#!/usr/bin/env python
"""
ChemBERTa fine-tuning for scaffold-vs-random generalization (Ana M.'s model).

One run = one (dataset, split, seed). Resumable: a checkpoint is saved every epoch,
and re-running the same command continues where it stopped. Finished runs are skipped.

Example:
    python chemberta_finetune.py --dataset bace --split scaffold --seed 0
    python chemberta_finetune.py --dataset hiv  --split scaffold --seed 0 --split_file splits/hiv_scaffold.npz
    python chemberta_finetune.py --dataset bace --split random   --seed 0 --from_scratch

Outputs (all under /home/jovyan so they survive restarts):
    runs/<model_tag>/<dataset>_<split>_seed<seed>/{ckpt.pt, best.pt, preds.npz, done.json}
    results/chemberta_results.csv   columns: model,dataset,split,seed,metric,value
"""
import argparse
import json
import math
import os
import random
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import average_precision_score, roc_auc_score
from transformers import AutoConfig, AutoModel, AutoTokenizer

RDLogger.DisableLog("rdApp.*")

BASE = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
# name -> (url, smiles column, label column)
DATASETS = {
    "bace": (BASE + "bace.csv", "mol", "Class"),
    "bbbp": (BASE + "BBBP.csv", "smiles", "p_np"),
    "hiv": (BASE + "HIV.csv", "smiles", "HIV_active"),
}
DEFAULT_EPOCHS = {"bace": 30, "bbbp": 30, "hiv": 12}
DEFAULT_PATIENCE = {"bace": 8, "bbbp": 8, "hiv": 4}


# ----------------------------------------------------------------------------- data
def load_data(dataset, csv_path, data_dir):
    url, smi_col, lab_col = DATASETS[dataset]
    if csv_path is None:
        data_dir = Path(data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)
        csv_path = data_dir / f"{dataset}.csv"
        if not csv_path.exists():
            print(f"Downloading {url}")
            try:
                urllib.request.urlretrieve(url, csv_path)
            except Exception as e:  # network blocked -> tell the user what to do
                raise SystemExit(
                    f"Could not download {url} ({e}).\n"
                    f"Download the file elsewhere, upload it to {csv_path}, or pass --csv."
                )
    df = pd.read_csv(csv_path)
    smiles, labels, orig_idx = [], [], []
    for i, (s, y) in enumerate(zip(df[smi_col], df[lab_col])):
        if isinstance(s, str) and Chem.MolFromSmiles(s) is not None and not pd.isna(y):
            smiles.append(s)
            labels.append(int(y))
            orig_idx.append(i)
    print(f"{dataset}: {len(smiles)} valid molecules of {len(df)} rows, "
          f"{np.mean(labels) * 100:.1f}% positive")
    return smiles, np.array(labels, dtype=np.float32), np.array(orig_idx)


def scaffold_split(smiles, frac_train=0.8, frac_valid=0.1):
    """Deterministic scaffold split: whole scaffold groups go to one side only.
    Largest groups fill train first, so valid/test hold the rarer scaffolds."""
    groups = defaultdict(list)
    for i, s in enumerate(smiles):
        groups[MurckoScaffold.MurckoScaffoldSmiles(smiles=s, includeChirality=False)].append(i)
    ordered = sorted(groups.values(), key=lambda g: (len(g), g[0]), reverse=True)
    n = len(smiles)
    tr, va, te = [], [], []
    for g in ordered:
        if len(tr) + len(g) <= frac_train * n:
            tr += g
        elif len(va) + len(g) <= frac_valid * n:
            va += g
        else:
            te += g
    return np.array(tr), np.array(va), np.array(te)


def random_split(n, seed, frac_train=0.8, frac_valid=0.1):
    perm = np.random.RandomState(seed).permutation(n)
    a, b = int(frac_train * n), int((frac_train + frac_valid) * n)
    return perm[:a], perm[a:b], perm[b:]


def _norm_split_dict(raw):
    """Accept common key spellings: train/valid/test, *_idx, val, validation ..."""
    alias = {
        "train": ("train", "train_idx", "training", "train_indices", "train_index"),
        "valid": ("valid", "val", "validation", "valid_idx", "val_idx", "validation_idx",
                  "valid_indices", "val_indices", "valid_index"),
        "test": ("test", "test_idx", "test_indices", "test_index"),
    }
    lower = {str(k).lower(): k for k in raw}
    out = {}
    for name, opts in alias.items():
        for o in opts:
            if o in lower:
                out[name] = np.asarray(raw[lower[o]]).astype(int).ravel()
                break
        else:
            raise SystemExit(f"Split file has no '{name}' entry. Keys found: {list(raw)}")
    return out


def load_split_file(path, orig_idx, smiles):
    """Shared split files from Ana B. Supported:
      .npz / .json : train/valid/test arrays of row indices into the ORIGINAL csv
      .csv         : a column named split/subset/set/fold with train/valid/test per row.
                     If it also has a smiles column, molecules are matched by canonical SMILES
                     (safest); otherwise the csv row number is used as the index.
    Indices are mapped to positions in our filtered molecule list."""
    path = str(path)
    low = path.lower()
    if low.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        raw = {k: d[k] for k in d.files}
    elif low.endswith(".json"):
        raw = json.load(open(path))
    elif low.endswith(".csv"):
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        scol = next((cols[c] for c in ("split", "subset", "set", "fold") if c in cols), None)
        if scol is None:
            raise SystemExit(f"CSV split file needs a split/subset/set/fold column; columns: {list(df.columns)}")
        lab = df[scol].astype(str).str.lower().replace({"val": "valid", "validation": "valid", "training": "train"})
        smi_col = next((cols[c] for c in ("smiles", "smile", "mol") if c in cols), None)
        if smi_col is not None:
            def canon(s):
                m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
                return Chem.MolToSmiles(m) if m is not None else None
            lookup = {}
            for pos_, s in enumerate(smiles):
                lookup.setdefault(canon(s), pos_)
            res = []
            for k in ("train", "valid", "test"):
                found = [lookup.get(canon(s)) for s in df.loc[lab == k, smi_col]]
                miss = sum(f is None for f in found)
                if miss:
                    print(f"  split file: {miss} '{k}' molecules not found in our dataset")
                res.append(np.array([f for f in found if f is not None]))
            return _check_split(res)
        raw = {k: np.where(lab == k)[0] for k in ("train", "valid", "test")}
    else:
        raise SystemExit(f"Unsupported split file type: {path} (use .npz, .json or .csv)")
    idx = _norm_split_dict(raw)
    pos = {int(o): p for p, o in enumerate(orig_idx)}
    res = []
    for k in ("train", "valid", "test"):
        keep = [pos[int(i)] for i in idx[k] if int(i) in pos]
        if len(keep) != len(idx[k]):
            print(f"  split file: {len(idx[k]) - len(keep)} '{k}' indices not in our dataset (dropped)")
        res.append(np.array(keep))
    return _check_split(res)


def _check_split(res):
    tr, va, te = res
    if set(tr) & set(te) or set(tr) & set(va) or set(va) & set(te):
        raise SystemExit("Split file has molecules in more than one of train/valid/test. Wrong index convention?")
    print(f"  split file ok: train {len(tr)}, valid {len(va)}, test {len(te)}")
    return tuple(res)


class SmilesData(torch.utils.data.Dataset):
    def __init__(self, ids, labels, idx):
        self.ids, self.labels, self.idx = ids, labels, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        i = self.idx[k]
        return self.ids[i], self.labels[i]


def make_collate(pad_id):
    def collate(batch):
        L = max(len(x[0]) for x in batch)
        ids = torch.full((len(batch), L), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for r, (seq, _) in enumerate(batch):
            ids[r, : len(seq)] = torch.tensor(seq)
            mask[r, : len(seq)] = 1
        y = torch.tensor([x[1] for x in batch], dtype=torch.float32)
        return ids, mask, y
    return collate


# ----------------------------------------------------------------------------- model
class ChemBERTaClassifier(nn.Module):
    def __init__(self, name, from_scratch, dropout):
        super().__init__()
        cfg = AutoConfig.from_pretrained(name)
        self.enc = AutoModel.from_config(cfg) if from_scratch else AutoModel.from_pretrained(name)
        h = cfg.hidden_size
        self.head = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(h, h), nn.GELU(), nn.Dropout(dropout), nn.Linear(h, 1)
        )

    def forward(self, ids, mask):
        cls = self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0]
        return self.head(cls).squeeze(-1)


# ----------------------------------------------------------------------------- metrics
def ece_binary(y, p, bins=10):
    """Expected calibration error of the positive-class probability."""
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def metrics(y, p, prefix=""):
    out = {}
    if len(np.unique(y)) == 2:
        out[prefix + "roc_auc"] = float(roc_auc_score(y, p))
        out[prefix + "pr_auc"] = float(average_precision_score(y, p))
    else:
        out[prefix + "roc_auc"] = float("nan")
        out[prefix + "pr_auc"] = float("nan")
    out[prefix + "ece"] = ece_binary(y, p)
    return out


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    ps, ys = [], []
    for ids, mask, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logit = model(ids.to(device), mask.to(device))
        ps.append(torch.sigmoid(logit.float()).cpu())
        ys.append(y)
    return torch.cat(ps).numpy(), torch.cat(ys).numpy()


@torch.no_grad()
def mc_predict(model, loader, device, samples):
    """MC dropout: eval mode (no batchnorm-style surprises) but dropout layers stay active."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    runs = []
    for _ in range(samples):
        ps = []
        for ids, mask, _ in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(ids.to(device), mask.to(device))
            ps.append(torch.sigmoid(logit.float()).cpu())
        runs.append(torch.cat(ps).numpy())
    model.eval()
    runs = np.stack(runs)
    return runs.mean(0), runs.std(0)


# ----------------------------------------------------------------------------- utils
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj, path):
    tmp = str(path) + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def append_results(csv_path, rows):
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new = not csv_path.exists()
    pd.DataFrame(rows).to_csv(csv_path, mode="a", header=new, index=False)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=list(DATASETS), required=True)
    ap.add_argument("--split", choices=["random", "scaffold"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model_name", default="seyonec/ChemBERTa-zinc-base-v1")
    ap.add_argument("--from_scratch", action="store_true",
                    help="random init instead of pretrained weights (scratch-vs-pretrained comparison)")
    ap.add_argument("--csv", default=None, help="local csv instead of downloading")
    ap.add_argument("--split_seed", type=int, default=None,
                    help="random split only: seed for the split itself. Use the SAME value for all seeds "
                         "so every seed shares one test set (required to ensemble them). Default: split uses --seed.")
    ap.add_argument("--split_file", default=None,
                    help="shared split (.npz/.json with train/valid/test row indices). "
                         "If omitted, a split is built here (scaffold = deterministic).")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=5e-5, help="encoder learning rate")
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1, help="head dropout (encoder uses its own config)")
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--pos_weight", type=float, default=1.0,
                    help=">1 up-weights positives (helps imbalance but distorts calibration)")
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--data_dir", default="/home/jovyan/data")
    ap.add_argument("--runs_dir", default="/home/jovyan/runs")
    ap.add_argument("--results_csv", default="/home/jovyan/results/chemberta_results.csv")
    args = ap.parse_args()

    epochs = args.epochs or DEFAULT_EPOCHS[args.dataset]
    patience = args.patience or DEFAULT_PATIENCE[args.dataset]
    tag = "chemberta_scratch" if args.from_scratch else "chemberta_pretrained"
    if args.split_file:
        ss = "_sf"  # shared split from the group: separate folder so it never collides with own-split runs
    elif args.split_seed is not None and args.split == "random":
        ss = f"_ss{args.split_seed}"
    else:
        ss = ""
    out_dir = Path(args.runs_dir) / tag / f"{args.dataset}_{args.split}_seed{args.seed}{ss}"
    out_dir.mkdir(parents=True, exist_ok=True)
    if (out_dir / "done.json").exists():
        print(f"Already finished: {out_dir}. Skipping.")
        return

    set_seed(args.seed)
    device = torch.device("cuda")

    # ---- data and split
    smiles, labels, orig_idx = load_data(args.dataset, args.csv, args.data_dir)
    if args.split_file:
        tr, va, te = load_split_file(args.split_file, orig_idx, smiles)
    elif args.split == "scaffold":
        tr, va, te = scaffold_split(smiles)
    else:
        tr, va, te = random_split(len(smiles), args.seed if args.split_seed is None else args.split_seed)
    print(f"split={args.split}: train {len(tr)}, valid {len(va)}, test {len(te)}; "
          f"test positives {int(labels[te].sum())}")

    tok = AutoTokenizer.from_pretrained(args.model_name)
    ids = tok(smiles, truncation=True, max_length=args.max_len)["input_ids"]
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 1
    collate = make_collate(pad_id)

    def loader(idx, shuffle):
        return torch.utils.data.DataLoader(
            SmilesData(ids, labels, idx), batch_size=args.batch_size if shuffle else 128,
            shuffle=shuffle, collate_fn=collate, num_workers=args.workers,
            drop_last=False, persistent_workers=args.workers > 0)

    train_dl, val_dl, test_dl = loader(tr, True), loader(va, False), loader(te, False)

    # ---- model, optimizer, schedule
    model = ChemBERTaClassifier(args.model_name, args.from_scratch, args.dropout).to(device)
    opt = torch.optim.AdamW([
        {"params": model.enc.parameters(), "lr": args.lr},
        {"params": model.head.parameters(), "lr": args.head_lr},
    ], weight_decay=args.weight_decay)
    total_steps = epochs * len(train_dl)
    warmup = max(1, int(0.1 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else max(0.0, (total_steps - s) / (total_steps - warmup)))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=device))

    # ---- resume
    start_epoch, best_val, bad = 0, -math.inf, 0
    ckpt_path = out_dir / "ckpt.pt"
    if ckpt_path.exists():
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start_epoch, best_val, bad = ck["epoch"], ck["best_val"], ck["bad"]
        print(f"Resumed from epoch {start_epoch} (best val ROC-AUC {best_val:.4f})")

    # ---- train
    for epoch in range(start_epoch, epochs):
        model.train()
        t0, run_loss = time.time(), 0.0
        for ids_b, mask_b, y_b in train_dl:
            ids_b, mask_b, y_b = ids_b.to(device), mask_b.to(device), y_b.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logit = model(ids_b, mask_b)
            loss = loss_fn(logit.float(), y_b)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
        p, y = predict(model, val_dl, device)
        val = metrics(y, p)["roc_auc"]
        if np.isnan(val):
            val = -1.0
        improved = val > best_val
        if improved:
            best_val, bad = val, 0
            atomic_save(model.state_dict(), out_dir / "best.pt")
        else:
            bad += 1
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "epoch": epoch + 1, "best_val": best_val, "bad": bad}, ckpt_path)
        print(f"epoch {epoch + 1}/{epochs}  loss {run_loss / len(train_dl):.4f}  "
              f"val ROC-AUC {val:.4f}{' *' if improved else ''}  ({time.time() - t0:.0f}s)", flush=True)
        if bad >= patience:
            print(f"Early stop (no improvement for {patience} epochs)")
            break

    # ---- final evaluation with the best-validation weights
    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    p_val, y_val = predict(model, val_dl, device)
    p_test, y_test = predict(model, test_dl, device)
    mc_mean, mc_std = mc_predict(model, test_dl, device, args.mc_samples)

    res = {}
    res.update(metrics(y_val, p_val, "val_"))
    res.update(metrics(y_test, p_test))
    res.update(metrics(y_test, mc_mean, "mcdrop_"))
    res["mcdrop_mean_std"] = float(mc_std.mean())
    print(json.dumps(res, indent=2))

    # test-set predictions saved for the familiarity analysis and cross-seed ensembling.
    # test_orig_idx = row numbers in the ORIGINAL csv (same convention as the shared split files).
    np.savez(out_dir / "preds.npz", y=y_test, p=p_test, mc_mean=mc_mean, mc_std=mc_std,
             test_orig_idx=orig_idx[te], test_smiles=np.array([smiles[i] for i in te]))

    append_results(args.results_csv, [
        {"model": tag, "dataset": args.dataset, "split": args.split, "seed": args.seed,
         "metric": k, "value": v} for k, v in res.items()])
    json.dump({"epochs_run": epoch + 1 if epochs > start_epoch else start_epoch, **res},
              open(out_dir / "done.json", "w"))
    print(f"Done. Results appended to {args.results_csv}")


if __name__ == "__main__":
    main()
