"""
fp_baselines.py - fingerprint baselines (random forest + feed-forward network) for the CS 553 final project.

Reads the data exactly as Ana B. saved it in the shared Drive folder:

    <project>/data/mapping/<ds>/mol.csv.gz               labels (+ SMILES), in OGB row order
    <project>/data/fingerprints/<ds>_r2_b2048.npy        Morgan fingerprints, same row order
    <project>/data/splits/<ds>_<split>_seed<k>.json      train / valid / test row numbers

with <ds> in {molhiv, bace, bbbp}. The harness names the first dataset "ogbg-molhiv"; this file
translates between the two.

IMPORTANT convention (from Ana B.'s split files): split file seed k is a DIFFERENT split from seed k-1.
So "seed k" means "use split file k AND model seed k". Mean +/- std over the 3 seeds therefore includes
split-to-split variation. (A deep ensemble needs ONE shared test set, so ensembles must use a single split
file, see the notes in the README.)

Everything is logged through harness.py, so the numbers are computed exactly like every other model's.

Quick use (Colab, after the setup cell):

    import fp_baselines
    fp_baselines.check_all()                  # validates the data and prints split sizes
    fp_baselines.run_rf("bace")               # both splits, 3 seeds -> results/rf.csv
    fp_baselines.run_rf("ogbg-molhiv")
    fp_baselines.run_ffn("bace")              # needs torch (Colab has it); logs ffn, ffn_mcd, ffn_ens
    fp_baselines.run_all()                    # EVERYTHING: both models x all 3 datasets x both splits x 3 seeds
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import harness

SHORT = {"ogbg-molhiv": "molhiv", "bace": "bace", "bbbp": "bbbp"}  # harness name -> Ana B.'s file prefix
FP_TAG = "r2_b2048"  # Morgan radius 2, 2048 bits


# --------------------------------------------------------------------------
# Loading Ana B.'s data
# --------------------------------------------------------------------------
def _data_dir(data_dir) -> Path:
    return Path(data_dir) if data_dir else harness.ROOT / "data"


def _short(dataset: str) -> str:
    if dataset not in SHORT:
        raise ValueError(f"Unknown dataset {dataset!r}. Use one of {tuple(SHORT)}")
    return SHORT[dataset]


def load_dataset(dataset: str, data_dir=None):
    """Return (X, y): fingerprints (n, 2048) uint8 and 0/1 labels (n,), in the shared row order."""
    d, short = _data_dir(data_dir), _short(dataset)
    df = pd.read_csv(d / "mapping" / short / "mol.csv.gz")
    label_cols = [c for c in df.columns if c not in ("smiles", "mol_id")]
    if len(label_cols) != 1:
        raise ValueError(f"{short}: expected one label column, found {label_cols}")
    y = df[label_cols[0]].to_numpy()
    if np.isnan(y.astype(float)).any() or not set(np.unique(y)).issubset({0, 1}):
        raise ValueError(f"{short}: labels must be 0/1 with no missing values")
    y = y.astype(int)
    X = np.load(d / "fingerprints" / f"{short}_{FP_TAG}.npy")
    if X.shape[0] != len(y):
        raise ValueError(f"{short}: {X.shape[0]} fingerprint rows but {len(y)} labels")
    return X, y


def load_split(dataset: str, split: str, seed: int, data_dir=None, n: int | None = None) -> dict:
    """
    Read split file <ds>_<split>_seed<seed>.json -> {"train": idx, "valid": idx, "test": idx} (numpy arrays).
    If `n` (number of molecules) is given, also checks that the indices are valid and do not overlap.
    """
    path = _data_dir(data_dir) / "splits" / f"{_short(dataset)}_{split}_seed{seed}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    rec = json.loads(path.read_text())
    out = {part: np.asarray(rec[part], dtype=np.int64) for part in ("train", "valid", "test")}
    if n is not None:
        for part, idx in out.items():
            if idx.size == 0:
                raise ValueError(f"{path.name}: {part} is empty")
            if idx.min() < 0 or idx.max() >= n:
                raise ValueError(f"{path.name}: {part} has indices outside 0..{n - 1}")
        for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")):
            shared = np.intersect1d(out[a], out[b]).size
            if shared:
                raise ValueError(f"{path.name}: {a} and {b} share {shared} molecules (leak!)")
    return out


def check_all(data_dir=None, datasets=tuple(SHORT), splits=harness.SPLITS, seeds=harness.SEEDS) -> pd.DataFrame:
    """Validate every dataset and split file we use. Returns a table of sizes and percent active."""
    rows = []
    for ds in datasets:
        X, y = load_dataset(ds, data_dir)
        for split in splits:
            for seed in seeds:
                sp = load_split(ds, split, seed, data_dir, n=len(y))
                rows.append(
                    {
                        "dataset": ds,
                        "split": split,
                        "seed": seed,
                        "n_train": len(sp["train"]),
                        "n_valid": len(sp["valid"]),
                        "n_test": len(sp["test"]),
                        "pct_active_train": round(100 * y[sp["train"]].mean(), 2),
                        "pct_active_test": round(100 * y[sp["test"]].mean(), 2),
                    }
                )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Random forest
# --------------------------------------------------------------------------
# 5 configs = the same small tuning budget for every model, as in the plan.
# No class weighting on purpose: class_weight="balanced" inflates the predicted probability of the rare
# active class, which would make ECE (calibration) look bad for a reason unrelated to the research question.
# ROC-AUC and PR-AUC only rank molecules, so they do not need the reweighting.
RF_CONFIGS = [
    dict(n_estimators=200, min_samples_leaf=1),
    dict(n_estimators=200, min_samples_leaf=2),
    dict(n_estimators=200, min_samples_leaf=5),
    dict(n_estimators=200, min_samples_leaf=2, max_features="log2"),
    dict(n_estimators=200, min_samples_leaf=1, max_depth=30),
]


def fit_predict_rf(X, y, sp: dict, config: dict, seed: int, eval_part: str = "test"):
    """Train on sp['train']; return (labels, P(active), row numbers) for the `eval_part` molecules."""
    rf = RandomForestClassifier(**config, n_jobs=-1, random_state=seed)
    rf.fit(X[sp["train"]], y[sp["train"]])
    if list(rf.classes_) != [0, 1]:
        raise ValueError(f"training set must contain both classes, got {list(rf.classes_)}")
    part = sp[eval_part]
    p = rf.predict_proba(X[part])[:, 1]
    return y[part], p, part


def select_config(X, y, sp: dict, configs=RF_CONFIGS, seed: int = 0):
    """Pick the config with the best ROC-AUC on the VALIDATION molecules (never the test set)."""
    rows = []
    for i, cfg in enumerate(configs):
        y_val, p_val, _ = fit_predict_rf(X, y, sp, cfg, seed, eval_part="valid")
        rows.append({"config_id": i, "config": json.dumps(cfg, sort_keys=True), **harness.compute_metrics(y_val, p_val)})
    table = pd.DataFrame(rows)
    return configs[int(table["roc_auc"].idxmax())], table


def log_config(model: str, dataset: str, split: str, config: dict, results_dir=None) -> None:
    """Record the chosen hyperparameters (needed for the report) in results/<model>_configs.json."""
    results_dir = Path(results_dir) if results_dir else harness.RESULTS_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{model}_configs.json"
    saved = json.loads(path.read_text()) if path.exists() else {}
    saved[f"{dataset}|{split}"] = config
    path.write_text(json.dumps(saved, indent=2, sort_keys=True))


def run_rf(
    dataset: str,
    data_dir=None,
    splits=harness.SPLITS,
    seeds=harness.SEEDS,
    configs=RF_CONFIGS,
    model_name: str = "rf",
    results_dir=None,
    pred_dir=None,
    verbose: bool = True,
) -> dict:
    """
    RF baseline for one dataset. For each split type:
      1. pick the best of the configs on the validation set of the FIRST seed's split file,
      2. for every seed k: load split file k, train with that config and model seed k, test,
         and log ROC-AUC / PR-AUC / ECE + predictions through the harness.
    Returns {split: chosen_config}.
    """
    X, y = load_dataset(dataset, data_dir)
    chosen = {}
    for split in splits:
        tune_sp = load_split(dataset, split, seeds[0], data_dir, n=len(y))
        best, table = select_config(X, y, tune_sp, configs, seed=seeds[0])
        chosen[split] = best
        log_config(model_name, dataset, split, best, results_dir)
        if verbose:
            print(f"\n[{model_name} | {dataset} | {split}] validation results (split seed {seeds[0]}):")
            print(table[["config_id", "config", "roc_auc", "pr_auc", "ece"]].round(4).to_string(index=False))
            print(f"chosen: {best}")

        def train_predict(seed, split=split, best=best):
            sp = load_split(dataset, split, seed, data_dir, n=len(y))
            y_true, p, part = fit_predict_rf(X, y, sp, best, seed)
            return {"y_true": y_true, "y_prob": p, "test_idx": part}

        harness.run_seeds(model_name, train_predict, dataset, split, seeds, results_dir, pred_dir, verbose)
    return chosen


# --------------------------------------------------------------------------
# Feed-forward network (FFN) + uncertainty (MC dropout, deep ensemble)
# --------------------------------------------------------------------------
# What gets logged for each dataset and split type (all through the harness):
#   ffn      one network per seed k, trained and tested on split file k          -> seeds 0, 1, 2
#   ffn_mcd  the SAME network as ffn, but predicting with dropout switched on:
#            20 stochastic passes averaged (Gal & Ghahramani); spread saved as y_std -> seeds 0, 1, 2
#   ffn_ens  deep ensemble: 3 networks with different model seeds, all trained and tested on ONE split
#            file (the first seed's), probabilities averaged; spread saved as y_std    -> logged as seed 0
# The ensemble uses a single split because every member must be tested on the same molecules.
#
# No class weighting (same reason as the RF: it would inflate the predicted probability of the rare class
# and make ECE look bad for the wrong reason). Early stopping uses validation log-loss, which rewards
# honest probabilities; the hyperparameter config is chosen by validation ROC-AUC, like the RF.
# Every config has dropout > 0, because MC dropout needs it.
FFN_CONFIGS = [
    dict(hidden=(512, 256), dropout=0.2, lr=1e-3),
    dict(hidden=(512, 256), dropout=0.5, lr=1e-3),
    dict(hidden=(1024, 256), dropout=0.3, lr=1e-3),
    dict(hidden=(256, 128), dropout=0.2, lr=1e-3),
    dict(hidden=(512, 256), dropout=0.3, lr=3e-4),
]


def _build_ffn(n_in: int, hidden, dropout: float):
    import torch.nn as nn

    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))  # one logit: P(active) = sigmoid(logit)
    return nn.Sequential(*layers)


def train_ffn(X_tr, y_tr, X_va, y_va, config: dict, seed: int, max_epochs: int = 60, patience: int = 8,
              batch_size: int = 128, weight_decay: float = 1e-5, device: str | None = None):
    """
    Train one FFN with Adam and binary cross-entropy. After every epoch the validation log-loss is checked;
    training stops after `patience` epochs without improvement and the best weights are restored.
    """
    import torch
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)  # fixes the initial weights and the dropout masks
    shuffle_gen = torch.Generator().manual_seed(seed)  # fixes the batch order

    model = _build_ffn(X_tr.shape[1], config["hidden"], config["dropout"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()

    Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y_tr, dtype=torch.float32, device=device).unsqueeze(1)
    Xv = torch.as_tensor(X_va, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_va, dtype=torch.float32, device=device).unsqueeze(1)

    best_loss, best_state, bad = float("inf"), None, 0
    n = Xt.shape[0]
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n, generator=shuffle_gen).to(device)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            opt.zero_grad()
            loss_fn(model(Xt[idx]), yt[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xv), yv).item()
        if val_loss < best_loss - 1e-5:
            best_loss, bad = val_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def predict_proba(model, X, batch_size: int = 2048) -> np.ndarray:
    """Normal prediction: dropout OFF. Returns P(active) for every row of X."""
    import torch

    device = next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.as_tensor(X[s : s + batch_size], dtype=torch.float32, device=device)
            out.append(torch.sigmoid(model(xb)).squeeze(1).cpu().numpy())
    return np.concatenate(out).astype(np.float64)


def predict_mc_dropout(model, X, n_samples: int = 20, seed: int = 0, batch_size: int = 2048):
    """
    MC dropout: keep the dropout layers ON, predict the same molecules `n_samples` times, and return
    (mean probability, standard deviation). The standard deviation is the model's uncertainty estimate.
    """
    import torch
    import torch.nn as nn

    device = next(model.parameters()).device
    torch.manual_seed(seed)
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()  # only the dropout layers, nothing else
    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = []
            for s in range(0, len(X), batch_size):
                xb = torch.as_tensor(X[s : s + batch_size], dtype=torch.float32, device=device)
                out.append(torch.sigmoid(model(xb)).squeeze(1).cpu().numpy())
            samples.append(np.concatenate(out))
    model.eval()  # switch dropout back off
    S = np.stack(samples).astype(np.float64)
    return S.mean(axis=0), S.std(axis=0)


def select_config_ffn(X, y, sp: dict, configs=FFN_CONFIGS, seed: int = 0, device=None):
    """Pick the FFN config with the best ROC-AUC on the VALIDATION set (never the test set)."""
    rows = []
    for i, cfg in enumerate(configs):
        model = train_ffn(X[sp["train"]], y[sp["train"]], X[sp["valid"]], y[sp["valid"]], cfg, seed, device=device)
        p_val = predict_proba(model, X[sp["valid"]])
        rows.append({"config_id": i, "config": json.dumps(cfg, sort_keys=True), **harness.compute_metrics(y[sp["valid"]], p_val)})
    table = pd.DataFrame(rows)
    return configs[int(table["roc_auc"].idxmax())], table


def run_ffn(
    dataset: str,
    data_dir=None,
    splits=harness.SPLITS,
    seeds=harness.SEEDS,
    configs=FFN_CONFIGS,
    mc_samples: int = 20,
    n_ensemble: int = 3,
    model_name: str = "ffn",
    results_dir=None,
    pred_dir=None,
    device=None,
    verbose: bool = True,
) -> dict:
    """
    Full FFN pipeline for one dataset. For each split type:
      1. pick the best config on the validation set of the first seed's split file,
      2. for every seed k: train on split file k (model seed k) and log `ffn` and `ffn_mcd`,
      3. train the ensemble members on the first seed's split file and log `ffn_ens` (as the first seed).
    Returns {split: chosen_config}.
    """
    X, y = load_dataset(dataset, data_dir)
    chosen = {}
    for split in splits:
        first_sp = load_split(dataset, split, seeds[0], data_dir, n=len(y))
        best, table = select_config_ffn(X, y, first_sp, configs, seed=seeds[0], device=device)
        chosen[split] = best
        log_config(model_name, dataset, split, best, results_dir)
        if verbose:
            print(f"\n[{model_name} | {dataset} | {split}] validation results (split seed {seeds[0]}):")
            print(table[["config_id", "config", "roc_auc", "pr_auc", "ece"]].round(4).to_string(index=False))
            print(f"chosen: {best}")

        kept = {}  # the network trained for the first seed is also ensemble member 0

        def train_predict(seed, split=split, best=best):
            sp = load_split(dataset, split, seed, data_dir, n=len(y))
            model = train_ffn(X[sp["train"]], y[sp["train"]], X[sp["valid"]], y[sp["valid"]], best, seed, device=device)
            X_te, y_te, te = X[sp["test"]], y[sp["test"]], sp["test"]
            p = predict_proba(model, X_te)
            p_mc, p_std = predict_mc_dropout(model, X_te, mc_samples, seed=seed)
            harness.evaluate_and_log(
                model_name + "_mcd", dataset, split, seed, y_te, p_mc, test_idx=te,
                results_dir=results_dir, pred_dir=pred_dir, y_std=p_std,
            )
            if seed == seeds[0]:
                kept["model"] = model
            return {"y_true": y_te, "y_prob": p, "test_idx": te}

        harness.run_seeds(model_name, train_predict, dataset, split, seeds, results_dir, pred_dir, verbose)

        # deep ensemble on ONE split file (members need the same test molecules)
        members = [kept["model"]]
        for i in range(1, n_ensemble):
            members.append(
                train_ffn(X[first_sp["train"]], y[first_sp["train"]], X[first_sp["valid"]], y[first_sp["valid"]],
                          best, seeds[0] + i, device=device)
            )
        X_te, y_te, te = X[first_sp["test"]], y[first_sp["test"]], first_sp["test"]
        P = np.stack([predict_proba(m, X_te) for m in members])
        m_ens = harness.evaluate_and_log(
            model_name + "_ens", dataset, split, seeds[0], y_te, P.mean(axis=0), test_idx=te,
            results_dir=results_dir, pred_dir=pred_dir, y_std=P.std(axis=0), n_members=np.array(len(members)),
        )
        if verbose:
            print(f"[{model_name}_ens | {dataset} | {split} | {len(members)} members on split seed {seeds[0]}] "
                  f"ROC-AUC={m_ens['roc_auc']:.4f}  PR-AUC={m_ens['pr_auc']:.4f}  ECE={m_ens['ece']:.4f}")
    return chosen


# --------------------------------------------------------------------------
# One command that does everything
# --------------------------------------------------------------------------
def _is_done(model_name: str, dataset: str, splits, seeds, results_dir=None) -> bool:
    """True if results/<model_name>.csv already holds all 3 metrics for every split and seed of this dataset."""
    results_dir = Path(results_dir) if results_dir else harness.RESULTS_DIR
    path = results_dir / f"{model_name}.csv"
    if not path.exists():
        return False
    df = pd.read_csv(path)
    df = df[df["dataset"] == dataset]
    for split in splits:
        for seed in seeds:
            n = ((df["split"] == split) & (df["seed"] == seed)).sum()
            if n < len(harness.METRICS):
                return False
    return True


def run_all(
    models=("rf", "ffn"),
    datasets=("bace", "bbbp", "ogbg-molhiv"),  # smallest first, so any problem shows up in minutes, not hours
    data_dir=None,
    splits=harness.SPLITS,
    seeds=harness.SEEDS,
    force: bool = False,
    rf_kwargs: dict | None = None,
    ffn_kwargs: dict | None = None,
) -> None:
    """
    Run every requested model on every requested dataset (both splits, all seeds) and log everything through
    the harness. Safe to re-run after a Colab disconnect: a (model, dataset) pair that is already complete in
    results/ is skipped, so you only pay for what is missing. Use force=True to redo everything.
    """
    rf_kwargs, ffn_kwargs = rf_kwargs or {}, ffn_kwargs or {}
    for dataset in datasets:
        for model in models:
            if model not in ("rf", "ffn"):
                raise ValueError(f"Unknown model {model!r}. Use 'rf' and/or 'ffn'.")
            done = _is_done(model, dataset, splits, seeds, rf_kwargs.get("results_dir") if model == "rf" else ffn_kwargs.get("results_dir"))
            if model == "ffn" and done:  # the ensemble is logged separately, as the first seed
                done = _is_done("ffn_ens", dataset, splits, seeds[:1], ffn_kwargs.get("results_dir"))
            if done and not force:
                print(f"\n=== {model} | {dataset}: already complete, skipping (use force=True to redo) ===")
                continue
            print(f"\n=== {model} | {dataset}: running ===")
            if model == "rf":
                run_rf(dataset, data_dir, splits=splits, seeds=seeds, **rf_kwargs)
            else:
                run_ffn(dataset, data_dir, splits=splits, seeds=seeds, **ffn_kwargs)
    print("\nAll requested runs are complete.")
