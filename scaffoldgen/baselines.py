"""
Fingerprint baselines: Random Forest and a feed-forward net on Morgan bits.
Owner: Ani

Includes the uncertainty machinery (deep ensemble + MC dropout) that the
other three models reuse -- the pattern is identical, only the network
body changes.

Run:
    python -m scaffoldgen.baselines --dataset molhiv --split scaffold --seed 0 --model rf
    python -m scaffoldgen.baselines --dataset molhiv --split scaffold --seed 0 --model ffn
"""

from __future__ import annotations

import numpy as np

from . import data as D
from . import metrics as M
from . import runner as R


# --------------------------------------------------------------------------
# random forest
# --------------------------------------------------------------------------
def run_rf(dataset: str, split: str, seed: int, n_estimators: int = 500,
           log: bool = True):
    from sklearn.ensemble import RandomForestClassifier

    sp = D.load_split(dataset, split, seed)
    df = D.load_dataset(dataset)
    X = D.fingerprint_cache(dataset)
    y = df["y"].to_numpy()

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        class_weight="balanced_subsample",   # 3.5% positives on MolHIV
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X[sp.train], y[sp.train])
    prob = clf.predict_proba(X[sp.test])[:, 1]

    # Per-tree spread is a free ensemble-style uncertainty estimate.
    tree_probs = np.stack([t.predict_proba(X[sp.test])[:, 1] for t in clf.estimators_])
    unc = tree_probs.std(axis=0)

    met = M.summarize(y[sp.test], prob, unc)
    if log:
        R.log_metrics("rf", dataset, split, seed, met, "none")
        R.save_predictions("rf", dataset, split, seed, sp.test,
                           y[sp.test], prob, unc, kind="none")
    return met, prob, unc


# --------------------------------------------------------------------------
# feed-forward net
# --------------------------------------------------------------------------
def _build_ffn(n_in: int, hidden=(512, 256), dropout: float = 0.3):
    import torch.nn as nn

    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def train_ffn(X_tr, y_tr, X_va, y_va, seed: int, epochs: int = 50,
              batch_size: int = 128, lr: float = 1e-3, dropout: float = 0.3,
              hidden=(512, 256), device: str | None = None, verbose: bool = False):
    """Train one FFN, early-stopping on validation PR-AUC. Returns the model."""
    import torch
    import torch.nn as nn

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = _build_ffn(X_tr.shape[1], hidden, dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    # Positive class weight -- without it the net predicts all-negative.
    pos = max(1, int(y_tr.sum()))
    pw = torch.tensor([(len(y_tr) - pos) / pos], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pw)

    Xt = torch.tensor(X_tr, dtype=torch.float32)
    yt = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    ds = torch.utils.data.TensorDataset(Xt, yt)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                     drop_last=False)

    best, best_state, patience, bad = -np.inf, None, 10, 0
    for ep in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()

        va_prob = predict_ffn(model, X_va, device)
        score = M.pr_auc(y_va, va_prob)
        if np.isnan(score):
            score = M.roc_auc(y_va, va_prob)
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
        if verbose:
            print(f"  epoch {ep:3d}  val_pr_auc={score:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_ffn(model, X, device: str | None = None, batch_size: int = 1024):
    """Deterministic prediction (dropout off)."""
    import torch

    device = device or next(model.parameters()).device
    model.eval()
    out = []
    with torch.no_grad():
        for s in range(0, len(X), batch_size):
            xb = torch.tensor(X[s : s + batch_size], dtype=torch.float32).to(device)
            out.append(torch.sigmoid(model(xb)).squeeze(1).cpu().numpy())
    return np.concatenate(out)


def predict_mc_dropout(model, X, n_samples: int = 20, device: str | None = None,
                       batch_size: int = 1024):
    """Keep dropout ON at prediction time and sample. Gal & Ghahramani.

    Returns (mean_prob, std_prob). The std is the uncertainty estimate.
    """
    import torch
    import torch.nn as nn

    device = device or next(model.parameters()).device
    model.eval()
    # Re-enable ONLY dropout layers -- leaving BatchNorm in train mode would
    # corrupt the running stats.
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d)):
            m.train()

    samples = []
    with torch.no_grad():
        for _ in range(n_samples):
            out = []
            for s in range(0, len(X), batch_size):
                xb = torch.tensor(X[s : s + batch_size], dtype=torch.float32).to(device)
                out.append(torch.sigmoid(model(xb)).squeeze(1).cpu().numpy())
            samples.append(np.concatenate(out))
    S = np.stack(samples)
    return S.mean(axis=0), S.std(axis=0)


def ensemble_predict(models, X, device: str | None = None):
    """Deep ensemble: mean probability, std as uncertainty. Lakshminarayanan."""
    S = np.stack([predict_ffn(m, X, device) for m in models])
    return S.mean(axis=0), S.std(axis=0)


def run_ffn(dataset: str, split: str, seed: int, n_ensemble: int = 1,
            mc_samples: int = 0, epochs: int = 50, log: bool = True, **kw):
    """One FFN run. n_ensemble>1 trains an ensemble; mc_samples>0 adds MC dropout.

    Logged under uncertainty='none' | 'ensemble' | 'mcdropout' so the
    three are comparable in the same CSV.
    """
    sp = D.load_split(dataset, split, seed)
    df = D.load_dataset(dataset)
    X = D.fingerprint_cache(dataset).astype(np.float32)
    y = df["y"].to_numpy()

    models = [
        train_ffn(X[sp.train], y[sp.train], X[sp.valid], y[sp.valid],
                  seed=seed * 100 + m, epochs=epochs, **kw)
        for m in range(max(1, n_ensemble))
    ]

    results = {}
    # single model, no uncertainty
    prob = predict_ffn(models[0], X[sp.test])
    met = M.summarize(y[sp.test], prob)
    results["none"] = met
    if log:
        R.log_metrics("ffn", dataset, split, seed, met, "none")
        R.save_predictions("ffn", dataset, split, seed, sp.test,
                           y[sp.test], prob, kind="none")

    if n_ensemble > 1:
        p, u = ensemble_predict(models, X[sp.test])
        met_e = M.summarize(y[sp.test], p, u)
        results["ensemble"] = met_e
        if log:
            R.log_metrics("ffn", dataset, split, seed, met_e, "ensemble",
                          notes=f"members={n_ensemble}")
            R.save_predictions("ffn", dataset, split, seed, sp.test,
                               y[sp.test], p, u, kind="ensemble")

    if mc_samples > 0:
        p, u = predict_mc_dropout(models[0], X[sp.test], mc_samples)
        met_m = M.summarize(y[sp.test], p, u)
        results["mcdropout"] = met_m
        if log:
            R.log_metrics("ffn", dataset, split, seed, met_m, "mcdropout",
                          notes=f"samples={mc_samples}")
            R.save_predictions("ffn", dataset, split, seed, sp.test,
                               y[sp.test], p, u, kind="mcdropout")

    return results


# --------------------------------------------------------------------------
def _main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="molhiv", choices=D.DATASETS)
    ap.add_argument("--split", default="scaffold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default="rf", choices=("rf", "ffn", "both"))
    ap.add_argument("--ensemble", type=int, default=1)
    ap.add_argument("--mc", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    a = ap.parse_args()

    if a.model in ("rf", "both"):
        met, _, _ = run_rf(a.dataset, a.split, a.seed)
        print("RF ", {k: round(v, 4) for k, v in met.items()})
    if a.model in ("ffn", "both"):
        res = run_ffn(a.dataset, a.split, a.seed, a.ensemble, a.mc, a.epochs)
        for kind, met in res.items():
            print(f"FFN[{kind}]", {k: round(v, 4) for k, v in met.items()})


if __name__ == "__main__":
    _main()
