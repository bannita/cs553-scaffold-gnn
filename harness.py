"""
harness.py - shared evaluation harness for the CS 553 final project
(GNNs and scaffold generalization in molecular bioactivity prediction).

Every model (RF, FFN, D-MPNN, GIN, ChemBERTa, ensembles, MC dropout) is
evaluated through this file, so all numbers are computed the same way.

What it does
------------
1. Computes ROC-AUC, PR-AUC and ECE from true labels + predicted probabilities.
2. Runs a model over several seeds (default 0, 1, 2).
3. Logs results to results/<model>.csv  (columns: model, dataset, split, seed, metric, value)
4. Saves per-molecule predictions to predictions/<model>__<dataset>__<split>__seed<k>.npz

Quick use
---------
    import harness

    def train_and_predict(seed):
        # ... train your model with this seed on the TRAIN split ...
        # ... predict probabilities of the positive class on the TEST split ...
        return {"y_true": y_test, "y_prob": p_test, "test_idx": test_idx}

    harness.run_seeds("rf", train_and_predict, dataset="bace", split="scaffold")

See README.md for the team conventions (names, folders, rules).
"""
from __future__ import annotations

import os
import random
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

# --------------------------------------------------------------------------
# Team-wide constants. Do NOT change these without telling everyone.
# --------------------------------------------------------------------------
DATASETS = ("ogbg-molhiv", "bace", "bbbp")
SPLITS = ("random", "scaffold")
SEEDS = (0, 1, 2)
METRICS = ("roc_auc", "pr_auc", "ece")
COLUMNS = ["model", "dataset", "split", "seed", "metric", "value"]
N_BINS = 10  # number of equal-width confidence bins used for ECE

# Folders live next to this file (so inside the shared Drive folder), unless
# the environment variable CS553_PROJECT points somewhere else.
ROOT = Path(os.environ.get("CS553_PROJECT", Path(__file__).resolve().parent))
RESULTS_DIR = ROOT / "results"
PRED_DIR = ROOT / "predictions"

_MODEL_NAME_RE = re.compile(r"^[A-Za-z0-9_.\-]+$")


# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    """Seed python, numpy and (if installed) torch."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# --------------------------------------------------------------------------
# Input checking
# --------------------------------------------------------------------------
def _check_arrays(y_true, y_prob):
    y_true = np.asarray(y_true).ravel()
    y_prob = np.asarray(y_prob, dtype=float).ravel()
    if y_true.shape != y_prob.shape:
        raise ValueError(f"y_true and y_prob must have the same length, got {y_true.shape} and {y_prob.shape}")
    if y_true.size == 0:
        raise ValueError("Empty predictions.")
    if np.isnan(y_prob).any() or np.isnan(y_true.astype(float)).any():
        raise ValueError("NaN found in y_true or y_prob (drop or fill missing labels before calling the harness).")
    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("y_true must contain only 0/1 labels.")
    if y_prob.min() < 0.0 or y_prob.max() > 1.0:
        raise ValueError(
            "y_prob must be probabilities in [0, 1]. If your model outputs logits, apply a sigmoid first."
        )
    return y_true.astype(int), y_prob


def _check_names(model: str, dataset: str, split: str, seed: int) -> None:
    if not _MODEL_NAME_RE.match(model):
        raise ValueError(f"Bad model name {model!r}: use only letters, digits, '_', '-' and '.'")
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}. Use one of {DATASETS}")
    if split not in SPLITS:
        raise ValueError(f"Unknown split {split!r}. Use one of {SPLITS}")
    if not isinstance(seed, (int, np.integer)):
        raise ValueError(f"seed must be an integer, got {seed!r}")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def reliability_bins(y_true, y_prob, n_bins: int = N_BINS) -> pd.DataFrame:
    """
    Table behind ECE and reliability plots.

    Predictions (probability of the POSITIVE / active class) are put into
    `n_bins` equal-width bins on [0, 1]. For every bin we report:
      count          number of molecules in the bin
      mean_prob      average predicted probability
      frac_positive  fraction of molecules that are really active
    A perfectly calibrated model has mean_prob == frac_positive in every bin.
    """
    y_true, y_prob = _check_arrays(y_true, y_prob)
    idx = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)  # prob == 1.0 goes in the last bin
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        rows.append(
            {
                "bin": b,
                "lower": b / n_bins,
                "upper": (b + 1) / n_bins,
                "count": n,
                "mean_prob": float(y_prob[mask].mean()) if n else np.nan,
                "frac_positive": float(y_true[mask].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ece(y_true, y_prob, n_bins: int = N_BINS) -> float:
    """
    Expected Calibration Error (positive-class version).

    ECE = sum over bins of (bin_size / N) * |mean_prob - frac_positive|.
    0 is perfect; higher means the model's probabilities are less honest.
    """
    table = reliability_bins(y_true, y_prob, n_bins)
    table = table[table["count"] > 0]
    weights = table["count"] / table["count"].sum()
    return float((weights * (table["mean_prob"] - table["frac_positive"]).abs()).sum())


def compute_metrics(y_true, y_prob) -> dict:
    """Return {'roc_auc': ..., 'pr_auc': ..., 'ece': ...} (AUCs are NaN if only one class is present)."""
    y_true, y_prob = _check_arrays(y_true, y_prob)
    has_both = len(np.unique(y_true)) == 2
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if has_both else float("nan"),
        "pr_auc": float(average_precision_score(y_true, y_prob)) if has_both else float("nan"),
        "ece": ece(y_true, y_prob),
    }


# --------------------------------------------------------------------------
# Logging results and predictions
# --------------------------------------------------------------------------
def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file then rename, so a crash can't leave a half-written CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def log_results(model, dataset, split, seed, metrics: dict, results_dir=None) -> Path:
    """
    Write one row per metric to results/<model>.csv.
    Re-running the same (model, dataset, split, seed) REPLACES the old rows, never duplicates them.
    Each model has its own file, so teammates never overwrite each other.
    """
    _check_names(model, dataset, split, seed)
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    path = results_dir / f"{model}.csv"
    new = pd.DataFrame(
        [
            {"model": model, "dataset": dataset, "split": split, "seed": int(seed), "metric": k, "value": float(v)}
            for k, v in metrics.items()
        ],
        columns=COLUMNS,
    )
    if path.exists():
        old = pd.read_csv(path)
        same = (old["dataset"] == dataset) & (old["split"] == split) & (old["seed"] == int(seed))
        old = old[~same]
        new = pd.concat([old, new], ignore_index=True)
    _atomic_write_csv(new, path)
    return path


def save_predictions(model, dataset, split, seed, y_true, y_prob, test_idx=None, pred_dir=None, **extra) -> Path:
    """
    Save per-molecule predictions to predictions/<model>__<dataset>__<split>__seed<k>.npz

    Stored arrays: y_true, y_prob, and optionally
      test_idx   index of each test molecule in the dataset (needed for the familiarity analysis)
      anything passed as extra keyword arguments, e.g. y_std=... for ensemble / MC-dropout spread
    """
    _check_names(model, dataset, split, seed)
    y_true, y_prob = _check_arrays(y_true, y_prob)
    pred_dir = Path(pred_dir) if pred_dir else PRED_DIR
    pred_dir.mkdir(parents=True, exist_ok=True)
    arrays = {"y_true": y_true, "y_prob": y_prob}
    if test_idx is not None:
        test_idx = np.asarray(test_idx)
        if test_idx.shape[0] != y_true.shape[0]:
            raise ValueError("test_idx must have one entry per test molecule.")
        arrays["test_idx"] = test_idx
    for k, v in extra.items():
        arrays[k] = np.asarray(v)
    path = pred_dir / f"{model}__{dataset}__{split}__seed{int(seed)}.npz"
    np.savez_compressed(path, **arrays)
    return path


def evaluate_and_log(
    model, dataset, split, seed, y_true, y_prob, test_idx=None, results_dir=None, pred_dir=None, **extra
) -> dict:
    """Compute metrics for one run, log them, save predictions. Returns the metrics dict."""
    _check_names(model, dataset, split, seed)
    metrics = compute_metrics(y_true, y_prob)
    log_results(model, dataset, split, seed, metrics, results_dir)
    save_predictions(model, dataset, split, seed, y_true, y_prob, test_idx, pred_dir, **extra)
    return metrics


def run_seeds(model, train_predict_fn, dataset, split, seeds=SEEDS, results_dir=None, pred_dir=None, verbose=True):
    """
    Run `train_predict_fn(seed)` for every seed and log everything.

    `train_predict_fn` must train the model with that seed and return a dict:
        {"y_true": ..., "y_prob": ..., "test_idx": ... (optional), <extra arrays> (optional)}
    y_prob is the predicted probability of the positive (active) class on the TEST set.
    Returns a DataFrame with one row per seed.
    """
    rows = []
    for seed in seeds:
        set_seed(seed)
        out = train_predict_fn(seed)
        if "y_true" not in out or "y_prob" not in out:
            raise ValueError("train_predict_fn must return a dict with at least 'y_true' and 'y_prob'.")
        out = dict(out)
        y_true, y_prob = out.pop("y_true"), out.pop("y_prob")
        test_idx = out.pop("test_idx", None)
        metrics = evaluate_and_log(
            model, dataset, split, seed, y_true, y_prob, test_idx, results_dir, pred_dir, **out
        )
        rows.append({"seed": seed, **metrics})
        if verbose:
            print(
                f"[{model} | {dataset} | {split} | seed {seed}] "
                f"ROC-AUC={metrics['roc_auc']:.4f}  PR-AUC={metrics['pr_auc']:.4f}  ECE={metrics['ece']:.4f}"
            )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Reading results back
# --------------------------------------------------------------------------
def load_results(results_dir=None) -> pd.DataFrame:
    """Combine every results/<model>.csv into one long table (model, dataset, split, seed, metric, value)."""
    results_dir = Path(results_dir) if results_dir else RESULTS_DIR
    files = sorted(results_dir.glob("*.csv"))
    if not files:
        return pd.DataFrame(columns=COLUMNS)
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


def summarize(results_dir=None) -> pd.DataFrame:
    """Mean, std and number of seeds for every (model, dataset, split, metric)."""
    df = load_results(results_dir)
    if df.empty:
        return df
    out = (
        df.groupby(["model", "dataset", "split", "metric"])["value"]
        .agg(mean="mean", std=lambda s: s.std(ddof=1) if len(s) > 1 else float("nan"), n_seeds="count")
        .reset_index()
    )
    return out


def scaffold_gap(metric: str = "roc_auc", results_dir=None) -> pd.DataFrame:
    """
    The core result of the project: random-split score minus scaffold-split score
    (mean over seeds) for every model and dataset. A big gap = poor generalization to new scaffolds.
    """
    s = summarize(results_dir)
    if s.empty:
        return s
    s = s[s["metric"] == metric]
    wide = s.pivot_table(index=["model", "dataset"], columns="split", values="mean").reset_index()
    if not {"random", "scaffold"}.issubset(wide.columns):
        return wide
    wide["gap"] = wide["random"] - wide["scaffold"]
    return wide
