"""
Results logging + the gap table.
Owner: Ani

One CSV for the whole team: results/results.csv
    model,dataset,split,seed,metric,value,uncertainty,timestamp,notes

Append-only, one row per (model, dataset, split, seed, metric). Safe to
call from five notebooks at once (file lock + atomic append).

Also saves raw predictions to results/preds/, which is what Ana B.'s
familiarity analysis and Ani's calibration plots read. Save predictions
even when a run looks fine -- rerunning a model to recover them costs
GPU hours you do not have.
"""

from __future__ import annotations

import csv
import datetime as _dt
import json
import os

import numpy as np

RESULTS_DIR = os.environ.get("SCAFFOLDGEN_RESULTS", "/home/jovyan/scaffoldgen/results")
RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")
PREDS_DIR = os.path.join(RESULTS_DIR, "preds")

FIELDS = ["model", "dataset", "split", "seed", "metric", "value",
          "uncertainty", "timestamp", "notes"]


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def _append_rows(rows: list[dict]) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    new_file = not os.path.exists(RESULTS_CSV)
    if os.name == "nt":
        _append_rows_windows(rows, new_file)
        return
    # fcntl is POSIX-only, which is all we need on VERNE.
    import fcntl

    with open(RESULTS_CSV, "a", newline="") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new_file:
                w.writeheader()
            for r in rows:
                w.writerow(r)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def _append_rows_windows(rows: list[dict], new_file: bool) -> None:
    """Windows version of the locked append (no fcntl there).

    msvcrt locks a byte range at the current file position, and an
    append-mode file's position is its end, which moves. So the lock is
    taken on byte 0 of a separate, never-written lock file instead, which
    every process agrees on. LK_LOCK retries for about 10 s, then raises.
    """
    import msvcrt

    with open(RESULTS_CSV + ".lock", "a+b") as lk:
        lk.seek(0)
        msvcrt.locking(lk.fileno(), msvcrt.LK_LOCK, 1)
        try:
            # re-check under the lock, so two first writers can't both add a header
            new_file = not os.path.exists(RESULTS_CSV)
            with open(RESULTS_CSV, "a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                for r in rows:
                    w.writerow(r)
                fh.flush()
                os.fsync(fh.fileno())
        finally:
            lk.seek(0)
            msvcrt.locking(lk.fileno(), msvcrt.LK_UNLCK, 1)


def log_metrics(model: str, dataset: str, split: str, seed: int,
                metrics: dict, uncertainty: str = "none", notes: str = "") -> None:
    """Write one row per metric. `uncertainty` is 'none' | 'ensemble' | 'mcdropout'."""
    ts = _utcnow()
    rows = []
    for k, v in metrics.items():
        if isinstance(v, dict):          # skip nested (use log_binned_metrics)
            continue
        rows.append({
            "model": model, "dataset": dataset, "split": split, "seed": seed,
            "metric": k, "value": "" if v is None else f"{float(v):.6g}",
            "uncertainty": uncertainty, "timestamp": ts, "notes": notes,
        })
    _append_rows(rows)
    print(f"[log] {model}/{dataset}/{split}/seed{seed} ({uncertainty}) "
          f"-> {len(rows)} metrics")


def log_binned_metrics(model: str, dataset: str, split: str, seed: int,
                       binned: dict, uncertainty: str = "none") -> None:
    """Familiarity-bin metrics, logged as metric names like 'roc_auc@low'."""
    flat = {}
    for bin_name, m in binned.items():
        for k, v in m.items():
            flat[f"{k}@{bin_name}"] = v
    log_metrics(model, dataset, split, seed, flat, uncertainty, notes="familiarity_bin")


def save_predictions(model: str, dataset: str, split: str, seed: int,
                     test_idx, y_true, y_prob, uncertainty=None,
                     kind: str = "none", extra: dict | None = None) -> str:
    """Persist raw test-set predictions as .npz. Always do this."""
    os.makedirs(PREDS_DIR, exist_ok=True)
    name = f"{model}_{dataset}_{split}_seed{seed}_{kind}.npz"
    path = os.path.join(PREDS_DIR, name)
    arrs = {
        "test_idx": np.asarray(test_idx, dtype=np.int64),
        "y_true": np.asarray(y_true, dtype=np.int8),
        "y_prob": np.asarray(y_prob, dtype=np.float32),
    }
    if uncertainty is not None:
        arrs["uncertainty"] = np.asarray(uncertainty, dtype=np.float32)
    np.savez_compressed(path, **arrs)
    if extra:
        with open(path.replace(".npz", ".json"), "w") as fh:
            json.dump(extra, fh, indent=2)
    print(f"[log] predictions -> {path}")
    return path


def load_predictions(model: str, dataset: str, split: str, seed: int,
                     kind: str = "none"):
    path = os.path.join(PREDS_DIR,
                        f"{model}_{dataset}_{split}_seed{seed}_{kind}.npz")
    return np.load(path)


# --------------------------------------------------------------------------
# the table that goes in the paper
# --------------------------------------------------------------------------
def gap_table(metric: str = "roc_auc", dataset: str = "molhiv",
              random_split: str = "random", scaffold_split: str = "scaffold",
              uncertainty: str = "none"):
    """mean +/- std per model on each split, and the generalization gap.

    This is the central result of the project: how much each
    representation loses when the test scaffolds are new.
    """
    import pandas as pd

    df = pd.read_csv(RESULTS_CSV)
    df = df[(df.dataset == dataset) & (df.metric == metric)
            & (df.uncertainty == uncertainty) & (df.notes != "familiarity_bin")]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")

    out = []
    for model, g in df.groupby("model"):
        r = g[g.split == random_split]["value"].dropna()
        s = g[g.split == scaffold_split]["value"].dropna()
        if len(r) == 0 or len(s) == 0:
            continue
        out.append({
            "model": model,
            "n_seeds": min(len(r), len(s)),
            f"{metric}_random": r.mean(),
            "random_std": r.std(ddof=1) if len(r) > 1 else 0.0,
            f"{metric}_scaffold": s.mean(),
            "scaffold_std": s.std(ddof=1) if len(s) > 1 else 0.0,
            "gap": r.mean() - s.mean(),
        })
    tab = pd.DataFrame(out).sort_values("gap")
    return tab.reset_index(drop=True)


def coverage_report(expected_models=None, datasets=("molhiv", "bace", "bbbp"),
                    splits=("random", "scaffold"), seeds=(0, 1, 2),
                    metric: str = "roc_auc"):
    """What is still missing. Nino runs this daily in week 2."""
    import pandas as pd

    if not os.path.exists(RESULTS_CSV):
        print("[coverage] no results yet")
        return None
    df = pd.read_csv(RESULTS_CSV)
    df = df[(df.metric == metric) & (df.notes != "familiarity_bin")]
    have = set(zip(df.model, df.dataset, df.split, df.seed))
    models = expected_models or sorted(df.model.unique())

    missing = [
        {"model": m, "dataset": d, "split": sp, "seed": sd}
        for m in models for d in datasets for sp in splits for sd in seeds
        if (m, d, sp, sd) not in have
    ]
    print(f"[coverage] {len(have)} runs present, {len(missing)} missing "
          f"(of {len(models) * len(datasets) * len(splits) * len(seeds)})")
    return pd.DataFrame(missing)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--gap", action="store_true")
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--metric", default="roc_auc")
    ap.add_argument("--dataset", default="molhiv")
    a = ap.parse_args()
    if a.gap:
        print(gap_table(a.metric, a.dataset).to_string(index=False))
    if a.coverage:
        m = coverage_report(metric=a.metric)
        if m is not None and len(m):
            print(m.to_string(index=False))
