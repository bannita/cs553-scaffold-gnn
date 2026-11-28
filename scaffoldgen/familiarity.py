"""
Familiarity analysis: does performance degrade as test molecules get less
similar to the training set?
Owner: Ana B.

Reads saved predictions from results/preds/ -- it does NOT retrain anything,
so it runs on CPU after everyone else is done.

    python -m scaffoldgen.familiarity --dataset molhiv --split scaffold --seed 0
"""

from __future__ import annotations

import glob
import os

import numpy as np

from . import data as D
from . import metrics as M
from . import runner as R


def analyze(model: str, dataset: str, split: str, seed: int,
            kind: str = "none", log: bool = True, binning: str = "fixed",
            n_bins: int = 3):
    """Per-familiarity-bin metrics for one saved prediction file.

    binning='fixed'    -> FAMILIARITY_EDGES cutoffs (0.3 / 0.5). Comparable
                          across datasets and splits, but a bin can end up
                          with too few actives to score: on BACE the low and
                          medium bins hold a handful of molecules.
    binning='quantile' -> equal-count bins, which guarantees usable bin
                          sizes. The edges are dataset-specific, so they are
                          printed and returned -- report them in the write-up.

    Quantile bins are logged under the metric names 'roc_auc@q_low' etc., so
    they never overwrite the fixed-cutoff rows in results.csv.

    Returns (binned, sim, bins, edges). edges is None for fixed binning.
    """
    z = R.load_predictions(model, dataset, split, seed, kind)
    test_idx, y_true, y_prob = z["test_idx"], z["y_true"], z["y_prob"]

    sp = D.load_split(dataset, split, seed)
    if list(test_idx) != list(sp.test):
        raise ValueError(
            f"{model}: saved test indices do not match the split file. "
            "Someone re-split locally -- rerun that model with load_split()."
        )

    fp = D.fingerprint_cache(dataset)
    sim = D.tanimoto_max_similarity(fp[sp.test], fp[sp.train])
    if binning == "quantile":
        bins, edges = D.familiarity_bins_quantile(sim, n_bins)
        labels = tuple(f"q_{lab}" for lab in D.FAMILIARITY_LABELS[: len(edges) - 1])
        print(f"[familiarity] {model}: quantile bin edges "
              f"{[round(float(e), 3) for e in edges[1:-1]]}")
    elif binning == "fixed":
        bins, edges = D.familiarity_bins(sim), None
        labels = D.FAMILIARITY_LABELS
    else:
        raise ValueError(f"binning must be 'fixed' or 'quantile', got {binning!r}")

    binned = M.summarize_by_bin(y_true, y_prob, bins, labels)
    if log:
        R.log_binned_metrics(model, dataset, split, seed, binned, kind)
    return binned, sim, bins, edges


def analyze_all(dataset: str = "molhiv", split: str = "scaffold",
                seed: int = 0, kind: str = "none", binning: str = "fixed",
                n_bins: int = 3):
    """Every model that has predictions for this (dataset, split, seed)."""
    pattern = os.path.join(R.PREDS_DIR, f"*_{dataset}_{split}_seed{seed}_{kind}.npz")
    rows = []
    for path in sorted(glob.glob(pattern)):
        model = os.path.basename(path).split(f"_{dataset}_")[0]
        try:
            binned, sim, _, edges = analyze(model, dataset, split, seed, kind,
                                            binning=binning, n_bins=n_bins)
        except Exception as exc:
            print(f"[familiarity] {model}: {exc}")
            continue
        for bin_name, m in binned.items():
            rows.append({"model": model, "bin": bin_name, **m})
        print(f"[familiarity] {model}: " + "  ".join(
            f"{b}(n={int(m.get('n', 0))}) auc={m.get('roc_auc', float('nan')):.3f}"
            for b, m in binned.items() if m.get("n", 0) > 0))
    import pandas as pd
    return pd.DataFrame(rows)


def similarity_summary(dataset: str, seed: int = 0):
    """Sanity check that the scaffold split really is harder: mean
    nearest-neighbour similarity should be clearly lower than on random."""
    out = {}
    for split in ("random", "scaffold"):
        sim, bins = D.test_familiarity(dataset, split, seed)
        out[split] = {
            "mean_nn_sim": float(sim.mean()),
            "median_nn_sim": float(np.median(sim)),
            **{f"frac_{lab}": float((bins == i).mean())
               for i, lab in enumerate(D.FAMILIARITY_LABELS)},
        }
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="molhiv", choices=D.DATASETS)
    ap.add_argument("--split", default="scaffold")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--kind", default="none")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--binning", default="fixed", choices=("fixed", "quantile"),
                    help="'quantile' gives equal-count bins; use it when a "
                         "fixed-cutoff bin has too few actives to score")
    ap.add_argument("--n-bins", type=int, default=3)
    a = ap.parse_args()

    if a.summary:
        import json
        print(json.dumps(similarity_summary(a.dataset, a.seed), indent=2))
    df = analyze_all(a.dataset, a.split, a.seed, a.kind,
                     binning=a.binning, n_bins=a.n_bins)
    if len(df):
        print(df.to_string(index=False))
