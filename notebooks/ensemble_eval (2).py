#!/usr/bin/env python
"""
Combine the seed runs of one model into a deep ensemble and report uncertainty quality.

For every (dataset, split) it finds all finished seed runs, checks they share the same test set,
and reports:
  - single-model metrics: mean +/- std over seeds
  - deep ensemble (average of the seeds' probabilities)
  - ensemble of MC-dropout means (members x dropout samples)
  - ECE, and whether member disagreement actually flags the mistakes (Spearman correlation)
  - reliability-diagram data (CSV, ready to plot)
  - random-vs-scaffold gap per dataset

Usage:
    python ensemble_eval.py                              # pretrained runs
    python ensemble_eval.py --tag chemberta_scratch      # from-scratch runs
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

DIR_RE = re.compile(r"^(bace|bbbp|hiv)_(random|scaffold)_seed(\d+)(?:_(ss\d+|sf))?$")


def ece_binary(y, p, bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    e = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            e += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(e)


def reliability(y, p, bins=10):
    idx = np.clip(np.digitize(p, np.linspace(0, 1, bins + 1)[1:-1]), 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"bin": b, "n": int(m.sum()),
                     "mean_pred": float(p[m].mean()) if m.any() else np.nan,
                     "frac_pos": float(y[m].mean()) if m.any() else np.nan})
    return rows


def metrics(y, p):
    ok = len(np.unique(y)) == 2
    return {"roc_auc": float(roc_auc_score(y, p)) if ok else np.nan,
            "pr_auc": float(average_precision_score(y, p)) if ok else np.nan,
            "ece": ece_binary(y, p)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="/home/jovyan/runs")
    ap.add_argument("--tag", default="chemberta_pretrained")
    ap.add_argument("--out_dir", default="/home/jovyan/results")
    args = ap.parse_args()

    root = Path(args.runs_dir) / args.tag
    groups = {}
    for d in sorted(root.glob("*")):
        m = DIR_RE.match(d.name)
        if m and (d / "preds.npz").exists() and (d / "done.json").exists():
            ds, sp, seed, ss = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            groups.setdefault((ds, sp, ss), []).append((seed, d))
    if not groups:
        raise SystemExit(f"No finished runs found under {root}")
    # If runs on the group's shared split ("sf") exist for a dataset/split, use only those.
    for (ds, sp) in {(k[0], k[1]) for k in groups}:
        if (ds, sp, "sf") in groups:
            for k in [k for k in groups if k[0] == ds and k[1] == sp and k[2] != "sf"]:
                print(f"[note] {ds} {sp}: ignoring runs on own split ({k[2] or 'default'}), shared-split runs exist")
                del groups[k]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary, rel_rows, ens_rows = [], [], []

    for (ds, sp, ss), members in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2] or "")):
        members.sort()
        P = [np.load(d / "preds.npz", allow_pickle=True) for _, d in members]
        ref = P[0]["test_orig_idx"]
        if any(len(q["test_orig_idx"]) != len(ref) or not np.array_equal(q["test_orig_idx"], ref) for q in P[1:]):
            print(f"[skip ensemble] {ds} {sp}: seeds have DIFFERENT test sets. "
                  f"Rerun with a shared split (--split_seed 0, see run_all.sh).")
            continue
        y = P[0]["y"]
        n = len(members)
        p_all = np.stack([q["p"] for q in P])
        mc_all = np.stack([q["mc_mean"] for q in P])

        singles = [metrics(y, p) for p in p_all]
        mc_singles = [metrics(y, p) for p in mc_all]
        ens = metrics(y, p_all.mean(0))
        mc_ens = metrics(y, mc_all.mean(0))
        disagree = p_all.std(0)            # spread across ensemble members
        err = np.abs(y - p_all.mean(0))    # how wrong the ensemble is on each molecule
        rho = spearmanr(disagree, err).statistic if n > 1 else np.nan

        row = {"model": args.tag, "dataset": ds, "split": sp, "split_seed": ss or "", "n_members": n,
               "n_test": len(y), "n_test_pos": int(y.sum())}
        for k in ("roc_auc", "pr_auc", "ece"):
            v = np.array([s[k] for s in singles])
            row[f"single_{k}_mean"] = v.mean()
            row[f"single_{k}_std"] = v.std(ddof=1) if n > 1 else np.nan
            row[f"mcdrop_{k}_mean"] = np.mean([s[k] for s in mc_singles])
            row[f"ensemble_{k}"] = ens[k]
            row[f"ens_mcdrop_{k}"] = mc_ens[k]
        row["member_disagreement_mean"] = float(disagree.mean())
        row["disagreement_vs_error_spearman"] = float(rho) if rho == rho else np.nan
        summary.append(row)

        for kind, p in (("single_seed0", p_all[0]), ("ensemble", p_all.mean(0))):
            for r in reliability(y, p):
                rel_rows.append({"model": args.tag, "dataset": ds, "split": sp, "kind": kind, **r})
        for k, v in ens.items():
            ens_rows.append({"model": args.tag + "_ensemble", "dataset": ds, "split": sp,
                             "seed": "all", "metric": k, "value": v})
        for k, v in mc_ens.items():
            ens_rows.append({"model": args.tag + "_ensemble", "dataset": ds, "split": sp,
                             "seed": "all", "metric": "mcdrop_" + k, "value": v})

    S = pd.DataFrame(summary)
    S.to_csv(out_dir / f"{args.tag}_summary.csv", index=False)
    pd.DataFrame(rel_rows).to_csv(out_dir / f"{args.tag}_reliability.csv", index=False)
    pd.DataFrame(ens_rows).to_csv(out_dir / f"{args.tag}_ensemble_results.csv", index=False)

    pd.set_option("display.width", 200)
    show = ["dataset", "split", "n_members", "single_roc_auc_mean", "single_roc_auc_std", "ensemble_roc_auc",
            "single_ece_mean", "ensemble_ece", "disagreement_vs_error_spearman"]
    print(S[show].round(3).to_string(index=False))

    if {"random", "scaffold"} <= set(S["split"]):
        g = S.pivot_table(index="dataset", columns="split", values="single_roc_auc_mean")
        g["gap_random_minus_scaffold"] = g["random"] - g["scaffold"]
        print("\nRandom vs scaffold gap (single-model ROC-AUC, mean over seeds):")
        print(g.round(3).to_string())
        g.to_csv(out_dir / f"{args.tag}_gap.csv")
    print(f"\nWrote summary, reliability, gap and ensemble CSVs to {out_dir}")


if __name__ == "__main__":
    main()
