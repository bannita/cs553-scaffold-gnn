#!/usr/bin/env python
"""
Send my ChemBERTa results through the team's shared harness (harness.py, by Ani),
so they are computed and logged exactly like everyone else's.

What it writes (into --project_dir, same layout as the shared Drive folder):
    results/chemberta.csv        single model, seeds 0-2
    results/chemberta_mcd.csv    MC dropout (20 samples averaged), seeds 0-2
    results/chemberta_ens.csv    deep ensemble of the 3 seeds (ONE ensemble -> logged as seed 0)
    predictions/<model>__<dataset>__<split>__seed<k>.npz
For the from-scratch model use --prefix chemberta_scratch (-> chemberta_scratch, _mcd, _ens).

Usage (on VERNE):
    python chemberta_to_harness.py --harness_dir /home/jovyan/repo
    python chemberta_to_harness.py --harness_dir /home/jovyan/repo --tag chemberta_scratch --prefix chemberta_scratch

Then download results/ and predictions/ from CS553_Project (see the guide) and upload them to the shared Drive.
Re-running is safe: the harness replaces old rows instead of duplicating them.
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np

DIR_RE = re.compile(r"^(bace|bbbp|hiv)_(random|scaffold)_seed(\d+)(?:_(ss\d+|sf))?$")
DS_NAME = {"bace": "bace", "bbbp": "bbbp", "hiv": "ogbg-molhiv"}  # my names -> team names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness_dir", required=True, help="folder that contains harness.py")
    ap.add_argument("--project_dir", default="/home/jovyan/CS553_Project")
    ap.add_argument("--runs_dir", default="/home/jovyan/runs")
    ap.add_argument("--tag", default="chemberta_pretrained", help="run folder name under runs/")
    ap.add_argument("--prefix", default="chemberta", help="model name used in the team files")
    args = ap.parse_args()

    # the harness reads this at import time, so it must be set first
    os.environ["CS553_PROJECT"] = str(Path(args.project_dir))
    sys.path.insert(0, args.harness_dir)
    import harness

    root = Path(args.runs_dir) / args.tag
    groups = {}
    for d in sorted(root.glob("*")):
        m = DIR_RE.match(d.name)
        if m and (d / "preds.npz").exists() and (d / "done.json").exists():
            ds, sp, seed, ss = m.group(1), m.group(2), int(m.group(3)), m.group(4)
            if seed in harness.SEEDS:
                groups.setdefault((ds, sp), {}).setdefault(ss, []).append((seed, d))
    if not groups:
        raise SystemExit(f"No finished runs under {root}")

    for (ds, sp), variants in sorted(groups.items()):
        if len(variants) > 1:  # e.g. own split and the group's shared split both present
            best = "sf" if "sf" in variants else max(variants, key=lambda k: len(variants[k]))
            print(f"[note] {ds} {sp}: several split variants {list(variants)}; using {best!r}")
            members = variants[best]
        else:
            members = next(iter(variants.values()))
        members.sort()
        name = DS_NAME[ds]
        P = {seed: np.load(d / "preds.npz", allow_pickle=True) for seed, d in members}

        for seed, q in P.items():
            harness.evaluate_and_log(args.prefix, name, sp, seed, q["y"], q["p"], test_idx=q["test_orig_idx"])
            harness.evaluate_and_log(args.prefix + "_mcd", name, sp, seed, q["y"], q["mc_mean"],
                                     test_idx=q["test_orig_idx"], y_std=q["mc_std"])

        seeds = sorted(P)
        ref = P[seeds[0]]["test_orig_idx"]
        same = all(np.array_equal(P[s]["test_orig_idx"], ref) for s in seeds)
        if len(seeds) >= 2 and same:
            stack = np.stack([P[s]["p"] for s in seeds])
            harness.evaluate_and_log(args.prefix + "_ens", name, sp, 0, P[seeds[0]]["y"], stack.mean(0),
                                     test_idx=ref, y_std=stack.std(0), n_members=np.array(len(seeds)))
        elif len(seeds) >= 2:
            print(f"[skip ensemble] {ds} {sp}: seeds have different test sets (needs one shared split)")
        print(f"ok  {name:12s} {sp:9s} seeds {seeds}")

    s = harness.summarize()
    s = s[s["model"].str.startswith(args.prefix)]
    print("\n", s.round(3).to_string(index=False))
    print(f"\nFiles are in {args.project_dir}/results and {args.project_dir}/predictions")


if __name__ == "__main__":
    main()
