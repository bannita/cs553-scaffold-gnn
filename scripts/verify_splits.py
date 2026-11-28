"""Ana B.'s hand-off check. Run this after building the splits, and paste the
output into the group chat. It is the team's evidence that the splits are sound.

    python scripts/verify_splits.py                 # all 3 datasets
    python scripts/verify_splits.py --dataset molhiv

What it checks, per dataset, split and seed:
  * every molecule lands in exactly one of train / valid / test
  * no index appears in two sets
  * the validation set is non-empty and contains at least one active
  * on a scaffold split, no ring-bearing scaffold appears in both train and test
  * the scaffold split's test molecules really are less similar to training
    than the random split's are

The last one matters most. A scaffold split that leaves test molecules nearly
as familiar as a random split would make the project's headline gap small for
an uninteresting reason.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from scaffoldgen import data as D


def check(dataset: str, seeds=(0, 1, 2)) -> list[str]:
    problems: list[str] = []
    notes: list[str] = []
    df = D.load_dataset(dataset)
    smiles = df["smiles"].tolist()
    y = df["y"].to_numpy()
    scaf = [D.murcko_scaffold(s) for s in smiles]
    n_acyclic = scaf.count("")

    groups = D.scaffold_groups(smiles, acyclic_as_own_group=False)
    ring_families = len([k for k in groups if k != ""])

    print(f"\n{'=' * 78}\n{dataset}: {len(df)} molecules | "
          f"pos_rate {y.mean():.4f} | {ring_families} ring scaffolds | "
          f"{n_acyclic} acyclic ({n_acyclic / len(df):.1%})")
    if n_acyclic:
        print(f"  note: acyclic molecules have no Murcko scaffold. Each is "
              f"treated as its own family, so they appear on both sides of a "
              f"scaffold split. That is intended -- they are not relatives.")

    print(f"\n{'split':<18}{'seed':>5}{'train':>8}{'valid':>7}{'test':>7}"
          f"{'leak':>6}{'NN sim':>8}{'pos tr':>8}{'pos te':>8}")
    print("-" * 78)

    sim_by_split: dict[str, list[float]] = {}
    splits = ["random", "scaffold", "scaffold_official"]

    for split in splits:
        for seed in seeds:
            if split == "scaffold_official" and seed != 0:
                continue
            try:
                sp = D.load_split(dataset, split, seed)
            except FileNotFoundError as exc:
                problems.append(f"{dataset}/{split}/seed{seed}: {exc}")
                continue

            total = len(sp.train) + len(sp.valid) + len(sp.test)
            if total != len(df):
                problems.append(
                    f"{dataset}/{split}/seed{seed}: {total} indices for "
                    f"{len(df)} molecules")
            for a, b in (("train", "test"), ("train", "valid"), ("valid", "test")):
                if set(getattr(sp, a)) & set(getattr(sp, b)):
                    problems.append(
                        f"{dataset}/{split}/seed{seed}: {a} and {b} overlap")
            if not sp.valid:
                problems.append(f"{dataset}/{split}/seed{seed}: empty valid set")

            leak = ""
            if split.startswith("scaffold"):
                tr = {scaf[i] for i in sp.train} - {""}
                te = {scaf[i] for i in sp.test} - {""}
                n_shared = len(tr & te)
                leak = str(n_shared)
                if n_shared and split == "scaffold_official":
                    # OGB's published split, which we don't control. Its
                    # splitter keeps stereochemistry in the scaffold
                    # (includeChirality=True), so mirror-image versions of
                    # one skeleton count as different families there but as
                    # the same family here. Report it, don't fail on it.
                    notes.append(
                        f"{split} seed{seed}: {n_shared} ring scaffolds appear "
                        f"in both train and test by our definition. This is "
                        f"OGB's published split: it counts stereoisomers "
                        f"(mirror-image versions of a scaffold) as different "
                        f"families, while murcko_scaffold() ignores "
                        f"stereochemistry. Not a problem with our splits.")
                elif n_shared:
                    problems.append(
                        f"{dataset}/{split}/seed{seed}: {n_shared} ring "
                        f"scaffolds in both train and test")
            else:
                leak = "n/a"

            sim, _ = D.test_familiarity(dataset, split, seed)
            sim_by_split.setdefault(split, []).append(float(sim.mean()))

            pos_tr = y[sp.train].mean()
            pos_va = y[sp.valid].mean() if sp.valid else float("nan")
            pos_te = y[sp.test].mean()
            if sp.valid and pos_va == 0:
                problems.append(
                    f"{dataset}/{split}/seed{seed}: validation set has no "
                    f"actives, so early stopping on PR-AUC will not work")

            print(f"{split:<18}{seed:>5}{len(sp.train):>8}{len(sp.valid):>7}"
                  f"{len(sp.test):>7}{leak:>6}{sim.mean():>8.3f}"
                  f"{pos_tr:>8.4f}{pos_te:>8.4f}")

    if "random" in sim_by_split and "scaffold" in sim_by_split:
        r = float(np.mean(sim_by_split["random"]))
        s = float(np.mean(sim_by_split["scaffold"]))
        print(f"\nmean nearest-neighbour Tanimoto similarity to train:")
        print(f"  random   {r:.3f}")
        print(f"  scaffold {s:.3f}   (drop of {r - s:.3f})")
        if s >= r:
            problems.append(
                f"{dataset}: scaffold split is not less familiar than random "
                f"({s:.3f} vs {r:.3f}) -- the premise of the project does not "
                f"hold on this dataset")
        elif r - s < 0.05:
            print("  warning: the drop is small. Check that the scaffold "
                  "split is doing real work on this dataset.")

    for note in notes:
        print(f"\n  note: {note}")

    sim, bins = D.test_familiarity(dataset, "scaffold", seeds[0])
    sp = D.load_split(dataset, "scaffold", seeds[0])
    yt = y[sp.test]
    print(f"\nfamiliarity bins, scaffold split seed {seeds[0]} "
          f"(cutoffs {D.FAMILIARITY_EDGES[1]}/{D.FAMILIARITY_EDGES[2]}):")
    thin = []
    for b, lab in enumerate(D.FAMILIARITY_LABELS):
        m = bins == b
        if m.sum() == 0:
            print(f"  {lab:<7} empty")
            thin.append(lab)
            continue
        act = int(yt[m].sum())
        print(f"  {lab:<7} {int(m.sum()):>5} molecules, {act:>4} actives, "
              f"sim {sim[m].min():.2f}-{sim[m].max():.2f}")
        if act < 5:
            thin.append(lab)
    if thin:
        print(f"  warning: {', '.join(thin)} has too few actives to score "
              f"reliably. Use binning='quantile' and report the edges.")
        _, bq = D.test_familiarity(dataset, "scaffold", seeds[0], binning="quantile")
        print("  quantile bin sizes:",
              {D.FAMILIARITY_LABELS[b]: int((bq == b).sum()) for b in range(3)})

    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", choices=("all",) + D.DATASETS)
    ap.add_argument("--seeds", default="0,1,2")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    names = D.DATASETS if args.dataset == "all" else (args.dataset,)

    all_problems: list[str] = []
    for name in names:
        try:
            all_problems += check(name, seeds)
        except Exception as exc:
            all_problems.append(f"{name}: check failed -- {type(exc).__name__}: {exc}")

    print(f"\n{'=' * 78}")
    if all_problems:
        print(f"{len(all_problems)} PROBLEM(S) -- do not share these splits yet:\n")
        for p in all_problems:
            print(f"  * {p}")
        sys.exit(1)
    print("ALL CHECKS PASSED. Splits are safe to share with the team.")


if __name__ == "__main__":
    main()
