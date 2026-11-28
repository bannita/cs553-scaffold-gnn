"""
Data layer: loading, scaffolds, splits, fingerprints, similarity bins.
Owner: Ana B.

Everything downstream (RF, FFN, D-MPNN, GIN, ChemBERTa) consumes
split index files written by this module, so every model sees
exactly the same train/valid/test molecules.

Key invariant
-------------
For ogbg-molhiv, the row order of the dataframe returned by
load_dataset() is IDENTICAL to the order of the PyG dataset from
    PygGraphPropPredDataset('ogbg-molhiv')
because we read SMILES from OGB's own mapping file. So an index in a
split file can be used directly to index the PyG dataset (graph
models) or the dataframe (fingerprint / SMILES models).
"""

from __future__ import annotations

import gzip
import json
import os
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd

DATA_ROOT = os.environ.get("SCAFFOLDGEN_DATA", "/home/jovyan/data")
SPLIT_DIR = os.environ.get("SCAFFOLDGEN_SPLITS", "/home/jovyan/scaffoldgen/splits")

# All three datasets come through OGB. OGB adopts the MoleculeNet datasets and
# republishes them with unified splits and graph features, so these ARE the
# MoleculeNet datasets the proposal cites -- just packaged consistently.
#
# Why not the raw MoleculeNet CSVs: going through OGB gives us (a) molecule
# counts that match the published figures exactly (1,513 / 2,039), (b) the PyG
# graph version of all three datasets, not just MolHIV, so Nino and Nata can
# run BACE and BBBP too, and (c) row order that matches those graphs, which is
# what lets one split file serve every model. A CSV read with unparseable rows
# dropped would silently break that alignment.
OGB_NAMES = {
    "molhiv": "ogbg-molhiv",
    "bace": "ogbg-molbace",
    "bbbp": "ogbg-molbbbp",
}

# Fallback only, if the cluster cannot reach the OGB servers. These give
# slightly different molecule counts, and the row order will NOT match the PyG
# graphs -- see load_dataset().
MOLNET_URLS = {
    "bace": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/bace.csv",
    "bbbp": "https://deepchemdata.s3.us-west-1.amazonaws.com/datasets/BBBP.csv",
}
# (smiles column, label column) per MoleculeNet dataset
MOLNET_COLS = {
    "bace": ("mol", "Class"),
    "bbbp": ("smiles", "p_np"),
}

DATASETS = ("molhiv", "bace", "bbbp")


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_dataset(name: str, allow_csv_fallback: bool = True) -> pd.DataFrame:
    """Return a dataframe with columns: smiles, y (int 0/1).

    name in {"molhiv", "bace", "bbbp"}. Row order matches the PyG graph
    dataset of the same name, so one split file serves every model.

    If OGB cannot be reached, BACE and BBBP fall back to the MoleculeNet CSVs.
    That fallback drops unparseable SMILES, so the row order no longer matches
    the PyG graphs -- the graph models must then use the CSV order too, or not
    use these datasets. load_dataset() prints a warning when it happens.
    """
    name = name.lower()
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; expected one of {DATASETS}")
    try:
        return _load_ogb(name)
    except Exception as exc:
        if name == "molhiv" or not allow_csv_fallback:
            raise
        print(f"[data] warning: OGB load of {name} failed ({type(exc).__name__}: "
              f"{exc}). Falling back to the MoleculeNet CSV. Row order will NOT "
              f"match the PyG graphs for this dataset -- tell the team.")
        return _load_molnet(name)


def _download_ogb(ogb_name: str) -> None:
    """Fetch an OGB dataset, preferring the PyG loader.

    The PyG loader also builds the graph tensors the GNNs need, so when it is
    available this one call sets up everyone's data. But it needs
    torch_geometric, and data prep blocks the whole team -- so if that import
    fails we fall back to OGB's library-agnostic loader, which needs only
    numpy and pandas and downloads the same archive. The split files can then
    be built on a machine with no deep-learning stack at all.

    Note on a confusing error: ogb/graphproppred/__init__.py wraps its PyG
    import in `try: ... except ImportError: pass`. So a broken or missing
    torch_geometric shows up as "cannot import name PygGraphPropPredDataset",
    which looks like the class does not exist. It does. Check torch_geometric.
    """
    try:
        from ogb.graphproppred import PygGraphPropPredDataset

        PygGraphPropPredDataset(ogb_name, root=DATA_ROOT)
        return
    except ImportError as exc:
        print(f"[data] PyG loader unavailable ({exc}). This usually means "
              f"torch_geometric is missing or broken, NOT that OGB changed. "
              f"Downloading with the library-agnostic loader instead; the "
              f"graph models will need torch_geometric before they can run.")

    from ogb.graphproppred import GraphPropPredDataset

    GraphPropPredDataset(ogb_name, root=DATA_ROOT)


def _load_ogb(name: str) -> pd.DataFrame:
    """SMILES + labels straight from the OGB download, in OGB row order."""
    ogb_name = OGB_NAMES[name]
    folder = ogb_name.replace("-", "_")
    base = os.path.join(DATA_ROOT, folder, "mapping", "mol.csv.gz")
    if not os.path.exists(base):
        _download_ogb(ogb_name)
    with gzip.open(base, "rt") as fh:
        df = pd.read_csv(fh)

    # The label column is named differently per dataset (HIV_active, Class,
    # p_np), so identify it by elimination rather than hard-coding each one.
    label_cols = [c for c in df.columns if c not in ("smiles", "mol_id")]
    if len(label_cols) != 1:
        raise ValueError(
            f"{ogb_name}: expected one label column, found {label_cols}. "
            f"These loaders assume a single-task dataset.")
    out = pd.DataFrame({
        "smiles": df["smiles"].astype(str),
        "y": df[label_cols[0]].astype(int),
    })
    return out.reset_index(drop=True)


def _load_molnet(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_ROOT, f"{name}.csv")
    if not os.path.exists(path):
        os.makedirs(DATA_ROOT, exist_ok=True)
        import urllib.request

        urllib.request.urlretrieve(MOLNET_URLS[name], path)
    df = pd.read_csv(path)
    smi_col, y_col = MOLNET_COLS[name]
    out = pd.DataFrame({"smiles": df[smi_col].astype(str), "y": df[y_col].astype(int)})
    # Drop molecules RDKit cannot parse; graph models would fail on them too.
    from rdkit import Chem, RDLogger

    RDLogger.DisableLog("rdApp.*")
    keep = [i for i, s in enumerate(out["smiles"]) if Chem.MolFromSmiles(s) is not None]
    dropped = len(out) - len(keep)
    if dropped:
        print(f"[data] {name}: dropped {dropped} unparseable SMILES")
    return out.iloc[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# scaffolds
# --------------------------------------------------------------------------
def murcko_scaffold(smiles: str, include_chirality: bool = False) -> str:
    """Bemis-Murcko scaffold as a SMILES string; '' if unparseable."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)
    except Exception:
        return ""


def scaffold_groups(smiles_list, acyclic_as_own_group: bool = True) -> dict[str, list[int]]:
    """scaffold SMILES -> list of row indices sharing it.

    A molecule with no ring system has no Bemis-Murcko scaffold, and
    MurckoScaffoldSmiles returns "" for it. Those molecules are NOT chemical
    relatives of each other -- they have nothing in common but the absence of
    a ring. Lumping them under one key makes a single enormous pseudo-family:
    on 5,000 real NCI compounds that family held 1,149 molecules, 23% of the
    set, and the splitter handed the whole thing to the validation set, which
    then contained zero actives and broke early stopping.

    So by default each acyclic molecule becomes its own group. Set
    acyclic_as_own_group=False to reproduce the DeepChem / Chemprop behaviour
    of treating them as one family.
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = murcko_scaffold(smi)
        if scaf == "" and acyclic_as_own_group:
            groups[f"__acyclic_{i}"].append(i)
        else:
            groups[scaf].append(i)
    return dict(groups)


def count_acyclic(smiles_list) -> int:
    """How many molecules have no ring system. Report this alongside splits."""
    return sum(1 for s in smiles_list if murcko_scaffold(s) == "")


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------
@dataclass
class Split:
    train: list[int]
    valid: list[int]
    test: list[int]

    def as_dict(self):
        return {"train": self.train, "valid": self.valid, "test": self.test}


def random_split(n: int, seed: int, frac=(0.8, 0.1, 0.1)) -> Split:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_tr = int(frac[0] * n)
    n_va = int(frac[1] * n)
    return Split(
        train=sorted(int(i) for i in idx[:n_tr]),
        valid=sorted(int(i) for i in idx[n_tr : n_tr + n_va]),
        test=sorted(int(i) for i in idx[n_tr + n_va :]),
    )


def scaffold_split(smiles_list, seed: int | None = None, frac=(0.8, 0.1, 0.1),
                   balanced: bool = True, acyclic_as_own_group: bool = True) -> Split:
    """Split so that no scaffold family appears in two sets.

    balanced=True  : shuffle scaffold groups (big groups to train first),
                     so different seeds give different scaffold splits.
                     This is Chemprop's 'scaffold_balanced'.
    balanced=False : deterministic -- groups sorted by size, largest to
                     train. Same split for every seed.
    """
    groups = scaffold_groups(smiles_list, acyclic_as_own_group)
    sets = list(groups.values())
    n = len(smiles_list)
    n_tr, n_va = int(frac[0] * n), int(frac[1] * n)

    if balanced:
        rng = np.random.default_rng(seed)
        big = [s for s in sets if len(s) > n_va / 2]
        small = [s for s in sets if len(s) <= n_va / 2]
        rng.shuffle(big)
        rng.shuffle(small)
        sets = big + small
    else:
        sets = sorted(sets, key=lambda s: (-len(s), s[0]))

    n_te = n - n_tr - n_va
    buckets = {"train": [], "valid": [], "test": []}
    targets = {"train": n_tr, "valid": n_va, "test": n_te}

    # Greedy fill-the-emptiest: assign each scaffold family to whichever set
    # is furthest below its target. A plain "train, else valid, else test"
    # cascade can leave valid empty when a dataset has few large families,
    # and an empty valid set silently disables early stopping.
    for grp in sets:
        # relative deficit, so the small valid/test targets are not starved
        # by train's much larger one
        rel = {k: (targets[k] - len(buckets[k])) / max(1, targets[k]) for k in buckets}
        pick = max(rel, key=lambda k: (rel[k], k == "train"))
        buckets[pick] += grp

    if not buckets["valid"]:
        raise RuntimeError(
            "scaffold split produced an empty validation set -- too few "
            "scaffold families for these fractions. Lower the valid fraction "
            "or check the SMILES column."
        )
    return Split(sorted(buckets["train"]), sorted(buckets["valid"]),
                 sorted(buckets["test"]))


def official_split(name: str) -> Split:
    """OGB's own scaffold split for this dataset.

    All three datasets have one, so each gets a published reference point to
    check our own splitting code against. OGB uses the most challenging
    version of the scaffold split, where test molecules are maximally diverse,
    so expect it to be at least as hard as ours.
    """
    # Read OGB's split files directly instead of building the dataset object.
    # Building it torch.load()s OGB's processed graph file, which fails on
    # PyTorch >= 2.6 ("Weights only load failed"). The split itself is three
    # plain CSVs of row indices -- exactly what ds.get_idx_split() reads -- so
    # no torch.load and no weights_only=False is needed.
    ogb_name = OGB_NAMES[name]
    split_dir = os.path.join(DATA_ROOT, ogb_name.replace("-", "_"), "split", "scaffold")
    if not os.path.exists(os.path.join(split_dir, "train.csv.gz")):
        _download_ogb(ogb_name)
    s = {
        part: pd.read_csv(os.path.join(split_dir, f"{part}.csv.gz"),
                          compression="gzip", header=None).values.T[0]
        for part in ("train", "valid", "test")
    }
    return Split(
        train=[int(i) for i in s["train"]],
        valid=[int(i) for i in s["valid"]],
        test=[int(i) for i in s["test"]],
    )


def split_path(dataset: str, split: str, seed: int) -> str:
    return os.path.join(SPLIT_DIR, f"{dataset}_{split}_seed{seed}.json")


def build_splits(dataset: str, seeds=(0, 1, 2), overwrite: bool = False) -> None:
    """Write split files for one dataset: random + scaffold, one per seed.

    Each dataset also gets 'scaffold_official' (seed 0 only), which is OGB's
    own published split -- a fixed reference to check our splitter against.
    """
    os.makedirs(SPLIT_DIR, exist_ok=True)
    df = load_dataset(dataset)
    n = len(df)
    smiles = df["smiles"].tolist()

    for seed in seeds:
        for split_name, fn in (
            ("random", lambda: random_split(n, seed)),
            ("scaffold", lambda: scaffold_split(smiles, seed=seed, balanced=True)),
        ):
            p = split_path(dataset, split_name, seed)
            if os.path.exists(p) and not overwrite:
                continue
            sp = fn()
            _write_split(p, sp, dataset, split_name, seed, df)

    p = split_path(dataset, "scaffold_official", 0)
    if overwrite or not os.path.exists(p):
        try:
            _write_split(p, official_split(dataset), dataset, "scaffold_official", 0, df)
        except Exception as exc:  # OGB not installed / download failed
            print(f"[data] skipped official split for {dataset}: {exc}")


def _write_split(path, sp: Split, dataset, split_name, seed, df):
    rec = sp.as_dict()
    n = len(df)
    sizes = {k: len(v) for k, v in sp.as_dict().items()}
    rec["meta"] = {
        "dataset": dataset,
        "split": split_name,
        "seed": seed,
        "n_total": int(n),
        "sizes": sizes,
        "pos_rate": {
            k: round(float(df["y"].iloc[v].mean()), 5) if len(v) else None
            for k, v in sp.as_dict().items()
        },
    }
    with open(path, "w") as fh:
        json.dump(rec, fh)
    m = rec["meta"]
    print(f"[data] wrote {os.path.basename(path)}  sizes={sizes}  pos_rate={m['pos_rate']}")

    # A scaffold split cannot hit the target fractions exactly -- families are
    # indivisible. But a big miss means one family dominated, which is worth
    # knowing before anyone trains on it.
    for name, target in (("valid", 0.1 * n), ("test", 0.1 * n)):
        got = sizes[name]
        if target > 0 and abs(got - target) / target > 0.5:
            print(f"[data]   warning: {name} set is {got}, target was about "
                  f"{int(target)} -- one scaffold family may dominate")
    if m["pos_rate"]["valid"] == 0:
        print("[data]   warning: validation set has no actives, so early "
              "stopping on PR-AUC will not work for this split")


def load_split(dataset: str, split: str, seed: int) -> Split:
    """Read a split file. Everyone must use this -- never re-split locally."""
    p = split_path(dataset, split, seed)
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"{p} missing. Run: python -m scaffoldgen.data --dataset {dataset}"
        )
    with open(p) as fh:
        rec = json.load(fh)
    return Split(rec["train"], rec["valid"], rec["test"])


# --------------------------------------------------------------------------
# fingerprints + similarity
# --------------------------------------------------------------------------
def morgan_fingerprints(smiles_list, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """(n_molecules, n_bits) uint8 array. Unparseable SMILES -> all zeros."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator

    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        fp = gen.GetFingerprintAsNumPy(mol)
        out[i] = fp.astype(np.uint8)
    return out


def fingerprint_cache(dataset: str, radius: int = 2, n_bits: int = 2048) -> np.ndarray:
    """Fingerprints for a whole dataset, cached to disk."""
    os.makedirs(os.path.join(DATA_ROOT, "fp"), exist_ok=True)
    path = os.path.join(DATA_ROOT, "fp", f"{dataset}_r{radius}_b{n_bits}.npy")
    if os.path.exists(path):
        return np.load(path)
    fp = morgan_fingerprints(load_dataset(dataset)["smiles"].tolist(), radius, n_bits)
    np.save(path, fp)
    return fp


def tanimoto_max_similarity(query_fp: np.ndarray, ref_fp: np.ndarray,
                            block: int = 512) -> np.ndarray:
    """For each query molecule, its Tanimoto similarity to the NEAREST
    reference molecule. Used to measure how familiar a test molecule is.

    Returns array of shape (n_query,) in [0, 1].
    """
    q = query_fp.astype(np.float32)
    r = ref_fp.astype(np.float32)
    r_sum = r.sum(axis=1)
    out = np.zeros(len(q), dtype=np.float32)
    for start in range(0, len(q), block):
        qb = q[start : start + block]
        inter = qb @ r.T                                  # |A & B|
        union = qb.sum(axis=1)[:, None] + r_sum[None, :] - inter
        with np.errstate(divide="ignore", invalid="ignore"):
            sim = np.where(union > 0, inter / union, 0.0)
        out[start : start + block] = sim.max(axis=1)
    return out


FAMILIARITY_EDGES = (0.0, 0.3, 0.5, 1.01)   # low / medium / high
FAMILIARITY_LABELS = ("low", "medium", "high")


def familiarity_bins(sim: np.ndarray, edges=FAMILIARITY_EDGES) -> np.ndarray:
    """Bin index (0..len(edges)-2) for each similarity value."""
    return np.clip(np.digitize(sim, np.asarray(edges[1:-1]), right=False), 0, len(edges) - 2)


def familiarity_bins_quantile(sim: np.ndarray, n_bins: int = 3):
    """Equal-COUNT bins instead of fixed similarity cutoffs.

    Prefer this on MolHIV: with 3.5% positives, a fixed-cutoff bin can end up
    with too few actives to compute an AUC at all. Returns (bin_idx, edges) --
    report the edges in the write-up, since they are dataset-specific.
    """
    edges = np.quantile(sim, np.linspace(0, 1, n_bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    edges = np.unique(edges)
    idx = np.clip(np.digitize(sim, edges[1:-1], right=False), 0, len(edges) - 2)
    return idx, edges


def test_familiarity(dataset: str, split: str, seed: int,
                     radius: int = 2, n_bits: int = 2048,
                     binning: str = "fixed", n_bins: int = 3):
    """Nearest-neighbour similarity of every TEST molecule to the TRAIN set.

    binning='fixed'    -> FAMILIARITY_EDGES cutoffs (comparable across splits)
    binning='quantile' -> equal-count bins (guarantees usable bin sizes)

    Returns (sim, bin_idx) aligned with load_split(...).test.
    """
    sp = load_split(dataset, split, seed)
    fp = fingerprint_cache(dataset, radius, n_bits)
    sim = tanimoto_max_similarity(fp[sp.test], fp[sp.train])
    if binning == "quantile":
        return sim, familiarity_bins_quantile(sim, n_bins)[0]
    return sim, familiarity_bins(sim)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _main():
    import argparse

    ap = argparse.ArgumentParser(description="Build shared split files.")
    ap.add_argument("--dataset", default="all", choices=("all",) + DATASETS)
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--report", action="store_true",
                    help="print scaffold overlap + familiarity summary")
    args = ap.parse_args()

    seeds = tuple(int(s) for s in args.seeds.split(","))
    names = DATASETS if args.dataset == "all" else (args.dataset,)
    for name in names:
        print(f"\n=== {name} ===")
        build_splits(name, seeds=seeds, overwrite=args.overwrite)
        if args.report:
            _report(name, seeds[0])


def _report(dataset: str, seed: int):
    df = load_dataset(dataset)
    smiles = df["smiles"].tolist()
    groups = scaffold_groups(smiles)
    print(f"[report] {len(df)} molecules, {len(groups)} scaffolds, "
          f"pos_rate={df['y'].mean():.4f}")
    for split in ("random", "scaffold"):
        sp = load_split(dataset, split, seed)
        # Count ring-bearing scaffolds only. Acyclic molecules share the
        # empty scaffold string and are deliberately on both sides of a
        # scaffold split, so counting "" here would report a leak of 1 on a
        # split that is actually clean. verify_splits.py excludes it the
        # same way.
        tr = {murcko_scaffold(smiles[i]) for i in sp.train} - {""}
        te = {murcko_scaffold(smiles[i]) for i in sp.test} - {""}
        shared = len(tr & te)
        sim, bins = test_familiarity(dataset, split, seed)
        counts = {FAMILIARITY_LABELS[b]: int((bins == b).sum()) for b in range(3)}
        print(f"[report] {split:9s} shared scaffolds train/test = {shared:5d} | "
              f"mean NN sim = {sim.mean():.3f} | bins {counts}")


if __name__ == "__main__":
    _main()
