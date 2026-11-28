"""Validate the data pipeline on 5,000 real NCI compounds (ships with RDKit).
No network, no GPU. Checks the properties Ana B. must guarantee to the team.
"""
import os, sys, tempfile, numpy as np, pandas as pd
tmp = tempfile.mkdtemp()
os.environ["SCAFFOLDGEN_DATA"] = tmp + "/data"
os.environ["SCAFFOLDGEN_SPLITS"] = tmp + "/splits"
os.environ["SCAFFOLDGEN_RESULTS"] = tmp + "/results"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
from scaffoldgen import data as D

SMI = "/usr/local/lib/python3.13/site-packages/rdkit/Data/NCI/first_5K.smi"
smis = [l.split("\t")[0].strip() for l in open(SMI) if l.strip()]
smis = [s for s in smis if Chem.MolFromSmiles(s) is not None]

# Fake labels correlated with scaffold, at MolHIV's 3.5% positive rate, so the
# split's label-balance behaviour is realistic.
rng = np.random.default_rng(0)
scaf = [D.murcko_scaffold(s) for s in smis]
uniq = sorted(set(scaf))
hot = set(rng.choice(uniq, size=max(1, len(uniq)//12), replace=False))
y = [1 if (sc in hot and rng.random() < 0.42) else 0 for sc in scaf]
df_real = pd.DataFrame({"smiles": smis, "y": y})
D.load_dataset = lambda name: df_real.copy()

print(f"{len(df_real)} real molecules | pos_rate={df_real.y.mean():.4f} "
      f"| {len(uniq)} distinct scaffolds")
print(f"empty-scaffold molecules (acyclic, no ring system): {scaf.count('')}")
sizes = pd.Series(scaf).value_counts()
print(f"largest scaffold family: {sizes.iloc[0]} molecules | "
      f"families with 1 member: {(sizes==1).sum()}")

D.build_splits("nci", seeds=(0, 1, 2), overwrite=True)

print("\n--- the guarantees Ana B. owes the team ---")
ok = True
for split in ("random", "scaffold"):
    for seed in (0, 1, 2):
        sp = D.load_split("nci", split, seed)
        n = len(sp.train) + len(sp.valid) + len(sp.test)
        assert n == len(df_real), f"{split}/{seed}: {n} != {len(df_real)}"
        assert not (set(sp.train) & set(sp.test)), "train/test index overlap"
        assert not (set(sp.train) & set(sp.valid)), "train/valid index overlap"
        assert len(sp.valid) > 0, "empty validation set"
        # Ring-bearing scaffolds must never be shared. Acyclic molecules
        # (scaffold "") are not relatives of each other and are expected on
        # both sides -- count them separately.
        tr = {scaf[i] for i in sp.train} - {""}
        te = {scaf[i] for i in sp.test} - {""}
        shared = len(tr & te)
        acyc_tr = sum(1 for i in sp.train if scaf[i] == "")
        acyc_te = sum(1 for i in sp.test if scaf[i] == "")
        sim, bins = D.test_familiarity("nci", split, seed)
        ytr = df_real.y.iloc[sp.train].mean()
        yte = df_real.y.iloc[sp.test].mean()
        flag = ""
        if split == "scaffold" and shared != 0:
            flag = "  <-- SCAFFOLD LEAK"; ok = False
        print(f"{split:9s} seed{seed} | sizes {len(sp.train):4d}/{len(sp.valid):4d}/{len(sp.test):4d}"
              f" | shared scaffolds {shared:4d} | mean NN sim {sim.mean():.3f}"
              f" | pos train {ytr:.3f} test {yte:.3f}"
              f" | acyclic {acyc_tr}/{acyc_te}{flag}")

print("\n--- is the scaffold split actually harder? ---")
for seed in (0, 1, 2):
    sr, _ = D.test_familiarity("nci", "random", seed)
    ss, _ = D.test_familiarity("nci", "scaffold", seed)
    print(f"seed{seed}: random {sr.mean():.3f} vs scaffold {ss.mean():.3f}"
          f"  (median {np.median(sr):.3f} vs {np.median(ss):.3f})")
    assert sr.mean() > ss.mean(), "scaffold split not less familiar than random!"

print("\n--- familiarity bins on the scaffold split (seed 0) ---")
sim, bins = D.test_familiarity("nci", "scaffold", 0)
sp = D.load_split("nci", "scaffold", 0)
yv = df_real.y.to_numpy()[sp.test]
for b, lab in enumerate(D.FAMILIARITY_LABELS):
    m = bins == b
    if m.sum() == 0:
        print(f"  {lab:7s}: 0 molecules  <-- unusable bin"); continue
    print(f"  {lab:7s}: {m.sum():4d} molecules | {int(yv[m].sum()):3d} actives"
          f" | sim range {sim[m].min():.2f}-{sim[m].max():.2f}")
print("  (fixed cutoffs 0.3 / 0.5)")
sim_q, bins_q = D.test_familiarity("nci", "scaffold", 0, binning="quantile")
print("  quantile bins:", {D.FAMILIARITY_LABELS[b]: int((bins_q == b).sum()) for b in range(3)})

print("\nALL CHECKS PASSED" if ok else "\nFAILURES ABOVE")
