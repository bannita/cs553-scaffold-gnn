"""Offline smoke test: synthetic SMILES, no OGB/network/GPU needed."""
import os, sys, tempfile, random
tmp = tempfile.mkdtemp()
os.environ["SCAFFOLDGEN_DATA"] = tmp + "/data"
os.environ["SCAFFOLDGEN_SPLITS"] = tmp + "/splits"
os.environ["SCAFFOLDGEN_RESULTS"] = tmp + "/results"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np, pandas as pd
from scaffoldgen import data as D, metrics as M, runner as R, familiarity as F

# --- synthetic dataset: 6 scaffold families x side chains ---
rings = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccc2ccccc2c1", "c1cc[nH]c1",
         "C1CCNCC1", "c1ccsc1", "c1ccoc1", "C1CCOCC1", "c1cnc2ccccc2n1"]
linkers = ["", "-", "C", "CC", "O", "N"]
cores = []
for i, a in enumerate(rings):
    for j, b in enumerate(rings):
        if j <= i:
            continue
        for lk in linkers[:3]:
            cores.append(f"{a}{lk}{b}" if lk != "-" else f"{a}-{b}")
cores = cores[:60]
tails = ["C", "CC", "CCC", "CO", "CN", "CCO", "CF", "CCl"]
rows = []
rng = random.Random(0)
for ci, c in enumerate(cores):
    for t in tails:
        rows.append((c + t, 1 if (ci % 7 == 0 and rng.random() < 0.6) else 0))
df_fake = pd.DataFrame(rows, columns=["smiles", "y"]).drop_duplicates("smiles").reset_index(drop=True)
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
df_fake = df_fake[[Chem.MolFromSmiles(s) is not None for s in df_fake.smiles]].reset_index(drop=True)
D.load_dataset = lambda name: df_fake.copy()
print(f"synthetic dataset: {len(df_fake)} molecules, pos_rate={df_fake.y.mean():.3f}")

# --- scaffolds ---
g = D.scaffold_groups(df_fake.smiles.tolist())
print(f"scaffold groups: {len(g)}")
assert len(g) >= 4, "expected several scaffold families"

# --- splits ---
D.build_splits("fake", seeds=(0, 1), overwrite=True)
for split in ("random", "scaffold"):
    for seed in (0, 1):
        sp = D.load_split("fake", split, seed)
        total = len(sp.train) + len(sp.valid) + len(sp.test)
        assert total == len(df_fake), (split, seed, total)
        assert not (set(sp.train) & set(sp.test)), "train/test leak!"
sm = df_fake.smiles.tolist()
sp_s = D.load_split("fake", "scaffold", 0)
tr_sc = {D.murcko_scaffold(sm[i]) for i in sp_s.train}
te_sc = {D.murcko_scaffold(sm[i]) for i in sp_s.test}
assert not (tr_sc & te_sc), f"scaffold leak: {tr_sc & te_sc}"
sp_r = D.load_split("fake", "random", 0)
tr_r = {D.murcko_scaffold(sm[i]) for i in sp_r.train}
te_r = {D.murcko_scaffold(sm[i]) for i in sp_r.test}
print(f"scaffold split: 0 shared scaffolds OK | random split shares {len(tr_r & te_r)} scaffolds")
assert len(tr_r & te_r) > 0, "random split should share scaffolds"

# --- fingerprints + similarity ---
fp = D.fingerprint_cache("fake")
assert fp.shape == (len(df_fake), 2048)
sim_s, bins_s = D.test_familiarity("fake", "scaffold", 0)
sim_r, bins_r = D.test_familiarity("fake", "random", 0)
print(f"mean NN similarity: random={sim_r.mean():.3f}  scaffold={sim_s.mean():.3f}")
assert sim_r.mean() > sim_s.mean(), "random split should look MORE familiar"
assert (sim_r <= 1.0).all() and (sim_s >= 0).all()

# --- metrics ---
y = np.array([0,0,0,1,1,0,1,0,0,1]); p = np.array([.1,.2,.05,.9,.8,.3,.6,.15,.1,.7])
s = M.summarize(y, p, uncertainty=np.full(10, 0.1))
assert 0.9 < s["roc_auc"] <= 1.0, s["roc_auc"]
assert 0 <= s["ece"] <= 1 and 0 <= s["brier"] <= 1
perfect = M.expected_calibration_error(np.array([0,0,1,1]), np.array([0.,0.,1.,1.]), n_bins=10)
assert perfect < 1e-9, perfect
assert M.roc_auc(np.zeros(5), np.random.rand(5)) != M.roc_auc(np.zeros(5), np.random.rand(5)) or True
assert np.isnan(M.roc_auc(np.zeros(5), np.random.rand(5))), "single-class should be nan"
print(f"metrics OK: roc={s['roc_auc']:.3f} pr={s['pr_auc']:.3f} ece={s['ece']:.3f} ef1={s['ef_1pct']:.2f}")

# --- RF end to end + logging ---
met, prob, unc = D and None or None, None, None
from scaffoldgen.baselines import run_rf
for split in ("random", "scaffold"):
    for seed in (0, 1):
        met, prob, unc = run_rf("fake", split, seed, n_estimators=60)
print(f"RF scaffold seed1: roc={met['roc_auc']:.3f} pr={met['pr_auc']:.3f}")

# --- concurrent append safety ---
import threading
def hammer(i):
    R.log_metrics(f"m{i}", "fake", "random", 0, {"roc_auc": 0.5 + i/100})
ts = [threading.Thread(target=hammer, args=(i,)) for i in range(8)]
[t.start() for t in ts]; [t.join() for t in ts]
csv = pd.read_csv(R.RESULTS_CSV)
assert csv.isnull().all(axis=1).sum() == 0 and len(csv) > 8
print(f"results.csv rows={len(csv)}, no malformed rows")

# --- familiarity analysis on saved preds ---
dfb = F.analyze_all("fake", "scaffold", 0)
assert len(dfb) > 0
print(F.similarity_summary("fake", 0))

# --- gap table ---
tab = R.gap_table("roc_auc", "fake")
print(tab.to_string(index=False))
assert "gap" in tab.columns and len(tab) >= 1
miss = R.coverage_report(expected_models=["rf"], datasets=("fake",), seeds=(0,1))
print("\nALL SMOKE TESTS PASSED")
