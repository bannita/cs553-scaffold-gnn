"""Test _load_ogb against OGB's real file layout, without a download.
Builds the mapping/mol.csv.gz files OGB would produce, with each dataset's
real label column name, and checks the loader finds the right column.
"""
import os, sys, gzip, tempfile
import pandas as pd
tmp = tempfile.mkdtemp()
os.environ["SCAFFOLDGEN_DATA"] = tmp
os.environ["SCAFFOLDGEN_SPLITS"] = tmp + "/splits"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scaffoldgen import data as D

LAYOUTS = {
    "molhiv": ("ogbg_molhiv", ["smiles", "HIV_active"]),
    "bace":   ("ogbg_molbace", ["smiles", "Class"]),
    "bbbp":   ("ogbg_molbbbp", ["smiles", "p_np"]),
}
SMI = ["CCO", "c1ccccc1", "CC(=O)O", "c1ccncc1"]

for name, (folder, cols) in LAYOUTS.items():
    d = os.path.join(tmp, folder, "mapping")
    os.makedirs(d, exist_ok=True)
    df = pd.DataFrame({cols[0]: SMI, cols[1]: [0, 1, 0, 1]})
    with gzip.open(os.path.join(d, "mol.csv.gz"), "wt") as fh:
        df.to_csv(fh, index=False)
    out = D.load_dataset(name)
    assert list(out.columns) == ["smiles", "y"], out.columns
    assert out.y.tolist() == [0, 1, 0, 1], out.y.tolist()
    assert out.smiles.tolist() == SMI
    print(f"{name:8s} folder={folder:14s} label column '{cols[1]}' -> y   OK")

# the mol_id variant some OGB versions ship
d = os.path.join(tmp, "ogbg_molbbbp", "mapping")
with gzip.open(os.path.join(d, "mol.csv.gz"), "wt") as fh:
    pd.DataFrame({"mol_id": ["BBBP-1","BBBP-2","BBBP-3","BBBP-4"],
                  "smiles": SMI, "p_np": [1,1,0,0]}).to_csv(fh, index=False)
out = D.load_dataset("bbbp")
assert out.y.tolist() == [1,1,0,0]
print("bbbp     with a mol_id column present                  OK")

# an unexpected multi-task file must raise, not silently pick a column
d = os.path.join(tmp, "ogbg_molbace", "mapping")
with gzip.open(os.path.join(d, "mol.csv.gz"), "wt") as fh:
    pd.DataFrame({"smiles": SMI, "Class": [0,1,0,1],
                  "extra": [1,1,1,1]}).to_csv(fh, index=False)
try:
    D.load_dataset("bace", allow_csv_fallback=False)
    print("FAIL: multi-column file did not raise")
    sys.exit(1)
except ValueError as e:
    assert "expected one label column" in str(e), e
    print("bace     ambiguous label columns -> raises            OK")

print("\nOGB LOADER TESTS PASSED")
