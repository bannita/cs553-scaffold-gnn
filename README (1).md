# CS 553 Final Project: shared folder

Graph Neural Networks and Scaffold Generalization in Molecular Bioactivity Prediction.

## Where things live

**GitHub repo = code.** `harness.py`, `test_harness.py`, `README.md`, and `notebooks/` (each person keeps their own notebook files and only edits those, which avoids merge conflicts).

**Shared Google Drive folder `CS553_Project` = data and outputs.**

```
CS553_Project/
  results/            one CSV per model: rf.csv, ffn.csv, dmpnn.csv, ...  (written by the harness)
  predictions/        per-molecule predictions (.npz), written by the harness
  data/               Ana B.'s data: mapping/<ds>/mol.csv.gz, fingerprints/<ds>_r2_b2048.npy, splits/<ds>_<split>_seed<k>.json
  checkpoints/        saved model weights (Nata, Ana M., Nino)
```

The harness writes `results/` and `predictions/` into whatever folder the environment variable
`CS553_PROJECT` points to. In Colab, set it to the Drive folder BEFORE `import harness`
(see the setup cell below). Nobody pushes results to GitHub, so results never cause merge conflicts.

Never edit another person's file in `results/` by hand. The harness writes them.

## Colab setup cell (put this at the top of every notebook)

```python
import os, sys
from google.colab import drive
drive.mount('/content/drive')

# data + results live on shared Drive (must be set BEFORE importing harness)
os.environ['CS553_PROJECT'] = '/content/drive/MyDrive/CS553_Project'

# code lives in the public GitHub repo
REPO = '/content/repo'
if not os.path.exists(REPO):
    !git clone https://github.com/bannita/cs553-scaffold-gnn.git {REPO}
else:
    !git -C {REPO} pull
sys.path.insert(0, REPO)

!pip -q install rdkit ogb scikit-learn pandas matplotlib
import harness
```

To save your own notebook to GitHub: in Colab use **File > Save a copy in GitHub** (authorize GitHub once).
The repo is public, so never commit passwords, tokens, or personal info.

## Data and split conventions (read this before running a model)

- Ana B.'s files use the dataset names `molhiv`, `bace`, `bbbp`. The harness uses `ogbg-molhiv`, `bace`, `bbbp`.
  Always pass the HARNESS name to `harness.*`. (`ogbg-molhiv` <-> `molhiv` is the only difference.)
- **There is one split file per seed, and they are different splits**: `<ds>_<split>_seed0.json`, `..._seed1.json`,
  `..._seed2.json`. Convention: **run seed k = split file k + model seed k**. Mean +/- std over the 3 seeds
  therefore includes split-to-split variation.
- Rows in split files index the molecules in OGB order. `test_idx` in the saved predictions is that row number.
- Never re-split locally and never regenerate the split files.
- **Ensembles** need every member to be tested on the SAME molecules. Train all members on ONE split file
  (use seed 0's) with different model seeds, and log the ensemble as seed 0.
- `scaffoldgen/runner.py` and `scaffoldgen/metrics.py` write to a different results file (`results.csv`) with
  different dataset names. The team standard is `harness.py`. Do not mix the two.

## Fingerprint baselines (RF + FFN)

Easiest: open `notebooks/ani_run_baselines.ipynb` in Colab and run the cells top to bottom. Or, from any notebook
that ran the setup cell:

```python
import fp_baselines
fp_baselines.check_all()            # validates all data and split files, prints split sizes
fp_baselines.run_all()              # EVERYTHING: rf + ffn x 3 datasets x 2 splits x 3 seeds (skips finished work)
fp_baselines.run_rf("bace")         # or one model on one dataset
fp_baselines.run_ffn("bace")        # needs torch; Colab has it. Switch to a GPU for MolHIV.
```

Results and predictions are written to the shared Drive automatically; nobody copies files by hand.
`run_ffn` logs three models per dataset and split: `ffn` (one network per seed), `ffn_mcd` (the same network predicting
with dropout on, 20 passes averaged) and `ffn_ens` (3 networks on ONE split file, logged as seed 0).
Chosen hyperparameters are saved in `results/<model>_configs.json`.

## Team conventions (the harness rejects anything else)

| Thing | Allowed values |
|---|---|
| dataset | `ogbg-molhiv`, `bace`, `bbbp` |
| split | `random`, `scaffold` |
| seeds | `0`, `1`, `2` |
| metrics | `roc_auc`, `pr_auc`, `ece` |

Suggested model names (one results file each):
`rf`, `ffn`, `ffn_ens`, `ffn_mcd`, `dmpnn`, `dmpnn_ens`, `dmpnn_mcd`, `gin_scratch`, `gin_pretrained`, `gin_ens`, `gin_mcd`, `chemberta`, `chemberta_ens`, `chemberta_mcd`.
Letters, digits, `_`, `-` and `.` only.

## How to use the harness

1. Train your model on the TRAIN split with the given seed.
2. Predict the **probability of the positive (active) class** on the TEST split. If your model outputs logits, apply a sigmoid first.
3. Return a dict from a function and let the harness do the rest:

```python
import harness

def train_and_predict(seed):
    # ... train with this seed, predict on the test split ...
    return {
        "y_true": y_test,        # 0/1 labels
        "y_prob": p_test,        # probabilities in [0, 1]
        "test_idx": test_idx,    # index of each test molecule in the dataset (needed for familiarity analysis)
        # optional extras, saved as-is, e.g. "y_std": spread from ensemble / MC dropout
    }

harness.run_seeds("rf", train_and_predict, dataset="bace", split="scaffold")
```

Re-running the same model/dataset/split/seed replaces the old rows. It never duplicates them.

## Reading results

```python
harness.load_results()          # every row from every model
harness.summarize()             # mean, std, n_seeds per model/dataset/split/metric
harness.scaffold_gap("roc_auc") # random-split score minus scaffold-split score (the core result)
```

## What each metric means

- **ROC-AUC**: chance the model ranks a random active molecule above a random inactive one (0.5 = guessing, 1 = perfect).
- **PR-AUC**: precision-recall area. More informative when actives are rare (HIV is about 3.5% active).
- **ECE**: expected calibration error on the active-class probability, 10 equal-width bins. 0 = honest confidence.

## Rules

1. Same seeds, same split files, same harness for everyone.
2. Test on the TEST split only. Never tune on it.
3. If you find a bug in `harness.py`, tell Ani instead of patching your own copy, so everyone stays consistent.
