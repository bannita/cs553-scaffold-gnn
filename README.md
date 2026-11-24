# CS 553 Final Project: shared folder

Graph Neural Networks and Scaffold Generalization in Molecular Bioactivity Prediction.

## Where things live

**GitHub repo = code.** `harness.py`, `test_harness.py`, `README.md`, and `notebooks/` (each person keeps their own notebook files and only edits those, which avoids merge conflicts).

**Shared Google Drive folder `CS553_Project` = data and outputs.**

```
CS553_Project/
  results/            one CSV per model: rf.csv, ffn.csv, dmpnn.csv, ...  (written by the harness)
  predictions/        per-molecule predictions (.npz), written by the harness
  data/               datasets, split files, fingerprints (Ana B.)
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
    !git clone https://github.com/OWNER/REPO_NAME.git {REPO}
else:
    !git -C {REPO} pull
sys.path.insert(0, REPO)

!pip -q install rdkit ogb scikit-learn pandas matplotlib
import harness
```

Replace `OWNER/REPO_NAME` with the real repo path.

To save your own notebook to GitHub: in Colab use **File > Save a copy in GitHub** (authorize GitHub once).
The repo is public, so never commit passwords, tokens, or personal info.

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
