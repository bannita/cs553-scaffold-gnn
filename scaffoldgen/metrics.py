"""
Shared metrics: ROC-AUC, PR-AUC, calibration (ECE), reliability curves,
and per-familiarity-bin breakdowns.
Owner: Ani

Every model reports through these functions so the numbers are comparable.
Convention everywhere: y_true is 0/1 int, y_prob is P(active) in [0, 1].
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


# --------------------------------------------------------------------------
# discrimination
# --------------------------------------------------------------------------
def roc_auc(y_true, y_prob) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")          # happens on tiny scaffold-split bins
    return float(roc_auc_score(y_true, y_prob))


def pr_auc(y_true, y_prob) -> float:
    """Average precision. The metric that matters at 3.5% positives."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_prob))


def enrichment_factor(y_true, y_prob, frac: float = 0.01) -> float:
    """How many more actives you find in the top `frac` of a ranked library
    than by picking at random. This is the number a screening chemist cares
    about. EF=10 at 1% means a tenfold enrichment.
    """
    y_true = np.asarray(y_true)
    n = len(y_true)
    k = max(1, int(round(frac * n)))
    base = y_true.mean()
    if base == 0:
        return float("nan")
    top = np.argsort(-np.asarray(y_prob))[:k]
    return float(y_true[top].mean() / base)


# --------------------------------------------------------------------------
# calibration
# --------------------------------------------------------------------------
def reliability_curve(y_true, y_prob, n_bins: int = 10, strategy: str = "uniform"):
    """Returns per-bin (count, mean_confidence, observed_frequency).

    strategy='uniform'  : equal-width probability bins (standard ECE)
    strategy='quantile' : equal-count bins (better with skewed probabilities,
                          which is what you get on MolHIV)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1], right=False), 0, len(edges) - 2)

    rows = []
    for b in range(len(edges) - 1):
        m = idx == b
        cnt = int(m.sum())
        if cnt == 0:
            rows.append((0, float("nan"), float("nan")))
        else:
            rows.append((cnt, float(y_prob[m].mean()), float(y_true[m].mean())))
    return rows, edges


def expected_calibration_error(y_true, y_prob, n_bins: int = 10,
                               strategy: str = "uniform") -> float:
    """Weighted mean |confidence - accuracy| across bins.

    Note: this is the binary-probability form -- we compare predicted
    P(active) against the observed positive rate in the bin, NOT
    top-label confidence. That is the right version for a screening task.
    """
    rows, _ = reliability_curve(y_true, y_prob, n_bins, strategy)
    n = len(y_true)
    ece = 0.0
    for cnt, conf, obs in rows:
        if cnt == 0:
            continue
        ece += (cnt / n) * abs(conf - obs)
    return float(ece)


def maximum_calibration_error(y_true, y_prob, n_bins: int = 10,
                              strategy: str = "uniform") -> float:
    rows, _ = reliability_curve(y_true, y_prob, n_bins, strategy)
    gaps = [abs(c - o) for cnt, c, o in rows if cnt > 0]
    return float(max(gaps)) if gaps else float("nan")


def brier(y_true, y_prob) -> float:
    return float(brier_score_loss(np.asarray(y_true), np.asarray(y_prob)))


def negative_log_likelihood(y_true, y_prob, eps: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.clip(np.asarray(y_prob, dtype=float), eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


# --------------------------------------------------------------------------
# uncertainty quality
# --------------------------------------------------------------------------
def selective_auc(y_true, y_prob, uncertainty, keep: float = 0.5) -> float:
    """ROC-AUC on the `keep` fraction of molecules the model is most certain
    about. If the uncertainty estimate is any good, this beats the full-set
    AUC -- that is the practical test of "can a chemist trust it".
    """
    order = np.argsort(np.asarray(uncertainty))
    k = max(2, int(round(keep * len(order))))
    sel = order[:k]
    return roc_auc(np.asarray(y_true)[sel], np.asarray(y_prob)[sel])


# --------------------------------------------------------------------------
# one-call summary
# --------------------------------------------------------------------------
def summarize(y_true, y_prob, uncertainty=None, n_bins: int = 10) -> dict[str, float]:
    """The standard row of numbers for one (model, dataset, split, seed)."""
    out = {
        "roc_auc": roc_auc(y_true, y_prob),
        "pr_auc": pr_auc(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "ece_quantile": expected_calibration_error(y_true, y_prob, n_bins, "quantile"),
        "mce": maximum_calibration_error(y_true, y_prob, n_bins),
        "brier": brier(y_true, y_prob),
        "nll": negative_log_likelihood(y_true, y_prob),
        "ef_1pct": enrichment_factor(y_true, y_prob, 0.01),
        "n": float(len(y_true)),
        "pos_rate": float(np.mean(y_true)),
    }
    if uncertainty is not None:
        out["selective_auc_50"] = selective_auc(y_true, y_prob, uncertainty, 0.5)
        out["mean_uncertainty"] = float(np.mean(uncertainty))
    return out


def summarize_by_bin(y_true, y_prob, bin_idx, labels=("low", "medium", "high")) -> dict:
    """Familiarity-binned metrics: {'low': {...}, 'medium': {...}, ...}.

    Bins on a scaffold split can be small and sometimes all-negative, in
    which case AUC is nan -- report the count alongside and never average
    over nans silently.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_idx = np.asarray(bin_idx)
    out = {}
    for b, name in enumerate(labels):
        m = bin_idx == b
        if m.sum() == 0:
            out[name] = {"n": 0.0}
            continue
        out[name] = {
            "n": float(m.sum()),
            "pos_rate": float(y_true[m].mean()),
            "roc_auc": roc_auc(y_true[m], y_prob[m]),
            "pr_auc": pr_auc(y_true[m], y_prob[m]),
            "ece": expected_calibration_error(y_true[m], y_prob[m]),
        }
    return out
