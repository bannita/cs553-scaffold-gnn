"""
Self-test for harness.py. Run it in Colab or locally with: python test_harness.py
Uses a temporary folder, so it never touches real results/ or predictions/.
"""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import harness


def test_ece_perfectly_calibrated():
    rng = np.random.default_rng(0)
    y_prob = np.full(10000, 0.7)
    y_true = (rng.random(10000) < 0.7).astype(int)
    assert harness.ece(y_true, y_prob) < 0.02


def test_ece_overconfident():
    y_prob = np.full(1000, 0.9)
    y_true = np.array([1, 0] * 500)  # only 50% positive
    assert abs(harness.ece(y_true, y_prob) - 0.4) < 1e-9


def test_ece_prob_exactly_one():
    # prob == 1.0 must fall in the last bin, not crash
    assert harness.ece([1, 1, 0], [1.0, 1.0, 1.0]) >= 0.0


def test_auc_values():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    m = harness.compute_metrics(y_true, perfect)
    assert m["roc_auc"] == 1.0 and m["pr_auc"] == 1.0
    inverted = 1 - perfect
    assert harness.compute_metrics(y_true, inverted)["roc_auc"] == 0.0


def test_single_class_gives_nan_auc():
    m = harness.compute_metrics([0, 0, 0, 0], [0.1, 0.2, 0.3, 0.4])
    assert np.isnan(m["roc_auc"]) and np.isnan(m["pr_auc"]) and not np.isnan(m["ece"])


def test_bad_inputs_raise():
    for bad in [
        lambda: harness.compute_metrics([0, 1], [0.5]),  # length mismatch
        lambda: harness.compute_metrics([0, 2], [0.5, 0.5]),  # label not 0/1
        lambda: harness.compute_metrics([0, 1], [-2.0, 3.0]),  # logits, not probabilities
        lambda: harness.compute_metrics([0, 1], [np.nan, 0.5]),  # NaN
    ]:
        try:
            bad()
        except ValueError:
            continue
        raise AssertionError("expected ValueError")


def test_bad_names_raise():
    with tempfile.TemporaryDirectory() as d:
        for kwargs in [
            dict(model="rf", dataset="hiv", split="random", seed=0),  # wrong dataset name
            dict(model="rf", dataset="bace", split="scaf", seed=0),  # wrong split name
            dict(model="r f", dataset="bace", split="random", seed=0),  # bad model name
        ]:
            try:
                harness.log_results(metrics={"roc_auc": 0.5}, results_dir=d, **kwargs)
            except ValueError:
                continue
            raise AssertionError("expected ValueError")


def _fake_fn_factory(quality):
    def fn(seed):
        rng = np.random.default_rng(seed)
        y = (rng.random(500) < 0.2).astype(int)
        p = np.clip(0.2 + quality * (y - 0.2) + rng.normal(0, 0.15, 500), 0, 1)
        return {"y_true": y, "y_prob": p, "test_idx": np.arange(500) + 1000, "y_std": rng.random(500) * 0.1}

    return fn


def test_run_seeds_logs_and_overwrites():
    with tempfile.TemporaryDirectory() as d:
        res, pred = Path(d) / "results", Path(d) / "predictions"
        for _ in range(2):  # run twice: must NOT duplicate rows
            out = harness.run_seeds(
                "rf", _fake_fn_factory(0.8), "bace", "random", results_dir=res, pred_dir=pred, verbose=False
            )
        assert len(out) == 3
        df = pd.read_csv(res / "rf.csv")
        assert list(df.columns) == harness.COLUMNS
        assert len(df) == 3 * 3  # 3 seeds x 3 metrics
        assert set(df["metric"]) == set(harness.METRICS)
        # predictions saved and loadable, with extras
        f = pred / "rf__bace__random__seed1.npz"
        assert f.exists()
        z = np.load(f)
        assert set(z.files) == {"y_true", "y_prob", "test_idx", "y_std"}
        assert z["test_idx"][0] == 1000


def test_summary_and_gap():
    with tempfile.TemporaryDirectory() as d:
        res, pred = Path(d) / "results", Path(d) / "predictions"
        harness.run_seeds("rf", _fake_fn_factory(0.9), "bace", "random", results_dir=res, pred_dir=pred, verbose=False)
        harness.run_seeds("rf", _fake_fn_factory(0.4), "bace", "scaffold", results_dir=res, pred_dir=pred, verbose=False)
        harness.run_seeds("ffn", _fake_fn_factory(0.7), "bace", "random", results_dir=res, pred_dir=pred, verbose=False)
        allr = harness.load_results(res)
        assert set(allr["model"]) == {"rf", "ffn"}  # two models, two files, combined
        s = harness.summarize(res)
        assert (s["n_seeds"] == 3).all()
        g = harness.scaffold_gap("roc_auc", res)
        rf_gap = g[(g["model"] == "rf") & (g["dataset"] == "bace")]["gap"].iloc[0]
        assert rf_gap > 0  # random split is easier than scaffold in this fake setup


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
