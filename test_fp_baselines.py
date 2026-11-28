"""
Self-test for fp_baselines.py. Run with:   python test_fp_baselines.py

Builds a tiny fake copy of the shared Drive data layout in a temporary folder (mapping/, fingerprints/,
splits/), so it needs no real data and never touches your real results.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import fp_baselines as fb
import harness


def make_fake_data_dir(root: Path, n=700, n_bits=2048, seed=0, short="bace"):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    X = (rng.random((n, n_bits)) < 0.02).astype(np.uint8)
    X[:, :6] = ((rng.random((n, 6)) < 0.05) | ((y[:, None] == 1) & (rng.random((n, 6)) < 0.6))).astype(np.uint8)
    (root / "mapping" / short).mkdir(parents=True)
    (root / "fingerprints").mkdir(parents=True)
    (root / "splits").mkdir(parents=True)
    pd.DataFrame({"Class": y, "smiles": ["C"] * n, "mol_id": range(n)}).to_csv(
        root / "mapping" / short / "mol.csv.gz", index=False
    )
    np.save(root / "fingerprints" / f"{short}_{fb.FP_TAG}.npy", X)
    for split in ("random", "scaffold"):
        for k in range(3):  # a DIFFERENT split per seed, like Ana B.'s files
            perm = np.random.default_rng(100 * k + (split == "scaffold")).permutation(n)
            rec = {"train": sorted(perm[:560].tolist()), "valid": sorted(perm[560:630].tolist()), "test": sorted(perm[630:].tolist())}
            (root / "splits" / f"{short}_{split}_seed{k}.json").write_text(json.dumps(rec))
    return X, y


def _expect_error(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_loaders_and_check_all():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        X, y = make_fake_data_dir(d)
        X2, y2 = fb.load_dataset("bace", d)
        assert X2.shape == X.shape and np.array_equal(y2, y)
        sp = fb.load_split("bace", "scaffold", 1, d, n=len(y))
        assert set(sp) == {"train", "valid", "test"} and len(sp["test"]) == 70
        table = fb.check_all(d, datasets=("bace",))
        assert len(table) == 6 and (table["n_test"] == 70).all()
        # different seeds give different splits
        a = fb.load_split("bace", "random", 0, d)["test"]
        b = fb.load_split("bace", "random", 1, d)["test"]
        assert not np.array_equal(a, b)


def test_bad_data_is_rejected():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        X, y = make_fake_data_dir(d)
        _expect_error(lambda: fb.load_dataset("hiv", d))  # wrong dataset name
        # leak: test molecule also in train
        p = d / "splits" / "bace_random_seed0.json"
        rec = json.loads(p.read_text())
        rec["test"] = rec["test"] + rec["train"][:1]
        p.write_text(json.dumps(rec))
        _expect_error(lambda: fb.load_split("bace", "random", 0, d, n=len(y)))
        # index out of range
        rec = {"train": [0, 1], "valid": [2], "test": [10**6]}
        p.write_text(json.dumps(rec))
        _expect_error(lambda: fb.load_split("bace", "random", 0, d, n=len(y)))
        # missing file
        _expect_error(lambda: fb.load_split("bace", "random", 9, d), FileNotFoundError)
        # fingerprints and labels disagree
        np.save(d / "fingerprints" / f"bace_{fb.FP_TAG}.npy", X[:10])
        _expect_error(lambda: fb.load_dataset("bace", d))


def test_run_rf_end_to_end():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        X, y = make_fake_data_dir(d / "data")
        res, pred = d / "results", d / "predictions"
        small = fb.RF_CONFIGS[:2]
        chosen = fb.run_rf("bace", d / "data", configs=small, results_dir=res, pred_dir=pred, verbose=False)
        assert set(chosen) == {"random", "scaffold"}
        df = pd.read_csv(res / "rf.csv")
        assert list(df.columns) == harness.COLUMNS
        assert len(df) == 2 * 3 * 3  # 2 splits x 3 seeds x 3 metrics
        auc = df[df["metric"] == "roc_auc"]["value"]
        assert (auc > 0.65).all(), f"RF should find the planted signal, got {auc.tolist()}"
        # seed k used split file k: saved test rows equal that file's test rows
        for k in range(3):
            z = np.load(pred / f"rf__bace__scaffold__seed{k}.npz")
            want = fb.load_split("bace", "scaffold", k, d / "data")["test"]
            assert np.array_equal(z["test_idx"], want)
            assert np.array_equal(z["y_true"], y[want])
        assert (res / "rf_configs.json").exists()


def test_tuning_never_touches_test_set():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        X, y = make_fake_data_dir(d)
        sp = fb.load_split("bace", "random", 0, d)
        best_a, _ = fb.select_config(X, y, sp, fb.RF_CONFIGS[:3])
        y2 = y.copy()
        y2[sp["test"]] = 1 - y2[sp["test"]]  # scramble the TEST labels only
        best_b, _ = fb.select_config(X, y2, sp, fb.RF_CONFIGS[:3])
        assert best_a == best_b


# ---------------- FFN: bookkeeping tests with a FAKE network (no torch needed) ----------------
class _FakeNet:
    """Stands in for a torch model: a logistic regression fit on a bootstrap sample chosen by the seed."""

    def __init__(self, clf):
        self.clf = clf


def _install_fake_ffn(monkey: list):
    from sklearn.linear_model import LogisticRegression

    def fake_train(X_tr, y_tr, X_va, y_va, config, seed, device=None, **kw):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X_tr), len(X_tr), replace=True)
        return _FakeNet(LogisticRegression(max_iter=200).fit(X_tr[idx].astype(float), y_tr[idx]))

    def fake_predict(model, X, **kw):
        return model.clf.predict_proba(X.astype(float))[:, 1]

    def fake_mc(model, X, n_samples=20, seed=0, **kw):
        p = fake_predict(model, X)
        return np.clip(p + 0.01, 0, 1), np.full(len(p), 0.05)

    for name, fn in (("train_ffn", fake_train), ("predict_proba", fake_predict), ("predict_mc_dropout", fake_mc)):
        monkey.append((name, getattr(fb, name)))
        setattr(fb, name, fn)


def _remove_fake_ffn(monkey: list):
    for name, original in monkey:
        setattr(fb, name, original)


def test_run_ffn_bookkeeping_with_fake_network():
    monkey = []
    _install_fake_ffn(monkey)
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            X, y = make_fake_data_dir(d / "data")
            res, pred = d / "results", d / "predictions"
            chosen = fb.run_ffn("bace", d / "data", configs=fb.FFN_CONFIGS[:2], results_dir=res, pred_dir=pred, verbose=False)
            assert set(chosen) == {"random", "scaffold"}
            # one row per (split, seed, metric) for ffn and ffn_mcd; the ensemble is logged once per split, as seed 0
            ffn, mcd, ens = (pd.read_csv(res / f"{m}.csv") for m in ("ffn", "ffn_mcd", "ffn_ens"))
            assert len(ffn) == 2 * 3 * 3 and len(mcd) == 2 * 3 * 3
            assert len(ens) == 2 * 1 * 3 and set(ens["seed"]) == {0}
            for k in range(3):  # seed k used split file k
                z = np.load(pred / f"ffn__bace__scaffold__seed{k}.npz")
                assert np.array_equal(z["test_idx"], fb.load_split("bace", "scaffold", k, d / "data")["test"])
            # MC-dropout predictions carry their spread
            z = np.load(pred / "ffn_mcd__bace__random__seed1.npz")
            assert "y_std" in z.files and np.allclose(z["y_std"], 0.05)
            # ensemble: tested on the FIRST seed's split file, has member spread and member count
            z = np.load(pred / "ffn_ens__bace__scaffold__seed0.npz")
            assert np.array_equal(z["test_idx"], fb.load_split("bace", "scaffold", 0, d / "data")["test"])
            assert "y_std" in z.files and int(z["n_members"]) == 3 and z["y_std"].max() > 0  # members differ
            assert (res / "ffn_configs.json").exists()
            # everything flows back through the harness
            assert {"ffn", "ffn_mcd", "ffn_ens"} <= set(harness.load_results(res)["model"])
    finally:
        _remove_fake_ffn(monkey)


def test_run_all_runs_everything_and_skips_finished_work():
    monkey = []
    _install_fake_ffn(monkey)
    calls = {"n": 0}
    fake_train = fb.train_ffn

    def counting_train(*a, **kw):
        calls["n"] += 1
        return fake_train(*a, **kw)

    fb.train_ffn = counting_train
    try:
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            make_fake_data_dir(d / "data")
            res, pred = d / "results", d / "predictions"
            kw = dict(results_dir=res, pred_dir=pred, verbose=False)
            fb.run_all(datasets=("bace",), data_dir=d / "data",
                       rf_kwargs=dict(configs=fb.RF_CONFIGS[:1], **kw),
                       ffn_kwargs=dict(configs=fb.FFN_CONFIGS[:1], **kw))
            assert {p.stem for p in res.glob("*.csv")} == {"rf", "ffn", "ffn_mcd", "ffn_ens"}
            first_calls = calls["n"]
            assert first_calls > 0
            before = {p.name: p.read_bytes() for p in res.glob("*.csv")}
            # second call: everything is complete, so nothing is retrained and no file changes
            fb.run_all(datasets=("bace",), data_dir=d / "data",
                       rf_kwargs=dict(configs=fb.RF_CONFIGS[:1], **kw),
                       ffn_kwargs=dict(configs=fb.FFN_CONFIGS[:1], **kw))
            assert calls["n"] == first_calls
            assert before == {p.name: p.read_bytes() for p in res.glob("*.csv")}
            # force=True redoes the work
            fb.run_all(models=("ffn",), datasets=("bace",), data_dir=d / "data", force=True,
                       ffn_kwargs=dict(configs=fb.FFN_CONFIGS[:1], **kw))
            assert calls["n"] > first_calls
            # an unknown model name is rejected
            try:
                fb.run_all(models=("svm",), datasets=("bace",), data_dir=d / "data")
            except ValueError:
                pass
            else:
                raise AssertionError("expected ValueError")
    finally:
        fb.train_ffn = fake_train
        _remove_fake_ffn(monkey)


# ---------------- FFN: real torch tests (run these in Colab; skipped where torch is missing) ----------------
def test_torch_ffn_learns_and_mc_dropout_works():
    try:
        import torch  # noqa: F401
    except ImportError:
        return "skip"
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        X, y = make_fake_data_dir(d)
        sp = fb.load_split("bace", "random", 0, d)
        cfg = dict(hidden=(64, 32), dropout=0.3, lr=1e-3)
        model = fb.train_ffn(X[sp["train"]], y[sp["train"]], X[sp["valid"]], y[sp["valid"]], cfg, seed=0, device="cpu")
        p = fb.predict_proba(model, X[sp["test"]])
        assert p.shape == (len(sp["test"]),) and p.min() >= 0 and p.max() <= 1
        assert harness.compute_metrics(y[sp["test"]], p)["roc_auc"] > 0.65  # finds the planted signal
        assert np.array_equal(p, fb.predict_proba(model, X[sp["test"]]))  # deterministic with dropout off
        mean, std = fb.predict_mc_dropout(model, X[sp["test"]], n_samples=20, seed=0)
        assert mean.shape == p.shape and std.mean() > 0  # dropout on -> predictions vary between passes
        assert not model.training  # dropout switched back off afterwards
        assert np.array_equal(p, fb.predict_proba(model, X[sp["test"]]))
        # same seed -> same network
        model2 = fb.train_ffn(X[sp["train"]], y[sp["train"]], X[sp["valid"]], y[sp["valid"]], cfg, seed=0, device="cpu")
        assert np.allclose(p, fb.predict_proba(model2, X[sp["test"]]), atol=1e-5)
        # different seed -> different network
        model3 = fb.train_ffn(X[sp["train"]], y[sp["train"]], X[sp["valid"]], y[sp["valid"]], cfg, seed=1, device="cpu")
        assert not np.allclose(p, fb.predict_proba(model3, X[sp["test"]]), atol=1e-5)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    skipped = 0
    for t in tests:
        if t() == "skip":
            skipped += 1
            print(f"SKIP  {t.__name__} (torch not installed here; run this file in Colab)")
        else:
            print(f"PASS  {t.__name__}")
    print(f"\nFinished {len(tests)} tests, {skipped} skipped.")
