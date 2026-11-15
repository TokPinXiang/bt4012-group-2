"""Baseline contract, leakage and saved-model checks on tiny synthetic CSVs.

Run: python -m unittest discover -s tests -p test_gbt.py -v
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.gbt import (
    TrainConfig, baseline_metrics, check_splits, choose_threshold,
    feature_columns, fit_preprocessor, load_model, load_split, predict,
    train_baseline, transform,
)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "outputs" / "tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_data(self):
        data = self.root / "data"
        data.mkdir()
        rng = np.random.default_rng(13)
        for offset, (name, size) in enumerate((("train", 1000), ("validation", 400), ("test", 400))):
            amount = rng.random(size)
            frame = pd.DataFrame({
                "source_row": np.arange(offset * 10000, offset * 10000 + size),
                "timestamp": pd.date_range(f"2022-09-{1 + offset:02d}", periods=size, freq="min"),
                "amount_paid": amount, "src_hist_out_count": rng.integers(0, 10, size),
                "payment_format": np.where(np.arange(size) % 2, "ACH", "WIRE"),
                "is_laundering": (amount > 0.85).astype(int),
                "src_account_id": "bank:account", "pattern_id": "DO_NOT_USE",
                "typology": "DO_NOT_USE", "tx_id": "DO_NOT_USE",
            })
            if offset:
                frame.loc[0, "payment_format"] = "UNSEEN"
            frame.to_csv(data / f"{name}.csv", index=False)
        return data

    def test_feature_exclusions_and_train_only_categories(self):
        columns = ["amount_paid", "src_hist_out_count", "payment_format", "source_row",
                   "timestamp", "is_laundering", "src_account_id", "pattern_id", "typology"]
        self.assertEqual(feature_columns(columns, "row"), ["amount_paid", "payment_format"])
        train = pd.DataFrame({"amount_paid": [1., 2.], "payment_format": ["ACH", "WIRE"]})
        meta = fit_preprocessor(train, feature_columns(columns, "row"))
        future = pd.DataFrame({"amount_paid": [3., np.nan], "payment_format": ["NEW", "ACH"]})
        x = transform(future, meta)
        self.assertTrue(pd.isna(x.payment_format.iloc[0]))
        self.assertEqual(meta["categories"]["payment_format"], ["ACH", "WIRE"])
        self.assertEqual(future.payment_format.iloc[0], "NEW")

    def test_invalid_labels_and_duplicate_ids_rejected(self):
        data = self.make_data()
        path = data / "train.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "is_laundering"] = 256
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "binary classes"):
            load_split(data, "train")
        frame.loc[0, "is_laundering"] = 0
        frame.loc[0, "source_row"] = frame.loc[1, "source_row"]
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            load_split(data, "train")

    def test_split_overlap_rejected(self):
        a = pd.DataFrame({"source_row": [1], "timestamp": pd.to_datetime(["2022-09-01"])})
        b = pd.DataFrame({"source_row": [2], "timestamp": pd.to_datetime(["2022-09-01"])})
        with self.assertRaisesRegex(ValueError, "time"):
            check_splits(a, b)
        b["timestamp"] = pd.to_datetime(["2022-09-02"])
        b["source_row"] = 1
        with self.assertRaisesRegex(ValueError, "IDs"):
            check_splits(a, b)

    def test_metrics_use_validation_threshold(self):
        y, s = np.array([0, 1, 0, 1]), np.array([0.1, 0.8, 0.2, 0.7])
        threshold = choose_threshold(y, s)
        self.assertAlmostEqual(threshold, 0.7)
        m = baseline_metrics(y, s, threshold)
        self.assertEqual(m["auprc"], 1.0)
        self.assertEqual(m["f1"], 1.0)
        self.assertEqual(m["tp"], 2)
        self.assertEqual(m["recall@500"], 1.0)
        # Applying the same threshold on different scores does not retune it.
        m = baseline_metrics(y, s / 2, threshold)
        self.assertEqual(m["threshold"], threshold)
        self.assertEqual(m["alerts"], 0)

    def test_roundtrip_shap_and_test_labels_cannot_change_selection(self):
        data = self.make_data()
        cfg = TrainConfig(trials=3, max_rounds=15, patience=3, threads=1, shap_rows=20, smoke=True)
        run_a, run_b = self.root / "a", self.root / "b"
        first = train_baseline(data, run_a, cfg)
        test = pd.read_csv(data / "test.csv")
        test["is_laundering"] = 1 - test.is_laundering
        test.to_csv(data / "test.csv", index=False)
        second = train_baseline(data, run_b, cfg)
        self.assertEqual(first["val"], second["val"])
        self.assertEqual(first["summary"]["selected_trial"], second["summary"]["selected_trial"])
        pd.testing.assert_frame_equal(pd.read_parquet(run_a / "predictions.parquet"),
                                      pd.read_parquet(run_b / "predictions.parquet"))
        self.assertNotEqual(first["test"]["auprc"], second["test"]["auprc"])
        model, meta = load_model(run_a)
        scores = predict(model, test, meta, threads=1)
        saved = pd.read_parquet(run_a / "predictions.parquet").query("split == 'test'")
        np.testing.assert_allclose(scores, saved.score, rtol=1e-10, atol=1e-12)
        self.assertEqual(list(saved.columns), ["edge_id", "split", "score"])
        shap = pd.read_parquet(run_a / "shap_values.parquet")
        inputs = pd.read_parquet(run_a / "shap_inputs.parquet")
        raw = model.predict(transform(inputs, meta), raw_score=True, num_threads=1)
        np.testing.assert_allclose(shap[meta["features"]].sum(axis=1) + shap.base_value, raw,
                                   rtol=1e-5, atol=1e-6)
        # CLI inference works without labels, and smoke runs cannot be published.
        test.drop(columns="is_laundering").to_csv(self.root / "unlabelled.csv", index=False)
        cli = [sys.executable, str(ROOT / "scripts" / "03_train_gbt.py")]
        subprocess.run([*cli, "--predict-run", str(run_a), "--input", str(self.root / "unlabelled.csv"),
                        "--output", str(self.root / "scores.parquet"), "--threads", "1"],
                       check=True, capture_output=True, text=True)
        np.testing.assert_allclose(pd.read_parquet(self.root / "scores.parquet").score, saved.score)
        rejected = subprocess.run([*cli, "--publish-run", str(run_a)], capture_output=True, text=True)
        self.assertNotEqual(rejected.returncode, 0)
        self.assertIn("non-smoke", rejected.stderr)
        with self.assertRaises(FileExistsError):
            train_baseline(data, run_a, cfg)


if __name__ == "__main__":
    unittest.main()
