"""Run with: python -m unittest discover -s tests -p test_evaluation.py -v"""
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.eval.compare import load_annotations, load_predictions, run_comparison
from src.eval.metrics import choose_threshold, evaluate_scores, top_k_metrics, typology_metrics


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        parent = ROOT / "outputs" / "tests"
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_metrics_and_frozen_threshold(self):
        y, s = [0, 1, 0, 1], [0.1, 0.8, 0.2, 0.7]
        threshold = choose_threshold(y, s)
        self.assertEqual(threshold, 0.7)
        m = evaluate_scores(y, s, threshold, [1, 2, 3, 4], ks=(1, 2, 100))
        self.assertEqual(m["auprc"], 1)
        self.assertEqual(m["precision@1"], 1)
        self.assertEqual(m["recall@1"], 0.5)
        self.assertEqual(m["precision@100"], 0.5)  # denominator capped at four rows
        self.assertEqual(m["tp"], 2)
        test = evaluate_scores(y, [0.6, 0.65, 0.3, 0.4], threshold, [1, 2, 3, 4], ks=(2,))
        self.assertEqual(test["alerts"], 0)
        self.assertEqual(test["threshold"], threshold)
        no_positive = evaluate_scores([0, 0], [0.1, 0.2], 0.5, [1, 2], ks=(1,))
        self.assertIsNone(no_positive["roc_auc"])
        self.assertIsNone(no_positive["auprc"])
        self.assertIsNone(no_positive["recall@1"])
        with self.assertRaises(ValueError):
            choose_threshold([0, 0], [0.1, 0.2])

    def test_top_k_ties_are_independent_of_row_order(self):
        y = np.array([0, 1, 1, 0])
        s = np.array([0.9, 0.9, 0.2, 0.1])
        ids = np.array([20, 10, 30, 40])
        expected = top_k_metrics(y, s, ids, ks=(1, 2))
        self.assertEqual(expected[0]["caught"], 1)
        self.assertEqual(expected[0]["transactions_tied_at_cutoff"], 2)
        perm = [3, 2, 0, 1]
        self.assertEqual(expected, top_k_metrics(y[perm], s[perm], ids[perm], ks=(1, 2)))

    def test_prediction_integrity_and_alignment(self):
        labels = pd.DataFrame({"edge_id": [1, 2, 3], "split": ["val", "val", "test"], "label": [0, 1, 1]})
        good = pd.DataFrame({"edge_id": [3, 1, 2], "split": ["test", "val", "val"], "score": [0.8, 0.2, 0.9]})
        path = self.root / "predictions.parquet"
        good.to_parquet(path)
        aligned = load_predictions(path, labels)
        self.assertEqual(aligned.score.tolist(), [0.2, 0.9, 0.8])
        bads = [good.iloc[:-1], pd.concat([good, good.iloc[[0]]]),
                good.assign(split="val"), good.assign(score=np.nan),
                good.assign(score=1.1), good.assign(edge_id=good.edge_id + 0.5),
                good.assign(edge_id=[3, 1, 99])]
        for bad in bads:
            with self.subTest(data=bad.to_dict()):
                bad.to_parquet(path)
                with self.assertRaises(ValueError):
                    load_predictions(path, labels)

    def test_typology_uses_global_queue_and_keeps_unknown_separate(self):
        frame = pd.DataFrame({"edge_id": [1, 2, 3, 4, 5, 6], "label": [0, 1, 0, 1, 1, 1],
                              "score": [0.99, 0.95, 0.9, 0.8, 0.7, 0.6]})
        annotations = pd.DataFrame({"edge_id": [2, 4, 5], "typology": ["CYCLE", "CYCLE", "UNKNOWN_UNLISTED"],
                                    "pattern_id": ["a", "a", None],
                                    "annotation_status": ["ANNOTATED", "ANNOTATED", "UNKNOWN_UNLISTED"]})
        types, groups = typology_metrics(frame, annotations, 0.75, ks=(2,))
        types = {r["typology"]: r for r in types}
        self.assertEqual(types["CYCLE"]["recall@2"], 0.5)
        self.assertEqual(types["CYCLE"]["schemes"], 1)
        self.assertEqual(types["CYCLE"]["schemes_detected"], 1)
        self.assertEqual(types["CYCLE"]["recall"], 1)
        self.assertEqual(types["MISSING_ANNOTATION"]["positives"], 1)
        self.assertEqual(types["UNKNOWN_UNLISTED"]["positives"], 1)
        self.assertEqual(sum(r["detected"] for r in groups), 2)
        self.assertIsNone(types["UNKNOWN_UNLISTED"]["schemes"])

    def test_annotation_label_mismatch_rejected(self):
        labels = pd.DataFrame({"edge_id": [1], "split": ["test"], "label": [0]})
        path = self.root / "patterns.csv"
        pd.DataFrame({"source_row": [1], "split": ["test"], "is_laundering": [1],
                      "pattern_id": ["p"], "typology": ["CYCLE"], "annotation_status": ["ANNOTATED"]}).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "normal"):
            load_annotations(path, labels)

    def test_full_export_reference_audit_and_test_independence(self):
        data = self.root / "data"
        data.mkdir()
        for split, offset in (("validation", 0), ("test", 10)):
            pd.DataFrame({"source_row": np.arange(4) + offset,
                          "timestamp": pd.date_range(f"2022-09-{1 if split == 'validation' else 2:02d}", periods=4, freq="min"),
                          "is_laundering": [0, 1, 0, 1]}).to_csv(data / f"{split}.csv", index=False)
        predictions = {}
        for name, scores in (("lightgbm", [0.1, 0.8, 0.2, 0.7]), ("gnn", [0.9, 0.8, 0.2, 0.7])):
            path = self.root / f"{name}.parquet"
            pd.DataFrame({"edge_id": [0, 1, 2, 3, 10, 11, 12, 13], "split": ["val"] * 4 + ["test"] * 4,
                          "score": scores + scores}).sample(frac=1, random_state=2).to_parquet(path, index=False)
            predictions[name] = path
        reference = self.root / "metrics.json"
        reference.write_text(json.dumps({"val": {"threshold": 0.7, "f1": 1.0},
                                         "test": {"threshold": 0.7, "f1": 1.0}}))
        a = run_comparison(data, predictions, self.root / "a", references={"lightgbm": reference},
                           ks=(1, 2), make_plots=False)
        self.assertTrue(a["reference_checks"]["all_match"])
        self.assertEqual(a["reference_checks"]["checked"], 4)
        table = pd.read_csv(self.root / "a" / "model_comparison.csv")
        self.assertEqual(len(table), 4)
        self.assertTrue((self.root / "a" / "manifest.json").exists())
        test = pd.read_csv(data / "test.csv")
        test["is_laundering"] = 1 - test.is_laundering
        test.to_csv(data / "test.csv", index=False)
        b = run_comparison(data, predictions, self.root / "b", ks=(1, 2), make_plots=False)
        for model in predictions:
            self.assertEqual(a["thresholds"][model]["threshold"], b["thresholds"][model]["threshold"])
            self.assertEqual(a["models"][model]["val"], b["models"][model]["val"])
            self.assertNotEqual(a["models"][model]["test"]["f1"], b["models"][model]["test"]["f1"])
        with self.assertRaisesRegex(ValueError, "Saved metrics do not match"):
            run_comparison(data, predictions, self.root / "bad", references={"lightgbm": reference},
                           ks=(1, 2), make_plots=False)
        self.assertEqual(json.loads((self.root / "bad" / "status.json").read_text())["state"], "failed")
        self.assertTrue((self.root / "bad" / "reference_metric_checks.csv").exists())


if __name__ == "__main__":
    unittest.main()
