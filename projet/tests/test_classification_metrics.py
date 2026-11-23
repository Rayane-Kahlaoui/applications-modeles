"""Check binary metrics and timing using deterministic fixtures."""

import unittest

from projet.classification_metrics import binary_metrics, threshold_for_recall
from projet.timing import benchmark_inference


class ClassificationMetricsTest(unittest.TestCase):
    def test_recall_target_threshold_with_ties(self):
        labels = [1, 1, 1, 0]
        scores = [0.2, 0.5, 0.5, 0.8]
        self.assertEqual(threshold_for_recall(labels, scores, 2 / 3), 0.5)
        self.assertEqual(threshold_for_recall(labels, scores, 1), 0.2)
        with self.assertRaises(ValueError):
            threshold_for_recall([0], [0.2], 0.95)

    def test_counts_and_undefined_curves(self):
        result = binary_metrics([1, 1, 0, 0], [0.9, 0.1, 0.8, 0.2], 0.5)
        self.assertEqual((result["tp"], result["fp"], result["fn"], result["tn"]), (1, 1, 1, 1))
        self.assertEqual(result["recall"], 0.5)
        self.assertEqual(result["specificity"], 0.5)
        self.assertEqual(result["f1"], 0.5)
        self.assertIsNone(binary_metrics([0, 0], [0.1, 0.2], 0.5)["roc_auc"])
        self.assertIsNone(binary_metrics([0, 0], [0.1, 0.2], 0.5)["average_precision"])

    def test_timing_counts_warmup_separately(self):
        calls = []
        result = benchmark_inference(calls.append, [1, 2, 3], warmup=2)
        self.assertEqual(calls, [1, 2, 1, 2, 3])
        self.assertEqual(len(result["per_image_seconds"]), 3)
        self.assertGreaterEqual(result["p95_seconds"], result["median_seconds"])
