"""Hand-checkable fixtures for the evaluation protocol."""

import unittest
from collections import Counter

import numpy as np

from projet.evaluation import (choose_threshold, class_metrics, detection_summary, evaluate_boxes, image_alert_metrics,
                        mask_metrics, match_boxes, review_queue, threshold_sweep)


class EvaluationFixtures(unittest.TestCase):
    def setUp(self):
        self.truth = {"class_id": 1, "box": [0, 0, 10, 10]}
        self.prediction = {**self.truth, "score": 0.8}

    def test_matching_and_duplicates(self):
        self.assertEqual(match_boxes([self.truth], [self.prediction], 0.5),
                         {"tp": 1, "fp": 0, "fn": 0})
        self.assertEqual(match_boxes([self.truth], [self.prediction, self.prediction], 0.5),
                         {"tp": 1, "fp": 1, "fn": 0})
        wrong = {**self.prediction, "class_id": 2}
        self.assertEqual(match_boxes([self.truth], [wrong], 0.5),
                         {"tp": 0, "fp": 1, "fn": 1})
        self.assertEqual(match_boxes([], [], 0.5), {"tp": 0, "fp": 0, "fn": 0})
        self.assertEqual(match_boxes([self.truth], [self.prediction], 0.9),
                         {"tp": 0, "fp": 0, "fn": 1})

    def test_detection_summary_excludes_unsupported_classes(self):
        images = {"a": {"ground_truth": [self.truth], "predictions": [self.prediction]}}
        result = detection_summary(images, 0.5, [1, 2])
        self.assertEqual(result["map50"], 1.0)
        self.assertEqual(result["map50_95"], 1.0)
        self.assertEqual(result["per_class"][1]["f1"], 1.0)
        self.assertIsNone(result["per_class"][2]["ap50"])

    def test_sweep_and_class_metrics(self):
        images = {"a": {"ground_truth": [self.truth], "predictions": [self.prediction]},
                  "b": {"ground_truth": [], "predictions": []}}
        result = threshold_sweep(images)
        self.assertLess(result[0]["threshold"], 0.8)
        self.assertGreater(result[-1]["threshold"], 0.8)
        self.assertEqual(choose_threshold(result)["threshold"], 0.8)
        self.assertEqual(class_metrics(images, 0.8, 1)["ap50"], 1.0)
        self.assertEqual(image_alert_metrics(images, 0.8), {"tp": 1, "fp": 0, "fn": 0, "tn": 1})
        self.assertIsNone(class_metrics(images, 0.8, 2)["ap50"])

    def test_mask_metrics(self):
        truth = np.array([[[[1, 1], [0, 0]]]])
        pred = np.array([[[[1, 0], [1, 0]]]])
        metric = mask_metrics(truth, pred)[0]
        self.assertEqual((metric["tp"], metric["fp"], metric["fn"]), (1, 1, 1))
        self.assertAlmostEqual(metric["iou"], 1 / 3)
        self.assertAlmostEqual(metric["dice"], 0.5)
        self.assertIsNone(mask_metrics(np.zeros_like(truth), np.zeros_like(truth))[0]["iou"])
        streamed = mask_metrics(iter([truth[0], truth[0]]), iter([pred[0], pred[0]]))[0]
        self.assertEqual((streamed["tp"], streamed["fp"], streamed["fn"]), (2, 2, 2))
        with self.assertRaises(ValueError):
            mask_metrics(iter([truth[0]]), iter([]))

    def test_sweep_matches_independent_threshold_matching(self):
        images = {
            "a": {"ground_truth": [self.truth], "predictions": [
                {**self.prediction, "score": 0.6},
                {**self.prediction, "score": 0.6},
                {**self.prediction, "score": 0.3, "class_id": 2}]},
            "b": {"ground_truth": [self.truth], "predictions": [
                {**self.prediction, "score": 0.6},
                {**self.prediction, "score": 0.9, "box": [20, 20, 30, 30]}]},
            "c": {"ground_truth": [], "predictions": []},
        }
        for row in threshold_sweep(images):
            expected = Counter()
            for image in images.values():
                expected.update(match_boxes(image["ground_truth"], image["predictions"], row["threshold"]))
            self.assertEqual({key: row[key] for key in ("tp", "fp", "fn")},
                             {key: expected[key] for key in ("tp", "fp", "fn")})

    def test_tie_break_and_empty_sweep(self):
        candidates = [{"cost_per_image": 1, "fn": 2, "threshold": 0.9},
                      {"cost_per_image": 1, "fn": 1, "threshold": 0.4},
                      {"cost_per_image": 1, "fn": 1, "threshold": 0.6}]
        self.assertEqual(choose_threshold(candidates)["threshold"], 0.6)
        images = {"a": {"ground_truth": [self.truth], "predictions": []}}
        self.assertEqual(threshold_sweep(images)[0]["cost_per_image"], 10)

    def test_report_tracks_score_floor_and_validates_inputs(self):
        images = {"a": {"ground_truth": [self.truth], "predictions": [self.prediction]}}
        report = evaluate_boxes(images, [1, 2], score_floor=0.1)
        self.assertEqual(report["score_floor"], 0.1)
        self.assertEqual(report["per_class"][1]["recall"], 1.0)
        self.assertEqual(report["image_alerts"]["tp"], 1)
        with self.assertRaises(ValueError):
            evaluate_boxes(images, [1, 2], score_floor=0.9)
        with self.assertRaises(ValueError):
            evaluate_boxes(images, [2], score_floor=0.1)

    def test_review_band_and_overflow(self):
        uncertain = {"ground_truth": [self.truth], "predictions": [{**self.prediction, "score": 0.2}]}
        upper = {"ground_truth": [], "predictions": [{**self.prediction, "score": 0.8}]}
        empty = {"ground_truth": [self.truth], "predictions": []}
        result = review_queue([uncertain, uncertain, upper, empty], 0.2, 0.8, 1)
        self.assertEqual((result["queued"], result["reviewed"], result["overflow"]), (2, 1, 1))
        self.assertEqual(result["bypassed_misses"], 1)
        self.assertEqual(result["before"], {"tp": 0, "fp": 1, "fn": 3})
        self.assertEqual(result["after"], {"tp": 1, "fp": 1, "fn": 2})
        self.assertEqual(review_queue([uncertain], 0.2, 0.8, 0)["overflow"], 1)
        windows = review_queue([uncertain] * 61, 0.2, 0.8, 9)
        self.assertEqual((windows["reviewed"], windows["overflow"]), (10, 51))
        self.assertEqual(windows["review_seconds"], 50)


if __name__ == "__main__":
    unittest.main()
