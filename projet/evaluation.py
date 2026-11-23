"""Evaluation primitives for scored box predictions and review queues."""

import math
from collections import Counter
from itertools import zip_longest

import numpy as np


def iou(a, b):
    """Intersection over union for boxes in x_min, y_min, x_max, y_max order."""
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def match_boxes(ground_truth, predictions, threshold, iou_threshold=0.5):
    """Match predictions to the highest-IoU unmatched box of the same class."""
    matched = set()
    counts = Counter()
    for prediction in sorted(predictions, key=lambda item: -item["score"]):
        if prediction["score"] < threshold:
            continue
        candidates = [
            (iou(prediction["box"], truth["box"]), index)
            for index, truth in enumerate(ground_truth)
            if index not in matched and truth["class_id"] == prediction["class_id"]
        ]
        best = max(candidates, default=(0.0, -1), key=lambda item: item[0])
        if best[0] >= iou_threshold:
            matched.add(best[1])
            counts["tp"] += 1
        else:
            counts["fp"] += 1
    counts["fn"] = len(ground_truth) - len(matched)
    return {key: counts[key] for key in ("tp", "fp", "fn")}


def threshold_sweep(images, miss_cost=10.0, false_alert_cost=1.0):
    """Compute box counts and cost for a mapping of image IDs to annotations."""
    if not images:
        raise ValueError("At least one image is required")
    if not math.isfinite(miss_cost) or not math.isfinite(false_alert_cost) or min(miss_cost, false_alert_cost) < 0:
        raise ValueError("Costs must be finite and nonnegative")
    predictions = [(prediction["score"], image_id, prediction)
                   for image_id, image in images.items()
                   for prediction in image["predictions"]]
    if any(not math.isfinite(score) for score, _, _ in predictions):
        raise ValueError("Prediction scores must be finite")
    # Stable ordering preserves the per-image order of equal-score predictions.
    predictions.sort(key=lambda item: -item[0])
    matched = {image_id: set() for image_id in images}
    counts = Counter(fn=sum(len(image["ground_truth"]) for image in images.values()))

    def result(threshold):
        return {"threshold": threshold, "tp": counts["tp"], "fp": counts["fp"],
                "fn": counts["fn"],
                "cost_per_image": (miss_cost * counts["fn"] + false_alert_cost * counts["fp"]) / len(images)}

    if not predictions:
        return [result(0.0)]
    results = [result(math.nextafter(predictions[0][0], math.inf))]
    for index, (score, image_id, prediction) in enumerate(predictions):
        truths = images[image_id]["ground_truth"]
        candidates = [(iou(prediction["box"], truth["box"]), truth_index)
                      for truth_index, truth in enumerate(truths)
                      if truth_index not in matched[image_id]
                      and truth["class_id"] == prediction["class_id"]]
        best = max(candidates, default=(0.0, -1), key=lambda item: item[0])
        if best[0] >= 0.5:
            matched[image_id].add(best[1])
            counts["tp"] += 1
            counts["fn"] -= 1
        else:
            counts["fp"] += 1
        if index + 1 == len(predictions) or predictions[index + 1][0] != score:
            results.append(result(score))
    results.append(result(math.nextafter(predictions[-1][0], -math.inf)))
    return list(reversed(results))


def choose_threshold(results):
    """Choose minimum cost, then fewest misses, then the highest threshold."""
    if not results:
        raise ValueError("Threshold sweep is empty")
    return min(results, key=lambda row: (row["cost_per_image"], row["fn"], -row["threshold"]))


def class_metrics(images, threshold, class_id, iou_threshold=0.5):
    """Return per-class counts, precision, recall and 101-point interpolated AP."""
    truths = {
        image_id: [box for box in image["ground_truth"] if box["class_id"] == class_id]
        for image_id, image in images.items()
    }
    predictions = sorted(
        ((p["score"], image_id, p) for image_id, image in images.items()
         for p in image["predictions"] if p["class_id"] == class_id),
        key=lambda item: (-item[0], item[1]),
    )
    support = sum(len(items) for items in truths.values())
    matched = {image_id: set() for image_id in images}
    ranked = []
    for score, image_id, prediction in predictions:
        candidates = [
            (iou(prediction["box"], truth["box"]), index)
            for index, truth in enumerate(truths[image_id]) if index not in matched[image_id]
        ]
        best = max(candidates, default=(0.0, -1), key=lambda item: item[0])
        hit = best[0] >= iou_threshold
        if hit:
            matched[image_id].add(best[1])
        ranked.append((score, hit))
    tp = sum(hit for score, hit in ranked if score >= threshold)
    fp = sum(not hit for score, hit in ranked if score >= threshold)
    fn = support - tp
    ap = None
    if support:
        cumulative_tp = np.cumsum([hit for _, hit in ranked])
        recalls = cumulative_tp / support
        precisions = cumulative_tp / np.arange(1, len(ranked) + 1)
        ap = float(np.mean([max(precisions[recalls >= point], default=0.0)
                            for point in np.linspace(0, 1, 101)]))
    return {"tp": tp, "fp": fp, "fn": fn, "support": support,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / support if support else None, "ap50": ap}


def mask_metrics(ground_truth, predicted):
    """Compute per-class pixel counts and micro-overlap across images.

    Each iterable yields binary masks with shape (classes, height, width).
    Images may have different sizes; class counts must match. Undefined
    overlaps are returned as None.
    """
    counts = None
    sentinel = object()
    for truth, prediction in zip_longest(ground_truth, predicted, fillvalue=sentinel):
        if truth is sentinel or prediction is sentinel:
            raise ValueError("Ground truth and prediction image counts differ")
        truth = np.asarray(truth)
        prediction = np.asarray(prediction)
        if truth.shape != prediction.shape or truth.ndim != 3 or not truth.shape[0]:
            raise ValueError("Each mask pair must share shape (classes, height, width)")
        if not np.isin(truth, (0, 1)).all() or not np.isin(prediction, (0, 1)).all():
            raise ValueError("Masks must be binary")
        if counts is None:
            counts = [Counter() for _ in range(truth.shape[0])]
        if len(counts) != truth.shape[0]:
            raise ValueError("Class count differs between images")
        truth = truth.astype(bool)
        prediction = prediction.astype(bool)
        for class_id, class_counts in enumerate(counts):
            t, p = truth[class_id], prediction[class_id]
            class_counts.update(tp=int(np.count_nonzero(t & p)),
                                fp=int(np.count_nonzero(~t & p)),
                                fn=int(np.count_nonzero(t & ~p)))
    if counts is None:
        raise ValueError("At least one image is required")
    results = []
    for class_id, class_counts in enumerate(counts):
        tp, fp, fn = (class_counts[key] for key in ("tp", "fp", "fn"))
        union = tp + fp + fn
        total = 2 * tp + fp + fn
        results.append({"class_id": class_id, "tp": tp, "fp": fp, "fn": fn,
                        "iou": tp / union if union else None,
                        "dice": 2 * tp / total if total else None})
    return results


def image_alert_metrics(images, threshold):
    """Count image-level alerts against presence of any annotated defect."""
    counts = Counter()
    for image in images.values():
        positive = bool(image["ground_truth"])
        alert = any(p["score"] >= threshold for p in image["predictions"])
        counts["tp" if positive and alert else "fn" if positive else "fp" if alert else "tn"] += 1
    return {key: counts[key] for key in ("tp", "fp", "fn", "tn")}


def evaluate_boxes(images, class_ids, score_floor, miss_cost=10.0, false_alert_cost=1.0):
    """Evaluate one complete split and retain the settings needed for replay."""
    if not math.isfinite(score_floor):
        raise ValueError("Score floor must be finite")
    class_ids = list(class_ids)
    if not class_ids or len(set(class_ids)) != len(class_ids):
        raise ValueError("Class IDs must be nonempty and unique")
    for image in images.values():
        for item in image["ground_truth"] + image["predictions"]:
            if item["class_id"] not in class_ids:
                raise ValueError("An annotation has an unknown class ID")
            box = item["box"]
            if (len(box) != 4 or any(not math.isfinite(value) for value in box)
                    or box[0] >= box[2] or box[1] >= box[3]):
                raise ValueError("Boxes must be finite and have positive width and height")
        if any(not math.isfinite(item["score"]) or item["score"] < score_floor
               for item in image["predictions"]):
            raise ValueError("Prediction scores must be finite and respect the score floor")
    sweep = threshold_sweep(images, miss_cost, false_alert_cost)
    best = choose_threshold(sweep)
    return {
        "image_count": len(images), "score_floor": score_floor, "iou_threshold": 0.5,
        "miss_cost": miss_cost, "false_alert_cost": false_alert_cost,
        "threshold_sweep": sweep, "selected": best,
        "per_class": {class_id: class_metrics(images, best["threshold"], class_id)
                      for class_id in class_ids},
        "image_alerts": image_alert_metrics(images, best["threshold"]),
    }


def review_queue(images, lower, upper, capacity_per_60=9, seconds_per_image=5.0):
    """Replay ordered images in 60-frame windows with an optimistic review bound."""
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("Review bounds must be finite and lower must be below upper")
    if not isinstance(capacity_per_60, int) or isinstance(capacity_per_60, bool) or capacity_per_60 < 0:
        raise ValueError("Capacity must be a nonnegative integer per 60 frames")
    if not math.isfinite(seconds_per_image) or seconds_per_image <= 0:
        raise ValueError("Review time must be finite and positive")
    before = Counter()
    after = Counter()
    queued = reviewed = overflow = bypassed_misses = 0
    for index, image in enumerate(images):
        if index % 60 == 0:
            remaining = capacity_per_60
        automatic = match_boxes(image["ground_truth"], image["predictions"], upper)
        before.update(automatic)
        uncertain = any(lower <= p["score"] < upper for p in image["predictions"])
        if uncertain:
            queued += 1
            if remaining:
                remaining -= 1
                reviewed += 1
                # Perfect review corrects all errors only for this reviewed image.
                after.update({"tp": len(image["ground_truth"]), "fp": 0, "fn": 0})
            else:
                overflow += 1
                after.update(automatic)
        else:
            bypassed_misses += automatic["fn"]
            after.update(automatic)
    return {
        "images": len(images), "queued": queued, "reviewed": reviewed,
        "overflow": overflow, "bypassed_misses": bypassed_misses,
        "queued_per_60": 60 * queued / len(images) if images else 0.0,
        "review_seconds": reviewed * seconds_per_image,
        "capacity_per_60": capacity_per_60, "seconds_per_image": seconds_per_image,
        "before": {key: before[key] for key in ("tp", "fp", "fn")},
        "after": {key: after[key] for key in ("tp", "fp", "fn")},
    }
