"""Classical multi-class region-proposal detector for Severstal boxes."""

import argparse
from itertools import zip_longest
import json
import time
from pathlib import Path

import cv2
import joblib
import mlflow
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from projet.evaluation import detection_summary, evaluate_boxes, iou, match_boxes
from projet.mlops.tracking import start_run
from projet.preprocessing.severstal import SeverstalDataset


def crop_features(image, box, feature_set="hog"):
    x1, y1, x2, y2 = box
    crop = image[max(0, int(np.floor(y1))):min(image.shape[0], int(np.ceil(y2))),
                 max(0, int(np.floor(x1))):min(image.shape[1], int(np.ceil(x2)))]
    if not crop.size:
        raise ValueError("Empty crop")
    resized = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    features = hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    if feature_set == "texture":
        lbp = local_binary_pattern(resized, 8, 1, method="uniform")
        histogram = np.bincount(lbp.astype(np.int32).ravel(), minlength=10).astype(np.float32)
        histogram /= histogram.sum()
        shape = np.array([np.log1p(crop.shape[1]), np.log1p(crop.shape[0]),
                          float(crop.mean()) / 255, float(crop.std()) / 255])
        features = np.concatenate((features, histogram, shape))
    return features


def propose(image, cutoff, min_area, max_regions, sigma=9, kernel=5, mode="single"):
    if mode == "multiscale":
        boxes = []
        groups = [propose(image, threshold, min_area, max_regions, scale, kernel)
                  for scale, threshold in ((3, 10), (9, 20), (20, 25))]
        for candidates in zip_longest(*groups):
            for box in candidates:
                if box is not None and box not in boxes:
                    boxes.append(box)
                    if len(boxes) == max_regions:
                        return boxes
        return boxes[:max_regions]
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    contrast = cv2.absdiff(image, blurred)
    binary = (contrast >= cutoff).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions = [(int(area), [int(x), int(y), int(x + width), int(y + height)])
               for x, y, width, height, area in stats[1:count]
               if area >= min_area and width >= 3 and height >= 3]
    regions.sort(key=lambda item: -item[0])
    return [box for _, box in regions[:max_regions]]


def sample_training(data, count, seed, *, hard_negatives=False, cutoff=25, min_area=60,
                    max_regions=100, sigma=9, kernel=5, proposal_mode="single", feature_set="hog"):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(data))[:count]
    vectors, labels, ids = [], [], []
    for n, index in enumerate(indices, 1):
        item = data[int(index)]
        ids.append(item["image_id"])
        image, boxes = item["image"], item["boxes"]
        for box, label in zip(boxes, item["labels"]):
            vectors.append(crop_features(image, box, feature_set))
            labels.append(int(label))
        if hard_negatives:
            positive_proposals, negative_proposals = [], []
            for proposal in propose(image, cutoff, min_area, max_regions, sigma, kernel, proposal_mode):
                overlaps = [iou(proposal, box) for box in boxes]
                best = max(overlaps, default=0)
                if best >= 0.5:
                    positive_proposals.append((proposal, int(item["labels"][int(np.argmax(overlaps))])))
                elif best < 0.1:
                    negative_proposals.append(proposal)
            for proposal, label in positive_proposals[:10]:
                vectors.append(crop_features(image, proposal, feature_set))
                labels.append(label)
            if negative_proposals:
                selected = rng.choice(len(negative_proposals), min(4, len(negative_proposals)), replace=False)
                for selected_index in selected:
                    vectors.append(crop_features(image, negative_proposals[int(selected_index)], feature_set))
                    labels.append(4)
        height, width = image.shape
        accepted = 0
        for _ in range(40):
            crop_width, crop_height = ((64, 64), (128, 64), (128, 128))[rng.integers(3)]
            x, y = int(rng.integers(width - crop_width + 1)), int(rng.integers(height - crop_height + 1))
            box = [x, y, x + crop_width, y + crop_height]
            if any(iou(box, truth) > 0 for truth in boxes):
                continue
            vectors.append(crop_features(image, box, feature_set))
            labels.append(4)
            accepted += 1
            if accepted == 2:
                break
        if n % 500 == 0:
            print(f"region detector: sampled {n}/{len(indices)}", flush=True)
    return np.asarray(vectors, dtype=np.float32), np.asarray(labels), ids


def predict(image, model, cutoff, min_area, max_regions, sigma=9, kernel=5, proposal_mode="single", feature_set="hog"):
    boxes = propose(image, cutoff, min_area, max_regions, sigma, kernel, proposal_mode)
    if not boxes:
        return [], boxes
    features = np.asarray([crop_features(image, box, feature_set) for box in boxes], dtype=np.float32)
    margins = model.decision_function(features) if hasattr(model, "decision_function") else model.predict_proba(features)
    classes = model.classes_
    background = int(np.flatnonzero(classes == 4)[0])
    defect_indices = np.flatnonzero(classes != 4)
    predictions = []
    for box, row in zip(boxes, margins):
        best = defect_indices[np.argmax(row[defect_indices])]
        predictions.append({"class_id": int(classes[best]), "box": box,
                            "score": float(row[best] - row[background])})
    kept = []
    for prediction in sorted(predictions, key=lambda value: -value["score"]):
        if not any(other["class_id"] == prediction["class_id"] and iou(other["box"], prediction["box"]) > 0.5
                   for other in kept):
            kept.append(prediction)
    return kept, boxes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cutoff", type=int, default=25)
    parser.add_argument("--min-area", type=int, default=60)
    parser.add_argument("--max-regions", type=int, default=100)
    parser.add_argument("--sigma", type=float, default=9)
    parser.add_argument("--kernel", type=int, default=5)
    parser.add_argument("--validation-ids", type=Path, help="Proposal study JSON containing validation_tuning_ids")
    parser.add_argument("--validation-id-key", default="validation_tuning_ids")
    parser.add_argument("--fixed-threshold", type=float, help="Use a threshold previously selected on tuning data")
    parser.add_argument("--model-from", type=Path, help="Load a saved run directory instead of fitting again")
    parser.add_argument("--hard-negatives", action="store_true", help="Train on proposals and hard background regions")
    parser.add_argument("--classifier", choices=("linear-svc", "sgd", "extra-trees"), default="linear-svc")
    parser.add_argument("--svm-alpha", type=float, default=0.0001, help="SGD linear SVM regularization strength")
    parser.add_argument("--proposal-mode", choices=("single", "multiscale"), default="single")
    parser.add_argument("--feature-set", choices=("hog", "texture"), default="hog")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.train_count, args.validation_count, args.min_area, args.max_regions) < 1 or args.seed < 0 or not 0 <= args.cutoff <= 255:
        parser.error("Counts and areas must be positive, seed nonnegative and cutoff between 0 and 255")
    if args.sigma <= 0 or args.svm_alpha <= 0 or not np.isfinite(args.svm_alpha) or args.kernel < 1 or args.kernel % 2 == 0:
        parser.error("Sigma must be positive and kernel must be a positive odd integer")
    if args.output.exists():
        parser.error("Output directory already exists")
    train, validation = SeverstalDataset("train"), SeverstalDataset("validation")
    saved = (json.loads((args.model_from / "results.json").read_text(encoding="utf-8"))
             if args.model_from else None)
    with start_run("contrast-hog-svm", phase="detection-validation", seed=args.seed,
                   description="Classical proposal detector; threshold tuned on validation.") as run:
        mlflow.log_params({"train_count_requested": saved["config"]["train_count"] if saved else args.train_count,
                           "validation_count_requested": args.validation_count,
                           "contrast_cutoff": args.cutoff, "min_area": args.min_area,
                           "max_regions": args.max_regions, "sigma": args.sigma, "kernel": args.kernel,
                           "classifier": args.classifier, "class_weight": "balanced",
                           "hard_negatives": saved["config"]["hard_negatives"] if saved else args.hard_negatives,
                           "proposal_mode": args.proposal_mode, "feature_set": args.feature_set,
                           "threshold_mode": "fixed" if args.fixed_threshold is not None else "validation-selected",
                           "validation_id_key": args.validation_id_key if args.validation_ids else "random sample"})
        if args.model_from:
            for setting in ("classifier", "proposal_mode", "feature_set", "cutoff", "min_area", "max_regions", "sigma", "kernel"):
                if saved["config"].get(setting, {"proposal_mode": "single", "feature_set": "hog"}.get(setting)) != getattr(args, setting):
                    raise ValueError(f"Saved model setting differs: {setting}")
            if args.fixed_threshold is None and not args.validation_ids:
                raise ValueError("Threshold tuning for a saved model requires explicit validation IDs")
            model = joblib.load(args.model_from / "model.joblib")
            train_ids = saved["train_ids"]
            feature_seconds = fit_seconds = 0.0
            mlflow.log_param("source_run_id", saved["run_id"])
        else:
            start = time.perf_counter()
            vectors, labels, train_ids = sample_training(train, args.train_count, args.seed,
                hard_negatives=args.hard_negatives, cutoff=args.cutoff, min_area=args.min_area,
                max_regions=args.max_regions, sigma=args.sigma, kernel=args.kernel,
                proposal_mode=args.proposal_mode, feature_set=args.feature_set)
            if set(labels) != {0, 1, 2, 3, 4}:
                raise ValueError(f"Training subset lacks classes: {sorted(set(labels))}")
            feature_seconds = time.perf_counter() - start
            if args.classifier == "sgd":
                classifier = SGDClassifier(loss="hinge", alpha=args.svm_alpha, class_weight="balanced",
                                           random_state=args.seed, max_iter=1000, tol=0.001, average=True)
            elif args.classifier == "linear-svc":
                classifier = LinearSVC(C=1, class_weight="balanced", random_state=args.seed, max_iter=10000)
            else:
                classifier = ExtraTreesClassifier(n_estimators=150, class_weight="balanced", random_state=args.seed, n_jobs=4)
            mlflow.log_param("classifier_settings", str(classifier.get_params()))
            model = make_pipeline(StandardScaler(), classifier)
            start = time.perf_counter()
            model.fit(vectors, labels)
            fit_seconds = time.perf_counter() - start
        if args.validation_ids:
            selected_ids = json.loads(args.validation_ids.read_text(encoding="utf-8"))[args.validation_id_key]
            if len(selected_ids) != len(set(selected_ids)) or not set(selected_ids) <= set(validation.ids):
                raise ValueError("Validation IDs must be unique members of the validation split")
            lookup = {image_id: index for index, image_id in enumerate(validation.ids)}
            indices = [lookup[image_id] for image_id in selected_ids[:args.validation_count]]
        else:
            indices = np.random.default_rng(args.seed).permutation(len(validation))[:args.validation_count]
        images, candidate_found, candidate_total, durations = {}, 0, 0, []
        for n, index in enumerate(indices, 1):
            sample = validation[int(index)]
            truth = [{"class_id": int(label), "box": box.tolist()}
                     for label, box in zip(sample["labels"], sample["boxes"])]
            start = time.perf_counter()
            predictions, candidates = predict(sample["image"], model, args.cutoff, args.min_area,
                                               args.max_regions, args.sigma, args.kernel,
                                               args.proposal_mode, args.feature_set)
            durations.append(time.perf_counter() - start)
            candidate_found += sum(any(iou(item["box"], box) >= 0.5 for box in candidates) for item in truth)
            candidate_total += len(truth)
            images[sample["image_id"]] = {"ground_truth": truth, "predictions": predictions}
            if n % 25 == 0:
                print(f"region detector: evaluated {n}/{len(indices)}", flush=True)
        scores = [prediction["score"] for item in images.values() for prediction in item["predictions"]]
        floor = min(scores, default=0.0)
        if args.fixed_threshold is None:
            report = evaluate_boxes(images, range(4), floor)
        else:
            if not np.isfinite(args.fixed_threshold):
                raise ValueError("Fixed threshold must be finite")
            counts = [match_boxes(item["ground_truth"], item["predictions"], args.fixed_threshold)
                      for item in images.values()]
            totals = {key: sum(row[key] for row in counts) for key in ("tp", "fp", "fn")}
            report = {"score_floor": floor, "iou_threshold": 0.5,
                      "selected": {"threshold": args.fixed_threshold, **totals,
                                   "cost_per_image": (10 * totals["fn"] + totals["fp"]) / len(images)}}
        summary = detection_summary(images, report["selected"]["threshold"], range(4))
        metrics = {"cost_per_image": report["selected"]["cost_per_image"],
                   "tp": report["selected"]["tp"], "fp": report["selected"]["fp"], "fn": report["selected"]["fn"],
                   "threshold": report["selected"]["threshold"], "fit_seconds": fit_seconds,
                   "train_feature_seconds": feature_seconds, "inference_median_seconds": float(np.median(durations)),
                   "inference_p95_seconds": float(np.percentile(durations, 95))}
        if candidate_total:
            metrics["proposal_recall"] = candidate_found / candidate_total
        for name in ("map50", "map50_95"):
            if summary[name] is not None:
                metrics[name] = summary[name]
        for class_id, result in summary["per_class"].items():
            for name in ("support", "precision", "recall", "f1", "ap50", "ap50_95"):
                if result[name] is not None:
                    metrics[f"class_{class_id + 1}_{name}"] = result[name]
        mlflow.log_metrics(metrics)
        args.output.mkdir(parents=True)
        joblib.dump(model, args.output / "model.joblib")
        results = {"run_id": run.info.run_id, "train_ids": train_ids, "validation_ids": list(images),
                   "config": vars(args) | {"output": str(args.output), "model_from": str(args.model_from) if args.model_from else None,
                                           "hard_negatives": saved["config"]["hard_negatives"] if saved else args.hard_negatives,
                                           "train_count": saved["config"]["train_count"] if saved else args.train_count,
                                           "validation_ids": str(args.validation_ids) if args.validation_ids else None}, "metrics": metrics,
                   "proposal_found": candidate_found, "proposal_total": candidate_total,
                   "detection_summary": summary, "report": report, "predictions": images,
                   "timing_boundary": "decoded image to proposal generation, HOG, classification and NMS; no warmup"}
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifacts(str(args.output))
        mlflow.log_artifact(__file__, artifact_path="source")
        print(f"region detector: {metrics}", flush=True)


if __name__ == "__main__":
    main()
