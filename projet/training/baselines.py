"""Train classical Severstal image-alert baselines on fixed data splits."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import cv2
import joblib
import mlflow
import numpy as np
from skimage.feature import hog
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from projet.classification_metrics import binary_metrics
from projet.mlops.tracking import start_run
from projet.preprocessing.severstal import SeverstalDataset
from projet.timing import benchmark_inference


def extract_features(image, method):
    """Resize the complete source image consistently, then compute features."""
    if method == "texture-multiscale":
        resized = cv2.resize(image, (512, 82), interpolation=cv2.INTER_AREA)
        gray = resized.astype(np.float32) / 255
        gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
        contrast = cv2.absdiff(gray, cv2.GaussianBlur(gray, (0, 0), 3))
        features = []
        for values in (gray, gradient, contrast):
            for size in ((32, 8), (64, 8)):
                mean = cv2.resize(values, size, interpolation=cv2.INTER_AREA)
                squared = cv2.resize(values * values, size, interpolation=cv2.INTER_AREA)
                features.extend((mean.ravel(), np.sqrt(np.maximum(squared - mean * mean, 0)).ravel()))
        histogram = cv2.calcHist([resized], [0], None, [32], [0, 256]).reshape(-1) / resized.size
        return np.concatenate((*features, histogram))
    resized = cv2.resize(image, (256, 64), interpolation=cv2.INTER_AREA)
    if method == "hog-sgd-svm":
        return hog(resized, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
    gray = resized.astype(np.float32) / 255
    gradient = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    histogram = cv2.calcHist([resized], [0], None, [32], [0, 256]).reshape(-1) / resized.size
    return np.concatenate((cv2.resize(gray, (32, 8), interpolation=cv2.INTER_AREA).ravel(),
                           cv2.resize(gradient, (32, 8), interpolation=cv2.INTER_AREA).ravel(), histogram))


def choose_threshold(labels, scores):
    """Select a validation threshold under the stated image-level 10:1 scenario."""
    _, _, thresholds = precision_recall_curve(labels, scores)
    candidates = np.append(thresholds, np.nextafter(scores.max(), np.inf))
    best = None
    for threshold in candidates:
        predictions = scores >= threshold
        false_negatives = int(np.count_nonzero((labels == 1) & ~predictions))
        false_positives = int(np.count_nonzero((labels == 0) & predictions))
        key = (10 * false_negatives + false_positives, false_negatives, -threshold)
        if best is None or key < best[0]:
            best = (key, float(threshold))
    return best[1]


def extract_split(dataset, method, count, seed):
    indices = np.random.default_rng(seed).permutation(len(dataset))[:count]
    features, labels, ids, class_presence = [], [], [], []
    start = time.perf_counter()
    for number, index in enumerate(indices, start=1):
        sample = dataset[int(index)]
        features.append(extract_features(sample["image"], method))
        labels.append(int(bool(len(sample["labels"]))))
        class_presence.append([int(class_id in sample["labels"]) for class_id in range(4)])
        ids.append(sample["image_id"])
        if number % 500 == 0:
            print(f"{method}: extracted {number}/{len(indices)}", flush=True)
    return np.asarray(features, dtype=np.float32), np.asarray(labels), ids, np.asarray(class_presence), time.perf_counter() - start


def run(method, train, validation, args):
    with start_run(method, phase="binary-validation", seed=args.seed,
                   description="Image-level defect alert baseline; threshold tuned on validation.") as run:
        mlflow.log_params({"image_size": "256x64", "train_count_requested": args.train_count,
                           "validation_count_requested": args.validation_count,
                           "augmentation": args.augment, "false_negative_cost": 10,
                           "false_positive_cost": 1})
        train_x, train_y, train_ids, _, feature_seconds = extract_split(train, method, args.train_count, args.seed)
        valid_x, valid_y, valid_ids, class_presence, validation_feature_seconds = extract_split(
            validation, method, args.validation_count, args.seed)
        if len(np.unique(train_y)) != 2 or len(np.unique(valid_y)) != 2:
            raise ValueError("Training and validation samples must each contain both image classes")
        if method == "hog-sgd-svm":
            model = make_pipeline(StandardScaler(), SGDClassifier(loss="hinge", alpha=0.0001,
                class_weight="balanced", random_state=args.seed, max_iter=1000, tol=0.001, average=True))
            mlflow.log_params({"classifier": "SGDClassifier", "loss": "hinge", "alpha": 0.0001,
                               "max_iter": 1000, "tolerance": 0.001, "average": True,
                               "class_weight": "balanced"})
        else:
            model = RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
                                           class_weight="balanced_subsample", random_state=args.seed,
                                           n_jobs=-1)
            mlflow.log_params({"classifier": "RandomForestClassifier", "trees": 150,
                               "min_samples_leaf": 2, "class_weight": "balanced_subsample"})
        start = time.perf_counter()
        model.fit(train_x, train_y)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        scores = (model.decision_function(valid_x) if method == "hog-sgd-svm"
                  else model.predict_proba(valid_x)[:, 1])
        scoring_seconds = time.perf_counter() - start
        threshold = choose_threshold(valid_y, scores)
        metrics = binary_metrics(valid_y, scores, threshold)
        metrics["cost_per_image"] = (10 * metrics["fn"] + metrics["fp"]) / len(valid_y)
        class_metrics = {}
        for class_id in range(4):
            present = class_presence[:, class_id].astype(bool)
            support = int(present.sum())
            recall = float(np.mean(scores[present] >= threshold)) if support else None
            class_metrics[str(class_id + 1)] = {"support": support, "image_alert_recall": recall}
            if recall is not None:
                mlflow.log_metric(f"class_{class_id + 1}_image_alert_recall", recall)
                mlflow.log_metric(f"class_{class_id + 1}_image_support", support)
        images = []
        for image_id in valid_ids[:50]:
            image = cv2.imread(str(validation.root / "images" / image_id), cv2.IMREAD_GRAYSCALE)
            if image is None:
                raise FileNotFoundError(image_id)
            images.append(image)

        def predict_image(image):
            features = extract_features(image, method).reshape(1, -1)
            return model.decision_function(features) if method == "hog-sgd-svm" else model.predict_proba(features)

        timing = benchmark_inference(predict_image, images, warmup=3)
        mlflow.log_metrics({f"inference_{key}": timing[key]
                            for key in ("mean_seconds", "median_seconds", "p95_seconds", "max_seconds")})
        metrics.update({"fit_seconds": fit_seconds, "train_feature_seconds": feature_seconds,
                        "validation_feature_seconds": validation_feature_seconds,
                        "validation_scoring_seconds": scoring_seconds})
        mlflow.log_metrics({key: value for key, value in metrics.items() if value is not None})
        mlflow.log_params({"train_count_actual": len(train_ids), "validation_count_actual": len(valid_ids),
                           "python": platform.python_version(), "opencv": cv2.__version__,
                           "manifest_sha256": hashlib.sha256((train.root / "split.json").read_bytes()).hexdigest()})
        output = args.output / method
        output.mkdir(parents=True, exist_ok=False)
        joblib.dump(model, output / "model.joblib")
        results = {"method": method, "run_id": run.info.run_id, "metrics": metrics,
                   "class_image_alert_recall": class_metrics, "timing": timing,
                   "timing_ids": valid_ids[:50],
                   "timing_boundary": "decoded grayscale image through feature extraction and prediction; disk excluded",
                   "train_ids": train_ids, "validation_ids": valid_ids,
                   "validation_labels": valid_y.tolist(), "validation_scores": scores.tolist(),
                   "threshold_selection": "minimum 10*FN + FP on this validation sample"}
        (output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifacts(str(output))
        mlflow.log_artifact(__file__, artifact_path="source")
        mlflow.log_artifact(str(Path(__file__).resolve().parents[2] / "requirements.txt"), artifact_path="source")
        print(f"{method}: {metrics}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-count", type=int, default=10050)
    parser.add_argument("--validation-count", type=int, default=1252)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--methods", nargs="+", choices=("hog-sgd-svm", "texture-forest"),
                        default=("hog-sgd-svm", "texture-forest"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.train_count < 2 or args.validation_count < 2 or args.seed < 0:
        parser.error("Counts must be at least two and seed nonnegative")
    if args.output.exists():
        parser.error("Output directory already exists")
    train = SeverstalDataset("train", augment_training=args.augment, seed=args.seed)
    validation = SeverstalDataset("validation")
    for method in args.methods:
        run(method, train, validation, args)


if __name__ == "__main__":
    main()
