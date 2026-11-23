"""Compare classical image-alert models with held-out validation reporting."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import sklearn
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split

from projet.classification_metrics import binary_metrics
from projet.mlops.tracking import start_run
from projet.preprocessing.severstal import SeverstalDataset
from projet.training.baselines import choose_threshold, extract_split


def metrics_for(labels, scores, threshold):
    result = binary_metrics(labels, scores, threshold)
    result["cost_per_image"] = (10 * result["fn"] + result["fp"]) / len(labels)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--methods", nargs="+", choices=("forest-150-leaf2", "forest-300-leaf1",
                        "extra-trees-300-leaf1", "hist-gradient-boosting"))
    parser.add_argument("--comparison-fraction", type=float, default=0.5)
    args = parser.parse_args()
    if args.output.exists() or args.seed < 0 or not 0 < args.comparison_fraction < 1:
        parser.error("Output must be new, seed nonnegative, and comparison fraction between 0 and 1")
    train, validation = SeverstalDataset("train"), SeverstalDataset("validation")
    train_x, train_y, train_ids, _, train_extract_seconds = extract_split(train, "texture-forest", len(train), args.seed)
    valid_x, valid_y, valid_ids, classes, valid_extract_seconds = extract_split(validation, "texture-forest", len(validation), args.seed)
    tune, comparison = train_test_split(np.arange(len(valid_y)), test_size=args.comparison_fraction,
                                        random_state=args.seed, stratify=valid_y)
    methods = {
        "forest-150-leaf2": RandomForestClassifier(n_estimators=150, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1),
        "forest-300-leaf1": RandomForestClassifier(n_estimators=300, min_samples_leaf=1,
            class_weight="balanced_subsample", random_state=args.seed, n_jobs=-1),
        "extra-trees-300-leaf1": ExtraTreesClassifier(n_estimators=300, min_samples_leaf=1,
            class_weight="balanced", random_state=args.seed, n_jobs=-1),
        "hist-gradient-boosting": HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1,
            max_leaf_nodes=31, l2_regularization=1, random_state=args.seed),
    }
    args.output.mkdir(parents=True)
    for name in args.methods or methods:
        model = methods[name]
        with start_run(name, phase="binary-validation-tuning", seed=args.seed,
                       description="Threshold tuned on one validation half; other half reported separately.") as run:
            params = model.get_params()
            mlflow.log_params({"estimator": type(model).__name__, "model_settings": json.dumps(params, sort_keys=True),
                               "features": "spatial intensity, gradient, intensity histogram at 256x64",
                               "train_count": len(train_ids), "tune_count": len(tune),
                               "comparison_count": len(comparison), "comparison_fraction": args.comparison_fraction,
                               "miss_cost": 10, "false_alert_cost": 1,
                               "manifest_sha256": hashlib.sha256((train.root / "split.json").read_bytes()).hexdigest(),
                               "python": platform.python_version(), "scikit_learn": sklearn.__version__,
                               "train_augmentation": False})
            start = time.perf_counter()
            model.fit(train_x, train_y)
            fit_seconds = time.perf_counter() - start
            scores = model.predict_proba(valid_x)[:, 1]
            threshold = choose_threshold(valid_y[tune], scores[tune])
            tune_metrics = metrics_for(valid_y[tune], scores[tune], threshold)
            comparison_metrics = metrics_for(valid_y[comparison], scores[comparison], threshold)
            mlflow.log_metrics({f"tune_{key}": value for key, value in tune_metrics.items() if value is not None})
            mlflow.log_metrics({f"comparison_{key}": value for key, value in comparison_metrics.items() if value is not None})
            mlflow.log_metrics({"fit_seconds": fit_seconds, "train_feature_seconds": train_extract_seconds,
                                "validation_feature_seconds": valid_extract_seconds})
            class_results = {}
            for class_id in range(4):
                present = classes[comparison, class_id].astype(bool)
                support = int(present.sum())
                recall = float(np.mean(scores[comparison][present] >= threshold)) if support else None
                class_results[str(class_id + 1)] = {"support": support, "image_alert_recall": recall}
                if recall is not None:
                    mlflow.log_metric(f"comparison_class_{class_id + 1}_image_alert_recall", recall)
                    mlflow.log_metric(f"comparison_class_{class_id + 1}_support", support)
            output = args.output / name
            output.mkdir()
            joblib.dump(model, output / "model.joblib")
            result = {"run_id": run.info.run_id, "model": name, "params": params,
                      "train_ids": train_ids, "validation_ids": valid_ids,
                      "tune_ids": [valid_ids[i] for i in tune],
                      "comparison_ids": [valid_ids[i] for i in comparison],
                      "validation_labels": valid_y.tolist(), "validation_scores": scores.tolist(),
                      "tune_metrics": tune_metrics, "comparison_metrics": comparison_metrics,
                      "comparison_class_image_alert_recall": class_results,
                      "fit_seconds": fit_seconds, "train_feature_seconds": train_extract_seconds,
                      "validation_feature_seconds": valid_extract_seconds}
            (output / "results.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
            mlflow.log_artifacts(str(output))
            mlflow.log_artifact(__file__, artifact_path="source")
            mlflow.log_artifact(str(Path(__file__).with_name("baselines.py")), artifact_path="source")
            print(f"{name}: tuning={tune_metrics} comparison={comparison_metrics} class2={class_results['2']}", flush=True)


if __name__ == "__main__":
    main()
