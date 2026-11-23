"""Run a bounded pretrained YOLO detection experiment on Severstal."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import mlflow
import numpy as np
from ultralytics import YOLO

from projet.evaluation import detection_summary, evaluate_boxes, match_boxes
from projet.mlops.tracking import start_run
from projet.preprocessing.severstal import SeverstalDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--validation-ids", type=Path, required=True)
    parser.add_argument("--validation-id-key", default="validation_tuning_ids")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=500)
    parser.add_argument("--validation-count", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--fixed-threshold", type=float)
    args = parser.parse_args()
    if args.output.exists() or min(args.train_count, args.validation_count, args.epochs, args.image_size, args.batch) < 1 or args.seed < 0:
        parser.error("Output must be new, counts positive and seed nonnegative")
    if not args.weights.is_file():
        parser.error("Pretrained weights file does not exist")
    if args.evaluate_only and (args.fixed_threshold is None or not np.isfinite(args.fixed_threshold)):
        parser.error("Evaluation-only requires a finite threshold selected beforehand")
    train, validation = SeverstalDataset("train"), SeverstalDataset("validation")
    selected = np.random.default_rng(args.seed).permutation(len(train))[:args.train_count]
    train_ids = [train.ids[index] for index in selected]
    validation_ids = json.loads(args.validation_ids.read_text(encoding="utf-8"))[args.validation_id_key][:args.validation_count]
    if not validation_ids or len(validation_ids) != len(set(validation_ids)) or not set(validation_ids) <= set(validation.ids):
        raise ValueError("Validation IDs must be unique members of the validation split")
    args.output.mkdir(parents=True)
    for split, ids in (("train", train_ids), ("validation", validation_ids)):
        (args.output / f"{split}.txt").write_text(
            "".join(f"{(train.root / 'images' / image_id).as_posix()}\n" for image_id in ids), encoding="utf-8")
    config = {"path": str(train.root), "train": str((args.output / "train.txt").resolve()),
              "val": str((args.output / "validation.txt").resolve()),
              "names": {index: f"class_{index + 1}" for index in range(4)}}
    config_path = args.output / "dataset.yaml"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "severstal-inspection"
    with start_run("yolo11n-pretrained", phase="detection-comparison" if args.evaluate_only else "detection-pilot", seed=args.seed,
                   description="Bounded CPU pilot on training and validation tuning images.") as run:
        params = {"input_weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                  "pilot_validation_count": len(validation_ids), "pilot_image_size": args.image_size,
                  "mode": "evaluate-only" if args.evaluate_only else "train"}
        if not args.evaluate_only:
            params.update({"pilot_train_count": len(train_ids), "pilot_epochs": args.epochs})
        mlflow.log_params(params)
        training_seconds = None
        checkpoint = args.weights
        if not args.evaluate_only:
            model = YOLO(str(args.weights))
            start = time.perf_counter()
            model.train(data=str(config_path.resolve()), epochs=args.epochs, imgsz=args.image_size,
                        batch=args.batch, workers=0, device="cpu", seed=args.seed, plots=False,
                        project=str(args.output.resolve()), name="training", exist_ok=False, verbose=False)
            training_seconds = time.perf_counter() - start
            checkpoint = args.output / "training" / "weights" / "best.pt"
        model = YOLO(str(checkpoint))
        lookup = {image_id: index for index, image_id in enumerate(validation.ids)}
        images = {}
        for index, image_id in enumerate(validation_ids, start=1):
            sample = validation[lookup[image_id]]
            output = model.predict(source=str(validation.root / "images" / image_id), imgsz=args.image_size,
                                   conf=0.001, iou=0.7, max_det=1000, device="cpu", verbose=False)[0]
            predictions = [{"class_id": int(label), "box": box.tolist(), "score": float(score)}
                           for box, score, label in zip(output.boxes.xyxy.cpu().numpy(),
                               output.boxes.conf.cpu().numpy(), output.boxes.cls.cpu().numpy())]
            truths = [{"class_id": int(label), "box": box.tolist()}
                      for label, box in zip(sample["labels"], sample["boxes"])]
            images[image_id] = {"ground_truth": truths, "predictions": predictions}
            if index % 25 == 0:
                print(f"YOLO pilot: evaluated {index}/{len(validation_ids)}", flush=True)
        if args.fixed_threshold is None:
            report = evaluate_boxes(images, range(4), score_floor=0.001)
        else:
            counts = [match_boxes(item["ground_truth"], item["predictions"], args.fixed_threshold)
                      for item in images.values()]
            totals = {key: sum(row[key] for row in counts) for key in ("tp", "fp", "fn")}
            report = {"score_floor": 0.001, "iou_threshold": 0.5,
                      "selected": {"threshold": args.fixed_threshold, **totals,
                                   "cost_per_image": (10 * totals["fn"] + totals["fp"]) / len(images)}}
        summary = detection_summary(images, report["selected"]["threshold"], range(4))
        metrics = {"pilot_training_seconds": training_seconds, "pilot_map50": summary["map50"],
                   "pilot_map50_95": summary["map50_95"], **{f"pilot_{key}": value
                   for key, value in report["selected"].items()}}
        mlflow.log_metrics({key: value for key, value in metrics.items() if value is not None})
        results = {"run_id": run.info.run_id, "train_ids": [] if args.evaluate_only else train_ids, "validation_ids": validation_ids,
                   "metrics": metrics, "summary": summary, "report": report, "predictions": images,
                   "threshold_selection": "fixed from prior tuning" if args.fixed_threshold is not None
                                          else "selected on these validation tuning images"}
        (args.output / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(args.output / "results.json"))
        mlflow.log_artifact(str(checkpoint), artifact_path="model")
        mlflow.log_artifact(__file__, artifact_path="source")
        print(f"YOLO pilot: {metrics}", flush=True)


if __name__ == "__main__":
    main()
