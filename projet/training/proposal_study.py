"""Compare contrast-region proposals on a validation tuning subset."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import mlflow
import numpy as np
from sklearn.model_selection import train_test_split

from projet.evaluation import iou
from projet.mlops.tracking import start_run
from projet.preprocessing.severstal import SeverstalDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sigmas", type=float, nargs="+", default=[3, 9, 20])
    parser.add_argument("--cutoffs", type=int, nargs="+", default=[10, 20, 25, 30])
    parser.add_argument("--kernels", type=int, nargs="+", default=[3, 5, 7])
    parser.add_argument("--min-areas", type=int, nargs="+", default=[20, 60])
    parser.add_argument("--max-regions", type=int, default=100)
    args = parser.parse_args()
    if (args.output.exists() or args.count < 1 or args.seed < 0 or args.max_regions < 1
            or any(not np.isfinite(x) or x <= 0 for x in args.sigmas)
            or any(x < 0 or x > 255 for x in args.cutoffs)
            or any(x < 1 or x % 2 == 0 for x in args.kernels)
            or any(x < 1 for x in args.min_areas)):
        parser.error("Invalid output, count, seed, proposal limit or grid values")
    data = SeverstalDataset("validation")
    labels = []
    for image_id in data.ids:
        labels.append(int(bool((data.root / "labels" / f"{Path(image_id).stem}.txt").read_text().strip())))
    # Reproduce the binary study's seeded permutation and stratified partition.
    indices = np.random.default_rng(args.seed).permutation(len(data))
    ordered_labels = np.asarray(labels)[indices]
    tune, _ = train_test_split(np.arange(len(indices)), test_size=0.5, random_state=args.seed, stratify=ordered_labels)
    selected = indices[tune[:args.count]]
    results = defaultdict(lambda: {"found": 0, "total": 0, "proposals": 0})
    ids = []
    with start_run("contrast-proposals", phase="localization-tuning", seed=args.seed,
                   description="Proposal recall study on validation tuning images."):
        mlflow.log_params({"sigmas": args.sigmas, "cutoffs": args.cutoffs, "kernels": args.kernels,
                           "min_areas": args.min_areas, "max_regions": args.max_regions,
                           "requested_images": args.count})
        for number, index in enumerate(selected, 1):
            item = data[int(index)]
            image, truth = item["image"], item["boxes"]
            ids.append(item["image_id"])
            for sigma in args.sigmas:
                contrast = cv2.absdiff(image, cv2.GaussianBlur(image, (0, 0), sigma))
                for cutoff in args.cutoffs:
                    binary = (contrast >= cutoff).astype(np.uint8) * 255
                    for kernel in args.kernels:
                        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((kernel, kernel), np.uint8))
                        count, _, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
                        for minimum_area in args.min_areas:
                            components = sorted(((int(area), [int(x), int(y), int(x + w), int(y + h)])
                                for x, y, w, h, area in stats[1:count]
                                if area >= minimum_area and w >= 3 and h >= 3),
                                key=lambda pair: -pair[0])[:args.max_regions]
                            boxes = [box for _, box in components]
                            row = results[f"sigma{sigma}_cutoff{cutoff}_kernel{kernel}_area{minimum_area}"]
                            row["found"] += sum(any(iou(box, target) >= 0.5 for box in boxes) for target in truth)
                            row["total"] += len(truth)
                            row["proposals"] += len(boxes)
            if number % 25 == 0:
                print(f"proposals: {number}/{len(selected)}", flush=True)
        for name, row in results.items():
            row["recall"] = row["found"] / row["total"] if row["total"] else None
            row["mean_proposals"] = row["proposals"] / len(ids)
            if row["recall"] is not None:
                mlflow.log_metric(f"{name}_recall", row["recall"])
        result = {"validation_tuning_ids": ids, "settings": dict(results), "seed": args.seed}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        mlflow.log_artifact(str(args.output))
        mlflow.log_artifact(__file__, artifact_path="source")
        for name, row in sorted(results.items(), key=lambda pair: -(pair[1]["recall"] or 0))[:5]:
            print(name, row, flush=True)


if __name__ == "__main__":
    main()
