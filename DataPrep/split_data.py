"""Create seeded image-level manifests for the Severstal experiments."""

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
OUTPUT = ROOT / "manifests"
SEED = 0
RARE_CLASS = 2


def main():
    images = sorted(path.name for path in (RAW / "train_images").iterdir() if path.is_file())
    if not images:
        raise SystemExit("No training images found")
    classes = {image: set() for image in images}
    with (RAW / "train.csv").open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["EncodedPixels"].strip():
                classes[row["ImageId"]].add(int(row["ClassId"]))

    # Stratify by the full set of classes, including the empty set.
    groups = defaultdict(list)
    for image in images:
        groups[tuple(sorted(classes[image]))].append(image)
    rng = random.Random(SEED)
    splits = {"train": [], "validation": [], "test": []}
    for group in sorted(groups):
        members = groups[group]
        rng.shuffle(members)
        train_end = int(len(members) * 0.8)
        validation_end = train_end + int(len(members) * 0.1)
        splits["train"].extend(members[:train_end])
        splits["validation"].extend(members[train_end:validation_end])
        splits["test"].extend(members[validation_end:])
    for members in splits.values():
        members.sort()

    rare = [image for image in splits["train"] if RARE_CLASS in classes[image]]
    rng.shuffle(rare)
    if len(rare) < 15:
        raise SystemExit("Fewer than 15 rare-class training images")
    rare_order = rare[:15]
    target_free = [image for image in splits["train"] if RARE_CLASS not in classes[image]]
    conditions = {
        str(count): sorted(target_free + rare_order[:count]) for count in (5, 10, 15)
    }
    summary = {}
    for split, members in splits.items():
        counts = Counter(class_id for image in members for class_id in classes[image])
        summary[split] = {
            "images": len(members),
            "no_annotated_defect": sum(not classes[image] for image in members),
            "images_by_class": {str(class_id): counts[class_id] for class_id in range(1, 5)},
        }

    manifest = {
        "seed": SEED,
        "method": "stratified by exact image-level class set, 80/10/10 per group",
        "rare_class": RARE_CLASS,
        "splits": splits,
        "rare_order": rare_order,
        "rare_training_images": conditions,
        "summary": summary,
    }
    OUTPUT.mkdir(exist_ok=True)
    path = OUTPUT / "split.json"
    if path.exists():
        raise SystemExit(f"{path} already exists; preserve the existing manifest")
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
