"""Create Ultralytics dataset files from the reviewed image-level manifest."""

import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
RAW_IMAGES = ROOT / "raw" / "train_images"
LABELS = ROOT / "processed" / "labels"
OUTPUT = ROOT.parent / "donnees" / "severstal"
WIDTH, HEIGHT = 1600, 256


def main():
    manifest = json.loads((ROOT / "manifests" / "split.json").read_text(encoding="utf-8"))
    splits = manifest["splits"]
    train = set(splits["train"])
    validation = set(splits["validation"])
    test = set(splits["test"])
    if train & validation or train & test or validation & test:
        raise SystemExit("The image splits overlap")
    images = train | validation | test
    if len(images) != sum(len(ids) for ids in splits.values()):
        raise SystemExit("An image occurs more than once in the split manifest")
    missing = [name for name in sorted(images) if not (RAW_IMAGES / name).is_file()
               or not (LABELS / (Path(name).stem + ".txt")).is_file()]
    if missing:
        raise SystemExit(f"Missing {len(missing)} source images or labels, first: {missing[0]}")
    if OUTPUT.exists():
        raise SystemExit(f"{OUTPUT} already exists; preserve existing outputs")

    variants = {"full": splits["train"]}
    for count in (5, 10, 15):
        subset = manifest["rare_training_images"][str(count)]
        if len(subset) != len(set(subset)) or not set(subset) <= train:
            raise SystemExit(f"Invalid training subset for {count} rare images")
        variants[f"rare_{count}"] = subset

    # Stage the entire layout first, so an interrupted run can be retried safely.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="severstal-", dir=OUTPUT.parent) as temporary:
        stage = Path(temporary)
        image_dir = stage / "images"
        label_dir = stage / "labels"
        image_dir.mkdir()
        label_dir.mkdir()
        for name in sorted(images):
            source_image = RAW_IMAGES / name
            source_label = LABELS / (Path(name).stem + ".txt")
            for source, destination in ((source_image, image_dir / name),
                                        (source_label, label_dir / source_label.name)):
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)

        # Save each nonempty class mask independently so overlaps are preserved.
        mask_dir = stage / "masks"
        for class_id in range(1, 5):
            (mask_dir / f"class_{class_id}").mkdir(parents=True)
        mask_count = 0
        with (ROOT / "raw" / "train.csv").open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                encoded = row["EncodedPixels"].strip()
                if not encoded:
                    continue
                image_id = row["ImageId"]
                class_id = int(row["ClassId"])
                if image_id not in images or class_id not in range(1, 5):
                    raise ValueError(f"Unexpected image or class in annotation: {image_id}, {class_id}")
                values = [int(value) for value in encoded.split()]
                if len(values) % 2:
                    raise ValueError(f"Invalid RLE for {image_id}, class {class_id}")
                mask = np.zeros(WIDTH * HEIGHT, dtype=np.uint8)
                last_end = 0
                for start, length in zip(values[::2], values[1::2]):
                    end = start + length - 1
                    if start < 1 or length < 1 or start <= last_end or end > mask.size:
                        raise ValueError(f"Invalid RLE for {image_id}, class {class_id}")
                    mask[start - 1:end] = 255
                    last_end = end
                mask_path = mask_dir / f"class_{class_id}" / f"{Path(image_id).stem}.png"
                if mask_path.exists():
                    raise ValueError(f"Duplicate mask for {image_id}, class {class_id}")
                Image.fromarray(mask.reshape((HEIGHT, WIDTH), order="F")).convert("1").save(mask_path)
                mask_count += 1
        shutil.copy2(ROOT / "raw" / "train.csv", stage / "annotations.csv")
        shutil.copy2(ROOT / "manifests" / "split.json", stage / "split.json")

        lists = {"validation": splits["validation"], "test": splits["test"], **variants}
        for name, members in lists.items():
            (stage / f"{name}.txt").write_text(
                "".join(f"./images/{image}\n" for image in members), encoding="utf-8"
            )
        for name in variants:
            config = {
                "path": str(OUTPUT.resolve()),
                "train": f"{name}.txt",
                "val": "validation.txt",
                "test": "test.txt",
                "names": {0: "class_1", 1: "class_2", 2: "class_3", 3: "class_4"},
            }
            # JSON is valid YAML and safely escapes Windows paths.
            (stage / f"{name}.yaml").write_text(
                json.dumps(config, indent=2) + "\n", encoding="utf-8"
            )
        stage.rename(OUTPUT)
    print(f"Created {len(images)} image/label pairs in {OUTPUT}")
    print(f"Created {mask_count} nonempty class masks")
    for name, members in variants.items():
        print(f"{name}: {len(members)} training images")
    print(f"Validation: {len(validation)} images; test: {len(test)} images")


if __name__ == "__main__":
    main()
