"""Check the original Severstal training files before creating a split."""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image


RAW = Path(__file__).resolve().parent / "raw"
IMAGE_DIR = RAW / "train_images"
ANNOTATIONS = RAW / "train.csv"
EXPECTED_SIZE = (1600, 256)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    images = {path.name: path for path in IMAGE_DIR.iterdir() if path.is_file()}
    if not images:
        raise SystemExit(f"No images found in {IMAGE_DIR}")

    classes_by_image = defaultdict(set)
    seen_pairs = set()
    errors = []
    rows = Counter()
    with ANNOTATIONS.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not {"ImageId", "ClassId", "EncodedPixels"}.issubset(reader.fieldnames or []):
            raise SystemExit("Unexpected CSV columns; expected ImageId, ClassId, EncodedPixels")
        for line, row in enumerate(reader, start=2):
            image_id = row["ImageId"]
            try:
                class_id = int(row["ClassId"])
                if class_id not in range(1, 5):
                    raise ValueError("class ID outside 1–4")
                if image_id not in images:
                    raise ValueError("image is missing")
                pair = (image_id, class_id)
                if pair in seen_pairs:
                    raise ValueError("duplicate image/class row")
                seen_pairs.add(pair)
                encoded = (row["EncodedPixels"] or "").strip()
                if not encoded:
                    rows["empty_masks"] += 1
                    continue
                values = [int(value) for value in encoded.split()]
                if len(values) % 2:
                    raise ValueError("RLE has an odd number of values")
                last_end = 0
                for start, length in zip(values[::2], values[1::2]):
                    if start < 1 or length < 1 or start <= last_end:
                        raise ValueError("invalid or overlapping RLE run")
                    last_end = start + length - 1
                    if last_end > EXPECTED_SIZE[0] * EXPECTED_SIZE[1]:
                        raise ValueError("RLE run exceeds image size")
                classes_by_image[image_id].add(class_id)
                rows["nonempty_masks"] += 1
            except (ValueError, TypeError) as exc:
                errors.append(f"CSV line {line}: {exc}")

    hashes = defaultdict(list)
    pixel_hashes = defaultdict(list)
    sizes = Counter()
    modes = Counter()
    for image_id, path in sorted(images.items()):
        try:
            hashes[sha256(path)].append(image_id)
            with Image.open(path) as image:
                sizes[str(image.size)] += 1
                modes[image.mode] += 1
                image.load()
                pixel_hashes[hashlib.sha256(image.tobytes()).hexdigest()].append(image_id)
                if image.size != EXPECTED_SIZE:
                    errors.append(f"{image_id}: unexpected size {image.size}")
        except (OSError, ValueError) as exc:
            errors.append(f"{image_id}: {exc}")

    duplicates = [names for names in hashes.values() if len(names) > 1]
    class_counts = Counter(class_id for values in classes_by_image.values() for class_id in values)
    report = {
        "annotation_sha256": sha256(ANNOTATIONS),
        "training_images": len(images),
        "image_sizes": dict(sizes),
        "image_modes": dict(modes),
        "annotation_rows": dict(rows),
        "images_by_class": dict(sorted(class_counts.items())),
        "images_with_no_annotated_defects": len(images) - len(classes_by_image),
        "images_with_multiple_classes": sum(len(values) > 1 for values in classes_by_image.values()),
        "identical_file_groups": duplicates,
        "identical_pixel_groups": [names for names in pixel_hashes.values() if len(names) > 1],
        "errors": errors,
    }
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
