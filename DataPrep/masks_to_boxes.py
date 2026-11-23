"""Convert class masks to YOLO boxes using 8-connected components."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1600, 256


def boxes_from_rle(encoded):
    values = np.asarray([int(value) for value in encoded.split()], dtype=np.int64)
    if len(values) % 2:
        raise ValueError("RLE must contain start/length pairs")
    flat = np.zeros(WIDTH * HEIGHT, dtype=np.uint8)
    last_end = 0
    for start, length in zip(values[::2], values[1::2]):
        if start < 1 or length < 1 or start <= last_end:
            raise ValueError("invalid or overlapping RLE run")
        last_end = start + length - 1
        if last_end > flat.size:
            raise ValueError("RLE exceeds image dimensions")
        flat[start - 1:last_end] = 1
    mask = flat.reshape((HEIGHT, WIDTH), order="F")
    labels, _ = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    return [(region[1].start, region[0].start, region[1].stop, region[0].stop)
            for region in ndimage.find_objects(labels) if region is not None]


def main():
    manifest = json.loads((ROOT / "manifests" / "split.json").read_text(encoding="utf-8"))
    all_images = {image for ids in manifest["splits"].values() for image in ids}
    annotations = defaultdict(list)
    with (ROOT / "raw" / "train.csv").open(newline="", encoding="utf-8-sig") as file:
        for row in csv.DictReader(file):
            if row["EncodedPixels"].strip():
                annotations[row["ImageId"]].append((int(row["ClassId"]), row["EncodedPixels"]))

    output = ROOT / "processed" / "labels"
    if output.exists():
        raise SystemExit(f"{output} already exists; preserve existing outputs")
    output.mkdir(parents=True)
    for image in sorted(all_images):
        lines = []
        for class_id, encoded in sorted(annotations[image]):
            for x0, y0, x1, y1 in boxes_from_rle(encoded):
                xc = (x0 + x1) / (2 * WIDTH)
                yc = (y0 + y1) / (2 * HEIGHT)
                width = (x1 - x0) / WIDTH
                height = (y1 - y0) / HEIGHT
                lines.append(f"{class_id - 1} {xc:.8f} {yc:.8f} {width:.8f} {height:.8f}")
        (output / f"{Path(image).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
    print(f"Wrote labels for {len(all_images)} images to {output}")


if __name__ == "__main__":
    main()
