"""Load prepared Severstal images and aligned annotations from a fixed split."""

import json
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import Dataset

from projet.preprocessing.augmentation import augment


DATA_ROOT = Path(__file__).resolve().parents[2] / "donnees" / "severstal"


class SeverstalDataset(Dataset):
    """Return original-size grayscale images, xyxy boxes and four binary masks."""

    def __init__(self, split, *, root=DATA_ROOT, augment_training=False,
                 augmentation_settings=None, seed=0, rare_count=None):
        if split not in ("train", "validation", "test"):
            raise ValueError("Split must be train, validation or test")
        if augment_training and split != "train":
            raise ValueError("Augmentation is only available for training")
        if rare_count is not None and (split != "train" or rare_count not in (5, 10, 15)):
            raise ValueError("Rare subsets are available for training with 5, 10 or 15 examples")
        self.root = Path(root)
        manifest = json.loads((self.root / "split.json").read_text(encoding="utf-8"))
        self.ids = manifest["splits"][split] if rare_count is None else manifest["rare_training_images"][str(rare_count)]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Split contains duplicate image IDs")
        if rare_count is not None and not set(self.ids) <= set(manifest["splits"]["train"]):
            raise ValueError("Rare subset must contain only training images")
        self.augment_training = augment_training
        self.augmentation_settings = dict(augmentation_settings or {})
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.ids)

    def set_epoch(self, epoch):
        if not isinstance(epoch, int) or epoch < 0:
            raise ValueError("Epoch must be a nonnegative integer")
        self.epoch = epoch

    def __getitem__(self, index):
        image_id = self.ids[index]
        image = cv2.imread(str(self.root / "images" / image_id), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Cannot read image {image_id}")
        height, width = image.shape
        boxes, labels = [], []
        label_path = self.root / "labels" / f"{Path(image_id).stem}.txt"
        for line in label_path.read_text(encoding="utf-8").splitlines():
            class_id, x, y, w, h = map(float, line.split())
            if not class_id.is_integer() or int(class_id) not in range(4):
                raise ValueError(f"Invalid class in {label_path}")
            box = np.array([(x - w / 2) * width, (y - h / 2) * height,
                            (x + w / 2) * width, (y + h / 2) * height])
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height)
            if not np.isfinite(box).all() or box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError(f"Invalid box in {label_path}")
            boxes.append(box)
            labels.append(int(class_id))
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64)
        masks = np.zeros((4, height, width), dtype=np.uint8)
        for class_id in range(4):
            path = self.root / "masks" / f"class_{class_id + 1}" / f"{Path(image_id).stem}.png"
            if path.exists():
                mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if mask is None or mask.shape != image.shape:
                    raise ValueError(f"Invalid mask {path}")
                masks[class_id] = mask > 0
            if bool(np.any(masks[class_id])) != (class_id in labels):
                raise ValueError(f"Mask and boxes disagree for {image_id}, class {class_id + 1}")
        if self.augment_training:
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.epoch, index]))
            image, boxes, masks, kept = augment(image, boxes, masks, rng, **self.augmentation_settings)
            labels = labels[kept]
        return {"image_id": image_id, "image": image, "boxes": boxes,
                "labels": labels, "masks": masks}


def collate_samples(samples):
    """Preserve variable-sized annotations for a PyTorch DataLoader."""
    return samples
