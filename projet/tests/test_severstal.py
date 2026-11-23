"""Check split selection and aligned augmentation on a tiny dataset."""

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from torch.utils.data import DataLoader

from projet.preprocessing.severstal import SeverstalDataset, collate_samples


class SeverstalLoaderTest(unittest.TestCase):
    def test_loader_and_training_augmentation(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "masks" / "class_2").mkdir(parents=True)
            manifest = {"splits": {"train": ["defect.png", "clean.png"],
                                    "validation": ["valid.png"], "test": []},
                        "rare_training_images": {"5": ["defect.png"]}}
            (root / "split.json").write_text(json.dumps(manifest), encoding="utf-8")
            for name in ("defect", "clean", "valid"):
                cv2.imwrite(str(root / "images" / f"{name}.png"), np.full((32, 64), 128, np.uint8))
                text = "1 0.5 0.5 0.25 0.5\n" if name == "defect" else ""
                (root / "labels" / f"{name}.txt").write_text(text, encoding="utf-8")
            mask = np.zeros((32, 64), np.uint8)
            mask[8:24, 24:40] = 255
            cv2.imwrite(str(root / "masks" / "class_2" / "defect.png"), mask)
            data = SeverstalDataset("train", root=root, augment_training=True, seed=3)
            first = data[0]
            np.testing.assert_array_equal(first["image"], data[0]["image"])
            self.assertEqual(first["labels"].tolist(), [1])
            self.assertEqual(first["masks"].shape, (4, 32, 64))
            self.assertEqual(data[1]["boxes"].shape, (0, 4))
            data.set_epoch(1)
            self.assertFalse(np.array_equal(first["image"], data[0]["image"]))
            self.assertEqual(len(next(iter(DataLoader(data, batch_size=2, collate_fn=collate_samples)))), 2)
            self.assertEqual(len(SeverstalDataset("train", root=root, rare_count=5)), 1)
            with self.assertRaises(ValueError):
                SeverstalDataset("validation", root=root, augment_training=True)
