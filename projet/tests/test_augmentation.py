"""Check augmentation alignment and deterministic random seeds."""

import unittest

import numpy as np

from projet.preprocessing.augmentation import augment


class AugmentationFixtures(unittest.TestCase):
    def test_masks_and_boxes_follow_rotation(self):
        image = np.full((64, 128), 128, dtype=np.uint8)
        masks = np.zeros((2, 64, 128), dtype=np.uint8)
        masks[0, 20:40, 30:60] = 1
        boxes = [[30, 20, 60, 40]]
        result = augment(image, boxes, masks, np.random.default_rng(4))
        repeated = augment(image, boxes, masks, np.random.default_rng(4))
        for first, second in zip(result, repeated):
            np.testing.assert_array_equal(first, second)
        augmented, transformed_boxes, transformed_masks, kept = result
        self.assertEqual(augmented.shape, image.shape)
        self.assertEqual(kept.tolist(), [0])
        self.assertEqual(set(np.unique(transformed_masks)), {0, 1})
        y, x = np.nonzero(transformed_masks[0])
        x1, y1, x2, y2 = transformed_boxes[0]
        self.assertGreaterEqual(x.min(), np.floor(x1) - 1)
        self.assertLessEqual(x.max(), np.ceil(x2))
        self.assertGreaterEqual(y.min(), np.floor(y1) - 1)
        self.assertLessEqual(y.max(), np.ceil(y2))
        self.assertFalse(transformed_masks[1].any())

    def test_empty_boxes_and_invalid_masks(self):
        image = np.zeros((16, 32), dtype=np.uint8)
        masks = np.zeros((4, 16, 32), dtype=np.uint8)
        _, boxes, _, kept = augment(image, [], masks, np.random.default_rng(0))
        self.assertEqual(boxes.shape, (0, 4))
        self.assertEqual(len(kept), 0)
        masks[0, 0, 0] = 2
        with self.assertRaises(ValueError):
            augment(image, [], masks, np.random.default_rng(0))


if __name__ == "__main__":
    unittest.main()
