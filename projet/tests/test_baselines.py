"""Check binary baseline features and cost-based threshold selection."""

import unittest

import numpy as np
import torch

from projet.training.baselines import choose_threshold, extract_features
from projet.training.region_baseline import crop_features, propose
from projet.training.cnn_classifier import CustomCNN, ImageSamples


class BaselineTest(unittest.TestCase):
    def test_features_and_threshold(self):
        image = np.full((256, 1600), 128, dtype=np.uint8)
        for method in ("hog-sgd-svm", "texture-forest", "texture-multiscale"):
            features = extract_features(image, method)
            self.assertEqual(features.ndim, 1)
            self.assertTrue(np.isfinite(features).all())
        labels = np.array([0, 1, 1])
        scores = np.array([0.9, 0.8, 0.2])
        threshold = choose_threshold(labels, scores)
        self.assertEqual(((scores >= threshold) & (labels == 1)).sum(), 2)
        self.assertEqual(((scores >= threshold) & (labels == 0)).sum(), 1)

    def test_region_features_and_proposal_limit(self):
        image = np.full((128, 256), 128, dtype=np.uint8)
        image[30:50, 30:50] = 255
        image[70:90, 180:200] = 0
        self.assertTrue(np.isfinite(crop_features(image, [30, 30, 50, 50])).all())
        boxes = propose(image, cutoff=25, min_area=10, max_regions=1)
        self.assertEqual(len(boxes), 1)

    def test_multiscale_region_proposals_and_texture_features(self):
        image = np.full((128, 256), 128, dtype=np.uint8)
        image[30:50, 30:50] = 255
        image[70:90, 180:200] = 0
        boxes = propose(image, cutoff=25, min_area=10, max_regions=3, mode="multiscale")
        self.assertGreater(len(boxes), 0)
        self.assertLessEqual(len(boxes), 3)
        self.assertEqual(len(boxes), len({tuple(box) for box in boxes}))
        texture = crop_features(image, boxes[0], feature_set="texture")
        self.assertGreater(len(texture), len(crop_features(image, boxes[0])))
        self.assertTrue(np.isfinite(texture).all())

    def test_custom_cnn_and_seeded_augmentation(self):
        images = np.full((2, 64, 256), 128, dtype=np.uint8)
        masks = np.zeros_like(images)
        masks[1, 20:30, 120:130] = 1
        samples = ImageSamples(images, masks, np.array([0, 1]), np.array([0, 1]), augment=True, seed=2)
        image, label = samples[1]
        self.assertEqual(label, 1)
        self.assertTrue(torch.equal(image, samples[1][0]))
        samples.epoch = 1
        self.assertFalse(torch.equal(image, samples[1][0]))
        self.assertEqual(CustomCNN()(torch.stack([image, samples[0][0]])).shape, (2,))
