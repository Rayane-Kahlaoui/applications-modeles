"""Seeded training-image augmentation with aligned boxes and class masks."""

import cv2
import numpy as np


def augment(image, boxes, masks, rng, *, rotation_degrees=5.0,
            brightness_range=(0.8, 1.2), noise_std=5.0, blur_sigma=0.0):
    """Transform a grayscale image, xyxy boxes and binary (C,H,W) masks together.

    Returns the transformed image, boxes, masks and indices of boxes that remain
    at least partly inside the image. The caller must use these indices to keep
    class labels aligned. Each call samples one synthetic training condition.
    """
    image = np.asarray(image)
    masks = np.asarray(masks)
    boxes = np.asarray(boxes, dtype=float).reshape(-1, 4)
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("Image must be grayscale uint8")
    height, width = image.shape
    if masks.ndim != 3 or not masks.shape[0] or masks.shape[1:] != image.shape or not np.isin(masks, (0, 1)).all():
        raise ValueError("Masks must be binary with shape (classes, height, width)")
    if (not np.isfinite(boxes).all() or np.any(boxes[:, 0] < 0) or np.any(boxes[:, 1] < 0)
            or np.any(boxes[:, 2] > width) or np.any(boxes[:, 3] > height)
            or np.any(boxes[:, 0] >= boxes[:, 2]) or np.any(boxes[:, 1] >= boxes[:, 3])):
        raise ValueError("Boxes must be finite and within image bounds")
    if (not np.isfinite(rotation_degrees) or rotation_degrees < 0
            or not np.isfinite(noise_std) or noise_std < 0
            or not np.isfinite(blur_sigma) or blur_sigma < 0
            or len(brightness_range) != 2 or not np.isfinite(brightness_range).all()
            or brightness_range[0] <= 0 or brightness_range[0] > brightness_range[1]):
        raise ValueError("Invalid augmentation ranges")

    angle = float(rng.uniform(-rotation_degrees, rotation_degrees))
    matrix = cv2.getRotationMatrix2D(((width - 1) / 2, (height - 1) / 2), angle, 1.0)
    transformed = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=int(np.median(image)))
    transformed_masks = np.stack([
        cv2.warpAffine(mask.astype(np.uint8), matrix, (width, height),
                       flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        for mask in masks
    ])
    if blur_sigma:
        transformed = cv2.GaussianBlur(transformed, (0, 0), blur_sigma)
    brightness = float(rng.uniform(*brightness_range))
    noise = rng.normal(0, noise_std, image.shape)
    transformed = np.clip(transformed.astype(np.float32) * brightness + noise, 0, 255).astype(np.uint8)

    corners = np.stack((boxes[:, [0, 1]], boxes[:, [2, 1]],
                        boxes[:, [2, 3]], boxes[:, [0, 3]]), axis=1)
    rotated = corners @ matrix[:, :2].T + matrix[:, 2]
    lower = rotated.min(axis=1)
    upper = rotated.max(axis=1)
    lower = np.maximum(lower, 0)
    upper = np.minimum(upper, [width, height])
    keep = (lower[:, 0] < upper[:, 0]) & (lower[:, 1] < upper[:, 1])
    result_boxes = np.concatenate((lower[keep], upper[keep]), axis=1)
    return transformed, result_boxes, transformed_masks, np.flatnonzero(keep)
