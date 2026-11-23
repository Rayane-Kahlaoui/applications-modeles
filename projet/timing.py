"""Measure end-to-end single-image inference on a caller-supplied pipeline."""

import time

import numpy as np


def benchmark_inference(predict, samples, *, warmup=3):
    """Time predict(sample), including everything the supplied callable performs."""
    samples = list(samples)
    if not samples or not isinstance(warmup, int) or warmup < 0:
        raise ValueError("Provide samples and a nonnegative warmup count")
    for index in range(warmup):
        predict(samples[index % len(samples)])
    elapsed = []
    for sample in samples:
        start = time.perf_counter()
        predict(sample)
        elapsed.append(time.perf_counter() - start)
    return {"images": len(samples), "warmup": warmup,
            "mean_seconds": float(np.mean(elapsed)),
            "median_seconds": float(np.median(elapsed)),
            "p95_seconds": float(np.percentile(elapsed, 95)),
            "max_seconds": max(elapsed), "per_image_seconds": elapsed}
