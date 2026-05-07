from __future__ import annotations

import random

from court_detection.augmentation import random_crop_transform


def test_random_crop_transform_can_sample_aggressive_edge_zoom(monkeypatch) -> None:
    random_values = iter([0.0, 0.0, 0.25])
    monkeypatch.setattr(random, "random", lambda: next(random_values))
    monkeypatch.setattr(random, "randint", lambda low, high: low)

    crop = random_crop_transform((1000, 1600), (384, 640), augment=True)

    assert crop.height == 420
    assert crop.width == 700
    assert crop.x0 == 0
    assert crop.y0 == 0


def test_random_crop_transform_keeps_full_frame_without_augmentation() -> None:
    crop = random_crop_transform((1000, 1600), (384, 640), augment=False)

    assert crop.height == 1000
    assert crop.width == 1600
    assert crop.x0 == 0
    assert crop.y0 == 0
