"""Image and geometry augmentation helpers for court-detection datasets."""

from __future__ import annotations

import random
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CropTransform:
    x0: int
    y0: int
    width: int
    height: int
    out_width: int
    out_height: int


@dataclass(frozen=True)
class ScoreBarConfig:
    min_y_size: float = 0.25
    max_y_size: float = 0.50
    min_y_top: float = 0.50
    max_x_left: float = 0.25


def random_crop_transform(
    image_shape: tuple[int, int],
    output_size: tuple[int, int],
    augment: bool,
) -> CropTransform:
    in_h, in_w = image_shape
    out_h, out_w = output_size

    if augment:
        target_aspect = out_w / out_h
        crop_h = random.randint(max(2, int(0.72 * in_h)), in_h)
        crop_w = int(round(crop_h * target_aspect))
        if crop_w > in_w:
            crop_w = in_w
            crop_h = min(in_h, int(round(crop_w / target_aspect)))
        x0 = random.randint(0, max(0, in_w - crop_w))
        y0 = random.randint(0, max(0, in_h - crop_h))
    else:
        x0, y0 = 0, 0
        crop_w, crop_h = in_w, in_h

    return CropTransform(
        x0=x0,
        y0=y0,
        width=crop_w,
        height=crop_h,
        out_width=out_w,
        out_height=out_h,
    )


def crop_resize_image(image: np.ndarray, transform: CropTransform) -> np.ndarray:
    cropped = image[
        transform.y0:transform.y0 + transform.height,
        transform.x0:transform.x0 + transform.width,
    ]
    interpolation = (
        cv2.INTER_AREA
        if transform.out_width < transform.width or transform.out_height < transform.height
        else cv2.INTER_LINEAR
    )
    resized = cv2.resize(cropped, (transform.out_width, transform.out_height), interpolation=interpolation)
    return resized.clip(0.0, 1.0).astype(np.float32)


def crop_resize_mask(mask: np.ndarray, transform: CropTransform) -> np.ndarray:
    cropped = mask[
        transform.y0:transform.y0 + transform.height,
        transform.x0:transform.x0 + transform.width,
    ].astype(np.uint8)
    resized = cv2.resize(
        cropped,
        (transform.out_width, transform.out_height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def transform_line_uv(lines_uv: dict[str, np.ndarray], transform: CropTransform) -> dict[str, np.ndarray]:
    scale_x = transform.out_width / transform.width
    scale_y = transform.out_height / transform.height
    transformed: dict[str, np.ndarray] = {}
    for name, uv in lines_uv.items():
        new_uv = uv.copy()
        new_uv[:, 0] = (uv[:, 0] - transform.x0) * scale_x
        new_uv[:, 1] = (uv[:, 1] - transform.y0) * scale_y
        transformed[name] = new_uv
    return transformed


def augment_color(image: np.ndarray) -> np.ndarray:
    img = image.copy()
    img = img * random.uniform(0.80, 1.20) + random.uniform(-0.08, 0.08)
    mean = img.mean(axis=(0, 1), keepdims=True)
    img = (img - mean) * random.uniform(0.75, 1.25) + mean
    gray = np.dot(img, np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
    img = gray + (img - gray) * random.uniform(0.75, 1.25)
    if random.random() < 0.15:
        img = cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.2, 1.0))
    return img.clip(0.0, 1.0).astype(np.float32)


def apply_score_bar(
    image: np.ndarray,
    visible_mask: np.ndarray,
    config: ScoreBarConfig = ScoreBarConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    y_size = random.uniform(config.min_y_size, config.max_y_size)
    y_top_max = max(config.min_y_top, 1.0 - y_size)
    y0_frac = random.uniform(config.min_y_top, y_top_max)
    y1_frac = y0_frac + y_size

    x0_frac = random.uniform(0.0, config.max_x_left)
    x1_frac = 1.0 - x0_frac

    x0 = int(np.floor(np.clip(x0_frac, 0.0, 1.0) * w))
    x1 = int(np.ceil(np.clip(x1_frac, 0.0, 1.0) * w))
    y0 = int(np.floor(np.clip(y0_frac, 0.0, 1.0) * h))
    y1 = int(np.ceil(np.clip(y1_frac, 0.0, 1.0) * h))

    x0 = max(0, min(x0, w - 1))
    x1 = max(x0 + 1, min(x1, w))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))

    out_image = image.copy()
    out_mask = visible_mask.copy()
    color = np.array([random.random(), random.random(), random.random()], dtype=np.float32)
    out_image[y0:y1, x0:x1] = color
    out_mask[y0:y1, x0:x1] = False
    return out_image.astype(np.float32), out_mask
