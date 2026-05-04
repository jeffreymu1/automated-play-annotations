from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COURT_LENGTH_CM = 2800.0
COURT_WIDTH_CM = 1500.0

KEYPOINT_NAMES = (
    "corner_left_far",
    "corner_right_far",
    "corner_right_near",
    "corner_left_near",
    "half_court_far",
    "half_court_near",
)


@dataclass
class CameraCalibration:
    K: np.ndarray
    R: np.ndarray
    T: np.ndarray
    kc: np.ndarray
    width: int
    height: int

    @classmethod
    def from_json(cls, path: Path) -> "CameraCalibration":
        with open(path) as f:
            data = json.load(f)
        c = data["calibration"]
        return cls(
            K=np.array(c["KK"], dtype=float).reshape(3, 3),
            R=np.array(c["R"], dtype=float).reshape(3, 3),
            T=np.array(c["T"], dtype=float).reshape(3),
            kc=np.array(c["kc"], dtype=float).reshape(5),
            width=int(c["img_width"]),
            height=int(c["img_height"]),
        )


def project_world_to_image(points_w: np.ndarray, calib: CameraCalibration) -> np.ndarray:
    points_w = np.asarray(points_w, dtype=float).reshape(-1, 3)
    pc = points_w @ calib.R.T + calib.T
    Xc, Yc, Zc = pc[:, 0], pc[:, 1], pc[:, 2]
    valid = Zc > 1e-6
    x = np.where(valid, Xc / np.where(valid, Zc, 1.0), 0.0)
    y = np.where(valid, Yc / np.where(valid, Zc, 1.0), 0.0)
    k1, k2, p1, p2, k3 = calib.kc
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2**2 + k3 * r2**3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
    fx, fy = calib.K[0, 0], calib.K[1, 1]
    cx, cy = calib.K[0, 2], calib.K[1, 2]
    skew = calib.K[0, 1]
    u = fx * xd + skew * yd + cx
    v = fy * yd + cy
    uv = np.stack([u, v], axis=-1)
    uv[~valid] = np.nan
    return uv


def court_corners_world(
    length_cm: float = COURT_LENGTH_CM,
    width_cm: float = COURT_WIDTH_CM,
) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [length_cm, 0.0, 0.0],
            [length_cm, width_cm, 0.0],
            [0.0, width_cm, 0.0],
        ]
    )


def court_keypoints_world(
    length_cm: float = COURT_LENGTH_CM,
    width_cm: float = COURT_WIDTH_CM,
) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [length_cm, 0.0, 0.0],
            [length_cm, width_cm, 0.0],
            [0.0, width_cm, 0.0],
            [length_cm / 2, 0.0, 0.0],
            [length_cm / 2, width_cm, 0.0],
        ]
    )


def sample_segment(a: np.ndarray, b: np.ndarray, n: int = 400) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return a + t * (b - a)
