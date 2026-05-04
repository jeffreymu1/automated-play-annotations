from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class CameraCalibration:
    K: np.ndarray
    R: np.ndarray
    T: np.ndarray
    kc: np.ndarray
    width: int
    height: int

    @classmethod
    def from_json(cls, path: Path) -> CameraCalibration:
        with open(path) as f:
            data = json.load(f)
        c = data["calibration"]
        return cls(
            K=np.array(c["KK"], dtype=np.float64).reshape(3, 3),
            R=np.array(c["R"], dtype=np.float64).reshape(3, 3),
            T=np.array(c["T"], dtype=np.float64).reshape(3),
            kc=np.array(c["kc"], dtype=np.float64).reshape(5),
            width=int(c["img_width"]),
            height=int(c["img_height"]),
        )


def project_world_to_image(points_w: np.ndarray, calib: CameraCalibration) -> np.ndarray:
    points_w = np.asarray(points_w, dtype=np.float64).reshape(-1, 3)
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


def clip_xyxy(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    w: int,
    h: int,
) -> tuple[float, float, float, float]:
    x1 = float(np.clip(x1, 0, w - 1))
    x2 = float(np.clip(x2, 0, w - 1))
    y1 = float(np.clip(y1, 0, h - 1))
    y2 = float(np.clip(y2, 0, h - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2
