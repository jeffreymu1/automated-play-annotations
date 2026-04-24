"""Basketball court geometry, camera model, and world->image projection.

Centralises the *physical* assumptions the rest of the codebase builds
on top of:

    * FIBA court dimensions (28 m x 15 m) as centimetre constants.
    * The 6 supervised keypoints (4 corners + 2 half-court endpoints)
      with their canonical ordering and world coordinates.
    * :class:`CameraCalibration` -- intrinsics, extrinsics, and lens
      distortion loaded from a DeepSport calibration JSON.
    * :func:`project_world_to_image` -- the full forward camera model
      (extrinsics -> perspective divide -> radial/tangential distortion
      -> intrinsics), returning NaN for points behind the camera.
    * :func:`sample_segment` -- world-space line densification so that
      lens-distorted curves can be drawn correctly in the image.

World convention (deepsport-dataset-info.md):
    origin at furthest left court corner, +X along court length,
    +Y along court width, +Z pointing DOWN, units = centimetres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COURT_LENGTH_CM = 2800.0  # FIBA: 28 m
COURT_WIDTH_CM = 1500.0   # FIBA: 15 m

# With +X along length and +Y along width, X=0 is the "left" baseline,
# X=L is "right"; Y=0 is the "far" sideline, Y=W is "near".
KEYPOINT_NAMES = (
    "corner_left_far",    # (0,   0)
    "corner_right_far",   # (L,   0)
    "corner_right_near",  # (L,   W)
    "corner_left_near",   # (0,   W)
    "half_court_far",     # (L/2, 0)
    "half_court_near",    # (L/2, W)
)


@dataclass
class CameraCalibration:
    K: np.ndarray      # (3, 3)
    R: np.ndarray      # (3, 3)
    T: np.ndarray      # (3,)
    kc: np.ndarray     # (5,) -> k1, k2, p1, p2, k3
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
    """Project (N, 3) world points to (N, 2) image pixel coordinates.

    Points whose camera-space Z is non-positive are returned as NaN.
    """
    points_w = np.asarray(points_w, dtype=float).reshape(-1, 3)

    # Extrinsics: world -> camera
    pc = points_w @ calib.R.T + calib.T  # (N, 3)
    Xc, Yc, Zc = pc[:, 0], pc[:, 1], pc[:, 2]

    # Perspective divide (guard Zc)
    valid = Zc > 1e-6
    x = np.where(valid, Xc / np.where(valid, Zc, 1.0), 0.0)
    y = np.where(valid, Yc / np.where(valid, Zc, 1.0), 0.0)

    # Distortion: kc = [k1, k2, p1, p2, k3]
    k1, k2, p1, p2, k3 = calib.kc
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * radial + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
    yd = y * radial + p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y

    # Intrinsics
    fx, fy = calib.K[0, 0], calib.K[1, 1]
    cx, cy = calib.K[0, 2], calib.K[1, 2]
    skew = calib.K[0, 1]
    u = fx * xd + skew * yd + cx
    v = fy * yd + cy

    uv = np.stack([u, v], axis=-1)
    uv[~valid] = np.nan
    return uv


def court_corners_world(length_cm: float = COURT_LENGTH_CM,
                        width_cm: float = COURT_WIDTH_CM) -> np.ndarray:
    """Four court corners as a (4, 3) world-coordinate array (cm).

    Order: left_far, right_far, right_near, left_near (matches the
    first four entries of :data:`KEYPOINT_NAMES`).
    """
    return np.array([
        [0.0,        0.0,       0.0],
        [length_cm,  0.0,       0.0],
        [length_cm,  width_cm,  0.0],
        [0.0,        width_cm,  0.0],
    ])


def court_keypoints_world(length_cm: float = COURT_LENGTH_CM,
                          width_cm: float = COURT_WIDTH_CM) -> np.ndarray:
    """The 6 supervised keypoints as a (6, 3) array, in ``KEYPOINT_NAMES`` order."""
    return np.array([
        [0.0,           0.0,      0.0],  # corner_left_far
        [length_cm,     0.0,      0.0],  # corner_right_far
        [length_cm,     width_cm, 0.0],  # corner_right_near
        [0.0,           width_cm, 0.0],  # corner_left_near
        [length_cm / 2, 0.0,      0.0],  # half_court_far
        [length_cm / 2, width_cm, 0.0],  # half_court_near
    ])


def sample_segment(a: np.ndarray, b: np.ndarray, n: int = 400) -> np.ndarray:
    """Return n world-space points evenly spaced on the segment from a to b."""
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return a + t * (b - a)
