"""Basketball court geometry, camera model, and world->image projection."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

COURT_LENGTH_CM = 2800.0
COURT_WIDTH_CM = 1500.0
LINE_WIDTH_CM = 5.0
LANE_HALF_WIDTH_CM = 245.0
FREE_THROW_LINE_X_CM = 580.0
FREE_THROW_LINE_LENGTH_CM = 360.0
FREE_THROW_CIRCLE_RADIUS_CM = 180.0
THREE_POINT_RADIUS_CM = 675.0
THREE_POINT_SIDE_OFFSET_CM = 90.0
BASKET_CENTER_FROM_ENDLINE_CM = 157.5

KEYPOINT_NAMES = (
    "corner_left_far",
    "corner_right_far",
    "corner_right_near",
    "corner_left_near",
    "half_court_far",
    "half_court_near",
)

STRAIGHT_LINE_NAMES = (
    "sideline_far",
    "sideline_near",
    "baseline_left",
    "baseline_right",
    "halfcourt",
    "lane_left_far",
    "lane_left_near",
    "foul_left",
    "lane_right_far",
    "lane_right_near",
    "foul_right",
)

CURVE_LINE_NAMES = (
    "three_point_arc_left",
    "three_point_arc_right",
    "free_throw_circle_left",
    "free_throw_circle_right",
)

LINE_NAMES = STRAIGHT_LINE_NAMES + CURVE_LINE_NAMES

MARKING_CLASS_NAMES = (
    "sideline_far",
    "sideline_near",
    "baseline",
    "halfcourt",
    "lane_far",
    "lane_near",
    "foul",
    "three_point_arc",
    "free_throw_circle",
)

MARKING_CLASS_BY_GEOMETRY = {
    "sideline_far": "sideline_far",
    "sideline_near": "sideline_near",
    "baseline_left": "baseline",
    "baseline_right": "baseline",
    "halfcourt": "halfcourt",
    "lane_left_far": "lane_far",
    "lane_right_far": "lane_far",
    "lane_left_near": "lane_near",
    "lane_right_near": "lane_near",
    "foul_left": "foul",
    "foul_right": "foul",
    "three_point_arc_left": "three_point_arc",
    "three_point_arc_right": "three_point_arc",
    "free_throw_circle_left": "free_throw_circle",
    "free_throw_circle_right": "free_throw_circle",
}


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
    xc, yc, zc = pc[:, 0], pc[:, 1], pc[:, 2]

    valid = zc > 1e-6
    x = np.where(valid, xc / np.where(valid, zc, 1.0), 0.0)
    y = np.where(valid, yc / np.where(valid, zc, 1.0), 0.0)

    k1, k2, p1, p2, k3 = calib.kc
    r2 = x * x + y * y
    radial = 1.0 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
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
    return np.array([
        [0.0, 0.0, 0.0],
        [length_cm, 0.0, 0.0],
        [length_cm, width_cm, 0.0],
        [0.0, width_cm, 0.0],
    ])


def court_keypoints_world(
    length_cm: float = COURT_LENGTH_CM,
    width_cm: float = COURT_WIDTH_CM,
) -> np.ndarray:
    return np.array([
        [0.0, 0.0, 0.0],
        [length_cm, 0.0, 0.0],
        [length_cm, width_cm, 0.0],
        [0.0, width_cm, 0.0],
        [length_cm / 2, 0.0, 0.0],
        [length_cm / 2, width_cm, 0.0],
    ])


def sample_segment(a: np.ndarray, b: np.ndarray, n: int = 400) -> np.ndarray:
    t = np.linspace(0.0, 1.0, n).reshape(-1, 1)
    return a + t * (b - a)


def sample_circle_arc(
    center: np.ndarray,
    radius: float,
    start_angle_rad: float,
    end_angle_rad: float,
    n: int = 400,
) -> np.ndarray:
    angles = np.linspace(start_angle_rad, end_angle_rad, n)
    xy = np.column_stack([
        center[0] + radius * np.cos(angles),
        center[1] + radius * np.sin(angles),
    ])
    return np.column_stack([xy, np.zeros(len(xy), dtype=float)])


def court_lines_world(
    length_cm: float = COURT_LENGTH_CM,
    width_cm: float = COURT_WIDTH_CM,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Endpoints (a, b) in world coordinates for each named court line."""
    mid_y = width_cm / 2.0
    lane_far_y = mid_y - LANE_HALF_WIDTH_CM
    lane_near_y = mid_y + LANE_HALF_WIDTH_CM
    free_throw_left_x = FREE_THROW_LINE_X_CM
    free_throw_right_x = length_cm - FREE_THROW_LINE_X_CM
    foul_y0 = mid_y - FREE_THROW_LINE_LENGTH_CM / 2.0
    foul_y1 = mid_y + FREE_THROW_LINE_LENGTH_CM / 2.0
    return {
        "sideline_far": (np.array([0.0, 0.0, 0.0]), np.array([length_cm, 0.0, 0.0])),
        "sideline_near": (np.array([0.0, width_cm, 0.0]), np.array([length_cm, width_cm, 0.0])),
        "baseline_left": (np.array([0.0, 0.0, 0.0]), np.array([0.0, width_cm, 0.0])),
        "baseline_right": (np.array([length_cm, 0.0, 0.0]), np.array([length_cm, width_cm, 0.0])),
        "halfcourt": (np.array([length_cm / 2, 0.0, 0.0]), np.array([length_cm / 2, width_cm, 0.0])),
        "lane_left_far": (np.array([0.0, lane_far_y, 0.0]), np.array([free_throw_left_x, lane_far_y, 0.0])),
        "lane_left_near": (np.array([0.0, lane_near_y, 0.0]), np.array([free_throw_left_x, lane_near_y, 0.0])),
        "foul_left": (np.array([free_throw_left_x, foul_y0, 0.0]), np.array([free_throw_left_x, foul_y1, 0.0])),
        "lane_right_far": (np.array([length_cm, lane_far_y, 0.0]), np.array([free_throw_right_x, lane_far_y, 0.0])),
        "lane_right_near": (np.array([length_cm, lane_near_y, 0.0]), np.array([free_throw_right_x, lane_near_y, 0.0])),
        "foul_right": (np.array([free_throw_right_x, foul_y0, 0.0]), np.array([free_throw_right_x, foul_y1, 0.0])),
    }


def court_curves_world(
    length_cm: float = COURT_LENGTH_CM,
    width_cm: float = COURT_WIDTH_CM,
    n: int = 400,
) -> dict[str, np.ndarray]:
    """Sampled world-coordinate polylines for curved FIBA court markings."""
    mid_y = width_cm / 2.0
    basket_left = np.array([BASKET_CENTER_FROM_ENDLINE_CM, mid_y], dtype=float)
    basket_right = np.array([length_cm - BASKET_CENTER_FROM_ENDLINE_CM, mid_y], dtype=float)
    free_throw_left = np.array([FREE_THROW_LINE_X_CM, mid_y], dtype=float)
    free_throw_right = np.array([length_cm - FREE_THROW_LINE_X_CM, mid_y], dtype=float)
    dy = mid_y - THREE_POINT_SIDE_OFFSET_CM
    theta = float(np.arcsin(np.clip(dy / THREE_POINT_RADIUS_CM, -1.0, 1.0)))

    return {
        "three_point_arc_left": sample_circle_arc(
            basket_left, THREE_POINT_RADIUS_CM, -theta, theta, n=n
        ),
        "three_point_arc_right": sample_circle_arc(
            basket_right, THREE_POINT_RADIUS_CM, np.pi - theta, np.pi + theta, n=n
        ),
        "free_throw_circle_left": sample_circle_arc(
            free_throw_left, FREE_THROW_CIRCLE_RADIUS_CM, 0.0, 2.0 * np.pi, n=n
        ),
        "free_throw_circle_right": sample_circle_arc(
            free_throw_right, FREE_THROW_CIRCLE_RADIUS_CM, 0.0, 2.0 * np.pi, n=n
        ),
    }


def court_markings_world(
    length_cm: float = COURT_LENGTH_CM,
    width_cm: float = COURT_WIDTH_CM,
    n: int = 400,
) -> dict[str, np.ndarray]:
    """Sampled world-coordinate polylines for every learning-target marking."""
    markings = {
        name: sample_segment(a, b, n=n)
        for name, (a, b) in court_lines_world(length_cm, width_cm).items()
    }
    markings.update(court_curves_world(length_cm, width_cm, n=n))
    return markings
