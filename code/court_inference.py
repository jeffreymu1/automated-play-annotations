"""Infer quadrilateral corners of the playable court in pixel space.

Used as source points for a planar homography to field coordinates (see :mod:`field`).

Modes:
    * ``default`` — margins-based rectangle (fallback).
    * ``auto`` — HSV segmentation + largest region + minimum area quadrilateral heuristics.
    * ``manual`` — four user-provided ``(x, y)`` pixels in order TL, TR, BR, BL.
    * ``calib`` — project FIBA court corners with a DeepSport-style calibration JSON.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .calib import CameraCalibration, project_world_to_image
from .config import CourtMode

# Full FIBA court in cm: left-far, right-far, right-near, left-near (Z=0).
# Maps to field (0,0), (28,0), (28,15), (0,15) metres in :func:`field.build_homography_from_corners`.
COURT_WORLD_CORNERS_CM = np.array(
    [
        [0.0, 0.0, 0.0],
        [2800.0, 0.0, 0.0],
        [2800.0, 1500.0, 0.0],
        [0.0, 1500.0, 0.0],
    ],
    dtype=np.float64,
)


def order_corners_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    """Order four image points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    ys = pts[np.argsort(pts[:, 1])]
    top = ys[:2][np.argsort(ys[:2, 0])]
    bot = ys[2:][np.argsort(ys[2:, 0])]
    tl, tr = top[0], top[1]
    bl, br = bot[0], bot[1]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def default_margin_corners(width: int, height: int) -> list[tuple[float, float]]:
    margin_x = max(20, int(0.08 * width))
    margin_y = max(20, int(0.1 * height))
    return [
        (margin_x, margin_y),
        (width - margin_x, margin_y),
        (width - margin_x, height - margin_y),
        (margin_x, height - margin_y),
    ]


def corners_from_calibration_json(json_path: Path, frame_wh: tuple[int, int]) -> np.ndarray | None:
    """Return (4,2) float32 TL..BL image corners from calibration JSON, or None if invalid."""
    calib = CameraCalibration.from_json(json_path)
    w, h = frame_wh
    if calib.width != w or calib.height != h:
        # tolerate small tolerance for resized frames — still try
        pass
    uv = project_world_to_image(COURT_WORLD_CORNERS_CM, calib)
    if not np.isfinite(uv).all():
        return None
    out = []
    for row in uv:
        x, y = float(row[0]), float(row[1])
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        if not (-0.5 * w <= x < 1.5 * w and -0.5 * h <= y < 1.5 * h):
            return None
        out.append([x, y])
    # Keep row order = world TL,TR,BR,BL for getPerspectiveTransform dst mapping.
    return np.array(out, dtype=np.float32)


def detect_court_quad_auto(bgr: np.ndarray, scale: float = 0.5) -> np.ndarray | None:
    """Segment court-like colors and fit a quadrilateral. Returns (4,2) in full image coords."""
    h0, w0 = bgr.shape[:2]
    if scale != 1.0:
        bgr_s = cv2.resize(bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv2.INTER_AREA)
    else:
        bgr_s = bgr
    h, w = bgr_s.shape[:2]
    hsv = cv2.cvtColor(bgr_s, cv2.COLOR_BGR2HSV)

    # Hardwood / tan + court green (broadcast varies a lot).
    wood = cv2.inRange(hsv, (5, 25, 50), (35, 255, 255))
    green = cv2.inRange(hsv, (30, 30, 35), (95, 255, 255))
    mask = cv2.bitwise_or(wood, green)

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    if area < 0.04 * float(h * w):
        return None

    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = order_corners_tl_tr_br_bl(box.astype(np.float32))

    if scale != 1.0:
        box[:, 0] /= scale
        box[:, 1] /= scale

    box[:, 0] = np.clip(box[:, 0], 0.0, w0 - 1.0)
    box[:, 1] = np.clip(box[:, 1], 0.0, h0 - 1.0)
    return box


def parse_manual_corners(s: str) -> list[tuple[float, float]]:
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 8:
        raise ValueError("Expected 8 numbers: x1,y1,x2,y2,x3,y3,x4,y4 (TL,TR,BR,BL).")
    vals = [float(x) for x in parts]
    return [(vals[i], vals[i + 1]) for i in range(0, 8, 2)]


def try_auto_corners_update(bgr: np.ndarray) -> list[tuple[float, float]] | None:
    """If auto-detection succeeds, return four corners; else None (keep prior homography)."""
    quad = detect_court_quad_auto(bgr)
    if quad is None:
        return None
    return [(float(r[0]), float(r[1])) for r in quad]


def resolve_court_corners(
    frame_bgr: np.ndarray,
    mode: CourtMode,
    *,
    calibration_json: Path | None,
    manual_corners: list[tuple[float, float]] | None,
) -> tuple[list[tuple[float, float]], str]:
    """Return corners list TL,TR,BR,BL and a short tag describing source."""
    h, w = frame_bgr.shape[:2]

    if mode == "default":
        return default_margin_corners(w, h), "default_margin"

    if mode == "manual":
        if not manual_corners or len(manual_corners) != 4:
            raise ValueError("manual mode requires exactly four corners.")
        return list(manual_corners), "manual"

    if mode == "calib":
        if calibration_json is None:
            raise ValueError("calib mode requires --calibration-json.")
        arr = corners_from_calibration_json(calibration_json, (w, h))
        if arr is None:
            raise RuntimeError(
                "Could not project court corners from calibration JSON (check JSON matches frame size)."
            )
        pts = [(float(r[0]), float(r[1])) for r in arr]
        return pts, f"calib:{calibration_json.name}"

    if mode == "auto":
        quad = detect_court_quad_auto(frame_bgr)
        if quad is None:
            return default_margin_corners(w, h), "auto_failed_fallback_default_margin"
        pts = [(float(r[0]), float(r[1])) for r in quad]
        return pts, "auto_hsv_quad"

    raise ValueError(f"Unknown court mode {mode!r}")
