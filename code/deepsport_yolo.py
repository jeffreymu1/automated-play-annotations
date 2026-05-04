from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .calib import CameraCalibration, clip_xyxy, project_world_to_image

# mask: pixel = 1000 * class_code + instance_id; class 1 = human, 3 = ball
MASK_CLASS_PLAYER = 1
MASK_CLASS_BALL = 3

YOLO_CLASS_PLAYER = 0
YOLO_CLASS_BALL = 1


def _mask_class_code(pixel_value: int) -> int:
    return int(pixel_value) // 1000


def _pixel_value_to_yolo_class(pixel_value: int) -> int | None:
    cc = _mask_class_code(pixel_value)
    if cc == MASK_CLASS_PLAYER:
        return YOLO_CLASS_PLAYER
    if cc == MASK_CLASS_BALL:
        return YOLO_CLASS_BALL
    return None


def boxes_from_instance_mask(mask: np.ndarray) -> list[tuple[int, float, float, float, float]]:
    h, w = mask.shape[:2]
    out: list[tuple[int, float, float, float, float]] = []
    for v in np.unique(mask):
        vi = int(v)
        if vi < 1000:
            continue
        yc = _pixel_value_to_yolo_class(vi)
        if yc is None:
            continue
        ys, xs = np.where(mask == vi)
        if len(xs) == 0:
            continue
        x1, x2 = float(xs.min()), float(xs.max())
        y1, y2 = float(ys.min()), float(ys.max())
        x1, y1, x2, y2 = clip_xyxy(x1, y1, x2, y2, w, h)
        if (x2 - x1) < 1.5 or (y2 - y1) < 1.5:
            continue
        out.append((yc, x1, y1, x2, y2))
    return out


def _ball_box_from_center(
    u: float,
    v: float,
    w: int,
    h: int,
    half_side: float = 14.0,
) -> tuple[int, float, float, float, float] | None:
    if not (np.isfinite(u) and np.isfinite(v)):
        return None
    x1, y1, x2, y2 = clip_xyxy(
        u - half_side,
        v - half_side,
        u + half_side,
        v + half_side,
        w,
        h,
    )
    if (x2 - x1) < 2 or (y2 - y1) < 2:
        return None
    return (YOLO_CLASS_BALL, x1, y1, x2, y2)


def boxes_from_index_annotations(
    annotations: list[dict[str, Any]],
    calib: CameraCalibration,
    img_w: int,
    img_h: int,
    player_margin: float = 6.0,
) -> list[tuple[int, float, float, float, float]]:
    out: list[tuple[int, float, float, float, float]] = []
    for ann in annotations:
        t = ann.get("type")
        if t == "ball":
            if not ann.get("visible", True):
                continue
            center = ann["center"]
            uv = project_world_to_image(np.array([center], dtype=np.float64), calib)[0]
            bb = _ball_box_from_center(float(uv[0]), float(uv[1]), img_w, img_h)
            if bb is not None:
                out.append(bb)
        elif t == "player":
            pts = []
            for k in ("head", "hips", "foot1", "foot2"):
                if k in ann:
                    pts.append(ann[k])
            if not pts:
                continue
            arr = np.array(pts, dtype=np.float64)
            uv = project_world_to_image(arr, calib)
            good = np.isfinite(uv).all(axis=1)
            if not good.any():
                continue
            uv = uv[good]
            x1, y1 = float(uv[:, 0].min()), float(uv[:, 1].min())
            x2, y2 = float(uv[:, 0].max()), float(uv[:, 1].max())
            x1 -= player_margin
            y1 -= player_margin
            x2 += player_margin
            y2 += player_margin
            x1, y1, x2, y2 = clip_xyxy(x1, y1, x2, y2, img_w, img_h)
            if (x2 - x1) < 2 or (y2 - y1) < 2:
                continue
            out.append((YOLO_CLASS_PLAYER, x1, y1, x2, y2))
    return out


def xyxy_to_yolo_line(
    cls_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
) -> str:
    cx = ((x1 + x2) / 2.0) / img_w
    cy = ((y1 + y2) / 2.0) / img_h
    bw = (x2 - x1) / img_w
    bh = (y2 - y1) / img_h
    return f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


@dataclass
class IndexRecord:
    arena: str
    game_id: int
    timestamp: int
    annotations: list[dict[str, Any]]


def load_basketball_index(path: Path) -> list[IndexRecord]:
    data = json.loads(path.read_text())
    return [
        IndexRecord(
            arena=e["arena_label"],
            game_id=int(e["game_id"]),
            timestamp=int(e["timestamp"]),
            annotations=e["annotations"],
        )
        for e in data
    ]


def index_image_path(rec: IndexRecord, data_root: Path, image_cam_index: int) -> Path:
    stem = f"camcourt{image_cam_index + 1}_{rec.timestamp}"
    return data_root / rec.arena / str(rec.game_id) / f"{stem}_0.png"


def build_index_path_to_annotations(
    data_root: Path,
) -> dict[Path, tuple[list[dict[str, Any]], int]]:
    idx_path = data_root / "basketball-instants-dataset.json"
    if not idx_path.is_file():
        return {}
    mapping: dict[Path, tuple[list[dict[str, Any]], int]] = {}
    for rec in load_basketball_index(idx_path):
        by_cam: dict[int, list[dict[str, Any]]] = {}
        for ann in rec.annotations:
            cam = int(ann.get("image", 0))
            by_cam.setdefault(cam, []).append(ann)
        for cam, annots in by_cam.items():
            png = index_image_path(rec, data_root, cam)
            mapping[png.resolve()] = (annots, cam)
    return mapping


def humans_mask_path(frame0_png: Path) -> Path:
    return frame0_png.parent / (frame0_png.name.replace("_0.png", "_humans.png"))


def collect_labels_for_frame(
    frame0_png: Path,
    index_map: dict[Path, tuple[list[dict[str, Any]], int]],
) -> list[str] | None:
    with Image.open(frame0_png) as im:
        img_w, img_h = im.size

    mask_boxes: list[tuple[int, float, float, float, float]] = []
    mask_path = humans_mask_path(frame0_png)
    if mask_path.is_file():
        mask = np.array(Image.open(mask_path))
        if mask.ndim == 2:
            mask_boxes = boxes_from_instance_mask(mask)

    index_boxes: list[tuple[int, float, float, float, float]] = []
    key = frame0_png.resolve()
    json_calib = frame0_png.parent / (frame0_png.name.replace("_0.png", ".json"))
    if key in index_map and json_calib.is_file():
        index_annots, _cam_idx = index_map[key]
        calib = CameraCalibration.from_json(json_calib)
        index_boxes = boxes_from_index_annotations(index_annots, calib, img_w, img_h)

    players_m = [b for b in mask_boxes if b[0] == YOLO_CLASS_PLAYER]
    balls_m = [b for b in mask_boxes if b[0] == YOLO_CLASS_BALL]
    players_i = [b for b in index_boxes if b[0] == YOLO_CLASS_PLAYER]
    balls_i = [b for b in index_boxes if b[0] == YOLO_CLASS_BALL]

    merged: list[tuple[int, float, float, float, float]] = []
    if players_m:
        merged.extend(players_m)
    else:
        merged.extend(players_i)
    if balls_m:
        merged.extend(balls_m)
    elif balls_i:
        merged.extend(balls_i)

    if not merged:
        return None

    return [xyxy_to_yolo_line(c, x1, y1, x2, y2, img_w, img_h) for c, x1, y1, x2, y2 in merged]


def iter_frame0_pngs(data_root: Path) -> list[Path]:
    return sorted(data_root.glob("*/*/*_0.png"))


def default_train_val_split(frame: Path, val_mod: int = 10) -> str:
    try:
        game_id = int(frame.parent.name)
    except ValueError:
        return "train"
    return "val" if game_id % val_mod == 0 else "train"
