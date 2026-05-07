"""Warp frames to a metric 2D court view and overlay YOLO player locations."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from court_detection.geometry import COURT_LENGTH_CM, COURT_WIDTH_CM
from player_detection.yolo_player_locator import YoloPlayerLocator


@dataclass(frozen=True)
class WarpInput:
    image_path: Path
    homography_path: Path
    H_world_to_image: np.ndarray
    frame_index: int | None = None
    label: str | None = None


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    locator = YoloPlayerLocator(
        model_name=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        ankle_conf=args.ankle_conf,
    )

    entries = discover_inputs(args.homography_source, args.dataset_root, args.include_sibling_offsets)
    if not entries:
        raise RuntimeError(f"No usable homography/image pairs found in {args.homography_source}")

    manifest = []
    saved_count = 0
    for attempt_index, entry in enumerate(entries):
        if args.count is not None and saved_count >= args.count:
            break
        frame_bgr = cv2.imread(str(entry.image_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f"Could not read image: {entry.image_path}")

        players = locator.predict_frame(frame_bgr)
        warped, M_image_to_canvas = warp_frame_to_court_canvas(
            frame_bgr,
            entry.H_world_to_image,
            pixels_per_meter=args.pixels_per_meter,
            margin_meters=args.margin_meters,
        )
        player_points = np.array([player.xy for player in players], dtype=np.float32).reshape(-1, 2)
        warped_players = transform_points(player_points, M_image_to_canvas) if len(player_points) else np.zeros((0, 2))
        keep = on_court_mask(warped_players, args.pixels_per_meter, args.margin_meters)
        court_players = [player for player, ok in zip(players, keep) if ok]
        court_warped_players = warped_players[keep]
        if len(court_players) == 0:
            print(f"skip {attempt_index:03d} {entry.image_path}: no on-court players")
            continue

        if args.draw_court_outline:
            draw_court_outline(warped, args.pixels_per_meter, args.margin_meters)
        draw_players(warped, court_warped_players, court_players)

        clip_key = clip_key_from_path(entry.image_path)
        raw_dir = args.out / "raw"
        warped_dir = args.out / "warped"
        raw_dir.mkdir(parents=True, exist_ok=True)
        warped_dir.mkdir(parents=True, exist_ok=True)

        stem = f"{saved_count:03d}_{clip_key}_{safe_stem(entry.image_path)}"
        raw_path = raw_dir / f"{stem}.jpg"
        warped_path = warped_dir / f"{stem}_court_warp.png"
        cv2.imwrite(str(raw_path), frame_bgr)
        cv2.imwrite(str(warped_path), warped)
        manifest.append(
            {
                "index": saved_count,
                "attempt_index": attempt_index,
                "clip": clip_key,
                "image_path": str(entry.image_path),
                "homography_path": str(entry.homography_path),
                "frame_index": entry.frame_index,
                "label": entry.label,
                "raw_output_path": str(raw_path),
                "warped_output_path": str(warped_path),
                "num_players_total": len(players),
                "num_players_on_court": len(court_players),
                "court_pixels": {
                    "length": int(round(COURT_LENGTH_CM / 100.0 * args.pixels_per_meter)),
                    "width": int(round(COURT_WIDTH_CM / 100.0 * args.pixels_per_meter)),
                    "margin": int(round(args.margin_meters * args.pixels_per_meter)),
                },
                "players": [
                    {
                        **player.to_json(),
                        "warped_xy": warped_xy.astype(float).tolist(),
                    }
                    for player, warped_xy in zip(court_players, court_warped_players)
                ],
            }
        )
        print(f"{saved_count:03d} wrote {warped_path} ({len(court_players)}/{len(players)} players on court)")
        saved_count += 1

    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(json.dumps({"frames": manifest}, indent=2))
    print(f"wrote {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--homography-source", type=Path, default=Path("results/line_refinement"))
    parser.add_argument("--dataset-root", type=Path, default=Path("data/deepsport-dataset"))
    parser.add_argument("--out", type=Path, default=Path("player_detection/output/court_warp_vis"))
    parser.add_argument("--count", type=int, default=None)
    parser.add_argument("--include-sibling-offsets", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--draw-court-outline", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pixels-per-meter", type=float, default=35.0)
    parser.add_argument("--margin-meters", type=float, default=5.0)
    parser.add_argument("--model", default="player_detection/output/models/yolo11m-pose.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--ankle-conf", type=float, default=0.2)
    return parser.parse_args()


def discover_inputs(source: Path, dataset_root: Path, include_sibling_offsets: bool) -> list[WarpInput]:
    if source.is_file():
        return inputs_from_json(source, dataset_root, include_sibling_offsets)

    diagnostics = sorted(source.glob("*/diagnostics.json"))
    entries: list[WarpInput] = []
    for path in diagnostics:
        entries.extend(inputs_from_json(path, dataset_root, include_sibling_offsets=False))

    for path in sorted(source.glob("*.json")):
        entries.extend(inputs_from_json(path, dataset_root, include_sibling_offsets))
    return dedupe_inputs(entries)


def inputs_from_json(path: Path, dataset_root: Path, include_sibling_offsets: bool) -> list[WarpInput]:
    data = json.loads(path.read_text())
    if isinstance(data.get("frames"), list):
        entries = []
        for frame in data["frames"]:
            H = homography_from_record(frame)
            image_path = frame.get("image_path")
            if H is None or image_path is None:
                continue
            entries.append(
                WarpInput(
                    image_path=Path(image_path),
                    homography_path=path,
                    H_world_to_image=H,
                    frame_index=frame.get("frame_index"),
                    label=frame.get("source_name") or frame.get("camera"),
                )
            )
        return entries

    H = homography_from_record(data)
    if H is None:
        return []
    image_paths = infer_images_for_standalone_json(path, dataset_root)
    if not include_sibling_offsets:
        image_paths = image_paths[:1]
    return [
        WarpInput(image_path=image_path, homography_path=path, H_world_to_image=H, label=path.stem)
        for image_path in image_paths
    ]


def homography_from_record(record: dict[str, object]) -> np.ndarray | None:
    for key in ("homography",):
        if record.get(key) is not None:
            return np.asarray(record[key], dtype=float)
    stage3 = record.get("stage3")
    if isinstance(stage3, dict) and stage3.get("H") is not None:
        return np.asarray(stage3["H"], dtype=float)
    stage2 = record.get("stage2")
    if isinstance(stage2, dict) and stage2.get("H") is not None:
        return np.asarray(stage2["H"], dtype=float)
    return None


def infer_images_for_standalone_json(path: Path, dataset_root: Path) -> list[Path]:
    match = re.match(r"(?P<camera>camcourt\d+)_(?P<timestamp>\d+)_(?P<offset>\d+)(?:_.*)?$", path.stem)
    if match is None:
        return []

    camera = match.group("camera")
    timestamp = match.group("timestamp")
    offset = match.group("offset")
    base_name = f"{camera}_{timestamp}"
    primary = sorted(dataset_root.glob(f"*/*/{base_name}_{offset}.png"))
    siblings = sorted(dataset_root.glob(f"*/*/{base_name}_*.png"))
    return dedupe_paths(primary + [p for p in siblings if p.name != f"{base_name}_humans.png"])


def warp_frame_to_court_canvas(
    frame_bgr: np.ndarray,
    H_world_to_image: np.ndarray,
    pixels_per_meter: float,
    margin_meters: float,
) -> tuple[np.ndarray, np.ndarray]:
    scale = pixels_per_meter / 100.0
    margin_px = margin_meters * pixels_per_meter
    court_w_px = COURT_LENGTH_CM * scale
    court_h_px = COURT_WIDTH_CM * scale
    canvas_w = int(round(court_w_px + 2.0 * margin_px))
    canvas_h = int(round(court_h_px + 2.0 * margin_px))

    world_to_canvas = np.array(
        [
            [scale, 0.0, margin_px],
            [0.0, scale, margin_px],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    image_to_canvas = world_to_canvas @ np.linalg.inv(H_world_to_image)
    frame_bgra = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2BGRA)
    frame_bgra[:, :, 3] = 255
    warped = cv2.warpPerspective(
        frame_bgra,
        image_to_canvas,
        (canvas_w, canvas_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
    return warped, image_to_canvas


def transform_points(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    points_h = np.concatenate([points_xy, np.ones((len(points_xy), 1), dtype=points_xy.dtype)], axis=1)
    warped_h = points_h @ H.T
    return (warped_h[:, :2] / warped_h[:, 2:3]).astype(np.float32)


def draw_court_outline(image_bgr: np.ndarray, pixels_per_meter: float, margin_meters: float) -> None:
    scale = pixels_per_meter / 100.0
    margin_px = margin_meters * pixels_per_meter
    x0 = int(round(margin_px))
    y0 = int(round(margin_px))
    x1 = int(round(margin_px + COURT_LENGTH_CM * scale))
    y1 = int(round(margin_px + COURT_WIDTH_CM * scale))
    white = draw_color(image_bgr, (255, 255, 255))
    cv2.rectangle(image_bgr, (x0, y0), (x1, y1), white, 3, cv2.LINE_AA)
    cv2.line(image_bgr, ((x0 + x1) // 2, y0), ((x0 + x1) // 2, y1), white, 2, cv2.LINE_AA)


def on_court_mask(points_xy: np.ndarray, pixels_per_meter: float, margin_meters: float) -> np.ndarray:
    if len(points_xy) == 0:
        return np.zeros(0, dtype=bool)
    scale = pixels_per_meter / 100.0
    margin_px = margin_meters * pixels_per_meter
    x0 = margin_px
    y0 = margin_px
    x1 = margin_px + COURT_LENGTH_CM * scale
    y1 = margin_px + COURT_WIDTH_CM * scale
    return (
        np.isfinite(points_xy).all(axis=1)
        & (points_xy[:, 0] >= x0)
        & (points_xy[:, 0] <= x1)
        & (points_xy[:, 1] >= y0)
        & (points_xy[:, 1] <= y1)
    )


def draw_players(image_bgr: np.ndarray, points_xy: np.ndarray, players: list[object]) -> None:
    for i, (xy, player) in enumerate(zip(points_xy, players)):
        if not np.isfinite(xy).all():
            continue
        x, y = np.rint(xy).astype(int)
        if x < -50 or y < -50 or x > image_bgr.shape[1] + 50 or y > image_bgr.shape[0] + 50:
            continue
        color = draw_color(image_bgr, (0, 0, 255) if getattr(player, "conf", 0.0) > 0.0 else (0, 180, 255))
        white = draw_color(image_bgr, (255, 255, 255))
        black = draw_color(image_bgr, (0, 0, 0))
        cv2.circle(image_bgr, (x, y), 7, color, -1, cv2.LINE_AA)
        cv2.circle(image_bgr, (x, y), 9, white, 2, cv2.LINE_AA)
        cv2.putText(
            image_bgr,
            str(i),
            (x + 10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            white,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image_bgr,
            str(i),
            (x + 10, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            black,
            1,
            cv2.LINE_AA,
        )


def draw_color(image: np.ndarray, bgr: tuple[int, int, int]) -> tuple[int, ...]:
    return (*bgr, 255) if image.shape[2] == 4 else bgr


def dedupe_inputs(entries: list[WarpInput]) -> list[WarpInput]:
    seen: set[tuple[str, str]] = set()
    out = []
    for entry in entries:
        key = (str(entry.image_path), str(entry.homography_path))
        if key in seen:
            continue
        seen.add(key)
        out.append(entry)
    return out


def dedupe_paths(paths: list[Path]) -> list[Path]:
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def safe_stem(path: Path) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem).strip("._")[:80] or "frame"


def clip_key_from_path(path: Path) -> str:
    parts = path.parts
    if "deepsport-dataset" in parts:
        i = parts.index("deepsport-dataset")
        if len(parts) > i + 3:
            game = parts[i + 1]
            segment = parts[i + 2]
            camera = path.stem.split("_", 1)[0]
            return safe_name(f"{game}_{segment}_{camera}")
    encoded = re.match(r"(?:\d+_)?(?P<game>KS-FR-[^_]+)_(?P<segment>\d+)_(?P<camera>camcourt\d+)_", path.name)
    if encoded is not None:
        return safe_name(f"{encoded.group('game')}_{encoded.group('segment')}_{encoded.group('camera')}")
    if len(parts) >= 3 and parts[-2] == "frames":
        return safe_name(parts[-3])
    return safe_name(path.parent.name)


def safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._") or "clip"


if __name__ == "__main__":
    main()
