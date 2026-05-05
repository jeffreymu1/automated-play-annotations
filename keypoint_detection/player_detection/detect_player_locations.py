"""Run YOLO pose player floor-location inference on images or DeepSport frames."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from player_detection.yolo_player_locator import YoloPlayerLocator, draw_player_locations

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def main() -> None:
    args = parse_args()
    locator = YoloPlayerLocator(
        model_name=args.model,
        device=args.device,
        imgsz=args.imgsz,
        conf=args.conf,
        ankle_conf=args.ankle_conf,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.annotated_dir is not None:
        args.annotated_dir.mkdir(parents=True, exist_ok=True)

    frames = iter_frames(args)
    records = []
    for frame_index, (frame_id, frame_bgr, metadata) in enumerate(frames):
        players = locator.predict_frame(frame_bgr)
        record = {
            "frame_index": frame_index,
            "frame_id": frame_id,
            "image_shape": list(frame_bgr.shape),
            "metadata": metadata,
            "players": [player.to_json() for player in players],
        }
        records.append(record)

        if args.annotated_dir is not None:
            annotated = draw_player_locations(frame_bgr, players)
            cv2.imwrite(str(args.annotated_dir / f"{frame_index:06d}_{safe_stem(frame_id)}.jpg"), annotated)

        print(f"{frame_index:06d} {frame_id}: {len(players)} players")

    payload = {
        "model": args.model,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "ankle_conf": args.ankle_conf,
        "frames": records,
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames", type=Path, help="Image file or folder of image frames.")
    source.add_argument("--dataset-root", type=Path, help="DeepSport dataset root.")

    parser.add_argument("--out", type=Path, default=Path("player_detection/player_locations.json"))
    parser.add_argument("--annotated-dir", type=Path, default=None)
    parser.add_argument("--recursive", action="store_true", help="Search --frames folder recursively.")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--dataset-mode", choices=("samples", "clip"), default="samples")
    parser.add_argument("--clip", type=str, default=None, help="Clip index or name for --dataset-mode clip.")
    parser.add_argument("--camera", type=str, default=None, help="Optional DeepSport clip camera filter.")
    parser.add_argument("--start-index", type=int, default=0, help="First sample index for --dataset-mode samples.")

    parser.add_argument("--model", default="yolo11m-pose.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--ankle-conf", type=float, default=0.2)
    return parser.parse_args()


def iter_frames(args: argparse.Namespace) -> Iterator[tuple[str, np.ndarray, dict[str, object]]]:
    if args.frames is not None:
        yield from iter_image_frames(args.frames, args.recursive, args.limit)
        return

    yield from iter_deepsport_frames(args)


def iter_image_frames(
    frames: Path,
    recursive: bool,
    limit: int | None,
) -> Iterator[tuple[str, np.ndarray, dict[str, object]]]:
    paths = image_paths(frames, recursive)
    if limit is not None:
        paths = paths[:limit]
    for path in paths:
        frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        yield str(path), frame_bgr, {"source": "image", "path": str(path)}


def iter_deepsport_frames(args: argparse.Namespace) -> Iterator[tuple[str, np.ndarray, dict[str, object]]]:
    from court_detection.dataset import DeepSportDataset

    dataset = DeepSportDataset(args.dataset_root)
    if args.dataset_mode == "clip":
        if args.clip is None:
            raise ValueError("--clip is required when --dataset-mode clip is used")
        clip_id: int | str = int(args.clip) if args.clip.isdigit() else args.clip
        frames = dataset.load_clip_frames(clip_id, camera=args.camera)
        if args.limit is not None:
            frames = frames[: args.limit]
        for image_rgb_float, frame, _calib in frames:
            frame_bgr = rgb_float_to_bgr_u8(image_rgb_float)
            frame_id = str(frame.image_path)
            metadata = {
                "source": "deepsport_clip",
                "path": str(frame.image_path),
                "json_path": str(frame.json_path),
                "frame_number": frame.frame_number,
                "camera": frame.camera,
                "timestamp": frame.timestamp,
            }
            yield frame_id, frame_bgr, metadata
        return

    stop = len(dataset) if args.limit is None else min(len(dataset), args.start_index + args.limit)
    for idx in range(args.start_index, stop):
        image, _keypoints, _calib = dataset[idx]
        image_rgb_float = image.permute(1, 2, 0).numpy()
        image_path, json_path = dataset.samples[idx]
        frame_bgr = rgb_float_to_bgr_u8(image_rgb_float)
        metadata = {
            "source": "deepsport_sample",
            "dataset_index": idx,
            "path": str(image_path),
            "json_path": str(json_path),
        }
        yield str(image_path), frame_bgr, metadata


def image_paths(frames: Path, recursive: bool) -> list[Path]:
    if frames.is_file():
        if frames.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {frames}")
        return [frames]
    if not frames.is_dir():
        raise FileNotFoundError(f"Frame path not found: {frames}")

    suffixes = set(IMAGE_EXTENSIONS)
    iterator = frames.rglob("*") if recursive else frames.glob("*")
    return sorted(path for path in iterator if path.is_file() and path.suffix.lower() in suffixes)


def rgb_float_to_bgr_u8(image_rgb_float: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image_rgb_float * 255.0, 0.0, 255.0).astype(np.uint8)
    return cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR)


def safe_stem(frame_id: str) -> str:
    stem = Path(frame_id).stem
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in stem)[:80]


if __name__ == "__main__":
    main()
