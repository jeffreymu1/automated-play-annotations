from pathlib import Path

import cv2


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def video_writer_for(
    sample_frame,
    out_path: Path,
    fps: float,
):
    ensure_parent(out_path)
    height, width = sample_frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

