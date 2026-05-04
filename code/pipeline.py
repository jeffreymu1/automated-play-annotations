from collections import defaultdict, deque
from pathlib import Path

import cv2
from rich import print

from .config import PipelineConfig
from .detectors import YoloDetector
from .field import build_homography_from_corners
from .io_utils import video_writer_for
from .projection import project_image_point_to_field
from .tracking import CentroidTracker
from .visualize import draw_detections, save_field_plot


def list_frame_paths(input_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def run_pipeline(config: PipelineConfig, image_corners: list[tuple[float, float]]) -> None:
    input_path = config.input_video
    use_image_sequence = input_path.is_dir()

    cap: cv2.VideoCapture | None = None
    paths: list[Path] = []

    if use_image_sequence:
        paths = list_frame_paths(input_path)
        if not paths:
            raise FileNotFoundError(f"No images (.png/.jpg/...) in directory: {input_path}")
        first_frame = cv2.imread(str(paths[0]))
        if first_frame is None:
            raise RuntimeError(f"Could not read image: {paths[0]}")
        fps = float(config.sequence_fps)
    else:
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {input_path}")
        ok, first_frame = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError("Input video has no frames.")
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    writer = video_writer_for(first_frame, config.output_video, fps)

    detector = YoloDetector(config.yolo_model, config.confidence, config.iou)
    tracker = CentroidTracker()
    homography = build_homography_from_corners(
        image_corners,
        config.field_width_m,
        config.field_height_m,
    )

    trajectories: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    trail_len = max(2, config.trail_length)
    image_trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_len))
    track_class: dict[int, str] = {}
    frame_count = 0
    max_frames = config.max_frames

    def process_frame(frame) -> None:
        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        for tr in tracks:
            track_class[tr.track_id] = tr.cls_name
            ix, iy = int(tr.position[0]), int(tr.position[1])
            image_trails[tr.track_id].append((ix, iy))
            fx, fy = project_image_point_to_field(tr.position, homography)
            trajectories[(tr.cls_name, tr.track_id)].append((fx, fy))

        annotated = draw_detections(
            frame.copy(),
            detections,
            tracks,
            image_trails=image_trails,
            track_class=track_class,
        )
        writer.write(annotated)

    if use_image_sequence:
        for p in paths:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            if max_frames is not None and frame_count >= max_frames:
                break
            process_frame(frame)
            frame_count += 1
    else:
        assert cap is not None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_count >= max_frames:
                break
            process_frame(frame)
            frame_count += 1
        cap.release()

    writer.release()
    save_field_plot(
        dict(trajectories),
        config.output_plot,
        config.field_width_m,
        config.field_height_m,
    )
    print(f"[green]Done.[/green] Processed {frame_count} frames.")
    print(f"Annotated video: {config.output_video}")
    print(f"2D field plot: {config.output_plot}")


def default_corners_for_frame(frame_width: int, frame_height: int) -> list[tuple[float, float]]:
    margin_x = max(20, int(0.08 * frame_width))
    margin_y = max(20, int(0.1 * frame_height))
    return [
        (margin_x, margin_y),
        (frame_width - margin_x, margin_y),
        (frame_width - margin_x, frame_height - margin_y),
        (margin_x, frame_height - margin_y),
    ]


def infer_default_corners(input_path: Path) -> list[tuple[float, float]]:
    if input_path.is_dir():
        paths = list_frame_paths(input_path)
        if not paths:
            raise RuntimeError(f"No images in directory {input_path}")
        frame = cv2.imread(str(paths[0]))
        if frame is None:
            raise RuntimeError(f"Could not read {paths[0]}")
    else:
        cap = cv2.VideoCapture(str(input_path))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(f"Could not read first frame of {input_path}")
    h, w = frame.shape[:2]
    return default_corners_for_frame(w, h)

