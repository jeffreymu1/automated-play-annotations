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


def run_pipeline(config: PipelineConfig, image_corners: list[tuple[float, float]]) -> None:
    cap = cv2.VideoCapture(str(config.input_video))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {config.input_video}")

    ok, first_frame = cap.read()
    if not ok:
        raise RuntimeError("Input video has no frames.")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer = video_writer_for(first_frame, config.output_video, fps)

    detector = YoloDetector(config.yolo_model, config.confidence, config.iou)
    tracker = CentroidTracker()
    homography = build_homography_from_corners(
        image_corners,
        config.field_width_m,
        config.field_height_m,
    )

    projected_points: list[tuple[str, int, float, float]] = []
    frame_count = 0
    max_frames = config.max_frames

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_count += 1
        if max_frames is not None and frame_count > max_frames:
            break

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        annotated = draw_detections(frame.copy(), detections, tracks)
        writer.write(annotated)

        for tr in tracks:
            fx, fy = project_image_point_to_field(tr.position, homography)
            projected_points.append((tr.cls_name, tr.track_id, fx, fy))

    cap.release()
    writer.release()
    save_field_plot(
        projected_points,
        config.output_plot,
        config.field_width_m,
        config.field_height_m,
    )
    print(f"[green]Done.[/green] Processed {frame_count} frames.")
    print(f"Annotated video: {config.output_video}")
    print(f"2D field plot: {config.output_plot}")


def default_corners_for_frame(frame_width: int, frame_height: int) -> list[tuple[float, float]]:
    """Temporary default corners; replace with learned/automatic corner detector."""
    margin_x = max(20, int(0.08 * frame_width))
    margin_y = max(20, int(0.1 * frame_height))
    return [
        (margin_x, margin_y),
        (frame_width - margin_x, margin_y),
        (frame_width - margin_x, frame_height - margin_y),
        (margin_x, frame_height - margin_y),
    ]


def infer_default_corners(video_path: Path) -> list[tuple[float, float]]:
    cap = cv2.VideoCapture(str(video_path))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame of {video_path}")
    h, w = frame.shape[:2]
    return default_corners_for_frame(w, h)

