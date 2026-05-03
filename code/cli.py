import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import infer_default_corners, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic Sports Game Annotation")
    parser.add_argument("--input-video", type=Path, required=True, help="Path to input sports video")
    parser.add_argument(
        "--output-video",
        type=Path,
        default=Path("results/annotated.mp4"),
        help="Annotated output video path",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path("results/field_projection.png"),
        help="Output 2D field projection image path",
    )
    parser.add_argument("--yolo-model", default="yolov8n.pt", help="YOLOv8 model name/path")
    parser.add_argument("--confidence", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--field-width", type=float, default=28.0, help="Field width in meters")
    parser.add_argument("--field-height", type=float, default=15.0, help="Field height in meters")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional frame limit for quick runs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corners = infer_default_corners(args.input_video)
    cfg = PipelineConfig(
        input_video=args.input_video,
        output_video=args.output_video,
        output_plot=args.output_plot,
        field_width_m=args.field_width,
        field_height_m=args.field_height,
        yolo_model=args.yolo_model,
        confidence=args.confidence,
        iou=args.iou,
        max_frames=args.max_frames,
    )
    run_pipeline(cfg, corners)


if __name__ == "__main__":
    main()

