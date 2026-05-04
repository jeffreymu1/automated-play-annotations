import argparse
from pathlib import Path

from .config import PipelineConfig
from .pipeline import infer_default_corners, run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate video or image folder")
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, default=Path("results/annotated.mp4"))
    parser.add_argument("--output-plot", type=Path, default=Path("results/field_projection.png"))
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--field-width", type=float, default=28.0)
    parser.add_argument("--field-height", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--trail-length", type=int, default=48)
    parser.add_argument("--sequence-fps", type=float, default=24.0)
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
        yolo_model=str(args.yolo_model),
        confidence=args.confidence,
        iou=args.iou,
        max_frames=args.max_frames,
        trail_length=args.trail_length,
        sequence_fps=args.sequence_fps,
    )
    run_pipeline(cfg, corners)


if __name__ == "__main__":
    main()
