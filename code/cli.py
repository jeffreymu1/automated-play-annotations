import argparse
from pathlib import Path

from .config import PipelineConfig
from .court_inference import parse_manual_corners
from .pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Basketball video or frame folder: YOLO detect, track, court homography, "
            "annotated video + field plot."
        ),
    )
    parser.add_argument("--input-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, default=Path("results/annotated.mp4"))
    parser.add_argument("--output-plot", type=Path, default=Path("results/field_projection.png"))
    parser.add_argument("--output-whiteboard", type=Path, default=Path("results/play_whiteboard.png"))
    parser.add_argument("--yolo-model", default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--field-width", type=float, default=28.0)
    parser.add_argument("--field-height", type=float, default=15.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--trail-length", type=int, default=48)
    parser.add_argument("--sequence-fps", type=float, default=24.0)

    parser.add_argument(
        "--court-mode",
        choices=("default", "auto", "manual", "calib"),
        default="auto",
        help="Court quad: auto (HSV+contour heuristic), calib JSON, manual 8 coords, or default margin rectangle.",
    )
    parser.add_argument(
        "--calibration-json",
        type=Path,
        default=None,
        help="For calib mode: DeepSport-style calibration JSON matching the footage resolution.",
    )
    parser.add_argument(
        "--court-corners",
        type=str,
        default=None,
        help="manual mode: x1,y1,x2,y2,x3,y3,x4,y4 = TL,TR,BR,BL in pixels.",
    )
    parser.add_argument(
        "--court-refresh",
        type=int,
        default=0,
        help="If >0 with court-mode=auto, refine court polygon every N frames.",
    )
    parser.add_argument(
        "--no-draw-court",
        action="store_true",
        help="Skip translucent court polygon + corner labels on the output video.",
    )
    parser.add_argument(
        "--yolo-classes",
        type=str,
        default="",
        help='Comma-separated substrings to keep (e.g. "person,sports ball"). Empty = all classes.',
    )
    parser.add_argument(
        "--no-whiteboard-arrows",
        action="store_true",
        help="Disable movement arrows on whiteboard output.",
    )
    parser.add_argument(
        "--whiteboard-max-players",
        type=int,
        default=10,
        help="Max player markers in whiteboard snapshot (default 10).",
    )
    parser.add_argument(
        "--whiteboard-arrow-scale",
        type=float,
        default=1.0,
        help="Amplify whiteboard arrow length while keeping start positions fixed (e.g. 2.0).",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.court_mode == "manual" and not args.court_corners:
        raise SystemExit("--court-mode manual requires --court-corners with 8 numbers.")
    if args.court_mode == "calib" and args.calibration_json is None:
        raise SystemExit("--court-mode calib requires --calibration-json.")

    manual = parse_manual_corners(args.court_corners) if args.court_corners else None
    subs_list = [s.strip() for s in args.yolo_classes.split(",") if s.strip()]
    subs = subs_list or None

    cfg = PipelineConfig(
        input_video=args.input_video,
        output_video=args.output_video,
        output_plot=args.output_plot,
        output_whiteboard=args.output_whiteboard,
        field_width_m=args.field_width,
        field_height_m=args.field_height,
        yolo_model=str(args.yolo_model),
        confidence=args.confidence,
        iou=args.iou,
        max_frames=args.max_frames,
        trail_length=args.trail_length,
        sequence_fps=args.sequence_fps,
        court_mode=args.court_mode,
        court_calibration_json=args.calibration_json,
        court_corners_manual=manual,
        court_refresh_every=args.court_refresh,
        draw_court_overlay=not args.no_draw_court,
        yolo_class_substrings=subs,
        whiteboard_arrows=not args.no_whiteboard_arrows,
        whiteboard_max_players=max(1, int(args.whiteboard_max_players)),
        whiteboard_arrow_scale=max(0.1, float(args.whiteboard_arrow_scale)),
    )
    run_pipeline(cfg)


if __name__ == "__main__":
    main()
