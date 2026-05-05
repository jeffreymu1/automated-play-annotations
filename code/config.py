from dataclasses import dataclass
from pathlib import Path
from typing import Literal

CourtMode = Literal["default", "auto", "manual", "calib"]


@dataclass
class PipelineConfig:
    input_video: Path
    output_video: Path
    output_plot: Path
    field_width_m: float = 28.0
    field_height_m: float = 15.0
    yolo_model: str = "yolov8n.pt"
    confidence: float = 0.35
    iou: float = 0.5
    max_frames: int | None = None
    trail_length: int = 48
    sequence_fps: float = 24.0
    # Homography source
    court_mode: CourtMode = "auto"
    court_calibration_json: Path | None = None
    court_corners_manual: list[tuple[float, float]] | None = None
    court_refresh_every: int = 0
    draw_court_overlay: bool = True
    # YOLO: substring match against class names (e.g. coco "person", "sports ball")
    yolo_class_substrings: list[str] | None = None
    # Whiteboard-style static play output.
    output_whiteboard: Path | None = Path("results/play_whiteboard.png")
    whiteboard_arrows: bool = True
    whiteboard_max_players: int = 10
    whiteboard_arrow_scale: float = 1.0
