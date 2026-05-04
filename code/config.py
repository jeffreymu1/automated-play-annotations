from dataclasses import dataclass
from pathlib import Path


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

