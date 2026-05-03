from pathlib import Path

import cv2
import matplotlib.pyplot as plt

from .detectors import Detection
from .tracking import TrackState


def draw_detections(frame, detections: list[Detection], tracks: list[TrackState]):
    track_by_cls = {}
    for tr in tracks:
        track_by_cls.setdefault(tr.cls_name, []).append(tr)

    for det in detections:
        x1, y1, x2, y2 = (int(v) for v in det.xyxy)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        label = f"{det.cls_name}:{det.confidence:.2f}"
        cv2.putText(frame, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 1)

        near = min(
            track_by_cls.get(det.cls_name, []),
            key=lambda t: (t.position[0] - det.center[0]) ** 2 + (t.position[1] - det.center[1]) ** 2,
            default=None,
        )
        if near is not None:
            cv2.putText(
                frame,
                f"ID {near.track_id}",
                (x1, min(frame.shape[0] - 8, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 0),
                1,
            )
    return frame


def save_field_plot(
    points: list[tuple[str, int, float, float]],
    out_path: Path,
    field_width_m: float,
    field_height_m: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, field_width_m)
    ax.set_ylim(field_height_m, 0)
    ax.set_title("Projected Object Positions on 2D Field")
    ax.set_xlabel("Field X (m)")
    ax.set_ylabel("Field Y (m)")
    ax.grid(True, linestyle="--", alpha=0.3)

    for cls_name, track_id, x, y in points:
        color = "orange" if "ball" in cls_name else "royalblue"
        marker = "o" if "ball" in cls_name else "x"
        ax.scatter(x, y, c=color, marker=marker)
        ax.text(x, y, f"{cls_name}:{track_id}", fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

