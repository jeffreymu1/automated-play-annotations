from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .detectors import Detection
from .tracking import TrackState


def draw_detections(
    frame,
    detections: list[Detection],
    tracks: list[TrackState],
    *,
    image_trails: dict[int, deque[tuple[int, int]]] | None = None,
    track_class: dict[int, str] | None = None,
) -> np.ndarray:
    track_by_cls = {}
    for tr in tracks:
        track_by_cls.setdefault(tr.cls_name, []).append(tr)

    if image_trails and track_class:
        for tid, pts in image_trails.items():
            if len(pts) < 2:
                continue
            cls = track_class.get(tid, "")
            is_ball = "ball" in cls.lower()
            color = (0, 140, 255) if is_ball else (255, 128, 0)
            thickness = 3 if is_ball else 2
            arr = np.array(list(pts), dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [arr], isClosed=False, color=color, thickness=thickness)

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
    trajectories: dict[tuple[str, int], list[tuple[float, float]]],
    out_path: Path,
    field_width_m: float,
    field_height_m: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.set_xlim(0, field_width_m)
    ax.set_ylim(field_height_m, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Field tracks (m)")
    ax.set_xlabel("Field X (m)")
    ax.set_ylabel("Field Y (m)")
    ax.grid(True, linestyle="--", alpha=0.25)
    court_x = [0.0, field_width_m, field_width_m, 0.0]
    court_y = [0.0, 0.0, field_height_m, field_height_m]
    ax.fill(court_x + [0.0], court_y + [0.0], closed=True, color="#e8f0e8", zorder=0)
    ax.plot(court_x + [0.0], court_y + [0.0], color="#2d5a2d", linewidth=1.5, zorder=1)

    for (cls_name, track_id), pts in sorted(trajectories.items(), key=lambda x: (x[0][0], x[0][1])):
        if len(pts) < 1:
            continue
        is_ball = "ball" in cls_name.lower()
        color = "#cc5500" if is_ball else "#1f4e79"
        lw = 2.8 if is_ball else 1.6
        alpha = 0.95 if is_ball else 0.55
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        if len(pts) >= 2:
            ax.plot(xs, ys, color=color, linewidth=lw, alpha=alpha, solid_capstyle="round", label=None)
        ax.scatter(xs[-1], ys[-1], c=color, s=55 if is_ball else 28, zorder=3, edgecolors="white", linewidths=0.5)
        ax.annotate(
            f"{cls_name[:1].upper()}{track_id}",
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color="black",
        )
        if len(pts) >= 2:
            ax.scatter(xs[0], ys[0], c=color, s=22, marker="o", zorder=3, alpha=0.5)

    handles = [
        plt.Line2D([0], [0], color="#1f4e79", lw=2, label="player paths"),
        plt.Line2D([0], [0], color="#cc5500", lw=3, label="ball path"),
    ]
    ax.legend(handles=handles, loc="upper right", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

