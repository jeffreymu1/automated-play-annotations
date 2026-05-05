from collections import deque
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .detectors import Detection
from .tracking import TrackState


def _team_and_role_for_track(
    track_id: int,
    first_xy: tuple[float, float],
    field_width_m: float,
) -> tuple[str, bool]:
    """Deterministically map a track to (team_label, is_offense_like)."""
    team_label = "A" if first_xy[0] <= (field_width_m / 2.0) else "B"
    # Keep deterministic and simple: lower id parity flips role marker style inside each team.
    is_offense_like = (track_id % 2) == 0
    return team_label, is_offense_like


def _movement_style(
    pts: list[tuple[float, float]],
) -> tuple[str, float]:
    """Return (linestyle, curvature_rad) inferred from trajectory shape."""
    if len(pts) < 2:
        return "-", 0.0

    x0, y0 = pts[0]
    x1, y1 = pts[-1]
    disp = float(np.hypot(x1 - x0, y1 - y0))
    if disp < 0.35:
        return ":", 0.08

    if len(pts) < 3:
        return "--", 0.0

    # Curvature proxy: max distance to start-end line.
    p0 = np.array([x0, y0], dtype=np.float32)
    p1 = np.array([x1, y1], dtype=np.float32)
    v = p1 - p0
    nv = float(np.linalg.norm(v))
    if nv < 1e-6:
        return ":", 0.0

    max_off = 0.0
    for px, py in pts[1:-1]:
        p = np.array([px, py], dtype=np.float32)
        off = abs(np.cross(v, p - p0) / nv)
        max_off = max(max_off, float(off))
    if max_off > 0.55:
        # Sign by dominant side of midpoint to vary curve direction.
        pm = np.array(pts[len(pts) // 2], dtype=np.float32)
        sgn = np.sign(np.cross(v, pm - p0))
        rad = 0.28 if sgn >= 0 else -0.28
        return "--", rad
    return "--", 0.0


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


def draw_court_quad(
    frame: np.ndarray,
    corners_xy: list[tuple[float, float]],
    *,
    color: tuple[int, int, int] = (220, 200, 60),
    thickness: int = 2,
    fill_overlay: float = 0.12,
) -> np.ndarray:
    """Draw planar court polygon (TL, TR, BR, BL, closed) with optional translucent fill."""
    if len(corners_xy) != 4:
        return frame
    pts = np.round(np.array(corners_xy, dtype=np.float32)).astype(np.int32).reshape(-1, 1, 2)

    overlay = frame
    if fill_overlay > 0:
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        frame = cv2.addWeighted(overlay, fill_overlay, frame, 1.0 - fill_overlay, 0)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness, lineType=cv2.LINE_AA)
    for i, (x, y) in enumerate(corners_xy):
        cv2.circle(frame, (int(round(x)), int(round(y))), 6, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(
            frame,
            str(i),
            (int(round(x)) + 6, int(round(y)) - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
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


def save_whiteboard_play(
    trajectories: dict[tuple[str, int], list[tuple[float, float]]],
    out_path: Path,
    field_width_m: float,
    field_height_m: float,
    *,
    draw_arrows: bool = True,
    include_player_ids: set[int] | None = None,
    include_ball_id: int | None = None,
    team_by_player_id: dict[int, str] | None = None,
) -> int:
    """Render a whiteboard-style half-sketched play image with court and players."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, field_width_m)
    ax.set_ylim(field_height_m, 0)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    # Court outline and a few key lines to resemble coaching whiteboard style.
    ax.plot([0, field_width_m, field_width_m, 0, 0], [0, 0, field_height_m, field_height_m, 0], "k-", lw=2)
    mid_x = field_width_m / 2.0
    ax.plot([mid_x, mid_x], [0, field_height_m], "k-", lw=1.5)
    center_r = field_height_m * 0.12
    ax.add_patch(plt.Circle((mid_x, field_height_m / 2.0), center_r, fill=False, color="k", lw=1.2))

    # Simple paint/arc hints on both sides.
    paint_w = field_height_m * 0.34
    paint_l = field_width_m * 0.19
    y0 = (field_height_m - paint_w) / 2.0
    ax.add_patch(plt.Rectangle((0, y0), paint_l, paint_w, fill=False, color="k", lw=1.2))
    ax.add_patch(plt.Rectangle((field_width_m - paint_l, y0), paint_l, paint_w, fill=False, color="k", lw=1.2))
    hoop_off = field_width_m * 0.06
    hoop_r = field_height_m * 0.04
    ax.add_patch(plt.Circle((hoop_off, field_height_m / 2.0), hoop_r, fill=False, color="k", lw=1.2))
    ax.add_patch(
        plt.Circle((field_width_m - hoop_off, field_height_m / 2.0), hoop_r, fill=False, color="k", lw=1.2),
    )

    drawn_players = 0
    for (cls_name, track_id), pts in sorted(trajectories.items(), key=lambda x: (x[0][0], x[0][1])):
        if not pts:
            continue
        is_ball = "ball" in cls_name.lower()
        if is_ball and include_ball_id is not None and track_id != include_ball_id:
            continue
        if (not is_ball) and include_player_ids is not None and track_id not in include_player_ids:
            continue
        x0, y0 = pts[0]
        x1, y1 = pts[-1]
        if not (np.isfinite(x0) and np.isfinite(y0) and np.isfinite(x1) and np.isfinite(y1)):
            continue
        x0 = float(np.clip(x0, 0.0, field_width_m))
        y0 = float(np.clip(y0, 0.0, field_height_m))
        x1 = float(np.clip(x1, 0.0, field_width_m))
        y1 = float(np.clip(y1, 0.0, field_height_m))

        if is_ball:
            # Optional: keep ball endpoint for context.
            ax.plot([x1], [y1], marker="o", color="k", markersize=5, linestyle="None")
            ax.text(x1 + 0.12, y1 - 0.12, "ball", fontsize=8, color="k")
        else:
            # Draw ALL players as X markers at their starting locations.
            team = (team_by_player_id or {}).get(track_id, "A")
            team_color = "#1f4e79" if team == "A" else "#8b2f2f"
            dx = 0.18
            dy = 0.18
            ax.plot([x0 - dx, x0 + dx], [y0 - dy, y0 + dy], color=team_color, lw=2.0)
            ax.plot([x0 - dx, x0 + dx], [y0 + dy, y0 - dy], color=team_color, lw=2.0)
            drawn_players += 1

        if draw_arrows and len(pts) >= 2:
            ls, rad = _movement_style(pts)
            if is_ball:
                color = "k"
            else:
                team = (team_by_player_id or {}).get(track_id, "A")
                color = "#1f4e79" if team == "A" else "#8b2f2f"
            ax.annotate(
                "",
                xy=(x1, y1),
                xytext=(x0, y0),
                arrowprops=dict(
                    arrowstyle="->",
                    color=color,
                    lw=1.4 if is_ball else 1.25,
                    linestyle=ls,
                    connectionstyle=f"arc3,rad={rad:.3f}",
                ),
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return drawn_players

