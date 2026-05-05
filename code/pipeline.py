from collections import defaultdict, deque
from pathlib import Path

import cv2
from rich import print

from .config import PipelineConfig
from .court_inference import resolve_court_corners, try_auto_corners_update
from .detectors import YoloDetector
from .field import build_homography_from_corners
from .io_utils import video_writer_for
from .projection import project_image_point_to_field
from .tracking import CentroidTracker
from .visualize import draw_court_quad, draw_detections, save_field_plot, save_whiteboard_play


def list_frame_paths(input_dir: Path) -> list[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp"}
    return sorted(p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def run_pipeline(config: PipelineConfig) -> None:
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

    court_corners, court_src = resolve_court_corners(
        first_frame,
        config.court_mode,
        calibration_json=config.court_calibration_json,
        manual_corners=config.court_corners_manual,
    )
    print(f"[cyan]Court homography source:[/cyan] {court_src}")

    writer = video_writer_for(first_frame, config.output_video, fps)

    detector = YoloDetector(
        config.yolo_model,
        config.confidence,
        config.iou,
        class_substrings=config.yolo_class_substrings,
    )
    tracker = CentroidTracker()

    trajectories: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    image_trajectories: dict[tuple[str, int], list[tuple[float, float]]] = defaultdict(list)
    trail_len = max(2, config.trail_length)
    image_trails: dict[int, deque[tuple[int, int]]] = defaultdict(lambda: deque(maxlen=trail_len))
    track_class: dict[int, str] = {}
    track_last_seen: dict[int, int] = {}

    refresh = max(0, int(config.court_refresh_every))

    def process_frame(frame, frame_idx: int) -> None:
        nonlocal court_corners

        if (
            refresh > 0
            and frame_idx > 0
            and frame_idx % refresh == 0
            and config.court_mode == "auto"
        ):
            new_c = try_auto_corners_update(frame)
            if new_c is not None:
                court_corners = new_c

        homography = build_homography_from_corners(
            court_corners,
            config.field_width_m,
            config.field_height_m,
        )

        detections = detector.detect(frame)
        tracks = tracker.update(detections)
        for tr in tracks:
            track_class[tr.track_id] = tr.cls_name
            track_last_seen[tr.track_id] = frame_idx
            ix, iy = int(tr.position[0]), int(tr.position[1])
            image_trails[tr.track_id].append((ix, iy))
            image_trajectories[(tr.cls_name, tr.track_id)].append((float(ix), float(iy)))
            fx, fy = project_image_point_to_field(tr.position, homography)
            trajectories[(tr.cls_name, tr.track_id)].append((fx, fy))

        vis = frame.copy()
        if config.draw_court_overlay:
            vis = draw_court_quad(vis, court_corners)
        annotated = draw_detections(
            vis,
            detections,
            tracks,
            image_trails=image_trails,
            track_class=track_class,
        )
        writer.write(annotated)

    frame_idx = 0
    max_frames = config.max_frames

    if use_image_sequence:
        for p in paths:
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            if max_frames is not None and frame_idx >= max_frames:
                break
            process_frame(frame, frame_idx)
            frame_idx += 1
    else:
        assert cap is not None
        process_frame(first_frame, frame_idx)
        frame_idx += 1

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break
            process_frame(frame, frame_idx)
            frame_idx += 1
        cap.release()

    writer.release()
    save_field_plot(
        dict(trajectories),
        config.output_plot,
        config.field_width_m,
        config.field_height_m,
    )
    if config.output_whiteboard is not None:
        # Keep whiteboard readable: latest snapshot with up to max players.
        max_players = max(1, int(config.whiteboard_max_players))
        player_items: list[tuple[int, int, int, float]] = []
        ball_items: list[tuple[int, int, int]] = []
        for (cls_name, tid), pts in trajectories.items():
            if not pts:
                continue
            seen = track_last_seen.get(tid, -1)
            if "ball" in cls_name.lower():
                ball_items.append((seen, len(pts), tid))
            else:
                player_items.append((seen, len(pts), tid, pts[-1][0]))

        player_items.sort(key=lambda x: (x[0], x[1]), reverse=True)
        top_players = player_items[:max_players]
        selected_players = {t[2] for t in top_players}
        team_by_player_id: dict[int, str] = {}
        if top_players:
            by_x = sorted(top_players, key=lambda t: t[3])
            split = max(1, len(by_x) // 2)
            for _, _, tid, _ in by_x[:split]:
                team_by_player_id[tid] = "A"
            for _, _, tid, _ in by_x[split:]:
                team_by_player_id[tid] = "B"
        print(
            f"[cyan]Whiteboard players:[/cyan] selected {len(selected_players)} "
            f"(target {max_players})",
        )

        include_ball_id = None
        if ball_items:
            ball_items.sort(key=lambda x: (x[0], x[1]), reverse=True)
            include_ball_id = ball_items[0][2]

        # Build whiteboard trajectories from video-space start/end so movement matches clip better,
        # even when court homography is noisy.
        h_img, w_img = first_frame.shape[:2]
        wb_trajectories: dict[tuple[str, int], list[tuple[float, float]]] = {}
        selected_keys: list[tuple[str, int]] = []
        for (cls_name, tid), pts in image_trajectories.items():
            if "ball" in cls_name.lower():
                if include_ball_id is not None and tid == include_ball_id:
                    selected_keys.append((cls_name, tid))
            elif tid in selected_players:
                selected_keys.append((cls_name, tid))
        for key in selected_keys:
            pts = image_trajectories.get(key, [])
            if len(pts) < 1:
                continue
            start = pts[0]
            end = pts[-1]
            sx = (start[0] / max(1.0, float(w_img - 1))) * config.field_width_m
            sy = (start[1] / max(1.0, float(h_img - 1))) * config.field_height_m
            ex = (end[0] / max(1.0, float(w_img - 1))) * config.field_width_m
            ey = (end[1] / max(1.0, float(h_img - 1))) * config.field_height_m
            wb_trajectories[key] = [(sx, sy), (ex, ey)]

        drawn_players = save_whiteboard_play(
            wb_trajectories,
            config.output_whiteboard,
            config.field_width_m,
            config.field_height_m,
            draw_arrows=config.whiteboard_arrows,
            include_player_ids=selected_players,
            include_ball_id=include_ball_id,
            team_by_player_id=team_by_player_id,
        )
        print(
            f"[cyan]Whiteboard drawn:[/cyan] {drawn_players} players "
            f"(requested {len(selected_players)})",
        )
    print(f"[green]Done.[/green] Processed {frame_idx} frames.")
    print(f"Annotated video: {config.output_video}")
    print(f"2D field plot: {config.output_plot}")
    if config.output_whiteboard is not None:
        print(f"Whiteboard play: {config.output_whiteboard}")


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
    """Deprecated for internal use — prefer :func:`court_inference.resolve_court_corners` via CLI."""
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
