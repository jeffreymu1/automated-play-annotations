"""Run geometric court-line refinement and save stage-by-stage visualizations."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from court_detection.dataset import DeepSportDataset
from court_detection.geometry import LINE_NAMES, court_lines_world, sample_segment
from court_detection.lines import CourtLineDataModule, CourtLineLightning, class_palette, overlay_line_predictions
from court_detection.refinement import (
    RefinementConfig,
    Stage1Config,
    Stage2Config,
    Stage3Config,
    project_world_points,
    refine_frame,
)


def _json_mat(H: np.ndarray | None) -> list[list[float]] | None:
    return None if H is None else np.asarray(H, dtype=float).tolist()


def _result_diagnostics(result) -> dict:
    H = result.stage3.H if result.stage3.success and result.stage3.H is not None else result.stage2.H
    k1 = result.stage3.k1 if result.stage3.success else 0.0
    return {
        "homography": _json_mat(H),
        "distortion": {"k1": float(k1)},
        "stage1_lines": {
            name: {
                "p0": fit.p0.tolist(),
                "p1": fit.p1.tolist(),
                "line_homog": fit.line_homog.tolist(),
                "fit_residual": fit.fit_residual,
                "score": fit.score,
            }
            for name, fit in result.line_fits.items()
        },
        "stage2": {
            "success": result.stage2.success,
            "message": result.stage2.message,
            "H": _json_mat(result.stage2.H),
            "inlier_lines": list(result.stage2.inlier_lines),
            "inlier_corners": list(result.stage2.inlier_corners),
        },
        "stage3": {
            "success": result.stage3.success,
            "message": result.stage3.message,
            "H": _json_mat(result.stage3.H),
            "k1": result.stage3.k1,
            "corner_rms": result.stage3.corner_rms,
            "residuals_by_line": result.stage3.residuals_by_line,
        },
    }


def _predict_on_image(
    model: CourtLineLightning,
    image_np: np.ndarray,
    device: torch.device,
    model_image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    model_h, model_w = model_image_size
    if image_np.shape[:2] != model_image_size:
        model_input = cv2.resize(image_np, (model_w, model_h), interpolation=cv2.INTER_AREA)
    else:
        model_input = image_np
    image_t = torch.from_numpy(model_input).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        line_prob, class_probs = model.predict(image_t)
    image_size = image_np.shape[:2]
    line_full = F.interpolate(
        line_prob.unsqueeze(1), size=image_size, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    class_full = F.interpolate(
        class_probs, size=image_size, mode="bilinear", align_corners=False
    )[0].cpu().numpy()
    return line_full, class_full


def _load_dataset_frame(args: argparse.Namespace, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str]:
    dm = CourtLineDataModule(
        root=args.root,
        image_size=image_size,
        output_stride=args.output_stride,
        sigma=args.sigma,
        batch_size=1,
        num_workers=0,
        seed=args.seed,
    )
    dm.setup("test")
    dataset = dm.val_dataset if args.split == "val" else dm.test_dataset
    idx = min(max(args.index, 0), len(dataset) - 1)
    base_idx = dataset.indices[idx]
    base = DeepSportDataset(args.root)
    full_image, _, _ = base[base_idx]
    sample = dataset[idx]
    return (
        full_image.permute(1, 2, 0).numpy(),
        sample["image"].permute(1, 2, 0).numpy(),
        str(base.samples[base_idx][0]),
    )


def _load_plain_image(path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray, str]:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    full_image = image_rgb.astype(np.float32) / 255.0
    model_image = cv2.resize(full_image, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    return full_image, model_image, str(path)


def _draw_stage1(ax: plt.Axes, image: np.ndarray, result, palette: np.ndarray) -> None:
    ax.imshow(image)
    for fit in result.line_fits.values():
        color = palette[fit.class_id]
        ax.plot([fit.p0[0], fit.p1[0]], [fit.p0[1], fit.p1[1]], color=color, linewidth=2.5, label=fit.name)
        ax.scatter([fit.p0[0], fit.p1[0]], [fit.p0[1], fit.p1[1]], color=color, s=10)
    ax.set_title("Stage 1: fitted image lines")
    ax.axis("off")


def _draw_template(
    ax: plt.Axes,
    image: np.ndarray,
    H: np.ndarray | None,
    k1: float,
    title: str,
    palette: np.ndarray,
) -> None:
    ax.imshow(image)
    if H is not None:
        for class_id, name in enumerate(LINE_NAMES):
            if name not in court_lines_world():
                continue
            a, b = court_lines_world()[name]
            world = sample_segment(a[:2], b[:2], n=200)
            img = project_world_points(H, world, k1=k1, image_shape=image.shape[:2])
            finite = np.isfinite(img).all(axis=1)
            ax.plot(img[finite, 0], img[finite, 1], color=palette[class_id], linewidth=2.0)
    ax.set_title(title)
    ax.axis("off")


def _write_diagnostics(path: Path, result) -> None:
    path.write_text(json.dumps(_result_diagnostics(result), indent=2))


def _save_full_resolution_overlay(
    path: Path,
    image: np.ndarray,
    result,
    palette: np.ndarray,
    source_shape: tuple[int, int] | None = None,
) -> None:
    H = result.stage3.H if result.stage3.success and result.stage3.H is not None else result.stage2.H
    k1 = result.stage3.k1 if result.stage3.success else 0.0
    if H is not None:
        if source_shape is not None and source_shape != image.shape[:2]:
            sx = image.shape[1] / source_shape[1]
            sy = image.shape[0] / source_shape[0]
            H = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]]) @ H
            k1 = 0.0
    overlay = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8).copy()
    if H is not None:
        for class_id, name in enumerate(LINE_NAMES):
            if name not in court_lines_world():
                continue
            a, b = court_lines_world()[name]
            world = sample_segment(a[:2], b[:2], n=500)
            img = project_world_points(H, world, k1=k1, image_shape=image.shape[:2])
            finite = np.isfinite(img).all(axis=1) & (np.abs(img).max(axis=1) < 1.0e7)
            pts = np.round(img[finite]).astype(np.int32)
            if len(pts) >= 2:
                color = tuple(int(255 * c) for c in palette[class_id][::-1])
                cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _make_config(args: argparse.Namespace, scale: float) -> RefinementConfig:
    return RefinementConfig(
        stage1=Stage1Config(
            pixel_threshold=args.pixel_threshold,
            min_pixels_per_class=args.min_pixels_per_class,
            ransac_iter=args.stage1_ransac_iter,
            ransac_distance_threshold=args.ransac_distance_threshold * scale,
            irls_iter=args.irls_iter,
            huber_threshold=2.0 * scale,
            early_stop_shift=0.1 * scale,
            log_sigma=1.5 * scale,
            use_class_aware_filter=args.use_class_aware_filter,
            seed=args.seed,
        ),
        stage2=Stage2Config(
            ransac_iter=args.stage2_ransac_iter,
            line_distance_threshold=args.line_distance_threshold * scale,
            corner_distance_threshold=args.corner_distance_threshold * scale,
            max_stage1_residual=args.max_stage1_residual * scale,
            min_segment_length_px=args.min_segment_length_px * scale,
            max_corner_segment_overshoot_px=80.0 * scale,
            seed=args.seed,
        ),
        stage3=Stage3Config(
            enabled=not args.no_stage3,
            n_samples_per_line=args.stage3_samples,
            rms_reprojection_max=5.0 * scale,
        ),
    )


def _safe_folder_name(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "clip"


def _clip_output_name(game: str, segment: str, camera: str) -> str:
    return _safe_folder_name(f"{game}_{segment}_{camera}")


def _select_refine_image(
    args: argparse.Namespace,
    full_image_np: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, tuple[int, int] | None]:
    if args.refine_resolution == "full":
        return full_image_np, None
    model_image = cv2.resize(full_image_np, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    return model_image, model_image.shape[:2]


def _load_model(args: argparse.Namespace) -> tuple[CourtLineLightning, torch.device]:
    model = CourtLineLightning.load_from_checkpoint(args.checkpoint, map_location="cpu", pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()
    return model, device


def _run_ffmpeg(frame_pattern: Path, output_path: Path, fps: float) -> None:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frame_pattern),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed:\n{exc.stderr}") from exc


def _iter_image_folder(folder: Path, image_glob: str) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected an image folder: {folder}")
    images = sorted(path for path in folder.glob(image_glob) if path.is_file())
    if not images:
        raise RuntimeError(f"No images matched {image_glob!r} in {folder}")
    return images


def visualize(args: argparse.Namespace) -> None:
    image_size = (args.image_height, args.image_width)
    if args.image is None:
        full_image_np, image_np, source_name = _load_dataset_frame(args, image_size)
    else:
        full_image_np, image_np, source_name = _load_plain_image(args.image, image_size)

    model, device = _load_model(args)
    if args.refine_resolution == "full":
        refine_image = full_image_np
        overlay_source_shape = None
    else:
        refine_image = image_np
        overlay_source_shape = image_np.shape[:2]
    line_prob, class_probs = _predict_on_image(model, refine_image, device, image_size)

    scale = max(refine_image.shape[1] / image_size[1], refine_image.shape[0] / image_size[0])
    config = _make_config(args, scale)
    result = refine_frame(refine_image, line_prob, class_probs, config)

    args.out.mkdir(parents=True, exist_ok=True)
    palette = class_palette(len(LINE_NAMES))
    heatmap_overlay = overlay_line_predictions(refine_image, class_probs, line_prob, palette)

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes[0, 0].imshow(heatmap_overlay)
    axes[0, 0].set_title("Network line heatmaps")
    axes[0, 0].axis("off")
    _draw_stage1(axes[0, 1], refine_image, result, palette)
    _draw_template(
        axes[1, 0], refine_image, result.stage2.H, 0.0,
        f"Stage 2: homography ({'ok' if result.stage2.success else 'failed'})", palette,
    )
    _draw_template(
        axes[1, 1], refine_image, result.stage3.H, result.stage3.k1,
        f"Stage 3: H + k1={result.stage3.k1:.2e}", palette,
    )
    fig.suptitle(source_name, fontsize=10)
    fig.tight_layout()

    stem = args.name or f"refinement_{args.index:03d}"
    fig_path = args.out / f"{stem}.png"
    json_path = args.out / f"{stem}.json"
    full_overlay_path = args.out / f"{stem}_full_overlay.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    _write_diagnostics(json_path, result)
    _save_full_resolution_overlay(full_overlay_path, full_image_np, result, palette, overlay_source_shape)
    print(f"wrote {fig_path}")
    print(f"wrote {json_path}")
    print(f"wrote {full_overlay_path}")
    print(f"stage1 lines: {', '.join(result.line_fits) or 'none'}")
    print(f"stage2: {result.stage2.success} {result.stage2.message}")
    print(f"stage3: {result.stage3.success} {result.stage3.message}")


def visualize_image_folder(args: argparse.Namespace) -> None:
    image_size = (args.image_height, args.image_width)
    image_paths = _iter_image_folder(args.image_folder, args.image_glob)

    model, device = _load_model(args)
    palette = class_palette(len(LINE_NAMES))
    args.out.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict] = []

    for i, image_path in enumerate(image_paths):
        full_image_np, image_np, source_name = _load_plain_image(image_path, image_size)
        refine_image, overlay_source_shape = _select_refine_image(args, full_image_np, image_size)
        line_prob, class_probs = _predict_on_image(model, refine_image, device, image_size)
        scale = max(refine_image.shape[1] / image_size[1], refine_image.shape[0] / image_size[0])
        config = _make_config(args, scale)
        result = refine_frame(refine_image, line_prob, class_probs, config)

        overlay_path = args.out / f"{image_path.stem}_full_overlay.png"
        _save_full_resolution_overlay(overlay_path, full_image_np, result, palette, overlay_source_shape)

        frame_diag = _result_diagnostics(result)
        frame_diag.update(
            {
                "frame_index": i,
                "image_path": str(image_path),
                "source_name": source_name,
                "overlay_path": str(overlay_path),
                "image_width": int(full_image_np.shape[1]),
                "image_height": int(full_image_np.shape[0]),
            }
        )
        diagnostics.append(frame_diag)
        print(
            f"frame {i + 1}/{len(image_paths)}: wrote {overlay_path.name}; "
            f"stage2={result.stage2.success} stage3={result.stage3.success}"
        )

    json_path = args.out / "diagnostics.json"
    payload = {
        "image_folder": str(args.image_folder),
        "image_glob": args.image_glob,
        "num_frames": len(image_paths),
        "settings": {
            "checkpoint": str(args.checkpoint),
            "refine_resolution": args.refine_resolution,
            "image_size": list(image_size),
        },
        "frames": diagnostics,
    }
    json_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {json_path}")


def visualize_sequence(args: argparse.Namespace) -> None:
    if args.image is not None or args.image_folder is not None:
        raise ValueError("--image/--image-folder cannot be combined with sequence mode")

    image_size = (args.image_height, args.image_width)
    base = DeepSportDataset(args.root)
    clip_key: int | str = args.clip if args.clip is not None else args.index
    if isinstance(clip_key, str) and clip_key.isdigit():
        clip_key = int(clip_key)
    clip = base.get_clip(clip_key)
    cameras = clip.cameras
    if not cameras:
        raise RuntimeError(f"No camera streams found for clip {clip.name}")
    camera = args.camera if args.camera is not None else cameras[0]
    frames = base.load_clip_frames(clip_key, camera=camera)
    if not frames:
        raise RuntimeError(f"No frames found for clip {clip.name}, camera {camera}")

    model, device = _load_model(args)
    palette = class_palette(len(LINE_NAMES))
    clip_dir = args.out / _clip_output_name(clip.game, clip.segment, camera)
    clip_dir.mkdir(parents=True, exist_ok=True)
    diagnostics: list[dict] = []

    for i, (full_image_np, frame, calib) in enumerate(frames):
        refine_image, overlay_source_shape = _select_refine_image(args, full_image_np, image_size)
        line_prob, class_probs = _predict_on_image(model, refine_image, device, image_size)
        scale = max(refine_image.shape[1] / image_size[1], refine_image.shape[0] / image_size[0])
        config = _make_config(args, scale)
        result = refine_frame(refine_image, line_prob, class_probs, config)

        overlay_path = clip_dir / f"frame_{i:04d}.png"
        _save_full_resolution_overlay(overlay_path, full_image_np, result, palette, overlay_source_shape)

        frame_diag = _result_diagnostics(result)
        frame_diag.update(
            {
                "frame_index": i,
                "frame_number": frame.frame_number,
                "timestamp": frame.timestamp,
                "camera": frame.camera,
                "image_path": str(frame.image_path),
                "json_path": str(frame.json_path),
                "overlay_path": str(overlay_path),
                "image_width": int(calib.width),
                "image_height": int(calib.height),
            }
        )
        diagnostics.append(frame_diag)
        print(
            f"frame {i + 1}/{len(frames)}: wrote {overlay_path.name}; "
            f"stage2={result.stage2.success} stage3={result.stage3.success}"
        )

    json_path = clip_dir / "diagnostics.json"
    payload = {
        "clip": {
            "game": clip.game,
            "segment": clip.segment,
            "name": clip.name,
            "camera": camera,
            "available_cameras": list(cameras),
            "json_paths": [str(frame.json_path) for _, frame, _ in frames],
            "num_frames": len(frames),
        },
        "settings": {
            "checkpoint": str(args.checkpoint),
            "refine_resolution": args.refine_resolution,
            "image_size": list(image_size),
            "fps": args.fps,
        },
        "frames": diagnostics,
    }
    json_path.write_text(json.dumps(payload, indent=2))

    mp4_path = clip_dir / f"{_clip_output_name(clip.game, clip.segment, camera)}.mp4"
    _run_ffmpeg(clip_dir / "frame_%04d.png", mp4_path, args.fps)
    print(f"wrote {json_path}")
    print(f"wrote {mp4_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    parser.add_argument("--image", type=Path, default=None, help="Optional image path instead of a DeepSport split frame.")
    parser.add_argument("--image-folder", type=Path, default=None, help="Optional folder of images to refine in filename order.")
    parser.add_argument("--image-glob", default="*.jpg", help="Glob used with --image-folder.")
    parser.add_argument("--out", type=Path, default=Path("results/line_refinement"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sequence", action="store_true", help="Run refinement on every frame in a DeepSport clip.")
    parser.add_argument("--clip", default=None, help="Clip index, clip name, or game/segment/name. Defaults to --index in sequence mode.")
    parser.add_argument("--camera", default=None, help="Camera stream within a clip, such as camcourt1. Defaults to the first camera.")
    parser.add_argument("--fps", type=float, default=10.0, help="Frame rate for sequence mp4 output.")
    parser.add_argument("--seed", type=int, default=1430)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--output-stride", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--refine-resolution", choices=("full", "model"), default="full")
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--pixel-threshold", type=float, default=0.3)
    parser.add_argument("--min-pixels-per-class", type=int, default=30)
    parser.add_argument("--stage1-ransac-iter", type=int, default=100)
    parser.add_argument("--ransac-distance-threshold", type=float, default=3.0)
    parser.add_argument("--irls-iter", type=int, default=5)
    parser.add_argument("--use-class-aware-filter", action=argparse.BooleanOptionalAction, default=True)

    parser.add_argument("--stage2-ransac-iter", type=int, default=200)
    parser.add_argument("--line-distance-threshold", type=float, default=4.0)
    parser.add_argument("--corner-distance-threshold", type=float, default=5.0)
    parser.add_argument("--max-stage1-residual", type=float, default=30.0)
    parser.add_argument("--min-segment-length-px", type=float, default=25.0)

    parser.add_argument("--no-stage3", action="store_true")
    parser.add_argument("--stage3-samples", type=int, default=50)
    parser.set_defaults(func=visualize)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.image is not None and args.image_folder is not None:
        raise ValueError("--image and --image-folder are mutually exclusive")
    if args.sequence or args.clip is not None:
        visualize_sequence(args)
    elif args.image_folder is not None:
        visualize_image_folder(args)
    else:
        args.func(args)


if __name__ == "__main__":
    main()
