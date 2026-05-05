"""Debug GPU-batched structured homography RANSAC on images or image folders."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from court_detection.geometry import LINE_NAMES, court_lines_world, sample_segment
from court_detection.lines import CourtLineLightning, class_palette, overlay_line_predictions
from court_detection.structured_refinement import StructuredHomographyResult, project_world_points
from court_detection.structured_refinement_gpu import (
    StructuredGpuRansacConfig,
    fit_homography_from_heatmaps_gpu,
)


def _load_plain_image(path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    full_image = image_rgb.astype(np.float32) / 255.0
    model_image = cv2.resize(full_image, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    return full_image, model_image


def _iter_image_folder(folder: Path, image_glob: str) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    images = sorted(path for path in folder.glob(image_glob) if path.is_file())
    if not images:
        raise RuntimeError(f"No images matched {image_glob!r} in {folder}")
    return images


def _load_model(args: argparse.Namespace) -> tuple[CourtLineLightning, torch.device]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = CourtLineLightning.load_from_checkpoint(args.checkpoint, map_location="cpu", pretrained=False)
    model.to(device).eval()
    return model, device


def _predict_on_image(
    model: CourtLineLightning,
    image_np: np.ndarray,
    device: torch.device,
    model_image_size: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    model_h, model_w = model_image_size
    model_input = cv2.resize(image_np, (model_w, model_h), interpolation=cv2.INTER_AREA)
    image_t = torch.from_numpy(model_input).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        line_prob, class_probs, _ = model.predict(image_t)
        line_full = F.interpolate(
            line_prob.unsqueeze(1), size=image_np.shape[:2], mode="bilinear", align_corners=False
        )[0, 0]
        class_full = F.interpolate(
            class_probs, size=image_np.shape[:2], mode="bilinear", align_corners=False
        )[0]
    return line_full, class_full


def _make_config(args: argparse.Namespace) -> StructuredGpuRansacConfig:
    return StructuredGpuRansacConfig(
        ransac_iter=args.structured_ransac_iter,
        line_threshold=args.structured_line_threshold,
        class_threshold=args.structured_class_threshold,
        joint_threshold=args.structured_joint_threshold,
        min_pixels_per_class=args.min_pixels_per_class,
        line_refine_distance_px=args.line_refine_distance_px,
        max_refine_candidates=args.max_refine_candidates,
        n_samples_per_line=args.n_samples_per_line,
        score_radius_px=args.score_radius_px,
        min_scored_lines=args.min_scored_lines,
        require_baseline_inlier=not args.allow_no_baseline_inlier,
        seed=args.seed,
    )


def _json_mat(H: np.ndarray | None) -> list[list[float]] | None:
    return None if H is None else np.asarray(H, dtype=float).tolist()


def _result_diagnostics(result: StructuredHomographyResult, device: torch.device) -> dict:
    return {
        "success": result.success,
        "message": result.message,
        "device": device.type,
        "score": result.score,
        "homography": _json_mat(result.H),
        "candidate_pixels": result.candidate_pixels,
        "inlier_lines": list(result.inlier_lines),
        "inlier_corners": list(result.inlier_corners),
        "per_line_scores": result.per_line_scores,
        "sampled_lines": {
            name: {
                "line_homog": line.line_homog.tolist(),
                "p0": line.p0.tolist(),
                "p1": line.p1.tolist(),
                "support": line.support,
                "score": line.score,
            }
            for name, line in result.sampled_lines.items()
        },
        "corners": [
            {
                "name": corner.name,
                "world": corner.world.tolist(),
                "image": corner.image.tolist(),
                "line_a": corner.line_a,
                "line_b": corner.line_b,
            }
            for corner in result.corners
        ],
    }


def _save_heatmap_overlay(
    path: Path,
    image: np.ndarray,
    lineness: np.ndarray,
    class_probs: np.ndarray,
    palette: np.ndarray,
) -> None:
    _save_rgb(path, overlay_line_predictions(image, class_probs, lineness, palette))


def _save_line_debug_overlay(
    path: Path,
    image: np.ndarray,
    lineness: np.ndarray,
    class_probs: np.ndarray,
    result: StructuredHomographyResult,
    config: StructuredGpuRansacConfig,
    palette: np.ndarray,
) -> None:
    overlay = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8).copy()
    evidence = np.clip(class_probs * lineness[None], 0.0, 1.0)
    for class_id, name in enumerate(LINE_NAMES[: class_probs.shape[0]]):
        mask = (
            (lineness >= config.line_threshold)
            & (class_probs[class_id] >= config.class_threshold)
            & (evidence[class_id] >= config.joint_threshold)
        )
        if mask.any():
            color = np.array(255 * palette[class_id], dtype=np.uint8)
            overlay[mask] = (0.45 * overlay[mask] + 0.55 * color).astype(np.uint8)
    for line in result.sampled_lines.values():
        color = tuple(int(255 * c) for c in palette[line.class_id])
        p0 = np.round(line.p0).astype(int)
        p1 = np.round(line.p1).astype(int)
        cv2.line(overlay, tuple(p0), tuple(p1), color=color, thickness=3, lineType=cv2.LINE_AA)
    for corner in result.corners:
        x, y = np.round(corner.image).astype(int)
        cv2.circle(overlay, (x, y), 6, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    _save_rgb(path, overlay.astype(np.float32) / 255.0)


def _save_template_overlay(
    path: Path,
    image: np.ndarray,
    result: StructuredHomographyResult,
    palette: np.ndarray,
) -> None:
    overlay = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8).copy()
    if result.H is not None:
        for class_id, name in enumerate(LINE_NAMES):
            if name not in court_lines_world():
                continue
            a, b = court_lines_world()[name]
            world = sample_segment(a[:2], b[:2], n=500)
            img = project_world_points(result.H, world)
            finite = np.isfinite(img).all(axis=1) & (np.abs(img).max(axis=1) < 1.0e7)
            pts = np.round(img[finite]).astype(np.int32)
            if len(pts) >= 2:
                color = tuple(int(255 * c) for c in palette[class_id][::-1])
                cv2.polylines(overlay, [pts.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _save_rgb(path: Path, image: np.ndarray) -> None:
    out = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def _process_image(
    image_path: Path,
    args: argparse.Namespace,
    model: CourtLineLightning,
    device: torch.device,
    palette: np.ndarray,
) -> dict:
    image_size = (args.image_height, args.image_width)
    full_image, _ = _load_plain_image(image_path, image_size)
    line_prob_t, class_probs_t = _predict_on_image(model, full_image, device, image_size)
    config = _make_config(args)
    result = fit_homography_from_heatmaps_gpu(full_image, line_prob_t, class_probs_t, config, device=device)

    line_prob = line_prob_t.detach().cpu().numpy()
    class_probs = class_probs_t.detach().cpu().numpy()
    stem = args.name if args.name and args.image is not None else image_path.stem
    heatmap_path = args.out / f"{stem}_gpu_heatmap.png"
    line_debug_path = args.out / f"{stem}_gpu_structured_lines.png"
    template_path = args.out / f"{stem}_gpu_structured_template.png"
    json_path = args.out / f"{stem}_gpu_structured.json"
    _save_heatmap_overlay(heatmap_path, full_image, line_prob, class_probs, palette)
    _save_line_debug_overlay(line_debug_path, full_image, line_prob, class_probs, result, config, palette)
    _save_template_overlay(template_path, full_image, result, palette)

    frame_diag = _result_diagnostics(result, device)
    frame_diag.update(
        {
            "image_path": str(image_path),
            "heatmap_overlay_path": str(heatmap_path),
            "line_debug_path": str(line_debug_path),
            "template_overlay_path": str(template_path),
            "image_width": int(full_image.shape[1]),
            "image_height": int(full_image.shape[0]),
        }
    )
    json_path.write_text(json.dumps(frame_diag, indent=2))
    print(
        f"{image_path.name}: gpu_structured={result.success} score={result.score:.4f} "
        f"device={device.type}; wrote {template_path.name}"
    )
    return frame_diag


def run(args: argparse.Namespace) -> None:
    if args.image is None and args.image_folder is None:
        raise ValueError("Provide --image or --image-folder")
    if args.image is not None and args.image_folder is not None:
        raise ValueError("--image and --image-folder are mutually exclusive")
    args.out.mkdir(parents=True, exist_ok=True)
    model, device = _load_model(args)
    palette = class_palette(len(LINE_NAMES))
    image_paths = [args.image] if args.image is not None else _iter_image_folder(args.image_folder, args.image_glob)
    diagnostics = [_process_image(path, args, model, device, palette) for path in image_paths]
    payload = {
        "num_frames": len(image_paths),
        "settings": {
            "checkpoint": str(args.checkpoint),
            "image_size": [args.image_height, args.image_width],
            "device": device.type,
            "structured_ransac_iter": args.structured_ransac_iter,
            "structured_line_threshold": args.structured_line_threshold,
            "structured_class_threshold": args.structured_class_threshold,
            "structured_joint_threshold": args.structured_joint_threshold,
        },
        "frames": diagnostics,
    }
    folder_json = args.out / "gpu_structured_diagnostics.json"
    folder_json.write_text(json.dumps(payload, indent=2))
    print(f"wrote {folder_json}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, default=None)
    parser.add_argument("--image-folder", type=Path, default=None)
    parser.add_argument("--image-glob", default="*.jpg")
    parser.add_argument("--out", type=Path, default=Path("results/structured_homography_gpu_debug"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=1430)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--structured-ransac-iter", type=int, default=4096)
    parser.add_argument("--structured-line-threshold", type=float, default=0.55)
    parser.add_argument("--structured-class-threshold", type=float, default=0.55)
    parser.add_argument("--structured-joint-threshold", type=float, default=0.30)
    parser.add_argument("--min-pixels-per-class", type=int, default=8)
    parser.add_argument("--line-refine-distance-px", type=float, default=8.0)
    parser.add_argument("--max-refine-candidates", type=int, default=12000)
    parser.add_argument("--n-samples-per-line", type=int, default=80)
    parser.add_argument("--score-radius-px", type=int, default=3)
    parser.add_argument("--min-scored-lines", type=int, default=3)
    parser.add_argument("--allow-no-baseline-inlier", action="store_true")
    return parser


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
