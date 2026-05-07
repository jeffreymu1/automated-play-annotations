"""Visualize marking-primitive fits and homography refinement.

Examples:
    uv run python tests/visualize_marking_refinement.py dataset --count 8
    uv run python tests/visualize_marking_refinement.py frames --frames results/dominican_v_mexico/frames --count 8
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from court_detection.geometry import MARKING_CLASS_BY_GEOMETRY, court_markings_world
from court_detection.marking_refinement import (
    MarkingHomographyResult,
    MarkingRefinementConfig,
    fit_homography_from_marking_heatmaps,
)
from court_detection.markings import (
    FIBA_MARKING_NAMES,
    FibaCourtMarkingDataModule,
    FibaCourtMarkingLightning,
    FibaStructuredSideCourtMarkingLightning,
    class_palette,
    overlay_line_predictions,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".avi", ".mkv", ".webm")
MODEL_ARCHITECTURES = {
    "dense-side": FibaCourtMarkingLightning,
    "structured-side": FibaStructuredSideCourtMarkingLightning,
}


def _default_checkpoint() -> Path:
    preferred = Path("checkpoints/006_fiba_court_markings_structured_side_1to1_pan/last.ckpt")
    if preferred.exists():
        return preferred
    candidates = sorted(Path("checkpoints").glob("**/fiba-court-markings-*.ckpt"))
    if candidates:
        return candidates[-1]
    return preferred


def _load_model(args: argparse.Namespace) -> tuple[FibaCourtMarkingLightning, torch.device]:
    model_cls = MODEL_ARCHITECTURES[args.architecture]
    model = model_cls.load_from_checkpoint(args.checkpoint, map_location="cpu", pretrained=False)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()
    return model, device


def _predict_full_resolution(
    model: FibaCourtMarkingLightning,
    image_rgb: np.ndarray,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_image = cv2.resize(image_rgb, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    image_t = torch.from_numpy(model_image).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        line_prob, class_probs, side_prob, court_prob = model.predict(image_t)
    out_size = image_rgb.shape[:2]
    line_full = F.interpolate(line_prob.unsqueeze(1), size=out_size, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    class_full = F.interpolate(class_probs, size=out_size, mode="bilinear", align_corners=False)[0].cpu().numpy()
    side_full = F.interpolate(side_prob.unsqueeze(1), size=out_size, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    court_full = F.interpolate(court_prob.unsqueeze(1), size=out_size, mode="bilinear", align_corners=False)[0, 0].cpu().numpy()
    return line_full, class_full, side_full, court_full


def _line_names(model: FibaCourtMarkingLightning, num_classes: int) -> tuple[str, ...]:
    return tuple(getattr(model, "line_names", FIBA_MARKING_NAMES))[:num_classes]


def _make_config(args: argparse.Namespace) -> MarkingRefinementConfig:
    return MarkingRefinementConfig(
        line_threshold=args.line_threshold,
        class_threshold=args.class_threshold,
        joint_threshold=args.joint_threshold,
        court_threshold=args.court_threshold,
        min_component_pixels=args.min_component_pixels,
        max_components_per_class=args.max_components_per_class,
        side_margin=args.side_margin,
        require_winning_class=not args.allow_nonwinning_class,
        min_geometry_evidence_mass=args.min_geometry_evidence_mass,
        min_geometry_evidence_mass_per_megapixel=args.min_geometry_evidence_mass_per_megapixel,
        curve_geometry_mass_multiplier=args.curve_geometry_mass_multiplier,
        min_component_evidence_fraction=args.min_component_evidence_fraction,
        min_component_evidence_mass=args.min_component_evidence_mass,
        ransac_iter=args.ransac_iter,
        use_torch_ransac=args.torch_ransac,
        min_scored_markings=args.min_scored_markings,
        require_baseline=not args.allow_no_baseline,
        enable_nonlinear_refinement=args.enable_nonlinear_refinement,
        seed=args.seed,
    )


def _result_json(result: MarkingHomographyResult) -> dict:
    return {
        "success": result.success,
        "message": result.message,
        "score": result.score,
        "homography": None if result.H is None else result.H.tolist(),
        "candidate_pixels": result.candidate_pixels,
        "geometry_masses": result.geometry_masses,
        "per_geometry_scores": result.per_geometry_scores,
        "point_correspondences": [
            {
                "name": name,
                "world": world.tolist(),
                "image": image.tolist(),
            }
            for name, world, image in result.point_correspondences
        ],
        "selected_lines": {
            name: {
                "class_name": fit.class_name,
                "geometry_name": fit.geometry_name,
                "line_homog": fit.line_homog.tolist(),
                "p0": fit.p0.tolist(),
                "p1": fit.p1.tolist(),
                "support": fit.support,
                "score": fit.score,
            }
            for name, fit in result.selected_lines.items()
        },
        "line_fits": {
            class_name: [
                {
                    "geometry_name": fit.geometry_name,
                    "p0": fit.p0.tolist(),
                    "p1": fit.p1.tolist(),
                    "support": fit.support,
                    "score": fit.score,
                }
                for fit in fits
            ]
            for class_name, fits in result.line_fits.items()
        },
        "curve_fits": {
            class_name: [
                {
                    "geometry_name": fit.geometry_name,
                    "conic": fit.conic.tolist(),
                    "circle_center": fit.circle_center.tolist(),
                    "circle_radius": fit.circle_radius,
                    "support": fit.support,
                    "score": fit.score,
                }
                for fit in fits
            ]
            for class_name, fits in result.curve_fits.items()
        },
    }


def _save_panel(
    out_path: Path,
    image_rgb: np.ndarray,
    line_prob: np.ndarray,
    class_probs: np.ndarray,
    side_prob: np.ndarray,
    court_prob: np.ndarray,
    result: MarkingHomographyResult,
    palette: np.ndarray,
    marking_names: tuple[str, ...],
    title: str,
) -> None:
    court_mask = np.clip(court_prob, 0.0, 1.0)
    pred_overlay = overlay_line_predictions(image_rgb, class_probs * court_mask[None], line_prob * court_mask, palette)
    primitive_overlay = _primitive_overlay(image_rgb, result, palette, marking_names)
    template_overlay = _template_overlay(image_rgb, result, palette, marking_names)
    side_masked = np.ma.masked_where(court_prob < 0.5, side_prob)
    side_cmap = plt.get_cmap("coolwarm").copy()
    side_cmap.set_bad(color="black")

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    panels = [
        (axes[0, 0], image_rgb, "Input", None),
        (axes[0, 1], pred_overlay, "Prediction overlay", None),
        (axes[0, 2], side_masked, "Court-masked sidedness", side_cmap),
        (axes[1, 0], line_prob, "Lineness", "inferno"),
        (axes[1, 1], primitive_overlay, "Fitted lines / curves", None),
        (axes[1, 2], template_overlay, f"Homography score {result.score:.3f}", None),
    ]
    for ax, image, panel_title, cmap in panels:
        if image.ndim == 2:
            ax.imshow(image, cmap=cmap or "inferno", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(panel_title)
        ax.axis("off")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _primitive_overlay(
    image_rgb: np.ndarray,
    result: MarkingHomographyResult,
    palette: np.ndarray,
    marking_names: tuple[str, ...],
) -> np.ndarray:
    overlay = np.clip(image_rgb * 255.0, 0.0, 255.0).astype(np.uint8).copy()
    for class_name, fits in result.line_fits.items():
        class_id = marking_names.index(class_name)
        color = tuple(int(255 * c) for c in palette[class_id][::-1])
        for fit in fits:
            p0 = np.round(fit.p0).astype(int)
            p1 = np.round(fit.p1).astype(int)
            cv2.line(overlay, tuple(p0), tuple(p1), color, 2, cv2.LINE_AA)
    for class_name, fits in result.curve_fits.items():
        class_id = marking_names.index(class_name)
        color = tuple(int(255 * c) for c in palette[class_id][::-1])
        for fit in fits:
            ellipse = _conic_to_cv2_ellipse(fit.conic)
            if ellipse is not None:
                center, axes, angle = ellipse
                cv2.ellipse(overlay, center, axes, angle, 0.0, 360.0, color, 2, cv2.LINE_AA)
    _draw_dlt_points(overlay, result)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _draw_dlt_points(overlay: np.ndarray, result: MarkingHomographyResult) -> None:
    for name, _, image in result.point_correspondences:
        if not np.isfinite(image).all():
            continue
        x, y = np.round(image).astype(int)
        if not (-20 <= x <= overlay.shape[1] + 20 and -20 <= y <= overlay.shape[0] + 20):
            continue
        if "tangent_apex" in name:
            color = (255, 0, 255)
        elif "three_point_arc" in name or "free_throw_circle" in name:
            color = (255, 255, 0)
        else:
            color = (0, 255, 255)
        cv2.circle(overlay, (x, y), 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 5, color, -1, cv2.LINE_AA)
        cv2.circle(overlay, (x, y), 8, (255, 255, 255), 1, cv2.LINE_AA)


def _template_overlay(
    image_rgb: np.ndarray,
    result: MarkingHomographyResult,
    palette: np.ndarray,
    marking_names: tuple[str, ...],
) -> np.ndarray:
    overlay = np.clip(image_rgb * 255.0, 0.0, 255.0).astype(np.uint8).copy()
    if result.H is not None:
        for geometry_name, world_xyz in court_markings_world(n=240).items():
            class_name = MARKING_CLASS_BY_GEOMETRY[geometry_name]
            if class_name not in marking_names:
                continue
            class_id = marking_names.index(class_name)
            pts = _project(result.H, world_xyz[:, :2])
            finite = np.isfinite(pts).all(axis=1) & (np.abs(pts).max(axis=1) < 1e7)
            pix = np.round(pts[finite]).astype(np.int32)
            if len(pix) >= 2:
                color = tuple(int(255 * c) for c in palette[class_id][::-1])
                cv2.polylines(overlay, [pix.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)
    return cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _project(H: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    homog = np.column_stack([world_xy, np.ones(len(world_xy))])
    img_h = (H @ homog.T).T
    return img_h[:, :2] / img_h[:, 2:3]


def _conic_to_cv2_ellipse(conic: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], float] | None:
    A = float(conic[0, 0])
    B = float(2.0 * conic[0, 1])
    C = float(conic[1, 1])
    D = float(2.0 * conic[0, 2])
    E = float(2.0 * conic[1, 2])
    F0 = float(conic[2, 2])
    Q = np.array([[A, B / 2.0], [B / 2.0, C]], dtype=float)
    rhs = -0.5 * np.array([D, E], dtype=float)
    try:
        center = np.linalg.solve(Q, rhs)
    except np.linalg.LinAlgError:
        return None
    translated_f = float(center @ Q @ center + D * center[0] + E * center[1] + F0)
    evals, evecs = np.linalg.eigh(Q)
    if not np.isfinite(evals).all() or translated_f == 0.0:
        return None
    axis_sq = -translated_f / evals
    if not np.isfinite(axis_sq).all() or np.any(axis_sq <= 0.0):
        return None
    radii = np.sqrt(axis_sq)
    order = np.argsort(radii)[::-1]
    radii = radii[order]
    vec = evecs[:, order[0]]
    angle = float(np.degrees(np.arctan2(vec[1], vec[0])))
    if np.max(radii) > 10000.0:
        return None
    return (
        tuple(np.round(center).astype(int)),
        tuple(np.round(radii).astype(int)),
        angle,
    )


def visualize_dataset(args: argparse.Namespace) -> None:
    model, device = _load_model(args)
    image_size = (args.image_height, args.image_width)
    marking_names = _line_names(model, int(model.hparams.num_classes))
    palette = class_palette(len(marking_names))
    dm = FibaCourtMarkingDataModule(
        root=args.root,
        image_size=image_size,
        output_stride=int(model.hparams.output_stride),
        sigma=float(model.hparams.sigma),
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dm.setup("test")
    dataset = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}[args.split]
    args.out.mkdir(parents=True, exist_ok=True)
    config = _make_config(args)
    count = min(args.count, len(dataset))
    for i in range(count):
        sample = dataset[i]
        image_rgb = sample["image"].permute(1, 2, 0).numpy()
        line_prob, class_probs, side_prob, court_prob = _predict_full_resolution(model, image_rgb, image_size, device)
        result = fit_homography_from_marking_heatmaps(
            image_rgb, line_prob, class_probs, side_prob, court_prob, config, marking_names
        )
        stem = f"dataset_{args.split}_{i:03d}"
        _save_panel(
            args.out / f"{stem}_refinement.png",
            image_rgb,
            line_prob,
            class_probs,
            side_prob,
            court_prob,
            result,
            palette,
            marking_names,
            stem,
        )
        (args.out / f"{stem}_refinement.json").write_text(json.dumps(_result_json(result), indent=2))
        print(f"{stem}: success={result.success} score={result.score:.3f}")


def visualize_frames(args: argparse.Namespace) -> None:
    model, device = _load_model(args)
    image_size = (args.image_height, args.image_width)
    marking_names = _line_names(model, int(model.hparams.num_classes))
    palette = class_palette(len(marking_names))
    config = _make_config(args)
    args.out.mkdir(parents=True, exist_ok=True)
    paths = _iter_media(args.frames, args.recursive)
    written = 0
    for path in paths:
        if written >= args.count:
            break
        frame_iter = [(None, _read_image(path))] if path.suffix.lower() in IMAGE_EXTENSIONS else _iter_video_frames(path, args.video_stride)
        for frame_idx, image_rgb in frame_iter:
            if written >= args.count:
                break
            line_prob, class_probs, side_prob, court_prob = _predict_full_resolution(model, image_rgb, image_size, device)
            result = fit_homography_from_marking_heatmaps(
                image_rgb, line_prob, class_probs, side_prob, court_prob, config, marking_names
            )
            stem = _safe_name(path.stem if frame_idx is None else f"{path.stem}_frame_{frame_idx:06d}")
            _save_panel(
                args.out / f"{stem}_refinement.png",
                image_rgb,
                line_prob,
                class_probs,
                side_prob,
                court_prob,
                result,
                palette,
                marking_names,
                stem,
            )
            (args.out / f"{stem}_refinement.json").write_text(json.dumps(_result_json(result), indent=2))
            written += 1
            print(f"frame {written}/{args.count}: {stem} success={result.success} score={result.score:.3f}")


def _read_image(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _iter_video_frames(path: Path, stride: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    frame_idx = 0
    stride = max(1, int(stride))
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if frame_idx % stride == 0:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            yield frame_idx, rgb
        frame_idx += 1
    cap.release()


def _iter_media(root: Path, recursive: bool) -> list[Path]:
    if root.is_file():
        return [root]
    paths = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in paths if path.suffix.lower() in IMAGE_EXTENSIONS + VIDEO_EXTENSIONS)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "frame"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--checkpoint", type=Path, default=_default_checkpoint())
        p.add_argument("--architecture", choices=tuple(MODEL_ARCHITECTURES), default="structured-side")
        p.add_argument("--out", type=Path, default=Path("tests/output/marking_refinement_vis"))
        p.add_argument("--count", type=int, default=8)
        p.add_argument("--image-height", type=int, default=384)
        p.add_argument("--image-width", type=int, default=640)
        p.add_argument("--cpu", action="store_true")
        p.add_argument("--seed", type=int, default=1430)
        p.add_argument("--line-threshold", type=float, default=0.50)
        p.add_argument("--class-threshold", type=float, default=0.35)
        p.add_argument("--joint-threshold", type=float, default=0.20)
        p.add_argument("--court-threshold", type=float, default=0.20)
        p.add_argument("--min-component-pixels", type=int, default=40)
        p.add_argument("--max-components-per-class", type=int, default=4)
        p.add_argument("--side-margin", type=float, default=0.08)
        p.add_argument("--allow-nonwinning-class", action="store_true")
        p.add_argument("--min-geometry-evidence-mass", type=float, default=120.0)
        p.add_argument("--min-geometry-evidence-mass-per-megapixel", type=float, default=300.0)
        p.add_argument("--curve-geometry-mass-multiplier", type=float, default=2.0)
        p.add_argument("--min-component-evidence-fraction", type=float, default=0.10)
        p.add_argument("--min-component-evidence-mass", type=float, default=80.0)
        p.add_argument("--ransac-iter", type=int, default=1200)
        p.add_argument("--torch-ransac", action="store_true")
        p.add_argument("--min-scored-markings", type=int, default=5)
        p.add_argument("--allow-no-baseline", action="store_true")
        p.add_argument("--enable-nonlinear-refinement", action="store_true")

    dataset_parser = subparsers.add_parser("dataset")
    common(dataset_parser)
    dataset_parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    dataset_parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    dataset_parser.add_argument("--num-workers", type=int, default=0)
    dataset_parser.set_defaults(func=visualize_dataset)

    frames_parser = subparsers.add_parser("frames")
    common(frames_parser)
    frames_parser.add_argument("--frames", type=Path, required=True)
    frames_parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    frames_parser.add_argument("--video-stride", type=int, default=60)
    frames_parser.set_defaults(func=visualize_frames)
    return parser


def main() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
