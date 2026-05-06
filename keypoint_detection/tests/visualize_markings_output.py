"""Visualize predictions from the FIBA court-marking detector.

Examples:
    uv run python tests/visualize_markings_output.py dataset \
        --checkpoint checkpoints/fiba_court_markings/last.ckpt --count 8

    uv run python tests/visualize_markings_output.py frames \
        --checkpoint checkpoints/fiba_court_markings/last.ckpt --frames test_footage
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from court_detection.geometry import (
    COURT_LENGTH_CM,
    COURT_WIDTH_CM,
    MARKING_CLASS_BY_GEOMETRY,
    court_markings_world,
    project_world_to_image,
)
from court_detection.lines import _render_line_targets, _render_side_target
from court_detection.markings import (
    FIBA_MARKING_NAMES,
    FibaCourtMarkingDataModule,
    FibaCourtMarkingLightning,
    class_palette,
    overlay_line_predictions,
)
from court_detection.midcourt_stitch_dataset import DeepSportMidcourtStitchDataset

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def _load_model(args: argparse.Namespace) -> tuple[FibaCourtMarkingLightning, torch.device]:
    model = FibaCourtMarkingLightning.load_from_checkpoint(
        args.checkpoint,
        map_location="cpu",
        pretrained=False,
    )
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

    output_size = image_rgb.shape[:2]
    line_full = F.interpolate(
        line_prob.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    class_full = F.interpolate(
        class_probs,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0].cpu().numpy()
    side_full = F.interpolate(
        side_prob.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    court_full = F.interpolate(
        court_prob.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    return line_full, class_full, side_full, court_full


def _resize_array(arr: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    tensor = torch.from_numpy(arr)
    if tensor.dim() == 2:
        tensor = tensor[None, None]
    elif tensor.dim() == 3:
        tensor = tensor[None]
    if mode == "nearest":
        resized = F.interpolate(tensor.float(), size=size, mode="nearest")
    else:
        resized = F.interpolate(tensor.float(), size=size, mode=mode, align_corners=False)
    return resized.squeeze().numpy()


def _onehot(class_target: np.ndarray, num_classes: int) -> np.ndarray:
    out = np.zeros((num_classes, *class_target.shape), dtype=np.float32)
    for k in range(num_classes):
        out[k] = (class_target == k).astype(np.float32)
    return out


def _legend_handles(palette: np.ndarray, line_names: tuple[str, ...]) -> list[plt.Line2D]:
    return [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=palette[k], markersize=9, label=name)
        for k, name in enumerate(line_names)
    ]


def _save_panel(
    out_path: Path,
    image_rgb: np.ndarray,
    pred_overlay: np.ndarray,
    line_prob: np.ndarray,
    class_probs: np.ndarray,
    palette: np.ndarray,
    line_names: tuple[str, ...],
    title: str,
    gt_overlay: np.ndarray | None = None,
    side_prob: np.ndarray | None = None,
    court_prob: np.ndarray | None = None,
) -> None:
    if gt_overlay is None:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        panels = [
            (axes[0, 0], image_rgb, "Input frame"),
            (axes[0, 1], pred_overlay, "Prediction overlay"),
            (axes[0, 2], court_prob if court_prob is not None else line_prob, "Predicted court mask"),
            (axes[1, 0], line_prob, "Predicted lineness"),
            (axes[1, 1], _class_rgb(class_probs, palette), "Predicted class color"),
            (axes[1, 2], _side_rgb(side_prob, court_prob) if side_prob is not None and court_prob is not None else line_prob, "Predicted court side (masked)"),
        ]
    else:
        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        panels = [
            (axes[0, 0], image_rgb, "Input frame"),
            (axes[0, 1], gt_overlay, "GT overlay"),
            (axes[0, 2], court_prob if court_prob is not None else line_prob, "Predicted court mask"),
            (axes[1, 0], pred_overlay, "Prediction overlay"),
            (axes[1, 1], line_prob, "Predicted lineness"),
            (axes[1, 2], _side_rgb(side_prob, court_prob) if side_prob is not None and court_prob is not None else line_prob, "Predicted court side (masked)"),
        ]

    for ax, image, panel_title in panels:
        if image.ndim == 2:
            ax.imshow(image, cmap="inferno", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(panel_title)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.legend(
        handles=_legend_handles(palette, line_names),
        loc="lower center",
        ncol=min(5, len(line_names)),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _save_gt_panel(
    out_path: Path,
    image_rgb: np.ndarray,
    gt_overlay: np.ndarray,
    gt_line: np.ndarray,
    gt_class: np.ndarray,
    side_target: np.ndarray,
    court_mask: np.ndarray,
    palette: np.ndarray,
    line_names: tuple[str, ...],
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    panels = [
        (axes[0, 0], image_rgb, "Input frame"),
        (axes[0, 1], gt_overlay, "GT overlay"),
        (axes[0, 2], court_mask, "GT court mask"),
        (axes[1, 0], gt_line, "GT lineness"),
        (axes[1, 1], _class_rgb(_onehot(gt_class.astype(np.int64), len(line_names)), palette), "GT class color"),
        (axes[1, 2], _side_rgb(side_target, court_mask), "GT court side (masked)"),
    ]

    for ax, image, panel_title in panels:
        if image.ndim == 2:
            ax.imshow(image, cmap="inferno", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(panel_title)
        ax.axis("off")

    fig.suptitle(title, fontsize=11)
    fig.legend(
        handles=_legend_handles(palette, line_names),
        loc="lower center",
        ncol=min(5, len(line_names)),
        frameon=False,
        fontsize=8,
    )
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _class_rgb(class_probs: np.ndarray, palette: np.ndarray) -> np.ndarray:
    class_ids = class_probs.argmax(axis=0)
    return palette[class_ids]


def _side_rgb(side_prob: np.ndarray, court_prob: np.ndarray) -> np.ndarray:
    rgb = plt.get_cmap("coolwarm")(np.clip(side_prob, 0.0, 1.0))[..., :3].astype(np.float32)
    rgb[court_prob < 0.5] = 0.15
    return rgb


def _save_rgb(out_path: Path, image_rgb: np.ndarray) -> None:
    image_u8 = np.clip(image_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(out_path), cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR))


def visualize_dataset(args: argparse.Namespace) -> None:
    _validate_common_args(args)
    model, device = _load_model(args)
    num_classes = int(model.hparams.num_classes)
    line_names = _line_names(model, num_classes)
    image_size = (args.image_height, args.image_width)
    palette = class_palette(num_classes)

    dm = FibaCourtMarkingDataModule(
        root=args.root,
        image_size=image_size,
        output_stride=int(model.hparams.output_stride),
        sigma=float(model.hparams.sigma),
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        use_player_occlusion=args.use_player_occlusion,
    )
    dm.setup("test")
    dataset = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}[args.split]

    args.out.mkdir(parents=True, exist_ok=True)
    count = min(args.count, len(dataset))
    for i in range(count):
        sample = dataset[i]
        image_rgb = sample["image"].permute(1, 2, 0).numpy()
        line_prob, class_probs, side_prob, court_prob = _predict_full_resolution(model, image_rgb, image_size, device)
        pred_overlay = overlay_line_predictions(image_rgb, class_probs, line_prob, palette)

        gt_line = _resize_array(sample["lineness"].numpy(), image_rgb.shape[:2], mode="bilinear")
        gt_class = _resize_array(sample["class_target"].numpy().astype(np.float32), image_rgb.shape[:2], mode="nearest")
        gt_overlay = overlay_line_predictions(
            image_rgb,
            _onehot(gt_class.astype(np.int64), num_classes),
            gt_line,
            palette,
        )

        source = Path(sample["image_path"]).name
        out_path = args.out / f"dataset_markings_{i:03d}.png"
        title = f"{args.split} sample {i} | base index {sample['index']} | {source}"
        _save_panel(
            out_path,
            image_rgb,
            pred_overlay,
            line_prob,
            class_probs,
            palette,
            line_names,
            title,
            gt_overlay,
            side_prob=side_prob,
            court_prob=court_prob,
        )
        print(f"wrote {out_path}")


def _project_marking_lines(calib, line_names: tuple[str, ...], n_samples_per_line: int) -> dict[str, np.ndarray]:
    worlds = court_markings_world(n=n_samples_per_line)
    needed = {"halfcourt", "baseline_right"}
    for geometry_name, class_name in MARKING_CLASS_BY_GEOMETRY.items():
        if class_name in line_names:
            needed.add(geometry_name)

    lines_uv: dict[str, np.ndarray] = {}
    for geometry_name in sorted(needed):
        lines_uv[geometry_name] = project_world_to_image(worlds[geometry_name], calib).astype(np.float32)

    lines_uv["__court_full"] = project_world_to_image(
        np.array([
            [0.0, 0.0, 0.0],
            [COURT_LENGTH_CM, 0.0, 0.0],
            [COURT_LENGTH_CM, COURT_WIDTH_CM, 0.0],
            [0.0, COURT_WIDTH_CM, 0.0],
        ], dtype=float),
        calib,
    ).astype(np.float32)
    return lines_uv


def _midcourt_gt_targets(
    image_rgb: np.ndarray,
    calib,
    line_names: tuple[str, ...],
    output_stride: int,
    sigma: float,
    side_blur_sigma: float,
    n_samples_per_line: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lines_uv = _project_marking_lines(calib, line_names, n_samples_per_line)
    visible_mask = (image_rgb > 1.0 / 255.0).any(axis=2)
    lineness, class_target, _ = _render_line_targets(
        lines_uv,
        image_size=image_rgb.shape[:2],
        output_stride=output_stride,
        sigma=sigma,
        line_names=line_names,
        class_by_geometry=MARKING_CLASS_BY_GEOMETRY,
        visible_mask=visible_mask,
        foul_line_occlusion_names={
            "foul_left": "free_throw_circle_left",
            "foul_right": "free_throw_circle_right",
        },
    )
    side_target, court_mask = _render_side_target(
        lines_uv,
        image_size=image_rgb.shape[:2],
        output_stride=output_stride,
        blur_sigma=side_blur_sigma,
        visible_mask=visible_mask,
    )
    return lineness, class_target, side_target, court_mask


def visualize_midcourt(args: argparse.Namespace) -> None:
    _validate_common_args(args)
    if args.gt_only:
        model = None
        device = None
        num_classes = len(FIBA_MARKING_NAMES)
        line_names = FIBA_MARKING_NAMES
        output_stride = args.output_stride
        sigma = args.sigma
        side_blur_sigma = args.side_blur_sigma
    else:
        model, device = _load_model(args)
        num_classes = int(model.hparams.num_classes)
        line_names = _line_names(model, num_classes)
        output_stride = int(model.hparams.output_stride)
        sigma = float(model.hparams.sigma)
        side_blur_sigma = float(getattr(model.hparams, "side_blur_sigma", 1.0))
    image_size = (args.image_height, args.image_width)
    palette = class_palette(num_classes)

    dataset = DeepSportMidcourtStitchDataset(
        args.root,
        pairs_per_game=args.pairs_per_game,
        seed=args.seed,
        camera_portion=args.camera_portion,
        camera_portion_range=tuple(args.camera_portion_range),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    count = min(args.count, len(dataset))
    for i in range(count):
        rendered = dataset.render_pair(dataset.pairs[i])
        image_rgb = rendered.image

        gt_line, gt_class, side_target, court_mask = _midcourt_gt_targets(
            image_rgb,
            rendered.calibration,
            line_names,
            output_stride=output_stride,
            sigma=sigma,
            side_blur_sigma=side_blur_sigma,
            n_samples_per_line=args.n_samples_per_line,
        )
        gt_line = _resize_array(gt_line, image_rgb.shape[:2], mode="bilinear")
        gt_class = _resize_array(gt_class.astype(np.float32), image_rgb.shape[:2], mode="nearest")
        side_target = _resize_array(side_target, image_rgb.shape[:2], mode="bilinear")
        court_mask = _resize_array(court_mask, image_rgb.shape[:2], mode="nearest")
        gt_overlay = overlay_line_predictions(
            image_rgb,
            _onehot(gt_class.astype(np.int64), num_classes),
            gt_line,
            palette,
        )

        safe_game = _safe_name(rendered.pair.game)
        out_path = args.out / f"midcourt_markings_{i:03d}_{safe_game}.png"
        title = (
            f"midcourt stitch {i} | {rendered.pair.game} | "
            f"pan {rendered.camera_portion:.2f} | "
            f"L {rendered.pair.left.image_path.name} + R {rendered.pair.right.image_path.name}"
        )
        if args.gt_only:
            _save_gt_panel(
                out_path,
                image_rgb,
                gt_overlay,
                gt_line,
                gt_class,
                side_target,
                court_mask,
                palette,
                line_names,
                title,
            )
        else:
            assert model is not None and device is not None
            line_prob, class_probs, side_prob, court_prob = _predict_full_resolution(model, image_rgb, image_size, device)
            pred_overlay = overlay_line_predictions(image_rgb, class_probs, line_prob, palette)
            _save_panel(
                out_path,
                image_rgb,
                pred_overlay,
                line_prob,
                class_probs,
                palette,
                line_names,
                title,
                gt_overlay,
                side_prob=side_prob,
                court_prob=court_prob,
            )
        print(f"midcourt {i + 1}/{count}: wrote {out_path}")


def _read_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _iter_paths(root: Path, extensions: Iterable[str], recursive: bool) -> list[Path]:
    suffixes = {ext.lower() for ext in extensions}
    paths = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in suffixes)


def _iter_frame_folder(args: argparse.Namespace) -> Iterator[tuple[str, np.ndarray]]:
    if args.frames.is_file() and args.frames.suffix.lower() in IMAGE_EXTENSIONS:
        yield args.frames.stem, _read_image(args.frames)
        return
    if args.frames.is_file() and args.frames.suffix.lower() in VIDEO_EXTENSIONS:
        yield from _iter_video(args.frames, args.video_stride)
        return
    if not args.frames.is_dir():
        raise FileNotFoundError(f"Frame source not found: {args.frames}")

    image_paths = _iter_paths(args.frames, IMAGE_EXTENSIONS, args.recursive)
    for path in image_paths:
        yield _relative_stem(path, args.frames), _read_image(path)

    if args.include_videos:
        for video_path in _iter_paths(args.frames, VIDEO_EXTENSIONS, args.recursive):
            yield from _iter_video(video_path, args.video_stride, prefix=_relative_stem(video_path, args.frames))


def _iter_video(path: Path, stride: int, prefix: str | None = None) -> Iterator[tuple[str, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    label_prefix = prefix or path.stem
    frame_idx = 0
    emitted = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                yield f"{label_prefix}_frame_{frame_idx:06d}", image_rgb
                emitted += 1
            frame_idx += 1
    finally:
        cap.release()
    if emitted == 0:
        raise RuntimeError(f"No frames could be sampled from video: {path}")


def _relative_stem(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    stem = str(rel.with_suffix(""))
    return _safe_name(stem)


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "frame"


def visualize_frames(args: argparse.Namespace) -> None:
    _validate_common_args(args)
    if args.video_stride < 1:
        raise ValueError("--video-stride must be at least 1")
    model, device = _load_model(args)
    num_classes = int(model.hparams.num_classes)
    line_names = _line_names(model, num_classes)
    image_size = (args.image_height, args.image_width)
    palette = class_palette(num_classes)

    args.out.mkdir(parents=True, exist_ok=True)
    seen = 0
    for label, image_rgb in _iter_frame_folder(args):
        if seen >= args.count:
            break
        line_prob, class_probs, side_prob, court_prob = _predict_full_resolution(model, image_rgb, image_size, device)
        pred_overlay = overlay_line_predictions(image_rgb, class_probs, line_prob, palette)
        safe_label = _safe_name(label)
        if args.panel:
            out_path = args.out / f"{safe_label}_markings_panel.png"
            _save_panel(
                out_path,
                image_rgb,
                pred_overlay,
                line_prob,
                class_probs,
                palette,
                line_names,
                safe_label,
                side_prob=side_prob,
                court_prob=court_prob,
            )
        else:
            out_path = args.out / f"{safe_label}_markings_overlay.png"
            _save_rgb(out_path, pred_overlay)
        seen += 1
        print(f"frame {seen}/{args.count}: wrote {out_path}")

    if seen == 0:
        raise RuntimeError(f"No image or video frames found under {args.frames}")


def _line_names(model: FibaCourtMarkingLightning, num_classes: int) -> tuple[str, ...]:
    names = tuple(getattr(model, "line_names", FIBA_MARKING_NAMES))
    return names[:num_classes]


def _validate_common_args(args: argparse.Namespace) -> None:
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.image_height < 1 or args.image_width < 1:
        raise ValueError("--image-height and --image-width must be at least 1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--checkpoint", type=Path, default=Path("checkpoints/fiba_court_markings/last.ckpt"))
        p.add_argument("--out", type=Path, default=Path("tests/output/markings_output_vis"))
        p.add_argument("--count", type=int, default=8)
        p.add_argument("--image-height", type=int, default=384)
        p.add_argument("--image-width", type=int, default=640)
        p.add_argument("--cpu", action="store_true")

    dataset_parser = subparsers.add_parser("dataset", help="Visualize predictions on DeepSport loader samples.")
    add_common_args(dataset_parser)
    dataset_parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    dataset_parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    dataset_parser.add_argument("--num-workers", type=int, default=0)
    dataset_parser.add_argument("--seed", type=int, default=1430)
    dataset_parser.add_argument("--val-fraction", type=float, default=0.15)
    dataset_parser.add_argument("--test-fraction", type=float, default=0.15)
    dataset_parser.add_argument("--use-player-occlusion", action=argparse.BooleanOptionalAction, default=False)
    dataset_parser.set_defaults(func=visualize_dataset)

    midcourt_parser = subparsers.add_parser("midcourt", help="Visualize predictions on stitched mid-court samples.")
    add_common_args(midcourt_parser)
    midcourt_parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    midcourt_parser.add_argument("--pairs-per-game", type=int, default=5)
    midcourt_parser.add_argument("--seed", type=int, default=1430)
    midcourt_parser.add_argument(
        "--camera-portion",
        type=float,
        default=None,
        help="Fixed virtual camera pan: 0 is left, 1 is right. Default randomizes.",
    )
    midcourt_parser.add_argument(
        "--camera-portion-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 1.0),
        help="Uniform random pan range used when --camera-portion is omitted.",
    )
    midcourt_parser.add_argument("--n-samples-per-line", type=int, default=400)
    midcourt_parser.add_argument("--gt-only", action="store_true")
    midcourt_parser.add_argument("--output-stride", type=int, default=2)
    midcourt_parser.add_argument("--sigma", type=float, default=1.5)
    midcourt_parser.add_argument("--side-blur-sigma", type=float, default=1.0)
    midcourt_parser.set_defaults(func=visualize_midcourt)

    frames_parser = subparsers.add_parser("frames", help="Visualize predictions on image frames or sampled videos.")
    add_common_args(frames_parser)
    frames_parser.add_argument("--frames", type=Path, default=Path("test_footage"))
    frames_parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    frames_parser.add_argument("--include-videos", action=argparse.BooleanOptionalAction, default=True)
    frames_parser.add_argument("--video-stride", type=int, default=60)
    frames_parser.add_argument("--panel", action=argparse.BooleanOptionalAction, default=True)
    frames_parser.set_defaults(func=visualize_frames)

    return parser


def main() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
