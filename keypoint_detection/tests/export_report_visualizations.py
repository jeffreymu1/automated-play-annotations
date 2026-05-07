"""Export report visualizations as separate image files per frame.

Examples:
    uv run python tests/export_report_visualizations.py dataset \
        --split test --count 20 --random --out tests/output/report_visualizations_split

    uv run python tests/export_report_visualizations.py frames \
        --frames test_footage --count 4 --random --video-stride 120
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

import visualize_report_figures as report


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(image_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR))


def _heatmap_rgb(values: np.ndarray, cmap_name: str) -> np.ndarray:
    return plt.get_cmap(cmap_name)(np.clip(values, 0.0, 1.0))[..., :3].astype(np.float32)


def _write_frame_visualizations(
    out_dir: Path,
    image_rgb: np.ndarray,
    line_prob: np.ndarray,
    class_probs: np.ndarray,
    side_prob: np.ndarray,
    court_prob: np.ndarray,
    palette: np.ndarray,
    mask_side_with_court: bool,
) -> None:
    line_class_overlay = report.overlay_line_predictions(image_rgb, class_probs, line_prob, palette)
    outputs = {
        "01_raw_frame.png": image_rgb,
        "02_lineness.png": _heatmap_rgb(line_prob, "inferno"),
        "03_line_class.png": report._class_rgb(class_probs, palette),
        "04_court_side.png": report._side_rgb(side_prob, court_prob, mask_side_with_court),
        "05_court_mask.png": _heatmap_rgb(court_prob, "gray"),
        "06_lineness_class_overlay.png": line_class_overlay,
    }
    for filename, image in outputs.items():
        _write_rgb(out_dir / filename, image)


def export_dataset(args: argparse.Namespace) -> None:
    report._validate_common_args(args)
    model, device = report._load_model(args)
    image_size = (args.image_height, args.image_width)
    palette = report.class_palette(int(model.hparams.num_classes))

    dm = report.FibaCourtMarkingDataModule(
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
        pan_train_ratio=0.0,
    )
    dm.setup("test")
    dataset = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}[args.split]
    indices = report._sample_indices(len(dataset), args.count, args.random, args.seed)

    for out_idx, dataset_idx in enumerate(indices):
        sample = dataset[dataset_idx]
        image_rgb = sample["image"].permute(1, 2, 0).numpy()
        line_prob, class_probs, side_prob, court_prob = report._predict_full_resolution(
            model,
            image_rgb,
            image_size,
            device,
        )
        frame_dir = args.out / f"dataset_{args.split}_{out_idx:03d}"
        _write_frame_visualizations(
            frame_dir,
            image_rgb,
            line_prob,
            class_probs,
            side_prob,
            court_prob,
            palette,
            args.mask_side_with_court,
        )
        print(f"wrote {frame_dir}")


def export_frames(args: argparse.Namespace) -> None:
    report._validate_common_args(args)
    if args.video_stride < 1:
        raise ValueError("--video-stride must be at least 1")
    model, device = report._load_model(args)
    image_size = (args.image_height, args.image_width)
    palette = report.class_palette(int(model.hparams.num_classes))

    frame_iter = report._iter_random_frame_folder(args) if args.random else report._iter_frame_folder(args)
    seen = 0
    for label, image_rgb in frame_iter:
        if seen >= args.count:
            break
        line_prob, class_probs, side_prob, court_prob = report._predict_full_resolution(
            model,
            image_rgb,
            image_size,
            device,
        )
        frame_dir = args.out / report._safe_name(label)
        _write_frame_visualizations(
            frame_dir,
            image_rgb,
            line_prob,
            class_probs,
            side_prob,
            court_prob,
            palette,
            args.mask_side_with_court,
        )
        seen += 1
        print(f"frame {seen}/{args.count}: wrote {frame_dir}")

    if seen == 0:
        raise RuntimeError(f"No image or video frames found under {args.frames}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--checkpoint",
            type=Path,
            default=Path("checkpoints/006_fiba_court_markings_structured_side_1to1_pan/last-v1.ckpt"),
        )
        p.add_argument("--architecture", choices=("auto", *sorted(report.MODEL_ARCHITECTURES)), default="auto")
        p.add_argument("--out", type=Path, default=Path("tests/output/report_visualizations_split"))
        p.add_argument("--count", type=int, default=4)
        p.add_argument("--image-height", type=int, default=384)
        p.add_argument("--image-width", type=int, default=640)
        p.add_argument("--mask-side-with-court", action=argparse.BooleanOptionalAction, default=False)
        p.add_argument("--cpu", action="store_true")

    dataset_parser = subparsers.add_parser("dataset", help="Export split visualizations from DeepSport samples.")
    add_common_args(dataset_parser)
    dataset_parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    dataset_parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    dataset_parser.add_argument("--num-workers", type=int, default=0)
    dataset_parser.add_argument("--seed", type=int, default=1430)
    dataset_parser.add_argument("--val-fraction", type=float, default=0.15)
    dataset_parser.add_argument("--test-fraction", type=float, default=0.15)
    dataset_parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False)
    dataset_parser.add_argument("--use-player-occlusion", action=argparse.BooleanOptionalAction, default=False)
    dataset_parser.set_defaults(func=export_dataset)

    frames_parser = subparsers.add_parser("frames", help="Export split visualizations from image frames or videos.")
    add_common_args(frames_parser)
    frames_parser.add_argument("--frames", type=Path, default=Path("test_footage"))
    frames_parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    frames_parser.add_argument("--include-videos", action=argparse.BooleanOptionalAction, default=True)
    frames_parser.add_argument("--video-stride", type=int, default=60)
    frames_parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False)
    frames_parser.add_argument("--seed", type=int, default=1430)
    frames_parser.set_defaults(func=export_frames)

    return parser


def main() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
