"""Visualize the line ground-truth targets fed into the network.

Loads the DeepSport dataset through CourtLineFrameDataset (which invokes
_render_line_targets) so the saved figures match exactly what the model trains on.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from court_detection.dataset import DeepSportDataset
from court_detection.geometry import LINE_NAMES
from court_detection.lines import (
    CourtLineFrameDataset,
    class_palette,
    overlay_line_predictions,
)


def _split_indices(n: int, seed: int, val_fraction: float, test_fraction: float) -> dict[str, list[int]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n).tolist()
    n_test = max(1, int(round(n * test_fraction)))
    n_val = max(1, int(round(n * val_fraction)))
    n_train = max(1, n - n_val - n_test)
    return {
        "train": indices[:n_train],
        "val": indices[n_train:n_train + n_val],
        "test": indices[n_train + n_val:] or indices[n_train:n_train + n_val],
    }


def visualize(args: argparse.Namespace) -> None:
    base = DeepSportDataset(args.root)
    splits = _split_indices(len(base), args.seed, args.val_fraction, args.test_fraction)
    indices = splits[args.split]

    dataset = CourtLineFrameDataset(
        base,
        indices=indices,
        image_size=(args.image_height, args.image_width),
        output_stride=args.output_stride,
        sigma=args.sigma,
        augment=args.augment,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    num_classes = len(LINE_NAMES)
    palette = class_palette(num_classes)
    count = min(args.count, len(dataset))

    for i in range(count):
        sample = dataset[i]
        image_np = sample["image"].permute(1, 2, 0).numpy()
        lineness = sample["lineness"].numpy()
        class_target = sample["class_target"].numpy()
        visible = sample["visible"].numpy()

        class_onehot = np.zeros((num_classes, *lineness.shape), dtype=np.float32)
        for k in range(num_classes):
            class_onehot[k] = (class_target == k).astype(np.float32)

        out_h, out_w = lineness.shape
        image_lo = _resize_image(image_np, (out_h, out_w))
        gt_overlay = overlay_line_predictions(image_lo, class_onehot, lineness, palette)

        legend_handles = [
            plt.Line2D(
                [0], [0], marker="s", color="w",
                markerfacecolor=palette[k], markersize=10,
                label=f"{name}{'' if visible[k] else ' (hidden)'}",
            )
            for k, name in enumerate(LINE_NAMES)
        ]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title(f"Input frame ({args.image_height}x{args.image_width})")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(lineness, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[0, 1].set_title(f"Lineness target (sigma={args.sigma}, stride={args.output_stride})")
        axes[0, 1].axis("off")

        class_rgb = np.einsum("khw,kc->hwc", class_onehot, palette)
        axes[1, 0].imshow(np.clip(class_rgb, 0.0, 1.0))
        axes[1, 0].set_title("Class argmin (Voronoi, ungated)")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(gt_overlay)
        axes[1, 1].set_title("GT overlay (lineness-gated)")
        axes[1, 1].axis("off")

        fig.legend(handles=legend_handles, loc="lower center", ncol=min(5, num_classes), frameon=False)
        fig.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        out_path = args.out / f"line_target_{i:03d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


def _resize_image(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2
    h, w = size
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    parser.add_argument("--out", type=Path, default=Path("results/line_target_vis"))
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--seed", type=int, default=1430)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--output-stride", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--augment", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
