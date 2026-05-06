"""Visualize augmented court-marking learning targets for DeepSport frames."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from court_detection.dataset import DeepSportDataset
from court_detection.markings import FIBA_MARKING_NAMES, CourtLineFrameDataset, class_palette


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


def _onehot(class_target: np.ndarray, count: int) -> np.ndarray:
    out = np.zeros((count, *class_target.shape), dtype=np.float32)
    for k in range(count):
        out[k] = (class_target == k).astype(np.float32)
    return out


def _masked_side_rgb(side_target: np.ndarray, side_weight: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("coolwarm")
    rgb = cmap(np.clip(side_target, 0.0, 1.0))[..., :3].astype(np.float32)
    ignored = side_weight <= 0.0
    rgb[ignored] = 0.15
    return rgb


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
        line_names=FIBA_MARKING_NAMES,
        side_blur_sigma=args.side_blur_sigma,
        use_player_occlusion=args.use_player_occlusion,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    palette = class_palette(len(FIBA_MARKING_NAMES))
    count = min(args.count, len(dataset))

    for i in range(count):
        sample = dataset[i]
        image = sample["image"].permute(1, 2, 0).numpy()
        lineness = sample["lineness"].numpy()
        class_target = sample["class_target"].numpy()
        side_target = sample["side_target"].numpy()
        side_weight = sample["side_weight"].numpy()
        court_mask = sample["court_mask"].numpy()
        occlusion_mask = sample["annotation_occlusion_mask"].numpy()
        lineness_weight = sample["lineness_weight"].numpy()
        class_onehot = _onehot(class_target, len(FIBA_MARKING_NAMES))

        fig, axes = plt.subplots(2, 4, figsize=(17, 7.5), constrained_layout=True)

        ax = axes[0, 0]
        ax.imshow(image)
        ax.set_title("Augmented input" if sample["score_bar"] else "Input")
        ax.axis("off")

        ax = axes[0, 1]
        ax.imshow(lineness, cmap="inferno", vmin=0.0, vmax=1.0)
        ax.set_title("Lineness")
        ax.axis("off")

        ax = axes[1, 0]
        class_rgb = np.einsum("khw,kc->hwc", class_onehot, palette)
        ax.imshow(np.clip(class_rgb, 0.0, 1.0))
        ax.set_title("Class target")
        ax.axis("off")

        ax = axes[1, 1]
        ax.imshow(_masked_side_rgb(side_target, side_weight))
        ax.set_title("Court side target (masked)")
        ax.axis("off")

        ax = axes[0, 2]
        ax.imshow(court_mask, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title("Court mask")
        ax.axis("off")

        ax = axes[1, 2]
        ax.imshow(occlusion_mask, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title("Player/ball occlusion")
        ax.axis("off")

        ax = axes[0, 3]
        ax.imshow(lineness_weight, cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title("Lineness weight")
        ax.axis("off")

        axes[1, 3].axis("off")

        stem = f"court_learning_target_{i:03d}"
        source = Path(sample["image_path"]).name
        fig.suptitle(
            f"{stem} | base index {sample['index']} | {source} | "
            f"stride={args.output_stride} sigma={args.sigma} player_occlusion={args.use_player_occlusion}",
            fontsize=11,
        )
        fig.savefig(args.out / f"{stem}.png", dpi=140)
        plt.close(fig)

        np.savez_compressed(
            args.out / f"{stem}.npz",
            lineness=lineness,
            class_target=class_target,
            side_target=side_target,
            side_weight=side_weight,
            court_mask=court_mask,
            annotation_occlusion_mask=occlusion_mask,
            lineness_weight=lineness_weight,
            visible=sample["visible"].numpy(),
            line_names=np.array(FIBA_MARKING_NAMES),
            image_path=str(sample["image_path"]),
            base_index=int(sample["index"]),
            score_bar=bool(sample["score_bar"]),
            use_player_occlusion=bool(args.use_player_occlusion),
        )
        print(f"wrote {args.out / f'{stem}.png'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    parser.add_argument("--out", type=Path, default=Path("tests/output/court_learning_targets"))
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--split", choices=("train", "val", "test"), default="train")
    parser.add_argument("--seed", type=int, default=1430)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--output-stride", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--side-blur-sigma", type=float, default=1.0)
    parser.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-player-occlusion", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    visualize(args)


if __name__ == "__main__":
    main()
