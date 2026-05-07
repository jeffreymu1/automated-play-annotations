"""Export augmented, randomly panned frames for report figures.

This uses the same midcourt panning and frame augmentation path as training:
DeepSportMidcourtStitchDataset -> CourtLineFrameDataset(..., augment=True).
The augmented frame dataset alternates the no-scorebar and scorebar variants,
so this exporter intentionally takes the odd-index scorebar variant only.

Example:
    uv run python tests/export_augmented_report_frames.py --count 30 \
        --out tests/output/report_augmented_frames
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np

from court_detection.lines import CourtLineFrameDataset
from court_detection.markings import FIBA_MARKING_NAMES
from court_detection.midcourt_stitch_dataset import DeepSportMidcourtStitchDataset


def _write_rgb(path: Path, image_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image_u8 = np.clip(image_rgb * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR))


def export_augmented_frames(args: argparse.Namespace) -> None:
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.image_height < 1 or args.image_width < 1:
        raise ValueError("--image-height and --image-width must be at least 1")

    random.seed(args.seed)
    np_rng = np.random.default_rng(args.seed)

    pan_base = DeepSportMidcourtStitchDataset(
        args.root,
        pairs_per_game=args.pairs_per_game,
        seed=args.seed,
        camera_portion=None,
        camera_portion_range=tuple(args.camera_portion_range),
        return_annotation_occlusion_mask=args.use_player_occlusion,
    )
    replace = len(pan_base) < args.count
    selected_indices = np_rng.choice(len(pan_base), size=args.count, replace=replace).astype(int).tolist()

    dataset = CourtLineFrameDataset(
        pan_base,
        indices=selected_indices,
        image_size=(args.image_height, args.image_width),
        output_stride=args.output_stride,
        sigma=args.sigma,
        augment=True,
        line_names=FIBA_MARKING_NAMES,
        side_blur_sigma=args.side_blur_sigma,
        use_player_occlusion=args.use_player_occlusion,
    )

    args.out.mkdir(parents=True, exist_ok=True)
    metadata = []
    for out_idx, pan_idx in enumerate(selected_indices):
        sample = dataset[2 * out_idx + 1]
        image_rgb = sample["image"].permute(1, 2, 0).numpy()
        out_path = args.out / f"augmented_frame_{out_idx:03d}.png"
        _write_rgb(out_path, image_rgb)
        metadata.append(
            {
                "output": str(out_path),
                "pan_dataset_index": int(pan_idx),
                "source": str(sample["image_path"]),
                "score_bar": bool(sample["score_bar"]),
            }
        )
        print(f"wrote {out_path}")

    metadata_path = args.out / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {metadata_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    parser.add_argument("--out", type=Path, default=Path("tests/output/report_augmented_frames"))
    parser.add_argument("--count", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1430)
    parser.add_argument("--image-height", type=int, default=384)
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--output-stride", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.5)
    parser.add_argument("--side-blur-sigma", type=float, default=1.0)
    parser.add_argument("--pairs-per-game", type=int, default=8)
    parser.add_argument(
        "--camera-portion-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 1.0),
        help="Random virtual camera pan range, where 0 is left and 1 is right.",
    )
    parser.add_argument("--use-player-occlusion", action=argparse.BooleanOptionalAction, default=False)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    export_augmented_frames(args)


if __name__ == "__main__":
    main()
