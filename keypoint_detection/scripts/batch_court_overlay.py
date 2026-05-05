"""Render court overlays for a random sample of DeepSport frames."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from court_detection.dataset import DeepSportDataset
from court_overlay import draw_court_edges, draw_keypoints


def render_overlay(dataset: DeepSportDataset, idx: int) -> np.ndarray:
    image, keypoints, calib = dataset[idx]
    rgb_u8 = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    overlay = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    draw_court_edges(overlay, calib)
    draw_keypoints(overlay, keypoints)
    return overlay


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    parser.add_argument("--out", type=Path, default=Path("results"))
    parser.add_argument("--count", "-n", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    dataset = DeepSportDataset(args.root)
    if args.count <= 0:
        print(f"count={args.count}; nothing to render")
        return
    n = min(args.count, len(dataset))

    rng = random.Random(args.seed)
    indices = sorted(rng.sample(range(len(dataset)), n))

    args.out.mkdir(parents=True, exist_ok=True)
    for idx in indices:
        image_path, _ = dataset.samples[idx]
        try:
            overlay = render_overlay(dataset, idx)
        except Exception as exc:
            print(f"[fail] idx={idx} ({image_path.name}): {exc}")
            continue
        out_path = args.out / f"idx{idx:04d}__{image_path.stem}.png"
        cv2.imwrite(str(out_path), overlay)
        print(f"[ok]   idx={idx:4d}: {image_path.relative_to(args.root)} -> {out_path}")


if __name__ == "__main__":
    main()
