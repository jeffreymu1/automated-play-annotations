"""Inspect the packaged DeepSport dataset reader."""

from __future__ import annotations

import argparse
from pathlib import Path

from court_detection.dataset import DeepSportDataset
from court_detection.geometry import KEYPOINT_NAMES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    dataset = DeepSportDataset(args.root)
    print(f"len(dataset) = {len(dataset)}")

    image, keypoints, calib = dataset[args.idx]
    print(
        f"X.shape = {tuple(image.shape)}   dtype = {image.dtype}   "
        f"min = {image.min().item():.3f}   max = {image.max().item():.3f}"
    )
    print(f"image size (calib): {calib.width} x {calib.height}")
    print("y:")
    for name, row in zip(KEYPOINT_NAMES, keypoints):
        if row[2].item() == 0:
            print(f"  {name}: None")
        else:
            print(f"  {name}: ({row[0].item():.1f}, {row[1].item():.1f}, v={int(row[2].item())})")


if __name__ == "__main__":
    main()
