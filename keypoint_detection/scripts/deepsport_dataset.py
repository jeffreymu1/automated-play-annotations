"""PyTorch-style dataset over the DeepSport court-frames.

Walks a flat list of ``<set>/<game>/camcourtN_<ts>_0.png`` images under
a dataset root (default ``data/deepsport-dataset``) and yields, per
frame:

    X      -- grayscale image, torch.Tensor of shape (1, H, W), float32, [0, 1].
    y      -- dict of 6 keypoints (4 court corners + 2 half-court
              endpoints); each value is (u, v) in image pixels or None
              if the keypoint projects behind the camera or outside the
              image rectangle.
    calib  -- CameraCalibration for the frame (needed by tooling that
              still wants to sample lens-distorted curves).

Sets collapse into one flat index range; callers don't need to know
which set an index belongs to.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from court_geometry import (
    CameraCalibration,
    KEYPOINT_NAMES,
    court_keypoints_world,
    project_world_to_image,
)

Keypoints = dict[str, tuple[float, float] | None]


class DeepSportDataset(Dataset):
    def __init__(self, root: Path | str = Path("data/deepsport-dataset")) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        samples: list[tuple[Path, Path]] = []
        for image_path in sorted(self.root.glob("*/*/*_0.png")):
            json_path = image_path.with_name(image_path.stem[: -len("_0")] + ".json")
            if json_path.is_file():
                samples.append((image_path, json_path))
        if not samples:
            raise RuntimeError(f"No (png, json) pairs found under {self.root}")
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Keypoints, CameraCalibration]:
        image_path, json_path = self.samples[idx]

        calib = CameraCalibration.from_json(json_path)

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR_RGB)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        X = torch.from_numpy(img).to(torch.float32).div_(255.0).unsqueeze(0)

        uvs = project_world_to_image(court_keypoints_world(), calib)

        y: Keypoints = {}
        for name, uv in zip(KEYPOINT_NAMES, uvs):
            u, v = float(uv[0]), float(uv[1])
            if (np.isfinite(uv).all()
                    and 0.0 <= u < calib.width
                    and 0.0 <= v < calib.height):
                y[name] = (u, v)
            else:
                y[name] = None

        return X, y, calib


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    dataset = DeepSportDataset(args.root)
    print(f"len(dataset) = {len(dataset)}")

    X, y, calib = dataset[args.idx]
    print(f"X.shape = {tuple(X.shape)}   dtype = {X.dtype}   "
          f"min = {X.min().item():.3f}   max = {X.max().item():.3f}")
    print(f"image size (calib): {calib.width} x {calib.height}")
    print("y:")
    for name, uv in y.items():
        if uv is None:
            print(f"  {name}: None")
        else:
            print(f"  {name}: ({uv[0]:.1f}, {uv[1]:.1f})")


if __name__ == "__main__":
    main()
