"""Render the basketball court boundary + keypoints onto a DeepSport frame."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from court_detection.dataset import DeepSportDataset, keypoints_to_dict
from court_detection.geometry import (
    COURT_LENGTH_CM,
    COURT_WIDTH_CM,
    CameraCalibration,
    KEYPOINT_NAMES,
    court_corners_world,
    project_world_to_image,
    sample_segment,
)


def draw_polyline(
    image_bgr: np.ndarray,
    pts_uv: np.ndarray,
    color=(0, 255, 255),
    thickness: int = 2,
    closed: bool = False,
) -> None:
    finite = np.isfinite(pts_uv).all(axis=1)
    if finite.sum() < 2:
        return
    pts = np.round(pts_uv[finite]).astype(np.int32).reshape(-1, 1, 2)
    cv2.polylines(image_bgr, [pts], isClosed=closed, color=color, thickness=thickness, lineType=cv2.LINE_AA)


KEYPOINT_COLORS = {
    "corner_left_far": (0, 0, 255),
    "corner_right_far": (255, 0, 255),
    "corner_right_near": (255, 255, 0),
    "corner_left_near": (255, 128, 0),
    "half_court_far": (0, 255, 0),
    "half_court_near": (255, 255, 255),
}


def draw_court_edges(
    image_bgr: np.ndarray,
    calib: CameraCalibration,
    length_cm: float = COURT_LENGTH_CM,
    width_cm: float = COURT_WIDTH_CM,
    n_samples: int = 400,
) -> None:
    corners_w = court_corners_world(length_cm, width_cm)
    edges = [
        (corners_w[0], corners_w[1]),
        (corners_w[1], corners_w[2]),
        (corners_w[2], corners_w[3]),
        (corners_w[3], corners_w[0]),
    ]
    for a, b in edges:
        pts_uv = project_world_to_image(sample_segment(a, b, n=n_samples), calib)
        draw_polyline(image_bgr, pts_uv, color=(0, 255, 255), thickness=2, closed=False)

    half_top = np.array([length_cm / 2, 0.0, 0.0])
    half_bot = np.array([length_cm / 2, width_cm, 0.0])
    half_uv = project_world_to_image(sample_segment(half_top, half_bot, n=n_samples), calib)
    draw_polyline(image_bgr, half_uv, color=(255, 0, 0), thickness=2, closed=False)


def draw_keypoints(
    image_bgr: np.ndarray,
    keypoints: dict[str, tuple[float, float] | None] | np.ndarray,
    radius: int = 8,
) -> None:
    if not isinstance(keypoints, dict):
        keypoints = keypoints_to_dict(keypoints)
    for name, uv in keypoints.items():
        if uv is None:
            continue
        color = KEYPOINT_COLORS.get(name, (255, 255, 255))
        u, v = int(round(uv[0])), int(round(uv[1]))
        cv2.circle(image_bgr, (u, v), radius, color, -1, lineType=cv2.LINE_AA)
        cv2.putText(image_bgr, name, (u + 10, v - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idx", type=int, default=0, help="Flat frame index into the DeepSport dataset.")
    parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    parser.add_argument("--save", type=Path, default=None)
    args = parser.parse_args()

    dataset = DeepSportDataset(args.root)
    image, keypoints, calib = dataset[args.idx]

    print(f"Frame {args.idx} / {len(dataset)}  ({calib.width} x {calib.height})")
    print("Keypoints (u, v):")
    for name, row in zip(KEYPOINT_NAMES, keypoints):
        if row[2].item() == 0:
            print(f"  {name}: None")
        else:
            print(f"  {name}: ({row[0].item():.1f}, {row[1].item():.1f}, v={int(row[2].item())})")

    rgb_u8 = (image.permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    image_bgr = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)

    overlay = image_bgr.copy()
    draw_court_edges(overlay, calib)
    draw_keypoints(overlay, keypoints)

    if args.save:
        cv2.imwrite(str(args.save), overlay)
        print(f"Wrote overlay to {args.save}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original frame")
    axes[0].axis("off")
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title("Court region highlighted")
    axes[1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
