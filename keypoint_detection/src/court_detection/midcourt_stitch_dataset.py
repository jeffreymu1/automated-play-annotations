"""DeepSport mid-court stitching augmentation.

The source DeepSport frames are calibrated views of one side of the court. This
dataset pairs left-side and right-side frames from the same game, undistorts
them, reprojects both through an averaged virtual camera, and blends the
overlap to create a continuous mid-court view. The virtual camera can be
panned between the source views: 0.0 matches the left view and 1.0 matches the
right view.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from torch.utils.data import Dataset

from court_detection.dataset import DeepSportDataset
from court_detection.geometry import (
    COURT_LENGTH_CM,
    COURT_WIDTH_CM,
    KEYPOINT_NAMES,
    CameraCalibration,
    court_keypoints_world,
    project_world_to_image,
)

CourtSide = Literal["left", "right"]


@dataclass(frozen=True)
class MidcourtSourceFrame:
    base_index: int
    image_path: Path
    json_path: Path
    game: str
    segment: str
    side: CourtSide
    look_at_x_cm: float
    look_at_y_cm: float


@dataclass(frozen=True)
class MidcourtFramePair:
    game: str
    left: MidcourtSourceFrame
    right: MidcourtSourceFrame
    camera_portion: float = 0.5


@dataclass(frozen=True)
class RenderedMidcourtFrame:
    image: np.ndarray
    keypoints: np.ndarray
    calibration: CameraCalibration
    pair: MidcourtFramePair
    overlap_pixels: int
    camera_portion: float
    annotation_occlusion_mask: np.ndarray


def _camera_center(calib: CameraCalibration) -> np.ndarray:
    return -calib.R.T @ calib.T


def _optical_axis_world(calib: CameraCalibration) -> np.ndarray:
    return calib.R.T @ np.array([0.0, 0.0, 1.0], dtype=float)


def _look_at_court_plane(calib: CameraCalibration) -> np.ndarray | None:
    center = _camera_center(calib)
    direction = _optical_axis_world(calib)
    if abs(float(direction[2])) < 1e-8:
        return None
    t = -float(center[2]) / float(direction[2])
    if t <= 0.0:
        return None
    return center + t * direction


def classify_court_side(calib: CameraCalibration) -> tuple[CourtSide, np.ndarray]:
    """Classify whether the camera is aimed at the left or right half-court."""
    look_at = _look_at_court_plane(calib)
    if look_at is None:
        half_centers = np.array([
            [0.25 * COURT_LENGTH_CM, 0.5 * COURT_WIDTH_CM, 0.0],
            [0.75 * COURT_LENGTH_CM, 0.5 * COURT_WIDTH_CM, 0.0],
        ])
        uv = project_world_to_image(half_centers, calib)
        image_center = np.array([0.5 * calib.width, 0.5 * calib.height], dtype=float)
        distances = np.linalg.norm(uv - image_center, axis=1)
        if not np.isfinite(distances).all():
            distances = np.where(np.isfinite(distances), distances, np.inf)
        side: CourtSide = "left" if distances[0] <= distances[1] else "right"
        fallback_x = half_centers[0 if side == "left" else 1, 0]
        return side, np.array([fallback_x, 0.5 * COURT_WIDTH_CM, 0.0], dtype=float)

    side = "left" if float(look_at[0]) < 0.5 * COURT_LENGTH_CM else "right"
    return side, look_at


def _plane_homography_world_to_image(calib: CameraCalibration) -> np.ndarray:
    rt_plane = np.column_stack([calib.R[:, 0], calib.R[:, 1], calib.T])
    return calib.K @ rt_plane


def _average_intrinsics(
    left: CameraCalibration,
    right: CameraCalibration,
    output_size: tuple[int, int] | None,
    camera_portion: float = 0.5,
) -> tuple[np.ndarray, int, int]:
    t = _validate_camera_portion(camera_portion)
    width = int(round((1.0 - t) * left.width + t * right.width))
    height = int(round((1.0 - t) * left.height + t * right.height))

    fx = float(
        np.exp(
            (1.0 - t) * np.log(max(left.K[0, 0], 1e-12))
            + t * np.log(max(right.K[0, 0], 1e-12))
        )
    )
    fy = float(
        np.exp(
            (1.0 - t) * np.log(max(left.K[1, 1], 1e-12))
            + t * np.log(max(right.K[1, 1], 1e-12))
        )
    )
    skew = float((1.0 - t) * left.K[0, 1] + t * right.K[0, 1])
    cx = float((1.0 - t) * left.K[0, 2] + t * right.K[0, 2])
    cy = float((1.0 - t) * left.K[1, 2] + t * right.K[1, 2])

    K = np.array(
        [
            [fx, skew, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    if output_size is not None:
        out_h, out_w = output_size
        sx = float(out_w) / float(width)
        sy = float(out_h) / float(height)
        K[0, :] *= sx
        K[1, :] *= sy
        width = int(out_w)
        height = int(out_h)

    return K, width, height


def _validate_camera_portion(camera_portion: float) -> float:
    value = float(camera_portion)
    if not np.isfinite(value):
        raise ValueError("camera_portion must be finite")
    if value < 0.0 or value > 1.0:
        raise ValueError(
            "camera_portion must be in [0, 1], where 0 is left and 1 is right"
        )
    return value


def _validate_camera_portion_range(
    camera_portion_range: tuple[float, float],
) -> tuple[float, float]:
    lo = _validate_camera_portion(camera_portion_range[0])
    hi = _validate_camera_portion(camera_portion_range[1])
    if lo > hi:
        raise ValueError("camera_portion_range must be ordered from low to high")
    return lo, hi


def _interpolate_extrinsics(
    left: CameraCalibration,
    right: CameraCalibration,
    camera_portion: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    t = _validate_camera_portion(camera_portion)
    center = (1.0 - t) * _camera_center(left) + t * _camera_center(right)
    rotations = Rotation.from_matrix(np.stack([left.R, right.R], axis=0))
    rotation = Slerp([0.0, 1.0], rotations)([t]).as_matrix()[0]
    translation = -rotation @ center
    return rotation, translation


def virtual_calibration_between(
    left: CameraCalibration,
    right: CameraCalibration,
    camera_portion: float = 0.5,
    output_size: tuple[int, int] | None = None,
) -> CameraCalibration:
    """Interpolate an undistorted virtual camera between left=0 and right=1."""
    t = _validate_camera_portion(camera_portion)
    R, T = _interpolate_extrinsics(left, right, t)
    K, width, height = _average_intrinsics(left, right, output_size, t)
    return CameraCalibration(
        K=K,
        R=R,
        T=T,
        kc=np.zeros(5, dtype=float),
        width=width,
        height=height,
    )


def average_virtual_calibration(
    left: CameraCalibration,
    right: CameraCalibration,
    output_size: tuple[int, int] | None = None,
) -> CameraCalibration:
    """Average two calibrated side cameras into an undistorted virtual camera."""
    return virtual_calibration_between(left, right, camera_portion=0.5, output_size=output_size)


def _load_undistorted_rgb(path: Path, calib: CameraCalibration) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR_RGB)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image_f = image.astype(np.float32) / 255.0
    return cv2.undistort(image_f, calib.K, calib.kc, None, calib.K).astype(np.float32)


def _load_undistorted_annotation_occlusion_mask(
    json_path: Path,
    calib: CameraCalibration,
) -> np.ndarray:
    mask = DeepSportDataset.load_annotation_occlusion_mask(json_path)
    if not mask.size:
        return np.zeros((calib.height, calib.width), dtype=bool)
    undistorted = cv2.undistort(
        mask.astype(np.uint8),
        calib.K,
        calib.kc,
        None,
        calib.K,
    )
    return undistorted > 0


def _warp_to_virtual(
    image: np.ndarray,
    source: CameraCalibration,
    target: CameraCalibration,
) -> tuple[np.ndarray, np.ndarray]:
    h_src, w_src = image.shape[:2]
    h_tgt, w_tgt = target.height, target.width
    H_source = _plane_homography_world_to_image(source)
    H_target = _plane_homography_world_to_image(target)
    H_source_to_target = H_target @ np.linalg.inv(H_source)

    warped = cv2.warpPerspective(
        image,
        H_source_to_target,
        (w_tgt, h_tgt),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    ).astype(np.float32)
    valid = cv2.warpPerspective(
        np.ones((h_src, w_src), dtype=np.uint8),
        H_source_to_target,
        (w_tgt, h_tgt),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 0
    return warped, valid


def _warp_mask_to_virtual(
    mask: np.ndarray,
    source: CameraCalibration,
    target: CameraCalibration,
) -> np.ndarray:
    h_tgt, w_tgt = target.height, target.width
    H_source = _plane_homography_world_to_image(source)
    H_target = _plane_homography_world_to_image(target)
    H_source_to_target = H_target @ np.linalg.inv(H_source)

    warped = cv2.warpPerspective(
        mask.astype(np.uint8),
        H_source_to_target,
        (w_tgt, h_tgt),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return warped > 0


def _right_half_alpha(target: CameraCalibration, overlap: np.ndarray) -> np.ndarray:
    h, w = overlap.shape
    alpha = np.full((h, w), 0.5, dtype=np.float32)
    if not overlap.any():
        return alpha

    xs = np.flatnonzero(overlap.any(axis=0))
    if xs.size <= 1:
        return alpha

    lo = int(xs.min())
    hi = int(xs.max())
    ramp = (np.arange(w, dtype=np.float32) - float(lo)) / max(float(hi - lo), 1.0)
    ramp = np.clip(ramp, 0.0, 1.0)

    half_centers = np.array([
        [0.25 * COURT_LENGTH_CM, 0.5 * COURT_WIDTH_CM, 0.0],
        [0.75 * COURT_LENGTH_CM, 0.5 * COURT_WIDTH_CM, 0.0],
    ])
    uv = project_world_to_image(half_centers, target)
    if np.isfinite(uv).all() and uv[1, 0] < uv[0, 0]:
        ramp = 1.0 - ramp

    alpha = np.broadcast_to(ramp.reshape(1, w), (h, w)).astype(np.float32).copy()
    alpha[~overlap] = 0.5
    return alpha


def _blend_warped_halves(
    left_image: np.ndarray,
    left_valid: np.ndarray,
    right_image: np.ndarray,
    right_valid: np.ndarray,
    target: CameraCalibration,
) -> tuple[np.ndarray, int]:
    left_weight = left_valid.astype(np.float32)
    right_weight = right_valid.astype(np.float32)
    overlap = left_valid & right_valid

    if overlap.any():
        alpha_right = _right_half_alpha(target, overlap)
        left_weight[overlap] = 1.0 - alpha_right[overlap]
        right_weight[overlap] = alpha_right[overlap]

    denom = left_weight + right_weight
    safe_denom = np.where(denom > 0.0, denom, 1.0)
    blended = (
        left_image * left_weight[..., None]
        + right_image * right_weight[..., None]
    ) / safe_denom[..., None]
    blended[denom <= 0.0] = 0.0
    return blended.clip(0.0, 1.0).astype(np.float32), int(overlap.sum())


def _project_keypoints(calib: CameraCalibration) -> np.ndarray:
    uvs = project_world_to_image(court_keypoints_world(), calib)
    keypoints = np.zeros((len(KEYPOINT_NAMES), 3), dtype=np.float32)
    for i, uv in enumerate(uvs):
        u, v = float(uv[0]), float(uv[1])
        if np.isfinite(uv).all() and 0.0 <= u < calib.width and 0.0 <= v < calib.height:
            keypoints[i] = (u, v, 2.0)
    return keypoints


class DeepSportMidcourtStitchDataset(Dataset):
    """Random same-game left/right stitches with DeepSportDataset-style output."""

    def __init__(
        self,
        root: Path | str = Path("data/deepsport-dataset"),
        *,
        base: DeepSportDataset | None = None,
        pairs_per_game: int = 5,
        seed: int = 1430,
        output_size: tuple[int, int] | None = None,
        camera_portion: float | None = None,
        camera_portion_range: tuple[float, float] = (0.0, 1.0),
        return_annotation_occlusion_mask: bool = False,
    ) -> None:
        if pairs_per_game <= 0:
            raise ValueError("pairs_per_game must be positive")
        self.base = base if base is not None else DeepSportDataset(root)
        self.root = self.base.root
        self.pairs_per_game = int(pairs_per_game)
        self.seed = int(seed)
        self.output_size = output_size
        self.camera_portion = (
            None if camera_portion is None else _validate_camera_portion(camera_portion)
        )
        self.camera_portion_range = _validate_camera_portion_range(camera_portion_range)
        self.return_annotation_occlusion_mask = bool(return_annotation_occlusion_mask)

        self.frames_by_game = self._group_frames_by_game()
        self.games = tuple(
            game
            for game, groups in sorted(self.frames_by_game.items())
            if groups["left"] and groups["right"]
        )
        if not self.games:
            raise RuntimeError(f"No games with both left and right frames found under {self.root}")

        self.pairs: list[MidcourtFramePair] = []
        self.refresh_pairs()

    def _group_frames_by_game(self) -> dict[str, dict[CourtSide, list[MidcourtSourceFrame]]]:
        grouped: dict[str, dict[CourtSide, list[MidcourtSourceFrame]]] = defaultdict(
            lambda: {"left": [], "right": []}
        )
        for base_index, (image_path, json_path) in enumerate(self.base.samples):
            try:
                rel = image_path.relative_to(self.root)
                game, segment = rel.parts[0], rel.parts[1]
            except (ValueError, IndexError):
                continue

            calib = CameraCalibration.from_json(json_path)
            side, look_at = classify_court_side(calib)
            grouped[game][side].append(
                MidcourtSourceFrame(
                    base_index=base_index,
                    image_path=image_path,
                    json_path=json_path,
                    game=game,
                    segment=segment,
                    side=side,
                    look_at_x_cm=float(look_at[0]),
                    look_at_y_cm=float(look_at[1]),
                )
            )
        return dict(grouped)

    def refresh_pairs(self, seed: int | None = None) -> None:
        rng = np.random.default_rng(self.seed if seed is None else seed)
        pairs: list[MidcourtFramePair] = []
        for game in self.games:
            lefts = self.frames_by_game[game]["left"]
            rights = self.frames_by_game[game]["right"]
            for _ in range(self.pairs_per_game):
                left = lefts[int(rng.integers(len(lefts)))]
                right = rights[int(rng.integers(len(rights)))]
                camera_portion = self._sample_camera_portion(rng)
                pairs.append(
                    MidcourtFramePair(
                        game=game,
                        left=left,
                        right=right,
                        camera_portion=camera_portion,
                    )
                )
        self.pairs = pairs

    def __len__(self) -> int:
        return len(self.pairs)

    def _sample_camera_portion(self, rng: np.random.Generator) -> float:
        if self.camera_portion is not None:
            return self.camera_portion
        lo, hi = self.camera_portion_range
        return float(rng.uniform(lo, hi))

    def set_camera_portion(
        self,
        camera_portion: float | None = None,
        *,
        camera_portion_range: tuple[float, float] | None = None,
        seed: int | None = None,
    ) -> None:
        """Set a fixed pan fraction, or pass None to randomize within a range."""
        self.camera_portion = (
            None if camera_portion is None else _validate_camera_portion(camera_portion)
        )
        if camera_portion_range is not None:
            self.camera_portion_range = _validate_camera_portion_range(camera_portion_range)
        self.refresh_pairs(seed=seed)

    def render_pair_at_portion(
        self,
        pair: MidcourtFramePair,
        camera_portion: float,
    ) -> RenderedMidcourtFrame:
        """Render a pair from a specified pan fraction: 0 is left, 1 is right."""
        return self.render_pair(pair, camera_portion=camera_portion)

    def render_pair(
        self,
        pair: MidcourtFramePair,
        camera_portion: float | None = None,
    ) -> RenderedMidcourtFrame:
        target_portion = (
            _validate_camera_portion(pair.camera_portion)
            if camera_portion is None
            else _validate_camera_portion(camera_portion)
        )
        left_calib = CameraCalibration.from_json(pair.left.json_path)
        right_calib = CameraCalibration.from_json(pair.right.json_path)
        target_calib = virtual_calibration_between(
            left_calib,
            right_calib,
            camera_portion=target_portion,
            output_size=self.output_size,
        )

        left_image = _load_undistorted_rgb(pair.left.image_path, left_calib)
        right_image = _load_undistorted_rgb(pair.right.image_path, right_calib)

        left_warped, left_valid = _warp_to_virtual(left_image, left_calib, target_calib)
        right_warped, right_valid = _warp_to_virtual(right_image, right_calib, target_calib)
        image, overlap_pixels = _blend_warped_halves(
            left_warped,
            left_valid,
            right_warped,
            right_valid,
            target_calib,
        )
        annotation_occlusion_mask = np.zeros((target_calib.height, target_calib.width), dtype=bool)
        if self.return_annotation_occlusion_mask:
            left_occlusion = _load_undistorted_annotation_occlusion_mask(pair.left.json_path, left_calib)
            right_occlusion = _load_undistorted_annotation_occlusion_mask(pair.right.json_path, right_calib)
            left_occlusion_warped = _warp_mask_to_virtual(left_occlusion, left_calib, target_calib)
            right_occlusion_warped = _warp_mask_to_virtual(right_occlusion, right_calib, target_calib)
            annotation_occlusion_mask = (
                (left_occlusion_warped & left_valid)
                | (right_occlusion_warped & right_valid)
            )
        return RenderedMidcourtFrame(
            image=image,
            keypoints=_project_keypoints(target_calib),
            calibration=target_calib,
            pair=pair,
            overlap_pixels=overlap_pixels,
            camera_portion=target_portion,
            annotation_occlusion_mask=annotation_occlusion_mask,
        )

    def sample_label(self, idx: int) -> str:
        pair = self.pairs[idx]
        return (
            f"midcourt:{pair.game}:"
            f"{pair.left.image_path.stem}+{pair.right.image_path.stem}:"
            f"pan={pair.camera_portion:.3f}"
        )

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, CameraCalibration] | tuple[
        torch.Tensor,
        torch.Tensor,
        CameraCalibration,
        torch.Tensor,
    ]:
        rendered = self.render_pair(self.pairs[idx])
        image = torch.from_numpy(rendered.image).permute(2, 0, 1).contiguous().to(torch.float32)
        keypoints = torch.from_numpy(rendered.keypoints)
        if not self.return_annotation_occlusion_mask:
            return image, keypoints, rendered.calibration
        occlusion = torch.from_numpy(rendered.annotation_occlusion_mask)
        return image, keypoints, rendered.calibration, occlusion


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "sample"


def _calibration_to_jsonable(calib: CameraCalibration) -> dict[str, object]:
    return {
        "KK": calib.K.tolist(),
        "R": calib.R.tolist(),
        "T": calib.T.tolist(),
        "kc": calib.kc.tolist(),
        "img_width": calib.width,
        "img_height": calib.height,
    }


def save_stitched_examples(
    root: Path,
    out: Path,
    *,
    samples_per_game: int = 5,
    seed: int = 1430,
    camera_portion: float | None = None,
    camera_portion_range: tuple[float, float] = (0.0, 1.0),
) -> None:
    dataset = DeepSportMidcourtStitchDataset(
        root,
        pairs_per_game=samples_per_game,
        seed=seed,
        camera_portion=camera_portion,
        camera_portion_range=camera_portion_range,
    )
    out.mkdir(parents=True, exist_ok=True)
    per_game_count: dict[str, int] = defaultdict(int)

    for pair in dataset.pairs:
        rendered = dataset.render_pair(pair)
        game_dir = out / _safe_name(pair.game)
        game_dir.mkdir(parents=True, exist_ok=True)
        index = per_game_count[pair.game]
        per_game_count[pair.game] += 1

        stem = f"midcourt_stitch_{index:03d}"
        image_path = game_dir / f"{stem}.png"
        image_u8 = np.clip(np.round(rendered.image * 255.0), 0, 255).astype(np.uint8)
        cv2.imwrite(str(image_path), cv2.cvtColor(image_u8, cv2.COLOR_RGB2BGR))

        metadata = {
            "game": pair.game,
            "left": {
                "image_path": str(pair.left.image_path),
                "json_path": str(pair.left.json_path),
                "segment": pair.left.segment,
                "look_at_x_cm": pair.left.look_at_x_cm,
                "look_at_y_cm": pair.left.look_at_y_cm,
            },
            "right": {
                "image_path": str(pair.right.image_path),
                "json_path": str(pair.right.json_path),
                "segment": pair.right.segment,
                "look_at_x_cm": pair.right.look_at_x_cm,
                "look_at_y_cm": pair.right.look_at_y_cm,
            },
            "overlap_pixels": rendered.overlap_pixels,
            "camera_portion": rendered.camera_portion,
            "calibration": _calibration_to_jsonable(rendered.calibration),
            "keypoints": rendered.keypoints.tolist(),
        }
        (game_dir / f"{stem}.json").write_text(json.dumps(metadata, indent=2))
        print(f"wrote {image_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate same-game stitched mid-court DeepSport frames."
    )
    parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    parser.add_argument("--out", type=Path, default=Path("tests/midcourt_stitches"))
    parser.add_argument("--samples-per-game", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1430)
    parser.add_argument(
        "--camera-portion",
        type=float,
        default=None,
        help="Fixed virtual camera pan: 0 is left, 1 is right. Default randomizes.",
    )
    parser.add_argument(
        "--camera-portion-range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.0, 1.0),
        help="Uniform random pan range used when --camera-portion is omitted.",
    )
    args = parser.parse_args()
    save_stitched_examples(
        args.root,
        args.out,
        samples_per_game=args.samples_per_game,
        seed=args.seed,
        camera_portion=args.camera_portion,
        camera_portion_range=tuple(args.camera_portion_range),
    )


if __name__ == "__main__":
    main()
