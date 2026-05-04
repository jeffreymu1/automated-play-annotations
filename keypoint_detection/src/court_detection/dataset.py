"""Dataset readers and shared helpers for DeepSport court frames."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from court_detection.geometry import (
    CameraCalibration,
    KEYPOINT_NAMES,
    court_keypoints_world,
    project_world_to_image,
)

Keypoints = torch.Tensor


@dataclass(frozen=True)
class DeepSportClipFrame:
    image_path: Path
    json_path: Path
    frame_number: int
    camera: str
    timestamp: int | None = None


@dataclass(frozen=True)
class DeepSportClip:
    game: str
    segment: str
    name: str
    frames: tuple[DeepSportClipFrame, ...]

    @property
    def cameras(self) -> tuple[str, ...]:
        return tuple(sorted({frame.camera for frame in self.frames}))


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
        self.clips = self._discover_clips()
        self._clips_by_name: dict[str, int] = {}
        for i, clip in enumerate(self.clips):
            self._clips_by_name.setdefault(clip.name, i)

    def _discover_clips(self) -> list[DeepSportClip]:
        clips: list[DeepSportClip] = []
        for clip_dir in sorted(path for path in self.root.glob("*/*") if path.is_dir()):
            frames = self._clip_frames_from_dir(clip_dir)
            if not frames:
                continue
            try:
                rel = clip_dir.relative_to(self.root)
                game, segment = rel.parts[0], rel.parts[1]
            except (ValueError, IndexError):
                game, segment = clip_dir.parent.name, clip_dir.name
            clips.append(DeepSportClip(game=game, segment=segment, name=segment, frames=tuple(frames)))
        return clips

    def _clip_frames_from_dir(self, clip_dir: Path) -> list[DeepSportClipFrame]:
        frames: list[DeepSportClipFrame] = []
        for frame_path in sorted(clip_dir.glob("*.png"), key=self._frame_sort_key):
            stem, sep, suffix = frame_path.stem.rpartition("_")
            if sep == "" or not suffix.isdigit():
                continue
            json_path = frame_path.with_name(f"{stem}.json")
            if json_path.is_file():
                camera, timestamp = self._camera_timestamp_from_stem(stem)
                frames.append(DeepSportClipFrame(frame_path, json_path, int(suffix), camera, timestamp))
        return frames

    @staticmethod
    def _frame_sort_key(frame_path: Path) -> tuple[int, int, str, str]:
        parts = frame_path.stem.rsplit("_", 2)
        if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
            return int(parts[1]), int(parts[2]), parts[0], frame_path.name
        return 0, 0, frame_path.stem, frame_path.name

    @staticmethod
    def _camera_timestamp_from_stem(stem: str) -> tuple[str, int | None]:
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            return parts[0], int(parts[1])
        return stem, None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, Keypoints, CameraCalibration]:
        image_path, json_path = self.samples[idx]
        calib = CameraCalibration.from_json(json_path)

        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR_RGB)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        image = torch.from_numpy(img).permute(2, 0, 1).contiguous().to(torch.float32).div_(255.0)

        uvs = project_world_to_image(court_keypoints_world(), calib)
        keypoints = np.zeros((len(KEYPOINT_NAMES), 3), dtype=np.float32)
        for i, uv in enumerate(uvs):
            u, v = float(uv[0]), float(uv[1])
            if np.isfinite(uv).all() and 0.0 <= u < calib.width and 0.0 <= v < calib.height:
                keypoints[i] = (u, v, 2.0)

        return image, torch.from_numpy(keypoints), calib

    def get_clip(self, idx_or_name: int | str) -> DeepSportClip:
        if isinstance(idx_or_name, int):
            return self.clips[idx_or_name]
        if idx_or_name in self._clips_by_name:
            return self.clips[self._clips_by_name[idx_or_name]]
        for i, clip in enumerate(self.clips):
            identifiers = (
                f"{clip.game}/{clip.segment}/{clip.name}",
                f"{clip.game}/{clip.segment}",
                f"{clip.game}/{clip.name}",
            )
            if idx_or_name in identifiers:
                return self.clips[i]
        raise KeyError(f"Unknown clip: {idx_or_name}")

    def load_clip_frames(
        self,
        idx_or_name: int | str,
        camera: str | None = None,
    ) -> list[tuple[np.ndarray, DeepSportClipFrame, CameraCalibration]]:
        clip = self.get_clip(idx_or_name)
        selected = [frame for frame in clip.frames if camera is None or frame.camera == camera]
        if camera is not None and not selected:
            raise KeyError(f"Camera {camera!r} not found in clip {clip.name}; available: {clip.cameras}")
        frames: list[tuple[np.ndarray, DeepSportClipFrame, CameraCalibration]] = []
        for frame in selected:
            calib = CameraCalibration.from_json(frame.json_path)
            img = cv2.imread(str(frame.image_path), cv2.IMREAD_COLOR_RGB)
            if img is None:
                raise FileNotFoundError(f"Could not read image: {frame.image_path}")
            frames.append((img.astype(np.float32) / 255.0, frame, calib))
        return frames


def keypoints_to_dict(keypoints: torch.Tensor | np.ndarray) -> dict[str, tuple[float, float] | None]:
    arr = keypoints.detach().cpu().numpy() if isinstance(keypoints, torch.Tensor) else np.asarray(keypoints)
    out: dict[str, tuple[float, float] | None] = {}
    for name, row in zip(KEYPOINT_NAMES, arr):
        out[name] = (float(row[0]), float(row[1])) if row[2] > 0 else None
    return out
