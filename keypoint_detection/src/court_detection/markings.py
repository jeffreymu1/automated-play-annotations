"""Expanded FIBA court-marking model definitions."""

from __future__ import annotations

import copy
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
from torch.utils.data import ConcatDataset, Dataset

from court_detection.dataset import DeepSportDataset
from court_detection.geometry import MARKING_CLASS_NAMES
from court_detection.lines import (
    CourtLineDataModule,
    CourtLineFrameDataset,
    CourtLineLightning,
    FrozenDinoUNet,
    class_palette,
    focal_bce_loss,
    gated_cross_entropy,
    overlay_line_predictions,
    soft_dice_loss,
)
from court_detection.midcourt_stitch_dataset import DeepSportMidcourtStitchDataset
from court_detection.structured_side import StructuredSideCourtLineLightning, StructuredSideDinoUNet

FIBA_MARKING_NAMES = MARKING_CLASS_NAMES


class _SizedDataset(Dataset):
    """Expose a deterministic epoch length while cycling through a source dataset."""

    def __init__(self, dataset: Dataset, length: int) -> None:
        if length <= 0:
            raise ValueError("length must be positive")
        if len(dataset) <= 0:
            raise ValueError("dataset must be non-empty")
        self.dataset = dataset
        self.length = int(length)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[idx % len(self.dataset)]


def _subset_deepsport_dataset(base: DeepSportDataset, indices: list[int]) -> DeepSportDataset:
    subset = copy.copy(base)
    subset.samples = [base.samples[i] for i in indices]
    subset.clips = base._discover_clips()
    subset._clips_by_name = {}
    for i, clip in enumerate(subset.clips):
        subset._clips_by_name.setdefault(clip.name, i)
    return subset


class FibaCourtMarkingDataModule(CourtLineDataModule):
    """DeepSport data module for the expanded FIBA marking target set."""

    def __init__(
        self,
        root: Path,
        image_size: tuple[int, int] = (384, 640),
        output_stride: int = 2,
        sigma: float = 1.5,
        batch_size: int = 4,
        num_workers: int = 4,
        seed: int = 1430,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
        n_samples_per_line: int = 400,
        side_blur_sigma: float = 1.0,
        line_names: tuple[str, ...] = FIBA_MARKING_NAMES,
        use_player_occlusion: bool = False,
        pan_train_ratio: float = 1.0,
        pan_camera_portion_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        super().__init__(
            root=root,
            image_size=image_size,
            output_stride=output_stride,
            sigma=sigma,
            batch_size=batch_size,
            num_workers=num_workers,
            seed=seed,
            val_fraction=val_fraction,
            test_fraction=test_fraction,
            line_names=line_names,
            n_samples_per_line=n_samples_per_line,
            side_blur_sigma=side_blur_sigma,
            use_player_occlusion=use_player_occlusion,
        )
        if pan_train_ratio < 0.0:
            raise ValueError("pan_train_ratio must be non-negative")
        self.pan_train_ratio = float(pan_train_ratio)
        self.pan_camera_portion_range = tuple(float(v) for v in pan_camera_portion_range)

    def setup(self, stage: str | None = None) -> None:
        base = DeepSportDataset(self.root)
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(base)).tolist()

        n_total = len(indices)
        n_test = max(1, int(round(n_total * self.test_fraction)))
        n_val = max(1, int(round(n_total * self.val_fraction)))
        n_train = max(1, n_total - n_val - n_test)

        train_indices = indices[:n_train]
        val_indices = indices[n_train:n_train + n_val]
        test_indices = indices[n_train + n_val:]
        if not test_indices:
            test_indices = val_indices

        kwargs = dict(
            image_size=self.image_size,
            output_stride=self.output_stride,
            sigma=self.sigma,
            line_names=self.line_names,
            class_by_geometry=self.class_by_geometry,
            n_samples_per_line=self.n_samples_per_line,
            side_blur_sigma=self.side_blur_sigma,
            use_player_occlusion=self.use_player_occlusion,
        )
        real_train = CourtLineFrameDataset(base, train_indices, augment=True, **kwargs)
        self.train_dataset = real_train
        if stage in (None, "fit") and self.pan_train_ratio > 0.0:
            self.train_dataset = self._make_augmented_train_dataset(
                base,
                train_indices,
                real_train,
                kwargs,
            )
        self.val_dataset = CourtLineFrameDataset(base, val_indices, augment=False, **kwargs)
        self.test_dataset = CourtLineFrameDataset(base, test_indices, augment=False, **kwargs)

    def _make_augmented_train_dataset(
        self,
        base: DeepSportDataset,
        train_indices: list[int],
        real_train: CourtLineFrameDataset,
        kwargs: dict[str, Any],
    ) -> Dataset:
        target_pan_len = max(1, int(round(len(real_train) * self.pan_train_ratio)))
        train_base = _subset_deepsport_dataset(base, train_indices)
        try:
            probe = DeepSportMidcourtStitchDataset(
                self.root,
                base=train_base,
                pairs_per_game=1,
                seed=self.seed,
                camera_portion_range=self.pan_camera_portion_range,
                return_annotation_occlusion_mask=self.use_player_occlusion,
            )
            pairs_per_game = max(1, math.ceil(math.ceil(target_pan_len / 2) / len(probe.games)))
            pan_base = DeepSportMidcourtStitchDataset(
                self.root,
                base=train_base,
                pairs_per_game=pairs_per_game,
                seed=self.seed + 1,
                camera_portion_range=self.pan_camera_portion_range,
                return_annotation_occlusion_mask=self.use_player_occlusion,
            )
        except RuntimeError as exc:
            warnings.warn(
                f"Skipping panned training augmentation because no valid train pairs were found: {exc}",
                stacklevel=2,
            )
            return real_train
        pan_indices = list(range(len(pan_base)))
        pan_train = CourtLineFrameDataset(pan_base, pan_indices, augment=True, **kwargs)
        return ConcatDataset([real_train, _SizedDataset(pan_train, target_pan_len)])


class FibaCourtMarkingLightning(CourtLineLightning):
    """Frozen-DINO U-Net with shared marking classes plus a court-side head."""

    def __init__(
        self,
        line_names: tuple[str, ...] = FIBA_MARKING_NAMES,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("num_classes", len(line_names))
        super().__init__(line_names=line_names, **kwargs)


class FibaStructuredSideCourtMarkingLightning(StructuredSideCourtLineLightning):
    """FIBA marking model with a constrained linear court-side boundary."""

    def __init__(
        self,
        line_names: tuple[str, ...] = FIBA_MARKING_NAMES,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("num_classes", len(line_names))
        super().__init__(line_names=line_names, **kwargs)


__all__ = [
    "CourtLineFrameDataset",
    "FIBA_MARKING_NAMES",
    "FibaCourtMarkingDataModule",
    "FibaCourtMarkingLightning",
    "FibaStructuredSideCourtMarkingLightning",
    "FrozenDinoUNet",
    "StructuredSideDinoUNet",
    "class_palette",
    "focal_bce_loss",
    "gated_cross_entropy",
    "overlay_line_predictions",
    "soft_dice_loss",
]
