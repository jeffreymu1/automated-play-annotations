"""Expanded FIBA court-marking model definitions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

FIBA_MARKING_NAMES = MARKING_CLASS_NAMES


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
        )


class FibaCourtMarkingLightning(CourtLineLightning):
    """Frozen-DINO U-Net with shared marking classes plus a court-side head."""

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
    "FrozenDinoUNet",
    "class_palette",
    "focal_bce_loss",
    "gated_cross_entropy",
    "overlay_line_predictions",
    "soft_dice_loss",
]
