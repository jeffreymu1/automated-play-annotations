"""Heatmap model, training module, and decoding helpers."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import cv2
import lightning as L
import numpy as np
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from court_detection.dataset import DeepSportDataset
from court_detection.geometry import KEYPOINT_NAMES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise AssertionError("unreachable")


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


class CourtKeypointFrameDataset(Dataset):
    """Resize and augment DeepSport frames while keeping keypoints aligned."""

    def __init__(
        self,
        base: DeepSportDataset,
        indices: list[int],
        image_size: tuple[int, int],
        augment: bool = False,
    ) -> None:
        self.base = base
        self.indices = indices
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        base_idx = self.indices[idx]
        image, keypoints, _ = self.base[base_idx]
        image_np = image.permute(1, 2, 0).numpy()
        keypoints_np = keypoints.numpy().copy()

        image_np, keypoints_np = self._crop_resize(image_np, keypoints_np)
        if self.augment:
            image_np = self._augment_color(image_np)

        image_t = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float()
        keypoints_t = torch.from_numpy(keypoints_np).float()
        image_path, _ = self.base.samples[base_idx]
        return {
            "image": image_t,
            "keypoints": keypoints_t,
            "index": base_idx,
            "image_path": str(image_path),
        }

    def _crop_resize(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        in_h, in_w = image.shape[:2]
        out_h, out_w = self.image_size

        if self.augment:
            target_aspect = out_w / out_h
            crop_h = random.randint(max(2, int(0.72 * in_h)), in_h)
            crop_w = int(round(crop_h * target_aspect))
            if crop_w > in_w:
                crop_w = in_w
                crop_h = min(in_h, int(round(crop_w / target_aspect)))
            x0 = random.randint(0, max(0, in_w - crop_w))
            y0 = random.randint(0, max(0, in_h - crop_h))
        else:
            x0, y0 = 0, 0
            crop_w, crop_h = in_w, in_h

        cropped = image[y0:y0 + crop_h, x0:x0 + crop_w]
        interpolation = cv2.INTER_AREA if out_w < crop_w or out_h < crop_h else cv2.INTER_LINEAR
        resized = cv2.resize(cropped, (out_w, out_h), interpolation=interpolation)

        kp = keypoints.copy()
        visible = kp[:, 2] > 0
        kp[:, 0] = (kp[:, 0] - x0) * (out_w / crop_w)
        kp[:, 1] = (kp[:, 1] - y0) * (out_h / crop_h)
        visible &= (kp[:, 0] >= 0.0) & (kp[:, 0] < out_w)
        visible &= (kp[:, 1] >= 0.0) & (kp[:, 1] < out_h)
        kp[~visible, :2] = 0.0
        kp[~visible, 2] = 0.0

        return resized.clip(0.0, 1.0).astype(np.float32), kp.astype(np.float32)

    def _augment_color(self, image: np.ndarray) -> np.ndarray:
        img = image.copy()
        img = img * random.uniform(0.80, 1.20) + random.uniform(-0.08, 0.08)

        mean = img.mean(axis=(0, 1), keepdims=True)
        img = (img - mean) * random.uniform(0.75, 1.25) + mean

        gray = np.dot(img, np.array([0.299, 0.587, 0.114], dtype=np.float32))[..., None]
        img = gray + (img - gray) * random.uniform(0.75, 1.25)

        if random.random() < 0.15:
            img = cv2.GaussianBlur(img, (3, 3), sigmaX=random.uniform(0.2, 1.0))

        if random.random() < 0.35:
            h, w = img.shape[:2]
            for _ in range(random.randint(1, 3)):
                rect_w = random.randint(max(8, w // 24), max(12, w // 8))
                rect_h = random.randint(max(8, h // 24), max(12, h // 7))
                x0 = random.randint(0, max(0, w - rect_w))
                y0 = random.randint(0, max(0, h - rect_h))
                color = np.array([random.random(), random.random(), random.random()], dtype=np.float32)
                img[y0:y0 + rect_h, x0:x0 + rect_w] = 0.65 * img[y0:y0 + rect_h, x0:x0 + rect_w] + 0.35 * color

        return img.clip(0.0, 1.0).astype(np.float32)


class CourtKeypointDataModule(L.LightningDataModule):
    def __init__(
        self,
        root: Path,
        image_size: tuple[int, int] = (384, 640),
        batch_size: int = 8,
        num_workers: int = 4,
        seed: int = 1430,
        val_fraction: float = 0.15,
        test_fraction: float = 0.15,
    ) -> None:
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction

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

        self.train_dataset = CourtKeypointFrameDataset(base, train_indices, self.image_size, augment=True)
        self.val_dataset = CourtKeypointFrameDataset(base, val_indices, self.image_size, augment=False)
        self.test_dataset = CourtKeypointFrameDataset(base, test_indices, self.image_size, augment=False)

    def _loader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        generator = torch.Generator().manual_seed(self.seed)
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
            worker_init_fn=_seed_worker,
            generator=generator,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_dataset, shuffle=False)


class FrozenDinoHeatmapNet(nn.Module):
    def __init__(
        self,
        num_keypoints: int,
        model_name: str = "convnext_base.dinov3_lvd1689m",
        pretrained: bool = True,
        hidden_channels: int = 256,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        backbone_channels = self.backbone.feature_info.channels()[-1]

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        self.decoder = nn.Sequential(
            nn.Conv2d(backbone_channels, hidden_channels, kernel_size=1),
            _norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            _norm(hidden_channels),
            nn.GELU(),
            nn.Conv2d(hidden_channels, decoder_channels, kernel_size=3, padding=1),
            _norm(decoder_channels),
            nn.GELU(),
        )
        self.keypoint_head = nn.Sequential(
            nn.Conv2d(decoder_channels, num_keypoints, kernel_size=1),
            nn.Conv2d(num_keypoints, num_keypoints, kernel_size=3, padding=1, groups=num_keypoints),
        )

    def train(self, mode: bool = True) -> "FrozenDinoHeatmapNet":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor, target_heatmap_size: tuple[int, int]) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            features = self.backbone(images)[-1]
        decoded = self.decoder(features)
        logits = self.keypoint_head(decoded)
        if logits.shape[-2:] != target_heatmap_size:
            logits = F.interpolate(logits, size=target_heatmap_size, mode="bilinear", align_corners=False)
        return logits


class CourtKeypointLightning(L.LightningModule):
    def __init__(
        self,
        num_keypoints: int = len(KEYPOINT_NAMES),
        model_name: str = "convnext_base.dinov3_lvd1689m",
        pretrained: bool = True,
        input_height: int = 384,
        input_width: int = 640,
        heatmap_stride: int = 4,
        sigma: float = 2.0,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = FrozenDinoHeatmapNet(
            num_keypoints=num_keypoints,
            model_name=model_name,
            pretrained=pretrained,
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean) / self.std
        h, w = images.shape[-2:]
        target_size = (max(1, h // self.hparams.heatmap_stride), max(1, w // self.hparams.heatmap_stride))
        return self.net(images, target_size)

    def predict_heatmaps(self, images: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(images))

    def _make_targets(self, keypoints: torch.Tensor, image_size: tuple[int, int]) -> tuple[torch.Tensor, torch.Tensor]:
        image_h, image_w = image_size
        heatmap_h = max(1, image_h // self.hparams.heatmap_stride)
        heatmap_w = max(1, image_w // self.hparams.heatmap_stride)

        xs = torch.arange(heatmap_w, device=keypoints.device, dtype=keypoints.dtype).view(1, 1, 1, heatmap_w)
        ys = torch.arange(heatmap_h, device=keypoints.device, dtype=keypoints.dtype).view(1, 1, heatmap_h, 1)

        kp_x = keypoints[:, :, 0] * (heatmap_w / image_w)
        kp_y = keypoints[:, :, 1] * (heatmap_h / image_h)
        weights = (keypoints[:, :, 2] > 0).to(keypoints.dtype)

        dx2 = (xs - kp_x[:, :, None, None]) ** 2
        dy2 = (ys - kp_y[:, :, None, None]) ** 2
        targets = torch.exp(-0.5 * (dx2 + dy2) / (self.hparams.sigma ** 2))
        targets = targets * weights[:, :, None, None]
        return targets, weights

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        keypoints = batch["keypoints"]
        logits = self(images)
        pred = torch.sigmoid(logits)
        targets, weights = self._make_targets(keypoints, images.shape[-2:])

        pred_flat = pred.flatten(2)
        target_flat = targets.flatten(2)
        cosine = F.cosine_similarity(pred_flat, target_flat, dim=-1, eps=1e-8)
        loss = ((1.0 - cosine) * weights).sum() / weights.sum().clamp_min(1.0)

        batch_size = images.shape[0]
        self.log(
            f"{stage}/loss",
            loss,
            prog_bar=True,
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
        )
        mean_cosine = (cosine * weights).sum() / weights.sum().clamp_min(1.0)
        self.log(
            f"{stage}/mean_cosine",
            mean_cosine,
            prog_bar=stage != "train",
            on_step=stage == "train",
            on_epoch=True,
            batch_size=batch_size,
        )
        mean_error = self._mean_pixel_error(pred, keypoints, weights, images.shape[-2:])
        if mean_error is not None:
            self.log(
                f"{stage}/mean_pixel_error",
                mean_error,
                prog_bar=stage != "train",
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )
            if stage != "train":
                self.log(
                    f"{stage}_mean_pixel_error",
                    mean_error,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                    batch_size=batch_size,
                )
        return loss

    def _mean_pixel_error(
        self,
        heatmaps: torch.Tensor,
        keypoints: torch.Tensor,
        weights: torch.Tensor,
        image_size: tuple[int, int],
    ) -> torch.Tensor | None:
        image_h, image_w = image_size
        heatmap_h, heatmap_w = heatmaps.shape[-2:]
        flat_idx = heatmaps.flatten(2).argmax(dim=-1)
        pred_x = (flat_idx % heatmap_w).to(keypoints.dtype) * (image_w / heatmap_w)
        pred_y = (flat_idx // heatmap_w).to(keypoints.dtype) * (image_h / heatmap_h)
        error = torch.sqrt((pred_x - keypoints[:, :, 0]) ** 2 + (pred_y - keypoints[:, :, 1]) ** 2)
        visible_count = weights.sum()
        if visible_count.item() == 0:
            return None
        return (error * weights).sum() / visible_count

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        trainable = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)


def decode_heatmap_peaks(
    heatmaps: torch.Tensor,
    image_size: tuple[int, int],
    top_k: int = 1,
) -> list[list[tuple[float, float, float]]]:
    if heatmaps.ndim != 3:
        raise ValueError(f"Expected [K, H, W] heatmaps, got {tuple(heatmaps.shape)}")
    k, heatmap_h, heatmap_w = heatmaps.shape
    pooled = F.max_pool2d(heatmaps.unsqueeze(0), kernel_size=3, stride=1, padding=1).squeeze(0)
    local = torch.where(heatmaps == pooled, heatmaps, torch.zeros_like(heatmaps))
    scores, flat_idx = local.flatten(1).topk(min(top_k, heatmap_h * heatmap_w), dim=1)

    image_h, image_w = image_size
    peaks: list[list[tuple[float, float, float]]] = []
    for keypoint_idx in range(k):
        kp_peaks = []
        for score, idx in zip(scores[keypoint_idx], flat_idx[keypoint_idx]):
            x = float((idx % heatmap_w).item() * (image_w / heatmap_w))
            y = float((idx // heatmap_w).item() * (image_h / heatmap_h))
            kp_peaks.append((x, y, float(score.item())))
        peaks.append(kp_peaks)
    return peaks
