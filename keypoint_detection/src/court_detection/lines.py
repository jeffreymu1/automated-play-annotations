"""Line-segmentation model: lineness + per-class heatmaps over basketball court markings."""

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

from court_detection.augmentation import (
    apply_score_bar,
    augment_color,
    crop_resize_image,
    crop_resize_mask,
    random_crop_transform,
    transform_line_uv,
)
from court_detection.dataset import DeepSportDataset
from court_detection.geometry import (
    CameraCalibration,
    LINE_NAMES,
    court_lines_world,
    project_world_to_image,
    sample_segment,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2, 1):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    raise AssertionError("unreachable")


def _conv_block(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        _norm(out_ch),
        nn.GELU(),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        _norm(out_ch),
        nn.GELU(),
    )


def _render_line_targets(
    lines_uv: dict[str, np.ndarray],
    image_size: tuple[int, int],
    output_stride: int,
    sigma: float,
    line_names: tuple[str, ...] = LINE_NAMES,
    visible_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (lineness, class_id, visible) targets at output stride from per-line UV samples.

    `lines_uv[name]` is an (M, 2) float array of pixel coordinates already mapped into the
    network input frame. Points may be outside the frame; they're clipped to the output mask.
    """
    image_h, image_w = image_size
    out_h = max(1, image_h // output_stride)
    out_w = max(1, image_w // output_stride)
    num_classes = len(line_names)
    target_mask = None
    if visible_mask is not None:
        target_mask = cv2.resize(
            visible_mask.astype(np.uint8),
            (out_w, out_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)

    distances = np.full((num_classes, out_h, out_w), np.inf, dtype=np.float32)
    visible = np.zeros(num_classes, dtype=bool)

    for class_idx, name in enumerate(line_names):
        uv = lines_uv.get(name)
        if uv is None or len(uv) == 0:
            continue
        uv_out = uv / output_stride
        finite = np.isfinite(uv_out).all(axis=1)
        if finite.sum() < 2:
            continue
        mask = np.zeros((out_h, out_w), dtype=np.uint8)
        pts = uv_out[finite]
        for a, b in zip(pts[:-1], pts[1:]):
            x0, y0 = int(round(a[0])), int(round(a[1]))
            x1, y1 = int(round(b[0])), int(round(b[1]))
            cv2.line(mask, (x0, y0), (x1, y1), color=1, thickness=1, lineType=cv2.LINE_8)
        if target_mask is not None:
            mask[~target_mask] = 0
        if mask.any():
            visible[class_idx] = True
            dist = cv2.distanceTransform(1 - mask, cv2.DIST_L2, maskSize=3)
            distances[class_idx] = dist

    nearest_class = distances.argmin(axis=0).astype(np.int64)
    min_dist = distances.min(axis=0)
    lineness = np.exp(-0.5 * (min_dist ** 2) / (sigma ** 2)).astype(np.float32)
    lineness[~np.isfinite(min_dist)] = 0.0
    if target_mask is not None:
        lineness[~target_mask] = 0.0

    return lineness, nearest_class, visible


class CourtLineFrameDataset(Dataset):
    """Renders per-frame line ground truth aligned with augmented input frames."""

    def __init__(
        self,
        base: DeepSportDataset,
        indices: list[int],
        image_size: tuple[int, int],
        output_stride: int,
        sigma: float,
        augment: bool = False,
        line_names: tuple[str, ...] = LINE_NAMES,
        n_samples_per_line: int = 400,
    ) -> None:
        self.base = base
        self.indices = indices
        self.image_size = image_size
        self.output_stride = output_stride
        self.sigma = sigma
        self.augment = augment
        self.line_names = line_names
        self.n_samples_per_line = n_samples_per_line

    def __len__(self) -> int:
        if self.augment:
            return 2 * len(self.indices)
        return len(self.indices)

    def _project_lines(self, calib: CameraCalibration) -> dict[str, np.ndarray]:
        worlds = court_lines_world()
        out: dict[str, np.ndarray] = {}
        for name in self.line_names:
            a, b = worlds[name]
            samples = sample_segment(a, b, n=self.n_samples_per_line)
            uv = project_world_to_image(samples, calib)
            out[name] = uv.astype(np.float32)
        return out

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.augment:
            base_idx = self.indices[idx // 2]
            use_score_bar = idx % 2 == 1
        else:
            base_idx = self.indices[idx]
            use_score_bar = False

        image, _, calib = self.base[base_idx]
        image_np = image.permute(1, 2, 0).numpy()
        target_mask = np.ones(image_np.shape[:2], dtype=bool)

        lines_uv = self._project_lines(calib)
        if use_score_bar:
            image_np, target_mask = apply_score_bar(image_np, target_mask)

        crop = random_crop_transform(image_np.shape[:2], self.image_size, augment=self.augment)
        image_np = crop_resize_image(image_np, crop)
        target_mask = crop_resize_mask(target_mask, crop)
        lines_uv = transform_line_uv(lines_uv, crop)
        if self.augment:
            image_np = augment_color(image_np)

        lineness, class_target, visible = _render_line_targets(
            lines_uv,
            image_size=self.image_size,
            output_stride=self.output_stride,
            sigma=self.sigma,
            line_names=self.line_names,
            visible_mask=target_mask,
        )

        image_t = torch.from_numpy(image_np).permute(2, 0, 1).contiguous().float()
        image_path, _ = self.base.samples[base_idx]
        return {
            "image": image_t,
            "lineness": torch.from_numpy(lineness),
            "class_target": torch.from_numpy(class_target),
            "visible": torch.from_numpy(visible),
            "index": base_idx,
            "image_path": str(image_path),
            "score_bar": use_score_bar,
        }


class CourtLineDataModule(L.LightningDataModule):
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
    ) -> None:
        super().__init__()
        self.root = root
        self.image_size = image_size
        self.output_stride = output_stride
        self.sigma = sigma
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

        kwargs = dict(image_size=self.image_size, output_stride=self.output_stride, sigma=self.sigma)
        self.train_dataset = CourtLineFrameDataset(base, train_indices, augment=True, **kwargs)
        self.val_dataset = CourtLineFrameDataset(base, val_indices, augment=False, **kwargs)
        self.test_dataset = CourtLineFrameDataset(base, test_indices, augment=False, **kwargs)

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


class FrozenDinoUNet(nn.Module):
    """Frozen DINOv3 ConvNeXt backbone with a U-Net decoder feeding lineness + class heads."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "convnext_base.dinov3_lvd1689m",
        pretrained: bool = True,
        decoder_channels: int = 128,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        self.backbone.eval()

        ch = self.backbone.feature_info.channels()
        d = decoder_channels

        self.proj = nn.ModuleList([nn.Conv2d(c, d, kernel_size=1) for c in ch])
        self.dec_c4 = _conv_block(2 * d, d)
        self.dec_c3 = _conv_block(2 * d, d)
        self.dec_c2 = _conv_block(2 * d, d)
        self.up_to_s2 = _conv_block(d, d)

        self.line_head = nn.Conv2d(d, 1, kernel_size=1)
        self.class_head = nn.Conv2d(d, num_classes, kernel_size=1)

    def train(self, mode: bool = True) -> "FrozenDinoUNet":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.backbone.eval()
        with torch.no_grad():
            c2, c3, c4, c5 = self.backbone(images)
        p2 = self.proj[0](c2)
        p3 = self.proj[1](c3)
        p4 = self.proj[2](c4)
        p5 = self.proj[3](c5)

        u4 = F.interpolate(p5, size=p4.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.dec_c4(torch.cat([p4, u4], dim=1))

        u3 = F.interpolate(d4, size=p3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec_c3(torch.cat([p3, u3], dim=1))

        u2 = F.interpolate(d3, size=p2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec_c2(torch.cat([p2, u2], dim=1))

        h, w = images.shape[-2] // 2, images.shape[-1] // 2
        s2 = self.up_to_s2(F.interpolate(d2, size=(h, w), mode="bilinear", align_corners=False))

        line_logits = self.line_head(s2).squeeze(1)
        class_logits = self.class_head(s2)
        return line_logits, class_logits


def soft_dice_loss(prob: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    numerator = 2.0 * (prob * target).flatten(1).sum(dim=1) + eps
    denominator = (prob + target).flatten(1).sum(dim=1) + eps
    return (1.0 - numerator / denominator).mean()


def focal_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p = torch.sigmoid(logits)
    p_t = p * target + (1.0 - p) * (1.0 - target)
    a_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (a_t * (1.0 - p_t).pow(gamma) * bce).mean()


def gated_cross_entropy(
    class_logits: torch.Tensor,
    class_target: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    ce = F.cross_entropy(class_logits, class_target, reduction="none")
    return (weight * ce).sum() / (weight.sum() + eps)


class CourtLineLightning(L.LightningModule):
    def __init__(
        self,
        num_classes: int = len(LINE_NAMES),
        model_name: str = "convnext_base.dinov3_lvd1689m",
        pretrained: bool = True,
        decoder_channels: int = 128,
        output_stride: int = 2,
        sigma: float = 1.5,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
        lambda_class: float = 1.0,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        lineness_threshold: float = 0.5,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 200,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.net = FrozenDinoUNet(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            decoder_channels=decoder_channels,
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        normalized = (images - self.mean) / self.std
        return self.net(normalized)

    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        line_logits, class_logits = self(images)
        return torch.sigmoid(line_logits), torch.softmax(class_logits, dim=1)

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        target_line = batch["lineness"]
        target_class = batch["class_target"]

        line_logits, class_logits = self(images)
        if line_logits.shape[-2:] != target_line.shape[-2:]:
            line_logits = F.interpolate(
                line_logits.unsqueeze(1), size=target_line.shape[-2:], mode="bilinear", align_corners=False
            ).squeeze(1)
            class_logits = F.interpolate(
                class_logits, size=target_line.shape[-2:], mode="bilinear", align_corners=False
            )

        line_prob = torch.sigmoid(line_logits)
        dice = soft_dice_loss(line_prob, target_line)
        focal = focal_bce_loss(line_logits, target_line, self.hparams.focal_alpha, self.hparams.focal_gamma)
        line_loss = self.hparams.lambda_dice * dice + self.hparams.lambda_focal * focal

        class_loss = gated_cross_entropy(class_logits, target_class, target_line)
        loss = line_loss + self.hparams.lambda_class * class_loss

        with torch.no_grad():
            mask_pred = (line_prob > self.hparams.lineness_threshold).float()
            mask_gt = (target_line > 0.5).float()
            inter = (mask_pred * mask_gt).flatten(1).sum(dim=1)
            union = (mask_pred + mask_gt - mask_pred * mask_gt).flatten(1).sum(dim=1).clamp_min(1.0)
            line_iou = (inter / union).mean()

            class_pred = class_logits.argmax(dim=1)
            correct = (class_pred == target_class).float()
            class_acc_num = (correct * target_line).sum()
            class_acc_den = target_line.sum().clamp_min(1.0)
            class_acc = class_acc_num / class_acc_den

        batch_size = images.shape[0]
        log_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_line", line_loss, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_class", class_loss, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/dice", dice, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/line_iou", line_iou, prog_bar=stage != "train", on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/class_acc", class_acc, prog_bar=stage != "train", on_step=False, on_epoch=True, batch_size=batch_size)
        if stage != "train":
            self.log(f"{stage}_line_iou", line_iou, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss

    def training_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def test_step(self, batch: dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "test")

    def configure_optimizers(self) -> dict[str, Any]:
        trainable = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        warmup = self.hparams.warmup_steps

        def lr_lambda(step: int) -> float:
            if warmup <= 0:
                return 1.0
            return min(1.0, (step + 1) / warmup)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def class_palette(num_classes: int) -> np.ndarray:
    """Distinct RGB colors per class in [0, 1]."""
    base = np.array([
        [0.95, 0.30, 0.30],
        [0.30, 0.85, 0.40],
        [0.35, 0.55, 0.95],
        [0.95, 0.85, 0.30],
        [0.85, 0.40, 0.95],
        [0.30, 0.85, 0.85],
        [0.95, 0.55, 0.30],
        [0.55, 0.30, 0.85],
    ], dtype=np.float32)
    if num_classes <= len(base):
        return base[:num_classes]
    extra = np.random.default_rng(1430).random((num_classes - len(base), 3)).astype(np.float32)
    return np.concatenate([base, extra], axis=0)


def overlay_line_predictions(
    image_rgb: np.ndarray,
    class_probs: np.ndarray,
    lineness: np.ndarray,
    palette: np.ndarray,
    floor: float = 0.35,
    color_strength: float = 0.6,
) -> np.ndarray:
    """Render class colors over the image, then modulate brightness by lineness with a floor.

    image_rgb: (H, W, 3) float in [0, 1]
    class_probs: (K, H, W) softmax probs
    lineness: (H, W) probability in [0, 1]
    palette: (K, 3) RGB colors in [0, 1]
    """
    color_map = np.einsum("khw,kc->hwc", class_probs, palette)
    color_map = np.clip(color_map, 0.0, 1.0)
    blended = (1.0 - color_strength) * image_rgb + color_strength * color_map
    brightness = floor + (1.0 - floor) * lineness
    out = blended * brightness[..., None]
    return np.clip(out, 0.0, 1.0)
