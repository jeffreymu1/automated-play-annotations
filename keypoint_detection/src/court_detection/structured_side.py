"""Experimental marking model with a constrained linear court-side boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterator

import timm
import torch
from torch import nn
from torch.nn import functional as F

from court_detection.geometry import MARKING_CLASS_NAMES
from court_detection.lines import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    CourtLineLightning,
    _conv_block,
    focal_bce_loss,
    gated_cross_entropy,
    soft_dice_loss,
)


@dataclass(frozen=True)
class StructuredSideOutput:
    """Tuple-compatible network output with auxiliary side logits attached."""

    line_logits: torch.Tensor
    class_logits: torch.Tensor
    side_logits: torch.Tensor
    court_logits: torch.Tensor
    aux_side_logits: tuple[torch.Tensor, ...]
    side_params: torch.Tensor

    def __iter__(self) -> Iterator[torch.Tensor]:
        return iter((self.line_logits, self.class_logits, self.side_logits, self.court_logits))

    def __getitem__(self, index: int) -> torch.Tensor:
        return (self.line_logits, self.class_logits, self.side_logits, self.court_logits)[index]

    def __len__(self) -> int:
        return 4


class StructuredSideDinoUNet(nn.Module):
    """Frozen DINOv3 U-Net whose side head can only render one monotone line."""

    def __init__(
        self,
        num_classes: int,
        model_name: str = "convnext_base.dinov3_lvd1689m",
        pretrained: bool = True,
        decoder_channels: int = 128,
        layout_channels: int | None = None,
        max_slope: float = 2.0,
        residual_dx: tuple[float, float, float] = (48.0, 24.0, 12.0),
        residual_dm: tuple[float, float, float] = (0.45, 0.25, 0.12),
        min_side_sharpness: float = 0.25,
        max_side_sharpness: float = 8.0,
    ) -> None:
        super().__init__()
        if len(residual_dx) != 3 or len(residual_dm) != 3:
            raise ValueError("residual_dx and residual_dm must each have three values")

        self.max_slope = float(max_slope)
        self.residual_dx = tuple(float(v) for v in residual_dx)
        self.residual_dm = tuple(float(v) for v in residual_dm)
        self.min_side_sharpness = float(min_side_sharpness)
        self.max_side_sharpness = float(max_side_sharpness)

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
        layout_d = layout_channels if layout_channels is not None else max(32, decoder_channels // 2)

        self.proj = nn.ModuleList([nn.Conv2d(c, d, kernel_size=1) for c in ch])
        self.dec_c4 = _conv_block(2 * d, d)
        self.dec_c3 = _conv_block(2 * d, d)
        self.dec_c2 = _conv_block(2 * d, d)
        self.up_to_s2 = _conv_block(d, d)

        self.layout_proj = nn.ModuleList([nn.Conv2d(c, layout_d, kernel_size=1) for c in ch])
        self.layout_dec_c4 = _conv_block(2 * layout_d + 1, layout_d)
        self.layout_dec_c3 = _conv_block(2 * layout_d + 1, layout_d)
        self.layout_dec_c2 = _conv_block(2 * layout_d + 1, layout_d)

        self.coarse_head = nn.Linear(ch[3], 3)
        self.refine_heads = nn.ModuleList(
            [
                nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(layout_d, 2)),
                nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(layout_d, 2)),
                nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(layout_d, 2)),
            ]
        )

        self.line_head = nn.Conv2d(d, 1, kernel_size=1)
        self.class_head = nn.Conv2d(d, num_classes, kernel_size=1)
        self.court_head = nn.Conv2d(layout_d, 1, kernel_size=1)

    def train(self, mode: bool = True) -> "StructuredSideDinoUNet":
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, images: torch.Tensor) -> StructuredSideOutput:
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

        out_h, out_w = images.shape[-2] // 2, images.shape[-1] // 2
        s2 = self.up_to_s2(F.interpolate(d2, size=(out_h, out_w), mode="bilinear", align_corners=False))

        line_logits = self.line_head(s2).squeeze(1)
        class_logits = self.class_head(s2)

        raw = self.coarse_head(c5.mean(dim=(-2, -1)))
        x0 = (out_w - 1) * torch.sigmoid(raw[:, 0])
        slope = self.max_slope * torch.tanh(raw[:, 1])
        sharpness = self.min_side_sharpness + self.max_side_sharpness * torch.sigmoid(raw[:, 2])

        aux_logits: list[torch.Tensor] = [self._render_side_logits(x0, slope, sharpness, (out_h, out_w))]

        l2, l3, l4, l5 = [proj(feat) for proj, feat in zip(self.layout_proj, (c2, c3, c4, c5), strict=True)]
        layout = l5
        for level, (skip, block, head) in enumerate(
            (
                (l4, self.layout_dec_c4, self.refine_heads[0]),
                (l3, self.layout_dec_c3, self.refine_heads[1]),
                (l2, self.layout_dec_c2, self.refine_heads[2]),
            )
        ):
            layout = F.interpolate(layout, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            distance = self._normalized_signed_distance(x0, slope, skip.shape[-2:], (out_h, out_w))
            layout = block(torch.cat([skip, layout, distance], dim=1))
            delta = head(layout)
            x0 = (x0 + self.residual_dx[level] * torch.tanh(delta[:, 0])).clamp(0.0, float(out_w - 1))
            slope = (slope + self.residual_dm[level] * torch.tanh(delta[:, 1])).clamp(
                -self.max_slope,
                self.max_slope,
            )
            aux_logits.append(self._render_side_logits(x0, slope, sharpness, (out_h, out_w)))

        court_logits = self.court_head(layout).squeeze(1)
        side_logits = aux_logits[-1]
        side_params = torch.stack([x0, slope, sharpness], dim=1)
        return StructuredSideOutput(
            line_logits,
            class_logits,
            side_logits,
            court_logits,
            tuple(aux_logits[:-1]),
            side_params,
        )

    def _render_side_logits(
        self,
        x0: torch.Tensor,
        slope: torch.Tensor,
        sharpness: torch.Tensor,
        size: tuple[int, int],
    ) -> torch.Tensor:
        h, w = size
        y, x = self._grid(size, x0.device, x0.dtype)
        signed = x - x0[:, None, None] - slope[:, None, None] * (y - 0.5 * (h - 1))
        return sharpness[:, None, None] * signed

    def _normalized_signed_distance(
        self,
        x0: torch.Tensor,
        slope: torch.Tensor,
        feature_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        out_h, out_w = output_size
        feat_h, feat_w = feature_size
        y, x = self._grid(feature_size, x0.device, x0.dtype)
        x_scale = (out_w - 1) / max(1, feat_w - 1)
        y_scale = (out_h - 1) / max(1, feat_h - 1)
        x_out = x * x_scale
        y_out = y * y_scale
        signed = x_out - x0[:, None, None] - slope[:, None, None] * (y_out - 0.5 * (out_h - 1))
        denom = torch.tensor(max(1.0, float(out_w - 1)), device=x0.device, dtype=x0.dtype)
        return (signed / denom).unsqueeze(1)

    @staticmethod
    def _grid(size: tuple[int, int], device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        h, w = size
        y = torch.arange(h, device=device, dtype=dtype).view(1, h, 1)
        x = torch.arange(w, device=device, dtype=dtype).view(1, 1, w)
        return y, x


class StructuredSideCourtLineLightning(CourtLineLightning):
    """Lightning module for the constrained-line side architecture."""

    def __init__(
        self,
        num_classes: int = len(MARKING_CLASS_NAMES),
        line_names: tuple[str, ...] | None = None,
        model_name: str = "convnext_base.dinov3_lvd1689m",
        pretrained: bool = True,
        decoder_channels: int = 128,
        layout_channels: int | None = None,
        output_stride: int = 2,
        sigma: float = 1.5,
        lambda_dice: float = 1.0,
        lambda_focal: float = 1.0,
        lambda_class: float = 1.0,
        lambda_side: float = 0.25,
        lambda_side_aux: float = 0.2,
        lambda_court: float = 0.5,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        lineness_threshold: float = 0.5,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
        warmup_steps: int = 200,
        max_side_slope: float = 2.0,
        side_residual_dx: tuple[float, float, float] = (48.0, 24.0, 12.0),
        side_residual_dm: tuple[float, float, float] = (0.45, 0.25, 0.12),
        min_side_sharpness: float = 0.25,
        max_side_sharpness: float = 8.0,
    ) -> None:
        super().__init__(
            num_classes=num_classes,
            line_names=line_names,
            model_name=model_name,
            pretrained=False,
            decoder_channels=decoder_channels,
            layout_channels=layout_channels,
            output_stride=output_stride,
            sigma=sigma,
            lambda_dice=lambda_dice,
            lambda_focal=lambda_focal,
            lambda_class=lambda_class,
            lambda_side=lambda_side,
            lambda_court=lambda_court,
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            lineness_threshold=lineness_threshold,
            lr=lr,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
        )
        self.save_hyperparameters()
        self.net = StructuredSideDinoUNet(
            num_classes=num_classes,
            model_name=model_name,
            pretrained=pretrained,
            decoder_channels=decoder_channels,
            layout_channels=layout_channels,
            max_slope=max_side_slope,
            residual_dx=side_residual_dx,
            residual_dm=side_residual_dm,
            min_side_sharpness=min_side_sharpness,
            max_side_sharpness=max_side_sharpness,
        )
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1), persistent=False)

    def _shared_step(self, batch: dict[str, Any], stage: str) -> torch.Tensor:
        images = batch["image"]
        target_line = batch["lineness"]
        target_class = batch["class_target"]
        target_side = batch["side_target"]
        target_side_weight = batch.get("side_weight")
        target_court = batch.get("court_mask", target_side_weight)

        output = self(images)
        line_logits, class_logits, side_logits, court_logits = output
        aux_side_logits = getattr(output, "aux_side_logits", ())
        target_size = target_line.shape[-2:]
        line_logits = _resize_logits(line_logits, target_size)
        class_logits = _resize_class_logits(class_logits, target_size)
        side_logits = _resize_logits(side_logits, target_size)
        court_logits = _resize_logits(court_logits, target_size)
        aux_side_logits = tuple(_resize_logits(logits, target_size) for logits in aux_side_logits)
        if target_court is None:
            target_court = torch.ones_like(target_line)

        line_prob = torch.sigmoid(line_logits)
        dice = soft_dice_loss(line_prob, target_line)
        focal = focal_bce_loss(line_logits, target_line, self.hparams.focal_alpha, self.hparams.focal_gamma)
        line_loss = self.hparams.lambda_dice * dice + self.hparams.lambda_focal * focal

        class_loss = gated_cross_entropy(class_logits, target_class, target_line)
        court_loss = F.binary_cross_entropy_with_logits(court_logits, target_court)
        if target_side_weight is None:
            target_side_weight = torch.ones_like(target_side)
        side_loss = _weighted_side_bce(side_logits, target_side, target_side_weight)
        aux_side_loss = torch.zeros((), device=images.device, dtype=side_loss.dtype)
        if aux_side_logits:
            aux_side_loss = torch.stack(
                [_weighted_side_bce(logits, target_side, target_side_weight) for logits in aux_side_logits]
            ).mean()
        loss = (
            line_loss
            + self.hparams.lambda_class * class_loss
            + self.hparams.lambda_side * (side_loss + self.hparams.lambda_side_aux * aux_side_loss)
            + self.hparams.lambda_court * court_loss
        )

        with torch.no_grad():
            mask_pred = (line_prob > self.hparams.lineness_threshold).float()
            mask_gt = (target_line > 0.5).float()
            inter = (mask_pred * mask_gt).flatten(1).sum(dim=1)
            union = (mask_pred + mask_gt - mask_pred * mask_gt).flatten(1).sum(dim=1).clamp_min(1.0)
            line_iou = (inter / union).mean()

            class_pred = class_logits.argmax(dim=1)
            correct = (class_pred == target_class).float()
            class_acc = (correct * target_line).sum() / target_line.sum().clamp_min(1.0)
            side_pred = torch.sigmoid(side_logits) > 0.5
            side_correct = (side_pred == (target_side > 0.5)).float()
            side_acc = (side_correct * target_side_weight).sum() / target_side_weight.sum().clamp_min(1.0)
            court_prob = torch.sigmoid(court_logits)
            court_pred = (court_prob > 0.5).float()
            court_gt = (target_court > 0.5).float()
            court_inter = (court_pred * court_gt).flatten(1).sum(dim=1)
            court_union = (court_pred + court_gt - court_pred * court_gt).flatten(1).sum(dim=1).clamp_min(1.0)
            court_iou = (court_inter / court_union).mean()

        batch_size = images.shape[0]
        log_step = stage == "train"
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_line", line_loss, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_class", class_loss, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_side", side_loss, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_side_aux", aux_side_loss, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/loss_court", court_loss, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/dice", dice, on_step=log_step, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/line_iou", line_iou, prog_bar=stage != "train", on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/class_acc", class_acc, prog_bar=stage != "train", on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/side_acc", side_acc, prog_bar=stage != "train", on_step=False, on_epoch=True, batch_size=batch_size)
        self.log(f"{stage}/court_iou", court_iou, prog_bar=stage != "train", on_step=False, on_epoch=True, batch_size=batch_size)
        side_params = getattr(output, "side_params", None)
        if side_params is not None:
            self.log(f"{stage}/side_x0", side_params[:, 0].mean(), on_step=log_step, on_epoch=True, batch_size=batch_size)
            self.log(f"{stage}/side_slope", side_params[:, 1].mean(), on_step=log_step, on_epoch=True, batch_size=batch_size)
            self.log(f"{stage}/side_sharpness", side_params[:, 2].mean(), on_step=log_step, on_epoch=True, batch_size=batch_size)
        if stage != "train":
            self.log(f"{stage}_line_iou", line_iou, on_step=False, on_epoch=True, batch_size=batch_size)
        return loss


def _weighted_side_bce(logits: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    raw = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    return (raw * weight).sum() / weight.sum().clamp_min(1.0)


def _resize_logits(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == target_size:
        return logits
    return F.interpolate(logits.unsqueeze(1), size=target_size, mode="bilinear", align_corners=False).squeeze(1)


def _resize_class_logits(logits: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    if logits.shape[-2:] == target_size:
        return logits
    return F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)


__all__ = [
    "StructuredSideCourtLineLightning",
    "StructuredSideDinoUNet",
]
