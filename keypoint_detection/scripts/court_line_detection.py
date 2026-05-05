"""Train, evaluate, and visualize the expanded FIBA court-marking detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F

from court_detection.markings import (
    FIBA_MARKING_NAMES as LINE_NAMES,
    FibaCourtMarkingDataModule as CourtLineDataModule,
    FibaCourtMarkingLightning as CourtLineLightning,
    class_palette,
    overlay_line_predictions,
)


def _make_datamodule(args: argparse.Namespace, image_size: tuple[int, int] | None = None) -> CourtLineDataModule:
    if image_size is None:
        image_size = (args.image_height, args.image_width)
    return CourtLineDataModule(
        root=args.root,
        image_size=image_size,
        output_stride=args.output_stride,
        sigma=args.sigma,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        side_blur_sigma=args.side_blur_sigma,
    )


def train(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    dm = _make_datamodule(args)
    model = CourtLineLightning(
        model_name=args.model_name,
        pretrained=args.pretrained,
        decoder_channels=args.decoder_channels,
        layout_channels=args.layout_channels,
        output_stride=args.output_stride,
        sigma=args.sigma,
        lambda_dice=args.lambda_dice,
        lambda_focal=args.lambda_focal,
        lambda_class=args.lambda_class,
        lambda_side=args.lambda_side,
        lambda_court=args.lambda_court,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.out,
        filename="fiba-court-markings-{epoch:03d}-{val_line_iou:.3f}",
        monitor="val_line_iou",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
    )
    callbacks = [checkpoint_cb]
    if args.visualize_best:
        callbacks.append(
            BestCheckpointVisualizationCallback(
                root=args.root,
                out_root=args.best_vis_out,
                footage_root=args.test_footage,
                image_size=(args.image_height, args.image_width),
                min_epoch=args.visualize_min_epoch,
                dataset_count=args.visualize_dataset_count,
                footage_stride=args.visualize_footage_stride,
                num_workers=args.visualize_num_workers,
                seed=args.seed,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                side_blur_sigma=args.side_blur_sigma,
            )
        )
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.log_name)
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=1,
        fast_dev_run=args.fast_dev_run,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    trainer.fit(model, datamodule=dm)
    if checkpoint_cb.best_model_path:
        print(f"best checkpoint: {checkpoint_cb.best_model_path}")
    if checkpoint_cb.last_model_path:
        print(f"last checkpoint: {checkpoint_cb.last_model_path}")


class BestCheckpointVisualizationCallback(Callback):
    def __init__(
        self,
        root: Path,
        out_root: Path,
        footage_root: Path,
        image_size: tuple[int, int],
        min_epoch: int,
        dataset_count: int,
        footage_stride: int,
        num_workers: int,
        seed: int,
        val_fraction: float,
        test_fraction: float,
        side_blur_sigma: float,
    ) -> None:
        self.root = root
        self.out_root = out_root
        self.footage_root = footage_root
        self.image_size = image_size
        self.min_epoch = min_epoch
        self.dataset_count = dataset_count
        self.footage_stride = footage_stride
        self.num_workers = num_workers
        self.seed = seed
        self.val_fraction = val_fraction
        self.test_fraction = test_fraction
        self.side_blur_sigma = side_blur_sigma
        self.best_score = -float("inf")

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: CourtLineLightning) -> None:
        if trainer.sanity_checking:
            return
        score_t = trainer.callback_metrics.get("val_line_iou")
        if score_t is None:
            return
        score = float(score_t.detach().cpu()) if isinstance(score_t, torch.Tensor) else float(score_t)
        epoch_num = int(trainer.current_epoch) + 1
        if score <= self.best_score:
            return
        self.best_score = score
        if epoch_num < self.min_epoch:
            return

        out_dir = self.out_root / f"epoch_{epoch_num:03d}_val_line_iou_{score:.3f}"
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = out_dir / "checkpoint.ckpt"
        trainer.save_checkpoint(checkpoint_path)
        print(f"new best after epoch {epoch_num}: {score:.3f}; visualizing {checkpoint_path}")
        self._visualize_checkpoint(checkpoint_path, out_dir, trainer, pl_module, epoch_num)

    def _visualize_checkpoint(
        self,
        checkpoint_path: Path,
        out_dir: Path,
        trainer: L.Trainer,
        pl_module: CourtLineLightning,
        epoch_num: int,
    ) -> None:
        device = pl_module.device
        model = pl_module
        was_training = model.training
        model.eval()
        num_classes = int(model.hparams.num_classes)
        line_names = tuple(getattr(model, "line_names", LINE_NAMES))[:num_classes]
        palette = class_palette(num_classes)
        figure_writer = _figure_writer(trainer.logger)
        global_step = int(trainer.global_step)
        tag_prefix = f"best_checkpoint/epoch_{epoch_num:03d}"

        try:
            self._visualize_dataset(
                model,
                device,
                palette,
                line_names,
                out_dir / "dataset_test",
                figure_writer,
                global_step,
                f"{tag_prefix}/dataset_test",
            )
            self._visualize_footage(
                model,
                device,
                palette,
                line_names,
                out_dir / "test_footage",
                figure_writer,
                global_step,
                f"{tag_prefix}/test_footage",
            )
        finally:
            if was_training:
                model.train()

    def _visualize_dataset(
        self,
        model: CourtLineLightning,
        device: torch.device,
        palette: np.ndarray,
        line_names: tuple[str, ...],
        out_dir: Path,
        figure_writer: object | None,
        global_step: int,
        tag_prefix: str,
    ) -> None:
        dm = CourtLineDataModule(
            root=self.root,
            image_size=self.image_size,
            output_stride=int(model.hparams.output_stride),
            sigma=float(model.hparams.sigma),
            batch_size=1,
            num_workers=self.num_workers,
            seed=self.seed,
            val_fraction=self.val_fraction,
            test_fraction=self.test_fraction,
            side_blur_sigma=self.side_blur_sigma,
        )
        dm.setup("test")
        out_dir.mkdir(parents=True, exist_ok=True)
        count = min(self.dataset_count, len(dm.test_dataset))
        for i in range(count):
            sample = dm.test_dataset[i]
            image_rgb = sample["image"].permute(1, 2, 0).numpy()
            line_prob, class_probs, side_prob, court_prob = _predict_image_full(model, image_rgb, self.image_size, device)
            pred_overlay = overlay_line_predictions(image_rgb, class_probs, line_prob, palette)

            gt_line = _resize(sample["lineness"].numpy(), image_rgb.shape[:2], mode="bilinear")
            gt_class = _resize(sample["class_target"].numpy().astype(np.float32), image_rgb.shape[:2], mode="nearest")
            gt_side = _resize(sample["side_target"].numpy(), image_rgb.shape[:2], mode="bilinear")
            gt_side_weight = _resize(sample["side_weight"].numpy(), image_rgb.shape[:2], mode="nearest")
            gt_court = _resize(sample["court_mask"].numpy(), image_rgb.shape[:2], mode="nearest")
            gt_onehot = np.zeros((len(line_names), *image_rgb.shape[:2]), dtype=np.float32)
            for k in range(len(line_names)):
                gt_onehot[k] = (gt_class.astype(np.int64) == k).astype(np.float32)
            gt_overlay = overlay_line_predictions(image_rgb, gt_onehot, gt_line, palette)

            title = f"test sample {i} | base index {sample['index']} | {Path(sample['image_path']).name}"
            _save_prediction_panel(
                out_dir / f"dataset_test_{i:03d}.png",
                title,
                image_rgb,
                pred_overlay,
                line_prob,
                side_prob,
                court_prob,
                palette,
                line_names,
                gt_overlay=gt_overlay,
                gt_side=gt_side,
                gt_side_weight=gt_side_weight,
                gt_court=gt_court,
                figure_writer=figure_writer,
                tb_tag=f"{tag_prefix}/{i:03d}_{_safe_tag(Path(sample['image_path']).stem)}",
                global_step=global_step,
            )

    def _visualize_footage(
        self,
        model: CourtLineLightning,
        device: torch.device,
        palette: np.ndarray,
        line_names: tuple[str, ...],
        out_dir: Path,
        figure_writer: object | None,
        global_step: int,
        tag_prefix: str,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for clip_dir in sorted(path for path in self.footage_root.iterdir() if path.is_dir()):
            frames_dir = clip_dir / "frames"
            if not frames_dir.is_dir():
                continue
            clip_out = out_dir / clip_dir.name
            clip_out.mkdir(parents=True, exist_ok=True)
            frame_paths = sorted(
                path for path in frames_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
            )[:: self.footage_stride]
            for path in frame_paths:
                image_rgb = _read_rgb(path)
                line_prob, class_probs, side_prob, court_prob = _predict_image_full(model, image_rgb, self.image_size, device)
                pred_overlay = overlay_line_predictions(image_rgb, class_probs, line_prob, palette)
                _save_prediction_panel(
                    clip_out / f"{path.stem}_markings_panel.png",
                    f"{clip_dir.name} | {path.name}",
                    image_rgb,
                    pred_overlay,
                    line_prob,
                    side_prob,
                    court_prob,
                    palette,
                    line_names,
                    figure_writer=figure_writer,
                    tb_tag=f"{tag_prefix}/{_safe_tag(clip_dir.name)}/{_safe_tag(path.stem)}",
                    global_step=global_step,
                )


def _read_rgb(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _predict_image_full(
    model: CourtLineLightning,
    image_rgb: np.ndarray,
    image_size: tuple[int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    model_image = cv2.resize(image_rgb, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    image_t = torch.from_numpy(model_image).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        line_prob, class_probs, side_prob, court_prob = model.predict(image_t)
    output_size = image_rgb.shape[:2]
    line_full = F.interpolate(
        line_prob.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    class_full = F.interpolate(
        class_probs, size=output_size, mode="bilinear", align_corners=False
    )[0].cpu().numpy()
    side_full = F.interpolate(
        side_prob.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    court_full = F.interpolate(
        court_prob.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    return line_full, class_full, side_full, court_full


def _save_prediction_panel(
    out_path: Path,
    title: str,
    image_rgb: np.ndarray,
    pred_overlay: np.ndarray,
    line_prob: np.ndarray,
    side_prob: np.ndarray,
    court_prob: np.ndarray,
    palette: np.ndarray,
    line_names: tuple[str, ...],
    gt_overlay: np.ndarray | None = None,
    gt_side: np.ndarray | None = None,
    gt_side_weight: np.ndarray | None = None,
    gt_court: np.ndarray | None = None,
    figure_writer: object | None = None,
    tb_tag: str | None = None,
    global_step: int | None = None,
) -> None:
    gt_side_image = _side_rgb(gt_side, gt_side_weight) if gt_side is not None else None
    pred_side_weight = (court_prob >= 0.5).astype(np.float32)
    pred_side_image = _side_rgb(side_prob, pred_side_weight)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    panels = [
        (axes[0, 0], image_rgb, "Input frame", None),
        (axes[0, 1], gt_overlay if gt_overlay is not None else pred_overlay, "GT overlay" if gt_overlay is not None else "Prediction overlay", None),
        (axes[0, 2], gt_side_image if gt_side_image is not None else court_prob, "GT court side" if gt_side is not None else "Predicted court mask", None if gt_side is not None else "gray"),
        (axes[1, 0], pred_overlay, "Prediction overlay", None),
        (axes[1, 1], line_prob, "Predicted lineness", "inferno"),
        (axes[1, 2], pred_side_image, "Predicted court side (masked)", None),
    ]
    for ax, image, panel_title, cmap in panels:
        if image.ndim == 2:
            ax.imshow(image, cmap=cmap or "inferno", vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.clip(image, 0.0, 1.0))
        ax.set_title(panel_title)
        ax.axis("off")
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=palette[k], markersize=9, label=name)
        for k, name in enumerate(line_names)
    ]
    fig.suptitle(title, fontsize=11)
    fig.legend(handles=handles, loc="lower center", ncol=min(5, len(handles)), frameon=False, fontsize=8)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 0.95))
    fig.savefig(out_path, dpi=140)
    if figure_writer is not None and tb_tag is not None:
        figure_writer.add_figure(tb_tag, fig, global_step=global_step)
    plt.close(fig)


def _figure_writer(logger: object | None) -> object | None:
    experiment = getattr(logger, "experiment", None)
    if hasattr(experiment, "add_figure"):
        return experiment
    return None


def _safe_tag(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def _side_rgb(side: np.ndarray, weight: np.ndarray | None = None) -> np.ndarray:
    rgb = plt.get_cmap("coolwarm")(np.clip(side, 0.0, 1.0))[..., :3].astype(np.float32)
    if weight is not None:
        rgb[weight <= 0.0] = 0.15
    return rgb


def evaluate(args: argparse.Namespace) -> None:
    model = CourtLineLightning.load_from_checkpoint(args.checkpoint, map_location="cpu", pretrained=False)
    image_size = (args.image_height, args.image_width)
    dm = _make_datamodule(args, image_size=image_size)
    trainer = L.Trainer(accelerator="auto", devices="auto", precision=args.precision)
    trainer.test(model, datamodule=dm)


def _load_plain_image(path: Path, image_size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    full_image = image_rgb.astype(np.float32) / 255.0
    model_image = cv2.resize(full_image, (image_size[1], image_size[0]), interpolation=cv2.INTER_AREA)
    return full_image, model_image


def _iter_image_folder(folder: Path, image_glob: str) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"Expected an image folder: {folder}")
    image_paths = sorted(path for path in folder.glob(image_glob) if path.is_file())
    if not image_paths:
        raise RuntimeError(f"No images matched {image_glob!r} in {folder}")
    return image_paths


def _predict_full_resolution(
    model: CourtLineLightning,
    model_image: np.ndarray,
    output_size: tuple[int, int],
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    image_t = torch.from_numpy(model_image).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        line_prob, class_probs, side_prob, court_prob = model.predict(image_t)
    line_prob_full = F.interpolate(
        line_prob.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    class_probs_full = F.interpolate(
        class_probs, size=output_size, mode="bilinear", align_corners=False
    )[0].cpu().numpy()
    side_prob_full = F.interpolate(
        side_prob.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    court_prob_full = F.interpolate(
        court_prob.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    return line_prob_full, class_probs_full, side_prob_full, court_prob_full


def _save_rgb(path: Path, image: np.ndarray) -> None:
    out = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def _visualize(args: argparse.Namespace) -> None:
    model = CourtLineLightning.load_from_checkpoint(args.checkpoint, map_location="cpu", pretrained=False)
    num_classes = int(model.hparams.num_classes)
    output_stride = int(model.hparams.output_stride)
    sigma = float(model.hparams.sigma)
    side_blur_sigma = float(getattr(model.hparams, "side_blur_sigma", args.side_blur_sigma))
    image_size = (args.image_height, args.image_width)

    dm = CourtLineDataModule(
        root=args.root,
        image_size=image_size,
        output_stride=output_stride,
        sigma=sigma,
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
        side_blur_sigma=side_blur_sigma,
    )
    dm.setup("test")
    dataset = dm.val_dataset if args.split == "val" else dm.test_dataset

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()
    args.out.mkdir(parents=True, exist_ok=True)

    palette = class_palette(num_classes)
    count = min(args.count, len(dataset))
    for i in range(count):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            line_prob, class_probs, side_prob, court_prob = model.predict(image)
        line_prob_full = F.interpolate(
            line_prob.unsqueeze(1), size=image_size, mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        class_probs_full = F.interpolate(
            class_probs, size=image_size, mode="bilinear", align_corners=False
        )[0].cpu().numpy()
        side_prob_full = F.interpolate(
            side_prob.unsqueeze(1), size=image_size, mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        court_prob_full = F.interpolate(
            court_prob.unsqueeze(1), size=image_size, mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()

        image_np = sample["image"].permute(1, 2, 0).numpy()
        gt_line = sample["lineness"].numpy()
        gt_class = sample["class_target"].numpy()
        gt_side = sample["side_target"].numpy()
        gt_court = sample["court_mask"].numpy()

        gt_line_full = _resize(gt_line, image_size, mode="bilinear")
        gt_class_full = _resize(gt_class.astype(np.float32), image_size, mode="nearest").astype(np.int64)
        gt_side_full = _resize(gt_side, image_size, mode="bilinear")
        gt_court_full = _resize(gt_court, image_size, mode="nearest")
        gt_class_onehot = np.zeros((num_classes, *image_size), dtype=np.float32)
        for k in range(num_classes):
            gt_class_onehot[k] = (gt_class_full == k).astype(np.float32)

        pred_overlay = overlay_line_predictions(image_np, class_probs_full, line_prob_full, palette)
        gt_overlay = overlay_line_predictions(image_np, gt_class_onehot, gt_line_full, palette)

        legend_handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=palette[k], markersize=10, label=name)
            for k, name in enumerate(LINE_NAMES[:num_classes])
        ]

        fig, axes = plt.subplots(2, 3, figsize=(16, 8))
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title("Input frame")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(gt_overlay)
        axes[0, 1].set_title("GT overlay")
        axes[0, 1].axis("off")
        axes[0, 2].imshow(_side_rgb(gt_side_full, gt_court_full))
        axes[0, 2].set_title("GT court side/mask")
        axes[0, 2].axis("off")
        axes[1, 0].imshow(pred_overlay)
        axes[1, 0].set_title("Prediction overlay")
        axes[1, 0].axis("off")
        axes[1, 1].imshow(line_prob_full, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Predicted lineness")
        axes[1, 1].axis("off")
        axes[1, 2].imshow(_side_rgb(side_prob_full, court_prob_full >= 0.5))
        axes[1, 2].set_title("Predicted court side (masked)")
        axes[1, 2].axis("off")
        fig.legend(handles=legend_handles, loc="lower center", ncol=min(5, num_classes), frameon=False)
        fig.tight_layout(rect=(0.0, 0.04, 1.0, 1.0))
        out_path = args.out / f"line_vis_{i:03d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


def _visualize_folder(args: argparse.Namespace) -> None:
    model = CourtLineLightning.load_from_checkpoint(args.checkpoint, map_location="cpu", pretrained=False)
    num_classes = int(model.hparams.num_classes)
    image_size = (args.image_height, args.image_width)
    image_paths = _iter_image_folder(args.image_folder, args.image_glob)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()
    args.out.mkdir(parents=True, exist_ok=True)

    palette = class_palette(num_classes)
    for i, image_path in enumerate(image_paths):
        full_image, model_image = _load_plain_image(image_path, image_size)
        line_prob_full, class_probs_full, _, _ = _predict_full_resolution(
            model, model_image, full_image.shape[:2], device
        )
        heatmap_overlay = overlay_line_predictions(full_image, class_probs_full, line_prob_full, palette)
        out_path = args.out / f"{image_path.stem}_line_heatmap.png"
        _save_rgb(out_path, heatmap_overlay)
        print(f"frame {i + 1}/{len(image_paths)}: wrote {out_path}")


def _resize(arr: np.ndarray, size: tuple[int, int], mode: str) -> np.ndarray:
    h, w = size
    tensor = torch.from_numpy(arr)
    if tensor.dim() == 2:
        tensor = tensor[None, None]
    elif tensor.dim() == 3:
        tensor = tensor[None]
    align = False if mode == "bilinear" else None
    if mode == "nearest":
        out = F.interpolate(tensor.float(), size=(h, w), mode="nearest")
    else:
        out = F.interpolate(tensor.float(), size=(h, w), mode=mode, align_corners=align)
    return out.squeeze().numpy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
        p.add_argument("--batch-size", type=int, default=4)
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--seed", type=int, default=1430)
        p.add_argument("--val-fraction", type=float, default=0.15)
        p.add_argument("--test-fraction", type=float, default=0.15)
        p.add_argument("--output-stride", type=int, default=2)
        p.add_argument("--sigma", type=float, default=1.5)
        p.add_argument("--side-blur-sigma", type=float, default=1.0)
        p.add_argument("--image-height", type=int, default=384)
        p.add_argument("--image-width", type=int, default=640)

    train_parser = subparsers.add_parser("train", help="Train the expanded FIBA court-marking model.")
    add_data_args(train_parser)
    train_parser.add_argument("--model-name", default="convnext_base.dinov3_lvd1689m")
    train_parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--decoder-channels", type=int, default=128)
    train_parser.add_argument("--layout-channels", type=int, default=None)
    train_parser.add_argument("--lambda-dice", type=float, default=1.0)
    train_parser.add_argument("--lambda-focal", type=float, default=1.0)
    train_parser.add_argument("--lambda-class", type=float, default=1.0)
    train_parser.add_argument("--lambda-side", type=float, default=0.25)
    train_parser.add_argument("--lambda-court", type=float, default=0.5)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--warmup-steps", type=int, default=200)
    train_parser.add_argument("--max-epochs", type=int, default=20)
    train_parser.add_argument("--precision", default="32-true")
    train_parser.add_argument("--out", type=Path, default=Path("checkpoints/fiba_court_markings_side"))
    train_parser.add_argument("--log-dir", type=Path, default=Path("lightning_logs"))
    train_parser.add_argument("--log-name", default="fiba_court_markings_side")
    train_parser.add_argument("--fast-dev-run", action="store_true")
    train_parser.add_argument("--limit-train-batches", type=float, default=1.0)
    train_parser.add_argument("--limit-val-batches", type=float, default=1.0)
    train_parser.add_argument("--visualize-best", action=argparse.BooleanOptionalAction, default=False)
    train_parser.add_argument("--visualize-min-epoch", type=int, default=5)
    train_parser.add_argument("--visualize-dataset-count", type=int, default=10)
    train_parser.add_argument("--visualize-footage-stride", type=int, default=5)
    train_parser.add_argument("--visualize-num-workers", type=int, default=0)
    train_parser.add_argument("--test-footage", type=Path, default=Path("test_footage"))
    train_parser.add_argument("--best-vis-out", type=Path, default=Path("tests/fiba_court_markings_side_best_vis"))
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved checkpoint.")
    add_data_args(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--precision", default="32-true")
    eval_parser.set_defaults(func=evaluate)

    vis_parser = subparsers.add_parser("vis", help="Save line prediction visualizations.")
    vis_parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    vis_parser.add_argument("--checkpoint", type=Path, required=True)
    vis_parser.add_argument("--out", type=Path, default=Path("results/line_vis"))
    vis_parser.add_argument("--count", type=int, default=6)
    vis_parser.add_argument("--split", choices=("val", "test"), default="val")
    vis_parser.add_argument("--image-folder", type=Path, default=None, help="Optional folder of images to visualize in filename order.")
    vis_parser.add_argument("--image-glob", default="*.jpg", help="Glob used with --image-folder.")
    vis_parser.add_argument("--num-workers", type=int, default=0)
    vis_parser.add_argument("--seed", type=int, default=1430)
    vis_parser.add_argument("--side-blur-sigma", type=float, default=1.0)
    vis_parser.add_argument("--image-height", type=int, default=384)
    vis_parser.add_argument("--image-width", type=int, default=640)
    vis_parser.add_argument("--cpu", action="store_true")
    vis_parser.set_defaults(func=_visualize)

    return parser


def main() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    args = build_parser().parse_args()
    if args.command == "vis" and args.image_folder is not None:
        args.func = _visualize_folder
    args.func(args)


if __name__ == "__main__":
    main()
