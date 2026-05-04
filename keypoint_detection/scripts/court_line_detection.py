"""Train, evaluate, and visualize the basketball court line detector."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F

from court_detection.geometry import LINE_NAMES
from court_detection.lines import (
    CourtLineDataModule,
    CourtLineLightning,
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
    )


def train(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    dm = _make_datamodule(args)
    model = CourtLineLightning(
        model_name=args.model_name,
        pretrained=args.pretrained,
        decoder_channels=args.decoder_channels,
        output_stride=args.output_stride,
        sigma=args.sigma,
        lambda_dice=args.lambda_dice,
        lambda_focal=args.lambda_focal,
        lambda_class=args.lambda_class,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.out,
        filename="court-lines-{epoch:03d}-{val_line_iou:.3f}",
        monitor="val_line_iou",
        mode="max",
        save_last=True,
        auto_insert_metric_name=False,
    )
    logger = TensorBoardLogger(save_dir=args.log_dir, name=args.log_name)
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=args.precision,
        callbacks=[checkpoint_cb],
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
) -> tuple[np.ndarray, np.ndarray]:
    image_t = torch.from_numpy(model_image).permute(2, 0, 1).unsqueeze(0).to(device).float()
    with torch.no_grad():
        line_prob, class_probs = model.predict(image_t)
    line_prob_full = F.interpolate(
        line_prob.unsqueeze(1), size=output_size, mode="bilinear", align_corners=False
    )[0, 0].cpu().numpy()
    class_probs_full = F.interpolate(
        class_probs, size=output_size, mode="bilinear", align_corners=False
    )[0].cpu().numpy()
    return line_prob_full, class_probs_full


def _save_rgb(path: Path, image: np.ndarray) -> None:
    out = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))


def _visualize(args: argparse.Namespace) -> None:
    model = CourtLineLightning.load_from_checkpoint(args.checkpoint, map_location="cpu", pretrained=False)
    num_classes = int(model.hparams.num_classes)
    output_stride = int(model.hparams.output_stride)
    sigma = float(model.hparams.sigma)
    image_size = (args.image_height, args.image_width)

    dm = CourtLineDataModule(
        root=args.root,
        image_size=image_size,
        output_stride=output_stride,
        sigma=sigma,
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
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
            line_prob, class_probs = model.predict(image)
        line_prob_full = F.interpolate(
            line_prob.unsqueeze(1), size=image_size, mode="bilinear", align_corners=False
        )[0, 0].cpu().numpy()
        class_probs_full = F.interpolate(
            class_probs, size=image_size, mode="bilinear", align_corners=False
        )[0].cpu().numpy()

        image_np = sample["image"].permute(1, 2, 0).numpy()
        gt_line = sample["lineness"].numpy()
        gt_class = sample["class_target"].numpy()

        gt_line_full = _resize(gt_line, image_size, mode="bilinear")
        gt_class_full = _resize(gt_class.astype(np.float32), image_size, mode="nearest").astype(np.int64)
        gt_class_onehot = np.zeros((num_classes, *image_size), dtype=np.float32)
        for k in range(num_classes):
            gt_class_onehot[k] = (gt_class_full == k).astype(np.float32)

        pred_overlay = overlay_line_predictions(image_np, class_probs_full, line_prob_full, palette)
        gt_overlay = overlay_line_predictions(image_np, gt_class_onehot, gt_line_full, palette)

        legend_handles = [
            plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=palette[k], markersize=10, label=name)
            for k, name in enumerate(LINE_NAMES[:num_classes])
        ]

        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        axes[0, 0].imshow(image_np)
        axes[0, 0].set_title("Input frame")
        axes[0, 0].axis("off")
        axes[0, 1].imshow(gt_overlay)
        axes[0, 1].set_title("GT overlay")
        axes[0, 1].axis("off")
        axes[1, 0].imshow(pred_overlay)
        axes[1, 0].set_title("Prediction overlay")
        axes[1, 0].axis("off")
        axes[1, 1].imshow(line_prob_full, cmap="inferno", vmin=0.0, vmax=1.0)
        axes[1, 1].set_title("Predicted lineness")
        axes[1, 1].axis("off")
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
        line_prob_full, class_probs_full = _predict_full_resolution(
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
        p.add_argument("--image-height", type=int, default=384)
        p.add_argument("--image-width", type=int, default=640)

    train_parser = subparsers.add_parser("train", help="Train the line-segmentation model.")
    add_data_args(train_parser)
    train_parser.add_argument("--model-name", default="convnext_base.dinov3_lvd1689m")
    train_parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--decoder-channels", type=int, default=128)
    train_parser.add_argument("--lambda-dice", type=float, default=1.0)
    train_parser.add_argument("--lambda-focal", type=float, default=1.0)
    train_parser.add_argument("--lambda-class", type=float, default=1.0)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--warmup-steps", type=int, default=200)
    train_parser.add_argument("--max-epochs", type=int, default=10)
    train_parser.add_argument("--precision", default="32-true")
    train_parser.add_argument("--out", type=Path, default=Path("checkpoints/court_lines"))
    train_parser.add_argument("--log-dir", type=Path, default=Path("lightning_logs"))
    train_parser.add_argument("--log-name", default="court_lines")
    train_parser.add_argument("--fast-dev-run", action="store_true")
    train_parser.add_argument("--limit-train-batches", type=float, default=1.0)
    train_parser.add_argument("--limit-val-batches", type=float, default=1.0)
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
