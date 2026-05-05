"""Train, evaluate, and visualize the court-keypoint heatmap predictor."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import lightning as L
import matplotlib.pyplot as plt
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.nn import functional as F

from court_detection.heatmap import (
    CourtKeypointDataModule,
    CourtKeypointLightning,
    decode_heatmap_peaks,
)


def _draw_predictions(
    image,
    keypoints: torch.Tensor,
    peaks: list[list[tuple[float, float, float]]],
) -> object:
    canvas = (image * 255.0).clip(0, 255).astype("uint8").copy()
    for idx, kp_peaks in enumerate(peaks):
        if keypoints[idx, 2] > 0:
            x = int(round(float(keypoints[idx, 0])))
            y = int(round(float(keypoints[idx, 1])))
            cv2.circle(canvas, (x, y), 5, (80, 220, 80), -1, lineType=cv2.LINE_AA)
        if kp_peaks:
            x, y, score = kp_peaks[0]
            cv2.circle(canvas, (int(round(x)), int(round(y))), 6, (255, 80, 220), 2, lineType=cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"{idx}:{score:.2f}",
                (int(round(x)) + 7, int(round(y)) - 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 80, 220),
                1,
                cv2.LINE_AA,
            )
    return canvas


def _save_visualizations(args: argparse.Namespace) -> None:
    model = CourtKeypointLightning.load_from_checkpoint(args.checkpoint, map_location="cpu", pretrained=False)
    image_size = (int(model.hparams.input_height), int(model.hparams.input_width))
    dm = CourtKeypointDataModule(
        root=args.root,
        image_size=image_size,
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    dm.setup("test")

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()
    args.out.mkdir(parents=True, exist_ok=True)

    dataset = dm.val_dataset if args.split == "val" else dm.test_dataset
    count = min(args.count, len(dataset))
    for i in range(count):
        sample = dataset[i]
        image = sample["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            heatmaps = model.predict_heatmaps(image)[0].cpu()
        peaks = decode_heatmap_peaks(heatmaps, image_size=image_size, top_k=1)

        image_np = sample["image"].permute(1, 2, 0).numpy()
        pred_panel = _draw_predictions(image_np, sample["keypoints"], peaks)

        max_heatmap = heatmaps.max(dim=0).values[None, None]
        max_heatmap = F.interpolate(max_heatmap, size=image_size, mode="bilinear", align_corners=False)[0, 0]
        heat_np = max_heatmap.numpy()
        heat_np = (heat_np / max(heat_np.max(), 1e-6) * 255.0).astype("uint8")
        colored = cv2.applyColorMap(heat_np, cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        heat_panel = (0.55 * (image_np * 255.0) + 0.45 * colored).clip(0, 255).astype("uint8")

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        axes[0].imshow(pred_panel)
        axes[0].set_title("GT green / pred magenta")
        axes[0].axis("off")
        axes[1].imshow(heat_panel)
        axes[1].set_title("Max predicted heatmap")
        axes[1].axis("off")
        fig.tight_layout()
        out_path = args.out / f"heatmap_vis_{i:03d}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"wrote {out_path}")


def _make_datamodule(args: argparse.Namespace, image_size: tuple[int, int] | None = None) -> CourtKeypointDataModule:
    if image_size is None:
        image_size = (args.image_height, args.image_width)
    return CourtKeypointDataModule(
        root=args.root,
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
    )


def train(args: argparse.Namespace) -> None:
    L.seed_everything(args.seed, workers=True)
    dm = _make_datamodule(args)
    model = CourtKeypointLightning(
        model_name=args.model_name,
        pretrained=args.pretrained,
        input_height=args.image_height,
        input_width=args.image_width,
        heatmap_stride=args.heatmap_stride,
        sigma=args.sigma,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    checkpoint_cb = ModelCheckpoint(
        dirpath=args.out,
        filename="court-keypoint-{epoch:03d}-{val_mean_pixel_error:.2f}",
        monitor="val_mean_pixel_error",
        mode="min",
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
    model = CourtKeypointLightning.load_from_checkpoint(args.checkpoint, map_location="cpu", pretrained=False)
    image_size = (int(model.hparams.input_height), int(model.hparams.input_width))
    dm = _make_datamodule(args, image_size=image_size)
    trainer = L.Trainer(accelerator="auto", devices="auto", precision=args.precision)
    trainer.test(model, datamodule=dm)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_data_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--num-workers", type=int, default=4)
        p.add_argument("--seed", type=int, default=1430)
        p.add_argument("--val-fraction", type=float, default=0.15)
        p.add_argument("--test-fraction", type=float, default=0.15)

    train_parser = subparsers.add_parser("train", help="Train the heatmap predictor.")
    add_data_args(train_parser)
    train_parser.add_argument("--model-name", default="convnext_base.dinov3_lvd1689m")
    train_parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True)
    train_parser.add_argument("--image-height", type=int, default=384)
    train_parser.add_argument("--image-width", type=int, default=640)
    train_parser.add_argument("--heatmap-stride", type=int, default=4)
    train_parser.add_argument("--sigma", type=float, default=2.0)
    train_parser.add_argument("--lr", type=float, default=1e-3)
    train_parser.add_argument("--weight-decay", type=float, default=1e-4)
    train_parser.add_argument("--max-epochs", type=int, default=40)
    train_parser.add_argument("--precision", default="32-true")
    train_parser.add_argument("--out", type=Path, default=Path("checkpoints/court_keypoints"))
    train_parser.add_argument("--log-dir", type=Path, default=Path("lightning_logs"))
    train_parser.add_argument("--log-name", default="court_keypoints")
    train_parser.add_argument("--fast-dev-run", action="store_true")
    train_parser.add_argument("--limit-train-batches", type=float, default=1.0)
    train_parser.add_argument("--limit-val-batches", type=float, default=1.0)
    train_parser.set_defaults(func=train)

    eval_parser = subparsers.add_parser("eval", help="Evaluate a saved checkpoint.")
    add_data_args(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, required=True)
    eval_parser.add_argument("--precision", default="32-true")
    eval_parser.set_defaults(func=evaluate)

    vis_parser = subparsers.add_parser("vis", help="Save a few heatmap prediction visualizations.")
    vis_parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    vis_parser.add_argument("--checkpoint", type=Path, required=True)
    vis_parser.add_argument("--out", type=Path, default=Path("results/heatmap_vis"))
    vis_parser.add_argument("--count", type=int, default=8)
    vis_parser.add_argument("--split", choices=("val", "test"), default="val")
    vis_parser.add_argument("--num-workers", type=int, default=0)
    vis_parser.add_argument("--seed", type=int, default=1430)
    vis_parser.add_argument("--cpu", action="store_true")
    vis_parser.set_defaults(func=_save_visualizations)

    return parser


def main() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
