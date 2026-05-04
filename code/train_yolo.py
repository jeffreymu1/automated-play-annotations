from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

# Ultralytics augmentation bundle tuned for arena basketball (lighting, mild camera motion).
PRESET_DEEPSPORT: dict[str, Any] = {
    "degrees": 5.0,
    "translate": 0.1,
    "scale": 0.55,
    "shear": 1.0,
    "perspective": 0.0,
    "fliplr": 0.5,
    "flipud": 0.0,
    "hsv_h": 0.02,
    "hsv_s": 0.55,
    "hsv_v": 0.45,
    "mosaic": 1.0,
}


def resolve_train_device(device_arg: str) -> str | int:
    import torch

    d = device_arg.strip().lower()
    if d == "cpu":
        return "cpu"
    if d in ("auto", "cuda", "gpu"):
        if not torch.cuda.is_available():
            raise SystemExit(
                "CUDA not available. Use a GPU node or pass --device cpu.\n"
                "CPU-only: --device cpu"
            )
        return 0
    if d.isdigit() or "," in device_arg:
        if not torch.cuda.is_available():
            raise SystemExit(
                f"--device {device_arg!r} needs CUDA, or use --device cpu."
            )
    return device_arg


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO fine-tune")
    p.add_argument("--data", type=Path, default=Path("results/yolo_deepsport/data.yaml"))
    p.add_argument("--model", type=str, default="yolov8m.pt")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--project", type=Path, default=Path("results/train_runs"))
    p.add_argument("--name", type=str, default="deepsport_ball_player")
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--cos-lr", action="store_true")
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--close-mosaic", type=int, default=15)
    p.add_argument("--cache", choices=("off", "ram", "disk"), default="off")
    p.add_argument("--resume", action="store_true")
    p.add_argument(
        "--preset",
        choices=("none", "deepsport"),
        default="none",
        help="Augmentation preset: 'deepsport' for arena lighting / mild warp; 'none' for Ultralytics defaults.",
    )
    p.add_argument("--rect", action="store_true", help="Rectangular training batches (often faster at high imgsz)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.expanduser().resolve()
    if not data_yaml.is_file():
        raise SystemExit(
            f"Missing {data_yaml}. Run: uv run cs1430-export-yolo --data-root data --output results/yolo_deepsport"
        )

    from ultralytics import YOLO

    device = resolve_train_device(args.device)
    print(f"[train_yolo] Using device: {device}  preset={args.preset}")

    cache_map = {"off": False, "ram": "ram", "disk": "disk"}
    model = YOLO(args.model)

    aug: dict[str, Any] = {}
    if args.preset == "deepsport":
        aug.update(PRESET_DEEPSPORT)

    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.project),
        name=args.name,
        patience=args.patience,
        exist_ok=True,
        resume=args.resume,
        workers=args.workers,
        cos_lr=args.cos_lr,
        mixup=args.mixup,
        close_mosaic=args.close_mosaic,
        cache=cache_map[args.cache],
        rect=args.rect,
        **aug,
    )
    trainer = getattr(model, "trainer", None)
    save_dir = getattr(trainer, "save_dir", None) if trainer else None
    if save_dir:
        best = Path(save_dir) / "weights" / "best.pt"
        print(f"[train_yolo] Best weights: {best.resolve()}")
    else:
        print(f"[train_yolo] Best weights: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
