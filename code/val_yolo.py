from __future__ import annotations

import argparse
from pathlib import Path

from .train_yolo import resolve_train_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate a YOLO checkpoint on DeepSport-export data.yaml")
    p.add_argument("--weights", type=Path, required=True, help="Trained weights, e.g. results/train_runs/.../best.pt")
    p.add_argument("--data", type=Path, default=Path("results/yolo_deepsport/data.yaml"))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--split", type=str, default="val", choices=("train", "val", "test"))
    p.add_argument("--plots", action="store_true", help="Save PR / prediction plots under runs/")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data.expanduser().resolve()
    weights = args.weights.expanduser().resolve()
    if not data_yaml.is_file():
        raise SystemExit(f"Missing {data_yaml}")
    if not weights.is_file():
        raise SystemExit(f"Missing weights {weights}")

    from ultralytics import YOLO

    device = resolve_train_device(args.device)
    print(f"[val_yolo] device={device} weights={weights} data={data_yaml}")

    model = YOLO(str(weights))
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        split=args.split,
        plots=args.plots,
        verbose=True,
    )
    box = metrics.box
    print(f"[val_yolo] mAP50     {float(box.map50):.4f}")
    print(f"[val_yolo] mAP50-95  {float(box.map):.4f}")
    maps = getattr(box, "maps", None)
    if maps is not None:
        seq = maps.tolist() if hasattr(maps, "tolist") else list(maps)
        print(f"[val_yolo] per-class mAP50-95: {seq}")


if __name__ == "__main__":
    main()
