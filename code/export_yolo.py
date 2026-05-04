from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import yaml
from tqdm import tqdm

from .deepsport_yolo import (
    build_index_path_to_annotations,
    collect_labels_for_frame,
    default_train_val_split,
    iter_frame0_pngs,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLO dataset from DeepSport data/")
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--output", type=Path, default=Path("results/yolo_deepsport"))
    p.add_argument("--val-mod", type=int, default=10)
    p.add_argument("--copy-images", action="store_true")
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def _safe_link_or_copy(src: Path, dst: Path, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


def _flat_image_name(frame: Path) -> str:
    arena = frame.parent.parent.name
    game = frame.parent.name
    stem = frame.stem
    return f"{arena}__{game}__{stem}.png"


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    out = args.output.expanduser().resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Data root not found: {data_root}")

    index_map = build_index_path_to_annotations(data_root)
    frames = iter_frame0_pngs(data_root)
    labeled: list[tuple[Path, str, list[str]]] = []
    for frame in tqdm(frames, desc="Scan labels"):
        lines = collect_labels_for_frame(frame, index_map)
        if lines is None:
            continue
        split = default_train_val_split(frame, val_mod=args.val_mod)
        labeled.append((frame, split, lines))
        if args.limit is not None and len(labeled) >= args.limit:
            break

    if not labeled:
        raise SystemExit("No labeled frames found.")

    for frame, split, lines in tqdm(labeled, desc="Write files"):
        name = _flat_image_name(frame)
        img_dst = out / "images" / split / name
        lab_dst = out / "labels" / split / name.replace(".png", ".txt")
        _safe_link_or_copy(frame, img_dst, copy=args.copy_images)
        lab_dst.parent.mkdir(parents=True, exist_ok=True)
        lab_dst.write_text("\n".join(lines) + "\n")

    names = {0: "player", 1: "ball"}
    data_yaml = {
        "path": str(out),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }
    yaml_path = out / "data.yaml"
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data_yaml, f, sort_keys=False, default_flow_style=False)

    print(f"[export_yolo] Wrote {len(labeled)} samples to {out}")
    print(f"[export_yolo] data.yaml -> {yaml_path}")


if __name__ == "__main__":
    main()
