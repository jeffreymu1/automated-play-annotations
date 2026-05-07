"""Generate compact report figures for court-marking model outputs.

Examples:
    uv run python tests/visualize_report_figures.py dataset \
        --checkpoint checkpoints/003_fiba_court_markings_layout_side/last.ckpt \
        --split test --count 4

    uv run python tests/visualize_report_figures.py frames \
        --checkpoint checkpoints/003_fiba_court_markings_layout_side/last.ckpt \
        --frames test_footage --count 4 --video-stride 120
"""

from __future__ import annotations

import argparse
import re
from collections.abc import Iterable, Iterator
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

from court_detection.markings import (
    FibaCourtMarkingDataModule,
    FibaCourtMarkingLightning,
    FibaStructuredSideCourtMarkingLightning,
    class_palette,
    overlay_line_predictions,
)

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


MODEL_ARCHITECTURES = {
    "dense-side": FibaCourtMarkingLightning,
    "structured-side": FibaStructuredSideCourtMarkingLightning,
}


def _load_model(args: argparse.Namespace) -> tuple[FibaCourtMarkingLightning, torch.device]:
    if args.architecture == "auto":
        architecture_order = ("structured-side", "dense-side")
    else:
        architecture_order = (args.architecture,)

    errors: list[str] = []
    for architecture in architecture_order:
        model_cls = MODEL_ARCHITECTURES[architecture]
        try:
            model = model_cls.load_from_checkpoint(
                args.checkpoint,
                map_location="cpu",
                pretrained=False,
            )
            break
        except RuntimeError as exc:
            errors.append(f"{architecture}: {exc}")
    else:
        raise RuntimeError(f"Could not load checkpoint {args.checkpoint}:\n" + "\n".join(errors))

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model.to(device).eval()
    return model, device


def _predict_full_resolution(
    model: FibaCourtMarkingLightning,
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
        line_prob.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    class_full = F.interpolate(
        class_probs,
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0].cpu().numpy()
    side_full = F.interpolate(
        side_prob.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    court_full = F.interpolate(
        court_prob.unsqueeze(1),
        size=output_size,
        mode="bilinear",
        align_corners=False,
    )[0, 0].cpu().numpy()
    return line_full, class_full, side_full, court_full


def _class_rgb(class_probs: np.ndarray, palette: np.ndarray) -> np.ndarray:
    class_ids = class_probs.argmax(axis=0)
    return palette[class_ids]


def _side_rgb(side_prob: np.ndarray, court_prob: np.ndarray, mask_with_court: bool) -> np.ndarray:
    rgb = plt.get_cmap("coolwarm")(np.clip(side_prob, 0.0, 1.0))[..., :3].astype(np.float32)
    if mask_with_court:
        rgb[court_prob < 0.5] = 0.15
    return rgb


def _save_output_channels_figure(
    out_path: Path,
    image_rgb: np.ndarray,
    line_prob: np.ndarray,
    class_probs: np.ndarray,
    side_prob: np.ndarray,
    court_prob: np.ndarray,
    palette: np.ndarray,
    mask_side_with_court: bool,
    dpi: int,
) -> None:
    line_class_overlay = overlay_line_predictions(image_rgb, class_probs, line_prob, palette)
    fig, axes = plt.subplots(3, 2, figsize=(8, 9))
    panels = [
        (axes[0, 0], image_rgb, None),
        (axes[0, 1], line_prob, "inferno"),
        (axes[1, 0], _class_rgb(class_probs, palette), None),
        (axes[1, 1], _side_rgb(side_prob, court_prob, mask_side_with_court), None),
        (axes[2, 0], court_prob, "gray"),
        (axes[2, 1], line_class_overlay, None),
    ]
    for ax, image, cmap in panels:
        if image.ndim == 2:
            ax.imshow(image, cmap=cmap, vmin=0.0, vmax=1.0)
        else:
            ax.imshow(np.clip(image, 0.0, 1.0))
        ax.axis("off")

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=1.0, wspace=0.0, hspace=0.0)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)


def visualize_dataset(args: argparse.Namespace) -> None:
    _validate_common_args(args)
    model, device = _load_model(args)
    image_size = (args.image_height, args.image_width)
    palette = class_palette(int(model.hparams.num_classes))

    dm = FibaCourtMarkingDataModule(
        root=args.root,
        image_size=image_size,
        output_stride=int(model.hparams.output_stride),
        sigma=float(model.hparams.sigma),
        batch_size=1,
        num_workers=args.num_workers,
        seed=args.seed,
        val_fraction=args.val_fraction,
        test_fraction=args.test_fraction,
        use_player_occlusion=args.use_player_occlusion,
        pan_train_ratio=0.0,
    )
    dm.setup("test")
    dataset = {"train": dm.train_dataset, "val": dm.val_dataset, "test": dm.test_dataset}[args.split]
    indices = _sample_indices(len(dataset), args.count, args.random, args.seed)

    args.out.mkdir(parents=True, exist_ok=True)
    for out_idx, dataset_idx in enumerate(indices):
        sample = dataset[dataset_idx]
        image_rgb = sample["image"].permute(1, 2, 0).numpy()
        line_prob, class_probs, side_prob, court_prob = _predict_full_resolution(
            model,
            image_rgb,
            image_size,
            device,
        )
        out_path = args.out / f"dataset_{args.split}_{out_idx:03d}_output_channels.png"
        _save_output_channels_figure(
            out_path,
            image_rgb,
            line_prob,
            class_probs,
            side_prob,
            court_prob,
            palette,
            args.mask_side_with_court,
            args.dpi,
        )
        print(f"wrote {out_path}")


def _sample_indices(dataset_len: int, count: int, random: bool, seed: int) -> list[int]:
    if dataset_len <= 0:
        return []
    if not random:
        return list(range(min(count, dataset_len)))
    rng = np.random.default_rng(seed)
    replace = dataset_len < count
    return rng.choice(dataset_len, size=count, replace=replace).astype(int).tolist()


def visualize_frames(args: argparse.Namespace) -> None:
    _validate_common_args(args)
    if args.video_stride < 1:
        raise ValueError("--video-stride must be at least 1")
    model, device = _load_model(args)
    image_size = (args.image_height, args.image_width)
    palette = class_palette(int(model.hparams.num_classes))

    args.out.mkdir(parents=True, exist_ok=True)
    frame_iter = _iter_random_frame_folder(args) if args.random else _iter_frame_folder(args)
    seen = 0
    for label, image_rgb in frame_iter:
        line_prob, class_probs, side_prob, court_prob = _predict_full_resolution(
            model,
            image_rgb,
            image_size,
            device,
        )
        safe_label = _safe_name(label)
        out_path = args.out / f"{safe_label}_output_channels.png"
        _save_output_channels_figure(
            out_path,
            image_rgb,
            line_prob,
            class_probs,
            side_prob,
            court_prob,
            palette,
            args.mask_side_with_court,
            args.dpi,
        )
        seen += 1
        print(f"frame {seen}/{args.count}: wrote {out_path}")

    if seen == 0:
        raise RuntimeError(f"No image or video frames found under {args.frames}")


def _iter_random_frame_folder(args: argparse.Namespace) -> Iterator[tuple[str, np.ndarray]]:
    sources = _collect_frame_sources(args)
    if not sources:
        raise RuntimeError(f"No image or video frames found under {args.frames}")

    rng = np.random.default_rng(args.seed)
    replace = len(sources) < args.count
    sampled_indices = rng.choice(len(sources), size=args.count, replace=replace)
    for idx in sampled_indices:
        label, source_type, path, frame_idx = sources[int(idx)]
        if source_type == "image":
            yield label, _read_image(path)
        else:
            assert frame_idx is not None
            yield label, _read_video_frame(path, frame_idx)


def _collect_frame_sources(args: argparse.Namespace) -> list[tuple[str, str, Path, int | None]]:
    if args.frames.is_file() and args.frames.suffix.lower() in IMAGE_EXTENSIONS:
        return [(args.frames.stem, "image", args.frames, None)]
    if args.frames.is_file() and args.frames.suffix.lower() in VIDEO_EXTENSIONS:
        return _collect_video_sources(args.frames, args.video_stride)
    if not args.frames.is_dir():
        raise FileNotFoundError(f"Frame source not found: {args.frames}")

    sources: list[tuple[str, str, Path, int | None]] = []
    for path in _iter_paths(args.frames, IMAGE_EXTENSIONS, args.recursive):
        sources.append((_relative_stem(path, args.frames), "image", path, None))

    if args.include_videos:
        for video_path in _iter_paths(args.frames, VIDEO_EXTENSIONS, args.recursive):
            prefix = _relative_stem(video_path, args.frames)
            sources.extend(_collect_video_sources(video_path, args.video_stride, prefix=prefix))
    return sources


def _collect_video_sources(path: Path, stride: int, prefix: str | None = None) -> list[tuple[str, str, Path, int | None]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()
    if frame_count <= 0:
        raise RuntimeError(f"Could not determine frame count for video: {path}")

    label_prefix = prefix or path.stem
    return [
        (f"{label_prefix}_frame_{frame_idx:06d}", "video", path, frame_idx)
        for frame_idx in range(0, frame_count, stride)
    ]


def _read_video_frame(path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame_bgr = cap.read()
    finally:
        cap.release()
    if not ok:
        raise RuntimeError(f"Could not read frame {frame_idx} from video: {path}")
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _read_image(path: Path) -> np.ndarray:
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def _iter_paths(root: Path, extensions: Iterable[str], recursive: bool) -> list[Path]:
    suffixes = {ext.lower() for ext in extensions}
    paths = root.rglob("*") if recursive else root.glob("*")
    return sorted(path for path in paths if path.is_file() and path.suffix.lower() in suffixes)


def _iter_frame_folder(args: argparse.Namespace) -> Iterator[tuple[str, np.ndarray]]:
    if args.frames.is_file() and args.frames.suffix.lower() in IMAGE_EXTENSIONS:
        yield args.frames.stem, _read_image(args.frames)
        return
    if args.frames.is_file() and args.frames.suffix.lower() in VIDEO_EXTENSIONS:
        yield from _iter_video(args.frames, args.video_stride)
        return
    if not args.frames.is_dir():
        raise FileNotFoundError(f"Frame source not found: {args.frames}")

    for path in _iter_paths(args.frames, IMAGE_EXTENSIONS, args.recursive):
        yield _relative_stem(path, args.frames), _read_image(path)

    if args.include_videos:
        for video_path in _iter_paths(args.frames, VIDEO_EXTENSIONS, args.recursive):
            yield from _iter_video(video_path, args.video_stride, prefix=_relative_stem(video_path, args.frames))


def _iter_video(path: Path, stride: int, prefix: str | None = None) -> Iterator[tuple[str, np.ndarray]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    label_prefix = prefix or path.stem
    frame_idx = 0
    emitted = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % stride == 0:
                image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                yield f"{label_prefix}_frame_{frame_idx:06d}", image_rgb
                emitted += 1
            frame_idx += 1
    finally:
        cap.release()
    if emitted == 0:
        raise RuntimeError(f"No frames could be sampled from video: {path}")


def _relative_stem(path: Path, root: Path) -> str:
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path
    return _safe_name(str(rel.with_suffix("")))


def _safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "frame"


def _validate_common_args(args: argparse.Namespace) -> None:
    if args.count < 1:
        raise ValueError("--count must be at least 1")
    if args.image_height < 1 or args.image_width < 1:
        raise ValueError("--image-height and --image-width must be at least 1")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--checkpoint",
            type=Path,
            default=Path("checkpoints/006_fiba_court_markings_structured_side_1to1_pan/last-v1.ckpt"),
        )
        p.add_argument("--architecture", choices=("auto", *sorted(MODEL_ARCHITECTURES)), default="auto")
        p.add_argument("--out", type=Path, default=Path("tests/output/report_figures"))
        p.add_argument("--count", type=int, default=4)
        p.add_argument("--image-height", type=int, default=384)
        p.add_argument("--image-width", type=int, default=640)
        p.add_argument("--dpi", type=int, default=300)
        p.add_argument("--mask-side-with-court", action=argparse.BooleanOptionalAction, default=False)
        p.add_argument("--cpu", action="store_true")

    dataset_parser = subparsers.add_parser("dataset", help="Make report figures from DeepSport loader samples.")
    add_common_args(dataset_parser)
    dataset_parser.add_argument("--root", type=Path, default=Path("data/deepsport-dataset"))
    dataset_parser.add_argument("--split", choices=("train", "val", "test"), default="test")
    dataset_parser.add_argument("--num-workers", type=int, default=0)
    dataset_parser.add_argument("--seed", type=int, default=1430)
    dataset_parser.add_argument("--val-fraction", type=float, default=0.15)
    dataset_parser.add_argument("--test-fraction", type=float, default=0.15)
    dataset_parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False)
    dataset_parser.add_argument("--use-player-occlusion", action=argparse.BooleanOptionalAction, default=False)
    dataset_parser.set_defaults(func=visualize_dataset)

    frames_parser = subparsers.add_parser("frames", help="Make report figures from image frames or sampled videos.")
    add_common_args(frames_parser)
    frames_parser.add_argument("--frames", type=Path, default=Path("test_footage"))
    frames_parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    frames_parser.add_argument("--include-videos", action=argparse.BooleanOptionalAction, default=True)
    frames_parser.add_argument("--video-stride", type=int, default=60)
    frames_parser.add_argument("--random", action=argparse.BooleanOptionalAction, default=False)
    frames_parser.add_argument("--seed", type=int, default=1430)
    frames_parser.set_defaults(func=visualize_frames)

    return parser


def main() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("medium")
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
