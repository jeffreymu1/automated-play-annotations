# Court Detection

Basketball court line detection and geometric refinement for DeepSport frames or image folders.

## Setup

Use `uv` to create and sync the development environment:

```bash
uv sync
```

You can also install the package into an existing `uv` environment:

```bash
uv pip install .
```

For editable development:

```bash
uv pip install -e .
```

## Court Line Detection

Train a court line detector:

```bash
uv run python scripts/court_line_detection.py train --root data/deepsport-dataset
```

Checkpoints are written to `checkpoints/court_lines/`, with TensorBoard logs in `lightning_logs/court_lines/`.

Save line heatmap visualizations from a checkpoint:

```bash
uv run python scripts/court_line_detection.py vis \
  --checkpoint checkpoints/court_lines/last.ckpt
```

Outputs are PNG visualizations in `results/line_vis/`. To run on your own images, add `--image-folder path/to/images`.

## Court Line Refinement

Run geometric refinement from a trained line checkpoint:

```bash
uv run python scripts/court_line_refinement.py \
  --checkpoint checkpoints/court_lines/last.ckpt \
  --root data/deepsport-dataset
```

Outputs are written to `results/line_refinement/`:

- `<name>.png`: stage-by-stage heatmap, fitted lines, homography, and distortion visualization.
- `<name>.json`: homography, distortion, fitted line diagnostics, and refinement status.
- `<name>_full_overlay.png`: refined court overlay on the full-resolution frame.

For a folder of images:

```bash
uv run python scripts/court_line_refinement.py \
  --checkpoint checkpoints/court_lines/last.ckpt \
  --image-folder path/to/images
```

This writes one `*_full_overlay.png` per image plus `diagnostics.json` in `results/line_refinement/`.
