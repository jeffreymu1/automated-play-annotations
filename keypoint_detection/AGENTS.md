This repo implements keypoint and marking detection for a basketball court.
Dataset loading starts in `src/court_detection/dataset.py`.
The virtual environment is managed by uv; use `uv run ...` to run Python with the correct dependencies.

# Ground Rules

- Do not worry about `__pycache__`; those files are already in gitignore.
- When a user asks to launch a training job, do not start it automatically. Provide the exact `uv run python scripts/court_line_detection.py train ...` command for the user to run.
- When naming a new checkpoint, log, or visualization variant, scan `checkpoints/` for existing three-digit prefixes and use the next prefix, for example `001_fiba_court_markings_court_mask`, `002_...`, etc.
- When generating new refinement test results in `tests/output`, use four-digit prefixes, e.g. "0005_...". Use the next prefix as appropriate.
- When generating groundtruth visualization, additionally prefix with "gt_", e.g. "gt_0006_add_line". 

# Core Code Map

- `src/court_detection/dataset.py`: DeepSport dataset loading.
- `src/court_detection/geometry.py`: court geometry and physical markings.
- `src/court_detection/lines.py`: learning-target generation and `CourtLineFrameDataset`.
- `src/court_detection/augmentation.py`: crop, score-bar, overlay-box, and color augmentation helpers.
- `src/court_detection/midcourt_stitch_dataset.py`: virtual midcourt projection augmentation.
- `src/court_detection/markings.py`: expanded FIBA marking model.
- `src/court_detection/marking_refinement.py`: marking primitive fitting and homography refinement.
- `scripts/court_line_detection.py`: training entry point and best-checkpoint visualization callback.

# Court Geometry

Court geometry lives in `src/court_detection/geometry.py`. Use the dataset convention there: world units are centimetres, origin is the far-left court corner, x runs along court length, y along court width, and z points down.

Add or adjust court markings in `geometry.py` first, then validate the generated targets before training.

# Target Generation

Target image generation is handled by `src/court_detection/lines.py`. `CourtLineFrameDataset` projects physical markings through DeepSport camera calibration and renders:

- `lineness`
- shared per-marking class targets
- `court_mask`
- `side_target`
- `side_weight`

`court_mask` is the projected visible court polygon. `side_target` splits left and right by the projected halfcourt line, while `side_weight` mirrors `court_mask` so side loss ignores non-court pixels. Foul-line targets are masked by the free-throw circle.

Player and ball annotation masks come from the DeepSport `*_humans.png` sidecar masks (`1xxx` human instances and `3xxx` ball instances). `CourtLineFrameDataset` can use them with `use_player_occlusion=True` to zero `lineness` and class supervision under annotated players and balls. This is a supervision mask, not data augmentation.

# Data Augmentation

Image augmentation helpers live in `src/court_detection/augmentation.py` and are applied by `CourtLineFrameDataset`:

- random crop/resize keeps projected line coordinates aligned
- score-bar and overlay-box augmentation blank a random lower-screen rectangle and mask labels there
- color augmentation changes brightness, contrast, saturation, and occasional blur

Midcourt projection augmentation lives in `src/court_detection/midcourt_stitch_dataset.py`. It groups DeepSport frames by first-level game folder, classifies left/right side from calibration extrinsics, randomly pairs same-game halves, undistorts both frames, homography-warps them into an averaged virtual camera, and cross-fades overlap.

When validating augmented labels, remember shared classes such as `three_point_arc` and `free_throw_circle` should render on both court sides.

# Marking Model

The expanded FIBA marking model is defined in `src/court_detection/markings.py`, mostly by reusing `src/court_detection/lines.py`.

It uses shared left/right marking classes:

- `baseline`
- `lane_far`
- `lane_near`
- `foul`
- `three_point_arc`
- `free_throw_circle`

It also uses unique straddling classes:

- `sideline_far`
- `sideline_near`
- `halfcourt`

The lineness and class heads use the high-resolution U-Net marking decoder. The court-side and court-mask heads use a separate lightweight `c5+c4` DINO layout branch so they rely more on global court structure and less on shallow court-marking texture.

# Training

Train with:

    uv run python scripts/court_line_detection.py train --max-epochs 20 --visualize-best --visualize-min-epoch 5 --visualize-dataset-count 10 --visualize-footage-stride 5 --out checkpoints/003_fiba_court_markings_layout_side --log-name 003_fiba_court_markings_layout_side

This writes checkpoints to `checkpoints/003_fiba_court_markings_layout_side/` and TensorBoard logs to `lightning_logs/003_fiba_court_markings_layout_side/`.

Best-checkpoint visualization is handled by `BestCheckpointVisualizationCallback` in `scripts/court_line_detection.py`. After a new best `val_line_iou` at or after `--visualize-min-epoch`, it generates prediction panels for the first `--visualize-dataset-count` test samples and every `--visualize-footage-stride` frame from `test_footage`, then logs the figures to TensorBoard with `logger.experiment.add_figure`.

Prediction panels include the predicted court mask and display predicted court side masked by that predicted court mask.

# Tests

- `tests/test_line_targets.py`: unit tests for line-target rendering, shared marking classes, visible-mask suppression, and soft player/ball occlusion weighting.
- `tests/test_marking_refinement.py`: unit tests for marking-refinement geometry, conic fitting, and homography correspondence logic.
- `tests/test_midcourt_stitch_dataset.py`: unit tests for virtual midcourt camera calibration and stitch sampling geometry.
- `tests/test_structured_refinement.py`: legacy structured-refinement tests for CPU and GPU RANSAC-style homography fitting on synthetic heatmaps.

# Visualizers

- `tests/visualize_court_learning_targets.py`: writes target-generation sanity-check panels and `.npz` dumps under `tests/output/court_learning_targets` by default.
- `tests/visualize_markings_output.py`: visualizes marking model predictions on DeepSport samples, midcourt stitches, image folders, and videos; outputs to `tests/output/markings_output_vis` by default.
- `tests/visualize_marking_refinement.py`: visualizes marking primitive fits and linear-DLT homography refinement on dataset samples or frames; outputs to `tests/output/marking_refinement_vis` by default.

Visualize learning targets with:

    uv run python tests/visualize_court_learning_targets.py --count 10 --out tests/output/court_learning_targets

That script saves compact 2x2 PNG summaries and `.npz` target dumps. Use it after geometry or target-rendering changes to sanity-check labels before training.

Visualize marking model predictions with:

    uv run python tests/visualize_markings_output.py dataset --checkpoint checkpoints/003_fiba_court_markings_layout_side/last.ckpt --split test --count 20 --out tests/output/markings_dataset_test_vis
    uv run python tests/visualize_markings_output.py frames --checkpoint checkpoints/003_fiba_court_markings_layout_side/last.ckpt --frames test_footage --count 20 --video-stride 60 --out tests/output/markings_test_footage_vis

The prediction visualizer has `dataset` mode for DeepSport loader samples, including GT-vs-prediction panels, and `frames` mode for a single image, a frame folder, or sampled video files. `frames` mode searches recursively by default, includes videos by default, samples videos with `--video-stride`, and writes full panels by default. Use `--no-panel` only when plain overlay PNGs are desired.

Prediction panels show the input, prediction overlay, predicted court mask, lineness heatmap, predicted marking classes when no GT overlay is present, and predicted court side masked by predicted court mask.

# Linear DLT Marking Refinement

Current homography refinement work lives in `src/court_detection/marking_refinement.py`. For now, focus on the linear-DLT path, not nonlinear refinement or randomized torch RANSAC.

`MarkingRefinementConfig` defaults to:

- `use_torch_ransac=False`
- `enable_nonlinear_refinement=False`
- `max_excluded_primitives=2`

The intent is to try the small combinatorial space of plausible primitive subsets instead of relying on random dropout; with roughly 7-8 visible primitives this is still a small number of DLT solves.

Primitive fitting first extracts per-geometry line/curve fits from `lineness * class_probs`, gated by winning class, side prediction, and court mask. Shared marking classes are split into geometry names with side-aware weights, so left/right baselines, lane lines, arcs, and circles can be reasoned about separately even when they share a network class.

Homography candidates are evaluated against the full projected court-marking template, not just the primitives used for DLT. Scoring uses max-pooled class-lineness evidence, multiplies by sidedness where applicable, and also multiplies by predicted `court_prob` so score-bar/off-court false positives do not reward a homography.

The aggregate score is weighted by world-space marking length: visible projected samples contribute according to their physical polyline length, so dragging a long unsupported sideline or baseline into the frame is penalized more than missing or matching a short segment.

Use `tests/visualize_marking_refinement.py` to inspect this pipeline. The prediction overlay in that script is also court-mask-weighted (`class_probs * court_prob` and `lineness * court_prob`) to match the scorer visually.

The current comparison folder is:

    tests/output/0004_marking_refinement_systematic_dlt_caitlin_frame0

It contains 10 sampled Caitlin Clark frames and 10 sampled Dominican/Mexico frames generated with:

    uv run python tests/visualize_marking_refinement.py frames --frames "test_footage/caitlin_clark/Caitlin Clark TAKES OVER in FIBA Basketball #FIBAWWC - FIBA Basketball (1080p, h264, youtube).mp4" --count 10 --video-stride 120 --out tests/output/0004_marking_refinement_systematic_dlt_caitlin_frame0
    uv run python tests/visualize_marking_refinement.py frames --frames "test_footage/dominican_v_mexico/Dominican Republic 🇩🇴 v Mexico 🇲🇽 Extended Highlights FIBA Basketball World Cup 2027 Americas - FIBA Basketball (1080p, h264, youtube).mp4" --count 10 --video-stride 120 --out tests/output/0004_marking_refinement_systematic_dlt_caitlin_frame0

Recent 0004 results: Caitlin succeeded on 6/10 sampled frames and Dominican/Mexico succeeded on 5/10. The important Caitlin frame 0 behavior is that the selected DLT primitives are `sideline_far`, `baseline_left`, `lane_left_far`, and `lane_left_near`; the distorted near sideline is no longer selected.
