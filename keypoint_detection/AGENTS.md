This repo is implementng a keypoint detection for a basketball court.
Dataset loader is in deepsport_dataset.py. 
Virtual environment is managed by uv. You can use `uv run...` to run python with the correct dependencies.

# Training and Target Generation

Court geometry lives in `src/court_detection/geometry.py`. Use the dataset convention here: world units are centimetres, origin is the far-left court corner, x runs along court length, y along court width, and z points down. Add or adjust court markings there first.

Target image generation is handled by `src/court_detection/lines.py`: `CourtLineFrameDataset` projects physical markings through DeepSport camera calibration, applies crop/score-bar/color augmentation, and renders `lineness`, shared per-marking class targets, `court_mask`, `side_target`, and `side_weight`. `court_mask` is the projected visible court polygon. `side_target` splits left/right by the projected halfcourt line, while `side_weight` mirrors `court_mask` so side loss ignores non-court pixels. Foul-line targets are masked by the free-throw circle there.

The expanded FIBA marking model is defined in `src/court_detection/markings.py`, mostly by reusing `src/court_detection/lines.py`. It uses shared left/right marking classes (`baseline`, `lane_far`, `lane_near`, `foul`, `three_point_arc`, `free_throw_circle`) plus unique straddling classes (`sideline_far`, `sideline_near`, `halfcourt`). The lineness and class heads use the high-resolution U-Net marking decoder, while the court-side and court-mask heads use a separate lightweight `c5+c4` DINO layout branch so they rely more on global court structure and less on shallow court-marking texture.

Train with:

    uv run python scripts/court_line_detection.py train --max-epochs 20 --visualize-best --visualize-min-epoch 5 --visualize-dataset-count 10 --visualize-footage-stride 5 --out checkpoints/003_fiba_court_markings_layout_side --log-name 003_fiba_court_markings_layout_side --best-vis-out tests/003_fiba_court_markings_layout_side_best_vis

This writes checkpoints to `checkpoints/003_fiba_court_markings_layout_side/`, TensorBoard logs to `lightning_logs/003_fiba_court_markings_layout_side/`, and best-checkpoint PNG panels to `tests/003_fiba_court_markings_layout_side_best_vis/`. Best-checkpoint visualization is handled by `BestCheckpointVisualizationCallback` in `scripts/court_line_detection.py`: after a new best `val_line_iou` at or after `--visualize-min-epoch`, it saves a callback checkpoint, generates prediction panels for the first `--visualize-dataset-count` test samples and every `--visualize-footage-stride` frame from `test_footage`, and logs the same figures to TensorBoard with `logger.experiment.add_figure`. Prediction panels include the predicted court mask and display predicted court side masked by that predicted court mask.

When a user asks to launch a training job, do not start it automatically. Provide the exact `uv run python scripts/court_line_detection.py train ...` command for the user to run. When naming a new checkpoint/log/visualization variant, scan `checkpoints/` for existing three-digit prefixes and use the next prefix, for example `001_fiba_court_markings_court_mask`, `002_...`, etc. 

Visualize learning targets with:

    uv run python scripts/visualize_court_learning_targets.py --count 10 --out tests/court_learning_targets

That script saves compact 2x2 PNG summaries and `.npz` target dumps. Use it after geometry or target-rendering changes to sanity-check labels before training.

Visualize marking model predictions with:

    uv run python tests/visualize_markings_output.py dataset --checkpoint checkpoints/003_fiba_court_markings_layout_side/last.ckpt --split test --count 20 --out tests/markings_dataset_test_vis
    uv run python tests/visualize_markings_output.py frames --checkpoint checkpoints/003_fiba_court_markings_layout_side/last.ckpt --frames test_footage --count 20 --video-stride 60 --out tests/markings_test_footage_vis

The prediction visualizer has `dataset` mode for DeepSport loader samples, including GT-vs-prediction panels, and `frames` mode for a single image, a frame folder, or sampled video files. `frames` mode searches recursively by default, includes videos by default, samples videos with `--video-stride`, and writes full panels by default; use `--no-panel` only when plain overlay PNGs are desired. Prediction panels show the input, prediction overlay, predicted court mask, lineness heatmap, predicted marking classes when no GT overlay is present, and predicted court side masked by predicted court mask.
