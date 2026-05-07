# Refinement Handoff

## Model Outputs

The current marking model is `FibaCourtMarkingLightning` in `src/court_detection/markings.py`, inherited from `CourtLineLightning`.

`model.predict(image)` returns:

- `line_prob`: sigmoid lineness probability, shape `(B, H, W)`.
- `class_probs`: softmax marking-class probabilities, shape `(B, K, H, W)`.
- `side_prob`: sigmoid court-side probability, shape `(B, H, W)`.
- `court_prob`: sigmoid court-mask probability, shape `(B, H, W)`.

Important: `class_probs` is a softmax, so each pixel has one winning marking class. Primitive fitting should normally require `argmax(class_probs) == class_id`, not just `class_probs[class_id] >= threshold`.

Marking classes are shared left/right:

- Shared side-dependent classes: `baseline`, `lane_far`, `lane_near`, `foul`, `three_point_arc`, `free_throw_circle`.
- Straddling/non-sided classes: `sideline_far`, `sideline_near`, `halfcourt`.

`side_prob` convention:

- Near `1.0` means the `baseline_right` side of half court.
- Near `0.0` means the `baseline_left` side.

## Current Implementation

New refinement code lives in:

- `src/court_detection/marking_refinement.py`
- `tests/visualize_marking_refinement.py`

The visualizer supports:

- `dataset` mode for DeepSport test samples.
- `frames` mode for image folders and videos.

Visualization panels are currently a clean 2x3 grid:

- Input
- Prediction overlay
- Court-masked sidedness
- Lineness
- Fitted lines / curves
- Homography overlay

Existing scratch outputs were moved under `tests/marking_refinement_previous_runs/`.

The newest visualizations should use numbered iteration folders, 4 digits starting from `0001`, per user request. This convention has not been fully automated yet.

## Primitive Fitting

Primitive fitting now happens per physical geometry, not just per shared class.

Evidence is:

```python
evidence_k = line_prob * class_probs[k]
```

Pixels are gated by:

- `line_prob >= line_threshold`
- `class_probs[k] >= class_threshold`
- `evidence_k >= joint_threshold`
- `argmax(class_probs) == k` by default
- `court_prob >= court_threshold`

For sided geometries, evidence is multiplied by side compatibility:

- left geometry: `evidence_k * (1 - side_prob)`
- right geometry: `evidence_k * side_prob`

Each physical geometry must pass a total side-weighted evidence mass threshold before fitting. This avoids fitting arcs/circles to tiny noise blobs. Each physical marking now produces at most one primitive.

Straight primitives are fitted as homogeneous image lines with weighted total least squares.

Curved primitives are fitted as image conics/ellipses, not image circles. Circle parameters are only diagnostic.

## Current Homography Issue

The current homography path is incomplete.

It still effectively initializes from straight lines/intersections, then scores the projected template. Curves/conics are not properly part of the initial homography solve. This is why good primitive fits can still produce failed homography results.

The user explicitly clarified:

> There should not be a separate line DLT step. The DLT/solve should take into account all constraints: points, lines, and conic.

## Conic Constraint Direction

A fitted image ellipse alone does **not** provide point-to-point correspondence with sampled world-circle points. Equal-angle samples on the image ellipse do not correspond to equal-angle samples on the world circle under homography.

The correct geometric relationship is a conic correspondence.

For world conic `Cw`, image conic `Ci`, and world-to-image homography `H`:

```text
Ci ∝ H^{-T} Cw H^{-1}
```

Equivalently:

```text
Cw ∝ H^T Ci H
```

This constraint is nonlinear in `H`, unlike point/line DLT rows.

Next implementation should use a unified homography solver with one objective containing:

- point residuals from known line intersections
- line residuals from projected world lines to fitted image lines
- conic residuals from known world circles/arcs to fitted image conics

This should replace the current "line DLT first, conic later" structure.

