"""Homography refinement from expanded FIBA court-marking heatmaps.

This module handles shared marking classes such as ``baseline`` and curved
markings such as ``three_point_arc`` and ``free_throw_circle``.  The homography
is initialized with DLT from straight marking candidates, then scored and
optionally refined against the full court template, including curve evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from scipy.optimize import least_squares

from court_detection.geometry import (
    BASKET_CENTER_FROM_ENDLINE_CM,
    COURT_LENGTH_CM,
    COURT_WIDTH_CM,
    FREE_THROW_CIRCLE_RADIUS_CM,
    FREE_THROW_LINE_X_CM,
    MARKING_CLASS_BY_GEOMETRY,
    MARKING_CLASS_NAMES,
    STRAIGHT_LINE_NAMES,
    THREE_POINT_RADIUS_CM,
    court_lines_world,
    court_markings_world,
)
from court_detection.structured_refinement import line_point_dlt, project_world_points


CURVE_GEOMETRY_NAMES = tuple(name for name in MARKING_CLASS_BY_GEOMETRY if name not in STRAIGHT_LINE_NAMES)
GEOMETRIES_BY_CLASS = {
    class_name: tuple(name for name, mapped in MARKING_CLASS_BY_GEOMETRY.items() if mapped == class_name)
    for class_name in MARKING_CLASS_NAMES
}


@dataclass
class MarkingRefinementConfig:
    line_threshold: float = 0.50
    class_threshold: float = 0.35
    joint_threshold: float = 0.20
    court_threshold: float = 0.20
    min_component_pixels: int = 40
    max_components_per_class: int = 4
    side_margin: float = 0.08
    require_winning_class: bool = True
    min_geometry_evidence_mass: float = 120.0
    min_geometry_evidence_mass_per_megapixel: float = 300.0
    curve_geometry_mass_multiplier: float = 2.0
    min_component_evidence_fraction: float = 0.10
    min_component_evidence_mass: float = 80.0
    ransac_iter: int = 2048
    min_dlt_lines: int = 4
    use_torch_ransac: bool = False
    primitive_keep_probability: float = 0.65
    max_excluded_primitives: int = 2
    max_dlt_candidates_per_geometry: int = 2
    n_template_samples: int = 120
    score_radius_px: int = 3
    min_projected_coverage: float = 0.08
    inlier_score_threshold: float = 0.08
    min_scored_markings: int = 5
    require_baseline: bool = True
    dlt_curve_point_weight: float = 4.0
    curve_refine_weight: float = 0.35
    point_refine_weight: float = 1.0
    enable_nonlinear_refinement: bool = False
    max_refine_nfev: int = 60
    seed: int = 1430


@dataclass
class MarkingLineFit:
    class_name: str
    geometry_name: str | None
    line_homog: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    support: int
    score: float


@dataclass
class MarkingCurveFit:
    class_name: str
    geometry_name: str | None
    conic: np.ndarray
    circle_center: np.ndarray
    circle_radius: float
    points: np.ndarray
    support: int
    score: float


@dataclass
class MarkingHomographyResult:
    H: np.ndarray | None
    success: bool
    score: float
    line_fits: dict[str, list[MarkingLineFit]] = field(default_factory=dict)
    curve_fits: dict[str, list[MarkingCurveFit]] = field(default_factory=dict)
    selected_lines: dict[str, MarkingLineFit] = field(default_factory=dict)
    point_correspondences: list[tuple[str, np.ndarray, np.ndarray]] = field(default_factory=list)
    per_geometry_scores: dict[str, float] = field(default_factory=dict)
    candidate_pixels: dict[str, int] = field(default_factory=dict)
    geometry_masses: dict[str, float] = field(default_factory=dict)
    message: str = ""


def fit_homography_from_marking_heatmaps(
    image_rgb: np.ndarray,
    lineness: np.ndarray,
    class_probs: np.ndarray,
    side_prob: np.ndarray | None = None,
    court_prob: np.ndarray | None = None,
    config: MarkingRefinementConfig | None = None,
    marking_names: tuple[str, ...] = MARKING_CLASS_NAMES,
) -> MarkingHomographyResult:
    config = MarkingRefinementConfig() if config is None else config
    image_shape = image_rgb.shape[:2]
    lineness = _resize_2d(np.asarray(lineness, dtype=np.float32), image_shape)
    class_probs = np.asarray(class_probs, dtype=np.float32)
    if class_probs.shape[-2:] != image_shape:
        class_probs = np.stack([_resize_2d(class_probs[k], image_shape) for k in range(class_probs.shape[0])])
    if court_prob is not None:
        court_prob = _resize_2d(np.asarray(court_prob, dtype=np.float32), image_shape)
    if side_prob is not None:
        side_prob = _resize_2d(np.asarray(side_prob, dtype=np.float32), image_shape)

    evidence = np.clip(lineness[None] * class_probs, 0.0, 1.0)
    line_fits, curve_fits, candidate_pixels, geometry_masses = fit_marking_primitives(
        lineness, class_probs, evidence, side_prob, court_prob, config, marking_names
    )
    geometry_candidates = _geometry_line_candidates(line_fits)
    if not _has_enough_initializer_potential(geometry_candidates, curve_fits):
        return MarkingHomographyResult(
            None,
            False,
            -np.inf,
            line_fits=line_fits,
            curve_fits=curve_fits,
            candidate_pixels=candidate_pixels,
            geometry_masses=geometry_masses,
            message="Not enough linear initializer constraints",
        )

    rng = np.random.default_rng(config.seed)
    if config.use_torch_ransac:
        ransac_result = _torch_ransac_homography(
            geometry_candidates,
            curve_fits,
            evidence,
            side_prob,
            court_prob,
            image_shape,
            config,
            marking_names,
        )
    else:
        ransac_result = _cpu_ransac_homography(
            geometry_candidates,
            curve_fits,
            evidence,
            side_prob,
            court_prob,
            image_shape,
            config,
            marking_names,
            rng,
        )
    if ransac_result is None:
        best_H = None
        best_score = -np.inf
        best_selected = {}
        best_point_corrs = []
        best_per_geometry = {}
    else:
        best_H, best_score, best_selected, best_point_corrs, best_per_geometry, backend_message = ransac_result

    if best_H is None:
        return MarkingHomographyResult(
            None,
            False,
            -np.inf,
            line_fits=line_fits,
            curve_fits=curve_fits,
            candidate_pixels=candidate_pixels,
            geometry_masses=geometry_masses,
            message="No valid homography",
        )

    refined_H = (
        refine_homography_with_primitives(best_H, best_selected, curve_fits, config, best_point_corrs)
        if config.enable_nonlinear_refinement
        else None
    )
    if refined_H is not None and _homography_is_sane(refined_H, image_shape):
        refined_score, refined_per_geometry, coverage = score_homography_template(
            refined_H, evidence, image_shape, config, marking_names, side_prob, court_prob
        )
        if coverage >= config.min_projected_coverage and refined_score >= 0.90 * best_score:
            best_H = refined_H
            best_score = refined_score
            best_per_geometry = refined_per_geometry

    return MarkingHomographyResult(
        best_H,
        True,
        float(best_score),
        line_fits=line_fits,
        curve_fits=curve_fits,
        selected_lines=best_selected,
        point_correspondences=best_point_corrs,
        per_geometry_scores=best_per_geometry,
        candidate_pixels=candidate_pixels,
        geometry_masses=geometry_masses,
        message=backend_message,
    )


def fit_marking_primitives(
    lineness: np.ndarray,
    class_probs: np.ndarray,
    evidence: np.ndarray,
    side_prob: np.ndarray | None,
    court_prob: np.ndarray | None,
    config: MarkingRefinementConfig,
    marking_names: tuple[str, ...],
) -> tuple[dict[str, list[MarkingLineFit]], dict[str, list[MarkingCurveFit]], dict[str, int], dict[str, float]]:
    line_fits: dict[str, list[MarkingLineFit]] = {}
    curve_fits: dict[str, list[MarkingCurveFit]] = {}
    candidate_pixels: dict[str, int] = {}
    geometry_masses: dict[str, float] = {}
    class_winner = np.argmax(class_probs, axis=0)
    for class_id, class_name in enumerate(marking_names[: class_probs.shape[0]]):
        base_mask = (
            (lineness >= config.line_threshold)
            & (class_probs[class_id] >= config.class_threshold)
            & (evidence[class_id] >= config.joint_threshold)
        )
        if config.require_winning_class:
            base_mask &= class_winner == class_id
        if court_prob is not None:
            base_mask &= court_prob >= config.court_threshold
        candidate_pixels[class_name] = int(base_mask.sum())
        line_fits.setdefault(class_name, [])
        curve_fits.setdefault(class_name, [])

        for geometry_name in GEOMETRIES_BY_CLASS.get(class_name, ()):
            side_weights = _geometry_side_weight_map(geometry_name, side_prob, evidence[class_id].shape, config)
            geometry_weights = evidence[class_id] * side_weights
            geometry_mask = base_mask & (geometry_weights >= config.joint_threshold)
            geometry_mass = float(geometry_weights[geometry_mask].sum())
            candidate_pixels[geometry_name] = int(geometry_mask.sum())
            geometry_masses[geometry_name] = geometry_mass
            if geometry_mass < _minimum_geometry_mass(config, evidence[class_id].shape, geometry_name):
                continue

            components = _connected_components(geometry_mask, geometry_weights, config)
            components = _significant_components(components, geometry_mass, config)
            if not components:
                continue
            pts = np.concatenate([pts for pts, _ in components], axis=0)
            weights = np.concatenate([weights for _, weights in components], axis=0)
            if geometry_name in STRAIGHT_LINE_NAMES:
                fit = _fit_line_component(class_name, geometry_name, pts, weights)
                if fit is not None:
                    line_fits[class_name].append(fit)
            elif geometry_name in CURVE_GEOMETRY_NAMES:
                fit = _fit_curve_component(class_name, geometry_name, pts, weights)
                if fit is not None:
                    curve_fits[class_name].append(fit)
    return line_fits, curve_fits, candidate_pixels, geometry_masses


def score_homography_template(
    H: np.ndarray,
    evidence: np.ndarray,
    image_shape: tuple[int, int],
    config: MarkingRefinementConfig,
    marking_names: tuple[str, ...],
    side_prob: np.ndarray | None = None,
    court_prob: np.ndarray | None = None,
) -> tuple[float, dict[str, float], float]:
    h, w = image_shape
    pooled = np.stack([_max_filter(evidence[k], config.score_radius_px) for k in range(evidence.shape[0])])
    markings = court_markings_world(n=config.n_template_samples)
    per_geometry: dict[str, float] = {}
    visible_weight = 0.0
    total_weight = 0.0
    weighted_evidence = 0.0
    for geometry_name, world_xyz in markings.items():
        class_name = MARKING_CLASS_BY_GEOMETRY[geometry_name]
        if class_name not in marking_names:
            continue
        class_id = marking_names.index(class_name)
        world_xy = world_xyz[:, :2]
        sample_weights = _polyline_sample_weights(world_xy)
        geometry_total_weight = float(sample_weights.sum())
        total_weight += geometry_total_weight
        projected = project_world_points(H, world_xy)
        finite = np.isfinite(projected).all(axis=1)
        inside = (
            finite
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] <= w - 1)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] <= h - 1)
        )
        if not inside.any():
            per_geometry[geometry_name] = 0.0
            continue
        weights = sample_weights[inside]
        geometry_visible_weight = float(weights.sum())
        if geometry_visible_weight <= 0.0:
            per_geometry[geometry_name] = 0.0
            continue
        x = np.round(projected[inside, 0]).astype(int).clip(0, w - 1)
        y = np.round(projected[inside, 1]).astype(int).clip(0, h - 1)
        values = pooled[class_id, y, x]
        side_name = _geometry_side(geometry_name)
        if side_name is not None and side_prob is not None:
            side_values = side_prob[y, x] if side_name == "right" else 1.0 - side_prob[y, x]
            values = values * np.clip(side_values, 0.0, 1.0)
        if court_prob is not None:
            values = values * np.clip(court_prob[y, x], 0.0, 1.0)
        geometry_evidence = float(np.sum(weights * values))
        visible_weight += geometry_visible_weight
        weighted_evidence += geometry_evidence
        per_geometry[geometry_name] = float(geometry_evidence / geometry_visible_weight)
    if not per_geometry:
        return -np.inf, {}, 0.0
    coverage = float(visible_weight / total_weight) if total_weight > 0.0 else 0.0
    visible_mean = float(weighted_evidence / visible_weight) if visible_weight > 0.0 else 0.0
    score = float(np.clip(visible_mean, 0.0, 1.0) * coverage)
    return score, per_geometry, coverage


def _cpu_ransac_homography(
    geometry_candidates: dict[str, list[MarkingLineFit]],
    curve_fits: dict[str, list[MarkingCurveFit]],
    evidence: np.ndarray,
    side_prob: np.ndarray | None,
    court_prob: np.ndarray | None,
    image_shape: tuple[int, int],
    config: MarkingRefinementConfig,
    marking_names: tuple[str, ...],
    rng: np.random.Generator,
) -> (
    tuple[np.ndarray, float, dict[str, MarkingLineFit], list[tuple[str, np.ndarray, np.ndarray]], dict[str, float], str]
    | None
):
    best_H: np.ndarray | None = None
    best_score = -np.inf
    best_selected: dict[str, MarkingLineFit] = {}
    best_point_corrs: list[tuple[str, np.ndarray, np.ndarray]] = []
    best_per_geometry: dict[str, float] = {}
    del rng
    line_choices = {
        name: _top_fit_candidates(fits, config)
        for name, fits in geometry_candidates.items()
        if fits
    }
    curve_choices = _geometry_curve_candidates(curve_fits, config)
    primitive_names = tuple(line_choices) + tuple(curve_choices)
    max_excluded = min(max(0, int(config.max_excluded_primitives)), len(primitive_names))
    attempts = 0

    for excluded_count in range(max_excluded + 1):
        for excluded_tuple in combinations(primitive_names, excluded_count):
            excluded = set(excluded_tuple)
            active_line_names = [name for name in line_choices if name not in excluded]
            active_curve_names = [name for name in curve_choices if name not in excluded]
            if not active_line_names:
                continue
            for line_combo in product(*(line_choices[name] for name in active_line_names)):
                selected = dict(zip(active_line_names, line_combo, strict=True))
                curve_products = product(*(curve_choices[name] for name in active_curve_names)) if active_curve_names else [()]
                for curve_combo in curve_products:
                    selected_curves = _selected_curve_fit_dict(active_curve_names, curve_combo, curve_fits)
                    line_corrs = [(name, _world_line(name), fit.line_homog) for name, fit in selected.items()]
                    point_corrs = _all_point_correspondences(selected, selected_curves, image_shape)
                    if 2 * len(line_corrs) + 2 * len(point_corrs) < 8:
                        continue
                    attempts += 1
                    try:
                        H = line_point_dlt(
                            line_corrs,
                            point_corrs,
                            point_weights=_dlt_point_weights(point_corrs, config),
                        )
                    except (ValueError, np.linalg.LinAlgError):
                        continue
                    if not _homography_is_sane(H, image_shape):
                        continue
                    refit_point_corrs = _all_point_correspondences(selected, selected_curves, image_shape, H=H)
                    H_refit = _refit_linear_homography(selected, refit_point_corrs, H, config)
                    if H_refit is not None and _homography_is_sane(H_refit, image_shape):
                        H = H_refit
                        point_corrs = _all_point_correspondences(selected, selected_curves, image_shape, H=H)
                    score, per_geometry, coverage = score_homography_template(
                        H, evidence, image_shape, config, marking_names, side_prob, court_prob
                    )
                    if coverage < config.min_projected_coverage or not _passes_structure(per_geometry, config):
                        continue
                    if score > best_score:
                        best_H = H
                        best_score = score
                        best_selected = selected
                        best_point_corrs = point_corrs
                        best_per_geometry = per_geometry
    if best_H is None:
        return None
    return (
        best_H,
        float(best_score),
        best_selected,
        best_point_corrs,
        best_per_geometry,
        f"ok (cpu systematic dlt, attempts={attempts}, max_excluded={max_excluded})",
    )


def _top_fit_candidates(
    fits: list[MarkingLineFit] | list[MarkingCurveFit],
    config: MarkingRefinementConfig,
) -> list[MarkingLineFit] | list[MarkingCurveFit]:
    limit = max(1, int(config.max_dlt_candidates_per_geometry))
    return sorted(fits, key=lambda fit: float(fit.score), reverse=True)[:limit]


def _geometry_curve_candidates(
    curve_fits: dict[str, list[MarkingCurveFit]],
    config: MarkingRefinementConfig,
) -> dict[str, list[MarkingCurveFit]]:
    by_geometry: dict[str, list[MarkingCurveFit]] = {}
    for fits in curve_fits.values():
        for fit in fits:
            if fit.geometry_name is not None:
                by_geometry.setdefault(fit.geometry_name, []).append(fit)
    return {name: _top_fit_candidates(fits, config) for name, fits in by_geometry.items()}


def _selected_curve_fit_dict(
    active_curve_names: list[str],
    curve_combo: tuple[MarkingCurveFit, ...],
    curve_fits: dict[str, list[MarkingCurveFit]],
) -> dict[str, list[MarkingCurveFit]]:
    selected = {class_name: [] for class_name in curve_fits}
    for geometry_name, fit in zip(active_curve_names, curve_combo, strict=True):
        class_name = MARKING_CLASS_BY_GEOMETRY[geometry_name]
        selected.setdefault(class_name, []).append(fit)
    return selected


def _systematic_primitive_attempt_indices(
    line_choices: dict[str, list[MarkingLineFit]],
    curve_choices: dict[str, list[MarkingCurveFit]],
    config: MarkingRefinementConfig,
) -> tuple[
    list[str],
    list[str],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    int,
] | None:
    line_names = list(line_choices)
    curve_names = list(curve_choices)
    if not line_names:
        return None

    primitive_names = tuple(line_names) + tuple(curve_names)
    max_excluded = min(max(0, int(config.max_excluded_primitives)), len(primitive_names))
    line_indices: dict[str, list[int]] = {name: [] for name in line_names}
    line_active: dict[str, list[bool]] = {name: [] for name in line_names}
    curve_indices: dict[str, list[int]] = {name: [] for name in curve_names}
    curve_active: dict[str, list[bool]] = {name: [] for name in curve_names}

    for excluded_count in range(max_excluded + 1):
        for excluded_tuple in combinations(primitive_names, excluded_count):
            excluded = set(excluded_tuple)
            active_line_names = [name for name in line_names if name not in excluded]
            active_curve_names = [name for name in curve_names if name not in excluded]
            if not active_line_names:
                continue
            line_products = product(*(range(len(line_choices[name])) for name in active_line_names))
            for line_combo in line_products:
                curve_products = product(*(range(len(curve_choices[name])) for name in active_curve_names)) if active_curve_names else [()]
                for curve_combo in curve_products:
                    line_combo_by_name = dict(zip(active_line_names, line_combo, strict=True))
                    curve_combo_by_name = dict(zip(active_curve_names, curve_combo, strict=True))
                    for name in line_names:
                        is_active = name in line_combo_by_name
                        line_active[name].append(is_active)
                        line_indices[name].append(int(line_combo_by_name.get(name, 0)))
                    for name in curve_names:
                        is_active = name in curve_combo_by_name
                        curve_active[name].append(is_active)
                        curve_indices[name].append(int(curve_combo_by_name.get(name, 0)))

    if not line_names or not line_indices[line_names[0]]:
        return None
    return (
        line_names,
        curve_names,
        {name: np.asarray(values, dtype=np.int64) for name, values in line_indices.items()},
        {name: np.asarray(values, dtype=bool) for name, values in line_active.items()},
        {name: np.asarray(values, dtype=np.int64) for name, values in curve_indices.items()},
        {name: np.asarray(values, dtype=bool) for name, values in curve_active.items()},
        max_excluded,
    )


def _sample_curve_subset(
    curve_fits: dict[str, list[MarkingCurveFit]],
    rng: np.random.Generator,
    keep_probability: float,
) -> dict[str, list[MarkingCurveFit]]:
    out: dict[str, list[MarkingCurveFit]] = {class_name: [] for class_name in curve_fits}
    for class_name, fits in curve_fits.items():
        by_geometry: dict[str, list[MarkingCurveFit]] = {}
        for fit in fits:
            if fit.geometry_name is not None:
                by_geometry.setdefault(fit.geometry_name, []).append(fit)
        for geometry_name, candidates in by_geometry.items():
            if rng.random() <= keep_probability:
                out.setdefault(class_name, []).append(rng.choice(candidates))
    return out


def _torch_ransac_homography(
    geometry_candidates: dict[str, list[MarkingLineFit]],
    curve_fits: dict[str, list[MarkingCurveFit]],
    evidence: np.ndarray,
    side_prob: np.ndarray | None,
    court_prob: np.ndarray | None,
    image_shape: tuple[int, int],
    config: MarkingRefinementConfig,
    marking_names: tuple[str, ...],
) -> tuple[np.ndarray, float, dict[str, MarkingLineFit], list[tuple[str, np.ndarray, np.ndarray]], dict[str, float], str] | None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    line_choices = {
        name: _top_fit_candidates(fits, config)
        for name, fits in geometry_candidates.items()
        if fits
    }
    curve_choices = _geometry_curve_candidates(curve_fits, config)
    attempt_indices = _systematic_primitive_attempt_indices(line_choices, curve_choices, config)
    if attempt_indices is None:
        return None
    line_names, curve_names, line_index_np, line_active_np, curve_index_np, curve_active_np, max_excluded = attempt_indices
    batch = len(next(iter(line_index_np.values()))) if line_index_np else 0
    if batch <= 0:
        return None
    Tw, Ti = _torch_fixed_dlt_transforms(image_shape, device, dtype)
    Tw_inv = torch.linalg.inv(Tw)
    Ti_inv = torch.linalg.inv(Ti)
    Tw_line = Tw_inv.T
    Ti_line = Ti_inv.T

    line_tensors: dict[str, torch.Tensor] = {}
    line_indices: dict[str, torch.Tensor] = {}
    line_active: dict[str, torch.Tensor] = {}
    for name in line_names:
        candidates = line_choices[name]
        lines = torch.as_tensor(np.stack([fit.line_homog for fit in candidates]), device=device, dtype=dtype)
        idx = torch.as_tensor(line_index_np[name], device=device, dtype=torch.long)
        active = torch.as_tensor(line_active_np[name], device=device, dtype=torch.bool)
        line_tensors[name] = lines[idx]
        line_indices[name] = idx
        line_active[name] = active

    curve_tensors: dict[str, torch.Tensor] = {}
    curve_indices: dict[str, torch.Tensor] = {}
    curve_active: dict[str, torch.Tensor] = {}
    for name in curve_names:
        candidates = curve_choices[name]
        conics = torch.as_tensor(np.stack([fit.conic for fit in candidates]), device=device, dtype=dtype)
        idx = torch.as_tensor(curve_index_np[name], device=device, dtype=torch.long)
        active = torch.as_tensor(curve_active_np[name], device=device, dtype=torch.bool)
        curve_tensors[name] = conics[idx]
        curve_indices[name] = idx
        curve_active[name] = active

    A_parts: list[torch.Tensor] = []
    row_valid_parts: list[torch.Tensor] = []
    row_weight_parts: list[torch.Tensor] = []
    for name in line_names:
        world_line = Tw_line @ torch.as_tensor(_world_line(name), device=device, dtype=dtype)
        image_line = torch.einsum("ij,bj->bi", Ti_line, line_tensors[name])
        rows = _torch_line_rows(_torch_normalize_single_line(world_line), _torch_normalize_lines(image_line))
        valid = line_active[name][:, None].expand(batch, 2)
        A_parts.append(torch.where(valid[:, :, None], rows, torch.zeros_like(rows)))
        row_valid_parts.append(valid)
        row_weight_parts.append(torch.ones((batch, 2), device=device, dtype=dtype))

    for i, a_name in enumerate(line_names):
        for b_name in line_names[i + 1:]:
            world = _line_intersection(_world_line(a_name), _world_line(b_name))
            if world is None:
                continue
            if not (-50.0 <= world[0] <= COURT_LENGTH_CM + 50.0 and -50.0 <= world[1] <= COURT_WIDTH_CM + 50.0):
                continue
            image, valid_intersection = _torch_line_intersection(line_tensors[a_name], line_tensors[b_name], image_shape)
            valid = valid_intersection & line_active[a_name] & line_active[b_name]
            rows = _torch_point_rows(
                _torch_transform_point(Tw, torch.as_tensor(world, device=device, dtype=dtype)),
                _torch_transform_points(Ti, image),
            )
            A_parts.append(torch.where(valid[:, None, None], rows, torch.zeros_like(rows)))
            row_valid_parts.append(valid[:, None].expand(batch, 2))
            row_weight_parts.append(torch.ones((batch, 2), device=device, dtype=dtype))

    for _, curve_name, line_name, world_points in _conic_line_intersection_specs():
        if curve_name not in curve_tensors or line_name not in line_tensors:
            continue
        image_points, valid_points = _torch_conic_line_intersections(curve_tensors[curve_name], line_tensors[line_name], image_shape)
        active = curve_active[curve_name] & line_active[line_name]
        for point_idx in range(2):
            valid = valid_points & active
            rows = _torch_point_rows(
                _torch_transform_point(Tw, torch.as_tensor(world_points[point_idx], device=device, dtype=dtype)),
                _torch_transform_points(Ti, image_points[:, point_idx]),
            )
            A_parts.append(torch.where(valid[:, None, None], rows, torch.zeros_like(rows)))
            row_valid_parts.append(valid[:, None].expand(batch, 2))
            row_weight_parts.append(torch.full((batch, 2), _torch_dlt_curve_row_weight(config), device=device, dtype=dtype))

    line_score_tensors: dict[str, torch.Tensor] = {}
    for name in line_names:
        candidates = line_choices[name]
        scores = torch.as_tensor([fit.score for fit in candidates], device=device, dtype=dtype)
        line_score_tensors[name] = scores[line_indices[name]]

    for _, curve_name, line_name, world_point in _conic_tangent_point_specs():
        guide_names = [name for name in _tangent_guide_line_names(line_name) if name in line_tensors]
        if curve_name not in curve_tensors or not guide_names:
            continue
        guide_score = torch.stack(
            [
                torch.where(line_active[name], line_score_tensors[name], torch.full((batch,), -torch.inf, device=device, dtype=dtype))
                for name in guide_names
            ],
            dim=1,
        )
        best_guide_idx = guide_score.argmax(dim=1)
        best_guide_score = guide_score.gather(1, best_guide_idx[:, None])[:, 0]
        guide_lines = torch.stack([line_tensors[name] for name in guide_names], dim=1)
        chosen_line = guide_lines[torch.arange(batch, device=device), best_guide_idx]
        line_direction = torch.stack([-chosen_line[:, 1], chosen_line[:, 0]], dim=1)
        image_points, valid_points = _torch_conic_tangent_points(curve_tensors[curve_name], line_direction, image_shape)
        distances = (image_points @ chosen_line[:, :2, None]).squeeze(2) + chosen_line[:, 2:3]
        distances = distances.abs()
        prefer_farthest = torch.as_tensor(
            [_tangent_prefers_farthest(curve_name, name) for name in guide_names],
            device=device,
            dtype=torch.bool,
        )[best_guide_idx]
        choose_second = torch.where(
            prefer_farthest,
            distances[:, 1] > distances[:, 0],
            distances[:, 1] < distances[:, 0],
        )[:, None]
        image_point = torch.where(choose_second, image_points[:, 1], image_points[:, 0])
        valid = valid_points & curve_active[curve_name] & torch.isfinite(best_guide_score)
        rows = _torch_point_rows(
            _torch_transform_point(Tw, torch.as_tensor(world_point, device=device, dtype=dtype)),
            _torch_transform_points(Ti, image_point),
        )
        A_parts.append(torch.where(valid[:, None, None], rows, torch.zeros_like(rows)))
        row_valid_parts.append(valid[:, None].expand(batch, 2))
        row_weight_parts.append(torch.full((batch, 2), _torch_dlt_curve_row_weight(config), device=device, dtype=dtype))

    if not A_parts:
        return None
    A = torch.cat(A_parts, dim=1)
    row_valid = torch.cat(row_valid_parts, dim=1)
    row_weights = torch.cat(row_weight_parts, dim=1)
    row_count = row_valid.sum(dim=1)
    A = _normalize_torch_dlt_rows(A) * row_weights[:, :, None]
    try:
        _, _, vh = torch.linalg.svd(A)
    except RuntimeError:
        return None
    H = vh[:, -1].reshape(-1, 3, 3)
    H = Ti_inv[None] @ H @ Tw[None]
    H = H / torch.where(H[:, 2:3, 2:3].abs() > 1e-12, H[:, 2:3, 2:3], torch.ones_like(H[:, 2:3, 2:3]))

    evidence_t = torch.as_tensor(evidence, device=device, dtype=dtype)
    side_t = None if side_prob is None else torch.as_tensor(side_prob, device=device, dtype=dtype)
    court_t = None if court_prob is None else torch.as_tensor(court_prob, device=device, dtype=dtype)
    score, per_geometry_t, coverage = _torch_score_marking_homographies(
        H, evidence_t, side_t, court_t, image_shape, config, marking_names
    )
    sane = _torch_homography_sane(H, image_shape)
    per_stack = torch.stack(list(per_geometry_t.values()), dim=1) if per_geometry_t else torch.empty((batch, 0), device=device)
    inlier_count = (per_stack >= config.inlier_score_threshold).sum(dim=1) if per_stack.numel() else torch.zeros((batch,), device=device, dtype=torch.long)
    valid = (row_count >= 8) & sane & (coverage >= config.min_projected_coverage) & (inlier_count >= config.min_scored_markings)
    if config.require_baseline:
        baseline_ok = torch.zeros((batch,), device=device, dtype=torch.bool)
        for name in ("baseline_left", "baseline_right"):
            if name in per_geometry_t:
                baseline_ok = baseline_ok | (per_geometry_t[name] >= config.inlier_score_threshold)
        valid = valid & baseline_ok
    masked_score = torch.where(valid, score, torch.full_like(score, -torch.inf))
    best_score_t, best_idx_t = masked_score.max(dim=0)
    if not torch.isfinite(best_score_t):
        return None

    best_idx = int(best_idx_t.detach().cpu())
    H_seed = H[best_idx].detach().cpu().numpy()
    selected_lines = {
        name: line_choices[name][int(line_indices[name][best_idx].detach().cpu())]
        for name in line_names
        if bool(line_active[name][best_idx].detach().cpu())
    }
    selected_curves: dict[str, list[MarkingCurveFit]] = {class_name: [] for class_name in curve_fits}
    for name in curve_names:
        if not bool(curve_active[name][best_idx].detach().cpu()):
            continue
        class_name = MARKING_CLASS_BY_GEOMETRY[name]
        candidates = curve_choices[name]
        selected_curves.setdefault(class_name, []).append(candidates[int(curve_indices[name][best_idx].detach().cpu())])
    point_corrs_seed = _all_point_correspondences(selected_lines, selected_curves, image_shape, H=H_seed)
    H_refit = _refit_linear_homography(selected_lines, point_corrs_seed, H_seed, config)
    if H_refit is not None and _homography_is_sane(H_refit, image_shape):
        H_seed = H_refit
    point_corrs = _all_point_correspondences(selected_lines, selected_curves, image_shape, H=H_seed)
    final_score, final_per_geometry, coverage_np = score_homography_template(
        H_seed, evidence, image_shape, config, marking_names, side_prob, court_prob
    )
    if coverage_np < config.min_projected_coverage or not _passes_structure(final_per_geometry, config):
        final_score = float(best_score_t.detach().cpu())
        final_per_geometry = {name: float(values[best_idx].detach().cpu()) for name, values in per_geometry_t.items()}
    return (
        H_seed,
        float(final_score),
        selected_lines,
        point_corrs,
        final_per_geometry,
        f"ok (torch {device.type} systematic dlt, attempts={batch}, max_excluded={max_excluded})",
    )


def _refit_linear_homography(
    selected_lines: dict[str, MarkingLineFit],
    point_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    fallback_H: np.ndarray,
    config: MarkingRefinementConfig,
) -> np.ndarray | None:
    line_corrs = [(name, _world_line(name), fit.line_homog) for name, fit in selected_lines.items()]
    if 2 * len(line_corrs) + 2 * len(point_corrs) < 8:
        return fallback_H
    try:
        return line_point_dlt(
            line_corrs,
            point_corrs,
            point_weights=_dlt_point_weights(point_corrs, config),
        )
    except (ValueError, np.linalg.LinAlgError):
        return fallback_H


def _dlt_point_weights(
    point_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    config: MarkingRefinementConfig,
) -> dict[str, float]:
    curve_weight = _dlt_curve_point_weight(config)
    return {
        name: curve_weight
        for name, _, _ in point_corrs
        if name.startswith("three_point_arc_") or name.startswith("free_throw_circle_")
    }


def _dlt_curve_point_weight(config: MarkingRefinementConfig) -> float:
    return max(1.0, float(config.dlt_curve_point_weight))


def _torch_dlt_curve_row_weight(config: MarkingRefinementConfig) -> float:
    return float(np.sqrt(_dlt_curve_point_weight(config)))


def _torch_line_rows(lw: torch.Tensor, li: torch.Tensor) -> torch.Tensor:
    b = li.shape[0]
    rows = torch.zeros((b, 2, 9), device=li.device, dtype=li.dtype)

    def coeff(col: int) -> torch.Tensor:
        out = torch.zeros((b, 9), device=li.device, dtype=li.dtype)
        out[:, col] = li[:, 0]
        out[:, 3 + col] = li[:, 1]
        out[:, 6 + col] = li[:, 2]
        return out

    v0 = coeff(0)
    v1 = coeff(1)
    v2 = coeff(2)
    rows[:, 0] = lw[1] * v0 - lw[0] * v1
    rows[:, 1] = lw[2] * v0 - lw[0] * v2
    return rows


def _torch_fixed_dlt_transforms(
    image_shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image_shape
    world_center = torch.tensor([COURT_LENGTH_CM / 2.0, COURT_WIDTH_CM / 2.0], device=device, dtype=dtype)
    world_corners = torch.tensor(
        [[0.0, 0.0], [COURT_LENGTH_CM, 0.0], [COURT_LENGTH_CM, COURT_WIDTH_CM], [0.0, COURT_WIDTH_CM]],
        device=device,
        dtype=dtype,
    )
    world_scale = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype)) / torch.linalg.norm(
        world_corners - world_center[None], dim=1
    ).mean().clamp_min(1e-12)
    image_center = torch.tensor([(w - 1) / 2.0, (h - 1) / 2.0], device=device, dtype=dtype)
    image_corners = torch.tensor(
        [[0.0, 0.0], [w - 1.0, 0.0], [w - 1.0, h - 1.0], [0.0, h - 1.0]],
        device=device,
        dtype=dtype,
    )
    image_scale = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype)) / torch.linalg.norm(
        image_corners - image_center[None], dim=1
    ).mean().clamp_min(1e-12)

    def transform(scale: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        T = torch.eye(3, device=device, dtype=dtype)
        T[0, 0] = scale
        T[1, 1] = scale
        T[0, 2] = -scale * center[0]
        T[1, 2] = -scale * center[1]
        return T

    return transform(world_scale, world_center), transform(image_scale, image_center)


def _torch_transform_point(T: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    homog = torch.cat([point, torch.ones((1,), device=point.device, dtype=point.dtype)])
    transformed = T @ homog
    return transformed[:2] / _torch_safe_denominator(transformed[2:3])


def _torch_transform_points(T: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    ones = torch.ones((*points.shape[:-1], 1), device=points.device, dtype=points.dtype)
    homog = torch.cat([points, ones], dim=-1)
    transformed = torch.einsum("ij,...j->...i", T, homog)
    return transformed[..., :2] / _torch_safe_denominator(transformed[..., 2:3])


def _torch_normalize_single_line(line: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(line[:2]).clamp_min(1e-12)
    out = line / norm
    sign = torch.where(out[2] < 0, -1.0, 1.0)
    return torch.nan_to_num(out * sign, nan=0.0, posinf=0.0, neginf=0.0)


def _torch_point_rows(world: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
    b = image.shape[0]
    X, Y = world[0], world[1]
    u, v = image[:, 0], image[:, 1]
    rows = torch.zeros((b, 2, 9), device=image.device, dtype=image.dtype)
    rows[:, 0, 0] = -X
    rows[:, 0, 1] = -Y
    rows[:, 0, 2] = -1.0
    rows[:, 0, 6] = u * X
    rows[:, 0, 7] = u * Y
    rows[:, 0, 8] = u
    rows[:, 1, 3] = -X
    rows[:, 1, 4] = -Y
    rows[:, 1, 5] = -1.0
    rows[:, 1, 6] = v * X
    rows[:, 1, 7] = v * Y
    rows[:, 1, 8] = v
    return rows


def _torch_line_intersection(
    a: torch.Tensor,
    b: torch.Tensor,
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    h, w = image_shape
    p = torch.cross(a, b, dim=1)
    valid = torch.isfinite(p).all(dim=1) & (p[:, 2].abs() > 1e-9)
    xy = p[:, :2] / _torch_safe_denominator(p[:, 2:3])
    valid = (
        valid
        & torch.isfinite(xy).all(dim=1)
        & (xy[:, 0] >= -w)
        & (xy[:, 0] <= 2 * w)
        & (xy[:, 1] >= -h)
        & (xy[:, 1] <= 2 * h)
    )
    return xy, valid


def _torch_conic_line_intersections(
    conic: torch.Tensor,
    line: torch.Tensor,
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    line = _torch_normalize_lines(line)
    point_on_line = -line[:, 2:3] * line[:, :2]
    direction = torch.stack([-line[:, 1], line[:, 0]], dim=1)
    p = torch.cat([point_on_line, torch.ones((line.shape[0], 1), device=line.device, dtype=line.dtype)], dim=1)
    d = torch.cat([direction, torch.zeros((line.shape[0], 1), device=line.device, dtype=line.dtype)], dim=1)
    a = torch.einsum("bi,bij,bj->b", d, conic, d)
    b = 2.0 * torch.einsum("bi,bij,bj->b", p, conic, d)
    c = torch.einsum("bi,bij,bj->b", p, conic, p)
    disc = b * b - 4.0 * a * c
    valid = torch.isfinite(disc) & (disc > 1e-9) & (a.abs() > 1e-12)
    root = torch.sqrt(torch.clamp(disc, min=1e-9))
    denom = _torch_safe_denominator(2.0 * a[:, None])
    t = torch.stack([-b - root, -b + root], dim=1) / denom
    points = point_on_line[:, None] + t[:, :, None] * direction[:, None]
    projection = ((points - point_on_line[:, None]) * direction[:, None]).sum(dim=2)
    order = projection.argsort(dim=1)
    points = torch.gather(points, 1, order[:, :, None].expand(-1, -1, 2))
    valid = valid & _torch_points_in_loose_bounds(points, image_shape)
    return points, valid


def _torch_conic_tangent_points(
    conic: torch.Tensor,
    tangent_direction: torch.Tensor,
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    norm = torch.linalg.norm(tangent_direction, dim=1, keepdim=True).clamp_min(1e-12)
    direction = tangent_direction / norm
    homog_direction = torch.cat(
        [direction, torch.zeros((direction.shape[0], 1), device=direction.device, dtype=direction.dtype)],
        dim=1,
    )
    line = torch.einsum("bij,bj->bi", conic, homog_direction)
    return _torch_conic_line_intersections(conic, line, image_shape)


def _torch_points_in_loose_bounds(points: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    h, w = image_shape
    flat = points.reshape(points.shape[0], -1, 2)
    return (
        torch.isfinite(flat).all(dim=(1, 2))
        & (flat[:, :, 0] >= -w).all(dim=1)
        & (flat[:, :, 0] <= 2 * w).all(dim=1)
        & (flat[:, :, 1] >= -h).all(dim=1)
        & (flat[:, :, 1] <= 2 * h).all(dim=1)
    )


def _normalize_torch_dlt_rows(rows: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(rows, dim=2, keepdim=True)
    rows = torch.where(norm > 1e-12, rows / norm.clamp_min(1e-12), torch.zeros_like(rows))
    return torch.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)


def _torch_normalize_lines(lines: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(lines[:, :2], dim=1, keepdim=True).clamp_min(1e-12)
    out = lines / norm
    sign = torch.where(out[:, 2:3] < 0, -1.0, 1.0)
    return torch.nan_to_num(out * sign, nan=0.0, posinf=0.0, neginf=0.0)


def _torch_homography_sane(H: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    finite = torch.isfinite(H).flatten(1).all(dim=1)
    det_ok = torch.linalg.det(H).abs() > 1e-10
    corners = torch.as_tensor(
        [[0.0, 0.0], [COURT_LENGTH_CM, 0.0], [COURT_LENGTH_CM, COURT_WIDTH_CM], [0.0, COURT_WIDTH_CM]],
        device=H.device,
        dtype=H.dtype,
    )
    proj = _torch_project(H, corners)
    proj_ok = torch.isfinite(proj).flatten(1).all(dim=1)
    x = proj[:, :, 0]
    y = proj[:, :, 1]
    area = 0.5 * torch.abs((x * torch.roll(y, -1, 1)).sum(dim=1) - (y * torch.roll(x, -1, 1)).sum(dim=1))
    limit = 8.0 * max(image_shape)
    bounds_ok = proj.abs().flatten(1).max(dim=1).values <= limit
    return finite & det_ok & proj_ok & (area >= 100.0) & bounds_ok


def _torch_score_marking_homographies(
    H: torch.Tensor,
    evidence: torch.Tensor,
    side_prob: torch.Tensor | None,
    court_prob: torch.Tensor | None,
    image_shape: tuple[int, int],
    config: MarkingRefinementConfig,
    marking_names: tuple[str, ...],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    h, w = image_shape
    r = int(config.score_radius_px)
    pooled = F.max_pool2d(evidence[:, None], kernel_size=2 * r + 1, stride=1, padding=r)[:, 0]
    per_geometry: dict[str, torch.Tensor] = {}
    visible_weight = torch.zeros((H.shape[0],), device=H.device, dtype=H.dtype)
    total_weight = torch.zeros((H.shape[0],), device=H.device, dtype=H.dtype)
    weighted_evidence = torch.zeros((H.shape[0],), device=H.device, dtype=H.dtype)
    for geometry_name, world_xyz in court_markings_world(n=config.n_template_samples).items():
        class_name = MARKING_CLASS_BY_GEOMETRY[geometry_name]
        if class_name not in marking_names:
            continue
        class_id = marking_names.index(class_name)
        world_xy = world_xyz[:, :2]
        sample_weights = torch.as_tensor(_polyline_sample_weights(world_xy), device=H.device, dtype=H.dtype)
        geometry_total_weight = sample_weights.sum()
        total_weight = total_weight + geometry_total_weight
        world = torch.as_tensor(world_xy, device=H.device, dtype=H.dtype)
        projected = _torch_project(H, world)
        inside = (
            torch.isfinite(projected).all(dim=2)
            & (projected[:, :, 0] >= 0.0)
            & (projected[:, :, 0] <= w - 1)
            & (projected[:, :, 1] >= 0.0)
            & (projected[:, :, 1] <= h - 1)
        )
        x = torch.round(projected[:, :, 0]).long().clamp(0, w - 1)
        y = torch.round(projected[:, :, 1]).long().clamp(0, h - 1)
        values = pooled[class_id, y, x]
        side_name = _geometry_side(geometry_name)
        if side_name is not None and side_prob is not None:
            side_values = side_prob[y, x] if side_name == "right" else 1.0 - side_prob[y, x]
            values = values * torch.clamp(side_values, 0.0, 1.0)
        if court_prob is not None:
            values = values * torch.clamp(court_prob[y, x], 0.0, 1.0)
        weights = sample_weights[None, :] * inside.to(H.dtype)
        geometry_visible_weight = weights.sum(dim=1)
        geometry_evidence = (weights * values).sum(dim=1)
        visible_weight = visible_weight + geometry_visible_weight
        weighted_evidence = weighted_evidence + geometry_evidence
        per_geometry[geometry_name] = torch.where(
            geometry_visible_weight > 0.0,
            geometry_evidence / geometry_visible_weight.clamp_min(1e-12),
            torch.zeros_like(geometry_visible_weight),
        )
    if not per_geometry:
        score = torch.full((H.shape[0],), -torch.inf, device=H.device, dtype=H.dtype)
        return score, {}, torch.zeros_like(score)
    coverage = torch.where(
        total_weight > 0.0,
        visible_weight / total_weight.clamp_min(1e-12),
        torch.zeros_like(visible_weight),
    )
    visible_mean = torch.where(
        visible_weight > 0.0,
        weighted_evidence / visible_weight.clamp_min(1e-12),
        torch.zeros_like(visible_weight),
    )
    score = torch.clamp(visible_mean, 0.0, 1.0) * coverage
    return score, per_geometry, coverage


def _torch_project(H: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
    homog = torch.cat([world, torch.ones((world.shape[0], 1), device=world.device, dtype=world.dtype)], dim=1)
    img_h = torch.einsum("bij,nj->bni", H, homog)
    return img_h[:, :, :2] / _torch_safe_denominator(img_h[:, :, 2:3])


def _torch_safe_denominator(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    sign = torch.where(value < 0, -1.0, 1.0)
    return torch.where(value.abs() > eps, value, sign * eps)


def refine_homography_with_primitives(
    H0: np.ndarray,
    selected_lines: dict[str, MarkingLineFit],
    curve_fits: dict[str, list[MarkingCurveFit]],
    config: MarkingRefinementConfig,
    point_corrs: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> np.ndarray | None:
    point_corrs = [] if point_corrs is None else point_corrs
    curve_geometries = [
        name
        for name in CURVE_GEOMETRY_NAMES
        if any(fit.geometry_name == name for fit in curve_fits.get(MARKING_CLASS_BY_GEOMETRY[name], []))
    ]
    if not selected_lines and not curve_geometries and not point_corrs:
        return None

    def pack(H: np.ndarray) -> np.ndarray:
        H = H / H[2, 2]
        return H.reshape(-1)[:8]

    def unpack(p: np.ndarray) -> np.ndarray:
        return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]], dtype=float)

    def residuals(p: np.ndarray) -> np.ndarray:
        H = unpack(p)
        out = []
        for _, world, image in point_corrs:
            projected = project_world_points(H, world[None])[0]
            out.extend(config.point_refine_weight * np.clip((projected - image) / 12.0, -5.0, 5.0))
        for geometry_name, fit in selected_lines.items():
            world = court_markings_world(n=32)[geometry_name][:, :2]
            proj = project_world_points(H, world)
            out.extend(np.clip(_line_distances(fit.line_homog, proj) / 12.0, -5.0, 5.0))
        for geometry_name in curve_geometries:
            world = court_markings_world(n=48)[geometry_name][:, :2]
            proj = project_world_points(H, world)
            class_name = MARKING_CLASS_BY_GEOMETRY[geometry_name]
            candidates = [fit for fit in curve_fits[class_name] if fit.geometry_name == geometry_name]
            best = min(
                candidates,
                key=lambda fit: float(np.nanmean(np.abs(_conic_geometric_residual(fit.conic, proj)))),
            )
            out.extend(config.curve_refine_weight * np.clip(_conic_geometric_residual(best.conic, proj) / 12.0, -5.0, 5.0))
        arr = np.asarray(out, dtype=float)
        return arr[np.isfinite(arr)]

    try:
        result = least_squares(
            residuals,
            pack(H0),
            loss="huber",
            f_scale=1.0,
            max_nfev=config.max_refine_nfev,
        )
    except (ValueError, np.linalg.LinAlgError):
        return None
    return unpack(result.x) if result.success else None


def _connected_components(
    mask: np.ndarray,
    weights: np.ndarray,
    config: MarkingRefinementConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    comps = []
    for label in range(1, num):
        if stats[label, cv2.CC_STAT_AREA] < config.min_component_pixels:
            continue
        ys, xs = np.nonzero(labels == label)
        pts = np.column_stack([xs, ys]).astype(float)
        ws = weights[ys, xs].astype(float)
        comps.append((float(ws.sum()), pts, ws))
    comps.sort(key=lambda item: item[0], reverse=True)
    return [(pts, ws) for _, pts, ws in comps[: config.max_components_per_class]]


def _significant_components(
    components: list[tuple[np.ndarray, np.ndarray]],
    geometry_mass: float,
    config: MarkingRefinementConfig,
) -> list[tuple[np.ndarray, np.ndarray]]:
    threshold = max(
        float(config.min_component_evidence_mass),
        float(config.min_component_evidence_fraction) * float(geometry_mass),
    )
    return [(pts, weights) for pts, weights in components if float(weights.sum()) >= threshold]


def _component_geometry_names(
    class_name: str,
    points: np.ndarray,
    weights: np.ndarray,
    side_prob: np.ndarray | None,
    config: MarkingRefinementConfig,
) -> tuple[str, ...]:
    geometries = GEOMETRIES_BY_CLASS.get(class_name, ())
    if len(geometries) <= 1:
        return geometries

    left = tuple(name for name in geometries if _geometry_side(name) == "left")
    right = tuple(name for name in geometries if _geometry_side(name) == "right")
    if not left and not right:
        return geometries
    if side_prob is None or len(points) == 0 or weights.sum() <= 0:
        return left + right

    h, w = side_prob.shape
    x = np.round(points[:, 0]).astype(int).clip(0, w - 1)
    y = np.round(points[:, 1]).astype(int).clip(0, h - 1)
    side_score = float(np.average(side_prob[y, x], weights=weights))
    if side_score >= 0.5 + config.side_margin:
        return right
    if side_score <= 0.5 - config.side_margin:
        return left
    return left + right


def _geometry_side_weight_map(
    geometry_name: str,
    side_prob: np.ndarray | None,
    shape: tuple[int, int],
    config: MarkingRefinementConfig,
) -> np.ndarray:
    side = _geometry_side(geometry_name)
    if side is None or side_prob is None:
        return np.ones(shape, dtype=np.float32)
    side_map = side_prob if side == "right" else 1.0 - side_prob
    margin = max(0.0, min(0.49, float(config.side_margin)))
    hard_side = side_map >= 0.5 + margin
    weights = np.where(hard_side, side_map, 0.0)
    return np.clip(weights, 0.0, 1.0).astype(np.float32)


def _minimum_geometry_mass(
    config: MarkingRefinementConfig,
    shape: tuple[int, int],
    geometry_name: str,
) -> float:
    h, w = shape
    scaled = config.min_geometry_evidence_mass_per_megapixel * (h * w / 1_000_000.0)
    minimum = max(config.min_geometry_evidence_mass, scaled)
    if geometry_name in CURVE_GEOMETRY_NAMES:
        minimum *= config.curve_geometry_mass_multiplier
    return float(minimum)


def _geometry_side(geometry_name: str) -> str | None:
    if "_left" in geometry_name or geometry_name.endswith("_left"):
        return "left"
    if "_right" in geometry_name or geometry_name.endswith("_right"):
        return "right"
    return None


def _fit_line_component(
    class_name: str,
    geometry_name: str,
    points: np.ndarray,
    weights: np.ndarray,
) -> MarkingLineFit | None:
    if len(points) < 2 or weights.sum() <= 0:
        return None
    line = _fit_line_tls(points, weights)
    p0, p1 = _clip_line_endpoints(line, points)
    return MarkingLineFit(class_name, geometry_name, line, p0, p1, int(len(points)), float(weights.sum()))


def _fit_curve_component(
    class_name: str,
    geometry_name: str,
    points: np.ndarray,
    weights: np.ndarray,
) -> MarkingCurveFit | None:
    if len(points) < 6 or weights.sum() <= 0:
        return None
    conic = _fit_conic(points, weights)
    center, radius = _fit_circle(points, weights)
    return MarkingCurveFit(class_name, geometry_name, conic, center, radius, points, int(len(points)), float(weights.sum()))


def _geometry_line_candidates(line_fits: dict[str, list[MarkingLineFit]]) -> dict[str, list[MarkingLineFit]]:
    out: dict[str, list[MarkingLineFit]] = {}
    for geometry_name in STRAIGHT_LINE_NAMES:
        class_name = MARKING_CLASS_BY_GEOMETRY[geometry_name]
        out[geometry_name] = [f for f in line_fits.get(class_name, []) if f.geometry_name == geometry_name]
    return out


def _has_enough_initializer_potential(
    geometry_candidates: dict[str, list[MarkingLineFit]],
    curve_fits: dict[str, list[MarkingCurveFit]],
) -> bool:
    present_lines = [name for name, fits in geometry_candidates.items() if fits]
    point_count = 0
    for i, a_name in enumerate(present_lines):
        for b_name in present_lines[i + 1:]:
            world = _line_intersection(_world_line(a_name), _world_line(b_name))
            if world is not None and -50.0 <= world[0] <= COURT_LENGTH_CM + 50.0 and -50.0 <= world[1] <= COURT_WIDTH_CM + 50.0:
                point_count += 1
    for _, curve_name, line_name, _ in _conic_line_intersection_specs():
        class_name = MARKING_CLASS_BY_GEOMETRY[curve_name]
        if geometry_candidates.get(line_name) and any(
            fit.geometry_name == curve_name for fit in curve_fits.get(class_name, [])
        ):
            point_count += 2
    for _, curve_name, line_name, _ in _conic_tangent_point_specs():
        class_name = MARKING_CLASS_BY_GEOMETRY[curve_name]
        has_guide_line = any(geometry_candidates.get(guide_name) for guide_name in _tangent_guide_line_names(line_name))
        if has_guide_line and any(
            fit.geometry_name == curve_name for fit in curve_fits.get(class_name, [])
        ):
            point_count += 1
    return 2 * len(present_lines) + 2 * point_count >= 8


def _select_line_fits_for_homography(
    H: np.ndarray,
    geometry_candidates: dict[str, list[MarkingLineFit]],
) -> dict[str, MarkingLineFit]:
    selected = {}
    markings = court_markings_world(n=32)
    for geometry_name, candidates in geometry_candidates.items():
        if not candidates:
            continue
        projected = project_world_points(H, markings[geometry_name][:, :2])
        selected[geometry_name] = min(candidates, key=lambda fit: float(np.nanmean(_line_distances(fit.line_homog, projected))))
    return selected


def _intersection_point_correspondences(
    selected: dict[str, MarkingLineFit],
    image_shape: tuple[int, int],
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    h, w = image_shape
    names = list(selected)
    corrs = []
    for i, a_name in enumerate(names):
        for b_name in names[i + 1:]:
            world = _line_intersection(_world_line(a_name), _world_line(b_name))
            if world is None:
                continue
            if not (-50.0 <= world[0] <= COURT_LENGTH_CM + 50.0 and -50.0 <= world[1] <= COURT_WIDTH_CM + 50.0):
                continue
            image = _line_intersection(selected[a_name].line_homog, selected[b_name].line_homog)
            if image is None:
                continue
            if not (-w <= image[0] <= 2 * w and -h <= image[1] <= 2 * h):
                continue
            corrs.append((f"{a_name}__{b_name}", world, image))
    return corrs


def _all_point_correspondences(
    selected: dict[str, MarkingLineFit],
    curve_fits: dict[str, list[MarkingCurveFit]],
    image_shape: tuple[int, int],
    rng: np.random.Generator | None = None,
    H: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    return (
        _intersection_point_correspondences(selected, image_shape)
        + _conic_line_point_correspondences(selected, curve_fits, image_shape, rng=rng, H=H)
        + _conic_tangent_point_correspondences(selected, curve_fits, image_shape, rng=rng, H=H)
    )


def _conic_line_point_correspondences(
    selected: dict[str, MarkingLineFit],
    curve_fits: dict[str, list[MarkingCurveFit]],
    image_shape: tuple[int, int],
    rng: np.random.Generator | None = None,
    H: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    h, w = image_shape
    corrs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for spec_name, curve_name, line_name, world_points in _conic_line_intersection_specs():
        line_fit = selected.get(line_name)
        if line_fit is None:
            continue
        class_name = MARKING_CLASS_BY_GEOMETRY[curve_name]
        candidates = [fit for fit in curve_fits.get(class_name, []) if fit.geometry_name == curve_name]
        if not candidates:
            continue

        options: list[tuple[float, np.ndarray]] = []
        for curve_fit in candidates:
            image_points = _conic_line_intersections(curve_fit.conic, line_fit.line_homog)
            if image_points is None:
                continue
            image_points = _sort_points_along_line(image_points, line_fit.line_homog)
            if not _points_in_loose_image_bounds(image_points, h, w):
                continue
            if H is None:
                options.append((-float(curve_fit.score), image_points))
                continue
            projected = project_world_points(H, world_points)
            options.append((_pairing_cost(projected, image_points), image_points))
            options.append((_pairing_cost(projected, image_points[::-1]), image_points[::-1]))

        if not options:
            continue
        options.sort(key=lambda item: item[0])
        image_points = options[0][1]
        if H is None and rng is not None and bool(rng.integers(0, 2)):
            image_points = image_points[::-1]
        for i, (world, image) in enumerate(zip(world_points, image_points, strict=True)):
            corrs.append((f"{spec_name}_{i}", world, image))
    return corrs


def _conic_line_intersection_specs() -> tuple[tuple[str, str, str, np.ndarray], ...]:
    mid_y = COURT_WIDTH_CM / 2.0
    baseline_circle_dx = BASKET_CENTER_FROM_ENDLINE_CM
    three_dy = float(np.sqrt(max(0.0, THREE_POINT_RADIUS_CM ** 2 - baseline_circle_dx ** 2)))
    three_y = np.array([mid_y - three_dy, mid_y + three_dy], dtype=float)
    return (
        (
            "three_point_arc_left__baseline_left",
            "three_point_arc_left",
            "baseline_left",
            np.column_stack([np.zeros(2, dtype=float), three_y]),
        ),
        (
            "three_point_arc_right__baseline_right",
            "three_point_arc_right",
            "baseline_right",
            np.column_stack([np.full(2, COURT_LENGTH_CM, dtype=float), three_y]),
        ),
    )


def _conic_tangent_point_correspondences(
    selected: dict[str, MarkingLineFit],
    curve_fits: dict[str, list[MarkingCurveFit]],
    image_shape: tuple[int, int],
    rng: np.random.Generator | None = None,
    H: np.ndarray | None = None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    h, w = image_shape
    corrs: list[tuple[str, np.ndarray, np.ndarray]] = []
    for spec_name, curve_name, line_name, world_point in _conic_tangent_point_specs():
        guide_name, line_fit = _select_tangent_guide_line(selected, line_name)
        if line_fit is None:
            continue
        class_name = MARKING_CLASS_BY_GEOMETRY[curve_name]
        candidates = [fit for fit in curve_fits.get(class_name, []) if fit.geometry_name == curve_name]
        if not candidates:
            continue

        line_direction = np.array([-line_fit.line_homog[1], line_fit.line_homog[0]], dtype=float)
        options: list[tuple[float, np.ndarray]] = []
        for curve_fit in candidates:
            image_points = _conic_tangent_points(curve_fit.conic, line_direction)
            if image_points is None:
                continue
            if not _points_in_loose_image_bounds(image_points, h, w):
                continue
            image_point = _select_tangent_point_for_guide(image_points, line_fit.line_homog, curve_name, guide_name)
            if H is None:
                options.append((-float(curve_fit.score), image_point))
                continue
            projected = project_world_points(H, world_point[None])[0]
            options.append((float(np.linalg.norm(projected - image_point)), image_point))

        if not options:
            continue
        options.sort(key=lambda item: item[0])
        if H is None and rng is not None:
            tied = [point for cost, point in options if abs(cost - options[0][0]) <= 1e-9]
            image_point = tied[int(rng.integers(0, len(tied)))] if len(tied) > 1 else options[0][1]
        else:
            image_point = options[0][1]
        corr_name = spec_name if guide_name == line_name else f"{curve_name}__{guide_name}_tangent_apex"
        corrs.append((corr_name, world_point, image_point))
    return corrs


def _tangent_guide_line_names(line_name: str) -> tuple[str, ...]:
    if line_name in {"baseline_left", "baseline_right"}:
        return (line_name, "halfcourt")
    return (line_name,)


def _select_tangent_guide_line(
    selected: dict[str, MarkingLineFit],
    line_name: str,
) -> tuple[str, MarkingLineFit | None]:
    candidates = [(name, selected[name]) for name in _tangent_guide_line_names(line_name) if name in selected]
    if not candidates:
        return line_name, None
    return max(candidates, key=lambda item: float(item[1].score))


def _select_tangent_point_for_guide(
    points: np.ndarray,
    guide_line: np.ndarray,
    curve_name: str,
    guide_name: str,
) -> np.ndarray:
    distances = _line_distances(guide_line, points)
    index = int(np.argmax(distances) if _tangent_prefers_farthest(curve_name, guide_name) else np.argmin(distances))
    return points[index]


def _tangent_prefers_farthest(curve_name: str, guide_name: str) -> bool:
    if curve_name.startswith("free_throw_circle_"):
        return guide_name == "halfcourt"
    if curve_name.startswith("three_point_arc_"):
        return guide_name != "halfcourt"
    return guide_name == "halfcourt"


def _conic_tangent_point_specs() -> tuple[tuple[str, str, str, np.ndarray], ...]:
    mid_y = COURT_WIDTH_CM / 2.0
    free_left_x = FREE_THROW_LINE_X_CM
    free_right_x = COURT_LENGTH_CM - FREE_THROW_LINE_X_CM
    basket_left_x = BASKET_CENTER_FROM_ENDLINE_CM
    basket_right_x = COURT_LENGTH_CM - BASKET_CENTER_FROM_ENDLINE_CM
    return (
        (
            "three_point_arc_left__baseline_tangent_apex",
            "three_point_arc_left",
            "baseline_left",
            np.array([basket_left_x + THREE_POINT_RADIUS_CM, mid_y], dtype=float),
        ),
        (
            "free_throw_circle_left__baseline_tangent_apex",
            "free_throw_circle_left",
            "baseline_left",
            np.array([free_left_x - FREE_THROW_CIRCLE_RADIUS_CM, mid_y], dtype=float),
        ),
        (
            "three_point_arc_right__baseline_tangent_apex",
            "three_point_arc_right",
            "baseline_right",
            np.array([basket_right_x - THREE_POINT_RADIUS_CM, mid_y], dtype=float),
        ),
        (
            "free_throw_circle_right__baseline_tangent_apex",
            "free_throw_circle_right",
            "baseline_right",
            np.array([free_right_x + FREE_THROW_CIRCLE_RADIUS_CM, mid_y], dtype=float),
        ),
    )


def _conic_tangent_points(conic: np.ndarray, tangent_direction: np.ndarray) -> np.ndarray | None:
    direction = np.asarray(tangent_direction, dtype=float).reshape(2)
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-12:
        return None
    line = conic @ np.array([direction[0] / norm, direction[1] / norm, 0.0], dtype=float)
    return _conic_line_intersections(conic, line)


def _conic_line_intersections(conic: np.ndarray, line: np.ndarray) -> np.ndarray | None:
    line = _normalize_line(line)
    point_on_line = -line[2] * line[:2]
    direction = np.array([-line[1], line[0]], dtype=float)
    p = np.array([point_on_line[0], point_on_line[1], 1.0], dtype=float)
    d = np.array([direction[0], direction[1], 0.0], dtype=float)
    a = float(d @ conic @ d)
    b = float(2.0 * p @ conic @ d)
    c = float(p @ conic @ p)
    if abs(a) <= 1e-12:
        if abs(b) <= 1e-12:
            return None
        t = -c / b
        point = point_on_line + t * direction
        return np.stack([point, point])
    disc = b * b - 4.0 * a * c
    if disc <= 1e-9:
        return None
    root = float(np.sqrt(disc))
    t0 = (-b - root) / (2.0 * a)
    t1 = (-b + root) / (2.0 * a)
    pts = np.stack([point_on_line + t0 * direction, point_on_line + t1 * direction])
    return pts if np.isfinite(pts).all() else None


def _sort_points_along_line(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    line = _normalize_line(line)
    direction = np.array([-line[1], line[0]], dtype=float)
    order = np.argsort(points @ direction)
    return points[order]


def _points_in_loose_image_bounds(points: np.ndarray, h: int, w: int) -> bool:
    return bool(
        np.isfinite(points).all()
        and (points[:, 0] >= -w).all()
        and (points[:, 0] <= 2 * w).all()
        and (points[:, 1] >= -h).all()
        and (points[:, 1] <= 2 * h).all()
    )


def _pairing_cost(a: np.ndarray, b: np.ndarray) -> float:
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        return float("inf")
    return float(np.linalg.norm(a - b, axis=1).sum())


def _line_intersection(a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
    p = np.cross(a, b)
    if not np.isfinite(p).all() or abs(float(p[2])) <= 1e-9:
        return None
    xy = p[:2] / p[2]
    return xy if np.isfinite(xy).all() else None


def _passes_structure(per_geometry: dict[str, float], config: MarkingRefinementConfig) -> bool:
    inliers = [name for name, value in per_geometry.items() if value >= config.inlier_score_threshold]
    if len(inliers) < config.min_scored_markings:
        return False
    if config.require_baseline and not any(name in inliers for name in ("baseline_left", "baseline_right")):
        return False
    return True


def _world_line(name: str) -> np.ndarray:
    a, b = court_lines_world()[name]
    return _line_through(a[:2], b[:2])


def _line_through(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _normalize_line(np.cross(np.array([a[0], a[1], 1.0]), np.array([b[0], b[1], 1.0])))


def _fit_line_tls(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centroid = np.average(points, axis=0, weights=weights)
    centered = points - centroid
    cov = (centered * weights[:, None]).T @ centered / weights.sum()
    _, evecs = np.linalg.eigh(cov)
    normal = evecs[:, 0]
    return _normalize_line(np.array([normal[0], normal[1], -float(normal @ centroid)]))


def _normalize_line(line: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(line[:2]))
    if norm <= 1e-12:
        raise ValueError("Degenerate line")
    line = line / norm
    return -line if line[2] < 0 else line


def _clip_line_endpoints(line: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point_on_line = -line[2] * line[:2]
    direction = np.array([-line[1], line[0]])
    t = (points - point_on_line) @ direction
    lo, hi = np.percentile(t, (5.0, 95.0))
    return point_on_line + lo * direction, point_on_line + hi * direction


def _fit_circle(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, float]:
    x, y = points[:, 0], points[:, 1]
    A = np.column_stack([x, y, np.ones(len(points))])
    b = -(x * x + y * y)
    Aw = A * np.sqrt(weights)[:, None]
    bw = b * np.sqrt(weights)
    d, e, f = np.linalg.lstsq(Aw, bw, rcond=None)[0]
    center = np.array([-d / 2.0, -e / 2.0])
    radius = float(np.sqrt(max(1e-9, center @ center - f)))
    return center, radius


def _fit_conic(points: np.ndarray, weights: np.ndarray) -> np.ndarray:
    pts = points.astype(float)
    mean = np.average(pts, axis=0, weights=weights)
    scale = np.sqrt(2.0) / max(1e-9, np.linalg.norm(pts - mean, axis=1).mean())
    x = (pts[:, 0] - mean[0]) * scale
    y = (pts[:, 1] - mean[1]) * scale
    D = np.column_stack([x * x, x * y, y * y, x, y, np.ones(len(x))])
    _, _, vh = np.linalg.svd(D * np.sqrt(weights)[:, None])
    c = vh[-1]
    Cn = np.array([[c[0], c[1] / 2, c[3] / 2], [c[1] / 2, c[2], c[4] / 2], [c[3] / 2, c[4] / 2, c[5]]])
    T = np.array([[scale, 0.0, -scale * mean[0]], [0.0, scale, -scale * mean[1]], [0.0, 0.0, 1.0]])
    C = T.T @ Cn @ T
    return C / max(1e-12, np.linalg.norm(C))


def _conic_geometric_residual(conic: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts = np.column_stack([points, np.ones(len(points))])
    algebraic = np.einsum("ni,ij,nj->n", pts, conic, pts)
    grad_x = 2 * conic[0, 0] * points[:, 0] + 2 * conic[0, 1] * points[:, 1] + 2 * conic[0, 2]
    grad_y = 2 * conic[1, 1] * points[:, 1] + 2 * conic[0, 1] * points[:, 0] + 2 * conic[1, 2]
    return algebraic / np.sqrt(grad_x * grad_x + grad_y * grad_y + 1e-9)


def _line_distances(line: np.ndarray, points: np.ndarray) -> np.ndarray:
    return np.abs(points @ line[:2] + line[2])


def _polyline_sample_weights(points: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return np.empty((0,), dtype=np.float32)
    if len(points) == 1:
        return np.ones((1,), dtype=np.float32)
    segment_lengths = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    weights = np.zeros((len(points),), dtype=np.float64)
    weights[:-1] += 0.5 * segment_lengths
    weights[1:] += 0.5 * segment_lengths
    return weights.astype(np.float32)


def _max_filter(arr: np.ndarray, radius: int) -> np.ndarray:
    if radius <= 0:
        return arr
    k = 2 * int(radius) + 1
    return cv2.dilate(arr.astype(np.float32), np.ones((k, k), np.uint8))


def _class_has_straight_geometry(class_name: str) -> bool:
    return any(v == class_name for k, v in MARKING_CLASS_BY_GEOMETRY.items() if k in STRAIGHT_LINE_NAMES)


def _class_has_curve_geometry(class_name: str) -> bool:
    return any(v == class_name for k, v in MARKING_CLASS_BY_GEOMETRY.items() if k in CURVE_GEOMETRY_NAMES)


def _homography_is_sane(H: np.ndarray | None, image_shape: tuple[int, int]) -> bool:
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-10:
        return False
    corners = np.array([[0.0, 0.0], [2800.0, 0.0], [2800.0, 1500.0], [0.0, 1500.0]])
    proj = project_world_points(H, corners)
    if not np.isfinite(proj).all():
        return False
    area = 0.5 * abs(np.dot(proj[:, 0], np.roll(proj[:, 1], -1)) - np.dot(proj[:, 1], np.roll(proj[:, 0], -1)))
    limit = 8.0 * max(image_shape)
    return area >= 100.0 and np.abs(proj).max() <= limit


def _resize_2d(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
