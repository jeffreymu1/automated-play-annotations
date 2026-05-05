"""Homography refinement from expanded FIBA court-marking heatmaps.

This module handles shared marking classes such as ``baseline`` and curved
markings such as ``three_point_arc`` and ``free_throw_circle``.  The homography
is initialized with DLT from straight marking candidates, then scored and
optionally refined against the full court template, including curve evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import least_squares

from court_detection.geometry import (
    COURT_LENGTH_CM,
    COURT_WIDTH_CM,
    MARKING_CLASS_BY_GEOMETRY,
    MARKING_CLASS_NAMES,
    STRAIGHT_LINE_NAMES,
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
    ransac_iter: int = 1200
    min_dlt_lines: int = 4
    n_template_samples: int = 120
    score_radius_px: int = 3
    min_projected_coverage: float = 0.08
    inlier_score_threshold: float = 0.08
    min_scored_markings: int = 5
    require_baseline: bool = True
    curve_refine_weight: float = 0.35
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
    if sum(bool(v) for v in geometry_candidates.values()) < config.min_dlt_lines:
        return MarkingHomographyResult(
            None,
            False,
            -np.inf,
            line_fits=line_fits,
            curve_fits=curve_fits,
            candidate_pixels=candidate_pixels,
            geometry_masses=geometry_masses,
            message="Not enough straight marking candidates",
        )

    rng = np.random.default_rng(config.seed)
    best_H: np.ndarray | None = None
    best_score = -np.inf
    best_selected: dict[str, MarkingLineFit] = {}
    best_per_geometry: dict[str, float] = {}
    geometry_names = [name for name, fits in geometry_candidates.items() if fits]
    for _ in range(max(1, config.ransac_iter)):
        selected = {name: rng.choice(geometry_candidates[name]) for name in geometry_names}
        line_corrs = [
            (name, _world_line(name), fit.line_homog)
            for name, fit in selected.items()
        ]
        point_corrs = _intersection_point_correspondences(selected, image_shape)
        try:
            H = line_point_dlt(line_corrs, point_corrs)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not _homography_is_sane(H, image_shape):
            continue
        score, per_geometry, coverage = score_homography_template(
            H, evidence, image_shape, config, marking_names, side_prob
        )
        if coverage < config.min_projected_coverage:
            continue
        if not _passes_structure(per_geometry, config):
            continue
        if score > best_score:
            best_H = H
            best_score = score
            best_selected = _select_line_fits_for_homography(H, geometry_candidates)
            best_per_geometry = per_geometry

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

    refined_H = refine_homography_with_primitives(best_H, best_selected, curve_fits, config)
    if refined_H is not None and _homography_is_sane(refined_H, image_shape):
        refined_score, refined_per_geometry, coverage = score_homography_template(
            refined_H, evidence, image_shape, config, marking_names, side_prob
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
        per_geometry_scores=best_per_geometry,
        candidate_pixels=candidate_pixels,
        geometry_masses=geometry_masses,
        message="ok",
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
) -> tuple[float, dict[str, float], float]:
    h, w = image_shape
    pooled = np.stack([_max_filter(evidence[k], config.score_radius_px) for k in range(evidence.shape[0])])
    markings = court_markings_world(n=config.n_template_samples)
    per_geometry: dict[str, float] = {}
    coverages = []
    for geometry_name, world_xyz in markings.items():
        class_name = MARKING_CLASS_BY_GEOMETRY[geometry_name]
        if class_name not in marking_names:
            continue
        class_id = marking_names.index(class_name)
        projected = project_world_points(H, world_xyz[:, :2])
        finite = np.isfinite(projected).all(axis=1)
        inside = (
            finite
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] <= w - 1)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] <= h - 1)
        )
        coverages.append(float(inside.mean()))
        if not inside.any():
            per_geometry[geometry_name] = 0.0
            continue
        x = np.round(projected[inside, 0]).astype(int).clip(0, w - 1)
        y = np.round(projected[inside, 1]).astype(int).clip(0, h - 1)
        values = pooled[class_id, y, x]
        side_name = _geometry_side(geometry_name)
        if side_name is not None and side_prob is not None:
            side_values = side_prob[y, x] if side_name == "right" else 1.0 - side_prob[y, x]
            values = values * np.clip(side_values, 0.0, 1.0)
        per_geometry[geometry_name] = float(np.mean(values))
    if not per_geometry:
        return -np.inf, {}, 0.0
    coverage = float(np.mean(coverages)) if coverages else 0.0
    visible_scores = np.array(list(per_geometry.values()), dtype=float)
    score = float(np.mean(np.clip(visible_scores, 0.0, 1.0)) * coverage)
    return score, per_geometry, coverage


def refine_homography_with_primitives(
    H0: np.ndarray,
    selected_lines: dict[str, MarkingLineFit],
    curve_fits: dict[str, list[MarkingCurveFit]],
    config: MarkingRefinementConfig,
) -> np.ndarray | None:
    curve_geometries = [
        name
        for name in CURVE_GEOMETRY_NAMES
        if any(fit.geometry_name == name for fit in curve_fits.get(MARKING_CLASS_BY_GEOMETRY[name], []))
    ]
    if not selected_lines and not curve_geometries:
        return None

    def pack(H: np.ndarray) -> np.ndarray:
        H = H / H[2, 2]
        return H.reshape(-1)[:8]

    def unpack(p: np.ndarray) -> np.ndarray:
        return np.array([[p[0], p[1], p[2]], [p[3], p[4], p[5]], [p[6], p[7], 1.0]], dtype=float)

    def residuals(p: np.ndarray) -> np.ndarray:
        H = unpack(p)
        out = []
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
