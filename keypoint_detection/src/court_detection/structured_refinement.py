"""Structured homography RANSAC directly from court-line heatmaps."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from court_detection.geometry import LINE_NAMES, court_lines_world, sample_segment


CORNER_SPECS = (
    ("corner_left_far", "sideline_far", "baseline_left", (0.0, 0.0)),
    ("corner_right_far", "sideline_far", "baseline_right", (2800.0, 0.0)),
    ("corner_left_near", "sideline_near", "baseline_left", (0.0, 1500.0)),
    ("corner_right_near", "sideline_near", "baseline_right", (2800.0, 1500.0)),
    ("half_court_far", "sideline_far", "halfcourt", (1400.0, 0.0)),
    ("half_court_near", "sideline_near", "halfcourt", (1400.0, 1500.0)),
)


@dataclass
class StructuredRansacConfig:
    ransac_iter: int = 500
    line_threshold: float = 0.55
    class_threshold: float = 0.55
    joint_threshold: float = 0.30
    min_pixels_per_class: int = 8
    sample_line_count: int = 5
    line_refine_distance_px: float = 8.0
    min_refine_pixels: int = 6
    n_samples_per_line: int = 80
    score_radius_px: int = 3
    min_projected_coverage: float = 0.12
    inlier_line_score_threshold: float = 0.08
    min_scored_lines: int = 3
    corner_inlier_threshold_px: float = 8.0
    seed: int = 1430


@dataclass
class SampledLine:
    name: str
    class_id: int
    line_homog: np.ndarray
    p0: np.ndarray
    p1: np.ndarray
    support: int
    score: float


@dataclass
class SampledCorner:
    name: str
    world: np.ndarray
    image: np.ndarray
    line_a: str
    line_b: str


@dataclass
class StructuredHomographyResult:
    H: np.ndarray | None
    success: bool
    score: float
    sampled_lines: dict[str, SampledLine] = field(default_factory=dict)
    corners: tuple[SampledCorner, ...] = ()
    inlier_lines: tuple[str, ...] = ()
    inlier_corners: tuple[str, ...] = ()
    per_line_scores: dict[str, float] = field(default_factory=dict)
    candidate_pixels: dict[str, int] = field(default_factory=dict)
    message: str = ""


def fit_homography_from_heatmaps(
    image_rgb: np.ndarray,
    lineness: np.ndarray,
    class_probs: np.ndarray,
    config: StructuredRansacConfig | None = None,
    line_names: tuple[str, ...] = LINE_NAMES,
) -> StructuredHomographyResult:
    config = StructuredRansacConfig() if config is None else config
    image_shape = image_rgb.shape[:2]
    lineness = _resize_2d(np.asarray(lineness, dtype=np.float32), image_shape)
    class_probs = np.asarray(class_probs, dtype=np.float32)
    if class_probs.shape[-2:] != image_shape:
        class_probs = np.stack([_resize_2d(class_probs[k], image_shape) for k in range(class_probs.shape[0])])

    evidence = np.clip(class_probs * lineness[None], 0.0, 1.0)
    candidates = _candidate_pixels(lineness, class_probs, evidence, config, line_names)
    world_lines = court_lines_world()
    visible = [
        name for name in line_names[: class_probs.shape[0]]
        if name in world_lines and len(candidates.get(name, ())) >= config.min_pixels_per_class
    ]
    candidate_counts = {name: int(len(candidates.get(name, ()))) for name in line_names[: class_probs.shape[0]]}
    if len(visible) < 3:
        return StructuredHomographyResult(
            None, False, -np.inf, candidate_pixels=candidate_counts, message="Not enough line classes"
        )

    rng = np.random.default_rng(config.seed)
    best: StructuredHomographyResult | None = None
    for _ in range(max(1, config.ransac_iter)):
        selected = _sample_line_classes(visible, config.sample_line_count, rng)
        sampled_lines = _sample_lines_for_classes(selected, candidates, evidence, config, rng, line_names)
        if len(sampled_lines) < 3:
            continue
        corners = extract_structured_corners(sampled_lines, image_shape)
        line_corrs = _line_correspondences(sampled_lines)
        point_corrs = [(c.name, c.world, c.image) for c in corners]
        if not _has_enough_constraints(line_corrs, point_corrs):
            continue
        try:
            H = line_point_dlt(line_corrs, point_corrs)
        except (ValueError, np.linalg.LinAlgError):
            continue
        if not _homography_is_sane(H, image_shape):
            continue
        score, per_line_scores, coverage = score_homography_on_heatmaps(H, evidence, image_shape, config, line_names)
        if coverage < config.min_projected_coverage:
            continue
        line_inliers = tuple(name for name, value in per_line_scores.items() if value >= config.inlier_line_score_threshold)
        if len(line_inliers) < config.min_scored_lines:
            continue
        corner_inliers = tuple(_corner_inliers(H, corners, config))
        result = StructuredHomographyResult(
            H=H,
            success=True,
            score=score,
            sampled_lines=sampled_lines,
            corners=tuple(corners),
            inlier_lines=line_inliers,
            inlier_corners=corner_inliers,
            per_line_scores=per_line_scores,
            candidate_pixels=candidate_counts,
            message="ok",
        )
        if best is None or result.score > best.score:
            best = result

    if best is None:
        return StructuredHomographyResult(
            None, False, -np.inf, candidate_pixels=candidate_counts, message="No valid homography"
        )

    refined = _refit_best_homography(best, image_shape, config)
    if refined is not None:
        score, per_line_scores, coverage = score_homography_on_heatmaps(refined, evidence, image_shape, config, line_names)
        if coverage >= config.min_projected_coverage and score >= best.score * 0.95:
            best.H = refined
            best.score = score
            best.per_line_scores = per_line_scores
            best.inlier_lines = tuple(
                name for name, value in per_line_scores.items() if value >= config.inlier_line_score_threshold
            )
    return best


def extract_structured_corners(
    sampled_lines: dict[str, SampledLine],
    image_shape: tuple[int, int],
    margin_px: float = 80.0,
) -> list[SampledCorner]:
    h, w = image_shape
    out: list[SampledCorner] = []
    for name, line_a, line_b, world in CORNER_SPECS:
        if line_a not in sampled_lines or line_b not in sampled_lines:
            continue
        xh = np.cross(sampled_lines[line_a].line_homog, sampled_lines[line_b].line_homog)
        if abs(xh[2]) < 1e-9:
            continue
        xy = xh[:2] / xh[2]
        if not np.isfinite(xy).all():
            continue
        if -margin_px <= xy[0] <= w + margin_px and -margin_px <= xy[1] <= h + margin_px:
            out.append(SampledCorner(name, np.array(world, dtype=float), xy, line_a, line_b))
    return out


def line_point_dlt(
    line_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    point_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    line_weights: dict[str, float] | None = None,
    point_weights: dict[str, float] | None = None,
) -> np.ndarray:
    if 2 * len(line_corrs) + 2 * len(point_corrs) < 8:
        raise ValueError("Need at least 8 linear constraints for homography DLT")
    world_points = _normalization_points_from_corrs(line_corrs, point_corrs, world=True)
    image_points = _normalization_points_from_corrs(line_corrs, point_corrs, world=False)
    Tw = _hartley_transform(world_points)
    Ti = _hartley_transform(image_points)

    rows: list[np.ndarray] = []
    for name, lw, li in line_corrs:
        lw_n = np.linalg.inv(Tw).T @ lw
        li_n = np.linalg.inv(Ti).T @ li
        weight = _dlt_weight(name, line_weights)
        rows.extend(weight * row for row in _line_constraint_rows(_normalize_line(lw_n), _normalize_line(li_n)))
    for name, world, image in point_corrs:
        X = Tw @ np.array([world[0], world[1], 1.0], dtype=float)
        x = Ti @ np.array([image[0], image[1], 1.0], dtype=float)
        X = X / X[2]
        x = x / x[2]
        weight = _dlt_weight(name, point_weights)
        rows.extend(weight * row for row in _point_constraint_rows(X[:2], x[:2]))

    A = np.stack(rows)
    _, _, vh = np.linalg.svd(A)
    Hn = vh[-1].reshape(3, 3)
    H = np.linalg.inv(Ti) @ Hn @ Tw
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def _dlt_weight(name: str, weights: dict[str, float] | None) -> float:
    if weights is None:
        return 1.0
    weight = float(weights.get(name, 1.0))
    if not np.isfinite(weight) or weight <= 0.0:
        return 1.0
    return float(np.sqrt(weight))


def project_world_points(H: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    pts = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    homog = np.column_stack([pts, np.ones(len(pts))])
    img_h = (H @ homog.T).T
    return img_h[:, :2] / img_h[:, 2:3]


def score_homography_on_heatmaps(
    H: np.ndarray,
    evidence: np.ndarray,
    image_shape: tuple[int, int],
    config: StructuredRansacConfig,
    line_names: tuple[str, ...] = LINE_NAMES,
) -> tuple[float, dict[str, float], float]:
    h, w = image_shape
    per_line: dict[str, float] = {}
    coverages = []
    for class_id, name in enumerate(line_names[: evidence.shape[0]]):
        lines = court_lines_world()
        if name not in lines:
            continue
        a, b = lines[name]
        world = sample_segment(a[:2], b[:2], n=config.n_samples_per_line)
        projected = project_world_points(H, world)
        finite = np.isfinite(projected).all(axis=1)
        inside = (
            finite
            & (projected[:, 0] >= 0.0)
            & (projected[:, 0] <= w - 1)
            & (projected[:, 1] >= 0.0)
            & (projected[:, 1] <= h - 1)
        )
        coverages.append(float(inside.mean()))
        values = _sample_max_response(evidence[class_id], projected[inside], config.score_radius_px)
        per_line[name] = float(np.mean(np.clip(values, 0.0, 1.0))) if len(values) else 0.0
    if not per_line:
        return -np.inf, {}, 0.0
    coverage = float(np.mean(coverages)) if coverages else 0.0
    return float(np.mean(list(per_line.values())) * coverage), per_line, coverage


def _candidate_pixels(
    lineness: np.ndarray,
    class_probs: np.ndarray,
    evidence: np.ndarray,
    config: StructuredRansacConfig,
    line_names: tuple[str, ...],
) -> dict[str, np.ndarray]:
    yy, xx = np.indices(lineness.shape, dtype=np.float32)
    coords = np.stack([xx, yy], axis=-1)
    out: dict[str, np.ndarray] = {}
    for class_id, name in enumerate(line_names[: class_probs.shape[0]]):
        mask = (
            (lineness >= config.line_threshold)
            & (class_probs[class_id] >= config.class_threshold)
            & (evidence[class_id] >= config.joint_threshold)
        )
        out[name] = coords[mask].astype(float)
    return out


def _sample_line_classes(visible: list[str], sample_count: int, rng: np.random.Generator) -> list[str]:
    if len(visible) <= sample_count:
        return list(visible)
    return list(rng.choice(visible, size=max(3, sample_count), replace=False))


def _sample_lines_for_classes(
    selected: list[str],
    candidates: dict[str, np.ndarray],
    evidence: np.ndarray,
    config: StructuredRansacConfig,
    rng: np.random.Generator,
    line_names: tuple[str, ...],
) -> dict[str, SampledLine]:
    out: dict[str, SampledLine] = {}
    class_lookup = {name: i for i, name in enumerate(line_names)}
    for name in selected:
        class_id = class_lookup[name]
        pts = candidates[name]
        if len(pts) < 2:
            continue
        weights = _point_weights(pts, evidence[class_id])
        probs = weights / weights.sum() if weights.sum() > 0 else np.full(len(pts), 1.0 / len(pts))
        try:
            i, j = rng.choice(len(pts), size=2, replace=False, p=probs)
            if np.linalg.norm(pts[i] - pts[j]) < 1e-6:
                continue
            line = line_through_two_points(pts[i], pts[j])
            distances = line_distances(line, pts)
            support_mask = distances <= config.line_refine_distance_px
            if int(support_mask.sum()) >= config.min_refine_pixels:
                line = fit_line_tls(pts[support_mask], weights[support_mask])
            support_pts = pts[support_mask] if support_mask.any() else pts[[i, j]]
            p0, p1 = clip_line_endpoints(line, support_pts)
            out[name] = SampledLine(
                name=name,
                class_id=class_id,
                line_homog=line,
                p0=p0,
                p1=p1,
                support=int(support_mask.sum()),
                score=float(weights[support_mask].sum()) if support_mask.any() else float(weights[[i, j]].sum()),
            )
        except (ValueError, np.linalg.LinAlgError):
            continue
    return out


def _point_weights(points: np.ndarray, score_map: np.ndarray) -> np.ndarray:
    h, w = score_map.shape
    x = np.clip(np.round(points[:, 0]).astype(int), 0, w - 1)
    y = np.clip(np.round(points[:, 1]).astype(int), 0, h - 1)
    return np.clip(score_map[y, x].astype(float), 0.0, None)


def _line_correspondences(sampled_lines: dict[str, SampledLine]) -> list[tuple[str, np.ndarray, np.ndarray]]:
    world_lines = _world_line_segments_2d()
    return [
        (name, _line_from_segment(*world_lines[name]), sampled.line_homog)
        for name, sampled in sampled_lines.items()
        if name in world_lines
    ]


def _refit_best_homography(
    result: StructuredHomographyResult,
    image_shape: tuple[int, int],
    config: StructuredRansacConfig,
) -> np.ndarray | None:
    line_corrs = [
        corr for corr in _line_correspondences(result.sampled_lines)
        if corr[0] in result.inlier_lines or len(result.inlier_lines) < 4
    ]
    point_corrs = [(c.name, c.world, c.image) for c in result.corners if c.name in result.inlier_corners]
    if not _has_enough_constraints(line_corrs, point_corrs):
        point_corrs = [(c.name, c.world, c.image) for c in result.corners]
    if not _has_enough_constraints(line_corrs, point_corrs):
        return None
    try:
        H = line_point_dlt(line_corrs, point_corrs)
    except (ValueError, np.linalg.LinAlgError):
        return None
    return H if _homography_is_sane(H, image_shape) else None


def _corner_inliers(
    H: np.ndarray,
    corners: list[SampledCorner],
    config: StructuredRansacConfig,
) -> list[str]:
    out = []
    for corner in corners:
        projected = project_world_points(H, corner.world[None])[0]
        if np.isfinite(projected).all() and np.linalg.norm(projected - corner.image) <= config.corner_inlier_threshold_px:
            out.append(corner.name)
    return out


def line_through_two_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = np.cross(np.array([a[0], a[1], 1.0]), np.array([b[0], b[1], 1.0]))
    return _normalize_line(h)


def fit_line_tls(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if weights is None:
        weights = np.ones(len(points), dtype=float)
    weights = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, None)
    if len(points) < 2 or weights.sum() <= 0:
        raise ValueError("Need at least two positively weighted points")
    centroid = np.average(points, axis=0, weights=weights)
    centered = points - centroid
    cov = (centered * weights[:, None]).T @ centered / weights.sum()
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, np.argmin(evals)]
    return _normalize_line(np.array([normal[0], normal[1], -float(normal @ centroid)], dtype=float))


def clip_line_endpoints(
    line: np.ndarray,
    points: np.ndarray,
    percentiles: tuple[float, float] = (5.0, 95.0),
) -> tuple[np.ndarray, np.ndarray]:
    point_on_line = -line[2] * line[:2]
    direction = np.array([-line[1], line[0]], dtype=float)
    t = (points - point_on_line) @ direction
    lo, hi = np.percentile(t, percentiles)
    return point_on_line + lo * direction, point_on_line + hi * direction


def line_distances(line: np.ndarray, points: np.ndarray) -> np.ndarray:
    line = _normalize_line(line)
    return np.abs(points @ line[:2] + line[2])


def _normalize_line(line: np.ndarray) -> np.ndarray:
    line = np.asarray(line, dtype=float)
    norm = float(np.linalg.norm(line[:2]))
    if norm <= 1e-12:
        raise ValueError("Degenerate line")
    line = line / norm
    if line[2] < 0:
        line = -line
    return line


def _world_line_segments_2d() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: (a[:2].astype(float), b[:2].astype(float)) for name, (a, b) in court_lines_world().items()}


def _line_from_segment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return line_through_two_points(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def _has_enough_constraints(
    line_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    point_corrs: list[tuple[str, np.ndarray, np.ndarray]],
) -> bool:
    return 2 * len(line_corrs) + 2 * len(point_corrs) >= 8


def _line_constraint_rows(lw: np.ndarray, li: np.ndarray) -> list[np.ndarray]:
    def col_coeff(col: int) -> np.ndarray:
        coeff = np.zeros(9, dtype=float)
        coeff[col] = li[0]
        coeff[3 + col] = li[1]
        coeff[6 + col] = li[2]
        return coeff

    v0 = col_coeff(0)
    v1 = col_coeff(1)
    v2 = col_coeff(2)
    return [lw[1] * v0 - lw[0] * v1, lw[2] * v0 - lw[0] * v2]


def _point_constraint_rows(world: np.ndarray, image: np.ndarray) -> list[np.ndarray]:
    X, Y = world
    u, v = image
    return [
        np.array([-X, -Y, -1.0, 0.0, 0.0, 0.0, u * X, u * Y, u], dtype=float),
        np.array([0.0, 0.0, 0.0, -X, -Y, -1.0, v * X, v * Y, v], dtype=float),
    ]


def _normalization_points_from_corrs(
    line_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    point_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    world: bool,
) -> np.ndarray:
    pts = []
    segments = _world_line_segments_2d()
    for name, _, li in line_corrs:
        if world:
            pts.extend(segments[name])
        else:
            p = -li[2] * li[:2]
            d = np.array([-li[1], li[0]])
            pts.extend([p - 100.0 * d, p + 100.0 * d])
    for _, w, i in point_corrs:
        pts.append(w if world else i)
    return np.asarray(pts, dtype=float)


def _hartley_transform(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    centroid = points.mean(axis=0)
    dist = np.linalg.norm(points - centroid, axis=1).mean()
    scale = np.sqrt(2.0) / dist if dist > 1e-12 else 1.0
    return np.array([[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]])


def _homography_is_sane(H: np.ndarray | None, image_shape: tuple[int, int]) -> bool:
    if H is None or not np.isfinite(H).all() or abs(np.linalg.det(H)) < 1e-10:
        return False
    corners = np.array([[0.0, 0.0], [2800.0, 0.0], [2800.0, 1500.0], [0.0, 1500.0]])
    proj = project_world_points(H, corners)
    if not np.isfinite(proj).all():
        return False
    area = 0.5 * abs(
        np.dot(proj[:, 0], np.roll(proj[:, 1], -1))
        - np.dot(proj[:, 1], np.roll(proj[:, 0], -1))
    )
    diagonal = max(np.linalg.norm(proj[0] - proj[2]), np.linalg.norm(proj[1] - proj[3]))
    h, w = image_shape
    limit = 8.0 * max(h, w)
    return area >= 100.0 and diagonal >= 30.0 and np.abs(proj).max() <= limit


def _sample_max_response(score_map: np.ndarray, points: np.ndarray, radius: int) -> np.ndarray:
    if len(points) == 0:
        return np.empty(0, dtype=float)
    h, w = score_map.shape
    values = []
    r = max(0, int(radius))
    for x, y in np.round(points).astype(int):
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        if x0 < x1 and y0 < y1:
            values.append(float(score_map[y0:y1, x0:x1].max()))
    return np.asarray(values, dtype=float)


def _resize_2d(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if arr.shape == shape:
        return arr
    return cv2.resize(arr, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
