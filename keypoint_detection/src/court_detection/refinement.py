"""Geometric refinement from court-line heatmaps to world-to-image homographies."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import least_squares

from court_detection.geometry import LINE_NAMES, court_lines_world


OUTER_LINE_NAMES = frozenset(("sideline_far", "sideline_near", "baseline_left", "baseline_right"))


@dataclass
class Stage1Config:
    pixel_threshold: float = 0.3
    min_pixels_per_class: int = 30
    ransac_iter: int = 100
    ransac_distance_threshold: float = 3.0
    irls_iter: int = 5
    huber_threshold: float = 2.0
    early_stop_shift: float = 0.1
    edge_threshold_frac: float = 0.1
    class_prob_threshold: float = 0.2
    use_class_aware_filter: bool = True
    log_sigma: float = 1.5
    endpoint_percentiles: tuple[float, float] = (5.0, 95.0)
    seed: int = 1430


@dataclass
class Stage2Config:
    ransac_iter: int = 200
    line_distance_threshold: float = 4.0
    corner_distance_threshold: float = 5.0
    min_correspondences: int = 4
    max_stage1_residual: float = 30.0
    min_segment_length_px: float = 25.0
    max_corner_segment_overshoot_px: float = 80.0
    seed: int = 1430


@dataclass
class Stage3Config:
    enabled: bool = True
    n_samples_per_line: int = 50
    k1_max_abs: float = 1.0e-6
    rms_reprojection_max: float = 5.0
    max_nfev: int = 200
    xtol: float = 1.0e-8
    ftol: float = 1.0e-8


@dataclass
class RefinementConfig:
    stage1: Stage1Config = field(default_factory=Stage1Config)
    stage2: Stage2Config = field(default_factory=Stage2Config)
    stage3: Stage3Config = field(default_factory=Stage3Config)


@dataclass
class LineFit:
    name: str
    class_id: int
    p0: np.ndarray
    p1: np.ndarray
    line_homog: np.ndarray
    inliers: np.ndarray
    fit_residual: float
    score: float


@dataclass
class CornerCorrespondence:
    name: str
    world: np.ndarray
    image: np.ndarray
    line_a: str
    line_b: str


@dataclass
class Stage2Result:
    H: np.ndarray | None
    success: bool
    inlier_lines: tuple[str, ...] = ()
    inlier_corners: tuple[str, ...] = ()
    corners: tuple[CornerCorrespondence, ...] = ()
    message: str = ""


@dataclass
class Stage3Result:
    H: np.ndarray | None
    k1: float
    success: bool
    residuals_by_line: dict[str, float] = field(default_factory=dict)
    corner_rms: float | None = None
    message: str = ""


@dataclass
class RefinementResult:
    line_fits: dict[str, LineFit]
    stage2: Stage2Result
    stage3: Stage3Result


def line_through_two_points(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    h = np.cross(np.array([a[0], a[1], 1.0]), np.array([b[0], b[1], 1.0]))
    return _normalize_line(h)


def fit_line_tls(points: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    if weights is None:
        weights = np.ones(len(points), dtype=float)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    weights = np.clip(weights, 0.0, None)
    if len(points) < 2 or weights.sum() <= 0:
        raise ValueError("Need at least two positively weighted points to fit a line")

    centroid = np.average(points, axis=0, weights=weights)
    centered = points - centroid
    cov = (centered * weights[:, None]).T @ centered / weights.sum()
    evals, evecs = np.linalg.eigh(cov)
    normal = evecs[:, np.argmin(evals)]
    c = -float(normal @ centroid)
    return _normalize_line(np.array([normal[0], normal[1], c], dtype=float))


def ransac_line(
    pixels: np.ndarray,
    weights: np.ndarray,
    n_iter: int = 100,
    dist_threshold: float = 3.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng() if rng is None else rng
    pixels = np.asarray(pixels, dtype=float)
    weights = np.asarray(weights, dtype=float)
    probs = np.clip(weights, 0.0, None)
    probs = probs / probs.sum() if probs.sum() > 0 else np.full(len(pixels), 1.0 / len(pixels))

    best_score = -np.inf
    best_inliers: np.ndarray | None = None
    for _ in range(n_iter):
        i, j = rng.choice(len(pixels), size=2, replace=False, p=probs)
        if np.linalg.norm(pixels[i] - pixels[j]) < 1e-6:
            continue
        line = line_through_two_points(pixels[i], pixels[j])
        distances = line_distances(line, pixels)
        inliers = distances < dist_threshold
        score = float(weights[inliers].sum())
        if score > best_score:
            best_score = score
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 2:
        line = fit_line_tls(pixels, weights)
        inliers = np.ones(len(pixels), dtype=bool)
        return line, inliers, float(weights.sum())

    line = fit_line_tls(pixels[best_inliers], weights[best_inliers])
    return line, best_inliers, best_score


def refine_frame(
    image_rgb: np.ndarray,
    lineness: np.ndarray,
    class_probs: np.ndarray,
    config: RefinementConfig | None = None,
    line_names: tuple[str, ...] = LINE_NAMES,
) -> RefinementResult:
    """Run all refinement stages in image pixel coordinates.

    `image_rgb` is source-frame RGB in [0, 1] or [0, 255]. `lineness` is HxW and
    `class_probs` is KxHxW after any model-output resizing to the same source frame.
    """
    config = RefinementConfig() if config is None else config
    image = _as_float_image(image_rgb)
    lineness = np.asarray(lineness, dtype=np.float32)
    class_probs = np.asarray(class_probs, dtype=np.float32)
    if lineness.shape != image.shape[:2]:
        lineness = _resize_2d(lineness, image.shape[:2], cv2.INTER_LINEAR)
    if class_probs.shape[-2:] != image.shape[:2]:
        class_probs = np.stack([
            _resize_2d(class_probs[k], image.shape[:2], cv2.INTER_LINEAR) for k in range(class_probs.shape[0])
        ])

    line_fits = fit_lines_from_heatmaps(image, lineness, class_probs, config.stage1, line_names)
    stage2 = estimate_homography_from_lines(line_fits, config.stage2)
    stage3 = refine_homography_and_distortion(
        image, line_fits, stage2, config.stage3
    ) if stage2.success and stage2.H is not None and config.stage3.enabled else Stage3Result(
        H=stage2.H, k1=0.0, success=False, message="Stage 3 disabled or Stage 2 failed"
    )
    return RefinementResult(line_fits=line_fits, stage2=stage2, stage3=stage3)


def fit_lines_from_heatmaps(
    image_rgb: np.ndarray,
    lineness: np.ndarray,
    class_probs: np.ndarray,
    config: Stage1Config | None = None,
    line_names: tuple[str, ...] = LINE_NAMES,
) -> dict[str, LineFit]:
    config = Stage1Config() if config is None else config
    gray = _gray(image_rgb)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    log_resp = -ndimage.gaussian_laplace(gray, sigma=config.log_sigma).astype(np.float32)
    rng = np.random.default_rng(config.seed)

    yy, xx = np.indices(lineness.shape, dtype=np.float32)
    coords = np.stack([xx, yy], axis=-1).reshape(-1, 2)
    fits: dict[str, LineFit] = {}
    for class_id, name in enumerate(line_names[: class_probs.shape[0]]):
        prior = np.clip(lineness * class_probs[class_id], 0.0, 1.0)
        mask = prior > config.pixel_threshold
        if int(mask.sum()) < config.min_pixels_per_class:
            continue

        pixels = coords[mask.reshape(-1)]
        weights = prior.reshape(-1)[mask.reshape(-1)]
        try:
            line, inliers, score = ransac_line(
                pixels, weights, config.ransac_iter, config.ransac_distance_threshold, rng
            )
            response = _class_filter(name, grad_x, grad_y, grad_mag, log_resp, line, config)
            line = _irls_refine_line(line, prior, response, config)
            inlier_pixels = pixels[inliers]
            p0, p1 = clip_line_endpoints(line, inlier_pixels, config.endpoint_percentiles)
            residual = float(np.sqrt(np.mean(line_distances(line, inlier_pixels) ** 2))) if len(inlier_pixels) else np.inf
        except (ValueError, np.linalg.LinAlgError):
            continue
        fits[name] = LineFit(
            name=name,
            class_id=class_id,
            p0=p0,
            p1=p1,
            line_homog=line,
            inliers=inlier_pixels,
            fit_residual=residual,
            score=score,
        )
    return fits


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


def estimate_homography_from_lines(
    line_fits: dict[str, LineFit],
    config: Stage2Config | None = None,
) -> Stage2Result:
    config = Stage2Config() if config is None else config
    world_lines = _world_line_segments_2d()
    usable_fits = {
        name: fit
        for name, fit in line_fits.items()
        if fit.fit_residual <= config.max_stage1_residual
        and np.linalg.norm(fit.p1 - fit.p0) >= config.min_segment_length_px
    }
    line_corrs = [(name, _line_from_segment(*world_lines[name]), fit.line_homog) for name, fit in usable_fits.items()]
    line_corrs = [(n, lw, li) for n, lw, li in line_corrs if n in world_lines]
    corners = extract_corners(usable_fits, max_segment_overshoot_px=config.max_corner_segment_overshoot_px)
    point_corrs = [(c.name, c.world, c.image) for c in corners]

    if not _has_enough_homography_constraints(line_corrs, point_corrs, config.min_correspondences):
        return Stage2Result(None, False, corners=tuple(corners), message="Not enough visible correspondences")

    rng = np.random.default_rng(config.seed)
    best_score = -1
    best_H: np.ndarray | None = None
    best_line_inliers: list[str] = []
    best_corner_inliers: list[str] = []
    if len(point_corrs) >= 4:
        all_indices = [("point", i) for i in range(len(point_corrs))]
    else:
        all_indices = [("line", i) for i in range(len(line_corrs))] + [("point", i) for i in range(len(point_corrs))]
    sample_size = min(max(config.min_correspondences, 4), len(all_indices))

    for _ in range(max(1, config.ransac_iter)):
        sample = rng.choice(len(all_indices), size=sample_size, replace=False)
        sample_lines = [line_corrs[all_indices[i][1]] for i in sample if all_indices[i][0] == "line"]
        sample_points = [point_corrs[all_indices[i][1]] for i in sample if all_indices[i][0] == "point"]
        try:
            H = line_point_dlt(sample_lines, sample_points)
        except (ValueError, np.linalg.LinAlgError):
            continue
        line_inliers, corner_inliers = _score_homography(H, line_corrs, point_corrs, config)
        score = len(line_inliers) + len(corner_inliers)
        if score > best_score:
            best_score = score
            best_H = H
            best_line_inliers = line_inliers
            best_corner_inliers = corner_inliers

    if best_H is None:
        return Stage2Result(None, False, corners=tuple(corners), message="DLT failed")

    inlier_lines = [corr for corr in line_corrs if corr[0] in best_line_inliers]
    inlier_points = [corr for corr in point_corrs if corr[0] in best_corner_inliers]
    if not _has_enough_homography_constraints(inlier_lines, inlier_points, config.min_correspondences):
        return Stage2Result(
            best_H,
            False,
            inlier_lines=tuple(best_line_inliers),
            inlier_corners=tuple(best_corner_inliers),
            corners=tuple(corners),
            message="Not enough inlier correspondences",
        )
    if len(inlier_lines) + len(inlier_points) >= config.min_correspondences:
        try:
            best_H = line_point_dlt(inlier_lines, inlier_points)
        except (ValueError, np.linalg.LinAlgError):
            pass
    if not _homography_is_sane(best_H):
        return Stage2Result(
            best_H,
            False,
            inlier_lines=tuple(best_line_inliers),
            inlier_corners=tuple(best_corner_inliers),
            corners=tuple(corners),
            message="Rejected degenerate homography",
        )
    return Stage2Result(
        H=best_H,
        success=True,
        inlier_lines=tuple(best_line_inliers),
        inlier_corners=tuple(best_corner_inliers),
        corners=tuple(corners),
    )


def extract_corners(line_fits: dict[str, LineFit], max_segment_overshoot_px: float = 80.0) -> list[CornerCorrespondence]:
    specs = (
        ("corner_left_far", "sideline_far", "baseline_left", (0.0, 0.0)),
        ("corner_right_far", "sideline_far", "baseline_right", (2800.0, 0.0)),
        ("corner_left_near", "sideline_near", "baseline_left", (0.0, 1500.0)),
        ("corner_right_near", "sideline_near", "baseline_right", (2800.0, 1500.0)),
        ("half_court_far", "sideline_far", "halfcourt", (1400.0, 0.0)),
        ("half_court_near", "sideline_near", "halfcourt", (1400.0, 1500.0)),
    )
    out: list[CornerCorrespondence] = []
    for name, a, b, world in specs:
        if a not in line_fits or b not in line_fits:
            continue
        xh = np.cross(line_fits[a].line_homog, line_fits[b].line_homog)
        if abs(xh[2]) < 1e-9:
            continue
        xy = xh[:2] / xh[2]
        if not (_near_segment_support(xy, line_fits[a], max_segment_overshoot_px)
                and _near_segment_support(xy, line_fits[b], max_segment_overshoot_px)):
            continue
        out.append(CornerCorrespondence(name, np.array(world, dtype=float), xy, a, b))
    return out


def line_point_dlt(
    line_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    point_corrs: list[tuple[str, np.ndarray, np.ndarray]],
) -> np.ndarray:
    if 2 * len(line_corrs) + 2 * len(point_corrs) < 8:
        raise ValueError("Need at least 8 linear constraints for homography DLT")
    world_points = _normalization_points_from_corrs(line_corrs, point_corrs, world=True)
    image_points = _normalization_points_from_corrs(line_corrs, point_corrs, world=False)
    Tw = _hartley_transform(world_points)
    Ti = _hartley_transform(image_points)

    rows: list[np.ndarray] = []
    for _, lw, li in line_corrs:
        lw_n = np.linalg.inv(Tw).T @ lw
        li_n = np.linalg.inv(Ti).T @ li
        rows.extend(_line_constraint_rows(_normalize_line(lw_n), _normalize_line(li_n)))
    for _, world, image in point_corrs:
        X = Tw @ np.array([world[0], world[1], 1.0], dtype=float)
        x = Ti @ np.array([image[0], image[1], 1.0], dtype=float)
        X = X / X[2]
        x = x / x[2]
        rows.extend(_point_constraint_rows(X[:2], x[:2]))

    A = np.stack(rows)
    _, _, vh = np.linalg.svd(A)
    Hn = vh[-1].reshape(3, 3)
    H = np.linalg.inv(Ti) @ Hn @ Tw
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def brown_conrady_distort(points: np.ndarray, k1: float, principal_point: tuple[float, float]) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    c = np.array(principal_point, dtype=float)
    delta = points - c
    r2 = np.sum(delta * delta, axis=-1, keepdims=True)
    return c + delta * (1.0 + k1 * r2)


def project_world_points(H: np.ndarray, world_xy: np.ndarray, k1: float = 0.0, image_shape: tuple[int, int] | None = None) -> np.ndarray:
    pts = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    homog = np.column_stack([pts, np.ones(len(pts))])
    img_h = (H @ homog.T).T
    img = img_h[:, :2] / img_h[:, 2:3]
    if k1 != 0.0 and image_shape is not None:
        h, w = image_shape
        img = brown_conrady_distort(img, k1, (w / 2.0, h / 2.0))
    return img


def refine_homography_and_distortion(
    image_rgb: np.ndarray,
    line_fits: dict[str, LineFit],
    stage2: Stage2Result,
    config: Stage3Config | None = None,
) -> Stage3Result:
    config = Stage3Config() if config is None else config
    if stage2.H is None:
        return Stage3Result(None, 0.0, False, message="No Stage 2 homography")

    gray = _gray(image_rgb)
    grad_x = cv2.Scharr(gray, cv2.CV_32F, 1, 0)
    grad_y = cv2.Scharr(gray, cv2.CV_32F, 0, 1)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    max_grad = float(np.percentile(grad_mag, 99.5))
    if max_grad <= 1e-12:
        max_grad = 1.0
    grad_norm = np.clip(grad_mag / max_grad, 0.0, 1.0)

    world_lines = _world_line_segments_2d()
    active_lines = [name for name in line_fits if name in world_lines]
    if not active_lines:
        return Stage3Result(stage2.H, 0.0, False, message="No lines to refine")

    H0 = stage2.H / stage2.H[2, 2]
    theta0 = np.array([H0[0, 0], H0[0, 1], H0[0, 2], H0[1, 0], H0[1, 1], H0[1, 2], H0[2, 0], H0[2, 1], 0.0])
    h_img, w_img = image_rgb.shape[:2]

    line_samples: list[tuple[str, np.ndarray]] = []
    for name in active_lines:
        a, b = world_lines[name]
        t0, t1 = _visible_world_interval(stage2.H, line_fits[name], a, b)
        ts = np.linspace(t0, t1, config.n_samples_per_line)
        line_samples.append((name, a[None] + ts[:, None] * (b - a)[None]))

    corner_corrs = [c for c in stage2.corners if c.name in stage2.inlier_corners]

    def unpack(theta: np.ndarray) -> tuple[np.ndarray, float]:
        H = np.array([
            [theta[0], theta[1], theta[2]],
            [theta[3], theta[4], theta[5]],
            [theta[6], theta[7], 1.0],
        ], dtype=float)
        return H, float(theta[8])

    def residuals(theta: np.ndarray) -> np.ndarray:
        H, k1 = unpack(theta)
        res: list[np.ndarray] = []
        for name, samples in line_samples:
            img = project_world_points(H, samples, k1, (h_img, w_img))
            evidence = _bilinear_sample(grad_norm, img)
            in_frame = (
                (img[:, 0] >= 0.0) & (img[:, 0] <= w_img - 1.0)
                & (img[:, 1] >= 0.0) & (img[:, 1] <= h_img - 1.0)
            )
            weight = max(0.25, min(2.0, line_fits[name].score / 50.0))
            res.append((1.0 - evidence) * in_frame.astype(float) * weight)
        for corner in corner_corrs:
            pred = project_world_points(H, corner.world[None], k1, (h_img, w_img))[0]
            res.append((pred - corner.image) / 5.0)
        return np.concatenate([np.ravel(r) for r in res])

    try:
        result = least_squares(
            residuals, theta0, method="lm", max_nfev=config.max_nfev, xtol=config.xtol, ftol=config.ftol
        )
    except ValueError as exc:
        return Stage3Result(stage2.H, 0.0, False, message=f"LM failed: {exc}")

    H_opt, k1 = unpack(result.x)
    H_opt = H_opt / H_opt[2, 2]
    if abs(k1) > config.k1_max_abs:
        return Stage3Result(stage2.H, 0.0, False, message=f"Rejected k1={k1:.3g}")

    corner_rms = _corner_rms(H_opt, k1, corner_corrs, (h_img, w_img))
    if corner_rms is not None and corner_rms > config.rms_reprojection_max:
        return Stage3Result(stage2.H, 0.0, False, corner_rms=corner_rms, message="Rejected by corner RMS")

    residual_by_line = {}
    for name, samples in line_samples:
        img = project_world_points(H_opt, samples, k1, (h_img, w_img))
        residual_by_line[name] = float(np.mean(1.0 - _bilinear_sample(grad_norm, img)))
    return Stage3Result(H_opt, k1, bool(result.success), residual_by_line, corner_rms, result.message)


def _irls_refine_line(line: np.ndarray, prior: np.ndarray, response: np.ndarray, config: Stage1Config) -> np.ndarray:
    response = np.clip(response, 0.0, None)
    threshold = config.edge_threshold_frac * float(response.max())
    mask = (response > threshold) & (prior > config.class_prob_threshold)
    ys, xs = np.nonzero(mask)
    if len(xs) < 2:
        return line
    points = np.column_stack([xs, ys]).astype(float)
    base_weight = prior[ys, xs] * response[ys, xs]
    current = line.copy()
    for _ in range(config.irls_iter):
        distances = line_distances(current, points)
        huber = np.ones_like(distances)
        far = distances > config.huber_threshold
        huber[far] = config.huber_threshold / np.maximum(distances[far], 1e-6)
        weights = base_weight * huber
        if weights.sum() <= 0:
            break
        new_line = fit_line_tls(points, weights)
        if _line_shift(current, new_line, points.mean(axis=0)) < config.early_stop_shift:
            current = new_line
            break
        current = new_line
    return current


def _class_filter(
    name: str,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    grad_mag: np.ndarray,
    log_resp: np.ndarray,
    line: np.ndarray,
    config: Stage1Config,
) -> np.ndarray:
    if not config.use_class_aware_filter:
        return grad_mag
    if name not in OUTER_LINE_NAMES:
        return log_resp
    normal = line[:2]
    signed = grad_x * normal[0] + grad_y * normal[1]
    return np.maximum(signed, -signed)


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


def _line_shift(a: np.ndarray, b: np.ndarray, around: np.ndarray) -> float:
    pa = float(a[:2] @ around + a[2])
    pb = float(b[:2] @ around + b[2])
    return abs(pa - pb)


def _world_line_segments_2d() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    return {name: (a[:2].astype(float), b[:2].astype(float)) for name, (a, b) in court_lines_world().items()}


def _line_from_segment(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return line_through_two_points(np.asarray(a, dtype=float), np.asarray(b, dtype=float))


def _near_segment_support(point: np.ndarray, fit: LineFit, tolerance: float) -> bool:
    segment = fit.p1 - fit.p0
    length2 = float(segment @ segment)
    if length2 <= 1e-12:
        return False
    t = float((point - fit.p0) @ segment / length2)
    extra = tolerance / np.sqrt(length2)
    return -extra <= t <= 1.0 + extra


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
    for name, _, _ in line_corrs:
        if world:
            pts.extend(segments[name])
        else:
            continue
    for _, w, i in point_corrs:
        pts.append(w if world else i)
    arr = np.asarray(pts, dtype=float)
    if not world:
        img_pts = []
        for _, _, li in line_corrs:
            p = -li[2] * li[:2]
            img_pts.extend([p - 100.0 * np.array([-li[1], li[0]]), p + 100.0 * np.array([-li[1], li[0]])])
        img_pts.extend(i for _, _, i in point_corrs)
        arr = np.asarray(img_pts, dtype=float)
    return arr


def _hartley_transform(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    centroid = points.mean(axis=0)
    dist = np.linalg.norm(points - centroid, axis=1).mean()
    scale = np.sqrt(2.0) / dist if dist > 1e-12 else 1.0
    return np.array([[scale, 0.0, -scale * centroid[0]], [0.0, scale, -scale * centroid[1]], [0.0, 0.0, 1.0]])


def _score_homography(
    H: np.ndarray,
    line_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    point_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    config: Stage2Config,
) -> tuple[list[str], list[str]]:
    world_segments = _world_line_segments_2d()
    line_inliers = []
    for name, _, li in line_corrs:
        a, b = world_segments[name]
        proj = project_world_points(H, np.stack([a, b]))
        err = float(np.mean(line_distances(li, proj)))
        if np.isfinite(err) and err < config.line_distance_threshold:
            line_inliers.append(name)
    corner_inliers = []
    for name, world, image in point_corrs:
        proj = project_world_points(H, world[None])[0]
        err = float(np.linalg.norm(proj - image))
        if np.isfinite(err) and err < config.corner_distance_threshold:
            corner_inliers.append(name)
    return line_inliers, corner_inliers


def _has_enough_homography_constraints(
    line_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    point_corrs: list[tuple[str, np.ndarray, np.ndarray]],
    min_correspondences: int,
) -> bool:
    return len(line_corrs) + len(point_corrs) >= min_correspondences and 2 * len(line_corrs) + 2 * len(point_corrs) >= 8


def _homography_is_sane(H: np.ndarray | None) -> bool:
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
    diagonal = max(
        np.linalg.norm(proj[0] - proj[2]),
        np.linalg.norm(proj[1] - proj[3]),
    )
    return area >= 100.0 and diagonal >= 30.0


def _visible_world_interval(H: np.ndarray, fit: LineFit, a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    samples = a[None] + np.linspace(0.0, 1.0, 200)[:, None] * (b - a)[None]
    proj = project_world_points(H, samples)
    direction = fit.p1 - fit.p0
    length2 = float(direction @ direction)
    if length2 <= 1e-12:
        return 0.0, 1.0
    t_seg = ((proj - fit.p0) @ direction) / length2
    mask = (t_seg >= -0.05) & (t_seg <= 1.05) & np.isfinite(proj).all(axis=1)
    if mask.sum() < 2:
        return 0.0, 1.0
    ts = np.linspace(0.0, 1.0, 200)[mask]
    return float(ts.min()), float(ts.max())


def _corner_rms(
    H: np.ndarray,
    k1: float,
    corners: list[CornerCorrespondence],
    image_shape: tuple[int, int],
) -> float | None:
    if not corners:
        return None
    sq = []
    for corner in corners:
        pred = project_world_points(H, corner.world[None], k1, image_shape)[0]
        sq.append(float(np.sum((pred - corner.image) ** 2)))
    return float(np.sqrt(np.mean(sq)))


def _bilinear_sample(arr: np.ndarray, xy: np.ndarray) -> np.ndarray:
    x = xy[:, 0]
    y = xy[:, 1]
    coords = np.vstack([y, x])
    return ndimage.map_coordinates(arr, coords, order=1, mode="constant", cval=0.0)


def _resize_2d(arr: np.ndarray, shape_hw: tuple[int, int], interpolation: int) -> np.ndarray:
    return cv2.resize(np.asarray(arr), (shape_hw[1], shape_hw[0]), interpolation=interpolation)


def _as_float_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype.kind in "ui":
        return image.astype(np.float32) / 255.0
    return image.astype(np.float32)


def _gray(image_rgb: np.ndarray) -> np.ndarray:
    image = _as_float_image(image_rgb)
    if image.ndim == 2:
        return image.astype(np.float32)
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
