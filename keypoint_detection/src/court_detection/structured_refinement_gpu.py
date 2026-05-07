"""Batched Torch implementation of structured homography RANSAC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from court_detection.geometry import LINE_NAMES, court_lines_world, sample_segment
from court_detection.structured_refinement import (
    CORNER_SPECS,
    SampledCorner,
    SampledLine,
    StructuredHomographyResult,
    _line_from_segment,
    extract_structured_corners,
    line_point_dlt,
)


@dataclass
class StructuredGpuRansacConfig:
    ransac_iter: int = 4096
    line_threshold: float = 0.55
    class_threshold: float = 0.55
    joint_threshold: float = 0.30
    min_pixels_per_class: int = 8
    line_refine_distance_px: float = 8.0
    min_refine_pixels: int = 6
    max_refine_candidates: int = 12000
    n_samples_per_line: int = 80
    score_radius_px: int = 3
    min_projected_coverage: float = 0.12
    inlier_line_score_threshold: float = 0.08
    min_scored_lines: int = 3
    require_baseline_inlier: bool = True
    corner_inlier_threshold_px: float = 8.0
    seed: int = 1430
    dtype: torch.dtype = torch.float32


def fit_homography_from_heatmaps_gpu(
    image_rgb: np.ndarray | torch.Tensor,
    lineness: np.ndarray | torch.Tensor,
    class_probs: np.ndarray | torch.Tensor,
    config: StructuredGpuRansacConfig | None = None,
    line_names: tuple[str, ...] = LINE_NAMES,
    device: torch.device | str | None = None,
) -> StructuredHomographyResult:
    config = StructuredGpuRansacConfig() if config is None else config
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    image_shape = tuple(image_rgb.shape[:2])
    lineness_t = _as_tensor(lineness, device, config.dtype)
    class_t = _as_tensor(class_probs, device, config.dtype)
    if lineness_t.shape != image_shape:
        lineness_t = _resize_2d(lineness_t, image_shape)
    if class_t.shape[-2:] != image_shape:
        class_t = F.interpolate(class_t[None], size=image_shape, mode="bilinear", align_corners=False)[0]

    evidence = torch.clamp(class_t * lineness_t[None], 0.0, 1.0)
    candidates = _candidate_pools(lineness_t, class_t, evidence, config, line_names)
    candidate_counts = {name: int(pool["points"].shape[0]) for name, pool in candidates.items()}
    world_lines = court_lines_world()
    visible = [
        name for name, pool in candidates.items()
        if name in world_lines and pool["points"].shape[0] >= config.min_pixels_per_class
    ]
    if len(visible) < 3:
        return StructuredHomographyResult(
            None, False, -np.inf, candidate_pixels=candidate_counts, message="Not enough line classes"
        )

    generator = torch.Generator(device=device)
    generator.manual_seed(config.seed)
    sampled_lines_t: dict[str, torch.Tensor] = {}
    candidate_points_for_debug: dict[str, torch.Tensor] = {}
    candidate_weights_for_debug: dict[str, torch.Tensor] = {}
    for name in visible:
        class_id = line_names.index(name)
        pool = candidates[name]
        points = pool["points"]
        weights = pool["weights"]
        sampled_lines_t[name] = _sample_and_refine_lines(points, weights, config, generator)
        candidate_points_for_debug[name] = points
        candidate_weights_for_debug[name] = weights

    line_corrs = _batched_line_correspondence_rows(sampled_lines_t)
    point_rows, corner_points, corner_valid = _batched_corner_rows(sampled_lines_t, image_shape)
    if not line_corrs and point_rows is None:
        return StructuredHomographyResult(
            None, False, -np.inf, candidate_pixels=candidate_counts, message="Not enough constraints"
        )

    A_parts = []
    if line_corrs:
        A_parts.extend(line_corrs)
    if point_rows is not None:
        A_parts.append(point_rows)
    A = torch.cat(A_parts, dim=1)
    A = torch.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)
    try:
        _, _, vh = torch.linalg.svd(A)
    except RuntimeError as exc:
        return StructuredHomographyResult(
            None, False, -np.inf, candidate_pixels=candidate_counts, message=f"SVD failed: {exc}"
        )
    H = vh[:, -1].reshape(-1, 3, 3)
    H = H / torch.where(H[:, 2:3, 2:3].abs() > 1e-12, H[:, 2:3, 2:3], torch.ones_like(H[:, 2:3, 2:3]))

    sane = _homography_sanity(H, image_shape)
    score, per_line_scores, coverage = _score_homographies(H, evidence, image_shape, config, line_names)
    inlier_counts = (per_line_scores >= config.inlier_line_score_threshold).sum(dim=1)
    valid = sane & (coverage >= config.min_projected_coverage) & (inlier_counts >= config.min_scored_lines)
    if config.require_baseline_inlier:
        baseline_ids = [
            i for i, name in enumerate(line_names[: evidence.shape[0]])
            if name in ("baseline_left", "baseline_right")
        ]
        if baseline_ids:
            baseline_scores = per_line_scores[:, baseline_ids]
            valid = valid & (baseline_scores >= config.inlier_line_score_threshold).any(dim=1)
    masked_score = torch.where(valid, score, torch.full_like(score, -torch.inf))
    best_score, best_idx_t = masked_score.max(dim=0)
    if not torch.isfinite(best_score):
        return StructuredHomographyResult(
            None,
            False,
            -np.inf,
            candidate_pixels=candidate_counts,
            message="No valid homography",
        )

    best_idx = int(best_idx_t.item())
    best_H = H[best_idx].detach().cpu().numpy()
    sampled_lines = _debug_sampled_lines(
        sampled_lines_t,
        candidate_points_for_debug,
        candidate_weights_for_debug,
        best_idx,
        config,
        line_names,
    )
    corners = tuple(extract_structured_corners(sampled_lines, image_shape))
    best_per_line = {
        name: float(per_line_scores[best_idx, class_id].detach().cpu())
        for class_id, name in enumerate(line_names[: evidence.shape[0]])
    }
    inlier_lines = tuple(name for name, value in best_per_line.items() if value >= config.inlier_line_score_threshold)
    inlier_corners = tuple(_corner_inliers(best_H, corners, config))
    return StructuredHomographyResult(
        H=best_H,
        success=True,
        score=float(best_score.detach().cpu()),
        sampled_lines=sampled_lines,
        corners=corners,
        inlier_lines=inlier_lines,
        inlier_corners=inlier_corners,
        per_line_scores=best_per_line,
        candidate_pixels=candidate_counts,
        message=f"ok ({device.type})",
    )


def _as_tensor(value: np.ndarray | torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _resize_2d(arr: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(arr[None, None], size=shape, mode="bilinear", align_corners=False)[0, 0]


def _candidate_pools(
    lineness: torch.Tensor,
    class_probs: torch.Tensor,
    evidence: torch.Tensor,
    config: StructuredGpuRansacConfig,
    line_names: tuple[str, ...],
) -> dict[str, dict[str, torch.Tensor]]:
    out = {}
    for class_id, name in enumerate(line_names[: class_probs.shape[0]]):
        mask = (
            (lineness >= config.line_threshold)
            & (class_probs[class_id] >= config.class_threshold)
            & (evidence[class_id] >= config.joint_threshold)
        )
        yx = mask.nonzero(as_tuple=False)
        if len(yx):
            points = torch.stack([yx[:, 1], yx[:, 0]], dim=1).to(dtype=config.dtype)
            weights = evidence[class_id, yx[:, 0], yx[:, 1]].clamp_min(0.0)
        else:
            points = torch.empty((0, 2), device=lineness.device, dtype=config.dtype)
            weights = torch.empty((0,), device=lineness.device, dtype=config.dtype)
        out[name] = {"points": points, "weights": weights}
    return out


def _sample_and_refine_lines(
    points: torch.Tensor,
    weights: torch.Tensor,
    config: StructuredGpuRansacConfig,
    generator: torch.Generator,
) -> torch.Tensor:
    probs = weights / weights.sum() if weights.sum() > 0 else torch.full_like(weights, 1.0 / len(weights))
    sample_idx = torch.multinomial(probs, config.ransac_iter * 2, replacement=True, generator=generator).reshape(-1, 2)
    p0 = points[sample_idx[:, 0]]
    p1 = points[sample_idx[:, 1]]
    lines = _normalize_lines(torch.cross(_homog_points(p0), _homog_points(p1), dim=1))
    if points.shape[0] > config.max_refine_candidates:
        refine_idx = torch.multinomial(
            probs, config.max_refine_candidates, replacement=False, generator=generator
        )
        refine_points = points[refine_idx]
        refine_weights = weights[refine_idx]
    else:
        refine_points = points
        refine_weights = weights
    return _weighted_tls_refine(lines, refine_points, refine_weights, config)


def _weighted_tls_refine(
    lines: torch.Tensor,
    points: torch.Tensor,
    weights: torch.Tensor,
    config: StructuredGpuRansacConfig,
) -> torch.Tensor:
    distances = torch.abs(points @ lines[:, :2].T + lines[:, 2])
    support = distances.T <= config.line_refine_distance_px
    support_weights = support.to(lines.dtype) * weights[None]
    sumw = support_weights.sum(dim=1)
    good = sumw >= max(config.min_refine_pixels, 2) * 1e-6
    support_counts = support.sum(dim=1)
    good = good & (support_counts >= config.min_refine_pixels)
    if not bool(good.any()):
        return lines

    centroid = (support_weights @ points) / sumw.clamp_min(1e-6)[:, None]
    centered = points[None] - centroid[:, None]
    cov = (centered * support_weights[:, :, None]).transpose(1, 2) @ centered
    cov = cov / sumw.clamp_min(1e-6)[:, None, None]
    _, evecs = torch.linalg.eigh(cov)
    normal = evecs[:, :, 0]
    c = -(normal * centroid).sum(dim=1, keepdim=True)
    refined = _normalize_lines(torch.cat([normal, c], dim=1))
    return torch.where(good[:, None], refined, lines)


def _homog_points(points: torch.Tensor) -> torch.Tensor:
    return torch.cat([points, torch.ones((points.shape[0], 1), device=points.device, dtype=points.dtype)], dim=1)


def _normalize_lines(lines: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(lines[:, :2], dim=1, keepdim=True)
    out = lines / norm.clamp_min(1e-12)
    sign = torch.where(out[:, 2:3] < 0, -1.0, 1.0)
    out = out * sign
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _batched_line_correspondence_rows(sampled_lines: dict[str, torch.Tensor]) -> list[torch.Tensor]:
    rows = []
    world_lines = court_lines_world()
    for name, li in sampled_lines.items():
        if name not in world_lines:
            continue
        lw = torch.as_tensor(
            _line_from_segment(*[p[:2] for p in world_lines[name]]),
            device=li.device,
            dtype=li.dtype,
        )
        rows.append(_line_rows(lw, li))
    return rows


def _line_rows(lw: torch.Tensor, li: torch.Tensor) -> torch.Tensor:
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


def _batched_corner_rows(
    sampled_lines: dict[str, torch.Tensor],
    image_shape: tuple[int, int],
) -> tuple[torch.Tensor | None, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    all_rows = []
    points = {}
    valid_by_name = {}
    h, w = image_shape
    for name, line_a, line_b, world in CORNER_SPECS:
        if line_a not in sampled_lines or line_b not in sampled_lines:
            continue
        xh = torch.cross(sampled_lines[line_a], sampled_lines[line_b], dim=1)
        xy = xh[:, :2] / _safe_denominator(xh[:, 2:3])
        valid = (
            torch.isfinite(xy).all(dim=1)
            & (xh[:, 2].abs() > 1e-9)
            & (xy[:, 0] >= -80.0)
            & (xy[:, 0] <= w + 80.0)
            & (xy[:, 1] >= -80.0)
            & (xy[:, 1] <= h + 80.0)
        )
        rows = _point_rows(torch.as_tensor(world, device=xy.device, dtype=xy.dtype), xy)
        rows = torch.where(valid[:, None, None], rows, torch.zeros_like(rows))
        all_rows.append(rows)
        points[name] = xy
        valid_by_name[name] = valid
    return (torch.cat(all_rows, dim=1) if all_rows else None), points, valid_by_name


def _point_rows(world: torch.Tensor, image: torch.Tensor) -> torch.Tensor:
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


def _homography_sanity(H: torch.Tensor, image_shape: tuple[int, int]) -> torch.Tensor:
    finite = torch.isfinite(H).flatten(1).all(dim=1)
    det_ok = torch.linalg.det(H).abs() > 1e-10
    corners = torch.as_tensor(
        [[0.0, 0.0], [2800.0, 0.0], [2800.0, 1500.0], [0.0, 1500.0]],
        device=H.device,
        dtype=H.dtype,
    )
    proj = _project(H, corners)
    proj_ok = torch.isfinite(proj).flatten(1).all(dim=1)
    x = proj[:, :, 0]
    y = proj[:, :, 1]
    area = 0.5 * torch.abs((x * torch.roll(y, -1, 1)).sum(dim=1) - (y * torch.roll(x, -1, 1)).sum(dim=1))
    diag = torch.maximum(torch.linalg.norm(proj[:, 0] - proj[:, 2], dim=1), torch.linalg.norm(proj[:, 1] - proj[:, 3], dim=1))
    limit = 8.0 * max(image_shape)
    bounds_ok = proj.abs().flatten(1).max(dim=1).values <= limit
    return finite & det_ok & proj_ok & (area >= 100.0) & (diag >= 30.0) & bounds_ok


def _score_homographies(
    H: torch.Tensor,
    evidence: torch.Tensor,
    image_shape: tuple[int, int],
    config: StructuredGpuRansacConfig,
    line_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    h, w = image_shape
    r = int(config.score_radius_px)
    pooled = F.max_pool2d(evidence[:, None], kernel_size=2 * r + 1, stride=1, padding=r)[:, 0]
    per_line = []
    coverages = []
    world_lines = court_lines_world()
    for class_id, name in enumerate(line_names[: evidence.shape[0]]):
        if name not in world_lines:
            per_line.append(torch.zeros((H.shape[0],), device=H.device, dtype=H.dtype))
            coverages.append(torch.zeros((H.shape[0],), device=H.device, dtype=H.dtype))
            continue
        a, b = world_lines[name]
        world_np = sample_segment(a[:2], b[:2], n=config.n_samples_per_line)
        world = torch.as_tensor(world_np, device=H.device, dtype=H.dtype)
        projected = _project(H, world)
        inside = (
            torch.isfinite(projected).all(dim=2)
            & (projected[:, :, 0] >= 0.0)
            & (projected[:, :, 0] <= w - 1)
            & (projected[:, :, 1] >= 0.0)
            & (projected[:, :, 1] <= h - 1)
        )
        x = torch.round(projected[:, :, 0]).long().clamp(0, w - 1)
        y = torch.round(projected[:, :, 1]).long().clamp(0, h - 1)
        values = pooled[class_id, y, x] * inside.to(H.dtype)
        count = inside.sum(dim=1).clamp_min(1)
        per_line.append(values.sum(dim=1) / count)
        coverages.append(inside.to(H.dtype).mean(dim=1))
    per_line_t = torch.stack(per_line, dim=1)
    coverage_t = torch.stack(coverages, dim=1).mean(dim=1)
    return per_line_t.mean(dim=1) * coverage_t, per_line_t, coverage_t


def _project(H: torch.Tensor, world: torch.Tensor) -> torch.Tensor:
    homog = torch.cat([world, torch.ones((world.shape[0], 1), device=world.device, dtype=world.dtype)], dim=1)
    img_h = torch.einsum("bij,nj->bni", H, homog)
    return img_h[:, :, :2] / _safe_denominator(img_h[:, :, 2:3])


def _safe_denominator(value: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    sign = torch.where(value < 0, -1.0, 1.0)
    return torch.where(value.abs() > eps, value, sign * eps)


def _debug_sampled_lines(
    sampled_lines_t: dict[str, torch.Tensor],
    candidate_points: dict[str, torch.Tensor],
    candidate_weights: dict[str, torch.Tensor],
    best_idx: int,
    config: StructuredGpuRansacConfig,
    line_names: tuple[str, ...],
) -> dict[str, SampledLine]:
    out = {}
    for name, lines in sampled_lines_t.items():
        class_id = line_names.index(name)
        line = lines[best_idx].detach().cpu().numpy()
        pts = candidate_points[name].detach().cpu().numpy()
        weights = candidate_weights[name].detach().cpu().numpy()
        distances = np.abs(pts @ line[:2] + line[2])
        mask = distances <= config.line_refine_distance_px
        support_pts = pts[mask] if int(mask.sum()) >= 2 else pts[:2]
        p0, p1 = _clip_line_endpoints(line, support_pts)
        out[name] = SampledLine(
            name=name,
            class_id=class_id,
            line_homog=line,
            p0=p0,
            p1=p1,
            support=int(mask.sum()),
            score=float(weights[mask].sum()) if mask.any() else 0.0,
        )
    return out


def _clip_line_endpoints(line: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point_on_line = -line[2] * line[:2]
    direction = np.array([-line[1], line[0]], dtype=float)
    t = (points - point_on_line) @ direction
    lo, hi = np.percentile(t, (5.0, 95.0))
    return point_on_line + lo * direction, point_on_line + hi * direction


def _refit_best(
    sampled_lines: dict[str, SampledLine],
    corners: tuple[SampledCorner, ...],
    fallback_H: np.ndarray,
    inlier_lines: tuple[str, ...],
) -> np.ndarray | None:
    try:
        line_corrs = [
            (name, _line_from_segment(*[p[:2] for p in court_lines_world()[name]]), sampled.line_homog)
            for name, sampled in sampled_lines.items()
            if name in court_lines_world() and name in inlier_lines
        ]
        point_corrs = [
            (corner.name, corner.world, corner.image)
            for corner in corners
            if corner.line_a in inlier_lines and corner.line_b in inlier_lines
        ]
        if 2 * len(line_corrs) + 2 * len(point_corrs) < 8:
            return fallback_H
        return line_point_dlt(line_corrs, point_corrs)
    except (ValueError, np.linalg.LinAlgError):
        return fallback_H


def _corner_inliers(
    H: np.ndarray,
    corners: tuple[SampledCorner, ...],
    config: StructuredGpuRansacConfig,
) -> list[str]:
    out = []
    for corner in corners:
        projected = _project_np(H, corner.world[None])[0]
        if np.isfinite(projected).all() and np.linalg.norm(projected - corner.image) <= config.corner_inlier_threshold_px:
            out.append(corner.name)
    return out


def _project_np(H: np.ndarray, world: np.ndarray) -> np.ndarray:
    homog = np.column_stack([world, np.ones(len(world))])
    img_h = (H @ homog.T).T
    return img_h[:, :2] / img_h[:, 2:3]
