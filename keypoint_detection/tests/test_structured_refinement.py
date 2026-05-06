from __future__ import annotations

import unittest

import cv2
import numpy as np
import torch

from court_detection.geometry import STRAIGHT_LINE_NAMES, court_lines_world, sample_segment
from court_detection.structured_refinement import (
    StructuredRansacConfig,
    SampledLine,
    extract_structured_corners,
    fit_homography_from_heatmaps,
    line_point_dlt,
    line_through_two_points,
    project_world_points,
    score_homography_on_heatmaps,
    _line_from_segment,
)
from court_detection.structured_refinement_gpu import (
    StructuredGpuRansacConfig,
    fit_homography_from_heatmaps_gpu,
)


IMAGE_SHAPE = (140, 260)
H_TRUE = np.array([[0.08, 0.005, 12.0], [0.004, 0.06, 18.0], [1.0e-5, 1.0e-5, 1.0]])


def _projected_line(name: str) -> np.ndarray:
    a, b = court_lines_world()[name]
    pts = project_world_points(H_TRUE, np.stack([a[:2], b[:2]]))
    return line_through_two_points(pts[0], pts[1])


def _sampled_lines() -> dict[str, SampledLine]:
    out = {}
    for class_id, name in enumerate(STRAIGHT_LINE_NAMES):
        a, b = court_lines_world()[name]
        pts = project_world_points(H_TRUE, np.stack([a[:2], b[:2]]))
        out[name] = SampledLine(
            name=name,
            class_id=class_id,
            line_homog=line_through_two_points(pts[0], pts[1]),
            p0=pts[0],
            p1=pts[1],
            support=100,
            score=100.0,
        )
    return out


def _draw_synthetic_heatmaps(false_overlay: bool = False) -> tuple[np.ndarray, np.ndarray]:
    h, w = IMAGE_SHAPE
    lineness = np.zeros((h, w), dtype=np.float32)
    class_probs = np.zeros((len(STRAIGHT_LINE_NAMES), h, w), dtype=np.float32)
    for class_id, name in enumerate(STRAIGHT_LINE_NAMES):
        a, b = court_lines_world()[name]
        pts = project_world_points(H_TRUE, sample_segment(a[:2], b[:2], n=120))
        finite = np.isfinite(pts).all(axis=1)
        int_pts = np.round(pts[finite]).astype(np.int32)
        if len(int_pts) >= 2:
            cv2.polylines(lineness, [int_pts.reshape(-1, 1, 2)], False, 1.0, 3, cv2.LINE_AA)
            cv2.polylines(class_probs[class_id], [int_pts.reshape(-1, 1, 2)], False, 1.0, 3, cv2.LINE_AA)
    if false_overlay:
        near_id = STRAIGHT_LINE_NAMES.index("sideline_near")
        cv2.line(lineness, (0, h - 8), (w - 1, h - 8), 1.0, 5, cv2.LINE_AA)
        cv2.line(class_probs[near_id], (0, h - 8), (w - 1, h - 8), 1.0, 5, cv2.LINE_AA)
    return np.clip(lineness, 0.0, 1.0), np.clip(class_probs, 0.0, 1.0)


class StructuredRefinementTest(unittest.TestCase):
    def test_intersections_have_known_corner_coordinates(self) -> None:
        corners = extract_structured_corners(_sampled_lines(), IMAGE_SHAPE, margin_px=200.0)
        by_name = {corner.name: corner for corner in corners}
        self.assertIn("corner_left_far", by_name)
        expected = project_world_points(H_TRUE, np.array([[0.0, 0.0]]))[0]
        np.testing.assert_allclose(by_name["corner_left_far"].image, expected, atol=1.0)
        np.testing.assert_allclose(by_name["corner_left_far"].world, np.array([0.0, 0.0]))

    def test_line_point_dlt_recovers_synthetic_homography(self) -> None:
        sampled = _sampled_lines()
        corners = extract_structured_corners(sampled, IMAGE_SHAPE, margin_px=200.0)
        line_corrs = [
            (name, _line_from_segment(*[p[:2] for p in court_lines_world()[name]]), line.line_homog)
            for name, line in sampled.items()
        ]
        point_corrs = [(corner.name, corner.world, corner.image) for corner in corners]
        H = line_point_dlt(line_corrs, point_corrs)
        world = np.array([[0.0, 0.0], [2800.0, 0.0], [2800.0, 1500.0], [0.0, 1500.0]])
        np.testing.assert_allclose(project_world_points(H, world), project_world_points(H_TRUE, world), atol=1.5)

    def test_line_point_dlt_respects_point_weights(self) -> None:
        anchors = np.array([[0.0, 0.0], [2800.0, 0.0], [2800.0, 1500.0], [0.0, 1500.0]])
        image = project_world_points(H_TRUE, anchors)
        point_corrs = [(f"anchor_{idx}", world, img) for idx, (world, img) in enumerate(zip(anchors, image))]
        noisy_world = np.array([1400.0, 750.0])
        noisy_image = project_world_points(H_TRUE, noisy_world[None])[0] + np.array([30.0, -20.0])
        point_corrs.append(("noisy_center", noisy_world, noisy_image))

        equal_H = line_point_dlt([], point_corrs)
        weighted_H = line_point_dlt(
            [],
            point_corrs,
            point_weights={**{f"anchor_{idx}": 25.0 for idx in range(4)}, "noisy_center": 0.01},
        )

        probe = np.array([[0.0, 0.0], [2800.0, 0.0], [2800.0, 1500.0], [0.0, 1500.0], [1400.0, 750.0]])
        true_projected = project_world_points(H_TRUE, probe)
        equal_error = np.linalg.norm(project_world_points(equal_H, probe) - true_projected, axis=1).mean()
        weighted_error = np.linalg.norm(project_world_points(weighted_H, probe) - true_projected, axis=1).mean()
        self.assertLess(weighted_error, 0.5 * equal_error)

    def test_heatmap_score_prefers_correct_homography(self) -> None:
        lineness, class_probs = _draw_synthetic_heatmaps()
        evidence = lineness[None] * class_probs
        config = StructuredRansacConfig(n_samples_per_line=80, score_radius_px=2)
        shifted = H_TRUE.copy()
        shifted[0, 2] += 20.0
        shifted[1, 2] += 15.0
        true_score, _, _ = score_homography_on_heatmaps(H_TRUE, evidence, IMAGE_SHAPE, config)
        shifted_score, _, _ = score_homography_on_heatmaps(shifted, evidence, IMAGE_SHAPE, config)
        self.assertGreater(true_score, shifted_score + 0.2)

    def test_ransac_rejects_strong_single_class_overlay_line(self) -> None:
        lineness, class_probs = _draw_synthetic_heatmaps(false_overlay=True)
        image = np.zeros((*IMAGE_SHAPE, 3), dtype=np.float32)
        config = StructuredRansacConfig(
            ransac_iter=250,
            line_threshold=0.5,
            class_threshold=0.5,
            joint_threshold=0.25,
            min_pixels_per_class=4,
            line_refine_distance_px=5.0,
            score_radius_px=2,
            seed=1430,
        )
        result = fit_homography_from_heatmaps(image, lineness, class_probs, config)
        self.assertTrue(result.success, result.message)
        world = np.array([[0.0, 0.0], [2800.0, 0.0], [2800.0, 1500.0], [0.0, 1500.0]])
        err = np.linalg.norm(project_world_points(result.H, world) - project_world_points(H_TRUE, world), axis=1).mean()
        self.assertLess(err, 12.0)

    def test_gpu_ransac_runs_on_synthetic_heatmaps(self) -> None:
        lineness, class_probs = _draw_synthetic_heatmaps(false_overlay=True)
        image = np.zeros((*IMAGE_SHAPE, 3), dtype=np.float32)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = StructuredGpuRansacConfig(
            ransac_iter=256,
            line_threshold=0.5,
            class_threshold=0.5,
            joint_threshold=0.25,
            min_pixels_per_class=4,
            line_refine_distance_px=5.0,
            max_refine_candidates=2000,
            score_radius_px=2,
            seed=1430,
        )
        result = fit_homography_from_heatmaps_gpu(image, lineness, class_probs, config, device=device)
        self.assertTrue(result.success, result.message)
        self.assertGreaterEqual(len(result.inlier_lines), 3)


if __name__ == "__main__":
    unittest.main()
