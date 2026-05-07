from __future__ import annotations

import unittest

import numpy as np

from court_detection.geometry import (
    BASKET_CENTER_FROM_ENDLINE_CM,
    COURT_LENGTH_CM,
    COURT_WIDTH_CM,
    FREE_THROW_CIRCLE_RADIUS_CM,
    FREE_THROW_LINE_X_CM,
    MARKING_CLASS_BY_GEOMETRY,
    THREE_POINT_RADIUS_CM,
    court_lines_world,
)
from court_detection.marking_refinement import (
    MarkingRefinementConfig,
    MarkingCurveFit,
    MarkingLineFit,
    _conic_line_intersection_specs,
    _conic_line_intersections,
    _conic_line_point_correspondences,
    _conic_tangent_point_correspondences,
    _conic_tangent_point_specs,
    _conic_tangent_points,
    _dlt_point_weights,
    _line_through,
    _line_distances,
    _select_tangent_point_for_guide,
    _select_tangent_guide_line,
    _torch_ransac_homography,
    _world_line,
)
from court_detection.structured_refinement import line_point_dlt, project_world_points


IMAGE_SHAPE = (140, 260)
H_TRUE = np.array([[0.08, 0.005, 12.0], [0.004, 0.06, 18.0], [1.0e-5, 1.0e-5, 1.0]])


def _project_line(name: str) -> np.ndarray:
    a, b = court_lines_world()[name]
    pts = project_world_points(H_TRUE, np.stack([a[:2], b[:2]]))
    return _line_through(pts[0], pts[1])


def _circle_conic(center: np.ndarray, radius: float) -> np.ndarray:
    cx, cy = center
    conic = np.array(
        [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [-cx, -cy, cx * cx + cy * cy - radius * radius]],
        dtype=float,
    )
    return conic / np.linalg.norm(conic)


def _project_conic(conic_world: np.ndarray) -> np.ndarray:
    h_inv = np.linalg.inv(H_TRUE)
    conic_image = h_inv.T @ conic_world @ h_inv
    return conic_image / np.linalg.norm(conic_image)


def _line_fit(name: str) -> MarkingLineFit:
    line = _project_line(name)
    a, b = court_lines_world()[name]
    pts = project_world_points(H_TRUE, np.stack([a[:2], b[:2]]))
    return MarkingLineFit(
        class_name=MARKING_CLASS_BY_GEOMETRY[name],
        geometry_name=name,
        line_homog=line,
        p0=pts[0],
        p1=pts[1],
        support=100,
        score=100.0,
    )


def _curve_fit(name: str) -> MarkingCurveFit:
    mid_y = COURT_WIDTH_CM / 2.0
    if name == "free_throw_circle_left":
        center = np.array([FREE_THROW_LINE_X_CM, mid_y], dtype=float)
        radius = FREE_THROW_CIRCLE_RADIUS_CM
    elif name == "free_throw_circle_right":
        center = np.array([COURT_LENGTH_CM - FREE_THROW_LINE_X_CM, mid_y], dtype=float)
        radius = FREE_THROW_CIRCLE_RADIUS_CM
    elif name == "three_point_arc_left":
        center = np.array([BASKET_CENTER_FROM_ENDLINE_CM, mid_y], dtype=float)
        radius = THREE_POINT_RADIUS_CM
    elif name == "three_point_arc_right":
        center = np.array([COURT_LENGTH_CM - BASKET_CENTER_FROM_ENDLINE_CM, mid_y], dtype=float)
        radius = THREE_POINT_RADIUS_CM
    else:
        raise ValueError(name)
    conic = _project_conic(_circle_conic(center, radius))
    return MarkingCurveFit(
        class_name=MARKING_CLASS_BY_GEOMETRY[name],
        geometry_name=name,
        conic=conic,
        circle_center=np.zeros(2, dtype=float),
        circle_radius=0.0,
        points=np.empty((0, 2), dtype=float),
        support=100,
        score=100.0,
    )


def _pair_cost(a: np.ndarray, b: np.ndarray) -> float:
    return min(
        float(np.linalg.norm(a - b, axis=1).sum()),
        float(np.linalg.norm(a - b[::-1], axis=1).sum()),
    )


class MarkingRefinementConicTest(unittest.TestCase):
    def test_conic_line_intersections_match_projected_world_points(self) -> None:
        conic = _curve_fit("free_throw_circle_left").conic
        image_line = _project_line("foul_left")
        actual = _conic_line_intersections(conic, image_line)
        self.assertIsNotNone(actual)
        expected = project_world_points(H_TRUE, np.array([[580.0, 570.0], [580.0, 930.0]]))
        self.assertLess(_pair_cost(actual, expected), 1e-5)

    def test_conic_line_intersections_reject_non_intersecting_line(self) -> None:
        conic = _curve_fit("free_throw_circle_left").conic
        image_line = _project_line("sideline_far")
        self.assertIsNone(_conic_line_intersections(conic, image_line))

    def test_three_point_virtual_world_intersections_use_baselines(self) -> None:
        specs = {name: points for name, _, _, points in _conic_line_intersection_specs()}
        left = specs["three_point_arc_left__baseline_left"]
        right = specs["three_point_arc_right__baseline_right"]
        dy = np.sqrt(THREE_POINT_RADIUS_CM ** 2 - BASKET_CENTER_FROM_ENDLINE_CM ** 2)
        expected_y = np.array([COURT_WIDTH_CM / 2.0 - dy, COURT_WIDTH_CM / 2.0 + dy])
        np.testing.assert_allclose(left[:, 0], np.array([0.0, 0.0]))
        np.testing.assert_allclose(right[:, 0], np.array([COURT_LENGTH_CM, COURT_LENGTH_CM]))
        np.testing.assert_allclose(left[:, 1], expected_y)
        np.testing.assert_allclose(right[:, 1], expected_y)

    def test_conic_line_correspondences_feed_dlt_initializer(self) -> None:
        selected = {
            name: _line_fit(name)
            for name in ("baseline_left", "baseline_right", "foul_left", "foul_right", "sideline_far", "sideline_near")
        }
        curve_fits = {
            "free_throw_circle": [_curve_fit("free_throw_circle_left"), _curve_fit("free_throw_circle_right")],
            "three_point_arc": [_curve_fit("three_point_arc_left"), _curve_fit("three_point_arc_right")],
        }
        point_corrs = _conic_line_point_correspondences(selected, curve_fits, IMAGE_SHAPE, H=H_TRUE)
        tangent_corrs = _conic_tangent_point_correspondences(selected, curve_fits, IMAGE_SHAPE, H=H_TRUE)
        self.assertEqual(len(point_corrs), 4)
        self.assertEqual(len(tangent_corrs), 4)
        line_corrs = [(name, _world_line(name), fit.line_homog) for name, fit in selected.items()]
        H = line_point_dlt(line_corrs, point_corrs + tangent_corrs)
        corners = np.array([[0.0, 0.0], [2800.0, 0.0], [2800.0, 1500.0], [0.0, 1500.0]])
        err = np.linalg.norm(project_world_points(H, corners) - project_world_points(H_TRUE, corners), axis=1).mean()
        self.assertLess(err, 0.2)

    def test_curve_correspondences_receive_dlt_priority_weight(self) -> None:
        point_corrs = [
            ("baseline_left__sideline_far", np.zeros(2), np.zeros(2)),
            ("three_point_arc_left__baseline_left_0", np.zeros(2), np.zeros(2)),
            ("free_throw_circle_left__baseline_tangent_apex", np.zeros(2), np.zeros(2)),
        ]
        weights = _dlt_point_weights(point_corrs, MarkingRefinementConfig(dlt_curve_point_weight=5.0))
        self.assertNotIn("baseline_left__sideline_far", weights)
        self.assertEqual(weights["three_point_arc_left__baseline_left_0"], 5.0)
        self.assertEqual(weights["free_throw_circle_left__baseline_tangent_apex"], 5.0)

    def test_tangent_apex_world_points_use_baseline_direction(self) -> None:
        specs = {name: world for name, _, _, world in _conic_tangent_point_specs()}
        np.testing.assert_allclose(specs["three_point_arc_left__baseline_tangent_apex"], np.array([832.5, 750.0]))
        np.testing.assert_allclose(specs["free_throw_circle_left__baseline_tangent_apex"], np.array([400.0, 750.0]))
        np.testing.assert_allclose(specs["three_point_arc_right__baseline_tangent_apex"], np.array([1967.5, 750.0]))
        np.testing.assert_allclose(specs["free_throw_circle_right__baseline_tangent_apex"], np.array([2400.0, 750.0]))

    def test_tangent_guide_prefers_more_confident_halfcourt_line(self) -> None:
        baseline = _line_fit("baseline_left")
        baseline.score = 10.0
        halfcourt = _line_fit("halfcourt")
        halfcourt.score = 50.0

        guide_name, guide_fit = _select_tangent_guide_line(
            {"baseline_left": baseline, "halfcourt": halfcourt},
            "baseline_left",
        )

        self.assertEqual(guide_name, "halfcourt")
        self.assertIs(guide_fit, halfcourt)

    def test_tangent_correspondences_can_use_halfcourt_when_baseline_missing(self) -> None:
        selected = {"halfcourt": _line_fit("halfcourt")}
        curve_fits = {
            "free_throw_circle": [_curve_fit("free_throw_circle_left")],
            "three_point_arc": [_curve_fit("three_point_arc_left")],
        }

        corrs = _conic_tangent_point_correspondences(selected, curve_fits, IMAGE_SHAPE, H=H_TRUE)

        self.assertEqual({name for name, _, _ in corrs}, {
            "three_point_arc_left__halfcourt_tangent_apex",
            "free_throw_circle_left__halfcourt_tangent_apex",
        })

    def test_tangent_point_selection_uses_closest_baseline_and_farthest_halfcourt(self) -> None:
        points = _conic_tangent_points(
            _curve_fit("free_throw_circle_left").conic,
            np.array([-_project_line("baseline_left")[1], _project_line("baseline_left")[0]], dtype=float),
        )
        self.assertIsNotNone(points)

        baseline = _project_line("baseline_left")
        halfcourt = _project_line("halfcourt")
        baseline_point = _select_tangent_point_for_guide(points, baseline, "free_throw_circle_left", "baseline_left")
        halfcourt_point = _select_tangent_point_for_guide(points, halfcourt, "free_throw_circle_left", "halfcourt")

        self.assertAlmostEqual(float(_line_distances(baseline, baseline_point[None])[0]), float(_line_distances(baseline, points).min()))
        self.assertAlmostEqual(float(_line_distances(halfcourt, halfcourt_point[None])[0]), float(_line_distances(halfcourt, points).max()))

    def test_torch_ransac_uses_systematic_dlt_candidates(self) -> None:
        geometry_candidates = {
            name: [_line_fit(name)]
            for name in ("baseline_left", "baseline_right", "sideline_far", "sideline_near", "halfcourt")
        }
        evidence = np.ones((len(MARKING_CLASS_BY_GEOMETRY), *IMAGE_SHAPE), dtype=np.float32)
        config = MarkingRefinementConfig(
            use_torch_ransac=True,
            max_excluded_primitives=0,
            min_projected_coverage=0.01,
            min_scored_markings=1,
            require_baseline=False,
        )

        result = _torch_ransac_homography(
            geometry_candidates,
            {},
            evidence,
            None,
            None,
            IMAGE_SHAPE,
            config,
            tuple(dict.fromkeys(MARKING_CLASS_BY_GEOMETRY.values())),
        )

        self.assertIsNotNone(result)
        assert result is not None
        H, _, selected, _, _, message = result
        self.assertEqual(set(selected), set(geometry_candidates))
        self.assertIn("systematic dlt", message)
        corners = np.array([[0.0, 0.0], [2800.0, 0.0], [2800.0, 1500.0], [0.0, 1500.0]])
        err = np.linalg.norm(project_world_points(H, corners) - project_world_points(H_TRUE, corners), axis=1).mean()
        self.assertLess(err, 0.2)


if __name__ == "__main__":
    unittest.main()
