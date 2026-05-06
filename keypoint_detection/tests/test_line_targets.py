from __future__ import annotations

import numpy as np

from court_detection.geometry import MARKING_CLASS_NAMES
from court_detection.lines import _render_line_targets, _soft_occlusion_lineness_weight


def test_render_line_targets_merges_geometries_that_share_a_class() -> None:
    y = np.linspace(10.0, 90.0, 100, dtype=np.float32)
    lines_uv = {
        "baseline_left": np.column_stack([np.full_like(y, 20.0), y]),
        "baseline_right": np.column_stack([np.full_like(y, 80.0), y]),
    }

    lineness, class_target, visible = _render_line_targets(
        lines_uv,
        image_size=(100, 100),
        output_stride=1,
        sigma=1.5,
        line_names=MARKING_CLASS_NAMES,
    )

    baseline_idx = MARKING_CLASS_NAMES.index("baseline")
    assert visible[baseline_idx]
    assert lineness[50, 20] > 0.9
    assert lineness[50, 80] > 0.9
    assert class_target[50, 20] == baseline_idx
    assert class_target[50, 80] == baseline_idx


def test_render_line_targets_suppresses_lineness_outside_visible_mask() -> None:
    y = np.linspace(10.0, 90.0, 100, dtype=np.float32)
    lines_uv = {
        "halfcourt": np.column_stack([np.full_like(y, 50.0), y]),
    }
    visible_mask = np.ones((100, 100), dtype=bool)
    visible_mask[48:53, 48:53] = False

    lineness, _, visible = _render_line_targets(
        lines_uv,
        image_size=(100, 100),
        output_stride=1,
        sigma=1.5,
        line_names=MARKING_CLASS_NAMES,
        visible_mask=visible_mask,
    )

    halfcourt_idx = MARKING_CLASS_NAMES.index("halfcourt")
    assert visible[halfcourt_idx]
    assert lineness[50, 50] == 0.0
    assert lineness[60, 50] > 0.9


def test_soft_occlusion_lineness_weight_uses_lineness_sigma() -> None:
    occlusion = np.zeros((100, 100), dtype=bool)
    occlusion[50, 50] = True

    weight = _soft_occlusion_lineness_weight(occlusion, output_stride=1, sigma=2.0)

    assert weight[50, 50] == 0.0
    assert 0.0 < weight[50, 51] < weight[50, 54] < 1.0
    assert weight[10, 10] > 0.99
