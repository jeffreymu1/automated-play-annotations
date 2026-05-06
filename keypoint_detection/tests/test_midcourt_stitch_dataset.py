from __future__ import annotations

import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from court_detection.geometry import CameraCalibration
from court_detection.midcourt_stitch_dataset import (
    average_virtual_calibration,
    virtual_calibration_between,
)


def _calibration(
    *,
    K: np.ndarray,
    center: np.ndarray,
    rotation: np.ndarray,
    width: int,
    height: int,
) -> CameraCalibration:
    return CameraCalibration(
        K=K.astype(float),
        R=rotation.astype(float),
        T=-(rotation @ center.astype(float)),
        kc=np.zeros(5, dtype=float),
        width=width,
        height=height,
    )


class MidcourtVirtualCalibrationTest(unittest.TestCase):
    def test_camera_portion_endpoints_match_source_views(self) -> None:
        left_R = np.eye(3)
        right_R = Rotation.from_euler("z", 15.0, degrees=True).as_matrix()
        left = _calibration(
            K=np.array([[800.0, 0.0, 320.0], [0.0, 810.0, 240.0], [0.0, 0.0, 1.0]]),
            center=np.array([100.0, 200.0, -900.0]),
            rotation=left_R,
            width=640,
            height=480,
        )
        right = _calibration(
            K=np.array([[1000.0, 0.0, 400.0], [0.0, 990.0, 250.0], [0.0, 0.0, 1.0]]),
            center=np.array([500.0, 220.0, -920.0]),
            rotation=right_R,
            width=800,
            height=500,
        )

        left_view = virtual_calibration_between(left, right, camera_portion=0.0)
        right_view = virtual_calibration_between(left, right, camera_portion=1.0)

        np.testing.assert_allclose(left_view.K, left.K)
        np.testing.assert_allclose(left_view.R, left.R)
        np.testing.assert_allclose(left_view.T, left.T)
        self.assertEqual(left_view.width, left.width)
        self.assertEqual(left_view.height, left.height)

        np.testing.assert_allclose(right_view.K, right.K)
        np.testing.assert_allclose(right_view.R, right.R)
        np.testing.assert_allclose(right_view.T, right.T)
        self.assertEqual(right_view.width, right.width)
        self.assertEqual(right_view.height, right.height)

    def test_average_virtual_calibration_keeps_midpoint_behavior(self) -> None:
        left = _calibration(
            K=np.array([[800.0, 0.0, 320.0], [0.0, 800.0, 240.0], [0.0, 0.0, 1.0]]),
            center=np.array([0.0, 0.0, -1000.0]),
            rotation=np.eye(3),
            width=640,
            height=480,
        )
        right = _calibration(
            K=np.array([[900.0, 0.0, 360.0], [0.0, 900.0, 260.0], [0.0, 0.0, 1.0]]),
            center=np.array([200.0, 0.0, -1000.0]),
            rotation=Rotation.from_euler("y", 10.0, degrees=True).as_matrix(),
            width=720,
            height=520,
        )

        midpoint = virtual_calibration_between(left, right, camera_portion=0.5)
        averaged = average_virtual_calibration(left, right)

        np.testing.assert_allclose(averaged.K, midpoint.K)
        np.testing.assert_allclose(averaged.R, midpoint.R)
        np.testing.assert_allclose(averaged.T, midpoint.T)
        self.assertEqual(averaged.width, midpoint.width)
        self.assertEqual(averaged.height, midpoint.height)

    def test_camera_portion_rejects_out_of_range_values(self) -> None:
        calib = _calibration(
            K=np.eye(3),
            center=np.zeros(3),
            rotation=np.eye(3),
            width=10,
            height=10,
        )

        with self.assertRaises(ValueError):
            virtual_calibration_between(calib, calib, camera_portion=-0.1)
        with self.assertRaises(ValueError):
            virtual_calibration_between(calib, calib, camera_portion=1.1)


if __name__ == "__main__":
    unittest.main()
