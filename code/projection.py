import cv2
import numpy as np

from .field import FieldHomography


def project_image_point_to_field(
    image_point: tuple[float, float],
    homography: FieldHomography,
) -> tuple[float, float]:
    pts = np.array([[image_point]], dtype=np.float32)
    out = cv2.perspectiveTransform(pts, homography.image_to_field)
    x, y = out[0, 0].tolist()
    return float(x), float(y)

