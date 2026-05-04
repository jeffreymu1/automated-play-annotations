from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FieldHomography:
    image_to_field: np.ndarray
    field_to_image: np.ndarray


def build_homography_from_corners(
    image_corners: list[tuple[float, float]],
    field_width_m: float,
    field_height_m: float,
) -> FieldHomography:
    # image_corners: TL, TR, BR, BL
    if len(image_corners) != 4:
        raise ValueError("Exactly 4 corners required.")

    src = np.array(image_corners, dtype=np.float32)
    dst = np.array(
        [
            [0.0, 0.0],
            [field_width_m, 0.0],
            [field_width_m, field_height_m],
            [0.0, field_height_m],
        ],
        dtype=np.float32,
    )

    image_to_field = cv2.getPerspectiveTransform(src, dst)
    field_to_image = cv2.getPerspectiveTransform(dst, src)
    return FieldHomography(image_to_field=image_to_field, field_to_image=field_to_image)
