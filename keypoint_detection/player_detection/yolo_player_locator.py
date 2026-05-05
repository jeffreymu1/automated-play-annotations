"""Estimate player floor locations from YOLO pose ankle keypoints."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

LEFT_ANKLE = 15
RIGHT_ANKLE = 16
NUM_COCO_KEYPOINTS = 17


@dataclass
class PlayerLocation:
    xy: np.ndarray
    conf: float
    bbox_xyxy: np.ndarray
    keypoints_xy: np.ndarray
    keypoints_conf: np.ndarray
    bbox_conf: float | None = None
    source: str = "bbox_bottom_center"

    def to_json(self) -> dict[str, object]:
        return {
            "xy": self.xy.astype(float).tolist(),
            "conf": float(self.conf),
            "bbox_xyxy": self.bbox_xyxy.astype(float).tolist(),
            "bbox_conf": None if self.bbox_conf is None else float(self.bbox_conf),
            "keypoints_xy": self.keypoints_xy.astype(float).tolist(),
            "keypoints_conf": self.keypoints_conf.astype(float).tolist(),
            "source": self.source,
        }


class YoloPlayerLocator:
    def __init__(
        self,
        model_name: str = "yolo11m-pose.pt",
        device: str | int | None = None,
        imgsz: int = 1280,
        conf: float = 0.15,
        ankle_conf: float = 0.2,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Ultralytics is required for YOLO pose inference. Install it with "
                "`uv add ultralytics` or `uv pip install ultralytics`."
            ) from exc

        self.model = YOLO(model_name)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.ankle_conf = ankle_conf

    def predict_frame(self, frame_bgr: np.ndarray) -> list[PlayerLocation]:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError(f"Expected BGR frame with shape (H, W, 3), got {frame_bgr.shape}")

        results = self.model.predict(
            source=frame_bgr,
            imgsz=self.imgsz,
            conf=self.conf,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        result = results[0]
        boxes = _tensor_to_numpy(result.boxes.xyxy) if result.boxes is not None else np.zeros((0, 4), dtype=np.float32)
        box_confs = None
        if result.boxes is not None and getattr(result.boxes, "conf", None) is not None:
            box_confs = _tensor_to_numpy(result.boxes.conf).reshape(-1)

        keypoints_xy = np.zeros((len(boxes), NUM_COCO_KEYPOINTS, 2), dtype=np.float32)
        keypoints_conf = np.zeros((len(boxes), NUM_COCO_KEYPOINTS), dtype=np.float32)
        if result.keypoints is not None:
            if getattr(result.keypoints, "xy", None) is not None:
                xy = _tensor_to_numpy(result.keypoints.xy).astype(np.float32)
                keypoints_xy[: len(xy), : xy.shape[1], :] = xy[:, :NUM_COCO_KEYPOINTS, :]
            if getattr(result.keypoints, "conf", None) is not None and result.keypoints.conf is not None:
                kconf = _tensor_to_numpy(result.keypoints.conf).astype(np.float32)
                keypoints_conf[: len(kconf), : kconf.shape[1]] = kconf[:, :NUM_COCO_KEYPOINTS]

        detections: list[PlayerLocation] = []
        for i, bbox in enumerate(boxes.astype(np.float32)):
            xy, estimate_conf, source = _ground_point_from_ankles(
                bbox,
                keypoints_xy[i],
                keypoints_conf[i],
                self.ankle_conf,
            )
            bbox_conf = None if box_confs is None or i >= len(box_confs) else float(box_confs[i])
            detections.append(
                PlayerLocation(
                    xy=xy,
                    conf=estimate_conf,
                    bbox_xyxy=bbox,
                    keypoints_xy=keypoints_xy[i],
                    keypoints_conf=keypoints_conf[i],
                    bbox_conf=bbox_conf,
                    source=source,
                )
            )
        return detections


def _ground_point_from_ankles(
    bbox_xyxy: np.ndarray,
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray,
    ankle_conf: float,
) -> tuple[np.ndarray, float, str]:
    left_ok = bool(keypoints_conf[LEFT_ANKLE] >= ankle_conf)
    right_ok = bool(keypoints_conf[RIGHT_ANKLE] >= ankle_conf)

    if left_ok and right_ok:
        xy = (keypoints_xy[LEFT_ANKLE] + keypoints_xy[RIGHT_ANKLE]) * 0.5
        conf = float((keypoints_conf[LEFT_ANKLE] + keypoints_conf[RIGHT_ANKLE]) * 0.5)
        return xy.astype(np.float32), conf, "ankle_midpoint"
    if left_ok:
        return keypoints_xy[LEFT_ANKLE].astype(np.float32), float(keypoints_conf[LEFT_ANKLE]), "left_ankle"
    if right_ok:
        return keypoints_xy[RIGHT_ANKLE].astype(np.float32), float(keypoints_conf[RIGHT_ANKLE]), "right_ankle"

    x1, _, x2, y2 = bbox_xyxy
    return np.array([(x1 + x2) * 0.5, y2], dtype=np.float32), 0.0, "bbox_bottom_center"


def draw_player_locations(frame_bgr: np.ndarray, players: list[PlayerLocation]) -> np.ndarray:
    annotated = frame_bgr.copy()
    for i, player in enumerate(players):
        x1, y1, x2, y2 = np.rint(player.bbox_xyxy).astype(int)
        x, y = np.rint(player.xy).astype(int)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.circle(annotated, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            annotated,
            f"{i}: {player.conf:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            annotated,
            f"{i}: {player.conf:.2f}",
            (x1, max(15, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return annotated


def apply_homography(points_xy: np.ndarray, H: np.ndarray) -> np.ndarray:
    points_xy = np.asarray(points_xy)
    H = np.asarray(H)
    points_h = np.concatenate(
        [points_xy, np.ones((len(points_xy), 1), dtype=points_xy.dtype)],
        axis=1,
    )
    warped_h = points_h @ H.T
    return warped_h[:, :2] / warped_h[:, 2:3]


def _tensor_to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)
