from dataclasses import dataclass


@dataclass
class Detection:
    cls_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]

    @property
    def center(self) -> tuple[float, float]:
        x1, y1, x2, y2 = self.xyxy
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


class YoloDetector:
    def __init__(self, model_name: str, conf: float, iou: float) -> None:
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou

    def detect(self, frame) -> list[Detection]:
        result = self.model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        names = result.names
        detections: list[Detection] = []
        for box in result.boxes:
            cls_idx = int(box.cls.item())
            cls_name = names.get(cls_idx, str(cls_idx))
            xyxy = tuple(float(v) for v in box.xyxy[0].tolist())
            detections.append(
                Detection(
                    cls_name=cls_name,
                    confidence=float(box.conf.item()),
                    xyxy=xyxy,
                )
            )
        return detections

