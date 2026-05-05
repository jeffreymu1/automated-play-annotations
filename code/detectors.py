from dataclasses import dataclass


def _indices_for_substrings(names: dict, substrings: list[str]) -> list[int] | None:
    if not substrings:
        return None
    out: list[int] = []
    for idx_raw, nm in sorted(names.items(), key=lambda kv: int(str(kv[0]))):
        label = str(nm).lower()
        if any(ss.strip().lower() in label for ss in substrings if ss.strip()):
            out.append(int(str(idx_raw)))
    return out


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
    def __init__(
        self,
        model_name: str,
        conf: float,
        iou: float,
        *,
        class_substrings: list[str] | None = None,
    ) -> None:
        from rich import print
        from ultralytics import YOLO

        self.model = YOLO(model_name)
        self.conf = conf
        self.iou = iou
        self._class_idxs: list[int] | None = _indices_for_substrings(
            self.model.names or {}, class_substrings or []
        )
        if class_substrings and not self._class_idxs:
            print(
                "[yellow]YOLO class filter matched nothing; detecting all classes.[/yellow]",
            )

    def detect(self, frame) -> list[Detection]:
        kwargs: dict = {"conf": self.conf, "iou": self.iou, "verbose": False}
        if self._class_idxs:
            kwargs["classes"] = self._class_idxs
        result = self.model(frame, **kwargs)[0]
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

