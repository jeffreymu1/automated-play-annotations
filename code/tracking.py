from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .detectors import Detection


@dataclass
class TrackState:
    track_id: int
    cls_name: str
    position: tuple[float, float]
    stale: int = 0


class CentroidTracker:
    def __init__(self, max_distance: float = 45.0, max_stale: int = 20) -> None:
        self.max_distance = max_distance
        self.max_stale = max_stale
        self._next_id = 1
        self._tracks: dict[int, TrackState] = {}

    def update(self, detections: list[Detection]) -> list[TrackState]:
        by_class: dict[str, list[Detection]] = defaultdict(list)
        for det in detections:
            by_class[det.cls_name].append(det)

        assigned: set[int] = set()
        for cls_name, cls_dets in by_class.items():
            cls_tracks = [t for t in self._tracks.values() if t.cls_name == cls_name]
            for det in cls_dets:
                det_center = np.array(det.center, dtype=float)
                best_id = None
                best_dist = float("inf")
                for tr in cls_tracks:
                    if tr.track_id in assigned:
                        continue
                    dist = np.linalg.norm(det_center - np.array(tr.position, dtype=float))
                    if dist < best_dist:
                        best_dist = dist
                        best_id = tr.track_id

                if best_id is not None and best_dist <= self.max_distance:
                    self._tracks[best_id].position = det.center
                    self._tracks[best_id].stale = 0
                    assigned.add(best_id)
                else:
                    tid = self._next_id
                    self._next_id += 1
                    self._tracks[tid] = TrackState(track_id=tid, cls_name=cls_name, position=det.center)
                    assigned.add(tid)

        stale_ids = []
        for tid, tr in self._tracks.items():
            if tid not in assigned:
                tr.stale += 1
            if tr.stale > self.max_stale:
                stale_ids.append(tid)
        for tid in stale_ids:
            del self._tracks[tid]

        return list(self._tracks.values())

