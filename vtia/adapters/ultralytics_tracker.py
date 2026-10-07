"""Ultralytics adapter: YOLO + ByteTrack/BoT-SORT -> domain detections."""
from collections.abc import Iterator
from pathlib import Path

from ultralytics import YOLO

from vtia.domain.classes import COCO_TO_ROAD
from vtia.domain.models import Detection


class UltralyticsTracker:
    def __init__(self, weights: str = "yolo26n.pt", tracker: str = "bytetrack.yaml",
                 conf: float | None = None, device: str | None = None):
        self.model = YOLO(weights)
        self.tracker = tracker
        self.conf = conf  # None -> Ultralytics' tracking default
        self.device = device  # None -> auto (CPU here, GPU on Colab)
        # Model class id -> project class; ids outside the six classes are never detected.
        self.class_map = {i: COCO_TO_ROAD[n] for i, n in self.model.names.items() if n in COCO_TO_ROAD}

    def track(self, video: Path) -> Iterator[list[Detection]]:
        results = self.model.track(
            str(video), stream=True, persist=True, tracker=self.tracker,
            classes=list(self.class_map), conf=self.conf, device=self.device, verbose=False,
        )
        for frame, r in enumerate(results):
            boxes = r.boxes
            if boxes.id is None:  # nothing tracked in this frame
                yield []
                continue
            yield [
                Detection(frame, int(t), self.class_map[int(c)], float(s), tuple(xyxy))
                for t, c, s, xyxy in zip(boxes.id.tolist(), boxes.cls.tolist(),
                                         boxes.conf.tolist(), boxes.xyxy.tolist())
            ]
