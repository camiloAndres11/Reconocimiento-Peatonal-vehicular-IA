"""Use case: count road users crossing virtual lines in a video."""
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

from vtia.domain.counting import LineCounter
from vtia.domain.models import CrossingEvent, Detection, Line


class Tracker(Protocol):
    """Port: any detector + tracker that yields the tracked detections of each frame.

    Implementations: Ultralytics (YOLO, RT-DETR); Faster R-CNN arrives in phase 3.
    """

    def track(self, video: Path) -> Iterator[list[Detection]]: ...


def process_video(video: Path, tracker: Tracker, lines: list[Line]) -> tuple[list[CrossingEvent], int]:
    """Returns the crossing events and the number of frames processed."""
    counter = LineCounter(lines)
    events: list[CrossingEvent] = []
    frames = 0
    for detections in tracker.track(video):
        events += counter.update(detections)
        frames += 1
    return events, frames
