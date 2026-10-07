"""Domain entities. Pure data: no I/O, no third-party libraries."""
from dataclasses import dataclass

from vtia.domain.classes import RoadClass

Point = tuple[float, float]


@dataclass(frozen=True)
class Detection:
    """One tracked object in one frame. `box` is (x1, y1, x2, y2) in pixels."""
    frame: int
    track_id: int
    road_class: RoadClass
    confidence: float
    box: tuple[float, float, float, float]

    @property
    def anchor(self) -> Point:
        """Bottom-centre of the box: where the object touches the ground."""
        x1, _, x2, y2 = self.box
        return (x1 + x2) / 2, y2


@dataclass(frozen=True)
class Line:
    """Virtual counting line from `start` to `end`, in pixels.

    Walking from start to end as seen on screen, an object that crosses to the
    right-hand side is labelled `to_right`, otherwise `to_left`. Example: for a
    horizontal line drawn left to right, `to_right` means moving down the image.
    """
    name: str
    start: Point
    end: Point
    to_right: str
    to_left: str


@dataclass(frozen=True)
class CrossingEvent:
    frame: int
    track_id: int
    road_class: RoadClass
    line: str
    direction: str
