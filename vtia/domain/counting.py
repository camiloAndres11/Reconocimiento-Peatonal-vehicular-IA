"""Line-crossing counting: turns per-frame tracked detections into events."""
from collections import Counter, defaultdict
from collections.abc import Iterable

from vtia.domain.models import CrossingEvent, Detection, Line, Point


def _side(a: Point, b: Point, p: Point) -> float:
    """Cross product: > 0 if p is right of a->b on screen (y grows down), < 0 left, 0 on it."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


class LineCounter:
    """Stateful counter fed frame by frame.

    A crossing needs the anchor to change side between two off-line positions,
    with the movement passing through the segment (not its extension).
    The class of an event is the track's majority class so far, which absorbs
    frame-to-frame flicker (car/truck) from the detector.
    """

    def __init__(self, lines: list[Line]):
        self.lines = lines
        self._last: dict[tuple[int, str], tuple[bool, Point]] = {}  # (track, line) -> (right side?, anchor)
        self._counted: set[tuple[int, str]] = set()
        self._votes: defaultdict[int, Counter] = defaultdict(Counter)

    def update(self, detections: Iterable[Detection]) -> list[CrossingEvent]:
        events = []
        for d in detections:
            self._votes[d.track_id][d.road_class] += 1
            p = d.anchor
            for line in self.lines:
                s = _side(line.start, line.end, p)
                if s == 0:
                    continue  # on the line: wait until it leaves to decide
                key = (d.track_id, line.name)
                prev = self._last.get(key)
                self._last[key] = (s > 0, p)
                if prev is None or prev[0] == (s > 0) or key in self._counted:
                    continue
                # Endpoints of the line on opposite sides of the movement -> crossed within the segment.
                if _side(prev[1], p, line.start) * _side(prev[1], p, line.end) > 0:
                    continue
                # ponytail: one count per track per line, so jitter on the line doesn't double count;
                # an object that genuinely crosses back is missed. Revisit if dwell/return trips matter.
                self._counted.add(key)
                road_class = self._votes[d.track_id].most_common(1)[0][0]
                direction = line.to_right if s > 0 else line.to_left
                events.append(CrossingEvent(d.frame, d.track_id, road_class, line.name, direction))
        return events


def count_by(events: Iterable[CrossingEvent]) -> Counter:
    """Totals keyed by (line, direction, class)."""
    return Counter((e.line, e.direction, e.road_class) for e in events)
