"""Run: python -m unittest discover -s tests"""
import unittest

from vtia.domain.classes import RoadClass
from vtia.domain.counting import LineCounter, count_by
from vtia.domain.models import Detection, Line

LINE = Line("main", (0, 100), (200, 100), to_right="down", to_left="up")


def det(frame, x, y, cls=RoadClass.CAR, track=1):
    """Detection whose anchor (bottom-centre) is (x, y)."""
    return Detection(frame, cls, 0.9, (x - 10, y - 20, x + 10, y), track)


def run(*dets):
    counter = LineCounter([LINE])
    return [e for d in dets for e in counter.update([d])]


class LineCounterTest(unittest.TestCase):
    def test_crossing_down_and_up(self):
        self.assertEqual([e.direction for e in run(det(0, 50, 90), det(1, 50, 110))], ["down"])
        self.assertEqual([e.direction for e in run(det(0, 50, 110), det(1, 50, 90))], ["up"])

    def test_passing_beside_the_segment_is_ignored(self):
        self.assertEqual(run(det(0, 300, 90), det(1, 300, 110)), [])

    def test_jitter_on_the_line_counts_once(self):
        events = run(det(0, 50, 95), det(1, 50, 105), det(2, 50, 95), det(3, 50, 105))
        self.assertEqual(len(events), 1)

    def test_stopping_exactly_on_the_line(self):
        events = run(det(0, 50, 90), det(1, 50, 100), det(2, 50, 110))
        self.assertEqual([(e.frame, e.direction) for e in events], [(2, "down")])

    def test_class_is_track_majority(self):
        # Detector flickers to "truck" exactly on the crossing frame; the track is mostly "car".
        events = run(det(0, 50, 80), det(1, 50, 85), det(2, 50, 90), det(3, 50, 110, RoadClass.TRUCK))
        self.assertEqual(events[0].road_class, RoadClass.CAR)

    def test_tracks_are_independent(self):
        events = run(det(0, 50, 90, track=1), det(0, 150, 90, track=2), det(1, 50, 110, track=1),
                     det(1, 150, 110, track=2))
        self.assertEqual(count_by(events)[("main", "down", RoadClass.CAR)], 2)


if __name__ == "__main__":
    unittest.main()
