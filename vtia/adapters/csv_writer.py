"""Writes events and counts as CSV (UTF-8, opens fine in Excel and pandas)."""
import csv
from collections import Counter
from pathlib import Path

from vtia.domain.models import CrossingEvent


def write_events(events: list[CrossingEvent], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "track_id", "class", "line", "direction"])
        w.writerows((e.frame, e.track_id, e.road_class, e.line, e.direction) for e in events)


def write_counts(counts: Counter, path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["line", "direction", "class", "count"])
        w.writerows((*key, n) for key, n in sorted(counts.items()))
