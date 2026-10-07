"""Reads a camera-point YAML (see configs/points/) into domain lines."""
from pathlib import Path

import yaml

from vtia.domain.models import Line

_EXPECTED = "lines: [{name, start: [x, y], end: [x, y], to_right, to_left}]"


def load_lines(path: Path) -> list[Line]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    try:
        lines = [
            Line(str(item["name"]), _point(item["start"]), _point(item["end"]),
                 str(item["to_right"]), str(item["to_left"]))
            for item in data["lines"]
        ]
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{path}: invalid config ({e!r}). Expected {_EXPECTED}") from e
    names = [line.name for line in lines]
    if not lines or len(set(names)) != len(names):
        raise ValueError(f"{path}: need at least one line and unique line names, got {names}")
    return lines


def _point(value) -> tuple[float, float]:
    x, y = value
    return float(x), float(y)
