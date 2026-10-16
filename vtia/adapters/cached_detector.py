"""Runs a detector once per source and reuses its output afterwards.

Plan §1.2: heavy models run once (here or on Colab) and everything downstream
reads their detections. Cache: <dir>/<model>/<source>.jsonl (one line per frame)
plus <source>.json with frames and seconds, written last as the completion mark.
"""
import json
import time
from collections.abc import Iterator
from pathlib import Path

from vtia.application.ports import Detector
from vtia.domain.classes import RoadClass
from vtia.domain.models import Detection


class CachedDetector:
    def __init__(self, inner: Detector, cache_dir: Path):
        self.inner = inner
        self.name = inner.name
        self.dir = Path(cache_dir) / inner.name
        self.frames = 0      # totals over the sources detected, for throughput (fps)
        self.seconds = 0.0

    def detect(self, source: Path) -> Iterator[list[Detection]]:
        source = Path(source)
        cache = self.dir / f"{source.stem}.jsonl"
        meta = cache.with_suffix(".json")
        if not meta.exists():
            self.dir.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            frames = list(self.inner.detect(source))
            seconds = time.perf_counter() - start
            cache.write_text("".join(json.dumps([[d.road_class, d.confidence, *d.box] for d in f]) + "\n"
                                     for f in frames), encoding="utf-8")
            meta.write_text(json.dumps({"frames": len(frames), "seconds": seconds}), encoding="utf-8")
        info = json.loads(meta.read_text(encoding="utf-8"))
        self.frames += info["frames"]
        self.seconds += info["seconds"]
        with open(cache, encoding="utf-8") as f:
            for i, line in enumerate(f):
                yield [Detection(i, RoadClass(c), conf, tuple(box)) for c, conf, *box in json.loads(line)]
