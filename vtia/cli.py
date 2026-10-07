"""Entry point and composition root: wires adapters into the use case.

    python -m vtia.cli VIDEO --lines configs/points/<point>.yaml
"""
import argparse
import time
from pathlib import Path

from vtia.adapters.csv_writer import write_counts, write_events
from vtia.adapters.ultralytics_tracker import UltralyticsTracker
from vtia.adapters.yaml_config import load_lines
from vtia.application.process_video import process_video
from vtia.domain.counting import count_by


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Count road users crossing virtual lines in a video.")
    p.add_argument("video", type=Path)
    p.add_argument("--lines", type=Path, required=True, help="camera-point YAML with the virtual lines")
    p.add_argument("--weights", default="yolo26n.pt", help="Ultralytics weights (default: %(default)s)")
    p.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    p.add_argument("--conf", type=float, help="detector confidence threshold (Ultralytics default if omitted)")
    p.add_argument("--out", type=Path, default=Path("outputs"), help="output folder (default: %(default)s)")
    args = p.parse_args(argv)
    if not args.video.is_file():
        p.error(f"video not found: {args.video}")

    lines = load_lines(args.lines)
    tracker = UltralyticsTracker(args.weights, args.tracker, args.conf)

    start = time.perf_counter()
    events, frames = process_video(args.video, tracker, lines)
    elapsed = time.perf_counter() - start

    out = args.out / args.video.stem
    out.mkdir(parents=True, exist_ok=True)
    counts = count_by(events)
    write_events(events, out / "events.csv")
    write_counts(counts, out / "counts.csv")

    for (line, direction, road_class), n in sorted(counts.items()):
        print(f"{line:<12}{direction:<20}{road_class:<12}{n}")
    print(f"{len(events)} crossings | {frames} frames in {elapsed:.1f} s ({frames / elapsed:.1f} fps) -> {out}")


if __name__ == "__main__":
    main()
