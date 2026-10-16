"""Command-line entry point.

    python -m vtia.cli count VIDEO --point configs/points/<point>.yaml [--save-video]
    python -m vtia.cli anonymize data/raw --out data/anon
    python -m vtia.cli prelabel data/anon/P1_20261015_0705_s0312.mp4 --out data/prelabels
    python -m vtia.cli split data/anon --out data/splits.csv
    python -m vtia.cli evaluate --visdrone data/visdrone/VisDrone2019-MOT-val
    python -m vtia.cli evaluate --cvat data/annotations --videos data/anon
    python -m vtia.cli calibrate --visdrone data/visdrone/VisDrone2019-MOT-val --model yolo26n.pt
    python -m vtia.cli train --visdrone data/visdrone/VisDrone2019-MOT-val --val <seq> <seq>
"""
import argparse
import csv
from datetime import datetime
from pathlib import Path

from vtia.adapters.cached_detector import CachedDetector
from vtia.adapters.coco_map import coco_map
from vtia.adapters.mot_tracker import ALGORITHMS
from vtia.adapters.table_files import markdown_table, write_csv
from vtia.adapters.trackeval_metrics import mot_metrics
from vtia.adapters.video_io import VIDEO_SUFFIXES
from vtia.adapters.visdrone import load_visdrone
from vtia.application.evaluate import EvalResult, evaluate
from vtia.compose import (MODELS, PRELABEL_MODEL, analyze_file, anonymize_file, device_name, make_detector,
                          make_tracker, prelabel_file, with_postprocess)
from vtia.domain.classes import RoadClass
from vtia.domain.evaluation import BACKGROUND, MISSED
from vtia.domain.splits import split

DETECTORS = f"{', '.join(MODELS)} (or any Ultralytics weights, e.g. a fine-tuned best.pt)"


# --- count ---------------------------------------------------------------------------------

def count(args) -> None:
    if not args.video.is_file():
        raise SystemExit(f"video not found: {args.video}")
    run = analyze_file(args.video, args.point, model=args.model, tracker=args.tracker, track_thresh=args.track_thresh,
                       conf=args.conf, dedup=args.dedup, riders=args.riders, start=args.start, out=args.out,
                       save_video=args.save_video)
    a, s = run.analysis, run.summary
    for line, direction, road_class, n in s.by_line:
        print(f"{line:<14}{direction:<20}{road_class:<12}{n}")
    print(f"{len(a.events)} crossings, {len(a.visits)} zone visits, {len(a.alerts)} alerts | "
          f"{a.frames} frames in {run.seconds:.1f} s ({a.frames / run.seconds:.1f} fps, {device_name()})")
    print(f"start time: {a.start:%Y-%m-%d %H:%M}" if a.start else
          "start time: unknown (name the file P<point>_<YYYYMMDD>_<HHMM> or pass --start)")
    print(f"report: {run.out / 'report.pdf'}" + (f" | video: {run.out / 'annotated.mp4'}" if args.save_video else ""))


# --- data preparation (phase 2) ------------------------------------------------------------

def _videos(inputs: list[Path]) -> list[Path]:
    found = []
    for p in inputs:
        found += sorted(f for f in p.rglob("*") if f.suffix.lower() in VIDEO_SUFFIXES) if p.is_dir() else [p]
    if not found:
        raise SystemExit("no videos found")
    return found


def _progress(name: str):
    def show(i: int, total: int) -> None:
        if i == total or i % 50 == 0:
            print(f"\r{name}: frame {i}/{total}", end="" if i < total else "\n", flush=True)
    return show


def anonymize(args) -> None:
    for video in _videos(args.inputs):
        out, sheet = anonymize_file(video, args.out, model=args.model, conf=args.conf, hold_frames=args.hold,
                                    progress=_progress(video.name))
        print(f"  -> {out}\n  check by eye: {sheet}")
    print("Review every *_check.jpg (and play the videos) before deleting the raw files: "
          "only anonymised copies may leave this PC (capture protocol §8).")


def prelabel(args) -> None:
    for video in _videos(args.inputs):
        out, tracks = prelabel_file(video, args.out, model=args.model, keyframe_every=args.keyframe_every,
                                    progress=_progress(video.name))
        print(f"  -> {out} ({tracks} tracks). In CVAT: task > Actions > Upload annotations > MOT 1.1")


def split_cmd(args) -> None:
    names = [v.stem for v in _videos(args.inputs)]
    assignment = split(names, args.test_share, args.folds, args.seed)
    write_csv(args.out, ["clip", "split"], sorted(assignment.items()))
    for part in sorted(set(assignment.values())):
        print(f"{part:<6} {sum(1 for v in assignment.values() if v == part)} clips")
    print(f"-> {args.out}")


# --- evaluation (phases 3 and 5) -----------------------------------------------------------

def _sequences(args):
    if args.visdrone:
        sequences = load_visdrone(args.visdrone)
    elif args.cvat and args.videos:
        from vtia.adapters.cvat_mot import load_cvat_dir
        sequences = load_cvat_dir(args.cvat, args.videos)
    else:
        raise SystemExit("give --visdrone ROOT, or --cvat ANNOTATIONS_DIR together with --videos VIDEOS_DIR")
    if args.sequences:
        sequences = [s for s in sequences if s.name in args.sequences]
        missing = set(args.sequences) - {s.name for s in sequences}
        if missing:
            raise SystemExit(f"unknown sequences: {sorted(missing)}")
    return sequences


def _source_name(args) -> str:
    return args.visdrone.name if args.visdrone else Path(args.cvat).name


SUMMARY_HEADER = ["model", "tracker", "track_thresh", "mAP50", "mAP50_95", "AP_small", "macro_F1", "balanced_acc",
                  "accuracy", "HOTA", "MOTA", "MOTP", "IDF1", "IDSW", "IDSW_per_100", "fps"]


def evaluate_cmd(args) -> None:
    sequences = _sequences(args)
    out = args.out or Path("outputs/eval") / _source_name(args)
    print(f"{len(sequences)} sequences, {sum(len(s.truths) for s in sequences)} frames | device: {device_name()}")
    summary, sections = [], []
    for name in args.models:
        cached = CachedDetector(make_detector(name, conf=0.01), out / "detections")
        detector = with_postprocess(cached, args.dedup, args.riders)
        tracker = make_tracker(args.tracker, name, args.track_thresh)
        r = evaluate(sequences, detector, tracker, coco_map, mot_metrics, conf=args.conf)
        fps = cached.frames / cached.seconds if cached.seconds else 0.0  # detector only
        d, t, c = r.detection, r.tracking, r.classes
        summary.append([detector.name, tracker.algorithm, tracker.track_thresh, d["mAP50"], d["mAP50_95"],
                        d["AP_small"], c["macro_f1"], c["balanced_accuracy"], c["accuracy"], t["HOTA"], t["MOTA"],
                        t["MOTP"], t["IDF1"], t["IDSW"], t["IDSW_per_100_objects"], fps])
        sections.append(_write_model_tables(out / detector.name, detector.name, r))
        print(f"done: {detector.name}")

    suffix = ((f"_dedup{args.dedup}" if args.dedup else "") + (f"_riders{args.riders}" if args.riders else "")
              + (f"_{args.tracker}" if args.tracker != "bytetrack" else "")
              + (f"_tt{args.track_thresh}" if args.track_thresh is not None else "")
              + (f"_{'-'.join(args.sequences)}" if args.sequences else ""))
    write_csv(out / f"summary{suffix}.csv", SUMMARY_HEADER, summary)
    report = (f"# Evaluation on {_source_name(args)}\n\nDevice: {device_name()} | confusion threshold: "
              f"conf >= {args.conf}, IoU >= 0.5 | dedup: {args.dedup or 'off'} | riders: {args.riders or 'off'}"
              + (f" | sequences: {', '.join(args.sequences)}" if args.sequences else "") + "\n\n"
              + markdown_table(SUMMARY_HEADER, summary) + "\n" + "\n".join(sections))
    (out / f"summary{suffix}.md").write_text(report, encoding="utf-8")
    print(report)


def _write_model_tables(out: Path, name: str, r: EvalResult) -> str:
    out.mkdir(parents=True, exist_ok=True)
    classes = list(RoadClass)
    per_class_header = ["class", "precision", "recall", "f1", "support", "AP50", "AP50_95"]
    per_class = [[c, m["precision"], m["recall"], m["f1"], m["support"],
                  r.detection["per_class"][c]["AP50"], r.detection["per_class"][c]["AP50_95"]]
                 for c, m in r.classes["per_class"].items()]
    confusion_header = ["true \\ predicted", *classes, MISSED]
    confusion = [[row, *(r.confusion[(row, col)] for col in [*classes, MISSED])] for row in [*classes, BACKGROUND]]
    write_csv(out / "per_class.csv", per_class_header, per_class)
    write_csv(out / "confusion.csv", confusion_header, confusion)
    return (f"## {name}\n\n" + markdown_table(per_class_header, per_class) + "\n"
            + markdown_table(confusion_header, confusion))


def calibrate(args) -> None:
    """Tracking metrics for a grid of tracker thresholds, reusing cached detections (no mAP: it doesn't change)."""
    sequences = _sequences(args)
    out = args.out or Path("outputs/eval") / _source_name(args)
    cached = CachedDetector(make_detector(args.model, conf=0.01), out / "detections")
    detector = with_postprocess(cached, args.dedup, args.riders)
    header = ["track_thresh", "HOTA", "MOTA", "IDF1", "IDSW", "macro_F1"]
    rows = []
    for t in args.grid:
        r = evaluate(sequences, detector, make_tracker(args.tracker, args.model, t), lambda *_: {}, mot_metrics,
                     conf=args.conf)
        rows.append([t, r.tracking["HOTA"], r.tracking["MOTA"], r.tracking["IDF1"], r.tracking["IDSW"], r.classes["macro_f1"]])
        print(f"track_thresh {t}: HOTA {rows[-1][1]:.3f}  MOTA {rows[-1][2]:.3f}  IDF1 {rows[-1][3]:.3f}  IDSW {rows[-1][4]}")
    best = max(rows, key=lambda r: r[header.index(args.metric)])
    name = f"calibration_{detector.name}_{args.tracker}"
    write_csv(out / f"{name}.csv", header, rows)
    (out / f"{name}.md").write_text(
        f"# Tracker threshold calibration: {detector.name} + {args.tracker} on {_source_name(args)}\n\n"
        + markdown_table(header, rows) + f"\nBest by {args.metric}: **{best[0]}**\n", encoding="utf-8")
    print(f"best by {args.metric}: track_thresh {best[0]} -> {out / name}.md")


# --- fine-tuning (phase 4) -----------------------------------------------------------------

def train_cmd(args) -> None:
    from vtia.adapters.yolo_training import export_dataset, train  # heavy imports only for this command

    sequences = _sequences(args)
    if args.splits:
        with open(args.splits, encoding="utf-8") as f:
            part = {row["clip"]: row["split"] for row in csv.DictReader(f)}
        val_names = {n for n, p in part.items() if p == f"fold{args.fold}"}
        train_seqs = [s for s in sequences if part.get(s.name, "").startswith("fold") and s.name not in val_names]
        val_seqs = [s for s in sequences if s.name in val_names]  # the "test" split is never touched here
    else:
        val_seqs = [s for s in sequences if s.name in set(args.val or [])]
        train_seqs = [s for s in sequences if s not in val_seqs]
    if not train_seqs or not val_seqs:
        raise SystemExit("need at least one training and one validation sequence (--val NAMES or --splits + --fold)")
    print(f"train: {[s.name for s in train_seqs]}\nval:   {[s.name for s in val_seqs]}")
    data = export_dataset(train_seqs, val_seqs, Path("outputs/train") / args.name / "dataset", every=args.every)
    best = train(data, args.weights, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, name=args.name)
    print(f"best weights: {best}\nEvaluate them like any model: evaluate ... --models {best} --sequences <val>")


# --- arguments -----------------------------------------------------------------------------

def parse_start(text: str) -> datetime:
    return datetime.strptime(text, "%Y-%m-%d %H:%M")


def _postprocess_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--dedup", type=float, metavar="IOU",
                   help="drop boxes overlapping a more confident one by more than this IoU, any class (e.g. 0.7)")
    p.add_argument("--riders", type=float, metavar="OVERLAP",
                   help="experiment E2: merge a person into the bicycle/motorcycle they ride (e.g. 0.4)")


def _tracker_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--tracker", choices=ALGORITHMS, default="bytetrack", help="(default: %(default)s)")


def _source_options(p: argparse.ArgumentParser) -> None:
    p.add_argument("--visdrone", type=Path, help="VisDrone2019-MOT folder with sequences/ and annotations/")
    p.add_argument("--cvat", type=Path, help="folder of CVAT MOT 1.1 exports, one <clip>.zip per clip")
    p.add_argument("--videos", type=Path, help="folder with the (anonymised) videos the CVAT exports annotate")
    p.add_argument("--sequences", nargs="+", metavar="NAME", help="use only these sequences / clips")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="vtia")
    sub = p.add_subparsers(required=True)

    c = sub.add_parser("count", help="counts, zone visits, alerts and report for one video")
    c.add_argument("video", type=Path)
    c.add_argument("--point", "--lines", dest="point", type=Path, required=True,
                   help="camera-point YAML: lines, zones, restrictions (see configs/points/)")
    c.add_argument("--start", type=parse_start, metavar='"YYYY-MM-DD HH:MM"',
                   help="wall-clock start of the video, if its name does not follow the capture protocol")
    c.add_argument("--model", default="yolo26n.pt", help=f"{DETECTORS} (default: %(default)s)")
    c.add_argument("--conf", type=float, default=0.1, help="minimum detector confidence (default: %(default)s)")
    c.add_argument("--out", type=Path, default=Path("outputs"), help="output folder (default: %(default)s)")
    c.add_argument("--save-video", action="store_true", help="also write annotated.mp4 with tracks, zones, counts, alerts")
    c.add_argument("--track-thresh", type=float, metavar="T",
                   help="tracker high-confidence threshold; new tracks need T + 0.1 (default: 0.3 for yolo26n, else 0.6)")
    _tracker_options(c)
    _postprocess_options(c)
    c.set_defaults(func=count)

    a = sub.add_parser("anonymize", help="blur heads and plates; writes copies + a contact sheet to check by eye")
    a.add_argument("inputs", type=Path, nargs="+", help="videos or folders of videos")
    a.add_argument("--out", type=Path, required=True, help="folder for the anonymised copies")
    a.add_argument("--model", default="yolo26s.pt", help="(default: %(default)s)")
    a.add_argument("--conf", type=float, default=0.15, help="low on purpose: blurring a false positive is free (default: %(default)s)")
    a.add_argument("--hold", type=int, default=3, help="keep regions blurred this many extra frames (default: %(default)s)")
    a.set_defaults(func=anonymize)

    pl = sub.add_parser("prelabel", help="CVAT pre-annotations (MOT 1.1 zip) so annotators only correct")
    pl.add_argument("inputs", type=Path, nargs="+", help="anonymised videos or folders")
    pl.add_argument("--out", type=Path, required=True)
    pl.add_argument("--model", default=PRELABEL_MODEL, help="use a model outside the comparison (default: %(default)s)")
    pl.add_argument("--keyframe-every", type=int, default=5, help="(default: %(default)s)")
    pl.set_defaults(func=prelabel)

    sp = sub.add_parser("split", help="test set + k folds, whole capture sessions together")
    sp.add_argument("inputs", type=Path, nargs="+", help="clips or folders, named P<point>_<YYYYMMDD>_<HHMM>[_s<SSSS>]")
    sp.add_argument("--out", type=Path, required=True, help="CSV clip,split")
    sp.add_argument("--test-share", type=float, default=0.3, help="(default: %(default)s)")
    sp.add_argument("--folds", type=int, default=5, help="(default: %(default)s)")
    sp.add_argument("--seed", type=int, default=0, help="(default: %(default)s)")
    sp.set_defaults(func=split_cmd)

    e = sub.add_parser("evaluate", help="score detectors + tracker on VisDrone or on CVAT annotations")
    _source_options(e)
    e.add_argument("--models", nargs="+", default=MODELS, help=DETECTORS)
    e.add_argument("--conf", type=float, default=0.25, help="operating threshold for the confusion matrix (default: %(default)s)")
    e.add_argument("--out", type=Path, help="(default: outputs/eval/<dataset>)")
    e.add_argument("--track-thresh", type=float, metavar="T", help="(default: 0.3 for yolo26n, else 0.6)")
    _tracker_options(e)
    _postprocess_options(e)
    e.set_defaults(func=evaluate_cmd)

    k = sub.add_parser("calibrate", help="pick the tracker threshold of one model on annotated data")
    _source_options(k)
    k.add_argument("--model", required=True, help=DETECTORS)
    k.add_argument("--grid", type=float, nargs="+", default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7], help="(default: %(default)s)")
    k.add_argument("--metric", choices=["HOTA", "MOTA", "IDF1"], default="HOTA", help="(default: %(default)s)")
    k.add_argument("--conf", type=float, default=0.25)
    k.add_argument("--out", type=Path, help="(default: outputs/eval/<dataset>)")
    _tracker_options(k)
    _postprocess_options(k)
    k.set_defaults(func=calibrate)

    t = sub.add_parser("train", help="fine-tune a YOLO model (phase 4)")
    _source_options(t)
    t.add_argument("--val", nargs="+", metavar="NAME", help="validation sequences (the rest train)")
    t.add_argument("--splits", type=Path, help="CSV from `split`: trains on the folds except --fold, never on test")
    t.add_argument("--fold", type=int, default=1, help="validation fold when using --splits (default: %(default)s)")
    t.add_argument("--weights", default="yolo26n.pt", help="starting weights (default: %(default)s)")
    t.add_argument("--epochs", type=int, default=50)
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--imgsz", type=int, default=640)
    t.add_argument("--every", type=int, default=5, help="use 1 of every N frames (default: %(default)s)")
    t.add_argument("--name", default="run", help="run name under outputs/train/ (default: %(default)s)")
    t.set_defaults(func=train_cmd)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
