"""CVAT "MOT 1.1" archives: write pre-annotations to import, read finished annotations back.

Archive: gt/gt.txt rows `frame (from 1), track id, x, y, w, h, not ignored, class id, visibility`
and gt/labels.txt with one label per line (class id = line number, from 1).
"""
import csv
import io
import zipfile
from collections import defaultdict
from pathlib import Path

from vtia.adapters.video_io import VIDEO_SUFFIXES, video_info
from vtia.application.evaluate import AnnotatedSequence
from vtia.domain.classes import RoadClass
from vtia.domain.counting import ClassVotes
from vtia.domain.models import Detection, GroundTruth

LABELS = [c.value for c in RoadClass] + ["ignore"]  # same order as the CVAT project labels (README)


def write_preannotations(tracked_frames: list[list[Detection]], out: Path, keyframe_every: int = 5) -> int:
    """Tracks -> MOT 1.1 zip for CVAT's "Upload annotations". One label per track (its majority class).
    Only every `keyframe_every`-th frame plus each track's first and last are written; CVAT
    interpolates between them, which keeps corrections cheap. Returns the number of tracks."""
    votes = ClassVotes()
    frames_of: dict[int, list[Detection]] = defaultdict(list)
    for frame in tracked_frames:
        for d in frame:
            votes.add(d)
            frames_of[d.track_id].append(d)
    rows = []
    for track, dets in frames_of.items():
        class_id = LABELS.index(votes[track].value) + 1
        keep = [d for i, d in enumerate(dets) if i == 0 or i == len(dets) - 1 or d.frame % keyframe_every == 0]
        for d in keep:
            x1, y1, x2, y2 = d.box
            rows.append((d.frame + 1, track, round(x1, 1), round(y1, 1), round(x2 - x1, 1), round(y2 - y1, 1), 1, class_id, 1))
    rows.sort()
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerows(rows)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("gt/gt.txt", buf.getvalue())
        z.writestr("gt/labels.txt", "\n".join(LABELS) + "\n")
    return len(frames_of)


def read_annotations(archive: Path, video: Path) -> AnnotatedSequence:
    """CVAT export (zip or extracted folder) + the video it annotates -> evaluation sequence.
    Boxes labelled `ignore`, or flagged "not ignored" = 0, become ignore regions."""
    files = _read_files(Path(archive))
    labels = [l.strip() for l in files["gt/labels.txt"].splitlines() if l.strip()]
    _, n = video_info(video)
    truths = [[] for _ in range(n)]
    ignore = [[] for _ in range(n)]
    for row in csv.reader(io.StringIO(files["gt/gt.txt"])):
        if not row:
            continue
        frame, track = int(float(row[0])) - 1, int(float(row[1]))
        x, y, w, h = (float(v) for v in row[2:6])
        considered, label = int(float(row[6])), labels[int(float(row[7])) - 1]
        if not 0 <= frame < n:
            continue
        box = (x, y, x + w, y + h)
        if label == "ignore" or not considered:
            ignore[frame].append(box)
        else:
            truths[frame].append(GroundTruth(frame, track, RoadClass(label), box))
    return AnnotatedSequence(Path(video).stem, Path(video), truths, ignore)


def _read_files(archive: Path) -> dict[str, str]:
    names = ("gt/gt.txt", "gt/labels.txt")
    if archive.is_dir():
        return {n: (archive / n).read_text(encoding="utf-8") for n in names}
    with zipfile.ZipFile(archive) as z:  # entries may sit under a top-level folder
        return {n: z.read(next(i for i in z.namelist() if i.endswith(n))).decode("utf-8") for n in names}



def load_cvat_dir(annotations: Path, videos: Path) -> list[AnnotatedSequence]:
    """Every <clip>.zip (or extracted <clip>/ folder) in `annotations`, paired with <clip>.<ext> in `videos`."""
    sequences = []
    for archive in sorted(p for p in Path(annotations).iterdir() if p.suffix == ".zip" or p.is_dir()):
        video = next((Path(videos) / f"{archive.stem}{ext}" for ext in VIDEO_SUFFIXES
                      if (Path(videos) / f"{archive.stem}{ext}").exists()), None)
        if video is None:
            raise FileNotFoundError(f"no video named {archive.stem}.* in {videos} for annotations {archive.name}")
        sequences.append(read_annotations(archive, video))
    return sequences
