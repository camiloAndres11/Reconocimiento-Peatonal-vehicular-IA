"""Train/test partition by capture session (plan §2.3 and capture protocol §1).

Clips cut from the same session share light, place and even vehicles: they must all
land on the same side, or test metrics come out inflated. Sessions are shuffled within
each stratum (camera point x time band) so the test set covers every point and band.
"""
import random
import re
from collections import defaultdict
from dataclasses import dataclass

_NAME = re.compile(r"^(P[^_]+)_(\d{8})_(\d{2})(\d{2})(?:_s\d+)?$")
# Time bands of the capture protocol (§4); anything else is "other".
BANDS = (("morning", 6, 10), ("midday", 11, 14), ("afternoon", 16, 19))


@dataclass(frozen=True)
class Clip:
    name: str      # file name without extension
    session: str   # P<point>_<date>_<HHMM>
    stratum: str   # <point>/<band>


def parse_clip(name: str) -> Clip:
    m = _NAME.match(name)
    if not m:
        raise ValueError(f"{name}: expected P<point>_<YYYYMMDD>_<HHMM>[_s<SSSS>] (capture protocol §5)")
    point, date, hh, mm = m.groups()
    hour = int(hh)
    band = next((b for b, lo, hi in BANDS if lo <= hour < hi), "other")
    return Clip(name, f"{point}_{date}_{hh}{mm}", f"{point}/{band}")


def split(names: list[str], test_share: float = 0.3, folds: int = 5, seed: int = 0) -> dict[str, str]:
    """clip name -> "test" or "fold<i>" (i in 1..folds). Whole sessions move together.

    Per stratum, round(test_share x sessions) go to test (at least one when the stratum has
    two or more sessions); the rest are dealt to the folds round-robin, continuing across
    strata so folds stay balanced. Deterministic for a given seed.
    """
    clips = [parse_clip(n) for n in names]
    sessions_by_stratum: dict[str, set[str]] = defaultdict(set)
    for c in clips:
        sessions_by_stratum[c.stratum].add(c.session)

    rng = random.Random(seed)
    assignment: dict[str, str] = {}
    next_fold = 0
    for stratum in sorted(sessions_by_stratum):
        sessions = sorted(sessions_by_stratum[stratum])
        rng.shuffle(sessions)
        n_test = round(len(sessions) * test_share)
        if len(sessions) >= 2:
            n_test = max(1, n_test)
        for s in sessions[:n_test]:
            assignment[s] = "test"
        for s in sessions[n_test:]:
            assignment[s] = f"fold{next_fold % folds + 1}"
            next_fold += 1
    return {c.name: assignment[c.session] for c in clips}
