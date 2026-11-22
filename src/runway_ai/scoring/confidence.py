"""How well is each criterion actually identified, as opposed to inferred?

`significance` answers "is this difference bigger than the error bar". This
answers a different and prior question: **is this score measuring the thing it
names at all?**

Where the numbers come from
---------------------------
The generator draws the twelve skill qualities from a shared ability factor at
r ~ 0.6, because real walkers are like that. The side effect is that a model can
score one criterion by reading another's features - predict rhythm from postural
sway, because sway indicates general ability and general ability indicates
rhythm. On correlated data that shortcut looks like accuracy.

`scripts/leakage_audit.py` measures the size of that shortcut by re-running the
identical out-of-fold procedure on a corpus where the twelve qualities are drawn
*independently* (max cross-correlation 0.079 versus 0.797). Cross-criterion
shortcuts are worthless there, so what survives is genuine:

=====================  ===========  =============  =======
criterion              as reported  independent    own
=====================  ===========  =============  =======
foot_placement            0.929        0.937        0.915
confidence                0.913        0.896        0.872
shoulder_alignment        0.900        0.901        0.885
head_position             0.842        0.839        0.813
balance                   0.827        0.838        0.818
stage_presence            0.829        0.759          n/a
posture                   0.701        0.591        0.161
arm_swing                 0.590        0.392        0.384
stride_length             0.575        0.381        0.385
symmetry                  0.523        0.279        0.055
hip_movement              0.526        0.232        0.191
rhythm                    0.533        0.172        0.160
=====================  ===========  =============  =======

Mean R^2 falls from 0.724 to 0.601. Five criteria are unaffected - they were
always measuring themselves. Three lose most of their apparent accuracy.

What this changes about the product
-----------------------------------
Twelve scores presented identically imply twelve equally trustworthy
measurements, and that is now known to be false. A user acting on a rhythm score
of 4.1 is acting on something the system can barely identify; the same user
acting on a foot-placement score of 4.1 is on solid ground. Saying so is more
useful than hiding it behind a single headline figure.

Two honest caveats about the caveat. These are R^2 values against *synthetic*
ground truth, so they measure identifiability within this generator's world, not
against human judges - no expert-rated corpus exists here. And a low figure does
not make a score meaningless: rhythm still correlates with rhythm, just weakly
enough that only large differences are worth acting on.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..config import CRITERIA, REPORT_DIR

AUDIT_PATH = REPORT_DIR / "leakage_audit.csv"

#: Bands over the independent-latent R^2. The boundaries are round numbers
#: chosen to separate the three groups the audit actually produced, rather than
#: fitted to anything - a 0.60 cut would split "well identified" arbitrarily
#: between head_position (0.839) and posture (0.591), which is where the real gap
#: in the data lies.
BANDS = [
    (0.75, "strong", "Measured directly and reliably."),
    (0.50, "moderate", "Measured, but with a wider error bar - act on clear "
                       "differences rather than small ones."),
    (0.00, "weak", "Only weakly identifiable from pose alone. Treat as a hint, "
                   "not a verdict, and prefer the evidence traces over the number."),
]

#: Used when the audit has not been run. Deliberately not a guess at the real
#: value: absent evidence is reported as absent, not as moderate.
UNKNOWN = "unknown"

_cache: dict | None = None


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache

    table: dict[str, dict] = {}
    if AUDIT_PATH.exists():
        try:
            with AUDIT_PATH.open(newline="") as fh:
                for row in csv.DictReader(fh):
                    name = row.get("criterion")
                    if name not in CRITERIA:
                        continue
                    indep = _as_float(row.get("r2_independent"))
                    if indep is None:
                        continue
                    table[name] = {
                        "r2_independent": indep,
                        "r2_as_reported": _as_float(row.get("r2_as_reported")),
                        "r2_own_features": _as_float(row.get("r2_own_features")),
                        "leakage": _as_float(row.get("leakage")),
                        **_band(indep),
                    }
        except Exception:
            table = {}
    _cache = table
    return table


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _band(r2: float) -> dict:
    for floor, tier, note in BANDS:
        if r2 >= floor:
            return {"tier": tier, "note": note}
    return {"tier": UNKNOWN, "note": ""}


def available() -> bool:
    return bool(_load())


def for_criterion(criterion: str) -> dict | None:
    return _load().get(criterion)


def tier(criterion: str) -> str:
    entry = _load().get(criterion)
    return entry["tier"] if entry else UNKNOWN


def summary() -> dict:
    """Everything the API needs to annotate a set of scores."""
    table = _load()
    if not table:
        return {"available": False}

    by_tier: dict[str, list[str]] = {}
    for name, entry in table.items():
        by_tier.setdefault(entry["tier"], []).append(name)

    return {
        "available": True,
        "per_criterion": table,
        "by_tier": by_tier,
        "basis": ("out-of-fold R^2 on a 1500-clip corpus with the twelve skill "
                  "qualities drawn independently, so no criterion can be scored "
                  "by reading another's features"),
        "caveat": ("Measured against synthetic ground truth, so this is "
                   "identifiability within the generator's world - not agreement "
                   "with human judges, for which no rated corpus exists here."),
    }
