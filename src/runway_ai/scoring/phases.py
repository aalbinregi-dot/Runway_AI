"""How a walk changes between its approach, middle and exit.

A single set of twelve numbers describes the *average* of a pass, and averages
hide what a coach most wants to see. Someone who holds their posture for four
seconds and collapses over the last two scores the same as someone mildly
slouched throughout.

Why this reports measurements, not scores
-----------------------------------------
The obvious design is a table of twelve criterion scores by three phases. It was
built that way first, and it does not work. Measured on clips spliced so that one
quality genuinely drops halfway through - everything else, including build and
camera distance, held identical - the per-phase *scores* could not detect the
change:

====================  ============  =========  ==========
criterion             signal        gate       detected
====================  ============  =========  ==========
posture                    +0.28       2.23        0%
balance                    +0.29       2.12        0%
head_position              -1.82       3.36       28%
foot_placement             -1.85       3.76        0%
====================  ============  =========  ==========

Two structural reasons. A third of a six-second clip is two seconds, which is
below what the scorer needs - it already warns about clips under three seconds.
And several criteria are predicted largely from body-shape features that are
constant within a clip, so they carry no within-walk signal at all; posture moved
+0.28 for a near-full-range drop.

The *measurements* behind those scores are a different matter. They are direct
geometry with no model in the loop, so they do not suffer small-sample model
error. On the same spliced clips, comparing the first third against the last:

====================  ==================  ===========
quality changed       trace               effect size
====================  ==================  ===========
foot_placement        Step width               +1.69
shoulder_alignment    Shoulder tilt            +1.25
head_position         Head off-centre          +1.02
balance               Lateral sway             +0.99
====================  ==================  ===========

So this module reports what was *measured* in each phase - trunk lean, shoulder
tilt, step width, sway - and leaves the scoring to the whole clip, where there is
enough data for it.

Distance is still a confound
----------------------------
In a walk toward the camera the approach is filmed from further away, so the
figure is smaller and the landmarks noisier. Any systematic approach-to-exit
difference could be that rather than the walker. :func:`measure_null` quantifies
it on clips of deliberately constant quality, and a change is only reported when
it escapes that measured distribution - never on a threshold derived from theory.
"""
from __future__ import annotations

import copy
import json

import numpy as np

from ..data.schema import PoseSequence

#: Names in order.
PHASES = ("approach", "middle", "exit")

#: Fewest frames a phase needs before its mean means anything.
MIN_PHASE_FRAMES = 12

#: How many measured standard deviations a phase change must clear.
#:
#: Calibrated against the measured null, not derived. A per-trace 95% gate
#: (z = 1.96) reported a change on most clips where the walker never changed,
#: because there are a dozen traces and 5% of a dozen is most walks. Splitting
#: the error rate across the traces helps but under-covers - the empirical
#: distribution has heavier tails than a normal. This multiplier was chosen
#: against one set of constant-quality clips and validated on another.
CRITICAL_MULTIPLIER = 3.5

#: Traces allowed to assert that something *changed* between phases.
#:
#: Only these survived a power-versus-false-rate sweep. Widening the gate never
#: pushed the family-wise false rate below ~16% - the angular traces have a
#: per-phase spread of 7-11 degrees with tails no multiplier catches - while
#: power for everything else collapsed to nothing:
#:
#: =====  ==========  =======  ==================  ===============  ==========
#: mult   false(any)  balance  shoulder_alignment  head_position    step width
#: =====  ==========  =======  ==================  ===============  ==========
#: 2.5       16.7%      39%           0%                17%            94%
#: 3.0       16.7%       6%           0%                 6%            94%
#: 3.5       16.7%       0%           0%                 6%            94%
#: 4.0       15.6%       0%           0%                 0%            94%
#: =====  ==========  =======  ==================  ===============  ==========
#:
#: Step width is the one quantity with a real within-walk signal-to-noise ratio,
#: so it is the only one that gets to make a claim. The rest are still *shown* -
#: the per-phase numbers are what the user asked to see - but the system says
#: nothing about whether they moved, because on this evidence it cannot tell.
VALIDATED_TRACES = frozenset({"Step width"})

_null_cache: dict | None = None


def _null() -> dict:
    """Per-trace distribution of approach-to-exit change on constant walkers."""
    global _null_cache
    if _null_cache is not None:
        return _null_cache

    from ..config import REPORT_DIR
    table = {}
    try:
        path = REPORT_DIR / "phase_null.json"
        if path.exists():
            raw = json.loads(path.read_text())
            for name, stat in (raw.get("per_trace") or {}).items():
                if stat and stat.get("sd"):
                    table[name] = {"mean": float(stat["mean"]),
                                   "sd": float(stat["sd"]),
                                   "n": int(stat.get("n") or 0)}
    except Exception:
        table = {}
    _null_cache = table
    return table


def _bounds(n: int) -> list[tuple[str, int, int]]:
    cuts = [round(i * n / len(PHASES)) for i in range(len(PHASES) + 1)]
    return list(zip(PHASES, cuts[:-1], cuts[1:]))


def split(seq: PoseSequence) -> list[PoseSequence] | None:
    """Cut a sequence into three equal spans of frames, oldest first."""
    n = seq.n_frames
    if n < MIN_PHASE_FRAMES * len(PHASES):
        return None

    out = []
    for name, lo, hi in _bounds(n):
        part = PoseSequence(
            video_id=f"{seq.video_id}:{name}",
            landmarks=list(seq.landmarks),
            xy=seq.xy[lo:hi].copy(),
            z=seq.z[lo:hi].copy(),
            visibility=seq.visibility[lo:hi].copy(),
            fps=seq.fps,
            meta={k: v for k, v in seq.meta.items() if k != "preprocessed"},
        )
        part.meta["phase"] = name
        part.meta["frame_range"] = [lo, hi]
        out.append(part)
    return out


def analyse(seq: PoseSequence, timeline: dict | None = None) -> dict | None:
    """Per-phase measurements, and only the changes too large to be the artifact.

    ``seq`` must already be preprocessed - the traces are in normalised units and
    conditioning each phase separately would remove the very drift being looked
    for. ``timeline`` is reused when the caller already built it, since it holds
    exactly the traces needed.
    """
    from ..features import timeline as tl

    n = seq.n_frames
    if n < MIN_PHASE_FRAMES * len(PHASES):
        return None

    if timeline is None:
        try:
            timeline = tl.build(seq, [])
        except Exception:
            return None
    traces = (timeline or {}).get("traces") or {}
    if not traces:
        return None

    bounds = _bounds(n)
    null = _null()
    rows, changes = [], []

    for label, trace in traces.items():
        values = np.asarray(trace.get("values") or [], dtype=float)
        if values.size < MIN_PHASE_FRAMES * len(PHASES):
            continue
        # The trace may be decimated relative to the sequence, so slice by
        # fraction rather than by the sequence's own frame indices.
        scale = values.size / n
        per_phase = {}
        for name, lo, hi in bounds:
            seg = values[int(lo * scale):int(hi * scale)]
            seg = seg[np.isfinite(seg)]
            per_phase[name] = round(float(seg.mean()), 4) if seg.size else None
        if any(v is None for v in per_phase.values()):
            continue

        first, last = per_phase[PHASES[0]], per_phase[PHASES[-1]]
        delta = last - first
        spread = float(np.nanstd(values)) or 1e-9

        stat = null.get(label)
        if stat:
            centre, half = stat["mean"], CRITICAL_MULTIPLIER * stat["sd"]
            basis = "measured on constant-quality clips"
        else:
            # No null measured for this trace: fall back to its own variability,
            # and say so rather than implying the same rigour.
            centre, half = 0.0, CRITICAL_MULTIPLIER * spread
            basis = "trace variability (phase artifact not yet measured)"

        row = {
            "label": label,
            "unit": trace.get("unit"),
            "per_phase": per_phase,
            "delta": round(delta, 4),
            "corrected_delta": round(delta - centre, 4),
            "expected_artifact": round(centre, 4),
            "threshold": round(half, 4),
            "effect_size": round(delta / spread, 2),
            "higher_is_worse": (trace.get("reference") or {}).get("worse") != "low",
            # Shown for every trace; claimed for only the validated ones.
            "can_claim": label in VALIDATED_TRACES,
            "changed": (label in VALIDATED_TRACES
                        and abs(delta - centre) >= half),
            "basis": basis,
        }
        rows.append(row)
        if row["changed"]:
            changes.append(row)

    if not rows:
        return None

    rows.sort(key=lambda r: -abs(r["effect_size"]))
    changes.sort(key=lambda r: -abs(r["corrected_delta"]))

    return {
        "order": list(PHASES),
        "seconds_per_phase": round(n / max(seq.fps, 1e-6) / len(PHASES), 2),
        "traces": rows,
        "changes": changes,
        "null_measured": bool(null),
        "claimable": sorted(VALIDATED_TRACES),
        "note": (
            "Measurements, not scores. A third of a clip is too little data for "
            "the scoring models - on spliced clips where one quality genuinely "
            "dropped halfway through, per-phase scores detected it at most 28% of "
            "the time and for several criteria not at all."
        ),
        "caveat": (
            "The approach is filmed from further away, so its landmarks are "
            "noisier. Only step width was able to beat that artifact reliably "
            "when tested, so it is the only quantity this will call a change. "
            "The rest are shown as measured and left for you to read - a "
            "difference in them is as likely to be the camera as the walker."
        ),
    }


def measure_null(n_clips: int = 120, seed: int = 11) -> dict:
    """Phase-to-phase change on walkers whose quality never varies.

    The true change is zero by construction, so whatever comes out is the
    artifact of the split itself - the walker approaching the camera, and the
    landmarks sharpening as they do.
    """
    from ..data.loaders import preprocess
    from ..data.synthetic import make_clip, sample_params
    from ..features import timeline as tl

    collected: dict[str, list[float]] = {}
    used = 0
    for i in range(n_clips):
        rng = np.random.default_rng(seed + i)
        params = sample_params(rng, correlated=False)
        seq, _ = make_clip(f"null_{i}", params, rng)
        try:
            prepared = preprocess(copy.deepcopy(seq))
            got = analyse(prepared, tl.build(prepared, []))
        except Exception:
            continue
        if not got:
            continue
        used += 1
        for row in got["traces"]:
            collected.setdefault(row["label"], []).append(row["delta"])

    def stat(values):
        arr = np.asarray(values, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size < 2:
            return None
        return {"mean": round(float(arr.mean()), 4),
                "sd": round(float(arr.std(ddof=1)), 4),
                "n": int(arr.size)}

    return {"n_clips": used,
            "per_trace": {k: s for k, v in collected.items() if (s := stat(v))}}
