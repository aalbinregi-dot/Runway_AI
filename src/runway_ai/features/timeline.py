"""Per-frame evidence: where in the clip each measurement came from.

The scalar features answer *how much*; they cannot answer *when*. A user told
their head position scores 1.5 has no way to check that, and no way to act on
it, unless they can see the moments that produced it.

This module rebuilds the same quantities as per-frame traces, segments the clip
into the gait cycles the timing features were derived from, and locates the
worst moment for each measurement. Nothing here re-derives a score - the traces
are the identical expressions used in :mod:`runway_ai.features.builder`, so the
evidence and the number can never disagree.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import PoseSequence
from .signals import (
    angle_from_vertical,
    cycle_times,
    gait_highpass,
    transverse_tilt,
)

#: feature name -> (label, unit, worse_direction)
#:
#: ``worse_direction`` says which end of the trace is the fault, so the "worst
#: moment" is the right extremum rather than merely the largest number.
TRACES = {
    "posture_trunk_lean_mean": ("Trunk lean", "deg", "high"),
    "posture_trunk_lean_std": ("Trunk lean", "deg", "high"),
    "posture_neck_angle_mean": ("Neck angle", "deg", "high"),
    "shoulder_tilt_abs_mean": ("Shoulder tilt", "deg", "high"),
    "shoulder_tilt_amp": ("Shoulder tilt", "deg", "high"),
    "head_lateral_offset": ("Head off-centre", "torso", "high"),
    "head_lateral_amp": ("Head off-centre", "torso", "high"),
    "foot_step_width_mean": ("Step width", "torso", "high"),
    "foot_step_width_at_strike": ("Step width", "torso", "high"),
    "foot_placement_var": ("Step width", "torso", "high"),
    "balance_sway_amp": ("Lateral sway", "torso", "high"),
    "balance_sway_rms": ("Lateral sway", "torso", "high"),
    "balance_upper_body_sway": ("Shoulder sway", "torso", "high"),
    "hip_obliquity_amp": ("Hip drop", "deg", "high"),
    "hip_obliquity_std": ("Hip drop", "deg", "high"),
    # Amplitude and variability features have no single "moment" that caused
    # them - they are properties of the whole clip. Pointing at the extremum of
    # the underlying trace is still the most useful frame to look at: it is
    # where the motion was most pronounced.
    "arm_swing_amp_error": ("Arm reach", "torso", "high"),
    "arm_swing_amp_mean": ("Arm reach", "torso", "high"),
    "arm_swing_asymmetry": ("Arm reach L-R", "torso", "high"),
    "stride_leg_shorten_mean": ("Leg extension", "torso", "low"),
    "stride_knee_lift": ("Knee lift", "torso", "high"),
    "sym_overall_index": ("Arm reach L-R", "torso", "high"),
    "sym_ankle_index": ("Step width", "torso", "high"),
}

#: Feature-name endings that describe *variation* rather than a level. A trace
#: is a per-frame level, so shading it against a standard-deviation or amplitude
#: band would draw a line the trace is not measured against.
_SPREAD_SUFFIXES = ("_std", "_amp", "_var", "_cv", "_error", "_index", "_jerk")


def _references() -> dict[str, dict]:
    """Trace label -> the rubric band the trace can honestly be drawn against.

    Read from the live ``RUBRIC`` rather than restated here, so a threshold moved
    by ``rubric_calibration.json`` moves the shaded region on the chart too. A
    trace with no level-valued band (arm reach left-minus-right, for one) is
    simply absent, and the client draws no target for it.
    """
    from ..scoring.rubric import RUBRIC

    best: dict[str, tuple[tuple[int, float], dict]] = {}
    for criterion, bands in RUBRIC.items():
        for band in bands:
            meta = TRACES.get(band.feature)
            if not meta:
                continue
            label = meta[0]
            is_level = not band.feature.endswith(_SPREAD_SUFFIXES)
            rank = (1 if is_level else 0, band.weight)
            if label in best and best[label][0] >= rank:
                continue
            lo = None if band.mode == "lower" else band.ideal - band.tol
            hi = None if band.mode == "upper" else band.ideal + band.tol
            best[label] = (rank, {
                "criterion": criterion,
                "feature": band.feature,
                "ideal": round(float(band.ideal), 4),
                "tol": round(float(band.tol), 4),
                "mode": band.mode,
                # The acceptable region, so the client shades an area rather than
                # drawing a single line the reader would treat as a cliff edge.
                "acceptable": [None if lo is None else round(float(lo), 4),
                               None if hi is None else round(float(hi), 4)],
            })
    return {label: entry for label, (_, entry) in best.items()}


def _strides(events: list[float]) -> list[dict]:
    """Successive intervals between one foot's events, as ``{t, duration}``."""
    return [{"t": round(float(a), 3), "duration": round(float(b - a), 3)}
            for a, b in zip(events, events[1:])]


def _series(seq: PoseSequence) -> dict[str, np.ndarray]:
    """The per-frame traces, using the same expressions as the feature builder."""
    shoulder = seq.midpoint("left_shoulder", "right_shoulder")
    hip = seq.midpoint("left_hip", "right_hip")
    nose = seq.point("nose")
    la, ra = seq.point("left_ankle"), seq.point("right_ankle")

    pelvis = seq.meta.get("pelvis_raw")
    scale = np.clip(np.asarray(seq.meta.get("torso_scale", np.ones(seq.n_frames))), 1e-6, None)
    sway = (gait_highpass(pelvis[:, 0], seq.fps) / scale
            if pelvis is not None else np.zeros(seq.n_frames))

    reach_l = np.linalg.norm(seq.point("left_wrist") - seq.point("left_shoulder"), axis=1)
    reach_r = np.linalg.norm(seq.point("right_wrist") - seq.point("right_shoulder"), axis=1)
    leg_l = np.linalg.norm(la - seq.point("left_hip"), axis=1)
    leg_r = np.linalg.norm(ra - seq.point("right_hip"), axis=1)

    return {
        "Trunk lean": np.abs(angle_from_vertical(shoulder - hip)),
        "Arm reach": 0.5 * (reach_l + reach_r),
        "Arm reach L-R": np.abs(reach_l - reach_r),
        "Leg extension": np.minimum(leg_l, leg_r),
        "Knee lift": np.abs(gait_highpass(
            0.5 * (seq.point("left_knee")[:, 1] + seq.point("right_knee")[:, 1]), seq.fps)),
        "Neck angle": np.abs(angle_from_vertical(nose - shoulder)),
        "Shoulder tilt": np.abs(transverse_tilt(
            seq.point("left_shoulder") - seq.point("right_shoulder"))),
        "Hip drop": np.abs(transverse_tilt(
            seq.point("left_hip") - seq.point("right_hip"))),
        "Head off-centre": np.abs((nose - shoulder)[:, 0]),
        "Step width": np.abs(la[:, 0] - ra[:, 0]),
        "Lateral sway": np.abs(sway),
        "Shoulder sway": np.abs(gait_highpass(shoulder[:, 0], seq.fps)),
    }


def build(seq: PoseSequence, findings: list, max_points: int = 400) -> dict:
    """Timeline evidence for one clip.

    ``seq`` must be the **preprocessed** sequence, so the traces are in the same
    normalised units the features and the rubric thresholds use.
    """
    n = seq.n_frames
    step = max(1, -(-n // max_points))
    idx = np.arange(0, n, step)
    times = (idx / seq.fps).round(3)

    series = _series(seq)
    refs = _references()
    units = {label: unit for label, unit, _ in TRACES.values()}
    traces = {
        name: {
            "label": name,
            "unit": units.get(name, ""),
            "values": np.round(np.nan_to_num(values[idx], nan=0.0), 4).tolist(),
            # Where this trace sits relative to the band that scores it, so the
            # chart can shade tolerance instead of leaving the reader to guess
            # whether 9 degrees of lean is a lot.
            "reference": refs.get(name),
            "stats": {
                "min": round(float(np.nanmin(values)), 4),
                "mean": round(float(np.nanmean(values)), 4),
                "max": round(float(np.nanmax(values)), 4),
                "p95": round(float(np.nanpercentile(values, 95)), 4),
            } if values.size else None,
        }
        for name, values in series.items()
    }

    # Gait cycles, from the same phase-tracking the timing features use. The
    # ankles' lateral separation is the cleanest periodic signal in a frontal
    # view - the vertical trace is dominated by the perspective trend.
    gap = seq.point("left_ankle")[:, 0] - seq.point("right_ankle")[:, 0]
    cycles = cycle_times(gap, seq.fps)

    # Per-foot events come from the *signed gap between the ankles*, not from
    # each ankle's own vertical trace. Both ankles share the same perspective
    # drift as the subject approaches, and tracking their phase separately let
    # the two lock onto the same crossing - producing left and right "strikes"
    # at the identical instant, which is physically impossible. The gap is
    # differential, so it cancels the shared motion, and taking it with opposite
    # signs makes the two feet antiphase by construction.
    left_strikes = cycle_times(gap, seq.fps)
    right_strikes = cycle_times(-gap, seq.fps)

    # Where each reported fault is worst, so a finding can be jumped to.
    evidence = []
    for f in findings:
        meta = TRACES.get(f.feature)
        if not meta:
            continue
        label, unit, worse = meta
        values = series.get(label)
        if values is None or values.size == 0:
            continue
        j = int(np.nanargmax(values) if worse == "high" else np.nanargmin(values))
        evidence.append({
            "criterion": f.criterion,
            "feature": f.feature,
            "trace": label,
            "unit": unit,
            "t": round(float(j / seq.fps), 3),
            "value": round(float(values[j]), 4),
        })

    left_times = np.round(left_strikes, 3).tolist()
    right_times = np.round(right_strikes, 3).tolist()
    strides = {"left": _strides(left_times), "right": _strides(right_times)}
    durations = [s["duration"] for side in strides.values() for s in side]

    return {
        "duration_s": round(n / seq.fps, 3),
        "times": times.tolist(),
        "traces": traces,
        "cycles": np.round(cycles, 3).tolist(),
        "strikes": {"left": left_times, "right": right_times},
        # One entry per completed stride per foot, so cycle-to-cycle consistency
        # can be *seen*. Deliberately not summarised as a CV here: the rhythm
        # score's CV comes from each ankle's own phase track in the feature
        # builder, and a second CV computed from these events would differ
        # slightly and read as a contradiction.
        "strides": {
            **strides,
            "mean_s": round(float(np.mean(durations)), 3) if durations else None,
            "spread_s": round(float(max(durations) - min(durations)), 3)
                        if len(durations) > 1 else None,
            "basis": ("intervals between the foot events drawn above, taken from "
                      "the signed gap between the ankles"),
        },
        "evidence": evidence,
    }
