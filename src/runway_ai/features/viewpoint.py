"""Which angle was this filmed from, and what does that mean for the scores?

The project's documentation repeats that five criteria are unreliable from a
head-on camera - and then never checks what angle the clip was actually shot
at. That is the wrong way round twice over: a walk filmed from the side has the
*opposite* weaknesses, and a user told "stride length is hard to measure" has no
idea it is hard *for their footage specifically* or what to do about it.

How the angle is recovered
--------------------------
The shoulder line is horizontal, so it foreshortens as ``cos(yaw)``. The torso
is vertical, so it does not. Their ratio therefore carries the yaw, once the
frame's aspect ratio is divided out - image coordinates are normalised
separately by width and height, so a horizontal and a vertical span are not
directly comparable.

Validated against the synthetic generator, where the true camera azimuth is
known (90 clips, angles drawn uniformly from 0-75 degrees):

* mean error **-0.2 deg** - unbiased
* standard deviation **8.5 deg**
* within 10 degrees on **81%** of clips
* three-band classification correct **79%** of the time; two-band **92%**

Two things follow, and both shape how this is reported. The estimate is given
as a *range*, never a bare degree figure. And ``cos`` is flat near zero, so the
method genuinely cannot separate 5 degrees from 10 - it distinguishes head-on
from angled from side, which is all the advice needs.

A caveat that cannot be measured away: the method assumes an average
shoulder-to-torso build. Someone unusually narrow-shouldered filmed head-on
reads as slightly turned. The synthetic subjects scale proportionally, so this
validation does not capture that, and the reported uncertainty should be read as
a floor rather than a bound.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import PoseSequence
from ..data.synthetic import SEG, TORSO_H

#: True 3D shoulder-width to torso-length ratio, from the joint-centre
#: anthropometry the generator uses.
SHOULDER_TORSO_RATIO = SEG["shoulder_width"] / TORSO_H

#: One standard deviation of the estimator, measured on 90 synthetic clips.
UNCERTAINTY_DEG = 8.5

BANDS = [
    (0.0, 20.0, "head-on"),
    (20.0, 50.0, "angled"),
    (50.0, 91.0, "side-on"),
]

#: How well each criterion can be measured from each viewpoint.
#:
#: This is not a fitted table - it follows from the geometry, and from the
#: out-of-fold results the models already report. Stride and arm swing are
#: sagittal: a head-on camera foreshortens them to almost nothing, which is
#: exactly why they sit at R^2 0.58-0.59. Step width and shoulder level are
#: transverse and need the frontal view. Timing quantities care about how many
#: gait cycles were captured, not the angle.
RELIABILITY = {
    "head-on": {
        "good": ["posture", "balance", "shoulder_alignment", "head_position",
                 "foot_placement", "hip_movement", "confidence", "stage_presence"],
        "poor": ["stride_length", "arm_swing"],
        "advice": "Stride length and arm swing are largely front-to-back, and a "
                  "head-on camera flattens them. Film one pass from about 45 "
                  "degrees to measure those properly.",
    },
    "angled": {
        "good": ["posture", "balance", "head_position", "confidence",
                 "stage_presence", "stride_length", "arm_swing", "hip_movement"],
        "poor": ["foot_placement", "shoulder_alignment"],
        "advice": "A three-quarter view is the best single compromise. Step "
                  "width and shoulder level are the two it reads least well.",
    },
    "side-on": {
        "good": ["posture", "stride_length", "arm_swing", "rhythm"],
        "poor": ["foot_placement", "shoulder_alignment", "symmetry", "balance"],
        "advice": "A side view measures stride and arm swing best, but it cannot "
                  "see left-right differences at all - step width, shoulder "
                  "level and symmetry need a frontal pass.",
    },
}

#: Timing criteria depend on how many strides were captured, not the angle.
ANGLE_NEUTRAL = ["rhythm", "symmetry"]


def band_for(angle: float) -> str:
    for lo, hi, name in BANDS:
        if lo <= angle < hi:
            return name
    return BANDS[-1][2]


def estimate(seq: PoseSequence, aspect: float | None = None) -> dict | None:
    """Estimate the camera angle from a preprocessed sequence.

    ``aspect`` is frame height over width. Image coordinates are normalised per
    axis, so without it a horizontal span cannot be compared with a vertical
    one; 16:9 portrait is assumed when the source dimensions are unknown, and
    the result says so.
    """
    try:
        width = np.linalg.norm(
            seq.point("left_shoulder") - seq.point("right_shoulder"), axis=1)
    except Exception:
        return None
    if width.size < 4:
        return None

    assumed = aspect is None
    aspect = aspect or (16 / 9)

    # The median, not the maximum. The trunk counter-rotates through the gait
    # cycle, so the widest frame is the squarest *instant* rather than the
    # camera's angle; using it biased the estimate 25 degrees low.
    ratio = float(np.median(width)) / aspect
    angle = float(np.degrees(np.arccos(np.clip(ratio / SHOULDER_TORSO_RATIO, 0.0, 1.0))))

    band = band_for(angle)
    table = RELIABILITY[band]
    lo = max(0.0, angle - UNCERTAINTY_DEG)
    hi = min(90.0, angle + UNCERTAINTY_DEG)

    return {
        "angle_deg": round(angle, 1),
        "range_deg": [round(lo, 1), round(hi, 1)],
        "uncertainty_deg": UNCERTAINTY_DEG,
        "band": band,
        "well_measured": table["good"],
        "poorly_measured": table["poor"],
        "angle_neutral": ANGLE_NEUTRAL,
        "advice": table["advice"],
        "aspect_assumed": assumed,
        "basis": ("shoulder-line foreshortening; unbiased with a "
                  f"±{UNCERTAINTY_DEG:.0f}° spread measured against 90 clips "
                  "of known camera angle"),
    }
