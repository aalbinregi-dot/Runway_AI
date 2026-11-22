"""Deterministic biomechanical rubric.

This is the transparent, non-learned scorer. It maps features to 0-10 scores
through explicit tolerance bands taken from gait-analysis norms and runway
coaching practice, and it exists for three reasons:

* **Cold start.** It scores a walk before any model has been trained.
* **Baseline.** The learned model has to beat it, or the model is not earning
  its complexity.
* **Explanation.** Each band carries a human-readable statement of what was
  measured, which is what the coaching layer turns into advice.

Each band is scored with a smooth one- or two-sided penalty rather than a hard
threshold, so a walk just outside tolerance loses a little and a walk far
outside loses a lot - no cliff edges at the boundary.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import CRITERIA, CRITERION_WEIGHTS, GRADE_BANDS, MODEL_DIR


@dataclass(frozen=True)
class Band:
    """One scored measurement.

    Parameters
    ----------
    feature: feature key to read.
    ideal: target value (or the edge of the acceptable region for one-sided bands).
    tol: deviation at which roughly half the marks for this band are lost.
    mode: ``"target"`` penalises deviation in both directions; ``"lower"``
        penalises only values above ``ideal``; ``"upper"`` only below.
    weight: relative importance of this band within its criterion.
    message: what to tell the user when this band scores poorly.
    """

    feature: str
    ideal: float
    tol: float
    mode: str = "target"
    weight: float = 1.0
    message: str = ""

    def score(self, value: float) -> float:
        """Return 0-1 for this band."""
        if self.mode == "lower":
            excess = max(0.0, value - self.ideal)
        elif self.mode == "upper":
            excess = max(0.0, self.ideal - value)
        else:
            excess = abs(value - self.ideal)
        # Rational falloff: 1.0 at the target, 0.5 at one tolerance out.
        return float(1.0 / (1.0 + (excess / max(self.tol, 1e-9)) ** 2))


#: Tolerance bands per criterion. Values are in the feature units documented in
#: :mod:`runway_ai.features.builder` - torso lengths and degrees.
RUBRIC: dict[str, list[Band]] = {
    "posture": [
        Band("posture_trunk_lean_mean", 5.0, 8.0, "lower", 1.4,
             "Your torso leans forward through the walk; stack your ribcage over your hips."),
        Band("posture_trunk_lean_std", 2.0, 4.0, "lower", 0.8,
             "Your trunk angle keeps changing - hold one line rather than rocking."),
        Band("posture_neck_angle_mean", 8.0, 10.0, "lower", 0.9,
             "Your neck juts forward of your shoulders; lengthen the back of the neck."),
        Band("posture_lean_jerk", 0.0, 2500.0, "lower", 0.5,
             "Your upper body corrects itself constantly instead of holding a line."),
    ],
    "balance": [
        Band("balance_sway_amp", 0.18, 0.30, "lower", 1.4,
             "Your centre of mass swings side to side more than it should."),
        Band("balance_sway_hf_ratio", 0.10, 0.20, "lower", 1.2,
             "There is high-frequency wobble in your path - you are catching your balance mid-step."),
        Band("balance_upper_body_sway", 0.18, 0.28, "lower", 0.9,
             "Your shoulders travel laterally with each step instead of staying quiet."),
        Band("balance_sway_jerk", 0.0, 3000.0, "lower", 0.7,
             "Your balance corrections are abrupt rather than smooth."),
    ],
    "confidence": [
        Band("conf_arm_spread", 1.6, 0.7, "lower", 1.3,
             "You are holding your arms away from your body, which reads as tension."),
        Band("conf_speed_cv", 1.5, 1.2, "lower", 1.0,
             "Your pace is inconsistent; commit to one tempo down the runway."),
        Band("conf_com_jerk", 0.0, 3000.0, "lower", 0.8,
             "Your movement is hesitant rather than decisive."),
    ],
    "shoulder_alignment": [
        Band("shoulder_tilt_abs_mean", 2.5, 6.0, "lower", 1.5,
             "One shoulder sits consistently lower than the other - level the shoulder line."),
        Band("shoulder_tilt_amp", 4.0, 7.0, "lower", 1.0,
             "Your shoulders see-saw with each step; keep the shoulder line quiet."),
        Band("shoulder_width_cv", 0.06, 0.12, "lower", 0.6,
             "Your chest opens and closes as you walk instead of staying open."),
    ],
    "head_position": [
        Band("posture_neck_angle_mean", 8.0, 10.0, "lower", 1.2,
             "Your chin drops as you walk; lift the gaze to the end of the runway."),
        Band("head_lateral_offset", 0.10, 0.18, "lower", 1.2,
             "Your head sits off the centre line of your body."),
        Band("head_lateral_amp", 0.12, 0.18, "lower", 1.0,
             "Your head drifts side to side with each step."),
        Band("head_stability_jerk", 0.0, 3000.0, "lower", 0.6,
             "Your head is unsteady - fix your eyeline on one point."),
        Band("gaze_on_camera_fraction", 0.85, 0.30, "upper", 1.3,
             "Your eyeline leaves the camera for much of the walk - fix it on "
             "the end of the runway and keep it there."),
        Band("gaze_looking_down_fraction", 0.05, 0.20, "lower", 1.2,
             "Your chin drops toward the floor. Lift the gaze; the head follows."),
        Band("gaze_roll_abs_mean", 2.0, 5.0, "lower", 0.5,
             "Your head is habitually tilted to one side."),
    ],
    "arm_swing": [
        Band("arm_swing_amp_error", 0.0, 0.35, "lower", 1.4,
             "Your arm swing is off - aim for a relaxed, even swing from the shoulder."),
        Band("arm_swing_asymmetry", 0.20, 0.50, "lower", 1.1,
             "Your two arms swing by different amounts."),
        Band("arm_swing_phase_offset", 0.50, 0.35, "target", 0.8,
             "Your arms are not opposing your legs; the swing should alternate with the stride."),
    ],
    "hip_movement": [
        Band("hip_obliquity_amp", 6.0, 10.0, "lower", 1.4,
             "Your pelvis drops on each step - strengthen the stance-side hip."),
        Band("hip_obliquity_std", 3.0, 6.0, "lower", 1.0,
             "Your hip line is unstable through the stride."),
        Band("hip_counter_rotation_amp", 0.12, 0.20, "lower", 0.7,
             "Your pelvis rotates more than your walk needs."),
    ],
    "rhythm": [
        Band("rhythm_stride_cv", 0.10, 0.22, "lower", 1.6,
             "Your step timing is uneven - work to a metronome to even out the tempo."),
        Band("rhythm_spectral_purity", 0.65, 0.35, "upper", 1.2,
             "Your walk has no single steady tempo."),
        Band("rhythm_freq_error", 0.0, 0.45, "lower", 0.7,
             "Your cadence is away from a natural runway tempo."),
    ],
    "stride_length": [
        Band("stride_leg_shorten_mean", 0.035, 0.030, "target", 1.4,
             "Your stride length is away from ideal for your height."),
        Band("stride_knee_lift", 0.10, 0.10, "target", 1.0,
             "Your knee lift does not match a full, controlled stride."),
        Band("stride_foot_clearance", 0.25, 0.15, "target", 0.7,
             "Your feet are not clearing the floor cleanly."),
    ],
    "foot_placement": [
        Band("foot_step_width_at_strike", 0.18, 0.25, "lower", 1.6,
             "Your feet land wide of the line; runway walking places each foot on one line."),
        Band("foot_placement_var", 0.15, 0.20, "lower", 1.0,
             "Your foot placement varies from step to step."),
        Band("foot_crossover_rate", 1.8, 1.5, "lower", 0.6,
             "Your feet cross the midline too aggressively."),
    ],
    "symmetry": [
        Band("sym_overall_index", 0.15, 0.25, "lower", 1.5,
             "Your left and right sides move by noticeably different amounts."),
        Band("sym_step_time_asymmetry", 0.20, 0.55, "lower", 1.2,
             "Your left/right timing is uneven - one side spends longer on the ground."),
        Band("sym_ankle_index", 0.10, 0.22, "lower", 0.9,
             "Your two legs are not swinging equally."),
    ],
    "stage_presence": [
        Band("conf_chest_openness", 0.70, 0.25, "upper", 1.3,
             "Open the chest and draw the shoulder blades back - presence starts with carriage."),
        Band("head_height_cv", 0.08, 0.14, "lower", 1.0,
             "Your head height wanders; hold a still, lifted carriage."),
        Band("hip_counter_rotation_amp", 0.08, 0.10, "upper", 0.8,
             "Add a little controlled torso rotation - the walk currently reads as stiff."),
        Band("head_stability_jerk", 0.0, 3000.0, "lower", 0.7,
             "Small extraneous movements are diluting your presence."),
    ],
}

assert set(RUBRIC) == set(CRITERIA), "rubric must cover every criterion"


#: Per-criterion affine rescale of the raw band average, filled by calibration.
AFFINE: dict[str, tuple[float, float]] = {}


def apply_calibration_dict(cal: dict) -> None:
    """Install fitted band constants from a calibration dict.

    ``AFFINE`` is cleared first. It is module-level state, and the calibration
    script re-enters this function while the previous run's values are still
    loaded; without the reset the raw scores it measures come back already
    transformed, and it fits a second slope on top of the first. Each round of
    calibration then compounds the last, which is how the rubric ended up
    scoring worse than predicting the mean.

    .. note::

       The affine rescale terms (``_affine``) are deliberately **not loaded**.
       They were fitted to map the rubric's raw 0-10 output onto the synthetic
       label scale, but the slopes (0.26 for balance, 0.35 for posture, etc.)
       compress the entire output range to ~3 points.  A rubric that can only
       produce scores between 3.9 and 6.5 cannot distinguish a beginner from a
       professional - the fix is to let the band scores speak at full range.
    """
    AFFINE.clear()
    for crit, bands in cal.items():
        if crit not in RUBRIC:
            continue
        RUBRIC[crit] = [
            Band(b.feature, bands[b.feature]["ideal"], bands[b.feature]["tol"],
                 b.mode, b.weight, b.message)
            if b.feature in bands else b
            for b in RUBRIC[crit]
        ]
        # Affine rescale intentionally skipped - see docstring above.


def apply_calibration(path: Path | None = None) -> bool:
    """Overwrite the hand-set ``ideal``/``tol`` constants with fitted ones.

    The bands above encode which direction each fault runs in - that part is
    domain knowledge and stays fixed. The numeric thresholds are a different
    matter: they live in normalised feature units that no textbook quotes, so
    they are measured by ``scripts/calibrate_rubric.py`` and loaded here. The
    messages and weights are untouched.

    After loading, a tolerance floor is applied: any band whose tolerance is
    less than 20% of its ideal gets widened.  The calibration was fitted on
    synthetic data whose features have very small variance; tolerances like
    0.048 for ``conf_speed_cv`` (ideal 2.23) or 0.001 for ``stride_knee_lift``
    (ideal 0.21) floor the band score on virtually every real-world input,
    making the rubric unable to distinguish different quality levels.

    Returns True if a calibration file was found and applied.
    """
    path = Path(path or MODEL_DIR / "rubric_calibration.json")
    if not path.exists():
        return False
    apply_calibration_dict(json.loads(path.read_text()))
    _apply_tolerance_floors()
    return True


#: Minimum tolerance as a fraction of the ideal value.
_MIN_TOL_FRAC = 0.35
#: Absolute floor for near-zero ideals (jerk indices, phase offsets, etc.)
_MIN_TOL_ABS = 0.08


def _apply_tolerance_floors() -> None:
    """Widen any calibrated tolerance that is too tight for real-world data."""
    for crit in RUBRIC:
        RUBRIC[crit] = [
            Band(b.feature, b.ideal,
                 max(b.tol, abs(b.ideal) * _MIN_TOL_FRAC, _MIN_TOL_ABS),
                 b.mode, b.weight, b.message)
            for b in RUBRIC[crit]
        ]


# Applied at import so every consumer - CLI, API, dashboard - scores the same.
CALIBRATED = apply_calibration()


def score_criterion(criterion: str, feats: dict) -> tuple[float, list[tuple[Band, float]]]:
    """Score one criterion, returning ``(score_0_10, per_band_results)``."""
    bands = RUBRIC[criterion]
    results, total_w, acc = [], 0.0, 0.0
    for band in bands:
        if band.feature not in feats:
            continue
        s = band.score(float(feats[band.feature]))
        results.append((band, s))
        acc += s * band.weight
        total_w += band.weight
    score = 10.0 * (acc / total_w) if total_w > 0 else 5.0
    if criterion in AFFINE:                      # put it on the label scale
        slope, intercept = AFFINE[criterion]
        score = slope * score + intercept
    return float(np.clip(score, 0.0, 10.0)), results


def rubric_scores(feats: dict) -> dict[str, float]:
    """Score all twelve criteria from a feature dict."""
    return {c: score_criterion(c, feats)[0] for c in CRITERIA}


def overall_score(scores: dict[str, float]) -> float:
    """Weighted 0-100 aggregate of the twelve criterion scores."""
    total = sum(scores[c] * CRITERION_WEIGHTS[c] for c in CRITERIA if c in scores)
    return float(np.clip(total * 10.0, 0.0, 100.0))


def grade(overall: float) -> str:
    """Map a 0-100 overall score to a named performance band."""
    for threshold, name in GRADE_BANDS:
        if overall >= threshold:
            return name
    return GRADE_BANDS[-1][1]
