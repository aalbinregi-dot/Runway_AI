"""Why each criterion scored what it did, in the units it was measured in.

The result payload carried the twelve scores and - for criteria under 7/10 - a
ranked list of the rubric bands that failed. Everything else the rubric had
already computed was thrown away: the bands that *passed*, how close each
measurement sat to its own tolerance, and how many of the hundred marks a
criterion actually cost. So a user could be told "balance 4.2" with no way to
see that their sway amplitude was 0.31 torso lengths against a 0.16 target, and
no way to know that balance is worth 12 of the 100 marks and lost 7 of them.

Two blocks are built here, both from data that already existed:

``breakdown``
    Every band of every criterion, measured value beside target and tolerance,
    with the criterion's weight turned into marks. Read straight from
    :data:`runway_ai.scoring.rubric.RUBRIC` through the same
    :func:`score_criterion` the score came from, so the explanation cannot drift
    from the number - including after ``rubric_calibration.json`` moves a
    threshold.

``measurements``
    The rest of the feature vector, grouped and labelled. 102 features are
    computed for every walk and twelve numbers were shown; these are the ones a
    person can act on or sanity-check, in the units the pipeline uses.

Marks, not band scores, are attributed to a criterion. A band's shortfall is
reported as its own 0-1 score and never converted into "marks lost", because
when the twelve learned models are in play the score is *theirs* - the bands
explain what the fault is, they do not add up to the mark. Pretending otherwise
would put a precise-looking number on an inference the code does not make.
"""
from __future__ import annotations

from ..config import CRITERIA, CRITERION_LABELS, CRITERION_WEIGHTS
from .rubric import RUBRIC, score_criterion

#: feature -> (human label, unit). One entry for every feature a rubric band
#: reads, plus the ones surfaced in :func:`measurements`.
#:
#: Units are the pipeline's own: ``torso`` is torso lengths, ``share`` a 0-1
#: fraction, ``cv`` a coefficient of variation, ``index`` a normalised left-right
#: difference, ``jerk`` the amplitude-normalised third derivative (no physical
#: unit, only comparable against its own band), ``cycles`` a fraction of one
#: gait cycle.
FEATURE_LABELS: dict[str, tuple[str, str]] = {
    # posture
    "posture_trunk_lean_mean": ("Trunk lean", "°"),
    "posture_trunk_lean_signed": ("Lean direction", "°"),
    "posture_trunk_lean_std": ("Trunk lean variation", "°"),
    "posture_neck_angle_mean": ("Neck angle", "°"),
    "posture_body_length_mean": ("Nose-to-hip length", "torso"),
    "posture_body_length_cv": ("Body length variation", "cv"),
    "posture_torso_foreshortening": ("Torso foreshortening", "share"),
    "posture_lean_jerk": ("Trunk correction rate", "jerk"),
    # balance
    "balance_sway_amp": ("Lateral sway", "torso"),
    "balance_sway_rms": ("Sway RMS", "torso"),
    "balance_bob_amp": ("Vertical bob", "torso"),
    "balance_sway_hf_ratio": ("High-frequency wobble", "share"),
    "balance_sway_jerk": ("Sway abruptness", "jerk"),
    "balance_upper_body_sway": ("Shoulder sway", "torso"),
    "balance_path_ratio": ("Path wander", "×"),
    # shoulders
    "shoulder_tilt_abs_mean": ("Shoulder tilt", "°"),
    "shoulder_tilt_bias": ("Shoulder tilt bias", "°"),
    "shoulder_tilt_amp": ("Shoulder see-saw", "°"),
    "shoulder_tilt_std": ("Shoulder tilt spread", "°"),
    "shoulder_width_mean": ("Shoulder width", "torso"),
    "shoulder_width_cv": ("Chest opening variation", "cv"),
    # head
    "head_height_mean": ("Head height", "torso"),
    "head_height_cv": ("Head height variation", "cv"),
    "head_lateral_offset": ("Head off-centre", "torso"),
    "head_lateral_amp": ("Head drift", "torso"),
    "head_tilt_mean": ("Head tilt", "°"),
    "head_stability_jerk": ("Head steadiness", "jerk"),
    # gaze (measured from the face, only present when one was visible)
    "gaze_on_camera_fraction": ("Eyeline forward", "share"),
    "gaze_looking_down_fraction": ("Chin dropped", "share"),
    "gaze_roll_abs_mean": ("Head roll", "°"),
    "gaze_yaw_abs_mean": ("Head turn", "°"),
    "gaze_pitch_mean": ("Head pitch", "°"),
    # arms
    "arm_swing_amp_mean": ("Arm swing amplitude", "torso"),
    "arm_swing_amp_error": ("Arm swing vs ideal", "torso"),
    "arm_swing_asymmetry": ("Arm swing L/R gap", "index"),
    "arm_swing_phase_offset": ("Arm-to-leg phase", "cycles"),
    "arm_left_reach_amp": ("Left arm reach swing", "torso"),
    "arm_right_reach_amp": ("Right arm reach swing", "torso"),
    "arm_left_elbow_angle_mean": ("Left elbow angle", "°"),
    "arm_right_elbow_angle_mean": ("Right elbow angle", "°"),
    # hips
    "hip_obliquity_amp": ("Hip drop", "°"),
    "hip_obliquity_bias": ("Hip drop bias", "°"),
    "hip_obliquity_std": ("Hip line stability", "°"),
    "hip_width_mean": ("Hip width", "torso"),
    "hip_shoulder_ratio_mean": ("Hip-to-shoulder width", "×"),
    "hip_counter_rotation_amp": ("Pelvis counter-rotation", "torso"),
    "hip_vertical_osc": ("Pelvis rise and fall", "torso"),
    # rhythm
    "rhythm_stride_freq": ("Stride rate", "Hz"),
    "rhythm_freq_error": ("Cadence vs ideal", "Hz"),
    "rhythm_spectral_purity": ("Tempo purity", "share"),
    "rhythm_step_interval": ("Step interval", "s"),
    "rhythm_stride_cv": ("Stride timing spread", "cv"),
    "rhythm_cadence_spm": ("Cadence", "spm"),
    "rhythm_n_steps": ("Steps detected", ""),
    # stride
    "stride_leg_shorten_mean": ("Leg obliquity at strike", "share"),
    "stride_leg_extension_mean": ("Leg extension", "torso"),
    "stride_knee_lift": ("Knee lift", "torso"),
    "stride_foot_clearance": ("Foot clearance", "torso"),
    "stride_ankle_sep_max": ("Widest ankle separation", "torso"),
    "stride_approach_rate": ("Approach rate", "/s"),
    "stride_length_est": ("Stride estimate", "torso"),
    # feet
    "foot_step_width_mean": ("Step width", "torso"),
    "foot_step_width_at_strike": ("Step width at strike", "torso"),
    "foot_placement_var": ("Step width spread", "torso"),
    "foot_crossover_rate": ("Midline crossings", "/s"),
    "foot_lateral_amp_left": ("Left foot travel", "torso"),
    "foot_lateral_amp_right": ("Right foot travel", "torso"),
    # symmetry
    "sym_overall_index": ("Left/right amplitude gap", "index"),
    "sym_ankle_index": ("Leg swing L/R gap", "index"),
    "sym_wrist_index": ("Wrist swing L/R gap", "index"),
    "sym_knee_index": ("Knee lift L/R gap", "index"),
    "sym_elbow_index": ("Elbow travel L/R gap", "index"),
    "sym_ankle_phase_offset": ("Ankle phase offset", "cycles"),
    "sym_step_time_asymmetry": ("Step-time asymmetry", "index"),
    # confidence / presence
    "conf_speed_mean": ("Approach speed", "/s"),
    "conf_speed_cv": ("Pace consistency", "cv"),
    "conf_com_jerk": ("Movement hesitancy", "jerk"),
    "conf_chest_openness": ("Chest openness", "torso"),
    "conf_arm_spread": ("Wrist separation", "torso"),
    "conf_motion_energy": ("Motion energy", "/s"),
    # recording quality
    "qc_duration_s": ("Clip length", "s"),
    "qc_fps": ("Frame rate", "fps"),
    "qc_n_frames": ("Frames tracked", ""),
    "qc_mean_visibility": ("Mean landmark confidence", "share"),
    "qc_min_visibility": ("Weakest landmark", "share"),
    "qc_gap_fraction": ("Interpolated landmarks", "share"),
    "qc_torso_scale_ratio": ("Apparent size change", "×"),
}

#: Which measurements to surface, in reading order. Not the whole 102-column
#: vector: the rest are intermediate terms whose value carries no meaning on its
#: own, and listing them would bury the ones that do.
MEASUREMENT_GROUPS: list[tuple[str, str, list[str]]] = [
    ("timing", "Timing", [
        "rhythm_cadence_spm", "rhythm_stride_freq", "rhythm_step_interval",
        "rhythm_n_steps", "rhythm_stride_cv", "rhythm_spectral_purity",
    ]),
    ("travel", "Travel", [
        "stride_leg_shorten_mean", "stride_knee_lift", "stride_foot_clearance",
        "stride_leg_extension_mean", "stride_ankle_sep_max", "stride_approach_rate",
    ]),
    ("posture", "Posture", [
        "posture_trunk_lean_mean", "posture_trunk_lean_signed",
        "posture_trunk_lean_std", "posture_neck_angle_mean",
        "posture_torso_foreshortening", "head_height_cv",
    ]),
    ("balance", "Balance", [
        "balance_sway_amp", "balance_sway_rms", "balance_bob_amp",
        "balance_upper_body_sway", "balance_sway_hf_ratio", "balance_path_ratio",
    ]),
    ("arms", "Arms", [
        "arm_swing_amp_mean", "arm_left_reach_amp", "arm_right_reach_amp",
        "arm_swing_asymmetry", "arm_swing_phase_offset", "conf_arm_spread",
    ]),
    ("hips", "Hips and shoulders", [
        "hip_obliquity_amp", "hip_obliquity_bias", "hip_counter_rotation_amp",
        "hip_vertical_osc", "shoulder_tilt_abs_mean", "shoulder_width_mean",
    ]),
    ("feet", "Feet", [
        "foot_step_width_mean", "foot_step_width_at_strike",
        "foot_placement_var", "foot_crossover_rate",
        "foot_lateral_amp_left", "foot_lateral_amp_right",
    ]),
    ("symmetry", "Left versus right", [
        "sym_overall_index", "sym_ankle_index", "sym_wrist_index",
        "sym_knee_index", "sym_ankle_phase_offset", "sym_step_time_asymmetry",
    ]),
    ("gaze", "Gaze", [
        "gaze_on_camera_fraction", "gaze_looking_down_fraction",
        "gaze_roll_abs_mean", "gaze_yaw_abs_mean",
    ]),
    ("recording", "Recording", [
        "qc_duration_s", "qc_fps", "qc_n_frames", "qc_mean_visibility",
        "qc_gap_fraction", "qc_torso_scale_ratio",
    ]),
]

#: The band score at exactly one tolerance out. The rubric's falloff is
#: ``1 / (1 + (excess/tol)^2)``, so this is the tolerance edge itself rather than
#: a second, separately chosen threshold.
IN_TOLERANCE = 0.5


def label_of(feature: str) -> str:
    """Human label for a feature key, falling back to a readable form of it."""
    entry = FEATURE_LABELS.get(feature)
    return entry[0] if entry else feature.replace("_", " ")


def unit_of(feature: str) -> str:
    entry = FEATURE_LABELS.get(feature)
    return entry[1] if entry else ""


def _acceptable(band, measured: float) -> list[float]:
    """The band's acceptable interval, as ``[lo, hi]`` with None for open ends.

    Drawn rather than described in the client: a target with a tolerance is a
    region, and rendering it as a single tick invites the reader to treat a
    measurement one tolerance out as a failure when it scores 0.5.
    """
    if band.mode == "lower":
        return [None, band.ideal + band.tol]
    if band.mode == "upper":
        return [band.ideal - band.tol, None]
    return [band.ideal - band.tol, band.ideal + band.tol]


def band_details(criterion: str, feats: dict) -> list[dict]:
    """Every band of one criterion, measured beside its target.

    Bands whose feature is absent are skipped, exactly as
    :func:`~runway_ai.scoring.rubric.score_criterion` skips them - so a
    keypoint upload with no face does not report seven head-position bands of
    which three were never evaluated.
    """
    return _detail(criterion, feats)[1]


def _detail(criterion: str, feats: dict) -> tuple[float, list[dict]]:
    """``(rubric score, band rows)`` for one criterion, scoring it once."""
    rubric_score, scored = score_criterion(criterion, feats)
    total_w = sum(b.weight for b, _ in scored) or 1.0
    out = []
    for band, score in scored:
        measured = float(feats[band.feature])
        lo, hi = _acceptable(band, measured)
        # Read off the interval, not off the rounded score. Both express "excess
        # is within one tolerance", but the score is rounded to three decimals
        # for the payload, and a band sitting exactly on the edge then reported
        # 0.5 alongside "out of tolerance" - the two statements the client draws
        # side by side, contradicting each other at the one value where a reader
        # is most likely to look closely.
        in_tolerance = ((lo is None or measured >= lo - 1e-12)
                        and (hi is None or measured <= hi + 1e-12))
        out.append({
            "feature": band.feature,
            "label": label_of(band.feature),
            "unit": unit_of(band.feature),
            "measured": round(measured, 4),
            "ideal": round(float(band.ideal), 4),
            "tol": round(float(band.tol), 4),
            "mode": band.mode,
            "acceptable": [None if v is None else round(float(v), 4)
                           for v in (lo, hi)],
            # 0-1: how well this one measurement satisfied its band.
            "score": round(float(score), 3),
            "in_tolerance": in_tolerance,
            # Relative pull inside the criterion, so a reader can see that a
            # failing 0.5-weight band matters less than a passing 1.6-weight one.
            "share": round(float(band.weight / total_w), 3),
            "message": band.message,
        })
    return rubric_score, sorted(out, key=lambda b: (b["score"], -b["share"]))


def breakdown(feats: dict, scores: dict[str, float]) -> dict:
    """Per-criterion explanation: marks, weight, and every band behind the score.

    ``marks_earned`` is the criterion's actual contribution to the overall 0-100
    - ``score x weight x 10`` - and ``marks_available`` is what it could have
    contributed. That pair is what makes a 4.2 actionable: a 4.2 on posture
    costs 7.5 marks, a 4.2 on stage presence costs 2.9, and the same number on
    the same 0-10 scale is worth two and a half times as much in the first case.
    """
    out = {}
    for crit in CRITERIA:
        score = float(scores.get(crit, 0.0))
        weight = CRITERION_WEIGHTS[crit]
        rubric_score, bands = _detail(crit, feats)
        out[crit] = {
            "label": CRITERION_LABELS[crit],
            "score": round(score, 2),
            "weight": weight,
            "marks_earned": round(score * weight * 10.0, 2),
            "marks_available": round(weight * 100.0, 2),
            "marks_lost": round((10.0 - score) * weight * 10.0, 2),
            # What the transparent rubric alone made of it. Shown beside the
            # model's score rather than instead of it: where the two disagree
            # sharply, that disagreement is itself information.
            "rubric_score": round(float(rubric_score), 2),
            "bands": bands,
            "n_bands": len(bands),
            "n_out_of_tolerance": sum(1 for b in bands if not b["in_tolerance"]),
        }
    return out


def measurements(feats: dict) -> list[dict]:
    """The measured quantities behind the scores, grouped and labelled.

    Every entry carries ``scored_in`` - the criterion whose rubric reads this
    feature, or None. That is the difference between a number a user can chase
    and a number that is context: ``foot_step_width_at_strike`` is scored,
    ``foot_lateral_amp_left`` explains it.
    """
    scored_in = {band.feature: crit for crit in CRITERIA for band in RUBRIC[crit]}
    groups = []
    for key, title, features in MEASUREMENT_GROUPS:
        rows = []
        for feature in features:
            if feature not in feats:
                continue          # gaze on a keypoint upload, for instance
            rows.append({
                "feature": feature,
                "label": label_of(feature),
                "unit": unit_of(feature),
                "value": round(float(feats[feature]), 4),
                "scored_in": scored_in.get(feature),
            })
        if rows:
            groups.append({"key": key, "title": title, "rows": rows})
    return groups
