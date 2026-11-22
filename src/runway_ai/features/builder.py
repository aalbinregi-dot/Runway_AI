"""Turn a ``PoseSequence`` into a flat, interpretable feature vector.

Features are grouped by the criterion they primarily inform, and every name is
prefixed with that group. Keeping the mapping explicit means a low predicted
``balance`` score can always be traced back to the measurements that caused it,
which is what the coaching layer needs in order to say something specific.

All positional features are in **torso lengths** and all angles in **degrees**,
so they are comparable across subjects, camera distances and resolutions.
"""
from __future__ import annotations

import numpy as np

from ..data.schema import PoseSequence
from .signals import (
    amplitude,
    angle_from_horizontal,
    transverse_tilt,
    angle_from_vertical,
    gait_highpass,
    dominant_frequency,
    high_freq_ratio,
    interval_cv,
    jerk_index,
    joint_angle,
    phase_offset,
    rms,
    safe_div,
    cycle_times,
    step_events,
    symmetry_index,
)

IDEAL_STRIDE_TORSO = 2.7      # ~0.78 body height expressed in torso lengths
IDEAL_ARM_SWING_TORSO = 0.42  # wrist excursion for a relaxed, open swing
IDEAL_CADENCE_HZ = 0.95       # strides per second for a deliberate runway walk


def _span(x: np.ndarray) -> float:
    """Robust peak-to-peak (5-95 percentile) of an already-detrended signal."""
    return float(np.percentile(x, 95) - np.percentile(x, 5))


def _pelvis_deviation(seq: PoseSequence) -> np.ndarray:
    """Pelvis deviation from its own smooth path, in torso lengths, (T, 2).

    ``normalise`` pins the pelvis to the origin, so the body's actual motion has
    to come from the raw image track stashed in ``meta``. Sway is a property of
    the whole body, not of the landmarks relative to it.

    Order of operations matters. The high-pass runs **in image units first**,
    and only the residual is divided by the local torso scale. Dividing the raw
    pelvis *position* by a scale that itself changes threefold as the subject
    approaches produces a huge spurious swing - it measures the perspective, not
    the walker. High-passing first leaves a small local displacement, and
    dividing that by the apparent body size converts it honestly into torso
    lengths.
    """
    pelvis = seq.meta.get("pelvis_raw")
    scale = seq.meta.get("torso_scale")
    if pelvis is None or scale is None:
        return np.zeros((seq.n_frames, 2))
    scale = np.clip(np.asarray(scale), 1e-6, None)
    return np.stack([
        gait_highpass(pelvis[:, 0], seq.fps) / scale,
        gait_highpass(pelvis[:, 1], seq.fps) / scale,
    ], axis=1)


# --------------------------------------------------------------------------- #
# Feature groups
# --------------------------------------------------------------------------- #
def posture_features(seq: PoseSequence) -> dict:
    shoulder = seq.midpoint("left_shoulder", "right_shoulder")
    hip = seq.midpoint("left_hip", "right_hip")
    lean = angle_from_vertical(shoulder - hip)
    neck = angle_from_vertical(seq.point("nose") - shoulder)

    # Measuring the pelvis-to-shoulder distance here would be pointless: that is
    # exactly the quantity ``normalise`` divides by, so it is 1.0 in every frame
    # by construction. Nose-to-pelvis is not normalised away, and it shortens
    # when the walker slouches or leans toward the camera.
    body_len = np.linalg.norm(seq.point("nose") - hip, axis=1)
    # Residual variation in the *raw* apparent torso size, after removing the
    # perspective trend: the trunk foreshortening as it rotates or pitches.
    scale = np.asarray(seq.meta.get("torso_scale", np.ones(seq.n_frames)))
    fore = safe_div(float(np.std(gait_highpass(scale, seq.fps))), float(np.mean(scale)))

    return {
        "posture_trunk_lean_mean": float(np.mean(np.abs(lean))),
        "posture_trunk_lean_signed": float(np.mean(lean)),
        "posture_trunk_lean_std": float(np.std(lean)),
        "posture_neck_angle_mean": float(np.mean(np.abs(neck))),
        "posture_body_length_mean": float(np.mean(body_len)),
        "posture_body_length_cv": safe_div(float(np.std(body_len)), float(np.mean(body_len))),
        "posture_torso_foreshortening": fore,
        "posture_lean_jerk": jerk_index(lean, seq.fps),
    }


def balance_features(seq: PoseSequence) -> dict:
    com = _pelvis_deviation(seq)          # already gait-band, in torso lengths
    sway_x, bob_y = com[:, 0], com[:, 1]
    shoulder = seq.midpoint("left_shoulder", "right_shoulder")
    return {
        "balance_sway_amp": _span(sway_x),
        "balance_sway_rms": rms(sway_x),
        "balance_bob_amp": _span(bob_y),
        "balance_sway_hf_ratio": high_freq_ratio(com[:, 0], seq.fps),
        "balance_sway_jerk": jerk_index(com[:, 0], seq.fps),
        "balance_upper_body_sway": amplitude(shoulder[:, 0], seq.fps),
        "balance_path_ratio": safe_div(
            float(np.sum(np.abs(np.diff(sway_x)))), float(np.ptp(sway_x) + 1e-9)
        ),
    }


def shoulder_features(seq: PoseSequence) -> dict:
    vec = seq.point("left_shoulder") - seq.point("right_shoulder")
    tilt = transverse_tilt(vec)
    width = np.linalg.norm(vec, axis=1)
    return {
        "shoulder_tilt_bias": float(np.mean(tilt)),
        "shoulder_tilt_abs_mean": float(np.mean(np.abs(tilt))),
        "shoulder_tilt_amp": amplitude(tilt, seq.fps),
        "shoulder_tilt_std": float(np.std(tilt)),
        "shoulder_width_mean": float(np.mean(width)),
        "shoulder_width_cv": safe_div(float(np.std(width)), float(np.mean(width))),
    }


def head_features(seq: PoseSequence) -> dict:
    nose = seq.point("nose")
    shoulder = seq.midpoint("left_shoulder", "right_shoulder")
    rel = nose - shoulder
    # In normalised coordinates a lifted head sits far above the shoulder line;
    # a dropped chin shortens that vertical gap and shifts the nose forward.
    height = -rel[:, 1]
    return {
        "head_height_mean": float(np.mean(height)),
        "head_height_cv": safe_div(float(np.std(height)), float(np.mean(np.abs(height)))),
        "head_lateral_offset": float(np.mean(np.abs(rel[:, 0]))),
        "head_lateral_amp": amplitude(rel[:, 0], seq.fps),
        "head_tilt_mean": float(np.mean(np.abs(angle_from_vertical(rel)))),
        "head_stability_jerk": jerk_index(nose[:, 0], seq.fps),
    }


def arm_features(seq: PoseSequence) -> dict:
    """Arm-swing features.

    Arm swing is mostly *sagittal*, so in the near-frontal view a runway camera
    gives it barely projects into image ``y`` at all. What it does do is change
    the apparent shoulder-to-wrist distance: the limb foreshortens as it swings
    toward and away from the lens. That reach oscillation is therefore the
    primary amplitude measurement here, with the raw vertical excursion kept
    only as a secondary cue.
    """
    out = {}
    amps = {}
    for side in ("left", "right"):
        wrist = seq.point(f"{side}_wrist")
        elbow = seq.point(f"{side}_elbow")
        shoulder = seq.point(f"{side}_shoulder")
        hip = seq.point(f"{side}_hip")
        ang = joint_angle(shoulder, elbow, wrist)
        reach = np.linalg.norm(wrist - shoulder, axis=1)

        amp = amplitude(reach, seq.fps)
        amps[side] = amp
        out[f"arm_{side}_reach_amp"] = amp
        out[f"arm_{side}_reach_mean"] = float(np.mean(reach))
        out[f"arm_{side}_swing_amp"] = amplitude(wrist[:, 1], seq.fps)
        out[f"arm_{side}_lateral_amp"] = amplitude(wrist[:, 0], seq.fps)
        out[f"arm_{side}_elbow_angle_mean"] = float(np.mean(ang))
        out[f"arm_{side}_elbow_angle_std"] = float(np.std(ang))
        out[f"arm_{side}_wrist_hip_dist"] = float(np.mean(np.linalg.norm(wrist - hip, axis=1)))

    mean_amp = 0.5 * (amps["left"] + amps["right"])
    out["arm_swing_amp_mean"] = mean_amp
    out["arm_swing_amp_error"] = abs(mean_amp - IDEAL_ARM_SWING_TORSO)
    out["arm_swing_asymmetry"] = symmetry_index(amps["left"], amps["right"])
    out["arm_swing_phase_offset"] = phase_offset(
        np.linalg.norm(seq.point("left_wrist") - seq.point("left_shoulder"), axis=1),
        np.linalg.norm(seq.point("right_wrist") - seq.point("right_shoulder"), axis=1),
        seq.fps,
    )
    return out


def hip_features(seq: PoseSequence) -> dict:
    vec = seq.point("left_hip") - seq.point("right_hip")
    obliquity = transverse_tilt(vec)
    width = np.linalg.norm(vec, axis=1)
    shoulder_w = np.linalg.norm(
        seq.point("left_shoulder") - seq.point("right_shoulder"), axis=1
    )
    com = _pelvis_deviation(seq)
    # Shoulder/hip width ratio oscillates as the trunk counter-rotates against
    # the pelvis; its amplitude is a projection-robust proxy for that rotation.
    ratio = width / np.clip(shoulder_w, 1e-6, None)
    return {
        "hip_obliquity_amp": amplitude(obliquity, seq.fps),
        "hip_obliquity_bias": float(np.mean(obliquity)),
        "hip_obliquity_std": float(np.std(obliquity)),
        "hip_width_mean": float(np.mean(width)),
        "hip_shoulder_ratio_mean": float(np.mean(ratio)),
        "hip_counter_rotation_amp": amplitude(ratio, seq.fps),
        "hip_vertical_osc": _span(com[:, 1]),
    }


def rhythm_features(seq: PoseSequence) -> dict:
    lay = seq.point("left_ankle")[:, 1]
    ray = seq.point("right_ankle")[:, 1]
    gap = seq.point("left_ankle")[:, 0] - seq.point("right_ankle")[:, 0]

    freq, purity = dominant_frequency(gap, seq.fps)
    # Sub-frame cycle boundaries per foot; these carry the timing information.
    t_l = cycle_times(lay, seq.fps)
    t_r = cycle_times(ray, seq.fps)
    iv_l, cv_l = interval_cv(t_l)
    iv_r, cv_r = interval_cv(t_r)

    all_ev = np.sort(np.concatenate([t_l, t_r])) if (t_l.size + t_r.size) else np.array([])
    step_iv, step_cv = interval_cv(all_ev)

    # Per-foot stride CV is the clinical regularity measure and the one to
    # trust. Pooling left and right strikes into a single train inflates the CV
    # whenever the two feet are not exactly antiphase, because the merged
    # intervals then alternate short-long even at a perfectly steady tempo -
    # left/right asymmetry would leak in and read as poor rhythm.
    cvs = [c for c in (cv_l, cv_r) if c > 0]
    stride_cv = float(np.mean(cvs)) if cvs else step_cv

    return {
        "rhythm_stride_freq": freq,
        "rhythm_freq_error": abs(freq - IDEAL_CADENCE_HZ),
        "rhythm_spectral_purity": purity,
        "rhythm_step_interval": step_iv,
        "rhythm_stride_cv": stride_cv,
        "rhythm_step_cv": step_cv,
        "rhythm_stride_cv_left": cv_l,
        "rhythm_stride_cv_right": cv_r,
        "rhythm_cadence_spm": 60.0 / step_iv if step_iv > 0 else 0.0,
        "rhythm_n_steps": float(all_ev.size),
    }


def stride_features(seq: PoseSequence) -> dict:
    """Stride-length features.

    Stride is a sagittal quantity and a head-on camera foreshortens it almost
    completely, so measuring the ankles' apparent horizontal separation is
    hopeless. Two things *are* visible frontally and both are used here:

    * **Leg obliquity.** A long stride puts the leg at a steeper angle at heel
      strike, which shortens the projected hip-to-ankle distance. The depth of
      that shortening scales directly with stride length.
    * **Knee lift.** Reaching further raises the swing knee relative to the hip.

    The speed/cadence estimate is kept as a third cue, but it is the weakest:
    recovering metric speed from a monocular view needs the focal length, so
    only its *relative* form survives.
    """
    la = seq.point("left_ankle")
    ra = seq.point("right_ankle")
    lh = seq.point("left_hip")
    rh = seq.point("right_hip")
    sep = np.linalg.norm(la - ra, axis=1)
    freq, _ = dominant_frequency(la[:, 0] - ra[:, 0], seq.fps)

    leg_l = np.linalg.norm(la - lh, axis=1)
    leg_r = np.linalg.norm(ra - rh, axis=1)
    # Fully extended (mid-stance) vs most oblique (heel strike / toe off).
    ext_l = float(np.percentile(leg_l, 95))
    ext_r = float(np.percentile(leg_r, 95))

    # Shortening must be sampled *at foot strike*, not over the whole cycle.
    # The deepest shortening in a stride comes from the swing knee bending,
    # which is governed by foot clearance and is nearly independent of stride
    # length; it swamps the stride signal in any percentile taken over all
    # frames. At heel strike the knee is almost straight, so what remains is
    # purely the leg's forward obliquity - which is exactly the stride cue.
    def _shorten_at_strike(leg_len: np.ndarray, ankle_y: np.ndarray, ext: float) -> float:
        ev = step_events(ankle_y, seq.fps)
        ev = ev[ev < len(leg_len)]
        if ev.size == 0 or ext <= 1e-9:
            return safe_div(ext - float(np.percentile(leg_len, 5)), ext)
        return float(np.mean(1.0 - leg_len[ev] / ext))

    shorten_l = _shorten_at_strike(leg_l, la[:, 1], ext_l)
    shorten_r = _shorten_at_strike(leg_r, ra[:, 1], ext_r)

    knee_lift_l = amplitude(seq.point("left_knee")[:, 1], seq.fps)
    knee_lift_r = amplitude(seq.point("right_knee")[:, 1], seq.fps)

    scale = np.asarray(seq.meta.get("torso_scale", np.ones(seq.n_frames)))
    approach = float(np.polyfit(np.arange(len(scale)) / seq.fps, scale, 1)[0])
    approach_norm = safe_div(approach, float(np.mean(scale)))
    stride_est = abs(safe_div(approach_norm, freq)) if freq > 0 else 0.0

    return {
        "stride_leg_shorten_mean": 0.5 * (shorten_l + shorten_r),
        "stride_leg_shorten_amp": 0.5 * (amplitude(leg_l, seq.fps) + amplitude(leg_r, seq.fps)),
        "stride_leg_extension_mean": 0.5 * (ext_l + ext_r),
        "stride_knee_lift": 0.5 * (knee_lift_l + knee_lift_r),
        "stride_ankle_sep_max": float(np.percentile(sep, 95)),
        "stride_ankle_sep_mean": float(np.mean(sep)),
        "stride_ankle_sep_amp": amplitude(sep, seq.fps),
        "stride_foot_clearance": 0.5 * (amplitude(la[:, 1], seq.fps) + amplitude(ra[:, 1], seq.fps)),
        "stride_approach_rate": approach_norm,
        "stride_length_est": stride_est,
        "stride_length_error": abs(stride_est - IDEAL_STRIDE_TORSO / 10.0),
        "stride_pelvis_bob": _span(_pelvis_deviation(seq)[:, 1]),
    }


def foot_features(seq: PoseSequence) -> dict:
    la = seq.point("left_ankle")
    ra = seq.point("right_ankle")
    gap = la[:, 0] - ra[:, 0]

    strikes = np.sort(np.concatenate([
        step_events(la[:, 1], seq.fps), step_events(ra[:, 1], seq.fps)
    ])).astype(int)
    strikes = strikes[strikes < len(gap)]
    # Step width at foot strike is the runway-relevant quantity: models are
    # taught to land on a single line, so a good walk has near-zero mean width.
    width_at_strike = float(np.mean(np.abs(gap[strikes]))) if strikes.size else float(np.mean(np.abs(gap)))

    sign_changes = int(np.sum(np.diff(np.sign(gap)) != 0))
    return {
        "foot_step_width_mean": float(np.mean(np.abs(gap))),
        "foot_step_width_at_strike": width_at_strike,
        "foot_step_width_signed": float(np.mean(gap)),
        "foot_crossover_rate": safe_div(float(sign_changes), float(len(gap)) / seq.fps),
        "foot_lateral_amp_left": amplitude(la[:, 0], seq.fps),
        "foot_lateral_amp_right": amplitude(ra[:, 0], seq.fps),
        "foot_placement_var": float(np.std(gap)),
    }


def symmetry_features(seq: PoseSequence) -> dict:
    out = {}
    pairs = [("ankle", 1), ("wrist", 1), ("knee", 1), ("elbow", 0)]
    idx = []
    for joint, axis in pairs:
        l = amplitude(seq.point(f"left_{joint}")[:, axis], seq.fps)
        r = amplitude(seq.point(f"right_{joint}")[:, axis], seq.fps)
        si = symmetry_index(l, r)
        out[f"sym_{joint}_index"] = si
        idx.append(si)

    out["sym_overall_index"] = float(np.mean(idx))
    # Alternating gait should put the two ankles half a cycle apart.
    off = phase_offset(seq.point("left_ankle")[:, 1], seq.point("right_ankle")[:, 1], seq.fps)
    out["sym_ankle_phase_offset"] = off
    out["sym_phase_error"] = abs(off - 0.5)

    # Step-time asymmetry, the standard clinical measure. Merge both feet's
    # strikes into one train: the intervals then alternate left-to-right and
    # right-to-left, and in a symmetric walk those two alternate to the same
    # value. Any difference between the odd and even intervals is a direct read
    # on limb timing asymmetry - and unlike an amplitude ratio it survives
    # projection, because it is made of event times rather than distances.
    ev = np.sort(np.concatenate([
        cycle_times(seq.point("left_ankle")[:, 1], seq.fps),
        cycle_times(seq.point("right_ankle")[:, 1], seq.fps),
    ]))
    if ev.size >= 4:
        iv = np.diff(ev)
        a, b = iv[0::2], iv[1::2]
        out["sym_step_time_asymmetry"] = safe_div(
            abs(float(np.mean(a)) - float(np.mean(b))), float(np.mean(iv))
        )
    else:
        out["sym_step_time_asymmetry"] = 0.0
    return out


def confidence_features(seq: PoseSequence) -> dict:
    scale = np.asarray(seq.meta.get("torso_scale", np.ones(seq.n_frames)))
    t = np.arange(len(scale)) / seq.fps
    if len(scale) >= 11:
        from scipy.signal import savgol_filter
        w = min(11, len(scale) if len(scale) % 2 == 1 else len(scale) - 1)
        scale_s = savgol_filter(scale, window_length=w, polyorder=2)
    else:
        scale_s = scale
    speed = np.gradient(scale_s, t) / np.clip(np.mean(scale_s), 1e-9, None)
    speed_cv = safe_div(float(np.std(speed)), float(np.abs(np.mean(speed))))
    com = _pelvis_deviation(seq)
    shoulder_w = np.linalg.norm(
        seq.point("left_shoulder") - seq.point("right_shoulder"), axis=1
    )
    wrist_gap = np.linalg.norm(seq.point("left_wrist") - seq.point("right_wrist"), axis=1)
    return {
        "conf_speed_mean": float(np.mean(speed)),
        "conf_speed_cv": float(np.clip(speed_cv, 0.0, 5.0)),
        "conf_com_jerk": jerk_index(com[:, 0], seq.fps),
        "conf_chest_openness": float(np.mean(shoulder_w)),
        "conf_arm_spread": float(np.mean(wrist_gap)),
        "conf_motion_energy": float(np.mean(np.abs(np.diff(seq.xy, axis=0))) * seq.fps),
    }


def quality_features(seq: PoseSequence) -> dict:
    """Recording-quality covariates.

    These are not scored, but the model needs them: a noisy, gappy clip yields
    inflated jerk and sway, and without these columns the model would read that
    noise as genuine instability.
    """
    return {
        "qc_n_frames": float(seq.n_frames),
        "qc_fps": float(seq.fps),
        "qc_duration_s": float(seq.duration_s),
        "qc_mean_visibility": float(np.mean(seq.visibility)),
        "qc_min_visibility": float(np.min(np.mean(seq.visibility, axis=0))),
        "qc_gap_fraction": float(seq.meta.get("gap_fraction", 0.0)),
        "qc_torso_scale_ratio": float(
            np.max(seq.meta.get("torso_scale", [1.0])) / max(np.min(seq.meta.get("torso_scale", [1.0])), 1e-9)
        ),
    }


EXTRACTORS = (
    posture_features,
    balance_features,
    shoulder_features,
    head_features,
    arm_features,
    hip_features,
    rhythm_features,
    stride_features,
    foot_features,
    symmetry_features,
    confidence_features,
    quality_features,
)


#: Fewest frames the extractors can work with. Several fit a trend line over
#: time, and a least-squares fit through a single point has no solution - the
#: failure surfaced as "SVD did not converge in Linear Least Squares", a LAPACK
#: message shown verbatim to anyone who uploaded a one-row keypoint file. The
#: limit is where the maths breaks; clips that are merely *short* are handled by
#: the duration warning in `scoring.assess`, which can still say something useful.
MIN_ANALYSIS_FRAMES = 2


def extract_features(seq: PoseSequence) -> dict:
    """Run every extractor and return one flat, NaN-free feature dict."""
    if seq.n_frames < MIN_ANALYSIS_FRAMES:
        raise ValueError(
            f"Only {seq.n_frames} frame(s) of pose data - at least "
            f"{MIN_ANALYSIS_FRAMES} are needed to measure anything that changes "
            f"over time. For a usable score, film 4-6 seconds of walking.")
    feats: dict[str, float] = {}
    for fn in EXTRACTORS:
        feats.update(fn(seq))
    return {k: (0.0 if not np.isfinite(v) else float(v)) for k, v in feats.items()}


def feature_frame(seqs: dict[str, PoseSequence]) -> "np.ndarray":
    """Convenience wrapper: build a DataFrame indexed by ``video_id``."""
    import pandas as pd

    rows = []
    for vid, seq in seqs.items():
        row = {"video_id": vid}
        row.update(extract_features(seq))
        rows.append(row)
    return pd.DataFrame(rows).set_index("video_id").sort_index()
