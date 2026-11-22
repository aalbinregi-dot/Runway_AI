"""Biomechanically grounded synthetic runway-walk generator.

Why this exists
---------------
The supplied workbook stores the right *schema* but its motion is a placeholder:
every landmark follows one shared sinusoid, so there is no differential arm or
leg motion to learn from, and there are no labels. Training on it is impossible.

This module instead builds a **generative** dataset:

1. sample a vector of latent *skill* parameters ``q`` (one per judged criterion,
   0 = poor, 1 = flawless), correlated through a shared ability factor;
2. render ``q`` into a 3D walking skeleton using an explicit gait model
   (prescribed foot trajectories + inverse kinematics), so a low ``posture``
   score really does produce a forward-leaning spine;
3. project that skeleton through a pinhole camera at a random azimuth and
   distance, then corrupt it exactly the way a pose estimator would - jitter,
   dropout, occlusion, quantised frame rate;
4. emit keypoints in the workbook's schema, and labels ``10 * q`` **derived
   from the latents, not from the rendered pixels**.

Because the labels come from the hidden latents and the features are computed
from the noisy projection, recovering the score is a real estimation problem
rather than an algebraic identity. That is what makes the trained model
meaningful rather than a restatement of the rubric.

Anthropometric segment ratios follow Drillis & Contini (1966), expressed here in
units of torso length (pelvis centre to shoulder centre) so they match the
normalisation used in :mod:`runway_ai.data.loaders`.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from ..config import CORE_13, CRITERIA, RANDOM_SEED
from .schema import PoseSequence

# --------------------------------------------------------------------------- #
# Body model
# --------------------------------------------------------------------------- #
BODY_HEIGHT_M = 1.72          # nominal; jittered per subject
TORSO_H = 0.288               # pelvis centre -> shoulder centre, fraction of height
SEG = {                       # fractions of body height
    # Transverse widths are *joint-centre* separations, not the body-surface
    # breadths Drillis & Contini tabulate. MediaPipe's left/right shoulder and
    # hip landmarks estimate glenohumeral and femoral head centres, which sit
    # well inside the skin: measured against real footage, biacromial breadth
    # 0.259 over-predicts by ~22% and bi-iliac breadth 0.191 by ~58%. Using the
    # surface values makes every generated skeleton wider than anything a pose
    # estimator reports, and every width-derived feature lands out of range.
    "shoulder_width": 0.215,
    "hip_width": 0.125,
    # Long-bone lengths are unaffected: joint centre to joint centre is what
    # Drillis & Contini already give.
    "upper_arm": 0.186,
    "forearm": 0.146,
    "thigh": 0.245,
    "shank": 0.246,
    "neck_to_nose": 0.130,
}

GAIT_STANCE_FRACTION = 0.60   # proportion of the gait cycle spent in stance

#: Ankle travel during stance, relative to the pelvis, as a fraction of stride.
#: Not the same as the stance fraction. The *foot* is planted, but the ankle
#: joint is not: the foot rolls heel -> flat -> toe, so the ankle advances over
#: the contact patch during push-off. Measured gait puts the ankle's
#: pelvis-relative excursion near 0.40 stride rather than the 0.60 a
#: world-fixed ankle would imply. Using 0.60 demands a hip-to-ankle reach
#: longer than the leg at heel strike and silently saturates the IK.
ANKLE_STANCE_EXCURSION = 0.40

# --------------------------------------------------------------------------- #
# Amplitude calibration
# --------------------------------------------------------------------------- #
#: Scale factors on the movement amplitudes, fitted by
#: ``scripts/calibrate_generator.py``.
#:
#: The first real video analysed by this system fell outside the synthetic
#: training distribution on seven features, all in the same direction: the
#: generator moved every joint substantially more than a real walker does - hip
#: obliquity by about six times, lateral sway by three. Models trained on that
#: data extrapolate on real footage, which is the central risk of a
#: synthetic-data project and the hardest failure to spot, because nothing in
#: cross-validation reveals it.
#:
#: The targets these were fitted against are literature peak-to-peak ranges for
#: normal adult gait (Perry & Burnfield; Winter), not one subject's numbers,
#: so the correction is not tuned to a single person.
AMP = {
    "pelvic_drop": 0.40,   # pelvic obliquity ~10 deg peak-to-peak, not 45
    "sway": 0.53,          # mediolateral COM excursion ~3-5 cm
    "trunk_rot": 0.46,     # trunk axial rotation ~7-10 deg total
    "clearance": 0.52,     # swing-phase foot and knee rise
    "abduct": 0.88,        # arms carried nearer the body
}


@dataclass
class WalkParams:
    """Latent generative parameters for one clip.

    Every ``*_q`` field is a quality in [0, 1]; 1 means textbook execution. The
    label for the matching criterion is ``10 * q``.
    """

    posture_q: float
    balance_q: float
    confidence_q: float
    shoulder_alignment_q: float
    head_position_q: float
    arm_swing_q: float
    hip_movement_q: float
    rhythm_q: float
    stride_length_q: float
    foot_placement_q: float
    symmetry_q: float
    stage_presence_q: float

    # nuisance / recording conditions (not scored, but they shape the pixels)
    height_m: float = BODY_HEIGHT_M
    camera_azimuth_deg: float = 0.0
    camera_distance_m: float = 6.0
    camera_height_m: float = 1.25
    fps: float = 30.0
    duration_s: float = 6.0
    detector_noise: float = 0.004
    dropout_rate: float = 0.02

    def qualities(self) -> np.ndarray:
        return np.array([getattr(self, f"{c}_q") for c in CRITERIA], dtype=float)


# --------------------------------------------------------------------------- #
# Latent sampling
# --------------------------------------------------------------------------- #
def sample_params(
    rng: np.random.Generator,
    skill: float | None = None,
    correlated: bool = True,
) -> WalkParams:
    """Draw one plausible walker.

    A single latent ability factor drives all twelve qualities (r ~ 0.6), with
    independent per-criterion deviation on top. Real walkers are like this: good
    posture and good balance co-occur, but somebody can still have excellent
    posture and a badly timed arm swing.

    Set ``correlated=False`` to draw the twelve qualities independently. That
    destroys the realistic covariance but is essential for diagnostics: it is
    the only way to check that, say, the rhythm features respond to *rhythm*
    rather than riding on the shared ability factor.
    """
    if skill is None:
        skill = float(np.clip(rng.beta(2.6, 2.2), 0.02, 0.99))

    q = {}
    if correlated:
        shared = 0.62
        for crit in CRITERIA:
            indep = rng.normal(0.0, 0.20)
            val = shared * skill + (1 - shared) * (skill + indep * 2.2)
            q[f"{crit}_q"] = float(np.clip(val, 0.03, 0.99))

        # Stage presence is emergent: it tracks head, posture and confidence.
        q["stage_presence_q"] = float(np.clip(
            0.45 * q["stage_presence_q"]
            + 0.22 * q["confidence_q"]
            + 0.18 * q["posture_q"]
            + 0.15 * q["head_position_q"]
            + rng.normal(0, 0.05),
            0.03, 0.99,
        ))
    else:
        for crit in CRITERIA:
            q[f"{crit}_q"] = float(rng.uniform(0.03, 0.99))

    return WalkParams(
        **q,
        height_m=float(rng.normal(BODY_HEIGHT_M, 0.07)),
        camera_azimuth_deg=float(rng.normal(0.0, 12.0)),
        camera_distance_m=float(rng.uniform(3.5, 7.5)),
        camera_height_m=float(rng.uniform(1.0, 1.6)),
        fps=float(rng.choice([24.0, 25.0, 30.0, 30.0, 60.0])),
        duration_s=float(rng.uniform(4.0, 8.0)),
        # Measured against real MediaPipe output: the high-frequency residual on
        # a well-lit 1080p clip is ~0.0004 in normalised units. The original
        # 0.002-0.009 guess was thirteen times that, which swamped the finer
        # criteria and made the synthetic problem far harder than the real one.
        # The range still spans up to ~6x reality to cover dim, low-resolution
        # or motion-blurred footage.
        detector_noise=float(rng.uniform(0.0003, 0.0025)),
        dropout_rate=float(rng.uniform(0.0, 0.04)),
    )


# --------------------------------------------------------------------------- #
# Kinematics
# --------------------------------------------------------------------------- #
def _ik_two_link(root: np.ndarray, end: np.ndarray, l1: float, l2: float,
                 bend_dir: np.ndarray) -> np.ndarray:
    """Planar two-link IK: joint position between ``root`` and ``end``.

    Used for knee (hip->ankle) and elbow (shoulder->wrist). ``bend_dir`` selects
    which side the joint buckles toward, so knees bend forward and elbows back.
    Arrays are (T, 3); the chain is clamped if the target is out of reach.
    """
    d_vec = end - root
    d = np.linalg.norm(d_vec, axis=-1, keepdims=True)
    d = np.clip(d, 1e-6, l1 + l2 - 1e-6)
    unit = d_vec / d

    # Distance along the root->end line to the joint's projection.
    a = (d**2 + l1**2 - l2**2) / (2 * d)
    h = np.sqrt(np.clip(l1**2 - a**2, 0.0, None))

    # Component of bend_dir orthogonal to the limb axis.
    bd = np.broadcast_to(bend_dir, unit.shape).astype(float).copy()
    bd -= unit * np.sum(bd * unit, axis=-1, keepdims=True)
    n = np.linalg.norm(bd, axis=-1, keepdims=True)
    bd = np.where(n > 1e-8, bd / np.clip(n, 1e-8, None), 0.0)

    return root + unit * a + bd * h


def _foot_trajectory(phase: np.ndarray, stride: float, clearance: float) -> tuple[np.ndarray, np.ndarray]:
    """Ankle forward (z) and height (y) for one leg over a gait cycle.

    Prescribing the foot and solving IK for the knee - rather than prescribing
    joint angles - keeps stride length and ground contact under direct control,
    which is exactly what the stride and foot-placement features measure.

    The foot is world-fixed through stance, so *relative to the pelvis* it
    travels backward by however far the pelvis advances in that time: one
    stride times the stance fraction, not a whole stride. Using the whole
    stride here puts the ankle further from the hip than the leg can reach and
    silently drives the IK into its clamp.
    """
    p = np.mod(phase / (2 * np.pi), 1.0)
    stance = p < GAIT_STANCE_FRACTION
    excursion = stride * ANKLE_STANCE_EXCURSION

    z = np.empty_like(p)
    y = np.empty_like(p)

    # Stance: foot planted, translating backward relative to the pelvis.
    u_st = p / GAIT_STANCE_FRACTION
    z[stance] = (excursion / 2 - excursion * u_st)[stance]
    y[stance] = 0.0

    # Swing: foot lifts in an arc and travels forward.
    u_sw = (p - GAIT_STANCE_FRACTION) / (1 - GAIT_STANCE_FRACTION)
    sw = ~stance
    z[sw] = (-excursion / 2 + excursion * u_sw)[sw]
    y[sw] = (clearance * np.sin(np.pi * u_sw))[sw]
    return z, y


def render_skeleton(p: WalkParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Render latent qualities into a 3D landmark track.

    Returns
    -------
    pts : (T, 13, 3) world coordinates in metres. X = subject's left,
        Y = up, Z = walking direction (away from camera).
    t : (T,) timestamps in seconds.
    """
    H = p.height_m
    n = max(int(round(p.duration_s * p.fps)), 40)
    t = np.arange(n) / p.fps

    thigh, shank = SEG["thigh"] * H, SEG["shank"] * H
    upper_arm, forearm = SEG["upper_arm"] * H, SEG["forearm"] * H
    sh_w, hip_w = SEG["shoulder_width"] * H, SEG["hip_width"] * H
    torso = TORSO_H * H
    leg = thigh + shank

    # ---- temporal structure ------------------------------------------- #
    # Poor rhythm is *step-to-step* timing variability, not a slow tempo drift.
    # A random walk on phase velocity would produce the latter and leave the
    # step-interval CV - the quantity coaches and gait labs actually use -
    # almost unchanged. So the cadence is perturbed independently on each step,
    # with occasional hesitations on top.
    base_freq = 0.90 + 0.10 * p.confidence_q          # strides / s
    poor = (1 - p.rhythm_q) ** 1.2

    def _draw_tempo() -> float:
        s = 1.0 + poor * rng.normal(0, 0.32)
        if rng.random() < poor * 0.10:                # hesitation / stumble
            s *= rng.uniform(0.35, 0.60)
        return float(np.clip(s, 0.35, 2.2))

    theta = np.zeros(n)
    phase = 0.0
    half_cycle = 0          # each half cycle of theta is one step
    tempo = _draw_tempo()
    for i in range(1, n):
        phase += 2 * np.pi * base_freq * tempo / p.fps
        if int(phase / np.pi) != half_cycle:          # a step just completed
            half_cycle = int(phase / np.pi)
            tempo = _draw_tempo()
        theta[i] = phase
    freq = np.gradient(theta) * p.fps / (2 * np.pi)

    # ---- gross parameters driven by the latents ----------------------- #
    # Stride: ideal is ~0.78 of body height; error grows as quality drops, and
    # the sign is randomised so we generate both mincing and over-striding.
    stride_ideal = 0.78 * H
    stride_err = (1 - p.stride_length_q) * 0.45 * rng.choice([-1.0, 1.0])
    stride = stride_ideal * (1 + stride_err)

    # Runway technique: feet land on (or across) a single line. Poor placement
    # widens the base or produces an exaggerated crossover.
    step_width = 0.012 * H + (1 - p.foot_placement_q) * 0.16 * H * rng.choice([-1.0, 1.0])
    clearance = AMP["clearance"] * (0.045 + 0.02 * p.confidence_q) * H

    # Left/right mismatch from the symmetry latent.
    asym = (1 - p.symmetry_q) ** 1.3
    stride_L = stride * (1 + 0.22 * asym)
    stride_R = stride * (1 - 0.22 * asym)
    phase_off = np.pi * (1 + 0.20 * asym * rng.choice([-1.0, 1.0]))

    # ---- pelvis ------------------------------------------------------- #
    # Travel is driven by the instantaneous cadence, so an irregular rhythm
    # produces genuinely irregular progress rather than a smooth glide.
    speed = stride * freq
    z_travel = np.concatenate([[0.0], np.cumsum(speed[1:] / p.fps)])

    # Lateral sway: once per stride. Poor balance inflates it and adds tremor.
    sway_amp = AMP["sway"] * (0.020 + 0.075 * (1 - p.balance_q) ** 1.4) * H
    tremor = AMP["sway"] * (1 - p.balance_q) ** 2 * 0.030 * H * rng.normal(0, 1, n)
    tremor = np.convolve(tremor, np.ones(5) / 5, mode="same")   # ~3 Hz wobble
    sway = sway_amp * np.sin(theta) + tremor

    # Foot placement, needed before the pelvis height can be resolved.
    zL, yL = _foot_trajectory(theta, stride_L, clearance)
    zR, yR = _foot_trajectory(theta + phase_off, stride_R, clearance)
    xL = np.full(n, step_width / 2) + sway * 0.25
    xR = np.full(n, -step_width / 2) + sway * 0.25

    # Pelvic kinematics are resolved first because the hip *joints*, not the
    # pelvis centre, are what the legs hang from.
    pelvic_drop = (AMP["pelvic_drop"]
                   * np.deg2rad(1.5 + 14.0 * (1 - p.hip_movement_q)) * np.sin(theta))
    hip_rot = (-0.6 * AMP["trunk_rot"]
               * np.deg2rad(3.0 + 5.0 * p.confidence_q
                            + 4.0 * p.stage_presence_q) * np.sin(theta))

    def _hip_offset(sign: float) -> np.ndarray:
        half = sign * hip_w / 2
        return np.stack([
            half * np.cos(hip_rot) * np.cos(pelvic_drop),
            half * np.sin(pelvic_drop),
            half * np.sin(hip_rot) * np.cos(pelvic_drop),
        ], axis=1)

    hip_off_L, hip_off_R = _hip_offset(+1.0), _hip_offset(-1.0)

    # Pelvis height is *derived*, not prescribed: it sits as high as the more
    # constrained leg allows. This is the classical compass-gait result - the
    # hip dips at double support because the stance leg is oblique - and it is
    # what makes stride length observable from a head-on camera at all, since
    # a longer stride forces a visibly deeper dip. Prescribing the height
    # instead would demand a hip-to-ankle reach longer than the leg and quietly
    # saturate the IK.
    def _max_pelvis_height(foot_x: np.ndarray, foot_z: np.ndarray,
                           foot_y: np.ndarray, hip_off: np.ndarray) -> np.ndarray:
        # Lateral and depth separation between hip joint and ankle. The pelvis
        # sits at (sway, ?, z_travel) and the ankle at (foot_x, foot_y,
        # z_travel + foot_z), so the travel term cancels in depth.
        dx = sway + hip_off[:, 0] - foot_x
        dz = hip_off[:, 2] - foot_z
        reach = np.sqrt(np.clip((0.995 * leg) ** 2 - dx**2 - dz**2, 1e-4, None))
        return foot_y + reach - hip_off[:, 1]

    compass = np.minimum(
        _max_pelvis_height(xL, zL, yL, hip_off_L),
        _max_pelvis_height(xR, zR, yR, hip_off_R),
    )
    # With a realistic ankle excursion the compass geometry already lands on the
    # physiological 4-5 cm dip, so no artificial compression is applied: the hip
    # simply sits where the legs can actually hold it. Raising it above this
    # would put the ankle out of the leg's reach and clamp the IK.
    hip_h = compass
    # A small residual bob keeps the trace from looking mechanically perfect.
    hip_h = hip_h - 0.004 * H * np.cos(2 * theta)

    pelvis = np.stack([sway, hip_h, z_travel], axis=1)

    # ---- trunk -------------------------------------------------------- #
    # Poor posture = forward lean plus thoracic flexion, so the shoulder centre
    # sits ahead of and below the pelvis.
    lean = np.full(n, np.deg2rad(2.0 + 26.0 * (1 - p.posture_q) ** 1.25))
    # Shoulder roll: a constant bias (habitually dropped shoulder) plus swing.
    roll_bias = np.deg2rad(16.0 * (1 - p.shoulder_alignment_q) * rng.choice([-1.0, 1.0]))
    roll = roll_bias + np.deg2rad(2.0 + 9.0 * (1 - p.shoulder_alignment_q)) * np.sin(theta)
    # Trunk rotation about the vertical, counter to the pelvis. Presence shows
    # up here as a deliberate, committed counter-rotation rather than a timid
    # one, so both confidence and presence drive the amplitude. The pelvis
    # counter-rotates at -0.6 of this, already resolved above as ``hip_rot``.
    trunk_rot = -hip_rot / 0.6

    # Stage presence is a *held* quality: chest open, carriage steady, no
    # extraneous fidget. Without rendering it the label would exist with no
    # corresponding signal in the pixels, and no model could ever recover it.
    presence = p.stage_presence_q
    chest_open = 1.0 + 0.14 * (presence - 0.5)        # scapular retraction
    sh_w = sh_w * chest_open
    # Low presence adds slow, aimless carriage noise the gait model would not
    # otherwise produce: micro-adjustments of the trunk and head.
    fidget = (1 - presence) ** 1.5
    carriage_noise = fidget * 0.020 * rng.normal(0, 1, n)
    carriage_noise = np.convolve(carriage_noise, np.ones(9) / 9, mode="same")

    shoulder_c = pelvis + np.stack([
        torso * np.sin(roll) + carriage_noise,
        torso * np.cos(lean) * np.cos(roll),
        -torso * np.sin(lean),
    ], axis=1)

    def _lr_offset(width: np.ndarray | float, rot: np.ndarray, tilt: np.ndarray) -> np.ndarray:
        """Left-minus-centre offset for a transverse segment (shoulders/hips)."""
        half = width / 2
        return np.stack([
            half * np.cos(rot) * np.cos(tilt),
            half * np.sin(tilt),
            half * np.sin(rot) * np.cos(tilt),
        ], axis=1)

    sh_off = _lr_offset(sh_w, trunk_rot, roll)
    left_shoulder = shoulder_c + sh_off
    right_shoulder = shoulder_c - sh_off

    left_hip = pelvis + hip_off_L
    right_hip = pelvis + hip_off_R

    # ---- head --------------------------------------------------------- #
    # Chin drops and gaze falls as head_position_q falls; a nervous walker also
    # lets the head drift laterally off the spine axis.
    # A commanding walk also lifts the chin fractionally and holds it there.
    pitch = np.full(n, np.deg2rad(
        3.0 + 30.0 * (1 - p.head_position_q) ** 1.2 - 3.0 * (presence - 0.5)
    ))
    neck = SEG["neck_to_nose"] * H
    head_wobble = ((1 - p.head_position_q) * 0.035 * H * np.sin(2 * theta + 0.7)
                   + 1.4 * carriage_noise)
    nose = shoulder_c + np.stack([
        head_wobble + neck * 0.15 * np.sin(roll),
        neck * np.cos(pitch),
        -neck * np.sin(pitch) - torso * np.sin(lean) * 0.25,
    ], axis=1)

    # ---- legs --------------------------------------------------------- #
    # Crossover: negative step width puts each foot past the midline. The
    # lateral tracks already include their share of the sway.
    ankle_L = np.stack([xL, yL, z_travel + zL], axis=1)
    ankle_R = np.stack([xR, yR, z_travel + zR], axis=1)

    knee_bend = np.array([0.0, 0.0, 1.0])      # knees buckle forward (+Z)
    knee_L = _ik_two_link(left_hip, ankle_L, thigh, shank, knee_bend)
    knee_R = _ik_two_link(right_hip, ankle_R, thigh, shank, knee_bend)

    # ---- arms --------------------------------------------------------- #
    # Ideal swing ~24 deg, contralateral to the legs. Quality controls both the
    # amplitude error and how much the two sides disagree.
    swing_ideal = np.deg2rad(24.0)
    amp_err = (1 - p.arm_swing_q) * 0.85 * rng.choice([-1.0, 1.0])
    amp_L = swing_ideal * (1 + amp_err) * (1 + 0.35 * asym)
    amp_R = swing_ideal * (1 + amp_err) * (1 - 0.35 * asym)
    # A tense walker holds the arms away from the body.
    abduct = AMP["abduct"] * np.deg2rad(6.0 + 16.0 * (1 - p.confidence_q))

    def _arm(shoulder: np.ndarray, amp: float, phase: np.ndarray, side: float) -> tuple[np.ndarray, np.ndarray]:
        """Rigid pendulum from the shoulder: swing about the lateral axis, then
        abduct about the forward axis.

        Composing the two rotations this way keeps ``|wrist - shoulder|``
        exactly constant in 3D. Building the vector component-wise instead
        would let the arm stretch as it swings, and the projected reach - which
        the arm-swing feature reads as foreshortening - would then be measuring
        that artefact rather than the swing.
        """
        ang = amp * np.sin(phase)
        arm_len = (upper_arm + forearm) * 0.93
        ca, sa = np.cos(ang), np.sin(ang)
        cb, sb = np.cos(abduct), np.sin(abduct)
        direction = np.stack([side * ca * sb, -ca * cb, sa], axis=1)
        wrist = shoulder + arm_len * direction
        elbow = _ik_two_link(shoulder, wrist, upper_arm, forearm,
                             np.array([0.0, 0.0, -1.0]))   # elbows buckle back
        return elbow, wrist

    elbow_L, wrist_L = _arm(left_shoulder, amp_L, theta + phase_off, +1.0)
    elbow_R, wrist_R = _arm(right_shoulder, amp_R, theta, -1.0)

    pts = np.stack([
        nose,
        left_shoulder, right_shoulder,
        elbow_L, elbow_R,
        wrist_L, wrist_R,
        left_hip, right_hip,
        knee_L, knee_R,
        ankle_L, ankle_R,
    ], axis=1)
    return pts, t


# --------------------------------------------------------------------------- #
# Camera + sensor
# --------------------------------------------------------------------------- #
def project(pts: np.ndarray, p: WalkParams, rng: np.random.Generator,
            aspect: float = 16 / 9, margin: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole-project a (T, L, 3) world track to normalised image coords.

    The camera orbits the mid-point of the walk by ``camera_azimuth_deg`` and
    sits ``camera_distance_m`` away, so the subject stays framed no matter how
    far they travel. The focal length is then *auto-framed*: it starts from a
    "fill 78% of the height" target and is reduced if that would push any
    landmark outside the frame - the same thing an operator does when setting
    up the shot. Per-clip framing changes apparent scale, which the torso-length
    normalisation in :mod:`runway_ai.data.loaders` removes again.

    Returns ``(xy, visibility)`` with ``xy`` in [0, 1] and image ``y`` growing
    downward, matching MediaPipe's convention and the sample workbook.
    """
    # Aim at the middle of the walk, at hip height.
    pelvis = 0.5 * (pts[:, 7, :] + pts[:, 8, :])
    look_at = np.array([pelvis[:, 0].mean(), pelvis[:, 1].mean(), pelvis[:, 2].mean()])

    # The +180 puts the camera in *front* of the walker, looking back down the
    # runway, so the subject approaches the lens and grows in frame. Without it
    # the camera sits behind them and every perspective cue is inverted: the
    # torso shrinks instead of growing, and the approach-rate feature comes out
    # with the wrong sign against any real runway footage.
    az = np.deg2rad(p.camera_azimuth_deg + 180.0)
    c, s = np.cos(az), np.sin(az)
    rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])

    rel = (pts - look_at) @ rot.T
    # ``camera_distance_m`` is the *closest approach*: back the camera off by
    # half the travel so a long walk does not end up on the lens. Getting much
    # nearer than ~3 m makes limbs at different depths scale so differently
    # that torso normalisation can no longer cancel the perspective.
    travel = float(pelvis[:, 2].max() - pelvis[:, 2].min())
    eff_distance = p.camera_distance_m + 0.5 * travel * np.cos(az)

    xc = rel[:, :, 0]
    yc = rel[:, :, 1] - (p.camera_height_m - look_at[1])
    zc = np.clip(rel[:, :, 2] + eff_distance, 1.2, None)

    rx, ry = xc / zc, yc / zc

    # Auto-frame: desired zoom, capped so nothing leaves the visible area.
    half = 0.5 - margin
    fy = 0.78 * eff_distance / p.height_m
    fy = min(fy, half / max(np.abs(ry).max(), 1e-6),
             half / max(aspect * np.abs(rx).max(), 1e-6))
    fx = fy * aspect

    xy = np.stack([0.5 + fx * rx, 0.5 - fy * ry], axis=2)

    # --- sensor / detector degradation --------------------------------- #
    # Localisation error grows with distance: a far-away subject is fewer
    # pixels tall, so the same pixel error is a larger normalised error.
    noise = p.detector_noise * (zc / eff_distance)[:, :, None]
    xy = xy + rng.normal(0, 1, xy.shape) * noise

    vis = np.clip(rng.normal(0.96, 0.02, xy.shape[:2]), 0.0, 1.0)
    # Wrists and ankles self-occlude behind the torso near mid-swing.
    for idx in (5, 6, 11, 12):
        vis[:, idx] -= rng.uniform(0.0, 0.05)
    # Random detector dropouts.
    drop = rng.random(vis.shape) < p.dropout_rate
    vis[drop] = rng.uniform(0.05, 0.35, drop.sum())

    xy = np.clip(xy, -0.15, 1.15)
    return xy, np.clip(vis, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def make_clip(video_id: str, p: WalkParams, rng: np.random.Generator) -> tuple[PoseSequence, dict]:
    """Generate one labelled clip."""
    pts, _ = render_skeleton(p, rng)
    xy, vis = project(pts, p, rng)

    # Depth channel, mirroring MediaPipe: roughly hip-relative, in torso units.
    pelvis_z = 0.5 * (pts[:, 7, 2] + pts[:, 8, 2])
    z = (pts[:, :, 2] - pelvis_z[:, None]) / (TORSO_H * p.height_m)

    seq = PoseSequence(
        video_id=video_id, landmarks=list(CORE_13),
        xy=xy, z=z, visibility=vis, fps=p.fps,
        meta={"synthetic": True},
    )

    labels = {f"score_{c}": round(10.0 * getattr(p, f"{c}_q"), 4) for c in CRITERIA}
    meta = {"video_id": video_id, "n_frames": seq.n_frames, **labels,
            **{f"param_{k}": v for k, v in asdict(p).items()}}
    return seq, meta


def generate_dataset(
    n_clips: int = 900,
    seed: int = RANDOM_SEED,
    prefix: str = "syn",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a labelled synthetic corpus.

    Returns ``(keypoints_long, labels)`` where ``keypoints_long`` uses the
    sample workbook's exact seven columns.
    """
    rng = np.random.default_rng(seed)
    frames, metas = [], []
    for i in range(n_clips):
        vid = f"{prefix}_{i:04d}"
        seq, meta = make_clip(vid, sample_params(rng), rng)
        frames.append(seq.to_long())
        metas.append(meta)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(metas)
