"""Where the walker is looking, from facial geometry.

The gap this fills
------------------
One of the twelve criteria is called "Head Position **& Gaze**", but until now
nothing in the system measured gaze at all. Every head feature was derived from
the single ``nose`` pose landmark relative to the shoulders, which says where
the head *is* and nothing about where it is *pointing*. A walker looking at the
floor and one looking down the runway can put their nose in the same place.

The pose model cannot close that gap: BlazePose gives one nose point. Face
detection does, and for free - YuNet already returns five facial landmarks
alongside every box it finds (both eyes, nose tip, both mouth corners), and the
detector is already being run for enrolment.

Geometry
--------
From those five points, in the image plane:

* **roll** - the tilt of the eye line. Direct and exact.
* **yaw** - the nose's offset from the eye midpoint, in eye-widths. Zero when
  square to the camera; the nose slides toward whichever eye the head turns
  past. Proportional to sin(yaw) for small angles.
* **pitch** - the face's vertical extent (eye line to mouth line) divided by
  the eye separation. Dropping the chin foreshortens the face vertically while
  leaving eye separation untouched, so the ratio falls.

  The first attempt measured the nose's position *within* the face instead, and
  that carries no pitch signal at all: the nose-to-eye and mouth-to-eye
  distances foreshorten together and the ratio between them cancels. On a frame
  where the subject is visibly staring at the floor it read 0.586 against
  0.543-0.584 for upright frames - indistinguishable. The current ratio
  separates the same frame cleanly, 0.93 against 1.04-1.06.

These are *ratios*, not angles in degrees. Recovering true angles needs a 3D
face model and camera intrinsics, neither of which this system has, so the
values are reported as what they are: normalised, monotonic proxies calibrated
against measured neutral values rather than dressed up as degrees.

Not a learned score
-------------------
Gaze informs the *findings*, never the model's number. The synthetic generator
draws stick figures with no faces, so there is no gaze data to train on, and a
learned gaze score would be fabricated. These are measurements, shown as
measurements.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .signals import gait_highpass, safe_div

#: Neutral values, measured on the project's reference clip of a subject facing
#: the camera. A walker inside these bands reads as square-on.
NEUTRAL_PITCH = 1.045       # face height / eye separation, head level
YAW_NEUTRAL_BAND = 0.06     # |yaw| below this is squarely on-camera
PITCH_DOWN_BAND = 0.05      # pitch this far below neutral is a dropped chin

#: A face smaller than this is too coarse for the landmark geometry to be
#: trusted; the walker is simply too far away.
MIN_EYE_DISTANCE_PX = 12.0


def head_pose_series(video_path: str | Path, nose_track: np.ndarray | None = None,
                     stride: int = 4) -> dict | None:
    """Per-frame head pose across a clip.

    Returns ``None`` when the models are missing or too few faces are found -
    gaze is an enrichment, and its absence must never stop a walk being scored.
    """
    import cv2

    from .. import face as face_mod

    if not face_mod.available():
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    det = cv2.FaceDetectorYN.create(
        str(face_mod.DETECTOR_PATH), "", (w, h),
        face_mod.DETECT_SCORE, face_mod.DETECT_NMS, 5000)

    times, yaw, pitch, roll = [], [], [], []
    try:
        for f in range(0, total, stride):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ok, img = cap.read()
            if not ok:
                continue
            _, faces = det.detect(img)
            if faces is None or not len(faces):
                continue

            # Prefer the face nearest the tracked nose, so a bystander's head
            # pose is never mistaken for the walker's.
            chosen = None
            if nose_track is not None and len(nose_track):
                nx, ny = nose_track[min(f, len(nose_track) - 1)]
                nx, ny = nx * w, ny * h
                best = np.inf
                for r in faces:
                    cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2
                    dist = np.hypot(cx - nx, cy - ny) / max(r[2], 1)
                    if dist < best:
                        best, chosen = dist, r
                if best > face_mod.MAX_NOSE_DISTANCE:
                    chosen = None
            if chosen is None:
                chosen = max(faces, key=lambda r: r[2] * r[3])

            r_eye, l_eye = chosen[4:6], chosen[6:8]
            nose_pt = chosen[8:10]
            r_mouth, l_mouth = chosen[10:12], chosen[12:14]

            eye_d = float(np.linalg.norm(l_eye - r_eye))
            if eye_d < MIN_EYE_DISTANCE_PX:
                continue
            eye_mid = (r_eye + l_eye) / 2
            mouth_mid = (r_mouth + l_mouth) / 2
            face_h = float(np.linalg.norm(mouth_mid - eye_mid))
            if face_h < 1.0:
                continue

            times.append(f / fps)
            roll.append(float(np.degrees(np.arctan2(l_eye[1] - r_eye[1],
                                                    l_eye[0] - r_eye[0]))))
            yaw.append(float((nose_pt[0] - eye_mid[0]) / eye_d))
            pitch.append(face_h / eye_d)
    finally:
        cap.release()

    if len(times) < 8:
        return None
    return {"t": np.asarray(times), "yaw": np.asarray(yaw),
            "pitch": np.asarray(pitch), "roll": np.asarray(roll),
            "fps": fps / stride, "coverage": len(times)}


def features(series: dict | None) -> dict:
    """Gaze features. Empty when no usable face was found, by design."""
    if not series:
        return {}

    yaw, pitch, roll = series["yaw"], series["pitch"], series["roll"]
    fps = series["fps"]

    # Fraction of the walk spent square to the camera, which is what a runway
    # coach actually asks for: eyes to the end of the runway, not the floor.
    on_camera = float(np.mean(np.abs(yaw) <= YAW_NEUTRAL_BAND))
    looking_down = float(np.mean(pitch < NEUTRAL_PITCH - PITCH_DOWN_BAND))

    # Steadiness: gaze-band wander, not the slow drift of walking toward a lens.
    yaw_jitter = float(np.std(gait_highpass(yaw, fps))) if len(yaw) > 8 else 0.0

    return {
        "gaze_yaw_abs_mean": float(np.mean(np.abs(yaw))),
        "gaze_yaw_amp": float(np.percentile(yaw, 95) - np.percentile(yaw, 5)),
        "gaze_yaw_jitter": yaw_jitter,
        "gaze_on_camera_fraction": on_camera,
        "gaze_pitch_mean": float(np.mean(pitch)),
        "gaze_pitch_drop": float(max(0.0, NEUTRAL_PITCH - np.mean(pitch))),
        "gaze_looking_down_fraction": looking_down,
        "gaze_roll_abs_mean": float(np.mean(np.abs(roll))),
        "gaze_roll_amp": float(np.percentile(roll, 95) - np.percentile(roll, 5)),
        "gaze_frames": float(series["coverage"]),
    }


def summary(feats: dict) -> dict | None:
    """A plain-language read of the gaze measurements, for display.

    Deliberately phrased as observation rather than judgement: these are
    measured quantities, not a model output, and the copy should not imply a
    score was computed from them.
    """
    if "gaze_on_camera_fraction" not in feats:
        return None

    on = feats["gaze_on_camera_fraction"]
    down = feats["gaze_looking_down_fraction"]
    roll = feats["gaze_roll_abs_mean"]

    notes = []
    if on >= 0.75:
        notes.append(f"Eyeline held forward for {on * 100:.0f}% of the walk.")
    elif on >= 0.45:
        notes.append(f"Eyeline forward {on * 100:.0f}% of the time — it drifts "
                     "off to the side for the rest.")
    else:
        notes.append(f"Eyeline was square to the camera only {on * 100:.0f}% of "
                     "the walk; the head turns away for most of it.")

    if down >= 0.15:
        notes.append(f"Chin dropped toward the floor in {down * 100:.0f}% of "
                     "frames. Lift the gaze to the end of the runway.")
    if roll >= 5.0:
        notes.append(f"Head is tilted about {roll:.0f}° off vertical on average.")

    return {
        "on_camera_fraction": round(on, 3),
        "looking_down_fraction": round(down, 3),
        "roll_deg": round(roll, 1),
        "frames": int(feats.get("gaze_frames", 0)),
        "notes": notes,
        # Stated wherever this is shown: these are measurements, not a score.
        "basis": "measured from facial landmarks, not model-scored",
    }
