"""Render an annotated analysis video: skeleton, live measurements, scores.

A single number is hard to trust and impossible to argue with. Drawing the
tracked skeleton and the actual quantities being measured - trunk lean,
shoulder tilt, step width, foot strikes - back onto the original footage turns
the assessment into something a coach can check frame by frame, and makes
tracking failures obvious instead of silent.

Everything drawn here comes from the same ``PoseSequence`` and feature code the
score comes from, so the overlay cannot drift out of agreement with the report.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# OpenCV is imported defensively, not because it is optional to *this* module -
# every function here needs it - but because requirements.txt deliberately does
# not ship it. The analysis environment consumes keypoints rather than pixels, so
# it installs no video stack; only `render` needs one, and it needs it in the
# analysis environment rather than the MediaPipe one. Without this guard a fresh
# install fails at import time with a bare ModuleNotFoundError from a file the
# user never asked about.
try:
    import cv2
except ImportError:                              # pragma: no cover - env-specific
    cv2 = None


def _require_cv2() -> None:
    if cv2 is not None:
        return
    raise RuntimeError(
        "Rendering an annotated video needs OpenCV in this environment.\n"
        "  The analysis requirements leave it out on purpose - the pipeline reads\n"
        "  keypoints, not pixels - so add it where the renderer runs:\n"
        "      python -m pip install opencv-python-headless\n"
        "  Scores, reports, the API and the web app are unaffected without it."
    )

from ..config import CRITERION_LABELS
from ..data.schema import PoseSequence
from ..features.signals import angle_from_horizontal, angle_from_vertical, cycle_times

# BGR. Left limbs warm, right limbs cool, so a viewer can tell the sides apart
# and spot left/right asymmetry directly in the overlay.
C_LEFT = (92, 132, 240)
C_RIGHT = (222, 168, 78)
C_SPINE = (236, 236, 236)
C_GOOD = (110, 190, 120)
C_MID = (86, 176, 226)
C_POOR = (86, 92, 226)
C_PANEL = (28, 26, 24)
C_TEXT = (240, 240, 240)
C_DIM = (150, 150, 150)

SKELETON = [
    ("left_shoulder", "right_shoulder", C_SPINE),
    ("left_hip", "right_hip", C_SPINE),
    ("left_shoulder", "left_hip", C_SPINE),
    ("right_shoulder", "right_hip", C_SPINE),
    ("left_shoulder", "left_elbow", C_LEFT),
    ("left_elbow", "left_wrist", C_LEFT),
    ("left_hip", "left_knee", C_LEFT),
    ("left_knee", "left_ankle", C_LEFT),
    ("right_shoulder", "right_elbow", C_RIGHT),
    ("right_elbow", "right_wrist", C_RIGHT),
    ("right_hip", "right_knee", C_RIGHT),
    ("right_knee", "right_ankle", C_RIGHT),
]


def _grade_colour(score: float) -> tuple[int, int, int]:
    return C_GOOD if score >= 7.5 else (C_MID if score >= 5.0 else C_POOR)


def _panel(img: np.ndarray, x: int, y: int, w: int, h: int, alpha: float = 0.72) -> None:
    """Translucent backing so text stays readable over any footage."""
    x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
    x, y = max(x, 0), max(y, 0)
    if x2 <= x or y2 <= y:
        return
    roi = img[y:y2, x:x2]
    img[y:y2, x:x2] = cv2.addWeighted(
        roi, 1 - alpha, np.full_like(roi, C_PANEL, dtype=np.uint8), alpha, 0
    )


def _text(img, s, org, scale, colour=C_TEXT, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour,
                thick, cv2.LINE_AA)


def frame_metrics(seq: PoseSequence) -> dict[str, np.ndarray]:
    """Per-frame values of the quantities worth showing live."""
    sh = seq.midpoint("left_shoulder", "right_shoulder")
    hip = seq.midpoint("left_hip", "right_hip")
    la, ra = seq.point("left_ankle"), seq.point("right_ankle")

    strikes = np.zeros(seq.n_frames, dtype=bool)
    for ankle in (la[:, 1], ra[:, 1]):
        for t in cycle_times(ankle, seq.fps):
            i = int(round(t * seq.fps))
            if 0 <= i < seq.n_frames:
                strikes[i] = True

    return {
        "trunk_lean": np.abs(angle_from_vertical(sh - hip)),
        "shoulder_tilt": angle_from_horizontal(
            seq.point("left_shoulder") - seq.point("right_shoulder")),
        "step_width": np.abs(la[:, 0] - ra[:, 0]),
        "strike": strikes,
    }


def render(
    video_path: str | Path,
    seq_image: PoseSequence,
    result,
    out_path: str | Path,
    seq_norm: PoseSequence | None = None,
    max_side: int = 720,
) -> Path:
    """Write an annotated copy of ``video_path``.

    ``seq_image`` must hold **image-space** landmarks (0-1), i.e. the sequence
    as extracted, before normalisation moves the pelvis to the origin.
    ``seq_norm`` is the preprocessed sequence used for the metric readouts.
    """
    _require_cv2()
    video_path, out_path = Path(video_path), Path(out_path)
    metrics = frame_metrics(seq_norm if seq_norm is not None else seq_image)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {video_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or seq_image.fps

    # Downscale for a shareable file; 1080x1920 annotated is needlessly large.
    scale = min(1.0, max_side / max(src_w, src_h))
    w, h = int(src_w * scale), int(src_h * scale)

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"could not open a writer for {out_path}")

    a = result.assessment
    ranked = sorted(a.scores.items(), key=lambda kv: kv[1])
    worst3 = ranked[:3]
    best = ranked[-1]
    n = seq_image.n_frames
    s = h / 720.0                                    # scale UI to frame height

    try:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or idx >= n:
                break
            if scale < 1.0:
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)

            pts = seq_image.xy[idx]
            vis = seq_image.visibility[idx]

            def P(name):
                i = seq_image.index(name)
                return (int(pts[i, 0] * w), int(pts[i, 1] * h)), vis[i]

            # --- skeleton -------------------------------------------------- #
            for a_name, b_name, colour in SKELETON:
                (pa, va), (pb, vb) = P(a_name), P(b_name)
                if va < 0.4 or vb < 0.4:
                    continue
                cv2.line(frame, pa, pb, colour, max(int(3 * s), 2), cv2.LINE_AA)
            for name in seq_image.landmarks:
                p, v = P(name)
                if v >= 0.4:
                    side = C_LEFT if name.startswith("left") else (
                        C_RIGHT if name.startswith("right") else C_SPINE)
                    cv2.circle(frame, p, max(int(4 * s), 3), side, -1, cv2.LINE_AA)

            # --- plumb line from the shoulder centre ----------------------- #
            (lsh, _), (rsh, _) = P("left_shoulder"), P("right_shoulder")
            (lhp, _), (rhp, _) = P("left_hip"), P("right_hip")
            sh_c = ((lsh[0] + rsh[0]) // 2, (lsh[1] + rsh[1]) // 2)
            hip_c = ((lhp[0] + rhp[0]) // 2, (lhp[1] + rhp[1]) // 2)
            # Vertical reference through the pelvis, one torso long. Using a
            # fixed pixel length would shoot far above the head as the subject
            # approaches and the body grows in frame.
            torso_px = max(int(abs(sh_c[1] - hip_c[1])), 10)
            cv2.line(frame, (hip_c[0], hip_c[1] - torso_px), hip_c,
                     (120, 120, 120), 1, cv2.LINE_AA)
            cv2.line(frame, hip_c, sh_c, (255, 255, 255), max(int(2 * s), 1), cv2.LINE_AA)

            # --- foot-strike flash ----------------------------------------- #
            if metrics["strike"][idx]:
                for foot in ("left_ankle", "right_ankle"):
                    p, v = P(foot)
                    if v >= 0.4:
                        cv2.circle(frame, p, int(18 * s), (90, 230, 255),
                                   max(int(2 * s), 2), cv2.LINE_AA)

            # --- header: overall ------------------------------------------- #
            _panel(frame, 0, 0, w, int(64 * s))
            _text(frame, f"{a.overall:.0f}", (int(12 * s), int(46 * s)), 1.5 * s,
                  _grade_colour(a.overall / 10), 3)
            _text(frame, "/100", (int(62 * s), int(46 * s)), 0.6 * s, C_DIM, 1)
            _text(frame, a.grade, (int(120 * s), int(30 * s)), 0.62 * s, C_TEXT, 1)
            _text(frame, f"best: {CRITERION_LABELS[best[0]].split(' &')[0]} {best[1]:.1f}",
                  (int(120 * s), int(52 * s)), 0.45 * s, C_DIM, 1)

            # --- live measurements ----------------------------------------- #
            _panel(frame, 0, int(64 * s), int(178 * s), int(78 * s))
            rows = [
                ("trunk lean", f"{metrics['trunk_lean'][idx]:5.1f} deg"),
                ("shoulder tilt", f"{metrics['shoulder_tilt'][idx]:+5.1f} deg"),
                ("step width", f"{metrics['step_width'][idx]:5.2f} torso"),
            ]
            for r, (k, v) in enumerate(rows):
                y = int((84 + r * 22) * s)
                _text(frame, k, (int(10 * s), y), 0.4 * s, C_DIM, 1)
                _text(frame, v, (int(104 * s), y), 0.42 * s, C_TEXT, 1)

            # --- weakest criteria ------------------------------------------ #
            ph = int((26 + 22 * len(worst3)) * s)
            _panel(frame, 0, h - ph - int(18 * s), int(210 * s), ph)
            _text(frame, "NEEDS WORK", (int(10 * s), h - ph + int(2 * s)),
                  0.4 * s, C_DIM, 1)
            for r, (crit, sc) in enumerate(worst3):
                y = h - ph + int((24 + r * 22) * s)
                label = CRITERION_LABELS[crit].split(" &")[0][:16]
                _text(frame, label, (int(10 * s), y), 0.42 * s, C_TEXT, 1)
                _text(frame, f"{sc:.1f}", (int(160 * s), y), 0.44 * s,
                      _grade_colour(sc), 1)
                bar_y = y + int(4 * s)
                cv2.rectangle(frame, (int(120 * s), bar_y - int(7 * s)),
                              (int(120 * s) + int(34 * s), bar_y - int(2 * s)),
                              (70, 70, 70), -1)
                cv2.rectangle(frame, (int(120 * s), bar_y - int(7 * s)),
                              (int(120 * s) + int(34 * s * sc / 10), bar_y - int(2 * s)),
                              _grade_colour(sc), -1)

            # --- timeline -------------------------------------------------- #
            cv2.rectangle(frame, (0, h - int(6 * s)), (w, h), (48, 48, 48), -1)
            cv2.rectangle(frame, (0, h - int(6 * s)),
                          (int(w * (idx + 1) / n), h), (200, 200, 200), -1)

            writer.write(frame)
            idx += 1
    finally:
        writer.release()
        cap.release()

    return out_path
