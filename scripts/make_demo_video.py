"""Render a synthetic walk as an actual MP4.

Useful for two things: seeing what the generator produces, and having a video to
push through the full video -> pose -> score path without needing to film
anybody.

A caveat worth stating: MediaPipe is trained on photographs of people. A
rendered figure is a much easier or much harder target depending on how closely
it resembles one, so detection here is a check of the *plumbing*, not evidence
about accuracy on real footage. Real video is still the only honest test of
that.

Usage:
    python scripts/make_demo_video.py --skill 0.9 --out demo_expert.mp4
    python scripts/make_demo_video.py --skill 0.2 --out demo_novice.mp4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runway_ai.config import CORE_13                                # noqa: E402
from runway_ai.data.synthetic import (                              # noqa: E402
    WalkParams, project, render_skeleton, sample_params,
)

IDX = {n: i for i, n in enumerate(CORE_13)}

# Limb chains drawn as tapered capsules, outer-to-inner so nearer parts overlay.
LIMBS = [
    ("left_hip", "left_knee", 0.062), ("left_knee", "left_ankle", 0.048),
    ("right_hip", "right_knee", 0.062), ("right_knee", "right_ankle", 0.048),
    ("left_shoulder", "left_elbow", 0.044), ("left_elbow", "left_wrist", 0.034),
    ("right_shoulder", "right_elbow", 0.044), ("right_elbow", "right_wrist", 0.034),
]

SKIN = (150, 174, 205)      # BGR, mid tone
CLOTH = (78, 62, 58)
BG_TOP = (232, 228, 222)
BG_FLOOR = (198, 192, 184)


def draw_frame(pts2d: np.ndarray, w: int, h: int) -> np.ndarray:
    """Render one frame from (13, 2) normalised landmark coordinates."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[: int(h * 0.72)] = BG_TOP
    img[int(h * 0.72):] = BG_FLOOR
    cv2.line(img, (0, int(h * 0.72)), (w, int(h * 0.72)), (176, 170, 162), 2)

    def P(name: str) -> tuple[int, int]:
        x, y = pts2d[IDX[name]]
        return int(round(x * w)), int(round(y * h))

    scale = h  # limb thicknesses are fractions of frame height

    # Torso as a filled quad, so the trunk reads as a solid body not a stick.
    torso = np.array([P("left_shoulder"), P("right_shoulder"),
                      P("right_hip"), P("left_hip")], dtype=np.int32)
    cv2.fillConvexPoly(img, torso, CLOTH)

    for a, b, t in LIMBS:
        colour = CLOTH if ("hip" in a or "shoulder" in a) else SKIN
        cv2.line(img, P(a), P(b), colour, max(int(t * scale), 2), cv2.LINE_AA)
        cv2.circle(img, P(b), max(int(t * scale * 0.5), 2), colour, -1, cv2.LINE_AA)

    # Neck and head.
    sh_mid = tuple(((np.array(P("left_shoulder")) + np.array(P("right_shoulder"))) // 2).tolist())
    nose = P("nose")
    cv2.line(img, sh_mid, nose, SKIN, max(int(0.035 * scale), 2), cv2.LINE_AA)
    head_r = max(int(0.042 * scale), 4)
    cv2.circle(img, nose, head_r, SKIN, -1, cv2.LINE_AA)
    cv2.circle(img, nose, head_r, (120, 140, 168), 2, cv2.LINE_AA)

    return cv2.GaussianBlur(img, (3, 3), 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", type=float, default=0.85, help="0-1 walking ability")
    ap.add_argument("--out", default="demo_walk.mp4")
    ap.add_argument("--width", type=int, default=720)
    ap.add_argument("--height", type=int, default=1280)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seconds", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    base = sample_params(rng, skill=args.skill)
    p = WalkParams(
        **{f: getattr(base, f) for f in base.__dataclass_fields__
           if f.endswith("_q")},
        height_m=base.height_m,
        camera_azimuth_deg=6.0,
        camera_distance_m=5.0,
        camera_height_m=1.35,
        fps=args.fps,
        duration_s=args.seconds,
        detector_noise=0.0,        # rendering is the ground truth; no fake jitter
        dropout_rate=0.0,
    )

    pts3d, _ = render_skeleton(p, rng)
    xy, _ = project(pts3d, p, rng, aspect=args.height / args.width)

    out = Path(args.out)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps, (args.width, args.height))
    if not writer.isOpened():
        print(f"Could not open a writer for {out}")
        return 1
    try:
        for t in range(xy.shape[0]):
            writer.write(draw_frame(xy[t], args.width, args.height))
    finally:
        writer.release()

    truth = {c: round(10 * getattr(p, f"{c}_q"), 1)
             for c in ("posture", "balance", "confidence", "rhythm")}
    print(f"Wrote {out}  ({xy.shape[0]} frames, {args.width}x{args.height}, {args.fps:.0f} fps)")
    print(f"Generated at skill={args.skill}. Ground-truth sample: {truth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
