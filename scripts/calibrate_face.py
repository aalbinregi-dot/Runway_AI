"""Measure the face-matching operating point on this system.

The SFace model card quotes a cosine threshold of 0.363 (and L2 1.128) from
LFW. Adopting a published number without checking it against your own footage
is how biometric systems end up quietly failing: the threshold was set on
posed, well-lit portraits, and a walking subject at 3-8 m under room lighting
is not that.

What this measures
------------------
The **genuine** distribution only: how similar two embeddings of *the same
person* are, across the pose, scale and motion-blur variation of a real clip.
That is enough to answer the question that actually drives the design - how
many frames must be aggregated before same-person matching is reliable.

What this cannot measure
------------------------
The **impostor** distribution, which needs several different people. With one
subject there is no honest false-accept rate to report, and none should be
claimed. Until multi-subject footage exists, the published threshold is used as
the accept bar and the measured genuine margin is what justifies confidence in
it - not a fabricated FAR.

Usage:
    python scripts/calibrate_face.py video.mp4 [more.mp4 ...]
"""
from __future__ import annotations

import argparse
import itertools
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runway_ai.config import MODEL_DIR                    # noqa: E402

DETECTOR = MODEL_DIR / "face" / "face_detection_yunet_2023mar.onnx"
EMBEDDER = MODEL_DIR / "face" / "face_recognition_sface_2021dec.onnx"

PUBLISHED_COSINE = 0.363
PUBLISHED_L2 = 1.128


def models(width: int, height: int):
    if not DETECTOR.exists() or not EMBEDDER.exists():
        raise SystemExit(
            f"Face models missing. Expected:\n  {DETECTOR}\n  {EMBEDDER}\n"
            "Download them from the OpenCV Zoo (see docs/FACE.md)."
        )
    det = cv2.FaceDetectorYN.create(str(DETECTOR), "", (width, height), 0.6, 0.3, 5000)
    rec = cv2.FaceRecognizerSF.create(str(EMBEDDER), "")
    return det, rec


def embed_video(path: Path, stride: int = 6) -> np.ndarray:
    """Largest-face embedding from every ``stride``-th frame."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"could not open {path}")
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det, rec = models(w, h)

    out = []
    for f in range(0, n, stride):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        _, faces = det.detect(img)
        if faces is None or not len(faces):
            continue
        face = max(faces, key=lambda x: x[2] * x[3])
        out.append(rec.feature(rec.alignCrop(img, face)))
    cap.release()
    return np.asarray(out).squeeze(1) if out else np.empty((0, 128))


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def report_genuine(E: np.ndarray) -> None:
    pairs = np.array([cosine(a, b) for a, b in itertools.combinations(E, 2)])
    below = (pairs < PUBLISHED_COSINE).mean() * 100
    print(f"\nGENUINE pairs, single frame vs single frame  (n={len(pairs)})")
    print(f"  min {pairs.min():.3f}   p1 {np.percentile(pairs, 1):.3f}   "
          f"median {np.median(pairs):.3f}   max {pairs.max():.3f}")
    print(f"  below the published {PUBLISHED_COSINE} threshold: {below:.1f}%")
    if below > 0:
        print("  -> single-frame identification is NOT reliable; aggregate frames.")


def report_aggregation(E: np.ndarray, trials: int = 400, seed: int = 0) -> None:
    """How many enrolment and probe frames are needed for a safe margin."""
    rng = np.random.default_rng(seed)
    print(f"\nAGGREGATED matching  (disjoint enrolment and probe frames, "
          f"{trials} trials each)")
    print(f"  {'enrol':>6}{'probe':>7}{'worst':>9}{'p1':>8}{'median':>9}"
          f"{'  below ' + str(PUBLISHED_COSINE):>16}")
    best = None
    for k in (1, 3, 5, 10):
        for m in (1, 3, 5, 10):
            if k + m > len(E):
                continue
            scores = []
            for _ in range(trials):
                idx = rng.permutation(len(E))
                template = E[idx[:k]].mean(0)
                scores.append(np.median([cosine(template, E[p]) for p in idx[k:k + m]]))
            s = np.array(scores)
            fail = (s < PUBLISHED_COSINE).mean() * 100
            print(f"  {k:>6}{m:>7}{s.min():>9.3f}{np.percentile(s, 1):>8.3f}"
                  f"{np.median(s):>9.3f}{fail:>15.1f}%")
            if fail == 0 and best is None and k >= 3 and m >= 3:
                best = (k, m, s.min())
    if best:
        k, m, worst = best
        print(f"\n  Smallest reliable setting: enrol {k}, probe {m} "
              f"-> worst-case genuine score {worst:.3f}, "
              f"{worst / PUBLISHED_COSINE:.1f}x the accept threshold.")


def report_impostor(sets: dict[str, np.ndarray]) -> None:
    names = list(sets)
    if len(names) < 2:
        print("\nIMPOSTOR pairs: NOT MEASURED - only one subject was supplied.")
        print("  No false-accept rate can be claimed. Supply videos of several")
        print("  different people to measure it.")
        return
    print("\nIMPOSTOR pairs (different people)")
    worst = -1.0
    for a, b in itertools.combinations(names, 2):
        cross = np.array([cosine(x, y) for x in sets[a] for y in sets[b]])
        worst = max(worst, cross.max())
        print(f"  {a} vs {b}: max {cross.max():.3f}  median {np.median(cross):.3f}")
    print(f"\n  Highest impostor similarity seen: {worst:.3f} "
          f"({'SAFE' if worst < PUBLISHED_COSINE else 'ABOVE THRESHOLD - unsafe'})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+", help="one video per person")
    ap.add_argument("--stride", type=int, default=6)
    args = ap.parse_args()

    sets = {}
    for v in args.videos:
        path = Path(v)
        E = embed_video(path, args.stride)
        print(f"{path.name}: {len(E)} face embeddings")
        if len(E) >= 2:
            sets[path.stem] = E

    if not sets:
        print("No faces found in any input.")
        return 1

    first = next(iter(sets.values()))
    report_genuine(first)
    report_aggregation(first)
    report_impostor(sets)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
