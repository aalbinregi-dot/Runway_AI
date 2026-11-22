"""Opt-in face identification: which enrolled user does this walk belong to?

Scope, deliberately narrow
--------------------------
This performs 1:N matching against users who have **explicitly enrolled their
own face**. It exists to save someone picking their name from a list before
uploading their own walk. It is not, and must not become, a way to identify
people who have not consented: nothing is matched unless a user has enrolled,
and a clip whose face matches no enrolment returns ``None`` rather than a best
guess.

What is stored is a 128-dimensional embedding, never an image. Embeddings are
not anonymous - they are biometric data - but they are far less exposing than
the photographs they came from, and the raw enrolment frames are discarded once
the template is built.

Both models ship with OpenCV, so this adds no Python dependency:
* **YuNet** for detection (227 KB)
* **SFace** for embedding (37 MB), 128-d, cosine similarity

Operating point
---------------
Everything below was measured on this system by ``scripts/calibrate_face.py``,
not adopted from the model card:

* Single frame against single frame, same person: worst case **0.328**, with
  **1.2%** of genuine pairs falling below the published 0.363 accept threshold.
  Single-frame identification is therefore not reliable and is refused.
* Aggregating **5 enrolment frames and 5 probe frames** lifts the worst-case
  same-person score to **0.750** - roughly twice the accept threshold - with no
  genuine failures across 400 trials.

The impostor distribution has *not* been measured: that needs footage of
several different people, and only one subject was available. No false-accept
rate is claimed anywhere in this module, and the accept bar is the published
threshold rather than one fitted to a single face.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import MODEL_DIR

DETECTOR_PATH = MODEL_DIR / "face" / "face_detection_yunet_2023mar.onnx"
EMBEDDER_PATH = MODEL_DIR / "face" / "face_recognition_sface_2021dec.onnx"

EMBEDDING_DIM = 128

#: SFace's published cosine operating point. Used as the accept bar rather than
#: a threshold fitted here, because fitting one to a single subject would
#: overstate what has been validated.
ACCEPT_COSINE = 0.363

#: Frames required to enrol and to identify. Both come from the aggregation
#: measurement above; below these the worst-case genuine score drops under the
#: accept bar and identification starts silently failing.
MIN_ENROL_FRAMES = 5
MIN_PROBE_FRAMES = 5

#: In 1:N matching the runner-up matters as much as the winner. Two enrolled
#: users scoring 0.40 and 0.39 is not an identification, it is a coin toss, so
#: the best match must clear the second best by this margin. The value is a
#: conservative default, NOT a measured one - it cannot be calibrated without
#: multi-person footage.
MIN_MARGIN = 0.10

#: How far a detected face may sit from the pose track's nose landmark, in face
#: widths, to be accepted as the walker's. Measured on real footage: the
#: walker's own face never exceeded 0.26. A bystander sits far outside this.
MAX_NOSE_DISTANCE = 1.0

DETECT_SCORE = 0.6
DETECT_NMS = 0.3

#: Quality floor for *enrolment* frames only. Measured, not guessed: the
#: calibration templates came from ~50 px faces at ~0.92 confidence and reached
#: a 0.750 worst-case genuine score, so this sits just below that.
ENROL_MIN_SCORE = 0.80
ENROL_MIN_WIDTH = 45


class FaceUnavailable(RuntimeError):
    """Raised when the face models are not present."""


@dataclass
class Detection:
    box: tuple[int, int, int, int]      # x, y, w, h
    score: float
    embedding: np.ndarray               # (128,) float32


@dataclass
class Identification:
    """The outcome of matching a clip against enrolled users."""

    user_id: int | None                 # None means "not identified"
    score: float                        # best median cosine similarity
    margin: float                       # gap to the runner-up
    reason: str                         # why, in plain words
    frames_used: int

    @property
    def identified(self) -> bool:
        return self.user_id is not None

    def as_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "identified": self.identified,
            "score": round(self.score, 4),
            "margin": round(self.margin, 4),
            "reason": self.reason,
            "frames_used": self.frames_used,
        }


def available() -> bool:
    return DETECTOR_PATH.exists() and EMBEDDER_PATH.exists()


def _models(width: int, height: int):
    import cv2

    if not available():
        raise FaceUnavailable(
            "Face models are not installed. Fetch them once with:\n"
            "    python scripts/fetch_face_models.py"
        )
    det = cv2.FaceDetectorYN.create(
        str(DETECTOR_PATH), "", (width, height), DETECT_SCORE, DETECT_NMS, 5000)
    rec = cv2.FaceRecognizerSF.create(str(EMBEDDER_PATH), "")
    return det, rec


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
def detect_in_frame(img, det, rec) -> list[Detection]:
    _, faces = det.detect(img)
    if faces is None:
        return []
    out = []
    for f in faces:
        try:
            emb = rec.feature(rec.alignCrop(img, f)).ravel()
        except Exception:
            continue
        out.append(Detection(box=tuple(int(v) for v in f[:4]),
                             score=float(f[-1]), embedding=emb))
    return out


def _pick_walker(dets: list[Detection], nose_xy, width: int, height: int) -> Detection | None:
    """Choose the analysed subject's face, not a bystander's.

    Size alone is the obvious heuristic and it is wrong: someone standing
    closer to the camera behind the runway outranks the walker. This system
    already knows where the walker's head is - the pose track's nose landmark -
    so the face nearest that point wins. On real footage the walker's own face
    stayed within 0.26 face widths of it.
    """
    if not dets:
        return None
    if nose_xy is None:
        return max(dets, key=lambda d: d.box[2] * d.box[3])

    nx, ny = nose_xy[0] * width, nose_xy[1] * height
    best, best_d = None, np.inf
    for d in dets:
        cx, cy = d.box[0] + d.box[2] / 2, d.box[1] + d.box[3] / 2
        dist = float(np.hypot(cx - nx, cy - ny) / max(d.box[2], 1))
        if dist < best_d:
            best, best_d = d, dist
    return best if best_d <= MAX_NOSE_DISTANCE else None


def embeddings_from_video(
    path: str | Path,
    n_frames: int = 12,
    nose_track: np.ndarray | None = None,
) -> list[np.ndarray]:
    """Sample frames evenly and return the walker's face embedding from each.

    ``nose_track`` is the pose track's (T, 2) normalised nose trajectory. When
    supplied it disambiguates the walker from anyone else in shot.
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    det, rec = _models(w, h)

    # Skip the first and last 10%: the subject is furthest away there, and the
    # face is correspondingly smallest and least reliable.
    lo, hi = int(total * 0.1), int(total * 0.9)
    picks = np.linspace(lo, max(hi, lo + 1), n_frames).astype(int)

    out = []
    try:
        for f in picks:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(f))
            ok, img = cap.read()
            if not ok:
                continue
            nose = None
            if nose_track is not None and len(nose_track):
                nose = nose_track[min(int(f), len(nose_track) - 1)]
            chosen = _pick_walker(detect_in_frame(img, det, rec), nose, w, h)
            if chosen is not None:
                out.append(chosen.embedding)
    finally:
        cap.release()
    return out


def embeddings_from_captures(frames: list[np.ndarray]) -> list[np.ndarray]:
    """Embeddings from live webcam captures, for enrolment.

    Deliberately takes decoded frames rather than file paths. Accepting
    arbitrary image *files* would make enrolment the one operation in this
    system capable of registering somebody who never consented - upload a
    photograph of any person and they become identifiable. Enrolment is
    therefore restricted to frames the user captured of themselves, either from
    the webcam or from a walk video they just uploaded.

    This raises the bar; it does not make impersonation impossible. SFace has no
    liveness detection, so a webcam pointed at a printed photograph still
    enrols. Anything relying on this for access control would need a dedicated
    presentation-attack model, which this system does not have and does not
    claim.
    """
    import cv2

    out = []
    for img in frames:
        if img is None or getattr(img, "size", 0) == 0:
            continue
        det, rec = _models(img.shape[1], img.shape[0])
        dets = detect_in_frame(img, det, rec)
        if not dets:
            continue
        # Enrolment is worth gating on quality; probes are not, since a strict
        # gate starves the sampler. The numbers come from the calibration run
        # rather than intuition: the templates that produced a 0.750 worst-case
        # genuine score were built from faces around 50 px wide at detector
        # confidence ~0.92. Gating at 80 px - as a first attempt did - rejected
        # 8 of 9 frames from a perfectly usable clip.
        best = max(dets, key=lambda d: d.box[2] * d.box[3])
        if best.score >= ENROL_MIN_SCORE and best.box[2] >= ENROL_MIN_WIDTH:
            out.append(best.embedding)
    return out


# --------------------------------------------------------------------------- #
# Enrolment and matching
# --------------------------------------------------------------------------- #
def build_template(embeddings: list[np.ndarray]) -> np.ndarray:
    """Condense enrolment embeddings into one L2-normalised centroid.

    The centroid of several views is markedly more stable than any single one:
    measured here, going from one enrolment frame to five lifted the worst-case
    same-person score from 0.328 to 0.750.
    """
    if len(embeddings) < MIN_ENROL_FRAMES:
        raise ValueError(
            f"need at least {MIN_ENROL_FRAMES} usable face frames to enrol, "
            f"got {len(embeddings)}. Use a clearer, better-lit clip or photo set."
        )
    m = np.mean(np.stack([np.ravel(e) for e in embeddings]), axis=0)
    return (m / (np.linalg.norm(m) + 1e-12)).astype(np.float32)


def identify(
    probes: list[np.ndarray],
    templates: dict[int, np.ndarray],
    accept: float = ACCEPT_COSINE,
    margin: float = MIN_MARGIN,
) -> Identification:
    """Match probe embeddings against enrolled templates.

    Returns an unidentified result rather than a best guess whenever the
    evidence is weak - too few frames, nobody above the accept bar, or two
    users too close to separate.
    """
    if not templates:
        return Identification(None, 0.0, 0.0, "no users have enrolled a face", 0)

    if len(probes) < MIN_PROBE_FRAMES:
        return Identification(
            None, 0.0, 0.0,
            f"only {len(probes)} usable face frames; {MIN_PROBE_FRAMES} are "
            "needed for a reliable match",
            len(probes),
        )

    # Median across frames, not mean: a couple of blurred or half-turned frames
    # should not drag an otherwise clear identification down.
    scored = sorted(
        ((int(np.median([cosine(t, p) for p in probes])* 1e6) / 1e6, uid)
         for uid, t in templates.items()),
        reverse=True,
    )
    best_score, best_uid = scored[0]
    second = scored[1][0] if len(scored) > 1 else -1.0
    gap = best_score - second if len(scored) > 1 else float("inf")

    if best_score < accept:
        return Identification(
            None, best_score, gap if np.isfinite(gap) else 0.0,
            "this face does not match any enrolled user", len(probes))

    if gap < margin:
        return Identification(
            None, best_score, gap,
            "two enrolled users match too closely to tell apart", len(probes))

    return Identification(
        best_uid, best_score, gap if np.isfinite(gap) else 1.0,
        "matched an enrolled face", len(probes))
