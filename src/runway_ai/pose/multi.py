"""Analyse every walker in one clip, not just the nearest.

Why crop-and-run rather than ``num_poses``
------------------------------------------
MediaPipe's PoseLandmarker takes a ``num_poses`` argument, and on this footage
it does not work. Measured on a two-person fixture, sampling every tenth frame:

======================================  ===========================
approach                                both walkers found
======================================  ===========================
whole frame, ``num_poses=4``, VIDEO     3% of frames
whole frame, ``num_poses=4``, IMAGE     5% of frames
crop per person, ``num_poses=1``        **100% of frames**
======================================  ===========================

The reason is resolution. The model rescales whatever it is given to a small
fixed input, so on a wide frame each person shrinks below what the detector can
work with. Cropping restores their pixel size, and in VIDEO mode the tracker
also latches onto one subject and stays there.

Faces do the identity work
--------------------------
Cropping needs to know where each person is, and *staying with the same person*
across frames needs identity. Both come from the face detector that already
runs for enrolment: YuNet finds every face in a frame, and SFace turns each into
an embedding that matches the same person frame to frame even when they cross,
overlap, or briefly leave the shot. Nearest-box tracking cannot do that; it
swaps identities the moment two walkers pass each other.

The consequence, stated plainly: **a walker whose face is never visible cannot
be separated out.** For a runway pass toward the camera that is the normal case,
but a person filmed from behind will not be found, and the result says how many
walkers it located rather than implying it found everyone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import CORE_13, MEDIAPIPE_33
from ..data.schema import PoseSequence

#: How far a face embedding may sit from a track's running template and still be
#: taken as the same person. Deliberately looser than the identification
#: threshold in :mod:`runway_ai.face`: this is association within one clip,
#: where the alternative to a slightly wrong match is a broken track, not a
#: misattributed identity.
TRACK_COSINE = 0.30

#: Weight of the position tiebreaker, and the distance over which it decays
#: (as a fraction of frame width). Small on purpose - see :func:`_continuity`.
CONTINUITY_WEIGHT = 0.15
CONTINUITY_SPAN = 0.25

#: Body box around a face, in face widths. A standing adult is roughly seven
#: head-heights tall, so the crop runs from just above the head to well below
#: the feet, with lateral room for arm swing.
CROP_ABOVE = 1.2
CROP_BELOW = 8.5
CROP_SIDE = 3.0

#: A track seen in fewer frames than this is noise - a bystander glimpsed at the
#: edge, or a false detection - not a walker to be scored.
MIN_TRACK_FRAMES = 20


@dataclass
class PersonTrack:
    """One walker's detections through a clip."""

    track_id: int
    frames: list[int] = field(default_factory=list)
    boxes: list[tuple] = field(default_factory=list)
    embeddings: list[np.ndarray] = field(default_factory=list)
    landmarks: dict[int, np.ndarray] = field(default_factory=dict)
    visibility: dict[int, np.ndarray] = field(default_factory=dict)

    @property
    def template(self) -> np.ndarray:
        m = np.mean(np.stack(self.embeddings), axis=0)
        return m / (np.linalg.norm(m) + 1e-12)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.ravel(a), np.ravel(b)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def _continuity(track: PersonTrack, box, frame_w: int) -> float:
    """Small bonus for a detection near where this track last was.

    Face similarity decides who is who; this only breaks ties. The weight is set
    so it cannot override an appearance difference - two different people differ
    by far more than ``CONTINUITY_WEIGHT`` in cosine, so position never pulls a
    track onto the wrong person. But two people the recogniser cannot tell apart
    (look-alikes, motion blur, a face too small to embed well) differ by almost
    nothing, and then position is the only signal left. Without it the two
    tracks swap places at random from frame to frame.
    """
    if not track.boxes:
        return 0.0
    last = track.boxes[-1]
    dx = abs((box[0] + box[2] / 2) - (last[0] + last[2] / 2)) / max(frame_w, 1)
    return CONTINUITY_WEIGHT * max(0.0, 1.0 - dx / CONTINUITY_SPAN)


def _body_box(face_box, w: int, h: int) -> tuple[int, int, int, int]:
    """Expand a face box to a crop that should contain the whole body.

    Always returns ``x0 <= x1`` and ``y0 <= y1``. A detection reported outside
    the frame used to come back inverted (``x0`` past ``x1``), which numpy slices
    as an *empty* array rather than raising - the caller's size check happened to
    reject it, but a silently empty crop is the kind of thing that only surfaces
    once something downstream stops checking.
    """
    fx, fy, fw, fh = face_box
    cx = fx + fw / 2
    x0 = int(min(max(0, cx - CROP_SIDE * fw), w))
    x1 = int(min(max(0, cx + CROP_SIDE * fw), w))
    y0 = int(min(max(0, fy - CROP_ABOVE * fh), h))
    y1 = int(min(max(0, fy + CROP_BELOW * fh), h))
    return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)


def find_people(video_path: str | Path, stride: int = 2,
                on_progress=None) -> list[PersonTrack]:
    """Locate and track every walker whose face is visible."""
    import cv2

    from .. import face as face_mod

    if not face_mod.available():
        raise RuntimeError("Face models are required to separate multiple people.")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"could not open video: {video_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    det = cv2.FaceDetectorYN.create(str(face_mod.DETECTOR_PATH), "", (w, h),
                                    face_mod.DETECT_SCORE, face_mod.DETECT_NMS, 5000)
    rec = cv2.FaceRecognizerSF.create(str(face_mod.EMBEDDER_PATH), "")

    tracks: list[PersonTrack] = []
    try:
        # Decoded sequentially, not by seeking to each wanted frame: measured
        # 1.7s against 6.7s for the same 93 frames, because a seek forces the
        # decoder back to the previous keyframe every time. ``grab`` advances
        # without decoding, so skipped frames cost almost nothing - and unlike
        # a seek, the frame number is exact rather than snapped to a keyframe.
        f = -1
        while True:
            if not cap.grab():
                break
            f += 1
            if f % stride:
                continue
            ok, img = cap.retrieve()
            if not ok:
                continue
            _, faces = det.detect(img)
            if faces is None:
                continue

            found = []
            for row in faces:
                try:
                    emb = rec.feature(rec.alignCrop(img, row)).ravel()
                except Exception:
                    continue
                found.append((tuple(float(v) for v in row[:4]), emb))
            if not found:
                continue

            # Two faces visible at once are two different people - no matter how
            # alike the embeddings look. That constraint is stronger than any
            # similarity score, so each track may take at most one detection per
            # frame. Without it, look-alikes (and genuinely identical footage)
            # collapse into a single track and the group is scored as one
            # walker.
            # Ranked by face-plus-position, but *gated* on face alone: the
            # tiebreaker may reorder candidates that are all plausible, never
            # admit one the recogniser rejected.
            scored = sorted(
                ((_cosine(t.template, e) + _continuity(t, box, w),
                  _cosine(t.template, e), di, ti)
                 for ti, t in enumerate(tracks)
                 for di, (box, e) in enumerate(found)),
                reverse=True)

            taken_det, taken_track = set(), set()
            for _rank, face_score, di, ti in scored:
                if face_score < TRACK_COSINE or di in taken_det or ti in taken_track:
                    continue
                taken_det.add(di)
                taken_track.add(ti)
                box, emb = found[di]
                tracks[ti].frames.append(f)
                tracks[ti].boxes.append(box)
                tracks[ti].embeddings.append(emb)

            for di, (box, emb) in enumerate(found):
                if di in taken_det:
                    continue
                t = PersonTrack(track_id=len(tracks))
                t.frames.append(f)
                t.boxes.append(box)
                t.embeddings.append(emb)
                tracks.append(t)

            if on_progress:
                on_progress(f, total)
    finally:
        cap.release()

    return [t for t in tracks if len(t.frames) >= MIN_TRACK_FRAMES]


def pose_for_track(video_path: str | Path, track: PersonTrack,
                   fps: float | None = None) -> PoseSequence | None:
    """Run single-person pose inside this walker's crop, frame by frame.

    Landmarks come back in crop coordinates and are mapped straight back to the
    full frame, so everything downstream sees ordinary image-space keypoints and
    needs no knowledge that cropping happened.
    """
    import cv2
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    )

    from .extract import ensure_task_model

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = fps or cap.get(cv2.CAP_PROP_FPS) or 30.0

    keep = [MEDIAPIPE_33.index(n) for n in CORE_13]
    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_task_model())),
        running_mode=RunningMode.IMAGE, num_poses=1,
        min_pose_detection_confidence=0.5)

    # Sequential decode again, for the same reason as find_people. The wanted
    # frames are ascending, so one forward pass reaches all of them.
    wanted = dict(zip(track.frames, track.boxes))
    last_wanted = max(wanted) if wanted else -1

    xy, vis, frames = [], [], []
    try:
        with PoseLandmarker.create_from_options(options) as lm:
            f = -1
            while f < last_wanted:
                if not cap.grab():
                    break
                f += 1
                box = wanted.get(f)
                if box is None:
                    continue
                ok, img = cap.retrieve()
                if not ok:
                    continue
                x0, y0, x1, y1 = _body_box(box, w, h)
                if x1 - x0 < 40 or y1 - y0 < 60:
                    continue
                crop = img[y0:y1, x0:x1]
                res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB,
                                         data=cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
                if not res.pose_landmarks:
                    continue
                pts = res.pose_landmarks[0]
                cw, ch = x1 - x0, y1 - y0
                xy.append([[(x0 + pts[k].x * cw) / w, (y0 + pts[k].y * ch) / h]
                           for k in keep])
                vis.append([getattr(pts[k], "visibility", 1.0) or 1.0 for k in keep])
                frames.append(f)
    finally:
        cap.release()

    if len(frames) < 10:
        return None

    # Frames were sampled at a stride and some were dropped, so the effective
    # rate is the true one - getting this wrong would skew every cadence and
    # rhythm measurement.
    span = max(frames[-1] - frames[0], 1)
    effective_fps = fps * len(frames) / span

    return PoseSequence(
        video_id=f"person-{track.track_id}",
        landmarks=list(CORE_13),
        xy=np.asarray(xy, dtype=float),
        z=np.zeros((len(frames), len(CORE_13))),
        visibility=np.asarray(vis, dtype=float),
        fps=float(effective_fps),
        meta={"frame_size": [w, h], "track_id": track.track_id,
              "frames_used": len(frames), "source_frames": frames},
    )


def extract_all(video_path: str | Path, stride: int = 4, on_progress=None,
                on_person=None) -> list[tuple[PoseSequence, np.ndarray]]:
    """Every walker in the clip, as a pose sequence plus a face template.

    The template travels with the sequence so the scoring process - which is a
    *different interpreter*, see :mod:`runway_ai.pose.delegate` - can attribute
    each walk to an enrolled account without re-opening the video.

    ``on_person(seq, template, done, total)`` fires as each walker's pose is
    extracted, before the next one starts. Pose estimation is the expensive part
    - 80 seconds for two people against 0.2 seconds to score them - so a caller
    that waits for this list gets one long silence, whereas one that listens here
    can show each walker as it lands.
    """
    # Both phases report on one scale. Reported on their own the bar ran to
    # 100% and then back to 50%, because the face scan counts frames while the
    # pose pass counts people - and it was the slower phase that looked like it
    # had gone backwards. The face scan fills the first half, pose the second.
    scanned = {"frames": 0}

    def scan_progress(done: int, total: int) -> None:
        scanned["frames"] = total
        on_progress(done, total * 2)

    tracks = find_people(video_path, stride=stride,
                         on_progress=scan_progress if on_progress else None)

    span = scanned["frames"] or 1
    out = []
    for i, t in enumerate(tracks):
        if on_progress:
            on_progress(span + span * i // max(len(tracks), 1), span * 2)
        seq = pose_for_track(video_path, t)
        if seq is not None:
            out.append((seq, t.template))
            if on_person:
                on_person(seq, t.template, len(out), len(tracks))
    if on_progress:
        on_progress(span * 2, span * 2)
    # Left to right as they appear, so "person 1" means the same thing to the
    # viewer as it does in the numbering on screen.
    #
    # Only ``person_index`` is assigned here. ``video_id`` deliberately keeps the
    # track-based name it was written under: each walker's keypoints are streamed
    # to disk during extraction, before the ordering is known, and renaming them
    # afterwards left the streamed files and the manifest referring to different
    # ids - which reconciled into three results for a two-person clip.
    out.sort(key=lambda p: float(np.median(p[0].xy[:, :, 0])))
    for i, (seq, _) in enumerate(out):
        seq.meta["person_index"] = i + 1
    return out
