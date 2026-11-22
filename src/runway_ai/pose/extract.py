"""Video -> keypoints.

Pose estimation is deliberately behind a small interface. Everything downstream
consumes a :class:`~runway_ai.data.schema.PoseSequence`, never a detector, so
the backend can be swapped - MediaPipe today, a 3D lifter or a multi-camera rig
later - without touching the feature or model code.

**Two MediaPipe APIs are supported.** MediaPipe 1.0 removed the legacy
``mp.solutions.pose`` module in favour of the Tasks API
(``mediapipe.tasks.python.vision.PoseLandmarker``), so code written against
either one breaks on the other release line. Both are implemented here and the
available one is detected at runtime; the Tasks API is preferred where present.

MediaPipe also ships wheels only for CPython 3.9-3.12, so on a newer
interpreter the import fails outright. Rather than making the project unusable,
the backend is imported lazily and its absence reported with the exact remedy.
Keypoint CSV/Excel input works regardless.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import numpy as np

from ..config import CORE_13, DEFAULT_FPS, MEDIAPIPE_33, MIN_FRAMES, MODEL_DIR
from ..data.schema import PoseSequence

#: Tasks-API model. Downloaded once and cached; ~9 MB.
TASK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
TASK_MODEL_PATH = MODEL_DIR / "mediapipe" / "pose_landmarker_full.task"


class PoseBackendUnavailable(RuntimeError):
    """Raised when no pose estimator can be loaded."""


def detect_backend() -> str:
    """Return ``"tasks"``, ``"solutions"`` or ``"none"``."""
    try:
        import mediapipe as mp
    except Exception:
        return "none"
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        return "tasks"
    except Exception:
        pass
    if hasattr(mp, "solutions") and hasattr(mp.solutions, "pose"):
        return "solutions"
    return "none"


def mediapipe_available() -> bool:
    return detect_backend() != "none"


def _install_hint() -> str:
    from ..config import video_env_hint

    v = f"{sys.version_info.major}.{sys.version_info.minor}"
    return (
        f"No MediaPipe pose backend for this interpreter (Python {v}).\n"
        "  MediaPipe publishes wheels for CPython 3.9-3.12 only. On a newer\n"
        "  interpreter, create a compatible environment:\n"
        f"{video_env_hint()}\n"
        "  Then extract keypoints with that interpreter. Everything else in\n"
        "  this project - features, training, scoring, reports - runs on any\n"
        "  supported Python, because it consumes keypoints rather than pixels."
    )


def ensure_task_model(path: Path = TASK_MODEL_PATH) -> Path:
    """Download the Tasks-API model on first use, then reuse the cached copy."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading pose model (~9 MB) -> {path}", flush=True)
    try:
        urllib.request.urlretrieve(TASK_MODEL_URL, path)
    except Exception as exc:
        raise PoseBackendUnavailable(
            f"could not download the pose model from {TASK_MODEL_URL}: {exc}\n"
            f"  Download it manually and save it to {path}."
        ) from exc
    return path


# --------------------------------------------------------------------------- #
# Frame reading
# --------------------------------------------------------------------------- #
def _read_frames(path: Path, stride: int, max_frames: int | None):
    """Yield ``(index, rgb_frame)`` and report fps via the returned generator's
    ``fps`` attribute set on first use."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or DEFAULT_FPS
    frames = []
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                if max_frames and len(frames) >= max_frames:
                    break
            idx += 1
    finally:
        cap.release()
    return frames, float(fps)


def _blank_frame(n_keep: int):
    """A missing detection, kept in place so the time axis stays uniform.

    Dropping undetected frames instead would silently compress the timeline and
    corrupt every cadence and rhythm measurement.
    """
    return ([[np.nan, np.nan]] * n_keep, [0.0] * n_keep, [0.0] * n_keep)


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
def _extract_tasks(frames, fps, keep, model_complexity, on_progress=None):
    """MediaPipe >= 1.0 Tasks API."""
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        PoseLandmarker, PoseLandmarkerOptions, RunningMode,
    )

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_task_model())),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    xy, z, vis = [], [], []
    with PoseLandmarker.create_from_options(options) as landmarker:
        for i, rgb in enumerate(frames):
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(i * 1000 / max(fps, 1e-6))
            res = landmarker.detect_for_video(image, ts_ms)
            if on_progress and (i % 5 == 0 or i == len(frames) - 1):
                on_progress(i + 1, len(frames))
            if res.pose_landmarks:
                lms = res.pose_landmarks[0]
                xy.append([[lms[k].x, lms[k].y] for k in keep])
                z.append([lms[k].z for k in keep])
                vis.append([getattr(lms[k], "visibility", 1.0) or 1.0 for k in keep])
            else:
                a, b, c = _blank_frame(len(keep))
                xy.append(a); z.append(b); vis.append(c)
    return xy, z, vis


def _extract_solutions(frames, fps, keep, model_complexity, on_progress=None):
    """MediaPipe 0.10.x legacy solutions API."""
    import mediapipe as mp

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    xy, z, vis = [], [], []
    try:
        for i, rgb in enumerate(frames):
            res = pose.process(rgb)
            if on_progress and (i % 5 == 0 or i == len(frames) - 1):
                on_progress(i + 1, len(frames))
            if res.pose_landmarks:
                lms = res.pose_landmarks.landmark
                xy.append([[lms[k].x, lms[k].y] for k in keep])
                z.append([lms[k].z for k in keep])
                vis.append([lms[k].visibility for k in keep])
            else:
                a, b, c = _blank_frame(len(keep))
                xy.append(a); z.append(b); vis.append(c)
    finally:
        pose.close()
    return xy, z, vis


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #
def extract_from_video(
    path: str | Path,
    video_id: str | None = None,
    model_complexity: int = 1,
    max_frames: int | None = None,
    stride: int = 1,
    progress: bool = True,
    on_progress=None,
) -> PoseSequence:
    """Run MediaPipe Pose over a video file and return a ``PoseSequence``.

    Only the 13 analysis landmarks are kept, matching the sample workbook's
    schema and the feature code.
    """
    backend = detect_backend()
    if backend == "none":
        raise PoseBackendUnavailable(_install_hint())

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    video_id = video_id or path.stem

    if on_progress:
        on_progress(0, 0)          # signals "decoding", frame count unknown yet
    frames, fps = _read_frames(path, stride, max_frames)
    if progress:
        print(f"  {len(frames)} frames at {fps:.0f} fps, backend={backend}", flush=True)
    if not frames:
        raise ValueError(f"no readable frames in {path.name}")

    keep = [MEDIAPIPE_33.index(name) for name in CORE_13]
    runner = _extract_tasks if backend == "tasks" else _extract_solutions
    xy, z, vis = runner(frames, fps, keep, model_complexity, on_progress)

    detected = int(sum(1 for v in vis if max(v) > 0))
    if progress:
        print(f"  pose found in {detected}/{len(frames)} frames", flush=True)

    if detected < MIN_FRAMES:
        raise ValueError(
            f"a person was detected in only {detected} of {len(frames)} frames in "
            f"{path.name}; need at least {MIN_FRAMES}. Check that the whole body is "
            "in frame, the subject is well lit, and the clip is at least 4 seconds."
        )

    return PoseSequence(
        video_id=video_id,
        landmarks=list(CORE_13),
        xy=np.asarray(xy, dtype=float),
        z=np.asarray(z, dtype=float),
        visibility=np.asarray(vis, dtype=float),
        fps=float(fps) / stride,
        meta={"source": str(path), "backend": f"mediapipe:{backend}",
              "frames_detected": detected, "frames_total": len(frames),
              # Image coordinates are normalised per axis, so a horizontal span
              # cannot be compared with a vertical one without this.
              "frame_size": [int(frames[0].shape[1]), int(frames[0].shape[0])]},
    )
