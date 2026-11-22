"""Central configuration: landmark vocabulary, scoring criteria, paths.

The landmark names and the (video_id, frame, landmark, x, y, z, visibility)
long-format schema are taken verbatim from the supplied sample workbook
`Synthetic_Runway_Keypoints_Sample.xlsx` so that anything produced here stays
drop-in compatible with it.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Where a virtualenv keeps its interpreter, for *this* platform only.
#:
#: Selected rather than searched. Trying both layouts in order looks more
#: tolerant and is actively harmful: a project folder copied from Windows still
#: contains ``.venv311/Scripts/python.exe``, ``Path.exists()`` is true for it on
#: macOS, and handing that path to ``subprocess`` fails with "Exec format error"
#: - after the diagnostics have already reported the video pipeline as ready.
VENV_BIN = ("Scripts/python.exe",) if os.name == "nt" else ("bin/python", "bin/python3")

#: Virtualenv names that may hold a MediaPipe-capable interpreter, best first.
VENV_NAMES = (".venv312", ".venv311", ".venv310", ".venv39")


def launcher() -> str:
    """The launcher command for this platform, for use in user-facing messages."""
    return "run.bat" if os.name == "nt" else "./run.sh"


def find_video_interpreter(root: Path | None = None) -> Path | None:
    """A sibling virtualenv whose interpreter can run MediaPipe *on this platform*.

    Lives here rather than in :mod:`runway_ai.pose.delegate` - which re-exports it -
    because the two callers that most need it are status surfaces: ``runway_ai
    info`` and ``GET /diagnostics``. Importing the delegate pulls in pandas, and a
    command whose whole job is to report on a broken install must not itself fall
    over on a missing dependency.

    The layout comes from :data:`VENV_BIN`, selected by platform rather than tried
    in order. That is load-bearing; see the note on ``VENV_BIN``. The executable
    check is the other half: a ``.exe`` unpacked from a zip can arrive with the
    execute bit set, and what the caller needs is something it can actually run.
    """
    root = Path(root or PROJECT_ROOT)
    for name in VENV_NAMES:
        for rel in VENV_BIN:
            candidate = root / name / rel
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return candidate
    return None


def video_env_steps() -> tuple[str, str, str]:
    """``(create, install, alternative)`` for the MediaPipe environment.

    Returned as parts so a caller can lay them out for its own medium. The
    diagnostics report joins the two commands with " then " and needs the third to
    read as an aside; a terminal wants three lines.
    """
    if os.name == "nt":
        create = "py -3.12 -m venv .venv312"
        install = r".venv312\Scripts\pip install -r requirements-video.txt"
    else:
        create = "python3.12 -m venv .venv312"
        install = ".venv312/bin/pip install -r requirements-video.txt"
    return create, install, f"or let the launcher do it: {launcher()} setup"


def video_env_hint(indent: str = "      ") -> str:
    """How to create the MediaPipe environment, in this platform's syntax.

    One copy, because there were five: two in ``pose/delegate.py``, one in
    ``pose/extract.py``, one in ``app/diagnostics.py`` and one in the launcher.
    Every one of them told a macOS user to run ``py -3.11`` and then to use a
    path spelled with backslashes, which is the moment they most needed a
    command that works.
    """
    create, install, alternative = video_env_steps()
    return (
        f"{indent}{create}\n"
        f"{indent}{install}\n"
        f"{indent}({alternative})"
    )
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODEL_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Landmarks
# --------------------------------------------------------------------------- #
#: The full MediaPipe Pose (BlazePose) topology, index -> name.
MEDIAPIPE_33 = [
    "nose",
    "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]

#: The 13-point analysis subset used by the supplied workbook. Every downstream
#: feature is computed from these so the pipeline runs on either topology.
CORE_13 = [
    "nose",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

LANDMARK_INDEX = {name: i for i, name in enumerate(CORE_13)}

#: Left/right pairs, used for symmetry and alignment features.
BILATERAL_PAIRS = [
    ("left_shoulder", "right_shoulder"),
    ("left_elbow", "right_elbow"),
    ("left_wrist", "right_wrist"),
    ("left_hip", "right_hip"),
    ("left_knee", "right_knee"),
    ("left_ankle", "right_ankle"),
]

# Canonical long-format columns (matches the sample workbook exactly).
KEYPOINT_COLUMNS = ["video_id", "frame", "landmark", "x", "y", "z", "visibility"]

# --------------------------------------------------------------------------- #
# Scoring criteria
# --------------------------------------------------------------------------- #
#: The twelve judged criteria. Each is scored 0-10; the weighted sum is
#: rescaled to an overall 0-100. Weights reflect how runway coaches prioritise
#: faults: postural and balance errors are the most penalised, stage presence
#: and rhythm carry real but smaller weight.
CRITERIA = [
    "posture",
    "balance",
    "confidence",
    "shoulder_alignment",
    "head_position",
    "arm_swing",
    "hip_movement",
    "rhythm",
    "stride_length",
    "foot_placement",
    "symmetry",
    "stage_presence",
]

CRITERION_WEIGHTS = {
    "posture": 0.13,
    "balance": 0.12,
    "confidence": 0.10,
    "shoulder_alignment": 0.08,
    "head_position": 0.07,
    "arm_swing": 0.07,
    "hip_movement": 0.08,
    "rhythm": 0.09,
    "stride_length": 0.07,
    "foot_placement": 0.08,
    "symmetry": 0.06,
    "stage_presence": 0.05,
}

assert abs(sum(CRITERION_WEIGHTS.values()) - 1.0) < 1e-9, "weights must sum to 1"

CRITERION_LABELS = {
    "posture": "Posture & Spinal Alignment",
    "balance": "Balance & Stability",
    "confidence": "Walking Confidence",
    "shoulder_alignment": "Shoulder Alignment",
    "head_position": "Head Position & Gaze",
    "arm_swing": "Arm Swing",
    "hip_movement": "Hip Movement",
    "rhythm": "Walking Rhythm",
    "stride_length": "Stride Length",
    "foot_placement": "Foot Placement",
    "symmetry": "Body Symmetry",
    "stage_presence": "Stage Presence",
}

#: Performance bands applied to the overall 0-100 score.
GRADE_BANDS = [
    (90, "Professional"),
    (80, "Advanced"),
    (70, "Proficient"),
    (60, "Developing"),
    (45, "Beginner"),
    (0, "Foundational"),
]

# --------------------------------------------------------------------------- #
# Signal-processing defaults
# --------------------------------------------------------------------------- #
DEFAULT_FPS = 30.0
SMOOTHING_WINDOW = 7          # Savitzky-Golay window (frames, odd)
SMOOTHING_POLYORDER = 2
MIN_VISIBILITY = 0.5          # below this a landmark is treated as missing
MIN_FRAMES = 30               # refuse to score clips shorter than this
RANDOM_SEED = 42
