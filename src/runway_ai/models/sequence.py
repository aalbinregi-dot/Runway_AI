"""Keypoint sequence builder for the GRU temporal model.

Converts raw keypoint DataFrames (in the project's long-format schema) into
fixed-length padded tensors suitable for PyTorch's batch training.

The schema is::

    video_id | frame | landmark | x | y | z | visibility

Each clip is converted to a ``(T, D)`` float32 array where:
* ``T = MAX_FRAMES`` (clips are truncated or zero-padded to this length)
* ``D = n_landmarks × 4`` (x, y, z, visibility, interleaved by landmark)

A boolean ``mask`` of shape ``(T,)`` marks which frames are real (``True``)
vs. padding (``False``), so a masked mean-pooling layer can ignore padding
when computing clip-level representations.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CORE_13

# ---------------------------------------------------------------------------
# Sequence parameters
# ---------------------------------------------------------------------------
#: Pad / truncate all clips to this many frames. 120 frames ≈ 2 seconds at
#: 60 fps or 4 seconds at 30 fps — long enough to capture several gait cycles.
MAX_FRAMES: int = 120

#: Keypoint channels per landmark.
_CHANNELS = ["x", "y", "z", "visibility"]

#: Total feature dimension: n_landmarks × n_channels.
SEQ_DIM: int = len(CORE_13) * len(_CHANNELS)

_LANDMARK_ORDER = CORE_13  # canonical order


@dataclass
class ClipSequence:
    """One clip represented as a padded sequence tensor."""

    video_id: str
    frames: np.ndarray   # shape (MAX_FRAMES, SEQ_DIM), float32
    mask: np.ndarray     # shape (MAX_FRAMES,), bool  — True = real frame


def _extract_clip(df: pd.DataFrame) -> np.ndarray:
    """Convert a long-format keypoint DataFrame to ``(n_frames, SEQ_DIM)``."""
    frames_sorted = sorted(df["frame"].unique())
    n = len(frames_sorted)

    out = np.zeros((n, SEQ_DIM), dtype=np.float32)
    frame_map = {f: i for i, f in enumerate(frames_sorted)}

    for _, row in df.iterrows():
        fi = frame_map.get(row["frame"])
        li = _LANDMARK_ORDER.index(row["landmark"]) if row["landmark"] in _LANDMARK_ORDER else None
        if fi is None or li is None:
            continue
        base = li * len(_CHANNELS)
        out[fi, base    ] = float(row.get("x",          0.0))
        out[fi, base + 1] = float(row.get("y",          0.0))
        out[fi, base + 2] = float(row.get("z",          0.0))
        out[fi, base + 3] = float(row.get("visibility", 0.0))

    return out


def build_sequences(
    keypoints: pd.DataFrame,
    max_frames: int = MAX_FRAMES,
) -> list[ClipSequence]:
    """Build a list of ``ClipSequence`` objects from a long-format keypoint table.

    Parameters
    ----------
    keypoints:
        Long-format DataFrame with columns
        ``video_id, frame, landmark, x, y, z, visibility``.
    max_frames:
        Sequence length.  Clips shorter than this are zero-padded;
        longer clips are truncated from the end.

    Returns
    -------
    list of :class:`ClipSequence`, one per unique ``video_id``.
    """
    if "landmark" in keypoints.columns:
        # Filter to the CORE_13 landmarks only.
        keypoints = keypoints[keypoints["landmark"].isin(_LANDMARK_ORDER)]

    clips: list[ClipSequence] = []
    for vid, group in keypoints.groupby("video_id"):
        raw = _extract_clip(group)       # (n_frames, SEQ_DIM)
        n_real = raw.shape[0]

        # Pad / truncate.
        padded = np.zeros((max_frames, SEQ_DIM), dtype=np.float32)
        real_len = min(n_real, max_frames)
        padded[:real_len] = raw[:real_len]

        mask = np.zeros(max_frames, dtype=bool)
        mask[:real_len] = True

        clips.append(ClipSequence(video_id=str(vid), frames=padded, mask=mask))

    return clips


def load_sequences_from_csv(
    csv_path: str | Path,
    max_frames: int = MAX_FRAMES,
) -> list[ClipSequence]:
    """Convenience wrapper: load long-format CSV and build sequences."""
    df = pd.read_csv(csv_path)
    return build_sequences(df, max_frames=max_frames)
