"""Readers for keypoint tables plus the signal-conditioning stage.

Raw pose output is never clean: landmarks drop out behind the body, the
detector jitters frame to frame, and the subject changes apparent size as they
walk toward the camera. ``preprocess`` fixes all three before any feature is
computed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from ..config import (
    CORE_13,
    DEFAULT_FPS,
    MIN_VISIBILITY,
    SMOOTHING_POLYORDER,
    SMOOTHING_WINDOW,
)
from .schema import PoseSequence, long_to_sequences, validate_long


# --------------------------------------------------------------------------- #
# Readers
# --------------------------------------------------------------------------- #
def read_keypoints(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """Read a long-format keypoint table from .xlsx, .xls, .csv or .parquet."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls", ".xlsm"}:
        df = pd.read_excel(path, sheet_name=sheet)
    elif suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        # Parquet needs an engine, and neither pyarrow nor fastparquet is in
        # requirements.txt - both are large wheels for an input format almost
        # nobody uses here. pandas' own error for the missing engine names two
        # packages and not the reason, so it is worth one line of context.
        try:
            df = pd.read_parquet(path)
        except ImportError as exc:
            raise ImportError(
                "Reading .parquet needs a parquet engine, which this project does "
                "not install by default:\n"
                "    python -m pip install pyarrow\n"
                "CSV and XLSX keypoint files need nothing extra."
            ) from exc
    else:
        raise ValueError(f"unsupported keypoint file type: {suffix}")
    return validate_long(df)


def load_sequences(
    path: str | Path,
    landmarks: list[str] | None = None,
    fps: float = DEFAULT_FPS,
    sheet: str | int = 0,
    conditioned: bool = True,
) -> dict[str, PoseSequence]:
    """Read a file and return ``PoseSequence`` objects.

    ``conditioned=False`` returns the raw image-space tracks. Preprocessing
    moves the pelvis to the origin and rescales by torso length, which is what
    the features need and exactly what a skeleton overlay cannot use - so a
    caller that wants to keep the track for replay must ask for the raw form
    and let ``analyze_sequence`` condition it afterwards.
    """
    df = read_keypoints(path, sheet=sheet)
    seqs = long_to_sequences(df, landmarks=landmarks or CORE_13, fps=fps)
    return {k: (preprocess(v) if conditioned else v) for k, v in seqs.items()}


# --------------------------------------------------------------------------- #
# Conditioning
# --------------------------------------------------------------------------- #
def _interpolate_gaps(track: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Linearly fill invalid samples in a (T,) track; edges are held."""
    t = np.arange(track.shape[0])
    if valid.sum() == 0:
        return np.zeros_like(track)
    if valid.sum() == track.shape[0]:
        return track
    return np.interp(t, t[valid], track[valid])


def fill_missing(seq: PoseSequence, min_visibility: float = MIN_VISIBILITY) -> PoseSequence:
    """Interpolate across low-visibility or NaN samples, per landmark and axis."""
    xy = seq.xy.copy()
    valid = (seq.visibility >= min_visibility) & np.isfinite(seq.xy).all(axis=2)
    for l in range(xy.shape[1]):
        for axis in (0, 1):
            xy[:, l, axis] = _interpolate_gaps(xy[:, l, axis], valid[:, l])
    seq.xy = xy
    seq.meta["gap_fraction"] = float(1.0 - valid.mean())
    return seq


def smooth(
    seq: PoseSequence,
    window: int = SMOOTHING_WINDOW,
    polyorder: int = SMOOTHING_POLYORDER,
) -> PoseSequence:
    """Savitzky-Golay smoothing.

    Preferred over a moving average because it preserves the amplitude and
    timing of gait peaks, which the stride and rhythm features depend on.
    """
    t = seq.n_frames
    win = min(window, t if t % 2 == 1 else t - 1)
    if win < polyorder + 2:
        return seq
    seq.xy = savgol_filter(seq.xy, win, polyorder, axis=0)
    return seq


def normalise(seq: PoseSequence) -> PoseSequence:
    """Make coordinates translation- and scale-invariant.

    The pelvis midpoint becomes the origin and the torso length (pelvis to
    shoulder midpoint) becomes the unit, so a walk scores identically whether
    it was filmed from three metres or ten. The per-frame torso length is kept
    in ``meta['torso_scale']`` because its rate of change tells us how fast the
    subject is approaching the camera.
    """
    pelvis = seq.midpoint("left_hip", "right_hip")           # (T, 2)
    shoulder = seq.midpoint("left_shoulder", "right_shoulder")
    torso = np.linalg.norm(shoulder - pelvis, axis=1)        # (T,)

    median = float(np.nanmedian(torso))
    if not np.isfinite(median) or median < 1e-6:
        median = 1.0
    # Guard against per-frame collapse (e.g. a badly occluded frame).
    scale = np.where(torso > 0.2 * median, torso, median)

    seq.meta["torso_scale"] = scale
    seq.meta["torso_scale_median"] = median
    seq.meta["pelvis_raw"] = pelvis.copy()

    seq.xy = (seq.xy - pelvis[:, None, :]) / scale[:, None, None]
    return seq


def preprocess(
    seq: PoseSequence,
    min_visibility: float = MIN_VISIBILITY,
    window: int = SMOOTHING_WINDOW,
) -> PoseSequence:
    """Full conditioning chain: gap fill -> smooth -> normalise."""
    seq = fill_missing(seq, min_visibility=min_visibility)
    seq = smooth(seq, window=window)
    seq = normalise(seq)
    seq.meta["preprocessed"] = True
    return seq
