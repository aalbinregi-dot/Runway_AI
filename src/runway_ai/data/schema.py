"""Canonical keypoint representation and conversions.

Two equivalent views of the same data are used throughout:

* **long**   - one row per (video_id, frame, landmark); the format of the
  supplied workbook and the format written to CSV.
* **tensor** - a dense ``PoseSequence`` with ``xy`` of shape (T, L, 2),
  ``z`` (T, L) and ``visibility`` (T, L); what the feature code consumes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..config import CORE_13, KEYPOINT_COLUMNS, DEFAULT_FPS


class SchemaError(ValueError):
    """Raised when an input table does not satisfy the keypoint schema."""


@dataclass
class PoseSequence:
    """A dense pose track for a single clip.

    Attributes
    ----------
    video_id: clip identifier.
    landmarks: ordered landmark names, length ``L``.
    xy: float array (T, L, 2), image-normalised coordinates. ``y`` grows
        downward, matching MediaPipe and the sample workbook.
    z: float array (T, L), relative depth (may be all zeros).
    visibility: float array (T, L) in [0, 1].
    fps: frames per second of the source clip.
    """

    video_id: str
    landmarks: list[str]
    xy: np.ndarray
    z: np.ndarray
    visibility: np.ndarray
    fps: float = DEFAULT_FPS
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        t, l = self.xy.shape[0], self.xy.shape[1]
        if self.xy.shape != (t, l, 2):
            raise SchemaError(f"xy must be (T, L, 2), got {self.xy.shape}")
        if self.z.shape != (t, l) or self.visibility.shape != (t, l):
            raise SchemaError("z and visibility must both be (T, L)")
        if len(self.landmarks) != l:
            raise SchemaError(
                f"{len(self.landmarks)} landmark names for {l} columns"
            )

    # -- convenience ------------------------------------------------------- #
    @property
    def n_frames(self) -> int:
        return self.xy.shape[0]

    @property
    def duration_s(self) -> float:
        return self.n_frames / self.fps

    def index(self, name: str) -> int:
        try:
            return self.landmarks.index(name)
        except ValueError as exc:
            raise SchemaError(f"landmark {name!r} not in sequence") from exc

    def point(self, name: str) -> np.ndarray:
        """Return the (T, 2) trajectory of one landmark."""
        return self.xy[:, self.index(name), :]

    def vis(self, name: str) -> np.ndarray:
        """Return the (T,) visibility track of one landmark."""
        return self.visibility[:, self.index(name)]

    def midpoint(self, a: str, b: str) -> np.ndarray:
        """Return the (T, 2) midpoint trajectory between two landmarks."""
        return 0.5 * (self.point(a) + self.point(b))

    def to_long(self) -> pd.DataFrame:
        """Flatten back to the canonical long format."""
        t, l = self.xy.shape[:2]
        frames = np.repeat(np.arange(1, t + 1), l)
        names = np.tile(np.asarray(self.landmarks, dtype=object), t)
        return pd.DataFrame({
            "video_id": self.video_id,
            "frame": frames,
            "landmark": names,
            "x": self.xy[:, :, 0].ravel(),
            "y": self.xy[:, :, 1].ravel(),
            "z": self.z.ravel(),
            "visibility": self.visibility.ravel(),
        })[KEYPOINT_COLUMNS]


def validate_long(df: pd.DataFrame) -> pd.DataFrame:
    """Check and normalise a long-format keypoint table."""
    missing = [c for c in KEYPOINT_COLUMNS if c not in df.columns]
    if missing:
        raise SchemaError(f"missing required columns: {missing}")
    # Build a fresh frame rather than coercing columns in place: assigning into
    # a slice of the caller's DataFrame trips pandas' copy-on-write warnings and,
    # from pandas 3.0, would silently fail to write back.
    out = pd.DataFrame({
        "video_id": df["video_id"].astype(str),
        "frame": pd.to_numeric(df["frame"], errors="raise").astype(int),
        "landmark": df["landmark"].astype(str).str.strip().str.lower(),
        "x": pd.to_numeric(df["x"], errors="coerce").astype(float),
        "y": pd.to_numeric(df["y"], errors="coerce").astype(float),
        "z": pd.to_numeric(df["z"], errors="coerce").astype(float),
        "visibility": pd.to_numeric(df["visibility"], errors="coerce").astype(float),
    })[KEYPOINT_COLUMNS].reset_index(drop=True)
    dupes = out.duplicated(["video_id", "frame", "landmark"]).sum()
    if dupes:
        raise SchemaError(f"{dupes} duplicate (video_id, frame, landmark) rows")
    return out


def long_to_sequences(
    df: pd.DataFrame,
    landmarks: list[str] | None = None,
    fps: float = DEFAULT_FPS,
) -> dict[str, PoseSequence]:
    """Convert a long table into one ``PoseSequence`` per ``video_id``.

    Landmarks absent from a clip are filled with NaN and zero visibility so
    that downstream code sees a consistent tensor shape.
    """
    df = validate_long(df)
    landmarks = list(landmarks) if landmarks else list(CORE_13)
    out: dict[str, PoseSequence] = {}

    for video_id, g in df.groupby("video_id", sort=True):
        frames = np.sort(g["frame"].unique())
        f_idx = {f: i for i, f in enumerate(frames)}
        l_idx = {name: i for i, name in enumerate(landmarks)}
        t, l = len(frames), len(landmarks)

        xy = np.full((t, l, 2), np.nan)
        z = np.zeros((t, l))
        vis = np.zeros((t, l))

        known = g[g["landmark"].isin(l_idx)]
        rows = known["frame"].map(f_idx).to_numpy()
        cols = known["landmark"].map(l_idx).to_numpy()
        xy[rows, cols, 0] = known["x"].to_numpy()
        xy[rows, cols, 1] = known["y"].to_numpy()
        z[rows, cols] = known["z"].to_numpy()
        vis[rows, cols] = known["visibility"].to_numpy()

        out[str(video_id)] = PoseSequence(
            video_id=str(video_id),
            landmarks=landmarks,
            xy=xy, z=z, visibility=vis, fps=fps,
            meta={"source_frames": frames.tolist()},
        )
    return out


def sequences_to_long(seqs: dict[str, PoseSequence] | list[PoseSequence]) -> pd.DataFrame:
    """Concatenate sequences back into one long table."""
    items = seqs.values() if isinstance(seqs, dict) else seqs
    return pd.concat([s.to_long() for s in items], ignore_index=True)
