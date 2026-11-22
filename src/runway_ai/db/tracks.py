"""Persist the keypoint track of a saved walk.

Why store this at all
---------------------
Until now a saved session kept only its scores, features and findings, so a past
walk could not be replayed or compared - the shape of the movement was thrown
away the moment the request ended. History and side-by-side comparison are
impossible without it.

Storing keypoints is consistent with, not contrary to, this project's privacy
position. That position is "keypoints, not video": the pixels are the
identifying part, and they are still never written to disk. A stick figure is
far less exposing than footage of somebody's body, and it is what the analysis
actually operates on.

Encoding
--------
Measured on a real 369-frame, 13-landmark track:

======================  ========  ==============================
form                    size      worst error at 1080p
======================  ========  ==============================
JSON, 3 decimals        101.3 KB  --
float16 blob             23.4 KB  0.26 px
int16 at 1e-4 + gzip     ~8 KB    0.05 px
======================  ========  ==============================

int16 is chosen over float16 at identical width because float16's precision
degrades exactly where the data lives: near 1.0 its spacing is ~0.001, about a
pixel at 1080p, whereas fixed-point 1e-4 is uniform across the range.
"""
from __future__ import annotations

import gzip
import json

import numpy as np

from .store import connect

TRACK_SCHEMA = """
CREATE TABLE IF NOT EXISTS session_tracks (
    session_id INTEGER PRIMARY KEY REFERENCES sessions(session_id) ON DELETE CASCADE,
    landmarks  TEXT    NOT NULL,
    n_frames   INTEGER NOT NULL,
    fps        REAL    NOT NULL,
    xy         BLOB    NOT NULL,
    visibility BLOB    NOT NULL
)
"""

#: Fixed-point scale. int16 spans +/-3.2767 at this scale, which comfortably
#: covers normalised image coordinates even when a landmark is predicted just
#: outside the frame.
SCALE = 10000


def encode(xy: np.ndarray, visibility: np.ndarray) -> tuple[bytes, bytes]:
    q = np.round(np.clip(np.nan_to_num(xy, nan=0.0), -3.2, 3.2) * SCALE).astype(np.int16)
    v = np.round(np.clip(np.nan_to_num(visibility, nan=0.0), 0, 1) * 255).astype(np.uint8)
    return gzip.compress(q.tobytes(), 6), gzip.compress(v.tobytes(), 6)


def decode(xy_blob: bytes, vis_blob: bytes, n_frames: int, n_landmarks: int):
    xy = (np.frombuffer(gzip.decompress(xy_blob), dtype=np.int16)
          .reshape(n_frames, n_landmarks, 2).astype(np.float32) / SCALE)
    vis = (np.frombuffer(gzip.decompress(vis_blob), dtype=np.uint8)
           .reshape(n_frames, n_landmarks).astype(np.float32) / 255.0)
    return xy, vis


def save(session_id: int, track: dict, db_path=None) -> int:
    """Store the ``track`` block an analysis produced. Returns bytes written."""
    xy = np.asarray(track["xy"], dtype=np.float32)
    vis = np.asarray(track["visibility"], dtype=np.float32)
    xy_blob, vis_blob = encode(xy, vis)
    with connect(db_path) as con:
        con.execute(TRACK_SCHEMA)
        con.execute(
            "INSERT INTO session_tracks (session_id, landmarks, n_frames, fps, "
            "xy, visibility) VALUES (?,?,?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET landmarks=excluded.landmarks, "
            "n_frames=excluded.n_frames, fps=excluded.fps, xy=excluded.xy, "
            "visibility=excluded.visibility",
            (session_id, json.dumps(track["landmarks"]), int(xy.shape[0]),
             float(track.get("fps", 30.0)), xy_blob, vis_blob))
    return len(xy_blob) + len(vis_blob)


def load(session_id: int, db_path=None) -> dict | None:
    """Return a track in the same shape the analysis endpoint emits."""
    with connect(db_path) as con:
        con.execute(TRACK_SCHEMA)
        row = con.execute(
            "SELECT landmarks, n_frames, fps, xy, visibility FROM session_tracks "
            "WHERE session_id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    names = json.loads(row["landmarks"])
    xy, vis = decode(row["xy"], row["visibility"], row["n_frames"], len(names))
    return {
        "landmarks": names,
        "fps": float(row["fps"]),
        "n_frames": int(row["n_frames"]),
        "xy": np.round(xy, 4).tolist(),
        "visibility": np.round(vis, 2).tolist(),
    }


def load_sequence(session_id: int, db_path=None):
    """Rebuild a ``PoseSequence`` so a stored walk can be re-measured.

    The timeline and features are deliberately *not* persisted. They are
    derivable from the track, and recomputing them means a stored walk always
    reflects the current feature code rather than whatever version happened to
    be running the day it was uploaded.
    """
    from ..data.schema import PoseSequence

    with connect(db_path) as con:
        con.execute(TRACK_SCHEMA)
        row = con.execute(
            "SELECT landmarks, n_frames, fps, xy, visibility FROM session_tracks "
            "WHERE session_id = ?", (session_id,)).fetchone()
    if row is None:
        return None
    names = json.loads(row["landmarks"])
    xy, vis = decode(row["xy"], row["visibility"], row["n_frames"], len(names))
    return PoseSequence(
        video_id=f"session-{session_id}",
        landmarks=names,
        xy=xy.astype(np.float64),
        z=np.zeros(vis.shape),
        visibility=vis.astype(np.float64),
        fps=float(row["fps"]),
    )


def delete(session_id: int, db_path=None) -> bool:
    with connect(db_path) as con:
        con.execute(TRACK_SCHEMA)
        return con.execute("DELETE FROM session_tracks WHERE session_id = ?",
                           (session_id,)).rowcount > 0
