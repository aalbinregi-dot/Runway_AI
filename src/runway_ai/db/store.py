"""SQLite persistence for users, sessions and progress.

SQLite rather than a server database: the whole point is that a student can run
this on a laptop with no infrastructure. The schema is ordinary relational SQL,
so moving to Postgres or Firebase later is a connection-string change plus a
dialect pass, not a redesign.

Scores are stored both denormalised on the session row (fast dashboard reads)
and per-criterion in ``session_scores`` (so progress can be charted for any one
criterion without parsing JSON).
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from ..config import CRITERIA, PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "data" / "runway.db"

SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS users (
    user_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    email       TEXT    UNIQUE,
    height_cm   REAL,
    experience  TEXT    CHECK (experience IN ('beginner','intermediate','advanced','professional')),
    created_at  TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    video_id      TEXT    NOT NULL,
    source_path   TEXT,
    recorded_at   TEXT    NOT NULL,
    n_frames      INTEGER,
    fps           REAL,
    duration_s    REAL,
    overall_score REAL    NOT NULL,
    grade         TEXT,
    score_source  TEXT,                       -- 'model' or 'rubric'
    notes         TEXT
);
CREATE INDEX IF NOT EXISTS idx_sessions_user ON sessions(user_id, recorded_at);

CREATE TABLE IF NOT EXISTS session_scores (
    session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    criterion  TEXT    NOT NULL,
    score      REAL    NOT NULL,
    PRIMARY KEY (session_id, criterion)
);

CREATE TABLE IF NOT EXISTS session_features (
    session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    payload    TEXT    NOT NULL,              -- JSON: full feature vector
    PRIMARY KEY (session_id)
);

CREATE TABLE IF NOT EXISTS findings (
    finding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    criterion  TEXT    NOT NULL,
    feature    TEXT,
    measured   REAL,
    ideal      REAL,
    severity   REAL,
    message    TEXT
);
CREATE INDEX IF NOT EXISTS idx_findings_session ON findings(session_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path | str | None = None):
    path = Path(db_path or DB_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_user(name: str, email: str | None = None, height_cm: float | None = None,
                experience: str = "beginner", db_path=None) -> int:
    with connect(db_path) as c:
        cur = c.execute(
            "INSERT INTO users (name, email, height_cm, experience, created_at) "
            "VALUES (?,?,?,?,?)",
            (name, email, height_cm, experience, _now()),
        )
        return int(cur.lastrowid)


def find_user(name: str, db_path=None) -> int | None:
    """Look up a user by name without creating one.

    Read-only commands must use this rather than :func:`get_or_create_user`:
    reporting progress for a mistyped name should say so, not silently add an
    account to the database.
    """
    with connect(db_path) as c:
        row = c.execute("SELECT user_id FROM users WHERE name = ?", (name,)).fetchone()
    return int(row["user_id"]) if row else None


def get_or_create_user(name: str, db_path=None, **kw) -> int:
    existing = find_user(name, db_path=db_path)
    if existing is not None:
        return existing
    return create_user(name, db_path=db_path, **kw)


def save_session(user_id: int, video_id: str, assessment, feats: dict,
                 track: dict | None = None,
                 source_path: str | None = None, notes: str | None = None,
                 db_path=None) -> int:
    """Persist one analysed walk and return its session id."""
    with connect(db_path) as c:
        cur = c.execute(
            "INSERT INTO sessions (user_id, video_id, source_path, recorded_at, "
            "n_frames, fps, duration_s, overall_score, grade, score_source, notes) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (user_id, video_id, source_path, _now(),
             int(feats.get("qc_n_frames", 0)), float(feats.get("qc_fps", 0.0)),
             float(feats.get("qc_duration_s", 0.0)),
             float(assessment.overall), assessment.grade, assessment.source, notes),
        )
        sid = int(cur.lastrowid)

        c.executemany(
            "INSERT INTO session_scores (session_id, criterion, score) VALUES (?,?,?)",
            [(sid, k, float(v)) for k, v in assessment.scores.items()],
        )
        c.execute("INSERT INTO session_features (session_id, payload) VALUES (?,?)",
                  (sid, json.dumps({k: float(v) for k, v in feats.items()})))
        c.executemany(
            "INSERT INTO findings (session_id, criterion, feature, measured, ideal, "
            "severity, message) VALUES (?,?,?,?,?,?,?)",
            [(sid, f.criterion, f.feature, f.measured, f.ideal, f.severity, f.message)
             for f in assessment.findings],
        )

    # Written after the session row is committed, so the foreign key resolves.
    # A failure here must not lose the walk: the scores are the result, the
    # track only enables replay and comparison.
    if track:
        try:
            from . import tracks as track_store
            track_store.save(sid, track, db_path=db_path)
        except Exception:
            pass
    return sid


def user_history(user_id: int, db_path=None) -> list[dict]:
    """All sessions for a user, oldest first, with per-criterion scores."""
    with connect(db_path) as c:
        sessions = c.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY recorded_at, session_id",
            (user_id,),
        ).fetchall()
        out = []
        for s in sessions:
            scores = {
                r["criterion"]: r["score"] for r in c.execute(
                    "SELECT criterion, score FROM session_scores WHERE session_id = ?",
                    (s["session_id"],),
                )
            }
            out.append({**dict(s), "scores": scores})
        return out


def progress(user_id: int, db_path=None) -> dict:
    """Improvement between the first and most recent session.

    Deltas are reported per criterion so a user can see that, say, balance
    improved while rhythm regressed - an overall number alone hides exactly the
    information that should drive the next session's focus.
    """
    hist = user_history(user_id, db_path)
    if len(hist) < 2:
        return {"sessions": len(hist), "message": "Need at least two sessions to show progress."}

    first, last = hist[0], hist[-1]
    deltas = {
        c: round(last["scores"].get(c, 0.0) - first["scores"].get(c, 0.0), 2)
        for c in CRITERIA if c in last["scores"] and c in first["scores"]
    }
    improved = sorted((k for k in deltas if deltas[k] > 0), key=lambda k: -deltas[k])
    declined = sorted((k for k in deltas if deltas[k] < 0), key=lambda k: deltas[k])

    series = [{"recorded_at": h["recorded_at"], "overall": h["overall_score"],
               **h["scores"]} for h in hist]

    # Fitted through every walk, not just the two endpoints. The endpoint deltas
    # above are kept because they are what a user checks by eye, but the trend is
    # the estimate that should drive a decision - see `scoring.trend`.
    from ..scoring import trend as trend_mod
    trends = trend_mod.per_criterion(series, CRITERIA)
    overall_trend = trend_mod.overall([h["overall_score"] for h in hist])

    confirmed_up = sorted((k for k, t in trends.items() if t["direction"] == "improving"),
                          key=lambda k: -trends[k]["total_change"])
    confirmed_down = sorted((k for k, t in trends.items() if t["direction"] == "declining"),
                            key=lambda k: trends[k]["total_change"])

    return {
        "sessions": len(hist),
        "first_overall": round(first["overall_score"], 1),
        "latest_overall": round(last["overall_score"], 1),
        "overall_delta": round(last["overall_score"] - first["overall_score"], 1),
        "per_criterion_delta": deltas,
        "most_improved": improved[:3],
        "needs_attention": declined[:3],
        "series": series,
        "trends": trends,
        "overall_trend": overall_trend,
        # Separate from most_improved/needs_attention on purpose: those rank raw
        # endpoint deltas, which include changes indistinguishable from noise.
        # These two only contain criteria whose confidence interval clears zero.
        "confirmed_improving": confirmed_up,
        "confirmed_declining": confirmed_down,
        "min_walks_for_trend": trend_mod.MIN_WALKS,
    }


# --------------------------------------------------------------------------- #
# Personalized baselines (Phase 5)
# --------------------------------------------------------------------------- #
def get_user_baseline(
    user_id: int,
    criterion: str,
    n_sessions: int = 5,
    db_path=None,
) -> float | None:
    """Exponential moving average of the last ``n_sessions`` scores for one criterion.

    Returns ``None`` when the user has no history for that criterion.

    The EMA weights recent walks more heavily than old ones, so a fast improver
    quickly moves their baseline upward rather than being anchored to an early
    poor result. The smoothing factor alpha = 2 / (n + 1) is the standard
    span-based formula.
    """
    with connect(db_path) as c:
        rows = c.execute(
            "SELECT sc.score "
            "FROM session_scores sc "
            "JOIN sessions s ON s.session_id = sc.session_id "
            "WHERE s.user_id = ? AND sc.criterion = ? "
            "ORDER BY s.recorded_at DESC, s.session_id DESC "
            "LIMIT ?",
            (user_id, criterion, n_sessions),
        ).fetchall()

    if not rows:
        return None

    # Rows are newest-first; reverse for chronological EMA.
    scores = [float(r["score"]) for r in reversed(rows)]
    alpha = 2.0 / (len(scores) + 1)
    ema = scores[0]
    for s in scores[1:]:
        ema = alpha * s + (1.0 - alpha) * ema
    return round(ema, 3)


def get_historical_best(
    user_id: int,
    criterion: str,
    db_path=None,
) -> dict | None:
    """The session that achieved the highest score for one criterion.

    Returns a dict with ``score``, ``session_id``, and ``recorded_at``, or
    ``None`` if the user has no recorded sessions for this criterion.
    """
    with connect(db_path) as c:
        row = c.execute(
            "SELECT sc.score, s.session_id, s.recorded_at "
            "FROM session_scores sc "
            "JOIN sessions s ON s.session_id = sc.session_id "
            "WHERE s.user_id = ? AND sc.criterion = ? "
            "ORDER BY sc.score DESC, s.session_id DESC "
            "LIMIT 1",
            (user_id, criterion),
        ).fetchone()

    if row is None:
        return None
    return {
        "score":       round(float(row["score"]), 2),
        "session_id":  int(row["session_id"]),
        "recorded_at": row["recorded_at"],
    }
