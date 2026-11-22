"""Drill tracking: did the practice actually move the score?

The coaching plan already names what to work on and hands out drills. What it
could not do was close the loop — a user did the wall-alignment drill for two
weeks and the system had no idea, so the next walk's posture score arrived with
no reference to the effort that went into it.

This module records that a drill was taken on, freezes the criterion score at
that moment as a baseline, and reports what later walks measured against it.

The one thing it must not do is flatter the user
------------------------------------------------
Every criterion carries a model error of roughly 0.9/10, so two walks differing
by 0.4 is *nothing* — reporting that as "your drill is working" would be
inventing an effect out of noise. Outcomes are therefore gated on
:func:`runway_ai.scoring.significance.is_significant`, which uses the
per-criterion out-of-fold MAE measured by this project's own training run.
Below that threshold the honest answer is "no measurable change", and it is
worded so it cannot be misread as "no change".

Nor can it claim causality
--------------------------
A score that rises after a drill was practised is not proof the drill caused it.
There is no control condition here: the walk was filmed on a different day, at a
different time, possibly in different shoes. The wording throughout says what
was *measured* ("posture is up 1.4 since you started this drill"), never what
was caused. Weeks of consistent practice against a consistent trend is
suggestive, and that is as far as the data can carry it.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from ..feedback.coach import DRILLS
from ..scoring import significance

SCHEMA = """
CREATE TABLE IF NOT EXISTS drill_assignments (
    assignment_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
    session_id    INTEGER REFERENCES sessions(session_id) ON DELETE SET NULL,
    criterion     TEXT    NOT NULL,
    drill_name    TEXT    NOT NULL,
    baseline      REAL    NOT NULL,
    assigned_at   TEXT    NOT NULL,
    archived_at   TEXT
);
CREATE INDEX IF NOT EXISTS idx_drill_user ON drill_assignments(user_id, archived_at);

CREATE TABLE IF NOT EXISTS drill_practice (
    practice_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    assignment_id INTEGER NOT NULL REFERENCES drill_assignments(assignment_id)
                  ON DELETE CASCADE,
    practised_at  TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_practice_assignment
    ON drill_practice(assignment_id);
"""

#: One live assignment per drill, enforced by the database rather than by a
#: check in :func:`assign`.
#:
#: The check alone was a read-then-write race: two "take this on" clicks - a
#: double-click, or two open tabs - both saw no existing row and both inserted.
#: Six concurrent calls produced three live assignments, each with its own
#: baseline, and the drills page showed the same drill three times. Only the
#: database can make that impossible.
#:
#: Partial, so archiving a drill and taking it on again later still works;
#: without the WHERE clause a drill could only ever be tracked once.
UNIQUE_LIVE_DRILL = """
CREATE UNIQUE INDEX IF NOT EXISTS ux_drill_live
    ON drill_assignments(user_id, criterion, drill_name)
    WHERE archived_at IS NULL;
"""


class DrillError(Exception):
    """A drill request that cannot be honoured."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def migrate(db_path: Path | str | None = None) -> None:
    """Create the drill tables if they are missing.

    Purely additive - only ``CREATE ... IF NOT EXISTS`` - so it is idempotent
    and needs no version counter of its own. It deliberately does not touch
    ``PRAGMA user_version``, which :mod:`runway_ai.db.auth` owns.

    The order matters. A database written before the unique index existed may
    already hold duplicate live assignments, and ``CREATE UNIQUE INDEX`` fails
    outright against data that violates it. So duplicates are folded away
    first, then the constraint goes on.
    """
    from .store import connect

    with connect(db_path) as con:
        con.executescript(SCHEMA)
        _fold_duplicates(con)
        con.executescript(UNIQUE_LIVE_DRILL)


def _fold_duplicates(con) -> int:
    """Archive all but the earliest live assignment of each drill.

    The earliest is kept because it holds the baseline the user actually
    started from; keeping a later duplicate would silently move the goalposts.
    Practice rows are re-pointed at the survivor rather than discarded - the
    user did those sessions.
    """
    rows = con.execute(
        "SELECT MIN(assignment_id) AS keep, user_id, criterion, drill_name "
        "FROM drill_assignments WHERE archived_at IS NULL "
        "GROUP BY user_id, criterion, drill_name HAVING COUNT(*) > 1"
    ).fetchall()

    folded = 0
    for r in rows:
        dupes = [x[0] for x in con.execute(
            "SELECT assignment_id FROM drill_assignments "
            "WHERE archived_at IS NULL AND user_id = ? AND criterion = ? "
            "AND drill_name = ? AND assignment_id != ?",
            (r["user_id"], r["criterion"], r["drill_name"], r["keep"]))]
        for dupe in dupes:
            con.execute("UPDATE drill_practice SET assignment_id = ? "
                        "WHERE assignment_id = ?", (r["keep"], dupe))
            con.execute("DELETE FROM drill_assignments WHERE assignment_id = ?",
                        (dupe,))
            folded += 1
    return folded


def known_drill(criterion: str, drill_name: str) -> dict | None:
    """Look up a drill, so an assignment cannot name one that does not exist."""
    for d in DRILLS.get(criterion, []):
        if d.name == drill_name:
            return d.as_dict()
    return None


def assign(user_id: int, criterion: str, drill_name: str, baseline: float,
           session_id: int | None = None, db_path=None) -> int:
    """Take on a drill, freezing the current score as the baseline."""
    from .store import connect

    if known_drill(criterion, drill_name) is None:
        raise DrillError(f"no drill '{drill_name}' for {criterion}")

    import sqlite3

    migrate(db_path)
    with connect(db_path) as con:
        def live() -> int | None:
            row = con.execute(
                "SELECT assignment_id FROM drill_assignments "
                "WHERE user_id = ? AND criterion = ? AND drill_name = ? "
                "AND archived_at IS NULL",
                (user_id, criterion, drill_name)).fetchone()
            return int(row[0]) if row else None

        # Taking the same drill on twice must not reset the baseline - that
        # would discard the progress measured so far. The check is the fast
        # path; ``ux_drill_live`` is what actually guarantees it, because two
        # concurrent callers can both pass this check.
        existing = live()
        if existing is not None:
            return existing

        try:
            cur = con.execute(
                "INSERT INTO drill_assignments "
                "(user_id, session_id, criterion, drill_name, baseline, assigned_at) "
                "VALUES (?,?,?,?,?,?)",
                (user_id, session_id, criterion, drill_name, float(baseline), _now()))
            return int(cur.lastrowid)
        except sqlite3.IntegrityError:
            # Lost the race. The other caller's assignment is the real one, and
            # its baseline is the one to keep.
            won = live()
            if won is None:
                raise
            return won


def log_practice(user_id: int, assignment_id: int, db_path=None) -> int:
    """Record one practice session. Returns the new total."""
    from .store import connect

    migrate(db_path)
    with connect(db_path) as con:
        owner = con.execute(
            "SELECT user_id FROM drill_assignments WHERE assignment_id = ?",
            (assignment_id,)).fetchone()
        # Ownership is re-checked here rather than trusted from the route, so a
        # missing row and another user's row are indistinguishable to the caller.
        if owner is None or owner[0] != user_id:
            raise DrillError("no such drill")
        con.execute(
            "INSERT INTO drill_practice (assignment_id, practised_at) VALUES (?,?)",
            (assignment_id, _now()))
        return int(con.execute(
            "SELECT COUNT(*) FROM drill_practice WHERE assignment_id = ?",
            (assignment_id,)).fetchone()[0])


def archive(user_id: int, assignment_id: int, db_path=None) -> bool:
    """Stop tracking a drill, keeping its history."""
    from .store import connect

    migrate(db_path)
    with connect(db_path) as con:
        cur = con.execute(
            "UPDATE drill_assignments SET archived_at = ? "
            "WHERE assignment_id = ? AND user_id = ? AND archived_at IS NULL",
            (_now(), assignment_id, user_id))
        return cur.rowcount > 0


def _latest_score(con, user_id: int, criterion: str, after: str):
    """This criterion's score in the most recent walk since ``after``.

    ``recorded_at`` has one-second resolution, so two walks analysed back to
    back tie. Ordering on it alone leaves which row wins up to SQLite, and the
    consequence would be an outcome measured against the wrong walk. The
    session id breaks the tie exactly, since it is monotonic - the same
    tiebreaker every other ordering query in the project already uses.
    """
    return con.execute(
        "SELECT sc.score, s.recorded_at, s.session_id "
        "FROM session_scores sc JOIN sessions s ON s.session_id = sc.session_id "
        "WHERE s.user_id = ? AND sc.criterion = ? AND s.recorded_at > ? "
        "ORDER BY s.recorded_at DESC, s.session_id DESC LIMIT 1",
        (user_id, criterion, after)).fetchone()


def _walks_since(con, user_id: int, after: str) -> int:
    return int(con.execute(
        "SELECT COUNT(*) FROM sessions WHERE user_id = ? AND recorded_at > ?",
        (user_id, after)).fetchone()[0])


def active(user_id: int, db_path=None) -> list[dict]:
    """Live drills, each with whatever the walks since have measured."""
    from .store import connect

    migrate(db_path)
    with connect(db_path) as con:
        rows = con.execute(
            "SELECT * FROM drill_assignments "
            "WHERE user_id = ? AND archived_at IS NULL ORDER BY assigned_at DESC",
            (user_id,)).fetchall()

        out = []
        for r in rows:
            sessions = con.execute(
                "SELECT COUNT(*) FROM drill_practice WHERE assignment_id = ?",
                (r["assignment_id"],)).fetchone()[0]
            latest = _latest_score(con, user_id, r["criterion"], r["assigned_at"])
            meta = known_drill(r["criterion"], r["drill_name"]) or {}

            entry = {
                "assignment_id": r["assignment_id"],
                "criterion": r["criterion"],
                "drill_name": r["drill_name"],
                "instruction": meta.get("instruction"),
                "duration": meta.get("duration"),
                "baseline": round(r["baseline"], 2),
                "assigned_at": r["assigned_at"],
                "practice_sessions": int(sessions),
                "walks_since": _walks_since(con, user_id, r["assigned_at"]),
                "threshold": significance.threshold(r["criterion"]),
                **_outcome(r["criterion"], r["baseline"], latest),
            }
            out.append(entry)
        return out


def _outcome(criterion: str, baseline: float, latest) -> dict:
    """Turn a baseline and a later score into a verdict the user can trust."""
    if latest is None:
        return {
            "status": "awaiting_walk",
            "latest": None,
            "delta": None,
            "summary": "No walk recorded since you took this on. "
                       "Film one to measure the effect.",
        }

    delta = float(latest["score"]) - float(baseline)
    thresh = significance.threshold(criterion)

    if not significance.is_significant(criterion, delta):
        status, summary = "no_change", (
            f"Moved {delta:+.1f}, which is inside this criterion's measurement "
            f"error of ±{thresh:.1f}. That means no change could be *detected* — "
            f"not that nothing changed. Keep going and film again."
        )
    elif delta > 0:
        status, summary = "improved", (
            f"Up {delta:+.1f}, past the ±{thresh:.1f} needed to be sure it is "
            f"real rather than measurement noise."
        )
    else:
        status, summary = "declined", (
            f"Down {delta:+.1f}, past the ±{thresh:.1f} measurement error. Worth "
            f"checking the drill is being done as described, and that the two "
            f"walks were filmed from a similar angle."
        )

    return {
        "status": status,
        "latest": round(float(latest["score"]), 2),
        "latest_session_id": latest["session_id"],
        "latest_at": latest["recorded_at"],
        "delta": round(delta, 2),
        "summary": summary,
    }


def history(user_id: int, db_path=None) -> list[dict]:
    """Archived drills, for the record."""
    from .store import connect

    migrate(db_path)
    with connect(db_path) as con:
        rows = con.execute(
            "SELECT a.*, (SELECT COUNT(*) FROM drill_practice p "
            "             WHERE p.assignment_id = a.assignment_id) AS sessions "
            "FROM drill_assignments a "
            "WHERE a.user_id = ? AND a.archived_at IS NOT NULL "
            "ORDER BY a.archived_at DESC", (user_id,)).fetchall()
        return [{
            "assignment_id": r["assignment_id"],
            "criterion": r["criterion"],
            "drill_name": r["drill_name"],
            "baseline": round(r["baseline"], 2),
            "assigned_at": r["assigned_at"],
            "archived_at": r["archived_at"],
            "practice_sessions": int(r["sessions"]),
        } for r in rows]


# --------------------------------------------------------------------------- #
# Adaptive drill prioritization (Phase 5)
# --------------------------------------------------------------------------- #
def assign_drills(
    user_id: int,
    assessment,
    baseline: dict | None = None,
    n_recent: int = 3,
    max_drills: int = 3,
    db_path=None,
) -> list[dict]:
    """Return an ordered list of drills tailored to this user's recent history.

    Prioritization rules
    --------------------
    Each criterion with a finding is given a priority weight. Higher weight →
    recommended sooner.

    1. **Persistent weakness**: criterion was below FAULT_SCORE in the last
       ``n_recent`` sessions → weight multiplied by 1.5.
    2. **Rate of improvement**: criterion has already improved past the
       significance threshold since the user's last assignment → weight divided
       by 2 (criterion is resolving itself; steer attention elsewhere).
    3. **Severity**: base weight is the finding's own severity score from
       ``assess.diagnose``.

    When ``baseline`` is ``None`` (new user, no history) the plain
    severity-ranked list from the assessment is returned unchanged — the same
    behaviour as before this upgrade.

    Parameters
    ----------
    user_id:
        Database user ID.
    assessment:
        ``Assessment`` object from ``assess.assess()``.
    baseline:
        Dict mapping criterion → EMA score, from
        ``store.get_user_baseline()``. Pass ``None`` for new users.
    n_recent:
        How many past sessions to look back when checking persistence.
    max_drills:
        Cap on recommendations per call.
    """
    from .store import connect
    from ..scoring import significance

    # Build a set of criteria that have already resolved since the last
    # assignment — improvement past the significance threshold.
    resolved: set[str] = set()
    if baseline:
        for crit, base_score in baseline.items():
            current = assessment.scores.get(crit)
            if current is not None:
                delta = current - base_score
                if significance.is_significant(crit, delta) and delta > 0:
                    resolved.add(crit)

    # Check persistence: which weak criteria have been weak across recent walks.
    persistent: set[str] = set()
    if baseline:
        migrate(db_path)
        with connect(db_path) as con:
            for crit in assessment.scores:
                rows = con.execute(
                    "SELECT sc.score "
                    "FROM session_scores sc "
                    "JOIN sessions s ON s.session_id = sc.session_id "
                    "WHERE s.user_id = ? AND sc.criterion = ? "
                    "ORDER BY s.recorded_at DESC, s.session_id DESC "
                    "LIMIT ?",
                    (user_id, crit, n_recent),
                ).fetchall()
                if len(rows) >= n_recent:
                    from ..scoring.assess import FAULT_SCORE
                    if all(float(r["score"]) < FAULT_SCORE for r in rows):
                        persistent.add(crit)

    # Score every finding.
    scored_findings = []
    for finding in assessment.findings:
        crit = finding.criterion
        weight = finding.severity

        if crit in persistent:
            weight *= 1.5
        if crit in resolved:
            weight *= 0.5  # Already improving; let the user finish the drill.

        scored_findings.append((weight, finding))

    scored_findings.sort(key=lambda t: -t[0])

    # Pick top drills, one per criterion, from the DRILLS catalogue.
    seen_criteria: set[str] = set()
    recommendations = []
    for _, finding in scored_findings:
        if len(recommendations) >= max_drills:
            break
        crit = finding.criterion
        if crit in seen_criteria:
            continue
        drills_for_crit = DRILLS.get(crit, [])
        if drills_for_crit:
            drill = drills_for_crit[0]
            recommendations.append({
                "criterion": crit,
                "drill_name": drill.name,
                "instruction": drill.as_dict().get("instruction"),
                "duration": drill.as_dict().get("duration"),
                "finding_message": finding.message,
                "severity": round(finding.severity, 4),
                "is_persistent": crit in persistent,
                "is_resolving": crit in resolved,
            })
            seen_criteria.add(crit)

    return recommendations
