"""History, comparison and PDF export.

Every route here is scoped to the signed-in account and re-checks ownership on
each session id. Reading a walk by id without that check would be an IDOR: ids
are sequential integers, so guessing another user's is trivial.

A stored walk is replayed from its **keypoint track**, not from video - no
footage is ever written to disk. That means a past walk is shown as a skeleton
on a blank canvas. It is less rich than the original overlay, and the UI says so
rather than letting a stick figure imply the footage was kept.
"""
from __future__ import annotations

import sys
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from auth_routes import current_user                       # noqa: E402
from runway_ai.config import (CRITERIA, CRITERION_LABELS,  # noqa: E402
                              CRITERION_WEIGHTS)
from pydantic import BaseModel                             # noqa: E402
from runway_ai.db import auth, drills, store                # noqa: E402
from runway_ai.db import tracks as track_store             # noqa: E402
from runway_ai.scoring import significance                 # noqa: E402

router = APIRouter()


def _own_session(session_id: int, user: auth.User) -> dict:
    """Fetch a session, or 404 if it is not this user's.

    404 rather than 403 on purpose: telling a caller that a session exists but
    belongs to somebody else is itself a disclosure.
    """
    with store.connect() as con:
        row = con.execute(
            "SELECT * FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user.id)).fetchone()
    if row is None:
        raise HTTPException(404, "No such walk.")
    return dict(row)


def _quality(session_id: int) -> dict:
    """Recording-quality warnings and out-of-distribution status for a walk.

    Both were computed at analysis time but only ever shown on the result page.
    A walk flagged "1.7s clip, fewer than four steps" then reappeared in history
    as a bare number, which reads as far more solid than it is.
    """
    import json as _json

    from runway_ai.scoring.assess import check_quality
    from runway_ai.scoring.coverage import check as coverage_check

    with store.connect() as con:
        row = con.execute("SELECT payload FROM session_features WHERE session_id = ?",
                          (session_id,)).fetchone()
    if row is None:
        return {"warnings": [], "coverage": "unknown"}
    try:
        feats = _json.loads(row["payload"])
        return {"warnings": check_quality(feats),
                "coverage": coverage_check(feats).get("overall", "unknown")}
    except Exception:
        return {"warnings": [], "coverage": "unknown"}


def _scores(session_id: int) -> dict[str, float]:
    with store.connect() as con:
        rows = con.execute(
            "SELECT criterion, score FROM session_scores WHERE session_id = ?",
            (session_id,)).fetchall()
    return {r["criterion"]: r["score"] for r in rows}


def _findings(session_id: int) -> list[dict]:
    with store.connect() as con:
        rows = con.execute(
            "SELECT criterion, feature, measured, ideal, severity, message "
            "FROM findings WHERE session_id = ? ORDER BY severity DESC",
            (session_id,)).fetchall()
    return [dict(r) for r in rows]


@router.get("/me/walks")
def list_walks(user: auth.User = Depends(current_user)) -> dict:
    """Every saved walk, newest first."""
    with store.connect() as con:
        con.execute(track_store.TRACK_SCHEMA)
        rows = con.execute(
            "SELECT s.session_id, s.video_id, s.recorded_at, s.overall_score, "
            "       s.grade, s.duration_s, s.n_frames, s.score_source, "
            "       (SELECT 1 FROM session_tracks t "
            "          WHERE t.session_id = s.session_id) AS has_track "
            "FROM sessions s WHERE s.user_id = ? "
            "ORDER BY s.recorded_at DESC, s.session_id DESC", (user.id,)).fetchall()

    walks = []
    for r in rows:
        q = _quality(r["session_id"])
        walks.append(dict(r) | {"has_track": bool(r["has_track"]),
                                "warnings": len(q["warnings"]),
                                "coverage": q["coverage"]})
    scores = [w["overall_score"] for w in walks]
    return {
        "count": len(walks),
        "walks": walks,
        "best": max(scores) if scores else None,
        "average": round(sum(scores) / len(scores), 1) if scores else None,
    }


@router.get("/me/walks/{session_id}")
def get_walk(session_id: int, user: auth.User = Depends(current_user)) -> dict:
    """One walk in full, with its track and a freshly recomputed timeline.

    The timeline is derived rather than stored, so an old walk is always
    measured by the current feature code instead of whatever version was
    running the day it was uploaded.
    """
    row = _own_session(session_id, user)
    out = {
        "session": row,
        "quality": _quality(session_id),
        "scores": _scores(session_id),
        "findings": _findings(session_id),
        "track": track_store.load(session_id),
        "timeline": None,
        "breakdown": None,
        "measurements": None,
    }

    seq = track_store.load_sequence(session_id)
    if seq is not None:
        try:
            from runway_ai.data.loaders import preprocess
            from runway_ai.features import timeline as tl
            from runway_ai.features.builder import extract_features
            from runway_ai.scoring import explain
            from runway_ai.scoring.assess import assess

            prepared = preprocess(seq)
            feats = extract_features(prepared)
            a = assess(feats)
            out["timeline"] = tl.build(prepared, a.findings)
            # The bands are rebuilt against the *stored* scores, not against the
            # rubric scores just computed: the saved walk's number is what the
            # page shows, so its explanation has to be keyed to that number.
            out["breakdown"] = explain.breakdown(feats, out["scores"])
            out["measurements"] = a.measurements
        except Exception:
            out["timeline"] = None
    return out


@router.delete("/me/walks/{session_id}")
def delete_walk(session_id: int, user: auth.User = Depends(current_user)) -> dict:
    """Delete a walk. Scores, features, findings and the track all go with it."""
    _own_session(session_id, user)
    with store.connect() as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    return {"deleted": True}


@router.get("/me/compare")
def compare(a: int, b: int, user: auth.User = Depends(current_user)) -> dict:
    """Two walks side by side.

    Only per-criterion scores and findings are compared. The measurement traces
    are deliberately *not* aligned across walks: two clips have different
    durations, different numbers of gait cycles and possibly different camera
    distances, so overlaying their traces would invite reading a difference that
    is an artefact of the recording rather than of the walking.
    """
    left, right = _own_session(a, user), _own_session(b, user)
    sa, sb = _scores(a), _scores(b)

    deltas = []
    for c in CRITERIA:
        if c in sa and c in sb:
            d = sb[c] - sa[c]
            deltas.append({
                "criterion": c,
                "label": CRITERION_LABELS[c],
                "a": round(sa[c], 1),
                "b": round(sb[c], 1),
                "delta": round(d, 1),
                # A difference smaller than the model's own error on that
                # criterion is not evidence of change. Flagged rather than
                # hidden, so the number is still visible but not mistaken for
                # a result.
                "significant": significance.is_significant(c, d),
                "noise_floor": significance.threshold(c),
            })
    # Only a significant move can be headlined.
    ranked = sorted((d for d in deltas if d["significant"]),
                    key=lambda d: d["delta"])

    # A caveat the numbers cannot carry themselves.
    notes = []
    if left["duration_s"] and right["duration_s"]:
        ratio = max(left["duration_s"], right["duration_s"]) / max(
            min(left["duration_s"], right["duration_s"]), 1e-6)
        if ratio > 1.6:
            notes.append(
                f"These clips differ a lot in length "
                f"({left['duration_s']:.1f}s vs {right['duration_s']:.1f}s). "
                "Timing and symmetry scores are less comparable across very "
                "different durations.")
    if left["score_source"] != right["score_source"]:
        notes.append("These walks were scored by different methods "
                     f"({left['score_source']} vs {right['score_source']}).")

    overall_delta = right["overall_score"] - left["overall_score"]
    overall_floor = significance.overall_threshold()

    if not any(d["significant"] for d in deltas):
        notes.append(
            "No criterion changed by more than the model's own margin of error, "
            "so this pair does not show a measurable difference.")

    return {
        "a": {"session": left, "scores": sa, "track": track_store.load(a),
              "quality": _quality(a)},
        "b": {"session": right, "scores": sb, "track": track_store.load(b),
              "quality": _quality(b)},
        "deltas": deltas,
        "overall_delta": round(overall_delta, 1),
        "overall_significant": abs(overall_delta) >= overall_floor,
        "overall_noise_floor": overall_floor,
        "most_improved": ranked[-1] if ranked and ranked[-1]["delta"] > 0 else None,
        "most_regressed": ranked[0] if ranked and ranked[0]["delta"] < 0 else None,
        "notes": notes,
    }


@router.get("/me/walks/{session_id}/report.pdf")
def walk_pdf(session_id: int, user: auth.User = Depends(current_user)) -> Response:
    """A printable assessment for this walk."""
    from runway_ai.feedback import pdf as pdf_report

    row = _own_session(session_id, user)
    data = pdf_report.build(
        session=row,
        scores=_scores(session_id),
        findings=_findings(session_id),
        track=track_store.load(session_id),
        user_name=user.name,
        quality=_quality(session_id),
    )
    stamp = str(row["recorded_at"])[:10]
    return Response(
        data,
        media_type="application/pdf",
        headers={"Content-Disposition":
                 f'attachment; filename="runway-{stamp}-{session_id}.pdf"'},
    )


# ---------------------------------------------------------------------------
# Drill tracking
#
# The coaching plan hands out drills; these routes record that one was taken on
# and report what later walks measured against the score it started from. The
# verdict is gated on the per-criterion measurement error, so a drill is never
# credited with a change too small to distinguish from noise - see
# runway_ai.db.drills.
# ---------------------------------------------------------------------------

class DrillRequest(BaseModel):
    criterion: str
    drill_name: str
    session_id: int | None = None


@router.get("/me/drills")
def list_drills(user: auth.User = Depends(current_user)) -> dict:
    """Live drills with their measured outcome, plus the archived ones."""
    return {
        "active": drills.active(user.id),
        "archived": drills.history(user.id),
    }


@router.post("/me/drills")
def take_on_drill(req: DrillRequest,
                  user: auth.User = Depends(current_user)) -> dict:
    """Take on a drill, freezing the current criterion score as the baseline.

    The baseline is read from the user's own most recent walk rather than
    accepted from the request: a client-supplied baseline would let the number
    every future outcome is measured against be set to anything.
    """
    if req.criterion not in CRITERIA:
        raise HTTPException(422, f"unknown criterion '{req.criterion}'")

    with store.connect() as con:
        row = con.execute(
            "SELECT sc.score, s.session_id FROM session_scores sc "
            "JOIN sessions s ON s.session_id = sc.session_id "
            "WHERE s.user_id = ? AND sc.criterion = ? "
            "ORDER BY s.recorded_at DESC, s.session_id DESC LIMIT 1",
            (user.id, req.criterion)).fetchone()
    if row is None:
        raise HTTPException(
            409, "Analyse a walk first — a drill needs a starting score to "
                 "measure against.")

    try:
        assignment_id = drills.assign(
            user.id, req.criterion, req.drill_name, float(row["score"]),
            session_id=req.session_id or row["session_id"])
    except drills.DrillError as exc:
        raise HTTPException(422, str(exc)) from None

    return {"assignment_id": assignment_id,
            "baseline": round(float(row["score"]), 2),
            "threshold": significance.threshold(req.criterion)}


@router.post("/me/drills/{assignment_id}/practice")
def log_practice(assignment_id: int,
                 user: auth.User = Depends(current_user)) -> dict:
    """Record one practice session."""
    try:
        total = drills.log_practice(user.id, assignment_id)
    except drills.DrillError:
        # 404 rather than 403: another user's assignment must be
        # indistinguishable from one that does not exist.
        raise HTTPException(404, "no such drill") from None
    return {"assignment_id": assignment_id, "practice_sessions": total}


@router.delete("/me/drills/{assignment_id}")
def stop_drill(assignment_id: int,
               user: auth.User = Depends(current_user)) -> dict:
    """Stop tracking a drill. Its history is kept."""
    if not drills.archive(user.id, assignment_id):
        raise HTTPException(404, "no such drill")
    return {"archived": assignment_id}
