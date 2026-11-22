"""Regressions for bugs found by adversarial probing rather than by feature tests.

Each of these reproduced before its fix. They are grouped here because what they
share is how they were found: degenerate inputs, concurrent callers, and
comparisons that quietly do the wrong thing rather than raising.
"""
from __future__ import annotations

import threading

import numpy as np
import pytest

from runway_ai.analyze import _match_template, analyze_sequence
from runway_ai.config import CORE_13
from runway_ai.data.schema import PoseSequence
from runway_ai.db import drills, store
from runway_ai.features.builder import MIN_ANALYSIS_FRAMES, extract_features
from runway_ai.pose import multi
from runway_ai.scoring.assess import UnusableRecording


def _seq(n_frames: int, fill=None, gaps: float = 0.0) -> PoseSequence:
    rng = np.random.default_rng(0)
    xy = (rng.normal(0.5, 0.03, (n_frames, 13, 2)) if fill is None
          else np.full((n_frames, 13, 2), fill))
    if gaps:
        flat = xy.reshape(-1, 2)
        idx = rng.choice(flat.shape[0], size=int(gaps * flat.shape[0]), replace=False)
        flat[idx] = np.nan
    return PoseSequence("t", list(CORE_13), xy, np.zeros((n_frames, 13)),
                        np.ones((n_frames, 13)), 30.0, meta={"frame_size": [1080, 1920]})


# --------------------------------------------------------------------------- #
# A NaN embedding was attributed to a real account
# --------------------------------------------------------------------------- #
ENROLLED = {1: {"template": np.ones(128), "name": "alice"}}


@pytest.mark.parametrize("probe, why", [
    (np.full(128, np.nan), "NaN"),
    (np.full(128, np.inf), "infinite"),
    (np.zeros(128), "zero-length"),
    (np.array([]), "empty"),
    (np.ones(64), "wrong width"),
])
def test_degenerate_embedding_is_never_attributed(probe, why):
    """`best < THRESHOLD` is False for NaN, so the reject guard never fired.

    Every comparison against NaN is False, which let a degenerate vector through
    a check written as a test *for rejection*. The guard now tests for
    acceptance, so anything unexpected fails closed.
    """
    assert _match_template(probe, ENROLLED) is None, f"{why} probe was matched"


def test_a_genuine_embedding_still_matches():
    assert _match_template(np.ones(128) * 0.5, ENROLLED) is not None


def test_one_corrupt_stored_template_does_not_break_the_others():
    """A bad row must be skipped, not take down the whole group's attribution."""
    got = _match_template(np.ones(128), {
        1: {"template": np.full(128, np.nan), "name": "corrupt"},
        2: {"template": np.ones(128), "name": "good"},
    })
    assert got is not None and got["name"] == "good"


# --------------------------------------------------------------------------- #
# Concurrent "take this drill on" created duplicate assignments
# --------------------------------------------------------------------------- #
class _Walk:
    findings, strengths, quality_warnings, coverage = [], [], [], {}
    source = "model"
    scores = {"posture": 5.0}
    overall, grade = 55.0, "Developing"


def test_concurrent_assign_yields_one_live_drill(tmp_path):
    """Six simultaneous clicks produced three live assignments before the fix.

    The SELECT-then-INSERT check is a race; only the partial unique index makes
    one-live-assignment-per-drill actually true.
    """
    db = tmp_path / "race.db"
    uid = store.get_or_create_user("racer", db_path=db)
    store.save_session(uid, "w", _Walk(), {}, db_path=db)

    ids: list = []
    def go():
        try:
            ids.append(drills.assign(uid, "posture", "Wall alignment", 5.0, db_path=db))
        except Exception as exc:            # pragma: no cover - must not happen
            ids.append(f"raised {type(exc).__name__}")

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(ids)) == 1, f"callers disagreed on the assignment: {set(ids)}"
    assert len(drills.active(uid, db_path=db)) == 1


def test_migrate_folds_pre_existing_duplicates(tmp_path):
    """A database written before the index existed must still migrate.

    CREATE UNIQUE INDEX fails against data that violates it, so duplicates are
    folded first - keeping the earliest baseline and re-pointing practice rows,
    because the user really did those sessions.
    """
    db = tmp_path / "old.db"
    uid = store.get_or_create_user("legacy", db_path=db)
    store.save_session(uid, "w", _Walk(), {}, db_path=db)
    with store.connect(db) as con:
        con.executescript(drills.SCHEMA)
        for i in range(3):
            cur = con.execute(
                "INSERT INTO drill_assignments (user_id, session_id, criterion, "
                "drill_name, baseline, assigned_at) VALUES (?,?,?,?,?,?)",
                (uid, None, "posture", "Wall alignment", 5.0 + i, f"2026-01-0{i + 1}"))
            con.execute("INSERT INTO drill_practice (assignment_id, practised_at) "
                        "VALUES (?,?)", (cur.lastrowid, "2026-01-05"))

    drills.migrate(db)
    live = drills.active(uid, db_path=db)
    assert len(live) == 1
    assert live[0]["baseline"] == 5.0, "the earliest baseline must survive"
    assert live[0]["practice_sessions"] == 3, "practice history must be preserved"


# --------------------------------------------------------------------------- #
# Group progress ran backwards
# --------------------------------------------------------------------------- #
def test_body_box_is_always_a_valid_rectangle():
    """A detection outside the frame produced an inverted box.

    numpy slices an inverted range as *empty* rather than raising, so the crop
    silently contained nothing.
    """
    for box in [(9999, 9999, 50, 50), (-500, -500, 50, 50), (10, 10, 0, 0),
                (320, 240, 100, 100)]:
        x0, y0, x1, y1 = multi._body_box(box, 640, 480)
        assert 0 <= x0 <= x1 <= 640, box
        assert 0 <= y0 <= y1 <= 480, box


# --------------------------------------------------------------------------- #
# Degenerate recordings: crash with a LAPACK message, or a confident score
# --------------------------------------------------------------------------- #
def test_one_frame_gives_a_readable_error_not_a_lapack_message():
    """"SVD did not converge in Linear Least Squares" was shown to users."""
    with pytest.raises(ValueError, match="at least"):
        extract_features(_seq(1))
    assert MIN_ANALYSIS_FRAMES == 2


def test_a_recording_with_no_measurements_is_refused_not_scored():
    """All-NaN coordinates scored 52.6/100 next to a "100% missing" warning.

    Both statements cannot be useful at once: if every value was interpolated
    from nothing, the score describes the interpolator.
    """
    with pytest.raises(UnusableRecording):
        analyze_sequence(_seq(90, fill=np.nan), None)


@pytest.mark.parametrize("gaps", [0.0, 0.25, 0.5, 0.75, 0.85])
def test_a_partially_degraded_recording_is_still_scored(gaps):
    """The refusal must not swallow clips a caveated score still helps with."""
    result = analyze_sequence(_seq(90, gaps=gaps), None)
    assert 0 <= result.assessment.overall <= 100


# --------------------------------------------------------------------------- #
# Reads that wrote
# --------------------------------------------------------------------------- #
def test_showing_progress_for_an_unknown_name_creates_nothing(tmp_path):
    db = tmp_path / "p.db"
    assert store.find_user("typo", db_path=db) is None
    with store.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0
