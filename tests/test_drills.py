"""Drill tracking: baselines, ownership, and refusing to credit noise."""
from __future__ import annotations

import time

import pytest

from runway_ai.db import drills, store
from runway_ai.scoring import significance


class _Assessment:
    """Minimal stand-in for what save_session reads."""

    findings: list = []
    strengths: list = []
    source = "model"
    quality_warnings: list = []
    coverage: dict = {}

    def __init__(self, posture: float):
        self.scores = {"posture": posture, "balance": 6.0}
        self.overall = 55.0
        self.grade = "Developing"


@pytest.fixture()
def db(tmp_path):
    return tmp_path / "drills.db"


@pytest.fixture()
def user(db):
    uid = store.get_or_create_user("tester", db_path=db)
    store.save_session(uid, "w1", _Assessment(5.0), {}, db_path=db)
    return uid


def test_assign_records_the_baseline(db, user):
    aid = drills.assign(user, "posture", "Wall alignment", 5.0, db_path=db)
    entry = drills.active(user, db_path=db)[0]
    assert entry["assignment_id"] == aid
    assert entry["baseline"] == 5.0
    assert entry["instruction"], "the drill's text should be filled in from DRILLS"


def test_taking_the_same_drill_twice_keeps_the_original_baseline(db, user):
    """Re-assigning must not silently reset what progress is measured against."""
    first = drills.assign(user, "posture", "Wall alignment", 5.0, db_path=db)
    again = drills.assign(user, "posture", "Wall alignment", 9.9, db_path=db)
    assert again == first
    assert drills.active(user, db_path=db)[0]["baseline"] == 5.0


def test_unknown_drill_is_rejected(db, user):
    with pytest.raises(drills.DrillError):
        drills.assign(user, "posture", "Not a real drill", 5.0, db_path=db)


def test_practice_is_counted(db, user):
    aid = drills.assign(user, "posture", "Wall alignment", 5.0, db_path=db)
    assert drills.log_practice(user, aid, db_path=db) == 1
    assert drills.log_practice(user, aid, db_path=db) == 2
    assert drills.active(user, db_path=db)[0]["practice_sessions"] == 2


def test_another_users_drill_is_untouchable(db, user):
    """Ownership is checked in the data layer, not only in the route."""
    aid = drills.assign(user, "posture", "Wall alignment", 5.0, db_path=db)
    other = store.get_or_create_user("someone else", db_path=db)
    with pytest.raises(drills.DrillError):
        drills.log_practice(other, aid, db_path=db)
    assert drills.archive(other, aid, db_path=db) is False


def test_no_walk_yet_reports_awaiting_not_success(db, user):
    drills.assign(user, "posture", "Wall alignment", 5.0, db_path=db)
    assert drills.active(user, db_path=db)[0]["status"] == "awaiting_walk"


def _outcome_after(db, user, new_score: float) -> dict:
    drills.assign(user, "posture", "Wall alignment", 5.0, db_path=db)
    time.sleep(1.05)          # recorded_at has one-second resolution
    store.save_session(user, "w2", _Assessment(new_score), {}, db_path=db)
    return drills.active(user, db_path=db)[0]


def test_a_change_smaller_than_the_measurement_error_is_not_an_improvement(db, user):
    """The whole point of the feature: do not invent an effect out of noise.

    Posture's threshold is 1.41 x its out-of-fold MAE. A gain of 1.0 looks like
    progress and is not distinguishable from the model being wrong twice.
    """
    got = _outcome_after(db, user, 6.0)
    assert got["delta"] == 1.0
    assert got["status"] == "no_change"
    # And the wording must not let "undetectable" be read as "nothing happened".
    assert "not that nothing changed" in got["summary"]


def test_a_change_past_the_threshold_is_reported_as_real(db, user):
    got = _outcome_after(db, user, 5.0 + significance.threshold("posture") + 0.1)
    assert got["status"] == "improved"


def test_a_decline_past_the_threshold_is_reported_honestly(db, user):
    got = _outcome_after(db, user, 5.0 - significance.threshold("posture") - 0.1)
    assert got["status"] == "declined"
    assert got["delta"] < 0


def test_outcome_never_claims_the_drill_caused_the_change(db, user):
    """No wording anywhere may assert causation - there is no control here."""
    for score in (6.0, 8.0, 2.0):
        summary = _outcome_after(db, user, score)["summary"].lower()
        for banned in ("caused", "because of", "thanks to", "due to the drill"):
            assert banned not in summary


def test_archiving_moves_a_drill_to_history(db, user):
    aid = drills.assign(user, "posture", "Wall alignment", 5.0, db_path=db)
    assert drills.archive(user, aid, db_path=db) is True
    assert drills.active(user, db_path=db) == []
    assert len(drills.history(user, db_path=db)) == 1
    # Archiving twice is not an error, but it is also not a second archive.
    assert drills.archive(user, aid, db_path=db) is False


def test_migrate_is_idempotent(db):
    drills.migrate(db)
    drills.migrate(db)


def test_find_user_does_not_create(tmp_path):
    """A read must stay a read.

    `progress --user <typo>` used to call get_or_create_user, which quietly
    added an empty account to the database instead of reporting the name was
    unknown.
    """
    db = tmp_path / "lookup.db"
    assert store.find_user("nobody", db_path=db) is None
    with store.connect(db) as con:
        assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0

    uid = store.get_or_create_user("somebody", db_path=db)
    assert store.find_user("somebody", db_path=db) == uid
