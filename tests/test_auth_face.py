"""Accounts, sessions, and opt-in face matching.

The emphasis is on the properties that would be expensive to get wrong:
credentials never stored in the clear, failures indistinguishable to an
attacker, sessions revocable, biometrics deletable, and the matcher refusing
rather than guessing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def authdb(tmp_path, monkeypatch):
    """A migrated database, isolated from the live one."""
    import runway_ai.db.store as st
    from runway_ai.db import auth

    db = tmp_path / "auth.db"
    monkeypatch.setattr(st, "DB_PATH", db)
    auth.migrate(db)
    return db


class TestMigration:
    def test_preserves_existing_rows(self, tmp_path):
        """The live database predates auth and must survive the migration."""
        import sqlite3

        from runway_ai.db import auth, store

        db = tmp_path / "legacy.db"
        with store.connect(db) as con:
            con.execute("INSERT INTO users (name, experience, created_at) "
                        "VALUES ('Old', 'beginner', '2020-01-01T00:00:00+00:00')")
            con.execute(
                "INSERT INTO sessions (user_id, video_id, recorded_at, overall_score)"
                " VALUES (1, 'v1', '2020-01-01T00:00:00+00:00', 55.0)")

        auth.migrate(db)
        con = sqlite3.connect(db)
        assert con.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 1
        assert con.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 1
        assert con.execute("SELECT is_legacy FROM users").fetchone()[0] == 1
        assert con.execute("PRAGMA user_version").fetchone()[0] >= 1

    def test_is_idempotent(self, tmp_path):
        from runway_ai.db import auth

        db = tmp_path / "m.db"
        first = auth.migrate(db)
        assert first, "the first run must actually apply something"
        assert auth.migrate(db) == [], "the second run must be a no-op"


class TestAccounts:
    def test_signup_login_roundtrip(self, authdb):
        from runway_ai.db import auth

        u = auth.create_account("Ada", "ada@example.com", "a-long-enough-pass", authdb)
        back = auth.authenticate("  ADA@Example.com ", "a-long-enough-pass", authdb)
        assert back.id == u.id, "login must ignore case and surrounding space"

    def test_password_never_stored_in_clear(self, authdb):
        from runway_ai.db import auth, store

        auth.create_account("Ada", "ada@example.com", "hunter2-hunter2", authdb)
        with store.connect(authdb) as con:
            stored = con.execute("SELECT password_hash FROM users "
                                 "WHERE email_lower='ada@example.com'").fetchone()[0]
        assert "hunter2" not in stored
        assert stored.startswith("$argon2id$")

    @pytest.mark.parametrize("email,password", [
        ("ada@example.com", "wrong-password-x"),
        ("nobody@example.com", "a-long-enough-pass"),
    ])
    def test_failures_are_indistinguishable(self, authdb, email, password):
        """A wrong password and an unknown account must read identically."""
        from runway_ai.db import auth

        auth.create_account("Ada", "ada@example.com", "a-long-enough-pass", authdb)
        with pytest.raises(auth.AuthError) as exc:
            auth.authenticate(email, password, authdb)
        assert str(exc.value) == "Email or password is incorrect."

    def test_duplicate_email_rejected_case_insensitively(self, authdb):
        from runway_ai.db import auth

        auth.create_account("Ada", "ada@example.com", "a-long-enough-pass", authdb)
        with pytest.raises(auth.AuthError, match="already exists"):
            auth.create_account("Eve", "ADA@EXAMPLE.COM", "another-good-pass", authdb)

    def test_short_password_rejected(self, authdb):
        from runway_ai.db import auth

        with pytest.raises(auth.AuthError, match="10 characters"):
            auth.create_account("Ada", "ada@example.com", "short", authdb)

    def test_repeated_failures_lock_out(self, authdb):
        from runway_ai.db import auth

        auth.create_account("Ada", "ada@example.com", "a-long-enough-pass", authdb)
        for _ in range(auth.MAX_FAILED_LOGINS):
            with pytest.raises(auth.AuthError):
                auth.authenticate("ada@example.com", "nope-nope-nope", authdb)
        # Even the correct password is refused while the backoff holds.
        with pytest.raises(auth.AuthError, match="Too many"):
            auth.authenticate("ada@example.com", "a-long-enough-pass", authdb)

    def test_legacy_row_cannot_log_in(self, authdb):
        from runway_ai.db import auth, store

        with store.connect(authdb) as con:
            con.execute("INSERT INTO users (name, email, email_lower, experience,"
                        " created_at, is_legacy) VALUES "
                        "('Old','old@x.com','old@x.com','beginner','2020-01-01',1)")
        with pytest.raises(auth.AuthError):
            auth.authenticate("old@x.com", "anything-at-all", authdb)


class TestSessions:
    def test_token_is_not_stored_verbatim(self, authdb):
        from runway_ai.db import auth, store

        u = auth.create_account("Ada", "ada@example.com", "a-long-enough-pass", authdb)
        token = auth.issue_session(u.id, db_path=authdb)
        with store.connect(authdb) as con:
            rows = [r[0] for r in con.execute("SELECT token_hash FROM auth_sessions")]
        assert token not in rows, "a database leak must not hand over live sessions"
        assert auth.resolve_session(token, authdb).id == u.id

    def test_revoked_session_stops_resolving(self, authdb):
        from runway_ai.db import auth

        u = auth.create_account("Ada", "ada@example.com", "a-long-enough-pass", authdb)
        token = auth.issue_session(u.id, db_path=authdb)
        auth.revoke_session(token, authdb)
        assert auth.resolve_session(token, authdb) is None

    @pytest.mark.parametrize("bad", ["not-a-token", "", None, "../../etc/passwd"])
    def test_garbage_tokens_resolve_to_none(self, authdb, bad):
        from runway_ai.db import auth

        assert auth.resolve_session(bad, authdb) is None


class TestLegacyClaim:
    def test_claim_moves_walks_and_removes_the_legacy_row(self, authdb):
        from runway_ai.db import auth, store

        with store.connect(authdb) as con:
            con.execute("INSERT INTO users (name, experience, created_at, is_legacy)"
                        " VALUES ('Old','beginner','2020-01-01',1)")
            legacy_id = con.execute("SELECT user_id FROM users "
                                    "WHERE name='Old'").fetchone()[0]
            con.execute("INSERT INTO sessions (user_id, video_id, recorded_at,"
                        " overall_score) VALUES (?, 'v1', '2020-01-01', 55.0)",
                        (legacy_id,))
        u = auth.create_account("New", "new@x.com", "a-long-enough-pass", authdb)
        assert auth.claim_legacy(legacy_id, u.id, authdb) == 1
        with store.connect(authdb) as con:
            assert con.execute("SELECT COUNT(*) FROM sessions WHERE user_id=?",
                               (u.id,)).fetchone()[0] == 1
            assert con.execute("SELECT COUNT(*) FROM users WHERE user_id=?",
                               (legacy_id,)).fetchone()[0] == 0

    def test_cannot_claim_a_real_account(self, authdb):
        """Only legacy rows are claimable; a live account is not."""
        from runway_ai.db import auth

        a = auth.create_account("A", "a@x.com", "a-long-enough-pass", authdb)
        b = auth.create_account("B", "b@x.com", "a-long-enough-pass", authdb)
        with pytest.raises(auth.AuthError):
            auth.claim_legacy(a.id, b.id, authdb)


class TestFaceStorage:
    def test_template_roundtrip_and_delete(self, authdb):
        from runway_ai.db import auth

        u = auth.create_account("Ada", "ada@example.com", "a-long-enough-pass", authdb)
        emb = np.random.default_rng(0).normal(size=128).astype(np.float32)
        auth.save_face_template(u.id, emb, 7, authdb)

        assert np.allclose(auth.load_face_templates(authdb)[u.id], emb, atol=1e-6)
        assert auth.has_face_template(u.id, authdb)

        assert auth.delete_face_template(u.id, authdb) is True
        assert auth.load_face_templates(authdb) == {}
        # Removing biometrics must leave the account and its walks alone.
        assert auth.authenticate("ada@example.com", "a-long-enough-pass",
                                 authdb).id == u.id

    def test_wrong_dimension_refused(self, authdb):
        from runway_ai.db import auth

        u = auth.create_account("Ada", "ada@example.com", "a-long-enough-pass", authdb)
        with pytest.raises(ValueError, match="128"):
            auth.save_face_template(u.id, np.zeros(64, dtype=np.float32), 5, authdb)

    def test_deleting_the_account_deletes_the_biometrics(self, authdb):
        from runway_ai.db import auth, store

        u = auth.create_account("Ada", "ada@example.com", "a-long-enough-pass", authdb)
        auth.save_face_template(u.id, np.zeros(128, dtype=np.float32), 5, authdb)
        with store.connect(authdb) as con:
            con.execute("PRAGMA foreign_keys = ON")
            con.execute("DELETE FROM users WHERE user_id = ?", (u.id,))
        assert auth.load_face_templates(authdb) == {}, "cascade must remove it"


class TestFaceMatching:
    """Decision logic. Needs no model files, so it runs everywhere."""

    def _emb(self, seed):
        v = np.random.default_rng(seed).normal(size=128).astype(np.float32)
        return v / np.linalg.norm(v)

    def test_identifies_the_right_template(self):
        from runway_ai import face

        me = self._emb(1)
        probes = [me + 0.01 * self._emb(100 + i) for i in range(6)]
        r = face.identify(probes, {7: me, 8: self._emb(2)})
        assert r.user_id == 7 and r.identified

    def test_unknown_face_is_refused_not_guessed(self):
        from runway_ai import face

        r = face.identify([self._emb(50 + i) for i in range(6)],
                          {7: self._emb(1), 8: self._emb(2)})
        assert r.user_id is None
        assert "does not match" in r.reason

    def test_too_few_frames_is_refused(self):
        from runway_ai import face

        me = self._emb(1)
        r = face.identify([me, me], {7: me})
        assert r.user_id is None
        assert str(face.MIN_PROBE_FRAMES) in r.reason

    def test_ambiguous_match_is_refused(self):
        """Two near-identical templates must not yield a coin-toss answer."""
        from runway_ai import face

        me = self._emb(1)
        twin = me + 1e-4 * self._emb(9)
        r = face.identify([me] * 6, {7: me, 8: twin})
        assert r.user_id is None
        assert "too closely" in r.reason

    def test_no_enrolments_means_no_matching(self):
        from runway_ai import face

        r = face.identify([self._emb(1)] * 6, {})
        assert r.user_id is None and "no users" in r.reason

    def test_enrolment_requires_enough_frames(self):
        from runway_ai import face

        with pytest.raises(ValueError, match=str(face.MIN_ENROL_FRAMES)):
            face.build_template([self._emb(i) for i in range(2)])

    def test_template_is_normalised(self):
        from runway_ai import face

        t = face.build_template([self._emb(i) for i in range(6)])
        assert np.isclose(np.linalg.norm(t), 1.0, atol=1e-5)

    def test_no_photo_file_enrolment_path_exists(self):
        """Enrolling from an arbitrary image file would allow registering a
        person who never consented. That path must not come back."""
        from runway_ai import face

        assert not hasattr(face, "embeddings_from_images")
        assert hasattr(face, "embeddings_from_captures")


class TestSignificance:
    """A difference smaller than the model's own error is not a result."""

    def test_thresholds_are_per_criterion(self):
        from runway_ai.scoring import significance as sig

        # Foot placement is measured far more precisely than symmetry, so a
        # single global threshold would be wrong in both directions.
        assert sig.threshold("foot_placement") < sig.threshold("symmetry")
        assert sig.threshold("foot_placement") < 0.7
        assert sig.threshold("symmetry") > 1.5

    def test_small_change_is_not_significant(self):
        from runway_ai.scoring import significance as sig

        assert not sig.is_significant("symmetry", 0.4)
        assert sig.is_significant("symmetry", 2.5)

    def test_threshold_scales_with_error(self):
        """The bar is sqrt(2) x MAE: differencing two estimates compounds error."""
        from runway_ai.scoring import significance as sig

        mae = sig.per_criterion_mae()
        for c, m in mae.items():
            assert sig.threshold(c) == pytest.approx(m * sig.DIFFERENCE_FACTOR, abs=0.01)

    def test_overall_threshold_uses_quadrature_not_a_sum(self):
        """Summing twelve errors assumes they all point the same way at once."""
        from runway_ai.config import CRITERIA, CRITERION_WEIGHTS
        from runway_ai.scoring import significance as sig

        linear = sum(CRITERION_WEIGHTS[c] * sig.threshold(c) for c in CRITERIA) * 10
        combined = sig.overall_threshold()
        assert combined < linear / 2, "quadrature must be well below the linear sum"
        assert 2.0 < combined < 8.0, f"implausible overall floor: {combined}"

    def test_missing_report_falls_back_without_crashing(self, tmp_path, monkeypatch):
        from runway_ai.scoring import significance as sig

        monkeypatch.setattr(sig, "REPORT_DIR", tmp_path)
        monkeypatch.setattr(sig, "_cache", None)
        assert sig.threshold("posture") == pytest.approx(
            sig.FALLBACK_MAE * sig.DIFFERENCE_FACTOR, abs=0.01)


class TestGaze:
    """Head pose from facial landmarks — the gap in 'Head Position & Gaze'."""

    def _series(self, yaw, pitch, roll, fps=7.5):
        import numpy as np
        n = len(yaw)
        return {"t": np.arange(n) / fps, "yaw": np.asarray(yaw, float),
                "pitch": np.asarray(pitch, float), "roll": np.asarray(roll, float),
                "fps": fps, "coverage": n}

    def test_absent_series_yields_no_features(self):
        """A clip with no visible face must add nothing, not zeros.

        Zeros would read as a perfect forward eyeline and silently improve the
        head-position finding for every keypoint-only upload.
        """
        from runway_ai.features import gaze

        assert gaze.features(None) == {}
        assert gaze.summary({}) is None

    def test_forward_gaze_scores_high_on_camera(self):
        from runway_ai.features import gaze

        n = 40
        f = gaze.features(self._series([0.0] * n, [gaze.NEUTRAL_PITCH] * n, [0.0] * n))
        assert f["gaze_on_camera_fraction"] == 1.0
        assert f["gaze_looking_down_fraction"] == 0.0

    def test_turned_head_is_detected(self):
        from runway_ai.features import gaze

        n = 40
        f = gaze.features(self._series([0.25] * n, [gaze.NEUTRAL_PITCH] * n, [0.0] * n))
        assert f["gaze_on_camera_fraction"] == 0.0
        assert f["gaze_yaw_abs_mean"] > gaze.YAW_NEUTRAL_BAND

    def test_dropped_chin_is_detected(self):
        """The pitch proxy must actually respond to looking down.

        Regression: the first proxy measured the nose's position *within* the
        face, where the two distances foreshorten together and the ratio
        cancels. It read 0.586 on a visibly floor-staring frame against
        0.543-0.584 upright — no signal at all.
        """
        from runway_ai.features import gaze

        n = 40
        down = gaze.NEUTRAL_PITCH - gaze.PITCH_DOWN_BAND - 0.05
        f = gaze.features(self._series([0.0] * n, [down] * n, [0.0] * n))
        assert f["gaze_looking_down_fraction"] == 1.0
        assert f["gaze_pitch_drop"] > gaze.PITCH_DOWN_BAND

    def test_summary_says_it_is_measured_not_scored(self):
        from runway_ai.features import gaze

        n = 40
        f = gaze.features(self._series([0.3] * n, [0.9] * n, [8.0] * n))
        s = gaze.summary(f)
        assert "not model-scored" in s["basis"]
        assert s["notes"], "a walk this far off-camera must say something"

    def test_gaze_bands_are_optional_in_the_rubric(self):
        """Head position must score identically when no face was available."""
        from runway_ai.scoring.rubric import score_criterion

        base = {"posture_neck_angle_mean": 6.0, "head_lateral_offset": 0.08,
                "head_lateral_amp": 0.09, "head_stability_jerk": 500.0}
        without, _ = score_criterion("head_position", base)
        with_gaze, _ = score_criterion(
            "head_position",
            base | {"gaze_on_camera_fraction": 0.95,
                    "gaze_looking_down_fraction": 0.0,
                    "gaze_roll_abs_mean": 1.0})
        assert without == pytest.approx(without)
        # Adding a *good* gaze reading must not make the score worse.
        assert with_gaze >= without - 0.01


class TestViewpoint:
    """Camera-angle estimation, validated against known ground truth."""

    def _clip(self, azimuth, seed=5, skill=0.7):
        from runway_ai.data.loaders import preprocess
        from runway_ai.data.synthetic import make_clip, sample_params

        rng = np.random.default_rng(seed)
        p = sample_params(rng, skill=skill)
        p.camera_azimuth_deg = float(azimuth)
        seq, _ = make_clip("x", p, rng)
        return preprocess(seq)

    def test_recovers_a_known_camera_angle(self):
        """The generator knows the true azimuth, so this is checkable."""
        from runway_ai.features import viewpoint as vp

        for truth in (0.0, 25.0, 55.0):
            got = vp.estimate(self._clip(truth), aspect=16 / 9)
            assert got is not None
            # Two standard deviations of the measured 8.5 deg spread.
            assert abs(got["angle_deg"] - truth) < 2 * vp.UNCERTAINTY_DEG, (
                f"truth {truth}, got {got['angle_deg']}")

    def test_bands_separate_head_on_from_side(self):
        from runway_ai.features import viewpoint as vp

        assert vp.estimate(self._clip(0), aspect=16 / 9)["band"] == "head-on"
        assert vp.estimate(self._clip(70), aspect=16 / 9)["band"] == "side-on"

    def test_reported_range_brackets_the_estimate(self):
        from runway_ai.features import viewpoint as vp

        got = vp.estimate(self._clip(30), aspect=16 / 9)
        lo, hi = got["range_deg"]
        assert lo <= got["angle_deg"] <= hi
        assert hi > lo, "an uncertainty range must have width"

    def test_reliability_flips_with_angle(self):
        """The point of the feature: a side view has the opposite weaknesses."""
        from runway_ai.features import viewpoint as vp

        head_on = vp.RELIABILITY["head-on"]
        side = vp.RELIABILITY["side-on"]
        assert "stride_length" in head_on["poor"]
        assert "stride_length" in side["good"]
        assert "foot_placement" in head_on["good"]
        assert "foot_placement" in side["poor"]

    def test_every_named_criterion_is_real(self):
        from runway_ai.config import CRITERIA
        from runway_ai.features import viewpoint as vp

        for band, table in vp.RELIABILITY.items():
            for key in table["good"] + table["poor"]:
                assert key in CRITERIA, f"{band} names unknown criterion {key}"
        for key in vp.ANGLE_NEUTRAL:
            assert key in CRITERIA

    def test_missing_aspect_is_flagged_not_hidden(self):
        from runway_ai.features import viewpoint as vp

        got = vp.estimate(self._clip(20), aspect=None)
        assert got["aspect_assumed"] is True, (
            "a guessed aspect ratio changes the answer and must be disclosed")
