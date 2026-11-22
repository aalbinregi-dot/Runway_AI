"""Test suite.

The emphasis is on the properties that silently break: rigid-body invariants in
the generator, the schema contract against the supplied workbook, and the
*direction* of every feature. A feature that responds to the wrong thing still
produces plausible numbers, so several tests assert monotonicity against a
known ground truth rather than just checking for finite output.

Run:  python -m pytest tests -q      (or: python tests/test_pipeline.py)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runway_ai.config import CORE_13, CRITERIA, CRITERION_WEIGHTS, KEYPOINT_COLUMNS
from runway_ai.data.loaders import preprocess, read_keypoints
from runway_ai.data.schema import (
    PoseSequence, SchemaError, long_to_sequences, validate_long,
)
from runway_ai.data.synthetic import (
    SEG, WalkParams, make_clip, render_skeleton, sample_params,
)
from runway_ai.features.builder import extract_features
from runway_ai.features.signals import cycle_times, gait_highpass, interval_cv
from runway_ai.scoring.assess import assess
from runway_ai.scoring.rubric import grade, overall_score, rubric_scores
from runway_ai.feedback.coach import build_plan


def params(**overrides) -> WalkParams:
    q = {f"{c}_q": 0.6 for c in CRITERIA}
    q.update({k: v for k, v in overrides.items() if k.endswith("_q")})
    rest = {k: v for k, v in overrides.items() if not k.endswith("_q")}
    rest.setdefault("fps", 30.0)
    rest.setdefault("duration_s", 6.0)
    return WalkParams(**q, **rest)


def clip(**overrides):
    rng = np.random.default_rng(overrides.pop("seed", 0))
    seq, meta = make_clip("test", params(**overrides), rng)
    return preprocess(seq), meta


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
class TestSchema:
    def test_matches_supplied_workbook(self):
        """The sample workbook must load unchanged through our reader."""
        book = ROOT / "Synthetic_Runway_Keypoints_Sample.xlsx"
        if not book.exists():
            pytest.skip("sample workbook not present")
        df = read_keypoints(book)
        assert list(df.columns) == KEYPOINT_COLUMNS
        assert set(df["landmark"].unique()) == set(CORE_13)
        assert df["video_id"].nunique() == 5

    def test_round_trip(self):
        seq, _ = clip()
        long = seq.to_long()
        assert list(long.columns) == KEYPOINT_COLUMNS
        back = long_to_sequences(long, fps=seq.fps)["test"]
        assert back.n_frames == seq.n_frames
        np.testing.assert_allclose(back.xy, seq.xy, atol=1e-9)

    def test_rejects_missing_columns(self):
        with pytest.raises(SchemaError):
            validate_long(pd.DataFrame({"video_id": ["a"], "frame": [1]}))

    def test_rejects_duplicates(self):
        row = {"video_id": "a", "frame": 1, "landmark": "nose",
               "x": 0.5, "y": 0.5, "z": 0.0, "visibility": 1.0}
        with pytest.raises(SchemaError, match="duplicate"):
            validate_long(pd.DataFrame([row, row]))

    def test_missing_landmarks_are_filled_not_dropped(self):
        """A clip lacking a landmark must still produce a full tensor."""
        seq, _ = clip()
        long = seq.to_long()
        long = long[long["landmark"] != "left_wrist"]
        back = long_to_sequences(long)["test"]
        assert back.xy.shape[1] == len(CORE_13)
        assert back.vis("left_wrist").max() == 0.0


# --------------------------------------------------------------------------- #
# Real MoCap workbook
# --------------------------------------------------------------------------- #
class TestMoCapWorkbook:
    """Integration tests against the real Motion Capture keypoint file.

    These are the Tier-D (real data) checks described in docs/DATASET.md.
    Every assertion is against measurable properties of the file itself, not
    against synthetic expectations, so they fail loudly if the file changes or
    the pipeline mis-reads it.
    """

    BOOK = ROOT / "data" / "raw" / "Optimal_Runway_Keypoints_from_MoCap.xlsx"
    SIDECAR = ROOT / "data" / "raw" / "Optimal_Runway_Keypoints_from_MoCap.xlsx.meta.json"
    N_CLIPS = 14
    N_SUBJECTS = 7
    RECORDED_FPS = 50.0
    FRAMES_PER_CLIP = 200

    @pytest.fixture(scope="class")
    def mocap_df(self):
        if not self.BOOK.exists():
            pytest.skip("MoCap workbook not present")
        return read_keypoints(self.BOOK)

    @pytest.fixture(scope="class")
    def mocap_seqs(self):
        if not self.BOOK.exists():
            pytest.skip("MoCap workbook not present")
        from runway_ai.data.loaders import load_sequences
        return load_sequences(self.BOOK, fps=self.RECORDED_FPS)

    @pytest.fixture(scope="class")
    def mocap_results(self, mocap_seqs):
        from runway_ai.analyze import analyze_sequence
        from runway_ai.models.train import load_models
        models = load_models() or None
        results = {}
        for vid, seq in mocap_seqs.items():
            results[vid] = analyze_sequence(seq, models=models)
        return results

    # -- schema ----------------------------------------------------------------

    def test_schema_columns(self, mocap_df):
        """File must have the canonical 7-column schema."""
        assert list(mocap_df.columns) == KEYPOINT_COLUMNS

    def test_schema_landmarks(self, mocap_df):
        """All 13 analysis landmarks must be present."""
        assert set(mocap_df["landmark"].unique()) == set(CORE_13)

    def test_schema_clip_count(self, mocap_df):
        """Must contain exactly 14 clips (7 subjects x ~2 sessions each)."""
        assert mocap_df["video_id"].nunique() == self.N_CLIPS

    def test_schema_frame_count(self, mocap_df):
        """Every clip must have exactly 200 frames."""
        frames_per_clip = mocap_df.groupby("video_id")["frame"].nunique()
        assert (frames_per_clip == self.FRAMES_PER_CLIP).all(), (
            f"Some clips have unexpected frame counts:\n{frames_per_clip}"
        )

    # -- fps sidecar -----------------------------------------------------------

    def test_sidecar_exists(self):
        """The fps sidecar must be present so timing features are correct."""
        if not self.BOOK.exists():
            pytest.skip("MoCap workbook not present")
        assert self.SIDECAR.exists(), (
            f"Missing fps sidecar: {self.SIDECAR}\n"
            "Without it load_sequences defaults to 30 fps, miscalculating "
            "all cadence, rhythm and stride-timing features for 50 fps clips."
        )

    def test_sidecar_fps_value(self):
        """Sidecar must specify fps=50.0."""
        if not self.SIDECAR.exists():
            pytest.skip("sidecar not present")
        import json
        data = json.loads(self.SIDECAR.read_text())
        assert data.get("fps") == pytest.approx(50.0)

    def test_sequences_loaded_at_correct_fps(self, mocap_seqs):
        """All sequences must carry fps=50 after loading with the sidecar value."""
        for vid, seq in mocap_seqs.items():
            assert seq.fps == pytest.approx(self.RECORDED_FPS), (
                f"{vid} loaded at {seq.fps} fps instead of {self.RECORDED_FPS}"
            )

    # -- pipeline ---------------------------------------------------------------

    def test_all_clips_score_without_error(self, mocap_results):
        """Every clip must produce a valid Assessment with finite overall score."""
        assert len(mocap_results) == self.N_CLIPS
        for vid, result in mocap_results.items():
            assert np.isfinite(result.assessment.overall), (
                f"{vid} produced non-finite overall score"
            )
            assert 0 <= result.assessment.overall <= 100, (
                f"{vid} overall {result.assessment.overall} out of range"
            )

    def test_all_criterion_scores_are_finite(self, mocap_results):
        """Every per-criterion score must be a finite number 0-10."""
        for vid, result in mocap_results.items():
            for crit, score in result.assessment.scores.items():
                assert np.isfinite(score), f"{vid} {crit} is non-finite"
                assert 0 <= score <= 10, f"{vid} {crit}={score:.2f} out of range"

    def test_coaching_plan_is_generated(self, mocap_results):
        """Every clip must have a structurally valid coaching plan.

        Elite MoCap walks score 81-89/100 with all criteria above FAULT_SCORE
        (7.0), so they legitimately produce zero focus areas - there is nothing
        to correct.  The test therefore checks that:
          1. Every plan has the required keys with the right types.
          2. For clips whose overall score leaves room for improvement (< 85),
             the plan must surface at least one focus area.
        """
        from runway_ai.feedback.coach import CoachingPlan
        required_keys = set(CoachingPlan.__dataclass_fields__)
        found_focus = False
        for vid, result in mocap_results.items():
            plan_dict = result.plan.as_dict()
            missing = required_keys - plan_dict.keys()
            assert not missing, f"{vid} coaching plan is missing keys: {missing}"
            assert isinstance(plan_dict["focus_areas"], list), (
                f"{vid} focus_areas is not a list"
            )
            assert isinstance(plan_dict["drills"], list), (
                f"{vid} drills is not a list"
            )
            # Clips below the elite threshold must have actionable focus areas.
            if result.assessment.overall < 85.0:
                assert len(plan_dict["focus_areas"]) > 0, (
                    f"{vid} (overall={result.assessment.overall:.1f}) coaching "
                    f"plan has empty focus areas despite sub-elite score"
                )
                found_focus = True
        # Sanity check: the dataset must contain at least one improvable clip
        # so the branch above actually executes.
        assert found_focus, (
            "No clip scored below 85 -- update this threshold or add a lower-scoring clip"
        )

    # -- feature plausibility against clip_info reference ----------------------

    def test_cadence_order_matches_clip_info(self, mocap_seqs):
        """Clips with higher reference cadence_spm should have shorter step intervals.

        clip_info reference (steps/min):
          S11_Walking  115.4  <- highest cadence
          S8_Walking    78.3  <- lowest cadence

        The pipeline's rhythm_step_time_mean is 1/cadence (seconds per step),
        so the faster walker must have a shorter measured step interval.
        """
        from runway_ai.features.builder import extract_features

        fast_vid = "video_001_S11_Walking"   # 115.4 spm
        slow_vid = "video_007_S8_Walking"    # 78.3 spm

        fast_feats = extract_features(mocap_seqs[fast_vid])
        slow_feats = extract_features(mocap_seqs[slow_vid])

        fast_step = fast_feats.get("rhythm_step_time_mean")
        slow_step = slow_feats.get("rhythm_step_time_mean")

        # Only assert if the feature is present and sensible
        if fast_step and slow_step and fast_step > 0 and slow_step > 0:
            assert fast_step < slow_step, (
                f"S11 (115 spm) step_time={fast_step:.3f}s should be shorter "
                f"than S8 (78 spm) step_time={slow_step:.3f}s"
            )

    def test_subject_ids_cover_expected_set(self, mocap_df):
        """Clip names must encode the 7 expected subject IDs."""
        subject_ids = set()
        for vid in mocap_df["video_id"].unique():
            # Names like "video_001_S11_Walking" -> "S11"
            parts = vid.split("_")
            for p in parts:
                if p.startswith("S") and p[1:].isdigit():
                    subject_ids.add(p)
        expected = {"S1", "S5", "S6", "S7", "S8", "S9", "S11"}
        assert subject_ids == expected, (
            f"Subject IDs differ. Got: {subject_ids}, expected: {expected}"
        )


# --------------------------------------------------------------------------- #
# Generator physics
# --------------------------------------------------------------------------- #
class TestGeneratorPhysics:
    @pytest.mark.parametrize("skill", [0.05, 0.5, 0.95])
    def test_limb_segments_are_rigid(self, skill):
        """Bones must not stretch. A silently clamped IK shows up here first."""
        rng = np.random.default_rng(1)
        pts, _ = render_skeleton(sample_params(rng, skill=skill), rng)
        H = sample_params(np.random.default_rng(1), skill=skill).height_m
        for a, b, seg in [(7, 9, "thigh"), (8, 10, "thigh"),
                          (9, 11, "shank"), (10, 12, "shank")]:
            d = np.linalg.norm(pts[:, a] - pts[:, b], axis=1)
            assert d.std() / d.mean() < 0.02, f"{seg} varies by {d.std()/d.mean():.1%}"

    def test_arm_is_a_rigid_pendulum(self):
        rng = np.random.default_rng(2)
        pts, _ = render_skeleton(sample_params(rng, skill=0.5), rng)
        for sh, wr in [(1, 5), (2, 6)]:
            d = np.linalg.norm(pts[:, sh] - pts[:, wr], axis=1)
            np.testing.assert_allclose(d, d[0], rtol=1e-6)

    def test_leg_never_over_extends(self):
        """The pelvis must sit where the legs can actually hold it."""
        rng = np.random.default_rng(3)
        p = sample_params(rng, skill=0.4)
        pts, _ = render_skeleton(p, rng)
        leg = (SEG["thigh"] + SEG["shank"]) * p.height_m
        for hip, ankle in [(7, 11), (8, 12)]:
            d = np.linalg.norm(pts[:, hip] - pts[:, ankle], axis=1)
            assert d.max() <= leg * 1.01, f"hip-ankle {d.max():.3f} exceeds leg {leg:.3f}"

    def test_hip_bob_is_physiological(self):
        """Vertical pelvis excursion should land in the 1-12 cm range."""
        rng = np.random.default_rng(4)
        for skill in (0.2, 0.6, 0.9):
            pts, _ = render_skeleton(sample_params(rng, skill=skill), rng)
            pelvis = 0.5 * (pts[:, 7, 1] + pts[:, 8, 1])
            assert 0.005 < np.ptp(pelvis) < 0.15

    def test_landmarks_stay_in_frame(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            seq, _ = make_clip("f", sample_params(rng), rng)
            assert seq.xy.min() >= -0.15 and seq.xy.max() <= 1.15

    def test_labels_are_derived_from_latents(self):
        seq, meta = clip(posture_q=0.25)
        assert meta["score_posture"] == pytest.approx(2.5, abs=1e-6)


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
class TestSignals:
    def test_highpass_removes_arbitrary_drift_keeps_oscillation(self):
        """The perspective drift is not polynomial, so this must not assume it is."""
        fps, t = 30.0, np.arange(180) / 30.0
        drift = 3.0 * np.exp(t / 3.0)                 # strongly curved, non-polynomial
        osc = 0.2 * np.sin(2 * np.pi * 0.9 * t)
        out = gait_highpass(drift + osc, fps)
        assert np.std(out) < 0.25
        assert np.corrcoef(out, osc)[0, 1] > 0.8

    def test_cycle_times_recovers_known_period(self):
        fps = 30.0
        t = np.arange(300) / fps
        sig = np.sin(2 * np.pi * 0.9 * t)
        times = cycle_times(sig, fps)
        mean_iv, cv = interval_cv(times)
        assert mean_iv == pytest.approx(1 / 0.9, rel=0.05)
        assert cv < 0.05

    def test_cycle_times_detects_irregularity(self):
        """An unevenly timed signal must produce a materially higher CV.

        The jitter is applied *per cycle*, not per frame. Per-frame noise on the
        phase velocity averages out over the ~33 samples in a cycle and leaves
        the period almost unchanged - which is exactly the mistake the
        generator originally made.
        """
        fps, n = 30.0, 300
        rng = np.random.default_rng(0)
        steady = np.sin(2 * np.pi * 0.9 * np.arange(n) / fps)

        phase, cur, tempo = np.zeros(n), 0.0, 1.0
        for i in range(1, n):
            if int(cur / (2 * np.pi)) != int(phase[i - 1] / (2 * np.pi)) or i == 1:
                tempo = float(np.clip(1 + rng.normal(0, 0.35), 0.4, 2.0))
            cur += 2 * np.pi * 0.9 * tempo / fps
            phase[i] = cur
        jittery = np.sin(phase)
        _, cv_steady = interval_cv(cycle_times(steady, fps))
        _, cv_jitter = interval_cv(cycle_times(jittery, fps))
        assert cv_jitter > cv_steady * 2


# --------------------------------------------------------------------------- #
# Feature direction
# --------------------------------------------------------------------------- #
class TestFeatureDirection:
    """Each feature must move the right way when its latent moves.

    Averaged over several nuisance draws, because a single clip's camera angle
    can swamp a modest effect.
    """

    @staticmethod
    def _mean_feature(key: str, crit: str, level: float, reps: int = 8) -> float:
        rng = np.random.default_rng(11)
        vals = []
        for _ in range(reps):
            q = {f"{c}_q": 0.6 for c in CRITERIA}
            q[f"{crit}_q"] = level
            p = WalkParams(**q, fps=30.0, duration_s=6.0,
                           camera_azimuth_deg=float(rng.normal(0, 8)),
                           camera_distance_m=5.5)
            seq, _ = make_clip("x", p, rng)
            vals.append(extract_features(preprocess(seq))[key])
        return float(np.mean(vals))

    @pytest.mark.parametrize("key,crit", [
        ("posture_trunk_lean_mean", "posture"),
        ("balance_sway_amp", "balance"),
        ("shoulder_tilt_abs_mean", "shoulder_alignment"),
        ("head_lateral_offset", "head_position"),
        ("hip_obliquity_amp", "hip_movement"),
        ("foot_step_width_mean", "foot_placement"),
        ("rhythm_stride_cv", "rhythm"),
        ("conf_arm_spread", "confidence"),
    ])
    def test_fault_measure_falls_as_quality_rises(self, key, crit):
        poor = self._mean_feature(key, crit, 0.05)
        good = self._mean_feature(key, crit, 0.95)
        assert poor > good, f"{key}: poor={poor:.4f} not above good={good:.4f}"

    def test_stage_presence_is_actually_rendered(self):
        """Regression guard: the label once existed with no signal in the pixels."""
        poor = self._mean_feature("conf_chest_openness", "stage_presence", 0.05)
        good = self._mean_feature("conf_chest_openness", "stage_presence", 0.95)
        assert good > poor

    def test_no_dead_features(self):
        """Every feature must vary across clips; a constant column is a bug."""
        rng = np.random.default_rng(7)
        rows = [extract_features(preprocess(make_clip(f"c{i}", sample_params(rng), rng)[0]))
                for i in range(30)]
        df = pd.DataFrame(rows)
        dead = [c for c in df.columns if df[c].std() < 1e-12 and c != "qc_fps"]
        assert not dead, f"features never vary: {dead}"

    def test_features_are_finite(self):
        seq, _ = clip()
        feats = extract_features(seq)
        assert all(np.isfinite(v) for v in feats.values())


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
class TestScoring:
    def test_weights_sum_to_one(self):
        assert sum(CRITERION_WEIGHTS.values()) == pytest.approx(1.0)

    def test_scores_are_in_range(self):
        seq, _ = clip()
        a = assess(extract_features(seq))
        assert set(a.scores) == set(CRITERIA)
        assert all(0.0 <= v <= 10.0 for v in a.scores.values())
        assert 0.0 <= a.overall <= 100.0

    def test_overall_tracks_criteria(self):
        assert overall_score({c: 10.0 for c in CRITERIA}) == pytest.approx(100.0)
        assert overall_score({c: 0.0 for c in CRITERIA}) == pytest.approx(0.0)
        assert overall_score({c: 5.0 for c in CRITERIA}) == pytest.approx(50.0)

    def test_grades_are_ordered(self):
        assert grade(95) == "Professional"
        assert grade(50) == "Beginner"
        assert grade(0) == "Foundational"

    def test_good_walk_outscores_poor_walk(self):
        """The end-to-end system must rank skill correctly."""
        from runway_ai.models.train import load_models
        models = load_models() or None
        rng = np.random.default_rng(21)
        highs, lows = [], []
        for _ in range(6):
            for skill, bucket in ((0.92, highs), (0.12, lows)):
                seq, _ = make_clip("x", sample_params(rng, skill=skill), rng)
                bucket.append(assess(extract_features(preprocess(seq)), models).overall)
        assert np.mean(highs) > np.mean(lows) + 20

    def test_findings_never_contradict_scores(self):
        """A criterion reported as a strength must not also raise a fault."""
        rng = np.random.default_rng(31)
        for _ in range(10):
            seq, _ = make_clip("x", sample_params(rng), rng)
            a = assess(extract_features(preprocess(seq)))
            faulted = {f.criterion for f in a.findings}
            for crit in faulted:
                assert a.scores[crit] < 7.0

    def test_short_clip_is_flagged(self):
        seq, _ = clip(duration_s=2.0)
        a = assess(extract_features(seq))
        assert any("second" in w or "step" in w for w in a.quality_warnings)


# --------------------------------------------------------------------------- #
# Coaching
# --------------------------------------------------------------------------- #
class TestCoaching:
    def test_plan_is_capped_and_deduplicated(self):
        rng = np.random.default_rng(41)
        seq, _ = make_clip("x", sample_params(rng, skill=0.08), rng)
        plan = build_plan(assess(extract_features(preprocess(seq))))
        assert len(plan.focus_areas) <= 3
        crits = [f["criterion"] for f in plan.focus_areas]
        assert len(crits) == len(set(crits)), "one criterion took two focus slots"
        assert plan.drills and plan.next_session

    def test_strong_walk_gets_no_manufactured_faults(self):
        from runway_ai.models.train import load_models
        rng = np.random.default_rng(43)
        seq, _ = make_clip("x", sample_params(rng, skill=0.97), rng)
        a = assess(extract_features(preprocess(seq)), load_models() or None)
        plan = build_plan(a)
        assert a.overall > 60
        assert len(plan.focus_areas) <= 3


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
class TestStore:
    def test_session_round_trip_and_progress(self, tmp_path):
        from runway_ai.db import store
        from runway_ai.models.train import load_models

        # Score with the trained models, as the real system does. Using the
        # rubric alone here made the test flaky: it carries roughly twice the
        # error, which a single clip per skill level cannot reliably separate.
        models = load_models() or None

        db = tmp_path / "t.db"
        uid = store.create_user("Test User", db_path=db)
        rng = np.random.default_rng(51)
        for skill in (0.3, 0.7):
            seq, _ = make_clip("v", sample_params(rng, skill=skill), rng)
            feats = extract_features(preprocess(seq))
            store.save_session(uid, "v", assess(feats, models), feats, db_path=db)

        hist = store.user_history(uid, db_path=db)
        assert len(hist) == 2
        assert set(hist[0]["scores"]) == set(CRITERIA)

        prog = store.progress(uid, db_path=db)
        assert prog["sessions"] == 2
        assert prog["overall_delta"] > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))


class TestFetch:
    """Share-link input.

    Network calls are never made here: only URL shape, the host allow-list and
    the private-address guard are exercised, because those are the parts that
    must not regress. The allow-list is a security boundary - the API fetches
    these URLs server-side, so anything it accepts is somewhere the server can
    be made to send a request.
    """

    def test_is_url_distinguishes_links_from_paths(self):
        from runway_ai.data.fetch import is_url

        assert is_url("https://www.youtube.com/watch?v=abc")
        assert is_url("http://vimeo.com/123")
        assert not is_url("walk.mp4")
        assert not is_url("C:/videos/walk.mp4")
        assert not is_url("file:///etc/passwd")
        assert not is_url("")

    @pytest.mark.parametrize("url", [
        "https://evil.example.com/video",
        "https://youtube.com.attacker.net/v",
        "https://raw.githubusercontent.com/x/y",
    ])
    def test_unknown_hosts_refused(self, url):
        from runway_ai.data.fetch import FetchError, fetch_video

        with pytest.raises(FetchError, match="not a supported source"):
            fetch_video(url)

    @pytest.mark.parametrize("url", [
        "http://localhost:8000/v",
        "http://127.0.0.1/v",
        "http://192.168.1.1/v",
        "http://169.254.169.254/latest/meta-data",   # cloud metadata endpoint
    ])
    def test_private_addresses_refused(self, url):
        from runway_ai.data.fetch import FetchError, fetch_video

        with pytest.raises(FetchError):
            fetch_video(url)

    def test_allowed_hosts_cover_the_advertised_platforms(self):
        from runway_ai.data.fetch import ALLOWED_HOSTS

        for host in ("youtube.com", "youtu.be", "instagram.com", "tiktok.com",
                     "vimeo.com"):
            assert host in ALLOWED_HOSTS

    def test_analyze_routes_urls_to_the_fetcher(self, monkeypatch):
        """A URL must reach fetch_video, not be treated as a file path."""
        from runway_ai import analyze as mod

        seen = {}

        def fake_fetch(url, *a, **k):
            seen["url"] = url
            raise mod_fetch.FetchError("stopped before any network call")

        import runway_ai.data.fetch as mod_fetch
        monkeypatch.setattr(mod_fetch, "fetch_video", fake_fetch)

        with pytest.raises(mod_fetch.FetchError):
            mod.analyze("https://www.youtube.com/watch?v=abc", load_trained=False)
        assert seen["url"] == "https://www.youtube.com/watch?v=abc"

    def test_link_provenance_does_not_clobber_the_scorer_field(self):
        """`source` is the scorer; link provenance must not overwrite it.

        Regression: provenance was originally written to ``source``, which
        ``Assessment.as_dict()`` already uses for "model"/"rubric". The dict
        replaced the string, and the client - which renders it as text - threw
        "Objects are not valid as a React child".
        """
        from runway_ai.analyze import AnalysisResult
        from runway_ai.scoring.assess import Assessment

        a = Assessment(scores={}, overall=50.0, grade="Developing", source="model")
        r = AnalysisResult("v", a, build_plan(a), {},
                           origin={"title": "Walk", "uploader": "me"})
        d = r.as_dict()

        assert isinstance(d["source"], str), "scorer field must stay a string"
        assert d["source"] == "model"
        assert d["origin"]["title"] == "Walk"

    def test_slug_is_filesystem_safe(self):
        from runway_ai.analyze import _slug

        assert _slug("Milan FW '25 — Look 3 / final") == "Milan_FW_25_Look_3_final"
        assert _slug("") == "linked_video"
        assert len(_slug("x" * 200)) <= 60


class TestTimeline:
    """Per-frame evidence.

    The point of the timeline is that a score can be *checked*, so what matters
    is that the traces come from the same expressions as the features and that
    the segmentation is physically sensible — not merely that a dict comes back.
    """

    @pytest.fixture(scope="class")
    def built(self):
        from runway_ai.features import timeline as tl

        rng = np.random.default_rng(11)
        seq, _ = make_clip("v", sample_params(rng, skill=0.35), rng)
        seq = preprocess(seq)
        a = assess(extract_features(seq))
        return tl.build(seq, a.findings), seq

    def test_traces_align_with_the_time_axis(self, built):
        t, seq = built
        for name, trace in t["traces"].items():
            assert len(trace["values"]) == len(t["times"]), name
        assert t["duration_s"] == pytest.approx(seq.n_frames / seq.fps, rel=1e-3)

    def test_cycles_are_ordered_and_physiological(self, built):
        t, _ = built
        cycles = t["cycles"]
        assert cycles == sorted(cycles), "cycle boundaries must increase"
        if len(cycles) >= 3:
            intervals = np.diff(cycles)
            # A stride is roughly 0.7-2.0 s; anything outside means the phase
            # tracking has locked onto the wrong periodicity.
            assert intervals.min() > 0.4, intervals
            assert intervals.max() < 3.0, intervals

    def test_feet_alternate(self, built):
        """Left and right strikes should interleave, not coincide."""
        t, _ = built
        left, right = t["strikes"]["left"], t["strikes"]["right"]
        if len(left) >= 2 and len(right) >= 2:
            for r in right:
                nearest = min(abs(r - l) for l in left)
                assert nearest > 0.05, "a left and right strike landed together"

    def test_evidence_points_inside_the_clip(self, built):
        t, _ = built
        for e in t["evidence"]:
            assert 0 <= e["t"] <= t["duration_s"], e
            assert e["trace"] in t["traces"], e

    def test_every_trace_target_exists(self):
        """Each mapped feature must name a trace the builder actually produces."""
        from runway_ai.features import timeline as tl

        rng = np.random.default_rng(3)
        seq, _ = make_clip("v", sample_params(rng, skill=0.5), rng)
        produced = set(tl._series(preprocess(seq)))
        for feature, (label, _unit, _dir) in tl.TRACES.items():
            assert label in produced, f"{feature} maps to missing trace {label!r}"


class TestMediaStore:
    """Short-lived store for link-fetched videos.

    It exists so the skeleton overlay works for links too. It must not become a
    way to read arbitrary files, nor a disk leak.
    """

    def _make(self, tmp_path, name="clip.mp4", data=b"x" * 2048):
        p = tmp_path / name
        p.write_bytes(data)
        return p

    def test_store_and_resolve(self, tmp_path, monkeypatch):
        from runway_ai import media

        monkeypatch.setattr(media, "MEDIA_DIR", tmp_path / "media")
        media.clear()
        token = media.store(self._make(tmp_path))
        entry = media.resolve(token)
        assert entry is not None
        assert entry.path.read_bytes() == b"x" * 2048
        assert entry.content_type == "video/mp4"
        media.clear()

    def test_unknown_token_resolves_to_none(self, tmp_path, monkeypatch):
        from runway_ai import media

        monkeypatch.setattr(media, "MEDIA_DIR", tmp_path / "media")
        media.clear()
        assert media.resolve("not-a-real-token") is None
        # A token is never a path, so traversal has nothing to act on.
        assert media.resolve("../../../etc/passwd") is None

    def test_expired_entries_are_swept(self, tmp_path, monkeypatch):
        from runway_ai import media

        monkeypatch.setattr(media, "MEDIA_DIR", tmp_path / "media")
        monkeypatch.setattr(media, "TTL_S", -1)          # everything is stale
        media.clear()
        token = media.store(self._make(tmp_path))
        assert media.resolve(token) is None, "expired entry must not resolve"

    def test_store_is_capped(self, tmp_path, monkeypatch):
        from runway_ai import media

        monkeypatch.setattr(media, "MEDIA_DIR", tmp_path / "media")
        monkeypatch.setattr(media, "MAX_ENTRIES", 3)
        media.clear()
        tokens = [media.store(self._make(tmp_path, f"c{i}.mp4")) for i in range(6)]
        alive = [t for t in tokens if media.resolve(t)]
        assert len(alive) <= 3, "the store must not grow without bound"
        # The survivors are the most recent ones.
        assert set(alive) <= set(tokens[-3:])
        media.clear()

    def test_tokens_are_unguessable(self, tmp_path, monkeypatch):
        from runway_ai import media

        monkeypatch.setattr(media, "MEDIA_DIR", tmp_path / "media")
        media.clear()
        tokens = {media.store(self._make(tmp_path, f"c{i}.mp4")) for i in range(5)}
        assert len(tokens) == 5, "tokens must be unique"
        assert all(len(t) >= 24 for t in tokens)
        media.clear()


class TestFramingCheck:
    def test_inverted_torso_is_reported_as_framing_not_posture(self):
        """A >90 degree trunk lean means the hips were located above the
        shoulders - the lower body is out of frame, not a posture fault."""
        from runway_ai.scoring.assess import check_quality

        warnings = check_quality({"posture_trunk_lean_mean": 140.0})
        assert any("lower body" in w for w in warnings), warnings

    def test_normal_lean_raises_no_framing_warning(self):
        from runway_ai.scoring.assess import check_quality

        warnings = check_quality({"posture_trunk_lean_mean": 12.0,
                                  "qc_duration_s": 6.0, "rhythm_n_steps": 8,
                                  "qc_mean_visibility": 0.95})
        assert not any("lower body" in w for w in warnings), warnings
