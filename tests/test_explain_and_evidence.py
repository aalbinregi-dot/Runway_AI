"""The explanation layer: per-band breakdown, measurement catalogue, evidence.

These are the payload blocks the result page is built from, and every one of them
restates something the scorer already computed. That is the risk they carry: a
restatement can drift from the thing it restates, and the drift is invisible -
both numbers look plausible, they simply disagree. So the tests here are almost
all *consistency* assertions between the explanation and the score it explains.

Run:  python -m pytest tests/test_explain_and_evidence.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runway_ai.config import CRITERIA, CRITERION_WEIGHTS
from runway_ai.data.loaders import preprocess
from runway_ai.data.synthetic import WalkParams, make_clip
from runway_ai.features import timeline as tl
from runway_ai.features.builder import extract_features
from runway_ai.scoring import explain
from runway_ai.scoring.explain import IN_TOLERANCE
from runway_ai.scoring.assess import assess
from runway_ai.scoring.rubric import RUBRIC, overall_score


def clip(**overrides):
    q = {f"{c}_q": 0.6 for c in CRITERIA}
    q.update({k: v for k, v in overrides.items() if k.endswith("_q")})
    rest = {k: v for k, v in overrides.items() if not k.endswith("_q")}
    rest.setdefault("fps", 30.0)
    rest.setdefault("duration_s", 7.0)
    seq, _ = make_clip("t", WalkParams(**q, **rest), np.random.default_rng(3))
    return preprocess(seq)


@pytest.fixture(scope="module")
def analysed():
    seq = clip()
    feats = extract_features(seq)
    return seq, feats, assess(feats)


class TestBreakdown:
    def test_covers_every_criterion(self, analysed):
        _, _, a = analysed
        assert set(a.breakdown) == set(CRITERIA)

    def test_marks_reconstruct_the_overall_score(self, analysed):
        """The marks must add up to the number they are explaining.

        `marks_earned` is score x weight x 10, and the overall is the same
        weighted sum rescaled to 100. If these two ever disagree, the page is
        telling the user where 100 marks went while showing them a different
        total.
        """
        _, _, a = analysed
        total = sum(e["marks_earned"] for e in a.breakdown.values())
        assert total == pytest.approx(overall_score(a.scores), abs=0.2)
        assert total == pytest.approx(a.overall, abs=0.2)

    def test_marks_lost_completes_the_available(self, analysed):
        _, _, a = analysed
        for crit, e in a.breakdown.items():
            assert e["marks_earned"] + e["marks_lost"] == pytest.approx(
                e["marks_available"], abs=0.02), crit
            assert e["marks_available"] == pytest.approx(
                CRITERION_WEIGHTS[crit] * 100, abs=0.01)

    def test_in_tolerance_agrees_with_the_bands_own_score(self, analysed):
        """The tolerance edge is not a second threshold chosen here.

        The rubric's falloff is 1/(1 + (excess/tol)^2), which is exactly 0.5 at
        one tolerance out, so the flag and the score have to agree. The exception
        is a band sitting *on* the edge: the score is rounded to three decimals
        for the payload, so 0.4996 becomes 0.5 and the two readings tie. That is
        why the flag is derived from the interval and this check skips the tie.
        """
        _, _, a = analysed
        for e in a.breakdown.values():
            for band in e["bands"]:
                if abs(band["score"] - IN_TOLERANCE) < 0.001:
                    continue
                assert band["in_tolerance"] == (band["score"] > IN_TOLERANCE), band

    def test_acceptable_interval_agrees_with_the_flag(self, analysed):
        """The drawn region and the flag must describe the same interval.

        Both numbers are rounded to four decimals for the payload, so a
        measurement within ~1e-4 of an edge cannot be resolved from them - the
        first walk this ran on had a sway amplitude 7e-5 outside its tolerance,
        which rounds to exactly the edge. Those ties are skipped rather than
        asserted either way.
        """
        _, _, a = analysed
        edge_resolution = 5e-4
        for e in a.breakdown.values():
            for band in e["bands"]:
                lo, hi = band["acceptable"]
                if any(v is not None and abs(band["measured"] - v) < edge_resolution
                       for v in (lo, hi)):
                    continue
                inside = ((lo is None or band["measured"] >= lo)
                          and (hi is None or band["measured"] <= hi))
                assert inside == band["in_tolerance"], band

    def test_shares_sum_to_one_per_criterion(self, analysed):
        _, _, a = analysed
        for crit, e in a.breakdown.items():
            assert sum(b["share"] for b in e["bands"]) == pytest.approx(1.0, abs=0.01), crit

    def test_absent_features_raise_no_band(self, analysed):
        """A keypoint clip has no face, so the four gaze bands must not appear.

        `score_criterion` skips bands whose feature is missing; reporting them
        here would show a user four head-position measurements that were never
        evaluated, three of which would read as a perfect zero.
        """
        _, feats, a = analysed
        assert not any(f.startswith("gaze_") for f in feats)
        features = [b["feature"] for b in a.breakdown["head_position"]["bands"]]
        assert not any(f.startswith("gaze_") for f in features)

    def test_every_rubric_band_has_a_label(self):
        """A new band with no catalogue entry would render as a raw feature key."""
        missing = sorted(
            band.feature
            for bands in RUBRIC.values() for band in bands
            if band.feature not in explain.FEATURE_LABELS
        )
        assert missing == []


class TestMeasurements:
    def test_groups_are_labelled_and_non_empty(self, analysed):
        _, _, a = analysed
        assert a.measurements
        for group in a.measurements:
            assert group["title"] and group["rows"]
            for row in group["rows"]:
                assert row["label"] and row["label"] != row["feature"]

    def test_scored_in_names_only_criteria_that_read_the_feature(self, analysed):
        _, _, a = analysed
        for group in a.measurements:
            for row in group["rows"]:
                if row["scored_in"] is None:
                    continue
                read_by = [b.feature for b in RUBRIC[row["scored_in"]]]
                assert row["feature"] in read_by, row

    def test_values_match_the_feature_vector(self, analysed):
        _, feats, a = analysed
        for group in a.measurements:
            for row in group["rows"]:
                assert row["value"] == pytest.approx(feats[row["feature"]], abs=1e-4)

    def test_missing_features_are_omitted_not_zeroed(self, analysed):
        """A clip with no measurable face must not report 0% eyeline forward."""
        _, _, a = analysed
        shown = {row["feature"] for g in a.measurements for row in g["rows"]}
        assert not any(f.startswith("gaze_") for f in shown)


class TestTimelineEvidence:
    @pytest.fixture(scope="class")
    def built(self, analysed):
        seq, _, a = analysed
        return tl.build(seq, a.findings)

    def test_traces_carry_unit_and_stats(self, built):
        for name, trace in built["traces"].items():
            assert trace["unit"], name
            assert trace["stats"]["min"] <= trace["stats"]["mean"] <= trace["stats"]["max"]

    def test_reference_comes_from_the_rubric(self, built):
        """A shaded target has to be a real band, with the real threshold in it."""
        for name, trace in built["traces"].items():
            ref = trace["reference"]
            if ref is None:
                continue
            band = next(b for b in RUBRIC[ref["criterion"]] if b.feature == ref["feature"])
            assert ref["ideal"] == pytest.approx(band.ideal, abs=1e-3)
            assert ref["tol"] == pytest.approx(band.tol, abs=1e-3)
            assert ref["mode"] == band.mode

    def test_reference_prefers_a_level_band_over_a_spread_one(self, built):
        """Trunk lean must shade against the mean band, not the std band.

        A trace is a per-frame level. Shading it against a standard deviation
        would draw a line the trace is not measured against, and the reader would
        compare an instantaneous angle with a tolerance on its variability.
        """
        ref = built["traces"]["Trunk lean"]["reference"]
        assert ref["feature"] == "posture_trunk_lean_mean"

    def test_strides_are_the_intervals_between_the_drawn_events(self, built):
        for side in ("left", "right"):
            events = built["strikes"][side]
            strides = built["strides"][side]
            assert len(strides) == max(0, len(events) - 1)
            for i, stride in enumerate(strides):
                assert stride["t"] == pytest.approx(events[i], abs=1e-3)
                assert stride["duration"] == pytest.approx(
                    events[i + 1] - events[i], abs=2e-3)

    def test_stride_summary_matches_the_strides(self, built):
        durations = [s["duration"] for side in ("left", "right")
                     for s in built["strides"][side]]
        if len(durations) > 1:
            assert built["strides"]["mean_s"] == pytest.approx(np.mean(durations), abs=2e-3)
            assert built["strides"]["spread_s"] == pytest.approx(
                max(durations) - min(durations), abs=2e-3)

    def test_evidence_points_at_a_real_extremum(self, built):
        """Each finding's cited moment must be inside the clip and on its trace."""
        for ev in built["evidence"]:
            assert 0 <= ev["t"] <= built["duration_s"] + 1e-6
            values = built["traces"][ev["trace"]]["values"]
            stats = built["traces"][ev["trace"]]["stats"]
            assert stats["min"] - 1e-3 <= ev["value"] <= stats["max"] + 1e-3
            assert values


def test_a_bad_clip_still_explains_itself():
    """The explanation must survive the input it is most needed on.

    A degraded recording produces measurements many tolerances outside their
    bands. Those are exactly the results a user will interrogate, so the
    breakdown has to hold up rather than divide by a zero-width range.
    """
    seq = clip(duration_s=3.5, posture_q=0.05, balance_q=0.05, rhythm_q=0.05)
    feats = extract_features(seq)
    a = assess(feats)
    assert set(a.breakdown) == set(CRITERIA)
    for e in a.breakdown.values():
        for band in e["bands"]:
            assert np.isfinite(band["measured"])
            assert 0.0 <= band["score"] <= 1.0
