"""Trend fitting across walks, and per-criterion identifiability."""
from __future__ import annotations

import numpy as np
import pytest

from runway_ai.config import CRITERIA
from runway_ai.scoring import confidence, significance, trend


# --------------------------------------------------------------------------- #
# Trend
# --------------------------------------------------------------------------- #
def test_too_few_walks_has_no_trend():
    """Two points define a line exactly, leaving nothing to estimate error from."""
    assert trend.fit([5.0, 6.0], "posture") is None
    assert trend.fit([5.0], "posture") is None
    assert trend.MIN_WALKS == 3


def test_a_strong_sustained_improvement_is_detected():
    got = trend.fit([5.0 + 0.6 * i for i in range(8)], "posture")
    assert got["direction"] == "improving"
    assert got["significant"]
    assert got["ci"][0] > 0, "the interval must exclude zero"


def test_a_strong_sustained_decline_is_detected():
    got = trend.fit([8.0 - 0.6 * i for i in range(8)], "posture")
    assert got["direction"] == "declining"
    assert got["ci"][1] < 0


def test_pure_noise_is_not_called_a_trend():
    rng = np.random.default_rng(0)
    mae = significance.per_criterion_mae().get("posture", 0.9)
    flagged = 0
    for _ in range(400):
        y = 5.0 + rng.normal(0, mae, 8)
        got = trend.fit(list(y), "posture")
        flagged += bool(got and got["significant"])
    # Nominally 5%; the measurement-error floor makes it stricter in practice.
    assert flagged / 400 < 0.05, f"false positive rate {flagged / 400:.3f}"


def test_the_measurement_error_floor_makes_the_test_stricter():
    """Three nearly-collinear points must not pass on a tiny residual spread.

    Without the floor, the residual estimate can come out near zero by luck and
    declare a meaningless slope significant.
    """
    almost_perfect = [5.0, 5.05, 5.10, 5.15]
    with_floor = trend.fit(almost_perfect, "posture")
    without_floor = trend.fit(almost_perfect, None)
    assert not with_floor["significant"]
    assert without_floor["significant"], (
        "this is the case the floor exists to reject; if the unfloored fit also "
        "rejects it, the test no longer demonstrates anything")


def test_no_trend_says_how_many_walks_would_settle_it():
    """"No trend" alone reads as a verdict on training, not on sample size."""
    got = trend.fit([5.0, 5.6, 6.2, 6.8], "posture")
    assert not got["significant"]
    assert got["walks_needed"] and got["walks_needed"] > got["n"]
    assert "walks in total" in got["summary"]


def test_personal_best_is_flagged():
    assert trend.fit([4.0, 5.0, 6.0], "posture")["is_personal_best"]
    assert not trend.fit([4.0, 9.0, 6.0], "posture")["is_personal_best"]


def test_a_trend_summary_never_claims_causation():
    for values in ([5.0 + 0.6 * i for i in range(8)],
                   [8.0 - 0.6 * i for i in range(8)],
                   [5.0, 5.2, 4.9, 5.1, 5.3, 5.0]):
        text = trend.fit(values, "posture")["summary"].lower()
        for banned in ("caused", "because of", "thanks to", "proves"):
            assert banned not in text


def test_non_finite_input_is_rejected():
    assert trend.fit([5.0, float("nan"), 6.0], "posture") is None


def test_overall_trend_uses_the_quadrature_noise_floor():
    got = trend.overall([50.0 + 3.0 * i for i in range(8)])
    assert got["direction"] == "improving"
    # Halved because overall_threshold() is a *difference* floor.
    assert got["noise_floor"] == pytest.approx(
        significance.overall_threshold() / 1.41, abs=0.02)


def test_per_criterion_skips_criteria_with_too_little_history():
    series = [{"posture": 5.0 + i, "balance": 6.0} for i in range(4)]
    got = trend.per_criterion(series, ["posture", "balance", "rhythm"])
    assert "posture" in got
    assert "rhythm" not in got, "absent criteria must not appear"


# --------------------------------------------------------------------------- #
# Identifiability
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not confidence.available(),
                    reason="leakage audit has not been run in this checkout")
def test_every_criterion_has_an_identifiability_tier():
    table = confidence.summary()["per_criterion"]
    for c in CRITERIA:
        assert c in table, c
        assert table[c]["tier"] in {"strong", "moderate", "weak"}


@pytest.mark.skipif(not confidence.available(), reason="audit not run")
def test_the_criteria_the_audit_found_weak_are_reported_weak():
    """These three lost most of their accuracy on independent latents.

    Pinned so a future retrain that quietly restores the leakage - or drops the
    audit - is visible rather than silent.
    """
    for c in ("rhythm", "hip_movement", "symmetry"):
        assert confidence.tier(c) == "weak", c
    for c in ("foot_placement", "shoulder_alignment", "confidence"):
        assert confidence.tier(c) == "strong", c


@pytest.mark.skipif(not confidence.available(), reason="audit not run")
def test_reported_accuracy_is_never_below_independent_by_the_stated_gap():
    """Leakage is the reported-minus-independent gap; it must be self-consistent."""
    for c, e in confidence.summary()["per_criterion"].items():
        if e["r2_as_reported"] is None or e["leakage"] is None:
            continue
        assert e["leakage"] == pytest.approx(
            e["r2_as_reported"] - e["r2_independent"], abs=0.002), c


def test_a_missing_audit_degrades_to_unknown(monkeypatch, tmp_path):
    """A fresh checkout has not run the audit; that must not break scoring."""
    monkeypatch.setattr(confidence, "AUDIT_PATH", tmp_path / "absent.csv")
    monkeypatch.setattr(confidence, "_cache", None)
    assert confidence.available() is False
    assert confidence.tier("rhythm") == confidence.UNKNOWN
    assert confidence.summary() == {"available": False}
    monkeypatch.setattr(confidence, "_cache", None)
