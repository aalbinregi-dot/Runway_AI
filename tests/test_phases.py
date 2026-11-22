"""Per-phase measurements: split geometry, and refusing to narrate noise."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from runway_ai.config import CORE_13
from runway_ai.data.loaders import preprocess
from runway_ai.data.schema import PoseSequence
from runway_ai.features import timeline as tl
from runway_ai.scoring import phases


def _seq(n_frames: int) -> PoseSequence:
    rng = np.random.default_rng(0)
    xy = rng.normal(0.5, 0.03, (n_frames, 13, 2))
    return PoseSequence("t", list(CORE_13), xy, np.zeros((n_frames, 13)),
                        np.ones((n_frames, 13)), 30.0,
                        meta={"frame_size": [1080, 1920]})


def _analysed(n_frames: int = 120):
    prepared = preprocess(copy.deepcopy(_seq(n_frames)))
    return phases.analyse(prepared, tl.build(prepared, []))


def test_a_clip_too_short_to_split_returns_nothing():
    assert phases.split(_seq(10)) is None
    assert _analysed(12) is None


def test_split_covers_every_frame_exactly_once():
    seq = _seq(100)
    parts = phases.split(seq)
    assert [p.meta["phase"] for p in parts] == list(phases.PHASES)
    assert sum(p.n_frames for p in parts) == seq.n_frames
    ranges = [p.meta["frame_range"] for p in parts]
    assert ranges[0][0] == 0 and ranges[-1][1] == seq.n_frames
    for a, b in zip(ranges, ranges[1:]):
        assert a[1] == b[0], "phases must abut with no gap or overlap"


def test_every_trace_reports_all_three_phases():
    got = _analysed()
    assert got and got["traces"]
    for row in got["traces"]:
        assert set(row["per_phase"]) == set(phases.PHASES)
        assert all(v is not None for v in row["per_phase"].values())


def test_only_validated_traces_may_claim_a_change():
    """The sweep found only step width could beat the camera-distance artifact.

    Everything else is still shown - the per-phase numbers are the point - but
    must not assert a change, because widening the gate never got the false rate
    below ~16% while power collapsed to nothing.
    """
    got = _analysed()
    for row in got["traces"]:
        if row["label"] not in phases.VALIDATED_TRACES:
            assert row["can_claim"] is False
            assert row["changed"] is False
    assert all(c["label"] in phases.VALIDATED_TRACES for c in got["changes"])


def test_step_width_is_the_validated_trace():
    """Pinned: a retrain that silently changes this should be visible."""
    assert phases.VALIDATED_TRACES == frozenset({"Step width"})


def test_the_payload_states_what_it_is_not():
    """The note must say these are measurements, not per-phase scores."""
    got = _analysed()
    assert "not scores" in got["note"]
    assert "camera" in got["caveat"] or "further away" in got["caveat"]


def test_changes_carry_the_artifact_they_were_corrected_for():
    got = _analysed()
    for row in got["traces"]:
        assert "expected_artifact" in row
        assert "threshold" in row
        assert row["basis"]


def test_a_missing_null_table_degrades_rather_than_raising(monkeypatch):
    """A fresh checkout has not run scripts/phase_null.py."""
    monkeypatch.setattr(phases, "_null_cache", {})
    got = _analysed()
    assert got is not None
    assert got["null_measured"] is False
    for row in got["traces"]:
        assert "not yet measured" in row["basis"]
    monkeypatch.setattr(phases, "_null_cache", None)


def test_no_timeline_means_no_phase_view():
    prepared = preprocess(copy.deepcopy(_seq(120)))
    assert phases.analyse(prepared, {"traces": {}}) is None


@pytest.mark.parametrize("multiplier", [phases.CRITICAL_MULTIPLIER])
def test_the_gate_is_wider_than_a_nominal_95_percent(multiplier):
    """1.96 gave 0.80 false claims per clip; the calibrated value is wider."""
    assert multiplier > 1.96
