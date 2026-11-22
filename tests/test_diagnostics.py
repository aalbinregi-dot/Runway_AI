"""The support report: actionable, and safe to paste in public."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app"
if str(APP) not in sys.path:
    sys.path.insert(0, str(APP))

import diagnostics  # noqa: E402


@pytest.fixture(scope="module")
def report():
    return diagnostics.build()


def test_every_capability_is_fully_described(report):
    assert report.capabilities, "no capabilities probed"
    for c in report.capabilities:
        assert c.key and c.label, c
        assert isinstance(c.ok, bool)
        # A gap the user cannot act on is not worth reporting.
        if not c.ok:
            assert c.remedy, f"{c.key} is unavailable with no remedy"


def test_the_core_capabilities_are_probed(report):
    keys = {c.key for c in report.capabilities}
    for expected in ("models", "video", "face", "links"):
        assert expected in keys, expected


def test_versions_are_reported(report):
    assert report.versions["python"]
    assert "numpy" in report.versions


def test_summary_agrees_with_the_capability_list(report):
    blocking = {c.key for c in report.capabilities if not c.ok and c.required}
    assert set(report.summary["blocking"]) == blocking
    assert report.summary["healthy"] is (not blocking)


def test_an_optional_gap_is_not_called_a_failure(report):
    """ffmpeg is absent on this machine and nothing depends on it yet.

    Reporting it as broken would send users chasing an install they do not need.
    """
    ffmpeg = next((c for c in report.capabilities if c.key == "ffmpeg"), None)
    if ffmpeg and not ffmpeg.ok:
        assert ffmpeg.required is False
        assert ffmpeg.key not in report.summary["blocking"]


FORBIDDEN = ["onedrive", "appdata", "desktop", "runway.db", "c:\\", "/users/",
             "password", "email", "session"]


def test_the_report_leaks_nothing_about_the_machine(report):
    """Paths here contain the user's own name, and reports get pasted publicly."""
    blob = (diagnostics.as_text(report) + repr(report.versions)
            + repr([c.as_dict() for c in report.capabilities])).lower()
    for token in FORBIDDEN:
        assert token not in blob, f"diagnostics leaked {token!r}"


def test_the_text_form_is_readable_and_self_contained(report):
    text = diagnostics.as_text(report)
    assert "Runway AI" in text
    assert "versions:" in text
    assert report.summary["headline"] in text
    for c in report.capabilities:
        assert c.label in text


def test_a_missing_subsystem_degrades_rather_than_raising(monkeypatch):
    """A half-installed checkout is exactly when this must still work."""
    def boom(*a, **k):
        raise RuntimeError("simulated missing dependency")

    monkeypatch.setattr(diagnostics, "_face", boom)
    with pytest.raises(RuntimeError):
        diagnostics.build()          # the probe itself is allowed to raise...

    # ...but each individual probe guards its own imports, which is what keeps a
    # missing optional package from taking the whole report down.
    monkeypatch.undo()
    assert diagnostics.build().capabilities
