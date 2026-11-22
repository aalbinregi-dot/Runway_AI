"""Structured video quality assessment.

Evaluates how reliable a recorded clip is likely to be, based entirely on the
feature keys already produced by ``runway_ai.features.builder``.  No new
extraction step is needed.

Verdicts
--------
``GOOD``
    All checks pass; the score should be taken at face value.
``ACCEPTABLE``
    One or more minor issues (mild landmark gaps, borderline visibility,
    slightly short clip). The score is usable but comes with caveats.
``POOR``
    A significant issue that materially degrades scoring accuracy (severe gaps,
    low visibility, very few steps detected, extreme perspective distortion).
    The score should be treated as a rough guide only.
``UNANALYZABLE``
    The recording contains so little usable data that any score would describe
    the gap-filler rather than the walk. No score is issued in this case; see
    ``assess.assert_usable``.

All thresholds are derived from the project's own quality-control features
(``qc_*``) and the scoring features already computed by the pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


# ---------------------------------------------------------------------------
# Thresholds (all in the same units as the corresponding feature)
# ---------------------------------------------------------------------------
_GAP_UNUSABLE   = 0.90   # above this → UNANALYZABLE
_GAP_POOR       = 0.50   # above this → POOR
_GAP_ACCEPTABLE = 0.25   # above this → ACCEPTABLE

_VIS_POOR       = 0.55   # below this → POOR
_VIS_ACCEPTABLE = 0.70   # below this → ACCEPTABLE

_STEPS_MIN      = 4      # fewer → POOR (timing/symmetry not meaningful)
_DURATION_MIN   = 3.0    # shorter → POOR (seconds)

_SCALE_RATIO_MAX  = 4.0  # above this → POOR (extreme camera approach)
_TRUNK_LEAN_MAX   = 90.0 # above this → POOR (lower body out of frame)


@dataclass
class VideoQuality:
    """Outcome of evaluating a clip's recording quality.

    All fields are derived from the existing feature vector; this dataclass
    is purely a structured representation of the same information that
    ``assess.check_quality`` surfaced as plain strings.
    """

    #: Overall verdict: one of "GOOD", "ACCEPTABLE", "POOR", "UNANALYZABLE".
    overall: str

    #: Mean pose-landmark visibility (0–1).
    landmark_visibility: float

    #: Number of step events detected by the gait analyser.
    gait_cycles: int

    #: True when the gap fraction suggests occlusion or landmark dropout.
    has_occlusion: bool

    #: True when the camera angle appears frontal (trunk lean < 90°).
    camera_angle_ok: bool

    #: Clip duration in seconds.
    duration_s: float

    #: Fraction of landmark coordinates that were interpolated (0–1).
    gap_fraction: float

    #: Human-readable list of problems found.
    warnings: list[str] = field(default_factory=list)

    #: What to change for the next recording.
    recommendations: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Severity order: UNANALYZABLE > POOR > ACCEPTABLE > GOOD
# ---------------------------------------------------------------------------
_RANK = {"UNANALYZABLE": 3, "POOR": 2, "ACCEPTABLE": 1, "GOOD": 0}


def evaluate(feats: dict) -> VideoQuality:
    """Return a structured quality verdict for one clip.

    Parameters
    ----------
    feats:
        Feature dict as produced by ``runway_ai.features.builder.build``.
        Missing keys are treated as zero / best-case defaults so the function
        never raises.
    """
    gap      = float(feats.get("qc_gap_fraction",          0.0))
    vis      = float(feats.get("qc_mean_visibility",       1.0))
    steps    = int(feats.get("rhythm_n_steps",             0))
    dur      = float(feats.get("qc_duration_s",            0.0))
    scale    = float(feats.get("qc_torso_scale_ratio",     1.0))
    lean     = float(feats.get("posture_trunk_lean_mean",  0.0))

    warnings: list[str] = []
    recommendations: list[str] = []
    verdict = "GOOD"

    # ------------------------------------------------------------------ #
    # UNANALYZABLE — no measurement left
    # ------------------------------------------------------------------ #
    if gap >= _GAP_UNUSABLE:
        warnings.append(
            f"{gap * 100:.0f}% of landmark coordinates are missing. Any score "
            "would describe the gap-filler, not the walk."
        )
        recommendations.append(
            "Re-record with the entire body — head to feet — in frame and "
            "well-lit throughout."
        )
        verdict = "UNANALYZABLE"

    # ------------------------------------------------------------------ #
    # POOR — major quality issues
    # ------------------------------------------------------------------ #
    if gap >= _GAP_POOR and verdict != "UNANALYZABLE":
        _downgrade(verdict, "POOR")
        warnings.append(
            f"{gap * 100:.0f}% of landmarks missing or interpolated — "
            "scores for timing, symmetry and stride length are unreliable."
        )
        recommendations.append("Ensure the full body stays in frame.")
        verdict = _downgrade(verdict, "POOR")

    if vis < _VIS_POOR:
        warnings.append(
            f"Mean pose confidence {vis:.0%} — the pose estimator is struggling."
        )
        recommendations.append(
            "Improve lighting (face a window or use a ring light) and film "
            "against a plain, contrasting background."
        )
        verdict = _downgrade(verdict, "POOR")

    if steps < _STEPS_MIN:
        warnings.append(
            f"Only {steps} step{'s' if steps != 1 else ''} detected — "
            "timing and symmetry scores need at least four."
        )
        recommendations.append(
            "Walk the full length of the runway (aim for 6–10 seconds, 6+ strides)."
        )
        verdict = _downgrade(verdict, "POOR")

    if dur < _DURATION_MIN:
        warnings.append(
            f"Clip is only {dur:.1f} s — less than the 3 s minimum for stable "
            "feature estimation."
        )
        recommendations.append(
            "Record at least 4–6 seconds of continuous walking."
        )
        verdict = _downgrade(verdict, "POOR")

    if scale > _SCALE_RATIO_MAX:
        warnings.append(
            f"Subject distance changes sharply (scale ratio {scale:.1f}×). "
            "Perspective features may be distorted."
        )
        recommendations.append(
            "Film from further back so the subject fills the frame evenly "
            "throughout the walk."
        )
        verdict = _downgrade(verdict, "POOR")

    camera_ok = lean <= _TRUNK_LEAN_MAX
    if not camera_ok:
        warnings.append(
            f"Trunk lean is {lean:.0f}° — the lower body is likely out of frame "
            "and the hips could not be located."
        )
        recommendations.append(
            "Film from further back with the whole body, feet included, visible."
        )
        verdict = _downgrade(verdict, "POOR")

    # ------------------------------------------------------------------ #
    # ACCEPTABLE — minor issues
    # ------------------------------------------------------------------ #
    if _GAP_ACCEPTABLE <= gap < _GAP_POOR and verdict == "GOOD":
        warnings.append(
            f"{gap * 100:.0f}% of landmarks were interpolated. "
            "Scores are usable but come with caveats."
        )
        verdict = _downgrade(verdict, "ACCEPTABLE")

    if _VIS_ACCEPTABLE <= vis < _VIS_POOR and verdict == "GOOD":
        warnings.append(
            f"Pose confidence {vis:.0%} — borderline. "
            "Improve lighting if possible."
        )
        verdict = _downgrade(verdict, "ACCEPTABLE")

    has_occlusion = gap >= _GAP_ACCEPTABLE

    return VideoQuality(
        overall=verdict,
        landmark_visibility=round(vis, 3),
        gait_cycles=steps,
        has_occlusion=has_occlusion,
        camera_angle_ok=camera_ok,
        duration_s=round(dur, 2),
        gap_fraction=round(gap, 3),
        warnings=warnings,
        recommendations=list(dict.fromkeys(recommendations)),  # deduplicate
    )


def _downgrade(current: str, new: str) -> str:
    """Return whichever verdict is more severe."""
    return current if _RANK[current] >= _RANK[new] else new
