"""Turn a feature vector into a complete assessment.

Combines the learned per-criterion models with the deterministic rubric. The
models supply the score; the rubric supplies the *reason*, because a fitted
regressor can tell you a walk scores 4.2 for balance but not which measurement
put it there. Every reported fault is therefore traceable to a named
measurement and its tolerance band.

If no trained model is present the rubric alone is used, so the system degrades
to a working (if blunter) scorer rather than failing.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict

import numpy as np

from ..config import CRITERIA, CRITERION_LABELS, CRITERION_WEIGHTS
from .coverage import check as coverage_check, warnings_from
from .rubric import RUBRIC, grade, overall_score, rubric_scores, score_criterion


@dataclass
class Finding:
    """One diagnosed fault, tied to the measurement that produced it."""

    criterion: str
    feature: str
    measured: float
    ideal: float
    band_score: float          # 0-1, how well this band was satisfied
    severity: float            # 0-1, weighted shortfall; drives ranking
    message: str

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Assessment:
    scores: dict[str, float]                  # criterion -> 0-10
    overall: float                            # 0-100
    grade: str
    findings: list[Finding] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    source: str = "model"                     # "model" or "rubric"
    quality_warnings: list[str] = field(default_factory=list)
    coverage: dict = field(default_factory=dict)   # training-distribution check
    #: Every rubric band behind every criterion, measured beside its target,
    #: with the criterion's weight expressed in marks. See scoring.explain.
    breakdown: dict = field(default_factory=dict)
    #: The measured quantities themselves, grouped and labelled.
    measurements: list = field(default_factory=list)
    #: Per-criterion prediction uncertainty (fold + ensemble disagreement).
    uncertainty: dict = field(default_factory=dict)
    #: Structured recording-quality verdict.
    video_quality: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "scores": {k: round(v, 2) for k, v in self.scores.items()},
            "overall": round(self.overall, 1),
            "grade": self.grade,
            "source": self.source,
            "findings": [f.as_dict() for f in self.findings],
            "strengths": self.strengths,
            "quality_warnings": self.quality_warnings,
            "coverage": self.coverage,
            "breakdown": self.breakdown,
            "measurements": self.measurements,
            "uncertainty": self.uncertainty,
            "video_quality": self.video_quality,
            # How well each criterion is identifiable at all, as distinct from
            # how this particular clip was recorded. Twelve scores rendered
            # identically imply twelve equally trustworthy measurements, and the
            # leakage audit showed that is not true - see scoring.confidence.
            "measurement_confidence": _confidence_summary(),
        }


# --------------------------------------------------------------------------- #
# Recording-quality gate
# --------------------------------------------------------------------------- #
def _confidence_summary() -> dict:
    """Per-criterion identifiability, or an explicit absence of it.

    Imported lazily and failure-tolerant: the audit is an offline experiment and
    a fresh checkout has not run it, which must degrade to "unknown" rather than
    break every analysis.
    """
    try:
        from . import confidence
        return confidence.summary()
    except Exception:
        return {"available": False}


#: Above this fraction of interpolated landmarks there is no measurement left -
#: only the interpolator's guesses - so there is nothing to score.
UNUSABLE_GAP_FRACTION = 0.90


class UnusableRecording(ValueError):
    """The recording contains nothing measurable. Scoring it would be invention."""


def assert_usable(feats: dict) -> None:
    """Refuse to score a recording with no real measurements in it.

    ``check_quality`` warns about degraded input, which is the right response
    almost always - a caveated number still helps. But a keypoint file whose
    coordinates are entirely absent produced a confident 52.6/100 alongside a
    warning that 100% of the landmarks were missing. Those two statements cannot
    both be useful: if every value was interpolated from nothing, the score
    describes the interpolator, not the walk. The honest output is no score.
    """
    gap = feats.get("qc_gap_fraction", 0.0)
    if gap >= UNUSABLE_GAP_FRACTION:
        raise UnusableRecording(
            f"{gap * 100:.0f}% of the landmark coordinates are missing, so there "
            f"is nothing left to measure - any score would describe the gap-"
            f"filling rather than the walk. Check that the file has x and y "
            f"values, or re-record with the whole body in frame.")


def check_quality(feats: dict) -> list[str]:
    """Flag recording problems that make any score unreliable.

    Reporting a confident score on a two-second clip of a half-occluded subject
    would be worse than reporting nothing, so these conditions are surfaced
    alongside the result.
    """
    warnings = []
    if feats.get("qc_duration_s", 0) < 3.0:
        warnings.append(
            f"Clip is only {feats.get('qc_duration_s', 0):.1f}s; "
            "at least 4-6 seconds (4+ strides) gives a stable reading.")
    if feats.get("rhythm_n_steps", 0) < 4:
        warnings.append(
            "Fewer than four steps were detected - timing and symmetry scores "
            "are unreliable on this clip.")
    if feats.get("qc_mean_visibility", 1.0) < 0.70:
        warnings.append(
            "Pose confidence is low; check lighting and that the whole body is in frame.")
    if feats.get("qc_gap_fraction", 0.0) > 0.25:
        warnings.append(
            f"{feats['qc_gap_fraction'] * 100:.0f}% of landmarks were missing and "
            "had to be interpolated.")
    # A trunk "lean" beyond 90 degrees means the shoulder-to-hip vector points
    # downward - the pose estimator has put the hips above the shoulders. That
    # is not a posture fault, it is a detection failure, and it happens when the
    # lower body is out of frame and the hips are guessed. Reporting 140 degrees
    # of lean would be nonsense; saying the framing is wrong is actionable.
    if feats.get("posture_trunk_lean_mean", 0.0) > 90.0:
        warnings.append(
            "The lower body does not appear to be in frame - the hips could not "
            "be located, so posture and stride scores are not meaningful. Film "
            "from further back with the whole body, feet included, visible.")
    if feats.get("qc_torso_scale_ratio", 1.0) > 4.0:
        warnings.append(
            "The subject's distance changes very sharply; film from further back "
            "for a steadier view.")
    return warnings


# --------------------------------------------------------------------------- #
# Assessment
# --------------------------------------------------------------------------- #
#: A criterion at or above this is reported as a strength, never as a fault.
STRENGTH_SCORE = 7.5
#: Only criteria below this can raise findings.
FAULT_SCORE = 7.0


def diagnose(feats: dict, scores: dict[str, float],
             band_threshold: float = 0.65) -> tuple[list[Finding], list[str]]:
    """Explain the scores: which bands failed, and which criteria were strong.

    Findings are gated on the *criterion's own score*, not just on whether a
    rubric band tripped. The score comes from the learned model while the bands
    come from the rubric, and the two can disagree - without the gate a walk
    could be told its posture is 10/10 and simultaneously handed a posture
    correction, which reads as broken. The model's score decides whether there
    is a problem; the bands only explain what it is.

    Severity weights a band's shortfall by its weight inside the criterion, the
    criterion's weight in the overall score, and how far the criterion actually
    fell short - so the ranked list leads with the fault costing the most marks
    rather than merely the lowest raw band.
    """
    findings, strengths = [], []
    for crit in CRITERIA:
        score = scores.get(crit, 0.0)
        if score >= STRENGTH_SCORE:
            strengths.append(CRITERION_LABELS[crit])
        if score >= FAULT_SCORE:
            continue

        _, bands = score_criterion(crit, feats)
        total_w = sum(b.weight for b in RUBRIC[crit]) or 1.0
        for band, band_score in bands:
            if band_score < band_threshold and band.message:
                shortfall = 1.0 - band_score
                findings.append(Finding(
                    criterion=crit,
                    feature=band.feature,
                    measured=round(float(feats.get(band.feature, 0.0)), 4),
                    ideal=round(float(band.ideal), 4),
                    band_score=round(float(band_score), 3),
                    severity=round(float(
                        shortfall
                        * (band.weight / total_w)
                        * CRITERION_WEIGHTS[crit]
                        * (1.0 - score / 10.0)
                        * 12
                    ), 4),
                    message=band.message,
                ))

    findings.sort(key=lambda f: -f.severity)
    return findings, strengths


def assess(feats: dict, models: dict | None = None,
           fold_models: dict | None = None) -> Assessment:
    """Score and explain one walk.

    Parameters
    ----------
    feats:
        Feature dict produced by ``runway_ai.features.builder``.
    models:
        Per-criterion model bundles from ``train.load_models()``. When absent
        the deterministic rubric is used as a fallback scorer.
    fold_models:
        Optional per-fold model bundles from ``train.load_fold_models()``.
        When supplied, per-criterion uncertainty (fold + ensemble disagreement)
        is attached to the result.
    """
    from ..models.train import predict_scores

    if models:
        scores = predict_scores(feats, models)
        source = "model"
        # Fill any criterion the model set does not cover.
        rub = rubric_scores(feats)
        for crit in CRITERIA:
            scores.setdefault(crit, rub[crit])
    else:
        scores = rubric_scores(feats)
        source = "rubric"

    scores = {c: float(np.clip(scores[c], 0.0, 10.0)) for c in CRITERIA}
    findings, strengths = diagnose(feats, scores)
    total = overall_score(scores)

    # A confident-looking number on a clip unlike anything in training is worse
    # than no number, so the extrapolation check rides alongside every result.
    cov = coverage_check(feats)

    # The bands and the measurements are built here, from the same feature dict
    # and the same rubric that produced the scores, rather than being recomputed
    # by whoever renders them. A second computation is a second chance for the
    # explanation to disagree with the number it is explaining.
    from . import explain

    # --- Uncertainty quantification (Phase 3) ---------------------------------
    uncertainty: dict = {}
    if models:
        try:
            from .uncertainty import predict_with_uncertainty
            uncertainty = predict_with_uncertainty(feats, models, fold_models)
        except Exception:
            pass  # Non-fatal: uncertainty is a bonus, not a gating concern.

    # --- Structured video quality (Phase 4) -----------------------------------
    try:
        from ..data.quality import evaluate as _quality_eval
        vq = _quality_eval(feats).as_dict()
    except Exception:
        vq = {}

    return Assessment(
        scores=scores,
        overall=total,
        grade=grade(total),
        findings=findings,
        strengths=strengths,
        source=source,
        quality_warnings=check_quality(feats) + warnings_from(cov),
        coverage=cov,
        breakdown=explain.breakdown(feats, scores),
        measurements=explain.measurements(feats),
        uncertainty=uncertainty,
        video_quality=vq,
    )
