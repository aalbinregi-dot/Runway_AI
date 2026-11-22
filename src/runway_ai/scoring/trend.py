"""Is a criterion actually trending, or is it just noise across several walks?

Why not first-versus-last
-------------------------
The obvious way to show progress is to subtract the first walk's score from the
latest one. It has two problems, and both get worse the more a user trains.

It throws away data. With eight walks, a first-versus-last delta uses two of
them and ignores six. The six discarded walks are exactly what tells you whether
the change is a trend or a wobble.

And it is the *noisiest* possible estimator. Each score carries about 0.9/10 of
model error, so a difference of two of them carries roughly 1.27 - the errors
add in quadrature. A regression through all eight points averages that error
down instead of compounding it.

So this fits a least-squares line through every walk and reports the slope with
a confidence interval. A trend is only called real when the interval excludes
zero.

The noise floor matters more than the residuals
-----------------------------------------------
Ordinary regression estimates uncertainty from how far the points sit off the
fitted line. With three or four walks that estimate is itself unreliable: three
points can happen to be nearly collinear, the residual spread comes out tiny,
and a meaningless slope is declared significant.

But this project already *measured* its own error, per criterion, out of fold.
That number does not depend on how many walks a user has, and it is a floor no
amount of luck in the sampling can go below. The standard error used here is
therefore the larger of the two - the residual estimate and the known
measurement error. It makes the test strictly more conservative, and it is the
reason a three-walk "trend" almost never passes.

What it still cannot tell you
-----------------------------
That training caused the change. Walks are filmed on different days, in
different shoes, from slightly different angles, and there is no control group.
A confirmed upward trend means the improvement is too large and too consistent
to be measurement error - not that any particular practice produced it.
"""
from __future__ import annotations

import numpy as np

from . import significance

#: Fewest walks worth fitting a line through. Two points define a line exactly,
#: leaving no degrees of freedom to estimate uncertainty from, so a two-walk
#: "trend" cannot be tested at all - that case belongs to the plain
#: walk-versus-walk comparison.
MIN_WALKS = 3

#: Two-sided 95%.
CONFIDENCE = 0.95


def fit(values: list[float], criterion: str | None = None) -> dict | None:
    """Fit a trend line through one criterion's scores, oldest first.

    ``criterion`` supplies the measurement-error floor. Without it only the
    residual spread is used, which is the less conservative choice and is why
    every caller in this project passes it.
    """
    y = np.asarray([v for v in values if v is not None], dtype=float)
    n = y.size
    if n < MIN_WALKS or not np.all(np.isfinite(y)):
        return None

    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    sxx = float(xc @ xc)
    if sxx <= 0:                                    # cannot happen for n >= 2
        return None

    slope = float(xc @ (y - y.mean()) / sxx)
    intercept = float(y.mean() - slope * x.mean())
    resid = y - (intercept + slope * x)

    # Residual standard deviation, then the floor described in the module notes.
    resid_sd = float(np.sqrt(resid @ resid / (n - 2))) if n > 2 else 0.0
    floor = (significance.per_criterion_mae().get(criterion, significance.FALLBACK_MAE)
             if criterion else 0.0)
    sd = max(resid_sd, floor)
    se_slope = sd / np.sqrt(sxx)

    from scipy import stats
    t_crit = float(stats.t.ppf(0.5 + CONFIDENCE / 2, n - 2))

    # Reported over the whole span rather than per walk: "up 1.4 across six
    # walks" is a quantity a user can check against their own scores, whereas
    # "slope 0.28 per walk" is not.
    span = n - 1
    total = slope * span
    half_width = t_crit * se_slope * span
    lo, hi = total - half_width, total + half_width

    real = (lo > 0) or (hi < 0)
    direction = "flat" if not real else ("improving" if total > 0 else "declining")

    needed = None if real else _walks_needed(slope, sd, n)

    return {
        "n": n,
        "slope_per_walk": round(slope, 3),
        "total_change": round(total, 2),
        "ci": [round(lo, 2), round(hi, 2)],
        "confidence": CONFIDENCE,
        "significant": bool(real),
        "direction": direction,
        "noise_floor": round(floor, 2),
        "residual_sd": round(resid_sd, 2),
        "first": round(float(y[0]), 2),
        "latest": round(float(y[-1]), 2),
        "best": round(float(y.max()), 2),
        "is_personal_best": bool(y[-1] >= y.max() - 1e-9),
        "walks_needed": needed,
        "summary": _summary(criterion, total, (lo, hi), real, direction, n, needed),
    }


#: Stop looking beyond this. A change so slight it needs a hundred walks to
#: confirm is better described as "too small to detect" than as a target.
MAX_PROJECTION = 40


def _walks_needed(slope: float, sd: float, n: int) -> int | None:
    """How many walks in total before a trend this size could be confirmed.

    "No trend detected" is a dead end on its own - it gives a user nothing to do
    and reads like a verdict on their training rather than on the sample size.
    Holding the observed rate of change and the noise fixed, this asks how many
    walks it would take for the interval to clear zero.

    The answer is a projection, not a promise: it assumes the current rate keeps
    up, which is exactly what more walks are meant to find out.
    """
    if not np.isfinite(slope) or abs(slope) < 1e-9 or sd <= 0:
        return None

    from scipy import stats

    for m in range(n + 1, MAX_PROJECTION + 1):
        sxx = m * (m * m - 1) / 12.0            # evenly spaced walk indices
        t_crit = float(stats.t.ppf(0.5 + CONFIDENCE / 2, m - 2))
        if t_crit * sd / np.sqrt(sxx) < abs(slope):
            return m
    return None


def _summary(criterion, total, ci, real, direction, n, needed=None) -> str:
    lo, hi = ci
    if not real:
        text = (f"No trend that stands out from measurement error across {n} "
                f"walks. The fitted change is {total:+.1f}, but anything from "
                f"{lo:+.1f} to {hi:+.1f} fits the data equally well, and that "
                f"range includes no change at all.")
        if needed:
            text += (f" At this rate it would take about {needed} walks in total "
                     f"({needed - n} more) to tell the difference.")
        return text
    word = "up" if direction == "improving" else "down"
    return (f"Consistently {word} across {n} walks: {total:+.1f} overall, and the "
            f"data supports somewhere between {lo:+.1f} and {hi:+.1f}. That range "
            f"excludes zero, so this is larger than the measurement error rather "
            f"than a run of luck.")


def per_criterion(series: list[dict], criteria) -> dict:
    """Fit a trend for each criterion across a user's walk history.

    ``series`` is oldest-first, each entry mapping criterion -> score.
    """
    out = {}
    for c in criteria:
        values = [row.get(c) for row in series if row.get(c) is not None]
        got = fit(values, criterion=c)
        if got:
            out[c] = got
    return out


def overall(series: list[float]) -> dict | None:
    """Trend in the 0-100 overall.

    The floor is the overall's own noise threshold, which combines the twelve
    criterion errors in quadrature rather than summing them - see
    ``significance.overall_threshold``.
    """
    y = [v for v in series if v is not None]
    if len(y) < MIN_WALKS:
        return None

    n = len(y)
    x = np.arange(n, dtype=float)
    xc = x - x.mean()
    sxx = float(xc @ xc)
    arr = np.asarray(y, dtype=float)
    slope = float(xc @ (arr - arr.mean()) / sxx)
    resid = arr - (arr.mean() + slope * (x - x.mean()))
    resid_sd = float(np.sqrt(resid @ resid / (n - 2)))

    # The overall threshold is a *difference* floor; halve it to get a
    # single-measurement standard deviation on the same footing as resid_sd.
    floor = significance.overall_threshold() / 1.41
    sd = max(resid_sd, floor)

    from scipy import stats
    t_crit = float(stats.t.ppf(0.5 + CONFIDENCE / 2, n - 2))
    span = n - 1
    total = slope * span
    half = t_crit * (sd / np.sqrt(sxx)) * span
    lo, hi = total - half, total + half
    real = (lo > 0) or (hi < 0)

    return {
        "n": n,
        "total_change": round(total, 1),
        "ci": [round(lo, 1), round(hi, 1)],
        "significant": bool(real),
        "direction": "flat" if not real else ("improving" if total > 0 else "declining"),
        "first": round(float(arr[0]), 1),
        "latest": round(float(arr[-1]), 1),
        "best": round(float(arr.max()), 1),
        "is_personal_best": bool(arr[-1] >= arr.max() - 1e-9),
        "noise_floor": round(floor, 2),
    }


# --------------------------------------------------------------------------- #
# Personalised progress report (Phase 5)
# --------------------------------------------------------------------------- #
def personalised_report(
    user_id: int,
    current,        # Assessment object
    store,          # runway_ai.db.store module
    n_sessions: int = 5,
    db_path=None,
) -> dict:
    """Annotate an assessment with deltas against the user's personal EMA baseline.

    Each criterion is labelled with its delta vs. the EMA baseline and whether
    that delta is significant (above the per-criterion measurement noise floor).

    Parameters
    ----------
    user_id:
        Database user ID.
    current:
        The ``Assessment`` object from ``assess.assess()``.
    store:
        The ``runway_ai.db.store`` module (passed in to avoid a circular
        import — this module is already imported by store indirectly).
    n_sessions:
        Window size for the EMA baseline lookup.
    db_path:
        Optional database path override.

    Returns
    -------
    dict with keys::

        {
            "has_baseline": bool,
            "per_criterion": {
                criterion: {
                    "current":     float,
                    "baseline":    float | None,
                    "delta":       float | None,
                    "significant": bool,
                    "direction":   str,   # "improving" | "declining" | "stable"
                    "label":       str,   # human-readable sentence
                }
            },
            "narrative": list[str],    # 1-3 highlight sentences for the UI
            "personal_bests": {        # criteria where current == historical best
                criterion: float
            },
        }
    """
    per = {}
    narrative: list[str] = []
    personal_bests: dict[str, float] = {}

    has_baseline = False
    for crit in current.scores:
        baseline_val = store.get_user_baseline(
            user_id, crit, n_sessions=n_sessions, db_path=db_path
        )
        current_val = current.scores[crit]
        best = store.get_historical_best(user_id, crit, db_path=db_path)

        delta = None
        sig = False
        direction = "stable"
        label = "No previous sessions to compare against."

        if baseline_val is not None:
            has_baseline = True
            delta = round(current_val - baseline_val, 2)
            sig = significance.is_significant(crit, delta)
            if sig and delta > 0:
                direction = "improving"
                thresh = significance.threshold(crit)
                label = (
                    f"Up {delta:+.1f} since your recent average — larger than "
                    f"the ±{thresh:.1f} measurement noise."
                )
            elif sig and delta < 0:
                direction = "declining"
                thresh = significance.threshold(crit)
                label = (
                    f"Down {delta:+.1f} since your recent average — larger than "
                    f"the ±{thresh:.1f} measurement noise. Worth checking the "
                    f"filming angle and conditions match your earlier sessions."
                )
            else:
                label = (
                    f"Moved {delta:+.1f} — within measurement noise; no confirmed change."
                )

        if best is not None and abs(current_val - best["score"]) < 1e-6:
            personal_bests[crit] = round(current_val, 2)

        per[crit] = {
            "current":     round(current_val, 2),
            "baseline":    round(baseline_val, 3) if baseline_val is not None else None,
            "delta":       delta,
            "significant": sig,
            "direction":   direction,
            "label":       label,
        }

    # Build a short narrative (shown at the top of the result card).
    if has_baseline:
        improvers = [c for c, v in per.items()
                     if v["direction"] == "improving"]
        decliners = [c for c, v in per.items()
                     if v["direction"] == "declining"]
        if improvers:
            from ..config import CRITERION_LABELS
            names = ", ".join(CRITERION_LABELS.get(c, c) for c in improvers[:2])
            narrative.append(
                f"Confirmed improvement in {names}"
                + (" and others." if len(improvers) > 2 else ".")
            )
        if decliners:
            from ..config import CRITERION_LABELS
            names = ", ".join(CRITERION_LABELS.get(c, c) for c in decliners[:2])
            narrative.append(
                f"Notable decline in {names} — worth reviewing filming conditions."
            )
        if personal_bests:
            from ..config import CRITERION_LABELS
            names = ", ".join(
                CRITERION_LABELS.get(c, c) for c in list(personal_bests)[:2]
            )
            narrative.append(f"Personal best: {names}.")
        if not narrative:
            narrative.append(
                "Scores are within measurement noise of your recent average — "
                "no confirmed change this session."
            )
    else:
        narrative.append(
            "This is your first recorded session. "
            "Walk again to start tracking your progress."
        )

    return {
        "has_baseline":   has_baseline,
        "per_criterion":  per,
        "narrative":      narrative,
        "personal_bests": personal_bests,
    }
