"""Is a difference between two walks real, or is it noise?

Comparing two scores invites reading meaning into every decimal. It should not:
each score carries the model's own error, and the *difference* of two noisy
measurements is noisier still. Subtracting two estimates each with standard
error s gives a difference with error s*sqrt(2), so a gap has to clear roughly
1.4 times the per-criterion error before it says anything.

The error used is the **per-criterion** out-of-fold MAE from training, not the
0.902 mean. That distinction matters here: foot placement is measured to 0.33
while symmetry is only good to 1.35, so a 0.9 change means opposite things for
the two. Using one global figure would call real movement in foot placement
noise, and call noise in symmetry real.
"""
from __future__ import annotations

import csv
from pathlib import Path

from ..config import CRITERIA, REPORT_DIR

#: Difference of two independent estimates: sqrt(2) ~ 1.41.
DIFFERENCE_FACTOR = 1.41

#: Used when the per-criterion table is missing, e.g. before any training run.
#: The mean out-of-fold MAE across the twelve criteria.
FALLBACK_MAE = 0.902

_cache: dict[str, float] | None = None


def per_criterion_mae() -> dict[str, float]:
    """Out-of-fold MAE per criterion, from the last training run."""
    global _cache
    if _cache is not None:
        return _cache

    table: dict[str, float] = {}
    path = Path(REPORT_DIR) / "model_comparison.csv"
    if path.exists():
        try:
            with path.open(newline="", encoding="utf-8") as fh:
                for row in csv.DictReader(fh):
                    name = row.get("criterion") or row.get("")
                    if name in CRITERIA and row.get("MAE"):
                        table[name] = float(row["MAE"])
        except Exception:
            table = {}

    _cache = {c: table.get(c, FALLBACK_MAE) for c in CRITERIA}
    return _cache


def threshold(criterion: str) -> float:
    """Smallest difference in this criterion worth reporting as a change."""
    return round(per_criterion_mae().get(criterion, FALLBACK_MAE) * DIFFERENCE_FACTOR, 2)


def is_significant(criterion: str, delta: float) -> bool:
    return abs(delta) >= threshold(criterion)


def overall_threshold() -> float:
    """Smallest change in the 0-100 overall worth reporting.

    The overall is a weighted sum of twelve separate estimates, so its error
    combines **in quadrature**, not linearly. Summing the twelve errors instead
    assumes every model happens to be wrong in the same direction at the same
    time, which put the floor at +/-12.5 - large enough to dismiss genuine
    improvement as noise.

    Quadrature assumes the twelve errors are independent. They are not
    perfectly so: the models share input features and training data, so the
    true floor sits somewhat above this and somewhere below the linear sum.
    """
    import math

    from ..config import CRITERION_WEIGHTS

    var = sum((CRITERION_WEIGHTS[c] * per_criterion_mae()[c]) ** 2 for c in CRITERIA)
    return round(math.sqrt(var) * 10 * DIFFERENCE_FACTOR, 1)
