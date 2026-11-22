"""Fit the rubric's tolerance bands to data instead of guessing them.

The band constants were first written by hand from gait-analysis norms, which
got the *direction* of every band right but not its scale - several were off by
enough to make the rubric's scores nearly useless (confidence was out by ~5
points on a 0-10 scale). Feature units here are torso lengths and degrees under
a specific normalisation, so the honest thing is to measure the constants.

For each band:

* ``ideal`` <- the median value of that feature among clips scoring in the top
  decile of the criterion, i.e. what good execution actually measures.
* ``tol``   <- the gap between the top-decile and bottom-decile medians. That is
  the deviation at which a walk has gone from excellent to poor, which is
  exactly the point where the band should have shed about half its marks.

One-sided bands (``lower``/``upper``) keep their direction; only the numbers
move. The result is written to ``models/rubric_calibration.json`` and picked up
automatically by :mod:`runway_ai.scoring.rubric`.

Usage:  python scripts/calibrate_rubric.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runway_ai.config import CRITERIA, MODEL_DIR, PROCESSED_DIR  # noqa: E402
from runway_ai.scoring.rubric import RUBRIC                      # noqa: E402


#: The falloff is 1/(1 + (excess/tol)^2), so a band sits at 0.10 when the excess
#: reaches 3 tolerances. Setting tol to the *full* good-to-poor spread would put
#: a bottom-decile walk at 0.5 - a middling mark for a bad walk. Dividing by 3
#: makes the same spread land near 0.1, so the band's output uses its full range.
FALLOFF_AT_POOR = 3.0


def calibrate(X: pd.DataFrame, Y: pd.DataFrame) -> dict:
    out: dict[str, dict] = {}
    for crit in CRITERIA:
        col = f"score_{crit}"
        if col not in Y.columns:
            continue
        y = Y[col]
        good = X[y >= y.quantile(0.90)]
        poor = X[y <= y.quantile(0.10)]

        bands = {}
        for band in RUBRIC[crit]:
            f = band.feature
            if f not in X.columns:
                continue
            g = float(np.median(good[f]))
            p = float(np.median(poor[f]))
            spread = abs(p - g)
            # Fall back to the baseline band tol if the synthetic spread is near zero or uninformative
            if spread < 1e-6:
                spread = abs(float(np.subtract(*np.percentile(good[f], [75, 25])))) or 1.0
            
            # Floor tolerance against hand-set biomechanical baseline to prevent razor-thin overfit
            cal_tol = max(round(spread / FALLOFF_AT_POOR, 6), band.tol * 0.75, abs(g) * 0.25, 0.05)
            # Retain domain-expert ideal if synthetic median deviates wildly
            ideal_val = g if abs(g - band.ideal) < max(band.ideal * 1.5, 5.0) else band.ideal

            bands[f] = {"ideal": round(ideal_val, 6),
                        "tol": round(cal_tol, 6),
                        "mode": band.mode, "weight": band.weight,
                        "separation": round(spread, 6)}
        out[crit] = bands
    return out


def fit_affine(X: pd.DataFrame, Y: pd.DataFrame, cal: dict) -> dict:
    """Least-squares rescale of each criterion's raw rubric score.

    Even with well-placed bands, a weighted average of rational falloffs does
    not naturally span 0-10: it compresses toward the middle, because a walk
    rarely fails every band at once. A per-criterion slope and intercept fixes
    the scale without touching which measurements drive the score or in which
    direction - the rubric stays fully transparent, it just reports on the same
    scale as the labels.
    """
    import runway_ai.scoring.rubric as R

    R.apply_calibration_dict(cal)
    for crit in CRITERIA:
        col = f"score_{crit}"
        if col not in Y.columns:
            continue
        raw = np.array([R.score_criterion(crit, r)[0] for _, r in X.iterrows()])
        y = Y[col].to_numpy()

        if np.std(raw) < 1e-9:
            a, b = 0.0, float(np.mean(y))
        else:
            a, b = (float(v) for v in np.polyfit(raw, y, 1))

        # A least-squares slope is r * sy/sx, so when the raw score barely
        # varies (sx small) the slope blows up and amplifies noise. Refuse any
        # fit that does not actually beat predicting the mean - stride_length
        # produced an MAE of 4.5 on a 0-10 scale this way, worse than a
        # constant. Falling back keeps the rubric honest rather than confident.
        fitted_mae = float(np.mean(np.abs(np.clip(a * raw + b, 0, 10) - y)))
        mean_mae = float(np.mean(np.abs(np.mean(y) - y)))
        if not np.isfinite(fitted_mae) or fitted_mae >= mean_mae:
            a, b = 0.0, float(np.mean(y))
            print(f"  note: {crit} rubric carries no usable signal "
                  f"(fit MAE {fitted_mae:.2f} vs mean {mean_mae:.2f}); "
                  f"falling back to the mean.")

        cal[crit]["_affine"] = {"slope": round(float(a), 6), "intercept": round(float(b), 6)}
    return cal


def main() -> None:
    X = pd.read_csv(PROCESSED_DIR / "features.csv", index_col="video_id").sort_index()
    Y = pd.read_csv(PROCESSED_DIR / "labels.csv", index_col="video_id").sort_index()
    cal = calibrate(X, Y)
    cal = fit_affine(X, Y, cal)

    path = MODEL_DIR / "rubric_calibration.json"
    path.write_text(json.dumps(cal, indent=2))

    print(f"Calibrated {sum(len(v) for v in cal.values())} bands "
          f"across {len(cal)} criteria -> {path}\n")
    for crit, bands in cal.items():
        print(f"{crit}:")
        for f, b in bands.items():
            if f == "_affine":
                continue
            print(f"    {f:34s} ideal={b['ideal']:>10.4f}  tol={b['tol']:>9.4f}  ({b['mode']})")


if __name__ == "__main__":
    main()
