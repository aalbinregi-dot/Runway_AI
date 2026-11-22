"""Prediction uncertainty for per-criterion walk scores.

Two independent sources of disagreement are combined:

1. **Fold disagreement** — the spread of predictions across the ``n_splits``
   per-fold models saved alongside the final model. Each fold was held out in
   turn, so these represent the same estimator family evaluated on unseen data;
   their spread is a direct measure of how much the model's answer depends on
   which slice of the training data it saw.

2. **Ensemble disagreement** — the spread across component estimators inside a
   ``BlendRegressor`` or ``WeightedBlendRegressor``. A linear model and a
   boosted tree agree on a prediction when the relationship is genuinely close
   to linear; divergence signals a part of the feature space where the data is
   sparse and the choice of algorithm matters.

The combined standard deviation feeds a simple three-tier confidence label:

* **HIGH** — std < 0.5 (roughly half a marking band)
* **MEDIUM** — 0.5 ≤ std < 1.0 (a full marking band)
* **LOW** — std ≥ 1.0 (larger than the average model error)

A 95% interval is reported as ``lower``/``upper``, clipped to [0, 10].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CRITERIA

# Confidence tiers based on combined std.
_TIERS = [
    (0.5,  "HIGH",   "Prediction is stable across cross-validation folds and "
                     "estimator families."),
    (1.0,  "MEDIUM", "Moderate spread across folds or estimators. Treat "
                     "differences smaller than ±1 point with caution."),
    (None, "LOW",    "High spread across folds or estimators. The score is a "
                     "rough guide; prefer the rubric evidence traces over the "
                     "number itself."),
]

# 95% interval multiplier (1.96 ≈ two-sided 95% Normal).
_Z95 = 1.96


def _tier(std: float) -> tuple[str, str]:
    for threshold, label, note in _TIERS:
        if threshold is None or std < threshold:
            return label, note
    return "LOW", _TIERS[-1][2]


def _ensemble_std(bundle: dict, row: pd.DataFrame) -> float | None:
    """Std across component estimators inside a blend/weighted-blend.

    Returns ``None`` when the bundle does not expose sub-estimators (e.g. hgb,
    ridge, extra_trees), in which case only fold disagreement is used.
    """
    model = bundle.get("model")
    sub_models = getattr(model, "models_", None)
    if not sub_models or len(sub_models) < 2:
        return None
    preds = []
    for m in sub_models:
        try:
            preds.append(float(m.predict(row)[0]))
        except Exception:
            pass
    if len(preds) < 2:
        return None
    return float(np.std(preds, ddof=1))


def predict_with_uncertainty(
    feats: dict,
    models: dict,
    fold_models: dict | None = None,
) -> dict[str, dict]:
    """Score one clip and attach uncertainty estimates to each criterion.

    Parameters
    ----------
    feats:
        Feature dict as produced by ``runway_ai.features.builder``.
    models:
        Per-criterion model bundles from ``train.load_models()``.
    fold_models:
        Optional per-criterion fold-model bundles from
        ``train.load_fold_models()``. When absent, only ensemble disagreement
        is used.

    Returns
    -------
    dict mapping criterion name to::

        {
            "score":      float,   # central prediction, clipped [0, 10]
            "std":        float,   # combined std (0 when unavailable)
            "lower":      float,   # score − 1.96 × std, clipped [0, 10]
            "upper":      float,   # score + 1.96 × std, clipped [0, 10]
            "confidence": str,     # "HIGH" | "MEDIUM" | "LOW"
            "note":       str,     # human-readable confidence note
            "n_folds":    int,     # number of fold models used
        }
    """
    out: dict[str, dict] = {}

    for crit in CRITERIA:
        bundle = models.get(crit)
        if bundle is None:
            continue

        names = bundle["features"]
        row = pd.DataFrame(
            [[float(feats.get(f, 0.0)) for f in names]], columns=names
        )

        # --- Central prediction -----------------------------------------------
        try:
            score = float(np.clip(bundle["model"].predict(row)[0], 0.0, 10.0))
        except Exception:
            continue

        stds: list[float] = []

        # --- Fold disagreement ------------------------------------------------
        n_folds = 0
        fm_bundle = (fold_models or {}).get(crit)
        if fm_bundle:
            fold_preds = []
            for fm in fm_bundle.get("fold_models", []):
                try:
                    fp = float(fm.predict(row)[0])
                    fold_preds.append(np.clip(fp, 0.0, 10.0))
                except Exception:
                    pass
            n_folds = len(fold_preds)
            if n_folds >= 2:
                stds.append(float(np.std(fold_preds, ddof=1)))

        # --- Ensemble disagreement --------------------------------------------
        ens_std = _ensemble_std(bundle, row)
        if ens_std is not None:
            stds.append(ens_std)

        # --- Combine ----------------------------------------------------------
        # RSS when both are available; single source otherwise.
        if len(stds) == 2:
            combined = float(np.sqrt(stds[0] ** 2 + stds[1] ** 2) / np.sqrt(2))
        elif len(stds) == 1:
            combined = stds[0]
        else:
            combined = 0.0

        confidence, note = _tier(combined)
        half = _Z95 * combined

        out[crit] = {
            "score":      score,
            "std":        round(combined, 3),
            "lower":      round(float(np.clip(score - half, 0.0, 10.0)), 2),
            "upper":      round(float(np.clip(score + half, 0.0, 10.0)), 2),
            "confidence": confidence,
            "note":       note,
            "n_folds":    n_folds,
        }

    return out
