"""TreeSHAP-based feature attribution for per-criterion walk scores.

Each criterion is scored by a fitted estimator.  When that estimator is a
``HistGradientBoostingRegressor`` (or wraps one inside a ``BlendRegressor``),
TreeSHAP computes exact Shapley values in polynomial time and the result is
fully additive: the sum of all feature attributions plus the expected value
equals the model's prediction.

When the ``shap`` library is not installed the module falls back to an empty
dict rather than raising, so the rest of the pipeline continues without
attribution data.

Usage
-----
The main entry point is ``shap_values(feats, models)``.  The result is
returned by ``scoring.explain.breakdown()`` under the key ``"shap"``, so the
API and web app receive it automatically alongside the existing rubric bands.

Limitations
-----------
* Only the ``hgb`` component of a blend is explained, because TreeSHAP
  requires a tree-based model.  ``ridge`` contributions are omitted.
* SHAP values are computed on one row; the background dataset for
  ``TreeExplainer`` is the single prediction row, so these are
  interventional (not conditional) attributions.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..config import CRITERIA


def _get_hgb(bundle: dict):
    """Extract the HGB component from a model bundle.

    Handles plain ``HistGradientBoostingRegressor``, ``BlendRegressor``, and
    ``WeightedBlendRegressor``.  Returns ``None`` when none is found.
    """
    model = bundle.get("model")
    if model is None:
        return None

    from sklearn.ensemble import HistGradientBoostingRegressor

    # Direct HGB.
    if isinstance(model, HistGradientBoostingRegressor):
        return model

    # Blend / WeightedBlend: look inside .models_
    sub_models = getattr(model, "models_", [])
    for m in sub_models:
        if isinstance(m, HistGradientBoostingRegressor):
            return m

    return None


def shap_values(
    feats: dict,
    models: dict,
) -> dict[str, dict[str, float]]:
    """Compute TreeSHAP attributions for each criterion.

    Parameters
    ----------
    feats:
        Feature dict as produced by ``runway_ai.features.builder``.
    models:
        Per-criterion model bundles from ``train.load_models()``.

    Returns
    -------
    dict mapping criterion name to ``{feature_name: shap_value}``.
    Returns an empty dict for any criterion whose model is not tree-based or
    when the ``shap`` library is not installed.
    """
    try:
        import shap as _shap
    except ImportError:
        return {}

    out: dict[str, dict[str, float]] = {}

    for crit in CRITERIA:
        bundle = models.get(crit)
        if bundle is None:
            continue

        hgb = _get_hgb(bundle)
        if hgb is None:
            continue

        names = bundle["features"]
        row = pd.DataFrame(
            [[float(feats.get(f, 0.0)) for f in names]], columns=names
        )

        try:
            explainer = _shap.TreeExplainer(hgb)
            sv = explainer.shap_values(row)
            # shap_values returns shape (1, n_features) for a single row.
            if hasattr(sv, "values"):
                vals = sv.values[0]          # newer shap: Explanation object
            else:
                vals = np.asarray(sv)[0]     # older shap: ndarray
            out[crit] = {
                name: round(float(v), 5)
                for name, v in zip(names, vals)
            }
        except Exception:
            # SHAP is a bonus; never let it break the pipeline.
            pass

    return out
