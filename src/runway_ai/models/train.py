"""Model training: features -> twelve criterion scores.

Design
------
The task is **multi-output regression** on tabular features with a few hundred
to a few thousand clips. That rules gradient-boosted trees in and deep networks
out: with ~100 engineered features and no raw-pixel input, a sequence model has
nothing to exploit that the features do not already expose, and it would need
far more data to match a boosted ensemble.

One model is fitted **per criterion** rather than a single joint model. The
criteria have very different difficulty and very different informative features,
and a shared model would let the strong criteria dominate the loss. Per-criterion
models also make the feature importances directly interpretable, which the
coaching layer uses.

Validation is by **grouped, stratified split on overall skill**, so the easy
high-skill clips cannot all land in one fold and flatter the score. When a
``groups`` array is supplied the split is additionally person-level (GroupKFold),
so no clips from the same subject appear in both train and validation.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..config import CRITERIA, MODEL_DIR, RANDOM_SEED

# ---------------------------------------------------------------------------
# Optional heavy dependencies — imported lazily so the project works without
# them.  Register their kind-strings only when the library is present.
# ---------------------------------------------------------------------------
_EXTRA_KINDS: list[str] = []

try:
    from xgboost import XGBRegressor as _XGBRegressor  # type: ignore
    _EXTRA_KINDS.append("xgb")
except ImportError:
    _XGBRegressor = None  # type: ignore

try:
    from catboost import CatBoostRegressor as _CatBoostRegressor  # type: ignore
    _EXTRA_KINDS.append("catboost")
except ImportError:
    _CatBoostRegressor = None  # type: ignore


# --------------------------------------------------------------------------- #
# Model zoo
# --------------------------------------------------------------------------- #
def build_estimator(kind: str, seed: int = RANDOM_SEED):
    """Return an unfitted estimator.

    ``hgb`` is the default. ``ridge`` and ``rf`` exist as comparison baselines
    for the ablation table - a boosted ensemble that cannot beat ridge on a
    criterion is telling you that criterion's signal is essentially linear.

    ``extra_trees`` is always available (scikit-learn). ``xgb`` and ``catboost``
    are registered only when the corresponding library is installed.
    """
    if kind == "hgb":
        return HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.06,
            max_depth=None,
            max_leaf_nodes=31,
            min_samples_leaf=15,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=30,
            random_state=seed,
        )
    if kind == "rf":
        return RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=seed
        )
    if kind == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=400, min_samples_leaf=3, n_jobs=-1, random_state=seed
        )
    if kind == "ridge":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ])
    if kind == "mean":
        return DummyRegressor(strategy="mean")
    if kind == "xgb":
        if _XGBRegressor is None:
            raise ImportError("xgboost is not installed. pip install xgboost")
        return _XGBRegressor(
            n_estimators=400,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            n_jobs=-1,
            random_state=seed,
            verbosity=0,
        )
    if kind == "catboost":
        if _CatBoostRegressor is None:
            raise ImportError("catboost is not installed. pip install catboost")
        return _CatBoostRegressor(
            iterations=400,
            learning_rate=0.06,
            depth=6,
            l2_leaf_reg=1.0,
            random_seed=seed,
            verbose=0,
        )
    raise ValueError(f"unknown estimator kind: {kind!r}")


# --------------------------------------------------------------------------- #
# Compound estimators
# --------------------------------------------------------------------------- #
class BlendRegressor(RegressorMixin, BaseEstimator):
    """Unweighted average of several fitted regressors.

    A linear model and a boosted ensemble make different errors - the first is
    biased but low-variance, the second the reverse - so averaging them is
    usually at least as good as either and much steadier across criteria.

    Inheriting from the sklearn base classes is not cosmetic: without them the
    estimator tags are missing, and ``permutation_importance``,
    ``cross_val_score`` and the rest of the sklearn tooling refuse to accept it.
    """

    def __init__(self, kinds: tuple[str, ...] = ("hgb", "ridge"), seed: int = RANDOM_SEED):
        # sklearn's clone() requires __init__ to only store its arguments
        # verbatim, under the same names.
        self.kinds = kinds
        self.seed = seed

    def fit(self, X, y):
        self.models_ = [build_estimator(k, self.seed).fit(X, y) for k in self.kinds]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        return np.mean([m.predict(X) for m in self.models_], axis=0)


class WeightedBlendRegressor(RegressorMixin, BaseEstimator):
    """Non-negative weighted average of fitted regressors.

    Weights are fitted on OOF predictions using non-negative least squares
    (``np.linalg.lstsq`` with a non-negativity projection), then L1-normalised.
    Falls back to equal weights when the solve is degenerate.

    Like ``BlendRegressor`` this inherits sklearn base classes so the full
    sklearn tooling accepts it.
    """

    def __init__(self, kinds: tuple[str, ...] = ("hgb", "ridge", "extra_trees"),
                 seed: int = RANDOM_SEED):
        self.kinds = kinds
        self.seed = seed

    def fit(self, X, y):
        self.models_ = [build_estimator(k, self.seed).fit(X, y) for k in self.kinds]
        # Fit weights on training predictions (in-sample; used only when this
        # is itself a component of a cross-validated outer loop).
        preds = np.column_stack([m.predict(X) for m in self.models_])
        try:
            # Non-negative least squares via clip-then-normalise.
            raw, _, _, _ = np.linalg.lstsq(preds, np.asarray(y), rcond=None)
            raw = np.clip(raw, 0.0, None)
            s = raw.sum()
            self.weights_ = raw / s if s > 1e-12 else np.ones(len(self.kinds)) / len(self.kinds)
        except Exception:
            self.weights_ = np.ones(len(self.kinds)) / len(self.kinds)
        self.is_fitted_ = True
        return self

    def predict(self, X):
        preds = np.column_stack([m.predict(X) for m in self.models_])
        return preds @ self.weights_


class StackingRegressor(RegressorMixin, BaseEstimator):
    """Cross-validated stacking with a RidgeCV meta-learner.

    Base estimators are fitted on n-1 folds; their held-out predictions form
    the meta-features. The meta-learner is then fitted on all meta-features.
    Final prediction uses the full-data-fitted base models + meta-learner.
    """

    def __init__(self, kinds: tuple[str, ...] = ("hgb", "ridge", "extra_trees"),
                 n_splits: int = 5, seed: int = RANDOM_SEED):
        self.kinds = kinds
        self.n_splits = n_splits
        self.seed = seed

    def fit(self, X, y):
        y = np.asarray(y)
        n = len(y)
        meta_X = np.zeros((n, len(self.kinds)))

        # Build OOF meta-features for the meta-learner.
        order = np.argsort(y)
        folds = np.zeros(n, dtype=int)
        folds[order] = np.arange(n) % self.n_splits

        for k in range(self.n_splits):
            tr, te = folds != k, folds == k
            X_tr = X[tr] if not isinstance(X, pd.DataFrame) else X.iloc[tr]
            X_te = X[te] if not isinstance(X, pd.DataFrame) else X.iloc[te]
            y_tr = y[tr]
            for j, kind in enumerate(self.kinds):
                est = build_estimator(kind, self.seed).fit(X_tr, y_tr)
                meta_X[te, j] = est.predict(X_te)

        meta_X = np.clip(meta_X, 0.0, 10.0)
        self.meta_ = Pipeline([
            ("scale", StandardScaler()),
            ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 25))),
        ]).fit(meta_X, y)

        # Final base models fitted on all data.
        self.base_models_ = [build_estimator(k, self.seed).fit(X, y) for k in self.kinds]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        meta_X = np.column_stack([m.predict(X) for m in self.base_models_])
        meta_X = np.clip(meta_X, 0.0, 10.0)
        return self.meta_.predict(meta_X)


# ---------------------------------------------------------------------------
# Candidate pool — auto selection iterates over all available kinds.
# ---------------------------------------------------------------------------
CANDIDATES = ["hgb", "ridge", "blend", "extra_trees"] + _EXTRA_KINDS


def _available_candidates() -> list[str]:
    """Candidates that can actually be built on this installation."""
    out = []
    for k in CANDIDATES:
        try:
            build_estimator(k)
            out.append(k)
        except (ImportError, ValueError):
            pass
    return out


def _make(kind: str, seed: int):
    if kind == "blend":
        return BlendRegressor(("hgb", "ridge"), seed)
    if kind == "weighted_blend":
        base = ["hgb", "ridge", "extra_trees"] + [k for k in _EXTRA_KINDS]
        return WeightedBlendRegressor(tuple(base), seed)
    if kind == "stacking":
        base = ["hgb", "ridge", "extra_trees"] + [k for k in _EXTRA_KINDS]
        return StackingRegressor(tuple(base), seed=seed)
    return build_estimator(kind, seed)


# --------------------------------------------------------------------------- #
# Cross-validated evaluation
# --------------------------------------------------------------------------- #
def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    kind: str = "hgb",
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, CriterionResult, list]:
    """Out-of-fold predictions plus a metric summary for one criterion.

    Every reported number is out-of-fold. An in-sample score on a boosted
    ensemble is close to meaningless - it will happily reach R^2 ~ 0.99 by
    memorising - so nothing here is ever measured on data the fold was fitted
    on.

    Parameters
    ----------
    groups:
        Optional integer array of length ``len(X)``. When supplied, folds are
        person-level (GroupKFold) so no subject's clips appear in both train
        and validation. When ``None``, falls back to the original score-rank
        stratification.

    Returns
    -------
    oof: np.ndarray
        Out-of-fold predictions.
    res: CriterionResult
        Aggregated metrics.
    fold_models: list
        One fitted estimator per fold (used by uncertainty quantification).
    """
    y_np = y.to_numpy()
    X_np = X.to_numpy() if isinstance(X, pd.DataFrame) else X

    if groups is not None and len(np.unique(groups)) >= n_splits:
        # Person-level split: each group (subject) stays in one fold.
        gkf = GroupKFold(n_splits=n_splits)
        fold_indices = list(gkf.split(X_np, y_np, groups=groups))
    else:
        # Score-rank stratification (original behaviour).
        order = np.argsort(y_np)
        folds = np.zeros(len(y_np), dtype=int)
        folds[order] = np.arange(len(y_np)) % n_splits
        fold_indices = [(np.where(folds != k)[0], np.where(folds == k)[0])
                        for k in range(n_splits)]

    oof = np.full(len(y_np), np.nan)
    fold_models = []
    for tr_idx, te_idx in fold_indices:
        model = _make(kind, seed)
        if isinstance(X, pd.DataFrame):
            model.fit(X.iloc[tr_idx], y_np[tr_idx])
            oof[te_idx] = model.predict(X.iloc[te_idx])
        else:
            model.fit(X_np[tr_idx], y_np[tr_idx])
            oof[te_idx] = model.predict(X_np[te_idx])
        fold_models.append(model)

    oof = np.clip(oof, 0.0, 10.0)
    base = np.full(len(y_np), float(np.mean(y_np)))
    res = CriterionResult(
        criterion=str(y.name),
        mae=float(mean_absolute_error(y_np, oof)),
        r2=float(r2_score(y_np, oof)),
        baseline_mae=float(mean_absolute_error(y_np, base)),
        within_1=float(np.mean(np.abs(oof - y_np) <= 1.0)),
        n=int(len(y_np)),
    )
    return oof, res, fold_models


@dataclass
class CriterionResult:
    criterion: str
    mae: float
    r2: float
    baseline_mae: float
    within_1: float          # fraction of predictions within 1.0 of truth
    n: int

    def as_dict(self) -> dict:
        return {
            "criterion": self.criterion,
            "mae": round(self.mae, 4),
            "r2": round(self.r2, 4),
            "baseline_mae": round(self.baseline_mae, 4),
            "skill_vs_baseline": round(1.0 - self.mae / max(self.baseline_mae, 1e-9), 4),
            "within_1_point": round(self.within_1, 4),
            "n": self.n,
        }


# --------------------------------------------------------------------------- #
# Full training run
# --------------------------------------------------------------------------- #
def train_all(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    kind: str = "auto",
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
    out_dir: Path | None = None,
    groups: np.ndarray | None = None,
) -> dict:
    """Cross-validate then fit a final model per criterion, and persist them.

    With ``kind="auto"`` each criterion picks its own estimator by out-of-fold
    MAE. This is worth doing rather than forcing one family everywhere: the
    twelve criteria differ in how non-linear they are, and a boosted ensemble
    on an essentially linear criterion just adds variance. Selection is on
    out-of-fold error only, so no criterion is chosen on data it was fitted on.

    Parameters
    ----------
    groups:
        Optional integer array of length ``len(features)``. When supplied,
        cross-validation uses GroupKFold (person-level splits) so no subject's
        clips appear in both train and validation.

    Returns a report dict with per-criterion metrics and the feature list.
    """
    out_dir = Path(out_dir or MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    X = features.sort_index()
    feature_names = list(X.columns)
    results, oof_frame, chosen = [], {}, {}
    t0 = time.time()

    candidates = _available_candidates() if kind == "auto" else [kind]

    for crit in CRITERIA:
        col = f"score_{crit}"
        if col not in labels.columns:
            continue
        y = labels.sort_index()[col].rename(crit)

        scored = []
        for k in candidates:
            oof_k, res_k, fold_models_k = cross_validate(
                X, y, kind=k, n_splits=n_splits, seed=seed, groups=groups
            )
            scored.append((res_k.mae, k, oof_k, res_k, fold_models_k))
        scored.sort(key=lambda t: t[0])
        _, best_kind, oof, res, best_fold_models = scored[0]

        oof_frame[crit] = oof
        results.append(res)
        chosen[crit] = best_kind

        # Persist the winning final model (fitted on all data).
        final = _make(best_kind, seed).fit(X, y)
        joblib.dump(
            {"model": final, "features": feature_names, "criterion": crit, "kind": best_kind},
            out_dir / f"model_{crit}.joblib",
        )

        # Persist per-fold models for uncertainty quantification (Phase 3).
        joblib.dump(
            {"fold_models": best_fold_models, "features": feature_names,
             "criterion": crit, "kind": best_kind, "n_splits": n_splits},
            out_dir / f"fold_models_{crit}.joblib",
        )

        print(f"  {crit:20s} MAE={res.mae:.3f}  R2={res.r2:+.3f}  "
              f"[{best_kind}]  (baseline MAE {res.baseline_mae:.3f})")

    report = {
        "estimator": kind,
        "chosen_estimator": chosen,
        "n_clips": int(len(X)),
        "n_features": len(feature_names),
        "n_splits": n_splits,
        "seed": seed,
        "grouped": groups is not None,
        "train_seconds": round(time.time() - t0, 1),
        "per_criterion": [r.as_dict() for r in results],
        "features": feature_names,
    }
    report["mean_mae"] = round(float(np.mean([r.mae for r in results])), 4)
    report["mean_r2"] = round(float(np.mean([r.r2 for r in results])), 4)

    # Record the training feature ranges so inference can tell whether a new
    # clip is inside the distribution or being extrapolated.
    from ..scoring.coverage import build_reference
    build_reference(X, out_dir / "feature_reference.json")

    (out_dir / "training_report.json").write_text(json.dumps(report, indent=2))
    pd.DataFrame(oof_frame, index=X.index).to_csv(out_dir / "oof_predictions.csv")
    return report


# --------------------------------------------------------------------------- #
# Inference helpers
# --------------------------------------------------------------------------- #
def load_models(model_dir: Path | None = None) -> dict:
    """Load every persisted per-criterion model."""
    model_dir = Path(model_dir or MODEL_DIR)
    models = {}
    for crit in CRITERIA:
        path = model_dir / f"model_{crit}.joblib"
        if path.exists():
            models[crit] = joblib.load(path)
    return models


def load_fold_models(model_dir: Path | None = None) -> dict:
    """Load per-fold models for uncertainty quantification.

    Returns an empty dict when the fold-model files do not exist (e.g. on a
    checkout that has not been retrained since this feature was added).
    """
    model_dir = Path(model_dir or MODEL_DIR)
    fold_models = {}
    for crit in CRITERIA:
        path = model_dir / f"fold_models_{crit}.joblib"
        if path.exists():
            try:
                fold_models[crit] = joblib.load(path)
            except Exception:
                pass
    return fold_models


def _load_feature_reference() -> dict | None:
    """Lazily load the training-distribution reference for feature clamping."""
    try:
        from ..scoring.coverage import load_reference
        return load_reference()
    except Exception:
        return None


_FEATURE_REF: dict | None = None


def _clamp_to_training(value: float, feature: str, ref: dict) -> float:
    """Clamp a feature value to the training distribution's p01-p99 range.

    When a real video produces features far outside the synthetic training
    range (e.g. conf_speed_cv = 1634 against a training range of 1.6-4.5) the
    model has to extrapolate, and tree ensembles extrapolate by predicting the
    value of the nearest leaf they *do* have, which is arbitrary. Clamping
    keeps the prediction in the regime the model was actually fitted on.

    The p01-p99 range is used rather than q1-q3 because the model *was* trained
    on the tails - it just was not trained beyond them.
    """
    stats = ref.get(feature)
    if not stats:
        return value
    lo, hi = stats["p01"], stats["p99"]
    return float(np.clip(value, lo, hi))


def predict_scores(feats: dict, models: dict) -> dict[str, float]:
    """Score one feature dict with the loaded models.

    The row is passed as a named DataFrame, not a bare array: the estimators
    were fitted on named columns, and handing them a positional array both
    warns and silently relies on column order matching. Building from
    ``bundle["features"]`` guarantees the order the model was trained with.

    Features are clamped to the training distribution before prediction so that
    out-of-range values do not cause wild extrapolation. The coverage system
    already detects when a clip sits outside the training range; this makes the
    model's prediction match that detection rather than ignoring it.
    """
    global _FEATURE_REF
    if _FEATURE_REF is None:
        _FEATURE_REF = _load_feature_reference() or {}

    out = {}
    for crit, bundle in models.items():
        names = bundle["features"]
        values = []
        for f in names:
            v = float(feats.get(f, 0.0))
            if _FEATURE_REF:
                v = _clamp_to_training(v, f, _FEATURE_REF)
            values.append(v)
        row = pd.DataFrame([values], columns=names)
        out[crit] = float(np.clip(bundle["model"].predict(row)[0], 0.0, 10.0))
    return out
