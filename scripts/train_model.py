"""Train the twelve criterion regressors and compare them against baselines.

Every number printed is out-of-fold. Three references are reported so the
learned model's value is unambiguous:

* **mean**   - predict the training mean. Anything that cannot beat this is
  worthless.
* **ridge**  - regularised linear fit. Beating this shows the signal is
  genuinely non-linear rather than the ensemble adding noise.
* **rubric** - the hand-written biomechanical scorer. Beating this is what
  justifies having a learned model in the system at all.

Usage:  python scripts/train_model.py [--estimator auto] [--folds 5]
                                     [--groups-col subject_id]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runway_ai.config import CRITERIA, MODEL_DIR, PROCESSED_DIR, REPORT_DIR  # noqa: E402
from runway_ai.models.train import cross_validate, train_all                 # noqa: E402
from runway_ai.scoring.rubric import rubric_scores                           # noqa: E402


def rubric_reference(features: pd.DataFrame, labels: pd.DataFrame) -> dict[str, float]:
    """MAE of the deterministic rubric on the same clips."""
    scored = pd.DataFrame(
        [rubric_scores(r._asdict() if hasattr(r, "_asdict") else dict(r))
         for _, r in features.iterrows()],
        index=features.index,
    )
    return {
        c: float(np.mean(np.abs(scored[c] - labels[f"score_{c}"])))
        for c in CRITERIA if f"score_{c}" in labels.columns
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimator", default="auto",
                    choices=["auto", "hgb", "rf", "ridge", "blend",
                             "extra_trees", "xgb", "catboost",
                             "weighted_blend", "stacking"])
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--features", default=str(PROCESSED_DIR / "features.csv"))
    ap.add_argument("--labels", default=str(PROCESSED_DIR / "labels.csv"))
    ap.add_argument(
        "--groups-col", default=None, metavar="COL",
        help="Column in the features CSV holding a subject/person ID for "
             "GroupKFold (person-level) cross-validation. When omitted, "
             "falls back to the original score-rank stratification."
    )
    args = ap.parse_args()

    X = pd.read_csv(args.features, index_col="video_id").sort_index()
    Y = pd.read_csv(args.labels, index_col="video_id").sort_index()
    assert X.index.equals(Y.index), "feature/label clips are misaligned"
    print(f"Loaded {X.shape[0]} clips x {X.shape[1]} features\n")

    # Build groups array for person-level GroupKFold when requested.
    groups = None
    if args.groups_col:
        if args.groups_col in X.columns:
            groups = X[args.groups_col].to_numpy()
            n_subjects = len(np.unique(groups))
            print(f"GroupKFold: {n_subjects} unique subjects in '{args.groups_col}'\n")
        else:
            print(f"Warning: --groups-col '{args.groups_col}' not found in features "
                  f"CSV. Falling back to score-rank stratification.\n")

    print("Deterministic rubric reference (MAE):")
    rub = rubric_reference(X, Y)
    for c in CRITERIA:
        print(f"  {c:20s} {rub.get(c, float('nan')):.3f}")

    print(f"\nBaseline: ridge (out-of-fold)")
    ridge = {}
    for c in CRITERIA:
        _, res, _ = cross_validate(
            X, Y[f"score_{c}"].rename(c), "ridge", args.folds, groups=groups
        )
        ridge[c] = res.mae
        print(f"  {c:20s} MAE={res.mae:.3f}  R2={res.r2:+.3f}")

    grouped_note = f" [GroupKFold on '{args.groups_col}']" if groups is not None else ""
    print(f"\nModel: {args.estimator} (out-of-fold){grouped_note}")
    report = train_all(X, Y, kind=args.estimator, n_splits=args.folds, groups=groups)

    rows = []
    for r in report["per_criterion"]:
        c = r["criterion"]
        rows.append({
            "criterion": c,
            "MAE": r["mae"],
            "R2": r["r2"],
            "MAE_mean": r["baseline_mae"],
            "MAE_ridge": round(ridge[c], 3),
            "MAE_rubric": round(rub.get(c, float("nan")), 3),
            "within_1pt": r["within_1_point"],
        })
    table = pd.DataFrame(rows).set_index("criterion")
    table["beats_rubric"] = table["MAE"] < table["MAE_rubric"]
    table["beats_ridge"] = table["MAE"] <= table["MAE_ridge"] + 1e-9
    table["chosen"] = pd.Series(report.get("chosen_estimator", {}))

    print("\n" + "=" * 88)
    print(table.to_string())
    print("=" * 88)
    print(f"mean MAE {report['mean_mae']:.3f} | mean R2 {report['mean_r2']:+.3f} | "
          f"beats rubric on {int(table['beats_rubric'].sum())}/{len(table)} criteria")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(REPORT_DIR / "model_comparison.csv")
    print(f"\nmodels  -> {MODEL_DIR}")
    print(f"report  -> {REPORT_DIR / 'model_comparison.csv'}")


if __name__ == "__main__":
    main()
