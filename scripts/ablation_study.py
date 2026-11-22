"""Reproducible ablation study: estimator × feature-set.

Compares every combination of estimator family and feature subset on out-of-fold
MAE and R² per criterion, using the same cross_validate() machinery as the main
training run.  Output is a markdown table saved to ``reports/ablation.md``.

Usage
-----
    python scripts/ablation_study.py
    python scripts/ablation_study.py --estimators hgb ridge blend extra_trees
    python scripts/ablation_study.py --feature-sets all own_only --folds 5
    python scripts/ablation_study.py --out reports/my_ablation.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runway_ai.config import CRITERIA, MODEL_DIR, PROCESSED_DIR, REPORT_DIR  # noqa: E402
from runway_ai.models.train import cross_validate, _available_candidates        # noqa: E402

# ---------------------------------------------------------------------------
# Feature subsets
# ---------------------------------------------------------------------------
# "own_only" keeps only the features whose name prefix matches the criterion.
_CRITERION_PREFIXES: dict[str, list[str]] = {
    "posture":            ["posture_"],
    "balance":            ["balance_"],
    "confidence":         ["conf_"],
    "shoulder_alignment": ["shoulder_"],
    "head_position":      ["head_"],
    "arm_swing":          ["arm_"],
    "hip_movement":       ["hip_"],
    "rhythm":             ["rhythm_"],
    "stride_length":      ["stride_"],
    "foot_placement":     ["foot_"],
    "symmetry":           ["sym_"],
    "stage_presence":     ["conf_", "hip_shoulder_ratio_mean"],
}


def _own_features(crit: str, columns: list[str]) -> list[str]:
    prefixes = _CRITERION_PREFIXES.get(crit, [])
    keep = [c for c in columns if any(c.startswith(p) for p in prefixes)]
    # Always include the QC features so clip-quality effects are not mistaken
    # for genuine skill differences.
    keep += [c for c in columns if c.startswith("qc_") and c not in keep]
    return keep if keep else columns  # fallback: keep all


def _make_subset(X: pd.DataFrame, crit: str, feature_set: str) -> pd.DataFrame:
    if feature_set == "own_only":
        cols = _own_features(crit, list(X.columns))
        return X[cols]
    return X  # "all" — no filtering


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--estimators", nargs="+", default=None,
        help="Estimator kinds to compare. Defaults to all available on this install."
    )
    ap.add_argument(
        "--feature-sets", nargs="+", default=["all", "own_only"],
        choices=["all", "own_only"],
        help="Feature subsets to compare (default: all own_only)."
    )
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--features", default=str(PROCESSED_DIR / "features.csv"))
    ap.add_argument("--labels",   default=str(PROCESSED_DIR / "labels.csv"))
    ap.add_argument("--out",      default=str(REPORT_DIR / "ablation.md"))
    args = ap.parse_args()

    X = pd.read_csv(args.features, index_col="video_id").sort_index()
    Y = pd.read_csv(args.labels,   index_col="video_id").sort_index()
    assert X.index.equals(Y.index), "feature/label clips are misaligned"
    print(f"Loaded {X.shape[0]} clips × {X.shape[1]} features\n")

    available = _available_candidates()
    estimators: list[str] = args.estimators or available
    # Silently skip any requested estimator not available on this install.
    estimators = [k for k in estimators if k in available]
    if not estimators:
        print("No requested estimators are available. Install xgboost/catboost or "
              "remove --estimators.")
        return

    feature_sets: list[str] = args.feature_sets
    n_folds = args.folds

    print(f"Estimators:   {estimators}")
    print(f"Feature sets: {feature_sets}")
    print(f"Folds:        {n_folds}\n")

    # Columns: (estimator, feature_set) combinations.
    combos = [(e, fs) for fs in feature_sets for e in estimators]
    col_names = [f"{e}/{fs}" for e, fs in combos]

    # Results: rows = criteria, columns = combos (MAE).
    mae_rows: dict[str, list[float]] = {}
    r2_rows: dict[str, list[float]] = {}

    for crit in CRITERIA:
        col = f"score_{crit}"
        if col not in Y.columns:
            continue
        y = Y[col].rename(crit)
        mae_row, r2_row = [], []

        for est, fs in combos:
            X_sub = _make_subset(X, crit, fs)
            _, res, _ = cross_validate(X_sub, y, kind=est, n_splits=n_folds)
            mae_row.append(round(res.mae, 4))
            r2_row.append(round(res.r2, 4))
            print(f"  {crit:20s} [{est:15s}/{fs:8s}] "
                  f"MAE={res.mae:.3f}  R²={res.r2:+.3f}")

        mae_rows[crit] = mae_row
        r2_rows[crit]  = r2_row

    # Mean row.
    mae_rows["**mean**"] = [
        round(float(np.mean([mae_rows[c][i] for c in mae_rows if c != "**mean**"])), 4)
        for i in range(len(combos))
    ]
    r2_rows["**mean**"] = [
        round(float(np.mean([r2_rows[c][i] for c in r2_rows if c != "**mean**"])), 4)
        for i in range(len(combos))
    ]

    # ---------------------------------------------------------------------------
    # Build markdown table
    # ---------------------------------------------------------------------------
    lines: list[str] = [
        "# Ablation Study",
        "",
        f"**Clips:** {X.shape[0]}  |  **Folds:** {n_folds}",
        "",
        "All numbers are **out-of-fold** (OOF). The column header is "
        "`estimator / feature-set`.",
        "",
        "## MAE (lower is better)",
        "",
    ]

    header = "| Criterion | " + " | ".join(col_names) + " |"
    sep    = "|---" + "|---" * len(combos) + "|"
    lines += [header, sep]
    for crit, row in mae_rows.items():
        best_idx = int(np.argmin(row))
        cells = []
        for i, v in enumerate(row):
            cells.append(f"**{v}**" if i == best_idx else str(v))
        lines.append(f"| {crit} | " + " | ".join(cells) + " |")

    lines += ["", "## R² (higher is better)", ""]
    lines += [header, sep]
    for crit, row in r2_rows.items():
        best_idx = int(np.argmax(row))
        cells = []
        for i, v in enumerate(row):
            cells.append(f"**{v}**" if i == best_idx else str(v))
        lines.append(f"| {crit} | " + " | ".join(cells) + " |")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nAblation report written to {out_path}")


if __name__ == "__main__":
    main()
