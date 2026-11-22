"""How much of each criterion's reported accuracy is actually measuring *it*?

The generator draws the twelve skill qualities from a shared ability factor at
r ~ 0.6, because real walkers are like that - good posture and good balance
co-occur. That realism has a cost the generator's own docstring names:

    Set ``correlated=False`` to draw the twelve qualities independently. That
    destroys the realistic covariance but is essential for diagnostics: it is
    the only way to check that, say, the rhythm features respond to *rhythm*
    rather than riding on the shared ability factor.

That check had never been run. `--independent` existed in `build_dataset.py` and
nothing called it, so every headline R^2 in this project included an unmeasured
amount of shared-factor leakage: a model can score rhythm well by reading sway,
because on correlated data sway predicts general ability and general ability
predicts rhythm.

This script measures the size of that effect, three ways per criterion:

1. **as reported** - every feature, correlated latents. The number the project
   has been quoting.
2. **independent latents** - every feature, but the twelve qualities drawn
   independently. Cross-criterion shortcuts are worthless here, so whatever
   survives is genuine.
3. **own features only, independent latents** - restricted to that criterion's
   own feature group. Answers the sharpest version of the question: does the
   signal exist in the place it is supposed to be?

Every number is out of fold, using the same `cross_validate` the training script
uses, so the comparison is like for like.

    python scripts/leakage_audit.py [--folds 5] [--estimator blend]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runway_ai.config import CRITERIA, PROCESSED_DIR, REPORT_DIR  # noqa: E402
from runway_ai.models.train import cross_validate                 # noqa: E402

#: Which feature prefix belongs to which criterion. `stage_presence` has no
#: group of its own - it is emergent in the generator, built from confidence,
#: posture and head position - so column 3 is not defined for it.
OWN_PREFIX = {
    "posture": "posture_",
    "balance": "balance_",
    "confidence": "conf_",
    "shoulder_alignment": "shoulder_",
    "head_position": "head_",
    "arm_swing": "arm_",
    "hip_movement": "hip_",
    "rhythm": "rhythm_",
    "stride_length": "stride_",
    "foot_placement": "foot_",
    "symmetry": "sym_",
    "stage_presence": None,
}


def load(suffix: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    fpath = PROCESSED_DIR / f"features{suffix}.csv"
    lpath = PROCESSED_DIR / f"labels{suffix}.csv"
    if not fpath.exists() or not lpath.exists():
        raise SystemExit(
            f"missing {fpath.name} / {lpath.name}. Build it first:\n"
            f"    python scripts/build_dataset.py --n 1500 --seed 42"
            f"{' --independent' if suffix else ''} --no-keypoints")
    X = pd.read_csv(fpath).set_index("video_id").sort_index()
    Y = pd.read_csv(lpath).set_index("video_id").sort_index()
    X = X.select_dtypes("number").fillna(0.0)
    return X, Y


def score(X: pd.DataFrame, y: pd.Series, kind: str, folds: int) -> dict:
    _, res = cross_validate(X, y, kind, folds)
    return {"mae": float(res.mae), "r2": float(res.r2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--estimator", default="hgb",
                    choices=["hgb", "rf", "ridge", "blend"])
    args = ap.parse_args()

    Xc, Yc = load("")
    Xi, Yi = load("_indep")
    print(f"correlated corpus : {Xc.shape[0]} clips x {Xc.shape[1]} features")
    print(f"independent corpus: {Xi.shape[0]} clips x {Xi.shape[1]} features")
    print(f"estimator: {args.estimator}, {args.folds}-fold, all numbers out of fold\n")

    rows = []
    for c in CRITERIA:
        col = f"score_{c}"
        if col not in Yc or col not in Yi:
            continue
        yc, yi = Yc[col], Yi[col]

        as_reported = score(Xc, yc.rename(c), args.estimator, args.folds)
        independent = score(Xi, yi.rename(c), args.estimator, args.folds)

        prefix = OWN_PREFIX.get(c)
        own = None
        if prefix:
            cols = [k for k in Xi.columns if k.startswith(prefix)]
            if cols:
                own = score(Xi[cols], yi.rename(c), args.estimator, args.folds)

        rows.append({
            "criterion": c,
            "r2_as_reported": round(as_reported["r2"], 3),
            "r2_independent": round(independent["r2"], 3),
            "r2_own_features": round(own["r2"], 3) if own else None,
            "mae_as_reported": round(as_reported["mae"], 3),
            "mae_independent": round(independent["mae"], 3),
            "leakage": round(as_reported["r2"] - independent["r2"], 3),
            "n_own_features": len([k for k in Xi.columns if prefix and k.startswith(prefix)]),
            "target_sd_correlated": round(float(yc.std()), 2),
            "target_sd_independent": round(float(yi.std()), 2),
        })
        r = rows[-1]
        own_txt = f"{r['r2_own_features']:+.3f}" if own else "   n/a"
        print(f"  {c:20s} reported {r['r2_as_reported']:+.3f}   "
              f"independent {r['r2_independent']:+.3f}   own {own_txt}   "
              f"leak {r['leakage']:+.3f}")

    df = pd.DataFrame(rows)
    out = REPORT_DIR / "leakage_audit.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"\n{'=' * 74}")
    print(f"mean R2 as reported : {df.r2_as_reported.mean():+.3f}")
    print(f"mean R2 independent : {df.r2_independent.mean():+.3f}")
    print(f"mean leakage        : {df.leakage.mean():+.3f}")
    print(f"\ntarget spread: correlated sd ~{df.target_sd_correlated.mean():.2f}, "
          f"independent sd ~{df.target_sd_independent.mean():.2f} - R2 is relative to "
          f"each corpus's own variance, so MAE (absolute, /10) is the safer "
          f"cross-corpus comparison.")

    worst = df.nlargest(3, "leakage")[["criterion", "leakage"]].values.tolist()
    print("\nmost dependent on the shared ability factor:")
    for name, gap in worst:
        print(f"  {name:20s} {gap:+.3f}")

    weak = df[(df.r2_own_features.notna()) & (df.r2_own_features < 0.2)]
    if len(weak):
        print("\nown feature group carries little signal (R2 < 0.20):")
        for _, r in weak.iterrows():
            print(f"  {r.criterion:20s} own R2 {r.r2_own_features:+.3f} "
                  f"from {int(r.n_own_features)} features")

    (REPORT_DIR / "leakage_audit.json").write_text(json.dumps(rows, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
