"""Feature sensitivity analysis.

Sweeps one latent quality at a time from 0.05 to 0.95 with everything else held
at 0.5, and reports which features actually respond. This is the diagnostic
that tells us whether a criterion is *observable* at all: if no feature moves
when the latent moves, no model can ever learn that criterion, and the fault is
in the generator or the feature set - not in the regressor.

Usage:  python scripts/sensitivity.py [criterion ...]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runway_ai.config import CRITERIA                       # noqa: E402
from runway_ai.data.loaders import preprocess               # noqa: E402
from runway_ai.data.synthetic import WalkParams, make_clip  # noqa: E402
from runway_ai.features.builder import extract_features     # noqa: E402


def sweep(criterion: str, n_levels: int = 9, reps: int = 30, seed: int = 3,
          nuisance: float = 1.0) -> pd.DataFrame:
    """Vary one latent, average over ``reps`` nuisance draws per level.

    ``reps`` must be large. With ~90 features, the largest |r| you can pull out
    of pure noise is roughly ``2.5 / sqrt(n)``; at n=54 that is 0.34, which is
    big enough to look like a real effect. n=270 puts the floor near 0.15.
    """
    rng = np.random.default_rng(seed)
    levels = np.linspace(0.05, 0.95, n_levels)
    rows = []
    for lvl in levels:
        for r in range(reps):
            q = {f"{c}_q": 0.5 for c in CRITERIA}
            q[f"{criterion}_q"] = float(lvl)
            p = WalkParams(
                **q,
                height_m=float(rng.normal(1.72, 0.05 * nuisance)),
                camera_azimuth_deg=float(rng.normal(0, 10 * nuisance)),
                camera_distance_m=float(5.5 + rng.uniform(-2.0, 2.0) * nuisance),
                camera_height_m=float(1.3 + rng.uniform(-0.3, 0.3) * nuisance),
                fps=30.0,
                duration_s=6.0,
                detector_noise=0.005 * nuisance,
                dropout_rate=0.02 * nuisance,
            )
            seq, _ = make_clip(f"s{r}", p, rng)
            feats = extract_features(preprocess(seq))
            feats["level"] = lvl
            rows.append(feats)
    return pd.DataFrame(rows)


def report(criterion: str, top: int = 8, **kw) -> float:
    """Rank features by how strongly they track the swept latent.

    Ranking is by **mutual information**, not correlation. Several faults are
    two-sided by construction - over-striding and mincing are both bad, and the
    generator randomises the sign - so the feature-to-quality relationship is
    U-shaped and its Pearson r is near zero even when the feature is a perfect
    predictor. MI has no such blind spot. Pearson is still printed alongside so
    the *direction* of a monotone effect stays visible.
    """
    from sklearn.feature_selection import mutual_info_regression

    df = sweep(criterion, **kw)
    y = df.pop("level").to_numpy()
    n = len(df)
    usable = [c for c in df.columns if df[c].std() > 1e-9]
    x = df[usable].to_numpy()

    mi = mutual_info_regression(x, y, random_state=0)
    # Empirical null: MI of shuffled targets, to know what "nothing" looks like.
    rs = np.random.default_rng(0)
    null = np.concatenate([
        mutual_info_regression(x, rs.permutation(y), random_state=0) for _ in range(3)
    ])
    floor = float(np.percentile(null, 99))

    pearson = {c: float(np.corrcoef(df[c], y)[0, 1]) for c in usable}
    ranked = sorted(zip(usable, mi), key=lambda kv: -kv[1])[:top]
    best = float(ranked[0][1]) if ranked else 0.0

    flag = "OK  " if best > 3 * floor else ("WEAK" if best > 1.5 * floor else "FAIL")
    print(f"\n[{flag}] {criterion}   best MI = {best:.2f}   (n={n}, null 99th pct {floor:.2f})")
    for name, m in ranked:
        mark = " " if m > 1.5 * floor else "."
        print(f"      {mark} {name:34s} MI={m:.3f}  r={pearson[name]:+.2f}")
    return best


if __name__ == "__main__":
    targets = sys.argv[1:] or CRITERIA
    scores = {c: report(c) for c in targets}
    print("\n" + "=" * 60)
    print("SUMMARY (best single-feature |r| against the swept latent)")
    for c, s in sorted(scores.items(), key=lambda kv: kv[1]):
        print(f"  {c:22s} {s:.2f}")
