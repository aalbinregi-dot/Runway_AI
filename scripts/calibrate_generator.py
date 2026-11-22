"""Measure - and correct - the gap between generated and real walking.

Why this exists
---------------
The first real video pushed through the system landed *outside* the synthetic
training distribution on seven features, every one of them in the same
direction: the generator produced far more movement than a real walker does.
Hip obliquity was roughly six times too large, lateral sway nearly three times.
A model trained on that data has to extrapolate on real footage, which is
exactly the failure mode a synthetic-data project is most at risk of and least
likely to notice.

Method
------
Reference values come from gait-analysis literature (peak-to-peak ranges for
normal adult walking), not from a single subject - fitting the generator to one
person would trade one bias for another. A real clip, when supplied, is used
only as a cross-check that the targets are in the right region.

Each target names the generator constant that controls it, so the output is a
concrete list of scale factors to apply rather than an observation.

Usage:
    python scripts/calibrate_generator.py
    python scripts/calibrate_generator.py --real data/interim/_last_video.csv --fps 59.64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runway_ai.data.loaders import preprocess              # noqa: E402
from runway_ai.data.schema import long_to_sequences        # noqa: E402
from runway_ai.features.builder import extract_features    # noqa: E402

#: feature -> (target value for a *good* walker, generator knob it comes from)
#:
#: Targets are the value a competent walker should produce, so they are aimed at
#: roughly the 25th percentile of the synthetic spread: better than average, not
#: flawless. Sources are noted per line.
TARGETS: dict[str, tuple[float, str]] = {
    # Pelvic obliquity in normal gait spans ~10 deg peak-to-peak (Perry &
    # Burnfield); the projected measurement inflates that somewhat.
    "hip_obliquity_amp": (4.0, "pelvic_drop coefficient"),
    "hip_obliquity_std": (1.6, "pelvic_drop coefficient"),
    # Mediolateral COM excursion is ~3-5 cm, i.e. ~0.1 torso lengths.
    "balance_sway_amp": (0.22, "sway_amp coefficient"),
    "balance_upper_body_sway": (0.16, "sway_amp coefficient"),
    # Trunk axial rotation is ~7-10 deg total.
    "hip_counter_rotation_amp": (0.055, "trunk_rot coefficient"),
    # Swing-phase knee rise relative to the hip.
    "stride_knee_lift": (0.14, "clearance coefficient"),
    # Wrist-to-wrist span; a relaxed carriage keeps the arms near the body.
    "conf_arm_spread": (1.95, "abduct coefficient"),
}

PCT_FOR_GOOD = 25.0     # where a good walker should sit in the synthetic spread


def synthetic_stats(path: Path) -> pd.DataFrame:
    X = pd.read_csv(path, index_col="video_id")
    rows = []
    for k, (target, knob) in TARGETS.items():
        if k not in X.columns:
            continue
        at_pct = float(np.percentile(X[k], PCT_FOR_GOOD))
        rows.append({
            "feature": k,
            "target": target,
            f"synth_p{PCT_FOR_GOOD:.0f}": round(at_pct, 4),
            "synth_median": round(float(X[k].median()), 4),
            "scale_needed": round(target / at_pct, 3) if at_pct > 1e-9 else float("nan"),
            "knob": knob,
        })
    return pd.DataFrame(rows)


def real_values(path: Path, fps: float) -> dict:
    seq = next(iter(long_to_sequences(pd.read_csv(path), fps=fps).values()))
    return extract_features(preprocess(seq))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.csv")
    ap.add_argument("--real", default=None, help="a real extracted keypoint CSV")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    df = synthetic_stats(Path(args.features))
    if args.real and Path(args.real).exists():
        real = real_values(Path(args.real), args.fps)
        df = df.assign(real_clip=[round(real.get(k, float("nan")), 4)
                                  for k in df["feature"]])

    pd.set_option("display.width", 200)
    print("Generator amplitude calibration")
    print("Targets are literature values for a competent adult walker.\n")
    print(df.to_string(index=False))

    print("\nSuggested scale factors, grouped by generator constant:")
    for knob, group in df.groupby("knob"):
        factors = group["scale_needed"].dropna()
        if factors.empty:
            continue
        print(f"  {knob:28s} x{factors.mean():.2f}  "
              f"(from {', '.join(group['feature'])})")

    print("\nA factor well below 1.0 means the generator moves that joint more "
          "than a real walker does,\nso models trained on it extrapolate when "
          "shown real footage.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
