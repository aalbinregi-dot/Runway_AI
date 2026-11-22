"""Generate the synthetic corpus and extract its feature matrix.

Writes, under data/:
  interim/synthetic_keypoints.csv   long-format keypoints (workbook schema)
  processed/features.csv            one feature row per clip
  processed/labels.csv              twelve criterion scores + latent params

Usage:  python scripts/build_dataset.py [--n 1200] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runway_ai.config import CORE_13, INTERIM_DIR, PROCESSED_DIR, RANDOM_SEED  # noqa: E402
from runway_ai.data.loaders import preprocess                                  # noqa: E402
from runway_ai.data.synthetic import make_clip, sample_params                  # noqa: E402
from runway_ai.features.builder import extract_features                        # noqa: E402


def build(n_clips: int, seed: int, correlated: bool, prefix: str) -> tuple[pd.DataFrame, ...]:
    rng = np.random.default_rng(seed)
    keypoints, feats, labels = [], [], []
    t0 = time.time()

    for i in range(n_clips):
        vid = f"{prefix}_{i:05d}"
        params = sample_params(rng, correlated=correlated)
        seq, meta = make_clip(vid, params, rng)
        keypoints.append(seq.to_long())

        row = {"video_id": vid}
        row.update(extract_features(preprocess(seq)))
        feats.append(row)
        labels.append(meta)

        if (i + 1) % 100 == 0:
            rate = (i + 1) / (time.time() - t0)
            print(f"  {i + 1}/{n_clips} clips  ({rate:.0f}/s)", flush=True)

    return (
        pd.concat(keypoints, ignore_index=True),
        pd.DataFrame(feats).set_index("video_id").sort_index(),
        pd.DataFrame(labels).set_index("video_id").sort_index(),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200, help="number of clips")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--prefix", default="syn")
    ap.add_argument("--independent", action="store_true",
                    help="draw the twelve qualities independently (diagnostic set)")
    ap.add_argument("--no-keypoints", action="store_true",
                    help="skip writing the (large) raw keypoint CSV")
    args = ap.parse_args()

    print(f"Generating {args.n} clips (seed={args.seed}, "
          f"{'independent' if args.independent else 'correlated'} latents)")
    kp, feats, labels = build(args.n, args.seed, not args.independent, args.prefix)

    suffix = "_indep" if args.independent else ""
    if not args.no_keypoints:
        path = INTERIM_DIR / f"synthetic_keypoints{suffix}.csv"
        kp.to_csv(path, index=False)
        print(f"  keypoints -> {path}  ({len(kp):,} rows)")

    feats.to_csv(PROCESSED_DIR / f"features{suffix}.csv")
    labels.to_csv(PROCESSED_DIR / f"labels{suffix}.csv")
    print(f"  features  -> {PROCESSED_DIR / f'features{suffix}.csv'}  {feats.shape}")
    print(f"  labels    -> {PROCESSED_DIR / f'labels{suffix}.csv'}  {labels.shape}")

    # Landmark vocabulary check against the supplied workbook's schema.
    assert set(kp["landmark"].unique()) == set(CORE_13), "landmark set drifted"
    print("  schema matches the sample workbook (13 landmarks, 7 columns)")


if __name__ == "__main__":
    main()
