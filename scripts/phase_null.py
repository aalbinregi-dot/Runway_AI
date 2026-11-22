"""Measure what the phase split reports on walkers who never change.

Splitting a pass into approach / middle / exit invites a narrative: "the shoulder
line holds and then drops". The trouble is that the three phases are not filmed
alike. In a walk toward the camera the approach is further away, the figure is
smaller, and the landmarks are noisier - so a systematic approach-to-exit
difference can be the camera rather than the walker.

Every clip here is generated with the walker's twelve qualities held constant for
its whole duration, so the true phase-to-phase change is exactly zero. Whatever
comes out is the artifact, and it becomes the null distribution a real change has
to beat.

The measurement also calibrates the gate. A per-trace 95% test reported a change
on most constant-quality clips, simply because there are a dozen traces. The
multiplier in `scoring.phases` was chosen against one batch of clips and then
validated on another, which is what ``--validate`` does.

    python scripts/phase_null.py [--n 120] [--validate 150]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runway_ai.config import REPORT_DIR                        # noqa: E402
from runway_ai.data.loaders import preprocess                   # noqa: E402
from runway_ai.data.synthetic import make_clip, sample_params   # noqa: E402
from runway_ai.features import timeline as tl                   # noqa: E402
from runway_ai.scoring import phases                            # noqa: E402


def false_rate(seed_start: int, n: int) -> tuple[float, float, int]:
    """Fraction of constant-quality clips reporting at least one change."""
    clips = any_change = claims = 0
    for i in range(n):
        rng = np.random.default_rng(seed_start + i)
        params = sample_params(rng, correlated=False)
        seq, _ = make_clip(f"v_{i}", params, rng)
        try:
            prepared = preprocess(copy.deepcopy(seq))
            got = phases.analyse(prepared, tl.build(prepared, []))
        except Exception:
            continue
        if not got:
            continue
        clips += 1
        if got["changes"]:
            any_change += 1
        claims += len(got["changes"])
    if not clips:
        return 0.0, 0.0, 0
    return any_change / clips, claims / clips, clips


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120, help="clips for the null table")
    ap.add_argument("--validate", type=int, default=150,
                    help="further clips, never used to fit, for the false rate")
    ap.add_argument("--seed", type=int, default=11)
    args = ap.parse_args()

    print(f"Measuring the phase-split artifact on {args.n} constant-quality clips...")
    result = phases.measure_null(n_clips=args.n, seed=args.seed)
    print(f"  {result['n_clips']} clips were long enough to split\n")

    print(f"  {'trace':22s} {'mean':>9s} {'sd':>8s}   "
          f"gate (x{phases.CRITICAL_MULTIPLIER})")
    for name, stat in sorted(result["per_trace"].items(),
                             key=lambda kv: -abs(kv[1]["mean"])):
        gate = phases.CRITICAL_MULTIPLIER * stat["sd"]
        print(f"  {name:22s} {stat['mean']:+9.4f} {stat['sd']:8.4f}   +/-{gate:.4f}")

    out = REPORT_DIR / "phase_null.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    phases._null_cache = None            # pick up what was just written

    if args.validate:
        print(f"\nValidating on {args.validate} clips never used to fit the null...")
        rate, per_clip, clips = false_rate(90_000, args.validate)
        print(f"  clips                      : {clips}")
        print(f"  reporting any false change : {rate * 100:.1f}%   (target ~5%)")
        print(f"  false claims per clip      : {per_clip:.3f}")
        result["validation"] = {
            "clips": clips, "family_wise_false_rate": round(rate, 3),
            "claims_per_clip": round(per_clip, 3),
            "multiplier": phases.CRITICAL_MULTIPLIER,
            "note": ("held-out constant-quality clips; the true phase change is "
                     "zero by construction, so any reported change is spurious"),
        }
        out.write_text(json.dumps(result, indent=2))

    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
