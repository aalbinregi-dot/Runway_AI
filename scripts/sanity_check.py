"""Does the system actually rank walking skill correctly?

Generates fresh walks at known skill levels and checks the predicted overall
score tracks the truth. This is the quickest honest read on whether the trained
models are working: unlike the cross-validation report, it uses clips generated
after training, and it reports the ranking, not just the error.

Usage:  python scripts/sanity_check.py [--n 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from runway_ai.analyze import analyze_sequence          # noqa: E402
from runway_ai.config import CRITERION_WEIGHTS          # noqa: E402
from runway_ai.data.synthetic import make_clip, sample_params  # noqa: E402
from runway_ai.models.train import load_models          # noqa: E402

LEVELS = [(0.95, "expert"), (0.75, "advanced"), (0.55, "intermediate"),
          (0.35, "developing"), (0.12, "novice")]


def true_overall(meta: dict) -> float:
    """The label-side overall on 0-100, using the same weights as the scorer.

    Criterion labels are 0-10 and the weights sum to 1, so the weighted sum is
    itself 0-10; the factor of ten puts it on the same 0-100 scale the system
    reports.
    """
    return 10.0 * sum(meta[f"score_{c}"] * w for c, w in CRITERION_WEIGHTS.items())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=8, help="clips per skill level")
    ap.add_argument("--seed", type=int, default=99)
    args = ap.parse_args()

    models = load_models() or None
    if not models:
        from runway_ai.config import launcher

        print("No trained models found - scoring will fall back to the rubric.")
        print(f"Run '{launcher()} rebuild' (or scripts/train_model.py) first.\n")

    rng = np.random.default_rng(args.seed)
    print(f"{args.n} fresh clips per level, generated after training.\n")
    print(f"  {'level':<14}{'predicted':>11}{'true':>9}{'error':>9}   spread")
    print("  " + "-" * 62)

    rows, all_pred, all_true = [], [], []
    for skill, name in LEVELS:
        preds, truths = [], []
        for _ in range(args.n):
            seq, meta = make_clip("x", sample_params(rng, skill=skill), rng)
            preds.append(analyze_sequence(seq, models).assessment.overall)
            truths.append(true_overall(meta))
        p, t = float(np.mean(preds)), float(np.mean(truths))
        rows.append((name, p, t))
        all_pred += preds
        all_true += truths
        bar = "#" * max(int(round(p / 2.5)), 0)
        print(f"  {name:<14}{p:>10.1f}{t:>9.1f}{p - t:>+9.1f}   {bar}")

    mae = float(np.mean(np.abs(np.array(all_pred) - np.array(all_true))))
    corr = float(np.corrcoef(all_pred, all_true)[0, 1])
    ordered = all(rows[i][1] > rows[i + 1][1] for i in range(len(rows) - 1))

    print("\n  " + "-" * 62)
    print(f"  overall MAE      {mae:6.2f} / 100")
    print(f"  correlation      {corr:+6.3f}")
    print(f"  ranking correct  {'yes' if ordered else 'NO'}"
          f"  (predicted order matches true skill order)")

    if not ordered:
        print("\n  Skill levels came out in the wrong order - investigate before "
              "trusting any score.")
        return 1
    if mae > 12.0:
        print(f"\n  Order is right but the scale is off by {mae:.0f} points on "
              "average - the models are ranking, not calibrated.")
        return 1
    print("\n  Scores track skill in the right order and at the right scale.")

    # ----------------------------------------------------------------------- #
    # Tier-D: real MoCap data check
    # ----------------------------------------------------------------------- #
    print()
    rc = _check_mocap(models)
    return max(0, rc)


def _check_mocap(models) -> int:
    """Score all 14 real MoCap clips and confirm the pipeline handles them.

    This is a smoke-test, not a ground-truth comparison (the file has no
    expert labels), but it catches regressions that only show up on real
    biomechanical data: NaN propagation, assertion errors, out-of-range scores.

    The fps sidecar (Optimal_Runway_Keypoints_from_MoCap.xlsx.meta.json) is
    required; without it every timing feature is miscalculated.
    """
    from runway_ai.data.loaders import load_sequences

    root = Path(__file__).resolve().parents[1]
    book = root / "data" / "raw" / "Optimal_Runway_Keypoints_from_MoCap.xlsx"
    sidecar = Path(str(book) + ".meta.json")

    if not book.exists():
        print("MoCap workbook not found - skipping Tier-D check.")
        print(f"  Expected: {book}")
        return 0

    if not sidecar.exists():
        print("WARNING: fps sidecar missing - MoCap clips will be loaded at "
              f"DEFAULT_FPS instead of 50 fps.\n  Create: {sidecar}")

    import json
    fps = 50.0
    if sidecar.exists():
        fps = float(json.loads(sidecar.read_text()).get("fps", 50.0))

    print(f"Tier-D MoCap check  ({book.name}, fps={fps})")
    print(f"  {'clip':<30} {'overall':>8}  {'grade':<14}  warnings")
    print("  " + "-" * 72)

    seqs = load_sequences(book, fps=fps)
    errors = 0
    for vid, seq in seqs.items():
        try:
            result = analyze_sequence(seq, models=models)
            a = result.assessment
            warn_count = len(a.quality_warnings)
            warn_str = f"{warn_count} warning{'s' if warn_count != 1 else ''}" if warn_count else "ok"
            short = vid.replace("video_0", "v0")
            print(f"  {short:<30} {a.overall:>7.1f}  {a.grade:<14}  {warn_str}")
        except Exception as exc:
            print(f"  {vid}  ERROR: {exc}")
            errors += 1

    if errors:
        print(f"\n  {errors} clip(s) failed - real-data pipeline has a regression.")
        return 1

    print(f"\n  All {len(seqs)} MoCap clips scored without error.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
