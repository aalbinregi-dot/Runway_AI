"""Print the model accuracy table.

A separate script rather than an inline one-liner in run.bat: batch eats bare
``%`` signs, so format specifiers like ``%.3f`` silently become ``.3f`` and the
command dies with a confusing TypeError.

Usage:  python scripts/show_results.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

CSV = ROOT / "reports" / "model_comparison.csv"


def main() -> int:
    if not CSV.exists():
        from runway_ai.config import launcher

        print(f"No results yet. Run '{launcher()} rebuild' first.")
        return 1

    d = pd.read_csv(CSV, index_col=0)
    print(f"  {'criterion':<20}{'MAE':>7}{'R2':>8}{'vs mean':>9}"
          f"{'vs ridge':>10}{'vs rubric':>11}{'within 1':>10}  model")
    print("  " + "-" * 84)
    for name, r in d.sort_values("R2", ascending=False).iterrows():
        print(f"  {name:<20}{r['MAE']:>7.3f}{r['R2']:>8.3f}"
              f"{r['MAE_mean']:>9.3f}{r['MAE_ridge']:>10.3f}{r['MAE_rubric']:>11.3f}"
              f"{r['within_1pt'] * 100:>9.1f}%  {r['chosen']}")

    print("  " + "-" * 84)
    print(f"  {'MEAN':<20}{d['MAE'].mean():>7.3f}{d['R2'].mean():>8.3f}"
          f"{d['MAE_mean'].mean():>9.3f}{d['MAE_ridge'].mean():>10.3f}"
          f"{d['MAE_rubric'].mean():>11.3f}{d['within_1pt'].mean() * 100:>9.1f}%")
    print()
    print(f"  Everything is out-of-fold. Lower MAE is better; 'vs mean' is the")
    print(f"  predict-the-average baseline, so the model must beat it to matter.")
    print(f"  Beats the rubric on {int(d['beats_rubric'].sum())}/{len(d)} criteria, "
          f"ridge on {int(d['beats_ridge'].sum())}/{len(d)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
