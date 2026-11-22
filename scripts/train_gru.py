"""Train the bidirectional GRU per-criterion regressors.

This script is the GRU equivalent of ``scripts/train_model.py``. It reads the
raw keypoint CSV produced by ``build_dataset.py``, builds padded sequence
tensors, and trains one GRU per criterion using 5-fold out-of-fold evaluation.

Epoch metrics are logged to ``reports/gru_history/{criterion}.csv`` and
training curves are saved as ``reports/gru_history/{criterion}_curves.png``.
Trained models are saved as ``models/gru_{criterion}.pt``.

Usage
-----
    python scripts/train_gru.py
    python scripts/train_gru.py --criteria posture balance
    python scripts/train_gru.py --epochs 30 --device mps
    python scripts/train_gru.py --folds 3 --hidden 64
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runway_ai.config import CRITERIA, PROCESSED_DIR                            # noqa: E402
from runway_ai.models.gru import train_criterion, _TORCH_AVAILABLE             # noqa: E402
from runway_ai.models.sequence import build_sequences, load_sequences_from_csv  # noqa: E402


def main() -> None:
    if not _TORCH_AVAILABLE:
        print(
            "PyTorch is not installed. Install it with:\n"
            "  pip install torch\n"
            "or see https://pytorch.org for platform-specific instructions."
        )
        sys.exit(1)

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--criteria", nargs="+", default=None, metavar="CRITERION",
        help="Which criteria to train (default: all twelve)."
    )
    ap.add_argument("--folds",   type=int, default=5,   help="OOF folds (default 5)")
    ap.add_argument("--epochs",  type=int, default=50,  help="Max epochs (default 50)")
    ap.add_argument("--patience",type=int, default=5,   help="Early-stop patience (default 5)")
    ap.add_argument("--hidden",  type=int, default=128, help="GRU hidden dim (default 128)")
    ap.add_argument("--batch",   type=int, default=32,  help="Batch size (default 32)")
    ap.add_argument("--lr",      type=float, default=1e-3)
    ap.add_argument(
        "--device", default="cpu",
        help="PyTorch device string: cpu | cuda | mps (default cpu)"
    )
    ap.add_argument(
        "--keypoints", default=None,
        help="Long-format keypoints CSV. Defaults to data/interim/keypoints.csv "
             "if it exists, otherwise data/raw/keypoints.csv."
    )
    ap.add_argument(
        "--labels", default=str(PROCESSED_DIR / "labels.csv"),
        help="Labels CSV with columns score_{criterion} (default: processed/labels.csv)."
    )
    args = ap.parse_args()

    # -----------------------------------------------------------------------
    # Locate keypoints CSV
    # -----------------------------------------------------------------------
    if args.keypoints:
        kp_path = Path(args.keypoints)
    else:
        interim = ROOT / "data" / "interim" / "keypoints.csv"
        raw     = ROOT / "data" / "raw"     / "keypoints.csv"
        kp_path = interim if interim.exists() else raw

    if not kp_path.exists():
        print(
            f"Keypoints CSV not found at {kp_path}.\n"
            "Run ``python scripts/build_dataset.py`` first, or pass --keypoints."
        )
        sys.exit(1)

    print(f"Loading keypoints from {kp_path} ...")
    sequences = load_sequences_from_csv(kp_path)
    print(f"  {len(sequences)} clips loaded.\n")

    # -----------------------------------------------------------------------
    # Labels
    # -----------------------------------------------------------------------
    labels_df = pd.read_csv(args.labels, index_col="video_id")

    # -----------------------------------------------------------------------
    # Train per criterion
    # -----------------------------------------------------------------------
    target_criteria = args.criteria or CRITERIA
    results = []

    for crit in target_criteria:
        col = f"score_{crit}"
        if col not in labels_df.columns:
            print(f"  Skipping {crit}: no column '{col}' in labels CSV.")
            continue

        # Map video_id -> label for this criterion.
        lmap = {str(vid): float(score)
                for vid, score in labels_df[col].items()
                if not np.isnan(float(score))}

        if not lmap:
            print(f"  Skipping {crit}: all labels are NaN.")
            continue

        res = train_criterion(
            sequences=sequences,
            labels=lmap,
            criterion=crit,
            n_splits=args.folds,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch,
            lr=args.lr,
            hidden=args.hidden,
            device=args.device,
        )
        results.append(res)

    # -----------------------------------------------------------------------
    # Summary table
    # -----------------------------------------------------------------------
    if results:
        valid = [r for r in results if "mae" in r]
        print("\n" + "=" * 60)
        print(f"{'Criterion':<22} {'MAE':>7} {'R²':>7} {'RMSE':>7}")
        print("-" * 60)
        for r in valid:
            print(f"  {r['criterion']:<20} {r['mae']:>7.3f} {r['r2']:>+7.3f} {r['rmse']:>7.3f}")
        print("=" * 60)
        if valid:
            print(
                f"mean MAE {np.mean([r['mae'] for r in valid]):.3f} | "
                f"mean R² {np.mean([r['r2'] for r in valid]):+.3f}"
            )
        print(f"\nModels  -> {ROOT / 'models'}")
        print(f"History -> {ROOT / 'reports' / 'gru_history'}")


if __name__ == "__main__":
    main()
