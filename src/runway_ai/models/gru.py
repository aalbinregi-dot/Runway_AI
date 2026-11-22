"""Bidirectional GRU regressor for temporal runway-walk scoring.

Operates on the keypoint sequence tensors produced by
``runway_ai.models.sequence`` rather than the tabular feature vectors used
by the main ensemble.  This gives the model access to the raw timing and
shape trajectory, at the cost of requiring PyTorch and a longer training time.

Architecture
------------
* 2-layer bidirectional GRU, hidden dim 128 → effective hidden dim 256.
* Global-average pooling over the time axis (real frames only, via mask).
* A single linear head → 1 output clipped to [0, 10].

One model is trained **per criterion**, matching the per-criterion strategy of
the tabular models.

Training
--------
* Adam, lr=1e-3, weight_decay=1e-4.
* Huber loss (delta=1.0).
* Early stopping: patience=5 epochs without validation improvement.
* Best weights are restored before returning.
* Epoch metrics are logged to ``reports/gru_history/{criterion}.csv`` and a
  four-panel training curve PNG is saved alongside it.

Integration with the main pipeline
-----------------------------------
When ``models/gru_{criterion}.pt`` exists, ``assess.assess()`` averages the
GRU prediction with the tabular model's prediction, giving a light ensemble
boost without requiring GRU training to be mandatory.
"""
from __future__ import annotations

import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import CRITERIA, MODEL_DIR, RANDOM_SEED, REPORT_DIR
from .sequence import ClipSequence, SEQ_DIM

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------
if _TORCH_AVAILABLE:
    class _GRUNet(nn.Module):
        def __init__(self, input_dim: int = SEQ_DIM, hidden: int = 128,
                     n_layers: int = 2, dropout: float = 0.2):
            super().__init__()
            self.gru = nn.GRU(
                input_dim, hidden,
                num_layers=n_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.head = nn.Linear(hidden * 2, 1)  # *2 for bidirectional

        def forward(self, x: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
            # x:    (B, T, D)
            # mask: (B, T) bool — True = real frame
            out, _ = self.gru(x)             # (B, T, hidden*2)
            # Masked mean pooling.
            m = mask.unsqueeze(-1).float()   # (B, T, 1)
            pooled = (out * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (B, hidden*2)
            return self.head(pooled).squeeze(-1)  # (B,)
else:
    class _GRUNet:  # type: ignore
        pass


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------
@dataclass
class GRUEpochLog:
    epoch: int
    train_loss: float
    val_loss: float
    val_mae: float
    val_rmse: float
    val_r2: float

    def as_row(self) -> dict:
        return {
            "epoch":      self.epoch,
            "train_loss": round(self.train_loss, 5),
            "val_loss":   round(self.val_loss, 5),
            "val_mae":    round(self.val_mae, 4),
            "val_rmse":   round(self.val_rmse, 4),
            "val_r2":     round(self.val_r2, 4),
        }


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / (ss_tot + 1e-12)


def _save_history(history: list[GRUEpochLog], criterion: str) -> None:
    hist_dir = REPORT_DIR / "gru_history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    csv_path = hist_dir / f"{criterion}.csv"
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(history[0].as_row()))
        writer.writeheader()
        writer.writerows(row.as_row() for row in history)


def _save_curves(history: list[GRUEpochLog], criterion: str) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return  # matplotlib is optional here

    epochs = [h.epoch for h in history]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(f"GRU training — {criterion}", fontsize=13)

    def _plot(ax, y_train, y_val, title, ylabel):
        ax.plot(epochs, y_train, label="train")
        ax.plot(epochs, y_val, label="val")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(alpha=0.3)

    _plot(axes[0, 0], [h.train_loss for h in history], [h.val_loss for h in history],
          "Loss (Huber)", "loss")
    _plot(axes[0, 1], [h.val_mae for h in history], [h.val_mae for h in history],
          "Validation MAE", "MAE")
    _plot(axes[1, 0], [h.val_rmse for h in history], [h.val_rmse for h in history],
          "Validation RMSE", "RMSE")
    _plot(axes[1, 1], [h.val_r2 for h in history], [h.val_r2 for h in history],
          "Validation R²", "R²")

    plt.tight_layout()
    hist_dir = REPORT_DIR / "gru_history"
    hist_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(hist_dir / f"{criterion}_curves.png", dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------
def train_criterion(
    sequences: list[ClipSequence],
    labels: dict[str, float],
    criterion: str,
    n_splits: int = 5,
    max_epochs: int = 50,
    patience: int = 5,
    batch_size: int = 32,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden: int = 128,
    device: str = "cpu",
    seed: int = RANDOM_SEED,
    out_dir: Path | None = None,
) -> dict:
    """Train and evaluate a GRU for one criterion.

    Returns an OOF metrics dict and saves the final model to
    ``models/gru_{criterion}.pt``.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError(
            "PyTorch is required for the GRU model. "
            "Install it with: pip install torch"
        )

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)
    out_dir = Path(out_dir or MODEL_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Filter to clips that have a label for this criterion.
    paired = [(s, labels[s.video_id]) for s in sequences if s.video_id in labels]
    if len(paired) < n_splits * 2:
        return {"error": f"Too few labeled clips for {criterion}: {len(paired)}"}

    # Build tensors.
    X_t = torch.tensor(np.stack([p[0].frames for p in paired]), dtype=torch.float32)
    M_t = torch.tensor(np.stack([p[0].mask   for p in paired]), dtype=torch.bool)
    y_np = np.array([p[1] for p in paired], dtype=np.float32)
    y_t  = torch.tensor(y_np, dtype=torch.float32)

    n = len(paired)
    order = np.argsort(y_np)
    folds = np.zeros(n, dtype=int)
    folds[order] = np.arange(n) % n_splits

    oof = np.full(n, np.nan)
    history: list[GRUEpochLog] = []

    # Cross-validation for OOF metrics.
    print(f"  GRU [{criterion}] — {n} clips, {n_splits}-fold OOF ...")
    for k in range(n_splits):
        tr_idx = np.where(folds != k)[0]
        te_idx = np.where(folds == k)[0]

        tr_ds = TensorDataset(X_t[tr_idx], M_t[tr_idx], y_t[tr_idx])
        te_ds = TensorDataset(X_t[te_idx], M_t[te_idx], y_t[te_idx])
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True)
        te_loader = DataLoader(te_ds, batch_size=batch_size)

        model = _GRUNet(SEQ_DIM, hidden).to(dev)
        opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_fn = nn.HuberLoss(delta=1.0)

        best_val, best_state, no_improve = float("inf"), None, 0
        fold_hist: list[GRUEpochLog] = []

        for epoch in range(1, max_epochs + 1):
            # -- Train --
            model.train()
            train_losses = []
            for xb, mb, yb in tr_loader:
                xb, mb, yb = xb.to(dev), mb.to(dev), yb.to(dev)
                opt.zero_grad()
                pred = model(xb, mb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                train_losses.append(float(loss.item()))

            # -- Validate --
            model.eval()
            val_preds, val_true = [], []
            with torch.no_grad():
                for xb, mb, yb in te_loader:
                    xb, mb = xb.to(dev), mb.to(dev)
                    val_preds.append(model(xb, mb).cpu().numpy())
                    val_true.append(yb.numpy())
            vp = np.clip(np.concatenate(val_preds), 0, 10)
            vt = np.concatenate(val_true)
            val_loss = float(np.mean(np.abs(vp - vt)))  # MAE as val loss

            log = GRUEpochLog(
                epoch=epoch,
                train_loss=float(np.mean(train_losses)),
                val_loss=val_loss,
                val_mae=float(np.mean(np.abs(vp - vt))),
                val_rmse=float(np.sqrt(np.mean((vp - vt) ** 2))),
                val_r2=_r2(vt, vp),
            )
            fold_hist.append(log)

            if val_loss < best_val - 1e-4:
                best_val, best_state, no_improve = val_loss, {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        if best_state:
            model.load_state_dict(best_state)
        # Store OOF predictions.
        model.eval()
        with torch.no_grad():
            for xb, mb, yb in te_loader:
                p = model(xb.to(dev), mb.to(dev)).cpu().numpy()
                oof[te_idx[:len(p)]] = np.clip(p, 0, 10)

        # Accumulate history from all folds (epoch 1 of each fold appended).
        if not history:
            history = fold_hist

    # Save epoch history from the last fold (representative training curve).
    _save_history(history, criterion)
    _save_curves(history, criterion)

    # Train final model on all data.
    print(f"  GRU [{criterion}] — training final model on all {n} clips ...")
    full_ds     = TensorDataset(X_t, M_t, y_t)
    full_loader = DataLoader(full_ds, batch_size=batch_size, shuffle=True)
    final_model = _GRUNet(SEQ_DIM, hidden).to(dev)
    final_opt   = torch.optim.Adam(final_model.parameters(), lr=lr, weight_decay=weight_decay)
    final_loss_fn = nn.HuberLoss(delta=1.0)

    for _ in range(max_epochs):
        final_model.train()
        for xb, mb, yb in full_loader:
            xb, mb, yb = xb.to(dev), mb.to(dev), yb.to(dev)
            final_opt.zero_grad()
            final_loss_fn(final_model(xb, mb), yb).backward()
            final_opt.step()

    torch.save(
        {"model_state": final_model.state_dict(), "input_dim": SEQ_DIM,
         "hidden": hidden, "criterion": criterion},
        out_dir / f"gru_{criterion}.pt",
    )

    oof_valid = ~np.isnan(oof)
    y_valid   = y_np[oof_valid]
    oof_v     = oof[oof_valid]
    result = {
        "criterion": criterion,
        "n": n,
        "mae":    round(float(np.mean(np.abs(oof_v - y_valid))), 4),
        "r2":     round(_r2(y_valid, oof_v), 4),
        "rmse":   round(float(np.sqrt(np.mean((oof_v - y_valid) ** 2))), 4),
        "epochs_logged": len(history),
    }
    print(f"  GRU [{criterion}] OOF MAE={result['mae']:.3f}  R²={result['r2']:+.3f}")
    return result


def load_gru_model(criterion: str, model_dir: Path | None = None) -> object | None:
    """Load a saved GRU model, returning ``None`` if not found or torch unavailable."""
    if not _TORCH_AVAILABLE:
        return None
    model_dir = Path(model_dir or MODEL_DIR)
    path = model_dir / f"gru_{criterion}.pt"
    if not path.exists():
        return None
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        net = _GRUNet(ckpt.get("input_dim", SEQ_DIM), ckpt.get("hidden", 128))
        net.load_state_dict(ckpt["model_state"])
        net.eval()
        return net
    except Exception:
        return None


def predict_gru(
    sequence: ClipSequence,
    criterion: str,
    model_dir: Path | None = None,
) -> float | None:
    """Predict one criterion score from a sequence using the saved GRU model.

    Returns ``None`` when no model is available.
    """
    net = load_gru_model(criterion, model_dir)
    if net is None:
        return None
    try:
        import torch
        x = torch.tensor(sequence.frames[None], dtype=torch.float32)
        m = torch.tensor(sequence.mask[None],   dtype=torch.bool)
        with torch.no_grad():
            score = float(net(x, m).squeeze().item())
        return float(np.clip(score, 0.0, 10.0))
    except Exception:
        return None
