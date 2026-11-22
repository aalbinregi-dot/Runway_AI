"""Out-of-distribution detection: is this clip one the models have seen before?

The models are trained on generated walks. Cross-validation says nothing about
what happens when a real clip lands outside that distribution - the regressor
still returns a confident-looking number, it is simply extrapolating.

That is not hypothetical. The first real video pushed through this system sat
outside the training range on seven features at once, every one in the same
direction, because the generator moved every joint more than a real walker
does. The scores looked entirely plausible.

So the training feature ranges are stored at fit time, and every prediction is
checked against them. A criterion whose evidence is out of range is reported
with that caveat attached rather than presented as equally trustworthy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import CRITERIA, MODEL_DIR
from .rubric import RUBRIC

REFERENCE_PATH = MODEL_DIR / "feature_reference.json"

#: A feature is "outside" if it sits beyond this many IQRs past the training
#: quartiles. 1.5 is the standard Tukey fence; anything past 3 is far out.
FENCE = 1.5
FAR_FENCE = 3.0


def build_reference(features: pd.DataFrame, path: Path | None = None) -> dict:
    """Record the training distribution of every feature. Called after training."""
    path = Path(path or REFERENCE_PATH)
    ref = {}
    for col in features.columns:
        q1, q50, q3 = (float(v) for v in np.percentile(features[col], [25, 50, 75]))
        ref[col] = {
            "p01": float(np.percentile(features[col], 1)),
            "q1": q1, "median": q50, "q3": q3,
            "p99": float(np.percentile(features[col], 99)),
            "iqr": max(q3 - q1, 1e-9),
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(ref, indent=2))
    return ref


def load_reference(path: Path | None = None) -> dict | None:
    path = Path(path or REFERENCE_PATH)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def feature_deviation(value: float, stats: dict) -> float:
    """How far outside the training quartiles a value sits, in IQRs. 0 = inside."""
    if value > stats["q3"]:
        return (value - stats["q3"]) / stats["iqr"]
    if value < stats["q1"]:
        return (stats["q1"] - value) / stats["iqr"]
    return 0.0


def check(feats: dict, reference: dict | None = None) -> dict:
    """Assess how far this clip sits from the training distribution.

    Returns per-criterion confidence plus the specific features responsible, so
    a low-confidence score can be explained rather than merely doubted.
    """
    reference = reference or load_reference()
    if not reference:
        return {"available": False, "criteria": {}, "outliers": [], "overall": "unknown"}

    outliers = []
    for name, value in feats.items():
        stats = reference.get(name)
        if not stats:
            continue
        dev = feature_deviation(float(value), stats)
        if dev > FENCE:
            outliers.append({
                "feature": name,
                "value": round(float(value), 4),
                "train_q1": round(stats["q1"], 4),
                "train_q3": round(stats["q3"], 4),
                "iqrs_outside": round(dev, 2),
                "far": dev > FAR_FENCE,
            })
    outliers.sort(key=lambda o: -o["iqrs_outside"])

    by_feature = {o["feature"]: o for o in outliers}
    criteria = {}
    for crit in CRITERIA:
        hits = [by_feature[b.feature] for b in RUBRIC[crit] if b.feature in by_feature]
        if not hits:
            level = "high"
        elif any(h["far"] for h in hits) or len(hits) >= 2:
            level = "low"
        else:
            level = "moderate"
        criteria[crit] = {"confidence": level,
                          "features_outside": [h["feature"] for h in hits]}

    n_low = sum(1 for c in criteria.values() if c["confidence"] == "low")
    overall = "low" if n_low >= 4 else ("moderate" if n_low >= 1 else "high")
    return {"available": True, "criteria": criteria, "outliers": outliers[:12],
            "overall": overall, "n_features_outside": len(outliers)}


def warnings_from(coverage: dict) -> list[str]:
    """Human-readable warnings for the report."""
    if not coverage.get("available") or coverage["overall"] == "high":
        return []

    low = [c for c, v in coverage["criteria"].items() if v["confidence"] == "low"]
    msgs = []
    if low:
        names = ", ".join(c.replace("_", " ") for c in low)
        msgs.append(
            f"This walk sits outside the range the model was trained on for: "
            f"{names}. Those scores are extrapolated - treat them as indicative "
            f"rather than precise."
        )
    if coverage.get("n_features_outside", 0) >= 6:
        msgs.append(
            f"{coverage['n_features_outside']} of the measured quantities fall "
            "outside the training distribution, which usually means the "
            "recording setup differs from what the model has seen."
        )
    return msgs
