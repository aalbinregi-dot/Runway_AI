"""Download the OpenCV face models used for opt-in face matching.

Both ship as ONNX from the OpenCV Zoo and are consumed by OpenCV's own
FaceDetectorYN / FaceRecognizerSF, so no extra Python package is needed.

Usage:  python scripts/fetch_face_models.py
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runway_ai.config import MODEL_DIR          # noqa: E402

BASE = "https://github.com/opencv/opencv_zoo/raw/main/models"
FILES = [
    (f"{BASE}/face_detection_yunet/face_detection_yunet_2023mar.onnx",
     "face_detection_yunet_2023mar.onnx", "YuNet face detector"),
    (f"{BASE}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
     "face_recognition_sface_2021dec.onnx", "SFace embedder"),
]


def main() -> int:
    out = MODEL_DIR / "face"
    out.mkdir(parents=True, exist_ok=True)
    for url, name, label in FILES:
        dest = out / name
        if dest.exists() and dest.stat().st_size > 100_000:
            print(f"  {label:22s} already present ({dest.stat().st_size / 1e6:.1f} MB)")
            continue
        print(f"  {label:22s} downloading...", flush=True)
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            print(f"    FAILED: {exc}\n    Fetch it manually to {dest}")
            return 1
        print(f"    -> {dest} ({dest.stat().st_size / 1e6:.1f} MB)")
    print("\nFace matching is ready. It stays off until a user enrols.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
