"""
Enhanced Dataset Creation + Multi-Target Runway Score Prediction
================================================================
Steps
-----
1. Load raw MoCap keypoints + processed synthetic features/labels
2. Engineer rich biomechanical features from raw keypoints
3. Merge real MoCap features with synthetic corpus -> enhanced dataset
4. Train an ensemble (RandomForest + XGBoost + Ridge stacking)
   for all 12 runway score dimensions
5. Evaluate with K-fold CV and report MAE / R² per criterion
6. Save the enhanced dataset and a predictions report
"""

from __future__ import annotations
import os
import sys
import warnings
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import skew, kurtosis
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
)
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────── paths ────────────────────────────
ROOT = Path(__file__).parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MOCAP_FILE = ROOT / "Optimal_Runway_Keypoints_from_MoCap.xlsx"
FEATURES_FILE = PROC_DIR / "features.csv"
LABELS_FILE = PROC_DIR / "labels.csv"
ENHANCED_FILE = ROOT / "data" / "enhanced_dataset.csv"

SCORE_COLS = [
    "score_posture", "score_balance", "score_confidence",
    "score_shoulder_alignment", "score_head_position", "score_arm_swing",
    "score_hip_movement", "score_rhythm", "score_stride_length",
    "score_foot_placement", "score_symmetry", "score_stage_presence",
]

# ══════════════════════════════════════════════════════════════
# 1.  Biomechanical feature engineering from raw MoCap keypoints
# ══════════════════════════════════════════════════════════════

def smooth(x: np.ndarray, w: int = 7) -> np.ndarray:
    if len(x) < w:
        return x
    from scipy.signal import savgol_filter
    return savgol_filter(x, window_length=w, polyorder=2)


def extract_mocap_features(kp: pd.DataFrame, clip_info: pd.DataFrame) -> pd.DataFrame:
    """
    For every video in the MoCap dataset extract a rich feature vector
    from the raw (frame, landmark, x, y, z) keypoints.
    Returns one row per video.
    """
    records = []

    for vid_id, grp in kp.groupby("video_id"):
        frames_sorted = sorted(grp["frame"].unique())
        n_frames = len(frames_sorted)

        def ts(landmark: str, axis: str) -> np.ndarray:
            sub = grp[grp["landmark"] == landmark].set_index("frame")[axis]
            return (sub.reindex(frames_sorted)
                       .interpolate()
                       .ffill()
                       .bfill()
                       .values)

        def mid_ts(lm_a: str, lm_b: str, axis: str) -> np.ndarray:
            return (ts(lm_a, axis) + ts(lm_b, axis)) / 2.0

        mid_sh_x = mid_ts("left_shoulder", "right_shoulder", "x")
        mid_sh_y = mid_ts("left_shoulder", "right_shoulder", "y")
        mid_hip_x = mid_ts("left_hip", "right_hip", "x")
        mid_hip_y = mid_ts("left_hip", "right_hip", "y")

        nose_x, nose_y = ts("nose", "x"), ts("nose", "y")

        l_ankle_y, r_ankle_y = ts("left_ankle", "y"), ts("right_ankle", "y")
        l_ankle_x, r_ankle_x = ts("left_ankle", "x"), ts("right_ankle", "x")

        l_wrist_x, r_wrist_x = ts("left_wrist", "x"), ts("right_wrist", "x")
        l_wrist_y, r_wrist_y = ts("left_wrist", "y"), ts("right_wrist", "y")

        l_knee_y, r_knee_y = ts("left_knee", "y"), ts("right_knee", "y")
        l_hip_x, r_hip_x   = ts("left_hip",  "x"), ts("right_hip",  "x")
        l_hip_y, r_hip_y   = ts("left_hip",  "y"), ts("right_hip",  "y")

        l_shoulder_y = ts("left_shoulder", "y")
        r_shoulder_y = ts("right_shoulder", "y")

        # ── POSTURE ──
        dx = mid_sh_x - mid_hip_x
        dy = mid_sh_y - mid_hip_y
        trunk_lean = np.degrees(np.arctan2(dx, np.abs(dy) + 1e-9))
        trunk_lean_s = smooth(trunk_lean)
        neck_angle = np.degrees(np.arctan2(nose_x - mid_sh_x, np.abs(nose_y - mid_sh_y) + 1e-9))
        body_len = np.sqrt((mid_sh_x - mid_hip_x)**2 + (mid_sh_y - mid_hip_y)**2)

        # ── BALANCE ──
        cop_x = (mid_sh_x + mid_hip_x) / 2
        cop_y = (mid_sh_y + mid_hip_y) / 2
        sway_amp = np.ptp(cop_x)
        sway_rms = np.std(cop_x)
        bob_amp   = np.ptp(cop_y)
        sway_jerk = np.mean(np.abs(np.diff(np.diff(cop_x))))
        path_ratio = np.sum(np.abs(np.diff(cop_x))) / (np.ptp(cop_x) + 1e-9)
        upper_body_sway = np.std(mid_sh_x)

        # ── SHOULDER ──
        sh_tilt = l_shoulder_y - r_shoulder_y
        sh_width = np.abs(ts("left_shoulder", "x") - ts("right_shoulder", "x"))

        # ── HEAD ──
        head_lat_offset = np.mean(nose_x - mid_sh_x)
        head_lat_amp    = np.ptp(nose_x - mid_sh_x)
        head_stability  = np.mean(np.abs(np.diff(np.diff(nose_x))))

        # ── ARM SWING ──
        arm_l_reach = np.ptp(l_wrist_x)
        arm_r_reach = np.ptp(r_wrist_x)
        arm_sym     = 1.0 - abs(arm_l_reach - arm_r_reach) / (arm_l_reach + arm_r_reach + 1e-9)
        arm_height_l = np.mean(mid_sh_y - l_wrist_y)
        arm_height_r = np.mean(mid_sh_y - r_wrist_y)
        wrist_amp    = (arm_l_reach + arm_r_reach) / 2

        # ── HIP ──
        hip_sway    = np.ptp(mid_hip_x)
        hip_osc     = np.std(mid_hip_x)
        hip_tilt_amp = np.ptp(l_hip_y - r_hip_y)
        hip_forward  = np.mean(mid_hip_y)

        # ── RHYTHM ──
        ankle_diff   = l_ankle_y - r_ankle_y
        step_signal  = smooth(ankle_diff)
        zc = np.where(np.diff(np.sign(step_signal)))[0]
        cadence_spm  = len(zc) * 30.0 / n_frames if n_frames > 0 else 0
        step_intervals = np.diff(zc) if len(zc) > 1 else np.array([0])
        rhythm_regularity = min(1.0 / (np.std(step_intervals) + 1e-9), 100.0)

        # ── STRIDE / FOOT ──
        l_ankle_amp = np.ptp(l_ankle_x)
        r_ankle_amp = np.ptp(r_ankle_x)
        stride_len  = (l_ankle_amp + r_ankle_amp) / 2
        foot_sym    = 1.0 - abs(l_ankle_amp - r_ankle_amp) / (l_ankle_amp + r_ankle_amp + 1e-9)
        foot_placement_x = np.std(l_ankle_x - r_ankle_x)

        # ── SYMMETRY ──
        sym_hips     = 1.0 - abs(np.std(l_hip_y) - np.std(r_hip_y)) / (np.std(l_hip_y) + np.std(r_hip_y) + 1e-9)
        sym_shoulders= 1.0 - abs(np.std(l_shoulder_y) - np.std(r_shoulder_y)) / (np.std(l_shoulder_y) + np.std(r_shoulder_y) + 1e-9)
        sym_knees    = 1.0 - abs(np.std(l_knee_y) - np.std(r_knee_y)) / (np.std(l_knee_y) + np.std(r_knee_y) + 1e-9)

        # ── CONFIDENCE ──
        velocity_x = np.abs(np.diff(mid_sh_x))
        velocity_y = np.abs(np.diff(mid_sh_y))
        movement_energy = np.mean(velocity_x**2 + velocity_y**2)
        smoothness      = min(1.0 / (np.mean(np.abs(np.diff(velocity_x))) + 1e-9), 1000.0)
        vel_full        = np.sqrt(np.diff(mid_sh_x)**2 + np.diff(mid_sh_y)**2)
        vel_skew        = float(skew(vel_full)) if len(vel_full) > 3 else 0.0
        vel_kurt        = float(kurtosis(vel_full)) if len(vel_full) > 3 else 0.0

        # ── FREQUENCY ──
        def dominant_freq(x: np.ndarray, fps: float = 50.0) -> float:
            if len(x) < 4:
                return 0.0
            fft_mag = np.abs(np.fft.rfft(x - x.mean()))
            freqs   = np.fft.rfftfreq(len(x), d=1.0/fps)
            idx     = np.argmax(fft_mag[1:]) + 1
            return float(freqs[idx])

        ankle_dom_freq = dominant_freq(ankle_diff)
        hip_dom_freq   = dominant_freq(mid_hip_x)
        wrist_dom_freq = dominant_freq(l_wrist_x - r_wrist_x)

        records.append({
            "video_id":           vid_id,
            "n_frames":           n_frames,
            "trunk_lean_mean":    float(np.mean(np.abs(trunk_lean_s))),
            "trunk_lean_signed":  float(np.mean(trunk_lean_s)),
            "trunk_lean_std":     float(np.std(trunk_lean_s)),
            "neck_angle_mean":    float(np.mean(np.abs(neck_angle))),
            "body_length_mean":   float(np.mean(body_len)),
            "body_length_cv":     float(np.std(body_len) / (np.mean(body_len) + 1e-9)),
            "sway_amp":           sway_amp,
            "sway_rms":           sway_rms,
            "bob_amp":            bob_amp,
            "sway_jerk":          sway_jerk,
            "path_ratio":         path_ratio,
            "upper_body_sway":    upper_body_sway,
            "sh_tilt_bias":       float(np.mean(sh_tilt)),
            "sh_tilt_abs_mean":   float(np.mean(np.abs(sh_tilt))),
            "sh_tilt_amp":        float(np.ptp(sh_tilt)),
            "sh_tilt_std":        float(np.std(sh_tilt)),
            "sh_width_mean":      float(np.mean(sh_width)),
            "sh_width_cv":        float(np.std(sh_width) / (np.mean(sh_width) + 1e-9)),
            "head_lat_offset":    head_lat_offset,
            "head_lat_amp":       head_lat_amp,
            "head_stability":     head_stability,
            "arm_l_reach":        arm_l_reach,
            "arm_r_reach":        arm_r_reach,
            "arm_sym":            arm_sym,
            "arm_height_l":       arm_height_l,
            "arm_height_r":       arm_height_r,
            "wrist_amp":          wrist_amp,
            "hip_sway":           hip_sway,
            "hip_osc":            hip_osc,
            "hip_tilt_amp":       hip_tilt_amp,
            "hip_forward":        hip_forward,
            "cadence_spm":        cadence_spm,
            "rhythm_regularity":  rhythm_regularity,
            "step_interval_std":  float(np.std(step_intervals)),
            "stride_len":         stride_len,
            "foot_sym":           foot_sym,
            "foot_placement_x":   foot_placement_x,
            "sym_hips":           sym_hips,
            "sym_shoulders":      sym_shoulders,
            "sym_knees":          sym_knees,
            "movement_energy":    movement_energy,
            "smoothness":         smoothness,
            "vel_skew":           vel_skew,
            "vel_kurt":           vel_kurt,
            "ankle_dom_freq":     ankle_dom_freq,
            "hip_dom_freq":       hip_dom_freq,
            "wrist_dom_freq":     wrist_dom_freq,
            "source":             "mocap",
        })

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════
# 2.  Remap synthetic column names to common schema
# ══════════════════════════════════════════════════════════════
SYN_TO_COMMON = {
    "posture_trunk_lean_mean":    "trunk_lean_mean",
    "posture_trunk_lean_signed":  "trunk_lean_signed",
    "posture_trunk_lean_std":     "trunk_lean_std",
    "posture_neck_angle_mean":    "neck_angle_mean",
    "posture_body_length_mean":   "body_length_mean",
    "posture_body_length_cv":     "body_length_cv",
    "balance_sway_amp":           "sway_amp",
    "balance_sway_rms":           "sway_rms",
    "balance_bob_amp":            "bob_amp",
    "balance_sway_jerk":          "sway_jerk",
    "balance_path_ratio":         "path_ratio",
    "balance_upper_body_sway":    "upper_body_sway",
    "shoulder_tilt_bias":         "sh_tilt_bias",
    "shoulder_tilt_abs_mean":     "sh_tilt_abs_mean",
    "shoulder_tilt_amp":          "sh_tilt_amp",
    "shoulder_tilt_std":          "sh_tilt_std",
    "shoulder_width_mean":        "sh_width_mean",
    "shoulder_width_cv":          "sh_width_cv",
    "head_lateral_offset":        "head_lat_offset",
    "head_lateral_amp":           "head_lat_amp",
    "head_stability_jerk":        "head_stability",
    "arm_left_reach_amp":         "arm_l_reach",
    "arm_right_reach_amp":        "arm_r_reach",
    "arm_symmetry":               "arm_sym",
    "arm_left_height":            "arm_height_l",
    "arm_right_height":           "arm_height_r",
    "arm_wrist_osc_amp":          "wrist_amp",
    "hip_sway_amp":               "hip_sway",
    "hip_sway_rms":               "hip_osc",
    "hip_tilt_amp":               "hip_tilt_amp",
    "hip_forward_lean":           "hip_forward",
    "rhythm_cadence_spm":         "cadence_spm",
    "rhythm_regularity":          "rhythm_regularity",
    "rhythm_step_time_std":       "step_interval_std",
    "stride_length_mean":         "stride_len",
    "stride_symmetry":            "foot_sym",
    "foot_placement_std":         "foot_placement_x",
    "symmetry_hip":               "sym_hips",
    "symmetry_shoulder":          "sym_shoulders",
    "symmetry_knee":              "sym_knees",
    "confidence_movement_energy": "movement_energy",
    "confidence_smoothness":      "smoothness",
    "confidence_vel_skew":        "vel_skew",
    "confidence_vel_kurt":        "vel_kurt",
    "rhythm_ankle_dom_freq":      "ankle_dom_freq",
    "rhythm_hip_dom_freq":        "hip_dom_freq",
    "rhythm_wrist_dom_freq":      "wrist_dom_freq",
}


def remap_synthetic(feat: pd.DataFrame, lbl: pd.DataFrame) -> pd.DataFrame:
    merged = feat.merge(lbl[["video_id", "n_frames"] + SCORE_COLS], on="video_id", how="inner")
    merged = merged.rename(columns=SYN_TO_COMMON)
    merged["source"] = "synthetic"
    return merged


# ══════════════════════════════════════════════════════════════
# 3.  Score estimation for MoCap clips (biomechanical heuristics)
# ══════════════════════════════════════════════════════════════
def estimate_mocap_scores(df: pd.DataFrame) -> pd.DataFrame:
    def clip10(x):
        return float(np.clip(x, 0, 10))

    rows = []
    for _, r in df.iterrows():
        score_posture   = clip10(10 - r["trunk_lean_mean"] * 0.4
                                    - r["neck_angle_mean"] * 0.3
                                    - r["body_length_cv"] * 20)
        score_balance   = clip10(10 - r["sway_amp"] * 25
                                    - r["bob_amp"] * 15
                                    - r["sway_jerk"] * 50)
        score_confidence = clip10(5
                                  + np.log1p(r["movement_energy"]) * 1.2
                                  + r["smoothness"] * 0.005
                                  - abs(r["vel_skew"]) * 0.5)
        score_shoulder  = clip10(10 - r["sh_tilt_abs_mean"] * 40
                                     - r["sh_tilt_amp"] * 20)
        score_head      = clip10(10 - abs(r["head_lat_offset"]) * 30
                                     - r["head_stability"] * 150)
        score_arm       = clip10(r["arm_sym"] * 7
                                 + r["wrist_amp"] * 10
                                 - abs(r["arm_height_l"] - r["arm_height_r"]) * 10)
        score_hip       = clip10(r["hip_sway"] * 15
                                 + r["hip_osc"] * 10
                                 + r["hip_tilt_amp"] * 8)
        score_rhythm    = clip10(5 + np.log1p(r["rhythm_regularity"]) * 0.8
                                    - r["step_interval_std"] * 3)
        score_stride    = clip10(r["stride_len"] * 30 + r["foot_sym"] * 4)
        score_foot      = clip10(10 - r["foot_placement_x"] * 60 + r["foot_sym"] * 2)
        score_sym       = clip10((r["sym_hips"] + r["sym_shoulders"] + r["sym_knees"]) / 3 * 10)
        score_stage     = clip10(score_confidence * 0.35 + score_sym * 0.35 + score_balance * 0.30)

        rows.append({
            **r.to_dict(),
            "score_posture":            score_posture,
            "score_balance":            score_balance,
            "score_confidence":         score_confidence,
            "score_shoulder_alignment": score_shoulder,
            "score_head_position":      score_head,
            "score_arm_swing":          score_arm,
            "score_hip_movement":       score_hip,
            "score_rhythm":             score_rhythm,
            "score_stride_length":      score_stride,
            "score_foot_placement":     score_foot,
            "score_symmetry":           score_sym,
            "score_stage_presence":     score_stage,
        })
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════
# 4.  Augmentation (Gaussian jitter on numeric features)
# ══════════════════════════════════════════════════════════════
def augment_mocap(df: pd.DataFrame, n_copies: int = 8,
                  noise_std: float = 0.04, rng_seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    score_cols_present = [c for c in SCORE_COLS if c in df.columns]
    augmented = [df.copy()]
    for i in range(n_copies):
        aug = df.copy()
        noise = rng.normal(0, noise_std, (len(aug), len(num_cols)))
        aug[num_cols] = aug[num_cols].values + noise
        for sc in score_cols_present:
            aug[sc] = aug[sc].clip(0, 10)
        aug["video_id"] = aug["video_id"].astype(str) + f"_aug{i}"
        aug["source"]   = "mocap_aug"
        augmented.append(aug)
    return pd.concat(augmented, ignore_index=True)


# ══════════════════════════════════════════════════════════════
# 5.  Common feature columns
# ══════════════════════════════════════════════════════════════
COMMON_FEATURE_COLS = [
    "n_frames",
    "trunk_lean_mean", "trunk_lean_signed", "trunk_lean_std",
    "neck_angle_mean", "body_length_mean", "body_length_cv",
    "sway_amp", "sway_rms", "bob_amp", "sway_jerk", "path_ratio", "upper_body_sway",
    "sh_tilt_bias", "sh_tilt_abs_mean", "sh_tilt_amp", "sh_tilt_std",
    "sh_width_mean", "sh_width_cv",
    "head_lat_offset", "head_lat_amp", "head_stability",
    "arm_l_reach", "arm_r_reach", "arm_sym",
    "arm_height_l", "arm_height_r", "wrist_amp",
    "hip_sway", "hip_osc", "hip_tilt_amp", "hip_forward",
    "cadence_spm", "rhythm_regularity", "step_interval_std",
    "stride_len", "foot_sym", "foot_placement_x",
    "sym_hips", "sym_shoulders", "sym_knees",
    "movement_energy", "smoothness", "vel_skew", "vel_kurt",
    "ankle_dom_freq", "hip_dom_freq", "wrist_dom_freq",
]


# ══════════════════════════════════════════════════════════════
# 6.  Stacked ensemble model
# ══════════════════════════════════════════════════════════════
def build_model():
    base_estimators = [
        ("rf",  RandomForestRegressor(n_estimators=300, max_depth=10,
                                       min_samples_leaf=4, n_jobs=-1, random_state=42)),
        ("xgb", xgb.XGBRegressor(n_estimators=300, learning_rate=0.05,
                                   max_depth=5, subsample=0.8, colsample_bytree=0.8,
                                   reg_alpha=0.1, reg_lambda=1.0, tree_method="hist",
                                   random_state=42, verbosity=0)),
        ("gbm", GradientBoostingRegressor(n_estimators=200, learning_rate=0.08,
                                           max_depth=4, subsample=0.8, random_state=42)),
    ]
    stack = StackingRegressor(estimators=base_estimators,
                               final_estimator=Ridge(alpha=1.0),
                               cv=5, passthrough=True, n_jobs=-1)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  RobustScaler()),
        ("model",   stack),
    ])


# ══════════════════════════════════════════════════════════════
# 7.  Main
# ══════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  Enhanced Dataset Creation & Multi-Target Prediction")
    print("=" * 60)

    # Load MoCap
    print("\n[1/6] Loading raw MoCap keypoints …")
    kp        = pd.read_excel(MOCAP_FILE, sheet_name="keypoints")
    clip_info = pd.read_excel(MOCAP_FILE, sheet_name="clip_info")
    print(f"      Keypoints rows: {len(kp):,}  |  Clips: {clip_info.shape[0]}")

    # Engineer features
    print("[2/6] Engineering biomechanical features from MoCap …")
    mocap_feat = extract_mocap_features(kp, clip_info)
    mocap_feat = estimate_mocap_scores(mocap_feat)
    mocap_feat = augment_mocap(mocap_feat, n_copies=8)
    print(f"      MoCap rows after augmentation: {len(mocap_feat)}")

    # Load synthetic
    print("[3/6] Loading synthetic features & labels …")
    syn_feat = pd.read_csv(FEATURES_FILE)
    syn_lbl  = pd.read_csv(LABELS_FILE)
    syn_df   = remap_synthetic(syn_feat, syn_lbl)
    print(f"      Synthetic rows: {len(syn_df)}")

    # Merge
    print("[4/6] Merging into enhanced dataset …")
    available_feat = [c for c in COMMON_FEATURE_COLS
                      if c in mocap_feat.columns and c in syn_df.columns]
    target_cols    = [c for c in SCORE_COLS
                      if c in mocap_feat.columns and c in syn_df.columns]
    meta_cols      = ["video_id", "source"]
    keep_cols      = meta_cols + available_feat + target_cols

    mocap_sub = mocap_feat[[c for c in keep_cols if c in mocap_feat.columns]]
    syn_sub   = syn_df[[c for c in keep_cols if c in syn_df.columns]]
    enhanced  = pd.concat([syn_sub, mocap_sub], ignore_index=True)
    enhanced.to_csv(ENHANCED_FILE, index=False)
    print(f"      Enhanced dataset: {enhanced.shape[0]:,} rows x {enhanced.shape[1]} cols")
    print(f"      Saved → {ENHANCED_FILE}")

    # Train & evaluate
    print("\n[5/6] Training stacked ensemble (RF + XGB + GBM → Ridge) …")
    X = enhanced[available_feat].copy()
    results, models = {}, {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for sc in target_cols:
        y = enhanced[sc].copy()
        valid = y.notna() & np.isfinite(y)
        X_v, y_v = X[valid], y[valid]
        model = build_model()
        maes, r2s = [], []
        for tr, va in kf.split(X_v):
            model.fit(X_v.iloc[tr], y_v.iloc[tr])
            p = model.predict(X_v.iloc[va])
            maes.append(mean_absolute_error(y_v.iloc[va], p))
            r2s.append(r2_score(y_v.iloc[va], p))
        model.fit(X_v, y_v)
        models[sc] = model
        results[sc] = {
            "mae_mean": float(np.mean(maes)), "mae_std": float(np.std(maes)),
            "r2_mean":  float(np.mean(r2s)),  "r2_std":  float(np.std(r2s)),
        }
        print(f"      {sc:<30}  MAE={np.mean(maes):.3f}±{np.std(maes):.3f}"
              f"  R²={np.mean(r2s):.3f}±{np.std(r2s):.3f}")

    # Save models
    for name, mdl in models.items():
        joblib.dump(mdl, MODELS_DIR / f"enhanced_{name}.pkl")

    # Reports & plots
    print("\n[6/6] Generating reports & visualisations …")
    res_df = pd.DataFrame(results).T.reset_index().rename(columns={"index": "criterion"})
    res_df.to_csv(REPORTS_DIR / "enhanced_cv_results.csv", index=False)
    with open(REPORTS_DIR / "enhanced_cv_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot 1: MAE + R² bar charts
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    criteria_short = [c.replace("score_", "") for c in target_cols]
    maes_m = [results[c]["mae_mean"] for c in target_cols]
    r2s_m  = [results[c]["r2_mean"]  for c in target_cols]
    mae_e  = [results[c]["mae_std"]  for c in target_cols]
    r2_e   = [results[c]["r2_std"]   for c in target_cols]
    pal    = sns.color_palette("viridis", len(criteria_short))

    axes[0].barh(criteria_short, maes_m, xerr=mae_e, color=pal, edgecolor="white", capsize=3)
    axes[0].set_xlabel("MAE (± 1 SD)"); axes[0].set_title("CV MAE per Criterion — Enhanced Dataset", fontweight="bold")
    axes[0].axvline(np.mean(maes_m), ls="--", color="red", lw=1.5, label=f"Mean={np.mean(maes_m):.3f}")
    axes[0].legend(); axes[0].invert_yaxis()

    axes[1].barh(criteria_short, r2s_m, xerr=r2_e, color=pal, edgecolor="white", capsize=3)
    axes[1].set_xlabel("R² (± 1 SD)"); axes[1].set_title("CV R² per Criterion — Enhanced Dataset", fontweight="bold")
    axes[1].axvline(np.mean(r2s_m), ls="--", color="red", lw=1.5, label=f"Mean={np.mean(r2s_m):.3f}")
    axes[1].axvline(0, ls="-", color="gray", lw=0.8); axes[1].legend(); axes[1].invert_yaxis()
    plt.tight_layout(); fig.savefig(REPORTS_DIR / "enhanced_cv_performance.png", dpi=150); plt.close(fig)

    # Plot 2: Score distributions
    fig, axes = plt.subplots(3, 4, figsize=(18, 12))
    axes = axes.flatten()
    for i, sc in enumerate(target_cols):
        ax = axes[i]
        syn_s = enhanced[enhanced["source"] == "synthetic"][sc].dropna()
        moc_s = enhanced[enhanced["source"].str.startswith("mocap")][sc].dropna()
        ax.hist(syn_s, bins=30, alpha=0.6, color="#4ECDC4", label="Synthetic", density=True)
        ax.hist(moc_s, bins=15, alpha=0.6, color="#FF6B6B", label="MoCap (aug)", density=True)
        ax.set_title(sc.replace("score_", "").replace("_", " ").title(), fontsize=10)
        ax.set_xlabel("Score [0–10]", fontsize=8); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.suptitle("Score Distributions: Synthetic vs MoCap (Augmented)", fontsize=14, fontweight="bold")
    plt.tight_layout(); fig.savefig(REPORTS_DIR / "enhanced_score_distributions.png", dpi=150); plt.close(fig)

    # Plot 3: Feature importance
    try:
        rf_mdl = models["score_posture"].named_steps["model"].estimators_[0][1]
        imp = pd.Series(rf_mdl.feature_importances_, index=available_feat).sort_values(ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x=imp.values, y=imp.index, palette="magma", ax=ax)
        ax.set_title("Top-20 Feature Importances (RF, posture)\nEnhanced Dataset", fontweight="bold")
        ax.set_xlabel("Importance"); plt.tight_layout()
        fig.savefig(REPORTS_DIR / "enhanced_feature_importance.png", dpi=150); plt.close(fig)
    except Exception as e:
        print(f"      [warn] feature importance plot skipped: {e}")

    # Summary
    overall_mae = np.mean(maes_m)
    overall_r2  = np.mean(r2s_m)
    best  = target_cols[int(np.argmin(maes_m))]
    worst = target_cols[int(np.argmax(maes_m))]
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Enhanced dataset rows : {len(enhanced):,}")
    print(f"  Feature columns used  : {len(available_feat)}")
    print(f"  Score dimensions      : {len(target_cols)}")
    print(f"  Overall MAE (avg)     : {overall_mae:.3f} / 10")
    print(f"  Overall R²  (avg)     : {overall_r2:.3f}")
    print(f"  Best criterion        : {best}  MAE={results[best]['mae_mean']:.3f}")
    print(f"  Worst criterion       : {worst}  MAE={results[worst]['mae_mean']:.3f}")
    print(f"\n  Saved files:")
    print(f"    data/enhanced_dataset.csv")
    print(f"    reports/enhanced_cv_results.csv + .json")
    print(f"    reports/enhanced_cv_performance.png")
    print(f"    reports/enhanced_score_distributions.png")
    print(f"    reports/enhanced_feature_importance.png")
    print(f"    models/enhanced_score_*.pkl  (12 models)")
    print("=" * 60)


if __name__ == "__main__":
    main()
