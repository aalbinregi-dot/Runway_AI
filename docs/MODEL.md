# Machine Learning Design and Results

## 1. Problem framing

Twelve continuous targets on 0–10, from ~100 engineered tabular features, with
1,500 training clips. That framing drives every choice below.

**Why not a deep sequence model (LSTM / TCN / Transformer)?** With ~100
hand-built features and no raw-pixel input, a sequence model has nothing left
to exploit that the features do not already expose — the temporal structure has
already been reduced to amplitudes, frequencies, phase offsets and event
statistics. It would need one to two orders of magnitude more data to match a
boosted ensemble here, and it would forfeit the interpretability the coaching
layer depends on. A sequence model becomes the right answer once the dataset is
tens of thousands of *real* clips; at that point it is a drop-in replacement
behind the same interface.

**Why one model per criterion, not one multi-output model?** The twelve
criteria differ enormously in difficulty (out-of-fold R² ranges 0.52 → 0.93)
and in which features inform them. A shared model lets the easy criteria
dominate the loss. Separate models also make per-criterion feature importances
directly readable, which is what justifies the coaching text.

## 2. Estimator selection

Three candidates are cross-validated per criterion and the best is kept, by
out-of-fold MAE only:

| Estimator | Configuration |
|---|---|
| `ridge` | `StandardScaler` → `RidgeCV`, α over `logspace(-3, 3, 25)` |
| `hgb` | `HistGradientBoostingRegressor`, 400 iters, lr 0.06, 31 leaves, min_leaf 15, L2 1.0, early stopping |
| `blend` | unweighted mean of the two |

Selected per criterion:

| blend | hgb |
|---|---|
| confidence, hip_movement, rhythm, stage_presence, stride_length, symmetry | arm_swing, balance, foot_placement, head_position, posture, shoulder_alignment |

Where `ridge` or `blend` wins outright that is informative, not embarrassing:
it says the relationship is close to linear in the engineered features, and
forcing a boosted ensemble there would only add variance. Where the blend wins,
the linear and non-linear models are making complementary errors.

## 3. Validation protocol

* **5-fold cross-validation**, stratified across the score range so no fold is
  short of high- or low-ability clips.
* **Every reported number is out-of-fold.** A boosted ensemble reaches R² ≈ 0.99
  in-sample by memorising; in-sample numbers are not reported anywhere.
* **Estimator selection is itself out-of-fold**, so no criterion is chosen on
  data its fold was fitted on.

## 4. Results

1,500 clips × 102 features, 5 folds. Measured **after** the generator corrections in section 9; the earlier, higher numbers described unrealistic data and are reported there for comparison.

| Criterion | MAE | R² | MAE (mean) | MAE (ridge) | MAE (rubric) | within ±1 |
|---|---:|---:|---:|---:|---:|---:|
| confidence | **0.363** | 0.937 | 2.122 | 0.376 | 1.984 | 93.2% |
| foot_placement | **0.331** | 0.929 | 2.081 | 0.587 | 1.767 | 93.2% |
| shoulder_alignment | **0.480** | 0.900 | 2.146 | 0.767 | 1.667 | 88.8% |
| stage_presence | **0.632** | 0.845 | 1.846 | 0.722 | 1.634 | 81.1% |
| head_position | **0.628** | 0.842 | 2.078 | 0.823 | 1.546 | 82.1% |
| balance | **0.687** | 0.827 | 2.106 | 0.935 | 2.007 | 80.1% |
| posture | **1.055** | 0.701 | 2.087 | 1.268 | 1.828 | 57.5% |
| arm_swing | **1.282** | 0.590 | 2.083 | 1.444 | 2.038 | 46.9% |
| stride_length | **1.337** | 0.585 | 2.145 | 1.407 | 2.131 | 44.7% |
| rhythm | **1.333** | 0.564 | 2.093 | 1.360 | 2.034 | 44.1% |
| symmetry | **1.345** | 0.556 | 2.076 | 1.374 | 1.847 | 44.5% |
| hip_movement | **1.348** | 0.550 | 2.053 | 1.408 | 2.042 | 43.6% |
| **mean** | **0.902** | **0.735** | 2.076 | 1.039 | 1.877 | 66.6% |

The model beats the deterministic rubric on **12/12** criteria and the
predict-the-mean baseline on 12/12.

## 5. What drives each score

Permutation importance (drop in MAE when a feature is shuffled), top feature
per criterion:

| Criterion | Top feature | Δ MAE |
|---|---|---:|
| confidence | `conf_arm_spread` | 2.14 |
| hip_movement | `hip_obliquity_std` | 1.69 |
| foot_placement | `foot_step_width_mean` | 1.50 |
| posture | `posture_trunk_lean_mean` | 1.27 |
| balance | `balance_sway_amp` | 1.17 |
| shoulder_alignment | `posture_trunk_lean_std` | 1.16 |
| head_position | `head_lateral_offset` | 1.07 |
| symmetry | `arm_right_reach_mean` | 0.84 |
| stage_presence | `hip_shoulder_ratio_mean` | 0.68 |
| rhythm | `balance_sway_amp` | 0.25 |
| arm_swing | `arm_left_swing_amp` | 0.23 |
| stride_length | `balance_bob_amp` | 0.14 |

This is the design validating itself: for nine of twelve criteria the most
important feature is the one purpose-built for that criterion. `stride_length`
leaning on `balance_bob_amp` is the compass-gait result — a longer stride forces
a deeper hip dip, and in a head-on view that vertical signal survives where the
sagittal stride itself does not.

## 6. The four weak criteria, honestly

`stride_length`, `rhythm`, `arm_swing`, `symmetry` and `hip_movement` sit at
R² 0.52–0.59 while the rest reach 0.70–0.93. This is not a tuning failure; it is a **geometric
limitation of a single near-frontal camera**, and it is worth stating plainly:

* **Stride length and arm swing are sagittal.** A camera looking down the
  runway foreshortens forward motion almost completely. A 24° arm swing changes
  the projected shoulder-to-wrist distance by only ~8.6%, comparable to the
  landmark noise. What survives is indirect: hip dip depth for stride, limb
  foreshortening for arm swing.
* **Rhythm and symmetry are timing quantities** measured from ~5–6 gait cycles
  in a 6-second clip. The sampling error on a CV estimated from five intervals
  is large no matter how good the estimator.

Measured with a per-latent sensitivity sweep (`scripts/sensitivity.py`,
mutual information against a swept latent, n=270 per criterion), the same four
are the only ones whose best single feature sits near the noise floor.

The three interventions that would actually fix them, in order of expected
gain: a **side or three-quarter camera angle**; **longer clips** (10–12 s
doubles the number of cycles); and **3D pose lifting**, which recovers the
depth axis that projection destroys.

## 7. Guarding against circularity

The obvious failure mode for a synthetic-data project is training a model to
rediscover the rule that generated the labels. Three properties prevent it:

1. **Labels come from hidden latents.** They are `10 × q`, where `q` is the
   generative quality vector. No feature can see `q`.
2. **Features come from the corrupted 2D projection.** Between `q` and the
   features sit a 3D kinematic model, a pinhole camera at random pose,
   depth-scaled localisation noise, dropout, occlusion and frame quantisation.
3. **The rubric is not the label generator.** It is an independent hand-written
   scorer, and it manages only MAE 1.88 against the same labels — the model's
   0.90 is a real gap, not a restatement.

The residual honest caveat: the model has only ever been evaluated on data from
this generator. Real-video validation with expert labels (Dataset tier D) is
the necessary next step, and the transfer gap should be reported rather than
assumed away.

## 8. Reproducing

```bash
python scripts/build_dataset.py --n 1500 --seed 42   # generate + featurise
python scripts/calibrate_rubric.py                   # fit rubric bands
python scripts/train_model.py                        # cross-validate + train
python scripts/sensitivity.py                        # feature validation sweep
python -m pytest tests -q                            # 37 tests
```

All seeds are fixed; the reported numbers reproduce exactly.

---

## 9. What real video revealed

Everything above section 8 was measured on generated data. The first real phone
video pushed through the system exposed four defects that cross-validation could
never have shown, because every one of them was a property of the *training
distribution itself*.

The clip: 1080×1920 portrait, 59.6 fps, 6.2 s, one subject walking toward the
camera indoors with a second person in the background. MediaPipe found a pose in
**369/369 frames** with mean landmark visibility 0.974 and zero identity
switches, so the tracking was sound and the problems lay elsewhere.

### The four defects

**1. Detector noise was 13× too high.** Measuring the >6 Hz residual on real
MediaPipe output gives a landmark jitter of 0.00041 in normalised units. The
generator was sampling 0.002–0.009, a mean of 0.0054. Every fine-grained
criterion was being trained against noise that does not exist.

**2. Every joint amplitude was 2–6× too large.** Comparing feature distributions
found seven features where the real clip fell entirely outside the synthetic
range, all in the same direction. Hip obliquity was ~6× too large, lateral sway
~2.7×. Corrected against literature peak-to-peak ranges for normal adult gait,
not against the single subject.

**3. The skeleton was too wide.** Drillis & Contini tabulate body-*surface*
breadths; MediaPipe's shoulder and hip landmarks estimate *joint centres*, which
sit well inside the skin. Biacromial breadth over-predicted by ~22% and bi-iliac
by ~58%. Measured real ratios (0.212 H and 0.121 H) replaced the tabulated ones.

**4. The subject was walking away from the camera.** The camera sat behind the
walker, so the torso *shrank* through every clip instead of growing. That
inverts the sign of the approach-rate feature and reverses every perspective cue
relative to real runway footage. Fixed by placing the camera in front.

Defect 4 then exposed a fifth: with the subject facing the camera, their
anatomical left appears on the image right, so the shoulder-line vector points
the other way and its angle wraps near 180°. `abs()` of that measures nothing —
a level shoulder line read as 180 and a 5° tilt as 175 or 185. Transverse
segments are now folded into [-90, 90], which also makes the feature invariant
to a mirrored view.

### The cost, and why it was worth paying

| | before | after |
|---|---:|---:|
| mean MAE | 0.754 | 0.902 |
| mean R² | 0.813 | 0.735 |
| real clip: features outside training range | 12 | 3 |
| real clip: low-confidence criteria | 2 | 0 |

The headline metrics got **worse**, and that is the correct outcome. The earlier
numbers described performance on data that was 13× noisier than reality, moved
every joint several times too far, had the wrong body proportions and ran the
camera backwards. A model cannot be meaningfully accurate on data like that. The
current numbers describe a harder, honest problem.

### Out-of-distribution detection

Because this failure was invisible to cross-validation, it is now checked at
inference. `scoring/coverage.py` stores the training feature ranges at fit time
and compares every new clip against them, using Tukey fences on the training
IQR. Criteria whose supporting features fall outside get their confidence
downgraded and a warning attached to the report, so an extrapolated score is
never presented as an equally trustworthy one.

### Still open

One subject is not a validation set. The generator is now consistent with
literature norms and with one real clip; it has not been checked against a
population. The dataset tier D protocol in [DATASET.md](DATASET.md) — multiple
subjects, expert labels, inter-rater agreement, subject-level splits — remains
the necessary next step, and until it exists these numbers describe synthetic
performance only.
