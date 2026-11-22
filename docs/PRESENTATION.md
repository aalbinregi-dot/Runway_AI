# Final-Year Presentation

A 15-slide deck with the talking point for each. The through-line is
deliberately not "we built a system" but **"we made the scores mean something,
and we can prove it"** — that is what distinguishes this from a pipeline demo.

---

### Slide 1 — Title
**AI-Powered Runway Performance Analysis and Virtual Coaching System**
Computer vision + gait biomechanics + machine learning.
One line: *upload a walk, get a professional-grade assessment and a coaching plan.*

### Slide 2 — The problem
Runway coaching is expensive, subjective and unavailable to most aspiring
models. Feedback is verbal, inconsistent between coaches, and impossible to
track over time. There is no objective record of whether you improved.

### Slide 3 — What the system does
Demo screenshot: overall 0–100, twelve criterion scores, three ranked focus
areas with the measurement behind each, prescribed drills, progress chart.
Emphasise: *it says what to fix, why, and by how much.*

### Slide 4 — The starting data, honestly
Show the workbook: correct schema, 5 clips × 50 frames × 13 landmarks.
Then the killer chart: **per-landmark temporal std = 0.0143 for all thirteen.**
Every landmark moves identically — the body translates rigidly. No arm swing,
no gait, no labels.
**Decision: keep the schema, build the motion.** This slide sets up everything
that follows and shows you interrogated your data rather than accepting it.

### Slide 5 — Generating data that can actually be learned from
The circularity trap: compute a score from a rule, train a model to
rediscover the rule, prove nothing.
The fix — go through physics:

`latent qualities q → 3D gait model + IK → pinhole camera → sensor noise → keypoints`
`labels = 10 × q` (from the *hidden* latents)

Features see only the corrupted 2D projection. Recovering the score is a real
estimation problem.

### Slide 6 — The gait model
Drillis & Contini segment ratios. Prescribed foot trajectories with IK for the
knee. Three details that each fixed a measurable bug:
* ankle stance excursion is 0.40 stride, not 0.60 (the foot rolls heel→toe)
* pelvis height is *derived* from leg reach (compass gait), not prescribed
* the arm is a rigid pendulum from composed rotations

Show the before/after: thigh stretching 13% → under 0.3%.

### Slide 7 — From pixels to measurements
Normalisation: pelvis → origin, torso length → unit. A walk scores the same at
3 m or 8 m.
The signal-processing insight: the pelvis traverses **7 torso lengths** across
the frame as the subject approaches, while the sway being measured is **0.2**.
A polynomial detrend leaves residuals bigger than the signal. Zero-phase
Butterworth high-pass at 0.45 Hz fixes it.

### Slide 8 — Measuring gait timing properly
Peak-picking on the ankle trace found **27 steps in a 6-second clip** (truth:
~11) and gave a stride CV of 0.7 regardless of the actual rhythm.
Replaced with **Hilbert instantaneous phase** → one event per cycle by
construction, sub-frame precision. Step count corrected to 9.8–11.4, and CV
became monotone in the true rhythm quality.
*Good slide to show a plot: measured CV vs true rhythm, before and after.*

### Slide 9 — Validating the features before training anything
`scripts/sensitivity.py`: sweep one latent, see which features move.
Two methodological points worth making:
* ranked by **mutual information**, not correlation — over- and under-striding
  are both faults, so the relationship is U-shaped and Pearson r is ~0
* an **empirical noise floor** from permuted targets. At n=54 every criterion
  showed a "best feature" at |r| ≈ 0.35 — pure multiple-comparison artefact.

This caught that `stage_presence` had a label with **no signal in the pixels at
all** (MI 0.15 → 0.94 after fixing the generator).

### Slide 10 — The model
Twelve regressors, one per criterion. Each picks its own estimator
(ridge / gradient boosting / blend) by out-of-fold MAE.
Why not deep learning: ~100 engineered features, 1,500 clips, and the
interpretability the coaching layer needs. Cite Grinsztajn et al. 2022.
Everything reported is out-of-fold.

### Slide 11 — Results
| | MAE | R² |
|---|---:|---:|
| Predict the mean | 2.076 | 0.00 |
| Rule-based rubric | 1.877 | — |
| Ridge baseline | 1.039 | — |
| **Trained models** | **0.902** | **0.735** |

Beats the rubric on **12/12** criteria. Best: confidence R² 0.94, foot
placement 0.93, shoulder alignment 0.90.

### Slide 12 — The design validating itself
Permutation importance table. For nine of twelve criteria the top feature is
the one purpose-built for it.
The interesting one: **stride length's top feature is hip bob amplitude** — the
compass-gait result. A longer stride forces a deeper hip dip, and in a head-on
view that vertical signal survives where the sagittal stride does not.

### Slide 13 — What real video revealed *(the strongest slide in the deck)*
The first real phone clip landed **outside the training distribution on seven
features at once**, all in the same direction. Cross-validation could not have
caught it — every defect was a property of the training data itself:
* detector noise **13× too high** (0.0054 generated vs 0.00041 measured)
* every joint amplitude **2–6× too large**
* the skeleton too wide — Drillis gives body *surface* breadths, MediaPipe
  reports *joint centres*
* the camera behind the walker, so the subject **receded** instead of approaching

Fixing them moved mean MAE 0.754 → 0.902. **The metric got worse and that is
correct**: the earlier number described unrealistic data. Out-of-distribution
detection now runs at inference so this cannot hide again.

*This is the slide that shows engineering judgement rather than tuning.*

### Slide 13b — Remaining limits
Five criteria sit at R² 0.52–0.59 (stride, rhythm, arm swing, symmetry, hip).
**Geometry, not tuning**: a frontal camera foreshortens sagittal motion, and a
24° arm swing changes projected reach by only 8.6% — comparable to landmark
noise. And one subject is not a validation set.

### Slide 14 — Engineering
Architecture diagram. 37 tests, including regression guards for every bug
above, plus a React client over the same API. CLI + FastAPI + a React client over one shared entry point. SQLite session
tracking. Privacy: store keypoints, not video; cascade delete; no demographic
inference.

### Slide 15 — Future work
Ranked by expected benefit, not novelty:
1. Side / three-quarter camera — largest available accuracy gain
2. Real video + expert labels (report inter-rater agreement first)
3. 3D pose lifting — recovers the axis projection destroys
4. Real-time webcam feedback
5. Multi-camera fusion, IMU cross-validation, mobile deployment

---

## Live demo script (3 minutes)

```bash
python -m runway_ai info                                    # 12/12 models loaded
python -m runway_ai analyze Synthetic_Runway_Keypoints_Sample.xlsx
```
Point out the quality warning — the system *knows* the supplied clip is too
short and says so instead of pretending.

```bash
python scripts/build_dataset.py --n 40 --no-keypoints       # generation is fast
run.bat web                                                 # React app: upload, report, progress
./run.sh web                                                # the same, on macOS or Linux
```

## Likely questions

**"Isn't training on synthetic data circular?"**
Labels come from hidden latents; features come from a noisy 2D projection of a
3D physical model. Between them sit a camera at random pose, depth-scaled
noise, dropout and frame quantisation. And the independent rule-based scorer
only reaches MAE 1.88 against the same labels — the model's 0.90 is a real gap.

**"Why should I believe a feature measures what you say?"**
Sensitivity sweeps with an empirical null, plus permutation importance showing
each criterion's top feature is the one designed for it. Both are in the repo
and reproducible.

**"Why not a neural network?"**
At 1,500 clips with pre-extracted kinematic features there is nothing left for
a sequence model to learn that the features do not expose, and it would cost
interpretability. It becomes correct at tens of thousands of real clips — and
it drops in behind the same interface.

**"What happens on a bad recording?"**
Five explicit quality checks — duration, step count, mean visibility, gap
fraction, scale change — surface as warnings alongside the score rather than
being hidden.
