# Testing Strategy

`python -m pytest tests -q` → **199 tests**.
`npm test` in `web/` → **18 tests** (`node --test`, no dependencies), covering the
live scene classifier.

The suite is built around one observation: in a numerical pipeline, the
dangerous bugs do not raise exceptions. A feature that measures the wrong thing
still returns a plausible float, and a silently clamped inverse-kinematics
solve still produces a skeleton that looks fine when plotted. So most of these
tests assert *direction* and *invariants* against known ground truth rather than
merely checking that output exists.

## What each group protects

| Group | Tests | Protects against |
|---|---:|---|
| `TestSchema` | 5 | Drift from the supplied workbook's format; dropped landmarks |
| `TestGeneratorPhysics` | 8 | Stretching bones, over-extended legs, out-of-frame projection |
| `TestSignals` | 3 | Filters that leak drift or fabricate timing variability |
| `TestFeatureDirection` | 11 | Features that respond to the wrong quantity, or not at all |
| `TestScoring` | 7 | Score/finding contradictions, mis-scaled aggregation |
| `TestBreakdown` | 8 | An explanation that contradicts the score it explains |
| `TestMeasurements` | 4 | Absent measurements reported as zero; mislabelled features |
| `TestTimelineEvidence` | 6 | Evidence pointing at the wrong moment or the wrong tolerance |
| `scene.test.mjs` | 18 | Live coaching offered on something that is not a walk |
| `TestCoaching` | 2 | Duplicate focus areas, manufactured faults on good walks |
| `TestStore` | 1 | Persistence round-trip and progress computation |

## Tests that caught real bugs

Each of these was written because something was actually wrong.

**`test_limb_segments_are_rigid`** — the thigh was stretching 13% because the
foot trajectory used the stance *fraction* (0.60) for the ankle's
pelvis-relative excursion. The foot is planted but the ankle joint is not: it
rolls heel to flat to toe. The correct value is about 0.40 stride, and the wrong
one demanded a hip-to-ankle reach longer than the leg, silently saturating the
IK.

**`test_arm_is_a_rigid_pendulum`** — the arm's 3D length varied with swing angle
because the wrist offset was built component-wise. The arm-swing feature
measures projected foreshortening, so it was reading that artefact rather than
the swing.

**`test_highpass_removes_arbitrary_drift_keeps_oscillation`** — a one-stride
moving average has poor stopband rejection; against the near-exponential
perspective drift it left a residual comparable to the sway being measured.
Replaced with a zero-phase Butterworth high-pass at 0.45 Hz.

**`test_cycle_times_recovers_known_period`** — Hilbert phase estimation is
distorted at signal boundaries, putting a spurious CV of 0.096 on a *perfectly*
periodic input. That alone masked the real rhythm variability. Fixed by
discarding a margin at each end.

**`test_no_dead_features`** — caught `posture_spine_len_mean` and
`posture_spine_len_cv`, which were identically 1.0 and 0.0 in every clip:
pelvis-to-shoulder distance is exactly what `normalise` divides by, so those
features were tautological.

**`test_findings_never_contradict_scores`** — the report once said
"Posture 10.0/10" and simultaneously listed posture as a focus area, because
findings came from the rubric while scores came from the model. Findings are now
gated on the model's score.

**`test_stage_presence_is_actually_rendered`** — `stage_presence` had a label
but no corresponding signal in the generated pixels, so no model could ever have
recovered it. A regression guard now holds the generator to rendering it.

**`test_in_tolerance_agrees_with_the_bands_own_score`** — the band panel drew
"50% of this band" beside "out of tolerance" on the same row. The flag was
derived from the unrounded score while the number shown was rounded to three
decimals, so a measurement 7e-5 outside its tolerance displayed as sitting
exactly on it. The flag now comes from the interval, and the two can only tie
where the payload's own precision cannot separate them.

**`test_reference_prefers_a_level_band_over_a_spread_one`** — the trace chart
shades the rubric's tolerance behind the line. Trunk lean has two bands, a mean
and a standard deviation, and shading a per-frame angle against a tolerance on
its *variability* invites a comparison that means nothing. The picker prefers
level-valued bands, and this holds it there.

**`test_does_not_manufacture_lateral_drift_from_perspective`** (JS) — the live
classifier divided the pelvis position by each frame's own torso length. Since
the torso grows as the walker approaches, that alone produced 0.09 torso/s of
apparent sideways travel on a walk straight down the centre line — enough to read
as "walking across the frame". The pelvis is now normalised by the window's
median torso, which is the same order of operations the feature builder documents
for the same quantity.

## Statistical testing

`TestFeatureDirection` averages each measurement over several nuisance draws
before comparing, because a single clip's camera azimuth can swamp a modest
effect.

The related diagnostic, `scripts/sensitivity.py`, sweeps one latent at a time
and ranks features by **mutual information** rather than correlation. Two faults
(over- versus under-striding, over- versus under-swinging) are symmetric about
the ideal, so their feature relationships are U-shaped and Pearson r sits near
zero even for a perfect predictor.

That script also reports an empirical noise floor from permuted targets. This
matters more than it sounds: an early run at n=54 showed a "best feature" at
|r| about 0.35 for *every* criterion, which looked like signal and was entirely
the multiple-comparison floor for around 90 features at that sample size. The
fix was more samples (n=270) and an explicit null, not a different feature set.

## Coverage gaps, stated plainly

* **No real-video test.** MediaPipe cannot be installed on this interpreter, so
  `pose/extract.py` is exercised only through its unavailable-backend path.
* **No browser test.** The React components are exercised by rendering them to
  static markup against a real analysis payload — which catches crashes and
  values leaking as `NaN` — but nothing drives a pointer, so the drag-to-zoom and
  hover readouts are verified by hand. The live classifier is tested directly
  instead, as a plain module, which is where its logic lives.
* **The live thresholds are judgement, not measurement.** `TH` in
  `web/src/live/scene.js` separates walking from not-walking well enough for the
  eleven staged scenarios, on a synthetic body. They have not been calibrated
  against a labelled corpus of real scenes, and nothing derived from them is
  offered as a score.
* **No transfer validation.** Every metric is measured on data from this
  project's own generator.
* **No inter-rater comparison**, because there are no human labels yet.

The first is an environment constraint. The second and third require the
real-video dataset tier described in [DATASET.md](DATASET.md), and until that
exists the reported accuracy should be read as "accurate on synthetic data of
this kind", not as a claim about real runway footage.
