/**
 * Live scene reading: is what the camera is looking at a runway walk at all?
 *
 * Why this exists
 * ---------------
 * The live page used to compute three posture angles per frame and show them
 * whatever the camera was pointed at. Stand still and it read "holding" on all
 * three. Sit down, walk across the room, film a wall with a chair in it, walk
 * *away* from the lens — all of those produced the same confident cues, because
 * a single frame's trunk angle is perfectly measurable in every one of them.
 * That is the failure: the numbers were right and the coaching was nonsense,
 * because nothing checked the premise that a runway walk was happening.
 *
 * So this module answers the premise first. It keeps a rolling window of body
 * geometry, derives the same quantities the offline pipeline derives — stride
 * timing, cadence, lateral sway, approach rate, camera-facing angle — and
 * classifies the scene before any coaching is offered. If the scene is not a
 * walk toward the camera, the cues are withheld and the reason is named.
 *
 * How it relates to the offline pipeline
 * -------------------------------------
 * Every measurement here is deliberately in the *same units and conventions* as
 * `runway_ai.features`, so a live number and an analysed number mean the same
 * thing:
 *
 * - Lengths are in torso lengths, taken from raw normalised landmark
 *   coordinates exactly as `data.loaders.preprocess` does.
 * - The gait signal is the signed lateral gap between the ankles, which is what
 *   `features.timeline` and `rhythm_features` use — the cleanest periodic signal
 *   in a frontal view, and differential, so the perspective drift shared by both
 *   ankles cancels.
 * - The camera angle uses the shoulder-line foreshortening of
 *   `features.viewpoint`, including its aspect-ratio correction.
 *
 * Two things are honestly weaker than the offline versions and are treated as
 * such. The trend removal is a moving-mean rather than a zero-phase Butterworth,
 * because a rolling window cannot filter forward and backward through data it
 * has not seen; and cycle boundaries come from a Schmitt trigger on the gap
 * rather than from Hilbert phase, which costs sub-frame precision. Both are
 * fine for *deciding what is happening*; neither is offered as a score, and the
 * page says so.
 */

/* BlazePose's 33-landmark indices. */
export const L = {
  nose: 0,
  leftEye: 2, rightEye: 5,
  leftShoulder: 11, rightShoulder: 12,
  leftElbow: 13, rightElbow: 14,
  leftWrist: 15, rightWrist: 16,
  leftHip: 23, rightHip: 24,
  leftKnee: 25, rightKnee: 26,
  leftAnkle: 27, rightAnkle: 28,
  leftHeel: 29, rightHeel: 30,
  leftFoot: 31, rightFoot: 32,
};

/**
 * True shoulder-width to torso-length ratio, from the joint-centre anthropometry
 * the synthetic generator uses (`SEG.shoulder_width / TORSO_H` in
 * `runway_ai.data.synthetic`: 0.215 / 0.288). The same constant drives the
 * server's camera-angle estimate, so both agree on what "head-on" looks like.
 */
const SHOULDER_TORSO_RATIO = 0.215 / 0.288;

/**
 * Thresholds. These are scene *decisions*, not scores: each one only has to
 * separate "this is a walk toward the camera" from "this is something else", and
 * where a boundary is genuinely fuzzy the state machine's dwell time (below)
 * stops the readout flickering across it.
 *
 * Values in torso lengths and degrees unless noted. The cadence window is wide
 * on purpose — a deliberate runway walk sits near 100 spm, but a slow rehearsal
 * pass at 60 is still a walk and must not be reported as "not walking".
 */
export const TH = {
  /** Torso length as a fraction of frame height. Below this the body is too
   *  small for any measurement to survive landmark noise; above it the frame is
   *  cropping the walk. */
  torso: [0.10, 0.62],
  /** A landmark below this visibility is treated as missing, as MIN_VISIBILITY
   *  does server-side. */
  visibility: 0.5,
  /** Fraction of the window in which both ankles must be visible and inside the
   *  frame. The lower body leaving frame does not merely add noise: the hips get
   *  guessed, and trunk lean stops meaning anything. */
  legFrames: 0.6,
  /** Peak-to-peak of the detrended ankle gap below which nobody is stepping. */
  stillGap: 0.14,
  /** Pelvis travel below which the subject is on the spot. */
  stillTravel: 0.14,
  /** Autocorrelation peak of the gap signal. Periodic gait sits well above this;
   *  arm-waving, fidgeting and camera shake do not. */
  periodicity: 0.4,
  /** Steps per minute accepted as walking, and the edge above which it is a run. */
  cadence: [50, 150],
  /** Degrees off head-on. The rubric's frontal criteria — step width, shoulder
   *  level, symmetry — stop being measurable much beyond this. */
  facing: 45,
  /** Growth in apparent torso size per second. Below this the subject is
   *  receding from the lens. */
  approach: -0.03,
  /** Lateral pelvis drift, torso lengths per second, that reads as crossing the
   *  frame rather than coming down it. */
  crossing: 0.35,
  /** Fewest trigger events in the window before timing means anything. */
  steps: 2,
};

/** Rolling window length, and the shortest run of data worth reading. */
const WINDOW_S = 4.0;
const MIN_DATA_S = 1.4;

/** How long a candidate scene must hold before it is committed, in seconds.
 *  Without this the readout oscillates between "walking" and "standing still"
 *  at every mid-stance pause, which is unreadable and undermines trust in the
 *  parts that are right. Committing *to* a walk is slower than committing away
 *  from one: a false "not a ramp walk" is a minor annoyance, a false "walking"
 *  attaches coaching to something that is not a walk. */
const HOLD_S = { default: 0.5, walk: 1.0, searching: 0.4 };

/** Deadband of the step trigger, in torso lengths. The gap has to swing this
 *  far either side of centre to count as a step, which is what stops landmark
 *  jitter around zero from being counted as a cadence of 400 spm. */
const STEP_DEADBAND = 0.05;

/** Trend-removal window. One nominal stride (~1.15 s) so a mean over it sits
 *  near the cycle's centre and the gait band survives. */
const DETREND_S = 1.2;

/** Posture cues. Held behind a median filter and a hysteresis band, and shown
 *  only while the scene is a walk. Thresholds match the rubric bands used by the
 *  full analysis (`head_lateral_offset` ideal 0.10 / tol 0.16, and the trunk and
 *  shoulder bands alongside it). */
export const CUES = [
  { id: "lean", label: "Trunk lean", unit: "°", good: 6, warn: 11,
    hint: "Stack your ribcage over your hips" },
  { id: "tilt", label: "Shoulder level", unit: "°", good: 4, warn: 8,
    hint: "Level the shoulder line" },
  { id: "head", label: "Head centred", unit: "", good: 0.14, warn: 0.24,
    hint: "Bring your head back over your spine" },
];

const MEDIAN_WINDOW = 9;      // ~0.3 s at 30 fps: damps jitter, stays usable

/* ------------------------------------------------------------------------- *
 * Small numeric helpers. Kept local and dependency-free: this runs inside the
 * per-frame loop, so allocation and library overhead are visible as dropped
 * frames on a laptop already running a pose model.
 * ------------------------------------------------------------------------- */

function median(values) {
  if (!values.length) return 0;
  const s = [...values].sort((a, b) => a - b);
  const h = s.length >> 1;
  return s.length % 2 ? s[h] : 0.5 * (s[h - 1] + s[h]);
}

function percentile(sorted, p) {
  if (!sorted.length) return 0;
  const i = Math.min(sorted.length - 1, Math.max(0, Math.round((p / 100) * (sorted.length - 1))));
  return sorted[i];
}

function mean(values) {
  if (!values.length) return 0;
  let s = 0;
  for (const v of values) s += v;
  return s / values.length;
}

function stdev(values) {
  if (values.length < 2) return 0;
  const m = mean(values);
  let s = 0;
  for (const v of values) s += (v - m) ** 2;
  return Math.sqrt(s / (values.length - 1));
}

/** Least-squares slope of `ys` against `ts`, per unit of t. */
function slope(ts, ys) {
  const n = ts.length;
  if (n < 3) return 0;
  const mt = mean(ts);
  const my = mean(ys);
  let num = 0;
  let den = 0;
  for (let i = 0; i < n; i += 1) {
    const dt = ts[i] - mt;
    num += dt * (ys[i] - my);
    den += dt * dt;
  }
  return den > 1e-9 ? num / den : 0;
}

/**
 * Remove the slow trend with a centred moving mean.
 *
 * The offline pipeline high-passes with a zero-phase Butterworth and documents
 * why a moving mean is not good enough for *features*: over a strongly curved
 * drift the mean of a convex function sits above its centre value, leaving a
 * residual comparable to the sway being measured. That objection stands, and is
 * the reason nothing derived here is offered as a score. For deciding whether a
 * signal is periodic and at what rate, a residual bias that varies slowly is
 * harmless — it moves the whole trace, not its rhythm.
 */
function detrend(values, halfWidth) {
  const n = values.length;
  const out = new Array(n);
  for (let i = 0; i < n; i += 1) {
    const lo = Math.max(0, i - halfWidth);
    const hi = Math.min(n - 1, i + halfWidth);
    let s = 0;
    for (let j = lo; j <= hi; j += 1) s += values[j];
    out[i] = values[i] - s / (hi - lo + 1);
  }
  return out;
}

/**
 * Strongest normalised autocorrelation of a uniformly sampled signal within a
 * plausible stride-period band, returned as `{ r, periodS }`.
 *
 * This is the periodicity test that separates walking from every other kind of
 * movement. Waving, shifting weight and camera shake all produce motion energy;
 * only gait produces motion energy concentrated at one repeating period.
 */
function periodicity(values, fps, minPeriodS = 0.55, maxPeriodS = 2.2) {
  const n = values.length;
  const minLag = Math.max(2, Math.round(minPeriodS * fps));
  const maxLag = Math.min(n - 4, Math.round(maxPeriodS * fps));
  if (maxLag <= minLag) return { r: 0, periodS: 0 };

  let energy = 0;
  for (let i = 0; i < n; i += 1) energy += values[i] * values[i];
  if (energy < 1e-12) return { r: 0, periodS: 0 };

  let best = { r: 0, periodS: 0 };
  for (let lag = minLag; lag <= maxLag; lag += 1) {
    let num = 0;
    let a = 0;
    let b = 0;
    for (let i = 0; i + lag < n; i += 1) {
      num += values[i] * values[i + lag];
      a += values[i] * values[i];
      b += values[i + lag] * values[i + lag];
    }
    const den = Math.sqrt(a * b);
    const r = den > 1e-12 ? num / den : 0;
    if (r > best.r) best = { r, periodS: lag / fps };
  }
  return best;
}

/**
 * Step events from the detrended ankle gap, by Schmitt trigger.
 *
 * The gap crosses zero once per step: positive when the left ankle is the
 * leading foot, negative when the right one is. Requiring the signal to reach
 * ±`STEP_DEADBAND` before the state flips means a step has to actually happen —
 * plain zero-crossing detection counts landmark noise as steps whenever the feet
 * are together, which is precisely when someone is standing still.
 *
 * Returns `[{ t, dir }]`, `dir` being +1 when the left foot took the lead.
 */
function stepEvents(times, gap, deadband = STEP_DEADBAND) {
  const events = [];
  let state = 0;
  for (let i = 0; i < gap.length; i += 1) {
    const next = gap[i] > deadband ? 1 : gap[i] < -deadband ? -1 : 0;
    if (next !== 0 && next !== state) {
      if (state !== 0) events.push({ t: times[i], dir: next });
      state = next;
    }
  }
  return events;
}

function diffs(values) {
  const out = [];
  for (let i = 1; i < values.length; i += 1) out.push(values[i] - values[i - 1]);
  return out;
}

/* ------------------------------------------------------------------------- *
 * Scene vocabulary
 *
 * Every state carries what was seen, what it means for coaching, and the one
 * thing to change. "This is not a ramp walk" on its own is a dead end; "you are
 * walking across the frame — turn and walk toward the lens" is actionable, and
 * it is the same sentence a coach in the room would say.
 * ------------------------------------------------------------------------- */
const SCENES = {
  searching: {
    title: "No one in frame",
    detail: "Nothing that looks like a body has been found in the last moment.",
    fix: "Step into view, with the whole body from head to feet inside the frame.",
  },
  settling: {
    title: "Reading the scene",
    detail: "Collecting a second or two of movement before saying anything.",
    fix: "Keep walking — the first verdict lands in about a second.",
  },
  crowded: {
    title: "More than one person in frame",
    detail:
      "Live coaching follows a single walker. With two bodies in shot the "
      + "landmarks can jump between them mid-stride, which would show up as "
      + "sway and asymmetry neither person has.",
    fix: "Clear the frame, or upload the clip to Group analysis, which scores each walker separately.",
  },
  framing: {
    title: "Not the whole body",
    detail:
      "Your feet are out of frame for much of the time. When the lower body is "
      + "missing the hips are guessed rather than seen, and trunk lean, step "
      + "width and stride all stop meaning anything.",
    fix: "Step back, or tilt the camera down, until your feet stay in shot.",
  },
  far: {
    title: "Too far from the camera",
    detail:
      "Your body is a small part of the frame, so landmark noise is a large "
      + "fraction of every measurement.",
    fix: "Come closer, or move the camera in, until you fill most of the frame's height.",
  },
  close: {
    title: "Too close to the camera",
    detail:
      "You fill the frame, which means the walk is being cropped and the "
      + "perspective is changing faster than the measurements can follow.",
    fix: "Move the camera further back — about three or four metres works.",
  },
  still: {
    title: "Standing still — not a walk",
    detail:
      "You are in frame and being tracked, but your feet are not stepping and "
      + "your body is not travelling. Posture can be checked standing; nothing "
      + "about a walk can.",
    fix: "Start walking toward the camera to get a walk read.",
  },
  fidget: {
    title: "Moving, but not walking",
    detail:
      "There is movement, but it has no repeating stride rhythm in it — so this "
      + "is not gait. Shifting weight, turning, gesturing and adjusting clothes "
      + "all land here.",
    fix: "Take at least four unbroken steps toward the camera.",
  },
  crossing: {
    title: "Walking across the frame",
    detail:
      "You are walking, but across the camera rather than down the runway "
      + "toward it. A side view cannot see left-right differences at all — step "
      + "width, shoulder level and symmetry are invisible from there.",
    fix: "Turn and walk toward the lens; keep the camera at the end of the runway.",
  },
  profile: {
    title: "Turned away from the lens",
    detail:
      "You are walking, but at a steep angle to the camera. Shoulder level, "
      + "step width and symmetry all need a near-frontal view.",
    fix: "Square your shoulders to the camera and walk straight at it.",
  },
  receding: {
    title: "Walking away from the camera",
    detail:
      "You are getting smaller in frame, so this is the return leg. It is a "
      + "real walk, but the face, gaze and chest are all facing the wrong way "
      + "for the cues below.",
    fix: "Turn at the end and walk back toward the camera.",
  },
  running: {
    title: "That is a run, not a walk",
    detail:
      "Your cadence is above what a runway walk reaches. The rubric's timing "
      + "bands are built around a walking gait and would score this as chaos.",
    fix: "Slow to a deliberate walking pace — around 100 steps a minute.",
  },
  walk: {
    title: "Ramp walk detected",
    detail: "Walking toward the camera, whole body in frame, near head-on. Cues are live.",
    fix: "",
  },
};

/** Order matters: the first failing test names the scene. Framing problems come
 *  before movement ones because a cropped body makes the movement tests
 *  unreliable, and saying "you are standing still" to somebody whose legs are
 *  simply out of shot is worse than saying nothing. */
export const SCENE_IDS = Object.keys(SCENES);

/* ------------------------------------------------------------------------- */

/**
 * Rolling scene reader. Feed it one frame of landmarks at a time; ask it what is
 * happening whenever the UI wants to repaint.
 */
export class SceneAnalyser {
  constructor({ windowS = WINDOW_S } = {}) {
    this.windowS = windowS;
    this.reset();
  }

  reset() {
    /** One entry per frame, trimmed to the rolling window. */
    this.samples = [];
    /** Median filters for the three posture cues, and their hysteresis state. */
    this.cueBuf = { lean: [], tilt: [], head: [] };
    this.cueState = { lean: "good", tilt: "good", head: "good" };
    /** Committed scene, the candidate trying to replace it, and when each began. */
    this.scene = "searching";
    this.sceneSince = 0;
    this.candidate = null;
    this.candidateSince = 0;
    /** Completed walking passes, and the one in progress. */
    this.passes = [];
    this.pass = null;
    this.lastT = 0;
    this.people = 0;
  }

  /**
   * Add one frame.
   *
   * @param points   33 landmarks (`{x, y, visibility}` in normalised frame
   *                 coordinates) or null when nothing was detected.
   * @param t        Timestamp in seconds; any consistent clock will do.
   * @param aspect   Frame height divided by frame width. Normalised coordinates
   *                 are scaled per axis, so without this a horizontal span
   *                 cannot be compared with a vertical one.
   * @param people   How many plausible bodies the detector returned.
   */
  push(points, t, aspect = 0.75, people = points ? 1 : 0) {
    this.lastT = t;
    this.people = people;

    if (points) {
      const vis = (i) => points[i]?.visibility ?? 1;
      const mid = (a, b) => [
        (points[a].x + points[b].x) / 2,
        (points[a].y + points[b].y) / 2,
      ];
      const [sx, sy] = mid(L.leftShoulder, L.rightShoulder);
      const [hx, hy] = mid(L.leftHip, L.rightHip);
      const torso = Math.hypot(sx - hx, sy - hy) || 1e-6;

      // Same expressions as features.builder, on raw normalised coordinates.
      const lean = Math.abs(Math.atan2(sx - hx, -(sy - hy)) * 180 / Math.PI);
      let tilt = Math.atan2(
        -(points[L.leftShoulder].y - points[L.rightShoulder].y),
        points[L.leftShoulder].x - points[L.rightShoulder].x,
      ) * 180 / Math.PI;
      // A shoulder line is undirected: fold to +/-90 so a level line reads 0
      // whichever way the vector was taken. transverse_tilt does the same.
      if (tilt > 90) tilt -= 180;
      else if (tilt < -90) tilt += 180;

      const shoulderSpan = Math.hypot(
        points[L.leftShoulder].x - points[L.rightShoulder].x,
        points[L.leftShoulder].y - points[L.rightShoulder].y,
      );

      this.samples.push({
        t,
        torso,
        lean,
        tilt: Math.abs(tilt),
        head: Math.abs(points[L.nose].x - sx) / torso,
        // Signed ankle gap in torso lengths: the gait signal.
        gap: (points[L.leftAnkle].x - points[L.rightAnkle].x) / torso,
        stepWidth: Math.abs(points[L.leftAnkle].x - points[L.rightAnkle].x) / torso,
        // Kept in raw image units, *not* divided by this frame's torso. The
        // pelvis is normalised later, by the window's median torso: dividing a
        // position by a scale that itself grows as the subject approaches
        // manufactures a lateral drift that is pure perspective. The offline
        // builder makes the same point about the same quantity - filter first in
        // image units, then convert to torso lengths.
        pelvisX: hx,
        // cos(yaw) = (shoulder span / torso) / (anthropometric ratio x aspect).
        // features.viewpoint derives the identical expression.
        facingCos: shoulderSpan / torso / Math.max(SHOULDER_TORSO_RATIO * aspect, 1e-6),
        legsVisible: (vis(L.leftAnkle) > TH.visibility || vis(L.rightAnkle) > TH.visibility)
          && points[L.leftAnkle].y < 0.995 && points[L.rightAnkle].y < 0.995,
        headVisible: vis(L.nose) > TH.visibility && points[L.nose].y > 0.005,
        present: true,
      });

      // The cue medians are fed here rather than in read(): the UI reads at a
      // fraction of the frame rate, and a nine-sample median over UI ticks is a
      // second of lag instead of the third of a second it is meant to be.
      const last = this.samples.at(-1);
      for (const cue of CUES) {
        const buf = this.cueBuf[cue.id];
        buf.push(last[cue.id]);
        if (buf.length > MEDIAN_WINDOW) buf.shift();
      }
    } else {
      this.samples.push({ t, present: false });
    }

    while (this.samples.length && t - this.samples[0].t > this.windowS) {
      this.samples.shift();
    }
    return this;
  }

  /** Everything the UI needs, recomputed from the current window. */
  read() {
    const s = this.samples;
    const present = s.filter((x) => x.present);
    const span = s.length > 1 ? s[s.length - 1].t - s[0].t : 0;
    const fps = span > 0.2 ? (s.length - 1) / span : 30;

    const m = this._measure(present, fps);
    const candidate = this._classify(present, span, m);
    this._commit(candidate, this.lastT);
    const isWalk = this.scene === "walk";

    // Cues follow the same median-plus-hysteresis treatment as before, but are
    // only *offered* during a walk. Feeding them while the subject is turning
    // round at the end of a pass is what made the old readout meaningless.
    const cues = this._cues(present.at(-1), isWalk);
    this._track(isWalk, m, cues);

    return {
      ready: span >= MIN_DATA_S && present.length > 4,
      scene: {
        id: this.scene,
        ...SCENES[this.scene],
        isWalk,
        since: Math.max(0, this.lastT - this.sceneSince),
      },
      // What the *next* verdict is leaning toward, so a state about to change
      // does not look frozen.
      pending: this.candidate && this.candidate !== this.scene
        ? { id: this.candidate, title: SCENES[this.candidate].title }
        : null,
      checks: this._checks(present, span, m),
      metrics: m,
      cues,
      series: this._series(present, m),
      passes: this.passes,
      live: this.pass,
      people: this.people,
    };
  }

  /* --- measurement ------------------------------------------------------- */

  _measure(present, fps) {
    const n = present.length;
    if (n < 4) {
      return { n, fps, torso: 0, coverage: 0, legShare: 0, headShare: 0,
               facing: 90, approach: 0, lateral: 0, gapAmp: 0, sway: 0,
               travel: 0, periodicity: 0, steps: 0, cadence: 0, strideCv: 0,
               asymmetry: 0, stepWidth: 0, stridePeriod: 0, events: [],
               recentGap: 0, recentTravel: 0 };
    }

    const times = present.map((x) => x.t);
    const torso = median(present.map((x) => x.torso));
    const halfWidth = Math.max(2, Math.round((DETREND_S * fps) / 2));

    const gap = detrend(present.map((x) => x.gap), halfWidth);
    const gapSorted = [...gap].sort((a, b) => a - b);
    // Pelvis motion in torso lengths: detrended in image units, then divided by
    // the window's median torso rather than each frame's own.
    const pelvis = present.map((x) => x.pelvisX / torso);
    const swayTrace = detrend(pelvis, halfWidth);
    const swaySorted = [...swayTrace].sort((a, b) => a - b);

    const { r, periodS } = periodicity(gap, fps);
    const events = stepEvents(times, gap);
    const stepIntervals = diffs(events.map((e) => e.t));
    const cadence = stepIntervals.length ? 60 / mean(stepIntervals) : 0;

    // Stride time is the interval between *same-foot* events, which is every
    // other trigger. Pooling both feet into one train inflates the spread
    // whenever the two are not exactly antiphase — the feature builder makes the
    // same distinction and for the same reason.
    const strideCvs = [];
    for (const dir of [1, -1]) {
      const iv = diffs(events.filter((e) => e.dir === dir).map((e) => e.t));
      if (iv.length >= 2) strideCvs.push(stdev(iv) / Math.max(mean(iv), 1e-6));
    }

    // Step-time asymmetry: in a symmetric walk the left-to-right and
    // right-to-left intervals are equal. Made of event times, so unlike an
    // amplitude ratio it survives projection.
    const odd = stepIntervals.filter((_, i) => i % 2 === 0);
    const even = stepIntervals.filter((_, i) => i % 2 === 1);
    const asymmetry = odd.length && even.length
      ? Math.abs(mean(odd) - mean(even)) / Math.max(mean(stepIntervals), 1e-6)
      : 0;

    // "Has stopped" is judged on the last second and a half rather than the
    // whole window. A four-second window still holds four seconds of walking
    // when somebody stops dead, so a full-window test keeps insisting the walk
    // is live for seconds after it ended - and the pass log then credits the
    // pause to the pass. The raw gap serves here without detrending: it is
    // already differential, so it carries no perspective drift to remove.
    const recentFrom = present.at(-1).t - 1.5;
    const recent = present.filter((x) => x.t >= recentFrom);
    const recentGaps = recent.map((x) => x.gap);
    const recentPelvis = recent.map((x) => x.pelvisX / torso);

    return {
      n,
      fps,
      torso,
      coverage: this.samples.length ? n / this.samples.length : 0,
      legShare: present.filter((x) => x.legsVisible).length / n,
      headShare: present.filter((x) => x.headVisible).length / n,
      // acos is flat near zero, so this separates head-on from angled from
      // side-on and nothing finer. features.viewpoint says the same.
      facing: Math.acos(Math.min(1, Math.max(0, median(present.map((x) => x.facingCos)))))
        * 180 / Math.PI,
      approach: slope(times, present.map((x) => x.torso)) / Math.max(torso, 1e-6),
      lateral: slope(times, pelvis),
      travel: Math.max(...pelvis) - Math.min(...pelvis),
      gapAmp: percentile(gapSorted, 95) - percentile(gapSorted, 5),
      recentGap: recent.length > 3 ? Math.max(...recentGaps) - Math.min(...recentGaps) : 0,
      recentTravel: recent.length > 3
        ? Math.max(...recentPelvis) - Math.min(...recentPelvis) : 0,
      sway: percentile(swaySorted, 95) - percentile(swaySorted, 5),
      periodicity: r,
      stridePeriod: periodS,
      steps: events.length,
      cadence,
      strideCv: strideCvs.length ? mean(strideCvs) : 0,
      asymmetry,
      stepWidth: median(present.map((x) => x.stepWidth)),
      events,
    };
  }

  /* --- classification --------------------------------------------------- */

  _classify(present, span, m) {
    if (!present.length || m.coverage < 0.4) return "searching";
    if (this.people > 1) return "crowded";
    if (span < MIN_DATA_S || m.n < 8) return "settling";

    if (m.legShare < TH.legFrames || m.headShare < TH.legFrames) return "framing";
    if (m.torso < TH.torso[0]) return "far";
    if (m.torso > TH.torso[1]) return "close";

    const stepping = m.recentGap >= TH.stillGap;
    const travelling = m.recentTravel >= TH.stillTravel || Math.abs(m.approach) > 0.04;
    if (!stepping && !travelling) return "still";
    if (m.steps < TH.steps || m.periodicity < TH.periodicity) return "fidget";
    if (m.cadence > TH.cadence[1]) return "running";
    if (m.facing > TH.facing) {
      return Math.abs(m.lateral) > TH.crossing ? "crossing" : "profile";
    }
    if (m.approach < TH.approach) return "receding";
    return "walk";
  }

  /** Dwell-time state machine: a candidate has to hold before it takes over. */
  _commit(candidate, t) {
    if (candidate === this.scene) {
      this.candidate = null;
      return;
    }
    if (candidate !== this.candidate) {
      this.candidate = candidate;
      this.candidateSince = t;
      return;
    }
    const hold = HOLD_S[candidate] ?? HOLD_S.default;
    if (t - this.candidateSince >= hold) {
      this.scene = candidate;
      this.sceneSince = t;
      this.candidate = null;
    }
  }

  /**
   * The six things a live walk read needs, each with its measured value.
   *
   * Shown as a checklist rather than folded into one verdict because the failing
   * item *is* the instruction: a user who can see "facing the lens — 61° off"
   * knows what to change without reading a paragraph, and a user who sees all
   * six green knows the cues below can be trusted.
   */
  _checks(present, span, m) {
    const pct = (v) => `${Math.round(v * 100)}%`;
    if (!present.length) {
      return [{ id: "person", label: "Someone in frame", ok: false, detail: "nobody detected" }];
    }
    return [
      {
        id: "frame",
        label: "Whole body in frame",
        ok: m.legShare >= TH.legFrames && m.headShare >= TH.legFrames,
        detail: `feet visible ${pct(m.legShare)} of the time`,
      },
      {
        id: "distance",
        label: "Distance from camera",
        ok: m.torso >= TH.torso[0] && m.torso <= TH.torso[1],
        detail: `torso fills ${pct(m.torso)} of frame height`,
      },
      {
        id: "gait",
        label: "Repeating stride rhythm",
        ok: m.steps >= TH.steps && m.periodicity >= TH.periodicity,
        detail: m.steps
          ? `${m.steps} step${m.steps === 1 ? "" : "s"}, rhythm ${m.periodicity.toFixed(2)}`
          : "no steps detected",
      },
      {
        id: "toward",
        label: "Coming toward the camera",
        ok: m.approach >= TH.approach && Math.abs(m.lateral) <= TH.crossing,
        detail: `${m.approach >= 0 ? "+" : ""}${(m.approach * 100).toFixed(0)}% size/s`
          + `, ${Math.abs(m.lateral).toFixed(2)} torso/s sideways`,
      },
      {
        id: "facing",
        label: "Facing the lens",
        ok: m.facing <= TH.facing,
        detail: `about ${Math.round(m.facing)}° off head-on`,
      },
      {
        id: "pace",
        label: "Walking pace",
        ok: m.cadence >= TH.cadence[0] && m.cadence <= TH.cadence[1],
        detail: m.cadence ? `${Math.round(m.cadence)} steps/min` : "not measurable yet",
      },
    ];
  }

  /* --- cues and passes -------------------------------------------------- */

  _cues(last, isWalk) {
    if (!last) return {};
    const out = {};
    for (const cue of CUES) {
      const buf = this.cueBuf[cue.id];
      if (!buf.length) continue;
      const value = median(buf);
      const was = this.cueState[cue.id];
      // Hysteresis: only cross into "off" past `warn`, and back to "good" only
      // below `good`. Between them the previous state is held.
      const state = value > cue.warn ? "off" : value < cue.good ? "good" : was;
      this.cueState[cue.id] = state;
      out[cue.id] = { value, state, live: isWalk };
    }
    return out;
  }

  /**
   * Accumulate one pass per continuous walk, so a session reads as a list of
   * attempts rather than a number that resets every time you turn round.
   */
  _track(isWalk, m, cues) {
    if (isWalk && !this.pass) {
      this.pass = {
        index: this.passes.length + 1,
        start: this.sceneSince,
        seconds: 0,
        frames: 0,
        steps: 0,
        cadence: 0,
        sway: 0,
        facing: 0,
        off: { lean: 0, tilt: 0, head: 0 },
      };
    }
    if (isWalk && this.pass) {
      const p = this.pass;
      p.seconds = Math.max(0, this.lastT - p.start);
      p.frames += 1;
      p.steps = Math.max(p.steps, m.steps);
      // Running means, so a long pass is not dominated by its first second.
      p.cadence += (m.cadence - p.cadence) / p.frames;
      p.sway += (m.sway - p.sway) / p.frames;
      p.facing += (m.facing - p.facing) / p.frames;
      for (const cue of CUES) {
        if (cues[cue.id]?.state === "off") p.off[cue.id] += 1;
      }
    }
    if (!isWalk && this.pass) {
      const p = this.pass;
      this.pass = null;
      // A pass shorter than two seconds is a glance, not an attempt; logging it
      // would fill the list with noise every time somebody crossed the frame.
      if (p.seconds >= 2 && p.frames > 10) {
        this.passes.unshift({
          ...p,
          off: Object.fromEntries(
            Object.entries(p.off).map(([k, v]) => [k, v / p.frames]),
          ),
        });
        this.passes = this.passes.slice(0, 8);
      }
    }
  }

  /** The gait trace and its step marks, for the live sparkline. */
  _series(present, m) {
    if (present.length < 6) return null;
    const t0 = present[0].t;
    const t1 = present.at(-1).t;
    const halfWidth = Math.max(2, Math.round((DETREND_S * m.fps) / 2));
    const gap = detrend(present.map((x) => x.gap), halfWidth);
    const scale = Math.max(...gap.map(Math.abs), 0.12);
    return {
      t0,
      t1,
      // Normalised to +/-1 so the sparkline is readable at any stride amplitude;
      // the amplitude itself is reported as a number beside it.
      points: present.map((x, i) => [(x.t - t0) / Math.max(t1 - t0, 1e-6), gap[i] / scale]),
      steps: (m.events ?? []).map((e) => ({
        x: (e.t - t0) / Math.max(t1 - t0, 1e-6),
        dir: e.dir,
      })),
      amplitude: m.gapAmp,
    };
  }
}

export default SceneAnalyser;
