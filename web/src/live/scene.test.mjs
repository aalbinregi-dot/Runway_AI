/**
 * The live scene classifier, against staged scenarios.
 *
 * Run:  npm test        (from web/ — uses node --test, no dependencies)
 *
 * These are the cases the page exists to get right. Before this classifier the
 * live view showed the same three posture cues whether you were walking toward
 * the camera, standing still, crossing the frame or out of shot, so every one of
 * them is asserted here with a synthetic body driven through the exact motion.
 *
 * The generator is deliberately crude — a stick figure with the anthropometry the
 * server's camera-angle estimator assumes — because the classifier's inputs are
 * ratios of landmark distances, and those are the only properties that need to be
 * right for the decisions under test.
 */
import { strict as assert } from "node:assert";
import { describe, it } from "node:test";

import { SceneAnalyser, TH } from "./scene.js";

const RATIO = 0.215 / 0.288;   // shoulder width over torso, from data.synthetic
const ASPECT = 0.75;           // a 960x720 webcam
const FPS = 30;

/** One frame of landmarks from a handful of body parameters. */
function pose({ hipX = 0.5, hipY = 0.62, torso = 0.22, yawDeg = 0, gapTorso = 0,
                ankleY = 0.92, ankleVis = 1 } = {}) {
  const p = Array.from({ length: 33 }, () => ({ x: hipX, y: hipY, visibility: 1 }));
  const shoulderY = hipY - torso;
  // The shoulder line foreshortens as cos(yaw); the aspect factor is what makes a
  // horizontal span comparable with a vertical one in normalised coordinates.
  const span = RATIO * Math.cos((yawDeg * Math.PI) / 180) * ASPECT * torso;
  p[0] = { x: hipX, y: shoulderY - 0.06, visibility: 1 };                  // nose
  p[11] = { x: hipX + span / 2, y: shoulderY, visibility: 1 };             // shoulders
  p[12] = { x: hipX - span / 2, y: shoulderY, visibility: 1 };
  p[23] = { x: hipX + 0.2 * span, y: hipY, visibility: 1 };                // hips
  p[24] = { x: hipX - 0.2 * span, y: hipY, visibility: 1 };
  p[25] = { x: hipX + 0.05, y: (hipY + ankleY) / 2, visibility: 1 };       // knees
  p[26] = { x: hipX - 0.05, y: (hipY + ankleY) / 2, visibility: 1 };
  p[27] = { x: hipX + (gapTorso * torso) / 2, y: ankleY, visibility: ankleVis };
  p[28] = { x: hipX - (gapTorso * torso) / 2, y: ankleY, visibility: ankleVis };
  p[15] = { x: hipX + span * 0.7, y: shoulderY + torso * 0.9, visibility: 1 };
  p[16] = { x: hipX - span * 0.7, y: shoulderY + torso * 0.9, visibility: 1 };
  return p;
}

/** Drive the analyser through a scenario and return its final reading. */
function play(frames, { seconds = 5, people = 1, analyser } = {}) {
  const a = analyser ?? new SceneAnalyser();
  const n = Math.round(seconds * FPS);
  const t0 = a.lastT;
  for (let i = 0; i < n; i += 1) {
    const t = t0 + i / FPS;
    const f = frames(t, i);
    a.push(f, t, ASPECT, f ? people : 0);
    a.read();          // the page reads several times a second; dwell needs it
  }
  return a.read();
}

/** A walk: periodic ankle gap, growing apparent size, near head-on. */
const walking = ({ hz = 0.85, grow = 0.05, yaw = 5, drift = 0, torso = 0.22 } = {}) =>
  (t) => pose({
    hipX: 0.5 + drift * t,
    torso: torso * (1 + grow * t),
    yawDeg: yaw,
    gapTorso: 0.45 * Math.sin(2 * Math.PI * hz * t),
  });

const standing = () => (t) => pose({ gapTorso: 0.01 * Math.sin(t) });

describe("scene classification", () => {
  it("recognises a walk toward the camera", () => {
    const r = play(walking());
    assert.equal(r.scene.id, "walk");
    assert.equal(r.scene.isWalk, true);
    assert.ok(r.checks.every((c) => c.ok), "every requirement should pass");
  });

  it("says standing still is not a walk", () => {
    const r = play(standing());
    assert.equal(r.scene.id, "still");
    assert.equal(r.scene.isWalk, false);
    assert.ok(r.scene.fix.length > 0);
  });

  it("reports nobody in frame", () => {
    const r = play(() => null);
    assert.equal(r.scene.id, "searching");
  });

  it("separates crossing the frame from walking down it", () => {
    // Same cadence, but side-on and travelling laterally.
    const r = play(walking({ grow: 0, yaw: 72, drift: 0.09 }));
    assert.equal(r.scene.id, "crossing");
    assert.ok(r.metrics.facing > TH.facing);
  });

  it("separates a steep angle from a lateral crossing", () => {
    const r = play(walking({ yaw: 62, drift: 0.005 }));
    assert.equal(r.scene.id, "profile");
  });

  it("notices the return leg", () => {
    const r = play(walking({ grow: -0.08 }));
    assert.equal(r.scene.id, "receding");
    assert.ok(r.metrics.approach < TH.approach);
  });

  it("rejects movement with no stride rhythm", () => {
    // Motion energy without periodicity: shifting weight, gesturing, fidgeting.
    let seed = 1;
    const random = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const r = play((t) => pose({
      hipX: 0.5 + 0.02 * Math.sin(7.3 * t) + 0.02 * random(),
      torso: 0.22 + 0.004 * Math.sin(11 * t),
      gapTorso: 0.35 * (random() - 0.5),
    }));
    assert.equal(r.scene.id, "fidget");
    assert.ok(r.metrics.periodicity < TH.periodicity);
  });

  it("calls a run a run", () => {
    const r = play(walking({ hz: 1.6, grow: 0.12 }));
    assert.equal(r.scene.id, "running");
    assert.ok(r.metrics.cadence > TH.cadence[1]);
  });

  it("asks for the feet when the lower body is cropped", () => {
    const r = play((t) => {
      const p = walking()(t);
      p[27] = { ...p[27], y: 0.999, visibility: 0.2 };
      p[28] = { ...p[28], y: 0.999, visibility: 0.2 };
      return p;
    });
    assert.equal(r.scene.id, "framing");
    assert.equal(r.checks.find((c) => c.id === "frame").ok, false);
  });

  it("asks the walker to come closer when they are too small in frame", () => {
    const r = play(walking({ torso: 0.07, grow: 0 }));
    assert.equal(r.scene.id, "far");
  });

  it("refuses to coach two people at once", () => {
    const r = play(walking(), { people: 2 });
    assert.equal(r.scene.id, "crowded");
  });
});

describe("rolling measurements", () => {
  it("recovers a known cadence", () => {
    // 0.85 Hz stride is 1.7 steps a second: 102 steps a minute.
    const r = play(walking({ hz: 0.85 }));
    assert.ok(Math.abs(r.metrics.cadence - 102) < 8,
      `cadence ${r.metrics.cadence} should be near 102`);
  });

  it("finds a steady walk regular", () => {
    const r = play(walking(), { seconds: 8 });
    assert.ok(r.metrics.strideCv < 0.08, `stride cv ${r.metrics.strideCv}`);
    assert.ok(r.metrics.asymmetry < 0.1, `asymmetry ${r.metrics.asymmetry}`);
    assert.ok(r.metrics.periodicity > 0.8);
  });

  it("does not manufacture lateral drift from perspective", () => {
    // The pelvis is dead centre throughout; only the apparent size changes.
    // Dividing pelvis position by each frame's own torso once turned that into
    // 0.09 torso/s of sideways travel, which read as crossing the frame.
    const r = play(walking({ grow: 0.08, drift: 0 }));
    assert.ok(Math.abs(r.metrics.lateral) < 0.05,
      `lateral drift ${r.metrics.lateral} should be ~0`);
  });

  it("measures the camera angle it was given", () => {
    for (const yaw of [0, 30, 60]) {
      const r = play(walking({ yaw }));
      assert.ok(Math.abs(r.metrics.facing - yaw) < 10,
        `facing ${r.metrics.facing} for a ${yaw} degree camera`);
    }
  });

  it("withholds what a short window cannot measure", () => {
    // Six tenths of a second holds at most one step and no complete stride, so
    // the verdict is "settling" and the variability measures stay at zero rather
    // than reporting a spread computed from one interval.
    const r = play(walking(), { seconds: 0.6 });
    assert.equal(r.ready, false);
    assert.equal(r.scene.isWalk, false);
    assert.ok(["searching", "settling"].includes(r.scene.id), r.scene.id);
    assert.equal(r.metrics.strideCv, 0);
    assert.equal(r.metrics.asymmetry, 0);
  });
});

describe("passes", () => {
  it("logs one pass per continuous walk and closes it when the walk stops", () => {
    const a = new SceneAnalyser();
    play(walking(), { seconds: 6, analyser: a });
    assert.ok(a.pass, "a pass should be open while walking");
    play(standing(), { seconds: 5, analyser: a });
    assert.equal(a.pass, null, "the pass should close when the walk stops");
    assert.equal(a.passes.length, 1);
    const [pass] = a.passes;
    assert.ok(pass.seconds >= 2);
    assert.ok(pass.steps >= 4);
    assert.ok(Math.abs(pass.cadence - 102) < 15);
  });

  it("does not log a glance across the frame as a pass", () => {
    const a = new SceneAnalyser();
    play(walking({ yaw: 75, drift: 0.09, grow: 0 }), { seconds: 6, analyser: a });
    play(standing(), { seconds: 4, analyser: a });
    assert.equal(a.passes.length, 0);
  });
});
