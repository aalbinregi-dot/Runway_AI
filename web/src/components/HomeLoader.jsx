import { useEffect, useRef, useState } from "react";

const BONES = [
  ["ls", "rs"], ["lh", "rh"], ["ls", "lh"], ["rs", "rh"],
  ["ls", "le"], ["le", "lw"], ["rs", "re"], ["re", "rw"],
  ["lh", "lk"], ["lk", "la"], ["rh", "rk"], ["rk", "ra"],
  ["neck", "head"]
];

const JOINT_NAMES = [
  "head", "ls", "rs", "le", "re", "lw", "rw",
  "lh", "rh", "lk", "rk", "la", "ra"
];

const STAR_COORDS = [
  [48, 60, 5], [152, 70, 6], [40, 120, 4],
  [160, 128, 5], [70, 38, 4], [132, 40, 4]
];

function at(p, len, a) {
  return { x: p.x + len * Math.sin(a), y: p.y + len * Math.cos(a) };
}

const BEAT_SPEED = 5.2;

function buildPose(o) {
  const P = {};
  const pel = { x: 100 + (o.px || 0), y: 118 + (o.py || 0) };
  P.lh = { x: pel.x - 8, y: pel.y };
  P.rh = { x: pel.x + 8, y: pel.y };

  const up = Math.PI + (o.tilt || 0);
  const neck = at(pel, 38, up);
  P.neck = neck;

  const sh = o.sh || 0;
  P.ls = { x: neck.x - 15 * Math.cos(sh), y: neck.y - 15 * Math.sin(sh) };
  P.rs = { x: neck.x + 15 * Math.cos(sh), y: neck.y + 15 * Math.sin(sh) };
  P.head = at(neck, 17, up + (o.ht || 0));

  P.le = at(P.ls, 21, -o.aL);
  P.lw = at(P.le, 19, -(o.aL + o.bL));

  if (o.hipR) {
    P.re = { x: P.rs.x + 13, y: P.rs.y + 13 };
    P.rw = { x: P.rh.x + 3, y: P.rh.y - 2 };
  } else {
    P.re = at(P.rs, 21, o.aR);
    P.rw = at(P.re, 19, o.aR + o.bR);
  }

  if (o.hipL) {
    P.le = { x: P.ls.x - 13, y: P.ls.y + 13 };
    P.lw = { x: P.lh.x - 3, y: P.lh.y - 2 };
  }

  P.lk = at(P.lh, 28, o.lt);
  P.la = at(P.lk, 28, o.lt + o.ls);
  P.rk = at(P.rh, 28, o.rt);
  P.ra = at(P.rk, 28, o.rt + o.rs);
  return P;
}

function spinX(P, k) {
  const out = {};
  for (const n in P) {
    out[n] = { x: 100 + (P[n].x - 100) * k, y: P[n].y };
  }
  return out;
}

const MOVES = [
  // 0: Groove walk - natural arm swing, relaxed tilt
  function (t) {
    const s2 = Math.sin(t * BEAT_SPEED);
    const b = Math.abs(s2);
    return buildPose({
      py: 3 * b, px: 4 * s2, tilt: 0.14 * s2,
      sh: 0.12 * Math.sin(t * BEAT_SPEED + 1.2), ht: 0.1 * Math.sin(t * BEAT_SPEED + 0.5),
      aL: 1.7 + Math.sin(t * BEAT_SPEED), bL: 0.6 + 0.3 * Math.sin(t * 2 * BEAT_SPEED),
      aR: 1.7 - Math.sin(t * BEAT_SPEED), bR: 0.6 - 0.3 * Math.sin(t * 2 * BEAT_SPEED),
      lt: -0.1 + 0.4 * s2, ls: -0.5 * Math.max(0, s2),
      rt: 0.1 - 0.4 * s2, rs: 0.5 * Math.max(0, -s2),
    });
  },
  // 1: Catwalk strut - hand on hip, knee lift, pelvic sway
  function (t) {
    const s2 = Math.sin(t * BEAT_SPEED);
    const l = Math.max(0, s2);
    const r = Math.max(0, -s2);
    return buildPose({
      px: 7 * s2, py: 2 * Math.abs(Math.cos(t * BEAT_SPEED)),
      tilt: -0.1 * s2, sh: -0.14 * s2, ht: 0.08 * s2,
      aL: 0.5 + 0.5 * s2, bL: 0.15, aR: 0, bR: 0, hipR: true,
      lt: -0.02 - 0.1 * s2, ls: -0.9 * l,
      rt: 0.02 - 0.1 * s2, rs: 0.9 * r,
    });
  },
  // 2: 360-degree runway spin - arms elevated
  function (t) {
    const ph = Math.min(1, t / 2.2);
    const k = Math.cos(ph * Math.PI * 4);
    const P = buildPose({
      py: -3 * Math.sin(ph * Math.PI), tilt: 0,
      aL: 2.7, bL: 0.25, aR: 2.7, bR: 0.25,
      lt: -0.1, ls: 0.2, rt: 0.28, rs: -0.9,
    });
    return spinX(P, Math.abs(k) < 0.12 ? 0.12 * Math.sign(k || 1) : k);
  },
  // 3: Signature pose - hands on both hips, hold, sparkle
  function (t) {
    const e = Math.min(1, t / 0.5);
    const h = Math.sin(t * BEAT_SPEED * 0.5) * 0.03;
    return buildPose({
      py: 1, px: 3 * e, tilt: 0.08 * e + h, sh: 0.1 * e, ht: -0.18 * e,
      hipL: true, hipR: true, aL: 0.6, bL: 0, aR: 0, bR: 0,
      lt: -0.22 * e, ls: 0.1, rt: 0.2 * e, rs: -0.05,
    });
  },
  // 4: Runway wave - greeting the judges/camera
  function (t) {
    const s2 = Math.sin(t * BEAT_SPEED);
    const w = Math.sin(t * BEAT_SPEED * 2.2);
    return buildPose({
      px: 2 * s2, py: 2 * Math.abs(s2), tilt: 0.07 * s2, ht: 0.12 * s2,
      aL: 2.55 + 0.12 * w, bL: 0.55 * w, aR: 0, bR: 0, hipR: true,
      lt: -0.16, ls: 0.1, rt: 0.16, rs: -0.1,
    });
  },
];

const MOVE_DURATION = 2.6;
const FADE_DURATION = 0.35;

function computePose(T) {
  const idx = Math.floor(T / MOVE_DURATION);
  const lt = T - idx * MOVE_DURATION;
  const m = idx % MOVES.length;
  const P = MOVES[m](lt);

  if (lt < FADE_DURATION) {
    const prevIdx = (m + MOVES.length - 1) % MOVES.length;
    const Q = MOVES[prevIdx](MOVE_DURATION);
    let k = lt / FADE_DURATION;
    k = k * k * (3 - 2 * k); // smoothstep
    for (const n in P) {
      P[n] = {
        x: Q[n].x + (P[n].x - Q[n].x) * k,
        y: Q[n].y + (P[n].y - Q[n].y) * k,
      };
    }
  }

  P.m = m;
  P.lt = lt;
  return P;
}

const DEFAULT_STEPS = [
  "Finding the body",
  "Cleaning the motion",
  "Measuring 102 kinematic features",
  "Scoring 12 criteria",
  "Preparing your coaching plan",
];

export default function HomeLoader({ states, elapsed = 0, filename }) {
  const figRef = useRef(null);
  const headRef = useRef(null);
  const bonesRefs = useRef([]);
  const dotsRefs = useRef({});
  const starsRefs = useRef([]);
  const rafId = useRef(0);

  const [stepText, setStepText] = useState(DEFAULT_STEPS[0]);
  const [progressPct, setProgressPct] = useState(15);

  // Synchronize with server SSE stages or fallback to graceful progression
  useEffect(() => {
    if (states && Object.keys(states).length > 0) {
      if (states.score?.status === "active" || states.score?.status === "done") {
        setStepText(DEFAULT_STEPS[3]);
        setProgressPct(88);
      } else if (states.measure?.status === "active" || states.measure?.status === "done") {
        setStepText(DEFAULT_STEPS[2]);
        setProgressPct(62);
      } else if (states.pose?.status === "active") {
        const d = states.pose.done;
        const tot = states.pose.total;
        if (tot > 0) {
          const ratio = Math.min(1, d / tot);
          setStepText(`Tracking body landmarks (${d}/${tot})`);
          setProgressPct(20 + Math.floor(ratio * 35));
        } else {
          setStepText(DEFAULT_STEPS[0]);
          setProgressPct(25);
        }
      } else if (states.fetch?.status === "active") {
        setStepText("Downloading video from link");
        setProgressPct(12);
      }
    } else {
      // Dynamic fallback rotation
      let stepIdx = 0;
      const interval = setInterval(() => {
        stepIdx = (stepIdx + 1) % DEFAULT_STEPS.length;
        setStepText(DEFAULT_STEPS[stepIdx]);
        setProgressPct(Math.min(94, Math.floor(((stepIdx + 1) / DEFAULT_STEPS.length) * 92)));
      }, 2300);
      return () => clearInterval(interval);
    }
  }, [states]);

  // Biomechanical stick figure animation loop
  useEffect(() => {
    const reducedMotion =
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    function renderPose(t) {
      const P = computePose(t);

      // Bones
      BONES.forEach((bn, i) => {
        const line = bonesRefs.current[i];
        if (line && P[bn[0]] && P[bn[1]]) {
          line.setAttribute("x1", P[bn[0]].x);
          line.setAttribute("y1", P[bn[0]].y);
          line.setAttribute("x2", P[bn[1]].x);
          line.setAttribute("y2", P[bn[1]].y);
        }
      });

      // Head
      if (headRef.current && P.head) {
        headRef.current.setAttribute("cx", P.head.x);
        headRef.current.setAttribute("cy", P.head.y);
      }

      // Sparkles (active on signature pose m === 3)
      const sparkleOn = P.m === 3 ? Math.min(1, P.lt / 0.4) : 0;
      starsRefs.current.forEach((p, i) => {
        if (p) {
          const sinVal = Math.sin(P.lt * 7 + i * 1.7);
          p.style.opacity = sparkleOn * (0.45 + 0.55 * sinVal * sinVal);
        }
      });

      // Joints
      JOINT_NAMES.forEach((n) => {
        if (n === "head") return;
        const dot = dotsRefs.current[n];
        if (dot && P[n]) {
          dot.setAttribute("cx", P[n].x);
          dot.setAttribute("cy", P[n].y);
        }
      });
    }

    if (reducedMotion) {
      renderPose(MOVE_DURATION * 3 + 0.6);
      return;
    }

    const t0 = performance.now();
    function loop(now) {
      renderPose((now - t0) / 1000);
      rafId.current = requestAnimationFrame(loop);
    }

    rafId.current = requestAnimationFrame(loop);

    return () => {
      if (rafId.current) cancelAnimationFrame(rafId.current);
    };
  }, []);

  return (
    <div className="home-loader-container">
      <div className="home-loader-card" role="status" aria-live="polite">
        <div className="hl-stage">
          <svg viewBox="0 0 200 200" aria-hidden="true" className="hl-svg">
            <defs>
              <clipPath id="hl-clip">
                <circle cx="100" cy="100" r="88" />
              </clipPath>
            </defs>

            {/* Circular tracking telemetry rings */}
            <circle className="hl-ring-track" cx="100" cy="100" r="94" />
            <circle className="hl-ring-dots" cx="100" cy="100" r="99" />
            <circle className="hl-ring-spin" cx="100" cy="100" r="94" />

            <g clipPath="url(#hl-clip)">
              <circle className="hl-glow" cx="100" cy="100" r="88" />
              <line className="hl-floor" x1="55" y1="178" x2="145" y2="178" />

              {/* Sparkles */}
              <g id="hl-spk">
                {STAR_COORDS.map(([x, y, r], i) => (
                  <path
                    key={i}
                    ref={(el) => (starsRefs.current[i] = el)}
                    d={`M${x} ${y - r}L${x + r * 0.28} ${y - r * 0.28}L${x + r} ${y}L${x + r * 0.28} ${y + r * 0.28}L${x} ${y + r}L${x - r * 0.28} ${y + r * 0.28}L${x - r} ${y}L${x - r * 0.28} ${y - r * 0.28}Z`}
                    fill="#ffd9a8"
                    style={{ opacity: 0 }}
                  />
                ))}
              </g>

              {/* Skeleton Lines & Joints */}
              <g ref={figRef} id="hl-fig">
                {BONES.map((_, i) => (
                  <line
                    key={i}
                    ref={(el) => (bonesRefs.current[i] = el)}
                    className="hl-bone"
                  />
                ))}
                <circle ref={headRef} className="hl-head" r="9" />
                {JOINT_NAMES.map((n) => (
                  <circle
                    key={n}
                    ref={(el) => (dotsRefs.current[n] = el)}
                    className="hl-joint"
                    r={n === "head" ? 0 : 3.2}
                  />
                ))}
              </g>
            </g>
          </svg>
        </div>

        <div className="hl-content">
          <h2 className="hl-title">Analysing your walk</h2>
          <p className="hl-step">{stepText}...</p>

          <div className="hl-bar">
            <i className="hl-bar-fill" style={{ width: `${progressPct}%` }} />
          </div>

          {(filename || elapsed > 0) && (
            <div className="hl-meta">
              {filename && <span className="hl-file">{filename}</span>}
              {elapsed > 0 && (
                <span className="hl-time">{elapsed.toFixed(0)}s</span>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
