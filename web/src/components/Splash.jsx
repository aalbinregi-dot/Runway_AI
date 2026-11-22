import { useCallback, useEffect, useRef, useState } from "react";
import "./Splash.css";

const PTS = {
  head: [100, 48],
  ls: [85, 70],
  rs: [115, 70],
  le: [72, 92],
  re: [128, 92],
  lw: [68, 114],
  rw: [132, 114],
  lh: [92, 112],
  rh: [108, 112],
  lk: [90, 142],
  rk: [110, 142],
  la: [88, 172],
  ra: [112, 172],
};

const LINKS = [
  ["ls", "rs"],
  ["lh", "rh"],
  ["ls", "lh"],
  ["rs", "rh"],
  ["ls", "le"],
  ["le", "lw"],
  ["rs", "re"],
  ["re", "rw"],
  ["lh", "lk"],
  ["lk", "la"],
  ["rh", "rk"],
  ["rk", "ra"],
];

const TITLE_LETTERS = "RUNWAY COACH".split("");

/**
 * Animated splash screen matching splash.html.
 *
 * Props:
 *  - ready (boolean): true when the application prerequisites (auth, health) have resolved.
 *  - onDone (function): invoked after the exit animation completes (~900ms) to unmount splash.
 *  - minMs (number): minimum visible duration in milliseconds (default: 3200ms).
 */
export default function Splash({ ready = false, onDone, minMs = 3200 }) {
  const [exiting, setExiting] = useState(false);
  const startRef = useRef(Date.now());
  const finishedRef = useRef(false);

  const finish = useCallback(() => {
    if (finishedRef.current) return;
    finishedRef.current = true;
    setExiting(true);

    // Allow exit transition (~0.9s: beam scale + fade) to play before unmounting
    setTimeout(() => {
      onDone?.();
    }, 900);
  }, [onDone]);

  // Finish only when BOTH ready is true AND minMs has elapsed since mount
  useEffect(() => {
    if (!ready || finishedRef.current) return;
    const elapsed = Date.now() - startRef.current;
    const remaining = Math.max(0, minMs - elapsed);
    const timer = setTimeout(finish, remaining);
    return () => clearTimeout(timer);
  }, [ready, minMs, finish]);

  // Safety timeout: 8000ms always finishes it regardless of external state
  useEffect(() => {
    const safety = setTimeout(finish, 8000);
    return () => clearTimeout(safety);
  }, [finish]);

  return (
    <div
      className={`splash${exiting ? " exiting" : ""}`}
      role="status"
      aria-live="polite"
      aria-label="Loading Runway Coach"
      onClick={finish}
    >
      <div className="scene">
        <div className="beam" />
        <div className="runway" />

        <div className="fig">
          <svg viewBox="0 20 200 170" aria-hidden="true">
            <g id="bones">
              <circle
                cx={100}
                cy={48}
                r={10}
                className="bone"
                style={{
                  strokeDasharray: 70,
                  strokeDashoffset: 70,
                  animationDelay: "1.8s",
                }}
              />
              <line
                x1={100}
                y1={58}
                x2={100}
                y2={70}
                className="bone"
                style={{ animationDelay: "1.8s" }}
              />
              {LINKS.map(([aKey, bKey], i) => {
                const a = PTS[aKey];
                const b = PTS[bKey];
                return (
                  <line
                    key={`${aKey}-${bKey}`}
                    x1={a[0]}
                    y1={a[1]}
                    x2={b[0]}
                    y2={b[1]}
                    className="bone"
                    style={{ animationDelay: `${1.8 + i * 0.06}s` }}
                  />
                );
              })}
            </g>

            <g id="dots">
              {Object.keys(PTS).map((k, i) => (
                <circle
                  key={k}
                  cx={PTS[k][0]}
                  cy={PTS[k][1]}
                  r={k === "head" ? 5 : 3.8}
                  className="dot"
                  style={{ animationDelay: `${0.9 + i * 0.07}s` }}
                />
              ))}
            </g>
          </svg>
        </div>

        <div className="title" aria-hidden="true">
          {TITLE_LETTERS.map((ch, i) => (
            <span key={i} style={{ animationDelay: `${1.5 + i * 0.07}s` }}>
              {ch === " " ? "\u00A0" : ch}
            </span>
          ))}
        </div>

        <div className="tag">Video in, marks and drills out</div>
        <div className="progress">
          <i />
        </div>
        <div className="skip">Tap to skip</div>
      </div>
    </div>
  );
}
