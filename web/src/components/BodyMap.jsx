/**
 * The scores, on the body.
 *
 * Twelve rows of text require the reader to translate "hip_movement 3.1" into a
 * part of themselves before they can act on it. A figure does that translation:
 * the pelvis is dim, so the pelvis is the problem. It also makes the coverage
 * legible at a glance — five of the twelve criteria are not any one body part
 * (rhythm, symmetry, balance, confidence, presence) and those are drawn as what
 * they are: the ground under the figure, the axis through it, the footprints
 * behind it, the space around it.
 *
 * Brightness carries the score, as everywhere else in this interface: bright and
 * solid reads as excellent, dim as poor, which keeps the whole page working for
 * anyone with colour vision deficiency and keeps a monochrome brief honest.
 */
import { useState } from "react";
import { label } from "../criteria.js";

/** Region -> criterion. Every criterion appears exactly once, so the figure is a
 *  complete view of the twelve rather than a decorative subset. */
const REGIONS = [
  { key: "head_position", name: "Head" },
  { key: "shoulder_alignment", name: "Shoulders" },
  { key: "posture", name: "Spine" },
  { key: "hip_movement", name: "Pelvis" },
  { key: "arm_swing", name: "Arms" },
  { key: "stride_length", name: "Legs" },
  { key: "foot_placement", name: "Feet" },
  { key: "balance", name: "Base" },
  { key: "rhythm", name: "Steps behind" },
  { key: "symmetry", name: "Centre line" },
  { key: "confidence", name: "Travel" },
  { key: "stage_presence", name: "Presence" },
];

/** Opacity from a 0-10 score. Floored at 0.16 so a zero is still visible as a
 *  shape — an invisible region reads as "not measured", which is a different and
 *  much more forgiving statement than "measured, and bad". */
const shade = (score) => 0.16 + 0.74 * Math.max(0, Math.min(10, score ?? 0)) / 10;

export default function BodyMap({ scores, selected, onSelect }) {
  const [hover, setHover] = useState(null);
  if (!scores) return null;

  const focus = hover ?? selected;
  const fill = (key) => `rgba(246,245,243,${shade(scores[key]).toFixed(3)})`;
  const props = (key) => ({
    className: `bm-part${focus === key ? " on" : ""}`,
    onPointerEnter: () => setHover(key),
    onPointerLeave: () => setHover(null),
    onFocus: () => setHover(key),
    onBlur: () => setHover(null),
    onClick: () => onSelect?.(key),
    onKeyDown: (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect?.(key); }
    },
    role: "button",
    tabIndex: 0,
    "aria-label": `${label(key)}, ${(scores[key] ?? 0).toFixed(1)} out of 10`,
  });

  return (
    <div className="bodymap">
      {/* The viewBox starts above the origin: the travel arrow and its caption sit
          over the figure's head and would be clipped by a 0-based box. */}
      <svg viewBox="0 -14 140 226" role="img" aria-label="Scores mapped onto the body">
        {/* Presence: the space the walk occupies. Drawn first, behind everything. */}
        <g {...props("stage_presence")}>
          <ellipse cx="70" cy="104" rx="56" ry="94" fill="none"
                   stroke={fill("stage_presence")} strokeWidth="1"
                   strokeDasharray="2 4" />
        </g>

        {/* Rhythm: the steps already taken, evenly spaced when the tempo is even. */}
        <g {...props("rhythm")}>
          {[0, 1, 2, 3].map((i) => (
            <rect key={i} x={16 + i * 8} y={190 - i * 2} width="5" height="3" rx="1.5"
                  fill={fill("rhythm")} />
          ))}
          {[0, 1, 2, 3].map((i) => (
            <rect key={`r${i}`} x={112 - i * 8} y={190 - i * 2} width="5" height="3" rx="1.5"
                  fill={fill("rhythm")} />
          ))}
        </g>

        {/* Symmetry: the mirror axis the two sides are compared across. */}
        <g {...props("symmetry")}>
          <line x1="70" y1="12" x2="70" y2="196" stroke={fill("symmetry")}
                strokeWidth="1" strokeDasharray="3 5" />
        </g>

        {/* Balance: what the whole figure stands on. */}
        <g {...props("balance")}>
          <ellipse cx="70" cy="197" rx="34" ry="5" fill="none"
                   stroke={fill("balance")} strokeWidth="2" />
        </g>

        {/* Confidence: forward travel down the runway. */}
        <g {...props("confidence")}>
          <path d="M70 12 v-6 M64 10 l6-6 6 6" fill="none" stroke={fill("confidence")}
                strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
          <text x="70" y="-2" textAnchor="middle" className="bm-tag">toward camera</text>
        </g>

        {/* Head */}
        <g {...props("head_position")}>
          <circle cx="70" cy="30" r="12" fill={fill("head_position")} />
        </g>

        {/* Shoulders */}
        <g {...props("shoulder_alignment")}>
          <rect x="43" y="47" width="54" height="7" rx="3.5"
                fill={fill("shoulder_alignment")} />
        </g>

        {/* Spine and trunk */}
        <g {...props("posture")}>
          <path d="M56 56 h28 l-4 44 h-20 z" fill={fill("posture")} />
        </g>

        {/* Pelvis */}
        <g {...props("hip_movement")}>
          <rect x="54" y="102" width="32" height="12" rx="5" fill={fill("hip_movement")} />
        </g>

        {/* Arms */}
        <g {...props("arm_swing")}>
          <path d="M45 54 l-9 42 l4 1 l11 -41 z" fill={fill("arm_swing")} />
          <path d="M95 54 l9 42 l-4 1 l-11 -41 z" fill={fill("arm_swing")} />
        </g>

        {/* Legs */}
        <g {...props("stride_length")}>
          <path d="M58 114 l-3 56 h8 l3 -56 z" fill={fill("stride_length")} />
          <path d="M82 114 l3 56 h-8 l-3 -56 z" fill={fill("stride_length")} />
        </g>

        {/* Feet */}
        <g {...props("foot_placement")}>
          <rect x="49" y="170" width="16" height="6" rx="3" fill={fill("foot_placement")} />
          <rect x="75" y="170" width="16" height="6" rx="3" fill={fill("foot_placement")} />
        </g>
      </svg>

      <p className="bm-note">
        {focus ? (
          <>
            <strong>{label(focus)}</strong> {(scores[focus] ?? 0).toFixed(1)}/10 ·{" "}
            {REGIONS.find((r) => r.key === focus)?.name}
          </>
        ) : (
          <>Brighter is better. Tap a part of the figure to see what was measured there.</>
        )}
      </p>
    </div>
  );
}
