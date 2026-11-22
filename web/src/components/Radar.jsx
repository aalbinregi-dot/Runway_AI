/**
 * The twelve criteria as one shape.
 *
 * A sorted list of twelve numbers answers "what is my worst score". It does not
 * answer "what shape is my walk", and that is the question a coach asks first:
 * a walk that is flat at 6 everywhere needs a different session from one that is
 * 9 on posture and 2 on rhythm, even though both average the same.
 *
 * Interactive because the shape is a way in, not a decoration. Clicking an axis
 * opens that criterion's band-by-band breakdown; hovering names it and its
 * score. Axes whose criterion is only weakly identifiable from pose are drawn
 * dashed, so the polygon cannot imply that all twelve vertices are equally
 * trustworthy — the identifiability audit says plainly that they are not.
 */
import { useState } from "react";
import { ORDER, label, short } from "../criteria.js";

const R = 86;               // outer radius, in view units
const RINGS = [2, 4, 6, 8, 10];

/** Angle of the i-th axis, in radians, starting at twelve o'clock. */
const angleOf = (i, n) => (i / n) * Math.PI * 2 - Math.PI / 2;

const point = (i, n, radius) => [
  Math.cos(angleOf(i, n)) * radius,
  Math.sin(angleOf(i, n)) * radius,
];

export default function Radar({ scores, selected, onSelect, confidence, reference,
                                referenceLabel = "previous" }) {
  const [hover, setHover] = useState(null);
  const keys = ORDER.filter((k) => k in (scores ?? {}));
  if (keys.length < 3) return null;

  const n = keys.length;
  const per = confidence?.available ? confidence.per_criterion : null;
  const polygon = (source) => keys
    .map((k, i) => point(i, n, (Math.max(0, Math.min(10, source[k] ?? 0)) / 10) * R))
    .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`)
    .join(" ");

  const focus = hover ?? selected;

  return (
    <div className="radar">
      <svg viewBox="-118 -112 236 224" role="img"
           aria-label="The twelve criterion scores as a radar chart">
        {RINGS.map((ring) => (
          <polygon
            key={ring}
            points={keys.map((_, i) => point(i, n, (ring / 10) * R))
              .map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ")}
            fill="none"
            stroke={ring === 10 ? "rgba(246,245,243,0.22)" : "rgba(246,245,243,0.09)"}
            strokeWidth="0.8"
          />
        ))}

        {keys.map((k, i) => {
          const [x, y] = point(i, n, R);
          const weak = per?.[k]?.tier === "weak";
          return (
            <line key={`spoke-${k}`} x1="0" y1="0" x2={x} y2={y}
                  stroke={focus === k ? "rgba(246,245,243,0.5)" : "rgba(246,245,243,0.13)"}
                  strokeWidth="0.8"
                  strokeDasharray={weak ? "2 3" : undefined} />
          );
        })}

        {reference && (
          <polygon points={polygon(reference)} fill="none"
                   stroke="rgba(246,245,243,0.34)" strokeWidth="1"
                   strokeDasharray="3 2" />
        )}

        <polygon points={polygon(scores)} fill="rgba(246,245,243,0.13)"
                 stroke="rgba(246,245,243,0.85)" strokeWidth="1.4"
                 strokeLinejoin="round" />

        {keys.map((k, i) => {
          const value = Math.max(0, Math.min(10, scores[k] ?? 0));
          const [x, y] = point(i, n, (value / 10) * R);
          const on = focus === k;
          return (
            <circle key={`dot-${k}`} cx={x} cy={y} r={on ? 3.4 : 2}
                    fill={on ? "var(--ink)" : "rgba(246,245,243,0.7)"} />
          );
        })}

        {/* Hit targets and labels last, so they take the pointer. A wide
            transparent spoke is a far easier target than the label alone,
            especially with a thumb. */}
        {keys.map((k, i) => {
          const [lx, ly] = point(i, n, R + 17);
          const [hx, hy] = point(i, n, R);
          const anchor = Math.abs(lx) < 12 ? "middle" : lx > 0 ? "start" : "end";
          const on = focus === k;
          const value = scores[k] ?? 0;
          return (
            <g key={`hit-${k}`}
               className={`radar-hit${on ? " on" : ""}`}
               role="button"
               tabIndex={0}
               aria-label={`${label(k)}, ${value.toFixed(1)} out of 10`}
               onClick={() => onSelect?.(k)}
               onKeyDown={(e) => {
                 if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect?.(k); }
               }}
               onPointerEnter={() => setHover(k)}
               onPointerLeave={() => setHover(null)}
               onFocus={() => setHover(k)}
               onBlur={() => setHover(null)}>
              <line x1="0" y1="0" x2={hx} y2={hy} stroke="transparent" strokeWidth="14" />
              <text x={lx} y={ly + 3} textAnchor={anchor} className="radar-label">
                {short(k)}
              </text>
              <text x={lx} y={ly + 12} textAnchor={anchor} className="radar-num">
                {value.toFixed(1)}
              </text>
            </g>
          );
        })}
      </svg>

      <p className="radar-note">
        {focus ? (
          <>
            <strong>{label(focus)}</strong> {(scores[focus] ?? 0).toFixed(1)}/10
            {per?.[focus]?.tier && per[focus].tier !== "unknown"
              ? ` · ${per[focus].tier}ly identifiable from pose`
              : ""}
            {" — click for the measurements behind it"}
          </>
        ) : (
          <>
            Tap an axis for the measurements behind it. Dashed spokes are criteria
            only weakly identifiable from pose alone
            {reference ? `; the dashed outline is the ${referenceLabel} walk` : ""}.
          </>
        )}
      </p>
    </div>
  );
}
