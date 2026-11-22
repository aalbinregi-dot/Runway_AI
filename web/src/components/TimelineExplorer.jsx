/**
 * The evidence strip under the player, as an instrument rather than a picture.
 *
 * The old strip drew one trace at a time on an unlabelled axis with tick marks
 * for the gait cycles. It proved the measurements existed; it did not let anyone
 * interrogate them. There was no way to see whether 9 degrees of lean was a lot,
 * no way to compare two traces that plainly move together, no way to look
 * closely at the two seconds where it all went wrong, and no way to read a value
 * off the chart at the frame you were looking at.
 *
 * So this version adds the four things that turn a chart into an instrument:
 *
 * - **A target.** The shaded band is the rubric's own tolerance for the trace,
 *   read from the server's `reference` block, so the reader can see in-tolerance
 *   from out without knowing any of the thresholds.
 * - **Comparison.** Up to three traces at once. Shoulder tilt and hip drop
 *   moving in antiphase is a fact about the walk that no single trace shows.
 * - **Zoom.** Drag to inspect a range; the whole strip, foot events included,
 *   follows.
 * - **A readout.** Hover or scrub and every visible trace reports its value at
 *   that instant, next to the frame the player is showing.
 *
 * Everything is derived from the payload `features.timeline` already sent. No
 * charting dependency: one polyline per trace costs a few lines and a library
 * would outweigh the entire bundle.
 */
import { useEffect, useMemo, useRef, useState } from "react";
import { tight, withUnit } from "../format.js";

/** Plot geometry, in the SVG's own user units. */
const VB = { w: 100, h: 34, pad: 3 };

/** How many traces can be shown at once. Four is legible on a laptop and a mess
 *  on a phone; three is the compromise, and the primary one carries the shading. */
const MAX_ACTIVE = 3;

const STROKE = [
  "rgba(246,245,243,0.95)",
  "rgba(246,245,243,0.55)",
  "rgba(246,245,243,0.34)",
];
const DASH = [null, "3 2", "1 2"];

/** A drag shorter than this (as a fraction of the width) is a click, not a
 *  brush — otherwise every click-to-seek zooms to a two-pixel window. */
const DRAG_MIN = 0.02;

export default function TimelineExplorer({ timeline, current = 0, onSeek, onSelect }) {
  const names = useMemo(() => Object.keys(timeline?.traces ?? {}), [timeline]);
  const evidence = timeline?.evidence ?? [];

  // Open on the trace that explains the most severe finding: the first thing
  // shown should be the thing that cost the most marks, not whichever trace
  // happens to sort first.
  const [active, setActive] = useState(() => {
    const first = evidence[0]?.trace;
    return first && names.includes(first) ? [first] : names.slice(0, 1);
  });
  const [zoom, setZoom] = useState(null);          // [from, to] in seconds
  const [hoverT, setHoverT] = useState(null);
  const [drag, setDrag] = useState(null);          // [fromFrac, toFrac] while brushing
  const plotRef = useRef(null);
  /** Has the playhead been inside the current zoom yet? See the reset effect. */
  const enteredRef = useRef(true);

  const duration = timeline?.duration_s || 1;
  const [from, to] = zoom ?? [0, duration];
  const span = Math.max(to - from, 1e-3);

  // Index range of the visible window, computed once per zoom change rather than
  // per trace: `times` is shared by every trace.
  const window_ = useMemo(() => {
    const times = timeline?.times ?? [];
    let lo = 0;
    let hi = times.length - 1;
    while (lo < hi && times[lo] < from) lo += 1;
    while (hi > lo && times[hi] > to) hi -= 1;
    return { lo: Math.max(0, lo - 1), hi: Math.min(times.length - 1, hi + 1), times };
  }, [timeline, from, to]);

  const series = useMemo(
    () => active
      .filter((name) => timeline?.traces?.[name])
      .map((name, i) => build(timeline.traces[name], window_, i === 0)),
    [active, timeline, window_],
  );

  useEffect(() => {
    // A zoom the playhead has *left* is disorienting: the head vanishes and the
    // chart looks frozen while the video plays on. But a zoom the playhead has
    // not entered yet is the normal case at the instant of zooming — you drag
    // across seconds three to five while the video sits at zero — and resetting
    // then made drag-to-zoom appear not to work at all, which is how this was
    // found. So the reset waits until the playhead has actually been inside.
    if (!zoom) {
      enteredRef.current = true;
      return;
    }
    const inside = current >= zoom[0] - 0.05 && current <= zoom[1] + 0.05;
    if (inside) enteredRef.current = true;
    else if (enteredRef.current) setZoom(null);
  }, [current, zoom]);

  if (!timeline || !names.length) return null;

  const xOf = (t) => VB.pad + ((t - from) / span) * (VB.w - 2 * VB.pad);
  const leftOf = (t) => `${Math.max(0, Math.min(100, ((t - from) / span) * 100))}%`;
  const inWindow = (t) => t >= from - 1e-6 && t <= to + 1e-6;

  const fracFrom = (e) => {
    const rect = plotRef.current.getBoundingClientRect();
    return Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
  };
  const timeFrom = (e) => from + fracFrom(e) * span;

  function toggle(name) {
    setActive((cur) => {
      if (cur.includes(name)) return cur.length === 1 ? cur : cur.filter((n) => n !== name);
      return [name, ...cur].slice(0, MAX_ACTIVE);
    });
  }

  /* Pointer events rather than mouse events, so a drag works the same with a
     finger on the phone this is most likely to be filmed on. */
  function onPointerDown(e) {
    e.currentTarget.setPointerCapture?.(e.pointerId);
    const f = fracFrom(e);
    setDrag([f, f]);
  }

  function onPointerMove(e) {
    setHoverT(timeFrom(e));
    if (drag) setDrag([drag[0], fracFrom(e)]);
  }

  function onPointerUp(e) {
    const f = fracFrom(e);
    if (drag && Math.abs(f - drag[0]) >= DRAG_MIN) {
      const a = from + Math.min(drag[0], f) * span;
      const b = from + Math.max(drag[0], f) * span;
      setZoom([a, b]);
      enteredRef.current = false;
      // Zooming is navigation: if the playhead is outside the range just chosen,
      // take the video there rather than leaving the head off-screen.
      if (current < a || current > b) onSeek?.(a);
    } else {
      onSeek?.(from + f * span);
    }
    setDrag(null);
  }

  function onKeyDown(e) {
    const step = e.shiftKey ? 1 : 0.1;
    if (e.key === "ArrowRight") { onSeek?.(Math.min(duration, current + step)); }
    else if (e.key === "ArrowLeft") { onSeek?.(Math.max(0, current - step)); }
    else if (e.key === "Home") { onSeek?.(from); }
    else if (e.key === "End") { onSeek?.(to); }
    else if (e.key === "Escape") { setZoom(null); }
    else return;
    e.preventDefault();
  }

  const readAt = hoverT ?? current;
  const primary = series[0];
  const cycles = (timeline.cycles ?? []).filter(inWindow);
  const worst = evidence.filter((e) => active.includes(e.trace));

  return (
    <div className="tlx">
      <div className="tlx-tabs" role="group" aria-label="Measurement traces">
        {names.map((n) => {
          const i = active.indexOf(n);
          const flagged = evidence.some((e) => e.trace === n);
          return (
            <button
              key={n}
              className={`tl-tab${i >= 0 ? " on" : ""}${i > 0 ? " second" : ""}`}
              aria-pressed={i >= 0}
              onClick={() => toggle(n)}
              title={i >= 0 ? "Hide this trace" : "Add this trace"}
            >
              {n}
              {flagged && <span className="tlx-flag" title="A reported fault came from this trace">·</span>}
            </button>
          );
        })}
      </div>

      <div
        className="tlx-plot"
        ref={plotRef}
        tabIndex={0}
        role="slider"
        aria-label="Measurement timeline; arrow keys move the playhead"
        aria-valuemin={0}
        aria-valuemax={duration}
        aria-valuenow={Number(current.toFixed(2))}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        /* Only the crosshair is cleared on leave. Clearing the brush here as well
           cancelled any drag that ran past the edge of the plot — which is most
           of them, since the interesting range usually starts at one end. */
        onPointerLeave={() => setHoverT(null)}
        onPointerCancel={() => setDrag(null)}
        onLostPointerCapture={() => setDrag(null)}
        onKeyDown={onKeyDown}
      >
        <svg viewBox={`0 0 ${VB.w} ${VB.h}`} preserveAspectRatio="none" aria-hidden="true">
          {/* The rubric's acceptable region for the primary trace. Drawn first so
              every line sits over it. */}
          {primary?.band && (
            <rect x="0" y={primary.band.y} width={VB.w} height={primary.band.h}
                  fill="rgba(246,245,243,0.09)" />
          )}
          {primary?.band?.edge != null && (
            <line x1="0" y1={primary.band.edge} x2={VB.w} y2={primary.band.edge}
                  stroke="rgba(246,245,243,0.4)" strokeWidth="0.4" strokeDasharray="2 2"
                  vectorEffect="non-scaling-stroke" />
          )}

          {/* Gait cycles as alternating shading: the segmentation every timing
              score rests on, read as bands rather than counted as ticks. */}
          {cycles.map((t, i) => {
            const next = cycles[i + 1] ?? to;
            return i % 2 === 0 ? (
              <rect key={`cy${i}`} x={xOf(t)} y="0" width={Math.max(0, xOf(next) - xOf(t))}
                    height={VB.h} fill="rgba(255,255,255,0.028)" />
            ) : null;
          })}

          {series.map((s, i) => (
            <polyline key={s.name} points={s.points} fill="none" stroke={STROKE[i]}
                      strokeWidth={i === 0 ? 0.9 : 0.7}
                      strokeDasharray={DASH[i] ?? undefined}
                      vectorEffect="non-scaling-stroke" strokeLinejoin="round" />
          ))}

          {drag && (
            <rect x={Math.min(drag[0], drag[1]) * VB.w} y="0"
                  width={Math.abs(drag[1] - drag[0]) * VB.w} height={VB.h}
                  fill="rgba(246,245,243,0.12)" />
          )}
        </svg>

        {/* Worst-moment markers for the findings the visible traces produced. */}
        {worst.filter((e) => inWindow(e.t)).map((e, i) => (
          <button
            key={`ev${i}`}
            className="tlx-evidence"
            style={{ left: leftOf(e.t) }}
            title={`${e.trace} peaks at ${e.t.toFixed(2)}s — ${withUnit(e.value, e.unit)}`}
            onPointerDown={(ev) => ev.stopPropagation()}
            onClick={(ev) => { ev.stopPropagation(); onSeek?.(e.t); onSelect?.(e.criterion); }}
            aria-label={`Jump to ${e.trace} peak at ${e.t.toFixed(1)} seconds`}
          />
        ))}

        {hoverT != null && (
          <span className="tlx-hover" style={{ left: leftOf(hoverT) }} />
        )}
        {inWindow(current) && (
          <span className="tl-head" style={{ left: leftOf(current) }} />
        )}
      </div>

      {/* Foot events, following the same window as the plot. */}
      <div className="tl-feet">
        {["left", "right"].map((side) => (
          <div className="tl-foot" key={side}>
            <span className="tl-foot-label">{side[0].toUpperCase()}</span>
            <div className="tl-foot-track">
              {(timeline.strikes?.[side] ?? []).filter(inWindow).map((t, i) => (
                <button
                  key={i}
                  className={`tl-strike ${side}`}
                  style={{ left: leftOf(t) }}
                  title={`${side} foot event at ${t.toFixed(2)}s — click to jump`}
                  onClick={() => onSeek?.(t)}
                  aria-label={`Jump to ${side} foot event at ${t.toFixed(1)} seconds`}
                />
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* The readout. Values at the hovered instant if hovering, at the playhead
          otherwise, so the numbers always describe the frame being looked at. */}
      <div className="tlx-read">
        <span className="tlx-time">
          {readAt.toFixed(2)}s
          <small>{hoverT != null ? "cursor" : "playhead"}</small>
        </span>
        {series.map((s, i) => (
          <button
            key={s.name}
            className="tlx-value"
            onClick={() => onSelect?.(s.reference?.criterion)}
            disabled={!s.reference?.criterion}
            title={s.reference
              ? `Scored under ${s.reference.criterion.replace(/_/g, " ")} — click to open`
              : "No single band scores this trace"}
          >
            <span className="tlx-swatch" style={{ opacity: 1 - i * 0.3 }} />
            <span className="tlx-value-name">{s.name}</span>
            <strong>{withUnit(s.at(readAt), s.unit)}</strong>
            {/* Keyed on the reference, not on the shading: only the primary trace
                gets its tolerance drawn, but every trace that has one can say
                what it is. */}
            <span className="tlx-range">
              {tight(s.stats.min, s.unit)}–{tight(s.stats.max, s.unit)}
              {s.targetText ? ` · target ${s.targetText}` : ""}
            </span>
          </button>
        ))}
      </div>

      <div className="tl-legend">
        <span>{(timeline.cycles ?? []).length} gait cycles</span>
        <span>
          {zoom
            ? `${from.toFixed(2)}–${to.toFixed(2)}s of ${duration.toFixed(1)}s`
            : `${duration.toFixed(1)}s · drag to zoom, click to seek`}
        </span>
        {zoom && (
          <button className="tlx-reset" onClick={() => setZoom(null)}>reset zoom</button>
        )}
        {worst.length > 0 && (
          <button className="tlx-reset"
                  onClick={() => { onSeek?.(worst[0].t); onSelect?.(worst[0].criterion); }}>
            worst moment
          </button>
        )}
      </div>
    </div>
  );
}

/**
 * Prepare one trace for drawing: a polyline in view units, its own value range,
 * where its tolerance band falls, and a value lookup by time.
 *
 * Each trace is scaled to its own range rather than to a shared axis. That is a
 * deliberate compromise: degrees and torso lengths have no common scale, and the
 * question these overlays answer is "do these two move together", which is about
 * shape. The legend prints each range so nobody reads a crossing as equality.
 */
function build(trace, window_, isPrimary) {
  const { lo, hi, times } = window_;
  const values = trace.values ?? [];
  const view = values.slice(lo, hi + 1);
  const unit = trace.unit ?? "";
  const reference = trace.reference ?? null;

  let min = Math.min(...view);
  let max = Math.max(...view);
  // Include the tolerance edge, or a band drawn outside the plot would silently
  // disappear and an out-of-tolerance trace would look centred.
  if (isPrimary && reference) {
    for (const edge of reference.acceptable ?? []) {
      if (edge != null) { min = Math.min(min, edge); max = Math.max(max, edge); }
    }
  }
  if (!(max > min)) { max = min + 1e-6; }
  const pad = (max - min) * 0.12;
  min -= pad;
  max += pad;

  const yOf = (v) => VB.h - VB.pad - ((v - min) / (max - min)) * (VB.h - 2 * VB.pad);
  const xOf = (t) => {
    const t0 = times[lo];
    const t1 = times[hi];
    return VB.pad + ((t - t0) / Math.max(t1 - t0, 1e-6)) * (VB.w - 2 * VB.pad);
  };

  const points = view
    .map((v, i) => `${xOf(times[lo + i]).toFixed(2)},${yOf(v).toFixed(2)}`)
    .join(" ");

  let band = null;
  if (isPrimary && reference) {
    const [aLo, aHi] = reference.acceptable ?? [null, null];
    const yTop = yOf(aHi != null ? aHi : max);
    const yBottom = yOf(aLo != null ? aLo : min);
    band = {
      y: Math.min(yTop, yBottom),
      h: Math.abs(yBottom - yTop),
      edge: aHi != null ? yOf(aHi) : aLo != null ? yOf(aLo) : null,
    };
  }

  return {
    name: trace.label,
    unit,
    reference,
    points,
    band,
    stats: trace.stats ?? { min: Math.min(...view), max: Math.max(...view) },
    targetText: reference
      ? reference.mode === "lower"
        ? `under ${withUnit(reference.ideal + reference.tol, unit)}`
        : reference.mode === "upper"
          ? `over ${withUnit(reference.ideal - reference.tol, unit)}`
          : `${withUnit(reference.ideal, unit)} ± ${withUnit(reference.tol, unit)}`
      : null,
    /** Nearest sampled value to a time — the traces are sampled at up to 400
     *  points, so interpolating would imply a resolution they do not have. */
    at(t) {
      if (!times.length) return NaN;
      let best = 0;
      let bestGap = Infinity;
      for (let i = 0; i < times.length; i += 1) {
        const gap = Math.abs(times[i] - t);
        if (gap < bestGap) { bestGap = gap; best = i; }
      }
      return values[best];
    },
  };
}
