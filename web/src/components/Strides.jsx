/**
 * Cycle-to-cycle consistency: every stride the clip contained, side by side.
 *
 * The rhythm score is a coefficient of variation — one number for the whole walk.
 * It is the right number, and it is unreadable: 0.09 means nothing to a person
 * who has never seen 0.03. Drawing the strides themselves makes the same fact
 * obvious. Four bars of equal height are a steady tempo; one long bar among short
 * ones is the stride where something happened, and it can be jumped to.
 *
 * Everything here is derived from events the timeline already draws — the foot
 * marks under the trace — so the bars and the marks cannot disagree. No CV is
 * computed here on purpose: the rhythm score's CV comes from each ankle's own
 * Hilbert phase in the feature builder, and a second one computed from these
 * events would differ slightly and read as a contradiction. The spread is shown
 * as what it is, a range in seconds.
 */
import { useState } from "react";

const SIDES = [
  { id: "left", name: "Left", colour: "#f0844c" },
  { id: "right", name: "Right", colour: "#4ea8de" },
];

export default function Strides({ timeline, onSeek, current = 0 }) {
  const [hover, setHover] = useState(null);
  const strides = timeline?.strides;
  if (!strides) return null;

  const all = SIDES.flatMap((s) => (strides[s.id] ?? []).map((x) => ({ ...x, side: s.id })));
  if (all.length < 2) {
    return (
      <section className="glass card">
        <h3>Stride consistency</h3>
        <p className="lede" style={{ color: "var(--ink-2)" }}>
          {all.length === 1
            ? "Only one complete stride was captured, so there is nothing to compare it with."
            : "No complete strides were detected in this clip."}{" "}
          Four or more strides — six to ten seconds of walking — gives a stable reading.
        </p>
      </section>
    );
  }

  const longest = Math.max(...all.map((s) => s.duration));
  const mean = strides.mean_s ?? all.reduce((a, s) => a + s.duration, 0) / all.length;
  // The lane shares the clip's time axis with the traces above it, so a stride's
  // position means something. Falling back to the last event keeps the geometry
  // sane if a payload ever arrives without a duration.
  const duration = timeline.duration_s
    || Math.max(...all.map((s) => s.t + s.duration), 1e-6);
  const meanOf = (side) => {
    const list = strides[side] ?? [];
    return list.length ? list.reduce((a, s) => a + s.duration, 0) / list.length : null;
  };
  const leftMean = meanOf("left");
  const rightMean = meanOf("right");
  const gap = leftMean != null && rightMean != null ? leftMean - rightMean : null;

  // Phase: how far into the left stride the right foot's event falls. Alternating
  // gait puts it at half a cycle; a limp pushes it away from 0.5. Derived from the
  // events drawn below, and labelled as such — `sym_ankle_phase_offset` in the
  // measurements table is the pipeline's own version of this from the full clip.
  const phase = phaseOffset(timeline.strikes?.left ?? [], timeline.strikes?.right ?? []);

  return (
    <section className="glass card">
      <h3>Stride consistency</h3>

      <div className="stride-stats">
        <div className="stat">
          <dt>Strides</dt>
          <dd>{all.length}</dd>
        </div>
        <div className="stat">
          <dt>Mean stride</dt>
          <dd>{mean.toFixed(2)}<small>s</small></dd>
        </div>
        <div className="stat">
          <dt>Spread</dt>
          <dd>{(strides.spread_s ?? (longest - Math.min(...all.map((s) => s.duration))))
            .toFixed(2)}<small>s</small></dd>
        </div>
        <div className="stat">
          <dt>L − R</dt>
          <dd>{gap == null ? "—" : `${gap >= 0 ? "+" : ""}${gap.toFixed(2)}`}<small>s</small></dd>
        </div>
        <div className="stat">
          <dt>Phase</dt>
          <dd>{phase == null ? "—" : phase.toFixed(2)}</dd>
        </div>
      </div>

      {SIDES.map((side) => {
        const list = strides[side.id] ?? [];
        return (
          <div className="stride-lane" key={side.id}>
            <span className="stride-side">{side.name}</span>
            <div className="stride-bars">
              {list.length === 0 && <span className="stride-empty">no complete strides</span>}
              {list.map((s, i) => {
                const on = current >= s.t && current < s.t + s.duration;
                const key = `${side.id}-${i}`;
                return (
                  <button
                    key={key}
                    className={`stride-bar${on ? " on" : ""}`}
                    // Placed at the time it happened, not packed left to right.
                    // Three bars in a flex row left a lane two-thirds empty and
                    // said nothing about *when*; positioned in time they tile the
                    // clip and line up with the traces above, so a long stride can
                    // be read against the moment it happened.
                    style={{
                      left: `${(s.t / duration) * 100}%`,
                      width: `${(s.duration / duration) * 100}%`,
                      height: `${Math.max(12, (s.duration / longest) * 100)}%`,
                      background: side.colour,
                      opacity: hover && hover !== key ? 0.45 : 1,
                    }}
                    onPointerEnter={() => setHover(key)}
                    onPointerLeave={() => setHover(null)}
                    onClick={() => onSeek?.(s.t)}
                    title={`${side.name} stride ${i + 1}: ${s.duration.toFixed(2)}s at ${s.t.toFixed(2)}s`}
                    aria-label={`${side.name} stride ${i + 1}, ${s.duration.toFixed(2)} seconds, jump to it`}
                  />
                );
              })}
              {/* The mean, as a line to read the bars against. */}
              <span className="stride-mean" style={{ bottom: `${(mean / longest) * 100}%` }} />
              {current > 0 && current <= duration && (
                <span className="stride-head" style={{ left: `${(current / duration) * 100}%` }} />
              )}
            </div>
          </div>
        );
      })}

      <p className="live-caveat">
        One bar per completed stride, placed at the moment it happened and drawn
        against the mean — so the lane shares the time axis with the traces above.
        Click a bar to jump to it. Phase is where the right foot's event falls
        inside the left foot's stride — alternating gait sits near 0.50.{" "}
        {strides.basis}.
      </p>
    </section>
  );
}

/** Mean fraction of a left stride at which the next right event happens. */
function phaseOffset(left, right) {
  if (left.length < 2 || right.length < 1) return null;
  const fractions = [];
  for (let i = 0; i < left.length - 1; i += 1) {
    const period = left[i + 1] - left[i];
    if (period <= 0) continue;
    const next = right.find((t) => t >= left[i] && t < left[i + 1]);
    if (next != null) fractions.push((next - left[i]) / period);
  }
  if (!fractions.length) return null;
  return fractions.reduce((a, b) => a + b, 0) / fractions.length;
}
