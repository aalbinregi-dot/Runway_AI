/**
 * Where the hundred marks went.
 *
 * The overall score is a weighted sum, and the weights are not equal: posture is
 * worth 13 marks, stage presence 5. So the twelve 0-10 scores, drawn as twelve
 * identical bars, actively mislead — a 5/10 on posture and a 5/10 on presence
 * look like the same problem and cost 6.5 and 2.5 marks respectively.
 *
 * This ranks the criteria by *marks lost*, which is the only order that answers
 * "what should I work on first". The bar shows what was earned against what was
 * available, so the two quantities that matter — how bad it was, and how much it
 * was worth — are visible in one shape.
 */
import { label } from "../criteria.js";
import { tier } from "../api.js";

export default function Contribution({ breakdown, overall, selected, onSelect }) {
  if (!breakdown) return null;
  const rows = Object.entries(breakdown)
    .map(([key, e]) => ({ key, ...e }))
    .sort((a, b) => b.marks_lost - a.marks_lost);
  if (!rows.length) return null;

  const widest = Math.max(...rows.map((r) => r.marks_available));
  const lost = rows.reduce((sum, r) => sum + r.marks_lost, 0);
  const top = rows.slice(0, 3).filter((r) => r.marks_lost > 0.5);

  return (
    <section className="glass card">
      <h3>Where the marks went</h3>
      <p className="lede" style={{ marginBottom: "1rem", color: "var(--ink-2)" }}>
        {overall != null && <>{overall.toFixed(1)} earned, {lost.toFixed(1)} lost. </>}
        {top.length > 0 && (
          <>
            {top.map((r) => label(r.key)).join(", ")} account for{" "}
            {top.reduce((s, r) => s + r.marks_lost, 0).toFixed(1)} of them.
          </>
        )}
      </p>

      <div className="marks">
        {rows.map((r) => (
          <button
            key={r.key}
            className={`marks-row q-${tier(r.score)}${selected === r.key ? " on" : ""}`}
            onClick={() => onSelect?.(r.key)}
            aria-label={`${label(r.key)} earned ${r.marks_earned} of ${r.marks_available} marks`}
          >
            <span className="marks-name">{label(r.key)}</span>
            {/* A criterion that lost nothing showed "−0.0", which reads as a
                rounding artefact rather than a full mark. */}
            <span className="marks-num">
              {r.marks_lost < 0.05 ? "full marks" : `−${r.marks_lost.toFixed(1)}`}
            </span>
            <span className="marks-track"
                  style={{ width: `${(r.marks_available / widest) * 100}%` }}>
              <i style={{ width: `${(r.marks_earned / r.marks_available) * 100}%` }} />
            </span>
            <span className="marks-weight">{r.marks_available.toFixed(0)} available</span>
          </button>
        ))}
      </div>

      <p className="live-caveat">
        Bar length is what the criterion is worth; the filled part is what it
        earned. The twelve weights are the server's, fetched with the rubric, so
        this cannot drift from how the overall score is actually computed.
      </p>
    </section>
  );
}
