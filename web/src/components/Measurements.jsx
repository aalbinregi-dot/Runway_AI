/**
 * The measurements themselves, searchable.
 *
 * A hundred and two features are computed for every walk and twelve numbers were
 * shown. The rest were not secret, they were simply unreachable — which meant a
 * user who wanted to know their cadence, or whether their left arm swings further
 * than their right, had no way to find out from a page built entirely out of those
 * measurements.
 *
 * These are the ones a person can act on or sanity-check, grouped as the server
 * groups them. Rows the rubric actually scores are marked and link through to the
 * band that scores them; the rest are context, and saying which is which is the
 * difference between a number to chase and a number to read.
 */
import { useMemo, useState } from "react";
import { label } from "../criteria.js";
import { withUnit } from "../format.js";

export default function Measurements({ measurements, onSelect }) {
  const [query, setQuery] = useState("");
  const [scoredOnly, setScoredOnly] = useState(false);

  const groups = useMemo(() => {
    const q = query.trim().toLowerCase();
    return (measurements ?? [])
      .map((g) => ({
        ...g,
        rows: g.rows.filter((r) =>
          (!scoredOnly || r.scored_in)
          && (!q || r.label.toLowerCase().includes(q) || r.feature.includes(q)
              || (r.scored_in ?? "").includes(q))),
      }))
      .filter((g) => g.rows.length);
  }, [measurements, query, scoredOnly]);

  if (!measurements?.length) return null;
  const total = measurements.reduce((n, g) => n + g.rows.length, 0);
  const shown = groups.reduce((n, g) => n + g.rows.length, 0);

  return (
    <section className="glass card">
      <h3>Measurements</h3>

      <div className="meas-controls">
        <input
          className="field-input"
          type="search"
          placeholder="Filter — cadence, sway, hip, symmetry…"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          aria-label="Filter measurements"
        />
        <button
          className={`chip clip-pick${scoredOnly ? " on" : ""}`}
          aria-pressed={scoredOnly}
          onClick={() => setScoredOnly((s) => !s)}
        >
          scored only
        </button>
      </div>

      {groups.map((group) => (
        <details key={group.key} className="meas-group" open={!!query || groups.length <= 3}>
          <summary>
            {group.title}
            <span className="meas-count">{group.rows.length}</span>
          </summary>
          <div className="meas-rows">
            {group.rows.map((row) => (
              <div className="meas-row" key={row.feature}>
                <span className="meas-label" title={row.feature}>{row.label}</span>
                <span className="meas-value">{withUnit(row.value, row.unit)}</span>
                {row.scored_in ? (
                  <button className="meas-link" onClick={() => onSelect?.(row.scored_in)}>
                    {label(row.scored_in)}
                  </button>
                ) : (
                  <span className="meas-context">context</span>
                )}
              </div>
            ))}
          </div>
        </details>
      ))}

      {shown === 0 && <p className="muted tiny">Nothing matches “{query}”.</p>}

      <p className="live-caveat">
        Showing {shown} of {total}. Lengths are in torso lengths and angles in
        degrees, so they compare across subjects, camera distances and
        resolutions — not in centimetres, which a single camera cannot recover
        without knowing its own focal length. Rows marked with a criterion are read
        by the rubric; the others explain them.
      </p>
    </section>
  );
}
