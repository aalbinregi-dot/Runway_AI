import { useEffect, useState } from "react";
import { api } from "../api.js";

const LABELS = {
  posture: "Posture",
  balance: "Balance",
  confidence: "Confidence",
  shoulder_alignment: "Shoulder Alignment",
  head_position: "Head Position",
  arm_swing: "Arm Swing",
  hip_movement: "Hip Movement",
  rhythm: "Rhythm",
  stride_length: "Stride Length",
  foot_placement: "Foot Placement",
  symmetry: "Symmetry",
  stage_presence: "Stage Presence",
};

/** Sparkline of overall score across sessions. Hand-drawn SVG rather than a
 *  chart library: it is one polyline, and pulling in a charting dependency for
 *  it would outweigh the whole bundle. */
function Trend({ series }) {
  if (!series || series.length < 2) return null;
  const w = 100;
  const h = 30;
  const vals = series.map((s) => s.overall);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const span = Math.max(hi - lo, 1);
  const pts = vals.map((v, i) => {
    const x = (i / (vals.length - 1)) * w;
    const y = h - ((v - lo) / span) * (h - 4) - 2;
    return `${x.toFixed(2)},${y.toFixed(2)}`;
  });
  const last = pts[pts.length - 1].split(",");

  return (
    <svg
      viewBox={`0 0 ${w} ${h}`}
      preserveAspectRatio="none"
      style={{ width: "100%", height: "72px", display: "block", marginTop: "0.9rem" }}
      role="img"
      aria-label={`Overall score across ${vals.length} sessions`}
    >
      <polyline
        points={pts.join(" ")}
        fill="none"
        stroke="rgba(246,245,243,0.85)"
        strokeWidth="0.8"
        vectorEffect="non-scaling-stroke"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx={last[0]} cy={last[1]} r="1.6" fill="var(--ink)" vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

/* Fitted through every walk, not just the endpoints. The endpoint deltas below
   are kept because they are what a user checks by eye, but a delta between two
   noisy numbers is the noisiest possible estimate — see scoring/trend.py. */
function Trends({ trends, overall, minWalks, sessions }) {
  const entries = Object.entries(trends ?? {});
  if (!entries.length) {
    return (
      <section className="glass card">
        <h3>Trend across all walks</h3>
        <p className="muted small">
          Needs at least {minWalks ?? 3} walks to fit a trend — you have {sessions}.
        </p>
      </section>
    );
  }

  const real = entries.filter(([, t]) => t.significant)
                      .sort((a, b) => Math.abs(b[1].total_change) - Math.abs(a[1].total_change));
  const flat = entries.filter(([, t]) => !t.significant);

  return (
    <section className="glass card">
      <h3>Trend across all walks</h3>
      <p className="muted small" style={{ marginTop: "-0.5rem" }}>
        A line fitted through every walk, with a 95% interval. Only changes whose
        interval excludes zero are called real — the rest are inside this system’s
        own measurement error.
      </p>

      {overall && (
        <div className={`trend-overall ${overall.direction}`}>
          <div>
            <strong>Overall {overall.total_change > 0 ? "+" : ""}
            {overall.total_change?.toFixed(1)}</strong>
            <span className="muted tiny">
              {" "}across {overall.n} walks · interval [{overall.ci[0]}, {overall.ci[1]}]
            </span>
          </div>
          <span className="trend-verdict">
            {overall.significant ? overall.direction : "not distinguishable from noise"}
          </span>
        </div>
      )}

      {real.length > 0 && (
        <ul className="trend-list">
          {real.map(([key, t]) => (
            <li key={key}>
              <span className="trend-name">{LABELS[key] ?? key}</span>
              <span className="trend-num">
                {t.total_change > 0 ? "+" : ""}{t.total_change.toFixed(1)}
              </span>
              <span className="muted tiny">
                [{t.ci[0]}, {t.ci[1]}] · {t.direction}
                {t.is_personal_best && " · personal best"}
              </span>
            </li>
          ))}
        </ul>
      )}

      {real.length === 0 && (
        <p className="muted small">
          Nothing yet stands out from measurement error across {sessions} walks.
          That is a statement about the sample size, not about your training —
          each criterion below says how many walks would settle it.
        </p>
      )}

      {flat.length > 0 && (
        <details className="trend-details">
          <summary>{flat.length} criteria with no confirmed trend</summary>
          <ul className="trend-list">
            {flat.map(([key, t]) => (
              <li key={key}>
                <span className="trend-name">{LABELS[key] ?? key}</span>
                <span className="trend-num muted">
                  {t.total_change > 0 ? "+" : ""}{t.total_change.toFixed(1)}
                </span>
                <span className="muted tiny">
                  [{t.ci[0]}, {t.ci[1]}]
                  {t.walks_needed ? ` · ~${t.walks_needed} walks would settle it` : ""}
                </span>
              </li>
            ))}
          </ul>
        </details>
      )}
    </section>
  );
}

export default function Progress({ user }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    if (!user) return;
    let live = true;                     // guard against a stale response landing
    setBusy(true);
    setError(null);
    api
      .progress()
      .then((d) => live && setData(d))
      .catch((e) => live && setError(e.message))
      .finally(() => live && setBusy(false));
    return () => {
      live = false;
    };
  }, [user?.id]);

  if (!user) {
    return (
      <div className="glass card">
        <h3>Progress</h3>
        <p style={{ color: "var(--ink-2)" }}>
          Sign in to keep a history and track improvement across sessions.
          Analysis itself works without an account — it just is not saved.
        </p>
      </div>
    );
  }

  if (busy) {
    return (
      <div className="glass card loading">
        <span className="pulse" />
        <p>Loading history</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="glass card">
        <h3>Progress</h3>
        <p style={{ color: "var(--ink-2)" }}>{error}</p>
      </div>
    );
  }

  if (!data || data.sessions < 2) {
    return (
      <div className="glass card">
        <h3>Progress</h3>
        <p style={{ color: "var(--ink-2)" }}>
          {data?.message ?? "No sessions recorded yet."} Analyse at least two walks
          to see how you are improving.
        </p>
      </div>
    );
  }

  const deltas = Object.entries(data.per_criterion_delta ?? {}).sort((a, b) => b[1] - a[1]);
  const widest = Math.max(...deltas.map(([, v]) => Math.abs(v)), 1);

  return (
    <>
      <section className="glass card">
        <h3>Overall</h3>
        <dl className="stat-row">
          <div className="stat">
            <dt>Sessions</dt>
            <dd>{data.sessions}</dd>
          </div>
          <div className="stat">
            <dt>First</dt>
            <dd>{data.first_overall?.toFixed(1)}</dd>
          </div>
          <div className="stat">
            <dt>Latest</dt>
            <dd>{data.latest_overall?.toFixed(1)}</dd>
          </div>
          <div className="stat">
            <dt>Change</dt>
            <dd>
              {data.overall_delta > 0 ? "+" : ""}
              {data.overall_delta?.toFixed(1)}
            </dd>
          </div>
        </dl>
        <Trend series={data.series} />
      </section>

      <Trends trends={data.trends} overall={data.overall_trend}
              minWalks={data.min_walks_for_trend} sessions={data.sessions} />

      <section className="glass card">
        <h3>Per-criterion change (first vs latest)</h3>
        <div className="delta">
          {deltas.map(([key, v]) => (
            <div key={key}>
              <div className="delta-row">
                <span>{LABELS[key] ?? key}</span>
                <span
                  style={{
                    fontFamily: "var(--mono)",
                    fontVariantNumeric: "tabular-nums",
                    textAlign: "right",
                    color: v === 0 ? "var(--ink-3)" : "var(--ink)",
                  }}
                >
                  {v > 0 ? "+" : ""}
                  {v.toFixed(1)}
                </span>
              </div>
              <div className="delta-bar">
                {/* Centre line at 50%: gains extend right, regressions left. */}
                <i
                  style={{
                    left: v >= 0 ? "50%" : `${50 - (Math.abs(v) / widest) * 50}%`,
                    width: `${(Math.abs(v) / widest) * 50}%`,
                    opacity: v === 0 ? 0.25 : 1,
                  }}
                />
              </div>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
