/**
 * Saved walks.
 *
 * There are no thumbnails, because no video is kept — the project stores
 * keypoints, not footage. Each row therefore leads with what was actually
 * measured: the score, the grade, and how long the walk was.
 */
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { SignInRequired } from "../Shell.jsx";

const fmtDate = (s) =>
  new Date(s).toLocaleDateString(undefined,
    { day: "numeric", month: "short", year: "numeric" });

export default function History({ user }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [picked, setPicked] = useState([]);
  const navigate = useNavigate();

  useEffect(() => {
    if (!user) return;
    api.walks().then(setData).catch((e) => setError(e.message));
  }, [user]);

  if (!user) return <SignInRequired what="Your walks" />;
  if (error) return <div className="glass card"><h3>History</h3><p>{error}</p></div>;
  if (!data) {
    return <div className="glass card loading"><span className="pulse" /><p>Loading</p></div>;
  }

  if (!data.count) {
    return (
      <div className="glass card">
        <h3>No walks yet</h3>
        <p style={{ color: "var(--ink-2)" }}>
          Analyse a walk while signed in and it will be saved here.
        </p>
        <Link to="/" className="btn solid"
              style={{ marginTop: "1rem", display: "inline-block", textDecoration: "none" }}>
          Analyse a walk
        </Link>
      </div>
    );
  }

  // At most two selected: a third replaces the oldest, which is less annoying
  // than refusing the click.
  const toggle = (id) =>
    setPicked((p) => (p.includes(id) ? p.filter((x) => x !== id) : [...p, id].slice(-2)));

  return (
    <>
      <section className="glass card">
        <h3>Your walks</h3>
        <dl className="stat-row">
          <div className="stat"><dt>Saved</dt><dd>{data.count}</dd></div>
          <div className="stat"><dt>Best</dt><dd>{data.best?.toFixed(1)}</dd></div>
          <div className="stat"><dt>Average</dt><dd>{data.average?.toFixed(1)}</dd></div>
        </dl>
      </section>

      <section className="glass card">
        <h3>
          {picked.length === 2 ? "Two selected"
            : picked.length === 1 ? "Pick one more to compare" : "History"}
        </h3>

        <div className="walks">
          {data.walks.map((w) => {
            const on = picked.includes(w.session_id);
            return (
              <div className={`walk${on ? " on" : ""}`} key={w.session_id}>
                <button className="walk-pick" onClick={() => toggle(w.session_id)}
                        aria-pressed={on}
                        aria-label={on ? "Deselect walk" : "Select walk to compare"}>
                  {on ? "✓" : ""}
                </button>
                <Link to={`/history/${w.session_id}`} className="walk-main">
                  <span className="walk-score">{w.overall_score.toFixed(1)}</span>
                  <span className="walk-meta">
                    {fmtDate(w.recorded_at)} · {w.grade}
                    {w.duration_s ? ` · ${w.duration_s.toFixed(1)}s` : ""}
                    {w.has_track ? "" : " · no replay"}
                    {w.warnings > 0 ? ` · ${w.warnings} quality flag${w.warnings > 1 ? "s" : ""}` : ""}
                    {w.coverage === "low" ? " · low confidence" : ""}
                  </span>
                </Link>
              </div>
            );
          })}
        </div>

        {picked.length === 2 && (
          <div className="btn-row" style={{ marginTop: "1.1rem" }}>
            <button className="btn solid"
                    onClick={() => navigate(`/compare/${picked[0]}/${picked[1]}`)}>
              Compare these two
            </button>
            <button className="btn" onClick={() => setPicked([])}>Clear</button>
          </div>
        )}
      </section>
    </>
  );
}
