import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { label } from "../criteria.js";

/* How each outcome reads. The wording is deliberate: "no measurable change" is
   not "no change", and nothing here claims the drill *caused* anything — one
   walk against another is not a controlled comparison. */
const STATUS = {
  improved: { title: "Measurably better", tone: "good" },
  no_change: { title: "No measurable change", tone: "flat" },
  declined: { title: "Measurably worse", tone: "bad" },
  awaiting_walk: { title: "Waiting on a walk", tone: "wait" },
};

function Outcome({ drill }) {
  const s = STATUS[drill.status] ?? STATUS.awaiting_walk;
  return (
    <div className={`outcome ${s.tone}`}>
      <div className="outcome-head">
        <strong>{s.title}</strong>
        {drill.delta != null && (
          <span className="outcome-delta">
            {drill.delta > 0 ? "+" : ""}{drill.delta.toFixed(1)}
          </span>
        )}
      </div>
      <p className="muted small">{drill.summary}</p>
      {drill.latest != null && (
        <p className="muted tiny">
          {drill.baseline.toFixed(1)} → {drill.latest.toFixed(1)} on {label(drill.criterion)}
          {drill.latest_session_id && (
            <> · <Link to={`/history/${drill.latest_session_id}`}>see that walk</Link></>
          )}
        </p>
      )}
    </div>
  );
}

function DrillCard({ drill, onPractice, onStop, busy }) {
  return (
    <article className="glass card drill-card">
      <header className="drill-head">
        <div>
          <span className="chip tiny">{label(drill.criterion)}</span>
          <h3 className="drill-name">{drill.drill_name}</h3>
        </div>
        <div className="drill-count">
          <span className="drill-num">{drill.practice_sessions}</span>
          <span className="muted tiny">
            {drill.practice_sessions === 1 ? "session" : "sessions"}
          </span>
        </div>
      </header>

      {drill.instruction && <p className="drill-how">{drill.instruction}</p>}
      {drill.duration && <p className="muted tiny">{drill.duration}</p>}

      <Outcome drill={drill} />

      <div className="drill-actions">
        <button className="btn solid" disabled={busy}
                onClick={() => onPractice(drill.assignment_id)}>
          I practised this
        </button>
        <button className="btn" disabled={busy}
                onClick={() => onStop(drill.assignment_id)}>
          Stop tracking
        </button>
      </div>
    </article>
  );
}

export default function Drills({ user }) {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);

  const load = useCallback(async () => {
    if (!user) return;
    try {
      setData(await api.drills());
      setError(null);
    } catch (e) {
      setError(e.message);
    }
  }, [user]);

  useEffect(() => { load(); }, [load]);

  async function act(fn) {
    setBusy(true);
    try {
      await fn();
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  if (!user) {
    return (
      <section className="stack">
        <header className="page-head">
          <h1>Drills</h1>
          <p className="muted">
            Track the drills you are working on and see whether your walks
            actually move.
          </p>
        </header>
        <div className="glass card">
          <p>Sign in to track drills — they are measured against your own walk history.</p>
          <Link className="btn solid" to="/account">Sign in</Link>
        </div>
      </section>
    );
  }

  const active = data?.active ?? [];
  const archived = data?.archived ?? [];

  return (
    <section className="stack">
      <header className="page-head">
        <h1>Drills</h1>
        <p className="muted">
          Each drill is measured against the score it started from. A change is
          only called real when it is larger than this system's own measurement
          error on that criterion.
        </p>
      </header>

      {error && (
        <div className="glass card error">
          <p>{error}</p>
          <button className="btn" onClick={() => { setError(null); load(); }}>Retry</button>
        </div>
      )}

      {data && active.length === 0 && (
        <div className="glass card">
          <h3>Nothing being tracked yet</h3>
          <p className="muted small">
            Analyse a walk, then take on one of the drills in its coaching plan.
            The score at that moment becomes the baseline everything after is
            measured against.
          </p>
          <Link className="btn solid" to="/">Analyse a walk</Link>
        </div>
      )}

      {active.map((d) => (
        <DrillCard key={d.assignment_id} drill={d} busy={busy}
                   onPractice={(id) => act(() => api.logPractice(id))}
                   onStop={(id) => act(() => api.stopDrill(id))} />
      ))}

      {active.length > 0 && (
        <div className="glass card tight">
          <p className="muted small">
            One walk improving after practice is not proof the drill caused it —
            different day, different light, different shoes. What the number
            shows is that the change is too large to be measurement noise.
            A consistent trend across several walks is the stronger signal.
          </p>
        </div>
      )}

      {archived.length > 0 && (
        <div className="glass card">
          <h3>No longer tracked</h3>
          <ul className="past-drills">
            {archived.map((d) => (
              <li key={d.assignment_id}>
                <span>{d.drill_name}</span>
                <span className="muted tiny">
                  {label(d.criterion)} · {d.practice_sessions}{" "}
                  {d.practice_sessions === 1 ? "session" : "sessions"}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </section>
  );
}
