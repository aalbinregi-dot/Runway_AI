/**
 * One saved walk.
 *
 * The scores and findings are read back from the database, but the *timeline* and
 * the per-band breakdown are recomputed from the stored keypoints on each visit.
 * That means an old walk is always measured by the current feature code rather
 * than by whatever version happened to be running the day it was uploaded — and
 * the breakdown is keyed to the stored scores, so the explanation belongs to the
 * number the page is showing.
 *
 * The replay is a skeleton, not footage: no video is kept. It drives the clock
 * for the evidence strip below it, so clicking a moment in the traces moves the
 * figure to that frame.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api, tier } from "../api.js";
import { LABELS } from "../criteria.js";
import { SignInRequired } from "../Shell.jsx";
import BodyMap from "../components/BodyMap.jsx";
import Contribution from "../components/Contribution.jsx";
import CriterionDetail from "../components/CriterionDetail.jsx";
import Measurements from "../components/Measurements.jsx";
import Radar from "../components/Radar.jsx";
import SkeletonFigure from "../components/SkeletonFigure.jsx";
import Strides from "../components/Strides.jsx";
import TimelineExplorer from "../components/TimelineExplorer.jsx";

export default function WalkDetail({ user }) {
  const { id } = useParams();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState(null);
  const [playhead, setPlayhead] = useState(0);
  const seekRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    if (!user) return;
    setData(null);
    api.walk(id).then(setData).catch((e) => setError(e.message));
  }, [user, id]);

  const seek = useCallback((t) => seekRef.current?.(t), []);

  if (!user) return <SignInRequired what="This walk" />;
  if (error) return <div className="glass card"><h3>Walk</h3><p>{error}</p></div>;
  if (!data) {
    return <div className="glass card loading"><span className="pulse" /><p>Loading</p></div>;
  }

  const s = data.session;
  const rows = Object.entries(data.scores).sort((a, b) => b[1] - a[1]);

  async function remove() {
    if (!window.confirm("Delete this walk permanently?")) return;
    setBusy(true);
    try {
      await api.deleteWalk(id);
      navigate("/history");
    } catch (e) {
      setError(e.message);
      setBusy(false);
    }
  }

  return (
    <>
      <div className="btn-row" style={{ marginBottom: "var(--gap)" }}>
        <Link to="/history" className="btn" style={{ textDecoration: "none" }}>
          ← All walks
        </Link>
        <a className="btn" href={api.reportUrl(id)}>Download PDF</a>
        <button className="btn" onClick={remove} disabled={busy}>Delete</button>
      </div>

      <section className="glass card hero">
        <div className="hero-grid">
          <div className="numeral">
            {s.overall_score.toFixed(1)}<small>/100</small>
          </div>
          <div>
            <p className="grade">{s.grade}</p>
            <p className="meta">
              {new Date(s.recorded_at).toLocaleString()}
              <br />
              {s.duration_s ? `${s.duration_s.toFixed(1)}s · ` : ""}
              {s.n_frames ? `${s.n_frames} frames · ` : ""}
              scored by {s.score_source}
            </p>
          </div>
        </div>
      </section>

      <section className="glass card" style={{ marginTop: "var(--gap)" }}>
        <h3>Replay</h3>
        <SkeletonFigure track={data.track} height={300}
                        seekRef={seekRef} onTime={setPlayhead} />
        <p className="live-caveat" style={{ marginTop: "0.8rem" }}>
          The original video is not kept — only the tracked keypoints — so a saved
          walk replays as a skeleton. Tap to pause.
        </p>
        {data.timeline && (
          <TimelineExplorer
            timeline={data.timeline}
            current={playhead}
            onSeek={seek}
            onSelect={setSelected}
          />
        )}
      </section>

      <div className="split" style={{ marginTop: "var(--gap)" }}>
        <section className="glass card">
          <h3>Shape of the walk</h3>
          <Radar scores={data.scores} selected={selected} onSelect={setSelected} />
        </section>
        <section className="glass card">
          <h3>On the body</h3>
          <BodyMap scores={data.scores} selected={selected} onSelect={setSelected} />
        </section>
      </div>

      <div className="split" style={{ marginTop: "var(--gap)" }}>
        <div>
          <section className="glass card">
            <h3>Criterion scores</h3>
            <div className="rows">
              {rows.map(([key, value], i) => (
                <button
                  className={`row pickable q-${tier(value)}${selected === key ? " on" : ""}`}
                  key={key}
                  onClick={() => setSelected(key)}
                >
                  <span className="name">{LABELS[key] ?? key}</span>
                  <span className="val">{value.toFixed(1)}</span>
                  <span className="track">
                    <i style={{ width: `${value * 10}%`, animationDelay: `${i * 40}ms` }} />
                  </span>
                </button>
              ))}
            </div>
          </section>

          {data.findings.length > 0 && (
            <section className="glass card">
              <h3>What was flagged</h3>
              <div className="focus">
                {/* One row per criterion: findings are raised per rubric band,
                    so a single weak criterion can otherwise fill the list. */}
                {data.findings
                  .filter((f, i, all) =>
                    all.findIndex((g) => g.criterion === f.criterion) === i)
                  .slice(0, 5)
                  .map((f, i) => (
                    <button
                      className="focus-item seekable"
                      key={f.criterion}
                      onClick={() => setSelected(f.criterion)}
                    >
                      <span className="rank">{String(i + 1).padStart(2, "0")}</span>
                      <div>
                        <h4>
                          {LABELS[f.criterion] ?? f.criterion} —{" "}
                          {(data.scores[f.criterion] ?? 0).toFixed(1)}/10
                        </h4>
                        <p>{f.message}</p>
                      </div>
                    </button>
                  ))}
              </div>
            </section>
          )}
        </div>

        <div>
          <Strides timeline={data.timeline} onSeek={seek} current={playhead} />
          <div style={{ marginTop: "var(--gap)" }}>
            <Contribution breakdown={data.breakdown} overall={s.overall_score}
                          selected={selected} onSelect={setSelected} />
          </div>
        </div>
      </div>

      {data.breakdown && (
        <div style={{ marginTop: "var(--gap)" }}>
          <CriterionDetail
            breakdown={data.breakdown}
            criterion={selected}
            onSelect={setSelected}
            scores={data.scores}
            evidence={data.timeline?.evidence}
            onSeek={seek}
            source={s.score_source}
          />
        </div>
      )}

      {data.measurements && (
        <div style={{ marginTop: "var(--gap)" }}>
          <Measurements measurements={data.measurements} onSelect={setSelected} />
        </div>
      )}
    </>
  );
}
