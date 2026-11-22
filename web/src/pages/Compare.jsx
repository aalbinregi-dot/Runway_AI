/**
 * Two walks side by side.
 *
 * Only scores are compared. The measurement traces are deliberately not
 * overlaid: two clips have different durations, different numbers of gait
 * cycles and possibly different camera distances, so a difference between their
 * traces can be an artefact of the recording rather than of the walking. Where
 * the recordings differ enough for that to matter, the page says so instead of
 * letting the reader assume otherwise.
 */
import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import { SignInRequired } from "../Shell.jsx";
import Radar from "../components/Radar.jsx";
import SkeletonFigure from "../components/SkeletonFigure.jsx";

const fmtDate = (s) =>
  new Date(s).toLocaleDateString(undefined, { day: "numeric", month: "short" });

export default function Compare({ user }) {
  const { a, b } = useParams();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!user || !a || !b) return;
    setData(null);
    setError(null);
    api.compare(a, b).then(setData).catch((e) => setError(e.message));
  }, [user, a, b]);

  if (!user) return <SignInRequired what="Compare walks" />;

  if (!a || !b) {
    return (
      <div className="glass card">
        <h3>Compare</h3>
        <p style={{ color: "var(--ink-2)" }}>
          Choose two walks in your history, then press Compare.
        </p>
        <Link to="/history" className="btn solid"
              style={{ marginTop: "1rem", display: "inline-block", textDecoration: "none" }}>
          Go to history
        </Link>
      </div>
    );
  }
  if (error) return <div className="glass card"><h3>Compare</h3><p>{error}</p></div>;
  if (!data) {
    return <div className="glass card loading"><span className="pulse" /><p>Loading</p></div>;
  }

  const widest = Math.max(...data.deltas.map((d) => Math.abs(d.delta)), 1);
  // Scores, not deltas: the same twelve numbers the delta list is built from,
  // shaped so the two walks can be seen as two shapes. A list of twelve signed
  // changes says what moved; the two outlines say whether the walk changed
  // character or simply drifted.
  const scoresOf = (side) =>
    Object.fromEntries(data.deltas.map((d) => [d.criterion, d[side]]));

  return (
    <>
      <section className="glass card">
        <h3>Comparison</h3>
        <div className="cmp-head">
          {[data.a, data.b].map((side, i) => (
            <div className="cmp-side" key={i}>
              <span className="cmp-when">{fmtDate(side.session.recorded_at)}</span>
              <span className="cmp-score">{side.session.overall_score.toFixed(1)}</span>
              <span className="cmp-grade">{side.session.grade}</span>
              <SkeletonFigure track={side.track} height={150} />
            </div>
          ))}
        </div>
        <div className={`cmp-delta${data.overall_significant ? "" : " weak"}`}>
          {data.overall_delta > 0 ? "+" : ""}{data.overall_delta.toFixed(1)} overall
          {!data.overall_significant && (
            <span className="cmp-noise">
              {" "}· within the ±{data.overall_noise_floor} margin of error
            </span>
          )}
        </div>
        {data.notes.map((n, i) => (
          <div className="notice" key={i} style={{ marginTop: "0.8rem" }}>
            <span>{n}</span>
          </div>
        ))}
      </section>

      <section className="glass card">
        <h3>Then and now</h3>
        <Radar scores={scoresOf("b")} reference={scoresOf("a")}
               referenceLabel="earlier" />
      </section>

      <section className="glass card">
        <h3>Criterion by criterion</h3>
        <div className="delta">
          {[...data.deltas].sort((x, y) => y.delta - x.delta).map((d) => (
            <div key={d.criterion}>
              <div className="cmp-row">
                <span>{d.label}</span>
                <span className="cmp-nums">
                  {d.a.toFixed(1)} <span className="cmp-arrow">→</span> {d.b.toFixed(1)}
                </span>
                <span className={`cmp-diff${d.significant ? "" : " weak"}`}
                      title={d.significant ? "Larger than the model's error on this criterion"
                                           : `Smaller than the ±${d.noise_floor} error on this criterion`}>
                  {d.delta > 0 ? "+" : ""}{d.delta.toFixed(1)}
                </span>
              </div>
              <div className="delta-bar">
                {/* Centre line at 50%: gains extend right, regressions left. */}
                <i style={{
                  left: d.delta >= 0 ? "50%" : `${50 - (Math.abs(d.delta) / widest) * 50}%`,
                  width: `${(Math.abs(d.delta) / widest) * 50}%`,
                  // Faded when the change is inside the model's error, so the
                  // eye is not drawn to movement that is not evidence.
                  opacity: d.delta === 0 ? 0.25 : d.significant ? 1 : 0.3,
                }} />
              </div>
            </div>
          ))}
        </div>
      </section>

      {(data.most_improved || data.most_regressed) && (
        <section className="glass card">
          <h3>Headline — changes beyond the margin of error</h3>
          {data.most_improved && (
            <p style={{ margin: "0 0 0.5rem" }}>
              Biggest gain: <strong>{data.most_improved.label}</strong>{" "}
              +{data.most_improved.delta.toFixed(1)}
            </p>
          )}
          {data.most_regressed && (
            <p style={{ margin: 0, color: "var(--ink-2)" }}>
              Slipped most: <strong>{data.most_regressed.label}</strong>{" "}
              {data.most_regressed.delta.toFixed(1)}
            </p>
          )}
        </section>
      )}
    </>
  );
}
