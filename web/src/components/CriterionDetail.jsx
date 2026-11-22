/**
 * One criterion, opened up: every measurement behind the score, beside the
 * tolerance it was judged against.
 *
 * This is the panel the result page was missing. A user told "balance 4.2" could
 * previously see, at most, the one or two bands that failed badly enough to raise
 * a finding — and only if the criterion scored under 7. Everything else the
 * rubric computed was discarded: the bands that passed, how far each measurement
 * sat from its tolerance, and what the criterion was worth. So the number could
 * not be checked, argued with, or acted on precisely.
 *
 * The server sends all of it now (`breakdown`, from `scoring.explain`), read from
 * the same rubric that produced the score, including any thresholds moved by
 * calibration. Each band is drawn as a gauge with the acceptable region shaded,
 * because a target with a tolerance is a region and a single tick invites the
 * reader to treat one tolerance out — worth half marks — as a failure.
 */
import { ORDER, short } from "../criteria.js";
import { bandAxis, pct, targetText, withUnit } from "../format.js";

/** Model and rubric disagreeing by more than this is worth saying out loud: it
 *  means the twelve learned estimators and the transparent bands are reading the
 *  same measurements differently, and the bands below explain the rubric's view,
 *  not the model's. */
const DISAGREEMENT = 1.5;

export default function CriterionDetail({ breakdown, criterion, onSelect, scores,
                                          confidence, evidence = [], onSeek, source }) {
  if (!breakdown) return null;
  const keys = ORDER.filter((k) => breakdown[k]);
  const key = breakdown[criterion] ? criterion : keys[0];
  const entry = breakdown[key];
  if (!entry) return null;

  const per = confidence?.available ? confidence.per_criterion?.[key] : null;
  const disagrees = source === "model"
    && Math.abs(entry.rubric_score - entry.score) >= DISAGREEMENT;

  return (
    <section className="glass card detail">
      <h3>Behind the score</h3>

      <div className="detail-picker" role="tablist" aria-label="Criteria">
        {keys.map((k) => (
          <button
            key={k}
            role="tab"
            aria-selected={k === key}
            className={`chip clip-pick${k === key ? " on" : ""}`}
            onClick={() => onSelect?.(k)}
          >
            {short(k)} · {(scores?.[k] ?? breakdown[k].score).toFixed(1)}
          </button>
        ))}
      </div>

      <div className="detail-head">
        <div>
          <p className="detail-name">{entry.label}</p>
          <p className="detail-sub">
            {entry.n_out_of_tolerance === 0
              ? `all ${entry.n_bands} measurements inside tolerance`
              : `${entry.n_out_of_tolerance} of ${entry.n_bands} measurements outside tolerance`}
            {per?.tier && per.tier !== "unknown" ? ` · ${per.tier}ly identifiable` : ""}
          </p>
        </div>
        <div className="detail-marks">
          <span className="detail-score">{entry.score.toFixed(1)}<small>/10</small></span>
          <span className="detail-weight">
            {entry.marks_earned.toFixed(1)} of {entry.marks_available.toFixed(0)} marks
          </span>
        </div>
      </div>

      {/* Marks, not the 0-10 score, are what the overall grade is made of: the
          same 4.2 costs 7.5 marks on posture and 2.9 on stage presence. */}
      <div className="detail-bar">
        <i style={{ width: `${(entry.marks_earned / entry.marks_available) * 100}%` }} />
      </div>

      {disagrees && (
        <p className="muted tiny detail-disagree">
          The rubric alone scores this {entry.rubric_score.toFixed(1)}; the trained
          model scores it {entry.score.toFixed(1)}. The bands below are the rubric's
          reasoning — where the two differ this much, treat the number as less
          settled than it looks.
        </p>
      )}

      <div className="bands">
        {entry.bands.map((band) => {
          const [lo, hi, over, under] = bandAxis(band);
          const [aLo, aHi] = band.acceptable;
          const left = pct(aLo ?? lo, lo, hi);
          const right = pct(aHi ?? hi, lo, hi);
          const ev = evidence.find((e) => e.feature === band.feature);
          return (
            <div className={`band${band.in_tolerance ? "" : " out"}`} key={band.feature}>
              <div className="band-row">
                <span className="band-label">{band.label}</span>
                <span className="band-value">{withUnit(band.measured, band.unit)}</span>
              </div>
              <div className="band-gauge">
                <span className="band-ok" style={{ left: `${left}%`, width: `${right - left}%` }} />
                <span className="band-ideal" style={{ left: `${pct(band.ideal, lo, hi)}%` }} />
                <span className={`band-at${over ? " over" : ""}${under ? " under" : ""}`}
                      style={{ left: `${pct(band.measured, lo, hi)}%` }} />
              </div>
              <div className="band-foot">
                <span>target {targetText(band)}</span>
                <span>{Math.round(band.score * 100)}% of this band</span>
                <span>{Math.round(band.share * 100)}% of the criterion</span>
                {(over || under) && (
                  <span className="band-off">
                    off the scale — {withUnit(band.measured, band.unit)} against an
                    axis to {withUnit(hi, band.unit)}
                  </span>
                )}
              </div>
              {!band.in_tolerance && band.message && (
                <p className="band-message">{band.message}</p>
              )}
              {ev && onSeek && (
                <button className="band-seek" onClick={() => onSeek(ev.t)}>
                  ▸ worst at {ev.t.toFixed(1)}s — {ev.trace} {withUnit(ev.value, ev.unit)}
                </button>
              )}
            </div>
          );
        })}
      </div>

      <p className="live-caveat">
        Every row is a rubric band: the measurement, the target, and how much of
        this criterion it carries. The shaded strip is the region that scores at
        least half marks, so a marker just outside it is a near miss rather than a
        failure; the axis runs to three tolerances, and a measurement past that is
        marked off the scale with its value printed beside it.{" "}
        {per?.note ? per.note : ""}
      </p>
    </section>
  );
}
