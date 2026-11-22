import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { api, tier, GRADE_NOTE } from "../api.js";
import { label, short } from "../criteria.js";
import { fmt } from "../format.js";
import Processing, { useAnalysisStream } from "../components/Processing.jsx";
import HomeLoader from "../components/HomeLoader.jsx";
import SkeletonPlayer from "../components/SkeletonPlayer.jsx";
import BodyMap from "../components/BodyMap.jsx";
import Contribution from "../components/Contribution.jsx";
import CriterionDetail from "../components/CriterionDetail.jsx";
import Measurements from "../components/Measurements.jsx";
import Radar from "../components/Radar.jsx";
import Strides from "../components/Strides.jsx";

const ACCEPT = ".mp4,.mov,.avi,.mkv,.webm,.m4v,.csv,.xlsx,.xls";

/** The result page is four different questions, and stacking all four produced a
 *  page tens of thousands of pixels tall. The player stays mounted above the tabs
 *  so a "see it at 3.2s" jump works from any of them. */
const LENSES = [
  { id: "overview", label: "Overview" },
  { id: "motion", label: "Motion" },
  { id: "breakdown", label: "Breakdown" },
  { id: "coaching", label: "Coaching" },
];

function InfoIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <circle cx="8" cy="8" r="7" stroke="currentColor" strokeWidth="1.2" />
      <path d="M8 7.2v4M8 4.7v.9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function DropZone({ onFile, disabled }) {
  const [over, setOver] = useState(false);
  const input = useRef(null);

  const take = useCallback(
    (files) => {
      if (files?.length) onFile(files[0]);
    },
    [onFile],
  );

  return (
    <div
      className={`glass drop${over ? " over" : ""}`}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) take(e.dataTransfer.files);
      }}
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          input.current?.click();
        }
      }}
      role="button"
      tabIndex={0}
      aria-label="Upload a walk to analyse"
    >
      <span className="ring" aria-hidden="true">
        <svg width="19" height="19" viewBox="0 0 24 24" fill="none">
          <path d="M12 16V4m0 0L7 9m5-5 5 5M4 18v2h16v-2"
                stroke="currentColor" strokeWidth="1.3"
                strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </span>
      <h2>Upload your walk</h2>
      <p>Film the full body head-on for six to ten seconds, walking toward the camera.</p>
      <span className="filetypes">MP4 · MOV · CSV · XLSX</span>
      <input
        ref={input}
        type="file"
        accept={ACCEPT}
        hidden
        onChange={(e) => {
          take(e.target.files);
          e.target.value = "";   // let the same file be re-selected after a retry
        }}
      />
    </div>
  );
}

function LinkInput({ onSubmit, disabled }) {
  const [url, setUrl] = useState("");
  const [hosts, setHosts] = useState(null);

  useEffect(() => {
    // Ask the server which hosts it accepts rather than duplicating the list
    // here, so the two can never disagree.
    api.sources().then(setHosts).catch(() => setHosts({ available: false }));
  }, []);

  if (hosts && hosts.available === false) {
    return (
      <div className="glass card">
        <div className="notice">
          <InfoIcon />
          <span>
            Link fetching is unavailable — the server needs <code>yt-dlp</code>.
            Install it with <code>pip install yt-dlp</code>, then restart the API.
          </span>
        </div>
      </div>
    );
  }

  return (
    <form
      className="glass card"
      onSubmit={(e) => {
        e.preventDefault();
        if (url.trim()) onSubmit(url.trim());
      }}
    >
      <h3>From a link</h3>
      <div className="link-row">
        <input
          className="field-input"
          type="url"
          inputMode="url"
          placeholder="https://www.youtube.com/watch?v=..."
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={disabled}
          aria-label="Video link"
        />
        <button className="btn solid" type="submit" disabled={disabled || !url.trim()}>
          Analyse
        </button>
      </div>
      <div className="link-hosts">
        {(hosts?.platforms ?? ["YouTube", "Instagram", "TikTok", "Vimeo"]).join(" · ")}
        {` · up to ${Math.round((hosts?.max_duration_s ?? 300) / 60)} min`}
      </div>
      <p style={{ margin: "0.7rem 0 0", fontSize: "0.8rem", color: "var(--ink-3)" }}>
        Works best on your own posted videos. Private accounts need a login, so
        download those and upload the file instead.
      </p>
    </form>
  );
}

/**
 * The six grade bands, with this walk marked on them.
 *
 * A bare 61.4/100 does not say what it is close to. The bands are what the grade
 * name comes from, so drawing them turns the number into a position — and the
 * distance to the next one is the most concrete goal the page can offer.
 *
 * The thresholds come from the server with the rubric, not from a copy here.
 */
function GradeScale({ overall, bands }) {
  if (!bands?.length) return null;
  // Served highest-first; ascending is the order they are drawn in.
  const rising = [...bands].sort((a, b) => a.from - b.from);
  const next = rising.find((b) => b.from > overall);

  return (
    <div className="grades">
      <div className="grade-track">
        {rising.map((band, i) => {
          const top = rising[i + 1]?.from ?? 100;
          return (
            <span
              key={band.label}
              className={`grade-band${overall >= band.from && overall < top ? " on" : ""}`}
              style={{ left: `${band.from}%`, width: `${top - band.from}%` }}
              title={`${band.label}: ${band.from} and above`}
            />
          );
        })}
        <span className="grade-mark" style={{ left: `${Math.min(100, overall)}%` }} />
      </div>
      <div className="grade-ticks">
        {rising.map((band) => (
          <span key={band.label} style={{ left: `${band.from}%` }}>{band.from}</span>
        ))}
      </div>
      {next && (
        <p className="grade-gap">
          {(next.from - overall).toFixed(1)} marks to {next.label}
        </p>
      )}
    </div>
  );
}

function ScoreHero({ result, bands }) {
  const { overall, grade, source } = result;
  return (
    <section className="glass card hero">
      <div className="hero-grid">
        <div className="numeral">
          {overall.toFixed(1)}
          <small>/100</small>
        </div>
        <div>
          <p className="grade">{grade}</p>
          <p style={{ margin: "0 0 0.7rem", color: "var(--ink-2)", fontSize: "0.9rem" }}>
            {GRADE_NOTE[grade] ?? ""}
          </p>
          <p className="meta">
            {result.coaching?.strengths?.length
              ? `Strongest — ${result.coaching.strengths[0]}`
              : "Twelve criteria assessed"}
            <br />
            {result.origin?.title ? (
              <>
                {String(result.origin.title).slice(0, 46)}
                {result.origin.uploader ? ` · ${result.origin.uploader}` : ""}
                <br />
              </>
            ) : null}
            Scored by {typeof source === "string" ? source : "model"}
            {result.coverage?.overall && result.coverage.overall !== "high"
              ? ` · ${result.coverage.overall} confidence`
              : ""}
          </p>
          <GradeScale overall={overall} bands={bands} />
        </div>
      </div>
    </section>
  );
}

/* How well a criterion can be identified at all — a property of the model, not
   of this clip. Weak ones are marked so twelve scores drawn identically do not
   imply twelve equally trustworthy measurements. */
function Identifiability({ entry }) {
  if (!entry || entry.tier === "strong" || entry.tier === "unknown") return null;
  return (
    <span className={`ident ${entry.tier}`}
          title={`${entry.note} Identifiability R² ${entry.r2_independent?.toFixed(2)} `
                 + `against independently varied ground truth.`}>
      {entry.tier}
    </span>
  );
}

function Criteria({ scores, confidence, breakdown, selected, onSelect }) {
  const rows = Object.entries(scores).sort((a, b) => b[1] - a[1]);
  const per = confidence?.available ? confidence.per_criterion : null;
  const weak = per
    ? rows.filter(([k]) => per[k]?.tier === "weak").map(([k]) => label(k))
    : [];

  return (
    <section className="glass card">
      <h3>Criterion scores</h3>
      <div className="rows">
        {rows.map(([key, value], i) => {
          const entry = breakdown?.[key];
          // Every row is a control now. The score is the summary; the bands
          // behind it are one tap away, which is the difference between being
          // told a number and being able to check it.
          const Row = onSelect ? "button" : "div";
          return (
            <Row
              className={`row q-${tier(value)}${onSelect ? " pickable" : ""}`
                + `${selected === key ? " on" : ""}`}
              key={key}
              onClick={onSelect ? () => onSelect(key) : undefined}
              aria-label={onSelect
                ? `${label(key)} ${value.toFixed(1)} out of 10, open the measurements behind it`
                : undefined}
            >
              <span className="name">
                {label(key)}
                <Identifiability entry={per?.[key]} />
              </span>
              <span className="val">{value.toFixed(1)}</span>
              <span className="track">
                <i style={{ width: `${value * 10}%`, animationDelay: `${i * 45}ms` }} />
              </span>
              {entry && (
                <span className="row-marks">
                  {entry.marks_earned.toFixed(1)}/{entry.marks_available.toFixed(0)} marks
                  {entry.n_out_of_tolerance > 0
                    ? ` · ${entry.n_out_of_tolerance} of ${entry.n_bands} out of tolerance`
                    : " · all measurements in tolerance"}
                </span>
              )}
            </Row>
          );
        })}
      </div>
      {weak.length > 0 && (
        <p className="muted tiny" style={{ marginTop: "0.9rem" }}>
          {weak.join(", ")} {weak.length === 1 ? "is" : "are"} only weakly
          identifiable from pose alone — read those as hints and lean on the
          evidence traces rather than the number.{" "}
          <Link to="/about">Why</Link>
        </p>
      )}
    </section>
  );
}

/* Start tracking a drill. The baseline is *not* sent from here — the server
   reads it from the user's own latest walk, so the number every future outcome
   is measured against cannot be set by the client. */
function TakeOn({ drill }) {
  const [state, setState] = useState("idle");
  const [detail, setDetail] = useState(null);

  if (state === "done") {
    return (
      <p className="drill-tracked">
        Tracking from {detail?.baseline?.toFixed(1)}/10 — a change over
        ±{detail?.threshold?.toFixed(1)} counts as real.{" "}
        <Link to="/drills">See drills</Link>
      </p>
    );
  }

  return (
    <>
      <button
        className="btn small"
        disabled={state === "busy"}
        onClick={async () => {
          setState("busy");
          try {
            setDetail(await api.takeOnDrill(drill.criterion, drill.name));
            setState("done");
          } catch (e) {
            setDetail({ error: e.message });
            setState("idle");
          }
        }}
      >
        {state === "busy" ? "Starting…" : "Track this drill"}
      </button>
      {detail?.error && <p className="muted tiny">{detail.error}</p>}
    </>
  );
}

function Coaching({ coaching, evidence = [], onSeek, onSelect, user }) {
  if (!coaching) return null;
  const { focus_areas: focus = [], drills = [], next_session: next = [], summary } = coaching;

  return (
    <>
      {focus.length > 0 && (
        <section className="glass card">
          <h3>Focus areas</h3>
          {summary && (
            <p style={{ margin: "0 0 1.1rem", color: "var(--ink-2)", fontSize: "0.89rem" }}>
              {summary}
            </p>
          )}
          <div className="focus">
            {focus.map((f, i) => {
              const ev = evidence.find((e) => e.feature === f.feature)
                ?? evidence.find((e) => e.criterion === f.criterion);
              const jump = ev && onSeek ? () => onSeek(ev.t) : null;
              return (
              <div
                className={`focus-item${jump ? " seekable" : ""}`}
                key={f.criterion}
                onClick={jump ?? undefined}
                role={jump ? "button" : undefined}
                tabIndex={jump ? 0 : undefined}
                onKeyDown={jump ? (e) => {
                  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); jump(); }
                } : undefined}
              >
                <span className="rank">{String(i + 1).padStart(2, "0")}</span>
                <div>
                  <h4>
                    {short(f.criterion)} — {f.score}/10
                  </h4>
                  <p>{f.issue}</p>
                  {f.feature && (
                    <div className="measure">
                      {f.feature}: {fmt(f.measured)} · target {fmt(f.target)}
                    </div>
                  )}
                  <div className="focus-actions">
                    {jump && (
                      <span className="seek-hint">
                        ▸ see it at {ev.t.toFixed(1)}s — {ev.trace} {fmt(ev.value)} {ev.unit}
                      </span>
                    )}
                    {onSelect && (
                      <button
                        className="focus-why"
                        onClick={(e) => { e.stopPropagation(); onSelect(f.criterion); }}
                      >
                        all measurements
                      </button>
                    )}
                  </div>
                </div>
              </div>
              );
            })}
          </div>
        </section>
      )}

      {drills.length > 0 && (
        <section className="glass card">
          <h3>Prescribed drills</h3>
          {drills.map((d, i) => (
            <div className="drill" key={`${d.name}-${i}`}>
              <span className="for">
                {short(d.criterion)} · {d.duration}
              </span>
              <b>{d.name}</b>
              <p>{d.instruction}</p>
              {user && <TakeOn drill={d} />}
            </div>
          ))}
          {!user && (
            <p className="muted tiny">
              Sign in to track a drill and have the next walk measure whether it moved.
            </p>
          )}
        </section>
      )}

      {next.length > 0 && (
        <section className="glass card">
          <h3>Next session</h3>
          <div className="chips">
            {next.map((n) => (
              <span className="chip" key={n}>{n}</span>
            ))}
          </div>
        </section>
      )}
    </>
  );
}

export default function Analyse({ user }) {
  const location = useLocation();
  // Accept pre-loaded results from Live coaching → "See full analysis"
  const preloaded = location.state ?? {};

  const [file, setFile] = useState(preloaded.videoFile ?? null);
  const [videoFile, setVideoFile] = useState(
    preloaded.videoFile ?? null,
  );
  const [error, setError] = useState(null);
  const [results, setResults] = useState(preloaded.results ?? null);
  const [clip, setClip] = useState(0);
  const [lens, setLens] = useState("overview");
  const [selected, setSelected] = useState(null);
  const [playhead, setPlayhead] = useState(0);
  const [bands, setBands] = useState(null);
  const stream = useAnalysisStream();
  const seekRef = useRef(null);
  const busy = stream.busy;

  // The grade thresholds belong to the rubric, so they are fetched rather than
  // restated. Cached in the api module, so revisiting the page costs nothing and
  // a failure simply means the scale is not drawn.
  useEffect(() => {
    api.criteriaInfo().then((info) => setBands(info.grade_bands)).catch(() => {});
  }, []);

  /* Selecting a criterion — from the radar, the body map, a score row, a focus
     area, a measurement — always means the same thing: show me what was measured.
     So it opens the breakdown lens rather than expanding something in place, and
     the page has one place where that question is answered instead of four. */
  const select = useCallback((criterion) => {
    if (!criterion) return;
    setSelected(criterion);
    setLens("breakdown");
  }, []);

  async function runUrl(url) {
    setFile({ name: url });
    setVideoFile(null);          // the browser never holds a linked video
    setError(null);
    setResults(null);
    setClip(0);
    try {
      setResults(await stream.run(() => api.analyzeUrlStream(url)));
    } catch (e) {
      setError(e.message);
    }
  }

  async function run(f) {
    setFile(f);
    setVideoFile(f.type?.startsWith("video/") ? f : null);
    setError(null);
    setResults(null);
    setClip(0);
    try {
      setResults(await stream.run(() => api.analyzeStream(f)));
    } catch (e) {
      setError(e.message);
    }
  }

  const searchParams = new URLSearchParams(location.search);
  const isPreview = searchParams.get("loading") === "true";

  if (busy || isPreview) {
    return (
      <HomeLoader
        stages={stream.stages}
        states={stream.states}
        elapsed={stream.elapsed}
        filename={
          file?.name?.length > 58
            ? `${file.name.slice(0, 58)}…`
            : file?.name || (isPreview ? "runway_walk_preview.mp4" : undefined)
        }
      />
    );
  }

  if (error) {
    return (
      <div className="glass card">
        <h3>Could not analyse that file</h3>
        <div className="notice">
          <InfoIcon />
          <span>{error}</span>
        </div>
        <div className="btn-row" style={{ marginTop: "1.2rem" }}>
          <button className="btn solid" onClick={() => { setError(null); setResults(null); }}>
            Try another file
          </button>
          <Link className="btn" to="/guide#problems">What this means</Link>
        </div>
      </div>
    );
  }

  if (!results) {
    return (
      <div className="drop-wrap">
        <DropZone onFile={run} disabled={busy} />
        <div className="or">paste a link</div>
        <LinkInput onSubmit={runUrl} disabled={busy} />
      </div>
    );
  }

  if (results.length === 0) {
    return (
      <div className="glass card">
        <h3>Nothing to score</h3>
        <p style={{ color: "var(--ink-2)" }}>No analysable clip was found in that file.</p>
        <button className="btn solid" onClick={() => setResults(null)}>Upload another</button>
      </div>
    );
  }

  // A keypoint file can hold many clips. Rendering them all stacked produced a
  // page tens of thousands of pixels tall on a phone, so only the selected clip
  // is shown and the rest sit behind a picker.
  const r = results[Math.min(clip, results.length - 1)];

  const playable = Boolean(videoFile || r.origin?.video_url);
  const onSeek = playable ? (t) => seekRef.current?.(t) : null;

  return (
    <>
      <div className="btn-row" style={{ marginBottom: "var(--gap)" }}>
        <button
          className="btn"
          onClick={() => {
            setResults(null); setFile(null); setVideoFile(null); setClip(0);
            setLens("overview"); setSelected(null);
          }}
        >
          New analysis
        </button>
      </div>

      {results.length > 1 && (
        <div className="glass card" style={{ marginBottom: "var(--gap)" }}>
          <h3>{results.length} clips in this file</h3>
          <div className="chips" role="tablist" aria-label="Clips">
            {results.map((c, i) => (
              <button
                key={c.video_id}
                className={`chip clip-pick${i === clip ? " on" : ""}`}
                role="tab"
                aria-selected={i === clip}
                onClick={() => setClip(i)}
              >
                {c.video_id} · {c.overall.toFixed(0)}
              </button>
            ))}
          </div>
        </div>
      )}

      <ScoreHero result={r} bands={bands} />

      {/* The player stays put across the lenses below. A finding that says "see it
          at 3.2s" has to be able to move the video from wherever it was clicked,
          and remounting the element on every tab change would reload the clip and
          lose the frame. */}
      {results.length === 1 && (
        <div style={{ marginTop: "var(--gap)" }}>
          <SkeletonPlayer
            file={videoFile}
            src={r.origin?.video_url ? api.mediaUrl(r.origin.video_url) : null}
            track={r.track}
            timeline={r.timeline}
            seekRef={seekRef}
            onTime={setPlayhead}
            onSelect={select}
          />
        </div>
      )}

      <div className="tabs lens-tabs" role="tablist" aria-label="Result sections"
           style={{ marginTop: "var(--gap)" }}>
        {LENSES.map((l) => (
          <button
            key={l.id}
            role="tab"
            className="tab"
            aria-selected={lens === l.id}
            onClick={() => setLens(l.id)}
          >
            {l.label}
          </button>
        ))}
      </div>

      {lens === "overview" && (
        <>
          <div className="split">
            <section className="glass card">
              <h3>Shape of the walk</h3>
              <Radar
                scores={r.scores}
                selected={selected}
                onSelect={select}
                confidence={r.measurement_confidence}
              />
            </section>
            <section className="glass card">
              <h3>On the body</h3>
              <BodyMap scores={r.scores} selected={selected} onSelect={select} />
            </section>
          </div>

          <div style={{ marginTop: "var(--gap)" }}>
            <Criteria
              scores={r.scores}
              confidence={r.measurement_confidence}
              breakdown={r.breakdown}
              selected={selected}
              onSelect={select}
            />
          </div>

          {r.strengths?.length > 0 && (
            <section className="glass card" style={{ marginTop: "var(--gap)" }}>
              <h3>Holding up well</h3>
              <div className="chips">
                {r.strengths.map((s) => <span className="chip" key={s}>{s}</span>)}
              </div>
              <p className="muted tiny" style={{ marginTop: "0.8rem" }}>
                At or above 7.5/10, so these are reported as strengths and never
                raise a correction.
              </p>
            </section>
          )}

          <FaceCheck attribution={r.attribution} />
          <Quality warnings={r.quality_warnings} />
        </>
      )}

      {lens === "motion" && (
        <>
          <Strides timeline={r.timeline} onSeek={onSeek} current={playhead} />
          {!r.timeline && (
            <section className="glass card">
              <h3>Motion</h3>
              <p className="lede" style={{ color: "var(--ink-2)" }}>
                No per-frame traces came back for this clip, so there is nothing to
                segment into strides. That happens when the recording was too short
                or too gappy for the phase tracker to find a cycle.
              </p>
            </section>
          )}
          <Viewpoint viewpoint={r.viewpoint} onSelect={select} />
          <Gaze gaze={r.gaze} />
        </>
      )}

      {lens === "breakdown" && (
        <>
          <CriterionDetail
            breakdown={r.breakdown}
            criterion={selected}
            onSelect={setSelected}
            scores={r.scores}
            confidence={r.measurement_confidence}
            evidence={r.timeline?.evidence}
            onSeek={onSeek}
            source={r.source}
          />
          <div style={{ marginTop: "var(--gap)" }}>
            <Contribution
              breakdown={r.breakdown}
              overall={r.overall}
              selected={selected}
              onSelect={setSelected}
            />
          </div>
          <div style={{ marginTop: "var(--gap)" }}>
            <Measurements measurements={r.measurements} onSelect={setSelected} />
          </div>
          {!r.breakdown && (
            <section className="glass card">
              <h3>Behind the score</h3>
              <p className="lede" style={{ color: "var(--ink-2)" }}>
                This result came from an older version of the API, which did not
                send the per-band breakdown. Re-run the analysis to see it.
              </p>
            </section>
          )}
        </>
      )}

      {lens === "coaching" && (
        <Coaching
          coaching={r.coaching}
          user={user}
          evidence={r.timeline?.evidence}
          onSeek={onSeek}
          onSelect={select}
        />
      )}
    </>
  );
}

function Viewpoint({ viewpoint, onSelect }) {
  if (!viewpoint) return null;
  const pill = (keys) => keys.map((k) => (
    <button key={k} className="chip clip-pick" onClick={() => onSelect?.(k)}>
      {short(k)}
    </button>
  ));
  return (
    <section className="glass card" style={{ marginTop: "var(--gap)" }}>
      <h3>Camera angle — {viewpoint.band}</h3>
      <div className="vp-scale">
        <div className="vp-track">
          <i className="vp-range"
             style={{ left: `${(viewpoint.range_deg[0] / 90) * 100}%`,
                      width: `${((viewpoint.range_deg[1] - viewpoint.range_deg[0]) / 90) * 100}%` }} />
          <span className="vp-mark" style={{ left: `${(viewpoint.angle_deg / 90) * 100}%` }} />
        </div>
        <div className="vp-ticks"><span>head-on</span><span>45°</span><span>side</span></div>
      </div>
      <p className="vp-read">
        about <strong>{Math.round(viewpoint.angle_deg)}°</strong> off head-on
        <span className="vp-pm"> ± {Math.round(viewpoint.uncertainty_deg)}°</span>
      </p>

      <div className="vp-cols">
        <div>
          <span className="vp-h">Measures well here</span>
          <div className="chips">{pill(viewpoint.well_measured)}</div>
        </div>
        <div>
          <span className="vp-h">Least reliable here</span>
          <div className="chips">{pill(viewpoint.poorly_measured)}</div>
        </div>
      </div>

      <div className="notice" style={{ marginTop: "1rem" }}>
        <InfoIcon />
        <span>{viewpoint.advice}</span>
      </div>
      <p className="live-caveat">{viewpoint.basis}.</p>
    </section>
  );
}

function Gaze({ gaze }) {
  if (!gaze) return null;
  return (
    <section className="glass card" style={{ marginTop: "var(--gap)" }}>
      <h3>Gaze — measured from your face</h3>
      <div className="gaze">
        <div className="gaze-stat">
          <span className="gaze-num">{Math.round(gaze.on_camera_fraction * 100)}%</span>
          <span className="gaze-lab">eyeline forward</span>
          <div className="gaze-bar">
            <i style={{ width: `${gaze.on_camera_fraction * 100}%` }} />
          </div>
        </div>
        <div className="gaze-stat">
          <span className="gaze-num">{Math.round(gaze.looking_down_fraction * 100)}%</span>
          <span className="gaze-lab">chin dropped</span>
          <div className="gaze-bar">
            <i className="bad" style={{ width: `${gaze.looking_down_fraction * 100}%` }} />
          </div>
        </div>
        <div className="gaze-stat">
          <span className="gaze-num">{gaze.roll_deg}°</span>
          <span className="gaze-lab">head tilt</span>
          <div className="gaze-bar">
            <i style={{ width: `${Math.min(100, (gaze.roll_deg / 10) * 100)}%` }} />
          </div>
        </div>
      </div>
      {gaze.notes?.map((n, i) => (
        <p className="gaze-note" key={i}>{n}</p>
      ))}
      <p className="live-caveat">
        Head pose is read directly from your eyes, nose and mouth across{" "}
        {gaze.frames} frames. These are measurements, not model predictions — no
        score is derived from them, they only explain the head position result.
      </p>
    </section>
  );
}

function FaceCheck({ attribution }) {
  if (!attribution?.checked) return null;
  return (
    <div className="glass card" style={{ marginTop: "var(--gap)" }}>
      <h3>Face check</h3>
      <div className="notice">
        <InfoIcon />
        <span>
          {attribution.is_account_holder ? (
            <>This walk matches your enrolled face
              {" "}({attribution.frames_used} frames checked). Filed to your account.</>
          ) : (
            <>This walk does <strong>not</strong> match your enrolled face. It has
              still been saved to your account — face matching is a convenience, not
              a gate, and it can be wrong. If this is you, re-enrol in better light.</>
          )}
        </span>
      </div>
    </div>
  );
}

function Quality({ warnings }) {
  if (!warnings?.length) return null;
  return (
    <div className="glass card" style={{ marginTop: "var(--gap)" }}>
      <h3>Recording quality</h3>
      {warnings.map((w, i) => (
        <div className="notice" key={i}>
          <InfoIcon />
          <span>{w}</span>
        </div>
      ))}
    </div>
  );
}
