import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, tier } from "../api.js";
import { label } from "../criteria.js";
import Processing, { useAnalysisStream } from "../components/Processing.jsx";
import SkeletonPlayer from "../components/SkeletonPlayer.jsx";

const ACCEPT = ".mp4,.mov,.avi,.mkv,.webm,.m4v";


function DropZone({ onFile, disabled }) {
  const [over, setOver] = useState(false);
  const input = useRef(null);

  const take = useCallback((files) => { if (files?.length) onFile(files[0]); }, [onFile]);

  return (
    <div
      className={`glass drop${over ? " over" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setOver(true); }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        if (!disabled) take(e.dataTransfer.files);
      }}
      onClick={() => !disabled && input.current?.click()}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.current?.click(); }
      }}
      role="button"
      tabIndex={0}
      aria-label="Upload a group walk to analyse"
    >
      <span className="ring" aria-hidden="true">
        <svg width="19" height="19" viewBox="0 0 24 24" fill="none">
          <circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.3" />
          <circle cx="16.5" cy="9" r="2.4" stroke="currentColor" strokeWidth="1.3" />
          <path d="M3 19c0-2.8 2.2-5 5-5s5 2.2 5 5M14 19c0-2.2 1.4-4 3.5-4s3.5 1.8 3.5 4"
                stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" />
        </svg>
      </span>
      <h2>Upload a group walk</h2>
      <p>Everyone whose face is visible is found, scored separately, and ranked.</p>
      <span className="filetypes">MP4 · MOV · WEBM</span>
      <input ref={input} type="file" accept={ACCEPT} hidden
             onChange={(e) => { take(e.target.files); e.target.value = ""; }} />
    </div>
  );
}

/* One walker: rank, name, score, and their strongest and weakest criteria. */
function PersonCard({ result, rank, selected, onSelect }) {
  const person = result.person ?? {};
  const scores = result.scores ?? {};
  const entries = Object.entries(scores);
  const best = entries.reduce((a, b) => (b[1] > a[1] ? b : a), entries[0] ?? ["", 0]);
  const worst = entries.reduce((a, b) => (b[1] < a[1] ? b : a), entries[0] ?? ["", 0]);

  return (
    <button
      type="button"
      className={`glass card person-card${selected ? " selected" : ""}`}
      onClick={onSelect}
      aria-pressed={selected}
    >
      <div className="person-head">
        <span className="person-rank">{rank}</span>
        <div className="person-id">
          <strong>{person.label ?? "Person"}</strong>
          {person.matched_user_id ? (
            <span className="chip tiny">matched by face</span>
          ) : (
            <span className="muted tiny">not recognised</span>
          )}
        </div>
        <div className="person-score">
          <span className={`person-num q-${tier(result.overall / 10)}`}>
            {result.overall?.toFixed(1)}
          </span>
          <span className="muted tiny">{result.grade}</span>
        </div>
      </div>
      <dl className="person-extremes">
        <div><dt>Strongest</dt><dd>{label(best[0])}</dd></div>
        <div><dt>Weakest</dt><dd>{label(worst[0])}</dd></div>
      </dl>
    </button>
  );
}

/* Side-by-side criterion bars, so the group is comparable at a glance rather
   than only rankable by one number. */
function GroupMatrix({ results }) {
  const criteria = useMemo(() => Object.keys(results[0]?.scores ?? {}), [results]);
  if (!criteria.length) return null;

  return (
    <div className="glass card">
      <h3>Every criterion, side by side</h3>
      <p className="muted small">
        The same scoring model as a solo upload, run once per walker.
      </p>
      <div className="matrix-scroll">
        <table className="matrix">
          <thead>
            <tr>
              <th scope="col">Criterion</th>
              {results.map((r, i) => (
                <th scope="col" key={i}>{r.person?.label ?? `Person ${i + 1}`}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {criteria.map((c) => {
              const vals = results.map((r) => r.scores?.[c] ?? 0);
              const top = Math.max(...vals);
              return (
                <tr key={c}>
                  <th scope="row">{label(c)}</th>
                  {vals.map((v, i) => (
                    <td key={i} className={v === top ? "lead" : ""}>
                      <span className="bar-wrap">
                        <span className={`bar q-${tier(v)}`} style={{ width: `${v * 10}%` }} />
                        <span className="bar-num">{v.toFixed(1)}</span>
                      </span>
                    </td>
                  ))}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}


/* Paste a share link instead of uploading. Same fetcher, host allow-list and
   SSRF guards as the solo path - group support costs none of that twice. The
   markup mirrors the Analyse page's link form so the two cannot drift apart
   visually. */
function LinkInput({ onSubmit, disabled }) {
  const [url, setUrl] = useState("");
  const [hosts, setHosts] = useState(null);

  useEffect(() => {
    api.sources().then(setHosts).catch(() => setHosts(null));
  }, []);

  return (
    <form
      className="glass card"
      onSubmit={(e) => {
        e.preventDefault();
        if (url.trim() && !disabled) onSubmit(url.trim());
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
          aria-label="Link to a group walk"
        />
        <button className="btn solid" type="submit" disabled={disabled || !url.trim()}>
          Analyse
        </button>
      </div>
      <div className="link-hosts">
        {(hosts?.platforms ?? ["YouTube", "Instagram", "TikTok", "Vimeo"]).join(" · ")}
        {` · up to ${Math.round((hosts?.max_duration_s ?? 300) / 60)} min`}
      </div>
      <p className="muted small">
        Everyone in the clip still needs to face the camera at least once to be
        told apart.
      </p>
    </form>
  );
}

/* Per-phase measurements for one walker. Deliberately not per-phase *scores*:
   a third of a clip is too little data for the models, and measured on spliced
   clips they detected a real mid-walk change at most 28% of the time. These are
   direct geometry. Only step width beat the camera-distance artifact reliably
   when tested, so it is the only row allowed to assert a change. */
function PhaseTable({ phases }) {
  if (!phases?.traces?.length) return null;
  const order = phases.order ?? ["approach", "middle", "exit"];

  return (
    <div className="glass card">
      <h3>Through the pass</h3>
      <p className="muted small">
        Each phase is about {phases.seconds_per_phase}s. These are measurements,
        not scores — {phases.caveat}
      </p>
      <div className="guide-table-wrap">
        <table className="guide-table compact">
          <thead>
            <tr>
              <th scope="col">Measurement</th>
              {order.map((o) => <th scope="col" key={o} className="num">{o}</th>)}
              <th scope="col" className="num">change</th>
            </tr>
          </thead>
          <tbody>
            {phases.traces.map((t) => (
              <tr key={t.label} className={t.changed ? "lead" : ""}>
                <th scope="row">
                  {t.label}
                  {t.unit ? <span className="muted tiny"> {t.unit}</span> : null}
                </th>
                {order.map((o) => (
                  <td key={o} className="num">
                    {t.per_phase?.[o] != null ? t.per_phase[o].toFixed(2) : "—"}
                  </td>
                ))}
                <td className="num">
                  {t.changed ? (
                    <strong>{t.delta > 0 ? "+" : ""}{t.delta.toFixed(2)}</strong>
                  ) : (
                    <span className="muted" title={t.can_claim
                        ? "Inside the measured artifact of the phase split"
                        : "This quantity could not be told apart from the camera artifact when tested, so no change is claimed"}>
                      {t.can_claim ? "no change" : "not claimed"}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export default function Group({ user }) {
  const { run, stages, states, elapsed, busy, partial, progress } = useAnalysisStream();
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [picked, setPicked] = useState(0);

  async function start(request) {
    setError(null);
    setResults(null);
    setPicked(0);
    try {
      const out = await request();
      // Highest first: the page's job is to rank, so the order must not depend
      // on where people happened to stand.
      setResults([...out].sort((a, b) => (b.overall ?? 0) - (a.overall ?? 0)));
    } catch (e) {
      setError(e.message);
    }
  }

  const analyse = (file) => start(() => run(() => api.analyzeGroupStream(file)));
  const analyseLink = (url) => start(() => run(() => api.analyzeGroupUrlStream(url)));

  const chosen = results?.[picked];
  const mine = results?.find((r) => r.person?.matched_user_id && user &&
                                    r.person.matched_user_id === user.id);

  return (
    <section className="stack">
      <header className="page-head">
        <h1>Group analysis</h1>
        <p className="muted">
          One clip, everyone in it scored separately. Walkers are told apart by
          face, so they stay correctly labelled even when they cross paths.
        </p>
      </header>

      {!results && !busy && (
        <>
          <DropZone onFile={analyse} disabled={busy} />
          <div className="or">paste a link</div>
          <LinkInput onSubmit={analyseLink} disabled={busy} />
        </>
      )}

      {busy && (
        <>
          <Processing stages={stages} states={states} elapsed={elapsed} />
          {progress && (
            <div className="glass card tight">
              <p className="lede">
                Found <strong>{progress.total}</strong> walker
                {progress.total === 1 ? "" : "s"} · scoring {progress.done} of{" "}
                {progress.total}
              </p>
              <p className="muted small">
                Each walker is cropped out and measured on their own pass, so
                results fill in one at a time.
              </p>
            </div>
          )}
          {partial.length > 0 && (
            <div className="person-grid">
              {[...partial]
                .sort((a, b) => (b.overall ?? 0) - (a.overall ?? 0))
                .map((r, i) => (
                  <PersonCard key={r.video_id ?? i} result={r} rank={i + 1}
                              selected={false} onSelect={() => {}} />
                ))}
            </div>
          )}
        </>
      )}

      {error && (
        <div className="glass card error">
          <h3>That did not work</h3>
          <p>{error}</p>
          <div className="btn-row">
            <button className="btn" onClick={() => setError(null)}>Try another clip</button>
            <Link className="btn" to="/guide#problems">What this means</Link>
          </div>
        </div>
      )}

      {results && (
        <>
          <div className="glass card tight">
            <p className="lede">
              Found <strong>{results.length}</strong> walker
              {results.length === 1 ? "" : "s"} with a visible face.
              {mine && <> Your walk was saved to your history.</>}
            </p>
            {results.length === 1 && (
              <p className="muted small">
                Only one face was visible throughout. Someone filmed from behind,
                or too small in frame, cannot be separated out — see How it works.
              </p>
            )}
          </div>

          <div className="person-grid">
            {results.map((r, i) => (
              <PersonCard key={i} result={r} rank={i + 1} selected={i === picked}
                          onSelect={() => setPicked(i)} />
            ))}
          </div>

          {results.length > 1 && <GroupMatrix results={results} />}

          {chosen?.phases && <PhaseTable phases={chosen.phases} />}

          {chosen?.track && (
            <div className="glass card">
              <h3>{chosen.person?.label ?? "Selected walker"}</h3>
              <p className="muted small">
                The skeleton the score was measured from. Tap another card above
                to switch walker.
              </p>
              <SkeletonPlayer
                track={chosen.track}
                timeline={chosen.timeline}
                src={chosen.origin?.video_url ? api.mediaUrl(chosen.origin.video_url) : undefined}
              />
            </div>
          )}

          <button className="btn" onClick={() => { setResults(null); setError(null); }}>
            Analyse another clip
          </button>
        </>
      )}

      {!results && !busy && (
        <div className="glass card tight">
          <h3>Filming a group</h3>
          <ul className="tips">
            <li>Have everyone walk toward the camera — faces must be visible at least once.</li>
            <li>Keep the whole body of each person in frame; feet matter for half the criteria.</li>
            <li>Six to ten seconds is enough. Longer clips take proportionally longer.</li>
            <li>Enrolled accounts are named automatically; everyone else is “Person 2”, and so on.</li>
          </ul>
        </div>
      )}
    </section>
  );
}
