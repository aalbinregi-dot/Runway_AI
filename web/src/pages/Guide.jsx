import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { LABELS } from "../criteria.js";

/* Everything on this page is the system's own behaviour, read from the API where
   possible rather than restated by hand — a guide that drifts from what the code
   does is worse than none. The numbers below (thresholds, angle bands, refusal
   limits) come from /criteria and /diagnostics. */

const SECTIONS = [
  { id: "filming", label: "Filming" },
  { id: "reading", label: "Reading a result" },
  { id: "trust", label: "How much to trust it" },
  { id: "drills", label: "Drills & progress" },
  { id: "problems", label: "If something goes wrong" },
  { id: "support", label: "Diagnostics" },
];

function Section({ id, title, children }) {
  return (
    <section className="glass card guide-section" id={id}>
      <h3>{title}</h3>
      {children}
    </section>
  );
}

function Filming() {
  return (
    <Section id="filming" title="Filming a walk">
      <ol className="guide-steps">
        <li>
          <strong>Whole body in frame, feet included.</strong> Half the criteria are
          measured from the feet and hips. If the lower body is cut off, the pose
          estimator guesses the hips and the walk is flagged as unusable rather than
          scored.
        </li>
        <li>
          <strong>Six to ten seconds.</strong> Timing and symmetry need four or more
          strides. Shorter clips are still scored but carry a warning, and anything
          under two frames of pose data is refused outright.
        </li>
        <li>
          <strong>Walk toward the camera.</strong> Phone at roughly chest height,
          held still — a moving camera reads as body sway until the drift filter
          removes it, and it cannot remove all of it.
        </li>
        <li>
          <strong>Pick the angle for what you want measured.</strong> The camera
          angle is estimated automatically and each result says which criteria that
          angle reads well.
        </li>
      </ol>

      <div className="guide-table-wrap">
        <table className="guide-table">
          <thead>
            <tr><th>Angle</th><th>Reads well</th><th>Reads poorly</th></tr>
          </thead>
          <tbody>
            <tr>
              <th scope="row">Head-on<br /><span className="muted tiny">0–20°</span></th>
              <td>Posture, balance, shoulder level, step width</td>
              <td>Stride length, arm swing — both are front-to-back and flatten</td>
            </tr>
            <tr>
              <th scope="row">Angled<br /><span className="muted tiny">20–50°</span></th>
              <td>Best single compromise — most criteria</td>
              <td>Step width, shoulder level</td>
            </tr>
            <tr>
              <th scope="row">Side-on<br /><span className="muted tiny">50–90°</span></th>
              <td>Stride length, arm swing, posture</td>
              <td>Anything left-versus-right — it cannot see across the body</td>
            </tr>
          </tbody>
        </table>
      </div>
      <p className="muted small">
        Two passes — one head-on, one at about 45° — covers everything. Filming from
        a consistent distance and angle also makes scores comparable between sessions,
        which is what the progress tracking depends on.
      </p>
    </Section>
  );
}

function Reading() {
  return (
    <Section id="reading" title="Reading a result">
      <dl className="guide-defs">
        <div>
          <dt>The overall score, 0–100</dt>
          <dd>
            A weighted sum of the twelve criteria — posture and balance carry the most
            weight, stage presence the least. Useful for tracking yourself over time,
            not for comparing against anyone else.
          </dd>
        </div>
        <div>
          <dt>Criterion scores, 0–10</dt>
          <dd>
            Where the actionable detail lives. A middling overall can hide excellent
            balance and poor stride length, and those need opposite work.
          </dd>
        </div>
        <div>
          <dt>Evidence, not just numbers</dt>
          <dd>
            Every result carries the skeleton it was measured from, the gait cycles it
            found, and twelve per-frame traces. These are the same expressions the
            features use, so the evidence and the score cannot disagree. If the
            skeleton is visibly wrong, the score is wrong — and you can see it.
          </dd>
        </div>
        <div>
          <dt>Jump to the fault</dt>
          <dd>
            Each coaching point names the moment it was worst. Clicking it seeks the
            video there, so you can check the finding rather than take it on trust.
          </dd>
        </div>
        <div>
          <dt>Recording-quality warnings</dt>
          <dd>
            Short clip, low pose confidence, missing landmarks, lower body out of
            frame, or a setup unlike anything in training. These describe the{" "}
            <em>recording</em>, not your walking, and they are worth fixing first —
            a caveated score is much less useful than a clean one.
          </dd>
        </div>
      </dl>
    </Section>
  );
}

function Trust({ criteria, thresholds }) {
  return (
    <Section id="trust" title="How much to trust a score">
      <p className="guide-lede">
        Two different things limit a score, and the result reports both.
      </p>

      <h4 className="guide-sub">1. Measurement error — how big a change is real</h4>
      <p className="muted small">
        Each criterion carries a known out-of-fold error, so a difference smaller than
        that is not evidence of anything. Walk comparisons and drill outcomes are gated
        on these, which is why a walk that scores 1.0 higher may still be reported as
        “no measurable change”. Overall needs {thresholds?.overall ?? "≈4"}/100.
      </p>
      {criteria?.length > 0 && (
        <div className="guide-table-wrap">
          <table className="guide-table compact">
            <thead>
              <tr><th>Criterion</th><th>Smallest change worth reporting</th></tr>
            </thead>
            <tbody>
              {criteria.map((c) => (
                <tr key={c.key}>
                  <th scope="row">{LABELS[c.key] ?? c.label}</th>
                  <td className="num">±{c.threshold?.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <h4 className="guide-sub">2. Identifiability — whether it measures what it names</h4>
      <p className="muted small">
        Separately, some criteria are only weakly recoverable from pose at all. Those
        are marked <span className="ident weak">weak</span> on the score list. It does
        not mean the number is meaningless, but only large differences are worth acting
        on, and the traces tell you more than the score does.{" "}
        <Link to="/about">How that was measured</Link>.
      </p>
      <p className="muted small">
        What neither number covers: agreement with a human judge. There is no
        expert-rated corpus here, so that comparison has never been made and is not
        claimed anywhere.
      </p>
    </Section>
  );
}

function Drills() {
  return (
    <Section id="drills" title="Drills and progress">
      <dl className="guide-defs">
        <div>
          <dt>Tracking a drill</dt>
          <dd>
            Each result prescribes drills for its focus areas. Choosing{" "}
            <em>Track this drill</em> freezes that criterion's current score as a
            baseline; later walks are compared against it, and a change is only called
            real once it clears that criterion's measurement error.
          </dd>
        </div>
        <div>
          <dt>Why it sometimes says “no measurable change”</dt>
          <dd>
            Because the change was smaller than the error bar. That is not the same as
            nothing happening — it means this system cannot yet tell the difference.
            Keep practising and film again.
          </dd>
        </div>
        <div>
          <dt>Trends need at least three walks</dt>
          <dd>
            Progress fits a line through <em>every</em> walk rather than subtracting the
            first from the latest, because a difference between two noisy numbers is the
            noisiest possible estimate. Where no trend is confirmed, the page says
            roughly how many walks would settle it.
          </dd>
        </div>
        <div>
          <dt>What is never claimed</dt>
          <dd>
            That a drill <em>caused</em> an improvement. Different day, different light,
            possibly different shoes, and no control group. A confirmed trend means the
            change is too large to be measurement error — not that the practice produced
            it.
          </dd>
        </div>
      </dl>
    </Section>
  );
}

function Problems() {
  const items = [
    ["“No walkers with a visible face were found”",
     "Group analysis tells people apart by face, so each walker must face the camera at least once. Someone filmed only from behind cannot be separated out."],
    ["“Only 1 frame of pose data”",
     "The file or clip has too little to measure anything that changes over time. Film 4–6 seconds of walking."],
    ["“90% of the landmark coordinates are missing”",
     "There is nothing left to measure — any score would describe the gap-filling rather than the walk. Check the file actually has x and y values, or re-record with the whole body in frame."],
    ["“The lower body does not appear to be in frame”",
     "The hips could not be located and were guessed. Step further back so the feet are visible."],
    ["“This walk sits outside the range the model was trained on”",
     "The recording setup differs from anything in training, so those scores are extrapolated. Usually distance, angle or frame rate — filming closer to a straight-on 6-second pass fixes it."],
    ["Video upload does nothing, or the camera preview stays black",
     "Video needs MediaPipe, which needs Python 3.9–3.12. Check Diagnostics below — it will say whether a video environment was found. For the camera, the browser must have been granted permission and no other app can be holding it."],
    ["A share link fails",
     "Links only work for a fixed list of hosts, and both YouTube and Instagram restrict downloading in their terms. Private accounts need a login. Download the file and upload it instead."],
  ];
  return (
    <Section id="problems" title="If something goes wrong">
      <dl className="guide-defs">
        {items.map(([q, a]) => (
          <div key={q}>
            <dt>{q}</dt>
            <dd>{a}</dd>
          </div>
        ))}
      </dl>
    </Section>
  );
}

function Diagnostics() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    api.diagnostics().then(setData).catch((e) => setError(e.message));
  }, []);

  async function copy() {
    try {
      await navigator.clipboard.writeText(data.text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      setCopied(false);
    }
  }

  return (
    <Section id="support" title="Diagnostics">
      <p className="muted small">
        What this installation can and cannot do. Each gap comes with the command that
        fixes it. Copy this into a bug report — it contains no file paths, no account
        details and nothing from your walk history.
      </p>

      {/* The most likely reason this section cannot load is the very thing it
          exists to diagnose, so it answers that instead of relaying a bare
          status line. The rest of the guide is static and still readable. */}
      {error && (
        <div className="notice">
          <span>
            The analysis service is not responding, which is itself the most
            common fault. Start it with <code>run.bat api</code> on Windows or{" "}
            <code>./run.sh api</code> on macOS and Linux — or the{" "}
            <code>web</code> command on either, for both servers — then reload
            this page.
            <br />
            <span className="muted tiny">Reported: {error}</span>
          </span>
        </div>
      )}
      {!data && !error && <p className="muted small">Checking…</p>}

      {data && (
        <>
          <p className={`diag-headline ${data.summary.healthy ? "good" : "bad"}`}>
            {data.summary.headline}
          </p>
          <ul className="diag-list">
            {data.capabilities.map((c) => (
              <li key={c.key} className={c.ok ? "ok" : (c.required ? "bad" : "off")}>
                <span className="diag-state">
                  {c.ok ? "ready" : c.required ? "needs attention" : "not installed"}
                </span>
                <span className="diag-label">{c.label}</span>
                <span className="muted tiny">{c.detail}</span>
                {c.remedy && <span className="diag-remedy">{c.remedy}</span>}
              </li>
            ))}
          </ul>

          <details className="trend-details">
            <summary>Versions</summary>
            <ul className="diag-versions">
              {Object.entries(data.versions).map(([k, v]) => (
                <li key={k}><span>{k.replace(/_/g, " ")}</span><span>{v}</span></li>
              ))}
            </ul>
          </details>

          <button className="btn" style={{ marginTop: "1rem" }} onClick={copy}>
            {copied ? "Copied" : "Copy report"}
          </button>
        </>
      )}
    </Section>
  );
}

export default function Guide() {
  const [criteria, setCriteria] = useState([]);
  const [thresholds, setThresholds] = useState(null);

  useEffect(() => {
    api.criteria()
       .then((r) => {
         setCriteria(r.criteria ?? []);
         setThresholds(r.thresholds ?? null);
       })
       .catch(() => { /* the guide is still worth reading offline */ });
  }, []);

  return (
    <section className="stack">
      <header className="page-head">
        <h1>Guide</h1>
        <p className="muted">
          How to film a walk the system can actually measure, how to read what comes
          back, and what to do when something does not work.
        </p>
      </header>

      <nav className="guide-nav" aria-label="Guide sections">
        {SECTIONS.map((s) => (
          <a key={s.id} href={`#${s.id}`}>{s.label}</a>
        ))}
      </nav>

      <Filming />
      <Reading />
      <Trust criteria={criteria} thresholds={thresholds} />
      <Drills />
      <Problems />
      <Diagnostics />
    </section>
  );
}
