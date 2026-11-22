/**
 * Live processing view.
 *
 * A spinner for thirty seconds tells the user nothing — not what is happening,
 * not how far along it is, not whether it has hung. Every number here comes
 * from the server's own stage events, so the bar cannot claim progress the
 * pipeline has not actually made. The one honest exception is the decode phase,
 * where the frame count is genuinely unknown until the file has been read;
 * that is shown as an indeterminate shimmer rather than a fake percentage.
 */
import { useEffect, useRef, useState } from "react";

const COPY = {
  fetch: {
    label: "Fetching video",
    detail: "Downloading from the link",
  },
  pose: {
    label: "Tracking the body",
    detail: "MediaPipe locates 33 landmarks per frame; 13 are kept for analysis",
  },
  measure: {
    label: "Measuring the walk",
    detail: "102 gait and posture features, in torso lengths and degrees",
  },
  score: {
    label: "Scoring",
    detail: "Twelve models score it, then the rubric explains each result",
  },
};

function Tick() {
  return (
    <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden="true">
      <path d="M3 8.5l3.2 3.2L13 5" stroke="currentColor" strokeWidth="1.8"
            strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

export default function Processing({ stages, states, elapsed, filename }) {
  const list = stages?.length ? stages : ["pose", "measure", "score"];

  return (
    <div className="glass card proc">
      <div className="proc-head">
        <h3>Analysing</h3>
        <span className="proc-time">{elapsed.toFixed(0)}s</span>
      </div>
      {filename && <p className="proc-file">{filename}</p>}

      <ol className="proc-list">
        {list.map((id) => {
          const st = states[id] ?? {};
          const copy = COPY[id] ?? { label: id, detail: "" };
          const done = st.status === "done";
          const active = st.status === "active";
          const pct =
            st.total > 0 ? Math.min(100, (st.done / st.total) * 100) : null;

          return (
            <li
              key={id}
              className={`proc-step${done ? " done" : ""}${active ? " active" : ""}`}
            >
              <span className="proc-dot" aria-hidden="true">
                {done ? <Tick /> : active ? <span className="proc-spin" /> : null}
              </span>
              <div className="proc-body">
                <div className="proc-row">
                  <span className="proc-label">{copy.label}</span>
                  {active && pct !== null && (
                    <span className="proc-count">
                      {st.done}/{st.total}
                    </span>
                  )}
                  {done && st.detected != null && (
                    <span className="proc-count">
                      {st.detected}/{st.total} tracked
                    </span>
                  )}
                </div>
                {(active || done) && <p className="proc-detail">{copy.detail}</p>}
                {active && (
                  <div className={`proc-bar${pct === null ? " indet" : ""}`}>
                    <i style={pct === null ? undefined : { width: `${pct}%` }} />
                  </div>
                )}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}

/** Drive an SSE analysis and expose its stage states. */
export function useAnalysisStream() {
  // Partial results, pushed as each walker in a group finishes.
  const [partial, setPartial] = useState([]);
  const [progress, setProgress] = useState(null);
  const [stages, setStages] = useState([]);
  const [states, setStates] = useState({});
  const [elapsed, setElapsed] = useState(0);
  const [busy, setBusy] = useState(false);
  const timer = useRef(null);

  useEffect(() => () => clearInterval(timer.current), []);

  function reset() {
    setStages([]);
    setStates({});
    setElapsed(0);
    setPartial([]);
    setProgress(null);
  }

  async function run(request) {
    reset();
    setBusy(true);
    const started = Date.now();
    timer.current = setInterval(() => setElapsed((Date.now() - started) / 1000), 250);

    try {
      const res = await request();
      if (!res.ok || !res.body) {
        // The stream never opened, so the body is a normal JSON error.
        let detail = `${res.status} ${res.statusText}`;
        try {
          const b = await res.json();
          if (b?.detail) detail = b.detail;
        } catch { /* no JSON body */ }
        throw new Error(detail);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        // SSE frames are separated by a blank line; the tail may be partial.
        const frames = buffer.split("\n\n");
        buffer = frames.pop() ?? "";

        for (const frame of frames) {
          const line = frame.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const ev = JSON.parse(line.slice(6));

          if (ev.type === "start") setStages(ev.stages);
          else if (ev.type === "stage") {
            setStates((prev) => ({ ...prev, [ev.id]: { ...prev[ev.id], ...ev } }));
          } else if (ev.type === "person") {
            // One walker finished. Appending rather than replacing is what lets
            // the page fill in progressively instead of waiting for the lot.
            setPartial((prev) => [...prev, ev.result]);
            setProgress({ done: ev.done, total: ev.total });
          } else if (ev.type === "error") throw new Error(ev.message);
          else if (ev.type === "result") return ev.results ?? [];
        }
      }
      throw new Error("the connection closed before a result arrived");
    } finally {
      clearInterval(timer.current);
      setBusy(false);
    }
  }

  return { run, stages, states, elapsed, busy, partial, progress };
}
