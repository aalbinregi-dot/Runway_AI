/**
 * Catches render errors so a failure shows something actionable.
 *
 * React unmounts the whole tree when a render throws, which paints an entirely
 * blank page — no message, no clue, nothing to act on. That happened here after
 * a refactor left an open tab holding a stale module graph, and the only
 * symptom was white.
 *
 * A stale bundle is the most likely cause during development, so reloading is
 * offered first and the underlying error is shown rather than hidden.
 */
import { Component } from "react";

export default class ErrorBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    // Keep the real stack in the console; the panel below is the summary.
    console.error("Runway AI crashed while rendering:", error, info);
  }

  render() {
    if (!this.state.error) return this.props.children;

    return (
      <>
        <div className="field" aria-hidden="true" />
        <div className="shell">
          <header className="top">
            <h1 className="wordmark">Runway <span>ai</span></h1>
          </header>
          <section className="glass card">
            <h3>Something broke while drawing this page</h3>
            <p style={{ color: "var(--ink-2)" }}>
              If the app was updated while this tab was open, it is probably
              holding an old copy of the code. A hard reload fixes that.
            </p>
            <div className="btn-row" style={{ margin: "1.1rem 0" }}>
              <button className="btn solid"
                      onClick={() => window.location.reload(true)}>
                Reload
              </button>
              <button className="btn" onClick={() => { window.location.href = "/"; }}>
                Back to start
              </button>
            </div>
            <pre style={{
              margin: 0, padding: "0.9rem 1rem",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid var(--ink-4)", borderRadius: "var(--r-sm)",
              font: "500 0.72rem/1.6 var(--mono)", color: "var(--ink-3)",
              overflowX: "auto", whiteSpace: "pre-wrap",
            }}>
              {String(this.state.error?.message || this.state.error)}
            </pre>
          </section>
        </div>
      </>
    );
  }
}
