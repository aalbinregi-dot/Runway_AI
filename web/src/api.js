/**
 * Thin client for the FastAPI backend.
 *
 * The base URL is overridable at build time so a deployed bundle can point at a
 * real host without a code change; in dev it defaults to the local uvicorn.
 */
// Default to the dev-server proxy path, so requests are same-origin and CORS
// never enters the picture during development. Set VITE_API_URL to point a
// production build at a real host.
const BASE = import.meta.env.VITE_API_URL ?? "/api";

/** Cookie-based sessions mean every request must opt into sending credentials. */
const withCreds = (opts = {}) => ({ credentials: "include", ...opts });

/** In-flight or resolved /criteria request; see `api.criteriaInfo`. */
let criteriaCache = null;

async function unwrap(res) {
  if (res.ok) return res.json();
  // FastAPI puts the useful message in `detail`. Surfacing the raw status
  // instead would hide things the user can act on, like "clip too short".
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
  } catch {
    /* response had no JSON body; keep the status line */
  }
  throw new Error(detail);
}

export const api = {
  base: BASE,

  /** Absolute URL for a server-relative media path. */
  mediaUrl(path) {
    return path?.startsWith("http") ? path : `${BASE}${path}`;
  },

  async health() {
    return unwrap(await fetch(`${BASE}/health`, withCreds()));
  },

  async diagnostics() {
    return unwrap(await fetch(`${BASE}/diagnostics`, withCreds()));
  },

  async criteria() {
    return unwrap(await fetch(`${BASE}/criteria`, withCreds()));
  },

  /** The rubric's own weights, thresholds and grade bands, fetched once.
   *
   *  Several views draw the score against these — the grade scale, the marks
   *  breakdown — and they are constants on the server, so the request is cached
   *  for the lifetime of the tab. Cached as the *promise*, not the result, so two
   *  components mounting together share one request instead of racing. */
  criteriaInfo() {
    criteriaCache ??= this.criteria().catch((err) => {
      criteriaCache = null;      // a failure must not be cached forever
      throw err;
    });
    return criteriaCache;
  },

  async analyze(file) {
    const form = new FormData();
    form.append("file", file);
    return unwrap(await fetch(`${BASE}/analyze`, withCreds({ method: "POST", body: form })));
  },

  async analyzeUrl(url) {
    return unwrap(
      await fetch(`${BASE}/analyze-url`, withCreds({
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ url }),
      })),
    );
  },

  // Streaming variants return the raw Response so the caller can read the
  // SSE body incrementally; unwrap() would buffer it to completion.
  analyzeStream(file) {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/analyze/stream`, withCreds({ method: "POST", body: form }));
  },

  async drills() {
    return unwrap(await fetch(`${BASE}/me/drills`, withCreds()));
  },

  async takeOnDrill(criterion, drillName, sessionId) {
    return unwrap(await fetch(`${BASE}/me/drills`, withCreds({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ criterion, drill_name: drillName, session_id: sessionId ?? null }),
    })));
  },

  async logPractice(assignmentId) {
    return unwrap(await fetch(`${BASE}/me/drills/${assignmentId}/practice`,
                              withCreds({ method: "POST" })));
  },

  async stopDrill(assignmentId) {
    return unwrap(await fetch(`${BASE}/me/drills/${assignmentId}`,
                              withCreds({ method: "DELETE" })));
  },

  analyzeGroupStream(file) {
    const form = new FormData();
    form.append("file", file);
    return fetch(`${BASE}/analyze-group/stream`, withCreds({ method: "POST", body: form }));
  },

  analyzeGroupUrlStream(url) {
    return fetch(`${BASE}/analyze-group-url/stream`, withCreds({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }));
  },

  analyzeUrlStream(url) {
    return fetch(`${BASE}/analyze-url/stream`, withCreds({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }));
  },

  async sources() {
    return unwrap(await fetch(`${BASE}/sources`, withCreds()));
  },

  // --- saved walks --------------------------------------------------------
  async walks() {
    return unwrap(await fetch(`${BASE}/me/walks`, withCreds()));
  },

  async walk(id) {
    return unwrap(await fetch(`${BASE}/me/walks/${id}`, withCreds()));
  },

  async deleteWalk(id) {
    return unwrap(await fetch(`${BASE}/me/walks/${id}`,
      withCreds({ method: "DELETE" })));
  },

  async compare(a, b) {
    return unwrap(await fetch(`${BASE}/me/compare?a=${a}&b=${b}`, withCreds()));
  },

  /** Plain link, so the browser's own download handling applies. */
  reportUrl(id) {
    return `${BASE}/me/walks/${id}/report.pdf`;
  },

  async progress() {
    return unwrap(await fetch(`${BASE}/me/progress`, withCreds()));
  },

  // --- accounts -----------------------------------------------------------
  async me() {
    return unwrap(await fetch(`${BASE}/auth/me`, withCreds()));
  },

  async signup(name, email, password) {
    return unwrap(await fetch(`${BASE}/auth/signup`, withCreds({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, email, password }),
    })));
  },

  async login(email, password) {
    return unwrap(await fetch(`${BASE}/auth/login`, withCreds({
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ email, password }),
    })));
  },

  async logout() {
    return unwrap(await fetch(`${BASE}/auth/logout`, withCreds({ method: "POST" })));
  },

  // --- face ---------------------------------------------------------------
  async faceStatus() {
    return unwrap(await fetch(`${BASE}/face/status`, withCreds()));
  },

  async faceEnrol(blobs) {
    const form = new FormData();
    blobs.forEach((b, i) => form.append("frames", b, `capture-${i}.jpg`));
    return unwrap(await fetch(`${BASE}/face/enrol`,
      withCreds({ method: "POST", body: form })));
  },

  async faceDelete() {
    return unwrap(await fetch(`${BASE}/face/enrol`,
      withCreds({ method: "DELETE" })));
  },

  async reportHtml(file) {
    const form = new FormData();
    form.append("file", file);
    const res = await fetch(`${BASE}/report`, { method: "POST", body: form });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.text();
  },
};

/** Map a 0-10 score to a luminance tier. The palette is monochrome, so quality
 *  is carried by brightness and weight rather than hue. */
export function tier(score) {
  if (score >= 7.5) return 1;
  if (score >= 6) return 2;
  if (score >= 4) return 3;
  return 4;
}

export const GRADE_NOTE = {
  Professional: "Runway-ready. Refine, don't rebuild.",
  Advanced: "Strong technique with a few points to sharpen.",
  Proficient: "Solid base and clear next steps.",
  Developing: "The fundamentals need attention.",
  Beginner: "Start with posture and balance.",
  Foundational: "Build the basics before adding polish.",
};
