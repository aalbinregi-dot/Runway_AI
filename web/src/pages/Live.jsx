/**
 * Live webcam coaching.
 *
 * Runs MediaPipe PoseLandmarker in the browser via WASM, so there is no server
 * round-trip and no dependency on the Python 3.11 pose environment.
 *
 * What changed, and why
 * ---------------------
 * This page used to show three posture angles per frame regardless of what the
 * camera was looking at. Standing still read "holding" on all three; walking
 * across the room, walking away from the lens, or filming an empty chair read the
 * same. The numbers were correct and the coaching was meaningless, because
 * nothing checked that a runway walk was happening.
 *
 * So the premise is checked first. `live/scene.js` keeps a rolling window of body
 * geometry and classifies the scene — nobody in frame, standing still, moving
 * without a stride rhythm, crossing the frame, turned away, walking away,
 * running, or a genuine walk toward the camera — and names the one thing to
 * change. Cues are offered only while the scene is a walk.
 *
 * What it will and will not tell you
 * ----------------------------------
 * The three instantaneous posture cues are unchanged: trunk lean, shoulder tilt
 * and head offset are genuine single-frame geometry, held behind a median filter
 * and a hysteresis band.
 *
 * The timing measurements — cadence, stride regularity, step-time asymmetry —
 * are new, and they exist because they can be honestly measured over a rolling
 * window even though they cannot be measured from one frame. They are gated on
 * having seen enough strides, and they are labelled as measurements, not scores.
 *
 * No score out of 100 is shown. The twelve models were trained on complete clips
 * with a conditioning stage — zero-phase filtering, torso normalisation,
 * Hilbert-phase gait segmentation — that a live window cannot supply.
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { CUES, L, SceneAnalyser, TH } from "../live/scene.js";
import { api, tier, GRADE_NOTE } from "../api.js";
import { short } from "../criteria.js";
import Processing, { useAnalysisStream } from "../components/Processing.jsx";

const MODEL_URL = "/mediapipe/models/pose_landmarker_lite.task";
const WASM_PATH = "/mediapipe/wasm";

/** iOS Safari does not support the GPU delegate for WASM — it crashes silently.
 *  Detect iOS and force CPU so the model still runs (slower, but works). */
const IS_IOS = /iphone|ipad|ipod/i.test(navigator.userAgent) ||
  (navigator.platform === "MacIntel" && navigator.maxTouchPoints > 1);
const MEDIAPIPE_DELEGATE = IS_IOS ? "CPU" : "GPU";

const BONES = [
  [L.leftShoulder, L.rightShoulder, "spine"], [L.leftHip, L.rightHip, "spine"],
  [L.leftShoulder, L.leftHip, "spine"], [L.rightShoulder, L.rightHip, "spine"],
  [L.leftShoulder, L.leftElbow, "left"], [L.leftElbow, L.leftWrist, "left"],
  [L.leftHip, L.leftKnee, "left"], [L.leftKnee, L.leftAnkle, "left"],
  [L.rightShoulder, L.rightElbow, "right"], [L.rightElbow, L.rightWrist, "right"],
  [L.rightHip, L.rightKnee, "right"], [L.rightKnee, L.rightAnkle, "right"],
];
const COLOUR = { left: "#f0844c", right: "#4ea8de", spine: "rgba(246,245,243,0.92)" };
const COLOUR_DIM = {
  left: "rgba(240,132,76,0.34)",
  right: "rgba(78,168,222,0.34)",
  spine: "rgba(246,245,243,0.3)",
};

/** React repaints at this rate while the canvas keeps drawing every frame.
 *  Setting state 30 times a second re-renders the whole page 30 times a second,
 *  which on a laptop already running a pose model shows up as dropped frames in
 *  the very video the user is trying to read. */
const UI_HZ = 8;

/** A second walker is only worth mentioning if the detector is reasonably sure
 *  of it — a partially visible bystander at the back of the room should not stop
 *  a session. Measured against the subject's own torso so it scales with
 *  distance. */
const RIVAL_TORSO_SHARE = 0.55;

/** Live measurements, with what each needs before it means anything. `strides`
 *  is the number of same-foot intervals required: a cadence needs two step
 *  events, a spread needs several. */
const LIVE_METRICS = [
  { id: "cadence", label: "Cadence", unit: "spm", steps: 2,
    format: (m) => Math.round(m.cadence),
    note: "steps per minute over the last few seconds" },
  { id: "strideCv", label: "Stride spread", unit: "cv", steps: 5,
    format: (m) => m.strideCv.toFixed(3),
    note: "how much stride time varies; steady walking sits near 0.03" },
  { id: "asymmetry", label: "Step-time L/R", unit: "index", steps: 4,
    format: (m) => m.asymmetry.toFixed(3),
    note: "difference between left and right step times; 0 is even" },
  { id: "sway", label: "Lateral sway", unit: "torso", steps: 2,
    format: (m) => m.sway.toFixed(3),
    note: "side-to-side travel of the pelvis, in torso lengths" },
  { id: "stepWidth", label: "Step width", unit: "torso", steps: 2,
    format: (m) => m.stepWidth.toFixed(3),
    note: "ankle separation; runway walking places each foot near one line" },
  { id: "facing", label: "Camera angle", unit: "°", steps: 0,
    format: (m) => Math.round(m.facing),
    note: "shoulder-line foreshortening, +/- 8 degrees" },
];

export default function Live() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const landmarkerRef = useRef(null);
  const rafRef = useRef(0);
  const sceneRef = useRef(new SceneAnalyser());
  const lastUiRef = useRef(0);
  const mirrorRef = useRef(true);

  const isCoachingRef = useRef(false);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);

  const [status, setStatus] = useState("idle");   // idle|loading|ready|error
  const [isCoaching, setIsCoaching] = useState(false);
  const [error, setError] = useState(null);
  const [facingMode, setFacingMode] = useState("user");
  const [reading, setReading] = useState(null);
  const [detail, setDetail] = useState(false);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState(null);
  const [recordedDuration, setRecordedDuration] = useState(0);
  const recordStartRef = useRef(0);

  // Analysis state
  const stream = useAnalysisStream();
  const [analysisResult, setAnalysisResult] = useState(null);
  const [analysisError, setAnalysisError] = useState(null);
  const navigate = useNavigate();

  const mirrored = facingMode === "user";

  // Keep refs in sync for the animation frame loop
  useEffect(() => { mirrorRef.current = mirrored; }, [mirrored]);
  useEffect(() => { isCoachingRef.current = isCoaching; }, [isCoaching]);

  const startCoaching = useCallback(() => {
    sceneRef.current.reset();
    isCoachingRef.current = true;
    setIsCoaching(true);
    setReading(null);
    // Clear previous recording when starting fresh coaching
    setRecordedBlob(null);
    setAnalysisResult(null);
    setAnalysisError(null);
  }, []);

  const stopCoaching = useCallback(() => {
    isCoachingRef.current = false;
    setIsCoaching(false);
    // Also stop recording if active
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
  }, []);

  const startRecording = useCallback(() => {
    const v = videoRef.current;
    if (!v?.srcObject) return;
    chunksRef.current = [];
    const mimeType = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
      ? "video/webm;codecs=vp9"
      : MediaRecorder.isTypeSupported("video/webm")
        ? "video/webm"
        : "video/mp4";
    const rec = new MediaRecorder(v.srcObject, { mimeType });
    rec.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    rec.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mimeType });
      const dur = (performance.now() - recordStartRef.current) / 1000;
      setRecordedBlob(blob);
      setRecordedDuration(dur);
      setIsRecording(false);
      recorderRef.current = null;
    };
    rec.start(500); // collect chunks every 500 ms
    recorderRef.current = rec;
    recordStartRef.current = performance.now();
    setIsRecording(true);
    setRecordedBlob(null);
    setAnalysisResult(null);
    setAnalysisError(null);
  }, []);

  const stopRecording = useCallback(() => {
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
  }, []);

  const stop = useCallback(() => {
    cancelAnimationFrame(rafRef.current);
    isCoachingRef.current = false;
    setIsCoaching(false);
    if (recorderRef.current && recorderRef.current.state === "recording") {
      recorderRef.current.stop();
    }
    const v = videoRef.current;
    v?.srcObject?.getTracks?.().forEach((t) => t.stop());
    if (v) v.srcObject = null;
    setStatus("idle");
    setReading(null);
  }, []);

  useEffect(() => stop, [stop]);

  async function analyseClip() {
    if (!recordedBlob) return;
    setAnalysisError(null);
    setAnalysisResult(null);
    const ext = recordedBlob.type.includes("mp4") ? "mp4" : "webm";
    const file = new File([recordedBlob], `live-walk.${ext}`, { type: recordedBlob.type });
    try {
      const results = await stream.run(() => api.analyzeStream(file));
      setAnalysisResult(results);
    } catch (e) {
      setAnalysisError(e.message);
    }
  }

  function openFullAnalysis() {
    if (!analysisResult || !recordedBlob) return;
    const ext = recordedBlob.type.includes("mp4") ? "mp4" : "webm";
    const file = new File([recordedBlob], `live-walk.${ext}`, { type: recordedBlob.type });
    // Navigate to analyse page carrying the result + file via state so it renders
    // immediately without re-uploading.
    navigate("/", { state: { results: analysisResult, videoFile: file } });
  }

  async function start(mode) {
    setError(null);
    setStatus("loading");
    setIsCoaching(false);
    isCoachingRef.current = false;
    sceneRef.current.reset();

    // On iOS only Safari supports getUserMedia — Chrome and Firefox on iOS
    // use the same WebKit engine but Apple blocks camera access in them.
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        IS_IOS
          ? "Camera access requires Safari on iOS. Please open this page in Safari."
          : "Your browser does not support camera access."
      );
      setStatus("error");
      return;
    }

    try {
      const { FilesetResolver, PoseLandmarker } = await import("@mediapipe/tasks-vision");
      if (!landmarkerRef.current) {
        // Served from public/, not a CDN, so the page works offline and does not
        // leak a request to a third party every time somebody opens it.
        const fileset = await FilesetResolver.forVisionTasks(WASM_PATH);
        landmarkerRef.current = await PoseLandmarker.createFromOptions(fileset, {
          // iOS Safari crashes with the GPU delegate (limited WebGL/WebGPU WASM
          // support). CPU is slower but runs reliably on all platforms.
          baseOptions: { modelAssetPath: MODEL_URL, delegate: MEDIAPIPE_DELEGATE },
          runningMode: "VIDEO",
          // Two, not one: with numPoses at 1 the detector silently switches
          // between people when a second walker is in shot, and the jump lands
          // in the traces as sway and asymmetry that neither person has. Asking
          // for two lets the page notice and say so.
          numPoses: 2,
        });
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: mode, width: 960, height: 720 },
      });
      const v = videoRef.current;
      v.srcObject = stream;
      // iOS requires the play() call to originate from a user-gesture context.
      // Since start() is always called from a button click, this is satisfied;
      // adding the catch here prevents an unhandled rejection if the element
      // is removed before playback begins (e.g. fast page navigation).
      await v.play().catch(() => {});
      setStatus("ready");
      rafRef.current = requestAnimationFrame(loop);
    } catch (err) {
      const name = err?.name || "";
      setError(
        name === "NotAllowedError" ? "Camera access was blocked. Allow it in Safari Settings → Camera, then try again."
          : name === "NotFoundError" ? "No camera was found on this device."
          : name === "NotReadableError" ? "The camera is in use by another app."
          : name === "NotSupportedError" ? "Camera not supported. Make sure you are using Safari on iOS."
          : `Could not start: ${err?.message || name || "unknown error"}`);
      setStatus("error");
    }
  }

  function switchCamera() {
    const next = facingMode === "user" ? "environment" : "user";
    stop();
    setFacingMode(next);
    start(next);
  }

  function loop() {
    rafRef.current = requestAnimationFrame(loop);
    const v = videoRef.current;
    const c = canvasRef.current;
    const lm = landmarkerRef.current;
    if (!v || !c || !lm || v.readyState < 2 || !v.videoWidth) return;

    const res = lm.detectForVideo(v, performance.now());
    const poses = res?.landmarks ?? [];

    // The subject is the largest body in frame, not the first one the detector
    // happened to return: ordering is not stable between frames, and following
    // it would make the traces jump between people.
    const torsoOf = (p) => Math.hypot(
      (p[L.leftShoulder].x + p[L.rightShoulder].x) / 2 - (p[L.leftHip].x + p[L.rightHip].x) / 2,
      (p[L.leftShoulder].y + p[L.rightShoulder].y) / 2 - (p[L.leftHip].y + p[L.rightHip].y) / 2,
    );
    const ranked = poses
      .map((p) => ({ points: p, torso: torsoOf(p) }))
      .sort((a, b) => b.torso - a.torso);
    const subject = ranked[0] ?? null;
    const people = subject
      ? 1 + ranked.slice(1).filter((r) => r.torso >= subject.torso * RIVAL_TORSO_SHARE).length
      : 0;

    const aspect = v.videoHeight / v.videoWidth;
    const analyser = sceneRef.current;

    let snapshot = null;
    if (isCoachingRef.current) {
      analyser.push(subject?.points ?? null, performance.now() / 1000, aspect, people);

      const now = performance.now();
      if (now - lastUiRef.current > 1000 / UI_HZ) {
        lastUiRef.current = now;
        snapshot = analyser.read();
        setReading(snapshot);
      }
    }

    const isWalking = isCoachingRef.current && (snapshot?.scene?.isWalk ?? analyser.scene === "walk");
    draw(c, v, subject?.points ?? null, ranked.slice(1), isWalking, mirrorRef.current);
  }

  const r = reading;
  const scene = r?.scene;
  const walking = isCoaching && (scene?.isWalk ?? false);

  return (
    <>
      <div className="page-head">
        <h1>Live coaching</h1>
        <p style={{ color: "var(--ink-2)" }}>
          Stand back until your whole body, feet included, is in frame, then walk
          toward the camera. The page checks that what it can see is a runway
          walk before it says anything about your posture.
        </p>
      </div>

      <section className="glass card">
        <div className="player" style={{ background: "#000" }}>
          <video ref={videoRef} playsInline muted
                 style={{ transform: mirrored ? "scaleX(-1)" : "none", width: "100%",
                          maxHeight: "58vh", objectFit: "contain", display: "block" }} />
          <canvas ref={canvasRef} style={{ position: "absolute", inset: 0 }} />

          {status === "ready" && (
            isCoaching ? (
              <div className={`scene-badge active${walking ? " on" : ""}`}>
                <span className="scene-dot" />
                <span>{scene ? scene.title : "Analysing…"}</span>
                {scene && <span className="scene-since">{scene.since.toFixed(0)}s</span>}
              </div>
            ) : (
              <div className="scene-badge paused">
                <span className="scene-dot off" />
                <span>{reading ? "Coaching paused" : "Camera ready · Coaching stopped"}</span>
              </div>
            )
          )}

          {status !== "ready" && (
            <div className="live-overlay">
              {status === "loading"
                ? <><span className="pulse" /><p>Loading the pose model (about 6 MB, once)</p></>
                : <button className="btn solid" onClick={() => start(facingMode)}>
                    Start camera
                  </button>}
            </div>
          )}
        </div>

        {status === "ready" && (
          <div className="btn-row" style={{ marginTop: "0.9rem" }}>
            {!isCoaching ? (
              <button className="btn solid btn-start" onClick={startCoaching}>
                ▶ {reading ? "Resume coaching" : "Start coaching"}
              </button>
            ) : (
              <button className="btn btn-stop" onClick={stopCoaching}>
                ⏹ Stop coaching
              </button>
            )}
            {isCoaching && (
              isRecording ? (
                <button className="btn btn-rec-stop" onClick={stopRecording}>
                  ⏺ Stop recording
                </button>
              ) : (
                <button className="btn btn-rec" onClick={startRecording}>
                  ⏺ Record clip
                </button>
              )
            )}
            <button className="btn" onClick={stop}>
              Turn off camera
            </button>
            <button className="btn" onClick={switchCamera}>
              {mirrored ? "Use back camera" : "Use front camera"}
            </button>
            <button className="btn" onClick={() => setDetail((d) => !d)}>
              {detail ? "Hide numbers" : "Show numbers"}
            </button>
          </div>
        )}

        {error && (
          <div className="notice" style={{ marginTop: "1rem" }}><span>{error}</span></div>
        )}
      </section>

      {status === "ready" && (
        <>
          {!isCoaching && !reading && (
            <section className="glass card" style={{ marginTop: "var(--gap)", textAlign: "center", padding: "2.2rem 1.5rem" }}>
              <h3 style={{ margin: 0, fontSize: "1.15rem" }}>Camera Ready · Coaching Stopped</h3>
              <p style={{ maxWidth: "52ch", margin: "0.75rem auto 1.4rem", color: "var(--ink-2)", fontSize: "0.9rem" }}>
                Stand back until your full body is visible in the preview above. When you are positioned at the start of your runway and ready to walk, press <strong>Start coaching</strong>.
              </p>
              <button className="btn solid btn-start" onClick={startCoaching} style={{ padding: "0.9rem 2rem" }}>
                ▶ Start coaching
              </button>
            </section>
          )}

          {reading && scene && (
            <>
              {!isCoaching && (
                <div className="notice" style={{ marginTop: "var(--gap)", borderLeft: "3px solid var(--ink-3)" }}>
                  <span><strong>Coaching stopped.</strong> Showing data from your last run. Click <strong>Resume coaching</strong> to start another walk pass.</span>
                </div>
              )}
              <Verdict scene={scene} pending={r.pending} people={r.people} />
              <div className="split" style={{ marginTop: "var(--gap)" }}>
                <div>
                  <Checklist checks={r.checks} ready={r.ready} />
                  <Cues cues={r.cues} walking={walking} />
                </div>
                <div>
                  <Rhythm series={r.series} metrics={r.metrics} walking={walking} />
                  <Metrics metrics={r.metrics} walking={walking} detail={detail} />
                  <Passes passes={r.passes} live={r.live} />
                </div>
              </div>
            </>
          )}
        </>
      )}

      {/* ── Recorded clip + analysis ─────────────────────────────────────── */}
      {(isRecording || recordedBlob || stream.busy || analysisResult || analysisError) && (
        <RecordPanel
          isRecording={isRecording}
          recordedBlob={recordedBlob}
          recordedDuration={recordedDuration}
          onStopRecording={stopRecording}
          onAnalyse={analyseClip}
          onFullAnalysis={openFullAnalysis}
          stream={stream}
          analysisResult={analysisResult}
          analysisError={analysisError}
          onRetry={() => { setAnalysisResult(null); setAnalysisError(null); }}
        />
      )}
    </>
  );
}

/* ------------------------------------------------------------------------- *
 * Drawing
 * ------------------------------------------------------------------------- */

/**
 * Skeleton, plus two things a mirror cannot show you: your own centre line and
 * where your feet actually land relative to it.
 *
 * The skeleton dims when the scene is not a walk. That is deliberate feedback in
 * itself — a bright, confident overlay while the system has decided it cannot
 * read the walk is the same mixed message the page was written to remove.
 */
function draw(canvas, video, points, others, walking, mirrored) {
  const rect = video.getBoundingClientRect();
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    canvas.style.width = `${rect.width}px`;
    canvas.style.height = `${rect.height}px`;
  }
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!points) return;

  // The video is letterboxed inside its box by object-fit: contain, so the drawn
  // area is not the element's full rect.
  const scale = Math.min(rect.width / video.videoWidth, rect.height / video.videoHeight);
  const w = video.videoWidth * scale;
  const h = video.videoHeight * scale;
  const ox = (rect.width - w) / 2;
  const oy = (rect.height - h) / 2;
  // The preview is mirrored for a front camera; the skeleton must be mirrored
  // too or it will sit on the wrong side of the body.
  const at = (p) => [ox + (mirrored ? 1 - p.x : p.x) * w, oy + p.y * h];
  const P = (i) => at(points[i]);

  const palette = walking ? COLOUR : COLOUR_DIM;

  // Anyone else in shot, drawn faintly: seeing the second skeleton is what makes
  // "more than one person in frame" believable rather than a mystery.
  ctx.lineWidth = Math.max(rect.width * 0.003, 1);
  ctx.strokeStyle = "rgba(246,245,243,0.16)";
  for (const other of others) {
    for (const [a, b] of BONES) {
      const pa = other.points[a];
      const pb = other.points[b];
      if ((pa?.visibility ?? 1) < 0.4 || (pb?.visibility ?? 1) < 0.4) continue;
      const [ax, ay] = at(pa);
      const [bx, by] = at(pb);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    }
  }

  ctx.lineCap = "round";
  ctx.lineWidth = Math.max(rect.width * 0.006, 2);
  for (const [a, b, kind] of BONES) {
    if ((points[a]?.visibility ?? 1) < 0.4 || (points[b]?.visibility ?? 1) < 0.4) continue;
    ctx.strokeStyle = palette[kind];
    ctx.beginPath();
    const [ax, ay] = P(a);
    const [bx, by] = P(b);
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  }

  // Centre line through the pelvis, and a tick per ankle on a baseline: a live
  // read of foot placement against your own midline, which is the thing runway
  // walking is judged on and the one thing a mirror cannot show you.
  const hipMid = {
    x: (points[L.leftHip].x + points[L.rightHip].x) / 2,
    y: (points[L.leftHip].y + points[L.rightHip].y) / 2,
  };
  const [cx] = at(hipMid);
  const baseY = oy + h - Math.max(h * 0.02, 6);
  ctx.setLineDash([4, 6]);
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgba(246,245,243,0.28)";
  ctx.beginPath();
  ctx.moveTo(cx, oy + h * 0.35);
  ctx.lineTo(cx, baseY);
  ctx.stroke();
  ctx.setLineDash([]);

  for (const [side, index] of [["left", L.leftAnkle], ["right", L.rightAnkle]]) {
    const p = points[index];
    if ((p?.visibility ?? 1) < 0.4) continue;
    const [x] = at(p);
    ctx.strokeStyle = side === "left" ? COLOUR.left : COLOUR.right;
    ctx.lineWidth = Math.max(rect.width * 0.004, 2);
    ctx.beginPath();
    ctx.moveTo(x, baseY - Math.max(h * 0.018, 5));
    ctx.lineTo(x, baseY);
    ctx.stroke();
  }
}

/* ------------------------------------------------------------------------- *
 * Panels
 * ------------------------------------------------------------------------- */

function Verdict({ scene, pending, people }) {
  return (
    <section className={`glass card verdict${scene.isWalk ? " good" : ""}`}>
      <h3>What the camera sees</h3>
      <p className="verdict-title">{scene.title}</p>
      <p className="verdict-detail">{scene.detail}</p>
      {scene.fix && <p className="verdict-fix">{scene.fix}</p>}
      <div className="verdict-foot">
        <span>{scene.since.toFixed(0)}s in this state</span>
        {pending && <span>settling toward “{pending.title}”</span>}
        {people > 1 && <span>{people} people detected</span>}
        {scene.id === "crowded" && (
          <Link to="/group" className="verdict-link">Group analysis</Link>
        )}
      </div>
    </section>
  );
}

function Checklist({ checks, ready }) {
  const passed = checks.filter((c) => c.ok).length;
  return (
    <section className="glass card">
      <h3>Ramp-walk check · {passed}/{checks.length}</h3>
      <ul className="checks">
        {checks.map((c) => (
          <li key={c.id} className={c.ok ? "ok" : "no"}>
            <span className="check-mark" aria-hidden="true">{c.ok ? "✓" : "✕"}</span>
            <span className="check-label">{c.label}</span>
            <span className="check-detail">{c.detail}</span>
          </li>
        ))}
      </ul>
      {!ready && (
        <p className="muted tiny" style={{ marginTop: "0.8rem" }}>
          Still filling the window — it needs a second or two of movement, and at
          least {TH.steps} steps, before any of this is worth reading.
        </p>
      )}
    </section>
  );
}

function Cues({ cues, walking }) {
  return (
    <section className={`glass card${walking ? "" : " paused"}`}>
      <h3>Posture cues {walking ? "" : "· paused"}</h3>
      {!walking && (
        <p className="muted tiny" style={{ margin: "0 0 0.9rem" }}>
          Held while the scene is not a walk. These three are real single-frame
          geometry, but a trunk angle measured while you turn round or stand still
          is not coaching.
        </p>
      )}
      <div className="cues">
        {CUES.map((cue) => {
          const c = cues[cue.id];
          const off = walking && c?.state === "off";
          return (
            <div className={`cue${off ? " off" : ""}`} key={cue.id}>
              <div className="cue-row">
                <span className="cue-label">{cue.label}</span>
                <span className="cue-state">
                  {!walking ? "—" : off ? "adjust" : "holding"}
                  {walking && c ? ` · ${c.value.toFixed(cue.unit === "°" ? 0 : 2)}${cue.unit}` : ""}
                </span>
              </div>
              <div className="cue-bar">
                <i style={{ width: `${walking
                  ? Math.min(100, ((c?.value ?? 0) / (cue.warn * 1.6)) * 100) : 0}%` }} />
                <span className="cue-mark"
                      style={{ left: `${(cue.good / (cue.warn * 1.6)) * 100}%` }} />
              </div>
              {off && <p className="cue-hint">{cue.hint}</p>}
            </div>
          );
        })}
      </div>
    </section>
  );
}

/**
 * The gait signal itself: the lateral gap between the ankles, with a mark at
 * every step the trigger found.
 *
 * This is the evidence for the scene verdict. "Not walking" is an assertion; a
 * flat line with no step marks on it is a demonstration, and a clean alternating
 * wave is the user seeing their own rhythm for the first time.
 */
function Rhythm({ series, metrics, walking }) {
  const path = series?.points?.length
    ? series.points.map(([x, y], i) =>
        `${i ? "L" : "M"}${(x * 100).toFixed(2)},${(15 - y * 12).toFixed(2)}`).join(" ")
    : null;

  return (
    <section className="glass card">
      <h3>Stride signal</h3>
      <div className="rhythm-plot">
        <svg viewBox="0 0 100 30" preserveAspectRatio="none" aria-hidden="true">
          <line x1="0" y1="15" x2="100" y2="15" stroke="rgba(246,245,243,0.16)"
                strokeWidth="0.4" vectorEffect="non-scaling-stroke" />
          {path && (
            <path d={path} fill="none" vectorEffect="non-scaling-stroke"
                  stroke={walking ? "rgba(246,245,243,0.9)" : "rgba(246,245,243,0.4)"}
                  strokeWidth="1" strokeLinejoin="round" />
          )}
        </svg>
        {(series?.steps ?? []).map((s, i) => (
          <span key={i} className={`rhythm-step ${s.dir > 0 ? "left" : "right"}`}
                style={{ left: `${s.x * 100}%` }} />
        ))}
      </div>
      <div className="tl-legend">
        <span>{metrics.steps} steps in {Math.min(4, Math.round(metrics.n / Math.max(metrics.fps, 1)))}s</span>
        <span>rhythm {metrics.periodicity.toFixed(2)} / 1.00</span>
        <span>amplitude {(series?.amplitude ?? 0).toFixed(2)} torso</span>
      </div>
      <p className="muted tiny" style={{ marginTop: "0.7rem" }}>
        Ankle separation with the slow drift removed. One crossing of the centre
        line is one step; orange marks the left foot taking the lead, blue the
        right. A walk makes a regular alternating wave — anything else is not gait,
        which is how the check above decides.
      </p>
    </section>
  );
}

function Metrics({ metrics, walking, detail }) {
  return (
    <section className="glass card">
      <h3>Measured live</h3>
      <div className="live-metrics">
        {LIVE_METRICS.map((m) => {
          // The camera angle needs a body, not a walk — it is the one reading
          // that is still true while you stand still, and the one people check
          // when setting the camera up. Everything else needs steps, and the
          // window needs to hold enough frames for any of it to mean anything.
          const enough = metrics.n > 8 && metrics.steps >= m.steps;
          const value = enough && (walking || m.id === "facing") ? m.format(metrics) : "—";
          return (
            <div className="live-metric" key={m.id}>
              <span className="live-metric-value">
                {value}
                {value !== "—" && m.unit !== "index" && m.unit !== "cv" && (
                  <small>{m.unit}</small>
                )}
              </span>
              <span className="live-metric-label">{m.label}</span>
              {detail && <span className="live-metric-note">{m.note}</span>}
              {!enough && (
                <span className="live-metric-note">
                  {m.steps > 0 ? `needs ${m.steps} steps` : "needs a body in frame"}
                </span>
              )}
            </div>
          );
        })}
      </div>
      <p className="live-caveat">
        Measurements, not scores. They use the same definitions as the full
        analysis — torso lengths, degrees, steps per minute — but over a
        four-second rolling window with a moving-mean trend removal rather than the
        whole clip and a zero-phase filter, so read them as a live check, not a
        result. Use <strong>Record clip</strong> above to capture a walk for full scores,
        stride length, arm swing and a coaching plan.
      </p>
    </section>
  );
}

function Passes({ passes, live }) {
  if (!passes.length && !live) return null;
  return (
    <section className="glass card">
      <h3>This session</h3>
      {live && (
        <p className="pass-live">
          Pass {live.index} in progress · {live.seconds.toFixed(0)}s · {live.steps} steps
        </p>
      )}
      <ul className="passes">
        {passes.map((p) => {
          const worst = Object.entries(p.off).sort((a, b) => b[1] - a[1])[0];
          const cue = CUES.find((c) => c.id === worst?.[0]);
          return (
            <li key={p.index}>
              <span className="pass-index">{String(p.index).padStart(2, "0")}</span>
              <span className="pass-main">
                {p.seconds.toFixed(0)}s · {p.steps} steps · {Math.round(p.cadence)} spm
              </span>
              <span className="pass-note">
                {worst && worst[1] > 0.15
                  ? `${cue?.label ?? worst[0]} off for ${Math.round(worst[1] * 100)}%`
                  : "cues held throughout"}
              </span>
            </li>
          );
        })}
      </ul>
      <p className="muted tiny" style={{ marginTop: "0.8rem" }}>
        A pass is one continuous walk toward the camera, at least two seconds
        long. Nothing here is saved — close the tab and it is gone.
      </p>
    </section>
  );
}

/* ------------------------------------------------------------------------- *
 * Record → Analyse panel
 * ------------------------------------------------------------------------- */

function RecordPanel({
  isRecording, recordedBlob, recordedDuration,
  onStopRecording, onAnalyse, onFullAnalysis,
  stream, analysisResult, analysisError, onRetry,
}) {
  const blobUrl = recordedBlob ? URL.createObjectURL(recordedBlob) : null;
  const r0 = analysisResult?.[0] ?? null;

  return (
    <section className="glass card rec-panel" style={{ marginTop: "var(--gap)" }}>
      <h3>Recorded clip</h3>

      {/* ── Recording in progress ──────────────────────────────────────── */}
      {isRecording && (
        <div className="rec-live">
          <span className="rec-dot" aria-hidden="true" />
          <span>Recording in progress…</span>
          <button className="btn btn-rec-stop" onClick={onStopRecording} style={{ marginLeft: "auto" }}>
            ⏺ Stop recording
          </button>
        </div>
      )}

      {/* ── Clip ready ─────────────────────────────────────────────────── */}
      {recordedBlob && !isRecording && (
        <div className="rec-ready">
          {blobUrl && (
            <video
              src={blobUrl}
              controls
              playsInline
              className="rec-preview"
              style={{ width: "100%", maxHeight: "40vh", objectFit: "contain",
                       display: "block", borderRadius: "var(--r-sm)", background: "#000" }}
            />
          )}
          <div className="rec-meta">
            <span>{(recordedBlob.size / 1e6).toFixed(1)} MB · {recordedDuration.toFixed(0)}s</span>
            {blobUrl && (
              <a href={blobUrl}
                 download="live-walk.webm"
                 className="btn small"
                 style={{ marginLeft: "auto" }}
              >Download clip</a>
            )}
          </div>

          {!analysisResult && !stream.busy && !analysisError && (
            <div className="rec-actions">
              <button className="btn solid btn-analyse" onClick={onAnalyse}>
                ✨ Analyse this walk
              </button>
              <p className="muted tiny" style={{ margin: "0.6rem 0 0" }}>
                Runs the full 12-criterion pipeline on the recorded clip.
              </p>
            </div>
          )}
        </div>
      )}

      {/* ── Analysis running ───────────────────────────────────────────── */}
      {stream.busy && (
        <Processing
          stages={stream.stages}
          states={stream.states}
          elapsed={stream.elapsed}
          filename="live-walk"
        />
      )}

      {/* ── Analysis error ─────────────────────────────────────────────── */}
      {analysisError && !stream.busy && (
        <div>
          <div className="notice" style={{ marginTop: "0.8rem" }}>
            <span>{analysisError}</span>
          </div>
          <button className="btn" onClick={onRetry} style={{ marginTop: "0.8rem" }}>Try again</button>
        </div>
      )}

      {/* ── Condensed result ───────────────────────────────────────────── */}
      {r0 && !stream.busy && (
        <div className="rec-result">
          <div className="rec-score">
            <span className="rec-numeral">{r0.overall.toFixed(1)}<small>/100</small></span>
            <div>
              <p className="rec-grade">{r0.grade}</p>
              <p className="rec-grade-note" style={{ color: "var(--ink-2)", fontSize: "0.85rem", margin: 0 }}>
                {GRADE_NOTE[r0.grade] ?? ""}
              </p>
            </div>
          </div>

          {r0.coaching?.focus_areas?.length > 0 && (
            <div className="rec-focus">
              <h4 style={{ margin: "1.2rem 0 0.6rem", fontSize: "0.8rem",
                           textTransform: "uppercase", letterSpacing: "0.1em",
                           color: "var(--ink-3)" }}>Top focus areas</h4>
              {r0.coaching.focus_areas.slice(0, 3).map((f) => (
                <div key={f.criterion} className="rec-focus-item">
                  <span className={`rec-score-pill q-${tier(f.score)}`}>{f.score}/10</span>
                  <div>
                    <strong>{short(f.criterion)}</strong>
                    <p style={{ margin: "0.2rem 0 0", fontSize: "0.82rem", color: "var(--ink-2)" }}>{f.issue}</p>
                  </div>
                </div>
              ))}
            </div>
          )}

          <div className="rec-actions" style={{ marginTop: "1.2rem" }}>
            <button className="btn solid" onClick={onFullAnalysis}>
              See full analysis
            </button>
            <button className="btn" onClick={onRetry} style={{ marginLeft: "0.6rem" }}>
              Analyse again
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
