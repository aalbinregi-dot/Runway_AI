/**
 * Sign in, sign up, and face enrolment.
 *
 * The app deliberately still works signed out - anonymous analysis returns a
 * full score and simply saves nothing. An account buys history, not analysis,
 * so gating the whole app behind a login would take away working behaviour to
 * add a feature.
 *
 * The session lives in an httpOnly cookie, so there is no token here to read,
 * store, or leak through an XSS bug. The only client-side state is who the
 * server says you are.
 */
import { useEffect, useRef, useState } from "react";
import { api } from "../api.js";

function Field({ label, ...props }) {
  return (
    <label style={{ display: "block", marginBottom: "0.8rem" }}>
      <span className="lab">{label}</span>
      <input className="field-input" {...props} />
    </label>
  );
}

export function AuthForm({ onSignedIn }) {
  const [mode, setMode] = useState("login");
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const signup = mode === "signup";

  async function submit(e) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const r = signup
        ? await api.signup(name, email, password)
        : await api.login(email, password);
      onSignedIn(r.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <form className="glass card" onSubmit={submit}>
      <h3>{signup ? "Create an account" : "Sign in"}</h3>
      <p style={{ margin: "0 0 1.1rem", fontSize: "0.85rem", color: "var(--ink-2)" }}>
        Keeps your walks together so you can see progress over time. You can
        analyse a walk without one — it just will not be saved.
      </p>

      {signup && (
        <Field label="Name" value={name} autoComplete="name" required
               onChange={(e) => setName(e.target.value)} />
      )}
      <Field label="Email" type="email" value={email} required
             autoComplete="email"
             onChange={(e) => setEmail(e.target.value)} />
      <Field label="Password" type="password" value={password} required
             autoComplete={signup ? "new-password" : "current-password"}
             minLength={signup ? 10 : undefined}
             onChange={(e) => setPassword(e.target.value)} />

      {signup && (
        <p style={{ margin: "-0.3rem 0 1rem", fontSize: "0.78rem", color: "var(--ink-3)" }}>
          At least 10 characters. A short phrase beats a short password.
        </p>
      )}

      {error && (
        <div className="notice" style={{ marginBottom: "1rem" }}>
          <span>{error}</span>
        </div>
      )}

      <div className="btn-row">
        <button className="btn solid" type="submit" disabled={busy}>
          {busy ? "…" : signup ? "Create account" : "Sign in"}
        </button>
        <button className="btn" type="button"
                onClick={() => { setMode(signup ? "login" : "signup"); setError(null); }}>
          {signup ? "I have an account" : "Create one instead"}
        </button>
      </div>
    </form>
  );
}

/**
 * Face enrolment.
 *
 * Webcam capture only. Accepting an uploaded photograph would make this the one
 * place in the system capable of registering somebody who never agreed to it,
 * which is precisely what the feature must not become.
 */
export function FaceEnrol({ user, onChange }) {
  const videoRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [status, setStatus] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [captured, setCaptured] = useState(0);

  useEffect(() => {
    api.faceStatus().then(setStatus).catch(() => setStatus(null));
  }, [user?.id]);

  // A live camera left running after the panel closes is a privacy problem in
  // its own right, so the tracks are stopped on unmount.
  useEffect(() => () => stream?.getTracks().forEach((t) => t.stop()), [stream]);

  // Attaching the stream has to happen *after* React has rendered the <video>.
  // Doing it inside startCamera left videoRef.current null - the element does
  // not exist until `stream` is set, and setState does not apply synchronously -
  // so srcObject was never assigned. The camera light came on and the preview
  // stayed black.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !stream) return;
    v.srcObject = stream;
    v.play().catch(() => {
      setError("The camera started but the preview would not play.");
    });
  }, [stream]);

  async function startCamera() {
    setError(null);
    if (!navigator.mediaDevices?.getUserMedia) {
      setError("This browser will not share a camera with the page. Note that "
               + "camera access needs https, or localhost.");
      return;
    }
    try {
      setStream(await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user", width: 640, height: 480 },
      }));
    } catch (err) {
      // Distinguish "you said no" from "there is no camera": the fixes differ.
      const name = err?.name || "";
      setError(
        name === "NotAllowedError"
          ? "Camera access was blocked. Allow it in the address-bar permission "
            + "icon, then try again."
          : name === "NotFoundError"
            ? "No camera was found on this device."
            : name === "NotReadableError"
              ? "The camera is in use by another app. Close it and try again."
              : `Could not open the camera (${name || "unknown error"}).`);
    }
  }

  function stopCamera() {
    stream?.getTracks().forEach((t) => t.stop());
    setStream(null);
  }

  async function capture() {
    const v = videoRef.current;
    if (!v || !v.videoWidth) return;
    setBusy(true);
    setError(null);
    try {
      const shots = [];
      // Several shots a moment apart, so the template spans small changes in
      // pose rather than freezing one instant. Measured on this system: one
      // enrolment frame gives a 0.33 worst-case match, five gives 0.75.
      for (let i = 0; i < 8; i++) {
        const c = document.createElement("canvas");
        c.width = v.videoWidth;
        c.height = v.videoHeight;
        c.getContext("2d").drawImage(v, 0, 0);
        shots.push(await new Promise((res) => c.toBlob(res, "image/jpeg", 0.92)));
        setCaptured(i + 1);
        await new Promise((r) => setTimeout(r, 220));
      }
      await api.faceEnrol(shots.filter(Boolean));
      stopCamera();
      setStatus(await api.faceStatus());
      onChange?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
      setCaptured(0);
    }
  }

  async function remove() {
    setBusy(true);
    try {
      await api.faceDelete();
      setStatus(await api.faceStatus());
      onChange?.();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  }

  if (status && !status.available) {
    return (
      <section className="glass card">
        <h3>Face matching</h3>
        <div className="notice"><span>
          Not available — the face models are not installed on the server.
        </span></div>
      </section>
    );
  }

  return (
    <section className="glass card">
      <h3>Face matching — optional</h3>
      <p style={{ margin: "0 0 1rem", fontSize: "0.85rem", color: "var(--ink-2)" }}>
        Enrol your face and the app will confirm that an uploaded walk is yours.
        It is a convenience for filing your own videos — <strong>not</strong> a
        way to sign in, and not identity verification.
      </p>
      <ul style={{ margin: "0 0 1.2rem", paddingLeft: "1.1rem",
                   fontSize: "0.82rem", color: "var(--ink-3)", lineHeight: 1.7 }}>
        <li>A numeric face signature is stored. Your photos are not.</li>
        <li>Only enrolled people are ever matched.</li>
        <li>You can delete it at any time, without deleting your account.</li>
      </ul>

      {status?.enrolled ? (
        <div className="btn-row">
          <span className="chip">Enrolled</span>
          <button className="btn" onClick={remove} disabled={busy}>
            {busy ? "…" : "Delete my face data"}
          </button>
        </div>
      ) : (
        <>
          {stream && (
            <div className="player" style={{ marginBottom: "0.9rem" }}>
              <video ref={videoRef} playsInline muted
                     style={{ transform: "scaleX(-1)" }} />
            </div>
          )}
          <div className="btn-row">
            {!stream ? (
              <button className="btn solid" onClick={startCamera} disabled={busy}>
                Turn on camera
              </button>
            ) : (
              <>
                <button className="btn solid" onClick={capture} disabled={busy}>
                  {busy ? `Capturing ${captured}/8…` : "Capture and enrol"}
                </button>
                <button className="btn" onClick={stopCamera} disabled={busy}>
                  Cancel
                </button>
              </>
            )}
          </div>
        </>
      )}

      {error && (
        <div className="notice" style={{ marginTop: "1rem" }}>
          <span>{error}</span>
        </div>
      )}
    </section>
  );
}

export default function Account({ user, onUser }) {
  if (!user) return <AuthForm onSignedIn={onUser} />;

  return (
    <>
      <section className="glass card">
        <h3>Account</h3>
        <p style={{ margin: "0 0 0.3rem", font: "400 1.15rem/1.3 var(--display)" }}>
          {user.name}
        </p>
        <p className="meta" style={{ marginTop: 0 }}>{user.email}</p>
        <div className="btn-row" style={{ marginTop: "1.1rem" }}>
          <button className="btn" onClick={async () => {
            await api.logout();
            onUser(null);
          }}>
            Sign out
          </button>
        </div>
      </section>
      <FaceEnrol user={user} onChange={() => api.me().then((r) => onUser(r.user))} />
    </>
  );
}
