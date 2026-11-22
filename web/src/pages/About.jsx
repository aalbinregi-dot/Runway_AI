import { api } from "../api.js";

export default function About({ health }) {
  return (
    <>
      <section className="glass card">
        <h3>How the scoring works</h3>
        <div className="focus">
          {[
            ["Pose estimation", "MediaPipe Pose returns 33 landmarks per frame; 13 are kept for analysis."],
            ["Conditioning", "Gaps interpolated, tracks smoothed, then normalised so the pelvis is the origin and torso length the unit — a walk scores the same filmed from three metres or eight."],
            ["Measurement", "About 100 gait and posture features in torso lengths and degrees. A zero-phase high-pass keeps the camera's approach from being read as body sway."],
            ["Scoring", "Twelve regressors, each choosing its own estimator by out-of-fold error."],
            ["Explanation", "A transparent biomechanical rubric identifies which measurement caused a low score, and the coaching is built from that."],
          ].map(([title, body], i) => (
            <div className="focus-item" key={title}>
              <span className="rank">{String(i + 1).padStart(2, "0")}</span>
              <div>
                <h4>{title}</h4>
                <p>{body}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="glass card">
        <h3>Filming for a reliable score</h3>
        <div className="chips">
          {["Whole body in frame", "6–10 seconds", "Walk toward the camera",
            "Chest height, 5–8 m away", "Even lighting", "Plain background"].map((c) => (
            <span className="chip" key={c}>{c}</span>
          ))}
        </div>
      </section>

      <section className="glass card">
        <h3>Why some scores are marked “weak”</h3>
        <p style={{ color: "var(--ink-2)", fontSize: "0.89rem", margin: "0 0 0.7rem" }}>
          The training corpus gives each walker a shared ability level, because real
          walkers are like that — good posture and good balance co-occur. The side
          effect is that a model can score one criterion by reading another’s
          features: predict rhythm from postural sway, because sway indicates general
          ability and general ability indicates rhythm. On such data that shortcut
          looks like accuracy.
        </p>
        <p style={{ color: "var(--ink-2)", fontSize: "0.89rem", margin: "0 0 0.7rem" }}>
          So the whole evaluation was re-run on a second 1,500-clip corpus with the
          twelve qualities varied <em>independently</em>, where cross-criterion
          shortcuts are worthless. Average R² fell from 0.72 to 0.60. Foot placement,
          shoulder alignment, confidence, head position and balance were unaffected —
          they were always measuring themselves. Rhythm fell from 0.53 to 0.17, hip
          movement from 0.53 to 0.23, symmetry from 0.52 to 0.28.
        </p>
        <p style={{ color: "var(--ink-2)", fontSize: "0.89rem", margin: 0 }}>
          Those three are marked <strong>weak</strong> on the score list. It does not
          mean the number is meaningless — rhythm still tracks rhythm — but only large
          differences are worth acting on, and the evidence traces tell you more than
          the score does. Two limits worth stating: this measures identifiability
          against synthetic ground truth, not agreement with human judges, and no
          expert-rated corpus exists here to provide the latter.
        </p>
      </section>

      <section className="glass card">
        <h3>Tracking a drill</h3>
        <p style={{ color: "var(--ink-2)", fontSize: "0.89rem", margin: "0 0 0.7rem" }}>
          Taking a drill on freezes that criterion's current score as a baseline. Later
          walks are compared against it — but a change is only called real when it
          exceeds this system's own measurement error on that criterion, which is
          1.41 × the out-of-fold MAE measured during training (about ±1.5 for posture).
          Below that, the honest answer is “no measurable change”, and it is worded so
          it cannot be read as “nothing changed”.
        </p>
        <p style={{ color: "var(--ink-2)", fontSize: "0.89rem", margin: 0 }}>
          What it will never say is that the drill <em>caused</em> the change. There is
          no control here: the next walk is a different day, different light, possibly
          different shoes. A single walk clearing the threshold means the change is too
          large to be noise — not that the practice produced it. A consistent trend over
          several walks is the stronger evidence, and that is as far as this data goes.
        </p>
      </section>

      <section className="glass card">
        <h3>Analysing a group</h3>
        <p style={{ color: "var(--ink-2)", fontSize: "0.89rem", margin: "0 0 0.7rem" }}>
          A group clip is not scored differently — each walker is cut out of the frame
          and put through exactly the same model as a solo upload, so the numbers mean
          the same thing and are fair to compare. Cropping is what makes it work at all:
          run on the whole frame, the pose model found both people in 5% of frames; run
          on one crop per person, 100%.
        </p>
        <p style={{ color: "var(--ink-2)", fontSize: "0.89rem", margin: 0 }}>
          Walkers are told apart by face rather than by position, which is what keeps
          them labelled correctly when they cross paths. Two consequences follow.
          Someone whose face is never visible — filmed from behind, or too small in
          frame — cannot be separated out, so the page reports how many walkers it
          found rather than claiming it found everyone. And only walks matching your
          own enrolled face are saved to your history; the rest are shown and then
          discarded.
        </p>
      </section>

      <section className="glass card">
        <h3>Limitations</h3>
        <p style={{ color: "var(--ink-2)", fontSize: "0.89rem", margin: 0 }}>
          Stride length, rhythm, arm swing and symmetry are the hardest criteria to judge
          from a single head-on camera, because stride and arm swing are largely sagittal
          and a frontal view foreshortens them. A side or three-quarter angle would improve
          those. Every score is checked against the range the model was trained on, and any
          criterion outside it is flagged rather than presented as equally certain.
        </p>
      </section>

      {health && (
        <section className="glass card">
          <h3>System</h3>
          <div className="chips">
            <span className="chip">{health.models_loaded}/{health.criteria} models</span>
            <span className="chip">video: {health.video_backend}</span>
            <span className="chip">api: {api.base}</span>
          </div>
        </section>
      )}
    </>
  );
}
