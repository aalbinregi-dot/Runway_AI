# System Architecture

## 1. Layered view

The system is a pipeline of five layers. Each depends only on the layer below,
which is why the pose estimator can be swapped without touching the model, and
the model can be retrained without touching the interfaces.

```mermaid
flowchart TB
    subgraph L5["Presentation"]
        CLI["CLI<br/>python -m runway_ai"]
        WEB["React web app<br/>Vite · glass UI"]
        API["FastAPI service"]
    end
    subgraph L4["Application"]
        AN["analyze.py<br/>single entry point"]
        RP["feedback/report.py<br/>txt · md · html · json"]
        DB[("SQLite<br/>users · sessions · findings")]
    end
    subgraph L3["Intelligence"]
        MOD["models/train.py<br/>12 regressors"]
        RUB["scoring/rubric.py<br/>tolerance bands"]
        ASS["scoring/assess.py<br/>score + explain"]
        CO["feedback/coach.py<br/>drills + plan"]
    end
    subgraph L2["Representation"]
        FE["features/builder.py<br/>~100 features"]
        SIG["features/signals.py<br/>filters · phase · events"]
    end
    subgraph L1["Acquisition"]
        VID["pose/extract.py<br/>MediaPipe Pose"]
        KP["data/loaders.py<br/>CSV · XLSX · Parquet"]
        URL["data/fetch.py<br/>share links via yt-dlp"]
        SYN["data/synthetic.py<br/>labelled generator"]
    end

    URL --> VID
    VID --> PRE["data/loaders.preprocess<br/>fill · smooth · normalise"]
    KP --> PRE
    SYN --> PRE
    PRE --> SIG --> FE
    FE --> MOD --> ASS
    FE --> RUB --> ASS
    ASS --> CO --> AN
    AN --> CLI & WEB & API
    AN --> RP
    AN --> DB
```

## 2. Data flow for one analysis

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant I as Interface
    participant P as Pose backend
    participant C as Conditioning
    participant F as Features
    participant M as Models
    participant R as Rubric
    participant K as Coach
    participant D as SQLite

    U->>I: upload walk (video or keypoints)
    I->>P: extract landmarks
    P-->>I: PoseSequence (T x 13 x 2, +z, +visibility)
    I->>C: preprocess
    Note over C: interpolate gaps → Savitzky-Golay →<br/>pelvis-centre, torso-scale normalise
    C-->>F: normalised sequence
    F-->>M: ~100 features (torso lengths, degrees)
    M-->>I: 12 criterion scores (0-10)
    F->>R: same features
    R-->>I: which tolerance bands failed
    I->>K: scores + failed bands
    K-->>I: ranked focus areas + drills
    I->>D: persist session
    I-->>U: report + coaching plan
```

### 2a. Data flow for a group clip

Only the front of the pipeline differs. Once each walker has a `PoseSequence`,
they rejoin the sequence above unchanged — same conditioning, features, models,
rubric and coach — which is what makes scores comparable between people and
between a group clip and a solo upload.

```mermaid
sequenceDiagram
    autonumber
    actor U as User
    participant I as Interface
    participant Y as YuNet + SFace
    participant T as Tracker
    participant P as Pose backend
    participant A as Per-person analysis

    U->>I: upload group clip
    I->>Y: detect faces, every Nth frame
    Y-->>T: boxes + 128-d embeddings
    Note over T: one detection per track per frame<br/>(two faces at once = two people);<br/>face decides, position breaks ties
    T-->>I: PersonTrack per walker
    loop each walker
        I->>P: crop to this walker's body box
        Note over P: num_poses=1 on a crop found both<br/>walkers in 100% of frames;<br/>num_poses=4 on the full frame, 5%
        P-->>I: landmarks in crop space → mapped to full frame
        I->>A: PoseSequence
        A-->>I: scores, findings, coaching plan
    end
    I->>I: attribute each face template to an enrolled account
    Note over I: refuses to guess: needs the acceptance<br/>threshold *and* a margin over the runner-up
    I-->>U: ranked walkers + per-criterion matrix
```

Two design consequences worth stating explicitly:

* **A walker whose face is never visible cannot be separated out.** Faces are
  what make the tracks identifiable at all, so the result reports how many
  walkers it found rather than implying it found everyone.
* **Only a walk matching the signed-in account's own enrolled face is
  persisted.** Enrolling is consent to be recognised, not consent for anyone
  who films you to append to your training record.

## 3. Domain model

```mermaid
classDiagram
    class PoseSequence {
        +str video_id
        +list~str~ landmarks
        +ndarray xy "(T,L,2)"
        +ndarray z "(T,L)"
        +ndarray visibility "(T,L)"
        +float fps
        +dict meta
        +point(name) ndarray
        +midpoint(a,b) ndarray
        +to_long() DataFrame
    }
    class WalkParams {
        +float posture_q
        +float balance_q
        +"...10 more qualities"
        +float camera_azimuth_deg
        +float camera_distance_m
        +qualities() ndarray
    }
    class Band {
        +str feature
        +float ideal
        +float tol
        +str mode
        +float weight
        +str message
        +score(value) float
    }
    class Finding {
        +str criterion
        +str feature
        +float measured
        +float severity
        +str message
    }
    class Assessment {
        +dict scores
        +float overall
        +str grade
        +list~Finding~ findings
        +list~str~ strengths
        +list~str~ quality_warnings
    }
    class CoachingPlan {
        +str headline
        +str summary
        +list focus_areas
        +list drills
        +list next_session
    }
    class AnalysisResult {
        +str video_id
        +Assessment assessment
        +CoachingPlan plan
        +dict features
    }

    WalkParams ..> PoseSequence : renders
    PoseSequence ..> Assessment : features feed
    Band ..> Finding : produces
    Assessment "1" *-- "0..*" Finding
    AnalysisResult *-- Assessment
    AnalysisResult *-- CoachingPlan
```

## 4. Processing flowchart

```mermaid
flowchart TD
    A([Input]) --> B{File type}
    B -->|video| C[MediaPipe Pose<br/>33 landmarks/frame]
    B -->|csv / xlsx| D[Read long table<br/>validate schema]
    C --> E[Keep 13 analysis landmarks]
    D --> E
    E --> F{>= 30 frames?}
    F -->|no| G([Reject: clip too short])
    F -->|yes| H[Interpolate visibility &lt; 0.5]
    H --> I[Savitzky-Golay smooth<br/>preserves gait peak timing]
    I --> J[Normalise:<br/>pelvis → origin, torso → unit]
    J --> K[Zero-phase high-pass 0.45 Hz<br/>removes perspective drift]
    K --> L[Hilbert phase → cycle events<br/>sub-frame stride timing]
    L --> M[~100 features]
    M --> N[12 regressors → scores]
    M --> O[Rubric bands → causes]
    N --> P{score &lt; 7?}
    O --> P
    P -->|yes| Q[Raise finding]
    P -->|no| R[Report as strength]
    Q --> S[Rank by severity, cap at 3]
    R --> S
    S --> T[Attach drills]
    T --> U([Report + plan + DB row])
```

## 5. Why the layers sit where they do

**Pose estimation is isolated behind one function.** Everything downstream
consumes a `PoseSequence`, never a detector. That is what lets the project run
end-to-end on Python 3.14 even though MediaPipe publishes wheels only up to
3.12 — keypoint input works on any interpreter, and only video ingestion needs
the constrained environment. It is also what makes the 3D-pose and multi-camera
extensions a backend change rather than a rewrite.

**Normalisation sits before features, not inside them.** Pelvis-centring and
torso-scaling happen once, so every feature is automatically invariant to
camera distance and subject height. Doing it per-feature would guarantee
inconsistency.

**Scoring and explanation are separate.** The learned models are accurate but
opaque; the rubric is transparent but blunter. Using the model for the number
and the rubric for the reason gives both, and `assess.diagnose` gates findings
on the model's score so the two can never contradict each other in the output.
