"""End-to-end analysis: any supported input -> scores, findings, coaching plan.

This is the single entry point the CLI, the API and the dashboard all call, so
they cannot drift apart in how a walk is scored.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .data.loaders import load_sequences, preprocess
from .data.schema import PoseSequence
from .feedback.coach import CoachingPlan, build_plan
from .features.builder import extract_features
from .models.train import load_models
from .scoring.assess import Assessment, assess

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
KEYPOINT_SUFFIXES = {".csv", ".xlsx", ".xls", ".xlsm", ".parquet"}


@dataclass
class AnalysisResult:
    video_id: str
    assessment: Assessment
    plan: CoachingPlan
    features: dict
    #: Where a linked clip came from, else None. Deliberately *not* named
    #: ``source``: ``Assessment.as_dict()`` already emits that key for the
    #: scorer ("model" / "rubric"), and writing provenance to it silently
    #: replaced a string with an object in the API payload.
    origin: dict | None = None
    #: Image-space keypoint track for playback, else None. See ``_track``.
    track: dict | None = None
    #: Per-frame traces, gait cycles and per-finding evidence times.
    timeline: dict | None = None
    #: Face-verification outcome when the uploader has enrolled, else None.
    attribution: dict | None = None
    #: Head-pose measurements, when a face was visible. Measured, not scored.
    gaze: dict | None = None
    #: Estimated camera angle and which criteria it can measure well.
    viewpoint: dict | None = None
    #: Which walker this is, in a group clip. None for a solo upload.
    person: dict | None = None
    #: Per-phase measurements (approach / middle / exit). See scoring.phases.
    phases: dict | None = None

    def as_dict(self) -> dict:
        out = {
            "video_id": self.video_id,
            **self.assessment.as_dict(),
            "coaching": self.plan.as_dict(),
        }
        if self.origin:
            out["origin"] = self.origin
        if self.track:
            out["track"] = self.track
        if self.timeline:
            out["timeline"] = self.timeline
        if self.attribution:
            out["attribution"] = self.attribution
        if self.gaze:
            out["gaze"] = self.gaze
        if self.viewpoint:
            out["viewpoint"] = self.viewpoint
        if self.person:
            out["person"] = self.person
        if self.phases:
            out["phases"] = self.phases
        return out


def _sidecar_fps(path: Path) -> float | None:
    """Read the frame rate written alongside an extracted keypoint file.

    ``runway_ai extract`` writes ``<file>.meta.json`` because the seven-column
    keypoint schema has no field for fps, and getting it wrong silently skews
    every cadence, rhythm and timing feature.
    """
    sidecar = path.with_suffix(path.suffix + ".meta.json")
    if not sidecar.exists():
        return None
    try:
        value = float(json.loads(sidecar.read_text()).get("fps", 0.0))
        return value if value > 0 else None
    except Exception:
        return None


def _track(seq: PoseSequence, max_frames: int = 400) -> dict:
    """Compact image-space keypoint track, for drawing over the source video.

    Taken *before* normalisation: ``preprocess`` moves the pelvis to the origin
    and rescales by torso length, which is exactly what a feature needs and
    exactly what an overlay cannot use.

    Coordinates are rounded to four decimals and visibility to two. Three was
    the first choice and it cost 0.2/100 on the overall score when a stored
    walk was re-measured; four matches the 1e-4 fixed point the track store
    uses, so persisting and re-scoring is lossless.
    """
    import numpy as np

    n = seq.n_frames
    step = max(1, -(-n // max_frames))         # ceil division
    idx = list(range(0, n, step))
    xy = np.nan_to_num(seq.xy[idx], nan=0.0)
    return {
        "landmarks": list(seq.landmarks),
        "fps": round(seq.fps / step, 4),
        "n_frames": len(idx),
        "xy": np.round(xy, 4).tolist(),
        "visibility": np.round(seq.visibility[idx], 2).tolist(),
    }


def analyze_sequence(seq: PoseSequence, models: dict | None = None,
                     with_track: bool = False) -> AnalysisResult:
    """Score an already-loaded pose sequence."""
    track = _track(seq) if (with_track and not seq.meta.get("preprocessed")) else None
    gaze_feats = seq.meta.get("gaze_features") or {}
    if not seq.meta.get("preprocessed"):
        seq = preprocess(seq)
    feats = extract_features(seq)
    # Checked before scoring, not after: a recording with no measurements in it
    # must produce an explanation, not a number with a caveat attached.
    from .scoring.assess import assert_usable
    assert_usable(feats)
    # Merged after extraction, never into the model input: the trained models
    # were fitted on a fixed feature list that has no gaze in it, and could not
    # have - the synthetic generator draws faceless stick figures. predict_scores
    # selects its own columns, so these reach the rubric and nothing else.
    feats.update(gaze_feats)
    a = assess(feats, models)

    # Which angle the clip was shot from decides which scores deserve weight,
    # so it is estimated for every walk, not just videos.
    viewpoint = None
    try:
        from .features import viewpoint as vp
        size = seq.meta.get("frame_size")
        viewpoint = vp.estimate(seq, aspect=(size[1] / size[0]) if size else None)
    except Exception:
        viewpoint = None

    timeline = None
    phase_view = None
    if with_track:
        # Built from the *preprocessed* sequence so the traces carry the same
        # normalised units as the features and the rubric thresholds.
        from .features import timeline as tl
        try:
            timeline = tl.build(seq, a.findings)
        except Exception:
            timeline = None       # evidence is a bonus; never fail a score for it

        if timeline:
            # Reuses that timeline rather than rebuilding: the per-phase view is
            # the same traces, summarised in thirds. See scoring.phases for why
            # this reports measurements instead of per-phase scores.
            try:
                from .scoring import phases as phase_mod
                phase_view = phase_mod.analyse(seq, timeline)
            except Exception:
                phase_view = None

    return AnalysisResult(seq.video_id, a, build_plan(a), feats,
                          track=track, timeline=timeline, viewpoint=viewpoint,
                          phases=phase_view)


def analyze_group(path: str | Path, models: dict | None = None,
                  stride: int = 4, on_stage=None,
                  attribute: bool = True, on_person=None,
                  keep_fetched: bool = False) -> list[AnalysisResult]:
    """Score every walker in one clip, attributed by face where possible.

    Each walker is scored by exactly the same code as a solo upload - there is
    no separate "group" rubric - so a score means the same thing either way and
    the comparison between people is meaningful.

    Attribution is opt-in twice over: only accounts that chose to enrol a face
    can be matched, and a walker who matches nobody is simply reported as
    "Person 2" rather than guessed at.
    """
    if models is None:
        models = load_models() or None

    def stage(name: str, **kw) -> None:
        if on_stage:
            on_stage(name, kw)

    # A share link is fetched to a temporary file and then treated exactly like
    # an upload, so link support costs the rest of this function nothing.
    from .data.fetch import FetchedVideo, fetch_video, is_url

    fetched: FetchedVideo | None = None
    if isinstance(path, str) and is_url(path):
        stage("fetch", status="active")
        fetched = fetch_video(path)
        stage("fetch", status="done", title=fetched.title)
        path = fetched.path

    path = Path(path)
    if path.suffix.lower() not in VIDEO_SUFFIXES:
        raise ValueError("group analysis needs a video file")

    stage("pose", status="active", done=0, total=0)
    from .pose.extract import detect_backend

    templates = _enrolled_templates() if attribute else {}
    results: list[AnalysisResult] = []
    seen: set[str] = set()

    def score_one(seq, template, done: int, total: int) -> None:
        """Score one walker the moment its pose lands.

        Pose estimation dominates the runtime - roughly 40 seconds a person
        against 0.2 to score one - so scoring here, rather than after every
        walker has been extracted, is the difference between a client that fills
        in progressively and one that shows nothing for a minute and then
        everything at once.
        """
        if seq.video_id in seen:
            return
        seen.add(seq.video_id)
        try:
            r = analyze_sequence(seq, models, with_track=True)
        except Exception:
            return                      # one unusable walker must not stop the rest
        index = seq.meta.get("person_index") or len(results) + 1
        r.person = {"index": index, "label": f"Person {index}",
                    "frames": seq.n_frames}
        if templates and template is not None and len(template):
            match = _match_template(template, templates)
            if match:
                r.person["label"] = match["name"]
                r.person["matched_user_id"] = match["user_id"]
                r.person["match_similarity"] = match["similarity"]
        results.append(r)
        if on_person:
            on_person(r, done or len(results), total or len(results))

    if detect_backend() != "none":
        from .pose.multi import extract_all
        extract_all(path, stride=stride,
                    on_progress=lambda d, t: stage("pose", status="active",
                                                   done=d, total=t),
                    on_person=score_one)
    else:
        from .pose.delegate import extract_multi_delegated
        # The delegated child streams each walker as its CSV is written; the face
        # template only arrives with the manifest at the end, so a streamed
        # walker is scored immediately and attributed in the reconciliation pass
        # below.
        streamed: dict[str, AnalysisResult] = {}

        def from_stream(seq, done: int, total: int) -> None:
            before = len(results)
            score_one(seq, None, done, total)
            if len(results) > before:
                streamed[seq.video_id] = results[-1]

        people = extract_multi_delegated(
            path, stride=stride,
            on_progress=lambda d, t: stage("pose", status="active", done=d, total=t),
            on_person=from_stream)

        # Reconcile: apply the ordering and face attribution the manifest carries,
        # and pick up anyone the stream missed.
        for seq, template in people:
            existing = streamed.get(seq.video_id)
            if existing is None:
                score_one(seq, template, 0, len(people))
                continue
            index = seq.meta.get("person_index") or existing.person.get("index")
            existing.person["index"] = index
            existing.person["label"] = f"Person {index}"
            if templates and template:
                match = _match_template(template, templates)
                if match:
                    existing.person["label"] = match["name"]
                    existing.person["matched_user_id"] = match["user_id"]
                    existing.person["match_similarity"] = match["similarity"]

    if not results:
        raise RuntimeError(
            "No walkers with a visible face were found. Group analysis tells "
            "people apart by face, so each walker has to face the camera at "
            "least once.")
    results.sort(key=lambda r: r.person.get("index") or 0)
    stage("pose", status="done", people=len(results))

    stage("measure", status="done")
    stage("score", status="done", people=len(results))

    if fetched is not None:
        info = fetched.as_dict()
        if keep_fetched:
            info["local_path"] = str(fetched.path)
        else:
            fetched.path.unlink(missing_ok=True)
        for r in results:
            r.origin = info
    return results


def _enrolled_templates() -> dict:
    """Face templates of accounts that chose to enrol, keyed by user id."""
    try:
        from .db import auth
        return auth.all_face_templates()
    except Exception:
        return {}


def _match_template(template, enrolled: dict) -> dict | None:
    """Attribute one walker to an enrolled account, or to nobody.

    Uses the same acceptance threshold and best-versus-runner-up margin as
    sign-in identification. Refusing to guess matters more here than there: a
    wrong name on a group report attaches one person's posture scores to
    another person's history.

    Every reject path is written as a *positive* test of acceptance rather than
    a negative test for rejection. That is not style: a NaN embedding made
    ``best < ACCEPT_COSINE`` evaluate False - every comparison against NaN is
    False - so the guard waved it through and a degenerate vector was attributed
    to a real account. Requiring ``best >= threshold`` fails closed instead.
    """
    import numpy as np

    from . import face as face_mod

    probe = np.asarray(template, dtype=float).ravel()
    if probe.size == 0 or not np.all(np.isfinite(probe)):
        return None
    norm = float(np.linalg.norm(probe))
    if not np.isfinite(norm) or norm <= 0:
        return None
    probe = probe / norm

    scored = []
    for uid, info in enrolled.items():
        ref = np.asarray(info["template"], dtype=float).ravel()
        # A template of the wrong width means a different embedder wrote it.
        # Skip it rather than raising: one unreadable row must not take down the
        # whole group's analysis.
        if ref.shape != probe.shape or not np.all(np.isfinite(ref)):
            continue
        ref_norm = float(np.linalg.norm(ref))
        if not np.isfinite(ref_norm) or ref_norm <= 0:
            continue
        score = float(probe @ (ref / ref_norm))
        if np.isfinite(score):
            scored.append((score, uid, info.get("name")))
    if not scored:
        return None

    scored.sort(reverse=True)
    best, uid, name = scored[0]
    runner_up = scored[1][0] if len(scored) > 1 else -1.0
    if not (best >= face_mod.ACCEPT_COSINE
            and (best - runner_up) >= face_mod.MIN_MARGIN):
        return None
    return {"user_id": uid, "name": name, "similarity": round(best, 3)}


def analyze(path: str | Path, models: dict | None = None,
            video_id: str | None = None, fps: float | None = None,
            load_trained: bool = True,
            on_stage=None, keep_fetched: bool = False) -> list[AnalysisResult]:
    """Analyse a video file, a keypoint table, or a share link.

    A keypoint file may hold many clips, so a list is always returned.
    """
    if models is None and load_trained:
        models = load_models() or None

    # A share link is fetched to a temporary file first, then handled exactly
    # like any other video - so link support costs the rest of the pipeline
    # nothing and cannot diverge from it.
    from .data.fetch import FetchedVideo, fetch_video, is_url

    def stage(name: str, **kw) -> None:
        if on_stage:
            on_stage(name, kw)

    fetched: FetchedVideo | None = None
    if isinstance(path, str) and is_url(path):
        stage("fetch", status="active")
        fetched = fetch_video(path)
        stage("fetch", status="done", title=fetched.title)
        path = fetched.path
        video_id = video_id or _slug(fetched.title)

    path = Path(path)
    try:
        results = _analyze_path(path, models, video_id, fps, stage)
    finally:
        # Normally the download is discarded - the keypoints and scores are what
        # matter. A caller that intends to serve the video back (so the client
        # can show the skeleton overlay, which it otherwise cannot for a link)
        # asks to keep it, and becomes responsible for deleting it.
        if fetched is not None and not keep_fetched:
            fetched.path.unlink(missing_ok=True)

    if fetched is not None:
        info = fetched.as_dict()
        if keep_fetched:
            info["local_path"] = str(fetched.path)
        for r in results:
            r.origin = info
    return results


def _slug(title: str, limit: int = 60) -> str:
    """A filesystem- and database-safe id from a video title."""
    keep = [c if (c.isalnum() or c in " -_") else " " for c in title]
    slug = "_".join("".join(keep).split())[:limit]
    return slug or "linked_video"


def _analyze_path(path: Path, models: dict | None, video_id: str | None,
                  fps: float | None, stage=None) -> list[AnalysisResult]:
    stage = stage or (lambda *a, **k: None)

    suffix = path.suffix.lower()
    if suffix in VIDEO_SUFFIXES:
        from .pose.extract import detect_backend, extract_from_video

        def pose_progress(done: int, total: int) -> None:
            # total == 0 means frames are still being decoded, so the
            # denominator is not known yet.
            stage("pose", status="active", done=done, total=total)

        stage("pose", status="active", done=0, total=0)
        if detect_backend() != "none":
            seq = extract_from_video(path, video_id=video_id,
                                     progress=False, on_progress=pose_progress)
        else:
            # This interpreter has no MediaPipe. Rather than failing - which
            # would make video unusable from the API and the web client - hand
            # the extraction to a sibling environment that does. Scoring still
            # happens here, so the models never meet a second scikit-learn.
            from .pose.delegate import extract_delegated
            seq = extract_delegated(path, video_id=video_id,
                                    on_progress=pose_progress)
        stage("pose", status="done", done=seq.n_frames, total=seq.n_frames,
              detected=int(seq.meta.get("frames_detected", seq.n_frames)))

        stage("measure", status="active")
        # Gaze is measured from the video itself, not from the pose track: the
        # facial landmarks the head pose needs simply do not exist in a
        # keypoint file. It is folded into the features *before* scoring so the
        # rubric's gaze bands can fire, and it is optional throughout - a clip
        # with no visible face is scored exactly as it always was.
        gaze_summary = None
        try:
            from .features import gaze as gaze_mod
            series = gaze_mod.head_pose_series(path, nose_track=seq.point("nose"))
            gaze_feats = gaze_mod.features(series)
            if gaze_feats:
                seq.meta["gaze_features"] = gaze_feats
                gaze_summary = gaze_mod.summary(gaze_feats)
        except Exception:
            pass

        result = analyze_sequence(seq, models, with_track=True)
        result.gaze = gaze_summary
        stage("measure", status="done")
        stage("score", status="done", overall=round(result.assessment.overall, 1))
        return [result]

    if suffix in KEYPOINT_SUFFIXES:
        stage("pose", status="skipped")
        stage("measure", status="active")
        # Loaded raw so the image-space track survives for replay;
        # analyze_sequence conditions each one itself.
        seqs = load_sequences(path, fps=fps or _sidecar_fps(path) or 30.0,
                              conditioned=False)
        out = [analyze_sequence(s, models, with_track=True) for s in seqs.values()]
        stage("measure", status="done")
        stage("score", status="done",
              overall=round(out[0].assessment.overall, 1) if out else None)
        return out

    raise ValueError(
        f"unsupported input '{suffix}'. Provide a video "
        f"({', '.join(sorted(VIDEO_SUFFIXES))}) or a keypoint table "
        f"({', '.join(sorted(KEYPOINT_SUFFIXES))})."
    )
