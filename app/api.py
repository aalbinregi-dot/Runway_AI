"""FastAPI service.

Run:  uvicorn app.api:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import threading
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
# app/ itself, so `auth_routes` resolves whether this is imported as `app.api`
# (uvicorn) or executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from runway_ai.analyze import KEYPOINT_SUFFIXES, VIDEO_SUFFIXES, analyze  # noqa: E402
from runway_ai.config import (CRITERIA, CRITERION_LABELS,  # noqa: E402
                              CRITERION_WEIGHTS, GRADE_BANDS)
from runway_ai.db import store                                # noqa: E402
from runway_ai.feedback import report as report_mod           # noqa: E402
from runway_ai import media                                  # noqa: E402
from runway_ai.models.train import load_models                # noqa: E402
from runway_ai.pose.extract import mediapipe_available        # noqa: E402
from runway_ai.scoring import significance                    # noqa: E402
from runway_ai.db import auth as auth_db                     # noqa: E402
from runway_ai import face                                   # noqa: E402
from auth_routes import current_user_optional, router as auth_router  # noqa: E402
from history_routes import router as history_router          # noqa: E402

app = FastAPI(
    title="Runway Performance AI",
    version="1.0.0",
    description="Analyse a runway walk and return scores, findings and coaching.",
)

# The React client runs on its own dev server, so the browser treats calls to
# this API as cross-origin. Localhost only - widen deliberately for deployment
# rather than shipping a wildcard.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5180", "http://127.0.0.1:5180",   # vite dev
        "http://localhost:4173", "http://127.0.0.1:4173",   # vite preview
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Bring the schema up to date before the first request touches it.
auth_db.migrate()
app.include_router(auth_router)
app.include_router(history_router)

_MODELS: dict | None = None


def models():
    """Load once, reuse. Loading twelve estimators per request would dominate latency."""
    global _MODELS
    if _MODELS is None:
        _MODELS = load_models() or {}
    return _MODELS or None


class Health(BaseModel):
    status: str
    models_loaded: int
    criteria: int
    video_backend: str


@app.get("/health", response_model=Health)
def health() -> Health:
    return Health(
        status="ok",
        models_loaded=len(models() or {}),
        criteria=len(CRITERIA),
        video_backend="mediapipe" if mediapipe_available() else "unavailable",
    )


@app.get("/diagnostics")
def diagnostics_report() -> dict:
    """What this installation can and cannot do, with a remedy for each gap.

    Unauthenticated on purpose: someone whose install is broken often cannot
    sign in, and that is exactly when this is needed. Safe to expose because it
    carries no user data and no filesystem paths - see app/diagnostics.py.
    """
    import diagnostics as diag

    report = diag.build()
    return {**report.as_dict(), "text": diag.as_text(report)}


@app.get("/sources")
def sources() -> dict:
    """Which link hosts the fetcher accepts."""
    from runway_ai.data.fetch import ALLOWED_HOSTS, MAX_DURATION_S, PLATFORMS

    try:
        import yt_dlp  # noqa: F401
        available = True
    except ImportError:
        available = False
    return {
        "available": available,
        "max_duration_s": MAX_DURATION_S,
        "platforms": PLATFORMS,
        "hosts": sorted({h.removeprefix("www.") for h in ALLOWED_HOSTS}),
    }


@app.get("/criteria")
def criteria() -> dict:
    """The scoring rubric's criteria, weights, and the limits on each.

    Thresholds and identifiability tiers are served here rather than restated in
    the client, so the Guide cannot drift out of step with what the code actually
    does when a model is retrained.
    """
    from runway_ai.scoring import confidence

    conf = confidence.summary()
    per = conf.get("per_criterion", {}) if conf.get("available") else {}

    return {
        "criteria": [
            {
                "key": c,
                "label": CRITERION_LABELS[c],
                "weight": CRITERION_WEIGHTS[c],
                # Smallest difference in this criterion worth reporting: 1.41 x
                # its out-of-fold MAE.
                "threshold": significance.threshold(c),
                "identifiability": per.get(c, {}).get("tier"),
            }
            for c in CRITERIA
        ],
        "score_range": [0, 10],
        "overall_range": [0, 100],
        # Served rather than restated in the client, for the same reason the
        # weights are: the client draws the overall score against these bands, and
        # a second copy of the thresholds would go stale the first time one moved.
        "grade_bands": [{"from": threshold, "label": name}
                        for threshold, name in GRADE_BANDS],
        "thresholds": {
            "overall": round(significance.overall_threshold(), 1),
            "basis": ("1.41 x out-of-fold MAE per criterion; the overall combines "
                      "the twelve in quadrature rather than summing them"),
        },
        "measurement_confidence": conf,
    }


@app.post("/analyze")
async def analyze_upload(
    file: UploadFile = File(...),
    fps: float | None = Query(None, description="fps override for keypoint uploads"),
    account=Depends(current_user_optional),
) -> dict:
    """Analyse an uploaded video or keypoint table."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES | KEYPOINT_SUFFIXES:
        raise HTTPException(415, f"unsupported file type '{suffix}'")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    try:
        results = analyze(path, models=models(), fps=fps)
        _attribute(results, account, path if suffix in VIDEO_SUFFIXES else None)
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        Path(path).unlink(missing_ok=True)

    # Saved only when signed in, and only ever to the caller's own account.
    # The old ?user=<name> parameter let any caller write into any history.
    if account:
        for r in results:
            store.save_session(account.id, r.video_id, r.assessment, r.features,
                               track=r.track, source_path=file.filename)

    return {"count": len(results), "results": [r.as_dict() for r in results],
            "saved": bool(account)}


def _attribute(results, account, video_path: str | None) -> None:
    """Check a walk's face against the signed-in user's own enrolment.

    v1 does *verification*, not open identification: it answers "is this you?"
    and nothing else. Telling an uploader "this looks like Ana" would disclose
    that Ana enrolled, to somebody with no relationship to her, so no other
    account is ever named. A mismatch simply says the face is not the account
    holder's and lets them decide what to do.

    Silent on every failure path. Face attribution is a convenience; it must
    never be the reason a score fails to come back.
    """
    if not (account and video_path and face.available()):
        return
    if not auth_db.has_face_template(account.id):
        return
    try:
        template = auth_db.load_face_templates().get(account.id)
        if template is None:
            return
        nose = None
        track = getattr(results[0], "track", None) if results else None
        if track:
            import numpy as np
            idx = track["landmarks"].index("nose")
            nose = np.asarray([f[idx] for f in track["xy"]])
        probes = face.embeddings_from_video(video_path, n_frames=12, nose_track=nose)
        verdict = face.identify(probes, {account.id: template})
    except Exception:
        return
    for r in results:
        r.attribution = {
            "checked": True,
            "is_account_holder": verdict.identified,
            "score": round(verdict.score, 3),
            "reason": verdict.reason,
            "frames_used": verdict.frames_used,
        }


def _publish_media(results) -> None:
    """Move a kept download into the media store and attach its URL.

    Called for link analyses only. On any failure the file is deleted rather
    than left behind: losing the overlay is a much smaller problem than leaking
    videos onto disk.
    """
    for r in results:
        path = (r.origin or {}).pop("local_path", None)
        if not path:
            continue
        try:
            r.origin["video_url"] = f"/media/{media.store(path)}"
        except Exception:
            Path(path).unlink(missing_ok=True)


class LinkRequest(BaseModel):
    url: str


@app.post("/analyze-url")
def analyze_link(req: LinkRequest, account=Depends(current_user_optional)) -> dict:
    """Analyse a walk from a YouTube / Instagram / TikTok share link.

    The fetch is done server-side, so the URL is validated against a host
    allow-list and checked for private addresses before anything is requested -
    see :mod:`runway_ai.data.fetch`.
    """
    from runway_ai.data.fetch import FetchError, is_url

    # Without this, a non-URL falls through to the file-path branch and comes
    # back as "unsupported input ''", which tells the caller nothing.
    if not is_url(req.url):
        raise HTTPException(
            422, "That does not look like a link. Paste the full URL, "
                 "starting with https://"
        )

    try:
        results = analyze(req.url, models=models(), keep_fetched=True)
        _publish_media(results)
    except FetchError as exc:
        # A rejected or unfetchable link is the caller's problem to fix, and the
        # message already says how; 422 with that text is more use than a 500.
        raise HTTPException(422, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc

    if account:
        for r in results:
            store.save_session(account.id, r.video_id, r.assessment, r.features,
                               track=r.track, source_path=req.url)

    return {"count": len(results), "results": [r.as_dict() for r in results],
            "saved": bool(account)}


#: The stages a client shows, in order. ``fetch`` only appears for links and
#: ``pose`` is skipped for keypoint uploads, so the client is told which apply.
STAGES = [
    {"id": "fetch", "label": "Fetching video", "detail": "downloading from the link"},
    {"id": "pose", "label": "Tracking the body",
     "detail": "MediaPipe finds 33 landmarks per frame; 13 are kept"},
    {"id": "measure", "label": "Measuring the walk",
     "detail": "102 gait and posture features in torso lengths and degrees"},
    {"id": "score", "label": "Scoring",
     "detail": "twelve models, then the rubric explains each result"},
]


@app.get("/stages")
def stages() -> dict:
    return {"stages": STAGES}


async def _stream_analysis(run, is_link: bool):
    """Run a blocking analysis on a worker thread, streaming stage events.

    The analysis is CPU-bound and synchronous. Running it inline would block the
    event loop and no event could be flushed until it finished, which defeats
    the point. So it runs in a thread and pushes events into a queue the
    generator drains.
    """
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def on_stage(name: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {"type": "stage", "id": name, **data})

    # Group analysis scores one walker at a time, so each finished walker is
    # pushed as its own event and the client can render it immediately rather
    # than holding a blank screen until the whole clip is done.
    def on_person(result, done: int, total: int) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, {
            "type": "person", "done": done, "total": total,
            "result": result.as_dict()})

    def worker():
        try:
            import inspect
            # The solo and link routes take only the stage callback; the group
            # route takes both. Inspecting keeps one relay serving all three
            # rather than duplicating the generator.
            wants = len(inspect.signature(run).parameters)
            results = run(on_stage, on_person) if wants >= 2 else run(on_stage)
            payload = {"type": "result", "count": len(results),
                       "results": [r.as_dict() for r in results]}
        except Exception as exc:
            payload = {"type": "error", "message": str(exc)}
        loop.call_soon_threadsafe(queue.put_nowait, payload)

    threading.Thread(target=worker, daemon=True).start()

    applicable = [s["id"] for s in STAGES if s["id"] != "fetch" or is_link]
    yield _sse({"type": "start", "stages": applicable})
    while True:
        event = await queue.get()
        yield _sse(event)
        if event["type"] in ("result", "error"):
            break


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    # Without this, nginx and similar buffer the whole response and the client
    # sees nothing until the analysis has already finished.
    "X-Accel-Buffering": "no",
}


@app.post("/analyze/stream")
async def analyze_stream(
    file: UploadFile = File(...),
    account=Depends(current_user_optional),
) -> StreamingResponse:
    """Analyse an upload, streaming progress as server-sent events."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES | KEYPOINT_SUFFIXES:
        raise HTTPException(415, f"unsupported file type '{suffix}'")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    def run(on_stage):
        try:
            results = analyze(path, models=models(), on_stage=on_stage)
            _attribute(results, account, path if suffix in VIDEO_SUFFIXES else None)
        finally:
            Path(path).unlink(missing_ok=True)
        if account:
            for r in results:
                store.save_session(account.id, r.video_id, r.assessment,
                                   r.features, track=r.track,
                                   source_path=file.filename)
        return results

    return StreamingResponse(_stream_analysis(run, is_link=False),
                             media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/analyze-url/stream")
async def analyze_url_stream(req: LinkRequest,
                            account=Depends(current_user_optional)) -> StreamingResponse:
    """Analyse a share link, streaming progress as server-sent events."""
    from runway_ai.data.fetch import is_url

    if not is_url(req.url):
        raise HTTPException(
            422, "That does not look like a link. Paste the full URL, "
                 "starting with https://")

    def run(on_stage):
        results = analyze(req.url, models=models(), on_stage=on_stage,
                          keep_fetched=True)
        _publish_media(results)
        if account:
            for r in results:
                store.save_session(account.id, r.video_id, r.assessment,
                                   r.features, track=r.track,
                                   source_path=req.url)
        return results

    return StreamingResponse(_stream_analysis(run, is_link=True),
                             media_type="text/event-stream", headers=SSE_HEADERS)


class GroupLinkRequest(BaseModel):
    url: str


@app.post("/analyze-group-url/stream")
async def analyze_group_url_stream(
    req: GroupLinkRequest,
    account=Depends(current_user_optional),
) -> StreamingResponse:
    """Analyse every walker in a linked clip, streaming each as it finishes."""
    from runway_ai.analyze import analyze_group
    from runway_ai.data.fetch import is_url

    if not is_url(req.url):
        raise HTTPException(
            422, "That does not look like a link. Paste the full URL, "
                 "starting with https://")

    def run(on_stage, on_person):
        results = analyze_group(req.url, models=models(), on_stage=on_stage,
                                on_person=on_person, keep_fetched=True)
        _publish_media(results)
        if account:
            for r in results:
                if (r.person or {}).get("matched_user_id") == account.id:
                    store.save_session(account.id, r.video_id, r.assessment,
                                       r.features, track=r.track,
                                       source_path=req.url)
        return results

    return StreamingResponse(_stream_analysis(run, is_link=True),
                             media_type="text/event-stream", headers=SSE_HEADERS)


@app.post("/analyze-group/stream")
async def analyze_group_stream(
    file: UploadFile = File(...),
    account=Depends(current_user_optional),
) -> StreamingResponse:
    """Score every walker in one clip, streaming progress as it goes.

    Only the walk that matches the *signed-in* account is written to history.
    The others are shown and then discarded, deliberately: enrolling a face is
    consent to be recognised, not consent for anyone who films you to append to
    your training record. Saving them would also mean a stranger's upload could
    silently reshape your progress chart.
    """
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in VIDEO_SUFFIXES:
        raise HTTPException(415, "Group analysis needs a video file.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name

    def run(on_stage, on_person):
        from runway_ai.analyze import analyze_group

        kept = False
        try:
            results = analyze_group(path, models=models(), on_stage=on_stage,
                                    on_person=on_person)
            # One shared video for every walker: the overlay differs per person
            # only in which skeleton is drawn, and storing N copies of the same
            # upload would multiply disk for nothing.
            try:
                url = f"/media/{media.store(path)}"
                kept = True
                for r in results:
                    r.origin = {**(r.origin or {}), "video_url": url}
            except Exception:
                pass
        finally:
            if not kept:
                Path(path).unlink(missing_ok=True)

        if account:
            for r in results:
                if (r.person or {}).get("matched_user_id") == account.id:
                    store.save_session(account.id, r.video_id, r.assessment,
                                       r.features, track=r.track,
                                       source_path=file.filename)
        return results

    return StreamingResponse(_stream_analysis(run, is_link=False),
                             media_type="text/event-stream", headers=SSE_HEADERS)


@app.get("/media/{token}")
def media_file(token: str) -> FileResponse:
    """Serve a video fetched from a link, so the client can overlay the skeleton.

    The token maps to a path *the server chose*; nothing from the request
    reaches the filesystem, so this cannot be walked into an arbitrary read.
    Starlette's FileResponse answers Range requests, which is what makes the
    video seekable — without 206 support, clicking a finding could not jump.
    """
    entry = media.resolve(token)
    if entry is None:
        raise HTTPException(404, "that video has expired; analyse the link again")
    return FileResponse(entry.path, media_type=entry.content_type)


@app.post("/report", response_class=HTMLResponse)
async def html_report(file: UploadFile = File(...)) -> str:
    """Analyse an upload and return a rendered HTML report."""
    suffix = Path(file.filename or "").suffix.lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        path = tmp.name
    try:
        results = analyze(path, models=models())
    except Exception as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        Path(path).unlink(missing_ok=True)
    if not results:
        raise HTTPException(422, "no analysable clip found")
    return report_mod.to_html(results[0])


@app.get("/me/progress")
def my_progress(account=Depends(current_user_optional)) -> dict:
    """Progress for the signed-in account.

    Keyed on the session rather than a name in the path. The previous
    /users/{name}/progress route was an IDOR: any caller could read any
    history by guessing a name, and get_or_create_user silently created the
    account if the name did not exist.
    """
    if account is None:
        return {"sessions": 0, "message": "Sign in to track progress across sessions."}
    return store.progress(account.id)


@app.get("/me/sessions")
def my_sessions(account=Depends(current_user_optional)) -> dict:
    if account is None:
        return {"count": 0, "sessions": []}
    hist = store.user_history(account.id)
    return {"count": len(hist), "sessions": hist}
