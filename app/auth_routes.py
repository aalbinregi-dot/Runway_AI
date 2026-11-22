"""Authentication and face-enrolment endpoints.

Session transport
-----------------
An opaque token in an **httpOnly, SameSite=Lax** cookie. Only its SHA-256 is
stored server-side, so a database leak does not hand over live sessions.

Two consequences of that choice, stated rather than assumed:

* **httpOnly** means script cannot read the token, so an XSS bug cannot
  exfiltrate it. A JWT in ``localStorage`` can be read by any injected script,
  and revoking it before expiry needs a server-side blocklist anyway - at which
  point it has the storage cost of a session with none of the safety.
* **SameSite=Lax** is the CSRF control. A cross-site form POST does not carry a
  Lax cookie, so it arrives unauthenticated and is rejected before any handler
  runs. If the cookie ever has to become ``SameSite=None`` for a cross-origin
  deployment, that protection disappears and a double-submit CSRF token must be
  added at the same time.

Anonymous use is preserved: ``current_user_optional`` returns ``None`` rather
than raising, so analysing a walk without an account keeps working exactly as
before. Authentication gates *storage*, not *scoring*.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Cookie, Depends, File, HTTPException, Response, UploadFile
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from runway_ai import face                                  # noqa: E402
from runway_ai.db import auth                               # noqa: E402

router = APIRouter()

COOKIE_NAME = "runway_session"
COOKIE_MAX_AGE = auth.SESSION_TTL_DAYS * 24 * 3600


def _set_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME, token,
        max_age=COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
        # secure=True would break plain-HTTP localhost development. Turn it on
        # together with SameSite=None behind TLS; see the module docstring.
        secure=False,
        path="/",
    )


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
def current_user_optional(
    runway_session: str | None = Cookie(default=None),
) -> auth.User | None:
    """The signed-in user, or None. Never raises - anonymous use is supported."""
    return auth.resolve_session(runway_session)


def current_user(user: auth.User | None = Depends(current_user_optional)) -> auth.User:
    """The signed-in user, or 401. For endpoints that genuinely need identity."""
    if user is None:
        raise HTTPException(401, "Sign in to do that.")
    return user


# --------------------------------------------------------------------------- #
# Accounts
# --------------------------------------------------------------------------- #
class SignupRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


@router.post("/auth/signup")
def signup(req: SignupRequest, response: Response) -> dict:
    try:
        user = auth.create_account(req.name, req.email, req.password)
    except auth.AuthError as exc:
        raise HTTPException(400, str(exc)) from exc
    _set_cookie(response, auth.issue_session(user.id))
    return {"user": user.as_dict()}


@router.post("/auth/login")
def login(req: LoginRequest, response: Response) -> dict:
    try:
        user = auth.authenticate(req.email, req.password)
    except auth.AuthError as exc:
        # 401 for bad credentials, 429 when the backoff has kicked in.
        code = 429 if "Too many" in str(exc) else 401
        raise HTTPException(code, str(exc)) from exc
    _set_cookie(response, auth.issue_session(user.id))
    return {"user": user.as_dict()}


@router.post("/auth/logout")
def logout(response: Response,
           runway_session: str | None = Cookie(default=None)) -> dict:
    if runway_session:
        auth.revoke_session(runway_session)
    response.delete_cookie(COOKIE_NAME, path="/")
    return {"ok": True}


@router.get("/auth/me")
def me(user: auth.User | None = Depends(current_user_optional)) -> dict:
    return {"user": user.as_dict() if user else None,
            "face_available": face.available()}


# --------------------------------------------------------------------------- #
# Face enrolment
# --------------------------------------------------------------------------- #
@router.get("/face/status")
def face_status(user: auth.User = Depends(current_user)) -> dict:
    return {
        "available": face.available(),
        "enrolled": auth.has_face_template(user.id),
        "min_frames": face.MIN_ENROL_FRAMES,
        # Stated so the UI can be honest about what enrolment does and does not
        # provide, rather than implying it is a security measure.
        "note": ("Face matching is a convenience for attributing your own "
                 "walks. It is not identity verification and is not used to "
                 "sign you in."),
    }


@router.post("/face/enrol")
async def enrol(
    frames: list[UploadFile] = File(...),
    user: auth.User = Depends(current_user),
) -> dict:
    """Enrol from webcam captures the user just took of themselves.

    Frames are decoded, embedded, and discarded. Only the resulting centroid is
    stored - never an image.
    """
    if not face.available():
        raise HTTPException(503, "Face models are not installed on the server.")

    import cv2

    images = []
    for f in frames[:20]:
        raw = np.frombuffer(await f.read(), dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is not None:
            images.append(img)

    embeddings = face.embeddings_from_captures(images)
    if len(embeddings) < face.MIN_ENROL_FRAMES:
        raise HTTPException(
            422,
            f"Only {len(embeddings)} of {len(images)} captures showed a clear "
            f"face; {face.MIN_ENROL_FRAMES} are needed. Face the camera in even "
            "light and try again.")

    template = face.build_template(embeddings)
    auth.save_face_template(user.id, template, len(embeddings))
    return {"enrolled": True, "frames_used": len(embeddings)}


@router.delete("/face/enrol")
def unenrol(user: auth.User = Depends(current_user)) -> dict:
    """Delete face data without touching the account or its walk history."""
    return {"deleted": auth.delete_face_template(user.id)}
