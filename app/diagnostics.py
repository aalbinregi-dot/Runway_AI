"""What this installation can and cannot do, and what to do about the gaps.

Exists because "it doesn't work" is unactionable. This project spans two Python
interpreters, four downloadable model files and an optional Node app, and every
one of those can be half-installed. When something fails the useful question is
which capability is missing and what command fixes it - so each check reports a
remedy alongside its status, and the whole thing is one copyable block a user can
paste into a support request.

Deliberately excluded: filesystem paths, account names, walk data, anything from
the database. The paths on this machine contain the user's own name, and a
diagnostics report is the kind of thing people paste into public issue trackers.
Versions and yes/no capability flags are enough to diagnose every failure this
project actually produces.
"""
from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import asdict, dataclass, field


@dataclass
class Capability:
    """One thing the system can or cannot do."""

    key: str
    label: str
    ok: bool
    detail: str = ""
    #: What the user should run or do. Empty when nothing is wrong.
    remedy: str = ""
    #: False when the feature is optional - a missing one is not a fault.
    required: bool = True

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class Report:
    capabilities: list[Capability] = field(default_factory=list)
    versions: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "capabilities": [c.as_dict() for c in self.capabilities],
            "versions": self.versions,
            "summary": self.summary,
        }


def _version(module: str) -> str | None:
    if importlib.util.find_spec(module) is None:
        return None
    try:
        import importlib.metadata as md
        return md.version(module.replace("_", "-"))
    except Exception:
        try:
            return getattr(__import__(module), "__version__", "present")
        except Exception:
            return "present"


def _scoring() -> list[Capability]:
    from runway_ai.config import CRITERIA, launcher, video_env_hint

    out = []
    try:
        from runway_ai.models.train import load_models
        models = load_models() or {}
    except Exception as exc:
        models = {}
        out.append(Capability(
            "models_load", "Trained models load", False,
            f"loading raised {type(exc).__name__}",
            "The models were pickled by a different scikit-learn. Retrain with: "
            f"{launcher()} rebuild"))

    if models:
        kinds = {}
        for bundle in models.values():
            kinds[bundle["kind"]] = kinds.get(bundle["kind"], 0) + 1
        out.append(Capability(
            "models", "Trained models", len(models) == len(CRITERIA),
            f"{len(models)}/{len(CRITERIA)} criteria; "
            + ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())),
            "" if len(models) == len(CRITERIA)
            else f"Retrain with: {launcher()} rebuild"))
    elif not out:
        out.append(Capability(
            "models", "Trained models", False, "none found",
            f"Build the corpus and train: {launcher()} rebuild (about 5 minutes). "
            "Scoring falls back to the transparent rubric until then."))

    try:
        from runway_ai.scoring.rubric import CALIBRATED
        out.append(Capability(
            "rubric", "Rubric calibrated", bool(CALIBRATED),
            "tolerance bands fitted" if CALIBRATED else "using default bands",
            "" if CALIBRATED else "Fit the bands: python scripts/calibrate_rubric.py"))
    except Exception:
        pass

    try:
        from runway_ai.scoring import confidence
        ok = confidence.available()
        tiers = confidence.summary().get("by_tier", {}) if ok else {}
        out.append(Capability(
            "identifiability", "Per-criterion identifiability", ok,
            (", ".join(f"{len(v)} {k}" for k, v in sorted(tiers.items()))
             if ok else "not measured"),
            "" if ok else "Run the leakage audit so weakly measured criteria are "
                          "flagged: python scripts/leakage_audit.py",
            required=False))
    except Exception:
        pass
    return out


def _video() -> list[Capability]:
    out = []
    here = False
    try:
        from runway_ai.pose.extract import detect_backend
        backend = detect_backend()
        here = backend != "none"
    except Exception:
        backend = "none"

    sibling = None
    try:
        from runway_ai.pose.delegate import find_video_interpreter
        found = find_video_interpreter()
        sibling = found.parents[1].name if found else None
    except Exception:
        pass

    if here:
        detail = f"MediaPipe in this interpreter ({backend} API)"
    elif sibling:
        detail = f"delegated to the {sibling} environment"
    else:
        detail = "no MediaPipe anywhere"

    # Remedies are served over HTTP and rendered in the Guide page, so they have
    # to be runnable on the machine reading them rather than on the machine this
    # was written on - and they have to read as prose. Joining every line of the
    # terminal hint with " then " turned the trailing "or let the launcher do it"
    # into a third command in the sequence.
    from runway_ai.config import video_env_steps

    create, install, alternative = video_env_steps()
    out.append(Capability(
        "video", "Video analysis", bool(here or sibling), detail,
        "" if (here or sibling) else
        f"MediaPipe needs Python 3.9-3.12. Create the environment next to the "
        f"project: {create} then {install} - {alternative}."))

    try:
        from runway_ai.pose.extract import TASK_MODEL_PATH
        present = TASK_MODEL_PATH.exists()
        out.append(Capability(
            "pose_model", "Pose model file", present,
            "pose_landmarker_full.task present" if present else "not downloaded",
            "" if present else "Downloads automatically on first video analysis; "
                               "needs internet access once."))
    except Exception:
        pass
    return out


def _face() -> list[Capability]:
    out = []
    try:
        from runway_ai import face
        ok = face.available()
        out.append(Capability(
            "face", "Face recognition", ok,
            "YuNet + SFace present" if ok else "model files missing",
            "" if ok else "Fetch them once: python scripts/fetch_face_models.py",
            required=False))
    except Exception:
        out.append(Capability("face", "Face recognition", False, "unavailable",
                              "Requires OpenCV 4.8+", required=False))
    return out


def _links() -> list[Capability]:
    out = []
    try:
        from runway_ai.data import fetch
        hosts = len(getattr(fetch, "ALLOWED_HOSTS", []) or [])
        has = importlib.util.find_spec("yt_dlp") is not None
        out.append(Capability(
            "links", "Share-link analysis", has,
            f"{hosts} hosts allowed" if has else "yt-dlp not installed",
            "" if has else "pip install yt-dlp", required=False))
    except Exception:
        pass
    return out


def _audio() -> list[Capability]:
    """ffmpeg gates the unbuilt music features; reported so its absence is explicit."""
    import shutil

    found = shutil.which("ffmpeg")
    return [Capability(
        "ffmpeg", "Audio tools (ffmpeg)", bool(found),
        "on PATH" if found else "not installed",
        "" if found else "Only needed for audio features, none of which ship yet. "
                         "Nothing else is affected.",
        required=False)]


def build() -> Report:
    caps = _scoring() + _video() + _face() + _links() + _audio()

    versions = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "numpy": _version("numpy"),
        "scikit_learn": _version("scikit_learn") or _version("sklearn"),
        "opencv": _version("opencv_python") or _version("cv2"),
        "mediapipe": _version("mediapipe"),
        "fastapi": _version("fastapi"),
        "yt_dlp": _version("yt_dlp"),
    }
    # A sibling interpreter is where MediaPipe usually lives, and its absence
    # here is normal rather than a fault - say so instead of showing a bare null.
    if versions["mediapipe"] is None:
        versions["mediapipe"] = "not in this interpreter"

    broken = [c for c in caps if not c.ok and c.required]
    optional_off = [c for c in caps if not c.ok and not c.required]

    return Report(
        capabilities=caps,
        versions={k: v for k, v in versions.items() if v},
        summary={
            "healthy": not broken,
            "blocking": [c.key for c in broken],
            "optional_unavailable": [c.key for c in optional_off],
            "headline": ("Everything needed to score a walk is working."
                         if not broken else
                         f"{len(broken)} thing(s) need attention before "
                         f"analysis will work."),
        },
    )


def as_text(report: Report | None = None) -> str:
    """The copy-into-a-support-request form."""
    r = report or build()
    lines = ["Runway AI - diagnostics", ""]
    for c in r.capabilities:
        mark = "ok  " if c.ok else ("FAIL" if c.required else "off ")
        lines.append(f"  [{mark}] {c.label}: {c.detail}")
        if c.remedy:
            lines.append(f"         -> {c.remedy}")
    lines += ["", "  versions:"]
    for k, v in r.versions.items():
        lines.append(f"    {k}: {v}")
    lines += ["", f"  {r.summary['headline']}"]
    return "\n".join(lines)
