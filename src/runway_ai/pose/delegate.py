"""Run pose extraction in a sibling interpreter when this one cannot.

MediaPipe pins Python to 3.9-3.12; the analysis environment is often newer. The
CLI works around that by asking the user to invoke the right interpreter, but
the API and the web client cannot - a video uploaded through the browser has to
just work.

So when MediaPipe is missing locally and a ``.venv311``/``.venv312`` exists next
to the project, extraction is delegated to it as a subprocess and the keypoints
come back as a CSV. Scoring still happens in *this* interpreter, which keeps the
trained models from ever being unpickled against a different scikit-learn build.
"""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path

import pandas as pd

from ..config import PROJECT_ROOT, video_env_hint
from ..config import find_video_interpreter as find_video_interpreter
from ..data.schema import PoseSequence, long_to_sequences

# `find_video_interpreter` is re-exported rather than defined here. It used to
# live in this module and try `Scripts/python.exe` before `bin/python`, which
# looks portable and is not: a project folder copied from Windows carries its
# `.venv311/Scripts/python.exe`, `exists()` is true for it on macOS, and every
# video upload then died with "Exec format error" - while app/diagnostics.py
# reported the video pipeline as ready, because it asks this same function.
# Creating a correct `.venv311/bin/python` alongside did not help either: the
# stale `.exe` still won the search. It now lives in config.py, selected by
# platform, and importable without pulling pandas into a status check.


def _no_environment_error() -> RuntimeError:
    return RuntimeError(
        "No MediaPipe environment found. Create one next to the project:\n"
        + video_env_hint("    ")
    )


def extract_delegated(
    video: str | Path,
    video_id: str | None = None,
    python_exe: Path | None = None,
    timeout: int = 900,
    on_progress=None,
) -> PoseSequence:
    """Extract keypoints from ``video`` using the sibling interpreter."""
    python_exe = python_exe or find_video_interpreter()
    if python_exe is None:
        raise _no_environment_error()

    video = Path(video).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "keypoints.csv"
        cmd = [str(python_exe), "-m", "runway_ai", "extract", str(video),
               "-o", str(out), *(["--video-id", video_id] if video_id else [])]
        if on_progress:
            cmd.append("--progress-events")

        proc = _run_streaming(cmd, timeout, on_progress)
        if proc.returncode != 0 or not out.exists():
            # The child's stderr carries the actionable message ("only 12 of 300
            # frames had a detectable person"), so surface it rather than a
            # generic non-zero exit.
            msg = (proc.stderr or proc.stdout or "").strip().splitlines()
            detail = msg[-1] if msg else f"exit code {proc.returncode}"
            raise RuntimeError(f"pose extraction failed: {detail}")

        fps, frame_size = 30.0, None
        sidecar = out.with_suffix(out.suffix + ".meta.json")
        if sidecar.exists():
            try:
                info = json.loads(sidecar.read_text())
                fps = float(info.get("fps", fps))
                frame_size = info.get("frame_size")
            except Exception:
                pass

        seqs = long_to_sequences(pd.read_csv(out), fps=fps)

    seq = next(iter(seqs.values()))
    seq.meta["backend"] = f"delegated:{python_exe.parents[1].name}"
    if frame_size:
        seq.meta["frame_size"] = frame_size
    return seq


def extract_multi_delegated(
    video: str | Path,
    stride: int = 4,
    python_exe: Path | None = None,
    timeout: int = 1800,
    on_progress=None,
    on_person=None,
) -> list[tuple[PoseSequence, list]]:
    """Every walker in a group clip, extracted in the MediaPipe environment.

    Returns ``(sequence, face_template)`` pairs. The timeout is double the
    single-person one: a group clip runs pose estimation once per walker.
    """
    python_exe = python_exe or find_video_interpreter()
    if python_exe is None:
        raise _no_environment_error()

    video = Path(video).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "people"
        cmd = [str(python_exe), "-m", "runway_ai", "extract-multi", str(video),
               "-o", str(out_dir), "--stride", str(stride)]
        if on_progress:
            cmd.append("--progress-events")

        # The child writes each walker's CSV as it finishes and prints
        # "PERSON done total file". Loading it here means the parent scores
        # walker one while the child is still extracting walker two - pose
        # estimation takes ~40s per person against ~0.2s to score, so without
        # this the client waits for everything and then fills in at once.
        def relay(line: str) -> None:
            if not on_person:
                return
            try:
                _, done, total, name = line.split(maxsplit=3)
                seq = _load_person(out_dir / name.strip())
                if seq is not None:
                    on_person(seq, int(done), int(total))
            except Exception:
                pass

        proc = _run_streaming(cmd, timeout, on_progress, on_line=relay)
        manifest_path = out_dir / "manifest.json"
        if proc.returncode != 0 or not manifest_path.exists():
            msg = (proc.stderr or proc.stdout or "").strip().splitlines()
            detail = msg[-1] if msg else f"exit code {proc.returncode}"
            raise RuntimeError(f"group extraction failed: {detail}")

        manifest = json.loads(manifest_path.read_text())
        people = []
        for entry in manifest.get("people", []):
            csv_path = out_dir / entry["csv"]
            if not csv_path.exists():
                continue
            seqs = long_to_sequences(pd.read_csv(csv_path),
                                     fps=float(entry.get("fps") or 30.0))
            seq = next(iter(seqs.values()))
            seq.video_id = entry["video_id"]
            seq.meta["person_index"] = entry.get("person_index")
            seq.meta["backend"] = f"delegated:{python_exe.parents[1].name}"
            if entry.get("frame_size"):
                seq.meta["frame_size"] = entry["frame_size"]
            people.append((seq, entry.get("face_template") or []))

    if not people:
        raise RuntimeError("no walkers with a visible face were found")
    return people


def _load_person(csv_path: Path) -> PoseSequence | None:
    """Read one streamed walker's keypoints, without waiting for the manifest."""
    if not csv_path.exists():
        return None
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        return None
    seqs = long_to_sequences(frame, fps=30.0)
    if not seqs:
        return None
    seq = next(iter(seqs.values()))
    seq.video_id = csv_path.stem
    return seq


class _Completed:
    """Minimal stand-in for CompletedProcess, carrying what callers read."""

    def __init__(self, returncode: int, stdout: str, stderr: str):
        self.returncode, self.stdout, self.stderr = returncode, stdout, stderr


def _run_streaming(cmd: list[str], timeout: int, on_progress,
                   on_line=None) -> _Completed:
    """Run the child and relay its PROGRESS lines as they appear.

    ``subprocess.run`` buffers until exit, which would make progress arrive all
    at once at the end - useless. Reading stderr line by line is what lets the
    caller show a live frame count during a run that takes tens of seconds.
    """
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(PROJECT_ROOT), env=_child_env(),
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
        )
    except OSError as exc:
        # The interpreter exists but cannot run here - the classic case is a
        # virtualenv built on another platform, which raises errno 8, "Exec
        # format error". Uncaught, that reached the browser verbatim as a 422.
        raise RuntimeError(
            f"the MediaPipe environment at {cmd[0]} cannot be executed on this "
            f"platform ({exc}). It was probably created on a different operating "
            f"system; delete it and recreate it:\n" + video_env_hint("    ")
        ) from exc
    err_lines: list[str] = []
    try:
        for line in proc.stderr:
            line = line.rstrip()
            if line.startswith("PROGRESS pose "):
                if on_progress:
                    try:
                        _, _, done, total = line.split()
                        on_progress(int(done), int(total))
                    except ValueError:
                        pass
                continue
            if line.startswith("PERSON ") and on_line:
                on_line(line)
                continue
            err_lines.append(line)
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        raise RuntimeError("pose extraction timed out") from None
    finally:
        if proc.stderr:
            proc.stderr.close()

    stdout = proc.stdout.read() if proc.stdout else ""
    if proc.stdout:
        proc.stdout.close()
    return _Completed(proc.returncode, stdout, "\n".join(err_lines))


def _child_env() -> dict:
    """Environment for the subprocess: PYTHONPATH must reach src/."""
    env = dict(os.environ)
    src = str(PROJECT_ROOT / "src")
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = src if not existing else f"{src}{os.pathsep}{existing}"
    return env
