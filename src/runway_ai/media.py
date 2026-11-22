"""Short-lived store for videos fetched from a link.

A linked video is downloaded and scored server-side, so the browser never holds
those pixels and cannot draw the skeleton over them. Keeping the file briefly
and serving it back is what makes the evidence view work for links as well as
uploads.

Two things this must not become: a way to read arbitrary files, or a disk leak.
Entries are keyed by an unguessable token that maps to a path the server chose -
nothing from the request ever reaches the filesystem - and every access sweeps
out anything past its TTL.
"""
from __future__ import annotations

import secrets
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR

MEDIA_DIR = DATA_DIR / "media"
TTL_S = 45 * 60          # long enough to review a result, short enough to forget
MAX_ENTRIES = 24


@dataclass
class Entry:
    path: Path
    added: float
    content_type: str


_entries: dict[str, Entry] = {}
_lock = threading.Lock()


def _sweep(now: float) -> None:
    """Drop expired entries, then the oldest if still over the cap."""
    for token in [t for t, e in _entries.items() if now - e.added > TTL_S]:
        _drop(token)
    while len(_entries) > MAX_ENTRIES:
        _drop(min(_entries, key=lambda t: _entries[t].added))


def _drop(token: str) -> None:
    entry = _entries.pop(token, None)
    if entry:
        entry.path.unlink(missing_ok=True)


CONTENT_TYPES = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".webm": "video/webm", ".mkv": "video/x-matroska", ".avi": "video/x-msvideo",
}


def store(source: Path | str) -> str:
    """Move ``source`` into the store and return its access token."""
    source = Path(source)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    token = secrets.token_urlsafe(24)
    dest = MEDIA_DIR / f"{token}{source.suffix.lower() or '.mp4'}"
    shutil.move(str(source), dest)

    with _lock:
        _sweep(time.time())
        _entries[token] = Entry(
            path=dest, added=time.time(),
            content_type=CONTENT_TYPES.get(dest.suffix, "application/octet-stream"),
        )
    return token


def resolve(token: str) -> Entry | None:
    """Look up a token. Returns None if unknown or expired."""
    with _lock:
        _sweep(time.time())
        entry = _entries.get(token)
    if entry and entry.path.exists():
        return entry
    return None


def clear() -> None:
    """Drop everything. Used by tests and on shutdown."""
    with _lock:
        for token in list(_entries):
            _drop(token)
