"""Fetch a walk video from a share link (YouTube, Instagram, TikTok, ...).

Uses yt-dlp, which handles the extractor churn these platforms generate.

Three constraints shape this module:

**Format.** ffmpeg is not assumed to be installed, and yt-dlp needs it to merge
separate video and audio streams. Pose analysis never touches audio, so the
format selector deliberately prefers *progressive* single-file MP4 and falls
back to video-only rather than requesting anything that would need merging.

**Size.** A share link can point at a two-hour stream. Duration and file size
are capped before the download starts, using yt-dlp's metadata probe, because
the useful clip here is six to ten seconds.

**Where it points.** The API accepts URLs from the network and fetches them
server-side, which is a server-side request forgery vector if left open. Only
known video hosts are allowed, and private or loopback addresses are refused
outright.

Note on terms of service: YouTube and Instagram both restrict downloading in
their terms. This is intended for analysing your own footage. Whoever runs the
service is responsible for having the right to fetch what they point it at.
"""
from __future__ import annotations

import ipaddress
import shutil
import socket
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

#: Hosts the fetcher will accept. Extend deliberately - every entry is a place
#: the server can be made to send a request.
ALLOWED_HOSTS = {
    "youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com",
    "youtu.be", "youtube-nocookie.com", "www.youtube-nocookie.com",
    "instagram.com", "www.instagram.com",
    "tiktok.com", "www.tiktok.com", "vm.tiktok.com",
    "vimeo.com", "www.vimeo.com", "player.vimeo.com",
    "facebook.com", "www.facebook.com", "fb.watch",
    "twitter.com", "x.com", "www.x.com",
    "drive.google.com",
}

#: Display names, most useful first. The raw host list is the security
#: boundary; this is what a user should be told, and ordering it by usefulness
#: beats an alphabetical dump that leads with "drive.google.com".
PLATFORMS = ["YouTube", "Instagram", "TikTok", "Vimeo", "Facebook", "X", "Google Drive"]

MAX_DURATION_S = 300        # five minutes; a runway pass is a few seconds
MAX_FILESIZE_MB = 250
DOWNLOAD_TIMEOUT_S = 300


class FetchError(RuntimeError):
    """Raised when a link cannot be turned into a usable local video."""


@dataclass
class FetchedVideo:
    path: Path
    title: str
    duration_s: float
    source: str
    uploader: str = ""

    def as_dict(self) -> dict:
        return {"title": self.title, "duration_s": round(self.duration_s, 1),
                "source": self.source, "uploader": self.uploader}


def is_url(text: str) -> bool:
    """True if ``text`` looks like an http(s) URL rather than a file path."""
    try:
        p = urlparse(text.strip())
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


def _check_host(url: str) -> str:
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower()
    if not host:
        raise FetchError("that does not look like a valid link")

    if host not in ALLOWED_HOSTS:
        raise FetchError(
            f"'{host}' is not a supported source. Supported: "
            f"{', '.join(PLATFORMS)}. "
            "For anything else, download the file and upload it directly."
        )

    # Refuse anything resolving to a private or loopback address, so a
    # look-alike host cannot be used to reach the machine's own network.
    try:
        for info in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(info[4][0])
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                raise FetchError(f"'{host}' resolves to a non-public address")
    except socket.gaierror as exc:
        raise FetchError(f"could not resolve '{host}': {exc}") from exc

    return host


def _friendly(message: str) -> str:
    """Turn yt-dlp's diagnostics into something a user can act on."""
    m = message.lower()
    if "private" in m or "login required" in m or "log in" in m or "cookies" in m:
        return ("That video is private or needs a login. Make it public, or "
                "download it and upload the file directly.")
    if "unavailable" in m or "removed" in m or "does not exist" in m:
        return "That video is unavailable — check the link is correct and still live."
    if "age" in m and "restrict" in m:
        return "That video is age-restricted and cannot be fetched automatically."
    if "sign in to confirm" in m or "bot" in m or "403" in m or "forbidden" in m:
        return ("The platform blocked the request (bot check or HTTP 403 Forbidden). "
                "Download the video directly and upload the file.")
    if "unsupported url" in m:
        return "That link is not a video page — paste the direct link to the video."
    if "geo" in m and "restrict" in m:
        return "That video is not available in this region."
    return message.strip().removeprefix("ERROR: ")


def fetch_video(
    url: str,
    out_dir: Path | str | None = None,
    max_duration_s: int = MAX_DURATION_S,
    max_filesize_mb: int = MAX_FILESIZE_MB,
) -> FetchedVideo:
    """Download the video behind ``url`` and return where it landed."""
    try:
        from yt_dlp import YoutubeDL
        from yt_dlp.utils import DownloadError
    except ImportError as exc:
        raise FetchError(
            "yt-dlp is not installed. Run:  python -m pip install yt-dlp"
        ) from exc

    if not is_url(url):
        raise FetchError("that does not look like a link (expected http:// or https://)")
    host = _check_host(url)

    out_dir = Path(out_dir or tempfile.mkdtemp(prefix="runway_fetch_"))
    out_dir.mkdir(parents=True, exist_ok=True)

    opts = {
        # Progressive MP4 first so no merge - and therefore no ffmpeg - is
        # needed; then video-only MP4; then any video format; only then merged fallback.
        "format": (
            f"best[ext=mp4][acodec!=none][vcodec!=none][filesize<{max_filesize_mb}M]/"
            f"bestvideo[ext=mp4][filesize<{max_filesize_mb}M]/"
            f"best[ext=mp4][filesize<{max_filesize_mb}M]/"
            f"bestvideo/best"
        ),
        "extractor_args": {
            "youtube": {
                "player_client": ["android", "ios", "mweb", "web"]
            }
        },
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        # yt-dlp writes a carriage-return progress bar straight to stdout, which
        # corrupts CLI output and shows up in API logs as noise.
        "noprogress": True,
        "consoletitle": False,
        "noplaylist": True,
        "socket_timeout": 30,
        "retries": 2,
        "max_filesize": max_filesize_mb * 1024 * 1024,
        # Resolution beyond 1080p buys nothing: MediaPipe downsamples anyway and
        # the extra pixels cost download time and decode time.
        "format_sort": ["res:1080", "ext:mp4"],
    }

    if shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}

    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info.get("_type") == "playlist":
                entries = [e for e in (info.get("entries") or []) if e]
                if not entries:
                    raise FetchError("that link is an empty playlist")
                info = entries[0]
                url = info.get("webpage_url", url)

            duration = float(info.get("duration") or 0)
            if duration > max_duration_s:
                raise FetchError(
                    f"that video is {duration / 60:.0f} minutes long; the limit is "
                    f"{max_duration_s // 60} minutes. Trim it to the walk itself "
                    "and upload the file."
                )

            info = ydl.extract_info(url, download=True)
            path = Path(ydl.prepare_filename(info))

    except FetchError:
        raise
    except DownloadError as exc:
        raise FetchError(_friendly(str(exc))) from exc
    except Exception as exc:
        raise FetchError(_friendly(str(exc))) from exc

    if not path.exists():
        # yt-dlp remuxes some formats, so the predicted name can be stale.
        matches = sorted(out_dir.glob(f"{info.get('id', '*')}.*"))
        if not matches:
            raise FetchError("the download produced no file")
        path = matches[0]

    return FetchedVideo(
        path=path,
        title=info.get("title") or path.stem,
        duration_s=float(info.get("duration") or 0),
        source=host,
        uploader=info.get("uploader") or info.get("channel") or "",
    )
