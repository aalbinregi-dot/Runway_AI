"""Preflight check for the API the web client talks to.

A missing python-multipart makes every file upload fail with a 400 that says
nothing useful, and a missing uvicorn produces a bare ModuleNotFoundError. This
turns both into one actionable line before the server starts.

Usage:  python scripts/check_env.py api
"""
from __future__ import annotations

import sys


def _version_tuple(v: str) -> tuple[int, ...]:
    out = []
    for part in v.split("."):
        digits = "".join(c for c in part if c.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out) or (0,)


def fail(msg: str, fix: str) -> int:
    print()
    print(f"  {msg}")
    print()
    for line in fix.splitlines():
        print(f"      {line}")
    print()
    return 1


def check_api() -> int:
    missing = []
    for mod, pkg in [("fastapi", "fastapi"), ("uvicorn", "uvicorn[standard]"),
                     ("multipart", "python-multipart")]:
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        return fail(
            f"Missing packages for the API: {', '.join(missing)}",
            "python -m pip install " + " ".join(f'"{m}"' for m in missing),
        )
    return 0


def main() -> int:
    return check_api()


if __name__ == "__main__":
    raise SystemExit(main())
