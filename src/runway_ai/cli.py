"""Command-line interface.

    python -m runway_ai analyze  <video|keypoints|url> [--user NAME] [--save-report]
    python -m runway_ai progress --user NAME
    python -m runway_ai info
"""
from __future__ import annotations

import argparse
import json
import sys

import pandas as pd
from pathlib import Path

from .config import CRITERIA, REPORT_DIR

# `analyze` and the report writer are imported inside the commands that use them,
# not here. Importing them at module scope pulls in scikit-learn and scipy, which
# made `python -m runway_ai extract` - the one command the *video* environment
# runs - refuse to start unless that environment also carried the whole analysis
# stack. requirements-video.txt does not, and installing it as documented
# therefore produced an environment in which extraction failed at import time,
# before it ever reached MediaPipe.


def cmd_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze
    from .feedback import report as report_mod

    try:
        results = analyze(args.input, fps=args.fps)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for r in results:
        if args.json:
            print(json.dumps(r.as_dict(), indent=2))
        else:
            print(report_mod.to_text(r))

        if args.save_report:
            paths = report_mod.save(r, Path(args.out or REPORT_DIR), args.formats)
            for p in paths:
                print(f"wrote {p}")

        if args.user:
            from .db import store
            uid = store.get_or_create_user(args.user)
            sid = store.save_session(uid, r.video_id, r.assessment, r.features,
                                     source_path=str(args.input))
            print(f"saved session {sid} for user '{args.user}' (id {uid})")
    return 0


def cmd_extract(args: argparse.Namespace) -> int:
    """Video -> keypoint CSV, with no scoring.

    Kept as its own command because pose estimation and scoring often have to
    run in *different* interpreters: MediaPipe pins Python to 3.9-3.12, while
    the trained models are pickled against whatever scikit-learn the analysis
    environment has. Writing keypoints to disk in between means neither
    constraint leaks into the other.
    """
    from .pose.extract import extract_from_video

    src = Path(args.input)
    out = Path(args.output) if args.output else src.with_suffix(".keypoints.csv")
    # Progress goes to stderr in a fixed machine-readable form. The delegate in
    # pose/delegate.py runs this as a subprocess and relays these lines to the
    # API, which is the only way progress survives the interpreter hop.
    def emit(done: int, total: int) -> None:
        print(f"PROGRESS pose {done} {total}", file=sys.stderr, flush=True)

    try:
        seq = extract_from_video(src, video_id=args.video_id or src.stem,
                                 stride=args.stride,
                                 on_progress=emit if args.progress_events else None)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    out.parent.mkdir(parents=True, exist_ok=True)
    seq.to_long().to_csv(out, index=False)

    # The seven-column schema has nowhere to record the frame rate, and fps
    # drives every cadence and rhythm feature. Rather than widen the schema (and
    # break compatibility with the supplied workbook) or make the caller
    # remember to pass --fps, write a sidecar that `analyze` picks up
    # automatically.
    meta = {"fps": seq.fps, "video_id": seq.video_id, "source": str(src),
            "n_frames": seq.n_frames,
            "frame_size": seq.meta.get("frame_size"),
            **{k: v for k, v in seq.meta.items()
               if isinstance(v, (int, float, str))}}
    sidecar = out.with_suffix(out.suffix + ".meta.json")
    sidecar.write_text(json.dumps(meta, indent=2))

    print(f"wrote {out}  ({seq.n_frames} frames x {len(seq.landmarks)} landmarks, "
          f"{seq.fps:.1f} fps)")
    print(f"wrote {sidecar}")
    return 0


def cmd_extract_multi(args: argparse.Namespace) -> int:
    """Every walker in a group clip -> one keypoint CSV each, plus a manifest.

    Exists for the same reason as ``extract``: pose estimation needs MediaPipe
    and scoring needs the trained models, and those two rarely live in the same
    interpreter. The manifest carries each walker's face template across the
    process boundary so the scoring side can attribute walks to accounts
    without opening the video a second time.
    """
    import numpy as np

    from .pose.multi import extract_all

    src = Path(args.input)
    out_dir = Path(args.output or src.parent / f"{src.stem}_people")
    out_dir.mkdir(parents=True, exist_ok=True)

    def emit(done: int, total: int) -> None:
        print(f"PROGRESS pose {done} {total}", file=sys.stderr, flush=True)

    # Each walker is written the moment their pose is extracted, and announced on
    # stderr, so the parent process can score them while this one carries on with
    # the next. Waiting for the whole list first is what made the web client sit
    # blank for eighty seconds and then fill in all at once.
    written: list[dict] = []

    def entry_for(seq, template) -> dict:
        import numpy as np

        csv_path = out_dir / f"{seq.video_id}.csv"
        seq.to_long().to_csv(csv_path, index=False)
        return {
            "video_id": seq.video_id,
            "person_index": seq.meta.get("person_index"),
            "csv": csv_path.name,
            "fps": seq.fps,
            "n_frames": seq.n_frames,
            "frame_size": seq.meta.get("frame_size"),
            "face_template": np.asarray(template, dtype=float).ravel().tolist(),
        }

    def announce(seq, template, done, total) -> None:
        # person_index is only assigned once every walker is known and they can
        # be ordered left to right, so the streamed id is provisional.
        entry = entry_for(seq, template)
        written.append(entry)
        print(f"PERSON {done} {total} {entry['csv']}", file=sys.stderr, flush=True)

    try:
        people = extract_all(src, stride=args.stride,
                             on_progress=emit if args.progress_events else None,
                             on_person=announce if args.progress_events else None)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if not people:
        print("error: no walkers with a visible face were found", file=sys.stderr)
        return 1

    # Written once. `announce` already produced each CSV during extraction when
    # progress events were on; without them nothing has been written yet.
    by_id = {e["video_id"]: e for e in written}   # keyed on the stable track id
    manifest = {"source": str(src), "people": []}
    for seq, template in people:
        entry = by_id.get(seq.video_id) or entry_for(seq, template)
        # person_index is only known once every walker is ordered left to right,
        # which happens after the last one is extracted.
        entry["person_index"] = seq.meta.get("person_index")
        manifest["people"].append(entry)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"found {len(people)} walker(s); wrote {out_dir}")
    for p in manifest["people"]:
        print(f"  {p['video_id']}: {p['n_frames']} frames @ {p['fps']:.1f} fps")
    return 0


def cmd_render(args: argparse.Namespace) -> int:
    """Burn the skeleton, live measurements and scores onto the source video."""
    from .data.loaders import preprocess
    from .data.schema import long_to_sequences
    from .features.builder import extract_features
    from .models.train import load_models
    from .scoring.assess import assess
    from .feedback.coach import build_plan
    from .analyze import AnalysisResult, _sidecar_fps
    from .viz.overlay import render

    video = Path(args.input)
    if not video.exists():
        print(f"error: no such video: {video}", file=sys.stderr)
        return 1

    kp = Path(args.keypoints) if args.keypoints else None
    if kp is None or not kp.exists():
        print("error: need extracted keypoints. Run:\n"
              f"  python -m runway_ai extract \"{video}\" -o keypoints.csv",
              file=sys.stderr)
        return 1

    fps = args.fps or _sidecar_fps(kp) or 30.0
    seqs = long_to_sequences(pd.read_csv(kp), fps=fps)
    seq_image = next(iter(seqs.values()))

    # preprocess() mutates in place, so the image-space copy is taken first -
    # the overlay needs original pixel coordinates, the metrics need normalised
    # ones.
    import copy
    seq_norm = preprocess(copy.deepcopy(seq_image))

    feats = extract_features(seq_norm)
    a = assess(feats, load_models() or None)
    result = AnalysisResult(seq_image.video_id, a, build_plan(a), feats)

    out = Path(args.output) if args.output else video.with_name(video.stem + "_analysed.mp4")
    print(f"Rendering {seq_image.n_frames} frames...", flush=True)
    render(video, seq_image, result, out, seq_norm=seq_norm, max_side=args.max_side)
    print(f"wrote {out}")
    print(f"  overall {a.overall:.1f}/100 ({a.grade})")
    return 0


def cmd_accounts(args: argparse.Namespace) -> int:
    """List accounts. Operator-only, deliberately not an HTTP route."""
    from .db import auth

    auth.migrate()
    rows = auth.list_accounts()
    if not rows:
        print("No accounts yet.")
        return 0
    print(f"  {'id':>4}  {'name':22s} {'email':26s} {'walks':>5}  flags")
    for r in rows:
        flags = []
        if r["is_legacy"]:
            flags.append("legacy")
        if r["has_face"]:
            flags.append("face")
        print(f"  {r['user_id']:>4}  {(r['name'] or '')[:22]:22s} "
              f"{(r['email'] or '-')[:26]:26s} {r['walks']:>5}  {','.join(flags) or '-'}")
    if any(r["is_legacy"] for r in rows):
        print("\n  'legacy' rows predate accounts and cannot sign in. Move their")
        print("  walks onto a real account with:")
        print("      python -m runway_ai claim --from <id> --into <id>")
    return 0


def cmd_claim(args: argparse.Namespace) -> int:
    """Move a legacy account's walks onto a registered account.

    Not exposed over HTTP on purpose: the only thing such an endpoint could
    check is possession of a name, and names are neither unique nor secret.
    """
    from .db import auth

    auth.migrate()
    try:
        moved = auth.claim_legacy(args.from_id, args.into)
    except auth.AuthError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"moved {moved} walk(s) from account {args.from_id} to {args.into}")
    return 0


def cmd_progress(args: argparse.Namespace) -> int:
    from .db import store

    # Deliberately a lookup, not get-or-create: showing progress is a read, and
    # a mistyped --user used to add an empty account to the database instead of
    # saying the name was unknown.
    uid = store.find_user(args.user)
    if uid is None:
        print(f"error: no user named '{args.user}'. Existing accounts:",
              file=sys.stderr)
        from .db import auth
        auth.migrate()
        for row in auth.list_accounts():
            print(f"  {row['name']}", file=sys.stderr)
        return 1
    prog = store.progress(uid)
    if args.json:
        print(json.dumps(prog, indent=2))
        return 0

    if prog.get("sessions", 0) < 2:
        print(prog.get("message", "No history yet."))
        return 0

    print(f"Sessions: {prog['sessions']}")
    print(f"Overall:  {prog['first_overall']} -> {prog['latest_overall']} "
          f"({prog['overall_delta']:+.1f})")
    print("\nPer-criterion change:")
    for c in CRITERIA:
        d = prog["per_criterion_delta"].get(c)
        if d is None:
            continue
        mark = "+" if d > 0 else ("-" if d < 0 else " ")
        print(f"  {mark} {c:22s} {d:+.2f}")
    if prog["most_improved"]:
        print("\nMost improved:   " + ", ".join(prog["most_improved"]))
    if prog["needs_attention"]:
        print("Needs attention: " + ", ".join(prog["needs_attention"]))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    from .models.train import load_models
    from .pose.extract import mediapipe_available
    from .scoring.rubric import CALIBRATED

    models = load_models()
    print("Runway AI - system status")
    print(f"  trained models   : {len(models)}/{len(CRITERIA)} criteria")
    if models:
        kinds = {}
        for c, b in models.items():
            kinds[b['kind']] = kinds.get(b['kind'], 0) + 1
        print(f"  estimators       : " + ", ".join(f"{k} x{v}" for k, v in sorted(kinds.items())))
        print(f"  feature count    : {len(next(iter(models.values()))['features'])}")
    print(f"  rubric calibrated: {CALIBRATED}")

    from .config import PROJECT_ROOT
    from .pose.extract import detect_backend

    backend = detect_backend()
    if backend != "none":
        print(f"  video backend    : mediapipe ({backend} API), in this interpreter")
        return 0

    # This interpreter cannot read video, but the launcher and the API route
    # video through a sibling environment. Reporting a bare "UNAVAILABLE" here
    # would be misleading when that environment exists and works.
    #
    # Asks config rather than repeating the search: this had its own copy that
    # looked only for Scripts/python.exe, so on a correctly configured Mac it
    # reported the video backend as unavailable and then printed Windows-only
    # instructions for fixing a problem that did not exist. Imported from config,
    # not from pose.delegate, so `info` stays runnable without pandas - which is
    # the state somebody running `info` is most likely to be in.
    from .config import find_video_interpreter, launcher

    sibling = find_video_interpreter()
    if sibling:
        print(f"  video backend    : not here, but {sibling.parents[1].name} is set up")
        print(f"                     {launcher()} routes video files to it automatically")
    else:
        print("  video backend    : UNAVAILABLE")
        from .pose.extract import _install_hint
        print("\n" + _install_hint())
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="runway_ai",
                                 description="AI runway performance analysis and coaching")
    sub = ap.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="score a walk from video or keypoints")
    a.add_argument("input", help="video, keypoint table, or a YouTube/Instagram/TikTok link")
    a.add_argument("--user", help="store the result against this user")
    a.add_argument("--fps", type=float, default=None, help="fps for keypoint input")
    a.add_argument("--json", action="store_true")
    a.add_argument("--save-report", action="store_true")
    a.add_argument("--out", default=None)
    a.add_argument("--formats", nargs="+", default=["txt", "md", "html", "json"])
    a.set_defaults(func=cmd_analyze)

    e = sub.add_parser("extract", help="video -> keypoint CSV (no scoring)")
    e.add_argument("input")
    e.add_argument("-o", "--output", default=None)
    e.add_argument("--video-id", default=None)
    e.add_argument("--progress-events", action="store_true",
                   help="emit 'PROGRESS pose done total' lines on stderr")
    e.add_argument("--stride", type=int, default=1,
                   help="process every Nth frame (2 halves the time, halves cadence resolution)")
    e.set_defaults(func=cmd_extract)

    em = sub.add_parser("extract-multi", help="group video -> one keypoint CSV per walker")
    em.add_argument("input")
    em.add_argument("-o", "--output", default=None, help="output directory")
    em.add_argument("--stride", type=int, default=4,
                    help="scan every Nth frame for faces (4 is a good default)")
    em.add_argument("--progress-events", action="store_true")
    em.set_defaults(func=cmd_extract_multi)

    v = sub.add_parser("render", help="annotated video: skeleton + live metrics + scores")
    v.add_argument("input", help="the original video")
    # The help text used to promise "(default: data/interim/_last_video.csv)".
    # Nothing implemented it - cmd_render treats a missing -k as a hard error -
    # and implementing it would have been worse: a stale _last_video.csv from a
    # previous clip would have been drawn over this one. The launcher supplies the
    # path, so the flag is simply required.
    v.add_argument("-k", "--keypoints", default=None,
                   help="extracted keypoint CSV (required; run `extract` first)")
    v.add_argument("-o", "--output", default=None)
    v.add_argument("--fps", type=float, default=None)
    v.add_argument("--max-side", type=int, default=720,
                   help="downscale the long edge to this many pixels")
    v.set_defaults(func=cmd_render)

    ac = sub.add_parser("accounts", help="list accounts (operator only)")
    ac.set_defaults(func=cmd_accounts)

    cl = sub.add_parser("claim", help="move a legacy account's walks onto a real one")
    cl.add_argument("--from", dest="from_id", type=int, required=True,
                    help="legacy account id, from `runway_ai accounts`")
    cl.add_argument("--into", type=int, required=True, help="target account id")
    cl.set_defaults(func=cmd_claim)

    p = sub.add_parser("progress", help="show a user's improvement history")
    p.add_argument("--user", required=True)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_progress)

    i = sub.add_parser("info", help="show system status")
    i.set_defaults(func=cmd_info)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
