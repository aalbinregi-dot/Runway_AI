#!/usr/bin/env bash
# ===========================================================================
#  Runway Performance AI - launcher (macOS / Linux)
#
#  The exact counterpart of run.bat. Same commands, same two-interpreter
#  routing, same menu:
#
#      ./run.sh                        interactive menu
#      ./run.sh info | test | verify | sample | results | sanity | diagnose
#      ./run.sh analyze <file> [flags] | group <video> | render <video>
#      ./run.sh web | api | train | rebuild | sensitivity | setup | clean
#
#  Three things this handles that are easy to get wrong by hand:
#
#    * PYTHONPATH must point at src/ or "python -m runway_ai" will not resolve.
#    * Video input needs MediaPipe, which publishes wheels for CPython 3.9-3.12
#      only. On macOS that is usually a *non*-problem: Homebrew's python@3.12
#      satisfies both the analysis pipeline and MediaPipe, so `setup` puts
#      everything in one .venv and no delegation is needed. If the main
#      interpreter is newer, a second environment is created and video files are
#      routed to it, exactly as on Windows.
#    * A folder copied from Windows brings a Windows .venv with it. Its
#      Scripts/python.exe cannot run here, and something has to say so rather
#      than fail with "Exec format error".
#
#  Fresh Mac, from nothing:
#      chmod +x run.sh          # once, if the executable bit did not survive
#      ./run.sh setup           # checks the toolchain, builds the environments
#      ./run.sh verify          # proves the install
#      ./run.sh web             # the app
#
#  If this file arrived with Windows line endings the shell says either
#  "env: bash\r: No such file or directory" or "$'\r': command not found". Fix it
#  with tr, not sed - BSD sed does not understand \r in a pattern, so the obvious
#  `sed 's/\r$//'` is a silent no-op on macOS:
#
#      tr -d '\r' < run.sh > run.sh.tmp && mv run.sh.tmp run.sh && chmod +x run.sh
# ===========================================================================

# Work from this file's directory, whatever it was launched from - but keep where
# the user was first, because the cd throws it away. run.bat has the same blind
# spot and gets away with it by being double-clicked; on macOS
# `./run.sh analyze clip.mov` from another directory is the normal way in, and
# resolving that against the project root would report a file the user is looking
# straight at as missing.
#
# Everything below is quoted throughout: the usual home for this folder - iCloud,
# OneDrive, Desktop - has spaces in the path.
ORIG_PWD="$PWD"
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"

export PYTHONPATH="$ROOT/src"

# --- interpreters ---------------------------------------------------------- #

# Order matches config.VENV_NAMES, so the launcher and the library agree on which
# environment is "the video one" when a machine has more than one.
VENV_NAMES=".venv312 .venv311 .venv310 .venv39"

PY=""
PY_VERSION=""
PYVID=""
PYVID_VERSION=""
STRANDED=""          # venvs built on Windows, unusable here

# Echo "3.12" if the argument is a working Python 3, or fail.
#
# Existence is not enough, and this is not a hypothetical: a Mac without the
# Xcode Command Line Tools has a /usr/bin/python3 that is a *stub* - it exists,
# it is executable, and running it opens an installer prompt instead of a Python.
# Windows has the same trap in reverse, a WindowsApps/python3 shim that prints an
# advert for the Microsoft Store and exits 0. Both must be rejected here rather
# than three commands later.
usable_python() {
  [ -n "$1" ] || return 1
  v="$("$1" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)"
  case "$v" in
    3.*) printf '%s' "$v"; return 0 ;;
  esac
  return 1
}

find_interpreters() {
  PY=""
  PY_VERSION=""
  PYVID=""
  PYVID_VERSION=""
  STRANDED=""

  for candidate in "$ROOT/.venv/bin/python3" "$ROOT/.venv/bin/python"; do
    if [ -z "$PY" ] && [ -x "$candidate" ]; then
      PY_VERSION="$(usable_python "$candidate")" && PY="$candidate"
    fi
  done
  if [ -z "$PY" ] && [ -f ".venv/Scripts/python.exe" ]; then
    STRANDED="$STRANDED .venv"
  fi

  # Then whatever is on PATH. `python` is accepted only if it reports 3.x: on
  # macOS a bare `python` is either absent or a Python 2 from an old install.
  if [ -z "$PY" ]; then
    for candidate in python3 python; do
      if command -v "$candidate" >/dev/null 2>&1; then
        resolved="$(command -v "$candidate")"
        if PY_VERSION="$(usable_python "$resolved")"; then
          PY="$resolved"
          break
        fi
      fi
    done
  fi
  [ -n "$PY" ] || PY_VERSION=""

  for name in $VENV_NAMES; do
    if [ -z "$PYVID" ]; then
      for candidate in "$ROOT/$name/bin/python3" "$ROOT/$name/bin/python"; do
        if [ -z "$PYVID" ] && [ -x "$candidate" ]; then
          PYVID_VERSION="$(usable_python "$candidate")" && PYVID="$candidate"
        fi
      done
    fi
    if [ -f "$name/Scripts/python.exe" ] && [ ! -x "$name/bin/python" ] \
       && [ ! -x "$name/bin/python3" ]; then
      STRANDED="$STRANDED $name"
    fi
  done
  [ -n "$PYVID" ] || PYVID_VERSION=""
}

# Does the main interpreter read video itself? When setup has put MediaPipe in
# the single .venv - the normal outcome on macOS - there is no second
# environment and no two-step routing.
main_has_mediapipe() {
  [ -n "$PY" ] || return 1
  "$PY" - <<'PY' >/dev/null 2>&1
import sys
try:
    from runway_ai.pose.extract import mediapipe_available
except Exception:
    sys.exit(1)
sys.exit(0 if mediapipe_available() else 1)
PY
}

# --- presentation ---------------------------------------------------------- #

if [ -t 1 ]; then
  B="$(printf '\033[1m')"; DIM="$(printf '\033[2m')"; R="$(printf '\033[0m')"
else
  B=""; DIM=""; R=""
fi

say()  { printf '%s\n' "$*"; }
head2() { printf '\n%s\n\n' "  ${B}--- $* ---${R}"; }
warn() { printf '\n  %s\n' "$*"; }

# The macOS instructions for building the video environment, printed wherever
# run.bat prints the `py -3.11` ones. Kept in one function so the two paths
# through the script cannot drift.
video_env_help() {
  say ""
  say "  Video input needs MediaPipe, which supports CPython 3.9-3.12 only."
  say ""
  say "  The simplest fix on macOS is one environment on 3.12:"
  say "      brew install python@3.12"
  say "      python3.12 -m venv .venv"
  say "      .venv/bin/pip install -r requirements.txt -r requirements-video.txt"
  say ""
  say "  Or keep your current .venv and add a second environment beside it:"
  say "      brew install python@3.12"
  say "      python3.12 -m venv .venv312"
  say "      .venv312/bin/pip install -r requirements-video.txt"
  say ""
  say "  ./run.sh setup does either of these for you, and this launcher picks"
  say "  the environment up automatically once it exists."
}

# The Homebrew bootstrap, in full. The installer itself does NOT put brew on PATH:
# on Apple Silicon the prefix is /opt/homebrew, and path_helper only seeds
# /usr/local/bin from /etc/paths. Printing `brew install ...` without the shellenv
# line therefore hands an Apple Silicon user a `command not found` for the very
# command the previous line told them would now exist - while working fine on an
# Intel Mac, which is how instructions like this survive review.
brew_bootstrap_help() {
  say "      xcode-select --install       # once, for the compiler toolchain"
  say '      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
  say '      eval "$(/opt/homebrew/bin/brew shellenv)" 2>/dev/null || \'
  say '        eval "$(/usr/local/bin/brew shellenv)"    # Apple Silicon or Intel'
  say "      brew install python@3.12 node"
}

stranded_warning() {
  [ -n "$STRANDED" ] || return 0
  warn "${B}This folder carries a Windows virtualenv:${R}${STRANDED}"
  say  "  Those hold Scripts/python.exe, which cannot run on macOS. They are"
  say  "  ignored here; delete them and let ./run.sh setup rebuild:"
  say  "      rm -rf${STRANDED}"
  say  ""
}

# Run one labelled step; stop on failure. The batch file's :RUNSTEP.
runstep() {
  label="$1"; shift
  head2 "$label"
  if ! "$@"; then
    warn "Step failed: $label"
    return 1
  fi
  return 0
}

# A path typed or dragged into Terminal. Finder escapes spaces with
# backslashes; quotes arrive when the path was pasted from elsewhere.
clean_path() {
  p="$1"
  p="${p#\"}"; p="${p%\"}"
  p="${p#\'}"; p="${p%\'}"
  p="$(printf '%s' "$p" | sed 's/\\\(.\)/\1/g')"
  case "$p" in
    "~") p="$HOME" ;;
    "~/"*) p="$HOME/${p#\~/}" ;;
  esac
  printf '%s' "$p"
}

require_py() {
  if [ -z "$PY" ]; then
    warn "${B}No working Python 3 found.${R}"
    # The remedy is platform-specific, and this script runs on Linux too - where
    # every line of the macOS version is wrong.
    case "$(uname -s)" in
      Darwin)
        say "  macOS does not ship a usable Python 3 for this project: its"
        say "  /usr/bin/python3 is an Xcode stub, not an interpreter."
        brew_bootstrap_help
        ;;
      *)
        say "  Install Python 3.10 or newer, then run setup:"
        say "      sudo apt install python3 python3-venv    # Debian, Ubuntu"
        say "      sudo dnf install python3                 # Fedora, RHEL"
        ;;
    esac
    say  "  Then: ./run.sh setup"
    say  ""
    return 1
  fi
  return 0
}

is_video() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    *.mp4|*.mov|*.avi|*.mkv|*.webm|*.m4v) return 0 ;;
    *) return 1 ;;
  esac
}

is_url() {
  case "$1" in
    http://*|https://*) return 0 ;;
    *) return 1 ;;
  esac
}

# Echo a usable path for a file argument, or fail.
#
# Tries it as given (absolute, or relative to the project root this script cd'd
# to), then relative to where the user actually was. Without the second attempt a
# relative path typed from another directory is reported missing while sitting in
# plain sight.
resolve_input() {
  if [ -f "$1" ]; then
    printf '%s' "$1"
    return 0
  fi
  case "$1" in
    /*) return 1 ;;
  esac
  if [ -n "$ORIG_PWD" ] && [ -f "$ORIG_PWD/$1" ]; then
    printf '%s' "$ORIG_PWD/$1"
    return 0
  fi
  return 1
}

# ===========================================================================
#  Commands
# ===========================================================================

cmd_info() {
  require_py || return 1
  head2 "System status"
  "$PY" -m runway_ai info
  say ""
  say "  Interpreter: $PY  (Python $PY_VERSION)"
  if main_has_mediapipe; then
    say "  Video pipeline: in this environment - no delegation needed."
  elif [ -n "$PYVID" ]; then
    say "  Video pipeline: $PYVID  (Python $PYVID_VERSION)"
  else
    say "  Video pipeline: not available."
    video_env_help
  fi
  stranded_warning
  say ""
  say "  For a full capability report with a fix for each gap:"
  say "      ./run.sh diagnose"
}

cmd_diagnose() {
  require_py || return 1
  head2 "Diagnostics"
  # The same report the Guide page shows, so somebody without the web app
  # running can still produce something worth pasting into a bug report.
  "$PY" -c "import sys; sys.path.insert(0,'app'); import diagnostics; print(diagnostics.as_text())"
}

cmd_test() {
  require_py || return 1
  head2 "Test suite"
  "$PY" -m pytest tests -q
  pyfail=$?

  # The live scene classifier is browser-side JavaScript, so pytest cannot reach
  # it. It runs on node's own test runner with no dependencies - but only if the
  # web app's node_modules exist, since a fresh checkout has not run npm install
  # and a missing toolchain must not read as a test failure.
  jsfail=0
  if [ -d "web/node_modules" ]; then
    head2 "Live scene classifier - node --test"
    ( cd web && npm test --silent )
    jsfail=$?
  else
    warn "Skipping the browser tests - run ./run.sh web once to install them."
  fi

  if [ "$pyfail" -eq 0 ] && [ "$jsfail" -eq 0 ]; then
    warn "All tests passed."
    return 0
  fi
  warn "${B}TESTS FAILED.${R}"
  return 1
}

cmd_sample() {
  require_py || return 1
  head2 "Scoring the supplied workbook"
  if [ ! -f "Synthetic_Runway_Keypoints_Sample.xlsx" ]; then
    warn "Sample workbook not found in $ROOT"
    return 1
  fi
  "$PY" -m runway_ai analyze "Synthetic_Runway_Keypoints_Sample.xlsx"
}

cmd_results() {
  require_py || return 1
  head2 "Model accuracy (all out-of-fold)"
  if [ ! -f "reports/model_comparison.csv" ]; then
    warn "No results yet. Run ./run.sh rebuild first."
    return 1
  fi
  "$PY" scripts/show_results.py
}

cmd_sanity() {
  require_py || return 1
  head2 "Does it rank skill correctly?"
  "$PY" scripts/sanity_check.py
}

cmd_verify() {
  require_py || return 1
  runstep "System status" "$PY" -m runway_ai info || return 1
  runstep "Test suite"    "$PY" -m pytest tests -q || return 1
  runstep "Sanity check"  "$PY" scripts/sanity_check.py || return 1
  head2 "Verification complete"
}

# Score one file. Video goes through pose extraction first; a keypoint table is
# scored directly.
cmd_analyze() {
  target="$1"; shift
  require_py || return 1
  if [ -z "$target" ]; then
    warn "Usage: ./run.sh analyze <file|url> [--user NAME] [--save-report] [--json]"
    return 1
  fi

  # A share link is a documented input - README advertises it - and the pipeline
  # fetches it server-side. It goes to the main interpreter, which is the one with
  # yt-dlp; requirements-video.txt does not include it, so a URL must never be
  # routed to the video environment.
  if is_url "$target"; then
    head2 "Analysing link: $target"
    "$PY" -m runway_ai analyze "$target" "$@"
    return $?
  fi

  if ! resolved_target="$(resolve_input "$target")"; then
    warn "File not found: $target"
    return 1
  fi
  target="$resolved_target"

  if ! is_video "$target"; then
    head2 "Analysing: $target"
    "$PY" -m runway_ai analyze "$target" "$@"
    return $?
  fi

  # A video in an environment that can read video: `analyze` ingests it directly,
  # so one step and no round trip. Unlike the delegated path below, this leaves no
  # keypoint CSV behind - it never writes one.
  if main_has_mediapipe; then
    head2 "Analysing: $target"
    "$PY" -m runway_ai analyze "$target" "$@"
    return $?
  fi

  if [ -z "$PYVID" ]; then
    warn "\"$target\" is a video, which needs MediaPipe."
    video_env_help
    stranded_warning
    return 1
  fi

  # Step 1 in the video environment, step 2 in the main one. Splitting them
  # means the trained models are never unpickled against a different
  # scikit-learn build. Keypoints land in data/interim so they can be
  # inspected, kept, or re-scored without repeating the slow extraction.
  mkdir -p data/interim
  kpfile="$ROOT/data/interim/_last_video.csv"

  head2 "Step 1/2: extracting pose from $target"
  if ! "$PYVID" -m runway_ai extract "$target" -o "$kpfile"; then
    warn "Pose extraction failed. Common causes: the whole body is not in"
    say  "  frame, the clip is too short, or the subject is too dark."
    return 1
  fi

  # The frame rate travels in a .meta.json sidecar written by `extract`, so
  # there is nothing to parse out of stdout here.
  head2 "Step 2/2: scoring"
  "$PY" -m runway_ai analyze "$kpfile" "$@"
}

cmd_group() {
  target="$1"
  require_py || return 1
  if [ -z "$target" ]; then
    warn "Usage: ./run.sh group <video>"
    return 1
  fi
  if ! resolved_target="$(resolve_input "$target")"; then
    warn "File not found: $target"
    return 1
  fi
  target="$resolved_target"

  runner="$PYVID"
  main_has_mediapipe && runner="$PY"
  if [ -z "$runner" ]; then
    warn "Group analysis reads video, which needs MediaPipe."
    video_env_help
    stranded_warning
    return 1
  fi

  say ""
  say "  Finding everyone with a visible face, then measuring each separately."
  say "  Pose estimation runs once per person, so this takes longer than a solo"
  say "  clip. The web app (./run.sh web) also scores and ranks them."
  say ""
  "$runner" -m runway_ai extract-multi "$target"
}

cmd_render() {
  target="$1"
  require_py || return 1
  if [ -z "$target" ]; then
    warn "Usage: ./run.sh render <video>"
    return 1
  fi
  if ! resolved_target="$(resolve_input "$target")"; then
    warn "File not found: $target"
    return 1
  fi
  target="$resolved_target"

  # `render` consumes a keypoint CSV and nothing else - it has no default path and
  # errors out without -k - so both branches extract first. The only difference is
  # which interpreter does the extracting. Copying cmd_analyze's one-step shortcut
  # here was wrong: `analyze` ingests video itself, `render` does not.
  extractor="$PYVID"
  main_has_mediapipe && extractor="$PY"
  if [ -z "$extractor" ]; then
    warn "Rendering needs pose data, which needs MediaPipe."
    video_env_help
    stranded_warning
    return 1
  fi

  mkdir -p data/interim
  kpfile="$ROOT/data/interim/_last_video.csv"
  head2 "Step 1/2: extracting pose"
  "$extractor" -m runway_ai extract "$target" -o "$kpfile" || return 1
  head2 "Step 2/2: rendering annotated video"
  "$PY" -m runway_ai render "$target" -k "$kpfile"
}

cmd_api() {
  require_py || return 1
  "$PY" scripts/check_env.py api || return 1
  say ""
  say "  Starting FastAPI. Interactive docs:  http://127.0.0.1:8000/docs"
  say "  Press Ctrl+C to stop."
  say ""
  "$PY" -m uvicorn app.api:app --reload
}

# Node's floor is Vite 8's own: "^20.19.0 || >=22.12.0". Comparing the major
# version alone is not the same rule and lets three ranges through that Vite
# rejects - 20.0-20.18, all of 21.x, and 22.0-22.11 - so the test is written as
# the engines field actually reads. Asking node to evaluate it keeps the
# comparison out of shell arithmetic, where minor versions are easy to fumble.
NODE_REQUIREMENT="^20.19.0 || >=22.12.0"

check_node() {
  if ! command -v node >/dev/null 2>&1; then
    warn "Node.js is required for the web app."
    say  "      brew install node          # or download from nodejs.org"
    say  "  Then run this again."
    return 1
  fi
  ok="$(node -p '
    const [a, b] = process.versions.node.split(".").map(Number);
    ((a === 20 && b >= 19) || a > 22 || (a === 22 && b >= 12)) ? "ok" : "old"
  ' 2>/dev/null)"
  # An unreadable answer is not a failure: let node speak for itself rather than
  # blocking the app on a version string this script could not parse.
  [ "$ok" = "old" ] || return 0
  warn "Node $(node -v) is not supported by Vite 8, which needs $NODE_REQUIREMENT."
  say  "      brew install node          # or nodejs.org"
  return 1
}

# Whether web/node_modules can actually run *here*.
#
# Directory existence is not the question, and assuming it was is a trap with
# teeth: a folder copied from Windows brings a populated node_modules whose only
# native bindings are @rolldown/binding-win32-x64-msvc and the oxlint equivalent.
# The platform-specific packages are optional dependencies, so npm never fetched
# the macOS ones. `npm test` then passes - node --test needs no binding - while
# `npm run dev` dies with "Cannot find module @rolldown/binding-darwin-arm64".
# Green tests and an app that cannot start is the worst possible signal, so the
# actual binding is probed instead.
# Matched with a trailing glob, not an exact name: the binding packages carry an
# ABI suffix that platform+arch alone does not predict - win32-x64-msvc,
# linux-x64-gnu, linux-x64-musl, but plain darwin-arm64. Requiring
# "@rolldown/binding-$platform-$arch" therefore fails on every platform except
# macOS, which would send a correctly installed Windows or Linux tree through a
# pointless reinstall on every run.
web_deps_ready() {
  [ -d "web/node_modules" ] || return 1
  pa="$(node -p 'process.platform + "-" + process.arch' 2>/dev/null)" || return 1
  [ -n "$pa" ] || return 1
  for dir in "web/node_modules/@rolldown/binding-$pa"*; do
    [ -d "$dir" ] && return 0
  done
  return 1
}

install_web_deps() {
  if [ -d "web/node_modules" ]; then
    head2 "Reinstalling web dependencies for $(uname -s) $(uname -m)"
    say "  The existing web/node_modules has no native binding for this platform,"
    say "  which happens when the folder was copied from another machine."
    say ""
  else
    head2 "Installing web dependencies (first run only)"
  fi
  # `npm ci` when there is a lockfile: it wipes node_modules first, which is
  # exactly what a tree carrying another platform's bindings needs.
  if [ -f "web/package-lock.json" ]; then
    ( cd web && npm ci ) || return 1
  else
    ( cd web && npm install ) || return 1
  fi
  return 0
}

cmd_web() {
  require_py || return 1
  "$PY" scripts/check_env.py api || return 1
  check_node || return 1
  web_deps_ready || install_web_deps || return 1

  # A previous run that was killed rather than stopped can leave the port bound,
  # and uvicorn's own error for that is a stack trace.
  if command -v lsof >/dev/null 2>&1 && lsof -ti tcp:8000 >/dev/null 2>&1; then
    warn "Port 8000 is already in use."
    say  "  Something is already serving the API - probably a previous run."
    say  "      lsof -ti tcp:8000 | xargs kill"
    return 1
  fi

  say ""
  say "  Starting both servers."
  say "    API  http://127.0.0.1:8000"
  say "    App  http://localhost:5180"
  say ""
  say "  Ctrl+C stops both."
  say ""

  # Backgrounded rather than opened in a second window (run.bat's approach,
  # which has no macOS equivalent worth having): one Ctrl+C, one trap, both
  # processes gone. Without the trap the API survives the dev server and the
  # next run finds port 8000 taken.
  "$PY" -m uvicorn app.api:app --host 127.0.0.1 --port 8000 &
  api_pid=$!
  trap 'kill "$api_pid" 2>/dev/null; wait "$api_pid" 2>/dev/null; trap - EXIT INT TERM' EXIT INT TERM
  ( cd web && npm run dev )
  # The dev server's status is the command's status. Ending on `trap -`, which
  # always succeeds, made `./run.sh web` exit 0 even when vite failed to start -
  # and this is the one command people wrap in scripts. The `wait` matters too:
  # without it the script can exit while uvicorn is still holding port 8000, and
  # the next run refuses to start.
  rc=$?
  kill "$api_pid" 2>/dev/null
  wait "$api_pid" 2>/dev/null
  trap - EXIT INT TERM
  return $rc
}

cmd_train() {
  require_py || return 1
  runstep "Calibrating rubric" "$PY" scripts/calibrate_rubric.py || return 1
  runstep "Training models"    "$PY" scripts/train_model.py || return 1
}

cmd_rebuild() {
  require_py || return 1
  say ""
  say "  Regenerates the corpus, refits the rubric and retrains."
  say "  Takes about 5 minutes. Existing models will be overwritten."
  say ""
  printf '  Continue? [y/N]: '
  read -r ok
  case "$ok" in [yY]*) ;; *) return 0 ;; esac

  runstep "Generating dataset" "$PY" scripts/build_dataset.py --n 1500 --seed 42 --no-keypoints || return 1
  runstep "Calibrating rubric" "$PY" scripts/calibrate_rubric.py || return 1
  runstep "Training models"    "$PY" scripts/train_model.py || return 1
  runstep "Test suite"         "$PY" -m pytest tests -q || return 1
  head2 "Rebuild complete"
  say "  Seeds are fixed, so this should reproduce mean MAE 0.902 / R2 0.735."
}

cmd_sensitivity() {
  require_py || return 1
  say ""
  say "  Sweeps each latent quality and reports which features respond."
  say "  This is the check that a feature measures what it claims to."
  say "  Takes about 10 minutes; output also written to reports/sensitivity.txt"
  say ""
  "$PY" scripts/sensitivity.py
}

# --- setup: the fresh-Mac path -------------------------------------------- #

# Newest interpreter that MediaPipe still supports, preferred first. Homebrew
# keeps versioned binaries on PATH as python3.12 etc; the python.org installer
# puts them in /Library/Frameworks and also links them into /usr/local/bin.
find_mediapipe_python() {
  for exe in python3.12 python3.11 python3.10 python3.9 \
             /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3.11 \
             /usr/local/bin/python3.12 /usr/local/bin/python3.11 \
             /Library/Frameworks/Python.framework/Versions/3.12/bin/python3 \
             /Library/Frameworks/Python.framework/Versions/3.11/bin/python3; do
    if command -v "$exe" >/dev/null 2>&1; then
      resolved="$(command -v "$exe")"
      # Validated, not merely present - and the version is re-checked rather
      # than trusted from the filename, because `python3.12` on PATH can be a
      # shim, an alias, or a pyenv stub for something else entirely.
      if found="$(usable_python "$resolved")"; then
        case "$found" in
          3.9|3.10|3.11|3.12) printf '%s' "$resolved"; return 0 ;;
        esac
      fi
    fi
  done
  return 1
}

version_at_least() {                # version_at_least 3.12 3.10  -> true
  awk -v a="$1" -v b="$2" 'BEGIN {
    split(a, x, "."); split(b, y, ".");
    exit !((x[1] > y[1]) || (x[1] == y[1] && x[2] >= y[2]))
  }'
}

cmd_setup() {
  head2 "Setup"

  case "$(uname -s)" in
    Darwin)
      if ! xcode-select -p >/dev/null 2>&1; then
        warn "${B}Xcode Command Line Tools are missing.${R}"
        say  "  Several wheels need them. Install, then run setup again:"
        say  "      xcode-select --install"
        return 1
      fi
      say "  Xcode Command Line Tools: present"
      if command -v brew >/dev/null 2>&1; then
        say "  Homebrew: $(brew --prefix)"
      else
        say "  Homebrew: not installed (optional, but the easiest way to get"
        say "            Python 3.12 and Node)"
      fi
      ;;
  esac

  find_interpreters
  stranded_warning

  # Offered rather than done: deleting a directory the user did not ask about is
  # not the launcher's decision to make, even when it is certainly dead weight.
  # A stranded environment is inert once find_interpreters ignores it, so "no" is
  # a perfectly good answer - it just leaves a few hundred MB behind.
  if [ -n "$STRANDED" ]; then
    printf '  Delete the Windows environment(s) now?%s [y/N]: ' "$STRANDED"
    read -r drop
    case "$drop" in
      [yY]*)
        for name in $STRANDED; do
          rm -rf "$name" && say "  removed $name"
        done
        find_interpreters
        ;;
      *) say "  Left in place." ;;
    esac
  fi

  # Two separate questions, and answering both with one probe was a bug.
  #
  #   (a) Which interpreter should own .venv?  Needs >= 3.10.
  #   (b) Which interpreter can run MediaPipe? Needs 3.9-3.12.
  #
  # `find_mediapipe_python` answers (b) and accepts 3.9 - a fine video interpreter
  # and an unusable base. Using its result for (a) meant a machine whose only
  # MediaPipe-capable Python was 3.9 aborted setup on the 3.10 floor while a
  # perfectly good 3.13 sat on the PATH. And because `single=yes` was set exactly
  # when that probe succeeded, the two-environment branch below re-probed for
  # something already known to be absent: it could never run at all.
  mp_python="$(find_mediapipe_python || true)"
  mp_version=""
  [ -n "$mp_python" ] && mp_version="$(usable_python "$mp_python")"

  base=""
  base_version=""
  # An interpreter that satisfies both gets to own the single environment. That
  # overlap is the whole macOS simplification, so it is worth preferring even when
  # something newer is available.
  if [ -n "$mp_python" ] && version_at_least "$mp_version" 3.10; then
    base="$mp_python"
    base_version="$mp_version"
  else
    for candidate in python3 python; do
      if command -v "$candidate" >/dev/null 2>&1; then
        resolved="$(command -v "$candidate")"
        if base_version="$(usable_python "$resolved")"; then
          base="$resolved"
          break
        fi
      fi
    done
  fi

  if [ -z "$base" ]; then
    warn "${B}No working Python 3 found.${R}"
    say  "  On macOS: brew install python@3.12   (see ./run.sh info for the"
    say  "  full first-time sequence, Homebrew included)"
    say  "  Or download it from python.org, then run setup again."
    return 1
  fi
  if ! version_at_least "$base_version" 3.10; then
    warn "${B}$base is Python $base_version; the pipeline needs 3.10 or newer.${R}"
    say  "  macOS ships an old python3 for Xcode's own use. Install a real one:"
    say  "      brew install python@3.12"
    say  "  then run setup again."
    return 1
  fi

  say ""
  say "  Base interpreter: $base  (Python $base_version)"

  if [ ! -d ".venv" ]; then
    runstep "Creating .venv" "$base" -m venv .venv || return 1
  else
    say "  .venv exists - reusing it."
  fi

  venv_py="$ROOT/.venv/bin/python"
  [ -x "$venv_py" ] || venv_py="$ROOT/.venv/bin/python3"
  if [ ! -x "$venv_py" ]; then
    warn "${B}.venv has no bin/python.${R}"
    say  "  If this folder came from Windows, its .venv is a Windows one:"
    say  "      rm -rf .venv && ./run.sh setup"
    return 1
  fi

  # `single` is decided by the interpreter that *owns* .venv, not by the one that
  # would have been chosen. A .venv carried over from another machine, or built
  # earlier on a different Python, is exactly the case where those two differ -
  # and getting it from the probe meant installing MediaPipe into an environment
  # that could not use it, or skipping the second environment that was needed.
  venv_version="$(usable_python "$venv_py")"
  single="no"
  case "$venv_version" in
    3.9|3.10|3.11|3.12) single="yes" ;;
  esac
  say "  .venv runs Python ${venv_version:-unknown}"
  if [ "$single" = "yes" ]; then
    say "  MediaPipe supports that, so one environment covers everything -"
    say "  video included, with no delegation."
  else
    say "  MediaPipe does not support that, so video ingestion needs a second"
    say "  environment on 3.9-3.12."
  fi

  runstep "Upgrading pip" "$venv_py" -m pip install --upgrade pip --quiet || return 1
  runstep "Installing requirements.txt" "$venv_py" -m pip install -r requirements.txt || return 1

  if [ "$single" = "yes" ]; then
    runstep "Installing requirements-video.txt (same environment)" \
      "$venv_py" -m pip install -r requirements-video.txt || {
        warn "Video packages failed to install."
        say  "  Everything else works: the pipeline consumes keypoints, so CSV"
        say  "  and XLSX input, scoring, the API and the web app are unaffected."
      }
  elif [ -n "$mp_python" ]; then
    # Reuses the interpreter already found rather than probing again.
    if [ -d ".venv312" ] && [ ! -x ".venv312/bin/python" ] \
       && [ ! -x ".venv312/bin/python3" ]; then
      warn "${B}.venv312 exists but has no bin/python.${R}"
      say  "  It was built on another platform. Remove it and run setup again:"
      say  "      rm -rf .venv312"
      return 1
    fi
    if [ ! -d ".venv312" ]; then
      runstep "Creating .venv312 (Python $mp_version)" \
        "$mp_python" -m venv .venv312 || return 1
    fi
    video_py="$ROOT/.venv312/bin/python"
    [ -x "$video_py" ] || video_py="$ROOT/.venv312/bin/python3"
    runstep "Upgrading pip in .venv312" \
      "$video_py" -m pip install --upgrade pip --quiet || return 1
    runstep "Installing requirements-video.txt" \
      "$video_py" -m pip install -r requirements-video.txt || return 1
  else
    say ""
    say "  No Python 3.9-3.12 found, so video input stays unavailable."
    video_env_help
  fi

  # Rendering an annotated video runs in *this* environment and needs OpenCV,
  # which requirements.txt leaves out on purpose. In the single-environment case
  # requirements-video.txt has already supplied it.
  if [ "$single" != "yes" ]; then
    if ! "$venv_py" -c "import cv2" >/dev/null 2>&1; then
      runstep "Installing OpenCV for the annotated-video renderer" \
        "$venv_py" -m pip install opencv-python-headless || {
          say "  Rendering will be unavailable; nothing else is affected."
        }
    fi
  fi

  if command -v node >/dev/null 2>&1; then
    if check_node && ! web_deps_ready; then
      install_web_deps || return 1
    elif web_deps_ready; then
      say ""
      say "  web/node_modules is already correct for this platform."
    fi
  else
    say ""
    say "  Node.js not found. The API and the CLI work without it; the React"
    say "  app needs it:  brew install node"
  fi

  find_interpreters
  head2 "Setup complete"
  say "  Analysis:  ${PY:-none}"
  if main_has_mediapipe; then
    say "  Video:     same environment"
  else
    say "  Video:     ${PYVID:-not available}"
  fi

  # The twelve .joblib bundles were pickled by whichever scikit-learn built them.
  # A different one here cannot unpickle them, and the symptom is a silent
  # fallback to the rubric - a working app with quietly worse scores. Worth one
  # check now rather than a puzzle later.
  #
  # The three outcomes are distinguished rather than lumped together as "pickled
  # by a different scikit-learn": models can also be simply absent (a fresh
  # checkout, or after `clean`), and a genuine load failure has a message worth
  # reading rather than a guess worth printing.
  say ""
  model_status="$("$PY" - <<'PY' 2>&1
import glob
import os
import sys

files = glob.glob(os.path.join(os.getcwd(), "models", "model_*.joblib"))
if not files:
    print("MISSING no models/model_*.joblib present")
    sys.exit(0)
try:
    from runway_ai.models.train import load_models
    models = load_models() or {}
except Exception as exc:
    print(f"ERROR {type(exc).__name__}: {exc}")
    sys.exit(0)
print(f"OK {len(models)}" if models else "ERROR load_models returned nothing")
PY
)"
  case "$model_status" in
    OK*)
      say "  Trained models load correctly (${model_status#OK })/12."
      ;;
    MISSING*)
      say "  No trained models present - scoring will use the transparent rubric."
      say "      ./run.sh rebuild            (about 5 minutes, seeds are fixed)"
      ;;
    *)
      say "  ${B}The trained models did not load.${R} Scoring would fall back to the"
      say "  rubric, which is blunter. Reported:"
      say "      ${model_status#ERROR }"
      say "  Usually a scikit-learn built elsewhere. Retrain here:"
      say "      ./run.sh rebuild            (about 5 minutes, seeds are fixed)"
      ;;
  esac

  say ""
  say "  Next:  ./run.sh verify        proves the install"
  say "         ./run.sh web           the app on http://localhost:5180"
  say ""
  say "  Optional: group analysis and face sign-in need a one-off 39 MB download:"
  say "      $venv_py scripts/fetch_face_models.py"
}

cmd_clean() {
  say ""
  say "  Deletes generated data, trained models and reports."
  say "  Source code, docs and the supplied workbook are NOT touched."
  say ""
  printf '  Delete generated files? [y/N]: '
  read -r ok
  case "$ok" in [yY]*) ;; *) return 0 ;; esac

  rm -rf data/interim data/processed models reports .pytest_cache
  rm -f data/runway.db
  # Pruned rather than descended into: .venv and node_modules hold tens of
  # thousands of files whose caches are not ours to clear.
  find . \( -name .venv -o -name '.venv3*' -o -name node_modules -o -name .git \) -prune \
       -o -type d -name __pycache__ -print0 2>/dev/null | xargs -0 rm -rf 2>/dev/null

  warn "Cleaned. Run ./run.sh rebuild to regenerate."
}

cmd_help() {
  cat <<'EOF'

  Usage: ./run.sh [command]

    info           system status: models, rubric, video backend
    diagnose       full capability report, with a fix for each gap
    test           run the Python suite and the browser tests
    sample         score the supplied workbook
    results        print per-criterion accuracy
    sanity         check expert/intermediate/novice ranking
    verify         info + test + sanity, in one pass
    analyze FILE   score a video, csv or xlsx  (extra flags pass through)
    group FILE     find and measure everyone in one clip
    render FILE    annotated video: skeleton, live metrics, scores
    web            start the React app and the API together
    api            launch the FastAPI service
    train          recalibrate the rubric and retrain the models
    rebuild        regenerate data, recalibrate rubric, retrain
    sensitivity    feature validation sweep
    setup          build the environments (safe to re-run)
    clean          delete generated data, models and reports

  Run with no arguments for an interactive menu.

EOF
}

# ===========================================================================
#  Menu
# ===========================================================================

pause_menu() {
  say ""
  printf '  Press return to continue... '
  read -r _
}

menu() {
  while :; do
    clear 2>/dev/null
    cat <<EOF

  ===========================================================
   ${B}RUNWAY PERFORMANCE AI${R}
  ===========================================================

   CHECK
     1.  System status          (models loaded, backends)
     2.  Run test suite         (Python + browser, ~40s)
     3.  Score the sample file  (the supplied workbook)
     4.  Show model results     (per-criterion accuracy)
     5.  Sanity check           (expert vs novice ranking)

   RUN
     6.  Analyse a file...      (video, csv or xlsx)
     7.  Analyse a group...     (everyone in one clip)
     8.  Annotated video        (skeleton + live metrics)
     9.  Web app                (React + API, both servers)
    10.  REST API               (FastAPI + docs)

   BUILD
    11.  Full rebuild           (data + rubric + train, ~5 min)
    12.  Feature sensitivity    (validation sweep, ~10 min)
    13.  Set up / install       (environments and dependencies)
    14.  Clean generated files

     0.  Exit

EOF
    printf '  Select: '
    read -r choice || return 0

    case "$choice" in
      1)  cmd_info; pause_menu ;;
      2)  cmd_test; pause_menu ;;
      3)  cmd_sample; pause_menu ;;
      4)  cmd_results; pause_menu ;;
      5)  cmd_sanity; pause_menu ;;
      6)  ask_and_run analyze "video/csv/xlsx" ;;
      7)  ask_and_run group "a group video" ;;
      8)  ask_and_run render "the video" ;;
      9)  cmd_web; pause_menu ;;
      10) cmd_api; pause_menu ;;
      11) cmd_rebuild; pause_menu ;;
      12) cmd_sensitivity; pause_menu ;;
      13) cmd_setup; pause_menu ;;
      14) cmd_clean; pause_menu ;;
      0|q|Q) return 0 ;;
      "") ;;
      *)  warn "Not an option."; sleep 1 ;;
    esac
  done
}

ask_and_run() {
  action="$1"; what="$2"
  say ""
  printf '  Path to %s (or drag the file in): ' "$what"
  read -r raw
  [ -n "$raw" ] || return 0
  target="$(clean_path "$raw")"
  if ! resolved_target="$(resolve_input "$target")"; then
    warn "File not found: $target"
    pause_menu
    return 0
  fi
  target="$resolved_target"
  case "$action" in
    analyze) cmd_analyze "$target" --save-report ;;
    group)   cmd_group "$target" ;;
    render)  cmd_render "$target" ;;
  esac
  pause_menu
}

# ===========================================================================
#  Dispatch
# ===========================================================================

find_interpreters

if [ $# -eq 0 ]; then
  menu
  exit 0
fi

command_name="$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')"
shift

case "$command_name" in
  info)                 cmd_info ;;
  diagnose|diagnostics) cmd_diagnose ;;
  test|tests)           cmd_test ;;
  sample)               cmd_sample ;;
  results)              cmd_results ;;
  sanity)               cmd_sanity ;;
  verify)               cmd_verify ;;
  analyze|analyse)      cmd_analyze "$@" ;;
  group)                cmd_group "$@" ;;
  render)               cmd_render "$@" ;;
  web)                  cmd_web ;;
  api)                  cmd_api ;;
  train)                cmd_train ;;
  rebuild)              cmd_rebuild ;;
  sensitivity)          cmd_sensitivity ;;
  setup)                cmd_setup ;;
  clean)                cmd_clean ;;
  help|-h|--help)       cmd_help ;;
  *)
    warn "Unknown command: $command_name"
    cmd_help
    exit 1
    ;;
esac
