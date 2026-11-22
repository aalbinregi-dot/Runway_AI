#!/usr/bin/env bash
# Double-clickable entry point for macOS Finder.
#
# run.bat is double-clicked on Windows and opens a menu; this is the same thing
# here. Finder runs a .command file in Terminal, so all it has to do is move to
# its own directory - Finder starts it in $HOME - and hand over to run.sh.
#
# If double-clicking does nothing, the executable bit did not survive the copy
# (OneDrive, iCloud and zip archives all drop it). In Terminal, once:
#
#     cd "path/to/albins" && chmod +x run.sh run.command
#
# macOS may also refuse to open a file downloaded from the internet. Right-click
# it and choose Open, or clear the quarantine flag:
#
#     xattr -d com.apple.quarantine run.command run.sh
cd "$(dirname "$0")" || exit 1
exec ./run.sh "$@"
