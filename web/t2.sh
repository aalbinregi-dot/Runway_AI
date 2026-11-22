web_deps_ready() {
  [ -d "web/node_modules" ] || return 1
  pa="$1"
  [ -n "$pa" ] || return 1
  for dir in "web/node_modules/@rolldown/binding-$pa"*; do
    [ -d "$dir" ] && return 0
  done
  return 1
}
cd g
mkdir -p web/node_modules/@rolldown/binding-win32-x64-msvc
web_deps_ready darwin-arm64; echo "darwin-arm64 (only win32 present) -> $?"
web_deps_ready win32-x64;    echo "win32-x64 -> $?"
mkdir -p web/node_modules/@rolldown/binding-darwin-arm64
web_deps_ready darwin-arm64; echo "after adding darwin -> $?"
echo "--- glob literal check:"
for dir in "web/node_modules/@rolldown/binding-nope"*; do echo "dir=[$dir]"; done
