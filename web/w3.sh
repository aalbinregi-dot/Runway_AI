sleep 20 &
api_pid=$!
trap 'echo "[EXIT/INT/TERM trap fired]"; kill "$api_pid" 2>/dev/null; wait "$api_pid" 2>/dev/null; trap - EXIT INT TERM' EXIT INT TERM
( exit 7 )     # stand-in for npm run dev failing
echo "fg rc=$?"
kill "$api_pid" 2>/dev/null; echo "post kill rc=$?"
trap - EXIT INT TERM
echo "cmd_web returns $?  <-- npm failure status is DISCARDED"
