echo "start pid=$$"
sleep 30 &
api_pid=$!
echo "api_pid=$api_pid"
trap 'echo "[trap fired]"; kill "$api_pid" 2>/dev/null; echo "kill rc=$?"; wait "$api_pid" 2>/dev/null; echo "wait rc=$?"; trap - EXIT INT TERM' EXIT INT TERM
( sleep 30 )
echo "npm returned rc=$?"
kill "$api_pid" 2>/dev/null; echo "post kill rc=$?"
trap - EXIT INT TERM
echo "cmd_web would return $?"
