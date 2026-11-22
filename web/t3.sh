version_at_least() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    split(a, x, "."); split(b, y, ".");
    exit !((x[1] > y[1]) || (x[1] == y[1] && x[2] >= y[2]))
  }'
}
for pair in "3.12 3.10" "3.10 3.10" "3.9 3.10" "3.13 3.10" "3.8 3.10" "4.0 3.10" "3.100 3.10" " 3.10"; do
  set -- $pair
  version_at_least "$1" "$2" && echo "$1 >= $2 : YES" || echo "$1 >= $2 : no"
done
