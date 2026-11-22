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
