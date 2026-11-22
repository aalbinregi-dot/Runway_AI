clean_path() {
  p="$1"
  p="${p#\"}"; p="${p%\"}"
  p="${p#\'}"; p="${p%\'}"
  p="$(printf '%s' "$p" | sed 's/\\(.\)/\1/g')"
  case "$p" in
    "~") p="$HOME" ;;
    "~/"*) p="$HOME/${p#\~/}" ;;
  esac
  printf '%s' "$p"
}
HOME=/Users/me
printf 'A[%s]\n' "$(clean_path '/Users/me/My\ Walks/clip.mov')"
printf 'B[%s]\n' "$(clean_path '"/Users/me/My Walks/clip.mov"')"
printf 'C[%s]\n' "$(clean_path "'/Users/me/My Walks/clip.mov'")"
printf 'D[%s]\n' "$(clean_path '~/Movies/clip.mov')"
printf 'E[%s]\n' "$(clean_path '~')"
printf 'F[%s]\n' "$(clean_path '/Users/me/O'"'"'Brien\ Walks/a\(1\).mov')"
printf 'G[%s]\n' "$(clean_path '/Volumes/My\ Book/x.mov')"
printf 'H-onequote[%s]\n' "$(clean_path '/Users/me/O'"'"'Brien/x.mov')"
printf 'I-icloud[%s]\n' "$(clean_path '/Users/me/Library/Mobile\ Documents/com~apple~CloudDocs/clip.mov')"
