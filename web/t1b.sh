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
HOME=/Users/me
printf 'A escaped-space   [%s]\n' "$(clean_path '/Users/me/My\ Walks/clip.mov')"
printf 'B dquoted         [%s]\n' "$(clean_path '"/Users/me/My Walks/clip.mov"')"
printf 'C squoted         [%s]\n' "$(clean_path "'/Users/me/My Walks/clip.mov'")"
printf 'D tilde slash     [%s]\n' "$(clean_path '~/Movies/clip.mov')"
printf 'E tilde           [%s]\n' "$(clean_path '~')"
printf 'F parens+apost    [%s]\n' "$(clean_path '/Users/me/O'"'"'Brien\ Walks/a\(1\).mov')"
printf 'G icloud tildes   [%s]\n' "$(clean_path '/Users/me/Library/Mobile\ Documents/com~apple~CloudDocs/clip.mov')"
printf 'H tilde-escaped   [%s]\n' "$(clean_path '\~/Movies/clip.mov')"
printf 'I onedrive        [%s]\n' "$(clean_path '/Users/me/Library/CloudStorage/OneDrive-Personal/Desktop/albins/data/My\ Walks/c.mov')"
printf 'J backslash-name  [%s]\n' "$(clean_path '/Users/me/a\b.mov')"
printf 'K dquote-and-esc  [%s]\n' "$(clean_path '"/Users/me/My\ Walks/clip.mov"')"
printf 'L trailing-esc-sp [%s]\n' "$(clean_path '/Users/me/x.mov\ ')"
