#!/usr/bin/env bash
# Fails when a comment line (`//`, `///`, `//!`) in a Rust file is wider than 100 columns, counted
# in characters, not bytes. rustfmt leaves comments alone, so nothing else catches a comment
# edited and not rewrapped. A comment line that is only a URL, or a row of a table (`| ... |`), is
# exempt, and so is `patches/`, upstream code carried nearly unchanged. Prints the file, line and
# width of each line it refuses.
#
#   tools/comment-width.sh BASE [HEAD]   the comment lines HEAD adds or changes since it left BASE
#   tools/comment-width.sh               every comment line in the tracked files
#
# CI runs the first form on every pull request, against its base. The whole tree is not yet
# clean: older lines are rewrapped crate by crate, and then CI is to run the second form.
set -u
export LC_ALL=C.UTF-8
limit=100
status=0

# Refuses `line`, line `n` of `file`, if it is an over-wide comment line that is not exempt.
check() {
  local file=$1 n=$2 line=$3
  [ "${#line}" -le "$limit" ] && return
  local t="${line#"${line%%[![:space:]]*}"}"
  case "$t" in
    //*) ;;
    *) return ;;
  esac
  local body="${t#//}"
  body="${body#[/!]}"
  body="${body# }"
  case "$body" in
    \|*) return ;;
  esac
  # Only a URL: no space in it, or a link reference `[name]: url`.
  if [[ "$body" =~ ^\<?https?://[^[:space:]]*$ || "$body" =~ ^\[[^]]*\]:\ https?://[^[:space:]]*$ ]]; then
    return
  fi
  echo "$file:$n: ${#line} columns"
  status=1
}

if [ $# -ge 1 ]; then
  head=${2:-HEAD}
  base=$(git merge-base "$1" "$head") || exit 2
  file="" n=0
  # Only added lines, with their numbers in the new file: a hunk header gives where they start.
  while IFS= read -r line; do
    case "$line" in
      "+++ b/"*) file=${line#+++ b/} ;;
      "+++ "*) file="" ;;
      @@*)
        n=${line#*+}
        n=${n%%[, ]*}
        ;;
      +*)
        [ -n "$file" ] && check "$file" "$n" "${line#+}"
        n=$((n + 1))
        ;;
    esac
  done < <(git diff -U0 --no-color --no-ext-diff "$base" "$head" -- '*.rs' ':(exclude)patches/')
else
  while IFS= read -r -d '' f; do
    n=0
    while IFS= read -r line || [ -n "$line" ]; do
      n=$((n + 1))
      check "$f" "$n" "$line"
    done < "$f"
  done < <(git ls-files -z -- '*.rs' ':(exclude)patches/')
fi
exit $status
