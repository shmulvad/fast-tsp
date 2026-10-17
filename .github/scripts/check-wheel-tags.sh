#!/usr/bin/env bash
# Fails unless the directory holds exactly the three expected wheels:
# abi3 (3.10+), free-threaded 3.14, and abi3t (3.15+, GIL and free-threaded).
set -euo pipefail
shopt -s nullglob

dir="${1:?usage: check-wheel-tags.sh DIR}"
status=0
for pattern in '*-cp310-abi3-*.whl' '*-cp314-cp314t-*.whl' '*-cp315-abi3.abi3t-*.whl'; do
    # shellcheck disable=SC2206  # the pattern is meant to glob
    matches=("$dir"/$pattern)
    if [ "${#matches[@]}" -ne 1 ]; then
        echo "Expected exactly one wheel matching $pattern, found ${#matches[@]}"
        status=1
    fi
done
all=("$dir"/*.whl)
if [ "${#all[@]}" -ne 3 ]; then
    echo "Expected 3 wheels, found ${#all[@]}"
    status=1
fi
ls -1 "$dir"
exit "$status"
