#!/usr/bin/env bash
# Fails unless the directory holds exactly the expected wheels: abi3 (3.10+),
# free-threaded 3.14, and abi3t (3.15+, GIL and free-threaded).
#   usage: check-wheel-tags.sh DIR [--no-cp314t]
# Pass --no-cp314t on 32-bit targets, where PyO3 supports free-threading
# only from 3.15 (the abi3t wheel covers those users).
set -euo pipefail
shopt -s nullglob

dir="${1:?usage: check-wheel-tags.sh DIR [--no-cp314t]}"
patterns=('*-cp310-abi3-*.whl' '*-cp314-cp314t-*.whl' '*-cp315-abi3.abi3t-*.whl')
if [ "${2:-}" = "--no-cp314t" ]; then
    patterns=('*-cp310-abi3-*.whl' '*-cp315-abi3.abi3t-*.whl')
fi
status=0
for pattern in "${patterns[@]}"; do
    # shellcheck disable=SC2206  # the pattern is meant to glob
    matches=("$dir"/$pattern)
    if [ "${#matches[@]}" -ne 1 ]; then
        echo "Expected exactly one wheel matching $pattern, found ${#matches[@]}"
        status=1
    fi
done
all=("$dir"/*.whl)
if [ "${#all[@]}" -ne "${#patterns[@]}" ]; then
    echo "Expected ${#patterns[@]} wheels, found ${#all[@]}"
    status=1
fi
ls -1 "$dir"
exit "$status"
