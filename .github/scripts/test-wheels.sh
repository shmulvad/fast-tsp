#!/usr/bin/env bash
# Installs the built wheel (never building from source) into a fresh venv
# for each given uv Python request and runs the test suite.
#   usage: test-wheels.sh DIST_DIR PYTHON...
set -euo pipefail

dist="${1:?usage: test-wheels.sh DIST_DIR PYTHON...}"
shift
for py in "$@"; do
    echo "::group::Testing on $py"
    rm -rf .venv-test
    uv venv -p "$py" .venv-test
    if [ -x .venv-test/Scripts/python.exe ]; then
        python=.venv-test/Scripts/python.exe
    else
        python=.venv-test/bin/python
    fi
    uv pip install -p "$python" --no-index --find-links "$dist" --only-binary fast_tsp --no-deps fast_tsp
    uv pip install -p "$python" numpy pytest
    "$python" -c 'import sys, fast_tsp; print(sys.version, fast_tsp._core.__file__)'
    "$python" -m pytest -q tests
    echo "::endgroup::"
done
