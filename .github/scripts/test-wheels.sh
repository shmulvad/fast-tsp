#!/usr/bin/env bash
# Installs the built wheel (never building from source) into a fresh venv
# for each given uv Python request and runs the test suite.
#   usage: test-wheels.sh DIST_DIR PYTHON...
# Requests ending in "t" or naming "freethreaded" must resolve to a
# free-threaded interpreter, and all others to a GIL build. Use "3.14+gil"
# rather than "3.14", which may pick an installed 3.14t. uv will not download
# for a "+gil" request, so the GIL build is installed first.
set -euo pipefail

dist="${1:?usage: test-wheels.sh DIST_DIR PYTHON...}"
shift
for py in "$@"; do
    echo "::group::Testing on $py"
    if [[ $py == *t || $py == *freethreaded* ]]; then
        want_gil=False
    else
        want_gil=True
    fi
    if [[ $py == *+gil ]]; then
        uv python install --no-bin "${py%+gil}"
    fi
    rm -rf .venv-test
    uv venv -p "$py" .venv-test
    if [ -x .venv-test/Scripts/python.exe ]; then
        python=.venv-test/Scripts/python.exe
    else
        python=.venv-test/bin/python
    fi
    uv pip install -p "$python" --no-index --find-links "$dist" --only-binary fast_tsp --no-deps fast_tsp
    uv pip install -p "$python" numpy pytest
    "$python" -c '
import sys, fast_tsp
gil = getattr(sys, "_is_gil_enabled", lambda: True)()
print(sys.version, "GIL" if gil else "free-threaded", fast_tsp._core.__file__)
sys.exit(f"expected GIL={sys.argv[1]}, got {gil}" if str(gil) != sys.argv[1] else 0)
' "$want_gil"
    "$python" -m pytest -q tests
    echo "::endgroup::"
done
