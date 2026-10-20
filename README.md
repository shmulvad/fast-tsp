# fast-tsp

[![Documentation Status](https://readthedocs.org/projects/fast-tsp/badge/?version=latest)](https://fast-tsp.readthedocs.io/en/latest/?badge=latest)
[![Pip Actions Status][actions-pip-badge]][actions-pip-link]
[![PyPI - Version](https://img.shields.io/pypi/v/fast-tsp)][pypi]
[![GitHub issues](https://img.shields.io/github/issues/shmulvad/fast-tsp?style=flat-square)](https://github.com/shmulvad/fast-tsp/issues)
[![GitHub license](https://img.shields.io/github/license/shmulvad/fast-tsp?style=flat-square)][LICENSE]


A library for computing near-optimal solutions to large instances of the TSP (Travelling Salesman Problem) fast, using a local search. The core is written in Rust and provides Python bindings. It is a heuristic: on the TSPLIB instances it returns the optimal tour for most instances below 1000 nodes within 2 seconds and stays within a few tenths of a percent of the optimum on instances of several thousand nodes, but it cannot prove a tour optimal (use [Concorde](https://www.math.uwaterloo.ca/tsp/concorde.html) if you need that). See the [comparisons](https://fast-tsp.readthedocs.io/en/latest/comparisons.html) against OR-Tools, LKH-3 and other packages.

# Quickstart

First install the library

```bash
$ pip install fast-tsp
```

numpy is optional. The solver accepts plain lists of lists, and reads integer numpy arrays directly (without per-element conversion) when numpy is installed. To install it alongside: `pip install "fast-tsp[numpy]"`.


Then run the problem on your 2D distance matrix. It must be square and symmetric (`dists[i][j] == dists[j][i]`), with integer distances between 0 and 65535 and zeros on the diagonal. The heuristic used for more than 20 nodes assumes symmetric distances and will not find good tours for asymmetric ones.

```python
import fast_tsp
dists = [
    [ 0, 63, 72, 70],
    [63,  0, 57, 53],
    [72, 57,  0,  4],
    [70, 53,  4,  0],
]
tour = fast_tsp.find_tour(dists)
print(tour)  # [0, 1, 3, 2]
```

For larger problems, `find_tour(dists, max_threads=None)` searches on all but one of the available CPU cores (the default is 1 thread). The threads share the best tour found so far, so 4 threads find tours about as good as 1 thread given 4 times as long. `duration_seconds` sets the time budget, which is used in full unless you pass `stop_when_converged=True`, and `candidates` picks the neighbour lists the search works with (`"auto"`, `"nearest"` or `"alpha"`, see the docs).

### Documentation

Documentation can be found at <https://fast-tsp.readthedocs.io/>.

You can build the documentation by `cd`ing to `docs/` and running `make clean && make html`.

### Development

The solver core is written in Rust (`src/`) and the Python package lives in `python/fast_tsp/`. Prebuilt wheels are published to PyPI, so installing from PyPI needs no Rust toolchain. To build from source you need [Rust](https://rustup.rs) and [uv](https://docs.astral.sh/uv/):

```sh
$ uv venv -p 3.12
$ uv pip install -e ".[test]"   # builds the extension with maturin
$ uv run pytest tests/
$ cargo test                    # Rust unit tests
```

### Benchmarking

`examples/bench.rs` runs the solver on every instance in `cases/` and scores each tour as on [Kattis](https://open.kattis.com/problems/tsp): `0.02^((cost - opt) / (naive - opt))`, where `naive` is the greedy nearest-neighbour cost. `cases/` holds the TSPLIB instances that use Euclidean distance (`EUC_2D`) with fewer than 5000 nodes, plus three random 19-node instances. The seven larger TSPLIB instances are in `cases/large/` and run with `--cases cases/large` (about 1.6 GB of memory for the largest). A comma list for `--time` or `--threads` runs a sweep over those values:

```sh
$ cargo run --release --example bench -- --time 2
$ cargo run --release --example bench -- --time 0.1,2,10 --threads 1,4 --repeats 3 --quiet
$ cargo run --release --example bench -- --help
```

Pass `--seed <s>` to make the random choices repeatable (repeat `r` uses seed `s + r`); the deadline is still wall-clock time, so totals still move by a few tenths between runs.

Total score over the 73 instances (maximum 73) on an Apple M5 Pro, seed 1, before and after the local search was rewritten around a don't-look-bit queue, a Lin-Kernighan step, local kicks and walk restarts (the "before" column is the previous 2-opt/3-opt iterated local search):

| budget | 1 thread, before → after | 4 threads, before → after |
|--------|--------------------------|---------------------------|
| 0.2 s  | 69.22 → 71.84 | 70.01 → 72.16 |
| 2 s    | 71.06 → 72.57 | 71.35 → 72.78 |
| 10 s   | 71.36 → 72.87 | 71.38 → 72.96 |
| 30 s   | 71.45 → 72.96 | 71.57 → 72.97 |

On the two largest instances checked in `cases/large/` the scores went from 0.86–0.89 to 0.98–1.00 (rl5915) and from 0.60–0.69 to 0.93–0.98 (usa13509) at 2 s and 10 s.

0.2.2 added LKH-style alpha-nearness candidate lists with a deeper Lin-Kernighan step on top of that (`candidates="auto"`), which took the totals to 72.09 / 72.06 at 0.2 s, 72.79 / 72.96 at 2 s, 72.98 / 72.97 at 10 s and 72.99 / 73.00 at 30 s (1 / 4 threads).

### Acknowledgements

The local search re-implements published methods: Lin-Kernighan moves, don't-look bits and neighbour lists (Bentley), double-bridge kicks (Martin, Otto and Felten), and from Keld Helsgaun's [LKH](http://webhotel4.ruc.dk/~keld/research/LKH-3/) the alpha-nearness candidate lists with a subgradient ascent (Helsgaun 2000) and the sequential 3-opt basis move (Helsgaun 2009). No code from LKH is included; see the [About page](https://fast-tsp.readthedocs.io/en/latest/about.html) for the references.

### Citation

If you find that this project helps your research, please consider citing it using the metadata from the `CITATION.cff` file.

### License

This library is licensed under the MIT license, which can be found in the [LICENSE] file. By using, distributing, or contributing to this project, you agree to the terms and conditions of this license.

[LICENSE]: https://github.com/shmulvad/fast-tsp/blob/main/LICENSE
[actions-pip-link]: https://github.com/shmulvad/fast-tsp/actions?query=workflow%3APip
[actions-pip-badge]: https://github.com/shmulvad/fast-tsp/workflows/Pip/badge.svg
[pypi]: https://pypi.org/project/fast-tsp/
