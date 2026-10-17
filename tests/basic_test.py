from __future__ import annotations

import _thread
import math
import random
import sys
import threading
import time

import fast_tsp
import pytest

try:
    import numpy as np
except ImportError:  # numpy is optional
    np = None


def test_errors_with_wrong_signature():
    with pytest.raises(ValueError, match='Distance matrix must be 2D'):
        fast_tsp.find_tour([0])  # type: ignore


def test_errors_with_too_small_dist():
    with pytest.raises(ValueError, match='Need at least 2 nodes'):
        fast_tsp.find_tour([[0]])


def test_errors_with_non_square_matrix():
    exp_msg = r'Distance matrix must be square, but got \(3, 2\)'
    with pytest.raises(ValueError, match=exp_msg):
        fast_tsp.find_tour([[0, 1], [2, 0], [3, 0]])


def test_errors_with_floats():
    exp_msg = 'Distance matrix must contain integers, but found'
    with pytest.raises(ValueError, match=exp_msg):
        fast_tsp.find_tour([[0.0, 1.1], [1.1, 0.0]])  # type: ignore


def test_errors_with_negative_value():
    exp_msg = 'All distances must be non-negative, but found -5'
    with pytest.raises(ValueError, match=exp_msg):
        fast_tsp.find_tour([[0, -5], [-5, 0]])


def test_errors_with_too_large_value():
    too_large_val = fast_tsp._UINT16_MAX + 1
    err_msg = fast_tsp.is_valid_dist_matrix([[0, too_large_val], [too_large_val, 0]])
    assert err_msg == f'All distances must be <= {fast_tsp._UINT16_MAX:,}, but found {too_large_val:,}'


def test_errors_with_negative_duration():
    with pytest.raises(ValueError, match='Duration must be strictly positive'):
        fast_tsp.find_tour([[0, 1], [1, 0]], duration_seconds=-1)


def test_basic_tour():
    tour = fast_tsp.find_tour([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    assert tour == [0, 1, 2]


def _euclidean(n: int, seed: int) -> list[list[int]]:
    rng = random.Random(seed)
    pts = [(rng.uniform(0, 1000), rng.uniform(0, 1000)) for _ in range(n)]
    return [[round(math.dist(a, b)) for b in pts] for a in pts]


def test_errors_with_ragged_matrix():
    with pytest.raises(ValueError):
        fast_tsp.find_tour([[0, 1, 2], [1, 0], [2, 3, 0]])


def test_tall_input_fails_fast():
    # Used to allocate n * n up front, which could abort the interpreter.
    with pytest.raises(ValueError, match='must be square'):
        fast_tsp.find_tour([[0, 1]] * 100_000)


@pytest.mark.parametrize('n', [2, 5, 12, 20])
def test_small_problems_are_solved_exactly(n: int):
    dists = _euclidean(n, n)
    tour = fast_tsp.find_tour(dists)
    assert tour == fast_tsp.solve_tsp_exact(dists)
    assert tour[0] == 0
    greedy = fast_tsp.greedy_nearest_neighbor(dists)
    assert fast_tsp.compute_cost(tour, dists) <= fast_tsp.compute_cost(greedy, dists)


def test_exact_tour_follows_asymmetric_direction():
    # Going 0 -> 1 -> 2 -> 3 -> 0 costs 4. The reverse direction costs 400.
    dists = [[0, 1, 100, 100], [100, 0, 1, 100], [100, 100, 0, 1], [1, 100, 100, 0]]
    assert fast_tsp.solve_tsp_exact(dists) == [0, 1, 2, 3]
    assert fast_tsp.find_tour(dists) == [0, 1, 2, 3]


@pytest.mark.parametrize('max_threads', [1, 4, None])
@pytest.mark.parametrize('n', [21, 100, 500])
def test_heuristic_gives_valid_tour_within_time(n: int, max_threads: int | None):
    dists = _euclidean(n, n)
    start = time.perf_counter()
    tour = fast_tsp.find_tour(dists, duration_seconds=0.3, max_threads=max_threads)
    # Generous slack: CI runners (emulated, free-threaded, ...) can be slow.
    assert time.perf_counter() - start < 0.3 + 3.0
    assert fast_tsp.is_valid_tour(n, tour)
    greedy = fast_tsp.greedy_nearest_neighbor(dists)
    assert fast_tsp.compute_cost(tour, dists) <= fast_tsp.compute_cost(greedy, dists)


@pytest.mark.parametrize('max_threads', [1, 4])
def test_keyboard_interrupt_stops_solver(max_threads: int):
    dists = _euclidean(1000, 0)
    timer = threading.Timer(0.3, _thread.interrupt_main)
    timer.start()
    start = time.perf_counter()
    try:
        with pytest.raises(KeyboardInterrupt):
            fast_tsp.find_tour(dists, duration_seconds=60, max_threads=max_threads)
    finally:
        timer.cancel()
    assert time.perf_counter() - start < 10


@pytest.mark.parametrize('max_threads', [0, -1])
def test_errors_with_invalid_max_threads(max_threads: int):
    with pytest.raises(ValueError, match='max_threads must be a positive integer or None'):
        fast_tsp.find_tour(_euclidean(30, 0), duration_seconds=0.1, max_threads=max_threads)


def test_max_threads_is_ignored_for_exact_problems():
    dists = _euclidean(10, 0)
    assert fast_tsp.find_tour(dists, max_threads=None) == fast_tsp.solve_tsp_exact(dists)


def test_greedy_nearest_neighbor():
    dists = [[0, 5, 1, 9], [5, 0, 2, 1], [1, 2, 0, 7], [9, 1, 7, 0]]
    assert fast_tsp.greedy_nearest_neighbor(dists) == [0, 2, 1, 3]


def test_compute_cost():
    dists = [[0, 1, 2], [1, 0, 3], [2, 3, 0]]
    assert fast_tsp.compute_cost([0, 1, 2], dists) == 6
    with pytest.raises(ValueError, match='Tour contains node 3'):
        fast_tsp.compute_cost([0, 1, 3], dists)
    with pytest.raises(ValueError, match='Tour contains node -1'):
        fast_tsp.compute_cost([0, -1, 2], dists)
    with pytest.raises(ValueError, match='Tour contains node -1'):
        fast_tsp.score_tour([0, -1, 2], dists, 6)


def test_is_valid_tour():
    assert fast_tsp.is_valid_tour(3, [2, 0, 1])
    assert not fast_tsp.is_valid_tour(3, [0, 0, 1])
    assert not fast_tsp.is_valid_tour(3, [0, 1])
    assert not fast_tsp.is_valid_tour(3, [0, 1, 3])
    assert not fast_tsp.is_valid_tour(3, [0, 1, -2])


def test_score_tour():
    dists = _euclidean(15, 1)
    opt = fast_tsp.solve_tsp_exact(dists)
    opt_cost = fast_tsp.compute_cost(opt, dists)
    assert fast_tsp.score_tour(opt, dists, opt_cost) == pytest.approx(1.0)
    greedy = fast_tsp.greedy_nearest_neighbor(dists)
    assert 0.0 <= fast_tsp.score_tour(greedy, dists, opt_cost) <= 0.02 + 1e-9
    with pytest.raises(ValueError, match='Received cost < opt_cost'):
        fast_tsp.score_tour(opt, dists, opt_cost + 1)


def test_exact_rejects_too_large_n():
    dists = [[0] * 26 for _ in range(26)]
    with pytest.warns(UserWarning, match='not recommended for n > 20'):
        with pytest.raises(ValueError, match='at most 25 nodes'):
            fast_tsp.solve_tsp_exact(dists)


def test_version():
    assert isinstance(fast_tsp.__version__, str)
    assert fast_tsp.__version__.count('.') == 2


INVALID_MATRICES = [
    [0],
    [[0, 1], [2, 0], [3, 0]],
    [[0.0, 1.1], [1.1, 0.0]],
    [[0, -5], [-5, 0]],
    [[0, 70_000], [70_000, 0]],
    [[1, 1], [1, 0]],
    [[0, 1], [2, 0]],
    [[0, 1, 9], [1, 0, 1], [9, 1, 0]],
    [[0, 1, 2], [1, 0, 3], [2, 3, 0]],
]


@pytest.mark.parametrize('dists', INVALID_MATRICES)
@pytest.mark.skipif(np is None, reason='numpy is not installed')
def test_validation_without_numpy(dists, monkeypatch: pytest.MonkeyPatch):
    with_numpy = fast_tsp.is_valid_dist_matrix(dists)
    monkeypatch.setitem(sys.modules, 'numpy', None)  # makes `import numpy` fail
    without_numpy = fast_tsp.is_valid_dist_matrix(dists)
    # Messages match, except for the reported type name (numpy.float64 vs float).
    assert (with_numpy or '').split('found <')[0] == (without_numpy or '').split('found <')[0]


def test_validation_without_numpy_empty(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, 'numpy', None)
    assert fast_tsp.is_valid_dist_matrix([]) == 'Distance matrix must be non-empty'


def test_find_tour_without_numpy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setitem(sys.modules, 'numpy', None)
    assert fast_tsp.find_tour(_euclidean(8, 0)) == fast_tsp.solve_tsp_exact(_euclidean(8, 0))
    with pytest.raises(ValueError, match='All distances must be non-negative'):
        fast_tsp.find_tour([[0, -5], [-5, 0]])


@pytest.mark.skipif(np is None, reason='numpy is not installed')
class TestNumpyInput:
    @pytest.mark.parametrize('dtype', ['int64', 'int32', 'int16', 'int8', 'uint64', 'uint32', 'uint16', 'uint8'])
    def test_integer_dtypes(self, dtype: str):
        dists = _euclidean(12, 0)
        expected = fast_tsp.find_tour(dists)
        arr = np.array(dists).clip(0, 127).astype(dtype)
        assert fast_tsp.find_tour(arr) == fast_tsp.find_tour(arr.tolist())
        assert fast_tsp.find_tour(np.array(dists)) == expected

    def test_non_contiguous_array(self):
        dists = np.array(_euclidean(25, 3))
        dists[0, 1] += 1  # asymmetric, so the transpose differs
        assert fast_tsp.compute_cost(list(range(25)), dists.T) == fast_tsp.compute_cost(list(range(25)), dists.T.tolist())
        assert fast_tsp.compute_cost(list(range(25)), np.asfortranarray(dists)) == fast_tsp.compute_cost(
            list(range(25)), dists.tolist()
        )
        tour = fast_tsp.find_tour(dists[::1, ::1], duration_seconds=0.1)
        assert fast_tsp.is_valid_tour(25, tour)

    def test_invalid_arrays(self):
        with pytest.raises(ValueError, match='must contain integers'):
            fast_tsp.find_tour(np.array([[0.0, 1.0], [1.0, 0.0]]))
        with pytest.raises(ValueError, match='must be 2D'):
            fast_tsp.find_tour(np.zeros((2, 2, 2), dtype=np.int64))
        with pytest.raises(ValueError, match='must be square'):
            fast_tsp.find_tour(np.zeros((100_000, 2), dtype=np.int64))
        with pytest.raises(ValueError, match='non-negative'):
            fast_tsp.find_tour(np.array([[0, -5], [-5, 0]]))
        with pytest.raises(ValueError, match='between 0 and 65535'):
            fast_tsp.compute_cost([0, 1], np.array([[0, 70_000], [70_000, 0]]))
