import time
import pytest
import fast_tsp


def test_version():
    assert fast_tsp.__version__ == "0.0.1"


def test_errors_with_wrong_signature():
    with pytest.raises(TypeError):
        fast_tsp.find_tour([0])


def test_errors_with_too_small_dist():
    with pytest.raises(ValueError, match=r'Need at least 2 nodes'):
        fast_tsp.find_tour([[0]])


def test_errors_with_non_square_matrix():
    with pytest.raises(ValueError) as exc_info:
        fast_tsp.find_tour([[0, 1], [2, 0], [3, 0]])

    exp_msg = f'Distance matrix must be square but got (3, 2)'
    assert str(exc_info.value) == exp_msg


def test_errors_with_negative_duration():
    with pytest.raises(ValueError, match='Duration must be strictly positive'):
        fast_tsp.find_tour([[0, 1], [1, 0]], duration_seconds=-1)


def test_basic_tour():
    tour = fast_tsp.find_tour([[0, 1, 2], [1, 0, 3], [2, 3, 0]])
    assert tour == [0, 1, 2]


@pytest.mark.parametrize('duration_seconds', [0.5, 0.6, 0.7])
def test_running_duration(duration_seconds):
    n = 50
    dists = [[1] * n] * n
    start_time = time.perf_counter()
    tour = fast_tsp.find_tour(dists, duration_seconds=duration_seconds)
    duration = time.perf_counter() - start_time
    assert duration == pytest.approx(duration_seconds, abs=0.05)
    assert len(tour) == n
