"""
TSP Solver
---------------------------------

Main module for the TSP solver. This module contains the main entry point
for the TSP solver. To use it, first import it

.. code-block:: python

    import fast_tsp


.. currentmodule:: fast_tsp

.. autosummary::
    :toctree: _generate

    find_tour
    greedy_nearest_neighbor
    solve_tsp_exact
    is_valid_tour
    compute_cost
    score_tour
"""
from typing import Union, List
import warnings
import numpy as np

from ._core import (  # type: ignore
    # __doc__,
    __version__,
    find_tour as __find_tour,
    is_valid_tour as __is_valid_tour,
    compute_cost as __compute_cost,
    greedy_nearest_neighbor as __greedy_nearest_neighbor,
    solve_tsp_exact as __solve_tsp_exact,
    score_tour as __score_tour,
)

Tour = List[int]
DistMatrix = Union[List[List[int]], np.ndarray]


def find_tour(dists: DistMatrix, duration_seconds: float = 2.0) -> Tour:
    """Find a tour using the fast heuristic.

    Run a local solver to find a near-optimal TSP tour. For small
    problems, the exact solution is returned.

    Args:
        dists: A distance matrix.
        duration_seconds: The maximum duration of the tour.

    Returns:
        A tour that is ideally near-optimal.

    .. code-block:: python

      dists = [[0, 1], [1, 0]]
      tour = fast_tsp.find_tour(dists)
      print(tour)
      # [0, 1]
    """
    return __find_tour(dists, duration_seconds)


def greedy_nearest_neighbor(dists: DistMatrix) -> Tour:
    """Solve the TSP using the greedy nearest neighbor algorithm.

    Solve TSP using the greedy nearest neighbor heuristic. This is a very
    rough approximation to the exact solution. The preferred way is to
    use `find_tour`.

    Args:
        dists: The distance matrix.

    Returns:
        The tour computed by the nearest neighbor algorithm.
    """
    return __greedy_nearest_neighbor(dists)


def solve_tsp_exact(dists: DistMatrix) -> Tour:
    """Solve the TSP using the exact algorithm.

    Solves TSP optimally using the bottom-up Held-Karp's algorithm in
    :math:`O(2^n * n^2)` time. This is tractable for small n but quickly becomes
    untractable for n even of medium size.

    Args:
        dists: The distance matrix.

    Returns:
        An optimal tour for the problem instance.
    """
    if len(dists) > 20:
        warnings.warn(
            "The exact algorithm is not recommended for n > 20. "
            "Consider using the fast heuristic instead."
        )
    return __solve_tsp_exact(dists)


def is_valid_tour(n: int, tour: Tour) -> bool:
    """Check if a tour is valid.

    Args:
        n: The number of cities.
        tour: The tour to test whether it is valid.

    Returns:
        True if the tour is valid, False otherwise.
    """
    return __is_valid_tour(n, tour)


def compute_cost(tour: Tour, dists: DistMatrix) -> float:
    """Returns the cost of a tour.

    Computes the cost of a tour on a distance matrix. No validation on
    whether the tour is valid is performed.

    Args:
        tour: The tour.
        dists: The distance matrix.

    Returns:
        The cost of the tour.
    """
    return __compute_cost(tour, dists)


def score_tour(tour: Tour, dists: DistMatrix, opt_cost: int) -> float:
    """Return the score of a tour, where 1 is better and 0 is worst.

    Returns a double in the range 0..1, where 0 denotes a solution that
    is worse than or equivalent to the naive solution and 1 denotes a
    solution that is equivalent to the optimal solution.

    Args:
        tour: The tour.
        dists: The distance matrix.
        opt_cost: The cost of the optimal tour.

    Returns:
        The score of the tour.
    """
    return __score_tour(tour, dists, opt_cost)


__all__ = [
    '__version__',
    'find_tour',
    'greedy_nearest_neighbor',
    'solve_tsp_exact',
    'is_valid_tour',
    'compute_cost',
    'score_tour',
]
