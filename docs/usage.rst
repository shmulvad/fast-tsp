.. role:: pyt(code)
   :language: python

Usage
=============

This page describes how to use the :pyt:`fast_tsp` module.


Installation
#############

The module can be installed using pip; prebuilt wheels exist for Linux, macOS
and Windows, so no Rust toolchain is needed:

.. code-block:: bash

    $ pip install fast-tsp

numpy is optional. Distance matrices can be plain lists of lists, or integer
numpy arrays, which are read directly without per-element conversion. To
install numpy alongside, use :pyt:`pip install "fast-tsp[numpy]"`.


The distance matrix
####################

The solver works on a distance matrix: a square 2D matrix of integers where
entry :pyt:`[i][j]` is the distance from node :pyt:`i` to node :pyt:`j`. It
must be

* **symmetric**: the distance from :pyt:`i` to :pyt:`j` equals the distance
  from :pyt:`j` to :pyt:`i`. The heuristic used for more than 20 nodes relies on
  this and will not find good tours for asymmetric matrices.
* **integer**, between 0 and 65535 (the range of an unsigned 16-bit integer),
  with zeros on the diagonal. If your distances are floating point, scale them
  up to the precision you need and round to the nearest integer; if they are
  larger than 65535, scale them down. The tour only depends on the relative
  distances, so a common scale factor changes nothing.
* ideally **metric**, i.e. satisfy the triangle inequality. The solver works
  on any symmetric matrix, but its candidate lists and moves are designed for
  distances that behave like distances, and the quality of the tours degrades
  on matrices that do not.

:pyt:`is_valid_dist_matrix(dists)` checks these requirements and returns a
message describing the first problem it finds, or :pyt:`None`.


From coordinates to a distance matrix
######################################

If you have 2D coordinates rather than a distance matrix, build it with numpy
or scipy. Rounding to the nearest integer is the convention TSPLIB uses for
Euclidean instances:

.. code-block:: python

    import numpy as np
    from scipy.spatial.distance import cdist

    coords = np.random.rand(1500, 2) * 1000
    dists = np.rint(cdist(coords, coords)).astype(np.int64)
    print(dists[:3, :3])
    # [[  0, 581, 352],
    #  [581,   0, 427],
    #  [352, 427,   0]]

:pyt:`cdist` takes a :pyt:`metric` argument for other distances (for example
:pyt:`'cityblock'`), and :pyt:`sklearn.metrics.pairwise_distances` does the
same job. For geographic coordinates, compute great-circle distances in
metres, or in tens of metres if they would exceed 65535.


Finding a tour
###############

Once you have a distance matrix, finding a tour is one call:

.. code-block:: python

    import fast_tsp
    tour = fast_tsp.find_tour(dists)
    print(tour)
    # [0, 52, 1, 9, 22, ...]

The tour is a list of indices into the distance matrix, in visiting order,
starting at node 0. :pyt:`[0, 52, 1]` means the tour starts at node 0, goes to
node 52, then to node 1, and finally back to node 0 to close the tour. The
tour is a cycle, so if you need to start somewhere else, rotate the list.

Problems of 20 nodes or fewer are solved exactly and return immediately;
larger ones run the local search described in :doc:`about` for the time
budget. :pyt:`compute_cost(tour, dists)` gives the length of a tour.


Choosing the time budget
#########################

:pyt:`duration_seconds` (default 2.0) is a wall-clock budget for the whole
call, including setup, and the solver keeps looking for a better tour until it
runs out. As a rule of thumb from the TSPLIB instances (see
:doc:`comparisons`): instances of up to about 1000 nodes are solved to
optimality or within a few tenths of a percent of it in 0.2 seconds and to
optimality in 2 seconds; instances of 3000–4500 nodes are within about 0.6 %
of the optimum after 0.2 seconds, 0.2 % after 2 seconds and 0.1 % after
10 seconds. Larger budgets keep improving the tour slowly. Pressing Ctrl-C
stops the search and raises :pyt:`KeyboardInterrupt`.

.. code-block:: python

    tour = fast_tsp.find_tour(dists, duration_seconds=0.5)

If you would rather get the answer back as soon as the search stops making
progress, pass :pyt:`stop_when_converged=True`. The solver then returns once
it has gone :pyt:`max(100_000, 500 * n)` perturbations without improving the
best tour, which on small problems happens within a second or two. A budget
of :pyt:`float('inf')` always stops on convergence.

.. code-block:: python

    tour = fast_tsp.find_tour(dists, duration_seconds=30, stop_when_converged=True)


Using multiple cores
#####################

By default the search runs on one thread. Pass :pyt:`max_threads` to let
several threads search together. They share the best tour found so far, so
4 threads find tours about as good as 1 thread given 4 times as long. Pass
:pyt:`None` to use one fewer than the number of available CPU cores:

.. code-block:: python

    tour = fast_tsp.find_tour(dists, duration_seconds=1.0, max_threads=None)

The time limit is the same whatever the number of threads. Problems of 20
nodes or fewer are solved exactly on one thread, and :pyt:`max_threads` is
never allowed to exceed the number of available cores. If you are already
solving many problems in parallel (e.g. with :pyt:`multiprocessing`), keep the
default of 1.


Candidate lists
################

The local search only considers moves between each node and a short list of
candidate neighbours. :pyt:`candidates` chooses how those lists are built:

* :pyt:`'nearest'`: the nearest neighbours of each node. Cheap to build.
* :pyt:`'alpha'`: LKH-style alpha-nearness. A minimum 1-tree with
  subgradient-optimised node penalties ranks neighbours by how likely their
  edge is to be in an optimal tour, and the local search then uses a deeper
  Lin-Kernighan step (a sequential 3-opt basis move, as in LKH) over the five
  best-ranked candidates of each node. The full subgradient ascent takes
  about 0.2 s for 1000 nodes and 2 s for 4500 nodes on one thread (it is
  capped at half the budget), and pays off on problems of more than a few
  hundred nodes.
* :pyt:`'auto'` (default): a shortened ascent that fits in about a tenth of
  the budget, or the nearest neighbours when not even a short one fits. With
  the default 2 second budget that means alpha lists for every TSPLIB
  instance up to 4461 nodes; with 0.2 seconds, up to about 1800 nodes.

Forcing :pyt:`'alpha'` on a short budget spends up to half of it on the
preprocessing. :pyt:`'nearest'` is what versions up to 0.2.1 used, and it is
still the better choice on a few instances whose alpha lists are poor (the
TSPLIB instance ``rl5915`` is one), so try it if a result disappoints.

.. code-block:: python

    tour = fast_tsp.find_tour(dists, duration_seconds=10, candidates='alpha')


Exact solutions
################

:pyt:`solve_tsp_exact(dists)` finds a provably optimal tour with the Held-Karp
dynamic program, in O(2^n · n^2) time and O(2^n · n) memory. That takes a
few hundredths of a second and under 100 MB at 20 nodes but doubles with every
node, which is why :pyt:`find_tour` uses it up to 20 nodes and the heuristic
above that.

If you need a guaranteed optimum for larger instances, use a dedicated exact
solver. `Concorde <https://www.math.uwaterloo.ca/tsp/concorde.html>`_ is the
state of the art: a branch-and-cut code that has solved every TSPLIB instance
to proven optimality, typically in seconds to minutes for a few thousand nodes
(it is free for academic research use, and `pyconcorde
<https://github.com/jvkersch/pyconcorde>`_ wraps it for Python).
:pyt:`fast_tsp` is a heuristic: on the instances in :doc:`comparisons` it
returns the optimal tour most of the time, but it cannot tell you that it did.


Other helpers
##############

* :pyt:`greedy_nearest_neighbor(dists)` returns the nearest-neighbour tour the
  local search starts from. On the TSPLIB instances it is 5–40 % (typically
  25 %) longer than the optimum, and useful as a quick baseline.
* :pyt:`compute_cost(tour, dists)` returns the length of a tour.
* :pyt:`is_valid_tour(n, tour)` checks that a tour visits each of the
  :pyt:`n` nodes exactly once.
* :pyt:`score_tour(tour, dists, opt_cost)` scores a tour between 0 and 1
  against a known optimal cost, as the `Kattis TSP problem
  <https://open.kattis.com/problems/tsp>`_ does: with
  :math:`d = (actual - opt) / (naive - opt)`, where :math:`naive` is the
  nearest-neighbour cost, the score is :math:`0.02^{\max(d, 0)}`. It is 1.0
  for an optimal tour and 0.0 for one no better than the nearest-neighbour
  tour. (If the nearest-neighbour tour is itself optimal, the score is 1.0
  for a tour of that cost and 0.0 for anything worse.)

.. code-block:: python

    import fast_tsp
    tour = fast_tsp.find_tour(dists)
    print(fast_tsp.compute_cost(tour, dists))  # 5984
    print(fast_tsp.compute_cost(fast_tsp.greedy_nearest_neighbor(dists), dists))  # 6812
    print(fast_tsp.is_valid_tour(len(dists), tour))  # True


Troubleshooting
################

* **"Invalid distance matrix" errors**: check the requirements listed under
  *The distance matrix* above with :pyt:`is_valid_dist_matrix`. The most
  common causes are floating point entries, negative entries, distances above
  65535 and non-square inputs.
* **Poor tours**: make sure the matrix is symmetric and that you gave the
  solver enough time; a few thousand nodes need seconds, not milliseconds.
  Try :pyt:`max_threads=None`, and :pyt:`candidates='nearest'` or
  :pyt:`'alpha'` to see whether the candidate lists are the issue.
* **The call takes the whole budget on a tiny problem**: that is the default
  since 0.2.2; pass :pyt:`stop_when_converged=True` or a shorter budget.

If you are still stuck, please open an issue on the
`GitHub page <https://github.com/shmulvad/fast-tsp>`_.
