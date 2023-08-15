.. role:: pyt(code)
   :language: python

Usage
=============

This is a short description of how to use the :pyt:`fast-tsp` module.


Installation
#############

The module can be installed using pip:

.. code-block:: bash

    $ pip install fast-tsp


Example usage
##############

The program works on a distance matrix. The distance matrix should be a square
2D matrix of integers. Note that the distance matrix should satisfy the
triangle inequality, otherwise the results might not make sense. For most
real-world problems, this inequality is satisfied. Note that for speed reasons,
this module requires the distance matrix to be integers. If you have a use-case
that requires floating point arithmetic, you can scale the distance matrix
up (multiplying by some given number that is aligned with the level of accuracy you want) and then round to the nearest integer.


I don't have a distance matrix, but I have 2D coordinates
##############################################################

If you have a list of 2D coordinates, you can use `sklearn` to generate this
matrix for various distance functions using `pairwise_distances`.

.. code-block:: python

    import numpy as np
    from sklearn.metrics import pairwise_distances
    coords = np.round(np.random.rand(1500, 2) * 100)
    print(coords[:3].astype(np.int64))
    # [[81, 32],
    #  [23, 32],
    #  [57,  6]]

    dists = pairwise_distances(coords).astype(np.int64)
    print(dists[:3, :3])
    # [[ 0, 58, 35],
    #  [58,  0, 42],
    #  [35, 42,  0]]



How to run main module
########################

Once you have a distance matrix, it is very easy to run the module:

.. code-block:: python

    import fast_tsp
    tour = fast_tsp.find_tour(dists)
    print(tour)
    # [0, 52, 1, 9, 22, ...]

The tour is a list of indices that correspond to the order of the nodes in the
supplied distance matrix. E.g. [0, 52, 1] would imply that the tour starts at
index 0, goes to the node at index 52, goes to node 1, and finally ends at node
0 again, closing the tour.



Help, I'm getting errors when I run the module
######################################################

If you are getting errors while running the module, make sure you are fulfilling
the following requirements:

* The distance matrix should be a square 2D matrix of integers.
* The distance matrix should satisfy the triangle inequality.
* The distance matrix should be symmetric.
* The distance matrix should be positive.
* The distance matrix should be zero on the diagonal.
* The distance matrix should be less than 2^16 - 1 (the maximum value for an unsigned 16-bit integer).

You can check whether your distance matrix fulfills these requirements by using
the :pyt:`is_valid_distance_matrix` function. If you are still getting errors,
please open an issue on the `GitHub page <https://github.com/shmulvad/fast-tsp>`.



Exact solution
###############

If you are running the program on a small problem, you can use the :pyt:`solve_tsp_exact`
function to get the optimal solution.
This solves TSP optimally using the bottom-up Held-Karp's algorithm in
O(2^n * n^2) time. This is tractable for small `n` but quickly becomes
untractable for `n` even of medium size. This function will also be used in the
main :pyt:`find_tour` function for problem instances of less than or equal to `n = 20` in size.



Greedy Nearest Neighbor
########################

This is the algorithm that is used in the local solver to get an initial solution
that is then improved upon using iterative local search. If you just desire
a very rough approximation of the optimal solution, you can use
:pyt:`greedy_nearest_neighbor`.



Scoring
###########

A helper function for computing the cost of a given tour is included. You can
use this to get an idea of the quality of the returned solution:

.. code-block:: python

    import fast_tsp
    tour = fast_tsp.find_tour(dists)  # Assume `dists` are defined
    cost_solver = fast_tsp.compute_cost(tour, dists)
    print(cost_solver)  # 5984
    cost_greedy = fast_tsp.greedy_nearest_neighbor(dists)
    print(cost_greedy)  # 6812


It can be interesting to know how well a solution performs, taking into account
the greedy solution and the optimal cost. If you know what the optimal cost is,
you can use :pyt:`score`-function to get a score for your solution.



Validation
###########
You can check whether a solution is valid by using the :pyt:`is_valid_tour` function.
A solution is valid if it is visits every node in the graph exactly once.
If you know what the optimal solution is, you can also score the solution
using the :pyt:`score_tour` function. This compares the solution to the one
returned by the greedy nearest neighbor algorithm and then computes the score
as follows:


.. math::

    \begin{align}
    d = \frac{actual - opt}{naive - opt},
    \ \ \
    score = 0.02^{\max(d, 0)}
    \end{align}

This will be 1.0 if the solution is optimal, 0.0 if it is worse than or equal to
the naive solution and somewhere in-between for values in between.
In the special case where the optimal solution is the naive solution, the score
will be 1.0 if actual is equal to naive, and 0.0 if actual is less than naive.
