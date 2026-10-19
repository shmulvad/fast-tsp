About
===========

This is a Travelling Salesman Problem (TSP) solver with the core written in Rust
and Python bindings on top. It works on a symmetric integer distance matrix and
finds near-optimal tours within a time budget you choose.

Problems of up to 20 nodes are solved exactly with the Held-Karp dynamic
program. Larger problems use an iterated local search: a nearest-neighbour
start tour is improved with a Lin-Kernighan style variable-depth move (with
2-opt and 3-opt as fallback) restricted to each node's candidate neighbours,
and then repeatedly perturbed with small double-bridge kicks and re-optimised
around the change. The candidates are the 20 nearest neighbours, and when the
budget allows it also LKH's alpha-nearest neighbours, ranked with a minimum
1-tree after a subgradient ascent on node penalties; with those lists the
variable-depth move is built from sequential 3-opt moves instead of 2-opt
flips, which is what closes most of the gap to LKH on large instances. Only the nodes touched by a kick are re-examined, so each
kick costs work proportional to the change rather than to the problem size,
and a walk that stops improving is restarted from a perturbed copy of the best
tour. Several threads can search together and share the best tour.

On the TSPLIB instances with up to 4461 nodes the solver typically returns the
optimal tour within 2 seconds for instances below about 1000 nodes and stays
within 1 % of the optimum on the largest ones even with a 0.2 second budget.
See :doc:`comparisons` for measurements against other Python packages, and
the ``examples/bench.rs`` benchmark in the repository for the full 73-instance
scoreboard.
