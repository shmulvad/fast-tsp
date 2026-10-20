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
optimal tour within 2 seconds for instances below about 1000 nodes, and on the
largest ones stays within 0.6 % of the optimum with a 0.2 second budget and
within 0.1 % after 10 seconds. It is a heuristic, so it cannot prove a tour
optimal; see :doc:`usage` for when to reach for an exact solver instead.
:doc:`comparisons` has measurements against other Python packages and LKH-3,
and the ``examples/bench.rs`` benchmark in the repository the full 73-instance
scoreboard.


References and acknowledgements
################################

The local search builds on well-known published methods, re-implemented here
in Rust. In particular the candidate lists and the deeper move follow Keld
Helsgaun's `LKH <http://webhotel4.ruc.dk/~keld/research/LKH-3/>`_, which
remains the stronger solver when its preprocessing time is not an issue:

* S. Lin and B. W. Kernighan, *An effective heuristic algorithm for the
  traveling-salesman problem*, Operations Research 21 (1973).
* K. Helsgaun, *An effective implementation of the Lin-Kernighan traveling
  salesman heuristic*, European Journal of Operational Research 126 (2000):
  alpha-nearness candidates from a minimum 1-tree and the subgradient ascent
  on node penalties.
* K. Helsgaun, *General k-opt submoves for the Lin-Kernighan TSP heuristic*,
  Mathematical Programming Computation 1 (2009): the sequential 3-opt basis
  move.
* O. Martin, S. W. Otto and E. W. Felten, *Large-step Markov chains for the
  traveling salesman problem*, Complex Systems 5 (1991): the double-bridge
  kick and iterated local search.
* J. L. Bentley, *Fast algorithms for geometric traveling salesman problems*,
  ORSA Journal on Computing 4 (1992): neighbour lists and don't-look bits.

No code from these projects is included; ``fast-tsp`` is MIT licensed.
