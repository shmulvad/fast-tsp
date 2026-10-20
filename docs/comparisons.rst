Comparisons
============

This page compares ``fast-tsp`` with the well-known TSP solvers available from
Python (OR-Tools, python-tsp, tsp-solver2, frigidum and elkai) and with
`LKH-3 <http://webhotel4.ruc.dk/~keld/research/LKH-3/>`_, the reference heuristic
solver, on a few `TSPLIB <http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>`_
instances with known optimal tours. The numbers are how far the returned tour is
above the optimum (0.00 % means the optimal tour was found). Lower is better.

All of these, ``fast-tsp`` included, are heuristics: they return a short tour
but no proof that it is the shortest. If you need a guaranteed optimum, use an
exact solver such as `Concorde <https://www.math.uwaterloo.ca/tsp/concorde.html>`_
(free for academic research use; `pyconcorde <https://github.com/jvkersch/pyconcorde>`_
wraps it for Python), which solves instances of this size to proven optimality
in seconds to minutes.

Setup
#####

* Instances: ``berlin52``, ``kroA200``, ``pcb442``, ``pr1002``, ``pcb3038`` and
  ``fnl4461`` (the number is the node count), all with Euclidean distances rounded
  to integers as TSPLIB prescribes. Every solver gets the same integer distance
  matrix (or, for LKH-3, the coordinates that produce it) and every tour is scored
  with the same matrix.
* Machine: Apple M5 Pro, one solver running at a time, Python 3.12. Unless noted,
  each solver uses one thread.
* Each solver is called the way its documentation shows, with the time limit it
  offers:

  - ``fast-tsp`` 0.2.2: ``fast_tsp.find_tour(dists, duration_seconds=t)``, i.e. the
    defaults (``candidates='auto'``, the whole budget is used), and
    ``max_threads=4`` for the 4-thread column. Median of 3 runs.
  - LKH-3.0.13, built from source and run on the TSPLIB file with
    ``TIME_LIMIT = TOTAL_TIME_LIMIT = t``, ``RUNS = 1000``, ``SEED = 1`` and
    otherwise default parameters, which include a subgradient ascent before the
    search. "No ascent" is the same with ``SUBGRADIENT = NO``. LKH is deterministic
    for a given seed, so one run each.
  - `OR-Tools <https://developers.google.com/optimization/routing/tsp>`_ 9.15:
    routing model with ``PATH_CHEAPEST_ARC`` as first solution and
    ``GUIDED_LOCAL_SEARCH`` as metaheuristic, ``time_limit`` set to ``t``.
  - `python-tsp <https://pypi.org/project/python-tsp/>`_ 0.5: the two heuristics
    ``solve_tsp_local_search`` and ``solve_tsp_simulated_annealing`` with
    ``max_processing_time=t``. Both start from a random permutation and run in
    pure Python, which is why they fall so far behind on larger instances. The
    exact solvers of the package are not usable at these sizes.
  - `elkai <https://pypi.org/project/elkai/>`_ 2.0.1 (LKH-3 as a Python package,
    ``solve_tsp(runs=1)``), `tsp-solver2 <https://pypi.org/project/tsp-solver2/>`_
    0.4 (``solve_tsp``) and `frigidum <https://pypi.org/project/frigidum/>`_ 0.4
    (simulated annealing with the settings of its bundled TSP example) have no
    time limit, so they are run once per instance and their wall time is
    reported instead.

Time limits are wall-clock budgets handed to the solver. ``fast-tsp``, OR-Tools and
python-tsp stayed within a few hundred milliseconds of their budget (OR-Tools needs
up to half a second extra to build its model on the largest instance). LKH-3's
time limit only applies to its search: the preprocessing before it (candidate
lists, and by default the subgradient ascent) is not bounded, and when it does not
finish within the limit no tour is written at all. Those cells say "no tour" with
the time LKH took.

Results
#######

Percent above the optimal tour length with a **0.2 second** budget:

+------------------+----------+---------------------+-----------------+------------------+----------------+-------------------------+----------------------+
| Instance         | fast-tsp | fast-tsp, 4 threads | LKH-3           | LKH-3, no ascent | OR-Tools (GLS) | python-tsp local search | python-tsp annealing |
+==================+==========+=====================+=================+==================+================+=========================+======================+
| berlin52 (n=52)  | 0.00 %   | 0.00 %              | 0.00 %          | 0.00 %           | 4.77 %         | 8.04 %                  | 5.52 %               |
+------------------+----------+---------------------+-----------------+------------------+----------------+-------------------------+----------------------+
| kroA200 (n=200)  | 0.00 %   | 0.00 %              | 0.00 %          | 0.00 %           | 1.72 %         | 78.25 %                 | 150.93 %             |
+------------------+----------+---------------------+-----------------+------------------+----------------+-------------------------+----------------------+
| pcb442 (n=442)   | 0.00 %   | 0.00 %              | 0.00 %          | 0.00 %           | 7.09 %         | 560.73 %                | 655.03 %             |
+------------------+----------+---------------------+-----------------+------------------+----------------+-------------------------+----------------------+
| pr1002 (n=1002)  | 0.24 %   | 0.00 %              | no tour (0.5 s) | 1.09 %           | 18.29 %        | 1803.95 %               | 1875.65 %            |
+------------------+----------+---------------------+-----------------+------------------+----------------+-------------------------+----------------------+
| pcb3038 (n=3038) | 0.36 %   | 0.27 %              | no tour (4.4 s) | 0.40 %           | 24.53 %        | 3681.92 %               | 3757.30 %            |
+------------------+----------+---------------------+-----------------+------------------+----------------+-------------------------+----------------------+
| fnl4461 (n=4461) | 0.62 %   | 0.31 %              | no tour (8.9 s) | no tour (0.6 s)  | 24.44 %        | 4385.24 %               | 4443.87 %            |
+------------------+----------+---------------------+-----------------+------------------+----------------+-------------------------+----------------------+

With a **2 second** budget:

+------------------+----------+---------------------+------------------+------------------+----------------+-------------------------+----------------------+
| Instance         | fast-tsp | fast-tsp, 4 threads | LKH-3            | LKH-3, no ascent | OR-Tools (GLS) | python-tsp local search | python-tsp annealing |
+==================+==========+=====================+==================+==================+================+=========================+======================+
| berlin52 (n=52)  | 0.00 %   | 0.00 %              | 0.00 %           | 0.00 %           | 0.00 %         | 18.58 %                 | 8.37 %               |
+------------------+----------+---------------------+------------------+------------------+----------------+-------------------------+----------------------+
| kroA200 (n=200)  | 0.00 %   | 0.00 %              | 0.00 %           | 0.00 %           | 1.72 %         | 7.96 %                  | 16.43 %              |
+------------------+----------+---------------------+------------------+------------------+----------------+-------------------------+----------------------+
| pcb442 (n=442)   | 0.00 %   | 0.00 %              | 0.00 %           | 0.00 %           | 2.24 %         | 90.89 %                 | 199.25 %             |
+------------------+----------+---------------------+------------------+------------------+----------------+-------------------------+----------------------+
| pr1002 (n=1002)  | 0.00 %   | 0.00 %              | 0.00 %           | 1.03 %           | 9.47 %         | 603.82 %                | 906.87 %             |
+------------------+----------+---------------------+------------------+------------------+----------------+-------------------------+----------------------+
| pcb3038 (n=3038) | 0.24 %   | 0.09 %              | no tour (3.8 s)  | 0.10 %           | 16.64 %        | 2673.52 %               | 2899.77 %            |
+------------------+----------+---------------------+------------------+------------------+----------------+-------------------------+----------------------+
| fnl4461 (n=4461) | 0.11 %   | 0.08 %              | no tour (10.4 s) | 0.02 %           | 22.13 %        | 3753.33 %               | 3927.47 %            |
+------------------+----------+---------------------+------------------+------------------+----------------+-------------------------+----------------------+

With a **10 second** budget:

+------------------+----------+---------------------+--------+------------------+----------------+-------------------------+----------------------+
| Instance         | fast-tsp | fast-tsp, 4 threads | LKH-3  | LKH-3, no ascent | OR-Tools (GLS) | python-tsp local search | python-tsp annealing |
+==================+==========+=====================+========+==================+================+=========================+======================+
| berlin52 (n=52)  | 0.00 %   | 0.00 %              | 0.00 % | 0.00 %           | 0.00 %         | 15.00 %                 | 5.03 %               |
+------------------+----------+---------------------+--------+------------------+----------------+-------------------------+----------------------+
| kroA200 (n=200)  | 0.00 %   | 0.00 %              | 0.00 % | 0.00 %           | 0.76 %         | 9.98 %                  | 10.57 %              |
+------------------+----------+---------------------+--------+------------------+----------------+-------------------------+----------------------+
| pcb442 (n=442)   | 0.00 %   | 0.00 %              | 0.00 % | 0.00 %           | 1.87 %         | 13.82 %                 | 35.58 %              |
+------------------+----------+---------------------+--------+------------------+----------------+-------------------------+----------------------+
| pr1002 (n=1002)  | 0.00 %   | 0.00 %              | 0.00 % | 0.00 %           | 4.66 %         | 195.86 %                | 301.66 %             |
+------------------+----------+---------------------+--------+------------------+----------------+-------------------------+----------------------+
| pcb3038 (n=3038) | 0.11 %   | 0.03 %              | 0.02 % | 0.01 %           | 14.59 %        | 1457.47 %               | 1707.85 %            |
+------------------+----------+---------------------+--------+------------------+----------------+-------------------------+----------------------+
| fnl4461 (n=4461) | 0.07 %   | 0.04 %              | 0.02 % | 0.02 %           | 17.31 %        | 2498.94 %               | 2605.36 %            |
+------------------+----------+---------------------+--------+------------------+----------------+-------------------------+----------------------+

Solvers **without a time limit** (one run each, wall time in parentheses;
"killed" means no result after 5 minutes, "error" means the library crashed on
that input):

+------------------+----------------+----------------+----------------+
| Instance         | elkai          | tsp-solver2    | frigidum       |
+==================+================+================+================+
| berlin52 (n=52)  | 0.0 % (0.0 s)  | 7.4 % (0.0 s)  | 2.3 % (2.9 s)  |
+------------------+----------------+----------------+----------------+
| kroA200 (n=200)  | 0.0 % (0.2 s)  | 8.7 % (0.0 s)  | 1.6 % (4.9 s)  |
+------------------+----------------+----------------+----------------+
| pcb442 (n=442)   | 0.0 % (1.2 s)  | 12.2 % (0.1 s) | 3.4 % (8.3 s)  |
+------------------+----------------+----------------+----------------+
| pr1002 (n=1002)  | 0.0 % (2.0 s)  | 9.8 % (0.8 s)  | 7.8 % (40.3 s) |
+------------------+----------------+----------------+----------------+
| pcb3038 (n=3038) | 0.0 % (59.5 s) | 7.7 % (7.3 s)  | killed         |
+------------------+----------------+----------------+----------------+
| fnl4461 (n=4461) | 0.0 % (93.3 s) | 7.1 % (17.4 s) | error          |
+------------------+----------------+----------------+----------------+

Reading the tables
##################

* ``fast-tsp`` finds the optimal tour of every instance up to 1002 nodes within
  2 seconds (and within 0.2 seconds with 4 threads), and stays within 0.6 % of
  the optimum on the 3038- and 4461-node instances even at 0.2 seconds. Four
  threads help on the large instances at every budget.
* LKH-3 finds better tours per second of *search* on the large instances: at
  10 seconds it is at 0.01–0.02 % on ``pcb3038`` and ``fnl4461`` where
  ``fast-tsp`` is at 0.03–0.11 %. Its preprocessing is the catch: with the
  default ascent LKH returns nothing within 2 seconds on either large instance,
  and even without the ascent it has nothing for ``fnl4461`` at 0.2 seconds.
  ``fast-tsp`` runs a shortened version of the same ascent only when it fits the
  budget, which is why it always returns a tour. elkai runs LKH's default
  configuration to convergence and finds every optimum, taking a minute for
  3038 nodes and a minute and a half for 4461.
* OR-Tools' guided local search is a good choice when you need routing
  constraints beyond a plain tour. On a plain TSP it finds the optimum of
  ``berlin52`` given 2 seconds, but stays 1–18 % above it on the instances with
  200–1002 nodes (depending on the budget) and 15–25 % above it on the two large
  ones, because its local search is a general routing search rather than a
  dedicated TSP one.
* ``tsp-solver2`` is quick, a greedy construction plus a limited improvement
  step, and lands 7–12 % above the optimum regardless of how long it takes.
* ``frigidum`` reaches 1.6–3.4 % on the small instances in a few seconds but
  takes 40 seconds for 1002 nodes and does not finish 3038 nodes in 5 minutes.
* The ``python-tsp`` heuristics are fine for a few dozen nodes but their pure
  Python neighbourhood scans cannot make progress from a random start on
  hundreds of nodes within these budgets.

``examples/compare_libraries.py`` in the repository produced these tables; it
runs every solver in a fresh process on the instance files in ``cases/`` and
adds the LKH-3 columns when given ``--lkh /path/to/LKH``. The Rust benchmark in
``examples/bench.rs`` covers all 73 instances for ``fast-tsp`` alone.
