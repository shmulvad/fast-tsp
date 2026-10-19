Comparisons
============

This page compares ``fast-tsp`` with the other Python TSP packages that come up in
`this StackOverflow thread <https://stackoverflow.com/questions/71196078/scalable-implementation-of-the-travelling-salesman-problem-in-python>`_
on a few `TSPLIB <http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/>`_
instances with known optimal tours. The numbers are how far the returned tour is
above the optimum (0.0 % means the optimal tour was found). Lower is better.

Setup
#####

* Instances: ``berlin52``, ``kroA200``, ``pcb442``, ``pr1002``, ``pcb3038`` and
  ``fnl4461`` (the number is the node count), all with Euclidean distances rounded
  to integers as TSPLIB prescribes. Every solver gets the same integer distance
  matrix and every tour is scored with the same matrix.
* Machine: Apple M5 Pro, one solver running at a time, Python 3.12. Unless noted,
  each solver uses one thread.
* Each solver is called the way the StackOverflow answers show, with the time
  limit it offers:

  - ``fast-tsp``: ``fast_tsp.find_tour(dists, duration_seconds=t)`` (and
    ``max_threads=4`` for the 4-thread column).
  - `OR-Tools <https://developers.google.com/optimization/routing/tsp>`_ 9.15:
    routing model with ``PATH_CHEAPEST_ARC`` as first solution and
    ``GUIDED_LOCAL_SEARCH`` as metaheuristic, ``time_limit`` set to ``t``.
  - `python-tsp <https://pypi.org/project/python-tsp/>`_ 0.5: the two heuristics
    ``solve_tsp_local_search`` and ``solve_tsp_simulated_annealing`` with
    ``max_processing_time=t``. Both start from a random permutation and run in
    pure Python, which is why they fall so far behind on larger instances. The
    exact solvers of the package are not usable at these sizes.
  - `tsp-solver2 <https://pypi.org/project/tsp-solver2/>`_ 0.4 (``solve_tsp``)
    and `frigidum <https://pypi.org/project/frigidum/>`_ 0.4 (simulated annealing
    with the parameters from the StackOverflow answer) have no time limit, so
    they are run once per instance and their wall time is reported instead.
    ``tsp-c`` from the thread no longer installs.

Time limits are wall-clock budgets handed to the solver, and all of them stayed
within a few hundred milliseconds of their budget (OR-Tools needs up to half a
second extra to build its model on the largest instance). ``fast-tsp`` returns
earlier than the budget when its search has converged, for example on the
instances where it reports 0.0 %.

Results
#######

Percent above the optimal tour length with a **0.2 second** budget:

+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| Instance         | fast-tsp (this version) | fast-tsp, 4 threads | OR-Tools (GLS) | python-tsp local search | python-tsp annealing |
+==================+=========================+=====================+================+=========================+======================+
| berlin52 (n=52)  | 0.0 %                   | 0.0 %               | 4.8 %          | 8.0 %                   | 5.5 %                |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| kroA200 (n=200)  | 0.0 %                   | 0.0 %               | 1.7 %          | 78.2 %                  | 150.9 %              |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| pcb442 (n=442)   | 0.0 %                   | 0.0 %               | 7.1 %          | 560.7 %                 | 655.0 %              |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| pr1002 (n=1002)  | 0.4 %                   | 0.2 %               | 18.3 %         | 1804.0 %                | 1875.6 %             |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| pcb3038 (n=3038) | 0.4 %                   | 0.3 %               | 24.5 %         | 3681.9 %                | 3757.3 %             |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| fnl4461 (n=4461) | 0.6 %                   | 0.3 %               | 24.4 %         | 4385.2 %                | 4443.9 %             |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+

With a **2 second** budget:

+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| Instance         | fast-tsp (this version) | fast-tsp, 4 threads | OR-Tools (GLS) | python-tsp local search | python-tsp annealing |
+==================+=========================+=====================+================+=========================+======================+
| berlin52 (n=52)  | 0.0 %                   | 0.0 %               | 0.0 %          | 18.6 %                  | 8.4 %                |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| kroA200 (n=200)  | 0.0 %                   | 0.0 %               | 1.7 %          | 8.0 %                   | 16.4 %               |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| pcb442 (n=442)   | 0.0 %                   | 0.0 %               | 2.2 %          | 90.9 %                  | 199.2 %              |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| pr1002 (n=1002)  | 0.0 %                   | 0.0 %               | 9.5 %          | 603.8 %                 | 906.9 %              |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| pcb3038 (n=3038) | 0.3 %                   | 0.1 %               | 16.6 %         | 2673.5 %                | 2899.8 %             |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| fnl4461 (n=4461) | 0.2 %                   | 0.1 %               | 22.1 %         | 3753.3 %                | 3927.5 %             |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+

With a **10 second** budget:

+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| Instance         | fast-tsp (this version) | fast-tsp, 4 threads | OR-Tools (GLS) | python-tsp local search | python-tsp annealing |
+==================+=========================+=====================+================+=========================+======================+
| berlin52 (n=52)  | 0.0 %                   | 0.0 %               | 0.0 %          | 15.0 %                  | 5.0 %                |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| kroA200 (n=200)  | 0.0 %                   | 0.0 %               | 0.8 %          | 10.0 %                  | 10.6 %               |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| pcb442 (n=442)   | 0.0 %                   | 0.0 %               | 1.9 %          | 13.8 %                  | 35.6 %               |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| pr1002 (n=1002)  | 0.0 %                   | 0.0 %               | 4.7 %          | 195.9 %                 | 301.7 %              |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| pcb3038 (n=3038) | 0.1 %                   | 0.1 %               | 14.6 %         | 1457.5 %                | 1707.8 %             |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+
| fnl4461 (n=4461) | 0.1 %                   | 0.1 %               | 17.3 %         | 2498.9 %                | 2605.4 %             |
+------------------+-------------------------+---------------------+----------------+-------------------------+----------------------+

Solvers **without a time limit** (one run each, wall time in parentheses;
"killed" means no result after 5 minutes, "error" means the library crashed on
that input):

+------------------+----------------+----------------+
| Instance         | tsp-solver2    | frigidum       |
+==================+================+================+
| berlin52 (n=52)  | 7.4 % (0.0 s)  | 2.3 % (2.9 s)  |
+------------------+----------------+----------------+
| kroA200 (n=200)  | 8.7 % (0.0 s)  | 1.6 % (4.9 s)  |
+------------------+----------------+----------------+
| pcb442 (n=442)   | 12.2 % (0.1 s) | 3.4 % (8.3 s)  |
+------------------+----------------+----------------+
| pr1002 (n=1002)  | 9.8 % (0.8 s)  | 7.8 % (40.3 s) |
+------------------+----------------+----------------+
| pcb3038 (n=3038) | 7.7 % (7.3 s)  | killed         |
+------------------+----------------+----------------+
| fnl4461 (n=4461) | 7.1 % (17.4 s) | error          |
+------------------+----------------+----------------+

Reading the tables
##################

* ``fast-tsp`` finds the optimal tour of every instance up to 1002 nodes within
  2 seconds, and stays within 0.6 % of the optimum on the 3038- and 4461-node
  instances even at 0.2 seconds. Four threads help on the large instances at the
  short budgets.
* OR-Tools' guided local search is the closest competitor and is a good choice
  when you need routing constraints beyond a plain tour. On a plain TSP it
  finds the optimum of ``berlin52`` given 2 seconds, but stays 1–18 % above it
  on the instances with 200–1002 nodes (depending on the budget) and 15–25 %
  above it on the two large ones, because its local search is a general
  routing search rather than a dedicated TSP one.
* ``tsp-solver2`` is quick, a greedy construction plus a limited improvement
  step, and lands 7–12 % above the optimum regardless of how long it takes.
* ``frigidum`` reaches 1.6–3.4 % on the small instances in a few seconds but
  takes 40 seconds for 1002 nodes and does not finish 3038 nodes in 5 minutes.
* The ``python-tsp`` heuristics are fine for a few dozen nodes but their pure
  Python neighbourhood scans cannot make progress from a random start on
  hundreds of nodes within these budgets.

``examples/compare_libraries.py`` in the repository produced these tables; it
runs every solver in a fresh process on the instance files in ``cases/``. The
Rust benchmark in ``examples/bench.rs`` covers all 73 instances for ``fast-tsp``
alone.
