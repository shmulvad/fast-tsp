"""Compares fast-tsp with other Python TSP packages on TSPLIB instances.

Produces the tables in docs/comparisons.rst. Install the competitors first:

    pip install fast-tsp ortools python-tsp tsp-solver2 frigidum numpy

Then run from the repository root (every solver runs in a fresh process, one
at a time, and the reStructuredText tables are printed at the end):

    python examples/compare_libraries.py [--cases berlin52,kroA200,...] [--budgets 0.2,2,10]

Internally the script calls itself as
`compare_libraries.py run <solver> <case file> <budget>` for each measurement.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess  # noqa: S404  (runs this script itself)
import sys
import time
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parent.parent
DEFAULT_CASES = ['berlin52', 'kroA200', 'pcb442', 'pr1002', 'pcb3038', 'fnl4461']
DEFAULT_BUDGETS = [0.2, 2.0, 10.0]
# Solvers that accept a time limit, and the column label for each.
BUDGETED = [
    ('fast_tsp', 'fast-tsp'),
    ('fast_tsp_4t', 'fast-tsp, 4 threads'),
    ('ortools', 'OR-Tools (GLS)'),
    ('python_tsp_ls', 'python-tsp local search'),
    ('python_tsp_sa', 'python-tsp annealing'),
]
# Solvers without a time limit: one run per instance.
UNBUDGETED = [('tsp_solver2', 'tsp-solver2'), ('frigidum', 'frigidum')]
UNBUDGETED_TIMEOUT = 300.0


def load_case(path: pathlib.Path):
    """Returns (opt, n, points) from a file in cases/."""
    with pathlib.Path(path).open() as f:
        opt = int(f.readline().split()[0])
        n = int(f.readline())
        pts = [[float(x) for x in f.readline().split()] for _ in range(n)]
    return opt, n, pts


def run_one(solver: str, case_file: str, budget: float) -> dict:
    import numpy as np

    opt, n, pts_list = load_case(pathlib.Path(case_file))
    pts = np.array(pts_list)
    diff = pts[:, None, :] - pts[None, :, :]
    euclid = np.sqrt((diff**2).sum(-1))
    # TSPLIB EUC_2D: Euclidean distance rounded to the nearest integer.
    exact = np.rint(euclid).astype(np.int64)
    # fast-tsp needs distances below 2^16; scale down if an instance exceeds that.
    factor = 1.0 if exact.max() <= 65535 else 65535.0 / float(euclid.max())
    scaled = np.minimum(np.rint(euclid * factor), 65535).astype(np.int32)

    # Import before timing: loading a library is not solving.
    if solver.startswith('fast_tsp'):
        import fast_tsp
    elif solver == 'ortools':
        from ortools.constraint_solver import pywrapcp
        from ortools.constraint_solver import routing_enums_pb2
    elif solver == 'python_tsp_ls':
        from python_tsp.heuristics import solve_tsp_local_search
    elif solver == 'python_tsp_sa':
        from python_tsp.heuristics import solve_tsp_simulated_annealing
    elif solver == 'tsp_solver2':
        from tsp_solver.greedy import solve_tsp
    elif solver == 'frigidum':
        import frigidum
        from frigidum.examples import tsp

    start = time.perf_counter()
    if solver == 'fast_tsp':
        tour = fast_tsp.find_tour(scaled, duration_seconds=budget)
    elif solver == 'fast_tsp_4t':
        tour = fast_tsp.find_tour(scaled, duration_seconds=budget, max_threads=4)
    elif solver == 'ortools':
        manager = pywrapcp.RoutingIndexManager(n, 1, 0)
        routing = pywrapcp.RoutingModel(manager)
        cb = routing.RegisterTransitMatrix(exact.tolist())
        routing.SetArcCostEvaluatorOfAllVehicles(cb)
        params = pywrapcp.DefaultRoutingSearchParameters()
        params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromMilliseconds(int(budget * 1000))
        sol = routing.SolveWithParameters(params)
        if sol is None:
            raise RuntimeError('OR-Tools found no solution')
        tour, idx = [], routing.Start(0)
        while not routing.IsEnd(idx):
            tour.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
    elif solver == 'python_tsp_ls':
        tour, _ = solve_tsp_local_search(exact, max_processing_time=budget)
    elif solver == 'python_tsp_sa':
        tour, _ = solve_tsp_simulated_annealing(exact, max_processing_time=budget)
    elif solver == 'tsp_solver2':
        tour = solve_tsp(exact, endpoints=(0, 0))
        tour = tour[:-1] if len(tour) == n + 1 else tour
    elif solver == 'frigidum':
        # The configuration from the StackOverflow answer.
        tsp.nodes = pts
        tsp.nodes_count = n
        tsp.dist_eu = exact.astype(float)
        local_opt = frigidum.sa(
            random_start=tsp.random_start,
            objective_function=tsp.objective_function,
            neighbours=[
                tsp.euclidian_bomb_and_fix,
                tsp.euclidian_nuke_and_fix,
                tsp.route_bomb_and_fix,
                tsp.route_nuke_and_fix,
                tsp.random_disconnect_vertices_and_fix,
            ],
            copy_state=frigidum.annealing.naked,
            T_start=5,
            alpha=0.8,
            T_stop=0.001,
            repeats=10**2,
            post_annealing=tsp.local_search_2opt,
        )
        tour = list(local_opt[0])
    else:
        raise SystemExit(f'unknown solver {solver}')
    elapsed = time.perf_counter() - start

    t = np.asarray(tour, dtype=np.int64)
    if len(t) != n or len(set(t.tolist())) != n:
        raise RuntimeError('invalid tour')
    cost = int(exact[t, np.roll(t, -1)].sum())
    return {
        'solver': solver,
        'case': pathlib.Path(case_file).stem,
        'n': n,
        'budget': budget,
        'cost': cost,
        'opt': opt,
        'gap_pct': 100.0 * (cost - opt) / opt,
        'elapsed': round(elapsed, 3),
    }


def measure(solver: str, case: str, budget: float, timeout: float) -> dict:
    """Runs one measurement in a fresh interpreter so solvers cannot affect each other."""
    cmd = [sys.executable, '-I', __file__, 'run', solver, str(REPO / 'cases' / f'{case}.txt'), str(budget)]
    started = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)  # noqa: S603
    except subprocess.TimeoutExpired:
        return {'solver': solver, 'case': case, 'budget': budget, 'error': f'killed after {timeout:.0f} s'}
    lines = [line for line in p.stdout.splitlines() if line.startswith('{')]
    if p.returncode != 0 or not lines:
        return {'solver': solver, 'case': case, 'budget': budget, 'error': (p.stderr or 'no output')[-300:]}
    rec = json.loads(lines[-1])
    rec['wall'] = round(time.perf_counter() - started, 3)
    return rec


def cell(r: dict | None, show_time: bool = False) -> str:
    if r is None:
        return '—'
    if 'error' in r:
        return 'killed' if r['error'].startswith('killed') else 'error'
    s = f'{r["gap_pct"]:.1f} %'
    if show_time:
        s += f' ({r["elapsed"]:.1f} s)'
    return s


def rst_table(header: list[str], rows: list[list[str]]) -> str:
    widths = [max(len(x) for x in col) for col in zip(header, *rows)]
    line = '+' + '+'.join('-' * (w + 2) for w in widths) + '+'
    head = '+' + '+'.join('=' * (w + 2) for w in widths) + '+'

    def fmt(row: list[str]) -> str:
        return '|' + '|'.join(f' {x:<{w}} ' for x, w in zip(row, widths)) + '|'

    out = [line, fmt(header), head]
    for row in rows:
        out += [fmt(row), line]
    return '\n'.join(out)


def report(recs: list[dict], cases: list[str], budgets: list[float]) -> str:
    res: dict = defaultdict(dict)
    n_of = {}
    for r in recs:
        res[r['solver'], r['case']][r['budget']] = r
        if 'n' in r:
            n_of[r['case']] = r['n']
    out = []
    for budget in budgets:
        out.append(f'\nTime budget {budget:g} s\n')
        rows = [[f'{c} (n={n_of.get(c, "?")})'] + [cell(res[s, c].get(budget)) for s, _ in BUDGETED] for c in cases]
        out.append(rst_table(['Instance'] + [label for _, label in BUDGETED], rows))
    out.append('\nNo time budget (one run, wall time in parentheses)\n')
    rows = [[f'{c} (n={n_of.get(c, "?")})'] + [cell(res[s, c].get(0.0), show_time=True) for s, _ in UNBUDGETED] for c in cases]
    out.append(rst_table(['Instance'] + [label for _, label in UNBUDGETED], rows))
    return '\n'.join(out)


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == 'run':
        print(json.dumps(run_one(sys.argv[2], sys.argv[3], float(sys.argv[4]))))
        return
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cases', default=','.join(DEFAULT_CASES))
    ap.add_argument('--budgets', default=','.join(str(b) for b in DEFAULT_BUDGETS))
    ap.add_argument('--out', default='compare_results.jsonl', help='where to append the raw results')
    args = ap.parse_args()
    cases = args.cases.split(',')
    budgets = [float(b) for b in args.budgets.split(',')]
    recs = []
    with pathlib.Path(args.out).open('a') as out:
        for case in cases:
            for solver, _ in BUDGETED:
                for budget in budgets:
                    recs.append(measure(solver, case, budget, timeout=3 * budget + 120))
                    out.write(json.dumps(recs[-1]) + '\n')
                    out.flush()
                    print(recs[-1], file=sys.stderr, flush=True)
            for solver, _ in UNBUDGETED:
                recs.append(measure(solver, case, 0.0, timeout=UNBUDGETED_TIMEOUT))
                out.write(json.dumps(recs[-1]) + '\n')
                out.flush()
                print(recs[-1], file=sys.stderr, flush=True)
    print(report(recs, cases, budgets))


if __name__ == '__main__':
    main()
