"""Compares fast-tsp with other Python TSP packages on TSPLIB instances.

Produces the tables in docs/comparisons.rst. Install the competitors first:

    pip install fast-tsp ortools python-tsp tsp-solver2 frigidum elkai numpy

Then run from the repository root (every solver runs in a fresh process, one
at a time, and the reStructuredText tables are printed at the end):

    python examples/compare_libraries.py [--cases berlin52,kroA200,...] [--budgets 0.2,2,10] [--repeats 3]

With `--repeats`, the tables show the median over the runs of each cell.

LKH-3 is included when `--lkh /path/to/LKH` points at a binary built from
http://webhotel4.ruc.dk/~keld/research/LKH-3/ (it is run with TIME_LIMIT and
TOTAL_TIME_LIMIT set to the budget, both with its default subgradient ascent
and with SUBGRADIENT = NO). elkai wraps LKH without a time limit, so it gets
one run per instance like tsp-solver2 and frigidum.

Internally the script calls itself as
`compare_libraries.py run <solver> <case file> <budget>` for each measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import statistics
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
# LKH-3 variants, only measured when --lkh is given.
LKH = [('lkh3', 'LKH-3'), ('lkh3_nosub', 'LKH-3, no ascent')]
LKH_PARAMS = {'lkh3': '', 'lkh3_nosub': 'SUBGRADIENT = NO\n'}
# Solvers without a time limit: one run per instance.
UNBUDGETED = [('tsp_solver2', 'tsp-solver2'), ('frigidum', 'frigidum'), ('elkai', 'elkai (LKH)')]
UNBUDGETED_TIMEOUT = 300.0


def load_case(path: pathlib.Path):
    """Returns (opt, n, points) from a file in cases/."""
    with pathlib.Path(path).open() as f:
        opt = int(f.readline().split()[0])
        n = int(f.readline())
        pts = [[float(x) for x in f.readline().split()] for _ in range(n)]
    return opt, n, pts


def _import_solver(solver: str) -> dict:
    """Imports the solver's library before timing starts and returns its namespace."""
    if solver.startswith('fast_tsp'):
        import fast_tsp

        return {'fast_tsp': fast_tsp}
    elif solver == 'ortools':
        from ortools.constraint_solver import pywrapcp
        from ortools.constraint_solver import routing_enums_pb2

        return {'pywrapcp': pywrapcp, 'routing_enums_pb2': routing_enums_pb2}
    elif solver == 'python_tsp_ls':
        from python_tsp.heuristics import solve_tsp_local_search

        return {'solve_tsp_local_search': solve_tsp_local_search}
    elif solver == 'python_tsp_sa':
        from python_tsp.heuristics import solve_tsp_simulated_annealing

        return {'solve_tsp_simulated_annealing': solve_tsp_simulated_annealing}
    elif solver == 'tsp_solver2':
        from tsp_solver.greedy import solve_tsp

        return {'solve_tsp': solve_tsp}
    elif solver == 'frigidum':
        import frigidum
        from frigidum.examples import tsp

        return {'frigidum': frigidum, 'tsp': tsp}
    elif solver == 'elkai':
        import elkai

        return {'elkai': elkai}

    return {}


def _solve(solver: str, lib: dict, case_file: str, budget: float, n: int, pts, exact, scaled) -> list[int]:
    """Runs one solver and returns its tour (the timed part of a measurement)."""
    if solver == 'fast_tsp':
        tour = lib['fast_tsp'].find_tour(scaled, duration_seconds=budget)
    elif solver == 'fast_tsp_4t':
        tour = lib['fast_tsp'].find_tour(scaled, duration_seconds=budget, max_threads=4)
    elif solver == 'ortools':
        manager = lib['pywrapcp'].RoutingIndexManager(n, 1, 0)
        routing = lib['pywrapcp'].RoutingModel(manager)
        cb = routing.RegisterTransitMatrix(exact.tolist())
        routing.SetArcCostEvaluatorOfAllVehicles(cb)
        params = lib['pywrapcp'].DefaultRoutingSearchParameters()
        params.first_solution_strategy = lib['routing_enums_pb2'].FirstSolutionStrategy.PATH_CHEAPEST_ARC
        params.local_search_metaheuristic = lib['routing_enums_pb2'].LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        params.time_limit.FromMilliseconds(int(budget * 1000))
        sol = routing.SolveWithParameters(params)
        if sol is None:
            raise RuntimeError('OR-Tools found no solution')
        tour, idx = [], routing.Start(0)
        while not routing.IsEnd(idx):
            tour.append(manager.IndexToNode(idx))
            idx = sol.Value(routing.NextVar(idx))
    elif solver == 'python_tsp_ls':
        tour, _ = lib['solve_tsp_local_search'](exact, max_processing_time=budget)
    elif solver == 'python_tsp_sa':
        tour, _ = lib['solve_tsp_simulated_annealing'](exact, max_processing_time=budget)
    elif solver == 'tsp_solver2':
        tour = lib['solve_tsp'](exact, endpoints=(0, 0))
        tour = tour[:-1] if len(tour) == n + 1 else tour
    elif solver == 'frigidum':
        # The configuration of frigidum's bundled TSP example.
        lib['tsp'].nodes = pts
        lib['tsp'].nodes_count = n
        lib['tsp'].dist_eu = exact.astype(float)
        local_opt = lib['frigidum'].sa(
            random_start=lib['tsp'].random_start,
            objective_function=lib['tsp'].objective_function,
            neighbours=[
                lib['tsp'].euclidian_bomb_and_fix,
                lib['tsp'].euclidian_nuke_and_fix,
                lib['tsp'].route_bomb_and_fix,
                lib['tsp'].route_nuke_and_fix,
                lib['tsp'].random_disconnect_vertices_and_fix,
            ],
            copy_state=lib['frigidum'].annealing.naked,
            T_start=5,
            alpha=0.8,
            T_stop=0.001,
            repeats=10**2,
            post_annealing=lib['tsp'].local_search_2opt,
        )
        tour = list(local_opt[0])
    elif solver == 'elkai':
        tour = lib['elkai'].DistanceMatrix(exact.tolist()).solve_tsp(runs=1)
        tour = tour[:-1] if len(tour) == n + 1 else tour
    elif solver in LKH_PARAMS:
        tour = run_lkh(os.environ['LKH_BINARY'], pathlib.Path(case_file).stem, pts, budget, LKH_PARAMS[solver])
    else:
        raise SystemExit(f'unknown solver {solver}')
    return tour


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

    lib = _import_solver(solver)
    start = time.perf_counter()
    tour = _solve(solver, lib, case_file, budget, n, pts, exact, scaled)
    elapsed = time.perf_counter() - start

    t = np.asarray(tour, dtype=np.int64)
    if len(t) == 0 and solver in LKH_PARAMS:
        raise RuntimeError('no tour within the time limit')
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


def run_lkh(binary: str, name: str, pts, budget: float, extra_params: str) -> list[int]:
    """Runs the LKH-3 binary with a wall-clock limit of `budget` seconds and returns its tour.

    The instance is written as a TSPLIB EUC_2D file, so LKH computes the same
    rounded distances as this script. An empty list means LKH wrote no tour
    within the limit (its preprocessing alone can take seconds).
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        work = pathlib.Path(tmp)
        tsp_file, tour_file, par_file = work / f'{name}.tsp', work / f'{name}.tour', work / f'{name}.par'
        with tsp_file.open('w') as f:
            f.write(f'NAME: {name}\nTYPE: TSP\nDIMENSION: {len(pts)}\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n')
            for i, (x, y) in enumerate(pts, 1):
                f.write(f'{i} {x} {y}\n')
            f.write('EOF\n')
        par_file.write_text(
            f'PROBLEM_FILE = {tsp_file}\nTOUR_FILE = {tour_file}\nRUNS = 1000\n'
            f'TIME_LIMIT = {budget}\nTOTAL_TIME_LIMIT = {budget}\nSEED = 1\nTRACE_LEVEL = 0\n' + extra_params
        )
        subprocess.run([binary, str(par_file)], capture_output=True, text=True, timeout=max(60.0, 20 * budget))  # noqa: S603
        if not tour_file.exists():
            return []
        tour, in_tour = [], False
        for line in tour_file.read_text().splitlines():
            line = line.strip()
            if line == 'TOUR_SECTION':
                in_tour = True
            elif in_tour:
                if line in ('-1', 'EOF'):
                    break
                tour.append(int(line) - 1)
        return tour


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
        if r['error'].startswith('killed'):
            return 'killed'
        return f'no tour ({r["elapsed"]:.1f} s)' if 'no tour within' in r['error'] else 'error'
    s = f'{r["gap_pct"]:.2f} %'
    if show_time:
        s += f' ({r["elapsed"]:.1f} s)'
    return s


def median_of(runs: list[dict]) -> dict:
    """One record per cell: the median gap and elapsed time over the successful runs, else the last error."""
    good = [r for r in runs if 'error' not in r]
    if not good:
        return runs[-1]
    med = dict(good[len(good) // 2])
    med['gap_pct'] = statistics.median(r['gap_pct'] for r in good)
    med['elapsed'] = statistics.median(r['elapsed'] for r in good)
    return med


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


def report(recs: list[dict], cases: list[str], budgets: list[float], budgeted=BUDGETED) -> str:
    runs: dict = defaultdict(list)
    n_of = {}
    for r in recs:
        runs[r['solver'], r['case'], r['budget']].append(r)
        if 'n' in r:
            n_of[r['case']] = r['n']
    res: dict = defaultdict(dict)
    for (solver, case, budget), rs in runs.items():
        res[solver, case][budget] = median_of(rs)
    out = []
    for budget in budgets:
        out.append(f'\nTime budget {budget:g} s\n')
        rows = [[f'{c} (n={n_of.get(c, "?")})'] + [cell(res[s, c].get(budget)) for s, _ in budgeted] for c in cases]
        out.append(rst_table(['Instance'] + [label for _, label in budgeted], rows))
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
    ap.add_argument('--lkh', default=None, help='path to an LKH-3 binary; adds the LKH-3 columns')
    ap.add_argument('--repeats', type=int, default=1, help='runs per cell; the tables show the median')
    args = ap.parse_args()
    cases = args.cases.split(',')
    budgets = [float(b) for b in args.budgets.split(',')]
    budgeted = list(BUDGETED)
    if args.lkh:
        os.environ['LKH_BINARY'] = str(pathlib.Path(args.lkh).resolve())
        budgeted = budgeted[:2] + LKH + budgeted[2:]
    recs = []
    with pathlib.Path(args.out).open('a') as out:
        for case in cases:
            for solver, _ in budgeted:
                for budget in budgets:
                    for _ in range(args.repeats):
                        recs.append(measure(solver, case, budget, timeout=3 * budget + 120))
                        out.write(json.dumps(recs[-1]) + '\n')
                        out.flush()
                        print(recs[-1], file=sys.stderr, flush=True)
            for solver, _ in UNBUDGETED:
                recs.append(measure(solver, case, 0.0, timeout=UNBUDGETED_TIMEOUT))
                out.write(json.dumps(recs[-1]) + '\n')
                out.flush()
                print(recs[-1], file=sys.stderr, flush=True)
    print(report(recs, cases, budgets, budgeted))


if __name__ == '__main__':
    main()
