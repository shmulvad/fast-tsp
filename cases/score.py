"""
This program tests your solutions for the various problems on the
test data provided in the subfolder 'tests'.

The program can be run by

    (2) Command line:

        > run_tests.py prog.cpp
        > run_tests.py prog.py

The optional argument --verbose=<value> determines the amount of output:

    -1 : Only print statistics on number of tests passed
    0 : Print names of failed tests
    1 : Print names of all tests
    2 : Print names of all tests, and details for failed tests (default)
    3 : Print details for all tests

On failed tests the script by default continues with the next test.
To abort after first failed test use --abort. E.g.

    > python run_tests.py tsp.cpp --verbose=3 --abort
"""
import re
import subprocess
from glob import glob
import os
import sys
from tempfile import TemporaryFile

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


DEFAULT_FILES = None
TIMEOUT = 2  # seconds before test killed
TEST_FOLDER = 'tests/cases/'
TEST_LOG_FILE = 'run_tests.log'
DEFAULT_VERBOSE = 2
DEFAULT_ABORT = False
NUM_KATTIS_TESTS = 50

IS_TSP = True  # Set to True for TSP, False for MWVC

GCC_ARGS = ['g++-11', '-std=c++17']  # The gcc compiler args for cpp programs
PYTHON = sys.executable  # The Python interpreter for running Python programs

INF = float('inf')
log_indent = 0  # current indent for terminal log messages


def coords_to_dist_matrix(coords: np.ndarray) -> np.ndarray:
    return np.array(np.rint(euclidean_distances(coords)), dtype=np.int64)


def check_tsp_answer(answer, problem_input):
    try:
        coords = []
        problem_input_splitted = problem_input.split('\n')
        n = int(problem_input_splitted[0].strip())
        for line in problem_input_splitted[1:]:
            line_stripped = line.strip()
            if line_stripped:
                x, y = map(float, line.strip().split(' '))
                coords.append([x, y])
        coords = np.array(coords)
        dist_matrix = coords_to_dist_matrix(coords)
        tour = [int(val) for val in answer if val.strip()]
        is_correct_len = len(tour) == len(set(tour)) == n
        is_correct_bounds = min(tour) == 0 and max(tour) == n - 1
        is_valid = is_correct_len and is_correct_bounds
        if not is_valid:
            return False, INF, 'WA (Not a valid TSP tour)'

        cost = sum([dist_matrix[tour[i], tour[i1]]
                    for i, i1 in zip(range(-1, n-1), range(n))])
        return is_valid, cost, ''
    except Exception:
        return False, INF, 'WA (Error decoding output)'


def check_vc_answer(answer, problem_input):
    try:
        answer_vc = set(map(int, answer[1].strip().split()))
        problem_input_splitted = problem_input.split('\n')
        weigths = list(map(int, problem_input_splitted[1].strip().split(' ')))
        for line in problem_input_splitted[2:]:
            line_stripped = line.strip()
            if line_stripped:
                u, v = map(int, line.strip().split(' '))
                if not (u in answer_vc or v in answer_vc):
                    return False, INF, 'WA (Not a valid VC)'
        cost = sum(weigths[v] for v in answer_vc)
        return True, cost, ''
    except Exception:
        return False, INF, 'WA (Error decoding output)'


def purify(lines):
    '''Remove redundant white spaces from list of lines.'''
    lines = [re.sub(r'\s+', ' ', line).strip() for line in lines]
    lines = [line for line in lines if line]
    return lines


def load_text(file):
    '''Load & purify text file'''
    if isinstance(file, str):
        with open(file) as f:
            return purify(f.readlines())
    else:
        return purify(file.readlines())


def log(title='', lines=[], end=None, indent=None):
    '''Print log message to screen in below format:

       title
       > lines[0]
       > lines[1]
       > ...

       indent = indentation for this and successive calls to log.
    '''
    global log_indent
    if indent is not None:
        log_indent += indent
    prefix = ' ' * log_indent

    if type(lines) == str:
        lines = [lines]
    if title:
        print(prefix + title, end=end, flush=True)
    for line in lines:
        print(prefix + '> ' + line, end=end, flush=True)


def run_test(program, test_in, verbose, timeout=TIMEOUT, is_python=False):
    '''Run program with test_in as input. Returns True if and only if
       the generated output is test_output, and the execution did not
       generate any errors.

       verbose determines the amount of printed log.
    '''
    if verbose >= 1:
        log(f'{program} {test_in} ', end='')

    # Excecute program in subprocess
    opt, naive = None, None
    exception = ''

    with TemporaryFile('w+') as answer_file, \
         TemporaryFile('w+') as error_file, \
         open(test_in, 'r') as in_file:
        try:
            lines = in_file.readlines()
            opt, naive = map(int, lines[0].strip().split())
            problem_input = ''.join(line.strip() + '\n' for line in lines[1:])
            cmd = [PYTHON, program] if is_python else [f'./{program}']
            subprocess.run(
                cmd,
                input=problem_input,
                stdout=answer_file,
                stderr=error_file,
                timeout=timeout,
                text=True
            )
        except Exception as e:
            exception = str(e)

        answer_file.seek(0)
        error_file.seek(0)

        answer = load_text(answer_file)
        error = load_text(error_file)

        if exception:
            error.append(exception)

    ok = not error
    cost = INF
    msg = None
    check_func = check_tsp_answer if IS_TSP else check_vc_answer
    if ok:
        ok, cost, msg = check_func(answer, problem_input)

    x = INF
    if naive == opt:
        x = 0 if cost == opt else INF
    else:
        x = (cost - opt) / (naive - opt)
    score = 0.02**x

    if ok and verbose > 0:
        log(f'[ok] - {score:.3f}')
    if not ok and verbose >= 0:
        log(f'[failed] {msg}')
    if not ok and verbose >= 2:
        if not answer:
            log(lines='[no answer]')
        if error:
            log('Error', error, msg)

    return ok, score


def test_program(program, verbose):
    '''Given a program name, runs program on all test data in
       TEST_FOLDER/.... ending on .txt

       Returns (#tests passed, #tests performed, total_score).

       verbose = see run_test.
    '''
    is_python = False
    file_ending = ''
    if program.endswith('.py'):
        program = program[:-3]
        file_ending = '.py'
        is_python = True

    if program.endswith('.cpp'):
        print(f'Compiling {program}...')
        try:
            program_no_ext = program[:-4]
            program = program_no_ext
            file_ending = '.o'
            cmd = [*GCC_ARGS, '-o', program + '.o', program + '.cpp']
            ret_val = subprocess.call(cmd, text=True)
            if ret_val != 0:
                exit()
        except Exception as e:
            print(e)
            exit()

    in_files = [file
                for file in glob(TEST_FOLDER + '/*.txt*')
                if re.search(r'[.]txt[0-9]*$', file)]
    in_files.sort()

    passed = 0
    total_score = 0
    for test_in in in_files:
        result, score = run_test(program + file_ending, test_in, verbose=verbose, is_python=is_python)
        with open(TEST_LOG_FILE, 'a') as log_file:
            print(test_in, result, score, file=log_file)

        if result:
            passed += 1
            total_score += score
        elif on_fail_abort:
            raise StopIteration

    return passed, len(in_files), total_score


def test_all(programs=None, verbose=1):
    '''Runs test_program on all files in programs.

       Prints statistics on the number of passed tests.

       verbose = see run_test.
    '''
    passed, tests, score = test_program(programs[0], verbose=verbose)

    # Print statistics
    log('\nTests passed (score):\n')
    log(indent=2)
    score_kattis = score * NUM_KATTIS_TESTS / tests
    out = f'{passed}/{tests} ({score_kattis:.2f}/{NUM_KATTIS_TESTS})'
    log('-' * len(out))
    log(out)
    log('=' * len(out))
    log(indent=-2)

######################################################################


if __name__ == '__main__':
    args = sys.argv[1:]
    verbose = DEFAULT_VERBOSE
    on_fail_abort = DEFAULT_ABORT

    # identify --verbose=... command line argument
    for arg in args:
        if arg.startswith('--'):
            if re.match('--verbose=[-]?[0-9]+$', arg):
                verbose = int(arg.split('=')[1])
            elif arg == '--abort':
                on_fail_abort = True
            else:
                raise Exception('unknown option ' + arg)

    args = [arg for arg in args if not arg.startswith('--')]

    # identify files
    if args:
        files = args
    elif DEFAULT_FILES is not None:
        files = DEFAULT_FILES
    else:
        files = sorted(glob('*.py'))
        if '__file__' in globals():
            this_program = os.path.basename(__file__)
            if this_program in files:
                files.remove(this_program)

    # run the tests
    try:
        test_all(files, verbose=verbose)
    except StopIteration:  # raised on first fail if on_fail_abort == True
        pass

    if len(sys.argv) <= 1:  # potentially run by clicking
        input('\nPress [Enter] to exit')
