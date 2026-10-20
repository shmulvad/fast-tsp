//! PyO3 bindings: the `fast_tsp._core` extension module.

use std::cell::RefCell;
use std::time::Instant;

use pyo3::exceptions::{PyMemoryError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use crate::tsp::{self, Budget, Candidates, DistMatrix, SearchOptions, TspError};

fn to_py_err(err: TspError) -> PyErr {
    match err {
        TspError::Invalid(msg) => PyValueError::new_err(msg),
        TspError::OutOfMemory(msg) => PyMemoryError::new_err(msg),
        // Normally replaced by the KeyboardInterrupt (or other error) raised
        // by the signal handler; see `interruptible`.
        TspError::Interrupted => PyRuntimeError::new_err("Solver was interrupted"),
    }
}

fn to_distance(v: impl TryInto<u16> + Copy + std::fmt::Display) -> PyResult<u16> {
    v.try_into().map_err(|_| {
        PyValueError::new_err(format!(
            "All distances must be between 0 and {}, but found {v}",
            u16::MAX
        ))
    })
}

/// Reads a distance matrix from a 2D integer numpy array or any sequence of
/// integer sequences (e.g. a list of lists).
fn extract_dists(obj: &Bound<'_, PyAny>) -> PyResult<DistMatrix> {
    #[cfg(feature = "numpy")]
    if let Some(dists) = numpy_input::extract(obj)? {
        return Ok(dists);
    }
    extract_nested_sequence(obj)
}

/// Reads rows one at a time. Each row's length is checked before the full
/// `n` x `n` buffer is allocated, so a long non-square input fails fast.
fn extract_nested_sequence(obj: &Bound<'_, PyAny>) -> PyResult<DistMatrix> {
    let n = obj.len()?;
    let mut data = Vec::new();
    for (i, row) in obj.try_iter()?.enumerate() {
        let row: Vec<i64> = row?.extract()?;
        if row.len() != n {
            return Err(to_py_err(DistMatrix::not_square_error(n, row.len())));
        }
        if i == 0 {
            data = DistMatrix::alloc_buffer(n).map_err(to_py_err)?;
        }
        for v in row {
            data.push(to_distance(v)?);
        }
    }
    DistMatrix::from_flat(n, data).map_err(to_py_err)
}

#[cfg(feature = "numpy")]
mod numpy_input {
    use numpy::ndarray::ArrayView2;
    use numpy::{PyArray2, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods};
    use pyo3::exceptions::PyTypeError;
    use pyo3::prelude::*;
    use pyo3::types::{PyDict, PyModule};

    use super::{to_distance, to_py_err};
    use crate::tsp::DistMatrix;

    /// Returns `Ok(None)` if `obj` is not a numpy array, so the caller can
    /// fall back to the generic sequence path.
    pub fn extract(obj: &Bound<'_, PyAny>) -> PyResult<Option<DistMatrix>> {
        // The numpy crate imports numpy's C API on first use and panics if
        // that fails. An ndarray can only exist once numpy has been imported,
        // so bail out early unless numpy is loaded and importable.
        let py = obj.py();
        let modules = py.import("sys")?.getattr("modules")?;
        let numpy_loaded = modules
            .cast::<PyDict>()?
            .get_item("numpy")?
            .is_some_and(|m| m.is_instance_of::<PyModule>());
        if !numpy_loaded || numpy::get_array_module(py).is_err() {
            return Ok(None);
        }
        let Ok(array) = obj.cast::<PyUntypedArray>() else {
            return Ok(None);
        };
        if array.ndim() != 2 {
            return Err(PyTypeError::new_err(format!(
                "Distance matrix must be 2D, but got {} dimension(s)",
                array.ndim()
            )));
        }
        let (n, m) = (array.shape()[0], array.shape()[1]);
        if n != m {
            return Err(to_py_err(DistMatrix::not_square_error(n, m)));
        }

        macro_rules! try_dtypes {
            ($($t:ty),*) => {$(
                if let Ok(typed) = obj.cast::<PyArray2<$t>>() {
                    return convert(typed.readonly().as_array(), n).map(Some);
                }
            )*};
        }
        try_dtypes!(i64, i32, u16, u8, i16, i8, u32, u64);

        Err(PyTypeError::new_err(format!(
            "Distance matrix must contain integers, but has dtype {}",
            array.dtype()
        )))
    }

    fn convert<T>(view: ArrayView2<'_, T>, n: usize) -> PyResult<DistMatrix>
    where
        T: TryInto<u16> + Copy + std::fmt::Display,
    {
        let mut data = DistMatrix::alloc_buffer(n).map_err(to_py_err)?;
        // `iter` walks in logical row-major order whatever the memory layout.
        for &v in view.iter() {
            data.push(to_distance(v)?);
        }
        DistMatrix::from_flat(n, data).map_err(to_py_err)
    }
}

/// Runs `f` with the GIL released. `f` gets a callback that briefly retakes
/// the GIL to run Python's signal handlers, so Ctrl-C can stop the solver.
/// If a handler raised (normally KeyboardInterrupt), that error is returned.
fn interruptible<T: Send>(
    py: Python<'_>,
    f: impl FnOnce(&dyn Fn() -> bool) -> tsp::Result<T> + Send,
) -> PyResult<T> {
    let (result, signal_error) = py.detach(|| {
        let signal_error = RefCell::new(None);
        let check_signals = || {
            Python::attach(|py| match py.check_signals() {
                Ok(()) => false,
                Err(err) => {
                    *signal_error.borrow_mut() = Some(err);
                    true
                }
            })
        };
        let result = f(&check_signals);
        (result, signal_error.into_inner())
    });
    match signal_error {
        Some(err) => Err(err),
        None => result.map_err(to_py_err),
    }
}

/// The number of search threads for `max_threads`: None means one fewer
/// than the available cores (at least one), and any value is capped at the
/// number of cores, since more threads than cores cannot help.
fn resolve_threads(max_threads: Option<i64>) -> PyResult<usize> {
    let cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
    match max_threads {
        None => Ok(cores.saturating_sub(1).max(1)),
        Some(t) if t >= 1 => Ok(usize::try_from(t).unwrap_or(usize::MAX).min(cores)),
        Some(t) => Err(PyValueError::new_err(format!(
            "max_threads must be a positive integer or None, but got {t}"
        ))),
    }
}

#[pyo3::pymodule]
mod _core {
    use super::*;

    #[pymodule_init]
    fn init(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add("__version__", env!("CARGO_PKG_VERSION"))
    }

    /// Find a good TSP tour.
    ///
    /// Run a local solver to find a near-optimal TSP tour. For small
    /// problems, the exact solution is guaranteed to be returned.
    /// `max_threads` threads search together (None: one fewer than the
    /// number of available CPU cores), capped at the number of cores.
    /// `candidates` is 'auto', 'nearest' or 'alpha'. With
    /// `stop_when_converged` the search may return before the deadline.
    #[pyfunction]
    #[pyo3(signature = (dists, duration_seconds = 2.0, max_threads = Some(1), candidates = "auto", stop_when_converged = false))]
    fn find_tour(
        dists: &Bound<'_, PyAny>,
        duration_seconds: f64,
        max_threads: Option<i64>,
        candidates: &str,
        stop_when_converged: bool,
    ) -> PyResult<Vec<usize>> {
        // Time spent reading the input counts towards the duration.
        let start = Instant::now();
        let threads = resolve_threads(max_threads)?;
        let candidates: Candidates = candidates.parse().map_err(PyValueError::new_err)?;
        let options = SearchOptions {
            candidates,
            stop_when_converged,
        };
        let py = dists.py();
        let dists = extract_dists(dists)?;
        interruptible(py, |check_signals| {
            let budget = Budget::new(start, duration_seconds)?.with_interrupt(check_signals);
            tsp::find_tour_with(&dists, &budget, threads, None, &options)
        })
    }

    /// Determine whether a tour is a valid TSP tour.
    ///
    /// Returns True if the tour visits each of the nodes 0..n-1 exactly once.
    #[pyfunction]
    fn is_valid_tour(n: i64, tour: Vec<i64>) -> bool {
        let Ok(n) = usize::try_from(n) else {
            return false;
        };
        tsp::tour_from_ints(&tour, n).is_ok_and(|tour| tsp::is_valid_tour(n, &tour))
    }

    /// Compute the cost of a TSP tour.
    ///
    /// Computes the cost of a TSP tour, including the edge back to the start.
    #[pyfunction]
    fn compute_cost(tour: Vec<i64>, dists: &Bound<'_, PyAny>) -> PyResult<u64> {
        tsp::compute_cost(&tour, &extract_dists(dists)?).map_err(to_py_err)
    }

    /// Solve TSP using greedy nearest neighbor heuristic.
    ///
    /// Solve TSP using the greedy nearest neighbor heuristic. This is a very
    /// rough approximation to the exact solution. The preferred way is to
    /// use `find_tour`.
    #[pyfunction]
    fn greedy_nearest_neighbor(dists: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let py = dists.py();
        let dists = extract_dists(dists)?;
        py.detach(|| tsp::greedy_nearest_neighbor(&dists))
            .map_err(to_py_err)
    }

    /// Solve TSP exactly using dynamic programming. Should not be used for
    /// n much more than ~20.
    ///
    /// Solves TSP optimally using the bottom-up Held-Karp's algorithm in
    /// O(2^n * n^2) time. This is tractable for small n but quickly becomes
    /// untractable for n even of medium size.
    #[pyfunction]
    fn solve_tsp_exact(dists: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
        let py = dists.py();
        let dists = extract_dists(dists)?;
        interruptible(py, |check_signals| {
            tsp::solve_tsp_exact(&dists, &Budget::unlimited().with_interrupt(check_signals))
        })
    }

    /// Return the score of a tour, where 1 is better and 0 is worst.
    ///
    /// Returns a double in the range 0..1, where 0 denotes a solution that
    /// is worse than or equivalent to the naive solution and 1 denotes a
    /// solution that is equivalent to the optimal solution.
    #[pyfunction]
    fn score_tour(tour: Vec<i64>, dists: &Bound<'_, PyAny>, opt_cost: i64) -> PyResult<f64> {
        tsp::score_tour(&tour, &extract_dists(dists)?, opt_cost).map_err(to_py_err)
    }
}
