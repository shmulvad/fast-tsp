//! Pure-Rust TSP solver core. Has no Python dependencies so it can be unit
//! tested with plain `cargo test`.

mod exact;
mod greedy;
mod local_search;

use std::cell::Cell;
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

pub use exact::{MAX_EXACT_NODES, solve_tsp_exact};
pub use greedy::greedy_nearest_neighbor;

/// Problems up to this size are solved exactly in `find_tour`.
pub const EXACT_SOLUTION_THRESHOLD: usize = 20;

pub type Tour = Vec<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TspError {
    /// Bad input. Maps to Python's `ValueError`.
    Invalid(String),
    /// Could not allocate enough memory. Maps to Python's `MemoryError`.
    OutOfMemory(String),
    /// The interrupt callback asked the solver to stop (e.g. Ctrl-C).
    Interrupted,
}

impl fmt::Display for TspError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TspError::Invalid(msg) | TspError::OutOfMemory(msg) => f.write_str(msg),
            TspError::Interrupted => f.write_str("Interrupted"),
        }
    }
}

impl std::error::Error for TspError {}

pub type Result<T> = std::result::Result<T, TspError>;

fn invalid<T>(msg: impl Into<String>) -> Result<T> {
    Err(TspError::Invalid(msg.into()))
}

/// Square distance matrix stored row-major in one flat buffer.
#[derive(Debug, Clone)]
pub struct DistMatrix {
    n: usize,
    data: Vec<u16>,
}

impl DistMatrix {
    pub fn not_square_error(n: usize, m: usize) -> TspError {
        TspError::Invalid(format!("Distance matrix must be square but got ({n}, {m})"))
    }

    /// Allocates an empty buffer with room for an `n` x `n` matrix. Fails
    /// cleanly instead of aborting when the size overflows or the
    /// allocation is refused.
    pub fn alloc_buffer(n: usize) -> Result<Vec<u16>> {
        let out_of_memory =
            || TspError::OutOfMemory(format!("Not enough memory for a {n} x {n} distance matrix"));
        let size = n.checked_mul(n).ok_or_else(out_of_memory)?;
        let mut data = Vec::new();
        data.try_reserve_exact(size).map_err(|_| out_of_memory())?;
        Ok(data)
    }

    /// Wraps a row-major buffer of `n * n` distances.
    pub fn from_flat(n: usize, data: Vec<u16>) -> Result<Self> {
        if n.checked_mul(n) != Some(data.len()) {
            return invalid(format!(
                "Expected {n} x {n} distances, but got {}",
                data.len()
            ));
        }
        Ok(Self { n, data })
    }

    /// Builds a matrix from rows, checking that every row has length `n`
    /// before allocating anything.
    pub fn from_rows(rows: &[Vec<u16>]) -> Result<Self> {
        let n = rows.len();
        if let Some(row) = rows.iter().find(|row| row.len() != n) {
            return Err(Self::not_square_error(n, row.len()));
        }
        let mut data = Self::alloc_buffer(n)?;
        for row in rows {
            data.extend_from_slice(row);
        }
        Self::from_flat(n, data)
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    #[inline(always)]
    pub fn get(&self, i: usize, j: usize) -> i64 {
        i64::from(self.data[i * self.n + j])
    }
}

/// How often the interrupt callback is polled. It may be expensive (the
/// Python bindings briefly take the GIL), so it is not called on every check.
const INTERRUPT_POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Limits on how long the solver may run: an optional deadline, plus an
/// optional callback that returns true when the solver should stop early.
///
/// A budget belongs to one thread. Worker threads get their own budget from
/// [`Budget::worker`], which shares the deadline and watches a stop flag
/// that the owning thread sets when its callback fires.
pub struct Budget<'a> {
    deadline: Option<Instant>,
    interrupt: Option<&'a dyn Fn() -> bool>,
    stop: Option<&'a AtomicBool>,
    next_poll: Cell<Instant>,
    interrupted: Cell<bool>,
}

impl<'a> Budget<'a> {
    /// No deadline and no interrupt callback.
    pub fn unlimited() -> Self {
        Self {
            deadline: None,
            interrupt: None,
            stop: None,
            next_poll: Cell::new(Instant::now()),
            interrupted: Cell::new(false),
        }
    }

    /// A budget ending `duration_seconds` after `start`. Durations too large
    /// to represent (including infinity) mean no deadline.
    pub fn new(start: Instant, duration_seconds: f64) -> Result<Self> {
        // Also rejects NaN.
        if duration_seconds.is_nan() || duration_seconds <= 0.0 {
            return invalid("Duration must be strictly positive");
        }
        let deadline = Duration::try_from_secs_f64(duration_seconds)
            .ok()
            .and_then(|d| start.checked_add(d));
        Ok(Self {
            deadline,
            ..Self::unlimited()
        })
    }

    pub fn with_interrupt(mut self, interrupt: &'a dyn Fn() -> bool) -> Self {
        self.interrupt = Some(interrupt);
        self
    }

    pub(crate) fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// A budget for a worker thread: the given deadline, no callback, and
    /// exceeded as soon as `stop` is set. Built inside the worker thread,
    /// since a budget with a callback cannot be sent between threads.
    pub(crate) fn worker(deadline: Option<Instant>, stop: &'a AtomicBool) -> Self {
        Self {
            deadline,
            stop: Some(stop),
            ..Self::unlimited()
        }
    }

    fn poll_interrupt(&self, now: Instant) -> bool {
        if !self.interrupted.get()
            && let Some(interrupt) = self.interrupt
            && now >= self.next_poll.get()
        {
            self.next_poll.set(now + INTERRUPT_POLL_INTERVAL);
            if interrupt() {
                self.interrupted.set(true);
            }
        }
        self.interrupted.get()
    }

    /// True once the interrupt callback has asked to stop. Ignores the deadline.
    pub(crate) fn is_interrupted(&self) -> bool {
        self.poll_interrupt(Instant::now())
    }

    /// True once the deadline has passed or the solver was interrupted.
    pub(crate) fn is_exceeded(&self) -> bool {
        if self.stop.is_some_and(|stop| stop.load(Ordering::Relaxed)) {
            return true;
        }
        let now = Instant::now();
        self.poll_interrupt(now) || self.deadline.is_some_and(|d| now > d)
    }

    fn check_interrupted(&self) -> Result<()> {
        if self.interrupted.get() {
            Err(TspError::Interrupted)
        } else {
            Ok(())
        }
    }
}

/// Finds a good tour. Small problems are solved exactly. Larger ones use
/// local search on `threads` threads (at least one) until the budget runs
/// out or each thread has gone 5000 perturbations without improving the
/// best tour.
pub fn find_tour(dists: &DistMatrix, budget: &Budget, threads: usize) -> Result<Tour> {
    if dists.len() < 2 {
        return invalid("Need at least 2 nodes");
    }
    if dists.len() <= EXACT_SOLUTION_THRESHOLD {
        return solve_tsp_exact(dists, budget);
    }
    let tour = local_search::local_search(dists, budget, threads);
    budget.check_interrupted()?;
    Ok(tour)
}

/// True if `tour` visits each of the nodes `0..n` exactly once.
pub fn is_valid_tour(n: usize, tour: &[usize]) -> bool {
    if tour.len() != n {
        return false;
    }
    let mut visited = vec![false; n];
    for &v in tour {
        if v >= n || visited[v] {
            return false;
        }
        visited[v] = true;
    }
    true
}

/// Converts user-supplied node ids, rejecting any outside `0..n`.
pub fn tour_from_ints(tour: &[i64], n: usize) -> Result<Tour> {
    tour.iter()
        .map(|&v| match usize::try_from(v) {
            Ok(v) if v < n => Ok(v),
            _ => invalid(format!(
                "Tour contains node {v}, but the distance matrix only has {n} nodes"
            )),
        })
        .collect()
}

/// Cost of the closed tour, including the edge back to the start.
pub(crate) fn tour_cost(dists: &DistMatrix, tour: &[usize]) -> u64 {
    let Some((&first, &last)) = tour.first().zip(tour.last()) else {
        return 0;
    };
    let path: i64 = tour.windows(2).map(|w| dists.get(w[0], w[1])).sum();
    (path + dists.get(last, first)) as u64
}

pub fn compute_cost(tour: &[i64], dists: &DistMatrix) -> Result<u64> {
    Ok(tour_cost(dists, &tour_from_ints(tour, dists.len())?))
}

/// Score in 0..1: 1 matches the optimal cost, 0 is no better than the
/// greedy nearest-neighbour tour.
pub fn score(cost: i64, opt_cost: i64, naive_cost: i64) -> Result<f64> {
    if cost < opt_cost {
        return invalid(format!(
            "Received cost < opt_cost, which cannot be true: {cost} < {opt_cost}"
        ));
    }
    if naive_cost == opt_cost {
        return Ok(if cost == opt_cost { 1.0 } else { 0.0 });
    }
    let relative_diff = (cost - opt_cost) as f64 / (naive_cost - opt_cost) as f64;
    Ok(0.02_f64.powf(relative_diff.max(0.0)))
}

pub fn score_tour(tour: &[i64], dists: &DistMatrix, opt_cost: i64) -> Result<f64> {
    let cost = compute_cost(tour, dists)?;
    let greedy_cost = tour_cost(dists, &greedy_nearest_neighbor(dists)?);
    score(cost as i64, opt_cost, greedy_cost as i64)
}

#[cfg(test)]
pub(crate) mod test_utils {
    use super::DistMatrix;

    pub fn matrix(rows: &[Vec<u16>]) -> DistMatrix {
        DistMatrix::from_rows(rows).unwrap()
    }

    /// Random Euclidean instance with integer distances, seeded for repeatability.
    pub fn euclidean(n: usize, seed: u64) -> DistMatrix {
        matrix(&euclidean_rows(n, seed))
    }

    pub fn euclidean_rows(n: usize, seed: u64) -> Vec<Vec<u16>> {
        let mut rng = fastrand::Rng::with_seed(seed);
        let pts: Vec<(f64, f64)> = (0..n)
            .map(|_| (rng.f64() * 1000.0, rng.f64() * 1000.0))
            .collect();
        pts.iter()
            .map(|a| {
                pts.iter()
                    .map(|b| ((a.0 - b.0).hypot(a.1 - b.1)).round() as u16)
                    .collect()
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::test_utils::{euclidean, matrix};
    use super::*;

    fn run(d: &DistMatrix, secs: f64) -> Result<Tour> {
        find_tour(d, &Budget::new(Instant::now(), secs)?, 1)
    }

    #[test]
    fn rejects_bad_input() {
        let ok = matrix(&[vec![0, 1], vec![1, 0]]);
        assert_eq!(run(&ok, 0.0), invalid("Duration must be strictly positive"));
        assert_eq!(
            run(&ok, f64::NAN),
            invalid("Duration must be strictly positive")
        );
        assert_eq!(
            run(&matrix(&[vec![0]]), 1.0),
            invalid("Need at least 2 nodes")
        );
        assert_eq!(
            DistMatrix::from_rows(&[vec![0, 1], vec![2, 0], vec![3, 0]]).unwrap_err(),
            DistMatrix::not_square_error(3, 2)
        );
        assert_eq!(
            DistMatrix::from_rows(&[vec![0, 1, 2], vec![1, 0], vec![2, 3, 0]]).unwrap_err(),
            DistMatrix::not_square_error(3, 2)
        );
        assert!(DistMatrix::from_flat(2, vec![0; 3]).is_err());
        assert!(compute_cost(&[0, 5], &ok).is_err());
        assert!(compute_cost(&[0, -1], &ok).is_err());
    }

    #[test]
    fn tall_input_fails_without_allocating() {
        // 100_000 rows of length 2 would need 20 GB if allocated up front.
        let rows = vec![vec![0u16; 2]; 100_000];
        assert_eq!(
            DistMatrix::from_rows(&rows).unwrap_err(),
            DistMatrix::not_square_error(100_000, 2)
        );
    }

    #[test]
    fn huge_matrix_is_out_of_memory_not_abort() {
        assert!(matches!(
            DistMatrix::alloc_buffer(usize::MAX / 2),
            Err(TspError::OutOfMemory(_))
        ));
        assert!(matches!(
            DistMatrix::alloc_buffer(1 << 40),
            Err(TspError::OutOfMemory(_))
        ));
    }

    #[test]
    fn basic_tour() {
        let d = matrix(&[vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]]);
        assert_eq!(run(&d, 2.0).unwrap(), vec![0, 1, 2]);
    }

    #[test]
    fn infinite_duration_terminates() {
        let tour = run(&euclidean(30, 1), f64::INFINITY).unwrap();
        assert!(is_valid_tour(30, &tour));
    }

    #[test]
    fn budget_handles_huge_durations() {
        let start = Instant::now();
        assert!(
            Budget::new(start, f64::INFINITY)
                .unwrap()
                .deadline
                .is_none()
        );
        assert!(Budget::new(start, 1e300).unwrap().deadline.is_none());
        assert!(!Budget::new(start, 60.0).unwrap().is_exceeded());
        let past = start - Duration::from_secs(2);
        assert!(Budget::new(past, 1.0).unwrap().is_exceeded());
    }

    #[test]
    fn interrupt_stops_local_search_and_exact() {
        let calls = Cell::new(0);
        let stop = || {
            calls.set(calls.get() + 1);
            true
        };
        let budget = Budget::unlimited().with_interrupt(&stop);
        assert_eq!(
            find_tour(&euclidean(200, 3), &budget, 1),
            Err(TspError::Interrupted)
        );
        assert_eq!(calls.get(), 1, "callback is not called again once it fired");

        let budget = Budget::unlimited().with_interrupt(&stop);
        assert_eq!(
            find_tour(&euclidean(200, 3), &budget, 4),
            Err(TspError::Interrupted)
        );

        let budget = Budget::unlimited().with_interrupt(&stop);
        assert_eq!(
            solve_tsp_exact(&euclidean(18, 3), &budget),
            Err(TspError::Interrupted)
        );

        let never = || false;
        let budget = Budget::new(Instant::now(), 0.2)
            .unwrap()
            .with_interrupt(&never);
        assert!(find_tour(&euclidean(200, 3), &budget, 1).is_ok());
    }

    #[test]
    fn interrupt_stops_all_threads() {
        // The callback fires after ~0.3 s; without it the search would run
        // for an hour. All worker threads must notice and stop.
        let start = Instant::now();
        let stop = || start.elapsed() > Duration::from_millis(300);
        let budget = Budget::new(start, 3600.0).unwrap().with_interrupt(&stop);
        assert_eq!(
            find_tour(&euclidean(3000, 5), &budget, 4),
            Err(TspError::Interrupted)
        );
        assert!(start.elapsed() < Duration::from_secs(10));
    }

    #[test]
    fn parallel_search_respects_deadline_and_is_valid() {
        for &n in &[21, 50, 300] {
            let d = euclidean(n, 7 + n as u64);
            let start = Instant::now();
            let tour = find_tour(&d, &Budget::new(start, 0.3).unwrap(), 4).unwrap();
            assert!(start.elapsed() < Duration::from_secs(2));
            assert!(is_valid_tour(n, &tour));
            let greedy = greedy_nearest_neighbor(&d).unwrap();
            assert!(tour_cost(&d, &tour) <= tour_cost(&d, &greedy));
        }
    }

    #[test]
    fn cost_and_validity() {
        let d = matrix(&[vec![0, 1, 2], vec![1, 0, 3], vec![2, 3, 0]]);
        assert_eq!(compute_cost(&[0, 1, 2], &d).unwrap(), 6);
        assert_eq!(compute_cost(&[], &d).unwrap(), 0);
        assert!(is_valid_tour(3, &[2, 0, 1]));
        assert!(!is_valid_tour(3, &[0, 0, 1]));
        assert!(!is_valid_tour(3, &[0, 1]));
        assert!(!is_valid_tour(3, &[0, 1, 3]));
    }

    #[test]
    fn scoring() {
        assert_eq!(score(10, 10, 20).unwrap(), 1.0);
        assert!((score(20, 10, 20).unwrap() - 0.02).abs() < 1e-12);
        assert_eq!(score(10, 10, 10).unwrap(), 1.0);
        assert!(score(5, 10, 20).is_err());
    }

    #[test]
    fn local_search_gives_valid_and_reasonable_tours() {
        for &n in &[21, 50, 200] {
            let d = euclidean(n, n as u64);
            let tour = run(&d, 0.5).unwrap();
            assert!(is_valid_tour(n, &tour));
            let greedy = greedy_nearest_neighbor(&d).unwrap();
            assert!(tour_cost(&d, &tour) <= tour_cost(&d, &greedy));
        }
    }
}
