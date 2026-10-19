//! Candidate neighbour lists for 2-opt and 3-opt.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;

use super::{Budget, DistMatrix};

/// A row of the distance matrix.
#[inline(always)]
pub(crate) fn row(dists: &DistMatrix, i: usize) -> &[u16] {
    &dists.data[i * dists.n..(i + 1) * dists.n]
}

/// Inserts `key` into `best`, which holds the (at most `k`) smallest keys
/// seen so far in ascending order.
#[inline(always)]
fn push_best(best: &mut Vec<u64>, k: usize, key: u64) {
    if best.len() == k {
        if key >= best[k - 1] {
            return;
        }
        best.pop();
    }
    let at = best.partition_point(|&b| b < key);
    best.insert(at, key);
}

/// Selecting the k smallest of n values is done in blocks of this many.
const BLOCK: usize = 16;

/// The minimum of each `BLOCK`-sized block of `vals`, into `mins`.
/// Vectorises well.
#[inline]
fn block_minima(vals: &[u16], mins: &mut Vec<u16>) {
    mins.clear();
    mins.extend(
        vals.chunks(BLOCK)
            .map(|block| block.iter().fold(u16::MAX, |m, &v| m.min(v))),
    );
}

/// The k-th smallest block minimum, or `u16::MAX` with fewer than k
/// blocks. At least k values (one per block) are at most this, so it
/// bounds the k-th smallest value, and only blocks whose minimum is within
/// it need a closer look: typically about k of them.
#[inline]
fn kth_smallest(mins: &[u16], k: usize, scratch: &mut Vec<u16>) -> u16 {
    if mins.len() < k {
        return u16::MAX;
    }
    scratch.clear();
    scratch.extend_from_slice(mins);
    *scratch.select_nth_unstable(k - 1).1
}

/// Matrices with fewer entries than this build their neighbour lists on
/// one thread; below it, starting threads costs more than it saves.
const PARALLEL_MIN_ENTRIES: usize = 1 << 21;

/// For each node, its `k` closest other nodes, closest first (ties to the
/// lowest index). Each row is selected with `block_minima` and
/// `kth_smallest`, so almost all the work is a vectorised scan. Large
/// matrices are split into row ranges built on up to `threads` threads.
///
/// Returns `None` if the budget runs out (deadline or interrupt) first.
pub(crate) fn nearest(
    dists: &DistMatrix,
    k: usize,
    budget: &Budget<'_>,
    threads: usize,
) -> Option<Vec<u32>> {
    nearest_split(dists, k, budget, threads, PARALLEL_MIN_ENTRIES)
}

/// [`nearest`], with more than one thread only for matrices of at least
/// `min_entries` entries (and then about one thread per `min_entries`).
fn nearest_split(
    dists: &DistMatrix,
    k: usize,
    budget: &Budget<'_>,
    threads: usize,
    min_entries: usize,
) -> Option<Vec<u32>> {
    let n = dists.len();
    let k = k.min(n.saturating_sub(1));
    if k == 0 {
        return Some(Vec::new());
    }
    let threads = threads
        .min(n.saturating_mul(n).div_ceil(min_entries))
        .max(1);
    let mut data = vec![0; n * k];
    let rows_per_thread = n.div_ceil(threads);
    let mut parts = data.chunks_mut(rows_per_thread * k);
    let first = parts.next().expect("n >= 1");
    let complete = if threads == 1 {
        fill_rows(dists, k, 0, first, budget)
    } else {
        let stop = AtomicBool::new(false);
        let deadline = budget.deadline();
        thread::scope(|scope| {
            let workers: Vec<_> = parts
                .enumerate()
                .map(|(t, out)| {
                    let stop = &stop;
                    scope.spawn(move || {
                        let budget = Budget::worker(deadline, stop);
                        fill_rows(dists, k, (t + 1) * rows_per_thread, out, &budget)
                    })
                })
                .collect();
            // The calling thread takes the first range with the real
            // budget, the only one that can see an interrupt.
            let mine = fill_rows(dists, k, 0, first, budget);
            if !mine {
                stop.store(true, Ordering::Relaxed);
            }
            // `join` only fails if a worker panicked; re-raise it. Join
            // every worker before looking at the results.
            let theirs: Vec<bool> = workers
                .into_iter()
                .map(|w| w.join().unwrap_or_else(|e| std::panic::resume_unwind(e)))
                .collect();
            mine && theirs.into_iter().all(|done| done)
        })
    };
    complete.then_some(data)
}

/// Fills `out` with the `k` nearest neighbours of each row from
/// `first_row` on (`out.len() / k` rows). Checks the budget every 64 rows
/// and returns false if it ran out.
fn fill_rows(
    dists: &DistMatrix,
    k: usize,
    first_row: usize,
    out: &mut [u32],
    budget: &Budget<'_>,
) -> bool {
    let n = dists.len();
    let mut mins: Vec<u16> = Vec::with_capacity(n.div_ceil(BLOCK));
    let mut scratch: Vec<u16> = Vec::with_capacity(n.div_ceil(BLOCK));
    let mut best: Vec<u64> = Vec::with_capacity(k + 1);
    for (r, out) in out.chunks_mut(k).enumerate() {
        if r % 64 == 0 && budget.is_exceeded() {
            return false;
        }
        nearest_row(
            dists,
            k,
            first_row + r,
            out,
            &mut mins,
            &mut scratch,
            &mut best,
        );
    }
    true
}

/// The `k` nearest neighbours of node `i`, into `out`. `mins`, `scratch`
/// and `best` are reusable buffers.
#[inline]
fn nearest_row(
    dists: &DistMatrix,
    k: usize,
    i: usize,
    out: &mut [u32],
    mins: &mut Vec<u16>,
    scratch: &mut Vec<u16>,
    best: &mut Vec<u64>,
) {
    let n = dists.len();
    let r = row(dists, i);
    block_minima(r, mins);
    // The block holding i itself, without the diagonal entry.
    let own = i / BLOCK;
    mins[own] = r[own * BLOCK..n.min(own * BLOCK + BLOCK)]
        .iter()
        .enumerate()
        .filter(|&(o, _)| own * BLOCK + o != i)
        .fold(u16::MAX, |m, (_, &d)| m.min(d));
    let bound = kth_smallest(mins, k, scratch);
    best.clear();
    for (b, &m) in mins.iter().enumerate() {
        if m > bound {
            continue;
        }
        let base = b * BLOCK;
        for (o, &d) in r[base..n.min(base + BLOCK)].iter().enumerate() {
            let j = base + o;
            if d <= bound && j != i {
                push_best(best, k, (u64::from(d) << 32) | j as u64);
            }
        }
    }
    for (o, &key) in out.iter_mut().zip(best.iter()) {
        *o = key as u32;
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::euclidean;
    use super::*;

    /// Random symmetric matrix with entries in 1..=max, so many ties.
    fn tied(n: usize, max: u16, seed: u64) -> DistMatrix {
        let mut rng = fastrand::Rng::with_seed(seed);
        let mut data = vec![0u16; n * n];
        for i in 0..n {
            for j in i + 1..n {
                let v = rng.u16(1..=max);
                data[i * n + j] = v;
                data[j * n + i] = v;
            }
        }
        let rows: Vec<Vec<u16>> = data.chunks(n).map(<[u16]>::to_vec).collect();
        DistMatrix::from_rows(&rows).unwrap()
    }

    #[test]
    fn nearest_matches_brute_force() {
        let cases = [
            (euclidean(2, 1), 16),
            (euclidean(7, 1), 16),
            (euclidean(50, 3), 8),
            (euclidean(203, 5), 16),
            (tied(97, 3, 2), 10),
            (tied(160, 65535, 4), 20),
        ];
        for (d, k) in cases {
            let n = d.len();
            let k = k.min(n - 1);
            let expected: Vec<u32> = (0..n)
                .flat_map(|i| {
                    let mut all: Vec<(i64, u32)> = (0..n)
                        .filter(|&j| j != i)
                        .map(|j| (d.get(i, j), j as u32))
                        .collect();
                    all.sort_unstable();
                    all.truncate(k);
                    all.into_iter().map(|t| t.1)
                })
                .collect();
            assert_eq!(nearest(&d, k, &Budget::unlimited(), 1).unwrap(), expected);
            // Rows split between threads, including uneven and empty splits.
            for threads in [2, 3, 4, 6] {
                let near = nearest_split(&d, k, &Budget::unlimited(), threads, 1).unwrap();
                assert_eq!(near, expected, "n={n}, {threads} threads");
            }
        }
    }

    #[test]
    fn nearest_splits_large_matrices() {
        // Large enough for `nearest` itself to use several threads.
        let d = euclidean(1500, 6);
        let one = nearest(&d, 20, &Budget::unlimited(), 1).unwrap();
        assert_eq!(nearest(&d, 20, &Budget::unlimited(), 4).unwrap(), one);
    }

    #[test]
    fn nearest_stops_when_budget_is_exceeded() {
        let d = euclidean(300, 2);
        let start = std::time::Instant::now() - std::time::Duration::from_secs(2);
        let past = Budget::new(start, 1.0).unwrap();
        assert!(nearest(&d, 20, &past, 1).is_none());
        assert!(nearest_split(&d, 20, &past, 4, 1).is_none());
    }
}
