//! Stochastic local search: repeated perturbation (double bridge plus a
//! local shuffle) followed by 2-opt and 3-opt descent, keeping the best tour.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::Duration;

use super::{Budget, DistMatrix, Tour, greedy_nearest_neighbor};

/// How many nearest neighbours each node considers in 2-opt and 3-opt.
const NEIGH_LIMIT: usize = 20;
/// Width of the window that `shuffle_tour` shuffles.
const GEO_SHUFFLE_WIDTH: usize = 30;
/// Each search thread stops after this many perturbations in a row
/// without a strictly better tour. Tuned so tour quality matches running for the full time limit on
/// random Euclidean instances up to n = 500, while small instances finish
/// in a fraction of the limit. Larger instances are bound by the time limit.
const MAX_ITER_NO_IMPROVEMENT: u32 = 5000;

/// A node, or an index into the tour. Four bytes instead of `usize` keeps
/// the tour, positions and neighbour lists compact, which makes the 2-opt
/// and 3-opt scans noticeably faster on large inputs. A distance matrix
/// with more than `u32::MAX` rows could never fit in memory.
type Node = u32;

/// The index after `i` on a cycle of length `n`.
#[inline(always)]
fn next(i: usize, n: usize) -> usize {
    if i + 1 == n { 0 } else { i + 1 }
}

/// The index before `i` on a cycle of length `n`.
#[inline(always)]
fn prev(i: usize, n: usize) -> usize {
    if i == 0 { n - 1 } else { i - 1 }
}

/// For each node, its `k` closest other nodes, closest first, stored in
/// one flat buffer. Ties go to the lowest index.
struct Neighbours {
    k: usize,
    data: Vec<Node>,
}

impl Neighbours {
    /// O(n^2). If the budget is interrupted while building, the lists are
    /// left empty; the search loop will not run in that case anyway.
    fn new(dists: &DistMatrix, budget: &Budget<'_>) -> Self {
        let n = dists.len();
        let k = NEIGH_LIMIT.min(n.saturating_sub(1));
        let mut data = Vec::with_capacity(n * k);
        let mut row: Vec<(i64, Node)> = Vec::with_capacity(n);
        for i in 0..n {
            if i % 64 == 0 && budget.is_interrupted() {
                return Self {
                    k: 0,
                    data: Vec::new(),
                };
            }
            row.clear();
            row.extend(
                (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (dists.get(i, j), j as Node)),
            );
            if k < row.len() {
                row.select_nth_unstable(k);
                row.truncate(k);
            }
            row.sort_unstable();
            data.extend(row.iter().map(|&(_, j)| j));
        }
        Self { k, data }
    }

    #[inline(always)]
    fn of(&self, node: usize) -> &[Node] {
        &self.data[node * self.k..(node + 1) * self.k]
    }
}

fn cost(dists: &DistMatrix, tour: &[Node]) -> i64 {
    let n = tour.len();
    (0..n)
        .map(|i| dists.get(tour[i] as usize, tour[next(i, n)] as usize))
        .sum()
}

/// The best tour found so far by any search thread.
///
/// Threads publish each strict improvement here and, before every
/// perturbation, switch to the shared tour if it is better than their own.
/// All threads therefore work on the current best tour, each trying
/// different random perturbations of it.
struct SharedBest {
    /// Copy of the cost under the lock, so threads can check for a better
    /// tour without locking.
    cost: AtomicI64,
    best: Mutex<(i64, Vec<Node>)>,
}

impl SharedBest {
    fn new(cost: i64, tour: Vec<Node>) -> Self {
        Self {
            cost: AtomicI64::new(cost),
            best: Mutex::new((cost, tour)),
        }
    }

    fn lock(&self) -> MutexGuard<'_, (i64, Vec<Node>)> {
        // A poisoned lock means another thread panicked, and that panic is
        // re-raised when the threads are joined, so the data is never used.
        self.best.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn cost(&self) -> i64 {
        self.cost.load(Ordering::Relaxed)
    }

    /// Stores `tour` if it is strictly better than the shared tour.
    fn offer(&self, cost: i64, tour: &[Node]) {
        let mut best = self.lock();
        if cost < best.0 {
            best.0 = cost;
            best.1.clear();
            best.1.extend_from_slice(tour);
            self.cost.store(cost, Ordering::Relaxed);
        }
    }

    /// Copies the shared tour into `tour` and returns its cost.
    fn copy_into(&self, tour: &mut Vec<Node>) -> i64 {
        let best = self.lock();
        tour.clone_from(&best.1);
        best.0
    }

    fn into_tour(self) -> Vec<Node> {
        self.best
            .into_inner()
            .unwrap_or_else(PoisonError::into_inner)
            .1
    }
}

/// How often the calling thread checks the interrupt callback while it
/// waits for the other search threads to finish.
const WAIT_POLL_INTERVAL: Duration = Duration::from_millis(1);

/// Runs the local search on `threads` threads, starting from the greedy
/// tour, and returns the best tour found.
///
/// The calling thread is one of the search threads. It alone runs the
/// budget's interrupt callback (which may need the calling thread, e.g. to
/// handle Ctrl-C in Python), and tells the other threads to stop through a
/// shared flag.
pub(crate) fn local_search(
    dists: &DistMatrix,
    budget: &Budget<'_>,
    threads: usize,
    seed: Option<u64>,
) -> Tour {
    let mut rng = seed.map_or_else(fastrand::Rng::new, fastrand::Rng::with_seed);
    let nn = Neighbours::new(dists, budget);
    let start: Vec<Node> = greedy_nearest_neighbor(dists)
        .expect("n >= 2 is checked by the caller")
        .into_iter()
        .map(|v| v as Node)
        .collect();
    let shared = SharedBest::new(cost(dists, &start), start);

    if threads <= 1 {
        search(dists, &nn, &shared, budget, &mut rng);
    } else {
        let stop = AtomicBool::new(false);
        let deadline = budget.deadline();
        thread::scope(|scope| {
            let workers: Vec<_> = (1..threads)
                .map(|_| {
                    let (nn, shared, stop) = (&nn, &shared, &stop);
                    let mut worker_rng = rng.fork();
                    scope.spawn(move || {
                        let budget = Budget::worker(deadline, stop);
                        search(dists, nn, shared, &budget, &mut worker_rng);
                    })
                })
                .collect();
            search(dists, &nn, &shared, budget, &mut rng);
            // This thread may finish first (or be interrupted). Keep polling
            // the callback until the other threads are done.
            while !workers.iter().all(|w| w.is_finished()) {
                if budget.is_interrupted() {
                    stop.store(true, Ordering::Relaxed);
                }
                thread::sleep(WAIT_POLL_INTERVAL);
            }
        });
    }
    shared.into_tour().into_iter().map(|v| v as usize).collect()
}

/// One search thread: perturb the best tour, re-optimise it, and keep the
/// result if it is at least as good. Stops when the budget is exceeded or
/// after `MAX_ITER_NO_IMPROVEMENT` perturbations in a row that did not lead
/// to a better tour (found by this thread or any other).
fn search(
    dists: &DistMatrix,
    nn: &Neighbours,
    shared: &SharedBest,
    budget: &Budget<'_>,
    rng: &mut fastrand::Rng,
) {
    let n = dists.len();
    let mut best_tour = Vec::with_capacity(n);
    let mut best_cost = shared.copy_into(&mut best_tour);
    let mut tour = best_tour.clone();
    let mut position: Vec<Node> = vec![0; n];
    let mut scratch: Vec<Node> = Vec::with_capacity(n);
    let mut num_iter_no_improvement = 0;

    while !budget.is_exceeded() && num_iter_no_improvement < MAX_ITER_NO_IMPROVEMENT {
        if shared.cost() < best_cost {
            // Another thread found a better tour: continue from it.
            best_cost = shared.copy_into(&mut best_tour);
            tour.clone_from(&best_tour);
            num_iter_no_improvement = 0;
        }

        double_bridge_move(&mut tour, &mut scratch, rng);
        shuffle_tour(&mut tour, rng);
        for (i, &v) in tour.iter().enumerate() {
            position[v as usize] = i as Node;
        }

        two_opt(&mut tour, dists, nn, &mut position, budget);
        three_opt(&mut tour, dists, nn, &mut position, &mut scratch, budget);

        let cost = cost(dists, &tour);
        if cost <= best_cost {
            // Equal-cost tours are accepted so the search can drift across
            // plateaus, but only a strictly better tour resets the counter.
            // Otherwise ties would keep the search alive forever.
            if cost < best_cost {
                num_iter_no_improvement = 0;
                shared.offer(cost, &tour);
            } else {
                num_iter_no_improvement += 1;
            }
            best_cost = cost;
            best_tour.clone_from(&tour);
        } else {
            tour.clone_from(&best_tour);
            num_iter_no_improvement += 1;
        }
    }
}

/// Splits the tour into segments A|B|C|D and reorders them to A|D|C|B.
/// `scratch` is a reusable buffer.
fn double_bridge_move(tour: &mut Vec<Node>, scratch: &mut Vec<Node>, rng: &mut fastrand::Rng) {
    let n = tour.len();
    if n < 8 {
        rng.shuffle(tour);
        return;
    }
    let a = rng.usize(1..=n / 4);
    let b = a + rng.usize(1..=n / 4);
    let c = b + rng.usize(1..=n / 4);
    scratch.clear();
    scratch.extend_from_slice(&tour[..a]);
    scratch.extend_from_slice(&tour[c..]);
    scratch.extend_from_slice(&tour[b..c]);
    scratch.extend_from_slice(&tour[a..b]);
    std::mem::swap(tour, scratch);
}

/// Shuffles a random window of `GEO_SHUFFLE_WIDTH` consecutive nodes.
fn shuffle_tour(tour: &mut [Node], rng: &mut fastrand::Rng) {
    let n = tour.len();
    if n <= GEO_SHUFFLE_WIDTH {
        rng.shuffle(tour);
        return;
    }
    let left = rng.usize(0..=n - GEO_SHUFFLE_WIDTH);
    rng.shuffle(&mut tour[left..left + GEO_SHUFFLE_WIDTH]);
}

/// Reverses the cyclic segment `start..=end` in place and updates positions.
/// Adapted from estan/tsp (see LICENSE).
fn reverse_tour(tour: &mut [Node], start: usize, end: usize, position: &mut [Node]) {
    let n = tour.len();
    let len = if start <= end {
        end - start
    } else {
        end + n - start
    };
    let (mut i, mut j) = (start, end);
    for _ in 0..len.div_ceil(2) {
        tour.swap(i, j);
        position[tour[i] as usize] = i as Node;
        position[tour[j] as usize] = j as Node;
        i = next(i, n);
        j = prev(j, n);
    }
}

/// Replaces cyclic segment `a..=b` followed by `c..=d` with `c..=d`
/// followed by `a..=b`, starting at index `a`. `temp` is a reusable buffer.
fn swap_adj_seg(
    tour: &mut [Node],
    position: &mut [Node],
    temp: &mut Vec<Node>,
    (a, b): (usize, usize),
    (c, d): (usize, usize),
) {
    let n = tour.len();
    temp.clear();
    for (from, to) in [(c, d), (a, b)] {
        let mut cur = from;
        loop {
            temp.push(tour[cur]);
            if cur == to {
                break;
            }
            cur = next(cur, n);
        }
    }
    let mut idx = a;
    for &v in temp.iter() {
        tour[idx] = v;
        position[v as usize] = idx as Node;
        idx = next(idx, n);
    }
}

/// 2-opt with neighbour lists, taking the first improving move. Runs until
/// no move improves the tour or time runs out.
fn two_opt(
    tour: &mut [Node],
    dists: &DistMatrix,
    nn: &Neighbours,
    position: &mut [Node],
    budget: &Budget<'_>,
) {
    let n = tour.len();
    let mut idx1 = 0;
    let mut improving = true;
    while improving && !budget.is_exceeded() {
        improving = false;
        for _ in 0..n {
            let u1 = tour[idx1] as usize;
            let next_idx = next(idx1, n);
            let u2 = tour[next_idx] as usize;
            let d0 = dists.get(u1, u2);

            for &u3 in nn.of(u1) {
                let u3 = u3 as usize;
                let d1 = dists.get(u1, u3);
                // Later neighbours are even further away.
                if d0 <= d1 {
                    break;
                }
                let u3_pos = position[u3] as usize;
                let u4 = tour[next(u3_pos, n)] as usize;
                if u1 == u4 || u2 == u3 {
                    continue;
                }
                if d0 + dists.get(u3, u4) > d1 + dists.get(u2, u4) {
                    reverse_tour(tour, next_idx, u3_pos, position);
                    improving = true;
                    break;
                }
            }
            idx1 = next_idx;
        }
    }
}

/// 3-opt with neighbour lists, taking the first improving move. Runs until
/// no move improves the tour or time runs out.
fn three_opt(
    tour: &mut [Node],
    dists: &DistMatrix,
    nn: &Neighbours,
    position: &mut [Node],
    scratch: &mut Vec<Node>,
    budget: &Budget<'_>,
) {
    let n = tour.len();
    let d = |a: usize, b: usize| dists.get(a, b);
    let mut improving = true;
    while improving && !budget.is_exceeded() {
        improving = false;

        'next_u1: for u1_idx in 0..n {
            let u2_idx = next(u1_idx, n);
            let u1 = tour[u1_idx] as usize;
            let u2 = tour[u2_idx] as usize;
            let d12 = d(u1, u2);

            for &u3 in nn.of(u2) {
                let u3 = u3 as usize;
                let u3_idx = position[u3] as usize;
                let d23 = d(u2, u3);
                // Later neighbours are even further away.
                if d12 <= d23 {
                    break;
                }
                if u3 == u1 {
                    continue;
                }

                // u5 must lie in the right cyclic order relative to u2 and u3.
                let in_order = |u5_idx: usize| {
                    (u5_idx < u3_idx && u3_idx < u2_idx)
                        || (u3_idx < u2_idx && u2_idx < u5_idx)
                        || (u2_idx < u5_idx && u5_idx < u3_idx)
                };

                // Case 1: u4 follows u3. Reconnect by moving a segment.
                let u4_idx = next(u3_idx, n);
                let u4 = tour[u4_idx] as usize;
                if u4_idx != u1_idx {
                    let d34 = d(u3, u4);
                    for &u5 in nn.of(u4) {
                        let u5 = u5 as usize;
                        if u5 == u3 || u5 == u1 || u5 == u2 {
                            continue;
                        }
                        let u5_idx = position[u5] as usize;
                        if !in_order(u5_idx) {
                            continue;
                        }
                        let g = d23 - d12 + d(u4, u5);
                        if d34 <= g {
                            break;
                        }
                        let u6 = tour[next(u5_idx, n)] as usize;
                        if g - d34 + d(u1, u6) < d(u5, u6) {
                            improving = true;
                            swap_adj_seg(
                                tour,
                                position,
                                scratch,
                                (u4_idx, u1_idx),
                                (u2_idx, u5_idx),
                            );
                            continue 'next_u1;
                        }
                    }
                }

                // Case 2: u4 precedes u3. Reconnect with two reversals.
                let u4_idx = prev(u3_idx, n);
                let u4 = tour[u4_idx] as usize;
                let d34 = d(u3, u4);
                for &u5 in nn.of(u4) {
                    let u5 = u5 as usize;
                    if u5 == u3 || u5 == u1 || u5 == u2 {
                        continue;
                    }
                    let u5_idx = position[u5] as usize;
                    if !in_order(u5_idx) {
                        continue;
                    }
                    let g = d23 - d12 + d(u4, u5);
                    if d34 <= g {
                        break;
                    }
                    let u6_idx = next(u5_idx, n);
                    let u6 = tour[u6_idx] as usize;
                    if g - d34 + d(u1, u6) < d(u5, u6) {
                        improving = true;
                        reverse_tour(tour, u6_idx, u4_idx, position);
                        reverse_tour(tour, u3_idx, u1_idx, position);
                        continue 'next_u1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::euclidean;
    use super::super::{Budget, is_valid_tour};
    use super::*;

    fn nodes(n: usize) -> Vec<Node> {
        (0..n as Node).collect()
    }

    fn is_valid(tour: &[Node]) -> bool {
        let tour: Vec<usize> = tour.iter().map(|&v| v as usize).collect();
        is_valid_tour(tour.len(), &tour)
    }

    fn positions_match(tour: &[Node], position: &[Node]) -> bool {
        tour.iter()
            .enumerate()
            .all(|(i, &v)| position[v as usize] as usize == i)
    }

    fn positions_of(tour: &[Node]) -> Vec<Node> {
        let mut pos = vec![0; tour.len()];
        for (i, &v) in tour.iter().enumerate() {
            pos[v as usize] = i as Node;
        }
        pos
    }

    #[test]
    fn reverse_tour_wraps_around() {
        let mut tour = nodes(6);
        let mut pos = positions_of(&tour);
        reverse_tour(&mut tour, 4, 1, &mut pos);
        assert_eq!(tour, vec![5, 4, 2, 3, 1, 0]);
        assert!(positions_match(&tour, &pos));

        let mut tour = nodes(6);
        let mut pos = positions_of(&tour);
        reverse_tour(&mut tour, 1, 3, &mut pos);
        assert_eq!(tour, vec![0, 3, 2, 1, 4, 5]);
        assert!(positions_match(&tour, &pos));
    }

    #[test]
    fn swap_adj_seg_swaps_segments() {
        let mut temp = Vec::new();
        let mut tour = nodes(7);
        let mut pos = positions_of(&tour);
        // [1..=2] followed by [3..=5] becomes [3..=5] followed by [1..=2].
        swap_adj_seg(&mut tour, &mut pos, &mut temp, (1, 2), (3, 5));
        assert_eq!(tour, vec![0, 3, 4, 5, 1, 2, 6]);
        assert!(positions_match(&tour, &pos));

        // Wrapping case.
        let mut tour = nodes(7);
        let mut pos = positions_of(&tour);
        swap_adj_seg(&mut tour, &mut pos, &mut temp, (5, 6), (0, 1));
        assert_eq!(tour, vec![5, 6, 2, 3, 4, 0, 1]);
        assert!(positions_match(&tour, &pos));
    }

    #[test]
    fn perturbations_keep_tour_valid() {
        let mut rng = fastrand::Rng::with_seed(7);
        let mut scratch = Vec::new();
        for n in [5, 8, 31, 100] {
            let mut tour = nodes(n);
            for _ in 0..50 {
                double_bridge_move(&mut tour, &mut scratch, &mut rng);
                shuffle_tour(&mut tour, &mut rng);
                assert!(is_valid(&tour));
            }
        }
    }

    #[test]
    fn nearest_neighbours_are_sorted() {
        let d = euclidean(50, 3);
        let nn = Neighbours::new(&d, &Budget::unlimited());
        for i in 0..50 {
            let row = nn.of(i);
            assert_eq!(row.len(), NEIGH_LIMIT);
            assert!(!row.contains(&(i as Node)));
            let mut all: Vec<(i64, Node)> = (0..50)
                .filter(|&j| j != i)
                .map(|j| (d.get(i, j), j as Node))
                .collect();
            all.sort();
            let expected: Vec<Node> = all[..NEIGH_LIMIT].iter().map(|&(_, j)| j).collect();
            assert_eq!(row, &expected[..]);
        }
    }

    #[test]
    fn small_inputs_have_short_neighbour_lists() {
        let d = euclidean(5, 1);
        let nn = Neighbours::new(&d, &Budget::unlimited());
        for i in 0..5 {
            assert_eq!(nn.of(i).len(), 4);
        }
    }

    #[test]
    fn opt_moves_never_worsen_and_keep_positions() {
        let d = euclidean(120, 11);
        let nn = Neighbours::new(&d, &Budget::unlimited());
        let budget = Budget::unlimited();
        let mut rng = fastrand::Rng::with_seed(1);
        let mut tour = nodes(120);
        rng.shuffle(&mut tour);
        let mut pos = positions_of(&tour);
        let mut scratch = Vec::new();

        let before = cost(&d, &tour);
        two_opt(&mut tour, &d, &nn, &mut pos, &budget);
        let after_two = cost(&d, &tour);
        assert!(positions_match(&tour, &pos));
        three_opt(&mut tour, &d, &nn, &mut pos, &mut scratch, &budget);
        let after_three = cost(&d, &tour);
        assert!(positions_match(&tour, &pos));
        assert!(is_valid(&tour));
        assert!(after_two < before);
        assert!(after_three <= after_two);
    }
}
