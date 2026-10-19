//! Iterated local search: repeated local perturbation (a double bridge
//! confined to a short window, now and then a global one) followed by a
//! Lin-Kernighan + 2-opt/3-opt descent around the perturbed nodes, keeping
//! the best tour and walking from any tour within a small slack of it.
//! A walk that stops improving restarts from a perturbed best tour.
//! Every step costs time proportional to the work it causes, not to the
//! tour length.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::Duration;

use super::candidates::nearest;
use super::greedy::nearest_neighbor_tour;
#[cfg(test)]
use super::greedy_nearest_neighbor;
use super::{Budget, DistMatrix, Tour};

/// How many nearest neighbours each node considers in 2-opt, 3-opt and LK.
///
/// Measured with the LK descent: shorter lists (8-16, also for LK alone)
/// lose quality without making kicks measurably cheaper (the positive-gain
/// `break` ends most scans early), mainly on instances with many equal
/// distances (d1291, rl1889) and on fl*. Alpha-nearness lists (LKH style,
/// pure or mixed with nearest, gated by build cost) and minimum-spanning-
/// tree neighbours give better tours per kick on clustered instances but
/// were within noise of these lists in the timed benchmark.
const NEIGH_LIMIT: usize = 20;
/// Width of the window that the global kick shuffles.
const GEO_SHUFFLE_WIDTH: usize = 30;

/// Perturbation applied before each local re-optimisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kick {
    /// Rewrites three short consecutive segments X|Y|Z at a random place as
    /// Z|Y|X: a double bridge confined to a window. Changes four edges.
    SegmentDoubleBridge,
    /// Swaps two short consecutive segments X|Y into Y|X. Changes three
    /// edges (a 3-opt move, so the descent can often undo it directly).
    SegmentSwap,
    /// The original kick: a double bridge with segments up to n/4 plus a
    /// shuffle of a random 30-node window. Moves O(n) nodes per kick (but
    /// evaluates only the changed edges), and is undone through the journal.
    Global,
}

/// The kick `search` uses, except for one kick in `GLOBAL_KICK_ONE_IN`
/// (on average), which is `Kick::Global`. With LK in the descent a kick
/// costs about 10x more than with 2-opt/3-opt alone, so global kicks must be
/// about as much more frequent to keep their rate per second (clustered
/// instances such as fl1577 depend on that rate; 128 was tuned without LK).
/// With LK, 4..64 score the same on the dev set (3 seeds) except fl1577,
/// which looked best at 16.
const KICK: Kick = Kick::SegmentDoubleBridge;
const GLOBAL_KICK_ONE_IN: u32 = 16;
/// Longest segment of a local kick. Each of the three segments is drawn
/// uniformly from `1..=KICK_SEGMENT_MAX`. 200 rather than 100: the longer
/// segments let the local kick move whole clusters (fl1577 at 0.2 s, mean
/// over 8 seeds: 0.69 -> 0.88 with LK) and help most mid-sized instances;
/// 30-50 or a bias towards short segments is clearly worse.
const KICK_SEGMENT_MAX: usize = 200;
/// Acceptance ("record-to-record travel"): a kicked and re-optimised tour
/// becomes the tour the search continues from if its cost is at most
/// `best + slack`, where `slack` is this fraction of the mean edge length
/// of the first local optimum. The walk can thus drift a little uphill
/// out of a basin, but never far from the best tour, so no restarts are
/// needed. 0 gives the old rule (accept ties with the best).
const ACCEPT_SLACK_PER_EDGE: f64 = 0.1;
/// A search thread stops after `max(MIN_KICKS_WITHOUT_IMPROVEMENT,
/// KICKS_PER_NODE * n)` kicks in a row without a strictly better tour.
/// Larger instances are bound by the time limit. Before walk restarts (see
/// `RESTART_KICKS_PER_NODE`) this was 100 per node / 20 000, which ended
/// runs whose walk was trapped (d1291 after 9 s of a 10 s budget, d1655 and
/// pr1002 after ~12 s). With restarts the search keeps finding better tours
/// long after that, so the rule is only a ceiling for long or unlimited
/// budgets: at least 33 restarts in a row (about 170 for n > 1000) that all
/// failed to improve. At 10 s, inputs with n < 200 now return after 1-4 s
/// (was 0.3-0.8 s), always with the same (optimal) tours.
const KICKS_PER_NODE: usize = 500;
const MIN_KICKS_WITHOUT_IMPROVEMENT: u32 = 100_000;
/// Walk restarts. The record-to-record walk settles in one basin within a
/// second or two and cannot leave it with its small slack (d1291, fl1577,
/// d1655, pr1002 stopped improving after 1-2 s of a 10 s budget; growing
/// the slack up to a whole mean edge while stuck did not get them out). After
/// `max(MIN_KICKS_PER_RESTART, RESTART_KICKS_PER_NODE * n)` kicks in a row
/// that did not improve the walk's own record, the thread restarts the walk
/// from its best tour perturbed by `RESTART_PERTURBATION` segment double
/// bridges (each up to `3 * KICK_SEGMENT_MAX` nodes at a random place) and
/// a descent. The new walk is judged against its own record, so it can
/// settle in a different basin before it has to beat the best tour.
/// Restarting every 3 n non-improving kicks with 32 bridges was best among
/// 2-10 n and 8-64 bridges (10 s, d1291 d1655 d2103 fl1577 pr1002 rl1889
/// u1817 u2152, 2 seeds: 7.65 -> 7.97 of 8). At 0.2 s restarts are rare on
/// inputs with n >= 500, which make about `MIN_KICKS_PER_RESTART` kicks.
const RESTART_KICKS_PER_NODE: usize = 3;
const MIN_KICKS_PER_RESTART: u32 = 3_000;
const RESTART_PERTURBATION: u32 = 32;
/// Maximum number of flips in one Lin-Kernighan step.
const LK_MAX_DEPTH: usize = 25;
/// How many choices of `t3` an LK step tries at its first levels before
/// giving up on that level (backtracking). Deeper levels try only the best.
const LK_BREADTH: [usize; 2] = [3, 1];
const LK_MAX_BREADTH: usize = 3;
const _: () = assert!(LK_BREADTH[0] <= LK_MAX_BREADTH && LK_BREADTH[1] <= LK_MAX_BREADTH);

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
    /// O(n^2), on up to `threads` threads. Returns `None` if the budget
    /// runs out (deadline or interrupt) before the lists are complete.
    fn build(dists: &DistMatrix, budget: &Budget<'_>, threads: usize) -> Option<Self> {
        let k = NEIGH_LIMIT.min(dists.len().saturating_sub(1));
        nearest(dists, k, budget, threads).map(|data| Self { k, data })
    }

    /// `build` on one thread, for tests with a budget that cannot run out.
    #[cfg(test)]
    fn new(dists: &DistMatrix, budget: &Budget<'_>) -> Self {
        Self::build(dists, budget, 1).expect("the budget ran out")
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

/// Runs the local search on `threads` threads, starting from the
/// nearest-neighbour tour, and returns the best tour found. If the budget
/// runs out while the neighbour lists are built, returns the
/// nearest-neighbour tour right away.
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
    let Some(nn) = Neighbours::build(dists, budget, threads) else {
        // An interrupted run's tour is discarded; otherwise return the
        // cheapest reasonable tour.
        if budget.check_interrupted().is_err() {
            return (0..dists.len()).collect();
        }
        return nearest_neighbor_tour(dists)
            .into_iter()
            .map(|v| v as usize)
            .collect();
    };
    let start = nearest_neighbor_tour(dists);
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

/// One search thread: perturb the current tour, re-optimise it locally,
/// and continue from the result if it is within the acceptance slack of the
/// walk's record (see `ACCEPT_SLACK_PER_EDGE`), restarting the walk from a
/// perturbed best tour when it stagnates (see `RESTART_KICKS_PER_NODE`).
/// Stops when the budget is exceeded or after
/// `max_kicks_without_improvement(n)` perturbations in a row that did not
/// lead to a better tour (found by this thread or any other).
///
/// The cost is tracked incrementally (kick delta minus move gains), and a
/// rejected kick is rolled back through the undo journal, so a kick costs
/// time proportional to the work it causes, not to n.
fn search(
    dists: &DistMatrix,
    nn: &Neighbours,
    shared: &SharedBest,
    budget: &Budget<'_>,
    rng: &mut fastrand::Rng,
) {
    search_with_slack(dists, nn, shared, budget, rng, ACCEPT_SLACK_PER_EDGE);
}

/// `search` with the acceptance slack as a parameter. Returns the cost of
/// the tour the walk ended on, which may be worse than the best tour (the
/// best tour is only ever published to `shared`).
fn search_with_slack(
    dists: &DistMatrix,
    nn: &Neighbours,
    shared: &SharedBest,
    budget: &Budget<'_>,
    rng: &mut fastrand::Rng,
    slack_per_edge: f64,
) -> i64 {
    let n = dists.len();
    let mut tour = Vec::with_capacity(n);
    let mut cur_cost = shared.copy_into(&mut tour);
    let mut opt = Opt::new(dists, nn, tour);

    // Full descent from the start tour, every node active.
    opt.activate_all();
    cur_cost -= opt.descend(budget);
    opt.journal.clear();
    debug_assert_eq!(cur_cost, cost(dists, &opt.tour));
    shared.offer(cur_cost, &opt.tour);

    // The walk (`opt.tour`, `cur_cost`) may be up to `slack` worse than the
    // best tour this thread knows, so keep a copy of that. Copying on each
    // new best costs O(n), but new bests are rare compared with kicks.
    let mut best_cost = cur_cost;
    let mut best_tour = opt.tour.clone();
    let mean_edge = (best_cost as f64 / n as f64).max(1.0);
    let slack = (slack_per_edge * mean_edge).round() as i64;

    let max_fails = max_kicks_without_improvement(n);
    let mut num_iter_no_improvement = 0;
    // The walk's own record and the kicks since it last improved, for the
    // restarts (see `RESTART_KICKS_PER_NODE`). Until the first restart the
    // record is the best tour.
    let restart_after = kicks_per_restart(n);
    let mut walk_best = best_cost;
    let mut walk_fails = 0;
    while !budget.is_exceeded() && num_iter_no_improvement < max_fails {
        if shared.cost() < best_cost {
            // Another thread found a better tour: continue from it.
            best_cost = shared.copy_into(&mut best_tour);
            opt.tour.clone_from(&best_tour);
            opt.reset_positions();
            cur_cost = best_cost;
            num_iter_no_improvement = 0;
            walk_best = best_cost;
            walk_fails = 0;
        }

        if walk_fails >= restart_after {
            cur_cost = opt.restart_from(&best_tour, best_cost, budget, rng);
            walk_best = cur_cost;
            walk_fails = 0;
            if cur_cost < best_cost {
                debug_assert_eq!(cur_cost, self::cost(dists, &opt.tour));
                num_iter_no_improvement = 0;
                shared.offer(cur_cost, &opt.tour);
                best_tour.clone_from(&opt.tour);
                best_cost = cur_cost;
            }
            continue;
        }

        // Mostly local kicks. An occasional global one reconnects distant
        // parts of the tour, which local kicks cannot (clustered instances
        // such as fl1577 get stuck without it).
        let kick = if rng.u32(0..GLOBAL_KICK_ONE_IN) == 0 {
            Kick::Global
        } else {
            KICK
        };
        let delta = opt.kick(kick, cur_cost, rng);
        let gain = opt.descend(budget);
        let cost = cur_cost + delta - gain;
        if cost < best_cost {
            debug_assert_eq!(cost, self::cost(dists, &opt.tour));
            num_iter_no_improvement = 0;
            shared.offer(cost, &opt.tour);
            best_tour.clone_from(&opt.tour);
            best_cost = cost;
        } else {
            // Only a strictly better tour resets the counter; otherwise the
            // walk (and its restarts) could keep the search alive forever.
            num_iter_no_improvement += 1;
        }
        if cost < walk_best {
            walk_best = cost;
            walk_fails = 0;
        } else {
            walk_fails += 1;
        }
        // Record-to-record acceptance against the walk's record, which is
        // the best tour except after a restart.
        if cost <= walk_best + slack {
            debug_assert_eq!(cost, self::cost(dists, &opt.tour));
            cur_cost = cost;
            opt.journal.clear();
        } else {
            opt.undo();
        }
    }
    cur_cost
}

/// How many kicks in a row without improving the walk's own record make a
/// search thread restart its walk (see `RESTART_KICKS_PER_NODE`).
fn kicks_per_restart(n: usize) -> u32 {
    u32::try_from(n.saturating_mul(RESTART_KICKS_PER_NODE))
        .unwrap_or(u32::MAX)
        .max(MIN_KICKS_PER_RESTART)
}

/// How many kicks in a row without a strictly better tour end a search
/// thread. Kicks are cheap (local work only), so this is far larger than
/// the number of kicks a full-tour perturbation scheme would need; it still
/// guarantees termination without a deadline.
fn max_kicks_without_improvement(n: usize) -> u32 {
    u32::try_from(n.saturating_mul(KICKS_PER_NODE))
        .unwrap_or(u32::MAX)
        .max(MIN_KICKS_WITHOUT_IMPROVEMENT)
}

/// Reverses the `len` tour positions starting at `start` (wrapping around)
/// and updates positions.
fn reverse_positions(tour: &mut [Node], position: &mut [Node], start: usize, len: usize) {
    if len < 2 {
        return;
    }
    let n = tour.len();
    let (mut i, mut j) = (start, (start + len - 1) % n);
    for _ in 0..len / 2 {
        tour.swap(i, j);
        position[tour[i] as usize] = i as Node;
        position[tour[j] as usize] = j as Node;
        i = next(i, n);
        j = prev(j, n);
    }
}

/// Reverses the cyclic segment `start..=end`, or its complement when that
/// is shorter (both give the same cyclic tour), and updates positions.
/// Returns the reversed range as `(start, len)`. Indices computed before
/// the call are stale afterwards: re-read `position`.
fn reverse_tour(
    tour: &mut [Node],
    start: usize,
    end: usize,
    position: &mut [Node],
) -> (usize, usize) {
    let n = tour.len();
    let len = if start <= end {
        end - start + 1
    } else {
        end + n - start + 1
    };
    let (start, len) = if 2 * len > n {
        (next(end, n), n - len)
    } else {
        (start, len)
    };
    reverse_positions(tour, position, start, len);
    (start, len)
}

/// Rewrites the consecutive (wrapping) segments `X|Y|Z` of lengths `a`, `b`,
/// `c` starting at `start` as `Z|Y|X`, and updates positions. With `b == 0`
/// this swaps two adjacent segments. Its inverse is the same call with `a`
/// and `c` swapped. `scratch` is a reusable buffer.
fn rotate_segments(
    tour: &mut [Node],
    position: &mut [Node],
    scratch: &mut Vec<Node>,
    start: usize,
    (a, b, c): (usize, usize, usize),
) {
    let n = tour.len();
    let total = a + b + c;
    debug_assert!(total <= n);
    scratch.clear();
    let mut idx = start;
    for _ in 0..total {
        scratch.push(tour[idx]);
        idx = next(idx, n);
    }
    let (x, rest) = scratch.split_at(a);
    let (y, z) = rest.split_at(b);
    let mut idx = start;
    for &v in z.iter().chain(y).chain(x) {
        tour[idx] = v;
        position[v as usize] = idx as Node;
        idx = next(idx, n);
    }
}

/// A tour change recorded so a rejected kick can be rolled back.
#[derive(Clone, Copy, Debug)]
enum Undo {
    /// `len` positions from `start` were reversed (self-inverse).
    Reverse { start: Node, len: Node },
    /// `rotate_segments(start, (a, b, c))` was applied.
    Rotate {
        start: Node,
        a: Node,
        b: Node,
        c: Node,
    },
    /// Tour positions `i` and `j` were swapped (self-inverse).
    Swap(Node, Node),
}

/// A tour with node positions, a queue of active nodes for the 2-opt/3-opt
/// descent, and an undo journal.
struct Opt<'a> {
    dists: &'a DistMatrix,
    nn: &'a Neighbours,
    tour: Vec<Node>,
    position: Vec<Node>,
    scratch: Vec<Node>,
    /// Nodes whose tour edges may allow an improving move (don't-look bits).
    queue: VecDeque<Node>,
    queued: Vec<bool>,
    /// Changes since the last accepted tour, oldest first.
    journal: Vec<Undo>,
    /// Queue pops so far, to check the deadline every `POPS_PER_DEADLINE_CHECK`.
    pops: u32,
    /// Working state of the current Lin-Kernighan step.
    lk: LkState,
}

/// Working state of one Lin-Kernighan step (`Opt::lk_from`), kept between
/// steps so a step allocates nothing.
#[derive(Default)]
struct LkState {
    /// `(t3, t4)` of each level of the current step.
    steps: Vec<(Node, Node)>,
    /// Edges added and removed by the current step (tabu lists).
    added: Vec<(Node, Node)>,
    removed: Vec<(Node, Node)>,
    /// Best closed gain so far, its number of levels, and the journal
    /// length right after its last flip.
    best_gain: i64,
    best_len: usize,
    best_journal: usize,
}

/// An undirected edge, smaller node first.
#[inline(always)]
fn edge(a: usize, b: usize) -> (Node, Node) {
    if a < b {
        (a as Node, b as Node)
    } else {
        (b as Node, a as Node)
    }
}

/// The descent checks the deadline after this many queue pops (a single
/// descent from the start tour can take long on large inputs, and one LK
/// step can take tens of microseconds).
const POPS_PER_DEADLINE_CHECK: u32 = 32;

impl<'a> Opt<'a> {
    fn new(dists: &'a DistMatrix, nn: &'a Neighbours, tour: Vec<Node>) -> Self {
        let n = tour.len();
        let mut opt = Self {
            dists,
            nn,
            tour,
            position: vec![0; n],
            scratch: Vec::with_capacity(n),
            queue: VecDeque::with_capacity(n),
            queued: vec![false; n],
            journal: Vec::new(),
            pops: 0,
            lk: LkState::default(),
        };
        opt.reset_positions();
        opt
    }

    /// Rebuilds `position` after `tour` was replaced, and forgets the journal.
    fn reset_positions(&mut self) {
        for (i, &v) in self.tour.iter().enumerate() {
            self.position[v as usize] = i as Node;
        }
        self.journal.clear();
    }

    #[inline(always)]
    fn n(&self) -> usize {
        self.tour.len()
    }

    #[inline(always)]
    fn pos(&self, v: usize) -> usize {
        self.position[v] as usize
    }

    #[inline(always)]
    fn succ(&self, v: usize) -> usize {
        self.tour[next(self.pos(v), self.n())] as usize
    }

    #[inline(always)]
    fn pred(&self, v: usize) -> usize {
        self.tour[prev(self.pos(v), self.n())] as usize
    }

    /// The next node when walking the tour forwards (`FWD`) or backwards.
    #[inline(always)]
    fn next_dir<const FWD: bool>(&self, v: usize) -> usize {
        if FWD { self.succ(v) } else { self.pred(v) }
    }

    #[inline(always)]
    fn prev_dir<const FWD: bool>(&self, v: usize) -> usize {
        if FWD { self.pred(v) } else { self.succ(v) }
    }

    /// True if `b` lies on the forward path from `a` to `c` (inclusive).
    #[inline(always)]
    fn between(&self, a: usize, b: usize, c: usize) -> bool {
        let (a, b, c) = (self.pos(a), self.pos(b), self.pos(c));
        if a <= c {
            a <= b && b <= c
        } else {
            a <= b || b <= c
        }
    }

    /// `between` when walking in direction `FWD`.
    #[inline(always)]
    fn between_dir<const FWD: bool>(&self, a: usize, b: usize, c: usize) -> bool {
        if FWD {
            self.between(a, b, c)
        } else {
            self.between(c, b, a)
        }
    }

    #[inline(always)]
    fn push(&mut self, v: usize) {
        if !self.queued[v] {
            self.queued[v] = true;
            self.queue.push_back(v as Node);
        }
    }

    fn push_all(&mut self, nodes: &[usize]) {
        for &v in nodes {
            self.push(v);
        }
    }

    fn activate_all(&mut self) {
        for i in 0..self.n() {
            let v = self.tour[i] as usize;
            self.push(v);
        }
    }

    fn clear_queue(&mut self) {
        while let Some(v) = self.queue.pop_front() {
            self.queued[v as usize] = false;
        }
    }

    /// Reverses tour positions `start..=end` (or the complement), journaled.
    fn reverse(&mut self, start: usize, end: usize) {
        let (start, len) = reverse_tour(&mut self.tour, start, end, &mut self.position);
        if len >= 2 {
            self.journal.push(Undo::Reverse {
                start: start as Node,
                len: len as Node,
            });
        }
    }

    /// `rotate_segments`, journaled.
    fn rotate(&mut self, start: usize, (a, b, c): (usize, usize, usize)) {
        rotate_segments(
            &mut self.tour,
            &mut self.position,
            &mut self.scratch,
            start,
            (a, b, c),
        );
        self.journal.push(Undo::Rotate {
            start: start as Node,
            a: a as Node,
            b: b as Node,
            c: c as Node,
        });
    }

    /// Rolls the tour back to the last accepted state.
    fn undo(&mut self) {
        self.clear_queue();
        self.rollback(0);
    }

    /// Undoes the journaled changes past the first `keep`, newest first.
    fn rollback(&mut self, keep: usize) {
        while self.journal.len() > keep {
            let op = self.journal.pop().expect("longer than keep");
            match op {
                Undo::Reverse { start, len } => reverse_positions(
                    &mut self.tour,
                    &mut self.position,
                    start as usize,
                    len as usize,
                ),
                Undo::Rotate { start, a, b, c } => rotate_segments(
                    &mut self.tour,
                    &mut self.position,
                    &mut self.scratch,
                    start as usize,
                    (c as usize, b as usize, a as usize),
                ),
                Undo::Swap(i, j) => self.swap_raw(i as usize, j as usize),
            }
        }
    }

    /// 2-opt move on tour edges `(t1, t2)` and `(t4, t3)`, where `t4` is the
    /// neighbour of `t3` on the far side from the `t1 -> t2` direction:
    /// replaces them with `(t2, t3)` and `(t4, t1)`. Works in either tour
    /// orientation, so it can be chained after other flips.
    fn flip(&mut self, t1: usize, t2: usize, t3: usize) {
        if self.succ(t1) == t2 {
            // t1 t2 .. t4 t3: reverse t2..t4.
            let t4 = self.pred(t3);
            self.reverse(self.pos(t2), self.pos(t4));
        } else {
            debug_assert_eq!(self.pred(t1), t2);
            // t2 t1 .. t3 t4: reverse t1..t3.
            self.reverse(self.pos(t1), self.pos(t3));
        }
    }

    /// Pure 3-opt segment move without reversals: removes the three tour
    /// edges `(x, y)` with `y` following `x` in direction `FWD`, and
    /// reconnects the three resulting paths A|B|C as A|C|B. Moves the two
    /// shortest paths.
    fn or_move<const FWD: bool>(&mut self, edges: [(usize, usize); 3]) {
        let n = self.n();
        // Forward index of the first node after each removed edge.
        let mut cuts = edges.map(|(x, y)| if FWD { self.pos(y) } else { self.pos(x) });
        cuts.sort_unstable();
        let [c0, c1, c2] = cuts;
        let lens = [c1 - c0, c2 - c1, n - (c2 - c0)];
        // Leave the longest path in place and swap the other two.
        if lens[2] >= lens[0] && lens[2] >= lens[1] {
            self.rotate(c0, (lens[0], 0, lens[1]));
        } else if lens[0] >= lens[1] {
            self.rotate(c1, (lens[1], 0, lens[2]));
        } else {
            self.rotate(c2, (lens[2], 0, lens[0]));
        }
    }

    /// Tries 2-opt and 3-opt moves that remove the tour edge from `t1` to
    /// its neighbour in direction `FWD`, with neighbour-list candidates and
    /// positive partial gains. Applies the first improving move found,
    /// activates the endpoints of all changed edges, and returns its gain
    /// (0 if there is none).
    fn improve<const FWD: bool>(&mut self, t1: usize) -> i64 {
        let (dists, nn) = (self.dists, self.nn);
        let d = |a: usize, b: usize| dists.get(a, b);
        let t2 = self.next_dir::<FWD>(t1);
        let d12 = d(t1, t2);
        let t2_next = self.next_dir::<FWD>(t2);
        for &t3 in nn.of(t2) {
            let t3 = t3 as usize;
            let g1 = d12 - d(t2, t3);
            // Later neighbours are even further away.
            if g1 <= 0 {
                break;
            }
            if t3 == t1 || t3 == t2_next {
                continue;
            }
            for t4_after_t3 in [false, true] {
                let t4 = if t4_after_t3 {
                    self.next_dir::<FWD>(t3)
                } else {
                    self.prev_dir::<FWD>(t3)
                };
                // t4 == t1 (t3 just before t1) is fine: the moves below then
                // reinsert t1 elsewhere.
                let g2 = g1 + d(t3, t4);
                if !t4_after_t3 {
                    // t1 t2 .. t4 t3 ..: closing with (t4, t1) is a 2-opt move.
                    let gain = g2 - d(t4, t1);
                    if gain > 0 {
                        self.flip(t1, t2, t3);
                        self.push_all(&[t1, t2, t3, t4]);
                        return gain;
                    }
                }
                let (t4_succ, t4_pred) = (self.succ(t4), self.pred(t4));
                for &t5 in nn.of(t4) {
                    let t5 = t5 as usize;
                    let g3 = g2 - d(t4, t5);
                    if g3 <= 0 {
                        break;
                    }
                    if t5 == t1 || t5 == t4_succ || t5 == t4_pred {
                        continue;
                    }
                    if !t4_after_t3 {
                        // Either t1 [t2..t5][t6..t4][t3..] (A B' C' type) or
                        // t1 [t2..t4][t3..t6][t5..] (A C B' type); both are
                        // two flips.
                        let t6 = if self.between_dir::<FWD>(t2, t5, t4) {
                            self.next_dir::<FWD>(t5)
                        } else {
                            self.prev_dir::<FWD>(t5)
                        };
                        let gain = g3 + d(t5, t6) - d(t6, t1);
                        if gain > 0 {
                            self.flip(t1, t2, t3);
                            self.flip(t1, t4, t5);
                            self.push_all(&[t1, t2, t3, t4, t5, t6]);
                            return gain;
                        }
                    } else {
                        // t1 t2 .. t3 t4 ..: t5 must lie on t2..t3.
                        if !self.between_dir::<FWD>(t2, t5, t3) {
                            continue;
                        }
                        // t1 [t2..t5][t6..t3][t4..] -> A C B (segment move).
                        let t6 = self.next_dir::<FWD>(t5);
                        let gain = g3 + d(t5, t6) - d(t6, t1);
                        if gain > 0 {
                            self.or_move::<FWD>([(t1, t2), (t5, t6), (t3, t4)]);
                            self.push_all(&[t1, t2, t3, t4, t5, t6]);
                            return gain;
                        }
                        // t1 [t2..t6][t5..t3][t4..] -> A' B' C.
                        if t5 != t2 {
                            let t6 = self.prev_dir::<FWD>(t5);
                            let gain = g3 + d(t5, t6) - d(t6, t1);
                            if gain > 0 {
                                self.flip(t1, t2, t5);
                                self.flip(t2, t5, t4);
                                self.push_all(&[t1, t2, t3, t4, t5, t6]);
                                return gain;
                            }
                        }
                    }
                }
            }
        }
        0
    }

    /// Lin-Kernighan step from base node `t1`, built from journaled flips.
    ///
    /// For each tour neighbour `t2` of `t1`, the edge `t1-t2` is treated as
    /// removed, leaving a Hamiltonian path with ends `t1` and `t2`. Each
    /// level adds an edge `t2-t3` to a near neighbour `t3` and removes the
    /// edge `t3-t4` (`t4` the neighbour of `t3` on the `t2` side) that makes
    /// the path Hamiltonian again; `t4` becomes the new free end. On the array
    /// this is one `flip`, after which `t1-t4` is a tour edge, so the tour is
    /// valid after every level and closing the move costs `d(t4, t1)`.
    ///
    /// Partial gains must stay positive (the neighbour lists are sorted, so
    /// the scan stops at the first that fails), an edge removed in this step
    /// is never added back and an edge added is never removed. The first
    /// levels try up to `LK_BREADTH[level]` choices of `t3` (backtracking),
    /// deeper ones only the best, up to `LK_MAX_DEPTH` flips. The step keeps
    /// the prefix with the best closed gain and rolls the journal back past
    /// it, so tentative flips leave no trace.
    ///
    /// Applies the move and returns its gain if it is strictly positive (so
    /// zero-gain moves can never cycle), activating the ends of every changed
    /// edge. Otherwise returns 0 with the tour and journal as they were.
    fn lk_from(&mut self, t1: usize) -> i64 {
        if self.n() < 5 {
            return 0;
        }
        for t2 in [self.succ(t1), self.pred(t1)] {
            let start = self.journal.len();
            let lk = &mut self.lk;
            lk.steps.clear();
            lk.added.clear();
            lk.removed.clear();
            lk.removed.push(edge(t1, t2));
            lk.best_gain = 0;
            lk.best_len = 0;
            lk.best_journal = start;
            if self.lk_deepen(t1, t2, self.dists.get(t1, t2), 0) {
                self.rollback(self.lk.best_journal);
                self.push(t1);
                self.push(t2);
                for i in 0..self.lk.best_len {
                    let (t3, t4) = self.lk.steps[i];
                    self.push(t3 as usize);
                    self.push(t4 as usize);
                }
                return self.lk.best_gain;
            }
            debug_assert_eq!(self.journal.len(), start);
        }
        0
    }

    /// One level of `lk_from`: `t1-t2` is the open (virtual) edge and `g`
    /// the gain so far without closing. Returns true once an improving prefix
    /// was found, leaving its flips (and possibly more) applied; otherwise
    /// rolls its flips back and returns false.
    fn lk_deepen(&mut self, t1: usize, t2: usize, g: i64, level: usize) -> bool {
        let (dists, nn) = (self.dists, self.nn);
        let d = |a: usize, b: usize| dists.get(a, b);
        // Walking the tour from t1 to t2 defines the orientation.
        let forward = self.succ(t1) == t2;
        let succ2 = if forward {
            self.succ(t2)
        } else {
            self.pred(t2)
        };
        let breadth = LK_BREADTH.get(level).copied().unwrap_or(1);

        // The best `breadth` candidates by d(t3, t4) - d(t2, t3), best first.
        let mut cands = [(0_i64, 0 as Node, 0 as Node); LK_MAX_BREADTH];
        let mut num = 0;
        for &t3 in nn.of(t2) {
            let t3 = t3 as usize;
            let d23 = d(t2, t3);
            // Positive gain criterion. Later neighbours are even further away.
            if g - d23 <= 0 {
                break;
            }
            if t3 == t1 || t3 == succ2 {
                continue;
            }
            let t4 = if forward {
                self.pred(t3)
            } else {
                self.succ(t3)
            };
            let score = d(t3, t4) - d23;
            if num == breadth && score <= cands[num - 1].0 {
                continue;
            }
            if self.lk.removed.contains(&edge(t2, t3)) || self.lk.added.contains(&edge(t3, t4)) {
                continue;
            }
            let mut i = num.min(breadth - 1);
            while i > 0 && cands[i - 1].0 < score {
                cands[i] = cands[i - 1];
                i -= 1;
            }
            cands[i] = (score, t3 as Node, t4 as Node);
            num = (num + 1).min(breadth);
        }

        for &(score, t3, t4) in &cands[..num] {
            let (t3, t4) = (t3 as usize, t4 as usize);
            let mark = self.journal.len();
            // Removes t1-t2 and t4-t3, adds t2-t3 and t4-t1.
            self.flip(t1, t2, t3);
            let lk = &mut self.lk;
            lk.steps.push((t3 as Node, t4 as Node));
            lk.added.push(edge(t2, t3));
            lk.removed.push(edge(t3, t4));
            let g = g + score;
            let closed = g - d(t4, t1);
            if closed > lk.best_gain {
                lk.best_gain = closed;
                lk.best_len = lk.steps.len();
                lk.best_journal = self.journal.len();
            }
            if level + 1 < LK_MAX_DEPTH {
                self.lk_deepen(t1, t4, g, level + 1);
            }
            if self.lk.best_gain > 0 {
                return true;
            }
            self.rollback(mark);
            let lk = &mut self.lk;
            lk.steps.pop();
            lk.added.pop();
            lk.removed.pop();
        }
        false
    }

    /// `improve` from both tour edges of `v`.
    fn improve_both(&mut self, v: usize) -> i64 {
        let gain = self.improve::<true>(v);
        if gain > 0 {
            gain
        } else {
            self.improve::<false>(v)
        }
    }

    /// Applies improving moves around active nodes until none is left (or
    /// the budget runs out). Returns the total gain.
    fn descend(&mut self, budget: &Budget<'_>) -> i64 {
        let mut total = 0;
        while let Some(v) = self.queue.pop_front() {
            let v = v as usize;
            self.queued[v] = false;
            self.pops = self.pops.wrapping_add(1);
            if self.pops % POPS_PER_DEADLINE_CHECK == 0 && budget.is_exceeded() {
                self.clear_queue();
                break;
            }
            // LK first; the 2-opt/3-opt moves (in particular the pure
            // segment moves, which LK cannot express) only when it fails.
            let gain = self.lk_from(v);
            total += if gain > 0 { gain } else { self.improve_both(v) };
        }
        total
    }

    /// Perturbs the tour, activates the nodes around the change and returns
    /// the change in cost. `cost` is the current tour cost.
    fn kick(&mut self, kick: Kick, cost: i64, rng: &mut fastrand::Rng) -> i64 {
        let n = self.n();
        // The window needs at least one node outside it.
        let max_len = KICK_SEGMENT_MAX.min((n - 1) / 3);
        if kick == Kick::Global || max_len < 2 {
            return self.global_kick(cost, rng);
        }
        let a = rng.usize(1..=max_len);
        let b = if kick == Kick::SegmentSwap {
            0
        } else {
            rng.usize(1..=max_len)
        };
        let c = rng.usize(1..=max_len);
        let start = rng.usize(0..n);
        let tour = &self.tour;
        let at = |k: usize| tour[(start + k) % n] as usize;
        let (p, x0, x1) = (at(n - 1), at(0), at(a - 1));
        let (z0, z1, q) = (at(a + b), at(a + b + c - 1), at(a + b + c));
        let dists = self.dists;
        let d = |u: usize, v: usize| dists.get(u, v);
        let (delta, ys) = if b == 0 {
            // p X Z q -> p Z X q
            let delta = d(p, z0) + d(z1, x0) + d(x1, q) - d(p, x0) - d(x1, z0) - d(z1, q);
            (delta, None)
        } else {
            // p X Y Z q -> p Z Y X q
            let (y0, y1) = (at(a), at(a + b - 1));
            let delta = d(p, z0) + d(z1, y0) + d(y1, x0) + d(x1, q)
                - d(p, x0)
                - d(x1, y0)
                - d(y1, z0)
                - d(z1, q);
            (delta, Some([y0, y1]))
        };
        self.rotate(start, (a, b, c));
        self.push_all(&[p, x0, x1, z0, z1, q]);
        if let Some(ys) = ys {
            self.push_all(&ys);
        }
        delta
    }

    /// Replaces the tour with `tour` (of cost `cost`), applies
    /// `RESTART_PERTURBATION` local kicks and a descent, and returns the
    /// new cost. The result is the new accepted tour (the journal is empty).
    fn restart_from(
        &mut self,
        tour: &[Node],
        cost: i64,
        budget: &Budget<'_>,
        rng: &mut fastrand::Rng,
    ) -> i64 {
        self.clear_queue();
        self.tour.clear();
        self.tour.extend_from_slice(tour);
        self.reset_positions();
        let mut cost = cost;
        for _ in 0..RESTART_PERTURBATION {
            cost += self.kick(KICK, cost, rng);
        }
        cost -= self.descend(budget);
        self.journal.clear();
        cost
    }

    /// Swaps tour positions `i` and `j` and updates positions.
    fn swap_raw(&mut self, i: usize, j: usize) {
        self.tour.swap(i, j);
        self.position[self.tour[i] as usize] = i as Node;
        self.position[self.tour[j] as usize] = j as Node;
    }

    /// Shuffles the `len` tour positions from `start` (no wrap-around) with
    /// journaled swaps.
    fn shuffle(&mut self, start: usize, len: usize, rng: &mut fastrand::Rng) {
        for k in (1..len).rev() {
            let r = rng.usize(0..=k);
            if r != k {
                self.swap_raw(start + r, start + k);
                self.journal
                    .push(Undo::Swap((start + r) as Node, (start + k) as Node));
            }
        }
    }

    /// Sum of the `count` tour edges starting at position `start`.
    fn path_cost(&self, start: usize, count: usize) -> i64 {
        let n = self.n();
        let mut i = start;
        let mut sum = 0;
        for _ in 0..count {
            let j = next(i, n);
            sum += self.dists.get(self.tour[i] as usize, self.tour[j] as usize);
            i = j;
        }
        sum
    }

    /// `Kick::Global`: a double bridge A|B|C|D -> A|D|C|B with segments of
    /// up to n/4 nodes, then a shuffle of a random `GEO_SHUFFLE_WIDTH`-node
    /// window. Moves O(n) nodes but evaluates only the changed edges.
    fn global_kick(&mut self, cost: i64, rng: &mut fastrand::Rng) -> i64 {
        let n = self.n();
        if n < 8 {
            self.shuffle(0, n, rng);
            self.activate_all();
            return self::cost(self.dists, &self.tour) - cost;
        }
        let a = rng.usize(1..=n / 4);
        let b = a + rng.usize(1..=n / 4);
        let c = b + rng.usize(1..=n / 4);
        let at = |i: usize| self.tour[i] as usize;
        let ends = [
            at(0),
            at(a - 1),
            at(a),
            at(b - 1),
            at(b),
            at(c - 1),
            at(c),
            at(n - 1),
        ];
        let [fa, la, fb, lb, fc, lc, fd, ld] = ends;
        let d = |u: usize, v: usize| self.dists.get(u, v);
        let mut delta = d(la, fd) + d(ld, fc) + d(lc, fb) + d(lb, fa)
            - d(la, fb)
            - d(lb, fc)
            - d(lc, fd)
            - d(ld, fa);
        self.rotate(a, (b - a, c - b, n - c));
        self.push_all(&ends);
        if n > GEO_SHUFFLE_WIDTH + 1 {
            let left = rng.usize(0..=n - GEO_SHUFFLE_WIDTH);
            // The window's edges, including the two leading into and out of it.
            let first = prev(left, n);
            let before = self.path_cost(first, GEO_SHUFFLE_WIDTH + 1);
            self.shuffle(left, GEO_SHUFFLE_WIDTH, rng);
            delta += self.path_cost(first, GEO_SHUFFLE_WIDTH + 1) - before;
            let mut i = first;
            for _ in 0..GEO_SHUFFLE_WIDTH + 2 {
                let v = self.tour[i] as usize;
                self.push(v);
                i = next(i, n);
            }
        }
        delta
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

    /// True if both tours are the same cycle, in either direction.
    fn same_cycle(a: &[Node], b: &[Node]) -> bool {
        let n = a.len();
        let edges = |t: &[Node]| {
            let mut e: Vec<(Node, Node)> = (0..n)
                .map(|i| {
                    let (u, v) = (t[i], t[next(i, n)]);
                    (u.min(v), u.max(v))
                })
                .collect();
            e.sort_unstable();
            e
        };
        a.len() == b.len() && edges(a) == edges(b)
    }

    #[test]
    fn reverse_tour_wraps_around() {
        // Positions 4, 5, 0, 1 are more than half the tour: the complement
        // 2..=3 is reversed instead, which gives the same cycle.
        let mut tour = nodes(6);
        let mut pos = positions_of(&tour);
        assert_eq!(reverse_tour(&mut tour, 4, 1, &mut pos), (2, 2));
        assert_eq!(tour, vec![0, 1, 3, 2, 4, 5]);
        assert!(same_cycle(&tour, &[5, 4, 2, 3, 1, 0]));
        assert!(positions_match(&tour, &pos));

        let mut tour = nodes(6);
        let mut pos = positions_of(&tour);
        reverse_tour(&mut tour, 1, 3, &mut pos);
        assert_eq!(tour, vec![0, 3, 2, 1, 4, 5]);
        assert!(positions_match(&tour, &pos));

        // Short wrapping segment: reversed in place.
        let mut tour = nodes(8);
        let mut pos = positions_of(&tour);
        assert_eq!(reverse_tour(&mut tour, 6, 1, &mut pos), (6, 4));
        assert_eq!(tour, vec![7, 6, 2, 3, 4, 5, 1, 0]);
        assert!(positions_match(&tour, &pos));
    }

    #[test]
    fn reverse_tour_matches_naive_reversal() {
        let n = 9;
        for start in 0..n {
            for end in 0..n {
                let mut tour = nodes(n);
                let mut pos = positions_of(&tour);
                reverse_tour(&mut tour, start, end, &mut pos);
                let mut naive = nodes(n);
                let len = (end + n - start) % n + 1;
                for k in 0..len / 2 {
                    naive.swap((start + k) % n, (end + n - k) % n);
                }
                assert!(same_cycle(&tour, &naive), "{start}..={end}");
                assert!(positions_match(&tour, &pos));
            }
        }
    }

    #[test]
    fn rotate_segments_moves_segments() {
        let mut temp = Vec::new();
        let mut tour = nodes(7);
        let mut pos = positions_of(&tour);
        // [1..=2] followed by [3..=5] becomes [3..=5] followed by [1..=2].
        rotate_segments(&mut tour, &mut pos, &mut temp, 1, (2, 0, 3));
        assert_eq!(tour, vec![0, 3, 4, 5, 1, 2, 6]);
        assert!(positions_match(&tour, &pos));

        // Wrapping case.
        let mut tour = nodes(7);
        let mut pos = positions_of(&tour);
        rotate_segments(&mut tour, &mut pos, &mut temp, 5, (2, 0, 2));
        assert_eq!(tour, vec![5, 6, 2, 3, 4, 0, 1]);
        assert!(positions_match(&tour, &pos));

        // X|Y|Z -> Z|Y|X, wrapping, and its inverse.
        let mut tour = nodes(7);
        let mut pos = positions_of(&tour);
        rotate_segments(&mut tour, &mut pos, &mut temp, 6, (1, 2, 3));
        assert_eq!(tour, vec![3, 4, 0, 1, 6, 5, 2]);
        assert!(positions_match(&tour, &pos));
        rotate_segments(&mut tour, &mut pos, &mut temp, 6, (3, 2, 1));
        assert_eq!(tour, nodes(7));
        assert!(positions_match(&tour, &pos));
    }

    #[test]
    fn kicks_track_cost_and_undo_exactly() {
        let mut rng = fastrand::Rng::with_seed(3);
        for n in [5, 8, 9, 21, 40, 400] {
            let d = euclidean(n, n as u64);
            let nn = Neighbours::new(&d, &Budget::unlimited());
            for kick in [Kick::SegmentDoubleBridge, Kick::SegmentSwap, Kick::Global] {
                let mut opt = Opt::new(&d, &nn, nodes(n));
                for _ in 0..200 {
                    let before = opt.tour.clone();
                    let c = cost(&d, &opt.tour);
                    let delta = opt.kick(kick, c, &mut rng);
                    assert!(is_valid(&opt.tour));
                    assert!(positions_match(&opt.tour, &opt.position));
                    assert_eq!(c + delta, cost(&d, &opt.tour));
                    let gain = opt.descend(&Budget::unlimited());
                    assert_eq!(c + delta - gain, cost(&d, &opt.tour));
                    if rng.bool() {
                        opt.undo();
                        assert_eq!(opt.tour, before);
                        assert!(positions_match(&opt.tour, &opt.position));
                    } else {
                        opt.journal.clear();
                    }
                }
            }
        }
    }

    #[test]
    fn restart_tracks_cost_and_leaves_the_start_tour() {
        let mut rng = fastrand::Rng::with_seed(8);
        for n in [5, 9, 40, 700] {
            let d = euclidean(n, n as u64);
            let nn = Neighbours::new(&d, &Budget::unlimited());
            let mut opt = Opt::new(&d, &nn, nodes(n));
            opt.activate_all();
            opt.descend(&Budget::unlimited());
            opt.journal.clear();
            let best = opt.tour.clone();
            let best_cost = cost(&d, &best);
            // Start from some other tour, as a walk would be.
            let other_cost = opt.restart_from(
                &nodes(n),
                cost(&d, &nodes(n)),
                &Budget::unlimited(),
                &mut rng,
            );
            assert_eq!(other_cost, cost(&d, &opt.tour));
            let mut moved = false;
            for _ in 0..5 {
                let c = opt.restart_from(&best, best_cost, &Budget::unlimited(), &mut rng);
                assert!(is_valid(&opt.tour));
                assert!(positions_match(&opt.tour, &opt.position));
                assert!(opt.journal.is_empty() && opt.queue.is_empty());
                assert_eq!(c, cost(&d, &opt.tour));
                moved |= !same_cycle(&opt.tour, &best);
            }
            // The perturbation must actually move the walk somewhere else.
            assert!(n < 40 || moved, "n={n}");
        }
    }

    #[test]
    fn restarts_come_well_before_the_stop_rule() {
        for n in [1, 20, 500, 1000, 5000, 20_000] {
            assert!(kicks_per_restart(n) >= MIN_KICKS_PER_RESTART);
            // At least 20 restarts in a row must fail before a thread stops.
            assert!(20 * kicks_per_restart(n) <= max_kicks_without_improvement(n));
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
        let mut rng = fastrand::Rng::with_seed(1);
        let mut tour = nodes(120);
        rng.shuffle(&mut tour);
        let start = tour.clone();
        let before = cost(&d, &tour);

        let mut opt = Opt::new(&d, &nn, tour);
        opt.activate_all();
        let gain = opt.descend(&Budget::unlimited());
        assert!(gain > 0);
        assert!(is_valid(&opt.tour));
        assert!(positions_match(&opt.tour, &opt.position));
        assert_eq!(cost(&d, &opt.tour), before - gain);

        // The journal rolls every move back.
        opt.undo();
        assert_eq!(opt.tour, start);
        assert!(positions_match(&opt.tour, &opt.position));
    }

    /// Every 2-opt move and every pure 3-opt reconnection, by brute force.
    /// Returns the best gain found (0 if the tour is 3-opt optimal).
    fn best_brute_force_gain(d: &DistMatrix, t: &[Node]) -> (i64, i64) {
        let n = t.len();
        let c = |a: usize, b: usize| d.get(t[a % n] as usize, t[b % n] as usize);
        let mut best2 = 0;
        for i in 0..n {
            for j in i + 2..n {
                if (j + 1) % n == i {
                    continue;
                }
                best2 = best2.max(c(i, i + 1) + c(j, j + 1) - c(i, j) - c(i + 1, j + 1));
            }
        }
        let mut best3 = 0;
        for i in 0..n {
            for j in i + 1..n {
                for k in j + 1..n - 1 {
                    // A = ..=i, B = i+1..=j, C = j+1..=k, A continues at k+1.
                    let (a1, b0, b1, c0, c1, a0) = (i, i + 1, j, j + 1, k, k + 1);
                    let old = c(a1, b0) + c(b1, c0) + c(c1, a0);
                    for new in [
                        c(a1, b1) + c(b0, c1) + c(c0, a0), // A B' C'
                        c(a1, c0) + c(c1, b0) + c(b1, a0), // A C B
                        c(a1, c0) + c(c1, b1) + c(b0, a0), // A C B'
                        c(a1, c1) + c(c0, b0) + c(b1, a0), // A C' B
                    ] {
                        best3 = best3.max(old - new);
                    }
                }
            }
        }
        (best2, best3)
    }

    #[test]
    fn queue_descent_reaches_two_and_three_opt_optimum() {
        // With n = 21 the neighbour lists hold every other node, so a
        // descent with every node active must leave no improving 2-opt move
        // (in either orientation) and no improving pure 3-opt move: the
        // move search from both tour edges of every node is complete.
        //
        // After a kick only the kicked nodes are active. That descent can
        // (rarely, ~0.2% of kicks) miss a move between two untouched edges
        // whose relative orientation a reversal flipped, as with any
        // don't-look-bit scheme; the next kicks there pick it up.
        let n = NEIGH_LIMIT + 1;
        for seed in 0..40 {
            let d = euclidean(n, 100 + seed);
            let nn = Neighbours::new(&d, &Budget::unlimited());
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut tour = nodes(n);
            rng.shuffle(&mut tour);
            let mut opt = Opt::new(&d, &nn, tour);
            for _ in 0..10 {
                opt.activate_all();
                opt.descend(&Budget::unlimited());
                assert!(opt.queue.is_empty());
                assert_eq!(best_brute_force_gain(&d, &opt.tour), (0, 0), "seed {seed}");
                for _ in 0..10 {
                    let c = cost(&d, &opt.tour);
                    opt.kick(Kick::SegmentDoubleBridge, c, &mut rng);
                    opt.descend(&Budget::unlimited());
                    opt.journal.clear();
                }
            }
        }
    }

    #[test]
    fn between_matches_a_walk_along_the_cycle() {
        for n in 1..=7 {
            let d = euclidean(n, n as u64);
            let nn = Neighbours::new(&d, &Budget::unlimited());
            // With the identity tour, node v sits at position v.
            let opt = Opt::new(&d, &nn, nodes(n));
            for a in 0..n {
                for c in 0..n {
                    // Positions visited walking forward from a to c.
                    let mut on_path = vec![false; n];
                    let mut i = a;
                    loop {
                        on_path[i] = true;
                        if i == c {
                            break;
                        }
                        i = next(i, n);
                    }
                    for (b, &expected) in on_path.iter().enumerate() {
                        assert_eq!(opt.between(a, b, c), expected, "n={n} a={a} b={b} c={c}");
                        assert_eq!(opt.between_dir::<false>(c, b, a), expected);
                    }
                }
            }
        }
    }

    #[test]
    fn lk_step_gains_what_it_claims() {
        for (n, seed) in [(5, 1), (6, 2), (9, 3), (30, 4), (200, 5)] {
            let d = euclidean(n, seed);
            let nn = Neighbours::new(&d, &Budget::unlimited());
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut tour = nodes(n);
            rng.shuffle(&mut tour);
            let mut opt = Opt::new(&d, &nn, tour);
            let mut improved = 0;
            for _ in 0..3 {
                for t1 in 0..n {
                    let before_tour = opt.tour.clone();
                    let before = cost(&d, &opt.tour);
                    let mark = opt.journal.len();
                    let gain = opt.lk_from(t1);
                    assert!(gain >= 0);
                    assert!(is_valid(&opt.tour));
                    assert!(positions_match(&opt.tour, &opt.position));
                    assert_eq!(cost(&d, &opt.tour), before - gain);
                    if gain == 0 {
                        // A failed step leaves no trace.
                        assert_eq!(opt.tour, before_tour);
                        assert_eq!(opt.journal.len(), mark);
                        assert!(opt.queue.is_empty());
                        continue;
                    }
                    improved += 1;
                    // Every changed edge has both ends activated.
                    let old = edge_list(&before_tour);
                    for e in edge_list(&opt.tour) {
                        if old.binary_search(&e).is_err() {
                            assert!(opt.queued[e.0 as usize] && opt.queued[e.1 as usize]);
                        }
                    }
                    // The journal restores the old tour exactly, and the
                    // step is deterministic.
                    let after = opt.tour.clone();
                    opt.rollback(mark);
                    assert_eq!(opt.tour, before_tour);
                    assert!(positions_match(&opt.tour, &opt.position));
                    assert_eq!(opt.lk_from(t1), gain);
                    assert_eq!(opt.tour, after);
                    opt.clear_queue();
                }
            }
            assert!(n < 9 || improved > 0);
        }
    }

    /// The tour as a sorted list of undirected edges.
    fn edge_list(tour: &[Node]) -> Vec<(Node, Node)> {
        let n = tour.len();
        let mut e: Vec<_> = (0..n)
            .map(|i| edge(tour[i] as usize, tour[next(i, n)] as usize))
            .collect();
        e.sort_unstable();
        e
    }

    /// The queue descent with 2-opt and 3-opt moves only (no LK steps).
    fn descend_without_lk(opt: &mut Opt<'_>) -> i64 {
        let mut total = 0;
        while let Some(v) = opt.queue.pop_front() {
            opt.queued[v as usize] = false;
            total += opt.improve_both(v as usize);
        }
        total
    }

    #[test]
    fn lk_descent_beats_two_and_three_opt() {
        let d = euclidean(300, 13);
        let nn = Neighbours::new(&d, &Budget::unlimited());
        let mut start = nodes(300);
        fastrand::Rng::with_seed(3).shuffle(&mut start);

        let mut opt = Opt::new(&d, &nn, start.clone());
        opt.activate_all();
        descend_without_lk(&mut opt);
        let three = cost(&d, &opt.tour);

        let mut opt = Opt::new(&d, &nn, start);
        opt.activate_all();
        opt.descend(&Budget::unlimited());
        assert!(is_valid(&opt.tour));
        assert!(cost(&d, &opt.tour) < three);
    }

    #[test]
    fn descent_stops_when_time_is_up() {
        use std::time::{Duration, Instant};
        let d = euclidean(1000, 21);
        let nn = Neighbours::new(&d, &Budget::unlimited());
        let mut tour = nodes(1000);
        fastrand::Rng::with_seed(5).shuffle(&mut tour);
        let before = cost(&d, &tour);
        let mut opt = Opt::new(&d, &nn, tour);
        opt.activate_all();
        let past = Instant::now() - Duration::from_secs(2);
        let gain = opt.descend(&Budget::new(past, 1.0).unwrap());
        // At most one check interval of work, the queue is reset, and the
        // gain is exact.
        assert!(opt.queue.is_empty());
        assert!(opt.queued.iter().all(|&q| !q));
        assert!(is_valid(&opt.tour));
        assert!(positions_match(&opt.tour, &opt.position));
        assert_eq!(cost(&d, &opt.tour), before - gain);
    }

    /// Compares the first local optimum from the greedy tour, 2-opt + 3-opt
    /// against the descent with LK, on benchmark cases (run from the repo
    /// root): `CASES=pcb442,fnl4461 cargo test --release first_local_optimum
    /// -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn first_local_optimum() {
        use std::time::Instant;
        let names = std::env::var("CASES").unwrap_or_else(|_| "pcb442".into());
        for name in names.split(',') {
            let text = std::fs::read_to_string(format!("cases/{name}.txt")).unwrap();
            let mut nums = text.split_whitespace().map(|t| t.parse::<f64>().unwrap());
            let opt_cost = nums.next().unwrap();
            let _naive = nums.next().unwrap();
            let n = nums.next().unwrap() as usize;
            let pts: Vec<(f64, f64)> = (0..n)
                .map(|_| (nums.next().unwrap(), nums.next().unwrap()))
                .collect();
            let rows: Vec<Vec<u16>> = pts
                .iter()
                .map(|a| {
                    pts.iter()
                        .map(|b| (a.0 - b.0).hypot(a.1 - b.1).round() as u16)
                        .collect()
                })
                .collect();
            let d = super::super::test_utils::matrix(&rows);
            let nn = Neighbours::new(&d, &Budget::unlimited());
            let start: Vec<Node> = greedy_nearest_neighbor(&d)
                .unwrap()
                .into_iter()
                .map(|v| v as Node)
                .collect();
            let pct = |tour: &[Node]| 100.0 * (cost(&d, tour) as f64 - opt_cost) / opt_cost;

            let mut opt = Opt::new(&d, &nn, start.clone());
            opt.activate_all();
            let t = Instant::now();
            descend_without_lk(&mut opt);
            let (base, base_ms) = (pct(&opt.tour), t.elapsed().as_secs_f64() * 1e3);

            let mut opt = Opt::new(&d, &nn, start.clone());
            opt.activate_all();
            let t = Instant::now();
            opt.descend(&Budget::unlimited());
            let (lk, lk_ms) = (pct(&opt.tour), t.elapsed().as_secs_f64() * 1e3);
            assert!(is_valid(&opt.tour));
            println!(
                "{name:10} n={n:5} greedy {:6.2}% | 2-opt+3-opt {base:5.2}% {base_ms:5.2} ms \
                 | with LK {lk:5.2}% {lk_ms:5.2} ms",
                pct(&start)
            );
        }
    }

    #[test]
    fn expired_budget_still_returns_a_tour() {
        let d = euclidean(300, 2);
        let start = std::time::Instant::now() - Duration::from_secs(2);
        let past = Budget::new(start, 1.0).unwrap();
        for threads in [1, 4] {
            let tour = local_search(&d, &past, threads, Some(1));
            let tour: Vec<Node> = tour.into_iter().map(|v| v as Node).collect();
            assert!(is_valid(&tour));
        }
    }

    #[test]
    fn search_tracks_cost_exactly() {
        // `search` publishes its incrementally tracked cost with each tour.
        for n in [21, 200, 1000] {
            let d = euclidean(n, 5);
            let nn = Neighbours::new(&d, &Budget::unlimited());
            let start: Vec<Node> = greedy_nearest_neighbor(&d)
                .unwrap()
                .into_iter()
                .map(|v| v as Node)
                .collect();
            let shared = SharedBest::new(cost(&d, &start), start.clone());
            let budget = Budget::new(std::time::Instant::now(), 0.2).unwrap();
            search(&d, &nn, &shared, &budget, &mut fastrand::Rng::with_seed(4));
            let tracked = shared.cost();
            let tour = shared.into_tour();
            assert!(is_valid(&tour));
            assert!(tracked < cost(&d, &start));
            assert_eq!(tracked, cost(&d, &tour));
        }
    }

    #[test]
    fn search_publishes_best_tour_when_walk_ends_worse() {
        // With a huge acceptance slack the walk accepts almost every kick
        // and ends far above the best tour; `shared` must still hold the
        // best tour, with its exact cost.
        let n = 200;
        let d = euclidean(n, 9);
        let nn = Neighbours::new(&d, &Budget::unlimited());
        let start: Vec<Node> = greedy_nearest_neighbor(&d)
            .unwrap()
            .into_iter()
            .map(|v| v as Node)
            .collect();
        let shared = SharedBest::new(cost(&d, &start), start.clone());
        let budget = Budget::new(std::time::Instant::now(), 0.1).unwrap();
        let mut rng = fastrand::Rng::with_seed(2);
        let walk_cost = search_with_slack(&d, &nn, &shared, &budget, &mut rng, 1000.0);
        let best = shared.cost();
        let tour = shared.into_tour();
        assert!(is_valid(&tour));
        assert_eq!(best, cost(&d, &tour));
        assert!(best < cost(&d, &start));
        assert!(walk_cost > best, "walk {walk_cost}, best {best}");
    }
}
