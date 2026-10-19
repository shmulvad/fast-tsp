//! Candidate neighbour lists for the local search: the nearest neighbours,
//! or the alpha-nearest neighbours of a minimum 1-tree with node penalties
//! from a subgradient ascent (Held and Karp; Helsgaun 2000, as in LKH).
//!
//! The alpha value of an edge (i, j) is how much the minimum 1-tree grows
//! when it is forced to contain (i, j). The 1-tree is computed on the
//! transformed costs `D(i, j) = 100 d(i, j) + pi_i + pi_j`; the ascent
//! chooses the penalties `pi` so that the 1-tree is as long as possible
//! (its length minus `2 sum(pi)` is a lower bound on the optimal tour), which
//! pushes it towards a tour. Edges of an optimal tour then almost all have
//! small alpha values, much more reliably than they are short, so five
//! alpha-nearest candidates cover the optimal tour better than twenty
//! distance-nearest ones, above all on clustered instances.

use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use super::{Budget, Candidates, DistMatrix};

/// A row of the distance matrix.
#[inline(always)]
pub(crate) fn row(dists: &DistMatrix, i: usize) -> &[u16] {
    &dists.data[i * dists.n..(i + 1) * dists.n]
}

/// Inserts `key` into `best`, which holds the (at most `k`) smallest keys
/// seen so far in ascending order.
#[inline(always)]
fn push_best<T: Copy + Ord>(best: &mut Vec<T>, k: usize, key: T) {
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

/// A value type the block selection works on.
trait Lane: Copy + Ord {
    const MAX: Self;
}

impl Lane for u16 {
    const MAX: Self = u16::MAX;
}

impl Lane for i32 {
    const MAX: Self = i32::MAX;
}

/// The minimum of each `BLOCK`-sized block of `vals`, into `mins`.
/// Vectorises well.
#[inline]
fn block_minima<T: Lane>(vals: &[T], mins: &mut Vec<T>) {
    mins.clear();
    mins.extend(
        vals.chunks(BLOCK)
            .map(|block| block.iter().fold(T::MAX, |m, &v| m.min(v))),
    );
}

/// The k-th smallest block minimum, or `MAX` with fewer than k blocks. At
/// least k values (one per block) are at most this, so it bounds the k-th
/// smallest value, and only blocks whose minimum is within it need a closer
/// look: typically about k of them.
#[inline]
fn kth_smallest<T: Lane>(mins: &[T], k: usize, scratch: &mut Vec<T>) -> T {
    if mins.len() < k {
        return T::MAX;
    }
    scratch.clear();
    scratch.extend_from_slice(mins);
    *scratch.select_nth_unstable(k - 1).1
}

/// Matrices with fewer entries than this build their neighbour lists on
/// one thread; below it, starting threads costs more than it saves.
const PARALLEL_MIN_ENTRIES: usize = 1 << 21;
/// The same for the alpha pass, which costs several times more per entry.
const ALPHA_PARALLEL_MIN_ENTRIES: usize = 1 << 19;

/// How many threads a row-parallel pass over an n x n matrix uses.
fn threads_for(n: usize, threads: usize, min_entries: usize) -> usize {
    threads
        .min(n.saturating_mul(n).div_ceil(min_entries))
        .max(1)
}

/// Fills an `n * k` buffer row by row with `fill(first_row, out, budget)`,
/// splitting the rows into ranges on up to `threads` threads (about one
/// per `min_entries` matrix entries). `fill` returns false if its budget
/// ran out, and then so does the whole pass (`None`). Needs `n, k >= 1`.
fn split_rows<F>(
    n: usize,
    k: usize,
    threads: usize,
    min_entries: usize,
    budget: &Budget<'_>,
    fill: F,
) -> Option<Vec<u32>>
where
    F: Fn(usize, &mut [u32], &Budget<'_>) -> bool + Sync,
{
    let threads = threads_for(n, threads, min_entries);
    let mut data = vec![0; n * k];
    let rows_per_thread = n.div_ceil(threads);
    let mut parts = data.chunks_mut(rows_per_thread * k);
    let first = parts.next().expect("n >= 1");
    let complete = if threads == 1 {
        fill(0, first, budget)
    } else {
        let stop = AtomicBool::new(false);
        let deadline = budget.deadline();
        let fill = &fill;
        thread::scope(|scope| {
            let workers: Vec<_> = parts
                .enumerate()
                .map(|(t, out)| {
                    let stop = &stop;
                    scope.spawn(move || {
                        let budget = Budget::worker(deadline, stop);
                        fill((t + 1) * rows_per_thread, out, &budget)
                    })
                })
                .collect();
            // The calling thread takes the first range with the real
            // budget, the only one that can see an interrupt.
            let mine = fill(0, first, budget);
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
    split_rows(n, k, threads, min_entries, budget, |first, out, budget| {
        fill_rows(dists, k, first, out, budget)
    })
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

// Minimum 1-trees on transformed costs.

/// Distances are multiplied by this before the penalties are added, as
/// LKH's `PRECISION`, so the integer step sizes of the ascent have room.
const PRECISION: i32 = 100;
/// Penalties are clamped to this magnitude, so that a transformed cost
/// `100 d + pi_i + pi_j` (and the difference of two) fits in `i32`.
const PI_LIMIT: i64 = 1 << 28;

/// The transformed cost of an edge of length `d` between nodes with
/// penalties `pi_i` and `pi_j`.
#[inline(always)]
fn tcost(d: u16, pi_i: i32, pi_j: i32) -> i32 {
    PRECISION * i32::from(d) + pi_i + pi_j
}

/// A minimum 1-tree: a spanning tree rooted at node 0 plus one extra edge
/// from a leaf, the special node, to its cheapest node other than its tree
/// neighbour. All costs are transformed costs.
#[derive(Clone, Debug)]
pub(crate) struct OneTree {
    /// The nodes in the order they were added, so every parent comes
    /// before its children. `order[0]` is the root, node 0.
    pub(crate) order: Vec<u32>,
    /// The parent of every node; the root is its own parent.
    pub(crate) parent: Vec<u32>,
    /// The cost of the edge from every node to its parent (0 for the root).
    pub(crate) cost: Vec<i32>,
    /// The leaf with the extra edge, and the other end of that edge.
    pub(crate) special: usize,
    pub(crate) second: usize,
    /// Cost of the extra edge.
    pub(crate) second_cost: i32,
    /// The 1-tree's cost minus twice the penalty sum: a lower bound on 100
    /// times the optimal tour length.
    pub(crate) w: i64,
    /// Degree minus 2 of every node in the 1-tree; all 0 for a tour.
    pub(crate) v: Vec<i32>,
    /// The sum of squares of `v`.
    pub(crate) norm: i64,
    /// The cost of the spanning tree alone (no extra edge, no `-2 sum(pi)`).
    pub(crate) mst: i64,
}

impl OneTree {
    fn empty(n: usize) -> Self {
        Self {
            order: Vec::with_capacity(n),
            parent: vec![0; n],
            cost: vec![0; n],
            special: 0,
            second: 0,
            second_cost: 0,
            w: 0,
            v: vec![0; n],
            norm: 0,
            mst: 0,
        }
    }

    /// Sets `w`, `v` and `norm` from the tree and the extra edge.
    fn finish(&mut self, pi: &[i32]) {
        self.v.fill(-2);
        let mut mst = 0;
        for &x in &self.order[1..] {
            let x = x as usize;
            mst += i64::from(self.cost[x]);
            self.v[x] += 1;
            self.v[self.parent[x] as usize] += 1;
        }
        self.mst = mst;
        let w =
            mst + i64::from(self.second_cost) - 2 * pi.iter().map(|&p| i64::from(p)).sum::<i64>();
        self.v[self.special] += 1;
        self.v[self.second] += 1;
        self.w = w;
        self.norm = self.v.iter().map(|&v| i64::from(v) * i64::from(v)).sum();
    }

    /// Calls `f(a, b)` for every edge of the 1-tree.
    fn for_each_edge(&self, mut f: impl FnMut(usize, usize)) {
        for &x in &self.order[1..] {
            f(x as usize, self.parent[x as usize] as usize);
        }
        f(self.special, self.second);
    }
}

/// The minimum 1-tree of the complete graph under penalties `pi`, by
/// Prim's algorithm on the dense matrix: O(n^2) time, O(n) space. Each step
/// updates every node's cost to the tree with branch-free selects and takes
/// the minimum, then finds the first node with that cost; both loops
/// vectorise. The special node is the last node added (a leaf), with its
/// cheapest edge to a node other than its parent. Needs n >= 2.
///
/// Returns `None` if the budget runs out (deadline or interrupt) first.
pub(crate) fn one_tree_dense(
    dists: &DistMatrix,
    pi: &[i32],
    budget: &Budget<'_>,
) -> Option<OneTree> {
    /// Key of a node already in the tree; above every transformed cost.
    const DONE: i32 = i32::MAX;
    let n = dists.len();
    debug_assert!(n >= 2 && pi.len() == n);
    let mut tree = OneTree::empty(n);
    let mut parent = vec![0u32; n];
    let mut key: Vec<i32> = row(dists, 0)
        .iter()
        .zip(pi)
        .map(|(&d, &p)| tcost(d, pi[0], p))
        .collect();
    key[0] = DONE;
    tree.order.push(0);
    let mut last = 0;
    for step in 1..n {
        if step % 64 == 0 && budget.is_exceeded() {
            return None;
        }
        let r = row(dists, last);
        let pl = pi[last];
        let mut min = DONE;
        for (((k, p), &d), &pj) in key.iter_mut().zip(parent.iter_mut()).zip(r).zip(pi) {
            let c = PRECISION * i32::from(d) + pl + pj;
            let closer = c < *k && *k != DONE;
            *p = if closer { last as u32 } else { *p };
            *k = if closer { c } else { *k };
            min = min.min(*k);
        }
        let next = key.iter().position(|&k| k == min).expect("min is in key");
        tree.cost[next] = min;
        key[next] = DONE;
        tree.order.push(next as u32);
        last = next;
    }
    tree.parent = parent;
    let s = last;
    let sp = tree.parent[s] as usize;
    let r = row(dists, s);
    let (second, second_cost) = (0..n)
        .filter(|&j| j != s && j != sp)
        .map(|j| (j, tcost(r[j], pi[s], pi[j])))
        .min_by_key(|&(j, c)| (c, j))
        // n == 2: the special node has only one other node.
        .unwrap_or((sp, tree.cost[s]));
    tree.special = s;
    tree.second = second;
    tree.second_cost = second_cost;
    tree.finish(pi);
    Some(tree)
}

/// A sparse symmetric graph in compressed rows, with untransformed costs
/// `100 d` per directed entry.
pub(crate) struct Graph {
    start: Vec<u32>,
    adj: Vec<u32>,
    cost: Vec<i32>,
}

impl Graph {
    /// The union of the candidate lists `near` (`k` per node) and the edges
    /// of `tree`, made symmetric. It contains a spanning tree, so it is
    /// connected.
    pub(crate) fn new(dists: &DistMatrix, near: &[u32], k: usize, tree: &OneTree) -> Self {
        let n = dists.len();
        let mut keys: Vec<u64> = Vec::with_capacity(2 * (near.len() + n));
        let pair = |a: usize, b: usize| ((a as u64) << 32) | b as u64;
        if k > 0 {
            for (i, list) in near.chunks(k).enumerate() {
                for &j in list {
                    keys.push(pair(i, j as usize));
                    keys.push(pair(j as usize, i));
                }
            }
        }
        tree.for_each_edge(|a, b| {
            keys.push(pair(a, b));
            keys.push(pair(b, a));
        });
        Self::from_keys(dists, keys)
    }

    fn from_keys(dists: &DistMatrix, mut keys: Vec<u64>) -> Self {
        let n = dists.len();
        keys.sort_unstable();
        keys.dedup();
        keys.retain(|&key| (key >> 32) != (key & 0xFFFF_FFFF));
        let mut start = vec![0u32; n + 1];
        for &key in &keys {
            start[(key >> 32) as usize + 1] += 1;
        }
        for i in 0..n {
            start[i + 1] += start[i];
        }
        let adj: Vec<u32> = keys.iter().map(|&key| key as u32).collect();
        let cost = keys
            .iter()
            .map(|&key| PRECISION * dists.get((key >> 32) as usize, key as u32 as usize) as i32)
            .collect();
        Self { start, adj, cost }
    }

    /// Adds the edges of `tree`.
    fn add_tree(&mut self, dists: &DistMatrix, tree: &OneTree) {
        let n = dists.len();
        let mut keys: Vec<u64> = Vec::with_capacity(self.adj.len() + 2 * n);
        for i in 0..n {
            for &j in &self.adj[self.range(i)] {
                keys.push(((i as u64) << 32) | u64::from(j));
            }
        }
        tree.for_each_edge(|a, b| {
            keys.push(((a as u64) << 32) | b as u64);
            keys.push(((b as u64) << 32) | a as u64);
        });
        *self = Self::from_keys(dists, keys);
    }

    /// Number of directed entries (twice the number of edges).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn entries(&self) -> usize {
        self.adj.len()
    }

    #[inline(always)]
    fn range(&self, v: usize) -> std::ops::Range<usize> {
        self.start[v] as usize..self.start[v + 1] as usize
    }
}

/// Not in the heap.
const NONE: u32 = u32::MAX;

/// Prim's algorithm on a [`Graph`] with an indexed binary heap, O(E log n)
/// per tree. Keeps its buffers between calls.
pub(crate) struct SparsePrim {
    key: Vec<i32>,
    heap: Vec<u32>,
    loc: Vec<u32>,
    in_tree: Vec<bool>,
    degree: Vec<u32>,
}

impl SparsePrim {
    pub(crate) fn new(n: usize) -> Self {
        Self {
            key: vec![i32::MAX; n],
            heap: Vec::with_capacity(n),
            loc: vec![NONE; n],
            in_tree: vec![false; n],
            degree: vec![0; n],
        }
    }

    #[inline(always)]
    fn sift_up(&mut self, mut i: usize) {
        let v = self.heap[i];
        let kv = self.key[v as usize];
        while i > 0 {
            let p = (i - 1) / 2;
            let u = self.heap[p];
            if self.key[u as usize] <= kv {
                break;
            }
            self.heap[i] = u;
            self.loc[u as usize] = i as u32;
            i = p;
        }
        self.heap[i] = v;
        self.loc[v as usize] = i as u32;
    }

    #[inline(always)]
    fn sift_down(&mut self, mut i: usize) {
        let len = self.heap.len();
        let v = self.heap[i];
        let kv = self.key[v as usize];
        loop {
            let mut c = 2 * i + 1;
            if c >= len {
                break;
            }
            if c + 1 < len && self.key[self.heap[c + 1] as usize] < self.key[self.heap[c] as usize]
            {
                c += 1;
            }
            let u = self.heap[c];
            if self.key[u as usize] >= kv {
                break;
            }
            self.heap[i] = u;
            self.loc[u as usize] = i as u32;
            i = c;
        }
        self.heap[i] = v;
        self.loc[v as usize] = i as u32;
    }

    fn pop(&mut self) -> Option<usize> {
        let top = *self.heap.first()?;
        let last = self.heap.pop().expect("not empty");
        self.loc[top as usize] = NONE;
        if !self.heap.is_empty() {
            self.heap[0] = last;
            self.sift_down(0);
        }
        Some(top as usize)
    }

    /// The minimum 1-tree of `graph` under penalties `pi`, into `tree`. The
    /// special node is the leaf whose cheapest non-tree edge is the most
    /// expensive (LKH's rule, which maximises the bound among leaves).
    /// Returns false if the graph is not connected.
    pub(crate) fn one_tree(
        &mut self,
        graph: &Graph,
        dists: &DistMatrix,
        pi: &[i32],
        tree: &mut OneTree,
    ) -> bool {
        let n = pi.len();
        self.key.fill(i32::MAX);
        self.in_tree.fill(false);
        self.heap.clear();
        tree.order.clear();
        tree.parent[0] = 0;
        tree.cost[0] = 0;
        let mut v = 0;
        self.in_tree[0] = true;
        tree.order.push(0);
        loop {
            let pv = pi[v];
            for e in graph.range(v) {
                let w = graph.adj[e] as usize;
                if self.in_tree[w] {
                    continue;
                }
                let c = graph.cost[e] + pv + pi[w];
                if c < self.key[w] {
                    self.key[w] = c;
                    tree.parent[w] = v as u32;
                    if self.loc[w] == NONE {
                        self.heap.push(w as u32);
                        self.loc[w] = (self.heap.len() - 1) as u32;
                    }
                    self.sift_up(self.loc[w] as usize);
                }
            }
            let Some(u) = self.pop() else { break };
            self.in_tree[u] = true;
            tree.cost[u] = self.key[u];
            tree.order.push(u as u32);
            v = u;
        }
        if tree.order.len() < n {
            return false;
        }

        self.degree.fill(0);
        for &x in &tree.order[1..] {
            self.degree[x as usize] += 1;
            self.degree[tree.parent[x as usize] as usize] += 1;
        }
        let mut best: Option<(usize, usize)> = None;
        let mut max = i32::MIN;
        for v in 0..n {
            if self.degree[v] != 1 {
                continue;
            }
            let pv = if v == 0 { NONE } else { tree.parent[v] };
            let mut next = NONE;
            let mut next_cost = i32::MAX;
            for e in graph.range(v) {
                let w = graph.adj[e];
                if w == pv || tree.parent[w as usize] as usize == v {
                    continue;
                }
                let c = graph.cost[e] + pi[v] + pi[w as usize];
                if c < next_cost {
                    next_cost = c;
                    // This leaf cannot beat the best one any more.
                    if c <= max {
                        break;
                    }
                    next = w;
                }
            }
            if next != NONE && next_cost > max {
                max = next_cost;
                best = Some((v, next as usize));
            }
        }
        let (special, second) = best.unwrap_or_else(|| {
            // Every leaf's graph neighbours are its tree neighbours: scan
            // the matrix for the first leaf.
            let v = (0..n)
                .find(|&v| self.degree[v] == 1)
                .expect("a tree has leaves");
            let r = row(dists, v);
            let j = (0..n)
                .filter(|&j| {
                    j != v
                        && (v == 0 || j != tree.parent[v] as usize)
                        && (j == 0 || tree.parent[j] as usize != v)
                })
                .min_by_key(|&j| (tcost(r[j], pi[v], pi[j]), j))
                // n == 2: the only other node.
                .unwrap_or(1 - v.min(1));
            (v, j)
        });
        tree.special = special;
        tree.second = second;
        tree.second_cost = tcost(row(dists, special)[second], pi[special], pi[second]);
        tree.finish(pi);
        true
    }
}

// Subgradient ascent.

/// How the penalties move in each ascent iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Step {
    /// LKH's schedule: step `T` starting at 100, periods halving, the
    /// direction `(7 v + 3 v_last) / 10`.
    Lkh,
    /// Polyak's step `lambda (upper - w) / norm` along `v`, with `upper`
    /// a tour cost (times 100) and lambda halved after `period / 4`
    /// iterations without a better bound. For comparison only.
    #[cfg_attr(not(test), allow(dead_code))]
    Polyak { upper: i64 },
}

/// When the ascent stops.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Schedule {
    /// LKH's `INITIAL_PERIOD`: the length of the first period.
    pub(crate) initial_period: usize,
    /// Hard limit on the number of 1-trees.
    pub(crate) max_iterations: usize,
    /// The ascent stops (keeping the best penalties so far) at this time.
    pub(crate) deadline: Option<Instant>,
    /// Re-estimate the time per iteration after two iterations and fit the
    /// schedule (`initial_period`, at most LKH's `max(n / 2, 100)`) to the
    /// time left before `deadline`.
    pub(crate) adaptive: bool,
    pub(crate) step: Step,
}

/// The outcome of an ascent.
#[derive(Clone, Debug)]
pub(crate) struct AscentResult {
    /// The penalties with the best bound.
    pub(crate) pi: Vec<i32>,
    /// The best bound (times 100) and its norm.
    pub(crate) best_w: i64,
    pub(crate) best_norm: i64,
    /// Number of sparse 1-trees computed.
    pub(crate) iterations: usize,
    /// The best bound after each iteration.
    pub(crate) history: Vec<i64>,
    /// How often a dense check found the sparse graph lacking.
    pub(crate) repairs: usize,
}

/// Subgradient optimisation of the node penalties (`Ascent.c` in LKH),
/// starting from `tree0`, the dense minimum 1-tree at pi = 0. Each
/// iteration computes a minimum 1-tree on a fixed sparse graph, the
/// candidate lists `near` (`k` per node) plus the edges of `tree0`, which
/// costs O(E log n) instead of O(n^2).
///
/// Returns `None` if the budget runs out (deadline or interrupt), which
/// is checked every iteration; reaching `schedule.deadline` just ends the
/// ascent early.
pub(crate) fn ascent(
    dists: &DistMatrix,
    near: &[u32],
    k: usize,
    tree0: &OneTree,
    schedule: &Schedule,
    budget: &Budget<'_>,
) -> Option<AscentResult> {
    let n = dists.len();
    let mut out = AscentResult {
        pi: vec![0; n],
        best_w: tree0.w,
        best_norm: tree0.norm,
        iterations: 0,
        history: Vec::new(),
        repairs: 0,
    };
    if tree0.norm == 0 {
        return Some(out);
    }
    let started = Instant::now();
    let mut graph = Graph::new(dists, near, k, tree0);
    let mut prim = SparsePrim::new(n);
    let mut tree = OneTree::empty(n);
    let mut pi = vec![0i32; n];
    let mut v = tree0.v.clone();
    let mut last_v = v.clone();
    let mut w_now = tree0.w;
    let mut norm = tree0.norm;
    let mut w0 = tree0.w;
    let mut p0 = schedule.initial_period.max(1);
    let mut period = p0;
    let mut t = i64::from(PRECISION);
    let mut initial_phase = true;
    // Polyak only.
    let mut lambda = 2.0_f64;
    let mut since_better = 0;
    let polyak = matches!(schedule.step, Step::Polyak { .. });
    // The spanning tree cost at the best penalties, and the iterations
    // since the last dense check.
    let mut best_mst = tree0.mst;
    let mut since_check = 0;
    let check_every = (n / DENSE_CHECK_NODES_PER_ITERATION).max(DENSE_CHECK_MIN_ITERATIONS);

    'outer: while period > 0 && t > 0 && norm != 0 {
        let mut p = 1;
        while t > 0 && p <= period && norm != 0 {
            if out.iterations >= schedule.max_iterations
                || schedule.deadline.is_some_and(|d| Instant::now() >= d)
            {
                break 'outer;
            }
            if budget.is_exceeded() {
                return None;
            }
            match schedule.step {
                Step::Lkh => {
                    for ((p, &v), last) in pi.iter_mut().zip(&v).zip(last_v.iter_mut()) {
                        if v != 0 {
                            let delta = t * (7 * i64::from(v) + 3 * i64::from(*last)) / 10;
                            *p = (i64::from(*p) + delta).clamp(-PI_LIMIT, PI_LIMIT) as i32;
                        }
                        *last = v;
                    }
                }
                Step::Polyak { upper } => {
                    let step = lambda * (upper - w_now).max(1) as f64 / norm as f64;
                    for (p, &v) in pi.iter_mut().zip(&v) {
                        let delta = (step * f64::from(v)).round() as i64;
                        *p = (i64::from(*p) + delta).clamp(-PI_LIMIT, PI_LIMIT) as i32;
                    }
                }
            }
            let connected = prim.one_tree(&graph, dists, &pi, &mut tree);
            debug_assert!(connected, "the graph contains a spanning tree");
            out.iterations += 1;
            since_check += 1;
            let (mut w, mut nrm, mut mst) = (tree.w, tree.norm, tree.mst);
            v.copy_from_slice(&tree.v);
            // A bound far above the first one suggests that the sparse
            // graph misses edges of the true minimum 1-tree: add them.
            if (w > out.best_w || (w == out.best_w && nrm < out.best_norm)) && w - w0 > w0.abs() {
                let dense = one_tree_dense(dists, &pi, budget)?;
                graph.add_tree(dists, &dense);
                (w, nrm, mst) = (dense.w, dense.norm, dense.mst);
                v.copy_from_slice(&dense.v);
                w0 = w;
            }
            if w > out.best_w || (w == out.best_w && nrm < out.best_norm) {
                out.best_w = w;
                out.best_norm = nrm;
                best_mst = mst;
                out.pi.copy_from_slice(&pi);
                since_better = 0;
                if initial_phase && nrm > 0 {
                    t = (2 * t).min(PI_LIMIT);
                }
                if p == period {
                    period = (2 * period).min(p0);
                }
            } else {
                since_better += 1;
                if initial_phase && p > period / 2 {
                    initial_phase = false;
                    p = 0;
                    t = 3 * t / 4;
                }
            }
            w_now = w;
            norm = nrm;
            out.history.push(out.best_w);
            if polyak && since_better >= (schedule.initial_period / 4).max(4) {
                lambda /= 2.0;
                since_better = 0;
            }
            if schedule.adaptive
                && out.iterations == 2
                && let Some(deadline) = schedule.deadline
            {
                let per_iteration = started.elapsed().as_secs_f64() / 2.0;
                let left = deadline.saturating_duration_since(Instant::now());
                let fits = (left.as_secs_f64() / per_iteration.max(1e-9)) as usize;
                p0 = (fits / 2).clamp(MIN_INITIAL_PERIOD, (n / 2).max(100));
                period = p0;
            }
            p += 1;
            // Now and then, check the best penalties on the complete graph.
            // The sparse graph holds only the edges of the first tree
            // between the clusters of a clustered instance; once the
            // penalties move, the true minimum tree may connect them
            // elsewhere, and the sparse "bound" then overestimates and
            // steers the ascent wrongly. A cheaper dense tree means missing
            // edges: add them and re-rate the best penalties. One dense
            // tree costs about n / 400 sparse ones, so checking every n / 40
            // iterations costs about 10 %.
            if since_check >= check_every {
                since_check = 0;
                let dense = one_tree_dense(dists, &out.pi, budget)?;
                if dense.mst < best_mst {
                    out.repairs += 1;
                    graph.add_tree(dists, &dense);
                    let connected = prim.one_tree(&graph, dists, &out.pi, &mut tree);
                    debug_assert!(connected);
                    (out.best_w, out.best_norm, best_mst) = (tree.w, tree.norm, tree.mst);
                }
            }
        }
        if polyak {
            // Polyak's step ignores the periods: run until the limits.
            if norm == 0 || lambda < 1e-4 {
                break;
            }
            continue;
        }
        period /= 2;
        t /= 2;
    }
    Some(out)
}

// Alpha values.

/// The tree of a [`OneTree`] indexed by position in Prim order (parents
/// first), so the per-node pass over the tree reads memory in order.
struct TreeIndex {
    order: Vec<u32>,
    pos: Vec<u32>,
    parent_pos: Vec<u32>,
    parent_cost: Vec<i32>,
    pi_by_pos: Vec<i32>,
    special: usize,
    special_parent: usize,
    second: usize,
    second_cost: i32,
}

/// Per-thread buffers for the alpha pass.
struct BetaScratch {
    /// After a pass for node i: the most expensive edge on the tree path
    /// from i to the node at each position.
    beta: Vec<i32>,
    /// Positions on the path from i to the root.
    mark: Vec<u32>,
    alphas: Vec<i32>,
    mins: Vec<i32>,
    scratch: Vec<i32>,
    best: Vec<u128>,
}

impl BetaScratch {
    fn new(n: usize) -> Self {
        Self {
            beta: vec![0; n],
            mark: vec![NONE; n],
            alphas: vec![0; n],
            mins: Vec::with_capacity(n.div_ceil(BLOCK)),
            scratch: Vec::with_capacity(n.div_ceil(BLOCK)),
            best: Vec::new(),
        }
    }
}

impl TreeIndex {
    fn new(tree: &OneTree, pi: &[i32]) -> Self {
        let n = tree.order.len();
        let order = tree.order.clone();
        let mut pos = vec![0u32; n];
        for (p, &v) in order.iter().enumerate() {
            pos[v as usize] = p as u32;
        }
        let parent_pos = order
            .iter()
            .map(|&v| pos[tree.parent[v as usize] as usize])
            .collect();
        let parent_cost = order.iter().map(|&v| tree.cost[v as usize]).collect();
        let pi_by_pos = order.iter().map(|&v| pi[v as usize]).collect();
        Self {
            order,
            pos,
            parent_pos,
            parent_cost,
            pi_by_pos,
            special: tree.special,
            special_parent: tree.parent[tree.special] as usize,
            second: tree.second,
            second_cost: tree.second_cost,
        }
    }

    /// Fills `s.alphas` (by position) with alpha(i, j) for every j, and
    /// `i32::MAX` at i itself. O(n).
    fn alphas(&self, dists: &DistMatrix, pi: &[i32], i: usize, s: &mut BetaScratch) {
        let n = self.order.len();
        let r = row(dists, i);
        let pi_i = pi[i];
        let (sp, second) = (self.special_parent, self.second);
        if i == self.special {
            // Adding (s, j) to the 1-tree removes s's more expensive edge.
            for (p, a) in s.alphas.iter_mut().enumerate() {
                let j = self.order[p] as usize;
                *a = if j == sp || j == second {
                    0
                } else {
                    tcost(r[j], pi_i, self.pi_by_pos[p]) - self.second_cost
                };
            }
            s.alphas[self.pos[i] as usize] = i32::MAX;
            return;
        }
        // beta for i and its ancestors: walk up from i.
        let stamp = i as u32;
        let start = self.pos[i] as usize;
        s.beta[start] = i32::MIN;
        s.mark[start] = stamp;
        let mut t = start;
        while t != 0 {
            let p = self.parent_pos[t] as usize;
            s.beta[p] = s.beta[t].max(self.parent_cost[t]);
            s.mark[p] = stamp;
            t = p;
        }
        // Everyone else: the path from i passes through the parent, which
        // comes earlier in the order. The special node is a leaf, so it is
        // never inside the path between two other nodes.
        let (beta, mark, alphas) = (&mut s.beta[..], &s.mark[..], &mut s.alphas[..]);
        alphas[0] = tcost(r[self.order[0] as usize], pi_i, self.pi_by_pos[0]).wrapping_sub(beta[0]);
        for p in 1..n {
            if mark[p] != stamp {
                beta[p] = beta[self.parent_pos[p] as usize].max(self.parent_cost[p]);
            }
            alphas[p] =
                tcost(r[self.order[p] as usize], pi_i, self.pi_by_pos[p]).wrapping_sub(beta[p]);
        }
        alphas[start] = i32::MAX;
        let sv = self.special;
        alphas[self.pos[sv] as usize] = if i == sp || i == second {
            0
        } else {
            tcost(r[sv], pi_i, pi[sv]) - self.second_cost
        };
    }

    /// The `ka` nodes with the smallest alpha(i, .), ties to the shorter
    /// edge and then the lower index, in that order, into `out`.
    fn select(
        &self,
        dists: &DistMatrix,
        i: usize,
        ka: usize,
        s: &mut BetaScratch,
        out: &mut [u32],
    ) {
        let r = row(dists, i);
        block_minima(&s.alphas, &mut s.mins);
        let bound = kth_smallest(&s.mins, ka, &mut s.scratch);
        s.best.clear();
        for (b, &m) in s.mins.iter().enumerate() {
            if m > bound {
                continue;
            }
            for (p, &a) in s.alphas.iter().enumerate().skip(b * BLOCK).take(BLOCK) {
                if a <= bound && a != i32::MAX {
                    debug_assert!(a >= 0, "alpha({i}, {}) = {a}", self.order[p]);
                    let j = self.order[p];
                    let key = (u128::from(a.max(0) as u32) << 64)
                        | (u128::from(r[j as usize]) << 32)
                        | u128::from(j);
                    push_best(&mut s.best, ka, key);
                }
            }
        }
        for (o, &key) in out.iter_mut().zip(&s.best) {
            *o = key as u32;
        }
    }
}

/// For each node, the `ka` other nodes with the smallest alpha values on
/// `tree` (a minimum 1-tree under penalties `pi`), smallest first (ties to
/// the shorter edge, then the lower index). One O(n) pass over the tree
/// per node, so O(n^2) in total, split by rows over up to `threads`
/// threads. Returns `None` if the budget runs out first.
pub(crate) fn alpha_lists(
    dists: &DistMatrix,
    tree: &OneTree,
    pi: &[i32],
    ka: usize,
    budget: &Budget<'_>,
    threads: usize,
) -> Option<Vec<u32>> {
    let n = dists.len();
    let ka = ka.min(n.saturating_sub(1));
    if ka == 0 {
        return Some(Vec::new());
    }
    let index = TreeIndex::new(tree, pi);
    split_rows(
        n,
        ka,
        threads,
        ALPHA_PARALLEL_MIN_ENTRIES,
        budget,
        |first, out, budget| {
            let mut s = BetaScratch::new(n);
            for (r, out) in out.chunks_mut(ka).enumerate() {
                if r % 16 == 0 && budget.is_exceeded() {
                    return false;
                }
                index.alphas(dists, pi, first + r, &mut s);
                index.select(dists, first + r, ka, &mut s, out);
            }
            true
        },
    )
}

// List layouts and the budget gate.

/// For each node: its first `base` entries of `near` (`kn` per node,
/// nearest first) together with its `ka` entries of `alpha`, then further
/// entries of `near` until every node has as many as the largest such
/// union, sorted by (distance, index). Returns the common length and the
/// lists. `near` must hold `base + ka` (or all other) nodes per node, so
/// that the padding succeeds.
pub(crate) fn union_lists(
    dists: &DistMatrix,
    near: &[u32],
    kn: usize,
    base: usize,
    alpha: &[u32],
    ka: usize,
) -> (usize, Vec<u32>) {
    let n = dists.len();
    if kn == 0 {
        return (0, Vec::new());
    }
    let base = base.min(kn);
    let mut sets: Vec<Vec<u32>> = Vec::with_capacity(n);
    let mut k = 0;
    for i in 0..n {
        let mut set = near[i * kn..i * kn + base].to_vec();
        for &j in &alpha[i * ka..(i + 1) * ka] {
            if !set.contains(&j) {
                set.push(j);
            }
        }
        k = k.max(set.len());
        sets.push(set);
    }
    let mut data = Vec::with_capacity(n * k);
    for (i, mut set) in sets.into_iter().enumerate() {
        for &j in &near[i * kn + base..(i + 1) * kn] {
            if set.len() == k {
                break;
            }
            if !set.contains(&j) {
                set.push(j);
            }
        }
        debug_assert_eq!(set.len(), k, "near is too short to pad node {i}");
        let r = row(dists, i);
        set.sort_unstable_by_key(|&j| (r[j as usize], j));
        data.extend_from_slice(&set);
    }
    (k, data)
}

/// For each node: its `ka` entries of `alpha`, then its entries of `near`
/// (`k` per node) not already there until there are `k`; sorted by
/// (distance, index).
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn fill_up(
    dists: &DistMatrix,
    alpha: &[u32],
    ka: usize,
    near: &[u32],
    k: usize,
) -> Vec<u32> {
    let n = dists.len();
    let mut data = Vec::with_capacity(n * k);
    let mut set: Vec<u32> = Vec::with_capacity(k);
    for i in 0..n {
        set.clear();
        set.extend_from_slice(&alpha[i * ka..(i + 1) * ka]);
        for &j in &near[i * k..(i + 1) * k] {
            if set.len() == k {
                break;
            }
            if !set.contains(&j) {
                set.push(j);
            }
        }
        let r = row(dists, i);
        set.sort_unstable_by_key(|&j| (r[j as usize], j));
        data.extend_from_slice(&set);
    }
    data
}

/// The share of the time left (after the nearest lists) that `auto` may
/// spend on the ascent and the alpha lists. 0 makes `auto` the same as
/// `nearest`. On the dev instances 0.05, 0.10 and 0.20 were within noise
/// of each other with the current LK step.
pub(crate) const ASCENT_MAX_SHARE: f64 = 0.10;
/// `auto` runs the ascent only if at least this many iterations fit in
/// that share; otherwise it keeps the nearest lists. On the dev instances,
/// 30 iterations of the shortened schedule put 98.8 % of the edges of a
/// near-optimal tour among the 5 alpha-nearest, against 97.5 % at pi = 0,
/// 99.2 % after LKH's whole schedule and 95.9 % for the 5 nearest.
pub(crate) const ASCENT_MIN_ITERATIONS: usize = 30;
/// The ascent checks its sparse graph against a dense 1-tree every
/// `max(n / DENSE_CHECK_NODES_PER_ITERATION, DENSE_CHECK_MIN_ITERATIONS)`
/// iterations.
const DENSE_CHECK_NODES_PER_ITERATION: usize = 40;
const DENSE_CHECK_MIN_ITERATIONS: usize = 20;
/// The shortest first period of a shortened (`auto`) schedule.
const MIN_INITIAL_PERIOD: usize = 16;
/// `alpha` caps LKH's first period (`max(n / 2, 100)`) at this, and
/// stops the ascent after this share of the time left.
const ALPHA_MAX_INITIAL_PERIOD: usize = 1000;
const ALPHA_MAX_SHARE: f64 = 0.5;
/// Cost model of the gate. Before anything else runs, a dense 1-tree is
/// estimated at `DENSE_TREE_COST * u`, where `u` is the single-thread time
/// of the nearest-neighbour pass; `auto` builds the first tree only if
/// that is at most `FIRST_TREE_SHARE` of its allowance. The real time `t`
/// of that tree then calibrates the rest: the last dense tree costs `t`,
/// the alpha pass `ALPHA_PASS_PER_TREE * t / threads` and an iteration
/// `ITERATION_NODES_PER_TREE / n * t` (measured on the dev instances, one
/// thread: n = 783..4461; `u` overestimates small inputs, `t` does not).
const DENSE_TREE_COST: f64 = 1.2;
const FIRST_TREE_SHARE: f64 = 0.25;
const ALPHA_PASS_PER_TREE: f64 = 4.5;
const ITERATION_NODES_PER_TREE: f64 = 300.0;

/// What the candidate builder does after the nearest lists.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Plan {
    /// Keep the nearest lists.
    Nearest,
    /// Run the ascent with this first period, for at most `time` (`None`:
    /// no limit), then build alpha lists. `adaptive` (`auto`): `time` is the
    /// allowance for everything after the nearest lists, and the schedule
    /// is fitted to it with [`ascent_budget`] once the first 1-tree is
    /// timed, and again after two iterations.
    Ascent {
        initial_period: usize,
        time: Option<Duration>,
        adaptive: bool,
    },
}

/// Decides whether and how long to run the ascent. `nearest_time` is the
/// wall time of the nearest-neighbour pass on `threads` threads and
/// `time_left` the time left after it (`None`: no deadline). Pure.
pub(crate) fn ascent_plan(
    mode: Candidates,
    n: usize,
    threads: usize,
    nearest_time: Duration,
    time_left: Option<Duration>,
) -> Plan {
    let lkh_period = (n / 2).max(100);
    match (mode, time_left) {
        (Candidates::Nearest, _) => Plan::Nearest,
        (Candidates::Alpha, _) | (Candidates::Auto, None) => Plan::Ascent {
            initial_period: lkh_period.min(ALPHA_MAX_INITIAL_PERIOD),
            time: time_left.map(|t| t.mul_f64(ALPHA_MAX_SHARE)),
            adaptive: false,
        },
        (Candidates::Auto, Some(left)) => {
            let u = nearest_time.as_secs_f64()
                * threads_for(n, threads.max(1), PARALLEL_MIN_ENTRIES) as f64;
            let allowance = ASCENT_MAX_SHARE * left.as_secs_f64();
            if DENSE_TREE_COST * u >= FIRST_TREE_SHARE * allowance {
                return Plan::Nearest;
            }
            Plan::Ascent {
                initial_period: lkh_period,
                time: Some(Duration::from_secs_f64(allowance)),
                adaptive: true,
            }
        }
    }
}

/// The schedule for `auto` once the first dense 1-tree took `tree_time`:
/// with `left` of the allowance remaining, the first period and the time
/// for the ascent itself, leaving room for the last dense tree and the
/// alpha pass on `threads` threads. `None` if fewer than
/// `ASCENT_MIN_ITERATIONS` iterations fit. Pure.
pub(crate) fn ascent_budget(
    n: usize,
    threads: usize,
    tree_time: Duration,
    left: Duration,
) -> Option<(usize, Duration)> {
    let t = tree_time.as_secs_f64();
    let fixed = t * (1.0 + ALPHA_PASS_PER_TREE / threads.max(1) as f64);
    let per_iteration = (t * ITERATION_NODES_PER_TREE / n as f64).max(1e-9);
    let room = left.as_secs_f64() - fixed;
    let min_iterations = ASCENT_MIN_ITERATIONS as f64;
    if room < min_iterations * per_iteration {
        return None;
    }
    let iterations = (room / per_iteration).min(1e9) as usize;
    let period = (iterations / 2).clamp(MIN_INITIAL_PERIOD, (n / 2).max(100));
    Some((period, Duration::from_secs_f64(room)))
}

/// Candidate lists for the local search.
pub(crate) struct Lists {
    /// `k` per node, sorted by (distance, index).
    pub(crate) k: usize,
    pub(crate) data: Vec<u32>,
    /// `ka` per node: the alpha-nearest, smallest alpha first, or the
    /// nearest when no ascent ran.
    pub(crate) ka: usize,
    pub(crate) alpha: Vec<u32>,
    /// Whether `alpha` comes from a subgradient ascent (false: the nearest).
    pub(crate) from_ascent: bool,
}

/// The first `width` of every `k`-long row of `lists`.
fn prefix(lists: &[u32], k: usize, width: usize) -> Vec<u32> {
    if k == 0 {
        return Vec::new();
    }
    lists
        .chunks(k)
        .flat_map(|list| &list[..width])
        .copied()
        .collect()
}

/// Builds the candidate lists: the `k` nearest neighbours of every node,
/// and for `Alpha` (and for `Auto` when the budget allows it, see
/// [`ascent_plan`] and [`ascent_budget`]) the union with the `ka`
/// alpha-nearest after a subgradient ascent, padded to a common length with
/// further nearest neighbours (at most `k + ka`). The alpha-ordered lists
/// are kept as well (the `ka` nearest without an ascent).
///
/// Returns `None` only if the budget runs out before the nearest lists are
/// complete; if it runs out later, the nearest lists are returned.
pub(crate) fn candidate_lists(
    dists: &DistMatrix,
    budget: &Budget<'_>,
    threads: usize,
    mode: Candidates,
    k: usize,
    ka: usize,
) -> Option<Lists> {
    let n = dists.len();
    let k = k.min(n.saturating_sub(1));
    let ka = ka.min(k);
    let kn = if mode == Candidates::Nearest {
        k
    } else {
        (k + ka).min(n.saturating_sub(1))
    };
    let started = Instant::now();
    let near = nearest(dists, kn, budget, threads)?;
    let nearest_time = started.elapsed();
    let fallback = || Lists {
        k,
        data: prefix(&near, kn, k),
        ka,
        alpha: prefix(&near, kn, ka),
        from_ascent: false,
    };
    if n < 3 || ka == 0 {
        return Some(fallback());
    }
    let now = Instant::now();
    let time_left = budget.deadline().map(|d| d.saturating_duration_since(now));
    let Plan::Ascent {
        mut initial_period,
        time,
        adaptive,
    } = ascent_plan(mode, n, threads, nearest_time, time_left)
    else {
        return Some(fallback());
    };
    let mut deadline = time.and_then(|t| now.checked_add(t));
    let tree0 = one_tree_dense(dists, &vec![0; n], budget);
    let Some(tree0) = tree0 else {
        return Some(fallback());
    };
    if adaptive && let Some(end) = deadline {
        let after = Instant::now();
        let left = end.saturating_duration_since(after);
        let Some((period, time)) = ascent_budget(n, threads, after - now, left) else {
            return Some(fallback());
        };
        initial_period = period;
        deadline = after.checked_add(time);
    }
    let schedule = Schedule {
        initial_period,
        max_iterations: usize::MAX,
        deadline,
        adaptive,
        step: Step::Lkh,
    };
    let near_k = prefix(&near, kn, k);
    let Some(alpha) = alpha_after_ascent(dists, &near_k, k, tree0, &schedule, ka, budget, threads)
    else {
        return Some(fallback());
    };
    let (k_union, data) = union_lists(dists, &near, kn, k, &alpha, ka);
    Some(Lists {
        k: k_union,
        data,
        ka,
        alpha,
        from_ascent: true,
    })
}

/// The ascent from `tree0` (the dense 1-tree at pi = 0), the dense 1-tree
/// at the best penalties, and its `ka` alpha-nearest lists. `near` (`k` per
/// node) is the ascent's sparse graph. `None` if the budget runs out.
#[allow(clippy::too_many_arguments)]
pub(crate) fn alpha_after_ascent(
    dists: &DistMatrix,
    near: &[u32],
    k: usize,
    tree0: OneTree,
    schedule: &Schedule,
    ka: usize,
    budget: &Budget<'_>,
    threads: usize,
) -> Option<Vec<u32>> {
    let result = ascent(dists, near, k, &tree0, schedule, budget)?;
    let tree = if result.iterations == 0 {
        tree0
    } else {
        one_tree_dense(dists, &result.pi, budget)?
    };
    alpha_lists(dists, &tree, &result.pi, ka, budget, threads)
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

    use super::super::{solve_tsp_exact, tour_cost};

    fn random_pi(n: usize, spread: i32, seed: u64) -> Vec<i32> {
        let mut rng = fastrand::Rng::with_seed(seed);
        (0..n).map(|_| rng.i32(-spread..=spread)).collect()
    }

    fn d_t(d: &DistMatrix, pi: &[i32], i: usize, j: usize) -> i64 {
        100 * d.get(i, j) + i64::from(pi[i]) + i64::from(pi[j])
    }

    fn find(c: &mut [usize], x: usize) -> usize {
        let mut r = x;
        while c[r] != r {
            r = c[r];
        }
        let mut x = x;
        while c[x] != r {
            let next = c[x];
            c[x] = r;
            x = next;
        }
        r
    }

    /// Weight of a minimum spanning tree of all nodes but `skip`, with the
    /// edge `forced` in it, by Kruskal over the sorted `edges`.
    fn kruskal(
        n: usize,
        edges: &[(i64, usize, usize)],
        skip: usize,
        forced: Option<(i64, usize, usize)>,
    ) -> i64 {
        let mut comp: Vec<usize> = (0..n).collect();
        let mut total = 0;
        if let Some((w, a, b)) = forced {
            comp[a] = b;
            total += w;
        }
        for &(w, a, b) in edges {
            if a == skip || b == skip {
                continue;
            }
            let (ra, rb) = (find(&mut comp, a), find(&mut comp, b));
            if ra != rb {
                comp[ra] = rb;
                total += w;
            }
        }
        total
    }

    /// alpha(i, j) for every pair by definition: the cost of the minimum
    /// 1-tree (special node `s`) forced to contain (i, j), minus the cost of
    /// the minimum 1-tree. O(n^4); small n only.
    fn alpha_oracle(d: &DistMatrix, pi: &[i32], s: usize) -> Vec<i64> {
        let n = d.len();
        let mut edges: Vec<(i64, usize, usize)> = (0..n)
            .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
            .map(|(i, j)| (d_t(d, pi, i, j), i, j))
            .collect();
        edges.sort_unstable();
        let mut s_edges: Vec<(i64, usize)> = (0..n)
            .filter(|&j| j != s)
            .map(|j| (d_t(d, pi, s, j), j))
            .collect();
        s_edges.sort_unstable();
        let two = |skip: Option<usize>| -> i64 {
            s_edges
                .iter()
                .filter(|&&(_, j)| Some(j) != skip)
                .take(2)
                .map(|e| e.0)
                .sum()
        };
        let base = kruskal(n, &edges, s, None) + two(None);
        let mut out = vec![0; n * n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let forced = if i == s || j == s {
                    let other = if i == s { j } else { i };
                    let rest = s_edges
                        .iter()
                        .find(|&&(_, x)| x != other)
                        .map_or(0, |e| e.0);
                    kruskal(n, &edges, s, None) + d_t(d, pi, s, other) + rest
                } else {
                    kruskal(n, &edges, s, Some((d_t(d, pi, i, j), i, j))) + two(None)
                };
                out[i * n + j] = forced - base;
            }
        }
        out
    }

    /// All alpha values of `tree`, by node pairs.
    fn alpha_matrix(d: &DistMatrix, tree: &OneTree, pi: &[i32]) -> Vec<i64> {
        let n = d.len();
        let index = TreeIndex::new(tree, pi);
        let mut s = BetaScratch::new(n);
        let mut out = vec![0; n * n];
        for i in 0..n {
            index.alphas(d, pi, i, &mut s);
            for (p, &a) in s.alphas.iter().enumerate() {
                let j = index.order[p] as usize;
                if j != i {
                    out[i * n + j] = i64::from(a);
                }
            }
        }
        out
    }

    #[test]
    fn alpha_with_penalties_matches_forced_edge_oracle() {
        let cases = [
            (euclidean(2, 1), 0),
            (euclidean(3, 1), 0),
            (euclidean(3, 2), 500),
            (euclidean(25, 4), 0),
            (euclidean(25, 4), 3000),
            (euclidean(60, 9), 2000),
            (tied(30, 3, 5), 0),
            (tied(30, 3, 6), 50),
        ];
        for (seed, (d, spread)) in cases.into_iter().enumerate() {
            let n = d.len();
            let pi = random_pi(n, spread, seed as u64);
            let tree = one_tree_dense(&d, &pi, &Budget::unlimited()).unwrap();
            let expected = alpha_oracle(&d, &pi, tree.special);
            let got = alpha_matrix(&d, &tree, &pi);
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        assert!(expected[i * n + j] >= 0);
                        assert_eq!(
                            got[i * n + j],
                            expected[i * n + j],
                            "n={n} spread={spread} i={i} j={j} special={}",
                            tree.special
                        );
                    }
                }
            }
            // The 1-tree's own cost matches the oracle's base (with n = 2
            // the 1-tree is the one edge twice).
            if n == 2 {
                assert_eq!(
                    tree.w,
                    2 * d_t(&d, &pi, 0, 1) - 2 * i64::from(pi[0] + pi[1])
                );
                continue;
            }
            let sum: i64 = pi.iter().map(|&p| i64::from(p)).sum();
            let mut edges: Vec<(i64, usize, usize)> = (0..n)
                .flat_map(|i| (i + 1..n).map(move |j| (i, j)))
                .map(|(i, j)| (d_t(&d, &pi, i, j), i, j))
                .collect();
            edges.sort_unstable();
            let s = tree.special;
            let mut s_edges: Vec<i64> = (0..n)
                .filter(|&j| j != s)
                .map(|j| d_t(&d, &pi, s, j))
                .collect();
            s_edges.sort_unstable();
            let one_tree = kruskal(n, &edges, s, None) + s_edges.iter().take(2).sum::<i64>();
            assert_eq!(tree.w, one_tree - 2 * sum, "n={n}");
        }
    }

    #[test]
    fn alpha_lists_take_the_smallest_alphas_in_order() {
        for (d, ka) in [
            (euclidean(40, 3), 5),
            (tied(50, 4, 9), 6),
            (euclidean(4, 2), 5),
        ] {
            let n = d.len();
            let pi = random_pi(n, 800, 7);
            let tree = one_tree_dense(&d, &pi, &Budget::unlimited()).unwrap();
            let alphas = alpha_matrix(&d, &tree, &pi);
            let lists = alpha_lists(&d, &tree, &pi, ka, &Budget::unlimited(), 1).unwrap();
            let ka = ka.min(n - 1);
            for i in 0..n {
                let mut all: Vec<(i64, i64, u32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (alphas[i * n + j], d.get(i, j), j as u32))
                    .collect();
                all.sort_unstable();
                let expected: Vec<u32> = all[..ka].iter().map(|t| t.2).collect();
                assert_eq!(&lists[i * ka..(i + 1) * ka], &expected[..], "i={i}");
            }
        }
    }

    #[test]
    fn alpha_lists_in_parallel_equal_serial() {
        let d = euclidean(1500, 4);
        let pi = random_pi(1500, 1000, 1);
        let tree = one_tree_dense(&d, &pi, &Budget::unlimited()).unwrap();
        let one = alpha_lists(&d, &tree, &pi, 5, &Budget::unlimited(), 1).unwrap();
        for threads in [2, 3, 4] {
            let many = alpha_lists(&d, &tree, &pi, 5, &Budget::unlimited(), threads).unwrap();
            assert_eq!(many, one, "{threads} threads");
        }
    }

    /// Edges of the spanning tree part, as sorted pairs.
    fn tree_edges(tree: &OneTree) -> Vec<(u32, u32)> {
        let mut edges: Vec<(u32, u32)> = tree.order[1..]
            .iter()
            .map(|&v| {
                let p = tree.parent[v as usize];
                (v.min(p), v.max(p))
            })
            .collect();
        edges.sort_unstable();
        edges
    }

    #[test]
    fn sparse_one_tree_equals_dense_on_a_complete_graph() {
        for (n, spread, seed) in [
            (2, 0, 1),
            (3, 100, 2),
            (30, 0, 3),
            (80, 2000, 4),
            (150, 500, 5),
        ] {
            let d = euclidean(n, seed);
            // Random penalties make every spanning tree weight distinct.
            let pi = random_pi(n, spread, seed);
            let dense = one_tree_dense(&d, &pi, &Budget::unlimited()).unwrap();
            let all: Vec<u32> = (0..n)
                .flat_map(|i| (0..n).filter(move |&j| j != i).map(|j| j as u32))
                .collect();
            let graph = Graph::new(&d, &all, n - 1, &dense);
            assert_eq!(graph.entries(), n * (n - 1));
            let mut sparse = OneTree::empty(n);
            assert!(SparsePrim::new(n).one_tree(&graph, &d, &pi, &mut sparse));
            if spread > 0 || n <= 3 {
                assert_eq!(tree_edges(&sparse), tree_edges(&dense), "n={n}");
            }
            let weight = |t: &OneTree| -> i64 {
                t.order[1..]
                    .iter()
                    .map(|&v| i64::from(t.cost[v as usize]))
                    .sum()
            };
            assert_eq!(weight(&sparse), weight(&dense), "n={n}");
            // The sparse tree takes the leaf with the most expensive second
            // edge, so its bound is at least the dense one's.
            let mut degree = vec![0; n];
            for (a, b) in tree_edges(&sparse) {
                degree[a as usize] += 1;
                degree[b as usize] += 1;
            }
            let best_second = (0..n)
                .filter(|&v| degree[v] == 1)
                .filter_map(|v| {
                    let tree_nb = |j: usize| {
                        (v != 0 && j == sparse.parent[v] as usize)
                            || (j != 0 && sparse.parent[j] as usize == v)
                    };
                    (0..n)
                        .filter(|&j| j != v && !tree_nb(j))
                        .map(|j| d_t(&d, &pi, v, j))
                        .min()
                })
                .max()
                .unwrap_or_else(|| i64::from(dense.second_cost));
            let sum: i64 = pi.iter().map(|&p| i64::from(p)).sum();
            assert_eq!(sparse.w, weight(&sparse) + best_second - 2 * sum, "n={n}");
            assert!(sparse.w >= dense.w);
            assert_eq!(sparse.v.iter().sum::<i32>(), 0);
        }
    }

    fn full_schedule(n: usize) -> Schedule {
        Schedule {
            initial_period: (n / 2).max(100),
            max_iterations: usize::MAX,
            deadline: None,
            adaptive: false,
            step: Step::Lkh,
        }
    }

    fn run_ascent(d: &DistMatrix, schedule: &Schedule) -> (OneTree, AscentResult) {
        let n = d.len();
        let near = nearest(d, 20, &Budget::unlimited(), 1).unwrap();
        let tree0 = one_tree_dense(d, &vec![0; n], &Budget::unlimited()).unwrap();
        let k = 20.min(n - 1);
        let result = ascent(d, &near, k, &tree0, schedule, &Budget::unlimited()).unwrap();
        (tree0, result)
    }

    #[test]
    fn ascent_bound_is_a_lower_bound_and_improves() {
        for n in 5..=16 {
            let d = euclidean(n, n as u64);
            let (tree0, result) = run_ascent(&d, &full_schedule(n));
            let tour = solve_tsp_exact(&d, &Budget::unlimited()).unwrap();
            let opt = tour_cost(&d, &tour) as i64;
            // Held-Karp: the bound, rounded up, never exceeds the optimum.
            let bound = (result.best_w + 99).div_euclid(100);
            assert!(bound <= opt, "n={n}: bound {bound} > optimum {opt}");
            assert!(result.best_w >= tree0.w);
            assert!(tree0.w <= 100 * opt);
            // BestW never decreases and ends at the reported value.
            assert!(result.history.windows(2).all(|w| w[0] <= w[1]));
            assert_eq!(
                result.history.last().copied().unwrap_or(tree0.w),
                result.best_w
            );
            // The 1-tree at the best penalties has the best bound.
            let mut graph_tree = OneTree::empty(n);
            let all: Vec<u32> = (0..n)
                .flat_map(|i| (0..n).filter(move |&j| j != i).map(|j| j as u32))
                .collect();
            let graph = Graph::new(&d, &all, n - 1, &tree0);
            SparsePrim::new(n).one_tree(&graph, &d, &result.pi, &mut graph_tree);
            assert_eq!(graph_tree.w, result.best_w, "n={n}");
        }
        // Larger instances: the ascent raises the bound well above pi = 0.
        for (n, seed) in [(200, 1), (500, 2)] {
            let d = euclidean(n, seed);
            let (tree0, result) = run_ascent(&d, &full_schedule(n));
            if result.repairs == 0 {
                assert!(result.history.windows(2).all(|w| w[0] <= w[1]));
            }
            assert!(result.best_w as f64 > 1.02 * tree0.w as f64, "n={n}");
            assert!(result.iterations > 100);
        }
        // A clustered instance, where the sparse graph lacks edges between
        // the clusters: the dense checks repair it, and the bound stays
        // below a tour.
        let d = clustered(600, 3);
        let (_, result) = run_ascent(&d, &full_schedule(600));
        let greedy = super::super::greedy_nearest_neighbor(&d).unwrap();
        assert!(result.best_w <= 100 * tour_cost(&d, &greedy) as i64);
        let dense = one_tree_dense(&d, &result.pi, &Budget::unlimited()).unwrap();
        assert!(dense.w <= result.best_w + 100 * 3000);
    }

    #[test]
    fn ascent_on_a_convex_polygon_finds_the_optimal_tour() {
        for (n, seed) in [(9, 1), (12, 2), (16, 3)] {
            let mut rng = fastrand::Rng::with_seed(seed);
            let mut angles: Vec<f64> = (0..n).map(|_| rng.f64() * std::f64::consts::TAU).collect();
            angles.sort_by(f64::total_cmp);
            // Shuffle the node numbers so the tour is not 0, 1, 2, ...
            let mut ids: Vec<usize> = (0..n).collect();
            rng.shuffle(&mut ids);
            let mut pts = vec![(0.0, 0.0); n];
            for (a, &id) in angles.iter().zip(&ids) {
                pts[id] = (5000.0 + 4000.0 * a.cos(), 5000.0 + 2500.0 * a.sin());
            }
            let rows: Vec<Vec<u16>> = pts
                .iter()
                .map(|a| {
                    pts.iter()
                        .map(|b| (a.0 - b.0).hypot(a.1 - b.1).round() as u16)
                        .collect()
                })
                .collect();
            let d = DistMatrix::from_rows(&rows).unwrap();
            let (_, result) = run_ascent(&d, &full_schedule(n));
            let tour = solve_tsp_exact(&d, &Budget::unlimited()).unwrap();
            let opt = tour_cost(&d, &tour) as i64;
            assert_eq!(result.best_norm, 0, "n={n}");
            assert_eq!(result.best_w, 100 * opt, "n={n}");
        }
    }

    #[test]
    fn polyak_step_also_raises_the_bound() {
        let d = euclidean(300, 8);
        let greedy = super::super::greedy_nearest_neighbor(&d).unwrap();
        let upper = 100 * tour_cost(&d, &greedy) as i64;
        let schedule = Schedule {
            initial_period: 100,
            max_iterations: 200,
            deadline: None,
            adaptive: false,
            step: Step::Polyak { upper },
        };
        let (tree0, result) = run_ascent(&d, &schedule);
        assert!(result.best_w > tree0.w);
        assert!(result.best_w <= upper);
        assert!(result.iterations <= 200);
    }

    #[test]
    fn ascent_stops_at_its_limits() {
        let d = euclidean(400, 3);
        let schedule = Schedule {
            max_iterations: 30,
            ..full_schedule(400)
        };
        let (_, result) = run_ascent(&d, &schedule);
        assert_eq!(result.iterations, 30);
        // A past deadline ends the ascent at once, with pi = 0.
        let schedule = Schedule {
            deadline: Some(Instant::now()),
            ..full_schedule(400)
        };
        let (tree0, result) = run_ascent(&d, &schedule);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.best_w, tree0.w);
        assert!(result.pi.iter().all(|&p| p == 0));
        // An exceeded budget aborts the ascent.
        let near = nearest(&d, 20, &Budget::unlimited(), 1).unwrap();
        let past = Budget::new(Instant::now() - Duration::from_secs(2), 1.0).unwrap();
        assert!(ascent(&d, &near, 20, &tree0, &full_schedule(400), &past).is_none());
        assert!(one_tree_dense(&d, &vec![0; 400], &past).is_none());
        assert!(alpha_lists(&d, &tree0, &vec![0; 400], 5, &past, 1).is_none());
    }

    fn check_lists(d: &DistMatrix, lists: &Lists, base: usize) {
        let n = d.len();
        let near = nearest(d, base, &Budget::unlimited(), 1).unwrap();
        let base = base.min(n - 1);
        assert_eq!(lists.data.len(), n * lists.k);
        assert_eq!(lists.alpha.len(), n * lists.ka);
        for i in 0..n {
            let list = &lists.data[i * lists.k..(i + 1) * lists.k];
            let alpha = &lists.alpha[i * lists.ka..(i + 1) * lists.ka];
            assert!(!list.contains(&(i as u32)));
            let mut uniq = list.to_vec();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(uniq.len(), lists.k);
            assert!(
                list.windows(2)
                    .all(|w| (d.get(i, w[0] as usize), w[0]) < (d.get(i, w[1] as usize), w[1]))
            );
            assert!(
                near[i * base..(i + 1) * base]
                    .iter()
                    .all(|j| list.contains(j))
            );
            assert!(alpha.iter().all(|j| list.contains(j)));
        }
    }

    /// Two far-apart clusters, where alpha lists differ from nearest ones.
    fn clustered(n: usize, seed: u64) -> DistMatrix {
        let mut rng = fastrand::Rng::with_seed(seed);
        let pts: Vec<(f64, f64)> = (0..n)
            .map(|i| {
                let c = if i % 3 == 0 { 0.0 } else { 20000.0 };
                (c + rng.f64() * 3000.0, rng.f64() * 3000.0)
            })
            .collect();
        let rows: Vec<Vec<u16>> = pts
            .iter()
            .map(|a| {
                pts.iter()
                    .map(|b| (a.0 - b.0).hypot(a.1 - b.1).round() as u16)
                    .collect()
            })
            .collect();
        DistMatrix::from_rows(&rows).unwrap()
    }

    #[test]
    fn candidate_lists_contain_nearest_and_alpha() {
        for d in [
            euclidean(21, 3),
            euclidean(23, 4),
            euclidean(300, 5),
            clustered(400, 6),
        ] {
            let n = d.len();
            let lists =
                candidate_lists(&d, &Budget::unlimited(), 1, Candidates::Alpha, 20, 5).unwrap();
            assert!(lists.k >= 20.min(n - 1) && lists.k <= 25.min(n - 1));
            assert_eq!(lists.ka, 5);
            check_lists(&d, &lists, 20);
            // Without an ascent: the nearest lists, and their first 5 as the
            // alpha lists.
            let near =
                candidate_lists(&d, &Budget::unlimited(), 1, Candidates::Nearest, 20, 5).unwrap();
            assert_eq!(near.k, 20.min(n - 1));
            assert_eq!(near.data, nearest(&d, 20, &Budget::unlimited(), 1).unwrap());
            assert_eq!(near.alpha, nearest(&d, 5, &Budget::unlimited(), 1).unwrap());
            check_lists(&d, &near, 20);
        }
        // On the clustered instance the alpha lists bring in edges between
        // the clusters that no nearest-20 list holds.
        let d = clustered(400, 6);
        let lists = candidate_lists(&d, &Budget::unlimited(), 2, Candidates::Alpha, 20, 5).unwrap();
        assert!(lists.k > 20);
        let union_with_padding = union_lists(
            &d,
            &nearest(&d, 25, &Budget::unlimited(), 1).unwrap(),
            25,
            20,
            &lists.alpha,
            5,
        );
        assert_eq!(union_with_padding, (lists.k, lists.data.clone()));
        // Tiny inputs.
        for n in [2, 3, 4] {
            let d = euclidean(n, 1);
            for mode in [Candidates::Alpha, Candidates::Auto, Candidates::Nearest] {
                let lists = candidate_lists(&d, &Budget::unlimited(), 1, mode, 20, 5).unwrap();
                assert_eq!(lists.k, n - 1);
                assert_eq!(lists.ka, n - 1);
                check_lists(&d, &lists, 20);
            }
        }
    }

    #[test]
    fn fill_up_merges_and_sorts() {
        let d = euclidean(30, 8);
        let near = nearest(&d, 10, &Budget::unlimited(), 1).unwrap();
        let tree = one_tree_dense(&d, &[0; 30], &Budget::unlimited()).unwrap();
        let alpha = alpha_lists(&d, &tree, &[0; 30], 4, &Budget::unlimited(), 1).unwrap();
        let mixed = fill_up(&d, &alpha, 4, &near, 10);
        for i in 0..30 {
            let list = &mixed[i * 10..(i + 1) * 10];
            let mut uniq = list.to_vec();
            uniq.sort_unstable();
            uniq.dedup();
            assert_eq!(uniq.len(), 10);
            assert!(alpha[i * 4..(i + 1) * 4].iter().all(|j| list.contains(j)));
            assert!(near[i * 10..i * 10 + 6].iter().all(|j| list.contains(j)));
        }
        assert_eq!(fill_up(&d, &[], 0, &near, 10), near);
    }

    #[test]
    fn exceeded_budget_falls_back_to_nearest_lists() {
        // The callback fires on its second poll, 50 ms after the first,
        // which is long after the nearest lists of 400 nodes are built and
        // in the middle of the (unlimited) ascent.
        let d = euclidean(400, 2);
        let calls = std::cell::Cell::new(0);
        let stop = || {
            calls.set(calls.get() + 1);
            calls.get() > 1
        };
        let budget = Budget::unlimited().with_interrupt(&stop);
        let lists = candidate_lists(&d, &budget, 1, Candidates::Alpha, 20, 5).unwrap();
        assert!(budget.is_interrupted());
        assert_eq!(lists.k, 20);
        assert_eq!(
            lists.data,
            nearest(&d, 20, &Budget::unlimited(), 1).unwrap()
        );
        assert_eq!(
            lists.alpha,
            nearest(&d, 5, &Budget::unlimited(), 1).unwrap()
        );
        // A budget that has run out before the start gives no lists at all.
        let past = Budget::new(Instant::now() - Duration::from_secs(2), 1.0).unwrap();
        for mode in [Candidates::Alpha, Candidates::Auto, Candidates::Nearest] {
            assert!(candidate_lists(&d, &past, 1, mode, 20, 5).is_none());
        }
    }

    #[test]
    fn ascent_plan_follows_the_budget() {
        let ms = Duration::from_millis;
        let secs = Duration::from_secs_f64;
        // `nearest` never runs the ascent, `alpha` always does.
        for left in [None, Some(ms(1)), Some(secs(100.0))] {
            assert_eq!(
                ascent_plan(Candidates::Nearest, 1000, 1, ms(1), left),
                Plan::Nearest
            );
            assert!(matches!(
                ascent_plan(Candidates::Alpha, 1000, 1, ms(1), left),
                Plan::Ascent {
                    adaptive: false,
                    initial_period: 500,
                    ..
                }
            ));
        }
        assert_eq!(
            ascent_plan(Candidates::Alpha, 5000, 4, ms(9), Some(secs(2.0))),
            Plan::Ascent {
                initial_period: 1000,
                time: Some(secs(1.0)),
                adaptive: false
            }
        );
        // Without a deadline, `auto` is `alpha`.
        assert_eq!(
            ascent_plan(Candidates::Auto, 4461, 1, ms(8), None),
            ascent_plan(Candidates::Alpha, 4461, 1, ms(8), None)
        );
        // Nearest lists as measured on one thread: about 8 ms at n = 4461
        // and 50 ms at n = 13509. On 4 threads the wall time is a quarter.
        let plan = |n: usize, threads: usize, near_ms: f64, left: f64| {
            let wall = secs(near_ms / 1e3 / threads_for(n, threads, PARALLEL_MIN_ENTRIES) as f64);
            ascent_plan(Candidates::Auto, n, threads, wall, Some(secs(left)))
        };
        // Short budgets on large inputs keep the nearest lists.
        assert_eq!(plan(4461, 1, 8.0, 0.19), Plan::Nearest);
        assert_eq!(plan(4461, 4, 8.0, 0.19), Plan::Nearest);
        assert_eq!(plan(13509, 1, 50.0, 1.9), Plan::Nearest);
        // Otherwise the first tree is built, with the share as allowance.
        for (n, near_ms, left) in [(4461, 8.0, 1.9), (13509, 50.0, 9.9), (1000, 0.5, 0.19)] {
            for threads in [1, 4] {
                assert_eq!(
                    plan(n, threads, near_ms, left),
                    Plan::Ascent {
                        initial_period: (n / 2).max(100),
                        time: Some(secs(ASCENT_MAX_SHARE * left)),
                        adaptive: true
                    }
                );
            }
        }
        // No time left: nearest.
        assert_eq!(plan(1000, 1, 0.5, 0.0), Plan::Nearest);
    }

    #[test]
    fn ascent_budget_fits_the_schedule_to_the_time_left() {
        let ms = |v: f64| Duration::from_secs_f64(v / 1e3);
        // fnl4461: a dense tree takes about 9 ms, an iteration 0.6 ms.
        let tree = ms(9.0);
        let fixed_1 = 9.0 * (1.0 + ALPHA_PASS_PER_TREE);
        let per_iteration = 9.0 * ITERATION_NODES_PER_TREE / 4461.0;
        // Not even the minimum fits.
        let too_short = fixed_1 + (ASCENT_MIN_ITERATIONS as f64 - 1.0) * per_iteration;
        assert_eq!(ascent_budget(4461, 1, tree, ms(too_short)), None);
        // Just the minimum.
        let (period, time) =
            ascent_budget(4461, 1, tree, ms(fixed_1 + 31.0 * per_iteration)).unwrap();
        assert_eq!(period, MIN_INITIAL_PERIOD);
        assert!((time.as_secs_f64() * 1e3 - 31.0 * per_iteration).abs() < 1e-6);
        // More threads make the alpha pass cheaper and leave more room.
        let one = ascent_budget(4461, 1, tree, ms(190.0)).unwrap();
        let four = ascent_budget(4461, 4, tree, ms(190.0)).unwrap();
        assert!(one.0 < four.0 && one.1 < four.1);
        assert_eq!(one.0, ((190.0 - fixed_1) / per_iteration) as usize / 2);
        // Long budgets get LKH's whole first period, and never more.
        assert_eq!(ascent_budget(4461, 1, tree, ms(10_000.0)).unwrap().0, 2230);
        assert_eq!(
            ascent_budget(150, 1, ms(0.01), ms(10_000.0)).unwrap().0,
            100
        );
    }

    /// A benchmark case file (`<opt> <naive>`, `<n>`, `<x> <y>` lines) as
    /// the u16 matrix the bench gives the solver (scaled down if needed),
    /// and its optimum.
    fn load_case(path: &str) -> (DistMatrix, i64) {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut tokens = text.split_whitespace();
        let opt: i64 = tokens.next().unwrap().parse().unwrap();
        tokens.next();
        let n: usize = tokens.next().unwrap().parse().unwrap();
        let pts: Vec<(f64, f64)> = (0..n)
            .map(|_| {
                let x: f64 = tokens.next().unwrap().parse().unwrap();
                let y: f64 = tokens.next().unwrap().parse().unwrap();
                (x, y)
            })
            .collect();
        let dist = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).hypot(a.1 - b.1);
        let max = pts
            .iter()
            .flat_map(|&a| pts.iter().map(move |&b| dist(a, b).round()))
            .fold(0.0, f64::max);
        let factor = if max > 65535.0 { 65535.0 / max } else { 1.0 };
        let mut data = Vec::with_capacity(n * n);
        for &a in &pts {
            data.extend(pts.iter().map(|&b| (dist(a, b) * factor).round() as u16));
        }
        (DistMatrix::from_flat(n, data).unwrap(), opt)
    }

    /// A TSPLIB tour file (1-based ids after TOUR_SECTION, ending in -1).
    fn load_tour(path: &str) -> Vec<usize> {
        let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("{path}: {e}"));
        text.lines()
            .skip_while(|l| l.trim() != "TOUR_SECTION")
            .skip(1)
            .map(str::trim)
            .take_while(|l| *l != "-1" && *l != "EOF")
            .map(|l| l.parse::<usize>().unwrap() - 1)
            .collect()
    }

    fn dev_cases() -> Vec<String> {
        std::env::var("CASES")
            .unwrap_or_else(|_| {
                "fl1577,fl3795,fl1400,d1291,d1655,d2103,pcb1173,pcb3038,pr2392,u2152,rl1889,\
                 fnl4461,vm1748,rat783,pr1002,u1060"
                    .to_string()
            })
            .split(',')
            .map(str::to_string)
            .collect()
    }

    /// Coverage of a candidate ordering: for every node i and both of its
    /// neighbours j on the reference tour, the rank of j in i's ordering.
    struct Coverage {
        ranks: Vec<u32>,
    }

    impl Coverage {
        fn within(&self, k: u32) -> f64 {
            100.0 * self.ranks.iter().filter(|&&r| r <= k).count() as f64 / self.ranks.len() as f64
        }

        fn mean(&self) -> f64 {
            self.ranks.iter().map(|&r| f64::from(r)).sum::<f64>() / self.ranks.len() as f64
        }
    }

    fn tour_neighbours(tour: &[usize]) -> Vec<[usize; 2]> {
        let n = tour.len();
        let mut nb = vec![[0; 2]; n];
        for (p, &v) in tour.iter().enumerate() {
            nb[v] = [tour[(p + n - 1) % n], tour[(p + 1) % n]];
        }
        nb
    }

    fn distance_ranks(d: &DistMatrix, nb: &[[usize; 2]]) -> Coverage {
        let n = d.len();
        let mut ranks = Vec::with_capacity(2 * n);
        for (i, pair) in nb.iter().enumerate() {
            for &j in pair {
                let key = (d.get(i, j), j);
                let r = (0..n).filter(|&x| x != i && (d.get(i, x), x) < key).count();
                ranks.push(r as u32 + 1);
            }
        }
        Coverage { ranks }
    }

    fn alpha_ranks(d: &DistMatrix, tree: &OneTree, pi: &[i32], nb: &[[usize; 2]]) -> Coverage {
        let n = d.len();
        let index = TreeIndex::new(tree, pi);
        let mut s = BetaScratch::new(n);
        let mut by_node = vec![(0i32, 0i64); n];
        let mut ranks = Vec::with_capacity(2 * n);
        for (i, pair) in nb.iter().enumerate() {
            index.alphas(d, pi, i, &mut s);
            for (p, &a) in s.alphas.iter().enumerate() {
                let j = index.order[p] as usize;
                by_node[j] = (a, d.get(i, j));
            }
            for &j in pair {
                let key = (by_node[j], j);
                let r = (0..n).filter(|&x| x != i && (by_node[x], x) < key).count();
                ranks.push(r as u32 + 1);
            }
        }
        Coverage { ranks }
    }

    /// Ignored harness: how many edges of a near-optimal reference tour lie
    /// in each candidate list, for the dev cases. Run with
    /// `CASES_DIR=cases REF_DIR=<dir with <case>.tour> cargo test --release
    /// --lib coverage -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn coverage() {
        let (Ok(cases_dir), Ok(ref_dir)) = (std::env::var("CASES_DIR"), std::env::var("REF_DIR"))
        else {
            eprintln!("set CASES_DIR and REF_DIR");
            return;
        };
        let iterations: Vec<usize> = std::env::var("ITERS")
            .unwrap_or_else(|_| "30,100,300".into())
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect();
        let unl = Budget::unlimited();
        println!(
            "case,n,config,iters,ms,bound_gap_pct,top5,top8,top10,top20,mean_rank,union_n20_a5,either_end_top5"
        );
        for name in dev_cases() {
            let (d, opt) = load_case(&format!("{cases_dir}/{name}.txt"));
            let n = d.len();
            let tour = load_tour(&format!("{ref_dir}/{name}.tour"));
            assert_eq!(tour.len(), n, "{name}");
            let nb = tour_neighbours(&tour);
            let near_cov = distance_ranks(&d, &nb);
            let report = |config: &str, iters: usize, ms: f64, w: i64, cov: &Coverage| {
                let union = cov
                    .ranks
                    .iter()
                    .zip(&near_cov.ranks)
                    .filter(|&(&a, &r)| a <= 5 || r <= 20)
                    .count() as f64
                    * 100.0
                    / cov.ranks.len() as f64;
                // An edge counts if it is in the top 5 of either end.
                let mut either = 0;
                for (i, pair) in nb.iter().enumerate() {
                    for (side, &j) in pair.iter().enumerate() {
                        let back = usize::from(nb[j][0] != i);
                        if cov.ranks[2 * i + side] <= 5 || cov.ranks[2 * j + back] <= 5 {
                            either += 1;
                        }
                    }
                }
                println!(
                    "{name},{n},{config},{iters},{ms:.1},{:.3},{:.2},{:.2},{:.2},{:.2},{:.2},{union:.2},{:.2}",
                    100.0 * (100.0 * opt as f64 - w as f64) / (100.0 * opt as f64),
                    cov.within(5),
                    cov.within(8),
                    cov.within(10),
                    cov.within(20),
                    cov.mean(),
                    100.0 * f64::from(either) / cov.ranks.len() as f64,
                );
            };
            report("nearest", 0, 0.0, 0, &near_cov);
            let started = Instant::now();
            let tree0 = one_tree_dense(&d, &vec![0; n], &unl).unwrap();
            let ms0 = started.elapsed().as_secs_f64() * 1e3;
            report(
                "alpha_pi0",
                0,
                ms0,
                tree0.w,
                &alpha_ranks(&d, &tree0, &vec![0; n], &nb),
            );
            let near = nearest(&d, 20, &unl, 1).unwrap();
            let lkh_period = (n / 2).max(100);
            let greedy = super::super::greedy_nearest_neighbor(&d).unwrap();
            let upper = 100 * tour_cost(&d, &greedy) as i64;
            let mut configs: Vec<(String, Schedule)> = Vec::new();
            for &it in &iterations {
                configs.push((
                    "lkh".into(),
                    Schedule {
                        initial_period: (it / 2).clamp(MIN_INITIAL_PERIOD, lkh_period),
                        max_iterations: it,
                        deadline: None,
                        adaptive: false,
                        step: Step::Lkh,
                    },
                ));
                configs.push((
                    "polyak".into(),
                    Schedule {
                        initial_period: it,
                        max_iterations: it,
                        deadline: None,
                        adaptive: false,
                        step: Step::Polyak { upper },
                    },
                ));
            }
            configs.push((
                "lkh_alpha_mode".into(),
                Schedule {
                    initial_period: lkh_period.min(ALPHA_MAX_INITIAL_PERIOD),
                    ..full_schedule(n)
                },
            ));
            if std::env::var("FULL").is_ok() {
                configs.push(("lkh_full".into(), full_schedule(n)));
            }
            for (config, schedule) in configs {
                let started = Instant::now();
                let result = ascent(&d, &near, 20, &tree0, &schedule, &unl).unwrap();
                let tree = one_tree_dense(&d, &result.pi, &unl).unwrap();
                let ms = started.elapsed().as_secs_f64() * 1e3;
                let cov = alpha_ranks(&d, &tree, &result.pi, &nb);
                report(
                    &format!("{config}/r{}", result.repairs),
                    result.iterations,
                    ms,
                    result.best_w,
                    &cov,
                );
            }
        }
    }

    /// Ignored harness: the cost of each preprocessing step, for the gate's
    /// cost model. Run through the benchmark lock with
    /// `CASES_DIR=cases cargo test --release --lib build_times -- --ignored --nocapture`.
    #[test]
    #[ignore]
    fn build_times() {
        let Ok(cases_dir) = std::env::var("CASES_DIR") else {
            eprintln!("set CASES_DIR");
            return;
        };
        let unl = Budget::unlimited();
        let best_of = |f: &mut dyn FnMut()| -> f64 {
            (0..3)
                .map(|_| {
                    let t = Instant::now();
                    f();
                    t.elapsed().as_secs_f64() * 1e3
                })
                .fold(f64::MAX, f64::min)
        };
        println!(
            "case,n,near20_ms,near25_ms,near25_t4_ms,dense_ms,graph_ms,entries_per_node,iter_ms,alpha_t1_ms,alpha_t4_ms,dense/u,alpha/u,iter*n2/(E*u)"
        );
        for name in dev_cases() {
            let (d, _) = load_case(&format!("{cases_dir}/{name}.txt"));
            let n = d.len();
            let near20 = best_of(&mut || {
                nearest(&d, 20, &unl, 1).unwrap();
            });
            let near25 = best_of(&mut || {
                nearest(&d, 25, &unl, 1).unwrap();
            });
            let near25_t4 = best_of(&mut || {
                nearest(&d, 25, &unl, 4).unwrap();
            });
            let zero = vec![0; n];
            let dense = best_of(&mut || {
                one_tree_dense(&d, &zero, &unl).unwrap();
            });
            let tree0 = one_tree_dense(&d, &zero, &unl).unwrap();
            let near = nearest(&d, 20, &unl, 1).unwrap();
            let graph_ms = best_of(&mut || {
                Graph::new(&d, &near, 20, &tree0);
            });
            let entries = Graph::new(&d, &near, 20, &tree0).entries();
            let iters = 100;
            let schedule = Schedule {
                max_iterations: iters,
                ..full_schedule(n)
            };
            let mut result = None;
            let ascent_ms = best_of(&mut || {
                result = ascent(&d, &near, 20, &tree0, &schedule, &unl);
            });
            let iter_ms = (ascent_ms - graph_ms) / iters as f64;
            let pi = result.unwrap().pi;
            let tree = one_tree_dense(&d, &pi, &unl).unwrap();
            let alpha1 = best_of(&mut || {
                alpha_lists(&d, &tree, &pi, 5, &unl, 1).unwrap();
            });
            let alpha4 = best_of(&mut || {
                alpha_lists(&d, &tree, &pi, 5, &unl, 4).unwrap();
            });
            let u = near25;
            println!(
                "{name},{n},{near20:.2},{near25:.2},{near25_t4:.2},{dense:.2},{graph_ms:.2},{:.1},{iter_ms:.3},{alpha1:.2},{alpha4:.2},{:.2},{:.2},{:.1}",
                entries as f64 / n as f64,
                dense / u,
                alpha1 / u,
                iter_ms * (n * n) as f64 / (entries as f64 * u),
            );
        }
    }
}
