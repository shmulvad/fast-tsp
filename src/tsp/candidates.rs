//! Candidate neighbour lists for 2-opt and 3-opt.

use super::{Budget, DistMatrix};

/// A row of the distance matrix.
#[inline(always)]
fn row(dists: &DistMatrix, i: usize) -> &[u16] {
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

/// For each node, its `k` closest other nodes, closest first (ties to the
/// lowest index). Each row is selected with `block_minima` and
/// `kth_smallest`, so almost all the work is a vectorised scan.
pub(crate) fn nearest(dists: &DistMatrix, k: usize, budget: &Budget<'_>) -> Option<Vec<u32>> {
    let n = dists.len();
    let k = k.min(n.saturating_sub(1));
    if k == 0 {
        return Some(Vec::new());
    }
    let mut mins: Vec<u16> = Vec::with_capacity(n.div_ceil(BLOCK));
    let mut scratch: Vec<u16> = Vec::with_capacity(n.div_ceil(BLOCK));
    let mut best: Vec<u64> = Vec::with_capacity(k + 1);
    let mut data = Vec::with_capacity(n * k);
    for i in 0..n {
        if i % 64 == 0 && budget.is_interrupted() {
            return None;
        }
        let r = row(dists, i);
        block_minima(r, &mut mins);
        // The block holding i itself, without the diagonal entry.
        let own = i / BLOCK;
        mins[own] = r[own * BLOCK..n.min(own * BLOCK + BLOCK)]
            .iter()
            .enumerate()
            .filter(|&(o, _)| own * BLOCK + o != i)
            .fold(u16::MAX, |m, (_, &d)| m.min(d));
        let bound = kth_smallest(&mins, k, &mut scratch);
        best.clear();
        for (b, &m) in mins.iter().enumerate() {
            if m > bound {
                continue;
            }
            let base = b * BLOCK;
            for (o, &d) in r[base..n.min(base + BLOCK)].iter().enumerate() {
                let j = base + o;
                if d <= bound && j != i {
                    push_best(&mut best, k, (u64::from(d) << 32) | j as u64);
                }
            }
        }
        data.extend(best.iter().map(|&key| key as u32));
    }
    Some(data)
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
            let near = nearest(&d, k, &Budget::unlimited()).unwrap();
            assert_eq!(near.len(), n * k);
            for i in 0..n {
                let mut all: Vec<(i64, u32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (d.get(i, j), j as u32))
                    .collect();
                all.sort_unstable();
                let expected: Vec<u32> = all[..k].iter().map(|t| t.1).collect();
                assert_eq!(&near[i * k..(i + 1) * k], &expected[..], "n={n} i={i}");
            }
        }
    }
}
