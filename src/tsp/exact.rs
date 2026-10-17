use super::{Budget, DistMatrix, Result, Tour, TspError, invalid};

/// Largest problem the exact solver accepts. The DP table takes
/// `2^(n-1) * (n-1) * 4` bytes: about 1.6 GB and tens of seconds at n = 25.
/// Each extra node roughly doubles both, so larger inputs are refused
/// instead of risking the OS killing the process when memory runs out.
pub const MAX_EXACT_NODES: usize = 25;

const INF: u32 = u32::MAX / 2;

/// Check for interrupts every this many DP subsets.
const INTERRUPT_CHECK_MASK: usize = (1 << 12) - 1;

/// Bottom-up Held-Karp, O(2^n * n^2) time.
///
/// Node 0 is the fixed start, so subsets only cover nodes `1..n`: node `k`
/// is bit `k - 1`. `dp[S * m + (k - 1)]` is the cheapest path that starts at
/// node 0, visits exactly the nodes in `S` and ends at node `k`.
///
/// Only the interrupt callback in `budget` is honoured, not its deadline:
/// a partial DP gives no tour at all.
pub fn solve_tsp_exact(dists: &DistMatrix, budget: &Budget) -> Result<Tour> {
    let n = dists.len();
    if n == 0 {
        return invalid("Distance matrix must be non-empty");
    }
    if n > MAX_EXACT_NODES {
        return invalid(format!(
            "The exact solver supports at most {MAX_EXACT_NODES} nodes, but got {n}"
        ));
    }
    if n == 1 {
        return Ok(vec![0]);
    }

    let m = n - 1;
    let num_subsets = 1usize << m;
    let full = num_subsets - 1;
    // The path is reconstructed backwards from node 0 (each step picks the
    // previous node). Running the DP on the transposed matrix makes that
    // backwards walk come out as the tour in forward order, which matters
    // for asymmetric matrices. Max tour cost is 65535 * 25, well within u32.
    let d = |i: usize, j: usize| dists.get(j, i) as u32;

    let out_of_memory = || {
        TspError::OutOfMemory(format!(
            "Not enough memory to solve a problem with {n} nodes exactly"
        ))
    };
    let size = num_subsets.checked_mul(m).ok_or_else(out_of_memory)?;
    let mut dp: Vec<u32> = Vec::new();
    dp.try_reserve_exact(size).map_err(|_| out_of_memory())?;
    dp.resize(size, INF);

    for k in 1..n {
        dp[(1 << (k - 1)) * m + (k - 1)] = d(0, k);
    }

    for s in 1..num_subsets {
        if s & INTERRUPT_CHECK_MASK == 0 && budget.is_interrupted() {
            return Err(TspError::Interrupted);
        }
        let mut ks = s;
        while ks != 0 {
            let kb = ks.trailing_zeros() as usize;
            ks &= ks - 1;
            let prev = s & !(1 << kb);
            if prev == 0 {
                continue;
            }
            let mut best = dp[s * m + kb];
            let mut js = prev;
            while js != 0 {
                let jb = js.trailing_zeros() as usize;
                js &= js - 1;
                best = best.min(dp[prev * m + jb] + d(jb + 1, kb + 1));
            }
            dp[s * m + kb] = best;
        }
    }

    // Walk backwards from node 0: each step picks the node that should come
    // just before the ones already placed. Ties go to the lowest index.
    let mut tour = Vec::with_capacity(n);
    tour.push(0);
    let mut remaining = full;
    while remaining != 0 {
        let last = *tour.last().unwrap();
        let mut best = (u32::MAX, 0);
        let mut ks = remaining;
        while ks != 0 {
            let kb = ks.trailing_zeros() as usize;
            ks &= ks - 1;
            let cost = dp[remaining * m + kb] + d(kb + 1, last);
            if cost < best.0 {
                best = (cost, kb);
            }
        }
        tour.push(best.1 + 1);
        remaining &= !(1 << best.1);
    }
    Ok(tour)
}

#[cfg(test)]
mod tests {
    use super::super::test_utils::{euclidean, matrix};
    use super::super::{is_valid_tour, tour_cost};
    use super::*;

    fn exact(d: &DistMatrix) -> Result<Tour> {
        solve_tsp_exact(d, &Budget::unlimited())
    }

    fn brute_force(dists: &DistMatrix) -> u64 {
        fn permute(rest: &mut Vec<usize>, k: usize, d: &DistMatrix, best: &mut u64) {
            if k == rest.len() {
                let mut tour = vec![0];
                tour.extend_from_slice(rest);
                *best = (*best).min(tour_cost(d, &tour));
                return;
            }
            for i in k..rest.len() {
                rest.swap(k, i);
                permute(rest, k + 1, d, best);
                rest.swap(k, i);
            }
        }
        let mut rest: Vec<usize> = (1..dists.len()).collect();
        let mut best = u64::MAX;
        permute(&mut rest, 0, dists, &mut best);
        best
    }

    fn random_asymmetric(n: usize, seed: u64) -> DistMatrix {
        let mut rng = fastrand::Rng::with_seed(seed);
        let rows: Vec<Vec<u16>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| if i == j { 0 } else { rng.u16(1..1000) })
                    .collect()
            })
            .collect();
        matrix(&rows)
    }

    #[test]
    fn matches_brute_force() {
        for n in 2..=8 {
            for seed in 0..5 {
                for d in [
                    euclidean(n, seed * 100 + n as u64),
                    random_asymmetric(n, seed * 100 + n as u64),
                ] {
                    let tour = exact(&d).unwrap();
                    assert!(is_valid_tour(n, &tour));
                    assert_eq!(tour_cost(&d, &tour), brute_force(&d), "n={n} seed={seed}");
                }
            }
        }
    }

    #[test]
    fn asymmetric_tour_is_in_forward_order() {
        // Going 0 -> 1 -> 2 -> 3 -> 0 costs 4. The reverse direction costs 400.
        let d = matrix(&[
            vec![0, 1, 100, 100],
            vec![100, 0, 1, 100],
            vec![100, 100, 0, 1],
            vec![1, 100, 100, 0],
        ]);
        assert_eq!(exact(&d).unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn small_and_large_inputs() {
        assert_eq!(exact(&matrix(&[vec![0]])).unwrap(), vec![0]);
        assert!(exact(&matrix(&[])).is_err());
        let too_big = MAX_EXACT_NODES + 1;
        assert!(exact(&matrix(&vec![vec![0; too_big]; too_big])).is_err());
    }
}
