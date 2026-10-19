use super::candidates::row;
use super::{DistMatrix, Result, Tour, invalid};

/// Starts at node 0 and repeatedly moves to the closest unvisited node.
pub fn greedy_nearest_neighbor(dists: &DistMatrix) -> Result<Tour> {
    let n = dists.len();
    if n == 0 {
        return invalid("Distance matrix must be non-empty");
    }
    let mut used = vec![false; n];
    let mut tour = Vec::with_capacity(n);
    tour.push(0);
    used[0] = true;
    for _ in 1..n {
        let prev = *tour.last().unwrap();
        // Ties go to the lowest index.
        let best = (1..n)
            .filter(|&j| !used[j])
            .min_by_key(|&j| dists.get(prev, j))
            .unwrap();
        tour.push(best);
        used[best] = true;
    }
    Ok(tour)
}

/// The same tour as [`greedy_nearest_neighbor`], about five times faster:
/// visited nodes are masked to `u16::MAX`, so each step is one vectorised
/// pass for the minimum of each block plus a scan of the block with the
/// first overall minimum.
pub(crate) fn nearest_neighbor_tour(dists: &DistMatrix) -> Vec<u32> {
    const BLOCK: usize = 64;
    let n = dists.len();
    let mut mask = vec![0u16; n];
    let mut tour = Vec::with_capacity(n);
    let mut cur = 0;
    for _ in 1..n {
        tour.push(cur as u32);
        mask[cur] = u16::MAX;
        let row = row(dists, cur);
        let (mut best, mut best_block) = (u16::MAX, usize::MAX);
        for (b, (r, m)) in row.chunks(BLOCK).zip(mask.chunks(BLOCK)).enumerate() {
            let min = r
                .iter()
                .zip(m)
                .fold(u16::MAX, |acc, (&d, &m)| acc.min(d | m));
            if min < best {
                (best, best_block) = (min, b);
            }
        }
        cur = if best_block == usize::MAX {
            // Every unvisited node is at distance u16::MAX: take the first.
            mask.iter().position(|&m| m == 0)
        } else {
            let start = best_block * BLOCK;
            let end = n.min(start + BLOCK);
            (start..end).find(|&j| row[j] | mask[j] == best)
        }
        .expect("an unvisited node is left");
    }
    tour.push(cur as u32);
    tour
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn follows_nearest_neighbour() {
        let d = DistMatrix::from_rows(&[
            vec![0, 5, 1, 9],
            vec![5, 0, 2, 1],
            vec![1, 2, 0, 7],
            vec![9, 1, 7, 0],
        ])
        .unwrap();
        assert_eq!(greedy_nearest_neighbor(&d).unwrap(), vec![0, 2, 1, 3]);
    }

    #[test]
    fn fast_nearest_neighbour_matches_public_one() {
        let ties: Vec<Vec<u16>> = (0..300)
            .map(|i: usize| {
                (0..300)
                    .map(|j: usize| u16::from(i != j) * ((i + j) % 4) as u16)
                    .collect()
            })
            .collect();
        let far: Vec<Vec<u16>> = (0..70)
            .map(|i| (0..70).map(|j| if i == j { 0 } else { u16::MAX }).collect())
            .collect();
        for d in [
            crate::tsp::test_utils::euclidean(2, 1),
            crate::tsp::test_utils::euclidean(500, 9),
            DistMatrix::from_rows(&ties).unwrap(),
            DistMatrix::from_rows(&far).unwrap(),
        ] {
            let fast: Vec<usize> = nearest_neighbor_tour(&d)
                .iter()
                .map(|&v| v as usize)
                .collect();
            assert_eq!(fast, greedy_nearest_neighbor(&d).unwrap());
        }
    }

    #[test]
    fn edge_cases() {
        let one = DistMatrix::from_rows(&[vec![0]]).unwrap();
        assert_eq!(greedy_nearest_neighbor(&one).unwrap(), vec![0]);
        let empty = DistMatrix::from_rows(&[]).unwrap();
        assert!(greedy_nearest_neighbor(&empty).is_err());
    }
}
