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
    fn edge_cases() {
        let one = DistMatrix::from_rows(&[vec![0]]).unwrap();
        assert_eq!(greedy_nearest_neighbor(&one).unwrap(), vec![0]);
        let empty = DistMatrix::from_rows(&[]).unwrap();
        assert!(greedy_nearest_neighbor(&empty).is_err());
    }
}
