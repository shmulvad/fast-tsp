/*
 * Acknowledgement:
 * In our 48/50 submission for tsp, we adapted code/ideas from various online GitHub
 * repositories that contain solutions kattis tsp too.
 *
 * 1. Adapated code from this repository in the assignment PDF:
 *   https://github.com/estan/tsp
 *
 * 2. Adapated code from one of the previous top CS4234 teams on TSP:
 *   https://github.com/Lookuz/CS4234-Stochastic-Local-Search-Methods/tree/master/TSP
 */
#include <algorithm>
#include <iostream>
#include <limits>
#include <random>
#include <queue>
#include <cassert>
#include <tuple>
#include <stdexcept>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#define STRINGIFY(x) #x
#define MACRO_STRINGIFY(x) STRINGIFY(x)

namespace py = pybind11;

using namespace std;

typedef long long ll;
typedef pair<double, double> dd;
typedef vector<dd> vdd;
typedef vector<uint_fast16_t> vi;
typedef vector<ll> vll;
typedef pair<uint_fast32_t, uint_fast16_t> ii;
typedef vector<ii> vii;

#define EMPTY_SET 0
#define MAX_N 1350
#define DEFAULT_TIME_LIMIT 2.0

const ll kInf = 1e17;
const uint_fast16_t NEIGH_LIMIT = 20; // 24 is good for Kattis!
const static int GEO_SHUFFLE_WIDTH = 30;
const double CLOCKS_PER_SEC_INV = 1.0 / CLOCKS_PER_SEC;

int dist[MAX_N][MAX_N];
vector<vi> nearestNeighbour(MAX_N);

// Set-up RNG
random_device rd;
mt19937 gen(rd());


void fill_up_dist_matrix(uint_fast16_t n, const vector<vi> dists);
vi find_tour(const vector<vi> dists, const float duration_seconds);
void fill_up_nearest_neighbour_matrix(uint_fast16_t n);
vi local_search(uint_fast16_t n, const clock_t& clock_begin, float duration_seconds);
void naive_algo(uint_fast16_t n, vi& tour);
void print_tour(const vi& tour);
uint_fast32_t compute_cost(uint_fast16_t n, const vi& tour);
void two_opt(uint_fast16_t n, vi& tour, uint_fast16_t* position, const clock_t& clock_begin, float duration_seconds);
vi double_bridge_move(uint_fast16_t n, vi& tour);
void reverse_tour(uint_fast16_t n, vi& tour, uint_fast16_t start, uint_fast16_t end, uint_fast16_t* position);
void shuffle_tour(int n, vi& tour);
vi solve_dp_bitmask(uint_fast16_t n);
void three_opt(int n, vi& tour, uint_fast16_t* position, const clock_t& clock_begin, float duration_seconds);
void swap_adj_seg(uint_fast16_t n, vi& tour, uint_fast16_t* position, uint_fast16_t A, uint_fast16_t B, uint_fast16_t C, uint_fast16_t D);


vi find_tour(const vector<vi> dists, const float duration_seconds = DEFAULT_TIME_LIMIT) {
    clock_t clock_begin = clock();
    if (duration_seconds <= 0.0) throw invalid_argument("Duration must be strictly positive");

    uint_fast16_t n = dists.size();
    if (n < 2) throw invalid_argument("Need at least 2 nodes");
    uint_fast16_t m = dists[0].size();
    if (m != n) throw invalid_argument("Distance matrix must be square but got (" + to_string(n) + ", " + to_string(m) + ")");

    for (uint_fast16_t i = 0; i < n; ++i)
        dist[i][i] = 0;

    for (uint_fast16_t i = 0; i < n - 1; ++i) {
        for (uint_fast16_t j = i + 1; j < n; ++j) {
            dist[i][j] = dist[j][i] = dists[i][j];
        }
    }
    if (n < 20) return solve_dp_bitmask(n);

    // Solve it
    fill_up_nearest_neighbour_matrix(n);
    vi tour = local_search(n, clock_begin, duration_seconds);
    return tour;
}

inline bool is_time_exceeded(clock_t begin, float duration_seconds) {
    double elapsed_time = (double) (clock() - begin) * CLOCKS_PER_SEC_INV;
    return elapsed_time > duration_seconds;
}

// Fills up the distance matrix as per the problem specs as euclidean distance
// rounded to the nearest integer. Takes `n` number of 2D `coords` as input.
void fill_up_dist_matrix(uint_fast16_t n, const vector<vi> dists_inp) {
    for (uint_fast16_t i = 0; i < n; ++i)
        dist[i][i] = 0;

    for (uint_fast16_t i = 0; i < n - 1; ++i) {
        for (uint_fast16_t j = i + 1; j < n; ++j) {
            dist[i][j] = dist[j][i] = dists_inp[i][j];
        }
    }
}

// Fills up nearest neighbour matrix where for each node i, we compute the nearest
// neighbour. This method has to be called after populating the
// dist[MAX_n][MAX_n] array
void fill_up_nearest_neighbour_matrix(uint_fast16_t n) {
    uint_fast16_t size = min(NEIGH_LIMIT, n);
    for (uint_fast16_t i = 0; i < n; ++i) {
        nearestNeighbour[i] = vi(size);
    }

    priority_queue<ii, vii, greater<ii>> pq;

    uint_fast16_t j = 0;
    uint_fast16_t d, v;
    for (uint_fast16_t i = 0; i < n; ++i) {
        for (j = 0; j < n; ++j) {
            // we don't want a node to become it's own nearest neighbour
            if (i == j) continue;
            pq.push({dist[i][j], j});
        }

        j = 0;
        while (!pq.empty()) {
            if (j >= NEIGH_LIMIT) break;

            tie(d, v) = pq.top(); pq.pop();

            nearestNeighbour[i][j] = v;
            j++;
        }

        pq = priority_queue<ii, vii, greater<ii>>();
    }
}

// Implementation of bottom-up Held-Karp's algorithm, solving TSP in
// O(2^n * n^2) time.
vi solve_dp_bitmask(uint_fast16_t n) {
    const int num_combinations = 1 << n;
    const int max_mask = num_combinations - 1;

    // Vector is used instead because of seg fault
    vector<vll> dp(num_combinations);
    for (int i = 0; i < num_combinations; ++i) {
        dp[i] = vll(n);
        for (int j = 0; j < n; ++j) {
            dp[i][j] = kInf;
        }
    }

    // Fix node 0 as source for TSP tour
    for (int k = 1; k < n; ++k) {
        dp[1 << k][k] = dist[0][k];
    }

    // Iterate through all subsets of vertices/nodes
    for (int S = EMPTY_SET; S < num_combinations; ++S) {
        if (S & (1 << 0)) continue; // Don't care about subsets with source node

        // Each k nodes take turns to be dest!
        for (int k = 0; k < n; ++k) {
            if (!(S & (1 << k))) continue;

            for (int m = 0; m < n; ++m) {
                if ((m == k) || (!(S & (1 << m)))) continue;
                dp[S][k] = min(dp[S][k], dp[(S & ~(1 << k))][m] + dist[m][k]);
            }
        }
    }

    // Reconstruct the path from dp array
    ll ans = kInf;
    vi ans_path;
    int chosen_node = 0;

    ans_path.push_back(chosen_node);
    int mask = 1 << chosen_node;

    // Get the remaining n - 1 nodes
    for (int i = 0; i < n - 1; ++i) {
        ans = kInf;
        for (int k = 0; k < n; ++k) {
            // Only go through all potential dest not inside tour
            if (mask & (1 << k)) continue;
            int d = dp[max_mask & ~(mask)][k] + dist[k][ans_path[ans_path.size() - 1]];
            if (d < ans) {
                ans = d;
                chosen_node = k;
            }
        }

        ans_path.push_back(chosen_node);
        mask |= 1 << chosen_node;
    }

    return ans_path;
}

vi double_bridge_move(uint_fast16_t n, vi& tour) {
    vi newTour;
    newTour.reserve(n);

    if (n < 8) {
        newTour = tour;

        shuffle(newTour.begin(), newTour.end(), rd);

        return newTour;
    }

    uniform_int_distribution<uint_fast16_t> randomOffset(1, n / 4);
    uint_fast16_t A = randomOffset(rd);
    uint_fast16_t B = A + randomOffset(rd);
    uint_fast16_t C = B + randomOffset(rd);
    copy(tour.begin(), tour.begin() + A, back_inserter(newTour));
    copy(tour.begin() + C, tour.end(), back_inserter(newTour));
    copy(tour.begin() + B, tour.begin() + C, back_inserter(newTour));
    copy(tour.begin() + A, tour.begin() + B, back_inserter(newTour));
    return newTour;
}

// Copied reverse method from: https://github.com/estan/tsp/blob/master/tsp.cpp
void reverse_tour(uint_fast16_t n, vi& tour, uint_fast16_t start, uint_fast16_t end, uint_fast16_t* position) {
    uint_fast16_t numSwaps = (((start <= end ? end - start : (end + n) - start) + 1) / 2);
    uint_fast16_t i = start;
    uint_fast16_t j = end;
    while (numSwaps--) {
        swap(tour[i], tour[j]);

        position[tour[i]] = i;
        position[tour[j]] = j;

        i = (i + 1) % n;
        j = ((j + n) - 1) % n;
    }
}

// Gives a 2-approximate solution to TSP based on the greedy nearest neighbour
// approach. Used as the starting point for local search.
void naive_algo(uint_fast16_t n, vi& tour) {
    bool used[n];
    for (uint_fast16_t i = 0; i < n; ++i) {
        used[i] = false;
    }

    tour[0] = 0;
    used[0] = true;
    uint_fast16_t j;
    for (uint_fast16_t i = 1; i < n; ++i) {
        int best = -1;
        for (j = 1; j < n; ++j) {
            if (used[j]) continue;
            if (best == -1 || dist[tour[i - 1]][j] < dist[tour[i - 1]][best]) {
                best = j;
            }
        }
        tour[i] = best;
        used[best] = true;
    }
}


// Does local search over the problem until the time limit is reached
vi local_search(uint_fast16_t n, const clock_t& clock_begin, float duration_seconds) {
    /*
     * Start of with the greedy solution (in problem specs), then improve from there via
     * SLS
     */
    vi tour(n);
    naive_algo(n, tour);

    // Compute cost for this tour, takes O(n)
    uint_fast32_t cost = compute_cost(n, tour);

    // Perform 2-opt and 3 -top on tour
    vi bestTour = tour;
    uint_fast32_t bestCost = cost;

    // Find finding our first local maxima, hope we get a pretty good one
    uint_fast16_t* position = new uint_fast16_t[n];

    uint_fast16_t i;

    // Diversification
    while (!is_time_exceeded(clock_begin, duration_seconds)) {
        tour = double_bridge_move(n, tour);
        shuffle_tour(n, tour);

        // Next, perform subsidiary local search
        for (i = 0; i < n; ++i) {
            position[tour[i]] = i;
        }

        // Intensification
        two_opt(n, tour, position, clock_begin, duration_seconds);
        three_opt(n, tour, position, clock_begin, duration_seconds);
        cost = compute_cost(n, tour);

        //cout << "cost = " << cost << "\n";
        if (cost <= bestCost) {
            //cout << "better cost found!" << "\n";
            bestCost = cost;
            bestTour = tour;
        } else {
            tour = bestTour;
        }
    }
    return bestTour;
}

// Prints a given tour (linear fashion/order)
void print_tour(const vi& tour) {
    for (auto v : tour) {
        cout << v << "\n";
    }
}

// Computes the cost of a given tour, takes O(n)
uint_fast32_t compute_cost(uint_fast16_t n, const vi& tour) {
    uint_fast32_t cost = 0;
    uint_fast16_t prev = tour[0];
    uint_fast16_t v;
    for (uint_fast16_t i = 1; i < n; ++i) {
        v = tour[i];
        cost += dist[prev][v];
        prev = v;
    }

    cost += dist[prev][tour[0]]; // Close the loop

    return cost;
}

// Performs a 2-opt move on a given tour, where we select 2 distinct edges based on
// nearest neighbour heuristic (and take the first improving perturbative neighbour)
// Terminates when there are no improving neighbours or when time limit exeeded.
void two_opt(uint_fast16_t n, vi& tour, uint_fast16_t* position, const clock_t& clock_begin, float duration_seconds) {
    uint_fast16_t u1, u2, u3, u4;

    uint_fast16_t idx1 = 0;
    uint_fast16_t tempPos;
    uint_fast16_t tempIdx;
    bool improvingTSP = true; // Exit when no improving neighbours
    uint_fast16_t N;

    int d0, d1;

    /*
     * Exit SLS if no improving TSP (neighbours), i.e, we can't optimise the TSP any
     * further
     */
    while (!is_time_exceeded(clock_begin, duration_seconds) && improvingTSP) {
        improvingTSP = false;

        // Reset variables
        N = n;

        while (N--) {
            u1 = tour[idx1];
            tempIdx = (idx1 + 1) % n;
            u2 = tour[tempIdx];

            for (auto idx2 : nearestNeighbour[u1]) {
                u3 = idx2;

                d0 = dist[u1][u2];
                d1 = dist[u1][u3];

                // All subsequent neighbours of u1 guaranteed to have longer distances
                if (d0 <= d1) break;

                tempPos = position[idx2];
                u4 = tour[(tempPos + 1) % n];

                if (u1 == u4 || u2 == u3) continue;

                // Using technique for fast w(p') computation from lecture slides
                if (d0 + dist[u3][u4] > d1 + dist[u2][u4]) {
                    reverse_tour(n, tour, tempIdx, tempPos, position);
                    improvingTSP = true;

                    break;
                }
            }

            idx1 = tempIdx; // Increment index by 1 (mod n)
        }
    }
}

// This randomly selects a subtour and shuffles it
// modifies tour in place
void shuffle_tour(int n, vi& tour) {
    if (tour.size() <= GEO_SHUFFLE_WIDTH) {
        shuffle(tour.begin(), tour.end(), rd);
        return;
    }
    uniform_int_distribution<size_t> randomOffset(0, tour.size() - GEO_SHUFFLE_WIDTH); // [a, b] Inclusive
    int left = randomOffset(rd);
    shuffle(tour.begin() + left, tour.begin() + left + GEO_SHUFFLE_WIDTH, rd); // [a, b) exclusive
    return;
}

// Performs a 3-opt move on a given tour, where we select 3 distinct edges based on
// nearest neighbour heuristic (and take the first improving perturbative neighbour)
void three_opt(int n, vi& tour, uint_fast16_t* position, const clock_t& clock_begin, float duration_seconds) {
    uint_fast16_t u1, u2, u3, u4, u5, u6;
    uint_fast16_t u1Idx, u2Idx, u3Idx, u4Idx, u5Idx, u6Idx;

    bool improvingTSP = true; // Exit when no improving neighbours
    while (!is_time_exceeded(clock_begin, duration_seconds) && improvingTSP) {
        improvingTSP = false;

        for (uint_fast16_t idx1 = 0; idx1 < n; ++idx1) {

            u1Idx = idx1;
            u2Idx = (idx1 + 1) % n;
            u1 = tour[u1Idx];
            u2 = tour[u2Idx];

            for (auto idx2 : nearestNeighbour[u2]) {
                u3 = idx2;
                u3Idx = position[u3];

                // All subsequent neighbours of guaranteed to have longer distances
                if (dist[u1][u2] <= dist[u2][u3]) break;

                if (u3 == u1) continue;

                u4Idx = (position[u3] + 1) % n;
                u4 = tour[u4Idx];

                if (u4Idx != u1Idx) {
                    for (auto idx3 : nearestNeighbour[u4]) {
                        u5 = idx3;
                        u5Idx = position[u5];

                        if (u5 == u3 || u5 == u1 || u5 == u2) continue;

                        if (!((u5Idx < u3Idx && u3Idx < u2Idx) ||
                            (u3Idx < u2Idx && u2Idx < u5Idx) ||
                            (u2Idx < u5Idx && u5Idx < u3Idx))) continue;

                        // All subsequent neighbours of guaranteed to have longer distances
                        if (dist[u3][u4] <= ((dist[u2][u3] - dist[u1][u2]) + dist[u4][u5])) break;

                        u6Idx = (position[u5] + 1) % n;
                        u6 = tour[u6Idx];

                        if (((((dist[u2][u3] - dist[u1][u2]) + dist[u4][u5]) - dist[u3][u4]) + dist[u1][u6]) < (dist[u5][u6])) {
                            improvingTSP = true;

                            swap_adj_seg(n, tour, position, u4Idx, u1Idx, u2Idx, u5Idx);

                            goto nextU1U2;
                        }
                    }
                }

                u4Idx = ((position[u3] - 1) + n) % n;
                u4 = tour[u4Idx];

                for (auto idx3 : nearestNeighbour[u4]) {
                    u5 = idx3;
                    u5Idx = position[u5];

                    if (u5 == u3 || u5 == u1 || u5 == u2) continue;

                    if (!((u5Idx < u3Idx && u3Idx < u2Idx) ||
                        (u3Idx < u2Idx && u2Idx < u5Idx) ||
                        (u2Idx < u5Idx && u5Idx < u3Idx))) continue;

                    // All subsequent neighbours of guaranteed to have longer distances
                    if (dist[u3][u4] <= ((dist[u2][u3] - dist[u1][u2]) + dist[u4][u5])) break;

                    u6Idx = (position[u5] + 1) % n;
                    u6 = tour[u6Idx];

                    if (((((dist[u2][u3] - dist[u1][u2]) + dist[u4][u5]) - dist[u3][u4]) + dist[u1][u6]) < (dist[u5][u6])) {
                        improvingTSP = true;

                        reverse_tour(n, tour, u6Idx, u4Idx, position);
                        reverse_tour(n, tour, u3Idx, u1Idx, position);

                        goto nextU1U2;
                    }
                }
            }
            nextU1U2: continue;
        }
    }
}

void swap_adj_seg(uint_fast16_t n, vi& tour, uint_fast16_t* position, uint_fast16_t A, uint_fast16_t B, uint_fast16_t C, uint_fast16_t D) {
    vi temp;

    uint_fast16_t cur = C;
    while (cur != D) {
        temp.push_back(tour[cur]);
        cur = (cur + 1) % n;
    }
    temp.push_back(tour[cur]); // temp contains segment [C .. D]

    cur = A;
    while (cur != B) {
        temp.push_back(tour[cur]);
        cur = (cur + 1) % n;
    }
    temp.push_back(tour[cur]); // temp contains segment [C .. D, A .. B]

    for (uint_fast16_t i = 0; i < temp.size(); ++i) { // copy over to tour
        tour[(A + i) % n] = temp[i];
        position[temp[i]] = (A + i) % n;
    }
}


PYBIND11_MODULE(_core, m) {
    m.doc() = R"pbdoc(
        TSP solver using fast heuristic.
        -----------------------

        .. currentmodule:: fast_tsp

        .. autosummary::
           :toctree: _generate

           find_tour
    )pbdoc";

    m.def("find_tour", &find_tour, R"pbdoc(
        Find a good TSP tour

        Run a local solver to find a good TSP tour.
    )pbdoc", py::arg("dists"), py::arg("duration_seconds") = DEFAULT_TIME_LIMIT);

#ifdef VERSION_INFO
    m.attr("__version__") = MACRO_STRINGIFY(VERSION_INFO);
#else
    m.attr("__version__") = "dev";
#endif
}
