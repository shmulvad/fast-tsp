//! Kattis-style benchmark for the solver.
//!
//! Runs `find_tour` on every case in a folder and scores each tour with the
//! Kattis formula `0.02^((cost - opt) / (naive - opt))`, where `opt` and
//! `naive` (the greedy nearest-neighbour cost) come from the case file.
//!
//! ```text
//! cargo run --release --example bench -- --time 0.1,2,10 --threads 1,4 --repeats 3
//! ```
//!
//! Case files look like:
//!
//! ```text
//! <opt> <naive>
//! <n>
//! <x> <y>   (n lines)
//! ```
//!
//! The clock starts before the distance matrix is built, as on Kattis where
//! reading the input counts toward the time limit. Tours are scored on the
//! exact Kattis distances (Euclidean, rounded to the nearest integer). Cases
//! whose distances do not fit in `u16` are scaled down before solving, as a
//! library user would have to do; any precision lost shows up in the score.

use std::path::{Path, PathBuf};
use std::time::Instant;
use std::{env, fs, process};

use _core::tsp::{self, Budget, DistMatrix, Tour};

const USAGE: &str = "\
Usage: cargo run --release --example bench -- [options]

Options:
  --cases <dir>          Folder with *.txt cases [default: cases]
  --time <s[,s...]>      Time limit per case in seconds; a list sweeps [default: 2]
  --threads <n[,n...]>   Search threads; a list sweeps [default: 1]
  --repeats <k>          Runs per case per configuration [default: 1]
  --filter <text>        Only run cases whose file name contains <text>
  --scale-to <N>         Also report the total scaled to N cases [default: 50]
  --quiet                Only print the summaries
  -h, --help             Show this help";

/// A run counts as late when it overshoots the limit by more than this
/// fraction of the limit. The solver checks the deadline between moves, so
/// tiny overshoots are expected.
const LATE_TOLERANCE: f64 = 0.01;

type Point = (f64, f64);

#[derive(Debug)]
struct Case {
    name: String,
    opt: i64,
    naive: i64,
    points: Vec<Point>,
    /// Factor applied to distances before solving, when the largest one
    /// does not fit in `u16`.
    scale: Option<f64>,
}

#[derive(Debug)]
struct RunResult {
    score: f64,
    cost: Option<i64>,
    elapsed: f64,
    error: Option<String>,
}

#[derive(Debug)]
struct Args {
    cases: PathBuf,
    times: Vec<f64>,
    threads: Vec<usize>,
    repeats: usize,
    filter: Option<String>,
    scale_to: f64,
    quiet: bool,
}

#[derive(Debug)]
struct Summary {
    secs: f64,
    threads: usize,
    total: f64,
    total_std: f64,
}

fn kattis_dist(a: Point, b: Point) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// Cost of the closed tour using exact Kattis distances.
fn tour_cost(points: &[Point], tour: &[usize]) -> i64 {
    let n = tour.len();
    (0..n)
        .map(|i| kattis_dist(points[tour[i]], points[tour[(i + 1) % n]]).round() as i64)
        .sum()
}

fn max_dist(points: &[Point]) -> i64 {
    let mut max = 0;
    for (i, &a) in points.iter().enumerate() {
        for &b in &points[i + 1..] {
            max = max.max(kattis_dist(a, b).round() as i64);
        }
    }
    max
}

fn parse_case(name: &str, text: &str) -> Result<Case, String> {
    let mut tokens = text.split_whitespace();
    let mut next = |what: &str| {
        tokens
            .next()
            .ok_or_else(|| format!("{name}: missing {what}"))
    };
    let int = |s: &str, what: &str| {
        s.parse::<i64>()
            .map_err(|e| format!("{name}: bad {what} {s:?}: {e}"))
    };
    let opt = int(next("opt")?, "opt")?;
    let naive = int(next("naive")?, "naive")?;
    let n = next("n")?
        .parse::<usize>()
        .map_err(|e| format!("{name}: bad n: {e}"))?;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let mut coord = || -> Result<f64, String> {
            let s = next("coordinate")?;
            s.parse()
                .map_err(|e| format!("{name}: bad coordinate {s:?} for point {i}: {e}"))
        };
        points.push((coord()?, coord()?));
    }
    let max = max_dist(&points);
    let scale = (max > i64::from(u16::MAX)).then(|| f64::from(u16::MAX) / max as f64);
    Ok(Case {
        name: name.to_string(),
        opt,
        naive,
        points,
        scale,
    })
}

fn load_cases(dir: &Path, filter: Option<&str>) -> Result<Vec<Case>, String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("Cannot read {}: {e}", dir.display()))?;
    let mut paths: Vec<PathBuf> = entries
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|ext| ext == "txt"))
        .filter(|p| {
            filter.is_none_or(|f| {
                p.file_name()
                    .is_some_and(|n| n.to_string_lossy().contains(f))
            })
        })
        .collect();
    paths.sort();
    if paths.is_empty() {
        return Err(format!("No matching *.txt cases in {}", dir.display()));
    }
    paths
        .iter()
        .map(|p| {
            let text =
                fs::read_to_string(p).map_err(|e| format!("Cannot read {}: {e}", p.display()))?;
            let name = p.file_stem().unwrap_or_default().to_string_lossy();
            parse_case(&name, &text)
        })
        .collect()
}

/// The `u16` matrix the solver sees, with distances scaled by `scale` if given.
fn solver_matrix(points: &[Point], scale: Option<f64>) -> tsp::Result<DistMatrix> {
    let factor = scale.unwrap_or(1.0);
    let n = points.len();
    let mut data = DistMatrix::alloc_buffer(n)?;
    for &a in points {
        // `as u16` saturates, so rounding up past u16::MAX is harmless.
        data.extend(
            points
                .iter()
                .map(|&b| (kattis_dist(a, b) * factor).round() as u16),
        );
    }
    DistMatrix::from_flat(n, data)
}

fn solve(case: &Case, start: Instant, secs: f64, threads: usize) -> tsp::Result<Tour> {
    let dists = solver_matrix(&case.points, case.scale)?;
    let budget = Budget::new(start, secs)?;
    tsp::find_tour(&dists, &budget, threads)
}

fn run_case(case: &Case, secs: f64, threads: usize) -> RunResult {
    let start = Instant::now();
    let result = solve(case, start, secs, threads);
    let elapsed = start.elapsed().as_secs_f64();
    let failed = |error: String| RunResult {
        score: 0.0,
        cost: None,
        elapsed,
        error: Some(error),
    };
    let tour = match result {
        Ok(tour) => tour,
        Err(e) => return failed(format!("solver error: {e}")),
    };
    if !tsp::is_valid_tour(case.points.len(), &tour) {
        return failed("invalid tour".to_string());
    }
    let cost = tour_cost(&case.points, &tour);
    match tsp::score(cost, case.opt, case.naive) {
        Ok(score) => RunResult {
            score,
            cost: Some(cost),
            elapsed,
            error: None,
        },
        Err(e) => failed(format!("bad case data? {e}")),
    }
}

/// Mean and sample standard deviation (0 for fewer than two values).
fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 {
        return (mean, 0.0);
    }
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

fn is_late(elapsed: f64, secs: f64) -> bool {
    elapsed > secs * (1.0 + LATE_TOLERANCE)
}

fn run_config(cases: &[Case], args: &Args, secs: f64, threads: usize) -> Summary {
    println!(
        "\n== time {secs}s, threads {threads}, {} cases x {} repeats ==",
        cases.len(),
        args.repeats
    );
    // totals[r] is the summed score of repeat r.
    let mut totals = vec![0.0; args.repeats];
    let mut slowest = (0.0, "");
    let mut late_runs = 0;
    let mut failed_runs = 0;
    for case in cases {
        let runs: Vec<RunResult> = (0..args.repeats)
            .map(|_| run_case(case, secs, threads))
            .collect();
        for (total, run) in totals.iter_mut().zip(&runs) {
            *total += run.score;
        }
        let scores: Vec<f64> = runs.iter().map(|r| r.score).collect();
        let costs: Vec<f64> = runs
            .iter()
            .filter_map(|r| r.cost)
            .map(|c| c as f64)
            .collect();
        let max_elapsed = runs.iter().map(|r| r.elapsed).fold(0.0, f64::max);
        let late = runs.iter().filter(|r| is_late(r.elapsed, secs)).count();
        let errors: Vec<&str> = runs.iter().filter_map(|r| r.error.as_deref()).collect();
        late_runs += late;
        failed_runs += errors.len();
        if max_elapsed > slowest.0 {
            slowest = (max_elapsed, &case.name);
        }
        if args.quiet {
            continue;
        }
        let (score, score_std) = mean_std(&scores);
        let cost = if costs.is_empty() {
            "-".to_string()
        } else {
            format!("{:.0}", mean_std(&costs).0)
        };
        let mut flags = Vec::new();
        if case.scale.is_some() {
            flags.push("scaled".to_string());
        }
        if late > 0 {
            flags.push(format!("late x{late}"));
        }
        if let Some(first) = errors.first() {
            flags.push(format!("FAILED x{}: {first}", errors.len()));
        }
        let spread = if args.repeats > 1 {
            format!(" ±{score_std:.4}")
        } else {
            String::new()
        };
        let line = format!(
            "{:<14} n={:<5} score {score:.4}{spread}  cost {cost:>8} (opt {}, naive {})  {max_elapsed:.3}s  {}",
            case.name,
            case.points.len(),
            case.opt,
            case.naive,
            flags.join(", ")
        );
        println!("{}", line.trim_end());
    }
    let (total, total_std) = mean_std(&totals);
    let n = cases.len() as f64;
    let spread = if args.repeats > 1 {
        format!("  (±{total_std:.3} across repeats)")
    } else {
        String::new()
    };
    println!("total   {total:.3} / {}{spread}", cases.len());
    println!(
        "scaled  {:.2} / {}",
        total * args.scale_to / n,
        args.scale_to
    );
    println!("mean    {:.4} per case", total / n);
    println!(
        "timing  slowest run {:.4}s ({}), late runs {late_runs}, failed runs {failed_runs}",
        slowest.0, slowest.1
    );
    Summary {
        secs,
        threads,
        total,
        total_std,
    }
}

fn parse_list<T: std::str::FromStr>(flag: &str, value: &str) -> Result<Vec<T>, String> {
    value
        .split(',')
        .map(|s| {
            s.trim()
                .parse()
                .map_err(|_| format!("{flag}: cannot parse {s:?}"))
        })
        .collect()
}

fn parse_args(raw: impl IntoIterator<Item = String>) -> Result<Args, String> {
    let mut args = Args {
        cases: PathBuf::from("cases"),
        times: vec![2.0],
        threads: vec![1],
        repeats: 1,
        filter: None,
        scale_to: 50.0,
        quiet: false,
    };
    let mut raw = raw.into_iter();
    while let Some(arg) = raw.next() {
        let (flag, inline) = match arg.split_once('=') {
            Some((f, v)) => (f.to_string(), Some(v.to_string())),
            None => (arg, None),
        };
        match flag.as_str() {
            "-h" | "--help" => return Err(USAGE.to_string()),
            "--quiet" => {
                args.quiet = true;
                continue;
            }
            _ => {}
        }
        let value = inline
            .or_else(|| raw.next())
            .ok_or_else(|| format!("{flag} needs a value"))?;
        match flag.as_str() {
            "--cases" => args.cases = PathBuf::from(value),
            "--time" => args.times = parse_list(&flag, &value)?,
            "--threads" => args.threads = parse_list(&flag, &value)?,
            "--repeats" => args.repeats = parse_list(&flag, &value)?[0],
            "--filter" => args.filter = Some(value),
            "--scale-to" => args.scale_to = parse_list(&flag, &value)?[0],
            _ => return Err(format!("Unknown option {flag}\n\n{USAGE}")),
        }
    }
    if args.times.iter().any(|&t| t.is_nan() || t <= 0.0) {
        return Err("--time values must be positive".to_string());
    }
    if args.threads.contains(&0) || args.repeats == 0 {
        return Err("--threads and --repeats must be at least 1".to_string());
    }
    Ok(args)
}

fn main() {
    let args = parse_args(env::args().skip(1)).unwrap_or_else(|msg| {
        eprintln!("{msg}");
        process::exit(2);
    });
    let cases = load_cases(&args.cases, args.filter.as_deref()).unwrap_or_else(|msg| {
        eprintln!("{msg}");
        process::exit(1);
    });
    let scaled = cases.iter().filter(|c| c.scale.is_some()).count();
    println!(
        "Loaded {} cases from {} ({scaled} scaled to fit u16 distances)",
        cases.len(),
        args.cases.display()
    );

    let mut summaries = Vec::new();
    for &secs in &args.times {
        for &threads in &args.threads {
            summaries.push(run_config(&cases, &args, secs, threads));
        }
    }

    if let [only] = summaries.as_slice() {
        println!("\nSCORE {:.4}", only.total);
        return;
    }
    println!("\n== sweep ({} cases) ==", cases.len());
    println!(
        "{:>8} {:>8} {:>9} {:>8} {:>9}",
        "time(s)", "threads", "total", "±std", "scaled"
    );
    let n = cases.len() as f64;
    for s in &summaries {
        println!(
            "{:>8} {:>8} {:>9.3} {:>8.3} {:>9.2}",
            s.secs,
            s.threads,
            s.total,
            s.total_std,
            s.total * args.scale_to / n
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "10 20\n4\n0 0\n3 0\n3 4\n0 4\n";

    #[test]
    fn parses_case() {
        let case = parse_case("sample", SAMPLE).unwrap();
        assert_eq!((case.opt, case.naive), (10, 20));
        assert_eq!(
            case.points,
            vec![(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 4.0)]
        );
        assert!(case.scale.is_none());
        assert_eq!(tour_cost(&case.points, &[0, 1, 2, 3]), 14);
        assert_eq!(tour_cost(&case.points, &[0, 2, 1, 3]), 18);
        assert!(parse_case("short", "10 20\n3\n0 0\n1 1\n").is_err());
        assert!(parse_case("bad", "x 20\n1\n0 0\n").is_err());
    }

    #[test]
    fn scales_large_distances_into_u16() {
        let text = "0 0\n3\n0 0\n1000000 0\n0 500000\n";
        let case = parse_case("big", text).unwrap();
        let scale = case.scale.expect("should be scaled");
        let d = solver_matrix(&case.points, Some(scale)).unwrap();
        // The longest edge maps to u16::MAX and the others keep their ratio.
        let max = (0..3)
            .flat_map(|i| (0..3).map(move |j| (i, j)))
            .map(|(i, j)| d.get(i, j))
            .max()
            .unwrap();
        assert_eq!(max, i64::from(u16::MAX));
        assert!((d.get(0, 1) - 2 * d.get(0, 2)).abs() <= 1);
    }

    #[test]
    fn scores_runs() {
        let case = parse_case("sample", "14 18\n4\n0 0\n3 0\n3 4\n0 4\n").unwrap();
        let run = run_case(&case, 0.1, 1);
        assert!(run.error.is_none());
        assert_eq!(run.cost, Some(14));
        assert_eq!(run.score, 1.0);
        // An opt above the real optimum is reported, not a crash.
        let bad = parse_case("bad", "15 18\n4\n0 0\n3 0\n3 4\n0 4\n").unwrap();
        let run = run_case(&bad, 0.1, 1);
        assert_eq!(run.score, 0.0);
        assert!(run.error.is_some());
    }

    #[test]
    fn mean_and_std() {
        assert_eq!(mean_std(&[2.0]), (2.0, 0.0));
        let (m, s) = mean_std(&[1.0, 3.0]);
        assert_eq!(m, 2.0);
        assert!((s - 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn parses_args() {
        let args = parse_args(
            [
                "--time",
                "0.1,2",
                "--threads=1,4",
                "--repeats",
                "3",
                "--quiet",
            ]
            .map(String::from),
        )
        .unwrap();
        assert_eq!(args.times, vec![0.1, 2.0]);
        assert_eq!(args.threads, vec![1, 4]);
        assert_eq!(args.repeats, 3);
        assert!(args.quiet);
        assert!(parse_args(["--time", "0"].map(String::from)).is_err());
        assert!(parse_args(["--threads", "0"].map(String::from)).is_err());
        assert!(parse_args(["--nope", "1"].map(String::from)).is_err());
        assert!(parse_args(["--time"].map(String::from)).is_err());
    }
}
