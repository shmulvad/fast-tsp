//! TSP solver with Python bindings, exposed as `fast_tsp._core`.
//!
//! The solver itself lives in [`tsp`] and has no Python dependencies.
//! The bindings in `python.rs` are only built with the `python` feature.

pub mod tsp;

#[cfg(feature = "python")]
mod python;
