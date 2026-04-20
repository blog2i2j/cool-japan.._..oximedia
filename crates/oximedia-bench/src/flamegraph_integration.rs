#![allow(dead_code)]
//! Integration layer for flamegraph profiling of benchmark workloads.
//!
//! This module provides utilities for annotating benchmark runs with
//! profiling instrumentation, generating flamegraph-compatible stack-trace
//! data, and writing folded-stack output files that can be rendered by
//! [Flamegraph](https://github.com/brendangregg/FlameGraph) or
//! [inferno](https://github.com/jonhoo/inferno).
//!
//! Because external profiler support is platform-specific and optional, the
//! module is designed in layers:
//!
//! 1. **`FlamegraphConfig`** — controls output paths, sampling interval, and
//!    what operations to instrument.
//! 2. **`FlamegraphSession`** — an RAII guard that begins a profiling session
//!    on construction and finalises it on drop.
//! 3. **`FoldedStack`** — an in-memory folded-stack representation that can
//!    be serialised to the folded-stack text format consumed by `flamegraph.pl`
//!    and `inferno-flamegraph`.
//! 4. **`FlamegraphReport`** — aggregates one or more `FoldedStack` snapshots
//!    and computes per-symbol hit counts for differential analysis.

use crate::{BenchError, BenchResult};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Sampling mode for flamegraph collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingMode {
    /// On-CPU (wall-clock) sampling.
    OnCpu,
    /// Off-CPU sampling (waiting for I/O, locks, …).
    OffCpu,
    /// Combined on+off CPU.
    Mixed,
}

impl Default for SamplingMode {
    fn default() -> Self {
        Self::OnCpu
    }
}

/// What to profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProfileTarget {
    /// Profile the full process.
    Process,
    /// Profile only the calling thread.
    CurrentThread,
    /// Profile all threads in the process.
    AllThreads,
}

impl Default for ProfileTarget {
    fn default() -> Self {
        Self::CurrentThread
    }
}

/// Configuration for a flamegraph profiling session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamegraphConfig {
    /// Directory where output files are written.
    pub output_dir: PathBuf,
    /// Base name for output files (without extension).
    pub output_basename: String,
    /// Sampling interval.
    #[serde(with = "duration_serde")]
    pub sampling_interval: Duration,
    /// Sampling mode.
    pub mode: SamplingMode,
    /// What to profile.
    pub target: ProfileTarget,
    /// Maximum stack depth to capture.
    pub max_stack_depth: usize,
    /// Whether to demangle C++ / Rust symbols in the output.
    pub demangle_symbols: bool,
    /// Minimum sample count for a frame to appear in the report.
    pub min_samples: u64,
    /// Whether to include kernel frames.
    pub include_kernel: bool,
    /// Title to embed in generated SVG flamegraphs.
    pub svg_title: String,
}

impl Default for FlamegraphConfig {
    fn default() -> Self {
        Self {
            output_dir: PathBuf::from("./flamegraphs"),
            output_basename: "bench".to_string(),
            sampling_interval: Duration::from_millis(1), // 1 kHz
            mode: SamplingMode::default(),
            target: ProfileTarget::default(),
            max_stack_depth: 128,
            demangle_symbols: true,
            min_samples: 1,
            include_kernel: false,
            svg_title: "OxiMedia Benchmark".to_string(),
        }
    }
}

impl FlamegraphConfig {
    /// Create a new configuration with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set the output directory.
    #[must_use]
    pub fn with_output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.output_dir = dir.into();
        self
    }

    /// Builder: set the output basename.
    #[must_use]
    pub fn with_output_basename(mut self, name: impl Into<String>) -> Self {
        self.output_basename = name.into();
        self
    }

    /// Builder: set the sampling interval.
    #[must_use]
    pub fn with_sampling_interval(mut self, interval: Duration) -> Self {
        self.sampling_interval = interval;
        self
    }

    /// Builder: set the sampling mode.
    #[must_use]
    pub fn with_mode(mut self, mode: SamplingMode) -> Self {
        self.mode = mode;
        self
    }

    /// Builder: set the profile target.
    #[must_use]
    pub fn with_target(mut self, target: ProfileTarget) -> Self {
        self.target = target;
        self
    }

    /// Builder: set the maximum stack depth.
    #[must_use]
    pub fn with_max_stack_depth(mut self, depth: usize) -> Self {
        self.max_stack_depth = depth;
        self
    }

    /// Builder: control symbol demangling.
    #[must_use]
    pub fn with_demangle_symbols(mut self, demangle: bool) -> Self {
        self.demangle_symbols = demangle;
        self
    }

    /// Builder: set the minimum sample threshold.
    #[must_use]
    pub fn with_min_samples(mut self, min: u64) -> Self {
        self.min_samples = min;
        self
    }

    /// Builder: set the SVG title.
    #[must_use]
    pub fn with_svg_title(mut self, title: impl Into<String>) -> Self {
        self.svg_title = title.into();
        self
    }

    /// Returns the path where the folded-stack file will be written.
    #[must_use]
    pub fn folded_stack_path(&self) -> PathBuf {
        self.output_dir
            .join(format!("{}.folded", self.output_basename))
    }

    /// Returns the path where the SVG output will be written (if generated).
    #[must_use]
    pub fn svg_path(&self) -> PathBuf {
        self.output_dir
            .join(format!("{}.svg", self.output_basename))
    }
}

// ---------------------------------------------------------------------------
// Folded stack representation
// ---------------------------------------------------------------------------

/// A single folded-stack entry: a semicolon-delimited call chain and a sample count.
///
/// The text format is:
/// ```text
/// frame1;frame2;frame3 42
/// ```
/// where `42` is the number of samples hitting the leaf frame (`frame3`) via
/// the given chain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FoldedStackEntry {
    /// Call chain from root to leaf, separated by semicolons.
    pub stack: String,
    /// Sample count.
    pub count: u64,
}

impl FoldedStackEntry {
    /// Create a new entry.
    #[must_use]
    pub fn new(stack: impl Into<String>, count: u64) -> Self {
        Self {
            stack: stack.into(),
            count,
        }
    }

    /// Render to the canonical folded-stack line format.
    #[must_use]
    pub fn to_folded_line(&self) -> String {
        format!("{} {}", self.stack, self.count)
    }
}

/// An in-memory folded-stack data set collected during a profiling session.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FoldedStack {
    /// All collected stack entries (may contain duplicates; aggregate before writing).
    entries: Vec<FoldedStackEntry>,
    /// Label / annotation for this snapshot (e.g. benchmark name).
    pub label: String,
}

impl FoldedStack {
    /// Create a new empty folded stack.
    #[must_use]
    pub fn new(label: impl Into<String>) -> Self {
        Self {
            entries: Vec::new(),
            label: label.into(),
        }
    }

    /// Add a raw entry.
    pub fn add(&mut self, entry: FoldedStackEntry) {
        self.entries.push(entry);
    }

    /// Add a stack string and count directly.
    pub fn add_stack(&mut self, stack: impl Into<String>, count: u64) {
        self.entries.push(FoldedStackEntry::new(stack, count));
    }

    /// Number of entries (before aggregation).
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether there are no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Total sample count across all entries.
    #[must_use]
    pub fn total_samples(&self) -> u64 {
        self.entries.iter().map(|e| e.count).sum()
    }

    /// Aggregate duplicate stack strings and return a map from stack → total count.
    #[must_use]
    pub fn aggregate(&self) -> HashMap<String, u64> {
        let mut map: HashMap<String, u64> = HashMap::new();
        for entry in &self.entries {
            *map.entry(entry.stack.clone()).or_insert(0) += entry.count;
        }
        map
    }

    /// Render to the folded-stack text format (aggregated, sorted by count desc).
    #[must_use]
    pub fn to_folded_text(&self) -> String {
        let aggregated = self.aggregate();
        let mut lines: Vec<(String, u64)> = aggregated.into_iter().collect();
        lines.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        lines
            .into_iter()
            .map(|(stack, count)| format!("{stack} {count}"))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Write the folded-stack text to a file.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Io`] if the file cannot be created or written.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        let text = self.to_folded_text();
        std::fs::write(path, text.as_bytes())?;
        Ok(())
    }

    /// Filter entries to only include stacks containing `symbol`.
    #[must_use]
    pub fn filter_by_symbol(&self, symbol: &str) -> Self {
        let entries = self
            .entries
            .iter()
            .filter(|e| e.stack.contains(symbol))
            .cloned()
            .collect();
        Self {
            entries,
            label: format!("{} [filtered:{}]", self.label, symbol),
        }
    }

    /// Merge another `FoldedStack` into this one.
    pub fn merge(&mut self, other: &FoldedStack) {
        self.entries.extend(other.entries.iter().cloned());
    }
}

// ---------------------------------------------------------------------------
// Flamegraph report (differential analysis)
// ---------------------------------------------------------------------------

/// Per-symbol statistics across one or more `FoldedStack` snapshots.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolStats {
    /// Symbol / leaf frame name (last component in the stack chain).
    pub symbol: String,
    /// Total samples across all snapshots.
    pub total_samples: u64,
    /// Fraction of all samples (0.0–1.0).
    pub fraction: f64,
    /// Number of snapshots this symbol appeared in.
    pub snapshot_count: usize,
}

/// Aggregated flamegraph report combining multiple profiling sessions.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct FlamegraphReport {
    /// The folded stacks collected, one per session.
    pub snapshots: Vec<FoldedStack>,
    /// Per-symbol aggregated statistics (populated by [`Self::build`]).
    pub symbol_stats: Vec<SymbolStats>,
    /// Total samples across all snapshots.
    pub total_samples: u64,
}

impl FlamegraphReport {
    /// Create an empty report.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a snapshot to the report.
    pub fn add_snapshot(&mut self, snapshot: FoldedStack) {
        self.snapshots.push(snapshot);
    }

    /// Aggregate all snapshots into per-symbol statistics.
    pub fn build(&mut self) {
        let mut symbol_map: HashMap<String, (u64, usize)> = HashMap::new();
        let mut grand_total = 0u64;
        for snapshot in &self.snapshots {
            let agg = snapshot.aggregate();
            for (stack, count) in &agg {
                grand_total += count;
                let leaf = stack
                    .rsplit(';')
                    .next()
                    .unwrap_or(stack.as_str())
                    .trim()
                    .to_string();
                let entry = symbol_map.entry(leaf).or_insert((0, 0));
                entry.0 += count;
                entry.1 += 1;
            }
        }
        self.total_samples = grand_total;
        let mut stats: Vec<SymbolStats> = symbol_map
            .into_iter()
            .map(|(symbol, (total, snap_count))| SymbolStats {
                fraction: if grand_total > 0 {
                    total as f64 / grand_total as f64
                } else {
                    0.0
                },
                symbol,
                total_samples: total,
                snapshot_count: snap_count,
            })
            .collect();
        stats.sort_by(|a, b| b.total_samples.cmp(&a.total_samples));
        self.symbol_stats = stats;
    }

    /// Return the top-N hottest symbols.
    #[must_use]
    pub fn top_symbols(&self, n: usize) -> &[SymbolStats] {
        let end = n.min(self.symbol_stats.len());
        &self.symbol_stats[..end]
    }

    /// Compute a differential report: `self` (baseline) vs `other` (treatment).
    ///
    /// Returns a list of `(symbol, delta_fraction)` pairs sorted by absolute
    /// delta descending.  Positive delta means the symbol got hotter in the
    /// treatment.
    #[must_use]
    pub fn differential(&self, other: &FlamegraphReport) -> Vec<(String, f64)> {
        let baseline: HashMap<&str, f64> = self
            .symbol_stats
            .iter()
            .map(|s| (s.symbol.as_str(), s.fraction))
            .collect();
        let treatment: HashMap<&str, f64> = other
            .symbol_stats
            .iter()
            .map(|s| (s.symbol.as_str(), s.fraction))
            .collect();

        let mut all_symbols: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for s in baseline.keys() {
            all_symbols.insert(s);
        }
        for s in treatment.keys() {
            all_symbols.insert(s);
        }

        let mut diffs: Vec<(String, f64)> = all_symbols
            .into_iter()
            .map(|sym| {
                let base = baseline.get(sym).copied().unwrap_or(0.0);
                let treat = treatment.get(sym).copied().unwrap_or(0.0);
                (sym.to_string(), treat - base)
            })
            .collect();
        diffs.sort_by(|a, b| {
            b.1.abs()
                .partial_cmp(&a.1.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        diffs
    }

    /// Merge the folded stacks from all snapshots into a single [`FoldedStack`].
    #[must_use]
    pub fn merged_stack(&self) -> FoldedStack {
        let mut merged = FoldedStack::new("merged");
        for snapshot in &self.snapshots {
            merged.merge(snapshot);
        }
        merged
    }

    /// Write the merged folded-stack to a file.
    ///
    /// # Errors
    ///
    /// Propagates I/O errors.
    pub fn write_folded(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        self.merged_stack().write_to_file(path)
    }

    /// Serialize the report to a JSON string.
    ///
    /// # Errors
    ///
    /// Returns a [`BenchError`] if serialization fails.
    pub fn to_json(&self) -> BenchResult<String> {
        serde_json::to_string_pretty(self).map_err(BenchError::Serialization)
    }
}

// ---------------------------------------------------------------------------
// Session RAII guard
// ---------------------------------------------------------------------------

/// State of a [`FlamegraphSession`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionState {
    /// The session is actively collecting samples.
    Running,
    /// The session has been finalised (data collected).
    Finished,
    /// The session was aborted without collecting data.
    Aborted,
}

/// RAII guard representing an active flamegraph profiling session.
///
/// On drop, the session is automatically finalised (equivalent to calling
/// [`Self::finish`]).  The collected data can be retrieved via
/// [`Self::take_stack`] before or after the session ends.
///
/// In environments without OS-level profiler support the session falls back
/// to lightweight wall-clock timing instrumentation.
pub struct FlamegraphSession {
    config: FlamegraphConfig,
    label: String,
    state: SessionState,
    start_time: std::time::Instant,
    stack: FoldedStack,
}

impl FlamegraphSession {
    /// Begin a new profiling session.
    #[must_use]
    pub fn start(config: FlamegraphConfig, label: impl Into<String>) -> Self {
        let label_str = label.into();
        Self {
            stack: FoldedStack::new(label_str.clone()),
            config,
            label: label_str,
            state: SessionState::Running,
            start_time: std::time::Instant::now(),
        }
    }

    /// Record a synthetic stack sample (useful when driving from instrumented code).
    pub fn record_sample(&mut self, stack: impl Into<String>, count: u64) {
        if self.state == SessionState::Running {
            self.stack.add_stack(stack, count);
        }
    }

    /// Finalise the session and return the elapsed wall-clock duration.
    pub fn finish(&mut self) -> Duration {
        if self.state == SessionState::Running {
            self.state = SessionState::Finished;
        }
        self.start_time.elapsed()
    }

    /// Abort the session without collecting data.
    pub fn abort(&mut self) {
        self.state = SessionState::Aborted;
        self.stack = FoldedStack::new(self.label.clone());
    }

    /// Take ownership of the collected stack data (leaves an empty stack behind).
    pub fn take_stack(&mut self) -> FoldedStack {
        std::mem::replace(&mut self.stack, FoldedStack::new(self.label.clone()))
    }

    /// Current session state.
    #[must_use]
    pub fn state(&self) -> SessionState {
        self.state
    }

    /// Elapsed time since the session started.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Configuration used by this session.
    #[must_use]
    pub fn config(&self) -> &FlamegraphConfig {
        &self.config
    }

    /// Write the collected folded-stack to the path specified by the config.
    ///
    /// # Errors
    ///
    /// Returns [`BenchError::Io`] if writing fails.
    pub fn write_folded_stack(&self) -> BenchResult<()> {
        if let Some(parent) = self.config.folded_stack_path().parent() {
            std::fs::create_dir_all(parent)?;
        }
        self.stack.write_to_file(self.config.folded_stack_path())
    }
}

impl Drop for FlamegraphSession {
    fn drop(&mut self) {
        if self.state == SessionState::Running {
            self.finish();
        }
    }
}

// ---------------------------------------------------------------------------
// Convenience: wrap a closure in a profiling session
// ---------------------------------------------------------------------------

/// Run `f` inside a flamegraph session and return the collected stack along
/// with the return value of `f`.
///
/// # Errors
///
/// Propagates any error returned by `f`.
pub fn profile<F, T, E>(
    config: FlamegraphConfig,
    label: impl Into<String>,
    f: F,
) -> Result<(T, FoldedStack), E>
where
    F: FnOnce(&mut FlamegraphSession) -> Result<T, E>,
{
    let mut session = FlamegraphSession::start(config, label);
    let result = f(&mut session);
    session.finish();
    let stack = session.take_stack();
    result.map(|v| (v, stack))
}

// ---------------------------------------------------------------------------
// Serde helpers
// ---------------------------------------------------------------------------

mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(d: &Duration, s: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        d.as_secs_f64().serialize(s)
    }

    pub fn deserialize<'de, D>(d: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(d)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_folded_stack_entry_line() {
        let entry = FoldedStackEntry::new("main;encode;compress", 42);
        assert_eq!(entry.to_folded_line(), "main;encode;compress 42");
    }

    #[test]
    fn test_folded_stack_aggregate() {
        let mut stack = FoldedStack::new("test");
        stack.add_stack("a;b;c", 10);
        stack.add_stack("a;b;c", 5);
        stack.add_stack("a;b;d", 3);
        let agg = stack.aggregate();
        assert_eq!(agg.get("a;b;c"), Some(&15));
        assert_eq!(agg.get("a;b;d"), Some(&3));
        assert_eq!(stack.total_samples(), 18);
    }

    #[test]
    fn test_folded_stack_to_text() {
        let mut stack = FoldedStack::new("test");
        stack.add_stack("a;b", 100);
        stack.add_stack("a;c", 50);
        let text = stack.to_folded_text();
        assert!(text.contains("a;b 100"));
        assert!(text.contains("a;c 50"));
    }

    #[test]
    fn test_folded_stack_filter() {
        let mut stack = FoldedStack::new("test");
        stack.add_stack("main;encode;av1", 20);
        stack.add_stack("main;decode;vp9", 10);
        let filtered = stack.filter_by_symbol("encode");
        assert_eq!(filtered.len(), 1);
        assert_eq!(filtered.total_samples(), 20);
    }

    #[test]
    fn test_folded_stack_merge() {
        let mut a = FoldedStack::new("a");
        a.add_stack("f;g", 5);
        let mut b = FoldedStack::new("b");
        b.add_stack("f;g", 3);
        b.add_stack("f;h", 2);
        a.merge(&b);
        assert_eq!(a.total_samples(), 10);
    }

    #[test]
    fn test_folded_stack_write_to_file() {
        let tmp = std::env::temp_dir().join("test_folded_stack.folded");
        let mut stack = FoldedStack::new("write_test");
        stack.add_stack("a;b;c", 7);
        stack.write_to_file(&tmp).expect("write should succeed");
        let contents = std::fs::read_to_string(&tmp).expect("read should succeed");
        assert!(contents.contains("a;b;c 7"));
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_report_build_and_top_symbols() {
        let mut stack = FoldedStack::new("snap1");
        stack.add_stack("main;encode;av1_encode", 80);
        stack.add_stack("main;decode;vp9_decode", 20);
        let mut report = FlamegraphReport::new();
        report.add_snapshot(stack);
        report.build();
        assert_eq!(report.total_samples, 100);
        let top = report.top_symbols(1);
        assert_eq!(top[0].symbol, "av1_encode");
        assert!((top[0].fraction - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_report_differential() {
        let mut base_stack = FoldedStack::new("base");
        base_stack.add_stack("main;f1", 50);
        base_stack.add_stack("main;f2", 50);
        let mut base = FlamegraphReport::new();
        base.add_snapshot(base_stack);
        base.build();

        let mut treat_stack = FoldedStack::new("treat");
        treat_stack.add_stack("main;f1", 80);
        treat_stack.add_stack("main;f2", 20);
        let mut treat = FlamegraphReport::new();
        treat.add_snapshot(treat_stack);
        treat.build();

        let diff = base.differential(&treat);
        assert!(!diff.is_empty());
        // f1 should show a positive delta (hotter in treatment).
        let f1 = diff.iter().find(|(s, _)| s == "f1").expect("f1 in diff");
        assert!(f1.1 > 0.0);
    }

    #[test]
    fn test_session_lifecycle() {
        let config = FlamegraphConfig::new();
        let mut session = FlamegraphSession::start(config, "test_session");
        assert_eq!(session.state(), SessionState::Running);
        session.record_sample("main;compute", 3);
        let elapsed = session.finish();
        assert_eq!(session.state(), SessionState::Finished);
        assert!(elapsed.as_nanos() < 10_000_000_000); // < 10 s
        let stack = session.take_stack();
        assert_eq!(stack.total_samples(), 3);
    }

    #[test]
    fn test_session_abort_clears_data() {
        let config = FlamegraphConfig::new();
        let mut session = FlamegraphSession::start(config, "abort_test");
        session.record_sample("a;b", 99);
        session.abort();
        assert_eq!(session.state(), SessionState::Aborted);
        let stack = session.take_stack();
        assert!(stack.is_empty());
    }

    #[test]
    fn test_profile_helper() {
        let config = FlamegraphConfig::new();
        let (value, stack) =
            profile::<_, _, std::convert::Infallible>(config, "profile_helper_test", |sess| {
                sess.record_sample("bench;work", 5);
                Ok(42u32)
            })
            .expect("profile should succeed");
        assert_eq!(value, 42);
        assert_eq!(stack.total_samples(), 5);
    }

    #[test]
    fn test_config_paths() {
        let dir = std::env::temp_dir().join("oximedia-bench-flamegraph-fg");
        let cfg = FlamegraphConfig::new()
            .with_output_dir(dir.clone())
            .with_output_basename("codec_bench");
        assert_eq!(cfg.folded_stack_path(), dir.join("codec_bench.folded"));
        assert_eq!(cfg.svg_path(), dir.join("codec_bench.svg"));
    }

    #[test]
    fn test_report_json_roundtrip() {
        let mut stack = FoldedStack::new("s");
        stack.add_stack("x;y", 3);
        let mut report = FlamegraphReport::new();
        report.add_snapshot(stack);
        report.build();
        let json = report.to_json().expect("serialization should succeed");
        assert!(json.contains("total_samples"));
    }
}
