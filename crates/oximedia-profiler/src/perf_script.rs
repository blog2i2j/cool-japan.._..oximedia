//! Linux `perf script`–compatible output exporter.
//!
//! Produces text in the format emitted by `perf script`, which many
//! visualisation and post-processing tools can consume.  The canonical line
//! format is:
//!
//! ```text
//! process  pid [cpu] timestamp:  function+offset
//! ```
//!
//! For example:
//!
//! ```text
//! oximedia 12345 [000] 1.234567:  encode_frame+0x0
//! oximedia 12345 [000] 1.234600:  av1_encode+0x4c
//! ```
//!
//! # Reference
//!
//! The format is described in the `perf-script(1)` man page.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// PerfSample
// ---------------------------------------------------------------------------

/// A single profiling sample to be exported in `perf script` format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerfSample {
    /// Process name (up to 15 chars recommended for alignment).
    pub process: String,
    /// Process (or thread) id.
    pub pid: u32,
    /// CPU index on which the sample was taken.
    pub cpu: u32,
    /// Timestamp in fractional seconds since session start.
    pub timestamp_secs: f64,
    /// Function (symbol) name.
    pub function: String,
    /// Byte offset within the function.
    pub offset: u64,
}

impl PerfSample {
    /// Create a new `PerfSample`.
    #[must_use]
    pub fn new(
        process: impl Into<String>,
        pid: u32,
        cpu: u32,
        timestamp_secs: f64,
        function: impl Into<String>,
        offset: u64,
    ) -> Self {
        Self {
            process: process.into(),
            pid,
            cpu,
            timestamp_secs,
            function: function.into(),
            offset,
        }
    }

    /// Format this sample as a single `perf script` output line.
    ///
    /// Format:  `{process} {pid} [{cpu:03}] {ts:.6}:  {function}+0x{offset:x}`
    #[must_use]
    pub fn to_perf_line(&self) -> String {
        format!(
            "{} {} [{:03}] {:.6}:  {}+0x{:x}",
            self.process, self.pid, self.cpu, self.timestamp_secs, self.function, self.offset
        )
    }
}

// ---------------------------------------------------------------------------
// PerfScriptExporter
// ---------------------------------------------------------------------------

/// Exports a collection of [`PerfSample`]s in `perf script` output format.
///
/// # Example
///
/// ```
/// use oximedia_profiler::perf_script::{PerfSample, PerfScriptExporter};
///
/// let samples = vec![
///     PerfSample::new("oximedia", 1234, 0, 0.000100, "encode_frame", 0),
///     PerfSample::new("oximedia", 1234, 0, 0.000250, "av1_encode",   0x4c),
/// ];
/// let output = PerfScriptExporter::export(&samples);
/// assert!(output.contains("encode_frame+0x0"));
/// assert!(output.contains("av1_encode+0x4c"));
/// ```
pub struct PerfScriptExporter;

impl PerfScriptExporter {
    /// Convert a slice of `PerfSample`s into a `perf script`-compatible
    /// multi-line string.
    ///
    /// Samples are emitted in the order provided; no sorting is performed.
    #[must_use]
    pub fn export(samples: &[PerfSample]) -> String {
        let mut out = String::new();
        for sample in samples {
            out.push_str(&sample.to_perf_line());
            out.push('\n');
        }
        out
    }

    /// Export samples that fall within the half-open time range
    /// `[start_secs, end_secs)`.
    #[must_use]
    pub fn export_range(samples: &[PerfSample], start_secs: f64, end_secs: f64) -> String {
        let filtered: Vec<&PerfSample> = samples
            .iter()
            .filter(|s| s.timestamp_secs >= start_secs && s.timestamp_secs < end_secs)
            .collect();
        let mut out = String::new();
        for sample in filtered {
            out.push_str(&sample.to_perf_line());
            out.push('\n');
        }
        out
    }

    /// Return sample count within a time range (useful for density analysis).
    #[must_use]
    pub fn count_in_range(samples: &[PerfSample], start_secs: f64, end_secs: f64) -> usize {
        samples
            .iter()
            .filter(|s| s.timestamp_secs >= start_secs && s.timestamp_secs < end_secs)
            .count()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(ts: f64, func: &str) -> PerfSample {
        PerfSample::new("oximedia", 1000, 0, ts, func, 0)
    }

    #[test]
    fn test_export_empty() {
        let out = PerfScriptExporter::export(&[]);
        assert!(out.is_empty());
    }

    #[test]
    fn test_export_single() {
        let s = sample(1.234567, "my_function");
        let out = PerfScriptExporter::export(&[s]);
        assert!(out.contains("oximedia"));
        assert!(out.contains("my_function+0x0"));
        assert!(out.contains("1.234567"));
    }

    #[test]
    fn test_export_multiple_lines() {
        let samples = vec![sample(0.1, "fn_a"), sample(0.2, "fn_b"), sample(0.3, "fn_c")];
        let out = PerfScriptExporter::export(&samples);
        let lines: Vec<&str> = out.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("fn_a"));
        assert!(lines[1].contains("fn_b"));
        assert!(lines[2].contains("fn_c"));
    }

    #[test]
    fn test_perf_line_format() {
        let s = PerfSample::new("myproc", 9999, 2, 0.000100, "encode", 0x20);
        let line = s.to_perf_line();
        assert!(line.starts_with("myproc 9999 [002] 0.000100:  encode+0x20"));
    }

    #[test]
    fn test_export_range_filters_correctly() {
        let samples = vec![sample(0.0, "a"), sample(0.5, "b"), sample(1.0, "c")];
        let out = PerfScriptExporter::export_range(&samples, 0.25, 0.75);
        assert!(!out.contains("fn_a"));
        assert!(out.contains("fn_b") || out.contains("b+0x0"));
        assert!(!out.contains("fn_c"));
    }

    #[test]
    fn test_count_in_range() {
        let samples = vec![
            sample(0.0, "a"),
            sample(0.5, "b"),
            sample(1.0, "c"),
            sample(1.5, "d"),
        ];
        assert_eq!(PerfScriptExporter::count_in_range(&samples, 0.0, 1.0), 2);
        assert_eq!(PerfScriptExporter::count_in_range(&samples, 0.0, 2.0), 4);
        assert_eq!(PerfScriptExporter::count_in_range(&samples, 2.0, 3.0), 0);
    }

    #[test]
    fn test_offset_hex_formatting() {
        let s = PerfSample::new("p", 1, 0, 0.0, "fn", 0xdeadbeef);
        let line = s.to_perf_line();
        assert!(line.contains("fn+0xdeadbeef"));
    }

    #[test]
    fn test_cpu_index_zero_padded() {
        let s = PerfSample::new("p", 1, 5, 0.0, "fn", 0);
        let line = s.to_perf_line();
        // CPU index must be formatted as [005]
        assert!(line.contains("[005]"));
    }

    #[test]
    fn test_serialization_roundtrip() {
        let original = PerfSample::new("oximedia", 4242, 1, 3.141592, "codec_init", 0x100);
        let json = serde_json::to_string(&original).expect("serialization failed");
        let restored: PerfSample = serde_json::from_str(&json).expect("deserialization failed");
        assert_eq!(restored.process, original.process);
        assert_eq!(restored.pid, original.pid);
        assert_eq!(restored.cpu, original.cpu);
        assert_eq!(restored.function, original.function);
        assert_eq!(restored.offset, original.offset);
        assert!((restored.timestamp_secs - original.timestamp_secs).abs() < 1e-9);
    }
}
