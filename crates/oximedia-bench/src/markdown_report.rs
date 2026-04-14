//! Markdown report generation for benchmark results.
//!
//! Produces a self-contained Markdown document (GitHub-Flavoured Markdown)
//! containing:
//!
//! - Summary metadata (timestamp, duration, configuration overview)
//! - Per-codec result tables with quality metrics and throughput
//! - Aggregate statistics section
//!
//! The output is suitable for embedding in CI artefacts, GitHub PR comments,
//! or rendering in any GFM-capable viewer.
//!
//! # Example
//!
//! ```
//! use oximedia_bench::markdown_report::MarkdownReport;
//! use oximedia_bench::{BenchmarkResults, BenchmarkConfig};
//! use std::time::Duration;
//!
//! let results = BenchmarkResults {
//!     codec_results: vec![],
//!     timestamp: "2025-06-01T12:00:00Z".to_string(),
//!     total_duration: Duration::from_secs(42),
//!     config: BenchmarkConfig::default(),
//! };
//!
//! let md = MarkdownReport::new(&results).render();
//! assert!(md.contains("# OxiMedia Codec Benchmark Results"));
//! ```

use crate::{BenchResult, BenchmarkResults, BenchmarkUtils, CodecBenchmarkResult, SequenceResult};
use std::fmt::Write as FmtWrite;
use std::path::Path;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Controls what sections are included in the Markdown report.
#[derive(Debug, Clone)]
pub struct MarkdownReportOptions {
    /// Include the summary header section.
    pub include_summary: bool,
    /// Include per-codec detail tables.
    pub include_detail_tables: bool,
    /// Include aggregate statistics section.
    pub include_statistics: bool,
    /// Include configuration dump section.
    pub include_config: bool,
    /// Maximum number of decimal places for floating-point values.
    pub float_precision: usize,
}

impl Default for MarkdownReportOptions {
    fn default() -> Self {
        Self {
            include_summary: true,
            include_detail_tables: true,
            include_statistics: true,
            include_config: true,
            float_precision: 2,
        }
    }
}

// ---------------------------------------------------------------------------
// Report builder
// ---------------------------------------------------------------------------

/// Markdown report generator.
pub struct MarkdownReport<'a> {
    results: &'a BenchmarkResults,
    options: MarkdownReportOptions,
}

impl<'a> MarkdownReport<'a> {
    /// Create a new report with default options.
    #[must_use]
    pub fn new(results: &'a BenchmarkResults) -> Self {
        Self {
            results,
            options: MarkdownReportOptions::default(),
        }
    }

    /// Create a report with custom options.
    #[must_use]
    pub fn with_options(results: &'a BenchmarkResults, options: MarkdownReportOptions) -> Self {
        Self { results, options }
    }

    /// Render the full report as a Markdown string.
    #[must_use]
    pub fn render(&self) -> String {
        let mut md = String::with_capacity(4096);

        if self.options.include_summary {
            self.write_summary(&mut md);
        }

        if self.options.include_detail_tables {
            self.write_detail_tables(&mut md);
        }

        if self.options.include_statistics {
            self.write_statistics(&mut md);
        }

        if self.options.include_config {
            self.write_config_section(&mut md);
        }

        md
    }

    /// Render and write directly to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn write_to_file(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        let content = self.render();
        std::fs::write(path, content)?;
        Ok(())
    }

    // ------------------------------------------------------------------
    // Sections
    // ------------------------------------------------------------------

    fn write_summary(&self, md: &mut String) {
        let _ = writeln!(md, "# OxiMedia Codec Benchmark Results\n");
        let _ = writeln!(md, "| Field | Value |");
        let _ = writeln!(md, "|-------|-------|");
        let _ = writeln!(md, "| Timestamp | {} |", self.results.timestamp);
        let _ = writeln!(
            md,
            "| Total Duration | {} |",
            BenchmarkUtils::format_duration(self.results.total_duration)
        );
        let _ = writeln!(
            md,
            "| Codecs Tested | {} |",
            self.results.codec_results.len()
        );

        let total_sequences: usize = self
            .results
            .codec_results
            .iter()
            .map(|c| c.sequence_results.len())
            .sum();
        let _ = writeln!(md, "| Total Sequences | {total_sequences} |");
        let _ = writeln!(md);
    }

    fn write_detail_tables(&self, md: &mut String) {
        let _ = writeln!(md, "## Per-Codec Results\n");

        for codec_result in &self.results.codec_results {
            self.write_codec_table(md, codec_result);
        }
    }

    fn write_codec_table(&self, md: &mut String, codec: &CodecBenchmarkResult) {
        let _ = writeln!(md, "### {:?}\n", codec.codec_id);

        if let Some(ref preset) = codec.preset {
            let _ = writeln!(md, "**Preset:** {preset}\n");
        }
        if let Some(bitrate) = codec.bitrate_kbps {
            let _ = writeln!(md, "**Target Bitrate:** {bitrate} kbps\n");
        }
        if let Some(cq) = codec.cq_level {
            let _ = writeln!(md, "**CQ Level:** {cq}\n");
        }

        let _ = writeln!(
            md,
            "| Sequence | Enc FPS | Dec FPS | Size | PSNR (dB) | SSIM |"
        );
        let _ = writeln!(
            md,
            "|----------|---------|---------|------|-----------|------|"
        );

        let prec = self.options.float_precision;

        for seq in &codec.sequence_results {
            let _ = writeln!(
                md,
                "| {} | {:.prec$} | {:.prec$} | {} | {} | {} |",
                seq.sequence_name,
                seq.encoding_fps,
                seq.decoding_fps,
                BenchmarkUtils::format_bytes(seq.file_size_bytes),
                format_opt_f64(seq.metrics.psnr, prec),
                format_opt_f64(seq.metrics.ssim, prec + 2),
            );
        }

        let _ = writeln!(md);
    }

    fn write_statistics(&self, md: &mut String) {
        let _ = writeln!(md, "## Aggregate Statistics\n");

        let _ = writeln!(
            md,
            "| Codec | Mean Enc FPS | Mean Dec FPS | Mean PSNR | Mean SSIM |"
        );
        let _ = writeln!(
            md,
            "|-------|-------------|-------------|-----------|-----------|"
        );

        let prec = self.options.float_precision;
        for codec in &self.results.codec_results {
            let s = &codec.statistics;
            let label = if let Some(ref p) = codec.preset {
                format!("{:?} ({})", codec.codec_id, p)
            } else {
                format!("{:?}", codec.codec_id)
            };
            let _ = writeln!(
                md,
                "| {} | {:.prec$} | {:.prec$} | {} | {} |",
                label,
                s.mean_encoding_fps,
                s.mean_decoding_fps,
                format_opt_f64(s.mean_psnr, prec),
                format_opt_f64(s.mean_ssim, prec + 2),
            );
        }

        let _ = writeln!(md);
    }

    fn write_config_section(&self, md: &mut String) {
        let _ = writeln!(md, "## Configuration\n");
        let _ = writeln!(md, "```json");
        if let Ok(json) = serde_json::to_string_pretty(&self.results.config) {
            let _ = writeln!(md, "{json}");
        }
        let _ = writeln!(md, "```\n");
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_opt_f64(value: Option<f64>, precision: usize) -> String {
    value.map_or_else(|| "N/A".to_string(), |v| format!("{v:.precision$}"))
}

/// Render a single [`SequenceResult`] as a one-line Markdown summary.
///
/// Useful for embedding individual results in PR comments.
#[must_use]
pub fn sequence_result_one_liner(seq: &SequenceResult) -> String {
    let psnr = format_opt_f64(seq.metrics.psnr, 2);
    let ssim = format_opt_f64(seq.metrics.ssim, 4);
    format!(
        "**{}**: enc {:.1} fps, dec {:.1} fps, size {}, PSNR {}, SSIM {}",
        seq.sequence_name,
        seq.encoding_fps,
        seq.decoding_fps,
        BenchmarkUtils::format_bytes(seq.file_size_bytes),
        psnr,
        ssim,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::QualityMetrics;
    use crate::stats::Statistics;
    use crate::{BenchmarkConfig, CodecBenchmarkResult, CodecConfig, SequenceResult};
    use oximedia_core::types::CodecId;
    use std::time::Duration;

    fn sample_results() -> BenchmarkResults {
        let seq = SequenceResult {
            sequence_name: "test_sequence_720p".to_string(),
            frames_processed: 300,
            encoding_fps: 45.5,
            decoding_fps: 120.3,
            file_size_bytes: 1_500_000,
            metrics: QualityMetrics {
                psnr: Some(38.5),
                ssim: Some(0.965),
                vmaf: None,
                mse: None,
                psnr_y: None,
                psnr_u: None,
                psnr_v: None,
                ssim_y: None,
                ssim_u: None,
                ssim_v: None,
            },
            encoding_duration: Duration::from_secs(7),
            decoding_duration: Duration::from_secs(3),
        };
        let codec = CodecBenchmarkResult {
            codec_id: CodecId::Av1,
            preset: Some("medium".to_string()),
            bitrate_kbps: Some(2000),
            cq_level: None,
            sequence_results: vec![seq],
            statistics: Statistics {
                mean_encoding_fps: 45.5,
                mean_decoding_fps: 120.3,
                mean_psnr: Some(38.5),
                mean_ssim: Some(0.965),
                ..Statistics::default()
            },
        };
        BenchmarkResults {
            codec_results: vec![codec],
            timestamp: "2025-06-01T12:00:00Z".to_string(),
            total_duration: Duration::from_secs(42),
            config: BenchmarkConfig::default(),
        }
    }

    #[test]
    fn test_render_contains_title() {
        let results = sample_results();
        let md = MarkdownReport::new(&results).render();
        assert!(md.contains("# OxiMedia Codec Benchmark Results"));
    }

    #[test]
    fn test_render_contains_timestamp() {
        let results = sample_results();
        let md = MarkdownReport::new(&results).render();
        assert!(md.contains("2025-06-01T12:00:00Z"));
    }

    #[test]
    fn test_render_contains_codec_heading() {
        let results = sample_results();
        let md = MarkdownReport::new(&results).render();
        assert!(md.contains("### Av1"));
    }

    #[test]
    fn test_render_contains_sequence_data() {
        let results = sample_results();
        let md = MarkdownReport::new(&results).render();
        assert!(md.contains("test_sequence_720p"));
        assert!(md.contains("45.50"));
    }

    #[test]
    fn test_render_contains_statistics_table() {
        let results = sample_results();
        let md = MarkdownReport::new(&results).render();
        assert!(md.contains("## Aggregate Statistics"));
        assert!(md.contains("Mean Enc FPS"));
    }

    #[test]
    fn test_render_empty_results() {
        let results = BenchmarkResults {
            codec_results: vec![],
            timestamp: "2025-01-01T00:00:00Z".to_string(),
            total_duration: Duration::from_secs(0),
            config: BenchmarkConfig::default(),
        };
        let md = MarkdownReport::new(&results).render();
        assert!(md.contains("# OxiMedia Codec Benchmark Results"));
        // Tables should be present but empty
        assert!(md.contains("## Per-Codec Results"));
    }

    #[test]
    fn test_options_disable_sections() {
        let results = sample_results();
        let opts = MarkdownReportOptions {
            include_summary: false,
            include_detail_tables: false,
            include_statistics: false,
            include_config: false,
            float_precision: 2,
        };
        let md = MarkdownReport::with_options(&results, opts).render();
        assert!(!md.contains("# OxiMedia"));
        assert!(!md.contains("## Per-Codec Results"));
        assert!(!md.contains("## Aggregate Statistics"));
        assert!(!md.contains("## Configuration"));
    }

    #[test]
    fn test_write_to_file() {
        let results = sample_results();
        let dir = std::env::temp_dir();
        let path = dir.join("oximedia_bench_test_md_report.md");
        let report = MarkdownReport::new(&results);
        assert!(report.write_to_file(&path).is_ok());
        let content = std::fs::read_to_string(&path).expect("read back failed in test");
        assert!(content.contains("# OxiMedia Codec Benchmark Results"));
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_sequence_result_one_liner() {
        let seq = SequenceResult {
            sequence_name: "clip_1080p".to_string(),
            frames_processed: 100,
            encoding_fps: 30.0,
            decoding_fps: 200.0,
            file_size_bytes: 2_000_000,
            metrics: QualityMetrics {
                psnr: Some(40.0),
                ssim: Some(0.98),
                vmaf: None,
                mse: None,
                psnr_y: None,
                psnr_u: None,
                psnr_v: None,
                ssim_y: None,
                ssim_u: None,
                ssim_v: None,
            },
            encoding_duration: Duration::from_secs(3),
            decoding_duration: Duration::from_secs(1),
        };
        let line = sequence_result_one_liner(&seq);
        assert!(line.contains("clip_1080p"));
        assert!(line.contains("30.0"));
    }

    #[test]
    fn test_format_opt_f64_none() {
        assert_eq!(format_opt_f64(None, 2), "N/A");
    }

    #[test]
    fn test_format_opt_f64_some() {
        assert_eq!(format_opt_f64(Some(3.14159), 2), "3.14");
    }

    #[test]
    fn test_custom_float_precision() {
        let results = sample_results();
        let opts = MarkdownReportOptions {
            float_precision: 4,
            ..MarkdownReportOptions::default()
        };
        let md = MarkdownReport::with_options(&results, opts).render();
        // 45.5 with 4 decimals => 45.5000
        assert!(md.contains("45.5000"));
    }
}
