//! Comprehensive codec benchmarking suite for `OxiMedia`.
//!
//! This crate provides a complete benchmarking framework for evaluating codec performance,
//! quality metrics, and efficiency across different encoding parameters and content types.
//!
//! # Features
//!
//! - **Multi-codec support**: Benchmark AV1, VP9, VP8, and Theora
//! - **Quality metrics**: PSNR, SSIM, and optional VMAF
//! - **Performance metrics**: Encoding/decoding speed, memory usage
//! - **Statistical analysis**: Mean, median, percentiles, standard deviation
//! - **Parallel execution**: Multi-threaded benchmark execution
//! - **Report generation**: JSON, CSV, and HTML output formats
//! - **Incremental benchmarking**: Result caching and differential runs
//!
//! # Example
//!
//! ```
//! use oximedia_bench::{BenchmarkConfig, BenchmarkSuite, CodecConfig};
//! use oximedia_core::types::CodecId;
//!
//! # fn example() -> oximedia_bench::BenchResult<()> {
//! // Create a benchmark configuration
//! let config = BenchmarkConfig::builder()
//!     .add_codec(CodecConfig::new(CodecId::Av1))
//!     .add_codec(CodecConfig::new(CodecId::Vp9))
//!     .parallel_jobs(4)
//!     .build()?;
//!
//! // Create and run the benchmark suite
//! let suite = BenchmarkSuite::new(config);
//! let results = suite.run_all()?;
//!
//! // Generate reports
//! results.export_json("results.json")?;
//! results.export_csv("results.csv")?;
//! # Ok(())
//! # }
//! ```
//!
//! # Architecture
//!
//! The benchmarking suite consists of several key components:
//!
//! - **Sequences**: Test video sequences with various characteristics
//! - **Metrics**: Quality and performance measurement tools
//! - **Runner**: Execution engine for running benchmarks
//! - **Comparison**: Tools for comparing codec performance
//! - **Reports**: Export and visualization of results
//! - **Statistics**: Statistical analysis of benchmark data

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::too_many_lines)]

pub mod baseline;
pub mod bench_suite;
pub mod codec_bench;
pub mod comparison;
pub mod cpu_profile;
pub mod examples;
pub mod gpu_bench;
pub mod hardware_info;
pub mod io_bench;
pub mod latency;
pub mod memory;
pub mod metrics;
pub mod percentile_tracker;
pub mod perf_comparison;
pub mod pipeline_bench;
pub mod regression;
pub mod regression_bench;
pub mod regression_detect;
pub mod report;
pub mod resource_monitor;
pub mod runner;
pub mod scalability_bench;
pub mod sequences;
pub mod statistical;
pub mod stats;
pub mod throughput;
pub mod warmup_strategy;

use oximedia_core::types::CodecId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use thiserror::Error;

pub use comparison::{CodecComparison, ComparisonResult};
pub use metrics::{MetricsCalculator, QualityMetrics};
pub use report::{HtmlReport, ReportExporter};
pub use runner::{BenchmarkRunner, ExecutionResult};
pub use sequences::{ContentType, MotionCharacteristics, TestSequence};
pub use stats::{StatisticalAnalysis, Statistics};

/// Result type for benchmarking operations.
pub type BenchResult<T> = Result<T, BenchError>;

/// Errors that can occur during benchmarking.
#[derive(Debug, Error)]
pub enum BenchError {
    /// I/O error occurred
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// OxiMedia core error
    #[error("OxiMedia error: {0}")]
    Oxi(#[from] oximedia_core::error::OxiError),

    /// Codec error
    #[error("Codec error: {0}")]
    Codec(#[from] oximedia_codec::error::CodecError),

    /// Graph error
    #[error("Graph error: {0}")]
    Graph(#[from] oximedia_graph::error::GraphError),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// CSV error
    #[error("CSV error: {0}")]
    Csv(#[from] csv::Error),

    /// Invalid configuration
    #[error("Invalid configuration: {0}")]
    InvalidConfig(String),

    /// Sequence not found
    #[error("Test sequence not found: {0}")]
    SequenceNotFound(String),

    /// Codec not supported
    #[error("Codec not supported: {0:?}")]
    UnsupportedCodec(CodecId),

    /// Benchmark execution failed
    #[error("Benchmark execution failed: {0}")]
    ExecutionFailed(String),

    /// Metric calculation failed
    #[error("Metric calculation failed: {0}")]
    MetricFailed(String),

    /// Invalid benchmark results
    #[error("Invalid benchmark results: {0}")]
    InvalidResults(String),
}

/// Configuration for a codec benchmark.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecConfig {
    /// Codec identifier
    pub codec_id: CodecId,

    /// Encoding preset (if applicable)
    pub preset: Option<String>,

    /// Target bitrate in kbps (if applicable)
    pub bitrate_kbps: Option<u32>,

    /// Constant quality mode value (if applicable)
    pub cq_level: Option<u32>,

    /// Number of encoding passes
    pub passes: u32,

    /// Enable rate control
    pub rate_control: bool,

    /// Additional codec-specific parameters
    pub extra_params: HashMap<String, String>,
}

impl CodecConfig {
    /// Create a new codec configuration with default settings.
    #[must_use]
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            preset: None,
            bitrate_kbps: None,
            cq_level: None,
            passes: 1,
            rate_control: false,
            extra_params: HashMap::new(),
        }
    }

    /// Set the encoding preset.
    #[must_use]
    pub fn with_preset(mut self, preset: impl Into<String>) -> Self {
        self.preset = Some(preset.into());
        self
    }

    /// Set the target bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bitrate_kbps: u32) -> Self {
        self.bitrate_kbps = Some(bitrate_kbps);
        self
    }

    /// Set the constant quality level.
    #[must_use]
    pub fn with_cq_level(mut self, cq_level: u32) -> Self {
        self.cq_level = Some(cq_level);
        self
    }

    /// Set the number of encoding passes.
    #[must_use]
    pub fn with_passes(mut self, passes: u32) -> Self {
        self.passes = passes;
        self
    }

    /// Enable or disable rate control.
    #[must_use]
    pub fn with_rate_control(mut self, enabled: bool) -> Self {
        self.rate_control = enabled;
        self
    }

    /// Add a codec-specific parameter.
    #[must_use]
    pub fn with_param(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_params.insert(key.into(), value.into());
        self
    }
}

/// Configuration for the benchmark suite.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkConfig {
    /// Codec configurations to benchmark
    pub codecs: Vec<CodecConfig>,

    /// Test sequences to use
    pub sequences: Vec<PathBuf>,

    /// Number of parallel jobs
    pub parallel_jobs: usize,

    /// Enable quality metric calculation
    pub enable_psnr: bool,

    /// Enable SSIM calculation
    pub enable_ssim: bool,

    /// Enable VMAF calculation
    pub enable_vmaf: bool,

    /// Cache directory for intermediate results
    pub cache_dir: Option<PathBuf>,

    /// Output directory for results
    pub output_dir: PathBuf,

    /// Maximum number of frames to process per sequence
    pub max_frames: Option<usize>,

    /// Warmup iterations before measurement
    pub warmup_iterations: usize,

    /// Number of measurement iterations
    pub measurement_iterations: usize,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            codecs: Vec::new(),
            sequences: Vec::new(),
            parallel_jobs: num_cpus(),
            enable_psnr: true,
            enable_ssim: true,
            enable_vmaf: false,
            cache_dir: None,
            output_dir: PathBuf::from("./bench_results"),
            max_frames: None,
            warmup_iterations: 1,
            measurement_iterations: 3,
        }
    }
}

impl BenchmarkConfig {
    /// Create a new builder for benchmark configuration.
    #[must_use]
    pub fn builder() -> BenchmarkConfigBuilder {
        BenchmarkConfigBuilder::default()
    }
}

/// Builder for creating benchmark configurations.
#[derive(Debug, Default)]
pub struct BenchmarkConfigBuilder {
    config: BenchmarkConfig,
}

impl BenchmarkConfigBuilder {
    /// Add a codec configuration.
    #[must_use]
    pub fn add_codec(mut self, codec: CodecConfig) -> Self {
        self.config.codecs.push(codec);
        self
    }

    /// Add a test sequence.
    #[must_use]
    pub fn add_sequence(mut self, path: impl Into<PathBuf>) -> Self {
        self.config.sequences.push(path.into());
        self
    }

    /// Set the number of parallel jobs.
    #[must_use]
    pub fn parallel_jobs(mut self, jobs: usize) -> Self {
        self.config.parallel_jobs = jobs;
        self
    }

    /// Enable PSNR calculation.
    #[must_use]
    pub fn enable_psnr(mut self, enable: bool) -> Self {
        self.config.enable_psnr = enable;
        self
    }

    /// Enable SSIM calculation.
    #[must_use]
    pub fn enable_ssim(mut self, enable: bool) -> Self {
        self.config.enable_ssim = enable;
        self
    }

    /// Enable VMAF calculation.
    #[must_use]
    pub fn enable_vmaf(mut self, enable: bool) -> Self {
        self.config.enable_vmaf = enable;
        self
    }

    /// Set the cache directory.
    #[must_use]
    pub fn cache_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.cache_dir = Some(dir.into());
        self
    }

    /// Set the output directory.
    #[must_use]
    pub fn output_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.config.output_dir = dir.into();
        self
    }

    /// Set the maximum number of frames to process.
    #[must_use]
    pub fn max_frames(mut self, max: usize) -> Self {
        self.config.max_frames = Some(max);
        self
    }

    /// Set the number of warmup iterations.
    #[must_use]
    pub fn warmup_iterations(mut self, iterations: usize) -> Self {
        self.config.warmup_iterations = iterations;
        self
    }

    /// Set the number of measurement iterations.
    #[must_use]
    pub fn measurement_iterations(mut self, iterations: usize) -> Self {
        self.config.measurement_iterations = iterations;
        self
    }

    /// Build the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn build(self) -> BenchResult<BenchmarkConfig> {
        if self.config.codecs.is_empty() {
            return Err(BenchError::InvalidConfig("No codecs specified".to_string()));
        }

        if self.config.parallel_jobs == 0 {
            return Err(BenchError::InvalidConfig(
                "Parallel jobs must be greater than 0".to_string(),
            ));
        }

        if self.config.measurement_iterations == 0 {
            return Err(BenchError::InvalidConfig(
                "Measurement iterations must be greater than 0".to_string(),
            ));
        }

        Ok(self.config)
    }
}

/// Complete benchmark results for all codecs and sequences.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// Individual codec results
    pub codec_results: Vec<CodecBenchmarkResult>,

    /// Timestamp when benchmark was run
    pub timestamp: String,

    /// Total execution time
    #[serde(with = "duration_serde")]
    pub total_duration: Duration,

    /// Configuration used
    pub config: BenchmarkConfig,
}

impl BenchmarkResults {
    /// Export results to JSON format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn export_json(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        let file = std::fs::File::create(path)?;
        serde_json::to_writer_pretty(file, self)?;
        Ok(())
    }

    /// Export results to CSV format.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn export_csv(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        let mut writer = csv::Writer::from_path(path)?;

        // Write header
        writer.write_record([
            "Codec",
            "Sequence",
            "Preset",
            "Bitrate (kbps)",
            "Encoding FPS",
            "Decoding FPS",
            "File Size (bytes)",
            "PSNR (dB)",
            "SSIM",
            "VMAF",
        ])?;

        // Write data rows
        for codec_result in &self.codec_results {
            for seq_result in &codec_result.sequence_results {
                writer.write_record(&[
                    format!("{:?}", codec_result.codec_id),
                    seq_result.sequence_name.clone(),
                    codec_result.preset.clone().unwrap_or_default(),
                    codec_result
                        .bitrate_kbps
                        .map_or(String::new(), |b| b.to_string()),
                    format!("{:.2}", seq_result.encoding_fps),
                    format!("{:.2}", seq_result.decoding_fps),
                    seq_result.file_size_bytes.to_string(),
                    seq_result
                        .metrics
                        .psnr
                        .map_or(String::new(), |p| format!("{p:.2}")),
                    seq_result
                        .metrics
                        .ssim
                        .map_or(String::new(), |s| format!("{s:.4}")),
                    seq_result
                        .metrics
                        .vmaf
                        .map_or(String::new(), |v| format!("{v:.2}")),
                ])?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    /// Generate an HTML report.
    ///
    /// # Errors
    ///
    /// Returns an error if the report cannot be generated.
    pub fn export_html(&self, path: impl AsRef<Path>) -> BenchResult<()> {
        let report = HtmlReport::new(self);
        report.write_to_file(path)
    }

    /// Get all results for a specific codec.
    #[must_use]
    pub fn get_codec_results(&self, codec_id: CodecId) -> Vec<&CodecBenchmarkResult> {
        self.codec_results
            .iter()
            .filter(|r| r.codec_id == codec_id)
            .collect()
    }

    /// Compare two codecs.
    #[must_use]
    pub fn compare_codecs(&self, codec_a: CodecId, codec_b: CodecId) -> Option<ComparisonResult> {
        let results_a: Vec<_> = self.get_codec_results(codec_a);
        let results_b: Vec<_> = self.get_codec_results(codec_b);

        if results_a.is_empty() || results_b.is_empty() {
            return None;
        }

        Some(CodecComparison::compare(results_a, results_b))
    }
}

/// Benchmark results for a single codec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecBenchmarkResult {
    /// Codec identifier
    pub codec_id: CodecId,

    /// Preset used
    pub preset: Option<String>,

    /// Target bitrate
    pub bitrate_kbps: Option<u32>,

    /// Constant quality level
    pub cq_level: Option<u32>,

    /// Results for each sequence
    pub sequence_results: Vec<SequenceResult>,

    /// Aggregated statistics
    pub statistics: Statistics,
}

/// Results for a single test sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequenceResult {
    /// Sequence name/identifier
    pub sequence_name: String,

    /// Number of frames processed
    pub frames_processed: usize,

    /// Encoding frames per second
    pub encoding_fps: f64,

    /// Decoding frames per second
    pub decoding_fps: f64,

    /// Encoded file size in bytes
    pub file_size_bytes: u64,

    /// Quality metrics
    pub metrics: QualityMetrics,

    /// Encoding duration
    #[serde(with = "duration_serde")]
    pub encoding_duration: Duration,

    /// Decoding duration
    #[serde(with = "duration_serde")]
    pub decoding_duration: Duration,
}

/// The main benchmark suite.
pub struct BenchmarkSuite {
    config: BenchmarkConfig,
    runner: BenchmarkRunner,
}

impl BenchmarkSuite {
    /// Create a new benchmark suite with the given configuration.
    #[must_use]
    pub fn new(config: BenchmarkConfig) -> Self {
        let runner = BenchmarkRunner::new(&config);
        Self { config, runner }
    }

    /// Run all benchmarks.
    ///
    /// # Errors
    ///
    /// Returns an error if any benchmark fails.
    pub fn run_all(&self) -> BenchResult<BenchmarkResults> {
        let start_time = std::time::Instant::now();
        let mut codec_results = Vec::new();

        // Run benchmarks for each codec configuration
        for codec_config in &self.config.codecs {
            let result = self.run_codec_benchmark(codec_config)?;
            codec_results.push(result);
        }

        let total_duration = start_time.elapsed();

        Ok(BenchmarkResults {
            codec_results,
            timestamp: format_timestamp(),
            total_duration,
            config: self.config.clone(),
        })
    }

    /// Run benchmark for a single codec.
    fn run_codec_benchmark(&self, codec_config: &CodecConfig) -> BenchResult<CodecBenchmarkResult> {
        let sequence_results = self.runner.run_codec_sequences(codec_config)?;
        let statistics = stats::compute_statistics(&sequence_results);

        Ok(CodecBenchmarkResult {
            codec_id: codec_config.codec_id,
            preset: codec_config.preset.clone(),
            bitrate_kbps: codec_config.bitrate_kbps,
            cq_level: codec_config.cq_level,
            sequence_results,
            statistics,
        })
    }
}

/// Get the number of CPU cores available.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(1)
}

/// Format a timestamp in RFC3339 format.
fn format_timestamp() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};

    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();

    let secs = duration.as_secs();
    let nanos = duration.subsec_nanos();

    // Simple ISO 8601 / RFC3339-like format
    let days = secs / 86400;
    let hours = (secs % 86400) / 3600;
    let minutes = (secs % 3600) / 60;
    let seconds = secs % 60;

    // Approximate year/month/day calculation (good enough for benchmarking)
    let years = 1970 + (days / 365);
    let day_of_year = days % 365;
    let month = (day_of_year / 30).min(11) + 1;
    let day = (day_of_year % 30) + 1;

    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}Z",
        years, month, day, hours, minutes, seconds, nanos
    )
}

/// Serde serialization/deserialization for Duration.
mod duration_serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::time::Duration;

    pub fn serialize<S>(duration: &Duration, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        duration.as_secs_f64().serialize(serializer)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<Duration, D::Error>
    where
        D: Deserializer<'de>,
    {
        let secs = f64::deserialize(deserializer)?;
        Ok(Duration::from_secs_f64(secs))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_codec_config_builder() {
        let config = CodecConfig::new(CodecId::Av1)
            .with_preset("medium")
            .with_bitrate(2000)
            .with_passes(2);

        assert_eq!(config.codec_id, CodecId::Av1);
        assert_eq!(config.preset, Some("medium".to_string()));
        assert_eq!(config.bitrate_kbps, Some(2000));
        assert_eq!(config.passes, 2);
    }

    #[test]
    fn test_benchmark_config_builder() {
        let config = BenchmarkConfig::builder()
            .add_codec(CodecConfig::new(CodecId::Av1))
            .add_codec(CodecConfig::new(CodecId::Vp9))
            .parallel_jobs(4)
            .enable_psnr(true)
            .enable_ssim(true)
            .build();

        assert!(config.is_ok());
        let config = config.expect("config should be valid");
        assert_eq!(config.codecs.len(), 2);
        assert_eq!(config.parallel_jobs, 4);
        assert!(config.enable_psnr);
        assert!(config.enable_ssim);
    }

    #[test]
    fn test_invalid_config_no_codecs() {
        let config = BenchmarkConfig::builder().build();
        assert!(config.is_err());
        assert!(matches!(config.unwrap_err(), BenchError::InvalidConfig(_)));
    }

    #[test]
    fn test_invalid_config_zero_jobs() {
        let config = BenchmarkConfig::builder()
            .add_codec(CodecConfig::new(CodecId::Av1))
            .parallel_jobs(0)
            .build();

        assert!(config.is_err());
    }

    #[test]
    fn test_codec_config_with_params() {
        let config = CodecConfig::new(CodecId::Vp9)
            .with_param("cpu-used", "4")
            .with_param("threads", "8");

        assert_eq!(config.extra_params.get("cpu-used"), Some(&"4".to_string()));
        assert_eq!(config.extra_params.get("threads"), Some(&"8".to_string()));
    }

    #[test]
    fn test_num_cpus() {
        let cpus = num_cpus();
        assert!(cpus > 0);
    }

    #[test]
    fn test_format_timestamp() {
        let ts = format_timestamp();
        assert!(!ts.is_empty());
        assert!(ts.contains('T'));
        assert!(ts.contains('Z'));
    }
}

/// Benchmark preset configurations for common scenarios.
pub struct BenchmarkPresets;

impl BenchmarkPresets {
    /// Create a quick benchmark preset (fast, fewer iterations).
    #[must_use]
    pub fn quick() -> BenchmarkConfig {
        BenchmarkConfig {
            codecs: vec![
                CodecConfig::new(CodecId::Av1).with_preset("fast"),
                CodecConfig::new(CodecId::Vp9).with_preset("fast"),
            ],
            sequences: Vec::new(),
            parallel_jobs: num_cpus(),
            enable_psnr: true,
            enable_ssim: false,
            enable_vmaf: false,
            cache_dir: None,
            output_dir: PathBuf::from("./bench_quick"),
            max_frames: Some(30),
            warmup_iterations: 0,
            measurement_iterations: 1,
        }
    }

    /// Create a standard benchmark preset (balanced settings).
    #[must_use]
    pub fn standard() -> BenchmarkConfig {
        BenchmarkConfig {
            codecs: vec![
                CodecConfig::new(CodecId::Av1),
                CodecConfig::new(CodecId::Vp9),
            ],
            sequences: Vec::new(),
            parallel_jobs: num_cpus(),
            enable_psnr: true,
            enable_ssim: true,
            enable_vmaf: false,
            cache_dir: Some(PathBuf::from("./bench_cache")),
            output_dir: PathBuf::from("./bench_results"),
            max_frames: Some(300),
            warmup_iterations: 1,
            measurement_iterations: 3,
        }
    }

    /// Create a comprehensive benchmark preset (all metrics, high quality).
    #[must_use]
    pub fn comprehensive() -> BenchmarkConfig {
        BenchmarkConfig {
            codecs: vec![
                CodecConfig::new(CodecId::Av1).with_preset("medium"),
                CodecConfig::new(CodecId::Av1).with_preset("slow"),
                CodecConfig::new(CodecId::Vp9).with_preset("good"),
                CodecConfig::new(CodecId::Vp9).with_preset("best"),
            ],
            sequences: Vec::new(),
            parallel_jobs: num_cpus(),
            enable_psnr: true,
            enable_ssim: true,
            enable_vmaf: true,
            cache_dir: Some(PathBuf::from("./bench_cache")),
            output_dir: PathBuf::from("./bench_comprehensive"),
            max_frames: None,
            warmup_iterations: 2,
            measurement_iterations: 5,
        }
    }

    /// Create a quality-focused benchmark preset.
    #[must_use]
    pub fn quality_focus() -> BenchmarkConfig {
        BenchmarkConfig {
            codecs: vec![
                CodecConfig::new(CodecId::Av1).with_cq_level(20),
                CodecConfig::new(CodecId::Av1).with_cq_level(30),
                CodecConfig::new(CodecId::Av1).with_cq_level(40),
            ],
            sequences: Vec::new(),
            parallel_jobs: num_cpus() / 2,
            enable_psnr: true,
            enable_ssim: true,
            enable_vmaf: true,
            cache_dir: Some(PathBuf::from("./bench_cache")),
            output_dir: PathBuf::from("./bench_quality"),
            max_frames: None,
            warmup_iterations: 1,
            measurement_iterations: 3,
        }
    }

    /// Create a speed-focused benchmark preset.
    #[must_use]
    pub fn speed_focus() -> BenchmarkConfig {
        BenchmarkConfig {
            codecs: vec![
                CodecConfig::new(CodecId::Av1).with_preset("ultrafast"),
                CodecConfig::new(CodecId::Av1).with_preset("fast"),
                CodecConfig::new(CodecId::Vp9).with_preset("realtime"),
            ],
            sequences: Vec::new(),
            parallel_jobs: num_cpus(),
            enable_psnr: true,
            enable_ssim: false,
            enable_vmaf: false,
            cache_dir: Some(PathBuf::from("./bench_cache")),
            output_dir: PathBuf::from("./bench_speed"),
            max_frames: Some(100),
            warmup_iterations: 2,
            measurement_iterations: 5,
        }
    }
}

/// Benchmark utilities for common operations.
pub struct BenchmarkUtils;

impl BenchmarkUtils {
    /// Calculate bitrate from file size and duration.
    #[must_use]
    pub fn calculate_bitrate(file_size_bytes: u64, duration_seconds: f64) -> f64 {
        if duration_seconds == 0.0 {
            return 0.0;
        }
        (file_size_bytes as f64 * 8.0) / duration_seconds / 1000.0
    }

    /// Calculate bits per pixel.
    #[must_use]
    pub fn calculate_bpp(
        file_size_bytes: u64,
        width: usize,
        height: usize,
        frame_count: usize,
    ) -> f64 {
        let total_pixels = (width * height * frame_count) as f64;
        if total_pixels == 0.0 {
            return 0.0;
        }
        (file_size_bytes as f64 * 8.0) / total_pixels
    }

    /// Calculate compression ratio.
    #[must_use]
    pub fn calculate_compression_ratio(
        original_size_bytes: u64,
        compressed_size_bytes: u64,
    ) -> f64 {
        if compressed_size_bytes == 0 {
            return 0.0;
        }
        original_size_bytes as f64 / compressed_size_bytes as f64
    }

    /// Format bytes as human-readable string.
    #[must_use]
    pub fn format_bytes(bytes: u64) -> String {
        const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB"];
        let mut size = bytes as f64;
        let mut unit_index = 0;

        while size >= 1024.0 && unit_index < UNITS.len() - 1 {
            size /= 1024.0;
            unit_index += 1;
        }

        format!("{:.2} {}", size, UNITS[unit_index])
    }

    /// Format duration as human-readable string.
    #[must_use]
    pub fn format_duration(duration: Duration) -> String {
        let secs = duration.as_secs();
        let hours = secs / 3600;
        let minutes = (secs % 3600) / 60;
        let seconds = secs % 60;

        if hours > 0 {
            format!("{hours}h {minutes}m {seconds}s")
        } else if minutes > 0 {
            format!("{minutes}m {seconds}s")
        } else {
            format!("{seconds}s")
        }
    }

    /// Parse bitrate string (e.g., "2000kbps", "5Mbps").
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn parse_bitrate(bitrate_str: &str) -> BenchResult<u32> {
        let bitrate_str = bitrate_str.to_lowercase();

        if let Some(stripped) = bitrate_str.strip_suffix("kbps") {
            stripped
                .trim()
                .parse()
                .map_err(|_| BenchError::InvalidConfig(format!("Invalid bitrate: {bitrate_str}")))
        } else if let Some(stripped) = bitrate_str.strip_suffix("mbps") {
            stripped
                .trim()
                .parse::<f64>()
                .map(|v| (v * 1000.0) as u32)
                .map_err(|_| BenchError::InvalidConfig(format!("Invalid bitrate: {bitrate_str}")))
        } else {
            bitrate_str
                .parse()
                .map_err(|_| BenchError::InvalidConfig(format!("Invalid bitrate: {bitrate_str}")))
        }
    }

    /// Generate a benchmark summary.
    #[must_use]
    pub fn generate_summary(results: &BenchmarkResults) -> String {
        let mut summary = String::new();
        summary.push_str("# Benchmark Summary\n\n");

        for codec_result in &results.codec_results {
            summary.push_str(&format!("## {:?}\n", codec_result.codec_id));

            if let Some(preset) = &codec_result.preset {
                summary.push_str(&format!("Preset: {preset}\n"));
            }

            summary.push_str(&format!(
                "Mean Encoding FPS: {:.2}\n",
                codec_result.statistics.mean_encoding_fps
            ));

            summary.push_str(&format!(
                "Mean Decoding FPS: {:.2}\n",
                codec_result.statistics.mean_decoding_fps
            ));

            if let Some(psnr) = codec_result.statistics.mean_psnr {
                summary.push_str(&format!("Mean PSNR: {psnr:.2} dB\n"));
            }

            if let Some(ssim) = codec_result.statistics.mean_ssim {
                summary.push_str(&format!("Mean SSIM: {ssim:.4}\n"));
            }

            summary.push('\n');
        }

        summary
    }
}

/// Benchmark filter for filtering results based on criteria.
#[derive(Debug, Clone, Default)]
pub struct BenchmarkFilter {
    min_encoding_fps: Option<f64>,
    max_encoding_fps: Option<f64>,
    min_psnr: Option<f64>,
    max_psnr: Option<f64>,
    min_ssim: Option<f64>,
    max_ssim: Option<f64>,
    codec_ids: Vec<CodecId>,
}

impl BenchmarkFilter {
    /// Create a new filter.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set minimum encoding FPS.
    #[must_use]
    pub fn with_min_encoding_fps(mut self, fps: f64) -> Self {
        self.min_encoding_fps = Some(fps);
        self
    }

    /// Set maximum encoding FPS.
    #[must_use]
    pub fn with_max_encoding_fps(mut self, fps: f64) -> Self {
        self.max_encoding_fps = Some(fps);
        self
    }

    /// Set minimum PSNR.
    #[must_use]
    pub fn with_min_psnr(mut self, psnr: f64) -> Self {
        self.min_psnr = Some(psnr);
        self
    }

    /// Set maximum PSNR.
    #[must_use]
    pub fn with_max_psnr(mut self, psnr: f64) -> Self {
        self.max_psnr = Some(psnr);
        self
    }

    /// Set minimum SSIM.
    #[must_use]
    pub fn with_min_ssim(mut self, ssim: f64) -> Self {
        self.min_ssim = Some(ssim);
        self
    }

    /// Set maximum SSIM.
    #[must_use]
    pub fn with_max_ssim(mut self, ssim: f64) -> Self {
        self.max_ssim = Some(ssim);
        self
    }

    /// Set codec IDs to include.
    #[must_use]
    pub fn with_codec_ids(mut self, codec_ids: Vec<CodecId>) -> Self {
        self.codec_ids = codec_ids;
        self
    }

    /// Apply filter to results.
    #[must_use]
    pub fn apply<'a>(&self, results: &'a BenchmarkResults) -> Vec<&'a CodecBenchmarkResult> {
        results
            .codec_results
            .iter()
            .filter(|r| self.matches_codec(r))
            .collect()
    }

    fn matches_codec(&self, result: &CodecBenchmarkResult) -> bool {
        // Check codec ID
        if !self.codec_ids.is_empty() && !self.codec_ids.contains(&result.codec_id) {
            return false;
        }

        // Check encoding FPS
        if let Some(min) = self.min_encoding_fps {
            if result.statistics.mean_encoding_fps < min {
                return false;
            }
        }

        if let Some(max) = self.max_encoding_fps {
            if result.statistics.mean_encoding_fps > max {
                return false;
            }
        }

        // Check PSNR
        if let Some(min) = self.min_psnr {
            if result.statistics.mean_psnr.map_or(true, |psnr| psnr < min) {
                return false;
            }
        }

        if let Some(max) = self.max_psnr {
            if result.statistics.mean_psnr.map_or(true, |psnr| psnr > max) {
                return false;
            }
        }

        // Check SSIM
        if let Some(min) = self.min_ssim {
            if result.statistics.mean_ssim.map_or(true, |ssim| ssim < min) {
                return false;
            }
        }

        if let Some(max) = self.max_ssim {
            if result.statistics.mean_ssim.map_or(true, |ssim| ssim > max) {
                return false;
            }
        }

        true
    }
}

/// Command-line interface helpers for benchmark tool.
pub struct CliHelpers;

impl CliHelpers {
    /// Parse codec from string.
    ///
    /// # Errors
    ///
    /// Returns an error if codec string is invalid.
    pub fn parse_codec(codec_str: &str) -> BenchResult<CodecId> {
        match codec_str.to_lowercase().as_str() {
            "av1" => Ok(CodecId::Av1),
            "vp9" => Ok(CodecId::Vp9),
            "vp8" => Ok(CodecId::Vp8),
            "theora" => Ok(CodecId::Theora),
            _ => Err(BenchError::InvalidConfig(format!(
                "Unknown codec: {codec_str}"
            ))),
        }
    }

    /// Generate example configuration file.
    #[must_use]
    pub fn generate_example_config() -> String {
        serde_json::to_string_pretty(&BenchmarkConfig::default())
            .unwrap_or_else(|_| String::from("{}"))
    }

    /// Print progress bar.
    pub fn print_progress(current: usize, total: usize, bar_width: usize) {
        let progress = if total > 0 {
            current as f64 / total as f64
        } else {
            0.0
        };

        let filled = (bar_width as f64 * progress) as usize;
        let empty = bar_width - filled;

        print!("\r[");
        for _ in 0..filled {
            print!("=");
        }
        for _ in 0..empty {
            print!(" ");
        }
        print!("] {:.1}% ({}/{})", progress * 100.0, current, total);

        use std::io::Write;
        std::io::stdout().flush().ok();
    }

    /// Clear progress bar.
    pub fn clear_progress() {
        print!("\r");
        for _ in 0..100 {
            print!(" ");
        }
        print!("\r");

        use std::io::Write;
        std::io::stdout().flush().ok();
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    #[test]
    fn test_benchmark_presets_quick() {
        let config = BenchmarkPresets::quick();
        assert_eq!(config.codecs.len(), 2);
        assert_eq!(config.max_frames, Some(30));
        assert_eq!(config.warmup_iterations, 0);
    }

    #[test]
    fn test_benchmark_presets_standard() {
        let config = BenchmarkPresets::standard();
        assert_eq!(config.codecs.len(), 2);
        assert_eq!(config.measurement_iterations, 3);
        assert!(config.enable_psnr);
        assert!(config.enable_ssim);
    }

    #[test]
    fn test_benchmark_presets_comprehensive() {
        let config = BenchmarkPresets::comprehensive();
        assert_eq!(config.codecs.len(), 4);
        assert!(config.enable_vmaf);
        assert_eq!(config.measurement_iterations, 5);
    }

    #[test]
    fn test_calculate_bitrate() {
        let bitrate = BenchmarkUtils::calculate_bitrate(1_000_000, 10.0);
        assert_eq!(bitrate, 800.0); // 1MB over 10s = 800 kbps
    }

    #[test]
    fn test_calculate_bpp() {
        let bpp = BenchmarkUtils::calculate_bpp(1_000_000, 1920, 1080, 100);
        assert!(bpp > 0.0);
    }

    #[test]
    fn test_calculate_compression_ratio() {
        let ratio = BenchmarkUtils::calculate_compression_ratio(10_000_000, 1_000_000);
        assert_eq!(ratio, 10.0);
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(BenchmarkUtils::format_bytes(1024), "1.00 KB");
        assert_eq!(BenchmarkUtils::format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(BenchmarkUtils::format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_duration() {
        assert_eq!(
            BenchmarkUtils::format_duration(Duration::from_secs(30)),
            "30s"
        );
        assert_eq!(
            BenchmarkUtils::format_duration(Duration::from_secs(90)),
            "1m 30s"
        );
        assert_eq!(
            BenchmarkUtils::format_duration(Duration::from_secs(3665)),
            "1h 1m 5s"
        );
    }

    #[test]
    fn test_parse_bitrate() {
        assert_eq!(
            BenchmarkUtils::parse_bitrate("2000kbps").expect("test expectation failed"),
            2000
        );
        assert_eq!(
            BenchmarkUtils::parse_bitrate("5Mbps").expect("test expectation failed"),
            5000
        );
        assert_eq!(
            BenchmarkUtils::parse_bitrate("1500").expect("test expectation failed"),
            1500
        );
    }

    #[test]
    fn test_parse_codec() {
        assert!(matches!(
            CliHelpers::parse_codec("av1").expect("test expectation failed"),
            CodecId::Av1
        ));
        assert!(matches!(
            CliHelpers::parse_codec("vp9").expect("test expectation failed"),
            CodecId::Vp9
        ));
        assert!(matches!(
            CliHelpers::parse_codec("vp8").expect("test expectation failed"),
            CodecId::Vp8
        ));
    }

    #[test]
    fn test_benchmark_filter() {
        let filter = BenchmarkFilter::new()
            .with_min_encoding_fps(30.0)
            .with_min_psnr(35.0);

        assert_eq!(filter.min_encoding_fps, Some(30.0));
        assert_eq!(filter.min_psnr, Some(35.0));
    }

    #[test]
    fn test_generate_summary() {
        let results = BenchmarkResults {
            codec_results: vec![],
            timestamp: "2024-01-01T00:00:00Z".to_string(),
            total_duration: Duration::from_secs(100),
            config: BenchmarkConfig::default(),
        };

        let summary = BenchmarkUtils::generate_summary(&results);
        assert!(summary.contains("# Benchmark Summary"));
    }

    #[test]
    fn test_generate_example_config() {
        let config = CliHelpers::generate_example_config();
        assert!(!config.is_empty());
    }
}
