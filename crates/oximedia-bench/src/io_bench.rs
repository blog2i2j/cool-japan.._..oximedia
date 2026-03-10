#![allow(dead_code)]
//! I/O performance benchmarking for media read/write operations.
//!
//! Measures sequential and random read/write throughput, latency distribution,
//! and buffering efficiency to help tune I/O pipelines.

use std::collections::HashMap;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The access pattern to benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AccessPattern {
    /// Sequential reads/writes.
    Sequential,
    /// Random-offset reads/writes.
    Random,
    /// Mixed sequential and random.
    Mixed,
    /// Reverse sequential (seeking backwards).
    ReverseSequential,
}

/// Which direction to benchmark.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoDirection {
    /// Read only.
    Read,
    /// Write only.
    Write,
    /// Both read and write interleaved.
    ReadWrite,
}

/// Configuration for an I/O benchmark run.
#[derive(Debug, Clone)]
pub struct IoBenchConfig {
    /// Access pattern to use.
    pub pattern: AccessPattern,
    /// Direction.
    pub direction: IoDirection,
    /// Block size in bytes for each I/O operation.
    pub block_size: usize,
    /// Total bytes to transfer.
    pub total_bytes: u64,
    /// Number of warmup iterations.
    pub warmup_iterations: u32,
    /// Number of measurement iterations.
    pub measurement_iterations: u32,
    /// Buffer size hint (0 = use OS default).
    pub buffer_size: usize,
    /// Number of concurrent I/O threads (1 = single-threaded).
    pub concurrency: usize,
}

impl Default for IoBenchConfig {
    fn default() -> Self {
        Self {
            pattern: AccessPattern::Sequential,
            direction: IoDirection::Read,
            block_size: 4096,
            total_bytes: 64 * 1024 * 1024, // 64 MiB
            warmup_iterations: 1,
            measurement_iterations: 3,
            buffer_size: 0,
            concurrency: 1,
        }
    }
}

impl IoBenchConfig {
    /// Create a new default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: set access pattern.
    pub fn with_pattern(mut self, p: AccessPattern) -> Self {
        self.pattern = p;
        self
    }

    /// Builder: set direction.
    pub fn with_direction(mut self, d: IoDirection) -> Self {
        self.direction = d;
        self
    }

    /// Builder: set block size.
    pub fn with_block_size(mut self, bs: usize) -> Self {
        self.block_size = bs.max(1);
        self
    }

    /// Builder: set total bytes.
    pub fn with_total_bytes(mut self, tb: u64) -> Self {
        self.total_bytes = tb;
        self
    }

    /// Builder: set measurement iterations.
    pub fn with_measurement_iterations(mut self, n: u32) -> Self {
        self.measurement_iterations = n.max(1);
        self
    }

    /// Builder: set concurrency.
    pub fn with_concurrency(mut self, c: usize) -> Self {
        self.concurrency = c.max(1);
        self
    }

    /// Number of I/O operations per iteration.
    #[allow(clippy::cast_precision_loss)]
    pub fn ops_per_iteration(&self) -> u64 {
        if self.block_size == 0 {
            return 0;
        }
        self.total_bytes / self.block_size as u64
    }

    /// Validate the config.
    pub fn validate(&self) -> Result<(), String> {
        if self.block_size == 0 {
            return Err("block_size must be > 0".into());
        }
        if self.total_bytes == 0 {
            return Err("total_bytes must be > 0".into());
        }
        if self.measurement_iterations == 0 {
            return Err("measurement_iterations must be > 0".into());
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Latency histogram bucket.
#[derive(Debug, Clone)]
pub struct LatencyBucket {
    /// Lower bound of the bucket in microseconds.
    pub lower_us: u64,
    /// Upper bound of the bucket in microseconds.
    pub upper_us: u64,
    /// Number of operations in this bucket.
    pub count: u64,
}

/// Latency distribution.
#[derive(Debug, Clone)]
pub struct LatencyDistribution {
    /// Minimum latency in microseconds.
    pub min_us: u64,
    /// Maximum latency in microseconds.
    pub max_us: u64,
    /// Mean latency in microseconds.
    pub mean_us: f64,
    /// Median latency in microseconds.
    pub median_us: u64,
    /// 95th percentile latency in microseconds.
    pub p95_us: u64,
    /// 99th percentile latency in microseconds.
    pub p99_us: u64,
    /// Histogram buckets.
    pub buckets: Vec<LatencyBucket>,
}

impl LatencyDistribution {
    /// Build a distribution from a list of latency samples (in microseconds).
    #[allow(clippy::cast_precision_loss)]
    pub fn from_samples(mut samples: Vec<u64>) -> Self {
        if samples.is_empty() {
            return Self {
                min_us: 0,
                max_us: 0,
                mean_us: 0.0,
                median_us: 0,
                p95_us: 0,
                p99_us: 0,
                buckets: Vec::new(),
            };
        }
        samples.sort_unstable();
        let n = samples.len();
        let min_us = samples[0];
        let max_us = samples[n - 1];
        let mean_us = samples.iter().sum::<u64>() as f64 / n as f64;
        let median_us = samples[n / 2];
        let p95_us = samples[(n as f64 * 0.95) as usize];
        let p99_us = samples[(n as f64 * 0.99).min((n - 1) as f64) as usize];

        // Build simple log-scale buckets
        let buckets = Self::build_buckets(&samples);

        Self {
            min_us,
            max_us,
            mean_us,
            median_us,
            p95_us,
            p99_us,
            buckets,
        }
    }

    /// Build histogram buckets with powers-of-2 boundaries.
    fn build_buckets(sorted: &[u64]) -> Vec<LatencyBucket> {
        if sorted.is_empty() {
            return Vec::new();
        }
        let mut buckets = Vec::new();
        let mut boundary = 1_u64;
        while boundary <= sorted[sorted.len() - 1] * 2 {
            let lower = boundary;
            let upper = boundary * 2;
            let count = sorted.iter().filter(|&&v| v >= lower && v < upper).count() as u64;
            if count > 0 {
                buckets.push(LatencyBucket {
                    lower_us: lower,
                    upper_us: upper,
                    count,
                });
            }
            boundary *= 2;
            if boundary == 0 {
                break;
            } // overflow guard
        }
        buckets
    }

    /// Range of the distribution in microseconds.
    pub fn range_us(&self) -> u64 {
        self.max_us.saturating_sub(self.min_us)
    }
}

/// Result of a single I/O benchmark iteration.
#[derive(Debug, Clone)]
pub struct IoIterationResult {
    /// Bytes transferred.
    pub bytes_transferred: u64,
    /// Duration of this iteration.
    pub duration: Duration,
    /// Number of I/O operations.
    pub ops: u64,
}

impl IoIterationResult {
    /// Throughput in megabytes per second.
    #[allow(clippy::cast_precision_loss)]
    pub fn throughput_mbps(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.bytes_transferred as f64 / (1024.0 * 1024.0) / secs
    }

    /// I/O operations per second.
    #[allow(clippy::cast_precision_loss)]
    pub fn iops(&self) -> f64 {
        let secs = self.duration.as_secs_f64();
        if secs <= 0.0 {
            return 0.0;
        }
        self.ops as f64 / secs
    }
}

/// Aggregated I/O benchmark results.
#[derive(Debug, Clone)]
pub struct IoBenchResult {
    /// Configuration used.
    pub config: IoBenchConfig,
    /// Results per iteration.
    pub iterations: Vec<IoIterationResult>,
    /// Mean throughput in MB/s.
    pub mean_throughput_mbps: f64,
    /// Mean IOPS.
    pub mean_iops: f64,
    /// Latency distribution (if collected).
    pub latency: Option<LatencyDistribution>,
    /// Metadata tags.
    pub tags: HashMap<String, String>,
}

impl IoBenchResult {
    /// Build aggregated result from iterations.
    #[allow(clippy::cast_precision_loss)]
    pub fn from_iterations(config: IoBenchConfig, iterations: Vec<IoIterationResult>) -> Self {
        let n = iterations.len();
        let (mt, mi) = if n == 0 {
            (0.0, 0.0)
        } else {
            let t: f64 = iterations.iter().map(|i| i.throughput_mbps()).sum();
            let o: f64 = iterations.iter().map(|i| i.iops()).sum();
            (t / n as f64, o / n as f64)
        };
        Self {
            config,
            iterations,
            mean_throughput_mbps: mt,
            mean_iops: mi,
            latency: None,
            tags: HashMap::new(),
        }
    }

    /// Number of iterations.
    pub fn iteration_count(&self) -> usize {
        self.iterations.len()
    }

    /// Builder: attach a tag.
    pub fn with_tag(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.tags.insert(key.into(), value.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Runner (simulated)
// ---------------------------------------------------------------------------

/// Simulated I/O benchmark runner.
#[derive(Debug)]
pub struct IoBenchRunner {
    /// Config.
    config: IoBenchConfig,
}

impl IoBenchRunner {
    /// Create a new runner.
    pub fn new(config: IoBenchConfig) -> Self {
        Self { config }
    }

    /// Run the benchmark (simulated).
    #[allow(clippy::cast_precision_loss)]
    pub fn run(&self) -> IoBenchResult {
        let mut iterations = Vec::new();
        let ops = self.config.ops_per_iteration();

        for _ in 0..self.config.measurement_iterations {
            iterations.push(IoIterationResult {
                bytes_transferred: self.config.total_bytes,
                duration: Duration::from_millis(100), // simulated
                ops,
            });
        }

        IoBenchResult::from_iterations(self.config.clone(), iterations)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let cfg = IoBenchConfig::default();
        assert_eq!(cfg.pattern, AccessPattern::Sequential);
        assert_eq!(cfg.direction, IoDirection::Read);
        assert_eq!(cfg.block_size, 4096);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_builder() {
        let cfg = IoBenchConfig::new()
            .with_pattern(AccessPattern::Random)
            .with_direction(IoDirection::Write)
            .with_block_size(8192)
            .with_total_bytes(128 * 1024 * 1024)
            .with_measurement_iterations(5)
            .with_concurrency(4);
        assert_eq!(cfg.pattern, AccessPattern::Random);
        assert_eq!(cfg.direction, IoDirection::Write);
        assert_eq!(cfg.block_size, 8192);
        assert_eq!(cfg.concurrency, 4);
    }

    #[test]
    fn test_config_ops_per_iteration() {
        let cfg = IoBenchConfig::new()
            .with_block_size(1024)
            .with_total_bytes(1024 * 100);
        assert_eq!(cfg.ops_per_iteration(), 100);
    }

    #[test]
    fn test_config_validate_bad_block() {
        let mut cfg = IoBenchConfig::default();
        cfg.block_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_latency_distribution_empty() {
        let dist = LatencyDistribution::from_samples(vec![]);
        assert_eq!(dist.min_us, 0);
        assert_eq!(dist.max_us, 0);
        assert!(dist.buckets.is_empty());
    }

    #[test]
    fn test_latency_distribution_single() {
        let dist = LatencyDistribution::from_samples(vec![100]);
        assert_eq!(dist.min_us, 100);
        assert_eq!(dist.max_us, 100);
        assert_eq!(dist.median_us, 100);
    }

    #[test]
    fn test_latency_distribution_multiple() {
        let samples: Vec<u64> = (1..=100).collect();
        let dist = LatencyDistribution::from_samples(samples);
        assert_eq!(dist.min_us, 1);
        assert_eq!(dist.max_us, 100);
        assert!(dist.mean_us > 49.0 && dist.mean_us < 51.0);
        assert!(dist.p95_us >= 95);
        assert!(dist.p99_us >= 99);
    }

    #[test]
    fn test_latency_range() {
        let dist = LatencyDistribution::from_samples(vec![10, 20, 30, 40, 50]);
        assert_eq!(dist.range_us(), 40);
    }

    #[test]
    fn test_io_iteration_throughput() {
        let iter = IoIterationResult {
            bytes_transferred: 1024 * 1024, // 1 MiB
            duration: Duration::from_secs(1),
            ops: 256,
        };
        assert!((iter.throughput_mbps() - 1.0).abs() < 0.01);
        assert!((iter.iops() - 256.0).abs() < 0.01);
    }

    #[test]
    fn test_io_iteration_zero_duration() {
        let iter = IoIterationResult {
            bytes_transferred: 1024,
            duration: Duration::ZERO,
            ops: 1,
        };
        assert!((iter.throughput_mbps() - 0.0).abs() < 1e-9);
        assert!((iter.iops() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_io_bench_result_from_iterations() {
        let iters = vec![
            IoIterationResult {
                bytes_transferred: 1024 * 1024,
                duration: Duration::from_secs(1),
                ops: 256,
            },
            IoIterationResult {
                bytes_transferred: 1024 * 1024,
                duration: Duration::from_secs(1),
                ops: 256,
            },
        ];
        let result = IoBenchResult::from_iterations(IoBenchConfig::default(), iters);
        assert_eq!(result.iteration_count(), 2);
        assert!(result.mean_throughput_mbps > 0.0);
    }

    #[test]
    fn test_io_bench_runner() {
        let cfg = IoBenchConfig::new().with_measurement_iterations(3);
        let runner = IoBenchRunner::new(cfg);
        let result = runner.run();
        assert_eq!(result.iteration_count(), 3);
        assert!(result.mean_throughput_mbps > 0.0);
    }

    #[test]
    fn test_io_bench_result_tags() {
        let result = IoBenchResult::from_iterations(IoBenchConfig::default(), vec![])
            .with_tag("disk", "nvme")
            .with_tag("fs", "ext4");
        assert_eq!(result.tags.get("disk").expect("get should succeed"), "nvme");
        assert_eq!(result.tags.get("fs").expect("get should succeed"), "ext4");
    }
}
