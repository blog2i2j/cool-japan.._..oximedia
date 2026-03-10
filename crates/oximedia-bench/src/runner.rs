//! Benchmark execution engine for running codec benchmarks.
//!
//! This module provides the core execution engine that runs codec benchmarks,
//! manages parallel execution, handles warmup iterations, and collects performance
//! metrics.

use crate::metrics::{MetricsCalculator, QualityMetrics};
use crate::sequences::TestSequence;
use crate::{BenchError, BenchResult, BenchmarkConfig, CodecConfig, SequenceResult};
use oximedia_codec::VideoFrame;
use oximedia_core::types::CodecId;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Benchmark execution result for a single run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// Frames processed
    pub frames: usize,
    /// Time taken
    #[serde(with = "crate::duration_serde")]
    pub duration: Duration,
    /// FPS achieved
    pub fps: f64,
    /// Peak memory usage in bytes
    pub peak_memory_bytes: Option<u64>,
    /// CPU utilization percentage
    pub cpu_utilization: Option<f64>,
}

impl ExecutionResult {
    /// Create a new execution result.
    #[must_use]
    pub fn new(frames: usize, duration: Duration) -> Self {
        let fps = if duration.as_secs_f64() > 0.0 {
            frames as f64 / duration.as_secs_f64()
        } else {
            0.0
        };
        Self {
            frames,
            duration,
            fps,
            peak_memory_bytes: None,
            cpu_utilization: None,
        }
    }

    /// Create with memory tracking.
    #[must_use]
    pub fn with_memory(mut self, peak_memory_bytes: u64) -> Self {
        self.peak_memory_bytes = Some(peak_memory_bytes);
        self
    }

    /// Create with CPU tracking.
    #[must_use]
    pub fn with_cpu(mut self, cpu_utilization: f64) -> Self {
        self.cpu_utilization = Some(cpu_utilization);
        self
    }
}

/// Cache for benchmark results to enable incremental runs.
#[derive(Debug, Clone, Default)]
pub struct ResultCache {
    cache: Arc<Mutex<HashMap<String, SequenceResult>>>,
    cache_dir: Option<PathBuf>,
}

impl ResultCache {
    /// Create a new result cache.
    #[must_use]
    pub fn new(cache_dir: Option<PathBuf>) -> Self {
        Self {
            cache: Arc::new(Mutex::new(HashMap::new())),
            cache_dir,
        }
    }

    /// Get a cached result.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<SequenceResult> {
        let cache = self.cache.lock().ok()?;
        cache.get(key).cloned()
    }

    /// Store a result in the cache.
    pub fn set(&self, key: String, result: SequenceResult) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.insert(key, result);
        }
    }

    /// Load cache from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails.
    pub fn load_from_disk(&self) -> BenchResult<()> {
        if let Some(cache_dir) = &self.cache_dir {
            let cache_file = cache_dir.join("bench_cache.json");
            if cache_file.exists() {
                let data = std::fs::read_to_string(&cache_file)?;
                let cached: HashMap<String, SequenceResult> = serde_json::from_str(&data)?;
                if let Ok(mut cache) = self.cache.lock() {
                    *cache = cached;
                }
            }
        }
        Ok(())
    }

    /// Save cache to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if saving fails.
    pub fn save_to_disk(&self) -> BenchResult<()> {
        if let Some(cache_dir) = &self.cache_dir {
            std::fs::create_dir_all(cache_dir)?;
            let cache_file = cache_dir.join("bench_cache.json");
            if let Ok(cache) = self.cache.lock() {
                let data = serde_json::to_string_pretty(&*cache)?;
                std::fs::write(&cache_file, data)?;
            }
        }
        Ok(())
    }

    /// Clear the cache.
    pub fn clear(&self) {
        if let Ok(mut cache) = self.cache.lock() {
            cache.clear();
        }
    }

    /// Generate cache key for a benchmark run.
    #[must_use]
    pub fn generate_key(codec_config: &CodecConfig, sequence_name: &str) -> String {
        format!(
            "{:?}_{}_{}_{}_{}",
            codec_config.codec_id,
            codec_config.preset.as_deref().unwrap_or("default"),
            codec_config.bitrate_kbps.unwrap_or(0),
            codec_config.cq_level.unwrap_or(0),
            sequence_name
        )
    }
}

/// Progress tracking for benchmark execution.
#[derive(Debug, Clone)]
pub struct ProgressTracker {
    total_tasks: usize,
    completed_tasks: Arc<Mutex<usize>>,
    start_time: Instant,
}

impl ProgressTracker {
    /// Create a new progress tracker.
    #[must_use]
    pub fn new(total_tasks: usize) -> Self {
        Self {
            total_tasks,
            completed_tasks: Arc::new(Mutex::new(0)),
            start_time: Instant::now(),
        }
    }

    /// Mark a task as completed.
    pub fn complete_task(&self) {
        if let Ok(mut completed) = self.completed_tasks.lock() {
            *completed += 1;
        }
    }

    /// Get current progress (0.0 to 1.0).
    #[must_use]
    pub fn progress(&self) -> f64 {
        if self.total_tasks == 0 {
            return 1.0;
        }

        let completed = self.completed_tasks.lock().map(|c| *c).unwrap_or(0);

        completed as f64 / self.total_tasks as f64
    }

    /// Get elapsed time.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Estimate remaining time.
    #[must_use]
    pub fn estimate_remaining(&self) -> Option<Duration> {
        let progress = self.progress();
        if progress == 0.0 || progress >= 1.0 {
            return None;
        }

        let elapsed = self.elapsed().as_secs_f64();
        let total_estimate = elapsed / progress;
        let remaining = total_estimate - elapsed;

        Some(Duration::from_secs_f64(remaining))
    }
}

/// The benchmark runner handles execution of codec benchmarks.
#[allow(dead_code)]
pub struct BenchmarkRunner {
    parallel_jobs: usize,
    warmup_iterations: usize,
    measurement_iterations: usize,
    metrics_calculator: MetricsCalculator,
    result_cache: ResultCache,
    max_frames: Option<usize>,
}

#[allow(dead_code)]
impl BenchmarkRunner {
    /// Create a new benchmark runner.
    #[must_use]
    pub fn new(config: &BenchmarkConfig) -> Self {
        let metrics_calculator =
            MetricsCalculator::new(config.enable_psnr, config.enable_ssim, config.enable_vmaf);

        let result_cache = ResultCache::new(config.cache_dir.clone());

        Self {
            parallel_jobs: config.parallel_jobs,
            warmup_iterations: config.warmup_iterations,
            measurement_iterations: config.measurement_iterations,
            metrics_calculator,
            result_cache,
            max_frames: config.max_frames,
        }
    }

    /// Run benchmarks for all sequences with a specific codec.
    ///
    /// # Errors
    ///
    /// Returns an error if benchmark execution fails.
    pub fn run_codec_sequences(
        &self,
        _codec_config: &CodecConfig,
    ) -> BenchResult<Vec<SequenceResult>> {
        // Placeholder implementation
        // In reality, this would iterate over sequences and run benchmarks

        Ok(Vec::new())
    }

    /// Run benchmark for a single sequence.
    fn run_sequence_benchmark(
        &self,
        codec_config: &CodecConfig,
        sequence: &TestSequence,
    ) -> BenchResult<SequenceResult> {
        // Check cache first
        let cache_key = ResultCache::generate_key(codec_config, &sequence.name);
        if let Some(cached_result) = self.result_cache.get(&cache_key) {
            return Ok(cached_result);
        }

        // Run warmup iterations
        for _ in 0..self.warmup_iterations {
            let _ = self.run_iteration(codec_config, sequence)?;
        }

        // Run measurement iterations
        let mut results = Vec::new();
        for _ in 0..self.measurement_iterations {
            results.push(self.run_iteration(codec_config, sequence)?);
        }

        // Take median result
        results.sort_by(|a, b| {
            a.encoding_fps
                .partial_cmp(&b.encoding_fps)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let result = results[results.len() / 2].clone();

        // Cache the result
        self.result_cache.set(cache_key, result.clone());

        Ok(result)
    }

    /// Run a single benchmark iteration.
    fn run_iteration(
        &self,
        _codec_config: &CodecConfig,
        _sequence: &TestSequence,
    ) -> BenchResult<SequenceResult> {
        // Placeholder for running a single benchmark iteration

        Err(BenchError::ExecutionFailed(
            "Benchmark execution not fully implemented".to_string(),
        ))
    }

    /// Encode a sequence with the specified codec.
    fn encode_sequence(
        &self,
        _codec_id: CodecId,
        _frames: &[VideoFrame],
    ) -> BenchResult<(Vec<u8>, ExecutionResult)> {
        // Placeholder for encoding

        Err(BenchError::ExecutionFailed(
            "Encoding not fully implemented".to_string(),
        ))
    }

    /// Decode a sequence.
    fn decode_sequence(
        &self,
        _codec_id: CodecId,
        _data: &[u8],
    ) -> BenchResult<(Vec<VideoFrame>, ExecutionResult)> {
        // Placeholder for decoding

        Err(BenchError::ExecutionFailed(
            "Decoding not fully implemented".to_string(),
        ))
    }

    /// Calculate quality metrics between original and reconstructed frames.
    fn calculate_metrics(
        &self,
        original: &[VideoFrame],
        reconstructed: &[VideoFrame],
    ) -> BenchResult<QualityMetrics> {
        self.metrics_calculator
            .calculate_sequence(original, reconstructed)
    }

    /// Load result cache from disk.
    ///
    /// # Errors
    ///
    /// Returns an error if loading fails.
    pub fn load_cache(&self) -> BenchResult<()> {
        self.result_cache.load_from_disk()
    }

    /// Save result cache to disk.
    ///
    /// # Errors
    ///
    /// Returns an error if saving fails.
    pub fn save_cache(&self) -> BenchResult<()> {
        self.result_cache.save_to_disk()
    }

    /// Clear the result cache.
    pub fn clear_cache(&self) {
        self.result_cache.clear();
    }
}

/// Parallel benchmark executor for running multiple benchmarks concurrently.
pub struct ParallelExecutor {
    max_parallel: usize,
}

impl ParallelExecutor {
    /// Create a new parallel executor.
    #[must_use]
    pub fn new(max_parallel: usize) -> Self {
        Self { max_parallel }
    }

    /// Execute benchmarks in parallel.
    ///
    /// # Errors
    ///
    /// Returns an error if any benchmark fails.
    pub fn execute<F>(&self, tasks: Vec<String>, task_fn: F) -> BenchResult<Vec<SequenceResult>>
    where
        F: Fn(&str) -> BenchResult<SequenceResult> + Send + Sync,
    {
        // Configure rayon thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.max_parallel)
            .build()
            .map_err(|e| BenchError::ExecutionFailed(format!("Thread pool error: {e}")))?;

        // Execute tasks in parallel
        pool.install(|| {
            tasks
                .par_iter()
                .map(|task| task_fn(task))
                .collect::<Result<Vec<_>, _>>()
        })
    }
}

/// Run multiple iterations and return the median result.
pub fn run_multiple_iterations<F>(
    iterations: usize,
    mut benchmark_fn: F,
) -> BenchResult<ExecutionResult>
where
    F: FnMut() -> BenchResult<ExecutionResult>,
{
    if iterations == 0 {
        return Err(BenchError::InvalidConfig(
            "Iterations must be > 0".to_string(),
        ));
    }

    let mut results = Vec::with_capacity(iterations);
    for _ in 0..iterations {
        results.push(benchmark_fn()?);
    }

    // Return median result based on FPS
    results.sort_by(|a, b| {
        a.fps
            .partial_cmp(&b.fps)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    Ok(results[results.len() / 2].clone())
}

/// Memory profiler for tracking memory usage during benchmarks.
#[derive(Debug)]
pub struct MemoryProfiler {
    baseline: u64,
    peak: u64,
}

impl MemoryProfiler {
    /// Create a new memory profiler.
    #[must_use]
    pub fn new() -> Self {
        let baseline = Self::current_memory_usage();
        Self {
            baseline,
            peak: baseline,
        }
    }

    /// Update peak memory usage.
    pub fn update(&mut self) {
        let current = Self::current_memory_usage();
        if current > self.peak {
            self.peak = current;
        }
    }

    /// Get peak memory usage above baseline.
    #[must_use]
    pub fn peak_usage(&self) -> u64 {
        self.peak.saturating_sub(self.baseline)
    }

    /// Get current memory usage (placeholder - would use actual system calls).
    fn current_memory_usage() -> u64 {
        // Placeholder - in reality would use platform-specific APIs
        // like getrusage on Unix or GetProcessMemoryInfo on Windows
        0
    }
}

impl Default for MemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// CPU profiler for tracking CPU usage during benchmarks.
#[derive(Debug)]
pub struct CpuProfiler {
    start_time: Instant,
    total_cpu_time: Duration,
}

impl CpuProfiler {
    /// Create a new CPU profiler.
    #[must_use]
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
            total_cpu_time: Duration::ZERO,
        }
    }

    /// Calculate CPU utilization percentage.
    #[must_use]
    pub fn utilization(&self) -> f64 {
        let wall_time = self.start_time.elapsed();
        if wall_time.as_secs_f64() == 0.0 {
            return 0.0;
        }

        (self.total_cpu_time.as_secs_f64() / wall_time.as_secs_f64()) * 100.0
    }
}

impl Default for CpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Benchmark configuration validator.
pub struct ConfigValidator;

impl ConfigValidator {
    /// Validate codec configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid.
    pub fn validate_codec(config: &CodecConfig) -> BenchResult<()> {
        // Check for conflicting bitrate and CQ settings
        if config.bitrate_kbps.is_some() && config.cq_level.is_some() {
            return Err(BenchError::InvalidConfig(
                "Cannot specify both bitrate and CQ level".to_string(),
            ));
        }

        // Validate bitrate range
        if let Some(bitrate) = config.bitrate_kbps {
            if bitrate == 0 {
                return Err(BenchError::InvalidConfig(
                    "Bitrate must be greater than 0".to_string(),
                ));
            }
            if bitrate > 100_000 {
                return Err(BenchError::InvalidConfig(
                    "Bitrate unreasonably high (> 100 Mbps)".to_string(),
                ));
            }
        }

        // Validate passes
        if config.passes == 0 {
            return Err(BenchError::InvalidConfig(
                "Number of passes must be greater than 0".to_string(),
            ));
        }
        if config.passes > 3 {
            return Err(BenchError::InvalidConfig(
                "Number of passes > 3 not supported".to_string(),
            ));
        }

        Ok(())
    }

    /// Validate sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if sequence is invalid.
    pub fn validate_sequence(sequence: &TestSequence) -> BenchResult<()> {
        sequence.validate()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::BenchmarkConfig;

    #[test]
    fn test_execution_result() {
        let result = ExecutionResult::new(100, Duration::from_secs(2));
        assert_eq!(result.frames, 100);
        assert_eq!(result.fps, 50.0);
    }

    #[test]
    fn test_execution_result_zero_duration() {
        let result = ExecutionResult::new(100, Duration::ZERO);
        assert_eq!(result.fps, 0.0);
    }

    #[test]
    fn test_execution_result_with_memory() {
        let result =
            ExecutionResult::new(100, Duration::from_secs(1)).with_memory(1024 * 1024 * 100);
        assert_eq!(result.peak_memory_bytes, Some(1024 * 1024 * 100));
    }

    #[test]
    fn test_execution_result_with_cpu() {
        let result = ExecutionResult::new(100, Duration::from_secs(1)).with_cpu(75.5);
        assert_eq!(result.cpu_utilization, Some(75.5));
    }

    #[test]
    fn test_benchmark_runner_creation() {
        let config = BenchmarkConfig::default();
        let runner = BenchmarkRunner::new(&config);
        assert_eq!(runner.warmup_iterations, 1);
    }

    #[test]
    fn test_run_multiple_iterations() {
        let mut counter = 0;
        let result = run_multiple_iterations(3, || {
            counter += 1;
            Ok(ExecutionResult::new(
                100,
                Duration::from_millis(counter * 10),
            ))
        });

        assert!(result.is_ok());
        let result = result.expect("result should be valid");
        assert_eq!(result.frames, 100);
    }

    #[test]
    fn test_run_multiple_iterations_zero() {
        let result =
            run_multiple_iterations(0, || Ok(ExecutionResult::new(100, Duration::from_secs(1))));

        assert!(result.is_err());
    }

    #[test]
    fn test_result_cache() {
        let cache = ResultCache::new(None);
        assert!(cache.get("test").is_none());

        // Would need a full SequenceResult to test set
    }

    #[test]
    fn test_progress_tracker() {
        let tracker = ProgressTracker::new(10);
        assert_eq!(tracker.progress(), 0.0);

        tracker.complete_task();
        assert_eq!(tracker.progress(), 0.1);

        for _ in 0..9 {
            tracker.complete_task();
        }
        assert_eq!(tracker.progress(), 1.0);
    }

    #[test]
    fn test_progress_tracker_zero_tasks() {
        let tracker = ProgressTracker::new(0);
        assert_eq!(tracker.progress(), 1.0);
    }

    #[test]
    fn test_memory_profiler() {
        let mut profiler = MemoryProfiler::new();
        profiler.update();
        let _ = profiler.peak_usage();
    }

    #[test]
    fn test_cpu_profiler() {
        let profiler = CpuProfiler::new();
        let _ = profiler.utilization();
    }

    #[test]
    fn test_cache_key_generation() {
        let config = CodecConfig::new(CodecId::Av1)
            .with_preset("medium")
            .with_bitrate(2000);
        let key = ResultCache::generate_key(&config, "test_sequence");
        assert!(key.contains("Av1"));
        assert!(key.contains("medium"));
        assert!(key.contains("2000"));
        assert!(key.contains("test_sequence"));
    }

    #[test]
    fn test_config_validator_bitrate_and_cq() {
        let config = CodecConfig::new(CodecId::Av1)
            .with_bitrate(2000)
            .with_cq_level(30);

        assert!(ConfigValidator::validate_codec(&config).is_err());
    }

    #[test]
    fn test_config_validator_zero_bitrate() {
        let config = CodecConfig::new(CodecId::Av1).with_bitrate(0);

        assert!(ConfigValidator::validate_codec(&config).is_err());
    }

    #[test]
    fn test_config_validator_high_bitrate() {
        let config = CodecConfig::new(CodecId::Av1).with_bitrate(150_000);

        assert!(ConfigValidator::validate_codec(&config).is_err());
    }

    #[test]
    fn test_config_validator_zero_passes() {
        let mut config = CodecConfig::new(CodecId::Av1);
        config.passes = 0;

        assert!(ConfigValidator::validate_codec(&config).is_err());
    }

    #[test]
    fn test_config_validator_too_many_passes() {
        let mut config = CodecConfig::new(CodecId::Av1);
        config.passes = 4;

        assert!(ConfigValidator::validate_codec(&config).is_err());
    }

    #[test]
    fn test_config_validator_valid_config() {
        let config = CodecConfig::new(CodecId::Av1)
            .with_bitrate(2000)
            .with_passes(2);

        assert!(ConfigValidator::validate_codec(&config).is_ok());
    }

    #[test]
    fn test_parallel_executor_creation() {
        let executor = ParallelExecutor::new(4);
        assert_eq!(executor.max_parallel, 4);
    }
}
