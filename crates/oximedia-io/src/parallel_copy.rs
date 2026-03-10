#![allow(dead_code)]
//! Parallel file copy engine with chunked I/O and progress tracking.
//!
//! Provides efficient parallel copy operations by splitting files into
//! chunks and copying them concurrently, with real-time progress feedback.

use std::collections::VecDeque;

/// Default chunk size for parallel copy (1 MiB).
const DEFAULT_CHUNK_SIZE: usize = 1024 * 1024;

/// Maximum number of concurrent copy workers.
const MAX_WORKERS: usize = 16;

/// Copy progress information.
#[derive(Debug, Clone)]
pub struct CopyProgress {
    /// Total bytes to copy.
    pub total_bytes: u64,
    /// Bytes copied so far.
    pub copied_bytes: u64,
    /// Number of chunks completed.
    pub chunks_completed: usize,
    /// Total number of chunks.
    pub total_chunks: usize,
    /// Whether the copy is finished.
    pub finished: bool,
}

impl CopyProgress {
    /// Create a new copy progress tracker.
    pub fn new(total_bytes: u64, total_chunks: usize) -> Self {
        Self {
            total_bytes,
            copied_bytes: 0,
            chunks_completed: 0,
            total_chunks,
            finished: false,
        }
    }

    /// Get the progress fraction (0.0 to 1.0).
    #[allow(clippy::cast_precision_loss)]
    pub fn fraction(&self) -> f64 {
        if self.total_bytes == 0 {
            return 1.0;
        }
        self.copied_bytes as f64 / self.total_bytes as f64
    }

    /// Get progress as a percentage (0 to 100).
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_sign_loss)]
    pub fn percentage(&self) -> u8 {
        (self.fraction() * 100.0).min(100.0) as u8
    }

    /// Record completion of a chunk.
    pub fn complete_chunk(&mut self, bytes: u64) {
        self.copied_bytes = self.copied_bytes.saturating_add(bytes);
        self.chunks_completed += 1;
        if self.chunks_completed >= self.total_chunks {
            self.finished = true;
        }
    }
}

/// A chunk descriptor for parallel copy.
#[derive(Debug, Clone)]
pub struct CopyChunk {
    /// Offset within the source.
    pub offset: u64,
    /// Length of this chunk.
    pub length: usize,
    /// Index of this chunk.
    pub index: usize,
}

/// Configuration for the parallel copy engine.
#[derive(Debug, Clone)]
pub struct ParallelCopyConfig {
    /// Size of each chunk in bytes.
    pub chunk_size: usize,
    /// Number of concurrent workers.
    pub workers: usize,
    /// Whether to verify after copy.
    pub verify: bool,
    /// Whether to preserve file metadata.
    pub preserve_metadata: bool,
}

impl Default for ParallelCopyConfig {
    fn default() -> Self {
        Self {
            chunk_size: DEFAULT_CHUNK_SIZE,
            workers: 4,
            verify: false,
            preserve_metadata: true,
        }
    }
}

impl ParallelCopyConfig {
    /// Create a new config with the given chunk size.
    pub fn with_chunk_size(mut self, size: usize) -> Self {
        self.chunk_size = size.max(4096);
        self
    }

    /// Set the number of workers.
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.workers = workers.clamp(1, MAX_WORKERS);
        self
    }

    /// Enable or disable verification.
    pub fn with_verify(mut self, verify: bool) -> Self {
        self.verify = verify;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> Result<(), String> {
        if self.chunk_size < 512 {
            return Err("Chunk size must be at least 512 bytes".to_string());
        }
        if self.workers == 0 || self.workers > MAX_WORKERS {
            return Err(format!("Workers must be between 1 and {MAX_WORKERS}"));
        }
        Ok(())
    }
}

/// Plan for a parallel copy operation.
#[derive(Debug, Clone)]
pub struct CopyPlan {
    /// Source path.
    pub source: String,
    /// Destination path.
    pub destination: String,
    /// File size in bytes.
    pub file_size: u64,
    /// Chunks to copy.
    pub chunks: Vec<CopyChunk>,
    /// Configuration used.
    pub config: ParallelCopyConfig,
}

impl CopyPlan {
    /// Create a copy plan for a given file size.
    #[allow(clippy::cast_possible_truncation)]
    pub fn create(source: &str, destination: &str, file_size: u64, config: ParallelCopyConfig) -> Self {
        let chunk_size = config.chunk_size;
        let mut chunks = Vec::new();
        let mut offset = 0u64;
        let mut index = 0;

        while offset < file_size {
            let remaining = file_size - offset;
            let length = if remaining > chunk_size as u64 {
                chunk_size
            } else {
                remaining as usize
            };
            chunks.push(CopyChunk {
                offset,
                length,
                index,
            });
            offset += length as u64;
            index += 1;
        }

        Self {
            source: source.to_string(),
            destination: destination.to_string(),
            file_size,
            chunks,
            config,
        }
    }

    /// Get the total number of chunks.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Estimate throughput needed for a target duration (bytes/sec).
    #[allow(clippy::cast_precision_loss)]
    pub fn required_throughput(&self, target_seconds: f64) -> f64 {
        if target_seconds <= 0.0 {
            return f64::INFINITY;
        }
        self.file_size as f64 / target_seconds
    }
}

/// Copy work queue for scheduling chunks across workers.
#[derive(Debug)]
pub struct CopyWorkQueue {
    /// Pending chunks.
    pending: VecDeque<CopyChunk>,
    /// In-progress chunk indices.
    in_progress: Vec<usize>,
    /// Completed chunk indices.
    completed: Vec<usize>,
}

impl CopyWorkQueue {
    /// Create a new work queue from a copy plan.
    pub fn from_plan(plan: &CopyPlan) -> Self {
        let pending: VecDeque<CopyChunk> = plan.chunks.iter().cloned().collect();
        Self {
            pending,
            in_progress: Vec::new(),
            completed: Vec::new(),
        }
    }

    /// Take the next chunk to process.
    pub fn take_next(&mut self) -> Option<CopyChunk> {
        if let Some(chunk) = self.pending.pop_front() {
            self.in_progress.push(chunk.index);
            Some(chunk)
        } else {
            None
        }
    }

    /// Mark a chunk as completed.
    pub fn mark_completed(&mut self, index: usize) {
        self.in_progress.retain(|&i| i != index);
        self.completed.push(index);
    }

    /// Check if all chunks are completed.
    pub fn is_finished(&self) -> bool {
        self.pending.is_empty() && self.in_progress.is_empty()
    }

    /// Get the number of remaining chunks.
    pub fn remaining(&self) -> usize {
        self.pending.len() + self.in_progress.len()
    }

    /// Get the number of completed chunks.
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }
}

/// Throughput calculator for copy operations.
#[derive(Debug, Clone)]
pub struct ThroughputCalculator {
    /// Samples of (bytes, elapsed_nanos).
    samples: Vec<(u64, u64)>,
    /// Maximum number of samples to keep.
    max_samples: usize,
}

impl ThroughputCalculator {
    /// Create a new throughput calculator.
    pub fn new(max_samples: usize) -> Self {
        Self {
            samples: Vec::with_capacity(max_samples),
            max_samples: max_samples.max(1),
        }
    }

    /// Add a sample.
    pub fn add_sample(&mut self, bytes: u64, elapsed_nanos: u64) {
        if self.samples.len() >= self.max_samples {
            self.samples.remove(0);
        }
        self.samples.push((bytes, elapsed_nanos));
    }

    /// Get average throughput in bytes/sec.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_throughput(&self) -> f64 {
        if self.samples.is_empty() {
            return 0.0;
        }
        let total_bytes: u64 = self.samples.iter().map(|(b, _)| b).sum();
        let total_nanos: u64 = self.samples.iter().map(|(_, n)| n).sum();
        if total_nanos == 0 {
            return 0.0;
        }
        total_bytes as f64 / (total_nanos as f64 / 1_000_000_000.0)
    }

    /// Get the number of samples recorded.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Estimate time remaining for given bytes.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_remaining_secs(&self, remaining_bytes: u64) -> Option<f64> {
        let throughput = self.average_throughput();
        if throughput <= 0.0 {
            return None;
        }
        Some(remaining_bytes as f64 / throughput)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_copy_progress_new() {
        let p = CopyProgress::new(1000, 10);
        assert_eq!(p.total_bytes, 1000);
        assert_eq!(p.copied_bytes, 0);
        assert_eq!(p.chunks_completed, 0);
        assert!(!p.finished);
    }

    #[test]
    fn test_copy_progress_fraction() {
        let mut p = CopyProgress::new(1000, 2);
        assert!((p.fraction() - 0.0).abs() < f64::EPSILON);
        p.complete_chunk(500);
        assert!((p.fraction() - 0.5).abs() < f64::EPSILON);
        p.complete_chunk(500);
        assert!((p.fraction() - 1.0).abs() < f64::EPSILON);
        assert!(p.finished);
    }

    #[test]
    fn test_copy_progress_zero_size() {
        let p = CopyProgress::new(0, 0);
        assert!((p.fraction() - 1.0).abs() < f64::EPSILON);
        assert_eq!(p.percentage(), 100);
    }

    #[test]
    fn test_copy_progress_percentage() {
        let mut p = CopyProgress::new(200, 4);
        p.complete_chunk(50);
        assert_eq!(p.percentage(), 25);
    }

    #[test]
    fn test_parallel_copy_config_default() {
        let cfg = ParallelCopyConfig::default();
        assert_eq!(cfg.chunk_size, DEFAULT_CHUNK_SIZE);
        assert_eq!(cfg.workers, 4);
        assert!(!cfg.verify);
        assert!(cfg.preserve_metadata);
    }

    #[test]
    fn test_config_builders() {
        let cfg = ParallelCopyConfig::default()
            .with_chunk_size(8192)
            .with_workers(8)
            .with_verify(true);
        assert_eq!(cfg.chunk_size, 8192);
        assert_eq!(cfg.workers, 8);
        assert!(cfg.verify);
    }

    #[test]
    fn test_config_clamp_workers() {
        let cfg = ParallelCopyConfig::default().with_workers(100);
        assert_eq!(cfg.workers, MAX_WORKERS);
        let cfg2 = ParallelCopyConfig::default().with_workers(0);
        assert_eq!(cfg2.workers, 1);
    }

    #[test]
    fn test_config_validate() {
        let cfg = ParallelCopyConfig::default();
        assert!(cfg.validate().is_ok());

        let mut bad = ParallelCopyConfig::default();
        bad.chunk_size = 100;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn test_copy_plan_create() {
        let plan = CopyPlan::create("src.bin", "dst.bin", 3000, ParallelCopyConfig::default().with_chunk_size(1000));
        assert_eq!(plan.chunk_count(), 3);
        assert_eq!(plan.chunks[0].offset, 0);
        assert_eq!(plan.chunks[0].length, 1000);
        assert_eq!(plan.chunks[2].offset, 2000);
        assert_eq!(plan.chunks[2].length, 1000);
    }

    #[test]
    fn test_copy_plan_uneven() {
        let plan = CopyPlan::create("a", "b", 2500, ParallelCopyConfig::default().with_chunk_size(1000));
        assert_eq!(plan.chunk_count(), 3);
        assert_eq!(plan.chunks[2].length, 500);
    }

    #[test]
    fn test_copy_plan_empty() {
        let plan = CopyPlan::create("a", "b", 0, ParallelCopyConfig::default());
        assert_eq!(plan.chunk_count(), 0);
    }

    #[test]
    fn test_work_queue_lifecycle() {
        let plan = CopyPlan::create("a", "b", 3000, ParallelCopyConfig::default().with_chunk_size(1000));
        let mut queue = CopyWorkQueue::from_plan(&plan);
        assert_eq!(queue.remaining(), 3);
        assert!(!queue.is_finished());

        let c1 = queue.take_next().expect("take_next should return chunk");
        assert_eq!(c1.index, 0);
        assert_eq!(queue.remaining(), 3); // 2 pending + 1 in progress

        queue.mark_completed(0);
        assert_eq!(queue.remaining(), 2);
        assert_eq!(queue.completed_count(), 1);

        let _ = queue.take_next().expect("take_next should return chunk");
        let _ = queue.take_next().expect("take_next should return chunk");
        queue.mark_completed(1);
        queue.mark_completed(2);
        assert!(queue.is_finished());
    }

    #[test]
    fn test_throughput_calculator() {
        let mut calc = ThroughputCalculator::new(10);
        assert_eq!(calc.sample_count(), 0);
        assert!((calc.average_throughput() - 0.0).abs() < f64::EPSILON);

        // 1000 bytes in 1 second (1_000_000_000 nanos)
        calc.add_sample(1000, 1_000_000_000);
        assert!((calc.average_throughput() - 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_throughput_estimate_remaining() {
        let mut calc = ThroughputCalculator::new(10);
        assert!(calc.estimate_remaining_secs(1000).is_none());

        calc.add_sample(1000, 1_000_000_000);
        let est = calc.estimate_remaining_secs(5000).expect("estimate should succeed");
        assert!((est - 5.0).abs() < 0.01);
    }

    #[test]
    fn test_throughput_max_samples() {
        let mut calc = ThroughputCalculator::new(3);
        calc.add_sample(100, 100);
        calc.add_sample(200, 200);
        calc.add_sample(300, 300);
        calc.add_sample(400, 400);
        assert_eq!(calc.sample_count(), 3);
    }

    #[test]
    fn test_required_throughput() {
        let plan = CopyPlan::create("a", "b", 10_000_000, ParallelCopyConfig::default());
        let throughput = plan.required_throughput(10.0);
        assert!((throughput - 1_000_000.0).abs() < 1.0);
    }
}
