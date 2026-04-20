//! Job splitting: automatically partition large transcode jobs across workers.
//!
//! When a job's input is very large (e.g. a multi-hour video file), processing
//! it on a single worker creates a bottleneck.  This module splits such jobs
//! into smaller **chunks** that can be executed in parallel across a worker pool,
//! then merges the results back into the final output.
//!
//! ## Splitting strategies
//!
//! - **BySize**: split when the input exceeds a byte threshold.
//! - **ByDuration**: split by estimated media duration (requires a duration hint).
//! - **ByCount**: split a list of input files into N equal-sized groups.
//! - **Adaptive**: choose the strategy automatically based on input characteristics.
//!
//! ## Merge strategies
//!
//! - **Concatenate**: join chunk outputs sequentially.
//! - **MuxStreams**: multiplex audio/video streams from separate chunks.
//! - **Passthrough**: no merge needed (each chunk is self-contained output).

#![allow(dead_code)]

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{BatchError, Result};
use crate::types::JobId;

// ---------------------------------------------------------------------------
// Splitting strategy
// ---------------------------------------------------------------------------

/// How a large job should be partitioned.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum SplitStrategy {
    /// Split when total input size exceeds `threshold_bytes`.
    /// Each chunk targets approximately `chunk_bytes`.
    BySize {
        /// Maximum bytes per chunk.
        chunk_bytes: u64,
    },
    /// Split by estimated media duration.
    ByDuration {
        /// Maximum seconds per chunk.
        chunk_secs: f64,
    },
    /// Split a list of input files into `n` equal-sized groups.
    ByCount {
        /// Number of chunks to create.
        n: usize,
    },
    /// Automatically choose the best strategy based on input.
    Adaptive {
        /// Maximum bytes per chunk (used if splitting by size).
        max_chunk_bytes: u64,
        /// Maximum seconds per chunk (used if splitting by duration).
        max_chunk_secs: f64,
    },
}

impl std::fmt::Display for SplitStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BySize { chunk_bytes } => write!(f, "by_size({chunk_bytes} bytes)"),
            Self::ByDuration { chunk_secs } => write!(f, "by_duration({chunk_secs}s)"),
            Self::ByCount { n } => write!(f, "by_count({n})"),
            Self::Adaptive {
                max_chunk_bytes,
                max_chunk_secs,
            } => write!(f, "adaptive(size={max_chunk_bytes}, dur={max_chunk_secs}s)"),
        }
    }
}

/// How chunk outputs are reassembled into the final result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MergeStrategy {
    /// Concatenate chunks sequentially (e.g. for video segments).
    Concatenate,
    /// Multiplex audio and video from separate chunk streams.
    MuxStreams,
    /// No merge needed — each chunk is an independent output.
    Passthrough,
}

impl std::fmt::Display for MergeStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Concatenate => write!(f, "concatenate"),
            Self::MuxStreams => write!(f, "mux_streams"),
            Self::Passthrough => write!(f, "passthrough"),
        }
    }
}

// ---------------------------------------------------------------------------
// Input descriptor
// ---------------------------------------------------------------------------

/// Describes the input to be split.
#[derive(Debug, Clone)]
pub struct SplitInput {
    /// Parent job ID that owns this split.
    pub parent_job_id: JobId,
    /// Total input size in bytes.
    pub total_size_bytes: u64,
    /// Estimated total duration in seconds (if known).
    pub duration_secs: Option<f64>,
    /// Individual input file paths and their sizes.
    pub files: Vec<InputFile>,
    /// Metadata carried forward to each chunk.
    pub metadata: HashMap<String, String>,
}

/// A single input file with its size.
#[derive(Debug, Clone)]
pub struct InputFile {
    /// Path to the file.
    pub path: String,
    /// Size of the file in bytes.
    pub size_bytes: u64,
    /// Duration of the media in seconds (if known).
    pub duration_secs: Option<f64>,
}

impl InputFile {
    /// Create a new input file descriptor.
    #[must_use]
    pub fn new(path: impl Into<String>, size_bytes: u64) -> Self {
        Self {
            path: path.into(),
            size_bytes,
            duration_secs: None,
        }
    }

    /// Builder: set the duration.
    #[must_use]
    pub fn with_duration(mut self, secs: f64) -> Self {
        self.duration_secs = Some(secs);
        self
    }
}

// ---------------------------------------------------------------------------
// Chunk
// ---------------------------------------------------------------------------

/// A single chunk produced by the splitter.
#[derive(Debug, Clone)]
pub struct JobChunk {
    /// Unique chunk identifier.
    pub chunk_id: String,
    /// Parent job ID.
    pub parent_job_id: JobId,
    /// Zero-based index of this chunk.
    pub index: usize,
    /// Total number of chunks in the split.
    pub total_chunks: usize,
    /// Files assigned to this chunk.
    pub files: Vec<InputFile>,
    /// Byte range within a single file (for intra-file splitting).
    pub byte_range: Option<(u64, u64)>,
    /// Time range within a single file (for duration-based splitting).
    pub time_range: Option<(f64, f64)>,
    /// Estimated size of this chunk in bytes.
    pub estimated_size_bytes: u64,
    /// Estimated duration of this chunk in seconds.
    pub estimated_duration_secs: Option<f64>,
    /// Metadata inherited from the parent.
    pub metadata: HashMap<String, String>,
}

impl JobChunk {
    /// Completion percentage if this chunk is `index` out of `total_chunks`.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn progress_pct(&self) -> f64 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        ((self.index + 1) as f64 / self.total_chunks as f64) * 100.0
    }
}

// ---------------------------------------------------------------------------
// Split plan
// ---------------------------------------------------------------------------

/// The result of planning a job split.
#[derive(Debug, Clone)]
pub struct SplitPlan {
    /// The parent job ID.
    pub parent_job_id: JobId,
    /// Strategy that was used.
    pub strategy: SplitStrategy,
    /// Merge strategy for reassembly.
    pub merge_strategy: MergeStrategy,
    /// The chunks produced.
    pub chunks: Vec<JobChunk>,
    /// Whether splitting was actually needed (false if input was small enough).
    pub was_split: bool,
}

impl SplitPlan {
    /// Number of chunks.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Total estimated bytes across all chunks.
    #[must_use]
    pub fn total_estimated_bytes(&self) -> u64 {
        self.chunks.iter().map(|c| c.estimated_size_bytes).sum()
    }
}

// ---------------------------------------------------------------------------
// Job splitter
// ---------------------------------------------------------------------------

/// Minimum chunk size to avoid creating trivially small chunks (1 MiB).
const MIN_CHUNK_BYTES: u64 = 1024 * 1024;

/// Splits large jobs into parallelisable chunks.
#[derive(Debug, Clone)]
pub struct JobSplitter {
    /// Default split strategy.
    pub default_strategy: SplitStrategy,
    /// Default merge strategy.
    pub default_merge: MergeStrategy,
    /// Minimum input size (bytes) before splitting is considered.
    pub min_split_threshold_bytes: u64,
    /// Maximum number of chunks to create.
    pub max_chunks: usize,
}

impl JobSplitter {
    /// Create a new splitter with sensible defaults.
    #[must_use]
    pub fn new() -> Self {
        Self {
            default_strategy: SplitStrategy::Adaptive {
                max_chunk_bytes: 256 * 1024 * 1024, // 256 MiB
                max_chunk_secs: 300.0,              // 5 minutes
            },
            default_merge: MergeStrategy::Concatenate,
            min_split_threshold_bytes: 64 * 1024 * 1024, // 64 MiB
            max_chunks: 64,
        }
    }

    /// Set the default split strategy.
    #[must_use]
    pub fn with_strategy(mut self, strategy: SplitStrategy) -> Self {
        self.default_strategy = strategy;
        self
    }

    /// Set the default merge strategy.
    #[must_use]
    pub fn with_merge(mut self, merge: MergeStrategy) -> Self {
        self.default_merge = merge;
        self
    }

    /// Set the minimum split threshold.
    #[must_use]
    pub fn with_min_threshold(mut self, bytes: u64) -> Self {
        self.min_split_threshold_bytes = bytes;
        self
    }

    /// Set the maximum number of chunks.
    #[must_use]
    pub fn with_max_chunks(mut self, max: usize) -> Self {
        self.max_chunks = max.max(1);
        self
    }

    /// Plan a split for the given input.
    ///
    /// If the input is below the threshold, returns a single-chunk plan with
    /// `was_split = false`.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobConfig`] if the input has no files.
    pub fn plan(&self, input: &SplitInput) -> Result<SplitPlan> {
        self.plan_with_strategy(input, &self.default_strategy)
    }

    /// Plan a split using a specific strategy.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobConfig`] if the input has no files.
    pub fn plan_with_strategy(
        &self,
        input: &SplitInput,
        strategy: &SplitStrategy,
    ) -> Result<SplitPlan> {
        if input.files.is_empty() {
            return Err(BatchError::InvalidJobConfig(
                "Cannot split job with no input files".to_string(),
            ));
        }

        // If total size is below threshold, no split needed.
        if input.total_size_bytes < self.min_split_threshold_bytes {
            return Ok(self.single_chunk_plan(input));
        }

        let chunks = match strategy {
            SplitStrategy::BySize { chunk_bytes } => self.split_by_size(input, *chunk_bytes),
            SplitStrategy::ByDuration { chunk_secs } => {
                self.split_by_duration(input, *chunk_secs)?
            }
            SplitStrategy::ByCount { n } => self.split_by_count(input, *n),
            SplitStrategy::Adaptive {
                max_chunk_bytes,
                max_chunk_secs,
            } => self.split_adaptive(input, *max_chunk_bytes, *max_chunk_secs)?,
        };

        let was_split = chunks.len() > 1;

        Ok(SplitPlan {
            parent_job_id: input.parent_job_id.clone(),
            strategy: strategy.clone(),
            merge_strategy: self.default_merge.clone(),
            chunks,
            was_split,
        })
    }

    /// Create a single-chunk plan (no splitting).
    fn single_chunk_plan(&self, input: &SplitInput) -> SplitPlan {
        let chunk = JobChunk {
            chunk_id: format!("{}-chunk-0", input.parent_job_id.as_str()),
            parent_job_id: input.parent_job_id.clone(),
            index: 0,
            total_chunks: 1,
            files: input.files.clone(),
            byte_range: None,
            time_range: None,
            estimated_size_bytes: input.total_size_bytes,
            estimated_duration_secs: input.duration_secs,
            metadata: input.metadata.clone(),
        };

        SplitPlan {
            parent_job_id: input.parent_job_id.clone(),
            strategy: self.default_strategy.clone(),
            merge_strategy: self.default_merge.clone(),
            chunks: vec![chunk],
            was_split: false,
        }
    }

    /// Split by byte size: group files into chunks that each total ≤ `chunk_bytes`.
    fn split_by_size(&self, input: &SplitInput, chunk_bytes: u64) -> Vec<JobChunk> {
        let effective_chunk = chunk_bytes.max(MIN_CHUNK_BYTES);

        // If there's only one file, split it into byte ranges.
        if input.files.len() == 1 {
            return self.split_single_file_by_size(input, effective_chunk);
        }

        // Multiple files: group them into chunks by cumulative size.
        let mut chunks = Vec::new();
        let mut current_files: Vec<InputFile> = Vec::new();
        let mut current_size = 0u64;

        for file in &input.files {
            if !current_files.is_empty()
                && current_size + file.size_bytes > effective_chunk
                && chunks.len() < self.max_chunks - 1
            {
                let idx = chunks.len();
                chunks.push(self.make_chunk(input, idx, 0, current_files.clone(), current_size));
                current_files.clear();
                current_size = 0;
            }
            current_files.push(file.clone());
            current_size += file.size_bytes;
        }

        if !current_files.is_empty() {
            let idx = chunks.len();
            chunks.push(self.make_chunk(input, idx, 0, current_files, current_size));
        }

        // Fix total_chunks in all chunks.
        let total = chunks.len();
        for chunk in &mut chunks {
            chunk.total_chunks = total;
        }

        chunks
    }

    /// Split a single file into byte ranges.
    fn split_single_file_by_size(&self, input: &SplitInput, chunk_bytes: u64) -> Vec<JobChunk> {
        let file = &input.files[0];
        let file_size = file.size_bytes;

        let n_chunks =
            ((file_size as f64 / chunk_bytes as f64).ceil() as usize).clamp(1, self.max_chunks);

        let bytes_per_chunk = (file_size + n_chunks as u64 - 1) / n_chunks as u64;

        let mut chunks = Vec::with_capacity(n_chunks);
        let mut offset = 0u64;

        for i in 0..n_chunks {
            let end = (offset + bytes_per_chunk).min(file_size);
            let mut chunk = self.make_chunk(input, i, n_chunks, vec![file.clone()], end - offset);
            chunk.byte_range = Some((offset, end));

            // Estimate duration proportionally if known.
            if let Some(total_dur) = file.duration_secs {
                let frac = (end - offset) as f64 / file_size as f64;
                chunk.estimated_duration_secs = Some(total_dur * frac);
            }

            chunks.push(chunk);
            offset = end;
            if offset >= file_size {
                break;
            }
        }

        chunks
    }

    /// Split by duration: create chunks targeting `chunk_secs` each.
    fn split_by_duration(&self, input: &SplitInput, chunk_secs: f64) -> Result<Vec<JobChunk>> {
        let total_duration = input
            .duration_secs
            .or_else(|| {
                let sum: f64 = input.files.iter().filter_map(|f| f.duration_secs).sum();
                if sum > 0.0 {
                    Some(sum)
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                BatchError::InvalidJobConfig(
                    "Duration-based splitting requires duration hints on the input".to_string(),
                )
            })?;

        if chunk_secs <= 0.0 {
            return Err(BatchError::InvalidJobConfig(
                "chunk_secs must be positive".to_string(),
            ));
        }

        let n_chunks = ((total_duration / chunk_secs).ceil() as usize).clamp(1, self.max_chunks);
        let secs_per_chunk = total_duration / n_chunks as f64;

        let bytes_per_sec = if total_duration > 0.0 {
            input.total_size_bytes as f64 / total_duration
        } else {
            0.0
        };

        let mut chunks = Vec::with_capacity(n_chunks);
        let mut time_offset = 0.0f64;

        for i in 0..n_chunks {
            let time_end = (time_offset + secs_per_chunk).min(total_duration);
            let est_bytes = ((time_end - time_offset) * bytes_per_sec) as u64;

            let mut chunk = self.make_chunk(input, i, n_chunks, input.files.clone(), est_bytes);
            chunk.time_range = Some((time_offset, time_end));
            chunk.estimated_duration_secs = Some(time_end - time_offset);

            chunks.push(chunk);
            time_offset = time_end;
            if time_offset >= total_duration {
                break;
            }
        }

        Ok(chunks)
    }

    /// Split by count: distribute files evenly across `n` chunks.
    fn split_by_count(&self, input: &SplitInput, n: usize) -> Vec<JobChunk> {
        let effective_n = n.clamp(1, self.max_chunks).min(input.files.len());
        let files_per_chunk = (input.files.len() + effective_n - 1) / effective_n;

        let mut chunks = Vec::with_capacity(effective_n);
        for (i, group) in input.files.chunks(files_per_chunk).enumerate() {
            let size: u64 = group.iter().map(|f| f.size_bytes).sum();
            chunks.push(self.make_chunk(input, i, effective_n, group.to_vec(), size));
        }

        // Fix total_chunks.
        let total = chunks.len();
        for chunk in &mut chunks {
            chunk.total_chunks = total;
        }

        chunks
    }

    /// Adaptive splitting: choose size or duration based on what's available.
    fn split_adaptive(
        &self,
        input: &SplitInput,
        max_chunk_bytes: u64,
        max_chunk_secs: f64,
    ) -> Result<Vec<JobChunk>> {
        // Prefer duration-based if we have duration info.
        let has_duration =
            input.duration_secs.is_some() || input.files.iter().any(|f| f.duration_secs.is_some());

        if has_duration {
            self.split_by_duration(input, max_chunk_secs)
        } else {
            Ok(self.split_by_size(input, max_chunk_bytes))
        }
    }

    /// Helper: create a `JobChunk` with common fields.
    fn make_chunk(
        &self,
        input: &SplitInput,
        index: usize,
        total: usize,
        files: Vec<InputFile>,
        estimated_size_bytes: u64,
    ) -> JobChunk {
        JobChunk {
            chunk_id: format!("{}-chunk-{index}", input.parent_job_id.as_str()),
            parent_job_id: input.parent_job_id.clone(),
            index,
            total_chunks: total,
            files,
            byte_range: None,
            time_range: None,
            estimated_size_bytes,
            estimated_duration_secs: None,
            metadata: input.metadata.clone(),
        }
    }
}

impl Default for JobSplitter {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Merge tracker
// ---------------------------------------------------------------------------

/// Tracks which chunks have been completed and determines when a merge can proceed.
#[derive(Debug)]
pub struct MergeTracker {
    parent_job_id: JobId,
    total_chunks: usize,
    completed: Vec<bool>,
    failed: Vec<Option<String>>,
    merge_strategy: MergeStrategy,
}

impl MergeTracker {
    /// Create a new merge tracker for a split plan.
    #[must_use]
    pub fn from_plan(plan: &SplitPlan) -> Self {
        let n = plan.chunk_count();
        Self {
            parent_job_id: plan.parent_job_id.clone(),
            total_chunks: n,
            completed: vec![false; n],
            failed: vec![None; n],
            merge_strategy: plan.merge_strategy.clone(),
        }
    }

    /// Mark a chunk as completed.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobConfig`] if the index is out of bounds.
    pub fn mark_completed(&mut self, chunk_index: usize) -> Result<()> {
        if chunk_index >= self.total_chunks {
            return Err(BatchError::InvalidJobConfig(format!(
                "Chunk index {chunk_index} out of bounds (total: {})",
                self.total_chunks
            )));
        }
        self.completed[chunk_index] = true;
        Ok(())
    }

    /// Mark a chunk as failed.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::InvalidJobConfig`] if the index is out of bounds.
    pub fn mark_failed(&mut self, chunk_index: usize, error: impl Into<String>) -> Result<()> {
        if chunk_index >= self.total_chunks {
            return Err(BatchError::InvalidJobConfig(format!(
                "Chunk index {chunk_index} out of bounds (total: {})",
                self.total_chunks
            )));
        }
        self.failed[chunk_index] = Some(error.into());
        Ok(())
    }

    /// Whether all chunks are completed (none failed).
    #[must_use]
    pub fn is_ready_for_merge(&self) -> bool {
        self.completed.iter().all(|&c| c) && self.failed.iter().all(|f| f.is_none())
    }

    /// Whether any chunk has failed.
    #[must_use]
    pub fn has_failures(&self) -> bool {
        self.failed.iter().any(|f| f.is_some())
    }

    /// Number of completed chunks.
    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.completed.iter().filter(|&&c| c).count()
    }

    /// Number of failed chunks.
    #[must_use]
    pub fn failed_count(&self) -> usize {
        self.failed.iter().filter(|f| f.is_some()).count()
    }

    /// Completion progress as a percentage.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn progress_pct(&self) -> f64 {
        if self.total_chunks == 0 {
            return 0.0;
        }
        (self.completed_count() as f64 / self.total_chunks as f64) * 100.0
    }

    /// The merge strategy to use.
    #[must_use]
    pub fn merge_strategy(&self) -> &MergeStrategy {
        &self.merge_strategy
    }

    /// Parent job ID.
    #[must_use]
    pub fn parent_job_id(&self) -> &JobId {
        &self.parent_job_id
    }

    /// Collect all failure messages.
    #[must_use]
    pub fn failure_messages(&self) -> Vec<(usize, String)> {
        self.failed
            .iter()
            .enumerate()
            .filter_map(|(i, f)| f.as_ref().map(|msg| (i, msg.clone())))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(files: Vec<InputFile>) -> SplitInput {
        let total: u64 = files.iter().map(|f| f.size_bytes).sum();
        let dur: Option<f64> = {
            let sum: f64 = files.iter().filter_map(|f| f.duration_secs).sum();
            if sum > 0.0 {
                Some(sum)
            } else {
                None
            }
        };
        SplitInput {
            parent_job_id: JobId::from("test-job"),
            total_size_bytes: total,
            duration_secs: dur,
            files,
            metadata: HashMap::new(),
        }
    }

    fn large_single_file(size: u64) -> SplitInput {
        let tmp_path = std::env::temp_dir()
            .join("oximedia-batch-big.mp4")
            .to_string_lossy()
            .into_owned();
        make_input(vec![InputFile::new(tmp_path, size)])
    }

    fn large_single_file_with_duration(size: u64, dur: f64) -> SplitInput {
        let tmp_path = std::env::temp_dir()
            .join("oximedia-batch-big.mp4")
            .to_string_lossy()
            .into_owned();
        make_input(vec![InputFile::new(tmp_path, size).with_duration(dur)])
    }

    // ── Strategy display ────────────────────────────────────────────────
    #[test]
    fn test_split_strategy_display() {
        assert_eq!(
            SplitStrategy::BySize { chunk_bytes: 1024 }.to_string(),
            "by_size(1024 bytes)"
        );
        assert_eq!(
            SplitStrategy::ByDuration { chunk_secs: 60.0 }.to_string(),
            "by_duration(60s)"
        );
        assert_eq!(SplitStrategy::ByCount { n: 4 }.to_string(), "by_count(4)");
    }

    #[test]
    fn test_merge_strategy_display() {
        assert_eq!(MergeStrategy::Concatenate.to_string(), "concatenate");
        assert_eq!(MergeStrategy::Passthrough.to_string(), "passthrough");
        assert_eq!(MergeStrategy::MuxStreams.to_string(), "mux_streams");
    }

    // ── No split needed ─────────────────────────────────────────────────
    #[test]
    fn test_no_split_below_threshold() {
        let splitter = JobSplitter::new().with_min_threshold(100 * 1024 * 1024);
        let input = large_single_file(50 * 1024 * 1024); // 50 MiB
        let plan = splitter.plan(&input).expect("plan should succeed");
        assert!(!plan.was_split);
        assert_eq!(plan.chunk_count(), 1);
    }

    // ── Empty input ─────────────────────────────────────────────────────
    #[test]
    fn test_empty_input_returns_error() {
        let splitter = JobSplitter::new();
        let input = make_input(vec![]);
        let result = splitter.plan(&input);
        assert!(result.is_err());
    }

    // ── Split by size (single file) ─────────────────────────────────────
    #[test]
    fn test_split_by_size_single_file() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let input = large_single_file(100 * 1024 * 1024); // 100 MiB
        let strategy = SplitStrategy::BySize {
            chunk_bytes: 25 * 1024 * 1024, // 25 MiB chunks
        };
        let plan = splitter
            .plan_with_strategy(&input, &strategy)
            .expect("plan should succeed");
        assert!(plan.was_split);
        assert_eq!(plan.chunk_count(), 4);
        // Each chunk should have a byte range.
        for chunk in &plan.chunks {
            assert!(chunk.byte_range.is_some());
        }
        // Byte ranges should be contiguous.
        let ranges: Vec<(u64, u64)> = plan.chunks.iter().filter_map(|c| c.byte_range).collect();
        for i in 1..ranges.len() {
            assert_eq!(ranges[i].0, ranges[i - 1].1);
        }
    }

    // ── Split by size (multiple files) ──────────────────────────────────
    #[test]
    fn test_split_by_size_multiple_files() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let tmp_base = std::env::temp_dir();
        let files: Vec<InputFile> = (0..10)
            .map(|i| {
                InputFile::new(
                    tmp_base
                        .join(format!("oximedia-batch-file{i}.mp4"))
                        .to_string_lossy()
                        .into_owned(),
                    10 * 1024 * 1024,
                )
            })
            .collect();
        let input = make_input(files);
        let strategy = SplitStrategy::BySize {
            chunk_bytes: 25 * 1024 * 1024,
        };
        let plan = splitter
            .plan_with_strategy(&input, &strategy)
            .expect("plan should succeed");
        assert!(plan.was_split);
        // Each chunk should have some files.
        for chunk in &plan.chunks {
            assert!(!chunk.files.is_empty());
        }
    }

    // ── Split by duration ───────────────────────────────────────────────
    #[test]
    fn test_split_by_duration() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let input = large_single_file_with_duration(500 * 1024 * 1024, 600.0); // 10 min
        let strategy = SplitStrategy::ByDuration { chunk_secs: 120.0 }; // 2 min chunks
        let plan = splitter
            .plan_with_strategy(&input, &strategy)
            .expect("plan should succeed");
        assert!(plan.was_split);
        assert_eq!(plan.chunk_count(), 5);
        // Each chunk should have a time range.
        for chunk in &plan.chunks {
            assert!(chunk.time_range.is_some());
            assert!(chunk.estimated_duration_secs.is_some());
        }
    }

    #[test]
    fn test_split_by_duration_no_duration_returns_error() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let input = large_single_file(500 * 1024 * 1024);
        let strategy = SplitStrategy::ByDuration { chunk_secs: 120.0 };
        let result = splitter.plan_with_strategy(&input, &strategy);
        assert!(result.is_err());
    }

    #[test]
    fn test_split_by_duration_zero_secs_returns_error() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let input = large_single_file_with_duration(500 * 1024 * 1024, 600.0);
        let strategy = SplitStrategy::ByDuration { chunk_secs: 0.0 };
        let result = splitter.plan_with_strategy(&input, &strategy);
        assert!(result.is_err());
    }

    // ── Split by count ──────────────────────────────────────────────────
    #[test]
    fn test_split_by_count() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let tmp_base = std::env::temp_dir();
        let files: Vec<InputFile> = (0..12)
            .map(|i| {
                InputFile::new(
                    tmp_base
                        .join(format!("oximedia-batch-f{i}.mp4"))
                        .to_string_lossy()
                        .into_owned(),
                    10 * 1024 * 1024,
                )
            })
            .collect();
        let input = make_input(files);
        let strategy = SplitStrategy::ByCount { n: 4 };
        let plan = splitter
            .plan_with_strategy(&input, &strategy)
            .expect("plan should succeed");
        assert!(plan.was_split);
        assert_eq!(plan.chunk_count(), 4);
        // All files should be distributed.
        let total_files: usize = plan.chunks.iter().map(|c| c.files.len()).sum();
        assert_eq!(total_files, 12);
    }

    #[test]
    fn test_split_by_count_more_chunks_than_files() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let tmp_base = std::env::temp_dir();
        let files = vec![
            InputFile::new(
                tmp_base
                    .join("oximedia-batch-a.mp4")
                    .to_string_lossy()
                    .into_owned(),
                1024,
            ),
            InputFile::new(
                tmp_base
                    .join("oximedia-batch-b.mp4")
                    .to_string_lossy()
                    .into_owned(),
                1024,
            ),
        ];
        let input = make_input(files);
        let strategy = SplitStrategy::ByCount { n: 10 };
        let plan = splitter
            .plan_with_strategy(&input, &strategy)
            .expect("plan should succeed");
        // Should clamp to file count.
        assert!(plan.chunk_count() <= 2);
    }

    // ── Adaptive ────────────────────────────────────────────────────────
    #[test]
    fn test_adaptive_prefers_duration_when_available() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let input = large_single_file_with_duration(500 * 1024 * 1024, 600.0);
        let plan = splitter.plan(&input).expect("plan should succeed");
        assert!(plan.was_split);
        // Should use duration-based splitting.
        for chunk in &plan.chunks {
            assert!(chunk.time_range.is_some());
        }
    }

    #[test]
    fn test_adaptive_falls_back_to_size() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let input = large_single_file(500 * 1024 * 1024);
        let plan = splitter.plan(&input).expect("plan should succeed");
        assert!(plan.was_split);
        // Should use size-based splitting.
        for chunk in &plan.chunks {
            assert!(chunk.byte_range.is_some());
        }
    }

    // ── Builder ─────────────────────────────────────────────────────────
    #[test]
    fn test_splitter_builder() {
        let splitter = JobSplitter::new()
            .with_strategy(SplitStrategy::ByCount { n: 8 })
            .with_merge(MergeStrategy::Passthrough)
            .with_min_threshold(1024)
            .with_max_chunks(32);
        assert_eq!(splitter.max_chunks, 32);
        assert_eq!(splitter.min_split_threshold_bytes, 1024);
    }

    // ── Chunk progress ──────────────────────────────────────────────────
    #[test]
    fn test_chunk_progress_pct() {
        let chunk = JobChunk {
            chunk_id: "test-chunk-2".to_string(),
            parent_job_id: JobId::from("parent"),
            index: 2,
            total_chunks: 4,
            files: vec![],
            byte_range: None,
            time_range: None,
            estimated_size_bytes: 0,
            estimated_duration_secs: None,
            metadata: HashMap::new(),
        };
        assert!((chunk.progress_pct() - 75.0).abs() < f64::EPSILON);
    }

    // ── SplitPlan helpers ───────────────────────────────────────────────
    #[test]
    fn test_split_plan_total_bytes() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let input = large_single_file(100 * 1024 * 1024);
        let plan = splitter
            .plan_with_strategy(
                &input,
                &SplitStrategy::BySize {
                    chunk_bytes: 25 * 1024 * 1024,
                },
            )
            .expect("plan should succeed");
        assert_eq!(plan.total_estimated_bytes(), 100 * 1024 * 1024);
    }

    // ── Merge tracker ───────────────────────────────────────────────────
    #[test]
    fn test_merge_tracker_basic() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let input = large_single_file(100 * 1024 * 1024);
        let plan = splitter
            .plan_with_strategy(
                &input,
                &SplitStrategy::BySize {
                    chunk_bytes: 25 * 1024 * 1024,
                },
            )
            .expect("plan should succeed");

        let mut tracker = MergeTracker::from_plan(&plan);
        assert!(!tracker.is_ready_for_merge());
        assert_eq!(tracker.completed_count(), 0);
        assert!((tracker.progress_pct()).abs() < f64::EPSILON);

        tracker.mark_completed(0).expect("should succeed");
        tracker.mark_completed(1).expect("should succeed");
        assert_eq!(tracker.completed_count(), 2);
        assert!(!tracker.is_ready_for_merge());

        tracker.mark_completed(2).expect("should succeed");
        tracker.mark_completed(3).expect("should succeed");
        assert!(tracker.is_ready_for_merge());
        assert!((tracker.progress_pct() - 100.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_merge_tracker_failure() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let input = large_single_file(100 * 1024 * 1024);
        let plan = splitter
            .plan_with_strategy(
                &input,
                &SplitStrategy::BySize {
                    chunk_bytes: 50 * 1024 * 1024,
                },
            )
            .expect("plan should succeed");

        let mut tracker = MergeTracker::from_plan(&plan);
        tracker.mark_completed(0).expect("should succeed");
        tracker.mark_failed(1, "disk full").expect("should succeed");
        assert!(tracker.has_failures());
        assert!(!tracker.is_ready_for_merge());
        assert_eq!(tracker.failed_count(), 1);
        let failures = tracker.failure_messages();
        assert_eq!(failures.len(), 1);
        assert_eq!(failures[0].0, 1);
        assert_eq!(failures[0].1, "disk full");
    }

    #[test]
    fn test_merge_tracker_out_of_bounds() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let input = large_single_file(100 * 1024 * 1024);
        let plan = splitter
            .plan_with_strategy(
                &input,
                &SplitStrategy::BySize {
                    chunk_bytes: 50 * 1024 * 1024,
                },
            )
            .expect("plan should succeed");

        let mut tracker = MergeTracker::from_plan(&plan);
        assert!(tracker.mark_completed(99).is_err());
        assert!(tracker.mark_failed(99, "err").is_err());
    }

    // ── InputFile builder ───────────────────────────────────────────────
    #[test]
    fn test_input_file_with_duration() {
        let tmp_path = std::env::temp_dir()
            .join("oximedia-batch-test.mp4")
            .to_string_lossy()
            .into_owned();
        let f = InputFile::new(tmp_path, 1024).with_duration(120.5);
        assert_eq!(f.duration_secs, Some(120.5));
    }

    // ── Max chunks limit ────────────────────────────────────────────────
    #[test]
    fn test_max_chunks_respected() {
        let splitter = JobSplitter::new().with_min_threshold(0).with_max_chunks(3);
        let input = large_single_file(1000 * 1024 * 1024); // 1 GiB
        let plan = splitter
            .plan_with_strategy(
                &input,
                &SplitStrategy::BySize {
                    chunk_bytes: 10 * 1024 * 1024, // Would produce 100 chunks
                },
            )
            .expect("plan should succeed");
        assert!(plan.chunk_count() <= 3);
    }

    // ── Default splitter ────────────────────────────────────────────────
    #[test]
    fn test_default_splitter() {
        let splitter = JobSplitter::default();
        assert_eq!(splitter.max_chunks, 64);
    }

    // ── Metadata propagation ────────────────────────────────────────────
    #[test]
    fn test_metadata_propagated_to_chunks() {
        let splitter = JobSplitter::new().with_min_threshold(0);
        let mut input = large_single_file(100 * 1024 * 1024);
        input.metadata.insert("project".into(), "test".into());
        let plan = splitter
            .plan_with_strategy(
                &input,
                &SplitStrategy::BySize {
                    chunk_bytes: 50 * 1024 * 1024,
                },
            )
            .expect("plan should succeed");
        for chunk in &plan.chunks {
            assert_eq!(
                chunk.metadata.get("project").map(|s| s.as_str()),
                Some("test")
            );
        }
    }
}
