//! Batch RPU (Reference Processing Unit) frame processor.
//!
//! Provides facilities for processing sequences of Dolby Vision frames in bulk:
//! accumulating per-frame statistics, applying error-recovery policies, and
//! reporting progress via callbacks.
//!
//! # Design
//!
//! [`BatchRpuProcessor`] owns a configurable [`BatchConfig`] and processes
//! frames one-by-one through [`BatchRpuProcessor::process_frame`].  After all
//! frames have been submitted, call [`BatchRpuProcessor::finish`] to obtain a
//! [`BatchProcessingResult`] with aggregate statistics and any per-frame errors.
//!
//! Error-recovery behaviour is controlled by [`ErrorPolicy`]:
//! - [`ErrorPolicy::Abort`] — stop at the first error (returns early with an
//!   overall `Err`).
//! - [`ErrorPolicy::Skip`] — record the error and continue processing.
//! - [`ErrorPolicy::UseDefault`] — substitute a default RPU on error and
//!   continue.

use crate::{DolbyVisionError, DolbyVisionRpu, Level1Metadata, Profile, Result};

// ── ErrorPolicy ───────────────────────────────────────────────────────────────

/// Policy that governs how [`BatchRpuProcessor`] reacts to per-frame errors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ErrorPolicy {
    /// Stop processing immediately and propagate the error.
    Abort,
    /// Record the error for the frame and continue with the next frame.
    #[default]
    Skip,
    /// Substitute a profile-default RPU for the erroneous frame and continue.
    UseDefault,
}

// ── BatchConfig ───────────────────────────────────────────────────────────────

/// Configuration for a [`BatchRpuProcessor`].
#[derive(Debug, Clone)]
pub struct BatchConfig {
    /// Fallback profile used when [`ErrorPolicy::UseDefault`] is active.
    pub default_profile: Profile,
    /// How errors in individual frames are handled.
    pub error_policy: ErrorPolicy,
    /// Maximum number of errors to tolerate before aborting, regardless of
    /// `error_policy`.  `None` means unlimited.
    pub max_errors: Option<usize>,
}

impl BatchConfig {
    /// Create a configuration with the given default profile and skip-on-error
    /// policy.
    #[must_use]
    pub fn new(default_profile: Profile) -> Self {
        Self {
            default_profile,
            error_policy: ErrorPolicy::Skip,
            max_errors: None,
        }
    }

    /// Set the error policy.
    #[must_use]
    pub fn with_error_policy(mut self, policy: ErrorPolicy) -> Self {
        self.error_policy = policy;
        self
    }

    /// Set the maximum number of tolerated errors.
    #[must_use]
    pub fn with_max_errors(mut self, max: usize) -> Self {
        self.max_errors = Some(max);
        self
    }
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self::new(Profile::Profile8)
    }
}

// ── FrameError ────────────────────────────────────────────────────────────────

/// A per-frame error recorded during batch processing.
#[derive(Debug, Clone)]
pub struct FrameError {
    /// 0-based frame index at which the error occurred.
    pub frame_index: u64,
    /// Human-readable description of the error.
    pub message: String,
}

// ── BatchStats ────────────────────────────────────────────────────────────────

/// Aggregate statistics accumulated during batch processing.
#[derive(Debug, Clone, Default)]
pub struct BatchStats {
    /// Total number of frames submitted.
    pub total_frames: u64,
    /// Number of frames processed successfully.
    pub ok_frames: u64,
    /// Number of frames that produced an error (skipped or substituted).
    pub error_frames: u64,
    /// Number of frames for which a default RPU was substituted.
    pub substituted_frames: u64,
    /// Running sum of Level 1 max-PQ values (used to compute `avg_max_pq`).
    sum_max_pq: f64,
    /// Number of frames that contributed to `sum_max_pq`.
    pq_sample_count: u64,
    /// Global maximum PQ seen across all valid frames.
    pub global_max_pq: u16,
    /// Global minimum PQ seen across all valid frames (initialised to u16::MAX).
    pub global_min_pq: u16,
}

impl BatchStats {
    fn new() -> Self {
        Self {
            global_min_pq: u16::MAX,
            ..Default::default()
        }
    }

    fn record_level1(&mut self, level1: &Level1Metadata) {
        self.sum_max_pq += f64::from(level1.max_pq);
        self.pq_sample_count += 1;
        if level1.max_pq > self.global_max_pq {
            self.global_max_pq = level1.max_pq;
        }
        if level1.min_pq < self.global_min_pq {
            self.global_min_pq = level1.min_pq;
        }
    }

    /// Average max-PQ across all frames that had Level 1 metadata.
    ///
    /// Returns `0.0` when no Level 1 samples have been recorded.
    #[must_use]
    pub fn avg_max_pq(&self) -> f64 {
        if self.pq_sample_count == 0 {
            return 0.0;
        }
        self.sum_max_pq / self.pq_sample_count as f64
    }

    /// Fraction of frames that processed without error, in `[0.0, 1.0]`.
    ///
    /// Returns `1.0` when no frames have been submitted.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total_frames == 0 {
            return 1.0;
        }
        self.ok_frames as f64 / self.total_frames as f64
    }
}

// ── BatchProcessingResult ─────────────────────────────────────────────────────

/// Result returned by [`BatchRpuProcessor::finish`].
#[derive(Debug, Clone)]
pub struct BatchProcessingResult {
    /// Aggregate statistics for the entire batch.
    pub stats: BatchStats,
    /// Per-frame errors recorded during processing (empty when all frames OK).
    pub errors: Vec<FrameError>,
    /// Processed RPU frames in submission order.
    pub frames: Vec<DolbyVisionRpu>,
}

impl BatchProcessingResult {
    /// Returns `true` when every submitted frame processed without error.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.errors.is_empty()
    }

    /// Number of per-frame errors.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

// ── BatchRpuProcessor ─────────────────────────────────────────────────────────

/// Batch processor for sequences of Dolby Vision RPU frames.
///
/// # Example
///
/// ```rust
/// use oximedia_dolbyvision::batch_rpu::{BatchConfig, BatchRpuProcessor, ErrorPolicy};
/// use oximedia_dolbyvision::{DolbyVisionRpu, Profile};
///
/// let config = BatchConfig::new(Profile::Profile8)
///     .with_error_policy(ErrorPolicy::Skip);
/// let mut processor = BatchRpuProcessor::new(config);
///
/// let rpu = DolbyVisionRpu::new(Profile::Profile8);
/// processor.process_frame(0, Ok(rpu)).unwrap();
///
/// let result = processor.finish();
/// assert_eq!(result.stats.total_frames, 1);
/// assert!(result.is_clean());
/// ```
#[derive(Debug)]
pub struct BatchRpuProcessor {
    config: BatchConfig,
    stats: BatchStats,
    errors: Vec<FrameError>,
    frames: Vec<DolbyVisionRpu>,
}

impl BatchRpuProcessor {
    /// Create a new processor with the given configuration.
    #[must_use]
    pub fn new(config: BatchConfig) -> Self {
        Self {
            config,
            stats: BatchStats::new(),
            errors: Vec::new(),
            frames: Vec::new(),
        }
    }

    /// Submit a frame result for processing.
    ///
    /// - `frame_index` — the 0-based index of the frame in the source sequence.
    /// - `frame` — either an already-parsed `DolbyVisionRpu` or the error that
    ///   occurred while parsing/producing it.
    ///
    /// # Errors
    ///
    /// Returns an error only when `config.error_policy == ErrorPolicy::Abort`
    /// and `frame` is `Err`, or when the error limit (`max_errors`) is reached.
    pub fn process_frame(
        &mut self,
        frame_index: u64,
        frame: Result<DolbyVisionRpu>,
    ) -> Result<()> {
        self.stats.total_frames += 1;

        match frame {
            Ok(rpu) => {
                if let Some(ref l1) = rpu.level1 {
                    self.stats.record_level1(l1);
                }
                self.stats.ok_frames += 1;
                self.frames.push(rpu);
                Ok(())
            }
            Err(err) => {
                let msg = err.to_string();
                self.stats.error_frames += 1;

                // Check global error limit
                if let Some(max) = self.config.max_errors {
                    if self.errors.len() >= max {
                        return Err(DolbyVisionError::Generic(format!(
                            "Error limit ({max}) exceeded at frame {frame_index}: {msg}"
                        )));
                    }
                }

                match self.config.error_policy {
                    ErrorPolicy::Abort => {
                        Err(DolbyVisionError::Generic(format!(
                            "Frame {frame_index} failed: {msg}"
                        )))
                    }
                    ErrorPolicy::Skip => {
                        self.errors.push(FrameError {
                            frame_index,
                            message: msg,
                        });
                        Ok(())
                    }
                    ErrorPolicy::UseDefault => {
                        self.errors.push(FrameError {
                            frame_index,
                            message: msg,
                        });
                        let default_rpu = DolbyVisionRpu::new(self.config.default_profile);
                        self.stats.substituted_frames += 1;
                        self.frames.push(default_rpu);
                        Ok(())
                    }
                }
            }
        }
    }

    /// Finish processing and return the aggregated result.
    ///
    /// This method consumes the processor.
    #[must_use]
    pub fn finish(self) -> BatchProcessingResult {
        BatchProcessingResult {
            stats: self.stats,
            errors: self.errors,
            frames: self.frames,
        }
    }

    /// Number of frames successfully accumulated so far.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Number of errors recorded so far.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.errors.len()
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Level1Metadata;

    fn make_rpu_with_level1(max_pq: u16, min_pq: u16) -> DolbyVisionRpu {
        let mut rpu = DolbyVisionRpu::new(Profile::Profile8);
        rpu.level1 = Some(Level1Metadata {
            min_pq,
            max_pq,
            avg_pq: (max_pq / 2).max(min_pq),
        });
        rpu
    }

    fn make_error() -> Result<DolbyVisionRpu> {
        Err(DolbyVisionError::Generic("parse error".to_string()))
    }

    #[test]
    fn test_batch_process_all_ok() {
        let mut proc = BatchRpuProcessor::new(BatchConfig::default());
        for i in 0..5u64 {
            proc.process_frame(i, Ok(DolbyVisionRpu::new(Profile::Profile8)))
                .unwrap();
        }
        let result = proc.finish();
        assert_eq!(result.stats.total_frames, 5);
        assert_eq!(result.stats.ok_frames, 5);
        assert_eq!(result.stats.error_frames, 0);
        assert!(result.is_clean());
    }

    #[test]
    fn test_batch_skip_errors() {
        let config = BatchConfig::new(Profile::Profile8).with_error_policy(ErrorPolicy::Skip);
        let mut proc = BatchRpuProcessor::new(config);
        proc.process_frame(0, Ok(DolbyVisionRpu::new(Profile::Profile8)))
            .unwrap();
        proc.process_frame(1, make_error()).unwrap();
        proc.process_frame(2, Ok(DolbyVisionRpu::new(Profile::Profile8)))
            .unwrap();
        let result = proc.finish();
        assert_eq!(result.stats.total_frames, 3);
        assert_eq!(result.stats.ok_frames, 2);
        assert_eq!(result.stats.error_frames, 1);
        assert_eq!(result.error_count(), 1);
        assert_eq!(result.frames.len(), 2);
    }

    #[test]
    fn test_batch_use_default_substitution() {
        let config =
            BatchConfig::new(Profile::Profile8).with_error_policy(ErrorPolicy::UseDefault);
        let mut proc = BatchRpuProcessor::new(config);
        proc.process_frame(0, make_error()).unwrap();
        let result = proc.finish();
        assert_eq!(result.stats.substituted_frames, 1);
        assert_eq!(result.frames.len(), 1);
        assert_eq!(result.frames[0].profile, Profile::Profile8);
    }

    #[test]
    fn test_batch_abort_on_error() {
        let config = BatchConfig::new(Profile::Profile8).with_error_policy(ErrorPolicy::Abort);
        let mut proc = BatchRpuProcessor::new(config);
        let err = proc.process_frame(0, make_error());
        assert!(err.is_err());
    }

    #[test]
    fn test_batch_max_errors_limit() {
        let config = BatchConfig::new(Profile::Profile8)
            .with_error_policy(ErrorPolicy::Skip)
            .with_max_errors(1);
        let mut proc = BatchRpuProcessor::new(config);
        proc.process_frame(0, make_error()).unwrap(); // first error: OK
        let err = proc.process_frame(1, make_error()); // second: exceeds limit
        assert!(err.is_err());
    }

    #[test]
    fn test_batch_level1_stats_accumulation() {
        let mut proc = BatchRpuProcessor::new(BatchConfig::default());
        proc.process_frame(0, Ok(make_rpu_with_level1(3000, 100)))
            .unwrap();
        proc.process_frame(1, Ok(make_rpu_with_level1(1000, 50)))
            .unwrap();
        let result = proc.finish();
        assert_eq!(result.stats.global_max_pq, 3000);
        assert_eq!(result.stats.global_min_pq, 50);
        let avg = result.stats.avg_max_pq();
        assert!((avg - 2000.0).abs() < 1.0, "avg={avg}");
    }

    #[test]
    fn test_batch_success_rate_all_ok() {
        let mut proc = BatchRpuProcessor::new(BatchConfig::default());
        for i in 0..4u64 {
            proc.process_frame(i, Ok(DolbyVisionRpu::new(Profile::Profile8)))
                .unwrap();
        }
        let result = proc.finish();
        assert!((result.stats.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_success_rate_with_errors() {
        let config = BatchConfig::new(Profile::Profile8).with_error_policy(ErrorPolicy::Skip);
        let mut proc = BatchRpuProcessor::new(config);
        proc.process_frame(0, Ok(DolbyVisionRpu::new(Profile::Profile8)))
            .unwrap();
        proc.process_frame(1, make_error()).unwrap();
        let result = proc.finish();
        assert!((result.stats.success_rate() - 0.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_empty_success_rate_is_one() {
        let proc = BatchRpuProcessor::new(BatchConfig::default());
        let result = proc.finish();
        assert!((result.stats.success_rate() - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_batch_frame_count_query() {
        let mut proc = BatchRpuProcessor::new(BatchConfig::default());
        assert_eq!(proc.frame_count(), 0);
        proc.process_frame(0, Ok(DolbyVisionRpu::new(Profile::Profile8)))
            .unwrap();
        assert_eq!(proc.frame_count(), 1);
    }

    #[test]
    fn test_error_policy_default_is_skip() {
        assert_eq!(ErrorPolicy::default(), ErrorPolicy::Skip);
    }

    #[test]
    fn test_batch_config_builder() {
        let cfg = BatchConfig::new(Profile::Profile5)
            .with_error_policy(ErrorPolicy::Abort)
            .with_max_errors(10);
        assert_eq!(cfg.default_profile, Profile::Profile5);
        assert_eq!(cfg.error_policy, ErrorPolicy::Abort);
        assert_eq!(cfg.max_errors, Some(10));
    }

    #[test]
    fn test_batch_result_is_clean_true() {
        let mut proc = BatchRpuProcessor::new(BatchConfig::default());
        proc.process_frame(0, Ok(DolbyVisionRpu::new(Profile::Profile8)))
            .unwrap();
        let result = proc.finish();
        assert!(result.is_clean());
    }

    #[test]
    fn test_batch_result_error_count() {
        let config = BatchConfig::new(Profile::Profile8).with_error_policy(ErrorPolicy::Skip);
        let mut proc = BatchRpuProcessor::new(config);
        proc.process_frame(0, make_error()).unwrap();
        proc.process_frame(1, make_error()).unwrap();
        let result = proc.finish();
        assert_eq!(result.error_count(), 2);
    }
}
