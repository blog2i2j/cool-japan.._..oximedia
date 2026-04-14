//! Batch restoration of multiple audio buffers using Rayon parallel iterators.
//!
//! The [`BatchRestorer`] allows you to apply a single shared [`RestoreChain`]
//! configuration to many independent audio buffers simultaneously, exploiting
//! all available CPU cores via Rayon.
//!
//! Because [`RestoreChain`] contains mutable stateful processors, each worker
//! thread works with its **own clone** of the chain built from a
//! [`BatchRestorationConfig`] factory.  This preserves thread safety while
//! ensuring every buffer is processed with the same settings.
//!
//! # Example
//!
//! ```
//! use oximedia_restore::batch::{BatchRestorer, BatchJob, BatchRestorationConfig};
//! use oximedia_restore::{RestoreChain, RestorationStep};
//! use oximedia_restore::dc::DcRemover;
//!
//! // Define how to build a fresh chain for each job
//! let config = BatchRestorationConfig::new(|| {
//!     let mut chain = RestoreChain::new();
//!     chain.add_step(RestorationStep::DcRemoval(DcRemover::new(10.0, 44100)));
//!     chain
//! });
//!
//! let restorer = BatchRestorer::new(config);
//!
//! let jobs: Vec<BatchJob> = vec![
//!     BatchJob { id: 0, samples: vec![0.1f32; 1024], sample_rate: 44100 },
//!     BatchJob { id: 1, samples: vec![-0.05f32; 512], sample_rate: 48000 },
//! ];
//!
//! let results = restorer.process(jobs).expect("batch should succeed");
//! assert_eq!(results.len(), 2);
//! ```

use crate::error::{RestoreError, RestoreResult};
use crate::RestoreChain;
use rayon::prelude::*;

// ---------------------------------------------------------------------------
// BatchJob
// ---------------------------------------------------------------------------

/// A single unit of work in a batch restoration run.
#[derive(Debug, Clone)]
pub struct BatchJob {
    /// Caller-assigned identifier (preserved in the result for correlation).
    pub id: usize,
    /// Mono audio samples.
    pub samples: Vec<f32>,
    /// Sample rate in Hz.
    pub sample_rate: u32,
}

// ---------------------------------------------------------------------------
// BatchResult
// ---------------------------------------------------------------------------

/// The result of processing a single [`BatchJob`].
#[derive(Debug)]
pub struct BatchResult {
    /// The `id` from the originating [`BatchJob`].
    pub id: usize,
    /// Restored audio samples, or an error description if the job failed.
    pub output: Result<Vec<f32>, String>,
}

impl BatchResult {
    /// Returns `true` if the job completed successfully.
    #[must_use]
    pub fn is_ok(&self) -> bool {
        self.output.is_ok()
    }

    /// Borrow the restored samples.
    ///
    /// Returns `None` if the job failed.
    #[must_use]
    pub fn samples(&self) -> Option<&[f32]> {
        self.output.as_deref().ok()
    }
}

// ---------------------------------------------------------------------------
// BatchRestorationConfig
// ---------------------------------------------------------------------------

/// Factory that constructs a fresh [`RestoreChain`] for each parallel worker.
///
/// The factory closure is called once per Rayon thread that participates in
/// the batch, not necessarily once per job.
pub struct BatchRestorationConfig {
    /// Closure returning a ready-to-use [`RestoreChain`].
    factory: Box<dyn Fn() -> RestoreChain + Send + Sync>,
}

impl BatchRestorationConfig {
    /// Create a new configuration with the given chain factory.
    ///
    /// `factory` must be `Send + Sync` (a plain closure capturing only
    /// clonable or `Send + Sync` state qualifies automatically).
    pub fn new<F>(factory: F) -> Self
    where
        F: Fn() -> RestoreChain + Send + Sync + 'static,
    {
        Self {
            factory: Box::new(factory),
        }
    }

    /// Build a fresh chain using the factory.
    pub(crate) fn build(&self) -> RestoreChain {
        (self.factory)()
    }
}

impl std::fmt::Debug for BatchRestorationConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchRestorationConfig")
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// BatchRestorer
// ---------------------------------------------------------------------------

/// Parallel batch audio restorer.
///
/// Uses Rayon to process multiple [`BatchJob`]s in parallel.  Each job runs
/// in a Rayon worker thread with its own clone of the restoration chain.
///
/// Jobs that fail are represented in the output as [`BatchResult`] entries
/// whose `output` field contains an error string rather than panicking the
/// thread.
#[derive(Debug)]
pub struct BatchRestorer {
    config: BatchRestorationConfig,
}

impl BatchRestorer {
    /// Create a new batch restorer using the given configuration.
    #[must_use]
    pub fn new(config: BatchRestorationConfig) -> Self {
        Self { config }
    }

    /// Process all jobs in parallel and collect results.
    ///
    /// The output [`Vec`] is in the **same order** as the input `jobs` slice.
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidParameter`] if `jobs` is empty.
    /// Individual job failures are captured in [`BatchResult::output`] rather
    /// than propagated as a top-level error, so the call succeeds even if some
    /// jobs fail.
    pub fn process(&self, jobs: Vec<BatchJob>) -> RestoreResult<Vec<BatchResult>> {
        if jobs.is_empty() {
            return Err(RestoreError::InvalidParameter(
                "batch jobs list must not be empty".to_owned(),
            ));
        }

        let results: Vec<BatchResult> = jobs
            .into_par_iter()
            .map(|job| {
                let mut chain = self.config.build();
                let output = chain
                    .process(&job.samples, job.sample_rate)
                    .map_err(|e| e.to_string());
                BatchResult { id: job.id, output }
            })
            .collect();

        Ok(results)
    }

    /// Process all jobs in parallel with an optional per-job **progress callback**.
    ///
    /// `on_complete` is called on the Rayon worker thread immediately after each
    /// job finishes.  It receives the job `id` and whether the job succeeded.
    ///
    /// # Errors
    ///
    /// Same as [`process`](Self::process).
    pub fn process_with_progress<F>(
        &self,
        jobs: Vec<BatchJob>,
        on_complete: F,
    ) -> RestoreResult<Vec<BatchResult>>
    where
        F: Fn(usize, bool) + Send + Sync,
    {
        if jobs.is_empty() {
            return Err(RestoreError::InvalidParameter(
                "batch jobs list must not be empty".to_owned(),
            ));
        }

        let results: Vec<BatchResult> = jobs
            .into_par_iter()
            .map(|job| {
                let mut chain = self.config.build();
                let output = chain
                    .process(&job.samples, job.sample_rate)
                    .map_err(|e| e.to_string());
                let succeeded = output.is_ok();
                let result = BatchResult { id: job.id, output };
                on_complete(result.id, succeeded);
                result
            })
            .collect();

        Ok(results)
    }

    /// Process stereo jobs in parallel.
    ///
    /// Each job is a tuple `(id, left, right, sample_rate)`.  Results are
    /// returned as `(id, Result<(left_out, right_out), String>)` in input
    /// order.
    pub fn process_stereo(
        &self,
        jobs: Vec<(usize, Vec<f32>, Vec<f32>, u32)>,
    ) -> RestoreResult<Vec<(usize, Result<(Vec<f32>, Vec<f32>), String>)>> {
        if jobs.is_empty() {
            return Err(RestoreError::InvalidParameter(
                "stereo batch jobs list must not be empty".to_owned(),
            ));
        }

        let results = jobs
            .into_par_iter()
            .map(|(id, left, right, sr)| {
                let mut chain = self.config.build();
                let output = chain
                    .process_stereo(&left, &right, sr)
                    .map_err(|e| e.to_string());
                (id, output)
            })
            .collect();

        Ok(results)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dc::DcRemover;
    use crate::RestorationStep;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    fn make_dc_chain() -> RestoreChain {
        let mut chain = RestoreChain::new();
        chain.add_step(RestorationStep::DcRemoval(DcRemover::new(10.0, 44100)));
        chain
    }

    fn make_restorer() -> BatchRestorer {
        BatchRestorer::new(BatchRestorationConfig::new(make_dc_chain))
    }

    #[test]
    fn test_batch_empty_jobs_returns_error() {
        let restorer = make_restorer();
        let result = restorer.process(vec![]);
        assert!(result.is_err(), "empty job list should return an error");
    }

    #[test]
    fn test_batch_single_job() {
        let restorer = make_restorer();
        let jobs = vec![BatchJob {
            id: 42,
            samples: vec![0.1f32; 512],
            sample_rate: 44100,
        }];
        let results = restorer.process(jobs).expect("single job should succeed");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, 42);
        assert!(results[0].is_ok());
        assert_eq!(results[0].samples().map(|s| s.len()), Some(512));
    }

    #[test]
    fn test_batch_multiple_jobs_output_order() {
        let restorer = make_restorer();
        let jobs: Vec<BatchJob> = (0..8)
            .map(|i| BatchJob {
                id: i,
                samples: vec![0.0f32; 256 + i * 64],
                sample_rate: 44100,
            })
            .collect();

        let results = restorer
            .process(jobs)
            .expect("multi-job batch should succeed");
        assert_eq!(results.len(), 8);

        // Results must be in the same order as the input jobs
        for (i, r) in results.iter().enumerate() {
            assert_eq!(r.id, i, "result id {}, expected {i}", r.id);
            assert!(r.is_ok(), "job {i} should succeed");
        }
    }

    #[test]
    fn test_batch_output_length_matches_input() {
        let restorer = make_restorer();
        let n = 1024usize;
        let jobs = vec![BatchJob {
            id: 0,
            samples: vec![0.5f32; n],
            sample_rate: 44100,
        }];
        let results = restorer.process(jobs).expect("should succeed");
        let out_len = results[0].samples().map(|s| s.len());
        assert_eq!(out_len, Some(n), "output length should match input");
    }

    #[test]
    fn test_batch_with_progress_callback() {
        let completed = Arc::new(AtomicUsize::new(0));
        let completed_clone = Arc::clone(&completed);

        let restorer = make_restorer();
        let jobs: Vec<BatchJob> = (0..4)
            .map(|i| BatchJob {
                id: i,
                samples: vec![0.0f32; 256],
                sample_rate: 44100,
            })
            .collect();

        let results = restorer
            .process_with_progress(jobs, move |_id, success| {
                if success {
                    completed_clone.fetch_add(1, Ordering::Relaxed);
                }
            })
            .expect("progress batch should succeed");

        assert_eq!(results.len(), 4);
        assert_eq!(
            completed.load(Ordering::Relaxed),
            4,
            "all 4 jobs should call on_complete"
        );
    }

    #[test]
    fn test_batch_result_is_ok_and_samples() {
        let restorer = make_restorer();
        let jobs = vec![BatchJob {
            id: 0,
            samples: vec![0.0f32; 64],
            sample_rate: 44100,
        }];
        let results = restorer.process(jobs).expect("should succeed");
        assert!(results[0].is_ok());
        assert!(results[0].samples().is_some());
    }

    #[test]
    fn test_batch_stereo_empty_returns_error() {
        let restorer = make_restorer();
        let result = restorer.process_stereo(vec![]);
        assert!(result.is_err(), "empty stereo job list should error");
    }

    #[test]
    fn test_batch_stereo_single_job() {
        let restorer = make_restorer();
        let left = vec![0.1f32; 512];
        let right = vec![-0.1f32; 512];
        let jobs = vec![(0usize, left, right, 44100u32)];
        let results = restorer
            .process_stereo(jobs)
            .expect("stereo batch should succeed");
        assert_eq!(results.len(), 1);
        let (id, outcome) = &results[0];
        assert_eq!(*id, 0);
        assert!(outcome.is_ok(), "stereo job should succeed");
        let (l, r) = outcome.as_ref().expect("should have output");
        assert_eq!(l.len(), 512);
        assert_eq!(r.len(), 512);
    }

    #[test]
    fn test_batch_different_sample_rates() {
        // Chain is built with 44100 but jobs may use different rates
        let restorer = make_restorer();
        let jobs = vec![
            BatchJob {
                id: 0,
                samples: vec![0.0f32; 256],
                sample_rate: 44100,
            },
            BatchJob {
                id: 1,
                samples: vec![0.0f32; 256],
                sample_rate: 48000,
            },
            BatchJob {
                id: 2,
                samples: vec![0.0f32; 256],
                sample_rate: 96000,
            },
        ];
        let results = restorer
            .process(jobs)
            .expect("mixed sample rates should succeed");
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(r.is_ok(), "job {} should succeed", r.id);
        }
    }
}
