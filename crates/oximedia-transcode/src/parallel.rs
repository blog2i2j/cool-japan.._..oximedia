//! Parallel encoding for multiple outputs simultaneously.

use crate::{Result, TranscodeConfig, TranscodeError, TranscodeOutput};
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Configuration for parallel encoding.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of parallel encodes.
    pub max_parallel: usize,
    /// CPU cores to use per encode.
    pub cores_per_encode: Option<usize>,
    /// Whether to use thread pools.
    pub use_thread_pool: bool,
    /// Priority for parallel jobs.
    pub priority: ParallelPriority,
}

/// Priority levels for parallel jobs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelPriority {
    /// Low priority (background processing).
    Low,
    /// Normal priority.
    Normal,
    /// High priority (time-sensitive).
    High,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_parallel: num_cpus(),
            cores_per_encode: None,
            use_thread_pool: true,
            priority: ParallelPriority::Normal,
        }
    }
}

impl ParallelConfig {
    /// Creates a new parallel config with automatic core detection.
    #[must_use]
    pub fn auto() -> Self {
        Self::default()
    }

    /// Creates a config with a specific number of parallel jobs.
    #[must_use]
    pub fn with_max_parallel(max: usize) -> Self {
        Self {
            max_parallel: max,
            ..Self::default()
        }
    }

    /// Sets the number of cores per encode job.
    #[must_use]
    pub fn cores_per_encode(mut self, cores: usize) -> Self {
        self.cores_per_encode = Some(cores);
        self
    }

    /// Sets the priority level.
    #[must_use]
    pub fn priority(mut self, priority: ParallelPriority) -> Self {
        self.priority = priority;
        self
    }

    /// Validates the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid.
    pub fn validate(&self) -> Result<()> {
        if self.max_parallel == 0 {
            return Err(TranscodeError::ValidationError(
                crate::ValidationError::Unsupported(
                    "max_parallel must be greater than 0".to_string(),
                ),
            ));
        }

        if let Some(cores) = self.cores_per_encode {
            if cores == 0 {
                return Err(TranscodeError::ValidationError(
                    crate::ValidationError::Unsupported(
                        "cores_per_encode must be greater than 0".to_string(),
                    ),
                ));
            }
        }

        Ok(())
    }
}

/// Gets the number of CPU cores available.
///
/// Falls back to 4 if the system query fails.
fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4) // unwrap_or is safe — this is a fallback, not unwrap()
}

/// Parallel encoder for processing multiple outputs simultaneously.
pub struct ParallelEncoder {
    config: ParallelConfig,
    jobs: Vec<TranscodeConfig>,
    results: Arc<Mutex<Vec<Result<TranscodeOutput>>>>,
}

impl ParallelEncoder {
    /// Creates a new parallel encoder with the given configuration.
    #[must_use]
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            jobs: Vec::new(),
            results: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Adds a job to the parallel encoder.
    pub fn add_job(&mut self, job: TranscodeConfig) {
        self.jobs.push(job);
    }

    /// Adds multiple jobs at once.
    pub fn add_jobs(&mut self, jobs: Vec<TranscodeConfig>) {
        self.jobs.extend(jobs);
    }

    /// Gets the number of jobs queued.
    #[must_use]
    pub fn job_count(&self) -> usize {
        self.jobs.len()
    }

    /// Executes all jobs in parallel.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid. Individual job errors
    /// are captured in the results.
    pub async fn execute_all(&mut self) -> Result<Vec<Result<TranscodeOutput>>> {
        self.config.validate()?;

        // Configure thread pool
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.config.max_parallel)
            .build()
            .map_err(|e| {
                TranscodeError::PipelineError(format!("Failed to create thread pool: {e}"))
            })?;

        let jobs = std::mem::take(&mut self.jobs);

        // Execute jobs in parallel and collect results directly.
        let job_results: Vec<Result<TranscodeOutput>> = pool.install(|| {
            jobs.into_par_iter()
                .map(Self::execute_job)
                .collect::<Vec<_>>()
        });

        // Store results for later retrieval.
        match self.results.lock() {
            Ok(mut guard) => {
                guard.extend(job_results.iter().cloned());
            }
            Err(poisoned) => {
                poisoned.into_inner().extend(job_results.iter().cloned());
            }
        }

        Ok(job_results)
    }

    /// Executes all jobs sequentially (for debugging).
    ///
    /// # Errors
    ///
    /// Returns an error if any job fails.
    pub async fn execute_sequential(&mut self) -> Result<Vec<TranscodeOutput>> {
        let mut outputs = Vec::new();

        for job in &self.jobs {
            let output = Self::execute_job(job.clone())?;
            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Executes a single transcode job synchronously.
    ///
    /// Validates the job configuration and then delegates to the
    /// pipeline builder for actual transcoding. The pipeline is
    /// executed on a per-thread tokio runtime so that async I/O
    /// works within the rayon thread pool.
    fn execute_job(job: TranscodeConfig) -> Result<TranscodeOutput> {
        let input = job
            .input
            .as_deref()
            .ok_or_else(|| TranscodeError::InvalidInput("No input file specified".to_string()))?;

        let output = job
            .output
            .as_deref()
            .ok_or_else(|| TranscodeError::InvalidOutput("No output file specified".to_string()))?;

        // Build a pipeline from the job config.
        let mut pipeline_builder = crate::pipeline::TranscodePipelineBuilder::new()
            .input(input)
            .output(output);

        if let Some(ref vc) = job.video_codec {
            pipeline_builder = pipeline_builder.video_codec(vc);
        }
        if let Some(ref ac) = job.audio_codec {
            pipeline_builder = pipeline_builder.audio_codec(ac);
        }
        if let Some(mode) = job.multi_pass {
            pipeline_builder = pipeline_builder.multipass(mode);
        }

        let mut pipeline = pipeline_builder.build()?;

        // Create a per-thread tokio runtime to drive the async pipeline.
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| {
                TranscodeError::PipelineError(format!("Failed to create async runtime: {e}"))
            })?;

        rt.block_on(pipeline.execute())
    }

    /// Gets the results of completed jobs.
    #[must_use]
    pub fn get_results(&self) -> Vec<Result<TranscodeOutput>> {
        match self.results.lock() {
            Ok(guard) => guard.clone(),
            Err(poisoned) => poisoned.into_inner().clone(),
        }
    }

    /// Clears all jobs and results.
    pub fn clear(&mut self) {
        self.jobs.clear();
        match self.results.lock() {
            Ok(mut guard) => guard.clear(),
            Err(poisoned) => poisoned.into_inner().clear(),
        }
    }
}

/// Builder for creating parallel encode jobs.
pub struct ParallelEncodeBuilder {
    config: ParallelConfig,
    jobs: Vec<TranscodeConfig>,
}

impl ParallelEncodeBuilder {
    /// Creates a new parallel encode builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ParallelConfig::default(),
            jobs: Vec::new(),
        }
    }

    /// Sets the maximum number of parallel jobs.
    #[must_use]
    pub fn max_parallel(mut self, max: usize) -> Self {
        self.config.max_parallel = max;
        self
    }

    /// Sets cores per encode job.
    #[must_use]
    pub fn cores_per_encode(mut self, cores: usize) -> Self {
        self.config.cores_per_encode = Some(cores);
        self
    }

    /// Sets the priority level.
    #[must_use]
    pub fn priority(mut self, priority: ParallelPriority) -> Self {
        self.config.priority = priority;
        self
    }

    /// Adds a job to the builder.
    #[must_use]
    pub fn add_job(mut self, job: TranscodeConfig) -> Self {
        self.jobs.push(job);
        self
    }

    /// Adds multiple jobs.
    #[must_use]
    pub fn add_jobs(mut self, jobs: Vec<TranscodeConfig>) -> Self {
        self.jobs.extend(jobs);
        self
    }

    /// Builds the parallel encoder.
    #[must_use]
    pub fn build(self) -> ParallelEncoder {
        let mut encoder = ParallelEncoder::new(self.config);
        encoder.add_jobs(self.jobs);
        encoder
    }
}

impl Default for ParallelEncodeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parallel_config_default() {
        let config = ParallelConfig::default();
        assert!(config.max_parallel > 0);
        assert_eq!(config.priority, ParallelPriority::Normal);
        assert!(config.use_thread_pool);
    }

    #[test]
    fn test_parallel_config_validation() {
        let valid = ParallelConfig::with_max_parallel(4);
        assert!(valid.validate().is_ok());

        let invalid = ParallelConfig {
            max_parallel: 0,
            ..Default::default()
        };
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_parallel_config_cores_validation() {
        let valid = ParallelConfig::default().cores_per_encode(2);
        assert!(valid.validate().is_ok());

        let invalid = ParallelConfig::default().cores_per_encode(0);
        assert!(invalid.validate().is_err());
    }

    #[test]
    fn test_parallel_encoder_job_count() {
        let mut encoder = ParallelEncoder::new(ParallelConfig::default());
        assert_eq!(encoder.job_count(), 0);

        let job = TranscodeConfig {
            input: Some("/tmp/input.mp4".to_string()),
            output: Some("/tmp/output.mp4".to_string()),
            ..Default::default()
        };

        encoder.add_job(job);
        assert_eq!(encoder.job_count(), 1);
    }

    #[test]
    fn test_parallel_encoder_add_jobs() {
        let mut encoder = ParallelEncoder::new(ParallelConfig::default());

        let jobs = vec![
            TranscodeConfig {
                input: Some("/tmp/input1.mp4".to_string()),
                output: Some("/tmp/output1.mp4".to_string()),
                ..Default::default()
            },
            TranscodeConfig {
                input: Some("/tmp/input2.mp4".to_string()),
                output: Some("/tmp/output2.mp4".to_string()),
                ..Default::default()
            },
        ];

        encoder.add_jobs(jobs);
        assert_eq!(encoder.job_count(), 2);
    }

    #[test]
    fn test_parallel_encoder_clear() {
        let mut encoder = ParallelEncoder::new(ParallelConfig::default());

        let job = TranscodeConfig {
            input: Some("/tmp/input.mp4".to_string()),
            output: Some("/tmp/output.mp4".to_string()),
            ..Default::default()
        };

        encoder.add_job(job);
        assert_eq!(encoder.job_count(), 1);

        encoder.clear();
        assert_eq!(encoder.job_count(), 0);
    }

    #[test]
    fn test_parallel_builder() {
        let job = TranscodeConfig {
            input: Some("/tmp/input.mp4".to_string()),
            output: Some("/tmp/output.mp4".to_string()),
            ..Default::default()
        };

        let encoder = ParallelEncodeBuilder::new()
            .max_parallel(4)
            .cores_per_encode(2)
            .priority(ParallelPriority::High)
            .add_job(job)
            .build();

        assert_eq!(encoder.config.max_parallel, 4);
        assert_eq!(encoder.config.cores_per_encode, Some(2));
        assert_eq!(encoder.config.priority, ParallelPriority::High);
        assert_eq!(encoder.job_count(), 1);
    }

    #[test]
    fn test_num_cpus() {
        let cpus = num_cpus();
        assert!(cpus > 0);
        assert!(cpus <= 1024); // Reasonable upper bound
    }
}
