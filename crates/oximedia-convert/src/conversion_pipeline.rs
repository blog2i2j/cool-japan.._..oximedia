//! A composable, sequential conversion pipeline.
//!
//! Build a pipeline of named steps, execute them in order, and collect
//! per-step results into a final `ConversionResult`.

#![allow(dead_code)]

use std::time::{Duration, Instant};

/// A single step in a conversion pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversionStep {
    /// Detect and validate the source format.
    FormatDetection,
    /// Demux streams from the container.
    Demux,
    /// Decode video frames.
    VideoDecode,
    /// Decode audio samples.
    AudioDecode,
    /// Apply video filters (scale, crop, etc.).
    VideoFilter,
    /// Apply audio filters (resample, normalise, etc.).
    AudioFilter,
    /// Encode video to the target codec.
    VideoEncode,
    /// Encode audio to the target codec.
    AudioEncode,
    /// Mux encoded streams into the output container.
    Mux,
    /// Write and finalise the output file.
    Finalise,
    /// A named custom step.
    Custom(String),
}

impl ConversionStep {
    /// Return a human-readable name for this step.
    #[must_use]
    pub fn step_name(&self) -> &str {
        match self {
            Self::FormatDetection => "format_detection",
            Self::Demux => "demux",
            Self::VideoDecode => "video_decode",
            Self::AudioDecode => "audio_decode",
            Self::VideoFilter => "video_filter",
            Self::AudioFilter => "audio_filter",
            Self::VideoEncode => "video_encode",
            Self::AudioEncode => "audio_encode",
            Self::Mux => "mux",
            Self::Finalise => "finalise",
            Self::Custom(name) => name.as_str(),
        }
    }

    /// Returns `true` if this step is a decode step.
    #[must_use]
    pub fn is_decode(&self) -> bool {
        matches!(self, Self::VideoDecode | Self::AudioDecode)
    }

    /// Returns `true` if this step is an encode step.
    #[must_use]
    pub fn is_encode(&self) -> bool {
        matches!(self, Self::VideoEncode | Self::AudioEncode)
    }

    /// Returns `true` if this step is a filter step.
    #[must_use]
    pub fn is_filter(&self) -> bool {
        matches!(self, Self::VideoFilter | Self::AudioFilter)
    }
}

/// Outcome of executing a single pipeline step.
#[derive(Debug, Clone)]
pub struct StepOutcome {
    /// Which step produced this outcome.
    pub step: ConversionStep,
    /// Whether the step succeeded.
    pub success: bool,
    /// Optional error message when `success == false`.
    pub error: Option<String>,
    /// Wall-clock time the step took.
    pub duration: Duration,
}

/// The aggregate result of running a full conversion pipeline.
#[derive(Debug, Clone)]
pub struct ConversionResult {
    /// Individual step outcomes.
    pub steps: Vec<StepOutcome>,
    /// Total wall-clock time for the whole pipeline.
    pub total_duration: Duration,
}

impl ConversionResult {
    /// Fraction of steps that succeeded in [0.0, 1.0].
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.steps.is_empty() {
            return 1.0;
        }
        let ok = self.steps.iter().filter(|s| s.success).count();
        ok as f64 / self.steps.len() as f64
    }

    /// Total elapsed milliseconds for the whole pipeline.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn elapsed_ms(&self) -> f64 {
        self.total_duration.as_secs_f64() * 1000.0
    }

    /// Returns `true` if all steps succeeded.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.steps.iter().all(|s| s.success)
    }

    /// Returns the first failed `StepOutcome`, if any.
    #[must_use]
    pub fn first_failure(&self) -> Option<&StepOutcome> {
        self.steps.iter().find(|s| !s.success)
    }

    /// Collect error messages from all failed steps.
    #[must_use]
    pub fn error_messages(&self) -> Vec<String> {
        self.steps.iter().filter_map(|s| s.error.clone()).collect()
    }
}

/// Simulated step handler type.
/// Returns `Ok(())` on success, `Err(msg)` on failure.
type StepHandler = Box<dyn Fn() -> Result<(), String> + Send + Sync>;

/// A sequential conversion pipeline.
pub struct ConversionPipeline {
    steps: Vec<(ConversionStep, StepHandler)>,
    /// Whether the pipeline should abort on the first failure.
    abort_on_failure: bool,
}

impl ConversionPipeline {
    /// Create a new empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            abort_on_failure: true,
        }
    }

    /// Set whether to abort execution on the first step failure.
    #[must_use]
    pub fn abort_on_failure(mut self, abort: bool) -> Self {
        self.abort_on_failure = abort;
        self
    }

    /// Append a step to the pipeline.
    ///
    /// `handler` is a closure that performs the step work and returns
    /// `Ok(())` or `Err(message)`.
    pub fn add_step<F>(&mut self, step: ConversionStep, handler: F)
    where
        F: Fn() -> Result<(), String> + Send + Sync + 'static,
    {
        self.steps.push((step, Box::new(handler)));
    }

    /// Return the number of steps registered.
    #[must_use]
    pub fn step_count(&self) -> usize {
        self.steps.len()
    }

    /// Return the list of step names in order.
    #[must_use]
    pub fn step_names(&self) -> Vec<&str> {
        self.steps.iter().map(|(s, _)| s.step_name()).collect()
    }

    /// Execute all steps sequentially, collecting outcomes.
    #[must_use]
    pub fn execute(self) -> ConversionResult {
        let pipeline_start = Instant::now();
        let mut outcomes = Vec::with_capacity(self.steps.len());

        for (step, handler) in self.steps {
            let step_start = Instant::now();
            let result = handler();
            let duration = step_start.elapsed();

            let (success, error) = match result {
                Ok(()) => (true, None),
                Err(msg) => (false, Some(msg)),
            };

            let abort = !success && self.abort_on_failure;

            outcomes.push(StepOutcome {
                step,
                success,
                error,
                duration,
            });

            if abort {
                break;
            }
        }

        ConversionResult {
            steps: outcomes,
            total_duration: pipeline_start.elapsed(),
        }
    }
}

impl Default for ConversionPipeline {
    fn default() -> Self {
        Self::new()
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn all_success_pipeline() -> ConversionPipeline {
        let mut p = ConversionPipeline::new();
        p.add_step(ConversionStep::FormatDetection, || Ok(()));
        p.add_step(ConversionStep::Demux, || Ok(()));
        p.add_step(ConversionStep::VideoDecode, || Ok(()));
        p.add_step(ConversionStep::VideoEncode, || Ok(()));
        p.add_step(ConversionStep::Mux, || Ok(()));
        p
    }

    #[test]
    fn test_step_name_predefined() {
        assert_eq!(ConversionStep::Demux.step_name(), "demux");
        assert_eq!(ConversionStep::VideoEncode.step_name(), "video_encode");
        assert_eq!(ConversionStep::Finalise.step_name(), "finalise");
    }

    #[test]
    fn test_step_name_custom() {
        let step = ConversionStep::Custom("my_step".to_string());
        assert_eq!(step.step_name(), "my_step");
    }

    #[test]
    fn test_step_is_decode() {
        assert!(ConversionStep::VideoDecode.is_decode());
        assert!(ConversionStep::AudioDecode.is_decode());
        assert!(!ConversionStep::VideoEncode.is_decode());
    }

    #[test]
    fn test_step_is_encode() {
        assert!(ConversionStep::VideoEncode.is_encode());
        assert!(!ConversionStep::VideoDecode.is_encode());
    }

    #[test]
    fn test_step_is_filter() {
        assert!(ConversionStep::VideoFilter.is_filter());
        assert!(ConversionStep::AudioFilter.is_filter());
        assert!(!ConversionStep::Mux.is_filter());
    }

    #[test]
    fn test_pipeline_step_count() {
        let p = all_success_pipeline();
        assert_eq!(p.step_count(), 5);
    }

    #[test]
    fn test_pipeline_step_names_order() {
        let p = all_success_pipeline();
        let names = p.step_names();
        assert_eq!(names[0], "format_detection");
        assert_eq!(names[1], "demux");
        assert_eq!(names[4], "mux");
    }

    #[test]
    fn test_execute_all_success() {
        let p = all_success_pipeline();
        let result = p.execute();
        assert!(result.is_success());
        assert_eq!(result.steps.len(), 5);
        assert!((result.success_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_execute_abort_on_failure() {
        let mut p = ConversionPipeline::new();
        p.add_step(ConversionStep::FormatDetection, || Ok(()));
        p.add_step(ConversionStep::Demux, || Err("demux error".to_string()));
        p.add_step(ConversionStep::VideoDecode, || Ok(()));

        let result = p.execute();
        // Only 2 steps should have been run (aborts after Demux fails).
        assert_eq!(result.steps.len(), 2);
        assert!(!result.is_success());
    }

    #[test]
    fn test_execute_continue_on_failure() {
        let mut p = ConversionPipeline::new().abort_on_failure(false);
        p.add_step(ConversionStep::FormatDetection, || {
            Err("fmt err".to_string())
        });
        p.add_step(ConversionStep::Demux, || Ok(()));
        p.add_step(ConversionStep::Mux, || Ok(()));

        let result = p.execute();
        assert_eq!(result.steps.len(), 3);
        assert!(!result.is_success());
        assert!((result.success_rate() - 2.0 / 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_success_rate_empty_pipeline() {
        let result = ConversionResult {
            steps: vec![],
            total_duration: Duration::ZERO,
        };
        assert!((result.success_rate() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_elapsed_ms_nonzero() {
        let p = all_success_pipeline();
        let result = p.execute();
        // elapsed_ms should be >= 0.
        assert!(result.elapsed_ms() >= 0.0);
    }

    #[test]
    fn test_first_failure_returns_correct_step() {
        let mut p = ConversionPipeline::new().abort_on_failure(false);
        p.add_step(ConversionStep::FormatDetection, || Ok(()));
        p.add_step(ConversionStep::Demux, || Err("oops".to_string()));
        p.add_step(ConversionStep::Mux, || Ok(()));
        let result = p.execute();
        let failure = result.first_failure().unwrap();
        assert_eq!(failure.step.step_name(), "demux");
        assert_eq!(failure.error.as_deref(), Some("oops"));
    }

    #[test]
    fn test_error_messages_collected() {
        let mut p = ConversionPipeline::new().abort_on_failure(false);
        p.add_step(ConversionStep::FormatDetection, || Err("e1".to_string()));
        p.add_step(ConversionStep::Mux, || Err("e2".to_string()));
        let result = p.execute();
        let msgs = result.error_messages();
        assert_eq!(msgs.len(), 2);
        assert!(msgs.contains(&"e1".to_string()));
        assert!(msgs.contains(&"e2".to_string()));
    }

    #[test]
    fn test_add_custom_step() {
        let mut p = ConversionPipeline::new();
        p.add_step(ConversionStep::Custom("watermark".to_string()), || Ok(()));
        assert_eq!(p.step_count(), 1);
        assert_eq!(p.step_names()[0], "watermark");
    }
}
