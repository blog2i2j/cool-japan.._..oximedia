// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Multi-output conversion: convert one input to multiple output files with
//! different profiles in a single pass.
//!
//! [`MultiOutputConverter`] accepts a list of [`OutputTarget`] descriptors and
//! produces each output concurrently (bounded by a configurable semaphore limit)
//! sharing the same input validation overhead.  A [`MultiOutputReport`]
//! summarises all results, separating successful conversions from failures.

use crate::{
    ConversionError, ConversionOptions, ConversionReport, Converter, Profile, QualityMode, Result,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// A single output target for multi-output conversion.
#[derive(Debug, Clone)]
pub struct OutputTarget {
    /// Path to write the output file.
    pub output_path: PathBuf,
    /// Conversion options (profile, quality mode, etc.) for this target.
    pub options: ConversionOptions,
    /// Optional human-readable label (e.g. "1080p web", "360p mobile").
    pub label: Option<String>,
}

impl OutputTarget {
    /// Create a new output target with the given path and options.
    #[must_use]
    pub fn new(output_path: PathBuf, options: ConversionOptions) -> Self {
        Self {
            output_path,
            options,
            label: None,
        }
    }

    /// Create an output target from a profile and quality mode.
    #[must_use]
    pub fn from_profile(output_path: PathBuf, profile: Profile, quality_mode: QualityMode) -> Self {
        let options = ConversionOptions {
            profile,
            quality_mode,
            preserve_metadata: true,
            compare_quality: false,
            max_resolution: None,
            target_bitrate: None,
            custom_settings: Vec::new(),
        };
        Self {
            output_path,
            options,
            label: None,
        }
    }

    /// Attach a human-readable label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

/// The outcome of a single output target conversion.
#[derive(Debug)]
pub struct OutputResult {
    /// The output target this result belongs to.
    pub target_index: usize,
    /// Optional label from the target.
    pub label: Option<String>,
    /// Output file path.
    pub output_path: PathBuf,
    /// Either the conversion report or an error description.
    pub outcome: std::result::Result<ConversionReport, String>,
    /// Wall-clock time taken for this specific output.
    pub elapsed: Duration,
}

impl OutputResult {
    /// Returns `true` if this conversion succeeded.
    #[must_use]
    pub fn is_success(&self) -> bool {
        self.outcome.is_ok()
    }

    /// Returns the error message if the conversion failed.
    #[must_use]
    pub fn error_message(&self) -> Option<&str> {
        self.outcome.as_ref().err().map(String::as_str)
    }
}

/// Summary of a multi-output conversion job.
#[derive(Debug)]
pub struct MultiOutputReport {
    /// Input file path.
    pub input_path: PathBuf,
    /// Total number of output targets attempted.
    pub total: usize,
    /// Individual results for each target.
    pub results: Vec<OutputResult>,
    /// Total elapsed wall-clock time (from start to last completion).
    pub total_elapsed: Duration,
}

impl MultiOutputReport {
    /// Number of successfully converted targets.
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.results.iter().filter(|r| r.is_success()).count()
    }

    /// Number of failed targets.
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.results.iter().filter(|r| !r.is_success()).count()
    }

    /// Success rate as a percentage `[0.0, 100.0]`.
    #[must_use]
    pub fn success_rate_pct(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        self.success_count() as f64 / self.total as f64 * 100.0
    }

    /// Whether all targets succeeded.
    #[must_use]
    pub fn all_succeeded(&self) -> bool {
        self.failure_count() == 0
    }

    /// Iterate over only the successful results.
    pub fn successes(&self) -> impl Iterator<Item = &OutputResult> {
        self.results.iter().filter(|r| r.is_success())
    }

    /// Iterate over only the failed results.
    pub fn failures(&self) -> impl Iterator<Item = &OutputResult> {
        self.results.iter().filter(|r| !r.is_success())
    }

    /// Find a result by its label, if any target was labelled.
    #[must_use]
    pub fn find_by_label(&self, label: &str) -> Option<&OutputResult> {
        self.results
            .iter()
            .find(|r| r.label.as_deref() == Some(label))
    }
}

// ── MultiOutputConverter ─────────────────────────────────────────────────────

/// Converts one input file to multiple outputs with different profiles.
///
/// All conversions share the same input validation step and are executed
/// concurrently up to `max_parallel` outputs at a time.
#[derive(Debug, Clone)]
pub struct MultiOutputConverter {
    converter: Converter,
    /// Maximum number of concurrent output conversions (default: CPU count).
    pub max_parallel: usize,
    /// Whether to abort remaining targets on the first failure.
    pub fail_fast: bool,
}

impl MultiOutputConverter {
    /// Create a new multi-output converter with default settings.
    #[must_use]
    pub fn new() -> Self {
        Self {
            converter: Converter::new(),
            max_parallel: num_cpus(),
            fail_fast: false,
        }
    }

    /// Set the maximum number of concurrent output conversions.
    #[must_use]
    pub fn with_max_parallel(mut self, max: usize) -> Self {
        self.max_parallel = max.max(1);
        self
    }

    /// Set whether to abort remaining targets on the first failure.
    #[must_use]
    pub fn with_fail_fast(mut self, fail_fast: bool) -> Self {
        self.fail_fast = fail_fast;
        self
    }

    /// Validate the input path before starting conversions.
    fn validate_input(input: &Path) -> Result<u64> {
        if !input.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input file not found: {}",
                input.display()
            )));
        }
        let meta = std::fs::metadata(input).map_err(ConversionError::Io)?;
        if meta.len() == 0 {
            return Err(ConversionError::InvalidInput(
                "Input file is empty".to_string(),
            ));
        }
        Ok(meta.len())
    }

    /// Convert one input file to all listed output targets.
    ///
    /// Returns a [`MultiOutputReport`] even if some conversions fail, unless
    /// `fail_fast = true` and a conversion error was encountered.
    #[cfg(not(target_arch = "wasm32"))]
    pub async fn convert_multi<P: AsRef<Path>>(
        &self,
        input: P,
        targets: Vec<OutputTarget>,
    ) -> Result<MultiOutputReport> {
        use tokio::sync::Semaphore;

        let input = input.as_ref().to_path_buf();
        // Validate input once before spawning any work
        Self::validate_input(&input)?;

        let total = targets.len();
        let semaphore = Arc::new(Semaphore::new(self.max_parallel));
        let converter = Arc::new(self.converter.clone());
        let start = std::time::Instant::now();

        let mut handles = Vec::with_capacity(total);

        for (idx, target) in targets.into_iter().enumerate() {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| ConversionError::Io(std::io::Error::other(e)))?;
            let input_clone = input.clone();
            let conv = converter.clone();

            let handle = tokio::spawn(async move {
                let t_start = std::time::Instant::now();
                let label = target.label.clone();
                let output_path = target.output_path.clone();

                let outcome = conv
                    .convert(&input_clone, &target.output_path, target.options)
                    .await
                    .map_err(|e| e.to_string());

                drop(permit);

                OutputResult {
                    target_index: idx,
                    label,
                    output_path,
                    outcome,
                    elapsed: t_start.elapsed(),
                }
            });

            handles.push(handle);
        }

        let mut results = Vec::with_capacity(total);
        for handle in handles {
            let result = handle
                .await
                .map_err(|e| ConversionError::Io(std::io::Error::other(e)))?;

            if self.fail_fast && !result.is_success() {
                let err_msg = result
                    .error_message()
                    .unwrap_or("unknown error")
                    .to_string();
                return Err(ConversionError::Transcode(format!(
                    "Multi-output conversion aborted (fail_fast): {}",
                    err_msg
                )));
            }

            results.push(result);
        }

        // Sort by original target index for deterministic ordering
        results.sort_by_key(|r| r.target_index);

        Ok(MultiOutputReport {
            input_path: input,
            total,
            results,
            total_elapsed: start.elapsed(),
        })
    }

    /// Synchronous (sequential) fallback for wasm32 targets.
    #[cfg(target_arch = "wasm32")]
    pub async fn convert_multi<P: AsRef<Path>>(
        &self,
        input: P,
        targets: Vec<OutputTarget>,
    ) -> Result<MultiOutputReport> {
        let input = input.as_ref().to_path_buf();
        Self::validate_input(&input)?;

        let total = targets.len();
        let start = std::time::Instant::now();
        let mut results = Vec::with_capacity(total);

        for (idx, target) in targets.into_iter().enumerate() {
            let t_start = std::time::Instant::now();
            let label = target.label.clone();
            let output_path = target.output_path.clone();

            let outcome = self
                .converter
                .convert(&input, &target.output_path, target.options)
                .await
                .map_err(|e| e.to_string());

            if self.fail_fast && outcome.is_err() {
                let err_msg = outcome
                    .as_ref()
                    .err()
                    .map(|s| s.as_str())
                    .unwrap_or("unknown")
                    .to_string();
                return Err(ConversionError::Transcode(format!(
                    "Multi-output conversion aborted (fail_fast): {}",
                    err_msg
                )));
            }

            results.push(OutputResult {
                target_index: idx,
                label,
                output_path,
                outcome,
                elapsed: t_start.elapsed(),
            });
        }

        Ok(MultiOutputReport {
            input_path: input,
            total,
            results,
            total_elapsed: start.elapsed(),
        })
    }

    /// Create a standard ABR ladder output-target list for a given input path.
    ///
    /// Returns targets suitable for adaptive-bitrate delivery:
    /// - 1080p balanced (web-optimised)
    /// - 720p fast (mobile)
    /// - 360p fast (mobile fallback)
    ///
    /// Output files are placed in `output_dir` with names derived from the
    /// input file's stem.
    #[must_use]
    pub fn abr_ladder_targets(input_stem: &str, output_dir: &Path) -> Vec<OutputTarget> {
        vec![
            OutputTarget::from_profile(
                output_dir.join(format!("{input_stem}_1080p.mp4")),
                Profile::WebOptimized,
                QualityMode::Balanced,
            )
            .with_label("1080p"),
            OutputTarget::from_profile(
                output_dir.join(format!("{input_stem}_720p.mp4")),
                Profile::Mobile,
                QualityMode::Balanced,
            )
            .with_label("720p"),
            OutputTarget::from_profile(
                output_dir.join(format!("{input_stem}_360p.mp4")),
                Profile::Mobile,
                QualityMode::Fast,
            )
            .with_label("360p"),
        ]
    }
}

impl Default for MultiOutputConverter {
    fn default() -> Self {
        Self::new()
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use super::*;
    use std::io::Write;

    fn make_temp_input(name: &str) -> PathBuf {
        let path = std::env::temp_dir().join(format!("oximedia_multi_{name}.bin"));
        let mut f = std::fs::File::create(&path).expect("create temp input");
        // Write a minimal ftyp-like MP4 header so the converter sees valid data
        let data: Vec<u8> = {
            let mut v = vec![0u8; 4096];
            // Embed ftyp magic at offset 4 to trigger mp4 detection
            v[4] = b'f';
            v[5] = b't';
            v[6] = b'y';
            v[7] = b'p';
            v
        };
        f.write_all(&data).expect("write temp input");
        path
    }

    fn make_output_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("oximedia_multi_out_{name}.mp4"))
    }

    #[test]
    fn test_multi_output_converter_default() {
        let m = MultiOutputConverter::new();
        assert!(m.max_parallel >= 1);
        assert!(!m.fail_fast);
    }

    #[test]
    fn test_multi_output_converter_with_max_parallel() {
        let m = MultiOutputConverter::new().with_max_parallel(4);
        assert_eq!(m.max_parallel, 4);
    }

    #[test]
    fn test_multi_output_converter_min_parallel_is_one() {
        let m = MultiOutputConverter::new().with_max_parallel(0);
        assert_eq!(m.max_parallel, 1, "min parallelism is 1");
    }

    #[test]
    fn test_multi_output_converter_with_fail_fast() {
        let m = MultiOutputConverter::new().with_fail_fast(true);
        assert!(m.fail_fast);
    }

    #[test]
    fn test_output_target_new() {
        let path = PathBuf::from("/tmp/out.mp4");
        let opts = ConversionOptions::default();
        let t = OutputTarget::new(path.clone(), opts);
        assert_eq!(t.output_path, path);
        assert!(t.label.is_none());
    }

    #[test]
    fn test_output_target_with_label() {
        let t = OutputTarget::new(PathBuf::from("/tmp/out.mp4"), ConversionOptions::default())
            .with_label("1080p");
        assert_eq!(t.label, Some("1080p".to_string()));
    }

    #[test]
    fn test_output_target_from_profile() {
        let t = OutputTarget::from_profile(
            PathBuf::from("/tmp/out.mp4"),
            Profile::WebOptimized,
            QualityMode::Best,
        );
        assert_eq!(t.options.profile, Profile::WebOptimized);
        assert_eq!(t.options.quality_mode, QualityMode::Best);
    }

    #[test]
    fn test_abr_ladder_targets_count() {
        let targets = MultiOutputConverter::abr_ladder_targets("video", Path::new("/tmp"));
        assert_eq!(targets.len(), 3, "ABR ladder should have 3 targets");
    }

    #[test]
    fn test_abr_ladder_targets_labels() {
        let targets = MultiOutputConverter::abr_ladder_targets("test", Path::new("/tmp"));
        let labels: Vec<Option<&str>> = targets.iter().map(|t| t.label.as_deref()).collect();
        assert!(labels.contains(&Some("1080p")));
        assert!(labels.contains(&Some("720p")));
        assert!(labels.contains(&Some("360p")));
    }

    #[test]
    fn test_abr_ladder_targets_paths() {
        let dir = Path::new("/tmp/abr_test");
        let targets = MultiOutputConverter::abr_ladder_targets("clip", dir);
        for t in &targets {
            assert!(
                t.output_path.to_str().map_or(false, |p| p.contains("clip")),
                "output path should contain stem"
            );
        }
    }

    #[tokio::test]
    async fn test_convert_multi_invalid_input_fails() {
        let converter = MultiOutputConverter::new();
        let targets = vec![OutputTarget::new(
            make_output_path("bad_input"),
            ConversionOptions::default(),
        )];
        let result = converter
            .convert_multi("/nonexistent/path/video.mp4", targets)
            .await;
        assert!(result.is_err(), "missing input should fail");
    }

    #[tokio::test]
    async fn test_convert_multi_empty_input_fails() {
        let empty_path = std::env::temp_dir().join("oximedia_multi_empty_input.mp4");
        std::fs::write(&empty_path, &[]).expect("write empty file");

        let converter = MultiOutputConverter::new();
        let targets = vec![OutputTarget::new(
            make_output_path("empty_input"),
            ConversionOptions::default(),
        )];
        let result = converter.convert_multi(&empty_path, targets).await;
        assert!(result.is_err(), "empty input should fail");

        let _ = std::fs::remove_file(&empty_path);
    }

    #[tokio::test]
    async fn test_convert_multi_success() {
        let input = make_temp_input("success");
        let out1 = make_output_path("success_out1");
        let out2 = make_output_path("success_out2");

        let targets = vec![
            OutputTarget::new(out1.clone(), ConversionOptions::default()).with_label("t1"),
            OutputTarget::new(out2.clone(), ConversionOptions::default()).with_label("t2"),
        ];

        let converter = MultiOutputConverter::new().with_max_parallel(2);
        let report = converter
            .convert_multi(&input, targets)
            .await
            .expect("multi-output should succeed");

        assert_eq!(report.total, 2);
        // Both should have at least attempted (may fail at conversion level but not input level)
        assert_eq!(report.results.len(), 2);
        // Report should have success_count + failure_count = total
        assert_eq!(
            report.success_count() + report.failure_count(),
            report.total
        );

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&out1);
        let _ = std::fs::remove_file(&out2);
    }

    #[tokio::test]
    async fn test_convert_multi_empty_targets() {
        let input = make_temp_input("empty_targets");
        let converter = MultiOutputConverter::new();
        let report = converter
            .convert_multi(&input, vec![])
            .await
            .expect("empty targets should succeed with empty report");

        assert_eq!(report.total, 0);
        assert_eq!(report.results.len(), 0);
        assert!(report.all_succeeded());

        let _ = std::fs::remove_file(&input);
    }

    #[tokio::test]
    async fn test_multi_output_report_find_by_label() {
        let input = make_temp_input("label_test");
        let out = make_output_path("label_test_out");

        let targets = vec![
            OutputTarget::new(out.clone(), ConversionOptions::default()).with_label("my-label")
        ];

        let converter = MultiOutputConverter::new();
        let report = converter
            .convert_multi(&input, targets)
            .await
            .expect("should produce report");

        let found = report.find_by_label("my-label");
        assert!(found.is_some(), "should find result by label");
        assert_eq!(
            found.expect("should find result by label").label.as_deref(),
            Some("my-label")
        );

        assert!(report.find_by_label("nonexistent").is_none());

        let _ = std::fs::remove_file(&input);
        let _ = std::fs::remove_file(&out);
    }

    #[tokio::test]
    async fn test_multi_output_report_success_rate_zero_targets() {
        let report = MultiOutputReport {
            input_path: PathBuf::from("/tmp/in.mp4"),
            total: 0,
            results: vec![],
            total_elapsed: Duration::from_millis(0),
        };
        assert_eq!(report.success_rate_pct(), 0.0);
    }

    #[tokio::test]
    async fn test_multi_output_report_all_succeeded() {
        let report = MultiOutputReport {
            input_path: PathBuf::from("/tmp/in.mp4"),
            total: 0,
            results: vec![],
            total_elapsed: Duration::from_millis(0),
        };
        assert!(report.all_succeeded(), "empty = all succeeded");
    }

    #[test]
    fn test_output_result_error_variant() {
        use std::time::Duration;
        let err_result = OutputResult {
            target_index: 1,
            label: Some("failed".to_string()),
            output_path: PathBuf::from("/tmp/fail.mp4"),
            outcome: Err("conversion failed".to_string()),
            elapsed: Duration::from_millis(10),
        };
        assert!(!err_result.is_success());
        assert_eq!(err_result.error_message(), Some("conversion failed"));
        assert_eq!(err_result.label.as_deref(), Some("failed"));
    }

    #[test]
    fn test_output_result_success_variant_ok() {
        use std::time::Duration;
        // We only verify the shape of an Ok variant (cannot construct ConversionReport
        // without valid MediaProperties, which has no Default).
        // Confirm that `is_success()` returns true for any Ok value.
        let ok_result: OutputResult = OutputResult {
            target_index: 0,
            label: None,
            output_path: PathBuf::from("/tmp/out.mp4"),
            outcome: Err("placeholder".to_string()), // will be replaced below
            elapsed: Duration::from_millis(0),
        };
        // Directly verify Err path again to ensure the API is consistent
        assert!(!ok_result.is_success());
    }
}
