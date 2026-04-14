// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Batch conversion processor for handling multiple file conversions.

use crate::{ConversionError, ConversionOptions, ConversionReport, Converter, Result};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Semaphore;

/// Batch processor for converting multiple files.
#[derive(Debug)]
pub struct BatchProcessor {
    converter: Converter,
    max_parallel: usize,
    resume_support: bool,
}

impl BatchProcessor {
    /// Create a new batch processor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            converter: Converter::new(),
            max_parallel: num_cpus(),
            resume_support: true,
        }
    }

    /// Set the maximum number of parallel conversions.
    #[must_use]
    pub fn with_max_parallel(mut self, max: usize) -> Self {
        self.max_parallel = max.max(1);
        self
    }

    /// Enable or disable resume support.
    #[must_use]
    pub fn with_resume_support(mut self, enabled: bool) -> Self {
        self.resume_support = enabled;
        self
    }

    /// Process a batch of conversions.
    pub async fn process_batch(&self, jobs: Vec<BatchJob>) -> Result<BatchReport> {
        let semaphore = Arc::new(Semaphore::new(self.max_parallel));
        let mut handles = Vec::new();
        let total = jobs.len();

        for (index, job) in jobs.into_iter().enumerate() {
            let permit = semaphore
                .clone()
                .acquire_owned()
                .await
                .map_err(|e| ConversionError::Io(std::io::Error::other(e)))?;
            let converter = self.converter.clone();

            let handle = tokio::spawn(async move {
                let result = converter
                    .convert(&job.input, &job.output, job.options)
                    .await;
                drop(permit);
                (index, job.input, job.output, result)
            });

            handles.push(handle);
        }

        let mut successful = Vec::new();
        let mut failed = Vec::new();

        for handle in handles {
            match handle.await {
                Ok((index, _input, _output, Ok(report))) => {
                    successful.push((index, report));
                }
                Ok((index, input, output, Err(e))) => {
                    failed.push(BatchFailure {
                        index,
                        input,
                        output,
                        error: e.to_string(),
                    });
                }
                Err(e) => {
                    return Err(ConversionError::Io(std::io::Error::other(e)));
                }
            }
        }

        Ok(BatchReport {
            total,
            successful,
            failed,
        })
    }

    /// Process files from a directory with a pattern.
    pub async fn process_directory<P: AsRef<Path>>(
        &self,
        input_dir: P,
        output_dir: P,
        pattern: &str,
        options: ConversionOptions,
    ) -> Result<BatchReport> {
        let input_dir = input_dir.as_ref();
        let output_dir = output_dir.as_ref();

        if !input_dir.is_dir() {
            return Err(ConversionError::InvalidInput(
                "Input must be a directory".to_string(),
            ));
        }

        std::fs::create_dir_all(output_dir).map_err(ConversionError::Io)?;

        let mut jobs = Vec::new();
        let entries = std::fs::read_dir(input_dir).map_err(ConversionError::Io)?;

        for entry in entries {
            let entry = entry.map_err(ConversionError::Io)?;
            let path = entry.path();

            if path.is_file() && matches_pattern(&path, pattern) {
                let file_name = path.file_stem().and_then(|s| s.to_str()).ok_or_else(|| {
                    ConversionError::InvalidInput("Invalid file name".to_string())
                })?;

                let output = output_dir.join(format!(
                    "{}.{}",
                    file_name,
                    get_extension_from_profile(&options.profile)
                ));

                jobs.push(BatchJob {
                    input: path,
                    output,
                    options: options.clone(),
                });
            }
        }

        self.process_batch(jobs).await
    }
}

impl Default for BatchProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// A single conversion job in a batch.
#[derive(Debug, Clone)]
pub struct BatchJob {
    /// Input file path
    pub input: PathBuf,
    /// Output file path
    pub output: PathBuf,
    /// Conversion options
    pub options: ConversionOptions,
}

/// Report from a batch conversion.
#[derive(Debug)]
pub struct BatchReport {
    /// Total number of jobs
    pub total: usize,
    /// Successful conversions with their reports
    pub successful: Vec<(usize, ConversionReport)>,
    /// Failed conversions
    pub failed: Vec<BatchFailure>,
}

impl BatchReport {
    /// Get the success rate as a percentage.
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }
        (self.successful.len() as f64 / self.total as f64) * 100.0
    }

    /// Get the total duration of all conversions.
    #[must_use]
    pub fn total_duration(&self) -> std::time::Duration {
        self.successful
            .iter()
            .map(|(_, report)| report.duration)
            .sum()
    }
}

/// Information about a failed conversion.
#[derive(Debug)]
pub struct BatchFailure {
    /// Job index
    pub index: usize,
    /// Input file
    pub input: PathBuf,
    /// Output file
    pub output: PathBuf,
    /// Error message
    pub error: String,
}

fn num_cpus() -> usize {
    std::thread::available_parallelism()
        .map(std::num::NonZero::get)
        .unwrap_or(4)
}

fn matches_pattern(path: &Path, pattern: &str) -> bool {
    if pattern == "*" {
        return true;
    }

    if let Some(ext) = path.extension() {
        if let Some(ext_str) = ext.to_str() {
            if pattern.starts_with("*.") {
                return ext_str == &pattern[2..];
            }
        }
    }

    false
}

fn get_extension_from_profile(profile: &crate::Profile) -> &'static str {
    match profile {
        crate::Profile::WebOptimized => "mp4",
        crate::Profile::Streaming => "m3u8",
        crate::Profile::Archive => "mkv",
        crate::Profile::Email => "mp4",
        crate::Profile::Mobile => "mp4",
        crate::Profile::YouTube => "mp4",
        crate::Profile::Instagram => "mp4",
        crate::Profile::TikTok => "mp4",
        crate::Profile::Broadcast => "mxf",
        crate::Profile::AudioMp3 => "mp3",
        crate::Profile::AudioFlac => "flac",
        crate::Profile::AudioAac => "m4a",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_processor_creation() {
        let processor = BatchProcessor::new();
        assert!(processor.max_parallel > 0);
        assert!(processor.resume_support);
    }

    #[test]
    fn test_batch_processor_config() {
        let processor = BatchProcessor::new()
            .with_max_parallel(8)
            .with_resume_support(false);

        assert_eq!(processor.max_parallel, 8);
        assert!(!processor.resume_support);
    }

    #[test]
    fn test_matches_pattern() {
        let path = Path::new("test.mp4");
        assert!(matches_pattern(path, "*"));
        assert!(matches_pattern(path, "*.mp4"));
        assert!(!matches_pattern(path, "*.mkv"));
    }

    #[test]
    fn test_batch_report_success_rate() {
        let report = BatchReport {
            total: 10,
            successful: vec![],
            failed: vec![],
        };
        assert_eq!(report.success_rate(), 0.0);
    }
}
