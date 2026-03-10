//! Batch processing utilities for caption operations

use crate::error::Result;
use crate::export::Exporter;
use crate::import::Importer;
use crate::types::CaptionTrack;
use crate::validation::{ValidationReport, Validator};
use crate::CaptionFormat;
use std::path::{Path, PathBuf};

/// Batch processor for caption files
pub struct BatchProcessor {
    /// Input directory
    input_dir: PathBuf,
    /// Output directory
    output_dir: PathBuf,
    /// Validator for quality control
    validator: Option<Validator>,
}

impl BatchProcessor {
    /// Create a new batch processor
    #[must_use]
    pub fn new(input_dir: PathBuf, output_dir: PathBuf) -> Self {
        Self {
            input_dir,
            output_dir,
            validator: None,
        }
    }

    /// Set validator for quality control
    #[must_use]
    pub fn with_validator(mut self, validator: Validator) -> Self {
        self.validator = Some(validator);
        self
    }

    /// Convert all caption files from one format to another
    pub fn convert_all(
        &self,
        from_format: CaptionFormat,
        to_format: CaptionFormat,
    ) -> Result<BatchResult> {
        let files = self.find_caption_files(from_format)?;
        let mut result = BatchResult::new();

        for file in &files {
            match self.convert_file(file, from_format, to_format) {
                Ok(output_path) => {
                    result.successful.push(output_path);
                }
                Err(e) => {
                    result.failed.push((file.clone(), e.to_string()));
                }
            }
        }

        Ok(result)
    }

    fn convert_file(
        &self,
        input_path: &Path,
        from_format: CaptionFormat,
        to_format: CaptionFormat,
    ) -> Result<PathBuf> {
        // Import
        let track = Importer::import_from_file(input_path, Some(from_format))?;

        // Validate if validator is set
        if let Some(ref validator) = self.validator {
            let report = validator.validate(&track)?;
            if !report.passed() {
                return Err(crate::error::CaptionError::Validation(format!(
                    "Validation failed: {} errors",
                    report.statistics.error_count
                )));
            }
        }

        // Export
        let filename = input_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let output_path = self
            .output_dir
            .join(format!("{}.{}", filename, to_format.extension()));

        Exporter::export_to_file(&track, &output_path, to_format)?;

        Ok(output_path)
    }

    fn find_caption_files(&self, format: CaptionFormat) -> Result<Vec<PathBuf>> {
        let extension = format.extension();
        let mut files = Vec::new();

        if !self.input_dir.exists() {
            return Ok(files);
        }

        for entry in std::fs::read_dir(&self.input_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Some(ext) = path.extension() {
                    if ext == extension {
                        files.push(path);
                    }
                }
            }
        }

        Ok(files)
    }

    /// Validate all caption files in directory
    pub fn validate_all(&self, validator: &Validator) -> Result<Vec<(PathBuf, ValidationReport)>> {
        let mut results = Vec::new();

        for entry in std::fs::read_dir(&self.input_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Ok(track) = Importer::import_from_file(&path, None) {
                    let report = validator.validate(&track)?;
                    results.push((path, report));
                }
            }
        }

        Ok(results)
    }

    /// Generate quality reports for all captions
    pub fn generate_reports(&self) -> Result<Vec<(PathBuf, crate::report::QualityReport)>> {
        let mut reports = Vec::new();

        for entry in std::fs::read_dir(&self.input_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                if let Ok(track) = Importer::import_from_file(&path, None) {
                    let report = crate::report::QualityReport::generate(&track)?;
                    reports.push((path, report));
                }
            }
        }

        Ok(reports)
    }

    /// Apply operation to all caption files
    pub fn apply_operation<F>(&self, operation: F) -> Result<BatchResult>
    where
        F: Fn(&mut CaptionTrack) -> Result<()>,
    {
        let mut result = BatchResult::new();

        for entry in std::fs::read_dir(&self.input_dir)? {
            let entry = entry?;
            let path = entry.path();

            if path.is_file() {
                match self.apply_to_file(&path, &operation) {
                    Ok(output_path) => {
                        result.successful.push(output_path);
                    }
                    Err(e) => {
                        result.failed.push((path, e.to_string()));
                    }
                }
            }
        }

        Ok(result)
    }

    fn apply_to_file<F>(&self, path: &Path, operation: F) -> Result<PathBuf>
    where
        F: Fn(&mut CaptionTrack) -> Result<()>,
    {
        let mut track = Importer::import_from_file(path, None)?;

        operation(&mut track)?;

        let file_name = path.file_name().ok_or_else(|| {
            crate::error::CaptionError::Import("Input path has no file name component".to_string())
        })?;
        let output_path = self.output_dir.join(file_name);
        let format = Importer::detect_format_from_extension(path)
            .ok_or_else(|| crate::error::CaptionError::Import("Unknown format".to_string()))?;

        Exporter::export_to_file(&track, &output_path, format)?;

        Ok(output_path)
    }
}

/// Batch processing result
#[derive(Debug, Clone)]
pub struct BatchResult {
    /// Successful operations
    pub successful: Vec<PathBuf>,
    /// Failed operations (path, error message)
    pub failed: Vec<(PathBuf, String)>,
}

impl BatchResult {
    /// Create a new batch result
    #[must_use]
    pub fn new() -> Self {
        Self {
            successful: Vec::new(),
            failed: Vec::new(),
        }
    }

    /// Get success count
    #[must_use]
    pub fn success_count(&self) -> usize {
        self.successful.len()
    }

    /// Get failure count
    #[must_use]
    pub fn failure_count(&self) -> usize {
        self.failed.len()
    }

    /// Get success rate
    #[must_use]
    pub fn success_rate(&self) -> f64 {
        let total = self.success_count() + self.failure_count();
        if total == 0 {
            0.0
        } else {
            self.success_count() as f64 / total as f64
        }
    }
}

impl Default for BatchResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Batch operations
pub struct BatchOperations;

impl BatchOperations {
    /// Shift timing for all captions in directory
    pub fn shift_timing_batch(processor: &BatchProcessor, offset_ms: i64) -> Result<BatchResult> {
        processor.apply_operation(|track| {
            let offset = crate::types::Duration::from_millis(offset_ms);
            for caption in &mut track.captions {
                caption.start = caption.start.add(offset);
                caption.end = caption.end.add(offset);
            }
            Ok(())
        })
    }

    /// Fix overlaps in all captions
    pub fn fix_overlaps_batch(
        processor: &BatchProcessor,
        min_gap_frames: u32,
        fps: f64,
    ) -> Result<BatchResult> {
        processor.apply_operation(|track| {
            crate::authoring::TimingControl::fix_overlaps(track, min_gap_frames, fps)?;
            Ok(())
        })
    }

    /// Apply line breaking to all captions
    pub fn apply_line_breaking_batch(
        processor: &BatchProcessor,
        max_chars_per_line: usize,
        max_lines: usize,
    ) -> Result<BatchResult> {
        processor.apply_operation(|track| {
            for caption in &mut track.captions {
                crate::authoring::LineBreaker::apply_to_caption(
                    caption,
                    max_chars_per_line,
                    max_lines,
                );
            }
            Ok(())
        })
    }

    /// Apply style template to all captions
    pub fn apply_template_batch(
        processor: &BatchProcessor,
        template: &crate::templates::Template,
    ) -> Result<BatchResult> {
        processor.apply_operation(|track| {
            for caption in &mut track.captions {
                caption.style = template.style.clone();
            }
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    // use std::fs;
    #[allow(unused_imports)]
    use tempfile::TempDir;

    #[test]
    fn test_batch_result() {
        let mut result = BatchResult::new();
        assert_eq!(result.success_count(), 0);
        assert_eq!(result.failure_count(), 0);

        result.successful.push(PathBuf::from("test.srt"));
        assert_eq!(result.success_count(), 1);
        assert_eq!(result.success_rate(), 1.0);

        result
            .failed
            .push((PathBuf::from("fail.srt"), "Error".to_string()));
        assert_eq!(result.success_rate(), 0.5);
    }
}
