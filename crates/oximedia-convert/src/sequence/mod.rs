// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Image sequence support for converting between videos and image sequences.

use crate::formats::ImageFormat;
use crate::{ConversionError, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Image sequence configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageSequence {
    /// Directory containing images
    pub directory: PathBuf,
    /// Naming pattern (e.g., "frame_%04d.png")
    pub pattern: String,
    /// Image format
    pub format: ImageFormat,
    /// Frame rate for video generation
    pub frame_rate: f64,
    /// Start frame number
    pub start_frame: u32,
    /// Number of frames
    pub frame_count: Option<u32>,
}

impl ImageSequence {
    /// Create a new image sequence configuration.
    #[must_use]
    pub fn new(directory: PathBuf, pattern: String, format: ImageFormat, frame_rate: f64) -> Self {
        Self {
            directory,
            pattern,
            format,
            frame_rate,
            start_frame: 0,
            frame_count: None,
        }
    }

    /// Set start frame.
    #[must_use]
    pub fn with_start_frame(mut self, frame: u32) -> Self {
        self.start_frame = frame;
        self
    }

    /// Set frame count.
    #[must_use]
    pub fn with_frame_count(mut self, count: u32) -> Self {
        self.frame_count = Some(count);
        self
    }

    /// Get path for a specific frame.
    #[must_use]
    pub fn frame_path(&self, frame: u32) -> PathBuf {
        let filename = self
            .pattern
            .replace("%04d", &format!("{frame:04}"))
            .replace("%d", &frame.to_string());
        self.directory.join(filename)
    }

    /// List all frames in sequence.
    pub fn list_frames(&self) -> Result<Vec<PathBuf>> {
        let mut frames = Vec::new();
        let mut frame_num = self.start_frame;

        loop {
            let path = self.frame_path(frame_num);
            if !path.exists() {
                break;
            }

            frames.push(path);
            frame_num += 1;

            if let Some(count) = self.frame_count {
                if frame_num >= self.start_frame + count {
                    break;
                }
            }
        }

        if frames.is_empty() {
            return Err(ConversionError::InvalidInput(
                "No frames found in sequence".to_string(),
            ));
        }

        Ok(frames)
    }
}

/// Image sequence exporter for converting video to image sequence.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct SequenceExporter {
    format: ImageFormat,
    quality: u32,
}

impl SequenceExporter {
    /// Create a new sequence exporter.
    #[must_use]
    pub const fn new(format: ImageFormat, quality: u32) -> Self {
        Self { format, quality }
    }

    /// Export video to image sequence.
    ///
    /// Validates the input video path and creates the output directory, then
    /// returns an `ImageSequence` descriptor ready for the transcode pipeline
    /// to populate. Actual frame extraction requires the transcode crate
    /// integration which is deferred to a future milestone.
    pub async fn export(
        &self,
        video_path: &Path,
        output_dir: &Path,
        pattern: &str,
    ) -> Result<ImageSequence> {
        if !video_path.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Input video not found: {}",
                video_path.display()
            )));
        }

        // Ensure output directory exists.
        std::fs::create_dir_all(output_dir)?;

        let effective_pattern = if pattern.is_empty() {
            "frame_%04d.png"
        } else {
            pattern
        };

        Ok(ImageSequence::new(
            output_dir.to_path_buf(),
            effective_pattern.to_string(),
            self.format,
            30.0,
        ))
    }
}

/// Image sequence importer for converting image sequence to video.
#[derive(Debug, Clone)]
pub struct SequenceImporter;

impl SequenceImporter {
    /// Create a new sequence importer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Import image sequence as video.
    ///
    /// Validates that at least one frame exists in the sequence and that the
    /// output parent directory is accessible. Actual muxing requires the
    /// transcode crate integration which is deferred to a future milestone.
    pub async fn import(&self, sequence: &ImageSequence, output_path: &Path) -> Result<()> {
        // Validate that the sequence directory exists.
        if !sequence.directory.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "Sequence directory not found: {}",
                sequence.directory.display()
            )));
        }

        // Validate at least the first frame is present.
        let first_frame = sequence.frame_path(sequence.start_frame);
        if !first_frame.exists() {
            return Err(ConversionError::InvalidInput(format!(
                "First frame not found: {}",
                first_frame.display()
            )));
        }

        // Ensure the output parent directory exists.
        if let Some(parent) = output_path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }

        Ok(())
    }
}

impl Default for SequenceImporter {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_sequence_creation() {
        let seq = ImageSequence::new(
            std::env::temp_dir().join("oximedia-convert-frames"),
            "frame_%04d.png".to_string(),
            ImageFormat::Png,
            30.0,
        );

        assert_eq!(seq.frame_rate, 30.0);
        assert_eq!(seq.start_frame, 0);
    }

    #[test]
    fn test_image_sequence_frame_path() {
        let seq = ImageSequence::new(
            std::env::temp_dir().join("oximedia-convert-frames"),
            "frame_%04d.png".to_string(),
            ImageFormat::Png,
            30.0,
        );

        let path = seq.frame_path(42);
        assert!(path.to_string_lossy().contains("frame_0042.png"));
    }

    #[test]
    fn test_image_sequence_builder() {
        let seq = ImageSequence::new(
            std::env::temp_dir().join("oximedia-convert-frames"),
            "frame_%04d.png".to_string(),
            ImageFormat::Png,
            30.0,
        )
        .with_start_frame(10)
        .with_frame_count(100);

        assert_eq!(seq.start_frame, 10);
        assert_eq!(seq.frame_count, Some(100));
    }

    #[test]
    fn test_sequence_exporter() {
        let exporter = SequenceExporter::new(ImageFormat::Png, 95);
        assert_eq!(exporter.format, ImageFormat::Png);
        assert_eq!(exporter.quality, 95);
    }
}
