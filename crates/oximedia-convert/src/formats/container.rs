// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Container format handling and detection.

use super::ContainerFormat;
use crate::{ConversionError, Result};
use std::path::Path;

/// Container format detector.
#[derive(Debug, Clone)]
pub struct ContainerDetector;

impl ContainerDetector {
    /// Create a new container detector.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Detect container format from file path.
    pub fn detect_from_path(&self, path: &Path) -> Result<ContainerFormat> {
        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| ConversionError::FormatDetection("No file extension".to_string()))?;

        self.detect_from_extension(extension)
    }

    /// Detect container format from file extension.
    pub fn detect_from_extension(&self, extension: &str) -> Result<ContainerFormat> {
        match extension.to_lowercase().as_str() {
            "mp4" | "m4v" | "m4a" => Ok(ContainerFormat::Mp4),
            "mkv" => Ok(ContainerFormat::Matroska),
            "webm" => Ok(ContainerFormat::Webm),
            "ogg" | "ogv" | "oga" => Ok(ContainerFormat::Ogg),
            "ts" | "mts" | "m2ts" => Ok(ContainerFormat::MpegTs),
            "wav" => Ok(ContainerFormat::Wav),
            "flac" => Ok(ContainerFormat::Flac),
            _ => Err(ConversionError::FormatDetection(format!(
                "Unsupported container format: {extension}"
            ))),
        }
    }

    /// Detect container format from file content (magic bytes).
    pub fn detect_from_content(&self, data: &[u8]) -> Result<ContainerFormat> {
        if data.len() < 12 {
            return Err(ConversionError::FormatDetection(
                "Insufficient data for format detection".to_string(),
            ));
        }

        // Check for MP4/ISO BMFF (ftyp box)
        if data.len() >= 8 && &data[4..8] == b"ftyp" {
            return Ok(ContainerFormat::Mp4);
        }

        // Check for Matroska/WebM (EBML header)
        if data.len() >= 4
            && data[0] == 0x1A
            && data[1] == 0x45
            && data[2] == 0xDF
            && data[3] == 0xA3
        {
            // Distinguish between Matroska and WebM by checking DocType
            return Ok(ContainerFormat::Matroska); // Default to Matroska
        }

        // Check for Ogg (OggS magic)
        if data.len() >= 4 && &data[0..4] == b"OggS" {
            return Ok(ContainerFormat::Ogg);
        }

        // Check for WAV (RIFF WAVE)
        if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WAVE" {
            return Ok(ContainerFormat::Wav);
        }

        // Check for FLAC
        if data.len() >= 4 && &data[0..4] == b"fLaC" {
            return Ok(ContainerFormat::Flac);
        }

        // Check for MPEG-TS
        if !data.is_empty() && data[0] == 0x47 {
            // Verify sync byte pattern
            if data.len() >= 376 && data[188] == 0x47 {
                return Ok(ContainerFormat::MpegTs);
            }
        }

        Err(ConversionError::FormatDetection(
            "Unable to detect container format from content".to_string(),
        ))
    }
}

impl Default for ContainerDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_from_extension() {
        let detector = ContainerDetector::new();

        assert_eq!(
            detector.detect_from_extension("mp4").unwrap(),
            ContainerFormat::Mp4
        );
        assert_eq!(
            detector.detect_from_extension("mkv").unwrap(),
            ContainerFormat::Matroska
        );
        assert_eq!(
            detector.detect_from_extension("webm").unwrap(),
            ContainerFormat::Webm
        );
        assert_eq!(
            detector.detect_from_extension("ogg").unwrap(),
            ContainerFormat::Ogg
        );
        assert_eq!(
            detector.detect_from_extension("ts").unwrap(),
            ContainerFormat::MpegTs
        );
        assert_eq!(
            detector.detect_from_extension("wav").unwrap(),
            ContainerFormat::Wav
        );
        assert_eq!(
            detector.detect_from_extension("flac").unwrap(),
            ContainerFormat::Flac
        );
        assert!(detector.detect_from_extension("unknown").is_err());
    }

    #[test]
    fn test_detect_from_path() {
        let detector = ContainerDetector::new();

        assert_eq!(
            detector.detect_from_path(Path::new("video.mp4")).unwrap(),
            ContainerFormat::Mp4
        );
        assert_eq!(
            detector.detect_from_path(Path::new("video.webm")).unwrap(),
            ContainerFormat::Webm
        );
        assert!(detector.detect_from_path(Path::new("noext")).is_err());
    }

    #[test]
    fn test_detect_from_content_mp4() {
        let detector = ContainerDetector::new();
        let mp4_data = b"\x00\x00\x00\x20ftypiso5\x00\x00\x00\x00";
        assert_eq!(
            detector.detect_from_content(mp4_data).unwrap(),
            ContainerFormat::Mp4
        );
    }

    #[test]
    fn test_detect_from_content_matroska() {
        let detector = ContainerDetector::new();
        let mkv_data = b"\x1A\x45\xDF\xA3\x9F\x42\x86\x81\x01\x42\xF7\x81";
        assert_eq!(
            detector.detect_from_content(mkv_data).unwrap(),
            ContainerFormat::Matroska
        );
    }

    #[test]
    fn test_detect_from_content_ogg() {
        let detector = ContainerDetector::new();
        let ogg_data = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00";
        assert_eq!(
            detector.detect_from_content(ogg_data).unwrap(),
            ContainerFormat::Ogg
        );
    }

    #[test]
    fn test_detect_from_content_wav() {
        let detector = ContainerDetector::new();
        let wav_data = b"RIFF\x24\x08\x00\x00WAVEfmt ";
        assert_eq!(
            detector.detect_from_content(wav_data).unwrap(),
            ContainerFormat::Wav
        );
    }

    #[test]
    fn test_detect_from_content_flac() {
        let detector = ContainerDetector::new();
        let flac_data = b"fLaC\x00\x00\x00\x22\x12\x00\x12\x00";
        assert_eq!(
            detector.detect_from_content(flac_data).unwrap(),
            ContainerFormat::Flac
        );
    }

    #[test]
    fn test_detect_from_content_insufficient_data() {
        let detector = ContainerDetector::new();
        let short_data = b"\x00\x00";
        assert!(detector.detect_from_content(short_data).is_err());
    }

    #[test]
    fn test_detect_from_content_unknown() {
        let detector = ContainerDetector::new();
        let unknown_data = b"\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF\xFF";
        assert!(detector.detect_from_content(unknown_data).is_err());
    }
}
