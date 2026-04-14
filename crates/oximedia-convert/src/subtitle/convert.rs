// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Subtitle format conversion.

use crate::{ConversionError, Result};
use std::path::Path;

/// Converter for subtitle formats.
#[derive(Debug, Clone)]
pub struct SubtitleConverter {
    encoding: String,
}

impl SubtitleConverter {
    /// Create a new subtitle converter.
    #[must_use]
    pub fn new() -> Self {
        Self {
            encoding: "UTF-8".to_string(),
        }
    }

    /// Set the text encoding.
    pub fn with_encoding<S: Into<String>>(mut self, encoding: S) -> Self {
        self.encoding = encoding.into();
        self
    }

    /// Convert subtitle format.
    pub async fn convert<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
        target_format: SubtitleFormat,
    ) -> Result<()> {
        let input = input.as_ref();
        let output = output.as_ref();

        if !input.exists() {
            return Err(ConversionError::InvalidInput(
                "Input file not found".to_string(),
            ));
        }

        let source_format = self.detect_format(input)?;

        if source_format == target_format {
            std::fs::copy(input, output).map_err(ConversionError::Io)?;
            return Ok(());
        }

        // Placeholder for actual conversion
        // In a real implementation, this would use oximedia-subtitle
        Ok(())
    }

    /// Detect subtitle format from file.
    pub fn detect_format<P: AsRef<Path>>(&self, path: P) -> Result<SubtitleFormat> {
        let path = path.as_ref();

        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| ConversionError::FormatDetection("No file extension".to_string()))?;

        match ext.to_lowercase().as_str() {
            "srt" => Ok(SubtitleFormat::Srt),
            "vtt" => Ok(SubtitleFormat::WebVtt),
            "ass" | "ssa" => Ok(SubtitleFormat::Ass),
            "sub" => Ok(SubtitleFormat::SubRip),
            "sbv" => Ok(SubtitleFormat::Sbv),
            "ttml" => Ok(SubtitleFormat::Ttml),
            _ => Err(ConversionError::UnsupportedCodec(format!(
                "Unknown subtitle format: {ext}"
            ))),
        }
    }

    /// Convert SRT to `WebVTT`.
    pub async fn srt_to_webvtt<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<()> {
        self.convert(input, output, SubtitleFormat::WebVtt).await
    }

    /// Convert `WebVTT` to SRT.
    pub async fn webvtt_to_srt<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<()> {
        self.convert(input, output, SubtitleFormat::Srt).await
    }

    /// Convert any format to SRT.
    pub async fn to_srt<P: AsRef<Path>, Q: AsRef<Path>>(&self, input: P, output: Q) -> Result<()> {
        self.convert(input, output, SubtitleFormat::Srt).await
    }

    /// Convert any format to `WebVTT`.
    pub async fn to_webvtt<P: AsRef<Path>, Q: AsRef<Path>>(
        &self,
        input: P,
        output: Q,
    ) -> Result<()> {
        self.convert(input, output, SubtitleFormat::WebVtt).await
    }
}

impl Default for SubtitleConverter {
    fn default() -> Self {
        Self::new()
    }
}

/// Supported subtitle formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubtitleFormat {
    /// `SubRip` (.srt)
    Srt,
    /// `WebVTT` (.vtt)
    WebVtt,
    /// Advanced `SubStation` Alpha (.ass)
    Ass,
    /// `SubRip` (.sub)
    SubRip,
    /// `YouTube` SBV (.sbv)
    Sbv,
    /// Timed Text Markup Language (.ttml)
    Ttml,
}

impl SubtitleFormat {
    /// Get the file extension for this format.
    #[must_use]
    pub fn extension(&self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::WebVtt => "vtt",
            Self::Ass => "ass",
            Self::SubRip => "sub",
            Self::Sbv => "sbv",
            Self::Ttml => "ttml",
        }
    }

    /// Get the MIME type for this format.
    #[must_use]
    pub fn mime_type(&self) -> &'static str {
        match self {
            Self::Srt => "text/srt",
            Self::WebVtt => "text/vtt",
            Self::Ass => "text/x-ssa",
            Self::SubRip => "text/plain",
            Self::Sbv => "text/sbv",
            Self::Ttml => "application/ttml+xml",
        }
    }

    /// Get the format name.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Srt => "SubRip",
            Self::WebVtt => "WebVTT",
            Self::Ass => "Advanced SubStation Alpha",
            Self::SubRip => "SubRip",
            Self::Sbv => "YouTube SBV",
            Self::Ttml => "TTML",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_converter_creation() {
        let converter = SubtitleConverter::new();
        assert_eq!(converter.encoding, "UTF-8");
    }

    #[test]
    fn test_format_extension() {
        assert_eq!(SubtitleFormat::Srt.extension(), "srt");
        assert_eq!(SubtitleFormat::WebVtt.extension(), "vtt");
        assert_eq!(SubtitleFormat::Ass.extension(), "ass");
    }

    #[test]
    fn test_format_mime_type() {
        assert_eq!(SubtitleFormat::Srt.mime_type(), "text/srt");
        assert_eq!(SubtitleFormat::WebVtt.mime_type(), "text/vtt");
    }

    #[test]
    fn test_format_name() {
        assert_eq!(SubtitleFormat::Srt.name(), "SubRip");
        assert_eq!(SubtitleFormat::WebVtt.name(), "WebVTT");
    }

    #[test]
    fn test_detect_format() {
        let converter = SubtitleConverter::new();

        let path = Path::new("test.srt");
        assert_eq!(converter.detect_format(path).unwrap(), SubtitleFormat::Srt);

        let path = Path::new("test.vtt");
        assert_eq!(
            converter.detect_format(path).unwrap(),
            SubtitleFormat::WebVtt
        );

        let path = Path::new("test.ass");
        assert_eq!(converter.detect_format(path).unwrap(), SubtitleFormat::Ass);
    }
}
