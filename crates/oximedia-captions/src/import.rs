//! Caption import functionality

use crate::error::{CaptionError, Result};
use crate::formats::{detect_format, get_parser};
use crate::types::CaptionTrack;
use crate::CaptionFormat;
use std::path::Path;

/// Caption importer
pub struct Importer;

impl Importer {
    /// Import a caption track from bytes
    pub fn import(data: &[u8], format: CaptionFormat) -> Result<CaptionTrack> {
        if let Some(parser) = get_parser(format) {
            parser.parse(data)
        } else {
            Err(CaptionError::UnsupportedFormat(format!("{format:?}")))
        }
    }

    /// Import from a file
    pub fn import_from_file(path: &Path, format: Option<CaptionFormat>) -> Result<CaptionTrack> {
        let data = std::fs::read(path)
            .map_err(|e| CaptionError::Import(format!("Failed to read file: {e}")))?;

        let format = if let Some(fmt) = format {
            fmt
        } else {
            Self::detect_format_from_file(path, &data)?
        };

        Self::import(&data, format)
    }

    /// Auto-detect format from file content
    pub fn import_auto(data: &[u8]) -> Result<CaptionTrack> {
        let format = detect_format(data)
            .ok_or_else(|| CaptionError::Import("Could not detect caption format".to_string()))?;

        Self::import(data, format)
    }

    /// Detect format from file extension and content
    pub fn detect_format_from_file(path: &Path, data: &[u8]) -> Result<CaptionFormat> {
        // Try extension first
        if let Some(format) = Self::detect_format_from_extension(path) {
            return Ok(format);
        }

        // Fall back to content detection
        detect_format(data)
            .ok_or_else(|| CaptionError::Import("Could not determine caption format".to_string()))
    }

    /// Detect format from file extension
    #[must_use]
    pub fn detect_format_from_extension(path: &Path) -> Option<CaptionFormat> {
        path.extension()?
            .to_str()
            .and_then(|ext| match ext.to_lowercase().as_str() {
                "srt" => Some(CaptionFormat::Srt),
                "vtt" => Some(CaptionFormat::WebVtt),
                "ass" => Some(CaptionFormat::Ass),
                "ssa" => Some(CaptionFormat::Ssa),
                "ttml" => Some(CaptionFormat::Ttml),
                "dfxp" => Some(CaptionFormat::Dfxp),
                "scc" => Some(CaptionFormat::Scc),
                "stl" => Some(CaptionFormat::EbuStl),
                "itt" => Some(CaptionFormat::ITt),
                _ => None,
            })
    }

    /// Detect encoding of caption file
    #[must_use]
    pub fn detect_encoding(data: &[u8]) -> &'static str {
        // Check for BOM
        if data.starts_with(&[0xEF, 0xBB, 0xBF]) {
            return "UTF-8";
        }
        if data.starts_with(&[0xFF, 0xFE]) {
            return "UTF-16LE";
        }
        if data.starts_with(&[0xFE, 0xFF]) {
            return "UTF-16BE";
        }

        // Try to decode as UTF-8
        if std::str::from_utf8(data).is_ok() {
            return "UTF-8";
        }

        // Assume Latin-1 as fallback
        "Latin-1"
    }
}

/// Import options
#[derive(Debug, Clone, Default)]
pub struct ImportOptions {
    /// Force a specific encoding
    pub encoding: Option<String>,
    /// Skip invalid captions
    pub skip_invalid: bool,
    /// Merge overlapping captions
    pub merge_overlaps: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_import_srt() {
        let srt = b"1\n00:00:01,000 --> 00:00:03,000\nTest caption\n\n";
        let track = Importer::import_auto(srt).expect("auto import should succeed");
        assert_eq!(track.captions.len(), 1);
        assert_eq!(track.captions[0].text, "Test caption");
    }

    #[test]
    fn test_import_webvtt() {
        let vtt = b"WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nTest caption\n\n";
        let track = Importer::import_auto(vtt).expect("auto import should succeed");
        assert_eq!(track.captions.len(), 1);
        assert_eq!(track.captions[0].text, "Test caption");
    }

    #[test]
    fn test_format_detection() {
        let srt = b"1\n00:00:01,000 --> 00:00:03,000\nTest\n\n";
        assert_eq!(detect_format(srt), Some(CaptionFormat::Srt));

        let vtt = b"WEBVTT\n\n00:00:01.000 --> 00:00:03.000\nTest\n\n";
        assert_eq!(detect_format(vtt), Some(CaptionFormat::WebVtt));
    }

    #[test]
    fn test_encoding_detection() {
        let utf8 = b"Test string";
        assert_eq!(Importer::detect_encoding(utf8), "UTF-8");

        let utf8_bom = b"\xEF\xBB\xBFTest string";
        assert_eq!(Importer::detect_encoding(utf8_bom), "UTF-8");
    }
}
