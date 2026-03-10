//! Caption export functionality

use crate::error::{CaptionError, Result};
use crate::formats::get_writer;
use crate::types::CaptionTrack;
use crate::CaptionFormat;
use std::path::Path;

/// Caption exporter
pub struct Exporter;

impl Exporter {
    /// Export a caption track to a specific format
    pub fn export(track: &CaptionTrack, format: CaptionFormat) -> Result<Vec<u8>> {
        if let Some(writer) = get_writer(format) {
            writer.write(track)
        } else {
            Err(CaptionError::UnsupportedFormat(format!("{format:?}")))
        }
    }

    /// Export to a file
    pub fn export_to_file(track: &CaptionTrack, path: &Path, format: CaptionFormat) -> Result<()> {
        let data = Self::export(track, format)?;
        std::fs::write(path, data)
            .map_err(|e| CaptionError::Export(format!("Failed to write file: {e}")))?;
        Ok(())
    }

    /// Auto-detect format from file extension
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

    /// Export with auto-detected format
    pub fn export_auto(track: &CaptionTrack, path: &Path) -> Result<()> {
        let format = Self::detect_format_from_extension(path).ok_or_else(|| {
            CaptionError::Export("Could not determine format from extension".to_string())
        })?;
        Self::export_to_file(track, path, format)
    }

    /// Batch export to multiple formats
    pub fn batch_export(
        track: &CaptionTrack,
        formats: &[CaptionFormat],
    ) -> Result<Vec<(CaptionFormat, Vec<u8>)>> {
        let mut results = Vec::new();
        for &format in formats {
            let data = Self::export(track, format)?;
            results.push((format, data));
        }
        Ok(results)
    }
}

/// Export options
#[derive(Debug, Clone)]
pub struct ExportOptions {
    /// Encoding (default: UTF-8)
    pub encoding: Encoding,
    /// Line ending style
    pub line_ending: LineEnding,
    /// Include BOM (byte order mark)
    pub include_bom: bool,
}

impl Default for ExportOptions {
    fn default() -> Self {
        Self {
            encoding: Encoding::Utf8,
            line_ending: LineEnding::Unix,
            include_bom: false,
        }
    }
}

/// Text encoding
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    /// UTF-8
    Utf8,
    /// UTF-16 LE
    Utf16Le,
    /// UTF-16 BE
    Utf16Be,
    /// Latin-1 (ISO-8859-1)
    Latin1,
}

/// Line ending style
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineEnding {
    /// Unix (LF)
    Unix,
    /// Windows (CRLF)
    Windows,
    /// Mac (CR)
    Mac,
}

impl LineEnding {
    /// Get the line ending bytes
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Unix => "\n",
            Self::Windows => "\r\n",
            Self::Mac => "\r",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Caption, Language, Timestamp};

    #[test]
    fn test_export_srt() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let data = Exporter::export(&track, CaptionFormat::Srt).expect("export should succeed");
        let text = String::from_utf8(data).expect("output should be valid UTF-8");
        assert!(text.contains("Test"));
        assert!(text.contains("-->"));
    }

    #[test]
    fn test_format_detection() {
        let path = Path::new("test.srt");
        let format = Exporter::detect_format_from_extension(path);
        assert_eq!(format, Some(CaptionFormat::Srt));

        let path = Path::new("test.vtt");
        let format = Exporter::detect_format_from_extension(path);
        assert_eq!(format, Some(CaptionFormat::WebVtt));
    }

    #[test]
    fn test_batch_export() {
        let mut track = CaptionTrack::new(Language::english());
        track
            .add_caption(Caption::new(
                Timestamp::from_secs(1),
                Timestamp::from_secs(3),
                "Test".to_string(),
            ))
            .expect("operation should succeed in test");

        let formats = vec![CaptionFormat::Srt, CaptionFormat::WebVtt];
        let results =
            Exporter::batch_export(&track, &formats).expect("batch export should succeed");
        assert_eq!(results.len(), 2);
    }
}
