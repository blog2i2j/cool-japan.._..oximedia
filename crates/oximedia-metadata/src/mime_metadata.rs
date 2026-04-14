//! MIME type detection and metadata inference.
//!
//! # Overview
//!
//! This module bridges media metadata and MIME type classification.  It
//! provides:
//!
//! - **[`MimeType`]** – a typed MIME type enum covering audio, video, image,
//!   and container formats commonly handled by OxiMedia.
//! - **[`MimeDetector`]** – infers MIME type from magic-byte signatures embedded
//!   in the first bytes of a file or from metadata fields (codec, container,
//!   format hints).
//! - **[`MimeMetadataMapper`]** – maps a detected MIME type to the appropriate
//!   [`MetadataFormat`] and a set of expected/recommended metadata fields.
//! - **[`MimeHint`]** – structured output combining detected MIME type,
//!   confidence level, container format, and codec information extracted from
//!   metadata fields.
//!
//! # Pure Rust
//!
//! No C/Fortran dependencies; detection is purely byte-pattern matching and
//! string heuristics.

#![allow(dead_code)]

use crate::{Error, Metadata, MetadataFormat, MetadataValue};
use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// MimeType
// ---------------------------------------------------------------------------

/// A typed MIME type covering formats handled by OxiMedia.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MimeType {
    // --- Audio ---
    /// `audio/mpeg` (MP3)
    AudioMpeg,
    /// `audio/ogg` (Ogg Vorbis / Opus)
    AudioOgg,
    /// `audio/flac`
    AudioFlac,
    /// `audio/wav` / `audio/x-wav`
    AudioWav,
    /// `audio/aac`
    AudioAac,
    /// `audio/opus`
    AudioOpus,
    /// `audio/mp4` (M4A)
    AudioMp4,
    /// `audio/x-aiff`
    AudioAiff,

    // --- Video ---
    /// `video/mp4`
    VideoMp4,
    /// `video/webm`
    VideoWebm,
    /// `video/x-matroska` (MKV)
    VideoMatroska,
    /// `video/ogg`
    VideoOgg,
    /// `video/quicktime` (MOV)
    VideoQuicktime,
    /// `video/avi`
    VideoAvi,
    /// `video/x-msvideo` (alias for AVI)
    VideoXMsvideo,

    // --- Image ---
    /// `image/jpeg`
    ImageJpeg,
    /// `image/png`
    ImagePng,
    /// `image/gif`
    ImageGif,
    /// `image/webp`
    ImageWebp,
    /// `image/tiff`
    ImageTiff,
    /// `image/avif`
    ImageAvif,
    /// `image/jxl` (JPEG XL)
    ImageJxl,
    /// `image/x-adobe-dng`
    ImageDng,

    // --- Application ---
    /// `application/xml` / `text/xml` (XMP sidecar)
    ApplicationXml,

    // --- Unknown ---
    /// Unknown or unrecognised MIME type.
    Unknown,
}

impl MimeType {
    /// Return the canonical MIME type string.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AudioMpeg => "audio/mpeg",
            Self::AudioOgg => "audio/ogg",
            Self::AudioFlac => "audio/flac",
            Self::AudioWav => "audio/wav",
            Self::AudioAac => "audio/aac",
            Self::AudioOpus => "audio/opus",
            Self::AudioMp4 => "audio/mp4",
            Self::AudioAiff => "audio/x-aiff",
            Self::VideoMp4 => "video/mp4",
            Self::VideoWebm => "video/webm",
            Self::VideoMatroska => "video/x-matroska",
            Self::VideoOgg => "video/ogg",
            Self::VideoQuicktime => "video/quicktime",
            Self::VideoAvi => "video/avi",
            Self::VideoXMsvideo => "video/x-msvideo",
            Self::ImageJpeg => "image/jpeg",
            Self::ImagePng => "image/png",
            Self::ImageGif => "image/gif",
            Self::ImageWebp => "image/webp",
            Self::ImageTiff => "image/tiff",
            Self::ImageAvif => "image/avif",
            Self::ImageJxl => "image/jxl",
            Self::ImageDng => "image/x-adobe-dng",
            Self::ApplicationXml => "application/xml",
            Self::Unknown => "application/octet-stream",
        }
    }

    /// Whether this MIME type is in the `audio/*` top-level type.
    #[must_use]
    pub fn is_audio(self) -> bool {
        matches!(
            self,
            Self::AudioMpeg
                | Self::AudioOgg
                | Self::AudioFlac
                | Self::AudioWav
                | Self::AudioAac
                | Self::AudioOpus
                | Self::AudioMp4
                | Self::AudioAiff
        )
    }

    /// Whether this MIME type is in the `video/*` top-level type.
    #[must_use]
    pub fn is_video(self) -> bool {
        matches!(
            self,
            Self::VideoMp4
                | Self::VideoWebm
                | Self::VideoMatroska
                | Self::VideoOgg
                | Self::VideoQuicktime
                | Self::VideoAvi
                | Self::VideoXMsvideo
        )
    }

    /// Whether this MIME type is in the `image/*` top-level type.
    #[must_use]
    pub fn is_image(self) -> bool {
        matches!(
            self,
            Self::ImageJpeg
                | Self::ImagePng
                | Self::ImageGif
                | Self::ImageWebp
                | Self::ImageTiff
                | Self::ImageAvif
                | Self::ImageJxl
                | Self::ImageDng
        )
    }

    /// Parse a MIME type string into a [`MimeType`].  Returns
    /// [`MimeType::Unknown`] for unrecognised strings.
    #[must_use]
    pub fn from_str(s: &str) -> Self {
        let lower = s.to_lowercase();
        // Strip any parameters (e.g. "audio/mpeg; charset=utf-8").
        let base = lower.split(';').next().unwrap_or("").trim();
        match base {
            "audio/mpeg" | "audio/mp3" => Self::AudioMpeg,
            "audio/ogg" => Self::AudioOgg,
            "audio/flac" | "audio/x-flac" => Self::AudioFlac,
            "audio/wav" | "audio/x-wav" | "audio/vnd.wave" => Self::AudioWav,
            "audio/aac" | "audio/x-aac" => Self::AudioAac,
            "audio/opus" => Self::AudioOpus,
            "audio/mp4" | "audio/m4a" | "audio/x-m4a" => Self::AudioMp4,
            "audio/aiff" | "audio/x-aiff" => Self::AudioAiff,
            "video/mp4" => Self::VideoMp4,
            "video/webm" => Self::VideoWebm,
            "video/x-matroska" | "video/mkv" => Self::VideoMatroska,
            "video/ogg" | "video/x-ogg" => Self::VideoOgg,
            "video/quicktime" | "video/x-quicktime" => Self::VideoQuicktime,
            "video/avi" | "video/x-msvideo" => Self::VideoAvi,
            "image/jpeg" | "image/jpg" => Self::ImageJpeg,
            "image/png" => Self::ImagePng,
            "image/gif" => Self::ImageGif,
            "image/webp" => Self::ImageWebp,
            "image/tiff" | "image/x-tiff" => Self::ImageTiff,
            "image/avif" => Self::ImageAvif,
            "image/jxl" | "image/x-jxl" => Self::ImageJxl,
            "image/x-adobe-dng" | "image/dng" => Self::ImageDng,
            "application/xml" | "text/xml" => Self::ApplicationXml,
            _ => Self::Unknown,
        }
    }
}

impl fmt::Display for MimeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

// ---------------------------------------------------------------------------
// Confidence
// ---------------------------------------------------------------------------

/// Confidence level for a MIME type detection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// Detection is uncertain (fallback heuristic).
    Low,
    /// Reasonable match based on a partial signature or metadata field.
    Medium,
    /// Strong match from a known magic-byte signature.
    High,
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Low => write!(f, "low"),
            Self::Medium => write!(f, "medium"),
            Self::High => write!(f, "high"),
        }
    }
}

// ---------------------------------------------------------------------------
// MimeHint
// ---------------------------------------------------------------------------

/// Combined MIME detection result.
#[derive(Debug, Clone)]
pub struct MimeHint {
    /// Best guess for the MIME type.
    pub mime_type: MimeType,
    /// Confidence level.
    pub confidence: Confidence,
    /// Primary codec identifier derived from metadata, if any.
    pub codec: Option<String>,
    /// Container format name, if determinable.
    pub container: Option<String>,
    /// How the MIME type was determined.
    pub method: DetectionMethod,
}

/// How the MIME type was determined.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DetectionMethod {
    /// Magic bytes in the binary data.
    MagicBytes,
    /// Metadata field value (e.g., codec name, format tag).
    MetadataField(String),
    /// File extension heuristic (low confidence).
    Extension,
    /// Fallback / unknown.
    Fallback,
}

// ---------------------------------------------------------------------------
// MimeDetector
// ---------------------------------------------------------------------------

/// Detects MIME types from magic-byte signatures and metadata fields.
pub struct MimeDetector;

impl MimeDetector {
    /// Attempt to detect MIME type from the leading bytes of file data.
    ///
    /// Returns `None` if no known signature matches.
    #[must_use]
    pub fn from_bytes(data: &[u8]) -> Option<MimeHint> {
        // Need at least 12 bytes for the broadest checks.
        if data.len() < 4 {
            return None;
        }

        let magic = detect_magic(data);
        magic.map(|(mime, container)| MimeHint {
            mime_type: mime,
            confidence: Confidence::High,
            codec: None,
            container: Some(container.to_string()),
            method: DetectionMethod::MagicBytes,
        })
    }

    /// Infer MIME type from metadata fields (codec, format, container).
    ///
    /// Checks well-known field names used by various metadata formats.
    ///
    /// Returns `None` if no reliable inference is possible.
    #[must_use]
    pub fn from_metadata(metadata: &Metadata) -> Option<MimeHint> {
        // Codec fields to probe.
        const CODEC_FIELDS: &[&str] = &[
            "codec",
            "codec_name",
            "CODEC_NAME",
            "audio_codec",
            "video_codec",
            "format",
            "FORMAT",
            "container",
            "CONTAINER",
        ];

        for field in CODEC_FIELDS {
            if let Some(MetadataValue::Text(val)) = metadata.get(field) {
                if let Some(hint) = infer_from_codec_string(val, field) {
                    return Some(hint);
                }
            }
        }

        // Check MIME type fields directly.
        const MIME_FIELDS: &[&str] = &["mime_type", "MIME_TYPE", "content_type", "CONTENT_TYPE"];
        for field in MIME_FIELDS {
            if let Some(MetadataValue::Text(val)) = metadata.get(field) {
                let mime = MimeType::from_str(val);
                if mime != MimeType::Unknown {
                    return Some(MimeHint {
                        mime_type: mime,
                        confidence: Confidence::Medium,
                        codec: None,
                        container: None,
                        method: DetectionMethod::MetadataField((*field).to_string()),
                    });
                }
            }
        }

        None
    }

    /// Infer MIME type from a file extension string (e.g., `"mp3"` or `".mp3"`).
    #[must_use]
    pub fn from_extension(ext: &str) -> MimeHint {
        let ext_lower = ext.trim_start_matches('.').to_lowercase();
        let mime = match ext_lower.as_str() {
            "mp3" => MimeType::AudioMpeg,
            "ogg" => MimeType::AudioOgg,
            "flac" => MimeType::AudioFlac,
            "wav" | "wave" => MimeType::AudioWav,
            "aac" => MimeType::AudioAac,
            "opus" => MimeType::AudioOpus,
            "m4a" => MimeType::AudioMp4,
            "aif" | "aiff" => MimeType::AudioAiff,
            "mp4" | "m4v" => MimeType::VideoMp4,
            "webm" => MimeType::VideoWebm,
            "mkv" => MimeType::VideoMatroska,
            "ogv" => MimeType::VideoOgg,
            "mov" => MimeType::VideoQuicktime,
            "avi" => MimeType::VideoAvi,
            "jpg" | "jpeg" => MimeType::ImageJpeg,
            "png" => MimeType::ImagePng,
            "gif" => MimeType::ImageGif,
            "webp" => MimeType::ImageWebp,
            "tif" | "tiff" => MimeType::ImageTiff,
            "avif" => MimeType::ImageAvif,
            "jxl" => MimeType::ImageJxl,
            "dng" => MimeType::ImageDng,
            "xmp" | "xml" => MimeType::ApplicationXml,
            _ => MimeType::Unknown,
        };
        MimeHint {
            mime_type: mime,
            confidence: Confidence::Low,
            codec: None,
            container: None,
            method: DetectionMethod::Extension,
        }
    }

    /// Detect MIME type using all available strategies, returning the highest-
    /// confidence result.
    ///
    /// Strategy priority: magic bytes > metadata fields > extension.
    #[must_use]
    pub fn detect(
        data: Option<&[u8]>,
        metadata: Option<&Metadata>,
        extension: Option<&str>,
    ) -> MimeHint {
        // Try magic bytes first (highest confidence).
        if let Some(bytes) = data {
            if let Some(hint) = Self::from_bytes(bytes) {
                return hint;
            }
        }

        // Try metadata fields.
        if let Some(meta) = metadata {
            if let Some(hint) = Self::from_metadata(meta) {
                return hint;
            }
        }

        // Fall back to extension.
        if let Some(ext) = extension {
            let hint = Self::from_extension(ext);
            if hint.mime_type != MimeType::Unknown {
                return hint;
            }
        }

        // Give up.
        MimeHint {
            mime_type: MimeType::Unknown,
            confidence: Confidence::Low,
            codec: None,
            container: None,
            method: DetectionMethod::Fallback,
        }
    }
}

// ---------------------------------------------------------------------------
// MimeMetadataMapper
// ---------------------------------------------------------------------------

/// Maps a [`MimeType`] to the preferred [`MetadataFormat`] and a list of
/// recommended metadata fields.
#[derive(Debug, Clone)]
pub struct MimeMetadataMapper;

impl MimeMetadataMapper {
    /// Return the preferred [`MetadataFormat`] for writing tags to the given
    /// MIME type.
    ///
    /// Returns `None` for MIME types that have no standard embedded tag format
    /// (e.g., raw XML sidecar).
    #[must_use]
    pub fn preferred_format(mime: MimeType) -> Option<MetadataFormat> {
        match mime {
            MimeType::AudioMpeg => Some(MetadataFormat::Id3v2),
            MimeType::AudioOgg | MimeType::AudioFlac | MimeType::AudioOpus => {
                Some(MetadataFormat::VorbisComments)
            }
            MimeType::AudioWav => Some(MetadataFormat::Id3v2),
            MimeType::AudioMp4 | MimeType::VideoMp4 | MimeType::VideoQuicktime => {
                Some(MetadataFormat::iTunes)
            }
            MimeType::VideoMatroska | MimeType::VideoWebm => Some(MetadataFormat::Matroska),
            MimeType::ImageJpeg | MimeType::ImageTiff | MimeType::ImageDng => {
                Some(MetadataFormat::Exif)
            }
            MimeType::ApplicationXml => Some(MetadataFormat::Xmp),
            _ => None,
        }
    }

    /// Return a list of recommended metadata field names for the MIME type.
    ///
    /// The returned field names are *suggested* keys; the actual tag names
    /// depend on the target format.
    #[must_use]
    pub fn recommended_fields(mime: MimeType) -> Vec<&'static str> {
        if mime.is_audio() {
            vec![
                "title",
                "artist",
                "album",
                "track_number",
                "genre",
                "date",
                "comment",
                "album_artist",
                "disc_number",
                "composer",
                "lyricist",
                "bpm",
                "isrc",
                "replaygain_track_gain",
                "replaygain_album_gain",
            ]
        } else if mime.is_video() {
            vec![
                "title",
                "comment",
                "date",
                "language",
                "encoder",
                "duration",
                "video_codec",
                "audio_codec",
                "width",
                "height",
                "frame_rate",
            ]
        } else if mime.is_image() {
            vec![
                "title",
                "description",
                "author",
                "copyright",
                "date_time_original",
                "make",
                "model",
                "gps_latitude",
                "gps_longitude",
                "software",
                "color_space",
                "width",
                "height",
            ]
        } else {
            vec![]
        }
    }

    /// Validate that a [`Metadata`] record contains the minimum required fields
    /// for the given MIME type.
    ///
    /// Returns a list of missing recommended fields.  An empty return value
    /// means all recommended fields are present.
    #[must_use]
    pub fn missing_fields(mime: MimeType, metadata: &Metadata) -> Vec<&'static str> {
        Self::recommended_fields(mime)
            .into_iter()
            .filter(|field| !metadata.contains(field))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Attempt to identify a MIME type from magic bytes.
///
/// Returns `(MimeType, container_name)` on success.
fn detect_magic(data: &[u8]) -> Option<(MimeType, &'static str)> {
    // Helper: check slice prefix.
    let starts_with = |magic: &[u8]| data.len() >= magic.len() && data[..magic.len()] == *magic;

    // JPEG: FF D8 FF
    if starts_with(&[0xFF, 0xD8, 0xFF]) {
        return Some((MimeType::ImageJpeg, "JPEG"));
    }
    // PNG: 89 50 4E 47 0D 0A 1A 0A
    if starts_with(&[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]) {
        return Some((MimeType::ImagePng, "PNG"));
    }
    // GIF: 47 49 46 38
    if starts_with(b"GIF8") {
        return Some((MimeType::ImageGif, "GIF"));
    }
    // RIFF/WAVE: 52 49 46 46 ... 57 41 56 45
    if starts_with(b"RIFF") && data.len() >= 12 && &data[8..12] == b"WAVE" {
        return Some((MimeType::AudioWav, "WAVE"));
    }
    // RIFF/AVI
    if starts_with(b"RIFF") && data.len() >= 12 && &data[8..12] == b"AVI " {
        return Some((MimeType::VideoAvi, "AVI"));
    }
    // RIFF/WEBP
    if starts_with(b"RIFF") && data.len() >= 12 && &data[8..12] == b"WEBP" {
        return Some((MimeType::ImageWebp, "WebP"));
    }
    // FLAC: 66 4C 61 43
    if starts_with(b"fLaC") {
        return Some((MimeType::AudioFlac, "FLAC"));
    }
    // OGG: 4F 67 67 53
    if starts_with(b"OggS") {
        return Some((MimeType::AudioOgg, "Ogg"));
    }
    // TIFF LE: 49 49 2A 00
    if starts_with(&[0x49, 0x49, 0x2A, 0x00]) {
        return Some((MimeType::ImageTiff, "TIFF"));
    }
    // TIFF BE: 4D 4D 00 2A
    if starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
        return Some((MimeType::ImageTiff, "TIFF"));
    }
    // EBML/Matroska: 1A 45 DF A3
    if starts_with(&[0x1A, 0x45, 0xDF, 0xA3]) {
        return Some((MimeType::VideoMatroska, "Matroska/WebM"));
    }
    // ID3 (MP3 with ID3 header): 49 44 33
    if starts_with(b"ID3") {
        return Some((MimeType::AudioMpeg, "MP3/ID3"));
    }
    // MP3 sync word: FF FB / FF FA / FF F3
    if data.len() >= 2 && data[0] == 0xFF && (data[1] & 0xE0 == 0xE0) {
        return Some((MimeType::AudioMpeg, "MP3"));
    }
    // ftyp box (MP4/MOV/M4A): bytes 4-7 == "ftyp"
    if data.len() >= 8 && &data[4..8] == b"ftyp" {
        // Sub-brand discrimination.
        let subtype = if data.len() >= 12 {
            &data[8..12]
        } else {
            b"    "
        };
        let mime = match subtype {
            b"M4A " | b"M4B " | b"M4P " => MimeType::AudioMp4,
            b"qt  " => MimeType::VideoQuicktime,
            _ => MimeType::VideoMp4,
        };
        return Some((mime, "ISOBMFF"));
    }
    // AIFF: 46 4F 52 4D ... 41 49 46 46
    if starts_with(b"FORM") && data.len() >= 12 && &data[8..12] == b"AIFF" {
        return Some((MimeType::AudioAiff, "AIFF"));
    }
    // XML / XMP sidecar: starts with "<?xml" or "<x:xmpmeta"
    if starts_with(b"<?xml") || starts_with(b"<x:xmpmeta") {
        return Some((MimeType::ApplicationXml, "XML"));
    }
    // JPEG XL: FF 0A (naked codestream) or 0000000C 4A584C20 (ISOBMFF)
    if starts_with(&[0xFF, 0x0A]) {
        return Some((MimeType::ImageJxl, "JXL"));
    }
    if data.len() >= 12 && &data[4..8] == b"JXL " {
        return Some((MimeType::ImageJxl, "JXL-ISOBMFF"));
    }

    None
}

/// Infer MIME type from a codec / format string found in a metadata field.
fn infer_from_codec_string(value: &str, field: &str) -> Option<MimeHint> {
    let lower = value.to_lowercase();
    let mime = match lower.as_str() {
        "mp3" | "mpeg" | "mpeg1audio" | "mpeg-1 audio" => MimeType::AudioMpeg,
        "flac" => MimeType::AudioFlac,
        "vorbis" | "ogg" | "ogg/vorbis" => MimeType::AudioOgg,
        "opus" | "ogg/opus" => MimeType::AudioOpus,
        "wav" | "pcm" | "lpcm" => MimeType::AudioWav,
        "aac" | "mpeg-4 aac" => MimeType::AudioAac,
        "m4a" | "alac" => MimeType::AudioMp4,
        "aiff" | "aif" => MimeType::AudioAiff,
        "h264" | "avc" | "h.264" | "mp4" | "mpeg-4" => MimeType::VideoMp4,
        "vp8" | "vp9" | "av1" | "webm" => MimeType::VideoWebm,
        "matroska" | "mkv" | "hevc" | "h265" => MimeType::VideoMatroska,
        "mov" | "quicktime" => MimeType::VideoQuicktime,
        "jpeg" | "jpg" => MimeType::ImageJpeg,
        "png" => MimeType::ImagePng,
        "webp" => MimeType::ImageWebp,
        "tiff" | "tif" => MimeType::ImageTiff,
        "avif" => MimeType::ImageAvif,
        "jxl" | "jpeg xl" => MimeType::ImageJxl,
        "dng" => MimeType::ImageDng,
        _ => return None,
    };
    Some(MimeHint {
        mime_type: mime,
        confidence: Confidence::Medium,
        codec: Some(value.to_string()),
        container: None,
        method: DetectionMethod::MetadataField(field.to_string()),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Metadata, MetadataFormat, MetadataValue};

    #[test]
    fn test_mime_type_classification() {
        assert!(MimeType::AudioMpeg.is_audio());
        assert!(MimeType::AudioFlac.is_audio());
        assert!(!MimeType::AudioMpeg.is_video());
        assert!(MimeType::VideoMp4.is_video());
        assert!(MimeType::ImageJpeg.is_image());
        assert!(!MimeType::ImageJpeg.is_audio());
    }

    #[test]
    fn test_mime_type_from_str_roundtrip() {
        let types = [
            MimeType::AudioMpeg,
            MimeType::AudioFlac,
            MimeType::VideoMp4,
            MimeType::ImageJpeg,
            MimeType::ImagePng,
        ];
        for mime in &types {
            let parsed = MimeType::from_str(mime.as_str());
            assert_eq!(parsed, *mime, "Round-trip failed for {}", mime.as_str());
        }
    }

    #[test]
    fn test_detect_magic_jpeg() {
        let jpeg_magic = [0xFF_u8, 0xD8, 0xFF, 0xE0, 0x00, 0x10];
        let hint = MimeDetector::from_bytes(&jpeg_magic).expect("Should detect JPEG");
        assert_eq!(hint.mime_type, MimeType::ImageJpeg);
        assert_eq!(hint.confidence, Confidence::High);
        assert_eq!(hint.method, DetectionMethod::MagicBytes);
    }

    #[test]
    fn test_detect_magic_flac() {
        let flac_magic = b"fLaC\x00\x00\x00\x22";
        let hint = MimeDetector::from_bytes(flac_magic).expect("Should detect FLAC");
        assert_eq!(hint.mime_type, MimeType::AudioFlac);
    }

    #[test]
    fn test_detect_magic_mp3_id3() {
        let id3_magic = b"ID3\x04\x00\x00\x00\x00\x00\x00";
        let hint = MimeDetector::from_bytes(id3_magic).expect("Should detect MP3/ID3");
        assert_eq!(hint.mime_type, MimeType::AudioMpeg);
    }

    #[test]
    fn test_detect_magic_wav() {
        let wav = b"RIFF\x00\x00\x00\x00WAVE";
        let hint = MimeDetector::from_bytes(wav).expect("Should detect WAV");
        assert_eq!(hint.mime_type, MimeType::AudioWav);
    }

    #[test]
    fn test_detect_from_extension() {
        let mp3 = MimeDetector::from_extension("mp3");
        assert_eq!(mp3.mime_type, MimeType::AudioMpeg);

        let flac = MimeDetector::from_extension(".flac");
        assert_eq!(flac.mime_type, MimeType::AudioFlac);

        let unknown = MimeDetector::from_extension("xyz123");
        assert_eq!(unknown.mime_type, MimeType::Unknown);
    }

    #[test]
    fn test_detect_from_metadata_field() {
        let mut meta = Metadata::new(MetadataFormat::VorbisComments);
        meta.insert("codec".to_string(), MetadataValue::Text("flac".to_string()));
        let hint = MimeDetector::from_metadata(&meta).expect("Should detect from metadata");
        assert_eq!(hint.mime_type, MimeType::AudioFlac);
        assert_eq!(hint.confidence, Confidence::Medium);
    }

    #[test]
    fn test_detect_from_metadata_mime_field() {
        let mut meta = Metadata::new(MetadataFormat::VorbisComments);
        meta.insert("mime_type".to_string(), MetadataValue::Text("video/mp4".to_string()));
        let hint = MimeDetector::from_metadata(&meta).expect("Should detect from mime_type field");
        assert_eq!(hint.mime_type, MimeType::VideoMp4);
    }

    #[test]
    fn test_detect_priority_magic_over_extension() {
        // JPEG magic but extension says .flac — magic should win.
        let jpeg_magic = vec![0xFF_u8, 0xD8, 0xFF, 0xE0, 0x00, 0x00, 0x00, 0x00];
        let hint = MimeDetector::detect(Some(&jpeg_magic), None, Some("flac"));
        assert_eq!(hint.mime_type, MimeType::ImageJpeg);
        assert_eq!(hint.method, DetectionMethod::MagicBytes);
    }

    #[test]
    fn test_preferred_format_mapping() {
        assert_eq!(
            MimeMetadataMapper::preferred_format(MimeType::AudioMpeg),
            Some(MetadataFormat::Id3v2)
        );
        assert_eq!(
            MimeMetadataMapper::preferred_format(MimeType::AudioFlac),
            Some(MetadataFormat::VorbisComments)
        );
        assert_eq!(
            MimeMetadataMapper::preferred_format(MimeType::VideoMp4),
            Some(MetadataFormat::iTunes)
        );
        assert_eq!(
            MimeMetadataMapper::preferred_format(MimeType::Unknown),
            None
        );
    }

    #[test]
    fn test_recommended_fields_non_empty() {
        assert!(!MimeMetadataMapper::recommended_fields(MimeType::AudioMpeg).is_empty());
        assert!(!MimeMetadataMapper::recommended_fields(MimeType::VideoMp4).is_empty());
        assert!(!MimeMetadataMapper::recommended_fields(MimeType::ImageJpeg).is_empty());
    }

    #[test]
    fn test_missing_fields() {
        let mut meta = Metadata::new(MetadataFormat::VorbisComments);
        meta.insert("title".to_string(), MetadataValue::Text("Song".to_string()));
        meta.insert("artist".to_string(), MetadataValue::Text("Alice".to_string()));

        let missing = MimeMetadataMapper::missing_fields(MimeType::AudioMpeg, &meta);
        // "title" and "artist" are recommended; they're present, so should not be in missing.
        assert!(!missing.contains(&"title"));
        assert!(!missing.contains(&"artist"));
        // "album" is recommended but absent.
        assert!(missing.contains(&"album"));
    }

    #[test]
    fn test_mime_type_display() {
        assert_eq!(MimeType::AudioMpeg.to_string(), "audio/mpeg");
        assert_eq!(MimeType::VideoWebm.to_string(), "video/webm");
        assert_eq!(MimeType::ImagePng.to_string(), "image/png");
    }

    #[test]
    fn test_detect_matroska_magic() {
        let mkv = [0x1A_u8, 0x45, 0xDF, 0xA3, 0x01, 0x00, 0x00, 0x00];
        let hint = MimeDetector::from_bytes(&mkv).expect("Should detect Matroska");
        assert_eq!(hint.mime_type, MimeType::VideoMatroska);
    }
}
