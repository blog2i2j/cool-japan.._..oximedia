// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Codec compatibility matrix helpers.
//!
//! Provides a static lookup for validating codec/container compatibility
//! across the OxiMedia ecosystem. Only patent-free codecs from [`CodecId`]
//! are listed, consistent with OxiMedia's pure-Rust green-list policy.

use crate::types::CodecId;
use serde::{Deserialize, Serialize};

/// A codec/container compatibility entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompatEntry {
    /// Codec identifier.
    pub codec: CodecId,
    /// Container name (e.g. `"webm"`, `"mp4"`, `"mkv"`).
    pub container: String,
    /// Whether this combination is considered broadly compatible.
    pub compatible: bool,
}

/// Codec compatibility matrix.
pub struct CodecMatrix;

impl CodecMatrix {
    /// Return `true` if the given codec is compatible with the named container.
    ///
    /// Container names are matched case-insensitively by extension.
    #[must_use]
    pub fn is_compatible(codec: CodecId, container: &str) -> bool {
        let container = container.to_ascii_lowercase();
        match codec {
            CodecId::Vp8 => matches!(container.as_str(), "webm" | "mkv"),
            CodecId::Vp9 => matches!(container.as_str(), "webm" | "mkv"),
            CodecId::Av1 => matches!(container.as_str(), "webm" | "mkv" | "mp4" | "isobmff"),
            CodecId::Theora => matches!(container.as_str(), "ogg" | "ogv" | "mkv"),
            CodecId::H263 => matches!(container.as_str(), "3gp" | "mkv"),
            CodecId::Ffv1 => matches!(container.as_str(), "mkv" | "nut"),
            CodecId::RawVideo => matches!(container.as_str(), "y4m" | "mkv" | "avi"),
            CodecId::Opus => matches!(container.as_str(), "webm" | "mkv" | "ogg" | "opus"),
            CodecId::Vorbis => matches!(container.as_str(), "webm" | "mkv" | "ogg"),
            CodecId::Flac => matches!(
                container.as_str(),
                "flac" | "mkv" | "ogg" | "mp4" | "isobmff"
            ),
            CodecId::Mp3 => matches!(container.as_str(), "mp3" | "mkv" | "mp4"),
            CodecId::Pcm => matches!(container.as_str(), "wav" | "mkv" | "aiff"),
            // Image codecs — typically not in containers
            CodecId::JpegXl
            | CodecId::Dng
            | CodecId::WebP
            | CodecId::Gif
            | CodecId::Png
            | CodecId::Tiff
            | CodecId::OpenExr => matches!(container.as_str(), "mkv"),
            // Subtitle formats
            CodecId::WebVtt => matches!(container.as_str(), "webm" | "mkv"),
            CodecId::Ass | CodecId::Ssa | CodecId::Srt => {
                matches!(container.as_str(), "mkv" | "mp4")
            }
        }
    }

    /// Return all containers known to be compatible with the given codec.
    #[must_use]
    pub fn compatible_containers(codec: CodecId) -> &'static [&'static str] {
        match codec {
            CodecId::Vp8 => &["webm", "mkv"],
            CodecId::Vp9 => &["webm", "mkv"],
            CodecId::Av1 => &["webm", "mkv", "mp4"],
            CodecId::Theora => &["ogg", "ogv", "mkv"],
            CodecId::H263 => &["3gp", "mkv"],
            CodecId::Ffv1 => &["mkv", "nut"],
            CodecId::RawVideo => &["y4m", "mkv"],
            CodecId::Opus => &["webm", "mkv", "ogg", "opus"],
            CodecId::Vorbis => &["webm", "mkv", "ogg"],
            CodecId::Flac => &["flac", "mkv", "ogg"],
            CodecId::Mp3 => &["mp3", "mkv"],
            CodecId::Pcm => &["wav", "mkv", "aiff"],
            CodecId::WebVtt => &["webm", "mkv"],
            CodecId::Ass | CodecId::Ssa | CodecId::Srt => &["mkv", "mp4"],
            _ => &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vp9_webm_compatible() {
        assert!(CodecMatrix::is_compatible(CodecId::Vp9, "webm"));
    }

    #[test]
    fn test_vp9_mp4_incompatible() {
        assert!(!CodecMatrix::is_compatible(CodecId::Vp9, "mp4"));
    }

    #[test]
    fn test_av1_mp4_compatible() {
        assert!(CodecMatrix::is_compatible(CodecId::Av1, "mp4"));
    }

    #[test]
    fn test_opus_ogg_compatible() {
        assert!(CodecMatrix::is_compatible(CodecId::Opus, "ogg"));
    }

    #[test]
    fn test_case_insensitive() {
        assert!(CodecMatrix::is_compatible(CodecId::Vp8, "WebM"));
        assert!(CodecMatrix::is_compatible(CodecId::Vp8, "WEBM"));
    }

    #[test]
    fn test_compatible_containers_non_empty() {
        assert!(!CodecMatrix::compatible_containers(CodecId::Vp9).is_empty());
    }
}
