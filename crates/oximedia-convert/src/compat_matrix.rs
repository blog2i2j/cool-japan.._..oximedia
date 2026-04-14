// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Container/codec compatibility matrix.
//!
//! Provides a static lookup for validating whether a given codec is compatible
//! with a target container format, based on widely-accepted media standards.

/// A static compatibility matrix for container/codec combinations.
///
/// Covers the most common patent-free and widely-supported combinations.
/// Rules are conservative: only combinations known to be broadly compatible
/// are listed as `true`.
pub struct ContainerCodecMatrix;

impl ContainerCodecMatrix {
    /// Return `true` if the given `codec` is compatible with `container`.
    ///
    /// Container names are matched case-insensitively by extension or common
    /// identifiers (e.g. `"webm"`, `"mkv"`, `"mp4"`, `"ogg"`, `"ts"`, `"mov"`).
    /// Codec names follow common short identifiers: `"vp8"`, `"vp9"`, `"av1"`,
    /// `"avc"` / `"h264"`, `"hevc"` / `"h265"`, `"opus"`, `"vorbis"`,
    /// `"flac"`, `"aac"`, `"mp3"`, `"pcm"`, `"theora"`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_convert::compat_matrix::ContainerCodecMatrix;
    ///
    /// assert!(ContainerCodecMatrix::is_compatible("webm", "vp9"));
    /// assert!(ContainerCodecMatrix::is_compatible("mkv", "av1"));
    /// assert!(!ContainerCodecMatrix::is_compatible("webm", "avc"));
    /// ```
    pub fn is_compatible(container: &str, codec: &str) -> bool {
        let c = container.to_ascii_lowercase();
        let k = codec.to_ascii_lowercase();
        match c.as_str() {
            // WebM: VP8, VP9, AV1 video; Opus, Vorbis audio
            "webm" => matches!(k.as_str(), "vp8" | "vp9" | "av1" | "opus" | "vorbis"),

            // Matroska: accepts virtually any modern codec
            "mkv" | "matroska" => matches!(
                k.as_str(),
                "vp8"
                    | "vp9"
                    | "av1"
                    | "avc"
                    | "h264"
                    | "hevc"
                    | "h265"
                    | "theora"
                    | "opus"
                    | "vorbis"
                    | "flac"
                    | "aac"
                    | "mp3"
                    | "pcm"
                    | "ffv1"
            ),

            // MP4 / M4A / M4V: H.264, H.265, AV1, AAC, MP3, AC-3
            "mp4" | "m4a" | "m4v" => matches!(
                k.as_str(),
                "avc" | "h264" | "hevc" | "h265" | "av1" | "aac" | "mp3" | "ac3" | "pcm"
            ),

            // QuickTime MOV: same family as MP4
            "mov" => matches!(
                k.as_str(),
                "avc" | "h264" | "hevc" | "h265" | "av1" | "prores" | "aac" | "pcm"
            ),

            // MPEG-TS / Transport Stream: broadcast standards
            "ts" | "m2ts" | "mts" => matches!(
                k.as_str(),
                "avc" | "h264" | "hevc" | "h265" | "mpeg2" | "aac" | "mp3" | "ac3" | "pcm"
            ),

            // Ogg: Theora, Vorbis, Opus, FLAC, Speex
            "ogg" | "ogv" | "oga" | "opus" => {
                matches!(k.as_str(), "theora" | "vorbis" | "opus" | "flac" | "speex")
            }

            // FLAC container
            "flac" => k == "flac",

            // WAV container
            "wav" => matches!(k.as_str(), "pcm" | "adpcm"),

            // AVI: legacy format, broad but not modern
            "avi" => matches!(
                k.as_str(),
                "avc" | "h264" | "mpeg2" | "mpeg4" | "pcm" | "mp3" | "ac3"
            ),

            // Unknown container: conservatively return false
            _ => false,
        }
    }

    /// Return all codecs known to be compatible with `container`.
    pub fn compatible_codecs(container: &str) -> Vec<&'static str> {
        let known_codecs = [
            "vp8", "vp9", "av1", "avc", "h264", "hevc", "h265", "theora", "mpeg2", "mpeg4",
            "prores", "ffv1", "opus", "vorbis", "flac", "aac", "mp3", "ac3", "pcm", "adpcm",
            "speex",
        ];
        known_codecs
            .iter()
            .copied()
            .filter(|&codec| Self::is_compatible(container, codec))
            .collect()
    }

    /// Return all containers known to accept `codec`.
    pub fn compatible_containers(codec: &str) -> Vec<&'static str> {
        let known_containers = [
            "webm", "mkv", "mp4", "mov", "ts", "ogg", "flac", "wav", "avi",
        ];
        known_containers
            .iter()
            .copied()
            .filter(|&container| Self::is_compatible(container, codec))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn webm_accepts_vp8_vp9_av1() {
        assert!(ContainerCodecMatrix::is_compatible("webm", "vp8"));
        assert!(ContainerCodecMatrix::is_compatible("webm", "vp9"));
        assert!(ContainerCodecMatrix::is_compatible("webm", "av1"));
    }

    #[test]
    fn webm_rejects_h264() {
        assert!(!ContainerCodecMatrix::is_compatible("webm", "avc"));
        assert!(!ContainerCodecMatrix::is_compatible("webm", "h264"));
        assert!(!ContainerCodecMatrix::is_compatible("webm", "hevc"));
    }

    #[test]
    fn mkv_accepts_any_common_codec() {
        assert!(ContainerCodecMatrix::is_compatible("mkv", "avc"));
        assert!(ContainerCodecMatrix::is_compatible("mkv", "vp9"));
        assert!(ContainerCodecMatrix::is_compatible("mkv", "av1"));
        assert!(ContainerCodecMatrix::is_compatible("mkv", "flac"));
        assert!(ContainerCodecMatrix::is_compatible("mkv", "opus"));
    }

    #[test]
    fn mp4_accepts_avc_hevc() {
        assert!(ContainerCodecMatrix::is_compatible("mp4", "avc"));
        assert!(ContainerCodecMatrix::is_compatible("mp4", "h264"));
        assert!(ContainerCodecMatrix::is_compatible("mp4", "hevc"));
        assert!(ContainerCodecMatrix::is_compatible("mp4", "h265"));
    }

    #[test]
    fn mp4_rejects_vp8() {
        assert!(!ContainerCodecMatrix::is_compatible("mp4", "vp8"));
    }

    #[test]
    fn case_insensitive() {
        assert!(ContainerCodecMatrix::is_compatible("WebM", "VP9"));
        assert!(ContainerCodecMatrix::is_compatible("MKV", "AV1"));
    }

    #[test]
    fn unknown_container_rejects_all() {
        assert!(!ContainerCodecMatrix::is_compatible("xyz", "vp9"));
    }

    #[test]
    fn compatible_codecs_webm() {
        let codecs = ContainerCodecMatrix::compatible_codecs("webm");
        assert!(codecs.contains(&"vp9"));
        assert!(codecs.contains(&"av1"));
        assert!(!codecs.contains(&"avc"));
    }

    #[test]
    fn compatible_containers_vp9() {
        let containers = ContainerCodecMatrix::compatible_containers("vp9");
        assert!(containers.contains(&"webm"));
        assert!(containers.contains(&"mkv"));
        assert!(!containers.contains(&"mp4"));
    }
}
