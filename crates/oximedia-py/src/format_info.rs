#![allow(dead_code)]
//! Media format information and capability queries for Python bindings.
//!
//! Provides data structures that describe supported container formats,
//! codec capabilities, and compatibility matrices that the Python layer
//! can expose as read-only objects.

use std::collections::HashMap;
use std::fmt;

/// Kind of media container.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerKind {
    /// Matroska / WebM family.
    Matroska,
    /// MPEG-4 Part 14.
    Mp4,
    /// Ogg container.
    Ogg,
    /// MPEG Transport Stream.
    MpegTs,
    /// Flash Video.
    Flv,
    /// Audio-only WAV.
    Wav,
    /// Raw bitstream (no container).
    Raw,
}

impl fmt::Display for ContainerKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Matroska => write!(f, "matroska"),
            Self::Mp4 => write!(f, "mp4"),
            Self::Ogg => write!(f, "ogg"),
            Self::MpegTs => write!(f, "mpegts"),
            Self::Flv => write!(f, "flv"),
            Self::Wav => write!(f, "wav"),
            Self::Raw => write!(f, "raw"),
        }
    }
}

/// Describes a container format and its capabilities.
#[derive(Debug, Clone)]
pub struct ContainerInfo {
    /// Container kind.
    pub kind: ContainerKind,
    /// Common file extensions (e.g. `["mkv", "webm"]`).
    pub extensions: Vec<String>,
    /// MIME type (e.g. `"video/webm"`).
    pub mime_type: String,
    /// Whether demuxing (reading) is supported.
    pub can_demux: bool,
    /// Whether muxing (writing) is supported.
    pub can_mux: bool,
    /// Maximum number of streams supported (0 = unlimited).
    pub max_streams: u32,
}

impl ContainerInfo {
    /// Create a new container info.
    pub fn new(kind: ContainerKind, mime: impl Into<String>) -> Self {
        Self {
            kind,
            extensions: Vec::new(),
            mime_type: mime.into(),
            can_demux: false,
            can_mux: false,
            max_streams: 0,
        }
    }

    /// Add an extension and return self.
    pub fn with_extension(mut self, ext: impl Into<String>) -> Self {
        self.extensions.push(ext.into());
        self
    }

    /// Set demux capability.
    pub fn with_demux(mut self, v: bool) -> Self {
        self.can_demux = v;
        self
    }

    /// Set mux capability.
    pub fn with_mux(mut self, v: bool) -> Self {
        self.can_mux = v;
        self
    }

    /// Set max streams.
    pub fn with_max_streams(mut self, n: u32) -> Self {
        self.max_streams = n;
        self
    }

    /// Returns `true` if both demux and mux are supported.
    #[must_use]
    pub fn is_fully_supported(&self) -> bool {
        self.can_demux && self.can_mux
    }

    /// Check whether a given file extension matches this container.
    #[must_use]
    pub fn matches_extension(&self, ext: &str) -> bool {
        let lower = ext.to_lowercase();
        self.extensions.iter().any(|e| e.to_lowercase() == lower)
    }
}

/// Capability level for a codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodecCapability {
    /// Decode only.
    DecodeOnly,
    /// Encode only.
    EncodeOnly,
    /// Both encode and decode.
    Full,
}

impl fmt::Display for CodecCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DecodeOnly => write!(f, "decode-only"),
            Self::EncodeOnly => write!(f, "encode-only"),
            Self::Full => write!(f, "full"),
        }
    }
}

/// Information about a supported codec.
#[derive(Debug, Clone)]
pub struct CodecFormatInfo {
    /// Codec short name.
    pub name: String,
    /// Whether this is video or audio.
    pub media_type: String,
    /// Capability level.
    pub capability: CodecCapability,
    /// Known compatible containers.
    pub compatible_containers: Vec<ContainerKind>,
    /// Human-readable description.
    pub description: String,
}

impl CodecFormatInfo {
    /// Create a new codec format info.
    pub fn new(
        name: impl Into<String>,
        media_type: impl Into<String>,
        capability: CodecCapability,
    ) -> Self {
        Self {
            name: name.into(),
            media_type: media_type.into(),
            capability,
            compatible_containers: Vec::new(),
            description: String::new(),
        }
    }

    /// Add a compatible container.
    pub fn with_container(mut self, c: ContainerKind) -> Self {
        self.compatible_containers.push(c);
        self
    }

    /// Set description.
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = d.into();
        self
    }

    /// Whether the codec supports decoding.
    #[must_use]
    pub fn can_decode(&self) -> bool {
        matches!(
            self.capability,
            CodecCapability::DecodeOnly | CodecCapability::Full
        )
    }

    /// Whether the codec supports encoding.
    #[must_use]
    pub fn can_encode(&self) -> bool {
        matches!(
            self.capability,
            CodecCapability::EncodeOnly | CodecCapability::Full
        )
    }

    /// Whether this codec is compatible with a specific container.
    #[must_use]
    pub fn is_compatible_with(&self, container: ContainerKind) -> bool {
        self.compatible_containers.contains(&container)
    }
}

/// Registry that holds all known format and codec information.
#[derive(Debug, Clone, Default)]
pub struct FormatRegistry {
    /// Container infos keyed by kind.
    containers: HashMap<String, ContainerInfo>,
    /// Codec infos keyed by name.
    codecs: HashMap<String, CodecFormatInfo>,
}

impl FormatRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a container.
    pub fn add_container(&mut self, info: ContainerInfo) {
        self.containers.insert(info.kind.to_string(), info);
    }

    /// Register a codec.
    pub fn add_codec(&mut self, info: CodecFormatInfo) {
        self.codecs.insert(info.name.clone(), info);
    }

    /// Look up a container by kind string.
    pub fn container(&self, kind: &str) -> Option<&ContainerInfo> {
        self.containers.get(kind)
    }

    /// Look up a codec by name.
    pub fn codec(&self, name: &str) -> Option<&CodecFormatInfo> {
        self.codecs.get(name)
    }

    /// List all registered container kind strings.
    pub fn container_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.containers.keys().cloned().collect();
        names.sort();
        names
    }

    /// List all registered codec names.
    pub fn codec_names(&self) -> Vec<String> {
        let mut names: Vec<String> = self.codecs.keys().cloned().collect();
        names.sort();
        names
    }

    /// Number of registered containers.
    pub fn container_count(&self) -> usize {
        self.containers.len()
    }

    /// Number of registered codecs.
    pub fn codec_count(&self) -> usize {
        self.codecs.len()
    }

    /// Find a container by file extension.
    pub fn find_by_extension(&self, ext: &str) -> Option<&ContainerInfo> {
        self.containers.values().find(|c| c.matches_extension(ext))
    }

    /// Build a default registry with OxiMedia's supported formats.
    pub fn default_registry() -> Self {
        let mut reg = Self::new();

        reg.add_container(
            ContainerInfo::new(ContainerKind::Matroska, "video/x-matroska")
                .with_extension("mkv")
                .with_extension("webm")
                .with_demux(true)
                .with_mux(true),
        );
        reg.add_container(
            ContainerInfo::new(ContainerKind::Ogg, "audio/ogg")
                .with_extension("ogg")
                .with_extension("ogv")
                .with_demux(true)
                .with_mux(true),
        );
        reg.add_container(
            ContainerInfo::new(ContainerKind::Wav, "audio/wav")
                .with_extension("wav")
                .with_demux(true)
                .with_mux(false),
        );

        reg.add_codec(
            CodecFormatInfo::new("av1", "video", CodecCapability::Full)
                .with_container(ContainerKind::Matroska)
                .with_container(ContainerKind::Mp4)
                .with_description("AV1 video codec"),
        );
        reg.add_codec(
            CodecFormatInfo::new("vp9", "video", CodecCapability::DecodeOnly)
                .with_container(ContainerKind::Matroska)
                .with_description("VP9 video codec"),
        );
        reg.add_codec(
            CodecFormatInfo::new("opus", "audio", CodecCapability::Full)
                .with_container(ContainerKind::Ogg)
                .with_container(ContainerKind::Matroska)
                .with_description("Opus audio codec"),
        );

        reg
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ContainerKind ──────────────────────────────────────────────────────

    #[test]
    fn test_container_kind_display() {
        assert_eq!(ContainerKind::Matroska.to_string(), "matroska");
        assert_eq!(ContainerKind::Mp4.to_string(), "mp4");
        assert_eq!(ContainerKind::Raw.to_string(), "raw");
    }

    // ── ContainerInfo ──────────────────────────────────────────────────────

    #[test]
    fn test_container_info_new() {
        let ci = ContainerInfo::new(ContainerKind::Ogg, "audio/ogg");
        assert_eq!(ci.kind, ContainerKind::Ogg);
        assert_eq!(ci.mime_type, "audio/ogg");
        assert!(!ci.can_demux);
    }

    #[test]
    fn test_container_info_builder() {
        let ci = ContainerInfo::new(ContainerKind::Matroska, "video/x-matroska")
            .with_extension("mkv")
            .with_extension("webm")
            .with_demux(true)
            .with_mux(true)
            .with_max_streams(32);
        assert_eq!(ci.extensions.len(), 2);
        assert!(ci.can_demux);
        assert!(ci.can_mux);
        assert_eq!(ci.max_streams, 32);
    }

    #[test]
    fn test_container_fully_supported() {
        let ci = ContainerInfo::new(ContainerKind::Matroska, "video/x-matroska")
            .with_demux(true)
            .with_mux(true);
        assert!(ci.is_fully_supported());
    }

    #[test]
    fn test_container_not_fully_supported() {
        let ci = ContainerInfo::new(ContainerKind::Wav, "audio/wav")
            .with_demux(true)
            .with_mux(false);
        assert!(!ci.is_fully_supported());
    }

    #[test]
    fn test_matches_extension_case_insensitive() {
        let ci =
            ContainerInfo::new(ContainerKind::Matroska, "video/x-matroska").with_extension("mkv");
        assert!(ci.matches_extension("mkv"));
        assert!(ci.matches_extension("MKV"));
        assert!(!ci.matches_extension("mp4"));
    }

    // ── CodecCapability ────────────────────────────────────────────────────

    #[test]
    fn test_codec_capability_display() {
        assert_eq!(CodecCapability::Full.to_string(), "full");
        assert_eq!(CodecCapability::DecodeOnly.to_string(), "decode-only");
    }

    // ── CodecFormatInfo ────────────────────────────────────────────────────

    #[test]
    fn test_codec_can_decode_full() {
        let c = CodecFormatInfo::new("av1", "video", CodecCapability::Full);
        assert!(c.can_decode());
        assert!(c.can_encode());
    }

    #[test]
    fn test_codec_decode_only() {
        let c = CodecFormatInfo::new("vp9", "video", CodecCapability::DecodeOnly);
        assert!(c.can_decode());
        assert!(!c.can_encode());
    }

    #[test]
    fn test_codec_encode_only() {
        let c = CodecFormatInfo::new("x", "video", CodecCapability::EncodeOnly);
        assert!(!c.can_decode());
        assert!(c.can_encode());
    }

    #[test]
    fn test_codec_is_compatible_with() {
        let c = CodecFormatInfo::new("av1", "video", CodecCapability::Full)
            .with_container(ContainerKind::Matroska);
        assert!(c.is_compatible_with(ContainerKind::Matroska));
        assert!(!c.is_compatible_with(ContainerKind::Wav));
    }

    // ── FormatRegistry ─────────────────────────────────────────────────────

    #[test]
    fn test_registry_empty() {
        let reg = FormatRegistry::new();
        assert_eq!(reg.container_count(), 0);
        assert_eq!(reg.codec_count(), 0);
    }

    #[test]
    fn test_registry_add_and_lookup() {
        let mut reg = FormatRegistry::new();
        reg.add_container(ContainerInfo::new(ContainerKind::Mp4, "video/mp4"));
        reg.add_codec(CodecFormatInfo::new("h264", "video", CodecCapability::Full));
        assert!(reg.container("mp4").is_some());
        assert!(reg.codec("h264").is_some());
    }

    #[test]
    fn test_registry_names_sorted() {
        let mut reg = FormatRegistry::new();
        reg.add_codec(CodecFormatInfo::new(
            "vp9",
            "video",
            CodecCapability::DecodeOnly,
        ));
        reg.add_codec(CodecFormatInfo::new("av1", "video", CodecCapability::Full));
        let names = reg.codec_names();
        assert_eq!(names, vec!["av1", "vp9"]);
    }

    #[test]
    fn test_default_registry() {
        let reg = FormatRegistry::default_registry();
        assert!(reg.container_count() >= 3);
        assert!(reg.codec_count() >= 3);
        assert!(reg.codec("av1").expect("codec should succeed").can_decode());
    }

    #[test]
    fn test_find_by_extension() {
        let reg = FormatRegistry::default_registry();
        let c = reg.find_by_extension("mkv").expect("c should be valid");
        assert_eq!(c.kind, ContainerKind::Matroska);
    }

    #[test]
    fn test_find_by_extension_not_found() {
        let reg = FormatRegistry::default_registry();
        assert!(reg.find_by_extension("xyz").is_none());
    }
}
