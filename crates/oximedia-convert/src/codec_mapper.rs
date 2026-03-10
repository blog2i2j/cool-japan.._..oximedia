//! Codec family grouping and container-codec compatibility mapping.
//!
//! `CodecMapper` answers questions like "which codecs can be placed in an MKV
//! container?" and "what is the best match for H.264 when targeting `WebM`?".

#![allow(dead_code)]

/// High-level family that groups related codecs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CodecFamily {
    /// H.264 / AVC and its profiles.
    H264,
    /// H.265 / HEVC.
    H265,
    /// AV1 (`AOMedia` Video 1).
    Av1,
    /// VP8 / VP9.
    Vpx,
    /// MPEG-2 Video.
    Mpeg2Video,
    /// `ProRes` family.
    ProRes,
    /// `DNxHD` / `DNxHR`.
    Dnx,
    /// AAC audio.
    Aac,
    /// MP3 (MPEG-1 Layer III) audio.
    Mp3,
    /// Opus audio.
    Opus,
    /// FLAC lossless audio.
    Flac,
    /// PCM / uncompressed audio.
    Pcm,
    /// Vorbis audio.
    Vorbis,
    /// AC-3 / Dolby Digital audio.
    Ac3,
    /// EAC-3 / Dolby Digital Plus.
    Eac3,
    /// Unknown / unlisted codec.
    Unknown,
}

impl CodecFamily {
    /// Returns `true` if this is a video codec.
    #[must_use]
    pub fn is_video(&self) -> bool {
        matches!(
            self,
            Self::H264
                | Self::H265
                | Self::Av1
                | Self::Vpx
                | Self::Mpeg2Video
                | Self::ProRes
                | Self::Dnx
        )
    }

    /// Returns `true` if this is an audio codec.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            Self::Aac
                | Self::Mp3
                | Self::Opus
                | Self::Flac
                | Self::Pcm
                | Self::Vorbis
                | Self::Ac3
                | Self::Eac3
        )
    }

    /// Returns `true` if this codec produces lossless output.
    #[must_use]
    pub fn is_lossless(&self) -> bool {
        matches!(self, Self::Flac | Self::Pcm | Self::ProRes | Self::Dnx)
    }

    /// Display name string.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::H264 => "H.264/AVC",
            Self::H265 => "H.265/HEVC",
            Self::Av1 => "AV1",
            Self::Vpx => "VP8/VP9",
            Self::Mpeg2Video => "MPEG-2 Video",
            Self::ProRes => "Apple ProRes",
            Self::Dnx => "Avid DNxHD/HR",
            Self::Aac => "AAC",
            Self::Mp3 => "MP3",
            Self::Opus => "Opus",
            Self::Flac => "FLAC",
            Self::Pcm => "PCM",
            Self::Vorbis => "Vorbis",
            Self::Ac3 => "AC-3",
            Self::Eac3 => "E-AC-3",
            Self::Unknown => "Unknown",
        }
    }
}

/// Describes the pairing of a source codec with a target codec for a specific
/// container format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodecMapping {
    /// Source codec family.
    pub source: CodecFamily,
    /// Recommended target codec family.
    pub target: CodecFamily,
    /// Target container format string (e.g. "mp4", "mkv").
    pub container: &'static str,
    /// Whether source and target are directly mux-compatible (no transcode).
    pub direct_copy: bool,
    /// Compatibility score 0..100 (higher = better fit).
    pub score: u8,
}

impl CodecMapping {
    /// Returns `true` if source and target are the same codec family (direct
    /// stream copy is possible) or the mapping is flagged as copy-compatible.
    #[must_use]
    pub fn is_compatible(&self) -> bool {
        self.direct_copy || self.source == self.target
    }

    /// Returns `true` if a transcode step is required.
    #[must_use]
    pub fn needs_transcode(&self) -> bool {
        !self.is_compatible()
    }
}

/// Maps source codecs to recommended targets given a destination container.
#[derive(Debug, Clone, Default)]
pub struct CodecMapper;

impl CodecMapper {
    /// Create a new mapper.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Return all registered `CodecMapping` entries in the built-in table.
    fn all_mappings(&self) -> Vec<CodecMapping> {
        vec![
            // ── MP4 mappings ─────────────────────────────────────────────
            CodecMapping {
                source: CodecFamily::H264,
                target: CodecFamily::H264,
                container: "mp4",
                direct_copy: true,
                score: 100,
            },
            CodecMapping {
                source: CodecFamily::H265,
                target: CodecFamily::H265,
                container: "mp4",
                direct_copy: true,
                score: 100,
            },
            CodecMapping {
                source: CodecFamily::Av1,
                target: CodecFamily::H264,
                container: "mp4",
                direct_copy: false,
                score: 70,
            },
            CodecMapping {
                source: CodecFamily::ProRes,
                target: CodecFamily::H264,
                container: "mp4",
                direct_copy: false,
                score: 80,
            },
            CodecMapping {
                source: CodecFamily::Aac,
                target: CodecFamily::Aac,
                container: "mp4",
                direct_copy: true,
                score: 100,
            },
            CodecMapping {
                source: CodecFamily::Mp3,
                target: CodecFamily::Aac,
                container: "mp4",
                direct_copy: false,
                score: 75,
            },
            // ── MKV mappings ─────────────────────────────────────────────
            CodecMapping {
                source: CodecFamily::H264,
                target: CodecFamily::H264,
                container: "mkv",
                direct_copy: true,
                score: 100,
            },
            CodecMapping {
                source: CodecFamily::Av1,
                target: CodecFamily::Av1,
                container: "mkv",
                direct_copy: true,
                score: 100,
            },
            CodecMapping {
                source: CodecFamily::Flac,
                target: CodecFamily::Flac,
                container: "mkv",
                direct_copy: true,
                score: 100,
            },
            CodecMapping {
                source: CodecFamily::Opus,
                target: CodecFamily::Opus,
                container: "mkv",
                direct_copy: true,
                score: 100,
            },
            // ── WebM mappings ─────────────────────────────────────────────
            CodecMapping {
                source: CodecFamily::H264,
                target: CodecFamily::Vpx,
                container: "webm",
                direct_copy: false,
                score: 70,
            },
            CodecMapping {
                source: CodecFamily::Av1,
                target: CodecFamily::Av1,
                container: "webm",
                direct_copy: true,
                score: 100,
            },
            CodecMapping {
                source: CodecFamily::Aac,
                target: CodecFamily::Opus,
                container: "webm",
                direct_copy: false,
                score: 80,
            },
            CodecMapping {
                source: CodecFamily::Vorbis,
                target: CodecFamily::Vorbis,
                container: "webm",
                direct_copy: true,
                score: 100,
            },
        ]
    }

    /// Look up the mapping for a specific `(source, container)` pair.
    /// Returns `None` if no entry exists.
    #[must_use]
    pub fn get_mapping(&self, source: CodecFamily, container: &str) -> Option<CodecMapping> {
        let container_lc = container.to_ascii_lowercase();
        self.all_mappings()
            .into_iter()
            .find(|m| m.source == source && m.container == container_lc.as_str())
    }

    /// Return the best-matching target `CodecFamily` for a given source and
    /// container, falling back to `CodecFamily::Unknown` if not found.
    #[must_use]
    pub fn best_match(&self, source: CodecFamily, container: &str) -> CodecFamily {
        self.get_mapping(source, container)
            .map_or(CodecFamily::Unknown, |m| m.target)
    }

    /// List all codec families that have at least one mapping for `container`.
    #[must_use]
    pub fn available_codecs(&self, container: &str) -> Vec<CodecFamily> {
        let container_lc = container.to_ascii_lowercase();
        let mut codecs: Vec<CodecFamily> = self
            .all_mappings()
            .into_iter()
            .filter(|m| m.container == container_lc.as_str())
            .map(|m| m.target)
            .collect();
        codecs.sort_by_key(CodecFamily::name);
        codecs.dedup_by_key(|c| c.name());
        codecs
    }

    /// Return all mappings for a given container.
    #[must_use]
    pub fn mappings_for_container(&self, container: &str) -> Vec<CodecMapping> {
        let container_lc = container.to_ascii_lowercase();
        self.all_mappings()
            .into_iter()
            .filter(|m| m.container == container_lc.as_str())
            .collect()
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn mapper() -> CodecMapper {
        CodecMapper::new()
    }

    #[test]
    fn test_codec_family_is_video() {
        assert!(CodecFamily::H264.is_video());
        assert!(CodecFamily::Av1.is_video());
        assert!(!CodecFamily::Aac.is_video());
    }

    #[test]
    fn test_codec_family_is_audio() {
        assert!(CodecFamily::Opus.is_audio());
        assert!(CodecFamily::Flac.is_audio());
        assert!(!CodecFamily::H265.is_audio());
    }

    #[test]
    fn test_codec_family_is_lossless() {
        assert!(CodecFamily::Flac.is_lossless());
        assert!(CodecFamily::Pcm.is_lossless());
        assert!(!CodecFamily::Aac.is_lossless());
        assert!(!CodecFamily::H264.is_lossless());
    }

    #[test]
    fn test_codec_family_name() {
        assert_eq!(CodecFamily::H264.name(), "H.264/AVC");
        assert_eq!(CodecFamily::Opus.name(), "Opus");
    }

    #[test]
    fn test_mapping_is_compatible_direct_copy() {
        let m = CodecMapping {
            source: CodecFamily::H264,
            target: CodecFamily::H264,
            container: "mp4",
            direct_copy: true,
            score: 100,
        };
        assert!(m.is_compatible());
        assert!(!m.needs_transcode());
    }

    #[test]
    fn test_mapping_is_compatible_transcode() {
        let m = CodecMapping {
            source: CodecFamily::ProRes,
            target: CodecFamily::H264,
            container: "mp4",
            direct_copy: false,
            score: 80,
        };
        assert!(!m.is_compatible());
        assert!(m.needs_transcode());
    }

    #[test]
    fn test_get_mapping_found() {
        let m = mapper();
        let mapping = m.get_mapping(CodecFamily::H264, "mp4");
        assert!(mapping.is_some());
        let mapping = mapping.expect("H264→mp4 mapping should exist");
        assert_eq!(mapping.target, CodecFamily::H264);
        assert!(mapping.direct_copy);
    }

    #[test]
    fn test_get_mapping_not_found() {
        let m = mapper();
        assert!(m.get_mapping(CodecFamily::Mpeg2Video, "webm").is_none());
    }

    #[test]
    fn test_best_match_known() {
        let m = mapper();
        assert_eq!(m.best_match(CodecFamily::H264, "webm"), CodecFamily::Vpx);
    }

    #[test]
    fn test_best_match_unknown_fallback() {
        let m = mapper();
        assert_eq!(
            m.best_match(CodecFamily::Mpeg2Video, "webm"),
            CodecFamily::Unknown
        );
    }

    #[test]
    fn test_available_codecs_mp4_non_empty() {
        let m = mapper();
        let codecs = m.available_codecs("mp4");
        assert!(!codecs.is_empty());
        assert!(codecs.contains(&CodecFamily::H264));
        assert!(codecs.contains(&CodecFamily::Aac));
    }

    #[test]
    fn test_available_codecs_unknown_container() {
        let m = mapper();
        let codecs = m.available_codecs("xyz");
        assert!(codecs.is_empty());
    }

    #[test]
    fn test_mappings_for_container_webm() {
        let m = mapper();
        let mappings = m.mappings_for_container("webm");
        assert!(!mappings.is_empty());
        assert!(mappings.iter().all(|m| m.container == "webm"));
    }

    #[test]
    fn test_case_insensitive_container_lookup() {
        let m = mapper();
        let a = m.available_codecs("MP4");
        let b = m.available_codecs("mp4");
        assert_eq!(a.len(), b.len());
    }
}
