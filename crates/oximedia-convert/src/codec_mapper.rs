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
            CodecMapping {
                source: CodecFamily::Opus,
                target: CodecFamily::Opus,
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

    /// Validate the compatibility of a proposed codec/container combination and
    /// return a structured [`CompatibilityReport`].
    ///
    /// The report includes:
    /// - Whether the combination is directly compatible (no transcode needed).
    /// - A recommended target codec when transcoding is required.
    /// - Human-readable issues and suggestions.
    #[must_use]
    pub fn validate_compatibility(
        &self,
        source: CodecFamily,
        container: &str,
    ) -> CompatibilityReport {
        let container_lc = container.to_ascii_lowercase();
        let mapping = self.get_mapping(source, container);

        let is_compatible;
        let recommended_codec;
        let mut issues: Vec<String> = Vec::new();
        let mut suggestions: Vec<String> = Vec::new();

        if let Some(ref m) = mapping {
            is_compatible = m.is_compatible();
            recommended_codec = m.target;
            if m.needs_transcode() {
                issues.push(format!(
                    "Codec '{}' cannot be directly placed in '{}'; transcode required.",
                    source.name(),
                    container_lc,
                ));
                suggestions.push(format!(
                    "Transcode to '{}' for best compatibility (score {}/100).",
                    recommended_codec.name(),
                    m.score,
                ));
            }
            if m.score < 80 && !m.direct_copy {
                suggestions.push(format!(
                    "Compatibility score is {}/100; consider a higher-compatibility codec.",
                    m.score,
                ));
            }
        } else {
            is_compatible = false;
            recommended_codec = CodecFamily::Unknown;
            issues.push(format!(
                "No known mapping for codec '{}' → container '{}'. \
                 This combination may be unsupported.",
                source.name(),
                container_lc,
            ));
            let alternatives = self.available_codecs(container);
            if alternatives.is_empty() {
                issues.push(format!(
                    "Container '{}' has no registered codec mappings.",
                    container_lc
                ));
            } else {
                let alt_names: Vec<&str> = alternatives.iter().map(CodecFamily::name).collect();
                suggestions.push(format!(
                    "Supported codecs for '{}': {}.",
                    container_lc,
                    alt_names.join(", ")
                ));
            }
        }

        // Lossless-in-lossy-container advisory.
        if source.is_lossless() && matches!(container_lc.as_str(), "mp4" | "webm") {
            suggestions.push(format!(
                "Lossless codec '{}' in '{}' may produce very large files; \
                 consider a lossless container such as 'mkv'.",
                source.name(),
                container_lc,
            ));
        }

        // Video-in-audio-only container advisory.
        if source.is_video() && matches!(container_lc.as_str(), "ogg" | "flac") {
            issues.push(format!(
                "Video codec '{}' is not suitable for audio-only container '{}'.",
                source.name(),
                container_lc,
            ));
        }

        CompatibilityReport {
            source_codec: source,
            container: container_lc,
            is_compatible,
            recommended_codec,
            issues,
            suggestions,
        }
    }
}

/// A structured report describing the compatibility of a codec/container pair.
#[derive(Debug, Clone)]
pub struct CompatibilityReport {
    /// The source codec family that was evaluated.
    pub source_codec: CodecFamily,
    /// The target container (lowercase extension string).
    pub container: String,
    /// Whether the codec can be placed directly in the container without
    /// re-encoding (`true`) or a transcode is required (`false`).
    pub is_compatible: bool,
    /// The recommended codec family for the target container.
    /// `CodecFamily::Unknown` when no mapping exists.
    pub recommended_codec: CodecFamily,
    /// Human-readable compatibility issues (empty means no issues).
    pub issues: Vec<String>,
    /// Human-readable suggestions for resolving issues or improving quality.
    pub suggestions: Vec<String>,
}

impl CompatibilityReport {
    /// Returns `true` if there are no compatibility issues.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.issues.is_empty()
    }

    /// Returns the number of detected issues.
    #[must_use]
    pub fn issue_count(&self) -> usize {
        self.issues.len()
    }
}

// ── Severity / Findings ───────────────────────────────────────────────────────

/// Severity level attached to a compatibility finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum IssueSeverity {
    /// Informational note; no action required.
    Info,
    /// A potential problem that may affect quality or file size.
    Warning,
    /// A hard incompatibility that will cause the conversion to fail or produce
    /// an invalid output without intervention.
    Error,
}

impl IssueSeverity {
    /// Short string tag for display.
    #[must_use]
    pub const fn tag(&self) -> &'static str {
        match self {
            Self::Info => "INFO",
            Self::Warning => "WARN",
            Self::Error => "ERROR",
        }
    }
}

/// A single annotated compatibility finding.
#[derive(Debug, Clone)]
pub struct CompatibilityFinding {
    /// Codec that the finding applies to.
    pub codec: CodecFamily,
    /// Severity of the finding.
    pub severity: IssueSeverity,
    /// Human-readable message.
    pub message: String,
    /// Optional suggested action.
    pub suggestion: Option<String>,
}

impl CompatibilityFinding {
    /// Construct an error-level finding.
    #[must_use]
    pub fn error(codec: CodecFamily, msg: impl Into<String>) -> Self {
        Self {
            codec,
            severity: IssueSeverity::Error,
            message: msg.into(),
            suggestion: None,
        }
    }

    /// Construct a warning-level finding.
    #[must_use]
    pub fn warning(codec: CodecFamily, msg: impl Into<String>) -> Self {
        Self {
            codec,
            severity: IssueSeverity::Warning,
            message: msg.into(),
            suggestion: None,
        }
    }

    /// Construct an informational finding.
    #[must_use]
    pub fn info(codec: CodecFamily, msg: impl Into<String>) -> Self {
        Self {
            codec,
            severity: IssueSeverity::Info,
            message: msg.into(),
            suggestion: None,
        }
    }

    /// Attach a suggestion to the finding.
    #[must_use]
    pub fn with_suggestion(mut self, suggestion: impl Into<String>) -> Self {
        self.suggestion = Some(suggestion.into());
        self
    }
}

/// Summary produced by validating a complete set of codecs against a container.
#[derive(Debug, Clone)]
pub struct CodecValidationSummary {
    /// Target container format (lowercase).
    pub container: String,
    /// All findings, ordered by severity (errors first).
    pub findings: Vec<CompatibilityFinding>,
    /// Codecs that are directly compatible (no transcoding required).
    pub compatible_codecs: Vec<CodecFamily>,
    /// Codecs that require transcoding, paired with their recommended target.
    pub transcode_required: Vec<(CodecFamily, CodecFamily)>,
    /// Codecs with no known mapping for the target container.
    pub unmapped_codecs: Vec<CodecFamily>,
}

impl CodecValidationSummary {
    /// Returns `true` when there are no error-level findings.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        !self
            .findings
            .iter()
            .any(|f| f.severity == IssueSeverity::Error)
    }

    /// Number of error-level findings.
    #[must_use]
    pub fn error_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == IssueSeverity::Error)
            .count()
    }

    /// Number of warning-level findings.
    #[must_use]
    pub fn warning_count(&self) -> usize {
        self.findings
            .iter()
            .filter(|f| f.severity == IssueSeverity::Warning)
            .count()
    }

    /// Returns `true` when at least one codec is directly compatible.
    #[must_use]
    pub fn has_compatible_codecs(&self) -> bool {
        !self.compatible_codecs.is_empty()
    }

    /// Returns findings whose severity is at least `min_severity`.
    #[must_use]
    pub fn findings_at_least(&self, min_severity: IssueSeverity) -> Vec<&CompatibilityFinding> {
        self.findings
            .iter()
            .filter(|f| f.severity >= min_severity)
            .collect()
    }
}

/// Validates a complete set of codecs against a single target container.
///
/// Unlike [`CodecMapper::validate_compatibility`] — which operates on one
/// `(source, container)` pair — `MultiCodecValidator` batches the operation
/// across an entire set of codecs and produces a unified
/// [`CodecValidationSummary`].
///
/// # Usage
///
/// ```
/// use oximedia_convert::codec_mapper::{MultiCodecValidator, CodecFamily};
///
/// let v = MultiCodecValidator::new();
/// let summary = v.validate(&[CodecFamily::H264, CodecFamily::Aac], "webm");
/// // H264 → WebM needs transcode (warning level)
/// assert!(summary.warning_count() > 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MultiCodecValidator {
    mapper: CodecMapper,
}

impl MultiCodecValidator {
    /// Create a new validator backed by a default `CodecMapper`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Validate `codecs` against `container` and return a unified summary.
    #[must_use]
    pub fn validate(&self, codecs: &[CodecFamily], container: &str) -> CodecValidationSummary {
        let container_lc = container.to_ascii_lowercase();
        let mut findings: Vec<CompatibilityFinding> = Vec::new();
        let mut compatible_codecs: Vec<CodecFamily> = Vec::new();
        let mut transcode_required: Vec<(CodecFamily, CodecFamily)> = Vec::new();
        let mut unmapped_codecs: Vec<CodecFamily> = Vec::new();

        for &codec in codecs {
            let report = self.mapper.validate_compatibility(codec, &container_lc);

            if report.recommended_codec == CodecFamily::Unknown && !report.is_compatible {
                unmapped_codecs.push(codec);
                let mut finding = CompatibilityFinding::error(
                    codec,
                    format!(
                        "Codec '{}' has no known mapping for container '{container_lc}'.",
                        codec.name()
                    ),
                );
                if let Some(first) = report.suggestions.first() {
                    finding = finding.with_suggestion(first.clone());
                }
                findings.push(finding);
            } else if report.is_compatible {
                compatible_codecs.push(codec);
                for s in &report.suggestions {
                    findings.push(CompatibilityFinding::info(codec, s.clone()));
                }
            } else {
                transcode_required.push((codec, report.recommended_codec));
                let msg = format!(
                    "Codec '{}' requires transcoding to '{}' for '{container_lc}'.",
                    codec.name(),
                    report.recommended_codec.name(),
                );
                let mut finding = CompatibilityFinding::warning(codec, msg);
                if let Some(first) = report.issues.first() {
                    finding = finding.with_suggestion(first.clone());
                }
                findings.push(finding);
                for s in &report.suggestions {
                    findings.push(CompatibilityFinding::info(codec, s.clone()));
                }
            }
        }

        // Sort: errors first, then warnings, then info.
        findings.sort_by(|a, b| b.severity.cmp(&a.severity));

        CodecValidationSummary {
            container: container_lc,
            findings,
            compatible_codecs,
            transcode_required,
            unmapped_codecs,
        }
    }

    /// Check whether a single codec is directly compatible with a container.
    #[must_use]
    pub fn is_direct_compatible(&self, codec: CodecFamily, container: &str) -> bool {
        self.mapper
            .validate_compatibility(codec, container)
            .is_compatible
    }

    /// Return all codecs from `candidates` that are directly compatible with
    /// `container`, sorted by compatibility score (best first).
    #[must_use]
    pub fn filter_compatible(
        &self,
        candidates: &[CodecFamily],
        container: &str,
    ) -> Vec<CodecFamily> {
        let mut compat: Vec<(CodecFamily, u8)> = candidates
            .iter()
            .filter_map(|&c| {
                let m = self.mapper.get_mapping(c, container)?;
                if m.is_compatible() {
                    Some((c, m.score))
                } else {
                    None
                }
            })
            .collect();
        compat.sort_by(|a, b| b.1.cmp(&a.1));
        compat.into_iter().map(|(c, _)| c).collect()
    }

    /// Suggest the best alternative codec for `source` in `container`.
    ///
    /// Returns `None` if the source is already directly compatible or has no
    /// known mapping.
    #[must_use]
    pub fn suggest_alternative(&self, source: CodecFamily, container: &str) -> Option<CodecFamily> {
        let report = self.mapper.validate_compatibility(source, container);
        if report.is_compatible {
            return None;
        }
        if report.recommended_codec == CodecFamily::Unknown {
            return None;
        }
        Some(report.recommended_codec)
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

    // ── validate_compatibility ────────────────────────────────────────────────

    #[test]
    fn test_validate_h264_mp4_is_compatible() {
        let report = mapper().validate_compatibility(CodecFamily::H264, "mp4");
        assert!(report.is_compatible, "H264 in mp4 should be compatible");
        assert!(report.is_clean(), "no issues expected for direct-copy");
    }

    #[test]
    fn test_validate_h264_webm_needs_transcode() {
        let report = mapper().validate_compatibility(CodecFamily::H264, "webm");
        assert!(!report.is_compatible, "H264 cannot go directly into WebM");
        assert!(report.issue_count() > 0, "should report at least one issue");
        assert_eq!(report.recommended_codec, CodecFamily::Vpx);
    }

    #[test]
    fn test_validate_unknown_codec_has_no_mapping() {
        let report = mapper().validate_compatibility(CodecFamily::Mpeg2Video, "webm");
        assert!(!report.is_compatible);
        assert_eq!(report.recommended_codec, CodecFamily::Unknown);
        assert!(!report.issues.is_empty());
        assert!(!report.suggestions.is_empty());
    }

    #[test]
    fn test_validate_av1_mkv_is_compatible() {
        let report = mapper().validate_compatibility(CodecFamily::Av1, "mkv");
        assert!(report.is_compatible, "AV1 → MKV is direct copy");
        assert!(report.is_clean());
    }

    #[test]
    fn test_validate_video_in_ogg_reports_issue() {
        let report = mapper().validate_compatibility(CodecFamily::H264, "ogg");
        let has_issue = report.issues.iter().any(|s| {
            s.contains("not suitable") || s.contains("No known") || s.contains("no known")
        });
        assert!(has_issue, "should report video-in-ogg issue");
    }

    #[test]
    fn test_validate_flac_in_mp4_lossless_advisory() {
        let report = mapper().validate_compatibility(CodecFamily::Flac, "mp4");
        let advisory = report
            .suggestions
            .iter()
            .any(|s| s.contains("lossless") || s.contains("large"));
        assert!(advisory, "should warn about lossless in mp4");
    }

    // ── MultiCodecValidator tests ─────────────────────────────────────────────

    fn validator() -> MultiCodecValidator {
        MultiCodecValidator::new()
    }

    #[test]
    fn multi_empty_codecs_produces_clean_summary() {
        let summary = validator().validate(&[], "webm");
        assert!(summary.is_clean());
        assert_eq!(summary.error_count(), 0);
        assert!(summary.compatible_codecs.is_empty());
        assert!(summary.transcode_required.is_empty());
        assert!(summary.unmapped_codecs.is_empty());
    }

    #[test]
    fn multi_all_compatible_is_clean() {
        // AV1 + Opus are both direct-copy in WebM
        let summary = validator().validate(&[CodecFamily::Av1, CodecFamily::Opus], "webm");
        assert!(summary.is_clean(), "all-compatible should be clean");
        assert!(summary.has_compatible_codecs());
        assert!(summary.transcode_required.is_empty());
    }

    #[test]
    fn multi_h264_in_webm_needs_transcode() {
        let summary = validator().validate(&[CodecFamily::H264, CodecFamily::Opus], "webm");
        assert!(
            summary
                .transcode_required
                .iter()
                .any(|(src, _)| *src == CodecFamily::H264),
            "H264 should be in transcode_required"
        );
        assert!(summary.compatible_codecs.contains(&CodecFamily::Opus));
    }

    #[test]
    fn multi_unmapped_codec_is_error() {
        let summary = validator().validate(&[CodecFamily::Mpeg2Video], "webm");
        assert!(
            summary.error_count() > 0,
            "unmapped codec should be an error"
        );
        assert!(summary.unmapped_codecs.contains(&CodecFamily::Mpeg2Video));
    }

    #[test]
    fn multi_findings_sorted_errors_first() {
        let codecs = [CodecFamily::Mpeg2Video, CodecFamily::H264, CodecFamily::Av1];
        let summary = validator().validate(&codecs, "webm");
        let severities: Vec<IssueSeverity> = summary.findings.iter().map(|f| f.severity).collect();
        for w in severities.windows(2) {
            assert!(
                w[0] >= w[1],
                "findings must be in descending severity order"
            );
        }
    }

    #[test]
    fn multi_transcode_includes_recommended_codec() {
        let summary = validator().validate(&[CodecFamily::H264], "webm");
        assert!(!summary.transcode_required.is_empty());
        let (src, tgt) = summary.transcode_required[0];
        assert_eq!(src, CodecFamily::H264);
        assert_eq!(tgt, CodecFamily::Vpx);
    }

    #[test]
    fn multi_is_direct_compatible_true() {
        assert!(validator().is_direct_compatible(CodecFamily::Av1, "webm"));
        assert!(validator().is_direct_compatible(CodecFamily::H264, "mp4"));
    }

    #[test]
    fn multi_is_direct_compatible_false() {
        assert!(!validator().is_direct_compatible(CodecFamily::H264, "webm"));
    }

    #[test]
    fn multi_filter_compatible_excludes_transcode_codecs() {
        let candidates = [
            CodecFamily::Av1,
            CodecFamily::H264,
            CodecFamily::Vorbis,
            CodecFamily::Opus,
        ];
        let compat = validator().filter_compatible(&candidates, "webm");
        assert!(compat.contains(&CodecFamily::Av1));
        assert!(!compat.contains(&CodecFamily::H264));
    }

    #[test]
    fn multi_suggest_alternative_h264_to_webm() {
        let alt = validator().suggest_alternative(CodecFamily::H264, "webm");
        assert_eq!(alt, Some(CodecFamily::Vpx));
    }

    #[test]
    fn multi_suggest_alternative_none_for_compatible() {
        let alt = validator().suggest_alternative(CodecFamily::Av1, "webm");
        assert!(
            alt.is_none(),
            "no alternative needed for already-compatible codec"
        );
    }

    #[test]
    fn multi_summary_container_normalised() {
        let summary = validator().validate(&[CodecFamily::Av1], "WebM");
        assert_eq!(summary.container, "webm");
    }

    // ── IssueSeverity / CompatibilityFinding tests ────────────────────────────

    #[test]
    fn severity_ordering() {
        assert!(IssueSeverity::Error > IssueSeverity::Warning);
        assert!(IssueSeverity::Warning > IssueSeverity::Info);
    }

    #[test]
    fn severity_tags() {
        assert_eq!(IssueSeverity::Error.tag(), "ERROR");
        assert_eq!(IssueSeverity::Warning.tag(), "WARN");
        assert_eq!(IssueSeverity::Info.tag(), "INFO");
    }

    #[test]
    fn finding_with_suggestion() {
        let f = CompatibilityFinding::error(CodecFamily::H264, "test error")
            .with_suggestion("use VP9 instead");
        assert_eq!(f.severity, IssueSeverity::Error);
        assert_eq!(f.suggestion.as_deref(), Some("use VP9 instead"));
    }

    #[test]
    fn summary_findings_at_least_error() {
        let summary = validator().validate(&[CodecFamily::Mpeg2Video, CodecFamily::H264], "webm");
        for f in summary.findings_at_least(IssueSeverity::Error) {
            assert!(f.severity >= IssueSeverity::Error);
        }
    }
}
