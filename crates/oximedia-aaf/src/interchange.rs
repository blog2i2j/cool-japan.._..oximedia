#![allow(dead_code)]
//! AAF interchange format conversion between AAF, MXF, and OMF.
//!
//! Provides bidirectional conversion utilities for interchanging project data
//! between different professional post-production formats including:
//!
//! - **MXF** (Material Exchange Format, SMPTE ST 377-1)
//! - **OMF** (Open Media Framework, legacy Avid format)
//! - **FCP XML** (Final Cut Pro XML interchange)
//!
//! This module handles track mapping, essence reference translation, metadata
//! preservation, and timeline fidelity during format conversion.

use std::collections::HashMap;
use uuid::Uuid;

/// Supported interchange target formats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum InterchangeFormat {
    /// Material Exchange Format (SMPTE ST 377-1)
    Mxf,
    /// Open Media Framework (legacy Avid)
    Omf,
    /// Final Cut Pro XML
    FcpXml,
    /// DaVinci Resolve project XML
    ResolveXml,
    /// Adobe Premiere Pro project XML
    PremiereXml,
}

impl InterchangeFormat {
    /// Returns the typical file extension for this format.
    #[must_use]
    pub const fn extension(self) -> &'static str {
        match self {
            Self::Mxf => "mxf",
            Self::Omf => "omf",
            Self::FcpXml => "fcpxml",
            Self::ResolveXml => "xml",
            Self::PremiereXml => "prproj",
        }
    }

    /// Whether this format supports embedded essence data.
    #[must_use]
    pub const fn supports_embedded_essence(self) -> bool {
        matches!(self, Self::Mxf | Self::Omf)
    }

    /// Whether this format supports multiple video tracks.
    #[must_use]
    pub const fn supports_multi_video_tracks(self) -> bool {
        matches!(
            self,
            Self::Mxf | Self::FcpXml | Self::ResolveXml | Self::PremiereXml
        )
    }

    /// Whether this format is considered legacy.
    #[must_use]
    pub const fn is_legacy(self) -> bool {
        matches!(self, Self::Omf)
    }
}

/// Options controlling how interchange conversion proceeds.
#[derive(Debug, Clone)]
pub struct InterchangeOptions {
    /// Target format for conversion.
    pub target_format: InterchangeFormat,
    /// Whether to embed essence data in the output.
    pub embed_essence: bool,
    /// Whether to preserve all metadata (may increase file size).
    pub preserve_metadata: bool,
    /// Whether to flatten nested compositions.
    pub flatten_compositions: bool,
    /// Maximum number of video tracks to export.
    pub max_video_tracks: Option<usize>,
    /// Maximum number of audio tracks to export.
    pub max_audio_tracks: Option<usize>,
    /// Whether to convert timecode to match target format conventions.
    pub convert_timecode: bool,
    /// Whether to include marker/locator data.
    pub include_markers: bool,
}

impl Default for InterchangeOptions {
    fn default() -> Self {
        Self {
            target_format: InterchangeFormat::Mxf,
            embed_essence: false,
            preserve_metadata: true,
            flatten_compositions: false,
            max_video_tracks: None,
            max_audio_tracks: None,
            convert_timecode: true,
            include_markers: true,
        }
    }
}

/// Represents a mapping between source and destination track IDs during conversion.
#[derive(Debug, Clone)]
pub struct TrackMapping {
    /// Source track identifier.
    pub source_track_id: u32,
    /// Destination track identifier.
    pub dest_track_id: u32,
    /// Track type (video, audio, timecode, etc.).
    pub track_type: TrackKind,
    /// Optional label for the mapped track.
    pub label: Option<String>,
    /// Whether this track is muted in output.
    pub muted: bool,
}

/// Kind of track in the interchange mapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TrackKind {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Timecode track.
    Timecode,
    /// Data/metadata track.
    Data,
    /// Auxiliary track.
    Auxiliary,
}

impl TrackKind {
    /// Returns a human-readable label for this track kind.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Video => "Video",
            Self::Audio => "Audio",
            Self::Timecode => "Timecode",
            Self::Data => "Data",
            Self::Auxiliary => "Auxiliary",
        }
    }
}

/// Essence reference in the interchange domain, abstracting over format-specific references.
#[derive(Debug, Clone)]
pub struct EssenceRef {
    /// Unique identifier for the essence.
    pub essence_id: Uuid,
    /// Original file path (if external reference).
    pub file_path: Option<String>,
    /// MIME type of the essence data.
    pub mime_type: Option<String>,
    /// Data definition (e.g., "Picture", "Sound").
    pub data_def: String,
    /// Byte size of the essence, if known.
    pub byte_size: Option<u64>,
}

/// Result of an interchange conversion operation.
#[derive(Debug, Clone)]
pub struct InterchangeResult {
    /// Target format that was generated.
    pub format: InterchangeFormat,
    /// Number of tracks converted.
    pub tracks_converted: usize,
    /// Number of clips converted.
    pub clips_converted: usize,
    /// Number of effects that were dropped (unsupported in target).
    pub effects_dropped: usize,
    /// Number of transitions that were dropped.
    pub transitions_dropped: usize,
    /// Warnings generated during conversion.
    pub warnings: Vec<String>,
    /// Track mappings used.
    pub track_mappings: Vec<TrackMapping>,
    /// Total duration in seconds.
    pub duration_seconds: f64,
}

impl InterchangeResult {
    /// Create a new empty interchange result.
    #[must_use]
    pub fn new(format: InterchangeFormat) -> Self {
        Self {
            format,
            tracks_converted: 0,
            clips_converted: 0,
            effects_dropped: 0,
            transitions_dropped: 0,
            warnings: Vec::new(),
            track_mappings: Vec::new(),
            duration_seconds: 0.0,
        }
    }

    /// Whether the conversion completed without any warnings.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.warnings.is_empty() && self.effects_dropped == 0 && self.transitions_dropped == 0
    }

    /// Total items dropped during conversion.
    #[must_use]
    pub fn total_dropped(&self) -> usize {
        self.effects_dropped + self.transitions_dropped
    }

    /// Add a warning message.
    pub fn add_warning(&mut self, msg: impl Into<String>) {
        self.warnings.push(msg.into());
    }
}

/// Manages the conversion pipeline between interchange formats.
#[derive(Debug, Clone)]
pub struct InterchangeConverter {
    /// Configuration options.
    options: InterchangeOptions,
    /// Track mappings from source to destination.
    track_map: HashMap<u32, TrackMapping>,
    /// Essence references collected during conversion.
    essence_refs: Vec<EssenceRef>,
}

impl InterchangeConverter {
    /// Create a new converter with the given options.
    #[must_use]
    pub fn new(options: InterchangeOptions) -> Self {
        Self {
            options,
            track_map: HashMap::new(),
            essence_refs: Vec::new(),
        }
    }

    /// Create a converter targeting MXF with default options.
    #[must_use]
    pub fn to_mxf() -> Self {
        Self::new(InterchangeOptions {
            target_format: InterchangeFormat::Mxf,
            ..Default::default()
        })
    }

    /// Create a converter targeting OMF with default options.
    #[must_use]
    pub fn to_omf() -> Self {
        Self::new(InterchangeOptions {
            target_format: InterchangeFormat::Omf,
            ..Default::default()
        })
    }

    /// Get the target format.
    #[must_use]
    pub fn target_format(&self) -> InterchangeFormat {
        self.options.target_format
    }

    /// Add a track mapping.
    pub fn add_track_mapping(&mut self, mapping: TrackMapping) {
        self.track_map.insert(mapping.source_track_id, mapping);
    }

    /// Register an essence reference.
    pub fn register_essence(&mut self, essence_ref: EssenceRef) {
        self.essence_refs.push(essence_ref);
    }

    /// Get the number of registered track mappings.
    #[must_use]
    pub fn track_mapping_count(&self) -> usize {
        self.track_map.len()
    }

    /// Get the number of registered essence references.
    #[must_use]
    pub fn essence_ref_count(&self) -> usize {
        self.essence_refs.len()
    }

    /// Look up a track mapping by source track ID.
    #[must_use]
    pub fn get_track_mapping(&self, source_id: u32) -> Option<&TrackMapping> {
        self.track_map.get(&source_id)
    }

    /// Build the interchange result based on current state.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn build_result(
        &self,
        clips: usize,
        dropped_fx: usize,
        dropped_trans: usize,
        duration_secs: f64,
    ) -> InterchangeResult {
        let mut result = InterchangeResult::new(self.options.target_format);
        result.tracks_converted = self.track_map.len();
        result.clips_converted = clips;
        result.effects_dropped = dropped_fx;
        result.transitions_dropped = dropped_trans;
        result.duration_seconds = duration_secs;
        result.track_mappings = self.track_map.values().cloned().collect();

        if !self.options.target_format.supports_embedded_essence() && self.options.embed_essence {
            result.add_warning(
                "Target format does not support embedded essence; references will be external"
                    .to_string(),
            );
        }

        if self.options.target_format.is_legacy() {
            result.add_warning("Target format is legacy; some features may be lost".to_string());
        }

        result
    }

    /// Validate that the current configuration is consistent.
    #[must_use]
    pub fn validate_config(&self) -> Vec<String> {
        let mut errors = Vec::new();

        if self.options.embed_essence && !self.options.target_format.supports_embedded_essence() {
            errors.push(format!(
                "Cannot embed essence in {} format",
                self.options.target_format.extension()
            ));
        }

        if let Some(max_v) = self.options.max_video_tracks {
            let video_count = self
                .track_map
                .values()
                .filter(|m| m.track_type == TrackKind::Video)
                .count();
            if video_count > max_v {
                errors.push(format!(
                    "Video track count ({video_count}) exceeds max ({max_v})"
                ));
            }
        }

        if let Some(max_a) = self.options.max_audio_tracks {
            let audio_count = self
                .track_map
                .values()
                .filter(|m| m.track_type == TrackKind::Audio)
                .count();
            if audio_count > max_a {
                errors.push(format!(
                    "Audio track count ({audio_count}) exceeds max ({max_a})"
                ));
            }
        }

        errors
    }
}

/// Capability matrix describing what features a target format supports.
#[derive(Debug, Clone)]
pub struct FormatCapabilities {
    /// Format this describes.
    pub format: InterchangeFormat,
    /// Maximum number of video tracks (None = unlimited).
    pub max_video_tracks: Option<usize>,
    /// Maximum number of audio tracks (None = unlimited).
    pub max_audio_tracks: Option<usize>,
    /// Whether nested compositions are supported.
    pub nested_compositions: bool,
    /// Whether effects/transitions are supported.
    pub effects_supported: bool,
    /// Whether markers/locators are supported.
    pub markers_supported: bool,
    /// Whether timecode tracks are supported.
    pub timecode_tracks: bool,
}

impl FormatCapabilities {
    /// Get capabilities for a specific format.
    #[must_use]
    pub fn for_format(format: InterchangeFormat) -> Self {
        match format {
            InterchangeFormat::Mxf => Self {
                format,
                max_video_tracks: None,
                max_audio_tracks: None,
                nested_compositions: true,
                effects_supported: true,
                markers_supported: true,
                timecode_tracks: true,
            },
            InterchangeFormat::Omf => Self {
                format,
                max_video_tracks: Some(1),
                max_audio_tracks: Some(24),
                nested_compositions: false,
                effects_supported: false,
                markers_supported: true,
                timecode_tracks: true,
            },
            InterchangeFormat::FcpXml => Self {
                format,
                max_video_tracks: None,
                max_audio_tracks: None,
                nested_compositions: true,
                effects_supported: true,
                markers_supported: true,
                timecode_tracks: false,
            },
            InterchangeFormat::ResolveXml => Self {
                format,
                max_video_tracks: None,
                max_audio_tracks: None,
                nested_compositions: true,
                effects_supported: true,
                markers_supported: true,
                timecode_tracks: true,
            },
            InterchangeFormat::PremiereXml => Self {
                format,
                max_video_tracks: None,
                max_audio_tracks: None,
                nested_compositions: true,
                effects_supported: true,
                markers_supported: true,
                timecode_tracks: true,
            },
        }
    }

    /// Check if a feature set fits within this format's capabilities.
    #[must_use]
    pub fn can_accommodate(&self, video_tracks: usize, audio_tracks: usize) -> bool {
        let video_ok = self
            .max_video_tracks
            .map_or(true, |max| video_tracks <= max);
        let audio_ok = self
            .max_audio_tracks
            .map_or(true, |max| audio_tracks <= max);
        video_ok && audio_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interchange_format_extension() {
        assert_eq!(InterchangeFormat::Mxf.extension(), "mxf");
        assert_eq!(InterchangeFormat::Omf.extension(), "omf");
        assert_eq!(InterchangeFormat::FcpXml.extension(), "fcpxml");
        assert_eq!(InterchangeFormat::ResolveXml.extension(), "xml");
        assert_eq!(InterchangeFormat::PremiereXml.extension(), "prproj");
    }

    #[test]
    fn test_format_supports_embedded_essence() {
        assert!(InterchangeFormat::Mxf.supports_embedded_essence());
        assert!(InterchangeFormat::Omf.supports_embedded_essence());
        assert!(!InterchangeFormat::FcpXml.supports_embedded_essence());
        assert!(!InterchangeFormat::ResolveXml.supports_embedded_essence());
    }

    #[test]
    fn test_format_supports_multi_video() {
        assert!(InterchangeFormat::Mxf.supports_multi_video_tracks());
        assert!(!InterchangeFormat::Omf.supports_multi_video_tracks());
        assert!(InterchangeFormat::FcpXml.supports_multi_video_tracks());
    }

    #[test]
    fn test_format_is_legacy() {
        assert!(InterchangeFormat::Omf.is_legacy());
        assert!(!InterchangeFormat::Mxf.is_legacy());
    }

    #[test]
    fn test_interchange_options_default() {
        let opts = InterchangeOptions::default();
        assert_eq!(opts.target_format, InterchangeFormat::Mxf);
        assert!(!opts.embed_essence);
        assert!(opts.preserve_metadata);
        assert!(!opts.flatten_compositions);
        assert!(opts.convert_timecode);
        assert!(opts.include_markers);
    }

    #[test]
    fn test_track_kind_label() {
        assert_eq!(TrackKind::Video.label(), "Video");
        assert_eq!(TrackKind::Audio.label(), "Audio");
        assert_eq!(TrackKind::Timecode.label(), "Timecode");
        assert_eq!(TrackKind::Data.label(), "Data");
        assert_eq!(TrackKind::Auxiliary.label(), "Auxiliary");
    }

    #[test]
    fn test_interchange_result_new() {
        let result = InterchangeResult::new(InterchangeFormat::Mxf);
        assert_eq!(result.format, InterchangeFormat::Mxf);
        assert_eq!(result.tracks_converted, 0);
        assert_eq!(result.clips_converted, 0);
        assert!(result.is_clean());
        assert_eq!(result.total_dropped(), 0);
    }

    #[test]
    fn test_interchange_result_warnings() {
        let mut result = InterchangeResult::new(InterchangeFormat::Omf);
        assert!(result.is_clean());
        result.add_warning("test warning");
        assert!(!result.is_clean());
        assert_eq!(result.warnings.len(), 1);
    }

    #[test]
    fn test_interchange_result_dropped() {
        let mut result = InterchangeResult::new(InterchangeFormat::Mxf);
        result.effects_dropped = 3;
        result.transitions_dropped = 2;
        assert_eq!(result.total_dropped(), 5);
        assert!(!result.is_clean());
    }

    #[test]
    fn test_converter_creation() {
        let conv = InterchangeConverter::to_mxf();
        assert_eq!(conv.target_format(), InterchangeFormat::Mxf);
        assert_eq!(conv.track_mapping_count(), 0);
        assert_eq!(conv.essence_ref_count(), 0);
    }

    #[test]
    fn test_converter_track_mapping() {
        let mut conv = InterchangeConverter::to_mxf();
        conv.add_track_mapping(TrackMapping {
            source_track_id: 1,
            dest_track_id: 10,
            track_type: TrackKind::Video,
            label: Some("V1".to_string()),
            muted: false,
        });
        assert_eq!(conv.track_mapping_count(), 1);
        let mapping = conv.get_track_mapping(1).expect("mapping should be valid");
        assert_eq!(mapping.dest_track_id, 10);
        assert_eq!(mapping.track_type, TrackKind::Video);
    }

    #[test]
    fn test_converter_essence_ref() {
        let mut conv = InterchangeConverter::to_omf();
        conv.register_essence(EssenceRef {
            essence_id: Uuid::new_v4(),
            file_path: Some("/media/clip001.mxf".to_string()),
            mime_type: Some("video/mxf".to_string()),
            data_def: "Picture".to_string(),
            byte_size: Some(1_000_000),
        });
        assert_eq!(conv.essence_ref_count(), 1);
    }

    #[test]
    fn test_converter_build_result() {
        let mut conv = InterchangeConverter::to_omf();
        conv.add_track_mapping(TrackMapping {
            source_track_id: 1,
            dest_track_id: 1,
            track_type: TrackKind::Video,
            label: None,
            muted: false,
        });
        let result = conv.build_result(10, 2, 1, 120.5);
        assert_eq!(result.tracks_converted, 1);
        assert_eq!(result.clips_converted, 10);
        assert_eq!(result.effects_dropped, 2);
        assert_eq!(result.transitions_dropped, 1);
        assert!((result.duration_seconds - 120.5).abs() < f64::EPSILON);
        // OMF is legacy, should have a warning
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_converter_validate_config() {
        let mut conv = InterchangeConverter::new(InterchangeOptions {
            target_format: InterchangeFormat::FcpXml,
            embed_essence: true,
            ..Default::default()
        });
        conv.add_track_mapping(TrackMapping {
            source_track_id: 1,
            dest_track_id: 1,
            track_type: TrackKind::Video,
            label: None,
            muted: false,
        });
        let errors = conv.validate_config();
        // FcpXml doesn't support embedded essence
        assert!(!errors.is_empty());
    }

    #[test]
    fn test_format_capabilities_mxf() {
        let caps = FormatCapabilities::for_format(InterchangeFormat::Mxf);
        assert!(caps.nested_compositions);
        assert!(caps.effects_supported);
        assert!(caps.markers_supported);
        assert!(caps.timecode_tracks);
        assert!(caps.can_accommodate(10, 64));
    }

    #[test]
    fn test_format_capabilities_omf_limits() {
        let caps = FormatCapabilities::for_format(InterchangeFormat::Omf);
        assert!(!caps.nested_compositions);
        assert!(!caps.effects_supported);
        assert!(caps.can_accommodate(1, 24));
        assert!(!caps.can_accommodate(2, 24)); // OMF max 1 video track
        assert!(!caps.can_accommodate(1, 25)); // OMF max 24 audio tracks
    }
}
