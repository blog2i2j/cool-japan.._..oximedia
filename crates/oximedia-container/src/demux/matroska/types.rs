//! Matroska type definitions.
//!
//! This module contains all the type definitions used for parsing and
//! representing Matroska/`WebM` container data structures.

use oximedia_core::CodecId;

use super::matroska_v4::BlockAdditionMapping;

// ============================================================================
// Document Type
// ============================================================================

/// Matroska document type.
///
/// Indicates whether the file is a full Matroska container or a `WebM` subset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum DocType {
    /// Full Matroska container (.mkv, .mka, .mks).
    #[default]
    Matroska,
    /// `WebM` container (.webm) - restricted codec set.
    WebM,
}

impl DocType {
    /// Returns true if this is a `WebM` document.
    #[must_use]
    pub const fn is_webm(&self) -> bool {
        matches!(self, Self::WebM)
    }

    /// Returns the document type string as used in the EBML header.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Matroska => "matroska",
            Self::WebM => "webm",
        }
    }
}

// ============================================================================
// EBML Header
// ============================================================================

/// EBML header information.
///
/// Contains metadata about the EBML document structure and version requirements.
#[derive(Clone, Debug)]
pub struct EbmlHeader {
    /// EBML version used to create the file.
    pub version: u64,
    /// Minimum EBML version required to read the file.
    pub read_version: u64,
    /// Maximum ID length in bytes (default 4).
    pub max_id_length: u64,
    /// Maximum size length in bytes (default 8).
    pub max_size_length: u64,
    /// Document type (matroska or webm).
    pub doc_type: DocType,
    /// Document type version.
    pub doc_type_version: u64,
    /// Minimum document type version required to read.
    pub doc_type_read_version: u64,
}

impl Default for EbmlHeader {
    fn default() -> Self {
        Self {
            version: 1,
            read_version: 1,
            max_id_length: 4,
            max_size_length: 8,
            doc_type: DocType::Matroska,
            doc_type_version: 4,
            doc_type_read_version: 2,
        }
    }
}

// ============================================================================
// Segment Information
// ============================================================================

/// Segment information.
///
/// Contains metadata about the segment including timing and application info.
#[derive(Clone, Debug)]
pub struct SegmentInfo {
    /// Timestamp scale in nanoseconds (default 1,000,000 = 1ms).
    pub timecode_scale: u64,
    /// Duration of the segment in timecode scale units.
    pub duration: Option<f64>,
    /// Title of the segment.
    pub title: Option<String>,
    /// Application that created the muxed file.
    pub muxing_app: Option<String>,
    /// Application that wrote the file.
    pub writing_app: Option<String>,
    /// Date the file was created (nanoseconds since 2001-01-01).
    pub date_utc: Option<i64>,
    /// Unique segment identifier.
    pub segment_uid: Option<Vec<u8>>,
}

impl Default for SegmentInfo {
    fn default() -> Self {
        Self {
            timecode_scale: 1_000_000, // 1 millisecond
            duration: None,
            title: None,
            muxing_app: None,
            writing_app: None,
            date_utc: None,
            segment_uid: None,
        }
    }
}

impl SegmentInfo {
    /// Returns the duration in seconds, if available.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self) -> Option<f64> {
        self.duration
            .map(|d| d * self.timecode_scale as f64 / 1_000_000_000.0)
    }
}

// ============================================================================
// Track Types
// ============================================================================

/// Track types as defined by Matroska.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum TrackType {
    /// Video track.
    Video = 1,
    /// Audio track.
    Audio = 2,
    /// Complex track (combined audio/video).
    Complex = 3,
    /// Logo overlay track.
    Logo = 0x10,
    /// Subtitle track.
    Subtitle = 0x11,
    /// Button track (DVD menus).
    Buttons = 0x12,
    /// Control track.
    Control = 0x20,
    /// Metadata track.
    Metadata = 0x21,
}

impl TryFrom<u64> for TrackType {
    type Error = ();

    fn try_from(value: u64) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Video),
            2 => Ok(Self::Audio),
            3 => Ok(Self::Complex),
            0x10 => Ok(Self::Logo),
            0x11 => Ok(Self::Subtitle),
            0x12 => Ok(Self::Buttons),
            0x20 => Ok(Self::Control),
            0x21 => Ok(Self::Metadata),
            _ => Err(()),
        }
    }
}

// ============================================================================
// Video Settings
// ============================================================================

/// Video track settings.
#[derive(Clone, Debug, Default)]
pub struct VideoSettings {
    /// Width of encoded video in pixels.
    pub pixel_width: u32,
    /// Height of encoded video in pixels.
    pub pixel_height: u32,
    /// Display width (for anamorphic video).
    pub display_width: Option<u32>,
    /// Display height (for anamorphic video).
    pub display_height: Option<u32>,
    /// Display unit (0=pixels, 1=cm, 2=inches, 3=aspect ratio).
    pub display_unit: u8,
    /// Whether the video is interlaced.
    pub interlaced: bool,
    /// Field order for interlaced content.
    pub field_order: Option<u8>,
    /// Stereo-3D mode.
    pub stereo_mode: Option<u8>,
    /// Alpha mode.
    pub alpha_mode: Option<u8>,
    /// Pixel crop from bottom.
    pub pixel_crop_bottom: u32,
    /// Pixel crop from top.
    pub pixel_crop_top: u32,
    /// Pixel crop from left.
    pub pixel_crop_left: u32,
    /// Pixel crop from right.
    pub pixel_crop_right: u32,
    /// Color settings.
    pub colour: Option<ColourSettings>,
}

impl VideoSettings {
    /// Returns the effective display width.
    #[must_use]
    pub fn effective_display_width(&self) -> u32 {
        self.display_width.unwrap_or(self.pixel_width)
    }

    /// Returns the effective display height.
    #[must_use]
    pub fn effective_display_height(&self) -> u32 {
        self.display_height.unwrap_or(self.pixel_height)
    }
}

/// Video colour settings.
#[derive(Clone, Debug, Default)]
pub struct ColourSettings {
    /// Matrix coefficients (ITU-R BT series).
    pub matrix_coefficients: Option<u8>,
    /// Bits per channel.
    pub bits_per_channel: Option<u8>,
    /// Chroma subsampling horizontal.
    pub chroma_subsampling_horz: Option<u8>,
    /// Chroma subsampling vertical.
    pub chroma_subsampling_vert: Option<u8>,
    /// Cb subsampling horizontal.
    pub cb_subsampling_horz: Option<u8>,
    /// Cb subsampling vertical.
    pub cb_subsampling_vert: Option<u8>,
    /// Chroma siting horizontal.
    pub chroma_siting_horz: Option<u8>,
    /// Chroma siting vertical.
    pub chroma_siting_vert: Option<u8>,
    /// Range (0=unspecified, 1=broadcast, 2=full, 3=defined by matrix).
    pub range: Option<u8>,
    /// Transfer characteristics (gamma curve).
    pub transfer_characteristics: Option<u8>,
    /// Color primaries.
    pub primaries: Option<u8>,
    /// Maximum content light level (cd/m^2).
    pub max_cll: Option<u64>,
    /// Maximum frame-average light level (cd/m^2).
    pub max_fall: Option<u64>,
    /// Mastering display metadata.
    pub mastering_metadata: Option<MasteringMetadata>,
}

/// HDR mastering display metadata.
#[derive(Clone, Debug, Default)]
pub struct MasteringMetadata {
    /// Red X chromaticity.
    pub primary_r_chromaticity_x: Option<f64>,
    /// Red Y chromaticity.
    pub primary_r_chromaticity_y: Option<f64>,
    /// Green X chromaticity.
    pub primary_g_chromaticity_x: Option<f64>,
    /// Green Y chromaticity.
    pub primary_g_chromaticity_y: Option<f64>,
    /// Blue X chromaticity.
    pub primary_b_chromaticity_x: Option<f64>,
    /// Blue Y chromaticity.
    pub primary_b_chromaticity_y: Option<f64>,
    /// White point X chromaticity.
    pub white_point_chromaticity_x: Option<f64>,
    /// White point Y chromaticity.
    pub white_point_chromaticity_y: Option<f64>,
    /// Maximum luminance (cd/m^2).
    pub luminance_max: Option<f64>,
    /// Minimum luminance (cd/m^2).
    pub luminance_min: Option<f64>,
}

// ============================================================================
// Audio Settings
// ============================================================================

/// Audio track settings.
#[derive(Clone, Debug)]
pub struct AudioSettings {
    /// Sample rate in Hz.
    pub sampling_frequency: f64,
    /// Output sample rate (for SBR audio).
    pub output_sampling_frequency: Option<f64>,
    /// Number of audio channels.
    pub channels: u8,
    /// Bits per sample.
    pub bit_depth: Option<u8>,
}

impl Default for AudioSettings {
    fn default() -> Self {
        Self {
            sampling_frequency: 8000.0,
            output_sampling_frequency: None,
            channels: 1,
            bit_depth: None,
        }
    }
}

impl AudioSettings {
    /// Returns the effective output sample rate.
    #[must_use]
    pub fn effective_sample_rate(&self) -> f64 {
        self.output_sampling_frequency
            .unwrap_or(self.sampling_frequency)
    }
}

// ============================================================================
// Track Entry
// ============================================================================

/// A single track entry.
#[derive(Clone, Debug)]
#[allow(clippy::struct_excessive_bools)]
pub struct TrackEntry {
    /// Track number (1-based, used in block headers).
    pub number: u64,
    /// Unique track identifier.
    pub uid: u64,
    /// Track type.
    pub track_type: TrackType,
    /// Whether the track is enabled.
    pub enabled: bool,
    /// Whether this is the default track.
    pub default: bool,
    /// Whether the track is forced.
    pub forced: bool,
    /// Whether lacing is used.
    pub lacing: bool,
    /// Minimum cache required (frames).
    pub min_cache: u64,
    /// Maximum cache allowed (frames).
    pub max_cache: Option<u64>,
    /// Default frame duration in nanoseconds.
    pub default_duration: Option<u64>,
    /// Track timecode scale (deprecated, should be 1.0).
    pub track_timecode_scale: f64,
    /// Human-readable track name.
    pub name: Option<String>,
    /// Track language (ISO 639-2).
    pub language: String,
    /// Track language (BCP 47).
    pub language_ietf: Option<String>,
    /// Codec identifier string.
    pub codec_id: String,
    /// Codec-specific private data.
    pub codec_private: Option<Vec<u8>>,
    /// Human-readable codec name.
    pub codec_name: Option<String>,
    /// Codec delay in nanoseconds.
    pub codec_delay: Option<u64>,
    /// Seek pre-roll in nanoseconds.
    pub seek_pre_roll: Option<u64>,
    /// Video settings (if video track).
    pub video: Option<VideoSettings>,
    /// Audio settings (if audio track).
    pub audio: Option<AudioSettings>,
    /// Matroska v4 block addition mappings attached to this track.
    pub block_addition_mappings: Vec<BlockAdditionMapping>,
    /// Mapped `OxiMedia` codec ID.
    pub oxi_codec: Option<CodecId>,
}

impl Default for TrackEntry {
    fn default() -> Self {
        Self {
            number: 0,
            uid: 0,
            track_type: TrackType::Video,
            enabled: true,
            default: true,
            forced: false,
            lacing: true,
            min_cache: 0,
            max_cache: None,
            default_duration: None,
            track_timecode_scale: 1.0,
            name: None,
            language: "eng".to_string(),
            language_ietf: None,
            codec_id: String::new(),
            codec_private: None,
            codec_name: None,
            codec_delay: None,
            seek_pre_roll: None,
            video: None,
            audio: None,
            block_addition_mappings: Vec::new(),
            oxi_codec: None,
        }
    }
}

impl TrackEntry {
    /// Returns true if this is a video track.
    #[must_use]
    pub const fn is_video(&self) -> bool {
        matches!(self.track_type, TrackType::Video)
    }

    /// Returns true if this is an audio track.
    #[must_use]
    pub const fn is_audio(&self) -> bool {
        matches!(self.track_type, TrackType::Audio)
    }

    /// Returns true if this is a subtitle track.
    #[must_use]
    pub const fn is_subtitle(&self) -> bool {
        matches!(self.track_type, TrackType::Subtitle)
    }
}

// ============================================================================
// Cue Points (Seek Index)
// ============================================================================

/// A cue point for seeking.
#[derive(Clone, Debug, Default)]
pub struct CuePoint {
    /// Timestamp in timecode scale units.
    pub time: u64,
    /// Track positions for this cue point.
    pub track_positions: Vec<CueTrackPosition>,
}

/// Track position within a cue point.
#[derive(Clone, Debug, Default)]
pub struct CueTrackPosition {
    /// Track number.
    pub track: u64,
    /// Position of the cluster in the segment.
    pub cluster_position: u64,
    /// Relative position within the cluster.
    pub relative_position: Option<u64>,
    /// Duration of the cue.
    pub duration: Option<u64>,
    /// Block number within the cluster.
    pub block_number: Option<u64>,
}

// ============================================================================
// Chapters
// ============================================================================

/// An edition (collection of chapters).
#[derive(Clone, Debug, Default)]
pub struct Edition {
    /// Unique edition identifier.
    pub uid: Option<u64>,
    /// Whether the edition is hidden.
    pub hidden: bool,
    /// Whether this is the default edition.
    pub default: bool,
    /// Whether chapters are ordered.
    pub ordered: bool,
    /// Chapters in this edition.
    pub chapters: Vec<Chapter>,
}

/// A chapter entry.
#[derive(Clone, Debug, Default)]
pub struct Chapter {
    /// Unique chapter identifier.
    pub uid: u64,
    /// String UID for the chapter.
    pub string_uid: Option<String>,
    /// Start time in nanoseconds.
    pub time_start: u64,
    /// End time in nanoseconds.
    pub time_end: Option<u64>,
    /// Whether the chapter is hidden.
    pub hidden: bool,
    /// Whether the chapter is enabled.
    pub enabled: bool,
    /// Display information (title, language).
    pub display: Vec<ChapterDisplay>,
    /// Nested chapters.
    pub children: Vec<Chapter>,
}

/// Chapter display information.
#[derive(Clone, Debug, Default)]
pub struct ChapterDisplay {
    /// Chapter title.
    pub string: String,
    /// Language of the title (ISO 639-2).
    pub language: String,
    /// Language of the title (BCP 47).
    pub language_ietf: Option<String>,
    /// Country code.
    pub country: Option<String>,
}

// ============================================================================
// Tags (Metadata)
// ============================================================================

/// A tag element containing metadata.
#[derive(Clone, Debug, Default)]
pub struct Tag {
    /// Targeting information.
    pub targets: TagTargets,
    /// Simple tag entries.
    pub simple_tags: Vec<SimpleTag>,
}

/// Tag targeting information.
#[derive(Clone, Debug, Default)]
pub struct TagTargets {
    /// Target type value (50=album, 60=movie, etc.).
    pub target_type_value: Option<u64>,
    /// Target type name.
    pub target_type: Option<String>,
    /// Target track UIDs.
    pub track_uid: Vec<u64>,
    /// Target edition UIDs.
    pub edition_uid: Vec<u64>,
    /// Target chapter UIDs.
    pub chapter_uid: Vec<u64>,
    /// Target attachment UIDs.
    pub attachment_uid: Vec<u64>,
}

/// A simple tag entry.
#[derive(Clone, Debug, Default)]
pub struct SimpleTag {
    /// Tag name.
    pub name: String,
    /// Tag language.
    pub language: String,
    /// Language (BCP 47).
    pub language_ietf: Option<String>,
    /// Whether this is the default language.
    pub default: bool,
    /// String value.
    pub string: Option<String>,
    /// Binary value.
    pub binary: Option<Vec<u8>>,
    /// Nested tags.
    pub children: Vec<SimpleTag>,
}

// ============================================================================
// Attachments
// ============================================================================

/// An attached file.
#[derive(Clone, Debug, Default)]
pub struct AttachedFile {
    /// File description.
    pub description: Option<String>,
    /// File name.
    pub name: String,
    /// MIME type.
    pub mime_type: String,
    /// File data.
    pub data: Vec<u8>,
    /// Unique file identifier.
    pub uid: u64,
}

// ============================================================================
// Seek Head
// ============================================================================

/// Seek entry pointing to a segment element.
#[derive(Clone, Debug, Default)]
pub struct SeekEntry {
    /// Element ID being indexed.
    pub id: u32,
    /// Position in the segment (relative to segment start).
    pub position: u64,
}

// ============================================================================
// Cluster and Block Types
// ============================================================================

/// Cluster state during demuxing.
#[derive(Clone, Debug, Default)]
pub struct ClusterState {
    /// Cluster timestamp in timecode scale units.
    pub timecode: u64,
    /// Position of the cluster in the file.
    pub position: u64,
    /// Size of the cluster (if known).
    pub size: Option<u64>,
    /// Position within the cluster data.
    pub data_position: u64,
}

/// Block header parsed from `SimpleBlock` or Block.
#[derive(Clone, Debug, Default)]
pub struct BlockHeader {
    /// Track number (VINT encoded).
    pub track_number: u64,
    /// Timecode relative to cluster (signed 16-bit).
    pub timecode: i16,
    /// Whether this is a keyframe (from flags).
    pub keyframe: bool,
    /// Whether the frame is invisible.
    pub invisible: bool,
    /// Lacing type used.
    pub lacing: LacingType,
    /// Whether the frame is discardable.
    pub discardable: bool,
}

/// Lacing type for block data.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum LacingType {
    /// No lacing.
    #[default]
    None,
    /// Xiph-style lacing.
    Xiph,
    /// EBML lacing.
    Ebml,
    /// Fixed-size lacing.
    FixedSize,
}

impl TryFrom<u8> for LacingType {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Xiph),
            2 => Ok(Self::FixedSize),
            3 => Ok(Self::Ebml),
            _ => Err(()),
        }
    }
}

/// A parsed block with its data.
#[derive(Clone, Debug, Default)]
pub struct Block {
    /// Block header information.
    pub header: BlockHeader,
    /// Frame data (multiple frames if lacing is used).
    pub frames: Vec<Vec<u8>>,
    /// Block duration (from `BlockGroup`).
    pub duration: Option<u64>,
    /// Reference block timestamps (from `BlockGroup`).
    pub references: Vec<i64>,
    /// Discard padding in nanoseconds.
    pub discard_padding: Option<i64>,
}

impl Block {
    /// Returns true if this is a keyframe.
    #[must_use]
    pub fn is_keyframe(&self) -> bool {
        // A block is a keyframe if the keyframe flag is set
        // OR if there are no reference blocks
        self.header.keyframe || self.references.is_empty()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doc_type() {
        assert!(!DocType::Matroska.is_webm());
        assert!(DocType::WebM.is_webm());
        assert_eq!(DocType::Matroska.as_str(), "matroska");
        assert_eq!(DocType::WebM.as_str(), "webm");
    }

    #[test]
    fn test_ebml_header_default() {
        let header = EbmlHeader::default();
        assert_eq!(header.version, 1);
        assert_eq!(header.max_id_length, 4);
        assert_eq!(header.max_size_length, 8);
        assert_eq!(header.doc_type, DocType::Matroska);
    }

    #[test]
    fn test_segment_info_duration() {
        let mut info = SegmentInfo::default();
        info.duration = Some(60000.0); // 60000 ms in default scale

        // Duration should be 60 seconds
        let secs = info.duration_seconds().expect("operation should succeed");
        assert!((secs - 60.0).abs() < 0.001);
    }

    #[test]
    fn test_track_type_conversion() {
        assert_eq!(TrackType::try_from(1u64), Ok(TrackType::Video));
        assert_eq!(TrackType::try_from(2u64), Ok(TrackType::Audio));
        assert_eq!(TrackType::try_from(0x11u64), Ok(TrackType::Subtitle));
        assert!(TrackType::try_from(255u64).is_err());
    }

    #[test]
    fn test_video_settings_effective_dimensions() {
        let mut settings = VideoSettings::default();
        settings.pixel_width = 1920;
        settings.pixel_height = 1080;

        assert_eq!(settings.effective_display_width(), 1920);
        assert_eq!(settings.effective_display_height(), 1080);

        settings.display_width = Some(1920);
        settings.display_height = Some(800);

        assert_eq!(settings.effective_display_width(), 1920);
        assert_eq!(settings.effective_display_height(), 800);
    }

    #[test]
    fn test_audio_settings_effective_sample_rate() {
        let mut settings = AudioSettings::default();
        settings.sampling_frequency = 44100.0;

        assert!((settings.effective_sample_rate() - 44100.0).abs() < f64::EPSILON);

        settings.output_sampling_frequency = Some(48000.0);
        assert!((settings.effective_sample_rate() - 48000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_track_entry_type_checks() {
        let mut track = TrackEntry::default();

        track.track_type = TrackType::Video;
        assert!(track.is_video());
        assert!(!track.is_audio());
        assert!(!track.is_subtitle());

        track.track_type = TrackType::Audio;
        assert!(!track.is_video());
        assert!(track.is_audio());
        assert!(!track.is_subtitle());

        track.track_type = TrackType::Subtitle;
        assert!(!track.is_video());
        assert!(!track.is_audio());
        assert!(track.is_subtitle());
    }

    #[test]
    fn test_lacing_type_conversion() {
        assert_eq!(LacingType::try_from(0u8), Ok(LacingType::None));
        assert_eq!(LacingType::try_from(1u8), Ok(LacingType::Xiph));
        assert_eq!(LacingType::try_from(2u8), Ok(LacingType::FixedSize));
        assert_eq!(LacingType::try_from(3u8), Ok(LacingType::Ebml));
        assert!(LacingType::try_from(4u8).is_err());
    }

    #[test]
    fn test_block_is_keyframe() {
        let mut block = Block::default();

        // No keyframe flag, no references -> keyframe
        assert!(block.is_keyframe());

        // Keyframe flag set -> keyframe
        block.header.keyframe = true;
        assert!(block.is_keyframe());

        // Keyframe flag not set, has references -> not keyframe
        block.header.keyframe = false;
        block.references.push(-33);
        assert!(!block.is_keyframe());
    }
}
