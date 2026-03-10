//! EBML (Extensible Binary Meta Language) parser.
//!
//! EBML is the binary markup language used by Matroska and `WebM` containers.
//! It uses variable-length integers (VINTs) for element IDs and sizes.
//!
//! # Errors
//!
//! All parser functions return `IResult` from nom. Errors occur for:
//! - Incomplete input (not enough bytes)
//! - Invalid VINT encoding (zero first byte, too long)
//! - Integer overflow

// Allow missing_errors_doc for nom parser functions - error behavior is documented above
#![allow(clippy::missing_errors_doc)]
// Allow unreadable literals for EBML element IDs which have standard hex representations
#![allow(clippy::unreadable_literal)]
#![allow(clippy::unusual_byte_groupings)]

use nom::bytes::complete::take;
use nom::IResult;
use oximedia_core::{OxiError, OxiResult};

// ============================================================================
// EBML Element IDs Module
// ============================================================================

/// All Matroska Element IDs organized by category.
///
/// This module contains constants for all supported EBML/Matroska element IDs.
/// Elements are organized by their function in the container structure.
pub mod element_id {

    // ========================================================================
    // EBML Header Elements
    // ========================================================================

    /// EBML Header - marks the start of an EBML document.
    pub const EBML: u32 = 0x1A45_DFA3;

    /// EBML Version - version of EBML used to create the file.
    pub const EBML_VERSION: u32 = 0x4286;

    /// EBML Read Version - minimum EBML version required to read the file.
    pub const EBML_READ_VERSION: u32 = 0x42F7;

    /// EBML Max ID Length - maximum ID length in bytes.
    pub const EBML_MAX_ID_LENGTH: u32 = 0x42F2;

    /// EBML Max Size Length - maximum size length in bytes.
    pub const EBML_MAX_SIZE_LENGTH: u32 = 0x42F3;

    /// Doc Type - string identifying the document type (e.g., "matroska", "webm").
    pub const DOC_TYPE: u32 = 0x4282;

    /// Doc Type Version - version of the document type.
    pub const DOC_TYPE_VERSION: u32 = 0x4287;

    /// Doc Type Read Version - minimum version required to read the document.
    pub const DOC_TYPE_READ_VERSION: u32 = 0x4285;

    // ========================================================================
    // Segment Elements
    // ========================================================================

    /// Segment - top-level container for all Matroska data.
    pub const SEGMENT: u32 = 0x1853_8067;

    /// Seek Head - index of other elements in the segment.
    pub const SEEK_HEAD: u32 = 0x114D_9B74;

    /// Seek - single seek entry.
    pub const SEEK: u32 = 0x4DBB;

    /// Seek ID - element ID being indexed.
    pub const SEEK_ID: u32 = 0x53AB;

    /// Seek Position - position of the element in the segment.
    pub const SEEK_POSITION: u32 = 0x53AC;

    // ========================================================================
    // Info Elements
    // ========================================================================

    /// Info - segment information (duration, title, etc.).
    pub const INFO: u32 = 0x1549_A966;

    /// Timecode Scale - timestamp scale in nanoseconds (default 1,000,000).
    pub const TIMECODE_SCALE: u32 = 0x2AD7_B1;

    /// Duration - duration of the segment in timecode scale units.
    pub const DURATION: u32 = 0x4489;

    /// Title - title of the segment.
    pub const TITLE: u32 = 0x7BA9;

    /// Muxing App - application that created the file.
    pub const MUXING_APP: u32 = 0x4D80;

    /// Writing App - application that wrote the file.
    pub const WRITING_APP: u32 = 0x5741;

    /// Date UTC - date the file was created.
    pub const DATE_UTC: u32 = 0x4461;

    /// Segment UID - unique identifier for the segment.
    pub const SEGMENT_UID: u32 = 0x73A4;

    /// Segment Filename - filename of the segment.
    pub const SEGMENT_FILENAME: u32 = 0x7384;

    /// Prev UID - UID of the previous segment.
    pub const PREV_UID: u32 = 0x3CB923;

    /// Next UID - UID of the next segment.
    pub const NEXT_UID: u32 = 0x3EB923;

    // ========================================================================
    // Track Elements
    // ========================================================================

    /// Tracks - track descriptions.
    pub const TRACKS: u32 = 0x1654_AE6B;

    /// Track Entry - describes a single track.
    pub const TRACK_ENTRY: u32 = 0xAE;

    /// Track Number - track number for this entry.
    pub const TRACK_NUMBER: u32 = 0xD7;

    /// Track UID - unique identifier for the track.
    pub const TRACK_UID: u32 = 0x73C5;

    /// Track Type - type of track (video, audio, etc.).
    pub const TRACK_TYPE: u32 = 0x83;

    /// Flag Enabled - whether the track is usable.
    pub const FLAG_ENABLED: u32 = 0xB9;

    /// Flag Default - whether this is the default track.
    pub const FLAG_DEFAULT: u32 = 0x88;

    /// Flag Forced - whether the track is forced.
    pub const FLAG_FORCED: u32 = 0x55AA;

    /// Flag Lacing - whether lacing is used.
    pub const FLAG_LACING: u32 = 0x9C;

    /// Min Cache - minimum cache required.
    pub const MIN_CACHE: u32 = 0x6DE7;

    /// Max Cache - maximum cache allowed.
    pub const MAX_CACHE: u32 = 0x6DF8;

    /// Default Duration - default duration of each frame in nanoseconds.
    pub const DEFAULT_DURATION: u32 = 0x23E383;

    /// Track Timecode Scale - DEPRECATED, should be 1.0.
    pub const TRACK_TIMECODE_SCALE: u32 = 0x23314F;

    /// Max Block Addition ID - maximum block addition ID.
    pub const MAX_BLOCK_ADDITION_ID: u32 = 0x55EE;

    /// Name - human-readable track name.
    pub const NAME: u32 = 0x536E;

    /// Language - language of the track (ISO 639-2).
    pub const LANGUAGE: u32 = 0x22B5_9C;

    /// Language IETF - language using BCP 47.
    pub const LANGUAGE_IETF: u32 = 0x22B5_9D;

    /// Codec ID - codec identifier string.
    pub const CODEC_ID: u32 = 0x86;

    /// Codec Private - codec-specific private data.
    pub const CODEC_PRIVATE: u32 = 0x63A2;

    /// Codec Name - human-readable codec name.
    pub const CODEC_NAME: u32 = 0x2586_88;

    /// Codec Delay - codec delay in nanoseconds.
    pub const CODEC_DELAY: u32 = 0x56AA;

    /// Seek Pre Roll - seek pre-roll in nanoseconds.
    pub const SEEK_PRE_ROLL: u32 = 0x56BB;

    /// Attachment Link - UID of an attachment.
    pub const ATTACHMENT_LINK: u32 = 0x7446;

    // ========================================================================
    // Video Elements
    // ========================================================================

    /// Video - video settings element.
    pub const VIDEO: u32 = 0xE0;

    /// Flag Interlaced - interlacing flag.
    pub const FLAG_INTERLACED: u32 = 0x9A;

    /// Field Order - field order for interlaced content.
    pub const FIELD_ORDER: u32 = 0x9D;

    /// Stereo Mode - stereo-3D video mode.
    pub const STEREO_MODE: u32 = 0x53B8;

    /// Alpha Mode - alpha video mode.
    pub const ALPHA_MODE: u32 = 0x53C0;

    /// Pixel Width - width in pixels.
    pub const PIXEL_WIDTH: u32 = 0xB0;

    /// Pixel Height - height in pixels.
    pub const PIXEL_HEIGHT: u32 = 0xBA;

    /// Pixel Crop Bottom - pixels cropped from bottom.
    pub const PIXEL_CROP_BOTTOM: u32 = 0x54AA;

    /// Pixel Crop Top - pixels cropped from top.
    pub const PIXEL_CROP_TOP: u32 = 0x54BB;

    /// Pixel Crop Left - pixels cropped from left.
    pub const PIXEL_CROP_LEFT: u32 = 0x54CC;

    /// Pixel Crop Right - pixels cropped from right.
    pub const PIXEL_CROP_RIGHT: u32 = 0x54DD;

    /// Display Width - display width.
    pub const DISPLAY_WIDTH: u32 = 0x54B0;

    /// Display Height - display height.
    pub const DISPLAY_HEIGHT: u32 = 0x54BA;

    /// Display Unit - display unit (pixels, cm, inches, aspect ratio).
    pub const DISPLAY_UNIT: u32 = 0x54B2;

    /// Aspect Ratio Type - aspect ratio behavior.
    pub const ASPECT_RATIO_TYPE: u32 = 0x54B3;

    /// Colour Space - color space binary data.
    pub const COLOUR_SPACE: u32 = 0x2EB5_24;

    // ========================================================================
    // Video Colour Elements
    // ========================================================================

    /// Colour - color settings element.
    pub const COLOUR: u32 = 0x55B0;

    /// Matrix Coefficients - color matrix coefficients.
    pub const MATRIX_COEFFICIENTS: u32 = 0x55B1;

    /// Bits Per Channel - bits per color channel.
    pub const BITS_PER_CHANNEL: u32 = 0x55B2;

    /// Chroma Subsampling Horizontal - chroma horizontal subsampling.
    pub const CHROMA_SUBSAMPLING_HORZ: u32 = 0x55B3;

    /// Chroma Subsampling Vertical - chroma vertical subsampling.
    pub const CHROMA_SUBSAMPLING_VERT: u32 = 0x55B4;

    /// Cb Subsampling Horizontal - Cb horizontal subsampling.
    pub const CB_SUBSAMPLING_HORZ: u32 = 0x55B5;

    /// Cb Subsampling Vertical - Cb vertical subsampling.
    pub const CB_SUBSAMPLING_VERT: u32 = 0x55B6;

    /// Chroma Siting Horizontal - chroma horizontal siting.
    pub const CHROMA_SITING_HORZ: u32 = 0x55B7;

    /// Chroma Siting Vertical - chroma vertical siting.
    pub const CHROMA_SITING_VERT: u32 = 0x55B8;

    /// Range - color range (broadcast vs full).
    pub const RANGE: u32 = 0x55B9;

    /// Transfer Characteristics - transfer function.
    pub const TRANSFER_CHARACTERISTICS: u32 = 0x55BA;

    /// Primaries - color primaries.
    pub const PRIMARIES: u32 = 0x55BB;

    /// Max CLL - maximum content light level.
    pub const MAX_CLL: u32 = 0x55BC;

    /// Max FALL - maximum frame-average light level.
    pub const MAX_FALL: u32 = 0x55BD;

    /// Mastering Metadata - mastering display metadata.
    pub const MASTERING_METADATA: u32 = 0x55D0;

    /// Primary R Chromaticity X - red chromaticity X.
    pub const PRIMARY_R_CHROMATICITY_X: u32 = 0x55D1;

    /// Primary R Chromaticity Y - red chromaticity Y.
    pub const PRIMARY_R_CHROMATICITY_Y: u32 = 0x55D2;

    /// Primary G Chromaticity X - green chromaticity X.
    pub const PRIMARY_G_CHROMATICITY_X: u32 = 0x55D3;

    /// Primary G Chromaticity Y - green chromaticity Y.
    pub const PRIMARY_G_CHROMATICITY_Y: u32 = 0x55D4;

    /// Primary B Chromaticity X - blue chromaticity X.
    pub const PRIMARY_B_CHROMATICITY_X: u32 = 0x55D5;

    /// Primary B Chromaticity Y - blue chromaticity Y.
    pub const PRIMARY_B_CHROMATICITY_Y: u32 = 0x55D6;

    /// White Point Chromaticity X - white point X.
    pub const WHITE_POINT_CHROMATICITY_X: u32 = 0x55D7;

    /// White Point Chromaticity Y - white point Y.
    pub const WHITE_POINT_CHROMATICITY_Y: u32 = 0x55D8;

    /// Luminance Max - maximum luminance.
    pub const LUMINANCE_MAX: u32 = 0x55D9;

    /// Luminance Min - minimum luminance.
    pub const LUMINANCE_MIN: u32 = 0x55DA;

    // ========================================================================
    // Audio Elements
    // ========================================================================

    /// Audio - audio settings element.
    pub const AUDIO: u32 = 0xE1;

    /// Sampling Frequency - sample rate in Hz.
    pub const SAMPLING_FREQUENCY: u32 = 0xB5;

    /// Output Sampling Frequency - output sample rate.
    pub const OUTPUT_SAMPLING_FREQUENCY: u32 = 0x78B5;

    /// Channels - number of audio channels.
    pub const CHANNELS: u32 = 0x9F;

    /// Bit Depth - bits per sample.
    pub const BIT_DEPTH: u32 = 0x6264;

    // ========================================================================
    // Content Encoding Elements
    // ========================================================================

    /// Content Encodings - content encoding settings.
    pub const CONTENT_ENCODINGS: u32 = 0x6D80;

    /// Content Encoding - single content encoding.
    pub const CONTENT_ENCODING: u32 = 0x6240;

    /// Content Encoding Order - encoding order.
    pub const CONTENT_ENCODING_ORDER: u32 = 0x5031;

    /// Content Encoding Scope - encoding scope.
    pub const CONTENT_ENCODING_SCOPE: u32 = 0x5032;

    /// Content Encoding Type - encoding type.
    pub const CONTENT_ENCODING_TYPE: u32 = 0x5033;

    /// Content Compression - compression settings.
    pub const CONTENT_COMPRESSION: u32 = 0x5034;

    /// Content Comp Algo - compression algorithm.
    pub const CONTENT_COMP_ALGO: u32 = 0x4254;

    /// Content Comp Settings - compression settings data.
    pub const CONTENT_COMP_SETTINGS: u32 = 0x4255;

    // ========================================================================
    // Cluster Elements
    // ========================================================================

    /// Cluster - block of data for one or more tracks.
    pub const CLUSTER: u32 = 0x1F43_B675;

    /// Timestamp - cluster timestamp in timecode scale units.
    pub const TIMESTAMP: u32 = 0xE7;

    /// Silent Tracks - tracks with no data in this cluster.
    pub const SILENT_TRACKS: u32 = 0x5854;

    /// Silent Track Number - track number of silent track.
    pub const SILENT_TRACK_NUMBER: u32 = 0x58D7;

    /// Position - position of the cluster in the segment.
    pub const POSITION: u32 = 0xA7;

    /// Prev Size - size of the previous cluster.
    pub const PREV_SIZE: u32 = 0xAB;

    /// Simple Block - a block without lacing.
    pub const SIMPLE_BLOCK: u32 = 0xA3;

    /// Block Group - a group containing a block and related elements.
    pub const BLOCK_GROUP: u32 = 0xA0;

    /// Block - actual data block.
    pub const BLOCK: u32 = 0xA1;

    /// Block Virtual - virtual block (DEPRECATED).
    pub const BLOCK_VIRTUAL: u32 = 0xA2;

    /// Block Additions - additional blocks.
    pub const BLOCK_ADDITIONS: u32 = 0x75A1;

    /// Block More - additional block data.
    pub const BLOCK_MORE: u32 = 0xA6;

    /// Block Add ID - ID for block addition.
    pub const BLOCK_ADD_ID: u32 = 0xEE;

    /// Block Additional - additional block data.
    pub const BLOCK_ADDITIONAL: u32 = 0xA5;

    /// Block Duration - duration of the block.
    pub const BLOCK_DURATION: u32 = 0x9B;

    /// Reference Priority - reference priority.
    pub const REFERENCE_PRIORITY: u32 = 0xFA;

    /// Reference Block - reference to another block.
    pub const REFERENCE_BLOCK: u32 = 0xFB;

    /// Reference Virtual - virtual reference (DEPRECATED).
    pub const REFERENCE_VIRTUAL: u32 = 0xFD;

    /// Codec State - codec state data.
    pub const CODEC_STATE: u32 = 0xA4;

    /// Discard Padding - discard padding in nanoseconds.
    pub const DISCARD_PADDING: u32 = 0x75A2;

    // ========================================================================
    // Cues Elements
    // ========================================================================

    /// Cues - seek index for the segment.
    pub const CUES: u32 = 0x1C53_BB6B;

    /// Cue Point - single cue entry.
    pub const CUE_POINT: u32 = 0xBB;

    /// Cue Time - cue timestamp.
    pub const CUE_TIME: u32 = 0xB3;

    /// Cue Track Positions - track positions for the cue.
    pub const CUE_TRACK_POSITIONS: u32 = 0xB7;

    /// Cue Track - track number.
    pub const CUE_TRACK: u32 = 0xF7;

    /// Cue Cluster Position - position of the cluster.
    pub const CUE_CLUSTER_POSITION: u32 = 0xF1;

    /// Cue Relative Position - relative position in the cluster.
    pub const CUE_RELATIVE_POSITION: u32 = 0xF0;

    /// Cue Duration - duration of the cue.
    pub const CUE_DURATION: u32 = 0xB2;

    /// Cue Block Number - block number within the cluster.
    pub const CUE_BLOCK_NUMBER: u32 = 0x5378;

    /// Cue Codec State - codec state at this cue point.
    pub const CUE_CODEC_STATE: u32 = 0xEA;

    /// Cue Reference - reference information.
    pub const CUE_REFERENCE: u32 = 0xDB;

    /// Cue Ref Time - reference time.
    pub const CUE_REF_TIME: u32 = 0x96;

    // ========================================================================
    // Attachments Elements
    // ========================================================================

    /// Attachments - attached files.
    pub const ATTACHMENTS: u32 = 0x1941_A469;

    /// Attached File - single attached file.
    pub const ATTACHED_FILE: u32 = 0x61A7;

    /// File Description - description of the file.
    pub const FILE_DESCRIPTION: u32 = 0x467E;

    /// File Name - name of the file.
    pub const FILE_NAME: u32 = 0x466E;

    /// File Mime Type - MIME type of the file.
    pub const FILE_MIME_TYPE: u32 = 0x4660;

    /// File Data - file content.
    pub const FILE_DATA: u32 = 0x465C;

    /// File UID - unique identifier for the file.
    pub const FILE_UID: u32 = 0x46AE;

    // ========================================================================
    // Chapters Elements
    // ========================================================================

    /// Chapters - chapter information.
    pub const CHAPTERS: u32 = 0x1043_A770;

    /// Edition Entry - single edition.
    pub const EDITION_ENTRY: u32 = 0x45B9;

    /// Edition UID - unique identifier for the edition.
    pub const EDITION_UID: u32 = 0x45BC;

    /// Edition Flag Hidden - whether the edition is hidden.
    pub const EDITION_FLAG_HIDDEN: u32 = 0x45BD;

    /// Edition Flag Default - whether this is the default edition.
    pub const EDITION_FLAG_DEFAULT: u32 = 0x45DB;

    /// Edition Flag Ordered - whether chapters are ordered.
    pub const EDITION_FLAG_ORDERED: u32 = 0x45DD;

    /// Chapter Atom - single chapter.
    pub const CHAPTER_ATOM: u32 = 0xB6;

    /// Chapter UID - unique identifier for the chapter.
    pub const CHAPTER_UID: u32 = 0x73C4;

    /// Chapter String UID - string UID for the chapter.
    pub const CHAPTER_STRING_UID: u32 = 0x5654;

    /// Chapter Time Start - start time of the chapter.
    pub const CHAPTER_TIME_START: u32 = 0x91;

    /// Chapter Time End - end time of the chapter.
    pub const CHAPTER_TIME_END: u32 = 0x92;

    /// Chapter Flag Hidden - whether the chapter is hidden.
    pub const CHAPTER_FLAG_HIDDEN: u32 = 0x98;

    /// Chapter Flag Enabled - whether the chapter is enabled.
    pub const CHAPTER_FLAG_ENABLED: u32 = 0x4598;

    /// Chapter Segment UID - UID of the segment containing the chapter.
    pub const CHAPTER_SEGMENT_UID: u32 = 0x6E67;

    /// Chapter Segment Edition UID - edition UID in the linked segment.
    pub const CHAPTER_SEGMENT_EDITION_UID: u32 = 0x6EBC;

    /// Chapter Physical Equiv - physical equivalent level.
    pub const CHAPTER_PHYSICAL_EQUIV: u32 = 0x63C3;

    /// Chapter Track - tracks associated with the chapter.
    pub const CHAPTER_TRACK: u32 = 0x8F;

    /// Chapter Track Number - track number in the chapter.
    pub const CHAPTER_TRACK_NUMBER: u32 = 0x89;

    /// Chapter Display - display information for the chapter.
    pub const CHAPTER_DISPLAY: u32 = 0x80;

    /// Chap String - chapter title string.
    pub const CHAP_STRING: u32 = 0x85;

    /// Chap Language - language of the chapter title.
    pub const CHAP_LANGUAGE: u32 = 0x437C;

    /// Chap Language IETF - language using BCP 47.
    pub const CHAP_LANGUAGE_IETF: u32 = 0x437D;

    /// Chap Country - country for the chapter title.
    pub const CHAP_COUNTRY: u32 = 0x437E;

    /// Chapter Process - chapter process information.
    pub const CHAPTER_PROCESS: u32 = 0x6944;

    /// Chapter Process Codec ID - codec ID for chapter process.
    pub const CHAPTER_PROCESS_CODEC_ID: u32 = 0x6955;

    /// Chapter Process Private - private data for chapter process.
    pub const CHAPTER_PROCESS_PRIVATE: u32 = 0x450D;

    /// Chapter Process Command - command for chapter process.
    pub const CHAPTER_PROCESS_COMMAND: u32 = 0x6911;

    /// Chapter Process Time - time for chapter process command.
    pub const CHAPTER_PROCESS_TIME: u32 = 0x6922;

    /// Chapter Process Data - data for chapter process command.
    pub const CHAPTER_PROCESS_DATA: u32 = 0x6933;

    // ========================================================================
    // Tags Elements
    // ========================================================================

    /// Tags - metadata tags.
    pub const TAGS: u32 = 0x1254_C367;

    /// Tag - single tag element.
    pub const TAG: u32 = 0x7373;

    /// Targets - targeting information for the tag.
    pub const TARGETS: u32 = 0x63C0;

    /// Target Type Value - numeric target type.
    pub const TARGET_TYPE_VALUE: u32 = 0x68CA;

    /// Target Type - string target type.
    pub const TARGET_TYPE: u32 = 0x63CA;

    /// Tag Track UID - UID of the target track.
    pub const TAG_TRACK_UID: u32 = 0x63C5;

    /// Tag Edition UID - UID of the target edition.
    pub const TAG_EDITION_UID: u32 = 0x63C9;

    /// Tag Chapter UID - UID of the target chapter.
    pub const TAG_CHAPTER_UID: u32 = 0x63C4;

    /// Tag Attachment UID - UID of the target attachment.
    pub const TAG_ATTACHMENT_UID: u32 = 0x63C6;

    /// Simple Tag - single metadata entry.
    pub const SIMPLE_TAG: u32 = 0x67C8;

    /// Tag Name - name of the tag.
    pub const TAG_NAME: u32 = 0x45A3;

    /// Tag Language - language of the tag value.
    pub const TAG_LANGUAGE: u32 = 0x447A;

    /// Tag Language IETF - language using BCP 47.
    pub const TAG_LANGUAGE_IETF: u32 = 0x447B;

    /// Tag Default - whether this is the default language.
    pub const TAG_DEFAULT: u32 = 0x4484;

    /// Tag String - string value of the tag.
    pub const TAG_STRING: u32 = 0x4487;

    /// Tag Binary - binary value of the tag.
    pub const TAG_BINARY: u32 = 0x4485;

    // ========================================================================
    // Void and CRC Elements
    // ========================================================================

    /// Void - padding element.
    pub const VOID: u32 = 0xEC;

    /// CRC-32 - CRC-32 checksum.
    pub const CRC32: u32 = 0xBF;
}

// Re-export legacy constants for backwards compatibility
pub use element_id::EBML as EBML_HEADER;
pub use element_id::INFO as SEGMENT_INFO;
pub use element_id::{
    ATTACHMENTS, BLOCK, BLOCK_DURATION, BLOCK_GROUP, CHAPTERS, CLUSTER, CODEC_ID, CODEC_PRIVATE,
    CUES, DOC_TYPE, DOC_TYPE_READ_VERSION, DOC_TYPE_VERSION, EBML_MAX_ID_LENGTH,
    EBML_MAX_SIZE_LENGTH, EBML_READ_VERSION, EBML_VERSION, SEGMENT, SIMPLE_BLOCK, TAGS, TIMESTAMP,
    TRACKS, TRACK_ENTRY, TRACK_NUMBER, TRACK_TYPE, TRACK_UID,
};

// ============================================================================
// EBML Element Structure
// ============================================================================

/// Parsed EBML element header.
///
/// An EBML element consists of:
/// - Element ID (variable-length)
/// - Element size (variable-length)
/// - Element data (size bytes)
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EbmlElement {
    /// Element ID.
    pub id: u32,

    /// Size of the element data in bytes.
    ///
    /// A size of `u64::MAX` indicates an unknown/unbounded size,
    /// which is valid for streaming or the Segment element.
    pub size: u64,

    /// Size of the header (ID + size fields) in bytes.
    pub header_size: usize,
}

impl EbmlElement {
    /// Returns true if this element has an unknown/unbounded size.
    #[must_use]
    pub const fn is_unbounded(&self) -> bool {
        self.size == u64::MAX
    }

    /// Returns the total size of the element (header + data).
    ///
    /// Returns `None` for unbounded elements.
    #[must_use]
    pub const fn total_size(&self) -> Option<u64> {
        if self.is_unbounded() {
            None
        } else {
            Some(self.header_size as u64 + self.size)
        }
    }

    /// Returns true if this is a master element (contains other elements).
    #[must_use]
    pub const fn is_master(&self) -> bool {
        matches!(
            self.id,
            element_id::EBML
                | element_id::SEGMENT
                | element_id::SEEK_HEAD
                | element_id::SEEK
                | element_id::INFO
                | element_id::TRACKS
                | element_id::TRACK_ENTRY
                | element_id::VIDEO
                | element_id::AUDIO
                | element_id::COLOUR
                | element_id::MASTERING_METADATA
                | element_id::CONTENT_ENCODINGS
                | element_id::CONTENT_ENCODING
                | element_id::CONTENT_COMPRESSION
                | element_id::CLUSTER
                | element_id::BLOCK_GROUP
                | element_id::BLOCK_ADDITIONS
                | element_id::BLOCK_MORE
                | element_id::CUES
                | element_id::CUE_POINT
                | element_id::CUE_TRACK_POSITIONS
                | element_id::CUE_REFERENCE
                | element_id::ATTACHMENTS
                | element_id::ATTACHED_FILE
                | element_id::CHAPTERS
                | element_id::EDITION_ENTRY
                | element_id::CHAPTER_ATOM
                | element_id::CHAPTER_TRACK
                | element_id::CHAPTER_DISPLAY
                | element_id::CHAPTER_PROCESS
                | element_id::CHAPTER_PROCESS_COMMAND
                | element_id::TAGS
                | element_id::TAG
                | element_id::TARGETS
                | element_id::SIMPLE_TAG
                | element_id::SILENT_TRACKS
        )
    }
}

// ============================================================================
// VINT Parsing
// ============================================================================

/// Parse an EBML variable-length integer (VINT).
///
/// VINTs use a leading 1-bit to indicate length:
/// - `1xxxxxxx` = 1 byte (7 bits of data)
/// - `01xxxxxx xxxxxxxx` = 2 bytes (14 bits of data)
/// - `001xxxxx ...` = 3 bytes (21 bits of data)
/// - etc.
///
/// # Arguments
///
/// * `input` - Input bytes to parse
///
/// # Returns
///
/// The parsed value and remaining input, or an error.
///
/// # Errors
///
/// Returns a nom error if input is empty, starts with zero, or is incomplete.
///
/// # Example
///
/// ```
/// use oximedia_container::demux::matroska::ebml::parse_vint;
///
/// // Single byte: 0x81 = 1xxxxxxx, value = 1
/// let (remaining, value) = parse_vint(&[0x81, 0x00])?;
/// assert_eq!(value, 1);
/// assert_eq!(remaining, &[0x00]);
///
/// // Two bytes: 0x40 0x01 = 01xxxxxx xxxxxxxx, value = 1
/// let (remaining, value) = parse_vint(&[0x40, 0x01])?;
/// assert_eq!(value, 1);
/// ```
pub fn parse_vint(input: &[u8]) -> IResult<&[u8], u64> {
    if input.is_empty() {
        return Err(nom::Err::Incomplete(nom::Needed::new(1)));
    }

    let first = input[0];
    if first == 0 {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Fail,
        )));
    }

    // Count leading zeros to determine length
    let len = first.leading_zeros() as usize + 1;
    if len > 8 {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Fail,
        )));
    }

    if input.len() < len {
        return Err(nom::Err::Incomplete(nom::Needed::new(len - input.len())));
    }

    // Extract value, masking off the length marker bit
    let mask = (1u8 << (8 - len)) - 1;
    let mut value = u64::from(first & mask);

    for byte in input.iter().take(len).skip(1) {
        value = (value << 8) | u64::from(*byte);
    }

    Ok((&input[len..], value))
}

/// Parse an EBML element ID.
///
/// Element IDs are VINTs that include the length marker bit as part of the ID.
/// This is different from size VINTs where the marker is stripped.
///
/// # Errors
///
/// Returns a nom error if input is empty, starts with zero, exceeds 4 bytes, or is incomplete.
///
/// # Example
///
/// ```
/// use oximedia_container::demux::matroska::ebml::parse_element_id;
///
/// // EBML Header ID: 0x1A 0x45 0xDF 0xA3 = 0x1A45DFA3
/// let (remaining, id) = parse_element_id(&[0x1A, 0x45, 0xDF, 0xA3, 0x00])?;
/// assert_eq!(id, 0x1A45DFA3);
/// ```
pub fn parse_element_id(input: &[u8]) -> IResult<&[u8], u32> {
    if input.is_empty() {
        return Err(nom::Err::Incomplete(nom::Needed::new(1)));
    }

    let first = input[0];
    if first == 0 {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Fail,
        )));
    }

    // Count leading zeros to determine length
    let len = first.leading_zeros() as usize + 1;
    if len > 4 {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Fail,
        )));
    }

    if input.len() < len {
        return Err(nom::Err::Incomplete(nom::Needed::new(len - input.len())));
    }

    // For IDs, we keep the marker bit
    let mut id = u32::from(first);
    for byte in input.iter().take(len).skip(1) {
        id = (id << 8) | u32::from(*byte);
    }

    Ok((&input[len..], id))
}

/// Parse an EBML element size.
///
/// Returns `u64::MAX` for unknown/unbounded sizes (all 1-bits after marker).
///
/// # Errors
///
/// Returns a nom error if input is empty, starts with zero, exceeds 8 bytes, or is incomplete.
pub fn parse_element_size(input: &[u8]) -> IResult<&[u8], u64> {
    if input.is_empty() {
        return Err(nom::Err::Incomplete(nom::Needed::new(1)));
    }

    let first = input[0];
    if first == 0 {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Fail,
        )));
    }

    let len = first.leading_zeros() as usize + 1;
    if len > 8 {
        return Err(nom::Err::Error(nom::error::Error::new(
            input,
            nom::error::ErrorKind::Fail,
        )));
    }

    if input.len() < len {
        return Err(nom::Err::Incomplete(nom::Needed::new(len - input.len())));
    }

    // Parse the VINT value
    let (remaining, value) = parse_vint(input)?;

    // Check for unknown size (all 1-bits in data portion)
    let max_value = (1u64 << (7 * len)) - 1;
    if value == max_value {
        return Ok((remaining, u64::MAX));
    }

    Ok((remaining, value))
}

/// Parse an EBML element header (ID + size).
///
/// # Errors
///
/// Returns a nom error if parsing the element ID or size fails.
///
/// # Example
///
/// ```
/// use oximedia_container::demux::matroska::ebml::{parse_element_header, EBML_HEADER};
///
/// // EBML Header: ID=0x1A45DFA3, Size=0x1F (31 bytes)
/// let data = [0x1A, 0x45, 0xDF, 0xA3, 0x9F, 0x00];
/// let (remaining, element) = parse_element_header(&data)?;
///
/// assert_eq!(element.id, EBML_HEADER);
/// assert_eq!(element.size, 31);
/// assert_eq!(element.header_size, 5);
/// ```
pub fn parse_element_header(input: &[u8]) -> IResult<&[u8], EbmlElement> {
    let start_len = input.len();

    // Parse element ID
    let (after_id, id) = parse_element_id(input)?;

    // Parse element size
    let (remaining, size) = parse_element_size(after_id)?;

    let header_size = start_len - remaining.len();

    Ok((
        remaining,
        EbmlElement {
            id,
            size,
            header_size,
        },
    ))
}

/// Read an EBML unsigned integer.
///
/// EBML stores unsigned integers in big-endian format with minimal bytes.
///
/// # Errors
///
/// Returns a nom error if the input is empty or exceeds 8 bytes.
pub fn read_uint(data: &[u8]) -> IResult<&[u8], u64> {
    if data.is_empty() {
        return Ok((data, 0));
    }

    if data.len() > 8 {
        return Err(nom::Err::Error(nom::error::Error::new(
            data,
            nom::error::ErrorKind::TooLarge,
        )));
    }

    let (remaining, bytes) = take(data.len())(data)?;
    let mut value = 0u64;
    for &byte in bytes {
        value = (value << 8) | u64::from(byte);
    }

    Ok((remaining, value))
}

/// Read an EBML signed integer.
///
/// EBML stores signed integers in two's complement big-endian format.
///
/// # Errors
///
/// Returns a nom error if the input is empty or exceeds 8 bytes.
pub fn read_int(data: &[u8]) -> IResult<&[u8], i64> {
    let (remaining, unsigned) = read_uint(data)?;

    // Sign extend if necessary
    #[allow(clippy::cast_possible_wrap)]
    let signed = if data.is_empty() {
        0i64
    } else if data[0] & 0x80 != 0 {
        // Negative number - sign extend
        let shift = 64 - (data.len() * 8);
        ((unsigned << shift) as i64) >> shift
    } else {
        unsigned as i64
    };

    Ok((remaining, signed))
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Read a UTF-8 string from EBML data.
///
/// # Arguments
///
/// * `data` - Raw bytes containing the string
/// * `len` - Number of bytes to read
///
/// # Errors
///
/// Returns an error if the data is not valid UTF-8 or if len exceeds data length.
///
/// # Example
///
/// ```
/// use oximedia_container::demux::matroska::ebml::read_string;
///
/// let result = read_string(b"webm\0\0\0", 4)?;
/// assert_eq!(result, "webm");
/// ```
pub fn read_string(data: &[u8], len: usize) -> OxiResult<String> {
    if len > data.len() {
        return Err(OxiError::InvalidData(format!(
            "String length {} exceeds data length {}",
            len,
            data.len()
        )));
    }

    let bytes = &data[..len];
    // Strip trailing null bytes (common in EBML strings)
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(len);
    let trimmed = &bytes[..end];

    String::from_utf8(trimmed.to_vec())
        .map_err(|e| OxiError::InvalidData(format!("Invalid UTF-8 string: {e}")))
}

/// Read a floating-point number from EBML data.
///
/// EBML supports 4-byte (f32) and 8-byte (f64) floats in big-endian format.
///
/// # Arguments
///
/// * `data` - Raw bytes containing the float
/// * `len` - Number of bytes (must be 0, 4, or 8)
///
/// # Errors
///
/// Returns an error if len is not 0, 4, or 8.
///
/// # Example
///
/// ```
/// use oximedia_container::demux::matroska::ebml::read_float;
///
/// // 8-byte double representing 48000.0
/// let bytes = 48000.0_f64.to_be_bytes();
/// let result = read_float(&bytes, 8)?;
/// assert!((result - 48000.0).abs() < f64::EPSILON);
/// ```
pub fn read_float(data: &[u8], len: usize) -> OxiResult<f64> {
    match len {
        0 => Ok(0.0),
        4 => {
            if data.len() < 4 {
                return Err(OxiError::InvalidData("Not enough bytes for f32".into()));
            }
            let bytes: [u8; 4] = data[..4]
                .try_into()
                .map_err(|_| OxiError::InvalidData("Failed to convert bytes to f32".into()))?;
            Ok(f64::from(f32::from_be_bytes(bytes)))
        }
        8 => {
            if data.len() < 8 {
                return Err(OxiError::InvalidData("Not enough bytes for f64".into()));
            }
            let bytes: [u8; 8] = data[..8]
                .try_into()
                .map_err(|_| OxiError::InvalidData("Failed to convert bytes to f64".into()))?;
            Ok(f64::from_be_bytes(bytes))
        }
        _ => Err(OxiError::InvalidData(format!(
            "Invalid float size: {len} (expected 0, 4, or 8)"
        ))),
    }
}

/// Read an EBML date value.
///
/// EBML dates are stored as signed 64-bit integers representing nanoseconds
/// since 2001-01-01T00:00:00.000000000 UTC.
///
/// # Arguments
///
/// * `data` - Raw bytes containing the date
/// * `len` - Number of bytes (must be 0 or 8)
///
/// # Errors
///
/// Returns an error if len is not 0 or 8.
///
/// # Example
///
/// ```
/// use oximedia_container::demux::matroska::ebml::read_date;
///
/// // Zero timestamp (2001-01-01)
/// let result = read_date(&[0; 8], 8)?;
/// assert_eq!(result, 0);
/// ```
pub fn read_date(data: &[u8], len: usize) -> OxiResult<i64> {
    match len {
        0 => Ok(0),
        8 => {
            if data.len() < 8 {
                return Err(OxiError::InvalidData("Not enough bytes for date".into()));
            }
            let bytes: [u8; 8] = data[..8]
                .try_into()
                .map_err(|_| OxiError::InvalidData("Failed to convert bytes to i64".into()))?;
            Ok(i64::from_be_bytes(bytes))
        }
        _ => {
            // For non-standard lengths, try to read as a signed integer
            if len > 8 || len > data.len() {
                return Err(OxiError::InvalidData(format!("Invalid date size: {len}")));
            }
            let (_, value) = read_int(&data[..len])
                .map_err(|_| OxiError::InvalidData("Failed to parse date as integer".into()))?;
            Ok(value)
        }
    }
}

/// Read a binary blob from EBML data.
///
/// Simply extracts the specified number of bytes.
///
/// # Arguments
///
/// * `data` - Raw bytes to extract from
/// * `len` - Number of bytes to read
///
/// # Errors
///
/// Returns an error if len exceeds data length.
pub fn read_binary(data: &[u8], len: usize) -> OxiResult<Vec<u8>> {
    if len > data.len() {
        return Err(OxiError::InvalidData(format!(
            "Binary length {} exceeds data length {}",
            len,
            data.len()
        )));
    }
    Ok(data[..len].to_vec())
}

/// Get the element name for debugging purposes.
///
/// Returns a human-readable name for known element IDs.
#[must_use]
pub fn element_name(id: u32) -> &'static str {
    match id {
        element_id::EBML => "EBML",
        element_id::EBML_VERSION => "EBMLVersion",
        element_id::EBML_READ_VERSION => "EBMLReadVersion",
        element_id::EBML_MAX_ID_LENGTH => "EBMLMaxIDLength",
        element_id::EBML_MAX_SIZE_LENGTH => "EBMLMaxSizeLength",
        element_id::DOC_TYPE => "DocType",
        element_id::DOC_TYPE_VERSION => "DocTypeVersion",
        element_id::DOC_TYPE_READ_VERSION => "DocTypeReadVersion",
        element_id::SEGMENT => "Segment",
        element_id::SEEK_HEAD => "SeekHead",
        element_id::SEEK => "Seek",
        element_id::SEEK_ID => "SeekID",
        element_id::SEEK_POSITION => "SeekPosition",
        element_id::INFO => "Info",
        element_id::TIMECODE_SCALE => "TimecodeScale",
        element_id::DURATION => "Duration",
        element_id::TITLE => "Title",
        element_id::MUXING_APP => "MuxingApp",
        element_id::WRITING_APP => "WritingApp",
        element_id::DATE_UTC => "DateUTC",
        element_id::SEGMENT_UID => "SegmentUID",
        element_id::TRACKS => "Tracks",
        element_id::TRACK_ENTRY => "TrackEntry",
        element_id::TRACK_NUMBER => "TrackNumber",
        element_id::TRACK_UID => "TrackUID",
        element_id::TRACK_TYPE => "TrackType",
        element_id::FLAG_ENABLED => "FlagEnabled",
        element_id::FLAG_DEFAULT => "FlagDefault",
        element_id::FLAG_FORCED => "FlagForced",
        element_id::FLAG_LACING => "FlagLacing",
        element_id::DEFAULT_DURATION => "DefaultDuration",
        element_id::NAME => "Name",
        element_id::LANGUAGE => "Language",
        element_id::CODEC_ID => "CodecID",
        element_id::CODEC_PRIVATE => "CodecPrivate",
        element_id::CODEC_NAME => "CodecName",
        element_id::CODEC_DELAY => "CodecDelay",
        element_id::SEEK_PRE_ROLL => "SeekPreRoll",
        element_id::VIDEO => "Video",
        element_id::PIXEL_WIDTH => "PixelWidth",
        element_id::PIXEL_HEIGHT => "PixelHeight",
        element_id::DISPLAY_WIDTH => "DisplayWidth",
        element_id::DISPLAY_HEIGHT => "DisplayHeight",
        element_id::FLAG_INTERLACED => "FlagInterlaced",
        element_id::COLOUR => "Colour",
        element_id::PRIMARIES => "Primaries",
        element_id::TRANSFER_CHARACTERISTICS => "TransferCharacteristics",
        element_id::MATRIX_COEFFICIENTS => "MatrixCoefficients",
        element_id::AUDIO => "Audio",
        element_id::SAMPLING_FREQUENCY => "SamplingFrequency",
        element_id::OUTPUT_SAMPLING_FREQUENCY => "OutputSamplingFrequency",
        element_id::CHANNELS => "Channels",
        element_id::BIT_DEPTH => "BitDepth",
        element_id::CLUSTER => "Cluster",
        element_id::TIMESTAMP => "Timestamp",
        element_id::SIMPLE_BLOCK => "SimpleBlock",
        element_id::BLOCK_GROUP => "BlockGroup",
        element_id::BLOCK => "Block",
        element_id::BLOCK_DURATION => "BlockDuration",
        element_id::REFERENCE_BLOCK => "ReferenceBlock",
        element_id::DISCARD_PADDING => "DiscardPadding",
        element_id::CUES => "Cues",
        element_id::CUE_POINT => "CuePoint",
        element_id::CUE_TIME => "CueTime",
        element_id::CUE_TRACK_POSITIONS => "CueTrackPositions",
        element_id::CUE_TRACK => "CueTrack",
        element_id::CUE_CLUSTER_POSITION => "CueClusterPosition",
        element_id::CUE_RELATIVE_POSITION => "CueRelativePosition",
        element_id::CUE_DURATION => "CueDuration",
        element_id::CUE_BLOCK_NUMBER => "CueBlockNumber",
        element_id::ATTACHMENTS => "Attachments",
        element_id::ATTACHED_FILE => "AttachedFile",
        element_id::FILE_NAME => "FileName",
        element_id::FILE_MIME_TYPE => "FileMimeType",
        element_id::FILE_DATA => "FileData",
        element_id::FILE_UID => "FileUID",
        element_id::CHAPTERS => "Chapters",
        element_id::EDITION_ENTRY => "EditionEntry",
        element_id::CHAPTER_ATOM => "ChapterAtom",
        element_id::CHAPTER_UID => "ChapterUID",
        element_id::CHAPTER_TIME_START => "ChapterTimeStart",
        element_id::CHAPTER_TIME_END => "ChapterTimeEnd",
        element_id::CHAPTER_DISPLAY => "ChapterDisplay",
        element_id::CHAP_STRING => "ChapString",
        element_id::CHAP_LANGUAGE => "ChapLanguage",
        element_id::TAGS => "Tags",
        element_id::TAG => "Tag",
        element_id::TARGETS => "Targets",
        element_id::SIMPLE_TAG => "SimpleTag",
        element_id::TAG_NAME => "TagName",
        element_id::TAG_STRING => "TagString",
        element_id::TAG_BINARY => "TagBinary",
        element_id::VOID => "Void",
        element_id::CRC32 => "CRC-32",
        _ => "Unknown",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_vint_1byte() {
        // 0x81 = 1xxxxxxx, value = 1
        let (remaining, value) = parse_vint(&[0x81]).expect("operation should succeed");
        assert_eq!(value, 1);
        assert!(remaining.is_empty());

        // 0xFF = 1xxxxxxx, value = 127
        let (_, value) = parse_vint(&[0xFF]).expect("operation should succeed");
        assert_eq!(value, 127);
    }

    #[test]
    fn test_parse_vint_2bytes() {
        // 0x40 0x01 = 01xxxxxx xxxxxxxx, value = 1
        let (_, value) = parse_vint(&[0x40, 0x01]).expect("operation should succeed");
        assert_eq!(value, 1);

        // 0x7F 0xFF = 01xxxxxx xxxxxxxx, value = 16383
        let (_, value) = parse_vint(&[0x7F, 0xFF]).expect("operation should succeed");
        assert_eq!(value, 16383);
    }

    #[test]
    fn test_parse_element_id() {
        // Single byte ID: 0xA3 (SimpleBlock)
        let (_, id) = parse_element_id(&[0xA3]).expect("operation should succeed");
        assert_eq!(id, 0xA3);

        // Four byte ID: EBML_HEADER
        let (_, id) =
            parse_element_id(&[0x1A, 0x45, 0xDF, 0xA3]).expect("operation should succeed");
        assert_eq!(id, EBML_HEADER);

        // Four byte ID: SEGMENT
        let (_, id) =
            parse_element_id(&[0x18, 0x53, 0x80, 0x67]).expect("operation should succeed");
        assert_eq!(id, SEGMENT);
    }

    #[test]
    fn test_parse_element_header() {
        // EBML header with size 31
        let data = [0x1A, 0x45, 0xDF, 0xA3, 0x9F];
        let (remaining, element) = parse_element_header(&data).expect("operation should succeed");

        assert_eq!(element.id, EBML_HEADER);
        assert_eq!(element.size, 31);
        assert_eq!(element.header_size, 5);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_parse_element_unknown_size() {
        // ID followed by 0xFF (unknown size in 1 byte)
        let data = [0xA3, 0xFF];
        let (_, element) = parse_element_header(&data).expect("operation should succeed");

        assert!(element.is_unbounded());
        assert_eq!(element.size, u64::MAX);
        assert!(element.total_size().is_none());
    }

    #[test]
    fn test_read_uint() {
        let (_, value) = read_uint(&[0x01]).expect("operation should succeed");
        assert_eq!(value, 1);

        let (_, value) = read_uint(&[0x01, 0x00]).expect("operation should succeed");
        assert_eq!(value, 256);

        let (_, value) = read_uint(&[0xFF, 0xFF]).expect("operation should succeed");
        assert_eq!(value, 65535);

        // Empty data returns 0
        let (_, value) = read_uint(&[]).expect("operation should succeed");
        assert_eq!(value, 0);
    }

    #[test]
    fn test_read_int() {
        // Positive
        let (_, value) = read_int(&[0x01]).expect("operation should succeed");
        assert_eq!(value, 1);

        // Negative (-1 in 1 byte)
        let (_, value) = read_int(&[0xFF]).expect("operation should succeed");
        assert_eq!(value, -1);

        // Negative (-128 in 1 byte)
        let (_, value) = read_int(&[0x80]).expect("operation should succeed");
        assert_eq!(value, -128);
    }

    #[test]
    fn test_element_total_size() {
        let bounded = EbmlElement {
            id: 0xA3,
            size: 100,
            header_size: 2,
        };
        assert_eq!(bounded.total_size(), Some(102));
        assert!(!bounded.is_unbounded());

        let unbounded = EbmlElement {
            id: SEGMENT,
            size: u64::MAX,
            header_size: 5,
        };
        assert!(unbounded.is_unbounded());
        assert!(unbounded.total_size().is_none());
    }

    #[test]
    fn test_read_string() {
        // Normal string
        let result = read_string(b"webm", 4).expect("operation should succeed");
        assert_eq!(result, "webm");

        // String with null terminator
        let result = read_string(b"webm\0\0\0\0", 8).expect("operation should succeed");
        assert_eq!(result, "webm");

        // Empty string
        let result = read_string(b"", 0).expect("operation should succeed");
        assert_eq!(result, "");

        // Matroska string
        let result = read_string(b"matroska", 8).expect("operation should succeed");
        assert_eq!(result, "matroska");
    }

    #[test]
    fn test_read_string_invalid_length() {
        let result = read_string(b"test", 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_float_f64() {
        let bytes = 48000.0_f64.to_be_bytes();
        let result = read_float(&bytes, 8).expect("operation should succeed");
        assert!((result - 48000.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_float_f32() {
        let bytes = 44100.0_f32.to_be_bytes();
        let result = read_float(&bytes, 4).expect("operation should succeed");
        assert!((result - 44100.0).abs() < 1.0);
    }

    #[test]
    fn test_read_float_zero() {
        let result = read_float(&[], 0).expect("operation should succeed");
        assert!((result - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_read_float_invalid_size() {
        let result = read_float(&[0; 5], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_read_date() {
        // Zero timestamp
        let result = read_date(&[0; 8], 8).expect("operation should succeed");
        assert_eq!(result, 0);

        // Empty date
        let result = read_date(&[], 0).expect("operation should succeed");
        assert_eq!(result, 0);
    }

    #[test]
    fn test_read_binary() {
        let data = [1, 2, 3, 4, 5];
        let result = read_binary(&data, 3).expect("operation should succeed");
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_read_binary_invalid_length() {
        let result = read_binary(&[1, 2], 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_element_name() {
        assert_eq!(element_name(element_id::EBML), "EBML");
        assert_eq!(element_name(element_id::SEGMENT), "Segment");
        assert_eq!(element_name(element_id::TRACKS), "Tracks");
        assert_eq!(element_name(element_id::CLUSTER), "Cluster");
        assert_eq!(element_name(0xDEADBEEF), "Unknown");
    }

    #[test]
    fn test_is_master() {
        let master = EbmlElement {
            id: element_id::SEGMENT,
            size: 1000,
            header_size: 8,
        };
        assert!(master.is_master());

        let not_master = EbmlElement {
            id: element_id::SIMPLE_BLOCK,
            size: 100,
            header_size: 4,
        };
        assert!(!not_master.is_master());
    }

    // Additional comprehensive EBML tests

    #[test]
    fn test_parse_vint_single_byte_values() {
        // Test all single-byte VINT values (0-127)
        for value in 0..128 {
            let byte = 0x80 | value; // 1xxxxxxx format
            let data = [byte];
            let (remaining, parsed) = parse_vint(&data).expect("operation should succeed");
            assert_eq!(parsed, u64::from(value));
            assert!(remaining.is_empty());
        }
    }

    #[test]
    fn test_parse_vint_two_byte_values() {
        // Test two-byte VINT: 0x4001 = 01xxxxxx xxxxxxxx, value = 1
        let (_, value) = parse_vint(&[0x40, 0x01]).expect("operation should succeed");
        assert_eq!(value, 1);

        // Test maximum two-byte value: 0x7FFF
        let (_, value) = parse_vint(&[0x7F, 0xFF]).expect("operation should succeed");
        assert_eq!(value, 16383); // 2^14 - 1
    }

    #[test]
    fn test_parse_vint_three_byte_values() {
        // Test three-byte VINT: 0x200001 = 001xxxxx xxxxxxxx xxxxxxxx
        let (_, value) = parse_vint(&[0x20, 0x00, 0x01]).expect("operation should succeed");
        assert_eq!(value, 1);

        // Test larger three-byte value
        let (_, value) = parse_vint(&[0x3F, 0xFF, 0xFF]).expect("operation should succeed");
        assert_eq!(value, 2_097_151); // 2^21 - 1
    }

    #[test]
    fn test_parse_vint_invalid_zero_byte() {
        let result = parse_vint(&[0x00, 0x01]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_vint_incomplete() {
        // Two-byte VINT with only one byte available
        let result = parse_vint(&[0x40]);
        assert!(result.is_err());

        // Three-byte VINT with only two bytes available
        let result = parse_vint(&[0x20, 0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_element_id_ebml_header() {
        // EBML Header: 0x1A 0x45 0xDF 0xA3
        let (remaining, id) =
            parse_element_id(&[0x1A, 0x45, 0xDF, 0xA3, 0xFF]).expect("operation should succeed");
        assert_eq!(id, element_id::EBML);
        assert_eq!(remaining, &[0xFF]);
    }

    #[test]
    fn test_parse_element_id_segment() {
        // Segment: 0x18 0x53 0x80 0x67
        let (_, id) =
            parse_element_id(&[0x18, 0x53, 0x80, 0x67]).expect("operation should succeed");
        assert_eq!(id, element_id::SEGMENT);
    }

    #[test]
    fn test_parse_element_id_single_byte() {
        // Single-byte IDs like TrackType: 0x83
        let (remaining, id) = parse_element_id(&[0x83, 0x00]).expect("operation should succeed");
        assert_eq!(id, element_id::TRACK_TYPE);
        assert_eq!(remaining, &[0x00]);
    }

    #[test]
    fn test_parse_element_size_unknown() {
        // Unknown size indicator: all 1s after length marker
        // Single byte: 0xFF = 1xxxxxxx with all x=1
        let (_, size) = parse_element_size(&[0xFF]).expect("operation should succeed");
        assert_eq!(size, u64::MAX);

        // Two bytes: 0x7FFF
        let (_, size) = parse_element_size(&[0x7F, 0xFF]).expect("operation should succeed");
        assert_eq!(size, u64::MAX);
    }

    #[test]
    fn test_parse_element_size_normal_values() {
        // Size = 0: 0x80
        let (_, size) = parse_element_size(&[0x80]).expect("operation should succeed");
        assert_eq!(size, 0);

        // Size = 127: 0xFF (but not unknown)
        let (_, size) = parse_element_size(&[0xFE]).expect("operation should succeed");
        assert_eq!(size, 126);

        // Two-byte size: 0x4000 = size 0
        let (_, size) = parse_element_size(&[0x40, 0x00]).expect("operation should succeed");
        assert_eq!(size, 0);
    }

    #[test]
    fn test_parse_element_header_complete() {
        // Parse a complete element header: Track Entry (0xAE) with size 100
        // ID: 0xAE (1 byte), Size: 0x64 + 0x80 = 0xE4 (1 byte)
        let data = [0xAE, 0xE4];
        let (remaining, element) = parse_element_header(&data).expect("operation should succeed");
        assert_eq!(element.id, element_id::TRACK_ENTRY);
        assert_eq!(element.size, 100);
        assert_eq!(element.header_size, 2);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_ebml_element_total_size() {
        let element = EbmlElement {
            id: element_id::SEGMENT,
            size: 1000,
            header_size: 8,
        };
        assert_eq!(element.total_size(), Some(1008));

        // Unknown size
        let element_unknown = EbmlElement {
            id: element_id::SEGMENT,
            size: u64::MAX,
            header_size: 8,
        };
        assert!(element_unknown.total_size().is_none());
    }

    #[test]
    fn test_read_uint_various_sizes() {
        // 1 byte - reads all bytes provided
        let (remaining, result) = read_uint(&[0x12]).expect("operation should succeed");
        assert_eq!(result, 0x12);
        assert!(remaining.is_empty());

        // 2 bytes
        let (remaining, result) = read_uint(&[0x12, 0x34]).expect("operation should succeed");
        assert_eq!(result, 0x1234);
        assert!(remaining.is_empty());

        // 4 bytes
        let (remaining, result) =
            read_uint(&[0x12, 0x34, 0x56, 0x78]).expect("operation should succeed");
        assert_eq!(result, 0x12345678);
        assert!(remaining.is_empty());

        // 8 bytes
        let (remaining, result) = read_uint(&[0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE, 0xF0])
            .expect("operation should succeed");
        assert_eq!(result, 0x123456789ABCDEF0);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_read_uint_empty() {
        // Empty data returns 0
        let (remaining, result) = read_uint(&[]).expect("operation should succeed");
        assert_eq!(result, 0);
        assert!(remaining.is_empty());
    }

    #[test]
    fn test_read_int_positive_and_negative() {
        // Positive: 0x12
        let (_, result) = read_int(&[0x12]).expect("operation should succeed");
        assert_eq!(result, 0x12);

        // Negative: 0xFF = -1 in two's complement
        let (_, result) = read_int(&[0xFF]).expect("operation should succeed");
        assert_eq!(result, -1);

        // Two-byte negative: 0xFF 0xFF = -1
        let (_, result) = read_int(&[0xFF, 0xFF]).expect("operation should succeed");
        assert_eq!(result, -1);

        // Two-byte positive: 0x7F 0xFF
        let (_, result) = read_int(&[0x7F, 0xFF]).expect("operation should succeed");
        assert_eq!(result, 32767);
    }

    #[test]
    fn test_read_string_utf8() {
        let result = read_string("hello".as_bytes(), 5).expect("operation should succeed");
        assert_eq!(result, "hello");

        // String with spaces
        let result = read_string("hello world".as_bytes(), 11).expect("operation should succeed");
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_read_string_null_padding() {
        // String with null padding
        let data = b"test\0\0\0\0";
        let result = read_string(data, 8).expect("operation should succeed");
        assert_eq!(result, "test");
    }

    #[test]
    fn test_element_name_coverage() {
        // Test various element names
        assert_eq!(element_name(element_id::EBML), "EBML");
        assert_eq!(element_name(element_id::TRACK_ENTRY), "TrackEntry");
        assert_eq!(element_name(element_id::VIDEO), "Video");
        assert_eq!(element_name(element_id::AUDIO), "Audio");
        assert_eq!(element_name(element_id::CLUSTER), "Cluster");
        assert_eq!(element_name(element_id::SIMPLE_BLOCK), "SimpleBlock");
        assert_eq!(element_name(element_id::CODEC_ID), "CodecID");
    }

    #[test]
    fn test_is_master_all_variants() {
        // Test various master elements
        let master_ids = [
            element_id::EBML,
            element_id::SEGMENT,
            element_id::TRACKS,
            element_id::TRACK_ENTRY,
            element_id::VIDEO,
            element_id::AUDIO,
            element_id::CLUSTER,
            element_id::CUES,
        ];

        for id in master_ids {
            let element = EbmlElement {
                id,
                size: 100,
                header_size: 4,
            };
            assert!(
                element.is_master(),
                "Expected {id:08X} to be a master element"
            );
        }
    }

    #[test]
    fn test_vint_consecutive_parsing() {
        // Multiple VINTs in sequence
        let data = [
            0x81, // VINT: value = 1
            0x82, // VINT: value = 2
            0x83, // VINT: value = 3
        ];

        let (rest, v1) = parse_vint(&data).expect("operation should succeed");
        assert_eq!(v1, 1);

        let (rest, v2) = parse_vint(rest).expect("operation should succeed");
        assert_eq!(v2, 2);

        let (rest, v3) = parse_vint(rest).expect("operation should succeed");
        assert_eq!(v3, 3);
        assert!(rest.is_empty());
    }
}
