//! Essence handling
//!
//! This module implements essence-related functionality:
//! - Essence descriptors (file, physical, digital image, sound)
//! - Essence locators (file, network, streaming)
//! - Essence access (embedded vs external)
//! - Codec identification
//! - Format metadata

use crate::dictionary::Auid;
use crate::structured_storage::StorageReader;
use crate::{EssenceData, Result};
use serde::{Deserialize, Serialize};
use std::io::{Read, Seek};
use std::path::PathBuf;
use uuid::Uuid;

/// Essence descriptor - describes essence format
#[derive(Debug, Clone)]
pub enum EssenceDescriptor {
    /// File descriptor (generic)
    File(FileDescriptor),
    /// Physical descriptor (tape, film)
    Physical(PhysicalDescriptor),
    /// Digital image descriptor (video)
    DigitalImage(DigitalImageDescriptor),
    /// Sound descriptor (audio)
    Sound(SoundDescriptor),
    /// Data essence descriptor
    Data(DataEssenceDescriptor),
    /// Multiple descriptors
    Multiple(MultipleDescriptor),
}

impl EssenceDescriptor {
    /// Get the sample rate
    #[must_use]
    pub fn sample_rate(&self) -> Option<(u32, u32)> {
        match self {
            EssenceDescriptor::DigitalImage(d) => Some(d.sample_rate),
            EssenceDescriptor::Sound(d) => Some(d.audio_sample_rate),
            _ => None,
        }
    }

    /// Get the essence container
    #[must_use]
    pub fn essence_container(&self) -> Option<Auid> {
        match self {
            EssenceDescriptor::File(d) => Some(d.essence_container),
            EssenceDescriptor::DigitalImage(d) => Some(d.file.essence_container),
            EssenceDescriptor::Sound(d) => Some(d.file.essence_container),
            _ => None,
        }
    }

    /// Get codec
    #[must_use]
    pub fn codec(&self) -> Option<Auid> {
        match self {
            EssenceDescriptor::File(d) => d.codec,
            EssenceDescriptor::DigitalImage(d) => d.file.codec,
            EssenceDescriptor::Sound(d) => d.file.codec,
            _ => None,
        }
    }

    /// Get locators
    #[must_use]
    pub fn locators(&self) -> Vec<&Locator> {
        match self {
            EssenceDescriptor::File(d) => &d.locators,
            EssenceDescriptor::DigitalImage(d) => &d.file.locators,
            EssenceDescriptor::Sound(d) => &d.file.locators,
            _ => return Vec::new(),
        }
        .iter()
        .collect()
    }
}

/// File descriptor - generic essence descriptor
#[derive(Debug, Clone)]
pub struct FileDescriptor {
    /// Linked track ID
    pub linked_track_id: Option<u32>,
    /// Sample rate (numerator, denominator)
    pub sample_rate: (u32, u32),
    /// Container length
    pub length: Option<i64>,
    /// Essence container label
    pub essence_container: Auid,
    /// Codec
    pub codec: Option<Auid>,
    /// Locators
    pub locators: Vec<Locator>,
}

impl FileDescriptor {
    /// Create a new file descriptor
    #[must_use]
    pub fn new(sample_rate: (u32, u32), essence_container: Auid) -> Self {
        Self {
            linked_track_id: None,
            sample_rate,
            length: None,
            essence_container,
            codec: None,
            locators: Vec::new(),
        }
    }

    /// Add a locator
    pub fn add_locator(&mut self, locator: Locator) {
        self.locators.push(locator);
    }
}

/// Physical descriptor - for tape/film sources
#[derive(Debug, Clone)]
pub struct PhysicalDescriptor {
    /// File descriptor
    pub file: FileDescriptor,
}

/// Digital image descriptor - for video essence
#[derive(Debug, Clone)]
pub struct DigitalImageDescriptor {
    /// File descriptor
    pub file: FileDescriptor,
    /// Sample rate (frame rate)
    pub sample_rate: (u32, u32),
    /// Stored dimensions
    pub stored_width: u32,
    pub stored_height: u32,
    /// Sampled dimensions
    pub sampled_width: Option<u32>,
    pub sampled_height: Option<u32>,
    /// Sampled offset
    pub sampled_x_offset: Option<u32>,
    pub sampled_y_offset: Option<u32>,
    /// Display dimensions
    pub display_width: Option<u32>,
    pub display_height: Option<u32>,
    /// Display offset
    pub display_x_offset: Option<u32>,
    pub display_y_offset: Option<u32>,
    /// Aspect ratio (horizontal:vertical)
    pub aspect_ratio: Option<(u32, u32)>,
    /// Frame layout
    pub frame_layout: FrameLayout,
    /// Video line map
    pub video_line_map: Option<Vec<u32>>,
    /// Alpha transparency
    pub alpha_transparency: Option<AlphaTransparency>,
    /// Component depth
    pub component_depth: Option<u32>,
    /// Horizontal subsampling
    pub horizontal_subsampling: Option<u32>,
    /// Vertical subsampling
    pub vertical_subsampling: Option<u32>,
    /// Color siting
    pub color_siting: Option<ColorSiting>,
    /// Image alignment factor
    pub image_alignment_factor: Option<u32>,
    /// Compression
    pub compression: Option<Auid>,
}

impl DigitalImageDescriptor {
    /// Create a new digital image descriptor
    #[must_use]
    pub fn new(
        sample_rate: (u32, u32),
        stored_width: u32,
        stored_height: u32,
        essence_container: Auid,
    ) -> Self {
        Self {
            file: FileDescriptor::new(sample_rate, essence_container),
            sample_rate,
            stored_width,
            stored_height,
            sampled_width: None,
            sampled_height: None,
            sampled_x_offset: None,
            sampled_y_offset: None,
            display_width: None,
            display_height: None,
            display_x_offset: None,
            display_y_offset: None,
            aspect_ratio: None,
            frame_layout: FrameLayout::FullFrame,
            video_line_map: None,
            alpha_transparency: None,
            component_depth: None,
            horizontal_subsampling: None,
            vertical_subsampling: None,
            color_siting: None,
            image_alignment_factor: None,
            compression: None,
        }
    }

    /// Set aspect ratio
    #[must_use]
    pub fn with_aspect_ratio(mut self, horizontal: u32, vertical: u32) -> Self {
        self.aspect_ratio = Some((horizontal, vertical));
        self
    }

    /// Set frame layout
    #[must_use]
    pub fn with_frame_layout(mut self, layout: FrameLayout) -> Self {
        self.frame_layout = layout;
        self
    }
}

/// Frame layout
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FrameLayout {
    /// Full frame (progressive)
    FullFrame,
    /// Separate fields (interlaced)
    SeparateFields,
    /// Single field
    SingleField,
    /// Mixed fields
    MixedFields,
}

/// Alpha transparency
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlphaTransparency {
    /// No alpha
    None,
    /// Minimum value is transparent
    MinValueTransparent,
    /// Maximum value is transparent
    MaxValueTransparent,
}

/// Color siting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ColorSiting {
    /// Cositing (4:2:2)
    CoSiting,
    /// Averaging (4:2:0)
    Averaging,
    /// Three tap (4:2:0 advanced)
    ThreeTap,
    /// Quincunx (diamond pattern)
    Quincunx,
    /// Rec601
    Rec601,
    /// Line alternating
    LineAlternating,
    /// Vertical midpoint
    VerticalMidpoint,
    /// Unknown
    Unknown,
}

/// Sound descriptor - for audio essence
#[derive(Debug, Clone)]
pub struct SoundDescriptor {
    /// File descriptor
    pub file: FileDescriptor,
    /// Audio sampling rate
    pub audio_sample_rate: (u32, u32),
    /// Locked to video rate
    pub locked: bool,
    /// Audio reference level
    pub audio_ref_level: Option<i8>,
    /// Electrospatial formulation
    pub electrospatial_formulation: Option<ElectrospatialFormulation>,
    /// Channel count
    pub channel_count: u32,
    /// Quantization bits
    pub quantization_bits: u32,
    /// Dial norm
    pub dial_norm: Option<i8>,
    /// Compression
    pub compression: Option<Auid>,
}

impl SoundDescriptor {
    /// Create a new sound descriptor
    #[must_use]
    pub fn new(
        audio_sample_rate: (u32, u32),
        channel_count: u32,
        quantization_bits: u32,
        essence_container: Auid,
    ) -> Self {
        Self {
            file: FileDescriptor::new(audio_sample_rate, essence_container),
            audio_sample_rate,
            locked: false,
            audio_ref_level: None,
            electrospatial_formulation: None,
            channel_count,
            quantization_bits,
            dial_norm: None,
            compression: None,
        }
    }

    /// Set locked to video
    #[must_use]
    pub fn with_locked(mut self, locked: bool) -> Self {
        self.locked = locked;
        self
    }

    /// Set audio reference level
    #[must_use]
    pub fn with_audio_ref_level(mut self, level: i8) -> Self {
        self.audio_ref_level = Some(level);
        self
    }
}

/// Electrospatial formulation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ElectrospatialFormulation {
    /// Default (mono/stereo)
    Default,
    /// Two channel mode
    TwoChannelMode,
    /// Single channel mode
    SingleChannelMode,
    /// Primary/Secondary mode
    PrimarySecondaryMode,
    /// Stereophonic mode
    StereophoniccMode,
    /// Single channel double sampling
    SingleChannelDoubleSampling,
    /// Stereo left channel double sampling
    StereoLeftChannelDoubleSampling,
    /// Stereo right channel double sampling
    StereoRightChannelDoubleSampling,
    /// Multi-channel mode
    MultiChannelMode,
}

/// Data essence descriptor
#[derive(Debug, Clone)]
pub struct DataEssenceDescriptor {
    /// File descriptor
    pub file: FileDescriptor,
    /// Data essence coding
    pub data_essence_coding: Option<Auid>,
}

/// Multiple descriptor - for multi-track essence
#[derive(Debug, Clone)]
pub struct MultipleDescriptor {
    /// File descriptor
    pub file: FileDescriptor,
    /// Sub-descriptors
    pub sub_descriptors: Vec<EssenceDescriptor>,
}

impl MultipleDescriptor {
    /// Create a new multiple descriptor
    #[must_use]
    pub fn new(essence_container: Auid) -> Self {
        Self {
            file: FileDescriptor::new((1, 1), essence_container),
            sub_descriptors: Vec::new(),
        }
    }

    /// Add a sub-descriptor
    pub fn add_sub_descriptor(&mut self, descriptor: EssenceDescriptor) {
        self.sub_descriptors.push(descriptor);
    }
}

/// Locator - points to essence data
#[derive(Debug, Clone)]
pub enum Locator {
    /// File locator (path to external file)
    File(FileLocator),
    /// Network locator (URL)
    Network(NetworkLocator),
    /// Text locator (generic text)
    Text(TextLocator),
}

/// File locator
#[derive(Debug, Clone)]
pub struct FileLocator {
    /// File path
    pub path: PathBuf,
}

impl FileLocator {
    /// Create a new file locator
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self { path: path.into() }
    }
}

/// Network locator
#[derive(Debug, Clone)]
pub struct NetworkLocator {
    /// URL
    pub url: String,
}

impl NetworkLocator {
    /// Create a new network locator
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }
}

/// Text locator
#[derive(Debug, Clone)]
pub struct TextLocator {
    /// Text
    pub text: String,
}

impl TextLocator {
    /// Create a new text locator
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into() }
    }
}

/// Essence reference - how essence is stored
#[derive(Debug, Clone)]
pub enum EssenceReference {
    /// Embedded in AAF file
    Embedded {
        /// Mob ID
        mob_id: Uuid,
        /// Slot ID
        slot_id: u32,
    },
    /// External file
    External {
        /// File path
        path: PathBuf,
        /// Mob ID (optional)
        mob_id: Option<Uuid>,
    },
    /// Network location
    Network {
        /// URL
        url: String,
        /// Mob ID (optional)
        mob_id: Option<Uuid>,
    },
}

impl EssenceReference {
    /// Check if essence is embedded
    #[must_use]
    pub fn is_embedded(&self) -> bool {
        matches!(self, EssenceReference::Embedded { .. })
    }

    /// Check if essence is external
    #[must_use]
    pub fn is_external(&self) -> bool {
        matches!(
            self,
            EssenceReference::External { .. } | EssenceReference::Network { .. }
        )
    }

    /// Get mob ID if available
    #[must_use]
    pub fn mob_id(&self) -> Option<Uuid> {
        match self {
            EssenceReference::Embedded { mob_id, .. } => Some(*mob_id),
            EssenceReference::External { mob_id, .. } => *mob_id,
            EssenceReference::Network { mob_id, .. } => *mob_id,
        }
    }
}

/// Essence access - for reading/writing essence
pub struct EssenceAccess {
    /// Descriptor
    descriptor: EssenceDescriptor,
    /// Reference
    reference: EssenceReference,
}

impl EssenceAccess {
    /// Create a new essence access
    #[must_use]
    pub fn new(descriptor: EssenceDescriptor, reference: EssenceReference) -> Self {
        Self {
            descriptor,
            reference,
        }
    }

    /// Get the descriptor
    #[must_use]
    pub fn descriptor(&self) -> &EssenceDescriptor {
        &self.descriptor
    }

    /// Get the reference
    #[must_use]
    pub fn reference(&self) -> &EssenceReference {
        &self.reference
    }

    /// Check if essence is embedded
    #[must_use]
    pub fn is_embedded(&self) -> bool {
        self.reference.is_embedded()
    }

    /// Check if essence is external
    #[must_use]
    pub fn is_external(&self) -> bool {
        self.reference.is_external()
    }
}

/// Read essence data from AAF file
pub fn read_essence_data<R: Read + Seek>(
    _storage: &mut StorageReader<R>,
) -> Result<Vec<EssenceData>> {
    // In a real implementation, we would read embedded essence data
    // from the AAF file's essence streams
    Ok(Vec::new())
}

/// Codec registry for mapping AUIDs to codec names
pub struct CodecRegistry {
    codecs: std::collections::HashMap<Auid, CodecInfo>,
}

impl CodecRegistry {
    /// Create a new codec registry
    #[must_use]
    pub fn new() -> Self {
        let mut registry = Self {
            codecs: std::collections::HashMap::new(),
        };
        registry.add_standard_codecs();
        registry
    }

    /// Add a codec
    pub fn add_codec(&mut self, auid: Auid, info: CodecInfo) {
        self.codecs.insert(auid, info);
    }

    /// Get codec info
    #[must_use]
    pub fn get_codec(&self, auid: &Auid) -> Option<&CodecInfo> {
        self.codecs.get(auid)
    }

    /// Add standard AAF codecs
    fn add_standard_codecs(&mut self) {
        // Add common codecs
        // These are placeholder AUIDs - real AAF uses SMPTE ULs
        let codecs = vec![
            (
                Auid::null(),
                CodecInfo::new("Uncompressed", CodecType::Video),
            ),
            // Would add more codecs here
        ];

        for (auid, info) in codecs {
            self.add_codec(auid, info);
        }
    }
}

impl Default for CodecRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Codec information
#[derive(Debug, Clone)]
pub struct CodecInfo {
    /// Codec name
    pub name: String,
    /// Codec type
    pub codec_type: CodecType,
    /// Description
    pub description: Option<String>,
}

impl CodecInfo {
    /// Create new codec info
    pub fn new(name: impl Into<String>, codec_type: CodecType) -> Self {
        Self {
            name: name.into(),
            codec_type,
            description: None,
        }
    }

    /// Set description
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }
}

/// Codec type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodecType {
    /// Video codec
    Video,
    /// Audio codec
    Audio,
    /// Data codec
    Data,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_file_descriptor() {
        let mut desc = FileDescriptor::new((25, 1), Auid::null());
        assert_eq!(desc.sample_rate, (25, 1));

        desc.add_locator(Locator::File(FileLocator::new("/path/to/file.mov")));
        assert_eq!(desc.locators.len(), 1);
    }

    #[test]
    fn test_digital_image_descriptor() {
        let desc = DigitalImageDescriptor::new((25, 1), 1920, 1080, Auid::null())
            .with_aspect_ratio(16, 9)
            .with_frame_layout(FrameLayout::FullFrame);

        assert_eq!(desc.stored_width, 1920);
        assert_eq!(desc.stored_height, 1080);
        assert_eq!(desc.aspect_ratio, Some((16, 9)));
        assert_eq!(desc.frame_layout, FrameLayout::FullFrame);
    }

    #[test]
    fn test_sound_descriptor() {
        let desc = SoundDescriptor::new((48000, 1), 2, 24, Auid::null())
            .with_locked(true)
            .with_audio_ref_level(-20);

        assert_eq!(desc.audio_sample_rate, (48000, 1));
        assert_eq!(desc.channel_count, 2);
        assert_eq!(desc.quantization_bits, 24);
        assert!(desc.locked);
        assert_eq!(desc.audio_ref_level, Some(-20));
    }

    #[test]
    fn test_multiple_descriptor() {
        let mut multi = MultipleDescriptor::new(Auid::null());

        let video = EssenceDescriptor::DigitalImage(DigitalImageDescriptor::new(
            (25, 1),
            1920,
            1080,
            Auid::null(),
        ));
        let audio = EssenceDescriptor::Sound(SoundDescriptor::new((48000, 1), 2, 24, Auid::null()));

        multi.add_sub_descriptor(video);
        multi.add_sub_descriptor(audio);

        assert_eq!(multi.sub_descriptors.len(), 2);
    }

    #[test]
    fn test_file_locator() {
        let locator = FileLocator::new("/media/video.mov");
        assert_eq!(
            locator.path.to_str().expect("to_str should succeed"),
            "/media/video.mov"
        );
    }

    #[test]
    fn test_network_locator() {
        let locator = NetworkLocator::new("http://server.com/video.mov");
        assert_eq!(locator.url, "http://server.com/video.mov");
    }

    #[test]
    fn test_essence_reference() {
        let embedded = EssenceReference::Embedded {
            mob_id: Uuid::new_v4(),
            slot_id: 1,
        };
        assert!(embedded.is_embedded());
        assert!(!embedded.is_external());

        let external = EssenceReference::External {
            path: PathBuf::from("/path/to/file.mov"),
            mob_id: None,
        };
        assert!(external.is_external());
        assert!(!external.is_embedded());
    }

    #[test]
    fn test_essence_access() {
        let desc = EssenceDescriptor::File(FileDescriptor::new((25, 1), Auid::null()));
        let reference = EssenceReference::External {
            path: PathBuf::from("/path/to/file.mov"),
            mob_id: None,
        };

        let access = EssenceAccess::new(desc, reference);
        assert!(access.is_external());
    }

    #[test]
    fn test_codec_registry() {
        let mut registry = CodecRegistry::new();
        let codec_info = CodecInfo::new("TestCodec", CodecType::Video);
        let auid = Auid::null();

        registry.add_codec(auid, codec_info);
        assert!(registry.get_codec(&auid).is_some());
    }
}
