//! Essence descriptor types for IMF track files.
//!
//! Models the `EssenceDescriptor` element hierarchy from SMPTE ST 377-1
//! (MXF Descriptive Metadata) and ST 2067-5 (IMF Essence Component) in a
//! pure-Rust, allocation-light form.

#![allow(dead_code)]

/// High-level category of an essence track.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum EssenceType {
    /// Picture essence (video frames).
    Picture,
    /// Sound essence (audio samples).
    Sound,
    /// Data essence (subtitles, captions, ancillary data …).
    Data,
    /// Timecode track.
    Timecode,
    /// Descriptive metadata.
    DescriptiveMetadata,
}

impl EssenceType {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Picture => "Picture",
            Self::Sound => "Sound",
            Self::Data => "Data",
            Self::Timecode => "Timecode",
            Self::DescriptiveMetadata => "Descriptive Metadata",
        }
    }

    /// Returns `true` for essence types that carry media content (i.e. not
    /// timecode or descriptive metadata).
    #[must_use]
    pub fn is_media(&self) -> bool {
        matches!(self, Self::Picture | Self::Sound | Self::Data)
    }
}

/// Picture-specific essence parameters.
#[derive(Debug, Clone)]
pub struct PictureParams {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Aspect ratio as `(width_part, height_part)` e.g. `(16, 9)`.
    pub aspect_ratio: (u32, u32),
    /// Frame rate as `(numerator, denominator)`.
    pub frame_rate: (u32, u32),
    /// Whether the picture is progressive (`true`) or interlaced (`false`).
    pub progressive: bool,
    /// Colour primaries label (e.g. `"BT.709"`, `"P3D65"`, `"BT.2020"`).
    pub color_primaries: String,
    /// Bit depth of the picture samples.
    pub bit_depth: u8,
}

impl Default for PictureParams {
    fn default() -> Self {
        Self {
            width: 1920,
            height: 1080,
            aspect_ratio: (16, 9),
            frame_rate: (24, 1),
            progressive: true,
            color_primaries: "BT.709".to_string(),
            bit_depth: 12,
        }
    }
}

/// Sound-specific essence parameters.
#[derive(Debug, Clone)]
pub struct SoundParams {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channel_count: u32,
    /// Bit depth per sample.
    pub bit_depth: u8,
    /// Audio layout label (e.g. `"Stereo"`, `"5.1"`, `"7.1"`).
    pub audio_layout: String,
}

impl Default for SoundParams {
    fn default() -> Self {
        Self {
            sample_rate: 48000,
            channel_count: 2,
            bit_depth: 24,
            audio_layout: "Stereo".to_string(),
        }
    }
}

/// An essence descriptor combining type-specific parameters and common fields.
#[derive(Debug, Clone)]
pub struct EssenceDescriptor {
    /// UUID of the track file this descriptor belongs to.
    pub track_file_id: String,
    /// Type of essence.
    pub essence_type: EssenceType,
    /// Codec label or SMPTE UL string (e.g. `"JPEG 2000"`, `"PCM"`).
    pub codec: String,
    /// Edit rate of the essence as `(numerator, denominator)`.
    pub edit_rate: (u32, u32),
    /// Duration in edit units.
    pub duration: u64,
    /// Picture-specific parameters (present only for picture essence).
    pub picture: Option<PictureParams>,
    /// Sound-specific parameters (present only for sound essence).
    pub sound: Option<SoundParams>,
    /// Container format label (e.g. `"MXF OP-1a"`).
    pub container_format: String,
}

impl EssenceDescriptor {
    /// Create a picture [`EssenceDescriptor`] with the given parameters.
    #[must_use]
    pub fn picture(
        track_file_id: impl Into<String>,
        params: PictureParams,
        edit_rate: (u32, u32),
        duration: u64,
    ) -> Self {
        Self {
            track_file_id: track_file_id.into(),
            essence_type: EssenceType::Picture,
            codec: "JPEG 2000".to_string(),
            edit_rate,
            duration,
            picture: Some(params),
            sound: None,
            container_format: "MXF OP-1a".to_string(),
        }
    }

    /// Create a sound [`EssenceDescriptor`] with the given parameters.
    #[must_use]
    pub fn sound(
        track_file_id: impl Into<String>,
        params: SoundParams,
        edit_rate: (u32, u32),
        duration: u64,
    ) -> Self {
        Self {
            track_file_id: track_file_id.into(),
            essence_type: EssenceType::Sound,
            codec: "PCM".to_string(),
            edit_rate,
            duration,
            picture: None,
            sound: Some(params),
            container_format: "MXF OP-1a".to_string(),
        }
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        let (num, den) = self.edit_rate;
        if num == 0 {
            return 0.0;
        }
        self.duration as f64 * den as f64 / num as f64
    }

    /// Returns `true` when this descriptor represents picture essence.
    #[must_use]
    pub fn is_picture(&self) -> bool {
        self.essence_type == EssenceType::Picture
    }

    /// Returns `true` when this descriptor represents sound essence.
    #[must_use]
    pub fn is_sound(&self) -> bool {
        self.essence_type == EssenceType::Sound
    }
}

/// A registry of [`EssenceDescriptor`]s indexed by track-file UUID.
#[derive(Debug, Clone, Default)]
pub struct EssenceRegistry {
    entries: Vec<EssenceDescriptor>,
}

impl EssenceRegistry {
    /// Create an empty [`EssenceRegistry`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new descriptor.
    pub fn register(&mut self, descriptor: EssenceDescriptor) {
        self.entries.push(descriptor);
    }

    /// Look up an essence descriptor by track-file UUID.
    #[must_use]
    pub fn get(&self, track_file_id: &str) -> Option<&EssenceDescriptor> {
        self.entries
            .iter()
            .find(|d| d.track_file_id == track_file_id)
    }

    /// Return all descriptors of a given [`EssenceType`].
    #[must_use]
    pub fn by_type(&self, essence_type: EssenceType) -> Vec<&EssenceDescriptor> {
        self.entries
            .iter()
            .filter(|d| d.essence_type == essence_type)
            .collect()
    }

    /// Total number of registered descriptors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when no descriptors have been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all registered descriptors.
    pub fn iter(&self) -> impl Iterator<Item = &EssenceDescriptor> {
        self.entries.iter()
    }
}

// ── unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn picture_desc() -> EssenceDescriptor {
        EssenceDescriptor::picture("urn:uuid:pic-001", PictureParams::default(), (24, 1), 2400)
    }

    fn sound_desc() -> EssenceDescriptor {
        EssenceDescriptor::sound("urn:uuid:snd-001", SoundParams::default(), (24, 1), 2400)
    }

    // ── EssenceType ───────────────────────────────────────────────────────

    #[test]
    fn test_essence_type_labels() {
        assert_eq!(EssenceType::Picture.label(), "Picture");
        assert_eq!(EssenceType::Sound.label(), "Sound");
        assert_eq!(EssenceType::Timecode.label(), "Timecode");
        assert_eq!(
            EssenceType::DescriptiveMetadata.label(),
            "Descriptive Metadata"
        );
    }

    #[test]
    fn test_essence_type_is_media() {
        assert!(EssenceType::Picture.is_media());
        assert!(EssenceType::Sound.is_media());
        assert!(EssenceType::Data.is_media());
        assert!(!EssenceType::Timecode.is_media());
        assert!(!EssenceType::DescriptiveMetadata.is_media());
    }

    // ── PictureParams ─────────────────────────────────────────────────────

    #[test]
    fn test_picture_params_default() {
        let p = PictureParams::default();
        assert_eq!(p.width, 1920);
        assert_eq!(p.height, 1080);
        assert!(p.progressive);
        assert_eq!(p.bit_depth, 12);
    }

    // ── SoundParams ───────────────────────────────────────────────────────

    #[test]
    fn test_sound_params_default() {
        let s = SoundParams::default();
        assert_eq!(s.sample_rate, 48000);
        assert_eq!(s.channel_count, 2);
        assert_eq!(s.bit_depth, 24);
    }

    // ── EssenceDescriptor ─────────────────────────────────────────────────

    #[test]
    fn test_picture_descriptor_is_picture() {
        let d = picture_desc();
        assert!(d.is_picture());
        assert!(!d.is_sound());
        assert!(d.picture.is_some());
        assert!(d.sound.is_none());
    }

    #[test]
    fn test_sound_descriptor_is_sound() {
        let d = sound_desc();
        assert!(d.is_sound());
        assert!(!d.is_picture());
        assert!(d.sound.is_some());
        assert!(d.picture.is_none());
    }

    #[test]
    fn test_duration_secs() {
        let d = picture_desc(); // 2400 frames @ 24 fps
        assert!((d.duration_secs() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_duration_secs_zero_numerator() {
        let mut d = picture_desc();
        d.edit_rate = (0, 1);
        assert_eq!(d.duration_secs(), 0.0);
    }

    // ── EssenceRegistry ───────────────────────────────────────────────────

    #[test]
    fn test_registry_empty() {
        let reg = EssenceRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = EssenceRegistry::new();
        reg.register(picture_desc());
        let found = reg.get("urn:uuid:pic-001");
        assert!(found.is_some());
        assert!(found.expect("test expectation failed").is_picture());
    }

    #[test]
    fn test_registry_get_missing() {
        let reg = EssenceRegistry::new();
        assert!(reg.get("urn:uuid:nonexistent").is_none());
    }

    #[test]
    fn test_registry_by_type() {
        let mut reg = EssenceRegistry::new();
        reg.register(picture_desc());
        reg.register(sound_desc());
        let pics = reg.by_type(EssenceType::Picture);
        assert_eq!(pics.len(), 1);
        let sounds = reg.by_type(EssenceType::Sound);
        assert_eq!(sounds.len(), 1);
    }

    #[test]
    fn test_registry_len() {
        let mut reg = EssenceRegistry::new();
        reg.register(picture_desc());
        reg.register(sound_desc());
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn test_registry_iter() {
        let mut reg = EssenceRegistry::new();
        reg.register(picture_desc());
        reg.register(sound_desc());
        let ids: Vec<&str> = reg.iter().map(|d| d.track_file_id.as_str()).collect();
        assert!(ids.contains(&"urn:uuid:pic-001"));
        assert!(ids.contains(&"urn:uuid:snd-001"));
    }
}
