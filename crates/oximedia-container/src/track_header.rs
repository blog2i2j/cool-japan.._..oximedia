#![allow(dead_code)]
//! ISO Base Media File Format track header (`tkhd`) abstraction.
//!
//! Models the immutable properties carried in the `tkhd` box: track ID,
//! creation/modification dates, duration, visual dimensions, volume, and
//! the transformation matrix.

/// Track-enable flags stored in the `tkhd` box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TrackFlags {
    /// Raw 24-bit flags value.
    bits: u32,
}

impl TrackFlags {
    /// Flag: track is enabled.
    pub const ENABLED: u32 = 0x0000_0001;
    /// Flag: track is used in the movie.
    pub const IN_MOVIE: u32 = 0x0000_0002;
    /// Flag: track is used in the movie's preview.
    pub const IN_PREVIEW: u32 = 0x0000_0004;
    /// Flag: track size is in the aspect ratio.
    pub const SIZE_IS_ASPECT_RATIO: u32 = 0x0000_0008;

    /// Creates flags from a raw 24-bit value.
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self {
            bits: bits & 0x00FF_FFFF,
        }
    }

    /// Creates the default enabled-and-in-movie flags.
    #[must_use]
    pub const fn default_flags() -> Self {
        Self {
            bits: Self::ENABLED | Self::IN_MOVIE,
        }
    }

    /// Returns the raw bits.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.bits
    }

    /// Returns `true` if the track is enabled.
    #[must_use]
    pub const fn is_enabled(self) -> bool {
        self.bits & Self::ENABLED != 0
    }

    /// Returns `true` if the track is in the movie presentation.
    #[must_use]
    pub const fn is_in_movie(self) -> bool {
        self.bits & Self::IN_MOVIE != 0
    }

    /// Returns `true` if the track is used in preview.
    #[must_use]
    pub const fn is_in_preview(self) -> bool {
        self.bits & Self::IN_PREVIEW != 0
    }

    /// Enables or disables the track.
    #[must_use]
    pub const fn set_enabled(mut self, enabled: bool) -> Self {
        if enabled {
            self.bits |= Self::ENABLED;
        } else {
            self.bits &= !Self::ENABLED;
        }
        self
    }
}

impl Default for TrackFlags {
    fn default() -> Self {
        Self::default_flags()
    }
}

/// A 3x3 affine transformation matrix stored as 9 fixed-point values.
///
/// Used by `tkhd` to describe rotation, scale, and translation.
/// Stored in row-major order as `[a, b, u, c, d, v, tx, ty, w]`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TransformMatrix {
    /// The 9 matrix entries in row-major order.
    pub values: [f64; 9],
}

impl TransformMatrix {
    /// Identity matrix (no transformation).
    pub const IDENTITY: Self = Self {
        values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    };

    /// Creates a matrix from raw values.
    #[must_use]
    pub const fn new(values: [f64; 9]) -> Self {
        Self { values }
    }

    /// Returns `true` if this is the identity matrix (within epsilon).
    #[must_use]
    pub fn is_identity(&self) -> bool {
        let id = Self::IDENTITY.values;
        self.values
            .iter()
            .zip(id.iter())
            .all(|(a, b)| (a - b).abs() < 1e-9)
    }

    /// Creates a rotation matrix for the given angle in degrees.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rotation(degrees: f64) -> Self {
        let rad = degrees * std::f64::consts::PI / 180.0;
        let cos = rad.cos();
        let sin = rad.sin();
        Self {
            values: [cos, sin, 0.0, -sin, cos, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Creates a scale matrix.
    #[must_use]
    pub fn scale(sx: f64, sy: f64) -> Self {
        Self {
            values: [sx, 0.0, 0.0, 0.0, sy, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Creates a translation matrix.
    #[must_use]
    pub fn translation(tx: f64, ty: f64) -> Self {
        Self {
            values: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, tx, ty, 1.0],
        }
    }

    /// Extracts the rotation angle in degrees from the matrix.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn rotation_degrees(&self) -> f64 {
        self.values[1].atan2(self.values[0]) * 180.0 / std::f64::consts::PI
    }

    /// Extracts translation components `(tx, ty)`.
    #[must_use]
    pub fn translation_xy(&self) -> (f64, f64) {
        (self.values[6], self.values[7])
    }
}

impl Default for TransformMatrix {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// Track type classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackType {
    /// Video track.
    Video,
    /// Audio track.
    Audio,
    /// Subtitle / text track.
    Subtitle,
    /// Hint track (RTP streaming hints).
    Hint,
    /// Metadata track.
    Metadata,
    /// Unknown or unsupported type.
    Unknown,
}

impl TrackType {
    /// Returns `true` for media tracks (video/audio).
    #[must_use]
    pub fn is_media(self) -> bool {
        matches!(self, Self::Video | Self::Audio)
    }
}

/// The track header (`tkhd`) box contents.
#[derive(Debug, Clone)]
pub struct TrackHeader {
    /// Version (0 or 1).
    pub version: u8,
    /// Track flags.
    pub flags: TrackFlags,
    /// Track ID (unique within the movie).
    pub track_id: u32,
    /// Track duration in movie timescale units.
    pub duration: u64,
    /// Visual layer ordering (lower = closer to viewer).
    pub layer: i16,
    /// Alternate group identifier (0 = not in a group).
    pub alternate_group: u16,
    /// Audio volume (1.0 = full, 0.0 = mute). Only meaningful for audio.
    pub volume: f32,
    /// Visual width in pixels (fixed-point 16.16 in the file).
    pub width: f64,
    /// Visual height in pixels (fixed-point 16.16 in the file).
    pub height: f64,
    /// Transformation matrix.
    pub matrix: TransformMatrix,
    /// Derived track type.
    pub track_type: TrackType,
}

impl TrackHeader {
    /// Creates a new video track header with sensible defaults.
    #[must_use]
    pub fn video(track_id: u32, width: f64, height: f64, duration: u64) -> Self {
        Self {
            version: 0,
            flags: TrackFlags::default(),
            track_id,
            duration,
            layer: 0,
            alternate_group: 0,
            volume: 0.0,
            width,
            height,
            matrix: TransformMatrix::IDENTITY,
            track_type: TrackType::Video,
        }
    }

    /// Creates a new audio track header with sensible defaults.
    #[must_use]
    pub fn audio(track_id: u32, duration: u64) -> Self {
        Self {
            version: 0,
            flags: TrackFlags::default(),
            track_id,
            duration,
            layer: 0,
            alternate_group: 0,
            volume: 1.0,
            width: 0.0,
            height: 0.0,
            matrix: TransformMatrix::IDENTITY,
            track_type: TrackType::Audio,
        }
    }

    /// Returns the aspect ratio (width / height). Returns `None` for
    /// non-video tracks or zero dimensions.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect_ratio(&self) -> Option<f64> {
        if self.height.abs() < f64::EPSILON {
            return None;
        }
        Some(self.width / self.height)
    }

    /// Returns `true` if this is a video track.
    #[must_use]
    pub fn is_video(&self) -> bool {
        self.track_type == TrackType::Video
    }

    /// Returns `true` if this is an audio track.
    #[must_use]
    pub fn is_audio(&self) -> bool {
        self.track_type == TrackType::Audio
    }

    /// Returns `true` if the track is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.flags.is_enabled()
    }

    /// Returns `(width, height)` as integers.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn dimensions(&self) -> (u32, u32) {
        (self.width as u32, self.height as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_flags_default() {
        let f = TrackFlags::default();
        assert!(f.is_enabled());
        assert!(f.is_in_movie());
        assert!(!f.is_in_preview());
    }

    #[test]
    fn test_track_flags_from_bits() {
        let f = TrackFlags::from_bits(0x07);
        assert!(f.is_enabled());
        assert!(f.is_in_movie());
        assert!(f.is_in_preview());
    }

    #[test]
    fn test_track_flags_set_enabled() {
        let f = TrackFlags::default().set_enabled(false);
        assert!(!f.is_enabled());
        assert!(f.is_in_movie());

        let f2 = f.set_enabled(true);
        assert!(f2.is_enabled());
    }

    #[test]
    fn test_transform_identity() {
        let m = TransformMatrix::IDENTITY;
        assert!(m.is_identity());
    }

    #[test]
    fn test_transform_rotation() {
        let m = TransformMatrix::rotation(90.0);
        assert!(!m.is_identity());
        assert!((m.rotation_degrees() - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_transform_scale() {
        let m = TransformMatrix::scale(2.0, 3.0);
        assert!((m.values[0] - 2.0).abs() < f64::EPSILON);
        assert!((m.values[4] - 3.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_transform_translation() {
        let m = TransformMatrix::translation(10.0, 20.0);
        let (tx, ty) = m.translation_xy();
        assert!((tx - 10.0).abs() < f64::EPSILON);
        assert!((ty - 20.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_track_type_is_media() {
        assert!(TrackType::Video.is_media());
        assert!(TrackType::Audio.is_media());
        assert!(!TrackType::Subtitle.is_media());
        assert!(!TrackType::Hint.is_media());
    }

    #[test]
    fn test_video_header() {
        let h = TrackHeader::video(1, 1920.0, 1080.0, 90000);
        assert!(h.is_video());
        assert!(!h.is_audio());
        assert!(h.is_enabled());
        assert_eq!(h.dimensions(), (1920, 1080));
    }

    #[test]
    fn test_audio_header() {
        let h = TrackHeader::audio(2, 48000);
        assert!(h.is_audio());
        assert!(!h.is_video());
        assert!((h.volume - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_aspect_ratio() {
        let h = TrackHeader::video(1, 1920.0, 1080.0, 0);
        let ratio = h.aspect_ratio().expect("operation should succeed");
        assert!((ratio - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_aspect_ratio_zero_height() {
        let h = TrackHeader::video(1, 1920.0, 0.0, 0);
        assert!(h.aspect_ratio().is_none());
    }

    #[test]
    fn test_flags_bits_masking() {
        // Only lower 24 bits should survive.
        let f = TrackFlags::from_bits(0xFF000003);
        assert_eq!(f.bits(), 0x000003);
    }
}
