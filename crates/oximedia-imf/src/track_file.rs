//! IMF Virtual Track File resolution and metadata.
//!
//! Implements the virtual track file concept from SMPTE ST 2067-2:
//! a virtual track file is a named MXF essence file referenced by a
//! resource inside a Composition Playlist sequence.

use std::time::Duration;
use uuid::Uuid;

/// Essence type carried by a track file
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EssenceKind {
    /// Picture (video) essence
    Picture,
    /// Sound (audio) essence
    Sound,
    /// Timed text / subtitle
    TimedText,
    /// Data / metadata
    Data,
}

impl EssenceKind {
    /// Return the typical MIME type for this essence kind.
    #[allow(dead_code)]
    pub fn mime_type(self) -> &'static str {
        match self {
            EssenceKind::Picture | EssenceKind::Sound | EssenceKind::Data => "application/mxf",
            EssenceKind::TimedText => "application/ttml+xml",
        }
    }
}

/// Edit rate represented as a rational number (numerator / denominator).
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EditRate {
    /// Edit rate numerator (frames per second numerator).
    pub numerator: u32,
    /// Edit rate denominator (frames per second denominator).
    pub denominator: u32,
}

impl EditRate {
    /// Create a new edit rate.
    #[allow(dead_code)]
    pub fn new(numerator: u32, denominator: u32) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// Compute the frame duration.
    #[allow(dead_code)]
    #[allow(clippy::cast_precision_loss)]
    pub fn frame_duration(&self) -> Duration {
        if self.numerator == 0 {
            return Duration::ZERO;
        }
        let secs = self.denominator as f64 / self.numerator as f64;
        Duration::from_secs_f64(secs)
    }

    /// Frames per second as a floating-point value.
    #[allow(dead_code)]
    #[allow(clippy::cast_precision_loss)]
    pub fn fps(&self) -> f64 {
        if self.denominator == 0 {
            return 0.0;
        }
        self.numerator as f64 / self.denominator as f64
    }
}

impl std::fmt::Display for EditRate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

/// Metadata describing a virtual track file.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TrackFileMetadata {
    /// UUID of the track file as referenced in the CPL
    pub track_file_id: Uuid,
    /// Kind of essence stored
    pub essence_kind: EssenceKind,
    /// Edit rate of the contained essence
    pub edit_rate: EditRate,
    /// Duration in edit units (frames)
    pub duration: u64,
    /// SHA-1 hash of the file (hex string), for integrity checks
    pub hash: Option<String>,
    /// File size in bytes
    pub size_bytes: u64,
    /// Path to the MXF file on disk
    pub file_path: String,
}

impl TrackFileMetadata {
    /// Create a new metadata record.
    #[allow(dead_code)]
    pub fn new(
        track_file_id: Uuid,
        essence_kind: EssenceKind,
        edit_rate: EditRate,
        duration: u64,
        file_path: impl Into<String>,
        size_bytes: u64,
    ) -> Self {
        Self {
            track_file_id,
            essence_kind,
            edit_rate,
            duration,
            hash: None,
            size_bytes,
            file_path: file_path.into(),
        }
    }

    /// Attach a SHA-1 hash to this record.
    #[allow(dead_code)]
    pub fn with_hash(mut self, hash: impl Into<String>) -> Self {
        self.hash = Some(hash.into());
        self
    }

    /// Wall-clock duration based on `duration` and `edit_rate`.
    #[allow(dead_code)]
    pub fn wall_clock_duration(&self) -> Duration {
        self.edit_rate.frame_duration() * self.duration as u32
    }
}

/// A registry of track file metadata for an IMF package
#[allow(dead_code)]
#[derive(Debug, Default)]
pub struct TrackFileRegistry {
    entries: std::collections::HashMap<Uuid, TrackFileMetadata>,
}

impl TrackFileRegistry {
    /// Create an empty registry.
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a track file.
    #[allow(dead_code)]
    pub fn register(&mut self, meta: TrackFileMetadata) {
        self.entries.insert(meta.track_file_id, meta);
    }

    /// Look up a track file by UUID.
    #[allow(dead_code)]
    pub fn get(&self, id: &Uuid) -> Option<&TrackFileMetadata> {
        self.entries.get(id)
    }

    /// Return all picture (video) track files.
    #[allow(dead_code)]
    pub fn picture_tracks(&self) -> Vec<&TrackFileMetadata> {
        self.entries
            .values()
            .filter(|m| m.essence_kind == EssenceKind::Picture)
            .collect()
    }

    /// Return all sound (audio) track files.
    #[allow(dead_code)]
    pub fn sound_tracks(&self) -> Vec<&TrackFileMetadata> {
        self.entries
            .values()
            .filter(|m| m.essence_kind == EssenceKind::Sound)
            .collect()
    }

    /// Total number of registered track files.
    #[allow(dead_code)]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if no track files are registered.
    #[allow(dead_code)]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_meta(kind: EssenceKind, rate: EditRate, duration: u64) -> TrackFileMetadata {
        let track = std::env::temp_dir()
            .join("oximedia-imf-trackfile-track.mxf")
            .to_string_lossy()
            .into_owned();
        TrackFileMetadata::new(Uuid::new_v4(), kind, rate, duration, track, 1024)
    }

    #[test]
    fn test_essence_kind_mime_type_picture() {
        assert_eq!(EssenceKind::Picture.mime_type(), "application/mxf");
    }

    #[test]
    fn test_essence_kind_mime_type_timed_text() {
        assert_eq!(EssenceKind::TimedText.mime_type(), "application/ttml+xml");
    }

    #[test]
    fn test_edit_rate_display() {
        let r = EditRate::new(24, 1);
        assert_eq!(r.to_string(), "24/1");
    }

    #[test]
    fn test_edit_rate_fps() {
        let r = EditRate::new(24000, 1001);
        let fps = r.fps();
        assert!((fps - 23.976).abs() < 0.001);
    }

    #[test]
    fn test_edit_rate_frame_duration_24fps() {
        let r = EditRate::new(24, 1);
        let dur = r.frame_duration();
        // 1/24 s ≈ 41.667 ms
        assert!((dur.as_secs_f64() - 1.0 / 24.0).abs() < 1e-9);
    }

    #[test]
    fn test_edit_rate_fps_zero_denominator() {
        let r = EditRate::new(24, 0);
        assert_eq!(r.fps(), 0.0);
    }

    #[test]
    fn test_edit_rate_frame_duration_zero_numerator() {
        let r = EditRate::new(0, 1);
        assert_eq!(r.frame_duration(), Duration::ZERO);
    }

    #[test]
    fn test_track_file_metadata_new() {
        let id = Uuid::new_v4();
        let m = TrackFileMetadata::new(
            id,
            EssenceKind::Picture,
            EditRate::new(24, 1),
            100,
            "/x.mxf",
            512,
        );
        assert_eq!(m.track_file_id, id);
        assert_eq!(m.duration, 100);
        assert!(m.hash.is_none());
    }

    #[test]
    fn test_with_hash() {
        let m = make_meta(EssenceKind::Sound, EditRate::new(48000, 1), 48000).with_hash("deadbeef");
        assert_eq!(m.hash.as_deref(), Some("deadbeef"));
    }

    #[test]
    fn test_wall_clock_duration() {
        let m = make_meta(EssenceKind::Picture, EditRate::new(24, 1), 240);
        // 240 frames at 24fps = 10 seconds
        let dur = m.wall_clock_duration();
        assert!((dur.as_secs_f64() - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_registry_empty_on_creation() {
        let reg = TrackFileRegistry::new();
        assert!(reg.is_empty());
    }

    #[test]
    fn test_registry_register_and_get() {
        let mut reg = TrackFileRegistry::new();
        let m = make_meta(EssenceKind::Picture, EditRate::new(25, 1), 50);
        let id = m.track_file_id;
        reg.register(m);
        assert!(reg.get(&id).is_some());
    }

    #[test]
    fn test_registry_picture_and_sound_filter() {
        let mut reg = TrackFileRegistry::new();
        reg.register(make_meta(EssenceKind::Picture, EditRate::new(24, 1), 100));
        reg.register(make_meta(
            EssenceKind::Sound,
            EditRate::new(48000, 1),
            48000,
        ));
        reg.register(make_meta(
            EssenceKind::Sound,
            EditRate::new(48000, 1),
            48000,
        ));
        assert_eq!(reg.picture_tracks().len(), 1);
        assert_eq!(reg.sound_tracks().len(), 2);
    }

    #[test]
    fn test_registry_len() {
        let mut reg = TrackFileRegistry::new();
        reg.register(make_meta(EssenceKind::Data, EditRate::new(1, 1), 1));
        assert_eq!(reg.len(), 1);
    }
}
