//! AAF media file reference model.
//!
//! Provides `MediaFileKind`, `MediaFileRef`, and `MediaFileRegistry`
//! for tracking external or embedded media file references within an AAF.

#![allow(dead_code)]

use std::collections::HashMap;
use std::path::PathBuf;

/// The broad category of a referenced media file.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MediaFileKind {
    /// A video essence file (MXF, MOV, etc.).
    Video,
    /// An audio-only essence file (WAV, AIFF, etc.).
    Audio,
    /// A file containing both video and audio.
    AudioVideo,
    /// A still-image file (DPX, TIFF, etc.).
    StillImage,
    /// An auxiliary or sidecar file.
    Auxiliary,
}

impl MediaFileKind {
    /// Returns `true` if the kind contains a video track.
    #[must_use]
    pub fn is_video(self) -> bool {
        matches!(self, Self::Video | Self::AudioVideo)
    }

    /// Returns `true` if the kind contains an audio track.
    #[must_use]
    pub fn is_audio(self) -> bool {
        matches!(self, Self::Audio | Self::AudioVideo)
    }

    /// Human-readable label for this kind.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Video => "Video",
            Self::Audio => "Audio",
            Self::AudioVideo => "AudioVideo",
            Self::StillImage => "StillImage",
            Self::Auxiliary => "Auxiliary",
        }
    }
}

/// A reference to a single media file used within an AAF structure.
#[derive(Debug, Clone)]
pub struct MediaFileRef {
    /// Unique identifier within the registry.
    id: u32,
    /// File name (without directory component).
    name: String,
    /// Optional full path on disk.
    path: Option<PathBuf>,
    /// Kind of the referenced media.
    kind: MediaFileKind,
    /// Optional size in bytes.
    size_bytes: Option<u64>,
}

impl MediaFileRef {
    /// Create a new `MediaFileRef`.
    #[must_use]
    pub fn new(id: u32, name: impl Into<String>, kind: MediaFileKind) -> Self {
        Self {
            id,
            name: name.into(),
            path: None,
            kind,
            size_bytes: None,
        }
    }

    /// Set the full file path.
    pub fn set_path(&mut self, path: PathBuf) {
        self.path = Some(path);
    }

    /// Set the file size in bytes.
    pub fn set_size(&mut self, bytes: u64) {
        self.size_bytes = Some(bytes);
    }

    /// Returns `true` if a path has been associated with this reference.
    #[must_use]
    pub fn has_path(&self) -> bool {
        self.path.is_some()
    }

    /// Return the file name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Return the optional path.
    #[must_use]
    pub fn path(&self) -> Option<&PathBuf> {
        self.path.as_ref()
    }

    /// Return the file kind.
    #[must_use]
    pub fn kind(&self) -> MediaFileKind {
        self.kind
    }

    /// Return the unique id.
    #[must_use]
    pub fn id(&self) -> u32 {
        self.id
    }

    /// Return the optional size.
    #[must_use]
    pub fn size_bytes(&self) -> Option<u64> {
        self.size_bytes
    }
}

/// A registry of all `MediaFileRef` entries known to an AAF document.
#[derive(Debug, Clone, Default)]
pub struct MediaFileRegistry {
    entries: HashMap<u32, MediaFileRef>,
    next_id: u32,
}

impl MediaFileRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a `MediaFileRef`, assigning a fresh id automatically.
    pub fn add(&mut self, name: impl Into<String>, kind: MediaFileKind) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.entries.insert(id, MediaFileRef::new(id, name, kind));
        id
    }

    /// Find a ref by exact file name (returns the first match).
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<&MediaFileRef> {
        self.entries.values().find(|r| r.name() == name)
    }

    /// Find a ref by id.
    #[must_use]
    pub fn find_by_id(&self, id: u32) -> Option<&MediaFileRef> {
        self.entries.get(&id)
    }

    /// Return a mutable reference by id.
    pub fn find_by_id_mut(&mut self, id: u32) -> Option<&mut MediaFileRef> {
        self.entries.get_mut(&id)
    }

    /// Total number of registered file refs.
    #[must_use]
    pub fn count(&self) -> usize {
        self.entries.len()
    }

    /// Return refs filtered by kind.
    #[must_use]
    pub fn by_kind(&self, kind: MediaFileKind) -> Vec<&MediaFileRef> {
        self.entries.values().filter(|r| r.kind() == kind).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- MediaFileKind tests ---

    #[test]
    fn test_video_kind_is_video() {
        assert!(MediaFileKind::Video.is_video());
    }

    #[test]
    fn test_video_kind_not_audio() {
        assert!(!MediaFileKind::Video.is_audio());
    }

    #[test]
    fn test_audio_kind_is_audio() {
        assert!(MediaFileKind::Audio.is_audio());
    }

    #[test]
    fn test_audiovideo_is_both() {
        assert!(MediaFileKind::AudioVideo.is_video());
        assert!(MediaFileKind::AudioVideo.is_audio());
    }

    #[test]
    fn test_still_image_not_video() {
        assert!(!MediaFileKind::StillImage.is_video());
    }

    #[test]
    fn test_kind_label() {
        assert_eq!(MediaFileKind::Video.label(), "Video");
        assert_eq!(MediaFileKind::Audio.label(), "Audio");
        assert_eq!(MediaFileKind::StillImage.label(), "StillImage");
    }

    // --- MediaFileRef tests ---

    #[test]
    fn test_ref_new_no_path() {
        let r = MediaFileRef::new(0, "clip.mxf", MediaFileKind::Video);
        assert!(!r.has_path());
        assert_eq!(r.name(), "clip.mxf");
    }

    #[test]
    fn test_ref_set_path() {
        let mut r = MediaFileRef::new(1, "audio.wav", MediaFileKind::Audio);
        r.set_path(PathBuf::from("/mnt/media/audio.wav"));
        assert!(r.has_path());
    }

    #[test]
    fn test_ref_size() {
        let mut r = MediaFileRef::new(2, "img.dpx", MediaFileKind::StillImage);
        assert_eq!(r.size_bytes(), None);
        r.set_size(1_024_000);
        assert_eq!(r.size_bytes(), Some(1_024_000));
    }

    #[test]
    fn test_ref_id() {
        let r = MediaFileRef::new(99, "x.mxf", MediaFileKind::Video);
        assert_eq!(r.id(), 99);
    }

    // --- MediaFileRegistry tests ---

    #[test]
    fn test_registry_add_increments_count() {
        let mut reg = MediaFileRegistry::new();
        assert_eq!(reg.count(), 0);
        reg.add("a.mxf", MediaFileKind::Video);
        reg.add("b.wav", MediaFileKind::Audio);
        assert_eq!(reg.count(), 2);
    }

    #[test]
    fn test_registry_find_by_name() {
        let mut reg = MediaFileRegistry::new();
        reg.add("target.mxf", MediaFileKind::Video);
        let found = reg.find_by_name("target.mxf");
        assert!(found.is_some());
        assert_eq!(found.expect("test expectation failed").name(), "target.mxf");
    }

    #[test]
    fn test_registry_find_by_name_missing() {
        let reg = MediaFileRegistry::new();
        assert!(reg.find_by_name("ghost.mxf").is_none());
    }

    #[test]
    fn test_registry_find_by_id() {
        let mut reg = MediaFileRegistry::new();
        let id = reg.add("clip.mov", MediaFileKind::AudioVideo);
        assert!(reg.find_by_id(id).is_some());
        assert!(reg.find_by_id(id + 999).is_none());
    }

    #[test]
    fn test_registry_find_mut_and_set_path() {
        let mut reg = MediaFileRegistry::new();
        let id = reg.add("clip.mxf", MediaFileKind::Video);
        if let Some(r) = reg.find_by_id_mut(id) {
            r.set_path(PathBuf::from("/mnt/clip.mxf"));
        }
        assert!(reg
            .find_by_id(id)
            .expect("find_by_id should succeed")
            .has_path());
    }

    #[test]
    fn test_registry_by_kind() {
        let mut reg = MediaFileRegistry::new();
        reg.add("v1.mxf", MediaFileKind::Video);
        reg.add("v2.mxf", MediaFileKind::Video);
        reg.add("a1.wav", MediaFileKind::Audio);
        assert_eq!(reg.by_kind(MediaFileKind::Video).len(), 2);
        assert_eq!(reg.by_kind(MediaFileKind::Audio).len(), 1);
    }
}
