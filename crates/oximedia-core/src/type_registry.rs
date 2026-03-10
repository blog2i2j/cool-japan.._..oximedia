//! Runtime type registry for media format negotiation.
//!
//! Provides [`TypeKind`] (a classification of media data), [`TypeInfo`]
//! (metadata about a single registered type), and [`TypeRegistry`] (a
//! simple lookup table keyed on string names).
//!
//! # Examples
//!
//! ```
//! use oximedia_core::type_registry::{TypeKind, TypeInfo, TypeRegistry};
//!
//! let mut reg = TypeRegistry::new();
//! reg.register(TypeInfo::new("yuv420p", TypeKind::VideoFrame, 3));
//! let info = reg.lookup("yuv420p").expect("type not found");
//! assert_eq!(info.kind, TypeKind::VideoFrame);
//! ```

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use std::collections::HashMap;

/// Classification of a registered media type.
///
/// Used by [`TypeInfo`] to describe what category of data a type represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeKind {
    /// Raw or encoded video frame data.
    VideoFrame,
    /// Raw or encoded audio sample buffer.
    AudioBuffer,
    /// Subtitle or caption packet.
    Subtitle,
    /// Arbitrary binary data packet.
    DataPacket,
    /// Container-level metadata record.
    Metadata,
}

impl TypeKind {
    /// Returns a human-readable label for this kind.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::VideoFrame => "video_frame",
            Self::AudioBuffer => "audio_buffer",
            Self::Subtitle => "subtitle",
            Self::DataPacket => "data_packet",
            Self::Metadata => "metadata",
        }
    }

    /// Returns `true` if this kind carries time-coded media essence.
    #[inline]
    #[must_use]
    pub fn is_essence(self) -> bool {
        matches!(self, Self::VideoFrame | Self::AudioBuffer)
    }
}

impl std::fmt::Display for TypeKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Metadata record for a single registered type.
///
/// Each record holds the canonical name, classification, and the number of
/// component planes (e.g. 3 for YUV 4:2:0, 1 for packed RGBA).
///
/// # Examples
///
/// ```
/// use oximedia_core::type_registry::{TypeInfo, TypeKind};
///
/// let info = TypeInfo::new("nv12", TypeKind::VideoFrame, 2);
/// assert_eq!(info.name, "nv12");
/// assert_eq!(info.planes, 2);
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TypeInfo {
    /// Canonical name used as the lookup key (lower-case, ASCII).
    pub name: String,
    /// Classification of this type.
    pub kind: TypeKind,
    /// Number of distinct memory planes (0 for packed / non-planar types).
    pub planes: u8,
}

impl TypeInfo {
    /// Creates a new [`TypeInfo`] with the given parameters.
    #[must_use]
    pub fn new(name: &str, kind: TypeKind, planes: u8) -> Self {
        Self {
            name: name.to_owned(),
            kind,
            planes,
        }
    }

    /// Returns `true` if this type uses a planar memory layout.
    #[inline]
    #[must_use]
    pub fn is_planar(&self) -> bool {
        self.planes > 1
    }
}

/// A registry that maps type names to their [`TypeInfo`] records.
///
/// Types are stored in a [`HashMap`] keyed on the canonical name.
/// Lookups are case-sensitive; callers should normalise to lower-case before
/// calling [`register`](Self::register) or [`lookup`](Self::lookup).
///
/// # Examples
///
/// ```
/// use oximedia_core::type_registry::{TypeInfo, TypeKind, TypeRegistry};
///
/// let mut reg = TypeRegistry::new();
/// reg.register(TypeInfo::new("opus", TypeKind::AudioBuffer, 0));
/// assert!(reg.lookup("opus").is_some());
/// assert!(reg.lookup("flac").is_none());
/// ```
#[derive(Debug, Default)]
pub struct TypeRegistry {
    entries: HashMap<String, TypeInfo>,
}

impl TypeRegistry {
    /// Creates an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }

    /// Creates a registry pre-populated with common video/audio format types.
    #[must_use]
    pub fn with_defaults() -> Self {
        let mut reg = Self::new();
        reg.register(TypeInfo::new("yuv420p", TypeKind::VideoFrame, 3));
        reg.register(TypeInfo::new("yuv422p", TypeKind::VideoFrame, 3));
        reg.register(TypeInfo::new("yuv444p", TypeKind::VideoFrame, 3));
        reg.register(TypeInfo::new("nv12", TypeKind::VideoFrame, 2));
        reg.register(TypeInfo::new("rgba", TypeKind::VideoFrame, 0));
        reg.register(TypeInfo::new("pcm_s16le", TypeKind::AudioBuffer, 0));
        reg.register(TypeInfo::new("pcm_f32le", TypeKind::AudioBuffer, 0));
        reg.register(TypeInfo::new("srt", TypeKind::Subtitle, 0));
        reg
    }

    /// Registers a type, replacing any existing entry with the same name.
    pub fn register(&mut self, info: TypeInfo) {
        self.entries.insert(info.name.clone(), info);
    }

    /// Removes the type with the given name, returning it if it existed.
    pub fn unregister(&mut self, name: &str) -> Option<TypeInfo> {
        self.entries.remove(name)
    }

    /// Looks up a type by its canonical name.
    ///
    /// Returns `None` if no entry with that name has been registered.
    #[must_use]
    pub fn lookup(&self, name: &str) -> Option<&TypeInfo> {
        self.entries.get(name)
    }

    /// Returns `true` if the given name is registered.
    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    /// Returns the number of registered types.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no types have been registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Returns an iterator over all registered [`TypeInfo`] records.
    pub fn iter(&self) -> impl Iterator<Item = &TypeInfo> {
        self.entries.values()
    }

    /// Returns all entries whose [`TypeKind`] matches `kind`.
    #[must_use]
    pub fn by_kind(&self, kind: TypeKind) -> Vec<&TypeInfo> {
        self.entries.values().filter(|i| i.kind == kind).collect()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registry_is_empty() {
        let reg = TypeRegistry::new();
        assert!(reg.is_empty());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn register_and_lookup() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeInfo::new("yuv420p", TypeKind::VideoFrame, 3));
        let info = reg.lookup("yuv420p").expect("not found");
        assert_eq!(info.name, "yuv420p");
        assert_eq!(info.kind, TypeKind::VideoFrame);
        assert_eq!(info.planes, 3);
    }

    #[test]
    fn lookup_missing_returns_none() {
        let reg = TypeRegistry::new();
        assert!(reg.lookup("unknown").is_none());
    }

    #[test]
    fn register_overwrites_existing() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeInfo::new("x", TypeKind::DataPacket, 0));
        reg.register(TypeInfo::new("x", TypeKind::Metadata, 1));
        assert_eq!(
            reg.lookup("x").expect("lookup should succeed").kind,
            TypeKind::Metadata
        );
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn unregister_removes_entry() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeInfo::new("opus", TypeKind::AudioBuffer, 0));
        let removed = reg.unregister("opus");
        assert!(removed.is_some());
        assert!(!reg.contains("opus"));
    }

    #[test]
    fn unregister_missing_returns_none() {
        let mut reg = TypeRegistry::new();
        assert!(reg.unregister("nonexistent").is_none());
    }

    #[test]
    fn contains_returns_correct_bool() {
        let mut reg = TypeRegistry::new();
        reg.register(TypeInfo::new("nv12", TypeKind::VideoFrame, 2));
        assert!(reg.contains("nv12"));
        assert!(!reg.contains("rgb24"));
    }

    #[test]
    fn with_defaults_has_common_types() {
        let reg = TypeRegistry::with_defaults();
        assert!(reg.contains("yuv420p"));
        assert!(reg.contains("nv12"));
        assert!(reg.contains("pcm_s16le"));
        assert!(reg.contains("srt"));
    }

    #[test]
    fn by_kind_filters_correctly() {
        let reg = TypeRegistry::with_defaults();
        let video = reg.by_kind(TypeKind::VideoFrame);
        assert!(!video.is_empty());
        for entry in &video {
            assert_eq!(entry.kind, TypeKind::VideoFrame);
        }
    }

    #[test]
    fn iter_covers_all_entries() {
        let reg = TypeRegistry::with_defaults();
        assert_eq!(reg.iter().count(), reg.len());
    }

    #[test]
    fn type_kind_label_is_non_empty() {
        for kind in [
            TypeKind::VideoFrame,
            TypeKind::AudioBuffer,
            TypeKind::Subtitle,
            TypeKind::DataPacket,
            TypeKind::Metadata,
        ] {
            assert!(!kind.label().is_empty());
        }
    }

    #[test]
    fn type_kind_is_essence() {
        assert!(TypeKind::VideoFrame.is_essence());
        assert!(TypeKind::AudioBuffer.is_essence());
        assert!(!TypeKind::Subtitle.is_essence());
        assert!(!TypeKind::DataPacket.is_essence());
    }

    #[test]
    fn type_info_is_planar() {
        let planar = TypeInfo::new("yuv420p", TypeKind::VideoFrame, 3);
        assert!(planar.is_planar());
        let packed = TypeInfo::new("rgba", TypeKind::VideoFrame, 0);
        assert!(!packed.is_planar());
    }

    #[test]
    fn default_registry_is_empty() {
        let reg = TypeRegistry::default();
        assert!(reg.is_empty());
    }

    #[test]
    fn type_kind_display() {
        assert_eq!(TypeKind::VideoFrame.to_string(), "video_frame");
        assert_eq!(TypeKind::AudioBuffer.to_string(), "audio_buffer");
    }
}
