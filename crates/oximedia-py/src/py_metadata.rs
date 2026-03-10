#![allow(dead_code)]
//! # Python Metadata Bindings
//!
//! Pure-Rust side of metadata management for OxiMedia's Python bindings.
//! Provides strongly-typed metadata fields, a container type, and a
//! converter that maps between the Rust domain model and flat string maps
//! as used by the Python layer.

use std::collections::HashMap;

/// A well-known metadata field for media assets.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetadataField {
    /// Title of the programme / track.
    Title,
    /// Primary creator or director.
    Author,
    /// ISO 8601 creation timestamp.
    CreationDate,
    /// Duration in milliseconds.
    DurationMs,
    /// Video codec identifier (e.g. `"AV1"`).
    VideoCodec,
    /// Audio codec identifier (e.g. `"Opus"`).
    AudioCodec,
    /// File container format (e.g. `"MKV"`).
    Container,
    /// Width in pixels.
    Width,
    /// Height in pixels.
    Height,
    /// Frame rate as a rational string `"num/den"`.
    FrameRate,
    /// Bit-rate in bits per second.
    BitRate,
    /// Arbitrary user-defined tag.
    Custom(String),
}

impl MetadataField {
    /// Return the canonical string key used in flat maps.
    pub fn key(&self) -> String {
        match self {
            MetadataField::Title => "title".into(),
            MetadataField::Author => "author".into(),
            MetadataField::CreationDate => "creation_date".into(),
            MetadataField::DurationMs => "duration_ms".into(),
            MetadataField::VideoCodec => "video_codec".into(),
            MetadataField::AudioCodec => "audio_codec".into(),
            MetadataField::Container => "container".into(),
            MetadataField::Width => "width".into(),
            MetadataField::Height => "height".into(),
            MetadataField::FrameRate => "frame_rate".into(),
            MetadataField::BitRate => "bit_rate".into(),
            MetadataField::Custom(k) => format!("custom.{k}"),
        }
    }

    /// Parse a canonical key string back into a [`MetadataField`].
    pub fn from_key(key: &str) -> Self {
        match key {
            "title" => MetadataField::Title,
            "author" => MetadataField::Author,
            "creation_date" => MetadataField::CreationDate,
            "duration_ms" => MetadataField::DurationMs,
            "video_codec" => MetadataField::VideoCodec,
            "audio_codec" => MetadataField::AudioCodec,
            "container" => MetadataField::Container,
            "width" => MetadataField::Width,
            "height" => MetadataField::Height,
            "frame_rate" => MetadataField::FrameRate,
            "bit_rate" => MetadataField::BitRate,
            other => {
                let stripped = other.strip_prefix("custom.").unwrap_or(other);
                MetadataField::Custom(stripped.to_string())
            }
        }
    }
}

/// A typed metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    /// UTF-8 text value.
    Text(String),
    /// Integer value.
    Int(i64),
    /// Floating-point value.
    Float(f64),
    /// Boolean flag.
    Bool(bool),
}

impl MetadataValue {
    /// Serialize to a string for Python interop.
    pub fn to_string_repr(&self) -> String {
        match self {
            MetadataValue::Text(s) => s.clone(),
            MetadataValue::Int(i) => i.to_string(),
            MetadataValue::Float(f) => format!("{f:.6}"),
            MetadataValue::Bool(b) => if *b { "true" } else { "false" }.into(),
        }
    }

    /// Parse a string back into a `MetadataValue` (always `Text`).
    pub fn from_string_repr(s: impl Into<String>) -> Self {
        MetadataValue::Text(s.into())
    }
}

impl std::fmt::Display for MetadataValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_string_repr())
    }
}

/// Container for media metadata entries.
#[derive(Debug, Clone, Default)]
pub struct PyMetadata {
    entries: HashMap<MetadataField, MetadataValue>,
}

impl PyMetadata {
    /// Create an empty metadata container.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite a field.
    pub fn set(&mut self, field: MetadataField, value: MetadataValue) {
        self.entries.insert(field, value);
    }

    /// Retrieve a field value.
    pub fn get(&self, field: &MetadataField) -> Option<&MetadataValue> {
        self.entries.get(field)
    }

    /// Remove a field, returning the old value if present.
    pub fn remove(&mut self, field: &MetadataField) -> Option<MetadataValue> {
        self.entries.remove(field)
    }

    /// Return `true` if the field is present.
    pub fn contains(&self, field: &MetadataField) -> bool {
        self.entries.contains_key(field)
    }

    /// Number of fields stored.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Return `true` if no fields are stored.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Merge another `PyMetadata` into this one. Fields from `other` overwrite.
    pub fn merge(&mut self, other: PyMetadata) {
        for (k, v) in other.entries {
            self.entries.insert(k, v);
        }
    }
}

/// Converts between [`PyMetadata`] and flat `HashMap<String, String>`.
pub struct MetadataConverter;

impl MetadataConverter {
    /// Export metadata to a flat string map (for Python dict interop).
    pub fn to_flat(meta: &PyMetadata) -> HashMap<String, String> {
        meta.entries
            .iter()
            .map(|(k, v)| (k.key(), v.to_string_repr()))
            .collect()
    }

    /// Import metadata from a flat string map.
    pub fn from_flat(map: &HashMap<String, String>) -> PyMetadata {
        let mut meta = PyMetadata::new();
        for (k, v) in map {
            meta.set(
                MetadataField::from_key(k),
                MetadataValue::from_string_repr(v),
            );
        }
        meta
    }
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_field_key_title() {
        assert_eq!(MetadataField::Title.key(), "title");
    }

    #[test]
    fn test_field_key_custom() {
        assert_eq!(MetadataField::Custom("foo".into()).key(), "custom.foo");
    }

    #[test]
    fn test_field_from_key_roundtrip() {
        let f = MetadataField::VideoCodec;
        assert_eq!(MetadataField::from_key(&f.key()), f);
    }

    #[test]
    fn test_field_from_key_custom() {
        let f = MetadataField::from_key("custom.bar");
        assert_eq!(f, MetadataField::Custom("bar".into()));
    }

    #[test]
    fn test_field_from_key_unknown() {
        // Unknown keys become Custom(_)
        let f = MetadataField::from_key("weird_key");
        assert_eq!(f, MetadataField::Custom("weird_key".into()));
    }

    #[test]
    fn test_value_text_repr() {
        let v = MetadataValue::Text("hello".into());
        assert_eq!(v.to_string_repr(), "hello");
    }

    #[test]
    fn test_value_int_repr() {
        let v = MetadataValue::Int(42);
        assert_eq!(v.to_string_repr(), "42");
    }

    #[test]
    fn test_value_bool_repr() {
        assert_eq!(MetadataValue::Bool(true).to_string_repr(), "true");
        assert_eq!(MetadataValue::Bool(false).to_string_repr(), "false");
    }

    #[test]
    fn test_value_display() {
        let v = MetadataValue::Int(-7);
        assert_eq!(format!("{v}"), "-7");
    }

    #[test]
    fn test_metadata_set_get() {
        let mut m = PyMetadata::new();
        m.set(MetadataField::Title, MetadataValue::Text("My Film".into()));
        assert_eq!(
            m.get(&MetadataField::Title),
            Some(&MetadataValue::Text("My Film".into()))
        );
    }

    #[test]
    fn test_metadata_overwrite() {
        let mut m = PyMetadata::new();
        m.set(MetadataField::Width, MetadataValue::Int(1920));
        m.set(MetadataField::Width, MetadataValue::Int(3840));
        assert_eq!(
            m.get(&MetadataField::Width),
            Some(&MetadataValue::Int(3840))
        );
    }

    #[test]
    fn test_metadata_remove() {
        let mut m = PyMetadata::new();
        m.set(MetadataField::Author, MetadataValue::Text("Alice".into()));
        let old = m
            .remove(&MetadataField::Author)
            .expect("old should be valid");
        assert_eq!(old, MetadataValue::Text("Alice".into()));
        assert!(!m.contains(&MetadataField::Author));
    }

    #[test]
    fn test_metadata_merge() {
        let mut a = PyMetadata::new();
        a.set(MetadataField::Title, MetadataValue::Text("A".into()));
        let mut b = PyMetadata::new();
        b.set(MetadataField::Author, MetadataValue::Text("B".into()));
        b.set(MetadataField::Title, MetadataValue::Text("B-Title".into()));
        a.merge(b);
        assert_eq!(a.len(), 2);
        assert_eq!(
            a.get(&MetadataField::Title),
            Some(&MetadataValue::Text("B-Title".into()))
        );
    }

    #[test]
    fn test_converter_roundtrip() {
        let mut m = PyMetadata::new();
        m.set(MetadataField::BitRate, MetadataValue::Int(5_000_000));
        let flat = MetadataConverter::to_flat(&m);
        let restored = MetadataConverter::from_flat(&flat);
        assert_eq!(
            restored
                .get(&MetadataField::BitRate)
                .map(|v| v.to_string_repr()),
            Some("5000000".to_string())
        );
    }

    #[test]
    fn test_metadata_is_empty() {
        let m = PyMetadata::new();
        assert!(m.is_empty());
    }

    #[test]
    fn test_metadata_len() {
        let mut m = PyMetadata::new();
        m.set(MetadataField::Width, MetadataValue::Int(1920));
        m.set(MetadataField::Height, MetadataValue::Int(1080));
        assert_eq!(m.len(), 2);
    }
}
