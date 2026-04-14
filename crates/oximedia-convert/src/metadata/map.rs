// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Metadata mapping between different formats.

use std::collections::HashMap;

/// Mapper for converting metadata between formats.
#[derive(Debug, Clone)]
pub struct MetadataMapper {
    mappings: HashMap<FormatPair, HashMap<String, String>>,
}

impl MetadataMapper {
    /// Create a new metadata mapper with default mappings.
    #[must_use]
    pub fn new() -> Self {
        let mut mapper = Self {
            mappings: HashMap::new(),
        };
        mapper.init_default_mappings();
        mapper
    }

    /// Map a metadata key from one format to another.
    #[must_use]
    pub fn map_key(&self, from_format: &str, to_format: &str, key: &str) -> Option<String> {
        let pair = FormatPair {
            from: from_format.to_string(),
            to: to_format.to_string(),
        };

        self.mappings
            .get(&pair)
            .and_then(|map| map.get(key))
            .cloned()
            .or_else(|| Some(key.to_string()))
    }

    /// Add a custom mapping.
    pub fn add_mapping(
        &mut self,
        from_format: &str,
        to_format: &str,
        from_key: &str,
        to_key: &str,
    ) {
        let pair = FormatPair {
            from: from_format.to_string(),
            to: to_format.to_string(),
        };

        self.mappings
            .entry(pair)
            .or_default()
            .insert(from_key.to_string(), to_key.to_string());
    }

    /// Get all mappings for a format pair.
    #[must_use]
    pub fn get_mappings(
        &self,
        from_format: &str,
        to_format: &str,
    ) -> Option<&HashMap<String, String>> {
        let pair = FormatPair {
            from: from_format.to_string(),
            to: to_format.to_string(),
        };

        self.mappings.get(&pair)
    }

    fn init_default_mappings(&mut self) {
        // MP4 to MKV
        self.add_mapping("mp4", "mkv", "com.apple.quicktime.title", "title");
        self.add_mapping("mp4", "mkv", "com.apple.quicktime.artist", "artist");
        self.add_mapping("mp4", "mkv", "com.apple.quicktime.description", "comment");

        // MKV to MP4
        self.add_mapping("mkv", "mp4", "title", "com.apple.quicktime.title");
        self.add_mapping("mkv", "mp4", "artist", "com.apple.quicktime.artist");
        self.add_mapping("mkv", "mp4", "comment", "com.apple.quicktime.description");

        // MP3 to M4A
        self.add_mapping("mp3", "m4a", "TIT2", "©nam");
        self.add_mapping("mp3", "m4a", "TPE1", "©ART");
        self.add_mapping("mp3", "m4a", "TALB", "©alb");
        self.add_mapping("mp3", "m4a", "TDRC", "©day");
        self.add_mapping("mp3", "m4a", "COMM", "©cmt");

        // M4A to MP3
        self.add_mapping("m4a", "mp3", "©nam", "TIT2");
        self.add_mapping("m4a", "mp3", "©ART", "TPE1");
        self.add_mapping("m4a", "mp3", "©alb", "TALB");
        self.add_mapping("m4a", "mp3", "©day", "TDRC");
        self.add_mapping("m4a", "mp3", "©cmt", "COMM");
    }
}

impl Default for MetadataMapper {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FormatPair {
    from: String,
    to: String,
}

/// Common metadata field names across formats.
#[derive(Debug, Clone, Copy)]
pub enum CommonField {
    /// Title/Name
    Title,
    /// Artist/Author
    Artist,
    /// Album
    Album,
    /// Date/Year
    Date,
    /// Comment/Description
    Comment,
    /// Genre
    Genre,
    /// Track number
    Track,
    /// Copyright
    Copyright,
    /// Encoder
    Encoder,
}

impl CommonField {
    /// Get the field name for a specific format.
    #[must_use]
    pub fn for_format(self, format: &str) -> &'static str {
        match (self, format) {
            (Self::Title, "mp4") => "com.apple.quicktime.title",
            (Self::Title, "mkv") => "title",
            (Self::Title, "mp3") => "TIT2",
            (Self::Title, "m4a") => "©nam",

            (Self::Artist, "mp4") => "com.apple.quicktime.artist",
            (Self::Artist, "mkv") => "artist",
            (Self::Artist, "mp3") => "TPE1",
            (Self::Artist, "m4a") => "©ART",

            (Self::Album, "mp4") => "com.apple.quicktime.album",
            (Self::Album, "mkv") => "album",
            (Self::Album, "mp3") => "TALB",
            (Self::Album, "m4a") => "©alb",

            (Self::Date, "mp4") => "com.apple.quicktime.creationdate",
            (Self::Date, "mkv") => "date",
            (Self::Date, "mp3") => "TDRC",
            (Self::Date, "m4a") => "©day",

            (Self::Comment, "mp4") => "com.apple.quicktime.description",
            (Self::Comment, "mkv") => "comment",
            (Self::Comment, "mp3") => "COMM",
            (Self::Comment, "m4a") => "©cmt",

            (Self::Genre, "mp4") => "com.apple.quicktime.genre",
            (Self::Genre, "mkv") => "genre",
            (Self::Genre, "mp3") => "TCON",
            (Self::Genre, "m4a") => "©gen",

            (Self::Track, "mp4") => "com.apple.quicktime.track",
            (Self::Track, "mkv") => "part_number",
            (Self::Track, "mp3") => "TRCK",
            (Self::Track, "m4a") => "trkn",

            (Self::Copyright, "mp4") => "com.apple.quicktime.copyright",
            (Self::Copyright, "mkv") => "copyright",
            (Self::Copyright, "mp3") => "TCOP",
            (Self::Copyright, "m4a") => "cprt",

            (Self::Encoder, "mp4") => "encoder",
            (Self::Encoder, "mkv") => "encoder",
            (Self::Encoder, "mp3") => "TSSE",
            (Self::Encoder, "m4a") => "©too",

            _ => "unknown",
        }
    }

    /// Get all common fields.
    #[must_use]
    pub fn all() -> Vec<Self> {
        vec![
            Self::Title,
            Self::Artist,
            Self::Album,
            Self::Date,
            Self::Comment,
            Self::Genre,
            Self::Track,
            Self::Copyright,
            Self::Encoder,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapper_creation() {
        let mapper = MetadataMapper::new();
        assert!(!mapper.mappings.is_empty());
    }

    #[test]
    fn test_map_key() {
        let mapper = MetadataMapper::new();

        // MP4 to MKV
        let mapped = mapper.map_key("mp4", "mkv", "com.apple.quicktime.title");
        assert_eq!(mapped, Some("title".to_string()));

        // MKV to MP4
        let mapped = mapper.map_key("mkv", "mp4", "title");
        assert_eq!(mapped, Some("com.apple.quicktime.title".to_string()));
    }

    #[test]
    fn test_add_custom_mapping() {
        let mut mapper = MetadataMapper::new();

        mapper.add_mapping("custom1", "custom2", "field1", "field2");

        let mapped = mapper.map_key("custom1", "custom2", "field1");
        assert_eq!(mapped, Some("field2".to_string()));
    }

    #[test]
    fn test_get_mappings() {
        let mapper = MetadataMapper::new();

        let mappings = mapper.get_mappings("mp4", "mkv");
        assert!(mappings.is_some());
        assert!(!mappings.unwrap().is_empty());

        let no_mappings = mapper.get_mappings("unknown1", "unknown2");
        assert!(no_mappings.is_none());
    }

    #[test]
    fn test_common_field_for_format() {
        assert_eq!(
            CommonField::Title.for_format("mp4"),
            "com.apple.quicktime.title"
        );
        assert_eq!(CommonField::Title.for_format("mkv"), "title");
        assert_eq!(CommonField::Title.for_format("mp3"), "TIT2");
        assert_eq!(CommonField::Title.for_format("m4a"), "©nam");
    }

    #[test]
    fn test_all_common_fields() {
        let fields = CommonField::all();
        assert_eq!(fields.len(), 9);
    }

    #[test]
    fn test_mp3_to_m4a_mapping() {
        let mapper = MetadataMapper::new();

        assert_eq!(
            mapper.map_key("mp3", "m4a", "TIT2"),
            Some("©nam".to_string())
        );
        assert_eq!(
            mapper.map_key("mp3", "m4a", "TPE1"),
            Some("©ART".to_string())
        );
    }
}
