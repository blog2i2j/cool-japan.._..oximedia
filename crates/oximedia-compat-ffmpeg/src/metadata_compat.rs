//! FFmpeg `-metadata` tag translation for OxiMedia.
//!
//! FFmpeg accepts `-metadata KEY=VALUE` pairs that embed container-level and
//! stream-level tags into media files. This module parses those pairs and maps
//! them to OxiMedia's canonical metadata model, which normalises key names and
//! provides typed accessors for well-known fields.
//!
//! ## Key normalisation
//!
//! FFmpeg accepts both `title` and `TITLE` (case-insensitive). OxiMedia
//! stores all keys in lowercase. A small set of well-known aliases are also
//! normalised (e.g. `track` → `tracknumber`, `date` → `year`).
//!
//! ## Supported well-known tags
//!
//! | FFmpeg key(s)          | Normalised OxiMedia key | Notes                       |
//! |------------------------|-------------------------|-----------------------------|
//! | `title`                | `title`                 | Track / film title          |
//! | `artist`, `author`     | `artist`                | Performing artist           |
//! | `album`                | `album`                 | Album or collection name    |
//! | `comment`              | `comment`               | Free-form comment           |
//! | `year`, `date`         | `year`                  | Release year                |
//! | `track`, `tracknumber` | `tracknumber`           | Track number in album       |
//! | `genre`                | `genre`                 | Musical genre               |
//! | `copyright`            | `copyright`             | Copyright notice            |
//! | `language`, `lang`     | `language`              | Language code (ISO 639)     |
//! | `encoder`              | `encoder`               | Encoding tool name          |
//! | `description`          | `description`           | Longer description          |
//! | `synopsis`             | `synopsis`              | Short synopsis              |
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_compat_ffmpeg::metadata_compat::{
//!     parse_metadata_arg, MetadataMap, MetadataScope,
//! };
//!
//! // Parse a single FFmpeg -metadata argument value
//! let (key, value) = parse_metadata_arg("title=My Great Video").unwrap();
//! assert_eq!(key, "title");
//! assert_eq!(value, "My Great Video");
//!
//! // Build a MetadataMap from multiple pairs
//! let args: Vec<String> = vec![
//!     "title=My Film".into(),
//!     "artist=COOLJAPAN OU".into(),
//!     "year=2026".into(),
//! ];
//! let map = MetadataMap::from_args(&args, MetadataScope::Global).unwrap();
//! assert_eq!(map.title(), Some("My Film"));
//! assert_eq!(map.artist(), Some("COOLJAPAN OU"));
//! ```

use std::collections::HashMap;

use thiserror::Error;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned when a `-metadata` argument cannot be parsed.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum MetadataError {
    /// The metadata argument does not contain an `=` separator.
    #[error("metadata argument '{0}' is missing '=' separator")]
    MissingSeparator(String),

    /// The metadata key is empty.
    #[error("metadata key is empty in argument '{0}'")]
    EmptyKey(String),
}

// ─────────────────────────────────────────────────────────────────────────────
// MetadataScope — global vs. per-stream
// ─────────────────────────────────────────────────────────────────────────────

/// The scope of a metadata block.
///
/// FFmpeg allows container-level (`-metadata`) and stream-level
/// (`-metadata:s:0`) tags. OxiMedia mirrors this with [`MetadataScope`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetadataScope {
    /// Container-level (global) tags, equivalent to FFmpeg `-metadata`.
    Global,
    /// Tags scoped to a specific video stream index.
    VideoStream(usize),
    /// Tags scoped to a specific audio stream index.
    AudioStream(usize),
    /// Tags scoped to a specific subtitle stream index.
    SubtitleStream(usize),
}

impl MetadataScope {
    /// Parse a metadata stream specifier suffix such as `s:0`, `s:a:0`, `g`.
    ///
    /// Returns `None` if the specifier cannot be recognised.
    pub fn from_specifier(spec: &str) -> Option<Self> {
        match spec.trim() {
            "" | "g" | "global" => Some(Self::Global),
            s if s.starts_with("s:v:") => {
                let idx = s["s:v:".len()..].parse::<usize>().ok()?;
                Some(Self::VideoStream(idx))
            }
            s if s.starts_with("s:a:") => {
                let idx = s["s:a:".len()..].parse::<usize>().ok()?;
                Some(Self::AudioStream(idx))
            }
            s if s.starts_with("s:s:") => {
                let idx = s["s:s:".len()..].parse::<usize>().ok()?;
                Some(Self::SubtitleStream(idx))
            }
            s if s.starts_with("s:") => {
                // Generic stream index — treat as video if numeric
                let idx = s["s:".len()..].parse::<usize>().ok()?;
                Some(Self::VideoStream(idx))
            }
            _ => None,
        }
    }
}

impl std::fmt::Display for MetadataScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Global => write!(f, "global"),
            Self::VideoStream(i) => write!(f, "video:{}", i),
            Self::AudioStream(i) => write!(f, "audio:{}", i),
            Self::SubtitleStream(i) => write!(f, "subtitle:{}", i),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Key normalisation table
// ─────────────────────────────────────────────────────────────────────────────

/// Normalise a FFmpeg metadata key to the OxiMedia canonical lowercase form.
///
/// Returns the input unchanged if no alias is known.
pub fn normalise_key(key: &str) -> &'static str {
    match key.to_lowercase().as_str() {
        "title" => "title",
        "artist" | "author" | "album_artist" => "artist",
        "album" => "album",
        "comment" | "comments" => "comment",
        "year" | "date" | "creation_time" => "year",
        "track" | "tracknumber" | "track_number" => "tracknumber",
        "genre" => "genre",
        "copyright" => "copyright",
        "language" | "lang" => "language",
        "encoder" | "encoded_by" => "encoder",
        "description" => "description",
        "synopsis" => "synopsis",
        "composer" => "composer",
        "lyricist" => "lyricist",
        "performer" => "performer",
        "disc" | "discnumber" => "discnumber",
        "show" | "series" => "show",
        "episode_id" | "episode" => "episode_id",
        "network" => "network",
        "location" => "location",
        "keywords" => "keywords",
        "rating" => "rating",
        "bpm" => "bpm",
        _ => {
            // Return a static str from a leak-free approach:
            // since we can't return dynamically, fall back to known fallbacks.
            // For unknown keys, we let the caller use the key as-is (lowercase).
            "__unknown__"
        }
    }
}

/// Normalise a key to a lowercase-owned String, handling both known aliases
/// and arbitrary user-defined keys.
pub fn normalise_key_owned(key: &str) -> String {
    let known = normalise_key(key);
    if known == "__unknown__" {
        key.to_lowercase()
    } else {
        known.to_string()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parsing
// ─────────────────────────────────────────────────────────────────────────────

/// Parse a single FFmpeg `-metadata` argument value of the form `KEY=VALUE`.
///
/// The key is normalised to lowercase. The value is taken verbatim.
///
/// # Errors
///
/// Returns [`MetadataError::MissingSeparator`] if `=` is absent, or
/// [`MetadataError::EmptyKey`] if the key portion is empty.
pub fn parse_metadata_arg(arg: &str) -> Result<(String, String), MetadataError> {
    let sep = arg.find('=').ok_or_else(|| MetadataError::MissingSeparator(arg.to_string()))?;
    let raw_key = &arg[..sep];
    if raw_key.is_empty() {
        return Err(MetadataError::EmptyKey(arg.to_string()));
    }
    let value = arg[sep + 1..].to_string();
    let key = normalise_key_owned(raw_key);
    Ok((key, value))
}

/// Parse multiple `-metadata KEY=VALUE` argument strings into a raw
/// `HashMap<String, String>`.
///
/// Duplicate keys are resolved by keeping the *last* value (matching FFmpeg
/// behaviour). Keys are normalised via [`normalise_key_owned`].
pub fn parse_metadata_args(args: &[String]) -> Result<HashMap<String, String>, MetadataError> {
    let mut map = HashMap::new();
    for arg in args {
        let (k, v) = parse_metadata_arg(arg)?;
        map.insert(k, v);
    }
    Ok(map)
}

// ─────────────────────────────────────────────────────────────────────────────
// MetadataMap — typed accessor wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// A typed metadata map with well-known field accessors.
///
/// Wraps a raw `HashMap<String, String>` and exposes ergonomic accessors for
/// commonly-used tags like `title`, `artist`, `year`, etc.
#[derive(Debug, Clone)]
pub struct MetadataMap {
    /// The underlying key/value store (keys are normalised lowercase).
    inner: HashMap<String, String>,
    /// The scope (global, stream-specific) of this metadata block.
    pub scope: MetadataScope,
}

impl Default for MetadataMap {
    fn default() -> Self {
        Self {
            inner: HashMap::new(),
            scope: MetadataScope::Global,
        }
    }
}

impl MetadataMap {
    /// Create an empty [`MetadataMap`] with the given scope.
    pub fn new(scope: MetadataScope) -> Self {
        Self {
            inner: HashMap::new(),
            scope,
        }
    }

    /// Build a [`MetadataMap`] from a slice of `KEY=VALUE` strings.
    pub fn from_args(
        args: &[String],
        scope: MetadataScope,
    ) -> Result<Self, MetadataError> {
        let inner = parse_metadata_args(args)?;
        Ok(Self { inner, scope })
    }

    /// Build a [`MetadataMap`] directly from a pre-parsed `HashMap`.
    ///
    /// Keys in the map are normalised automatically.
    pub fn from_map(raw: HashMap<String, String>, scope: MetadataScope) -> Self {
        let inner = raw
            .into_iter()
            .map(|(k, v)| (normalise_key_owned(&k), v))
            .collect();
        Self { inner, scope }
    }

    /// Insert or overwrite a tag.
    pub fn insert(&mut self, key: &str, value: impl Into<String>) {
        self.inner.insert(normalise_key_owned(key), value.into());
    }

    /// Retrieve a tag by raw key (normalised before lookup).
    pub fn get(&self, key: &str) -> Option<&str> {
        self.inner.get(&normalise_key_owned(key)).map(|s| s.as_str())
    }

    /// Remove a tag. Returns the old value if present.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.inner.remove(&normalise_key_owned(key))
    }

    /// Return `true` if the map contains no tags.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Return the number of tags.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Iterate over all (key, value) pairs in arbitrary order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &str)> {
        self.inner.iter().map(|(k, v)| (k.as_str(), v.as_str()))
    }

    /// Consume the map and return the underlying `HashMap`.
    pub fn into_inner(self) -> HashMap<String, String> {
        self.inner
    }

    // ── Well-known field accessors ────────────────────────────────────────

    /// Return the `title` tag value, if set.
    pub fn title(&self) -> Option<&str> {
        self.inner.get("title").map(|s| s.as_str())
    }

    /// Return the `artist` tag value, if set.
    pub fn artist(&self) -> Option<&str> {
        self.inner.get("artist").map(|s| s.as_str())
    }

    /// Return the `album` tag value, if set.
    pub fn album(&self) -> Option<&str> {
        self.inner.get("album").map(|s| s.as_str())
    }

    /// Return the `comment` tag value, if set.
    pub fn comment(&self) -> Option<&str> {
        self.inner.get("comment").map(|s| s.as_str())
    }

    /// Return the `year` tag value (also covers `date`), if set.
    pub fn year(&self) -> Option<&str> {
        self.inner.get("year").map(|s| s.as_str())
    }

    /// Return the `tracknumber` tag value (also covers `track`), if set.
    pub fn tracknumber(&self) -> Option<&str> {
        self.inner.get("tracknumber").map(|s| s.as_str())
    }

    /// Return the `genre` tag value, if set.
    pub fn genre(&self) -> Option<&str> {
        self.inner.get("genre").map(|s| s.as_str())
    }

    /// Return the `copyright` tag value, if set.
    pub fn copyright(&self) -> Option<&str> {
        self.inner.get("copyright").map(|s| s.as_str())
    }

    /// Return the `language` tag value, if set.
    pub fn language(&self) -> Option<&str> {
        self.inner.get("language").map(|s| s.as_str())
    }

    /// Return the `encoder` tag value, if set.
    pub fn encoder(&self) -> Option<&str> {
        self.inner.get("encoder").map(|s| s.as_str())
    }

    /// Return the `description` tag value, if set.
    pub fn description(&self) -> Option<&str> {
        self.inner.get("description").map(|s| s.as_str())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FFmpeg argument extraction helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Extract all `-metadata[:<scope>]` pairs from a raw FFmpeg argument slice.
///
/// Returns a `Vec` of `(MetadataScope, key, value)` tuples in the order they
/// appear in `args`.
///
/// Recognised forms:
/// - `-metadata TITLE=My Movie` — global scope
/// - `-metadata:s:v:0 title=Chapter 1` — video stream 0
/// - `-metadata:s:a:1 language=eng` — audio stream 1
pub fn extract_metadata_from_args(
    args: &[String],
) -> Vec<(MetadataScope, String, String)> {
    let mut result = Vec::new();
    let mut it = args.iter().peekable();

    while let Some(arg) = it.next() {
        if arg == "-metadata" {
            // Global metadata
            if let Some(val) = it.next() {
                if let Ok((k, v)) = parse_metadata_arg(val) {
                    result.push((MetadataScope::Global, k, v));
                }
            }
        } else if let Some(suffix) = arg.strip_prefix("-metadata:") {
            // Scoped metadata
            let scope = MetadataScope::from_specifier(suffix)
                .unwrap_or(MetadataScope::Global);
            if let Some(val) = it.next() {
                if let Ok((k, v)) = parse_metadata_arg(val) {
                    result.push((scope, k, v));
                }
            }
        }
    }

    result
}

/// Build a [`MetadataMap`] from a raw `HashMap<String, String>` that came from
/// an [`FfmpegArgs`] output specification's `.metadata` field.
///
/// Keys are normalised in the process.
pub fn metadata_map_from_ffmpeg_output(
    raw: &HashMap<String, String>,
) -> MetadataMap {
    let inner: HashMap<String, String> = raw
        .iter()
        .map(|(k, v)| (normalise_key_owned(k), v.clone()))
        .collect();
    MetadataMap {
        inner,
        scope: MetadataScope::Global,
    }
}

/// Merge two [`MetadataMap`]s, with `overlay` taking precedence on key conflicts.
pub fn merge_metadata(base: MetadataMap, overlay: MetadataMap) -> MetadataMap {
    let mut inner = base.inner;
    for (k, v) in overlay.inner {
        inner.insert(k, v);
    }
    MetadataMap {
        inner,
        scope: base.scope,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── parse_metadata_arg ───────────────────────────────────────────────────

    #[test]
    fn test_parse_metadata_arg_basic() {
        let (k, v) = parse_metadata_arg("title=My Great Film").expect("should parse");
        assert_eq!(k, "title");
        assert_eq!(v, "My Great Film");
    }

    #[test]
    fn test_parse_metadata_arg_alias_artist() {
        let (k, _) = parse_metadata_arg("author=Someone").expect("should parse");
        assert_eq!(k, "artist", "author should normalise to artist");
    }

    #[test]
    fn test_parse_metadata_arg_alias_date_to_year() {
        let (k, v) = parse_metadata_arg("date=2026").expect("should parse");
        assert_eq!(k, "year", "date should normalise to year");
        assert_eq!(v, "2026");
    }

    #[test]
    fn test_parse_metadata_arg_alias_track_to_tracknumber() {
        let (k, _) = parse_metadata_arg("track=3").expect("should parse");
        assert_eq!(k, "tracknumber", "track should normalise to tracknumber");
    }

    #[test]
    fn test_parse_metadata_arg_missing_separator() {
        let err = parse_metadata_arg("titleOnly").expect_err("should fail");
        assert!(matches!(err, MetadataError::MissingSeparator(_)));
    }

    #[test]
    fn test_parse_metadata_arg_empty_key() {
        let err = parse_metadata_arg("=value").expect_err("should fail on empty key");
        assert!(matches!(err, MetadataError::EmptyKey(_)));
    }

    #[test]
    fn test_parse_metadata_arg_empty_value() {
        // Empty value is allowed
        let (k, v) = parse_metadata_arg("comment=").expect("should allow empty value");
        assert_eq!(k, "comment");
        assert_eq!(v, "");
    }

    #[test]
    fn test_parse_metadata_arg_value_with_equals() {
        // Only first `=` is the separator; value may contain `=`
        let (k, v) = parse_metadata_arg("comment=key=value").expect("should parse");
        assert_eq!(k, "comment");
        assert_eq!(v, "key=value");
    }

    // ── MetadataMap ──────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_map_from_args() {
        let args: Vec<String> = vec![
            "title=Test Film".into(),
            "artist=COOLJAPAN".into(),
            "year=2026".into(),
        ];
        let map = MetadataMap::from_args(&args, MetadataScope::Global).expect("should succeed");
        assert_eq!(map.title(), Some("Test Film"));
        assert_eq!(map.artist(), Some("COOLJAPAN"));
        assert_eq!(map.year(), Some("2026"));
    }

    #[test]
    fn test_metadata_map_accessors() {
        let args: Vec<String> = vec![
            "title=T".into(),
            "album=A".into(),
            "comment=C".into(),
            "genre=G".into(),
            "copyright=CR".into(),
            "language=eng".into(),
            "encoder=OxiMedia".into(),
            "tracknumber=5".into(),
            "description=D".into(),
        ];
        let map = MetadataMap::from_args(&args, MetadataScope::Global).expect("should succeed");
        assert_eq!(map.album(), Some("A"));
        assert_eq!(map.comment(), Some("C"));
        assert_eq!(map.genre(), Some("G"));
        assert_eq!(map.copyright(), Some("CR"));
        assert_eq!(map.language(), Some("eng"));
        assert_eq!(map.encoder(), Some("OxiMedia"));
        assert_eq!(map.tracknumber(), Some("5"));
        assert_eq!(map.description(), Some("D"));
    }

    #[test]
    fn test_metadata_map_insert_and_get() {
        let mut map = MetadataMap::new(MetadataScope::Global);
        map.insert("title", "Inserted");
        assert_eq!(map.get("title"), Some("Inserted"));
        assert_eq!(map.len(), 1);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_metadata_map_remove() {
        let mut map = MetadataMap::new(MetadataScope::Global);
        map.insert("title", "To Remove");
        let old = map.remove("title");
        assert_eq!(old, Some("To Remove".to_string()));
        assert!(map.is_empty());
    }

    #[test]
    fn test_metadata_map_from_map_normalises_keys() {
        let mut raw = HashMap::new();
        raw.insert("TITLE".to_string(), "Uppercase Title".to_string());
        raw.insert("Author".to_string(), "Someone".to_string());
        let map = MetadataMap::from_map(raw, MetadataScope::Global);
        assert_eq!(map.title(), Some("Uppercase Title"));
        assert_eq!(map.artist(), Some("Someone"));
    }

    #[test]
    fn test_metadata_map_duplicate_key_last_wins() {
        let args: Vec<String> = vec![
            "title=First".into(),
            "title=Second".into(),
        ];
        let map = MetadataMap::from_args(&args, MetadataScope::Global).expect("should succeed");
        assert_eq!(map.title(), Some("Second"), "last value should win");
    }

    // ── MetadataScope ────────────────────────────────────────────────────────

    #[test]
    fn test_metadata_scope_from_specifier_global() {
        assert_eq!(MetadataScope::from_specifier("g"), Some(MetadataScope::Global));
        assert_eq!(MetadataScope::from_specifier(""), Some(MetadataScope::Global));
        assert_eq!(MetadataScope::from_specifier("global"), Some(MetadataScope::Global));
    }

    #[test]
    fn test_metadata_scope_from_specifier_video() {
        assert_eq!(
            MetadataScope::from_specifier("s:v:0"),
            Some(MetadataScope::VideoStream(0))
        );
        assert_eq!(
            MetadataScope::from_specifier("s:v:2"),
            Some(MetadataScope::VideoStream(2))
        );
    }

    #[test]
    fn test_metadata_scope_from_specifier_audio() {
        assert_eq!(
            MetadataScope::from_specifier("s:a:1"),
            Some(MetadataScope::AudioStream(1))
        );
    }

    #[test]
    fn test_metadata_scope_from_specifier_subtitle() {
        assert_eq!(
            MetadataScope::from_specifier("s:s:0"),
            Some(MetadataScope::SubtitleStream(0))
        );
    }

    #[test]
    fn test_metadata_scope_display() {
        assert_eq!(MetadataScope::Global.to_string(), "global");
        assert_eq!(MetadataScope::VideoStream(0).to_string(), "video:0");
        assert_eq!(MetadataScope::AudioStream(2).to_string(), "audio:2");
        assert_eq!(MetadataScope::SubtitleStream(1).to_string(), "subtitle:1");
    }

    // ── extract_metadata_from_args ───────────────────────────────────────────

    #[test]
    fn test_extract_metadata_from_args_global() {
        let args: Vec<String> = vec![
            "-i".into(), "in.mp4".into(),
            "-metadata".into(), "title=My Movie".into(),
            "-metadata".into(), "artist=Director".into(),
            "out.mkv".into(),
        ];
        let extracted = extract_metadata_from_args(&args);
        assert_eq!(extracted.len(), 2);
        assert_eq!(extracted[0].0, MetadataScope::Global);
        assert_eq!(extracted[0].1, "title");
        assert_eq!(extracted[0].2, "My Movie");
    }

    #[test]
    fn test_extract_metadata_from_args_scoped() {
        let args: Vec<String> = vec![
            "-i".into(), "in.mp4".into(),
            "-metadata:s:a:0".into(), "language=eng".into(),
            "out.mkv".into(),
        ];
        let extracted = extract_metadata_from_args(&args);
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].0, MetadataScope::AudioStream(0));
        assert_eq!(extracted[0].1, "language");
    }

    #[test]
    fn test_extract_metadata_from_args_empty() {
        let args: Vec<String> = vec!["-i".into(), "in.mp4".into(), "out.mkv".into()];
        let extracted = extract_metadata_from_args(&args);
        assert!(extracted.is_empty());
    }

    // ── merge_metadata ───────────────────────────────────────────────────────

    #[test]
    fn test_merge_metadata_overlay_wins() {
        let mut base = MetadataMap::new(MetadataScope::Global);
        base.insert("title", "Base Title");
        base.insert("artist", "Base Artist");

        let mut overlay = MetadataMap::new(MetadataScope::Global);
        overlay.insert("title", "Override Title");
        overlay.insert("year", "2026");

        let merged = merge_metadata(base, overlay);
        assert_eq!(merged.title(), Some("Override Title"), "overlay should win");
        assert_eq!(merged.artist(), Some("Base Artist"), "base-only keys preserved");
        assert_eq!(merged.year(), Some("2026"), "overlay-only keys present");
    }

    // ── normalise_key_owned ──────────────────────────────────────────────────

    #[test]
    fn test_normalise_key_owned_unknown() {
        // Unknown keys should be returned as lowercase
        let k = normalise_key_owned("MyCustomTag");
        assert_eq!(k, "mycustomtag");
    }

    #[test]
    fn test_metadata_map_iter() {
        let args: Vec<String> = vec![
            "title=T".into(),
            "artist=A".into(),
        ];
        let map = MetadataMap::from_args(&args, MetadataScope::Global).expect("should succeed");
        let pairs: Vec<(&str, &str)> = map.iter().collect();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_metadata_map_into_inner() {
        let args: Vec<String> = vec!["title=T".into()];
        let map = MetadataMap::from_args(&args, MetadataScope::Global).expect("should succeed");
        let inner = map.into_inner();
        assert_eq!(inner.get("title").map(|s| s.as_str()), Some("T"));
    }
}
