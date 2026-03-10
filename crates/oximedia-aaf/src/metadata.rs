//! Metadata support
//!
//! This module implements AAF metadata functionality:
//! - Comments (name/value pairs)
//! - Tagged values (typed metadata)
//! - KLV data (key-length-value)
//! - Descriptive metadata framework (DMF)
//! - Timecode metadata

use crate::dictionary::Auid;
use crate::timeline::{EditRate, Position};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Comment - simple name/value metadata pair
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Comment {
    /// Category or namespace
    pub category: Option<String>,
    /// Comment name
    pub name: String,
    /// Comment value
    pub value: String,
}

impl Comment {
    /// Create a new comment
    pub fn new(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self {
            category: None,
            name: name.into(),
            value: value.into(),
        }
    }

    /// Set category
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }
}

/// Tagged value - typed metadata with AUID
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaggedValue {
    /// Tag name
    pub name: String,
    /// Value
    pub value: TaggedValueData,
}

impl TaggedValue {
    /// Create a new tagged value
    pub fn new(name: impl Into<String>, value: TaggedValueData) -> Self {
        Self {
            name: name.into(),
            value,
        }
    }

    /// Create a string tagged value
    pub fn string(name: impl Into<String>, value: impl Into<String>) -> Self {
        Self::new(name, TaggedValueData::String(value.into()))
    }

    /// Create an integer tagged value
    pub fn integer(name: impl Into<String>, value: i64) -> Self {
        Self::new(name, TaggedValueData::Integer(value))
    }

    /// Create a float tagged value
    pub fn float(name: impl Into<String>, value: f64) -> Self {
        Self::new(name, TaggedValueData::Float(value))
    }

    /// Create a boolean tagged value
    pub fn boolean(name: impl Into<String>, value: bool) -> Self {
        Self::new(name, TaggedValueData::Boolean(value))
    }
}

/// Tagged value data types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TaggedValueData {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Binary data
    Binary(Vec<u8>),
    /// AUID
    Auid(Auid),
    /// Rational (numerator, denominator)
    Rational(i64, i64),
}

/// KLV data - key-length-value metadata
#[derive(Debug, Clone)]
pub struct KlvData {
    /// Key (UL or UUID)
    pub key: Vec<u8>,
    /// Value
    pub value: Vec<u8>,
}

impl KlvData {
    /// Create new KLV data
    #[must_use]
    pub fn new(key: Vec<u8>, value: Vec<u8>) -> Self {
        Self { key, value }
    }

    /// Get key as bytes
    #[must_use]
    pub fn key(&self) -> &[u8] {
        &self.key
    }

    /// Get value as bytes
    #[must_use]
    pub fn value(&self) -> &[u8] {
        &self.value
    }

    /// Get key length
    #[must_use]
    pub fn key_length(&self) -> usize {
        self.key.len()
    }

    /// Get value length
    #[must_use]
    pub fn value_length(&self) -> usize {
        self.value.len()
    }
}

/// Timecode for AAF (wrapper around oximedia-timecode)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Timecode {
    /// Hours (0-23)
    pub hours: u8,
    /// Minutes (0-59)
    pub minutes: u8,
    /// Seconds (0-59)
    pub seconds: u8,
    /// Frames (0 to fps-1)
    pub frames: u8,
    /// Drop frame flag
    pub drop_frame: bool,
    /// Frame rate
    pub fps: u8,
}

impl Timecode {
    /// Create a new timecode
    #[must_use]
    pub fn new(hours: u8, minutes: u8, seconds: u8, frames: u8, fps: u8, drop_frame: bool) -> Self {
        Self {
            hours,
            minutes,
            seconds,
            frames,
            drop_frame,
            fps,
        }
    }

    /// Create from position and edit rate
    #[must_use]
    pub fn from_position(position: Position, edit_rate: EditRate) -> Self {
        let fps = edit_rate.to_float().round() as u8;
        let total_frames = position.to_frames(edit_rate);

        let hours = (total_frames / (i64::from(fps) * 3600)) as u8;
        let remaining = total_frames % (i64::from(fps) * 3600);
        let minutes = (remaining / (i64::from(fps) * 60)) as u8;
        let remaining = remaining % (i64::from(fps) * 60);
        let seconds = (remaining / i64::from(fps)) as u8;
        let frames = (remaining % i64::from(fps)) as u8;

        Self {
            hours,
            minutes,
            seconds,
            frames,
            drop_frame: edit_rate.is_ntsc(),
            fps,
        }
    }

    /// Convert to position given edit rate
    #[must_use]
    pub fn to_position(&self, edit_rate: EditRate) -> Position {
        let fps = i64::from(self.fps);
        let total_frames = i64::from(self.hours) * 3600 * fps
            + i64::from(self.minutes) * 60 * fps
            + i64::from(self.seconds) * fps
            + i64::from(self.frames);

        Position::from_frames(total_frames, edit_rate)
    }

    /// Parse from string (format: HH:MM:SS:FF or HH:MM:SS;FF for drop frame)
    pub fn parse(s: &str, fps: u8) -> Result<Self, MetadataError> {
        let parts: Vec<&str> = s.split(&[':', ';'][..]).collect();
        if parts.len() != 4 {
            return Err(MetadataError::InvalidTimecode(s.to_string()));
        }

        let hours = parts[0]
            .parse::<u8>()
            .map_err(|_| MetadataError::InvalidTimecode(s.to_string()))?;
        let minutes = parts[1]
            .parse::<u8>()
            .map_err(|_| MetadataError::InvalidTimecode(s.to_string()))?;
        let seconds = parts[2]
            .parse::<u8>()
            .map_err(|_| MetadataError::InvalidTimecode(s.to_string()))?;
        let frames = parts[3]
            .parse::<u8>()
            .map_err(|_| MetadataError::InvalidTimecode(s.to_string()))?;

        let drop_frame = s.contains(';');

        Ok(Self {
            hours,
            minutes,
            seconds,
            frames,
            drop_frame,
            fps,
        })
    }
}

impl std::fmt::Display for Timecode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let separator = if self.drop_frame { ';' } else { ':' };
        write!(
            f,
            "{:02}:{:02}:{:02}{}{:02}",
            self.hours, self.minutes, self.seconds, separator, self.frames
        )
    }
}

/// Descriptive metadata framework (DMF)
#[derive(Debug, Clone)]
pub struct DescriptiveMetadata {
    /// Metadata items
    items: HashMap<String, MetadataValue>,
    /// Linked objects
    linked_objects: Vec<DescriptiveObjectReference>,
}

impl DescriptiveMetadata {
    /// Create new descriptive metadata
    #[must_use]
    pub fn new() -> Self {
        Self {
            items: HashMap::new(),
            linked_objects: Vec::new(),
        }
    }

    /// Add metadata item
    pub fn add_item(&mut self, key: impl Into<String>, value: MetadataValue) {
        self.items.insert(key.into(), value);
    }

    /// Get metadata item
    #[must_use]
    pub fn get_item(&self, key: &str) -> Option<&MetadataValue> {
        self.items.get(key)
    }

    /// Add linked object
    pub fn add_linked_object(&mut self, reference: DescriptiveObjectReference) {
        self.linked_objects.push(reference);
    }

    /// Get all items
    #[must_use]
    pub fn items(&self) -> &HashMap<String, MetadataValue> {
        &self.items
    }

    /// Get linked objects
    #[must_use]
    pub fn linked_objects(&self) -> &[DescriptiveObjectReference] {
        &self.linked_objects
    }
}

impl Default for DescriptiveMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// Date/time (ISO 8601 string)
    DateTime(String),
    /// URI
    Uri(String),
    /// Array of values
    Array(Vec<MetadataValue>),
    /// Nested metadata
    Object(HashMap<String, MetadataValue>),
}

/// Reference to a descriptive object
#[derive(Debug, Clone)]
pub struct DescriptiveObjectReference {
    /// Object ID
    pub object_id: String,
    /// Object type
    pub object_type: String,
}

/// Production metadata
#[derive(Debug, Clone)]
pub struct ProductionMetadata {
    /// Production title
    pub title: Option<String>,
    /// Episode title
    pub episode_title: Option<String>,
    /// Series title
    pub series_title: Option<String>,
    /// Production number
    pub production_number: Option<String>,
    /// Copyright
    pub copyright: Option<String>,
    /// Creation date
    pub creation_date: Option<String>,
    /// Production company
    pub production_company: Option<String>,
    /// Director
    pub director: Option<String>,
    /// Producer
    pub producer: Option<String>,
    /// Additional metadata
    pub additional: HashMap<String, String>,
}

impl ProductionMetadata {
    /// Create new production metadata
    #[must_use]
    pub fn new() -> Self {
        Self {
            title: None,
            episode_title: None,
            series_title: None,
            production_number: None,
            copyright: None,
            creation_date: None,
            production_company: None,
            director: None,
            producer: None,
            additional: HashMap::new(),
        }
    }

    /// Set title
    pub fn with_title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }

    /// Set production company
    pub fn with_company(mut self, company: impl Into<String>) -> Self {
        self.production_company = Some(company.into());
        self
    }

    /// Add additional metadata
    pub fn add_metadata(&mut self, key: impl Into<String>, value: impl Into<String>) {
        self.additional.insert(key.into(), value.into());
    }
}

impl Default for ProductionMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Technical metadata
#[derive(Debug, Clone)]
pub struct TechnicalMetadata {
    /// Video format
    pub video_format: Option<String>,
    /// Audio format
    pub audio_format: Option<String>,
    /// Frame rate
    pub frame_rate: Option<EditRate>,
    /// Aspect ratio
    pub aspect_ratio: Option<String>,
    /// Resolution
    pub resolution: Option<(u32, u32)>,
    /// Duration
    pub duration: Option<i64>,
    /// File size
    pub file_size: Option<u64>,
    /// Codec
    pub codec: Option<String>,
    /// Additional metadata
    pub additional: HashMap<String, String>,
}

impl TechnicalMetadata {
    /// Create new technical metadata
    #[must_use]
    pub fn new() -> Self {
        Self {
            video_format: None,
            audio_format: None,
            frame_rate: None,
            aspect_ratio: None,
            resolution: None,
            duration: None,
            file_size: None,
            codec: None,
            additional: HashMap::new(),
        }
    }

    /// Set video format
    pub fn with_video_format(mut self, format: impl Into<String>) -> Self {
        self.video_format = Some(format.into());
        self
    }

    /// Set frame rate
    #[must_use]
    pub fn with_frame_rate(mut self, rate: EditRate) -> Self {
        self.frame_rate = Some(rate);
        self
    }

    /// Set resolution
    #[must_use]
    pub fn with_resolution(mut self, width: u32, height: u32) -> Self {
        self.resolution = Some((width, height));
        self
    }
}

impl Default for TechnicalMetadata {
    fn default() -> Self {
        Self::new()
    }
}

/// Metadata error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MetadataError {
    /// Invalid timecode format
    InvalidTimecode(String),
    /// Invalid metadata value
    InvalidValue(String),
    /// Metadata not found
    NotFound(String),
}

impl std::fmt::Display for MetadataError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetadataError::InvalidTimecode(s) => write!(f, "Invalid timecode: {s}"),
            MetadataError::InvalidValue(s) => write!(f, "Invalid metadata value: {s}"),
            MetadataError::NotFound(s) => write!(f, "Metadata not found: {s}"),
        }
    }
}

impl std::error::Error for MetadataError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_comment() {
        let comment = Comment::new("Author", "John Doe").with_category("Production");
        assert_eq!(comment.name, "Author");
        assert_eq!(comment.value, "John Doe");
        assert_eq!(comment.category, Some("Production".to_string()));
    }

    #[test]
    fn test_tagged_value() {
        let tv_str = TaggedValue::string("Title", "My Video");
        assert_eq!(tv_str.name, "Title");
        if let TaggedValueData::String(s) = &tv_str.value {
            assert_eq!(s, "My Video");
        } else {
            panic!("Expected string value");
        }

        let tv_int = TaggedValue::integer("FrameCount", 1000);
        assert_eq!(tv_int.name, "FrameCount");
    }

    #[test]
    fn test_klv_data() {
        let key = vec![1, 2, 3, 4];
        let value = vec![5, 6, 7, 8, 9];
        let klv = KlvData::new(key.clone(), value.clone());

        assert_eq!(klv.key(), &key);
        assert_eq!(klv.value(), &value);
        assert_eq!(klv.key_length(), 4);
        assert_eq!(klv.value_length(), 5);
    }

    #[test]
    fn test_timecode() {
        let tc = Timecode::new(1, 2, 3, 4, 25, false);
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 2);
        assert_eq!(tc.seconds, 3);
        assert_eq!(tc.frames, 4);
        assert_eq!(tc.to_string(), "01:02:03:04");
    }

    #[test]
    fn test_timecode_parse() {
        let tc = Timecode::parse("01:02:03:04", 25).expect("tc should be valid");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 2);
        assert_eq!(tc.seconds, 3);
        assert_eq!(tc.frames, 4);
        assert!(!tc.drop_frame);

        let tc_df = Timecode::parse("01:02:03;04", 30).expect("tc_df should be valid");
        assert!(tc_df.drop_frame);
    }

    #[test]
    fn test_timecode_position_conversion() {
        let edit_rate = EditRate::new(25, 1);
        let tc = Timecode::new(0, 0, 1, 0, 25, false);
        let pos = tc.to_position(edit_rate);
        assert_eq!(pos.0, 25);

        let tc2 = Timecode::from_position(pos, edit_rate);
        assert_eq!(tc2.seconds, 1);
        assert_eq!(tc2.frames, 0);
    }

    #[test]
    fn test_descriptive_metadata() {
        let mut dm = DescriptiveMetadata::new();
        dm.add_item("title", MetadataValue::String("My Film".to_string()));
        dm.add_item("year", MetadataValue::Integer(2024));

        assert!(dm.get_item("title").is_some());
        assert!(dm.get_item("year").is_some());
        assert_eq!(dm.items().len(), 2);
    }

    #[test]
    fn test_production_metadata() {
        let pm = ProductionMetadata::new()
            .with_title("Episode 1")
            .with_company("ABC Productions");

        assert_eq!(pm.title, Some("Episode 1".to_string()));
        assert_eq!(pm.production_company, Some("ABC Productions".to_string()));
    }

    #[test]
    fn test_technical_metadata() {
        let tm = TechnicalMetadata::new()
            .with_video_format("HD")
            .with_frame_rate(EditRate::new(25, 1))
            .with_resolution(1920, 1080);

        assert_eq!(tm.video_format, Some("HD".to_string()));
        assert_eq!(tm.resolution, Some((1920, 1080)));
    }
}
