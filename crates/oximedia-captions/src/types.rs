//! Core types for caption authoring

use crate::error::{CaptionError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;
use uuid::Uuid;

/// Unique identifier for a caption
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CaptionId(Uuid);

impl CaptionId {
    /// Create a new random caption ID
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Create a caption ID from a UUID
    #[must_use]
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the UUID
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }
}

impl Default for CaptionId {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for CaptionId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Timestamp in microseconds
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct Timestamp(i64);

impl Timestamp {
    /// Create a timestamp from microseconds
    #[must_use]
    pub const fn from_micros(micros: i64) -> Self {
        Self(micros)
    }

    /// Create a timestamp from milliseconds
    #[must_use]
    pub const fn from_millis(millis: i64) -> Self {
        Self(millis * 1000)
    }

    /// Create a timestamp from seconds
    #[must_use]
    pub const fn from_secs(secs: i64) -> Self {
        Self(secs * 1_000_000)
    }

    /// Create a timestamp from hours, minutes, seconds, and milliseconds
    #[must_use]
    pub const fn from_hmsm(hours: u32, minutes: u32, seconds: u32, millis: u32) -> Self {
        let total_secs = (hours as i64 * 3600) + (minutes as i64 * 60) + (seconds as i64);
        Self((total_secs * 1_000_000) + (millis as i64 * 1000))
    }

    /// Get the timestamp as microseconds
    #[must_use]
    pub const fn as_micros(&self) -> i64 {
        self.0
    }

    /// Get the timestamp as milliseconds
    #[must_use]
    pub const fn as_millis(&self) -> i64 {
        self.0 / 1000
    }

    /// Get the timestamp as seconds
    #[must_use]
    pub const fn as_secs(&self) -> i64 {
        self.0 / 1_000_000
    }

    /// Get the timestamp as (hours, minutes, seconds, milliseconds)
    #[must_use]
    pub const fn as_hmsm(&self) -> (u32, u32, u32, u32) {
        let total_millis = self.0 / 1000;
        let millis = (total_millis % 1000) as u32;
        let total_secs = total_millis / 1000;
        let secs = (total_secs % 60) as u32;
        let total_mins = total_secs / 60;
        let mins = (total_mins % 60) as u32;
        let hours = (total_mins / 60) as u32;
        (hours, mins, secs, millis)
    }

    /// Zero timestamp
    #[must_use]
    pub const fn zero() -> Self {
        Self(0)
    }

    /// Add a duration to this timestamp
    #[must_use]
    pub const fn add(&self, duration: Duration) -> Self {
        Self(self.0 + duration.0)
    }

    /// Subtract a duration from this timestamp
    #[must_use]
    pub const fn sub(&self, duration: Duration) -> Self {
        Self(self.0 - duration.0)
    }

    /// Calculate the duration between two timestamps
    #[must_use]
    pub const fn duration_since(&self, other: Self) -> Duration {
        Duration(self.0 - other.0)
    }
}

impl fmt::Display for Timestamp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let (h, m, s, ms) = self.as_hmsm();
        write!(f, "{h:02}:{m:02}:{s:02}.{ms:03}")
    }
}

/// Duration in microseconds
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Default,
)]
pub struct Duration(i64);

impl Duration {
    /// Create a duration from microseconds
    #[must_use]
    pub const fn from_micros(micros: i64) -> Self {
        Self(micros)
    }

    /// Create a duration from milliseconds
    #[must_use]
    pub const fn from_millis(millis: i64) -> Self {
        Self(millis * 1000)
    }

    /// Create a duration from seconds
    #[must_use]
    pub const fn from_secs(secs: i64) -> Self {
        Self(secs * 1_000_000)
    }

    /// Get the duration as microseconds
    #[must_use]
    pub const fn as_micros(&self) -> i64 {
        self.0
    }

    /// Get the duration as milliseconds
    #[must_use]
    pub const fn as_millis(&self) -> i64 {
        self.0 / 1000
    }

    /// Get the duration as seconds
    #[must_use]
    pub const fn as_secs(&self) -> i64 {
        self.0 / 1_000_000
    }

    /// Zero duration
    #[must_use]
    pub const fn zero() -> Self {
        Self(0)
    }
}

impl fmt::Display for Duration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let millis = self.as_millis();
        write!(f, "{millis}ms")
    }
}

/// RGB color with alpha channel
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Color {
    /// Red component (0-255)
    pub r: u8,
    /// Green component (0-255)
    pub g: u8,
    /// Blue component (0-255)
    pub b: u8,
    /// Alpha component (0-255, 255 = opaque)
    pub a: u8,
}

impl Color {
    /// Create a new color
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8, a: u8) -> Self {
        Self { r, g, b, a }
    }

    /// Create an opaque color
    #[must_use]
    pub const fn rgb(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b, a: 255 }
    }

    /// White color
    #[must_use]
    pub const fn white() -> Self {
        Self::rgb(255, 255, 255)
    }

    /// Black color
    #[must_use]
    pub const fn black() -> Self {
        Self::rgb(0, 0, 0)
    }

    /// Transparent color
    #[must_use]
    pub const fn transparent() -> Self {
        Self::new(0, 0, 0, 0)
    }

    /// Parse a hex color string (#RRGGBB or #RRGGBBAA)
    pub fn from_hex(hex: &str) -> Result<Self> {
        let hex = hex.trim_start_matches('#');
        let len = hex.len();

        if len != 6 && len != 8 {
            return Err(CaptionError::InvalidColor(format!(
                "Invalid hex color length: {len}"
            )));
        }

        let r = u8::from_str_radix(&hex[0..2], 16)
            .map_err(|e| CaptionError::InvalidColor(e.to_string()))?;
        let g = u8::from_str_radix(&hex[2..4], 16)
            .map_err(|e| CaptionError::InvalidColor(e.to_string()))?;
        let b = u8::from_str_radix(&hex[4..6], 16)
            .map_err(|e| CaptionError::InvalidColor(e.to_string()))?;
        let a = if len == 8 {
            u8::from_str_radix(&hex[6..8], 16)
                .map_err(|e| CaptionError::InvalidColor(e.to_string()))?
        } else {
            255
        };

        Ok(Self { r, g, b, a })
    }

    /// Convert to hex string
    #[must_use]
    pub fn to_hex(&self) -> String {
        if self.a == 255 {
            format!("#{:02X}{:02X}{:02X}", self.r, self.g, self.b)
        } else {
            format!("#{:02X}{:02X}{:02X}{:02X}", self.r, self.g, self.b, self.a)
        }
    }

    /// Calculate luminance (for WCAG contrast ratio)
    #[must_use]
    pub fn luminance(&self) -> f64 {
        let r = f64::from(self.r) / 255.0;
        let g = f64::from(self.g) / 255.0;
        let b = f64::from(self.b) / 255.0;

        let r = if r <= 0.03928 {
            r / 12.92
        } else {
            ((r + 0.055) / 1.055).powf(2.4)
        };
        let g = if g <= 0.03928 {
            g / 12.92
        } else {
            ((g + 0.055) / 1.055).powf(2.4)
        };
        let b = if b <= 0.03928 {
            b / 12.92
        } else {
            ((b + 0.055) / 1.055).powf(2.4)
        };

        0.2126 * r + 0.7152 * g + 0.0722 * b
    }

    /// Calculate contrast ratio with another color (WCAG 2.1)
    #[must_use]
    pub fn contrast_ratio(&self, other: &Self) -> f64 {
        let l1 = self.luminance();
        let l2 = other.luminance();
        let (lighter, darker) = if l1 > l2 { (l1, l2) } else { (l2, l1) };
        (lighter + 0.05) / (darker + 0.05)
    }
}

impl fmt::Display for Color {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

/// Text alignment
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Alignment {
    /// Left-aligned
    Left,
    /// Center-aligned
    #[default]
    Center,
    /// Right-aligned
    Right,
    /// Justified
    Justified,
}

/// Vertical position
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum VerticalPosition {
    /// Top of frame
    Top,
    /// Middle of frame
    Middle,
    /// Bottom of frame
    #[default]
    Bottom,
    /// Custom percentage (0-100)
    Custom(u8),
}

/// Position on screen
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Position {
    /// Vertical position
    pub vertical: VerticalPosition,
    /// Horizontal position (0.0 = left, 0.5 = center, 1.0 = right)
    pub horizontal: f32,
    /// Line number (for CEA-608, 1-15)
    pub line: Option<u8>,
    /// Column number (for CEA-608, 1-32)
    pub column: Option<u8>,
}

impl Default for Position {
    fn default() -> Self {
        Self {
            vertical: VerticalPosition::Bottom,
            horizontal: 0.5,
            line: None,
            column: None,
        }
    }
}

/// Font weight
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FontWeight {
    /// Normal weight
    #[default]
    Normal,
    /// Bold weight
    Bold,
}

/// Font style
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum FontStyle {
    /// Normal style
    #[default]
    Normal,
    /// Italic style
    Italic,
}

/// Text decoration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct TextDecoration {
    /// Underline
    pub underline: bool,
    /// Strikethrough
    pub strikethrough: bool,
}

/// Caption style
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionStyle {
    /// Font family
    pub font_family: String,
    /// Font size in points
    pub font_size: u32,
    /// Font weight
    pub font_weight: FontWeight,
    /// Font style
    pub font_style: FontStyle,
    /// Text decoration
    pub text_decoration: TextDecoration,
    /// Text color
    pub color: Color,
    /// Background color (optional)
    pub background_color: Option<Color>,
    /// Outline color (optional)
    pub outline_color: Option<Color>,
    /// Outline width in pixels
    pub outline_width: u32,
    /// Shadow color (optional)
    pub shadow_color: Option<Color>,
    /// Shadow offset (x, y) in pixels
    pub shadow_offset: (i32, i32),
    /// Text alignment
    pub alignment: Alignment,
}

impl Default for CaptionStyle {
    fn default() -> Self {
        Self {
            font_family: "Arial".to_string(),
            font_size: 32,
            font_weight: FontWeight::Normal,
            font_style: FontStyle::Normal,
            text_decoration: TextDecoration::default(),
            color: Color::white(),
            background_color: Some(Color::new(0, 0, 0, 180)),
            outline_color: Some(Color::black()),
            outline_width: 1,
            shadow_color: Some(Color::new(0, 0, 0, 128)),
            shadow_offset: (2, 2),
            alignment: Alignment::Center,
        }
    }
}

/// Caption effect
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum CaptionEffect {
    /// Pop-on (entire caption appears at once)
    #[default]
    PopOn,
    /// Roll-up (captions scroll up)
    RollUp,
    /// Paint-on (characters appear one at a time)
    PaintOn,
    /// Fade in
    FadeIn,
    /// Fade out
    FadeOut,
}

/// A single caption
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Caption {
    /// Unique identifier
    pub id: CaptionId,
    /// Start timestamp
    pub start: Timestamp,
    /// End timestamp
    pub end: Timestamp,
    /// Caption text
    pub text: String,
    /// Style
    pub style: CaptionStyle,
    /// Position
    pub position: Position,
    /// Effect
    pub effect: CaptionEffect,
    /// Speaker identification (optional)
    pub speaker: Option<String>,
    /// Custom metadata
    pub metadata: HashMap<String, String>,
}

impl Caption {
    /// Create a new caption
    #[must_use]
    pub fn new(start: Timestamp, end: Timestamp, text: String) -> Self {
        Self {
            id: CaptionId::new(),
            start,
            end,
            text,
            style: CaptionStyle::default(),
            position: Position::default(),
            effect: CaptionEffect::default(),
            speaker: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the duration of this caption
    #[must_use]
    pub const fn duration(&self) -> Duration {
        self.end.duration_since(self.start)
    }

    /// Check if this caption overlaps with another
    #[must_use]
    pub const fn overlaps(&self, other: &Self) -> bool {
        !(self.end.0 <= other.start.0 || self.start.0 >= other.end.0)
    }

    /// Count characters (excluding whitespace)
    #[must_use]
    pub fn character_count(&self) -> usize {
        self.text.chars().filter(|c| !c.is_whitespace()).count()
    }

    /// Count words
    #[must_use]
    pub fn word_count(&self) -> usize {
        self.text.split_whitespace().count()
    }

    /// Calculate reading speed in words per minute
    #[must_use]
    pub fn reading_speed_wpm(&self) -> f64 {
        let words = self.word_count() as f64;
        let duration_secs = self.duration().as_secs() as f64;
        if duration_secs == 0.0 {
            0.0
        } else {
            (words / duration_secs) * 60.0
        }
    }

    /// Count lines
    #[must_use]
    pub fn line_count(&self) -> usize {
        self.text.lines().count()
    }

    /// Get the maximum characters per line
    #[must_use]
    pub fn max_chars_per_line(&self) -> usize {
        self.text
            .lines()
            .map(|line| line.chars().count())
            .max()
            .unwrap_or(0)
    }
}

/// Language code (ISO 639-1/2/3)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Language {
    /// ISO 639-1 code (2 letters)
    pub code: String,
    /// Language name
    pub name: String,
    /// Right-to-left script
    pub rtl: bool,
}

impl Language {
    /// Create a new language
    #[must_use]
    pub fn new(code: String, name: String, rtl: bool) -> Self {
        Self { code, name, rtl }
    }

    /// English
    #[must_use]
    pub fn english() -> Self {
        Self::new("en".to_string(), "English".to_string(), false)
    }

    /// Spanish
    #[must_use]
    pub fn spanish() -> Self {
        Self::new("es".to_string(), "Spanish".to_string(), false)
    }

    /// French
    #[must_use]
    pub fn french() -> Self {
        Self::new("fr".to_string(), "French".to_string(), false)
    }

    /// German
    #[must_use]
    pub fn german() -> Self {
        Self::new("de".to_string(), "German".to_string(), false)
    }

    /// Japanese
    #[must_use]
    pub fn japanese() -> Self {
        Self::new("ja".to_string(), "Japanese".to_string(), false)
    }

    /// Arabic
    #[must_use]
    pub fn arabic() -> Self {
        Self::new("ar".to_string(), "Arabic".to_string(), true)
    }
}

/// Metadata for a caption track
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Metadata {
    /// Title
    pub title: Option<String>,
    /// Description
    pub description: Option<String>,
    /// Author
    pub author: Option<String>,
    /// Creation date
    pub created: Option<chrono::DateTime<chrono::Utc>>,
    /// Last modified date
    pub modified: Option<chrono::DateTime<chrono::Utc>>,
    /// Copyright notice
    pub copyright: Option<String>,
    /// Custom metadata
    pub custom: HashMap<String, String>,
}

impl Default for Metadata {
    fn default() -> Self {
        Self {
            title: None,
            description: None,
            author: None,
            created: Some(chrono::Utc::now()),
            modified: Some(chrono::Utc::now()),
            copyright: None,
            custom: HashMap::new(),
        }
    }
}

/// A track of captions
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CaptionTrack {
    /// Language
    pub language: Language,
    /// Captions (sorted by start time)
    pub captions: Vec<Caption>,
    /// Metadata
    pub metadata: Metadata,
}

impl CaptionTrack {
    /// Create a new caption track
    #[must_use]
    pub fn new(language: Language) -> Self {
        Self {
            language,
            captions: Vec::new(),
            metadata: Metadata::default(),
        }
    }

    /// Add a caption to the track
    pub fn add_caption(&mut self, caption: Caption) -> Result<()> {
        // Insert in sorted order by start time
        let pos = self
            .captions
            .binary_search_by_key(&caption.start, |c| c.start)
            .unwrap_or_else(|e| e);
        self.captions.insert(pos, caption);
        self.metadata.modified = Some(chrono::Utc::now());
        Ok(())
    }

    /// Remove a caption by ID
    pub fn remove_caption(&mut self, id: CaptionId) -> Result<()> {
        if let Some(pos) = self.captions.iter().position(|c| c.id == id) {
            self.captions.remove(pos);
            self.metadata.modified = Some(chrono::Utc::now());
            Ok(())
        } else {
            Err(CaptionError::CaptionNotFound(id.to_string()))
        }
    }

    /// Get a caption by ID
    #[must_use]
    pub fn get_caption(&self, id: CaptionId) -> Option<&Caption> {
        self.captions.iter().find(|c| c.id == id)
    }

    /// Get a mutable caption by ID
    pub fn get_caption_mut(&mut self, id: CaptionId) -> Option<&mut Caption> {
        self.captions.iter_mut().find(|c| c.id == id)
    }

    /// Get all captions in a time range
    #[must_use]
    pub fn get_captions_in_range(&self, start: Timestamp, end: Timestamp) -> Vec<&Caption> {
        self.captions
            .iter()
            .filter(|c| !(c.end <= start || c.start >= end))
            .collect()
    }

    /// Sort captions by start time
    pub fn sort(&mut self) {
        self.captions.sort_by_key(|c| c.start);
    }

    /// Count total words
    #[must_use]
    pub fn total_words(&self) -> usize {
        self.captions.iter().map(Caption::word_count).sum()
    }

    /// Get total duration of all captions
    #[must_use]
    pub fn total_duration(&self) -> Duration {
        self.captions
            .iter()
            .map(|c| c.duration().as_micros())
            .sum::<i64>()
            .into()
    }

    /// Count captions
    #[must_use]
    pub fn count(&self) -> usize {
        self.captions.len()
    }
}

impl From<i64> for Duration {
    fn from(micros: i64) -> Self {
        Self(micros)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timestamp_conversions() {
        let ts = Timestamp::from_hmsm(1, 30, 45, 500);
        assert_eq!(ts.as_hmsm(), (1, 30, 45, 500));
        assert_eq!(ts.as_secs(), 5445);
    }

    #[test]
    fn test_color_parsing() {
        let color = Color::from_hex("#FF0000").expect("hex color parsing should succeed");
        assert_eq!(color, Color::rgb(255, 0, 0));

        let color = Color::from_hex("#00FF00AA").expect("hex color parsing should succeed");
        assert_eq!(color, Color::new(0, 255, 0, 170));
    }

    #[test]
    fn test_color_contrast() {
        let white = Color::white();
        let black = Color::black();
        let ratio = white.contrast_ratio(&black);
        assert!((ratio - 21.0).abs() < 0.1); // Should be 21:1
    }

    #[test]
    fn test_caption_creation() {
        let start = Timestamp::from_secs(10);
        let end = Timestamp::from_secs(15);
        let caption = Caption::new(start, end, "Test caption".to_string());

        assert_eq!(caption.duration().as_secs(), 5);
        assert_eq!(caption.word_count(), 2);
    }

    #[test]
    fn test_caption_overlap() {
        let cap1 = Caption::new(
            Timestamp::from_secs(10),
            Timestamp::from_secs(15),
            "Caption 1".to_string(),
        );
        let cap2 = Caption::new(
            Timestamp::from_secs(12),
            Timestamp::from_secs(17),
            "Caption 2".to_string(),
        );
        assert!(cap1.overlaps(&cap2));
    }

    #[test]
    fn test_reading_speed() {
        let caption = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(10),
            "This is a test caption with ten words here".to_string(),
        );
        let wpm = caption.reading_speed_wpm();
        assert!((wpm - 54.0).abs() < 1.0); // 9 words in 10 seconds = 54 WPM
    }

    #[test]
    fn test_caption_track() {
        let mut track = CaptionTrack::new(Language::english());
        let cap1 = Caption::new(
            Timestamp::from_secs(10),
            Timestamp::from_secs(15),
            "First".to_string(),
        );
        let cap2 = Caption::new(
            Timestamp::from_secs(5),
            Timestamp::from_secs(8),
            "Second".to_string(),
        );

        track
            .add_caption(cap1)
            .expect("adding caption should succeed");
        track
            .add_caption(cap2)
            .expect("adding caption should succeed");

        assert_eq!(track.count(), 2);
        // Should be sorted by start time
        assert_eq!(track.captions[0].text, "Second");
        assert_eq!(track.captions[1].text, "First");
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    #[test]
    fn test_timestamp_arithmetic() {
        let ts1 = Timestamp::from_secs(10);
        let ts2 = Timestamp::from_secs(5);
        let duration = ts1.duration_since(ts2);
        assert_eq!(duration.as_secs(), 5);

        let ts3 = ts1.add(Duration::from_secs(5));
        assert_eq!(ts3.as_secs(), 15);

        let ts4 = ts1.sub(Duration::from_secs(3));
        assert_eq!(ts4.as_secs(), 7);
    }

    #[test]
    fn test_color_operations() {
        let red = Color::rgb(255, 0, 0);
        let green = Color::rgb(0, 255, 0);
        let blue = Color::rgb(0, 0, 255);

        assert_ne!(red, green);
        assert_ne!(green, blue);

        let hex = red.to_hex();
        assert_eq!(hex, "#FF0000");

        let parsed = Color::from_hex(&hex).expect("hex color parsing should succeed");
        assert_eq!(parsed, red);
    }

    #[test]
    fn test_color_hex_with_alpha() {
        let color = Color::new(255, 128, 64, 200);
        let hex = color.to_hex();
        assert_eq!(hex, "#FF8040C8");

        let parsed = Color::from_hex(&hex).expect("hex color parsing should succeed");
        assert_eq!(parsed, color);
    }

    #[test]
    fn test_caption_timing_edge_cases() {
        let cap = Caption::new(
            Timestamp::zero(),
            Timestamp::from_millis(1),
            "Test".to_string(),
        );

        assert_eq!(cap.duration().as_millis(), 1);
    }

    #[test]
    fn test_caption_metadata() {
        let mut cap = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "Test".to_string(),
        );

        cap.metadata
            .insert("custom".to_string(), "value".to_string());
        assert_eq!(cap.metadata.get("custom"), Some(&"value".to_string()));
    }

    #[test]
    fn test_caption_style_defaults() {
        let style = CaptionStyle::default();
        assert_eq!(style.font_family, "Arial");
        assert_eq!(style.font_size, 32);
        assert_eq!(style.alignment, Alignment::Center);
    }

    #[test]
    fn test_position_custom_vertical() {
        let pos = Position {
            vertical: VerticalPosition::Custom(75),
            horizontal: 0.5,
            line: None,
            column: None,
        };

        match pos.vertical {
            VerticalPosition::Custom(percent) => assert_eq!(percent, 75),
            _ => panic!("Expected Custom variant"),
        }
    }

    #[test]
    fn test_language_rtl() {
        let arabic = Language::arabic();
        assert!(arabic.rtl);

        let english = Language::english();
        assert!(!english.rtl);
    }

    #[test]
    fn test_track_operations() {
        let mut track = CaptionTrack::new(Language::english());

        let cap1 = Caption::new(
            Timestamp::from_secs(5),
            Timestamp::from_secs(10),
            "Second".to_string(),
        );
        let cap2 = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(3),
            "First".to_string(),
        );

        track
            .add_caption(cap1)
            .expect("adding caption should succeed");
        track
            .add_caption(cap2)
            .expect("adding caption should succeed");

        // Should be sorted by start time
        assert_eq!(track.captions[0].text, "First");
        assert_eq!(track.captions[1].text, "Second");
    }

    #[test]
    fn test_caption_word_metrics() {
        let cap = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(10),
            "one two three four five".to_string(),
        );

        assert_eq!(cap.word_count(), 5);
        assert_eq!(cap.line_count(), 1);
    }

    #[test]
    fn test_caption_multiline_metrics() {
        let cap = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(10),
            "Line one\nLine two\nLine three".to_string(),
        );

        assert_eq!(cap.line_count(), 3);
        assert_eq!(cap.max_chars_per_line(), 10); // "Line three" is 10 chars, longest
    }

    #[test]
    fn test_duration_conversions() {
        let dur = Duration::from_secs(1);
        assert_eq!(dur.as_millis(), 1000);
        assert_eq!(dur.as_micros(), 1_000_000);

        let dur2 = Duration::from_millis(500);
        assert_eq!(dur2.as_secs(), 0);
        assert_eq!(dur2.as_millis(), 500);
    }

    #[test]
    fn test_caption_overlaps_edge_cases() {
        let cap1 = Caption::new(
            Timestamp::from_secs(0),
            Timestamp::from_secs(5),
            "First".to_string(),
        );

        let cap2 = Caption::new(
            Timestamp::from_secs(5),
            Timestamp::from_secs(10),
            "Second".to_string(),
        );

        // Adjacent captions should not overlap
        assert!(!cap1.overlaps(&cap2));
    }

    #[test]
    fn test_track_get_captions_in_range() {
        let mut track = CaptionTrack::new(Language::english());

        track
            .add_caption(Caption::new(
                Timestamp::from_secs(0),
                Timestamp::from_secs(5),
                "First".to_string(),
            ))
            .expect("operation should succeed in test");

        track
            .add_caption(Caption::new(
                Timestamp::from_secs(10),
                Timestamp::from_secs(15),
                "Second".to_string(),
            ))
            .expect("operation should succeed in test");

        track
            .add_caption(Caption::new(
                Timestamp::from_secs(20),
                Timestamp::from_secs(25),
                "Third".to_string(),
            ))
            .expect("operation should succeed in test");

        let in_range =
            track.get_captions_in_range(Timestamp::from_secs(8), Timestamp::from_secs(22));

        assert_eq!(in_range.len(), 2);
    }

    #[test]
    fn test_metadata_timestamps() {
        let meta = Metadata::default();
        assert!(meta.created.is_some());
        assert!(meta.modified.is_some());
    }

    #[test]
    fn test_color_luminance_calculation() {
        let white = Color::white();
        let black = Color::black();

        let white_lum = white.luminance();
        let black_lum = black.luminance();

        assert!(white_lum > black_lum);
        assert!(white_lum > 0.9);
        assert!(black_lum < 0.1);
    }

    #[test]
    fn test_text_decoration() {
        let mut decor = TextDecoration::default();
        assert!(!decor.underline);
        assert!(!decor.strikethrough);

        decor.underline = true;
        assert!(decor.underline);
    }

    #[test]
    fn test_font_weight_style() {
        let weight = FontWeight::Bold;
        assert_eq!(weight, FontWeight::Bold);
        assert_ne!(weight, FontWeight::Normal);

        let style = FontStyle::Italic;
        assert_eq!(style, FontStyle::Italic);
        assert_ne!(style, FontStyle::Normal);
    }
}
