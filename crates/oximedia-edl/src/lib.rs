//! OxiMedia EDL - CMX 3600 Edit Decision List parser and generator.
//!
//! This crate provides comprehensive support for EDL (Edit Decision List) files,
//! with a focus on the CMX 3600 format and related formats.
//!
//! # Features
//!
//! - CMX 3600, CMX 3400, GVG, and Sony BVE-9000 format support
//! - Event types: Cut, Dissolve, Wipe, Key
//! - Timecode support: Drop-frame, Non-drop-frame (24, 25, 30, 60 fps)
//! - Reel names and source references
//! - Motion effects (speed changes, reverse playback, freeze frames)
//! - Audio channel mapping and routing
//! - EDL validation and compliance checking
//! - Format conversion and optimization
//!
//! # Example: Parsing an EDL
//!
//! ```
//! use oximedia_edl::{parse_edl, Edl};
//!
//! let edl_text = r#"
//! TITLE: Example EDL
//! FCM: DROP FRAME
//!
//! 001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00
//! * FROM CLIP NAME: shot001.mov
//! "#;
//!
//! let edl = parse_edl(edl_text)?;
//! assert_eq!(edl.title, Some("Example EDL".to_string()));
//! assert_eq!(edl.events.len(), 1);
//! ```
//!
//! # Example: Generating an EDL
//!
//! ```
//! use oximedia_edl::{Edl, EdlFormat, EdlGenerator};
//! use oximedia_edl::event::{EdlEvent, EditType, TrackType};
//! use oximedia_edl::timecode::{EdlFrameRate, EdlTimecode};
//!
//! let mut edl = Edl::new(EdlFormat::Cmx3600);
//! edl.set_title("My EDL".to_string());
//! edl.set_frame_rate(EdlFrameRate::Fps25);
//!
//! let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25)?;
//! let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25)?;
//!
//! let event = EdlEvent::new(
//!     1,
//!     "A001".to_string(),
//!     TrackType::Video,
//!     EditType::Cut,
//!     tc1,
//!     tc2,
//!     tc1,
//!     tc2,
//! );
//!
//! edl.add_event(event)?;
//!
//! let generator = EdlGenerator::new();
//! let output = generator.generate(&edl)?;
//! assert!(output.contains("TITLE: My EDL"));
//! ```
//!
//! # Example: Validating an EDL
//!
//! ```
//! use oximedia_edl::{Edl, EdlFormat, EdlValidator};
//! use oximedia_edl::validator::ValidationLevel;
//!
//! let edl = Edl::new(EdlFormat::Cmx3600);
//! let validator = EdlValidator::strict();
//! let report = validator.validate(&edl)?;
//! ```

#![warn(missing_docs)]
#![allow(
    clippy::module_name_repetitions,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    dead_code,
    clippy::pedantic
)]

pub mod audio;
pub mod batch_export;
pub mod cmx3600;
pub mod conform_report;
pub mod consolidate;
pub mod converter;
pub mod edl_comments;
pub mod edl_compare;
pub mod edl_event;
pub mod edl_filter;
pub mod edl_merge;
pub mod edl_statistics;
pub mod edl_timeline;
pub mod edl_validator;
pub mod error;
pub mod event;
pub mod event_list;
pub mod frame_count;
pub mod generator;
pub mod metadata;
pub mod motion;
pub mod optimizer;
pub mod parser;
pub mod reel;
pub mod reel_map;
pub mod reel_registry;
pub mod roundtrip;
pub mod timecode;
pub mod transition_events;
pub mod validator;

pub use error::{EdlError, EdlResult};
pub use generator::EdlGenerator;
pub use parser::{parse_edl, EdlParser};
pub use validator::EdlValidator;

use crate::event::EdlEvent;
use crate::reel::ReelTable;
use crate::timecode::EdlFrameRate;
use std::path::PathBuf;

/// EDL format identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EdlFormat {
    /// CMX 3600 format (most common).
    Cmx3600,
    /// CMX 3400 format (older).
    Cmx3400,
    /// CMX 340 format.
    Cmx340,
    /// GVG (Grass Valley Group) format.
    Gvg,
    /// Sony BVE-9000 format.
    SonyBve9000,
}

impl EdlFormat {
    /// Get the format name as a string.
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Cmx3600 => "CMX 3600",
            Self::Cmx3400 => "CMX 3400",
            Self::Cmx340 => "CMX 340",
            Self::Gvg => "GVG",
            Self::SonyBve9000 => "Sony BVE-9000",
        }
    }
}

impl std::fmt::Display for EdlFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Main EDL structure containing all events and metadata.
#[derive(Debug, Clone)]
pub struct Edl {
    /// EDL format.
    pub format: EdlFormat,

    /// Optional title of the EDL.
    pub title: Option<String>,

    /// Frame rate for timecodes.
    pub frame_rate: EdlFrameRate,

    /// List of events in the EDL.
    pub events: Vec<EdlEvent>,

    /// Reel table with source information.
    pub reel_table: ReelTable,

    /// Optional source file path (for reference).
    pub source_file: Option<PathBuf>,

    /// Optional comments not associated with specific events.
    pub global_comments: Vec<String>,
}

impl Edl {
    /// Create a new empty EDL with the specified format.
    #[must_use]
    pub fn new(format: EdlFormat) -> Self {
        Self {
            format,
            title: None,
            frame_rate: EdlFrameRate::Fps2997NDF,
            events: Vec::new(),
            reel_table: ReelTable::new(),
            source_file: None,
            global_comments: Vec::new(),
        }
    }

    /// Create a new CMX 3600 EDL (most common format).
    #[must_use]
    pub fn cmx3600() -> Self {
        Self::new(EdlFormat::Cmx3600)
    }

    /// Set the EDL title.
    pub fn set_title(&mut self, title: String) {
        self.title = Some(title);
    }

    /// Set the frame rate.
    pub fn set_frame_rate(&mut self, frame_rate: EdlFrameRate) {
        self.frame_rate = frame_rate;
    }

    /// Set the source file path.
    pub fn set_source_file(&mut self, path: PathBuf) {
        self.source_file = Some(path);
    }

    /// Add an event to the EDL.
    ///
    /// # Errors
    ///
    /// Returns an error if the event is invalid.
    pub fn add_event(&mut self, event: EdlEvent) -> EdlResult<()> {
        event.validate()?;
        self.events.push(event);
        Ok(())
    }

    /// Add a global comment.
    pub fn add_global_comment(&mut self, comment: String) {
        self.global_comments.push(comment);
    }

    /// Get an event by number.
    #[must_use]
    pub fn get_event(&self, number: u32) -> Option<&EdlEvent> {
        self.events.iter().find(|e| e.number == number)
    }

    /// Get a mutable event by number.
    pub fn get_event_mut(&mut self, number: u32) -> Option<&mut EdlEvent> {
        self.events.iter_mut().find(|e| e.number == number)
    }

    /// Remove an event by number.
    pub fn remove_event(&mut self, number: u32) -> Option<EdlEvent> {
        if let Some(index) = self.events.iter().position(|e| e.number == number) {
            Some(self.events.remove(index))
        } else {
            None
        }
    }

    /// Get the number of events.
    #[must_use]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Get the total duration of the EDL in frames.
    #[must_use]
    pub fn total_duration_frames(&self) -> u64 {
        self.events.iter().map(|e| e.duration_frames()).sum()
    }

    /// Get the total duration in seconds.
    #[must_use]
    pub fn total_duration_seconds(&self) -> f64 {
        self.total_duration_frames() as f64 / self.frame_rate.fps() as f64
    }

    /// Sort events by record in timecode.
    pub fn sort_events(&mut self) {
        self.events.sort_by_key(|e| e.record_in.to_frames());
    }

    /// Renumber events sequentially starting from 1.
    pub fn renumber_events(&mut self) {
        for (i, event) in self.events.iter_mut().enumerate() {
            event.number = (i + 1) as u32;
        }
    }

    /// Validate the entire EDL.
    ///
    /// # Errors
    ///
    /// Returns an error if the EDL is invalid.
    pub fn validate(&self) -> EdlResult<()> {
        let validator = EdlValidator::default();
        validator.validate(self)?;
        Ok(())
    }

    /// Generate the EDL as a string.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn to_string_format(&self) -> EdlResult<String> {
        let generator = EdlGenerator::new();
        generator.generate(self)
    }

    /// Parse an EDL from a string.
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(input: &str) -> EdlResult<Self> {
        parse_edl(input)
    }

    /// Load an EDL from a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_file(path: &std::path::Path) -> EdlResult<Self> {
        let content = std::fs::read_to_string(path)?;
        let mut edl = parse_edl(&content)?;
        edl.set_source_file(path.to_path_buf());
        Ok(edl)
    }

    /// Save the EDL to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn to_file(&self, path: &std::path::Path) -> EdlResult<()> {
        let generator = EdlGenerator::new();
        generator.generate_to_file(self, path)
    }
}

impl Default for Edl {
    fn default() -> Self {
        Self::cmx3600()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::EdlTimecode;

    #[test]
    fn test_create_edl() {
        let edl = Edl::new(EdlFormat::Cmx3600);
        assert_eq!(edl.format, EdlFormat::Cmx3600);
        assert_eq!(edl.events.len(), 0);
    }

    #[test]
    fn test_add_event() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event).expect("add_event should succeed");
        assert_eq!(edl.events.len(), 1);
    }

    #[test]
    fn test_get_event() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event).expect("add_event should succeed");

        let retrieved = edl.get_event(1).expect("get_event should succeed");
        assert_eq!(retrieved.number, 1);
    }

    #[test]
    fn test_remove_event() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event).expect("add_event should succeed");
        assert_eq!(edl.events.len(), 1);

        edl.remove_event(1);
        assert_eq!(edl.events.len(), 0);
    }

    #[test]
    fn test_renumber_events() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event1 = EdlEvent::new(
            10,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        let event2 = EdlEvent::new(
            20,
            "A002".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event1).expect("add_event should succeed");
        edl.add_event(event2).expect("add_event should succeed");

        edl.renumber_events();

        assert_eq!(edl.events[0].number, 1);
        assert_eq!(edl.events[1].number, 2);
    }

    #[test]
    fn test_total_duration() {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event).expect("add_event should succeed");

        let duration_frames = edl.total_duration_frames();
        assert_eq!(duration_frames, 125); // 5 seconds * 25 fps

        let duration_seconds = edl.total_duration_seconds();
        assert!((duration_seconds - 5.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_edl_format_display() {
        assert_eq!(EdlFormat::Cmx3600.to_string(), "CMX 3600");
        assert_eq!(EdlFormat::Cmx3400.to_string(), "CMX 3400");
        assert_eq!(EdlFormat::Gvg.to_string(), "GVG");
    }

    #[test]
    fn test_parse_and_generate_roundtrip() {
        let edl_text = r#"TITLE: Test EDL
FCM: NON-DROP FRAME

001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00

"#;

        let edl = parse_edl(edl_text).expect("operation should succeed");
        let generated = edl.to_string_format().expect("formatting should succeed");

        // Parse the generated EDL again
        let edl2 = parse_edl(&generated).expect("operation should succeed");

        assert_eq!(edl.title, edl2.title);
        assert_eq!(edl.events.len(), edl2.events.len());
        assert_eq!(edl.events[0].number, edl2.events[0].number);
    }
}
