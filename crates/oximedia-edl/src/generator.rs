//! EDL generator for creating EDL files from event data.
//!
//! This module provides functionality to generate CMX 3600 EDL files
//! and related formats from EDL structures.

use crate::error::{EdlError, EdlResult};
use crate::event::EdlEvent;
use crate::Edl;
use std::fmt::Write as FmtWrite;

/// EDL generator for creating EDL file content.
#[derive(Debug)]
pub struct EdlGenerator {
    /// Include comments in output.
    pub include_comments: bool,
    /// Include clip names in output.
    pub include_clip_names: bool,
    /// Include motion effects in output.
    pub include_motion_effects: bool,
    /// Indent size for formatting.
    pub indent_size: usize,
}

impl EdlGenerator {
    /// Create a new EDL generator with default settings.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            include_comments: true,
            include_clip_names: true,
            include_motion_effects: true,
            indent_size: 0,
        }
    }

    /// Create a minimal EDL generator (no comments, clip names, or motion effects).
    #[must_use]
    pub const fn minimal() -> Self {
        Self {
            include_comments: false,
            include_clip_names: false,
            include_motion_effects: false,
            indent_size: 0,
        }
    }

    /// Set whether to include comments.
    pub fn set_include_comments(&mut self, include: bool) {
        self.include_comments = include;
    }

    /// Set whether to include clip names.
    pub fn set_include_clip_names(&mut self, include: bool) {
        self.include_clip_names = include;
    }

    /// Set whether to include motion effects.
    pub fn set_include_motion_effects(&mut self, include: bool) {
        self.include_motion_effects = include;
    }

    /// Generate an EDL file as a string.
    ///
    /// # Errors
    ///
    /// Returns an error if the EDL cannot be generated.
    pub fn generate(&self, edl: &Edl) -> EdlResult<String> {
        let mut output = String::new();

        // Write header
        self.write_header(&mut output, edl)?;

        // Write events
        for event in &edl.events {
            self.write_event(&mut output, event)?;
        }

        Ok(output)
    }

    /// Write the EDL header.
    fn write_header(&self, output: &mut String, edl: &Edl) -> EdlResult<()> {
        // Write title
        if let Some(title) = &edl.title {
            writeln!(output, "TITLE: {title}")
                .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
        }

        // Write frame count mode
        writeln!(output, "FCM: {}", edl.frame_rate.fcm_string())
            .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

        // Blank line after header
        writeln!(output).map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

        Ok(())
    }

    /// Write a single event.
    fn write_event(&self, output: &mut String, event: &EdlEvent) -> EdlResult<()> {
        // Format event number (3 digits, zero-padded)
        write!(output, "{:03}  ", event.number)
            .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

        // Format reel name (8 characters, left-aligned)
        write!(output, "{:<8} ", event.reel)
            .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

        // Format track type (5 characters, left-aligned)
        write!(output, "{:<5} ", event.track)
            .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

        // Format edit type
        write!(output, "{}", event.edit_type)
            .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

        // Format transition duration if present
        if let Some(duration) = event.transition_duration {
            write!(output, " {:>4}", duration)
                .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
        } else {
            write!(output, "     ")
                .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
        }

        // Format timecodes
        writeln!(
            output,
            " {} {} {} {}",
            event.source_in, event.source_out, event.record_in, event.record_out
        )
        .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

        // Write clip name comment if enabled and present
        if self.include_clip_names {
            if let Some(clip_name) = &event.clip_name {
                writeln!(output, "* FROM CLIP NAME: {clip_name}")
                    .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
            }
        }

        // Write motion effect comment if enabled and present
        if self.include_motion_effects {
            if let Some(motion) = &event.motion_effect {
                let comment = motion.to_m2_comment(&event.reel, event.source_in.frames() as u32);
                writeln!(output, "* {comment}")
                    .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
            }
        }

        // Write wipe pattern if present
        if let Some(wipe) = &event.wipe_pattern {
            writeln!(output, "* WIPE: {wipe}")
                .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
        }

        // Write key type if present
        if let Some(key) = &event.key_type {
            writeln!(output, "* KEY TYPE: {key}")
                .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
        }

        // Write additional comments if enabled
        if self.include_comments {
            for comment in &event.comments {
                writeln!(output, "* {comment}")
                    .map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;
            }
        }

        // Blank line after event
        writeln!(output).map_err(|e| EdlError::ValidationError(format!("Write error: {e}")))?;

        Ok(())
    }

    /// Generate an EDL and write to a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be written.
    pub fn generate_to_file(&self, edl: &Edl, path: &std::path::Path) -> EdlResult<()> {
        let content = self.generate(edl)?;
        std::fs::write(path, content)?;
        Ok(())
    }
}

impl Default for EdlGenerator {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for creating EDL generators with custom settings.
#[derive(Debug, Default)]
pub struct EdlGeneratorBuilder {
    include_comments: bool,
    include_clip_names: bool,
    include_motion_effects: bool,
    indent_size: usize,
}

impl EdlGeneratorBuilder {
    /// Create a new EDL generator builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            include_comments: true,
            include_clip_names: true,
            include_motion_effects: true,
            indent_size: 0,
        }
    }

    /// Set whether to include comments.
    #[must_use]
    pub const fn include_comments(mut self, include: bool) -> Self {
        self.include_comments = include;
        self
    }

    /// Set whether to include clip names.
    #[must_use]
    pub const fn include_clip_names(mut self, include: bool) -> Self {
        self.include_clip_names = include;
        self
    }

    /// Set whether to include motion effects.
    #[must_use]
    pub const fn include_motion_effects(mut self, include: bool) -> Self {
        self.include_motion_effects = include;
        self
    }

    /// Set the indent size.
    #[must_use]
    pub const fn indent_size(mut self, size: usize) -> Self {
        self.indent_size = size;
        self
    }

    /// Build the EDL generator.
    #[must_use]
    pub const fn build(self) -> EdlGenerator {
        EdlGenerator {
            include_comments: self.include_comments,
            include_clip_names: self.include_clip_names,
            include_motion_effects: self.include_motion_effects,
            indent_size: self.indent_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, EdlEvent, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};

    #[test]
    fn test_generate_simple_edl() {
        let mut edl = Edl::new(crate::EdlFormat::Cmx3600);
        edl.set_title("Test EDL".to_string());
        edl.set_frame_rate(EdlFrameRate::Fps2997DF);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps2997DF).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps2997DF).expect("failed to create");

        let event = EdlEvent::new(
            1,
            "AX".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );

        edl.add_event(event).expect("add_event should succeed");

        let generator = EdlGenerator::new();
        let output = generator.generate(&edl).expect("failed to generate");

        assert!(output.contains("TITLE: Test EDL"));
        assert!(output.contains("FCM: DROP FRAME"));
        assert!(output.contains("001"));
        assert!(output.contains("AX"));
    }

    #[test]
    fn test_generate_with_dissolve() {
        let mut edl = Edl::new(crate::EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let mut event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Dissolve,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        event.set_transition_duration(30);

        edl.add_event(event).expect("add_event should succeed");

        let generator = EdlGenerator::new();
        let output = generator.generate(&edl).expect("failed to generate");

        assert!(output.contains("D"));
        assert!(output.contains("30"));
    }

    #[test]
    fn test_generate_with_clip_name() {
        let mut edl = Edl::new(crate::EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let mut event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        event.set_clip_name("test_clip.mov".to_string());

        edl.add_event(event).expect("add_event should succeed");

        let generator = EdlGenerator::new();
        let output = generator.generate(&edl).expect("failed to generate");

        assert!(output.contains("FROM CLIP NAME: test_clip.mov"));
    }

    #[test]
    fn test_minimal_generator() {
        let mut edl = Edl::new(crate::EdlFormat::Cmx3600);
        edl.set_frame_rate(EdlFrameRate::Fps25);

        let tc1 = EdlTimecode::new(1, 0, 0, 0, EdlFrameRate::Fps25).expect("failed to create");
        let tc2 = EdlTimecode::new(1, 0, 5, 0, EdlFrameRate::Fps25).expect("failed to create");

        let mut event = EdlEvent::new(
            1,
            "A001".to_string(),
            TrackType::Video,
            EditType::Cut,
            tc1,
            tc2,
            tc1,
            tc2,
        );
        event.set_clip_name("test_clip.mov".to_string());
        event.add_comment("This is a comment".to_string());

        edl.add_event(event).expect("add_event should succeed");

        let generator = EdlGenerator::minimal();
        let output = generator.generate(&edl).expect("failed to generate");

        assert!(!output.contains("FROM CLIP NAME"));
        assert!(!output.contains("This is a comment"));
    }

    #[test]
    fn test_generator_builder() {
        let generator = EdlGeneratorBuilder::new()
            .include_comments(false)
            .include_clip_names(true)
            .build();

        assert!(!generator.include_comments);
        assert!(generator.include_clip_names);
    }
}
