//! CMX 3600 EDL format parser and writer.
//!
//! The CMX 3600 format is the industry-standard EDL format for linear editing
//! systems. It originated with the CMX 3600 video editing system in the 1970s
//! and remains widely used today.
//!
//! # Format Specification
//!
//! A CMX 3600 EDL consists of:
//! - Title line: `TITLE: [title]`
//! - FCM line: `FCM: [DROP FRAME|NON-DROP FRAME]`
//! - Event entries with the format:
//!   ```text
//!   [event#] [reel] [track] [edit] [source-in] [source-out] [record-in] [record-out]
//!   ```
//!
//! # Example
//!
//! ```text
//! TITLE: My Project
//! FCM: DROP FRAME
//!
//! 001  AX       V     C        01:00:00;00 01:00:05;00 01:00:00;00 01:00:05;00
//! * FROM CLIP NAME: CLIP001.MOV
//! M2   AX       030   01:00:02;15
//! * FREEZE FRAME
//!
//! 002  AX       A     C        01:00:05;00 01:00:10;00 01:00:05;00 01:00:10;00
//! ```

use super::{
    Edl, EdlError, EdlEvent, EdlResult, EditType, MotionEffect, Timecode,
};
use oximedia_core::Rational;
use std::collections::HashMap;

/// CMX 3600 parser.
pub struct Cmx3600Parser {
    frame_rate: Rational,
    drop_frame: bool,
}

impl Cmx3600Parser {
    /// Create a new CMX 3600 parser with default 30fps non-drop-frame.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frame_rate: Rational::new(30, 1),
            drop_frame: false,
        }
    }

    /// Set frame rate.
    #[must_use]
    pub fn with_frame_rate(mut self, frame_rate: Rational) -> Self {
        self.frame_rate = frame_rate;
        self
    }

    /// Parse a CMX 3600 EDL.
    pub fn parse(&mut self, content: &str) -> EdlResult<Edl> {
        let lines: Vec<&str> = content.lines().collect();
        let mut title = String::new();
        let mut events = Vec::new();
        let mut comments = Vec::new();
        let mut metadata = HashMap::new();

        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();

            // Skip empty lines
            if line.is_empty() {
                i += 1;
                continue;
            }

            // Parse title
            if line.starts_with("TITLE:") {
                title = line[6..].trim().to_string();
                i += 1;
                continue;
            }

            // Parse FCM (Frame Code Mode)
            if line.starts_with("FCM:") {
                let fcm = line[4..].trim();
                self.drop_frame = fcm.contains("DROP");
                i += 1;
                continue;
            }

            // Parse comment lines
            if line.starts_with('*') {
                comments.push(line[1..].trim().to_string());
                i += 1;
                continue;
            }

            // Parse event line
            if let Ok(event) = self.parse_event_line(line, &lines, &mut i) {
                events.push(event);
            }

            i += 1;
        }

        Ok(Edl {
            title,
            frame_rate: self.frame_rate,
            drop_frame: self.drop_frame,
            events,
            comments,
            metadata,
        })
    }

    /// Parse a single event line and its associated metadata.
    fn parse_event_line(
        &self,
        line: &str,
        lines: &[&str],
        index: &mut usize,
    ) -> EdlResult<EdlEvent> {
        let parts: Vec<&str> = line.split_whitespace().collect();

        // Basic event line: [event#] [reel] [track] [edit] [src-in] [src-out] [rec-in] [rec-out]
        if parts.len() < 8 {
            return Err(EdlError::ParseError {
                line: *index + 1,
                message: "Invalid event line format".to_string(),
            });
        }

        let number = parts[0].parse().map_err(|_| EdlError::ParseError {
            line: *index + 1,
            message: format!("Invalid event number: {}", parts[0]),
        })?;

        let reel = parts[1].to_string();
        let track = parts[2].to_string();

        let edit_type = match parts[3] {
            "C" => EditType::Cut,
            "D" => EditType::Dissolve,
            "W" => EditType::Wipe,
            "K" | "KB" | "KO" => EditType::Key,
            _ => EditType::Cut,
        };

        let source_in = Timecode::parse(parts[4], self.frame_rate)?;
        let source_out = Timecode::parse(parts[5], self.frame_rate)?;
        let record_in = Timecode::parse(parts[6], self.frame_rate)?;
        let record_out = Timecode::parse(parts[7], self.frame_rate)?;

        // Parse transition duration if present (for dissolves/wipes)
        let transition_duration = if parts.len() > 8 {
            parts[8].parse().ok()
        } else {
            None
        };

        // Look ahead for associated metadata lines
        let mut event_comments = Vec::new();
        let mut event_metadata = HashMap::new();
        let mut motion_effect = None;

        let mut j = *index + 1;
        while j < lines.len() {
            let next_line = lines[j].trim();

            // Stop if we hit another event or empty line
            if next_line.is_empty() || next_line.chars().next().unwrap_or(' ').is_ascii_digit() {
                break;
            }

            // Comment line
            if next_line.starts_with('*') {
                let comment = next_line[1..].trim().to_string();
                event_comments.push(comment.clone());

                // Check for special metadata in comments
                if comment.starts_with("FROM CLIP NAME:") {
                    event_metadata.insert(
                        "clip_name".to_string(),
                        comment[15..].trim().to_string(),
                    );
                } else if comment.starts_with("TO CLIP NAME:") {
                    event_metadata.insert(
                        "to_clip_name".to_string(),
                        comment[13..].trim().to_string(),
                    );
                } else if comment.contains("FREEZE FRAME") {
                    motion_effect = Some(MotionEffect {
                        speed: 0.0,
                        freeze: true,
                        reverse: false,
                        entry: None,
                    });
                } else if comment.contains("REVERSE") {
                    let mut effect = motion_effect.unwrap_or(MotionEffect {
                        speed: 1.0,
                        freeze: false,
                        reverse: false,
                        entry: None,
                    });
                    effect.reverse = true;
                    motion_effect = Some(effect);
                }

                j += 1;
                continue;
            }

            // M2 line (motion effect)
            if next_line.starts_with("M2") {
                motion_effect = Some(self.parse_motion_effect(next_line)?);
                j += 1;
                continue;
            }

            // Other metadata lines
            if next_line.contains(':') {
                let kv: Vec<&str> = next_line.splitn(2, ':').collect();
                if kv.len() == 2 {
                    event_metadata.insert(
                        kv[0].trim().to_lowercase().replace(' ', "_"),
                        kv[1].trim().to_string(),
                    );
                }
            }

            j += 1;
        }

        // Update the outer index to skip processed lines
        *index = j - 1;

        Ok(EdlEvent {
            number,
            reel,
            track,
            edit_type,
            source_in,
            source_out,
            record_in,
            record_out,
            transition_duration,
            motion_effect,
            comments: event_comments,
            metadata: event_metadata,
        })
    }

    /// Parse motion effect line (M2).
    fn parse_motion_effect(&self, line: &str) -> EdlResult<MotionEffect> {
        let parts: Vec<&str> = line.split_whitespace().collect();

        if parts.len() < 3 {
            return Err(EdlError::ParseError {
                line: 0,
                message: "Invalid M2 line format".to_string(),
            });
        }

        // M2 format: M2 [reel] [speed] [entry-tc]
        let speed_str = parts[2];
        let speed = if speed_str.contains('.') {
            speed_str.parse().unwrap_or(1.0)
        } else {
            // Frame count format: speed = 100 * frames / duration
            speed_str.parse::<f64>().unwrap_or(100.0) / 100.0
        };

        let entry = if parts.len() > 3 {
            Some(Timecode::parse(parts[3], self.frame_rate)?)
        } else {
            None
        };

        Ok(MotionEffect {
            speed,
            freeze: speed == 0.0,
            reverse: speed < 0.0,
            entry,
        })
    }
}

impl Default for Cmx3600Parser {
    fn default() -> Self {
        Self::new()
    }
}

/// CMX 3600 writer.
pub struct Cmx3600Writer {
    include_comments: bool,
    include_metadata: bool,
}

impl Cmx3600Writer {
    /// Create a new CMX 3600 writer.
    #[must_use]
    pub fn new() -> Self {
        Self {
            include_comments: true,
            include_metadata: true,
        }
    }

    /// Set whether to include comments.
    #[must_use]
    pub fn with_comments(mut self, include: bool) -> Self {
        self.include_comments = include;
        self
    }

    /// Set whether to include metadata.
    #[must_use]
    pub fn with_metadata(mut self, include: bool) -> Self {
        self.include_metadata = include;
        self
    }

    /// Write an EDL to CMX 3600 format.
    pub fn write(&self, edl: &Edl) -> EdlResult<String> {
        let mut output = String::new();

        // Write title
        output.push_str(&format!("TITLE: {}\n", edl.title));

        // Write FCM
        let fcm = if edl.drop_frame {
            "DROP FRAME"
        } else {
            "NON-DROP FRAME"
        };
        output.push_str(&format!("FCM: {}\n\n", fcm));

        // Write global comments
        if self.include_comments {
            for comment in &edl.comments {
                output.push_str(&format!("* {}\n", comment));
            }
            if !edl.comments.is_empty() {
                output.push('\n');
            }
        }

        // Write events
        for event in &edl.events {
            self.write_event(&mut output, event);
        }

        Ok(output)
    }

    /// Write a single event.
    fn write_event(&self, output: &mut String, event: &EdlEvent) {
        // Write event line
        let edit_code = match event.edit_type {
            EditType::Cut => "C",
            EditType::Dissolve => "D",
            EditType::Wipe => "W",
            EditType::Key => "K",
        };

        output.push_str(&format!(
            "{:03}  {:8} {:5} {:1}        {} {} {} {}",
            event.number,
            event.reel,
            event.track,
            edit_code,
            event.source_in.format(),
            event.source_out.format(),
            event.record_in.format(),
            event.record_out.format()
        ));

        // Add transition duration for dissolves/wipes
        if let Some(duration) = event.transition_duration {
            if event.edit_type == EditType::Dissolve || event.edit_type == EditType::Wipe {
                output.push_str(&format!(" {:03}", duration));
            }
        }

        output.push('\n');

        // Write metadata comments
        if self.include_metadata {
            if let Some(clip_name) = event.metadata.get("clip_name") {
                output.push_str(&format!("* FROM CLIP NAME: {}\n", clip_name));
            }
            if let Some(to_clip_name) = event.metadata.get("to_clip_name") {
                output.push_str(&format!("* TO CLIP NAME: {}\n", to_clip_name));
            }
        }

        // Write comments
        if self.include_comments {
            for comment in &event.comments {
                output.push_str(&format!("* {}\n", comment));
            }
        }

        // Write motion effects
        if let Some(motion) = &event.motion_effect {
            if motion.freeze {
                output.push_str("* FREEZE FRAME\n");
            } else if motion.speed != 1.0 || motion.reverse {
                let speed_code = if motion.reverse {
                    -motion.speed.abs()
                } else {
                    motion.speed
                };

                output.push_str(&format!(
                    "M2   {:8} {:03}",
                    event.reel,
                    (speed_code * 100.0) as i32
                ));

                if let Some(entry) = &motion.entry {
                    output.push_str(&format!("   {}", entry.format()));
                }

                output.push('\n');

                if motion.reverse {
                    output.push_str("* REVERSE MOTION\n");
                }
            }
        }

        output.push('\n');
    }
}

impl Default for Cmx3600Writer {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse a CMX 3600 EDL from string.
pub fn parse(content: &str) -> EdlResult<Edl> {
    Cmx3600Parser::new().parse(content)
}

/// Write an EDL to CMX 3600 format.
pub fn write(edl: &Edl) -> EdlResult<String> {
    Cmx3600Writer::new().write(edl)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_edl() {
        let content = r"TITLE: Test Project
FCM: DROP FRAME

001  AX       V     C        01:00:00;00 01:00:05;00 01:00:00;00 01:00:05;00
* FROM CLIP NAME: CLIP001.MOV

002  BX       V     C        01:00:10;00 01:00:15;00 01:00:05;00 01:00:10;00
";

        let edl = parse(content).expect("edl should be valid");
        assert_eq!(edl.title, "Test Project");
        assert!(edl.drop_frame);
        assert_eq!(edl.events.len(), 2);

        let event1 = &edl.events[0];
        assert_eq!(event1.number, 1);
        assert_eq!(event1.reel, "AX");
        assert_eq!(event1.track, "V");
        assert_eq!(event1.edit_type, EditType::Cut);
    }

    #[test]
    fn test_parse_dissolve() {
        let content = r"TITLE: Dissolve Test
FCM: NON-DROP FRAME

001  AX       V     D        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00 030
";

        let edl = parse(content).expect("edl should be valid");
        assert_eq!(edl.events.len(), 1);

        let event = &edl.events[0];
        assert_eq!(event.edit_type, EditType::Dissolve);
        assert_eq!(event.transition_duration, Some(30));
    }

    #[test]
    fn test_parse_motion_effect() {
        let content = r"TITLE: Motion Test
FCM: DROP FRAME

001  AX       V     C        01:00:00;00 01:00:05;00 01:00:00;00 01:00:05;00
M2   AX       050   01:00:02;15
* SLOW MOTION
";

        let edl = parse(content).expect("edl should be valid");
        let event = &edl.events[0];

        assert!(event.motion_effect.is_some());
        let motion = event.motion_effect.as_ref().expect("motion should be valid");
        assert_eq!(motion.speed, 0.5);
        assert!(!motion.freeze);
    }

    #[test]
    fn test_parse_freeze_frame() {
        let content = r"TITLE: Freeze Test
FCM: DROP FRAME

001  AX       V     C        01:00:00;00 01:00:05;00 01:00:00;00 01:00:05;00
* FREEZE FRAME
";

        let edl = parse(content).expect("edl should be valid");
        let event = &edl.events[0];

        assert!(event.motion_effect.is_some());
        let motion = event.motion_effect.as_ref().expect("motion should be valid");
        assert!(motion.freeze);
    }

    #[test]
    fn test_write_edl() {
        let mut edl = Edl::new("Test Project".to_string(), Rational::new(30, 1), true);

        let event = EdlEvent {
            number: 1,
            reel: "AX".to_string(),
            track: "V".to_string(),
            edit_type: EditType::Cut,
            source_in: Timecode::new(1, 0, 0, 0, true, Rational::new(30, 1)),
            source_out: Timecode::new(1, 0, 5, 0, true, Rational::new(30, 1)),
            record_in: Timecode::new(1, 0, 0, 0, true, Rational::new(30, 1)),
            record_out: Timecode::new(1, 0, 5, 0, true, Rational::new(30, 1)),
            transition_duration: None,
            motion_effect: None,
            comments: vec!["Test comment".to_string()],
            metadata: HashMap::new(),
        };

        edl.add_event(event);

        let output = write(&edl).expect("output should be valid");
        assert!(output.contains("TITLE: Test Project"));
        assert!(output.contains("FCM: DROP FRAME"));
        assert!(output.contains("001  AX       V     C"));
        assert!(output.contains("* Test comment"));
    }

    #[test]
    fn test_roundtrip() {
        let content = r"TITLE: Roundtrip Test
FCM: DROP FRAME

001  AX       V     C        01:00:00;00 01:00:05;00 01:00:00;00 01:00:05;00
* FROM CLIP NAME: CLIP001.MOV

002  BX       V     D        01:00:10;00 01:00:15;00 01:00:05;00 01:00:10;00 030
";

        let edl = parse(content).expect("edl should be valid");
        let output = write(&edl).expect("output should be valid");
        let edl2 = parse(&output).expect("edl2 should be valid");

        assert_eq!(edl.title, edl2.title);
        assert_eq!(edl.drop_frame, edl2.drop_frame);
        assert_eq!(edl.events.len(), edl2.events.len());
    }

    #[test]
    fn test_timecode_parse() {
        let tc = Timecode::parse("01:23:45:29", Rational::new(30, 1)).expect("tc should be valid");
        assert_eq!(tc.hours, 1);
        assert_eq!(tc.minutes, 23);
        assert_eq!(tc.seconds, 45);
        assert_eq!(tc.frames, 29);
        assert!(!tc.drop_frame);

        let tc_df = Timecode::parse("01:23:45;29", Rational::new(30, 1)).expect("tc_df should be valid");
        assert!(tc_df.drop_frame);
    }

    #[test]
    fn test_timecode_format() {
        let tc = Timecode::new(1, 23, 45, 29, false, Rational::new(30, 1));
        assert_eq!(tc.format(), "01:23:45:29");

        let tc_df = Timecode::new(1, 23, 45, 29, true, Rational::new(30, 1));
        assert_eq!(tc_df.format(), "01:23:45;29");
    }

    #[test]
    fn test_timecode_conversion() {
        let tc = Timecode::new(0, 1, 30, 15, false, Rational::new(30, 1));
        let frames = tc.to_frames();
        let tc2 = Timecode::from_frames(frames, Rational::new(30, 1), false);

        assert_eq!(tc.hours, tc2.hours);
        assert_eq!(tc.minutes, tc2.minutes);
        assert_eq!(tc.seconds, tc2.seconds);
        assert_eq!(tc.frames, tc2.frames);
    }
}
