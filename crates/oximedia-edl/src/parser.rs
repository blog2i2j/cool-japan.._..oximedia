//! EDL parser implementation.
//!
//! This module provides a parser for CMX 3600 EDL files and related formats,
//! using the nom parser combinator library.

use crate::audio::AudioChannel;
use crate::error::{EdlError, EdlResult};
use crate::event::{EditType, EdlEvent, TrackType};
use crate::motion::MotionEffect;
use crate::reel::ReelId;
use crate::timecode::{EdlFrameRate, EdlTimecode};
use crate::{Edl, EdlFormat};
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while1},
    character::complete::{space0, space1},
    combinator::{map_res, opt, value},
    sequence::terminated,
    IResult, Parser,
};

/// Parse a complete EDL from a string.
///
/// # Errors
///
/// Returns an error if the EDL cannot be parsed.
pub fn parse_edl(input: &str) -> EdlResult<Edl> {
    let mut parser = EdlParser::new();
    parser.parse(input)
}

/// EDL parser with state management.
#[derive(Debug)]
pub struct EdlParser {
    /// Parsing mode (strict or lenient).
    pub strict_mode: bool,
    /// Current line number (for error reporting).
    current_line: usize,
}

impl EdlParser {
    /// Create a new EDL parser.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            strict_mode: false,
            current_line: 0,
        }
    }

    /// Create a new EDL parser in strict mode.
    #[must_use]
    pub const fn strict() -> Self {
        Self {
            strict_mode: true,
            current_line: 0,
        }
    }

    /// Enable or disable strict mode.
    pub fn set_strict_mode(&mut self, strict: bool) {
        self.strict_mode = strict;
    }

    /// Parse an EDL from a string.
    ///
    /// # Errors
    ///
    /// Returns an error if the EDL cannot be parsed.
    #[allow(clippy::too_many_lines)]
    pub fn parse(&mut self, input: &str) -> EdlResult<Edl> {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        let mut current_event: Option<EdlEvent> = None;

        for (line_num, line) in input.lines().enumerate() {
            self.current_line = line_num + 1;
            let trimmed = line.trim();

            // Skip empty lines
            if trimmed.is_empty() {
                continue;
            }

            // Parse comment lines
            if trimmed.starts_with('*') {
                if let Some(comment) = Self::parse_comment_line(trimmed) {
                    // Check for special comments
                    if comment.starts_with("FROM CLIP NAME:") {
                        if let Some(event) = &mut current_event {
                            let name = comment.trim_start_matches("FROM CLIP NAME:").trim();
                            event.set_clip_name(name.to_string());
                        }
                    } else if comment.starts_with("TO CLIP NAME:") {
                        if let Some(event) = &mut current_event {
                            let name = comment.trim_start_matches("TO CLIP NAME:").trim();
                            event.add_comment(format!("TO CLIP NAME: {name}"));
                        }
                    } else if comment.starts_with("M2") {
                        // Motion effect comment
                        if let Some(event) = &mut current_event {
                            if let Ok(effect) = MotionEffect::from_m2_comment(&comment) {
                                event.set_motion_effect(effect);
                            }
                        }
                    } else if let Some(event) = &mut current_event {
                        event.add_comment(comment);
                    }
                }
                continue;
            }

            // Parse header lines
            if trimmed.starts_with("TITLE:") {
                let title = trimmed.trim_start_matches("TITLE:").trim();
                edl.set_title(title.to_string());
                continue;
            }

            if trimmed.starts_with("FCM:") {
                let fcm = trimmed.trim_start_matches("FCM:").trim();
                let fcm_upper = fcm.to_uppercase();
                let frame_rate = if fcm_upper.contains("NON") {
                    EdlFrameRate::Fps2997NDF
                } else if fcm_upper.contains("DROP") {
                    EdlFrameRate::Fps2997DF
                } else {
                    EdlFrameRate::Fps2997NDF
                };
                edl.set_frame_rate(frame_rate);
                continue;
            }

            // Parse event lines
            if let Ok(event) = self.parse_event_line(trimmed, edl.frame_rate) {
                // Save previous event if any
                if let Some(prev_event) = current_event.take() {
                    edl.add_event(prev_event)
                        .map_err(|e| EdlError::parse(self.current_line, format!("{e}")))?;
                }
                current_event = Some(event);
            }
        }

        // Add the last event
        if let Some(event) = current_event {
            edl.add_event(event)
                .map_err(|e| EdlError::parse(self.current_line, format!("{e}")))?;
        }

        Ok(edl)
    }

    /// Parse a comment line (starts with *).
    fn parse_comment_line(line: &str) -> Option<String> {
        line.strip_prefix('*').map(|s| s.trim().to_string())
    }

    /// Parse an event line.
    ///
    /// Format: EVENT_NUM REEL TRACK EDIT_TYPE [DURATION] SRC_IN SRC_OUT REC_IN REC_OUT
    ///
    /// # Errors
    ///
    /// Returns an error if the event line is malformed.
    #[allow(clippy::too_many_lines)]
    fn parse_event_line(&self, line: &str, frame_rate: EdlFrameRate) -> EdlResult<EdlEvent> {
        let result = Self::event_line_parser(line, frame_rate);

        match result {
            Ok((_, event)) => Ok(event),
            Err(e) => Err(EdlError::parse(
                self.current_line,
                format!("Failed to parse event line: {e}"),
            )),
        }
    }

    /// Nom parser for event lines.
    fn event_line_parser(input: &str, frame_rate: EdlFrameRate) -> IResult<&str, EdlEvent> {
        let (input, _) = space0.parse(input)?;

        // Parse event number (3 digits, zero-padded)
        let mut parse_num = map_res(take_while1(|c: char| c.is_ascii_digit()), |s: &str| {
            s.parse::<u32>()
        });
        let (input, event_num) = parse_num.parse(input)?;

        let (input, _) = space1.parse(input)?;

        // Parse reel name
        let (input, reel) = take_while1(|c: char| !c.is_whitespace()).parse(input)?;
        let reel_id = ReelId::new(reel).map_err(|_| {
            nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))
        })?;

        let (input, _) = space1.parse(input)?;

        // Parse track type
        let (input, track) = Self::track_type_parser(input)?;

        let (input, _) = space1.parse(input)?;

        // Parse edit type
        let (input, edit_type) = Self::edit_type_parser(input)?;

        let (input, _) = space0.parse(input)?;

        // Parse optional transition duration
        let parse_duration = map_res(take_while1(|c: char| c.is_ascii_digit()), |s: &str| {
            s.parse::<u32>()
        });
        let mut opt_duration = opt(terminated(parse_duration, space1));
        let (input, transition_duration) = opt_duration.parse(input)?;

        // Consume any remaining spaces before timecodes
        let (input, _) = space0.parse(input)?;

        // Parse timecodes
        let (input, source_in) = Self::timecode_parser(input, frame_rate)?;
        let (input, _) = space1.parse(input)?;
        let (input, source_out) = Self::timecode_parser(input, frame_rate)?;
        let (input, _) = space1.parse(input)?;
        let (input, record_in) = Self::timecode_parser(input, frame_rate)?;
        let (input, _) = space1.parse(input)?;
        let (input, record_out) = Self::timecode_parser(input, frame_rate)?;

        let mut event = EdlEvent::new(
            event_num,
            reel_id.to_string(),
            track,
            edit_type,
            source_in,
            source_out,
            record_in,
            record_out,
        );

        if let Some(duration) = transition_duration {
            event.set_transition_duration(duration);
        }

        Ok((input, event))
    }

    /// Parse track type.
    fn track_type_parser(input: &str) -> IResult<&str, TrackType> {
        alt((
            value(TrackType::AudioPairWithVideo, tag("AA/V")),
            value(TrackType::AudioWithVideo, tag("A/V")),
            value(TrackType::AudioPair, tag("AA")),
            value(TrackType::Audio(AudioChannel::A4), tag("A4")),
            value(TrackType::Audio(AudioChannel::A3), tag("A3")),
            value(TrackType::Audio(AudioChannel::A2), tag("A2")),
            value(TrackType::Audio(AudioChannel::A1), tag("A")),
            value(TrackType::Video, tag("V")),
        ))
        .parse(input)
    }

    /// Parse edit type.
    fn edit_type_parser(input: &str) -> IResult<&str, EditType> {
        alt((
            value(EditType::Cut, tag("C")),
            value(EditType::Dissolve, tag("D")),
            value(EditType::Wipe, tag("W")),
            value(EditType::Key, tag("K")),
        ))
        .parse(input)
    }

    /// Parse timecode (HH:MM:SS:FF or HH:MM:SS;FF).
    fn timecode_parser(input: &str, frame_rate: EdlFrameRate) -> IResult<&str, EdlTimecode> {
        let (input, tc_str) =
            take_while1(|c: char| c.is_ascii_digit() || c == ':' || c == ';').parse(input)?;

        let tc = EdlTimecode::parse(tc_str, frame_rate).map_err(|_| {
            nom::Err::Error(nom::error::Error::new(input, nom::error::ErrorKind::Tag))
        })?;

        Ok((input, tc))
    }
}

impl Default for EdlParser {
    fn default() -> Self {
        Self::new()
    }
}

/// Parse frame rate from FCM line.
#[allow(dead_code)]
fn parse_fcm(input: &str) -> EdlResult<EdlFrameRate> {
    let upper = input.to_uppercase();
    // Check for "NON" before checking for "DROP" since "NON DROP FRAME" contains "DROP"
    if upper.contains("NON") {
        Ok(EdlFrameRate::Fps2997NDF)
    } else if upper.contains("DROP") {
        Ok(EdlFrameRate::Fps2997DF)
    } else {
        Ok(EdlFrameRate::Fps2997NDF)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_edl() {
        let edl_text = r#"TITLE: Test EDL
FCM: DROP FRAME

001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00
* FROM CLIP NAME: SHOT_001.MOV

002  AX       V     D    030 01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00
* FROM CLIP NAME: SHOT_002.MOV
"#;

        let mut parser = EdlParser::new();
        let edl = parser.parse(edl_text).expect("failed to parse");

        assert_eq!(edl.title, Some("Test EDL".to_string()));
        assert_eq!(edl.events.len(), 2);
        assert_eq!(edl.events[0].number, 1);
        assert_eq!(edl.events[0].edit_type, EditType::Cut);
        assert_eq!(edl.events[1].number, 2);
        assert_eq!(edl.events[1].edit_type, EditType::Dissolve);
        assert_eq!(edl.events[1].transition_duration, Some(30));
    }

    #[test]
    fn test_parse_comment_line() {
        let comment = EdlParser::parse_comment_line("* This is a comment");
        assert_eq!(comment, Some("This is a comment".to_string()));
    }

    #[test]
    fn test_timecode_parser() {
        let (_, tc) = EdlParser::timecode_parser("01:02:03:04", EdlFrameRate::Fps25)
            .expect("operation should succeed");
        assert_eq!(tc.hours(), 1);
        assert_eq!(tc.minutes(), 2);
        assert_eq!(tc.seconds(), 3);
        assert_eq!(tc.frames(), 4);
    }

    #[test]
    fn test_track_type_parser() {
        let (_, track) = EdlParser::track_type_parser("V").expect("operation should succeed");
        assert_eq!(track, TrackType::Video);

        let (_, track) = EdlParser::track_type_parser("A").expect("operation should succeed");
        assert_eq!(track, TrackType::Audio(AudioChannel::A1));

        let (_, track) = EdlParser::track_type_parser("AA/V").expect("operation should succeed");
        assert_eq!(track, TrackType::AudioPairWithVideo);
    }

    #[test]
    fn test_edit_type_parser() {
        let (_, edit) = EdlParser::edit_type_parser("C").expect("operation should succeed");
        assert_eq!(edit, EditType::Cut);

        let (_, edit) = EdlParser::edit_type_parser("D").expect("operation should succeed");
        assert_eq!(edit, EditType::Dissolve);
    }

    #[test]
    fn test_event_line_parser() {
        let line = "001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00";
        let (_, event) = EdlParser::event_line_parser(line, EdlFrameRate::Fps2997DF)
            .expect("operation should succeed");

        assert_eq!(event.number, 1);
        assert_eq!(event.reel, "AX");
        assert_eq!(event.track, TrackType::Video);
        assert_eq!(event.edit_type, EditType::Cut);
    }

    #[test]
    fn test_event_with_transition() {
        let line = "002  AX       V     D    030 01:00:05:00 01:00:10:00 01:00:05:00 01:00:10:00";
        let (_, event) = EdlParser::event_line_parser(line, EdlFrameRate::Fps2997DF)
            .expect("operation should succeed");

        assert_eq!(event.number, 2);
        assert_eq!(event.edit_type, EditType::Dissolve);
        assert_eq!(event.transition_duration, Some(30));
    }

    #[test]
    fn test_parse_fcm() {
        assert_eq!(
            parse_fcm("DROP FRAME").expect("operation should succeed"),
            EdlFrameRate::Fps2997DF
        );
        assert_eq!(
            parse_fcm("NON-DROP FRAME").expect("operation should succeed"),
            EdlFrameRate::Fps2997NDF
        );
        assert_eq!(
            parse_fcm("NON DROP FRAME").expect("operation should succeed"),
            EdlFrameRate::Fps2997NDF
        );
    }

    #[test]
    fn test_parse_clip_name_comment() {
        let edl_text = r#"001  AX       V     C        01:00:00:00 01:00:05:00 01:00:00:00 01:00:05:00
* FROM CLIP NAME: test_clip.mov"#;

        let mut parser = EdlParser::new();
        let edl = parser.parse(edl_text).expect("failed to parse");

        assert_eq!(edl.events.len(), 1);
        assert_eq!(edl.events[0].clip_name, Some("test_clip.mov".to_string()));
    }
}
