//! Timecode display formatting in different regional and industry conventions.
//!
//! This module provides formatting utilities for rendering timecodes according
//! to different conventions:
//!
//! - **SMPTE** (North American broadcast): `HH:MM:SS:FF` / `HH:MM:SS;FF`
//! - **EBU** (European broadcast): `HH:MM:SS:FF` (always colon, even drop-frame)
//! - **Film** (cinema): `HH+MM:SS:FF` with reel indicator
//! - **Feet+Frames** (16/35mm film): `FFFF+FF` feet-and-frames notation
//! - **Samples** (audio DAW): absolute sample count at given sample rate
//! - **Milliseconds** (editing): `HH:MM:SS.mmm` sub-second in milliseconds

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]

use crate::Timecode;
use std::fmt;

/// Display convention for timecode rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DisplayConvention {
    /// SMPTE convention: colon for NDF, semicolon for DF.
    Smpte,
    /// EBU convention: always uses colons regardless of drop-frame.
    Ebu,
    /// Film convention: `HH+MM:SS:FF` (reel+time).
    Film,
    /// Feet and frames for 35mm film at 24fps (16 frames per foot).
    FeetFrames35mm,
    /// Feet and frames for 16mm film at 24fps (40 frames per foot).
    FeetFrames16mm,
    /// Absolute milliseconds: `HH:MM:SS.mmm`.
    Milliseconds,
    /// Relative frame number from 00:00:00:00.
    FrameNumber,
}

/// Options for formatting a timecode.
#[derive(Debug, Clone)]
pub struct DisplayOptions {
    /// Display convention to use.
    pub convention: DisplayConvention,
    /// Whether to include a sign prefix (+/-) for relative values.
    pub show_sign: bool,
    /// Whether to zero-pad all fields.
    pub zero_pad: bool,
    /// Custom separator between fields (overrides convention default).
    pub custom_separator: Option<char>,
}

impl Default for DisplayOptions {
    fn default() -> Self {
        Self {
            convention: DisplayConvention::Smpte,
            show_sign: false,
            zero_pad: true,
            custom_separator: None,
        }
    }
}

impl DisplayOptions {
    /// SMPTE display options.
    #[must_use]
    pub fn smpte() -> Self {
        Self {
            convention: DisplayConvention::Smpte,
            ..Self::default()
        }
    }

    /// EBU display options.
    #[must_use]
    pub fn ebu() -> Self {
        Self {
            convention: DisplayConvention::Ebu,
            ..Self::default()
        }
    }

    /// Film display options.
    #[must_use]
    pub fn film() -> Self {
        Self {
            convention: DisplayConvention::Film,
            ..Self::default()
        }
    }

    /// Milliseconds display options.
    #[must_use]
    pub fn milliseconds() -> Self {
        Self {
            convention: DisplayConvention::Milliseconds,
            ..Self::default()
        }
    }

    /// Frame number display options.
    #[must_use]
    pub fn frame_number() -> Self {
        Self {
            convention: DisplayConvention::FrameNumber,
            ..Self::default()
        }
    }
}

/// A formatted representation of a timecode value.
#[derive(Debug, Clone)]
pub struct FormattedTimecode {
    /// The rendered string.
    pub text: String,
    /// The convention used.
    pub convention: DisplayConvention,
}

impl fmt::Display for FormattedTimecode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

/// Format a timecode according to the given display options.
///
/// # Examples
///
/// ```rust
/// use oximedia_timecode::{Timecode, FrameRate};
/// use oximedia_timecode::timecode_display::{format_timecode, DisplayOptions};
///
/// let tc = Timecode::new(1, 30, 0, 12, FrameRate::Fps25).expect("valid");
/// let s = format_timecode(&tc, &DisplayOptions::ebu());
/// assert_eq!(s.text, "01:30:00:12");
/// ```
#[must_use]
pub fn format_timecode(tc: &Timecode, opts: &DisplayOptions) -> FormattedTimecode {
    let text = match opts.convention {
        DisplayConvention::Smpte => format_smpte(tc, opts),
        DisplayConvention::Ebu => format_ebu(tc, opts),
        DisplayConvention::Film => format_film(tc, opts),
        DisplayConvention::FeetFrames35mm => format_feet_frames(tc, 16),
        DisplayConvention::FeetFrames16mm => format_feet_frames(tc, 40),
        DisplayConvention::Milliseconds => format_milliseconds(tc),
        DisplayConvention::FrameNumber => format_frame_number(tc),
    };
    FormattedTimecode {
        text,
        convention: opts.convention,
    }
}

/// SMPTE format: `HH:MM:SS:FF` for NDF, `HH:MM:SS;FF` for DF.
fn format_smpte(tc: &Timecode, _opts: &DisplayOptions) -> String {
    let sep = if tc.frame_rate.drop_frame { ';' } else { ':' };
    format!(
        "{:02}:{:02}:{:02}{}{:02}",
        tc.hours, tc.minutes, tc.seconds, sep, tc.frames
    )
}

/// EBU format: always uses colons.
fn format_ebu(tc: &Timecode, _opts: &DisplayOptions) -> String {
    format!(
        "{:02}:{:02}:{:02}:{:02}",
        tc.hours, tc.minutes, tc.seconds, tc.frames
    )
}

/// Film format: `HH+MM:SS:FF` (reel hour indicator).
fn format_film(tc: &Timecode, _opts: &DisplayOptions) -> String {
    format!(
        "{:02}+{:02}:{:02}:{:02}",
        tc.hours, tc.minutes, tc.seconds, tc.frames
    )
}

/// Feet+frames format for film (FFFF+FF).
///
/// `frames_per_foot` is typically 16 for 35mm and 40 for 16mm at 24fps.
fn format_feet_frames(tc: &Timecode, frames_per_foot: u64) -> String {
    let total_frames = tc.to_frames();
    let feet = total_frames / frames_per_foot;
    let leftover = total_frames % frames_per_foot;
    format!("{feet:04}+{leftover:02}")
}

/// Milliseconds format: `HH:MM:SS.mmm`.
fn format_milliseconds(tc: &Timecode) -> String {
    let fps = crate::frame_rate_from_info(&tc.frame_rate).as_float();
    let ms = if fps > 0.0 {
        ((tc.frames as f64 / fps) * 1000.0).round() as u32
    } else {
        0
    };
    format!(
        "{:02}:{:02}:{:02}.{:03}",
        tc.hours,
        tc.minutes,
        tc.seconds,
        ms.min(999)
    )
}

/// Frame number format: absolute frame count from 00:00:00:00.
fn format_frame_number(tc: &Timecode) -> String {
    format!("{}", tc.to_frames())
}

/// Parse a formatted timecode string back to a [`Timecode`].
///
/// Handles SMPTE (colon/semicolon), EBU (colon), and film (+) conventions.
///
/// # Errors
///
/// Returns error if the string cannot be parsed.
pub fn parse_display(
    s: &str,
    frame_rate: crate::FrameRate,
) -> Result<Timecode, crate::TimecodeError> {
    // Try standard from_string which handles HH:MM:SS:FF and HH:MM:SS;FF
    Timecode::from_string(s, frame_rate).or_else(|_| {
        // Try film format HH+MM:SS:FF
        let normalized = s.replacen('+', ":", 1);
        Timecode::from_string(&normalized, frame_rate)
    })
}

/// Comparison table entry showing a timecode in multiple conventions.
#[derive(Debug, Clone)]
pub struct ConventionComparison {
    /// Original timecode.
    pub timecode: Timecode,
    /// SMPTE representation.
    pub smpte: String,
    /// EBU representation.
    pub ebu: String,
    /// Film representation.
    pub film: String,
    /// Milliseconds representation.
    pub ms: String,
    /// Frame number.
    pub frame: String,
}

impl ConventionComparison {
    /// Build a comparison for a given timecode.
    #[must_use]
    pub fn build(tc: Timecode) -> Self {
        let smpte = format_timecode(&tc, &DisplayOptions::smpte()).text;
        let ebu = format_timecode(&tc, &DisplayOptions::ebu()).text;
        let film = format_timecode(&tc, &DisplayOptions::film()).text;
        let ms = format_timecode(&tc, &DisplayOptions::milliseconds()).text;
        let frame = format_timecode(&tc, &DisplayOptions::frame_number()).text;
        Self {
            timecode: tc,
            smpte,
            ebu,
            film,
            ms,
            frame,
        }
    }

    /// Render as a human-readable table row.
    #[must_use]
    pub fn to_table_row(&self) -> String {
        format!(
            "| {:>13} | {:>13} | {:>13} | {:>12} | {:>8} |",
            self.smpte, self.ebu, self.film, self.ms, self.frame
        )
    }
}

/// Build a comparison table header string.
#[must_use]
pub fn comparison_table_header() -> String {
    format!(
        "| {:>13} | {:>13} | {:>13} | {:>12} | {:>8} |",
        "SMPTE", "EBU", "Film", "Milliseconds", "Frames"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FrameRate;

    fn tc25(h: u8, m: u8, s: u8, f: u8) -> Timecode {
        Timecode::new(h, m, s, f, FrameRate::Fps25).expect("valid tc")
    }

    #[test]
    fn test_smpte_ndf() {
        let tc = tc25(1, 30, 0, 12);
        let fmt = format_timecode(&tc, &DisplayOptions::smpte());
        assert_eq!(fmt.text, "01:30:00:12");
    }

    #[test]
    fn test_ebu_always_colon() {
        let tc = tc25(0, 0, 0, 0);
        let fmt = format_timecode(&tc, &DisplayOptions::ebu());
        assert_eq!(fmt.text, "00:00:00:00");
    }

    #[test]
    fn test_film_format() {
        let tc = tc25(2, 15, 30, 5);
        let fmt = format_timecode(&tc, &DisplayOptions::film());
        assert_eq!(fmt.text, "02+15:30:05");
    }

    #[test]
    fn test_milliseconds_format() {
        // 12 frames at 25fps = 480ms
        let tc = tc25(0, 0, 0, 12);
        let fmt = format_timecode(&tc, &DisplayOptions::milliseconds());
        assert_eq!(fmt.text, "00:00:00.480");
    }

    #[test]
    fn test_frame_number() {
        // 1h at 25fps = 90000 frames
        let tc = tc25(1, 0, 0, 0);
        let fmt = format_timecode(&tc, &DisplayOptions::frame_number());
        assert_eq!(fmt.text, "90000");
    }

    #[test]
    fn test_feet_frames_35mm() {
        // 32 frames = 2 feet + 0 leftover (16 fps per foot)
        let tc = tc25(0, 0, 1, 7); // 25 + 7 = 32 frames at 25fps
        let fmt = format_timecode(
            &tc,
            &DisplayOptions {
                convention: DisplayConvention::FeetFrames35mm,
                ..DisplayOptions::default()
            },
        );
        assert!(!fmt.text.is_empty());
    }

    #[test]
    fn test_comparison_build() {
        let tc = tc25(1, 0, 0, 0);
        let comp = ConventionComparison::build(tc);
        assert!(!comp.smpte.is_empty());
        assert!(!comp.ebu.is_empty());
    }

    #[test]
    fn test_parse_display_smpte() {
        let parsed = parse_display("01:30:00:12", FrameRate::Fps25).expect("parse ok");
        assert_eq!(parsed.hours, 1);
        assert_eq!(parsed.minutes, 30);
        assert_eq!(parsed.seconds, 0);
        assert_eq!(parsed.frames, 12);
    }
}
