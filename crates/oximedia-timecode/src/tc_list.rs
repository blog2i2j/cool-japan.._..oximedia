// Copyright 2025 OxiMedia Contributors
// Licensed under the Apache License, Version 2.0

//! Timecode list import / export.
//!
//! `TcList` parses a CSV file (or string) containing IN,OUT timecode pairs and
//! returns them as `Vec<(Timecode, Timecode)>`.
//!
//! # CSV format
//!
//! ```csv
//! # optional comment lines are ignored
//! 00:00:01:00,00:00:04:00
//! 00:01:00;00,00:01:30;00
//! ```
//!
//! - Separator: `,` (comma) or `\t` (tab).
//! - Timecode format: `HH:MM:SS:FF` (NDF) or `HH:MM:SS;FF` (DF with semicolon).
//! - Lines starting with `#` or empty lines are skipped.
//! - Additional trailing fields on a line are ignored.

use crate::{FrameRate, Timecode, TimecodeError};

/// A collection of IN/OUT timecode pairs.
pub struct TcList;

impl TcList {
    /// Parse a CSV (or TSV) string and return all valid IN/OUT pairs.
    ///
    /// Invalid rows (parse errors, missing fields, etc.) are silently skipped.
    ///
    /// The `frame_rate` parameter is applied to all parsed timecodes.
    pub fn from_csv_with_rate(csv: &str, frame_rate: FrameRate) -> Vec<(Timecode, Timecode)> {
        csv.lines()
            .filter(|l| {
                let trimmed = l.trim();
                !trimmed.is_empty() && !trimmed.starts_with('#')
            })
            .filter_map(|line| {
                let parts: Vec<&str> = if line.contains('\t') {
                    line.splitn(3, '\t').collect()
                } else {
                    line.splitn(3, ',').collect()
                };

                if parts.len() < 2 {
                    return None;
                }

                let tc_in = parse_timecode_str(parts[0].trim(), frame_rate).ok()?;
                let tc_out = parse_timecode_str(parts[1].trim(), frame_rate).ok()?;
                Some((tc_in, tc_out))
            })
            .collect()
    }

    /// Parse CSV using a default frame rate of 25 fps (PAL).
    ///
    /// Convenience wrapper around [`from_csv_with_rate`][Self::from_csv_with_rate].
    pub fn from_csv(csv: &str) -> Vec<(Timecode, Timecode)> {
        Self::from_csv_with_rate(csv, FrameRate::Fps25)
    }

    /// Serialise a list of IN/OUT pairs back to CSV.
    ///
    /// Uses `HH:MM:SS:FF` format for NDF and `HH:MM:SS;FF` for DF rates.
    pub fn to_csv(pairs: &[(Timecode, Timecode)]) -> String {
        let mut out = String::new();
        for (tc_in, tc_out) in pairs {
            let sep_in = if tc_in.frame_rate.drop_frame {
                ';'
            } else {
                ':'
            };
            let sep_out = if tc_out.frame_rate.drop_frame {
                ';'
            } else {
                ':'
            };
            out.push_str(&format!(
                "{:02}:{:02}:{:02}{sep_in}{:02},{:02}:{:02}:{:02}{sep_out}{:02}\n",
                tc_in.hours,
                tc_in.minutes,
                tc_in.seconds,
                tc_in.frames,
                tc_out.hours,
                tc_out.minutes,
                tc_out.seconds,
                tc_out.frames,
            ));
        }
        out
    }
}

/// Parse a timecode string in `HH:MM:SS:FF` or `HH:MM:SS;FF` format.
///
/// The final separator (`:` or `;`) determines whether the timecode is
/// interpreted as drop-frame (`;`) or non-drop-frame (`:`).
///
/// The `frame_rate` supplied is used unless the timecode is clearly drop-frame
/// (`;` separator), in which case the corresponding DF variant is chosen.
fn parse_timecode_str(s: &str, frame_rate: FrameRate) -> Result<Timecode, TimecodeError> {
    // Expected formats:
    //   HH:MM:SS:FF  (colon-colon-colon — NDF)
    //   HH:MM:SS;FF  (colon-colon-semicolon — DF)
    if s.len() < 11 {
        return Err(TimecodeError::InvalidConfiguration);
    }

    // Find the last non-digit, non-leading separator
    let last_sep = s
        .char_indices()
        .filter(|(_, c)| *c == ':' || *c == ';')
        .last();

    let (last_sep_pos, last_sep_char) = last_sep.ok_or(TimecodeError::InvalidConfiguration)?;

    // Split into "HH:MM:SS" and "FF"
    let tc_part = &s[..last_sep_pos];
    let ff_str = &s[(last_sep_pos + 1)..];

    let mut colon_parts = tc_part.splitn(4, ':');
    let hh: u8 = colon_parts
        .next()
        .and_then(|p| p.parse().ok())
        .ok_or(TimecodeError::InvalidHours)?;
    let mm: u8 = colon_parts
        .next()
        .and_then(|p| p.parse().ok())
        .ok_or(TimecodeError::InvalidMinutes)?;
    let ss: u8 = colon_parts
        .next()
        .and_then(|p| p.parse().ok())
        .ok_or(TimecodeError::InvalidSeconds)?;
    let ff: u8 = ff_str.parse().map_err(|_| TimecodeError::InvalidFrames)?;

    // Select drop-frame variant when ';' is used
    let effective_rate = if last_sep_char == ';' {
        to_drop_frame_variant(frame_rate)
    } else {
        frame_rate
    };

    Timecode::new(hh, mm, ss, ff, effective_rate)
}

/// Return the drop-frame variant of `rate` if one exists; otherwise return `rate`.
fn to_drop_frame_variant(rate: FrameRate) -> FrameRate {
    match rate {
        FrameRate::Fps2997NDF | FrameRate::Fps2997DF => FrameRate::Fps2997DF,
        FrameRate::Fps23976 | FrameRate::Fps23976DF => FrameRate::Fps23976DF,
        FrameRate::Fps5994 | FrameRate::Fps5994DF => FrameRate::Fps5994DF,
        FrameRate::Fps47952 | FrameRate::Fps47952DF => FrameRate::Fps47952DF,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_CSV: &str = "\
# comment line
00:00:01:00,00:00:04:00
00:01:00:00,00:01:30:00
";

    #[test]
    fn parse_two_pairs() {
        let pairs = TcList::from_csv(SAMPLE_CSV);
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn first_pair_values() {
        let pairs = TcList::from_csv(SAMPLE_CSV);
        let (tc_in, tc_out) = &pairs[0];
        assert_eq!(tc_in.seconds, 1);
        assert_eq!(tc_out.seconds, 4);
    }

    #[test]
    fn second_pair_values() {
        let pairs = TcList::from_csv(SAMPLE_CSV);
        let (tc_in, tc_out) = &pairs[1];
        assert_eq!(tc_in.minutes, 1);
        assert_eq!(tc_out.minutes, 1);
        assert_eq!(tc_out.seconds, 30);
    }

    #[test]
    fn empty_and_comment_lines_skipped() {
        let csv = "\n# ignored\n\n00:00:00:00,00:00:01:00\n";
        let pairs = TcList::from_csv(csv);
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn invalid_row_silently_skipped() {
        let csv = "bad,data\n00:00:01:00,00:00:02:00\n";
        let pairs = TcList::from_csv(csv);
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn drop_frame_semicolon_parsed() {
        let csv = "00:00:01;00,00:00:04;00\n";
        let pairs = TcList::from_csv_with_rate(csv, FrameRate::Fps2997NDF);
        assert_eq!(pairs.len(), 1);
        assert!(pairs[0].0.frame_rate.drop_frame);
    }

    #[test]
    fn tab_separated_accepted() {
        let csv = "00:00:01:00\t00:00:04:00\n";
        let pairs = TcList::from_csv(csv);
        assert_eq!(pairs.len(), 1);
    }

    #[test]
    fn round_trip_csv() {
        let original = TcList::from_csv(SAMPLE_CSV);
        let serialised = TcList::to_csv(&original);
        let reparsed = TcList::from_csv(&serialised);
        assert_eq!(original.len(), reparsed.len());
        for (a, b) in original.iter().zip(reparsed.iter()) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
        }
    }
}
