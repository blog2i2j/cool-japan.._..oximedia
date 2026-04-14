//! EDL import: parse CMX 3600 edit decision lists into [`crate::clip::Clip`] slices.
//!
//! `EdlImporter::from_cmx3600` is a lightweight, self-contained parser that
//! reads a CMX 3600 text EDL and converts each edit event into a `Clip`.
//!
//! Timecodes are expressed as `HH:MM:SS:FF` (non-drop-frame by default) and
//! converted to frames using the EDL frame rate (default 30 fps).  Source
//! in/out points are stored in `source_in` / `source_out`; record in/out
//! points are stored in `timeline_start` / `timeline_duration`.
//!
//! # CMX 3600 line format
//! ```text
//! NNN  REEL  TRACK  EDIT  SRC-IN  SRC-OUT  REC-IN  REC-OUT
//! ```
//!
//! Only cut (`C`) events are parsed; dissolves and wipes are skipped.
//!
//! # Example
//! ```rust
//! use oximedia_edit::edl_import::EdlImporter;
//!
//! let edl = "TITLE: Test\nFCM: NON-DROP FRAME\n\n\
//!             001  AX  V  C  00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00\n";
//! let clips = EdlImporter::from_cmx3600(edl);
//! assert_eq!(clips.len(), 1);
//! assert_eq!(clips[0].timeline_start, 0);
//! assert_eq!(clips[0].timeline_duration, 150); // 5s × 30fps
//! ```

#![allow(dead_code)]

use crate::clip::{Clip, ClipType};

/// Imports edit events from CMX 3600 EDL text into a `Vec<Clip>`.
pub struct EdlImporter {
    /// Frame rate used for timecode ↔ frame conversion (default 30).
    pub fps: u32,
}

impl Default for EdlImporter {
    fn default() -> Self {
        Self { fps: 30 }
    }
}

impl EdlImporter {
    /// Create an importer using the default 30 fps frame rate.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an importer with a custom frame rate.
    #[must_use]
    pub fn with_fps(fps: u32) -> Self {
        Self { fps }
    }

    /// Parse a CMX 3600 EDL text and return one `Clip` per cut event.
    ///
    /// Each clip carries:
    /// - `id`               — 1-based event number from the EDL.
    /// - `clip_type`        — `ClipType::Video` for tracks containing `V`,
    ///                        `ClipType::Audio` otherwise.
    /// - `timeline_start`   — record in-point converted to frames.
    /// - `timeline_duration`— record out-point minus record in-point (frames).
    /// - `source_in`        — source in-point converted to frames.
    /// - `source_out`       — source out-point converted to frames.
    /// - `metadata.label`   — reel name from the EDL.
    ///
    /// Lines that cannot be parsed as event entries (headers, comments, blanks)
    /// are silently skipped.
    #[must_use]
    pub fn from_cmx3600(edl_text: &str) -> Vec<Clip> {
        // Use a default importer (30 fps) for the associated function form.
        let importer = Self::default();
        importer.parse(edl_text)
    }

    /// Instance method form — uses `self.fps` for timecode conversion.
    #[must_use]
    pub fn parse(&self, edl_text: &str) -> Vec<Clip> {
        let mut clips = Vec::new();

        for line in edl_text.lines() {
            let line = line.trim();

            // Skip blank lines and comment/directive lines.
            if line.is_empty()
                || line.starts_with('*')
                || line.to_uppercase().starts_with("TITLE")
                || line.to_uppercase().starts_with("FCM")
            {
                continue;
            }

            if let Some(clip) = self.parse_event_line(line) {
                clips.push(clip);
            }
        }

        clips
    }

    /// Attempt to parse a single EDL event line.
    ///
    /// Expected token layout (whitespace-separated):
    /// `[event#] [reel] [track] [edit_type] [src_in] [src_out] [rec_in] [rec_out]`
    fn parse_event_line(&self, line: &str) -> Option<Clip> {
        let tokens: Vec<&str> = line.split_whitespace().collect();

        // Need at least 8 tokens for a valid event line.
        if tokens.len() < 8 {
            return None;
        }

        // Token 0: event number (must be a positive integer).
        let event_num: u64 = tokens[0].parse().ok()?;

        // Token 1: reel name.
        let reel = tokens[1];

        // Token 2: track type (V, A, A2, B, …).
        let track_token = tokens[2].to_uppercase();
        let clip_type = if track_token.contains('V') {
            ClipType::Video
        } else {
            ClipType::Audio
        };

        // Token 3: edit type — only handle cuts.
        let edit_type = tokens[3].to_uppercase();
        if edit_type != "C" {
            return None;
        }

        // Tokens 4–7: timecodes.
        let src_in = parse_timecode(tokens[4], self.fps)?;
        let src_out = parse_timecode(tokens[5], self.fps)?;
        let rec_in = parse_timecode(tokens[6], self.fps)?;
        let rec_out = parse_timecode(tokens[7], self.fps)?;

        if src_out < src_in || rec_out < rec_in {
            return None;
        }

        let rec_dur = (rec_out - rec_in) as i64;
        if rec_dur <= 0 {
            return None;
        }

        let mut clip = Clip::new(event_num, clip_type, rec_in as i64, rec_dur);
        clip.source_in = src_in as i64;
        clip.source_out = src_out as i64;
        clip.metadata.name = Some(reel.to_string());

        Some(clip)
    }
}

/// Parse a timecode string `HH:MM:SS:FF` or `HH:MM:SS;FF` into an absolute
/// frame count.
///
/// Returns `None` if the string does not match the expected pattern or any
/// field cannot be parsed.
fn parse_timecode(tc: &str, fps: u32) -> Option<u64> {
    // Accept both `:` and `;` as field separators (drop-frame uses `;`).
    let clean: String = tc.chars().map(|c| if c == ';' { ':' } else { c }).collect();
    let parts: Vec<&str> = clean.split(':').collect();
    if parts.len() != 4 {
        return None;
    }

    let hh: u64 = parts[0].parse().ok()?;
    let mm: u64 = parts[1].parse().ok()?;
    let ss: u64 = parts[2].parse().ok()?;
    let ff: u64 = parts[3].parse().ok()?;

    let fps = fps as u64;
    Some(((hh * 3600 + mm * 60 + ss) * fps) + ff)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIMPLE_EDL: &str = "TITLE: Test Project\n\
        FCM: NON-DROP FRAME\n\
        \n\
        001  AX  V  C  00:00:00:00 00:00:05:00 00:00:00:00 00:00:05:00\n\
        002  B1  A  C  00:00:10:00 00:00:20:00 00:00:05:00 00:00:15:00\n\
        003  AX  V  D  00:00:20:00 00:00:25:00 00:00:15:00 00:00:20:00\n";

    #[test]
    fn test_from_cmx3600_cut_events_only() {
        let clips = EdlImporter::from_cmx3600(SIMPLE_EDL);
        // Only cuts (C) are parsed; dissolve (D) event 003 is skipped.
        assert_eq!(clips.len(), 2);
    }

    #[test]
    fn test_clip_ids_match_event_numbers() {
        let clips = EdlImporter::from_cmx3600(SIMPLE_EDL);
        assert_eq!(clips[0].id, 1);
        assert_eq!(clips[1].id, 2);
    }

    #[test]
    fn test_clip_types_inferred_from_track() {
        let clips = EdlImporter::from_cmx3600(SIMPLE_EDL);
        assert_eq!(clips[0].clip_type, ClipType::Video);
        assert_eq!(clips[1].clip_type, ClipType::Audio);
    }

    #[test]
    fn test_timeline_position_in_frames() {
        let clips = EdlImporter::from_cmx3600(SIMPLE_EDL);
        // Event 001: rec_in = 00:00:00:00 → 0, rec_out = 00:00:05:00 → 150 (5s × 30fps)
        assert_eq!(clips[0].timeline_start, 0);
        assert_eq!(clips[0].timeline_duration, 150);
        // Event 002: rec_in = 00:00:05:00 → 150, dur = 10s → 300
        assert_eq!(clips[1].timeline_start, 150);
        assert_eq!(clips[1].timeline_duration, 300);
    }

    #[test]
    fn test_source_in_out_in_frames() {
        let clips = EdlImporter::from_cmx3600(SIMPLE_EDL);
        // Event 002: src_in = 00:00:10:00 → 300, src_out = 00:00:20:00 → 600
        assert_eq!(clips[1].source_in, 300);
        assert_eq!(clips[1].source_out, 600);
    }

    #[test]
    fn test_reel_name_in_metadata() {
        let clips = EdlImporter::from_cmx3600(SIMPLE_EDL);
        assert_eq!(clips[0].metadata.name.as_deref(), Some("AX"));
        assert_eq!(clips[1].metadata.name.as_deref(), Some("B1"));
    }

    #[test]
    fn test_empty_edl() {
        let clips = EdlImporter::from_cmx3600("");
        assert!(clips.is_empty());
    }

    #[test]
    fn test_with_fps_25() {
        let edl = "001  AX  V  C  00:00:00:00 00:00:04:00 00:00:00:00 00:00:04:00\n";
        let importer = EdlImporter::with_fps(25);
        let clips = importer.parse(edl);
        assert_eq!(clips.len(), 1);
        // 4s × 25fps = 100 frames
        assert_eq!(clips[0].timeline_duration, 100);
    }

    #[test]
    fn test_parse_timecode_basic() {
        assert_eq!(super::parse_timecode("01:00:00:00", 30), Some(108000));
        assert_eq!(super::parse_timecode("00:01:00:00", 30), Some(1800));
        assert_eq!(super::parse_timecode("00:00:01:00", 30), Some(30));
        assert_eq!(super::parse_timecode("00:00:00:15", 30), Some(15));
    }

    #[test]
    fn test_parse_timecode_drop_frame_semicolon() {
        // Semicolon separator (drop-frame notation) should also parse.
        assert_eq!(super::parse_timecode("00:00:01;00", 30), Some(30));
    }

    #[test]
    fn test_parse_timecode_invalid() {
        assert!(super::parse_timecode("not-a-tc", 30).is_none());
        assert!(super::parse_timecode("00:00:00", 30).is_none());
    }
}
