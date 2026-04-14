//! Timecode list export utilities.
//!
//! Provides [`TcListExporter`] which converts a slice of [`crate::event::EdlEvent`]s
//! into a comma-separated values (CSV) representation suitable for import into
//! spreadsheet tools or NLE timecode panels.
//!
//! # CSV Format
//!
//! The exported CSV contains the following columns (with a header row):
//!
//! ```text
//! event_number,reel,source_in,source_out,record_in,record_out
//! 001,A001,01:00:00:00,01:00:05:00,01:00:00:00,01:00:05:00
//! ```
//!
//! # Example
//!
//! ```
//! use oximedia_edl::tc_list::TcListExporter;
//! use oximedia_edl::event::{EdlEvent, EditType, TrackType};
//! use oximedia_edl::timecode::{EdlFrameRate, EdlTimecode};
//!
//! fn tc(s: u8) -> EdlTimecode {
//!     EdlTimecode::new(1, 0, s, 0, EdlFrameRate::Fps25).expect("valid timecode")
//! }
//!
//! let event = EdlEvent::new(
//!     1,
//!     "A001".to_string(),
//!     TrackType::Video,
//!     EditType::Cut,
//!     tc(0), tc(5), tc(0), tc(5),
//! );
//!
//! let csv = TcListExporter::to_csv(&[event]);
//! assert!(csv.starts_with("event_number,"));
//! assert!(csv.contains("A001"));
//! ```

#![allow(dead_code)]

use crate::event::EdlEvent;

// ─────────────────────────────────────────────────────────────────────────────
// TcListExporter
// ─────────────────────────────────────────────────────────────────────────────

/// Column selection for the TC list export.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TcColumn {
    /// Event number (zero-padded to 3 digits).
    EventNumber,
    /// Reel / tape name.
    Reel,
    /// Source in-point timecode.
    SourceIn,
    /// Source out-point timecode.
    SourceOut,
    /// Record (timeline) in-point timecode.
    RecordIn,
    /// Record (timeline) out-point timecode.
    RecordOut,
    /// Optional clip name (empty string when absent).
    ClipName,
    /// Duration in frames (record_out − record_in).
    DurationFrames,
}

impl TcColumn {
    /// Header label for this column.
    #[must_use]
    pub const fn header(&self) -> &'static str {
        match self {
            Self::EventNumber => "event_number",
            Self::Reel => "reel",
            Self::SourceIn => "source_in",
            Self::SourceOut => "source_out",
            Self::RecordIn => "record_in",
            Self::RecordOut => "record_out",
            Self::ClipName => "clip_name",
            Self::DurationFrames => "duration_frames",
        }
    }
}

/// Stateless exporter that converts `EdlEvent` slices to CSV text.
pub struct TcListExporter;

impl TcListExporter {
    /// Export `events` as a CSV string.
    ///
    /// The first row is a header; each subsequent row represents one event.
    /// Fields are separated by commas; no quoting is applied (timecode strings
    /// and reel names do not contain commas in well-formed EDLs).
    ///
    /// # Column Layout
    ///
    /// | Column | Description |
    /// |--------|-------------|
    /// | `event_number` | Sequential event number (zero-padded to 3 digits) |
    /// | `reel` | Source reel / tape name |
    /// | `source_in` | Source in-point timecode |
    /// | `source_out` | Source out-point timecode |
    /// | `record_in` | Record (timeline) in-point timecode |
    /// | `record_out` | Record (timeline) out-point timecode |
    pub fn to_csv(events: &[EdlEvent]) -> String {
        let mut out = String::with_capacity(64 + events.len() * 80);
        out.push_str("event_number,reel,source_in,source_out,record_in,record_out\n");
        for ev in events {
            let line = format!(
                "{:03},{},{},{},{},{}\n",
                ev.number,
                ev.reel,
                ev.source_in,
                ev.source_out,
                ev.record_in,
                ev.record_out,
            );
            out.push_str(&line);
        }
        out
    }

    /// Export `events` to CSV with an optional custom delimiter.
    ///
    /// The delimiter replaces `,` in each data row **and** the header.
    /// A common alternative is `\t` (TSV).
    pub fn to_delimited(events: &[EdlEvent], delimiter: char) -> String {
        let sep = delimiter.to_string();
        let mut out = String::with_capacity(64 + events.len() * 80);
        // Header
        let header_fields = [
            "event_number",
            "reel",
            "source_in",
            "source_out",
            "record_in",
            "record_out",
        ];
        out.push_str(&header_fields.join(&sep));
        out.push('\n');

        for ev in events {
            let row = [
                format!("{:03}", ev.number),
                ev.reel.clone(),
                ev.source_in.to_string(),
                ev.source_out.to_string(),
                ev.record_in.to_string(),
                ev.record_out.to_string(),
            ]
            .join(&sep);
            out.push_str(&row);
            out.push('\n');
        }
        out
    }

    /// Export `events` with a custom selection of columns.
    ///
    /// The caller provides a slice of [`TcColumn`] values which determines
    /// both the header labels and the data values, and their order.
    pub fn to_csv_columns(events: &[EdlEvent], columns: &[TcColumn]) -> String {
        let mut out = String::with_capacity(64 + events.len() * 80);

        // Header row
        let headers: Vec<&str> = columns.iter().map(|c| c.header()).collect();
        out.push_str(&headers.join(","));
        out.push('\n');

        for ev in events {
            let record_duration = ev.record_out.to_frames().saturating_sub(ev.record_in.to_frames());
            let cells: Vec<String> = columns
                .iter()
                .map(|col| match col {
                    TcColumn::EventNumber => format!("{:03}", ev.number),
                    TcColumn::Reel => ev.reel.clone(),
                    TcColumn::SourceIn => ev.source_in.to_string(),
                    TcColumn::SourceOut => ev.source_out.to_string(),
                    TcColumn::RecordIn => ev.record_in.to_string(),
                    TcColumn::RecordOut => ev.record_out.to_string(),
                    TcColumn::ClipName => ev
                        .clip_name
                        .as_deref()
                        .unwrap_or("")
                        .to_string(),
                    TcColumn::DurationFrames => record_duration.to_string(),
                })
                .collect();
            out.push_str(&cells.join(","));
            out.push('\n');
        }
        out
    }

    /// Export `events` as a compact reel-only list (one reel name per event,
    /// deduplicated, preserving first-occurrence order).
    ///
    /// Useful for extracting which reels an EDL draws from.
    #[must_use]
    pub fn to_reel_list(events: &[EdlEvent]) -> Vec<String> {
        let mut seen = std::collections::HashSet::new();
        let mut reels = Vec::new();
        for ev in events {
            if seen.insert(ev.reel.clone()) {
                reels.push(ev.reel.clone());
            }
        }
        reels
    }

    /// Count how many events reference each reel.
    ///
    /// Returns a `Vec<(reel_name, count)>` sorted descending by count.
    #[must_use]
    pub fn reel_event_counts(events: &[EdlEvent]) -> Vec<(String, usize)> {
        let mut counts: std::collections::HashMap<String, usize> = std::collections::HashMap::new();
        for ev in events {
            *counts.entry(ev.reel.clone()).or_insert(0) += 1;
        }
        let mut result: Vec<(String, usize)> = counts.into_iter().collect();
        result.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        result
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, EdlEvent, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};

    fn tc(s: u8) -> EdlTimecode {
        EdlTimecode::new(1, 0, s, 0, EdlFrameRate::Fps25).expect("valid timecode")
    }

    fn ev(n: u32, reel: &str, start: u8) -> EdlEvent {
        EdlEvent::new(
            n,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            tc(start),
            tc(start + 5),
            tc(start),
            tc(start + 5),
        )
    }

    #[test]
    fn test_to_csv_has_header() {
        let csv = TcListExporter::to_csv(&[]);
        assert!(csv.starts_with("event_number,reel,source_in,source_out,record_in,record_out\n"));
    }

    #[test]
    fn test_to_csv_single_event() {
        let csv = TcListExporter::to_csv(&[ev(1, "A001", 0)]);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2, "header + 1 data row");
        assert!(lines[1].starts_with("001,A001,"));
    }

    #[test]
    fn test_to_csv_event_number_zero_padded() {
        let csv = TcListExporter::to_csv(&[ev(7, "R7", 0)]);
        let data = csv.lines().nth(1).expect("data row");
        assert!(data.starts_with("007,"), "number should be zero-padded to 3 digits");
    }

    #[test]
    fn test_to_csv_multiple_events() {
        let events: Vec<_> = (1..=5u32)
            .map(|i| ev(i, "REEL", (i as u8 - 1) * 5))
            .collect();
        let csv = TcListExporter::to_csv(&events);
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 6, "header + 5 data rows");
    }

    #[test]
    fn test_to_csv_contains_reel_name() {
        let csv = TcListExporter::to_csv(&[ev(1, "MYREEL", 0)]);
        assert!(csv.contains("MYREEL"));
    }

    #[test]
    fn test_to_csv_empty_events() {
        let csv = TcListExporter::to_csv(&[]);
        // Only the header line.
        assert_eq!(csv.lines().count(), 1);
    }

    #[test]
    fn test_to_csv_timecodes_present() {
        let csv = TcListExporter::to_csv(&[ev(1, "AX", 10)]);
        // source_in timecode should appear — 01:00:10:00
        assert!(csv.contains("01:00:10:00"), "source_in should be 01:00:10:00; csv = {csv}");
    }

    #[test]
    fn test_to_delimited_tab_separator() {
        let csv = TcListExporter::to_delimited(&[ev(1, "AX", 0)], '\t');
        // Header should use tab.
        let header = csv.lines().next().expect("header");
        assert!(header.contains('\t'));
        assert!(!header.contains(','));
    }

    #[test]
    fn test_to_delimited_pipe_separator() {
        let csv = TcListExporter::to_delimited(&[ev(1, "AX", 0)], '|');
        let data = csv.lines().nth(1).expect("data row");
        assert!(data.contains('|'));
    }

    #[test]
    fn test_to_csv_columns_event_number_and_reel() {
        let columns = [TcColumn::EventNumber, TcColumn::Reel];
        let csv = TcListExporter::to_csv_columns(&[ev(5, "REEL5", 0)], &columns);
        let header = csv.lines().next().expect("header");
        assert_eq!(header, "event_number,reel");
        let data = csv.lines().nth(1).expect("data row");
        assert_eq!(data, "005,REEL5");
    }

    #[test]
    fn test_to_csv_columns_duration_frames() {
        let columns = [TcColumn::EventNumber, TcColumn::DurationFrames];
        // ev(n, reel, start) produces 5-second events at 25fps = 125 frames
        let csv = TcListExporter::to_csv_columns(&[ev(1, "AX", 0)], &columns);
        let data = csv.lines().nth(1).expect("data row");
        // 5 seconds × 25 fps = 125 frames
        assert!(data.contains("125"), "duration should be 125; row = {data}");
    }

    #[test]
    fn test_to_csv_columns_clip_name() {
        let mut event = ev(1, "AX", 0);
        event.set_clip_name("interview.mov".to_string());

        let columns = [TcColumn::ClipName];
        let csv = TcListExporter::to_csv_columns(&[event], &columns);
        let data = csv.lines().nth(1).expect("data row");
        assert_eq!(data, "interview.mov");
    }

    #[test]
    fn test_to_csv_columns_clip_name_absent() {
        let columns = [TcColumn::ClipName];
        let csv = TcListExporter::to_csv_columns(&[ev(1, "AX", 0)], &columns);
        let data = csv.lines().nth(1).expect("data row");
        assert_eq!(data, "", "absent clip name should export as empty string");
    }

    #[test]
    fn test_to_reel_list_deduplicates() {
        let events = vec![
            ev(1, "A001", 0),
            ev(2, "A002", 5),
            ev(3, "A001", 10), // duplicate A001
        ];
        let reels = TcListExporter::to_reel_list(&events);
        assert_eq!(reels.len(), 2);
        assert_eq!(reels[0], "A001");
        assert_eq!(reels[1], "A002");
    }

    #[test]
    fn test_to_reel_list_empty() {
        let reels = TcListExporter::to_reel_list(&[]);
        assert!(reels.is_empty());
    }

    #[test]
    fn test_reel_event_counts_sorted_descending() {
        let events = vec![
            ev(1, "A001", 0),
            ev(2, "A002", 5),
            ev(3, "A001", 10),
            ev(4, "A001", 15),
            ev(5, "A002", 20),
        ];
        let counts = TcListExporter::reel_event_counts(&events);
        // A001 appears 3 times, A002 appears 2 times.
        assert_eq!(counts[0].0, "A001");
        assert_eq!(counts[0].1, 3);
        assert_eq!(counts[1].0, "A002");
        assert_eq!(counts[1].1, 2);
    }

    #[test]
    fn test_tc_column_header_names() {
        assert_eq!(TcColumn::EventNumber.header(), "event_number");
        assert_eq!(TcColumn::Reel.header(), "reel");
        assert_eq!(TcColumn::SourceIn.header(), "source_in");
        assert_eq!(TcColumn::SourceOut.header(), "source_out");
        assert_eq!(TcColumn::RecordIn.header(), "record_in");
        assert_eq!(TcColumn::RecordOut.header(), "record_out");
        assert_eq!(TcColumn::ClipName.header(), "clip_name");
        assert_eq!(TcColumn::DurationFrames.header(), "duration_frames");
    }

    #[test]
    fn test_to_csv_columns_all_standard_columns() {
        let columns = [
            TcColumn::EventNumber,
            TcColumn::Reel,
            TcColumn::SourceIn,
            TcColumn::SourceOut,
            TcColumn::RecordIn,
            TcColumn::RecordOut,
        ];
        let csv = TcListExporter::to_csv_columns(&[ev(1, "AX", 0)], &columns);
        let header = csv.lines().next().expect("header");
        assert_eq!(header, "event_number,reel,source_in,source_out,record_in,record_out");
        let lines: Vec<&str> = csv.lines().collect();
        assert_eq!(lines.len(), 2);
    }
}
