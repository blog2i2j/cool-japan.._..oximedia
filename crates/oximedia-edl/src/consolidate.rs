//! EDL consolidation utilities.
//!
//! Provides tools to consolidate an EDL by removing duplicate events,
//! merging adjacent edits from the same source reel, and simplifying
//! transitions to produce a cleaner, optimised edit decision list.

#![allow(dead_code)]

use crate::event::{EditType, EdlEvent};

// ────────────────────────────────────────────────────────────────────────────
// ConsolidationOptions
// ────────────────────────────────────────────────────────────────────────────

/// Options controlling the consolidation pass.
#[derive(Debug, Clone)]
pub struct ConsolidationOptions {
    /// Remove events whose source duration is zero.
    pub remove_zero_duration: bool,
    /// Merge adjacent cut events from the same reel when the record timecodes
    /// are contiguous and the source timecodes are also contiguous.
    pub merge_adjacent_cuts: bool,
    /// Renumber events sequentially after all changes.
    pub renumber: bool,
}

impl Default for ConsolidationOptions {
    fn default() -> Self {
        Self {
            remove_zero_duration: true,
            merge_adjacent_cuts: true,
            renumber: true,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// ConsolidationReport
// ────────────────────────────────────────────────────────────────────────────

/// Summary of changes made during consolidation.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationReport {
    /// Number of zero-duration events removed.
    pub zero_duration_removed: usize,
    /// Number of duplicate events removed.
    pub duplicates_removed: usize,
    /// Number of event pairs merged.
    pub merges_performed: usize,
    /// Number of events after consolidation.
    pub final_event_count: usize,
}

impl ConsolidationReport {
    /// Total number of events removed or merged.
    #[must_use]
    pub fn total_changes(&self) -> usize {
        self.zero_duration_removed + self.duplicates_removed + self.merges_performed
    }

    /// Returns `true` if no changes were made.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.total_changes() == 0
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Consolidator
// ────────────────────────────────────────────────────────────────────────────

/// Consolidates an EDL in-place, returning a report of the changes.
#[derive(Debug, Default)]
pub struct Consolidator {
    options: ConsolidationOptions,
}

impl Consolidator {
    /// Create a new `Consolidator` with the given options.
    #[must_use]
    pub fn new(options: ConsolidationOptions) -> Self {
        Self { options }
    }

    /// Run the consolidation pass over `events`, returning a report.
    #[must_use]
    pub fn consolidate(&self, events: &mut Vec<EdlEvent>) -> ConsolidationReport {
        let mut report = ConsolidationReport::default();

        // Step 1: Remove zero-duration events.
        if self.options.remove_zero_duration {
            let before = events.len();
            events.retain(|e| e.duration_frames() > 0);
            report.zero_duration_removed = before - events.len();
        }

        // Step 2: Remove exact duplicates (same event number).
        {
            let before = events.len();
            let mut seen = std::collections::HashSet::new();
            events.retain(|e| seen.insert(e.number));
            report.duplicates_removed = before - events.len();
        }

        // Step 3: Merge adjacent cut events from the same reel.
        if self.options.merge_adjacent_cuts {
            report.merges_performed = merge_adjacent_cuts(events);
        }

        // Step 4: Renumber.
        if self.options.renumber {
            for (i, event) in events.iter_mut().enumerate() {
                event.number = (i + 1) as u32;
            }
        }

        report.final_event_count = events.len();
        report
    }
}

/// Merge adjacent cut events from the same reel where source and record
/// timecodes are contiguous.  Returns the number of merges performed.
fn merge_adjacent_cuts(events: &mut Vec<EdlEvent>) -> usize {
    let mut merges = 0usize;
    let mut i = 0;

    while i + 1 < events.len() {
        let can_merge = events[i].edit_type == EditType::Cut
            && events[i + 1].edit_type == EditType::Cut
            && events[i].reel == events[i + 1].reel
            && events[i].source_out.to_frames() == events[i + 1].source_in.to_frames()
            && events[i].record_out.to_frames() == events[i + 1].record_in.to_frames();

        if can_merge {
            // Extend event[i] to cover both spans.
            events[i].source_out = events[i + 1].source_out;
            events[i].record_out = events[i + 1].record_out;
            events.remove(i + 1);
            merges += 1;
            // Do not advance i — check if the newly extended event can also
            // be merged with the next one.
        } else {
            i += 1;
        }
    }

    merges
}

// ────────────────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────────────────

/// Return `true` if `events` is sorted by record-in timecode.
#[must_use]
pub fn is_sorted_by_record_in(events: &[EdlEvent]) -> bool {
    events
        .windows(2)
        .all(|w| w[0].record_in.to_frames() <= w[1].record_in.to_frames())
}

/// Return the total source duration in frames across all events.
#[must_use]
pub fn total_source_frames(events: &[EdlEvent]) -> u64 {
    events.iter().map(|e| e.duration_frames()).sum()
}

/// Find events whose record ranges overlap.
#[must_use]
pub fn find_overlapping_events(events: &[EdlEvent]) -> Vec<(usize, usize)> {
    let mut overlaps = Vec::new();
    for i in 0..events.len() {
        for j in (i + 1)..events.len() {
            let a_in = events[i].record_in.to_frames();
            let a_out = events[i].record_out.to_frames();
            let b_in = events[j].record_in.to_frames();
            let b_out = events[j].record_out.to_frames();
            if a_in < b_out && b_in < a_out {
                overlaps.push((i, j));
            }
        }
    }
    overlaps
}

// ────────────────────────────────────────────────────────────────────────────
// Tests
// ────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};

    fn tc(h: u8, m: u8, s: u8, f: u8) -> EdlTimecode {
        EdlTimecode::new(h, m, s, f, EdlFrameRate::Fps25).expect("failed to create")
    }

    fn cut(
        num: u32,
        reel: &str,
        s_in: EdlTimecode,
        s_out: EdlTimecode,
        r_in: EdlTimecode,
        r_out: EdlTimecode,
    ) -> EdlEvent {
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            s_in,
            s_out,
            r_in,
            r_out,
        )
    }

    #[test]
    fn test_default_options() {
        let opts = ConsolidationOptions::default();
        assert!(opts.remove_zero_duration);
        assert!(opts.merge_adjacent_cuts);
        assert!(opts.renumber);
    }

    #[test]
    fn test_report_is_clean_when_no_changes() {
        let report = ConsolidationReport::default();
        assert!(report.is_clean());
        assert_eq!(report.total_changes(), 0);
    }

    #[test]
    fn test_remove_zero_duration_events() {
        let mut events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
            // zero duration: source_in == source_out
            cut(
                2,
                "A002",
                tc(1, 0, 5, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 5, 0),
            ),
        ];
        let c = Consolidator::new(ConsolidationOptions::default());
        let report = c.consolidate(&mut events);
        assert_eq!(report.zero_duration_removed, 1);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_remove_duplicate_event_numbers() {
        let mut events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
        ];
        let opts = ConsolidationOptions {
            remove_zero_duration: false,
            merge_adjacent_cuts: false,
            renumber: false,
        };
        let c = Consolidator::new(opts);
        let report = c.consolidate(&mut events);
        assert_eq!(report.duplicates_removed, 1);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn test_merge_adjacent_cuts_same_reel() {
        // Two contiguous cuts from the same reel should be merged.
        let mut events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
            cut(
                2,
                "A001",
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
            ),
        ];
        let c = Consolidator::new(ConsolidationOptions::default());
        let report = c.consolidate(&mut events);
        assert_eq!(report.merges_performed, 1);
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].source_out.to_frames(),
            tc(1, 0, 10, 0).to_frames()
        );
    }

    #[test]
    fn test_no_merge_different_reels() {
        let mut events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
            cut(
                2,
                "B001",
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
            ),
        ];
        let c = Consolidator::new(ConsolidationOptions::default());
        let report = c.consolidate(&mut events);
        assert_eq!(report.merges_performed, 0);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_no_merge_non_contiguous() {
        // Gap between events — no merge.
        let mut events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
            cut(
                2,
                "A001",
                tc(1, 0, 6, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 6, 0),
                tc(1, 0, 10, 0),
            ),
        ];
        let c = Consolidator::new(ConsolidationOptions::default());
        let report = c.consolidate(&mut events);
        assert_eq!(report.merges_performed, 0);
        assert_eq!(events.len(), 2);
    }

    #[test]
    fn test_renumber_after_removal() {
        let mut events = vec![
            cut(
                5,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
            cut(
                10,
                "B001",
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
            ),
        ];
        let c = Consolidator::new(ConsolidationOptions::default());
        let _ = c.consolidate(&mut events);
        assert_eq!(events[0].number, 1);
        assert_eq!(events[1].number, 2);
    }

    #[test]
    fn test_is_sorted_by_record_in_true() {
        let events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
            cut(
                2,
                "B001",
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
            ),
        ];
        assert!(is_sorted_by_record_in(&events));
    }

    #[test]
    fn test_is_sorted_by_record_in_false() {
        let events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
            ),
            cut(
                2,
                "B001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
        ];
        assert!(!is_sorted_by_record_in(&events));
    }

    #[test]
    fn test_total_source_frames() {
        let events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
            cut(
                2,
                "B001",
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
            ),
        ];
        // Each event is 5 seconds × 25 fps = 125 frames; total = 250.
        assert_eq!(total_source_frames(&events), 250);
    }

    #[test]
    fn test_find_overlapping_events() {
        let events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 10, 0),
            ),
            cut(
                2,
                "B001",
                tc(1, 0, 5, 0),
                tc(1, 0, 15, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 15, 0),
            ),
        ];
        let overlaps = find_overlapping_events(&events);
        assert_eq!(overlaps.len(), 1);
        assert_eq!(overlaps[0], (0, 1));
    }

    #[test]
    fn test_find_no_overlaps() {
        let events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
            cut(
                2,
                "B001",
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
            ),
        ];
        let overlaps = find_overlapping_events(&events);
        assert!(overlaps.is_empty());
    }

    #[test]
    fn test_consolidation_report_total_changes() {
        let report = ConsolidationReport {
            zero_duration_removed: 2,
            duplicates_removed: 1,
            merges_performed: 3,
            final_event_count: 10,
        };
        assert_eq!(report.total_changes(), 6);
        assert!(!report.is_clean());
    }

    #[test]
    fn test_empty_edl_no_panic() {
        let mut events: Vec<EdlEvent> = vec![];
        let c = Consolidator::new(ConsolidationOptions::default());
        let report = c.consolidate(&mut events);
        assert_eq!(report.final_event_count, 0);
        assert!(report.is_clean());
    }

    #[test]
    fn test_triple_merge_chain() {
        // Three contiguous cuts → two merges → one event.
        let mut events = vec![
            cut(
                1,
                "A001",
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 0, 0),
                tc(1, 0, 5, 0),
            ),
            cut(
                2,
                "A001",
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 5, 0),
                tc(1, 0, 10, 0),
            ),
            cut(
                3,
                "A001",
                tc(1, 0, 10, 0),
                tc(1, 0, 15, 0),
                tc(1, 0, 10, 0),
                tc(1, 0, 15, 0),
            ),
        ];
        let c = Consolidator::new(ConsolidationOptions::default());
        let report = c.consolidate(&mut events);
        assert_eq!(report.merges_performed, 2);
        assert_eq!(events.len(), 1);
        assert_eq!(
            events[0].source_out.to_frames(),
            tc(1, 0, 15, 0).to_frames()
        );
    }
}
