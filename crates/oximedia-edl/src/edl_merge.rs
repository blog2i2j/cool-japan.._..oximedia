#![allow(dead_code)]
//! EDL merging utilities for combining multiple EDLs into one.
//!
//! This module supports several merge strategies:
//! - **Append**: place the second EDL after the first on the timeline.
//! - **Interleave**: interleave events by their record-in timecodes.
//! - **Replace**: replace matching events in a base EDL with events from an overlay.
//! - **Union**: combine all events de-duplicating by event number.

use crate::event::EdlEvent;
use crate::timecode::EdlFrameRate;
use crate::{Edl, EdlFormat};
use std::collections::HashSet;

/// Strategy for merging two EDLs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategy {
    /// Append the second EDL after the first; renumber and offset record timecodes.
    Append,
    /// Interleave events from both EDLs by record-in timecode.
    Interleave,
    /// Replace events in the base EDL with events from the overlay that share the same number.
    Replace,
    /// Union of events from both EDLs; first wins on duplicate numbers.
    Union,
}

/// Options controlling the merge operation.
#[derive(Debug, Clone)]
pub struct MergeOptions {
    /// Merge strategy.
    pub strategy: MergeStrategy,
    /// Whether to renumber events after merging.
    pub renumber: bool,
    /// Whether to sort events by record-in after merging.
    pub sort_by_record_in: bool,
    /// Frame rate for the merged output (uses the first EDL's rate if `None`).
    pub frame_rate: Option<EdlFrameRate>,
    /// Title for the merged EDL (auto-generated if `None`).
    pub title: Option<String>,
}

impl Default for MergeOptions {
    fn default() -> Self {
        Self {
            strategy: MergeStrategy::Append,
            renumber: true,
            sort_by_record_in: true,
            frame_rate: None,
            title: None,
        }
    }
}

impl MergeOptions {
    /// Create merge options with the given strategy.
    #[must_use]
    pub fn with_strategy(strategy: MergeStrategy) -> Self {
        Self {
            strategy,
            ..Self::default()
        }
    }

    /// Set whether to renumber events.
    #[must_use]
    pub fn renumber(mut self, value: bool) -> Self {
        self.renumber = value;
        self
    }

    /// Set whether to sort events by record-in.
    #[must_use]
    pub fn sort_by_record_in(mut self, value: bool) -> Self {
        self.sort_by_record_in = value;
        self
    }

    /// Set the output frame rate.
    #[must_use]
    pub fn frame_rate(mut self, rate: EdlFrameRate) -> Self {
        self.frame_rate = Some(rate);
        self
    }

    /// Set the output title.
    #[must_use]
    pub fn title(mut self, title: impl Into<String>) -> Self {
        self.title = Some(title.into());
        self
    }
}

/// Merge result with statistics.
#[derive(Debug, Clone)]
pub struct MergeResult {
    /// The merged EDL.
    pub edl: Edl,
    /// Number of events contributed by the first EDL.
    pub from_first: usize,
    /// Number of events contributed by the second EDL.
    pub from_second: usize,
    /// Number of events that were replaced (Replace strategy).
    pub replaced: usize,
    /// Number of duplicate events skipped (Union strategy).
    pub duplicates_skipped: usize,
}

impl MergeResult {
    /// Total number of events in the merged EDL.
    #[must_use]
    pub fn total_events(&self) -> usize {
        self.edl.events.len()
    }
}

/// Merge two EDLs according to the given options.
///
/// # Arguments
///
/// * `first` - The first (base) EDL.
/// * `second` - The second (overlay / appended) EDL.
/// * `options` - Merge configuration.
#[must_use]
pub fn merge_edls(first: &Edl, second: &Edl, options: &MergeOptions) -> MergeResult {
    let frame_rate = options.frame_rate.unwrap_or(first.frame_rate);
    let format = first.format;
    let title = options.title.clone().or_else(|| {
        let t1 = first.title.as_deref().unwrap_or("EDL1");
        let t2 = second.title.as_deref().unwrap_or("EDL2");
        Some(format!("{t1} + {t2}"))
    });

    let mut result = MergeResult {
        edl: Edl::new(format),
        from_first: 0,
        from_second: 0,
        replaced: 0,
        duplicates_skipped: 0,
    };

    result.edl.set_frame_rate(frame_rate);
    if let Some(t) = title {
        result.edl.set_title(t);
    }

    match options.strategy {
        MergeStrategy::Append => merge_append(first, second, &mut result),
        MergeStrategy::Interleave => merge_interleave(first, second, &mut result),
        MergeStrategy::Replace => merge_replace(first, second, &mut result),
        MergeStrategy::Union => merge_union(first, second, &mut result),
    }

    if options.sort_by_record_in {
        result.edl.events.sort_by_key(|e| e.record_in.to_frames());
    }

    if options.renumber {
        result.edl.renumber_events();
    }

    result
}

/// Append strategy: copy all events from first, then all from second.
fn merge_append(first: &Edl, second: &Edl, result: &mut MergeResult) {
    for e in &first.events {
        result.edl.events.push(e.clone());
        result.from_first += 1;
    }
    for e in &second.events {
        result.edl.events.push(e.clone());
        result.from_second += 1;
    }
}

/// Interleave strategy: merge events sorted by record-in timecode.
fn merge_interleave(first: &Edl, second: &Edl, result: &mut MergeResult) {
    let mut all: Vec<(usize, &EdlEvent)> = first
        .events
        .iter()
        .map(|e| (0_usize, e))
        .chain(second.events.iter().map(|e| (1_usize, e)))
        .collect();

    all.sort_by_key(|(_, e)| e.record_in.to_frames());

    for (source, e) in all {
        result.edl.events.push(e.clone());
        if source == 0 {
            result.from_first += 1;
        } else {
            result.from_second += 1;
        }
    }
}

/// Replace strategy: start with first, replace matching event numbers from second.
fn merge_replace(first: &Edl, second: &Edl, result: &mut MergeResult) {
    let overlay_numbers: HashSet<u32> = second.events.iter().map(|e| e.number).collect();

    for e in &first.events {
        if overlay_numbers.contains(&e.number) {
            // Will be replaced by second's event
            if let Some(replacement) = second.events.iter().find(|s| s.number == e.number) {
                result.edl.events.push(replacement.clone());
                result.replaced += 1;
                result.from_second += 1;
            }
        } else {
            result.edl.events.push(e.clone());
            result.from_first += 1;
        }
    }

    // Add any events from second that are not in first
    let first_numbers: HashSet<u32> = first.events.iter().map(|e| e.number).collect();
    for e in &second.events {
        if !first_numbers.contains(&e.number) {
            result.edl.events.push(e.clone());
            result.from_second += 1;
        }
    }
}

/// Union strategy: include all unique event numbers, first wins on duplicates.
fn merge_union(first: &Edl, second: &Edl, result: &mut MergeResult) {
    let mut seen: HashSet<u32> = HashSet::new();

    for e in &first.events {
        seen.insert(e.number);
        result.edl.events.push(e.clone());
        result.from_first += 1;
    }

    for e in &second.events {
        if seen.contains(&e.number) {
            result.duplicates_skipped += 1;
        } else {
            seen.insert(e.number);
            result.edl.events.push(e.clone());
            result.from_second += 1;
        }
    }
}

/// Convenience function: merge multiple EDLs sequentially using the Append strategy.
#[must_use]
pub fn merge_many(edls: &[&Edl], options: &MergeOptions) -> Edl {
    if edls.is_empty() {
        return Edl::new(EdlFormat::Cmx3600);
    }
    if edls.len() == 1 {
        return edls[0].clone();
    }

    let mut merged = edls[0].clone();
    for edl in &edls[1..] {
        let res = merge_edls(&merged, edl, options);
        merged = res.edl;
    }
    merged
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::{EditType, TrackType};
    use crate::timecode::{EdlFrameRate, EdlTimecode};
    use crate::{Edl, EdlFormat};

    fn make_event(num: u32, reel: &str, sec_in: u8, sec_out: u8) -> EdlEvent {
        let fr = EdlFrameRate::Fps25;
        EdlEvent::new(
            num,
            reel.to_string(),
            TrackType::Video,
            EditType::Cut,
            EdlTimecode::new(1, 0, sec_in, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, sec_out, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, sec_in, 0, fr).expect("failed to create"),
            EdlTimecode::new(1, 0, sec_out, 0, fr).expect("failed to create"),
        )
    }

    fn make_edl(title: &str, events: Vec<EdlEvent>) -> Edl {
        let mut edl = Edl::new(EdlFormat::Cmx3600);
        edl.set_title(title.to_string());
        edl.set_frame_rate(EdlFrameRate::Fps25);
        for e in events {
            edl.events.push(e);
        }
        edl
    }

    #[test]
    fn test_append_basic() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(2, "R2", 5, 10)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Append);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 2);
        assert_eq!(result.from_first, 1);
        assert_eq!(result.from_second, 1);
    }

    #[test]
    fn test_append_renumber() {
        let a = make_edl("A", vec![make_event(10, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(20, "R2", 5, 10)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Append).renumber(true);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.edl.events[0].number, 1);
        assert_eq!(result.edl.events[1].number, 2);
    }

    #[test]
    fn test_interleave() {
        let a = make_edl("A", vec![make_event(1, "R1", 10, 15)]);
        let b = make_edl("B", vec![make_event(2, "R2", 0, 5)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Interleave);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 2);
        // After sort, event from b (sec 0-5) should come first
        assert_eq!(result.edl.events[0].reel, "R2");
    }

    #[test]
    fn test_replace_matching() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(1, "R1_REPLACED", 0, 5)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Replace);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 1);
        assert_eq!(result.replaced, 1);
        assert_eq!(result.edl.events[0].reel, "R1_REPLACED");
    }

    #[test]
    fn test_replace_adds_new() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(2, "R2", 5, 10)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Replace);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 2);
    }

    #[test]
    fn test_union_first_wins() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(1, "R1_DUP", 0, 5)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Union);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 1);
        assert_eq!(result.duplicates_skipped, 1);
        assert_eq!(result.edl.events[0].reel, "R1");
    }

    #[test]
    fn test_union_adds_unique() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(2, "R2", 5, 10)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Union);
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.total_events(), 2);
    }

    #[test]
    fn test_merge_title_auto() {
        let a = make_edl("Reel_A", vec![]);
        let b = make_edl("Reel_B", vec![]);
        let opts = MergeOptions::default();
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.edl.title.as_deref(), Some("Reel_A + Reel_B"));
    }

    #[test]
    fn test_merge_title_custom() {
        let a = make_edl("A", vec![]);
        let b = make_edl("B", vec![]);
        let opts = MergeOptions::default().title("Custom Title");
        let result = merge_edls(&a, &b, &opts);
        assert_eq!(result.edl.title.as_deref(), Some("Custom Title"));
    }

    #[test]
    fn test_merge_many_empty() {
        let edls: Vec<&Edl> = vec![];
        let opts = MergeOptions::default();
        let merged = merge_many(&edls, &opts);
        assert!(merged.events.is_empty());
    }

    #[test]
    fn test_merge_many_single() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let opts = MergeOptions::default();
        let merged = merge_many(&[&a], &opts);
        assert_eq!(merged.events.len(), 1);
    }

    #[test]
    fn test_merge_many_multiple() {
        let a = make_edl("A", vec![make_event(1, "R1", 0, 5)]);
        let b = make_edl("B", vec![make_event(2, "R2", 5, 10)]);
        let c = make_edl("C", vec![make_event(3, "R3", 10, 15)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Append);
        let merged = merge_many(&[&a, &b, &c], &opts);
        assert_eq!(merged.events.len(), 3);
    }

    #[test]
    fn test_no_sort_option() {
        let a = make_edl("A", vec![make_event(1, "R1", 10, 15)]);
        let b = make_edl("B", vec![make_event(2, "R2", 0, 5)]);
        let opts = MergeOptions::with_strategy(MergeStrategy::Append)
            .sort_by_record_in(false)
            .renumber(false);
        let result = merge_edls(&a, &b, &opts);
        // Without sorting, first EDL's event stays first
        assert_eq!(result.edl.events[0].number, 1);
    }
}
