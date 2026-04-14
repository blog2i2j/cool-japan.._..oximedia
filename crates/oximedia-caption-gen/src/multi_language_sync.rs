//! Multi-language caption synchronization.
//!
//! This module provides anchor-point-based synchronization of caption tracks
//! in different languages.  Anchor points are shared time positions (e.g. scene
//! cuts, chapter boundaries, or manually placed cue points) that act as
//! synchronization constraints across language tracks.
//!
//! ## Approach
//!
//! 1. Define [`SyncAnchor`] points shared by all tracks.
//! 2. Each language track provides a [`LangTrack`] with its own [`CaptionBlock`]
//!    sequence and an ISO 639-1 language code.
//! 3. [`MultiLangSyncer`] aligns all tracks to the anchor points by
//!    proportionally stretching or compressing block timings within each
//!    inter-anchor segment.
//! 4. The result is a [`SyncedMultiLangOutput`] holding adjusted tracks.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_caption_gen::multi_language_sync::{
//!     SyncAnchor, LangTrack, MultiLangSyncer, MultiLangSyncConfig,
//! };
//! use oximedia_caption_gen::{CaptionBlock, CaptionPosition};
//!
//! let blocks_en = vec![CaptionBlock {
//!     id: 1, start_ms: 0, end_ms: 2000,
//!     lines: vec!["Hello.".to_string()],
//!     speaker_id: None, position: CaptionPosition::Bottom,
//! }];
//! let blocks_fr = vec![CaptionBlock {
//!     id: 1, start_ms: 50, end_ms: 2100,
//!     lines: vec!["Bonjour.".to_string()],
//!     speaker_id: None, position: CaptionPosition::Bottom,
//! }];
//!
//! let anchors = vec![SyncAnchor { time_ms: 0 }, SyncAnchor { time_ms: 5000 }];
//! let tracks = vec![
//!     LangTrack::new("en", blocks_en),
//!     LangTrack::new("fr", blocks_fr),
//! ];
//! let syncer = MultiLangSyncer::new(MultiLangSyncConfig::default());
//! let output = syncer.sync(&tracks, &anchors).unwrap();
//! assert_eq!(output.tracks.len(), 2);
//! ```

use crate::alignment::CaptionBlock;
use crate::CaptionGenError;

// ─── Types ────────────────────────────────────────────────────────────────────

/// A synchronization anchor point shared across all language tracks.
///
/// Anchors are typically placed at scene boundaries, chapter markers, or
/// manually curated synchronization cue points.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct SyncAnchor {
    /// Absolute time position in milliseconds in the *reference* timeline.
    pub time_ms: u64,
}

/// A caption track for a single language.
#[derive(Debug, Clone)]
pub struct LangTrack {
    /// ISO 639-1 two-letter language code (e.g. "en", "fr", "de").
    pub lang: String,
    /// Ordered caption blocks for this language.
    pub blocks: Vec<CaptionBlock>,
}

impl LangTrack {
    /// Create a new language track.
    pub fn new(lang: &str, blocks: Vec<CaptionBlock>) -> Self {
        Self {
            lang: lang.to_string(),
            blocks,
        }
    }

    /// Total character count across all blocks in this track.
    pub fn total_chars(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|b| b.lines.iter())
            .map(|l| l.chars().count())
            .sum()
    }
}

/// Configuration for the multi-language synchronizer.
#[derive(Debug, Clone)]
pub struct MultiLangSyncConfig {
    /// Maximum time stretch/compression ratio allowed per segment (e.g. 1.5
    /// means timings may expand by at most 50 % or compress to at most 1/1.5
    /// of their original duration).
    pub max_stretch_ratio: f64,
    /// Minimum block duration after adjustment in milliseconds.
    pub min_block_duration_ms: u64,
    /// When `true`, blocks that fall entirely outside any anchor segment are
    /// kept as-is rather than dropped.
    pub keep_out_of_range_blocks: bool,
}

impl Default for MultiLangSyncConfig {
    fn default() -> Self {
        Self {
            max_stretch_ratio: 2.0,
            min_block_duration_ms: 100,
            keep_out_of_range_blocks: true,
        }
    }
}

/// A single adjusted language track in the synchronized output.
#[derive(Debug, Clone)]
pub struct SyncedLangTrack {
    /// Language code.
    pub lang: String,
    /// Adjusted caption blocks aligned to anchor points.
    pub blocks: Vec<CaptionBlock>,
    /// Number of blocks whose duration was clamped by `max_stretch_ratio`.
    pub clamp_count: usize,
}

/// Output of a multi-language synchronization run.
#[derive(Debug, Clone)]
pub struct SyncedMultiLangOutput {
    /// One entry per input [`LangTrack`], in the same order.
    pub tracks: Vec<SyncedLangTrack>,
    /// The anchor points used for synchronization.
    pub anchors: Vec<SyncAnchor>,
}

impl SyncedMultiLangOutput {
    /// Find the synced track for a given language code.
    pub fn track_for_lang(&self, lang: &str) -> Option<&SyncedLangTrack> {
        self.tracks.iter().find(|t| t.lang == lang)
    }

    /// Return captions from all tracks that are active at `time_ms`.
    pub fn active_at(&self, time_ms: u64) -> Vec<(&str, &CaptionBlock)> {
        self.tracks
            .iter()
            .flat_map(|t| {
                t.blocks
                    .iter()
                    .filter(move |b| b.start_ms <= time_ms && b.end_ms > time_ms)
                    .map(move |b| (t.lang.as_str(), b))
            })
            .collect()
    }
}

// ─── Synchronizer ─────────────────────────────────────────────────────────────

/// Synchronizes multiple language caption tracks to shared anchor points.
pub struct MultiLangSyncer {
    config: MultiLangSyncConfig,
}

impl MultiLangSyncer {
    /// Create a new syncer with the given configuration.
    pub fn new(config: MultiLangSyncConfig) -> Self {
        Self { config }
    }

    /// Synchronize all language tracks to the provided anchor points.
    ///
    /// # Errors
    ///
    /// - [`CaptionGenError::InvalidParameter`] if fewer than 2 anchors are
    ///   provided (at least a start and end anchor are required).
    /// - [`CaptionGenError::EmptyTranscript`] if `tracks` is empty.
    pub fn sync(
        &self,
        tracks: &[LangTrack],
        anchors: &[SyncAnchor],
    ) -> Result<SyncedMultiLangOutput, CaptionGenError> {
        if anchors.len() < 2 {
            return Err(CaptionGenError::InvalidParameter(
                "at least 2 anchor points are required for synchronization".to_string(),
            ));
        }
        if tracks.is_empty() {
            return Err(CaptionGenError::EmptyTranscript);
        }

        // Sort anchors ascending.
        let mut sorted_anchors = anchors.to_vec();
        sorted_anchors.sort();
        sorted_anchors.dedup_by_key(|a| a.time_ms);

        let mut synced_tracks = Vec::with_capacity(tracks.len());
        for track in tracks {
            let synced = self.sync_track(track, &sorted_anchors)?;
            synced_tracks.push(synced);
        }

        Ok(SyncedMultiLangOutput {
            tracks: synced_tracks,
            anchors: sorted_anchors,
        })
    }

    /// Synchronize a single language track to the anchor timeline.
    fn sync_track(
        &self,
        track: &LangTrack,
        anchors: &[SyncAnchor],
    ) -> Result<SyncedLangTrack, CaptionGenError> {
        let mut adjusted = Vec::with_capacity(track.blocks.len());
        let mut clamp_count = 0usize;

        for block in &track.blocks {
            // Identify which anchor segment the block's midpoint falls in.
            let mid = block.start_ms + block.duration_ms() / 2;

            let segment = find_anchor_segment(anchors, mid);

            match segment {
                Some((seg_start, seg_end)) => {
                    let seg_duration = seg_end - seg_start;
                    if seg_duration == 0 {
                        // Degenerate segment — keep block as-is.
                        adjusted.push(block.clone());
                        continue;
                    }

                    // Map the block's start and end into the anchor segment using
                    // a linear affine transform within [seg_start, seg_end].
                    let (new_start, new_end, clamped) = self.remap_block(block, seg_start, seg_end);
                    if clamped {
                        clamp_count += 1;
                    }

                    let mut new_block = block.clone();
                    new_block.start_ms = new_start;
                    new_block.end_ms = new_end;
                    adjusted.push(new_block);
                }
                None => {
                    if self.config.keep_out_of_range_blocks {
                        adjusted.push(block.clone());
                    }
                }
            }
        }

        Ok(SyncedLangTrack {
            lang: track.lang.clone(),
            blocks: adjusted,
            clamp_count,
        })
    }

    /// Remap a block's timestamps into a reference anchor segment.
    ///
    /// The block's start and end are stretched/compressed proportionally within
    /// `[seg_start, seg_end]`.  If the stretch ratio exceeds `max_stretch_ratio`,
    /// the duration is clamped and the function returns `(start, end, true)`.
    fn remap_block(&self, block: &CaptionBlock, seg_start: u64, seg_end: u64) -> (u64, u64, bool) {
        let seg_duration = (seg_end - seg_start) as f64;

        // Clamp block times into segment bounds.
        let clamped_start = block.start_ms.max(seg_start).min(seg_end);
        let clamped_end = block.end_ms.max(seg_start).min(seg_end);

        // Position within segment [0.0, 1.0].
        let t_start = (clamped_start - seg_start) as f64 / seg_duration;
        let t_end = (clamped_end - seg_start) as f64 / seg_duration;

        let orig_duration = block.duration_ms() as f64;
        let new_duration_raw = (t_end - t_start) * seg_duration;

        // Check stretch ratio.
        let ratio = if orig_duration > 0.0 {
            new_duration_raw / orig_duration
        } else {
            1.0
        };

        let clamped = ratio > self.config.max_stretch_ratio
            || (orig_duration > 0.0 && ratio < 1.0 / self.config.max_stretch_ratio);

        let new_start = seg_start + (t_start * seg_duration).round() as u64;
        let new_end_unclamped = seg_start + (t_end * seg_duration).round() as u64;

        // Ensure minimum block duration.
        let new_end = new_end_unclamped.max(new_start + self.config.min_block_duration_ms);
        let new_end = new_end.min(seg_end);

        (new_start, new_end, clamped)
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/// Find the anchor segment `[anchors[i].time_ms, anchors[i+1].time_ms)` that
/// contains `time_ms`.  Returns `None` if `time_ms` is outside all segments.
fn find_anchor_segment(anchors: &[SyncAnchor], time_ms: u64) -> Option<(u64, u64)> {
    for pair in anchors.windows(2) {
        let start = pair[0].time_ms;
        let end = pair[1].time_ms;
        if time_ms >= start && time_ms < end {
            return Some((start, end));
        }
    }
    // Include the last anchor point itself (closed right boundary).
    if let (Some(last), Some(second_last)) = (anchors.last(), anchors.iter().rev().nth(1)) {
        if time_ms == last.time_ms {
            return Some((second_last.time_ms, last.time_ms));
        }
    }
    None
}

/// Merge blocks from two synced tracks by interleaving them chronologically.
///
/// The `primary` track's blocks appear first at any given timestamp; ties
/// are broken by track order.
pub fn interleave_tracks<'a>(
    primary: &'a SyncedLangTrack,
    secondary: &'a SyncedLangTrack,
) -> Vec<(/* lang */ &'a str, &'a CaptionBlock)> {
    let mut entries: Vec<(&str, &CaptionBlock)> = Vec::new();
    for b in &primary.blocks {
        entries.push((primary.lang.as_str(), b));
    }
    for b in &secondary.blocks {
        entries.push((secondary.lang.as_str(), b));
    }
    entries.sort_by_key(|(_, b)| b.start_ms);
    entries
}

/// Compute the average time offset (in ms) between corresponding blocks in two
/// synced tracks.  Returns `None` if either track is empty or the tracks have
/// different lengths.
pub fn average_timing_offset(a: &SyncedLangTrack, b: &SyncedLangTrack) -> Option<f64> {
    if a.blocks.is_empty() || b.blocks.is_empty() || a.blocks.len() != b.blocks.len() {
        return None;
    }
    let total: i64 = a
        .blocks
        .iter()
        .zip(b.blocks.iter())
        .map(|(ba, bb)| ba.start_ms as i64 - bb.start_ms as i64)
        .sum();
    Some(total as f64 / a.blocks.len() as f64)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alignment::CaptionPosition;

    fn make_block(id: u32, start_ms: u64, end_ms: u64, text: &str) -> CaptionBlock {
        CaptionBlock {
            id,
            start_ms,
            end_ms,
            lines: vec![text.to_string()],
            speaker_id: None,
            position: CaptionPosition::Bottom,
        }
    }

    fn make_anchors(times: &[u64]) -> Vec<SyncAnchor> {
        times.iter().map(|&t| SyncAnchor { time_ms: t }).collect()
    }

    #[test]
    fn sync_single_segment_identity() {
        // With two anchors spanning the whole track and blocks already aligned,
        // the output should be identical to the input.
        let blocks = vec![make_block(1, 0, 2000, "Hello.")];
        let track = LangTrack::new("en", blocks.clone());
        let anchors = make_anchors(&[0, 10_000]);
        let syncer = MultiLangSyncer::new(MultiLangSyncConfig::default());
        let out = syncer.sync(&[track], &anchors).unwrap();
        assert_eq!(out.tracks.len(), 1);
        assert_eq!(out.tracks[0].blocks[0].start_ms, 0);
        assert_eq!(out.tracks[0].blocks[0].end_ms, 2000);
    }

    #[test]
    fn requires_at_least_two_anchors() {
        let track = LangTrack::new("en", vec![make_block(1, 0, 1000, "Test")]);
        let syncer = MultiLangSyncer::new(MultiLangSyncConfig::default());
        let result = syncer.sync(&[track], &make_anchors(&[0]));
        assert!(result.is_err());
    }

    #[test]
    fn empty_tracks_returns_error() {
        let syncer = MultiLangSyncer::new(MultiLangSyncConfig::default());
        let result = syncer.sync(&[], &make_anchors(&[0, 5000]));
        assert!(result.is_err());
    }

    #[test]
    fn two_language_tracks_synced() {
        let blocks_en = vec![make_block(1, 100, 2000, "Hello.")];
        let blocks_fr = vec![make_block(1, 200, 2200, "Bonjour.")];
        let tracks = vec![
            LangTrack::new("en", blocks_en),
            LangTrack::new("fr", blocks_fr),
        ];
        let anchors = make_anchors(&[0, 10_000]);
        let syncer = MultiLangSyncer::new(MultiLangSyncConfig::default());
        let out = syncer.sync(&tracks, &anchors).unwrap();
        assert_eq!(out.tracks.len(), 2);
        assert_eq!(out.track_for_lang("en").unwrap().lang, "en");
        assert_eq!(out.track_for_lang("fr").unwrap().lang, "fr");
    }

    #[test]
    fn active_at_returns_correct_blocks() {
        let blocks_en = vec![make_block(1, 0, 3000, "Hello.")];
        let blocks_fr = vec![make_block(1, 500, 2500, "Bonjour.")];
        let tracks = vec![
            LangTrack::new("en", blocks_en),
            LangTrack::new("fr", blocks_fr),
        ];
        let anchors = make_anchors(&[0, 10_000]);
        let syncer = MultiLangSyncer::new(MultiLangSyncConfig::default());
        let out = syncer.sync(&tracks, &anchors).unwrap();
        let active = out.active_at(1000);
        assert_eq!(active.len(), 2);
    }

    #[test]
    fn active_at_outside_returns_empty() {
        let blocks_en = vec![make_block(1, 0, 1000, "Hello.")];
        let tracks = vec![LangTrack::new("en", blocks_en)];
        let anchors = make_anchors(&[0, 5000]);
        let syncer = MultiLangSyncer::new(MultiLangSyncConfig::default());
        let out = syncer.sync(&tracks, &anchors).unwrap();
        let active = out.active_at(9000);
        assert!(active.is_empty());
    }

    #[test]
    fn interleave_tracks_sorted_by_start() {
        let t1 = SyncedLangTrack {
            lang: "en".to_string(),
            blocks: vec![make_block(1, 0, 1000, "A"), make_block(2, 3000, 4000, "C")],
            clamp_count: 0,
        };
        let t2 = SyncedLangTrack {
            lang: "fr".to_string(),
            blocks: vec![make_block(1, 1500, 2500, "B")],
            clamp_count: 0,
        };
        let merged = interleave_tracks(&t1, &t2);
        assert_eq!(merged.len(), 3);
        assert!(merged[0].1.start_ms <= merged[1].1.start_ms);
        assert!(merged[1].1.start_ms <= merged[2].1.start_ms);
    }

    #[test]
    fn average_timing_offset_same_tracks_is_zero() {
        let t1 = SyncedLangTrack {
            lang: "en".to_string(),
            blocks: vec![
                make_block(1, 1000, 2000, "A"),
                make_block(2, 3000, 4000, "B"),
            ],
            clamp_count: 0,
        };
        let t2 = SyncedLangTrack {
            lang: "fr".to_string(),
            blocks: vec![
                make_block(1, 1000, 2000, "A"),
                make_block(2, 3000, 4000, "B"),
            ],
            clamp_count: 0,
        };
        let offset = average_timing_offset(&t1, &t2).unwrap();
        assert!((offset).abs() < 1e-6);
    }

    #[test]
    fn average_timing_offset_none_for_different_lengths() {
        let t1 = SyncedLangTrack {
            lang: "en".to_string(),
            blocks: vec![make_block(1, 0, 1000, "A")],
            clamp_count: 0,
        };
        let t2 = SyncedLangTrack {
            lang: "fr".to_string(),
            blocks: vec![make_block(1, 0, 1000, "A"), make_block(2, 1000, 2000, "B")],
            clamp_count: 0,
        };
        assert!(average_timing_offset(&t1, &t2).is_none());
    }

    #[test]
    fn anchors_sorted_on_input() {
        // Provide anchors out of order — syncer should still work.
        let blocks = vec![make_block(1, 1000, 2000, "Hello.")];
        let track = LangTrack::new("en", blocks);
        let anchors = vec![SyncAnchor { time_ms: 5000 }, SyncAnchor { time_ms: 0 }];
        let syncer = MultiLangSyncer::new(MultiLangSyncConfig::default());
        let out = syncer.sync(&[track], &anchors).unwrap();
        // Anchors in output should be sorted.
        let times: Vec<u64> = out.anchors.iter().map(|a| a.time_ms).collect();
        assert!(times.windows(2).all(|w| w[0] <= w[1]));
    }

    #[test]
    fn lang_track_total_chars() {
        let blocks = vec![
            make_block(1, 0, 1000, "Hello"),
            make_block(2, 1000, 2000, "World"),
        ];
        let track = LangTrack::new("en", blocks);
        assert_eq!(track.total_chars(), 10);
    }
}
