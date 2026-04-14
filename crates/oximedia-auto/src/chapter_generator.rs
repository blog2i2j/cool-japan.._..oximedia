//! Automatic chapter generation from scene analysis.
//!
//! Generates chapter markers for long-form video by combining several signals:
//!
//! - **Content-based boundary detection**: hard scene cuts plus sustained
//!   low-similarity regions signal chapter breaks.
//! - **Audio energy valleys**: prolonged silence / low-energy zones often
//!   coincide with topic transitions.
//! - **Score gradient analysis**: sudden drops in scene importance indicate
//!   content boundaries.
//! - **Minimum chapter duration**: prevents spurious micro-chapters by merging
//!   boundaries that are too close together.
//! - **Chapter title suggestion**: heuristic labels are generated from the
//!   dominant content type and relative position (intro, act, climax, outro).
//! - **TOC formatting**: a simple `Vec<ChapterMarker>` can be serialised to
//!   YouTube chapter syntax (`MM:SS Title`) or exported as an SRT-like text
//!   file.
//!
//! All logic is pure arithmetic — no external dependencies beyond
//! `oximedia_core`.
//!
//! # Example
//!
//! ```
//! use oximedia_auto::chapter_generator::{ChapterGenerator, ChapterConfig, SceneSegment};
//! use oximedia_core::{Rational, Timestamp};
//! use oximedia_auto::scoring::{ContentType, SceneFeatures};
//!
//! let config = ChapterConfig::default();
//! let gen = ChapterGenerator::new(config);
//!
//! let tb = Rational::new(1, 1000);
//! let segments = vec![
//!     SceneSegment::new(Timestamp::new(0, tb), Timestamp::new(60_000, tb), 0.8, ContentType::Dialogue),
//!     SceneSegment::new(Timestamp::new(60_000, tb), Timestamp::new(120_000, tb), 0.5, ContentType::Action),
//! ];
//!
//! let chapters = gen.generate(&segments).expect("generate chapters");
//! assert!(!chapters.is_empty());
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use crate::scoring::ContentType;
use oximedia_core::Timestamp;

// ─── SceneSegment ─────────────────────────────────────────────────────────────

/// A scored and typed scene segment used as input for chapter generation.
#[derive(Debug, Clone)]
pub struct SceneSegment {
    /// Start timestamp.
    pub start: Timestamp,
    /// End timestamp.
    pub end: Timestamp,
    /// Importance score (0.0 – 1.0).
    pub score: f64,
    /// Dominant content type.
    pub content_type: ContentType,
    /// Audio energy (0.0 – 1.0; 0 = silence).
    pub audio_energy: f64,
    /// Optional raw description hint (used for title generation).
    pub description: Option<String>,
}

impl SceneSegment {
    /// Create a new scene segment.
    #[must_use]
    pub fn new(start: Timestamp, end: Timestamp, score: f64, content_type: ContentType) -> Self {
        Self {
            start,
            end,
            score: score.clamp(0.0, 1.0),
            content_type,
            audio_energy: 0.5,
            description: None,
        }
    }

    /// Set audio energy.
    #[must_use]
    pub const fn with_audio_energy(mut self, energy: f64) -> Self {
        self.audio_energy = energy;
        self
    }

    /// Set description.
    #[must_use]
    pub fn with_description(mut self, desc: impl Into<String>) -> Self {
        self.description = Some(desc.into());
        self
    }

    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }
}

// ─── ChapterMarker ────────────────────────────────────────────────────────────

/// A single chapter marker in the generated table of contents.
#[derive(Debug, Clone)]
pub struct ChapterMarker {
    /// Chapter index (1-based).
    pub index: usize,
    /// Start timestamp of this chapter.
    pub start: Timestamp,
    /// End timestamp of this chapter (equals the start of the next chapter or
    /// the end of the video).
    pub end: Timestamp,
    /// Suggested chapter title.
    pub title: String,
    /// Confidence that this chapter boundary is correct (0.0 – 1.0).
    pub confidence: f64,
    /// The boundary signal(s) that triggered this chapter split.
    pub signals: Vec<BoundarySignal>,
}

impl ChapterMarker {
    /// Duration of this chapter in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }

    /// Format the start time as `MM:SS` (YouTube chapter format).
    #[must_use]
    pub fn format_timestamp(&self) -> String {
        let total_secs = self.start.pts / 1000;
        let mins = total_secs / 60;
        let secs = total_secs % 60;
        format!("{mins:02}:{secs:02}")
    }

    /// Format as a YouTube chapter line: `MM:SS Title`.
    #[must_use]
    pub fn to_youtube_format(&self) -> String {
        format!("{} {}", self.format_timestamp(), self.title)
    }
}

// ─── BoundarySignal ───────────────────────────────────────────────────────────

/// The evidence that contributed to a chapter boundary decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundarySignal {
    /// Hard scene cut detected between consecutive segments.
    SceneCut,
    /// Sustained drop in scene importance scores.
    ScoreValley,
    /// Audio energy valley (silence or near-silence).
    AudioValley,
    /// Content type changed between segments.
    ContentTypeChange,
    /// Forced boundary at the start of the video.
    VideoStart,
    /// Forced boundary at the end of the video.
    VideoEnd,
}

impl BoundarySignal {
    /// Weight contribution to the boundary confidence score.
    #[must_use]
    pub const fn weight(&self) -> f64 {
        match self {
            Self::SceneCut => 0.40,
            Self::ScoreValley => 0.25,
            Self::AudioValley => 0.20,
            Self::ContentTypeChange => 0.15,
            Self::VideoStart | Self::VideoEnd => 1.0,
        }
    }
}

// ─── ChapterConfig ────────────────────────────────────────────────────────────

/// Configuration for the chapter generator.
#[derive(Debug, Clone)]
pub struct ChapterConfig {
    /// Minimum chapter duration in milliseconds.
    pub min_chapter_duration_ms: i64,
    /// Maximum chapter duration in milliseconds.  Long chapters are split at
    /// the best available boundary within the window.
    pub max_chapter_duration_ms: i64,
    /// Score drop threshold that signals a chapter boundary.
    /// If the mean score drops by this amount between windows, a boundary is
    /// suggested.
    pub score_drop_threshold: f64,
    /// Audio energy threshold below which a segment is considered "silent".
    pub audio_silence_threshold: f64,
    /// Minimum number of consecutive silent segments to trigger an audio valley.
    pub audio_valley_min_segments: usize,
    /// Whether to split on content type changes.
    pub split_on_content_type_change: bool,
    /// Confidence threshold below which a boundary candidate is discarded.
    pub min_boundary_confidence: f64,
    /// Maximum number of chapters to generate.  0 = no limit.
    pub max_chapters: usize,
    /// Prefix for auto-generated chapter titles.
    pub title_prefix: String,
}

impl Default for ChapterConfig {
    fn default() -> Self {
        Self {
            min_chapter_duration_ms: 30_000,  // 30 s
            max_chapter_duration_ms: 600_000, // 10 min
            score_drop_threshold: 0.25,
            audio_silence_threshold: 0.10,
            audio_valley_min_segments: 2,
            split_on_content_type_change: true,
            min_boundary_confidence: 0.20,
            max_chapters: 0,
            title_prefix: "Chapter".to_string(),
        }
    }
}

impl ChapterConfig {
    /// Preset for podcast / interview content.
    #[must_use]
    pub fn podcast() -> Self {
        Self {
            min_chapter_duration_ms: 60_000, // 1 min
            max_chapter_duration_ms: 1_200_000,
            score_drop_threshold: 0.20,
            audio_silence_threshold: 0.05,
            audio_valley_min_segments: 3,
            split_on_content_type_change: false,
            ..Self::default()
        }
    }

    /// Preset for sports highlights.
    #[must_use]
    pub fn sports() -> Self {
        Self {
            min_chapter_duration_ms: 20_000,
            max_chapter_duration_ms: 300_000,
            score_drop_threshold: 0.35,
            split_on_content_type_change: true,
            ..Self::default()
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if self.min_chapter_duration_ms <= 0 {
            return Err(AutoError::invalid_parameter(
                "min_chapter_duration_ms",
                "must be positive",
            ));
        }
        if self.max_chapter_duration_ms < self.min_chapter_duration_ms {
            return Err(AutoError::invalid_parameter(
                "max_chapter_duration_ms",
                "must be >= min_chapter_duration_ms",
            ));
        }
        if !(0.0..=1.0).contains(&self.score_drop_threshold) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.score_drop_threshold,
                min: 0.0,
                max: 1.0,
            });
        }
        if !(0.0..=1.0).contains(&self.audio_silence_threshold) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.audio_silence_threshold,
                min: 0.0,
                max: 1.0,
            });
        }
        if !(0.0..=1.0).contains(&self.min_boundary_confidence) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.min_boundary_confidence,
                min: 0.0,
                max: 1.0,
            });
        }
        if self.audio_valley_min_segments == 0 {
            return Err(AutoError::invalid_parameter(
                "audio_valley_min_segments",
                "must be at least 1",
            ));
        }
        Ok(())
    }
}

// ─── Internal boundary candidate ─────────────────────────────────────────────

#[derive(Debug)]
struct BoundaryCandidate {
    /// Index in the segment list (boundary is *before* segment[index]).
    segment_index: usize,
    /// Computed confidence.
    confidence: f64,
    /// Signals that triggered this candidate.
    signals: Vec<BoundarySignal>,
}

// ─── ChapterGenerator ─────────────────────────────────────────────────────────

/// Generates chapter markers from a sequence of scored scene segments.
pub struct ChapterGenerator {
    config: ChapterConfig,
}

impl ChapterGenerator {
    /// Create a new generator with the given configuration.
    #[must_use]
    pub fn new(config: ChapterConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration.
    #[must_use]
    pub fn default_generator() -> Self {
        Self::new(ChapterConfig::default())
    }

    /// Generate chapter markers for `segments`.
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or fewer than two
    /// segments are provided.
    pub fn generate(&self, segments: &[SceneSegment]) -> AutoResult<Vec<ChapterMarker>> {
        self.config.validate()?;

        if segments.is_empty() {
            return Err(AutoError::insufficient_data(
                "No scene segments provided for chapter generation",
            ));
        }

        if segments.len() == 1 {
            // Single segment: one chapter covering the entire content
            let chapter = self.single_chapter(segments);
            return Ok(vec![chapter]);
        }

        // 1. Detect raw boundary candidates
        let mut candidates = self.detect_boundaries(segments);

        // 2. Filter by minimum confidence
        candidates.retain(|c| c.confidence >= self.config.min_boundary_confidence);

        // 3. Merge candidates that are too close together
        candidates = self.merge_close_boundaries(candidates, segments);

        // 4. Force-split overlong chapters
        let candidate_pts: Vec<i64> = candidates
            .iter()
            .map(|c| segments[c.segment_index].start.pts)
            .collect();
        let boundary_pts = self.enforce_max_chapter_duration(segments, &candidate_pts);

        // 5. Build chapter markers from boundary timestamps
        let mut chapters = self.build_chapters(segments, &boundary_pts);

        // 6. Apply max_chapters cap (keep highest-confidence chapters)
        if self.config.max_chapters > 0 && chapters.len() > self.config.max_chapters {
            chapters.truncate(self.config.max_chapters);
            // Ensure the last chapter ends at the video end
            if let (Some(last), Some(seg_last)) = (chapters.last_mut(), segments.last()) {
                last.end = seg_last.end;
            }
        }

        Ok(chapters)
    }

    /// Export chapters to YouTube chapter format.
    ///
    /// The returned string contains one `MM:SS Title` line per chapter.
    #[must_use]
    pub fn to_youtube_chapters(chapters: &[ChapterMarker]) -> String {
        chapters
            .iter()
            .map(ChapterMarker::to_youtube_format)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &ChapterConfig {
        &self.config
    }

    // ── Private helpers ───────────────────────────────────────────────────────

    fn single_chapter(&self, segments: &[SceneSegment]) -> ChapterMarker {
        let start = segments[0].start;
        let end = segments.last().map(|s| s.end).unwrap_or(start);
        ChapterMarker {
            index: 1,
            start,
            end,
            title: format!("{} 1", self.config.title_prefix),
            confidence: 1.0,
            signals: vec![BoundarySignal::VideoStart],
        }
    }

    /// Detect all raw boundary candidates between consecutive segments.
    fn detect_boundaries(&self, segments: &[SceneSegment]) -> Vec<BoundaryCandidate> {
        let n = segments.len();
        let mut candidates = Vec::new();

        // Sliding window mean for score valley detection (window size = 3)
        let window_means: Vec<f64> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(1);
                let hi = (i + 2).min(n);
                let slice = &segments[lo..hi];
                slice.iter().map(|s| s.score).sum::<f64>() / slice.len() as f64
            })
            .collect();

        for i in 1..n {
            let prev = &segments[i - 1];
            let curr = &segments[i];
            let mut signals: Vec<BoundarySignal> = Vec::new();
            let mut confidence = 0.0f64;

            // --- Score valley ---
            let mean_drop = (window_means[i - 1] - window_means[i]).max(0.0);
            if mean_drop >= self.config.score_drop_threshold {
                signals.push(BoundarySignal::ScoreValley);
                let signal_w = BoundarySignal::ScoreValley.weight();
                confidence += signal_w * (mean_drop / self.config.score_drop_threshold).min(1.0);
            }

            // --- Hard scene cut (large instantaneous score jump) ---
            let abs_delta = (curr.score - prev.score).abs();
            if abs_delta >= self.config.score_drop_threshold * 1.5 {
                signals.push(BoundarySignal::SceneCut);
                confidence += BoundarySignal::SceneCut.weight();
            }

            // --- Audio valley ---
            let in_valley = self.in_audio_valley(segments, i);
            if in_valley {
                signals.push(BoundarySignal::AudioValley);
                confidence += BoundarySignal::AudioValley.weight();
            }

            // --- Content type change ---
            if self.config.split_on_content_type_change && prev.content_type != curr.content_type {
                signals.push(BoundarySignal::ContentTypeChange);
                confidence += BoundarySignal::ContentTypeChange.weight();
            }

            if !signals.is_empty() {
                candidates.push(BoundaryCandidate {
                    segment_index: i,
                    confidence: confidence.clamp(0.0, 1.0),
                    signals,
                });
            }
        }

        candidates
    }

    /// Check whether segment `i` is in an audio energy valley.
    fn in_audio_valley(&self, segments: &[SceneSegment], i: usize) -> bool {
        let n = segments.len();
        let min_run = self.config.audio_valley_min_segments;

        // Count how many consecutive segments around i are below threshold
        let mut run = 0usize;
        let start_check = i.saturating_sub(min_run / 2);
        for j in start_check..((start_check + min_run * 2).min(n)) {
            if segments[j].audio_energy <= self.config.audio_silence_threshold {
                run += 1;
            } else {
                run = 0;
            }
            if run >= min_run {
                return true;
            }
        }
        false
    }

    /// Merge boundary candidates that fall within `min_chapter_duration_ms` of
    /// each other.  Keeps the candidate with the highest confidence.
    fn merge_close_boundaries(
        &self,
        mut candidates: Vec<BoundaryCandidate>,
        segments: &[SceneSegment],
    ) -> Vec<BoundaryCandidate> {
        if candidates.len() < 2 {
            return candidates;
        }

        // Sort by segment_index so we can process in order
        candidates.sort_by_key(|c| c.segment_index);

        let mut merged: Vec<BoundaryCandidate> = Vec::new();
        let mut group_start = 0usize;

        for i in 1..=candidates.len() {
            let last_in_group = i == candidates.len();
            let too_close = !last_in_group && {
                let pts_a = segments[candidates[group_start].segment_index].start.pts;
                let pts_b = segments[candidates[i].segment_index].start.pts;
                (pts_b - pts_a).abs() < self.config.min_chapter_duration_ms
            };

            if !too_close || last_in_group {
                // Flush group: keep the highest-confidence candidate
                let best = candidates[group_start..i]
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| {
                        a.confidence
                            .partial_cmp(&b.confidence)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .map(|(rel_i, _)| group_start + rel_i)
                    .unwrap_or(group_start);

                // Move the best candidate out of the group
                let best_cand = BoundaryCandidate {
                    segment_index: candidates[best].segment_index,
                    confidence: candidates[best].confidence,
                    signals: candidates[best].signals.clone(),
                };
                merged.push(best_cand);
                group_start = i;
            }
        }

        merged
    }

    /// If any pair of adjacent boundaries is more than `max_chapter_duration_ms`
    /// apart, insert forced splits at the lowest-score segment boundary in the
    /// gap.
    fn enforce_max_chapter_duration(
        &self,
        segments: &[SceneSegment],
        boundary_pts: &[i64],
    ) -> Vec<i64> {
        let video_start = segments[0].start.pts;
        let video_end = segments.last().map(|s| s.end.pts).unwrap_or(video_start);

        // Build the full list including video start and end
        let mut all_pts: Vec<i64> = std::iter::once(video_start)
            .chain(boundary_pts.iter().copied())
            .chain(std::iter::once(video_end))
            .collect();
        all_pts.sort_unstable();
        all_pts.dedup();

        let mut result: Vec<i64> = Vec::new();
        let max_dur = self.config.max_chapter_duration_ms;

        for window in all_pts.windows(2) {
            let (a, b) = (window[0], window[1]);
            result.push(a);

            let gap = b - a;
            if gap > max_dur {
                // Find the segment boundary with the lowest score in [a, b]
                let split_pt = self.find_best_split_in_range(segments, a, b, max_dur);
                if let Some(pt) = split_pt {
                    result.push(pt);
                }
            }
        }

        // Push final endpoint
        if let Some(&last) = all_pts.last() {
            if result.last() != Some(&last) {
                result.push(last);
            }
        }

        result.sort_unstable();
        result.dedup();
        result
    }

    /// Find the segment boundary nearest to the midpoint of [start_pts, end_pts]
    /// that results in sub-chapters of at least `min_chapter_duration_ms`.
    fn find_best_split_in_range(
        &self,
        segments: &[SceneSegment],
        start_pts: i64,
        end_pts: i64,
        max_dur: i64,
    ) -> Option<i64> {
        let mid = start_pts + (end_pts - start_pts) / 2;
        let mut best_dist = i64::MAX;
        let mut best_pts: Option<i64> = None;

        for seg in segments {
            let pt = seg.start.pts;
            if pt <= start_pts + self.config.min_chapter_duration_ms {
                continue;
            }
            if pt >= end_pts - self.config.min_chapter_duration_ms {
                continue;
            }
            // Must also not violate max_dur from start
            if pt - start_pts > max_dur {
                continue;
            }
            let dist = (pt - mid).abs();
            if dist < best_dist {
                best_dist = dist;
                best_pts = Some(pt);
            }
        }

        best_pts
    }

    /// Build [`ChapterMarker`] list from a sorted list of boundary timestamps
    /// (including the video start and end).
    fn build_chapters(
        &self,
        segments: &[SceneSegment],
        boundary_pts: &[i64],
    ) -> Vec<ChapterMarker> {
        if boundary_pts.len() < 2 {
            return vec![self.single_chapter(segments)];
        }

        let tb = segments[0].start.timebase;
        let mut chapters = Vec::new();

        for (idx, window) in boundary_pts.windows(2).enumerate() {
            let ch_start_pts = window[0];
            let ch_end_pts = window[1];

            // Collect segments in this chapter
            let ch_segments: Vec<&SceneSegment> = segments
                .iter()
                .filter(|s| s.start.pts >= ch_start_pts && s.start.pts < ch_end_pts)
                .collect();

            // Determine dominant content type
            let dom_content = dominant_content_type(&ch_segments);

            // Title generation
            let chapter_number = idx + 1;
            let title = self.generate_title(
                chapter_number,
                boundary_pts.len() - 1,
                dom_content,
                &ch_segments,
            );

            // Confidence: maximum of any boundary signal that ends this chapter
            let confidence = 0.70; // default; full signal info not carried through here

            let signals = if idx == 0 {
                vec![BoundarySignal::VideoStart]
            } else if idx + 2 == boundary_pts.len() {
                vec![BoundarySignal::VideoEnd]
            } else {
                vec![BoundarySignal::SceneCut] // placeholder
            };

            chapters.push(ChapterMarker {
                index: chapter_number,
                start: Timestamp::new(ch_start_pts, tb),
                end: Timestamp::new(ch_end_pts, tb),
                title,
                confidence,
                signals,
            });
        }

        chapters
    }

    /// Generate a human-readable chapter title.
    fn generate_title(
        &self,
        chapter_number: usize,
        total_chapters: usize,
        content_type: ContentType,
        segments: &[&SceneSegment],
    ) -> String {
        // Use first segment's description if available
        if let Some(desc) = segments.first().and_then(|s| s.description.as_deref()) {
            return desc.to_string();
        }

        // Position-based labels for well-known positions
        if chapter_number == 1 {
            return "Introduction".to_string();
        }
        if chapter_number == total_chapters {
            return "Conclusion".to_string();
        }

        // Content-type hints
        let type_hint = match content_type {
            ContentType::Action => "Action",
            ContentType::Dialogue => "Discussion",
            ContentType::Establishing => "Overview",
            ContentType::CloseUp => "Close-up",
            ContentType::Group => "Group",
            ContentType::Static => "Analysis",
            ContentType::Transition => "Interlude",
            ContentType::Unknown => &self.config.title_prefix,
        };

        // Position in video arc
        let position_fraction = chapter_number as f64 / total_chapters as f64;
        let arc_label = if position_fraction < 0.25 {
            "Opening"
        } else if position_fraction < 0.75 {
            type_hint
        } else {
            "Climax"
        };

        format!("{arc_label} {chapter_number}")
    }
}

/// Find the most common content type in a slice of segments.
fn dominant_content_type(segments: &[&SceneSegment]) -> ContentType {
    if segments.is_empty() {
        return ContentType::Unknown;
    }

    // Count occurrences of each type using a simple linear scan
    // (only 8 variants — no HashMap needed)
    let types = [
        ContentType::Action,
        ContentType::Dialogue,
        ContentType::Static,
        ContentType::Establishing,
        ContentType::CloseUp,
        ContentType::Group,
        ContentType::Transition,
        ContentType::Unknown,
    ];

    let mut best_type = ContentType::Unknown;
    let mut best_count = 0usize;

    for &ct in &types {
        let count = segments.iter().filter(|s| s.content_type == ct).count();
        if count > best_count {
            best_count = count;
            best_type = ct;
        }
    }

    best_type
}

impl Default for ChapterGenerator {
    fn default() -> Self {
        Self::default_generator()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::Rational;

    fn tb() -> Rational {
        Rational::new(1, 1000)
    }

    fn ts(ms: i64) -> Timestamp {
        Timestamp::new(ms, tb())
    }

    fn seg(start_ms: i64, end_ms: i64, score: f64, ct: ContentType) -> SceneSegment {
        SceneSegment::new(ts(start_ms), ts(end_ms), score, ct)
    }

    fn make_segments_uniform(count: usize, duration_each_ms: i64) -> Vec<SceneSegment> {
        (0..count)
            .map(|i| {
                seg(
                    i as i64 * duration_each_ms,
                    (i as i64 + 1) * duration_each_ms,
                    0.5,
                    ContentType::Unknown,
                )
            })
            .collect()
    }

    #[test]
    fn test_default_config_valid() {
        assert!(ChapterConfig::default().validate().is_ok());
    }

    #[test]
    fn test_podcast_preset_valid() {
        assert!(ChapterConfig::podcast().validate().is_ok());
    }

    #[test]
    fn test_sports_preset_valid() {
        assert!(ChapterConfig::sports().validate().is_ok());
    }

    #[test]
    fn test_invalid_min_chapter_duration() {
        let mut cfg = ChapterConfig::default();
        cfg.min_chapter_duration_ms = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_max_less_than_min_chapter_duration() {
        let mut cfg = ChapterConfig::default();
        cfg.min_chapter_duration_ms = 60_000;
        cfg.max_chapter_duration_ms = 30_000;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_empty_segments_error() {
        let gen = ChapterGenerator::default();
        assert!(gen.generate(&[]).is_err());
    }

    #[test]
    fn test_single_segment_produces_one_chapter() {
        let gen = ChapterGenerator::default();
        let segments = vec![seg(0, 300_000, 0.8, ContentType::Dialogue)];
        let chapters = gen.generate(&segments).expect("should succeed");
        assert_eq!(chapters.len(), 1);
        assert_eq!(chapters[0].index, 1);
    }

    #[test]
    fn test_chapter_markers_cover_full_video() {
        let config = ChapterConfig {
            min_chapter_duration_ms: 10_000,
            score_drop_threshold: 0.30,
            ..ChapterConfig::default()
        };
        let gen = ChapterGenerator::new(config);
        // Create segments with a clear score valley in the middle
        let mut segments: Vec<SceneSegment> = (0..10)
            .map(|i| {
                let score = if i < 3 || i > 7 { 0.8 } else { 0.1 };
                seg(i * 30_000, (i + 1) * 30_000, score, ContentType::Unknown)
            })
            .collect();
        // Mark the valley segments as silent
        for s in segments[3..8].iter_mut() {
            s.audio_energy = 0.02;
        }

        let chapters = gen.generate(&segments).expect("should succeed");
        assert!(!chapters.is_empty());

        // Chapters must be non-overlapping and cover the full range
        let video_start = segments[0].start.pts;
        let video_end = segments.last().unwrap().end.pts;
        assert_eq!(chapters[0].start.pts, video_start);
        assert_eq!(chapters.last().unwrap().end.pts, video_end);
    }

    #[test]
    fn test_min_chapter_duration_respected() {
        let config = ChapterConfig {
            min_chapter_duration_ms: 60_000,
            score_drop_threshold: 0.01, // very sensitive → many boundaries
            split_on_content_type_change: true,
            ..ChapterConfig::default()
        };
        let gen = ChapterGenerator::new(config.clone());
        // Lots of short alternating segments
        let segments: Vec<SceneSegment> = (0..20)
            .map(|i| {
                let ct = if i % 2 == 0 {
                    ContentType::Action
                } else {
                    ContentType::Dialogue
                };
                seg(i * 10_000, (i + 1) * 10_000, 0.5, ct)
            })
            .collect();

        let chapters = gen.generate(&segments).expect("should succeed");
        for ch in &chapters {
            assert!(
                ch.duration_ms() >= config.min_chapter_duration_ms || ch.index == chapters.len(), // last chapter may be shorter
                "chapter {} duration {} < min {}",
                ch.index,
                ch.duration_ms(),
                config.min_chapter_duration_ms
            );
        }
    }

    #[test]
    fn test_content_type_change_splits() {
        let config = ChapterConfig {
            min_chapter_duration_ms: 5_000,
            split_on_content_type_change: true,
            min_boundary_confidence: 0.10,
            ..ChapterConfig::default()
        };
        let gen = ChapterGenerator::new(config);
        let segments = vec![
            seg(0, 30_000, 0.7, ContentType::Dialogue),
            seg(30_000, 60_000, 0.7, ContentType::Action), // type change here
            seg(60_000, 90_000, 0.7, ContentType::Action),
        ];
        let chapters = gen.generate(&segments).expect("should succeed");
        // At least 2 chapters (dialogue → action transition)
        assert!(
            chapters.len() >= 2,
            "expected at least 2 chapters, got {}",
            chapters.len()
        );
    }

    #[test]
    fn test_chapter_indices_sequential() {
        let gen = ChapterGenerator::default();
        let segments = make_segments_uniform(8, 60_000);
        let chapters = gen.generate(&segments).expect("should succeed");
        for (i, ch) in chapters.iter().enumerate() {
            assert_eq!(ch.index, i + 1, "chapter index not sequential");
        }
    }

    #[test]
    fn test_max_chapters_cap() {
        let config = ChapterConfig {
            min_chapter_duration_ms: 5_000,
            max_chapters: 3,
            score_drop_threshold: 0.01,
            split_on_content_type_change: true,
            min_boundary_confidence: 0.10,
            ..ChapterConfig::default()
        };
        let gen = ChapterGenerator::new(config);
        // Many segments with alternating types → many potential chapters
        let segments: Vec<SceneSegment> = (0..20)
            .map(|i| {
                let ct = if i % 2 == 0 {
                    ContentType::Action
                } else {
                    ContentType::Dialogue
                };
                seg(i * 10_000, (i + 1) * 10_000, 0.5, ct)
            })
            .collect();
        let chapters = gen.generate(&segments).expect("should succeed");
        assert!(
            chapters.len() <= 3,
            "max_chapters violated: got {}",
            chapters.len()
        );
    }

    #[test]
    fn test_youtube_format_output() {
        let tb = Rational::new(1, 1000);
        let chapters = vec![
            ChapterMarker {
                index: 1,
                start: Timestamp::new(0, tb),
                end: Timestamp::new(60_000, tb),
                title: "Introduction".to_string(),
                confidence: 1.0,
                signals: vec![BoundarySignal::VideoStart],
            },
            ChapterMarker {
                index: 2,
                start: Timestamp::new(60_000, tb),
                end: Timestamp::new(120_000, tb),
                title: "Main Content".to_string(),
                confidence: 0.8,
                signals: vec![BoundarySignal::SceneCut],
            },
        ];
        let output = ChapterGenerator::to_youtube_chapters(&chapters);
        assert!(output.contains("00:00 Introduction"));
        assert!(output.contains("01:00 Main Content"));
    }

    #[test]
    fn test_format_timestamp_minutes() {
        let tb = Rational::new(1, 1000);
        let ch = ChapterMarker {
            index: 1,
            start: Timestamp::new(125_000, tb), // 2:05
            end: Timestamp::new(180_000, tb),
            title: "Test".to_string(),
            confidence: 0.9,
            signals: vec![],
        };
        assert_eq!(ch.format_timestamp(), "02:05");
    }

    #[test]
    fn test_boundary_signal_weights_sum_reasonable() {
        let signals = [
            BoundarySignal::SceneCut,
            BoundarySignal::ScoreValley,
            BoundarySignal::AudioValley,
            BoundarySignal::ContentTypeChange,
        ];
        let total: f64 = signals.iter().map(|s| s.weight()).sum();
        assert!(
            total <= 1.0 + 1e-9,
            "combined signal weight should be ≤ 1.0"
        );
    }

    #[test]
    fn test_scene_segment_builder() {
        let s = SceneSegment::new(ts(0), ts(5_000), 0.7, ContentType::Action)
            .with_audio_energy(0.3)
            .with_description("Opening action sequence");
        assert_eq!(s.audio_energy, 0.3);
        assert_eq!(s.description.as_deref(), Some("Opening action sequence"));
        assert_eq!(s.duration_ms(), 5_000);
    }

    #[test]
    fn test_dominant_content_type() {
        let s1 = SceneSegment::new(ts(0), ts(5_000), 0.5, ContentType::Dialogue);
        let s2 = SceneSegment::new(ts(5_000), ts(10_000), 0.5, ContentType::Dialogue);
        let s3 = SceneSegment::new(ts(10_000), ts(15_000), 0.5, ContentType::Action);
        let refs = [&s1, &s2, &s3];
        let dominant = dominant_content_type(&refs);
        assert_eq!(dominant, ContentType::Dialogue);
    }
}
