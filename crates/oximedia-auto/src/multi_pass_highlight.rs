//! Multi-pass highlight detection with configurable coarse-then-fine analysis.
//!
//! This module extends highlight detection by running a two-stage pipeline:
//!
//! 1. **Coarse pass**: Fast, low-resolution scan over the entire video to identify
//!    candidate regions.  Uses stride sampling and simplified feature extraction.
//! 2. **Fine pass**: High-fidelity, per-frame analysis restricted to regions that
//!    survived the coarse pass, enabling cheap rejection of uninteresting content.
//!
//! Each pass is independently configurable so the caller can tune the trade-off
//! between speed and recall.  Results from both passes are merged and deduplicated
//! before being returned.
//!
//! # Example
//!
//! ```
//! use oximedia_auto::multi_pass_highlight::{
//!     MultiPassDetector, MultiPassConfig, PassConfig,
//! };
//!
//! let config = MultiPassConfig::default();
//! let detector = MultiPassDetector::new(config);
//! ```

#![allow(dead_code)]

use crate::error::{AutoError, AutoResult};
use crate::highlights::{Highlight, HighlightType};
use crate::scoring::SceneFeatures;
use oximedia_core::Timestamp;

// ─── Pass Configuration ──────────────────────────────────────────────────────

/// Configuration for a single detection pass (coarse or fine).
#[derive(Debug, Clone)]
pub struct PassConfig {
    /// Frame stride: analyse every N-th frame.
    ///
    /// A stride of 1 means every frame is analysed; larger values trade recall
    /// for speed.
    pub frame_stride: usize,
    /// Minimum importance score required to keep a candidate region.
    pub score_threshold: f64,
    /// Minimum confidence required to keep a candidate region.
    pub confidence_threshold: f64,
    /// Temporal smoothing window (number of frames).
    pub smoothing_window: usize,
    /// Whether to merge overlapping detections within this pass.
    pub merge_overlaps: bool,
    /// Overlap ratio above which two detections are merged [0.0, 1.0].
    pub overlap_merge_ratio: f64,
}

impl Default for PassConfig {
    fn default() -> Self {
        Self {
            frame_stride: 1,
            score_threshold: 0.4,
            confidence_threshold: 0.4,
            smoothing_window: 3,
            merge_overlaps: true,
            overlap_merge_ratio: 0.5,
        }
    }
}

impl PassConfig {
    /// Create a coarse-pass configuration with aggressive stride and low threshold.
    #[must_use]
    pub fn coarse() -> Self {
        Self {
            frame_stride: 8,
            score_threshold: 0.25,
            confidence_threshold: 0.25,
            smoothing_window: 5,
            merge_overlaps: true,
            overlap_merge_ratio: 0.3,
        }
    }

    /// Create a fine-pass configuration that processes every frame precisely.
    #[must_use]
    pub fn fine() -> Self {
        Self {
            frame_stride: 1,
            score_threshold: 0.5,
            confidence_threshold: 0.5,
            smoothing_window: 2,
            merge_overlaps: true,
            overlap_merge_ratio: 0.6,
        }
    }

    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any value is out of range.
    pub fn validate(&self) -> AutoResult<()> {
        if self.frame_stride == 0 {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "frame_stride must be >= 1".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.score_threshold) {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "score_threshold must be in [0.0, 1.0]".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.confidence_threshold) {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "confidence_threshold must be in [0.0, 1.0]".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.overlap_merge_ratio) {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "overlap_merge_ratio must be in [0.0, 1.0]".to_string(),
            });
        }
        Ok(())
    }
}

// ─── Multi-Pass Configuration ─────────────────────────────────────────────────

/// Configuration for the multi-pass highlight detector.
#[derive(Debug, Clone)]
pub struct MultiPassConfig {
    /// Configuration for the coarse (first) pass.
    pub coarse: PassConfig,
    /// Configuration for the fine (second) pass.
    pub fine: PassConfig,
    /// Minimum region duration in milliseconds that a coarse candidate must
    /// have before it proceeds to the fine pass.
    pub min_region_duration_ms: i64,
    /// Expansion ratio applied to coarse candidate boundaries when feeding the
    /// fine pass.  E.g. 0.1 expands each boundary by 10% of the region length.
    pub boundary_expansion_ratio: f64,
    /// Final score threshold applied after merging both passes.
    pub final_score_threshold: f64,
    /// Maximum number of highlights returned.
    pub max_highlights: usize,
}

impl Default for MultiPassConfig {
    fn default() -> Self {
        Self {
            coarse: PassConfig::coarse(),
            fine: PassConfig::fine(),
            min_region_duration_ms: 500,
            boundary_expansion_ratio: 0.1,
            final_score_threshold: 0.45,
            max_highlights: 50,
        }
    }
}

impl MultiPassConfig {
    /// Validate the full multi-pass configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if any sub-configuration is invalid.
    pub fn validate(&self) -> AutoResult<()> {
        self.coarse.validate()?;
        self.fine.validate()?;
        if self.min_region_duration_ms < 0 {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "min_region_duration_ms must be >= 0".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.boundary_expansion_ratio) {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "boundary_expansion_ratio must be in [0.0, 1.0]".to_string(),
            });
        }
        if !(0.0..=1.0).contains(&self.final_score_threshold) {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "final_score_threshold must be in [0.0, 1.0]".to_string(),
            });
        }
        if self.max_highlights == 0 {
            return Err(AutoError::InvalidParameter {
                name: "config".to_string(),
                value: "max_highlights must be >= 1".to_string(),
            });
        }
        Ok(())
    }
}

// ─── Frame-Level Input ────────────────────────────────────────────────────────

/// Lightweight frame descriptor used as input to the multi-pass detector.
///
/// The caller pre-extracts per-frame features (motion, luminance, etc.) so the
/// detector itself does not depend on the concrete `VideoFrame` representation.
#[derive(Debug, Clone)]
pub struct FrameDescriptor {
    /// Timestamp of this frame.
    pub timestamp: Timestamp,
    /// Pre-computed scene features.
    pub features: SceneFeatures,
    /// Pre-computed importance hint (0.0–1.0) from upstream analysis, or 0.5
    /// if no prior estimate is available.
    pub prior_score: f64,
}

impl FrameDescriptor {
    /// Create a new frame descriptor.
    #[must_use]
    pub fn new(timestamp: Timestamp, features: SceneFeatures, prior_score: f64) -> Self {
        Self {
            timestamp,
            features,
            prior_score: prior_score.clamp(0.0, 1.0),
        }
    }
}

// ─── Candidate Region ─────────────────────────────────────────────────────────

/// An intermediate candidate highlight region produced by the coarse pass.
#[derive(Debug, Clone)]
pub struct CandidateRegion {
    /// Start timestamp.
    pub start: Timestamp,
    /// End timestamp.
    pub end: Timestamp,
    /// Average importance score within the region.
    pub avg_score: f64,
    /// Peak importance score within the region.
    pub peak_score: f64,
    /// Number of contributing frames.
    pub frame_count: usize,
}

impl CandidateRegion {
    /// Duration in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }

    /// Check whether this region overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start.pts < other.end.pts && self.end.pts > other.start.pts
    }

    /// Compute the overlap ratio relative to the smaller region.
    #[must_use]
    pub fn overlap_ratio(&self, other: &Self) -> f64 {
        let overlap_start = self.start.pts.max(other.start.pts);
        let overlap_end = self.end.pts.min(other.end.pts);
        if overlap_end <= overlap_start {
            return 0.0;
        }
        let overlap = (overlap_end - overlap_start) as f64;
        let self_dur = self.duration_ms() as f64;
        let other_dur = other.duration_ms() as f64;
        let min_dur = self_dur.min(other_dur);
        if min_dur <= 0.0 {
            0.0
        } else {
            overlap / min_dur
        }
    }

    /// Expand boundaries by a ratio of the region duration.
    #[must_use]
    pub fn expanded(&self, ratio: f64) -> Self {
        let dur = self.duration_ms();
        let expansion = (dur as f64 * ratio) as i64;
        let new_start = Timestamp::new((self.start.pts - expansion).max(0), self.start.timebase);
        let new_end = Timestamp::new(self.end.pts + expansion, self.end.timebase);
        Self {
            start: new_start,
            end: new_end,
            avg_score: self.avg_score,
            peak_score: self.peak_score,
            frame_count: self.frame_count,
        }
    }
}

// ─── Multi-Pass Detector ──────────────────────────────────────────────────────

/// Multi-pass highlight detector.
///
/// Runs a fast coarse pass over all frames to identify candidate regions, then
/// runs a fine pass limited to those regions.  The two sets of results are
/// merged, deduplicated, and filtered by the final score threshold.
pub struct MultiPassDetector {
    config: MultiPassConfig,
}

impl MultiPassDetector {
    /// Create a new multi-pass detector with the given configuration.
    #[must_use]
    pub fn new(config: MultiPassConfig) -> Self {
        Self { config }
    }

    /// Create a detector with default configuration.
    #[must_use]
    pub fn default_detector() -> Self {
        Self::new(MultiPassConfig::default())
    }

    /// Run the multi-pass detection pipeline.
    ///
    /// Returns highlights sorted by score descending, capped at
    /// [`MultiPassConfig::max_highlights`].
    ///
    /// # Errors
    ///
    /// Returns an error if the configuration is invalid or the frame list is
    /// empty.
    pub fn detect(&self, frames: &[FrameDescriptor]) -> AutoResult<Vec<Highlight>> {
        self.config.validate()?;
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        // --- Coarse pass ---
        let coarse_candidates = self.coarse_pass(frames);
        if coarse_candidates.is_empty() {
            return Ok(Vec::new());
        }

        // --- Fine pass (restricted to expanded coarse candidates) ---
        let fine_highlights = self.fine_pass(frames, &coarse_candidates);

        // --- Merge coarse + fine, deduplicate, filter ---
        let mut merged = self.merge_results(coarse_candidates, fine_highlights);

        // Sort by score descending.
        merged.sort_by(|a, b| {
            b.weighted_score()
                .partial_cmp(&a.weighted_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        merged.truncate(self.config.max_highlights);
        Ok(merged)
    }

    // ── Coarse pass ──────────────────────────────────────────────────────────

    fn coarse_pass(&self, frames: &[FrameDescriptor]) -> Vec<CandidateRegion> {
        let stride = self.config.coarse.frame_stride.max(1);
        let threshold = self.config.coarse.score_threshold;

        // Compute per-stride-frame scores.
        let scored: Vec<(usize, f64)> = frames
            .iter()
            .enumerate()
            .filter(|(i, _)| i % stride == 0)
            .map(|(i, fd)| {
                let raw = self.quick_score(&fd.features, fd.prior_score);
                (i, raw)
            })
            .collect();

        // Smooth scores.
        let smoothed = self.smooth_scores(&scored, self.config.coarse.smoothing_window);

        // Group consecutive frames above threshold into candidate regions.
        let mut candidates: Vec<CandidateRegion> = Vec::new();
        let mut region_start: Option<usize> = None;
        let mut region_scores: Vec<f64> = Vec::new();

        for (frame_idx, score) in &smoothed {
            if *score >= threshold {
                if region_start.is_none() {
                    region_start = Some(*frame_idx);
                }
                region_scores.push(*score);
            } else if let Some(start_idx) = region_start.take() {
                // Close the region.
                if let Some(region) =
                    self.build_region(frames, start_idx, *frame_idx, &region_scores)
                {
                    if region.duration_ms() >= self.config.min_region_duration_ms {
                        candidates.push(region);
                    }
                }
                region_scores.clear();
            }
        }

        // Handle open region at end of frames.
        if let Some(start_idx) = region_start {
            let last_idx = frames.len().saturating_sub(1);
            if let Some(region) = self.build_region(frames, start_idx, last_idx, &region_scores) {
                if region.duration_ms() >= self.config.min_region_duration_ms {
                    candidates.push(region);
                }
            }
        }

        if self.config.coarse.merge_overlaps {
            self.merge_candidate_regions(candidates, self.config.coarse.overlap_merge_ratio)
        } else {
            candidates
        }
    }

    // ── Fine pass ────────────────────────────────────────────────────────────

    fn fine_pass(
        &self,
        frames: &[FrameDescriptor],
        coarse_candidates: &[CandidateRegion],
    ) -> Vec<Highlight> {
        let stride = self.config.fine.frame_stride.max(1);
        let threshold = self.config.fine.score_threshold;
        let conf_threshold = self.config.fine.confidence_threshold;

        let mut highlights: Vec<Highlight> = Vec::new();

        for candidate in coarse_candidates {
            let expanded = candidate.expanded(self.config.boundary_expansion_ratio);

            // Collect frames within expanded region.
            let region_frames: Vec<&FrameDescriptor> = frames
                .iter()
                .enumerate()
                .filter(|(i, fd)| {
                    i % stride == 0
                        && fd.timestamp.pts >= expanded.start.pts
                        && fd.timestamp.pts <= expanded.end.pts
                })
                .map(|(_, fd)| fd)
                .collect();

            if region_frames.is_empty() {
                continue;
            }

            // Find peak frame within region.
            let peak = region_frames
                .iter()
                .map(|fd| {
                    let score = self.precise_score(&fd.features, fd.prior_score);
                    (fd, score)
                })
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if let Some((peak_fd, peak_score)) = peak {
                if peak_score < threshold {
                    continue;
                }
                let confidence = self.compute_confidence(&region_frames, peak_score);
                if confidence < conf_threshold {
                    continue;
                }

                let first_ts = region_frames
                    .first()
                    .map(|fd| fd.timestamp)
                    .unwrap_or(expanded.start);
                let last_ts = region_frames
                    .last()
                    .map(|fd| fd.timestamp)
                    .unwrap_or(expanded.end);

                let mut h = Highlight::new(
                    first_ts,
                    last_ts,
                    HighlightType::Composite,
                    peak_score,
                    confidence,
                );
                h.features = peak_fd.features.clone();
                h.description = format!(
                    "fine-pass highlight at {}ms (score={:.2})",
                    peak_fd.timestamp.pts, peak_score
                );
                highlights.push(h);
            }
        }

        if self.config.fine.merge_overlaps {
            self.merge_highlights(highlights, self.config.fine.overlap_merge_ratio)
        } else {
            highlights
        }
    }

    // ── Merge coarse + fine ───────────────────────────────────────────────────

    fn merge_results(
        &self,
        coarse: Vec<CandidateRegion>,
        mut fine: Vec<Highlight>,
    ) -> Vec<Highlight> {
        // Convert remaining coarse candidates (that didn't get a fine match) into
        // low-confidence highlights and append.
        let threshold = self.config.final_score_threshold;

        for candidate in coarse {
            if candidate.avg_score < threshold {
                continue;
            }
            // Check if any fine highlight already covers this region.
            let covered = fine
                .iter()
                .any(|h| h.start.pts <= candidate.end.pts && h.end.pts >= candidate.start.pts);
            if !covered {
                let mut h = Highlight::new(
                    candidate.start,
                    candidate.end,
                    HighlightType::Composite,
                    candidate.peak_score,
                    candidate.avg_score,
                );
                h.description = format!(
                    "coarse-pass candidate (avg_score={:.2})",
                    candidate.avg_score
                );
                fine.push(h);
            }
        }

        // Final dedup: merge overlapping highlights.
        self.merge_highlights(fine, 0.5)
    }

    // ── Scoring helpers ───────────────────────────────────────────────────────

    /// Fast approximation used in the coarse pass.
    fn quick_score(&self, features: &SceneFeatures, prior: f64) -> f64 {
        let feature_score = features.motion_intensity * 0.4
            + features.audio_peak * 0.3
            + features.face_coverage * 0.2
            + features.color_diversity * 0.1;
        (feature_score * 0.7 + prior * 0.3).clamp(0.0, 1.0)
    }

    /// Precise scoring used in the fine pass.
    fn precise_score(&self, features: &SceneFeatures, prior: f64) -> f64 {
        let weighted = features.motion_intensity * 0.25
            + features.audio_peak * 0.20
            + features.audio_energy * 0.10
            + features.face_coverage * 0.15
            + features.color_diversity * 0.05
            + features.edge_density * 0.05
            + features.contrast * 0.05
            + features.sharpness * 0.05
            + features.object_diversity * 0.10;
        (weighted * 0.8 + prior * 0.2).clamp(0.0, 1.0)
    }

    /// Compute confidence as consistency of scores within the region.
    fn compute_confidence(&self, frames: &[&FrameDescriptor], peak_score: f64) -> f64 {
        if frames.is_empty() {
            return 0.0;
        }
        let scores: Vec<f64> = frames
            .iter()
            .map(|fd| self.precise_score(&fd.features, fd.prior_score))
            .collect();
        let mean = scores.iter().sum::<f64>() / scores.len() as f64;
        // Confidence is proximity of mean to peak; high consistency → high confidence.
        let consistency = 1.0 - (peak_score - mean).abs();
        (peak_score * consistency).clamp(0.0, 1.0)
    }

    // ── Utility helpers ───────────────────────────────────────────────────────

    fn smooth_scores(&self, scored: &[(usize, f64)], window: usize) -> Vec<(usize, f64)> {
        if window <= 1 {
            return scored.to_vec();
        }
        let half = window / 2;
        scored
            .iter()
            .enumerate()
            .map(|(i, (idx, _))| {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(scored.len());
                let avg = scored[lo..hi].iter().map(|(_, s)| s).sum::<f64>() / (hi - lo) as f64;
                (*idx, avg)
            })
            .collect()
    }

    fn build_region(
        &self,
        frames: &[FrameDescriptor],
        start_idx: usize,
        end_idx: usize,
        scores: &[f64],
    ) -> Option<CandidateRegion> {
        let start_ts = frames.get(start_idx)?.timestamp;
        let end_ts = frames
            .get(end_idx.min(frames.len().saturating_sub(1)))?
            .timestamp;
        if scores.is_empty() {
            return None;
        }
        let avg_score = scores.iter().sum::<f64>() / scores.len() as f64;
        let peak_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        Some(CandidateRegion {
            start: start_ts,
            end: end_ts,
            avg_score,
            peak_score,
            frame_count: scores.len(),
        })
    }

    fn merge_candidate_regions(
        &self,
        mut regions: Vec<CandidateRegion>,
        overlap_ratio: f64,
    ) -> Vec<CandidateRegion> {
        if regions.len() < 2 {
            return regions;
        }
        regions.sort_by_key(|r| r.start.pts);
        let mut merged: Vec<CandidateRegion> = Vec::new();

        for region in regions {
            if let Some(last) = merged.last_mut() {
                if last.overlaps(&region) && region.overlap_ratio(last) >= overlap_ratio {
                    // Merge into last.
                    last.end = Timestamp::new(last.end.pts.max(region.end.pts), last.end.timebase);
                    let total = last.frame_count + region.frame_count;
                    last.avg_score = (last.avg_score * last.frame_count as f64
                        + region.avg_score * region.frame_count as f64)
                        / total as f64;
                    last.peak_score = last.peak_score.max(region.peak_score);
                    last.frame_count = total;
                    continue;
                }
            }
            merged.push(region);
        }
        merged
    }

    fn merge_highlights(
        &self,
        mut highlights: Vec<Highlight>,
        overlap_ratio: f64,
    ) -> Vec<Highlight> {
        if highlights.len() < 2 {
            return highlights;
        }
        highlights.sort_by_key(|h| h.start.pts);
        let mut merged: Vec<Highlight> = Vec::new();

        for h in highlights {
            if let Some(last) = merged.last_mut() {
                let last_dur = (last.end.pts - last.start.pts).max(0) as f64;
                let h_dur = (h.end.pts - h.start.pts).max(0) as f64;
                let min_dur = last_dur.min(h_dur);
                if min_dur > 0.0 {
                    let overlap_start = last.start.pts.max(h.start.pts);
                    let overlap_end = last.end.pts.min(h.end.pts);
                    let overlap = (overlap_end - overlap_start).max(0) as f64;
                    let ratio = overlap / min_dur;
                    if ratio >= overlap_ratio {
                        *last = last.merge(&h);
                        continue;
                    }
                }
            }
            merged.push(h);
        }
        merged
    }
}

// ─── Detection Statistics ─────────────────────────────────────────────────────

/// Statistics about a completed multi-pass detection run.
#[derive(Debug, Clone, Default)]
pub struct DetectionStats {
    /// Number of frames analysed in the coarse pass.
    pub coarse_frames_analysed: usize,
    /// Number of candidate regions produced by the coarse pass.
    pub coarse_candidates: usize,
    /// Number of frames analysed in the fine pass.
    pub fine_frames_analysed: usize,
    /// Number of highlights produced by the fine pass.
    pub fine_highlights: usize,
    /// Number of highlights in the final merged output.
    pub final_highlights: usize,
    /// Frame skip ratio (coarse frames / total frames).
    pub skip_ratio: f64,
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scoring::SceneFeatures;
    use oximedia_core::Timestamp;

    fn make_ts(pts: i64) -> Timestamp {
        Timestamp::new(pts, oximedia_core::Rational::new(1, 1000))
    }

    fn make_frame(pts: i64, motion: f64, audio: f64) -> FrameDescriptor {
        let mut features = SceneFeatures::default();
        features.motion_intensity = motion;
        features.audio_peak = audio;
        features.color_diversity = 0.3;
        FrameDescriptor::new(make_ts(pts), features, (motion + audio) / 2.0)
    }

    #[test]
    fn test_default_config_validates() {
        let config = MultiPassConfig::default();
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_pass_config_coarse_and_fine() {
        let coarse = PassConfig::coarse();
        assert!(coarse.frame_stride > 1, "coarse should skip frames");
        let fine = PassConfig::fine();
        assert_eq!(fine.frame_stride, 1, "fine should process every frame");
    }

    #[test]
    fn test_invalid_stride_rejected() {
        let mut cfg = PassConfig::default();
        cfg.frame_stride = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_invalid_threshold_rejected() {
        let mut cfg = PassConfig::default();
        cfg.score_threshold = 1.5;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_empty_frames_returns_empty() {
        let detector = MultiPassDetector::default_detector();
        let result = detector.detect(&[]).expect("detect should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_low_score_frames_produce_no_highlights() {
        let frames: Vec<FrameDescriptor> =
            (0..20).map(|i| make_frame(i * 100, 0.05, 0.05)).collect();
        let detector = MultiPassDetector::default_detector();
        let result = detector.detect(&frames).expect("detect should succeed");
        assert!(
            result.is_empty(),
            "low-score frames should not produce highlights"
        );
    }

    #[test]
    fn test_high_score_frames_produce_highlights() {
        let mut frames: Vec<FrameDescriptor> =
            (0..32).map(|i| make_frame(i * 100, 0.05, 0.05)).collect();
        // Insert a high-energy burst spanning frames 8-18 (includes stride-8
        // index at frame 8 and 16, ensuring the coarse pass detects the region).
        for i in 8..19 {
            frames[i] = make_frame((i as i64) * 100, 0.95, 0.90);
        }
        let detector = MultiPassDetector::default_detector();
        let result = detector.detect(&frames).expect("detect should succeed");
        assert!(
            !result.is_empty(),
            "high-score frames should produce at least one highlight"
        );
    }

    #[test]
    fn test_highlights_sorted_by_score_descending() {
        let mut frames: Vec<FrameDescriptor> =
            (0..50).map(|i| make_frame(i * 100, 0.05, 0.05)).collect();
        // Two separate bursts aligned with stride-8 sample points.
        // Burst 1 at frames 6-10 (includes index 8).
        for i in 6..11 {
            frames[i] = make_frame((i as i64) * 100, 0.70, 0.65);
        }
        // Burst 2 at frames 22-28 (includes indices 24 and potentially 16).
        for i in 22..29 {
            frames[i] = make_frame((i as i64) * 100, 0.95, 0.90);
        }
        let detector = MultiPassDetector::default_detector();
        let result = detector.detect(&frames).expect("detect should succeed");
        for pair in result.windows(2) {
            assert!(
                pair[0].weighted_score() >= pair[1].weighted_score(),
                "highlights must be sorted by weighted score descending"
            );
        }
    }

    #[test]
    fn test_max_highlights_capped() {
        let mut config = MultiPassConfig::default();
        config.max_highlights = 2;
        // Lower thresholds so more candidates survive.
        config.coarse.score_threshold = 0.1;
        config.fine.score_threshold = 0.1;
        config.fine.confidence_threshold = 0.1;
        config.final_score_threshold = 0.1;

        let mut frames: Vec<FrameDescriptor> =
            (0..80).map(|i| make_frame(i * 100, 0.05, 0.05)).collect();
        // Scatter many bursts.
        for burst_start in [0usize, 15, 30, 45, 60] {
            for i in burst_start..burst_start + 4 {
                if i < frames.len() {
                    frames[i] = make_frame((i as i64) * 100, 0.9, 0.85);
                }
            }
        }
        let detector = MultiPassDetector::new(config);
        let result = detector.detect(&frames).expect("detect should succeed");
        assert!(
            result.len() <= 2,
            "result length {} exceeds max_highlights cap of 2",
            result.len()
        );
    }

    #[test]
    fn test_candidate_region_overlap_ratio() {
        let r1 = CandidateRegion {
            start: make_ts(0),
            end: make_ts(1000),
            avg_score: 0.7,
            peak_score: 0.9,
            frame_count: 10,
        };
        let r2 = CandidateRegion {
            start: make_ts(500),
            end: make_ts(1500),
            avg_score: 0.6,
            peak_score: 0.8,
            frame_count: 10,
        };
        let ratio = r1.overlap_ratio(&r2);
        // Overlap is 500ms, smaller region is 1000ms → ratio = 0.5.
        assert!(
            (ratio - 0.5).abs() < 1e-9,
            "expected overlap ratio 0.5, got {ratio}"
        );
    }

    #[test]
    fn test_candidate_region_expansion() {
        let region = CandidateRegion {
            start: make_ts(1000),
            end: make_ts(3000),
            avg_score: 0.7,
            peak_score: 0.85,
            frame_count: 5,
        };
        let expanded = region.expanded(0.1);
        // Duration is 2000ms; 10% = 200ms on each side.
        assert!(expanded.start.pts <= region.start.pts);
        assert!(expanded.end.pts >= region.end.pts);
    }

    #[test]
    fn test_quick_score_range() {
        let mut features = SceneFeatures::default();
        features.motion_intensity = 0.8;
        features.audio_peak = 0.7;
        features.face_coverage = 0.5;
        features.color_diversity = 0.4;
        let config = MultiPassConfig::default();
        let detector = MultiPassDetector::new(config);
        let score = detector.quick_score(&features, 0.6);
        assert!((0.0..=1.0).contains(&score));
    }
}
