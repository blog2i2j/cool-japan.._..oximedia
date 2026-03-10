//! Highlight detection for automated video editing.
//!
//! This module provides algorithms for detecting highlights in video content
//! based on multiple analysis techniques:
//!
//! - **Motion analysis**: Detect action-packed moments
//! - **Face detection**: Track subjects and engagement
//! - **Audio peak detection**: Find cheers, applause, and music peaks
//! - **Object detection**: Identify important objects (YOLO integration)
//! - **Scene scoring**: Combine metrics for importance
//!
//! # Example
//!
//! ```
//! use oximedia_auto::highlights::{HighlightDetector, HighlightConfig};
//!
//! let config = HighlightConfig::default();
//! let detector = HighlightDetector::new(config);
//! ```

use crate::error::{AutoError, AutoResult};
use crate::scoring::{ImportanceScore, SceneFeatures};
use oximedia_codec::VideoFrame;
use oximedia_core::Timestamp;
use std::collections::HashMap;

/// Type of highlight detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HighlightType {
    /// High action moment.
    Action,
    /// Face-focused moment.
    Face,
    /// Audio peak (cheers, applause).
    AudioPeak,
    /// Object of interest detected.
    Object,
    /// Composite highlight (multiple factors).
    Composite,
    /// User-marked highlight.
    UserMarked,
}

impl HighlightType {
    /// Get the priority weight for this highlight type.
    #[must_use]
    pub const fn priority_weight(&self) -> f64 {
        match self {
            Self::Composite => 1.0,
            Self::Action => 0.85,
            Self::AudioPeak => 0.80,
            Self::Face => 0.75,
            Self::Object => 0.70,
            Self::UserMarked => 1.0,
        }
    }
}

/// A detected highlight in the video.
#[derive(Debug, Clone)]
pub struct Highlight {
    /// Start timestamp of the highlight.
    pub start: Timestamp,
    /// End timestamp of the highlight.
    pub end: Timestamp,
    /// Type of highlight.
    pub highlight_type: HighlightType,
    /// Importance score (0.0 to 1.0).
    pub score: ImportanceScore,
    /// Confidence level (0.0 to 1.0).
    pub confidence: f64,
    /// Scene features at this moment.
    pub features: SceneFeatures,
    /// Description of the highlight.
    pub description: String,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl Highlight {
    /// Create a new highlight.
    #[must_use]
    pub fn new(
        start: Timestamp,
        end: Timestamp,
        highlight_type: HighlightType,
        score: ImportanceScore,
        confidence: f64,
    ) -> Self {
        Self {
            start,
            end,
            highlight_type,
            score: score.clamp(0.0, 1.0),
            confidence: confidence.clamp(0.0, 1.0),
            features: SceneFeatures::default(),
            description: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get the duration of this highlight in milliseconds.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }

    /// Check if this highlight meets minimum score and confidence thresholds.
    #[must_use]
    pub fn meets_thresholds(&self, min_score: f64, min_confidence: f64) -> bool {
        self.score >= min_score && self.confidence >= min_confidence
    }

    /// Compute weighted score based on type priority.
    #[must_use]
    pub fn weighted_score(&self) -> f64 {
        self.score * self.highlight_type.priority_weight() * self.confidence
    }

    /// Check if this highlight overlaps with another.
    #[must_use]
    pub fn overlaps(&self, other: &Self) -> bool {
        self.start.pts < other.end.pts && self.end.pts > other.start.pts
    }

    /// Merge with another overlapping highlight.
    #[must_use]
    pub fn merge(&self, other: &Self) -> Self {
        let start = Timestamp::new(self.start.pts.min(other.start.pts), self.start.timebase);
        let end = Timestamp::new(self.end.pts.max(other.end.pts), self.end.timebase);

        // Use the higher score and composite type
        let score = self.score.max(other.score);
        let confidence = (self.confidence + other.confidence) / 2.0;

        let mut merged = Self::new(start, end, HighlightType::Composite, score, confidence);
        merged.description = format!("{} + {}", self.description, other.description);

        // Merge features (take max of each)
        merged.features.motion_intensity = self
            .features
            .motion_intensity
            .max(other.features.motion_intensity);
        merged.features.face_count = self.features.face_count.max(other.features.face_count);
        merged.features.face_coverage = self
            .features
            .face_coverage
            .max(other.features.face_coverage);
        merged.features.audio_peak = self.features.audio_peak.max(other.features.audio_peak);
        merged.features.audio_energy = self.features.audio_energy.max(other.features.audio_energy);

        merged
    }
}

/// Motion analysis parameters.
#[derive(Debug, Clone)]
pub struct MotionConfig {
    /// Motion threshold for detection (0.0 to 1.0).
    pub threshold: f64,
    /// Minimum duration in milliseconds.
    pub min_duration_ms: i64,
    /// Use optical flow for motion estimation.
    pub use_optical_flow: bool,
    /// Block size for motion estimation.
    pub block_size: usize,
}

impl Default for MotionConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            min_duration_ms: 300,
            use_optical_flow: true,
            block_size: 16,
        }
    }
}

/// Face detection parameters.
#[derive(Debug, Clone)]
pub struct FaceConfig {
    /// Minimum face coverage ratio (0.0 to 1.0).
    pub min_coverage: f64,
    /// Minimum face count.
    pub min_face_count: usize,
    /// Track faces across frames.
    pub enable_tracking: bool,
    /// Face detection confidence threshold.
    pub confidence_threshold: f64,
}

impl Default for FaceConfig {
    fn default() -> Self {
        Self {
            min_coverage: 0.1,
            min_face_count: 1,
            enable_tracking: true,
            confidence_threshold: 0.7,
        }
    }
}

/// Audio peak detection parameters.
#[derive(Debug, Clone)]
pub struct AudioPeakConfig {
    /// Peak threshold (0.0 to 1.0).
    pub threshold: f64,
    /// Minimum peak duration in milliseconds.
    pub min_duration_ms: i64,
    /// Look for sustained energy vs instant peaks.
    pub sustained_energy: bool,
    /// Energy window size in milliseconds.
    pub energy_window_ms: i64,
}

impl Default for AudioPeakConfig {
    fn default() -> Self {
        Self {
            threshold: 0.6,
            min_duration_ms: 200,
            sustained_energy: true,
            energy_window_ms: 500,
        }
    }
}

/// Object detection parameters.
#[derive(Debug, Clone)]
pub struct ObjectConfig {
    /// Enable object detection.
    pub enabled: bool,
    /// Object classes to detect (empty = all).
    pub target_classes: Vec<String>,
    /// Minimum detection confidence.
    pub confidence_threshold: f64,
    /// Minimum object size ratio (0.0 to 1.0).
    pub min_size_ratio: f64,
}

impl Default for ObjectConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_classes: Vec::new(),
            confidence_threshold: 0.6,
            min_size_ratio: 0.05,
        }
    }
}

/// Configuration for highlight detection.
#[derive(Debug, Clone)]
pub struct HighlightConfig {
    /// Motion analysis configuration.
    pub motion: MotionConfig,
    /// Face detection configuration.
    pub face: FaceConfig,
    /// Audio peak configuration.
    pub audio_peak: AudioPeakConfig,
    /// Object detection configuration.
    pub object: ObjectConfig,
    /// Minimum highlight score (0.0 to 1.0).
    pub min_score: f64,
    /// Minimum confidence (0.0 to 1.0).
    pub min_confidence: f64,
    /// Merge overlapping highlights.
    pub merge_overlaps: bool,
    /// Maximum gap to merge highlights (ms).
    pub merge_gap_ms: i64,
}

impl Default for HighlightConfig {
    fn default() -> Self {
        Self {
            motion: MotionConfig::default(),
            face: FaceConfig::default(),
            audio_peak: AudioPeakConfig::default(),
            object: ObjectConfig::default(),
            min_score: 0.5,
            min_confidence: 0.6,
            merge_overlaps: true,
            merge_gap_ms: 500,
        }
    }
}

impl HighlightConfig {
    /// Create a new highlight configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the minimum score threshold.
    #[must_use]
    pub const fn with_min_score(mut self, min_score: f64) -> Self {
        self.min_score = min_score;
        self
    }

    /// Set the minimum confidence threshold.
    #[must_use]
    pub const fn with_min_confidence(mut self, min_confidence: f64) -> Self {
        self.min_confidence = min_confidence;
        self
    }

    /// Enable or disable overlap merging.
    #[must_use]
    pub const fn with_merge_overlaps(mut self, merge: bool) -> Self {
        self.merge_overlaps = merge;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if !(0.0..=1.0).contains(&self.min_score) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.min_score,
                min: 0.0,
                max: 1.0,
            });
        }

        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.min_confidence,
                min: 0.0,
                max: 1.0,
            });
        }

        if !(0.0..=1.0).contains(&self.motion.threshold) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.motion.threshold,
                min: 0.0,
                max: 1.0,
            });
        }

        if self.motion.min_duration_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.motion.min_duration_ms,
            });
        }

        Ok(())
    }
}

/// Highlight detector for video analysis.
pub struct HighlightDetector {
    /// Configuration.
    config: HighlightConfig,
}

impl HighlightDetector {
    /// Create a new highlight detector.
    #[must_use]
    pub fn new(config: HighlightConfig) -> Self {
        Self { config }
    }

    /// Create a highlight detector with default configuration.
    #[must_use]
    pub fn default_detector() -> Self {
        Self::new(HighlightConfig::default())
    }

    /// Detect all highlights in a video sequence.
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails or configuration is invalid.
    pub fn detect_highlights(&self, frames: &[VideoFrame]) -> AutoResult<Vec<Highlight>> {
        self.config.validate()?;

        if frames.is_empty() {
            return Err(AutoError::insufficient_data("No frames provided"));
        }

        let mut highlights = Vec::new();

        // Detect motion-based highlights
        let motion_highlights = self.detect_motion_highlights(frames)?;
        highlights.extend(motion_highlights);

        // Detect face-based highlights
        let face_highlights = self.detect_face_highlights(frames)?;
        highlights.extend(face_highlights);

        // Detect object-based highlights
        if self.config.object.enabled {
            let object_highlights = self.detect_object_highlights(frames)?;
            highlights.extend(object_highlights);
        }

        // Filter by score and confidence
        highlights
            .retain(|h| h.meets_thresholds(self.config.min_score, self.config.min_confidence));

        // Merge overlapping highlights if configured
        if self.config.merge_overlaps {
            highlights = self.merge_overlapping_highlights(highlights);
        }

        // Sort by start time
        highlights.sort_by_key(|h| h.start.pts);

        if highlights.is_empty() {
            return Err(AutoError::NoHighlights);
        }

        Ok(highlights)
    }

    /// Detect highlights based on motion intensity.
    fn detect_motion_highlights(&self, frames: &[VideoFrame]) -> AutoResult<Vec<Highlight>> {
        let mut highlights = Vec::new();
        let mut current_highlight: Option<(usize, f64)> = None;

        for (i, frame) in frames.iter().enumerate() {
            let motion = self.estimate_motion(frame, frames.get(i + 1))?;

            if motion > self.config.motion.threshold {
                match current_highlight {
                    None => {
                        current_highlight = Some((i, motion));
                    }
                    Some((start_idx, max_motion)) => {
                        current_highlight = Some((start_idx, max_motion.max(motion)));
                    }
                }
            } else if let Some((start_idx, max_motion)) = current_highlight {
                // End of motion highlight
                let duration_ms =
                    (frames[i].timestamp.pts - frames[start_idx].timestamp.pts).max(0);

                if duration_ms >= self.config.motion.min_duration_ms {
                    let mut highlight = Highlight::new(
                        frames[start_idx].timestamp,
                        frames[i].timestamp,
                        HighlightType::Action,
                        max_motion,
                        0.8,
                    );
                    highlight.features.motion_intensity = max_motion;
                    highlight.description = format!("Action sequence ({duration_ms} ms)");
                    highlights.push(highlight);
                }

                current_highlight = None;
            }
        }

        Ok(highlights)
    }

    /// Estimate motion intensity between frames.
    fn estimate_motion(
        &self,
        _frame1: &VideoFrame,
        frame2: Option<&VideoFrame>,
    ) -> AutoResult<f64> {
        if frame2.is_none() {
            return Ok(0.0);
        }

        // Simplified motion estimation
        // In a real implementation, this would use optical flow or block matching
        let motion = 0.5; // Placeholder

        Ok(motion)
    }

    /// Detect highlights based on face presence and coverage.
    fn detect_face_highlights(&self, frames: &[VideoFrame]) -> AutoResult<Vec<Highlight>> {
        let mut highlights = Vec::new();
        let mut current_highlight: Option<(usize, usize, f64)> = None;

        for (i, _frame) in frames.iter().enumerate() {
            // Simplified face detection
            // In a real implementation, this would use actual face detection
            let (face_count, coverage) = self.detect_faces_in_frame(i)?;

            if face_count >= self.config.face.min_face_count
                && coverage >= self.config.face.min_coverage
            {
                match current_highlight {
                    None => {
                        current_highlight = Some((i, face_count, coverage));
                    }
                    Some((start, max_faces, max_coverage)) => {
                        current_highlight =
                            Some((start, max_faces.max(face_count), max_coverage.max(coverage)));
                    }
                }
            } else if let Some((start_idx, max_faces, max_coverage)) = current_highlight {
                let mut highlight = Highlight::new(
                    frames[start_idx].timestamp,
                    frames[i].timestamp,
                    HighlightType::Face,
                    max_coverage,
                    self.config.face.confidence_threshold,
                );
                highlight.features.face_count = max_faces;
                highlight.features.face_coverage = max_coverage;
                highlight.description = format!("{max_faces} face(s) detected");
                highlights.push(highlight);

                current_highlight = None;
            }
        }

        Ok(highlights)
    }

    /// Detect faces in a single frame.
    fn detect_faces_in_frame(&self, _frame_idx: usize) -> AutoResult<(usize, f64)> {
        // Placeholder for actual face detection
        Ok((0, 0.0))
    }

    /// Detect highlights based on object detection.
    fn detect_object_highlights(&self, frames: &[VideoFrame]) -> AutoResult<Vec<Highlight>> {
        let highlights = Vec::new();

        // Placeholder for object detection integration
        // In a real implementation, this would use YOLO or similar

        for _frame in frames {
            // Object detection logic here
        }

        Ok(highlights)
    }

    /// Detect highlights from audio peaks.
    #[allow(dead_code)]
    pub fn detect_audio_highlights(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
    ) -> AutoResult<Vec<Highlight>> {
        let mut highlights = Vec::new();

        if audio_samples.is_empty() {
            return Ok(highlights);
        }

        let samples_per_ms = sample_rate as usize / 1000;
        let window_samples = (self.config.audio_peak.energy_window_ms as usize) * samples_per_ms;

        let mut current_highlight: Option<(usize, f64)> = None;

        for i in (0..audio_samples.len()).step_by(window_samples.max(1)) {
            let end = (i + window_samples).min(audio_samples.len());
            let window = &audio_samples[i..end];

            let energy = if self.config.audio_peak.sustained_energy {
                // RMS energy
                (window.iter().map(|&s| s * s).sum::<f32>() / window.len() as f32).sqrt()
            } else {
                // Peak amplitude
                window
                    .iter()
                    .map(|&s| s.abs())
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or(0.0)
            };

            let energy_f64 = f64::from(energy);

            if energy_f64 > self.config.audio_peak.threshold {
                match current_highlight {
                    None => {
                        current_highlight = Some((i, energy_f64));
                    }
                    Some((start_sample, max_energy)) => {
                        current_highlight = Some((start_sample, max_energy.max(energy_f64)));
                    }
                }
            } else if let Some((start_sample, max_energy)) = current_highlight {
                let duration_ms = ((i - start_sample) as i64 * 1000) / i64::from(sample_rate);

                if duration_ms >= self.config.audio_peak.min_duration_ms {
                    let start_ms = (start_sample as i64 * 1000) / i64::from(sample_rate);
                    let end_ms = (i as i64 * 1000) / i64::from(sample_rate);

                    let timebase = oximedia_core::Rational::new(1, 1000);
                    let mut highlight = Highlight::new(
                        Timestamp::new(start_ms, timebase),
                        Timestamp::new(end_ms, timebase),
                        HighlightType::AudioPeak,
                        max_energy,
                        0.85,
                    );
                    highlight.features.audio_peak = max_energy;
                    highlight.features.audio_energy = max_energy;
                    highlight.description = format!("Audio peak ({duration_ms}ms)");
                    highlights.push(highlight);
                }

                current_highlight = None;
            }
        }

        Ok(highlights)
    }

    /// Merge overlapping highlights.
    fn merge_overlapping_highlights(&self, mut highlights: Vec<Highlight>) -> Vec<Highlight> {
        if highlights.len() < 2 {
            return highlights;
        }

        highlights.sort_by_key(|h| h.start.pts);

        let mut merged = Vec::new();
        let mut current = highlights[0].clone();

        for highlight in highlights.into_iter().skip(1) {
            let gap = highlight.start.pts - current.end.pts;

            if gap <= self.config.merge_gap_ms {
                current = current.merge(&highlight);
            } else {
                merged.push(current);
                current = highlight;
            }
        }

        merged.push(current);
        merged
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &HighlightConfig {
        &self.config
    }
}

impl Default for HighlightDetector {
    fn default() -> Self {
        Self::default_detector()
    }
}

/// Extract highlights from scored scenes.
#[allow(dead_code)]
pub fn highlights_from_scores(
    scenes: &[crate::scoring::ScoredScene],
    threshold: ImportanceScore,
) -> Vec<Highlight> {
    scenes
        .iter()
        .filter(|s| s.score >= threshold)
        .map(|s| {
            let mut highlight =
                Highlight::new(s.start, s.end, HighlightType::Composite, s.score, 0.9);
            highlight.features = s.features.clone();
            highlight.description = s.suggested_title.clone().unwrap_or_default();
            highlight
        })
        .collect()
}
