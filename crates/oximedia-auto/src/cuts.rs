//! Smart cutting and shot boundary detection.
//!
//! This module provides intelligent cutting algorithms for automated video editing:
//!
//! - **Shot boundary detection**: Find natural cut points
//! - **Transition recommendations**: Suggest appropriate transitions
//! - **Beat detection**: Synchronize cuts with music
//! - **Dialogue-aware cuts**: Preserve speech continuity
//! - **Jump cut removal**: Smooth out jarring edits
//!
//! # Example
//!
//! ```
//! use oximedia_auto::cuts::{CutDetector, CutConfig};
//!
//! let config = CutConfig::default();
//! let detector = CutDetector::new(config);
//! ```

use crate::error::{AutoError, AutoResult};
use oximedia_core::Timestamp;
use std::collections::HashMap;

/// Type of cut or transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CutType {
    /// Hard cut - instant transition.
    Hard,
    /// Soft cut - minimal transition.
    Soft,
    /// Fade transition.
    Fade,
    /// Dissolve/crossfade.
    Dissolve,
    /// Wipe transition.
    Wipe,
    /// Jump cut (to be removed).
    Jump,
    /// L-cut (audio leads video).
    LCut,
    /// J-cut (video leads audio).
    JCut,
}

impl CutType {
    /// Check if this is a gradual transition.
    #[must_use]
    pub const fn is_gradual(&self) -> bool {
        matches!(self, Self::Fade | Self::Dissolve | Self::Wipe)
    }

    /// Check if this is a hard cut.
    #[must_use]
    pub const fn is_hard(&self) -> bool {
        matches!(self, Self::Hard | Self::Soft)
    }

    /// Check if this is an audio-video split edit.
    #[must_use]
    pub const fn is_split_edit(&self) -> bool {
        matches!(self, Self::LCut | Self::JCut)
    }

    /// Get the recommended transition duration in milliseconds.
    #[must_use]
    pub const fn default_duration_ms(&self) -> i64 {
        match self {
            Self::Hard | Self::Soft => 0,
            Self::Jump => 0,
            Self::Fade => 500,
            Self::Dissolve => 750,
            Self::Wipe => 600,
            Self::LCut | Self::JCut => 300,
        }
    }
}

/// A detected or recommended cut point.
#[derive(Debug, Clone)]
pub struct CutPoint {
    /// Timestamp of the cut.
    pub timestamp: Timestamp,
    /// Type of cut.
    pub cut_type: CutType,
    /// Confidence in the detection (0.0 to 1.0).
    pub confidence: f64,
    /// Recommended transition duration in milliseconds.
    pub transition_duration_ms: i64,
    /// Whether this cut should be on a beat.
    pub on_beat: bool,
    /// Priority score for this cut (0.0 to 1.0).
    pub priority: f64,
    /// Reason for this cut.
    pub reason: String,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl CutPoint {
    /// Create a new cut point.
    #[must_use]
    pub fn new(timestamp: Timestamp, cut_type: CutType, confidence: f64) -> Self {
        Self {
            timestamp,
            cut_type,
            confidence: confidence.clamp(0.0, 1.0),
            transition_duration_ms: cut_type.default_duration_ms(),
            on_beat: false,
            priority: 0.5,
            reason: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Check if this cut meets a minimum confidence threshold.
    #[must_use]
    pub fn meets_threshold(&self, threshold: f64) -> bool {
        self.confidence >= threshold
    }

    /// Set the transition duration.
    #[must_use]
    pub const fn with_duration(mut self, duration_ms: i64) -> Self {
        self.transition_duration_ms = duration_ms;
        self
    }

    /// Set whether this cut is on a beat.
    #[must_use]
    pub const fn with_beat(mut self, on_beat: bool) -> Self {
        self.on_beat = on_beat;
        self
    }

    /// Set the priority.
    #[must_use]
    pub fn with_priority(mut self, priority: f64) -> Self {
        self.priority = priority.clamp(0.0, 1.0);
        self
    }

    /// Set the reason.
    #[must_use]
    pub fn with_reason<S: Into<String>>(mut self, reason: S) -> Self {
        self.reason = reason.into();
        self
    }
}

/// A detected beat in audio.
#[derive(Debug, Clone, Copy)]
pub struct Beat {
    /// Timestamp of the beat.
    pub timestamp: Timestamp,
    /// Strength of the beat (0.0 to 1.0).
    pub strength: f64,
    /// Whether this is a downbeat.
    pub is_downbeat: bool,
}

impl Beat {
    /// Create a new beat.
    #[must_use]
    pub fn new(timestamp: Timestamp, strength: f64, is_downbeat: bool) -> Self {
        Self {
            timestamp,
            strength: strength.clamp(0.0, 1.0),
            is_downbeat,
        }
    }
}

/// Dialogue segment detected in audio.
#[derive(Debug, Clone)]
pub struct DialogueSegment {
    /// Start timestamp.
    pub start: Timestamp,
    /// End timestamp.
    pub end: Timestamp,
    /// Confidence that this is dialogue (0.0 to 1.0).
    pub confidence: f64,
    /// Speaker ID if available.
    pub speaker_id: Option<usize>,
}

impl DialogueSegment {
    /// Create a new dialogue segment.
    #[must_use]
    pub fn new(start: Timestamp, end: Timestamp, confidence: f64) -> Self {
        Self {
            start,
            end,
            confidence: confidence.clamp(0.0, 1.0),
            speaker_id: None,
        }
    }

    /// Get the duration of this segment.
    #[must_use]
    pub fn duration_ms(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }

    /// Check if a timestamp falls within this segment.
    #[must_use]
    pub fn contains(&self, timestamp: Timestamp) -> bool {
        timestamp.pts >= self.start.pts && timestamp.pts <= self.end.pts
    }
}

/// Shot boundary detection configuration.
#[derive(Debug, Clone)]
pub struct ShotConfig {
    /// Detection threshold (0.0 to 1.0).
    pub threshold: f64,
    /// Minimum shot duration in milliseconds.
    pub min_shot_duration_ms: i64,
    /// Use histogram-based detection.
    pub use_histogram: bool,
    /// Use edge-based detection.
    pub use_edge: bool,
    /// Use motion-based detection.
    pub use_motion: bool,
}

impl Default for ShotConfig {
    fn default() -> Self {
        Self {
            threshold: 0.3,
            min_shot_duration_ms: 500,
            use_histogram: true,
            use_edge: true,
            use_motion: false,
        }
    }
}

/// Beat detection configuration.
#[derive(Debug, Clone)]
pub struct BeatConfig {
    /// Enable beat detection.
    pub enabled: bool,
    /// Tempo range (BPM) - min.
    pub tempo_min_bpm: f64,
    /// Tempo range (BPM) - max.
    pub tempo_max_bpm: f64,
    /// Beat detection sensitivity (0.0 to 1.0).
    pub sensitivity: f64,
}

impl Default for BeatConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            tempo_min_bpm: 60.0,
            tempo_max_bpm: 180.0,
            sensitivity: 0.5,
        }
    }
}

/// Dialogue detection configuration.
#[derive(Debug, Clone)]
pub struct DialogueConfig {
    /// Enable dialogue-aware cutting.
    pub enabled: bool,
    /// Minimum pause duration for cuts (ms).
    pub min_pause_duration_ms: i64,
    /// Voice activity detection threshold.
    pub vad_threshold: f64,
}

impl Default for DialogueConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_pause_duration_ms: 200,
            vad_threshold: 0.5,
        }
    }
}

/// Jump cut detection and removal configuration.
#[derive(Debug, Clone)]
pub struct JumpCutConfig {
    /// Enable jump cut detection.
    pub enabled: bool,
    /// Maximum duration for jump cut (ms).
    pub max_jump_duration_ms: i64,
    /// Similarity threshold for jump detection.
    pub similarity_threshold: f64,
    /// Remove detected jump cuts.
    pub auto_remove: bool,
}

impl Default for JumpCutConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_jump_duration_ms: 500,
            similarity_threshold: 0.85,
            auto_remove: false,
        }
    }
}

/// Configuration for cut detection.
#[derive(Debug, Clone, Default)]
pub struct CutConfig {
    /// Shot boundary detection configuration.
    pub shot: ShotConfig,
    /// Beat detection configuration.
    pub beat: BeatConfig,
    /// Dialogue detection configuration.
    pub dialogue: DialogueConfig,
    /// Jump cut configuration.
    pub jump_cut: JumpCutConfig,
    /// Minimum confidence for cut detection.
    pub min_confidence: f64,
    /// Prefer cuts on beats when available.
    pub prefer_beat_cuts: bool,
    /// Avoid cutting during dialogue.
    pub avoid_dialogue_cuts: bool,
}

impl CutConfig {
    /// Create a new cut configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            min_confidence: 0.6,
            prefer_beat_cuts: true,
            avoid_dialogue_cuts: true,
            ..Default::default()
        }
    }

    /// Set the minimum confidence threshold.
    #[must_use]
    pub const fn with_min_confidence(mut self, confidence: f64) -> Self {
        self.min_confidence = confidence;
        self
    }

    /// Enable or disable beat-aligned cuts.
    #[must_use]
    pub const fn with_prefer_beats(mut self, prefer: bool) -> Self {
        self.prefer_beat_cuts = prefer;
        self
    }

    /// Enable or disable dialogue avoidance.
    #[must_use]
    pub const fn with_avoid_dialogue(mut self, avoid: bool) -> Self {
        self.avoid_dialogue_cuts = avoid;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if !(0.0..=1.0).contains(&self.min_confidence) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.min_confidence,
                min: 0.0,
                max: 1.0,
            });
        }

        if !(0.0..=1.0).contains(&self.shot.threshold) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.shot.threshold,
                min: 0.0,
                max: 1.0,
            });
        }

        if self.shot.min_shot_duration_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.shot.min_shot_duration_ms,
            });
        }

        Ok(())
    }
}

/// Cut detector for intelligent video editing.
pub struct CutDetector {
    /// Configuration.
    config: CutConfig,
}

impl CutDetector {
    /// Create a new cut detector.
    #[must_use]
    pub fn new(config: CutConfig) -> Self {
        Self { config }
    }

    /// Create a cut detector with default configuration.
    #[must_use]
    pub fn default_detector() -> Self {
        Self::new(CutConfig::default())
    }

    /// Detect all cut points in a video.
    ///
    /// # Errors
    ///
    /// Returns an error if configuration is invalid or detection fails.
    pub fn detect_cuts(
        &self,
        scene_changes: &[oximedia_cv::scene::SceneChange],
        beats: Option<&[Beat]>,
        dialogue: Option<&[DialogueSegment]>,
    ) -> AutoResult<Vec<CutPoint>> {
        self.config.validate()?;

        let mut cut_points = Vec::new();

        // Convert scene changes to cut points
        for change in scene_changes {
            let cut_type = if change.change_type.is_gradual() {
                CutType::Dissolve
            } else {
                CutType::Hard
            };

            let mut cut = CutPoint::new(change.timestamp, cut_type, change.confidence);
            cut.reason = format!("Scene change detected ({:?})", change.change_type);
            cut_points.push(cut);
        }

        // Align with beats if available and preferred
        if self.config.prefer_beat_cuts && self.config.beat.enabled {
            if let Some(beats) = beats {
                cut_points = self.align_cuts_to_beats(cut_points, beats);
            }
        }

        // Avoid cuts during dialogue if configured
        if self.config.avoid_dialogue_cuts && self.config.dialogue.enabled {
            if let Some(dialogue_segments) = dialogue {
                cut_points = self.filter_dialogue_cuts(cut_points, dialogue_segments);
            }
        }

        // Detect and handle jump cuts
        if self.config.jump_cut.enabled {
            let jump_cuts = self.detect_jump_cuts(&cut_points);
            if self.config.jump_cut.auto_remove {
                cut_points.retain(|c| !jump_cuts.iter().any(|j| j.timestamp == c.timestamp));
            } else {
                cut_points.extend(jump_cuts);
            }
        }

        // Filter by confidence
        cut_points.retain(|c| c.meets_threshold(self.config.min_confidence));

        // Sort by timestamp
        cut_points.sort_by_key(|c| c.timestamp.pts);

        Ok(cut_points)
    }

    /// Align cut points to nearby beats.
    fn align_cuts_to_beats(&self, mut cuts: Vec<CutPoint>, beats: &[Beat]) -> Vec<CutPoint> {
        const MAX_ALIGNMENT_MS: i64 = 200; // Maximum ms to shift cut to align with beat

        for cut in &mut cuts {
            // Find nearest beat
            if let Some(nearest_beat) = beats
                .iter()
                .min_by_key(|b| (b.timestamp.pts - cut.timestamp.pts).abs())
            {
                let distance = (nearest_beat.timestamp.pts - cut.timestamp.pts).abs();

                if distance <= MAX_ALIGNMENT_MS {
                    cut.timestamp = nearest_beat.timestamp;
                    cut.on_beat = true;
                    cut.priority = (cut.priority + nearest_beat.strength) / 2.0;
                    cut.reason = format!("{} (aligned to beat)", cut.reason);
                }
            }
        }

        cuts
    }

    /// Filter out cuts that occur during dialogue.
    fn filter_dialogue_cuts(
        &self,
        mut cuts: Vec<CutPoint>,
        dialogue: &[DialogueSegment],
    ) -> Vec<CutPoint> {
        cuts.retain(|cut| {
            // Keep cut if it's not during dialogue
            !dialogue.iter().any(|d| d.contains(cut.timestamp))
        });

        // Add recommended L-cuts and J-cuts at dialogue boundaries
        for segment in dialogue {
            if segment.confidence >= self.config.dialogue.vad_threshold {
                // J-cut: video leads audio (cut video before dialogue starts)
                let j_cut = CutPoint::new(
                    Timestamp::new(segment.start.pts - 200, segment.start.timebase),
                    CutType::JCut,
                    segment.confidence,
                )
                .with_reason("J-cut before dialogue");
                cuts.push(j_cut);

                // L-cut: audio leads video (extend audio after video cuts)
                let l_cut = CutPoint::new(
                    Timestamp::new(segment.end.pts + 200, segment.end.timebase),
                    CutType::LCut,
                    segment.confidence,
                )
                .with_reason("L-cut after dialogue");
                cuts.push(l_cut);
            }
        }

        cuts
    }

    /// Detect jump cuts in the cut sequence.
    fn detect_jump_cuts(&self, cuts: &[CutPoint]) -> Vec<CutPoint> {
        let mut jump_cuts = Vec::new();

        for window in cuts.windows(2) {
            let duration = window[1].timestamp.pts - window[0].timestamp.pts;

            if duration > 0
                && duration <= self.config.jump_cut.max_jump_duration_ms
                && window[0].cut_type.is_hard()
                && window[1].cut_type.is_hard()
            {
                let mut jump = CutPoint::new(window[0].timestamp, CutType::Jump, 0.8);
                jump.reason = format!("Jump cut detected ({duration}ms gap)");
                jump_cuts.push(jump);
            }
        }

        jump_cuts
    }

    /// Detect beats in audio samples.
    #[allow(dead_code)]
    pub fn detect_beats(&self, audio_samples: &[f32], sample_rate: u32) -> AutoResult<Vec<Beat>> {
        if !self.config.beat.enabled {
            return Ok(Vec::new());
        }

        let mut beats = Vec::new();

        // Simplified beat detection using onset detection
        // In a real implementation, this would use proper beat tracking algorithms

        let window_size = sample_rate as usize / 10; // 100ms windows
        let mut prev_energy = 0.0;

        for (i, chunk) in audio_samples.chunks(window_size).enumerate() {
            let energy: f32 = chunk.iter().map(|&s| s * s).sum::<f32>() / chunk.len() as f32;
            let energy_f64 = f64::from(energy);

            // Detect sudden energy increase
            if energy_f64 > prev_energy * 1.5 && energy_f64 > self.config.beat.sensitivity {
                let timestamp_ms = (i * window_size * 1000) as i64 / i64::from(sample_rate);
                let timebase = oximedia_core::Rational::new(1, 1000);
                let beat = Beat::new(
                    Timestamp::new(timestamp_ms, timebase),
                    energy_f64.min(1.0),
                    false,
                );
                beats.push(beat);
            }

            prev_energy = energy_f64;
        }

        Ok(beats)
    }

    /// Detect dialogue segments using voice activity detection.
    #[allow(dead_code)]
    pub fn detect_dialogue(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
    ) -> AutoResult<Vec<DialogueSegment>> {
        if !self.config.dialogue.enabled {
            return Ok(Vec::new());
        }

        let mut segments = Vec::new();
        let window_size = sample_rate as usize / 100; // 10ms windows
        let mut in_speech = false;
        let mut speech_start = 0usize;

        for (i, chunk) in audio_samples.chunks(window_size).enumerate() {
            // Simplified VAD using energy threshold
            let energy: f32 = chunk.iter().map(|&s| s * s).sum::<f32>() / chunk.len() as f32;
            let is_speech = f64::from(energy) > self.config.dialogue.vad_threshold;

            if is_speech && !in_speech {
                speech_start = i * window_size;
                in_speech = true;
            } else if !is_speech && in_speech {
                let speech_end = i * window_size;
                let duration_ms =
                    ((speech_end - speech_start) * 1000) as i64 / i64::from(sample_rate);

                if duration_ms >= self.config.dialogue.min_pause_duration_ms {
                    let timebase = oximedia_core::Rational::new(1, 1000);
                    let start_ms = (speech_start * 1000) as i64 / i64::from(sample_rate);
                    let end_ms = (speech_end * 1000) as i64 / i64::from(sample_rate);

                    segments.push(DialogueSegment::new(
                        Timestamp::new(start_ms, timebase),
                        Timestamp::new(end_ms, timebase),
                        0.8,
                    ));
                }

                in_speech = false;
            }
        }

        Ok(segments)
    }

    /// Recommend transition types for cut points.
    pub fn recommend_transitions(&self, cuts: &mut [CutPoint]) {
        for cut in cuts {
            // Default recommendations based on context
            if cut.on_beat {
                // Beats work well with hard cuts
                if cut.cut_type == CutType::Dissolve {
                    cut.cut_type = CutType::Hard;
                }
            }

            // Adjust transition duration based on cut type
            cut.transition_duration_ms = cut.cut_type.default_duration_ms();
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &CutConfig {
        &self.config
    }
}

impl Default for CutDetector {
    fn default() -> Self {
        Self::default_detector()
    }
}

/// Suggest optimal cut points from highlights.
#[allow(dead_code)]
pub fn suggest_cuts_from_highlights(
    highlights: &[crate::highlights::Highlight],
    max_cuts: usize,
) -> Vec<CutPoint> {
    let mut cuts = Vec::new();

    for highlight in highlights {
        // Add cut at start of highlight
        let start_cut = CutPoint::new(highlight.start, CutType::Hard, highlight.confidence)
            .with_priority(highlight.weighted_score())
            .with_reason(format!("Start of {}", highlight.description));
        cuts.push(start_cut);

        // Add cut at end of highlight
        let end_cut = CutPoint::new(highlight.end, CutType::Soft, highlight.confidence)
            .with_priority(highlight.weighted_score() * 0.8)
            .with_reason(format!("End of {}", highlight.description));
        cuts.push(end_cut);
    }

    // Sort by priority and take top N
    cuts.sort_by(|a, b| {
        b.priority
            .partial_cmp(&a.priority)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    cuts.truncate(max_cuts);

    // Re-sort by timestamp
    cuts.sort_by_key(|c| c.timestamp.pts);

    cuts
}
