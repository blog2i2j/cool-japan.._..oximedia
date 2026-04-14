//! Lookahead buffer for scene-adaptive CRF adjustment.
//!
//! Traditional constant-rate-factor (CRF) encoding uses the same quality
//! target throughout the entire encode.  A *lookahead buffer* allows the
//! encoder to inspect a short window of upcoming frames before committing to
//! a CRF value, enabling it to:
//!
//! - Temporarily lower the CRF (higher quality) just before a scene cut so
//!   the first frames of the new scene look crisp.
//! - Raise the CRF (lower quality / smaller file) during visually uniform
//!   sections (e.g. talking-head footage with a static background) to
//!   reclaim bits.
//! - Stay within a configurable bitrate budget by accumulating "saved" bits
//!   and spending them on complex scenes.
//!
//! # Architecture
//!
//! ```text
//!  ┌───────────────────────────────────────────────────────────┐
//!  │                   LookaheadBuffer                         │
//!  │                                                           │
//!  │  deque: [F₀, F₁, F₂, … Fₙ]   ← push_back               │
//!  │            ↑ front = next frame to encode                 │
//!  │                                                           │
//!  │  analyse_window()  ─▶  SceneComplexity                    │
//!  │  suggest_crf()     ─▶  u8  (adjusted CRF value)          │
//!  └───────────────────────────────────────────────────────────┘
//! ```
//!
//! [`LookaheadBuffer`] is intentionally codec-agnostic.  The caller (e.g.
//! the CRF optimiser or a per-scene encoder) feeds [`FrameFeatures`] into
//! the buffer and queries it for an adjusted CRF before handing each frame
//! to the codec.

use std::collections::VecDeque;

use serde::{Deserialize, Serialize};

use crate::TranscodeError;

// ---------------------------------------------------------------------------
// Frame features
// ---------------------------------------------------------------------------

/// Lightweight per-frame feature vector extracted from a decoded frame.
///
/// All values are normalised to `[0.0, 1.0]` unless stated otherwise.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrameFeatures {
    /// Frame index (display order, 0-based).
    pub frame_index: u64,
    /// Spatial complexity estimate: 0.0 (uniform) → 1.0 (very detailed).
    ///
    /// A simple implementation can derive this from the DCT energy or the
    /// mean absolute difference of 8×8 blocks.
    pub spatial_complexity: f32,
    /// Temporal complexity estimate: 0.0 (static) → 1.0 (lots of motion).
    ///
    /// Usually derived from the motion vector magnitude distribution.
    pub temporal_complexity: f32,
    /// Mean luma of the frame (0.0 = black, 1.0 = white).
    pub mean_luma: f32,
    /// Confidence that a scene cut precedes this frame (`[0.0, 1.0]`).
    pub scene_cut_score: f32,
}

impl FrameFeatures {
    /// Returns the combined complexity as a weighted sum.
    ///
    /// The weights are 60 % spatial, 40 % temporal — reflecting that spatial
    /// detail is generally harder to compress.
    #[must_use]
    pub fn combined_complexity(&self) -> f32 {
        self.spatial_complexity * 0.6 + self.temporal_complexity * 0.4
    }

    /// Returns `true` if this frame is likely the start of a new scene.
    #[must_use]
    pub fn is_scene_cut(&self, threshold: f32) -> bool {
        self.scene_cut_score >= threshold
    }
}

// ---------------------------------------------------------------------------
// Window analysis result
// ---------------------------------------------------------------------------

/// Aggregate statistics about the current lookahead window.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowAnalysis {
    /// Number of frames currently in the window.
    pub frame_count: usize,
    /// Mean combined complexity across all frames in the window.
    pub mean_complexity: f32,
    /// Peak combined complexity in the window.
    pub peak_complexity: f32,
    /// Number of frames whose `scene_cut_score` exceeds the configured
    /// threshold.
    pub scene_cut_count: usize,
    /// Whether a scene cut is *imminent* (i.e. appears within the first
    /// `imminent_frames` positions).
    pub scene_cut_imminent: bool,
    /// Mean scene-cut score across the window.
    pub mean_scene_cut_score: f32,
}

// ---------------------------------------------------------------------------
// CRF adjustment strategy
// ---------------------------------------------------------------------------

/// How the buffer decides to adjust the CRF.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CrfAdjustStrategy {
    /// Adjust purely based on spatial complexity.
    ComplexityBased,
    /// Adjust based on spatial complexity + scene-cut lookahead.
    SceneAware,
    /// Aggressively lower CRF before every scene cut for maximum quality at
    /// scene boundaries.
    SceneCutOptimise,
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`LookaheadBuffer`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LookaheadConfig {
    /// Number of frames to buffer.  Larger values give better decisions at
    /// the cost of latency and memory.
    pub window_size: usize,
    /// Base CRF value (the encode-wide default).
    pub base_crf: u8,
    /// Minimum allowable CRF (highest quality).
    pub min_crf: u8,
    /// Maximum allowable CRF (lowest quality).
    pub max_crf: u8,
    /// Maximum amount the CRF may change relative to the base value.
    pub max_crf_delta: u8,
    /// Complexity threshold above which the CRF is lowered.
    pub high_complexity_threshold: f32,
    /// Complexity threshold below which the CRF is raised.
    pub low_complexity_threshold: f32,
    /// Scene-cut score above which a cut is declared.
    pub scene_cut_threshold: f32,
    /// Number of frames ahead that count as "imminent" for a scene cut.
    pub imminent_frames: usize,
    /// CRF adjustment strategy.
    pub strategy: CrfAdjustStrategy,
}

impl Default for LookaheadConfig {
    fn default() -> Self {
        Self {
            window_size: 24,
            base_crf: 28,
            min_crf: 15,
            max_crf: 51,
            max_crf_delta: 6,
            high_complexity_threshold: 0.65,
            low_complexity_threshold: 0.30,
            scene_cut_threshold: 0.75,
            imminent_frames: 4,
            strategy: CrfAdjustStrategy::SceneAware,
        }
    }
}

impl LookaheadConfig {
    /// Convenience constructor for H.264 / VP9 CRF range (0-63).
    #[must_use]
    pub fn for_vp9(base_crf: u8) -> Self {
        Self {
            base_crf,
            min_crf: 0,
            max_crf: 63,
            max_crf_delta: 8,
            ..Self::default()
        }
    }

    /// Convenience constructor for AV1 (0-63 q-index).
    #[must_use]
    pub fn for_av1(base_crf: u8) -> Self {
        Self::for_vp9(base_crf)
    }
}

// ---------------------------------------------------------------------------
// Main buffer
// ---------------------------------------------------------------------------

/// A fixed-capacity lookahead buffer that emits scene-adaptive CRF suggestions.
///
/// Frames are pushed in display order via [`push`][`LookaheadBuffer::push`] and
/// consumed one at a time via [`pop_with_crf`][`LookaheadBuffer::pop_with_crf`].
/// While the buffer has fewer frames than `window_size` it is considered
/// *priming* and suggestions are still provided but with reduced confidence.
pub struct LookaheadBuffer {
    config: LookaheadConfig,
    window: VecDeque<FrameFeatures>,
    /// Running bit-budget accumulator.  Positive = saved bits; negative = debt.
    bit_budget: f64,
    /// Number of frames that have been popped (emitted) so far.
    frames_emitted: u64,
    /// Last suggested CRF (used for smoothing).
    last_crf: u8,
}

impl LookaheadBuffer {
    /// Creates a new buffer with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::InvalidInput`] if `window_size` is 0 or if
    /// `min_crf > max_crf`.
    pub fn new(config: LookaheadConfig) -> Result<Self, TranscodeError> {
        if config.window_size == 0 {
            return Err(TranscodeError::InvalidInput(
                "LookaheadBuffer: window_size must be > 0".into(),
            ));
        }
        if config.min_crf > config.max_crf {
            return Err(TranscodeError::InvalidInput(format!(
                "LookaheadBuffer: min_crf ({}) > max_crf ({})",
                config.min_crf, config.max_crf
            )));
        }
        let base = config.base_crf;
        Ok(Self {
            config,
            window: VecDeque::new(),
            bit_budget: 0.0,
            frames_emitted: 0,
            last_crf: base,
        })
    }

    /// Creates a buffer with default configuration.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::new`].
    pub fn with_defaults() -> Result<Self, TranscodeError> {
        Self::new(LookaheadConfig::default())
    }

    /// Pushes `frame` into the lookahead window.
    ///
    /// If the buffer is already full the oldest frame is silently dropped to
    /// make room (this should not happen in a properly pipelined encoder, but
    /// provides a safety valve).
    pub fn push(&mut self, frame: FrameFeatures) {
        if self.window.len() >= self.config.window_size {
            self.window.pop_front();
        }
        self.window.push_back(frame);
    }

    /// Returns the number of frames currently in the buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.window.len()
    }

    /// Returns `true` when the buffer is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.window.is_empty()
    }

    /// Returns `true` while the buffer has not yet accumulated a full window.
    #[must_use]
    pub fn is_priming(&self) -> bool {
        self.window.len() < self.config.window_size
    }

    /// How many frames have been emitted so far.
    #[must_use]
    pub fn frames_emitted(&self) -> u64 {
        self.frames_emitted
    }

    /// Analyses the current window contents and returns aggregate statistics.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::PipelineError`] when the buffer is empty.
    pub fn analyse_window(&self) -> Result<WindowAnalysis, TranscodeError> {
        if self.window.is_empty() {
            return Err(TranscodeError::PipelineError(
                "LookaheadBuffer: cannot analyse empty window".into(),
            ));
        }

        let n = self.window.len() as f32;
        let mut sum_complexity = 0.0f32;
        let mut peak_complexity = 0.0f32;
        let mut sum_scene_cut = 0.0f32;
        let mut scene_cut_count = 0usize;

        for f in &self.window {
            let c = f.combined_complexity();
            sum_complexity += c;
            if c > peak_complexity {
                peak_complexity = c;
            }
            sum_scene_cut += f.scene_cut_score;
            if f.is_scene_cut(self.config.scene_cut_threshold) {
                scene_cut_count += 1;
            }
        }

        // "Imminent" = a scene cut within the first `imminent_frames` frames.
        let imminent = self
            .window
            .iter()
            .take(self.config.imminent_frames)
            .any(|f| f.is_scene_cut(self.config.scene_cut_threshold));

        Ok(WindowAnalysis {
            frame_count: self.window.len(),
            mean_complexity: sum_complexity / n,
            peak_complexity,
            scene_cut_count,
            scene_cut_imminent: imminent,
            mean_scene_cut_score: sum_scene_cut / n,
        })
    }

    /// Removes and returns the front (oldest) frame together with a suggested
    /// CRF value for encoding it.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::PipelineError`] when the buffer is empty.
    pub fn pop_with_crf(&mut self) -> Result<(FrameFeatures, u8), TranscodeError> {
        let analysis = self.analyse_window()?;
        let crf = self.compute_crf(&analysis);

        let frame = self
            .window
            .pop_front()
            .ok_or_else(|| TranscodeError::PipelineError("buffer empty".into()))?;

        self.last_crf = crf;
        self.frames_emitted += 1;

        // Update bit budget: positive delta means we used fewer bits than baseline.
        let baseline = self.config.base_crf as f64;
        self.bit_budget += baseline - crf as f64;

        Ok((frame, crf))
    }

    /// Peeks at the suggested CRF for the current front frame *without*
    /// consuming it.
    ///
    /// # Errors
    ///
    /// Returns [`TranscodeError::PipelineError`] when the buffer is empty.
    pub fn peek_crf(&self) -> Result<u8, TranscodeError> {
        let analysis = self.analyse_window()?;
        Ok(self.compute_crf(&analysis))
    }

    /// Flushes all remaining frames, returning each with its CRF suggestion.
    ///
    /// After this call the buffer is empty.
    pub fn drain_with_crf(&mut self) -> Vec<(FrameFeatures, u8)> {
        let mut results = Vec::with_capacity(self.window.len());
        while !self.window.is_empty() {
            match self.pop_with_crf() {
                Ok(pair) => results.push(pair),
                Err(_) => break,
            }
        }
        results
    }

    /// Resets the buffer, clearing all frames and resetting state.
    pub fn reset(&mut self) {
        self.window.clear();
        self.bit_budget = 0.0;
        self.frames_emitted = 0;
        self.last_crf = self.config.base_crf;
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    /// Core CRF decision logic.
    fn compute_crf(&self, analysis: &WindowAnalysis) -> u8 {
        let base = self.config.base_crf as i16;
        let max_delta = self.config.max_crf_delta as i16;

        let mut delta: i16 = match self.config.strategy {
            CrfAdjustStrategy::ComplexityBased => {
                self.complexity_delta(analysis.mean_complexity)
            }
            CrfAdjustStrategy::SceneAware => {
                let mut d = self.complexity_delta(analysis.mean_complexity);
                if analysis.scene_cut_imminent {
                    // Lower CRF (higher quality) to anticipate the cut.
                    d -= 2;
                }
                d
            }
            CrfAdjustStrategy::SceneCutOptimise => {
                if analysis.scene_cut_imminent {
                    // Maximum quality drop at scene boundaries.
                    -(max_delta)
                } else {
                    self.complexity_delta(analysis.mean_complexity)
                }
            }
        };

        // Clamp delta.
        delta = delta.clamp(-max_delta, max_delta);

        // Apply hysteresis: only change by more than 1 if the suggestion
        // differs significantly from last CRF to avoid oscillation.
        let candidate = base + delta;
        let last = self.last_crf as i16;
        let smoothed = if (candidate - last).abs() <= 1 {
            candidate
        } else {
            // Move toward candidate by at most half the distance per frame.
            last + (candidate - last) / 2
        };

        // Clamp to configured range.
        let clamped = smoothed.clamp(
            self.config.min_crf as i16,
            self.config.max_crf as i16,
        );
        clamped as u8
    }

    /// Returns a CRF delta based on the given complexity score.
    ///
    /// - High complexity → negative delta (lower CRF = higher quality).
    /// - Low complexity  → positive delta (higher CRF = smaller file).
    fn complexity_delta(&self, complexity: f32) -> i16 {
        let max_delta = self.config.max_crf_delta as i16;
        if complexity >= self.config.high_complexity_threshold {
            // Map [high_threshold, 1.0] → [-1, -max_delta].
            let range = 1.0 - self.config.high_complexity_threshold;
            let t = if range > 0.0 {
                ((complexity - self.config.high_complexity_threshold) / range).clamp(0.0, 1.0)
            } else {
                1.0
            };
            -((1.0 + t * (max_delta - 1) as f32).round() as i16)
        } else if complexity <= self.config.low_complexity_threshold {
            // Map [0.0, low_threshold] → [max_delta, 1].
            let range = self.config.low_complexity_threshold;
            let t = if range > 0.0 {
                (complexity / range).clamp(0.0, 1.0)
            } else {
                0.0
            };
            ((max_delta as f32 * (1.0 - t)).round() as i16).max(1)
        } else {
            // Mid-range: no adjustment.
            0
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_frame(idx: u64, spatial: f32, temporal: f32, scene_cut: f32) -> FrameFeatures {
        FrameFeatures {
            frame_index: idx,
            spatial_complexity: spatial,
            temporal_complexity: temporal,
            mean_luma: 0.5,
            scene_cut_score: scene_cut,
        }
    }

    #[test]
    fn test_new_with_zero_window_size_fails() {
        let cfg = LookaheadConfig {
            window_size: 0,
            ..LookaheadConfig::default()
        };
        assert!(LookaheadBuffer::new(cfg).is_err());
    }

    #[test]
    fn test_new_with_invalid_crf_range_fails() {
        let cfg = LookaheadConfig {
            min_crf: 40,
            max_crf: 10,
            ..LookaheadConfig::default()
        };
        assert!(LookaheadBuffer::new(cfg).is_err());
    }

    #[test]
    fn test_empty_buffer_returns_error_on_pop() {
        let mut buf = LookaheadBuffer::with_defaults().expect("defaults ok");
        assert!(buf.pop_with_crf().is_err());
    }

    #[test]
    fn test_push_and_pop_frame_count() {
        let mut buf = LookaheadBuffer::with_defaults().expect("defaults ok");
        for i in 0..5 {
            buf.push(simple_frame(i, 0.5, 0.3, 0.0));
        }
        assert_eq!(buf.len(), 5);
        let _ = buf.pop_with_crf().expect("pop ok");
        assert_eq!(buf.len(), 4);
        assert_eq!(buf.frames_emitted(), 1);
    }

    #[test]
    fn test_high_complexity_lowers_crf() {
        let cfg = LookaheadConfig {
            base_crf: 28,
            min_crf: 15,
            max_crf: 51,
            max_crf_delta: 6,
            strategy: CrfAdjustStrategy::ComplexityBased,
            ..LookaheadConfig::default()
        };
        let mut buf = LookaheadBuffer::new(cfg).expect("ok");
        // Fill with high-complexity frames.
        for i in 0..8 {
            buf.push(simple_frame(i, 0.9, 0.8, 0.0));
        }
        let (_, crf) = buf.pop_with_crf().expect("pop ok");
        assert!(crf < 28, "high complexity should lower CRF below base (got {crf})");
    }

    #[test]
    fn test_low_complexity_raises_crf() {
        let cfg = LookaheadConfig {
            base_crf: 28,
            min_crf: 15,
            max_crf: 51,
            max_crf_delta: 6,
            strategy: CrfAdjustStrategy::ComplexityBased,
            ..LookaheadConfig::default()
        };
        let mut buf = LookaheadBuffer::new(cfg).expect("ok");
        // Fill with low-complexity frames (talking head, static background).
        for i in 0..8 {
            buf.push(simple_frame(i, 0.1, 0.05, 0.0));
        }
        let (_, crf) = buf.pop_with_crf().expect("pop ok");
        assert!(crf > 28, "low complexity should raise CRF above base (got {crf})");
    }

    #[test]
    fn test_scene_cut_imminent_lowers_crf() {
        let cfg = LookaheadConfig {
            base_crf: 28,
            min_crf: 15,
            max_crf: 51,
            max_crf_delta: 6,
            strategy: CrfAdjustStrategy::SceneAware,
            imminent_frames: 4,
            scene_cut_threshold: 0.75,
            ..LookaheadConfig::default()
        };
        let mut buf = LookaheadBuffer::new(cfg).expect("ok");
        // First frame has medium complexity.
        buf.push(simple_frame(0, 0.5, 0.4, 0.0));
        // Scene cut imminent (score 0.9 > threshold 0.75) at position 1.
        buf.push(simple_frame(1, 0.5, 0.4, 0.9));
        for i in 2..8 {
            buf.push(simple_frame(i, 0.5, 0.4, 0.0));
        }
        let (_, crf) = buf.pop_with_crf().expect("pop ok");
        // Should be lower than base due to imminent scene cut.
        assert!(crf <= 28, "scene cut should not raise CRF above base (got {crf})");
    }

    #[test]
    fn test_scene_cut_optimise_maximises_quality_at_cut() {
        let cfg = LookaheadConfig {
            base_crf: 28,
            min_crf: 15,
            max_crf: 51,
            max_crf_delta: 6,
            strategy: CrfAdjustStrategy::SceneCutOptimise,
            imminent_frames: 4,
            scene_cut_threshold: 0.75,
            ..LookaheadConfig::default()
        };
        let mut buf = LookaheadBuffer::new(cfg).expect("ok");
        buf.push(simple_frame(0, 0.5, 0.3, 0.0));
        buf.push(simple_frame(1, 0.5, 0.3, 0.95)); // scene cut
        for i in 2..8 {
            buf.push(simple_frame(i, 0.5, 0.3, 0.0));
        }
        let crf = buf.peek_crf().expect("peek ok");
        assert!(crf < 28, "SceneCutOptimise should drop CRF significantly (got {crf})");
    }

    #[test]
    fn test_crf_stays_within_bounds() {
        let cfg = LookaheadConfig {
            base_crf: 28,
            min_crf: 20,
            max_crf: 35,
            max_crf_delta: 10,
            strategy: CrfAdjustStrategy::ComplexityBased,
            ..LookaheadConfig::default()
        };
        let mut buf = LookaheadBuffer::new(cfg.clone()).expect("ok");
        // Extreme high complexity.
        for i in 0..10 {
            buf.push(simple_frame(i, 1.0, 1.0, 0.0));
        }
        let (_, crf) = buf.pop_with_crf().expect("pop ok");
        assert!(
            crf >= cfg.min_crf && crf <= cfg.max_crf,
            "CRF {crf} should be within [{}, {}]",
            cfg.min_crf,
            cfg.max_crf
        );
    }

    #[test]
    fn test_analyse_window_counts_scene_cuts() {
        let mut buf = LookaheadBuffer::with_defaults().expect("defaults ok");
        buf.push(simple_frame(0, 0.5, 0.3, 0.0));
        buf.push(simple_frame(1, 0.5, 0.3, 0.9)); // cut
        buf.push(simple_frame(2, 0.5, 0.3, 0.0));
        buf.push(simple_frame(3, 0.5, 0.3, 0.8)); // cut
        let analysis = buf.analyse_window().expect("analysis ok");
        assert_eq!(analysis.scene_cut_count, 2);
        assert!(analysis.scene_cut_imminent, "cut at position 1 is imminent");
    }

    #[test]
    fn test_drain_with_crf_empties_buffer() {
        let mut buf = LookaheadBuffer::with_defaults().expect("defaults ok");
        for i in 0..6 {
            buf.push(simple_frame(i, 0.4, 0.3, 0.0));
        }
        let drained = buf.drain_with_crf();
        assert_eq!(drained.len(), 6);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_reset_clears_state() {
        let mut buf = LookaheadBuffer::with_defaults().expect("defaults ok");
        for i in 0..4 {
            buf.push(simple_frame(i, 0.6, 0.4, 0.0));
        }
        let _ = buf.pop_with_crf().expect("pop ok");
        buf.reset();
        assert!(buf.is_empty());
        assert_eq!(buf.frames_emitted(), 0);
    }

    #[test]
    fn test_window_size_capping() {
        let cfg = LookaheadConfig {
            window_size: 3,
            ..LookaheadConfig::default()
        };
        let mut buf = LookaheadBuffer::new(cfg).expect("ok");
        for i in 0..10 {
            buf.push(simple_frame(i, 0.5, 0.3, 0.0));
        }
        // Window should be capped at 3.
        assert_eq!(buf.len(), 3);
    }

    #[test]
    fn test_frame_features_combined_complexity() {
        let f = FrameFeatures {
            frame_index: 0,
            spatial_complexity: 0.8,
            temporal_complexity: 0.4,
            mean_luma: 0.5,
            scene_cut_score: 0.0,
        };
        let expected = 0.8 * 0.6 + 0.4 * 0.4;
        assert!((f.combined_complexity() - expected).abs() < 1e-5);
    }

    #[test]
    fn test_is_priming_while_filling() {
        let cfg = LookaheadConfig {
            window_size: 8,
            ..LookaheadConfig::default()
        };
        let mut buf = LookaheadBuffer::new(cfg).expect("ok");
        assert!(buf.is_priming(), "should be priming when empty");
        for i in 0..7 {
            buf.push(simple_frame(i, 0.5, 0.3, 0.0));
            assert!(buf.is_priming(), "still priming at {i}+1 frames");
        }
        buf.push(simple_frame(7, 0.5, 0.3, 0.0));
        assert!(!buf.is_priming(), "full window should not be priming");
    }
}
