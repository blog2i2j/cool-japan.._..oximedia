//! Automated highlight clip extraction from playout using scene analysis.
//!
//! This module extends the basic highlight automation with computational scene
//! analysis techniques to automatically detect and extract broadcast highlight
//! moments from a sequence of decoded video frames.
//!
//! # Analysis Methods
//!
//! | Detector | Signal used | Typical use |
//! |----------|-------------|-------------|
//! | [`MotionEnergyDetector`] | Mean absolute frame difference | Sports action, live events |
//! | [`LuminanceJumpDetector`] | Inter-frame mean luminance delta | Flash events, lighting changes |
//! | [`SceneCutDetector`] | Histogram correlation | Hard cuts, transitions |
//! | [`CompositeSceneAnalyser`] | Weighted combination | General-purpose highlight detection |
//!
//! # Pipeline
//!
//! 1. Each decoded frame is submitted to [`CompositeSceneAnalyser::submit`].
//! 2. The analyser internally runs all configured sub-detectors.
//! 3. When the composite score exceeds a configurable threshold, a
//!    [`SceneHighlight`] is emitted describing the highlight window.
//! 4. Callers collect highlights via [`CompositeSceneAnalyser::take_highlights`].
//!
//! # Design notes
//!
//! All detectors operate on [`GreyscaleFrame`] — an 8-bit per-pixel luma
//! representation.  Colour frames must be converted before submission.  The
//! conversion function [`rgb_to_luma`] is provided for convenience.

#![allow(dead_code)]

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// GreyscaleFrame
// ---------------------------------------------------------------------------

/// An 8-bit per-pixel luma (Y) frame used for scene analysis.
///
/// Pixels are stored in row-major order: pixel at `(x, y)` is at
/// `data[y * width + x]`.
#[derive(Debug, Clone)]
pub struct GreyscaleFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Luma values — must have `width * height` elements.
    pub data: Vec<u8>,
    /// Frame sequence number.
    pub sequence: u64,
    /// Presentation timestamp in nanoseconds.
    pub pts_ns: u64,
}

impl GreyscaleFrame {
    /// Construct a new greyscale frame, filling all pixels with `fill`.
    pub fn filled(width: u32, height: u32, sequence: u64, pts_ns: u64, fill: u8) -> Self {
        let size = (width as usize) * (height as usize);
        Self {
            width,
            height,
            data: vec![fill; size],
            sequence,
            pts_ns,
        }
    }

    /// Return the number of pixels.
    pub fn pixel_count(&self) -> usize {
        (self.width as usize) * (self.height as usize)
    }

    /// Compute the mean luma of the frame.
    pub fn mean_luma(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        let sum: u64 = self.data.iter().map(|&v| v as u64).sum();
        sum as f64 / self.data.len() as f64
    }

    /// Build an 8-bin luma histogram (each bin spans 32 luma values).
    pub fn histogram_8bin(&self) -> [u32; 8] {
        let mut hist = [0u32; 8];
        for &v in &self.data {
            let bin = (v / 32).min(7) as usize;
            hist[bin] += 1;
        }
        hist
    }
}

// ---------------------------------------------------------------------------
// Helper: RGB → luma
// ---------------------------------------------------------------------------

/// Convert a packed `[R, G, B]` or `[R, G, B, A]` byte slice to luma values
/// using the BT.709 coefficients:  Y = 0.2126·R + 0.7152·G + 0.0722·B.
///
/// `stride` is the number of bytes per pixel (3 for RGB, 4 for RGBA).
pub fn rgb_to_luma(data: &[u8], stride: usize) -> Vec<u8> {
    if stride < 3 {
        return Vec::new();
    }
    data.chunks(stride)
        .map(|px| {
            let r = px[0] as f64;
            let g = px[1] as f64;
            let b = px[2] as f64;
            (0.2126 * r + 0.7152 * g + 0.0722 * b).round() as u8
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Highlight
// ---------------------------------------------------------------------------

/// A detected scene highlight.
#[derive(Debug, Clone)]
pub struct SceneHighlight {
    /// Frame sequence at the start of the highlight window.
    pub start_sequence: u64,
    /// Frame sequence at the end of the highlight window (inclusive).
    pub end_sequence: u64,
    /// PTS of the first frame in nanoseconds.
    pub start_pts_ns: u64,
    /// PTS of the last frame in nanoseconds.
    pub end_pts_ns: u64,
    /// Composite detection score that triggered extraction (0.0–1.0+).
    pub score: f64,
    /// Human-readable reason for the detection.
    pub reason: HighlightReason,
}

impl SceneHighlight {
    /// Duration of the highlight in nanoseconds.
    pub fn duration_ns(&self) -> u64 {
        self.end_pts_ns.saturating_sub(self.start_pts_ns)
    }

    /// Duration of the highlight in seconds.
    pub fn duration_secs(&self) -> f64 {
        self.duration_ns() as f64 / 1_000_000_000.0
    }

    /// Number of frames in the highlight window.
    pub fn frame_count(&self) -> u64 {
        self.end_sequence.saturating_sub(self.start_sequence) + 1
    }
}

/// Which detector(s) fired to produce a highlight.
#[derive(Debug, Clone, PartialEq)]
pub enum HighlightReason {
    /// High inter-frame motion energy.
    HighMotion,
    /// Sudden luminance jump (e.g. flash, spotlight).
    LuminanceJump,
    /// Hard scene cut detected.
    SceneCut,
    /// Multiple detectors fired simultaneously.
    Composite,
}

// ---------------------------------------------------------------------------
// MotionEnergyDetector
// ---------------------------------------------------------------------------

/// Detects periods of high motion energy by computing the mean absolute
/// difference (MAD) between consecutive frames.
///
/// A score in `[0.0, 1.0]` is produced for each frame pair; values above
/// `threshold` indicate significant motion.
pub struct MotionEnergyDetector {
    /// MAD threshold above which motion is considered high (normalised 0–255).
    pub threshold: f64,
    /// Previous frame for difference computation.
    prev_frame: Option<GreyscaleFrame>,
}

impl MotionEnergyDetector {
    /// Create a new detector with the given threshold.
    pub fn new(threshold: f64) -> Self {
        Self {
            threshold,
            prev_frame: None,
        }
    }

    /// Submit a frame and return a motion score in `[0.0, 1.0]`, or `None` if
    /// this is the first frame (no previous to compare against).
    pub fn submit(&mut self, frame: &GreyscaleFrame) -> Option<f64> {
        let result = if let Some(prev) = &self.prev_frame {
            if prev.data.len() == frame.data.len() {
                let mad: f64 = prev
                    .data
                    .iter()
                    .zip(frame.data.iter())
                    .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs() as f64)
                    .sum::<f64>()
                    / frame.data.len() as f64;
                Some(mad / 255.0)
            } else {
                Some(0.0)
            }
        } else {
            None
        };
        self.prev_frame = Some(frame.clone());
        result
    }

    /// Return `true` if the last submitted score exceeded the threshold.
    pub fn is_high_motion(&self, score: f64) -> bool {
        score > self.threshold
    }
}

// ---------------------------------------------------------------------------
// LuminanceJumpDetector
// ---------------------------------------------------------------------------

/// Detects sudden changes in mean frame luminance — useful for flash detection
/// and dramatic lighting transitions.
pub struct LuminanceJumpDetector {
    /// Normalised luminance delta threshold (`[0.0, 1.0]`).
    pub threshold: f64,
    /// Ring of recent mean-luma values for smoothing.
    history: VecDeque<f64>,
    /// Number of frames to retain in the smoothing window.
    window: usize,
}

impl LuminanceJumpDetector {
    /// Create a new detector.
    ///
    /// `threshold` — normalised delta threshold (e.g. `0.15` = 15 luma units
    /// change).  `window` — smoothing window in frames (1 = no smoothing).
    pub fn new(threshold: f64, window: usize) -> Self {
        let window = window.max(1);
        Self {
            threshold,
            history: VecDeque::with_capacity(window + 1),
            window,
        }
    }

    /// Submit a frame and return the normalised luminance delta, or `None`
    /// for the very first frame.
    pub fn submit(&mut self, frame: &GreyscaleFrame) -> Option<f64> {
        let current = frame.mean_luma() / 255.0;
        if self.history.len() < self.window {
            self.history.push_back(current);
            return None;
        }
        let mean_prev: f64 = self.history.iter().sum::<f64>() / self.history.len() as f64;
        let delta = (current - mean_prev).abs();
        // Slide the window.
        if self.history.len() >= self.window {
            self.history.pop_front();
        }
        self.history.push_back(current);
        Some(delta)
    }

    /// Return `true` if the delta exceeds the configured threshold.
    pub fn is_jump(&self, delta: f64) -> bool {
        delta > self.threshold
    }
}

// ---------------------------------------------------------------------------
// SceneCutDetector
// ---------------------------------------------------------------------------

/// Detects hard cuts by comparing luma histograms of consecutive frames.
///
/// A Bhattacharyya-inspired correlation coefficient in `[0.0, 1.0]` is
/// produced; values below `cut_threshold` indicate a scene cut.
pub struct SceneCutDetector {
    /// Correlation threshold below which a cut is declared.
    pub cut_threshold: f64,
    prev_hist: Option<[u32; 8]>,
    prev_total: u32,
}

impl SceneCutDetector {
    pub fn new(cut_threshold: f64) -> Self {
        Self {
            cut_threshold,
            prev_hist: None,
            prev_total: 0,
        }
    }

    /// Submit a frame and return the histogram correlation `[0.0, 1.0]`,
    /// or `None` on the first frame.
    ///
    /// Low values indicate a scene cut (low similarity).
    pub fn submit(&mut self, frame: &GreyscaleFrame) -> Option<f64> {
        let hist = frame.histogram_8bin();
        let total = hist.iter().sum::<u32>().max(1);

        let corr = if let Some(prev) = &self.prev_hist {
            let prev_total = self.prev_total.max(1) as f64;
            let curr_total = total as f64;
            // Bhattacharyya-like coefficient: sum of geometric means of
            // normalised bin counts.
            let bc: f64 = prev
                .iter()
                .zip(hist.iter())
                .map(|(&p, &c)| {
                    let pn = p as f64 / prev_total;
                    let cn = c as f64 / curr_total;
                    (pn * cn).sqrt()
                })
                .sum();
            Some(bc.clamp(0.0, 1.0))
        } else {
            None
        };

        self.prev_hist = Some(hist);
        self.prev_total = total;
        corr
    }

    /// Return `true` if the given correlation indicates a scene cut.
    pub fn is_cut(&self, correlation: f64) -> bool {
        correlation < self.cut_threshold
    }
}

// ---------------------------------------------------------------------------
// CompositeSceneAnalyser
// ---------------------------------------------------------------------------

/// Configuration for the composite scene analyser.
#[derive(Debug, Clone)]
pub struct AnalyserConfig {
    /// Motion energy detector threshold (0.0–1.0).  Default: 0.10.
    pub motion_threshold: f64,
    /// Weight given to motion scores in the composite (default: 0.50).
    pub motion_weight: f64,
    /// Luminance jump threshold (0.0–1.0).  Default: 0.15.
    pub luma_threshold: f64,
    /// Weight given to luminance jump scores (default: 0.25).
    pub luma_weight: f64,
    /// Scene-cut correlation threshold.  Default: 0.70 (below = cut).
    pub cut_threshold: f64,
    /// Weight given to scene-cut score (default: 0.25).
    pub cut_weight: f64,
    /// Composite score threshold above which a highlight is declared.
    pub composite_threshold: f64,
    /// Minimum number of frames a highlight window must span.
    pub min_highlight_frames: u64,
    /// Maximum highlight window duration in frames before forced close.
    pub max_highlight_frames: u64,
    /// Luminance smoothing window size (frames).
    pub luma_window: usize,
}

impl Default for AnalyserConfig {
    fn default() -> Self {
        Self {
            motion_threshold: 0.10,
            motion_weight: 0.50,
            luma_threshold: 0.15,
            luma_weight: 0.25,
            cut_threshold: 0.70,
            cut_weight: 0.25,
            composite_threshold: 0.25,
            min_highlight_frames: 5,
            max_highlight_frames: 750, // 30 s at 25 fps
            luma_window: 5,
        }
    }
}

/// State of the current open highlight window.
#[derive(Debug, Clone)]
struct OpenHighlight {
    start_sequence: u64,
    start_pts_ns: u64,
    last_sequence: u64,
    last_pts_ns: u64,
    peak_score: f64,
    dominant_reason: HighlightReason,
}

/// Multi-detector scene analyser that combines motion, luminance, and scene-cut
/// signals to automatically extract broadcast highlights.
pub struct CompositeSceneAnalyser {
    config: AnalyserConfig,
    motion: MotionEnergyDetector,
    luma: LuminanceJumpDetector,
    cut: SceneCutDetector,
    /// Currently-open highlight window, if any.
    current: Option<OpenHighlight>,
    /// Closed highlights awaiting collection.
    highlights: Vec<SceneHighlight>,
    /// Total frames submitted.
    frames_analysed: u64,
}

impl CompositeSceneAnalyser {
    /// Create a new analyser with the given configuration.
    pub fn new(config: AnalyserConfig) -> Self {
        let motion = MotionEnergyDetector::new(config.motion_threshold);
        let luma = LuminanceJumpDetector::new(config.luma_threshold, config.luma_window);
        let cut = SceneCutDetector::new(config.cut_threshold);
        Self {
            config,
            motion,
            luma,
            cut,
            current: None,
            highlights: Vec::new(),
            frames_analysed: 0,
        }
    }

    /// Submit a frame for analysis.
    ///
    /// Internally updates all detectors and manages the open highlight window.
    pub fn submit(&mut self, frame: &GreyscaleFrame) {
        self.frames_analysed += 1;

        let motion_score = self.motion.submit(frame).unwrap_or(0.0);
        let luma_delta = self.luma.submit(frame).unwrap_or(0.0);
        let cut_corr = self.cut.submit(frame).unwrap_or(1.0);

        // Normalise cut signal: low correlation → high scene-change score.
        let cut_score = (1.0 - cut_corr).clamp(0.0, 1.0);

        let composite = motion_score * self.config.motion_weight
            + luma_delta * self.config.luma_weight
            + cut_score * self.config.cut_weight;

        // Determine dominant reason for transparency.
        let reason = if motion_score > luma_delta && motion_score > cut_score {
            HighlightReason::HighMotion
        } else if luma_delta >= cut_score {
            HighlightReason::LuminanceJump
        } else {
            HighlightReason::SceneCut
        };

        let above_threshold = composite >= self.config.composite_threshold;

        match &mut self.current {
            None if above_threshold => {
                // Start a new highlight window.
                self.current = Some(OpenHighlight {
                    start_sequence: frame.sequence,
                    start_pts_ns: frame.pts_ns,
                    last_sequence: frame.sequence,
                    last_pts_ns: frame.pts_ns,
                    peak_score: composite,
                    dominant_reason: reason,
                });
            }
            Some(open) if above_threshold => {
                // Extend the window.
                open.last_sequence = frame.sequence;
                open.last_pts_ns = frame.pts_ns;
                if composite > open.peak_score {
                    open.peak_score = composite;
                    open.dominant_reason = reason;
                }
                // Force-close if the window has grown too large.
                let window_len = open
                    .last_sequence
                    .saturating_sub(open.start_sequence)
                    + 1;
                if window_len >= self.config.max_highlight_frames {
                    self.close_highlight();
                }
            }
            Some(_) => {
                // Score fell below threshold — close the window.
                self.close_highlight();
            }
            None => {
                // Nothing happening.
            }
        }
    }

    /// Close the currently-open highlight window and add it to the output list
    /// if it meets the minimum duration requirement.
    fn close_highlight(&mut self) {
        if let Some(open) = self.current.take() {
            let frame_count = open
                .last_sequence
                .saturating_sub(open.start_sequence)
                + 1;
            if frame_count >= self.config.min_highlight_frames {
                let reason = if frame_count > 1 {
                    HighlightReason::Composite
                } else {
                    open.dominant_reason
                };
                self.highlights.push(SceneHighlight {
                    start_sequence: open.start_sequence,
                    end_sequence: open.last_sequence,
                    start_pts_ns: open.start_pts_ns,
                    end_pts_ns: open.last_pts_ns,
                    score: open.peak_score,
                    reason,
                });
            }
        }
    }

    /// Flush any open highlight window.
    ///
    /// Call at the end of a programme to ensure the last window is captured.
    pub fn flush(&mut self) {
        self.close_highlight();
    }

    /// Drain and return all completed highlights since the last call.
    pub fn take_highlights(&mut self) -> Vec<SceneHighlight> {
        std::mem::take(&mut self.highlights)
    }

    /// Total frames submitted to this analyser.
    pub fn frames_analysed(&self) -> u64 {
        self.frames_analysed
    }

    /// Return `true` if a highlight window is currently open.
    pub fn is_highlight_active(&self) -> bool {
        self.current.is_some()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_frame(seq: u64, luma: u8) -> GreyscaleFrame {
        GreyscaleFrame::filled(16, 16, seq, seq * 40_000_000, luma)
    }

    /// A frame where each pixel alternates between `base` and `base+noise`
    /// depending on pixel index XOR sequence — consecutive frames differ
    /// maximally, producing high MAD scores.
    fn noisy_frame(seq: u64, base: u8, noise: u8) -> GreyscaleFrame {
        let size = 16 * 16usize;
        // Use seq's LSB to flip the checker-board pattern each frame so
        // consecutive frames have high inter-frame difference.
        let flip = (seq & 1) as usize;
        let data: Vec<u8> = (0..size)
            .map(|i| {
                if (i ^ flip) % 2 == 0 {
                    base.saturating_add(noise)
                } else {
                    base
                }
            })
            .collect();
        GreyscaleFrame {
            width: 16,
            height: 16,
            data,
            sequence: seq,
            pts_ns: seq * 40_000_000,
        }
    }

    #[test]
    fn test_rgb_to_luma_pure_red() {
        // Pure red [255, 0, 0] → Y ≈ 54 (BT.709).
        let data = vec![255u8, 0, 0];
        let luma = rgb_to_luma(&data, 3);
        assert_eq!(luma.len(), 1);
        // 0.2126 * 255 ≈ 54.2 → rounds to 54.
        assert_eq!(luma[0], 54);
    }

    #[test]
    fn test_rgb_to_luma_white() {
        let data = vec![255u8, 255, 255];
        let luma = rgb_to_luma(&data, 3);
        assert_eq!(luma[0], 255);
    }

    #[test]
    fn test_motion_energy_detector_no_motion() {
        let mut det = MotionEnergyDetector::new(0.10);
        let f1 = flat_frame(0, 128);
        let f2 = flat_frame(1, 128);
        // First frame → no score.
        assert!(det.submit(&f1).is_none());
        // Identical second frame → score = 0.
        let score = det.submit(&f2).unwrap();
        assert!((score - 0.0).abs() < 1e-9);
        assert!(!det.is_high_motion(score));
    }

    #[test]
    fn test_motion_energy_detector_high_motion() {
        let mut det = MotionEnergyDetector::new(0.10);
        let f1 = flat_frame(0, 0);
        let f2 = flat_frame(1, 255); // Maximum change.
        det.submit(&f1);
        let score = det.submit(&f2).unwrap();
        assert!((score - 1.0).abs() < 1e-6, "expected ~1.0, got {score}");
        assert!(det.is_high_motion(score));
    }

    #[test]
    fn test_luma_jump_detector_smooth_sequence() {
        let mut det = LuminanceJumpDetector::new(0.15, 3);
        // Submit warm-up frames.
        for i in 0..5u64 {
            let f = flat_frame(i, 100);
            let _ = det.submit(&f);
        }
        // Another frame with same luma → delta near zero.
        let delta = det.submit(&flat_frame(5, 100)).unwrap();
        assert!(!det.is_jump(delta));
    }

    #[test]
    fn test_luma_jump_detector_large_jump() {
        let mut det = LuminanceJumpDetector::new(0.10, 1);
        // After warm-up with dark frame.
        det.submit(&flat_frame(0, 10));
        // Sudden bright frame.
        let delta = det.submit(&flat_frame(1, 240)).unwrap();
        assert!(det.is_jump(delta), "delta={delta} should exceed threshold");
    }

    #[test]
    fn test_scene_cut_detector_identical_frames() {
        let mut det = SceneCutDetector::new(0.70);
        let f = flat_frame(0, 128);
        assert!(det.submit(&f).is_none()); // first frame
        let corr = det.submit(&flat_frame(1, 128)).unwrap();
        // Identical histograms → correlation near 1.0.
        assert!(corr > 0.99, "expected high correlation, got {corr}");
        assert!(!det.is_cut(corr));
    }

    #[test]
    fn test_scene_cut_detector_different_histograms() {
        let mut det = SceneCutDetector::new(0.70);
        det.submit(&flat_frame(0, 0));
        let corr = det.submit(&flat_frame(1, 255)).unwrap();
        // Black vs white have non-overlapping histograms → low correlation.
        assert!(det.is_cut(corr), "expected scene cut, corr={corr}");
    }

    #[test]
    fn test_composite_analyser_quiet_scene_no_highlights() {
        let mut analyser = CompositeSceneAnalyser::new(AnalyserConfig::default());
        for seq in 0..30u64 {
            analyser.submit(&flat_frame(seq, 128));
        }
        analyser.flush();
        let highlights = analyser.take_highlights();
        assert!(
            highlights.is_empty(),
            "no highlights expected in static scene"
        );
    }

    #[test]
    fn test_composite_analyser_motion_burst_creates_highlight() {
        let config = AnalyserConfig {
            composite_threshold: 0.10,
            min_highlight_frames: 3,
            max_highlight_frames: 300,
            motion_threshold: 0.05,
            ..Default::default()
        };
        let mut analyser = CompositeSceneAnalyser::new(config);

        // 5 quiet frames.
        for i in 0..5u64 {
            analyser.submit(&flat_frame(i, 100));
        }
        // 10 frames of high motion (alternating luma).
        for i in 5..15u64 {
            let f = noisy_frame(i, 100, 180);
            analyser.submit(&f);
        }
        // 5 more quiet frames.
        for i in 15..20u64 {
            analyser.submit(&flat_frame(i, 100));
        }
        analyser.flush();
        let highlights = analyser.take_highlights();
        assert!(
            !highlights.is_empty(),
            "expected at least one highlight from motion burst"
        );
        let h = &highlights[0];
        assert!(h.score >= 0.10, "score should meet threshold: {}", h.score);
        assert!(h.frame_count() >= 3, "highlight too short: {}", h.frame_count());
    }

    #[test]
    fn test_composite_analyser_frames_analysed_counter() {
        let mut analyser = CompositeSceneAnalyser::new(AnalyserConfig::default());
        for seq in 0..12u64 {
            analyser.submit(&flat_frame(seq, 80));
        }
        assert_eq!(analyser.frames_analysed(), 12);
    }

    #[test]
    fn test_scene_highlight_duration_helpers() {
        let h = SceneHighlight {
            start_sequence: 0,
            end_sequence: 24,
            start_pts_ns: 0,
            end_pts_ns: 1_000_000_000,
            score: 0.5,
            reason: HighlightReason::HighMotion,
        };
        assert_eq!(h.duration_ns(), 1_000_000_000);
        assert!((h.duration_secs() - 1.0).abs() < 1e-9);
        assert_eq!(h.frame_count(), 25);
    }

    #[test]
    fn test_greyscale_frame_mean_luma() {
        let f = flat_frame(0, 100);
        assert!((f.mean_luma() - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_composite_analyser_max_window_forces_close() {
        let config = AnalyserConfig {
            composite_threshold: 0.01, // very low → almost every frame triggers
            min_highlight_frames: 1,
            max_highlight_frames: 5,   // force-close after 5 frames
            motion_threshold: 0.001,
            ..Default::default()
        };
        let mut analyser = CompositeSceneAnalyser::new(config);
        // Submit 20 frames of constant motion.
        for i in 0..20u64 {
            analyser.submit(&noisy_frame(i, 50, 200));
        }
        analyser.flush();
        let highlights = analyser.take_highlights();
        // With max_highlight_frames=5 we should have several closed windows.
        assert!(
            highlights.len() >= 2,
            "expected multiple closed windows, got {}",
            highlights.len()
        );
    }
}
