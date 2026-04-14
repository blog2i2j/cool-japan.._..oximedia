//! Action recognition using temporal feature analysis.
//!
//! This module provides a pure-Rust, CPU-based action recognition system that
//! analyses sequences of video frames to classify motion patterns without
//! requiring any external neural network runtime.
//!
//! # Algorithm overview
//!
//! 1. **Temporal HOG (T-HOG)** — Extract gradient orientation histograms from
//!    frame difference images to build a motion descriptor for each clip.
//! 2. **Temporal window** — Maintain a sliding window of `window_frames` frames.
//! 3. **Action classifier** — Match temporal descriptors against a set of named
//!    action templates using cosine similarity.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_cv::action_recognition::{ActionRecognizer, ActionConfig};
//!
//! let config = ActionConfig::default();
//! let mut recognizer = ActionRecognizer::new(config);
//!
//! // Feed 32×32 grayscale frames
//! let frame = vec![128u8; 32 * 32];
//! for _ in 0..16 {
//!     recognizer.push_frame(&frame, 32, 32);
//! }
//!
//! let prediction = recognizer.predict();
//! println!("Detected action: {:?}", prediction);
//! ```

use crate::error::{CvError, CvResult};

// ── Constants ─────────────────────────────────────────────────────────────────

/// Number of orientation bins for HOG features.
const HOG_BINS: usize = 9;

/// Cell size in pixels for HOG computation.
const HOG_CELL: usize = 8;

// ── Action label ─────────────────────────────────────────────────────────────

/// Recognised action categories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Action {
    /// No significant motion detected.
    Idle,
    /// Walking / slow locomotion.
    Walking,
    /// Running / fast locomotion.
    Running,
    /// Waving hand.
    Waving,
    /// Clapping.
    Clapping,
    /// Jumping.
    Jumping,
    /// Falling.
    Falling,
    /// Unknown / unclassified.
    Unknown,
}

impl Action {
    /// Return a human-readable name for this action.
    #[must_use]
    pub fn name(&self) -> &'static str {
        match self {
            Self::Idle => "idle",
            Self::Walking => "walking",
            Self::Running => "running",
            Self::Waving => "waving",
            Self::Clapping => "clapping",
            Self::Jumping => "jumping",
            Self::Falling => "falling",
            Self::Unknown => "unknown",
        }
    }
}

// ── Prediction result ─────────────────────────────────────────────────────────

/// Result of an action recognition prediction.
#[derive(Debug, Clone)]
pub struct ActionPrediction {
    /// The top-1 predicted action.
    pub action: Action,
    /// Confidence score in `[0.0, 1.0]`.
    pub confidence: f32,
    /// Top-k predictions with scores, sorted by score descending.
    pub top_k: Vec<(Action, f32)>,
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration for the action recogniser.
#[derive(Debug, Clone)]
pub struct ActionConfig {
    /// Number of frames to include in one temporal window.
    pub window_frames: usize,
    /// Step size between consecutive windows (1 = fully overlapping).
    pub window_step: usize,
    /// Minimum mean motion magnitude (0–255) to consider a clip non-idle.
    pub motion_threshold: f32,
    /// Top-k predictions to return.
    pub top_k: usize,
}

impl Default for ActionConfig {
    fn default() -> Self {
        Self {
            window_frames: 16,
            window_step: 4,
            motion_threshold: 5.0,
            top_k: 3,
        }
    }
}

// ── Temporal HOG descriptor ────────────────────────────────────────────────────

/// Compute a single-frame motion descriptor from a frame-difference image.
///
/// Returns a normalised L2 vector of length `(width/HOG_CELL) * (height/HOG_CELL) * HOG_BINS`.
#[allow(clippy::cast_precision_loss)]
fn frame_diff_hog(diff: &[f32], width: usize, height: usize) -> Vec<f32> {
    let cells_x = (width / HOG_CELL).max(1);
    let cells_y = (height / HOG_CELL).max(1);
    let num_cells = cells_x * cells_y;
    let mut descriptor = vec![0.0f32; num_cells * HOG_BINS];

    for cy in 0..cells_y {
        for cx in 0..cells_x {
            let x0 = cx * HOG_CELL;
            let y0 = cy * HOG_CELL;
            let cell_idx = cy * cells_x + cx;
            let bin_offset = cell_idx * HOG_BINS;

            for py in y0..(y0 + HOG_CELL).min(height) {
                for px in x0..(x0 + HOG_CELL).min(width) {
                    // Compute local gradient using finite differences
                    let get = |x: usize, y: usize| -> f32 {
                        if x < width && y < height {
                            diff[y * width + x]
                        } else {
                            0.0
                        }
                    };

                    let gx = get(px.saturating_add(1), py) - get(px.saturating_sub(1), py);
                    let gy = get(px, py.saturating_add(1)) - get(px, py.saturating_sub(1));
                    let mag = (gx * gx + gy * gy).sqrt();
                    let angle = gy.atan2(gx); // [-π, π]

                    // Map to [0, π) unsigned orientation
                    let angle_pos = if angle < 0.0 {
                        angle + std::f32::consts::PI
                    } else {
                        angle
                    };
                    let bin = ((angle_pos / std::f32::consts::PI) * HOG_BINS as f32) as usize;
                    let bin = bin.min(HOG_BINS - 1);
                    descriptor[bin_offset + bin] += mag;
                }
            }
        }
    }

    // L2 normalise
    let l2: f32 = descriptor.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if l2 > 1e-6 {
        for v in &mut descriptor {
            *v /= l2;
        }
    }

    descriptor
}

/// Compute temporal HOG descriptor for a clip (sequence of frames).
///
/// Each frame is converted to a difference image against the previous frame,
/// then per-frame HOG descriptors are averaged.
#[allow(clippy::cast_precision_loss)]
fn temporal_hog(frames: &[(Vec<f32>, usize, usize)]) -> Vec<f32> {
    if frames.len() < 2 {
        return Vec::new();
    }

    let (_, w, h) = &frames[0];
    let width = *w;
    let height = *h;
    let descriptor_len = (width / HOG_CELL).max(1) * (height / HOG_CELL).max(1) * HOG_BINS;
    let mut accumulator = vec![0.0f32; descriptor_len];
    let mut count = 0usize;

    for i in 1..frames.len() {
        let (prev, pw, ph) = &frames[i - 1];
        let (curr, cw, ch) = &frames[i];

        if pw != cw || ph != ch || prev.len() != curr.len() {
            continue;
        }

        // Frame difference
        let diff: Vec<f32> = prev
            .iter()
            .zip(curr.iter())
            .map(|(&p, &c)| (c - p).abs())
            .collect();

        let hog = frame_diff_hog(&diff, width, height);
        if hog.len() == descriptor_len {
            for (acc, h) in accumulator.iter_mut().zip(hog.iter()) {
                *acc += h;
            }
            count += 1;
        }
    }

    if count > 0 {
        for v in &mut accumulator {
            *v /= count as f32;
        }
    }

    // Re-normalise
    let l2: f32 = accumulator.iter().map(|&v| v * v).sum::<f32>().sqrt();
    if l2 > 1e-6 {
        for v in &mut accumulator {
            *v /= l2;
        }
    }

    accumulator
}

// ── Action templates (hand-crafted motion prototypes) ─────────────────────────

/// A named action template represented as a pre-computed HOG prototype vector.
struct ActionTemplate {
    action: Action,
    /// Prototype descriptor (L2-normalised).
    descriptor: Vec<f32>,
}

/// Build built-in action templates for a descriptor of `len` bins.
///
/// Templates are hand-crafted statistical prototypes rather than
/// data-learned centroids; they represent the rough spectral shape
/// that each action class tends to produce in HOG space.
#[allow(clippy::cast_precision_loss)]
fn build_templates(len: usize) -> Vec<ActionTemplate> {
    if len == 0 {
        return Vec::new();
    }

    let uniform = |weight: f32| -> Vec<f32> {
        let v = weight / len as f32;
        vec![v; len]
    };

    let ramp_up =
        |scale: f32| -> Vec<f32> { (0..len).map(|i| scale * (i as f32 / len as f32)).collect() };

    let ramp_down = |scale: f32| -> Vec<f32> {
        (0..len)
            .map(|i| scale * (1.0 - i as f32 / len as f32))
            .collect()
    };

    let mid_peak = |scale: f32| -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / len as f32;
                scale * (-((t - 0.5) * 4.0).powi(2)).exp()
            })
            .collect()
    };

    let l2_norm = |mut v: Vec<f32>| -> Vec<f32> {
        let l2: f32 = v.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if l2 > 1e-6 {
            for x in &mut v {
                *x /= l2;
            }
        }
        v
    };

    vec![
        // Idle: very small uniform motion
        ActionTemplate {
            action: Action::Idle,
            descriptor: l2_norm(uniform(0.01)),
        },
        // Walking: moderate low-frequency motion (ramp up)
        ActionTemplate {
            action: Action::Walking,
            descriptor: l2_norm(ramp_up(0.3)),
        },
        // Running: high-energy motion across all bins (ramp down = fast)
        ActionTemplate {
            action: Action::Running,
            descriptor: l2_norm(ramp_down(0.8)),
        },
        // Waving: mid-frequency oscillatory pattern (mid peak)
        ActionTemplate {
            action: Action::Waving,
            descriptor: l2_norm(mid_peak(0.6)),
        },
        // Clapping: high-frequency bursts (alternating)
        ActionTemplate {
            action: Action::Clapping,
            descriptor: l2_norm(
                (0..len)
                    .map(|i| if i % 3 == 0 { 0.7 } else { 0.1 })
                    .collect(),
            ),
        },
        // Jumping: vertical motion burst (spike at start, ramp down)
        ActionTemplate {
            action: Action::Jumping,
            descriptor: l2_norm(
                (0..len)
                    .map(|i| {
                        let t = i as f32 / len as f32;
                        (-t * 3.0).exp()
                    })
                    .collect(),
            ),
        },
        // Falling: rapid unidirectional motion (ramp down, high initial energy)
        ActionTemplate {
            action: Action::Falling,
            descriptor: l2_norm(
                (0..len)
                    .map(|i| {
                        let t = i as f32 / len as f32;
                        (-t * 1.5).exp() * 0.9
                    })
                    .collect(),
            ),
        },
    ]
}

/// Cosine similarity between two equal-length vectors.
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();
    if na < 1e-6 || nb < 1e-6 {
        0.0
    } else {
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }
}

// ── ActionRecognizer ──────────────────────────────────────────────────────────

/// CPU-based action recogniser using temporal HOG descriptors.
pub struct ActionRecognizer {
    config: ActionConfig,
    /// Sliding window of (grayscale f32 pixels, width, height).
    frame_buffer: std::collections::VecDeque<(Vec<f32>, usize, usize)>,
    /// Number of frames pushed since last prediction reset.
    frames_pushed: usize,
}

impl ActionRecognizer {
    /// Create a new action recogniser with the given configuration.
    #[must_use]
    pub fn new(config: ActionConfig) -> Self {
        let capacity = config.window_frames + 1;
        Self {
            config,
            frame_buffer: std::collections::VecDeque::with_capacity(capacity),
            frames_pushed: 0,
        }
    }

    /// Push a new grayscale frame into the temporal window.
    ///
    /// Pixels are expected as 8-bit grayscale values in row-major order.
    /// The frame is converted to normalised `f32` internally.
    pub fn push_frame(&mut self, pixels: &[u8], width: usize, height: usize) {
        if pixels.len() != width * height {
            return;
        }

        let f32_pixels: Vec<f32> = pixels.iter().map(|&p| f32::from(p) / 255.0).collect();

        if self.frame_buffer.len() >= self.config.window_frames {
            self.frame_buffer.pop_front();
        }
        self.frame_buffer.push_back((f32_pixels, width, height));
        self.frames_pushed += 1;
    }

    /// Push a pre-normalised f32 grayscale frame.
    pub fn push_frame_f32(&mut self, pixels: &[f32], width: usize, height: usize) {
        if pixels.len() != width * height {
            return;
        }

        if self.frame_buffer.len() >= self.config.window_frames {
            self.frame_buffer.pop_front();
        }
        self.frame_buffer
            .push_back((pixels.to_vec(), width, height));
        self.frames_pushed += 1;
    }

    /// Compute the mean motion magnitude in the current window.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn mean_motion(&self) -> f32 {
        if self.frame_buffer.len() < 2 {
            return 0.0;
        }

        let mut total = 0.0f32;
        let mut count = 0usize;

        let frames: Vec<_> = self.frame_buffer.iter().collect();
        for i in 1..frames.len() {
            let (prev, _, _) = frames[i - 1];
            let (curr, _, _) = frames[i];
            if prev.len() == curr.len() {
                let diff_sum: f32 = prev
                    .iter()
                    .zip(curr.iter())
                    .map(|(&p, &c)| (c - p).abs())
                    .sum();
                total += diff_sum / prev.len().max(1) as f32;
                count += 1;
            }
        }

        if count == 0 {
            0.0
        } else {
            total / count as f32 * 255.0 // Scale to 0–255 range
        }
    }

    /// Check if there are enough frames to make a prediction.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.frame_buffer.len() >= self.config.window_frames.min(2)
    }

    /// Compute the temporal HOG descriptor for the current window.
    ///
    /// Returns `None` if fewer than 2 frames are available.
    #[must_use]
    pub fn descriptor(&self) -> Option<Vec<f32>> {
        if self.frame_buffer.len() < 2 {
            return None;
        }

        let frames: Vec<(Vec<f32>, usize, usize)> = self
            .frame_buffer
            .iter()
            .map(|(p, w, h)| (p.clone(), *w, *h))
            .collect();

        let desc = temporal_hog(&frames);
        if desc.is_empty() {
            None
        } else {
            Some(desc)
        }
    }

    /// Make an action prediction for the current temporal window.
    ///
    /// Returns `None` if not enough frames have been pushed yet.
    #[allow(clippy::cast_precision_loss)]
    #[must_use]
    pub fn predict(&self) -> Option<ActionPrediction> {
        let desc = self.descriptor()?;

        // Check motion level
        let motion = self.mean_motion();
        if motion < self.config.motion_threshold {
            return Some(ActionPrediction {
                action: Action::Idle,
                confidence: 1.0 - motion / self.config.motion_threshold.max(1.0),
                top_k: vec![(Action::Idle, 1.0)],
            });
        }

        let templates = build_templates(desc.len());
        if templates.is_empty() {
            return Some(ActionPrediction {
                action: Action::Unknown,
                confidence: 0.0,
                top_k: vec![(Action::Unknown, 0.0)],
            });
        }

        // Compute cosine similarity with each template
        let mut scores: Vec<(Action, f32)> = templates
            .into_iter()
            .map(|t| {
                let sim = cosine_similarity(&desc, &t.descriptor);
                // Remap from [-1,1] to [0,1]
                let score = (sim + 1.0) / 2.0;
                (t.action, score)
            })
            .collect();

        // Sort by score descending
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let top_k_n = self.config.top_k.min(scores.len());
        let top_k: Vec<(Action, f32)> = scores[..top_k_n]
            .iter()
            .map(|(a, s)| (a.clone(), *s))
            .collect();

        let (best_action, best_score) = scores
            .into_iter()
            .next()
            .map_or_else(|| (Action::Unknown, 0.0), |(a, s)| (a, s));

        Some(ActionPrediction {
            action: best_action,
            confidence: best_score,
            top_k,
        })
    }

    /// Reset the frame buffer and counters.
    pub fn reset(&mut self) {
        self.frame_buffer.clear();
        self.frames_pushed = 0;
    }

    /// Return the number of frames currently in the buffer.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.frame_buffer.len()
    }

    /// Return the number of frames pushed since creation or last reset.
    #[must_use]
    pub fn frames_pushed(&self) -> usize {
        self.frames_pushed
    }
}

// ── Errors ────────────────────────────────────────────────────────────────────

/// Validate that a frame descriptor is consistent for recognition.
pub fn validate_frame(pixels: &[u8], width: usize, height: usize) -> CvResult<()> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_parameter("width/height", "must be > 0"));
    }
    if pixels.len() != width * height {
        return Err(CvError::invalid_parameter(
            "pixels",
            "length must equal width * height",
        ));
    }
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const W: usize = 32;
    const H: usize = 32;

    fn make_frame(val: u8) -> Vec<u8> {
        vec![val; W * H]
    }

    fn make_noisy_frame(base: u8, noise_step: u8) -> Vec<u8> {
        (0..W * H)
            .map(|i| base.wrapping_add((i as u8).wrapping_mul(noise_step)))
            .collect()
    }

    #[test]
    fn test_action_recognizer_new() {
        let config = ActionConfig::default();
        let recognizer = ActionRecognizer::new(config);
        assert_eq!(recognizer.buffer_size(), 0);
        assert_eq!(recognizer.frames_pushed(), 0);
    }

    #[test]
    fn test_push_frame_increments_counter() {
        let mut rec = ActionRecognizer::new(ActionConfig::default());
        rec.push_frame(&make_frame(100), W, H);
        assert_eq!(rec.frames_pushed(), 1);
        assert_eq!(rec.buffer_size(), 1);
    }

    #[test]
    fn test_push_frame_invalid_size() {
        let mut rec = ActionRecognizer::new(ActionConfig::default());
        // pixels.len() != W * H → should be ignored
        rec.push_frame(&[0u8; 10], W, H);
        assert_eq!(rec.buffer_size(), 0);
    }

    #[test]
    fn test_buffer_capped_at_window_frames() {
        let mut rec = ActionRecognizer::new(ActionConfig {
            window_frames: 4,
            ..Default::default()
        });
        for _ in 0..10 {
            rec.push_frame(&make_frame(128), W, H);
        }
        assert_eq!(rec.buffer_size(), 4);
    }

    #[test]
    fn test_not_ready_with_one_frame() {
        let mut rec = ActionRecognizer::new(ActionConfig::default());
        rec.push_frame(&make_frame(100), W, H);
        // Need at least 2 frames for temporal diff
        let config_min = ActionConfig {
            window_frames: 16,
            ..Default::default()
        };
        let rec2 = ActionRecognizer::new(config_min);
        assert!(!rec2.is_ready());
    }

    #[test]
    fn test_is_ready_with_two_frames() {
        let mut rec = ActionRecognizer::new(ActionConfig::default());
        rec.push_frame(&make_frame(100), W, H);
        rec.push_frame(&make_frame(110), W, H);
        assert!(rec.is_ready());
    }

    #[test]
    fn test_idle_prediction_for_static_frames() {
        let mut rec = ActionRecognizer::new(ActionConfig::default());
        // Push identical frames → zero motion → should classify as Idle
        for _ in 0..16 {
            rec.push_frame(&make_frame(128), W, H);
        }
        let pred = rec.predict();
        assert!(pred.is_some());
        let pred = pred.expect("prediction should be Some");
        assert_eq!(pred.action, Action::Idle);
    }

    #[test]
    fn test_predict_returns_some_for_moving_frames() {
        let mut rec = ActionRecognizer::new(ActionConfig {
            motion_threshold: 0.1,
            ..Default::default()
        });
        for i in 0..16u8 {
            rec.push_frame(&make_noisy_frame(i.wrapping_mul(10), 7), W, H);
        }
        let pred = rec.predict();
        assert!(pred.is_some());
        let pred = pred.expect("prediction should be Some");
        assert!(pred.confidence >= 0.0 && pred.confidence <= 1.0);
    }

    #[test]
    fn test_top_k_length() {
        let mut rec = ActionRecognizer::new(ActionConfig {
            top_k: 3,
            motion_threshold: 0.0,
            ..Default::default()
        });
        for i in 0..16u8 {
            rec.push_frame(&make_noisy_frame(i.wrapping_mul(15), 5), W, H);
        }
        if let Some(pred) = rec.predict() {
            assert!(pred.top_k.len() <= 3);
        }
    }

    #[test]
    fn test_descriptor_none_for_empty_buffer() {
        let rec = ActionRecognizer::new(ActionConfig::default());
        assert!(rec.descriptor().is_none());
    }

    #[test]
    fn test_descriptor_some_with_frames() {
        let mut rec = ActionRecognizer::new(ActionConfig::default());
        rec.push_frame(&make_frame(100), W, H);
        rec.push_frame(&make_frame(120), W, H);
        assert!(rec.descriptor().is_some());
    }

    #[test]
    fn test_mean_motion_zero_for_identical_frames() {
        let mut rec = ActionRecognizer::new(ActionConfig::default());
        rec.push_frame(&make_frame(128), W, H);
        rec.push_frame(&make_frame(128), W, H);
        let motion = rec.mean_motion();
        assert!(motion.abs() < 1e-3, "motion={motion}");
    }

    #[test]
    fn test_mean_motion_positive_for_different_frames() {
        let mut rec = ActionRecognizer::new(ActionConfig::default());
        rec.push_frame(&make_frame(0), W, H);
        rec.push_frame(&make_frame(200), W, H);
        let motion = rec.mean_motion();
        assert!(motion > 0.0, "expected positive motion, got {motion}");
    }

    #[test]
    fn test_reset_clears_buffer() {
        let mut rec = ActionRecognizer::new(ActionConfig::default());
        for _ in 0..8 {
            rec.push_frame(&make_frame(100), W, H);
        }
        rec.reset();
        assert_eq!(rec.buffer_size(), 0);
        assert_eq!(rec.frames_pushed(), 0);
    }

    #[test]
    fn test_validate_frame_ok() {
        let pixels = make_frame(100);
        assert!(validate_frame(&pixels, W, H).is_ok());
    }

    #[test]
    fn test_validate_frame_zero_dimension() {
        let pixels = vec![0u8; 0];
        assert!(validate_frame(&pixels, 0, H).is_err());
    }

    #[test]
    fn test_validate_frame_size_mismatch() {
        let pixels = vec![0u8; 10];
        assert!(validate_frame(&pixels, W, H).is_err());
    }

    #[test]
    fn test_action_name() {
        assert_eq!(Action::Idle.name(), "idle");
        assert_eq!(Action::Running.name(), "running");
        assert_eq!(Action::Unknown.name(), "unknown");
    }

    #[test]
    fn test_push_frame_f32() {
        let mut rec = ActionRecognizer::new(ActionConfig::default());
        let pixels = vec![0.5f32; W * H];
        rec.push_frame_f32(&pixels, W, H);
        assert_eq!(rec.buffer_size(), 1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Motion-feature-based action recognition API
// ═══════════════════════════════════════════════════════════════════════════════

// ── ActionLabel ───────────────────────────────────────────────────────────────

/// High-level action / camera-motion label derived from per-frame motion
/// features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ActionLabel {
    /// No significant motion (camera or scene is static).
    Static,
    /// Smooth, slow motion.
    SlowMotion,
    /// Rapid, energetic motion.
    FastMotion,
    /// Erratic / unstable camera movement (high spatial variance).
    Shake,
    /// Steady lateral camera pan.
    Pan,
    /// Zoom in or out (radially uniform optical flow).
    Zoom,
    /// Hard cut or very sudden scene change (extreme max_flow).
    Cut,
}

impl ActionLabel {
    /// Human-readable name for this label.
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::SlowMotion => "slow_motion",
            Self::FastMotion => "fast_motion",
            Self::Shake => "shake",
            Self::Pan => "pan",
            Self::Zoom => "zoom",
            Self::Cut => "cut",
        }
    }
}

impl std::fmt::Display for ActionLabel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

// ── MotionFeature ─────────────────────────────────────────────────────────────

/// Per-frame motion statistics, typically derived from optical flow or frame
/// differencing.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionFeature {
    /// Mean magnitude of the optical flow / frame difference over all pixels.
    pub mean_flow: f32,
    /// Maximum flow magnitude observed in the frame.
    pub max_flow: f32,
    /// Standard deviation of the per-pixel flow magnitudes.
    pub std_flow: f32,
    /// Fraction of pixels whose flow magnitude exceeds a significance threshold
    /// (a value in `[0.0, 1.0]`).
    pub motion_area_fraction: f32,
}

impl MotionFeature {
    /// Construct a new `MotionFeature`.
    #[must_use]
    pub fn new(mean_flow: f32, max_flow: f32, std_flow: f32, motion_area_fraction: f32) -> Self {
        Self {
            mean_flow,
            max_flow,
            std_flow,
            motion_area_fraction,
        }
    }
}

// ── Rule-based classification ─────────────────────────────────────────────────

/// Classify a single [`MotionFeature`] into an [`ActionLabel`] using a
/// deterministic rule-based decision tree.
///
/// The thresholds are designed for normalised flow values (pixels/frame) in
/// typical video content.  They may need tuning for specific resolutions or
/// frame rates.
///
/// Decision rules (applied in priority order):
///
/// 1. `max_flow > 80`  → [`ActionLabel::Cut`] (sudden hard cut)
/// 2. `mean_flow < 0.5` → [`ActionLabel::Static`]
/// 3. `std_flow / mean_flow > 2.0` → [`ActionLabel::Shake`]  (high relative std)
/// 4. `mean_flow < 3.0` → [`ActionLabel::SlowMotion`]
/// 5. `mean_flow > 20.0` → [`ActionLabel::FastMotion`]
/// 6. `std_flow / mean_flow < 0.4` → [`ActionLabel::Pan`] or [`ActionLabel::Zoom`]
///    - Pan vs Zoom: if `motion_area_fraction > 0.7` and low std → Zoom;
///      otherwise Pan.
/// 7. Default → [`ActionLabel::FastMotion`]
#[must_use]
pub fn classify_motion(feat: &MotionFeature) -> ActionLabel {
    // Rule 1: extreme max flow → hard cut
    if feat.max_flow > 80.0 {
        return ActionLabel::Cut;
    }

    // Rule 2: very low mean → static
    if feat.mean_flow < 0.5 {
        return ActionLabel::Static;
    }

    // Rule 3: high relative standard deviation → shake
    let rel_std = if feat.mean_flow > 1e-6 {
        feat.std_flow / feat.mean_flow
    } else {
        0.0
    };
    if rel_std > 2.0 {
        return ActionLabel::Shake;
    }

    // Rule 4: slow uniform motion
    if feat.mean_flow < 3.0 {
        return ActionLabel::SlowMotion;
    }

    // Rule 5: very fast motion
    if feat.mean_flow > 20.0 {
        return ActionLabel::FastMotion;
    }

    // Rule 6: steady flow (low relative std) → pan or zoom
    if rel_std < 0.4 {
        // Zoom tends to affect the whole frame (large area fraction) with
        // radially symmetric flow; pan has a large fraction too but the
        // threshold on area_fraction differentiates from localised motion.
        if feat.motion_area_fraction > 0.7 {
            return ActionLabel::Zoom;
        }
        return ActionLabel::Pan;
    }

    // Default
    ActionLabel::FastMotion
}

/// Classify a sequence of [`MotionFeature`]s by applying [`classify_motion`]
/// to each element.
///
/// Returns a `Vec` of `(frame_index, ActionLabel)` pairs.
#[must_use]
pub fn recognize(motion_features: &[MotionFeature]) -> Vec<(usize, ActionLabel)> {
    motion_features
        .iter()
        .enumerate()
        .map(|(i, feat)| (i, classify_motion(feat)))
        .collect()
}

// ── ActionSequence ────────────────────────────────────────────────────────────

/// A labelled sequence of frames with optional temporal smoothing.
#[derive(Debug, Clone)]
pub struct ActionSequence {
    /// `(frame_index, label)` pairs in ascending frame-index order.
    pub labels: Vec<(usize, ActionLabel)>,
}

impl ActionSequence {
    /// Construct an `ActionSequence` from a slice of motion features.
    ///
    /// This is equivalent to calling [`recognize`] and wrapping the result.
    #[must_use]
    pub fn from_features(features: &[MotionFeature]) -> Self {
        Self {
            labels: recognize(features),
        }
    }

    /// Construct an `ActionSequence` from a pre-computed label list.
    #[must_use]
    pub fn new(labels: Vec<(usize, ActionLabel)>) -> Self {
        Self { labels }
    }

    /// Temporally smooth the label sequence using a majority-vote window.
    ///
    /// Each label is replaced by the most frequent label in the surrounding
    /// `window` frames (half-window on each side).  Ties are broken by
    /// preferring the original label.
    ///
    /// A `window` of `0` or `1` leaves the sequence unchanged.
    #[must_use]
    pub fn smooth(&self, window: usize) -> Self {
        if window <= 1 || self.labels.is_empty() {
            return self.clone();
        }

        let half = window / 2;
        let n = self.labels.len();
        let smoothed: Vec<(usize, ActionLabel)> = (0..n)
            .map(|i| {
                let lo = i.saturating_sub(half);
                let hi = (i + half + 1).min(n);

                // Count votes
                use std::collections::HashMap;
                let mut votes: HashMap<ActionLabel, usize> = HashMap::new();
                for (_, lbl) in &self.labels[lo..hi] {
                    *votes.entry(*lbl).or_insert(0) += 1;
                }

                // Pick the majority label; prefer original on tie
                let original = self.labels[i].1;
                let winner = votes
                    .into_iter()
                    .max_by(|a, b| {
                        // Higher count wins; ties resolved by preferring original
                        a.1.cmp(&b.1).then_with(|| {
                            if a.0 == original {
                                std::cmp::Ordering::Greater
                            } else if b.0 == original {
                                std::cmp::Ordering::Less
                            } else {
                                std::cmp::Ordering::Equal
                            }
                        })
                    })
                    .map(|(lbl, _)| lbl)
                    .unwrap_or(original);

                (self.labels[i].0, winner)
            })
            .collect();

        Self { labels: smoothed }
    }

    /// Number of labelled frames.
    #[must_use]
    pub fn len(&self) -> usize {
        self.labels.len()
    }

    /// Returns `true` if the sequence contains no frames.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }

    /// Returns an iterator over `(frame_index, ActionLabel)` pairs.
    pub fn iter(&self) -> impl Iterator<Item = &(usize, ActionLabel)> {
        self.labels.iter()
    }
}

// ── Motion-feature tests ──────────────────────────────────────────────────────

#[cfg(test)]
mod motion_tests {
    use super::*;

    fn feat(mean: f32, max: f32, std: f32, area: f32) -> MotionFeature {
        MotionFeature::new(mean, max, std, area)
    }

    // ── ActionLabel classification ────────────────────────────────────────────

    #[test]
    fn test_classify_static() {
        let f = feat(0.1, 1.0, 0.05, 0.01);
        assert_eq!(classify_motion(&f), ActionLabel::Static);
    }

    #[test]
    fn test_classify_cut() {
        let f = feat(50.0, 100.0, 10.0, 0.9);
        assert_eq!(classify_motion(&f), ActionLabel::Cut);
    }

    #[test]
    fn test_classify_shake() {
        // rel_std = std / mean = 6 / 2 = 3 → shake
        let f = feat(2.0, 20.0, 6.0, 0.5);
        assert_eq!(classify_motion(&f), ActionLabel::Shake);
    }

    #[test]
    fn test_classify_slow_motion() {
        let f = feat(1.5, 5.0, 0.3, 0.4);
        assert_eq!(classify_motion(&f), ActionLabel::SlowMotion);
    }

    #[test]
    fn test_classify_fast_motion() {
        // mean=25 > 20, max not extreme, rel_std moderate
        let f = feat(25.0, 60.0, 8.0, 0.6);
        assert_eq!(classify_motion(&f), ActionLabel::FastMotion);
    }

    #[test]
    fn test_classify_pan() {
        // mean in (3, 20), low rel_std → pan or zoom
        // area_fraction <= 0.7 → pan
        let f = feat(8.0, 15.0, 1.5, 0.5); // rel_std = 1.5/8 ≈ 0.19 < 0.4
        assert_eq!(classify_motion(&f), ActionLabel::Pan);
    }

    #[test]
    fn test_classify_zoom() {
        // low rel_std + high area_fraction → zoom
        let f = feat(8.0, 15.0, 1.5, 0.85); // area_fraction > 0.7
        assert_eq!(classify_motion(&f), ActionLabel::Zoom);
    }

    // ── recognize() ──────────────────────────────────────────────────────────

    #[test]
    fn test_recognize_empty() {
        let result = recognize(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_recognize_preserves_indices() {
        let features = vec![feat(0.1, 0.5, 0.02, 0.01), feat(25.0, 60.0, 8.0, 0.6)];
        let result = recognize(&features);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0);
        assert_eq!(result[1].0, 1);
    }

    #[test]
    fn test_recognize_mixed_sequence() {
        let features = vec![
            feat(0.1, 0.5, 0.02, 0.01),   // Static
            feat(1.5, 5.0, 0.3, 0.4),     // SlowMotion
            feat(50.0, 100.0, 10.0, 0.9), // Cut
        ];
        let result = recognize(&features);
        assert_eq!(result[0].1, ActionLabel::Static);
        assert_eq!(result[1].1, ActionLabel::SlowMotion);
        assert_eq!(result[2].1, ActionLabel::Cut);
    }

    // ── ActionSequence ────────────────────────────────────────────────────────

    #[test]
    fn test_action_sequence_from_features() {
        let features = vec![feat(0.1, 0.5, 0.02, 0.01)];
        let seq = ActionSequence::from_features(&features);
        assert_eq!(seq.len(), 1);
        assert_eq!(seq.labels[0].1, ActionLabel::Static);
    }

    #[test]
    fn test_smooth_window_one_unchanged() {
        let labels = vec![
            (0, ActionLabel::Static),
            (1, ActionLabel::Cut),
            (2, ActionLabel::Shake),
        ];
        let seq = ActionSequence::new(labels.clone());
        let smoothed = seq.smooth(1);
        assert_eq!(smoothed.labels, labels);
    }

    #[test]
    fn test_smooth_majority_vote() {
        // Window=3: each frame sees itself + 1 neighbour on each side
        let labels = vec![
            (0, ActionLabel::Static),
            (1, ActionLabel::Static),
            (2, ActionLabel::Cut), // isolated → should be overridden
            (3, ActionLabel::Static),
            (4, ActionLabel::Static),
        ];
        let seq = ActionSequence::new(labels);
        let smoothed = seq.smooth(3);
        // Frame 2 is surrounded by Static labels → should become Static
        assert_eq!(smoothed.labels[2].1, ActionLabel::Static);
    }

    #[test]
    fn test_smooth_empty_sequence() {
        let seq = ActionSequence::new(vec![]);
        let smoothed = seq.smooth(5);
        assert!(smoothed.is_empty());
    }

    #[test]
    fn test_action_sequence_is_empty() {
        let seq = ActionSequence::new(vec![]);
        assert!(seq.is_empty());
        assert_eq!(seq.len(), 0);
    }

    #[test]
    fn test_action_label_display() {
        assert_eq!(ActionLabel::Static.to_string(), "static");
        assert_eq!(ActionLabel::Cut.to_string(), "cut");
        assert_eq!(ActionLabel::Zoom.to_string(), "zoom");
    }

    #[test]
    fn test_motion_feature_new() {
        let f = MotionFeature::new(1.0, 5.0, 0.5, 0.3);
        assert!((f.mean_flow - 1.0).abs() < 1e-6);
        assert!((f.max_flow - 5.0).abs() < 1e-6);
        assert!((f.std_flow - 0.5).abs() < 1e-6);
        assert!((f.motion_area_fraction - 0.3).abs() < 1e-6);
    }
}
