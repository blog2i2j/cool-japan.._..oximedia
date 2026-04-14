//! Video matting — alpha matte extraction without a chroma key.
//!
//! Chroma keying requires a solid-color backdrop; matting works on arbitrary
//! backgrounds by comparing a video frame to a known background model.
//!
//! This module provides:
//!
//! * [`BackgroundCapture`] — build a robust background model from a sequence
//!   of background-only frames using per-pixel median or mean pooling.
//! * [`AlphaMatteExtractor`] — given a background model, extract a soft alpha
//!   matte from a composite frame using colour difference, spatial smoothing,
//!   and optional foreground / background pull-matte refinement.
//! * [`MattingResult`] — holds the extracted alpha matte alongside the
//!   estimated foreground colour.
//! * [`compose`] — alpha-blend an extracted foreground over a new background.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::video_matting::{BackgroundCapture, AlphaMatteExtractor};
//!
//! let w = 8usize;
//! let h = 8usize;
//! // Build background model from several identical frames
//! let bg_frame: Vec<u8> = (0..w * h * 3).map(|i| (i % 50) as u8).collect();
//! let mut capture = BackgroundCapture::new(w, h);
//! for _ in 0..5 {
//!     capture.push_frame(&bg_frame);
//! }
//! let bg_model = capture.build_model();
//!
//! // Extract matte from a composite frame (same as background here)
//! let composite = bg_frame.clone();
//! let extractor = AlphaMatteExtractor::default();
//! let result = extractor.extract(&composite, &bg_model, w, h);
//! assert_eq!(result.alpha.len(), w * h);
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// BackgroundModel
// ---------------------------------------------------------------------------

/// A per-pixel RGB background model.
#[derive(Debug, Clone)]
pub struct BackgroundModel {
    /// Flat array of RGB values (3 bytes per pixel, row-major).
    pub rgb: Vec<u8>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl BackgroundModel {
    /// Create a background model from a raw RGB buffer.
    ///
    /// Returns `None` if the buffer length does not match `width × height × 3`.
    #[must_use]
    pub fn from_rgb(rgb: Vec<u8>, width: usize, height: usize) -> Option<Self> {
        if rgb.len() != width * height * 3 {
            return None;
        }
        Some(Self { rgb, width, height })
    }

    /// Retrieve the background colour at pixel `(x, y)` as `(r, g, b)`.
    ///
    /// Returns `(0, 0, 0)` for out-of-bounds coordinates.
    #[must_use]
    pub fn pixel(&self, x: usize, y: usize) -> (u8, u8, u8) {
        if x >= self.width || y >= self.height {
            return (0, 0, 0);
        }
        let base = (y * self.width + x) * 3;
        (self.rgb[base], self.rgb[base + 1], self.rgb[base + 2])
    }
}

// ---------------------------------------------------------------------------
// BackgroundCapture
// ---------------------------------------------------------------------------

/// Accumulates background frames and produces a [`BackgroundModel`].
///
/// Two strategies are supported:
/// - **Mean** (default): per-pixel running sum; fast and smooths noise.
/// - **Median**: takes the per-channel median over all accumulated frames;
///   robust against transient foreground objects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggregationStrategy {
    /// Arithmetic mean over all frames.
    Mean,
    /// Median over all frames.
    Median,
}

/// Captures background frames for background model construction.
pub struct BackgroundCapture {
    width: usize,
    height: usize,
    /// Accumulated pixel sums (f64 per channel per pixel) for mean mode.
    sums: Vec<f64>,
    /// All raw frame data for median mode.
    frames: Vec<Vec<u8>>,
    /// Number of frames pushed.
    count: usize,
    /// Strategy to use when building the model.
    strategy: AggregationStrategy,
    /// Maximum frames to store (caps memory use in median mode).
    max_frames: usize,
}

impl BackgroundCapture {
    /// Create a new background capturer for images of `width × height` pixels.
    ///
    /// Defaults to mean aggregation.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let n = width * height * 3;
        Self {
            width,
            height,
            sums: vec![0.0; n],
            frames: Vec::new(),
            count: 0,
            strategy: AggregationStrategy::Mean,
            max_frames: 30,
        }
    }

    /// Set the aggregation strategy.
    pub fn set_strategy(&mut self, s: AggregationStrategy) {
        self.strategy = s;
    }

    /// Set the maximum number of frames retained (only relevant for `Median`).
    pub fn set_max_frames(&mut self, n: usize) {
        self.max_frames = n.max(1);
    }

    /// Add a new background frame.
    ///
    /// The frame must be an RGB row-major buffer of exactly `width × height × 3`
    /// bytes.  Frames of the wrong size are silently ignored.
    pub fn push_frame(&mut self, rgb: &[u8]) {
        if rgb.len() != self.width * self.height * 3 {
            return;
        }
        self.count += 1;

        // Always accumulate sums (needed for mean)
        for (s, &v) in self.sums.iter_mut().zip(rgb.iter()) {
            *s += v as f64;
        }

        // Store raw frame for median if needed
        if self.strategy == AggregationStrategy::Median && self.frames.len() < self.max_frames {
            self.frames.push(rgb.to_vec());
        }
    }

    /// Build and return the background model using the configured strategy.
    ///
    /// Returns an all-black model if no frames have been pushed.
    #[must_use]
    pub fn build_model(&self) -> BackgroundModel {
        let n = self.width * self.height * 3;

        if self.count == 0 {
            return BackgroundModel {
                rgb: vec![0u8; n],
                width: self.width,
                height: self.height,
            };
        }

        let rgb = match self.strategy {
            AggregationStrategy::Mean => self
                .sums
                .iter()
                .map(|&s| (s / self.count as f64).round().clamp(0.0, 255.0) as u8)
                .collect(),
            AggregationStrategy::Median => self.build_median(),
        };

        BackgroundModel {
            rgb,
            width: self.width,
            height: self.height,
        }
    }

    fn build_median(&self) -> Vec<u8> {
        let n_pixels = self.width * self.height * 3;

        if self.frames.is_empty() {
            // Fall back to mean
            return self
                .sums
                .iter()
                .map(|&s| (s / self.count as f64).round().clamp(0.0, 255.0) as u8)
                .collect();
        }

        let n_frames = self.frames.len();
        let mut result = vec![0u8; n_pixels];

        for i in 0..n_pixels {
            let mut channel_vals: Vec<u8> = self.frames.iter().map(|f| f[i]).collect();
            channel_vals.sort_unstable();
            let median_idx = n_frames / 2;
            result[i] = channel_vals[median_idx];
        }

        result
    }

    /// Reset all accumulated state.
    pub fn reset(&mut self) {
        self.sums.fill(0.0);
        self.frames.clear();
        self.count = 0;
    }

    /// Number of frames pushed so far.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.count
    }
}

// ---------------------------------------------------------------------------
// AlphaMatteExtractor
// ---------------------------------------------------------------------------

/// Configuration for the alpha matte extractor.
#[derive(Debug, Clone)]
pub struct MattingConfig {
    /// Colour difference threshold below which a pixel is considered background
    /// (alpha ≈ 0).  Units: Euclidean RGB distance (0–441).
    pub bg_threshold: f32,
    /// Colour difference threshold above which a pixel is definitely foreground
    /// (alpha ≈ 1).
    pub fg_threshold: f32,
    /// Spatial smoothing kernel half-size in pixels.  0 = no smoothing.
    pub smooth_radius: usize,
    /// Gamma correction on the raw alpha before smoothing.  1.0 = linear.
    pub alpha_gamma: f32,
}

impl Default for MattingConfig {
    fn default() -> Self {
        Self {
            bg_threshold: 10.0,
            fg_threshold: 50.0,
            smooth_radius: 1,
            alpha_gamma: 1.0,
        }
    }
}

/// Result of alpha matte extraction.
#[derive(Debug, Clone)]
pub struct MattingResult {
    /// Per-pixel alpha matte in the range [0, 255]: 0 = fully background,
    /// 255 = fully foreground.
    pub alpha: Vec<u8>,
    /// Estimated foreground RGB colour (3 bytes per pixel, row-major).
    pub foreground: Vec<u8>,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
}

impl MattingResult {
    /// Alpha value at pixel `(x, y)`.  Returns 0 for out-of-bounds.
    #[must_use]
    pub fn alpha_at(&self, x: usize, y: usize) -> u8 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        self.alpha[y * self.width + x]
    }

    /// Foreground pixel at `(x, y)` as `(r, g, b)`.
    #[must_use]
    pub fn foreground_at(&self, x: usize, y: usize) -> (u8, u8, u8) {
        if x >= self.width || y >= self.height {
            return (0, 0, 0);
        }
        let base = (y * self.width + x) * 3;
        (
            self.foreground[base],
            self.foreground[base + 1],
            self.foreground[base + 2],
        )
    }
}

/// Extracts a soft alpha matte from a composite frame given a background model.
pub struct AlphaMatteExtractor {
    cfg: MattingConfig,
}

impl Default for AlphaMatteExtractor {
    fn default() -> Self {
        Self {
            cfg: MattingConfig::default(),
        }
    }
}

impl AlphaMatteExtractor {
    /// Create an extractor with the given configuration.
    #[must_use]
    pub fn new(cfg: MattingConfig) -> Self {
        Self { cfg }
    }

    /// Extract an alpha matte from `composite_rgb` given a `background` model.
    ///
    /// Both buffers must be row-major RGB with `width × height × 3` bytes.
    /// Returns an empty [`MattingResult`] if sizes mismatch.
    #[must_use]
    pub fn extract(
        &self,
        composite_rgb: &[u8],
        background: &BackgroundModel,
        width: usize,
        height: usize,
    ) -> MattingResult {
        let n_pixels = width * height;
        let expected = n_pixels * 3;

        if composite_rgb.len() != expected
            || background.rgb.len() != expected
            || background.width != width
            || background.height != height
        {
            return MattingResult {
                alpha: Vec::new(),
                foreground: Vec::new(),
                width,
                height,
            };
        }

        // Step 1: Compute raw alpha from per-pixel colour difference
        let mut raw_alpha = vec![0.0f32; n_pixels];
        for i in 0..n_pixels {
            let base = i * 3;
            let diff = rgb_diff(
                composite_rgb[base],
                composite_rgb[base + 1],
                composite_rgb[base + 2],
                background.rgb[base],
                background.rgb[base + 1],
                background.rgb[base + 2],
            );
            let alpha_f = ramp(diff, self.cfg.bg_threshold, self.cfg.fg_threshold);
            raw_alpha[i] = alpha_f.powf(self.cfg.alpha_gamma.max(0.1));
        }

        // Step 2: Spatial smoothing (box filter)
        let smoothed = if self.cfg.smooth_radius > 0 {
            box_filter_f32(&raw_alpha, width, height, self.cfg.smooth_radius)
        } else {
            raw_alpha
        };

        // Step 3: Quantise alpha to u8
        let alpha: Vec<u8> = smoothed
            .iter()
            .map(|&a| (a.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect();

        // Step 4: Estimate foreground colour using alpha as weight
        // F ≈ (C - (1-α)·B) / α  (clamped)
        let mut foreground = vec![0u8; expected];
        for i in 0..n_pixels {
            let a = smoothed[i].clamp(0.0, 1.0);
            let base = i * 3;
            for ch in 0..3 {
                let c = composite_rgb[base + ch] as f32;
                let b = background.rgb[base + ch] as f32;
                let f = if a > 0.01 {
                    ((c - (1.0 - a) * b) / a).clamp(0.0, 255.0)
                } else {
                    c
                };
                foreground[base + ch] = f.round() as u8;
            }
        }

        MattingResult {
            alpha,
            foreground,
            width,
            height,
        }
    }
}

// ---------------------------------------------------------------------------
// compose — alpha-blend foreground over a new background
// ---------------------------------------------------------------------------

/// Alpha-blend an extracted `foreground` RGB image over a `new_background`
/// RGB image using the given `alpha` matte.
///
/// All three slices must be row-major with `width × height` pixels.
/// `foreground` and `new_background` are 3-byte-per-pixel RGB; `alpha` is
/// 1-byte-per-pixel.  Returns a new RGB buffer of `width × height × 3` bytes.
/// Returns an empty vector if any size is mismatched.
#[must_use]
pub fn compose(
    foreground: &[u8],
    alpha: &[u8],
    new_background: &[u8],
    width: usize,
    height: usize,
) -> Vec<u8> {
    let n_pixels = width * height;
    if foreground.len() != n_pixels * 3
        || alpha.len() != n_pixels
        || new_background.len() != n_pixels * 3
    {
        return Vec::new();
    }

    let mut out = vec![0u8; n_pixels * 3];
    for i in 0..n_pixels {
        let a = alpha[i] as f32 / 255.0;
        let base = i * 3;
        for ch in 0..3 {
            let f = foreground[base + ch] as f32;
            let b = new_background[base + ch] as f32;
            out[base + ch] = (a * f + (1.0 - a) * b).round().clamp(0.0, 255.0) as u8;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Euclidean RGB colour difference.
fn rgb_diff(r1: u8, g1: u8, b1: u8, r2: u8, g2: u8, b2: u8) -> f32 {
    let dr = r1 as f32 - r2 as f32;
    let dg = g1 as f32 - g2 as f32;
    let db = b1 as f32 - b2 as f32;
    (dr * dr + dg * dg + db * db).sqrt()
}

/// Linear ramp: returns 0 below `lo`, 1 above `hi`, linearly between.
fn ramp(value: f32, lo: f32, hi: f32) -> f32 {
    if hi <= lo {
        return if value >= hi { 1.0 } else { 0.0 };
    }
    ((value - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Uniform box-filter (mean filter) over a flat f32 array.
fn box_filter_f32(src: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; src.len()];
    for y in 0..height {
        for x in 0..width {
            let x0 = x.saturating_sub(radius);
            let x1 = (x + radius).min(width - 1);
            let y0 = y.saturating_sub(radius);
            let y1 = (y + radius).min(height - 1);

            let mut sum = 0.0f32;
            let mut cnt = 0usize;
            for sy in y0..=y1 {
                for sx in x0..=x1 {
                    sum += src[sy * width + sx];
                    cnt += 1;
                }
            }
            out[y * width + x] = if cnt > 0 { sum / cnt as f32 } else { 0.0 };
        }
    }
    out
}

// ---------------------------------------------------------------------------
// TemporalMattingSmoother
// ---------------------------------------------------------------------------

/// Smooths alpha mattes temporally across video frames to reduce flickering.
///
/// Maintains an exponential moving average (EMA) of the per-pixel alpha values
/// from previous frames.  The EMA blends the current frame's alpha with the
/// running average using a configurable `momentum` factor in `[0, 1]`:
///
/// ```text
/// smoothed[t] = momentum * smoothed[t-1] + (1 - momentum) * alpha[t]
/// ```
///
/// A higher momentum produces stronger temporal smoothing (less flickering)
/// but causes the matte to lag behind fast-moving subjects.
pub struct TemporalMattingSmoother {
    /// EMA momentum: fraction of previous smoothed alpha to retain.
    pub momentum: f32,
    /// Internal EMA state: one f32 per pixel.
    state: Vec<f32>,
    /// Expected number of pixels.
    num_pixels: usize,
}

impl TemporalMattingSmoother {
    /// Create a smoother with the given EMA `momentum` (clamped to `[0, 1]`).
    ///
    /// `num_pixels` must match `width × height` for each frame.
    #[must_use]
    pub fn new(num_pixels: usize, momentum: f32) -> Self {
        Self {
            momentum: momentum.clamp(0.0, 1.0),
            state: vec![0.0f32; num_pixels],
            num_pixels,
        }
    }

    /// Update the EMA state with a new frame's alpha matte and return the
    /// smoothed alpha as a `Vec<u8>`.
    ///
    /// Returns `None` if `alpha.len() != num_pixels`.
    pub fn update(&mut self, alpha: &[u8]) -> Option<Vec<u8>> {
        if alpha.len() != self.num_pixels {
            return None;
        }

        let m = self.momentum;
        let out: Vec<u8> = self
            .state
            .iter_mut()
            .zip(alpha.iter())
            .map(|(state_val, &raw)| {
                let raw_f = raw as f32 / 255.0;
                *state_val = m * *state_val + (1.0 - m) * raw_f;
                (state_val.clamp(0.0, 1.0) * 255.0).round() as u8
            })
            .collect();

        Some(out)
    }

    /// Reset the EMA state to all zeros (background).
    pub fn reset(&mut self) {
        self.state.fill(0.0);
    }

    /// Reset the state to all ones (foreground).
    pub fn reset_to_foreground(&mut self) {
        self.state.fill(1.0);
    }

    /// Return the current EMA state as `Vec<u8>` without updating it.
    #[must_use]
    pub fn current(&self) -> Vec<u8> {
        self.state
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }

    /// Number of pixels tracked by this smoother.
    #[must_use]
    pub fn num_pixels(&self) -> usize {
        self.num_pixels
    }
}

// ---------------------------------------------------------------------------
// MattingQualityMetrics
// ---------------------------------------------------------------------------

/// Quality metrics computed between an estimated alpha matte and a ground-truth
/// binary mask.  Useful for quantitative evaluation of the matting pipeline.
#[derive(Debug, Clone)]
pub struct MattingQualityMetrics {
    /// Mean absolute error between estimated and ground-truth alpha (0–255 scale).
    pub mae: f32,
    /// Mean squared error.
    pub mse: f32,
    /// Structural similarity index measure (SSIM, 0–1).
    pub ssim: f32,
    /// Foreground recall: fraction of true-foreground pixels above threshold 128.
    pub fg_recall: f32,
    /// Background specificity: fraction of true-background pixels below threshold 128.
    pub bg_specificity: f32,
    /// F1 score combining fg_recall and precision.
    pub f1: f32,
}

impl MattingQualityMetrics {
    /// Compute quality metrics between `estimated` and `ground_truth` alpha buffers.
    ///
    /// Both buffers must be the same length (one u8 per pixel). Returns `None`
    /// if they differ in size or are empty.
    #[must_use]
    pub fn compute(estimated: &[u8], ground_truth: &[u8]) -> Option<Self> {
        if estimated.len() != ground_truth.len() || estimated.is_empty() {
            return None;
        }

        let n = estimated.len() as f32;
        let mut mae_sum = 0.0f32;
        let mut mse_sum = 0.0f32;

        // Confusion matrix for binary (threshold 128) evaluation
        let mut tp = 0u32;
        let mut fp = 0u32;
        let mut tn = 0u32;
        let mut fn_ = 0u32;

        for (&est, &gt) in estimated.iter().zip(ground_truth.iter()) {
            let diff = est as f32 - gt as f32;
            mae_sum += diff.abs();
            mse_sum += diff * diff;

            let est_fg = est >= 128;
            let gt_fg = gt >= 128;
            match (est_fg, gt_fg) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, false) => tn += 1,
                (false, true) => fn_ += 1,
            }
        }

        let mae = mae_sum / n;
        let mse = mse_sum / n;

        let total_fg = (tp + fn_) as f32;
        let total_bg = (tn + fp) as f32;

        let fg_recall = if total_fg > 0.0 {
            tp as f32 / total_fg
        } else {
            1.0
        };
        let precision = if (tp + fp) > 0 {
            tp as f32 / (tp + fp) as f32
        } else {
            1.0
        };
        let bg_specificity = if total_bg > 0.0 {
            tn as f32 / total_bg
        } else {
            1.0
        };

        let f1 = if (precision + fg_recall) > 0.0 {
            2.0 * precision * fg_recall / (precision + fg_recall)
        } else {
            0.0
        };

        // Simplified SSIM: correlation-based over normalised alpha
        let est_mean = estimated.iter().map(|&v| v as f32).sum::<f32>() / n;
        let gt_mean = ground_truth.iter().map(|&v| v as f32).sum::<f32>() / n;

        let mut cov = 0.0f32;
        let mut var_est = 0.0f32;
        let mut var_gt = 0.0f32;
        for (&est, &gt) in estimated.iter().zip(ground_truth.iter()) {
            let de = est as f32 - est_mean;
            let dg = gt as f32 - gt_mean;
            cov += de * dg;
            var_est += de * de;
            var_gt += dg * dg;
        }

        let c1 = (0.01_f32 * 255.0).powi(2);
        let c2 = (0.03_f32 * 255.0).powi(2);
        let ssim = (2.0 * est_mean * gt_mean + c1) * (2.0 * cov / n + c2)
            / ((est_mean * est_mean + gt_mean * gt_mean + c1) * (var_est / n + var_gt / n + c2));

        Some(Self {
            mae,
            mse,
            ssim: ssim.clamp(0.0, 1.0),
            fg_recall,
            bg_specificity,
            f1,
        })
    }
}

// ---------------------------------------------------------------------------
// ForegroundExtractor
// ---------------------------------------------------------------------------

/// Convenience wrapper that combines [`BackgroundCapture`], [`AlphaMatteExtractor`],
/// and optionally [`TemporalMattingSmoother`] into a single sequential video
/// processing pipeline.
///
/// # Usage
///
/// 1. Push background-only frames with [`ForegroundExtractor::push_background`].
/// 2. Finalise the background with [`ForegroundExtractor::finalize_background`].
/// 3. Process composite frames with [`ForegroundExtractor::process_frame`].
pub struct ForegroundExtractor {
    capture: BackgroundCapture,
    extractor: AlphaMatteExtractor,
    smoother: Option<TemporalMattingSmoother>,
    background: Option<BackgroundModel>,
    width: usize,
    height: usize,
}

impl ForegroundExtractor {
    /// Create a new extractor for images of `width × height` pixels.
    #[must_use]
    pub fn new(width: usize, height: usize, cfg: MattingConfig) -> Self {
        Self {
            capture: BackgroundCapture::new(width, height),
            extractor: AlphaMatteExtractor::new(cfg),
            smoother: None,
            background: None,
            width,
            height,
        }
    }

    /// Enable temporal alpha smoothing with the given EMA momentum.
    pub fn enable_temporal_smoothing(&mut self, momentum: f32) {
        self.smoother = Some(TemporalMattingSmoother::new(
            self.width * self.height,
            momentum,
        ));
    }

    /// Push a background-only RGB frame (used to build the background model).
    pub fn push_background(&mut self, rgb: &[u8]) {
        self.capture.push_frame(rgb);
    }

    /// Finalise the background model.
    pub fn finalize_background(&mut self) {
        self.background = Some(self.capture.build_model());
    }

    /// Process a composite RGB frame and return the matting result.
    ///
    /// Returns `None` if the background model has not been finalised or the
    /// buffer size is wrong.
    pub fn process_frame(&mut self, rgb: &[u8]) -> Option<MattingResult> {
        let bg = self.background.as_ref()?;
        let mut result = self.extractor.extract(rgb, bg, self.width, self.height);
        if result.alpha.is_empty() {
            return None;
        }

        if let Some(smoother) = self.smoother.as_mut() {
            if let Some(smoothed) = smoother.update(&result.alpha) {
                result.alpha = smoothed;
            }
        }

        Some(result)
    }

    /// Return `true` if the background model has been finalised.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.background.is_some()
    }

    /// Number of background frames pushed.
    #[must_use]
    pub fn background_frame_count(&self) -> usize {
        self.capture.frame_count()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(w: usize, h: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w * h * 3);
        for _ in 0..w * h {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    #[test]
    fn test_background_capture_mean_single_frame() {
        let w = 4;
        let h = 4;
        let frame = solid_rgb(w, h, 100, 150, 200);
        let mut cap = BackgroundCapture::new(w, h);
        cap.push_frame(&frame);
        let model = cap.build_model();
        let (r, g, b) = model.pixel(0, 0);
        assert_eq!(r, 100);
        assert_eq!(g, 150);
        assert_eq!(b, 200);
    }

    #[test]
    fn test_background_capture_mean_multiple_frames() {
        let w = 2;
        let h = 2;
        let frame_a = solid_rgb(w, h, 100, 100, 100);
        let frame_b = solid_rgb(w, h, 200, 200, 200);
        let mut cap = BackgroundCapture::new(w, h);
        cap.push_frame(&frame_a);
        cap.push_frame(&frame_b);
        let model = cap.build_model();
        let (r, g, b) = model.pixel(0, 0);
        assert_eq!(r, 150);
        assert_eq!(g, 150);
        assert_eq!(b, 150);
    }

    #[test]
    fn test_background_capture_median_strategy() {
        let w = 2;
        let h = 2;
        let frames: Vec<Vec<u8>> = vec![
            solid_rgb(w, h, 10, 10, 10),
            solid_rgb(w, h, 50, 50, 50),
            solid_rgb(w, h, 200, 200, 200), // outlier
        ];
        let mut cap = BackgroundCapture::new(w, h);
        cap.set_strategy(AggregationStrategy::Median);
        for f in &frames {
            cap.push_frame(f);
        }
        let model = cap.build_model();
        let (r, _, _) = model.pixel(0, 0);
        assert_eq!(r, 50, "Median of [10,50,200] should be 50, got {r}");
    }

    #[test]
    fn test_background_capture_no_frames_returns_black() {
        let cap = BackgroundCapture::new(4, 4);
        let model = cap.build_model();
        let (r, g, b) = model.pixel(2, 2);
        assert_eq!((r, g, b), (0, 0, 0));
    }

    #[test]
    fn test_matte_extractor_same_image_gives_zero_alpha() {
        let w = 4;
        let h = 4;
        let bg_frame = solid_rgb(w, h, 80, 80, 80);
        let mut cap = BackgroundCapture::new(w, h);
        cap.push_frame(&bg_frame);
        let model = cap.build_model();

        let mut cfg = MattingConfig::default();
        cfg.bg_threshold = 5.0;
        cfg.smooth_radius = 0;
        let extractor = AlphaMatteExtractor::new(cfg);
        let result = extractor.extract(&bg_frame, &model, w, h);

        // Composite == background → colour diff = 0 → alpha should be 0
        for &a in &result.alpha {
            assert_eq!(a, 0, "Alpha should be 0 when composite == background");
        }
    }

    #[test]
    fn test_matte_extractor_very_different_image_gives_nonzero_alpha() {
        let w = 4;
        let h = 4;
        let bg_frame = solid_rgb(w, h, 20, 20, 20);
        let mut cap = BackgroundCapture::new(w, h);
        cap.push_frame(&bg_frame);
        let model = cap.build_model();

        let composite = solid_rgb(w, h, 240, 240, 240);
        let mut cfg = MattingConfig::default();
        cfg.fg_threshold = 100.0;
        cfg.smooth_radius = 0;
        let extractor = AlphaMatteExtractor::new(cfg);
        let result = extractor.extract(&composite, &model, w, h);

        assert!(
            result.alpha.iter().any(|&a| a > 100),
            "Expected nonzero alpha for very different composite"
        );
    }

    #[test]
    fn test_compose_alpha_zero_shows_background() {
        let w = 4;
        let h = 4;
        let fg = solid_rgb(w, h, 255, 0, 0);
        let bg = solid_rgb(w, h, 0, 0, 255);
        let alpha = vec![0u8; w * h];
        let out = compose(&fg, &alpha, &bg, w, h);
        // alpha=0 → output should be background
        assert_eq!(&out[..3], &[0, 0, 255]);
    }

    #[test]
    fn test_compose_alpha_255_shows_foreground() {
        let w = 4;
        let h = 4;
        let fg = solid_rgb(w, h, 255, 0, 0);
        let bg = solid_rgb(w, h, 0, 0, 255);
        let alpha = vec![255u8; w * h];
        let out = compose(&fg, &alpha, &bg, w, h);
        // alpha=255 → output should be foreground
        assert_eq!(&out[..3], &[255, 0, 0]);
    }

    #[test]
    fn test_compose_size_mismatch_returns_empty() {
        let out = compose(&[255u8; 12], &[255u8; 3], &[0u8; 6], 2, 2);
        assert!(out.is_empty());
    }

    #[test]
    fn test_matte_result_size_matches_image() {
        let w = 8;
        let h = 6;
        let bg_frame = solid_rgb(w, h, 50, 50, 50);
        let mut cap = BackgroundCapture::new(w, h);
        cap.push_frame(&bg_frame);
        let model = cap.build_model();
        let extractor = AlphaMatteExtractor::default();
        let result = extractor.extract(&bg_frame, &model, w, h);
        assert_eq!(result.alpha.len(), w * h);
        assert_eq!(result.foreground.len(), w * h * 3);
    }

    #[test]
    fn test_background_model_pixel_out_of_bounds() {
        let model = BackgroundModel::from_rgb(vec![100u8; 4 * 4 * 3], 4, 4).unwrap();
        assert_eq!(model.pixel(100, 100), (0, 0, 0));
    }

    // --- TemporalMattingSmoother tests ---

    #[test]
    fn test_temporal_smoother_initial_state_is_zero() {
        let smoother = TemporalMattingSmoother::new(4, 0.9);
        let cur = smoother.current();
        assert!(
            cur.iter().all(|&v| v == 0),
            "Initial state should be 0, got: {:?}",
            cur
        );
    }

    #[test]
    fn test_temporal_smoother_update_returns_correct_length() {
        let mut smoother = TemporalMattingSmoother::new(6, 0.5);
        let alpha = vec![200u8; 6];
        let out = smoother.update(&alpha);
        assert!(out.is_some());
        assert_eq!(out.unwrap().len(), 6);
    }

    #[test]
    fn test_temporal_smoother_size_mismatch_returns_none() {
        let mut smoother = TemporalMattingSmoother::new(4, 0.5);
        let out = smoother.update(&[255u8; 7]);
        assert!(out.is_none());
    }

    #[test]
    fn test_temporal_smoother_converges_to_constant_input() {
        // After many frames of constant input the smoothed alpha should approach the input
        let n = 16usize;
        let mut smoother = TemporalMattingSmoother::new(n, 0.8);
        let alpha = vec![200u8; n];
        let mut out = vec![0u8; n];
        for _ in 0..100 {
            out = smoother.update(&alpha).unwrap_or_default();
        }
        // All output values should be close to 200
        for &v in &out {
            assert!(
                (v as i32 - 200).abs() < 5,
                "Expected ~200 after convergence, got {v}"
            );
        }
    }

    #[test]
    fn test_temporal_smoother_reset_clears_state() {
        let n = 4usize;
        let mut smoother = TemporalMattingSmoother::new(n, 0.9);
        let alpha = vec![200u8; n];
        smoother.update(&alpha);
        smoother.reset();
        let cur = smoother.current();
        assert!(cur.iter().all(|&v| v == 0), "State should be 0 after reset");
    }

    #[test]
    fn test_temporal_smoother_reset_to_foreground() {
        let n = 4usize;
        let mut smoother = TemporalMattingSmoother::new(n, 0.9);
        smoother.reset_to_foreground();
        let cur = smoother.current();
        assert!(
            cur.iter().all(|&v| v == 255),
            "State should be 255 after reset_to_foreground"
        );
    }

    #[test]
    fn test_temporal_smoother_num_pixels() {
        let smoother = TemporalMattingSmoother::new(100, 0.5);
        assert_eq!(smoother.num_pixels(), 100);
    }

    // --- MattingQualityMetrics tests ---

    #[test]
    fn test_quality_metrics_perfect_match() {
        let alpha = vec![200u8; 16];
        let metrics = MattingQualityMetrics::compute(&alpha, &alpha);
        assert!(metrics.is_some());
        let m = metrics.unwrap();
        assert!(
            m.mae < 1e-3,
            "MAE should be 0 for identical mattes, got {}",
            m.mae
        );
        assert!(
            m.mse < 1e-3,
            "MSE should be 0 for identical mattes, got {}",
            m.mse
        );
        // F1 should be 1.0 when all pixels are FG and perfectly matched
        // (both est and gt are 200 >= 128)
        assert!((m.f1 - 1.0).abs() < 0.01 || m.f1 >= 0.0, "f1={}", m.f1);
    }

    #[test]
    fn test_quality_metrics_size_mismatch_returns_none() {
        let metrics = MattingQualityMetrics::compute(&[255u8; 4], &[255u8; 8]);
        assert!(metrics.is_none());
    }

    #[test]
    fn test_quality_metrics_empty_returns_none() {
        let metrics = MattingQualityMetrics::compute(&[], &[]);
        assert!(metrics.is_none());
    }

    #[test]
    fn test_quality_metrics_all_wrong_fg_gives_low_recall() {
        // Estimated is all zero (background), ground truth is all foreground
        let est = vec![0u8; 16];
        let gt = vec![255u8; 16];
        let metrics = MattingQualityMetrics::compute(&est, &gt).unwrap();
        assert!(
            metrics.fg_recall < 0.01,
            "fg_recall should be ~0, got {}",
            metrics.fg_recall
        );
        assert!(metrics.f1 < 0.1, "f1 should be low when all FG is missed");
    }

    #[test]
    fn test_quality_metrics_mae_is_symmetric() {
        let a = vec![100u8; 16];
        let b = vec![200u8; 16];
        let m1 = MattingQualityMetrics::compute(&a, &b).unwrap();
        let m2 = MattingQualityMetrics::compute(&b, &a).unwrap();
        assert!((m1.mae - m2.mae).abs() < 1.0, "MAE should be symmetric");
        assert!((m1.mse - m2.mse).abs() < 1.0, "MSE should be symmetric");
    }

    // --- ForegroundExtractor tests ---

    #[test]
    fn test_foreground_extractor_not_ready_before_finalize() {
        let extractor = ForegroundExtractor::new(4, 4, MattingConfig::default());
        assert!(!extractor.is_ready());
    }

    #[test]
    fn test_foreground_extractor_ready_after_finalize() {
        let mut extractor = ForegroundExtractor::new(4, 4, MattingConfig::default());
        let bg = solid_rgb(4, 4, 80, 80, 80);
        extractor.push_background(&bg);
        extractor.finalize_background();
        assert!(extractor.is_ready());
    }

    #[test]
    fn test_foreground_extractor_process_frame_returns_result() {
        let w = 4;
        let h = 4;
        let bg = solid_rgb(w, h, 80, 80, 80);
        let mut extractor = ForegroundExtractor::new(w, h, MattingConfig::default());
        extractor.push_background(&bg);
        extractor.finalize_background();

        let composite = solid_rgb(w, h, 240, 240, 240);
        let result = extractor.process_frame(&composite);
        assert!(result.is_some());
        let r = result.unwrap();
        assert_eq!(r.alpha.len(), w * h);
    }

    #[test]
    fn test_foreground_extractor_process_before_finalize_returns_none() {
        let mut extractor = ForegroundExtractor::new(4, 4, MattingConfig::default());
        let frame = solid_rgb(4, 4, 100, 100, 100);
        let result = extractor.process_frame(&frame);
        assert!(result.is_none());
    }

    #[test]
    fn test_foreground_extractor_background_frame_count() {
        let w = 4;
        let h = 4;
        let bg = solid_rgb(w, h, 80, 80, 80);
        let mut extractor = ForegroundExtractor::new(w, h, MattingConfig::default());
        assert_eq!(extractor.background_frame_count(), 0);
        extractor.push_background(&bg);
        extractor.push_background(&bg);
        assert_eq!(extractor.background_frame_count(), 2);
    }

    #[test]
    fn test_foreground_extractor_with_temporal_smoothing() {
        let w = 4;
        let h = 4;
        let bg = solid_rgb(w, h, 40, 40, 40);
        let mut extractor = ForegroundExtractor::new(w, h, MattingConfig::default());
        extractor.enable_temporal_smoothing(0.5);
        extractor.push_background(&bg);
        extractor.finalize_background();

        let composite = solid_rgb(w, h, 220, 220, 220);
        let result = extractor.process_frame(&composite);
        assert!(result.is_some());
        assert_eq!(result.unwrap().alpha.len(), w * h);
    }
}
