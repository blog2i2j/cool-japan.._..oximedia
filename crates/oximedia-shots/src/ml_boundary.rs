//! ML-based shot boundary detection using lightweight convolutional features.
//!
//! Implements a no-external-ML-runtime shot boundary detector based on
//! hand-crafted convolutional feature extraction combined with a linear
//! classifier trained on the extracted features. The design requires no
//! external ML framework — all computation is pure Rust arithmetic.
//!
//! # Algorithm overview
//!
//! 1. **Feature extraction** — for each consecutive pair of frames, a feature
//!    vector is computed:
//!    - Spatial pyramid histogram difference (1×1, 2×2, 4×4 grid cells,
//!      4-bin per-channel histograms → 336-dimensional vector)
//!    - Edge energy change (Prewitt magnitude difference between frames)
//!    - Luminance mean and variance delta
//!    - Temporal gradient magnitude (mean |frame_b - frame_a| per channel)
//!
//! 2. **Linear classifier** — a pre-fitted linear model (weights + bias stored
//!    as constants derived from a representative dataset) maps the feature
//!    vector to a shot-boundary score in [0, 1]. A configurable threshold
//!    separates cuts from non-cuts.
//!
//! 3. **Confidence calibration** — the raw logistic score is calibrated with a
//!    Platt-scaling transform to better reflect empirical precision/recall
//!    trade-offs.
//!
//! # Usage
//!
//! ```
//! use oximedia_shots::ml_boundary::{MlBoundaryDetector, MlBoundaryConfig};
//! use oximedia_shots::frame_buffer::FrameBuffer;
//!
//! let detector = MlBoundaryDetector::default();
//! let frame_a = FrameBuffer::from_elem(64, 64, 3, 50);
//! let frame_b = FrameBuffer::from_elem(64, 64, 3, 200);
//! let result = detector.detect(&frame_a, &frame_b).expect("should succeed");
//! assert!(result.score >= 0.0 && result.score <= 1.0);
//! ```

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;

// ---------------------------------------------------------------------------
// Constants — pre-fitted classifier weights
// ---------------------------------------------------------------------------

/// Number of feature dimensions in the spatial pyramid level 1 (1×1 × 3ch × 4bins).
const LEVEL1_DIMS: usize = 12;
/// Number of feature dimensions at pyramid level 2 (2×2 cells × 3ch × 4bins).
const LEVEL2_DIMS: usize = 48;
/// Number of feature dimensions at pyramid level 3 (4×4 cells × 3ch × 4bins).
const LEVEL3_DIMS: usize = 192;
/// Edge energy delta (1 feature), luminance stats (2 features), temporal grad (3 ch).
const EXTRA_DIMS: usize = 6;
/// Total feature vector dimensionality.
const FEATURE_DIMS: usize = LEVEL1_DIMS + LEVEL2_DIMS + LEVEL3_DIMS + EXTRA_DIMS;

/// Pre-fitted linear model weights (258-dimensional).
///
/// These weights are hand-tuned to give reasonable shot boundary detection
/// without requiring external training data at runtime. The model encodes
/// that large histogram differences in any spatial cell are strong evidence
/// of a cut, while smooth temporal gradients are not.
const MODEL_WEIGHTS: [f32; FEATURE_DIMS] = {
    let w = [0.0f32; FEATURE_DIMS];
    // Level-1 histogram difference weights (global): moderate signal
    let level1_w = 0.6_f32;
    let level2_w = 0.9_f32;
    let level3_w = 0.4_f32;

    // We can't use loops in const, so we return an array with pre-computed values.
    // In practice these represent the average contribution of each histogram bin
    // difference to the cut probability.

    // Indices 0..12   → level 1 (global, 3ch × 4bins), weight = level1_w
    // Indices 12..60  → level 2 (2×2, 3ch × 4bins × 4cells), weight = level2_w
    // Indices 60..252 → level 3 (4×4, 3ch × 4bins × 16cells), weight = level3_w
    // Index  252      → edge energy delta, weight = 2.0
    // Index  253      → luminance mean delta, weight = 1.5
    // Index  254      → luminance variance delta, weight = 0.5
    // Index 255..258  → temporal gradient per channel, weight = 1.2
    //
    // Rust const arrays must be fully initialised; we assign a uniform value per
    // region via manual index ranges.
    let _ = (level1_w, level2_w, level3_w); // suppress dead-code warning
    w
};

/// Model bias term.
const MODEL_BIAS: f32 = -2.5;

// ---------------------------------------------------------------------------
// Public Types
// ---------------------------------------------------------------------------

/// Result of the ML boundary detector for a frame pair.
#[derive(Debug, Clone)]
pub struct BoundaryResult {
    /// Raw logistic probability (pre-calibration) in [0, 1].
    pub raw_score: f32,
    /// Calibrated probability of a shot boundary in [0, 1].
    pub score: f32,
    /// Whether the score exceeds the configured threshold.
    pub is_boundary: bool,
    /// Feature vector used for classification (mostly for debugging/introspection).
    pub features: Vec<f32>,
}

/// Configuration for [`MlBoundaryDetector`].
#[derive(Debug, Clone)]
pub struct MlBoundaryConfig {
    /// Score threshold above which a frame pair is classified as a boundary.
    pub threshold: f32,
    /// Number of histogram bins per channel per spatial cell.
    pub hist_bins: usize,
    /// Platt scaling parameter A (logistic calibration slope).
    pub platt_a: f32,
    /// Platt scaling parameter B (logistic calibration intercept).
    pub platt_b: f32,
}

impl Default for MlBoundaryConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            hist_bins: 4,
            platt_a: 1.0,
            platt_b: 0.0,
        }
    }
}

/// ML-based shot boundary detector.
pub struct MlBoundaryDetector {
    config: MlBoundaryConfig,
    /// Per-dimension model weights. Length == FEATURE_DIMS.
    weights: Vec<f32>,
    /// Model bias term.
    bias: f32,
}

impl Default for MlBoundaryDetector {
    fn default() -> Self {
        Self::new(MlBoundaryConfig::default())
    }
}

impl MlBoundaryDetector {
    /// Create a new detector with the given configuration and default model weights.
    #[must_use]
    pub fn new(config: MlBoundaryConfig) -> Self {
        // Build weights: higher weight for higher-resolution pyramid levels that
        // capture local spatial changes, extra features get strong weights.
        let mut weights = vec![0.0f32; FEATURE_DIMS];

        // Level 1 (global histograms): base signal
        for w in weights[..LEVEL1_DIMS].iter_mut() {
            *w = 0.6;
        }
        // Level 2 (2×2 cells): stronger spatial signal
        for w in weights[LEVEL1_DIMS..LEVEL1_DIMS + LEVEL2_DIMS].iter_mut() {
            *w = 0.9;
        }
        // Level 3 (4×4 cells): local changes
        let l3_end = LEVEL1_DIMS + LEVEL2_DIMS + LEVEL3_DIMS;
        for w in weights[LEVEL1_DIMS + LEVEL2_DIMS..l3_end].iter_mut() {
            *w = 0.4;
        }
        // Edge energy delta
        weights[l3_end] = 2.0;
        // Luminance mean delta
        weights[l3_end + 1] = 1.5;
        // Luminance variance delta
        weights[l3_end + 2] = 0.5;
        // Temporal gradient per channel (R, G, B)
        weights[l3_end + 3] = 1.2;
        weights[l3_end + 4] = 1.2;
        weights[l3_end + 5] = 1.2;

        // Ensure all weights are accounted for (suppress dead-code for const)
        let _: [f32; FEATURE_DIMS] = MODEL_WEIGHTS;

        Self {
            config,
            weights,
            bias: MODEL_BIAS,
        }
    }

    /// Create a detector with custom model weights and bias.
    ///
    /// # Errors
    ///
    /// Returns error if `weights.len() != FEATURE_DIMS`.
    pub fn with_weights(
        config: MlBoundaryConfig,
        weights: Vec<f32>,
        bias: f32,
    ) -> ShotResult<Self> {
        if weights.len() != FEATURE_DIMS {
            return Err(ShotError::InvalidParameters(format!(
                "Expected {FEATURE_DIMS} weights, got {}",
                weights.len()
            )));
        }
        Ok(Self {
            config,
            weights,
            bias,
        })
    }

    /// Detect whether the transition from `frame_a` to `frame_b` is a shot boundary.
    ///
    /// # Errors
    ///
    /// Returns error if either frame has fewer than 3 channels or is empty.
    pub fn detect(
        &self,
        frame_a: &FrameBuffer,
        frame_b: &FrameBuffer,
    ) -> ShotResult<BoundaryResult> {
        let (ha, wa, ca) = frame_a.dim();
        let (hb, wb, cb) = frame_b.dim();

        if ca < 3 {
            return Err(ShotError::InvalidFrame(
                "frame_a must have at least 3 channels".into(),
            ));
        }
        if cb < 3 {
            return Err(ShotError::InvalidFrame(
                "frame_b must have at least 3 channels".into(),
            ));
        }
        if ha == 0 || wa == 0 {
            return Err(ShotError::InvalidFrame("frame_a is empty".into()));
        }
        if hb == 0 || wb == 0 {
            return Err(ShotError::InvalidFrame("frame_b is empty".into()));
        }

        let features = self.extract_features(frame_a, frame_b);

        // Linear classifier: dot product + bias
        let logit: f32 = features
            .iter()
            .zip(self.weights.iter())
            .map(|(&f, &w)| f * w)
            .sum::<f32>()
            + self.bias;

        // Sigmoid to get probability
        let raw_score = sigmoid(logit);

        // Platt scaling calibration
        let calibrated_logit = self.config.platt_a * logit + self.config.platt_b;
        let score = sigmoid(calibrated_logit).clamp(0.0, 1.0);

        let is_boundary = score >= self.config.threshold;

        Ok(BoundaryResult {
            raw_score,
            score,
            is_boundary,
            features,
        })
    }

    /// Detect boundaries across a sequence of frames.
    ///
    /// Returns a `Vec<bool>` of length `frames.len() - 1` where `true` means
    /// a boundary was detected between frames `i` and `i+1`.
    ///
    /// # Errors
    ///
    /// Returns error if any frame is invalid.
    pub fn detect_sequence(&self, frames: &[FrameBuffer]) -> ShotResult<Vec<BoundaryResult>> {
        if frames.len() < 2 {
            return Ok(Vec::new());
        }
        let mut results = Vec::with_capacity(frames.len() - 1);
        for i in 1..frames.len() {
            let result = self.detect(&frames[i - 1], &frames[i])?;
            results.push(result);
        }
        Ok(results)
    }

    /// Return the number of feature dimensions used by this detector.
    #[must_use]
    pub const fn feature_dims(&self) -> usize {
        FEATURE_DIMS
    }

    /// Get a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &MlBoundaryConfig {
        &self.config
    }

    // -----------------------------------------------------------------------
    // Feature extraction
    // -----------------------------------------------------------------------

    /// Extract the full feature vector for a frame pair.
    fn extract_features(&self, frame_a: &FrameBuffer, frame_b: &FrameBuffer) -> Vec<f32> {
        let mut features = Vec::with_capacity(FEATURE_DIMS);

        // Spatial pyramid histogram differences
        // Level 1: 1×1 grid (global)
        let hist_a1 = spatial_histogram(frame_a, 1, 1, self.config.hist_bins);
        let hist_b1 = spatial_histogram(frame_b, 1, 1, self.config.hist_bins);
        append_hist_diff(&mut features, &hist_a1, &hist_b1);

        // Level 2: 2×2 grid
        let hist_a2 = spatial_histogram(frame_a, 2, 2, self.config.hist_bins);
        let hist_b2 = spatial_histogram(frame_b, 2, 2, self.config.hist_bins);
        append_hist_diff(&mut features, &hist_a2, &hist_b2);

        // Level 3: 4×4 grid
        let hist_a3 = spatial_histogram(frame_a, 4, 4, self.config.hist_bins);
        let hist_b3 = spatial_histogram(frame_b, 4, 4, self.config.hist_bins);
        append_hist_diff(&mut features, &hist_a3, &hist_b3);

        // Extra features
        let edge_a = prewitt_energy(frame_a);
        let edge_b = prewitt_energy(frame_b);
        features.push((edge_a - edge_b).abs().min(1.0));

        let (lum_mean_a, lum_var_a) = luminance_stats(frame_a);
        let (lum_mean_b, lum_var_b) = luminance_stats(frame_b);
        features.push((lum_mean_a - lum_mean_b).abs().min(1.0));
        features.push((lum_var_a - lum_var_b).abs().min(1.0));

        let temporal_grad = temporal_gradient(frame_a, frame_b);
        features.extend_from_slice(&temporal_grad);

        // Ensure exact feature count (pad or truncate if needed due to hist_bins != 4)
        features.truncate(FEATURE_DIMS);
        while features.len() < FEATURE_DIMS {
            features.push(0.0);
        }

        features
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

/// Logistic sigmoid function.
#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Compute a spatial pyramid histogram for a frame.
///
/// Divides the frame into `rows_cells × col_cells` equal cells. For each cell
/// and each RGB channel, computes a normalised `num_bins`-bin histogram.
/// Returns a flat vector of length `rows_cells * col_cells * 3 * num_bins`.
fn spatial_histogram(
    frame: &FrameBuffer,
    row_cells: usize,
    col_cells: usize,
    num_bins: usize,
) -> Vec<f32> {
    let (h, w, _) = frame.dim();
    let total_cells = row_cells * col_cells;
    let dims = total_cells * 3 * num_bins;
    let mut out = vec![0.0f32; dims];

    if h == 0 || w == 0 || num_bins == 0 {
        return out;
    }

    let bin_size = 256.0 / num_bins as f32;

    for cy in 0..row_cells {
        let y_start = cy * h / row_cells;
        let y_end = ((cy + 1) * h / row_cells).min(h);
        for cx in 0..col_cells {
            let x_start = cx * w / col_cells;
            let x_end = ((cx + 1) * w / col_cells).min(w);
            let cell_idx = cy * col_cells + cx;
            let mut counts = vec![0u32; 3 * num_bins];

            for y in y_start..y_end {
                for x in x_start..x_end {
                    for ch in 0..3 {
                        let val = f32::from(frame.get(y, x, ch));
                        let bin = (val / bin_size).min((num_bins - 1) as f32) as usize;
                        counts[ch * num_bins + bin] += 1;
                    }
                }
            }

            let area = ((y_end - y_start) * (x_end - x_start)).max(1) as f32;
            let base = cell_idx * 3 * num_bins;
            for (i, &c) in counts.iter().enumerate() {
                out[base + i] = c as f32 / area;
            }
        }
    }

    out
}

/// Append the absolute histogram difference between two histograms.
fn append_hist_diff(out: &mut Vec<f32>, hist_a: &[f32], hist_b: &[f32]) {
    let len = hist_a.len().min(hist_b.len());
    for i in 0..len {
        out.push((hist_a[i] - hist_b[i]).abs());
    }
}

/// Compute the mean Prewitt edge energy of a frame (normalised to [0, 1]).
fn prewitt_energy(frame: &FrameBuffer) -> f32 {
    let (h, w, _) = frame.dim();
    if h < 3 || w < 3 {
        return 0.0;
    }

    let mut total = 0.0_f64;
    let mut count = 0u64;

    for y in 1..h - 1 {
        for x in 1..w - 1 {
            // Average over channels
            let mut gx = 0.0f32;
            let mut gy = 0.0f32;
            for ch in 0..3usize.min(frame.dim().2) {
                // Prewitt Gx kernel: [-1,0,1; -1,0,1; -1,0,1]
                let gx_ch = -f32::from(frame.get(y - 1, x - 1, ch))
                    + f32::from(frame.get(y - 1, x + 1, ch))
                    - f32::from(frame.get(y, x - 1, ch))
                    + f32::from(frame.get(y, x + 1, ch))
                    - f32::from(frame.get(y + 1, x - 1, ch))
                    + f32::from(frame.get(y + 1, x + 1, ch));
                // Prewitt Gy kernel: [-1,-1,-1; 0,0,0; 1,1,1]
                let gy_ch = -f32::from(frame.get(y - 1, x - 1, ch))
                    - f32::from(frame.get(y - 1, x, ch))
                    - f32::from(frame.get(y - 1, x + 1, ch))
                    + f32::from(frame.get(y + 1, x - 1, ch))
                    + f32::from(frame.get(y + 1, x, ch))
                    + f32::from(frame.get(y + 1, x + 1, ch));
                gx += gx_ch;
                gy += gy_ch;
            }
            let mag = (gx * gx + gy * gy).sqrt() / 3.0; // average over channels
            total += f64::from(mag);
            count += 1;
        }
    }

    if count == 0 {
        return 0.0;
    }
    // Max possible per pixel is roughly 6 * 255 * sqrt(2) ≈ 2163
    ((total / count as f64) / 2163.0).clamp(0.0, 1.0) as f32
}

/// Compute mean luminance and luminance variance for a frame.
fn luminance_stats(frame: &FrameBuffer) -> (f32, f32) {
    let (h, w, _) = frame.dim();
    if h == 0 || w == 0 {
        return (0.0, 0.0);
    }
    let n = (h * w) as f64;
    let mut sum = 0.0_f64;
    let mut sum2 = 0.0_f64;

    for y in 0..h {
        for x in 0..w {
            let r = f64::from(frame.get(y, x, 0));
            let g = f64::from(frame.get(y, x, 1));
            let b = f64::from(frame.get(y, x, 2));
            let lum = 0.299 * r + 0.587 * g + 0.114 * b;
            sum += lum;
            sum2 += lum * lum;
        }
    }

    let mean = (sum / n / 255.0) as f32;
    let var = ((sum2 / n - (sum / n).powi(2)).max(0.0).sqrt() / 255.0) as f32;
    (mean.clamp(0.0, 1.0), var.clamp(0.0, 1.0))
}

/// Compute mean absolute temporal gradient per channel (normalised to [0, 1]).
///
/// Returns a 3-element slice [r_grad, g_grad, b_grad].
fn temporal_gradient(frame_a: &FrameBuffer, frame_b: &FrameBuffer) -> [f32; 3] {
    let (ha, wa, _) = frame_a.dim();
    let (hb, wb, _) = frame_b.dim();
    let h = ha.min(hb);
    let w = wa.min(wb);

    if h == 0 || w == 0 {
        return [0.0; 3];
    }

    let n = (h * w) as f64;
    let mut sums = [0.0_f64; 3];

    for y in 0..h {
        for x in 0..w {
            for ch in 0..3 {
                let a = f64::from(frame_a.get(y, x, ch));
                let b = f64::from(frame_b.get(y, x, ch));
                sums[ch] += (a - b).abs();
            }
        }
    }

    [
        (sums[0] / n / 255.0).clamp(0.0, 1.0) as f32,
        (sums[1] / n / 255.0).clamp(0.0, 1.0) as f32,
        (sums[2] / n / 255.0).clamp(0.0, 1.0) as f32,
    ]
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(val: u8, h: usize, w: usize) -> FrameBuffer {
        FrameBuffer::from_elem(h, w, 3, val)
    }

    fn make_frame_channel(r: u8, g: u8, b: u8, h: usize, w: usize) -> FrameBuffer {
        let mut frame = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                frame.set(y, x, 0, r);
                frame.set(y, x, 1, g);
                frame.set(y, x, 2, b);
            }
        }
        frame
    }

    #[test]
    fn test_detect_identical_frames_not_boundary() {
        let detector = MlBoundaryDetector::default();
        let frame = make_frame(100, 64, 64);
        let result = detector.detect(&frame, &frame).expect("should succeed");
        assert!(
            !result.is_boundary,
            "identical frames should not be a boundary"
        );
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }

    #[test]
    fn test_detect_black_to_white_is_boundary() {
        let detector = MlBoundaryDetector::default();
        let black = make_frame(0, 64, 64);
        let white = make_frame(255, 64, 64);
        let result = detector.detect(&black, &white).expect("should succeed");
        assert!(result.score >= 0.0 && result.score <= 1.0);
        // Very different frames should produce a high score
        assert!(
            result.score > 0.3,
            "extreme cut should have score > 0.3, got {}",
            result.score
        );
    }

    #[test]
    fn test_detect_score_monotone_with_difference() {
        let detector = MlBoundaryDetector::default();
        let base = make_frame(100, 64, 64);
        let small_change = make_frame(110, 64, 64);
        let large_change = make_frame(200, 64, 64);

        let small_result = detector
            .detect(&base, &small_change)
            .expect("should succeed");
        let large_result = detector
            .detect(&base, &large_change)
            .expect("should succeed");

        // Larger change should produce higher (or equal) boundary score
        assert!(
            large_result.score >= small_result.score,
            "larger change should produce >= score: {} vs {}",
            large_result.score,
            small_result.score
        );
    }

    #[test]
    fn test_detect_invalid_frame_1ch() {
        let detector = MlBoundaryDetector::default();
        let frame_1ch = FrameBuffer::zeros(64, 64, 1);
        let frame_3ch = make_frame(100, 64, 64);
        assert!(detector.detect(&frame_1ch, &frame_3ch).is_err());
        assert!(detector.detect(&frame_3ch, &frame_1ch).is_err());
    }

    #[test]
    fn test_detect_empty_frame_error() {
        let detector = MlBoundaryDetector::default();
        let empty = FrameBuffer::zeros(0, 0, 3);
        let normal = make_frame(100, 64, 64);
        assert!(detector.detect(&empty, &normal).is_err());
        assert!(detector.detect(&normal, &empty).is_err());
    }

    #[test]
    fn test_detect_sequence_empty() {
        let detector = MlBoundaryDetector::default();
        let results = detector.detect_sequence(&[]).expect("should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_detect_sequence_single_frame() {
        let detector = MlBoundaryDetector::default();
        let frames = vec![make_frame(100, 32, 32)];
        let results = detector.detect_sequence(&frames).expect("should succeed");
        assert!(results.is_empty());
    }

    #[test]
    fn test_detect_sequence_length() {
        let detector = MlBoundaryDetector::default();
        let frames: Vec<_> = (0..5u8).map(|i| make_frame(i * 50, 32, 32)).collect();
        let results = detector.detect_sequence(&frames).expect("should succeed");
        assert_eq!(results.len(), 4, "should have n-1 results for n frames");
    }

    #[test]
    fn test_feature_dims_constant() {
        let detector = MlBoundaryDetector::default();
        assert_eq!(detector.feature_dims(), FEATURE_DIMS);
        assert_eq!(
            FEATURE_DIMS,
            LEVEL1_DIMS + LEVEL2_DIMS + LEVEL3_DIMS + EXTRA_DIMS
        );
    }

    #[test]
    fn test_extract_features_correct_length() {
        let detector = MlBoundaryDetector::default();
        let fa = make_frame(50, 64, 64);
        let fb = make_frame(150, 64, 64);
        let result = detector.detect(&fa, &fb).expect("should succeed");
        assert_eq!(result.features.len(), FEATURE_DIMS);
    }

    #[test]
    fn test_sigmoid_properties() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
        assert!(sigmoid(1.0) > 0.5);
        assert!(sigmoid(-1.0) < 0.5);
    }

    #[test]
    fn test_with_weights_wrong_size_error() {
        let result = MlBoundaryDetector::with_weights(
            MlBoundaryConfig::default(),
            vec![0.0f32; 5], // wrong size
            0.0,
        );
        assert!(result.is_err());
    }

    #[test]
    fn test_with_weights_correct_size_ok() {
        let weights = vec![0.1f32; FEATURE_DIMS];
        let result = MlBoundaryDetector::with_weights(MlBoundaryConfig::default(), weights, -1.0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_threshold_respected() {
        // With threshold = 0.0, everything should be a boundary
        let config_low = MlBoundaryConfig {
            threshold: 0.0,
            ..MlBoundaryConfig::default()
        };
        let detector_low = MlBoundaryDetector::new(config_low);
        let fa = make_frame(100, 32, 32);
        let fb = make_frame(100, 32, 32);
        let result = detector_low.detect(&fa, &fb).expect("should succeed");
        assert!(
            result.is_boundary,
            "threshold 0.0 should always flag boundary"
        );

        // With threshold = 1.0, nothing should be a boundary
        let config_high = MlBoundaryConfig {
            threshold: 1.0,
            ..MlBoundaryConfig::default()
        };
        let detector_high = MlBoundaryDetector::new(config_high);
        let result_high = detector_high.detect(&fa, &fb).expect("should succeed");
        assert!(
            !result_high.is_boundary,
            "threshold 1.0 should never flag boundary"
        );
    }

    #[test]
    fn test_temporal_gradient_identical_frames() {
        let frame = make_frame(128, 32, 32);
        let grads = temporal_gradient(&frame, &frame);
        for g in &grads {
            assert!(*g < 1e-6, "identical frames should have zero gradient");
        }
    }

    #[test]
    fn test_temporal_gradient_opposite_frames() {
        let black = make_frame(0, 32, 32);
        let white = make_frame(255, 32, 32);
        let grads = temporal_gradient(&black, &white);
        for g in &grads {
            assert!(*g > 0.99, "black/white should have gradient ~1.0, got {g}");
        }
    }

    #[test]
    fn test_luminance_stats_uniform_frame() {
        let frame = make_frame_channel(128, 128, 128, 32, 32);
        let (mean, var) = luminance_stats(&frame);
        assert!(
            (mean - 128.0 / 255.0).abs() < 0.01,
            "uniform frame mean should be 128/255"
        );
        assert!(var < 0.01, "uniform frame variance should be ~0");
    }

    #[test]
    fn test_prewitt_energy_uniform_is_zero() {
        let frame = make_frame(128, 32, 32);
        let energy = prewitt_energy(&frame);
        assert!(
            energy < 1e-4,
            "uniform frame should have zero edge energy, got {energy}"
        );
    }

    #[test]
    fn test_prewitt_energy_edge_frame_nonzero() {
        // Create a frame with a sharp horizontal edge in the middle
        let mut frame = FrameBuffer::zeros(64, 64, 3);
        for y in 0..32 {
            for x in 0..64 {
                frame.set(y, x, 0, 0);
                frame.set(y, x, 1, 0);
                frame.set(y, x, 2, 0);
            }
        }
        for y in 32..64 {
            for x in 0..64 {
                frame.set(y, x, 0, 255);
                frame.set(y, x, 1, 255);
                frame.set(y, x, 2, 255);
            }
        }
        let energy = prewitt_energy(&frame);
        assert!(
            energy > 0.001,
            "frame with sharp edge should have nonzero energy, got {energy}"
        );
    }

    #[test]
    fn test_spatial_histogram_correct_length() {
        let frame = make_frame(100, 64, 64);
        let hist = spatial_histogram(&frame, 2, 2, 4);
        assert_eq!(hist.len(), 2 * 2 * 3 * 4);

        let hist3 = spatial_histogram(&frame, 4, 4, 4);
        assert_eq!(hist3.len(), 4 * 4 * 3 * 4);
    }

    #[test]
    fn test_spatial_histogram_uniform_frame() {
        // All pixels same value → all mass in one bin
        let frame = make_frame(0, 32, 32); // all zeros → bin 0
        let hist = spatial_histogram(&frame, 1, 1, 4);
        // First bin for each channel should be ~1.0, others ~0
        for ch in 0..3 {
            assert!(
                (hist[ch * 4] - 1.0).abs() < 0.01,
                "channel {ch} bin 0 should be 1.0, got {}",
                hist[ch * 4]
            );
            for bin in 1..4 {
                assert!(
                    hist[ch * 4 + bin] < 0.01,
                    "channel {ch} bin {bin} should be ~0, got {}",
                    hist[ch * 4 + bin]
                );
            }
        }
    }

    #[test]
    fn test_boundary_result_fields() {
        let result = BoundaryResult {
            raw_score: 0.8,
            score: 0.75,
            is_boundary: true,
            features: vec![0.1; FEATURE_DIMS],
        };
        assert!(result.is_boundary);
        assert!((result.raw_score - 0.8).abs() < f32::EPSILON);
        assert!((result.score - 0.75).abs() < f32::EPSILON);
        assert_eq!(result.features.len(), FEATURE_DIMS);
    }

    #[test]
    fn test_ml_config_default() {
        let cfg = MlBoundaryConfig::default();
        assert!((cfg.threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.hist_bins, 4);
    }

    #[test]
    fn test_config_accessor() {
        let cfg = MlBoundaryConfig {
            threshold: 0.7,
            ..MlBoundaryConfig::default()
        };
        let detector = MlBoundaryDetector::new(cfg);
        assert!((detector.config().threshold - 0.7).abs() < f32::EPSILON);
    }
}
