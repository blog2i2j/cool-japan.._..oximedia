//! Optical flow estimation model for motion analysis in video frames.
//!
//! This module provides a lightweight, pure-Rust optical flow estimator using a
//! PWC-Net–inspired coarse-to-fine convolutional architecture.  The model takes
//! two consecutive video frames as input and outputs a dense 2-channel flow
//! field representing per-pixel `(dx, dy)` motion vectors.
//!
//! ## Architecture
//!
//! ```text
//! Frame A [3, H, W] ─┐
//!                    ├──► Feature Encoder (shared weights)
//! Frame B [3, H, W] ─┘       │
//!                             │  [feat_channels, H/4, W/4]  (stride-4 encoder)
//!                             ▼
//!                    Correlation Volume (local neighbourhood)
//!                             │  [search_area², H/4, W/4]
//!                             ▼
//!                    Flow Decoder (Conv + upsample)
//!                             │
//!                             ▼
//!                    Flow field [2, H, W]  (bilinear upsampled)
//! ```
//!
//! The feature encoder applies three stride-2 convolutions (3→16→32→32) to
//! produce a quarter-resolution feature map.  The correlation volume is computed
//! by exhaustive local cross-correlation in a `(2*d+1) × (2*d+1)` search window
//! (default `d = 2`, i.e. 25 correlation channels).  The flow decoder maps the
//! correlation volume through two Conv2d layers and produces a 2-channel
//! quarter-resolution flow, which is bilinearly upsampled to full resolution.
//!
//! All weights are **zero-initialised** at construction time; the model can be
//! used immediately for testing or populated with pre-trained weights by
//! assigning the public layer fields.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::optical_flow::OpticalFlowEstimator;
//! use oximedia_neural::tensor::Tensor;
//!
//! let estimator = OpticalFlowEstimator::new(2).unwrap();
//! let frame_a = Tensor::zeros(vec![3, 32, 32]).unwrap();
//! let frame_b = Tensor::zeros(vec![3, 32, 32]).unwrap();
//! let flow = estimator.forward(&frame_a, &frame_b).unwrap();
//! // flow shape: [2, 32, 32]  (dx, dy per pixel)
//! assert_eq!(flow.shape(), &[2, 32, 32]);
//! ```

use crate::activations::{apply_activation, ActivationFn};
use crate::error::NeuralError;
use crate::layers::Conv2dLayer;
use crate::tensor::Tensor;

// ─────────────────────────────────────────────────────────────────────────────
// Flow vector type
// ─────────────────────────────────────────────────────────────────────────────

/// A single 2-D motion vector `(dx, dy)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FlowVector {
    /// Horizontal displacement in pixels (positive → right).
    pub dx: f32,
    /// Vertical displacement in pixels (positive → down).
    pub dy: f32,
}

impl FlowVector {
    /// Creates a new `FlowVector`.
    pub fn new(dx: f32, dy: f32) -> Self {
        Self { dx, dy }
    }

    /// Returns the Euclidean magnitude of the motion vector.
    pub fn magnitude(&self) -> f32 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Returns the angle (in radians, in `[-π, π]`) of the motion vector,
    /// measured counter-clockwise from the positive X axis.
    pub fn angle_rad(&self) -> f32 {
        self.dy.atan2(self.dx)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Correlation volume
// ─────────────────────────────────────────────────────────────────────────────

/// Computes the local cross-correlation volume between two feature maps.
///
/// For each spatial location `(h, w)` in feature map A, the correlation with
/// every offset `(dh, dw)` in `[-d, d] × [-d, d]` in feature map B is
/// computed as the dot product of the corresponding feature vectors (divided
/// by `sqrt(C)` for numerical stability).
///
/// Returns a tensor of shape `[(2d+1)², H, W]` where out-of-bounds positions
/// in B are treated as zero.
///
/// * `feat_a` — `[C, H, W]` feature tensor for frame A.
/// * `feat_b` — `[C, H, W]` feature tensor for frame B.
/// * `d`      — displacement radius (search window is `(2d+1) × (2d+1)`).
pub fn correlation_volume(
    feat_a: &Tensor,
    feat_b: &Tensor,
    d: usize,
) -> Result<Tensor, NeuralError> {
    if feat_a.ndim() != 3 || feat_b.ndim() != 3 {
        return Err(NeuralError::InvalidShape(
            "correlation_volume: both inputs must be [C, H, W]".to_string(),
        ));
    }
    if feat_a.shape() != feat_b.shape() {
        return Err(NeuralError::ShapeMismatch(format!(
            "correlation_volume: feat_a {:?} != feat_b {:?}",
            feat_a.shape(),
            feat_b.shape()
        )));
    }

    let c = feat_a.shape()[0];
    let h = feat_a.shape()[1];
    let w = feat_a.shape()[2];

    let search = 2 * d + 1;
    let num_disp = search * search;
    let scale = (c as f32).sqrt().max(1.0);

    let a = feat_a.data();
    let b = feat_b.data();

    let mut out = vec![0.0_f32; num_disp * h * w];

    for (k_dh, dh_signed) in (-(d as isize)..=(d as isize)).enumerate() {
        for (k_dw, dw_signed) in (-(d as isize)..=(d as isize)).enumerate() {
            let disp_idx = k_dh * search + k_dw;
            for ah in 0..h {
                let bh = ah as isize + dh_signed;
                for aw in 0..w {
                    let bw = aw as isize + dw_signed;
                    // Out-of-bounds → correlation is 0.
                    if bh < 0 || bh >= h as isize || bw < 0 || bw >= w as isize {
                        continue;
                    }
                    let bh = bh as usize;
                    let bw = bw as usize;
                    // Dot product over channel dimension.
                    let mut dot = 0.0_f32;
                    for ch in 0..c {
                        let a_idx = ch * h * w + ah * w + aw;
                        let b_idx = ch * h * w + bh * w + bw;
                        dot += a[a_idx] * b[b_idx];
                    }
                    let out_idx = disp_idx * h * w + ah * w + aw;
                    out[out_idx] = dot / scale;
                }
            }
        }
    }

    Tensor::from_data(out, vec![num_disp, h, w])
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear upsampling
// ─────────────────────────────────────────────────────────────────────────────

/// Bilinearly upsamples a `[C, H, W]` tensor by an integer scale factor.
///
/// Each output pixel is mapped back to the input grid and its value is
/// computed via bilinear interpolation, matching PyTorch `align_corners=False`.
///
/// Returns a tensor of shape `[C, H*scale, W*scale]`.
pub fn bilinear_upsample(input: &Tensor, scale: usize) -> Result<Tensor, NeuralError> {
    if input.ndim() != 3 {
        return Err(NeuralError::InvalidShape(format!(
            "bilinear_upsample: expected [C, H, W], got rank {}",
            input.ndim()
        )));
    }
    if scale == 0 {
        return Err(NeuralError::InvalidShape(
            "bilinear_upsample: scale must be > 0".to_string(),
        ));
    }
    if scale == 1 {
        return Ok(input.clone());
    }

    let c = input.shape()[0];
    let in_h = input.shape()[1];
    let in_w = input.shape()[2];
    let out_h = in_h * scale;
    let out_w = in_w * scale;

    let src = input.data();
    let mut out = vec![0.0_f32; c * out_h * out_w];

    for ch in 0..c {
        let ch_src_base = ch * in_h * in_w;
        let ch_dst_base = ch * out_h * out_w;
        for oy in 0..out_h {
            // align_corners=False: maps output pixel centre to input pixel centre.
            let fy = (oy as f32 + 0.5) / scale as f32 - 0.5;
            let y0 = fy.floor() as isize;
            let y1 = y0 + 1;
            let wy1 = fy - y0 as f32;
            let wy0 = 1.0 - wy1;

            for ox in 0..out_w {
                let fx = (ox as f32 + 0.5) / scale as f32 - 0.5;
                let x0 = fx.floor() as isize;
                let x1 = x0 + 1;
                let wx1 = fx - x0 as f32;
                let wx0 = 1.0 - wx1;

                // Clamp to valid range.
                let y0c = y0.clamp(0, in_h as isize - 1) as usize;
                let y1c = y1.clamp(0, in_h as isize - 1) as usize;
                let x0c = x0.clamp(0, in_w as isize - 1) as usize;
                let x1c = x1.clamp(0, in_w as isize - 1) as usize;

                let v00 = src[ch_src_base + y0c * in_w + x0c];
                let v01 = src[ch_src_base + y0c * in_w + x1c];
                let v10 = src[ch_src_base + y1c * in_w + x0c];
                let v11 = src[ch_src_base + y1c * in_w + x1c];

                let val = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
                out[ch_dst_base + oy * out_w + ox] = val;
            }
        }
    }

    Tensor::from_data(out, vec![c, out_h, out_w])
}

// ─────────────────────────────────────────────────────────────────────────────
// OpticalFlowEstimator
// ─────────────────────────────────────────────────────────────────────────────

/// Lightweight convolutional optical flow estimator.
///
/// The model encodes both input frames with a shared three-stage stride-2
/// encoder, computes a local correlation volume, and decodes the volume into a
/// 2-channel flow field that is bilinearly upsampled to the original resolution.
#[derive(Debug, Clone)]
pub struct OpticalFlowEstimator {
    /// Encoder stage 1: 3 → 16 channels, 3×3, stride 2.
    pub enc1: Conv2dLayer,
    /// Encoder stage 2: 16 → 32 channels, 3×3, stride 2.
    pub enc2: Conv2dLayer,
    /// Decoder stage 1: correlation_channels → 32, 3×3.
    pub dec1: Conv2dLayer,
    /// Decoder stage 2: 32 → 2 channels (flow), 1×1.
    pub dec2: Conv2dLayer,
    /// Displacement radius for correlation (search window = `(2d+1)²` channels).
    pub d: usize,
}

impl OpticalFlowEstimator {
    /// Creates a zero-initialised `OpticalFlowEstimator`.
    ///
    /// * `d` — displacement radius for the correlation volume; the search
    ///   window will be `(2*d+1) × (2*d+1)` locations.  Must be `>= 1`.
    pub fn new(d: usize) -> Result<Self, NeuralError> {
        if d == 0 {
            return Err(NeuralError::InvalidShape(
                "OpticalFlowEstimator: d must be >= 1".to_string(),
            ));
        }
        let search = 2 * d + 1;
        let corr_channels = search * search;

        let enc1 = Conv2dLayer::new(3, 16, 3, 3, (2, 2), (1, 1))?;
        let enc2 = Conv2dLayer::new(16, 32, 3, 3, (2, 2), (1, 1))?;
        let dec1 = Conv2dLayer::new(corr_channels, 32, 3, 3, (1, 1), (1, 1))?;
        let dec2 = Conv2dLayer::new(32, 2, 1, 1, (1, 1), (0, 0))?;

        Ok(Self {
            enc1,
            enc2,
            dec1,
            dec2,
            d,
        })
    }

    /// Encodes a single `[3, H, W]` RGB frame to a `[32, H/4, W/4]` feature map.
    fn encode(&self, frame: &Tensor) -> Result<Tensor, NeuralError> {
        let f1 = apply_activation(&self.enc1.forward(frame)?, &ActivationFn::Relu);
        let f2 = apply_activation(&self.enc2.forward(&f1)?, &ActivationFn::Relu);
        Ok(f2)
    }

    /// Runs the forward pass on two consecutive frames.
    ///
    /// Both frames must be `[3, H, W]` with identical spatial dimensions.
    /// Returns a `[2, H, W]` flow field where channel 0 is `dx` and channel 1
    /// is `dy`, in units of pixels.
    pub fn forward(&self, frame_a: &Tensor, frame_b: &Tensor) -> Result<Tensor, NeuralError> {
        // ── Input validation ──────────────────────────────────────────────────
        if frame_a.ndim() != 3 || frame_b.ndim() != 3 {
            return Err(NeuralError::InvalidShape(
                "OpticalFlowEstimator::forward: both frames must be [3, H, W]".to_string(),
            ));
        }
        if frame_a.shape() != frame_b.shape() {
            return Err(NeuralError::ShapeMismatch(format!(
                "OpticalFlowEstimator::forward: frame shapes differ: {:?} vs {:?}",
                frame_a.shape(),
                frame_b.shape()
            )));
        }
        if frame_a.shape()[0] != 3 {
            return Err(NeuralError::ShapeMismatch(format!(
                "OpticalFlowEstimator::forward: expected 3-channel frames, got {}",
                frame_a.shape()[0]
            )));
        }
        let orig_h = frame_a.shape()[1];
        let orig_w = frame_a.shape()[2];
        if orig_h < 4 || orig_w < 4 {
            return Err(NeuralError::InvalidShape(
                "OpticalFlowEstimator::forward: frames must be at least 4×4 pixels".to_string(),
            ));
        }

        // ── Encode both frames ────────────────────────────────────────────────
        let feat_a = self.encode(frame_a)?; // [32, H/4, W/4]
        let feat_b = self.encode(frame_b)?; // [32, H/4, W/4]

        // ── Correlation volume ────────────────────────────────────────────────
        let corr = correlation_volume(&feat_a, &feat_b, self.d)?; // [(2d+1)², H/4, W/4]

        // ── Flow decoder ──────────────────────────────────────────────────────
        let d1 = apply_activation(&self.dec1.forward(&corr)?, &ActivationFn::Relu); // [32, H/4, W/4]
        let flow_small = self.dec2.forward(&d1)?; // [2, H/4, W/4]

        // ── Bilinear upsample to original resolution ──────────────────────────
        // The encoder downsamples by stride-2 twice → factor 4.
        let flow = bilinear_upsample(&flow_small, 4)?; // [2, H, W]

        // Trim to exact input size (upsample may overshoot for non-divisible sizes).
        trim_to_size(flow, orig_h, orig_w)
    }

    /// Extracts the dominant motion vector by averaging the flow field.
    ///
    /// Useful for camera motion estimation or global motion compensation.
    pub fn mean_flow(flow: &Tensor) -> Result<FlowVector, NeuralError> {
        if flow.ndim() != 3 || flow.shape()[0] != 2 {
            return Err(NeuralError::InvalidShape(
                "mean_flow: expected [2, H, W] flow field".to_string(),
            ));
        }
        let h = flow.shape()[1];
        let w = flow.shape()[2];
        let n = (h * w) as f32;
        let data = flow.data();
        let dx_sum: f32 = data[..h * w].iter().sum();
        let dy_sum: f32 = data[h * w..].iter().sum();
        Ok(FlowVector::new(dx_sum / n, dy_sum / n))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper: trim flow to exact target dimensions
// ─────────────────────────────────────────────────────────────────────────────

fn trim_to_size(flow: Tensor, target_h: usize, target_w: usize) -> Result<Tensor, NeuralError> {
    let c = flow.shape()[0];
    let src_h = flow.shape()[1];
    let src_w = flow.shape()[2];

    // Fast path: already the right size.
    if src_h == target_h && src_w == target_w {
        return Ok(flow);
    }

    let actual_h = src_h.min(target_h);
    let actual_w = src_w.min(target_w);
    let src = flow.data();
    let mut out = vec![0.0_f32; c * actual_h * actual_w];

    for ch in 0..c {
        for y in 0..actual_h {
            for x in 0..actual_w {
                out[ch * actual_h * actual_w + y * actual_w + x] =
                    src[ch * src_h * src_w + y * src_w + x];
            }
        }
    }

    Tensor::from_data(out, vec![c, actual_h, actual_w])
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    fn make_estimator(d: usize) -> OpticalFlowEstimator {
        OpticalFlowEstimator::new(d).expect("OpticalFlowEstimator::new failed")
    }

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_valid() {
        let est = make_estimator(2);
        assert_eq!(est.d, 2);
        // Decoder input channels should be (2*2+1)^2 = 25.
        assert_eq!(est.dec1.in_channels, 25);
    }

    #[test]
    fn test_new_d_zero_fails() {
        assert!(OpticalFlowEstimator::new(0).is_err());
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    #[test]
    fn test_forward_output_shape() {
        let est = make_estimator(2);
        let fa = Tensor::zeros(vec![3, 32, 32]).expect("zeros");
        let fb = Tensor::zeros(vec![3, 32, 32]).expect("zeros");
        let flow = est.forward(&fa, &fb).expect("forward");
        assert_eq!(flow.shape(), &[2, 32, 32]);
    }

    #[test]
    fn test_forward_zero_flow_for_identical_frames() {
        // With zero weights, both frames encode to zero features → correlation
        // is also zero → flow is zero everywhere.
        let est = make_estimator(2);
        let fa = Tensor::zeros(vec![3, 16, 16]).expect("zeros");
        let fb = Tensor::zeros(vec![3, 16, 16]).expect("zeros");
        let flow = est.forward(&fa, &fb).expect("forward");
        assert!(flow.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_forward_wrong_rank_fails() {
        let est = make_estimator(1);
        let bad = Tensor::zeros(vec![32, 32]).expect("zeros");
        let fb = Tensor::zeros(vec![3, 32, 32]).expect("zeros");
        assert!(est.forward(&bad, &fb).is_err());
    }

    #[test]
    fn test_forward_mismatched_shapes_fails() {
        let est = make_estimator(1);
        let fa = Tensor::zeros(vec![3, 16, 16]).expect("zeros");
        let fb = Tensor::zeros(vec![3, 32, 32]).expect("zeros");
        assert!(est.forward(&fa, &fb).is_err());
    }

    #[test]
    fn test_forward_wrong_channels_fails() {
        let est = make_estimator(1);
        let fa = Tensor::zeros(vec![1, 16, 16]).expect("zeros");
        let fb = Tensor::zeros(vec![1, 16, 16]).expect("zeros");
        assert!(est.forward(&fa, &fb).is_err());
    }

    // ── Correlation volume ────────────────────────────────────────────────────

    #[test]
    fn test_correlation_volume_shape() {
        let fa = Tensor::zeros(vec![8, 4, 4]).expect("zeros");
        let fb = Tensor::zeros(vec![8, 4, 4]).expect("zeros");
        let cv = correlation_volume(&fa, &fb, 2).expect("corr");
        // d=2 → search=5 → 25 channels
        assert_eq!(cv.shape(), &[25, 4, 4]);
    }

    #[test]
    fn test_correlation_zero_features_is_zero() {
        let fa = Tensor::zeros(vec![4, 3, 3]).expect("zeros");
        let fb = Tensor::zeros(vec![4, 3, 3]).expect("zeros");
        let cv = correlation_volume(&fa, &fb, 1).expect("corr");
        assert!(cv.data().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_correlation_shape_mismatch_fails() {
        let fa = Tensor::zeros(vec![4, 3, 3]).expect("zeros");
        let fb = Tensor::zeros(vec![4, 4, 4]).expect("zeros");
        assert!(correlation_volume(&fa, &fb, 1).is_err());
    }

    // ── Bilinear upsampling ───────────────────────────────────────────────────

    #[test]
    fn test_bilinear_upsample_shape() {
        let t = Tensor::zeros(vec![2, 4, 4]).expect("zeros");
        let up = bilinear_upsample(&t, 4).expect("upsample");
        assert_eq!(up.shape(), &[2, 16, 16]);
    }

    #[test]
    fn test_bilinear_upsample_scale_one_is_identity() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let t = Tensor::from_data(data.clone(), vec![1, 3, 4]).expect("from_data");
        let up = bilinear_upsample(&t, 1).expect("upsample");
        assert_eq!(up.data(), t.data());
    }

    // ── FlowVector ────────────────────────────────────────────────────────────

    #[test]
    fn test_flow_vector_magnitude() {
        let fv = FlowVector::new(3.0, 4.0);
        assert!((fv.magnitude() - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_mean_flow_zero_field() {
        let flow = Tensor::zeros(vec![2, 8, 8]).expect("zeros");
        let mean = OpticalFlowEstimator::mean_flow(&flow).expect("mean_flow");
        assert_eq!(mean.dx, 0.0);
        assert_eq!(mean.dy, 0.0);
    }

    #[test]
    fn test_mean_flow_wrong_shape_fails() {
        let t = Tensor::zeros(vec![1, 8, 8]).expect("zeros");
        assert!(OpticalFlowEstimator::mean_flow(&t).is_err());
    }
}
