//! Deformable convolution for adaptive spatial sampling in object detection.
//!
//! # Background
//!
//! Standard convolution samples input values on a regular grid.  Deformable
//! convolution (Dai et al., 2017) learns an additional offset field that shifts
//! each sampling location, allowing the network to focus on relevant regions.
//!
//! For each output position `(oh, ow)` and kernel position `(kh_i, kw_i)`:
//!
//! ```text
//! sampling position = (oh * stride_h + kh_i - pad_h + Δh,
//!                      ow * stride_w + kw_i - pad_w + Δw)
//! ```
//!
//! where `Δh` and `Δw` are predicted by the offset network.
//!
//! Non-integer sampling locations are handled via bilinear interpolation on the
//! input feature map.
//!
//! # Modulated Deformable Convolution (DCNv2)
//!
//! An extended version multiplies each sample by a learned attention mask `m ∈
//! [0, 1]` before accumulation.  Enable with `modulated = true`.
//!
//! # Tensor shapes
//!
//! | Tensor | Shape |
//! |--------|-------|
//! | input | `[C_in, H, W]` |
//! | offsets | `[2*kH*kW, out_H, out_W]` (Δh then Δw per kernel position) |
//! | masks (modulated only) | `[kH*kW, out_H, out_W]` |
//! | output | `[C_out, out_H, out_W]` |

use crate::error::NeuralError;
use crate::tensor::Tensor;

// ──────────────────────────────────────────────────────────────────────────────
// DeformableConv2d
// ──────────────────────────────────────────────────────────────────────────────

/// Deformable 2-D convolution with learned per-kernel-position offsets.
///
/// Unlike standard [`Conv2dLayer`](crate::layers::Conv2dLayer), the sampling
/// locations are displaced by `offsets` at inference time.
///
/// For modulated deformable convolution set `modulated = true` and supply a
/// mask tensor to `forward_modulated`.
#[derive(Debug, Clone)]
pub struct DeformableConv2d {
    /// Convolution kernels, shape `[out_channels, in_channels, kH, kW]`.
    pub weight: Tensor,
    /// Per-output-channel bias, shape `[out_channels]`.
    pub bias: Tensor,
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// (stride_h, stride_w).
    pub stride: (usize, usize),
    /// (pad_h, pad_w).
    pub padding: (usize, usize),
    /// Whether to support modulated deformable convolution (DCNv2).
    pub modulated: bool,
}

impl DeformableConv2d {
    /// Creates a zero-initialized `DeformableConv2d`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: (usize, usize),
        padding: (usize, usize),
        modulated: bool,
    ) -> Result<Self, NeuralError> {
        if in_channels == 0 || out_channels == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NeuralError::InvalidShape(
                "DeformableConv2d: all channel/kernel sizes must be > 0".to_string(),
            ));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NeuralError::InvalidShape(
                "DeformableConv2d: stride must be > 0".to_string(),
            ));
        }
        let weight = Tensor::zeros(vec![out_channels, in_channels, kernel_h, kernel_w])?;
        let bias = Tensor::zeros(vec![out_channels])?;
        Ok(Self {
            weight,
            bias,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            modulated,
        })
    }

    // ── Helper: output spatial size ──────────────────────────────────────────

    /// Computes the output spatial dimensions given input spatial dimensions.
    pub fn output_size(&self, in_h: usize, in_w: usize) -> (usize, usize) {
        let (pad_h, pad_w) = self.padding;
        let (sh, sw) = self.stride;
        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;
        let out_h = if padded_h >= self.kernel_h {
            (padded_h - self.kernel_h) / sh + 1
        } else {
            0
        };
        let out_w = if padded_w >= self.kernel_w {
            (padded_w - self.kernel_w) / sw + 1
        } else {
            0
        };
        (out_h, out_w)
    }

    // ── Offset tensor validation ──────────────────────────────────────────────

    fn validate_offsets(
        &self,
        offsets: &Tensor,
        out_h: usize,
        out_w: usize,
    ) -> Result<(), NeuralError> {
        let expected_c = 2 * self.kernel_h * self.kernel_w;
        if offsets.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "DeformableConv2d: offsets must be rank 3, got {}",
                offsets.ndim()
            )));
        }
        if offsets.shape()[0] != expected_c {
            return Err(NeuralError::ShapeMismatch(format!(
                "DeformableConv2d: offsets channel dim {} != 2*kH*kW={}",
                offsets.shape()[0],
                expected_c
            )));
        }
        if offsets.shape()[1] != out_h || offsets.shape()[2] != out_w {
            return Err(NeuralError::ShapeMismatch(format!(
                "DeformableConv2d: offsets spatial {}×{} != output {}×{}",
                offsets.shape()[1],
                offsets.shape()[2],
                out_h,
                out_w
            )));
        }
        Ok(())
    }

    fn validate_masks(
        &self,
        masks: &Tensor,
        out_h: usize,
        out_w: usize,
    ) -> Result<(), NeuralError> {
        let expected_c = self.kernel_h * self.kernel_w;
        if masks.ndim() != 3 {
            return Err(NeuralError::InvalidShape(
                "DeformableConv2d: masks must be rank 3".to_string(),
            ));
        }
        if masks.shape()[0] != expected_c {
            return Err(NeuralError::ShapeMismatch(format!(
                "DeformableConv2d: masks channel dim {} != kH*kW={}",
                masks.shape()[0],
                expected_c
            )));
        }
        if masks.shape()[1] != out_h || masks.shape()[2] != out_w {
            return Err(NeuralError::ShapeMismatch(format!(
                "DeformableConv2d: masks spatial {}×{} != output {}×{}",
                masks.shape()[1],
                masks.shape()[2],
                out_h,
                out_w
            )));
        }
        Ok(())
    }

    // ── Core sampling ─────────────────────────────────────────────────────────

    /// Bilinear interpolation on a single-channel padded feature map.
    ///
    /// `ph` and `pw` are fractional row/column indices in the padded map
    /// (dimensions `[padded_h, padded_w]`).
    /// Returns 0 when the coordinates fall outside the padded region.
    fn bilinear_sample(feat: &[f32], padded_h: usize, padded_w: usize, ph: f32, pw: f32) -> f32 {
        if ph < 0.0 || pw < 0.0 || ph > (padded_h - 1) as f32 || pw > (padded_w - 1) as f32 {
            return 0.0;
        }
        let h0 = ph.floor() as usize;
        let w0 = pw.floor() as usize;
        let h1 = (h0 + 1).min(padded_h - 1);
        let w1 = (w0 + 1).min(padded_w - 1);
        let dh = ph - h0 as f32;
        let dw = pw - w0 as f32;

        let v00 = feat[h0 * padded_w + w0];
        let v01 = feat[h0 * padded_w + w1];
        let v10 = feat[h1 * padded_w + w0];
        let v11 = feat[h1 * padded_w + w1];

        v00 * (1.0 - dh) * (1.0 - dw)
            + v01 * (1.0 - dh) * dw
            + v10 * dh * (1.0 - dw)
            + v11 * dh * dw
    }

    /// Deformable convolution forward without modulation.
    ///
    /// `offsets` shape: `[2*kH*kW, out_H, out_W]`.
    /// Offset channels are ordered as `[Δh_00, Δh_01, …, Δw_00, Δw_01, …]`
    /// (all Δh channels first, then all Δw channels).
    pub fn forward(&self, input: &Tensor, offsets: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "DeformableConv2d::forward: expected [C,H,W] input, got rank {}",
                input.ndim()
            )));
        }
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if in_c != self.in_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "DeformableConv2d: input channels {} != in_channels {}",
                in_c, self.in_channels
            )));
        }

        let (out_h, out_w) = self.output_size(in_h, in_w);
        if out_h == 0 || out_w == 0 {
            return Err(NeuralError::InvalidShape(
                "DeformableConv2d::forward: output spatial size is zero".to_string(),
            ));
        }
        self.validate_offsets(offsets, out_h, out_w)?;

        let (pad_h, pad_w) = self.padding;
        let (sh, sw) = self.stride;
        let (kh, kw) = (self.kernel_h, self.kernel_w);
        let ksize = kh * kw;

        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;

        // Build padded input: [C, padded_H, padded_W].
        let mut padded = vec![0.0_f32; in_c * padded_h * padded_w];
        for c in 0..in_c {
            for h in 0..in_h {
                for w in 0..in_w {
                    let dst = c * padded_h * padded_w + (h + pad_h) * padded_w + (w + pad_w);
                    padded[dst] = input.data()[c * in_h * in_w + h * in_w + w];
                }
            }
        }

        let mut out_data = vec![0.0_f32; self.out_channels * out_h * out_w];

        for oc in 0..self.out_channels {
            let bias_val = self.bias.data()[oc];
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias_val;
                    for k in 0..ksize {
                        let kh_i = k / kw;
                        let kw_i = k % kw;

                        // Base sampling position.
                        let base_ph = (oh * sh + kh_i) as f32 - pad_h as f32;
                        let base_pw = (ow * sw + kw_i) as f32 - pad_w as f32;

                        // Apply offset (relative to padded coordinate system).
                        let off_h_idx = k * out_h * out_w + oh * out_w + ow;
                        let off_w_idx = (ksize + k) * out_h * out_w + oh * out_w + ow;
                        let delta_h = offsets.data()[off_h_idx];
                        let delta_w = offsets.data()[off_w_idx];

                        // Convert back to padded coordinates.
                        let sample_ph = base_ph + delta_h + pad_h as f32;
                        let sample_pw = base_pw + delta_w + pad_w as f32;

                        for ic in 0..in_c {
                            let feat_slice =
                                &padded[ic * padded_h * padded_w..(ic + 1) * padded_h * padded_w];
                            let val = Self::bilinear_sample(
                                feat_slice, padded_h, padded_w, sample_ph, sample_pw,
                            );
                            let w_idx = oc * (in_c * kh * kw) + ic * ksize + k;
                            acc += val * self.weight.data()[w_idx];
                        }
                    }
                    out_data[oc * out_h * out_w + oh * out_w + ow] = acc;
                }
            }
        }

        Tensor::from_data(out_data, vec![self.out_channels, out_h, out_w])
    }

    /// Modulated deformable convolution (DCNv2) forward pass.
    ///
    /// `offsets` shape: `[2*kH*kW, out_H, out_W]`.
    /// `masks` shape: `[kH*kW, out_H, out_W]` — attention weights applied
    /// before accumulation (sigmoid activation recommended but not enforced).
    pub fn forward_modulated(
        &self,
        input: &Tensor,
        offsets: &Tensor,
        masks: &Tensor,
    ) -> Result<Tensor, NeuralError> {
        if !self.modulated {
            return Err(NeuralError::InvalidShape(
                "DeformableConv2d::forward_modulated: layer was not created with modulated=true"
                    .to_string(),
            ));
        }
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "DeformableConv2d::forward_modulated: expected [C,H,W] input, got rank {}",
                input.ndim()
            )));
        }
        let (in_c, in_h, in_w) = (input.shape()[0], input.shape()[1], input.shape()[2]);
        if in_c != self.in_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "DeformableConv2d: input channels {} != in_channels {}",
                in_c, self.in_channels
            )));
        }

        let (out_h, out_w) = self.output_size(in_h, in_w);
        if out_h == 0 || out_w == 0 {
            return Err(NeuralError::InvalidShape(
                "DeformableConv2d::forward_modulated: output spatial size is zero".to_string(),
            ));
        }
        self.validate_offsets(offsets, out_h, out_w)?;
        self.validate_masks(masks, out_h, out_w)?;

        let (pad_h, pad_w) = self.padding;
        let (sh, sw) = self.stride;
        let (kh, kw) = (self.kernel_h, self.kernel_w);
        let ksize = kh * kw;

        let padded_h = in_h + 2 * pad_h;
        let padded_w = in_w + 2 * pad_w;

        let mut padded = vec![0.0_f32; in_c * padded_h * padded_w];
        for c in 0..in_c {
            for h in 0..in_h {
                for w in 0..in_w {
                    let dst = c * padded_h * padded_w + (h + pad_h) * padded_w + (w + pad_w);
                    padded[dst] = input.data()[c * in_h * in_w + h * in_w + w];
                }
            }
        }

        let mut out_data = vec![0.0_f32; self.out_channels * out_h * out_w];

        for oc in 0..self.out_channels {
            let bias_val = self.bias.data()[oc];
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias_val;
                    for k in 0..ksize {
                        let kh_i = k / kw;
                        let kw_i = k % kw;

                        let base_ph = (oh * sh + kh_i) as f32 - pad_h as f32;
                        let base_pw = (ow * sw + kw_i) as f32 - pad_w as f32;

                        let off_h_idx = k * out_h * out_w + oh * out_w + ow;
                        let off_w_idx = (ksize + k) * out_h * out_w + oh * out_w + ow;
                        let delta_h = offsets.data()[off_h_idx];
                        let delta_w = offsets.data()[off_w_idx];

                        let sample_ph = base_ph + delta_h + pad_h as f32;
                        let sample_pw = base_pw + delta_w + pad_w as f32;

                        let mask_val = masks.data()[k * out_h * out_w + oh * out_w + ow];

                        for ic in 0..in_c {
                            let feat_slice =
                                &padded[ic * padded_h * padded_w..(ic + 1) * padded_h * padded_w];
                            let val = Self::bilinear_sample(
                                feat_slice, padded_h, padded_w, sample_ph, sample_pw,
                            );
                            let w_idx = oc * (in_c * kh * kw) + ic * ksize + k;
                            acc += val * self.weight.data()[w_idx] * mask_val;
                        }
                    }
                    out_data[oc * out_h * out_w + oh * out_w + ow] = acc;
                }
            }
        }

        Tensor::from_data(out_data, vec![self.out_channels, out_h, out_w])
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() < 1e-4
    }

    /// Zero offsets should behave identically to standard convolution.
    #[test]
    fn test_zero_offsets_equals_standard_conv() {
        let mut layer = DeformableConv2d::new(1, 1, 3, 3, (1, 1), (1, 1), false)
            .expect("deformable conv2d new");
        // Identity kernel: only center pixel (kh=1, kw=1) = 1.0.
        layer.weight.data_mut()[4] = 1.0;

        let input = Tensor::ones(vec![1, 5, 5]).expect("tensor ones");
        let offsets = Tensor::zeros(vec![18, 5, 5]).expect("tensor zeros"); // 2*9=18 channels, zero deltas

        let out = layer.forward(&input, &offsets).expect("forward pass");
        assert_eq!(out.shape(), &[1, 5, 5]);
        // Identity kernel with zero offset → all 1.0.
        assert!(out.data().iter().all(|&v| close(v, 1.0)));
    }

    #[test]
    fn test_output_shape_no_pad() {
        let layer = DeformableConv2d::new(2, 4, 3, 3, (1, 1), (0, 0), false)
            .expect("deformable conv2d new");
        let input = Tensor::ones(vec![2, 7, 7]).expect("tensor ones");
        let (out_h, out_w) = layer.output_size(7, 7);
        assert_eq!((out_h, out_w), (5, 5));
        let offsets = Tensor::zeros(vec![18, out_h, out_w]).expect("tensor zeros");
        let out = layer.forward(&input, &offsets).expect("forward pass");
        assert_eq!(out.shape(), &[4, 5, 5]);
    }

    #[test]
    fn test_bias_only() {
        let mut layer = DeformableConv2d::new(1, 2, 1, 1, (1, 1), (0, 0), false)
            .expect("deformable conv2d new");
        layer.bias.data_mut()[0] = 3.0;
        layer.bias.data_mut()[1] = 7.0;
        let input = Tensor::zeros(vec![1, 4, 4]).expect("tensor zeros");
        let offsets = Tensor::zeros(vec![2, 4, 4]).expect("tensor zeros");
        let out = layer.forward(&input, &offsets).expect("forward pass");
        assert_eq!(out.shape(), &[2, 4, 4]);
        assert!(out.data()[..16].iter().all(|&v| close(v, 3.0)));
        assert!(out.data()[16..].iter().all(|&v| close(v, 7.0)));
    }

    #[test]
    fn test_offsets_shift_sampling() {
        // 1-channel 3×3 input, 1×1 kernel (ksize=1), stride=1, no pad.
        // A +1 row offset shifts the sampling point down by 1 row.
        let mut layer = DeformableConv2d::new(1, 1, 1, 1, (1, 1), (0, 0), false)
            .expect("deformable conv2d new");
        layer.weight.data_mut()[0] = 1.0; // passthrough

        // Input: row 0 = 0.0, row 1 = 1.0, row 2 = 2.0
        let input_data = vec![
            0.0, 0.0, 0.0, // row 0
            1.0, 1.0, 1.0, // row 1
            2.0, 2.0, 2.0, // row 2
        ];
        let input = Tensor::from_data(input_data, vec![1, 3, 3]).expect("tensor from_data");

        // Offsets: 2 channels (Δh, Δw), each 3×3.
        // All Δh = +1, all Δw = 0 → sample 1 row below.
        let mut offset_data = vec![0.0_f32; 2 * 3 * 3];
        for i in 0..9 {
            offset_data[i] = 1.0; // Δh = +1
        }
        let offsets = Tensor::from_data(offset_data, vec![2, 3, 3]).expect("tensor from_data");

        let out = layer.forward(&input, &offsets).expect("forward pass");
        assert_eq!(out.shape(), &[1, 3, 3]);

        // Row 0 of output: samples row 1 → value 1.0
        assert!(close(out.data()[0], 1.0), "got {}", out.data()[0]);
        // Row 1 of output: samples row 2 → value 2.0
        assert!(close(out.data()[3], 2.0), "got {}", out.data()[3]);
        // Row 2 of output: samples row 3 (out of bounds) → 0.0
        assert!(close(out.data()[6], 0.0), "got {}", out.data()[6]);
    }

    #[test]
    fn test_modulated_zero_mask_gives_bias_only() {
        let mut layer =
            DeformableConv2d::new(1, 1, 3, 3, (1, 1), (1, 1), true).expect("deformable conv2d new");
        layer.weight.data_mut()[4] = 1.0;
        layer.bias.data_mut()[0] = 5.0;

        let input = Tensor::ones(vec![1, 4, 4]).expect("tensor ones");
        let offsets = Tensor::zeros(vec![18, 4, 4]).expect("tensor zeros");
        let masks = Tensor::zeros(vec![9, 4, 4]).expect("tensor zeros"); // all mask = 0

        let out = layer
            .forward_modulated(&input, &offsets, &masks)
            .expect("forward_modulated");
        assert_eq!(out.shape(), &[1, 4, 4]);
        // Zero mask → no input contribution → all output = bias = 5.0
        assert!(out.data().iter().all(|&v| close(v, 5.0)));
    }

    #[test]
    fn test_modulated_unit_mask_equals_standard() {
        let mut layer =
            DeformableConv2d::new(1, 1, 3, 3, (1, 1), (1, 1), true).expect("deformable conv2d new");
        layer.weight.data_mut()[4] = 1.0; // center kernel = 1, others = 0

        let input = Tensor::ones(vec![1, 5, 5]).expect("tensor ones");
        let offsets = Tensor::zeros(vec![18, 5, 5]).expect("tensor zeros");
        let masks = Tensor::ones(vec![9, 5, 5]).expect("tensor ones"); // all mask = 1

        let out = layer
            .forward_modulated(&input, &offsets, &masks)
            .expect("forward_modulated");
        assert_eq!(out.shape(), &[1, 5, 5]);
        // Unit mask + identity kernel → all 1.0
        assert!(out.data().iter().all(|&v| close(v, 1.0)));
    }

    #[test]
    fn test_invalid_offset_shape() {
        let layer = DeformableConv2d::new(1, 1, 3, 3, (1, 1), (1, 1), false)
            .expect("deformable conv2d new");
        let input = Tensor::ones(vec![1, 5, 5]).expect("tensor ones");
        // Wrong channel count.
        let offsets = Tensor::zeros(vec![9, 5, 5]).expect("tensor zeros"); // should be 18
        assert!(layer.forward(&input, &offsets).is_err());
    }

    #[test]
    fn test_channel_mismatch_error() {
        let layer = DeformableConv2d::new(3, 1, 3, 3, (1, 1), (0, 0), false)
            .expect("deformable conv2d new");
        let input = Tensor::ones(vec![1, 5, 5]).expect("tensor ones"); // wrong channels
        let offsets = Tensor::zeros(vec![18, 3, 3]).expect("tensor zeros");
        assert!(layer.forward(&input, &offsets).is_err());
    }

    #[test]
    fn test_forward_modulated_on_non_modulated_layer_errors() {
        let layer = DeformableConv2d::new(1, 1, 3, 3, (1, 1), (1, 1), false)
            .expect("deformable conv2d new");
        let input = Tensor::ones(vec![1, 4, 4]).expect("tensor ones");
        let offsets = Tensor::zeros(vec![18, 4, 4]).expect("tensor zeros");
        let masks = Tensor::ones(vec![9, 4, 4]).expect("tensor ones");
        assert!(layer.forward_modulated(&input, &offsets, &masks).is_err());
    }
}
