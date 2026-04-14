//! Multi-threaded Conv2d using Rayon for parallel output-channel computation.
//!
//! The standard [`Conv2dLayer`](crate::layers::Conv2dLayer) computes each output
//! channel sequentially.  For models where `out_channels` is large (e.g. 128,
//! 256, 512) this module provides [`ParallelConv2dLayer`], which distributes the
//! output-channel loop across a Rayon thread pool.
//!
//! ## Algorithm
//!
//! Each output channel `oc` is computed independently:
//!
//! ```text
//! output[oc, oh, ow] = bias[oc]
//!   + Σ_{ic, kh, kw} weight[oc, ic, kh, kw] × input[ic, oh*s + kh - pad, ow*s + kw - pad]
//! ```
//!
//! Since output channels share no mutable state, they can be computed in
//! parallel with zero synchronisation.
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::parallel_conv::ParallelConv2dLayer;
//! use oximedia_neural::tensor::Tensor;
//!
//! let layer = ParallelConv2dLayer::new(3, 16, 3, 3, (1, 1), (1, 1)).unwrap();
//! let input = Tensor::zeros(vec![3, 32, 32]).unwrap();
//! let output = layer.forward(&input).unwrap();
//! assert_eq!(output.shape()[0], 16); // 16 output channels
//! ```

#![allow(dead_code)]

use crate::error::NeuralError;
use crate::tensor::Tensor;
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// ParallelConv2dLayer
// ─────────────────────────────────────────────────────────────────────────────

/// A 2-D convolution layer whose output channels are computed in parallel via
/// Rayon.
///
/// The public fields mirror [`Conv2dLayer`](crate::layers::Conv2dLayer) so that
/// weights can be shared / transferred between the two implementations.
#[derive(Debug, Clone)]
pub struct ParallelConv2dLayer {
    /// Convolution kernels, shape `[out_channels, in_channels, kH, kW]`.
    pub weight: Tensor,
    /// Per-output-channel bias, shape `[out_channels]`.
    pub bias: Tensor,
    /// `(stride_h, stride_w)`.
    pub stride: (usize, usize),
    /// `(pad_h, pad_w)` — zero-padding added to each side.
    pub padding: (usize, usize),
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
}

impl ParallelConv2dLayer {
    /// Creates a zero-initialised `ParallelConv2dLayer`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_h: usize,
        kernel_w: usize,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Result<Self, NeuralError> {
        if in_channels == 0 || out_channels == 0 || kernel_h == 0 || kernel_w == 0 {
            return Err(NeuralError::InvalidShape(
                "ParallelConv2dLayer: all dimensions must be > 0".to_string(),
            ));
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(NeuralError::InvalidShape(
                "ParallelConv2dLayer: stride must be > 0".to_string(),
            ));
        }
        let weight = Tensor::zeros(vec![out_channels, in_channels, kernel_h, kernel_w])?;
        let bias = Tensor::zeros(vec![out_channels])?;
        Ok(Self {
            weight,
            bias,
            stride,
            padding,
            in_channels,
            out_channels,
            kernel_h,
            kernel_w,
        })
    }

    /// Constructs a `ParallelConv2dLayer` from an existing
    /// [`Conv2dLayer`](crate::layers::Conv2dLayer) by cloning its weights.
    pub fn from_conv2d(conv: &crate::layers::Conv2dLayer) -> Self {
        Self {
            weight: conv.weight.clone(),
            bias: conv.bias.clone(),
            stride: conv.stride,
            padding: conv.padding,
            in_channels: conv.in_channels,
            out_channels: conv.out_channels,
            kernel_h: conv.kernel_h,
            kernel_w: conv.kernel_w,
        }
    }

    /// Runs the parallel forward pass on a `[C, H, W]` input tensor.
    ///
    /// Returns `[out_channels, out_H, out_W]`.
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, NeuralError> {
        if input.ndim() != 3 {
            return Err(NeuralError::InvalidShape(format!(
                "ParallelConv2dLayer::forward: expected [C, H, W], got rank {}",
                input.ndim()
            )));
        }
        if input.shape()[0] != self.in_channels {
            return Err(NeuralError::ShapeMismatch(format!(
                "ParallelConv2dLayer::forward: expected {} input channels, got {}",
                self.in_channels,
                input.shape()[0]
            )));
        }

        let in_h = input.shape()[1];
        let in_w = input.shape()[2];

        let out_h = (in_h + 2 * self.padding.0 - self.kernel_h) / self.stride.0 + 1;
        let out_w = (in_w + 2 * self.padding.1 - self.kernel_w) / self.stride.1 + 1;

        if out_h == 0 || out_w == 0 {
            return Err(NeuralError::InvalidShape(
                "ParallelConv2dLayer::forward: output spatial size would be 0".to_string(),
            ));
        }

        let in_data = input.data();
        let w_data = self.weight.data();
        let b_data = self.bias.data();
        let ic = self.in_channels;
        let kh = self.kernel_h;
        let kw = self.kernel_w;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let oc_count = self.out_channels;
        let kernel_size_per_oc = ic * kh * kw;

        // Each output channel is independent — compute in parallel.
        let channel_outputs: Vec<Vec<f32>> = (0..oc_count)
            .into_par_iter()
            .map(|oc| {
                let bias_val = b_data[oc];
                let w_base = oc * kernel_size_per_oc;
                let mut channel_buf = vec![0.0_f32; out_h * out_w];

                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = bias_val;
                        for ic_i in 0..ic {
                            for kh_i in 0..kh {
                                let ih = oh * sh + kh_i;
                                let ih = if ih >= ph { ih - ph } else { continue };
                                if ih >= in_h {
                                    continue;
                                }
                                for kw_i in 0..kw {
                                    let iw = ow * sw + kw_i;
                                    let iw = if iw >= pw { iw - pw } else { continue };
                                    if iw >= in_w {
                                        continue;
                                    }
                                    let w_idx = w_base + ic_i * kh * kw + kh_i * kw + kw_i;
                                    let in_idx = ic_i * in_h * in_w + ih * in_w + iw;
                                    sum += w_data[w_idx] * in_data[in_idx];
                                }
                            }
                        }
                        channel_buf[oh * out_w + ow] = sum;
                    }
                }

                channel_buf
            })
            .collect();

        // Flatten channel outputs into a single contiguous buffer.
        let total = oc_count * out_h * out_w;
        let mut out_data = Vec::with_capacity(total);
        for ch in channel_outputs {
            out_data.extend_from_slice(&ch);
        }

        Tensor::from_data(out_data, vec![oc_count, out_h, out_w])
    }

    /// Returns the output spatial dimensions for a given input size.
    pub fn output_size(&self, in_h: usize, in_w: usize) -> (usize, usize) {
        let out_h = (in_h + 2 * self.padding.0 - self.kernel_h) / self.stride.0 + 1;
        let out_w = (in_w + 2 * self.padding.1 - self.kernel_w) / self.stride.1 + 1;
        (out_h, out_w)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::Conv2dLayer;
    use crate::tensor::Tensor;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_new_valid() {
        let layer = ParallelConv2dLayer::new(3, 16, 3, 3, (1, 1), (1, 1));
        assert!(layer.is_ok());
        let layer = layer.expect("new");
        assert_eq!(layer.in_channels, 3);
        assert_eq!(layer.out_channels, 16);
    }

    #[test]
    fn test_new_zero_channels_fails() {
        assert!(ParallelConv2dLayer::new(0, 16, 3, 3, (1, 1), (1, 1)).is_err());
        assert!(ParallelConv2dLayer::new(3, 0, 3, 3, (1, 1), (1, 1)).is_err());
    }

    #[test]
    fn test_new_zero_kernel_fails() {
        assert!(ParallelConv2dLayer::new(3, 16, 0, 3, (1, 1), (1, 1)).is_err());
    }

    #[test]
    fn test_new_zero_stride_fails() {
        assert!(ParallelConv2dLayer::new(3, 16, 3, 3, (0, 1), (1, 1)).is_err());
    }

    // ── Forward pass ─────────────────────────────────────────────────────────

    #[test]
    fn test_forward_output_shape_no_padding() {
        let layer = ParallelConv2dLayer::new(1, 4, 3, 3, (1, 1), (0, 0)).expect("new");
        let input = Tensor::ones(vec![1, 8, 8]).expect("ones");
        let out = layer.forward(&input).expect("forward");
        assert_eq!(out.shape(), &[4, 6, 6]);
    }

    #[test]
    fn test_forward_output_shape_same_padding() {
        let layer = ParallelConv2dLayer::new(3, 8, 3, 3, (1, 1), (1, 1)).expect("new");
        let input = Tensor::zeros(vec![3, 16, 16]).expect("zeros");
        let out = layer.forward(&input).expect("forward");
        assert_eq!(out.shape(), &[8, 16, 16]);
    }

    #[test]
    fn test_forward_stride_2() {
        let layer = ParallelConv2dLayer::new(1, 2, 3, 3, (2, 2), (1, 1)).expect("new");
        let input = Tensor::zeros(vec![1, 8, 8]).expect("zeros");
        let out = layer.forward(&input).expect("forward");
        assert_eq!(out.shape(), &[2, 4, 4]);
    }

    #[test]
    fn test_forward_zero_weights_output_equals_bias() {
        let mut layer = ParallelConv2dLayer::new(1, 2, 1, 1, (1, 1), (0, 0)).expect("new");
        // Set biases to known values.
        layer.bias = Tensor::from_data(vec![3.0, -1.5], vec![2]).expect("bias");
        let input = Tensor::zeros(vec![1, 4, 4]).expect("zeros");
        let out = layer.forward(&input).expect("forward");
        // With zero weights each output pixel equals the bias.
        let od = out.data();
        for i in 0..16 {
            assert!((od[i] - 3.0).abs() < 1e-6);
        }
        for i in 16..32 {
            assert!((od[i] - (-1.5)).abs() < 1e-6);
        }
    }

    #[test]
    fn test_forward_wrong_rank_fails() {
        let layer = ParallelConv2dLayer::new(3, 4, 3, 3, (1, 1), (1, 1)).expect("new");
        let bad = Tensor::zeros(vec![3, 8]).expect("zeros");
        assert!(layer.forward(&bad).is_err());
    }

    #[test]
    fn test_forward_wrong_channels_fails() {
        let layer = ParallelConv2dLayer::new(3, 4, 3, 3, (1, 1), (1, 1)).expect("new");
        let bad = Tensor::zeros(vec![1, 8, 8]).expect("zeros");
        assert!(layer.forward(&bad).is_err());
    }

    // ── Matches sequential Conv2dLayer ────────────────────────────────────────

    #[test]
    fn test_matches_sequential_conv2d() {
        // Build a sequential Conv2dLayer with known weights.
        let mut seq = Conv2dLayer::new(2, 4, 3, 3, (1, 1), (1, 1)).expect("seq");
        let w_data: Vec<f32> = (0..(4 * 2 * 3 * 3)).map(|i| (i as f32) * 0.01).collect();
        seq.weight = Tensor::from_data(w_data, vec![4, 2, 3, 3]).expect("w");
        let b_data = vec![0.1_f32, -0.2, 0.3, -0.4];
        seq.bias = Tensor::from_data(b_data, vec![4]).expect("b");

        // Create the parallel layer from it.
        let par = ParallelConv2dLayer::from_conv2d(&seq);

        let input_data: Vec<f32> = (0..(2 * 6 * 6)).map(|i| (i as f32) * 0.005).collect();
        let input = Tensor::from_data(input_data, vec![2, 6, 6]).expect("input");

        let out_seq = seq.forward(&input).expect("seq forward");
        let out_par = par.forward(&input).expect("par forward");

        assert_eq!(out_seq.shape(), out_par.shape());
        for (a, b) in out_seq.data().iter().zip(out_par.data().iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "mismatch: sequential={}, parallel={}",
                a,
                b
            );
        }
    }

    // ── output_size helper ───────────────────────────────────────────────────

    #[test]
    fn test_output_size() {
        let layer = ParallelConv2dLayer::new(1, 1, 3, 3, (2, 2), (1, 1)).expect("new");
        let (oh, ow) = layer.output_size(8, 10);
        assert_eq!(oh, 4);
        assert_eq!(ow, 5);
    }
}
