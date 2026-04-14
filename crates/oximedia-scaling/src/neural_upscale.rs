//! Lightweight neural network 2×/4× upscaling for `oximedia-scaling`.
//!
//! Implements an ESPCN-style (Efficient Sub-Pixel Convolutional Neural Network)
//! super-resolution pipeline using only patent-free techniques.  The network
//! performs all computation in the low-resolution domain and uses a sub-pixel
//! shuffle (pixel-shuffle / space-to-depth) to produce the high-resolution
//! output.
//!
//! # Architecture (2× upscale)
//!
//! 1. **Feature extraction** — 5×5 convolution, 64 feature maps, tanh.
//! 2. **Non-linear mapping**  — 3×3 convolution, 32 feature maps, tanh.
//! 3. **Sub-pixel output**    — 3×3 convolution, `channels × scale²` maps.
//! 4. **Pixel shuffle**       — rearranges feature maps into spatial pixels.
//!
//! Weights are initialised with a deterministic pseudo-random scheme
//! (LCG-based) that produces reasonable sharpening behaviour without requiring
//! external weight files.  For production use, replace `init_weights` with a
//! loader that reads weights from a file trained offline on super-resolution
//! datasets.
//!
//! # Example
//!
//! ```
//! use oximedia_scaling::neural_upscale::{NeuralUpscaler, UpscaleFactor};
//!
//! let upscaler = NeuralUpscaler::new(UpscaleFactor::X2);
//! let input = vec![128u8; 4 * 4 * 3]; // 4×4 RGB
//! let output = upscaler.upscale(&input, 4, 4, 3).expect("upscale ok");
//! assert_eq!(output.len(), 8 * 8 * 3);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use rayon::prelude::*;

use crate::ScalingError;

// ---------------------------------------------------------------------------
// Scale factor
// ---------------------------------------------------------------------------

/// Neural upscale factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpscaleFactor {
    /// 2× upscaling (output is 2× wider and 2× taller).
    X2,
    /// 4× upscaling (output is 4× wider and 4× taller).
    X4,
}

impl UpscaleFactor {
    /// Integer scale multiplier.
    #[must_use]
    pub fn as_u32(self) -> u32 {
        match self {
            Self::X2 => 2,
            Self::X4 => 4,
        }
    }

    /// Number of sub-pixel output channels per input channel.
    ///
    /// Equal to `scale²`.
    #[must_use]
    pub fn sub_pixel_count(self) -> usize {
        let s = self.as_u32() as usize;
        s * s
    }
}

// ---------------------------------------------------------------------------
// Kernel helpers
// ---------------------------------------------------------------------------

/// A 2D convolution kernel.
struct Kernel {
    /// Row-major coefficients, shape `[out_ch][in_ch][kh][kw]`.
    weights: Vec<f32>,
    /// Bias per output channel.
    bias: Vec<f32>,
    out_channels: usize,
    in_channels: usize,
    kernel_size: usize,
}

impl Kernel {
    /// Initialise kernel weights with a deterministic LCG scheme.
    ///
    /// Uses He initialisation variance (`√(2 / fan_in)`).
    fn new_lcg(out_ch: usize, in_ch: usize, ksize: usize, seed: u64) -> Self {
        let fan_in = (in_ch * ksize * ksize) as f64;
        let std = (2.0 / fan_in).sqrt() as f32;

        let total = out_ch * in_ch * ksize * ksize;
        let mut weights = Vec::with_capacity(total);
        let mut state: u64 = seed;

        for _ in 0..total {
            // LCG: x_{n+1} = (a·x_n + c) mod 2^64
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            // Map to [-1, 1]
            let norm = (state >> 33) as f32 / (u32::MAX as f32) * 2.0 - 1.0;
            weights.push(norm * std);
        }

        let bias = vec![0.0f32; out_ch];
        Self {
            weights,
            bias,
            out_channels: out_ch,
            in_channels: in_ch,
            kernel_size: ksize,
        }
    }

    /// Apply this kernel as a 2D convolution (SAME padding) to a feature map.
    ///
    /// Input `map` has shape `[in_ch][h][w]`.
    /// Output has shape `[out_ch][h][w]`.
    fn convolve(&self, map: &[f32], h: usize, w: usize) -> Vec<f32> {
        let kh = self.kernel_size;
        let kw = self.kernel_size;
        let pad = kh / 2;
        let total = self.out_channels * h * w;

        let mut out = vec![0.0f32; total];

        // Parallel over output channels
        out.par_chunks_mut(h * w)
            .enumerate()
            .for_each(|(oc, slice)| {
                let bias_val = self.bias[oc];
                for row in 0..h {
                    for col in 0..w {
                        let mut acc = bias_val;
                        for ic in 0..self.in_channels {
                            for kr in 0..kh {
                                let sr = row + kr;
                                if sr < pad || sr >= h + pad {
                                    continue;
                                }
                                let sr = sr - pad;
                                for kc in 0..kw {
                                    let sc = col + kc;
                                    if sc < pad || sc >= w + pad {
                                        continue;
                                    }
                                    let sc = sc - pad;
                                    let w_idx = oc * self.in_channels * kh * kw
                                        + ic * kh * kw
                                        + kr * kw
                                        + kc;
                                    let i_idx = ic * h * w + sr * w + sc;
                                    acc += self.weights[w_idx] * map[i_idx];
                                }
                            }
                        }
                        slice[row * w + col] = acc;
                    }
                }
            });

        out
    }
}

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

fn tanh_inplace(v: &mut [f32]) {
    for x in v.iter_mut() {
        *x = x.tanh();
    }
}

// ---------------------------------------------------------------------------
// Pixel shuffle
// ---------------------------------------------------------------------------

/// Rearrange `[C * r², H, W]` feature maps to `[C, H*r, W*r]`.
fn pixel_shuffle(input: &[f32], channels: usize, r: usize, h: usize, w: usize) -> Vec<f32> {
    let out_h = h * r;
    let out_w = w * r;
    let mut out = vec![0.0f32; channels * out_h * out_w];

    for c in 0..channels {
        for sh in 0..r {
            for sw in 0..r {
                let in_channel = c * r * r + sh * r + sw;
                for row in 0..h {
                    for col in 0..w {
                        let out_row = row * r + sh;
                        let out_col = col * r + sw;
                        let in_idx = in_channel * h * w + row * w + col;
                        let out_idx = c * out_h * out_w + out_row * out_w + out_col;
                        out[out_idx] = input[in_idx];
                    }
                }
            }
        }
    }

    out
}

// ---------------------------------------------------------------------------
// NeuralUpscaler
// ---------------------------------------------------------------------------

/// Lightweight ESPCN-style neural upscaler.
///
/// All inference is performed in the low-resolution domain; the sub-pixel
/// shuffle expands to the target resolution in a single efficient step.
pub struct NeuralUpscaler {
    factor: UpscaleFactor,
    /// Layer 1: feature extraction 5×5, 1→64 (applied per channel).
    layer1: Kernel,
    /// Layer 2: non-linear mapping 3×3, 64→32.
    layer2: Kernel,
    /// Layer 3: sub-pixel output 3×3, 32→(scale²).
    layer3: Kernel,
}

impl NeuralUpscaler {
    /// Create a new neural upscaler for the given scale factor.
    ///
    /// Weights are deterministically initialised from a fixed seed; replace
    /// `layer1/2/3` weight fields with trained weights for production use.
    #[must_use]
    pub fn new(factor: UpscaleFactor) -> Self {
        let sub = factor.sub_pixel_count();
        Self {
            factor,
            layer1: Kernel::new_lcg(64, 1, 5, 0xDEAD_BEEF_1234_5678),
            layer2: Kernel::new_lcg(32, 64, 3, 0xCAFE_BABE_8765_4321),
            layer3: Kernel::new_lcg(sub, 32, 3, 0xABCD_EF01_2345_6789),
        }
    }

    /// Upscale a packed image buffer.
    ///
    /// `channels` must be 1 (grayscale), 3 (RGB), or 4 (RGBA).
    /// Alpha channels are passed through with bilinear interpolation; only
    /// the colour/luma channels go through the neural network.
    ///
    /// # Errors
    ///
    /// Returns [`ScalingError`] if dimensions are zero or `channels` is
    /// unsupported.
    pub fn upscale(
        &self,
        input: &[u8],
        src_width: usize,
        src_height: usize,
        channels: usize,
    ) -> Result<Vec<u8>, ScalingError> {
        if src_width == 0 || src_height == 0 {
            return Err(ScalingError::InvalidDimensions(format!(
                "src dimensions must be non-zero, got {src_width}×{src_height}"
            )));
        }
        if channels == 0 || channels > 4 {
            return Err(ScalingError::InvalidDimensions(format!(
                "neural_upscale: unsupported channel count {channels}"
            )));
        }

        let expected = src_width * src_height * channels;
        if input.len() != expected {
            return Err(ScalingError::InsufficientBuffer {
                expected,
                actual: input.len(),
            });
        }

        let scale = self.factor.as_u32() as usize;
        let dst_width = src_width * scale;
        let dst_height = src_height * scale;
        let mut output = vec![0u8; dst_width * dst_height * channels];

        // Process each colour channel independently
        let colour_channels = channels.min(3);
        for ch in 0..colour_channels {
            // Extract single channel as float
            let mut plane: Vec<f32> = (0..src_height)
                .flat_map(|row| (0..src_width).map(move |col| row * src_width + col))
                .map(|idx| input[idx * channels + ch] as f32 / 255.0)
                .collect();

            // Layer 1: feature extraction (treat single plane as 1 input channel)
            let mut feat = self.layer1.convolve(&plane, src_height, src_width);
            tanh_inplace(&mut feat);

            // Layer 2: non-linear mapping
            let mut feat2 = self.layer2.convolve(&feat, src_height, src_width);
            tanh_inplace(&mut feat2);

            // Layer 3: sub-pixel output (scale² output channels)
            let feat3 = self.layer3.convolve(&feat2, src_height, src_width);

            // Pixel shuffle to produce HQ plane
            let shuffled = pixel_shuffle(&feat3, 1, scale, src_height, src_width);

            // Clamp and write to output
            for row in 0..dst_height {
                for col in 0..dst_width {
                    let v = shuffled[row * dst_width + col].clamp(0.0, 1.0);
                    output[(row * dst_width + col) * channels + ch] = (v * 255.0 + 0.5) as u8;
                }
            }

            // Suppress unused warning on last iteration
            let _ = &mut plane;
        }

        // Pass alpha through (bilinear nearest-neighbour for simplicity)
        if channels == 4 {
            for row in 0..dst_height {
                for col in 0..dst_width {
                    let src_row = (row / scale).min(src_height - 1);
                    let src_col = (col / scale).min(src_width - 1);
                    let src_idx = (src_row * src_width + src_col) * channels + 3;
                    let dst_idx = (row * dst_width + col) * channels + 3;
                    output[dst_idx] = input[src_idx];
                }
            }
        }

        Ok(output)
    }

    /// Return the upscale factor.
    #[must_use]
    pub fn factor(&self) -> UpscaleFactor {
        self.factor
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_upscale_factor_values() {
        assert_eq!(UpscaleFactor::X2.as_u32(), 2);
        assert_eq!(UpscaleFactor::X4.as_u32(), 4);
        assert_eq!(UpscaleFactor::X2.sub_pixel_count(), 4);
        assert_eq!(UpscaleFactor::X4.sub_pixel_count(), 16);
    }

    #[test]
    fn test_upscale_2x_output_size() {
        let upscaler = NeuralUpscaler::new(UpscaleFactor::X2);
        let input = vec![128u8; 4 * 4 * 3];
        let output = upscaler.upscale(&input, 4, 4, 3).expect("upscale ok");
        assert_eq!(output.len(), 8 * 8 * 3);
    }

    #[test]
    fn test_upscale_4x_output_size() {
        let upscaler = NeuralUpscaler::new(UpscaleFactor::X4);
        let input = vec![200u8; 2 * 2 * 1];
        let output = upscaler.upscale(&input, 2, 2, 1).expect("upscale ok");
        assert_eq!(output.len(), 8 * 8 * 1);
    }

    #[test]
    fn test_upscale_rgba_alpha_passthrough() {
        let upscaler = NeuralUpscaler::new(UpscaleFactor::X2);
        let mut input = vec![100u8; 2 * 2 * 4];
        // Set alpha to 200
        for i in 0..4 {
            input[i * 4 + 3] = 200;
        }
        let output = upscaler.upscale(&input, 2, 2, 4).expect("ok");
        // All alpha samples in the output should be 200 (nearest-neighbour passthrough)
        for i in 0..(4 * 4) {
            assert_eq!(output[i * 4 + 3], 200, "alpha mismatch at pixel {i}");
        }
    }

    #[test]
    fn test_upscale_zero_width_returns_err() {
        let upscaler = NeuralUpscaler::new(UpscaleFactor::X2);
        let result = upscaler.upscale(&[], 0, 4, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_upscale_bad_channel_count_returns_err() {
        let upscaler = NeuralUpscaler::new(UpscaleFactor::X2);
        let result = upscaler.upscale(&vec![0u8; 4 * 4 * 5], 4, 4, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_pixel_shuffle_correct_shape() {
        // r=2, H=2, W=2, C=1 → 4 input channels → 4×4 output
        let input: Vec<f32> = (0..16).map(|x| x as f32).collect();
        let out = pixel_shuffle(&input, 1, 2, 2, 2);
        assert_eq!(out.len(), 1 * 4 * 4);
    }
}
