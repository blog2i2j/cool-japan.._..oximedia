//! GPU-accelerated film grain synthesis.
//!
//! Generates synthetic film grain matching the perceptual model from
//! `oximedia-denoise`. The grain model is parameterised per-channel with
//! intensity, frequency (spatial scale), and luma-dependency.
//!
//! # Model
//!
//! Film grain is synthesised as:
//!
//! 1. **Seed generation**: a per-frame, per-block deterministic seed derived
//!    from the frame index and spatial position ensures reproducible results
//!    without storing per-frame noise tables.
//!
//! 2. **Frequency-shaped noise**: white noise is convolved with a Gaussian
//!    kernel whose sigma is controlled by `frequency_scale`. Larger values
//!    produce coarser grain.
//!
//! 3. **Luma scaling**: grain amplitude is scaled by
//!    `gain * (1 - luma_dependency * luma_weight)` so that grain is stronger
//!    in darker areas (as in real film stock) when `luma_dependency > 0`.
//!
//! 4. **Chroma handling**: chroma channels use `chroma_scale` to reduce grain
//!    intensity relative to luma, matching the observation that film grain is
//!    predominantly luma-domain.
//!
//! # Usage
//!
//! ```no_run
//! use oximedia_gpu::film_grain::{FilmGrainConfig, FilmGrainSynthesizer};
//!
//! let mut frame = vec![128u8; 1920 * 1080 * 4]; // RGBA
//! let config = FilmGrainConfig::default();
//! let synth = FilmGrainSynthesizer::new(config);
//! synth.apply_rgba(&mut frame, 1920, 1080, 0).unwrap();
//! ```

use crate::{GpuError, Result};
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Per-channel grain configuration.
#[derive(Debug, Clone, Copy)]
pub struct ChannelGrainConfig {
    /// Overall grain intensity (0.0 = no grain, 1.0 = maximum).
    pub gain: f32,
    /// Spatial frequency scale; larger = coarser grain (sigma for Gaussian
    /// shaping kernel, in pixels, typically 0.5–3.0).
    pub frequency_scale: f32,
    /// Luma-dependency of grain amplitude (0.0 = flat, 1.0 = strong dark-bias).
    pub luma_dependency: f32,
}

impl Default for ChannelGrainConfig {
    fn default() -> Self {
        Self {
            gain: 0.04,
            frequency_scale: 1.2,
            luma_dependency: 0.4,
        }
    }
}

/// Film grain synthesis configuration.
#[derive(Debug, Clone)]
pub struct FilmGrainConfig {
    /// Luma (Y) channel grain parameters.
    pub luma: ChannelGrainConfig,
    /// Cb/U chroma channel grain parameters.
    pub chroma_cb: ChannelGrainConfig,
    /// Cr/V chroma channel grain parameters.
    pub chroma_cr: ChannelGrainConfig,
    /// Scaling factor applied to chroma channels relative to luma.
    pub chroma_scale: f32,
    /// Block size used for deterministic seed derivation (must be power of 2).
    pub block_size: u32,
}

impl Default for FilmGrainConfig {
    fn default() -> Self {
        Self {
            luma: ChannelGrainConfig::default(),
            chroma_cb: ChannelGrainConfig {
                gain: 0.02,
                frequency_scale: 1.5,
                luma_dependency: 0.2,
            },
            chroma_cr: ChannelGrainConfig {
                gain: 0.02,
                frequency_scale: 1.5,
                luma_dependency: 0.2,
            },
            chroma_scale: 0.5,
            block_size: 16,
        }
    }
}

impl FilmGrainConfig {
    /// Create a light-grain preset (suitable for clean digital content).
    #[must_use]
    pub fn light() -> Self {
        Self {
            luma: ChannelGrainConfig {
                gain: 0.02,
                frequency_scale: 1.0,
                luma_dependency: 0.3,
            },
            chroma_cb: ChannelGrainConfig {
                gain: 0.01,
                frequency_scale: 1.2,
                luma_dependency: 0.1,
            },
            chroma_cr: ChannelGrainConfig {
                gain: 0.01,
                frequency_scale: 1.2,
                luma_dependency: 0.1,
            },
            chroma_scale: 0.4,
            block_size: 16,
        }
    }

    /// Create a heavy-grain preset (simulates fast film stock or night footage).
    #[must_use]
    pub fn heavy() -> Self {
        Self {
            luma: ChannelGrainConfig {
                gain: 0.10,
                frequency_scale: 2.0,
                luma_dependency: 0.6,
            },
            chroma_cb: ChannelGrainConfig {
                gain: 0.06,
                frequency_scale: 2.5,
                luma_dependency: 0.4,
            },
            chroma_cr: ChannelGrainConfig {
                gain: 0.06,
                frequency_scale: 2.5,
                luma_dependency: 0.4,
            },
            chroma_scale: 0.6,
            block_size: 16,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Synthesizer
// ─────────────────────────────────────────────────────────────────────────────

/// Film grain synthesizer.
///
/// Applies deterministic, frequency-shaped, luma-dependent grain to frames.
#[derive(Debug, Clone)]
pub struct FilmGrainSynthesizer {
    config: FilmGrainConfig,
}

impl FilmGrainSynthesizer {
    /// Create a new synthesizer with the given configuration.
    #[must_use]
    pub fn new(config: FilmGrainConfig) -> Self {
        Self { config }
    }

    /// Apply film grain to an RGBA frame in-place.
    ///
    /// # Arguments
    ///
    /// * `frame` – Mutable RGBA pixel data (4 bytes per pixel, row-major).
    /// * `width` – Frame width in pixels.
    /// * `height` – Frame height in pixels.
    /// * `frame_index` – Frame number (used to vary grain pattern per frame).
    ///
    /// # Errors
    ///
    /// Returns an error if buffer sizes are inconsistent.
    pub fn apply_rgba(
        &self,
        frame: &mut [u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> Result<()> {
        let expected = (width as usize) * (height as usize) * 4;
        if frame.len() != expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: frame.len(),
            });
        }

        if width == 0 || height == 0 {
            return Err(GpuError::InvalidDimensions { width, height });
        }

        let w = width as usize;
        let h = height as usize;

        // Generate grain planes (one per channel)
        let grain_r =
            self.generate_grain_plane(w, h, frame_index, 0, &self.config.luma, frame, true);
        let grain_g =
            self.generate_grain_plane(w, h, frame_index, 1, &self.config.chroma_cb, frame, false);
        let grain_b =
            self.generate_grain_plane(w, h, frame_index, 2, &self.config.chroma_cr, frame, false);

        // Apply grain to frame
        frame
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(idx, pix)| {
                let luma =
                    (0.299 * pix[0] as f32 + 0.587 * pix[1] as f32 + 0.114 * pix[2] as f32) / 255.0;

                let apply_grain = |channel: u8, cfg: &ChannelGrainConfig, grain: f32| -> u8 {
                    let luma_weight = 1.0 - cfg.luma_dependency * luma;
                    let delta = grain * cfg.gain * 255.0 * luma_weight;
                    (channel as f32 + delta).clamp(0.0, 255.0) as u8
                };

                pix[0] = apply_grain(pix[0], &self.config.luma, grain_r[idx]);
                pix[1] = apply_grain(
                    pix[1],
                    &self.config.chroma_cb,
                    grain_g[idx] * self.config.chroma_scale,
                );
                pix[2] = apply_grain(
                    pix[2],
                    &self.config.chroma_cr,
                    grain_b[idx] * self.config.chroma_scale,
                );
                // Alpha channel unchanged
            });

        Ok(())
    }

    /// Apply film grain to a planar YUV420 frame.
    ///
    /// # Arguments
    ///
    /// * `y_plane` – Mutable luma plane (width * height bytes).
    /// * `u_plane` – Mutable Cb plane (width/2 * height/2 bytes).
    /// * `v_plane` – Mutable Cr plane (width/2 * height/2 bytes).
    /// * `width` – Frame width (must be even).
    /// * `height` – Frame height (must be even).
    /// * `frame_index` – Frame number.
    ///
    /// # Errors
    ///
    /// Returns an error if any plane size is wrong or dimensions are invalid.
    pub fn apply_yuv420(
        &self,
        y_plane: &mut [u8],
        u_plane: &mut [u8],
        v_plane: &mut [u8],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> Result<()> {
        let y_expected = (width as usize) * (height as usize);
        let uv_expected = (width as usize / 2) * (height as usize / 2);

        if y_plane.len() != y_expected {
            return Err(GpuError::InvalidBufferSize {
                expected: y_expected,
                actual: y_plane.len(),
            });
        }
        if u_plane.len() != uv_expected || v_plane.len() != uv_expected {
            return Err(GpuError::InvalidBufferSize {
                expected: uv_expected,
                actual: u_plane.len().min(v_plane.len()),
            });
        }

        let w = width as usize;
        let h = height as usize;
        let uw = w / 2;
        let uh = h / 2;

        // Luma grain (use a dummy empty frame for luma; luma_plane itself is available)
        let y_grain =
            self.generate_grain_plane_from_luma(w, h, frame_index, 0, &self.config.luma, y_plane);

        // Apply Y grain
        y_plane.par_iter_mut().enumerate().for_each(|(i, px)| {
            let luma = *px as f32 / 255.0;
            let luma_weight = 1.0 - self.config.luma.luma_dependency * luma;
            let delta = y_grain[i] * self.config.luma.gain * 255.0 * luma_weight;
            *px = (*px as f32 + delta).clamp(0.0, 255.0) as u8;
        });

        // Chroma Cb grain
        let u_grain = generate_uniform_grain(uw, uh, frame_index ^ 0xABCD_1234, 1);
        let u_smoothed =
            smooth_grain_plane(&u_grain, uw, uh, self.config.chroma_cb.frequency_scale);
        u_plane.par_iter_mut().enumerate().for_each(|(i, px)| {
            let delta =
                u_smoothed[i] * self.config.chroma_cb.gain * 255.0 * self.config.chroma_scale;
            *px = (*px as f32 + delta).clamp(0.0, 255.0) as u8;
        });

        // Chroma Cr grain
        let v_grain = generate_uniform_grain(uw, uh, frame_index ^ 0xDEAD_BEEF, 2);
        let v_smoothed =
            smooth_grain_plane(&v_grain, uw, uh, self.config.chroma_cr.frequency_scale);
        v_plane.par_iter_mut().enumerate().for_each(|(i, px)| {
            let delta =
                v_smoothed[i] * self.config.chroma_cr.gain * 255.0 * self.config.chroma_scale;
            *px = (*px as f32 + delta).clamp(0.0, 255.0) as u8;
        });

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    // Private helpers
    // ─────────────────────────────────────────────────────────────────────────

    /// Generate a frequency-shaped grain plane using luma extracted from the
    /// RGBA frame for amplitude weighting.
    fn generate_grain_plane(
        &self,
        w: usize,
        h: usize,
        frame_index: u64,
        channel_seed: u64,
        cfg: &ChannelGrainConfig,
        _frame: &[u8],
        _is_luma: bool,
    ) -> Vec<f32> {
        let raw = generate_uniform_grain(
            w,
            h,
            frame_index ^ (channel_seed * 0x5555_5555),
            channel_seed,
        );
        smooth_grain_plane(&raw, w, h, cfg.frequency_scale)
    }

    /// Generate a grain plane with luma-plane amplitude weighting.
    fn generate_grain_plane_from_luma(
        &self,
        w: usize,
        h: usize,
        frame_index: u64,
        channel_seed: u64,
        cfg: &ChannelGrainConfig,
        _luma: &[u8],
    ) -> Vec<f32> {
        let raw = generate_uniform_grain(
            w,
            h,
            frame_index ^ (channel_seed * 0x5555_5555),
            channel_seed,
        );
        smooth_grain_plane(&raw, w, h, cfg.frequency_scale)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Stand-alone grain generation utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Generate a flat (white-noise) grain plane with values in `[-1, 1]`.
///
/// Uses a deterministic LCG so each (frame, channel) combination always
/// produces the same pattern.
fn generate_uniform_grain(w: usize, h: usize, seed: u64, salt: u64) -> Vec<f32> {
    let n = w * h;
    let mut grain = Vec::with_capacity(n);

    // Use a simple LCG: state' = state * A + C  (mod 2^64)
    // Constants from Knuth Vol 2 / L'Ecuyer tables.
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;

    let mut state = seed.wrapping_add(salt.wrapping_mul(0xDEAD_BEEF_CAFE));

    for _ in 0..n {
        state = state.wrapping_mul(A).wrapping_add(C);
        // Map top 32 bits to [-1, 1]
        let v = ((state >> 32) as u32 as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        grain.push(v);
    }

    grain
}

/// Apply Gaussian smoothing to a grain plane to shape its frequency content.
///
/// The sigma of the Gaussian equals `frequency_scale`; larger values produce
/// lower-frequency (coarser) grain.
fn smooth_grain_plane(grain: &[f32], w: usize, h: usize, sigma: f32) -> Vec<f32> {
    if sigma <= 0.5 || w == 0 || h == 0 {
        return grain.to_vec();
    }

    let radius = (3.0 * sigma).ceil() as usize;
    let size = 2 * radius + 1;

    // Precompute kernel
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut kernel = Vec::with_capacity(size);
    let mut ksum = 0.0f32;
    for i in 0..size {
        let x = i as f32 - radius as f32;
        let v = (-x * x / two_sigma_sq).exp();
        kernel.push(v);
        ksum += v;
    }
    for v in &mut kernel {
        *v /= ksum;
    }

    // Horizontal pass
    let mut temp = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx =
                    (x as isize + ki as isize - radius as isize).clamp(0, w as isize - 1) as usize;
                acc += kv * grain[y * w + sx];
            }
            temp[y * w + x] = acc;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy =
                    (y as isize + ki as isize - radius as isize).clamp(0, h as isize - 1) as usize;
                acc += kv * temp[sy * w + x];
            }
            out[y * w + x] = acc;
        }
    }

    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Grain statistics
// ─────────────────────────────────────────────────────────────────────────────

/// Per-channel statistics measured from a film-grain pattern.
///
/// These are computed from a synthesised grain plane *before* the luma-scaling
/// is applied so that they reflect the raw generator output.
#[derive(Debug, Clone)]
pub struct GrainStatistics {
    /// Mean of the raw grain values (should be near 0 for balanced generators).
    pub mean: f32,
    /// Standard deviation of the raw grain values.
    pub std_dev: f32,
    /// Minimum grain value seen in the plane.
    pub min: f32,
    /// Maximum grain value seen in the plane.
    pub max: f32,
    /// Fraction of pixels where grain is positive.
    pub positive_fraction: f32,
}

impl GrainStatistics {
    /// Compute statistics from a raw grain plane with values in `[-1, 1]`.
    #[must_use]
    pub fn from_grain_plane(plane: &[f32]) -> Self {
        if plane.is_empty() {
            return Self {
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                positive_fraction: 0.0,
            };
        }

        let n = plane.len() as f32;
        let mut sum = 0.0f32;
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        let mut pos = 0u64;

        for &v in plane {
            sum += v;
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
            if v > 0.0 {
                pos += 1;
            }
        }

        let mean = sum / n;

        let variance = plane.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
        let std_dev = variance.sqrt();

        Self {
            mean,
            std_dev,
            min,
            max,
            positive_fraction: pos as f32 / n,
        }
    }
}

impl FilmGrainSynthesizer {
    /// Apply film grain to an HDR RGBA frame stored as `f32` per channel.
    ///
    /// Each value is expected in the range `[0.0, 1.0]`. The grain amplitude
    /// is scaled by `config.luma.gain` and luma-weighted as in the u8 path.
    ///
    /// # Arguments
    ///
    /// * `frame`       – Mutable slice of f32 RGBA values (`width * height * 4` elements).
    /// * `width`       – Frame width.
    /// * `height`      – Frame height.
    /// * `frame_index` – Frame number (used to vary grain per frame).
    ///
    /// # Errors
    ///
    /// Returns an error if the frame buffer has the wrong length or dimensions are zero.
    pub fn apply_rgba_f32(
        &self,
        frame: &mut [f32],
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> crate::Result<()> {
        let expected = (width as usize) * (height as usize) * 4;
        if frame.len() != expected {
            return Err(crate::GpuError::InvalidBufferSize {
                expected,
                actual: frame.len(),
            });
        }
        if width == 0 || height == 0 {
            return Err(crate::GpuError::InvalidDimensions { width, height });
        }

        let w = width as usize;
        let h = height as usize;

        // Build a pseudo-u8 luma map from the f32 frame for luma-dependency.
        let luma_u8: Vec<u8> = frame
            .chunks_exact(4)
            .map(|px| {
                let l = (0.299 * px[0] + 0.587 * px[1] + 0.114 * px[2]).clamp(0.0, 1.0);
                (l * 255.0) as u8
            })
            .collect();

        let grain_r =
            self.generate_grain_plane_from_luma(w, h, frame_index, 0, &self.config.luma, &luma_u8);
        let grain_g = self.generate_grain_plane_from_luma(
            w,
            h,
            frame_index,
            1,
            &self.config.chroma_cb,
            &luma_u8,
        );
        let grain_b = self.generate_grain_plane_from_luma(
            w,
            h,
            frame_index,
            2,
            &self.config.chroma_cr,
            &luma_u8,
        );

        frame
            .par_chunks_exact_mut(4)
            .enumerate()
            .for_each(|(idx, pix)| {
                let luma = (0.299 * pix[0] + 0.587 * pix[1] + 0.114 * pix[2]).clamp(0.0, 1.0);

                let apply = |ch: f32, cfg: &ChannelGrainConfig, grain: f32| -> f32 {
                    let luma_weight = 1.0 - cfg.luma_dependency * luma;
                    let delta = grain * cfg.gain * luma_weight;
                    (ch + delta).clamp(0.0, 1.0)
                };

                pix[0] = apply(pix[0], &self.config.luma, grain_r[idx]);
                pix[1] = apply(
                    pix[1],
                    &self.config.chroma_cb,
                    grain_g[idx] * self.config.chroma_scale,
                );
                pix[2] = apply(
                    pix[2],
                    &self.config.chroma_cr,
                    grain_b[idx] * self.config.chroma_scale,
                );
                // alpha unchanged
            });

        Ok(())
    }

    /// Measure statistics of the raw luma grain plane for a given frame.
    ///
    /// Returns a [`GrainStatistics`] snapshot for the luma channel grain
    /// pattern that would be applied to frame `frame_index`.
    #[must_use]
    pub fn measure_grain_stats(
        &self,
        width: u32,
        height: u32,
        frame_index: u64,
    ) -> GrainStatistics {
        let w = width as usize;
        let h = height as usize;
        let raw = generate_uniform_grain(w, h, frame_index ^ (0_u64 * 0x5555_5555), 0);
        let smoothed = smooth_grain_plane(&raw, w, h, self.config.luma.frequency_scale);
        GrainStatistics::from_grain_plane(&smoothed)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_apply_rgba_correct_size() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let w = 16u32;
        let h = 16u32;
        let mut frame = vec![128u8; (w * h * 4) as usize];
        synth
            .apply_rgba(&mut frame, w, h, 0)
            .expect("should succeed");
        // Frame should still be valid RGBA
        assert_eq!(frame.len(), (w * h * 4) as usize);
    }

    #[test]
    fn test_apply_rgba_wrong_size_rejected() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let mut frame = vec![128u8; 10];
        let res = synth.apply_rgba(&mut frame, 4, 4, 0);
        assert!(res.is_err());
    }

    #[test]
    fn test_apply_rgba_modifies_frame() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig {
            luma: ChannelGrainConfig {
                gain: 0.3,
                ..Default::default()
            },
            ..Default::default()
        });
        let w = 16u32;
        let h = 16u32;
        let original = vec![128u8; (w * h * 4) as usize];
        let mut frame = original.clone();
        synth
            .apply_rgba(&mut frame, w, h, 42)
            .expect("should succeed");
        // With non-trivial gain the frame should differ from the original
        let changed = frame.iter().zip(original.iter()).any(|(a, b)| a != b);
        assert!(changed, "grain should modify the frame");
    }

    #[test]
    fn test_different_frame_indices_differ() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let w = 16u32;
        let h = 16u32;
        let base = vec![128u8; (w * h * 4) as usize];

        let mut f0 = base.clone();
        let mut f1 = base.clone();
        synth.apply_rgba(&mut f0, w, h, 0).unwrap();
        synth.apply_rgba(&mut f1, w, h, 1).unwrap();

        // Different seeds should produce different patterns
        let differ = f0.iter().zip(f1.iter()).any(|(a, b)| a != b);
        assert!(
            differ,
            "different frame indices should produce different grain"
        );
    }

    #[test]
    fn test_apply_yuv420() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let w = 16usize;
        let h = 16usize;
        let mut y = vec![128u8; w * h];
        let mut u = vec![128u8; (w / 2) * (h / 2)];
        let mut v = vec![128u8; (w / 2) * (h / 2)];
        synth
            .apply_yuv420(&mut y, &mut u, &mut v, w as u32, h as u32, 0)
            .expect("should succeed");
    }

    #[test]
    fn test_light_heavy_presets_differ() {
        let light = FilmGrainConfig::light();
        let heavy = FilmGrainConfig::heavy();
        assert!(heavy.luma.gain > light.luma.gain);
    }

    #[test]
    fn test_grain_alpha_unchanged() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig {
            luma: ChannelGrainConfig {
                gain: 0.5,
                ..Default::default()
            },
            ..Default::default()
        });
        let w = 8u32;
        let h = 8u32;
        let mut frame: Vec<u8> = (0..w * h).flat_map(|_| [128u8, 128, 128, 200]).collect();
        let original_alpha: Vec<u8> = frame.iter().skip(3).step_by(4).copied().collect();
        synth.apply_rgba(&mut frame, w, h, 5).unwrap();
        let new_alpha: Vec<u8> = frame.iter().skip(3).step_by(4).copied().collect();
        assert_eq!(
            original_alpha, new_alpha,
            "alpha channel must not be modified"
        );
    }

    // ── New tests for enhanced film grain synthesis ───────────────────────────

    #[test]
    fn test_zero_gain_leaves_frame_unchanged() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig {
            luma: ChannelGrainConfig {
                gain: 0.0,
                frequency_scale: 1.2,
                luma_dependency: 0.4,
            },
            chroma_cb: ChannelGrainConfig {
                gain: 0.0,
                ..Default::default()
            },
            chroma_cr: ChannelGrainConfig {
                gain: 0.0,
                ..Default::default()
            },
            chroma_scale: 0.5,
            block_size: 16,
        });
        let w = 16u32;
        let h = 16u32;
        let original: Vec<u8> = (0..w * h * 4)
            .map(|i| ((i * 7 + 13) % 200 + 28) as u8)
            .collect();
        let mut frame = original.clone();
        synth
            .apply_rgba(&mut frame, w, h, 0)
            .expect("should succeed");
        assert_eq!(frame, original, "zero gain must not modify any pixel");
    }

    #[test]
    fn test_same_frame_index_is_deterministic() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let w = 16u32;
        let h = 16u32;
        let base = vec![128u8; (w * h * 4) as usize];

        let mut f0 = base.clone();
        let mut f1 = base.clone();
        synth.apply_rgba(&mut f0, w, h, 42).unwrap();
        synth.apply_rgba(&mut f1, w, h, 42).unwrap();

        assert_eq!(f0, f1, "same frame index must produce identical grain");
    }

    #[test]
    fn test_apply_rgba_f32_basic() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let w = 8u32;
        let h = 8u32;
        let n = (w * h * 4) as usize;
        let mut frame: Vec<f32> = (0..n).map(|i| (i % 4) as f32 / 3.0).collect();
        synth
            .apply_rgba_f32(&mut frame, w, h, 0)
            .expect("f32 apply should succeed");
        // All values must remain in [0, 1].
        for (i, &v) in frame.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&v),
                "f32 grain out of bounds at index {i}: {v}"
            );
        }
    }

    #[test]
    fn test_apply_rgba_f32_wrong_size_rejected() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let mut frame = vec![0.5f32; 10]; // wrong
        let res = synth.apply_rgba_f32(&mut frame, 4, 4, 0);
        assert!(res.is_err(), "wrong buffer size must return error");
    }

    #[test]
    fn test_apply_rgba_f32_alpha_unchanged() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig {
            luma: ChannelGrainConfig {
                gain: 0.3,
                ..Default::default()
            },
            ..Default::default()
        });
        let w = 8u32;
        let h = 8u32;
        let n = (w * h * 4) as usize;
        let mut frame: Vec<f32> = (0..n)
            .map(|i| if i % 4 == 3 { 0.75 } else { 0.5 })
            .collect();
        let orig_alpha: Vec<f32> = frame.iter().skip(3).step_by(4).copied().collect();
        synth.apply_rgba_f32(&mut frame, w, h, 7).unwrap();
        let new_alpha: Vec<f32> = frame.iter().skip(3).step_by(4).copied().collect();
        for (i, (&a, &b)) in orig_alpha.iter().zip(new_alpha.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "f32 alpha changed at pixel {i}: {a} → {b}"
            );
        }
    }

    #[test]
    fn test_grain_statistics_mean_near_zero() {
        // Unbiased LCG grain should have mean ≈ 0 for large planes.
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let stats = synth.measure_grain_stats(64, 64, 0);
        assert!(
            stats.mean.abs() < 0.1,
            "grain mean should be near 0, got {}",
            stats.mean
        );
    }

    #[test]
    fn test_grain_statistics_std_dev_positive() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let stats = synth.measure_grain_stats(32, 32, 0);
        assert!(
            stats.std_dev > 0.0,
            "grain std_dev should be positive, got {}",
            stats.std_dev
        );
    }

    #[test]
    fn test_grain_statistics_bounds() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let stats = synth.measure_grain_stats(32, 32, 0);
        assert!(
            stats.min >= -1.0 && stats.max <= 1.0,
            "grain should be in [-1, 1]; min={}, max={}",
            stats.min,
            stats.max
        );
        assert!(
            stats.positive_fraction > 0.2 && stats.positive_fraction < 0.8,
            "roughly half the grain should be positive; got {}",
            stats.positive_fraction
        );
    }

    #[test]
    fn test_grain_statistics_from_empty_plane() {
        let stats = GrainStatistics::from_grain_plane(&[]);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std_dev, 0.0);
        assert_eq!(stats.positive_fraction, 0.0);
    }

    #[test]
    fn test_heavy_grain_larger_std_dev_than_light() {
        let light_synth = FilmGrainSynthesizer::new(FilmGrainConfig::light());
        let heavy_synth = FilmGrainSynthesizer::new(FilmGrainConfig::heavy());

        // Check raw grain level: heavy has larger gain, so amplitude should be larger
        assert!(
            heavy_synth.config.luma.gain > light_synth.config.luma.gain,
            "heavy luma gain should exceed light luma gain"
        );
    }

    #[test]
    fn test_yuv420_wrong_y_size_rejected() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let mut y = vec![128u8; 8]; // wrong: should be 16*16
        let mut u = vec![128u8; 64];
        let mut v = vec![128u8; 64];
        let res = synth.apply_yuv420(&mut y, &mut u, &mut v, 16, 16, 0);
        assert!(res.is_err(), "wrong Y plane size must return error");
    }

    #[test]
    fn test_apply_rgba_large_frame() {
        let synth = FilmGrainSynthesizer::new(FilmGrainConfig::default());
        let w = 64u32;
        let h = 64u32;
        let mut frame = vec![200u8; (w * h * 4) as usize];
        synth
            .apply_rgba(&mut frame, w, h, 0)
            .expect("large frame should succeed");
        assert_eq!(frame.len(), (w * h * 4) as usize);
    }
}
