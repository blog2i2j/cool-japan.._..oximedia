//! Exposure bracketing and Debevec HDR merge.
//!
//! Implements Paul Debevec and Jitendra Malik's method for recovering the
//! camera response function and assembling a High Dynamic Range radiance map
//! from multiple LDR exposures ("Recovering High Dynamic Range Radiance Maps
//! from Photographs", SIGGRAPH 1997).
//!
//! # Algorithm
//!
//! 1. **Response curve recovery**: Solve the overdetermined linear system
//!    `g(Z[i,j]) = ln(E[i]) + ln(Δt[j])` by SVD/least-squares with
//!    smoothness regularisation. Uses a hat weighting function.
//! 2. **HDR assembly**: Reconstruct `ln(E)` per pixel by weighted average
//!    over all exposures, using the recovered response curve.
//! 3. Optional **tone mapping**: simple global Reinhard operator.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{ImageError, ImageResult};

// ── Hat weight function ───────────────────────────────────────────────────────

/// Hat weighting function for pixel value `z` in [0, 255].
///
/// Returns 0 for values near 0 or 255, and peaks at z=128.
#[must_use]
pub fn hat_weight(z: u8) -> f64 {
    let z = z as f64;
    let mid = 127.5;
    if z <= mid {
        z + 0.5
    } else {
        255.0 - z + 0.5
    }
}

/// Normalised hat weight: maps z → weight in (0, 1].
#[must_use]
pub fn hat_weight_normalised(z: u8) -> f64 {
    hat_weight(z) / 128.0
}

// ── Bracketed exposure set ────────────────────────────────────────────────────

/// A single bracketed exposure: pixel values + exposure time.
#[derive(Debug, Clone)]
pub struct BracketedExposure {
    /// Raw pixel values (uint8, interleaved RGB or grayscale).
    pub pixels: Vec<u8>,
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of channels (1 or 3).
    pub channels: u8,
    /// Exposure time in seconds (Δt).
    pub exposure_time: f64,
}

impl BracketedExposure {
    /// Create a new bracketed exposure.
    pub fn new(
        pixels: Vec<u8>,
        width: u32,
        height: u32,
        channels: u8,
        exposure_time: f64,
    ) -> ImageResult<Self> {
        let expected = width as usize * height as usize * channels as usize;
        if pixels.len() != expected {
            return Err(ImageError::invalid_format(format!(
                "Pixel buffer length {} != expected {}",
                pixels.len(),
                expected
            )));
        }
        if exposure_time <= 0.0 {
            return Err(ImageError::invalid_format("Exposure time must be positive"));
        }
        Ok(Self {
            pixels,
            width,
            height,
            channels,
            exposure_time,
        })
    }

    /// Sample a specific pixel's channel value.
    #[must_use]
    pub fn sample(&self, x: u32, y: u32, channel: u8) -> u8 {
        let idx = (y as usize * self.width as usize + x as usize) * self.channels as usize
            + channel as usize;
        if idx < self.pixels.len() {
            self.pixels[idx]
        } else {
            0
        }
    }

    /// Total number of pixels.
    #[must_use]
    pub fn pixel_count(&self) -> usize {
        self.width as usize * self.height as usize
    }
}

// ── Response curve recovery ───────────────────────────────────────────────────

/// Configuration for Debevec HDR merge.
#[derive(Debug, Clone)]
pub struct DebevecConfig {
    /// Smoothness regularisation weight (lambda). Typical: 10-50.
    pub lambda: f64,
    /// Number of sample pixels to use for response curve fitting.
    pub sample_pixels: usize,
    /// Random seed for sampling.
    pub seed: u64,
}

impl Default for DebevecConfig {
    fn default() -> Self {
        Self {
            lambda: 10.0,
            sample_pixels: 256,
            seed: 12345,
        }
    }
}

/// Recovered camera response function g(z) for z in [0, 255].
#[derive(Debug, Clone)]
pub struct ResponseCurve {
    /// g(z) values: g\[z\] = ln(irradiance) for camera output z.
    pub g: [f64; 256],
}

impl ResponseCurve {
    /// Evaluate g at value z.
    #[must_use]
    pub fn eval(&self, z: u8) -> f64 {
        self.g[z as usize]
    }

    /// Returns a simple linear (log-linear) response curve for testing.
    #[must_use]
    pub fn linear() -> Self {
        let mut g = [0.0f64; 256];
        for (z, gv) in g.iter_mut().enumerate() {
            // Use z+1 to ensure strict monotonicity across all 256 values
            *gv = ((z + 1) as f64 / 256.0).ln();
        }
        Self { g }
    }

    /// Compute the smoothness of this response curve (sum of second differences).
    #[must_use]
    pub fn smoothness(&self) -> f64 {
        let mut s = 0.0;
        for z in 1..255 {
            let d2 = self.g[z + 1] - 2.0 * self.g[z] + self.g[z - 1];
            s += d2 * d2;
        }
        s
    }
}

/// Recover the camera response function using the Debevec-Malik algorithm.
///
/// Requires at least 2 exposures.  Solves a least-squares system for each
/// channel independently.
///
/// # Arguments
/// * `exposures` – Bracketed exposures (same size/format).
/// * `config` – Algorithm configuration.
///
/// Returns a `ResponseCurve` per channel (1 for gray, 3 for RGB).
pub fn recover_response_curve(
    exposures: &[BracketedExposure],
    config: &DebevecConfig,
) -> ImageResult<Vec<ResponseCurve>> {
    if exposures.len() < 2 {
        return Err(ImageError::invalid_format(
            "At least 2 exposures required for Debevec HDR",
        ));
    }

    let channels = exposures[0].channels as usize;
    let pixel_count = exposures[0].pixel_count();
    // Validate all exposures have same dimensions
    for (i, exp) in exposures.iter().enumerate() {
        if exp.width != exposures[0].width || exp.height != exposures[0].height {
            return Err(ImageError::invalid_format(format!(
                "Exposure {i} has different dimensions"
            )));
        }
        if exp.channels != exposures[0].channels {
            return Err(ImageError::invalid_format(format!(
                "Exposure {i} has different channel count"
            )));
        }
    }

    // Sample pixel locations using a simple LCG
    let n_samples = config.sample_pixels.min(pixel_count);
    let mut sampled_pixels = Vec::with_capacity(n_samples);
    {
        let mut state = config.seed;
        let step = (pixel_count / n_samples).max(1);
        let mut idx = 0usize;
        while sampled_pixels.len() < n_samples && idx < pixel_count {
            sampled_pixels.push(idx);
            // LCG next
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            idx += step + (state >> 60) as usize;
        }
    }
    if sampled_pixels.is_empty() {
        sampled_pixels.push(0);
    }

    let n = sampled_pixels.len();
    let p = 256usize;
    let ln_times: Vec<f64> = exposures.iter().map(|e| e.exposure_time.ln()).collect();

    let mut result = Vec::with_capacity(channels);

    for ch in 0..channels {
        // Build the system: n*j + i rows for data, 254 rows for smoothness
        // unknowns: g[0..256] (256) + ln(E)[0..n] (n)
        // We use simplified solver: fix g[128] = 0 as anchor

        // Accumulate A^T A and A^T b (normal equations) for least squares
        let n_unknowns = p + n; // g[0..256] + lnE[0..n]
        let mut ata = vec![0.0f64; n_unknowns * n_unknowns];
        let mut atb = vec![0.0f64; n_unknowns];

        // Data equations: w(z) * [g(z) - lnE[i] - ln(dt[j])] = 0
        for (i, &pix_idx) in sampled_pixels.iter().enumerate() {
            for (j, exp) in exposures.iter().enumerate() {
                let z_raw = {
                    let px_base = pix_idx * channels + ch;
                    if px_base < exp.pixels.len() {
                        exp.pixels[px_base]
                    } else {
                        128
                    }
                };
                let z = z_raw as usize;
                let w = hat_weight(z_raw);
                if w < 1e-9 {
                    continue;
                }

                // Equation: w * g[z] - w * lnE[i] = w * ln(dt[j])
                let gz_idx = z;
                let le_idx = p + i;

                // Add w^2 to ata[gz_idx, gz_idx]
                ata[gz_idx * n_unknowns + gz_idx] += w * w;
                // Add -w^2 to ata[gz_idx, le_idx]
                ata[gz_idx * n_unknowns + le_idx] -= w * w;
                ata[le_idx * n_unknowns + gz_idx] -= w * w;
                // Add w^2 to ata[le_idx, le_idx]
                ata[le_idx * n_unknowns + le_idx] += w * w;
                // atb
                let rhs = w * ln_times[j];
                atb[gz_idx] += w * rhs;
                atb[le_idx] -= w * rhs;
            }
        }

        // Smoothness equations: lambda * w(z) * [g(z-1) - 2g(z) + g(z+1)] = 0
        for z in 1..255usize {
            let w = hat_weight(z as u8) * config.lambda;
            if w < 1e-9 {
                continue;
            }
            // Row: w*(g[z-1] - 2*g[z] + g[z+1]) = 0
            ata[(z - 1) * n_unknowns + (z - 1)] += w * w;
            ata[z * n_unknowns + z] += 4.0 * w * w;
            ata[(z + 1) * n_unknowns + (z + 1)] += w * w;
            ata[(z - 1) * n_unknowns + z] -= 2.0 * w * w;
            ata[z * n_unknowns + (z - 1)] -= 2.0 * w * w;
            ata[(z - 1) * n_unknowns + (z + 1)] += w * w;
            ata[(z + 1) * n_unknowns + (z - 1)] += w * w;
            ata[z * n_unknowns + (z + 1)] -= 2.0 * w * w;
            ata[(z + 1) * n_unknowns + z] -= 2.0 * w * w;
        }

        // Anchor: g[128] = 0 (add large weight to diagonal)
        let anchor = 128usize;
        ata[anchor * n_unknowns + anchor] += 1e8;
        // atb[anchor] += 0

        // Solve with Gauss-Seidel iteration (simpler than full LU for large sparse systems)
        let mut x = vec![0.0f64; n_unknowns];
        // Initialize g with linear assumption
        for z in 0..256 {
            x[z] = if z == 0 {
                (1.0f64 / 255.0).ln()
            } else {
                (z as f64 / 255.0).ln()
            };
        }

        for _iter in 0..50 {
            for i in 0..n_unknowns {
                let diag = ata[i * n_unknowns + i];
                if diag.abs() < 1e-12 {
                    continue;
                }
                let mut sum = atb[i];
                for j in 0..n_unknowns {
                    if j != i {
                        sum -= ata[i * n_unknowns + j] * x[j];
                    }
                }
                x[i] = sum / diag;
            }
        }

        // Extract g values
        let mut g = [0.0f64; 256];
        for z in 0..256 {
            g[z] = x[z];
        }

        result.push(ResponseCurve { g });
    }

    Ok(result)
}

// ── HDR radiance map assembly ──────────────────────────────────────────────────

/// HDR radiance map: linear float values per pixel per channel.
#[derive(Debug, Clone)]
pub struct HdrRadianceMap {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Number of channels.
    pub channels: u8,
    /// Linearised radiance values: [pixel * channels + channel].
    pub radiance: Vec<f32>,
}

impl HdrRadianceMap {
    /// Sample radiance at pixel (x, y), channel c.
    #[must_use]
    pub fn sample(&self, x: u32, y: u32, c: u8) -> f32 {
        let idx =
            (y as usize * self.width as usize + x as usize) * self.channels as usize + c as usize;
        if idx < self.radiance.len() {
            self.radiance[idx]
        } else {
            0.0
        }
    }

    /// Returns maximum radiance value across all channels.
    #[must_use]
    pub fn max_radiance(&self) -> f32 {
        self.radiance.iter().copied().fold(0.0f32, f32::max)
    }

    /// Returns minimum non-zero radiance.
    #[must_use]
    pub fn min_radiance(&self) -> f32 {
        self.radiance
            .iter()
            .copied()
            .filter(|&v| v > 0.0)
            .fold(f32::MAX, f32::min)
    }
}

/// Assemble an HDR radiance map from bracketed exposures and response curves.
///
/// For each pixel and channel, computes the weighted average of `g(z) - ln(Δt)`.
pub fn assemble_hdr(
    exposures: &[BracketedExposure],
    curves: &[ResponseCurve],
) -> ImageResult<HdrRadianceMap> {
    if exposures.is_empty() {
        return Err(ImageError::invalid_format("No exposures provided"));
    }
    if curves.len() != exposures[0].channels as usize {
        return Err(ImageError::invalid_format(
            "Mismatch between curves and channels",
        ));
    }

    let width = exposures[0].width;
    let height = exposures[0].height;
    let channels = exposures[0].channels as usize;
    let n_pixels = (width * height) as usize;
    let ln_times: Vec<f64> = exposures.iter().map(|e| e.exposure_time.ln()).collect();

    let mut radiance = vec![0.0f32; n_pixels * channels];

    for pix_idx in 0..n_pixels {
        for ch in 0..channels {
            let mut numerator = 0.0f64;
            let mut denominator = 0.0f64;

            for (j, exp) in exposures.iter().enumerate() {
                let raw_idx = pix_idx * channels + ch;
                let z = if raw_idx < exp.pixels.len() {
                    exp.pixels[raw_idx]
                } else {
                    128
                };
                let w = hat_weight(z);
                if w < 1e-9 {
                    continue;
                }
                let g_z = curves[ch].eval(z);
                numerator += w * (g_z - ln_times[j]);
                denominator += w;
            }

            let ln_e = if denominator > 1e-12 {
                numerator / denominator
            } else {
                0.0
            };
            radiance[pix_idx * channels + ch] = ln_e.exp() as f32;
        }
    }

    Ok(HdrRadianceMap {
        width,
        height,
        channels: channels as u8,
        radiance,
    })
}

// ── Tone mapping ──────────────────────────────────────────────────────────────

/// Global Reinhard tone mapping operator.
///
/// Maps HDR radiance to [0, 1] LDR using the formula:
/// `L_d = L_w / (1 + L_w)` after key-value scaling.
pub fn tone_map_reinhard_global(hdr: &HdrRadianceMap, key: f64) -> Vec<f32> {
    let n = hdr.radiance.len();
    if n == 0 {
        return Vec::new();
    }

    // Compute log-average luminance
    let eps = 1e-6;
    let channels = hdr.channels as usize;
    let pixel_count = (hdr.width * hdr.height) as usize;
    let log_avg_lum = if channels >= 3 {
        let sum: f64 = (0..pixel_count)
            .map(|i| {
                let base = i * channels;
                let r = hdr.radiance[base] as f64;
                let g = hdr.radiance[base + 1] as f64;
                let b = hdr.radiance[base + 2] as f64;
                let lum = 0.2126 * r + 0.7152 * g + 0.0722 * b;
                (lum + eps).ln()
            })
            .sum::<f64>();
        (sum / pixel_count as f64).exp()
    } else {
        let sum: f64 = hdr.radiance.iter().map(|&v| (v as f64 + eps).ln()).sum();
        (sum / n as f64).exp()
    };

    let scale = key / log_avg_lum.max(eps);

    let mut out = Vec::with_capacity(n);
    for &v in &hdr.radiance {
        let lw = v as f64 * scale;
        let ld = (lw / (1.0 + lw)).clamp(0.0, 1.0);
        out.push(ld as f32);
    }
    out
}

/// Convert a tone-mapped radiance map to 8-bit pixels.
#[must_use]
pub fn to_ldr_u8(tone_mapped: &[f32]) -> Vec<u8> {
    tone_mapped
        .iter()
        .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
        .collect()
}

/// Full Debevec HDR pipeline: recover curves + assemble radiance map.
pub fn debevec_hdr_merge(
    exposures: &[BracketedExposure],
    config: &DebevecConfig,
) -> ImageResult<HdrRadianceMap> {
    let curves = recover_response_curve(exposures, config)?;
    assemble_hdr(exposures, &curves)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_exposure(w: u32, h: u32, fill: u8, dt: f64) -> BracketedExposure {
        BracketedExposure::new(vec![fill; (w * h * 3) as usize], w, h, 3, dt).expect("exposure")
    }

    fn make_gray_exposure(w: u32, h: u32, fill: u8, dt: f64) -> BracketedExposure {
        BracketedExposure::new(vec![fill; (w * h) as usize], w, h, 1, dt).expect("exposure")
    }

    #[test]
    fn test_hat_weight_midpoint() {
        // At z=128, hat_weight should be near maximum
        let w128 = hat_weight(128);
        let w0 = hat_weight(0);
        let w255 = hat_weight(255);
        assert!(w128 > w0, "midpoint should have higher weight than 0");
        assert!(w128 > w255, "midpoint should have higher weight than 255");
    }

    #[test]
    fn test_hat_weight_boundaries() {
        let w0 = hat_weight(0);
        let w255 = hat_weight(255);
        // Should be small but non-zero (0.5)
        assert!(w0 > 0.0);
        assert!(w255 > 0.0);
    }

    #[test]
    fn test_hat_weight_normalised_range() {
        for z in 0u8..=255 {
            let w = hat_weight_normalised(z);
            assert!(w > 0.0 && w <= 1.0, "hat weight out of range at {z}: {w}");
        }
    }

    #[test]
    fn test_bracketed_exposure_creation() {
        let exp = make_exposure(4, 4, 128, 0.1);
        assert_eq!(exp.width, 4);
        assert_eq!(exp.height, 4);
        assert_eq!(exp.channels, 3);
        assert_eq!(exp.pixel_count(), 16);
    }

    #[test]
    fn test_bracketed_exposure_bad_size() {
        let result = BracketedExposure::new(vec![0u8; 10], 4, 4, 3, 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_bracketed_exposure_bad_time() {
        let result = BracketedExposure::new(vec![0u8; 48], 4, 4, 3, -1.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_bracketed_exposure_sample() {
        let mut pixels = vec![0u8; 3 * 3 * 3];
        pixels[4 * 3 + 1] = 200; // pixel (1,1) green = 200
        let exp = BracketedExposure::new(pixels, 3, 3, 3, 0.1).expect("exp");
        assert_eq!(exp.sample(1, 1, 1), 200);
    }

    #[test]
    fn test_response_curve_linear_monotonic() {
        let curve = ResponseCurve::linear();
        // ln(E) should be monotonically increasing
        for z in 1..255 {
            assert!(
                curve.g[z] > curve.g[z - 1],
                "linear curve not monotonic at z={z}: g[z]={} <= g[z-1]={}",
                curve.g[z],
                curve.g[z - 1]
            );
        }
    }

    #[test]
    fn test_response_curve_smoothness() {
        let curve = ResponseCurve::linear();
        let s = curve.smoothness();
        assert!(s.is_finite(), "smoothness must be finite");
    }

    #[test]
    fn test_recover_response_curve_needs_two() {
        let exp = make_exposure(4, 4, 128, 0.1);
        let config = DebevecConfig::default();
        let result = recover_response_curve(&[exp], &config);
        assert!(result.is_err());
    }

    #[test]
    fn test_recover_response_curve_basic() {
        let e1 = make_gray_exposure(8, 8, 64, 0.0625);
        let e2 = make_gray_exposure(8, 8, 128, 0.125);
        let e3 = make_gray_exposure(8, 8, 200, 0.25);
        let config = DebevecConfig {
            sample_pixels: 16,
            ..Default::default()
        };
        let curves = recover_response_curve(&[e1, e2, e3], &config).expect("recover curves");
        assert_eq!(curves.len(), 1);
        // g should be defined at all 256 values
        for (z, &g) in curves[0].g.iter().enumerate() {
            assert!(g.is_finite(), "g[{z}] = {g} is not finite");
        }
    }

    #[test]
    fn test_assemble_hdr_basic() {
        let curve = ResponseCurve::linear();
        let e1 = make_gray_exposure(4, 4, 64, 0.1);
        let e2 = make_gray_exposure(4, 4, 128, 0.2);
        let hdr = assemble_hdr(&[e1, e2], &[curve]).expect("hdr");
        assert_eq!(hdr.width, 4);
        assert_eq!(hdr.height, 4);
        assert_eq!(hdr.radiance.len(), 4 * 4);
        for &v in &hdr.radiance {
            assert!(
                v.is_finite() && v >= 0.0,
                "radiance must be non-negative finite"
            );
        }
    }

    #[test]
    fn test_assemble_hdr_channel_mismatch() {
        let curve = ResponseCurve::linear();
        let exp = make_exposure(4, 4, 128, 0.1); // 3 channels
                                                 // Only 1 curve but 3 channels
        let result = assemble_hdr(&[exp], &[curve]);
        assert!(result.is_err());
    }

    #[test]
    fn test_tone_map_reinhard_range() {
        let hdr = HdrRadianceMap {
            width: 4,
            height: 4,
            channels: 3,
            radiance: (0..48).map(|i| (i as f32 + 1.0) * 0.5).collect(),
        };
        let ldr = tone_map_reinhard_global(&hdr, 0.18);
        assert_eq!(ldr.len(), 48);
        for &v in &ldr {
            assert!(v >= 0.0 && v <= 1.0, "tone-mapped value out of range: {v}");
        }
    }

    #[test]
    fn test_tone_map_empty() {
        let hdr = HdrRadianceMap {
            width: 0,
            height: 0,
            channels: 3,
            radiance: vec![],
        };
        let ldr = tone_map_reinhard_global(&hdr, 0.18);
        assert!(ldr.is_empty());
    }

    #[test]
    fn test_to_ldr_u8_range() {
        let tone_mapped = vec![0.0f32, 0.25, 0.5, 0.75, 1.0];
        let u8_vals = to_ldr_u8(&tone_mapped);
        assert_eq!(u8_vals[0], 0);
        assert_eq!(u8_vals[4], 255);
        assert_eq!(u8_vals.len(), tone_mapped.len());
    }

    #[test]
    fn test_to_ldr_u8_clamping() {
        let tone_mapped = vec![-0.5f32, 1.5];
        let u8_vals = to_ldr_u8(&tone_mapped);
        assert_eq!(u8_vals[0], 0);
        assert_eq!(u8_vals[1], 255);
    }

    #[test]
    fn test_hdr_radiance_map_sample() {
        let radiance = vec![0.5f32, 1.0, 0.25, 0.8, 0.3, 0.6];
        let hdr = HdrRadianceMap {
            width: 2,
            height: 1,
            channels: 3,
            radiance,
        };
        assert!((hdr.sample(0, 0, 0) - 0.5).abs() < 1e-6);
        assert!((hdr.sample(0, 0, 1) - 1.0).abs() < 1e-6);
        assert!((hdr.sample(1, 0, 0) - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_hdr_max_min_radiance() {
        let radiance = vec![0.1f32, 5.0, 0.5, 2.0];
        let hdr = HdrRadianceMap {
            width: 2,
            height: 1,
            channels: 2,
            radiance,
        };
        assert!((hdr.max_radiance() - 5.0).abs() < 1e-6);
        assert!((hdr.min_radiance() - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_debevec_pipeline() {
        // Simulate 3-exposure bracket with known values
        let e1 = make_gray_exposure(6, 6, 50, 0.0625);
        let e2 = make_gray_exposure(6, 6, 100, 0.125);
        let e3 = make_gray_exposure(6, 6, 180, 0.25);
        let config = DebevecConfig {
            sample_pixels: 8,
            lambda: 5.0,
            seed: 42,
        };
        let hdr = debevec_hdr_merge(&[e1, e2, e3], &config).expect("debevec");
        assert_eq!(hdr.pixel_count(), 36);
        assert!(hdr.max_radiance() > 0.0);
    }

    impl HdrRadianceMap {
        fn pixel_count(&self) -> usize {
            (self.width * self.height) as usize
        }
    }

    #[test]
    fn test_different_dimensions_rejected() {
        let e1 = make_exposure(4, 4, 128, 0.1);
        let e2 = make_exposure(6, 4, 128, 0.2); // different width
        let config = DebevecConfig::default();
        let result = recover_response_curve(&[e1, e2], &config);
        assert!(result.is_err());
    }
}
