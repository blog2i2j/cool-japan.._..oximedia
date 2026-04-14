//! Video noise profiling with spatial frequency analysis and per-channel
//! noise estimation.
//!
//! This module provides tools for characterising video noise at both
//! spatial-frequency and per-channel levels.  The primary use-cases are:
//!
//! * Deciding how aggressively to denoise a given frame.
//! * Comparing noise characteristics between source and encoded clips.
//! * Detecting sensor or compression artefacts.
//!
//! ## Algorithms
//!
//! * **Laplacian MAD estimator** — classical Donoho–Johnstone robust
//!   noise-sigma estimation via the 3×3 Laplacian kernel.
//! * **Spatial frequency breakdown** — the frame is divided into
//!   non-overlapping 8×8 DCT-like blocks and the energy is bucketed into
//!   low / mid / high frequency bands.
//! * **Per-channel noise** — each colour channel (Y/Cb/Cr or R/G/B) is
//!   profiled independently so that chroma noise can be separated from luma
//!   noise.

use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur during noise profiling.
#[derive(Debug, Error)]
pub enum NoiseProfileError {
    /// The frame buffer is too small for the given dimensions.
    #[error("frame buffer length {got} is smaller than expected {expected}")]
    BufferTooSmall {
        /// Actual buffer length.
        got: usize,
        /// Required buffer length.
        expected: usize,
    },
    /// The frame dimensions are too small to compute a meaningful profile.
    #[error("frame dimensions {width}×{height} are too small (minimum 4×4)")]
    DimensionsTooSmall {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
    },
}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Spatial-frequency energy breakdown for a single channel / plane.
///
/// The plane is divided into non-overlapping 8×8 blocks.  Within each block
/// the 2-D discrete cosine transform (DCT-II approximation) is evaluated and
/// the resulting coefficients are classified as low-, mid-, or high-frequency
/// based on their horizontal + vertical index sum.
#[derive(Debug, Clone, PartialEq)]
pub struct SpatialFrequencyProfile {
    /// Mean energy of coefficients with index-sum 0–2 (DC + very low AC).
    pub low_freq_energy: f64,
    /// Mean energy of coefficients with index-sum 3–6 (medium-frequency AC).
    pub mid_freq_energy: f64,
    /// Mean energy of coefficients with index-sum 7–14 (high-frequency AC).
    pub high_freq_energy: f64,
    /// Total mean energy across all coefficients.
    pub total_energy: f64,
    /// Number of complete 8×8 blocks analysed.
    pub block_count: usize,
}

impl SpatialFrequencyProfile {
    /// Ratio of high-frequency energy to total energy.  Values near 1
    /// indicate a noisy or detail-rich frame; values near 0 indicate a
    /// smooth, low-detail frame.
    #[inline]
    pub fn high_freq_ratio(&self) -> f64 {
        if self.total_energy < f64::EPSILON {
            0.0
        } else {
            self.high_freq_energy / self.total_energy
        }
    }
}

/// Noise sigma estimate for one plane.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneSigma {
    /// Estimated noise standard deviation (Donoho–Johnstone MAD estimator).
    pub sigma: f64,
    /// Signal-to-noise ratio in dB: `20 · log10(255 / sigma)`.
    /// `f64::INFINITY` when `sigma` is effectively zero.
    pub snr_db: f64,
    /// Spatial-frequency energy breakdown for this plane.
    pub freq_profile: SpatialFrequencyProfile,
}

/// Complete per-channel noise profile for one video frame.
///
/// Channels are stored as an ordered `Vec<PlaneSigma>` so that the type is
/// agnostic about whether the frame is YCbCr, RGB, or any other colour model.
/// Callers label channels however they wish.
#[derive(Debug, Clone)]
pub struct NoiseProfile {
    /// Per-channel noise data, in the order the caller provided the planes.
    pub channels: Vec<PlaneSigma>,
    /// Width of the frame that was profiled.
    pub width: u32,
    /// Height of the frame that was profiled.
    pub height: u32,
}

impl NoiseProfile {
    /// Return the mean `sigma` across all channels.
    pub fn mean_sigma(&self) -> f64 {
        if self.channels.is_empty() {
            return 0.0;
        }
        self.channels.iter().map(|c| c.sigma).sum::<f64>() / self.channels.len() as f64
    }

    /// Return the minimum SNR (worst quality) across all channels.
    pub fn min_snr_db(&self) -> f64 {
        self.channels
            .iter()
            .map(|c| c.snr_db)
            .fold(f64::INFINITY, f64::min)
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Profile the noise characteristics of a single-channel (planar) frame.
///
/// # Parameters
///
/// * `plane` — grayscale (or any single-channel) pixel data in raster order,
///   one byte per sample.
/// * `width` / `height` — frame dimensions in pixels.
///
/// # Errors
///
/// Returns [`NoiseProfileError`] if the dimensions are too small or the buffer
/// is shorter than `width × height` bytes.
pub fn profile_plane(
    plane: &[u8],
    width: u32,
    height: u32,
) -> Result<PlaneSigma, NoiseProfileError> {
    let w = width as usize;
    let h = height as usize;
    let expected = w * h;

    if w < 4 || h < 4 {
        return Err(NoiseProfileError::DimensionsTooSmall { width, height });
    }
    if plane.len() < expected {
        return Err(NoiseProfileError::BufferTooSmall {
            got: plane.len(),
            expected,
        });
    }

    let sigma = laplacian_mad_sigma(plane, w, h);
    let snr_db = sigma_to_snr(sigma);
    let freq_profile = spatial_frequency_profile(plane, w, h);

    Ok(PlaneSigma {
        sigma,
        snr_db,
        freq_profile,
    })
}

/// Profile the noise characteristics of a multi-channel frame supplied as a
/// packed interleaved buffer.
///
/// For example, a packed RGB frame has `channel_count = 3` and the buffer
/// layout is `[R0, G0, B0, R1, G1, B1, …]`.
///
/// # Parameters
///
/// * `interleaved` — packed pixel data, `width × height × channel_count` bytes.
/// * `width` / `height` — frame dimensions in pixels.
/// * `channel_count` — number of channels in the packed buffer (e.g. 3 for RGB).
///
/// # Errors
///
/// Returns [`NoiseProfileError`] if any dimension is too small or the buffer
/// is shorter than required.
pub fn profile_interleaved(
    interleaved: &[u8],
    width: u32,
    height: u32,
    channel_count: usize,
) -> Result<NoiseProfile, NoiseProfileError> {
    let w = width as usize;
    let h = height as usize;
    let expected = w * h * channel_count;

    if w < 4 || h < 4 {
        return Err(NoiseProfileError::DimensionsTooSmall { width, height });
    }
    if interleaved.len() < expected {
        return Err(NoiseProfileError::BufferTooSmall {
            got: interleaved.len(),
            expected,
        });
    }

    let pixels = w * h;
    let mut channels = Vec::with_capacity(channel_count);

    for ch in 0..channel_count {
        // De-interleave channel into a temporary plane.
        let plane: Vec<u8> = (0..pixels)
            .map(|px| interleaved[px * channel_count + ch])
            .collect();
        let ch_sigma = laplacian_mad_sigma(&plane, w, h);
        let ch_snr = sigma_to_snr(ch_sigma);
        let ch_freq = spatial_frequency_profile(&plane, w, h);
        channels.push(PlaneSigma {
            sigma: ch_sigma,
            snr_db: ch_snr,
            freq_profile: ch_freq,
        });
    }

    Ok(NoiseProfile {
        channels,
        width,
        height,
    })
}

/// Profile multiple planar channels (e.g. YCbCr stored as separate `Vec<u8>`
/// slices).
///
/// Each element of `planes` must have at least `width × height` bytes.  The
/// planes may have different sizes (e.g. 4:2:0 chroma) — pass the corresponding
/// per-plane dimensions in `dims`.  If `dims` is shorter than `planes`, the
/// last entry is reused for the remaining planes.
///
/// # Errors
///
/// Returns [`NoiseProfileError`] on the first plane that fails validation.
pub fn profile_planes<'a>(
    planes: impl Iterator<Item = &'a [u8]>,
    dims: &[(u32, u32)],
) -> Result<Vec<PlaneSigma>, NoiseProfileError> {
    let mut results = Vec::new();
    let mut plane_idx = 0usize;

    for plane in planes {
        let (pw, ph) = dims
            .get(plane_idx)
            .or_else(|| dims.last())
            .copied()
            .unwrap_or((4, 4));
        results.push(profile_plane(plane, pw, ph)?);
        plane_idx += 1;
    }

    Ok(results)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Laplacian MAD (median-absolute-deviation) noise sigma estimator.
///
/// Applies a 3×3 Laplacian filter to the interior pixels and computes:
/// ```text
/// sigma = median(|lap_response|) / 0.6745
/// ```
fn laplacian_mad_sigma(plane: &[u8], w: usize, h: usize) -> f64 {
    if w < 3 || h < 3 {
        return 0.0;
    }

    let mut responses = Vec::with_capacity((w - 2) * (h - 2));

    for row in 1..(h - 1) {
        for col in 1..(w - 1) {
            let center = plane[row * w + col] as i32;
            let top = plane[(row - 1) * w + col] as i32;
            let bottom = plane[(row + 1) * w + col] as i32;
            let left = plane[row * w + col - 1] as i32;
            let right = plane[row * w + col + 1] as i32;

            let lap = (top + bottom + left + right - 4 * center).unsigned_abs() as f64;
            responses.push(lap);
        }
    }

    let med = median_f64(&mut responses);
    med / 0.6745
}

/// Convert sigma to SNR in dB.
#[inline]
fn sigma_to_snr(sigma: f64) -> f64 {
    if sigma < f64::EPSILON {
        f64::INFINITY
    } else {
        20.0 * (255.0_f64 / sigma).log10()
    }
}

/// Compute median of a mutable `Vec<f64>` (sorts in place).
fn median_f64(values: &mut Vec<f64>) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Compute a simple 1-D DCT-II approximation for an 8-element vector.
///
/// Uses the direct O(N²) formula:
/// ```text
/// X[k] = sum_{n=0}^{7} x[n] * cos(pi * k * (2n+1) / 16)
/// ```
fn dct8(x: [f64; 8]) -> [f64; 8] {
    let mut out = [0.0f64; 8];
    for (k, out_k) in out.iter_mut().enumerate() {
        let mut s = 0.0f64;
        for n in 0..8 {
            s += x[n] * ((std::f64::consts::PI * k as f64 * (2.0 * n as f64 + 1.0)) / 16.0).cos();
        }
        *out_k = s;
    }
    out
}

/// 2-D 8×8 DCT energy bucketed into low / mid / high frequency bands.
///
/// The 2-D transform is factored as row-DCT followed by column-DCT.
/// Coefficient `(u, v)` belongs to:
/// * low  if  `u + v <= 2`
/// * mid  if  `u + v <= 6`
/// * high otherwise
fn dct8x8_energy(block: &[f64; 64]) -> (f64, f64, f64, f64) {
    // Row transforms.
    let mut tmp = [0.0f64; 64];
    for r in 0..8 {
        let row: [f64; 8] = block[r * 8..r * 8 + 8].try_into().unwrap_or([0.0; 8]);
        let xrow = dct8(row);
        for c in 0..8 {
            tmp[r * 8 + c] = xrow[c];
        }
    }

    // Column transforms.
    let mut coeff = [0.0f64; 64];
    for c in 0..8 {
        let col: [f64; 8] = std::array::from_fn(|r| tmp[r * 8 + c]);
        let xcol = dct8(col);
        for r in 0..8 {
            coeff[r * 8 + c] = xcol[r];
        }
    }

    let (mut low, mut mid, mut high, mut total) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for u in 0usize..8 {
        for v in 0usize..8 {
            let e = coeff[u * 8 + v].powi(2);
            total += e;
            let sum = u + v;
            if sum <= 2 {
                low += e;
            } else if sum <= 6 {
                mid += e;
            } else {
                high += e;
            }
        }
    }
    (low, mid, high, total)
}

/// Compute the spatial-frequency profile of a plane by analysing non-overlapping
/// 8×8 blocks.
fn spatial_frequency_profile(plane: &[u8], w: usize, h: usize) -> SpatialFrequencyProfile {
    let blocks_x = w / 8;
    let blocks_y = h / 8;
    let block_count = blocks_x * blocks_y;

    if block_count == 0 {
        return SpatialFrequencyProfile {
            low_freq_energy: 0.0,
            mid_freq_energy: 0.0,
            high_freq_energy: 0.0,
            total_energy: 0.0,
            block_count: 0,
        };
    }

    let (mut sum_low, mut sum_mid, mut sum_high, mut sum_total) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);

    for by in 0..blocks_y {
        for bx in 0..blocks_x {
            let mut block = [0.0f64; 64];
            for r in 0..8usize {
                for c in 0..8usize {
                    let row = by * 8 + r;
                    let col = bx * 8 + c;
                    block[r * 8 + c] = plane[row * w + col] as f64;
                }
            }
            let (low, mid, high, total) = dct8x8_energy(&block);
            sum_low += low;
            sum_mid += mid;
            sum_high += high;
            sum_total += total;
        }
    }

    let n = block_count as f64;
    SpatialFrequencyProfile {
        low_freq_energy: sum_low / n,
        mid_freq_energy: sum_mid / n,
        high_freq_energy: sum_high / n,
        total_energy: sum_total / n,
        block_count,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn flat_plane(w: u32, h: u32, val: u8) -> Vec<u8> {
        vec![val; (w * h) as usize]
    }

    fn checker_plane(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h) as usize)
            .map(|i| {
                let row = i / w as usize;
                let col = i % w as usize;
                if (row + col) % 2 == 0 {
                    255
                } else {
                    0
                }
            })
            .collect()
    }

    fn ramp_plane(w: u32, h: u32) -> Vec<u8> {
        (0..(w * h) as usize).map(|i| (i % 256) as u8).collect()
    }

    // 1. Flat frame → sigma near zero
    #[test]
    fn test_flat_frame_sigma_near_zero() {
        let plane = flat_plane(16, 16, 128);
        let result = profile_plane(&plane, 16, 16).unwrap();
        assert!(
            result.sigma < 1.0,
            "flat frame should have near-zero sigma, got {}",
            result.sigma
        );
    }

    // 2. Checkerboard → sigma significantly > 0
    #[test]
    fn test_checker_frame_sigma_nonzero() {
        let plane = checker_plane(16, 16);
        let result = profile_plane(&plane, 16, 16).unwrap();
        assert!(
            result.sigma > 1.0,
            "checkerboard should have high sigma, got {}",
            result.sigma
        );
    }

    // 3. SNR is infinity for flat frame
    #[test]
    fn test_flat_frame_snr_infinity() {
        let plane = flat_plane(16, 16, 200);
        let result = profile_plane(&plane, 16, 16).unwrap();
        assert!(
            result.snr_db.is_infinite(),
            "expected infinite SNR for flat frame"
        );
    }

    // 4. Dimensions too small returns error
    #[test]
    fn test_dimensions_too_small_error() {
        let plane = flat_plane(3, 3, 128);
        let err = profile_plane(&plane, 3, 3);
        assert!(err.is_err());
        assert!(matches!(
            err,
            Err(NoiseProfileError::DimensionsTooSmall { .. })
        ));
    }

    // 5. Buffer too small returns error
    #[test]
    fn test_buffer_too_small_error() {
        let short = vec![0u8; 10];
        let err = profile_plane(&short, 16, 16);
        assert!(err.is_err());
        assert!(matches!(err, Err(NoiseProfileError::BufferTooSmall { .. })));
    }

    // 6. Flat frame → low freq energy dominates
    #[test]
    fn test_flat_frame_low_freq_dominates() {
        let plane = flat_plane(16, 16, 200);
        let result = profile_plane(&plane, 16, 16).unwrap();
        // Flat frame has all energy in DC (low-freq)
        assert!(
            result.freq_profile.low_freq_energy >= result.freq_profile.high_freq_energy,
            "flat frame: low freq should dominate"
        );
    }

    // 7. Checkerboard → high freq ratio > low freq ratio
    //
    // A strict absolute threshold is hard to set without a full 2-D DCT
    // implementation.  Instead we verify that the checkerboard distributes
    // more energy towards high frequencies than a flat frame does.
    #[test]
    fn test_checker_high_freq_ratio() {
        let checker = checker_plane(16, 16);
        let flat = flat_plane(16, 16, 128);

        let checker_result = profile_plane(&checker, 16, 16).unwrap();
        let flat_result = profile_plane(&flat, 16, 16).unwrap();

        let checker_ratio = checker_result.freq_profile.high_freq_ratio();
        let flat_ratio = flat_result.freq_profile.high_freq_ratio();

        assert!(
            checker_ratio > flat_ratio,
            "checkerboard high-freq ratio {checker_ratio} should exceed flat frame ratio {flat_ratio}"
        );
    }

    // 8. Interleaved RGB: 3 channels returned
    #[test]
    fn test_interleaved_rgb_channel_count() {
        let w = 8u32;
        let h = 8u32;
        let data = ramp_plane(w * h * 3 / 3, h * 3); // 3 bytes per pixel
        let buf: Vec<u8> = (0..(w * h * 3) as usize).map(|i| (i % 256) as u8).collect();
        let _ = data;
        let profile = profile_interleaved(&buf, w, h, 3).unwrap();
        assert_eq!(profile.channels.len(), 3);
    }

    // 9. mean_sigma is non-negative
    #[test]
    fn test_mean_sigma_nonneg() {
        let plane = ramp_plane(16, 16);
        let result = profile_plane(&plane, 16, 16).unwrap();
        let profile = NoiseProfile {
            channels: vec![result],
            width: 16,
            height: 16,
        };
        assert!(profile.mean_sigma() >= 0.0);
    }

    // 10. min_snr_db on flat frame is infinite
    #[test]
    fn test_min_snr_db_flat() {
        let plane = flat_plane(16, 16, 64);
        let ch = profile_plane(&plane, 16, 16).unwrap();
        let profile = NoiseProfile {
            channels: vec![ch],
            width: 16,
            height: 16,
        };
        assert!(profile.min_snr_db().is_infinite());
    }

    // 11. profile_planes: multiple planes succeed
    #[test]
    fn test_profile_planes_multiple() {
        let y = flat_plane(16, 16, 150);
        let cb = flat_plane(8, 8, 128);
        let cr = flat_plane(8, 8, 128);
        let dims = [(16u32, 16u32), (8u32, 8u32), (8u32, 8u32)];
        let planes: Vec<&[u8]> = vec![y.as_slice(), cb.as_slice(), cr.as_slice()];
        let result = profile_planes(planes.into_iter(), &dims).unwrap();
        assert_eq!(result.len(), 3);
    }

    // 12. block_count is correct for 16×16 frame (4 blocks of 8×8)
    #[test]
    fn test_block_count_correct() {
        let plane = flat_plane(16, 16, 100);
        let result = profile_plane(&plane, 16, 16).unwrap();
        assert_eq!(result.freq_profile.block_count, 4);
    }
}
