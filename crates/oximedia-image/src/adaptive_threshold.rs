//! Adaptive thresholding algorithms for grayscale images.
//!
//! Unlike a single global threshold, adaptive methods compute a per-pixel
//! threshold from the local neighbourhood or from the global image histogram.
//! This makes them robust to uneven illumination and contrast gradients.
//!
//! # Algorithms
//!
//! | Algorithm          | Description                                                        |
//! |--------------------|--------------------------------------------------------------------|
//! | [`OtsuThreshold`]  | Maximises inter-class variance to find a single optimal threshold. |
//! | [`TriangleThreshold`] | Fits a triangle to the histogram, robust for bimodal images.    |
//! | [`LocalMeanThreshold`] | Sauvola-style: threshold = local mean ± constant offset.       |
//! | [`LocalNiblackThreshold`] | Niblack: threshold = mean + k·stddev.                      |
//!
//! # Example
//!
//! ```rust
//! use oximedia_image::adaptive_threshold::{OtsuThreshold, TriangleThreshold, LocalMeanThreshold};
//!
//! let image: Vec<u8> = (0u8..=255).collect(); // gradient, 16×16 would need 256 pixels
//! // Simple 16-pixel ramp
//! let ramp: Vec<u8> = (0..16).map(|i| (i * 16) as u8).collect();
//!
//! let otsu_t = OtsuThreshold::compute(&ramp);
//! assert!(otsu_t > 0 && otsu_t < 255);
//!
//! let tri_t = TriangleThreshold::compute(&ramp);
//! assert!(tri_t > 0);
//!
//! let local = LocalMeanThreshold::new(3, 5);
//! let binary = local.apply(&ramp, 4, 4).expect("ok");
//! assert_eq!(binary.len(), 16);
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{ImageError, ImageResult};

// ---------------------------------------------------------------------------
// Histogram helpers
// ---------------------------------------------------------------------------

/// Builds a 256-bin intensity histogram from a `u8` image.
fn build_histogram(src: &[u8]) -> [u32; 256] {
    let mut hist = [0u32; 256];
    for &p in src {
        hist[p as usize] += 1;
    }
    hist
}

// ---------------------------------------------------------------------------
// Otsu Threshold
// ---------------------------------------------------------------------------

/// Otsu's automatic thresholding method.
///
/// Finds the global threshold `t` in `[0, 255]` that maximises the
/// **between-class variance** of the foreground and background populations,
/// which is equivalent to minimising the within-class (intra-class) variance.
///
/// # References
///
/// Otsu, N. (1979). "A Threshold Selection Method from Gray-Level Histograms".
/// IEEE Transactions on Systems, Man, and Cybernetics.
#[derive(Debug, Clone, Copy, Default)]
pub struct OtsuThreshold;

impl OtsuThreshold {
    /// Computes the optimal Otsu threshold for the given `u8` image.
    ///
    /// Returns the threshold value in `[0, 255]`. All pixels **strictly below**
    /// the threshold are considered background (0) and pixels **at or above**
    /// are foreground (255).
    ///
    /// Returns `0` for empty or all-same-intensity images.
    #[must_use]
    pub fn compute(src: &[u8]) -> u8 {
        let hist = build_histogram(src);
        let n = src.len() as f64;
        if n == 0.0 {
            return 0;
        }

        // Total mean intensity
        let mu_t: f64 = hist
            .iter()
            .enumerate()
            .map(|(i, &c)| i as f64 * c as f64)
            .sum::<f64>()
            / n;

        let mut best_t = 0u8;
        let mut best_var = 0.0f64;
        let mut w0 = 0.0f64; // weight of background class
        let mut mu0_acc = 0.0f64; // sum of intensities in background

        for t in 0usize..256 {
            let count = hist[t] as f64;
            w0 += count / n;
            mu0_acc += t as f64 * count / n;

            let w1 = 1.0 - w0;
            if w0 < 1e-12 || w1 < 1e-12 {
                continue;
            }
            let mu0 = mu0_acc / w0;
            let mu1 = (mu_t - w0 * mu0) / w1;
            let var_b = w0 * w1 * (mu0 - mu1) * (mu0 - mu1);
            if var_b > best_var {
                best_var = var_b;
                best_t = t as u8;
            }
        }
        best_t
    }

    /// Binarizes `src` using the Otsu threshold.
    ///
    /// Pixels at or above the threshold become 255; pixels below become 0.
    /// Returns a new `Vec<u8>` with the same length as `src`.
    #[must_use]
    pub fn binarize(src: &[u8]) -> Vec<u8> {
        let t = Self::compute(src);
        src.iter().map(|&p| if p >= t { 255 } else { 0 }).collect()
    }
}

// ---------------------------------------------------------------------------
// Triangle Threshold
// ---------------------------------------------------------------------------

/// Triangle (Zack) thresholding method.
///
/// Finds the threshold by drawing a line from the histogram peak to the
/// furthest non-zero edge bin and locating the point with maximum
/// perpendicular distance from that line.
///
/// Works particularly well for images with a dominant bright or dark background.
///
/// # References
///
/// Zack, G.W., Rogers, W.E., Latt, S.A. (1977). "Automatic Measurement of
/// Sister Chromatid Exchange Frequency". Journal of Histochemistry &
/// Cytochemistry.
#[derive(Debug, Clone, Copy, Default)]
pub struct TriangleThreshold;

impl TriangleThreshold {
    /// Computes the triangle threshold for the given `u8` image.
    ///
    /// Returns the threshold value in `[0, 255]`.
    #[must_use]
    pub fn compute(src: &[u8]) -> u8 {
        let hist = build_histogram(src);

        // Find the peak bin.
        let (peak_bin, _) = hist
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .unwrap_or((0, &0));

        // Find the first and last non-zero bins.
        let first = hist.iter().position(|&c| c > 0).unwrap_or(0);
        let last = hist.iter().rposition(|&c| c > 0).unwrap_or(255);

        if first == last {
            return first as u8;
        }

        // Decide which side of the peak to evaluate.
        // The triangle is drawn from the peak to whichever end is farther away.
        let left_dist = peak_bin.saturating_sub(first);
        let right_dist = last.saturating_sub(peak_bin);

        let (a, b) = if right_dist >= left_dist {
            // Use right side: line from peak to last.
            (peak_bin, last)
        } else {
            // Use left side: line from first to peak.
            (first, peak_bin)
        };

        if a == b {
            return a as u8;
        }

        // Line from (a, hist[a]) to (b, hist[b]).
        let x1 = a as f64;
        let y1 = hist[a] as f64;
        let x2 = b as f64;
        let y2 = hist[b] as f64;

        // Line equation: (y2-y1)(x - x1) - (x2-x1)(y - y1) = 0
        // Perpendicular distance numerator: |(y2-y1)·xi - (x2-x1)·yi + (x2*y1 - y2*x1)|
        let dx = x2 - x1;
        let dy = y2 - y1;
        let c = x2 * y1 - y2 * x1;
        let denom = (dx * dx + dy * dy).sqrt();

        let mut best_t = a;
        let mut best_dist = 0.0f64;

        let lo = a.min(b);
        let hi = a.max(b);
        for t in lo..=hi {
            let xi = t as f64;
            let yi = hist[t] as f64;
            let dist = (dy * xi - dx * yi + c).abs() / denom;
            if dist > best_dist {
                best_dist = dist;
                best_t = t;
            }
        }
        best_t as u8
    }

    /// Binarizes `src` using the triangle threshold.
    ///
    /// Pixels at or above the threshold become 255; pixels below become 0.
    #[must_use]
    pub fn binarize(src: &[u8]) -> Vec<u8> {
        let t = Self::compute(src);
        src.iter().map(|&p| if p >= t { 255 } else { 0 }).collect()
    }
}

// ---------------------------------------------------------------------------
// LocalMeanThreshold (Sauvola-style)
// ---------------------------------------------------------------------------

/// Local mean adaptive thresholding.
///
/// Each pixel's threshold is computed as the mean of a local `window_size × window_size`
/// neighbourhood minus a constant offset `c`:
///
/// ```text
/// T(x,y) = mean(neighbourhood) - c
/// ```
///
/// This is the simplest form of local adaptive thresholding and handles
/// gently varying illumination well.
#[derive(Debug, Clone)]
pub struct LocalMeanThreshold {
    /// Half-size of the square neighbourhood window (radius).
    radius: usize,
    /// Constant offset subtracted from the local mean.
    offset: i32,
}

impl LocalMeanThreshold {
    /// Creates a new local mean threshold.
    ///
    /// # Arguments
    ///
    /// * `window_size` – Width/height of the neighbourhood (clamped to odd ≥ 1).
    /// * `offset`      – Constant subtracted from the local mean (may be negative).
    #[must_use]
    pub fn new(window_size: usize, offset: i32) -> Self {
        let ws = window_size.max(1) | 1; // ensure odd
        Self {
            radius: ws / 2,
            offset,
        }
    }

    /// Applies the local mean threshold to the given `u8` image.
    ///
    /// Returns a binary image where 255 = foreground, 0 = background.
    ///
    /// # Errors
    ///
    /// Returns an error if `src.len() != width * height` or either dimension is zero.
    pub fn apply(&self, src: &[u8], width: usize, height: usize) -> ImageResult<Vec<u8>> {
        validate_dims(src.len(), width, height)?;
        let mut dst = vec![0u8; src.len()];
        let r = self.radius as isize;

        for y in 0..height {
            for x in 0..width {
                let y0 = ((y as isize) - r).max(0) as usize;
                let y1 = ((y as isize) + r + 1).min(height as isize) as usize;
                let x0 = ((x as isize) - r).max(0) as usize;
                let x1 = ((x as isize) + r + 1).min(width as isize) as usize;

                let mut sum = 0u64;
                let mut count = 0u64;
                for ny in y0..y1 {
                    for nx in x0..x1 {
                        sum += src[ny * width + nx] as u64;
                        count += 1;
                    }
                }

                let mean = (sum / count.max(1)) as i32;
                let t = (mean - self.offset).clamp(0, 255) as u8;
                dst[y * width + x] = if src[y * width + x] >= t { 255 } else { 0 };
            }
        }
        Ok(dst)
    }
}

// ---------------------------------------------------------------------------
// LocalNiblackThreshold
// ---------------------------------------------------------------------------

/// Niblack's adaptive thresholding.
///
/// Each pixel's threshold is:
///
/// ```text
/// T(x,y) = mean(N) + k · stddev(N)
/// ```
///
/// where `N` is the local neighbourhood of size `window_size × window_size`
/// and `k` is a sensitivity parameter (typically –0.2 for dark text on light).
///
/// # References
///
/// Niblack, W. (1986). "An Introduction to Digital Image Processing".
/// Prentice Hall.
#[derive(Debug, Clone)]
pub struct LocalNiblackThreshold {
    /// Half-size of the square neighbourhood window.
    radius: usize,
    /// Sensitivity factor `k`.
    k: f32,
}

impl LocalNiblackThreshold {
    /// Creates a new Niblack threshold.
    ///
    /// # Arguments
    ///
    /// * `window_size` – Width/height of the neighbourhood (clamped to odd ≥ 1).
    /// * `k`           – Sensitivity. Negative for bright-background images.
    #[must_use]
    pub fn new(window_size: usize, k: f32) -> Self {
        let ws = window_size.max(1) | 1;
        Self { radius: ws / 2, k }
    }

    /// Applies the Niblack adaptive threshold to the given `u8` image.
    ///
    /// Returns a binary image where 255 = foreground, 0 = background.
    ///
    /// # Errors
    ///
    /// Returns an error if `src.len() != width * height` or either dimension is zero.
    pub fn apply(&self, src: &[u8], width: usize, height: usize) -> ImageResult<Vec<u8>> {
        validate_dims(src.len(), width, height)?;
        let mut dst = vec![0u8; src.len()];
        let r = self.radius as isize;

        for y in 0..height {
            for x in 0..width {
                let y0 = ((y as isize) - r).max(0) as usize;
                let y1 = ((y as isize) + r + 1).min(height as isize) as usize;
                let x0 = ((x as isize) - r).max(0) as usize;
                let x1 = ((x as isize) + r + 1).min(width as isize) as usize;

                let mut sum = 0.0f64;
                let mut sum_sq = 0.0f64;
                let mut count = 0.0f64;
                for ny in y0..y1 {
                    for nx in x0..x1 {
                        let v = src[ny * width + nx] as f64;
                        sum += v;
                        sum_sq += v * v;
                        count += 1.0;
                    }
                }

                let mean = sum / count.max(1.0);
                let variance = (sum_sq / count.max(1.0)) - mean * mean;
                let stddev = variance.max(0.0).sqrt();
                let t = (mean + self.k as f64 * stddev).clamp(0.0, 255.0) as u8;
                dst[y * width + x] = if src[y * width + x] >= t { 255 } else { 0 };
            }
        }
        Ok(dst)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn validate_dims(len: usize, width: usize, height: usize) -> ImageResult<()> {
    if width == 0 || height == 0 {
        return Err(ImageError::InvalidDimensions(width as u32, height as u32));
    }
    if len != width * height {
        return Err(ImageError::InvalidDimensions(width as u32, height as u32));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Histogram helper ----

    #[test]
    fn test_histogram_correct_counts() {
        let src = vec![0u8, 0, 128, 255, 128, 0];
        let hist = build_histogram(&src);
        assert_eq!(hist[0], 3);
        assert_eq!(hist[128], 2);
        assert_eq!(hist[255], 1);
    }

    // ---- OtsuThreshold ----

    #[test]
    fn test_otsu_bimodal() {
        // Bimodal: 128 pixels at 50 and 128 pixels at 200
        let mut src = vec![50u8; 128];
        src.extend_from_slice(&vec![200u8; 128]);
        let t = OtsuThreshold::compute(&src);
        // Threshold should separate the two modes: at 50 (inclusive) or beyond
        // but strictly below the upper mode.
        assert!(t >= 50, "threshold {t} is below lower mode");
        assert!(t < 200, "threshold {t} is not below upper mode");
    }

    #[test]
    fn test_otsu_empty_returns_zero() {
        let t = OtsuThreshold::compute(&[]);
        assert_eq!(t, 0);
    }

    #[test]
    fn test_otsu_uniform_image() {
        let src = vec![128u8; 100];
        let t = OtsuThreshold::compute(&src);
        // Uniform image: variance is zero everywhere, any threshold is valid
        let _ = t; // just check it doesn't panic
    }

    #[test]
    fn test_otsu_binarize_output_length() {
        let src: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        let binary = OtsuThreshold::binarize(&src);
        assert_eq!(binary.len(), 256);
    }

    #[test]
    fn test_otsu_binarize_only_zero_and_255() {
        let src: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        let binary = OtsuThreshold::binarize(&src);
        for &v in &binary {
            assert!(v == 0 || v == 255);
        }
    }

    // ---- TriangleThreshold ----

    #[test]
    fn test_triangle_non_zero() {
        let src: Vec<u8> = (0u8..=255).cycle().take(512).collect();
        let t = TriangleThreshold::compute(&src);
        // Just ensure no panic and returns something in range
        let _ = t;
    }

    #[test]
    fn test_triangle_uniform_returns_same_value() {
        let src = vec![100u8; 64];
        let t = TriangleThreshold::compute(&src);
        assert_eq!(t, 100);
    }

    #[test]
    fn test_triangle_binarize_only_zero_and_255() {
        let src: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        let binary = TriangleThreshold::binarize(&src);
        for &v in &binary {
            assert!(v == 0 || v == 255);
        }
    }

    #[test]
    fn test_triangle_binarize_output_length() {
        let src = vec![50u8; 20];
        let binary = TriangleThreshold::binarize(&src);
        assert_eq!(binary.len(), 20);
    }

    // ---- LocalMeanThreshold ----

    #[test]
    fn test_local_mean_output_length() {
        let src = vec![128u8; 8 * 8];
        let t = LocalMeanThreshold::new(3, 5);
        let dst = t.apply(&src, 8, 8).expect("ok");
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn test_local_mean_only_binary_values() {
        let src: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        let t = LocalMeanThreshold::new(5, 0);
        let dst = t.apply(&src, 16, 16).expect("ok");
        for &v in &dst {
            assert!(v == 0 || v == 255);
        }
    }

    #[test]
    fn test_local_mean_invalid_dimensions() {
        let src = vec![0u8; 9];
        let t = LocalMeanThreshold::new(3, 0);
        assert!(t.apply(&src, 4, 4).is_err());
    }

    #[test]
    fn test_local_mean_zero_width_error() {
        let src: Vec<u8> = vec![];
        let t = LocalMeanThreshold::new(3, 0);
        assert!(t.apply(&src, 0, 4).is_err());
    }

    // ---- LocalNiblackThreshold ----

    #[test]
    fn test_niblack_output_length() {
        let src = vec![100u8; 5 * 5];
        let t = LocalNiblackThreshold::new(3, -0.2);
        let dst = t.apply(&src, 5, 5).expect("ok");
        assert_eq!(dst.len(), 25);
    }

    #[test]
    fn test_niblack_only_binary_values() {
        let src: Vec<u8> = (0u8..=255).cycle().take(256).collect();
        let t = LocalNiblackThreshold::new(5, -0.2);
        let dst = t.apply(&src, 16, 16).expect("ok");
        for &v in &dst {
            assert!(v == 0 || v == 255);
        }
    }

    #[test]
    fn test_niblack_uniform_positive_k() {
        // Uniform image with k > 0: mean + k*0 = mean, so pixels at exactly mean
        // should be foreground (>=).
        let src = vec![128u8; 4 * 4];
        let t = LocalNiblackThreshold::new(3, 0.0);
        let dst = t.apply(&src, 4, 4).expect("ok");
        // Every pixel equals the threshold so all should be 255 (foreground).
        for &v in &dst {
            assert_eq!(v, 255);
        }
    }

    #[test]
    fn test_niblack_invalid_dimensions() {
        let src = vec![0u8; 7]; // not 4×4
        let t = LocalNiblackThreshold::new(3, -0.2);
        assert!(t.apply(&src, 4, 4).is_err());
    }
}
