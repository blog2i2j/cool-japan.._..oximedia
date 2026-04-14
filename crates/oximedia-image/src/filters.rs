//! Image filter operations: convolution, thresholding, histogram equalization,
//! median filtering, and bilateral filtering for professional image processing
//! pipelines.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// A 2-D convolution kernel stored in row-major order.
#[derive(Debug, Clone)]
pub struct ConvolutionKernel {
    /// Kernel coefficients in row-major order.
    pub data: Vec<f32>,
    /// Kernel width in pixels.
    pub width: usize,
    /// Kernel height in pixels.
    pub height: usize,
}

impl ConvolutionKernel {
    /// Create a new kernel from raw data.
    ///
    /// # Panics
    ///
    /// Panics if `data.len() != width * height`.
    #[must_use]
    pub fn new(data: Vec<f32>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height, "kernel data length mismatch");
        Self {
            data,
            width,
            height,
        }
    }

    /// Divide every coefficient by the sum so the kernel preserves mean brightness.
    ///
    /// If the sum is zero the kernel is left unchanged.
    pub fn normalize(&mut self) {
        let sum: f32 = self.data.iter().sum();
        if sum.abs() > 1e-9 {
            for v in &mut self.data {
                *v /= sum;
            }
        }
    }

    /// Build a separable Gaussian kernel of the given `size` (must be odd) and `sigma`.
    #[must_use]
    pub fn gaussian(sigma: f32, size: usize) -> Self {
        let size = if size % 2 == 0 { size + 1 } else { size };
        let half = (size / 2) as i32;
        let mut data = Vec::with_capacity(size * size);
        for ky in -(half)..=half {
            for kx in -(half)..=half {
                let exp = -((kx * kx + ky * ky) as f32) / (2.0 * sigma * sigma);
                data.push(exp.exp());
            }
        }
        let mut k = Self {
            data,
            width: size,
            height: size,
        };
        k.normalize();
        k
    }

    /// Standard 3x3 unsharp / detail-enhance kernel.
    #[must_use]
    pub fn sharpen() -> Self {
        Self::new(vec![0.0, -1.0, 0.0, -1.0, 5.0, -1.0, 0.0, -1.0, 0.0], 3, 3)
    }

    /// Classic 3x3 emboss kernel.
    #[must_use]
    pub fn emboss() -> Self {
        Self::new(vec![-2.0, -1.0, 0.0, -1.0, 1.0, 1.0, 0.0, 1.0, 2.0], 3, 3)
    }
}

/// Apply a 2-D convolution kernel to a single-channel `f32` image.
///
/// `src` and `dst` must both have length `width * height`. Border pixels are
/// handled via clamp-to-edge.
pub fn apply_convolution(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    kernel: &ConvolutionKernel,
) {
    let kw = kernel.width as i64;
    let kh = kernel.height as i64;
    let hw = kw / 2;
    let hh = kh / 2;
    let iw = width as i64;
    let ih = height as i64;

    for py in 0..ih {
        for px in 0..iw {
            let mut acc = 0.0f32;
            for ky in 0..kh {
                for kx in 0..kw {
                    let sx = (px + kx - hw).clamp(0, iw - 1) as usize;
                    let sy = (py + ky - hh).clamp(0, ih - 1) as usize;
                    let coeff = kernel.data[(ky * kw + kx) as usize];
                    acc += coeff * src[sy * width + sx];
                }
            }
            dst[py as usize * width + px as usize] = acc;
        }
    }
}

/// Binarize a single-channel `u8` image: pixels < `threshold` to 0, else to 255.
pub fn threshold(src: &[u8], dst: &mut [u8], thr: u8) {
    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = if s < thr { 0 } else { 255 };
    }
}

/// Global histogram equalization for a single-channel `u8` image.
///
/// Remaps pixel intensities so that the cumulative distribution is linear.
pub fn equalize_histogram(src: &[u8], dst: &mut [u8]) {
    let n = src.len();
    if n == 0 {
        return;
    }

    let mut hist = [0u32; 256];
    for &p in src {
        hist[p as usize] += 1;
    }

    let mut cdf = [0u32; 256];
    cdf[0] = hist[0];
    for i in 1..256 {
        cdf[i] = cdf[i - 1] + hist[i];
    }

    let cdf_min = *cdf.iter().find(|&&v| v > 0).unwrap_or(&0);

    let range = (n as u32).saturating_sub(cdf_min);
    let mut lut = [0u8; 256];
    for (i, lut_val) in lut.iter_mut().enumerate() {
        if range == 0 {
            *lut_val = i as u8;
        } else {
            let eq = (cdf[i].saturating_sub(cdf_min) as f32 / range as f32 * 255.0)
                .clamp(0.0, 255.0)
                .round() as u8;
            *lut_val = eq;
        }
    }

    for (d, &s) in dst.iter_mut().zip(src.iter()) {
        *d = lut[s as usize];
    }
}

/// 3x3 median filter for a single-channel `u8` image.
///
/// Border pixels are clamped to the nearest valid source pixel.
pub fn median_filter_3x3(src: &[u8], dst: &mut [u8], width: usize, height: usize) {
    let iw = width as i64;
    let ih = height as i64;

    for py in 0..ih {
        for px in 0..iw {
            let mut window = [0u8; 9];
            let mut count = 0usize;
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let sy = (py + dy).clamp(0, ih - 1) as usize;
                    let sx = (px + dx).clamp(0, iw - 1) as usize;
                    window[count] = src[sy * width + sx];
                    count += 1;
                }
            }
            window[..count].sort_unstable();
            dst[py as usize * width + px as usize] = window[count / 2];
        }
    }
}

// ---------------------------------------------------------------------------
// Bilateral filter — edge-preserving denoising
// ---------------------------------------------------------------------------

/// Parameters for the bilateral filter.
///
/// The bilateral filter replaces each pixel value with a weighted average of
/// nearby pixels. The weights depend on both spatial distance (Gaussian kernel
/// with std-dev `sigma_spatial`) and intensity similarity (Gaussian kernel
/// with std-dev `sigma_range`, expressed on a 0-255 scale).
///
/// * Large `sigma_spatial`  — more spatial smoothing (wider neighbourhood).
/// * Large `sigma_range`    — less edge preservation (more intensity blurring).
#[derive(Debug, Clone, Copy)]
pub struct BilateralParams {
    /// Spatial Gaussian std-dev in pixels.
    pub sigma_spatial: f32,
    /// Range/intensity Gaussian std-dev (0-255 scale).
    pub sigma_range: f32,
    /// Half-size of the search window. Auto-computed as `ceil(3*sigma_spatial)` when 0.
    pub window_radius: usize,
}

impl BilateralParams {
    /// Creates parameters with explicit values.
    #[must_use]
    pub fn new(sigma_spatial: f32, sigma_range: f32, window_radius: usize) -> Self {
        Self {
            sigma_spatial,
            sigma_range,
            window_radius,
        }
    }

    /// Creates parameters and derives the window radius automatically as
    /// `ceil(3 * sigma_spatial).max(1)`.
    #[must_use]
    pub fn auto(sigma_spatial: f32, sigma_range: f32) -> Self {
        let radius = ((3.0 * sigma_spatial).ceil() as usize).max(1);
        Self::new(sigma_spatial, sigma_range, radius)
    }
}

impl Default for BilateralParams {
    fn default() -> Self {
        Self::auto(3.0, 30.0)
    }
}

/// Apply a bilateral filter to a single-channel `f32` image in `[0, 1]`.
///
/// Both `src` and `dst` must have length `width * height`.
///
/// # Panics
///
/// Panics if slice lengths do not match `width * height`.
pub fn bilateral_filter_f32(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    params: &BilateralParams,
) {
    assert_eq!(src.len(), width * height, "src length mismatch");
    assert_eq!(dst.len(), width * height, "dst length mismatch");

    let r = params.window_radius as i64;
    let two_ss_sq = 2.0 * params.sigma_spatial * params.sigma_spatial;
    let two_sr_sq = 2.0 * params.sigma_range * params.sigma_range;
    let iw = width as i64;
    let ih = height as i64;

    for cy in 0..ih {
        for cx in 0..iw {
            let center_val = src[(cy * iw + cx) as usize];
            let mut weight_sum = 0.0_f32;
            let mut val_sum = 0.0_f32;

            for dy in -r..=r {
                for dx in -r..=r {
                    let sx = (cx + dx).clamp(0, iw - 1);
                    let sy = (cy + dy).clamp(0, ih - 1);
                    let neighbor_val = src[(sy * iw + sx) as usize];
                    let spatial_dist_sq = (dx * dx + dy * dy) as f32;
                    // Convert range distance from [0,1] to 0-255 scale
                    let range_dist = (neighbor_val - center_val) * 255.0;
                    let spatial_w = (-spatial_dist_sq / two_ss_sq).exp();
                    let range_w = (-(range_dist * range_dist) / two_sr_sq).exp();
                    let w = spatial_w * range_w;
                    weight_sum += w;
                    val_sum += w * neighbor_val;
                }
            }

            dst[(cy * iw + cx) as usize] = if weight_sum > 1e-12 {
                val_sum / weight_sum
            } else {
                center_val
            };
        }
    }
}

/// Apply a bilateral filter to a single-channel `u8` image.
///
/// Converts to/from `f32 [0,1]` internally.
///
/// # Panics
///
/// Panics if slice lengths do not match `width * height`.
pub fn bilateral_filter_u8(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    params: &BilateralParams,
) {
    assert_eq!(src.len(), width * height, "src length mismatch");
    assert_eq!(dst.len(), width * height, "dst length mismatch");

    let src_f32: Vec<f32> = src.iter().map(|&v| v as f32 / 255.0).collect();
    let mut dst_f32 = vec![0.0_f32; width * height];
    bilateral_filter_f32(&src_f32, &mut dst_f32, width, height, params);
    for (d, &v) in dst.iter_mut().zip(dst_f32.iter()) {
        *d = (v * 255.0).clamp(0.0, 255.0).round() as u8;
    }
}

/// Apply a bilateral filter to an interleaved multi-channel `f32` image.
///
/// `channels` is the number of interleaved channels (e.g. 3 for RGB).
/// Spatial smoothing uses the mean channel (luminance) for the range kernel.
///
/// # Panics
///
/// Panics if `src.len() != width * height * channels`, `dst` has a different
/// length, or `channels == 0`.
pub fn bilateral_filter_rgb(
    src: &[f32],
    dst: &mut [f32],
    width: usize,
    height: usize,
    channels: usize,
    params: &BilateralParams,
) {
    let n = width * height;
    assert_eq!(src.len(), n * channels, "src length mismatch");
    assert_eq!(dst.len(), n * channels, "dst length mismatch");
    assert!(channels > 0, "channels must be > 0");

    let r = params.window_radius as i64;
    let two_ss_sq = 2.0 * params.sigma_spatial * params.sigma_spatial;
    let two_sr_sq = 2.0 * params.sigma_range * params.sigma_range;
    let iw = width as i64;
    let ih = height as i64;

    // Pre-compute luminance (mean of channels) for range distance
    let luma: Vec<f32> = (0..n)
        .map(|i| {
            let sum: f32 = (0..channels).map(|c| src[i * channels + c]).sum();
            sum / channels as f32
        })
        .collect();

    for cy in 0..ih {
        for cx in 0..iw {
            let center_idx = (cy * iw + cx) as usize;
            let center_luma = luma[center_idx];
            let mut weight_sum = 0.0_f32;
            let mut val_sums = vec![0.0_f32; channels];

            for dy in -r..=r {
                for dx in -r..=r {
                    let sx = (cx + dx).clamp(0, iw - 1);
                    let sy = (cy + dy).clamp(0, ih - 1);
                    let neighbor_idx = (sy * iw + sx) as usize;
                    let neighbor_luma = luma[neighbor_idx];
                    let spatial_dist_sq = (dx * dx + dy * dy) as f32;
                    let range_dist = (neighbor_luma - center_luma) * 255.0;
                    let spatial_w = (-spatial_dist_sq / two_ss_sq).exp();
                    let range_w = (-(range_dist * range_dist) / two_sr_sq).exp();
                    let w = spatial_w * range_w;
                    weight_sum += w;
                    for c in 0..channels {
                        val_sums[c] += w * src[neighbor_idx * channels + c];
                    }
                }
            }

            let base = center_idx * channels;
            if weight_sum > 1e-12 {
                for (c, &vs) in val_sums.iter().enumerate() {
                    dst[base + c] = vs / weight_sum;
                }
            } else {
                for c in 0..channels {
                    dst[base + c] = src[base + c];
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- ConvolutionKernel ----

    #[test]
    fn test_sharpen_kernel_size() {
        let k = ConvolutionKernel::sharpen();
        assert_eq!(k.width, 3);
        assert_eq!(k.height, 3);
        assert_eq!(k.data.len(), 9);
    }

    #[test]
    fn test_sharpen_kernel_sum() {
        let k = ConvolutionKernel::sharpen();
        let sum: f32 = k.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {sum}");
    }

    #[test]
    fn test_emboss_kernel_size() {
        let k = ConvolutionKernel::emboss();
        assert_eq!(k.data.len(), 9);
    }

    #[test]
    fn test_gaussian_kernel_sum_near_one() {
        let k = ConvolutionKernel::gaussian(1.0, 5);
        let sum: f32 = k.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "gaussian sum = {sum}");
    }

    #[test]
    fn test_gaussian_kernel_odd_size_enforced() {
        let k = ConvolutionKernel::gaussian(1.0, 4);
        assert_eq!(k.width, 5);
    }

    #[test]
    fn test_normalize_divides_by_sum() {
        let mut k = ConvolutionKernel::new(vec![1.0, 1.0, 1.0, 1.0], 2, 2);
        k.normalize();
        let sum: f32 = k.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_normalize_zero_sum_unchanged() {
        let mut k = ConvolutionKernel::new(vec![1.0, -1.0], 2, 1);
        k.normalize();
        assert!((k.data[0] - 1.0).abs() < 1e-6);
    }

    // ---- apply_convolution ----

    #[test]
    fn test_apply_convolution_identity() {
        let src = vec![0.1f32, 0.5, 0.9, 0.3];
        let mut dst = vec![0.0f32; 4];
        let kernel = ConvolutionKernel::new(vec![1.0], 1, 1);
        apply_convolution(&src, &mut dst, 2, 2, &kernel);
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!((s - d).abs() < 1e-6);
        }
    }

    #[test]
    fn test_apply_convolution_output_length() {
        let src = vec![0.5f32; 4 * 4];
        let mut dst = vec![0.0f32; 4 * 4];
        let k = ConvolutionKernel::sharpen();
        apply_convolution(&src, &mut dst, 4, 4, &k);
        assert_eq!(dst.len(), 16);
    }

    #[test]
    fn test_apply_convolution_uniform_sharpen() {
        let src = vec![0.5f32; 9];
        let mut dst = vec![0.0f32; 9];
        let k = ConvolutionKernel::sharpen();
        apply_convolution(&src, &mut dst, 3, 3, &k);
        for &v in &dst {
            assert!((v - 0.5).abs() < 1e-5, "v = {v}");
        }
    }

    // ---- threshold ----

    #[test]
    fn test_threshold_basic() {
        let src = vec![50u8, 100, 150, 200];
        let mut dst = vec![0u8; 4];
        threshold(&src, &mut dst, 128);
        assert_eq!(dst, vec![0, 0, 255, 255]);
    }

    #[test]
    fn test_threshold_boundary() {
        let src = vec![127u8, 128];
        let mut dst = vec![0u8; 2];
        threshold(&src, &mut dst, 128);
        assert_eq!(dst[0], 0);
        assert_eq!(dst[1], 255);
    }

    // ---- equalize_histogram ----

    #[test]
    fn test_equalize_histogram_length() {
        let src = vec![10u8, 20, 30, 40];
        let mut dst = vec![0u8; 4];
        equalize_histogram(&src, &mut dst);
        assert_eq!(dst.len(), 4);
    }

    #[test]
    fn test_equalize_histogram_uniform_input() {
        let src = vec![128u8; 16];
        let mut dst = vec![0u8; 16];
        equalize_histogram(&src, &mut dst);
        assert!(dst.iter().all(|&v| v == dst[0]));
    }

    #[test]
    fn test_equalize_histogram_empty() {
        let src: Vec<u8> = Vec::new();
        let mut dst: Vec<u8> = Vec::new();
        equalize_histogram(&src, &mut dst);
    }

    // ---- median_filter_3x3 ----

    #[test]
    fn test_median_filter_length() {
        let src = vec![0u8; 4 * 4];
        let mut dst = vec![0u8; 4 * 4];
        median_filter_3x3(&src, &mut dst, 4, 4);
        assert_eq!(dst.len(), 16);
    }

    #[test]
    fn test_median_filter_uniform_unchanged() {
        let src = vec![128u8; 3 * 3];
        let mut dst = vec![0u8; 3 * 3];
        median_filter_3x3(&src, &mut dst, 3, 3);
        assert!(dst.iter().all(|&v| v == 128));
    }

    #[test]
    fn test_median_filter_removes_spike() {
        let mut src = vec![10u8; 9];
        src[4] = 200;
        let mut dst = vec![0u8; 9];
        median_filter_3x3(&src, &mut dst, 3, 3);
        assert_eq!(dst[4], 10);
    }

    // ---- bilateral_filter ----

    #[test]
    fn test_bilateral_params_auto_radius() {
        let p = BilateralParams::auto(3.0, 30.0);
        assert_eq!(p.window_radius, 9);
    }

    #[test]
    fn test_bilateral_params_default() {
        let p = BilateralParams::default();
        assert!(p.sigma_spatial > 0.0);
        assert!(p.sigma_range > 0.0);
        assert!(p.window_radius >= 1);
    }

    #[test]
    fn test_bilateral_filter_f32_uniform_unchanged() {
        let src = vec![0.5_f32; 5 * 5];
        let mut dst = vec![0.0_f32; 5 * 5];
        let params = BilateralParams::auto(1.0, 30.0);
        bilateral_filter_f32(&src, &mut dst, 5, 5, &params);
        for &v in &dst {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_bilateral_filter_f32_output_length() {
        let src = vec![0.3_f32; 8 * 8];
        let mut dst = vec![0.0_f32; 8 * 8];
        let params = BilateralParams::auto(2.0, 20.0);
        bilateral_filter_f32(&src, &mut dst, 8, 8, &params);
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn test_bilateral_filter_f32_edge_preservation() {
        let src: Vec<f32> = (0..10).map(|i| if i < 5 { 0.0 } else { 1.0 }).collect();
        let mut dst = vec![0.0_f32; 10];
        let params = BilateralParams::new(2.0, 5.0, 2);
        bilateral_filter_f32(&src, &mut dst, 10, 1, &params);
        assert!(dst[0] < 0.1, "left far pixel: {}", dst[0]);
        assert!(dst[9] > 0.9, "right far pixel: {}", dst[9]);
    }

    #[test]
    fn test_bilateral_filter_u8_uniform_unchanged() {
        let src = vec![128u8; 6 * 6];
        let mut dst = vec![0u8; 6 * 6];
        let params = BilateralParams::auto(1.5, 25.0);
        bilateral_filter_u8(&src, &mut dst, 6, 6, &params);
        for &v in &dst {
            assert!((v as i32 - 128).abs() <= 1, "expected ~128, got {v}");
        }
    }

    #[test]
    fn test_bilateral_filter_u8_reduces_noise() {
        let mut src = vec![100u8; 7 * 7];
        src[10] = 200;
        src[20] = 10;
        src[30] = 220;
        let mut dst = vec![0u8; 7 * 7];
        let params = BilateralParams::new(2.0, 50.0, 3);
        bilateral_filter_u8(&src, &mut dst, 7, 7, &params);
        let out_max = *dst.iter().max().unwrap_or(&0);
        let out_min = *dst.iter().min().unwrap_or(&255);
        assert!(
            (out_max as i32 - out_min as i32) < (220 - 10),
            "range should compress: min={out_min} max={out_max}"
        );
    }

    #[test]
    fn test_bilateral_filter_rgb_uniform_unchanged() {
        let src = vec![0.6_f32; 4 * 4 * 3];
        let mut dst = vec![0.0_f32; 4 * 4 * 3];
        let params = BilateralParams::auto(1.5, 20.0);
        bilateral_filter_rgb(&src, &mut dst, 4, 4, 3, &params);
        for &v in &dst {
            assert!((v - 0.6).abs() < 1e-5, "expected 0.6, got {v}");
        }
    }

    #[test]
    fn test_bilateral_filter_rgb_output_length() {
        let src = vec![0.5_f32; 5 * 5 * 3];
        let mut dst = vec![0.0_f32; 5 * 5 * 3];
        let params = BilateralParams::auto(1.0, 15.0);
        bilateral_filter_rgb(&src, &mut dst, 5, 5, 3, &params);
        assert_eq!(dst.len(), 75);
    }

    #[test]
    fn test_bilateral_params_new_explicit() {
        let p = BilateralParams::new(5.0, 50.0, 4);
        assert!((p.sigma_spatial - 5.0).abs() < 1e-6);
        assert!((p.sigma_range - 50.0).abs() < 1e-6);
        assert_eq!(p.window_radius, 4);
    }
}
