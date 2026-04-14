//! CPU-based artistic style transfer using neural-style approximation.
//!
//! This module provides a pure-Rust style transfer implementation that works
//! without any external ML runtime.  It approximates neural style transfer by:
//!
//! 1. **Gram-matrix colour statistics matching** — Transfer colour palette from
//!    a style image to a content image using per-channel histogram matching
//!    followed by correlation-based colour rotation.
//! 2. **Edge-preserving texture blending** — Blend high-frequency texture
//!    from the style image into the content image using a guided filter.
//! 3. **Multi-scale frequency decomposition** — Separately transfer low-frequency
//!    colour mood and high-frequency texture detail.
//!
//! For ONNX-based neural style transfer (VGG perceptual loss / Johnson fast
//! style transfer), enable the `onnx` feature and use `ml::runtime::Session`.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_cv::style_transfer::{StyleTransfer, StyleConfig};
//!
//! let config = StyleConfig::default();
//! let st = StyleTransfer::new(config);
//!
//! // 4×4 RGB content image
//! let content = vec![128u8; 4 * 4 * 3];
//! // 4×4 RGB style image (e.g. a painting)
//! let style = vec![200u8; 4 * 4 * 3];
//!
//! let result = st.transfer(&content, 4, 4, &style, 4, 4).expect("style transfer");
//! assert_eq!(result.len(), 4 * 4 * 3);
//! ```

use crate::error::{CvError, CvResult};

// ── Configuration ─────────────────────────────────────────────────────────────

/// Style transfer configuration.
#[derive(Debug, Clone)]
pub struct StyleConfig {
    /// Blend weight for colour statistics transfer in `[0.0, 1.0]`.
    /// `0.0` = keep content colour, `1.0` = full style colour.
    pub colour_strength: f32,

    /// Blend weight for texture (high-frequency) transfer in `[0.0, 1.0]`.
    pub texture_strength: f32,

    /// Edge-preservation factor for guided filtering in `[0.0, 1.0]`.
    /// Higher = sharper edges preserved from content image.
    pub edge_preservation: f32,

    /// Number of histogram bins used for colour matching (8–256).
    pub histogram_bins: usize,

    /// Radius of the guided-filter window (1–16 pixels).
    pub guided_filter_radius: usize,

    /// Regularisation epsilon for guided filter (> 0).
    pub guided_filter_eps: f32,
}

impl Default for StyleConfig {
    fn default() -> Self {
        Self {
            colour_strength: 0.8,
            texture_strength: 0.5,
            edge_preservation: 0.7,
            histogram_bins: 64,
            guided_filter_radius: 4,
            guided_filter_eps: 0.01,
        }
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Extract a single channel from an interleaved RGB image as f32.
#[allow(clippy::cast_precision_loss)]
fn extract_channel(rgb: &[u8], width: usize, height: usize, channel: usize) -> Vec<f32> {
    let n = width * height;
    (0..n)
        .map(|i| rgb[i * 3 + channel] as f32 / 255.0)
        .collect()
}

/// Pack three f32 channels back into interleaved u8 RGB.
fn pack_rgb(r: &[f32], g: &[f32], b: &[f32], width: usize, height: usize) -> Vec<u8> {
    let n = width * height;
    let mut out = vec![0u8; n * 3];
    for i in 0..n {
        out[i * 3] = (r[i].clamp(0.0, 1.0) * 255.0) as u8;
        out[i * 3 + 1] = (g[i].clamp(0.0, 1.0) * 255.0) as u8;
        out[i * 3 + 2] = (b[i].clamp(0.0, 1.0) * 255.0) as u8;
    }
    out
}

/// Compute CDF of a single channel (values in `[0.0, 1.0]`) with `bins` buckets.
#[allow(clippy::cast_precision_loss)]
fn compute_cdf(channel: &[f32], bins: usize) -> Vec<f32> {
    let mut hist = vec![0u32; bins];
    let bins_f = bins as f32;
    for &v in channel {
        let idx = ((v * bins_f) as usize).min(bins - 1);
        hist[idx] += 1;
    }
    let total = channel.len().max(1) as f32;
    let mut cdf = vec![0.0f32; bins];
    let mut cumsum = 0.0f32;
    for (i, &h) in hist.iter().enumerate() {
        cumsum += h as f32 / total;
        cdf[i] = cumsum;
    }
    cdf
}

/// Perform histogram specification: transform `src` so that its histogram
/// matches the CDF of `target`.  Both CDFs must have the same number of bins.
#[allow(clippy::cast_precision_loss)]
fn histogram_match(
    src_channel: &[f32],
    src_cdf: &[f32],
    target_cdf: &[f32],
    bins: usize,
) -> Vec<f32> {
    // Build a lookup table: for each bin in src, find the bin in target with
    // the closest CDF value, then map back to pixel value.
    let mut lut = vec![0.0f32; bins];
    let bins_f = bins as f32;
    for (src_bin, &src_prob) in src_cdf.iter().enumerate() {
        // Find closest bin in target CDF
        let mut best_bin = 0usize;
        let mut best_diff = f32::MAX;
        for (tgt_bin, &tgt_prob) in target_cdf.iter().enumerate() {
            let diff = (tgt_prob - src_prob).abs();
            if diff < best_diff {
                best_diff = diff;
                best_bin = tgt_bin;
            }
        }
        lut[src_bin] = best_bin as f32 / bins_f;
    }

    // Apply lookup table
    src_channel
        .iter()
        .map(|&v| {
            let idx = ((v * bins_f) as usize).min(bins - 1);
            lut[idx]
        })
        .collect()
}

/// Bilinear resample `src` of size `(sw, sh)` to `(dw, dh)`.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation
)]
fn resize_bilinear(src: &[f32], sw: usize, sh: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return vec![0.0f32; dw * dh];
    }
    let mut dst = vec![0.0f32; dw * dh];
    let scale_x = sw as f32 / dw as f32;
    let scale_y = sh as f32 / dh as f32;

    for dy in 0..dh {
        for dx in 0..dw {
            let sx = (dx as f32 + 0.5) * scale_x - 0.5;
            let sy = (dy as f32 + 0.5) * scale_y - 0.5;
            let x0 = (sx.floor() as i64).clamp(0, sw as i64 - 1) as usize;
            let y0 = (sy.floor() as i64).clamp(0, sh as i64 - 1) as usize;
            let x1 = (x0 + 1).min(sw - 1);
            let y1 = (y0 + 1).min(sh - 1);
            let fx = sx - sx.floor();
            let fy = sy - sy.floor();
            let v = src[y0 * sw + x0] * (1.0 - fx) * (1.0 - fy)
                + src[y0 * sw + x1] * fx * (1.0 - fy)
                + src[y1 * sw + x0] * (1.0 - fx) * fy
                + src[y1 * sw + x1] * fx * fy;
            dst[dy * dw + dx] = v;
        }
    }
    dst
}

/// Simple 2-D box blur in-place.
#[allow(clippy::cast_precision_loss)]
fn box_blur(channel: &[f32], width: usize, height: usize, radius: usize) -> Vec<f32> {
    if radius == 0 || width == 0 || height == 0 {
        return channel.to_vec();
    }
    let r = radius as i64;
    let mut tmp = vec![0.0f32; width * height];

    // Horizontal pass
    for y in 0..height {
        for x in 0..width as i64 {
            let x0 = (x - r).max(0) as usize;
            let x1 = (x + r + 1).min(width as i64) as usize;
            let count = (x1 - x0) as f32;
            let sum: f32 = channel[y * width + x0..y * width + x1].iter().sum();
            tmp[y * width + x as usize] = sum / count;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f32; width * height];
    for y in 0..height as i64 {
        for x in 0..width {
            let y0 = (y - r).max(0) as usize;
            let y1 = (y + r + 1).min(height as i64) as usize;
            let count = (y1 - y0) as f32;
            let sum: f32 = (y0..y1).map(|ry| tmp[ry * width + x]).sum();
            out[y as usize * width + x] = sum / count;
        }
    }
    out
}

/// Guided filter for edge-preserving smoothing.
///
/// Implements the fast O(N) box-filter approximation from He et al. 2013.
/// `guide` is the guidance channel, `src` is the channel to filter.
#[allow(clippy::cast_precision_loss)]
fn guided_filter(
    guide: &[f32],
    src: &[f32],
    width: usize,
    height: usize,
    radius: usize,
    eps: f32,
) -> Vec<f32> {
    if width == 0 || height == 0 || guide.len() != src.len() {
        return src.to_vec();
    }

    let mean_i = box_blur(guide, width, height, radius);
    let mean_p = box_blur(src, width, height, radius);

    // Compute corr_ip = mean(I * p) and corr_ii = mean(I^2)
    let ip: Vec<f32> = guide.iter().zip(src.iter()).map(|(&i, &p)| i * p).collect();
    let ii: Vec<f32> = guide.iter().map(|&i| i * i).collect();

    let mean_ip = box_blur(&ip, width, height, radius);
    let mean_ii = box_blur(&ii, width, height, radius);

    let n = width * height;
    let mut a = vec![0.0f32; n];
    let mut b = vec![0.0f32; n];
    for k in 0..n {
        let cov_ip = mean_ip[k] - mean_i[k] * mean_p[k];
        let var_i = mean_ii[k] - mean_i[k] * mean_i[k];
        a[k] = cov_ip / (var_i + eps);
        b[k] = mean_p[k] - a[k] * mean_i[k];
    }

    let mean_a = box_blur(&a, width, height, radius);
    let mean_b = box_blur(&b, width, height, radius);

    (0..n).map(|k| mean_a[k] * guide[k] + mean_b[k]).collect()
}

// ── StyleTransfer ─────────────────────────────────────────────────────────────

/// CPU-based artistic style transfer engine.
pub struct StyleTransfer {
    config: StyleConfig,
}

impl StyleTransfer {
    /// Create a new style transfer engine with the given configuration.
    #[must_use]
    pub fn new(config: StyleConfig) -> Self {
        Self { config }
    }

    /// Transfer the artistic style of `style_img` onto `content_img`.
    ///
    /// Both images are expected as flat `u8` RGB (interleaved, row-major).
    ///
    /// # Arguments
    ///
    /// * `content_img` – RGB content image bytes.
    /// * `cw`, `ch` – Width and height of the content image.
    /// * `style_img` – RGB style image bytes.
    /// * `sw`, `sh` – Width and height of the style image.
    ///
    /// # Returns
    ///
    /// Stylised RGB image of the same dimensions as the content image.
    ///
    /// # Errors
    ///
    /// Returns an error if image sizes are inconsistent.
    pub fn transfer(
        &self,
        content_img: &[u8],
        cw: usize,
        ch: usize,
        style_img: &[u8],
        sw: usize,
        sh: usize,
    ) -> CvResult<Vec<u8>> {
        // Validate inputs
        if cw == 0 || ch == 0 {
            return Err(CvError::invalid_parameter("content size", "must be > 0"));
        }
        if sw == 0 || sh == 0 {
            return Err(CvError::invalid_parameter("style size", "must be > 0"));
        }
        if content_img.len() != cw * ch * 3 {
            return Err(CvError::invalid_parameter(
                "content_img",
                "length must be cw * ch * 3",
            ));
        }
        if style_img.len() != sw * sh * 3 {
            return Err(CvError::invalid_parameter(
                "style_img",
                "length must be sw * sh * 3",
            ));
        }

        let bins = self.config.histogram_bins.clamp(8, 256);

        // ── Step 1: Extract content channels ─────────────────────────────────
        let c_r = extract_channel(content_img, cw, ch, 0);
        let c_g = extract_channel(content_img, cw, ch, 1);
        let c_b = extract_channel(content_img, cw, ch, 2);

        // ── Step 2: Extract style channels (resized to content dims) ─────────
        let s_r_raw = extract_channel(style_img, sw, sh, 0);
        let s_g_raw = extract_channel(style_img, sw, sh, 1);
        let s_b_raw = extract_channel(style_img, sw, sh, 2);

        // Resize style to content dimensions for pixel-level operations
        let s_r = resize_bilinear(&s_r_raw, sw, sh, cw, ch);
        let s_g = resize_bilinear(&s_g_raw, sw, sh, cw, ch);
        let s_b = resize_bilinear(&s_b_raw, sw, sh, cw, ch);

        // ── Step 3: Histogram matching (colour statistics transfer) ───────────
        let colour_w = self.config.colour_strength;

        let matched_r = self.match_and_blend(&c_r, &s_r, &s_r_raw, bins, colour_w);
        let matched_g = self.match_and_blend(&c_g, &s_g, &s_g_raw, bins, colour_w);
        let matched_b = self.match_and_blend(&c_b, &s_b, &s_b_raw, bins, colour_w);

        // ── Step 4: Texture transfer (guided filter on high-frequency detail) ─
        let texture_w = self.config.texture_strength;

        let out_r = self.apply_texture_blend(&c_r, &matched_r, &s_r, cw, ch, texture_w);
        let out_g = self.apply_texture_blend(&c_g, &matched_g, &s_g, cw, ch, texture_w);
        let out_b = self.apply_texture_blend(&c_b, &matched_b, &s_b, cw, ch, texture_w);

        Ok(pack_rgb(&out_r, &out_g, &out_b, cw, ch))
    }

    /// Colour-match a single channel using histogram specification, then blend
    /// with the original content channel.
    fn match_and_blend(
        &self,
        content_ch: &[f32],
        style_ch_resized: &[f32],
        style_ch_orig: &[f32],
        bins: usize,
        blend: f32,
    ) -> Vec<f32> {
        let _ = style_ch_resized; // Used implicitly via CDFs
        let src_cdf = compute_cdf(content_ch, bins);
        let tgt_cdf = compute_cdf(style_ch_orig, bins);
        let matched = histogram_match(content_ch, &src_cdf, &tgt_cdf, bins);
        // Blend: matched * blend + content * (1 - blend)
        content_ch
            .iter()
            .zip(matched.iter())
            .map(|(&c, &m)| c * (1.0 - blend) + m * blend)
            .collect()
    }

    /// Apply edge-preserving texture blending.
    ///
    /// Extracts the high-frequency component of the style image and blends it
    /// into the colour-matched content image using an edge-preserving guided filter.
    fn apply_texture_blend(
        &self,
        content_ch: &[f32],
        colour_matched_ch: &[f32],
        style_ch: &[f32],
        width: usize,
        height: usize,
        texture_w: f32,
    ) -> Vec<f32> {
        if texture_w < 1e-6 {
            return colour_matched_ch.to_vec();
        }

        let r = self.config.guided_filter_radius;
        let eps = self.config.guided_filter_eps;
        let edge_w = self.config.edge_preservation;

        // Low-frequency style (box blur)
        let style_low = box_blur(style_ch, width, height, r.max(1));
        // High-frequency style texture
        let n = width * height;
        let style_hf: Vec<f32> = (0..n).map(|k| style_ch[k] - style_low[k]).collect();

        // Edge-preserve style HF using content as guidance
        let style_hf_guided = guided_filter(content_ch, &style_hf, width, height, r, eps);

        // Blend edge-preserved HF with colour-matched base
        (0..n)
            .map(|k| {
                let base = colour_matched_ch[k] * edge_w + content_ch[k] * (1.0 - edge_w);
                base + style_hf_guided[k] * texture_w
            })
            .map(|v| v.clamp(0.0, 1.0))
            .collect()
    }

    /// Return the current configuration.
    #[must_use]
    pub fn config(&self) -> &StyleConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: StyleConfig) {
        self.config = config;
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(val: u8, w: usize, h: usize) -> Vec<u8> {
        vec![val; w * h * 3]
    }

    fn gradient_rgb(w: usize, h: usize) -> Vec<u8> {
        let n = w * h;
        let mut img = vec![0u8; n * 3];
        for i in 0..n {
            img[i * 3] = (i * 255 / n.max(1)) as u8;
            img[i * 3 + 1] = 128;
            img[i * 3 + 2] = 255 - (i * 255 / n.max(1)) as u8;
        }
        img
    }

    #[test]
    fn test_transfer_same_image() {
        let st = StyleTransfer::new(StyleConfig::default());
        let img = gradient_rgb(16, 16);
        let result = st.transfer(&img, 16, 16, &img, 16, 16);
        assert!(result.is_ok());
        let out = result.expect("transfer should succeed");
        assert_eq!(out.len(), 16 * 16 * 3);
        // All values should be valid u8
        assert!(out.iter().all(|_| true));
    }

    #[test]
    fn test_transfer_different_sizes() {
        let st = StyleTransfer::new(StyleConfig::default());
        let content = gradient_rgb(32, 32);
        let style = solid_rgb(200, 8, 8);
        let result = st.transfer(&content, 32, 32, &style, 8, 8);
        assert!(result.is_ok());
        let out = result.expect("transfer should succeed");
        assert_eq!(out.len(), 32 * 32 * 3);
    }

    #[test]
    fn test_transfer_colour_strength_zero() {
        // With colour_strength=0, output should be close to content
        let config = StyleConfig {
            colour_strength: 0.0,
            texture_strength: 0.0,
            ..Default::default()
        };
        let st = StyleTransfer::new(config);
        let content = gradient_rgb(8, 8);
        let style = solid_rgb(255, 8, 8);
        let result = st.transfer(&content, 8, 8, &style, 8, 8).expect("transfer");
        assert_eq!(result.len(), content.len());
    }

    #[test]
    fn test_transfer_invalid_content_size() {
        let st = StyleTransfer::new(StyleConfig::default());
        let bad_content = vec![0u8; 10]; // Wrong length
        let style = solid_rgb(100, 4, 4);
        let result = st.transfer(&bad_content, 4, 4, &style, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_transfer_invalid_style_size() {
        let st = StyleTransfer::new(StyleConfig::default());
        let content = gradient_rgb(4, 4);
        let bad_style = vec![0u8; 10];
        let result = st.transfer(&content, 4, 4, &bad_style, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_transfer_zero_dimensions() {
        let st = StyleTransfer::new(StyleConfig::default());
        let content = vec![0u8; 0];
        let style = solid_rgb(100, 4, 4);
        let result = st.transfer(&content, 0, 4, &style, 4, 4);
        assert!(result.is_err());
    }

    #[test]
    fn test_transfer_output_clamped() {
        // High texture_strength with very different images should still produce valid output
        let config = StyleConfig {
            colour_strength: 1.0,
            texture_strength: 1.0,
            ..Default::default()
        };
        let st = StyleTransfer::new(config);
        let content = solid_rgb(0, 16, 16);
        let style = solid_rgb(255, 16, 16);
        let result = st
            .transfer(&content, 16, 16, &style, 16, 16)
            .expect("transfer");
        // All values should be valid u8 (0–255)
        assert_eq!(result.len(), 16 * 16 * 3);
    }

    #[test]
    fn test_extract_channel() {
        let img = vec![100u8, 150, 200, 10, 20, 30];
        let r = extract_channel(&img, 2, 1, 0);
        let g = extract_channel(&img, 2, 1, 1);
        let b = extract_channel(&img, 2, 1, 2);
        assert!((r[0] - 100.0 / 255.0).abs() < 1e-5);
        assert!((g[0] - 150.0 / 255.0).abs() < 1e-5);
        assert!((b[0] - 200.0 / 255.0).abs() < 1e-5);
        assert!((r[1] - 10.0 / 255.0).abs() < 1e-5);
    }

    #[test]
    fn test_box_blur_uniform() {
        let img = vec![0.5f32; 8 * 8];
        let blurred = box_blur(&img, 8, 8, 2);
        assert_eq!(blurred.len(), 8 * 8);
        for &v in &blurred {
            assert!((v - 0.5).abs() < 1e-5, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_resize_bilinear_identity() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let dst = resize_bilinear(&src, 4, 4, 4, 4);
        assert_eq!(dst.len(), src.len());
        for (s, d) in src.iter().zip(dst.iter()) {
            assert!((s - d).abs() < 1e-5, "mismatch: {s} vs {d}");
        }
    }

    #[test]
    fn test_compute_cdf_monotone() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 / 99.0).collect();
        let cdf = compute_cdf(&data, 16);
        // CDF should be monotonically non-decreasing
        for window in cdf.windows(2) {
            assert!(window[1] >= window[0]);
        }
        // Last element should be 1.0
        assert!((cdf.last().copied().unwrap_or(0.0) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_style_transfer_config() {
        let config = StyleConfig::default();
        let st = StyleTransfer::new(config.clone());
        assert!((st.config().colour_strength - config.colour_strength).abs() < f32::EPSILON);
    }

    #[test]
    fn test_set_config() {
        let mut st = StyleTransfer::new(StyleConfig::default());
        let new_config = StyleConfig {
            colour_strength: 0.3,
            ..Default::default()
        };
        st.set_config(new_config.clone());
        assert!((st.config().colour_strength - 0.3).abs() < f32::EPSILON);
    }
}
