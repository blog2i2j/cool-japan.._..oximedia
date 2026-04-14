//! Configurable scope resolution independent of output resolution.
//!
//! Professional scope monitors typically analyze at full frame resolution but
//! render the scope display at a lower (or different) resolution chosen for
//! the monitoring workstation.  This module provides [`ScopeResolutionConfig`]
//! and [`ResolutionScaler`] which decouple *analysis resolution* (the
//! sub-sampled pixel grid used to accumulate scope data) from *output
//! resolution* (the final rendered image dimensions).
//!
//! # Use cases
//!
//! | Scenario | Analysis res | Output res |
//! |----------|-------------|------------|
//! | Real-time preview | 25% (¼ pixels) | 512×512 |
//! | Full-accuracy offline | 100% | 1024×512 |
//! | Broadcast QC | 50% | 1920×1080 |
//!
//! # How it works
//!
//! `ResolutionScaler::downsample` takes an RGB24 frame and returns a
//! sub-sampled copy at the configured analysis resolution.  The caller
//! then passes the downsampled frame to any existing scope function.
//! The output scope image is subsequently upscaled (nearest-neighbor) to
//! the final output dimensions via `ResolutionScaler::upscale_scope`.
//!
//! # Example
//!
//! ```
//! use oximedia_scopes::scope_resolution::{ResolutionScaler, ScopeResolutionConfig, AnalysisQuality};
//!
//! let cfg = ScopeResolutionConfig::new(1920, 1080, AnalysisQuality::Quarter);
//! let scaler = ResolutionScaler::new(cfg);
//!
//! // Downsample a 1920×1080 frame to 480×270 for fast analysis.
//! let frame = vec![128u8; 1920 * 1080 * 3];
//! let (small, sw, sh) = scaler.downsample(&frame, 1920, 1080).unwrap();
//! assert_eq!(sw, 480);
//! assert_eq!(sh, 270);
//! ```

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use oximedia_core::{OxiError, OxiResult};

// ─────────────────────────────────────────────────────────────────────────────
// Analysis quality presets
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-defined analysis quality levels that map to a sub-sampling factor.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnalysisQuality {
    /// Analyze every pixel.  Highest accuracy, slowest (factor = 1).
    Full,

    /// Analyze every 2nd pixel in each dimension (¼ pixel count, factor = 2).
    Half,

    /// Analyze every 4th pixel in each dimension (1/16 pixel count, factor = 4).
    Quarter,

    /// Custom sub-sampling step in each dimension.
    Custom {
        /// Horizontal step (≥ 1).
        step_x: u32,
        /// Vertical step (≥ 1).
        step_y: u32,
    },
}

impl AnalysisQuality {
    /// Returns the horizontal and vertical sub-sampling steps.
    #[must_use]
    pub fn steps(self) -> (u32, u32) {
        match self {
            Self::Full => (1, 1),
            Self::Half => (2, 2),
            Self::Quarter => (4, 4),
            Self::Custom { step_x, step_y } => (step_x.max(1), step_y.max(1)),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration pairing an output scope size with an analysis quality level.
#[derive(Debug, Clone)]
pub struct ScopeResolutionConfig {
    /// Width of the final rendered scope image.
    pub output_width: u32,
    /// Height of the final rendered scope image.
    pub output_height: u32,
    /// Quality / sub-sampling factor used during analysis.
    pub quality: AnalysisQuality,
    /// Whether to apply a simple 2×2 box-filter average when downsampling
    /// (`true`) or use point-sample / skip (`false`).
    pub box_filter: bool,
}

impl ScopeResolutionConfig {
    /// Creates a new configuration.
    #[must_use]
    pub fn new(output_width: u32, output_height: u32, quality: AnalysisQuality) -> Self {
        Self {
            output_width,
            output_height,
            quality,
            box_filter: false,
        }
    }

    /// Enable box-filter averaging during downsampling.
    #[must_use]
    pub fn with_box_filter(mut self) -> Self {
        self.box_filter = true;
        self
    }

    /// Compute the analysis frame dimensions given a source frame size.
    ///
    /// Returns `(analysis_width, analysis_height)`.
    #[must_use]
    pub fn analysis_dims(&self, src_w: u32, src_h: u32) -> (u32, u32) {
        let (sx, sy) = self.quality.steps();
        let aw = ((src_w + sx - 1) / sx).max(1);
        let ah = ((src_h + sy - 1) / sy).max(1);
        (aw, ah)
    }
}

impl Default for ScopeResolutionConfig {
    fn default() -> Self {
        Self::new(512, 512, AnalysisQuality::Half)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ResolutionScaler
// ─────────────────────────────────────────────────────────────────────────────

/// Handles downsampling of input frames and upscaling of scope output images.
#[derive(Debug, Clone)]
pub struct ResolutionScaler {
    config: ScopeResolutionConfig,
}

impl ResolutionScaler {
    /// Creates a new `ResolutionScaler` with the given configuration.
    #[must_use]
    pub fn new(config: ScopeResolutionConfig) -> Self {
        Self { config }
    }

    /// Returns a reference to the configuration.
    #[must_use]
    pub fn config(&self) -> &ScopeResolutionConfig {
        &self.config
    }

    /// Downsample an RGB24 frame to the analysis resolution.
    ///
    /// Returns `(downsampled_rgb24, analysis_width, analysis_height)`.
    ///
    /// When [`ScopeResolutionConfig::box_filter`] is `true`, each output pixel
    /// is the arithmetic average of the pixels in the sub-sampling block
    /// (clamped to the frame boundary).  When `false`, every `step`-th pixel
    /// is picked directly (faster, slightly less accurate).
    ///
    /// # Errors
    ///
    /// Returns an error if the frame buffer is too small or dimensions are zero.
    pub fn downsample(
        &self,
        frame: &[u8],
        src_w: u32,
        src_h: u32,
    ) -> OxiResult<(Vec<u8>, u32, u32)> {
        validate_rgb_frame(frame, src_w, src_h)?;

        let (sx, sy) = self.config.quality.steps();
        if sx == 1 && sy == 1 {
            // No downsampling needed — return a clone.
            return Ok((frame[..(src_w as usize * src_h as usize * 3)].to_vec(), src_w, src_h));
        }

        let (aw, ah) = self.config.analysis_dims(src_w, src_h);
        let mut out = vec![0u8; (aw as usize) * (ah as usize) * 3];

        if self.config.box_filter {
            downsample_box_filter(frame, src_w, src_h, &mut out, aw, ah, sx, sy);
        } else {
            downsample_point_sample(frame, src_w, src_h, &mut out, aw, ah, sx, sy);
        }

        Ok((out, aw, ah))
    }

    /// Upscale an RGBA scope image (analysis resolution) to the output resolution
    /// using nearest-neighbor interpolation.
    ///
    /// # Arguments
    ///
    /// * `scope_rgba` – RGBA pixel buffer at analysis resolution.
    /// * `scope_w`    – Width of the scope image.
    /// * `scope_h`    – Height of the scope image.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is too small.
    pub fn upscale_scope(
        &self,
        scope_rgba: &[u8],
        scope_w: u32,
        scope_h: u32,
    ) -> OxiResult<Vec<u8>> {
        if scope_w == 0 || scope_h == 0 {
            return Err(OxiError::InvalidData(
                "Scope dimensions must be non-zero".into(),
            ));
        }
        let expected = (scope_w as usize) * (scope_h as usize) * 4;
        if scope_rgba.len() < expected {
            return Err(OxiError::InvalidData(format!(
                "Scope buffer too small: need {expected}, got {}",
                scope_rgba.len()
            )));
        }

        let out_w = self.config.output_width;
        let out_h = self.config.output_height;

        if out_w == scope_w && out_h == scope_h {
            return Ok(scope_rgba[..expected].to_vec());
        }

        let mut out = vec![0u8; (out_w as usize) * (out_h as usize) * 4];

        for oy in 0..out_h {
            let sy = (oy as u64 * scope_h as u64 / out_h as u64) as u32;
            for ox in 0..out_w {
                let sx = (ox as u64 * scope_w as u64 / out_w as u64) as u32;
                let src_idx = ((sy * scope_w + sx) * 4) as usize;
                let dst_idx = ((oy * out_w + ox) * 4) as usize;
                out[dst_idx..dst_idx + 4].copy_from_slice(&scope_rgba[src_idx..src_idx + 4]);
            }
        }

        Ok(out)
    }

    /// Compute the scale ratio between output and analysis resolution.
    ///
    /// Returns `(x_ratio, y_ratio)` where a value > 1.0 means upscaling.
    #[must_use]
    pub fn scale_ratio(&self, src_w: u32, src_h: u32) -> (f64, f64) {
        let (aw, ah) = self.config.analysis_dims(src_w, src_h);
        let xr = self.config.output_width as f64 / aw.max(1) as f64;
        let yr = self.config.output_height as f64 / ah.max(1) as f64;
        (xr, yr)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

fn validate_rgb_frame(frame: &[u8], w: u32, h: u32) -> OxiResult<()> {
    if w == 0 || h == 0 {
        return Err(OxiError::InvalidData(
            "Frame dimensions must be non-zero".into(),
        ));
    }
    let expected = (w as usize) * (h as usize) * 3;
    if frame.len() < expected {
        return Err(OxiError::InvalidData(format!(
            "Frame buffer too small: need {expected}, got {}",
            frame.len()
        )));
    }
    Ok(())
}

/// Point-sample downsampling: skip every `step_x` / `step_y` pixels.
fn downsample_point_sample(
    src: &[u8],
    src_w: u32,
    _src_h: u32,
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    sx: u32,
    sy: u32,
) {
    for dy in 0..dst_h {
        let sy_coord = (dy * sy) as usize;
        for dx in 0..dst_w {
            let sx_coord = (dx * sx) as usize;
            let src_idx = (sy_coord * src_w as usize + sx_coord) * 3;
            let dst_idx = ((dy * dst_w + dx) as usize) * 3;
            if src_idx + 2 < src.len() {
                dst[dst_idx] = src[src_idx];
                dst[dst_idx + 1] = src[src_idx + 1];
                dst[dst_idx + 2] = src[src_idx + 2];
            }
        }
    }
}

/// Box-filter downsampling: average over a `step_x × step_y` block.
fn downsample_box_filter(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dst: &mut [u8],
    dst_w: u32,
    dst_h: u32,
    sx: u32,
    sy: u32,
) {
    for dy in 0..dst_h {
        let y0 = (dy * sy) as usize;
        let y1 = ((dy * sy + sy) as usize).min(src_h as usize);
        for dx in 0..dst_w {
            let x0 = (dx * sx) as usize;
            let x1 = ((dx * sx + sx) as usize).min(src_w as usize);

            let mut r_sum = 0u32;
            let mut g_sum = 0u32;
            let mut b_sum = 0u32;
            let mut count = 0u32;

            for y in y0..y1 {
                for x in x0..x1 {
                    let idx = (y * src_w as usize + x) * 3;
                    if idx + 2 < src.len() {
                        r_sum += src[idx] as u32;
                        g_sum += src[idx + 1] as u32;
                        b_sum += src[idx + 2] as u32;
                        count += 1;
                    }
                }
            }

            let dst_idx = ((dy * dst_w + dx) as usize) * 3;
            if let Some(cnt) = count.checked_div(1).filter(|&c| c > 0) {
                dst[dst_idx] = (r_sum / cnt) as u8;
                dst[dst_idx + 1] = (g_sum / cnt) as u8;
                dst[dst_idx + 2] = (b_sum / cnt) as u8;
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        (0..w * h).flat_map(|_| [r, g, b]).collect()
    }

    fn solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        (0..w * h).flat_map(|_| [r, g, b, a]).collect()
    }

    // ── AnalysisQuality ──────────────────────────────────────────────────────

    #[test]
    fn test_quality_full_steps() {
        assert_eq!(AnalysisQuality::Full.steps(), (1, 1));
    }

    #[test]
    fn test_quality_half_steps() {
        assert_eq!(AnalysisQuality::Half.steps(), (2, 2));
    }

    #[test]
    fn test_quality_quarter_steps() {
        assert_eq!(AnalysisQuality::Quarter.steps(), (4, 4));
    }

    #[test]
    fn test_quality_custom_steps() {
        let q = AnalysisQuality::Custom { step_x: 3, step_y: 5 };
        assert_eq!(q.steps(), (3, 5));
    }

    // ── ScopeResolutionConfig ─────────────────────────────────────────────────

    #[test]
    fn test_config_analysis_dims_full() {
        let cfg = ScopeResolutionConfig::new(512, 512, AnalysisQuality::Full);
        assert_eq!(cfg.analysis_dims(1920, 1080), (1920, 1080));
    }

    #[test]
    fn test_config_analysis_dims_half() {
        let cfg = ScopeResolutionConfig::new(512, 512, AnalysisQuality::Half);
        assert_eq!(cfg.analysis_dims(1920, 1080), (960, 540));
    }

    #[test]
    fn test_config_analysis_dims_quarter() {
        let cfg = ScopeResolutionConfig::new(512, 512, AnalysisQuality::Quarter);
        assert_eq!(cfg.analysis_dims(1920, 1080), (480, 270));
    }

    // ── ResolutionScaler::downsample ──────────────────────────────────────────

    #[test]
    fn test_downsample_full_quality_returns_clone() {
        let frame = solid_rgb(64, 64, 100, 150, 200);
        let cfg = ScopeResolutionConfig::new(64, 64, AnalysisQuality::Full);
        let scaler = ResolutionScaler::new(cfg);
        let (out, ow, oh) = scaler.downsample(&frame, 64, 64).expect("should succeed");
        assert_eq!(ow, 64);
        assert_eq!(oh, 64);
        assert_eq!(out.len(), 64 * 64 * 3);
    }

    #[test]
    fn test_downsample_half_dimensions() {
        let frame = solid_rgb(64, 64, 128, 128, 128);
        let cfg = ScopeResolutionConfig::new(32, 32, AnalysisQuality::Half);
        let scaler = ResolutionScaler::new(cfg);
        let (out, ow, oh) = scaler.downsample(&frame, 64, 64).expect("should succeed");
        assert_eq!(ow, 32);
        assert_eq!(oh, 32);
        assert_eq!(out.len(), 32 * 32 * 3);
    }

    #[test]
    fn test_downsample_box_filter_solid_colour() {
        // Box-filter of a solid colour should produce the same colour.
        let frame = solid_rgb(64, 64, 200, 100, 50);
        let cfg = ScopeResolutionConfig::new(32, 32, AnalysisQuality::Half)
            .with_box_filter();
        let scaler = ResolutionScaler::new(cfg);
        let (out, _, _) = scaler.downsample(&frame, 64, 64).expect("should succeed");
        for px in out.chunks_exact(3) {
            assert_eq!(px[0], 200);
            assert_eq!(px[1], 100);
            assert_eq!(px[2], 50);
        }
    }

    #[test]
    fn test_downsample_zero_width_error() {
        let frame = solid_rgb(10, 10, 0, 0, 0);
        let cfg = ScopeResolutionConfig::default();
        let scaler = ResolutionScaler::new(cfg);
        assert!(scaler.downsample(&frame, 0, 10).is_err());
    }

    #[test]
    fn test_downsample_buffer_too_small_error() {
        let frame = vec![0u8; 5];
        let cfg = ScopeResolutionConfig::default();
        let scaler = ResolutionScaler::new(cfg);
        assert!(scaler.downsample(&frame, 100, 100).is_err());
    }

    // ── ResolutionScaler::upscale_scope ───────────────────────────────────────

    #[test]
    fn test_upscale_scope_identity() {
        let scope = solid_rgba(32, 32, 0, 128, 255, 255);
        let cfg = ScopeResolutionConfig::new(32, 32, AnalysisQuality::Half);
        let scaler = ResolutionScaler::new(cfg);
        let out = scaler.upscale_scope(&scope, 32, 32).expect("should succeed");
        assert_eq!(out.len(), 32 * 32 * 4);
        assert_eq!(out, scope);
    }

    #[test]
    fn test_upscale_scope_doubles_dimensions() {
        let scope = solid_rgba(32, 32, 255, 0, 0, 255);
        let cfg = ScopeResolutionConfig::new(64, 64, AnalysisQuality::Half);
        let scaler = ResolutionScaler::new(cfg);
        let out = scaler.upscale_scope(&scope, 32, 32).expect("should succeed");
        assert_eq!(out.len(), 64 * 64 * 4);
        // Every output pixel should be red
        for px in out.chunks_exact(4) {
            assert_eq!(px[0], 255);
            assert_eq!(px[1], 0);
        }
    }

    #[test]
    fn test_upscale_scope_zero_dimensions_error() {
        let scope = solid_rgba(32, 32, 0, 0, 0, 255);
        let cfg = ScopeResolutionConfig::new(64, 64, AnalysisQuality::Half);
        let scaler = ResolutionScaler::new(cfg);
        assert!(scaler.upscale_scope(&scope, 0, 32).is_err());
    }

    // ── scale_ratio ──────────────────────────────────────────────────────────

    #[test]
    fn test_scale_ratio_half_quality() {
        let cfg = ScopeResolutionConfig::new(1920, 1080, AnalysisQuality::Half);
        let scaler = ResolutionScaler::new(cfg);
        let (xr, yr) = scaler.scale_ratio(1920, 1080);
        // analysis = 960×540, output = 1920×1080 → ratio = 2.0
        assert!((xr - 2.0).abs() < 0.01, "xr={xr}");
        assert!((yr - 2.0).abs() < 0.01, "yr={yr}");
    }
}
