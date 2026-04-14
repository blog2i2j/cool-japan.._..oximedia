//! Tilt-shift miniature effect using a configurable gradient focus mask.
//!
//! Simulates the selective-focus look of a tilt-shift lens by blurring regions
//! outside a configurable in-focus band.  The focus band can be oriented
//! horizontally, vertically, or at an arbitrary angle and uses a smooth
//! Gaussian-like falloff into the blur zone.
//!
//! Blur is achieved via a separable box blur whose radius scales with the
//! gradient mask distance, approximating a depth-of-field ramp.
//!
//! # Example
//!
//! ```
//! use oximedia_vfx::tilt_shift::{TiltShiftEffect, TiltShiftConfig, FocusBandOrientation};
//! use oximedia_vfx::{Frame, EffectParams, VideoEffect};
//!
//! let config = TiltShiftConfig {
//!     orientation: FocusBandOrientation::Horizontal,
//!     focus_position: 0.4,
//!     focus_width: 0.15,
//!     max_blur_radius: 8,
//!     transition_width: 0.2,
//!     saturation_boost: 0.15,
//!     ..TiltShiftConfig::default()
//! };
//! let mut effect = TiltShiftEffect::new(config);
//! let input = Frame::new(64, 64).expect("frame");
//! let mut output = Frame::new(64, 64).expect("frame");
//! effect.apply(&input, &mut output, &EffectParams::new()).expect("apply");
//! ```

use crate::{EffectParams, Frame, VfxError, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// Orientation
// ─────────────────────────────────────────────────────────────────────────────

/// Orientation of the in-focus band.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FocusBandOrientation {
    /// Horizontal band: sharp strip across the image width.
    Horizontal,
    /// Vertical band: sharp strip across the image height.
    Vertical,
    /// Band at an arbitrary angle (radians, 0 = horizontal).
    Angled,
}

impl Default for FocusBandOrientation {
    fn default() -> Self {
        Self::Horizontal
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for the tilt-shift miniature effect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TiltShiftConfig {
    /// Orientation of the focus band.
    pub orientation: FocusBandOrientation,
    /// Centre position of the focus band along the perpendicular axis
    /// (normalised 0.0-1.0; 0.5 = centre).
    pub focus_position: f32,
    /// Half-width of the fully sharp zone (normalised 0.0-0.5).
    pub focus_width: f32,
    /// Width of the transition zone from sharp to maximum blur (normalised).
    pub transition_width: f32,
    /// Maximum blur radius in pixels at full distance from focus band.
    pub max_blur_radius: u32,
    /// Angle in radians (used only when `orientation == Angled`).
    pub angle_rad: f32,
    /// Optional saturation boost for the miniature look (0.0 = none).
    pub saturation_boost: f32,
    /// Number of box-blur passes (more passes approximate Gaussian better).
    pub blur_passes: u32,
}

impl Default for TiltShiftConfig {
    fn default() -> Self {
        Self {
            orientation: FocusBandOrientation::Horizontal,
            focus_position: 0.4,
            focus_width: 0.12,
            transition_width: 0.2,
            max_blur_radius: 10,
            angle_rad: 0.0,
            saturation_boost: 0.1,
            blur_passes: 2,
        }
    }
}

impl TiltShiftConfig {
    /// Validate and clamp parameters.
    #[must_use]
    pub fn validated(mut self) -> Self {
        self.focus_position = self.focus_position.clamp(0.0, 1.0);
        self.focus_width = self.focus_width.clamp(0.0, 0.5);
        self.transition_width = self.transition_width.clamp(0.01, 1.0);
        self.max_blur_radius = self.max_blur_radius.clamp(1, 64);
        self.saturation_boost = self.saturation_boost.clamp(-0.5, 1.0);
        self.blur_passes = self.blur_passes.clamp(1, 5);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Blur mask computation
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the blur amount (0.0 = sharp, 1.0 = max blur) for a normalised
/// position `(nx, ny)` given the tilt-shift config.
fn blur_amount(nx: f32, ny: f32, cfg: &TiltShiftConfig) -> f32 {
    // Distance from the focus band along the perpendicular axis.
    let perp = match cfg.orientation {
        FocusBandOrientation::Horizontal => ny,
        FocusBandOrientation::Vertical => nx,
        FocusBandOrientation::Angled => {
            // Signed distance from a line through (0.5, focus_position)
            // at angle `angle_rad`.
            let (sin_a, cos_a) = cfg.angle_rad.sin_cos();
            let dx = nx - 0.5;
            let dy = ny - cfg.focus_position;
            // Perpendicular distance = |dx*sin - dy*cos| but we want signed
            (-dx * sin_a + dy * cos_a).abs() + cfg.focus_position
        }
    };

    let dist = (perp - cfg.focus_position).abs();
    let inner = cfg.focus_width;
    let outer = cfg.focus_width + cfg.transition_width;

    if dist <= inner {
        0.0
    } else if dist >= outer {
        1.0
    } else {
        let t = (dist - inner) / (outer - inner);
        // Smoothstep for a natural falloff
        t * t * (3.0 - 2.0 * t)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Variable-radius box blur
// ─────────────────────────────────────────────────────────────────────────────

/// Generate the per-pixel blur radius map (in pixels).
fn build_blur_map(w: u32, h: u32, cfg: &TiltShiftConfig) -> Vec<u32> {
    let mut map = vec![0u32; (w as usize) * (h as usize)];
    let w_f = (w - 1).max(1) as f32;
    let h_f = (h - 1).max(1) as f32;
    for py in 0..h {
        let ny = py as f32 / h_f;
        for px in 0..w {
            let nx = px as f32 / w_f;
            let amt = blur_amount(nx, ny, cfg);
            let radius = (amt * cfg.max_blur_radius as f32).round() as u32;
            map[(py as usize) * (w as usize) + (px as usize)] = radius;
        }
    }
    map
}

/// Apply a single-pass box blur whose radius varies per-pixel according to `blur_map`.
///
/// This is an approximation: each pixel averages a square neighbourhood of
/// side `2 * radius + 1`.  For performance we keep it separable — horizontal
/// then vertical — but use the per-pixel radius for sampling.
fn variable_box_blur(
    src: &[u8],
    dst: &mut [u8],
    w: usize,
    h: usize,
    blur_map: &[u32],
) {
    // We need a temp buffer for the intermediate horizontal pass.
    let pixel_count = w * h;
    let mut tmp = vec![0u8; pixel_count * 4];

    // --- Horizontal pass ---
    for py in 0..h {
        for px in 0..w {
            let idx = py * w + px;
            let r = blur_map[idx] as usize;
            if r == 0 {
                let base = idx * 4;
                tmp[base..base + 4].copy_from_slice(&src[base..base + 4]);
                continue;
            }
            let x_start = px.saturating_sub(r);
            let x_end = (px + r + 1).min(w);
            let count = (x_end - x_start) as u32;
            let mut sum = [0u32; 4];
            for sx in x_start..x_end {
                let si = (py * w + sx) * 4;
                sum[0] += src[si] as u32;
                sum[1] += src[si + 1] as u32;
                sum[2] += src[si + 2] as u32;
                sum[3] += src[si + 3] as u32;
            }
            let base = idx * 4;
            tmp[base] = (sum[0] / count) as u8;
            tmp[base + 1] = (sum[1] / count) as u8;
            tmp[base + 2] = (sum[2] / count) as u8;
            tmp[base + 3] = (sum[3] / count) as u8;
        }
    }

    // --- Vertical pass ---
    for py in 0..h {
        for px in 0..w {
            let idx = py * w + px;
            let r = blur_map[idx] as usize;
            if r == 0 {
                let base = idx * 4;
                dst[base..base + 4].copy_from_slice(&tmp[base..base + 4]);
                continue;
            }
            let y_start = py.saturating_sub(r);
            let y_end = (py + r + 1).min(h);
            let count = (y_end - y_start) as u32;
            let mut sum = [0u32; 4];
            for sy in y_start..y_end {
                let si = (sy * w + px) * 4;
                sum[0] += tmp[si] as u32;
                sum[1] += tmp[si + 1] as u32;
                sum[2] += tmp[si + 2] as u32;
                sum[3] += tmp[si + 3] as u32;
            }
            let base = idx * 4;
            dst[base] = (sum[0] / count) as u8;
            dst[base + 1] = (sum[1] / count) as u8;
            dst[base + 2] = (sum[2] / count) as u8;
            dst[base + 3] = (sum[3] / count) as u8;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Saturation boost
// ─────────────────────────────────────────────────────────────────────────────

/// Boost saturation of an RGBA buffer in-place.
fn boost_saturation(data: &mut [u8], amount: f32) {
    if amount.abs() < f32::EPSILON {
        return;
    }
    let inv = 1.0 + amount;
    for chunk in data.chunks_exact_mut(4) {
        let r = chunk[0] as f32 / 255.0;
        let g = chunk[1] as f32 / 255.0;
        let b = chunk[2] as f32 / 255.0;
        let luma = 0.299 * r + 0.587 * g + 0.114 * b;
        let nr = luma + (r - luma) * inv;
        let ng = luma + (g - luma) * inv;
        let nb = luma + (b - luma) * inv;
        chunk[0] = (nr.clamp(0.0, 1.0) * 255.0) as u8;
        chunk[1] = (ng.clamp(0.0, 1.0) * 255.0) as u8;
        chunk[2] = (nb.clamp(0.0, 1.0) * 255.0) as u8;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Effect
// ─────────────────────────────────────────────────────────────────────────────

/// Tilt-shift miniature effect.
pub struct TiltShiftEffect {
    config: TiltShiftConfig,
}

impl TiltShiftEffect {
    /// Create a new tilt-shift effect with the given configuration.
    #[must_use]
    pub fn new(config: TiltShiftConfig) -> Self {
        Self {
            config: config.validated(),
        }
    }

    /// Get the current configuration.
    #[must_use]
    pub fn config(&self) -> &TiltShiftConfig {
        &self.config
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: TiltShiftConfig) {
        self.config = config.validated();
    }
}

impl Default for TiltShiftEffect {
    fn default() -> Self {
        Self::new(TiltShiftConfig::default())
    }
}

impl VideoEffect for TiltShiftEffect {
    fn name(&self) -> &str {
        "TiltShift"
    }

    fn description(&self) -> &'static str {
        "Miniature/tilt-shift selective focus with gradient blur mask"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        _params: &EffectParams,
    ) -> VfxResult<()> {
        if input.width != output.width || input.height != output.height {
            return Err(VfxError::InvalidDimensions {
                width: output.width,
                height: output.height,
            });
        }

        let w = input.width as usize;
        let h = input.height as usize;
        if w == 0 || h == 0 {
            return Ok(());
        }

        // Build per-pixel blur radius map
        let blur_map = build_blur_map(input.width, input.height, &self.config);

        // Multi-pass box blur for better approximation of Gaussian
        let mut current = input.data.clone();
        let mut scratch = vec![0u8; current.len()];

        for _ in 0..self.config.blur_passes {
            variable_box_blur(&current, &mut scratch, w, h, &blur_map);
            std::mem::swap(&mut current, &mut scratch);
        }

        // Saturation boost (miniature look)
        if self.config.saturation_boost.abs() > f32::EPSILON {
            boost_saturation(&mut current, self.config.saturation_boost);
        }

        output.data[..current.len()].copy_from_slice(&current);
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_gradient_frame(w: u32, h: u32) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        for py in 0..h {
            let val = ((py as f32 / (h - 1).max(1) as f32) * 255.0) as u8;
            for px in 0..w {
                f.set_pixel(px, py, [val, val, val, 255]);
            }
        }
        f
    }

    fn make_solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        f.clear(rgba);
        f
    }

    #[test]
    fn test_blur_amount_at_focus_centre() {
        let cfg = TiltShiftConfig::default().validated();
        let amt = blur_amount(0.5, cfg.focus_position, &cfg);
        assert!(
            amt < f32::EPSILON,
            "should be zero at focus centre, got {amt}"
        );
    }

    #[test]
    fn test_blur_amount_far_from_focus() {
        let cfg = TiltShiftConfig {
            focus_position: 0.5,
            focus_width: 0.05,
            transition_width: 0.1,
            ..TiltShiftConfig::default()
        }
        .validated();
        let amt = blur_amount(0.5, 0.0, &cfg);
        assert!(
            (amt - 1.0).abs() < f32::EPSILON,
            "should be 1.0 at edge, got {amt}"
        );
    }

    #[test]
    fn test_blur_amount_vertical() {
        let cfg = TiltShiftConfig {
            orientation: FocusBandOrientation::Vertical,
            focus_position: 0.5,
            focus_width: 0.05,
            transition_width: 0.1,
            ..TiltShiftConfig::default()
        }
        .validated();
        // At vertical centre should be sharp
        let sharp = blur_amount(0.5, 0.5, &cfg);
        assert!(sharp < f32::EPSILON);
        // At edge should be blurred
        let blurred = blur_amount(0.0, 0.5, &cfg);
        assert!(blurred > 0.5);
    }

    #[test]
    fn test_default_effect_applies() {
        let mut fx = TiltShiftEffect::default();
        let input = make_gradient_frame(32, 32);
        let mut output = Frame::new(32, 32).expect("frame");
        fx.apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        // Output should differ from input (blur applied)
        assert_ne!(input.data, output.data);
    }

    #[test]
    fn test_dimension_mismatch_error() {
        let mut fx = TiltShiftEffect::default();
        let input = Frame::new(32, 32).expect("frame");
        let mut output = Frame::new(16, 16).expect("frame");
        assert!(fx
            .apply(&input, &mut output, &EffectParams::new())
            .is_err());
    }

    #[test]
    fn test_focus_band_is_sharp() {
        // Set a narrow focus band at 0.5, with no grain and large blur
        let cfg = TiltShiftConfig {
            focus_position: 0.5,
            focus_width: 0.2,
            transition_width: 0.15,
            max_blur_radius: 4,
            saturation_boost: 0.0,
            blur_passes: 1,
            ..TiltShiftConfig::default()
        }
        .validated();
        let mut fx = TiltShiftEffect::new(cfg);
        let input = make_gradient_frame(32, 32);
        let mut output = Frame::new(32, 32).expect("frame");
        fx.apply(&input, &mut output, &EffectParams::new())
            .expect("apply");

        // Pixels at the focus centre (row 16) should match input closely
        let y_focus = 16u32;
        let x = 16u32;
        let inp = input.get_pixel(x, y_focus).unwrap_or([0; 4]);
        let out = output.get_pixel(x, y_focus).unwrap_or([0; 4]);
        assert!(
            (inp[0] as i32 - out[0] as i32).unsigned_abs() <= 2,
            "focus centre should be nearly unchanged: inp={}, out={}",
            inp[0],
            out[0]
        );
    }

    #[test]
    fn test_edges_are_blurred() {
        let cfg = TiltShiftConfig {
            focus_position: 0.5,
            focus_width: 0.1,
            transition_width: 0.1,
            max_blur_radius: 6,
            saturation_boost: 0.0,
            blur_passes: 2,
            ..TiltShiftConfig::default()
        }
        .validated();
        let mut fx = TiltShiftEffect::new(cfg);

        // Create a checkerboard so blur is visible
        let w = 32u32;
        let h = 32u32;
        let mut input = Frame::new(w, h).expect("frame");
        for py in 0..h {
            for px in 0..w {
                let val = if (px + py) % 2 == 0 { 255u8 } else { 0u8 };
                input.set_pixel(px, py, [val, val, val, 255]);
            }
        }
        let mut output = Frame::new(w, h).expect("frame");
        fx.apply(&input, &mut output, &EffectParams::new())
            .expect("apply");

        // Top-edge pixel should be blurred (not pure 0 or 255)
        let p = output.get_pixel(5, 0).unwrap_or([0; 4]);
        assert!(
            p[0] > 10 && p[0] < 245,
            "top edge should be blurred, got {}",
            p[0]
        );
    }

    #[test]
    fn test_saturation_boost_changes_output() {
        let mut fx_none = TiltShiftEffect::new(TiltShiftConfig {
            saturation_boost: 0.0,
            max_blur_radius: 1,
            blur_passes: 1,
            ..TiltShiftConfig::default()
        });
        let mut fx_boost = TiltShiftEffect::new(TiltShiftConfig {
            saturation_boost: 0.5,
            max_blur_radius: 1,
            blur_passes: 1,
            ..TiltShiftConfig::default()
        });
        let input = make_solid_frame(16, 16, [200, 100, 50, 255]);
        let mut out_none = Frame::new(16, 16).expect("frame");
        let mut out_boost = Frame::new(16, 16).expect("frame");
        fx_none
            .apply(&input, &mut out_none, &EffectParams::new())
            .expect("apply");
        fx_boost
            .apply(&input, &mut out_boost, &EffectParams::new())
            .expect("apply");
        // Boosted output should differ (more saturated)
        assert_ne!(out_none.data, out_boost.data);
    }

    #[test]
    fn test_config_validation() {
        let cfg = TiltShiftConfig {
            focus_position: -1.0,
            focus_width: 5.0,
            transition_width: 0.0,
            max_blur_radius: 200,
            saturation_boost: 5.0,
            blur_passes: 100,
            ..TiltShiftConfig::default()
        }
        .validated();
        assert_eq!(cfg.focus_position, 0.0);
        assert_eq!(cfg.focus_width, 0.5);
        assert_eq!(cfg.transition_width, 0.01);
        assert_eq!(cfg.max_blur_radius, 64);
        assert_eq!(cfg.saturation_boost, 1.0);
        assert_eq!(cfg.blur_passes, 5);
    }

    #[test]
    fn test_effect_name_and_description() {
        let fx = TiltShiftEffect::default();
        assert_eq!(fx.name(), "TiltShift");
        assert!(!fx.description().is_empty());
    }

    #[test]
    fn test_angled_orientation() {
        let cfg = TiltShiftConfig {
            orientation: FocusBandOrientation::Angled,
            angle_rad: std::f32::consts::FRAC_PI_4,
            focus_position: 0.5,
            focus_width: 0.1,
            transition_width: 0.2,
            max_blur_radius: 4,
            saturation_boost: 0.0,
            blur_passes: 1,
        }
        .validated();
        let mut fx = TiltShiftEffect::new(cfg);
        let input = make_gradient_frame(32, 32);
        let mut output = Frame::new(32, 32).expect("frame");
        fx.apply(&input, &mut output, &EffectParams::new())
            .expect("apply");
        // Just check it runs without error and produces output
        assert_eq!(output.data.len(), input.data.len());
    }
}
