//! GPU-accelerated image compositing with alpha blending and blend modes.
//!
//! Implements four standard Porter-Duff / Photoshop-style blend modes:
//!
//! - **Normal** – alpha-compositing straight over operation.
//! - **Multiply** – each channel is multiplied together.
//! - **Screen** – inverse-multiply: `1 - (1-a)(1-b)`.
//! - **Overlay** – hard-light combination of multiply/screen.
//!
//! All operations run on CPU with rayon SIMD-friendly parallel processing as
//! a CPU fallback path (GPU compute shader path can be wired in later via
//! the existing `GpuDevice` infrastructure).

use crate::{GpuDevice, GpuError, Result};
use rayon::prelude::*;

use super::utils;

// ─────────────────────────────────────────────────────────────────────────────
// Public API types
// ─────────────────────────────────────────────────────────────────────────────

/// Blend mode for layer compositing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BlendMode {
    /// Standard alpha-compositing (Porter-Duff over).
    Normal,
    /// Multiply blend: `dst * src` per channel.
    Multiply,
    /// Screen blend: `1 - (1 - dst) * (1 - src)` per channel.
    Screen,
    /// Overlay: multiply for dark areas, screen for light areas.
    Overlay,
    /// Hard Light: overlay with source and destination roles swapped.
    /// Multiply if src < 0.5, screen otherwise.
    HardLight,
    /// Color Dodge: brightens the base colour to reflect the blend colour.
    /// `dst / (1 - src)`, clamped to 1.
    ColorDodge,
    /// Color Burn: darkens the base colour to reflect the blend colour.
    /// `1 - (1 - dst) / src`, clamped to 0.
    ColorBurn,
    /// Linear Burn: `dst + src - 1`, clamped to 0.
    LinearBurn,
    /// Exclusion: `dst + src - 2 * dst * src`.
    Exclusion,
}

impl Default for BlendMode {
    fn default() -> Self {
        Self::Normal
    }
}

/// A single compositing layer.
///
/// The pixel data must be RGBA (4 bytes per pixel), packed row-major.
#[derive(Debug, Clone)]
pub struct BlendLayer<'a> {
    /// Raw RGBA pixel data.
    pub data: &'a [u8],
    /// Layer width in pixels.
    pub width: u32,
    /// Layer height in pixels.
    pub height: u32,
    /// Global opacity in `[0.0, 1.0]` (multiplied into the alpha channel).
    pub opacity: f32,
    /// Blend mode applied when compositing this layer onto the accumulator.
    pub blend_mode: BlendMode,
}

impl<'a> BlendLayer<'a> {
    /// Create a new layer with the given parameters.
    ///
    /// # Errors
    ///
    /// Returns an error if `opacity` is outside `[0.0, 1.0]` or if the
    /// buffer is too small for the given dimensions.
    pub fn new(
        data: &'a [u8],
        width: u32,
        height: u32,
        opacity: f32,
        blend_mode: BlendMode,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&opacity) {
            return Err(GpuError::Internal(format!(
                "Layer opacity {opacity} is outside [0,1]"
            )));
        }
        utils::validate_buffer_size(data, width, height, 4)?;
        Ok(Self {
            data,
            width,
            height,
            opacity,
            blend_mode,
        })
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerCompositor
// ─────────────────────────────────────────────────────────────────────────────

/// Composites a stack of [`BlendLayer`]s onto a destination buffer.
pub struct LayerCompositor;

impl LayerCompositor {
    /// Composite `layers` (bottom-to-top order) into `output`.
    ///
    /// * `output` must be `width * height * 4` bytes and is pre-cleared to
    ///   transparent black before compositing begins.
    /// * All layers must have the same `width` × `height` dimensions as the
    ///   output buffer.
    ///
    /// `device` is kept as a parameter for future GPU compute shader dispatch;
    /// the current implementation is a CPU-parallel fallback.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `output` is not `width * height * 4` bytes.
    /// - Any layer has mismatched dimensions.
    /// - Dimensions are zero or exceed 16 384.
    pub fn blend_layers(
        _device: &GpuDevice,
        layers: &[BlendLayer<'_>],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        Self::blend_layers_cpu(layers, output, width, height)
    }

    /// CPU-only variant — useful for unit tests and CPU fallback paths.
    ///
    /// # Errors
    ///
    /// Same conditions as `blend_layers`.
    pub fn blend_layers_cpu(
        layers: &[BlendLayer<'_>],
        output: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        utils::validate_dimensions(width, height)?;
        let expected = (width * height * 4) as usize;
        if output.len() < expected {
            return Err(GpuError::InvalidBufferSize {
                expected,
                actual: output.len(),
            });
        }

        // Validate every layer dimensions up-front.
        for (idx, layer) in layers.iter().enumerate() {
            if layer.width != width || layer.height != height {
                return Err(GpuError::Internal(format!(
                    "Layer {idx} dimensions {}×{} do not match output {}×{}",
                    layer.width, layer.height, width, height
                )));
            }
        }

        // Start with transparent black.
        output[..expected].fill(0);

        // Composite layers bottom-to-top.
        for layer in layers {
            Self::composite_layer(layer, output, width, height)?;
        }

        Ok(())
    }

    /// Composite a single layer onto the running accumulator.
    fn composite_layer(
        layer: &BlendLayer<'_>,
        acc: &mut [u8],
        width: u32,
        height: u32,
    ) -> Result<()> {
        let n_pixels = (width * height) as usize;
        let opacity = layer.opacity;
        let mode = layer.blend_mode;

        acc.par_chunks_exact_mut(4)
            .zip(layer.data.par_chunks_exact(4))
            .take(n_pixels)
            .for_each(|(dst, src)| {
                // Normalise to [0,1].
                let dr = dst[0] as f32 / 255.0;
                let dg = dst[1] as f32 / 255.0;
                let db = dst[2] as f32 / 255.0;
                let da = dst[3] as f32 / 255.0;

                let sr = src[0] as f32 / 255.0;
                let sg = src[1] as f32 / 255.0;
                let sb = src[2] as f32 / 255.0;
                let sa = (src[3] as f32 / 255.0) * opacity;

                // Apply blend function to colour channels.
                let (br, bg, bb) = apply_blend(mode, sr, sg, sb, dr, dg, db);

                // Porter-Duff over composite using blended colour.
                let out_a = sa + da * (1.0 - sa);
                let (or, og, ob) = if out_a > 1e-6 {
                    (
                        (br * sa + dr * da * (1.0 - sa)) / out_a,
                        (bg * sa + dg * da * (1.0 - sa)) / out_a,
                        (bb * sa + db * da * (1.0 - sa)) / out_a,
                    )
                } else {
                    (0.0, 0.0, 0.0)
                };

                dst[0] = (or.clamp(0.0, 1.0) * 255.0).round() as u8;
                dst[1] = (og.clamp(0.0, 1.0) * 255.0).round() as u8;
                dst[2] = (ob.clamp(0.0, 1.0) * 255.0).round() as u8;
                dst[3] = (out_a.clamp(0.0, 1.0) * 255.0).round() as u8;
            });

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Blend mode arithmetic
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a blend mode to a single colour channel (src over dst).
/// Returns (r, g, b) after blending — alpha compositing is handled by the caller.
#[inline(always)]
fn apply_blend(
    mode: BlendMode,
    sr: f32,
    sg: f32,
    sb: f32,
    dr: f32,
    dg: f32,
    db: f32,
) -> (f32, f32, f32) {
    match mode {
        BlendMode::Normal => (sr, sg, sb),
        BlendMode::Multiply => (sr * dr, sg * dg, sb * db),
        BlendMode::Screen => (screen(sr, dr), screen(sg, dg), screen(sb, db)),
        BlendMode::Overlay => (overlay(dr, sr), overlay(dg, sg), overlay(db, sb)),
        BlendMode::HardLight => (hard_light(sr, dr), hard_light(sg, dg), hard_light(sb, db)),
        BlendMode::ColorDodge => (
            color_dodge(sr, dr),
            color_dodge(sg, dg),
            color_dodge(sb, db),
        ),
        BlendMode::ColorBurn => (color_burn(sr, dr), color_burn(sg, dg), color_burn(sb, db)),
        BlendMode::LinearBurn => (
            linear_burn(sr, dr),
            linear_burn(sg, dg),
            linear_burn(sb, db),
        ),
        BlendMode::Exclusion => (exclusion(sr, dr), exclusion(sg, dg), exclusion(sb, db)),
    }
}

/// Screen blend for a single channel.
#[inline(always)]
fn screen(a: f32, b: f32) -> f32 {
    1.0 - (1.0 - a) * (1.0 - b)
}

/// Overlay blend for a single channel (base = dst, blend = src).
#[inline(always)]
fn overlay(base: f32, blend: f32) -> f32 {
    if base < 0.5 {
        2.0 * base * blend
    } else {
        1.0 - 2.0 * (1.0 - base) * (1.0 - blend)
    }
}

/// Hard Light blend: overlay with src/dst roles swapped.
/// Multiply when src < 0.5; screen otherwise.
#[inline(always)]
fn hard_light(src: f32, dst: f32) -> f32 {
    if src < 0.5 {
        2.0 * src * dst
    } else {
        1.0 - 2.0 * (1.0 - src) * (1.0 - dst)
    }
}

/// Color Dodge: `dst / (1 - src)` — brightens the destination.
#[inline(always)]
fn color_dodge(src: f32, dst: f32) -> f32 {
    if src >= 1.0 {
        1.0
    } else {
        (dst / (1.0 - src)).min(1.0)
    }
}

/// Color Burn: `1 - (1 - dst) / src` — darkens the destination.
#[inline(always)]
fn color_burn(src: f32, dst: f32) -> f32 {
    if src <= 0.0 {
        0.0
    } else {
        (1.0 - (1.0 - dst) / src).max(0.0)
    }
}

/// Linear Burn: `dst + src - 1`, clamped to 0.
#[inline(always)]
fn linear_burn(src: f32, dst: f32) -> f32 {
    (dst + src - 1.0).max(0.0)
}

/// Exclusion: `dst + src - 2 * dst * src`.
#[inline(always)]
fn exclusion(src: f32, dst: f32) -> f32 {
    dst + src - 2.0 * dst * src
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 4) as usize];
        for px in v.chunks_exact_mut(4) {
            px[0] = r;
            px[1] = g;
            px[2] = b;
            px[3] = a;
        }
        v
    }

    // ── blend mode functions ─────────────────────────────────────────────────

    #[test]
    fn test_screen_blend_identity() {
        // screen(0, x) == x
        assert!((screen(0.0, 0.7) - 0.7).abs() < 1e-6);
        // screen(1, x) == 1
        assert!((screen(1.0, 0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_multiply_blend_zero() {
        let (r, _g, b) = apply_blend(BlendMode::Multiply, 0.0, 0.5, 1.0, 0.5, 0.5, 0.5);
        assert!((r - 0.0).abs() < 1e-6);
        assert!((b - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_overlay_midpoint() {
        // overlay(0.5, 0.5) should be 0.5
        let v = overlay(0.5, 0.5);
        assert!((v - 0.5).abs() < 1e-6, "overlay midpoint: {v}");
    }

    // ── LayerCompositor::blend_layers ────────────────────────────────────────

    #[test]
    fn test_blend_zero_layers_produces_black() {
        let w = 4u32;
        let h = 4u32;
        let mut output = solid_rgba(w, h, 255, 255, 255, 255);
        // Overwrite with some non-zero data to confirm clearing happens.
        let result = LayerCompositor::blend_layers_cpu(&[], &mut output, w, h);
        assert!(result.is_ok());
        // Should have been cleared to transparent black.
        for &v in &output {
            assert_eq!(v, 0, "zero layers should produce transparent black");
        }
    }

    #[test]
    fn test_blend_single_fully_opaque_layer() {
        let w = 4u32;
        let h = 4u32;
        let src = solid_rgba(w, h, 200, 100, 50, 255);
        let layer =
            BlendLayer::new(&src, w, h, 1.0, BlendMode::Normal).expect("create blend layer");
        let mut out = vec![0u8; (w * h * 4) as usize];
        LayerCompositor::blend_layers_cpu(&[layer], &mut out, w, h).expect("blend single layer");
        for i in 0..(w * h) as usize {
            assert_eq!(out[i * 4], 200, "red mismatch at pixel {i}");
            assert_eq!(out[i * 4 + 1], 100, "green mismatch at pixel {i}");
            assert_eq!(out[i * 4 + 2], 50, "blue mismatch at pixel {i}");
            assert_eq!(out[i * 4 + 3], 255, "alpha mismatch at pixel {i}");
        }
    }

    #[test]
    fn test_blend_two_layers_normal_over() {
        let w = 4u32;
        let h = 4u32;
        let bg = solid_rgba(w, h, 0, 0, 255, 255); // solid blue
        let fg = solid_rgba(w, h, 255, 0, 0, 128); // semi-transparent red
        let layers = [
            BlendLayer::new(&bg, w, h, 1.0, BlendMode::Normal).expect("create bg layer"),
            BlendLayer::new(&fg, w, h, 1.0, BlendMode::Normal).expect("create fg layer"),
        ];
        let mut out = vec![0u8; (w * h * 4) as usize];
        LayerCompositor::blend_layers_cpu(&layers, &mut out, w, h).expect("blend two layers");
        // Output alpha should be fully opaque (255 + semi over solid).
        for i in 0..(w * h) as usize {
            assert_eq!(out[i * 4 + 3], 255, "composite alpha should be 255");
            // Red channel > 0 from the foreground layer.
            assert!(out[i * 4] > 0, "red should be present");
        }
    }

    #[test]
    fn test_blend_multiply_two_identical_layers() {
        let w = 4u32;
        let h = 4u32;
        // Both layers are 0.5 grey (128), multiply → 0.25 (64) expected.
        let layer_data = solid_rgba(w, h, 128, 128, 128, 255);
        let layers = [
            BlendLayer::new(&layer_data, w, h, 1.0, BlendMode::Normal)
                .expect("create normal layer"),
            BlendLayer::new(&layer_data, w, h, 1.0, BlendMode::Multiply)
                .expect("create multiply layer"),
        ];
        let mut out = vec![0u8; (w * h * 4) as usize];
        LayerCompositor::blend_layers_cpu(&layers, &mut out, w, h).expect("blend multiply layers");
        // Multiply of ~128/255 ≈ 0.502 with itself → ~0.252 → ~64.
        for i in 0..(w * h) as usize {
            let r = out[i * 4];
            assert!(
                r >= 60 && r <= 68,
                "multiply result {r} out of expected range [60,68]"
            );
        }
    }

    #[test]
    fn test_blend_layer_dimension_mismatch() {
        let w = 4u32;
        let h = 4u32;
        let small = solid_rgba(2, 2, 255, 0, 0, 255);
        let layer = BlendLayer {
            data: &small,
            width: 2,
            height: 2,
            opacity: 1.0,
            blend_mode: BlendMode::Normal,
        };
        let mut out = vec![0u8; (w * h * 4) as usize];
        let result = LayerCompositor::blend_layers_cpu(&[layer], &mut out, w, h);
        assert!(result.is_err(), "mismatched dimensions should error");
    }

    #[test]
    fn test_blend_layer_invalid_opacity() {
        let data = solid_rgba(4, 4, 0, 0, 0, 255);
        let result = BlendLayer::new(&data, 4, 4, 1.5, BlendMode::Normal);
        assert!(result.is_err(), "opacity > 1.0 should error");
    }

    #[test]
    fn test_blend_screen_mode() {
        let w = 4u32;
        let h = 4u32;
        let bg = solid_rgba(w, h, 128, 128, 128, 255);
        let fg = solid_rgba(w, h, 128, 128, 128, 255);
        let layers = [
            BlendLayer::new(&bg, w, h, 1.0, BlendMode::Normal).expect("create bg layer"),
            BlendLayer::new(&fg, w, h, 1.0, BlendMode::Screen).expect("create screen fg layer"),
        ];
        let mut out = vec![0u8; (w * h * 4) as usize];
        LayerCompositor::blend_layers_cpu(&layers, &mut out, w, h).expect("blend screen layers");
        // Screen of ~0.5 with ~0.5 → 1-(0.5*0.5) = 0.75 → ~191.
        for i in 0..(w * h) as usize {
            let r = out[i * 4];
            assert!(
                r >= 185 && r <= 197,
                "screen result {r} out of expected range [185,197]"
            );
        }
    }

    #[test]
    fn test_blend_overlay_mode() {
        let w = 4u32;
        let h = 4u32;
        let bg = solid_rgba(w, h, 100, 200, 50, 255);
        let fg = solid_rgba(w, h, 200, 100, 150, 255);
        let layers = [
            BlendLayer::new(&bg, w, h, 1.0, BlendMode::Normal).expect("create bg layer"),
            BlendLayer::new(&fg, w, h, 1.0, BlendMode::Overlay).expect("create overlay fg layer"),
        ];
        let mut out = vec![0u8; (w * h * 4) as usize];
        let result = LayerCompositor::blend_layers_cpu(&layers, &mut out, w, h);
        assert!(result.is_ok());
    }

    // ── New blend-mode tests ──────────────────────────────────────────────────

    #[test]
    fn test_hard_light_with_mid_grey_src() {
        // hard_light(0.5, x) = overlay(x, 0.5) which at 0.5 gives 0.5
        let v = hard_light(0.5, 0.5);
        assert!(
            (v - 0.5).abs() < 1e-5,
            "hard_light(0.5,0.5) should be 0.5, got {v}"
        );
    }

    #[test]
    fn test_hard_light_black_src_yields_zero() {
        // hard_light(0, x) = 2*0*x = 0
        assert!((hard_light(0.0, 0.8) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_hard_light_white_src_yields_screen_like_max() {
        // hard_light(1, x) = 1 - 2*(1-1)*(1-x) = 1
        let v = hard_light(1.0, 0.3);
        assert!(
            (v - 1.0).abs() < 1e-6,
            "hard_light(1, x) should be 1, got {v}"
        );
    }

    #[test]
    fn test_color_dodge_white_src_yields_one() {
        // color_dodge(1.0, x) = 1 (src >= 1 clamp)
        assert!((color_dodge(1.0, 0.5) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_color_dodge_black_src_yields_dst() {
        // color_dodge(0, x) = x / 1 = x
        let dst = 0.6;
        let v = color_dodge(0.0, dst);
        assert!(
            (v - dst).abs() < 1e-6,
            "color_dodge(0, {dst}) should be {dst}, got {v}"
        );
    }

    #[test]
    fn test_color_burn_white_src_yields_dst() {
        // color_burn(1, x) = 1 - (1-x)/1 = x
        let dst = 0.4;
        let v = color_burn(1.0, dst);
        assert!(
            (v - dst).abs() < 1e-6,
            "color_burn(1, {dst}) should be {dst}, got {v}"
        );
    }

    #[test]
    fn test_color_burn_black_src_yields_zero() {
        // color_burn(0, x) = 0 (clamped)
        assert!((color_burn(0.0, 0.8) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_burn_saturates_to_zero() {
        // linear_burn(0.3, 0.5) = 0.3+0.5-1 = -0.2 → clamped to 0
        assert!((linear_burn(0.3, 0.5) - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_linear_burn_normal_case() {
        // linear_burn(0.9, 0.8) = 0.9+0.8-1 = 0.7
        let v = linear_burn(0.9, 0.8);
        assert!((v - 0.7).abs() < 1e-5, "expected 0.7, got {v}");
    }

    #[test]
    fn test_exclusion_with_black_src_yields_dst() {
        // exclusion(0, x) = x + 0 - 0 = x
        let dst = 0.65;
        let v = exclusion(0.0, dst);
        assert!(
            (v - dst).abs() < 1e-6,
            "exclusion(0, {dst}) should be {dst}, got {v}"
        );
    }

    #[test]
    fn test_exclusion_midpoint_yields_zero_five() {
        // exclusion(0.5, 0.5) = 0.5+0.5-2*0.5*0.5 = 1-0.5 = 0.5
        let v = exclusion(0.5, 0.5);
        assert!(
            (v - 0.5).abs() < 1e-5,
            "exclusion midpoint should be 0.5, got {v}"
        );
    }

    #[test]
    fn test_hard_light_blend_layers_round_trip() {
        let w = 4u32;
        let h = 4u32;
        let bg = solid_rgba(w, h, 64, 64, 64, 255);
        let fg = solid_rgba(w, h, 64, 64, 64, 255);
        let layers = [
            BlendLayer::new(&bg, w, h, 1.0, BlendMode::Normal).expect("create bg layer"),
            BlendLayer::new(&fg, w, h, 1.0, BlendMode::HardLight).expect("create hard light layer"),
        ];
        let mut out = vec![0u8; (w * h * 4) as usize];
        LayerCompositor::blend_layers_cpu(&layers, &mut out, w, h).expect("blend hard light");
        // hard_light(64/255, 64/255) ≈ 2*(64/255)^2 ≈ 0.126 → ~32
        for i in 0..(w * h) as usize {
            let r = out[i * 4];
            assert!(r < 64, "hard light with dark src should darken; got {r}");
        }
    }

    #[test]
    fn test_color_dodge_blend_layers_brightens() {
        let w = 4u32;
        let h = 4u32;
        // bg = mid grey, fg = mid grey (almost 0.5 → dodge brightens significantly)
        let bg = solid_rgba(w, h, 100, 100, 100, 255);
        let fg = solid_rgba(w, h, 128, 128, 128, 255);
        let layers = [
            BlendLayer::new(&bg, w, h, 1.0, BlendMode::Normal).expect("create bg layer"),
            BlendLayer::new(&fg, w, h, 1.0, BlendMode::ColorDodge).expect("create dodge layer"),
        ];
        let mut out = vec![0u8; (w * h * 4) as usize];
        LayerCompositor::blend_layers_cpu(&layers, &mut out, w, h).expect("blend color dodge");
        // Color dodge should produce brighter result than the background alone.
        for i in 0..(w * h) as usize {
            assert!(
                out[i * 4] >= 100,
                "color dodge should not darken; got {}",
                out[i * 4]
            );
        }
    }

    #[test]
    fn test_exclusion_blend_layers_round_trip() {
        let w = 4u32;
        let h = 4u32;
        let bg = solid_rgba(w, h, 128, 128, 128, 255);
        let fg = solid_rgba(w, h, 128, 128, 128, 255);
        let layers = [
            BlendLayer::new(&bg, w, h, 1.0, BlendMode::Normal).expect("create bg layer"),
            BlendLayer::new(&fg, w, h, 1.0, BlendMode::Exclusion).expect("create exclusion layer"),
        ];
        let mut out = vec![0u8; (w * h * 4) as usize];
        LayerCompositor::blend_layers_cpu(&layers, &mut out, w, h).expect("blend exclusion");
        // exclusion(0.5, 0.5) = 0.5 → ~128
        for i in 0..(w * h) as usize {
            let r = out[i * 4];
            assert!(
                r >= 120 && r <= 136,
                "exclusion(0.5,0.5) should be ~128; got {r}"
            );
        }
    }
}
