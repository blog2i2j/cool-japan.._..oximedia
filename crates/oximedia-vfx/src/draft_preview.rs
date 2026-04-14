//! Downsampled preview path for `QualityMode::Draft`.
//!
//! When operating in Draft mode, the effect chain can use this module to process
//! at a reduced resolution (e.g. 1/4) and then upscale the result, dramatically
//! reducing computation for real-time preview.
//!
//! # Architecture
//!
//! [`DraftPreview`] wraps any [`VideoEffect`] and intercepts `apply()`:
//!
//! 1. **Downsample** the input frame by the configured factor using box filtering.
//! 2. **Apply** the wrapped effect at the smaller resolution.
//! 3. **Upscale** the result back to the original resolution using bilinear
//!    interpolation.
//!
//! When `QualityMode` is *not* Draft, the effect is applied at full resolution
//! (pass-through).
//!
//! # Example
//!
//! ```ignore
//! use oximedia_vfx::draft_preview::{DraftPreview, DownsampleFactor};
//! use oximedia_vfx::{EffectParams, QualityMode};
//!
//! let wrapped = some_expensive_effect();
//! let mut draft = DraftPreview::new(wrapped, DownsampleFactor::Quarter);
//! let params = EffectParams::new().with_quality(QualityMode::Draft);
//! draft.apply(&input, &mut output, &params)?;
//! ```

use crate::{EffectParams, Frame, QualityMode, VfxError, VfxResult, VideoEffect};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────────────
// DownsampleFactor
// ─────────────────────────────────────────────────────────────────────────────

/// Factor by which the frame is reduced during Draft preview.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DownsampleFactor {
    /// 1/2 resolution (each axis halved).
    Half,
    /// 1/4 resolution (each axis quartered).
    Quarter,
    /// 1/8 resolution.
    Eighth,
}

impl DownsampleFactor {
    /// The integer divisor for each axis.
    #[must_use]
    pub const fn divisor(self) -> u32 {
        match self {
            Self::Half => 2,
            Self::Quarter => 4,
            Self::Eighth => 8,
        }
    }

    /// Compute the downsampled dimensions, clamping to at least 1x1.
    #[must_use]
    pub fn downsampled_size(self, width: u32, height: u32) -> (u32, u32) {
        let d = self.divisor();
        ((width / d).max(1), (height / d).max(1))
    }
}

impl Default for DownsampleFactor {
    fn default() -> Self {
        Self::Quarter
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Box downsample
// ─────────────────────────────────────────────────────────────────────────────

/// Downsample `src` by averaging blocks of `factor x factor` pixels.
///
/// Returns a new `Frame` of size `(dst_w, dst_h)`.
fn box_downsample(src: &Frame, dst_w: u32, dst_h: u32, factor: u32) -> VfxResult<Frame> {
    let mut dst = Frame::new(dst_w, dst_h)?;
    let src_w = src.width;
    let src_h = src.height;

    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let sx_start = dx * factor;
            let sy_start = dy * factor;
            let sx_end = (sx_start + factor).min(src_w);
            let sy_end = (sy_start + factor).min(src_h);

            let mut sum = [0u32; 4];
            let mut count = 0u32;

            for sy in sy_start..sy_end {
                for sx in sx_start..sx_end {
                    if let Some(px) = src.get_pixel(sx, sy) {
                        sum[0] += px[0] as u32;
                        sum[1] += px[1] as u32;
                        sum[2] += px[2] as u32;
                        sum[3] += px[3] as u32;
                        count += 1;
                    }
                }
            }

            if let Some(c) = std::num::NonZero::new(count) {
                let avg = [
                    (sum[0] / c) as u8,
                    (sum[1] / c) as u8,
                    (sum[2] / c) as u8,
                    (sum[3] / c) as u8,
                ];
                dst.set_pixel(dx, dy, avg);
            }
        }
    }
    Ok(dst)
}

// ─────────────────────────────────────────────────────────────────────────────
// Bilinear upscale
// ─────────────────────────────────────────────────────────────────────────────

/// Upscale `src` to `(dst_w, dst_h)` using bilinear interpolation.
fn bilinear_upscale(src: &Frame, dst_w: u32, dst_h: u32) -> VfxResult<Frame> {
    let mut dst = Frame::new(dst_w, dst_h)?;

    if src.width == 0 || src.height == 0 {
        return Ok(dst);
    }

    let x_ratio = if dst_w > 1 {
        (src.width as f32 - 1.0) / (dst_w as f32 - 1.0)
    } else {
        0.0
    };
    let y_ratio = if dst_h > 1 {
        (src.height as f32 - 1.0) / (dst_h as f32 - 1.0)
    } else {
        0.0
    };

    for dy in 0..dst_h {
        let fy = dy as f32 * y_ratio;
        let y0 = fy.floor() as u32;
        let y1 = (y0 + 1).min(src.height - 1);
        let ty = fy - fy.floor();

        for dx in 0..dst_w {
            let fx = dx as f32 * x_ratio;
            let x0 = fx.floor() as u32;
            let x1 = (x0 + 1).min(src.width - 1);
            let tx = fx - fx.floor();

            let p00 = src.get_pixel(x0, y0).unwrap_or([0; 4]);
            let p10 = src.get_pixel(x1, y0).unwrap_or([0; 4]);
            let p01 = src.get_pixel(x0, y1).unwrap_or([0; 4]);
            let p11 = src.get_pixel(x1, y1).unwrap_or([0; 4]);

            let mut out = [0u8; 4];
            for ch in 0..4 {
                let top = p00[ch] as f32 * (1.0 - tx) + p10[ch] as f32 * tx;
                let bottom = p01[ch] as f32 * (1.0 - tx) + p11[ch] as f32 * tx;
                out[ch] = (top * (1.0 - ty) + bottom * ty).clamp(0.0, 255.0) as u8;
            }
            dst.set_pixel(dx, dy, out);
        }
    }
    Ok(dst)
}

// ─────────────────────────────────────────────────────────────────────────────
// DraftPreview wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Wraps a [`VideoEffect`] to process at reduced resolution in Draft mode.
pub struct DraftPreview {
    inner: Box<dyn VideoEffect>,
    factor: DownsampleFactor,
}

impl DraftPreview {
    /// Wrap an effect with a draft preview downsample factor.
    #[must_use]
    pub fn new(inner: impl VideoEffect + 'static, factor: DownsampleFactor) -> Self {
        Self {
            inner: Box::new(inner),
            factor,
        }
    }

    /// Get the downsample factor.
    #[must_use]
    pub fn factor(&self) -> DownsampleFactor {
        self.factor
    }

    /// Set the downsample factor.
    pub fn set_factor(&mut self, factor: DownsampleFactor) {
        self.factor = factor;
    }
}

impl VideoEffect for DraftPreview {
    fn name(&self) -> &str {
        "DraftPreview"
    }

    fn description(&self) -> &'static str {
        "Downsampled preview wrapper for Draft quality mode"
    }

    fn apply(
        &mut self,
        input: &Frame,
        output: &mut Frame,
        params: &EffectParams,
    ) -> VfxResult<()> {
        if input.width != output.width || input.height != output.height {
            return Err(VfxError::InvalidDimensions {
                width: output.width,
                height: output.height,
            });
        }

        // Only downsample in Draft mode
        if params.quality != QualityMode::Draft {
            return self.inner.apply(input, output, params);
        }

        let (small_w, small_h) = self.factor.downsampled_size(input.width, input.height);

        // If downsampled size equals original, no point in downsampling
        if small_w == input.width && small_h == input.height {
            return self.inner.apply(input, output, params);
        }

        // Step 1: Downsample input
        let small_input = box_downsample(input, small_w, small_h, self.factor.divisor())?;

        // Step 2: Apply effect at reduced resolution
        let mut small_output = Frame::new(small_w, small_h)?;
        self.inner.apply(&small_input, &mut small_output, params)?;

        // Step 3: Upscale back to original resolution
        let upscaled = bilinear_upscale(&small_output, output.width, output.height)?;

        output.data[..upscaled.data.len()].copy_from_slice(&upscaled.data);
        Ok(())
    }

    fn reset(&mut self) {
        self.inner.reset();
    }

    fn supports_gpu(&self) -> bool {
        self.inner.supports_gpu()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Standalone downsample/upscale API
// ─────────────────────────────────────────────────────────────────────────────

/// Downsample a frame by the given factor using box filtering.
///
/// # Errors
///
/// Returns an error if the resulting dimensions are invalid.
pub fn downsample(frame: &Frame, factor: DownsampleFactor) -> VfxResult<Frame> {
    let (w, h) = factor.downsampled_size(frame.width, frame.height);
    box_downsample(frame, w, h, factor.divisor())
}

/// Upscale a frame to the target dimensions using bilinear interpolation.
///
/// # Errors
///
/// Returns an error if the target dimensions are invalid.
pub fn upscale(frame: &Frame, target_width: u32, target_height: u32) -> VfxResult<Frame> {
    bilinear_upscale(frame, target_width, target_height)
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple effect that fills output with a solid colour (ignores input).
    struct FillEffect {
        color: [u8; 4],
    }

    impl VideoEffect for FillEffect {
        fn name(&self) -> &str {
            "FillEffect"
        }
        fn apply(
            &mut self,
            _input: &Frame,
            output: &mut Frame,
            _params: &EffectParams,
        ) -> VfxResult<()> {
            output.clear(self.color);
            Ok(())
        }
    }

    /// Effect that copies input to output unchanged.
    struct PassThrough;

    impl VideoEffect for PassThrough {
        fn name(&self) -> &str {
            "PassThrough"
        }
        fn apply(
            &mut self,
            input: &Frame,
            output: &mut Frame,
            _params: &EffectParams,
        ) -> VfxResult<()> {
            output.data[..input.data.len()].copy_from_slice(&input.data);
            Ok(())
        }
    }

    fn solid_frame(w: u32, h: u32, rgba: [u8; 4]) -> Frame {
        let mut f = Frame::new(w, h).expect("frame");
        f.clear(rgba);
        f
    }

    // ── DownsampleFactor ────────────────────────────────────────────────

    #[test]
    fn test_downsample_factor_divisor() {
        assert_eq!(DownsampleFactor::Half.divisor(), 2);
        assert_eq!(DownsampleFactor::Quarter.divisor(), 4);
        assert_eq!(DownsampleFactor::Eighth.divisor(), 8);
    }

    #[test]
    fn test_downsample_factor_size() {
        assert_eq!(DownsampleFactor::Half.downsampled_size(100, 80), (50, 40));
        assert_eq!(DownsampleFactor::Quarter.downsampled_size(100, 80), (25, 20));
        assert_eq!(DownsampleFactor::Eighth.downsampled_size(100, 80), (12, 10));
    }

    #[test]
    fn test_downsample_factor_minimum_1x1() {
        assert_eq!(DownsampleFactor::Eighth.downsampled_size(4, 4), (1, 1));
        assert_eq!(DownsampleFactor::Eighth.downsampled_size(1, 1), (1, 1));
    }

    // ── Box downsample ──────────────────────────────────────────────────

    #[test]
    fn test_box_downsample_solid() {
        let src = solid_frame(16, 16, [200, 100, 50, 255]);
        let dst = box_downsample(&src, 4, 4, 4).expect("downsample");
        assert_eq!(dst.width, 4);
        assert_eq!(dst.height, 4);
        // Solid colour should be preserved
        let p = dst.get_pixel(2, 2).unwrap_or([0; 4]);
        assert_eq!(p, [200, 100, 50, 255]);
    }

    #[test]
    fn test_box_downsample_averages() {
        // Create a 4x4 frame with checkerboard: 0 and 200
        let mut src = Frame::new(4, 4).expect("frame");
        for y in 0..4 {
            for x in 0..4 {
                let val = if (x + y) % 2 == 0 { 200u8 } else { 0u8 };
                src.set_pixel(x, y, [val, val, val, 255]);
            }
        }
        let dst = box_downsample(&src, 2, 2, 2).expect("downsample");
        // Each 2x2 block has two 200 and two 0 → average = 100
        let p = dst.get_pixel(0, 0).unwrap_or([0; 4]);
        assert_eq!(p[0], 100, "should average to 100, got {}", p[0]);
    }

    // ── Bilinear upscale ────────────────────────────────────────────────

    #[test]
    fn test_bilinear_upscale_solid() {
        let src = solid_frame(4, 4, [128, 64, 32, 255]);
        let dst = bilinear_upscale(&src, 16, 16).expect("upscale");
        assert_eq!(dst.width, 16);
        assert_eq!(dst.height, 16);
        let p = dst.get_pixel(8, 8).unwrap_or([0; 4]);
        assert_eq!(p, [128, 64, 32, 255]);
    }

    #[test]
    fn test_bilinear_upscale_1x1() {
        let src = solid_frame(1, 1, [42, 43, 44, 255]);
        let dst = bilinear_upscale(&src, 8, 8).expect("upscale");
        for y in 0..8 {
            for x in 0..8 {
                let p = dst.get_pixel(x, y).unwrap_or([0; 4]);
                assert_eq!(p, [42, 43, 44, 255], "pixel at ({x},{y})");
            }
        }
    }

    // ── DraftPreview wrapper ────────────────────────────────────────────

    #[test]
    fn test_draft_preview_draft_mode_downsamples() {
        let fill = FillEffect {
            color: [100, 200, 50, 255],
        };
        let mut draft = DraftPreview::new(fill, DownsampleFactor::Quarter);
        let input = solid_frame(64, 64, [0, 0, 0, 255]);
        let mut output = Frame::new(64, 64).expect("output");
        let params = EffectParams::new().with_quality(QualityMode::Draft);
        draft.apply(&input, &mut output, &params).expect("apply");

        // The fill effect fills with [100,200,50,255] at small res,
        // then upscaled — should be the same solid colour
        let p = output.get_pixel(32, 32).unwrap_or([0; 4]);
        assert_eq!(p, [100, 200, 50, 255]);
    }

    #[test]
    fn test_draft_preview_non_draft_mode_passthrough() {
        let pass = PassThrough;
        let mut draft = DraftPreview::new(pass, DownsampleFactor::Quarter);
        let input = solid_frame(16, 16, [42, 43, 44, 255]);
        let mut output = Frame::new(16, 16).expect("output");
        let params = EffectParams::new().with_quality(QualityMode::Final);
        draft.apply(&input, &mut output, &params).expect("apply");
        // Should be identical (no downsampling)
        assert_eq!(output.data, input.data);
    }

    #[test]
    fn test_draft_preview_dimension_mismatch() {
        let pass = PassThrough;
        let mut draft = DraftPreview::new(pass, DownsampleFactor::Half);
        let input = Frame::new(32, 32).expect("frame");
        let mut output = Frame::new(16, 16).expect("frame");
        assert!(draft
            .apply(&input, &mut output, &EffectParams::new())
            .is_err());
    }

    #[test]
    fn test_draft_preview_name_and_description() {
        let pass = PassThrough;
        let draft = DraftPreview::new(pass, DownsampleFactor::Half);
        assert_eq!(draft.name(), "DraftPreview");
        assert!(!draft.description().is_empty());
    }

    #[test]
    fn test_draft_preview_set_factor() {
        let pass = PassThrough;
        let mut draft = DraftPreview::new(pass, DownsampleFactor::Half);
        assert_eq!(draft.factor(), DownsampleFactor::Half);
        draft.set_factor(DownsampleFactor::Eighth);
        assert_eq!(draft.factor(), DownsampleFactor::Eighth);
    }

    // ── Standalone API ──────────────────────────────────────────────────

    #[test]
    fn test_standalone_downsample() {
        let src = solid_frame(32, 32, [150, 150, 150, 255]);
        let dst = downsample(&src, DownsampleFactor::Half).expect("downsample");
        assert_eq!(dst.width, 16);
        assert_eq!(dst.height, 16);
    }

    #[test]
    fn test_standalone_upscale() {
        let src = solid_frame(4, 4, [100, 100, 100, 255]);
        let dst = upscale(&src, 32, 32).expect("upscale");
        assert_eq!(dst.width, 32);
        assert_eq!(dst.height, 32);
    }

    #[test]
    fn test_downsample_upscale_roundtrip_solid() {
        let original = solid_frame(32, 32, [180, 90, 45, 255]);
        let small = downsample(&original, DownsampleFactor::Half).expect("down");
        let big = upscale(&small, 32, 32).expect("up");
        // For a solid frame, roundtrip should be exact
        let p = big.get_pixel(16, 16).unwrap_or([0; 4]);
        assert_eq!(p, [180, 90, 45, 255]);
    }
}
