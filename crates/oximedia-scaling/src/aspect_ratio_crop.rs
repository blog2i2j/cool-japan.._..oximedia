//! Aspect-ratio-preserving crop and pad operations.
//!
//! Provides four layout modes for fitting an arbitrary source image into a
//! fixed target canvas:
//!
//! | Mode | Behaviour |
//! |------|-----------|
//! | `Letterbox` | Fit entirely inside canvas; add black bars on top/bottom |
//! | `Pillarbox` | Fit entirely inside canvas; add black bars left/right |
//! | `Fill`      | Crop to fill the canvas completely (no bars) |
//! | `Fit`       | Alias for the smaller of Letterbox/Pillarbox |
//!
//! An `AnchorPoint` controls where the image is positioned when there is
//! surplus space or when the crop window must be placed within the source.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use thiserror::Error;

/// Errors that can arise during crop/pad operations.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum CropError {
    /// The source or target dimensions contain a zero axis.
    #[error("zero dimension: source {src_w}x{src_h}, target {dst_w}x{dst_h}")]
    ZeroDimension {
        /// Source width.
        src_w: u32,
        /// Source height.
        src_h: u32,
        /// Destination width.
        dst_w: u32,
        /// Destination height.
        dst_h: u32,
    },
    /// The pixel buffer length does not match the declared dimensions.
    #[error("buffer length {actual} does not match {src_w}x{src_h}x{channels}")]
    BufferMismatch {
        /// Actual buffer length.
        actual: usize,
        /// Declared source width.
        src_w: u32,
        /// Declared source height.
        src_h: u32,
        /// Channel count.
        channels: u32,
    },
}

/// How to fit the source into the target canvas.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FitMode {
    /// Scale to fit entirely; letterbox (horizontal bars) if source is wider.
    Letterbox,
    /// Scale to fit entirely; pillarbox (vertical bars) if source is taller.
    Pillarbox,
    /// Scale to fill the canvas completely, cropping the excess.
    Fill,
    /// Scale to fit the canvas with no bars; equivalent to the tighter of
    /// Letterbox and Pillarbox (same as `Letterbox` in most cases).
    Fit,
}

/// Anchor point for positioning within the canvas or crop window.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnchorPoint {
    /// Top-left corner.
    TopLeft,
    /// Top-centre.
    TopCenter,
    /// Top-right corner.
    TopRight,
    /// Centre-left.
    CenterLeft,
    /// Geometric centre (default).
    Center,
    /// Centre-right.
    CenterRight,
    /// Bottom-left corner.
    BottomLeft,
    /// Bottom-centre.
    BottomCenter,
    /// Bottom-right corner.
    BottomRight,
}

impl Default for AnchorPoint {
    fn default() -> Self {
        Self::Center
    }
}

impl AnchorPoint {
    /// Returns the (horizontal_fraction, vertical_fraction) in the range [0, 1].
    ///
    /// (0, 0) is top-left; (1, 1) is bottom-right; (0.5, 0.5) is centre.
    pub fn to_fractions(self) -> (f64, f64) {
        match self {
            Self::TopLeft => (0.0, 0.0),
            Self::TopCenter => (0.5, 0.0),
            Self::TopRight => (1.0, 0.0),
            Self::CenterLeft => (0.0, 0.5),
            Self::Center => (0.5, 0.5),
            Self::CenterRight => (1.0, 0.5),
            Self::BottomLeft => (0.0, 1.0),
            Self::BottomCenter => (0.5, 1.0),
            Self::BottomRight => (1.0, 1.0),
        }
    }
}

/// The result of a crop/pad layout calculation.
///
/// All coordinates are in *target canvas* space.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayoutRegion {
    /// X offset of the scaled image inside the target canvas.
    pub canvas_x: i32,
    /// Y offset of the scaled image inside the target canvas.
    pub canvas_y: i32,
    /// Width of the scaled image (may be equal to `dst_w` for Fill).
    pub scaled_w: u32,
    /// Height of the scaled image (may be equal to `dst_h` for Fill).
    pub scaled_h: u32,
    /// X offset into the *source* image where the crop begins (Fill only).
    pub src_crop_x: i32,
    /// Y offset into the *source* image where the crop begins (Fill only).
    pub src_crop_y: i32,
}

impl LayoutRegion {
    /// Returns true if the image fills the entire canvas without any bars.
    pub fn fills_canvas(&self, dst_w: u32, dst_h: u32) -> bool {
        self.canvas_x == 0 && self.canvas_y == 0 && self.scaled_w == dst_w && self.scaled_h == dst_h
    }
}

/// Configuration for a crop/pad operation.
#[derive(Debug, Clone)]
pub struct CropPadConfig {
    /// Target canvas width.
    pub dst_w: u32,
    /// Target canvas height.
    pub dst_h: u32,
    /// Fit mode.
    pub mode: FitMode,
    /// Anchor point for positioning.
    pub anchor: AnchorPoint,
    /// Fill/pad colour (RGBA, 4 bytes).  Default is opaque black.
    pub pad_color: [u8; 4],
}

impl CropPadConfig {
    /// Create a new config with the given target dimensions and mode.
    pub fn new(dst_w: u32, dst_h: u32, mode: FitMode) -> Self {
        Self {
            dst_w,
            dst_h,
            mode,
            anchor: AnchorPoint::Center,
            pad_color: [0, 0, 0, 255],
        }
    }

    /// Builder: set the anchor point.
    pub fn with_anchor(mut self, anchor: AnchorPoint) -> Self {
        self.anchor = anchor;
        self
    }

    /// Builder: set the padding colour (RGBA).
    pub fn with_pad_color(mut self, color: [u8; 4]) -> Self {
        self.pad_color = color;
        self
    }
}

/// Calculate the [`LayoutRegion`] for fitting `src_w × src_h` into the
/// canvas described by `config`, without performing any pixel operations.
///
/// This is the pure geometry step — use [`apply_crop_pad`] to produce pixels.
pub fn calculate_layout(
    src_w: u32,
    src_h: u32,
    config: &CropPadConfig,
) -> Result<LayoutRegion, CropError> {
    if src_w == 0 || src_h == 0 || config.dst_w == 0 || config.dst_h == 0 {
        return Err(CropError::ZeroDimension {
            src_w,
            src_h,
            dst_w: config.dst_w,
            dst_h: config.dst_h,
        });
    }

    let src_aspect = src_w as f64 / src_h as f64;
    let dst_aspect = config.dst_w as f64 / config.dst_h as f64;

    let (hfrac, vfrac) = config.anchor.to_fractions();

    match config.mode {
        FitMode::Letterbox | FitMode::Fit => {
            // Scale to fit entirely; bars appear on the narrow axis.
            let (scaled_w, scaled_h) = if src_aspect >= dst_aspect {
                // Source is relatively wider — fit to width.
                let sw = config.dst_w;
                let sh = (config.dst_w as f64 / src_aspect).round() as u32;
                (sw, sh.min(config.dst_h))
            } else {
                // Source is relatively taller — fit to height.
                let sh = config.dst_h;
                let sw = (config.dst_h as f64 * src_aspect).round() as u32;
                (sw.min(config.dst_w), sh)
            };

            let padding_x = config.dst_w.saturating_sub(scaled_w) as f64 * hfrac;
            let padding_y = config.dst_h.saturating_sub(scaled_h) as f64 * vfrac;

            Ok(LayoutRegion {
                canvas_x: padding_x.round() as i32,
                canvas_y: padding_y.round() as i32,
                scaled_w,
                scaled_h,
                src_crop_x: 0,
                src_crop_y: 0,
            })
        }

        FitMode::Pillarbox => {
            // Identical geometry to Letterbox — bars appear on the perpendicular axis.
            // The difference is semantic: callers request explicit Pillarbox behaviour.
            let (scaled_w, scaled_h) = if src_aspect <= dst_aspect {
                let sh = config.dst_h;
                let sw = (config.dst_h as f64 * src_aspect).round() as u32;
                (sw.min(config.dst_w), sh)
            } else {
                let sw = config.dst_w;
                let sh = (config.dst_w as f64 / src_aspect).round() as u32;
                (sw, sh.min(config.dst_h))
            };

            let padding_x = config.dst_w.saturating_sub(scaled_w) as f64 * hfrac;
            let padding_y = config.dst_h.saturating_sub(scaled_h) as f64 * vfrac;

            Ok(LayoutRegion {
                canvas_x: padding_x.round() as i32,
                canvas_y: padding_y.round() as i32,
                scaled_w,
                scaled_h,
                src_crop_x: 0,
                src_crop_y: 0,
            })
        }

        FitMode::Fill => {
            // Scale to fill the canvas entirely, then crop the excess.
            let (scaled_w, scaled_h, crop_x, crop_y) = if src_aspect >= dst_aspect {
                // Source is relatively wider — fit to height, crop horizontally.
                let sh = config.dst_h;
                let sw = (config.dst_h as f64 * src_aspect).round() as u32;
                let excess_x = sw.saturating_sub(config.dst_w) as f64 * hfrac;
                (config.dst_w, sh, excess_x.round() as i32, 0)
            } else {
                // Source is relatively taller — fit to width, crop vertically.
                let sw = config.dst_w;
                let sh = (config.dst_w as f64 / src_aspect).round() as u32;
                let excess_y = sh.saturating_sub(config.dst_h) as f64 * vfrac;
                (sw, config.dst_h, 0, excess_y.round() as i32)
            };

            Ok(LayoutRegion {
                canvas_x: 0,
                canvas_y: 0,
                scaled_w,
                scaled_h,
                src_crop_x: crop_x,
                src_crop_y: crop_y,
            })
        }
    }
}

/// Apply the crop/pad operation to a raw RGBA pixel buffer.
///
/// `pixels` must be a row-major RGBA buffer of length `src_w * src_h * 4`.
/// The output is a row-major RGBA canvas of `config.dst_w * config.dst_h * 4`.
///
/// Scaling uses bilinear interpolation for quality.
pub fn apply_crop_pad(
    pixels: &[u8],
    src_w: u32,
    src_h: u32,
    config: &CropPadConfig,
) -> Result<Vec<u8>, CropError> {
    let channels = 4u32;
    let expected = (src_w * src_h * channels) as usize;
    if pixels.len() != expected {
        return Err(CropError::BufferMismatch {
            actual: pixels.len(),
            src_w,
            src_h,
            channels,
        });
    }

    let layout = calculate_layout(src_w, src_h, config)?;
    let dst_w = config.dst_w;
    let dst_h = config.dst_h;
    let ch = channels as usize;

    // Allocate output canvas filled with the pad colour.
    let mut output = vec![0u8; (dst_w * dst_h * channels) as usize];
    for y in 0..dst_h {
        for x in 0..dst_w {
            let idx = (y * dst_w + x) as usize * ch;
            output[idx..idx + ch].copy_from_slice(&config.pad_color);
        }
    }

    let scaled_w = layout.scaled_w;
    let scaled_h = layout.scaled_h;

    // For Fill mode the intermediate "scaled" image is dst_w × dst_h but the
    // source crop offsets tell us where in the source to start sampling.
    let (render_w, render_h) = match config.mode {
        FitMode::Fill => (dst_w, dst_h),
        _ => (scaled_w, scaled_h),
    };

    // Scale factor from destination pixel space back into source pixel space.
    let x_scale = src_w as f64 / render_w as f64;
    let y_scale = src_h as f64 / render_h as f64;

    for dy in 0..render_h {
        let canvas_y = layout.canvas_y + dy as i32;
        if canvas_y < 0 || canvas_y >= dst_h as i32 {
            continue;
        }

        // Bilinear source coordinate (continuous).
        let src_y_f = (dy as f64 + 0.5) * y_scale + layout.src_crop_y as f64 - 0.5;
        let sy0 = src_y_f.floor().clamp(0.0, (src_h - 1) as f64) as u32;
        let sy1 = (sy0 + 1).min(src_h - 1);
        let ty = src_y_f - src_y_f.floor();

        for dx in 0..render_w {
            let canvas_x = layout.canvas_x + dx as i32;
            if canvas_x < 0 || canvas_x >= dst_w as i32 {
                continue;
            }

            let src_x_f = (dx as f64 + 0.5) * x_scale + layout.src_crop_x as f64 - 0.5;
            let sx0 = src_x_f.floor().clamp(0.0, (src_w - 1) as f64) as u32;
            let sx1 = (sx0 + 1).min(src_w - 1);
            let tx = src_x_f - src_x_f.floor();

            let dst_idx = (canvas_y as u32 * dst_w + canvas_x as u32) as usize * ch;

            // Bilinear sample — 4 taps.
            for c in 0..ch {
                let p00 = pixels[((sy0 * src_w + sx0) as usize) * ch + c] as f64;
                let p10 = pixels[((sy0 * src_w + sx1) as usize) * ch + c] as f64;
                let p01 = pixels[((sy1 * src_w + sx0) as usize) * ch + c] as f64;
                let p11 = pixels[((sy1 * src_w + sx1) as usize) * ch + c] as f64;

                let v = p00 * (1.0 - tx) * (1.0 - ty)
                    + p10 * tx * (1.0 - ty)
                    + p01 * (1.0 - tx) * ty
                    + p11 * tx * ty;

                output[dst_idx + c] = v.round().clamp(0.0, 255.0) as u8;
            }
        }
    }

    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── helper ───────────────────────────────────────────────────────────────

    fn solid_rgba(w: u32, h: u32, color: [u8; 4]) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 4) as usize);
        for _ in 0..w * h {
            v.extend_from_slice(&color);
        }
        v
    }

    // ── calculate_layout tests ───────────────────────────────────────────────

    #[test]
    fn test_zero_src_returns_error() {
        let cfg = CropPadConfig::new(1920, 1080, FitMode::Letterbox);
        let err = calculate_layout(0, 480, &cfg).unwrap_err();
        assert!(matches!(err, CropError::ZeroDimension { .. }));
    }

    #[test]
    fn test_zero_dst_returns_error() {
        let cfg = CropPadConfig::new(0, 1080, FitMode::Letterbox);
        let err = calculate_layout(720, 480, &cfg).unwrap_err();
        assert!(matches!(err, CropError::ZeroDimension { .. }));
    }

    #[test]
    fn test_letterbox_wider_source() {
        // 16:9 source into 4:3 canvas → horizontal bars.
        let cfg = CropPadConfig::new(800, 600, FitMode::Letterbox);
        let layout = calculate_layout(1920, 1080, &cfg).unwrap();
        // Scaled width should equal canvas width (800).
        assert_eq!(layout.scaled_w, 800);
        // Bars should appear above/below.
        assert!(
            layout.canvas_y > 0,
            "expected top bar, canvas_y={}",
            layout.canvas_y
        );
        assert_eq!(layout.canvas_x, 0);
    }

    #[test]
    fn test_letterbox_taller_source() {
        // 4:3 source into 16:9 canvas → vertical bars.
        let cfg = CropPadConfig::new(1920, 1080, FitMode::Letterbox);
        let layout = calculate_layout(640, 480, &cfg).unwrap();
        assert!(
            layout.canvas_x > 0,
            "expected left bar, canvas_x={}",
            layout.canvas_x
        );
    }

    #[test]
    fn test_fill_fills_canvas() {
        let cfg = CropPadConfig::new(1920, 1080, FitMode::Fill);
        let layout = calculate_layout(640, 480, &cfg).unwrap();
        assert!(
            layout.fills_canvas(1920, 1080),
            "Fill should cover entire canvas"
        );
    }

    #[test]
    fn test_fill_wide_source_crops_horizontally() {
        // 16:9 source into 4:3 canvas.
        let cfg = CropPadConfig::new(800, 600, FitMode::Fill).with_anchor(AnchorPoint::CenterLeft);
        let layout = calculate_layout(1920, 1080, &cfg).unwrap();
        assert!(layout.fills_canvas(800, 600));
        // CenterLeft anchor → crop should start at the left edge.
        assert_eq!(layout.src_crop_x, 0);
    }

    #[test]
    fn test_anchor_top_left_letterbox() {
        let cfg =
            CropPadConfig::new(1920, 1080, FitMode::Letterbox).with_anchor(AnchorPoint::TopLeft);
        let layout = calculate_layout(640, 480, &cfg).unwrap();
        // Bars should be to the right (canvas_x == 0) and bars only on the right
        // for a 4:3 source in 16:9 canvas with TopLeft anchor.
        assert_eq!(layout.canvas_x, 0);
        assert_eq!(layout.canvas_y, 0);
    }

    #[test]
    fn test_anchor_bottom_right_letterbox() {
        let cfg = CropPadConfig::new(1920, 1080, FitMode::Letterbox)
            .with_anchor(AnchorPoint::BottomRight);
        let layout_br = calculate_layout(640, 480, &cfg).unwrap();

        let cfg2 =
            CropPadConfig::new(1920, 1080, FitMode::Letterbox).with_anchor(AnchorPoint::TopLeft);
        let layout_tl = calculate_layout(640, 480, &cfg2).unwrap();

        // BottomRight should have larger canvas_x than TopLeft.
        assert!(
            layout_br.canvas_x >= layout_tl.canvas_x,
            "BottomRight canvas_x={} should be >= TopLeft canvas_x={}",
            layout_br.canvas_x,
            layout_tl.canvas_x
        );
    }

    #[test]
    fn test_fit_mode_same_as_letterbox_for_wider_source() {
        let cfg_lb = CropPadConfig::new(1920, 1080, FitMode::Letterbox);
        let cfg_fit = CropPadConfig::new(1920, 1080, FitMode::Fit);
        let layout_lb = calculate_layout(1920, 1080, &cfg_lb).unwrap();
        let layout_fit = calculate_layout(1920, 1080, &cfg_fit).unwrap();
        assert_eq!(layout_lb, layout_fit);
    }

    #[test]
    fn test_pillarbox_same_aspect_no_bars() {
        // Exact same aspect ratio — no bars.
        let cfg = CropPadConfig::new(1920, 1080, FitMode::Pillarbox);
        let layout = calculate_layout(1920, 1080, &cfg).unwrap();
        assert!(layout.fills_canvas(1920, 1080));
    }

    // ── apply_crop_pad tests ─────────────────────────────────────────────────

    #[test]
    fn test_apply_buffer_mismatch_error() {
        let cfg = CropPadConfig::new(100, 100, FitMode::Letterbox);
        let result = apply_crop_pad(&[0u8; 10], 16, 16, &cfg);
        assert!(matches!(result, Err(CropError::BufferMismatch { .. })));
    }

    #[test]
    fn test_apply_letterbox_output_size() {
        let pixels = solid_rgba(640, 480, [255, 0, 0, 255]);
        let cfg = CropPadConfig::new(1920, 1080, FitMode::Letterbox);
        let output = apply_crop_pad(&pixels, 640, 480, &cfg).unwrap();
        assert_eq!(output.len(), (1920 * 1080 * 4) as usize);
    }

    #[test]
    fn test_apply_fill_output_size() {
        let pixels = solid_rgba(1920, 1080, [0, 255, 0, 255]);
        let cfg = CropPadConfig::new(800, 600, FitMode::Fill);
        let output = apply_crop_pad(&pixels, 1920, 1080, &cfg).unwrap();
        assert_eq!(output.len(), (800 * 600 * 4) as usize);
    }

    #[test]
    fn test_apply_solid_color_preserved() {
        // A solid red canvas should remain red after fit.
        let pixels = solid_rgba(100, 100, [200, 100, 50, 255]);
        let cfg = CropPadConfig::new(50, 50, FitMode::Fill);
        let output = apply_crop_pad(&pixels, 100, 100, &cfg).unwrap();
        // Every pixel should be approximately red.
        for i in 0..(50 * 50usize) {
            assert_eq!(output[i * 4], 200, "R channel at pixel {i}");
            assert_eq!(output[i * 4 + 1], 100, "G channel at pixel {i}");
            assert_eq!(output[i * 4 + 2], 50, "B channel at pixel {i}");
        }
    }

    #[test]
    fn test_pad_color_applied_in_bars() {
        // 4:3 source into 16:9 canvas with bright green bars.
        let pixels = solid_rgba(640, 480, [255, 255, 255, 255]);
        let cfg = CropPadConfig::new(1920, 1080, FitMode::Letterbox)
            .with_pad_color([0, 255, 0, 255])
            .with_anchor(AnchorPoint::Center);
        let output = apply_crop_pad(&pixels, 640, 480, &cfg).unwrap();
        // Top-left pixel should be in the bar (green).
        assert_eq!(output[0], 0, "R should be 0 in bar");
        assert_eq!(output[1], 255, "G should be 255 in bar");
    }

    #[test]
    fn test_anchor_fractions_all_variants() {
        use AnchorPoint::*;
        let anchors = [
            (TopLeft, (0.0_f64, 0.0_f64)),
            (TopCenter, (0.5, 0.0)),
            (TopRight, (1.0, 0.0)),
            (CenterLeft, (0.0, 0.5)),
            (Center, (0.5, 0.5)),
            (CenterRight, (1.0, 0.5)),
            (BottomLeft, (0.0, 1.0)),
            (BottomCenter, (0.5, 1.0)),
            (BottomRight, (1.0, 1.0)),
        ];
        for (anchor, (hf, vf)) in anchors {
            let (h, v) = anchor.to_fractions();
            assert!((h - hf).abs() < 1e-9, "{anchor:?} hfrac");
            assert!((v - vf).abs() < 1e-9, "{anchor:?} vfrac");
        }
    }
}
