//! Color space conversion for video frames.
//!
//! Provides YUV↔RGB conversion supporting BT.601, BT.709, and BT.2020
//! color primaries, with both limited-range (studio swing) and full-range
//! (PC/JPEG swing) handling, plus chroma subsampling format conversion
//! (4:4:4, 4:2:2, 4:2:0).
//!
//! # Overview
//!
//! | Standard | Usage |
//! |----------|-------|
//! | BT.601   | SD (480i/576i), legacy video |
//! | BT.709   | HD (720p/1080p), modern SDR |
//! | BT.2020  | UHD (4K/8K), HDR content |
//!
//! Limited-range Y is `[16, 235]`; Cb/Cr are `[16, 240]`.
//! Full-range Y, Cb, Cr are all `[0, 255]`.

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors that can occur during color space conversion.
#[derive(Debug, thiserror::Error)]
pub enum ColorConvertError {
    /// Input buffer size does not match the declared dimensions.
    #[error("buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch {
        /// Expected buffer length.
        expected: usize,
        /// Actual buffer length.
        actual: usize,
    },
    /// Width or height is zero.
    #[error("invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Frame width.
        width: u32,
        /// Frame height.
        height: u32,
    },
    /// Width is not a multiple of 2, required for chroma subsampling.
    #[error("width {width} must be a multiple of 2 for chroma subsampling")]
    OddWidth {
        /// Frame width.
        width: u32,
    },
    /// Height is not a multiple of 2, required for 4:2:0 subsampling.
    #[error("height {height} must be a multiple of 2 for 4:2:0 subsampling")]
    OddHeight {
        /// Frame height.
        height: u32,
    },
}

// -----------------------------------------------------------------------
// Public types
// -----------------------------------------------------------------------

/// Color primaries / transfer matrix standard.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorStandard {
    /// ITU-R BT.601 — standard-definition video.
    Bt601,
    /// ITU-R BT.709 — high-definition video (default for modern SDR).
    #[default]
    Bt709,
    /// ITU-R BT.2020 — ultra-high-definition / HDR video.
    Bt2020,
}

/// Quantisation range for luma and chroma samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum QuantRange {
    /// Limited (studio) range: Y ∈ \[16,235\], Cb/Cr ∈ \[16,240\].
    #[default]
    Limited,
    /// Full (PC/JPEG) range: Y, Cb, Cr ∈ \[0,255\].
    Full,
}

/// Chroma subsampling layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChromaSubsampling {
    /// 4:4:4 — full chroma resolution, one Cb/Cr per luma sample.
    Yuv444,
    /// 4:2:2 — horizontal halving: one Cb/Cr per two horizontal luma samples.
    Yuv422,
    /// 4:2:0 — both horizontal and vertical halving.
    Yuv420,
}

/// Packed RGB pixel (R, G, B each 0–255).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rgb {
    /// Red channel.
    pub r: u8,
    /// Green channel.
    pub g: u8,
    /// Blue channel.
    pub b: u8,
}

/// Planar YCbCr frame with configurable subsampling.
#[derive(Debug, Clone)]
pub struct YcbcrFrame {
    /// Luma plane — `width × height` bytes, row-major.
    pub y: Vec<u8>,
    /// Cb (blue-difference) chroma plane.
    pub cb: Vec<u8>,
    /// Cr (red-difference) chroma plane.
    pub cr: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Chroma subsampling layout.
    pub subsampling: ChromaSubsampling,
}

impl YcbcrFrame {
    /// Returns the expected number of chroma samples for this frame's
    /// subsampling layout.
    pub fn chroma_size(&self) -> usize {
        let w = self.width as usize;
        let h = self.height as usize;
        match self.subsampling {
            ChromaSubsampling::Yuv444 => w * h,
            ChromaSubsampling::Yuv422 => (w / 2) * h,
            ChromaSubsampling::Yuv420 => (w / 2) * (h / 2),
        }
    }
}

// -----------------------------------------------------------------------
// Internal coefficient tables
// -----------------------------------------------------------------------

/// Luma weights (Kr, Kb) per standard.  Kg = 1 − Kr − Kb.
const fn kr_kb(std: ColorStandard) -> (f64, f64) {
    match std {
        ColorStandard::Bt601 => (0.299, 0.114),
        ColorStandard::Bt709 => (0.2126, 0.0722),
        ColorStandard::Bt2020 => (0.2627, 0.0593),
    }
}

// -----------------------------------------------------------------------
// Single-pixel conversions (floating-point, range-aware)
// -----------------------------------------------------------------------

/// Convert a single RGB pixel to YCbCr using the given standard and range.
///
/// Returns `(Y, Cb, Cr)` as `u8` values in the requested quantisation range.
pub fn rgb_to_ycbcr(rgb: Rgb, std: ColorStandard, range: QuantRange) -> (u8, u8, u8) {
    let r = rgb.r as f64 / 255.0;
    let g = rgb.g as f64 / 255.0;
    let b = rgb.b as f64 / 255.0;

    let (kr, kb) = kr_kb(std);
    let kg = 1.0 - kr - kb;

    let y_lin = kr * r + kg * g + kb * b;
    let cb_lin = (b - y_lin) / (2.0 * (1.0 - kb));
    let cr_lin = (r - y_lin) / (2.0 * (1.0 - kr));

    match range {
        QuantRange::Full => {
            let y = clamp_u8(y_lin * 255.0);
            let cb = clamp_u8(cb_lin * 255.0 + 128.0);
            let cr = clamp_u8(cr_lin * 255.0 + 128.0);
            (y, cb, cr)
        }
        QuantRange::Limited => {
            // Y: [16, 235], Cb/Cr: [16, 240]
            let y = clamp_u8(y_lin * 219.0 + 16.0);
            let cb = clamp_u8(cb_lin * 224.0 + 128.0);
            let cr = clamp_u8(cr_lin * 224.0 + 128.0);
            (y, cb, cr)
        }
    }
}

/// Convert a single YCbCr sample to RGB using the given standard and range.
///
/// Returns [`Rgb`] with channels clamped to [0, 255].
pub fn ycbcr_to_rgb(y: u8, cb: u8, cr: u8, std: ColorStandard, range: QuantRange) -> Rgb {
    let (y_f, cb_f, cr_f) = match range {
        QuantRange::Full => (
            y as f64 / 255.0,
            (cb as f64 - 128.0) / 255.0,
            (cr as f64 - 128.0) / 255.0,
        ),
        QuantRange::Limited => (
            (y as f64 - 16.0) / 219.0,
            (cb as f64 - 128.0) / 224.0,
            (cr as f64 - 128.0) / 224.0,
        ),
    };

    let (kr, kb) = kr_kb(std);
    let kg = 1.0 - kr - kb;

    let r = y_f + 2.0 * (1.0 - kr) * cr_f;
    let g = y_f - (2.0 * kb * (1.0 - kb) / kg) * cb_f - (2.0 * kr * (1.0 - kr) / kg) * cr_f;
    let b = y_f + 2.0 * (1.0 - kb) * cb_f;

    Rgb {
        r: clamp_u8(r * 255.0),
        g: clamp_u8(g * 255.0),
        b: clamp_u8(b * 255.0),
    }
}

// -----------------------------------------------------------------------
// Packed RGB ↔ planar YCbCr frame conversions
// -----------------------------------------------------------------------

/// Convert a packed RGB frame (3 bytes per pixel, R-G-B order) to a planar
/// [`YcbcrFrame`].
///
/// # Arguments
///
/// * `rgb` – row-major packed RGB buffer, `width * height * 3` bytes.
/// * `width` / `height` – frame dimensions.
/// * `subsampling` – desired chroma subsampling.
/// * `std` – colour matrix standard.
/// * `range` – quantisation range.
///
/// # Errors
///
/// Returns [`ColorConvertError`] if dimensions are invalid or the buffer is
/// the wrong size.
pub fn rgb_frame_to_ycbcr(
    rgb: &[u8],
    width: u32,
    height: u32,
    subsampling: ChromaSubsampling,
    std: ColorStandard,
    range: QuantRange,
) -> Result<YcbcrFrame, ColorConvertError> {
    validate_dims(width, height)?;
    if subsampling != ChromaSubsampling::Yuv444 {
        if width % 2 != 0 {
            return Err(ColorConvertError::OddWidth { width });
        }
        if subsampling == ChromaSubsampling::Yuv420 && height % 2 != 0 {
            return Err(ColorConvertError::OddHeight { height });
        }
    }
    let expected = width as usize * height as usize * 3;
    if rgb.len() != expected {
        return Err(ColorConvertError::BufferSizeMismatch {
            expected,
            actual: rgb.len(),
        });
    }

    let w = width as usize;
    let h = height as usize;
    let npix = w * h;

    // Convert every pixel to YCbCr at full 4:4:4 resolution first.
    let mut y_plane = vec![0u8; npix];
    let mut cb_full = vec![0u8; npix];
    let mut cr_full = vec![0u8; npix];

    for row in 0..h {
        for col in 0..w {
            let base = (row * w + col) * 3;
            let pixel = Rgb {
                r: rgb[base],
                g: rgb[base + 1],
                b: rgb[base + 2],
            };
            let (y, cb, cr) = rgb_to_ycbcr(pixel, std, range);
            let idx = row * w + col;
            y_plane[idx] = y;
            cb_full[idx] = cb;
            cr_full[idx] = cr;
        }
    }

    // Downsample chroma planes if needed.
    let (cb, cr) = match subsampling {
        ChromaSubsampling::Yuv444 => (cb_full, cr_full),
        ChromaSubsampling::Yuv422 => downsample_422(w, h, &cb_full, &cr_full),
        ChromaSubsampling::Yuv420 => downsample_420(w, h, &cb_full, &cr_full),
    };

    Ok(YcbcrFrame {
        y: y_plane,
        cb,
        cr,
        width,
        height,
        subsampling,
    })
}

/// Convert a planar [`YcbcrFrame`] back to a packed RGB buffer (R-G-B order).
///
/// # Errors
///
/// Returns [`ColorConvertError`] if any plane has an unexpected size or
/// dimensions are invalid.
pub fn ycbcr_frame_to_rgb(
    frame: &YcbcrFrame,
    std: ColorStandard,
    range: QuantRange,
) -> Result<Vec<u8>, ColorConvertError> {
    validate_dims(frame.width, frame.height)?;
    let w = frame.width as usize;
    let h = frame.height as usize;
    let npix = w * h;

    if frame.y.len() != npix {
        return Err(ColorConvertError::BufferSizeMismatch {
            expected: npix,
            actual: frame.y.len(),
        });
    }
    let chroma_expected = frame.chroma_size();
    if frame.cb.len() != chroma_expected {
        return Err(ColorConvertError::BufferSizeMismatch {
            expected: chroma_expected,
            actual: frame.cb.len(),
        });
    }
    if frame.cr.len() != chroma_expected {
        return Err(ColorConvertError::BufferSizeMismatch {
            expected: chroma_expected,
            actual: frame.cr.len(),
        });
    }

    // Upsample chroma to 4:4:4 if needed.
    let (cb_full, cr_full) = match frame.subsampling {
        ChromaSubsampling::Yuv444 => (frame.cb.clone(), frame.cr.clone()),
        ChromaSubsampling::Yuv422 => upsample_422(w, h, &frame.cb, &frame.cr),
        ChromaSubsampling::Yuv420 => upsample_420(w, h, &frame.cb, &frame.cr),
    };

    let mut out = vec![0u8; npix * 3];
    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            let rgb = ycbcr_to_rgb(frame.y[idx], cb_full[idx], cr_full[idx], std, range);
            let base = idx * 3;
            out[base] = rgb.r;
            out[base + 1] = rgb.g;
            out[base + 2] = rgb.b;
        }
    }
    Ok(out)
}

// -----------------------------------------------------------------------
// Chroma range conversion (limited ↔ full)
// -----------------------------------------------------------------------

/// Convert a [`YcbcrFrame`] between limited and full quantisation ranges
/// **in place** without changing the colour matrix.
///
/// # Errors
///
/// Returns [`ColorConvertError`] if the frame's planes have unexpected sizes.
pub fn convert_range(
    frame: &mut YcbcrFrame,
    from: QuantRange,
    to: QuantRange,
) -> Result<(), ColorConvertError> {
    if from == to {
        return Ok(());
    }
    validate_dims(frame.width, frame.height)?;
    let npix = frame.width as usize * frame.height as usize;
    if frame.y.len() != npix {
        return Err(ColorConvertError::BufferSizeMismatch {
            expected: npix,
            actual: frame.y.len(),
        });
    }

    match (from, to) {
        (QuantRange::Limited, QuantRange::Full) => {
            for v in frame.y.iter_mut() {
                *v = limited_luma_to_full(*v);
            }
            for v in frame.cb.iter_mut() {
                *v = limited_chroma_to_full(*v);
            }
            for v in frame.cr.iter_mut() {
                *v = limited_chroma_to_full(*v);
            }
        }
        (QuantRange::Full, QuantRange::Limited) => {
            for v in frame.y.iter_mut() {
                *v = full_luma_to_limited(*v);
            }
            for v in frame.cb.iter_mut() {
                *v = full_chroma_to_limited(*v);
            }
            for v in frame.cr.iter_mut() {
                *v = full_chroma_to_limited(*v);
            }
        }
        _ => {}
    }
    Ok(())
}

// -----------------------------------------------------------------------
// Chroma subsampling re-conversion
// -----------------------------------------------------------------------

/// Re-sample a [`YcbcrFrame`]'s chroma planes to a different subsampling layout.
///
/// The luma plane is left unchanged.
///
/// # Errors
///
/// Returns [`ColorConvertError`] if target requires even dimensions and
/// `frame` has odd width/height, or if existing plane sizes are inconsistent.
pub fn convert_subsampling(
    frame: &YcbcrFrame,
    target: ChromaSubsampling,
) -> Result<YcbcrFrame, ColorConvertError> {
    if frame.subsampling == target {
        return Ok(frame.clone());
    }
    validate_dims(frame.width, frame.height)?;
    let w = frame.width as usize;
    let h = frame.height as usize;

    if target != ChromaSubsampling::Yuv444 {
        if frame.width % 2 != 0 {
            return Err(ColorConvertError::OddWidth { width: frame.width });
        }
        if target == ChromaSubsampling::Yuv420 && frame.height % 2 != 0 {
            return Err(ColorConvertError::OddHeight {
                height: frame.height,
            });
        }
    }

    // Upsample to 4:4:4 then downsample to target.
    let (cb_444, cr_444) = match frame.subsampling {
        ChromaSubsampling::Yuv444 => (frame.cb.clone(), frame.cr.clone()),
        ChromaSubsampling::Yuv422 => upsample_422(w, h, &frame.cb, &frame.cr),
        ChromaSubsampling::Yuv420 => upsample_420(w, h, &frame.cb, &frame.cr),
    };

    let (cb, cr) = match target {
        ChromaSubsampling::Yuv444 => (cb_444, cr_444),
        ChromaSubsampling::Yuv422 => downsample_422(w, h, &cb_444, &cr_444),
        ChromaSubsampling::Yuv420 => downsample_420(w, h, &cb_444, &cr_444),
    };

    Ok(YcbcrFrame {
        y: frame.y.clone(),
        cb,
        cr,
        width: frame.width,
        height: frame.height,
        subsampling: target,
    })
}

// -----------------------------------------------------------------------
// Internal helpers — chroma downsampling / upsampling
// -----------------------------------------------------------------------

/// Average-box downsample chroma from 4:4:4 to 4:2:2 (horizontal halving).
fn downsample_422(w: usize, h: usize, cb: &[u8], cr: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let cw = w / 2;
    let mut cb_out = vec![0u8; cw * h];
    let mut cr_out = vec![0u8; cw * h];
    for row in 0..h {
        for col in 0..cw {
            let i0 = row * w + col * 2;
            let i1 = i0 + 1;
            cb_out[row * cw + col] = avg2(cb[i0], cb[i1]);
            cr_out[row * cw + col] = avg2(cr[i0], cr[i1]);
        }
    }
    (cb_out, cr_out)
}

/// Average-box downsample chroma from 4:4:4 to 4:2:0 (both halving).
fn downsample_420(w: usize, h: usize, cb: &[u8], cr: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let cw = w / 2;
    let ch = h / 2;
    let mut cb_out = vec![0u8; cw * ch];
    let mut cr_out = vec![0u8; cw * ch];
    for row in 0..ch {
        for col in 0..cw {
            let i00 = (row * 2) * w + col * 2;
            let i01 = i00 + 1;
            let i10 = i00 + w;
            let i11 = i10 + 1;
            cb_out[row * cw + col] = avg4(cb[i00], cb[i01], cb[i10], cb[i11]);
            cr_out[row * cw + col] = avg4(cr[i00], cr[i01], cr[i10], cr[i11]);
        }
    }
    (cb_out, cr_out)
}

/// Nearest-neighbour upsample from 4:2:2 to 4:4:4.
fn upsample_422(w: usize, h: usize, cb: &[u8], cr: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let cw = w / 2;
    let mut cb_out = vec![0u8; w * h];
    let mut cr_out = vec![0u8; w * h];
    for row in 0..h {
        for col in 0..cw {
            let src = row * cw + col;
            let dst0 = row * w + col * 2;
            let dst1 = dst0 + 1;
            cb_out[dst0] = cb[src];
            cb_out[dst1] = cb[src];
            cr_out[dst0] = cr[src];
            cr_out[dst1] = cr[src];
        }
    }
    (cb_out, cr_out)
}

/// Nearest-neighbour upsample from 4:2:0 to 4:4:4.
fn upsample_420(w: usize, h: usize, cb: &[u8], cr: &[u8]) -> (Vec<u8>, Vec<u8>) {
    let cw = w / 2;
    let ch = h / 2;
    let mut cb_out = vec![0u8; w * h];
    let mut cr_out = vec![0u8; w * h];
    for row in 0..ch {
        for col in 0..cw {
            let src = row * cw + col;
            let dst00 = (row * 2) * w + col * 2;
            let dst01 = dst00 + 1;
            let dst10 = dst00 + w;
            let dst11 = dst10 + 1;
            cb_out[dst00] = cb[src];
            cb_out[dst01] = cb[src];
            cb_out[dst10] = cb[src];
            cb_out[dst11] = cb[src];
            cr_out[dst00] = cr[src];
            cr_out[dst01] = cr[src];
            cr_out[dst10] = cr[src];
            cr_out[dst11] = cr[src];
        }
    }
    (cb_out, cr_out)
}

// -----------------------------------------------------------------------
// Internal helpers — range mapping
// -----------------------------------------------------------------------

#[inline]
fn limited_luma_to_full(v: u8) -> u8 {
    // Y_full = (Y_lim - 16) * 255 / 219
    let x = (v as i32 - 16).max(0);
    clamp_u8(x as f64 * 255.0 / 219.0)
}

#[inline]
fn limited_chroma_to_full(v: u8) -> u8 {
    // C_full = (C_lim - 128) * 255 / 224 + 128
    let x = v as f64 - 128.0;
    clamp_u8(x * 255.0 / 224.0 + 128.0)
}

#[inline]
fn full_luma_to_limited(v: u8) -> u8 {
    clamp_u8(v as f64 * 219.0 / 255.0 + 16.0)
}

#[inline]
fn full_chroma_to_limited(v: u8) -> u8 {
    let x = v as f64 - 128.0;
    clamp_u8(x * 224.0 / 255.0 + 128.0)
}

#[inline]
fn avg2(a: u8, b: u8) -> u8 {
    ((a as u16 + b as u16 + 1) / 2) as u8
}

#[inline]
fn avg4(a: u8, b: u8, c: u8, d: u8) -> u8 {
    ((a as u16 + b as u16 + c as u16 + d as u16 + 2) / 4) as u8
}

#[inline]
fn clamp_u8(v: f64) -> u8 {
    v.round().clamp(0.0, 255.0) as u8
}

fn validate_dims(width: u32, height: u32) -> Result<(), ColorConvertError> {
    if width == 0 || height == 0 {
        Err(ColorConvertError::InvalidDimensions { width, height })
    } else {
        Ok(())
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- Round-trip helpers ---

    fn make_rgb_frame(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        vec![[r, g, b]; (w * h) as usize]
            .into_iter()
            .flatten()
            .collect()
    }

    // 1. Pure white round-trip (4:4:4, BT.709, limited)
    #[test]
    fn test_white_roundtrip_444_limited() {
        let w = 4u32;
        let h = 4u32;
        let src = make_rgb_frame(w, h, 255, 255, 255);
        let yuv = rgb_frame_to_ycbcr(
            &src,
            w,
            h,
            ChromaSubsampling::Yuv444,
            ColorStandard::Bt709,
            QuantRange::Limited,
        )
        .unwrap();
        let out = ycbcr_frame_to_rgb(&yuv, ColorStandard::Bt709, QuantRange::Limited).unwrap();
        for (s, o) in src.iter().zip(out.iter()) {
            let diff = (*s as i16 - *o as i16).abs();
            assert!(diff <= 2, "channel mismatch: src={s} out={o}");
        }
    }

    // 2. Pure black round-trip (4:4:4, BT.601, full)
    #[test]
    fn test_black_roundtrip_444_full() {
        let w = 4u32;
        let h = 4u32;
        let src = make_rgb_frame(w, h, 0, 0, 0);
        let yuv = rgb_frame_to_ycbcr(
            &src,
            w,
            h,
            ChromaSubsampling::Yuv444,
            ColorStandard::Bt601,
            QuantRange::Full,
        )
        .unwrap();
        let out = ycbcr_frame_to_rgb(&yuv, ColorStandard::Bt601, QuantRange::Full).unwrap();
        for (s, o) in src.iter().zip(out.iter()) {
            let diff = (*s as i16 - *o as i16).abs();
            assert!(diff <= 2, "channel mismatch: src={s} out={o}");
        }
    }

    // 3. Red primary conversion checks for BT.709
    #[test]
    fn test_red_pixel_bt709_full() {
        let (y, cb, cr) = rgb_to_ycbcr(
            Rgb { r: 255, g: 0, b: 0 },
            ColorStandard::Bt709,
            QuantRange::Full,
        );
        // For pure red with BT.709: Y ≈ 0.2126*255 ≈ 54; Cr should be high; Cb low
        assert!(y > 40 && y < 70, "Y={y}");
        assert!(cr > 128, "Cr={cr}");
        assert!(cb < 128, "Cb={cb}");
    }

    // 4. Green primary
    #[test]
    fn test_green_pixel_bt709_full() {
        let (y, _cb, _cr) = rgb_to_ycbcr(
            Rgb { r: 0, g: 255, b: 0 },
            ColorStandard::Bt709,
            QuantRange::Full,
        );
        // Green has highest luma weight in BT.709 (0.7152)
        assert!(y > 150, "Y={y}");
    }

    // 5. Blue primary
    #[test]
    fn test_blue_pixel_bt709_full() {
        let (y, cb, cr) = rgb_to_ycbcr(
            Rgb { r: 0, g: 0, b: 255 },
            ColorStandard::Bt709,
            QuantRange::Full,
        );
        // Kb=0.0722 → Y ≈ 18
        assert!(y < 30, "Y={y}");
        assert!(cb > 128, "Cb={cb} should be high for blue");
        let _ = cr;
    }

    // 6. 4:2:0 round-trip
    #[test]
    fn test_roundtrip_420() {
        let w = 4u32;
        let h = 4u32;
        let src = make_rgb_frame(w, h, 100, 150, 200);
        let yuv = rgb_frame_to_ycbcr(
            &src,
            w,
            h,
            ChromaSubsampling::Yuv420,
            ColorStandard::Bt709,
            QuantRange::Full,
        )
        .unwrap();
        assert_eq!(yuv.cb.len(), 4);
        assert_eq!(yuv.cr.len(), 4);
        let out = ycbcr_frame_to_rgb(&yuv, ColorStandard::Bt709, QuantRange::Full).unwrap();
        for (s, o) in src.iter().zip(out.iter()) {
            let diff = (*s as i16 - *o as i16).abs();
            assert!(diff <= 4, "channel mismatch: src={s} out={o}");
        }
    }

    // 7. 4:2:2 chroma plane size
    #[test]
    fn test_422_chroma_size() {
        let w = 8u32;
        let h = 4u32;
        let src = make_rgb_frame(w, h, 120, 80, 40);
        let yuv = rgb_frame_to_ycbcr(
            &src,
            w,
            h,
            ChromaSubsampling::Yuv422,
            ColorStandard::Bt601,
            QuantRange::Limited,
        )
        .unwrap();
        assert_eq!(yuv.cb.len(), (w / 2 * h) as usize);
        assert_eq!(yuv.cr.len(), (w / 2 * h) as usize);
    }

    // 8. Range conversion limited→full→limited is close to identity
    #[test]
    fn test_range_convert_roundtrip() {
        let w = 4u32;
        let h = 4u32;
        let src = make_rgb_frame(w, h, 128, 64, 192);
        let mut yuv = rgb_frame_to_ycbcr(
            &src,
            w,
            h,
            ChromaSubsampling::Yuv444,
            ColorStandard::Bt709,
            QuantRange::Limited,
        )
        .unwrap();
        let orig_y = yuv.y.clone();
        convert_range(&mut yuv, QuantRange::Limited, QuantRange::Full).unwrap();
        convert_range(&mut yuv, QuantRange::Full, QuantRange::Limited).unwrap();
        for (a, b) in orig_y.iter().zip(yuv.y.iter()) {
            let diff = (*a as i16 - *b as i16).abs();
            assert!(diff <= 2, "Y drift: {a} -> {b}");
        }
    }

    // 9. Subsampling conversion 444→420→444 chroma is stable
    #[test]
    fn test_subsampling_convert() {
        let w = 4u32;
        let h = 4u32;
        let src = make_rgb_frame(w, h, 80, 160, 40);
        let yuv444 = rgb_frame_to_ycbcr(
            &src,
            w,
            h,
            ChromaSubsampling::Yuv444,
            ColorStandard::Bt709,
            QuantRange::Full,
        )
        .unwrap();
        let yuv420 = convert_subsampling(&yuv444, ChromaSubsampling::Yuv420).unwrap();
        assert_eq!(yuv420.chroma_size(), 4);
        let yuv444b = convert_subsampling(&yuv420, ChromaSubsampling::Yuv444).unwrap();
        assert_eq!(yuv444b.chroma_size(), 16);
    }

    // 10. BT.2020 luma coefficient check
    #[test]
    fn test_bt2020_luma() {
        // BT.2020 Kr=0.2627
        let (y, _, _) = rgb_to_ycbcr(
            Rgb { r: 255, g: 0, b: 0 },
            ColorStandard::Bt2020,
            QuantRange::Full,
        );
        let expected = (0.2627f64 * 255.0).round() as u8;
        let diff = (y as i16 - expected as i16).abs();
        assert!(diff <= 2, "BT.2020 red Y={y} expected≈{expected}");
    }

    // 11. Invalid dimensions error
    #[test]
    fn test_invalid_dimensions() {
        let result = rgb_frame_to_ycbcr(
            &[],
            0,
            4,
            ChromaSubsampling::Yuv444,
            ColorStandard::Bt709,
            QuantRange::Full,
        );
        assert!(matches!(
            result,
            Err(ColorConvertError::InvalidDimensions { .. })
        ));
    }

    // 12. Buffer size mismatch error
    #[test]
    fn test_buffer_size_mismatch() {
        let result = rgb_frame_to_ycbcr(
            &[0u8; 10],
            4,
            4,
            ChromaSubsampling::Yuv444,
            ColorStandard::Bt709,
            QuantRange::Full,
        );
        assert!(matches!(
            result,
            Err(ColorConvertError::BufferSizeMismatch { .. })
        ));
    }
}
