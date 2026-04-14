//! Porter-Duff alpha compositing operators.
//!
//! Implements the full set of Porter-Duff compositing modes for RGBA images
//! (stored as `u8` per channel, 4 channels per pixel).  All operations work in
//! **straight (un-premultiplied) alpha** by default; a `premultiplied` flag
//! allows operating on data that is already in premultiplied form.
//!
//! Supported operators (as defined in "Compositing Digital Images", Porter &
//! Duff, SIGGRAPH 1984):
//!
//! | Operator   | Formula (premultiplied)          |
//! |------------|----------------------------------|
//! | Over       | A + B × (1 − αA)                |
//! | Under      | A × (1 − αB) + B                |
//! | In         | A × αB                          |
//! | Out        | A × (1 − αB)                    |
//! | Atop       | A × αB + B × (1 − αA)           |
//! | Xor        | A × (1 − αB) + B × (1 − αA)    |
//! | Clear      | 0                               |
//! | Source     | A                               |
//! | Destination| B                               |
//! | DestOver   | A × (1 − αB) + B               |
//! | DestIn     | B × αA                         |
//! | DestOut    | B × (1 − αA)                   |
//! | DestAtop   | A × (1 − αB) + B × αA          |
//!
//! # Examples
//!
//! ```rust
//! use oximedia_image::alpha_composite::{composite, PorterDuffOp};
//!
//! // Two 1×1 RGBA pixels
//! let src = [255u8, 0, 0, 128];   // half-opaque red
//! let dst = [0u8,   0, 255, 255]; // opaque blue
//! let out = composite(&src, &dst, 1, 1, PorterDuffOp::Over, false).unwrap();
//! ```

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{ImageError, ImageResult};

// ── operator enum ─────────────────────────────────────────────────────────────

/// Porter-Duff compositing operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PorterDuffOp {
    /// Clear — output is fully transparent.
    Clear,
    /// Source — output is the source image unchanged.
    Source,
    /// Destination — output is the destination image unchanged.
    Destination,
    /// Source-over (the most common operator): A over B.
    Over,
    /// Source-under: A behind B.
    Under,
    /// Source-in: A masked by B's alpha.
    In,
    /// Source-out: A where B is transparent.
    Out,
    /// Source-atop: A atop B (A where B exists, B elsewhere).
    Atop,
    /// Destination-over: B over A.
    DestOver,
    /// Destination-in: B masked by A's alpha.
    DestIn,
    /// Destination-out: B where A is transparent.
    DestOut,
    /// Destination-atop: B atop A.
    DestAtop,
    /// Xor: A and B where they don't overlap.
    Xor,
}

impl PorterDuffOp {
    /// Human-readable name of this operator.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Clear => "Clear",
            Self::Source => "Source",
            Self::Destination => "Destination",
            Self::Over => "Over",
            Self::Under => "Under",
            Self::In => "In",
            Self::Out => "Out",
            Self::Atop => "Atop",
            Self::DestOver => "DestOver",
            Self::DestIn => "DestIn",
            Self::DestOut => "DestOut",
            Self::DestAtop => "DestAtop",
            Self::Xor => "Xor",
        }
    }
}

// ── per-pixel compositing kernel ──────────────────────────────────────────────

/// A normalised RGBA pixel in [0, 1] using premultiplied alpha internally.
#[derive(Debug, Clone, Copy)]
struct Rgba {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

impl Rgba {
    /// Construct from straight-alpha u8 values.
    fn from_straight(pixel: [u8; 4]) -> Self {
        let a = pixel[3] as f32 / 255.0;
        Self {
            r: (pixel[0] as f32 / 255.0) * a,
            g: (pixel[1] as f32 / 255.0) * a,
            b: (pixel[2] as f32 / 255.0) * a,
            a,
        }
    }

    /// Construct from premultiplied-alpha u8 values.
    fn from_premul(pixel: [u8; 4]) -> Self {
        Self {
            r: pixel[0] as f32 / 255.0,
            g: pixel[1] as f32 / 255.0,
            b: pixel[2] as f32 / 255.0,
            a: pixel[3] as f32 / 255.0,
        }
    }

    /// Convert back to straight-alpha u8 (round-trip compatible).
    fn to_straight_u8(self) -> [u8; 4] {
        if self.a < 1e-6 {
            return [0, 0, 0, 0];
        }
        let inv_a = 1.0 / self.a;
        let r = ((self.r * inv_a).clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = ((self.g * inv_a).clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = ((self.b * inv_a).clamp(0.0, 1.0) * 255.0).round() as u8;
        let a = (self.a.clamp(0.0, 1.0) * 255.0).round() as u8;
        [r, g, b, a]
    }

    /// Convert back to premultiplied u8.
    fn to_premul_u8(self) -> [u8; 4] {
        let r = (self.r.clamp(0.0, 1.0) * 255.0).round() as u8;
        let g = (self.g.clamp(0.0, 1.0) * 255.0).round() as u8;
        let b = (self.b.clamp(0.0, 1.0) * 255.0).round() as u8;
        let a = (self.a.clamp(0.0, 1.0) * 255.0).round() as u8;
        [r, g, b, a]
    }

    /// Multiply all components by a scalar.
    #[inline]
    fn scale(self, f: f32) -> Self {
        Self {
            r: self.r * f,
            g: self.g * f,
            b: self.b * f,
            a: self.a * f,
        }
    }

    /// Component-wise addition.
    #[inline]
    fn add(self, other: Self) -> Self {
        Self {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
            a: self.a + other.a,
        }
    }

    /// The zero (transparent black) pixel.
    #[inline]
    fn zero() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        }
    }
}

/// Apply a Porter-Duff operator to two premultiplied-alpha pixels.
///
/// Both `src` (A) and `dst` (B) should already be in premultiplied form.
fn apply_op(src: Rgba, dst: Rgba, op: PorterDuffOp) -> Rgba {
    let fa = src.a;
    let fb = dst.a;

    match op {
        PorterDuffOp::Clear => Rgba::zero(),
        PorterDuffOp::Source => src,
        PorterDuffOp::Destination => dst,
        PorterDuffOp::Over => src.add(dst.scale(1.0 - fa)),
        PorterDuffOp::Under => src.scale(1.0 - fb).add(dst),
        PorterDuffOp::In => src.scale(fb),
        PorterDuffOp::Out => src.scale(1.0 - fb),
        PorterDuffOp::Atop => src.scale(fb).add(dst.scale(1.0 - fa)),
        PorterDuffOp::DestOver => src.scale(1.0 - fb).add(dst),
        PorterDuffOp::DestIn => dst.scale(fa),
        PorterDuffOp::DestOut => dst.scale(1.0 - fa),
        PorterDuffOp::DestAtop => src.scale(1.0 - fb).add(dst.scale(fa)),
        PorterDuffOp::Xor => src.scale(1.0 - fb).add(dst.scale(1.0 - fa)),
    }
}

// ── public API ────────────────────────────────────────────────────────────────

/// Composite `src` over `dst` using the given Porter-Duff operator.
///
/// Both images must be RGBA (4 channels), same dimensions.  The `premultiplied`
/// flag indicates whether input data is already in premultiplied alpha form;
/// the output uses the same convention.
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if the buffer lengths do not match the
/// given dimensions or if the two images have different sizes.
pub fn composite(
    src: &[u8],
    dst: &[u8],
    width: u32,
    height: u32,
    op: PorterDuffOp,
    premultiplied: bool,
) -> ImageResult<Vec<u8>> {
    let n_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(ImageError::InvalidDimensions(width, height))?;
    let expected = n_pixels * 4;

    if src.len() != expected {
        return Err(ImageError::InvalidFormat(format!(
            "src buffer length {} does not match {}×{}×4 = {}",
            src.len(),
            width,
            height,
            expected
        )));
    }
    if dst.len() != expected {
        return Err(ImageError::InvalidFormat(format!(
            "dst buffer length {} does not match {}×{}×4 = {}",
            dst.len(),
            width,
            height,
            expected
        )));
    }

    let mut out = vec![0u8; expected];

    for i in 0..n_pixels {
        let off = i * 4;
        let sp: [u8; 4] = src[off..off + 4].try_into().unwrap_or([0; 4]);
        let dp: [u8; 4] = dst[off..off + 4].try_into().unwrap_or([0; 4]);

        let s = if premultiplied {
            Rgba::from_premul(sp)
        } else {
            Rgba::from_straight(sp)
        };
        let d = if premultiplied {
            Rgba::from_premul(dp)
        } else {
            Rgba::from_straight(dp)
        };

        let result = apply_op(s, d, op);

        let out_pixel = if premultiplied {
            result.to_premul_u8()
        } else {
            result.to_straight_u8()
        };
        out[off..off + 4].copy_from_slice(&out_pixel);
    }

    Ok(out)
}

/// Composite `src` onto `dst` **in-place** (mutates `dst`).
///
/// Equivalent to [`composite`] but avoids allocating a separate output buffer.
///
/// # Errors
/// Same conditions as [`composite`].
pub fn composite_inplace(
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    height: u32,
    op: PorterDuffOp,
    premultiplied: bool,
) -> ImageResult<()> {
    let n_pixels = (width as usize)
        .checked_mul(height as usize)
        .ok_or(ImageError::InvalidDimensions(width, height))?;
    let expected = n_pixels * 4;

    if src.len() != expected || dst.len() != expected {
        return Err(ImageError::InvalidFormat(
            "buffer length mismatch for in-place composite".to_string(),
        ));
    }

    for i in 0..n_pixels {
        let off = i * 4;
        let sp: [u8; 4] = src[off..off + 4].try_into().unwrap_or([0; 4]);
        let dp: [u8; 4] = dst[off..off + 4].try_into().unwrap_or([0; 4]);

        let s = if premultiplied {
            Rgba::from_premul(sp)
        } else {
            Rgba::from_straight(sp)
        };
        let d = if premultiplied {
            Rgba::from_premul(dp)
        } else {
            Rgba::from_straight(dp)
        };

        let result = apply_op(s, d, op);
        let out_pixel = if premultiplied {
            result.to_premul_u8()
        } else {
            result.to_straight_u8()
        };
        dst[off..off + 4].copy_from_slice(&out_pixel);
    }

    Ok(())
}

/// Convert a straight-alpha RGBA buffer to premultiplied alpha.
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if buffer length is not a multiple of 4.
pub fn premultiply(data: &[u8]) -> ImageResult<Vec<u8>> {
    if data.len() % 4 != 0 {
        return Err(ImageError::InvalidFormat(
            "straight-to-premul: buffer length must be a multiple of 4".to_string(),
        ));
    }
    let mut out = data.to_vec();
    for chunk in out.chunks_exact_mut(4) {
        let a = chunk[3] as f32 / 255.0;
        chunk[0] = ((chunk[0] as f32 * a).round() as u8).min(chunk[3]);
        chunk[1] = ((chunk[1] as f32 * a).round() as u8).min(chunk[3]);
        chunk[2] = ((chunk[2] as f32 * a).round() as u8).min(chunk[3]);
    }
    Ok(out)
}

/// Convert a premultiplied-alpha RGBA buffer back to straight alpha.
///
/// # Errors
/// Returns [`ImageError::InvalidFormat`] if buffer length is not a multiple of 4.
pub fn unpremultiply(data: &[u8]) -> ImageResult<Vec<u8>> {
    if data.len() % 4 != 0 {
        return Err(ImageError::InvalidFormat(
            "premul-to-straight: buffer length must be a multiple of 4".to_string(),
        ));
    }
    let mut out = data.to_vec();
    for chunk in out.chunks_exact_mut(4) {
        let a = chunk[3] as f32 / 255.0;
        if a > 1e-6 {
            let inv_a = 1.0 / a;
            chunk[0] = ((chunk[0] as f32 * inv_a).round().clamp(0.0, 255.0)) as u8;
            chunk[1] = ((chunk[1] as f32 * inv_a).round().clamp(0.0, 255.0)) as u8;
            chunk[2] = ((chunk[2] as f32 * inv_a).round().clamp(0.0, 255.0)) as u8;
        } else {
            chunk[0] = 0;
            chunk[1] = 0;
            chunk[2] = 0;
            chunk[3] = 0;
        }
    }
    Ok(out)
}

/// Gamma-correct compositing: linearise sRGB before blending, then re-apply gamma.
///
/// This avoids the dark-halo artefacts that appear when blending sRGB images
/// linearly (i.e., in the perceptual domain).
///
/// # Errors
/// Same conditions as [`composite`].
pub fn composite_gamma_correct(
    src: &[u8],
    dst: &[u8],
    width: u32,
    height: u32,
    op: PorterDuffOp,
) -> ImageResult<Vec<u8>> {
    // Linearise RGB channels (sRGB → linear), then composite, then re-encode.
    let src_lin = srgb_to_linear(src)?;
    let dst_lin = srgb_to_linear(dst)?;
    let out_lin = composite(&src_lin, &dst_lin, width, height, op, false)?;
    linear_to_srgb(&out_lin)
}

/// sRGB electro-optical transfer function: byte → normalised linear.
#[inline]
fn srgb_decode(v: u8) -> f32 {
    let x = v as f32 / 255.0;
    if x <= 0.04045 {
        x / 12.92
    } else {
        ((x + 0.055) / 1.055).powf(2.4)
    }
}

/// sRGB opto-electronic transfer function: normalised linear → byte.
#[inline]
fn srgb_encode(x: f32) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let encoded = if x <= 0.003_130_8 {
        x * 12.92
    } else {
        1.055 * x.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

/// Convert an RGBA buffer from sRGB to linear light (alpha stays linear).
fn srgb_to_linear(data: &[u8]) -> ImageResult<Vec<u8>> {
    if data.len() % 4 != 0 {
        return Err(ImageError::InvalidFormat(
            "srgb_to_linear: buffer length must be a multiple of 4".to_string(),
        ));
    }
    let mut out = data.to_vec();
    for chunk in out.chunks_exact_mut(4) {
        chunk[0] = (srgb_decode(chunk[0]) * 255.0).round() as u8;
        chunk[1] = (srgb_decode(chunk[1]) * 255.0).round() as u8;
        chunk[2] = (srgb_decode(chunk[2]) * 255.0).round() as u8;
        // alpha unchanged
    }
    Ok(out)
}

/// Convert an RGBA buffer from linear to sRGB (alpha stays linear).
fn linear_to_srgb(data: &[u8]) -> ImageResult<Vec<u8>> {
    if data.len() % 4 != 0 {
        return Err(ImageError::InvalidFormat(
            "linear_to_srgb: buffer length must be a multiple of 4".to_string(),
        ));
    }
    let mut out = data.to_vec();
    for chunk in out.chunks_exact_mut(4) {
        chunk[0] = srgb_encode(chunk[0] as f32 / 255.0);
        chunk[1] = srgb_encode(chunk[1] as f32 / 255.0);
        chunk[2] = srgb_encode(chunk[2] as f32 / 255.0);
    }
    Ok(out)
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn px(r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        vec![r, g, b, a]
    }

    #[test]
    fn test_clear_is_transparent() {
        let src = px(255, 0, 0, 200);
        let dst = px(0, 0, 255, 255);
        let out = composite(&src, &dst, 1, 1, PorterDuffOp::Clear, false).unwrap();
        assert_eq!(out, [0, 0, 0, 0]);
    }

    #[test]
    fn test_source_op() {
        let src = px(255, 0, 0, 200);
        let dst = px(0, 0, 255, 255);
        let out = composite(&src, &dst, 1, 1, PorterDuffOp::Source, false).unwrap();
        assert_eq!(out, src.as_slice());
    }

    #[test]
    fn test_destination_op() {
        let src = px(255, 0, 0, 200);
        let dst = px(0, 0, 255, 255);
        let out = composite(&src, &dst, 1, 1, PorterDuffOp::Destination, false).unwrap();
        assert_eq!(out, dst.as_slice());
    }

    #[test]
    fn test_over_opaque_src_hides_dst() {
        let src = px(255, 0, 0, 255); // fully opaque red
        let dst = px(0, 0, 255, 255); // fully opaque blue
        let out = composite(&src, &dst, 1, 1, PorterDuffOp::Over, false).unwrap();
        // Opaque src completely covers dst.
        assert_eq!(out[0], 255); // red
        assert_eq!(out[1], 0);
        assert_eq!(out[2], 0);
        assert_eq!(out[3], 255);
    }

    #[test]
    fn test_over_transparent_src_shows_dst() {
        let src = px(255, 0, 0, 0); // fully transparent red
        let dst = px(0, 0, 255, 255);
        let out = composite(&src, &dst, 1, 1, PorterDuffOp::Over, false).unwrap();
        assert_eq!(&out, &[0, 0, 255, 255]);
    }

    #[test]
    fn test_in_op_transparent_dst_gives_transparent() {
        let src = px(255, 0, 0, 255);
        let dst = px(0, 0, 0, 0); // fully transparent
        let out = composite(&src, &dst, 1, 1, PorterDuffOp::In, false).unwrap();
        assert_eq!(out[3], 0); // alpha should be 0
    }

    #[test]
    fn test_xor_opaque_over_opaque() {
        // Xor of two fully opaque pixels: both contributions vanish.
        let src = px(255, 0, 0, 255);
        let dst = px(0, 0, 255, 255);
        let out = composite(&src, &dst, 1, 1, PorterDuffOp::Xor, false).unwrap();
        // XOR of two 100% alpha = 0 alpha
        assert_eq!(out[3], 0);
    }

    #[test]
    fn test_premultiply_unpremultiply_roundtrip() {
        let orig = vec![200u8, 100, 50, 200];
        let premul = premultiply(&orig).unwrap();
        let back = unpremultiply(&premul).unwrap();
        // Allow ±1 rounding error per channel.
        for (a, b) in orig.iter().zip(back.iter()) {
            assert!(
                (*a as i16 - *b as i16).abs() <= 2,
                "channel mismatch: orig={a} back={b}"
            );
        }
    }

    #[test]
    fn test_composite_inplace_matches_composite() {
        let src = px(128, 64, 32, 180);
        let dst = px(20, 200, 100, 255);
        let expected = composite(&src, &dst, 1, 1, PorterDuffOp::Over, false).unwrap();
        let mut dst2 = dst.clone();
        composite_inplace(&src, &mut dst2, 1, 1, PorterDuffOp::Over, false).unwrap();
        assert_eq!(expected, dst2);
    }

    #[test]
    fn test_gamma_correct_composite_no_panic() {
        let src = px(255, 0, 0, 128);
        let dst = px(0, 0, 255, 255);
        let out = composite_gamma_correct(&src, &dst, 1, 1, PorterDuffOp::Over);
        assert!(out.is_ok());
        let out = out.unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn test_invalid_buffer_size_returns_error() {
        let src = vec![0u8; 3]; // wrong — not a multiple of 4
        let dst = vec![0u8; 4];
        let result = composite(&src, &dst, 1, 1, PorterDuffOp::Over, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_operator_names() {
        assert_eq!(PorterDuffOp::Over.name(), "Over");
        assert_eq!(PorterDuffOp::Clear.name(), "Clear");
        assert_eq!(PorterDuffOp::Xor.name(), "Xor");
        assert_eq!(PorterDuffOp::DestAtop.name(), "DestAtop");
    }

    #[test]
    fn test_out_op() {
        let src = px(255, 0, 0, 255);
        let dst = px(0, 0, 255, 255); // fully opaque dst → Out = transparent
        let out = composite(&src, &dst, 1, 1, PorterDuffOp::Out, false).unwrap();
        assert_eq!(out[3], 0);
    }

    #[test]
    fn test_multi_pixel_composite() {
        // 2×1 image: two side-by-side pixels
        let src = vec![255u8, 0, 0, 255, 0, 255, 0, 128]; // opaque red | half-green
        let dst = vec![0u8, 0, 255, 255, 0, 0, 255, 255]; // opaque blue | opaque blue
        let out = composite(&src, &dst, 2, 1, PorterDuffOp::Over, false).unwrap();
        assert_eq!(out.len(), 8);
        // Pixel 0: opaque red over blue → red
        assert_eq!(out[0], 255);
        assert_eq!(out[2], 0);
    }
}
