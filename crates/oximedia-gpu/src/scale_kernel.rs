//! GPU scaling kernels (CPU simulation via Rayon).
//!
//! Provides parallel image scaling operations that simulate GPU compute
//! shader dispatch. Supports packed RGBA and planar YUV formats.
//!
//! # Supported filters
//!
//! | Filter | Quality | Speed |
//! |--------|---------|-------|
//! | [`ScaleFilter::Nearest`] | Low | Fastest |
//! | [`ScaleFilter::Bilinear`] | Medium | Fast |
//! | [`ScaleFilter::Bicubic`] | High | Medium |
//! | [`ScaleFilter::Area`] | High (downscale) | Medium |
//!
//! # Example
//!
//! ```rust
//! use oximedia_gpu::scale_kernel::{ScaleKernel, ScaleFilter};
//!
//! let src = vec![0u8; 8 * 8 * 4]; // 8×8 RGBA
//! let mut dst = vec![0u8; 4 * 4 * 4]; // scale to 4×4
//! ScaleKernel::scale_rgba(&src, 8, 8, &mut dst, 4, 4, ScaleFilter::Bilinear)
//!     .expect("scale failed");
//! ```

use rayon::prelude::*;
use thiserror::Error;

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors returned by scale kernel operations.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum ScaleError {
    /// Source or destination buffer has incorrect length.
    #[error("Buffer size mismatch: expected {expected}, got {actual}")]
    BufferSizeMismatch { expected: usize, actual: usize },
    /// Image dimensions are zero or invalid.
    #[error("Invalid dimensions: {width}x{height}")]
    InvalidDimensions { width: u32, height: u32 },
    /// Pixel count would overflow.
    #[error("Pixel count overflow")]
    PixelCountOverflow,
    /// Planar planes are inconsistent in length.
    #[error("Planar plane size mismatch: plane '{plane}' has {actual} bytes, expected {expected}")]
    PlanarMismatch {
        plane: &'static str,
        expected: usize,
        actual: usize,
    },
}

// ─── ScaleFilter ──────────────────────────────────────────────────────────────

/// Interpolation filter used when scaling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleFilter {
    /// Nearest-neighbor: fast, blocky.
    Nearest,
    /// Bilinear: smooth, slight blur.
    Bilinear,
    /// Bicubic (Keys, a = -0.5): sharper than bilinear.
    Bicubic,
    /// Area averaging: best for significant downscale (anti-aliasing).
    Area,
}

impl ScaleFilter {
    /// Human-readable label.
    #[must_use]
    pub fn label(self) -> &'static str {
        match self {
            Self::Nearest => "nearest",
            Self::Bilinear => "bilinear",
            Self::Bicubic => "bicubic",
            Self::Area => "area",
        }
    }
}

// ─── ScaleStats ───────────────────────────────────────────────────────────────

/// Statistics produced after a scaling operation.
#[derive(Debug, Clone, Default)]
pub struct ScaleStats {
    /// Number of destination pixels computed.
    pub dst_pixels: u64,
    /// Number of source pixels read (may exceed dst_pixels for area filter).
    pub src_pixels_read: u64,
    /// Filter used.
    pub filter: Option<ScaleFilter>,
}

// ─── ScaleKernel ─────────────────────────────────────────────────────────────

/// GPU-style image scaling kernel (CPU simulation via Rayon).
///
/// Works with packed RGBA (4 bytes / pixel) and planar single-channel data.
#[derive(Debug, Clone, Default)]
pub struct ScaleKernel;

impl ScaleKernel {
    /// Validate a packed RGBA buffer against given dimensions.
    fn validate_rgba(buf: &[u8], width: u32, height: u32) -> Result<usize, ScaleError> {
        if width == 0 || height == 0 {
            return Err(ScaleError::InvalidDimensions { width, height });
        }
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or(ScaleError::PixelCountOverflow)?;
        let expected = pixels * 4;
        if buf.len() != expected {
            return Err(ScaleError::BufferSizeMismatch {
                expected,
                actual: buf.len(),
            });
        }
        Ok(pixels)
    }

    /// Validate a planar (single-channel) buffer.
    fn validate_plane(
        buf: &[u8],
        width: u32,
        height: u32,
        name: &'static str,
    ) -> Result<usize, ScaleError> {
        if width == 0 || height == 0 {
            return Err(ScaleError::InvalidDimensions { width, height });
        }
        let expected = (width as usize)
            .checked_mul(height as usize)
            .ok_or(ScaleError::PixelCountOverflow)?;
        if buf.len() != expected {
            return Err(ScaleError::PlanarMismatch {
                plane: name,
                expected,
                actual: buf.len(),
            });
        }
        Ok(expected)
    }

    // ── Core scaling logic ────────────────────────────────────────────────────

    /// Scale a single-channel (planar) image using the specified filter.
    ///
    /// # Errors
    ///
    /// Returns [`ScaleError`] if any buffer or dimension is invalid.
    pub fn scale_plane(
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        filter: ScaleFilter,
    ) -> Result<ScaleStats, ScaleError> {
        Self::validate_plane(src, src_w, src_h, "src")?;
        let dst_pixels = Self::validate_plane(dst, dst_w, dst_h, "dst")?;

        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;

        dst.par_iter_mut().enumerate().for_each(|(i, out)| {
            let dy = (i / dst_w as usize) as f32;
            let dx = (i % dst_w as usize) as f32;
            *out = match filter {
                ScaleFilter::Nearest => {
                    sample_nearest_plane(src, src_w, src_h, dx, dy, scale_x, scale_y)
                }
                ScaleFilter::Bilinear => {
                    sample_bilinear_plane(src, src_w, src_h, dx, dy, scale_x, scale_y)
                }
                ScaleFilter::Bicubic => {
                    sample_bicubic_plane(src, src_w, src_h, dx, dy, scale_x, scale_y)
                }
                ScaleFilter::Area => sample_area_plane(src, src_w, src_h, dx, dy, scale_x, scale_y),
            };
        });

        Ok(ScaleStats {
            dst_pixels: dst_pixels as u64,
            src_pixels_read: (src_w * src_h) as u64,
            filter: Some(filter),
        })
    }

    /// Scale a packed RGBA image.
    ///
    /// # Errors
    ///
    /// Returns [`ScaleError`] if buffer or dimension validation fails.
    pub fn scale_rgba(
        src: &[u8],
        src_w: u32,
        src_h: u32,
        dst: &mut [u8],
        dst_w: u32,
        dst_h: u32,
        filter: ScaleFilter,
    ) -> Result<ScaleStats, ScaleError> {
        Self::validate_rgba(src, src_w, src_h)?;
        let dst_pixels = Self::validate_rgba(dst, dst_w, dst_h)?;

        let scale_x = src_w as f32 / dst_w as f32;
        let scale_y = src_h as f32 / dst_h as f32;

        dst.par_chunks_mut(4).enumerate().for_each(|(i, out_px)| {
            let dy = (i / dst_w as usize) as f32;
            let dx = (i % dst_w as usize) as f32;

            match filter {
                ScaleFilter::Nearest => {
                    sample_nearest_rgba(src, src_w, src_h, dx, dy, scale_x, scale_y, out_px);
                }
                ScaleFilter::Bilinear => {
                    sample_bilinear_rgba(src, src_w, src_h, dx, dy, scale_x, scale_y, out_px);
                }
                ScaleFilter::Bicubic => {
                    sample_bicubic_rgba(src, src_w, src_h, dx, dy, scale_x, scale_y, out_px);
                }
                ScaleFilter::Area => {
                    sample_area_rgba(src, src_w, src_h, dx, dy, scale_x, scale_y, out_px);
                }
            }
        });

        Ok(ScaleStats {
            dst_pixels: dst_pixels as u64,
            src_pixels_read: (src_w * src_h) as u64,
            filter: Some(filter),
        })
    }

    /// Scale a planar YUV 4:2:0 image.
    ///
    /// Cb/Cr planes are at half resolution in each dimension.
    /// Returns scaled `(Y, Cb, Cr)` planes.
    ///
    /// # Errors
    ///
    /// Returns [`ScaleError`] on dimension or buffer validation failure.
    pub fn scale_yuv420(
        y_src: &[u8],
        cb_src: &[u8],
        cr_src: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        filter: ScaleFilter,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), ScaleError> {
        // Y plane: full resolution
        Self::validate_plane(y_src, src_w, src_h, "Y_src")?;
        // Cb/Cr planes: half resolution
        let chroma_src_w = (src_w + 1) / 2;
        let chroma_src_h = (src_h + 1) / 2;
        Self::validate_plane(cb_src, chroma_src_w, chroma_src_h, "Cb_src")?;
        Self::validate_plane(cr_src, chroma_src_w, chroma_src_h, "Cr_src")?;

        let chroma_dst_w = (dst_w + 1) / 2;
        let chroma_dst_h = (dst_h + 1) / 2;

        // Scale Y
        let mut y_dst = vec![0u8; (dst_w * dst_h) as usize];
        Self::scale_plane(y_src, src_w, src_h, &mut y_dst, dst_w, dst_h, filter)?;

        // Scale Cb
        let mut cb_dst = vec![0u8; (chroma_dst_w * chroma_dst_h) as usize];
        Self::scale_plane(
            cb_src,
            chroma_src_w,
            chroma_src_h,
            &mut cb_dst,
            chroma_dst_w,
            chroma_dst_h,
            filter,
        )?;

        // Scale Cr
        let mut cr_dst = vec![0u8; (chroma_dst_w * chroma_dst_h) as usize];
        Self::scale_plane(
            cr_src,
            chroma_src_w,
            chroma_src_h,
            &mut cr_dst,
            chroma_dst_w,
            chroma_dst_h,
            filter,
        )?;

        Ok((y_dst, cb_dst, cr_dst))
    }

    /// Scale a planar YUV 4:2:2 image.
    ///
    /// Cb/Cr planes are at half width, full height.
    /// Returns scaled `(Y, Cb, Cr)` planes.
    ///
    /// # Errors
    ///
    /// Returns [`ScaleError`] on validation failure.
    pub fn scale_yuv422(
        y_src: &[u8],
        cb_src: &[u8],
        cr_src: &[u8],
        src_w: u32,
        src_h: u32,
        dst_w: u32,
        dst_h: u32,
        filter: ScaleFilter,
    ) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), ScaleError> {
        Self::validate_plane(y_src, src_w, src_h, "Y_src")?;
        let chroma_src_w = (src_w + 1) / 2;
        Self::validate_plane(cb_src, chroma_src_w, src_h, "Cb_src")?;
        Self::validate_plane(cr_src, chroma_src_w, src_h, "Cr_src")?;

        let chroma_dst_w = (dst_w + 1) / 2;

        let mut y_dst = vec![0u8; (dst_w * dst_h) as usize];
        Self::scale_plane(y_src, src_w, src_h, &mut y_dst, dst_w, dst_h, filter)?;

        let mut cb_dst = vec![0u8; (chroma_dst_w * dst_h) as usize];
        Self::scale_plane(
            cb_src,
            chroma_src_w,
            src_h,
            &mut cb_dst,
            chroma_dst_w,
            dst_h,
            filter,
        )?;

        let mut cr_dst = vec![0u8; (chroma_dst_w * dst_h) as usize];
        Self::scale_plane(
            cr_src,
            chroma_src_w,
            src_h,
            &mut cr_dst,
            chroma_dst_w,
            dst_h,
            filter,
        )?;

        Ok((y_dst, cb_dst, cr_dst))
    }
}

// ─── Sampling helpers — planar ────────────────────────────────────────────────

/// Clamp a pixel coordinate to the valid range `[0, max-1]`.
#[inline]
fn clamp_coord(v: i32, max: u32) -> usize {
    v.clamp(0, max as i32 - 1) as usize
}

/// Read one byte from a planar buffer at `(x, y)` (clamped).
#[inline]
fn read_plane(src: &[u8], w: u32, _h: u32, x: i32, y: i32) -> f32 {
    let xi = x.clamp(0, w as i32 - 1) as usize;
    let yi = y.clamp(0, _h as i32 - 1) as usize;
    src[yi * w as usize + xi] as f32
}

/// Nearest-neighbor sample from a plane.
fn sample_nearest_plane(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dx: f32,
    dy: f32,
    scale_x: f32,
    scale_y: f32,
) -> u8 {
    let sx = ((dx + 0.5) * scale_x - 0.5).round() as i32;
    let sy = ((dy + 0.5) * scale_y - 0.5).round() as i32;
    let xi = clamp_coord(sx, src_w);
    let yi = clamp_coord(sy, src_h);
    src[yi * src_w as usize + xi]
}

/// Bilinear sample from a plane.
fn sample_bilinear_plane(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dx: f32,
    dy: f32,
    scale_x: f32,
    scale_y: f32,
) -> u8 {
    let sx = (dx + 0.5) * scale_x - 0.5;
    let sy = (dy + 0.5) * scale_y - 0.5;
    let x0 = sx.floor() as i32;
    let y0 = sy.floor() as i32;
    let wx = sx - x0 as f32;
    let wy = sy - y0 as f32;

    let v00 = read_plane(src, src_w, src_h, x0, y0);
    let v10 = read_plane(src, src_w, src_h, x0 + 1, y0);
    let v01 = read_plane(src, src_w, src_h, x0, y0 + 1);
    let v11 = read_plane(src, src_w, src_h, x0 + 1, y0 + 1);

    let result = v00 * (1.0 - wx) * (1.0 - wy)
        + v10 * wx * (1.0 - wy)
        + v01 * (1.0 - wx) * wy
        + v11 * wx * wy;
    result.round().clamp(0.0, 255.0) as u8
}

/// Keys bicubic weight function (a = -0.5).
#[inline]
fn bicubic_weight(t: f32) -> f32 {
    let t = t.abs();
    let a = -0.5_f32;
    if t <= 1.0 {
        (a + 2.0) * t * t * t - (a + 3.0) * t * t + 1.0
    } else if t < 2.0 {
        a * t * t * t - 5.0 * a * t * t + 8.0 * a * t - 4.0 * a
    } else {
        0.0
    }
}

/// Bicubic sample from a plane using 4×4 tap filter.
fn sample_bicubic_plane(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dx: f32,
    dy: f32,
    scale_x: f32,
    scale_y: f32,
) -> u8 {
    let sx = (dx + 0.5) * scale_x - 0.5;
    let sy = (dy + 0.5) * scale_y - 0.5;
    let x0 = sx.floor() as i32;
    let y0 = sy.floor() as i32;

    let mut sum = 0.0_f32;
    let mut weight_sum = 0.0_f32;
    for ky in -1_i32..=2 {
        let wy = bicubic_weight(sy - (y0 + ky) as f32);
        for kx in -1_i32..=2 {
            let wx = bicubic_weight(sx - (x0 + kx) as f32);
            let w = wx * wy;
            sum += read_plane(src, src_w, src_h, x0 + kx, y0 + ky) * w;
            weight_sum += w;
        }
    }

    if weight_sum.abs() < 1e-9 {
        return 0;
    }
    (sum / weight_sum).round().clamp(0.0, 255.0) as u8
}

/// Area-average sample from a plane (good for downscale).
fn sample_area_plane(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dx: f32,
    dy: f32,
    scale_x: f32,
    scale_y: f32,
) -> u8 {
    // Source region corresponding to this destination pixel.
    let sx0 = dx * scale_x;
    let sy0 = dy * scale_y;
    let sx1 = (dx + 1.0) * scale_x;
    let sy1 = (dy + 1.0) * scale_y;

    let xi0 = sx0.floor() as i32;
    let yi0 = sy0.floor() as i32;
    let xi1 = (sx1.ceil() as i32).min(src_w as i32);
    let yi1 = (sy1.ceil() as i32).min(src_h as i32);

    // For upscale or 1:1, fall back to bilinear.
    if xi1 <= xi0 + 1 && yi1 <= yi0 + 1 {
        return sample_bilinear_plane(src, src_w, src_h, dx, dy, scale_x, scale_y);
    }

    let mut sum = 0.0_f32;
    let mut total_weight = 0.0_f32;
    for sy in yi0..yi1 {
        let wy = partial_coverage(sy as f32, sy0, sy1);
        for sx in xi0..xi1 {
            let wx = partial_coverage(sx as f32, sx0, sx1);
            let w = wx * wy;
            sum += read_plane(src, src_w, src_h, sx, sy) * w;
            total_weight += w;
        }
    }
    if total_weight < 1e-9 {
        return 0;
    }
    (sum / total_weight).round().clamp(0.0, 255.0) as u8
}

/// Fraction of a `[start, end)` interval covered by pixel `[p, p+1)`.
#[inline]
fn partial_coverage(p: f32, start: f32, end: f32) -> f32 {
    let lo = p.max(start);
    let hi = (p + 1.0).min(end);
    (hi - lo).max(0.0)
}

// ─── Sampling helpers — packed RGBA ──────────────────────────────────────────

/// Read one RGBA pixel from a packed buffer (clamped coords).
#[inline]
fn read_rgba(src: &[u8], w: u32, h: u32, x: i32, y: i32) -> [f32; 4] {
    let xi = x.clamp(0, w as i32 - 1) as usize;
    let yi = y.clamp(0, h as i32 - 1) as usize;
    let base = (yi * w as usize + xi) * 4;
    [
        src[base] as f32,
        src[base + 1] as f32,
        src[base + 2] as f32,
        src[base + 3] as f32,
    ]
}

fn sample_nearest_rgba(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dx: f32,
    dy: f32,
    scale_x: f32,
    scale_y: f32,
    out: &mut [u8],
) {
    let sx = ((dx + 0.5) * scale_x - 0.5).round() as i32;
    let sy = ((dy + 0.5) * scale_y - 0.5).round() as i32;
    let px = read_rgba(src, src_w, src_h, sx, sy);
    for (o, &v) in out.iter_mut().zip(px.iter()) {
        *o = v.round().clamp(0.0, 255.0) as u8;
    }
}

fn sample_bilinear_rgba(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dx: f32,
    dy: f32,
    scale_x: f32,
    scale_y: f32,
    out: &mut [u8],
) {
    let sx = (dx + 0.5) * scale_x - 0.5;
    let sy = (dy + 0.5) * scale_y - 0.5;
    let x0 = sx.floor() as i32;
    let y0 = sy.floor() as i32;
    let wx = sx - x0 as f32;
    let wy = sy - y0 as f32;

    let v00 = read_rgba(src, src_w, src_h, x0, y0);
    let v10 = read_rgba(src, src_w, src_h, x0 + 1, y0);
    let v01 = read_rgba(src, src_w, src_h, x0, y0 + 1);
    let v11 = read_rgba(src, src_w, src_h, x0 + 1, y0 + 1);

    for c in 0..4 {
        let r = v00[c] * (1.0 - wx) * (1.0 - wy)
            + v10[c] * wx * (1.0 - wy)
            + v01[c] * (1.0 - wx) * wy
            + v11[c] * wx * wy;
        out[c] = r.round().clamp(0.0, 255.0) as u8;
    }
}

fn sample_bicubic_rgba(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dx: f32,
    dy: f32,
    scale_x: f32,
    scale_y: f32,
    out: &mut [u8],
) {
    let sx = (dx + 0.5) * scale_x - 0.5;
    let sy = (dy + 0.5) * scale_y - 0.5;
    let x0 = sx.floor() as i32;
    let y0 = sy.floor() as i32;

    let mut sum = [0.0_f32; 4];
    let mut weight_sum = 0.0_f32;
    for ky in -1_i32..=2 {
        let wy = bicubic_weight(sy - (y0 + ky) as f32);
        for kx in -1_i32..=2 {
            let wx = bicubic_weight(sx - (x0 + kx) as f32);
            let w = wx * wy;
            let px = read_rgba(src, src_w, src_h, x0 + kx, y0 + ky);
            for c in 0..4 {
                sum[c] += px[c] * w;
            }
            weight_sum += w;
        }
    }
    for c in 0..4 {
        let v = if weight_sum.abs() > 1e-9 {
            sum[c] / weight_sum
        } else {
            0.0
        };
        out[c] = v.round().clamp(0.0, 255.0) as u8;
    }
}

fn sample_area_rgba(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    dx: f32,
    dy: f32,
    scale_x: f32,
    scale_y: f32,
    out: &mut [u8],
) {
    let sx0 = dx * scale_x;
    let sy0 = dy * scale_y;
    let sx1 = (dx + 1.0) * scale_x;
    let sy1 = (dy + 1.0) * scale_y;

    let xi0 = sx0.floor() as i32;
    let yi0 = sy0.floor() as i32;
    let xi1 = (sx1.ceil() as i32).min(src_w as i32);
    let yi1 = (sy1.ceil() as i32).min(src_h as i32);

    if xi1 <= xi0 + 1 && yi1 <= yi0 + 1 {
        sample_bilinear_rgba(src, src_w, src_h, dx, dy, scale_x, scale_y, out);
        return;
    }

    let mut sum = [0.0_f32; 4];
    let mut total_weight = 0.0_f32;
    for sy in yi0..yi1 {
        let wy = partial_coverage(sy as f32, sy0, sy1);
        for sx in xi0..xi1 {
            let wx = partial_coverage(sx as f32, sx0, sx1);
            let w = wx * wy;
            let px = read_rgba(src, src_w, src_h, sx, sy);
            for c in 0..4 {
                sum[c] += px[c] * w;
            }
            total_weight += w;
        }
    }
    for c in 0..4 {
        let v = if total_weight > 1e-9 {
            sum[c] / total_weight
        } else {
            0.0
        };
        out[c] = v.round().clamp(0.0, 255.0) as u8;
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ScaleFilter ───────────────────────────────────────────────────────────

    #[test]
    fn test_scale_filter_labels() {
        assert_eq!(ScaleFilter::Nearest.label(), "nearest");
        assert_eq!(ScaleFilter::Bilinear.label(), "bilinear");
        assert_eq!(ScaleFilter::Bicubic.label(), "bicubic");
        assert_eq!(ScaleFilter::Area.label(), "area");
    }

    // ── Error handling ────────────────────────────────────────────────────────

    #[test]
    fn test_scale_rgba_invalid_src_dims() {
        let src = vec![0u8; 4];
        let mut dst = vec![0u8; 4];
        let err = ScaleKernel::scale_rgba(&src, 0, 1, &mut dst, 1, 1, ScaleFilter::Nearest);
        assert!(matches!(err, Err(ScaleError::InvalidDimensions { .. })));
    }

    #[test]
    fn test_scale_rgba_buffer_mismatch() {
        let src = vec![0u8; 8]; // wrong: 1×1 = 4 bytes
        let mut dst = vec![0u8; 4];
        let err = ScaleKernel::scale_rgba(&src, 1, 1, &mut dst, 1, 1, ScaleFilter::Bilinear);
        assert!(matches!(err, Err(ScaleError::BufferSizeMismatch { .. })));
    }

    #[test]
    fn test_scale_plane_invalid_dims() {
        let src = vec![0u8; 1];
        let mut dst = vec![0u8; 4];
        let err = ScaleKernel::scale_plane(&src, 0, 0, &mut dst, 2, 2, ScaleFilter::Nearest);
        assert!(matches!(err, Err(ScaleError::InvalidDimensions { .. })));
    }

    // ── 1:1 identity scale ────────────────────────────────────────────────────

    fn identity_scale(filter: ScaleFilter) {
        // A simple gradient 4×4 RGBA image.
        let src: Vec<u8> = (0..4 * 4 * 4).map(|i| (i * 7 % 256) as u8).collect();
        let mut dst = vec![0u8; 4 * 4 * 4];
        ScaleKernel::scale_rgba(&src, 4, 4, &mut dst, 4, 4, filter).unwrap();
        // Identity scale must reproduce the source exactly (or with 1 LSB for filtered).
        for (i, (&s, &d)) in src.iter().zip(dst.iter()).enumerate() {
            let diff = (s as i16 - d as i16).abs();
            assert!(
                diff <= 1,
                "filter={}: pixel {i}: src={s} dst={d}",
                filter.label()
            );
        }
    }

    #[test]
    fn test_identity_nearest() {
        identity_scale(ScaleFilter::Nearest);
    }

    #[test]
    fn test_identity_bilinear() {
        identity_scale(ScaleFilter::Bilinear);
    }

    #[test]
    fn test_identity_bicubic() {
        identity_scale(ScaleFilter::Bicubic);
    }

    // ── Downscale ─────────────────────────────────────────────────────────────

    #[test]
    fn test_downscale_2x_preserves_constant_color() {
        // Solid red 8×8 RGBA
        let src: Vec<u8> = (0..8 * 8).flat_map(|_| [255u8, 0, 0, 255]).collect();
        let mut dst = vec![0u8; 4 * 4 * 4];
        ScaleKernel::scale_rgba(&src, 8, 8, &mut dst, 4, 4, ScaleFilter::Bilinear).unwrap();
        for px in dst.chunks(4) {
            assert_eq!(px[0], 255, "R should be 255");
            assert_eq!(px[1], 0, "G should be 0");
            assert_eq!(px[2], 0, "B should be 0");
            assert_eq!(px[3], 255, "A should be 255");
        }
    }

    #[test]
    fn test_downscale_output_size() {
        let src = vec![128u8; 64 * 64 * 4];
        let mut dst = vec![0u8; 32 * 32 * 4];
        let stats =
            ScaleKernel::scale_rgba(&src, 64, 64, &mut dst, 32, 32, ScaleFilter::Area).unwrap();
        assert_eq!(stats.dst_pixels, 32 * 32);
    }

    // ── Upscale ───────────────────────────────────────────────────────────────

    #[test]
    fn test_upscale_output_size() {
        let src = vec![64u8; 8 * 8 * 4];
        let mut dst = vec![0u8; 16 * 16 * 4];
        let stats =
            ScaleKernel::scale_rgba(&src, 8, 8, &mut dst, 16, 16, ScaleFilter::Bicubic).unwrap();
        assert_eq!(stats.dst_pixels, 16 * 16);
    }

    // ── Plane scaling ─────────────────────────────────────────────────────────

    #[test]
    fn test_scale_plane_constant_value() {
        let src = vec![200u8; 16 * 16];
        let mut dst = vec![0u8; 8 * 8];
        ScaleKernel::scale_plane(&src, 16, 16, &mut dst, 8, 8, ScaleFilter::Bilinear).unwrap();
        assert!(
            dst.iter().all(|&v| v == 200),
            "constant plane should stay constant"
        );
    }

    // ── YUV 4:2:0 scaling ─────────────────────────────────────────────────────

    #[test]
    fn test_scale_yuv420_output_sizes() {
        let y_src = vec![128u8; 8 * 8];
        let cb_src = vec![128u8; 4 * 4];
        let cr_src = vec![128u8; 4 * 4];
        let (y_dst, cb_dst, cr_dst) =
            ScaleKernel::scale_yuv420(&y_src, &cb_src, &cr_src, 8, 8, 4, 4, ScaleFilter::Bilinear)
                .unwrap();
        assert_eq!(y_dst.len(), 4 * 4);
        assert_eq!(cb_dst.len(), 2 * 2);
        assert_eq!(cr_dst.len(), 2 * 2);
    }

    #[test]
    fn test_scale_yuv420_constant_neutral() {
        let y_src = vec![128u8; 8 * 8];
        let cb_src = vec![128u8; 4 * 4];
        let cr_src = vec![128u8; 4 * 4];
        let (y_dst, cb_dst, cr_dst) =
            ScaleKernel::scale_yuv420(&y_src, &cb_src, &cr_src, 8, 8, 16, 16, ScaleFilter::Nearest)
                .unwrap();
        assert_eq!(y_dst.len(), 16 * 16);
        assert!(y_dst.iter().all(|&v| v == 128));
        assert!(cb_dst.iter().all(|&v| v == 128));
        assert!(cr_dst.iter().all(|&v| v == 128));
    }

    // ── YUV 4:2:2 scaling ─────────────────────────────────────────────────────

    #[test]
    fn test_scale_yuv422_output_sizes() {
        let y_src = vec![128u8; 8 * 4]; // 8×4
        let cb_src = vec![128u8; 4 * 4]; // 4×4
        let cr_src = vec![128u8; 4 * 4]; // 4×4
        let (y_dst, cb_dst, cr_dst) =
            ScaleKernel::scale_yuv422(&y_src, &cb_src, &cr_src, 8, 4, 4, 2, ScaleFilter::Bilinear)
                .unwrap();
        assert_eq!(y_dst.len(), 4 * 2);
        assert_eq!(cb_dst.len(), 2 * 2);
        assert_eq!(cr_dst.len(), 2 * 2);
    }

    // ── bicubic_weight unit test ──────────────────────────────────────────────

    #[test]
    fn test_bicubic_weight_at_zero_is_one() {
        assert!((bicubic_weight(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_bicubic_weight_at_two_is_zero() {
        assert!(bicubic_weight(2.0).abs() < 1e-6);
    }

    // ── stats ─────────────────────────────────────────────────────────────────

    #[test]
    fn test_scale_stats_filter_recorded() {
        let src = vec![0u8; 4 * 4 * 4];
        let mut dst = vec![0u8; 2 * 2 * 4];
        let stats = ScaleKernel::scale_rgba(&src, 4, 4, &mut dst, 2, 2, ScaleFilter::Area).unwrap();
        assert_eq!(stats.filter, Some(ScaleFilter::Area));
        assert_eq!(stats.dst_pixels, 4);
    }
}
