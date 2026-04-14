//! VR180 equirectangular conversion support.
//!
//! VR180 video stores a 180-degree field-of-view half-frame (left eye) in
//! equirectangular projection covering [-90°, +90°] azimuth × [-90°, +90°]
//! elevation.  This module provides [`Vr180Converter`] to map such a half-frame
//! into a full equirectangular panorama (typically 2:1 aspect ratio) with the
//! right half left black.
//!
//! ## Pixel format
//!
//! Input and output pixel buffers are expected to be packed **RGBA** (4 bytes
//! per pixel, row-major).  `convert_nv12_to_rgba` on [`Vr180PixelFormat`] can
//! be used to convert an NV12 camera frame before calling
//! [`Vr180Converter::to_equirectangular`].
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::vr180::Vr180Converter;
//!
//! // A tiny 4×4 RGBA half-frame (all red)
//! let half_frame: Vec<u8> = (0..4 * 4 * 4).map(|i| if i % 4 == 0 { 200 } else if i % 4 == 3 { 255 } else { 0 }).collect();
//! let out = Vr180Converter::to_equirectangular(&half_frame, 4, 4);
//! // Output is 4×4 RGBA too (same dimensions as input, front hemisphere only)
//! assert_eq!(out.len(), 4 * 4 * 4);
//! ```

use crate::VrError;

// ─── Vr180PixelFormat ────────────────────────────────────────────────────────

/// Pixel-format helpers for VR180 camera output.
///
/// Camera sensors commonly output video in NV12 (YUV 4:2:0 semi-planar)
/// format.  This helper converts NV12 to RGBA so that the data can be
/// passed to [`Vr180Converter::to_equirectangular`].
pub struct Vr180PixelFormat;

impl Vr180PixelFormat {
    /// Convert an NV12 (YUV 4:2:0 semi-planar) buffer to packed RGBA.
    ///
    /// NV12 memory layout:
    /// * Y plane: `w × h` bytes (one luma sample per pixel, row-major)
    /// * UV plane (interleaved): `w × h/2` bytes immediately following the Y
    ///   plane (`U` byte, `V` byte, alternating, half-resolution chroma)
    ///
    /// The BT.601 limited-range coefficients are applied for the YCbCr → RGB
    /// conversion.  Alpha is set to 255 (fully opaque) for all pixels.
    ///
    /// # Parameters
    ///
    /// * `nv12` — raw NV12 byte buffer of length ≥ `w * h + w * h / 2`
    /// * `w`    — frame width in pixels (must be even)
    /// * `h`    — frame height in pixels (must be even)
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` of length `w * h * 4` in RGBA order, or an empty `Vec` on
    /// dimension errors.
    #[must_use]
    pub fn convert_nv12_to_rgba(nv12: &[u8], w: u32, h: u32) -> Vec<u8> {
        let ww = w as usize;
        let hh = h as usize;
        let y_size = ww * hh;
        let uv_size = ww * (hh / 2);

        if nv12.len() < y_size + uv_size {
            return Vec::new();
        }

        let y_plane = &nv12[..y_size];
        let uv_plane = &nv12[y_size..y_size + uv_size];

        let mut rgba = vec![0u8; ww * hh * 4];

        for row in 0..hh {
            for col in 0..ww {
                let y_val = y_plane[row * ww + col] as i32;
                // UV indices: each UV pair covers a 2×2 block
                let uv_row = row / 2;
                let uv_col = (col / 2) * 2;
                let uv_idx = uv_row * ww + uv_col;

                let u_val = uv_plane.get(uv_idx).copied().unwrap_or(128) as i32 - 128;
                let v_val = uv_plane.get(uv_idx + 1).copied().unwrap_or(128) as i32 - 128;

                // BT.601 limited range: Y in [16, 235], UV in [16, 240]
                let y_scaled = ((y_val - 16) * 298 + 128) >> 8;

                let r = (y_scaled + ((409 * v_val + 128) >> 8)).clamp(0, 255) as u8;
                let g = (y_scaled - ((100 * u_val + 128) >> 8) - ((208 * v_val + 128) >> 8))
                    .clamp(0, 255) as u8;
                let b = (y_scaled + ((516 * u_val + 128) >> 8)).clamp(0, 255) as u8;

                let dst = (row * ww + col) * 4;
                rgba[dst] = r;
                rgba[dst + 1] = g;
                rgba[dst + 2] = b;
                rgba[dst + 3] = 255;
            }
        }

        rgba
    }
}

// ─── Vr180Converter ───────────────────────────────────────────────────────────

/// Converter from a VR180 half-frame to full equirectangular.
///
/// VR180 covers the front hemisphere (azimuth [-90°, +90°], elevation
/// [-90°, +90°]).  The output equirectangular panorama maps the full sphere
/// (azimuth [-180°, +180°], elevation [-90°, +90°]).  The rear hemisphere
/// (azimuth outside ±90°) is filled with black transparent pixels.
///
/// Coordinate mapping:
/// * The VR180 half-frame uses a rectilinear-like equirectangular layout for
///   the front hemisphere only.
/// * For each output pixel we compute its sphere direction; if the azimuth
///   falls in the front hemisphere we bilinearly sample the half-frame,
///   otherwise we write black.
pub struct Vr180Converter;

impl Vr180Converter {
    /// Convert a VR180 equirectangular half-frame to a full-sphere equirectangular image.
    ///
    /// # Parameters
    ///
    /// * `half_frame` — RGBA pixel data for the front-hemisphere half-frame
    ///   (row-major, 4 bytes per pixel)
    /// * `w`          — width of the half-frame in pixels
    /// * `h`          — height of the half-frame in pixels
    ///
    /// # Returns
    ///
    /// A `Vec<u8>` of length `w × h × 4` representing the full equirectangular
    /// panorama at the same output dimensions.  Rear hemisphere pixels are
    /// black with alpha = 0.
    #[must_use]
    pub fn to_equirectangular(half_frame: &[u8], w: u32, h: u32) -> Vec<u8> {
        let ww = w as usize;
        let hh = h as usize;
        let expected = ww * hh * 4;
        if w == 0 || h == 0 || half_frame.len() < expected {
            return vec![0u8; expected];
        }

        let mut out = vec![0u8; expected];

        for oy in 0..hh {
            for ox in 0..ww {
                // Map output pixel to full-sphere spherical coordinates
                // azimuth ∈ [-π, π], elevation ∈ [-π/2, π/2]
                let az =
                    (ox as f32 / ww as f32) * 2.0 * std::f32::consts::PI - std::f32::consts::PI;
                let el = (1.0 - oy as f32 / hh as f32) * std::f32::consts::PI
                    - std::f32::consts::FRAC_PI_2;

                // VR180 front hemisphere: |azimuth| <= π/2
                let half_pi = std::f32::consts::FRAC_PI_2;
                if az.abs() > half_pi {
                    // Rear hemisphere → leave black/transparent
                    continue;
                }

                // Map front-hemisphere az ∈ [-π/2, π/2] → [0, 1]
                let src_u = (az + half_pi) / std::f32::consts::PI;
                // Map el ∈ [-π/2, π/2] → [1, 0] (top of image = +elevation)
                let src_v = (half_pi - el) / std::f32::consts::PI;

                let sx_f = src_u * ww as f32 - 0.5;
                let sy_f = src_v * hh as f32 - 0.5;

                let sx0 = (sx_f.floor() as i64).clamp(0, ww as i64 - 1) as usize;
                let sy0 = (sy_f.floor() as i64).clamp(0, hh as i64 - 1) as usize;
                let sx1 = (sx0 + 1).min(ww - 1);
                let sy1 = (sy0 + 1).min(hh - 1);
                let tx = sx_f - sx_f.floor();
                let ty = sy_f - sy_f.floor();

                let dst = (oy * ww + ox) * 4;

                for c in 0..4usize {
                    let p00 = half_frame[(sy0 * ww + sx0) * 4 + c] as f32;
                    let p10 = half_frame[(sy0 * ww + sx1) * 4 + c] as f32;
                    let p01 = half_frame[(sy1 * ww + sx0) * 4 + c] as f32;
                    let p11 = half_frame[(sy1 * ww + sx1) * 4 + c] as f32;

                    let top = p00 + (p10 - p00) * tx;
                    let bot = p01 + (p11 - p01) * tx;
                    let val = (top + (bot - top) * ty).round() as u8;
                    out[dst + c] = val;
                }
            }
        }

        out
    }

    /// Validate that a VR180 half-frame buffer has the correct minimum size.
    ///
    /// # Errors
    ///
    /// Returns [`VrError::BufferTooSmall`] if the buffer is too small, or
    /// [`VrError::InvalidDimensions`] if either dimension is zero.
    pub fn validate_half_frame(half_frame: &[u8], w: u32, h: u32) -> Result<(), VrError> {
        if w == 0 || h == 0 {
            return Err(VrError::InvalidDimensions(
                "VR180 half-frame dimensions must be > 0".into(),
            ));
        }
        let expected = w as usize * h as usize * 4;
        if half_frame.len() < expected {
            return Err(VrError::BufferTooSmall {
                expected,
                got: half_frame.len(),
            });
        }
        Ok(())
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_rgba(w: u32, h: u32, r: u8, g: u8, b: u8, a: u8) -> Vec<u8> {
        let n = (w * h * 4) as usize;
        let mut v = Vec::with_capacity(n);
        for _ in 0..(w * h) {
            v.push(r);
            v.push(g);
            v.push(b);
            v.push(a);
        }
        v
    }

    // ── Vr180PixelFormat ─────────────────────────────────────────────────────

    #[test]
    fn nv12_to_rgba_output_size() {
        let w = 4u32;
        let h = 4u32;
        let nv12_size = (w * h + w * h / 2) as usize;
        let nv12 = vec![128u8; nv12_size]; // grey Y, neutral UV
        let rgba = Vr180PixelFormat::convert_nv12_to_rgba(&nv12, w, h);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
    }

    #[test]
    fn nv12_to_rgba_alpha_is_opaque() {
        let w = 2u32;
        let h = 2u32;
        let nv12_size = (w * h + w * h / 2) as usize;
        let nv12 = vec![128u8; nv12_size];
        let rgba = Vr180PixelFormat::convert_nv12_to_rgba(&nv12, w, h);
        for pixel_alpha in rgba.chunks_exact(4).map(|p| p[3]) {
            assert_eq!(pixel_alpha, 255);
        }
    }

    #[test]
    fn nv12_to_rgba_too_small_returns_empty() {
        let rgba = Vr180PixelFormat::convert_nv12_to_rgba(&[0u8; 4], 4, 4);
        assert!(rgba.is_empty());
    }

    // ── Vr180Converter ───────────────────────────────────────────────────────

    #[test]
    fn to_equirectangular_output_size() {
        let half_frame = solid_rgba(8, 8, 200, 0, 0, 255);
        let out = Vr180Converter::to_equirectangular(&half_frame, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 4);
    }

    #[test]
    fn to_equirectangular_zero_dims_returns_zeros() {
        let out = Vr180Converter::to_equirectangular(&[], 0, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn to_equirectangular_centre_pixel_sampled_from_half_frame() {
        // A solid RGBA half-frame should produce the same colour in the centre
        // of the output (which maps to azimuth=0, elevation=0, i.e. front centre).
        let w = 32u32;
        let h = 32u32;
        let half_frame = solid_rgba(w, h, 100, 150, 200, 255);
        let out = Vr180Converter::to_equirectangular(&half_frame, w, h);

        // Centre pixel
        let cx = (w / 2) as usize;
        let cy = (h / 2) as usize;
        let base = (cy * w as usize + cx) * 4;

        // Centre should not be black (it is in the front hemisphere)
        let brightness = out[base] as u32 + out[base + 1] as u32 + out[base + 2] as u32;
        assert!(brightness > 0, "centre pixel should be sampled, not black");
    }

    #[test]
    fn to_equirectangular_rear_hemisphere_is_black() {
        let w = 32u32;
        let h = 32u32;
        let half_frame = solid_rgba(w, h, 200, 200, 200, 255);
        let out = Vr180Converter::to_equirectangular(&half_frame, w, h);

        // The very first column maps to azimuth ≈ -π (rear hemisphere) → should be black
        let base = 0; // first pixel
        assert_eq!(out[base], 0);
        assert_eq!(out[base + 1], 0);
        assert_eq!(out[base + 2], 0);
    }

    #[test]
    fn validate_half_frame_ok() {
        let buf = vec![0u8; 4 * 4 * 4];
        assert!(Vr180Converter::validate_half_frame(&buf, 4, 4).is_ok());
    }

    #[test]
    fn validate_half_frame_too_small() {
        let buf = vec![0u8; 4];
        assert!(Vr180Converter::validate_half_frame(&buf, 4, 4).is_err());
    }

    #[test]
    fn validate_half_frame_zero_dims() {
        assert!(Vr180Converter::validate_half_frame(&[], 0, 4).is_err());
        assert!(Vr180Converter::validate_half_frame(&[], 4, 0).is_err());
    }
}
