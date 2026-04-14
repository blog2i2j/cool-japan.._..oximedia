//! BT.709 RGB ↔ YUV colour conversion using scalar SIMD-friendly code.
//!
//! Provides [`Rgb2YuvCoeffs`] with the standard BT.709 matrix and two
//! conversion functions: [`rgb_to_yuv_bt709`] and [`yuv_to_rgb_bt709`].
//!
//! Values are full-range: RGB components and Y are in `[0, 255]`, while Cb/Cr
//! are centred on 128 (i.e. in `[0, 255]` with 128 representing zero chroma).

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

/// A packed RGB pixel (full-range, 8-bit per channel).
///
/// # Examples
///
/// ```
/// use oximedia_simd::color_convert_simd::RgbPixel;
/// let p = RgbPixel { r: 255, g: 0, b: 0 };
/// assert_eq!(p.r, 255);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RgbPixel {
    /// Red channel, `[0, 255]`.
    pub r: u8,
    /// Green channel, `[0, 255]`.
    pub g: u8,
    /// Blue channel, `[0, 255]`.
    pub b: u8,
}

/// A packed YCbCr pixel (full-range, 8-bit per channel, BT.709 convention).
///
/// Cb and Cr are offset so that zero chroma maps to 128.
///
/// # Examples
///
/// ```
/// use oximedia_simd::color_convert_simd::YuvPixel;
/// let p = YuvPixel { y: 128, cb: 128, cr: 128 };
/// assert_eq!(p.cb, 128);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct YuvPixel {
    /// Luma, `[0, 255]`.
    pub y: u8,
    /// Cb (blue-difference chroma), `[0, 255]`, 128 = neutral.
    pub cb: u8,
    /// Cr (red-difference chroma), `[0, 255]`, 128 = neutral.
    pub cr: u8,
}

/// BT.709 RGB-to-YCbCr conversion coefficients.
///
/// Stores the 3×3 floating-point matrix in row-major order (one row per output
/// channel).  The `offset` array holds the additive bias applied *after*
/// multiplication (e.g. 128 for Cb/Cr to centre them around zero).
///
/// # Examples
///
/// ```
/// use oximedia_simd::color_convert_simd::Rgb2YuvCoeffs;
/// let c = Rgb2YuvCoeffs::bt709();
/// // Y = 0.2126 R + 0.7152 G + 0.0722 B
/// assert!((c.row_y[0] - 0.2126_f32).abs() < 1e-4);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Rgb2YuvCoeffs {
    /// Coefficients for the Y (luma) row: `[kr, kg, kb]`.
    pub row_y: [f32; 3],
    /// Coefficients for the Cb (blue-difference) row.
    pub row_cb: [f32; 3],
    /// Coefficients for the Cr (red-difference) row.
    pub row_cr: [f32; 3],
    /// Additive offsets `[y_off, cb_off, cr_off]` applied after multiplication.
    pub offset: [f32; 3],
}

impl Rgb2YuvCoeffs {
    /// Returns the standard ITU-R BT.709 coefficients (full-range).
    ///
    /// Kr = 0.2126, Kb = 0.0722.
    #[must_use]
    pub fn bt709() -> Self {
        // Standard BT.709 primaries
        let kr: f32 = 0.2126;
        let kg: f32 = 0.7152;
        let kb: f32 = 0.0722;

        Self {
            row_y: [kr, kg, kb],
            row_cb: [-kr / (2.0 * (1.0 - kb)), -kg / (2.0 * (1.0 - kb)), 0.5],
            row_cr: [0.5, -kg / (2.0 * (1.0 - kr)), -kb / (2.0 * (1.0 - kr))],
            offset: [0.0, 128.0, 128.0],
        }
    }

    /// Returns ITU-R BT.2020 / BT.2100 coefficients (full-range, 8-bit encoding).
    ///
    /// BT.2020 primaries: Kr = 0.2627, Kg = 0.6780, Kb = 0.0593.
    /// These are also used for BT.2100 HLG and PQ transfers (the matrix is the
    /// same; only the OETF/EOTF differs).
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_simd::color_convert_simd::Rgb2YuvCoeffs;
    /// let c = Rgb2YuvCoeffs::bt2020();
    /// assert!((c.row_y[0] - 0.2627_f32).abs() < 1e-4);
    /// ```
    #[must_use]
    pub fn bt2020() -> Self {
        let kr: f32 = 0.2627;
        let kg: f32 = 0.6780;
        let kb: f32 = 0.0593;

        Self {
            row_y: [kr, kg, kb],
            row_cb: [-kr / (2.0 * (1.0 - kb)), -kg / (2.0 * (1.0 - kb)), 0.5],
            row_cr: [0.5, -kg / (2.0 * (1.0 - kr)), -kb / (2.0 * (1.0 - kr))],
            offset: [0.0, 128.0, 128.0],
        }
    }

    /// Returns ITU-R BT.2100 ICtCp coefficients (full-range, 8-bit encoding).
    ///
    /// ICtCp is the perceptual colour space used with BT.2100 HDR content.  It
    /// uses a different luma/chroma decomposition from the standard YCbCr matrix.
    /// The luma coefficients follow the same BT.2020 primaries; the Ct and Cp
    /// channels are a rotation of the YCbCr Cb/Cr axes to better decorrelate
    /// colour.  This implementation uses the simplified fixed-point-friendly
    /// approximation (`ICtCp ≈ rotated BT.2020 YCbCr`) and is suitable for
    /// 8-bit SDR proxy workflows.  Full ICtCp for HDR requires 12-bit or
    /// floating-point precision.
    ///
    /// For production HDR encoding at full precision, use the `oximedia-hdr`
    /// crate which implements the full SMPTE ST 2084/BT.2100 pipeline.
    #[must_use]
    pub fn bt2100_ictcp() -> Self {
        // ICtCp rotation approximation (Dolby/ITU simplified 8-bit version)
        // Rotation by ~45° in Cb-Cr space relative to BT.2020 YCbCr:
        //   Ct ≈  0.5 * Cb + 0.5 * Cr
        //   Cp ≈ -0.5 * Cb + 0.5 * Cr
        // Luma row is identical to BT.2020.
        let kr: f32 = 0.2627;
        let kg: f32 = 0.6780;
        let kb: f32 = 0.0593;

        // Standard BT.2020 Cb row
        let cb_r = -kr / (2.0 * (1.0 - kb));
        let cb_g = -kg / (2.0 * (1.0 - kb));
        let cb_b = 0.5_f32;

        // Standard BT.2020 Cr row
        let cr_r = 0.5_f32;
        let cr_g = -kg / (2.0 * (1.0 - kr));
        let cr_b = -kb / (2.0 * (1.0 - kr));

        // Ct = 0.5*(Cb + Cr),  Cp = 0.5*(-Cb + Cr)
        let ct_r = 0.5 * (cb_r + cr_r);
        let ct_g = 0.5 * (cb_g + cr_g);
        let ct_b = 0.5 * (cb_b + cr_b);
        let cp_r = 0.5 * (-cb_r + cr_r);
        let cp_g = 0.5 * (-cb_g + cr_g);
        let cp_b = 0.5 * (-cb_b + cr_b);

        Self {
            row_y: [kr, kg, kb],
            row_cb: [ct_r, ct_g, ct_b], // Ct channel
            row_cr: [cp_r, cp_g, cp_b], // Cp channel
            offset: [0.0, 128.0, 128.0],
        }
    }
}

/// Converts a slice of RGB pixels to YCbCr using the BT.709 matrix.
///
/// Both slices must have the same length; otherwise the conversion stops at
/// `min(rgb.len(), yuv.len())`.
///
/// # Examples
///
/// ```
/// use oximedia_simd::color_convert_simd::{RgbPixel, YuvPixel, rgb_to_yuv_bt709};
/// let src = [RgbPixel { r: 255, g: 255, b: 255 }];
/// let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
/// rgb_to_yuv_bt709(&src, &mut dst);
/// // White should give Y ≈ 255, Cb ≈ 128, Cr ≈ 128
/// assert!(dst[0].y > 250);
/// assert!((dst[0].cb as i32 - 128).abs() <= 2);
/// ```
pub fn rgb_to_yuv_bt709(rgb: &[RgbPixel], yuv: &mut [YuvPixel]) {
    let coeffs = Rgb2YuvCoeffs::bt709();
    let n = rgb.len().min(yuv.len());
    for i in 0..n {
        let r = f32::from(rgb[i].r);
        let g = f32::from(rgb[i].g);
        let b = f32::from(rgb[i].b);

        let y = coeffs.row_y[0] * r + coeffs.row_y[1] * g + coeffs.row_y[2] * b + coeffs.offset[0];
        let cb =
            coeffs.row_cb[0] * r + coeffs.row_cb[1] * g + coeffs.row_cb[2] * b + coeffs.offset[1];
        let cr =
            coeffs.row_cr[0] * r + coeffs.row_cr[1] * g + coeffs.row_cr[2] * b + coeffs.offset[2];

        yuv[i] = YuvPixel {
            y: y.clamp(0.0, 255.0) as u8,
            cb: cb.clamp(0.0, 255.0) as u8,
            cr: cr.clamp(0.0, 255.0) as u8,
        };
    }
}

/// Converts a slice of YCbCr pixels back to RGB using the inverse BT.709 matrix.
///
/// Both slices must have the same length; conversion stops at
/// `min(yuv.len(), rgb.len())`.
///
/// # Examples
///
/// ```
/// use oximedia_simd::color_convert_simd::{RgbPixel, YuvPixel, yuv_to_rgb_bt709};
/// let src = [YuvPixel { y: 235, cb: 128, cr: 128 }];
/// let mut dst = [RgbPixel { r: 0, g: 0, b: 0 }];
/// yuv_to_rgb_bt709(&src, &mut dst);
/// // Near-white with neutral chroma → R ≈ G ≈ B
/// assert!((dst[0].r as i32 - dst[0].g as i32).abs() <= 3);
/// ```
pub fn yuv_to_rgb_bt709(yuv: &[YuvPixel], rgb: &mut [RgbPixel]) {
    let n = yuv.len().min(rgb.len());
    // BT.709 inverse matrix (full-range, Cb/Cr centred on 128)
    for i in 0..n {
        let y = f32::from(yuv[i].y);
        let cb = f32::from(yuv[i].cb) - 128.0;
        let cr = f32::from(yuv[i].cr) - 128.0;

        // Standard BT.709 inverse
        let r = y + 1.574_72 * cr;
        let g = y - 0.187_324 * cb - 0.468_124 * cr;
        let b = y + 1.855_63 * cb;

        rgb[i] = RgbPixel {
            r: r.clamp(0.0, 255.0) as u8,
            g: g.clamp(0.0, 255.0) as u8,
            b: b.clamp(0.0, 255.0) as u8,
        };
    }
}

// ── BT.2020 (also used for BT.2100 HLG/PQ luma matrix) ─────────────────────

/// Convert RGB pixels to YCbCr using the ITU-R BT.2020 matrix (full-range).
///
/// BT.2020 is the standard colour space for Ultra-HD / HDR content.
/// The transfer function (HLG or PQ) is *not* applied here — callers should
/// apply the OETF before calling this function if needed.
///
/// # Examples
///
/// ```
/// use oximedia_simd::color_convert_simd::{RgbPixel, YuvPixel, rgb_to_yuv_bt2020};
/// let src = [RgbPixel { r: 255, g: 255, b: 255 }];
/// let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
/// rgb_to_yuv_bt2020(&src, &mut dst);
/// assert!(dst[0].y > 250);
/// ```
pub fn rgb_to_yuv_bt2020(rgb: &[RgbPixel], yuv: &mut [YuvPixel]) {
    let coeffs = Rgb2YuvCoeffs::bt2020();
    let n = rgb.len().min(yuv.len());
    for i in 0..n {
        let r = f32::from(rgb[i].r);
        let g = f32::from(rgb[i].g);
        let b = f32::from(rgb[i].b);

        let y = coeffs.row_y[0] * r + coeffs.row_y[1] * g + coeffs.row_y[2] * b + coeffs.offset[0];
        let cb =
            coeffs.row_cb[0] * r + coeffs.row_cb[1] * g + coeffs.row_cb[2] * b + coeffs.offset[1];
        let cr =
            coeffs.row_cr[0] * r + coeffs.row_cr[1] * g + coeffs.row_cr[2] * b + coeffs.offset[2];

        yuv[i] = YuvPixel {
            y: y.clamp(0.0, 255.0) as u8,
            cb: cb.clamp(0.0, 255.0) as u8,
            cr: cr.clamp(0.0, 255.0) as u8,
        };
    }
}

/// Convert BT.2020 YCbCr pixels back to RGB (full-range inverse matrix).
///
/// Inverse BT.2020 matrix coefficients (Kr=0.2627, Kb=0.0593):
/// - R = Y + 1.4746 * Cr
/// - G = Y - 0.1646 * Cb - 0.5714 * Cr
/// - B = Y + 1.8814 * Cb
///
/// # Examples
///
/// ```
/// use oximedia_simd::color_convert_simd::{RgbPixel, YuvPixel, yuv_to_rgb_bt2020};
/// let src = [YuvPixel { y: 128, cb: 128, cr: 128 }];
/// let mut dst = [RgbPixel { r: 0, g: 0, b: 0 }];
/// yuv_to_rgb_bt2020(&src, &mut dst);
/// // Neutral chroma → R ≈ G ≈ B
/// assert!((dst[0].r as i32 - dst[0].b as i32).abs() <= 5);
/// ```
pub fn yuv_to_rgb_bt2020(yuv: &[YuvPixel], rgb: &mut [RgbPixel]) {
    let n = yuv.len().min(rgb.len());
    for i in 0..n {
        let y = f32::from(yuv[i].y);
        let cb = f32::from(yuv[i].cb) - 128.0;
        let cr = f32::from(yuv[i].cr) - 128.0;

        // BT.2020 inverse (Kr=0.2627, Kb=0.0593)
        let r = y + 1.474_6 * cr;
        let g = y - 0.164_55 * cb - 0.571_35 * cr;
        let b = y + 1.881_4 * cb;

        rgb[i] = RgbPixel {
            r: r.clamp(0.0, 255.0) as u8,
            g: g.clamp(0.0, 255.0) as u8,
            b: b.clamp(0.0, 255.0) as u8,
        };
    }
}

// ── BT.2100 ICtCp ────────────────────────────────────────────────────────────

/// Convert RGB pixels to ICtCp (BT.2100 approximate, 8-bit full-range).
///
/// This is a low-complexity approximation suitable for 8-bit proxy workflows.
/// For full-precision HDR ICtCp processing use the `oximedia-hdr` crate.
///
/// # Examples
///
/// ```
/// use oximedia_simd::color_convert_simd::{RgbPixel, YuvPixel, rgb_to_ictcp_bt2100};
/// let src = [RgbPixel { r: 128, g: 128, b: 128 }];
/// let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
/// rgb_to_ictcp_bt2100(&src, &mut dst);
/// // Neutral grey → Ct ≈ 128, Cp ≈ 128
/// assert!((dst[0].cb as i32 - 128).abs() <= 5);
/// ```
pub fn rgb_to_ictcp_bt2100(rgb: &[RgbPixel], ictcp: &mut [YuvPixel]) {
    let coeffs = Rgb2YuvCoeffs::bt2100_ictcp();
    let n = rgb.len().min(ictcp.len());
    for i in 0..n {
        let r = f32::from(rgb[i].r);
        let g = f32::from(rgb[i].g);
        let b = f32::from(rgb[i].b);

        let intensity =
            coeffs.row_y[0] * r + coeffs.row_y[1] * g + coeffs.row_y[2] * b + coeffs.offset[0];
        let ct =
            coeffs.row_cb[0] * r + coeffs.row_cb[1] * g + coeffs.row_cb[2] * b + coeffs.offset[1];
        let cp =
            coeffs.row_cr[0] * r + coeffs.row_cr[1] * g + coeffs.row_cr[2] * b + coeffs.offset[2];

        ictcp[i] = YuvPixel {
            y: intensity.clamp(0.0, 255.0) as u8,
            cb: ct.clamp(0.0, 255.0) as u8,
            cr: cp.clamp(0.0, 255.0) as u8,
        };
    }

    // Apply `is_x86_feature_detected!`-gated AVX2 path for future optimisation.
    // The scalar path is already SIMDified by the compiler with -C opt-level=3.
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            // Currently uses the same scalar computation; a hand-vectorised
            // AVX2 path (8 pixels × _mm256_fmadd_ps) can be added here.
        }
    }
}

/// Convert ICtCp pixels back to RGB (BT.2100 approximate inverse).
///
/// # Examples
///
/// ```
/// use oximedia_simd::color_convert_simd::{RgbPixel, YuvPixel, ictcp_to_rgb_bt2100};
/// let src = [YuvPixel { y: 128, cb: 128, cr: 128 }];
/// let mut dst = [RgbPixel { r: 0, g: 0, b: 0 }];
/// ictcp_to_rgb_bt2100(&src, &mut dst);
/// // Near-neutral → R ≈ G ≈ B
/// assert!((dst[0].r as i32 - dst[0].g as i32).abs() <= 5);
/// ```
pub fn ictcp_to_rgb_bt2100(ictcp: &[YuvPixel], rgb: &mut [RgbPixel]) {
    // ICtCp inverse: first undo the rotation to get BT.2020 Cb/Cr, then apply
    // the BT.2020 inverse matrix.
    //  Cb = Ct - Cp   (un-rotate)
    //  Cr = Ct + Cp
    let n = ictcp.len().min(rgb.len());
    for i in 0..n {
        let intensity = f32::from(ictcp[i].y);
        let ct = f32::from(ictcp[i].cb) - 128.0;
        let cp = f32::from(ictcp[i].cr) - 128.0;

        // Undo rotation: Ct = 0.5*(Cb+Cr), Cp = 0.5*(-Cb+Cr)
        //  → Cb = Ct - Cp,  Cr = Ct + Cp
        let cb = ct - cp;
        let cr = ct + cp;

        // BT.2020 inverse matrix
        let r = intensity + 1.474_6 * cr;
        let g = intensity - 0.164_55 * cb - 0.571_35 * cr;
        let b = intensity + 1.881_4 * cb;

        rgb[i] = RgbPixel {
            r: r.clamp(0.0, 255.0) as u8,
            g: g.clamp(0.0, 255.0) as u8,
            b: b.clamp(0.0, 255.0) as u8,
        };
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(r: u8, g: u8, b: u8) -> (i32, i32, i32) {
        let src = [RgbPixel { r, g, b }];
        let mut yuv = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        let mut dst = [RgbPixel { r: 0, g: 0, b: 0 }];
        rgb_to_yuv_bt709(&src, &mut yuv);
        yuv_to_rgb_bt709(&yuv, &mut dst);
        (
            i32::from(r) - i32::from(dst[0].r),
            i32::from(g) - i32::from(dst[0].g),
            i32::from(b) - i32::from(dst[0].b),
        )
    }

    #[test]
    fn bt709_coeffs_luma_sum_to_one() {
        let c = Rgb2YuvCoeffs::bt709();
        let sum = c.row_y[0] + c.row_y[1] + c.row_y[2];
        assert!((sum - 1.0).abs() < 1e-4, "Y coefficients sum = {sum}");
    }

    #[test]
    fn white_converts_to_max_luma() {
        let src = [RgbPixel {
            r: 255,
            g: 255,
            b: 255,
        }];
        let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        rgb_to_yuv_bt709(&src, &mut dst);
        assert!(dst[0].y > 250, "Y for white = {}", dst[0].y);
    }

    #[test]
    fn black_converts_to_zero_luma() {
        let src = [RgbPixel { r: 0, g: 0, b: 0 }];
        let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        rgb_to_yuv_bt709(&src, &mut dst);
        assert_eq!(dst[0].y, 0);
    }

    #[test]
    fn white_has_neutral_chroma() {
        let src = [RgbPixel {
            r: 255,
            g: 255,
            b: 255,
        }];
        let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        rgb_to_yuv_bt709(&src, &mut dst);
        assert!((i32::from(dst[0].cb) - 128).abs() <= 2);
        assert!((i32::from(dst[0].cr) - 128).abs() <= 2);
    }

    #[test]
    fn round_trip_white() {
        let (dr, dg, db) = round_trip(255, 255, 255);
        assert!(dr.abs() <= 2 && dg.abs() <= 2 && db.abs() <= 2);
    }

    #[test]
    fn round_trip_black() {
        let (dr, dg, db) = round_trip(0, 0, 0);
        assert!(dr.abs() <= 2 && dg.abs() <= 2 && db.abs() <= 2);
    }

    #[test]
    fn round_trip_mid_grey() {
        let (dr, dg, db) = round_trip(128, 128, 128);
        assert!(dr.abs() <= 3 && dg.abs() <= 3 && db.abs() <= 3);
    }

    #[test]
    fn round_trip_red() {
        let (dr, _dg, _db) = round_trip(200, 0, 0);
        assert!(dr.abs() <= 3);
    }

    #[test]
    fn slice_length_mismatch_uses_min() {
        let src = [
            RgbPixel {
                r: 100,
                g: 100,
                b: 100,
            },
            RgbPixel {
                r: 200,
                g: 200,
                b: 200,
            },
        ];
        let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        // Only 1 output slot — must not panic
        rgb_to_yuv_bt709(&src, &mut dst);
        assert!(dst[0].y > 0);
    }

    #[test]
    fn yuv_to_rgb_neutral_chroma_gives_grey() {
        let src = [YuvPixel {
            y: 128,
            cb: 128,
            cr: 128,
        }];
        let mut dst = [RgbPixel { r: 0, g: 0, b: 0 }];
        yuv_to_rgb_bt709(&src, &mut dst);
        let r = i32::from(dst[0].r);
        let g = i32::from(dst[0].g);
        let b = i32::from(dst[0].b);
        // All channels should be close to each other
        assert!((r - g).abs() <= 3, "r={r} g={g}");
        assert!((r - b).abs() <= 3, "r={r} b={b}");
    }

    #[test]
    fn empty_slices_do_not_panic() {
        rgb_to_yuv_bt709(&[], &mut []);
        yuv_to_rgb_bt709(&[], &mut []);
    }

    #[test]
    fn pixel_structs_are_copy() {
        let rgb = RgbPixel { r: 1, g: 2, b: 3 };
        let _rgb2 = rgb; // copy
        let yuv = YuvPixel {
            y: 10,
            cb: 128,
            cr: 128,
        };
        let _yuv2 = yuv; // copy
    }

    #[test]
    fn rgb_to_yuv_multiple_pixels() {
        let src = vec![
            RgbPixel { r: 0, g: 0, b: 0 },
            RgbPixel {
                r: 128,
                g: 128,
                b: 128,
            },
            RgbPixel {
                r: 255,
                g: 255,
                b: 255,
            },
        ];
        let mut dst = vec![YuvPixel { y: 0, cb: 0, cr: 0 }; 3];
        rgb_to_yuv_bt709(&src, &mut dst);
        assert!(dst[0].y < dst[1].y);
        assert!(dst[1].y < dst[2].y);
    }

    // ── BT.2020 tests ─────────────────────────────────────────────────────────

    #[test]
    fn bt2020_coeffs_luma_sum_to_one() {
        let c = Rgb2YuvCoeffs::bt2020();
        let sum = c.row_y[0] + c.row_y[1] + c.row_y[2];
        assert!((sum - 1.0).abs() < 1e-4, "BT.2020 Y coefficients sum={sum}");
    }

    #[test]
    fn bt2020_kr_kg_kb_correct() {
        let c = Rgb2YuvCoeffs::bt2020();
        assert!(
            (c.row_y[0] - 0.2627_f32).abs() < 1e-4,
            "BT.2020 Kr={}",
            c.row_y[0]
        );
        assert!(
            (c.row_y[1] - 0.6780_f32).abs() < 1e-4,
            "BT.2020 Kg={}",
            c.row_y[1]
        );
        assert!(
            (c.row_y[2] - 0.0593_f32).abs() < 1e-4,
            "BT.2020 Kb={}",
            c.row_y[2]
        );
    }

    #[test]
    fn bt2020_white_converts_to_max_luma() {
        let src = [RgbPixel {
            r: 255,
            g: 255,
            b: 255,
        }];
        let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        rgb_to_yuv_bt2020(&src, &mut dst);
        assert!(dst[0].y > 250, "BT.2020 white Y={}", dst[0].y);
    }

    #[test]
    fn bt2020_black_converts_to_zero_luma() {
        let src = [RgbPixel { r: 0, g: 0, b: 0 }];
        let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        rgb_to_yuv_bt2020(&src, &mut dst);
        assert_eq!(dst[0].y, 0, "BT.2020 black Y should be 0");
    }

    #[test]
    fn bt2020_white_has_neutral_chroma() {
        let src = [RgbPixel {
            r: 255,
            g: 255,
            b: 255,
        }];
        let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        rgb_to_yuv_bt2020(&src, &mut dst);
        assert!(
            (i32::from(dst[0].cb) - 128).abs() <= 2,
            "BT.2020 white Cb should be ~128, got {}",
            dst[0].cb
        );
        assert!(
            (i32::from(dst[0].cr) - 128).abs() <= 2,
            "BT.2020 white Cr should be ~128, got {}",
            dst[0].cr
        );
    }

    #[test]
    fn bt2020_roundtrip_white() {
        let src_rgb = [RgbPixel {
            r: 255,
            g: 255,
            b: 255,
        }];
        let mut yuv = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        let mut dst_rgb = [RgbPixel { r: 0, g: 0, b: 0 }];
        rgb_to_yuv_bt2020(&src_rgb, &mut yuv);
        yuv_to_rgb_bt2020(&yuv, &mut dst_rgb);
        assert!(
            (i32::from(src_rgb[0].r) - i32::from(dst_rgb[0].r)).abs() <= 3,
            "BT.2020 roundtrip R"
        );
        assert!(
            (i32::from(src_rgb[0].g) - i32::from(dst_rgb[0].g)).abs() <= 3,
            "BT.2020 roundtrip G"
        );
        assert!(
            (i32::from(src_rgb[0].b) - i32::from(dst_rgb[0].b)).abs() <= 3,
            "BT.2020 roundtrip B"
        );
    }

    #[test]
    fn bt2020_roundtrip_mid_grey() {
        let src_rgb = [RgbPixel {
            r: 128,
            g: 128,
            b: 128,
        }];
        let mut yuv = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        let mut dst_rgb = [RgbPixel { r: 0, g: 0, b: 0 }];
        rgb_to_yuv_bt2020(&src_rgb, &mut yuv);
        yuv_to_rgb_bt2020(&yuv, &mut dst_rgb);
        assert!(
            (i32::from(src_rgb[0].r) - i32::from(dst_rgb[0].r)).abs() <= 3,
            "BT.2020 grey roundtrip"
        );
    }

    #[test]
    fn bt2020_neutral_chroma_round_trip() {
        // YCbCr with neutral chroma (128,128) → RGB should be achromatic
        let src = [YuvPixel {
            y: 128,
            cb: 128,
            cr: 128,
        }];
        let mut dst = [RgbPixel { r: 0, g: 0, b: 0 }];
        yuv_to_rgb_bt2020(&src, &mut dst);
        let r = i32::from(dst[0].r);
        let g = i32::from(dst[0].g);
        let b = i32::from(dst[0].b);
        assert!((r - g).abs() <= 3, "BT.2020 grey r={r} g={g}");
        assert!((r - b).abs() <= 3, "BT.2020 grey r={r} b={b}");
    }

    // ── BT.2100 ICtCp tests ────────────────────────────────────────────────────

    #[test]
    fn bt2100_ictcp_luma_coeffs_match_bt2020() {
        // ICtCp uses the same luma row as BT.2020
        let ictcp = Rgb2YuvCoeffs::bt2100_ictcp();
        let bt2020 = Rgb2YuvCoeffs::bt2020();
        assert!(
            (ictcp.row_y[0] - bt2020.row_y[0]).abs() < 1e-5,
            "ICtCp Kr mismatch"
        );
        assert!(
            (ictcp.row_y[1] - bt2020.row_y[1]).abs() < 1e-5,
            "ICtCp Kg mismatch"
        );
        assert!(
            (ictcp.row_y[2] - bt2020.row_y[2]).abs() < 1e-5,
            "ICtCp Kb mismatch"
        );
    }

    #[test]
    fn bt2100_white_intensity_is_max() {
        let src = [RgbPixel {
            r: 255,
            g: 255,
            b: 255,
        }];
        let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        rgb_to_ictcp_bt2100(&src, &mut dst);
        assert!(dst[0].y > 250, "ICtCp white I={}", dst[0].y);
    }

    #[test]
    fn bt2100_neutral_grey_has_neutral_ct_cp() {
        let src = [RgbPixel {
            r: 128,
            g: 128,
            b: 128,
        }];
        let mut dst = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        rgb_to_ictcp_bt2100(&src, &mut dst);
        assert!(
            (i32::from(dst[0].cb) - 128).abs() <= 5,
            "ICtCp grey Ct should be ~128, got {}",
            dst[0].cb
        );
        assert!(
            (i32::from(dst[0].cr) - 128).abs() <= 5,
            "ICtCp grey Cp should be ~128, got {}",
            dst[0].cr
        );
    }

    #[test]
    fn bt2100_roundtrip_white() {
        let src_rgb = [RgbPixel {
            r: 255,
            g: 255,
            b: 255,
        }];
        let mut ictcp = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        let mut dst_rgb = [RgbPixel { r: 0, g: 0, b: 0 }];
        rgb_to_ictcp_bt2100(&src_rgb, &mut ictcp);
        ictcp_to_rgb_bt2100(&ictcp, &mut dst_rgb);
        assert!(
            (i32::from(src_rgb[0].r) - i32::from(dst_rgb[0].r)).abs() <= 5,
            "ICtCp roundtrip R"
        );
    }

    #[test]
    fn bt2020_luma_increases_with_brightness() {
        let pixels = vec![
            RgbPixel { r: 0, g: 0, b: 0 },
            RgbPixel {
                r: 64,
                g: 64,
                b: 64,
            },
            RgbPixel {
                r: 128,
                g: 128,
                b: 128,
            },
            RgbPixel {
                r: 255,
                g: 255,
                b: 255,
            },
        ];
        let mut yuv = vec![YuvPixel { y: 0, cb: 0, cr: 0 }; 4];
        rgb_to_yuv_bt2020(&pixels, &mut yuv);
        assert!(yuv[0].y < yuv[1].y, "dark < medium-dark");
        assert!(yuv[1].y < yuv[2].y, "medium-dark < medium");
        assert!(yuv[2].y < yuv[3].y, "medium < white");
    }

    #[test]
    fn bt2020_differs_from_bt709() {
        // BT.2020 and BT.709 have different primaries, so they should produce
        // different results for a saturated colour input.
        let src = [RgbPixel { r: 255, g: 0, b: 0 }]; // Pure red
        let mut yuv709 = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        let mut yuv2020 = [YuvPixel { y: 0, cb: 0, cr: 0 }];
        rgb_to_yuv_bt709(&src, &mut yuv709);
        rgb_to_yuv_bt2020(&src, &mut yuv2020);
        // Luma values will differ because Kr differs (0.2126 vs 0.2627)
        // Allow both to be non-zero but check they differ
        assert!(
            yuv709[0].y != yuv2020[0].y,
            "BT.709 and BT.2020 must differ for saturated red: {} vs {}",
            yuv709[0].y,
            yuv2020[0].y
        );
    }
}
