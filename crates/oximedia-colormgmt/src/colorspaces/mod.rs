//! Standard RGB color space definitions.

use crate::error::Result;
use crate::math::matrix::{invert_matrix_3x3, multiply_matrix_vector, Matrix3x3};
use crate::xyz::Xyz;
use oximedia_core::hdr::{Primaries, TransferCharacteristic, WhitePoint};

/// RGB color space definition with primaries, white point, and transfer function.
#[derive(Clone, Debug, PartialEq)]
pub struct ColorSpace {
    /// Name of the color space
    pub name: String,
    /// RGB primaries in CIE xy chromaticity
    pub primaries: Primaries,
    /// White point
    pub white_point: WhitePoint,
    /// Transfer characteristic (EOTF)
    pub transfer: TransferCharacteristic,
    /// RGB to XYZ transformation matrix
    pub rgb_to_xyz: Matrix3x3,
    /// XYZ to RGB transformation matrix
    pub xyz_to_rgb: Matrix3x3,
}

impl ColorSpace {
    /// Creates a new color space from primaries, white point, and transfer function.
    ///
    /// # Errors
    ///
    /// Returns an error if the matrix calculation fails.
    pub fn new(
        name: String,
        primaries: Primaries,
        white_point: WhitePoint,
        transfer: TransferCharacteristic,
    ) -> Result<Self> {
        let rgb_to_xyz = compute_rgb_to_xyz_matrix(&primaries, &white_point)?;
        let xyz_to_rgb = invert_matrix_3x3(&rgb_to_xyz)?;

        Ok(Self {
            name,
            primaries,
            white_point,
            transfer,
            rgb_to_xyz,
            xyz_to_rgb,
        })
    }

    /// sRGB color space (IEC 61966-2-1).
    ///
    /// The standard for computer displays and web content.
    /// Uses BT.709 primaries with D65 white point and sRGB transfer function.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix calculation fails (should never happen for standard spaces).
    pub fn srgb() -> Result<Self> {
        Self::new(
            "sRGB".to_string(),
            Primaries::bt709(),
            WhitePoint::D65,
            TransferCharacteristic::Srgb,
        )
    }

    /// Adobe RGB (1998) color space.
    ///
    /// Wide gamut color space for photography and print.
    /// Uses pure gamma 2.2 transfer function.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix calculation fails.
    pub fn adobe_rgb() -> Result<Self> {
        Self::new(
            "Adobe RGB (1998)".to_string(),
            Primaries::adobe_rgb(),
            WhitePoint::D65,
            TransferCharacteristic::Bt709, // Adobe RGB uses gamma 2.2, approximated with BT.709
        )
    }

    /// `ProPhoto` RGB color space.
    ///
    /// Very wide gamut color space, larger than Adobe RGB.
    /// Uses gamma 1.8 transfer function.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix calculation fails.
    pub fn prophoto_rgb() -> Result<Self> {
        let primaries = Primaries::new(
            (0.734_699, 0.265_301),
            (0.159_597, 0.840_403),
            (0.036_598, 0.000_105),
        );

        Self::new(
            "ProPhoto RGB".to_string(),
            primaries,
            WhitePoint::D50,
            TransferCharacteristic::Bt709, // ProPhoto uses gamma 1.8, approximated
        )
    }

    /// Display P3 color space.
    ///
    /// DCI-P3 primaries with D65 white point, used by Apple displays.
    /// Uses sRGB transfer function.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix calculation fails.
    pub fn display_p3() -> Result<Self> {
        Self::new(
            "Display P3".to_string(),
            Primaries::display_p3(),
            WhitePoint::D65,
            TransferCharacteristic::Srgb,
        )
    }

    /// DCI-P3 color space (digital cinema).
    ///
    /// Uses DCI white point (~5900K) and gamma 2.6.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix calculation fails.
    pub fn dci_p3() -> Result<Self> {
        Self::new(
            "DCI-P3".to_string(),
            Primaries::dci_p3(),
            WhitePoint::Dci,
            TransferCharacteristic::Bt709, // DCI-P3 uses gamma 2.6, approximated
        )
    }

    /// Rec.709 (BT.709) color space.
    ///
    /// Standard for HDTV. Same primaries as sRGB but with BT.709 transfer function.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix calculation fails.
    pub fn rec709() -> Result<Self> {
        Self::new(
            "Rec.709".to_string(),
            Primaries::bt709(),
            WhitePoint::D65,
            TransferCharacteristic::Bt709,
        )
    }

    /// Rec.2020 (BT.2020) color space.
    ///
    /// Wide gamut standard for UHDTV and HDR content.
    /// Linear transfer function for HDR processing.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix calculation fails.
    pub fn rec2020() -> Result<Self> {
        Self::new(
            "Rec.2020".to_string(),
            Primaries::bt2020(),
            WhitePoint::D65,
            TransferCharacteristic::Linear,
        )
    }

    /// Rec.2020 with PQ transfer function (HDR10).
    ///
    /// # Errors
    ///
    /// Returns an error if matrix calculation fails.
    pub fn rec2020_pq() -> Result<Self> {
        Self::new(
            "Rec.2020 (PQ)".to_string(),
            Primaries::bt2020(),
            WhitePoint::D65,
            TransferCharacteristic::Pq,
        )
    }

    /// Rec.2020 with HLG transfer function.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix calculation fails.
    pub fn rec2020_hlg() -> Result<Self> {
        Self::new(
            "Rec.2020 (HLG)".to_string(),
            Primaries::bt2020(),
            WhitePoint::D65,
            TransferCharacteristic::Hlg,
        )
    }

    /// Linear RGB with Rec.709 primaries.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix calculation fails.
    pub fn linear_rec709() -> Result<Self> {
        Self::new(
            "Linear Rec.709".to_string(),
            Primaries::bt709(),
            WhitePoint::D65,
            TransferCharacteristic::Linear,
        )
    }

    /// Converts non-linear RGB to linear RGB using the transfer function.
    #[must_use]
    pub fn linearize(&self, rgb: [f64; 3]) -> [f64; 3] {
        [
            self.transfer.eotf(rgb[0]),
            self.transfer.eotf(rgb[1]),
            self.transfer.eotf(rgb[2]),
        ]
    }

    /// Converts linear RGB to non-linear RGB using the inverse transfer function.
    #[must_use]
    pub fn delinearize(&self, linear_rgb: [f64; 3]) -> [f64; 3] {
        [
            self.transfer.oetf(linear_rgb[0]),
            self.transfer.oetf(linear_rgb[1]),
            self.transfer.oetf(linear_rgb[2]),
        ]
    }

    /// Converts RGB to XYZ.
    #[must_use]
    pub fn rgb_to_xyz(&self, rgb: [f64; 3]) -> Xyz {
        let linear = self.linearize(rgb);
        let xyz = multiply_matrix_vector(&self.rgb_to_xyz, linear);
        Xyz::from_array(xyz)
    }

    /// Converts XYZ to RGB.
    #[must_use]
    pub fn xyz_to_rgb(&self, xyz: &Xyz) -> [f64; 3] {
        let linear = multiply_matrix_vector(&self.xyz_to_rgb, xyz.as_array());
        self.delinearize(linear)
    }

    /// Returns the white point in XYZ coordinates.
    #[must_use]
    pub fn white_point_xyz(&self) -> Xyz {
        let (x, y) = self.white_point.xy();
        Xyz::from_xyy(x, y, 1.0)
    }
}

/// Computes the RGB to XYZ transformation matrix from primaries and white point.
///
/// # Errors
///
/// Returns an error if matrix inversion fails.
fn compute_rgb_to_xyz_matrix(primaries: &Primaries, white_point: &WhitePoint) -> Result<Matrix3x3> {
    // Get chromaticity coordinates
    let (xr, yr) = primaries.red;
    let (xg, yg) = primaries.green;
    let (xb, yb) = primaries.blue;
    let (xw, yw) = white_point.xy();

    // Convert to XYZ
    let xr_xyz = xr / yr;
    let yr_xyz = 1.0;
    let zr_xyz = (1.0 - xr - yr) / yr;

    let xg_xyz = xg / yg;
    let yg_xyz = 1.0;
    let zg_xyz = (1.0 - xg - yg) / yg;

    let xb_xyz = xb / yb;
    let yb_xyz = 1.0;
    let zb_xyz = (1.0 - xb - yb) / yb;

    let xw_xyz = xw / yw;
    let yw_xyz = 1.0;
    let zw_xyz = (1.0 - xw - yw) / yw;

    // Create matrix of primaries
    let primaries_matrix = [
        [xr_xyz, xg_xyz, xb_xyz],
        [yr_xyz, yg_xyz, yb_xyz],
        [zr_xyz, zg_xyz, zb_xyz],
    ];

    // Invert to get scaling factors
    let inv_primaries = invert_matrix_3x3(&primaries_matrix)?;
    let white = [xw_xyz, yw_xyz, zw_xyz];
    let scale = multiply_matrix_vector(&inv_primaries, white);

    // Apply scaling to primaries
    let rgb_to_xyz = [
        [
            primaries_matrix[0][0] * scale[0],
            primaries_matrix[0][1] * scale[1],
            primaries_matrix[0][2] * scale[2],
        ],
        [
            primaries_matrix[1][0] * scale[0],
            primaries_matrix[1][1] * scale[1],
            primaries_matrix[1][2] * scale[2],
        ],
        [
            primaries_matrix[2][0] * scale[0],
            primaries_matrix[2][1] * scale[1],
            primaries_matrix[2][2] * scale[2],
        ],
    ];

    Ok(rgb_to_xyz)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_white_point() {
        let srgb = ColorSpace::srgb().expect("sRGB color space creation should succeed");
        let white_rgb = [1.0, 1.0, 1.0];
        let xyz = srgb.rgb_to_xyz(white_rgb);

        // Should be close to D65
        let d65 = Xyz::d65();
        assert!((xyz.x - d65.x).abs() < 0.01);
        assert!((xyz.y - d65.y).abs() < 0.01);
        assert!((xyz.z - d65.z).abs() < 0.01);
    }

    #[test]
    fn test_srgb_roundtrip() {
        let srgb = ColorSpace::srgb().expect("sRGB color space creation should succeed");
        let rgb = [0.5, 0.3, 0.7];

        let xyz = srgb.rgb_to_xyz(rgb);
        let rgb2 = srgb.xyz_to_rgb(&xyz);

        assert!((rgb2[0] - rgb[0]).abs() < 1e-6);
        assert!((rgb2[1] - rgb[1]).abs() < 1e-6);
        assert!((rgb2[2] - rgb[2]).abs() < 1e-6);
    }

    #[test]
    fn test_linearize_delinearize() {
        let srgb = ColorSpace::srgb().expect("sRGB color space creation should succeed");
        let rgb = [0.5, 0.3, 0.7];

        let linear = srgb.linearize(rgb);
        let rgb2 = srgb.delinearize(linear);

        assert!((rgb2[0] - rgb[0]).abs() < 1e-10);
        assert!((rgb2[1] - rgb[1]).abs() < 1e-10);
        assert!((rgb2[2] - rgb[2]).abs() < 1e-10);
    }

    #[test]
    fn test_color_space_creation() {
        assert!(ColorSpace::srgb().is_ok());
        assert!(ColorSpace::adobe_rgb().is_ok());
        assert!(ColorSpace::prophoto_rgb().is_ok());
        assert!(ColorSpace::display_p3().is_ok());
        assert!(ColorSpace::rec709().is_ok());
        assert!(ColorSpace::rec2020().is_ok());
    }
}
