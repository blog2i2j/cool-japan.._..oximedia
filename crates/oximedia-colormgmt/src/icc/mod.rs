//! ICC color profile support.
//!
//! This module provides parsing and application of ICC profiles (v2 and v4).
//! Full ICC profile support is complex; this is a simplified implementation
//! covering the most common use cases.

pub mod parser;

pub use parser::IccParser;

use crate::error::{ColorError, Result};
use crate::math::matrix::Matrix3x3;
use std::io::Read;

/// ICC profile class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProfileClass {
    /// Input device profile (camera, scanner)
    Input,
    /// Display device profile (monitor)
    Display,
    /// Output device profile (printer)
    Output,
    /// Device link profile
    DeviceLink,
    /// Color space conversion profile
    ColorSpace,
    /// Abstract profile
    Abstract,
    /// Named color profile
    NamedColor,
}

/// Rendering intent for profile application.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderingIntent {
    /// Perceptual rendering (photographic images)
    Perceptual,
    /// Relative colorimetric (preserves in-gamut colors)
    RelativeColorimetric,
    /// Saturation rendering (business graphics)
    Saturation,
    /// Absolute colorimetric (proofing)
    AbsoluteColorimetric,
}

/// ICC color profile.
#[derive(Clone, Debug)]
pub struct IccProfile {
    /// Profile class
    pub class: ProfileClass,
    /// Profile description
    pub description: String,
    /// Copyright information
    pub copyright: String,
    /// Color space (e.g., "RGB ", "CMYK")
    pub color_space: [u8; 4],
    /// Profile connection space (typically "XYZ " or "Lab ")
    pub pcs: [u8; 4],
    /// RGB to XYZ matrix (for matrix-based profiles)
    pub rgb_to_xyz: Option<Matrix3x3>,
    /// XYZ to RGB matrix
    pub xyz_to_rgb: Option<Matrix3x3>,
    /// Red, green, blue TRC (tone reproduction curves) as 1D LUTs
    pub trc: Option<(Vec<f32>, Vec<f32>, Vec<f32>)>,
}

impl IccProfile {
    /// Creates a simple sRGB ICC profile.
    #[must_use]
    pub fn srgb() -> Self {
        // sRGB to XYZ matrix (D65)
        let rgb_to_xyz = [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ];

        Self {
            class: ProfileClass::Display,
            description: "sRGB IEC61966-2.1".to_string(),
            copyright: "Public Domain".to_string(),
            color_space: *b"RGB ",
            pcs: *b"XYZ ",
            rgb_to_xyz: Some(rgb_to_xyz),
            xyz_to_rgb: crate::math::matrix::invert_matrix_3x3(&rgb_to_xyz).ok(),
            trc: None, // sRGB TRC is parametric, not LUT-based
        }
    }

    /// Parses an ICC profile from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile is invalid or unsupported.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        if data.len() < 128 {
            return Err(ColorError::IccProfile("Profile too small".to_string()));
        }

        // Read header (ICC spec section 7.2)
        let profile_size = u32::from_be_bytes([data[0], data[1], data[2], data[3]]) as usize;

        if data.len() < profile_size {
            return Err(ColorError::IccProfile("Truncated profile".to_string()));
        }

        // Check signature (bytes 36-39 should be "acsp")
        if &data[36..40] != b"acsp" {
            return Err(ColorError::IccProfile("Invalid signature".to_string()));
        }

        let class = Self::parse_profile_class(&data[12..16])?;
        let color_space = [data[16], data[17], data[18], data[19]];
        let pcs = [data[20], data[21], data[22], data[23]];

        Ok(Self {
            class,
            description: "ICC Profile".to_string(),
            copyright: String::new(),
            color_space,
            pcs,
            rgb_to_xyz: None,
            xyz_to_rgb: None,
            trc: None,
        })
    }

    /// Parses an ICC profile from a reader.
    ///
    /// # Errors
    ///
    /// Returns an error if reading fails or the profile is invalid.
    pub fn from_reader<R: Read>(reader: &mut R) -> Result<Self> {
        let mut data = Vec::new();
        reader
            .read_to_end(&mut data)
            .map_err(|e| ColorError::IccProfile(format!("Failed to read profile: {e}")))?;

        Self::from_bytes(&data)
    }

    fn parse_profile_class(bytes: &[u8]) -> Result<ProfileClass> {
        match bytes {
            b"scnr" => Ok(ProfileClass::Input),
            b"mntr" => Ok(ProfileClass::Display),
            b"prtr" => Ok(ProfileClass::Output),
            b"link" => Ok(ProfileClass::DeviceLink),
            b"spac" => Ok(ProfileClass::ColorSpace),
            b"abst" => Ok(ProfileClass::Abstract),
            b"nmcl" => Ok(ProfileClass::NamedColor),
            _ => Err(ColorError::IccProfile(format!(
                "Unknown profile class: {bytes:?}"
            ))),
        }
    }

    /// Applies the profile to convert RGB to XYZ.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile doesn't support this conversion.
    pub fn rgb_to_xyz(&self, rgb: [f64; 3]) -> Result<[f64; 3]> {
        let matrix = self
            .rgb_to_xyz
            .ok_or_else(|| ColorError::IccProfile("No RGB to XYZ matrix".to_string()))?;

        // Apply TRC if available, otherwise assume linear
        let linear = if let Some((r_trc, g_trc, b_trc)) = &self.trc {
            [
                interpolate_trc(rgb[0], r_trc),
                interpolate_trc(rgb[1], g_trc),
                interpolate_trc(rgb[2], b_trc),
            ]
        } else {
            // Assume sRGB TRC
            crate::colorspaces::ColorSpace::srgb()
                .map_err(|_| ColorError::IccProfile("Failed to create sRGB space".to_string()))?
                .linearize(rgb)
        };

        Ok(crate::math::matrix::multiply_matrix_vector(&matrix, linear))
    }

    /// Applies the profile to convert XYZ to RGB.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile doesn't support this conversion.
    pub fn xyz_to_rgb(&self, xyz: [f64; 3]) -> Result<[f64; 3]> {
        let matrix = self
            .xyz_to_rgb
            .ok_or_else(|| ColorError::IccProfile("No XYZ to RGB matrix".to_string()))?;

        let linear = crate::math::matrix::multiply_matrix_vector(&matrix, xyz);

        // Apply inverse TRC if available
        if let Some((r_trc, g_trc, b_trc)) = &self.trc {
            Ok([
                inverse_trc(linear[0], r_trc),
                inverse_trc(linear[1], g_trc),
                inverse_trc(linear[2], b_trc),
            ])
        } else {
            // Assume sRGB TRC
            Ok(crate::colorspaces::ColorSpace::srgb()
                .map_err(|_| ColorError::IccProfile("Failed to create sRGB space".to_string()))?
                .delinearize(linear))
        }
    }
}

fn interpolate_trc(value: f64, trc: &[f32]) -> f64 {
    if trc.is_empty() {
        return value;
    }

    let size = trc.len();
    let pos = value * (size - 1) as f64;
    let idx = pos.floor() as usize;
    let frac = pos - idx as f64;

    if idx >= size - 1 {
        return f64::from(trc[size - 1]);
    }

    let v0 = f64::from(trc[idx]);
    let v1 = f64::from(trc[idx + 1]);

    v0 + (v1 - v0) * frac
}

fn inverse_trc(value: f64, trc: &[f32]) -> f64 {
    // Simplified inverse - binary search through TRC
    if trc.is_empty() {
        return value;
    }

    // Binary search for closest value
    let mut low = 0;
    let mut high = trc.len() - 1;

    while low < high {
        let mid = (low + high) / 2;
        if f64::from(trc[mid]) < value {
            low = mid + 1;
        } else {
            high = mid;
        }
    }

    low as f64 / (trc.len() - 1) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_srgb_profile() {
        let profile = IccProfile::srgb();
        assert_eq!(profile.class, ProfileClass::Display);
        assert_eq!(profile.color_space, *b"RGB ");
        assert_eq!(profile.pcs, *b"XYZ ");
        assert!(profile.rgb_to_xyz.is_some());
        assert!(profile.xyz_to_rgb.is_some());
    }

    #[test]
    fn test_profile_class_parsing() {
        assert_eq!(
            IccProfile::parse_profile_class(b"mntr").expect("profile class parsing should succeed"),
            ProfileClass::Display
        );
        assert_eq!(
            IccProfile::parse_profile_class(b"scnr").expect("profile class parsing should succeed"),
            ProfileClass::Input
        );
    }

    #[test]
    fn test_rgb_to_xyz_conversion() {
        let profile = IccProfile::srgb();
        let white = [1.0, 1.0, 1.0];
        let xyz = profile
            .rgb_to_xyz(white)
            .expect("RGB to XYZ conversion should succeed");

        // Should be close to D65
        assert!((xyz[0] - 0.95047).abs() < 0.01);
        assert!((xyz[1] - 1.0).abs() < 0.01);
        assert!((xyz[2] - 1.08883).abs() < 0.01);
    }

    #[test]
    fn test_invalid_profile() {
        let data = vec![0u8; 100];
        assert!(IccProfile::from_bytes(&data).is_err());
    }
}
