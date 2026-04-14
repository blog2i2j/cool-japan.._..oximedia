#![allow(clippy::similar_names, clippy::many_single_char_names)]
//! Professional LUT (Look-Up Table) and color science library for `OxiMedia`.
//!
//! `oximedia-lut` provides comprehensive color management and transformation capabilities
//! for professional video and image processing workflows. This includes:
//!
//! - **LUT Operations**: 1D and 3D LUT application with multiple interpolation methods
//! - **LUT Formats**: Support for .cube, .3dl, .csp, and other industry-standard formats
//! - **Color Spaces**: Conversions between Rec.709, Rec.2020, DCI-P3, Adobe RGB, sRGB, ACES
//! - **Gamut Mapping**: Intelligent out-of-gamut color compression
//! - **Tone Mapping**: HDR to SDR and SDR to HDR transforms
//! - **ACES**: Complete ACES color management pipeline with ODTs
//! - **Color Science**: Matrix operations, chromatic adaptation, color temperature
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_lut::{Lut3d, LutInterpolation, ColorSpace};
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Load a 3D LUT from a .cube file
//! let lut = Lut3d::from_file("colorgrade.cube")?;
//!
//! // Apply the LUT to an RGB pixel
//! let input = [0.5, 0.3, 0.7];
//! let output = lut.apply(&input, LutInterpolation::Tetrahedral);
//!
//! // Convert between color spaces
//! let rec709_rgb = [0.8, 0.2, 0.4];
//! let rec2020_rgb = ColorSpace::Rec709.convert(
//!     ColorSpace::Rec2020,
//!     &rec709_rgb,
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! # Features
//!
//! ## LUT Support
//!
//! - **1D LUTs**: Per-channel curves with linear/cubic interpolation
//! - **3D LUTs**: Full RGB transforms with trilinear/tetrahedral interpolation
//! - **LUT Composition**: Chain multiple LUTs together
//! - **LUT Inversion**: Generate inverse LUTs where possible
//! - **LUT Analysis**: Validate and analyze LUT properties
//!
//! ## File Formats
//!
//! - `.cube` - Adobe/DaVinci Resolve format
//! - `.3dl` - Autodesk/Lustre format
//! - `.csp` - Cinespace format
//! - `.lut` - Generic LUT format
//!
//! ## Color Spaces
//!
//! - Rec.709 (BT.709, HD)
//! - Rec.2020 (BT.2020, UHD)
//! - DCI-P3 (Digital Cinema)
//! - Adobe RGB
//! - sRGB
//! - `ProPhoto` RGB
//! - ACES AP0 (ACES2065-1)
//! - ACES AP1 (`ACEScg`)
//!
//! ## Advanced Features
//!
//! - **Gamut Mapping**: Soft-clip, desaturate, and roll-off algorithms
//! - **Tone Mapping**: Reinhard, ACES, Hable (Uncharted 2) operators
//! - **ACES Workflow**: Full ACES color management with RRT+ODT
//! - **Chromatic Adaptation**: Bradford and Von Kries transforms
//! - **Color Temperature**: Convert between Kelvin (2000K-11000K) and RGB

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

pub mod aces;
pub mod baking;
pub mod builder;
pub mod chromatic;
pub mod color_cube;
pub mod colorspace;
pub mod compose;
pub mod creative;
pub mod cube_writer;
pub mod domain_clamp;
pub mod export;
pub mod formats;
pub mod gamut;
pub mod gamut_compress_lut;
pub mod hdr_lut;
pub mod hdr_metadata;
pub mod hdr_pipeline;
pub mod identity_lut;
pub mod interpolation;
pub mod lut1d;
pub mod lut3d;
pub mod lut_analysis;
pub mod lut_chain;
pub mod lut_combine;
pub mod lut_dither;
pub mod lut_fingerprint;
pub mod lut_gradient;
pub mod lut_interpolation;
pub mod lut_io;
pub mod lut_metadata;
pub mod lut_provenance;
pub mod lut_resample;
pub mod lut_stats;
pub mod lut_validate;
pub mod lut_version;
pub mod matrix;
pub mod preview;
pub mod temperature;
pub mod tetrahedral;
pub mod tonemap;

mod error;

pub use colorspace::{ColorSpace, TransferFunction};
pub use error::{LutError, LutResult};
pub use hdr_metadata::{
    ContentLightLevel, HdrColorSpace, HdrStandard, HdrTransferFunction, MasteringDisplayMetadata,
};
pub use hdr_pipeline::{
    aces_filmic as hdr_aces_filmic, drago as hdr_drago, hejl_filmic as hdr_hejl_filmic,
    reinhard as hdr_reinhard, reinhard_extended as hdr_reinhard_extended, rgb_luma, tone_map_pixel,
    HdrPipeline, HdrToSdrParams, ToneMappingAlgorithm,
};
pub use interpolation::LutInterpolation;
pub use lut1d::Lut1d;
pub use lut3d::Lut3d;

/// RGB color value (normalized to 0.0-1.0 range).
pub type Rgb = [f64; 3];

/// RGBA color value (normalized to 0.0-1.0 range).
pub type Rgba = [f64; 4];

/// XYZ tristimulus value.
pub type Xyz = [f64; 3];

/// 3x3 color matrix.
pub type Matrix3x3 = [[f64; 3]; 3];

/// 3x4 color matrix with offset.
pub type Matrix3x4 = [[f64; 4]; 3];

/// Size of a 3D LUT (number of entries per dimension).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LutSize {
    /// 17x17x17 LUT (standard for video).
    Size17 = 17,
    /// 33x33x33 LUT (common for color grading).
    Size33 = 33,
    /// 65x65x65 LUT (high precision).
    Size65 = 65,
}

impl LutSize {
    /// Get the size as a `usize`.
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self as usize
    }

    /// Get the total number of entries in the LUT.
    #[must_use]
    pub const fn total_entries(self) -> usize {
        let s = self.as_usize();
        s * s * s
    }
}

impl From<usize> for LutSize {
    fn from(size: usize) -> Self {
        #[allow(clippy::match_same_arms)]
        match size {
            17 => Self::Size17,
            33 => Self::Size33,
            65 => Self::Size65,
            _ => Self::Size33, // Default to 33
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lut_size() {
        assert_eq!(LutSize::Size17.as_usize(), 17);
        assert_eq!(LutSize::Size33.as_usize(), 33);
        assert_eq!(LutSize::Size65.as_usize(), 65);

        assert_eq!(LutSize::Size17.total_entries(), 17 * 17 * 17);
        assert_eq!(LutSize::Size33.total_entries(), 33 * 33 * 33);
        assert_eq!(LutSize::Size65.total_entries(), 65 * 65 * 65);
    }

    #[test]
    fn test_lut_size_from() {
        assert_eq!(LutSize::from(17), LutSize::Size17);
        assert_eq!(LutSize::from(33), LutSize::Size33);
        assert_eq!(LutSize::from(65), LutSize::Size65);
        assert_eq!(LutSize::from(42), LutSize::Size33); // Default
    }
}
