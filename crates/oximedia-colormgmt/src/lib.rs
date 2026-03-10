#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::too_many_arguments
)]
//! Professional color management system for `OxiMedia`.
//!
//! This crate provides comprehensive color management capabilities including:
//!
//! - **Standard Color Spaces**: sRGB, Adobe RGB, `ProPhoto` RGB, Display P3, Rec.709, Rec.2020, DCI-P3
//! - **ACES Support**: Full ACES workflow (IDT, RRT, ODT, LMT) with AP0 and AP1 primaries
//! - **ICC Profile Support**: Parse, validate, and apply ICC v2/v4 profiles
//! - **HDR Processing**: PQ/HLG transfer functions, tone mapping operators
//! - **Gamut Mapping**: Advanced gamut compression and expansion algorithms
//! - **Color Transforms**: Matrix-based, LUT-based, and parametric transforms
//! - **Professional Accuracy**: ΔE < 1 for standard conversions, proper linear-light processing
//!
//! # Examples
//!
//! ## Basic Color Space Conversion
//!
//! ```
//! use oximedia_colormgmt::{colorspaces::ColorSpace, transforms::rgb_to_rgb};
//!
//! let srgb = ColorSpace::srgb()?;
//! let rec2020 = ColorSpace::rec2020()?;
//!
//! let rgb = [0.5, 0.3, 0.2];
//! let converted = rgb_to_rgb(&rgb, &srgb, &rec2020);
//! ```
//!
//! ## ACES Workflow
//!
//! ```
//! use oximedia_colormgmt::aces::{AcesColorSpace, AcesTransform};
//!
//! // Convert from ACEScg to ACES2065-1
//! let acescg = AcesColorSpace::ACEScg;
//! let aces2065 = AcesColorSpace::ACES2065_1;
//!
//! let transform = AcesTransform::new(acescg, aces2065);
//! let converted = transform.apply([0.5, 0.3, 0.2]);
//! ```
//!
//! ## Color Pipeline
//!
//! ```
//! use oximedia_colormgmt::pipeline::{ColorPipeline, ColorTransform};
//! use oximedia_colormgmt::colorspaces::ColorSpace;
//!
//! let srgb = ColorSpace::srgb()?;
//! let mut pipeline = ColorPipeline::new();
//! pipeline.add_transform(ColorTransform::Linearize(srgb));
//! pipeline.add_transform(ColorTransform::Matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]));
//!
//! let result = pipeline.transform_pixel([0.5, 0.3, 0.2]);
//! ```

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod aces;
pub mod aces_config;
pub mod aces_pipeline;
pub mod chromatic_adapt;
pub mod chromatic_adaptation;
pub mod color_appearance;
pub mod color_blindness;
pub mod color_convert;
pub mod color_diff;
pub mod color_harmony;
pub mod color_quantize;
pub mod colorspaces;
pub mod curves;
pub mod display_profile;
pub mod error;
pub mod gamut;
pub mod gamut_clip;
pub mod gamut_mapping;
pub mod gamut_ops;
pub mod grading;
pub mod hdr;
pub mod hdr_color;
pub mod icc;
pub mod icc_profile;
pub mod lab_color;
pub mod match_color;
pub mod math;
pub mod pipeline;
pub mod rendering_intent;
pub mod spectral_data;
pub mod spectral_locus;
pub mod tone_map;
pub mod transfer_function;
pub mod transforms;
pub mod utils;
pub mod white_point;
pub mod xyz;

pub use color_convert::{ColorSpaceId, ColorTransformUtil, TransferFunctionId};

pub use error::{ColorError, Result};

/// Delta E (color difference) calculations.
pub mod delta_e {
    //! Color difference metrics (ΔE) for perceptual color comparison.

    use crate::xyz::Lab;

    /// Calculates ΔE 1976 (CIE76) - simple Euclidean distance in Lab space.
    ///
    /// # Arguments
    ///
    /// * `lab1` - First color in Lab
    /// * `lab2` - Second color in Lab
    ///
    /// # Returns
    ///
    /// Color difference value (ΔE). Values < 1 are imperceptible, < 2.3 are acceptable.
    #[must_use]
    pub fn delta_e_1976(lab1: &Lab, lab2: &Lab) -> f64 {
        let dl = lab1.l - lab2.l;
        let da = lab1.a - lab2.a;
        let db = lab1.b - lab2.b;
        (dl * dl + da * da + db * db).sqrt()
    }

    /// Calculates ΔE 2000 (CIEDE2000) - industry standard perceptual color difference.
    ///
    /// More accurate than ΔE 1976, accounting for perceptual non-uniformities.
    ///
    /// # Arguments
    ///
    /// * `lab1` - First color in Lab
    /// * `lab2` - Second color in Lab
    ///
    /// # Returns
    ///
    /// Color difference value (ΔE 2000). Values < 1 are imperceptible.
    #[must_use]
    #[allow(clippy::too_many_lines)]
    pub fn delta_e_2000(lab1: &Lab, lab2: &Lab) -> f64 {
        use std::f64::consts::PI;

        let l1 = lab1.l;
        let a1 = lab1.a;
        let b1 = lab1.b;
        let l2 = lab2.l;
        let a2 = lab2.a;
        let b2 = lab2.b;

        // Calculate C1 and C2
        let c1 = (a1 * a1 + b1 * b1).sqrt();
        let c2 = (a2 * a2 + b2 * b2).sqrt();
        let c_bar = (c1 + c2) / 2.0;

        // Calculate G
        let g = 0.5 * (1.0 - ((c_bar.powi(7)) / (c_bar.powi(7) + 25.0_f64.powi(7))).sqrt());

        // Calculate a'
        let a1_prime = a1 * (1.0 + g);
        let a2_prime = a2 * (1.0 + g);

        // Calculate C' and h'
        let c1_prime = (a1_prime * a1_prime + b1 * b1).sqrt();
        let c2_prime = (a2_prime * a2_prime + b2 * b2).sqrt();

        let h1_prime = if b1 == 0.0 && a1_prime == 0.0 {
            0.0
        } else {
            let mut h = b1.atan2(a1_prime).to_degrees();
            if h < 0.0 {
                h += 360.0;
            }
            h
        };

        let h2_prime = if b2 == 0.0 && a2_prime == 0.0 {
            0.0
        } else {
            let mut h = b2.atan2(a2_prime).to_degrees();
            if h < 0.0 {
                h += 360.0;
            }
            h
        };

        // Calculate ΔL', ΔC', ΔH'
        let delta_l_prime = l2 - l1;
        let delta_c_prime = c2_prime - c1_prime;

        let delta_h_prime = if c1_prime * c2_prime == 0.0 {
            0.0
        } else if (h2_prime - h1_prime).abs() <= 180.0 {
            h2_prime - h1_prime
        } else if h2_prime - h1_prime > 180.0 {
            h2_prime - h1_prime - 360.0
        } else {
            h2_prime - h1_prime + 360.0
        };

        let delta_big_h_prime =
            2.0 * (c1_prime * c2_prime).sqrt() * ((delta_h_prime / 2.0) * PI / 180.0).sin();

        // Calculate L', C', H' bar
        let l_bar_prime = (l1 + l2) / 2.0;
        let c_bar_prime = (c1_prime + c2_prime) / 2.0;

        let h_bar_prime = if c1_prime * c2_prime == 0.0 {
            h1_prime + h2_prime
        } else if (h1_prime - h2_prime).abs() <= 180.0 {
            (h1_prime + h2_prime) / 2.0
        } else if h1_prime + h2_prime < 360.0 {
            (h1_prime + h2_prime + 360.0) / 2.0
        } else {
            (h1_prime + h2_prime - 360.0) / 2.0
        };

        // Calculate T
        let t = 1.0 - 0.17 * ((h_bar_prime - 30.0) * PI / 180.0).cos()
            + 0.24 * ((2.0 * h_bar_prime) * PI / 180.0).cos()
            + 0.32 * ((3.0 * h_bar_prime + 6.0) * PI / 180.0).cos()
            - 0.20 * ((4.0 * h_bar_prime - 63.0) * PI / 180.0).cos();

        // Calculate S_L, S_C, S_H
        let s_l = 1.0
            + ((0.015 * (l_bar_prime - 50.0).powi(2))
                / (20.0 + (l_bar_prime - 50.0).powi(2)).sqrt());
        let s_c = 1.0 + 0.045 * c_bar_prime;
        let s_h = 1.0 + 0.015 * c_bar_prime * t;

        // Calculate R_T
        let delta_theta = 30.0 * (-(((h_bar_prime - 275.0) / 25.0).powi(2))).exp();
        let r_c = 2.0 * ((c_bar_prime.powi(7)) / (c_bar_prime.powi(7) + 25.0_f64.powi(7))).sqrt();
        let r_t = -r_c * (2.0 * delta_theta * PI / 180.0).sin();

        // Calculate ΔE 2000
        let k_l = 1.0;
        let k_c = 1.0;
        let k_h = 1.0;

        ((delta_l_prime / (k_l * s_l)).powi(2)
            + (delta_c_prime / (k_c * s_c)).powi(2)
            + (delta_big_h_prime / (k_h * s_h)).powi(2)
            + r_t * (delta_c_prime / (k_c * s_c)) * (delta_big_h_prime / (k_h * s_h)))
            .sqrt()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_crate_version() {
        assert!(!env!("CARGO_PKG_VERSION").is_empty());
    }
}
