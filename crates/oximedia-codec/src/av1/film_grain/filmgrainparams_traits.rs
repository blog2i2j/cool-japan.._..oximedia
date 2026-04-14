//! # FilmGrainParams - Trait Implementations
//!
//! This module contains trait implementations for `FilmGrainParams`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::functions::{
    MAX_AR_COEFFS_CHROMA, MAX_AR_COEFFS_LUMA, MAX_CHROMA_SCALING_POINTS, MAX_LUMA_SCALING_POINTS,
};
use super::types::{FilmGrainParams, ScalingPoint};

impl Default for FilmGrainParams {
    fn default() -> Self {
        Self {
            apply_grain: false,
            grain_seed: 0,
            update_grain: false,
            film_grain_params_present: false,
            num_y_points: 0,
            y_points: [ScalingPoint::default(); MAX_LUMA_SCALING_POINTS],
            chroma_scaling_from_luma: false,
            num_cb_points: 0,
            cb_points: [ScalingPoint::default(); MAX_CHROMA_SCALING_POINTS],
            num_cr_points: 0,
            cr_points: [ScalingPoint::default(); MAX_CHROMA_SCALING_POINTS],
            grain_scaling_minus_8: 0,
            ar_coeff_lag: 0,
            ar_coeffs_y: [0; MAX_AR_COEFFS_LUMA],
            ar_coeffs_cb: [0; MAX_AR_COEFFS_CHROMA],
            ar_coeffs_cr: [0; MAX_AR_COEFFS_CHROMA],
            ar_coeff_shift_minus_6: 0,
            grain_scale_shift: 0,
            cb_mult: 128,
            cb_luma_mult: 192,
            cb_offset: 256,
            cr_mult: 128,
            cr_luma_mult: 192,
            cr_offset: 256,
            overlap_flag: true,
            clip_to_restricted_range: false,
        }
    }
}
