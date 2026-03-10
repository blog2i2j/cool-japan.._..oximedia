//! AV1 Film Grain Table and Presets.
//!
//! This module provides film grain parameter storage, management, and preset patterns
//! for AV1 encoding and decoding.
//!
//! # Features
//!
//! - Film grain parameter storage and interpolation
//! - Per-frame grain parameter tracking
//! - Preset grain patterns (35mm, 16mm film stocks)
//! - Custom grain pattern support
//! - Grain intensity levels
//!
//! # Preset Patterns
//!
//! The module includes several preset grain patterns that simulate real film stocks:
//!
//! - **35mm Film** - Professional cinema film grain (light, medium, heavy)
//! - **16mm Film** - Documentary/indie film grain (light, medium, heavy)
//! - **Digital NR** - Noise reduction artifacts
//! - **Custom** - User-defined patterns
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::av1::film_grain_table::{FilmGrainTable, GrainPreset};
//!
//! let mut table = FilmGrainTable::new();
//!
//! // Use a preset
//! let params = GrainPreset::Film35mmMedium.to_params(8);
//! table.insert(0, params);
//!
//! // Retrieve parameters for frame
//! let frame_params = table.get(42)?;
//! ```

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::similar_names)]
#![allow(clippy::module_name_repetitions)]

use super::film_grain::{FilmGrainParams, ScalingPoint};
use std::collections::BTreeMap;

// =============================================================================
// Film Grain Table
// =============================================================================

/// Film grain parameter table.
///
/// Stores film grain parameters per frame and supports interpolation between frames.
#[derive(Clone, Debug, Default)]
pub struct FilmGrainTable {
    /// Frame number to parameters mapping.
    entries: BTreeMap<u64, FilmGrainParams>,
    /// Default parameters for frames without explicit entry.
    default_params: FilmGrainParams,
}

impl FilmGrainTable {
    /// Create a new empty film grain table.
    #[must_use]
    pub fn new() -> Self {
        Self {
            entries: BTreeMap::new(),
            default_params: FilmGrainParams::default(),
        }
    }

    /// Create a table with default parameters.
    #[must_use]
    pub fn with_default(default_params: FilmGrainParams) -> Self {
        Self {
            entries: BTreeMap::new(),
            default_params,
        }
    }

    /// Insert parameters for a specific frame.
    pub fn insert(&mut self, frame_num: u64, params: FilmGrainParams) {
        self.entries.insert(frame_num, params);
    }

    /// Get parameters for a specific frame.
    ///
    /// If no exact match exists, returns the nearest previous frame's parameters,
    /// or the default parameters if no previous frame exists.
    #[must_use]
    pub fn get(&self, frame_num: u64) -> Option<&FilmGrainParams> {
        // Try exact match first
        if let Some(params) = self.entries.get(&frame_num) {
            return Some(params);
        }

        // Find nearest previous frame
        self.entries
            .range(..=frame_num)
            .next_back()
            .map(|(_, params)| params)
            .or(Some(&self.default_params))
    }

    /// Get mutable parameters for a specific frame.
    pub fn get_mut(&mut self, frame_num: u64) -> Option<&mut FilmGrainParams> {
        self.entries.get_mut(&frame_num)
    }

    /// Get parameters with interpolation between keyframes.
    #[must_use]
    pub fn get_interpolated(&self, frame_num: u64) -> FilmGrainParams {
        // Find surrounding keyframes
        let prev = self.entries.range(..=frame_num).next_back();
        let next = self.entries.range(frame_num..).nth(1);

        match (prev, next) {
            (Some((prev_frame, prev_params)), Some((next_frame, next_params))) => {
                // Interpolate between keyframes
                let t = if next_frame > prev_frame {
                    (frame_num - prev_frame) as f32 / (next_frame - prev_frame) as f32
                } else {
                    0.0
                };
                interpolate_params(prev_params, next_params, t)
            }
            (Some((_, params)), None) => params.clone(),
            (None, Some((_, params))) => params.clone(),
            (None, None) => self.default_params.clone(),
        }
    }

    /// Remove parameters for a specific frame.
    pub fn remove(&mut self, frame_num: u64) -> Option<FilmGrainParams> {
        self.entries.remove(&frame_num)
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Get number of entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Iterate over all entries.
    pub fn iter(&self) -> impl Iterator<Item = (&u64, &FilmGrainParams)> {
        self.entries.iter()
    }

    /// Set default parameters.
    pub fn set_default(&mut self, params: FilmGrainParams) {
        self.default_params = params;
    }

    /// Get default parameters.
    #[must_use]
    pub const fn default_params(&self) -> &FilmGrainParams {
        &self.default_params
    }
}

/// Interpolate between two film grain parameter sets.
fn interpolate_params(p0: &FilmGrainParams, p1: &FilmGrainParams, t: f32) -> FilmGrainParams {
    let t = t.clamp(0.0, 1.0);
    let t_u8 = (t * 255.0) as u8;

    let mut result = p0.clone();

    // Interpolate grain_seed
    result.grain_seed = lerp_u16(p0.grain_seed, p1.grain_seed, t);

    // Interpolate scaling points (simplified - just blend scaling values)
    for i in 0..result.num_y_points.min(p1.num_y_points) {
        result.y_points[i].scaling = lerp_u8(p0.y_points[i].scaling, p1.y_points[i].scaling, t_u8);
    }

    // Interpolate chroma parameters
    result.cb_mult = lerp_u8(p0.cb_mult, p1.cb_mult, t_u8);
    result.cb_luma_mult = lerp_u8(p0.cb_luma_mult, p1.cb_luma_mult, t_u8);
    result.cb_offset = lerp_u16(p0.cb_offset, p1.cb_offset, t);
    result.cr_mult = lerp_u8(p0.cr_mult, p1.cr_mult, t_u8);
    result.cr_luma_mult = lerp_u8(p0.cr_luma_mult, p1.cr_luma_mult, t_u8);
    result.cr_offset = lerp_u16(p0.cr_offset, p1.cr_offset, t);

    result
}

/// Linear interpolation for u8.
fn lerp_u8(a: u8, b: u8, t: u8) -> u8 {
    let a = u16::from(a);
    let b = u16::from(b);
    let t = u16::from(t);
    ((a * (255 - t) + b * t + 127) / 255) as u8
}

/// Linear interpolation for u16.
fn lerp_u16(a: u16, b: u16, t: f32) -> u16 {
    let a = a as f32;
    let b = b as f32;
    (a + (b - a) * t) as u16
}

// =============================================================================
// Grain Presets
// =============================================================================

/// Grain intensity level.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GrainIntensity {
    /// Light grain (subtle).
    Light,
    /// Medium grain (moderate).
    Medium,
    /// Heavy grain (pronounced).
    Heavy,
}

/// Preset film grain patterns.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GrainPreset {
    /// No grain.
    None,
    /// 35mm film stock - light grain.
    Film35mmLight,
    /// 35mm film stock - medium grain.
    Film35mmMedium,
    /// 35mm film stock - heavy grain.
    Film35mmHeavy,
    /// 16mm film stock - light grain.
    Film16mmLight,
    /// 16mm film stock - medium grain.
    Film16mmMedium,
    /// 16mm film stock - heavy grain.
    Film16mmHeavy,
    /// Kodak Vision3 500T (tungsten-balanced, ISO 500).
    KodakVision3_500T,
    /// Kodak Vision3 250D (daylight-balanced, ISO 250).
    KodakVision3_250D,
    /// Fuji Eterna 400T (tungsten-balanced, ISO 400).
    FujiEterna400T,
    /// Digital noise reduction artifacts - light.
    DigitalNrLight,
    /// Digital noise reduction artifacts - medium.
    DigitalNrMedium,
    /// Custom pattern.
    Custom,
}

impl GrainPreset {
    /// Convert preset to film grain parameters.
    #[must_use]
    pub fn to_params(self, bit_depth: u8) -> FilmGrainParams {
        match self {
            Self::None => FilmGrainParams::default(),
            Self::Film35mmLight => create_35mm_params(GrainIntensity::Light, bit_depth),
            Self::Film35mmMedium => create_35mm_params(GrainIntensity::Medium, bit_depth),
            Self::Film35mmHeavy => create_35mm_params(GrainIntensity::Heavy, bit_depth),
            Self::Film16mmLight => create_16mm_params(GrainIntensity::Light, bit_depth),
            Self::Film16mmMedium => create_16mm_params(GrainIntensity::Medium, bit_depth),
            Self::Film16mmHeavy => create_16mm_params(GrainIntensity::Heavy, bit_depth),
            Self::KodakVision3_500T => create_kodak_vision3_500t_params(bit_depth),
            Self::KodakVision3_250D => create_kodak_vision3_250d_params(bit_depth),
            Self::FujiEterna400T => create_fuji_eterna_400t_params(bit_depth),
            Self::DigitalNrLight => create_digital_nr_params(GrainIntensity::Light, bit_depth),
            Self::DigitalNrMedium => create_digital_nr_params(GrainIntensity::Medium, bit_depth),
            Self::Custom => FilmGrainParams::default(),
        }
    }

    /// Get description of preset.
    #[must_use]
    pub const fn description(self) -> &'static str {
        match self {
            Self::None => "No film grain",
            Self::Film35mmLight => "35mm film - light grain (subtle cinema look)",
            Self::Film35mmMedium => "35mm film - medium grain (classic cinema look)",
            Self::Film35mmHeavy => "35mm film - heavy grain (gritty cinema look)",
            Self::Film16mmLight => "16mm film - light grain (clean documentary look)",
            Self::Film16mmMedium => "16mm film - medium grain (indie film look)",
            Self::Film16mmHeavy => "16mm film - heavy grain (vintage documentary look)",
            Self::KodakVision3_500T => "Kodak Vision3 500T - professional cinema film (tungsten)",
            Self::KodakVision3_250D => "Kodak Vision3 250D - professional cinema film (daylight)",
            Self::FujiEterna400T => "Fuji Eterna 400T - professional cinema film (tungsten)",
            Self::DigitalNrLight => "Digital NR artifacts - light (subtle processing)",
            Self::DigitalNrMedium => "Digital NR artifacts - medium (noticeable processing)",
            Self::Custom => "Custom grain pattern",
        }
    }
}

// =============================================================================
// Preset Pattern Generators
// =============================================================================

/// Create 35mm film grain parameters.
///
/// 35mm film has fine, relatively uniform grain with good detail preservation.
fn create_35mm_params(intensity: GrainIntensity, _bit_depth: u8) -> FilmGrainParams {
    let mut params = FilmGrainParams::new();
    params.apply_grain = true;
    params.film_grain_params_present = true;
    params.grain_seed = 1234;
    params.ar_coeff_lag = 2;
    params.grain_scaling_minus_8 = 1;
    params.ar_coeff_shift_minus_6 = 1;
    params.grain_scale_shift = 0;
    params.overlap_flag = true;

    // AV1 scaling points are always 8-bit values regardless of video bit depth.
    // The decoder handles bit-depth conversion internally.

    // Luma scaling points
    match intensity {
        GrainIntensity::Light => {
            params.add_y_point(0, 12);
            params.add_y_point(64, 16);
            params.add_y_point(128, 20);
            params.add_y_point(192, 16);
            params.add_y_point(255, 12);
        }
        GrainIntensity::Medium => {
            params.add_y_point(0, 20);
            params.add_y_point(64, 28);
            params.add_y_point(128, 36);
            params.add_y_point(192, 28);
            params.add_y_point(255, 20);
        }
        GrainIntensity::Heavy => {
            params.add_y_point(0, 32);
            params.add_y_point(64, 44);
            params.add_y_point(128, 56);
            params.add_y_point(192, 44);
            params.add_y_point(255, 32);
        }
    }

    // Chroma scaling (less grain in chroma)
    params.chroma_scaling_from_luma = false;

    let chroma_scale = match intensity {
        GrainIntensity::Light => [8, 12, 8],
        GrainIntensity::Medium => [12, 18, 12],
        GrainIntensity::Heavy => [20, 28, 20],
    };

    params.add_cb_point(0, chroma_scale[0]);
    params.add_cb_point(128, chroma_scale[1]);
    params.add_cb_point(255, chroma_scale[2]);

    params.add_cr_point(0, chroma_scale[0]);
    params.add_cr_point(128, chroma_scale[1]);
    params.add_cr_point(255, chroma_scale[2]);

    // AR coefficients for 35mm (smooth, correlated grain)
    params.ar_coeffs_y[0] = 4;
    params.ar_coeffs_y[1] = 3;
    params.ar_coeffs_y[2] = 2;
    params.ar_coeffs_y[3] = 3;
    params.ar_coeffs_y[4] = 0;
    params.ar_coeffs_y[5] = 2;
    params.ar_coeffs_y[6] = 2;
    params.ar_coeffs_y[7] = 0;
    params.ar_coeffs_y[8] = 1;
    params.ar_coeffs_y[9] = 3;
    params.ar_coeffs_y[10] = 2;
    params.ar_coeffs_y[11] = 1;

    params.ar_coeffs_cb[0] = 3;
    params.ar_coeffs_cb[1] = 2;
    params.ar_coeffs_cb[2] = 1;
    params.ar_coeffs_cb[3] = 2;
    params.ar_coeffs_cb[4] = 0;
    params.ar_coeffs_cb[5] = 1;
    params.ar_coeffs_cb[6] = 4; // Luma correlation

    params.ar_coeffs_cr = params.ar_coeffs_cb;

    // Chroma combination parameters
    params.cb_mult = 128;
    params.cb_luma_mult = 192;
    params.cb_offset = 256;
    params.cr_mult = 128;
    params.cr_luma_mult = 192;
    params.cr_offset = 256;

    params
}

/// Create 16mm film grain parameters.
///
/// 16mm film has coarser, more visible grain than 35mm.
fn create_16mm_params(intensity: GrainIntensity, _bit_depth: u8) -> FilmGrainParams {
    let mut params = FilmGrainParams::new();
    params.apply_grain = true;
    params.film_grain_params_present = true;
    params.grain_seed = 5678;
    params.ar_coeff_lag = 1;
    params.grain_scaling_minus_8 = 1;
    params.ar_coeff_shift_minus_6 = 0;
    params.grain_scale_shift = 0;
    params.overlap_flag = false;

    // AV1 scaling points are always 8-bit values regardless of video bit depth.

    // Luma scaling points (more grain than 35mm)
    match intensity {
        GrainIntensity::Light => {
            params.add_y_point(0, 24);
            params.add_y_point(64, 32);
            params.add_y_point(128, 40);
            params.add_y_point(192, 32);
            params.add_y_point(255, 24);
        }
        GrainIntensity::Medium => {
            params.add_y_point(0, 40);
            params.add_y_point(64, 52);
            params.add_y_point(128, 64);
            params.add_y_point(192, 52);
            params.add_y_point(255, 40);
        }
        GrainIntensity::Heavy => {
            params.add_y_point(0, 56);
            params.add_y_point(64, 72);
            params.add_y_point(128, 88);
            params.add_y_point(192, 72);
            params.add_y_point(255, 56);
        }
    }

    // Chroma scaling
    params.chroma_scaling_from_luma = false;

    let chroma_scale = match intensity {
        GrainIntensity::Light => [16, 24, 16],
        GrainIntensity::Medium => [28, 36, 28],
        GrainIntensity::Heavy => [40, 52, 40],
    };

    params.add_cb_point(0, chroma_scale[0]);
    params.add_cb_point(128, chroma_scale[1]);
    params.add_cb_point(255, chroma_scale[2]);

    params.add_cr_point(0, chroma_scale[0]);
    params.add_cr_point(128, chroma_scale[1]);
    params.add_cr_point(255, chroma_scale[2]);

    // AR coefficients for 16mm (coarser grain)
    params.ar_coeffs_y[0] = 6;
    params.ar_coeffs_y[1] = 5;
    params.ar_coeffs_y[2] = 5;
    params.ar_coeffs_y[3] = 6;

    params.ar_coeffs_cb[0] = 5;
    params.ar_coeffs_cb[1] = 4;
    params.ar_coeffs_cb[2] = 4;
    params.ar_coeffs_cb[3] = 5;
    params.ar_coeffs_cb[4] = 6; // Luma correlation

    params.ar_coeffs_cr = params.ar_coeffs_cb;

    params.cb_mult = 128;
    params.cb_luma_mult = 160;
    params.cb_offset = 256;
    params.cr_mult = 128;
    params.cr_luma_mult = 160;
    params.cr_offset = 256;

    params
}

/// Create Kodak Vision3 500T film grain parameters.
///
/// Vision3 500T is a tungsten-balanced professional cinema film with ISO 500.
/// Known for its fine grain structure, excellent shadow detail, and warm color rendering.
fn create_kodak_vision3_500t_params(_bit_depth: u8) -> FilmGrainParams {
    let mut params = FilmGrainParams::new();
    params.apply_grain = true;
    params.film_grain_params_present = true;
    params.grain_seed = 5007;
    params.ar_coeff_lag = 2;
    params.grain_scaling_minus_8 = 1;
    params.ar_coeff_shift_minus_6 = 1;
    params.grain_scale_shift = 0;
    params.overlap_flag = true;

    // AV1 scaling points are always 8-bit values regardless of video bit depth.

    // Vision3 500T: Fine grain, slightly more visible in shadows
    params.add_y_point(0, 28);
    params.add_y_point(32, 26);
    params.add_y_point(64, 24);
    params.add_y_point(96, 22);
    params.add_y_point(128, 20);
    params.add_y_point(160, 18);
    params.add_y_point(192, 16);
    params.add_y_point(224, 14);
    params.add_y_point(255, 12);

    // Chroma grain with slight warmth bias
    params.chroma_scaling_from_luma = false;

    params.add_cb_point(0, 14);
    params.add_cb_point(64, 16);
    params.add_cb_point(128, 18);
    params.add_cb_point(192, 16);
    params.add_cb_point(255, 14);

    params.add_cr_point(0, 16);
    params.add_cr_point(64, 18);
    params.add_cr_point(128, 20);
    params.add_cr_point(192, 18);
    params.add_cr_point(255, 16);

    // AR coefficients for Vision3 (smooth, organic grain)
    params.ar_coeffs_y[0] = 5;
    params.ar_coeffs_y[1] = 4;
    params.ar_coeffs_y[2] = 3;
    params.ar_coeffs_y[3] = 4;
    params.ar_coeffs_y[4] = 2;
    params.ar_coeffs_y[5] = 3;
    params.ar_coeffs_y[6] = 3;
    params.ar_coeffs_y[7] = 1;
    params.ar_coeffs_y[8] = 2;
    params.ar_coeffs_y[9] = 4;
    params.ar_coeffs_y[10] = 3;
    params.ar_coeffs_y[11] = 2;

    params.ar_coeffs_cb[0] = 4;
    params.ar_coeffs_cb[1] = 3;
    params.ar_coeffs_cb[2] = 2;
    params.ar_coeffs_cb[3] = 3;
    params.ar_coeffs_cb[4] = 1;
    params.ar_coeffs_cb[5] = 2;
    params.ar_coeffs_cb[6] = 5; // Luma correlation

    params.ar_coeffs_cr = params.ar_coeffs_cb;

    // Tungsten color characteristics
    params.cb_mult = 120;
    params.cb_luma_mult = 200;
    params.cb_offset = 240;
    params.cr_mult = 136;
    params.cr_luma_mult = 188;
    params.cr_offset = 272;

    params
}

/// Create Kodak Vision3 250D film grain parameters.
///
/// Vision3 250D is a daylight-balanced professional cinema film with ISO 250.
/// Finer grain than 500T, excellent color saturation, and cooler color balance.
fn create_kodak_vision3_250d_params(_bit_depth: u8) -> FilmGrainParams {
    let mut params = FilmGrainParams::new();
    params.apply_grain = true;
    params.film_grain_params_present = true;
    params.grain_seed = 2507;
    params.ar_coeff_lag = 2;
    params.grain_scaling_minus_8 = 1;
    params.ar_coeff_shift_minus_6 = 1;
    params.grain_scale_shift = 0;
    params.overlap_flag = true;

    // AV1 scaling points are always 8-bit values regardless of video bit depth.

    // Vision3 250D: Very fine grain, more uniform across tonal range
    params.add_y_point(0, 18);
    params.add_y_point(32, 17);
    params.add_y_point(64, 16);
    params.add_y_point(96, 15);
    params.add_y_point(128, 14);
    params.add_y_point(160, 13);
    params.add_y_point(192, 12);
    params.add_y_point(224, 11);
    params.add_y_point(255, 10);

    // Chroma grain with daylight balance
    params.chroma_scaling_from_luma = false;

    params.add_cb_point(0, 10);
    params.add_cb_point(64, 11);
    params.add_cb_point(128, 12);
    params.add_cb_point(192, 11);
    params.add_cb_point(255, 10);

    params.add_cr_point(0, 9);
    params.add_cr_point(64, 10);
    params.add_cr_point(128, 11);
    params.add_cr_point(192, 10);
    params.add_cr_point(255, 9);

    // AR coefficients for Vision3 250D (very smooth, fine grain)
    params.ar_coeffs_y[0] = 4;
    params.ar_coeffs_y[1] = 3;
    params.ar_coeffs_y[2] = 2;
    params.ar_coeffs_y[3] = 3;
    params.ar_coeffs_y[4] = 1;
    params.ar_coeffs_y[5] = 2;
    params.ar_coeffs_y[6] = 2;
    params.ar_coeffs_y[7] = 1;
    params.ar_coeffs_y[8] = 1;
    params.ar_coeffs_y[9] = 3;
    params.ar_coeffs_y[10] = 2;
    params.ar_coeffs_y[11] = 1;

    params.ar_coeffs_cb[0] = 3;
    params.ar_coeffs_cb[1] = 2;
    params.ar_coeffs_cb[2] = 1;
    params.ar_coeffs_cb[3] = 2;
    params.ar_coeffs_cb[4] = 1;
    params.ar_coeffs_cb[5] = 1;
    params.ar_coeffs_cb[6] = 4; // Luma correlation

    params.ar_coeffs_cr = params.ar_coeffs_cb;

    // Daylight color characteristics
    params.cb_mult = 136;
    params.cb_luma_mult = 188;
    params.cb_offset = 272;
    params.cr_mult = 120;
    params.cr_luma_mult = 200;
    params.cr_offset = 240;

    params
}

/// Create Fuji Eterna 400T film grain parameters.
///
/// Eterna 400T is a tungsten-balanced professional cinema film with ISO 400.
/// Known for its muted color palette, excellent skin tones, and distinctive grain character.
fn create_fuji_eterna_400t_params(_bit_depth: u8) -> FilmGrainParams {
    let mut params = FilmGrainParams::new();
    params.apply_grain = true;
    params.film_grain_params_present = true;
    params.grain_seed = 4007;
    params.ar_coeff_lag = 2;
    params.grain_scaling_minus_8 = 1;
    params.ar_coeff_shift_minus_6 = 1;
    params.grain_scale_shift = 0;
    params.overlap_flag = true;

    // AV1 scaling points are always 8-bit values regardless of video bit depth.

    // Eterna 400T: Distinctive grain structure, slightly coarser in midtones
    params.add_y_point(0, 24);
    params.add_y_point(32, 22);
    params.add_y_point(64, 20);
    params.add_y_point(96, 22);
    params.add_y_point(128, 24);
    params.add_y_point(160, 22);
    params.add_y_point(192, 18);
    params.add_y_point(224, 15);
    params.add_y_point(255, 12);

    // Chroma grain - Eterna's muted color palette
    params.chroma_scaling_from_luma = false;

    params.add_cb_point(0, 12);
    params.add_cb_point(64, 14);
    params.add_cb_point(128, 16);
    params.add_cb_point(192, 14);
    params.add_cb_point(255, 12);

    params.add_cr_point(0, 14);
    params.add_cr_point(64, 16);
    params.add_cr_point(128, 18);
    params.add_cr_point(192, 16);
    params.add_cr_point(255, 14);

    // AR coefficients for Eterna (organic, slightly clumpy grain)
    params.ar_coeffs_y[0] = 6;
    params.ar_coeffs_y[1] = 4;
    params.ar_coeffs_y[2] = 3;
    params.ar_coeffs_y[3] = 5;
    params.ar_coeffs_y[4] = 2;
    params.ar_coeffs_y[5] = 3;
    params.ar_coeffs_y[6] = 4;
    params.ar_coeffs_y[7] = 2;
    params.ar_coeffs_y[8] = 2;
    params.ar_coeffs_y[9] = 5;
    params.ar_coeffs_y[10] = 3;
    params.ar_coeffs_y[11] = 2;

    params.ar_coeffs_cb[0] = 5;
    params.ar_coeffs_cb[1] = 3;
    params.ar_coeffs_cb[2] = 2;
    params.ar_coeffs_cb[3] = 4;
    params.ar_coeffs_cb[4] = 2;
    params.ar_coeffs_cb[5] = 2;
    params.ar_coeffs_cb[6] = 6; // Luma correlation

    params.ar_coeffs_cr[0] = 5;
    params.ar_coeffs_cr[1] = 4;
    params.ar_coeffs_cr[2] = 2;
    params.ar_coeffs_cr[3] = 4;
    params.ar_coeffs_cr[4] = 2;
    params.ar_coeffs_cr[5] = 3;
    params.ar_coeffs_cr[6] = 5; // Luma correlation

    // Eterna color characteristics (muted, filmic)
    params.cb_mult = 116;
    params.cb_luma_mult = 204;
    params.cb_offset = 232;
    params.cr_mult = 132;
    params.cr_luma_mult = 192;
    params.cr_offset = 264;

    params
}

/// Create digital noise reduction artifact parameters.
///
/// Simulates artifacts from aggressive digital noise reduction.
fn create_digital_nr_params(intensity: GrainIntensity, _bit_depth: u8) -> FilmGrainParams {
    let mut params = FilmGrainParams::new();
    params.apply_grain = true;
    params.film_grain_params_present = true;
    params.grain_seed = 9012;
    params.ar_coeff_lag = 3;
    params.grain_scaling_minus_8 = 2;
    params.ar_coeff_shift_minus_6 = 2;
    params.grain_scale_shift = 1;
    params.overlap_flag = true;

    // AV1 scaling points are always 8-bit values regardless of video bit depth.

    // NR artifacts: more grain in shadows, less in highlights
    match intensity {
        GrainIntensity::Light => {
            params.add_y_point(0, 16);
            params.add_y_point(32, 12);
            params.add_y_point(96, 8);
            params.add_y_point(160, 4);
            params.add_y_point(224, 2);
            params.add_y_point(255, 1);
        }
        GrainIntensity::Medium => {
            params.add_y_point(0, 32);
            params.add_y_point(32, 24);
            params.add_y_point(96, 16);
            params.add_y_point(160, 8);
            params.add_y_point(224, 4);
            params.add_y_point(255, 2);
        }
        GrainIntensity::Heavy => {
            params.add_y_point(0, 48);
            params.add_y_point(32, 36);
            params.add_y_point(96, 24);
            params.add_y_point(160, 12);
            params.add_y_point(224, 6);
            params.add_y_point(255, 3);
        }
    }

    // Chroma artifacts
    params.chroma_scaling_from_luma = true;

    // AR coefficients for NR artifacts (blocky, less random)
    params.ar_coeffs_y[0] = 8;
    params.ar_coeffs_y[1] = 6;
    params.ar_coeffs_y[2] = 4;
    params.ar_coeffs_y[3] = 6;
    params.ar_coeffs_y[4] = 4;
    params.ar_coeffs_y[5] = 2;
    params.ar_coeffs_y[6] = 4;
    params.ar_coeffs_y[7] = 2;
    params.ar_coeffs_y[8] = 0;
    params.ar_coeffs_y[9] = 6;
    params.ar_coeffs_y[10] = 4;
    params.ar_coeffs_y[11] = 2;

    params.cb_mult = 128;
    params.cb_luma_mult = 224;
    params.cb_offset = 256;
    params.cr_mult = 128;
    params.cr_luma_mult = 224;
    params.cr_offset = 256;

    params
}

// =============================================================================
// Grain Pattern Builder
// =============================================================================

/// Builder for custom grain patterns.
#[derive(Clone, Debug)]
pub struct GrainPatternBuilder {
    params: FilmGrainParams,
}

impl GrainPatternBuilder {
    /// Create a new builder with default parameters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            params: FilmGrainParams::new(),
        }
    }

    /// Enable grain synthesis.
    pub fn enable(mut self) -> Self {
        self.params.apply_grain = true;
        self.params.film_grain_params_present = true;
        self
    }

    /// Set grain seed.
    pub fn seed(mut self, seed: u16) -> Self {
        self.params.grain_seed = seed;
        self
    }

    /// Set AR coefficient lag (0-3).
    pub fn ar_lag(mut self, lag: u8) -> Self {
        self.params.ar_coeff_lag = lag.min(3);
        self
    }

    /// Add luma scaling point.
    pub fn add_luma_point(mut self, value: u8, scaling: u8) -> Self {
        self.params.add_y_point(value, scaling);
        self
    }

    /// Add chroma scaling point (both Cb and Cr).
    pub fn add_chroma_point(mut self, value: u8, scaling: u8) -> Self {
        self.params.add_cb_point(value, scaling);
        self.params.add_cr_point(value, scaling);
        self
    }

    /// Enable chroma scaling from luma.
    pub fn chroma_from_luma(mut self, enable: bool) -> Self {
        self.params.chroma_scaling_from_luma = enable;
        self
    }

    /// Set overlap flag.
    pub fn overlap(mut self, enable: bool) -> Self {
        self.params.overlap_flag = enable;
        self
    }

    /// Set grain scaling shift.
    pub fn grain_scale_shift(mut self, shift: u8) -> Self {
        self.params.grain_scale_shift = shift.min(3);
        self
    }

    /// Build the final parameters.
    #[must_use]
    pub fn build(self) -> FilmGrainParams {
        self.params
    }
}

impl Default for GrainPatternBuilder {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_film_grain_table_creation() {
        let table = FilmGrainTable::new();
        assert!(table.is_empty());
        assert_eq!(table.len(), 0);
    }

    #[test]
    fn test_film_grain_table_insert_get() {
        let mut table = FilmGrainTable::new();
        let mut params = FilmGrainParams::new();
        params.grain_seed = 1234;

        table.insert(0, params.clone());
        assert_eq!(table.len(), 1);

        let retrieved = table.get(0).expect("get should return value");
        assert_eq!(retrieved.grain_seed, 1234);
    }

    #[test]
    fn test_film_grain_table_get_nearest() {
        let mut table = FilmGrainTable::new();

        let mut params0 = FilmGrainParams::new();
        params0.grain_seed = 100;
        table.insert(0, params0);

        let mut params10 = FilmGrainParams::new();
        params10.grain_seed = 200;
        table.insert(10, params10);

        // Frame 5 should get params from frame 0
        let params5 = table.get(5).expect("get should return value");
        assert_eq!(params5.grain_seed, 100);

        // Frame 15 should get params from frame 10
        let params15 = table.get(15).expect("get should return value");
        assert_eq!(params15.grain_seed, 200);
    }

    #[test]
    fn test_film_grain_table_interpolation() {
        let mut table = FilmGrainTable::new();

        let mut params0 = FilmGrainParams::new();
        params0.grain_seed = 100;
        params0.cb_mult = 64;
        table.insert(0, params0);

        let mut params10 = FilmGrainParams::new();
        params10.grain_seed = 200;
        params10.cb_mult = 192;
        table.insert(10, params10);

        let params5 = table.get_interpolated(5);
        assert!(params5.grain_seed >= 100 && params5.grain_seed <= 200);
        assert!(params5.cb_mult >= 64 && params5.cb_mult <= 192);
    }

    #[test]
    fn test_film_grain_table_remove() {
        let mut table = FilmGrainTable::new();
        let params = FilmGrainParams::new();

        table.insert(0, params);
        assert_eq!(table.len(), 1);

        table.remove(0);
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn test_film_grain_table_clear() {
        let mut table = FilmGrainTable::new();
        table.insert(0, FilmGrainParams::new());
        table.insert(10, FilmGrainParams::new());
        assert_eq!(table.len(), 2);

        table.clear();
        assert_eq!(table.len(), 0);
        assert!(table.is_empty());
    }

    #[test]
    fn test_grain_presets() {
        let presets = [
            GrainPreset::Film35mmLight,
            GrainPreset::Film35mmMedium,
            GrainPreset::Film35mmHeavy,
            GrainPreset::Film16mmLight,
            GrainPreset::Film16mmMedium,
            GrainPreset::Film16mmHeavy,
            GrainPreset::KodakVision3_500T,
            GrainPreset::KodakVision3_250D,
            GrainPreset::FujiEterna400T,
            GrainPreset::DigitalNrLight,
            GrainPreset::DigitalNrMedium,
        ];

        for preset in &presets {
            let params = preset.to_params(8);
            assert!(params.validate());
            assert!(!preset.description().is_empty());
        }
    }

    #[test]
    fn test_35mm_light() {
        let params = GrainPreset::Film35mmLight.to_params(8);
        assert!(params.apply_grain);
        assert!(params.film_grain_params_present);
        assert!(params.num_y_points > 0);
        assert!(params.validate());
    }

    #[test]
    fn test_35mm_medium() {
        let params = GrainPreset::Film35mmMedium.to_params(8);
        assert!(params.num_y_points > 0);
        assert!(params.overlap_flag);
        assert!(params.validate());
    }

    #[test]
    fn test_16mm_heavy() {
        let params = GrainPreset::Film16mmHeavy.to_params(8);
        assert!(params.num_y_points > 0);
        assert!(!params.overlap_flag); // 16mm doesn't use overlap
        assert!(params.validate());
    }

    #[test]
    fn test_digital_nr() {
        let params = GrainPreset::DigitalNrMedium.to_params(8);
        assert!(params.num_y_points > 0);
        assert!(params.chroma_scaling_from_luma);
        assert!(params.validate());
    }

    #[test]
    fn test_grain_pattern_builder() {
        let params = GrainPatternBuilder::new()
            .enable()
            .seed(12345)
            .ar_lag(2)
            .add_luma_point(0, 32)
            .add_luma_point(128, 48)
            .add_luma_point(255, 32)
            .add_chroma_point(0, 16)
            .add_chroma_point(255, 16)
            .chroma_from_luma(false)
            .overlap(true)
            .grain_scale_shift(0)
            .build();

        assert!(params.apply_grain);
        assert_eq!(params.grain_seed, 12345);
        assert_eq!(params.ar_coeff_lag, 2);
        assert_eq!(params.num_y_points, 3);
        assert!(params.overlap_flag);
        assert!(params.validate());
    }

    #[test]
    fn test_lerp_u8() {
        assert_eq!(lerp_u8(0, 255, 0), 0);
        assert_eq!(lerp_u8(0, 255, 255), 255);
        let mid = lerp_u8(0, 255, 128);
        assert!(mid >= 127 && mid <= 128);
    }

    #[test]
    fn test_lerp_u16() {
        assert_eq!(lerp_u16(0, 1000, 0.0), 0);
        assert_eq!(lerp_u16(0, 1000, 1.0), 1000);
        let mid = lerp_u16(0, 1000, 0.5);
        assert!(mid >= 499 && mid <= 501);
    }

    #[test]
    fn test_interpolate_params() {
        let mut p0 = FilmGrainParams::new();
        p0.grain_seed = 100;
        p0.cb_mult = 64;

        let mut p1 = FilmGrainParams::new();
        p1.grain_seed = 200;
        p1.cb_mult = 192;

        let mid = interpolate_params(&p0, &p1, 0.5);
        assert!(mid.grain_seed >= 100 && mid.grain_seed <= 200);
        assert!(mid.cb_mult >= 64 && mid.cb_mult <= 192);
    }

    #[test]
    fn test_bit_depth_scaling() {
        let params_8 = GrainPreset::Film35mmMedium.to_params(8);
        let params_10 = GrainPreset::Film35mmMedium.to_params(10);

        // 10-bit should have scaled values
        assert!(params_8.validate());
        assert!(params_10.validate());
    }

    #[test]
    fn test_table_with_default() {
        let mut default_params = FilmGrainParams::new();
        default_params.grain_seed = 9999;

        let table = FilmGrainTable::with_default(default_params);
        let params = table.get(100).expect("get should return value");
        assert_eq!(params.grain_seed, 9999);
    }

    #[test]
    fn test_table_iteration() {
        let mut table = FilmGrainTable::new();
        table.insert(0, FilmGrainParams::new());
        table.insert(10, FilmGrainParams::new());
        table.insert(20, FilmGrainParams::new());

        let count = table.iter().count();
        assert_eq!(count, 3);
    }

    #[test]
    fn test_kodak_vision3_500t() {
        let params = GrainPreset::KodakVision3_500T.to_params(8);
        assert!(params.apply_grain);
        assert!(params.film_grain_params_present);
        assert_eq!(params.grain_seed, 5007);
        assert!(params.num_y_points > 0);
        assert!(params.num_cb_points > 0);
        assert!(params.num_cr_points > 0);
        assert!(params.overlap_flag);
        assert!(params.validate());
    }

    #[test]
    fn test_kodak_vision3_250d() {
        let params = GrainPreset::KodakVision3_250D.to_params(8);
        assert!(params.apply_grain);
        assert!(params.film_grain_params_present);
        assert_eq!(params.grain_seed, 2507);
        assert!(params.num_y_points > 0);
        assert!(params.num_cb_points > 0);
        assert!(params.num_cr_points > 0);
        assert!(params.overlap_flag);
        assert!(params.validate());
    }

    #[test]
    fn test_fuji_eterna_400t() {
        let params = GrainPreset::FujiEterna400T.to_params(8);
        assert!(params.apply_grain);
        assert!(params.film_grain_params_present);
        assert_eq!(params.grain_seed, 4007);
        assert!(params.num_y_points > 0);
        assert!(params.num_cb_points > 0);
        assert!(params.num_cr_points > 0);
        assert!(params.overlap_flag);
        assert!(params.validate());
    }

    #[test]
    fn test_film_stock_grain_characteristics() {
        // Vision3 250D should have finer grain than 500T
        let v250d = GrainPreset::KodakVision3_250D.to_params(8);
        let v500t = GrainPreset::KodakVision3_500T.to_params(8);

        // 250D should have lower grain values (finer grain)
        assert!(v250d.y_points[0].scaling < v500t.y_points[0].scaling);

        // Eterna should have distinctive grain in midtones
        let eterna = GrainPreset::FujiEterna400T.to_params(8);
        assert!(eterna.num_y_points >= 5);
    }

    #[test]
    fn test_film_stock_bit_depth_scaling() {
        // Test that film stocks work with different bit depths
        for bit_depth in [8, 10, 12] {
            let v500t = GrainPreset::KodakVision3_500T.to_params(bit_depth);
            let v250d = GrainPreset::KodakVision3_250D.to_params(bit_depth);
            let eterna = GrainPreset::FujiEterna400T.to_params(bit_depth);

            assert!(v500t.validate());
            assert!(v250d.validate());
            assert!(eterna.validate());
        }
    }
}
