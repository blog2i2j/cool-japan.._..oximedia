//! Film grain synthesis for AV1.
//!
//! Film grain synthesis adds realistic grain patterns to decoded video,
//! allowing encoders to remove grain before encoding and synthesize it
//! during playback for better compression efficiency.

#![forbid(unsafe_code)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::identity_op)]
#![allow(clippy::range_plus_one)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::useless_conversion)]
#![allow(clippy::redundant_closure_for_method_calls)]
#![allow(clippy::single_match_else)]
#![allow(dead_code)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::unused_self)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]

use super::pipeline::FrameContext;
use super::{FrameBuffer, PlaneBuffer, PlaneType, ReconstructResult};

// =============================================================================
// Constants
// =============================================================================

/// Maximum number of AR (auto-regression) coefficients for luma.
pub const MAX_AR_COEFFS_Y: usize = 24;

/// Maximum number of AR coefficients for chroma.
pub const MAX_AR_COEFFS_UV: usize = 25;

/// Maximum AR lag.
pub const MAX_AR_LAG: usize = 3;

/// Grain block size.
pub const GRAIN_BLOCK_SIZE: usize = 32;

/// Maximum number of luma scaling points.
pub const MAX_LUMA_POINTS: usize = 14;

/// Maximum number of chroma scaling points.
pub const MAX_CHROMA_POINTS: usize = 10;

/// Grain seed base.
pub const GRAIN_SEED: u16 = 0xB524;

/// LUT size for grain values.
pub const GRAIN_LUT_SIZE: usize = 82;

// =============================================================================
// Film Grain Parameters
// =============================================================================

/// Scaling point for grain intensity.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScalingPoint {
    /// Input value (0-255 for 8-bit).
    pub value: u8,
    /// Scaling factor.
    pub scaling: u8,
}

impl ScalingPoint {
    /// Create a new scaling point.
    #[must_use]
    pub const fn new(value: u8, scaling: u8) -> Self {
        Self { value, scaling }
    }
}

/// Film grain parameters.
#[derive(Clone, Debug)]
pub struct FilmGrainParams {
    /// Apply grain to this frame.
    pub apply_grain: bool,
    /// Random seed for grain generation.
    pub grain_seed: u16,
    /// Update grain parameters.
    pub update_grain: bool,
    /// Number of Y scaling points.
    pub num_y_points: usize,
    /// Y scaling points.
    pub y_points: [ScalingPoint; MAX_LUMA_POINTS],
    /// Chroma scaling from luma.
    pub chroma_scaling_from_luma: bool,
    /// Number of Cb scaling points.
    pub num_cb_points: usize,
    /// Cb scaling points.
    pub cb_points: [ScalingPoint; MAX_CHROMA_POINTS],
    /// Number of Cr scaling points.
    pub num_cr_points: usize,
    /// Cr scaling points.
    pub cr_points: [ScalingPoint; MAX_CHROMA_POINTS],
    /// Grain scaling shift (8-11).
    pub grain_scaling_minus_8: u8,
    /// AR coefficients lag (0-3).
    pub ar_coeff_lag: u8,
    /// AR coefficients for Y.
    pub ar_coeffs_y: [i8; MAX_AR_COEFFS_Y],
    /// AR coefficients for Cb.
    pub ar_coeffs_cb: [i8; MAX_AR_COEFFS_UV],
    /// AR coefficients for Cr.
    pub ar_coeffs_cr: [i8; MAX_AR_COEFFS_UV],
    /// AR coefficient shift (6-9).
    pub ar_coeff_shift_minus_6: u8,
    /// Grain scale shift.
    pub grain_scale_shift: u8,
    /// Cb multiplier.
    pub cb_mult: u8,
    /// Cb luma multiplier.
    pub cb_luma_mult: u8,
    /// Cb offset.
    pub cb_offset: u16,
    /// Cr multiplier.
    pub cr_mult: u8,
    /// Cr luma multiplier.
    pub cr_luma_mult: u8,
    /// Cr offset.
    pub cr_offset: u16,
    /// Overlap flag.
    pub overlap_flag: bool,
    /// Clip to restricted range.
    pub clip_to_restricted_range: bool,
}

impl Default for FilmGrainParams {
    fn default() -> Self {
        Self {
            apply_grain: false,
            grain_seed: 0,
            update_grain: false,
            num_y_points: 0,
            y_points: [ScalingPoint::default(); MAX_LUMA_POINTS],
            chroma_scaling_from_luma: false,
            num_cb_points: 0,
            cb_points: [ScalingPoint::default(); MAX_CHROMA_POINTS],
            num_cr_points: 0,
            cr_points: [ScalingPoint::default(); MAX_CHROMA_POINTS],
            grain_scaling_minus_8: 0,
            ar_coeff_lag: 0,
            ar_coeffs_y: [0; MAX_AR_COEFFS_Y],
            ar_coeffs_cb: [0; MAX_AR_COEFFS_UV],
            ar_coeffs_cr: [0; MAX_AR_COEFFS_UV],
            ar_coeff_shift_minus_6: 0,
            grain_scale_shift: 0,
            cb_mult: 0,
            cb_luma_mult: 0,
            cb_offset: 0,
            cr_mult: 0,
            cr_luma_mult: 0,
            cr_offset: 0,
            overlap_flag: false,
            clip_to_restricted_range: false,
        }
    }
}

impl FilmGrainParams {
    /// Create new film grain parameters.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if grain should be applied.
    #[must_use]
    pub const fn is_enabled(&self) -> bool {
        self.apply_grain && (self.num_y_points > 0 || self.chroma_scaling_from_luma)
    }

    /// Get grain scaling value.
    #[must_use]
    pub const fn grain_scaling(&self) -> u8 {
        self.grain_scaling_minus_8 + 8
    }

    /// Get AR coefficient shift.
    #[must_use]
    pub const fn ar_coeff_shift(&self) -> u8 {
        self.ar_coeff_shift_minus_6 + 6
    }

    /// Get number of AR coefficients for Y.
    #[must_use]
    pub fn num_ar_coeffs_y(&self) -> usize {
        let lag = self.ar_coeff_lag as usize;
        2 * lag * (lag + 1)
    }

    /// Get number of AR coefficients for chroma.
    #[must_use]
    pub fn num_ar_coeffs_uv(&self) -> usize {
        let lag = self.ar_coeff_lag as usize;
        2 * lag * (lag + 1) + 1
    }

    /// Add a Y scaling point.
    pub fn add_y_point(&mut self, value: u8, scaling: u8) {
        if self.num_y_points < MAX_LUMA_POINTS {
            self.y_points[self.num_y_points] = ScalingPoint::new(value, scaling);
            self.num_y_points += 1;
        }
    }

    /// Add a Cb scaling point.
    pub fn add_cb_point(&mut self, value: u8, scaling: u8) {
        if self.num_cb_points < MAX_CHROMA_POINTS {
            self.cb_points[self.num_cb_points] = ScalingPoint::new(value, scaling);
            self.num_cb_points += 1;
        }
    }

    /// Add a Cr scaling point.
    pub fn add_cr_point(&mut self, value: u8, scaling: u8) {
        if self.num_cr_points < MAX_CHROMA_POINTS {
            self.cr_points[self.num_cr_points] = ScalingPoint::new(value, scaling);
            self.num_cr_points += 1;
        }
    }
}

// =============================================================================
// Grain Block
// =============================================================================

/// Grain pattern for a block.
#[derive(Clone, Debug)]
pub struct GrainBlock {
    /// Grain values.
    values: Vec<i16>,
    /// Block width.
    width: usize,
    /// Block height.
    height: usize,
}

impl GrainBlock {
    /// Create a new grain block.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            values: vec![0; width * height],
            width,
            height,
        }
    }

    /// Get grain value at position.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> i16 {
        if x < self.width && y < self.height {
            self.values[y * self.width + x]
        } else {
            0
        }
    }

    /// Set grain value at position.
    pub fn set(&mut self, x: usize, y: usize, value: i16) {
        if x < self.width && y < self.height {
            self.values[y * self.width + x] = value;
        }
    }

    /// Get values slice.
    #[must_use]
    pub fn values(&self) -> &[i16] {
        &self.values
    }

    /// Get mutable values slice.
    pub fn values_mut(&mut self) -> &mut [i16] {
        &mut self.values
    }
}

// =============================================================================
// Pseudo-Random Number Generator
// =============================================================================

/// Linear feedback shift register for grain generation.
struct GrainRng {
    state: u16,
}

impl GrainRng {
    /// Create a new grain RNG.
    fn new(seed: u16) -> Self {
        Self { state: seed }
    }

    /// Generate next random value.
    fn next(&mut self) -> i16 {
        // LFSR with taps at bits 0, 1, 3, 12
        let bit =
            ((self.state >> 0) ^ (self.state >> 1) ^ (self.state >> 3) ^ (self.state >> 12)) & 1;
        self.state = (self.state >> 1) | (bit << 15);
        (self.state as i16) >> 5 // Return signed 11-bit value
    }

    /// Generate a Gaussian-distributed value.
    fn gaussian(&mut self) -> i16 {
        // Approximate Gaussian using sum of uniform values
        let mut sum: i32 = 0;
        for _ in 0..4 {
            sum += i32::from(self.next());
        }
        (sum / 4) as i16
    }
}

// =============================================================================
// Film Grain Synthesizer
// =============================================================================

/// Film grain synthesizer.
#[derive(Debug)]
pub struct FilmGrainSynthesizer {
    /// Current parameters.
    params: FilmGrainParams,
    /// Bit depth.
    bit_depth: u8,
    /// Luma grain LUT.
    luma_grain: Vec<i16>,
    /// Cb grain LUT.
    cb_grain: Vec<i16>,
    /// Cr grain LUT.
    cr_grain: Vec<i16>,
    /// Luma scaling LUT.
    luma_scaling: Vec<u8>,
    /// Cb scaling LUT.
    cb_scaling: Vec<u8>,
    /// Cr scaling LUT.
    cr_scaling: Vec<u8>,
}

impl FilmGrainSynthesizer {
    /// Create a new film grain synthesizer.
    #[must_use]
    pub fn new(bit_depth: u8) -> Self {
        let lut_size = GRAIN_LUT_SIZE * GRAIN_LUT_SIZE;
        let scaling_size = 1 << bit_depth.min(8);

        Self {
            params: FilmGrainParams::default(),
            bit_depth,
            luma_grain: vec![0; lut_size],
            cb_grain: vec![0; lut_size],
            cr_grain: vec![0; lut_size],
            luma_scaling: vec![0; scaling_size],
            cb_scaling: vec![0; scaling_size],
            cr_scaling: vec![0; scaling_size],
        }
    }

    /// Set film grain parameters.
    pub fn set_params(&mut self, params: FilmGrainParams) {
        self.params = params;
        if self.params.is_enabled() {
            self.generate_grain_luts();
            self.generate_scaling_luts();
        }
    }

    /// Get current parameters.
    #[must_use]
    pub fn params(&self) -> &FilmGrainParams {
        &self.params
    }

    /// Generate grain lookup tables.
    fn generate_grain_luts(&mut self) {
        let mut rng = GrainRng::new(self.params.grain_seed ^ GRAIN_SEED);

        // Generate luma grain
        for val in &mut self.luma_grain {
            *val = rng.gaussian();
        }

        // Generate Cb grain
        for val in &mut self.cb_grain {
            *val = rng.gaussian();
        }

        // Generate Cr grain
        for val in &mut self.cr_grain {
            *val = rng.gaussian();
        }

        // Apply AR filtering if coefficients are present
        if self.params.ar_coeff_lag > 0 {
            self.apply_ar_filter();
        }
    }

    /// Apply auto-regressive filter to grain.
    fn apply_ar_filter(&mut self) {
        let lag = self.params.ar_coeff_lag as usize;
        let shift = self.params.ar_coeff_shift();

        // Apply AR filter to luma grain
        for y in lag..GRAIN_LUT_SIZE {
            for x in lag..(GRAIN_LUT_SIZE - lag) {
                let mut sum: i32 = 0;
                let mut coeff_idx = 0;

                for dy in 0..=lag {
                    for dx in 0..(2 * lag + 1) {
                        if dy == 0 && dx >= lag {
                            break;
                        }
                        let coeff = i32::from(self.params.ar_coeffs_y[coeff_idx]);
                        let grain_idx = (y - lag + dy) * GRAIN_LUT_SIZE + (x - lag + dx);
                        sum += coeff * i32::from(self.luma_grain[grain_idx]);
                        coeff_idx += 1;
                    }
                }

                let idx = y * GRAIN_LUT_SIZE + x;
                self.luma_grain[idx] = (i32::from(self.luma_grain[idx]) + (sum >> shift)) as i16;
            }
        }
    }

    /// Generate scaling lookup tables.
    fn generate_scaling_luts(&mut self) {
        // Copy points to avoid borrow issues
        let y_points: Vec<_> = self.params.y_points[..self.params.num_y_points].to_vec();
        let cb_points: Vec<_> = self.params.cb_points[..self.params.num_cb_points].to_vec();
        let cr_points: Vec<_> = self.params.cr_points[..self.params.num_cr_points].to_vec();
        let chroma_from_luma = self.params.chroma_scaling_from_luma;

        // Generate luma scaling LUT
        interpolate_scaling_points(&y_points, &mut self.luma_scaling);

        // Generate Cb scaling LUT
        if chroma_from_luma {
            self.cb_scaling.copy_from_slice(&self.luma_scaling);
        } else {
            interpolate_scaling_points(&cb_points, &mut self.cb_scaling);
        }

        // Generate Cr scaling LUT
        if chroma_from_luma {
            self.cr_scaling.copy_from_slice(&self.luma_scaling);
        } else {
            interpolate_scaling_points(&cr_points, &mut self.cr_scaling);
        }
    }

    /// Apply film grain to a frame.
    ///
    /// # Errors
    ///
    /// Returns error if grain application fails.
    pub fn apply(
        &mut self,
        frame: &mut FrameBuffer,
        _context: &FrameContext,
    ) -> ReconstructResult<()> {
        if !self.params.is_enabled() {
            return Ok(());
        }

        let bd = frame.bit_depth();

        // Apply to Y plane
        self.apply_to_plane(frame.y_plane_mut(), PlaneType::Y, bd);

        // Apply to chroma planes
        if let Some(u) = frame.u_plane_mut() {
            self.apply_to_plane(u, PlaneType::U, bd);
        }
        if let Some(v) = frame.v_plane_mut() {
            self.apply_to_plane(v, PlaneType::V, bd);
        }

        Ok(())
    }

    /// Apply grain to a single plane.
    fn apply_to_plane(&self, plane: &mut PlaneBuffer, plane_type: PlaneType, bd: u8) {
        let width = plane.width() as usize;
        let height = plane.height() as usize;
        let max_val = (1i32 << bd) - 1;

        let (grain_lut, scaling_lut) = match plane_type {
            PlaneType::Y => (&self.luma_grain, &self.luma_scaling),
            PlaneType::U => (&self.cb_grain, &self.cb_scaling),
            PlaneType::V => (&self.cr_grain, &self.cr_scaling),
        };

        let grain_scale = self.params.grain_scaling();
        let grain_shift = self.params.grain_scale_shift;

        // Process each pixel
        for y in 0..height {
            for x in 0..width {
                let pixel = plane.get(x as u32, y as u32);

                // Get scaling factor
                let scaling_idx = (pixel as usize).min(scaling_lut.len() - 1);
                let scaling = i32::from(scaling_lut[scaling_idx]);

                // Get grain value
                let grain_x = x % GRAIN_LUT_SIZE;
                let grain_y = y % GRAIN_LUT_SIZE;
                let grain_idx = grain_y * GRAIN_LUT_SIZE + grain_x;
                let grain = i32::from(grain_lut[grain_idx]);

                // Apply grain
                let scaled_grain = (grain * scaling) >> grain_scale;
                let adjusted_grain = scaled_grain >> grain_shift;
                let result = (i32::from(pixel) + adjusted_grain).clamp(0, max_val);

                plane.set(x as u32, y as u32, result as i16);
            }
        }
    }

    /// Generate a grain block for specific position.
    #[must_use]
    pub fn generate_block(&self, x: usize, y: usize, plane: PlaneType) -> GrainBlock {
        let mut block = GrainBlock::new(GRAIN_BLOCK_SIZE, GRAIN_BLOCK_SIZE);

        let grain_lut = match plane {
            PlaneType::Y => &self.luma_grain,
            PlaneType::U => &self.cb_grain,
            PlaneType::V => &self.cr_grain,
        };

        for by in 0..GRAIN_BLOCK_SIZE {
            for bx in 0..GRAIN_BLOCK_SIZE {
                let grain_x = (x + bx) % GRAIN_LUT_SIZE;
                let grain_y = (y + by) % GRAIN_LUT_SIZE;
                let grain_idx = grain_y * GRAIN_LUT_SIZE + grain_x;
                block.set(bx, by, grain_lut[grain_idx]);
            }
        }

        block
    }
}

/// Interpolate scaling points to create LUT.
fn interpolate_scaling_points(points: &[ScalingPoint], lut: &mut [u8]) {
    if points.is_empty() {
        lut.fill(0);
        return;
    }

    let lut_size = lut.len();

    // Fill before first point
    let first_scaling = points[0].scaling;
    for val in lut.iter_mut().take(points[0].value as usize) {
        *val = first_scaling;
    }

    // Interpolate between points
    for i in 0..points.len().saturating_sub(1) {
        let p0 = &points[i];
        let p1 = &points[i + 1];

        for x in p0.value as usize..=p1.value as usize {
            if x < lut_size {
                let t = (x - p0.value as usize) as f32 / (p1.value - p0.value).max(1) as f32;
                lut[x] = ((1.0 - t) * p0.scaling as f32 + t * p1.scaling as f32).round() as u8;
            }
        }
    }

    // Fill after last point
    let last_point = points.last().expect("points is non-empty by construction");
    for val in lut.iter_mut().skip(last_point.value as usize + 1) {
        *val = last_point.scaling;
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reconstruct::ChromaSubsampling;

    #[test]
    fn test_scaling_point() {
        let point = ScalingPoint::new(128, 64);
        assert_eq!(point.value, 128);
        assert_eq!(point.scaling, 64);
    }

    #[test]
    fn test_film_grain_params_default() {
        let params = FilmGrainParams::default();
        assert!(!params.apply_grain);
        assert!(!params.is_enabled());
        assert_eq!(params.num_y_points, 0);
    }

    #[test]
    fn test_film_grain_params_enabled() {
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.add_y_point(0, 32);
        params.add_y_point(255, 64);

        assert!(params.is_enabled());
        assert_eq!(params.num_y_points, 2);
    }

    #[test]
    fn test_film_grain_params_scaling_values() {
        let mut params = FilmGrainParams::new();
        params.grain_scaling_minus_8 = 2;
        params.ar_coeff_shift_minus_6 = 3;

        assert_eq!(params.grain_scaling(), 10);
        assert_eq!(params.ar_coeff_shift(), 9);
    }

    #[test]
    fn test_film_grain_params_ar_coeffs() {
        let mut params = FilmGrainParams::new();

        params.ar_coeff_lag = 0;
        assert_eq!(params.num_ar_coeffs_y(), 0);

        params.ar_coeff_lag = 1;
        assert_eq!(params.num_ar_coeffs_y(), 4);

        params.ar_coeff_lag = 2;
        assert_eq!(params.num_ar_coeffs_y(), 12);

        params.ar_coeff_lag = 3;
        assert_eq!(params.num_ar_coeffs_y(), 24);
    }

    #[test]
    fn test_grain_block() {
        let mut block = GrainBlock::new(32, 32);
        block.set(10, 20, 100);
        assert_eq!(block.get(10, 20), 100);
        assert_eq!(block.get(0, 0), 0);
    }

    #[test]
    fn test_grain_rng() {
        let mut rng = GrainRng::new(12345);

        // Generate some values and check they're in range
        for _ in 0..100 {
            let val = rng.next();
            assert!(val >= -2048 && val < 2048);
        }
    }

    #[test]
    fn test_grain_rng_gaussian() {
        let mut rng = GrainRng::new(12345);

        // Gaussian values should be centered around 0
        let mut sum: i32 = 0;
        for _ in 0..1000 {
            sum += i32::from(rng.gaussian());
        }
        let mean = sum / 1000;
        assert!(mean.abs() < 100);
    }

    #[test]
    fn test_film_grain_synthesizer_creation() {
        let synth = FilmGrainSynthesizer::new(8);
        assert_eq!(synth.bit_depth, 8);
        assert!(!synth.params().is_enabled());
    }

    #[test]
    fn test_film_grain_synthesizer_set_params() {
        let mut synth = FilmGrainSynthesizer::new(8);

        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.grain_seed = 12345;
        params.add_y_point(0, 32);
        params.add_y_point(255, 64);

        synth.set_params(params);
        assert!(synth.params().is_enabled());
    }

    #[test]
    fn test_film_grain_apply_disabled() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);
        let context = FrameContext::new(64, 64);

        let mut synth = FilmGrainSynthesizer::new(8);
        let result = synth.apply(&mut frame, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_film_grain_apply_enabled() {
        let mut frame = FrameBuffer::new(64, 64, 8, ChromaSubsampling::Cs420);

        // Set some initial values
        for y in 0..64 {
            for x in 0..64 {
                frame.y_plane_mut().set(x, y, 128);
            }
        }

        let context = FrameContext::new(64, 64);

        let mut synth = FilmGrainSynthesizer::new(8);
        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.grain_seed = 12345;
        params.add_y_point(0, 16);
        params.add_y_point(255, 32);
        synth.set_params(params);

        let result = synth.apply(&mut frame, &context);
        assert!(result.is_ok());
    }

    #[test]
    fn test_generate_block() {
        let mut synth = FilmGrainSynthesizer::new(8);

        let mut params = FilmGrainParams::new();
        params.apply_grain = true;
        params.grain_seed = 12345;
        params.add_y_point(0, 32);
        params.add_y_point(255, 64);
        synth.set_params(params);

        let block = synth.generate_block(0, 0, PlaneType::Y);
        assert_eq!(block.width, GRAIN_BLOCK_SIZE);
        assert_eq!(block.height, GRAIN_BLOCK_SIZE);
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_AR_COEFFS_Y, 24);
        assert_eq!(MAX_AR_COEFFS_UV, 25);
        assert_eq!(MAX_AR_LAG, 3);
        assert_eq!(GRAIN_BLOCK_SIZE, 32);
        assert_eq!(MAX_LUMA_POINTS, 14);
        assert_eq!(MAX_CHROMA_POINTS, 10);
        assert_eq!(GRAIN_LUT_SIZE, 82);
    }
}
