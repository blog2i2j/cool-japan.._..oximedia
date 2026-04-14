//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::film_grain_table::PerBlockGrainTable;
use crate::frame::VideoFrame;
use crate::CodecResult;

use super::functions::{
    GAUSSIAN_SEQUENCE_SIZE, GRAIN_BLOCK_SIZE, GRAIN_SEED_XOR, GRAIN_TEMPLATE_SIZE_LARGE,
    GRAIN_TEMPLATE_SIZE_SMALL, MAX_AR_COEFFS_CHROMA, MAX_AR_COEFFS_LUMA, MAX_AR_LAG,
    MAX_CHROMA_SCALING_POINTS, MAX_LUMA_SCALING_POINTS,
};

/// Scaling lookup table for grain intensity.
pub(crate) struct ScalingLut {
    pub(crate) values: Vec<u8>,
}
impl ScalingLut {
    /// Create scaling LUT from points.
    pub(crate) fn from_points(points: &[ScalingPoint], num_points: usize, bit_depth: u8) -> Self {
        let lut_size = 1 << bit_depth.min(8);
        let mut values = vec![0u8; lut_size];
        if num_points == 0 {
            return Self { values };
        }
        let first_scaling = points[0].scaling;
        for val in values.iter_mut().take(points[0].value as usize) {
            *val = first_scaling;
        }
        for i in 0..num_points.saturating_sub(1) {
            let p0 = points[i];
            let p1 = points[i + 1];
            let start = p0.value as usize;
            let end = (p1.value as usize).min(lut_size - 1);
            for x in start..=end {
                if x < lut_size {
                    let delta = (p1.value - p0.value) as i32;
                    if delta > 0 {
                        let t = ((x - start) as i32 * 256) / delta;
                        let s0 = i32::from(p0.scaling);
                        let s1 = i32::from(p1.scaling);
                        let interpolated = s0 + ((s1 - s0) * t + 128) / 256;
                        values[x] = interpolated.clamp(0, 255) as u8;
                    } else {
                        values[x] = p0.scaling;
                    }
                }
            }
        }
        let last_point = points[num_points - 1];
        let start = (last_point.value as usize + 1).min(lut_size);
        for val in values.iter_mut().skip(start) {
            *val = last_point.scaling;
        }
        Self { values }
    }
    /// Get scaling value for pixel.
    #[must_use]
    pub(crate) fn get(&self, pixel: i32, bit_depth: u8) -> u8 {
        if bit_depth <= 8 {
            let idx = pixel.clamp(0, 255) as usize;
            self.values[idx.min(self.values.len() - 1)]
        } else {
            let shift = bit_depth - 8;
            let idx = (pixel >> shift).clamp(0, 255) as usize;
            self.values[idx.min(self.values.len() - 1)]
        }
    }
}
/// AV1 Film Grain Parameters.
///
/// These parameters control the synthesis of film grain as specified in AV1 spec section 7.9.
#[derive(Clone, Debug)]
pub struct FilmGrainParams {
    /// Apply grain to this frame.
    pub apply_grain: bool,
    /// Random seed for grain generation (16-bit).
    pub grain_seed: u16,
    /// Update grain parameters from this frame.
    pub update_grain: bool,
    /// Film grain parameters present flag.
    pub film_grain_params_present: bool,
    /// Number of Y scaling points (0-14).
    pub num_y_points: usize,
    /// Y scaling points.
    pub y_points: [ScalingPoint; MAX_LUMA_SCALING_POINTS],
    /// Chroma scaling derived from luma.
    pub chroma_scaling_from_luma: bool,
    /// Number of Cb scaling points (0-10).
    pub num_cb_points: usize,
    /// Cb scaling points.
    pub cb_points: [ScalingPoint; MAX_CHROMA_SCALING_POINTS],
    /// Number of Cr scaling points (0-10).
    pub num_cr_points: usize,
    /// Cr scaling points.
    pub cr_points: [ScalingPoint; MAX_CHROMA_SCALING_POINTS],
    /// Grain scaling shift (8-11). Actual scaling is (8 + grain_scaling_minus_8).
    pub grain_scaling_minus_8: u8,
    /// AR coefficients lag (0-3).
    pub ar_coeff_lag: u8,
    /// AR coefficients for Y plane.
    pub ar_coeffs_y: [i8; MAX_AR_COEFFS_LUMA],
    /// AR coefficients for Cb plane.
    pub ar_coeffs_cb: [i8; MAX_AR_COEFFS_CHROMA],
    /// AR coefficients for Cr plane.
    pub ar_coeffs_cr: [i8; MAX_AR_COEFFS_CHROMA],
    /// AR coefficient shift (6-9). Actual shift is (6 + ar_coeff_shift_minus_6).
    pub ar_coeff_shift_minus_6: u8,
    /// Grain scale shift (0-3).
    pub grain_scale_shift: u8,
    /// Cb multiplier (0-255).
    pub cb_mult: u8,
    /// Cb luma multiplier (0-255).
    pub cb_luma_mult: u8,
    /// Cb offset (0-511).
    pub cb_offset: u16,
    /// Cr multiplier (0-255).
    pub cr_mult: u8,
    /// Cr luma multiplier (0-255).
    pub cr_luma_mult: u8,
    /// Cr offset (0-511).
    pub cr_offset: u16,
    /// Overlap flag - enables overlapping grain blocks.
    pub overlap_flag: bool,
    /// Clip to restricted range.
    pub clip_to_restricted_range: bool,
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
        self.apply_grain && self.film_grain_params_present && self.num_y_points > 0
    }
    /// Get grain scaling value (8-11).
    #[must_use]
    pub const fn grain_scaling(&self) -> u8 {
        self.grain_scaling_minus_8 + 8
    }
    /// Get AR coefficient shift (6-9).
    #[must_use]
    pub const fn ar_coeff_shift(&self) -> u8 {
        self.ar_coeff_shift_minus_6 + 6
    }
    /// Get number of AR coefficients for Y plane.
    #[must_use]
    pub fn num_ar_coeffs_y(&self) -> usize {
        let lag = self.ar_coeff_lag as usize;
        if lag == 0 {
            0
        } else {
            2 * lag * (lag + 1)
        }
    }
    /// Get number of AR coefficients for chroma planes.
    #[must_use]
    pub fn num_ar_coeffs_uv(&self) -> usize {
        let lag = self.ar_coeff_lag as usize;
        if lag == 0 {
            0
        } else {
            2 * lag * (lag + 1) + 1
        }
    }
    /// Add a Y scaling point.
    pub fn add_y_point(&mut self, value: u8, scaling: u8) {
        if self.num_y_points < MAX_LUMA_SCALING_POINTS {
            self.y_points[self.num_y_points] = ScalingPoint::new(value, scaling);
            self.num_y_points += 1;
        }
    }
    /// Add a Cb scaling point.
    pub fn add_cb_point(&mut self, value: u8, scaling: u8) {
        if self.num_cb_points < MAX_CHROMA_SCALING_POINTS {
            self.cb_points[self.num_cb_points] = ScalingPoint::new(value, scaling);
            self.num_cb_points += 1;
        }
    }
    /// Add a Cr scaling point.
    pub fn add_cr_point(&mut self, value: u8, scaling: u8) {
        if self.num_cr_points < MAX_CHROMA_SCALING_POINTS {
            self.cr_points[self.num_cr_points] = ScalingPoint::new(value, scaling);
            self.num_cr_points += 1;
        }
    }
    /// Validate parameters.
    #[must_use]
    pub fn validate(&self) -> bool {
        if self.num_y_points > MAX_LUMA_SCALING_POINTS {
            return false;
        }
        if self.num_cb_points > MAX_CHROMA_SCALING_POINTS {
            return false;
        }
        if self.num_cr_points > MAX_CHROMA_SCALING_POINTS {
            return false;
        }
        if self.ar_coeff_lag > MAX_AR_LAG as u8 {
            return false;
        }
        if self.grain_scaling_minus_8 > 3 {
            return false;
        }
        if self.ar_coeff_shift_minus_6 > 3 {
            return false;
        }
        if self.grain_scale_shift > 3 {
            return false;
        }
        for i in 1..self.num_y_points {
            if self.y_points[i].value <= self.y_points[i - 1].value {
                return false;
            }
        }
        for i in 1..self.num_cb_points {
            if self.cb_points[i].value <= self.cb_points[i - 1].value {
                return false;
            }
        }
        for i in 1..self.num_cr_points {
            if self.cr_points[i].value <= self.cr_points[i - 1].value {
                return false;
            }
        }
        true
    }
}
/// Pre-computed Gaussian sequence for grain generation.
///
/// This is generated once per grain seed and reused for the entire frame.
pub(super) struct GaussianSequence {
    pub(super) values: Vec<i16>,
}
impl GaussianSequence {
    /// Generate Gaussian sequence from seed.
    pub(super) fn generate(seed: u16) -> Self {
        let mut lcg = GrainLcg::new(seed ^ GRAIN_SEED_XOR);
        let mut values = Vec::with_capacity(GAUSSIAN_SEQUENCE_SIZE);
        for _ in 0..GAUSSIAN_SEQUENCE_SIZE {
            values.push(lcg.next_grain_value());
        }
        Self { values }
    }
    /// Get value at index (with wrapping).
    #[must_use]
    pub(super) fn get(&self, index: usize) -> i16 {
        self.values[index % GAUSSIAN_SEQUENCE_SIZE]
    }
}
/// AV1 Film Grain Synthesizer.
///
/// This synthesizes and applies film grain to decoded frames according to AV1 spec.
pub struct FilmGrainSynthesizer {
    /// Current parameters.
    params: FilmGrainParams,
    /// Bit depth (8, 10, or 12).
    pub(super) bit_depth: u8,
    /// Gaussian sequence.
    gaussian: Option<GaussianSequence>,
    /// Luma grain template.
    pub(super) luma_template: Option<LumaGrainTemplate>,
    /// Cb grain template.
    cb_template: Option<ChromaGrainTemplate>,
    /// Cr grain template.
    cr_template: Option<ChromaGrainTemplate>,
    /// Luma scaling LUT.
    pub(super) luma_scaling: Option<ScalingLut>,
    /// Cb scaling LUT.
    cb_scaling: Option<ScalingLut>,
    /// Cr scaling LUT.
    cr_scaling: Option<ScalingLut>,
}
impl FilmGrainSynthesizer {
    /// Create a new film grain synthesizer.
    #[must_use]
    pub fn new(bit_depth: u8) -> Self {
        Self {
            params: FilmGrainParams::default(),
            bit_depth,
            gaussian: None,
            luma_template: None,
            cb_template: None,
            cr_template: None,
            luma_scaling: None,
            cb_scaling: None,
            cr_scaling: None,
        }
    }
    /// Set film grain parameters and regenerate templates.
    pub fn set_params(&mut self, params: FilmGrainParams) {
        self.params = params;
        if self.params.is_enabled() {
            self.generate_templates();
            self.generate_scaling_luts();
        } else {
            self.clear_templates();
        }
    }
    /// Get current parameters.
    #[must_use]
    pub const fn params(&self) -> &FilmGrainParams {
        &self.params
    }
    /// Generate all grain templates.
    fn generate_templates(&mut self) {
        let gaussian = GaussianSequence::generate(self.params.grain_seed);
        let luma_template = LumaGrainTemplate::generate(&self.params, &gaussian, self.bit_depth);
        let subsampling_x = true;
        let subsampling_y = true;
        let cb_template = ChromaGrainTemplate::generate(
            &self.params,
            &gaussian,
            &luma_template,
            true,
            subsampling_x,
            subsampling_y,
            self.bit_depth,
        );
        let cr_template = ChromaGrainTemplate::generate(
            &self.params,
            &gaussian,
            &luma_template,
            false,
            subsampling_x,
            subsampling_y,
            self.bit_depth,
        );
        self.gaussian = Some(gaussian);
        self.luma_template = Some(luma_template);
        self.cb_template = Some(cb_template);
        self.cr_template = Some(cr_template);
    }
    /// Generate scaling lookup tables.
    fn generate_scaling_luts(&mut self) {
        self.luma_scaling = Some(ScalingLut::from_points(
            &self.params.y_points,
            self.params.num_y_points,
            self.bit_depth,
        ));
        if self.params.chroma_scaling_from_luma {
            let luma_lut = ScalingLut::from_points(
                &self.params.y_points,
                self.params.num_y_points,
                self.bit_depth,
            );
            self.cb_scaling = Some(ScalingLut {
                values: luma_lut.values.clone(),
            });
            self.cr_scaling = Some(ScalingLut {
                values: luma_lut.values.clone(),
            });
        } else {
            self.cb_scaling = Some(ScalingLut::from_points(
                &self.params.cb_points,
                self.params.num_cb_points,
                self.bit_depth,
            ));
            self.cr_scaling = Some(ScalingLut::from_points(
                &self.params.cr_points,
                self.params.num_cr_points,
                self.bit_depth,
            ));
        }
    }
    /// Clear all templates.
    fn clear_templates(&mut self) {
        self.gaussian = None;
        self.luma_template = None;
        self.cb_template = None;
        self.cr_template = None;
        self.luma_scaling = None;
        self.cb_scaling = None;
        self.cr_scaling = None;
    }
    /// Apply film grain to a frame.
    ///
    /// # Errors
    ///
    /// Returns error if grain application fails.
    pub fn apply_grain(&self, frame: &mut VideoFrame) -> CodecResult<()> {
        if !self.params.is_enabled() {
            return Ok(());
        }
        if let (Some(luma_template), Some(luma_scaling)) = (&self.luma_template, &self.luma_scaling)
        {
            self.apply_grain_plane_y(frame, luma_template, luma_scaling);
        }
        if let (Some(cb_template), Some(cb_scaling)) = (&self.cb_template, &self.cb_scaling) {
            if let Some(luma_template) = &self.luma_template {
                self.apply_grain_plane_cb(frame, cb_template, cb_scaling, luma_template);
            }
        }
        if let (Some(cr_template), Some(cr_scaling)) = (&self.cr_template, &self.cr_scaling) {
            if let Some(luma_template) = &self.luma_template {
                self.apply_grain_plane_cr(frame, cr_template, cr_scaling, luma_template);
            }
        }
        Ok(())
    }
    /// Apply grain to Y plane.
    fn apply_grain_plane_y(
        &self,
        frame: &mut VideoFrame,
        template: &LumaGrainTemplate,
        scaling: &ScalingLut,
    ) {
        let width = frame.plane(0).width() as usize;
        let height = frame.plane(0).height() as usize;
        let stride = frame.plane(0).stride() as usize;
        let data = frame.plane_mut(0).data_mut();
        let grain_scale = self.params.grain_scaling() as i32;
        let grain_shift = self.params.grain_scale_shift as i32;
        let max_value = (1 << self.bit_depth) - 1;
        let offset_x = (self.params.grain_seed as usize * 37) % template.size();
        let offset_y = (self.params.grain_seed as usize * 59) % template.size();
        for y in 0..height {
            for x in 0..width {
                let idx = y * stride + x;
                let pixel = i32::from(data[idx]);
                let scale = i32::from(scaling.get(pixel, self.bit_depth));
                let grain_x = (x + offset_x) % template.size();
                let grain_y = (y + offset_y) % template.size();
                let grain = i32::from(template.get(grain_x, grain_y));
                let scaled_grain = (grain * scale) >> grain_scale;
                let adjusted_grain = scaled_grain >> grain_shift;
                let result = (pixel + adjusted_grain).clamp(0, max_value);
                data[idx] = result as u8;
            }
        }
    }
    /// Apply grain to Cb plane.
    fn apply_grain_plane_cb(
        &self,
        frame: &mut VideoFrame,
        template: &ChromaGrainTemplate,
        scaling: &ScalingLut,
        luma_template: &LumaGrainTemplate,
    ) {
        self.apply_grain_plane_chroma(frame, template, scaling, luma_template, 1, true);
    }
    /// Apply grain to Cr plane.
    fn apply_grain_plane_cr(
        &self,
        frame: &mut VideoFrame,
        template: &ChromaGrainTemplate,
        scaling: &ScalingLut,
        luma_template: &LumaGrainTemplate,
    ) {
        self.apply_grain_plane_chroma(frame, template, scaling, luma_template, 2, false);
    }
    /// Apply grain to chroma plane.
    #[allow(clippy::needless_pass_by_value)]
    fn apply_grain_plane_chroma(
        &self,
        frame: &mut VideoFrame,
        template: &ChromaGrainTemplate,
        scaling: &ScalingLut,
        luma_template: &LumaGrainTemplate,
        plane_idx: usize,
        is_cb: bool,
    ) {
        let width = frame.plane(plane_idx).width() as usize;
        let height = frame.plane(plane_idx).height() as usize;
        let stride = frame.plane(plane_idx).stride() as usize;
        let luma_width = frame.plane(0).width() as usize;
        let luma_height = frame.plane(0).height() as usize;
        let luma_stride = frame.plane(0).stride() as usize;
        let luma_data: Vec<u8> = frame.plane(0).data().to_vec();
        let subsampling_x = width * 2 == luma_width;
        let subsampling_y = height * 2 == luma_height;
        let grain_scale = self.params.grain_scaling() as i32;
        let grain_shift = self.params.grain_scale_shift as i32;
        let max_value = (1 << self.bit_depth) - 1;
        let (mult, luma_mult, offset) = if is_cb {
            (
                i32::from(self.params.cb_mult),
                i32::from(self.params.cb_luma_mult),
                i32::from(self.params.cb_offset),
            )
        } else {
            (
                i32::from(self.params.cr_mult),
                i32::from(self.params.cr_luma_mult),
                i32::from(self.params.cr_offset),
            )
        };
        let data = frame.plane_mut(plane_idx).data_mut();
        let offset_x = (self.params.grain_seed as usize * 37) % template.width;
        let offset_y = (self.params.grain_seed as usize * 59) % template.height;
        for y in 0..height {
            for x in 0..width {
                let idx = y * stride + x;
                let pixel = i32::from(data[idx]);
                let luma_x = if subsampling_x { x * 2 } else { x };
                let luma_y = if subsampling_y { y * 2 } else { y };
                let luma_idx =
                    luma_y.min(luma_height - 1) * luma_stride + luma_x.min(luma_width - 1);
                let luma_pixel = i32::from(luma_data[luma_idx]);
                let scale_src = if self.params.chroma_scaling_from_luma {
                    luma_pixel
                } else {
                    pixel
                };
                let scale = i32::from(scaling.get(scale_src, self.bit_depth));
                let grain_x = (x + offset_x) % template.width;
                let grain_y = (y + offset_y) % template.height;
                let grain = i32::from(template.get(grain_x, grain_y));
                let luma_grain_x = (luma_x + offset_x) % luma_template.size();
                let luma_grain_y = (luma_y + offset_y) % luma_template.size();
                let luma_grain = i32::from(luma_template.get(luma_grain_x, luma_grain_y));
                let combined_grain = (mult * grain + luma_mult * luma_grain + 128) >> 8;
                let scaled_grain = (combined_grain * scale) >> grain_scale;
                let adjusted_grain = (scaled_grain + offset - 256) >> grain_shift;
                let result = (pixel + adjusted_grain).clamp(0, max_value);
                data[idx] = result as u8;
            }
        }
    }
    /// Apply grain with overlap blending.
    ///
    /// When `overlap_flag` is set, grain blocks overlap by 2 pixels at
    /// boundaries. The AV1 spec (Section 7.18.3.4) defines linear ramp
    /// weights for blending adjacent grain blocks so that seams are
    /// invisible.
    ///
    /// Block layout (GRAIN_BLOCK_SIZE = 32):
    /// ```text
    /// ┌───────────┬───────────┐
    /// │ block(0,0) │ block(1,0)│  <- 2-pixel horizontal overlap
    /// │            │←overlap→  │
    /// ├───────────┼───────────┤  <- 2-pixel vertical overlap
    /// │ block(0,1) │ block(1,1)│
    /// └───────────┴───────────┘
    /// ```
    ///
    /// # Errors
    ///
    /// Returns error if grain application fails.
    pub fn apply_grain_with_overlap(&self, frame: &mut VideoFrame) -> CodecResult<()> {
        if !self.params.overlap_flag {
            return self.apply_grain(frame);
        }
        if !self.params.is_enabled() {
            return Ok(());
        }
        if let (Some(luma_template), Some(luma_scaling)) = (&self.luma_template, &self.luma_scaling)
        {
            self.apply_grain_plane_y_overlap(frame, luma_template, luma_scaling);
        }
        if let (Some(cb_template), Some(cb_scaling)) = (&self.cb_template, &self.cb_scaling) {
            if let Some(luma_template) = &self.luma_template {
                self.apply_grain_plane_chroma_overlap(
                    frame,
                    cb_template,
                    cb_scaling,
                    luma_template,
                    1,
                    true,
                );
            }
        }
        if let (Some(cr_template), Some(cr_scaling)) = (&self.cr_template, &self.cr_scaling) {
            if let Some(luma_template) = &self.luma_template {
                self.apply_grain_plane_chroma_overlap(
                    frame,
                    cr_template,
                    cr_scaling,
                    luma_template,
                    2,
                    false,
                );
            }
        }
        Ok(())
    }
    /// Apply film grain with per-block parameter overrides from `table`.
    ///
    /// # Errors
    ///
    /// Returns error if grain application fails.
    pub fn apply_grain_per_block(
        &self,
        frame: &mut VideoFrame,
        table: &PerBlockGrainTable,
    ) -> CodecResult<()> {
        if !self.params.is_enabled() {
            return Ok(());
        }
        let width = frame.plane(0).width() as usize;
        let height = frame.plane(0).height() as usize;
        let stride = frame.plane(0).stride() as usize;
        let bsz = GRAIN_BLOCK_SIZE;
        let max_val = i32::from((1u16 << self.bit_depth.min(16)) - 1);
        for by in 0..height.div_ceil(bsz) {
            for bx in 0..width.div_ceil(bsz) {
                let res = table.resolve(&self.params, bx as u32, by as u32);
                let gauss = GaussianSequence::generate(res.grain_seed);
                let tpl = LumaGrainTemplate::generate(&res, &gauss, self.bit_depth);
                let n = res.num_y_points.min(MAX_LUMA_SCALING_POINTS);
                let lut = ScalingLut::from_points(&res.y_points[..n], n, self.bit_depth);
                let g_scale = i32::from(res.grain_scaling());
                let g_shift = i32::from(res.grain_scale_shift);
                let tsz = tpl.size().max(1);
                let ox = (res.grain_seed as usize * 37) % tsz;
                let oy = (res.grain_seed as usize * 59) % tsz;
                let x1 = ((bx + 1) * bsz).min(width);
                let y1 = ((by + 1) * bsz).min(height);
                let data = frame.plane_mut(0).data_mut();
                for y in (by * bsz)..y1 {
                    for x in (bx * bsz)..x1 {
                        let idx = y * stride + x;
                        let pix = i32::from(data[idx]);
                        let gs = i32::from(tpl.get((ox + x) % tsz, (oy + y) % tsz));
                        let sc = i32::from(lut.get(pix, self.bit_depth));
                        data[idx] =
                            ((pix + (gs * sc * g_scale)) >> (g_shift + 8)).clamp(0, max_val) as u8;
                    }
                }
            }
        }
        Ok(())
    }
    /// Apply film grain to a 10-bit or 12-bit luma plane stored as `u16`.
    ///
    /// # Errors
    ///
    /// Returns `CodecResult<()>` (currently infallible) for API consistency.
    pub fn apply_grain_per_block_u16(
        &self,
        luma_plane: &mut [u16],
        width: usize,
        height: usize,
        stride: usize,
        table: &PerBlockGrainTable,
    ) -> CodecResult<()> {
        if !self.params.is_enabled() {
            return Ok(());
        }
        let max_val = i32::from((1u16 << self.bit_depth.min(16)) - 1);
        let bsz = GRAIN_BLOCK_SIZE;
        for by in 0..height.div_ceil(bsz) {
            for bx in 0..width.div_ceil(bsz) {
                let res = table.resolve(&self.params, bx as u32, by as u32);
                let gauss = GaussianSequence::generate(res.grain_seed);
                let tpl = LumaGrainTemplate::generate(&res, &gauss, self.bit_depth);
                let n = res.num_y_points.min(MAX_LUMA_SCALING_POINTS);
                let lut = ScalingLut::from_points(&res.y_points[..n], n, self.bit_depth);
                let g_scale = i32::from(res.grain_scaling());
                let g_shift = i32::from(res.grain_scale_shift);
                let tsz = tpl.size().max(1);
                let ox = (res.grain_seed as usize * 37) % tsz;
                let oy = (res.grain_seed as usize * 59) % tsz;
                let x1 = ((bx + 1) * bsz).min(width);
                let y1 = ((by + 1) * bsz).min(height);
                for y in (by * bsz)..y1 {
                    for x in (bx * bsz)..x1 {
                        let idx = y * stride + x;
                        if idx >= luma_plane.len() {
                            continue;
                        }
                        let pix = i32::from(luma_plane[idx]);
                        let gs = i32::from(tpl.get((ox + x) % tsz, (oy + y) % tsz));
                        let sc = i32::from(lut.get(pix, self.bit_depth));
                        let result =
                            ((pix + (gs * sc * g_scale)) >> (g_shift + 8)).clamp(0, max_val);
                        luma_plane[idx] = result as u16;
                    }
                }
            }
        }
        Ok(())
    }
    /// Compute overlap blending weight for a coordinate within the overlap zone.
    ///
    /// AV1 uses a 2-pixel overlap region with linear ramp weights:
    /// - Position 0 in overlap: weight = 27 (out of 32)
    /// - Position 1 in overlap: weight = 17 (out of 32)
    /// The complementary block gets (32 - weight).
    pub(super) fn overlap_weight(pos_in_overlap: usize) -> (i32, i32) {
        match pos_in_overlap {
            0 => (27, 5),
            1 => (17, 15),
            _ => (32, 0),
        }
    }
    /// Apply grain to Y plane with overlap blending between grain blocks.
    fn apply_grain_plane_y_overlap(
        &self,
        frame: &mut VideoFrame,
        template: &LumaGrainTemplate,
        scaling: &ScalingLut,
    ) {
        let width = frame.plane(0).width() as usize;
        let height = frame.plane(0).height() as usize;
        let stride = frame.plane(0).stride() as usize;
        let data = frame.plane_mut(0).data_mut();
        let grain_scale = self.params.grain_scaling() as i32;
        let grain_shift = self.params.grain_scale_shift as i32;
        let max_value = (1 << self.bit_depth) - 1;
        let block_size = GRAIN_BLOCK_SIZE;
        let overlap = 2usize;
        let blocks_y = (height + block_size - 1) / block_size;
        let blocks_x = (width + block_size - 1) / block_size;
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_x0 = bx * block_size;
                let block_y0 = by * block_size;
                let seed_offset = (by * blocks_x + bx) as u16;
                let offset_x = ((self
                    .params
                    .grain_seed
                    .wrapping_add(seed_offset.wrapping_mul(37)))
                    as usize)
                    % template.size();
                let offset_y = ((self
                    .params
                    .grain_seed
                    .wrapping_add(seed_offset.wrapping_mul(59)))
                    as usize)
                    % template.size();
                for local_y in 0..block_size {
                    let y = block_y0 + local_y;
                    if y >= height {
                        break;
                    }
                    for local_x in 0..block_size {
                        let x = block_x0 + local_x;
                        if x >= width {
                            break;
                        }
                        let idx = y * stride + x;
                        let pixel = i32::from(data[idx]);
                        let scale = i32::from(scaling.get(pixel, self.bit_depth));
                        let grain_x = (local_x + offset_x) % template.size();
                        let grain_y = (local_y + offset_y) % template.size();
                        let mut grain_val = i32::from(template.get(grain_x, grain_y));
                        if bx > 0 && local_x < overlap {
                            let left_seed_offset = (by * blocks_x + (bx - 1)) as u16;
                            let left_ox = ((self
                                .params
                                .grain_seed
                                .wrapping_add(left_seed_offset.wrapping_mul(37)))
                                as usize)
                                % template.size();
                            let left_oy = ((self
                                .params
                                .grain_seed
                                .wrapping_add(left_seed_offset.wrapping_mul(59)))
                                as usize)
                                % template.size();
                            let left_local_x = block_size + local_x;
                            let left_gx = (left_local_x + left_ox) % template.size();
                            let left_gy = (local_y + left_oy) % template.size();
                            let left_grain = i32::from(template.get(left_gx, left_gy));
                            let (w_curr, w_left) = Self::overlap_weight(local_x);
                            grain_val = (grain_val * w_curr + left_grain * w_left + 16) >> 5;
                        }
                        if by > 0 && local_y < overlap {
                            let top_seed_offset = ((by - 1) * blocks_x + bx) as u16;
                            let top_ox = ((self
                                .params
                                .grain_seed
                                .wrapping_add(top_seed_offset.wrapping_mul(37)))
                                as usize)
                                % template.size();
                            let top_oy = ((self
                                .params
                                .grain_seed
                                .wrapping_add(top_seed_offset.wrapping_mul(59)))
                                as usize)
                                % template.size();
                            let top_local_y = block_size + local_y;
                            let top_gx = (local_x + top_ox) % template.size();
                            let top_gy = (top_local_y + top_oy) % template.size();
                            let top_grain = i32::from(template.get(top_gx, top_gy));
                            let (w_curr, w_top) = Self::overlap_weight(local_y);
                            grain_val = (grain_val * w_curr + top_grain * w_top + 16) >> 5;
                        }
                        let scaled_grain = (grain_val * scale) >> grain_scale;
                        let adjusted_grain = scaled_grain >> grain_shift;
                        let result = (pixel + adjusted_grain).clamp(0, max_value);
                        data[idx] = result as u8;
                    }
                }
            }
        }
    }
    /// Apply grain to chroma plane with overlap blending.
    fn apply_grain_plane_chroma_overlap(
        &self,
        frame: &mut VideoFrame,
        template: &ChromaGrainTemplate,
        scaling: &ScalingLut,
        luma_template: &LumaGrainTemplate,
        plane_idx: usize,
        is_cb: bool,
    ) {
        let width = frame.plane(plane_idx).width() as usize;
        let height = frame.plane(plane_idx).height() as usize;
        let stride = frame.plane(plane_idx).stride() as usize;
        let luma_width = frame.plane(0).width() as usize;
        let luma_height = frame.plane(0).height() as usize;
        let luma_stride = frame.plane(0).stride() as usize;
        let luma_data: Vec<u8> = frame.plane(0).data().to_vec();
        let subsampling_x = width * 2 == luma_width;
        let subsampling_y = height * 2 == luma_height;
        let grain_scale = self.params.grain_scaling() as i32;
        let grain_shift = self.params.grain_scale_shift as i32;
        let max_value = (1 << self.bit_depth) - 1;
        let block_size = GRAIN_BLOCK_SIZE;
        let chroma_block = if subsampling_x {
            block_size / 2
        } else {
            block_size
        };
        let overlap = 2usize;
        let (mult, luma_mult, offset) = if is_cb {
            (
                i32::from(self.params.cb_mult),
                i32::from(self.params.cb_luma_mult),
                i32::from(self.params.cb_offset),
            )
        } else {
            (
                i32::from(self.params.cr_mult),
                i32::from(self.params.cr_luma_mult),
                i32::from(self.params.cr_offset),
            )
        };
        let data = frame.plane_mut(plane_idx).data_mut();
        let blocks_y = (height + chroma_block - 1) / chroma_block;
        let blocks_x = (width + chroma_block - 1) / chroma_block;
        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let block_x0 = bx * chroma_block;
                let block_y0 = by * chroma_block;
                let seed_offset = (by * blocks_x + bx) as u16;
                let offset_x = ((self
                    .params
                    .grain_seed
                    .wrapping_add(seed_offset.wrapping_mul(41)))
                    as usize)
                    % template.width;
                let offset_y = ((self
                    .params
                    .grain_seed
                    .wrapping_add(seed_offset.wrapping_mul(67)))
                    as usize)
                    % template.height;
                for local_y in 0..chroma_block {
                    let y = block_y0 + local_y;
                    if y >= height {
                        break;
                    }
                    for local_x in 0..chroma_block {
                        let x = block_x0 + local_x;
                        if x >= width {
                            break;
                        }
                        let idx = y * stride + x;
                        let pixel = i32::from(data[idx]);
                        let luma_x = if subsampling_x { x * 2 } else { x };
                        let luma_y = if subsampling_y { y * 2 } else { y };
                        let luma_idx =
                            luma_y.min(luma_height - 1) * luma_stride + luma_x.min(luma_width - 1);
                        let luma_pixel = i32::from(luma_data[luma_idx]);
                        let scale_src = if self.params.chroma_scaling_from_luma {
                            luma_pixel
                        } else {
                            pixel
                        };
                        let scale = i32::from(scaling.get(scale_src, self.bit_depth));
                        let grain_x = (local_x + offset_x) % template.width;
                        let grain_y = (local_y + offset_y) % template.height;
                        let mut grain_val = i32::from(template.get(grain_x, grain_y));
                        let luma_grain_x = (luma_x + offset_x) % luma_template.size();
                        let luma_grain_y = (luma_y + offset_y) % luma_template.size();
                        let luma_grain = i32::from(luma_template.get(luma_grain_x, luma_grain_y));
                        if bx > 0 && local_x < overlap {
                            let left_seed = (by * blocks_x + (bx - 1)) as u16;
                            let left_ox = ((self
                                .params
                                .grain_seed
                                .wrapping_add(left_seed.wrapping_mul(41)))
                                as usize)
                                % template.width;
                            let left_oy = ((self
                                .params
                                .grain_seed
                                .wrapping_add(left_seed.wrapping_mul(67)))
                                as usize)
                                % template.height;
                            let left_gx = (chroma_block + local_x + left_ox) % template.width;
                            let left_gy = (local_y + left_oy) % template.height;
                            let left_grain = i32::from(template.get(left_gx, left_gy));
                            let (w_curr, w_left) = Self::overlap_weight(local_x);
                            grain_val = (grain_val * w_curr + left_grain * w_left + 16) >> 5;
                        }
                        if by > 0 && local_y < overlap {
                            let top_seed = ((by - 1) * blocks_x + bx) as u16;
                            let top_ox = ((self
                                .params
                                .grain_seed
                                .wrapping_add(top_seed.wrapping_mul(41)))
                                as usize)
                                % template.width;
                            let top_oy = ((self
                                .params
                                .grain_seed
                                .wrapping_add(top_seed.wrapping_mul(67)))
                                as usize)
                                % template.height;
                            let top_gx = (local_x + top_ox) % template.width;
                            let top_gy = (chroma_block + local_y + top_oy) % template.height;
                            let top_grain = i32::from(template.get(top_gx, top_gy));
                            let (w_curr, w_top) = Self::overlap_weight(local_y);
                            grain_val = (grain_val * w_curr + top_grain * w_top + 16) >> 5;
                        }
                        let combined_grain = (mult * grain_val + luma_mult * luma_grain + 128) >> 8;
                        let scaled_grain = (combined_grain * scale) >> grain_scale;
                        let adjusted_grain = (scaled_grain + offset - 256) >> grain_shift;
                        let result = (pixel + adjusted_grain).clamp(0, max_value);
                        data[idx] = result as u8;
                    }
                }
            }
        }
    }
}
/// Chroma grain template.
///
/// Chroma grain includes correlation with luma grain.
struct ChromaGrainTemplate {
    /// Grain values.
    data: Vec<i16>,
    /// Template width.
    width: usize,
    /// Template height.
    height: usize,
}
impl ChromaGrainTemplate {
    /// Generate chroma grain template.
    #[allow(clippy::needless_pass_by_value)]
    fn generate(
        params: &FilmGrainParams,
        gaussian: &GaussianSequence,
        luma_template: &LumaGrainTemplate,
        is_cb: bool,
        subsampling_x: bool,
        subsampling_y: bool,
        bit_depth: u8,
    ) -> Self {
        let luma_size = luma_template.size();
        let width = if subsampling_x {
            luma_size / 2
        } else {
            luma_size
        };
        let height = if subsampling_y {
            luma_size / 2
        } else {
            luma_size
        };
        let mut data = vec![0i16; width * height];
        let offset = if is_cb {
            GAUSSIAN_SEQUENCE_SIZE
        } else {
            GAUSSIAN_SEQUENCE_SIZE + width * height
        };
        for y in 0..height {
            for x in 0..width {
                let seq_idx = offset + y * width + x;
                data[y * width + x] = gaussian.get(seq_idx);
            }
        }
        let mut template = Self {
            data,
            width,
            height,
        };
        if params.ar_coeff_lag > 0 {
            template.apply_ar_filter(
                params,
                luma_template,
                is_cb,
                subsampling_x,
                subsampling_y,
                bit_depth,
            );
        }
        template
    }
    /// Apply auto-regressive filter to chroma grain.
    #[allow(clippy::needless_pass_by_value)]
    fn apply_ar_filter(
        &mut self,
        params: &FilmGrainParams,
        luma_template: &LumaGrainTemplate,
        is_cb: bool,
        subsampling_x: bool,
        subsampling_y: bool,
        bit_depth: u8,
    ) {
        let lag = params.ar_coeff_lag as usize;
        let shift = params.ar_coeff_shift() as i32;
        let num_coeffs = params.num_ar_coeffs_uv();
        let coeffs = if is_cb {
            &params.ar_coeffs_cb
        } else {
            &params.ar_coeffs_cr
        };
        let grain_min = -(256 << (bit_depth.saturating_sub(8)));
        let grain_max = (256 << (bit_depth.saturating_sub(8))) - 1;
        for y in lag..self.height {
            for x in lag..(self.width - lag) {
                let mut sum: i32 = 0;
                let mut coeff_idx = 0;
                for dy in 0..=lag {
                    for dx in 0..(2 * lag + 1) {
                        if dy == 0 && dx >= lag {
                            break;
                        }
                        if coeff_idx < num_coeffs - 1 {
                            let coeff = i32::from(coeffs[coeff_idx]);
                            let src_y = y - lag + dy;
                            let src_x = x - lag + dx;
                            let src_val = i32::from(self.data[src_y * self.width + src_x]);
                            sum += coeff * src_val;
                            coeff_idx += 1;
                        }
                    }
                }
                if coeff_idx < num_coeffs {
                    let luma_coeff = i32::from(coeffs[coeff_idx]);
                    let luma_x = if subsampling_x { x * 2 } else { x };
                    let luma_y = if subsampling_y { y * 2 } else { y };
                    let luma_val = i32::from(luma_template.get(luma_x, luma_y));
                    sum += luma_coeff * luma_val;
                }
                let idx = y * self.width + x;
                let current = i32::from(self.data[idx]);
                let filtered = current + (sum >> shift);
                self.data[idx] = filtered.clamp(grain_min, grain_max) as i16;
            }
        }
    }
    /// Get grain value at position (with wrapping).
    #[must_use]
    fn get(&self, x: usize, y: usize) -> i16 {
        let x_wrap = x % self.width;
        let y_wrap = y % self.height;
        self.data[y_wrap * self.width + x_wrap]
    }
}
/// Luma grain template.
///
/// This is the base grain pattern before scaling and AR filtering.
pub(super) struct LumaGrainTemplate {
    /// Grain values.
    data: Vec<i16>,
    /// Template size (64 or 128).
    size: usize,
}
impl LumaGrainTemplate {
    /// Generate luma grain template.
    fn generate(params: &FilmGrainParams, gaussian: &GaussianSequence, bit_depth: u8) -> Self {
        let size = if params.num_y_points == 0 {
            GRAIN_TEMPLATE_SIZE_SMALL
        } else {
            GRAIN_TEMPLATE_SIZE_LARGE
        };
        let mut data = vec![0i16; size * size];
        let mut seq_idx = 0;
        for y in 0..size {
            for x in 0..size {
                data[y * size + x] = gaussian.get(seq_idx);
                seq_idx += 1;
            }
        }
        let mut template = Self { data, size };
        if params.ar_coeff_lag > 0 {
            template.apply_ar_filter(params, bit_depth);
        }
        template
    }
    /// Apply auto-regressive filter to luma grain.
    fn apply_ar_filter(&mut self, params: &FilmGrainParams, bit_depth: u8) {
        let lag = params.ar_coeff_lag as usize;
        let shift = params.ar_coeff_shift() as i32;
        let num_coeffs = params.num_ar_coeffs_y();
        let grain_min = -(256 << (bit_depth.saturating_sub(8)));
        let grain_max = (256 << (bit_depth.saturating_sub(8))) - 1;
        for y in lag..self.size {
            for x in lag..(self.size - lag) {
                let mut sum: i32 = 0;
                let mut coeff_idx = 0;
                for dy in 0..=lag {
                    for dx in 0..(2 * lag + 1) {
                        if dy == 0 && dx >= lag {
                            break;
                        }
                        if coeff_idx < num_coeffs {
                            let coeff = i32::from(params.ar_coeffs_y[coeff_idx]);
                            let src_y = y - lag + dy;
                            let src_x = x - lag + dx;
                            let src_val = i32::from(self.data[src_y * self.size + src_x]);
                            sum += coeff * src_val;
                            coeff_idx += 1;
                        }
                    }
                }
                let idx = y * self.size + x;
                let current = i32::from(self.data[idx]);
                let filtered = current + (sum >> shift);
                self.data[idx] = filtered.clamp(grain_min, grain_max) as i16;
            }
        }
    }
    /// Get grain value at position (with wrapping).
    #[must_use]
    fn get(&self, x: usize, y: usize) -> i16 {
        let x_wrap = x % self.size;
        let y_wrap = y % self.size;
        self.data[y_wrap * self.size + x_wrap]
    }
    /// Get template size.
    #[must_use]
    const fn size(&self) -> usize {
        self.size
    }
}
/// Scaling point for grain intensity.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScalingPoint {
    /// Input value (0-255 for 8-bit, scaled for higher bit depths).
    pub value: u8,
    /// Scaling factor (0-255).
    pub scaling: u8,
}
impl ScalingPoint {
    /// Create a new scaling point.
    #[must_use]
    pub const fn new(value: u8, scaling: u8) -> Self {
        Self { value, scaling }
    }
}
/// Linear Congruential Generator for grain synthesis.
///
/// This implements the exact PRNG specified in the AV1 spec for bit-exact grain synthesis.
pub(super) struct GrainLcg {
    state: u16,
}
impl GrainLcg {
    /// Create a new LCG with given seed.
    pub(super) fn new(seed: u16) -> Self {
        Self { state: seed }
    }
    /// Generate next random value (0-2047).
    pub(super) fn next(&mut self) -> u16 {
        let bit =
            ((self.state >> 0) ^ (self.state >> 1) ^ (self.state >> 3) ^ (self.state >> 12)) & 1;
        self.state = (self.state >> 1) | (bit << 15);
        self.state
    }
    /// Generate next value in range [-2048, 2048).
    pub(super) fn next_grain_value(&mut self) -> i16 {
        let val = self.next();
        let val12 = val & 0x0FFF;
        if val12 >= 2048 {
            val12 as i16 - 4096
        } else {
            val12 as i16
        }
    }
}
