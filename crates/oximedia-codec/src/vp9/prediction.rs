//! VP9 Prediction buffer and interpolation filters.
//!
//! This module provides prediction buffers for storing prediction results
//! and interpolation filters for subpixel motion compensation in VP9.
//!
//! VP9 supports multiple interpolation filter types:
//! - Bilinear: Simple 2-tap filter for fast interpolation
//! - EIGHTTAP: Standard 8-tap filter
//! - EIGHTTAP_SMOOTH: Smoother 8-tap filter for less aliasing
//! - EIGHTTAP_SHARP: Sharper 8-tap filter for more detail

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::similar_names)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]

use super::inter::{InterPredContext, ScalingFactors};
use super::mv::MotionVector;
use super::partition::BlockSize;

/// Number of interpolation filter types.
pub const INTERP_FILTER_TYPES: usize = 4;

/// Number of subpixel positions (1/8 pixel precision).
pub const SUBPEL_SHIFTS: usize = 8;

/// Number of taps in the 8-tap filter.
pub const FILTER_TAPS: usize = 8;

/// Maximum prediction buffer size (64x64 block + filter margin).
pub const MAX_PRED_SIZE: usize = 64 + FILTER_TAPS;

/// Interpolation filter type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
#[repr(u8)]
pub enum InterpFilter {
    /// Standard 8-tap filter.
    #[default]
    EightTap = 0,
    /// Smoother 8-tap filter.
    EightTapSmooth = 1,
    /// Sharper 8-tap filter.
    EightTapSharp = 2,
    /// Simple 2-tap bilinear filter.
    Bilinear = 3,
}

impl InterpFilter {
    /// All interpolation filter types.
    pub const ALL: [InterpFilter; INTERP_FILTER_TYPES] = [
        InterpFilter::EightTap,
        InterpFilter::EightTapSmooth,
        InterpFilter::EightTapSharp,
        InterpFilter::Bilinear,
    ];

    /// Converts from u8 value to `InterpFilter`.
    #[must_use]
    pub const fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(Self::EightTap),
            1 => Some(Self::EightTapSmooth),
            2 => Some(Self::EightTapSharp),
            3 => Some(Self::Bilinear),
            _ => None,
        }
    }

    /// Returns the index of this filter type.
    #[must_use]
    pub const fn index(&self) -> usize {
        *self as usize
    }

    /// Returns true if this is a bilinear filter.
    #[must_use]
    pub const fn is_bilinear(&self) -> bool {
        matches!(self, Self::Bilinear)
    }

    /// Returns true if this is an 8-tap filter.
    #[must_use]
    pub const fn is_eight_tap(&self) -> bool {
        !self.is_bilinear()
    }

    /// Returns the number of taps for this filter.
    #[must_use]
    pub const fn num_taps(&self) -> usize {
        if self.is_bilinear() {
            2
        } else {
            8
        }
    }

    /// Returns the filter tap offset (for centering).
    #[must_use]
    pub const fn tap_offset(&self) -> usize {
        if self.is_bilinear() {
            0
        } else {
            3 // 8-tap filter is centered at tap 3
        }
    }
}

impl From<InterpFilter> for u8 {
    fn from(value: InterpFilter) -> Self {
        value as u8
    }
}

/// 8-tap filter coefficients for each subpixel position.
///
/// Filter coefficients sum to 128 (for 7-bit precision).
pub static EIGHTTAP_FILTER: [[i16; FILTER_TAPS]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 1, -5, 126, 8, -3, 1, 0],
    [-1, 3, -10, 122, 18, -6, 2, 0],
    [-1, 4, -13, 118, 27, -9, 3, -1],
    [-1, 4, -16, 112, 37, -11, 4, -1],
    [-1, 5, -18, 105, 48, -14, 4, -1],
    [-1, 5, -19, 97, 58, -16, 5, -1],
    [-1, 6, -19, 88, 68, -18, 5, -1],
];

/// 8-tap smooth filter coefficients for each subpixel position.
pub static EIGHTTAP_SMOOTH_FILTER: [[i16; FILTER_TAPS]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-3, -1, 32, 64, 38, 1, -3, 0],
    [-2, -2, 29, 63, 41, 2, -3, 0],
    [-2, -2, 26, 63, 43, 4, -4, 0],
    [-2, -3, 24, 62, 46, 5, -4, 0],
    [-2, -3, 21, 60, 49, 7, -4, 0],
    [-1, -4, 18, 59, 51, 9, -4, 0],
    [-1, -4, 16, 57, 53, 12, -4, -1],
];

/// 8-tap sharp filter coefficients for each subpixel position.
pub static EIGHTTAP_SHARP_FILTER: [[i16; FILTER_TAPS]; SUBPEL_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [-1, 3, -7, 127, 8, -3, 1, 0],
    [-2, 5, -13, 125, 17, -6, 3, -1],
    [-3, 7, -17, 121, 27, -10, 5, -2],
    [-4, 9, -20, 115, 37, -13, 6, -2],
    [-4, 10, -23, 108, 48, -16, 8, -3],
    [-4, 10, -24, 100, 59, -19, 9, -3],
    [-4, 11, -24, 90, 70, -21, 10, -4],
];

/// Bilinear filter coefficients for each subpixel position.
pub static BILINEAR_FILTER: [[i16; 2]; SUBPEL_SHIFTS] = [
    [128, 0],
    [112, 16],
    [96, 32],
    [80, 48],
    [64, 64],
    [48, 80],
    [32, 96],
    [16, 112],
];

/// Returns the filter coefficients for a given filter type and subpixel position.
#[must_use]
pub fn get_filter_coeffs(filter: InterpFilter, subpel: usize) -> &'static [i16] {
    let subpel = subpel & 7;
    match filter {
        InterpFilter::EightTap => &EIGHTTAP_FILTER[subpel],
        InterpFilter::EightTapSmooth => &EIGHTTAP_SMOOTH_FILTER[subpel],
        InterpFilter::EightTapSharp => &EIGHTTAP_SHARP_FILTER[subpel],
        InterpFilter::Bilinear => &BILINEAR_FILTER[subpel],
    }
}

/// Prediction buffer for storing prediction results.
#[derive(Clone, Debug)]
pub struct PredBuffer {
    /// Buffer data (Y plane).
    pub y: Vec<u8>,
    /// Buffer data (U plane).
    pub u: Vec<u8>,
    /// Buffer data (V plane).
    pub v: Vec<u8>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Y plane stride.
    pub y_stride: usize,
    /// UV plane stride.
    pub uv_stride: usize,
}

impl Default for PredBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl PredBuffer {
    /// Creates a new empty prediction buffer.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            y: Vec::new(),
            u: Vec::new(),
            v: Vec::new(),
            width: 0,
            height: 0,
            y_stride: 0,
            uv_stride: 0,
        }
    }

    /// Creates a prediction buffer with the given dimensions.
    #[must_use]
    pub fn with_size(width: usize, height: usize) -> Self {
        let y_stride = width;
        let uv_stride = width / 2;
        let uv_height = height / 2;

        Self {
            y: vec![128; width * height],
            u: vec![128; uv_stride * uv_height],
            v: vec![128; uv_stride * uv_height],
            width,
            height,
            y_stride,
            uv_stride,
        }
    }

    /// Creates a prediction buffer for a specific block size.
    #[must_use]
    pub fn for_block(block_size: BlockSize) -> Self {
        Self::with_size(block_size.width(), block_size.height())
    }

    /// Resizes the buffer if necessary.
    pub fn resize(&mut self, width: usize, height: usize) {
        if self.width != width || self.height != height {
            self.width = width;
            self.height = height;
            self.y_stride = width;
            self.uv_stride = width / 2;

            let uv_height = height / 2;
            self.y.resize(width * height, 128);
            self.u.resize(self.uv_stride * uv_height, 128);
            self.v.resize(self.uv_stride * uv_height, 128);
        }
    }

    /// Fills the Y plane with a value.
    pub fn fill_y(&mut self, value: u8) {
        self.y.fill(value);
    }

    /// Fills the U plane with a value.
    pub fn fill_u(&mut self, value: u8) {
        self.u.fill(value);
    }

    /// Fills the V plane with a value.
    pub fn fill_v(&mut self, value: u8) {
        self.v.fill(value);
    }

    /// Fills all planes with a value.
    pub fn fill(&mut self, value: u8) {
        self.fill_y(value);
        self.fill_u(value);
        self.fill_v(value);
    }

    /// Returns a reference to a row in the Y plane.
    #[must_use]
    pub fn y_row(&self, row: usize) -> &[u8] {
        let start = row * self.y_stride;
        let end = start + self.width;
        &self.y[start..end]
    }

    /// Returns a mutable reference to a row in the Y plane.
    pub fn y_row_mut(&mut self, row: usize) -> &mut [u8] {
        let start = row * self.y_stride;
        let end = start + self.width;
        &mut self.y[start..end]
    }

    /// Returns a reference to a row in the U plane.
    #[must_use]
    pub fn u_row(&self, row: usize) -> &[u8] {
        let start = row * self.uv_stride;
        let end = start + self.width / 2;
        &self.u[start..end]
    }

    /// Returns a mutable reference to a row in the U plane.
    pub fn u_row_mut(&mut self, row: usize) -> &mut [u8] {
        let start = row * self.uv_stride;
        let end = start + self.width / 2;
        &mut self.u[start..end]
    }

    /// Returns a reference to a row in the V plane.
    #[must_use]
    pub fn v_row(&self, row: usize) -> &[u8] {
        let start = row * self.uv_stride;
        let end = start + self.width / 2;
        &self.v[start..end]
    }

    /// Returns a mutable reference to a row in the V plane.
    pub fn v_row_mut(&mut self, row: usize) -> &mut [u8] {
        let start = row * self.uv_stride;
        let end = start + self.width / 2;
        &mut self.v[start..end]
    }

    /// Returns a pixel from the Y plane.
    #[must_use]
    pub fn y_pixel(&self, x: usize, y: usize) -> u8 {
        self.y[y * self.y_stride + x]
    }

    /// Sets a pixel in the Y plane.
    pub fn set_y_pixel(&mut self, x: usize, y: usize, value: u8) {
        self.y[y * self.y_stride + x] = value;
    }
}

/// Applies 8-tap horizontal interpolation filter.
///
/// # Arguments
///
/// * `src` - Source pixel buffer
/// * `src_stride` - Source buffer stride
/// * `filter` - Filter coefficients (8 taps)
/// * `width` - Output width
/// * `height` - Output height
/// * `output` - Output buffer
/// * `output_stride` - Output buffer stride
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn interp_horiz_8tap(
    src: &[u8],
    src_stride: usize,
    filter: &[i16; FILTER_TAPS],
    width: usize,
    height: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    for row in 0..height {
        let src_row = row * src_stride;
        let out_row = row * output_stride;

        for col in 0..width {
            let mut sum: i32 = 0;

            for (tap, &coef) in filter.iter().enumerate() {
                let src_col = col + tap;
                if src_col < src_stride {
                    sum += i32::from(src[src_row + src_col]) * i32::from(coef);
                }
            }

            // Round and clamp to 8-bit
            let result = ((sum + 64) >> 7).clamp(0, 255) as u8;
            output[out_row + col] = result;
        }
    }
}

/// Applies 8-tap vertical interpolation filter.
///
/// # Arguments
///
/// * `src` - Source pixel buffer
/// * `src_stride` - Source buffer stride
/// * `filter` - Filter coefficients (8 taps)
/// * `width` - Output width
/// * `height` - Output height
/// * `output` - Output buffer
/// * `output_stride` - Output buffer stride
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn interp_vert_8tap(
    src: &[u8],
    src_stride: usize,
    filter: &[i16; FILTER_TAPS],
    width: usize,
    height: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    for row in 0..height {
        let out_row = row * output_stride;

        for col in 0..width {
            let mut sum: i32 = 0;

            for (tap, &coef) in filter.iter().enumerate() {
                let src_row = row + tap;
                if src_row * src_stride + col < src.len() {
                    sum += i32::from(src[src_row * src_stride + col]) * i32::from(coef);
                }
            }

            // Round and clamp to 8-bit
            let result = ((sum + 64) >> 7).clamp(0, 255) as u8;
            output[out_row + col] = result;
        }
    }
}

/// Applies bilinear horizontal interpolation filter.
///
/// # Arguments
///
/// * `src` - Source pixel buffer
/// * `src_stride` - Source buffer stride
/// * `filter` - Filter coefficients (2 taps)
/// * `width` - Output width
/// * `height` - Output height
/// * `output` - Output buffer
/// * `output_stride` - Output buffer stride
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn interp_horiz_bilinear(
    src: &[u8],
    src_stride: usize,
    filter: &[i16; 2],
    width: usize,
    height: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    for row in 0..height {
        let src_row = row * src_stride;
        let out_row = row * output_stride;

        for col in 0..width {
            let p0 = i32::from(src[src_row + col]);
            let p1 = if col + 1 < src_stride {
                i32::from(src[src_row + col + 1])
            } else {
                p0
            };

            let sum = p0 * i32::from(filter[0]) + p1 * i32::from(filter[1]);
            let result = ((sum + 64) >> 7).clamp(0, 255) as u8;
            output[out_row + col] = result;
        }
    }
}

/// Applies bilinear vertical interpolation filter.
///
/// # Arguments
///
/// * `src` - Source pixel buffer
/// * `src_stride` - Source buffer stride
/// * `filter` - Filter coefficients (2 taps)
/// * `width` - Output width
/// * `height` - Output height
/// * `output` - Output buffer
/// * `output_stride` - Output buffer stride
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn interp_vert_bilinear(
    src: &[u8],
    src_stride: usize,
    filter: &[i16; 2],
    width: usize,
    height: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    for row in 0..height {
        let src_row0 = row * src_stride;
        let src_row1 = (row + 1) * src_stride;
        let out_row = row * output_stride;

        for col in 0..width {
            let p0 = i32::from(src[src_row0 + col]);
            let p1 = if src_row1 + col < src.len() {
                i32::from(src[src_row1 + col])
            } else {
                p0
            };

            let sum = p0 * i32::from(filter[0]) + p1 * i32::from(filter[1]);
            let result = ((sum + 64) >> 7).clamp(0, 255) as u8;
            output[out_row + col] = result;
        }
    }
}

/// Performs subpixel interpolation in the horizontal direction.
///
/// # Arguments
///
/// * `src` - Source pixel buffer (with margin for filter taps)
/// * `src_stride` - Source buffer stride
/// * `filter_type` - Interpolation filter type
/// * `subpel_x` - Horizontal subpixel offset (0-7)
/// * `width` - Output width
/// * `height` - Output height
/// * `output` - Output buffer
/// * `output_stride` - Output buffer stride
pub fn subpel_interp_horiz(
    src: &[u8],
    src_stride: usize,
    filter_type: InterpFilter,
    subpel_x: usize,
    width: usize,
    height: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    if subpel_x == 0 {
        // No horizontal interpolation needed, just copy
        for row in 0..height {
            let src_start = row * src_stride;
            let out_start = row * output_stride;
            output[out_start..out_start + width]
                .copy_from_slice(&src[src_start..src_start + width]);
        }
        return;
    }

    let coeffs = get_filter_coeffs(filter_type, subpel_x);

    if filter_type.is_bilinear() {
        if let Ok(f) = coeffs.try_into() {
            interp_horiz_bilinear(src, src_stride, f, width, height, output, output_stride);
        }
    } else if let Ok(f) = coeffs.try_into() {
        interp_horiz_8tap(src, src_stride, f, width, height, output, output_stride);
    }
}

/// Performs subpixel interpolation in the vertical direction.
///
/// # Arguments
///
/// * `src` - Source pixel buffer (with margin for filter taps)
/// * `src_stride` - Source buffer stride
/// * `filter_type` - Interpolation filter type
/// * `subpel_y` - Vertical subpixel offset (0-7)
/// * `width` - Output width
/// * `height` - Output height
/// * `output` - Output buffer
/// * `output_stride` - Output buffer stride
pub fn subpel_interp_vert(
    src: &[u8],
    src_stride: usize,
    filter_type: InterpFilter,
    subpel_y: usize,
    width: usize,
    height: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    if subpel_y == 0 {
        // No vertical interpolation needed, just copy
        for row in 0..height {
            let src_start = row * src_stride;
            let out_start = row * output_stride;
            output[out_start..out_start + width]
                .copy_from_slice(&src[src_start..src_start + width]);
        }
        return;
    }

    let coeffs = get_filter_coeffs(filter_type, subpel_y);

    if filter_type.is_bilinear() {
        if let Ok(f) = coeffs.try_into() {
            interp_vert_bilinear(src, src_stride, f, width, height, output, output_stride);
        }
    } else if let Ok(f) = coeffs.try_into() {
        interp_vert_8tap(src, src_stride, f, width, height, output, output_stride);
    }
}

/// Performs 2D subpixel interpolation (horizontal then vertical).
///
/// # Arguments
///
/// * `src` - Source pixel buffer (with margin for filter taps)
/// * `src_stride` - Source buffer stride
/// * `filter_type` - Interpolation filter type
/// * `subpel_x` - Horizontal subpixel offset (0-7)
/// * `subpel_y` - Vertical subpixel offset (0-7)
/// * `width` - Output width
/// * `height` - Output height
/// * `output` - Output buffer
/// * `output_stride` - Output buffer stride
pub fn subpel_interp_2d(
    src: &[u8],
    src_stride: usize,
    filter_type: InterpFilter,
    subpel_x: usize,
    subpel_y: usize,
    width: usize,
    height: usize,
    output: &mut [u8],
    output_stride: usize,
) {
    if subpel_x == 0 && subpel_y == 0 {
        // No interpolation needed
        for row in 0..height {
            let src_start = row * src_stride;
            let out_start = row * output_stride;
            output[out_start..out_start + width]
                .copy_from_slice(&src[src_start..src_start + width]);
        }
    } else if subpel_y == 0 {
        // Horizontal only
        subpel_interp_horiz(
            src,
            src_stride,
            filter_type,
            subpel_x,
            width,
            height,
            output,
            output_stride,
        );
    } else if subpel_x == 0 {
        // Vertical only
        subpel_interp_vert(
            src,
            src_stride,
            filter_type,
            subpel_y,
            width,
            height,
            output,
            output_stride,
        );
    } else {
        // Both horizontal and vertical
        // First apply horizontal filter to intermediate buffer
        let tap_offset = filter_type.tap_offset();
        let temp_height = height + tap_offset * 2;
        let temp_stride = width;
        let mut temp = vec![0u8; temp_stride * temp_height];

        subpel_interp_horiz(
            src,
            src_stride,
            filter_type,
            subpel_x,
            width,
            temp_height,
            &mut temp,
            temp_stride,
        );

        // Then apply vertical filter
        subpel_interp_vert(
            &temp,
            temp_stride,
            filter_type,
            subpel_y,
            width,
            height,
            output,
            output_stride,
        );
    }
}

/// Blends two prediction buffers for compound prediction.
///
/// The blend weights are 50/50 for simple averaging.
///
/// # Arguments
///
/// * `pred0` - First prediction buffer
/// * `pred1` - Second prediction buffer
/// * `output` - Output buffer
/// * `width` - Width in pixels
/// * `height` - Height in pixels
/// * `stride` - Buffer stride
#[allow(clippy::cast_possible_truncation)]
pub fn blend_predictions(
    pred0: &[u8],
    pred1: &[u8],
    output: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) {
    for row in 0..height {
        let row_start = row * stride;
        for col in 0..width {
            let idx = row_start + col;
            let p0 = i16::from(pred0[idx]);
            let p1 = i16::from(pred1[idx]);
            output[idx] = ((p0 + p1 + 1) >> 1) as u8;
        }
    }
}

/// Blends two prediction buffers with weighted averaging.
///
/// # Arguments
///
/// * `pred0` - First prediction buffer
/// * `pred1` - Second prediction buffer
/// * `weight0` - Weight for first prediction (0-64)
/// * `weight1` - Weight for second prediction (0-64)
/// * `output` - Output buffer
/// * `width` - Width in pixels
/// * `height` - Height in pixels
/// * `stride` - Buffer stride
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn blend_weighted(
    pred0: &[u8],
    pred1: &[u8],
    weight0: u8,
    weight1: u8,
    output: &mut [u8],
    width: usize,
    height: usize,
    stride: usize,
) {
    let w0 = i32::from(weight0);
    let w1 = i32::from(weight1);

    for row in 0..height {
        let row_start = row * stride;
        for col in 0..width {
            let idx = row_start + col;
            let p0 = i32::from(pred0[idx]);
            let p1 = i32::from(pred1[idx]);
            let weighted = (p0 * w0 + p1 * w1 + 32) >> 6;
            output[idx] = weighted.clamp(0, 255) as u8;
        }
    }
}

/// Context for inter prediction operations.
#[derive(Clone, Debug)]
pub struct InterPrediction {
    /// Prediction buffer for first reference.
    pub pred0: PredBuffer,
    /// Prediction buffer for second reference (compound).
    pub pred1: PredBuffer,
    /// Output buffer.
    pub output: PredBuffer,
    /// Interpolation filter for Y plane.
    pub filter_y: InterpFilter,
    /// Interpolation filter for UV planes.
    pub filter_uv: InterpFilter,
}

impl Default for InterPrediction {
    fn default() -> Self {
        Self::new()
    }
}

impl InterPrediction {
    /// Creates a new inter prediction context.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            pred0: PredBuffer::new(),
            pred1: PredBuffer::new(),
            output: PredBuffer::new(),
            filter_y: InterpFilter::EightTap,
            filter_uv: InterpFilter::EightTap,
        }
    }

    /// Sets the interpolation filters.
    pub fn set_filters(&mut self, filter_y: InterpFilter, filter_uv: InterpFilter) {
        self.filter_y = filter_y;
        self.filter_uv = filter_uv;
    }

    /// Prepares buffers for a block prediction.
    pub fn prepare(&mut self, block_size: BlockSize) {
        let width = block_size.width();
        let height = block_size.height();

        self.pred0.resize(width, height);
        self.pred1.resize(width, height);
        self.output.resize(width, height);
    }

    /// Blends predictions for compound mode.
    pub fn blend_compound(&mut self, width: usize, height: usize) {
        blend_predictions(
            &self.pred0.y,
            &self.pred1.y,
            &mut self.output.y,
            width,
            height,
            width,
        );

        let uv_width = width / 2;
        let uv_height = height / 2;
        blend_predictions(
            &self.pred0.u,
            &self.pred1.u,
            &mut self.output.u,
            uv_width,
            uv_height,
            uv_width,
        );
        blend_predictions(
            &self.pred0.v,
            &self.pred1.v,
            &mut self.output.v,
            uv_width,
            uv_height,
            uv_width,
        );
    }
}

/// Applies inter prediction for a block (skeleton implementation).
///
/// This function performs motion-compensated prediction using reference frames.
///
/// # Arguments
///
/// * `ctx` - Inter prediction context with mode and motion vector info
/// * `ref_y` - Reference Y plane data
/// * `ref_stride` - Reference plane stride
/// * `scaling` - Reference frame scaling factors
/// * `filter` - Interpolation filter type
/// * `output` - Output prediction buffer
/// * `output_stride` - Output buffer stride
///
/// # Note
///
/// This is a skeleton implementation that demonstrates the interface.
/// A complete implementation would fetch pixels from reference frames
/// and apply subpixel interpolation.
#[allow(clippy::too_many_arguments)]
pub fn apply_inter_prediction(
    ctx: &InterPredContext,
    ref_y: &[u8],
    ref_stride: usize,
    scaling: &ScalingFactors,
    filter: InterpFilter,
    output: &mut [u8],
    output_stride: usize,
) {
    let width = ctx.width();
    let height = ctx.height();

    // Get motion vector (or zero for intra)
    let mv = ctx.mode.mv0().unwrap_or(MotionVector::zero());

    // Scale motion vector if needed
    let scaled_mv = scaling.scale_mv(mv);

    // Get subpixel offsets
    let subpel_x = (scaled_mv.col & 7) as usize;
    let subpel_y = (scaled_mv.row & 7) as usize;

    // Get integer pixel offsets
    let ref_x = ctx.pixel_x() as i32 + (scaled_mv.col >> 3) as i32;
    let ref_y_pos = ctx.pixel_y() as i32 + (scaled_mv.row >> 3) as i32;

    // Bounds check and clamp
    let ref_x = ref_x.max(0) as usize;
    let ref_y_pos = ref_y_pos.max(0) as usize;

    // Calculate source offset
    let src_offset = ref_y_pos * ref_stride + ref_x;

    // Ensure we have enough source data
    if src_offset + (height - 1) * ref_stride + width <= ref_y.len() {
        // Perform subpixel interpolation
        subpel_interp_2d(
            &ref_y[src_offset..],
            ref_stride,
            filter,
            subpel_x,
            subpel_y,
            width,
            height,
            output,
            output_stride,
        );
    } else {
        // Fill with neutral value if out of bounds
        for row in 0..height {
            for col in 0..width {
                output[row * output_stride + col] = 128;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interp_filter() {
        assert_eq!(InterpFilter::EightTap.index(), 0);
        assert_eq!(InterpFilter::Bilinear.index(), 3);

        assert!(InterpFilter::Bilinear.is_bilinear());
        assert!(!InterpFilter::EightTap.is_bilinear());

        assert!(InterpFilter::EightTap.is_eight_tap());
        assert!(InterpFilter::EightTapSmooth.is_eight_tap());
        assert!(InterpFilter::EightTapSharp.is_eight_tap());
        assert!(!InterpFilter::Bilinear.is_eight_tap());

        assert_eq!(InterpFilter::Bilinear.num_taps(), 2);
        assert_eq!(InterpFilter::EightTap.num_taps(), 8);
    }

    #[test]
    fn test_interp_filter_from_u8() {
        assert_eq!(InterpFilter::from_u8(0), Some(InterpFilter::EightTap));
        assert_eq!(InterpFilter::from_u8(3), Some(InterpFilter::Bilinear));
        assert_eq!(InterpFilter::from_u8(4), None);
    }

    #[test]
    fn test_filter_coeffs() {
        // At subpel 0, all filters should have weight at center tap only
        let coeffs_8tap = get_filter_coeffs(InterpFilter::EightTap, 0);
        assert_eq!(coeffs_8tap[3], 128);
        assert_eq!(coeffs_8tap[0], 0);

        let coeffs_bilinear = get_filter_coeffs(InterpFilter::Bilinear, 0);
        assert_eq!(coeffs_bilinear[0], 128);
        assert_eq!(coeffs_bilinear[1], 0);
    }

    #[test]
    fn test_pred_buffer_new() {
        let buf = PredBuffer::new();
        assert_eq!(buf.width, 0);
        assert_eq!(buf.height, 0);
        assert!(buf.y.is_empty());
    }

    #[test]
    fn test_pred_buffer_with_size() {
        let buf = PredBuffer::with_size(16, 16);
        assert_eq!(buf.width, 16);
        assert_eq!(buf.height, 16);
        assert_eq!(buf.y.len(), 256);
        assert_eq!(buf.u.len(), 64);
        assert_eq!(buf.v.len(), 64);
    }

    #[test]
    fn test_pred_buffer_for_block() {
        let buf = PredBuffer::for_block(BlockSize::Block32x32);
        assert_eq!(buf.width, 32);
        assert_eq!(buf.height, 32);
    }

    #[test]
    fn test_pred_buffer_resize() {
        let mut buf = PredBuffer::with_size(8, 8);
        assert_eq!(buf.width, 8);

        buf.resize(16, 16);
        assert_eq!(buf.width, 16);
        assert_eq!(buf.y.len(), 256);
    }

    #[test]
    fn test_pred_buffer_fill() {
        let mut buf = PredBuffer::with_size(4, 4);
        buf.fill(100);

        assert!(buf.y.iter().all(|&x| x == 100));
        assert!(buf.u.iter().all(|&x| x == 100));
        assert!(buf.v.iter().all(|&x| x == 100));
    }

    #[test]
    fn test_pred_buffer_rows() {
        let mut buf = PredBuffer::with_size(4, 4);

        buf.y_row_mut(0).copy_from_slice(&[10, 20, 30, 40]);
        assert_eq!(buf.y_row(0), &[10, 20, 30, 40]);

        buf.u_row_mut(0).copy_from_slice(&[50, 60]);
        assert_eq!(buf.u_row(0), &[50, 60]);
    }

    #[test]
    fn test_pred_buffer_pixel() {
        let mut buf = PredBuffer::with_size(4, 4);
        buf.set_y_pixel(2, 1, 123);
        assert_eq!(buf.y_pixel(2, 1), 123);
    }

    #[test]
    fn test_blend_predictions() {
        let pred0 = vec![100u8; 16];
        let pred1 = vec![200u8; 16];
        let mut output = vec![0u8; 16];

        blend_predictions(&pred0, &pred1, &mut output, 4, 4, 4);

        // (100 + 200 + 1) / 2 = 150
        assert!(output.iter().all(|&x| x == 150));
    }

    #[test]
    fn test_blend_weighted() {
        let pred0 = vec![0u8; 16];
        let pred1 = vec![128u8; 16];
        let mut output = vec![0u8; 16];

        // 75% pred0, 25% pred1
        blend_weighted(&pred0, &pred1, 48, 16, &mut output, 4, 4, 4);

        // (0 * 48 + 128 * 16 + 32) / 64 = 32
        assert!(output.iter().all(|&x| x == 32));
    }

    #[test]
    fn test_subpel_interp_horiz_no_subpel() {
        let src = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let mut output = vec![0u8; 4];

        subpel_interp_horiz(&src, 8, InterpFilter::EightTap, 0, 4, 1, &mut output, 4);

        assert_eq!(output, &[10, 20, 30, 40]);
    }

    #[test]
    fn test_subpel_interp_vert_no_subpel() {
        // Source is 4 wide, 2 rows:
        // Row 0: [10, 20, 30, 40]
        // Row 1: [50, 60, 70, 80]
        let src = vec![10, 20, 30, 40, 50, 60, 70, 80];
        let mut output = vec![0u8; 4];

        // Copy width=2, height=2 with output_stride=2
        subpel_interp_vert(&src, 4, InterpFilter::EightTap, 0, 2, 2, &mut output, 2);

        // Expected: [10, 20] from row 0, [50, 60] from row 1
        assert_eq!(output, &[10, 20, 50, 60]);
    }

    #[test]
    fn test_subpel_interp_2d_no_subpel() {
        let src = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160,
        ];
        let mut output = vec![0u8; 16];

        subpel_interp_2d(&src, 4, InterpFilter::EightTap, 0, 0, 4, 4, &mut output, 4);

        assert_eq!(output, src);
    }

    #[test]
    fn test_inter_prediction_new() {
        let pred = InterPrediction::new();
        assert_eq!(pred.filter_y, InterpFilter::EightTap);
        assert_eq!(pred.filter_uv, InterpFilter::EightTap);
    }

    #[test]
    fn test_inter_prediction_prepare() {
        let mut pred = InterPrediction::new();
        pred.prepare(BlockSize::Block16x16);

        assert_eq!(pred.pred0.width, 16);
        assert_eq!(pred.pred0.height, 16);
        assert_eq!(pred.output.width, 16);
    }

    #[test]
    fn test_inter_prediction_set_filters() {
        let mut pred = InterPrediction::new();
        pred.set_filters(InterpFilter::Bilinear, InterpFilter::EightTapSmooth);

        assert_eq!(pred.filter_y, InterpFilter::Bilinear);
        assert_eq!(pred.filter_uv, InterpFilter::EightTapSmooth);
    }

    #[test]
    fn test_inter_prediction_blend() {
        let mut pred = InterPrediction::new();
        pred.prepare(BlockSize::Block8x8);

        pred.pred0.fill_y(100);
        pred.pred1.fill_y(200);

        pred.blend_compound(8, 8);

        // Check that output was blended
        assert!(pred.output.y.iter().all(|&x| x == 150));
    }
}
