//! Residual buffer management for transform coefficients.
//!
//! This module provides buffers for storing and manipulating transform
//! coefficients (residuals) during video decoding. Residuals are the
//! difference between the prediction and the actual pixel values.

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
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::manual_div_ceil)]

use super::{
    ChromaSubsampling, FrameBuffer, PlaneBuffer, PlaneType, ReconstructResult, ReconstructionError,
};

// =============================================================================
// Constants
// =============================================================================

/// Maximum transform size.
pub const MAX_TX_SIZE: usize = 64;

/// Minimum transform size.
pub const MIN_TX_SIZE: usize = 4;

/// Maximum coefficient value (for 12-bit).
pub const MAX_COEFF_VALUE: i32 = 32767;

/// Minimum coefficient value.
pub const MIN_COEFF_VALUE: i32 = -32768;

// =============================================================================
// Transform Block
// =============================================================================

/// Transform block containing coefficients.
#[derive(Clone, Debug)]
pub struct TransformBlock {
    /// Coefficient data in scan order.
    coeffs: Vec<i32>,
    /// Block width.
    width: usize,
    /// Block height.
    height: usize,
    /// End of block position (index of last non-zero coefficient + 1).
    eob: usize,
    /// Transform type.
    tx_type: TransformType,
}

impl TransformBlock {
    /// Create a new transform block.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        let size = width * height;
        Self {
            coeffs: vec![0; size],
            width,
            height,
            eob: 0,
            tx_type: TransformType::Dct,
        }
    }

    /// Get the block width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Get the block height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Get the end of block position.
    #[must_use]
    pub const fn eob(&self) -> usize {
        self.eob
    }

    /// Set the end of block position.
    pub fn set_eob(&mut self, eob: usize) {
        self.eob = eob;
    }

    /// Get the transform type.
    #[must_use]
    pub const fn tx_type(&self) -> TransformType {
        self.tx_type
    }

    /// Set the transform type.
    pub fn set_tx_type(&mut self, tx_type: TransformType) {
        self.tx_type = tx_type;
    }

    /// Get coefficient at position.
    #[must_use]
    pub fn get(&self, row: usize, col: usize) -> i32 {
        if row < self.height && col < self.width {
            self.coeffs[row * self.width + col]
        } else {
            0
        }
    }

    /// Set coefficient at position.
    pub fn set(&mut self, row: usize, col: usize, value: i32) {
        if row < self.height && col < self.width {
            self.coeffs[row * self.width + col] = value;
        }
    }

    /// Set coefficient with clamping.
    pub fn set_clamped(&mut self, row: usize, col: usize, value: i32) {
        let clamped = value.clamp(MIN_COEFF_VALUE, MAX_COEFF_VALUE);
        self.set(row, col, clamped);
    }

    /// Get coefficients as slice.
    #[must_use]
    pub fn coeffs(&self) -> &[i32] {
        &self.coeffs
    }

    /// Get coefficients as mutable slice.
    pub fn coeffs_mut(&mut self) -> &mut [i32] {
        &mut self.coeffs
    }

    /// Clear all coefficients.
    pub fn clear(&mut self) {
        self.coeffs.fill(0);
        self.eob = 0;
    }

    /// Check if the block is all zeros.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.eob == 0
    }

    /// Count non-zero coefficients.
    #[must_use]
    pub fn count_nonzero(&self) -> usize {
        self.coeffs.iter().filter(|&&c| c != 0).count()
    }

    /// Get coefficient at scan position.
    #[must_use]
    pub fn get_scan(&self, scan_pos: usize) -> i32 {
        if scan_pos < self.coeffs.len() {
            self.coeffs[scan_pos]
        } else {
            0
        }
    }

    /// Set coefficient at scan position.
    pub fn set_scan(&mut self, scan_pos: usize, value: i32) {
        if scan_pos < self.coeffs.len() {
            self.coeffs[scan_pos] = value;
        }
    }
}

/// Transform type for AV1/VP9.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransformType {
    /// Discrete Cosine Transform.
    #[default]
    Dct,
    /// Asymmetric Discrete Sine Transform.
    Adst,
    /// Identity transform.
    Identity,
    /// Flip DCT (reversed order).
    FlipDct,
    /// Flip ADST.
    FlipAdst,
}

impl TransformType {
    /// Get the horizontal transform type for a 2D transform.
    #[must_use]
    pub const fn horizontal(self) -> Self {
        self
    }

    /// Get the vertical transform type for a 2D transform.
    #[must_use]
    pub const fn vertical(self) -> Self {
        self
    }

    /// Check if this is an identity transform.
    #[must_use]
    pub const fn is_identity(self) -> bool {
        matches!(self, Self::Identity)
    }
}

// =============================================================================
// Residual Plane
// =============================================================================

/// Buffer for residuals of a single plane.
#[derive(Clone, Debug)]
pub struct ResidualPlane {
    /// Residual data.
    data: Vec<i32>,
    /// Plane width.
    width: u32,
    /// Plane height.
    height: u32,
    /// Row stride.
    stride: usize,
    /// Plane type.
    plane_type: PlaneType,
}

impl ResidualPlane {
    /// Create a new residual plane.
    #[must_use]
    pub fn new(width: u32, height: u32, plane_type: PlaneType) -> Self {
        // Align stride to 16 for SIMD
        let stride = ((width as usize + 15) / 16) * 16;
        let size = stride * height as usize;

        Self {
            data: vec![0; size],
            width,
            height,
            stride,
            plane_type,
        }
    }

    /// Get the plane width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Get the plane height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Get the row stride.
    #[must_use]
    pub const fn stride(&self) -> usize {
        self.stride
    }

    /// Get the plane type.
    #[must_use]
    pub const fn plane_type(&self) -> PlaneType {
        self.plane_type
    }

    /// Get a residual value at (x, y).
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> i32 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        let idx = y as usize * self.stride + x as usize;
        self.data.get(idx).copied().unwrap_or(0)
    }

    /// Set a residual value at (x, y).
    pub fn set(&mut self, x: u32, y: u32, value: i32) {
        if x < self.width && y < self.height {
            let idx = y as usize * self.stride + x as usize;
            if idx < self.data.len() {
                self.data[idx] = value;
            }
        }
    }

    /// Get a row of residuals.
    #[must_use]
    pub fn row(&self, y: u32) -> &[i32] {
        if y >= self.height {
            return &[];
        }
        let start = y as usize * self.stride;
        let end = start + self.width as usize;
        if end <= self.data.len() {
            &self.data[start..end]
        } else {
            &[]
        }
    }

    /// Get a mutable row of residuals.
    pub fn row_mut(&mut self, y: u32) -> &mut [i32] {
        if y >= self.height {
            return &mut [];
        }
        let start = y as usize * self.stride;
        let end = start + self.width as usize;
        if end <= self.data.len() {
            &mut self.data[start..end]
        } else {
            &mut []
        }
    }

    /// Get raw data slice.
    #[must_use]
    pub fn data(&self) -> &[i32] {
        &self.data
    }

    /// Get raw data as mutable slice.
    pub fn data_mut(&mut self) -> &mut [i32] {
        &mut self.data
    }

    /// Clear all residuals to zero.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Clear a block region.
    pub fn clear_block(&mut self, x: u32, y: u32, width: u32, height: u32) {
        for dy in 0..height {
            for dx in 0..width {
                self.set(x + dx, y + dy, 0);
            }
        }
    }

    /// Write a transform block to the residual plane.
    pub fn write_block(&mut self, x: u32, y: u32, block: &TransformBlock) {
        for row in 0..block.height() {
            for col in 0..block.width() {
                let value = block.get(row, col);
                self.set(x + col as u32, y + row as u32, value);
            }
        }
    }

    /// Read a transform block from the residual plane.
    #[must_use]
    pub fn read_block(&self, x: u32, y: u32, width: usize, height: usize) -> TransformBlock {
        let mut block = TransformBlock::new(width, height);
        for row in 0..height {
            for col in 0..width {
                let value = self.get(x + col as u32, y + row as u32);
                block.set(row, col, value);
            }
        }
        block
    }
}

// =============================================================================
// Residual Buffer
// =============================================================================

/// Buffer for all residual planes in a frame.
#[derive(Clone, Debug)]
pub struct ResidualBuffer {
    /// Y plane residuals.
    y_plane: ResidualPlane,
    /// U plane residuals.
    u_plane: Option<ResidualPlane>,
    /// V plane residuals.
    v_plane: Option<ResidualPlane>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Chroma subsampling.
    subsampling: ChromaSubsampling,
}

impl ResidualBuffer {
    /// Create a new residual buffer.
    #[must_use]
    pub fn new(width: u32, height: u32, subsampling: ChromaSubsampling) -> Self {
        let y_plane = ResidualPlane::new(width, height, PlaneType::Y);

        let (u_plane, v_plane) = match subsampling {
            ChromaSubsampling::Mono => (None, None),
            _ => {
                let (cw, ch) = subsampling.chroma_size(width, height);
                (
                    Some(ResidualPlane::new(cw, ch, PlaneType::U)),
                    Some(ResidualPlane::new(cw, ch, PlaneType::V)),
                )
            }
        };

        Self {
            y_plane,
            u_plane,
            v_plane,
            width,
            height,
            subsampling,
        }
    }

    /// Get the frame width.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Get the frame height.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Get the Y plane.
    #[must_use]
    pub fn y_plane(&self) -> &ResidualPlane {
        &self.y_plane
    }

    /// Get the Y plane mutably.
    pub fn y_plane_mut(&mut self) -> &mut ResidualPlane {
        &mut self.y_plane
    }

    /// Get the U plane.
    #[must_use]
    pub fn u_plane(&self) -> Option<&ResidualPlane> {
        self.u_plane.as_ref()
    }

    /// Get the U plane mutably.
    pub fn u_plane_mut(&mut self) -> Option<&mut ResidualPlane> {
        self.u_plane.as_mut()
    }

    /// Get the V plane.
    #[must_use]
    pub fn v_plane(&self) -> Option<&ResidualPlane> {
        self.v_plane.as_ref()
    }

    /// Get the V plane mutably.
    pub fn v_plane_mut(&mut self) -> Option<&mut ResidualPlane> {
        self.v_plane.as_mut()
    }

    /// Get a plane by type.
    #[must_use]
    pub fn plane(&self, plane_type: PlaneType) -> Option<&ResidualPlane> {
        match plane_type {
            PlaneType::Y => Some(&self.y_plane),
            PlaneType::U => self.u_plane.as_ref(),
            PlaneType::V => self.v_plane.as_ref(),
        }
    }

    /// Get a plane mutably by type.
    pub fn plane_mut(&mut self, plane_type: PlaneType) -> Option<&mut ResidualPlane> {
        match plane_type {
            PlaneType::Y => Some(&mut self.y_plane),
            PlaneType::U => self.u_plane.as_mut(),
            PlaneType::V => self.v_plane.as_mut(),
        }
    }

    /// Clear all residual planes.
    pub fn clear(&mut self) {
        self.y_plane.clear();
        if let Some(ref mut u) = self.u_plane {
            u.clear();
        }
        if let Some(ref mut v) = self.v_plane {
            v.clear();
        }
    }

    /// Add residuals to a frame buffer.
    ///
    /// # Errors
    ///
    /// Returns error if dimensions don't match.
    pub fn add_to_frame(&self, frame: &mut FrameBuffer) -> ReconstructResult<()> {
        if frame.width() != self.width || frame.height() != self.height {
            return Err(ReconstructionError::InvalidDimensions {
                width: self.width,
                height: self.height,
            });
        }

        // Add Y plane residuals
        add_residual_plane(&self.y_plane, frame.y_plane_mut());

        // Add U plane residuals
        if let (Some(ref resid), Some(ref mut plane)) = (&self.u_plane, frame.u_plane_mut()) {
            add_residual_plane(resid, plane);
        }

        // Add V plane residuals
        if let (Some(ref resid), Some(ref mut plane)) = (&self.v_plane, frame.v_plane_mut()) {
            add_residual_plane(resid, plane);
        }

        Ok(())
    }

    /// Add residuals to a frame buffer for a specific block.
    pub fn add_block_to_frame(
        &self,
        frame: &mut FrameBuffer,
        plane: PlaneType,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) -> ReconstructResult<()> {
        let resid_plane = self
            .plane(plane)
            .ok_or_else(|| ReconstructionError::InvalidInput("Plane not available".to_string()))?;
        let frame_plane = frame
            .plane_mut(plane)
            .ok_or_else(|| ReconstructionError::InvalidInput("Plane not available".to_string()))?;

        let max_val = frame_plane.max_value();

        for dy in 0..height {
            for dx in 0..width {
                let px = x + dx;
                let py = y + dy;

                let pixel = i32::from(frame_plane.get(px, py));
                let residual = resid_plane.get(px, py);
                let result = (pixel + residual).clamp(0, i32::from(max_val));

                frame_plane.set(px, py, result as i16);
            }
        }

        Ok(())
    }
}

/// Add residuals from a residual plane to a pixel plane.
fn add_residual_plane(residual: &ResidualPlane, pixels: &mut PlaneBuffer) {
    let max_val = pixels.max_value();
    let width = residual.width().min(pixels.width());
    let height = residual.height().min(pixels.height());

    for y in 0..height {
        let resid_row = residual.row(y);
        let pixel_row = pixels.row_mut(y);

        for x in 0..width as usize {
            let pixel = i32::from(pixel_row[x]);
            let resid = resid_row[x];
            let result = (pixel + resid).clamp(0, i32::from(max_val));
            pixel_row[x] = result as i16;
        }
    }
}

// =============================================================================
// Residual Operations
// =============================================================================

/// Add a residual block to prediction.
///
/// # Arguments
///
/// * `prediction` - The prediction buffer (modified in place).
/// * `residual` - The residual values.
/// * `width` - Block width.
/// * `height` - Block height.
/// * `bit_depth` - Bit depth for clamping.
pub fn add_residual(
    prediction: &mut [i16],
    residual: &[i32],
    width: usize,
    height: usize,
    bit_depth: u8,
) {
    let max_val = (1i32 << bit_depth) - 1;
    let size = width * height;

    for i in 0..size.min(prediction.len()).min(residual.len()) {
        let pred = i32::from(prediction[i]);
        let resid = residual[i];
        let result = (pred + resid).clamp(0, max_val);
        prediction[i] = result as i16;
    }
}

/// Add residual with stride.
pub fn add_residual_stride(
    prediction: &mut [i16],
    pred_stride: usize,
    residual: &[i32],
    resid_stride: usize,
    width: usize,
    height: usize,
    bit_depth: u8,
) {
    let max_val = (1i32 << bit_depth) - 1;

    for y in 0..height {
        let pred_row = &mut prediction[y * pred_stride..];
        let resid_row = &residual[y * resid_stride..];

        for x in 0..width {
            let pred = i32::from(pred_row[x]);
            let resid = resid_row[x];
            let result = (pred + resid).clamp(0, max_val);
            pred_row[x] = result as i16;
        }
    }
}

/// Clip residual values to valid range.
pub fn clip_residuals(residuals: &mut [i32], bit_depth: u8) {
    // Residuals can be in range [-(1<<bd), (1<<bd)-1]
    let max = (1i32 << bit_depth) - 1;
    let min = -(1i32 << bit_depth);

    for r in residuals.iter_mut() {
        *r = (*r).clamp(min, max);
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transform_block_new() {
        let block = TransformBlock::new(8, 8);
        assert_eq!(block.width(), 8);
        assert_eq!(block.height(), 8);
        assert_eq!(block.eob(), 0);
        assert!(block.is_zero());
    }

    #[test]
    fn test_transform_block_get_set() {
        let mut block = TransformBlock::new(8, 8);
        block.set(2, 3, 100);
        block.set_eob(25);

        assert_eq!(block.get(2, 3), 100);
        assert_eq!(block.eob(), 25);
        assert!(!block.is_zero());
    }

    #[test]
    fn test_transform_block_count_nonzero() {
        let mut block = TransformBlock::new(4, 4);
        block.set(0, 0, 10);
        block.set(1, 1, 20);
        block.set(2, 2, 30);

        assert_eq!(block.count_nonzero(), 3);
    }

    #[test]
    fn test_transform_type() {
        assert!(!TransformType::Dct.is_identity());
        assert!(TransformType::Identity.is_identity());
    }

    #[test]
    fn test_residual_plane_new() {
        let plane = ResidualPlane::new(64, 48, PlaneType::Y);
        assert_eq!(plane.width(), 64);
        assert_eq!(plane.height(), 48);
        assert_eq!(plane.plane_type(), PlaneType::Y);
    }

    #[test]
    fn test_residual_plane_get_set() {
        let mut plane = ResidualPlane::new(64, 48, PlaneType::Y);
        plane.set(10, 20, 500);
        assert_eq!(plane.get(10, 20), 500);
        assert_eq!(plane.get(0, 0), 0);
    }

    #[test]
    fn test_residual_plane_write_read_block() {
        let mut plane = ResidualPlane::new(64, 48, PlaneType::Y);

        let mut block = TransformBlock::new(4, 4);
        block.set(0, 0, 100);
        block.set(1, 1, 200);
        block.set(3, 3, 300);

        plane.write_block(8, 8, &block);

        let read_block = plane.read_block(8, 8, 4, 4);
        assert_eq!(read_block.get(0, 0), 100);
        assert_eq!(read_block.get(1, 1), 200);
        assert_eq!(read_block.get(3, 3), 300);
    }

    #[test]
    fn test_residual_buffer_new() {
        let buffer = ResidualBuffer::new(1920, 1080, ChromaSubsampling::Cs420);
        assert_eq!(buffer.width(), 1920);
        assert_eq!(buffer.height(), 1080);

        assert!(buffer.u_plane().is_some());
        assert!(buffer.v_plane().is_some());
    }

    #[test]
    fn test_residual_buffer_mono() {
        let buffer = ResidualBuffer::new(1920, 1080, ChromaSubsampling::Mono);
        assert!(buffer.u_plane().is_none());
        assert!(buffer.v_plane().is_none());
    }

    #[test]
    fn test_residual_buffer_clear() {
        let mut buffer = ResidualBuffer::new(64, 48, ChromaSubsampling::Cs420);
        buffer.y_plane_mut().set(10, 10, 1000);

        buffer.clear();
        assert_eq!(buffer.y_plane().get(10, 10), 0);
    }

    #[test]
    fn test_add_residual_to_frame() {
        let mut frame = FrameBuffer::new(64, 48, 8, ChromaSubsampling::Cs420);
        let mut residual = ResidualBuffer::new(64, 48, ChromaSubsampling::Cs420);

        // Set prediction value
        frame.y_plane_mut().set(10, 10, 100);

        // Set residual
        residual.y_plane_mut().set(10, 10, 50);

        // Add residual
        residual.add_to_frame(&mut frame).expect("should succeed");

        // Check result
        assert_eq!(frame.y_plane().get(10, 10), 150);
    }

    #[test]
    fn test_add_residual_clamping() {
        let mut frame = FrameBuffer::new(64, 48, 8, ChromaSubsampling::Cs420);
        let mut residual = ResidualBuffer::new(64, 48, ChromaSubsampling::Cs420);

        // Set prediction near max
        frame.y_plane_mut().set(10, 10, 250);

        // Set large positive residual
        residual.y_plane_mut().set(10, 10, 100);

        // Add residual (should clamp to 255)
        residual.add_to_frame(&mut frame).expect("should succeed");

        assert_eq!(frame.y_plane().get(10, 10), 255);
    }

    #[test]
    fn test_add_residual_function() {
        let mut prediction = vec![100i16, 150, 200, 50];
        let residual = vec![20i32, -30, 100, -100];

        add_residual(&mut prediction, &residual, 2, 2, 8);

        assert_eq!(prediction[0], 120); // 100 + 20
        assert_eq!(prediction[1], 120); // 150 - 30
        assert_eq!(prediction[2], 255); // 200 + 100, clamped
        assert_eq!(prediction[3], 0); // 50 - 100, clamped
    }

    #[test]
    fn test_clip_residuals() {
        let mut residuals = vec![100, -100, 500, -500];
        clip_residuals(&mut residuals, 8);

        assert_eq!(residuals[0], 100);
        assert_eq!(residuals[1], -100);
        assert_eq!(residuals[2], 255); // Clamped to max
        assert_eq!(residuals[3], -256); // Clamped to min
    }

    #[test]
    fn test_constants() {
        assert_eq!(MAX_TX_SIZE, 64);
        assert_eq!(MIN_TX_SIZE, 4);
        assert_eq!(MAX_COEFF_VALUE, 32767);
        assert_eq!(MIN_COEFF_VALUE, -32768);
    }
}
