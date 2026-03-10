//! Frame buffer management for decoded video frames.
//!
//! This module provides buffer types for storing decoded video data,
//! including frame buffers, plane buffers, reference frame management,
//! and a buffer pool for efficient memory reuse.

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
#![allow(clippy::many_single_char_names)]
#![allow(clippy::manual_div_ceil)]
#![allow(clippy::too_many_arguments)]

use super::{ChromaSubsampling, PlaneType, ReconstructResult, ReconstructionError, NUM_REF_FRAMES};
use std::collections::VecDeque;

// =============================================================================
// Plane Buffer
// =============================================================================

/// Buffer for a single video plane (Y, U, or V).
#[derive(Clone, Debug)]
pub struct PlaneBuffer {
    /// Pixel data stored as i16 for intermediate precision.
    data: Vec<i16>,
    /// Width in pixels.
    width: u32,
    /// Height in pixels.
    height: u32,
    /// Row stride in samples.
    stride: usize,
    /// Bit depth.
    bit_depth: u8,
    /// Plane type.
    plane_type: PlaneType,
}

impl PlaneBuffer {
    /// Create a new plane buffer.
    #[must_use]
    pub fn new(width: u32, height: u32, bit_depth: u8, plane_type: PlaneType) -> Self {
        // Add padding for SIMD alignment (16-byte aligned, 8 i16 values)
        let aligned_width = ((width as usize + 7) / 8) * 8;
        let stride = aligned_width;
        let size = stride * height as usize;

        Self {
            data: vec![0i16; size],
            width,
            height,
            stride,
            bit_depth,
            plane_type,
        }
    }

    /// Create a plane buffer with custom stride.
    #[must_use]
    pub fn with_stride(
        width: u32,
        height: u32,
        stride: usize,
        bit_depth: u8,
        plane_type: PlaneType,
    ) -> Self {
        let size = stride * height as usize;

        Self {
            data: vec![0i16; size],
            width,
            height,
            stride,
            bit_depth,
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

    /// Get the bit depth.
    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// Get the plane type.
    #[must_use]
    pub const fn plane_type(&self) -> PlaneType {
        self.plane_type
    }

    /// Get the maximum pixel value for this bit depth.
    #[must_use]
    pub fn max_value(&self) -> i16 {
        (1i16 << self.bit_depth) - 1
    }

    /// Get a pixel value at (x, y).
    #[must_use]
    pub fn get(&self, x: u32, y: u32) -> i16 {
        if x >= self.width || y >= self.height {
            return 0;
        }
        let idx = y as usize * self.stride + x as usize;
        self.data.get(idx).copied().unwrap_or(0)
    }

    /// Set a pixel value at (x, y).
    pub fn set(&mut self, x: u32, y: u32, value: i16) {
        if x < self.width && y < self.height {
            let idx = y as usize * self.stride + x as usize;
            if idx < self.data.len() {
                self.data[idx] = value;
            }
        }
    }

    /// Set a pixel value with clamping to valid range.
    pub fn set_clamped(&mut self, x: u32, y: u32, value: i16) {
        let max_val = self.max_value();
        let clamped = value.clamp(0, max_val);
        self.set(x, y, clamped);
    }

    /// Get a row of pixels.
    #[must_use]
    pub fn row(&self, y: u32) -> &[i16] {
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

    /// Get a mutable row of pixels.
    pub fn row_mut(&mut self, y: u32) -> &mut [i16] {
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

    /// Get the raw data slice.
    #[must_use]
    pub fn data(&self) -> &[i16] {
        &self.data
    }

    /// Get the raw data as mutable slice.
    pub fn data_mut(&mut self) -> &mut [i16] {
        &mut self.data
    }

    /// Clear the buffer to zero.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    /// Fill the buffer with a constant value.
    pub fn fill(&mut self, value: i16) {
        self.data.fill(value);
    }

    /// Copy from another plane buffer.
    ///
    /// # Errors
    ///
    /// Returns error if dimensions don't match.
    pub fn copy_from(&mut self, other: &PlaneBuffer) -> ReconstructResult<()> {
        if self.width != other.width || self.height != other.height {
            return Err(ReconstructionError::InvalidDimensions {
                width: other.width,
                height: other.height,
            });
        }

        if self.stride == other.stride {
            self.data.copy_from_slice(&other.data);
        } else {
            let copy_width = self.width as usize;
            for y in 0..self.height {
                let src_row = other.row(y);
                let dst_row = self.row_mut(y);
                dst_row[..copy_width].copy_from_slice(&src_row[..copy_width]);
            }
        }

        Ok(())
    }

    /// Copy a block from another plane buffer.
    pub fn copy_block_from(
        &mut self,
        other: &PlaneBuffer,
        src_x: u32,
        src_y: u32,
        dst_x: u32,
        dst_y: u32,
        width: u32,
        height: u32,
    ) {
        for dy in 0..height {
            for dx in 0..width {
                let value = other.get(src_x + dx, src_y + dy);
                self.set(dst_x + dx, dst_y + dy, value);
            }
        }
    }

    /// Convert to 8-bit output buffer.
    #[must_use]
    pub fn to_u8(&self) -> Vec<u8> {
        let mut output = Vec::with_capacity(self.width as usize * self.height as usize);

        for y in 0..self.height {
            let row = self.row(y);
            for &pixel in row.iter().take(self.width as usize) {
                let value = if self.bit_depth <= 8 {
                    pixel.clamp(0, 255) as u8
                } else {
                    // Scale down from higher bit depth
                    let shift = self.bit_depth - 8;
                    ((pixel >> shift).clamp(0, 255)) as u8
                };
                output.push(value);
            }
        }

        output
    }

    /// Calculate the sum of absolute differences (SAD) with another plane.
    #[must_use]
    pub fn sad(&self, other: &PlaneBuffer, block_x: u32, block_y: u32, size: u32) -> u32 {
        let mut sad: u32 = 0;
        for dy in 0..size {
            for dx in 0..size {
                let a = self.get(block_x + dx, block_y + dy);
                let b = other.get(block_x + dx, block_y + dy);
                sad += (a - b).unsigned_abs() as u32;
            }
        }
        sad
    }
}

// =============================================================================
// Frame Buffer
// =============================================================================

/// Buffer for a complete video frame (all planes).
#[derive(Clone, Debug)]
pub struct FrameBuffer {
    /// Y (luma) plane.
    y_plane: PlaneBuffer,
    /// U (chroma blue) plane.
    u_plane: Option<PlaneBuffer>,
    /// V (chroma red) plane.
    v_plane: Option<PlaneBuffer>,
    /// Frame width.
    width: u32,
    /// Frame height.
    height: u32,
    /// Bit depth.
    bit_depth: u8,
    /// Chroma subsampling.
    subsampling: ChromaSubsampling,
    /// Frame timestamp.
    timestamp: i64,
    /// Frame is a keyframe.
    is_keyframe: bool,
    /// Unique buffer ID.
    id: u64,
}

impl FrameBuffer {
    /// Create a new frame buffer.
    #[must_use]
    pub fn new(width: u32, height: u32, bit_depth: u8, subsampling: ChromaSubsampling) -> Self {
        static BUFFER_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

        let y_plane = PlaneBuffer::new(width, height, bit_depth, PlaneType::Y);

        let (u_plane, v_plane) = match subsampling {
            ChromaSubsampling::Mono => (None, None),
            _ => {
                let (cw, ch) = subsampling.chroma_size(width, height);
                (
                    Some(PlaneBuffer::new(cw, ch, bit_depth, PlaneType::U)),
                    Some(PlaneBuffer::new(cw, ch, bit_depth, PlaneType::V)),
                )
            }
        };

        Self {
            y_plane,
            u_plane,
            v_plane,
            width,
            height,
            bit_depth,
            subsampling,
            timestamp: 0,
            is_keyframe: false,
            id: BUFFER_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed),
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

    /// Get the bit depth.
    #[must_use]
    pub const fn bit_depth(&self) -> u8 {
        self.bit_depth
    }

    /// Get the chroma subsampling.
    #[must_use]
    pub const fn subsampling(&self) -> ChromaSubsampling {
        self.subsampling
    }

    /// Get the buffer ID.
    #[must_use]
    pub const fn id(&self) -> u64 {
        self.id
    }

    /// Get the timestamp.
    #[must_use]
    pub const fn timestamp(&self) -> i64 {
        self.timestamp
    }

    /// Set the timestamp.
    pub fn set_timestamp(&mut self, timestamp: i64) {
        self.timestamp = timestamp;
    }

    /// Check if this is a keyframe.
    #[must_use]
    pub const fn is_keyframe(&self) -> bool {
        self.is_keyframe
    }

    /// Set keyframe flag.
    pub fn set_keyframe(&mut self, is_keyframe: bool) {
        self.is_keyframe = is_keyframe;
    }

    /// Get the Y plane.
    #[must_use]
    pub fn y_plane(&self) -> &PlaneBuffer {
        &self.y_plane
    }

    /// Get the Y plane mutably.
    pub fn y_plane_mut(&mut self) -> &mut PlaneBuffer {
        &mut self.y_plane
    }

    /// Get the U plane.
    #[must_use]
    pub fn u_plane(&self) -> Option<&PlaneBuffer> {
        self.u_plane.as_ref()
    }

    /// Get the U plane mutably.
    pub fn u_plane_mut(&mut self) -> Option<&mut PlaneBuffer> {
        self.u_plane.as_mut()
    }

    /// Get the V plane.
    #[must_use]
    pub fn v_plane(&self) -> Option<&PlaneBuffer> {
        self.v_plane.as_ref()
    }

    /// Get the V plane mutably.
    pub fn v_plane_mut(&mut self) -> Option<&mut PlaneBuffer> {
        self.v_plane.as_mut()
    }

    /// Get a plane by type.
    #[must_use]
    pub fn plane(&self, plane_type: PlaneType) -> Option<&PlaneBuffer> {
        match plane_type {
            PlaneType::Y => Some(&self.y_plane),
            PlaneType::U => self.u_plane.as_ref(),
            PlaneType::V => self.v_plane.as_ref(),
        }
    }

    /// Get a plane mutably by type.
    pub fn plane_mut(&mut self, plane_type: PlaneType) -> Option<&mut PlaneBuffer> {
        match plane_type {
            PlaneType::Y => Some(&mut self.y_plane),
            PlaneType::U => self.u_plane.as_mut(),
            PlaneType::V => self.v_plane.as_mut(),
        }
    }

    /// Get the number of planes.
    #[must_use]
    pub const fn num_planes(&self) -> usize {
        self.subsampling.num_planes()
    }

    /// Clear all planes.
    pub fn clear(&mut self) {
        self.y_plane.clear();
        if let Some(ref mut u) = self.u_plane {
            u.clear();
        }
        if let Some(ref mut v) = self.v_plane {
            v.clear();
        }
    }

    /// Copy from another frame buffer.
    ///
    /// # Errors
    ///
    /// Returns error if dimensions or format don't match.
    pub fn copy_from(&mut self, other: &FrameBuffer) -> ReconstructResult<()> {
        if self.width != other.width
            || self.height != other.height
            || self.subsampling != other.subsampling
        {
            return Err(ReconstructionError::InvalidDimensions {
                width: other.width,
                height: other.height,
            });
        }

        self.y_plane.copy_from(&other.y_plane)?;

        if let (Some(ref mut dst), Some(ref src)) = (&mut self.u_plane, &other.u_plane) {
            dst.copy_from(src)?;
        }

        if let (Some(ref mut dst), Some(ref src)) = (&mut self.v_plane, &other.v_plane) {
            dst.copy_from(src)?;
        }

        Ok(())
    }

    /// Calculate total buffer size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        let y_size = self.y_plane.data().len() * 2; // i16 = 2 bytes
        let u_size = self.u_plane.as_ref().map_or(0, |p| p.data().len() * 2);
        let v_size = self.v_plane.as_ref().map_or(0, |p| p.data().len() * 2);
        y_size + u_size + v_size
    }
}

// =============================================================================
// Reference Frame Manager
// =============================================================================

/// Manages reference frames for inter prediction.
#[derive(Debug)]
pub struct ReferenceFrameManager {
    /// Reference frame slots.
    frames: [Option<FrameBuffer>; NUM_REF_FRAMES],
    /// Frame order hints for each slot.
    order_hints: [u32; NUM_REF_FRAMES],
    /// Current frame number.
    current_frame: u64,
}

impl Default for ReferenceFrameManager {
    fn default() -> Self {
        Self::new()
    }
}

impl ReferenceFrameManager {
    /// Create a new reference frame manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            frames: Default::default(),
            order_hints: [0; NUM_REF_FRAMES],
            current_frame: 0,
        }
    }

    /// Get a reference frame by slot index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&FrameBuffer> {
        self.frames.get(index).and_then(|f| f.as_ref())
    }

    /// Store a frame in a slot.
    pub fn store(&mut self, index: usize, frame: FrameBuffer, order_hint: u32) {
        if index < NUM_REF_FRAMES {
            self.frames[index] = Some(frame);
            self.order_hints[index] = order_hint;
        }
    }

    /// Clear a reference slot.
    pub fn clear_slot(&mut self, index: usize) {
        if index < NUM_REF_FRAMES {
            self.frames[index] = None;
            self.order_hints[index] = 0;
        }
    }

    /// Clear all reference frames.
    pub fn clear_all(&mut self) {
        for i in 0..NUM_REF_FRAMES {
            self.frames[i] = None;
            self.order_hints[i] = 0;
        }
        self.current_frame = 0;
    }

    /// Get the order hint for a slot.
    #[must_use]
    pub fn order_hint(&self, index: usize) -> u32 {
        self.order_hints.get(index).copied().unwrap_or(0)
    }

    /// Check if a slot has a valid reference.
    #[must_use]
    pub fn has_reference(&self, index: usize) -> bool {
        self.frames.get(index).is_some_and(|f| f.is_some())
    }

    /// Get the current frame number.
    #[must_use]
    pub const fn current_frame(&self) -> u64 {
        self.current_frame
    }

    /// Increment the frame counter.
    pub fn next_frame(&mut self) {
        self.current_frame += 1;
    }

    /// Get all valid reference indices.
    #[must_use]
    pub fn valid_references(&self) -> Vec<usize> {
        (0..NUM_REF_FRAMES)
            .filter(|&i| self.has_reference(i))
            .collect()
    }
}

// =============================================================================
// Buffer Pool
// =============================================================================

/// Pool of reusable frame buffers.
#[derive(Debug)]
pub struct BufferPool {
    /// Available buffers.
    available: VecDeque<FrameBuffer>,
    /// Buffer dimensions.
    width: u32,
    /// Buffer height.
    height: u32,
    /// Bit depth.
    bit_depth: u8,
    /// Chroma subsampling.
    subsampling: ChromaSubsampling,
    /// Maximum pool size.
    max_size: usize,
}

impl BufferPool {
    /// Create a new buffer pool.
    #[must_use]
    pub fn new(
        width: u32,
        height: u32,
        bit_depth: u8,
        subsampling: ChromaSubsampling,
        max_size: usize,
    ) -> Self {
        Self {
            available: VecDeque::with_capacity(max_size),
            width,
            height,
            bit_depth,
            subsampling,
            max_size,
        }
    }

    /// Acquire a buffer from the pool.
    ///
    /// Returns a buffer from the pool if available, otherwise allocates a new one.
    ///
    /// # Errors
    ///
    /// Returns error if allocation fails.
    pub fn acquire(&mut self) -> ReconstructResult<FrameBuffer> {
        if let Some(mut buffer) = self.available.pop_front() {
            buffer.clear();
            Ok(buffer)
        } else {
            Ok(FrameBuffer::new(
                self.width,
                self.height,
                self.bit_depth,
                self.subsampling,
            ))
        }
    }

    /// Release a buffer back to the pool.
    pub fn release(&mut self, buffer: FrameBuffer) {
        // Only keep buffers that match current dimensions
        if buffer.width == self.width
            && buffer.height == self.height
            && buffer.bit_depth == self.bit_depth
            && self.available.len() < self.max_size
        {
            self.available.push_back(buffer);
        }
    }

    /// Get the number of available buffers.
    #[must_use]
    pub fn available_count(&self) -> usize {
        self.available.len()
    }

    /// Check if the pool is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.available.is_empty()
    }

    /// Reset the pool, clearing all buffers.
    pub fn reset(&mut self) {
        self.available.clear();
    }

    /// Reconfigure the pool for new dimensions.
    pub fn reconfigure(
        &mut self,
        width: u32,
        height: u32,
        bit_depth: u8,
        subsampling: ChromaSubsampling,
    ) {
        self.width = width;
        self.height = height;
        self.bit_depth = bit_depth;
        self.subsampling = subsampling;
        self.available.clear();
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_plane_buffer_new() {
        let plane = PlaneBuffer::new(64, 48, 8, PlaneType::Y);
        assert_eq!(plane.width(), 64);
        assert_eq!(plane.height(), 48);
        assert_eq!(plane.bit_depth(), 8);
        assert_eq!(plane.plane_type(), PlaneType::Y);
    }

    #[test]
    fn test_plane_buffer_get_set() {
        let mut plane = PlaneBuffer::new(64, 48, 8, PlaneType::Y);
        plane.set(10, 20, 128);
        assert_eq!(plane.get(10, 20), 128);
        assert_eq!(plane.get(0, 0), 0);
    }

    #[test]
    fn test_plane_buffer_set_clamped() {
        let mut plane = PlaneBuffer::new(64, 48, 8, PlaneType::Y);
        plane.set_clamped(10, 20, 300);
        assert_eq!(plane.get(10, 20), 255);

        plane.set_clamped(10, 20, -50);
        assert_eq!(plane.get(10, 20), 0);
    }

    #[test]
    fn test_plane_buffer_row() {
        let mut plane = PlaneBuffer::new(64, 48, 8, PlaneType::Y);
        plane.set(5, 10, 100);
        let row = plane.row(10);
        assert_eq!(row[5], 100);
    }

    #[test]
    fn test_plane_buffer_max_value() {
        let plane8 = PlaneBuffer::new(64, 48, 8, PlaneType::Y);
        assert_eq!(plane8.max_value(), 255);

        let plane10 = PlaneBuffer::new(64, 48, 10, PlaneType::Y);
        assert_eq!(plane10.max_value(), 1023);

        let plane12 = PlaneBuffer::new(64, 48, 12, PlaneType::Y);
        assert_eq!(plane12.max_value(), 4095);
    }

    #[test]
    fn test_plane_buffer_copy_from() {
        let mut src = PlaneBuffer::new(64, 48, 8, PlaneType::Y);
        src.set(10, 20, 128);

        let mut dst = PlaneBuffer::new(64, 48, 8, PlaneType::Y);
        dst.copy_from(&src).expect("should succeed");

        assert_eq!(dst.get(10, 20), 128);
    }

    #[test]
    fn test_plane_buffer_to_u8() {
        let mut plane = PlaneBuffer::new(4, 4, 8, PlaneType::Y);
        plane.set(0, 0, 128);
        plane.set(1, 0, 255);

        let output = plane.to_u8();
        assert_eq!(output.len(), 16);
        assert_eq!(output[0], 128);
        assert_eq!(output[1], 255);
    }

    #[test]
    fn test_frame_buffer_new() {
        let frame = FrameBuffer::new(1920, 1080, 8, ChromaSubsampling::Cs420);
        assert_eq!(frame.width(), 1920);
        assert_eq!(frame.height(), 1080);
        assert_eq!(frame.bit_depth(), 8);
        assert_eq!(frame.num_planes(), 3);
    }

    #[test]
    fn test_frame_buffer_mono() {
        let frame = FrameBuffer::new(1920, 1080, 8, ChromaSubsampling::Mono);
        assert_eq!(frame.num_planes(), 1);
        assert!(frame.u_plane().is_none());
        assert!(frame.v_plane().is_none());
    }

    #[test]
    fn test_frame_buffer_planes() {
        let mut frame = FrameBuffer::new(64, 48, 8, ChromaSubsampling::Cs420);

        frame.y_plane_mut().set(10, 10, 100);
        assert_eq!(frame.y_plane().get(10, 10), 100);

        if let Some(u) = frame.u_plane_mut() {
            u.set(5, 5, 50);
        }
        assert_eq!(
            frame.u_plane().expect("get should return value").get(5, 5),
            50
        );
    }

    #[test]
    fn test_frame_buffer_copy_from() {
        let mut src = FrameBuffer::new(64, 48, 8, ChromaSubsampling::Cs420);
        src.y_plane_mut().set(10, 10, 100);

        let mut dst = FrameBuffer::new(64, 48, 8, ChromaSubsampling::Cs420);
        dst.copy_from(&src).expect("should succeed");

        assert_eq!(dst.y_plane().get(10, 10), 100);
    }

    #[test]
    fn test_reference_frame_manager() {
        let mut mgr = ReferenceFrameManager::new();

        assert!(!mgr.has_reference(0));

        let frame = FrameBuffer::new(64, 48, 8, ChromaSubsampling::Cs420);
        mgr.store(0, frame, 1);

        assert!(mgr.has_reference(0));
        assert_eq!(mgr.order_hint(0), 1);

        let refs = mgr.valid_references();
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0], 0);
    }

    #[test]
    fn test_reference_frame_manager_clear() {
        let mut mgr = ReferenceFrameManager::new();

        let frame = FrameBuffer::new(64, 48, 8, ChromaSubsampling::Cs420);
        mgr.store(0, frame, 1);

        mgr.clear_all();
        assert!(!mgr.has_reference(0));
    }

    #[test]
    fn test_buffer_pool() {
        let mut pool = BufferPool::new(64, 48, 8, ChromaSubsampling::Cs420, 4);

        assert!(pool.is_empty());

        let buf1 = pool.acquire().expect("should succeed");
        assert_eq!(buf1.width(), 64);

        pool.release(buf1);
        assert_eq!(pool.available_count(), 1);

        let buf2 = pool.acquire().expect("should succeed");
        assert!(pool.is_empty());
        assert_eq!(buf2.width(), 64);
    }

    #[test]
    fn test_buffer_pool_max_size() {
        let mut pool = BufferPool::new(64, 48, 8, ChromaSubsampling::Cs420, 2);

        let buf1 = pool.acquire().expect("should succeed");
        let buf2 = pool.acquire().expect("should succeed");
        let buf3 = pool.acquire().expect("should succeed");

        pool.release(buf1);
        pool.release(buf2);
        pool.release(buf3); // This one should be dropped (exceeds max_size)

        assert_eq!(pool.available_count(), 2);
    }

    #[test]
    fn test_buffer_pool_reconfigure() {
        let mut pool = BufferPool::new(64, 48, 8, ChromaSubsampling::Cs420, 4);

        let buf = pool.acquire().expect("should succeed");
        pool.release(buf);

        pool.reconfigure(128, 96, 10, ChromaSubsampling::Cs422);
        assert!(pool.is_empty());

        let new_buf = pool.acquire().expect("should succeed");
        assert_eq!(new_buf.width(), 128);
        assert_eq!(new_buf.height(), 96);
        assert_eq!(new_buf.bit_depth(), 10);
    }
}
