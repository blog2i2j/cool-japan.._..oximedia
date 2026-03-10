//! VP9 Reference frame management.
//!
//! This module provides reference frame structures and management for VP9
//! decoding. VP9 maintains a pool of reference frames that can be used
//! for inter prediction.
//!
//! Reference frames:
//! - LAST: The most recently decoded frame
//! - GOLDEN: A golden (high quality) reference frame
//! - ALTREF: An alternate reference frame (often future frame in GOP)

#![forbid(unsafe_code)]
#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::similar_names)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::bool_to_int_with_if)]

use super::inter::{RefFrameType, ScalingFactors, INTER_REFS, REF_FRAMES};

/// Number of reference frame buffer slots.
pub const REF_BUFFER_COUNT: usize = 8;

/// Maximum frame width.
pub const MAX_FRAME_WIDTH: usize = 4096;

/// Maximum frame height.
pub const MAX_FRAME_HEIGHT: usize = 2304;

/// Reference frame buffer.
///
/// Stores decoded frame data for use as a reference in inter prediction.
#[derive(Clone, Debug)]
pub struct RefFrameBuffer {
    /// Y plane data.
    pub y: Vec<u8>,
    /// U plane data.
    pub u: Vec<u8>,
    /// V plane data.
    pub v: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Y plane stride.
    pub y_stride: usize,
    /// UV plane stride.
    pub uv_stride: usize,
    /// Frame number (for debugging).
    pub frame_num: u64,
    /// Reference count (number of active references).
    pub ref_count: u32,
    /// Whether this buffer is valid (contains decoded data).
    pub valid: bool,
}

impl Default for RefFrameBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl RefFrameBuffer {
    /// Creates a new empty reference frame buffer.
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
            frame_num: 0,
            ref_count: 0,
            valid: false,
        }
    }

    /// Allocates buffer for the given frame dimensions.
    pub fn allocate(&mut self, width: u32, height: u32) {
        let w = width as usize;
        let h = height as usize;

        // Add margin for interpolation filters
        let margin = 16;
        let y_stride = w + margin * 2;
        let uv_stride = (w / 2) + margin;
        let y_height = h + margin * 2;
        let uv_height = (h / 2) + margin;

        self.y.resize(y_stride * y_height, 128);
        self.u.resize(uv_stride * uv_height, 128);
        self.v.resize(uv_stride * uv_height, 128);

        self.width = width;
        self.height = height;
        self.y_stride = y_stride;
        self.uv_stride = uv_stride;
    }

    /// Clears the buffer data.
    pub fn clear(&mut self) {
        self.y.fill(128);
        self.u.fill(128);
        self.v.fill(128);
        self.valid = false;
    }

    /// Marks the buffer as valid with a frame number.
    pub fn set_valid(&mut self, frame_num: u64) {
        self.frame_num = frame_num;
        self.valid = true;
    }

    /// Adds a reference to this buffer.
    pub fn add_ref(&mut self) {
        self.ref_count = self.ref_count.saturating_add(1);
    }

    /// Removes a reference from this buffer.
    pub fn remove_ref(&mut self) {
        self.ref_count = self.ref_count.saturating_sub(1);
    }

    /// Returns true if this buffer has no references.
    #[must_use]
    pub const fn is_unreferenced(&self) -> bool {
        self.ref_count == 0
    }

    /// Returns the Y plane pixel at the given position.
    #[must_use]
    pub fn y_pixel(&self, x: usize, y: usize) -> u8 {
        let margin = 16;
        let idx = (y + margin) * self.y_stride + (x + margin);
        self.y.get(idx).copied().unwrap_or(128)
    }

    /// Returns the U plane pixel at the given position.
    #[must_use]
    pub fn u_pixel(&self, x: usize, y: usize) -> u8 {
        let margin = 8;
        let idx = (y + margin) * self.uv_stride + (x + margin);
        self.u.get(idx).copied().unwrap_or(128)
    }

    /// Returns the V plane pixel at the given position.
    #[must_use]
    pub fn v_pixel(&self, x: usize, y: usize) -> u8 {
        let margin = 8;
        let idx = (y + margin) * self.uv_stride + (x + margin);
        self.v.get(idx).copied().unwrap_or(128)
    }

    /// Returns a reference to a row in the Y plane.
    #[must_use]
    pub fn y_row(&self, y: usize) -> &[u8] {
        let margin = 16;
        let start = (y + margin) * self.y_stride + margin;
        let end = start + self.width as usize;
        &self.y[start..end.min(self.y.len())]
    }

    /// Returns a mutable reference to a row in the Y plane.
    pub fn y_row_mut(&mut self, y: usize) -> &mut [u8] {
        let margin = 16;
        let start = (y + margin) * self.y_stride + margin;
        let end = start + self.width as usize;
        let len = self.y.len();
        &mut self.y[start..end.min(len)]
    }

    /// Copies Y plane data from a source buffer.
    pub fn copy_y_from(&mut self, src: &[u8], src_stride: usize, width: usize, height: usize) {
        let margin = 16;
        for row in 0..height {
            let src_start = row * src_stride;
            let dst_start = (row + margin) * self.y_stride + margin;
            let copy_len = width
                .min(src.len() - src_start)
                .min(self.y.len() - dst_start);
            self.y[dst_start..dst_start + copy_len]
                .copy_from_slice(&src[src_start..src_start + copy_len]);
        }
    }

    /// Copies U plane data from a source buffer.
    pub fn copy_u_from(&mut self, src: &[u8], src_stride: usize, width: usize, height: usize) {
        let margin = 8;
        for row in 0..height {
            let src_start = row * src_stride;
            let dst_start = (row + margin) * self.uv_stride + margin;
            let copy_len = width
                .min(src.len() - src_start)
                .min(self.u.len() - dst_start);
            self.u[dst_start..dst_start + copy_len]
                .copy_from_slice(&src[src_start..src_start + copy_len]);
        }
    }

    /// Copies V plane data from a source buffer.
    pub fn copy_v_from(&mut self, src: &[u8], src_stride: usize, width: usize, height: usize) {
        let margin = 8;
        for row in 0..height {
            let src_start = row * src_stride;
            let dst_start = (row + margin) * self.uv_stride + margin;
            let copy_len = width
                .min(src.len() - src_start)
                .min(self.v.len() - dst_start);
            self.v[dst_start..dst_start + copy_len]
                .copy_from_slice(&src[src_start..src_start + copy_len]);
        }
    }
}

/// Reference frame information.
///
/// Contains information about a reference frame including which buffer
/// it references and any scaling factors needed.
#[derive(Clone, Debug, Default)]
pub struct ReferenceFrame {
    /// Buffer index in the reference pool.
    pub buffer_index: usize,
    /// Scaling factors for this reference.
    pub scaling: ScalingFactors,
    /// Sign bias for motion vector prediction.
    pub sign_bias: bool,
    /// Whether this reference is valid.
    pub valid: bool,
}

impl ReferenceFrame {
    /// Creates a new invalid reference frame.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            buffer_index: 0,
            scaling: ScalingFactors::identity(),
            sign_bias: false,
            valid: false,
        }
    }

    /// Creates a valid reference frame with the given buffer index.
    #[must_use]
    pub const fn with_buffer(buffer_index: usize) -> Self {
        Self {
            buffer_index,
            scaling: ScalingFactors::identity(),
            sign_bias: false,
            valid: true,
        }
    }

    /// Sets the scaling factors.
    pub fn set_scaling(&mut self, scaling: ScalingFactors) {
        self.scaling = scaling;
    }

    /// Sets the sign bias.
    pub fn set_sign_bias(&mut self, bias: bool) {
        self.sign_bias = bias;
    }

    /// Marks this reference as valid.
    pub fn set_valid(&mut self, buffer_index: usize) {
        self.buffer_index = buffer_index;
        self.valid = true;
    }

    /// Marks this reference as invalid.
    pub fn invalidate(&mut self) {
        self.valid = false;
    }
}

/// Reference frame pool.
///
/// Manages a pool of reference frame buffers for VP9 decoding.
/// VP9 uses up to 8 buffer slots for reference frame management.
#[derive(Clone, Debug)]
pub struct ReferenceFramePool {
    /// Buffer pool.
    buffers: Vec<RefFrameBuffer>,
    /// Current reference frame assignments.
    /// Index: RefFrameType (Last, Golden, AltRef) -> buffer index
    ref_frame_map: [usize; INTER_REFS],
    /// Sign bias for each reference frame.
    sign_bias: [bool; REF_FRAMES],
    /// Reference frame information.
    refs: [ReferenceFrame; INTER_REFS],
    /// Current frame dimensions.
    width: u32,
    height: u32,
}

impl Default for ReferenceFramePool {
    fn default() -> Self {
        Self::new()
    }
}

impl ReferenceFramePool {
    /// Creates a new reference frame pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            buffers: (0..REF_BUFFER_COUNT)
                .map(|_| RefFrameBuffer::new())
                .collect(),
            ref_frame_map: [0; INTER_REFS],
            sign_bias: [false; REF_FRAMES],
            refs: [
                ReferenceFrame::new(),
                ReferenceFrame::new(),
                ReferenceFrame::new(),
            ],
            width: 0,
            height: 0,
        }
    }

    /// Sets the current frame dimensions.
    pub fn set_dimensions(&mut self, width: u32, height: u32) {
        self.width = width;
        self.height = height;
    }

    /// Allocates buffers for the current dimensions.
    pub fn allocate_buffers(&mut self) {
        for buffer in &mut self.buffers {
            if buffer.width != self.width || buffer.height != self.height {
                buffer.allocate(self.width, self.height);
            }
        }
    }

    /// Finds a free buffer slot.
    #[must_use]
    pub fn find_free_buffer(&self) -> Option<usize> {
        self.buffers
            .iter()
            .position(|buf| buf.is_unreferenced() || !buf.valid)
    }

    /// Gets a reference to a buffer.
    #[must_use]
    pub fn get_buffer(&self, index: usize) -> Option<&RefFrameBuffer> {
        self.buffers.get(index)
    }

    /// Gets a mutable reference to a buffer.
    pub fn get_buffer_mut(&mut self, index: usize) -> Option<&mut RefFrameBuffer> {
        self.buffers.get_mut(index)
    }

    /// Gets the buffer for a reference frame type.
    #[must_use]
    pub fn get_ref_buffer(&self, ref_type: RefFrameType) -> Option<&RefFrameBuffer> {
        match ref_type.inter_index() {
            Some(idx) => self.buffers.get(self.ref_frame_map[idx]),
            None => None,
        }
    }

    /// Gets the reference frame info.
    #[must_use]
    pub fn get_ref(&self, ref_type: RefFrameType) -> Option<&ReferenceFrame> {
        ref_type.inter_index().map(|idx| &self.refs[idx])
    }

    /// Sets the reference frame mapping.
    pub fn set_ref_frame(&mut self, ref_type: RefFrameType, buffer_index: usize) {
        if let Some(idx) = ref_type.inter_index() {
            // Remove reference from old buffer
            let old_buf_idx = self.ref_frame_map[idx];
            if let Some(buf) = self.buffers.get_mut(old_buf_idx) {
                buf.remove_ref();
            }

            // Add reference to new buffer
            self.ref_frame_map[idx] = buffer_index;
            if let Some(buf) = self.buffers.get_mut(buffer_index) {
                buf.add_ref();
            }

            // Update reference info
            self.refs[idx].set_valid(buffer_index);
        }
    }

    /// Sets the sign bias for a reference frame.
    pub fn set_sign_bias(&mut self, ref_type: RefFrameType, bias: bool) {
        self.sign_bias[ref_type.index()] = bias;
        if let Some(idx) = ref_type.inter_index() {
            self.refs[idx].set_sign_bias(bias);
        }
    }

    /// Gets the sign bias for a reference frame.
    #[must_use]
    pub const fn get_sign_bias(&self, ref_type: RefFrameType) -> bool {
        self.sign_bias[ref_type.index()]
    }

    /// Updates scaling factors for a reference frame.
    pub fn update_scaling(&mut self, ref_type: RefFrameType, ref_width: u32, ref_height: u32) {
        if let Some(idx) = ref_type.inter_index() {
            let scaling =
                ScalingFactors::from_dimensions(ref_width, ref_height, self.width, self.height);
            self.refs[idx].set_scaling(scaling);
        }
    }

    /// Invalidates all references.
    pub fn invalidate_all(&mut self) {
        for r in &mut self.refs {
            r.invalidate();
        }
        for buf in &mut self.buffers {
            buf.ref_count = 0;
            buf.valid = false;
        }
    }

    /// Returns true if a reference frame is valid.
    #[must_use]
    pub fn is_ref_valid(&self, ref_type: RefFrameType) -> bool {
        match ref_type.inter_index() {
            Some(idx) => {
                self.refs[idx].valid
                    && self
                        .buffers
                        .get(self.ref_frame_map[idx])
                        .is_some_and(|b| b.valid)
            }
            None => false,
        }
    }

    /// Gets the buffer index for a reference frame.
    #[must_use]
    pub fn get_buffer_index(&self, ref_type: RefFrameType) -> Option<usize> {
        ref_type.inter_index().map(|idx| self.ref_frame_map[idx])
    }

    /// Resets the pool.
    pub fn reset(&mut self) {
        self.invalidate_all();
        self.ref_frame_map = [0; INTER_REFS];
        self.sign_bias = [false; REF_FRAMES];
    }
}

/// Sign bias calculation for reference frames.
///
/// In VP9, sign bias is based on the order hint (temporal distance).
/// If a reference frame has a higher order hint than the current frame,
/// the sign bias is true (motion vectors are negated).
#[derive(Clone, Debug, Default)]
pub struct SignBiasInfo {
    /// Order hint for LAST reference.
    pub last_order_hint: u32,
    /// Order hint for GOLDEN reference.
    pub golden_order_hint: u32,
    /// Order hint for ALTREF reference.
    pub altref_order_hint: u32,
    /// Order hint for current frame.
    pub current_order_hint: u32,
}

impl SignBiasInfo {
    /// Creates a new sign bias info.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            last_order_hint: 0,
            golden_order_hint: 0,
            altref_order_hint: 0,
            current_order_hint: 0,
        }
    }

    /// Sets the current frame's order hint.
    pub fn set_current(&mut self, order_hint: u32) {
        self.current_order_hint = order_hint;
    }

    /// Sets the order hint for a reference frame.
    pub fn set_ref_order_hint(&mut self, ref_type: RefFrameType, order_hint: u32) {
        match ref_type {
            RefFrameType::Last => self.last_order_hint = order_hint,
            RefFrameType::Golden => self.golden_order_hint = order_hint,
            RefFrameType::AltRef => self.altref_order_hint = order_hint,
            RefFrameType::Intra => {}
        }
    }

    /// Computes the sign bias for a reference frame.
    #[must_use]
    pub const fn compute_sign_bias(&self, ref_type: RefFrameType) -> bool {
        let ref_hint = match ref_type {
            RefFrameType::Last => self.last_order_hint,
            RefFrameType::Golden => self.golden_order_hint,
            RefFrameType::AltRef => self.altref_order_hint,
            RefFrameType::Intra => return false,
        };

        // Sign bias is true if reference frame is "after" current frame
        ref_hint > self.current_order_hint
    }

    /// Computes all sign biases.
    #[must_use]
    pub const fn compute_all(&self) -> [bool; REF_FRAMES] {
        [
            false, // Intra
            self.compute_sign_bias(RefFrameType::Last),
            self.compute_sign_bias(RefFrameType::Golden),
            self.compute_sign_bias(RefFrameType::AltRef),
        ]
    }
}

/// Reference frame update flags.
///
/// These flags indicate which reference frames should be updated
/// after decoding the current frame.
#[derive(Clone, Copy, Debug, Default)]
pub struct RefUpdateFlags {
    /// Update the LAST reference.
    pub update_last: bool,
    /// Update the GOLDEN reference.
    pub update_golden: bool,
    /// Update the ALTREF reference.
    pub update_altref: bool,
}

impl RefUpdateFlags {
    /// Creates new update flags with all disabled.
    #[must_use]
    pub const fn none() -> Self {
        Self {
            update_last: false,
            update_golden: false,
            update_altref: false,
        }
    }

    /// Creates update flags for a keyframe (update all).
    #[must_use]
    pub const fn keyframe() -> Self {
        Self {
            update_last: true,
            update_golden: true,
            update_altref: true,
        }
    }

    /// Creates update flags from a bitmask.
    #[must_use]
    pub const fn from_bits(bits: u8) -> Self {
        Self {
            update_last: bits & 1 != 0,
            update_golden: bits & 2 != 0,
            update_altref: bits & 4 != 0,
        }
    }

    /// Converts to a bitmask.
    #[must_use]
    pub const fn to_bits(&self) -> u8 {
        let mut bits = 0;
        if self.update_last {
            bits |= 1;
        }
        if self.update_golden {
            bits |= 2;
        }
        if self.update_altref {
            bits |= 4;
        }
        bits
    }

    /// Returns true if any reference should be updated.
    #[must_use]
    pub const fn any(&self) -> bool {
        self.update_last || self.update_golden || self.update_altref
    }

    /// Returns true if a specific reference should be updated.
    #[must_use]
    pub const fn should_update(&self, ref_type: RefFrameType) -> bool {
        match ref_type {
            RefFrameType::Last => self.update_last,
            RefFrameType::Golden => self.update_golden,
            RefFrameType::AltRef => self.update_altref,
            RefFrameType::Intra => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ref_frame_buffer_new() {
        let buf = RefFrameBuffer::new();
        assert_eq!(buf.width, 0);
        assert_eq!(buf.height, 0);
        assert!(!buf.valid);
        assert_eq!(buf.ref_count, 0);
    }

    #[test]
    fn test_ref_frame_buffer_allocate() {
        let mut buf = RefFrameBuffer::new();
        buf.allocate(64, 64);

        assert_eq!(buf.width, 64);
        assert_eq!(buf.height, 64);
        assert!(!buf.y.is_empty());
        assert!(!buf.u.is_empty());
        assert!(!buf.v.is_empty());
    }

    #[test]
    fn test_ref_frame_buffer_clear() {
        let mut buf = RefFrameBuffer::new();
        buf.allocate(16, 16);
        buf.set_valid(1);
        assert!(buf.valid);

        buf.clear();
        assert!(!buf.valid);
    }

    #[test]
    fn test_ref_frame_buffer_ref_count() {
        let mut buf = RefFrameBuffer::new();
        assert!(buf.is_unreferenced());

        buf.add_ref();
        assert!(!buf.is_unreferenced());
        assert_eq!(buf.ref_count, 1);

        buf.add_ref();
        assert_eq!(buf.ref_count, 2);

        buf.remove_ref();
        assert_eq!(buf.ref_count, 1);

        buf.remove_ref();
        assert!(buf.is_unreferenced());
    }

    #[test]
    fn test_ref_frame_buffer_pixel_access() {
        let mut buf = RefFrameBuffer::new();
        buf.allocate(32, 32);

        // Default value should be 128
        assert_eq!(buf.y_pixel(0, 0), 128);
        assert_eq!(buf.u_pixel(0, 0), 128);
        assert_eq!(buf.v_pixel(0, 0), 128);
    }

    #[test]
    fn test_reference_frame_new() {
        let r = ReferenceFrame::new();
        assert!(!r.valid);
        assert!(!r.sign_bias);
    }

    #[test]
    fn test_reference_frame_with_buffer() {
        let r = ReferenceFrame::with_buffer(5);
        assert!(r.valid);
        assert_eq!(r.buffer_index, 5);
    }

    #[test]
    fn test_reference_frame_invalidate() {
        let mut r = ReferenceFrame::with_buffer(0);
        assert!(r.valid);

        r.invalidate();
        assert!(!r.valid);
    }

    #[test]
    fn test_reference_frame_pool_new() {
        let pool = ReferenceFramePool::new();
        assert_eq!(pool.buffers.len(), REF_BUFFER_COUNT);
    }

    #[test]
    fn test_reference_frame_pool_find_free() {
        let pool = ReferenceFramePool::new();
        let free = pool.find_free_buffer();
        assert!(free.is_some());
        assert_eq!(free.expect("should succeed"), 0);
    }

    #[test]
    fn test_reference_frame_pool_set_ref() {
        let mut pool = ReferenceFramePool::new();
        pool.set_dimensions(64, 64);
        pool.allocate_buffers();

        // Mark buffer 0 as valid
        pool.buffers[0].set_valid(1);

        pool.set_ref_frame(RefFrameType::Last, 0);

        assert!(pool.is_ref_valid(RefFrameType::Last));
        assert!(!pool.is_ref_valid(RefFrameType::Golden));
    }

    #[test]
    fn test_reference_frame_pool_sign_bias() {
        let mut pool = ReferenceFramePool::new();

        pool.set_sign_bias(RefFrameType::Golden, true);
        assert!(pool.get_sign_bias(RefFrameType::Golden));
        assert!(!pool.get_sign_bias(RefFrameType::Last));
    }

    #[test]
    fn test_reference_frame_pool_reset() {
        let mut pool = ReferenceFramePool::new();
        pool.set_dimensions(64, 64);
        pool.allocate_buffers();
        pool.buffers[0].set_valid(1);
        pool.set_ref_frame(RefFrameType::Last, 0);

        pool.reset();

        assert!(!pool.is_ref_valid(RefFrameType::Last));
    }

    #[test]
    fn test_sign_bias_info() {
        let mut info = SignBiasInfo::new();
        info.set_current(5);
        info.set_ref_order_hint(RefFrameType::Last, 4);
        info.set_ref_order_hint(RefFrameType::Golden, 2);
        info.set_ref_order_hint(RefFrameType::AltRef, 8);

        // LAST (4) < current (5) -> no sign bias
        assert!(!info.compute_sign_bias(RefFrameType::Last));
        // GOLDEN (2) < current (5) -> no sign bias
        assert!(!info.compute_sign_bias(RefFrameType::Golden));
        // ALTREF (8) > current (5) -> sign bias
        assert!(info.compute_sign_bias(RefFrameType::AltRef));
    }

    #[test]
    fn test_sign_bias_compute_all() {
        let mut info = SignBiasInfo::new();
        info.set_current(5);
        info.set_ref_order_hint(RefFrameType::Last, 4);
        info.set_ref_order_hint(RefFrameType::AltRef, 6);

        let biases = info.compute_all();
        assert!(!biases[0]); // Intra
        assert!(!biases[1]); // Last
        assert!(!biases[2]); // Golden (default 0)
        assert!(biases[3]); // AltRef
    }

    #[test]
    fn test_ref_update_flags() {
        let flags = RefUpdateFlags::none();
        assert!(!flags.any());

        let flags = RefUpdateFlags::keyframe();
        assert!(flags.update_last);
        assert!(flags.update_golden);
        assert!(flags.update_altref);
        assert!(flags.any());
    }

    #[test]
    fn test_ref_update_flags_from_bits() {
        let flags = RefUpdateFlags::from_bits(0b101);
        assert!(flags.update_last);
        assert!(!flags.update_golden);
        assert!(flags.update_altref);

        assert_eq!(flags.to_bits(), 0b101);
    }

    #[test]
    fn test_ref_update_flags_should_update() {
        let flags = RefUpdateFlags::from_bits(0b011);
        assert!(flags.should_update(RefFrameType::Last));
        assert!(flags.should_update(RefFrameType::Golden));
        assert!(!flags.should_update(RefFrameType::AltRef));
        assert!(!flags.should_update(RefFrameType::Intra));
    }
}
