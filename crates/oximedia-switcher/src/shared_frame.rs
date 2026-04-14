//! Zero-copy frame sharing via reference-counted buffers.
//!
//! Wraps a raw pixel buffer in an `Arc` so that multiple consumers (multiviewer
//! tiles, recording paths, downstream keyers …) can hold references to the same
//! frame data without copying. The inner data is immutable once the frame has
//! been constructed; writers must allocate a new frame for each output.

use std::sync::Arc;

/// Metadata describing the layout of a raw pixel plane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlaneDesc {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
    /// Row stride in bytes (may be > width * bytes_per_pixel for alignment).
    pub stride: u32,
    /// Number of bytes per pixel in this plane.
    pub bytes_per_pixel: u32,
}

impl PlaneDesc {
    /// Create a new `PlaneDesc`.
    pub fn new(width: u32, height: u32, stride: u32, bytes_per_pixel: u32) -> Self {
        Self {
            width,
            height,
            stride,
            bytes_per_pixel,
        }
    }

    /// Create a tightly-packed (no padding) plane descriptor.
    pub fn packed(width: u32, height: u32, bytes_per_pixel: u32) -> Self {
        Self::new(width, height, width * bytes_per_pixel, bytes_per_pixel)
    }

    /// Total byte size of the plane: `height * stride`.
    pub fn byte_size(&self) -> usize {
        (self.height as usize) * (self.stride as usize)
    }
}

/// The immutable interior of a shared frame.
///
/// Allocated once, then wrapped in `Arc<SharedFrameInner>` so that many readers
/// can access it concurrently without copying.
pub struct SharedFrameInner {
    /// Raw pixel data (all planes concatenated).
    pub data: Vec<u8>,
    /// Descriptor for each plane within `data`.
    pub planes: Vec<PlaneDesc>,
    /// Presentation timestamp in 90 kHz ticks.
    pub pts_90khz: i64,
    /// Input slot this frame originated from (0-based).
    pub source_id: usize,
}

impl SharedFrameInner {
    /// Create a new inner frame.
    pub fn new(data: Vec<u8>, planes: Vec<PlaneDesc>, pts_90khz: i64, source_id: usize) -> Self {
        Self {
            data,
            planes,
            pts_90khz,
            source_id,
        }
    }

    /// Total byte size of all planes.
    pub fn total_bytes(&self) -> usize {
        self.planes.iter().map(|p| p.byte_size()).sum()
    }

    /// Number of planes.
    pub fn plane_count(&self) -> usize {
        self.planes.len()
    }
}

/// A zero-copy, reference-counted video frame.
///
/// Cloning a `SharedFrame` increments an `Arc` reference count rather than
/// copying pixel data.  Multiple holder paths (program output, multiviewer,
/// upstream keyers …) can each hold an independent `SharedFrame` pointing at
/// the same allocation.
#[derive(Clone)]
pub struct SharedFrame(Arc<SharedFrameInner>);

impl SharedFrame {
    /// Construct a `SharedFrame` from the given inner data.
    pub fn new(inner: SharedFrameInner) -> Self {
        Self(Arc::new(inner))
    }

    /// Convenience constructor: pack raw bytes into a single-plane frame.
    pub fn from_raw(
        data: Vec<u8>,
        width: u32,
        height: u32,
        bytes_per_pixel: u32,
        pts_90khz: i64,
        source_id: usize,
    ) -> Self {
        let plane = PlaneDesc::packed(width, height, bytes_per_pixel);
        let inner = SharedFrameInner::new(data, vec![plane], pts_90khz, source_id);
        Self::new(inner)
    }

    /// Access the inner data.
    pub fn inner(&self) -> &SharedFrameInner {
        &self.0
    }

    /// Number of current owners of this frame (including `self`).
    pub fn ref_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }

    /// Attempt to take exclusive ownership if this is the sole reference.
    ///
    /// Returns `Ok(inner)` when the `Arc` reference count is exactly 1, or
    /// `Err(self)` when other references exist.
    pub fn try_unwrap(self) -> Result<SharedFrameInner, Self> {
        Arc::try_unwrap(self.0).map_err(Self)
    }

    /// PTS in 90 kHz ticks.
    pub fn pts_90khz(&self) -> i64 {
        self.0.pts_90khz
    }

    /// Source input slot.
    pub fn source_id(&self) -> usize {
        self.0.source_id
    }
}

impl std::fmt::Debug for SharedFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedFrame")
            .field("source_id", &self.0.source_id)
            .field("pts_90khz", &self.0.pts_90khz)
            .field("planes", &self.0.planes.len())
            .field("ref_count", &Arc::strong_count(&self.0))
            .finish()
    }
}

/// Manages a ring of the most-recent frame per source slot.
///
/// Each source slot holds at most one `SharedFrame` (the latest).  Writing a
/// new frame simply replaces the stored `Arc`; old consumers that still hold a
/// clone continue reading the previous data safely.
pub struct SharedFrameBuffer {
    /// One slot per source (index == source_id).
    slots: Vec<Option<SharedFrame>>,
}

impl SharedFrameBuffer {
    /// Create a buffer with `num_sources` independent slots.
    pub fn new(num_sources: usize) -> Self {
        Self {
            slots: vec![None; num_sources],
        }
    }

    /// Write the latest frame for `source_id`.
    ///
    /// Replaces the previous frame (if any).  Existing `SharedFrame` clones
    /// held by other threads remain valid — only the slot's pointer is updated.
    pub fn write(&mut self, source_id: usize, frame: SharedFrame) {
        if source_id < self.slots.len() {
            self.slots[source_id] = Some(frame);
        }
    }

    /// Read (clone the `Arc`) for `source_id`.
    ///
    /// Returns `None` when no frame has been written yet.
    pub fn read(&self, source_id: usize) -> Option<SharedFrame> {
        self.slots.get(source_id)?.clone()
    }

    /// Clear the frame for `source_id`.
    pub fn clear_slot(&mut self, source_id: usize) {
        if source_id < self.slots.len() {
            self.slots[source_id] = None;
        }
    }

    /// Number of source slots.
    pub fn slot_count(&self) -> usize {
        self.slots.len()
    }

    /// Number of slots that currently hold a frame.
    pub fn occupied_count(&self) -> usize {
        self.slots.iter().filter(|s| s.is_some()).count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(source_id: usize, pts: i64) -> SharedFrame {
        SharedFrame::from_raw(vec![0u8; 32], 4, 4, 2, pts, source_id)
    }

    #[test]
    fn test_plane_desc_byte_size() {
        let p = PlaneDesc::packed(1920, 1080, 2);
        assert_eq!(p.byte_size(), 1920 * 1080 * 2);
    }

    #[test]
    fn test_plane_desc_packed_stride() {
        let p = PlaneDesc::packed(320, 240, 3);
        assert_eq!(p.stride, 320 * 3);
    }

    #[test]
    fn test_shared_frame_from_raw() {
        let frame = make_frame(0, 90000);
        assert_eq!(frame.source_id(), 0);
        assert_eq!(frame.pts_90khz(), 90000);
    }

    #[test]
    fn test_shared_frame_clone_increments_refcount() {
        let frame = make_frame(1, 0);
        assert_eq!(frame.ref_count(), 1);
        let clone = frame.clone();
        assert_eq!(frame.ref_count(), 2);
        assert_eq!(clone.ref_count(), 2);
    }

    #[test]
    fn test_shared_frame_drop_decrements_refcount() {
        let frame = make_frame(2, 0);
        let clone = frame.clone();
        assert_eq!(frame.ref_count(), 2);
        drop(clone);
        assert_eq!(frame.ref_count(), 1);
    }

    #[test]
    fn test_shared_frame_try_unwrap_sole_owner() {
        let frame = make_frame(0, 0);
        let result = frame.try_unwrap();
        assert!(result.is_ok());
    }

    #[test]
    fn test_shared_frame_try_unwrap_shared_fails() {
        let frame = make_frame(0, 0);
        let _clone = frame.clone();
        let result = frame.try_unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn test_frame_buffer_write_and_read() {
        let mut buf = SharedFrameBuffer::new(4);
        let frame = make_frame(0, 1234);
        buf.write(0, frame);
        let read_back = buf.read(0).expect("slot 0 should have a frame");
        assert_eq!(read_back.source_id(), 0);
        assert_eq!(read_back.pts_90khz(), 1234);
    }

    #[test]
    fn test_frame_buffer_read_empty_slot() {
        let buf = SharedFrameBuffer::new(4);
        assert!(buf.read(0).is_none());
        assert!(buf.read(3).is_none());
    }

    #[test]
    fn test_frame_buffer_read_out_of_range() {
        let buf = SharedFrameBuffer::new(4);
        assert!(buf.read(99).is_none());
    }

    #[test]
    fn test_frame_buffer_overwrite() {
        let mut buf = SharedFrameBuffer::new(4);
        buf.write(0, make_frame(0, 1000));
        buf.write(0, make_frame(0, 2000));
        let f = buf.read(0).expect("slot 0 should have a frame");
        assert_eq!(f.pts_90khz(), 2000);
    }

    #[test]
    fn test_frame_buffer_clear_slot() {
        let mut buf = SharedFrameBuffer::new(4);
        buf.write(1, make_frame(1, 0));
        assert_eq!(buf.occupied_count(), 1);
        buf.clear_slot(1);
        assert_eq!(buf.occupied_count(), 0);
        assert!(buf.read(1).is_none());
    }

    #[test]
    fn test_frame_buffer_occupied_count() {
        let mut buf = SharedFrameBuffer::new(4);
        assert_eq!(buf.occupied_count(), 0);
        buf.write(0, make_frame(0, 0));
        buf.write(2, make_frame(2, 0));
        assert_eq!(buf.occupied_count(), 2);
    }

    #[test]
    fn test_old_clone_survives_overwrite() {
        let mut buf = SharedFrameBuffer::new(4);
        buf.write(0, make_frame(0, 1000));
        // Acquire a clone of the frame at pts=1000.
        let old_clone = buf.read(0).expect("should exist");
        // Overwrite the slot with a new frame.
        buf.write(0, make_frame(0, 2000));
        // The old clone still refers to pts=1000.
        assert_eq!(old_clone.pts_90khz(), 1000);
        // The slot now returns pts=2000.
        let new_frame = buf.read(0).expect("should exist");
        assert_eq!(new_frame.pts_90khz(), 2000);
    }

    #[test]
    fn test_inner_total_bytes() {
        let data = vec![0u8; 1920 * 1080 * 2];
        let plane = PlaneDesc::packed(1920, 1080, 2);
        let inner = SharedFrameInner::new(data, vec![plane], 0, 0);
        assert_eq!(inner.total_bytes(), 1920 * 1080 * 2);
    }
}
