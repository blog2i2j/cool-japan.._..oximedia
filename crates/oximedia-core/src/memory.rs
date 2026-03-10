//! Core memory management primitives for `OxiMedia`.
//!
//! This module provides low-level memory utilities optimised for multimedia
//! workloads:
//!
//! - [`MemoryLayout`] – alignment / stride calculations
//! - [`AlignedBuffer`] – a `Vec<u8>` wrapper with alignment guarantees
//! - [`MemoryPool`] – a pool of reusable `AlignedBuffer`s
//! - [`RingAllocator`] – a simple ring (circular) allocator
//!
//! # Example
//!
//! ```
//! use oximedia_core::memory::{AlignedBuffer, MemoryPool, RingAllocator};
//!
//! // Aligned buffer
//! let buf = AlignedBuffer::new(256, 16);
//! assert_eq!(buf.capacity(), 256);
//!
//! // Pool
//! let mut pool = MemoryPool::new(4, 1024, 16);
//! let idx = pool.allocate(512)?;
//! pool.deallocate(idx);
//! assert_eq!(pool.available(), 4);
//!
//! // Ring allocator
//! let mut ring = RingAllocator::new(256);
//! let offset = ring.allocate(64)?;
//! assert_eq!(offset, 0);
//! ```

#![allow(dead_code)]

/// Describes the memory layout of a buffer: alignment, total size, and stride.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryLayout {
    /// Required alignment in bytes (must be a power of two).
    pub alignment: usize,
    /// Usable size in bytes.
    pub size: usize,
    /// Row stride in bytes (may be ≥ `size` when padding is required).
    pub stride: usize,
}

impl MemoryLayout {
    /// Creates a new `MemoryLayout`.
    ///
    /// If `stride` is 0 it is set to `size`.
    #[must_use]
    pub fn new(alignment: usize, size: usize, stride: usize) -> Self {
        let stride = if stride == 0 { size } else { stride };
        Self {
            alignment,
            size,
            stride,
        }
    }

    /// Returns `true` when `ptr` is aligned to `self.alignment`.
    ///
    /// If `alignment` is 0 this always returns `true`.
    #[must_use]
    pub fn is_aligned(&self, ptr: usize) -> bool {
        if self.alignment == 0 {
            return true;
        }
        ptr % self.alignment == 0
    }

    /// Returns `size` rounded up to the nearest multiple of `alignment`.
    ///
    /// If `alignment` is 0 returns `size` unchanged.
    #[must_use]
    pub fn padded_size(&self) -> usize {
        if self.alignment == 0 {
            return self.size;
        }
        self.size.div_ceil(self.alignment) * self.alignment
    }
}

/// A heap-allocated byte buffer paired with its [`MemoryLayout`].
///
/// The buffer is backed by a `Vec<u8>` of length `layout.size`.
/// Note: the Rust allocator may not honour the requested alignment; this type
/// is a logical wrapper that tracks alignment metadata.
#[derive(Debug)]
pub struct AlignedBuffer {
    data: Vec<u8>,
    /// Layout metadata for this buffer.
    pub layout: MemoryLayout,
}

impl AlignedBuffer {
    /// Allocates a new zeroed buffer of `size` bytes with the given alignment.
    #[must_use]
    pub fn new(size: usize, align: usize) -> Self {
        let layout = MemoryLayout::new(align, size, size);
        Self {
            data: vec![0u8; size],
            layout,
        }
    }

    /// Returns an immutable slice over the buffer contents.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// Returns a mutable slice over the buffer contents.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Returns the capacity (usable size) of the buffer in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.layout.size
    }
}

/// A pool of reusable [`AlignedBuffer`]s.
///
/// Buffers are identified by an index into an internal array.
pub struct MemoryPool {
    buffers: Vec<AlignedBuffer>,
    free_list: Vec<usize>,
}

impl MemoryPool {
    /// Creates a pool with `count` buffers each of `size` bytes and `align` alignment.
    #[must_use]
    pub fn new(count: usize, size: usize, align: usize) -> Self {
        let buffers: Vec<AlignedBuffer> = (0..count)
            .map(|_| AlignedBuffer::new(size, align))
            .collect();
        let free_list: Vec<usize> = (0..count).collect();
        Self { buffers, free_list }
    }

    /// Takes a buffer from the free list and returns its index.
    ///
    /// The caller must ensure the requested `size` fits within the buffer.
    /// Returns `None` when the pool is exhausted.
    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        // Find a free buffer large enough.
        let pos = self
            .free_list
            .iter()
            .position(|&idx| self.buffers[idx].capacity() >= size)?;
        let idx = self.free_list.remove(pos);
        Some(idx)
    }

    /// Returns a buffer to the free list by its index.
    pub fn deallocate(&mut self, idx: usize) {
        if idx < self.buffers.len() && !self.free_list.contains(&idx) {
            self.free_list.push(idx);
        }
    }

    /// Number of buffers currently available.
    #[must_use]
    pub fn available(&self) -> usize {
        self.free_list.len()
    }

    /// Returns an immutable reference to the buffer at `idx`.
    #[must_use]
    pub fn buffer(&self, idx: usize) -> Option<&AlignedBuffer> {
        self.buffers.get(idx)
    }

    /// Returns a mutable reference to the buffer at `idx`.
    #[must_use]
    pub fn buffer_mut(&mut self, idx: usize) -> Option<&mut AlignedBuffer> {
        self.buffers.get_mut(idx)
    }
}

/// A simple ring (circular) allocator over a fixed-size backing buffer.
///
/// Allocations are bump-allocated from `tail` toward the end; when the
/// end is reached the allocator wraps around to the front.
///
/// Uses an explicit `used` byte counter to distinguish between the "empty"
/// and "full" states (which would otherwise both show `head == tail`).
pub struct RingAllocator {
    buffer: Vec<u8>,
    head: usize,
    tail: usize,
    /// Number of bytes currently allocated (not yet freed).
    used: usize,
}

impl RingAllocator {
    /// Creates a ring allocator over a backing buffer of `capacity` bytes.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: vec![0u8; capacity],
            head: 0,
            tail: 0,
            used: 0,
        }
    }

    /// Allocates `size` contiguous bytes and returns the starting offset into
    /// the backing buffer, or `None` when there is not enough free space.
    pub fn allocate(&mut self, size: usize) -> Option<usize> {
        if size == 0 || size > self.buffer.len() {
            return None;
        }
        if size > self.free_space() {
            return None;
        }
        let capacity = self.buffer.len();
        let offset = self.tail;
        self.tail = (self.tail + size) % capacity;
        self.used += size;
        Some(offset)
    }

    /// Free space available for allocation (in bytes).
    #[must_use]
    pub fn free_space(&self) -> usize {
        self.buffer.len().saturating_sub(self.used)
    }

    /// Resets both head and tail to 0, effectively freeing all allocations.
    pub fn reset(&mut self) {
        self.head = 0;
        self.tail = 0;
        self.used = 0;
        self.buffer.fill(0);
    }

    /// Returns the total capacity of the backing buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buffer.len()
    }

    /// Advances the head pointer by `size` bytes, reclaiming that space.
    pub fn free(&mut self, size: usize) {
        let capacity = self.buffer.len();
        self.head = (self.head + size) % capacity;
        self.used = self.used.saturating_sub(size);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. MemoryLayout::is_aligned – aligned pointer
    #[test]
    fn test_is_aligned_true() {
        let layout = MemoryLayout::new(16, 64, 64);
        assert!(layout.is_aligned(0));
        assert!(layout.is_aligned(16));
        assert!(layout.is_aligned(32));
    }

    // 2. MemoryLayout::is_aligned – misaligned pointer
    #[test]
    fn test_is_aligned_false() {
        let layout = MemoryLayout::new(16, 64, 64);
        assert!(!layout.is_aligned(1));
        assert!(!layout.is_aligned(15));
    }

    // 3. MemoryLayout::padded_size – no padding needed
    #[test]
    fn test_padded_size_exact() {
        let layout = MemoryLayout::new(16, 64, 64);
        assert_eq!(layout.padded_size(), 64);
    }

    // 4. MemoryLayout::padded_size – padding needed
    #[test]
    fn test_padded_size_with_padding() {
        let layout = MemoryLayout::new(16, 60, 60);
        assert_eq!(layout.padded_size(), 64);
    }

    // 5. MemoryLayout::padded_size – zero alignment
    #[test]
    fn test_padded_size_zero_alignment() {
        let layout = MemoryLayout::new(0, 60, 60);
        assert_eq!(layout.padded_size(), 60);
    }

    // 6. AlignedBuffer::new and capacity
    #[test]
    fn test_aligned_buffer_new() {
        let buf = AlignedBuffer::new(256, 16);
        assert_eq!(buf.capacity(), 256);
        assert_eq!(buf.as_slice().len(), 256);
        assert!(buf.as_slice().iter().all(|&b| b == 0));
    }

    // 7. AlignedBuffer::as_mut_slice write
    #[test]
    fn test_aligned_buffer_write() {
        let mut buf = AlignedBuffer::new(16, 8);
        buf.as_mut_slice()[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }

    // 8. MemoryPool::allocate – success
    #[test]
    fn test_pool_allocate_success() {
        let mut pool = MemoryPool::new(4, 1024, 16);
        assert_eq!(pool.available(), 4);
        let idx = pool.allocate(512).expect("allocate should succeed");
        assert_eq!(pool.available(), 3);
        assert!(pool.buffer(idx).is_some());
    }

    // 9. MemoryPool::allocate – exhausted returns None
    #[test]
    fn test_pool_allocate_exhausted() {
        let mut pool = MemoryPool::new(1, 64, 8);
        let _ = pool.allocate(64).expect("allocate should succeed");
        assert!(pool.allocate(64).is_none());
    }

    // 10. MemoryPool::deallocate – returns buffer
    #[test]
    fn test_pool_deallocate() {
        let mut pool = MemoryPool::new(2, 128, 16);
        let idx = pool.allocate(64).expect("allocate should succeed");
        assert_eq!(pool.available(), 1);
        pool.deallocate(idx);
        assert_eq!(pool.available(), 2);
    }

    // 11. MemoryPool::allocate – size too large skipped
    #[test]
    fn test_pool_size_too_large() {
        let mut pool = MemoryPool::new(2, 128, 16);
        // Requesting more than any buffer holds should return None
        let result = pool.allocate(256);
        assert!(result.is_none());
    }

    // 12. RingAllocator::allocate – sequential offsets
    #[test]
    fn test_ring_sequential() {
        let mut ring = RingAllocator::new(256);
        assert_eq!(ring.allocate(64), Some(0));
        assert_eq!(ring.allocate(64), Some(64));
        assert_eq!(ring.allocate(64), Some(128));
    }

    // 13. RingAllocator::free_space after allocations
    #[test]
    fn test_ring_free_space() {
        let mut ring = RingAllocator::new(256);
        let _ = ring.allocate(100);
        // After allocating 100 bytes the ring has not wrapped, so 156 bytes free
        // (capacity - (tail - head)) = 256 - 100 = 156
        assert_eq!(ring.free_space(), 156);
    }

    // 14. RingAllocator::reset
    #[test]
    fn test_ring_reset() {
        let mut ring = RingAllocator::new(128);
        let _ = ring.allocate(64);
        ring.reset();
        assert_eq!(ring.free_space(), 128);
        assert_eq!(ring.allocate(128), Some(0));
    }

    // 15. RingAllocator::allocate – None when full
    #[test]
    fn test_ring_full() {
        let mut ring = RingAllocator::new(64);
        let _ = ring.allocate(64);
        assert!(ring.allocate(1).is_none());
    }

    // 16. RingAllocator::free reclaims space
    #[test]
    fn test_ring_free_reclaims() {
        let mut ring = RingAllocator::new(128);
        let _ = ring.allocate(64);
        ring.free(64);
        assert_eq!(ring.free_space(), 128);
    }
}
