//! Frame buffer pool for zero-copy operations.
//!
//! This module provides a simple ID-based buffer pool (`BufferPool`) for efficient
//! frame buffer reuse with explicit ownership tracking via an `in_use` flag.

#![allow(dead_code)]

/// Descriptor for a pooled buffer slot.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BufferDesc {
    /// Size of the buffer in bytes.
    pub size_bytes: usize,
    /// Required memory alignment in bytes.
    pub alignment: usize,
    /// Pool identifier this descriptor belongs to.
    pub pool_id: u32,
}

impl BufferDesc {
    /// Creates a new `BufferDesc`.
    #[must_use]
    pub fn new(size_bytes: usize, alignment: usize, pool_id: u32) -> Self {
        Self {
            size_bytes,
            alignment,
            pool_id,
        }
    }

    /// Returns `true` if the alignment equals 4096 (one memory page).
    #[must_use]
    pub fn is_page_aligned(&self) -> bool {
        self.alignment == 4096
    }

    /// Returns how many slots of `slot_size` bytes are needed to hold this buffer.
    ///
    /// # Panics
    ///
    /// Panics if `slot_size` is zero.
    #[must_use]
    pub fn slots_needed(&self, slot_size: usize) -> usize {
        assert!(slot_size > 0, "slot_size must be non-zero");
        self.size_bytes.div_ceil(slot_size)
    }
}

/// A buffer managed by the pool with an associated unique ID.
#[derive(Debug)]
pub struct PooledBuffer {
    /// Unique identifier for this buffer within the pool.
    pub id: u64,
    /// Raw buffer data.
    pub data: Vec<u8>,
    /// Descriptor for this buffer.
    pub desc: BufferDesc,
    /// Whether this buffer is currently in use.
    pub in_use: bool,
}

impl PooledBuffer {
    /// Creates a new `PooledBuffer`.
    #[must_use]
    pub fn new(id: u64, desc: BufferDesc) -> Self {
        let data = vec![0u8; desc.size_bytes];
        Self {
            id,
            data,
            desc,
            in_use: false,
        }
    }

    /// Resets the buffer: zeroes the data and marks it as not in use.
    pub fn reset(&mut self) {
        self.data.fill(0);
        self.in_use = false;
    }

    /// Returns the number of bytes available (equal to the buffer size).
    #[must_use]
    pub fn available_size(&self) -> usize {
        self.data.len()
    }
}

/// A pool of frame buffers identified by integer IDs.
///
/// Buffers are acquired by ID and released back to the pool by ID.
#[derive(Debug)]
pub struct BufferPool {
    /// Managed buffers.
    pub buffers: Vec<PooledBuffer>,
    /// Counter for assigning unique IDs to new buffers.
    pub next_id: u64,
}

impl BufferPool {
    /// Creates a new `BufferPool` with `count` buffers each of `buf_size` bytes.
    ///
    /// All buffers share `pool_id = 0` and default alignment of 64.
    #[must_use]
    pub fn new(count: usize, buf_size: usize) -> Self {
        let mut buffers = Vec::with_capacity(count);
        for id in 0..count as u64 {
            let desc = BufferDesc::new(buf_size, 64, 0);
            buffers.push(PooledBuffer::new(id, desc));
        }
        Self {
            buffers,
            next_id: count as u64,
        }
    }

    /// Acquires an available buffer and returns its ID.
    ///
    /// Returns `None` if no buffer is free.
    #[must_use]
    pub fn acquire(&mut self) -> Option<u64> {
        for buf in &mut self.buffers {
            if !buf.in_use {
                buf.in_use = true;
                return Some(buf.id);
            }
        }
        None
    }

    /// Releases the buffer with the given `id` back to the pool.
    ///
    /// If the ID is not found this is a no-op.
    pub fn release(&mut self, id: u64) {
        if let Some(buf) = self.buffers.iter_mut().find(|b| b.id == id) {
            buf.reset();
        }
    }

    /// Returns the number of buffers not currently in use.
    #[must_use]
    pub fn available_count(&self) -> usize {
        self.buffers.iter().filter(|b| !b.in_use).count()
    }

    /// Returns the total number of buffers managed by this pool.
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.buffers.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- BufferDesc tests ---

    #[test]
    fn test_buffer_desc_new() {
        let desc = BufferDesc::new(1024, 64, 1);
        assert_eq!(desc.size_bytes, 1024);
        assert_eq!(desc.alignment, 64);
        assert_eq!(desc.pool_id, 1);
    }

    #[test]
    fn test_buffer_desc_is_page_aligned_true() {
        let desc = BufferDesc::new(8192, 4096, 0);
        assert!(desc.is_page_aligned());
    }

    #[test]
    fn test_buffer_desc_is_page_aligned_false() {
        let desc = BufferDesc::new(8192, 64, 0);
        assert!(!desc.is_page_aligned());
    }

    #[test]
    fn test_buffer_desc_slots_needed_exact() {
        let desc = BufferDesc::new(1024, 64, 0);
        assert_eq!(desc.slots_needed(512), 2);
    }

    #[test]
    fn test_buffer_desc_slots_needed_round_up() {
        let desc = BufferDesc::new(1025, 64, 0);
        // 1025 / 512 = 2.002... → ceil → 3
        assert_eq!(desc.slots_needed(512), 3);
    }

    #[test]
    fn test_buffer_desc_slots_needed_single_slot() {
        let desc = BufferDesc::new(100, 64, 0);
        assert_eq!(desc.slots_needed(200), 1);
    }

    // --- PooledBuffer tests ---

    #[test]
    fn test_pooled_buffer_initial_state() {
        let desc = BufferDesc::new(256, 64, 0);
        let buf = PooledBuffer::new(42, desc);
        assert_eq!(buf.id, 42);
        assert!(!buf.in_use);
        assert_eq!(buf.available_size(), 256);
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_pooled_buffer_reset() {
        let desc = BufferDesc::new(4, 64, 0);
        let mut buf = PooledBuffer::new(1, desc);
        buf.in_use = true;
        buf.data[0] = 0xFF;
        buf.reset();
        assert!(!buf.in_use);
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn test_pooled_buffer_available_size() {
        let desc = BufferDesc::new(512, 64, 0);
        let buf = PooledBuffer::new(0, desc);
        assert_eq!(buf.available_size(), 512);
    }

    // --- BufferPool tests ---

    #[test]
    fn test_pool_new() {
        let pool = BufferPool::new(4, 1024);
        assert_eq!(pool.total_count(), 4);
        assert_eq!(pool.available_count(), 4);
    }

    #[test]
    fn test_pool_acquire_returns_id() {
        let mut pool = BufferPool::new(2, 256);
        let id = pool.acquire();
        assert!(id.is_some());
    }

    #[test]
    fn test_pool_acquire_exhausts_buffers() {
        let mut pool = BufferPool::new(2, 256);
        let _id1 = pool.acquire().expect("acquire should succeed");
        let _id2 = pool.acquire().expect("acquire should succeed");
        assert!(pool.acquire().is_none());
    }

    #[test]
    fn test_pool_available_count_decrements_on_acquire() {
        let mut pool = BufferPool::new(3, 64);
        assert_eq!(pool.available_count(), 3);
        let _ = pool.acquire();
        assert_eq!(pool.available_count(), 2);
        let _ = pool.acquire();
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn test_pool_release_makes_buffer_available() {
        let mut pool = BufferPool::new(1, 64);
        let id = pool.acquire().expect("acquire should succeed");
        assert_eq!(pool.available_count(), 0);
        pool.release(id);
        assert_eq!(pool.available_count(), 1);
    }

    #[test]
    fn test_pool_release_unknown_id_is_noop() {
        let mut pool = BufferPool::new(2, 64);
        let before = pool.available_count();
        pool.release(999);
        assert_eq!(pool.available_count(), before);
    }

    #[test]
    fn test_pool_total_count_unchanged_after_ops() {
        let mut pool = BufferPool::new(5, 128);
        let ids: Vec<u64> = (0..5).filter_map(|_| pool.acquire()).collect();
        assert_eq!(pool.total_count(), 5);
        for id in ids {
            pool.release(id);
        }
        assert_eq!(pool.total_count(), 5);
    }
}
