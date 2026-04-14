//! Zero-copy buffer pool for GPU-style memory management.
//!
//! Provides a reuse-oriented pool of byte buffers, inspired by GPU memory
//! management patterns.  Buffers are acquired by size and alignment,
//! used by the caller, and released back to the pool rather than freed.
//! Unused buffers older than 60 seconds are evicted by [`BufferPool::defragment`].

#![allow(clippy::cast_precision_loss)]

use std::time::Instant;

// ---------------------------------------------------------------------------
// GpuBuffer
// ---------------------------------------------------------------------------

/// A raw byte buffer managed by a [`BufferPool`].
pub struct GpuBuffer {
    /// Unique identifier assigned by the owning pool.
    pub id: u64,
    /// Allocated capacity in bytes.
    pub size_bytes: usize,
    /// Alignment guarantee (in bytes).
    pub alignment: usize,
    /// Backing storage.
    data: Vec<u8>,
    /// Whether this buffer is currently checked out by a caller.
    pub(crate) in_use: bool,
    /// Monotonic timestamp of the most recent acquisition or release.
    pub(crate) created_at: Instant,
    /// Monotonic timestamp of last release (used for eviction).
    pub(crate) last_released_at: Option<Instant>,
}

impl GpuBuffer {
    /// Allocate a new buffer with the given `size` and `alignment`.
    ///
    /// The alignment hint is recorded but the backing `Vec<u8>` uses the
    /// default allocator.  For truly aligned allocations a custom allocator
    /// would be required; the pool still respects the alignment in
    /// compatibility checks.
    #[must_use]
    pub fn new(id: u64, size: usize, alignment: usize) -> Self {
        let effective_alignment = alignment.max(1);
        Self {
            id,
            size_bytes: size,
            alignment: effective_alignment,
            data: vec![0u8; size],
            in_use: false,
            created_at: Instant::now(),
            last_released_at: None,
        }
    }

    /// View the buffer contents as a byte slice.
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.data
    }

    /// View the buffer contents as a mutable byte slice.
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Fill the entire buffer with `value` (memset equivalent).
    pub fn fill(&mut self, value: u8) {
        self.data.fill(value);
    }

    /// Whether this buffer is currently checked out.
    #[must_use]
    pub fn is_in_use(&self) -> bool {
        self.in_use
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("id", &self.id)
            .field("size_bytes", &self.size_bytes)
            .field("alignment", &self.alignment)
            .field("in_use", &self.in_use)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Pool statistics
// ---------------------------------------------------------------------------

/// Snapshot of pool health metrics.
#[derive(Debug, Clone)]
pub struct PoolStats {
    /// Total number of buffers held by the pool (in-use + available).
    pub total_buffers: usize,
    /// Buffers currently checked out by callers.
    pub in_use_buffers: usize,
    /// Buffers available for immediate reuse.
    pub available_buffers: usize,
    /// Sum of all allocated buffer capacities in bytes.
    pub total_allocated_bytes: usize,
    /// Fraction of acquisitions satisfied from the pool (0.0 – 1.0).
    pub reuse_rate: f64,
}

// ---------------------------------------------------------------------------
// BufferPool
// ---------------------------------------------------------------------------

/// A pool of reusable GPU-style byte buffers.
///
/// Callers acquire a buffer via [`acquire`][BufferPool::acquire] (receiving its
/// ID), read/write via [`get_mut`][BufferPool::get_mut], then return it to the
/// pool with [`release`][BufferPool::release].
pub struct BufferPool {
    buffers: Vec<GpuBuffer>,
    next_id: u64,
    total_allocated: usize,
    max_pool_bytes: usize,
    /// Number of acquisitions satisfied by reusing an existing buffer.
    reuse_count: u64,
    /// Total acquisitions ever made (reused + newly allocated).
    alloc_count: u64,
}

impl BufferPool {
    /// Create a new pool that will hold at most `max_pool_bytes` of backing
    /// storage before refusing new allocations.
    #[must_use]
    pub fn new(max_pool_bytes: usize) -> Self {
        Self {
            buffers: Vec::new(),
            next_id: 1,
            total_allocated: 0,
            max_pool_bytes,
            reuse_count: 0,
            alloc_count: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Acquire
    // -----------------------------------------------------------------------

    /// Check out a buffer of at least `size_bytes` with at least `alignment`.
    ///
    /// Strategy: find the *smallest* existing compatible free buffer to
    /// minimise fragmentation.  If none exists, allocate a new one (provided
    /// the pool is below its byte budget).
    ///
    /// Returns the buffer `id` on success, or `None` if no buffer is available
    /// and allocating a new one would exceed the pool's byte budget.
    pub fn acquire(&mut self, size_bytes: usize, alignment: usize) -> Option<u64> {
        self.alloc_count += 1;

        // Find the best (smallest compatible) free buffer.
        let best_idx = self
            .buffers
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                !b.in_use && b.size_bytes >= size_bytes && b.alignment >= alignment.max(1)
            })
            .min_by_key(|(_, b)| b.size_bytes)
            .map(|(idx, _)| idx);

        if let Some(idx) = best_idx {
            self.buffers[idx].in_use = true;
            self.buffers[idx].created_at = Instant::now();
            self.reuse_count += 1;
            return Some(self.buffers[idx].id);
        }

        // No compatible free buffer — try to allocate a new one.
        let effective_alignment = alignment.max(1);
        let new_size = self.total_allocated + size_bytes;
        if new_size > self.max_pool_bytes {
            return None; // over budget
        }

        let id = self.next_id;
        self.next_id += 1;

        let mut buf = GpuBuffer::new(id, size_bytes, effective_alignment);
        buf.in_use = true;
        self.total_allocated += size_bytes;
        self.buffers.push(buf);

        Some(id)
    }

    // -----------------------------------------------------------------------
    // Release
    // -----------------------------------------------------------------------

    /// Return a buffer to the pool by `id`.
    ///
    /// The buffer is kept for future reuse but marked as available.
    /// Returns `true` if the buffer was found and released, `false` otherwise.
    pub fn release(&mut self, id: u64) -> bool {
        if let Some(buf) = self.buffers.iter_mut().find(|b| b.id == id) {
            buf.in_use = false;
            buf.last_released_at = Some(Instant::now());
            true
        } else {
            false
        }
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Borrow the buffer with the given `id`.
    #[must_use]
    pub fn get(&self, id: u64) -> Option<&GpuBuffer> {
        self.buffers.iter().find(|b| b.id == id)
    }

    /// Mutably borrow the buffer with the given `id`.
    #[must_use]
    pub fn get_mut(&mut self, id: u64) -> Option<&mut GpuBuffer> {
        self.buffers.iter_mut().find(|b| b.id == id)
    }

    // -----------------------------------------------------------------------
    // Defragmentation
    // -----------------------------------------------------------------------

    /// Evict all free buffers that have not been used for more than 60 seconds.
    ///
    /// In-use buffers are never evicted.
    pub fn defragment(&mut self) {
        let now = Instant::now();
        let eviction_threshold = std::time::Duration::from_secs(60);

        let mut bytes_freed = 0usize;
        self.buffers.retain(|buf| {
            if buf.in_use {
                return true; // never evict live buffers
            }
            let idle_since = buf.last_released_at.unwrap_or(buf.created_at);
            if now.duration_since(idle_since) > eviction_threshold {
                bytes_freed += buf.size_bytes;
                false // evict
            } else {
                true
            }
        });
        self.total_allocated = self.total_allocated.saturating_sub(bytes_freed);
    }

    // -----------------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------------

    /// Snapshot of pool metrics.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let in_use = self.buffers.iter().filter(|b| b.in_use).count();
        let available = self.buffers.len() - in_use;
        let reuse_rate = if self.alloc_count == 0 {
            0.0
        } else {
            self.reuse_count as f64 / self.alloc_count as f64
        };
        PoolStats {
            total_buffers: self.buffers.len(),
            in_use_buffers: in_use,
            available_buffers: available,
            total_allocated_bytes: self.total_allocated,
            reuse_rate,
        }
    }

    /// Total bytes currently under management.
    #[must_use]
    pub fn total_allocated_bytes(&self) -> usize {
        self.total_allocated
    }

    /// Maximum pool capacity in bytes.
    #[must_use]
    pub fn max_pool_bytes(&self) -> usize {
        self.max_pool_bytes
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BufferPool")
            .field("buffers", &self.buffers.len())
            .field("total_allocated", &self.total_allocated)
            .field("max_pool_bytes", &self.max_pool_bytes)
            .field("alloc_count", &self.alloc_count)
            .field("reuse_count", &self.reuse_count)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// SubAllocator — bump-pointer sub-allocator within a single large buffer
// ---------------------------------------------------------------------------

/// A sub-allocation record tracking a live region within a backing buffer.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SubAllocation {
    /// Unique identifier for this allocation.
    pub id: u64,
    /// Byte offset from the start of the backing buffer.
    pub offset: u64,
    /// Size of this allocation in bytes.
    pub size: u64,
}

/// Bump-pointer sub-allocator that partitions a single large backing buffer
/// into smaller regions.
///
/// Allocation is O(1); individual `free` is O(n) over live allocations;
/// `defrag` compacts the live allocations to the front of the buffer,
/// reclaiming all freed space.
pub struct SubAllocator {
    /// Total capacity of the backing buffer in bytes.
    backing_buffer_size: u64,
    /// Next byte offset to assign (the "bump pointer").
    current_offset: u64,
    /// All currently live allocations.
    allocations: Vec<SubAllocation>,
    /// Alignment requirement for every allocation (must be a power of two).
    alignment: u64,
    /// Counter for generating unique allocation IDs.
    next_id: u64,
    /// Set of IDs that have been freed (logically dead).
    freed_ids: std::collections::HashSet<u64>,
}

impl SubAllocator {
    /// Create a new `SubAllocator` backed by a buffer of `backing_size` bytes,
    /// with all offsets aligned to `alignment` bytes.
    ///
    /// `alignment` is clamped to at least 1.
    #[must_use]
    pub fn new(backing_size: u64, alignment: u64) -> Self {
        let alignment = alignment.max(1);
        Self {
            backing_buffer_size: backing_size,
            current_offset: 0,
            allocations: Vec::new(),
            alignment,
            next_id: 1,
            freed_ids: std::collections::HashSet::new(),
        }
    }

    /// Allocate `size` bytes from the backing buffer.
    ///
    /// Returns `Some(SubAllocation)` if there is room, `None` if the backing
    /// buffer is exhausted.
    pub fn alloc(&mut self, size: u64) -> Option<SubAllocation> {
        if size == 0 {
            return None;
        }

        // Align the current offset up to the required boundary.
        let aligned_offset = Self::align_up(self.current_offset, self.alignment);
        let end = aligned_offset.checked_add(size)?;

        if end > self.backing_buffer_size {
            return None; // not enough contiguous space
        }

        let id = self.next_id;
        self.next_id += 1;
        self.current_offset = end;

        let alloc = SubAllocation {
            id,
            offset: aligned_offset,
            size,
        };
        self.allocations.push(alloc.clone());
        Some(alloc)
    }

    /// Mark the allocation with the given `id` as freed.
    ///
    /// Freed allocations are not reclaimed until [`defrag`][Self::defrag] is called.
    pub fn free(&mut self, id: u64) {
        if let Some(pos) = self.allocations.iter().position(|a| a.id == id) {
            self.freed_ids.insert(id);
            self.allocations.remove(pos);
        }
    }

    /// Compact all live allocations to the front of the backing buffer,
    /// reclaiming the space left by freed allocations.
    ///
    /// After defragmentation the bump pointer is set to just after the last
    /// live allocation, making that space available for future `alloc` calls.
    pub fn defrag(&mut self) {
        // Remove any stale freed IDs (already removed on free(), but belt-and-suspenders).
        self.allocations.retain(|a| !self.freed_ids.contains(&a.id));
        self.freed_ids.clear();

        // Re-layout the live allocations from offset 0.
        let mut cursor: u64 = 0;
        for alloc in &mut self.allocations {
            let aligned = Self::align_up(cursor, self.alignment);
            alloc.offset = aligned;
            cursor = aligned + alloc.size;
        }
        self.current_offset = cursor;
    }

    /// Fraction of the backing buffer that is currently occupied by live
    /// allocations (0.0 = empty, 1.0 = full).
    #[must_use]
    pub fn utilization(&self) -> f64 {
        if self.backing_buffer_size == 0 {
            return 0.0;
        }
        let live_bytes: u64 = self.allocations.iter().map(|a| a.size).sum();
        live_bytes as f64 / self.backing_buffer_size as f64
    }

    /// Number of live allocations.
    #[must_use]
    pub fn allocation_count(&self) -> usize {
        self.allocations.len()
    }

    /// Current value of the bump pointer (first unassigned byte offset).
    #[must_use]
    pub fn current_offset(&self) -> u64 {
        self.current_offset
    }

    /// Total backing buffer size in bytes.
    #[must_use]
    pub fn capacity(&self) -> u64 {
        self.backing_buffer_size
    }

    /// Alignment used for all allocations.
    #[must_use]
    pub fn alignment(&self) -> u64 {
        self.alignment
    }

    // ── private helpers ──────────────────────────────────────────────────────

    /// Round `offset` up to the next multiple of `alignment`.
    fn align_up(offset: u64, alignment: u64) -> u64 {
        if alignment <= 1 {
            return offset;
        }
        let rem = offset % alignment;
        if rem == 0 {
            offset
        } else {
            offset + (alignment - rem)
        }
    }
}

impl std::fmt::Debug for SubAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SubAllocator")
            .field("capacity", &self.backing_buffer_size)
            .field("current_offset", &self.current_offset)
            .field("live_allocs", &self.allocations.len())
            .field("alignment", &self.alignment)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- GpuBuffer ---

    #[test]
    fn test_gpu_buffer_new() {
        let buf = GpuBuffer::new(1, 1024, 64);
        assert_eq!(buf.id, 1);
        assert_eq!(buf.size_bytes, 1024);
        assert_eq!(buf.alignment, 64);
        assert_eq!(buf.as_slice().len(), 1024);
        assert!(!buf.is_in_use());
    }

    #[test]
    fn test_gpu_buffer_fill() {
        let mut buf = GpuBuffer::new(2, 16, 4);
        buf.fill(0xAB);
        assert!(buf.as_slice().iter().all(|&b| b == 0xAB));
    }

    #[test]
    fn test_gpu_buffer_as_mut_slice() {
        let mut buf = GpuBuffer::new(3, 8, 1);
        buf.as_mut_slice()[0] = 42;
        assert_eq!(buf.as_slice()[0], 42);
    }

    // --- BufferPool::new ---

    #[test]
    fn test_pool_new_empty() {
        let pool = BufferPool::new(1024 * 1024);
        let stats = pool.stats();
        assert_eq!(stats.total_buffers, 0);
        assert_eq!(stats.reuse_rate, 0.0);
    }

    // --- acquire / release ---

    #[test]
    fn test_pool_acquire_and_release() {
        let mut pool = BufferPool::new(1024 * 1024);
        let id = pool.acquire(256, 4).expect("acquire failed");
        assert!(pool.get(id).expect("missing").is_in_use());

        let released = pool.release(id);
        assert!(released, "release should succeed");
        assert!(!pool.get(id).expect("missing").is_in_use());
    }

    #[test]
    fn test_pool_reuse() {
        let mut pool = BufferPool::new(1024 * 1024);
        let id1 = pool.acquire(512, 4).expect("first acquire");
        pool.release(id1);
        let id2 = pool.acquire(512, 4).expect("second acquire");
        // The pool should have reused the same buffer.
        assert_eq!(id1, id2, "expected buffer reuse");
        let stats = pool.stats();
        assert!(stats.reuse_rate > 0.0);
    }

    #[test]
    fn test_pool_smallest_compatible_preferred() {
        let mut pool = BufferPool::new(4 * 1024 * 1024);
        // Allocate two free buffers of different sizes.
        let big = pool.acquire(4096, 4).expect("big");
        let small = pool.acquire(256, 4).expect("small");
        pool.release(big);
        pool.release(small);
        // Requesting 128 bytes: should get the 256-byte buffer (smallest compat).
        let id = pool.acquire(128, 4).expect("reacquire");
        assert_eq!(id, small, "should prefer smaller buffer");
    }

    #[test]
    fn test_pool_budget_exceeded() {
        let mut pool = BufferPool::new(100);
        // First acquisition should succeed.
        let id = pool.acquire(80, 1).expect("first");
        // Second would exceed budget while first is in use.
        let result = pool.acquire(80, 1);
        assert!(result.is_none(), "should fail over budget");
        pool.release(id);
    }

    #[test]
    fn test_pool_release_unknown_id() {
        let mut pool = BufferPool::new(1024);
        assert!(
            !pool.release(9999),
            "releasing unknown id should return false"
        );
    }

    #[test]
    fn test_pool_get_missing() {
        let pool = BufferPool::new(1024);
        assert!(pool.get(42).is_none());
    }

    // --- get_mut ---

    #[test]
    fn test_pool_get_mut_write() {
        let mut pool = BufferPool::new(1024 * 1024);
        let id = pool.acquire(64, 1).expect("acquire");
        {
            let buf = pool.get_mut(id).expect("get_mut");
            buf.as_mut_slice()[0] = 0xFF;
        }
        assert_eq!(pool.get(id).expect("get").as_slice()[0], 0xFF);
    }

    // --- stats ---

    #[test]
    fn test_pool_stats_in_use_count() {
        let mut pool = BufferPool::new(1024 * 1024);
        let id1 = pool.acquire(128, 1).expect("a1");
        let _id2 = pool.acquire(128, 1).expect("a2");
        pool.release(id1);
        let stats = pool.stats();
        assert_eq!(stats.total_buffers, 2);
        assert_eq!(stats.in_use_buffers, 1);
        assert_eq!(stats.available_buffers, 1);
    }

    // --- defragment ---

    #[test]
    fn test_pool_defragment_keeps_in_use() {
        let mut pool = BufferPool::new(1024 * 1024);
        let id = pool.acquire(64, 1).expect("acquire");
        // Run defragment while buffer is in use — it should survive.
        pool.defragment();
        assert!(
            pool.get(id).is_some(),
            "in-use buffer should not be evicted"
        );
    }

    #[test]
    fn test_pool_defragment_recently_released_kept() {
        let mut pool = BufferPool::new(1024 * 1024);
        let id = pool.acquire(64, 1).expect("acquire");
        pool.release(id);
        // Buffer was just released — defragment should keep it (not 60s old).
        pool.defragment();
        assert!(
            pool.get(id).is_some(),
            "recently released buffer should survive"
        );
    }

    // ── SubAllocator tests ────────────────────────────────────────────────────

    #[test]
    fn test_sub_alloc_basic() {
        let mut sa = SubAllocator::new(1024, 4);
        let a = sa.alloc(64).expect("alloc 64 bytes");
        assert_eq!(a.offset, 0);
        assert_eq!(a.size, 64);
        assert_eq!(sa.allocation_count(), 1);
    }

    #[test]
    fn test_sub_alloc_fills_buffer() {
        let mut sa = SubAllocator::new(128, 1);
        sa.alloc(128).expect("should fill exactly");
        // Next alloc must fail — buffer is full.
        assert!(sa.alloc(1).is_none(), "buffer exhausted");
    }

    #[test]
    fn test_sub_alloc_alignment_respected() {
        let alignment = 16u64;
        let mut sa = SubAllocator::new(4096, alignment);
        // First alloc: offset must be 0 (already aligned).
        let a1 = sa.alloc(1).expect("first alloc");
        assert_eq!(a1.offset % alignment, 0, "offset must be aligned");
        // Second alloc: bump pointer is at 1, should jump to 16.
        let a2 = sa.alloc(1).expect("second alloc");
        assert_eq!(
            a2.offset, 16,
            "second alloc should start at aligned offset 16"
        );
        assert_eq!(a2.offset % alignment, 0, "all offsets must be aligned");
    }

    #[test]
    fn test_sub_alloc_free_reduces_count() {
        let mut sa = SubAllocator::new(1024, 4);
        let a1 = sa.alloc(100).expect("alloc 1");
        let a2 = sa.alloc(100).expect("alloc 2");
        assert_eq!(sa.allocation_count(), 2);
        sa.free(a1.id);
        assert_eq!(sa.allocation_count(), 1);
        sa.free(a2.id);
        assert_eq!(sa.allocation_count(), 0);
    }

    #[test]
    fn test_sub_alloc_defrag_reclaims_space() {
        let mut sa = SubAllocator::new(200, 1);
        let a1 = sa.alloc(100).expect("a1");
        let _a2 = sa.alloc(100).expect("a2");
        // Buffer is now full; next alloc must fail.
        assert!(sa.alloc(1).is_none(), "should be full before defrag");
        // Free a1 and defrag — that reclaims 100 bytes.
        sa.free(a1.id);
        sa.defrag();
        // Now there should be room for another 100-byte alloc.
        let a3 = sa.alloc(100).expect("a3 after defrag");
        assert!(a3.offset < 200, "a3 offset must be within backing buffer");
    }

    #[test]
    fn test_sub_alloc_defrag_zeroes_utilization_when_all_freed() {
        let mut sa = SubAllocator::new(512, 8);
        let a1 = sa.alloc(100).expect("a1");
        let a2 = sa.alloc(100).expect("a2");
        assert!(sa.utilization() > 0.0);
        sa.free(a1.id);
        sa.free(a2.id);
        sa.defrag();
        assert_eq!(
            sa.utilization(),
            0.0,
            "utilization must be 0 after all freed + defrag"
        );
        assert_eq!(sa.current_offset(), 0);
    }

    #[test]
    fn test_sub_alloc_utilization_rises_and_falls() {
        let mut sa = SubAllocator::new(1000, 1);
        assert_eq!(sa.utilization(), 0.0);
        let a = sa.alloc(500).expect("alloc 500");
        // utilization = 500/1000 = 0.5
        assert!((sa.utilization() - 0.5).abs() < 1e-9);
        sa.free(a.id);
        sa.defrag();
        assert_eq!(sa.utilization(), 0.0);
    }

    #[test]
    fn test_sub_alloc_zero_size_returns_none() {
        let mut sa = SubAllocator::new(1024, 4);
        assert!(sa.alloc(0).is_none(), "zero-size alloc must return None");
    }

    #[test]
    fn test_sub_alloc_ids_are_unique() {
        let mut sa = SubAllocator::new(4096, 4);
        let a1 = sa.alloc(10).expect("a1");
        let a2 = sa.alloc(10).expect("a2");
        let a3 = sa.alloc(10).expect("a3");
        assert_ne!(a1.id, a2.id);
        assert_ne!(a2.id, a3.id);
    }

    #[test]
    fn test_sub_alloc_capacity_and_alignment_accessors() {
        let sa = SubAllocator::new(8192, 64);
        assert_eq!(sa.capacity(), 8192);
        assert_eq!(sa.alignment(), 64);
    }

    #[test]
    fn test_sub_alloc_debug_fmt() {
        let sa = SubAllocator::new(1024, 4);
        let s = format!("{sa:?}");
        assert!(s.contains("SubAllocator"));
    }

    // ── Memory leak / allocate-free cycle tests ──────────────────────────────

    #[test]
    fn test_buffer_pool_alloc_free_100_cycles() {
        let mut pool = BufferPool::new(100 * 1024 * 1024); // 100 MB budget
        let mut ids = Vec::with_capacity(100);
        for _ in 0..100 {
            let id = pool.acquire(1024, 8).expect("acquire in cycle");
            ids.push(id);
        }
        for id in &ids {
            pool.release(*id);
        }
        let stats = pool.stats();
        assert_eq!(
            stats.in_use_buffers, 0,
            "all buffers must be freed after release"
        );
    }

    #[test]
    fn test_buffer_pool_alloc_free_alloc_reuse() {
        let mut pool = BufferPool::new(1024 * 1024);
        let id1 = pool.acquire(512, 4).expect("first alloc");
        pool.release(id1);
        let id2 = pool.acquire(512, 4).expect("second alloc after free");
        assert_eq!(id1, id2, "should reuse freed buffer");
        assert!(pool.stats().reuse_rate > 0.0);
        pool.release(id2);
    }

    #[test]
    fn test_buffer_pool_alloc_1000_then_free_all() {
        let budget = 1000 * 64 + 1024; // enough for 1000 x 64-byte buffers
        let mut pool = BufferPool::new(budget);
        let mut ids = Vec::with_capacity(1000);
        for _ in 0..1000 {
            let id = pool.acquire(64, 1).expect("acquire 64 bytes");
            ids.push(id);
        }
        for id in &ids {
            pool.release(*id);
        }
        let stats = pool.stats();
        assert_eq!(stats.in_use_buffers, 0);
        // total_allocated_bytes counter must not overflow (it is a usize, saturating)
        assert!(stats.total_allocated_bytes <= budget);
    }

    #[test]
    fn test_sub_alloc_alloc_free_cycle_many() {
        let mut sa = SubAllocator::new(1024 * 1024, 16);
        for _ in 0..100 {
            let a = sa.alloc(256).expect("alloc in cycle");
            sa.free(a.id);
            sa.defrag();
        }
        assert_eq!(sa.allocation_count(), 0);
        assert_eq!(sa.utilization(), 0.0);
    }
}
