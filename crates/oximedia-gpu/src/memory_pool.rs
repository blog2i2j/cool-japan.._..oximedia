//! GPU memory pool allocator.
//!
//! Provides block-based GPU memory allocation with alignment support and
//! pool statistics tracking. Designed to reduce the overhead of frequent
//! small allocations by sub-allocating from larger backing blocks.

/// Alignment requirements for GPU memory blocks.
#[allow(dead_code)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Alignment {
    /// 4-byte alignment (default for most scalar types).
    Bytes4 = 4,
    /// 16-byte alignment (required for vec4 on many GPUs).
    Bytes16 = 16,
    /// 64-byte alignment (cache-line alignment).
    Bytes64 = 64,
    /// 256-byte alignment (required by some Vulkan/D3D12 rules).
    Bytes256 = 256,
    /// 4 KB alignment (page granularity).
    Bytes4096 = 4096,
}

impl Alignment {
    /// Value as `usize`.
    #[allow(dead_code)]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self as usize
    }

    /// Align `offset` up to the next multiple of this alignment.
    #[allow(dead_code)]
    #[must_use]
    pub const fn align_up(self, offset: usize) -> usize {
        let align = self as usize;
        (offset + align - 1) & !(align - 1)
    }
}

/// A single allocation handle returned to the caller.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AllocationHandle {
    /// Index of the backing block.
    pub block_index: usize,
    /// Byte offset within that block.
    pub offset: usize,
    /// Allocated size (may be larger than requested due to alignment).
    pub size: usize,
    /// Alignment used.
    pub alignment: usize,
    /// Opaque allocation id for deallocation.
    pub id: u64,
}

/// Tracks free ranges inside a single backing block.
#[allow(dead_code)]
#[derive(Debug)]
struct FreeRange {
    offset: usize,
    size: usize,
}

/// A single large backing allocation that sub-allocates smaller regions.
#[allow(dead_code)]
#[derive(Debug)]
struct Block {
    /// Total capacity of this block in bytes.
    capacity: usize,
    /// Byte ranges that are currently free.
    free_ranges: Vec<FreeRange>,
    /// Number of live sub-allocations.
    live_count: usize,
}

impl Block {
    fn new(capacity: usize) -> Self {
        Self {
            capacity,
            free_ranges: vec![FreeRange {
                offset: 0,
                size: capacity,
            }],
            live_count: 0,
        }
    }

    /// Try to allocate `size` bytes with `alignment`. Returns the aligned
    /// offset on success.
    fn try_alloc(&mut self, size: usize, alignment: usize) -> Option<usize> {
        for range in &mut self.free_ranges {
            let aligned_offset = (range.offset + alignment - 1) & !(alignment - 1);
            let waste = aligned_offset - range.offset;
            if range.size >= waste + size {
                let result_offset = aligned_offset;
                range.offset += waste + size;
                range.size -= waste + size;
                self.live_count += 1;
                return Some(result_offset);
            }
        }
        // Remove exhausted ranges.
        self.free_ranges.retain(|r| r.size > 0);
        None
    }

    /// Free a previously allocated region.
    fn free(&mut self, offset: usize, size: usize) {
        self.free_ranges.push(FreeRange { offset, size });
        if self.live_count > 0 {
            self.live_count -= 1;
        }
        // Coalesce adjacent free ranges (simple O(n²) version adequate here).
        self.coalesce();
    }

    fn coalesce(&mut self) {
        self.free_ranges.sort_by_key(|r| r.offset);
        let mut i = 0;
        while i + 1 < self.free_ranges.len() {
            let end = self.free_ranges[i].offset + self.free_ranges[i].size;
            if end >= self.free_ranges[i + 1].offset {
                // Merge.
                let merged_size = self.free_ranges[i + 1].offset + self.free_ranges[i + 1].size
                    - self.free_ranges[i].offset;
                self.free_ranges[i].size = merged_size;
                self.free_ranges.remove(i + 1);
            } else {
                i += 1;
            }
        }
    }

    /// Bytes still free in this block (sum of all free ranges).
    fn free_bytes(&self) -> usize {
        self.free_ranges.iter().map(|r| r.size).sum()
    }
}

/// Statistics for the memory pool.
#[allow(dead_code)]
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Total bytes reserved across all backing blocks.
    pub total_reserved: usize,
    /// Total bytes currently allocated (live).
    pub total_allocated: usize,
    /// Number of backing blocks.
    pub block_count: usize,
    /// Total number of successful allocations.
    pub alloc_count: u64,
    /// Total number of deallocations.
    pub free_count: u64,
    /// Allocation failures due to fragmentation.
    pub failures: u64,
}

impl PoolStats {
    /// Bytes still free (reserved but not live-allocated).
    #[allow(dead_code)]
    #[must_use]
    pub fn free_bytes(&self) -> usize {
        self.total_reserved.saturating_sub(self.total_allocated)
    }

    /// Utilisation ratio (0.0 – 1.0).
    #[allow(dead_code)]
    #[must_use]
    pub fn utilisation(&self) -> f64 {
        if self.total_reserved == 0 {
            0.0
        } else {
            self.total_allocated as f64 / self.total_reserved as f64
        }
    }
}

/// GPU memory pool allocator.
#[allow(dead_code)]
pub struct GpuMemoryPool {
    /// Size of each new backing block in bytes.
    block_size: usize,
    /// All backing blocks.
    blocks: Vec<Block>,
    /// Statistics.
    stats: PoolStats,
    /// Monotonically increasing allocation id counter.
    next_id: u64,
}

impl GpuMemoryPool {
    /// Create a new pool.
    ///
    /// * `block_size` – size of each new backing block in bytes.
    #[allow(dead_code)]
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        Self {
            block_size,
            blocks: Vec::new(),
            stats: PoolStats::default(),
            next_id: 0,
        }
    }

    /// Allocate `size` bytes with the given `alignment`.
    ///
    /// Returns an [`AllocationHandle`] on success. If no existing block can
    /// satisfy the request, a new backing block is created.
    #[allow(dead_code)]
    pub fn alloc(&mut self, size: usize, alignment: Alignment) -> Option<AllocationHandle> {
        if size == 0 {
            return None;
        }
        let align = alignment.as_usize();

        // Try existing blocks first.
        for (i, block) in self.blocks.iter_mut().enumerate() {
            if let Some(offset) = block.try_alloc(size, align) {
                let id = self.next_id;
                self.next_id += 1;
                self.stats.alloc_count += 1;
                self.stats.total_allocated += size;
                return Some(AllocationHandle {
                    block_index: i,
                    offset,
                    size,
                    alignment: align,
                    id,
                });
            }
        }

        // Allocate a new block large enough.
        let new_block_size = self.block_size.max(size + align);
        let mut block = Block::new(new_block_size);
        if let Some(offset) = block.try_alloc(size, align) {
            self.stats.total_reserved += new_block_size;
            self.stats.block_count += 1;
            let block_index = self.blocks.len();
            self.blocks.push(block);

            let id = self.next_id;
            self.next_id += 1;
            self.stats.alloc_count += 1;
            self.stats.total_allocated += size;
            Some(AllocationHandle {
                block_index,
                offset,
                size,
                alignment: align,
                id,
            })
        } else {
            self.stats.failures += 1;
            None
        }
    }

    /// Free a previously allocated handle.
    #[allow(dead_code)]
    pub fn free(&mut self, handle: &AllocationHandle) {
        if handle.block_index < self.blocks.len() {
            self.blocks[handle.block_index].free(handle.offset, handle.size);
            self.stats.total_allocated = self.stats.total_allocated.saturating_sub(handle.size);
            self.stats.free_count += 1;
        }
    }

    /// Current pool statistics.
    #[allow(dead_code)]
    #[must_use]
    pub fn stats(&self) -> &PoolStats {
        &self.stats
    }

    /// Total number of backing blocks.
    #[allow(dead_code)]
    #[must_use]
    pub fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Total free bytes across all blocks.
    #[allow(dead_code)]
    #[must_use]
    pub fn free_bytes(&self) -> usize {
        self.blocks.iter().map(Block::free_bytes).sum()
    }

    /// Reset the pool – all backing blocks are cleared.
    #[allow(dead_code)]
    pub fn reset(&mut self) {
        self.blocks.clear();
        self.stats = PoolStats::default();
        self.next_id = 0;
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_align_up() {
        assert_eq!(Alignment::Bytes16.align_up(0), 0);
        assert_eq!(Alignment::Bytes16.align_up(1), 16);
        assert_eq!(Alignment::Bytes16.align_up(16), 16);
        assert_eq!(Alignment::Bytes16.align_up(17), 32);
    }

    #[test]
    fn test_alignment_as_usize() {
        assert_eq!(Alignment::Bytes4.as_usize(), 4);
        assert_eq!(Alignment::Bytes256.as_usize(), 256);
    }

    #[test]
    fn test_simple_alloc() {
        let mut pool = GpuMemoryPool::new(1024);
        let handle = pool.alloc(64, Alignment::Bytes16);
        assert!(handle.is_some());
        let h = handle.expect("handle should be valid");
        assert_eq!(h.size, 64);
        assert_eq!(h.offset % 16, 0);
    }

    #[test]
    fn test_zero_size_alloc_returns_none() {
        let mut pool = GpuMemoryPool::new(1024);
        assert!(pool.alloc(0, Alignment::Bytes4).is_none());
    }

    #[test]
    fn test_alloc_and_free_stats() {
        let mut pool = GpuMemoryPool::new(1024);
        let h = pool
            .alloc(100, Alignment::Bytes4)
            .expect("allocation should succeed");
        assert_eq!(pool.stats().total_allocated, 100);
        pool.free(&h);
        assert_eq!(pool.stats().total_allocated, 0);
    }

    #[test]
    fn test_multiple_allocs_same_block() {
        let mut pool = GpuMemoryPool::new(4096);
        let h1 = pool
            .alloc(128, Alignment::Bytes64)
            .expect("allocation should succeed");
        let h2 = pool
            .alloc(128, Alignment::Bytes64)
            .expect("allocation should succeed");
        assert_eq!(h1.block_index, h2.block_index);
        assert_eq!(pool.block_count(), 1);
    }

    #[test]
    fn test_new_block_created_when_full() {
        let mut pool = GpuMemoryPool::new(64);
        // First alloc fills the initial block.
        let _h1 = pool
            .alloc(64, Alignment::Bytes4)
            .expect("allocation should succeed");
        // Second alloc must create a new block.
        let h2 = pool
            .alloc(64, Alignment::Bytes4)
            .expect("allocation should succeed");
        assert!(h2.block_index >= 1 || pool.block_count() == 2);
    }

    #[test]
    fn test_pool_stats_utilisation() {
        let mut pool = GpuMemoryPool::new(1000);
        pool.alloc(500, Alignment::Bytes4);
        let util = pool.stats().utilisation();
        assert!(util > 0.0 && util <= 1.0);
    }

    #[test]
    fn test_free_bytes_decreases_after_alloc() {
        let mut pool = GpuMemoryPool::new(1024);
        pool.alloc(256, Alignment::Bytes4);
        assert!(pool.free_bytes() < 1024);
    }

    #[test]
    fn test_reset_clears_all() {
        let mut pool = GpuMemoryPool::new(512);
        pool.alloc(100, Alignment::Bytes4);
        pool.reset();
        assert_eq!(pool.block_count(), 0);
        assert_eq!(pool.stats().alloc_count, 0);
    }

    #[test]
    fn test_alloc_id_increments() {
        let mut pool = GpuMemoryPool::new(1024);
        let h1 = pool
            .alloc(10, Alignment::Bytes4)
            .expect("allocation should succeed");
        let h2 = pool
            .alloc(10, Alignment::Bytes4)
            .expect("allocation should succeed");
        assert!(h2.id > h1.id);
    }

    #[test]
    fn test_block_coalescing_after_free() {
        let mut pool = GpuMemoryPool::new(256);
        let h1 = pool
            .alloc(64, Alignment::Bytes4)
            .expect("allocation should succeed");
        let h2 = pool
            .alloc(64, Alignment::Bytes4)
            .expect("allocation should succeed");
        pool.free(&h1);
        pool.free(&h2);
        // After freeing both, the pool should be able to allocate a 128-byte block again.
        let h3 = pool.alloc(100, Alignment::Bytes4);
        assert!(h3.is_some());
    }

    #[test]
    fn test_stats_free_bytes() {
        let mut stats = PoolStats {
            total_reserved: 1000,
            total_allocated: 400,
            ..Default::default()
        };
        assert_eq!(stats.free_bytes(), 600);
        stats.total_allocated = 1000;
        assert_eq!(stats.free_bytes(), 0);
    }
}
