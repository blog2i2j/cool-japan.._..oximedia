//! GPU-style memory arena allocator.
//!
//! Provides a bump-allocator arena that carves sub-allocations from a
//! contiguous slab.  Useful for grouping per-frame GPU buffer allocations
//! so they can be freed in a single operation at the end of the frame.

#![allow(dead_code)]

use std::collections::HashMap;
use std::fmt;

/// Unique identifier for an arena allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AllocId(u64);

impl fmt::Display for AllocId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "alloc#{}", self.0)
    }
}

/// Metadata for a single sub-allocation inside the arena.
#[derive(Debug, Clone, Copy)]
pub struct AllocRecord {
    /// Byte offset from the start of the arena.
    pub offset: usize,
    /// Size in bytes.
    pub size: usize,
    /// Alignment that was requested.
    pub alignment: usize,
    /// Allocation id.
    pub id: AllocId,
}

/// Allocation strategy hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocStrategy {
    /// Simple bump / linear allocation (fastest, no individual free).
    Linear,
    /// Best-fit free-list (allows individual free, slower).
    BestFit,
}

/// Statistics for the arena.
#[derive(Debug, Clone, Default)]
pub struct ArenaStats {
    /// Total capacity in bytes.
    pub capacity: usize,
    /// Currently used bytes (including alignment padding).
    pub used: usize,
    /// Peak used bytes observed.
    pub peak_used: usize,
    /// Total number of allocations performed.
    pub alloc_count: u64,
    /// Total number of resets performed.
    pub reset_count: u64,
}

impl ArenaStats {
    /// Fraction of arena currently in use (0.0 .. 1.0).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.used as f64 / self.capacity as f64
    }

    /// Fraction of arena used at peak.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn peak_utilization(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        self.peak_used as f64 / self.capacity as f64
    }

    /// Remaining free bytes.
    #[must_use]
    pub fn free_bytes(&self) -> usize {
        self.capacity.saturating_sub(self.used)
    }
}

/// A bump-allocator memory arena.
///
/// Allocations are served from a contiguous virtual range and
/// freed collectively via [`MemoryArena::reset`].
pub struct MemoryArena {
    /// Total capacity in bytes.
    capacity: usize,
    /// Current write cursor (next free offset).
    cursor: usize,
    /// Running allocation id counter.
    next_id: u64,
    /// Record of live allocations.
    records: HashMap<AllocId, AllocRecord>,
    /// Strategy hint (stored for introspection; behaviour is always linear).
    strategy: AllocStrategy,
    /// Running statistics.
    stats: ArenaStats,
}

impl MemoryArena {
    /// Create a new arena with the given byte capacity.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity,
            cursor: 0,
            next_id: 0,
            records: HashMap::new(),
            strategy: AllocStrategy::Linear,
            stats: ArenaStats {
                capacity,
                ..ArenaStats::default()
            },
        }
    }

    /// Create a new arena with a specific strategy hint.
    #[must_use]
    pub fn with_strategy(capacity: usize, strategy: AllocStrategy) -> Self {
        let mut arena = Self::new(capacity);
        arena.strategy = strategy;
        arena
    }

    /// Total capacity in bytes.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Currently used bytes.
    #[must_use]
    pub fn used(&self) -> usize {
        self.cursor
    }

    /// Remaining free bytes.
    #[must_use]
    pub fn remaining(&self) -> usize {
        self.capacity.saturating_sub(self.cursor)
    }

    /// The strategy hint for this arena.
    #[must_use]
    pub fn strategy(&self) -> AllocStrategy {
        self.strategy
    }

    /// Number of live allocations.
    #[must_use]
    pub fn live_alloc_count(&self) -> usize {
        self.records.len()
    }

    /// Allocate `size` bytes with the given alignment.
    ///
    /// Returns `None` if the arena cannot satisfy the request.
    pub fn allocate(&mut self, size: usize, alignment: usize) -> Option<AllocRecord> {
        let align = alignment.max(1);
        // Round cursor up to alignment
        let aligned_offset = (self.cursor + align - 1) & !(align - 1);
        let end = aligned_offset.checked_add(size)?;
        if end > self.capacity {
            return None;
        }
        let id = AllocId(self.next_id);
        self.next_id += 1;
        let record = AllocRecord {
            offset: aligned_offset,
            size,
            alignment: align,
            id,
        };
        self.cursor = end;
        self.records.insert(id, record);
        self.stats.alloc_count += 1;
        self.stats.used = self.cursor;
        if self.cursor > self.stats.peak_used {
            self.stats.peak_used = self.cursor;
        }
        Some(record)
    }

    /// Convenience: allocate with default alignment of 1.
    pub fn allocate_unaligned(&mut self, size: usize) -> Option<AllocRecord> {
        self.allocate(size, 1)
    }

    /// Look up a record by its id.
    #[must_use]
    pub fn get_record(&self, id: AllocId) -> Option<&AllocRecord> {
        self.records.get(&id)
    }

    /// Reset the arena, freeing all allocations.
    pub fn reset(&mut self) {
        self.cursor = 0;
        self.records.clear();
        self.stats.used = 0;
        self.stats.reset_count += 1;
    }

    /// Snapshot of arena statistics.
    #[must_use]
    pub fn stats(&self) -> &ArenaStats {
        &self.stats
    }

    /// Resize the arena capacity. If the new capacity is smaller than the
    /// current cursor, this effectively invalidates existing allocations.
    pub fn resize(&mut self, new_capacity: usize) {
        self.capacity = new_capacity;
        self.stats.capacity = new_capacity;
        if self.cursor > new_capacity {
            self.cursor = new_capacity;
            self.stats.used = new_capacity;
        }
    }
}

impl fmt::Debug for MemoryArena {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryArena")
            .field("capacity", &self.capacity)
            .field("used", &self.cursor)
            .field("live_allocs", &self.records.len())
            .field("strategy", &self.strategy)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_id_display() {
        let id = AllocId(42);
        assert_eq!(id.to_string(), "alloc#42");
    }

    #[test]
    fn test_new_arena() {
        let arena = MemoryArena::new(1024);
        assert_eq!(arena.capacity(), 1024);
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.remaining(), 1024);
    }

    #[test]
    fn test_simple_allocation() {
        let mut arena = MemoryArena::new(256);
        let rec = arena.allocate(64, 1).expect("rec should be valid");
        assert_eq!(rec.offset, 0);
        assert_eq!(rec.size, 64);
        assert_eq!(arena.used(), 64);
        assert_eq!(arena.remaining(), 192);
    }

    #[test]
    fn test_aligned_allocation() {
        let mut arena = MemoryArena::new(256);
        arena.allocate(10, 1).expect("allocate should succeed"); // cursor at 10
        let rec = arena.allocate(32, 16).expect("rec should be valid"); // should align to 16
        assert_eq!(rec.offset, 16);
        assert_eq!(rec.size, 32);
    }

    #[test]
    fn test_allocation_overflow() {
        let mut arena = MemoryArena::new(64);
        assert!(arena.allocate(65, 1).is_none());
        assert_eq!(arena.live_alloc_count(), 0);
    }

    #[test]
    fn test_multiple_allocations() {
        let mut arena = MemoryArena::new(1024);
        for i in 0..10 {
            let rec = arena.allocate(32, 1).expect("rec should be valid");
            assert_eq!(rec.offset, i * 32);
        }
        assert_eq!(arena.live_alloc_count(), 10);
        assert_eq!(arena.used(), 320);
    }

    #[test]
    fn test_reset() {
        let mut arena = MemoryArena::new(256);
        arena.allocate(100, 1).expect("allocate should succeed");
        arena.allocate(50, 1).expect("allocate should succeed");
        arena.reset();
        assert_eq!(arena.used(), 0);
        assert_eq!(arena.live_alloc_count(), 0);
        assert_eq!(arena.stats().reset_count, 1);
    }

    #[test]
    fn test_peak_tracking() {
        let mut arena = MemoryArena::new(512);
        arena.allocate(200, 1).expect("allocate should succeed");
        arena.allocate(100, 1).expect("allocate should succeed");
        assert_eq!(arena.stats().peak_used, 300);
        arena.reset();
        arena.allocate(50, 1).expect("allocate should succeed");
        // peak should still be 300
        assert_eq!(arena.stats().peak_used, 300);
    }

    #[test]
    fn test_get_record() {
        let mut arena = MemoryArena::new(256);
        let rec = arena.allocate(16, 1).expect("rec should be valid");
        let found = arena.get_record(rec.id).expect("found should be valid");
        assert_eq!(found.offset, 0);
        assert_eq!(found.size, 16);
    }

    #[test]
    fn test_stats_utilization() {
        let mut arena = MemoryArena::new(100);
        arena.allocate(50, 1).expect("allocate should succeed");
        let s = arena.stats();
        assert!((s.utilization() - 0.5).abs() < 1e-9);
        assert_eq!(s.free_bytes(), 50);
    }

    #[test]
    fn test_strategy_hint() {
        let arena = MemoryArena::with_strategy(1024, AllocStrategy::BestFit);
        assert_eq!(arena.strategy(), AllocStrategy::BestFit);
    }

    #[test]
    fn test_resize_larger() {
        let mut arena = MemoryArena::new(100);
        arena.allocate(80, 1).expect("allocate should succeed");
        arena.resize(200);
        assert_eq!(arena.capacity(), 200);
        assert_eq!(arena.remaining(), 120);
    }

    #[test]
    fn test_resize_smaller_than_cursor() {
        let mut arena = MemoryArena::new(200);
        arena.allocate(150, 1).expect("allocate should succeed");
        arena.resize(100);
        assert_eq!(arena.capacity(), 100);
        // cursor clamped to capacity
        assert_eq!(arena.used(), 100);
    }

    #[test]
    fn test_allocate_unaligned() {
        let mut arena = MemoryArena::new(64);
        let rec = arena.allocate_unaligned(10).expect("rec should be valid");
        assert_eq!(rec.alignment, 1);
        assert_eq!(rec.offset, 0);
    }
}
