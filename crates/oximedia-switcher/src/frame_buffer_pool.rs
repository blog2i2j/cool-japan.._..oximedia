//! Frame buffer pool for reusable pixel data allocations.
//!
//! Eliminates per-frame heap allocation in `FrameSynchronizer` by maintaining a
//! free-list of `Vec<u8>` buffers. Callers acquire a buffer (popped from the
//! pool or freshly allocated if the pool is empty), use it for one frame, then
//! release it back (pushed onto the pool with capacity preserved but contents
//! cleared).
//!
//! The pool is bounded: once it holds `max_pool_size` idle buffers, excess
//! releases are dropped so memory does not grow unboundedly.

/// Pool of reusable raw byte buffers for video frame pixel data.
///
/// Acquiring a buffer returns a `Vec<u8>` whose length is initialised to
/// `size` bytes of zero.  Releasing it clears the contents but retains the
/// heap allocation for the next `acquire` call.
pub struct FrameBufferPool {
    /// Free buffers waiting to be reused.
    free_list: Vec<Vec<u8>>,
    /// Maximum number of idle buffers kept in the pool.
    max_pool_size: usize,
    /// Total number of acquires since the pool was created.
    total_acquires: u64,
    /// Total number of releases since the pool was created.
    total_releases: u64,
    /// Number of acquires that resulted in a fresh allocation.
    cold_allocs: u64,
}

impl FrameBufferPool {
    /// Create a new pool with `max_pool_size` as the upper bound on idle buffers.
    pub fn new(max_pool_size: usize) -> Self {
        Self {
            free_list: Vec::with_capacity(max_pool_size),
            max_pool_size,
            total_acquires: 0,
            total_releases: 0,
            cold_allocs: 0,
        }
    }

    /// Acquire a buffer of exactly `size` bytes, all initialised to zero.
    ///
    /// If a suitable buffer is available in the pool it is reused; otherwise a
    /// fresh `Vec` is allocated.
    pub fn acquire(&mut self, size: usize) -> Vec<u8> {
        self.total_acquires += 1;

        // Find the first pooled buffer with sufficient capacity.
        let pos = self.free_list.iter().position(|b| b.capacity() >= size);

        if let Some(idx) = pos {
            let mut buf = self.free_list.swap_remove(idx);
            // Safety: capacity >= size, all bytes set to 0 below.
            buf.clear();
            buf.resize(size, 0u8);
            buf
        } else {
            self.cold_allocs += 1;
            vec![0u8; size]
        }
    }

    /// Release a buffer back to the pool for future reuse.
    ///
    /// The buffer's contents are cleared (length set to 0) but the underlying
    /// heap allocation is retained.  Excess buffers beyond `max_pool_size` are
    /// simply dropped.
    pub fn release(&mut self, mut buf: Vec<u8>) {
        self.total_releases += 1;
        if self.free_list.len() < self.max_pool_size {
            buf.clear();
            self.free_list.push(buf);
        }
        // Else: drop the buffer — pool is at capacity.
    }

    /// Number of buffers currently sitting idle in the pool.
    pub fn idle_count(&self) -> usize {
        self.free_list.len()
    }

    /// Maximum number of idle buffers the pool will retain.
    pub fn max_pool_size(&self) -> usize {
        self.max_pool_size
    }

    /// Total number of `acquire` calls since creation.
    pub fn total_acquires(&self) -> u64 {
        self.total_acquires
    }

    /// Total number of `release` calls since creation.
    pub fn total_releases(&self) -> u64 {
        self.total_releases
    }

    /// Number of acquires that required a fresh heap allocation (cache miss).
    pub fn cold_allocs(&self) -> u64 {
        self.cold_allocs
    }

    /// Hit rate (0.0 – 1.0) of reuses vs total acquires.
    ///
    /// Returns `0.0` when no acquires have been made yet.
    pub fn hit_rate(&self) -> f64 {
        if self.total_acquires == 0 {
            return 0.0;
        }
        let hits = self.total_acquires.saturating_sub(self.cold_allocs);
        hits as f64 / self.total_acquires as f64
    }

    /// Drain all idle buffers, freeing their heap allocations.
    pub fn clear(&mut self) {
        self.free_list.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_acquire_fresh_when_empty() {
        let mut pool = FrameBufferPool::new(4);
        let buf = pool.acquire(1024);
        assert_eq!(buf.len(), 1024);
        assert!(buf.iter().all(|&b| b == 0));
        assert_eq!(pool.cold_allocs(), 1);
    }

    #[test]
    fn test_pool_release_and_reuse() {
        let mut pool = FrameBufferPool::new(4);
        let buf = pool.acquire(512);
        pool.release(buf);
        assert_eq!(pool.idle_count(), 1);

        // Second acquire should hit the pool.
        let buf2 = pool.acquire(512);
        assert_eq!(buf2.len(), 512);
        assert_eq!(pool.idle_count(), 0);
        assert_eq!(pool.cold_allocs(), 1); // only first was cold
    }

    #[test]
    fn test_pool_reuse_clears_contents() {
        let mut pool = FrameBufferPool::new(4);
        let mut buf = pool.acquire(64);
        // Write non-zero data.
        for b in &mut buf {
            *b = 0xFF;
        }
        pool.release(buf);

        // Should come back zeroed.
        let buf2 = pool.acquire(64);
        assert!(buf2.iter().all(|&b| b == 0), "reused buffer must be zeroed");
    }

    #[test]
    fn test_pool_bounded_idle_count() {
        let mut pool = FrameBufferPool::new(2);
        for _ in 0..5 {
            let buf = pool.acquire(128);
            pool.release(buf);
        }
        // Pool must not exceed max_pool_size.
        assert!(pool.idle_count() <= 2);
    }

    #[test]
    fn test_pool_hit_rate() {
        let mut pool = FrameBufferPool::new(4);
        // First acquire is cold.
        let buf = pool.acquire(256);
        pool.release(buf);
        // Second is a hit.
        let buf2 = pool.acquire(256);
        pool.release(buf2);

        let rate = pool.hit_rate();
        assert!(rate > 0.0 && rate <= 1.0);
        assert_eq!(pool.total_acquires(), 2);
        assert_eq!(pool.cold_allocs(), 1);
    }

    #[test]
    fn test_pool_clear() {
        let mut pool = FrameBufferPool::new(4);
        let buf = pool.acquire(128);
        pool.release(buf);
        assert_eq!(pool.idle_count(), 1);
        pool.clear();
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_pool_acquire_smaller_from_larger_capacity() {
        let mut pool = FrameBufferPool::new(4);
        // Acquire large buffer.
        let big = pool.acquire(1024);
        pool.release(big);

        // Acquire smaller — should reuse the large capacity buffer.
        let small = pool.acquire(256);
        assert_eq!(small.len(), 256);
        assert_eq!(pool.cold_allocs(), 1); // only the initial large alloc was cold
    }

    #[test]
    fn test_pool_stats_tracking() {
        let mut pool = FrameBufferPool::new(4);
        assert_eq!(pool.total_acquires(), 0);
        assert_eq!(pool.total_releases(), 0);

        let buf = pool.acquire(64);
        assert_eq!(pool.total_acquires(), 1);
        pool.release(buf);
        assert_eq!(pool.total_releases(), 1);
    }
}
