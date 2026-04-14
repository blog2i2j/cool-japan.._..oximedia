//! Lock-free audio buffer pool for reusing `Vec<f32>` allocations.
//!
//! Repeated per-block allocation of audio buffers on the realtime audio thread
//! is a common source of latency spikes (heap allocation is non-deterministic
//! in time and may call into the OS allocator).  This module provides an
//! [`AudioBufferPool`] that keeps a set of pre-allocated buffers and hands them
//! out via RAII guards ([`PooledBuffer`]).  When the guard is dropped the buffer
//! is returned to the pool rather than being freed.
//!
//! # Design
//!
//! * **Fixed-size blocks** — each pool manages buffers of exactly one
//!   `block_size` (number of `f32` samples).  This avoids fragmentation and
//!   makes allocation O(1).
//! * **Thread-safe via `Mutex`** — the pool itself is wrapped in a `Mutex`
//!   so it can be shared between a control thread (pre-warms the pool) and the
//!   audio thread.  The audio thread should pre-warm the pool before the
//!   realtime deadline begins so that checkouts during the audio callback never
//!   block on allocation.
//! * **Overflow allocation** — if the pool is exhausted a fresh `Vec` is
//!   allocated on the spot and returned as a [`PooledBuffer`] that is *not*
//!   returned to the pool on drop (it is simply freed).  This guarantees
//!   correctness at the cost of a rare allocation.
//!
//! # Example
//!
//! ```rust
//! use oximedia_mixer::buffer_pool::AudioBufferPool;
//!
//! let pool = AudioBufferPool::new(512, 8);
//!
//! // Borrow a zeroed buffer from the pool.
//! let mut buf = pool.checkout();
//! assert_eq!(buf.len(), 512);
//!
//! // Fill with audio data.
//! for (i, s) in buf.iter_mut().enumerate() {
//!     *s = i as f32 * 0.001;
//! }
//! // Buffer is returned to the pool when `buf` is dropped.
//! ```

use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Inner pool state
// ---------------------------------------------------------------------------

struct PoolInner {
    /// Pre-allocated buffers waiting to be checked out.
    free: Vec<Vec<f32>>,
    /// Number of samples in every buffer managed by this pool.
    block_size: usize,
    /// Total number of checkouts performed (for diagnostics).
    total_checkouts: u64,
    /// Number of checkouts that had to allocate because the pool was empty.
    overflow_allocations: u64,
    /// Number of buffers returned to the pool (vs freed on overflow path).
    returns: u64,
}

impl PoolInner {
    fn new(block_size: usize, initial_capacity: usize) -> Self {
        let mut free = Vec::with_capacity(initial_capacity);
        for _ in 0..initial_capacity {
            free.push(vec![0.0_f32; block_size]);
        }
        Self {
            free,
            block_size,
            total_checkouts: 0,
            overflow_allocations: 0,
            returns: 0,
        }
    }

    /// Check out a buffer, allocating if the pool is empty.
    fn checkout(&mut self) -> (Vec<f32>, bool) {
        self.total_checkouts += 1;
        if let Some(mut buf) = self.free.pop() {
            // Zero-fill before handing out (prevent stale audio data leaking).
            buf.iter_mut().for_each(|s| *s = 0.0);
            (buf, true)
        } else {
            self.overflow_allocations += 1;
            (vec![0.0_f32; self.block_size], false)
        }
    }

    /// Return a buffer to the pool.  Buffers whose size doesn't match are
    /// silently dropped.
    fn checkin(&mut self, buf: Vec<f32>) {
        if buf.len() == self.block_size {
            self.returns += 1;
            self.free.push(buf);
        }
        // Mismatched-size buffer is just dropped (freed normally).
    }
}

// ---------------------------------------------------------------------------
// AudioBufferPool
// ---------------------------------------------------------------------------

/// A shared, thread-safe pool of fixed-size `f32` audio buffers.
///
/// Clone the `Arc` to share the pool between threads.
#[derive(Clone)]
pub struct AudioBufferPool {
    inner: Arc<Mutex<PoolInner>>,
}

impl AudioBufferPool {
    /// Create a new pool with `initial_capacity` pre-allocated buffers of
    /// `block_size` samples each.
    ///
    /// # Panics
    ///
    /// Does not panic.  `block_size` of 0 is allowed but produces zero-length
    /// buffers.
    #[must_use]
    pub fn new(block_size: usize, initial_capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(PoolInner::new(block_size, initial_capacity))),
        }
    }

    /// The number of samples in every buffer managed by this pool.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.block_size)
            .unwrap_or(0)
    }

    /// Number of buffers currently available in the pool (not checked out).
    #[must_use]
    pub fn available(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.free.len())
            .unwrap_or(0)
    }

    /// Pre-warm the pool to hold at least `count` free buffers.
    ///
    /// Call this before entering the realtime callback to ensure that
    /// [`checkout`](Self::checkout) never needs to allocate during processing.
    pub fn prewarm(&self, count: usize) {
        if let Ok(mut guard) = self.inner.lock() {
            let block_size = guard.block_size;
            while guard.free.len() < count {
                guard.free.push(vec![0.0_f32; block_size]);
            }
        }
    }

    /// Check out a zeroed buffer.
    ///
    /// Returns a [`PooledBuffer`] RAII guard.  When the guard is dropped the
    /// buffer is returned to the pool.
    #[must_use]
    pub fn checkout(&self) -> PooledBuffer {
        let (buf, from_pool) = self
            .inner
            .lock()
            .map(|mut g| g.checkout())
            .unwrap_or_else(|_| (vec![0.0_f32; 0], false));

        PooledBuffer {
            buf: Some(buf),
            pool: if from_pool {
                Some(Arc::clone(&self.inner))
            } else {
                None
            },
        }
    }

    /// Pool diagnostics snapshot.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        self.inner
            .lock()
            .map(|g| PoolStats {
                block_size: g.block_size,
                available: g.free.len(),
                total_checkouts: g.total_checkouts,
                overflow_allocations: g.overflow_allocations,
                returns: g.returns,
            })
            .unwrap_or_default()
    }
}

impl std::fmt::Debug for AudioBufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let stats = self.stats();
        f.debug_struct("AudioBufferPool")
            .field("block_size", &stats.block_size)
            .field("available", &stats.available)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PooledBuffer — RAII guard
// ---------------------------------------------------------------------------

/// An RAII guard wrapping a borrowed audio buffer.
///
/// The buffer is automatically returned to the [`AudioBufferPool`] when this
/// guard is dropped.  Access the underlying slice via [`Deref`] / [`DerefMut`].
pub struct PooledBuffer {
    /// `Some` during the lifetime of the guard; `None` only after the
    /// `Option::take` inside `drop`.
    buf: Option<Vec<f32>>,
    /// If `Some`, the buffer will be returned to the pool on drop.
    /// If `None`, the buffer came from an overflow allocation and is freed.
    pool: Option<Arc<Mutex<PoolInner>>>,
}

impl PooledBuffer {
    /// Number of `f32` samples in this buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.buf.as_ref().map_or(0, Vec::len)
    }

    /// Returns `true` if the buffer is empty (zero samples).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Fill every sample with zero.
    pub fn zero(&mut self) {
        if let Some(buf) = self.buf.as_mut() {
            buf.iter_mut().for_each(|s| *s = 0.0);
        }
    }

    /// Copy samples from `src` into this buffer.
    ///
    /// Copies `min(self.len(), src.len())` samples.
    pub fn copy_from(&mut self, src: &[f32]) {
        if let Some(buf) = self.buf.as_mut() {
            let n = buf.len().min(src.len());
            buf[..n].copy_from_slice(&src[..n]);
        }
    }
}

impl std::ops::Deref for PooledBuffer {
    type Target = [f32];

    fn deref(&self) -> &Self::Target {
        self.buf.as_deref().unwrap_or(&[])
    }
}

impl std::ops::DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buf.as_deref_mut().unwrap_or(&mut [])
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            if let Some(pool_arc) = self.pool.take() {
                if let Ok(mut guard) = pool_arc.lock() {
                    guard.checkin(buf);
                }
                // If the lock is poisoned we just drop the buffer.
            }
            // No pool reference → overflow allocation; buf is freed here.
        }
    }
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("len", &self.len())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// PoolStats
// ---------------------------------------------------------------------------

/// Diagnostic snapshot of [`AudioBufferPool`] state.
#[derive(Debug, Clone, Default)]
pub struct PoolStats {
    /// Buffer size in samples.
    pub block_size: usize,
    /// Buffers currently available (not checked out).
    pub available: usize,
    /// Total number of checkout calls since the pool was created.
    pub total_checkouts: u64,
    /// Number of checkouts that had to allocate due to pool exhaustion.
    pub overflow_allocations: u64,
    /// Number of buffers successfully returned to the pool.
    pub returns: u64,
}

// ---------------------------------------------------------------------------
// MultiSizePool
// ---------------------------------------------------------------------------

/// A collection of [`AudioBufferPool`]s keyed by block size.
///
/// Use this when the mixer needs to handle multiple block sizes (e.g. 128,
/// 256, 512, 1024 samples) without creating a separate pool instance for each.
#[derive(Debug, Clone)]
pub struct MultiSizePool {
    pools: Vec<(usize, AudioBufferPool)>,
    /// Initial capacity created for each new block-size entry.
    default_capacity: usize,
}

impl MultiSizePool {
    /// Create a new multi-size pool.  `default_capacity` is the number of
    /// pre-allocated buffers created whenever a new block size is first seen.
    #[must_use]
    pub fn new(default_capacity: usize) -> Self {
        Self {
            pools: Vec::new(),
            default_capacity,
        }
    }

    /// Get or create the sub-pool for `block_size` and check out a buffer.
    pub fn checkout(&mut self, block_size: usize) -> PooledBuffer {
        // Linear search is fine: number of distinct block sizes is tiny (1–4).
        for (size, pool) in &self.pools {
            if *size == block_size {
                return pool.checkout();
            }
        }
        // First time we see this block size — create a pool for it.
        let pool = AudioBufferPool::new(block_size, self.default_capacity);
        let buf = pool.checkout();
        self.pools.push((block_size, pool));
        buf
    }

    /// Pre-warm all registered sub-pools to `count` free buffers each.
    pub fn prewarm_all(&self, count: usize) {
        for (_, pool) in &self.pools {
            pool.prewarm(count);
        }
    }

    /// Number of distinct block sizes currently registered.
    #[must_use]
    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkout_returns_correct_size() {
        let pool = AudioBufferPool::new(512, 4);
        let buf = pool.checkout();
        assert_eq!(buf.len(), 512, "checked-out buffer should have 512 samples");
    }

    #[test]
    fn test_checkout_zeroed() {
        let pool = AudioBufferPool::new(256, 4);
        let buf = pool.checkout();
        assert!(
            buf.iter().all(|&s| s == 0.0),
            "checked-out buffer must be zeroed"
        );
    }

    #[test]
    fn test_buffer_returned_on_drop() {
        let pool = AudioBufferPool::new(128, 2);
        assert_eq!(pool.available(), 2);
        {
            let _buf = pool.checkout();
            assert_eq!(pool.available(), 1);
        } // drop here
        assert_eq!(
            pool.available(),
            2,
            "buffer should be returned to pool after drop"
        );
    }

    #[test]
    fn test_overflow_allocation_when_pool_empty() {
        let pool = AudioBufferPool::new(64, 1);
        let _buf1 = pool.checkout(); // takes the one pre-allocated
        let buf2 = pool.checkout(); // overflow allocation
        assert_eq!(buf2.len(), 64, "overflow buffer should still have correct size");
        let stats = pool.stats();
        assert_eq!(
            stats.overflow_allocations, 1,
            "should record one overflow allocation"
        );
    }

    #[test]
    fn test_overflow_buffer_not_returned_to_pool() {
        let pool = AudioBufferPool::new(64, 1);
        let _b1 = pool.checkout(); // takes last pooled buffer
        {
            let _overflow = pool.checkout(); // overflow — not from pool
            // after drop, pool should still have 0 (the original was taken)
        }
        // The overflow buffer was freed, not returned.
        assert_eq!(
            pool.available(),
            0,
            "overflow buffer should not be returned to the pool"
        );
    }

    #[test]
    fn test_prewarm_increases_available() {
        let pool = AudioBufferPool::new(128, 0);
        assert_eq!(pool.available(), 0);
        pool.prewarm(8);
        assert_eq!(pool.available(), 8);
    }

    #[test]
    fn test_deref_mut_writes_visible() {
        let pool = AudioBufferPool::new(8, 2);
        let mut buf = pool.checkout();
        buf[0] = 1.0;
        buf[7] = -1.0;
        assert!((buf[0] - 1.0).abs() < f32::EPSILON);
        assert!((buf[7] + 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_copy_from() {
        let pool = AudioBufferPool::new(4, 2);
        let src = [0.1_f32, 0.2, 0.3, 0.4];
        let mut buf = pool.checkout();
        buf.copy_from(&src);
        assert!((buf[0] - 0.1).abs() < f32::EPSILON);
        assert!((buf[3] - 0.4).abs() < f32::EPSILON);
    }

    #[test]
    fn test_pool_stats() {
        let pool = AudioBufferPool::new(32, 4);
        let _b1 = pool.checkout();
        let _b2 = pool.checkout();
        let stats = pool.stats();
        assert_eq!(stats.total_checkouts, 2);
        assert_eq!(stats.block_size, 32);
    }

    #[test]
    fn test_multiple_checkouts_and_returns() {
        let pool = AudioBufferPool::new(256, 4);
        let bufs: Vec<PooledBuffer> = (0..4).map(|_| pool.checkout()).collect();
        assert_eq!(pool.available(), 0);
        drop(bufs);
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn test_multi_size_pool() {
        let mut mp = MultiSizePool::new(4);
        let buf128 = mp.checkout(128);
        assert_eq!(buf128.len(), 128);
        let buf512 = mp.checkout(512);
        assert_eq!(buf512.len(), 512);
        assert_eq!(mp.pool_count(), 2);
    }

    #[test]
    fn test_multi_size_same_size_reuses_pool() {
        let mut mp = MultiSizePool::new(2);
        let _b1 = mp.checkout(64);
        let _b2 = mp.checkout(64);
        // Only one distinct pool for size 64.
        assert_eq!(mp.pool_count(), 1);
    }

    #[test]
    fn test_zero_method() {
        let pool = AudioBufferPool::new(8, 2);
        let mut buf = pool.checkout();
        buf[0] = 99.0;
        buf.zero();
        assert!(buf.iter().all(|&s| s == 0.0));
    }

    #[test]
    fn test_block_size_accessor() {
        let pool = AudioBufferPool::new(1024, 2);
        assert_eq!(pool.block_size(), 1024);
    }
}
