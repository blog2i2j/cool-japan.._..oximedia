//! Buffer pool for encryption/decryption operations.
//!
//! Maintains a thread-safe pool of reusable byte buffers to reduce allocation
//! pressure during high-throughput DRM operations (e.g. per-segment CENC
//! encryption, PSSH box assembly, license payload marshalling).
//!
//! # Design
//!
//! Each pool holds buffers of a fixed capacity bucket. When a buffer is
//! requested and a suitable one exists in the pool it is leased immediately
//! without allocation. When the buffer is dropped it is returned to the pool
//! automatically via [`BufGuard`]'s `Drop` implementation.
//!
//! Multiple pools with different bucket sizes can coexist; callers should
//! choose the bucket size closest to (but not smaller than) the expected
//! working set size to minimise wasted capacity.

use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// BufPool
// ---------------------------------------------------------------------------

/// A pool of reusable byte buffers of a fixed bucket capacity.
///
/// # Example
///
/// ```rust
/// use oximedia_drm::buf_pool::BufPool;
///
/// let pool = BufPool::new(4096, 8);
/// let mut buf = pool.acquire();
/// buf.extend_from_slice(b"hello");
/// // buf is returned to the pool when dropped
/// ```
#[derive(Clone)]
pub struct BufPool {
    inner: Arc<Mutex<BufPoolInner>>,
    bucket_cap: usize,
}

struct BufPoolInner {
    free: Vec<Vec<u8>>,
    max_pooled: usize,
    total_acquired: u64,
    total_returned: u64,
}

impl BufPool {
    /// Create a new buffer pool.
    ///
    /// - `bucket_cap`: capacity in bytes of each pooled buffer.
    /// - `max_pooled`: maximum number of idle buffers retained in the pool.
    ///   Once the pool is full, returned buffers are discarded.
    pub fn new(bucket_cap: usize, max_pooled: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(BufPoolInner {
                free: Vec::with_capacity(max_pooled),
                max_pooled,
                total_acquired: 0,
                total_returned: 0,
            })),
            bucket_cap,
        }
    }

    /// Acquire a buffer from the pool, allocating a new one if none are free.
    ///
    /// The buffer is cleared (length = 0) but retains its allocated capacity.
    /// The caller owns the buffer until it (or the [`BufGuard`] wrapping it)
    /// is dropped.
    pub fn acquire(&self) -> BufGuard {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        inner.total_acquired += 1;
        let buf = if let Some(mut b) = inner.free.pop() {
            b.clear();
            b
        } else {
            Vec::with_capacity(self.bucket_cap)
        };
        BufGuard {
            buf: Some(buf),
            pool: self.inner.clone(),
        }
    }

    /// Number of buffers currently idle in the pool.
    pub fn idle_count(&self) -> usize {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .free
            .len()
    }

    /// Cumulative count of acquire calls (for metrics/monitoring).
    pub fn total_acquired(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .total_acquired
    }

    /// Cumulative count of return calls (for metrics/monitoring).
    pub fn total_returned(&self) -> u64 {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .total_returned
    }

    /// Pre-warm the pool by allocating `count` buffers immediately.
    ///
    /// Call this during startup to avoid allocation latency on the hot path.
    pub fn pre_warm(&self, count: usize) {
        let mut inner = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        let to_add = count.min(inner.max_pooled.saturating_sub(inner.free.len()));
        for _ in 0..to_add {
            inner.free.push(Vec::with_capacity(self.bucket_cap));
        }
    }
}

// ---------------------------------------------------------------------------
// BufGuard — RAII wrapper that returns the buffer to the pool on drop
// ---------------------------------------------------------------------------

/// A leased buffer from a [`BufPool`].
///
/// Derefs to `Vec<u8>` so it can be used wherever a `Vec<u8>` is expected.
/// On `Drop` the buffer is returned to the originating pool (unless the pool
/// is at capacity, in which case it is deallocated normally).
pub struct BufGuard {
    /// `Option` to allow `take()` in Drop without unsafe code.
    buf: Option<Vec<u8>>,
    pool: Arc<Mutex<BufPoolInner>>,
}

impl BufGuard {
    /// Consume the guard and take ownership of the inner `Vec<u8>`.
    ///
    /// The buffer will **not** be returned to the pool. Use this only when
    /// you need an owned `Vec<u8>` (e.g. to pass to an async boundary).
    pub fn into_vec(mut self) -> Vec<u8> {
        self.buf.take().unwrap_or_default()
    }
}

/// A shared empty `Vec<u8>` used as a last-resort fallback in `Deref` to avoid
/// panicking when `BufGuard` is dereffed after `into_vec()` has been called.
/// In well-formed code this path is never hit.
static EMPTY_BUF: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();

impl std::ops::Deref for BufGuard {
    type Target = Vec<u8>;

    fn deref(&self) -> &Self::Target {
        self.buf
            .as_ref()
            .unwrap_or_else(|| EMPTY_BUF.get_or_init(Vec::new))
    }
}

impl std::ops::DerefMut for BufGuard {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // If buf is None (after into_vec), we have no mutable reference to
        // return without unsafe; instead we re-initialise with an empty vec so
        // the caller does not panic.
        if self.buf.is_none() {
            self.buf = Some(Vec::new());
        }
        self.buf.as_mut().unwrap_or_else(|| unreachable!())
    }
}

impl Drop for BufGuard {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            let mut inner = self.pool.lock().unwrap_or_else(|e| e.into_inner());
            inner.total_returned += 1;
            if inner.free.len() < inner.max_pooled {
                let mut ret = buf;
                ret.clear();
                inner.free.push(ret);
            }
            // If pool is full, buf is dropped here (normal deallocation).
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-tier pool set
// ---------------------------------------------------------------------------

/// A set of [`BufPool`]s covering different size buckets.
///
/// Provides a convenient `acquire(size)` method that selects the smallest
/// bucket that is large enough for the requested size.
pub struct BufPoolSet {
    /// Sorted list of (capacity, pool) pairs.
    pools: Vec<(usize, BufPool)>,
}

impl BufPoolSet {
    /// Create a pool set covering the given capacities.
    ///
    /// Capacities are deduplicated and sorted. `max_pooled` applies to each
    /// tier uniformly.
    pub fn new(mut capacities: Vec<usize>, max_pooled: usize) -> Self {
        capacities.sort_unstable();
        capacities.dedup();
        let pools = capacities
            .into_iter()
            .map(|cap| (cap, BufPool::new(cap, max_pooled)))
            .collect();
        Self { pools }
    }

    /// Acquire a buffer of at least `min_size` bytes from the appropriate
    /// tier. If no tier is large enough a new buffer with exactly `min_size`
    /// capacity is allocated and returned without pooling.
    pub fn acquire(&self, min_size: usize) -> BufGuard {
        for (cap, pool) in &self.pools {
            if *cap >= min_size {
                return pool.acquire();
            }
        }
        // Fallback: allocate directly, return a guard tied to an empty pool
        // so that the buffer is simply deallocated on drop.
        let fallback_pool = BufPool::new(min_size, 0);
        fallback_pool.acquire()
    }

    /// Number of pool tiers.
    pub fn tier_count(&self) -> usize {
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
    fn test_acquire_and_use_buf() {
        let pool = BufPool::new(256, 4);
        let mut buf = pool.acquire();
        buf.extend_from_slice(b"encryption payload");
        assert_eq!(&**buf, b"encryption payload");
    }

    #[test]
    fn test_buf_returned_on_drop() {
        let pool = BufPool::new(128, 4);
        assert_eq!(pool.idle_count(), 0);
        {
            let _buf = pool.acquire();
            assert_eq!(pool.idle_count(), 0);
        }
        assert_eq!(pool.idle_count(), 1);
    }

    #[test]
    fn test_reuse_pooled_buf() {
        let pool = BufPool::new(64, 2);
        let buf1 = pool.acquire();
        drop(buf1);
        // The buffer should be reused
        assert_eq!(pool.idle_count(), 1);
        let _buf2 = pool.acquire();
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_buf_cleared_on_acquire() {
        let pool = BufPool::new(128, 2);
        {
            let mut buf = pool.acquire();
            buf.extend_from_slice(b"old data");
        }
        // New acquisition should be empty
        let buf = pool.acquire();
        assert!(buf.is_empty());
    }

    #[test]
    fn test_pool_max_capacity_limit() {
        let pool = BufPool::new(64, 2);
        // Acquire and release 5 buffers; only max_pooled=2 should be retained
        for _ in 0..5 {
            let _buf = pool.acquire();
        }
        assert!(pool.idle_count() <= 2);
    }

    #[test]
    fn test_total_acquired_counter() {
        let pool = BufPool::new(64, 4);
        for _ in 0..3 {
            let _buf = pool.acquire();
        }
        assert_eq!(pool.total_acquired(), 3);
    }

    #[test]
    fn test_total_returned_counter() {
        let pool = BufPool::new(64, 4);
        for _ in 0..3 {
            let buf = pool.acquire();
            drop(buf);
        }
        assert_eq!(pool.total_returned(), 3);
    }

    #[test]
    fn test_into_vec_does_not_return_to_pool() {
        let pool = BufPool::new(64, 4);
        let mut buf = pool.acquire();
        buf.extend_from_slice(b"take me");
        let v = buf.into_vec();
        assert_eq!(v, b"take me");
        // Buffer was not returned to pool
        assert_eq!(pool.idle_count(), 0);
    }

    #[test]
    fn test_pre_warm() {
        let pool = BufPool::new(64, 8);
        pool.pre_warm(4);
        assert_eq!(pool.idle_count(), 4);
    }

    #[test]
    fn test_pre_warm_respects_max_pooled() {
        let pool = BufPool::new(64, 3);
        pool.pre_warm(10);
        assert_eq!(pool.idle_count(), 3);
    }

    #[test]
    fn test_clone_shares_pool() {
        let pool = BufPool::new(64, 4);
        let pool2 = pool.clone();
        {
            let _buf = pool.acquire();
        }
        // pool and pool2 share the same inner pool
        assert_eq!(pool2.idle_count(), 1);
    }

    #[test]
    fn test_pool_set_selects_smallest_bucket() {
        let set = BufPoolSet::new(vec![256, 1024, 4096], 4);
        assert_eq!(set.tier_count(), 3);

        // For 200 bytes, should use the 256-byte bucket
        let buf = set.acquire(200);
        assert!(buf.capacity() >= 200);
    }

    #[test]
    fn test_pool_set_fallback_for_oversized() {
        let set = BufPoolSet::new(vec![128, 256], 4);
        // Request larger than any tier
        let buf = set.acquire(512);
        assert!(buf.capacity() >= 512);
    }

    #[test]
    fn test_pool_set_empty_buf_at_acquire() {
        let set = BufPoolSet::new(vec![64, 128], 4);
        let buf = set.acquire(64);
        assert!(buf.is_empty());
    }

    #[test]
    fn test_concurrent_acquire_and_drop() {
        use std::thread;

        let pool = BufPool::new(128, 16);
        let mut handles = Vec::new();

        for _ in 0..8 {
            let p = pool.clone();
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    let mut buf = p.acquire();
                    buf.extend_from_slice(b"data");
                    // buf dropped at end of loop iteration
                }
            }));
        }

        for h in handles {
            h.join().expect("thread should not panic");
        }

        // Pool should have some idle buffers and consistent counters
        assert!(pool.total_acquired() > 0);
        assert_eq!(pool.total_acquired(), pool.total_returned());
    }
}
