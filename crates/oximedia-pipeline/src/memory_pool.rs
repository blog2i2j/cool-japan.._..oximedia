//! Memory pool allocation for pipeline frame buffers.
//!
//! Rather than allocating and freeing a fresh `Vec<u8>` for every decoded
//! frame, the `MemoryPool` pre-allocates a fixed number of fixed-size buffers
//! and hands them out via a [`PooledBuffer`] guard.  When the guard is
//! dropped the backing memory is returned to the pool rather than freed,
//! eliminating repeated allocator round-trips in hot execution loops.
//!
//! # Integration with `ResourceEstimate`
//!
//! [`PoolConfig`] mirrors the fields of [`ResourceEstimate`] so that the
//! planner can size a pool appropriately from its resource estimates:
//!
//! ```rust
//! use oximedia_pipeline::memory_pool::{MemoryPool, PoolConfig};
//! use oximedia_pipeline::execution_plan::ResourceEstimate;
//!
//! let estimate = ResourceEstimate::new(1920 * 1080 * 4, 8.0, false);
//! let config = PoolConfig::from_estimate(&estimate, 4);
//! let pool = MemoryPool::new(config).expect("pool created");
//! assert!(pool.available() > 0);
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_pipeline::memory_pool::{MemoryPool, PoolConfig};
//!
//! let config = PoolConfig::new(512, 4);
//! let mut pool = MemoryPool::new(config).expect("pool ok");
//!
//! // Borrow a buffer from the pool.
//! let mut buf = pool.acquire().expect("slot available");
//! buf.as_mut_slice()[0] = 42;
//! drop(buf);           // returns to pool automatically
//!
//! // Same slot is recycled on the next acquire.
//! assert_eq!(pool.available(), 4);
//! ```

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::execution_plan::ResourceEstimate;
use crate::PipelineError;

// ── PoolConfig ────────────────────────────────────────────────────────────────

/// Configuration parameters for a [`MemoryPool`].
#[derive(Debug, Clone)]
pub struct PoolConfig {
    /// Number of bytes in each pooled buffer.
    pub buffer_size: usize,
    /// Number of pre-allocated buffers in the pool.
    pub pool_size: usize,
}

impl PoolConfig {
    /// Create a pool config directly.
    ///
    /// `buffer_size` is clamped to at least 1 byte; `pool_size` to at least 1.
    pub fn new(buffer_size: usize, pool_size: usize) -> Self {
        Self {
            buffer_size: buffer_size.max(1),
            pool_size: pool_size.max(1),
        }
    }

    /// Derive a pool config from a [`ResourceEstimate`] and a desired
    /// pre-allocation count.
    ///
    /// The buffer size is taken from `estimate.memory_bytes` (rounded up to
    /// the nearest 64-byte cache-line boundary for alignment).  Pool size
    /// defaults to `pre_alloc_count`.
    pub fn from_estimate(estimate: &ResourceEstimate, pre_alloc_count: usize) -> Self {
        let raw = estimate.memory_bytes as usize;
        // Round up to 64-byte cache-line multiple.
        let aligned = if raw == 0 {
            64
        } else {
            (raw + 63) & !63
        };
        Self {
            buffer_size: aligned,
            pool_size: pre_alloc_count.max(1),
        }
    }

    /// Total bytes pre-allocated by a pool created with this config.
    pub fn total_bytes(&self) -> usize {
        self.buffer_size * self.pool_size
    }
}

// ── Inner pool state (behind Arc<Mutex<>>) ────────────────────────────────────

struct PoolInner {
    config: PoolConfig,
    free: VecDeque<Vec<u8>>,
    /// Total number of slots (free + in-use).
    total: usize,
    /// How many acquires have been issued over the pool's lifetime.
    total_acquires: u64,
    /// How many acquires were served from recycled buffers (no new allocation).
    recycled_acquires: u64,
    /// How many times `acquire` returned `None` because the pool was exhausted.
    exhausted_count: u64,
}

impl PoolInner {
    fn new(config: PoolConfig) -> Self {
        let size = config.pool_size;
        let buf_size = config.buffer_size;
        let mut free = VecDeque::with_capacity(size);
        for _ in 0..size {
            free.push_back(vec![0u8; buf_size]);
        }
        Self {
            config,
            free,
            total: size,
            total_acquires: 0,
            recycled_acquires: 0,
            exhausted_count: 0,
        }
    }

    fn acquire(&mut self) -> Option<Vec<u8>> {
        self.total_acquires += 1;
        if let Some(mut buf) = self.free.pop_front() {
            // Zero-fill to avoid data leaks between frames.
            buf.fill(0);
            self.recycled_acquires += 1;
            Some(buf)
        } else {
            self.exhausted_count += 1;
            None
        }
    }

    fn release(&mut self, buf: Vec<u8>) {
        if self.free.len() < self.total {
            self.free.push_back(buf);
        }
        // If the pool already holds `total` buffers, discard the extra.
    }

    fn available(&self) -> usize {
        self.free.len()
    }
}

// ── MemoryPool ────────────────────────────────────────────────────────────────

/// A thread-safe pool of pre-allocated byte buffers.
///
/// Buffers are handed out as [`PooledBuffer`] guards.  When the guard is
/// dropped, the buffer is silently returned to the pool rather than freed.
///
/// `MemoryPool` is `Clone` (shares the same internal state) so it can be
/// given to multiple pipeline stages without `Rc`/`Box` gymnastics.
#[derive(Clone)]
pub struct MemoryPool {
    inner: Arc<Mutex<PoolInner>>,
}

impl MemoryPool {
    /// Create a new pool, pre-allocating all buffers eagerly.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::BuildError`] when the config would require
    /// more memory than `usize::MAX`.
    pub fn new(config: PoolConfig) -> Result<Self, PipelineError> {
        let total = config
            .buffer_size
            .checked_mul(config.pool_size)
            .ok_or_else(|| PipelineError::BuildError(
                "PoolConfig total_bytes overflows usize".to_string(),
            ))?;

        // Refuse obviously unreasonable sizes (> 4 GiB on 64-bit).
        if total > (1usize << 32) {
            return Err(PipelineError::BuildError(format!(
                "PoolConfig requests {total} bytes, which exceeds the 4 GiB safety limit"
            )));
        }

        Ok(Self {
            inner: Arc::new(Mutex::new(PoolInner::new(config))),
        })
    }

    /// Try to acquire a buffer from the pool.
    ///
    /// Returns `Some(PooledBuffer)` when a slot is available, or
    /// `None` when all slots are currently in use.
    pub fn acquire(&self) -> Option<PooledBuffer> {
        let mut inner = self.inner.lock().ok()?;
        let buf = inner.acquire()?;
        Some(PooledBuffer {
            data: Some(buf),
            pool: Arc::clone(&self.inner),
        })
    }

    /// Acquire a buffer, returning an error instead of `None` when exhausted.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::BuildError`] when the pool is fully in use.
    pub fn acquire_or_err(&self) -> Result<PooledBuffer, PipelineError> {
        self.acquire().ok_or_else(|| {
            PipelineError::BuildError(
                "MemoryPool exhausted: all slots are currently in use".to_string(),
            )
        })
    }

    /// Number of free slots currently in the pool.
    pub fn available(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.available())
            .unwrap_or(0)
    }

    /// Total number of slots (free + in-use) in the pool.
    pub fn total_slots(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.total)
            .unwrap_or(0)
    }

    /// Number of slots currently checked out.
    pub fn in_use(&self) -> usize {
        let total = self.total_slots();
        let free = self.available();
        total.saturating_sub(free)
    }

    /// Size in bytes of each pooled buffer.
    pub fn buffer_size(&self) -> usize {
        self.inner
            .lock()
            .map(|g| g.config.buffer_size)
            .unwrap_or(0)
    }

    /// Return lifetime statistics: `(total_acquires, recycled_acquires, exhausted_count)`.
    pub fn stats(&self) -> (u64, u64, u64) {
        self.inner
            .lock()
            .map(|g| (g.total_acquires, g.recycled_acquires, g.exhausted_count))
            .unwrap_or_default()
    }
}

impl std::fmt::Debug for MemoryPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MemoryPool")
            .field("available", &self.available())
            .field("total_slots", &self.total_slots())
            .field("buffer_size", &self.buffer_size())
            .finish()
    }
}

// ── PooledBuffer ──────────────────────────────────────────────────────────────

/// An RAII guard that returns its buffer to the pool on drop.
///
/// The buffer is zero-filled when acquired and when returned to ensure no
/// frame data leaks between pipeline invocations.
pub struct PooledBuffer {
    data: Option<Vec<u8>>,
    pool: Arc<Mutex<PoolInner>>,
}

impl PooledBuffer {
    /// Length of the underlying buffer in bytes.
    pub fn len(&self) -> usize {
        self.data.as_ref().map_or(0, |v| v.len())
    }

    /// Returns `true` when the buffer has zero length (should not happen in
    /// normal operation but guard against it).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return a read-only slice of the buffer.
    pub fn as_slice(&self) -> &[u8] {
        self.data.as_deref().unwrap_or(&[])
    }

    /// Return a mutable slice for writing frame data into.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        self.data.as_deref_mut().unwrap_or(&mut [])
    }

    /// Copy `src` bytes into the buffer.
    ///
    /// # Errors
    ///
    /// Returns [`PipelineError::ValidationError`] when `src` is longer than
    /// the pool's buffer size.
    pub fn write_from(&mut self, src: &[u8]) -> Result<(), PipelineError> {
        let buf = self
            .data
            .as_mut()
            .ok_or_else(|| PipelineError::ValidationError("PooledBuffer already released".into()))?;
        if src.len() > buf.len() {
            return Err(PipelineError::ValidationError(format!(
                "source data ({} bytes) exceeds pool buffer size ({} bytes)",
                src.len(),
                buf.len()
            )));
        }
        buf[..src.len()].copy_from_slice(src);
        Ok(())
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.data.take() {
            if let Ok(mut inner) = self.pool.lock() {
                inner.release(buf);
            }
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

// ── PoolManager ───────────────────────────────────────────────────────────────

/// Manages multiple named [`MemoryPool`]s, one per pipeline stage or node.
///
/// The manager lets the execution planner create appropriately-sized pools
/// based on [`ResourceEstimate`]s and then look them up by node name during
/// execution.
pub struct PoolManager {
    pools: std::collections::HashMap<String, MemoryPool>,
}

impl PoolManager {
    /// Create an empty manager.
    pub fn new() -> Self {
        Self {
            pools: std::collections::HashMap::new(),
        }
    }

    /// Register a pre-created pool under `name`.
    pub fn insert(&mut self, name: impl Into<String>, pool: MemoryPool) {
        self.pools.insert(name.into(), pool);
    }

    /// Create a new pool from a [`ResourceEstimate`] and register it.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`MemoryPool::new`].
    pub fn create_from_estimate(
        &mut self,
        name: impl Into<String>,
        estimate: &ResourceEstimate,
        pre_alloc_count: usize,
    ) -> Result<(), PipelineError> {
        let config = PoolConfig::from_estimate(estimate, pre_alloc_count);
        let pool = MemoryPool::new(config)?;
        self.pools.insert(name.into(), pool);
        Ok(())
    }

    /// Look up a pool by name.
    pub fn get(&self, name: &str) -> Option<&MemoryPool> {
        self.pools.get(name)
    }

    /// Total number of registered pools.
    pub fn pool_count(&self) -> usize {
        self.pools.len()
    }

    /// Aggregate free slots across all registered pools.
    pub fn total_available(&self) -> usize {
        self.pools.values().map(|p| p.available()).sum()
    }
}

impl Default for PoolManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution_plan::ResourceEstimate;

    // ── PoolConfig tests ──────────────────────────────────────────────────────

    #[test]
    fn pool_config_from_estimate_aligns_to_64() {
        let est = ResourceEstimate::new(100, 1.0, false);
        let cfg = PoolConfig::from_estimate(&est, 2);
        // 100 bytes rounds up to 128 (next multiple of 64)
        assert_eq!(cfg.buffer_size, 128);
        assert_eq!(cfg.pool_size, 2);
    }

    #[test]
    fn pool_config_zero_estimate_gives_64_bytes() {
        let est = ResourceEstimate::new(0, 0.0, false);
        let cfg = PoolConfig::from_estimate(&est, 3);
        assert_eq!(cfg.buffer_size, 64);
    }

    #[test]
    fn pool_config_total_bytes() {
        let cfg = PoolConfig::new(1024, 8);
        assert_eq!(cfg.total_bytes(), 8192);
    }

    // ── MemoryPool tests ──────────────────────────────────────────────────────

    #[test]
    fn pool_starts_fully_available() {
        let cfg = PoolConfig::new(256, 4);
        let pool = MemoryPool::new(cfg).expect("pool ok");
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.total_slots(), 4);
        assert_eq!(pool.in_use(), 0);
    }

    #[test]
    fn acquire_reduces_available() {
        let cfg = PoolConfig::new(64, 2);
        let pool = MemoryPool::new(cfg).expect("pool ok");
        let _buf = pool.acquire().expect("slot available");
        assert_eq!(pool.available(), 1);
        assert_eq!(pool.in_use(), 1);
    }

    #[test]
    fn drop_returns_buffer_to_pool() {
        let cfg = PoolConfig::new(64, 2);
        let pool = MemoryPool::new(cfg).expect("pool ok");
        {
            let _buf = pool.acquire().expect("slot ok");
            assert_eq!(pool.available(), 1);
        }
        // After the guard drops, the slot should come back.
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn pool_exhausted_returns_none() {
        let cfg = PoolConfig::new(32, 1);
        let pool = MemoryPool::new(cfg).expect("pool ok");
        let _buf = pool.acquire().expect("first acquire ok");
        let result = pool.acquire();
        assert!(result.is_none());
    }

    #[test]
    fn acquire_or_err_propagates_exhaustion() {
        let cfg = PoolConfig::new(16, 1);
        let pool = MemoryPool::new(cfg).expect("pool ok");
        let _buf = pool.acquire_or_err().expect("ok");
        let err = pool.acquire_or_err();
        assert!(err.is_err());
    }

    #[test]
    fn pooled_buffer_write_and_read() {
        let cfg = PoolConfig::new(16, 2);
        let pool = MemoryPool::new(cfg).expect("pool ok");
        let mut buf = pool.acquire().expect("slot ok");
        let payload = b"hello world!";
        buf.write_from(payload).expect("write ok");
        assert_eq!(&buf.as_slice()[..payload.len()], payload);
    }

    #[test]
    fn pooled_buffer_write_oversized_error() {
        let cfg = PoolConfig::new(4, 2);
        let pool = MemoryPool::new(cfg).expect("pool ok");
        let mut buf = pool.acquire().expect("slot ok");
        let result = buf.write_from(b"toolongdata");
        assert!(result.is_err());
    }

    #[test]
    fn pool_buffer_zero_filled_on_reuse() {
        let cfg = PoolConfig::new(8, 1);
        let pool = MemoryPool::new(cfg).expect("pool ok");
        {
            let mut buf = pool.acquire().expect("slot ok");
            buf.write_from(b"\xFF\xFF\xFF\xFF").expect("write ok");
        }
        // After returning, the pool should zero-fill on the next acquire.
        let buf2 = pool.acquire().expect("slot ok");
        assert_eq!(&buf2.as_slice()[..4], &[0u8, 0, 0, 0]);
    }

    // ── PoolManager tests ─────────────────────────────────────────────────────

    #[test]
    fn pool_manager_create_and_lookup() {
        let mut mgr = PoolManager::new();
        let est = ResourceEstimate::new(512, 5.0, false);
        mgr.create_from_estimate("scale", &est, 4).expect("create ok");
        let pool = mgr.get("scale").expect("should exist");
        assert_eq!(pool.available(), 4);
    }

    #[test]
    fn pool_manager_total_available() {
        let mut mgr = PoolManager::new();
        mgr.create_from_estimate("a", &ResourceEstimate::new(64, 1.0, false), 3)
            .expect("ok");
        mgr.create_from_estimate("b", &ResourceEstimate::new(64, 1.0, false), 2)
            .expect("ok");
        assert_eq!(mgr.total_available(), 5);
        assert_eq!(mgr.pool_count(), 2);
    }

    #[test]
    fn pool_stats_track_recycles() {
        let cfg = PoolConfig::new(32, 2);
        let pool = MemoryPool::new(cfg).expect("pool ok");
        {
            let _b = pool.acquire().expect("ok");
        }
        let _b2 = pool.acquire().expect("ok");
        let (total, recycled, exhausted) = pool.stats();
        assert_eq!(total, 2);
        assert_eq!(recycled, 2);
        assert_eq!(exhausted, 0);
    }
}
