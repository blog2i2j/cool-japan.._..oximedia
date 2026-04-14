//! Cache-line-aligned buffer allocation for SIMD-friendly media processing.
//!
//! SIMD instructions (SSE, AVX, NEON) require or strongly prefer data aligned
//! to 16, 32, or 64 bytes.  Misaligned loads/stores cause either a fault or a
//! significant performance penalty on many micro-architectures.  This module
//! provides:
//!
//! - **[`CacheAlignedBuffer`]** — a heap-allocated byte buffer whose start
//!   address is aligned to [`CACHE_LINE_BYTES`] (64 bytes).
//! - **[`CacheAlignedPool`]** — a fixed-size pool of `CacheAlignedBuffer`s
//!   with acquire/release semantics, suitable for reuse across frames.
//!
//! # Alignment strategy
//!
//! The Rust allocator does not guarantee any alignment stronger than the
//! type's natural alignment.  We achieve cache-line alignment by allocating
//! `size + CACHE_LINE_BYTES - 1` bytes via a plain `Vec<u8>` and then
//! choosing the first offset within that `Vec` whose address is a multiple of
//! `CACHE_LINE_BYTES`.  The buffer tracks this offset so it can be released
//! without leaking memory.
//!
//! This approach is **100 % safe Rust** — no `unsafe` blocks are used.
//!
//! # Example
//!
//! ```
//! use oximedia_core::alloc::cache_aligned::{CacheAlignedBuffer, CACHE_LINE_BYTES};
//!
//! let buf = CacheAlignedBuffer::new(256);
//! assert_eq!(buf.len(), 256);
//! // The data pointer is a multiple of CACHE_LINE_BYTES.
//! let ptr_val = buf.as_slice().as_ptr() as usize;
//! assert_eq!(ptr_val % CACHE_LINE_BYTES, 0);
//! ```

#![allow(dead_code)]

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/// The cache-line size assumed for alignment.
///
/// 64 bytes is the cache line on all modern x86-64, ARM64, and RISC-V cores
/// and also satisfies AVX-512 (64-byte) alignment requirements.
pub const CACHE_LINE_BYTES: usize = 64;

// ─────────────────────────────────────────────────────────────────────────────
// CacheAlignedBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// A heap-allocated byte buffer whose usable region starts at a cache-line
/// aligned address.
///
/// The allocation is slightly larger than `size` bytes to guarantee alignment.
/// The `len` and `offset` fields track exactly which slice of the backing
/// `Vec` is the usable, aligned region.
///
/// # Examples
///
/// ```
/// use oximedia_core::alloc::cache_aligned::{CacheAlignedBuffer, CACHE_LINE_BYTES};
///
/// let mut buf = CacheAlignedBuffer::new(1024);
/// assert_eq!(buf.len(), 1024);
///
/// buf.as_mut_slice().fill(0xAB);
/// assert!(buf.as_slice().iter().all(|&b| b == 0xAB));
///
/// let ptr = buf.as_slice().as_ptr() as usize;
/// assert_eq!(ptr % CACHE_LINE_BYTES, 0, "not cache-line aligned");
/// ```
pub struct CacheAlignedBuffer {
    /// Backing storage (possibly over-allocated for alignment padding).
    storage: Vec<u8>,
    /// Number of bytes to skip from the start of `storage` to reach the
    /// cache-line-aligned start.
    offset: usize,
    /// Number of usable bytes starting at `storage[offset]`.
    len: usize,
}

impl CacheAlignedBuffer {
    /// Allocates a new cache-line-aligned buffer of `len` bytes.
    ///
    /// The contents are initialised to zero.  The buffer's start address is
    /// guaranteed to be a multiple of [`CACHE_LINE_BYTES`].
    ///
    /// # Panics
    ///
    /// Panics if `len` is zero.
    #[must_use]
    pub fn new(len: usize) -> Self {
        assert!(len > 0, "CacheAlignedBuffer: len must be > 0");
        // Over-allocate so we can always find an aligned start within the vec.
        let alloc_size = len + CACHE_LINE_BYTES - 1;
        let mut storage = vec![0u8; alloc_size];

        // Find the first index whose address is cache-line-aligned.
        let base_ptr = storage.as_ptr() as usize;
        let offset = if base_ptr % CACHE_LINE_BYTES == 0 {
            0
        } else {
            CACHE_LINE_BYTES - (base_ptr % CACHE_LINE_BYTES)
        };

        // Touch the aligned slice to ensure it is inside the allocation.
        // (The index arithmetic is safe because offset < CACHE_LINE_BYTES and
        // alloc_size = len + CACHE_LINE_BYTES - 1 >= len + offset.)
        let _ = &mut storage[offset..offset + len];

        Self {
            storage,
            offset,
            len,
        }
    }

    /// Returns the number of usable bytes in this buffer.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the usable region is empty (zero bytes).
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns an immutable slice of the usable (aligned) region.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        &self.storage[self.offset..self.offset + self.len]
    }

    /// Returns a mutable slice of the usable (aligned) region.
    #[inline]
    #[must_use]
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        &mut self.storage[self.offset..self.offset + self.len]
    }

    /// Returns the raw pointer to the start of the aligned region.
    ///
    /// This is useful when passing buffers to SIMD intrinsics.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.as_slice().as_ptr()
    }

    /// Returns the raw mutable pointer to the start of the aligned region.
    #[inline]
    #[must_use]
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_mut_slice().as_mut_ptr()
    }

    /// Fills the entire buffer with `value`.
    #[inline]
    pub fn fill(&mut self, value: u8) {
        self.as_mut_slice().fill(value);
    }

    /// Copies data from `src` into the start of this buffer.
    ///
    /// Returns `Err(())` if `src.len() > self.len()`.
    pub fn copy_from_slice(&mut self, src: &[u8]) -> Result<(), ()> {
        if src.len() > self.len {
            return Err(());
        }
        self.as_mut_slice()[..src.len()].copy_from_slice(src);
        Ok(())
    }

    /// Returns the alignment offset used (bytes skipped from the raw
    /// allocation to reach the aligned start).  Exposed for diagnostics.
    #[must_use]
    pub fn alignment_offset(&self) -> usize {
        self.offset
    }

    /// Returns the total capacity of the backing storage in bytes
    /// (usable bytes + alignment padding).
    #[must_use]
    pub fn storage_capacity(&self) -> usize {
        self.storage.len()
    }
}

impl std::fmt::Debug for CacheAlignedBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheAlignedBuffer")
            .field("len", &self.len)
            .field("offset", &self.offset)
            .field(
                "aligned_ptr",
                &format_args!("{:#x}", self.as_ptr() as usize),
            )
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CacheAlignedPool
// ─────────────────────────────────────────────────────────────────────────────

/// A pool of reusable [`CacheAlignedBuffer`]s.
///
/// Intended for media pipelines that process many fixed-size frames where
/// allocation overhead is undesirable.  Buffers are acquired individually and
/// returned when done; the pool reuses them rather than dropping.
///
/// # Thread safety
///
/// `CacheAlignedPool` is **not** `Send` or `Sync` — it is designed for single-
/// threaded (or per-thread) use within a pipeline stage.  For shared access,
/// wrap in a `Mutex`.
///
/// # Example
///
/// ```
/// use oximedia_core::alloc::cache_aligned::CacheAlignedPool;
///
/// let mut pool = CacheAlignedPool::new(4, 4096);
/// let buf = pool.acquire().expect("pool has buffers");
/// assert_eq!(buf.len(), 4096);
/// pool.release(buf);
/// assert_eq!(pool.available(), 4);
/// ```
pub struct CacheAlignedPool {
    buffers: Vec<CacheAlignedBuffer>,
    buffer_size: usize,
    /// Peak number of simultaneously checked-out buffers.
    peak_outstanding: usize,
    /// Current number of checked-out buffers.
    outstanding: usize,
}

impl CacheAlignedPool {
    /// Creates a pool pre-populated with `count` buffers of `buffer_size`
    /// bytes each.
    ///
    /// # Panics
    ///
    /// Panics if `buffer_size` is zero.
    #[must_use]
    pub fn new(count: usize, buffer_size: usize) -> Self {
        assert!(buffer_size > 0, "CacheAlignedPool: buffer_size must be > 0");
        let buffers = (0..count)
            .map(|_| CacheAlignedBuffer::new(buffer_size))
            .collect();
        Self {
            buffers,
            buffer_size,
            peak_outstanding: 0,
            outstanding: 0,
        }
    }

    /// Acquires a buffer from the pool.
    ///
    /// Returns `None` if all buffers are currently in use.
    pub fn acquire(&mut self) -> Option<CacheAlignedBuffer> {
        let buf = self.buffers.pop()?;
        self.outstanding += 1;
        if self.outstanding > self.peak_outstanding {
            self.peak_outstanding = self.outstanding;
        }
        Some(buf)
    }

    /// Returns a buffer to the pool.
    ///
    /// The contents of the buffer are **not** zeroed on release; callers that
    /// need cleared buffers should call [`CacheAlignedBuffer::fill`] before
    /// use.
    pub fn release(&mut self, buf: CacheAlignedBuffer) {
        if self.outstanding > 0 {
            self.outstanding -= 1;
        }
        self.buffers.push(buf);
    }

    /// Returns the number of buffers currently available (not checked out).
    #[must_use]
    pub fn available(&self) -> usize {
        self.buffers.len()
    }

    /// Returns the number of buffers currently checked out.
    #[must_use]
    pub fn outstanding(&self) -> usize {
        self.outstanding
    }

    /// Returns the peak number of simultaneously outstanding buffers since
    /// the pool was created.
    #[must_use]
    pub fn peak_outstanding(&self) -> usize {
        self.peak_outstanding
    }

    /// Returns the size of each buffer in bytes.
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Extends the pool by allocating `additional` new buffers.
    pub fn grow(&mut self, additional: usize) {
        for _ in 0..additional {
            self.buffers.push(CacheAlignedBuffer::new(self.buffer_size));
        }
    }

    /// Returns `true` if no buffers are available.
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.buffers.is_empty()
    }
}

impl std::fmt::Debug for CacheAlignedPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CacheAlignedPool")
            .field("available", &self.available())
            .field("outstanding", &self.outstanding)
            .field("buffer_size", &self.buffer_size)
            .finish()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Alignment guarantee
    #[test]
    fn buffer_is_cache_line_aligned() {
        for size in [1_usize, 15, 64, 65, 128, 1024, 4096] {
            let buf = CacheAlignedBuffer::new(size);
            let ptr = buf.as_ptr() as usize;
            assert_eq!(
                ptr % CACHE_LINE_BYTES,
                0,
                "not aligned for size={size}: ptr={ptr:#x}"
            );
        }
    }

    // 2. Correct length reported
    #[test]
    fn buffer_len_is_exact() {
        let buf = CacheAlignedBuffer::new(256);
        assert_eq!(buf.len(), 256);
    }

    // 3. Initialised to zero
    #[test]
    fn buffer_initialised_to_zero() {
        let buf = CacheAlignedBuffer::new(128);
        assert!(buf.as_slice().iter().all(|&b| b == 0));
    }

    // 4. fill and read back
    #[test]
    fn buffer_fill_and_read() {
        let mut buf = CacheAlignedBuffer::new(64);
        buf.fill(0xDE);
        assert!(buf.as_slice().iter().all(|&b| b == 0xDE));
    }

    // 5. copy_from_slice success
    #[test]
    fn buffer_copy_from_slice_ok() {
        let mut buf = CacheAlignedBuffer::new(8);
        let data = [1u8, 2, 3, 4, 5, 6, 7, 8];
        buf.copy_from_slice(&data).expect("copy should succeed");
        assert_eq!(buf.as_slice(), &data);
    }

    // 6. copy_from_slice overflow returns Err
    #[test]
    fn buffer_copy_from_slice_overflow() {
        let mut buf = CacheAlignedBuffer::new(4);
        let data = [0u8; 8];
        assert!(buf.copy_from_slice(&data).is_err());
    }

    // 7. storage_capacity >= len
    #[test]
    fn storage_capacity_gte_len() {
        let buf = CacheAlignedBuffer::new(100);
        assert!(buf.storage_capacity() >= buf.len());
    }

    // 8. Pool acquire / release cycle
    #[test]
    fn pool_acquire_release() {
        let mut pool = CacheAlignedPool::new(4, 512);
        assert_eq!(pool.available(), 4);
        let b1 = pool.acquire().expect("available");
        let b2 = pool.acquire().expect("available");
        assert_eq!(pool.available(), 2);
        assert_eq!(pool.outstanding(), 2);
        pool.release(b1);
        pool.release(b2);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.outstanding(), 0);
    }

    // 9. Pool exhaustion returns None
    #[test]
    fn pool_exhaustion() {
        let mut pool = CacheAlignedPool::new(2, 64);
        let _b1 = pool.acquire().expect("ok");
        let _b2 = pool.acquire().expect("ok");
        assert!(pool.is_exhausted());
        assert!(pool.acquire().is_none());
    }

    // 10. Pool grow increases available count
    #[test]
    fn pool_grow() {
        let mut pool = CacheAlignedPool::new(2, 64);
        pool.grow(3);
        assert_eq!(pool.available(), 5);
    }

    // 11. Peak outstanding is tracked correctly
    #[test]
    fn pool_peak_outstanding() {
        let mut pool = CacheAlignedPool::new(4, 64);
        let b1 = pool.acquire().expect("ok");
        let b2 = pool.acquire().expect("ok");
        let b3 = pool.acquire().expect("ok");
        assert_eq!(pool.peak_outstanding(), 3);
        pool.release(b3);
        pool.release(b2);
        pool.release(b1);
        // Peak should still be 3.
        assert_eq!(pool.peak_outstanding(), 3);
    }

    // 12. buffer_size reported correctly
    #[test]
    fn pool_buffer_size() {
        let pool = CacheAlignedPool::new(2, 2048);
        assert_eq!(pool.buffer_size(), 2048);
    }

    // 13. Acquired buffers are individually aligned
    #[test]
    fn pool_acquired_buffers_aligned() {
        let mut pool = CacheAlignedPool::new(4, 1024);
        let mut bufs = Vec::new();
        while let Some(b) = pool.acquire() {
            bufs.push(b);
        }
        for buf in &bufs {
            let ptr = buf.as_ptr() as usize;
            assert_eq!(ptr % CACHE_LINE_BYTES, 0);
        }
    }

    // 14. as_mut_ptr returns aligned address
    #[test]
    fn buffer_mut_ptr_aligned() {
        let mut buf = CacheAlignedBuffer::new(64);
        let ptr = buf.as_mut_ptr() as usize;
        assert_eq!(ptr % CACHE_LINE_BYTES, 0);
    }

    // 15. is_empty always false for non-zero-length buffer
    #[test]
    fn buffer_not_empty() {
        let buf = CacheAlignedBuffer::new(1);
        assert!(!buf.is_empty());
    }

    // 16. Debug impl does not panic
    #[test]
    fn debug_impl() {
        let buf = CacheAlignedBuffer::new(32);
        let s = format!("{buf:?}");
        assert!(s.contains("CacheAlignedBuffer"));
        let pool = CacheAlignedPool::new(2, 32);
        let ps = format!("{pool:?}");
        assert!(ps.contains("CacheAlignedPool"));
    }
}
