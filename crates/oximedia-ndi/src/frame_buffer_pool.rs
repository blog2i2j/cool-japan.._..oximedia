//! Lock-free frame buffer pool for zero-copy NDI frame passing.
//!
//! Allocating a fresh `Vec<u8>` for every incoming video frame adds up fast at
//! production frame rates (60 fps × 4K = ~12 GB/s allocation pressure).  This
//! module solves that by pre-allocating a fixed number of [`FrameBuffer`]s and
//! returning them to the pool automatically when the [`PooledBuffer`] RAII
//! guard is dropped.
//!
//! ## Design
//!
//! * Each buffer is owned by either the pool's free-list or a live
//!   [`PooledBuffer`].  There is no shared mutable state — the pool communicates
//!   through an `Arc<Mutex<Vec<FrameBuffer>>>` free-list backed by
//!   `parking_lot::Mutex`, which is designed for tight short-critical-section
//!   workloads exactly like this one.
//! * [`PooledBuffer`] implements `Deref` / `DerefMut` to give transparent access
//!   to the inner [`FrameBuffer`] and drops the buffer back into the free-list
//!   on `Drop`.
//! * [`BufferPool::acquire`] returns `None` rather than blocking when all buffers
//!   are in use, giving callers explicit back-pressure control.

#![allow(dead_code)]

use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use parking_lot::Mutex;

// ── Pixel format ──────────────────────────────────────────────────────────────

/// Pixel (sample) formats supported by the NDI frame buffer pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// UYVY 4:2:2 packed — 2 bytes per pixel (luma is sampled at full rate,
    /// chroma at half rate horizontally).
    Uyvy422,
    /// BGRA 8-bit — 4 bytes per pixel.
    Bgra8,
    /// NV12 semi-planar YUV 4:2:0 — 1.5 bytes per pixel (Y plane + interleaved
    /// UV plane at half resolution).
    Nv12,
    /// YUV 4:2:0 planar — 1.5 bytes per pixel (Y + U + V planes).
    Yuv420p,
}

impl PixelFormat {
    /// Return the number of bytes required to store one complete video frame at
    /// the given dimensions.
    ///
    /// # Panics
    ///
    /// Never panics; overflow is guarded by saturating arithmetic.
    pub fn bytes_per_frame(self, width: u32, height: u32) -> usize {
        let pixels = (width as usize).saturating_mul(height as usize);
        match self {
            // 2 bytes per pixel
            Self::Uyvy422 => pixels.saturating_mul(2),
            // 4 bytes per pixel
            Self::Bgra8 => pixels.saturating_mul(4),
            // 1 byte Y + 0.5 byte UV = 1.5 bytes → multiply by 3 and divide by 2
            Self::Nv12 | Self::Yuv420p => pixels.saturating_mul(3) / 2,
        }
    }

    /// Human-readable name.
    pub fn name(self) -> &'static str {
        match self {
            Self::Uyvy422 => "UYVY422",
            Self::Bgra8 => "BGRA8",
            Self::Nv12 => "NV12",
            Self::Yuv420p => "YUV420p",
        }
    }
}

// ── Frame buffer ──────────────────────────────────────────────────────────────

/// A single pre-allocated video frame buffer.
#[derive(Debug, Clone)]
pub struct FrameBuffer {
    /// Raw pixel data.  Length must equal
    /// `format.bytes_per_frame(width, height)`.
    pub data: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format of the stored data.
    pub format: PixelFormat,
    /// Capture / render timestamp in microseconds.
    pub timestamp_us: u64,
}

impl FrameBuffer {
    /// Allocate a new frame buffer, zero-initialized, for the given dimensions.
    pub fn new(width: u32, height: u32, format: PixelFormat) -> Self {
        let size = format.bytes_per_frame(width, height);
        Self {
            data: vec![0u8; size],
            width,
            height,
            format,
            timestamp_us: 0,
        }
    }

    /// Reset the timestamp and zero-fill the data, preparing for reuse.
    pub fn reset(&mut self) {
        self.timestamp_us = 0;
        // Fill with zeros so stale data is never exposed to a new owner.
        self.data.iter_mut().for_each(|b| *b = 0);
    }
}

// ── Pool statistics ───────────────────────────────────────────────────────────

/// Snapshot of pool utilization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// Total number of buffers managed by this pool.
    pub total: usize,
    /// Number of buffers currently sitting in the free-list.
    pub available: usize,
    /// Number of buffers currently held by callers.
    pub in_use: usize,
    /// Historical peak of `in_use` since pool creation.
    pub peak_in_use: usize,
}

// ── Internal pool state ───────────────────────────────────────────────────────

/// Shared mutable state stored behind an `Arc` so [`PooledBuffer`] can return
/// its buffer on drop without holding a reference to the [`BufferPool`].
struct PoolInner {
    free: Mutex<Vec<FrameBuffer>>,
    total: usize,
    peak_in_use: AtomicUsize,
}

impl PoolInner {
    fn new(capacity: usize, width: u32, height: u32, format: PixelFormat) -> Self {
        let free: Vec<FrameBuffer> = (0..capacity)
            .map(|_| FrameBuffer::new(width, height, format))
            .collect();
        Self {
            free: Mutex::new(free),
            total: capacity,
            peak_in_use: AtomicUsize::new(0),
        }
    }

    fn acquire(&self) -> Option<FrameBuffer> {
        let mut free = self.free.lock();
        let buf = free.pop()?;
        // Update peak usage counter.
        let in_use = self.total - free.len();
        self.peak_in_use.fetch_max(in_use, Ordering::Relaxed);
        Some(buf)
    }

    fn release(&self, mut buf: FrameBuffer) {
        buf.reset();
        self.free.lock().push(buf);
    }

    fn stats(&self) -> PoolStats {
        let available = self.free.lock().len();
        let in_use = self.total.saturating_sub(available);
        PoolStats {
            total: self.total,
            available,
            in_use,
            peak_in_use: self.peak_in_use.load(Ordering::Relaxed),
        }
    }
}

// ── Public pool ───────────────────────────────────────────────────────────────

/// A fixed-capacity pool of pre-allocated [`FrameBuffer`]s.
///
/// Buffers are returned to the pool automatically when the associated
/// [`PooledBuffer`] is dropped.
#[derive(Clone)]
pub struct BufferPool {
    inner: Arc<PoolInner>,
}

impl BufferPool {
    /// Create a new pool with `capacity` pre-allocated buffers, each sized for
    /// `frame_width × frame_height` pixels in the specified `format`.
    pub fn new(
        capacity: usize,
        frame_width: u32,
        frame_height: u32,
        format: PixelFormat,
    ) -> Self {
        Self {
            inner: Arc::new(PoolInner::new(capacity, frame_width, frame_height, format)),
        }
    }

    /// Attempt to acquire a free buffer from the pool.
    ///
    /// Returns `Some(PooledBuffer)` when a buffer is available, `None` when
    /// the pool is exhausted.  The caller must not block waiting for a buffer
    /// — instead it should apply back-pressure (e.g. drop the oldest frame or
    /// wait for the next render cycle).
    pub fn acquire(&self) -> Option<PooledBuffer> {
        let buf = self.inner.acquire()?;
        Some(PooledBuffer {
            buf: Some(buf),
            pool: Arc::clone(&self.inner),
        })
    }

    /// Return a snapshot of current pool utilization.
    pub fn stats(&self) -> PoolStats {
        self.inner.stats()
    }

    /// Total number of buffers managed by this pool.
    pub fn capacity(&self) -> usize {
        self.inner.total
    }
}

impl std::fmt::Debug for BufferPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = self.stats();
        f.debug_struct("BufferPool")
            .field("total", &s.total)
            .field("available", &s.available)
            .field("in_use", &s.in_use)
            .finish()
    }
}

// ── RAII wrapper ──────────────────────────────────────────────────────────────

/// RAII wrapper around a [`FrameBuffer`] borrowed from a [`BufferPool`].
///
/// When this value is dropped, the inner buffer is automatically returned to
/// the pool it was taken from.
pub struct PooledBuffer {
    /// `Option` so we can `take()` the buffer in `Drop` without an unsafe move.
    buf: Option<FrameBuffer>,
    pool: Arc<PoolInner>,
}

/// Error returned when a [`PooledBuffer`] is accessed after its inner buffer
/// has been consumed (which can only happen if `Drop` has run — a logic error).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferConsumedError;

impl std::fmt::Display for BufferConsumedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PooledBuffer: inner buffer consumed before drop")
    }
}

impl std::error::Error for BufferConsumedError {}

impl PooledBuffer {
    /// Return a reference to the wrapped [`FrameBuffer`].
    ///
    /// # Errors
    ///
    /// Returns [`BufferConsumedError`] if the inner buffer has already been
    /// consumed (should never happen while the value is alive).
    pub fn buffer(&self) -> std::result::Result<&FrameBuffer, BufferConsumedError> {
        self.buf.as_ref().ok_or(BufferConsumedError)
    }

    /// Return a mutable reference to the wrapped [`FrameBuffer`].
    ///
    /// # Errors
    ///
    /// Returns [`BufferConsumedError`] if the inner buffer has already been
    /// consumed (should never happen while the value is alive).
    pub fn buffer_mut(&mut self) -> std::result::Result<&mut FrameBuffer, BufferConsumedError> {
        self.buf.as_mut().ok_or(BufferConsumedError)
    }
}

impl Deref for PooledBuffer {
    type Target = FrameBuffer;

    fn deref(&self) -> &FrameBuffer {
        // `buf` is `Some` from construction until `Drop::drop` runs.
        // `Deref` is only callable while the value is alive, so we use
        // a match with an unreachable fallback for safety.
        match self.buf.as_ref() {
            Some(b) => b,
            // This path is truly unreachable: `buf` is only `None` after
            // `Drop::drop`, at which point `self` is no longer accessible.
            None => unreachable!("PooledBuffer inner buffer consumed before drop"),
        }
    }
}

impl DerefMut for PooledBuffer {
    fn deref_mut(&mut self) -> &mut FrameBuffer {
        // See `Deref` impl for reasoning on why `buf` is always `Some` here.
        match self.buf.as_mut() {
            Some(b) => b,
            None => unreachable!("PooledBuffer inner buffer consumed before drop"),
        }
    }
}

impl Drop for PooledBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buf.take() {
            self.pool.release(buf);
        }
    }
}

impl std::fmt::Debug for PooledBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PooledBuffer")
            .field("width", &self.buf.as_ref().map(|b| b.width))
            .field("height", &self.buf.as_ref().map(|b| b.height))
            .field("format", &self.buf.as_ref().map(|b| b.format))
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Helper to acquire or fail the test gracefully.
    fn acquire_or_fail(pool: &BufferPool) -> PooledBuffer {
        pool.acquire()
            .unwrap_or_else(|| panic!("pool unexpectedly exhausted"))
    }

    // ── PixelFormat::bytes_per_frame ──────────────────────────────────────────

    #[test]
    fn uyvy422_bytes_per_frame() {
        assert_eq!(PixelFormat::Uyvy422.bytes_per_frame(1920, 1080), 1920 * 1080 * 2);
    }

    #[test]
    fn bgra8_bytes_per_frame() {
        assert_eq!(PixelFormat::Bgra8.bytes_per_frame(1280, 720), 1280 * 720 * 4);
    }

    #[test]
    fn nv12_bytes_per_frame() {
        let w = 1920_u32;
        let h = 1080_u32;
        let expected = (w as usize) * (h as usize) * 3 / 2;
        assert_eq!(PixelFormat::Nv12.bytes_per_frame(w, h), expected);
    }

    #[test]
    fn yuv420p_bytes_per_frame_matches_nv12() {
        let (w, h) = (640_u32, 480_u32);
        assert_eq!(
            PixelFormat::Yuv420p.bytes_per_frame(w, h),
            PixelFormat::Nv12.bytes_per_frame(w, h)
        );
    }

    #[test]
    fn zero_dimension_yields_zero_bytes() {
        assert_eq!(PixelFormat::Bgra8.bytes_per_frame(0, 1080), 0);
        assert_eq!(PixelFormat::Bgra8.bytes_per_frame(1920, 0), 0);
    }

    // ── FrameBuffer allocation ────────────────────────────────────────────────

    #[test]
    fn frame_buffer_allocates_correct_size() {
        let buf = FrameBuffer::new(1920, 1080, PixelFormat::Bgra8);
        assert_eq!(buf.data.len(), PixelFormat::Bgra8.bytes_per_frame(1920, 1080));
        assert_eq!(buf.width, 1920);
        assert_eq!(buf.height, 1080);
        assert_eq!(buf.format, PixelFormat::Bgra8);
        assert_eq!(buf.timestamp_us, 0);
    }

    #[test]
    fn frame_buffer_reset_clears_data() {
        let mut buf = FrameBuffer::new(64, 64, PixelFormat::Bgra8);
        buf.data[0] = 0xFF;
        buf.timestamp_us = 999;
        buf.reset();
        assert_eq!(buf.data[0], 0);
        assert_eq!(buf.timestamp_us, 0);
    }

    // ── Acquire / release cycle ───────────────────────────────────────────────

    #[test]
    fn acquire_and_drop_cycles_buffer_back() {
        let pool = BufferPool::new(2, 320, 240, PixelFormat::Uyvy422);
        assert_eq!(pool.stats().available, 2);

        let b1 = acquire_or_fail(&pool);
        assert_eq!(pool.stats().available, 1);
        assert_eq!(pool.stats().in_use, 1);

        drop(b1);
        assert_eq!(pool.stats().available, 2);
        assert_eq!(pool.stats().in_use, 0);
    }

    // ── Pool exhaustion returns None ──────────────────────────────────────────

    #[test]
    fn pool_exhaustion_returns_none() {
        let pool = BufferPool::new(2, 160, 120, PixelFormat::Nv12);
        let _b1 = acquire_or_fail(&pool);
        let _b2 = acquire_or_fail(&pool);
        let b3 = pool.acquire();
        assert!(b3.is_none(), "pool is exhausted; should return None");
    }

    // ── Stats accuracy ────────────────────────────────────────────────────────

    #[test]
    fn stats_reflect_correct_counts() {
        let pool = BufferPool::new(4, 640, 480, PixelFormat::Bgra8);
        let b1 = acquire_or_fail(&pool);
        let b2 = acquire_or_fail(&pool);
        {
            let s = pool.stats();
            assert_eq!(s.total, 4);
            assert_eq!(s.available, 2);
            assert_eq!(s.in_use, 2);
        }
        drop(b1);
        drop(b2);
        let s = pool.stats();
        assert_eq!(s.available, 4);
        assert_eq!(s.in_use, 0);
    }

    // ── Peak in_use tracking ──────────────────────────────────────────────────

    #[test]
    fn peak_in_use_tracked_correctly() {
        let pool = BufferPool::new(5, 128, 96, PixelFormat::Yuv420p);
        {
            let _b1 = acquire_or_fail(&pool);
            let _b2 = acquire_or_fail(&pool);
            let _b3 = acquire_or_fail(&pool);
        }
        let s = pool.stats();
        assert_eq!(s.in_use, 0);
        assert_eq!(s.peak_in_use, 3);
    }

    // ── Deref transparent access ──────────────────────────────────────────────

    #[test]
    fn pooled_buffer_deref_gives_access_to_frame() {
        let pool = BufferPool::new(1, 64, 64, PixelFormat::Bgra8);
        let mut pb = acquire_or_fail(&pool);
        pb.timestamp_us = 12345;
        assert_eq!(pb.timestamp_us, 12345);
        assert_eq!(pb.width, 64);
    }

    // ── Data zeroed after release ──────────────────────────────────────────────

    #[test]
    fn buffer_data_zeroed_on_return_to_pool() {
        let pool = BufferPool::new(1, 16, 16, PixelFormat::Bgra8);
        {
            let mut pb = acquire_or_fail(&pool);
            pb.data.iter_mut().for_each(|b| *b = 0xAB);
        }
        let pb2 = acquire_or_fail(&pool);
        assert!(pb2.data.iter().all(|&b| b == 0), "buffer data should be zeroed after return");
    }

    // ── Capacity helper ───────────────────────────────────────────────────────

    #[test]
    fn capacity_returns_total() {
        let pool = BufferPool::new(7, 1920, 1080, PixelFormat::Uyvy422);
        assert_eq!(pool.capacity(), 7);
    }

    // ========================================================================
    // New tests (8+)
    // ========================================================================

    #[test]
    fn pixel_format_name_labels() {
        assert_eq!(PixelFormat::Uyvy422.name(), "UYVY422");
        assert_eq!(PixelFormat::Bgra8.name(), "BGRA8");
        assert_eq!(PixelFormat::Nv12.name(), "NV12");
        assert_eq!(PixelFormat::Yuv420p.name(), "YUV420p");
    }

    #[test]
    fn buffer_method_returns_ok_on_live_pooled_buffer() {
        let pool = BufferPool::new(1, 8, 8, PixelFormat::Bgra8);
        let pb = acquire_or_fail(&pool);
        assert!(pb.buffer().is_ok());
        let buf = pb.buffer().unwrap_or_else(|_| panic!("buffer should be Ok"));
        assert_eq!(buf.width, 8);
        assert_eq!(buf.height, 8);
    }

    #[test]
    fn buffer_mut_method_allows_modification() {
        let pool = BufferPool::new(1, 4, 4, PixelFormat::Bgra8);
        let mut pb = acquire_or_fail(&pool);
        if let Ok(buf) = pb.buffer_mut() {
            buf.timestamp_us = 42;
        }
        assert_eq!(pb.timestamp_us, 42);
    }

    #[test]
    fn pool_clone_shares_state() {
        let pool1 = BufferPool::new(3, 32, 32, PixelFormat::Nv12);
        let pool2 = pool1.clone();
        let _b = acquire_or_fail(&pool1);
        // pool2 sees the same state
        assert_eq!(pool2.stats().in_use, 1);
        assert_eq!(pool2.stats().available, 2);
    }

    #[test]
    fn rapid_acquire_release_stress() {
        let pool = BufferPool::new(4, 64, 64, PixelFormat::Uyvy422);
        for _ in 0..100 {
            let mut bufs = Vec::new();
            for _ in 0..4 {
                if let Some(b) = pool.acquire() {
                    bufs.push(b);
                }
            }
            assert!(pool.acquire().is_none());
            drop(bufs);
            assert_eq!(pool.stats().available, 4);
        }
        assert_eq!(pool.stats().peak_in_use, 4);
    }

    #[test]
    fn frame_buffer_new_all_zeros() {
        let buf = FrameBuffer::new(16, 16, PixelFormat::Yuv420p);
        assert!(buf.data.iter().all(|&b| b == 0));
    }

    #[test]
    fn pool_debug_format_contains_total() {
        let pool = BufferPool::new(3, 10, 10, PixelFormat::Bgra8);
        let dbg = format!("{:?}", pool);
        assert!(dbg.contains("total"));
        assert!(dbg.contains("3"));
    }

    #[test]
    fn pooled_buffer_debug_format_contains_dimensions() {
        let pool = BufferPool::new(1, 640, 480, PixelFormat::Bgra8);
        let pb = acquire_or_fail(&pool);
        let dbg = format!("{:?}", pb);
        assert!(dbg.contains("640"));
        assert!(dbg.contains("480"));
    }

    #[test]
    fn frame_buffer_clone_is_independent() {
        let mut original = FrameBuffer::new(4, 4, PixelFormat::Bgra8);
        original.timestamp_us = 100;
        original.data[0] = 0xAA;
        let cloned = original.clone();
        original.data[0] = 0xBB;
        assert_eq!(cloned.data[0], 0xAA);
        assert_eq!(cloned.timestamp_us, 100);
    }

    #[test]
    fn pool_stats_equality() {
        let s1 = PoolStats { total: 4, available: 2, in_use: 2, peak_in_use: 3 };
        let s2 = PoolStats { total: 4, available: 2, in_use: 2, peak_in_use: 3 };
        let s3 = PoolStats { total: 4, available: 3, in_use: 1, peak_in_use: 3 };
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn buffer_consumed_error_display() {
        let err = BufferConsumedError;
        let msg = format!("{}", err);
        assert!(msg.contains("consumed"));
    }
}
