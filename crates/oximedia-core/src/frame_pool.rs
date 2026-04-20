//! Frame pool with configurable pre-allocation for low-latency pipelines.
//!
//! Provides [`FramePool`], a thread-safe pool of reusable video/audio frame
//! buffers.  Pre-allocating frames at startup eliminates per-frame heap
//! allocation in the hot path, which is critical for real-time media pipelines.
//!
//! # Design
//!
//! - Frames are wrapped in [`Arc<Mutex<PooledFrame>>`].
//! - When a caller calls [`FramePool::acquire`] it receives an
//!   [`AcquiredFrame`] RAII guard.  When the guard is dropped the frame is
//!   automatically returned to the pool.
//! - If the pool is exhausted and `allow_overflow` is true, a new frame is
//!   allocated on demand but **not** returned to the pool when dropped (it is
//!   just freed).
//! - Pool statistics (acquire count, release count, overflow count) are tracked
//!   with lock-free atomics.
//!
//! # Example
//!
//! ```
//! use oximedia_core::frame_pool::{FramePool, FramePoolConfig, FrameKind};
//!
//! let config = FramePoolConfig {
//!     pre_alloc: 4,
//!     max_pool_size: 8,
//!     frame_kind: FrameKind::Video { width: 1920, height: 1080, bytes_per_pixel: 3 },
//!     allow_overflow: true,
//! };
//! let pool = FramePool::new(config);
//! {
//!     let frame = pool.acquire().expect("pool has frames");
//!     // use frame.data() / frame.data_mut() ...
//! } // frame returned to pool here
//! assert_eq!(pool.stats().release_count, 1);
//! ```

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

// ─────────────────────────────────────────────────────────────────────────────
// FrameKind
// ─────────────────────────────────────────────────────────────────────────────

/// Describes the layout and size of frames managed by a [`FramePool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// Packed video frame: `width * height * bytes_per_pixel` bytes.
    Video {
        /// Frame width in pixels.
        width: u32,
        /// Frame height in pixels.
        height: u32,
        /// Bytes per pixel (e.g. 3 for RGB24, 4 for RGBA32, 2 for YUV422).
        bytes_per_pixel: u32,
    },
    /// Interleaved audio frame: `channel_count * samples_per_frame * bytes_per_sample` bytes.
    Audio {
        /// Number of channels.
        channel_count: u32,
        /// Number of PCM samples per channel per frame.
        samples_per_frame: u32,
        /// Bytes per sample (e.g. 4 for f32 PCM, 2 for i16 PCM).
        bytes_per_sample: u32,
    },
    /// Caller-specified raw byte buffer of `size` bytes.
    Raw {
        /// Buffer size in bytes.
        size: usize,
    },
}

impl FrameKind {
    /// Computes the total byte size of a single frame buffer.
    #[must_use]
    pub fn byte_size(self) -> usize {
        match self {
            Self::Video {
                width,
                height,
                bytes_per_pixel,
            } => (width as usize)
                .saturating_mul(height as usize)
                .saturating_mul(bytes_per_pixel as usize),
            Self::Audio {
                channel_count,
                samples_per_frame,
                bytes_per_sample,
            } => (channel_count as usize)
                .saturating_mul(samples_per_frame as usize)
                .saturating_mul(bytes_per_sample as usize),
            Self::Raw { size } => size,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FramePoolConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for a [`FramePool`].
#[derive(Debug, Clone, Copy)]
pub struct FramePoolConfig {
    /// Number of frames to allocate immediately when the pool is created.
    pub pre_alloc: usize,
    /// Maximum number of frames to keep in the pool simultaneously.
    /// Frames returned while the pool is at this limit are freed immediately.
    pub max_pool_size: usize,
    /// Layout/size descriptor for each frame.
    pub frame_kind: FrameKind,
    /// When `true`, [`FramePool::acquire`] allocates an overflow frame instead
    /// of returning `None` when the pool is empty.
    pub allow_overflow: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// PooledFrame
// ─────────────────────────────────────────────────────────────────────────────

/// A reusable frame buffer.
///
/// Normally obtained via [`FramePool::acquire`] and accessed through
/// [`AcquiredFrame`].  Direct construction is also possible for off-pool use.
pub struct PooledFrame {
    /// Raw byte buffer.
    data: Vec<u8>,
    /// Logical frame sequence number assigned at acquire time.
    seq: u64,
}

impl PooledFrame {
    fn new(size: usize) -> Self {
        Self {
            data: vec![0u8; size],
            seq: 0,
        }
    }

    /// Returns a shared reference to the raw frame data.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    /// Returns a mutable reference to the raw frame data.
    pub fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }

    /// Returns the byte length of the frame buffer.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns `true` when the frame buffer is zero-length.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns the logical sequence number assigned when the frame was acquired.
    #[must_use]
    pub fn seq(&self) -> u64 {
        self.seq
    }

    /// Zeroes out the frame buffer, useful before reuse.
    pub fn clear(&mut self) {
        self.data.fill(0);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// AcquiredFrame (RAII guard)
// ─────────────────────────────────────────────────────────────────────────────

/// RAII guard returned by [`FramePool::acquire`].
///
/// The underlying [`PooledFrame`] is locked for exclusive access while this
/// guard is alive.  Dropping the guard returns the frame to the pool (or frees
/// it if the pool is at capacity or if this was an overflow frame).
pub struct AcquiredFrame {
    /// The pooled frame, locked for exclusive access.
    frame: Arc<Mutex<PooledFrame>>,
    /// Weak back-reference to the pool's inner state.
    /// `None` for overflow frames that are not pool-managed.
    pool: Option<Weak<PoolInner>>,
}

impl AcquiredFrame {
    /// Runs `f` with a shared reference to the inner [`PooledFrame`].
    ///
    /// Returns `None` if the frame mutex is poisoned.
    pub fn with<R, F: FnOnce(&PooledFrame) -> R>(&self, f: F) -> R {
        // Frame mutex is never poisoned in normal operation; recover if needed.
        let guard = self.frame.lock().unwrap_or_else(|e| e.into_inner());
        f(&guard)
    }

    /// Runs `f` with a mutable reference to the inner [`PooledFrame`].
    ///
    /// Recovers from a poisoned mutex by clearing the poison.
    pub fn with_mut<R, F: FnOnce(&mut PooledFrame) -> R>(&self, f: F) -> R {
        let mut guard = self.frame.lock().unwrap_or_else(|e| e.into_inner());
        f(&mut guard)
    }

    /// Returns the sequence number of this frame (assigned at acquire time).
    pub fn seq(&self) -> u64 {
        self.frame.lock().unwrap_or_else(|e| e.into_inner()).seq
    }
}

impl Drop for AcquiredFrame {
    fn drop(&mut self) {
        if let Some(weak) = self.pool.take() {
            if let Some(inner) = weak.upgrade() {
                if let Ok(mut state) = inner.state.lock() {
                    if state.idle.len() < state.max_pool_size {
                        inner.release_count.fetch_add(1, Ordering::Relaxed);
                        state.idle.push(Arc::clone(&self.frame));
                    }
                    // If pool is at max_pool_size we just drop the Arc and let the
                    // frame be freed by the Arc's own Drop.
                }
            }
        }
        // For overflow frames (pool = None) the Arc is dropped here, freeing
        // the frame.
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PoolStats
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot of runtime statistics for a [`FramePool`].
#[derive(Debug, Clone, Copy, Default)]
pub struct PoolStats {
    /// Total number of frames pre-allocated at pool creation.
    pub pre_alloc: usize,
    /// Current number of idle (available) frames in the pool.
    pub idle_count: usize,
    /// Cumulative number of successful [`FramePool::acquire`] calls.
    pub acquire_count: u64,
    /// Cumulative number of frames returned to the pool.
    pub release_count: u64,
    /// Number of overflow frames allocated because the pool was empty.
    pub overflow_count: u64,
}

// ─────────────────────────────────────────────────────────────────────────────
// PoolInner
// ─────────────────────────────────────────────────────────────────────────────

struct PoolState {
    idle: Vec<Arc<Mutex<PooledFrame>>>,
    max_pool_size: usize,
}

struct PoolInner {
    state: Mutex<PoolState>,
    frame_byte_size: usize,
    allow_overflow: bool,
    pre_alloc: usize,
    acquire_count: AtomicU64,
    release_count: AtomicU64,
    overflow_count: AtomicU64,
}

// ─────────────────────────────────────────────────────────────────────────────
// FramePool
// ─────────────────────────────────────────────────────────────────────────────

/// A thread-safe pool of pre-allocated frame buffers.
///
/// See [module-level documentation](self) for design details and an example.
pub struct FramePool {
    inner: Arc<PoolInner>,
    seq: AtomicU64,
}

impl FramePool {
    /// Creates a new pool according to `config`, pre-allocating frames
    /// immediately.
    ///
    /// # Panics
    ///
    /// Panics if `config.max_pool_size` is zero or if the computed frame byte
    /// size overflows `usize`.
    #[must_use]
    pub fn new(config: FramePoolConfig) -> Self {
        assert!(
            config.max_pool_size > 0,
            "FramePool max_pool_size must be non-zero"
        );
        let byte_size = config.frame_kind.byte_size();
        let pre_alloc = config.pre_alloc.min(config.max_pool_size);
        let mut idle = Vec::with_capacity(pre_alloc);
        for _ in 0..pre_alloc {
            idle.push(Arc::new(Mutex::new(PooledFrame::new(byte_size))));
        }
        let state = PoolState {
            idle,
            max_pool_size: config.max_pool_size,
        };
        Self {
            inner: Arc::new(PoolInner {
                state: Mutex::new(state),
                frame_byte_size: byte_size,
                allow_overflow: config.allow_overflow,
                pre_alloc,
                acquire_count: AtomicU64::new(0),
                release_count: AtomicU64::new(0),
                overflow_count: AtomicU64::new(0),
            }),
            seq: AtomicU64::new(0),
        }
    }

    /// Acquires an idle frame from the pool.
    ///
    /// - If a frame is available in the pool, it is returned immediately (zero
    ///   allocation).
    /// - If the pool is empty and `allow_overflow` is `true`, a new frame is
    ///   allocated on the heap and returned as an overflow frame (not returned
    ///   to the pool on drop).
    /// - If the pool is empty and `allow_overflow` is `false`, `None` is
    ///   returned.
    pub fn acquire(&self) -> Option<AcquiredFrame> {
        let seq = self.seq.fetch_add(1, Ordering::Relaxed);
        let mut state = self.inner.state.lock().ok()?;
        if let Some(frame_arc) = state.idle.pop() {
            drop(state);
            if let Ok(mut frame) = frame_arc.lock() {
                frame.seq = seq;
            }
            self.inner.acquire_count.fetch_add(1, Ordering::Relaxed);
            Some(AcquiredFrame {
                frame: frame_arc,
                pool: Some(Arc::downgrade(&self.inner)),
            })
        } else if self.inner.allow_overflow {
            drop(state);
            self.inner.overflow_count.fetch_add(1, Ordering::Relaxed);
            let mut frame = PooledFrame::new(self.inner.frame_byte_size);
            frame.seq = seq;
            Some(AcquiredFrame {
                frame: Arc::new(Mutex::new(frame)),
                pool: None, // overflow: not returned to pool
            })
        } else {
            None
        }
    }

    /// Returns a snapshot of runtime pool statistics.
    #[must_use]
    pub fn stats(&self) -> PoolStats {
        let idle_count = self.inner.state.lock().map(|s| s.idle.len()).unwrap_or(0);
        PoolStats {
            pre_alloc: self.inner.pre_alloc,
            idle_count,
            acquire_count: self.inner.acquire_count.load(Ordering::Relaxed),
            release_count: self.inner.release_count.load(Ordering::Relaxed),
            overflow_count: self.inner.overflow_count.load(Ordering::Relaxed),
        }
    }

    /// Returns the number of frames currently idle (available for acquisition).
    pub fn idle_count(&self) -> usize {
        self.inner.state.lock().map(|s| s.idle.len()).unwrap_or(0)
    }

    /// Returns the configured maximum pool size.
    pub fn max_pool_size(&self) -> usize {
        self.inner
            .state
            .lock()
            .map(|s| s.max_pool_size)
            .unwrap_or(0)
    }

    /// Returns the byte size of each frame buffer.
    pub fn frame_byte_size(&self) -> usize {
        self.inner.frame_byte_size
    }

    /// Pre-warms the pool by ensuring at least `target` idle frames are
    /// available, allocating new ones as needed up to `max_pool_size`.
    ///
    /// Returns the number of newly allocated frames.
    pub fn warm(&self, target: usize) -> usize {
        let Ok(mut state) = self.inner.state.lock() else {
            return 0;
        };
        let current = state.idle.len();
        let to_add = target
            .saturating_sub(current)
            .min(state.max_pool_size.saturating_sub(current));
        for _ in 0..to_add {
            state.idle.push(Arc::new(Mutex::new(PooledFrame::new(
                self.inner.frame_byte_size,
            ))));
        }
        to_add
    }

    /// Shrinks the idle pool to at most `target` frames, dropping the rest.
    ///
    /// Returns the number of frames released from the pool.
    pub fn shrink_to(&self, target: usize) -> usize {
        let Ok(mut state) = self.inner.state.lock() else {
            return 0;
        };
        let current = state.idle.len();
        let to_remove = current.saturating_sub(target);
        for _ in 0..to_remove {
            state.idle.pop();
        }
        to_remove
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn video_config(pre_alloc: usize, max: usize) -> FramePoolConfig {
        FramePoolConfig {
            pre_alloc,
            max_pool_size: max,
            frame_kind: FrameKind::Video {
                width: 4,
                height: 4,
                bytes_per_pixel: 3,
            },
            allow_overflow: false,
        }
    }

    // 1. Pre-allocation creates idle frames
    #[test]
    fn test_pre_alloc() {
        let pool = FramePool::new(video_config(4, 8));
        assert_eq!(pool.idle_count(), 4);
        assert_eq!(pool.stats().pre_alloc, 4);
    }

    // 2. Acquire decrements idle count; drop restores it
    #[test]
    fn test_acquire_and_release() {
        let pool = FramePool::new(video_config(2, 4));
        {
            let _f = pool.acquire().expect("frame available");
            assert_eq!(pool.idle_count(), 1);
        }
        // frame returned on drop
        assert_eq!(pool.idle_count(), 2);
        assert_eq!(pool.stats().acquire_count, 1);
        assert_eq!(pool.stats().release_count, 1);
    }

    // 3. Acquire returns None when pool exhausted and allow_overflow=false
    #[test]
    fn test_exhausted_no_overflow() {
        let pool = FramePool::new(video_config(1, 2));
        let _f1 = pool.acquire().expect("first frame");
        assert!(pool.acquire().is_none());
    }

    // 4. Overflow frame is allocated when allow_overflow=true
    #[test]
    fn test_overflow_frame() {
        let config = FramePoolConfig {
            allow_overflow: true,
            ..video_config(0, 2)
        };
        let pool = FramePool::new(config);
        let frame = pool.acquire().expect("overflow frame");
        assert_eq!(pool.stats().overflow_count, 1);
        drop(frame);
        // overflow frames are not returned to the pool
        assert_eq!(pool.idle_count(), 0);
        assert_eq!(pool.stats().release_count, 0);
    }

    // 5. Frame data has correct byte size
    #[test]
    fn test_frame_byte_size() {
        let pool = FramePool::new(video_config(1, 1));
        assert_eq!(pool.frame_byte_size(), 4 * 4 * 3); // 48 bytes
        let frame = pool.acquire().expect("frame");
        frame.with(|f| assert_eq!(f.len(), 48));
    }

    // 6. data_mut / clear
    #[test]
    fn test_frame_data_mut() {
        let pool = FramePool::new(video_config(1, 1));
        let frame = pool.acquire().expect("frame");
        frame.with_mut(|f| {
            f.data_mut()[0] = 0xAB;
            assert_eq!(f.data()[0], 0xAB);
            f.clear();
            assert_eq!(f.data()[0], 0x00);
        });
    }

    // 7. Warm increases idle count
    #[test]
    fn test_warm() {
        let pool = FramePool::new(video_config(0, 8));
        assert_eq!(pool.idle_count(), 0);
        let added = pool.warm(4);
        assert_eq!(added, 4);
        assert_eq!(pool.idle_count(), 4);
    }

    // 8. Warm respects max_pool_size
    #[test]
    fn test_warm_clamps_to_max() {
        let pool = FramePool::new(video_config(2, 4));
        let added = pool.warm(10);
        assert_eq!(added, 2); // only 2 more slots available
        assert_eq!(pool.idle_count(), 4);
    }

    // 9. shrink_to drops excess idle frames
    #[test]
    fn test_shrink_to() {
        let pool = FramePool::new(video_config(6, 8));
        let removed = pool.shrink_to(2);
        assert_eq!(removed, 4);
        assert_eq!(pool.idle_count(), 2);
    }

    // 10. Audio frame byte size
    #[test]
    fn test_audio_frame_kind() {
        let config = FramePoolConfig {
            pre_alloc: 2,
            max_pool_size: 4,
            frame_kind: FrameKind::Audio {
                channel_count: 2,
                samples_per_frame: 1024,
                bytes_per_sample: 4,
            },
            allow_overflow: false,
        };
        let pool = FramePool::new(config);
        assert_eq!(pool.frame_byte_size(), 2 * 1024 * 4);
    }

    // 11. Raw frame kind
    #[test]
    fn test_raw_frame_kind() {
        let config = FramePoolConfig {
            pre_alloc: 1,
            max_pool_size: 2,
            frame_kind: FrameKind::Raw { size: 256 },
            allow_overflow: false,
        };
        let pool = FramePool::new(config);
        assert_eq!(pool.frame_byte_size(), 256);
    }

    // 12. Sequence numbers are monotonically increasing
    #[test]
    fn test_seq_monotonic() {
        let pool = FramePool::new(video_config(4, 8));
        let f1 = pool.acquire().expect("f1");
        let f2 = pool.acquire().expect("f2");
        let s1 = f1.seq();
        let s2 = f2.seq();
        assert!(s2 > s1);
    }

    // 13. Pool returned-over-max frames are freed, not re-pooled
    #[test]
    fn test_return_over_max_freed() {
        let pool = FramePool::new(video_config(0, 1));
        let config2 = FramePoolConfig {
            allow_overflow: true,
            ..video_config(0, 1)
        };
        let pool2 = FramePool::new(config2);
        // acquire one overflow frame and one pool frame
        let _f1 = pool2.acquire(); // overflow (pool empty)
                                   // pre-warm with 1
        pool2.warm(1);
        let f2 = pool2.acquire().expect("pool frame");
        drop(f2); // returned to pool (idle=1)
        let f3 = pool2.acquire().expect("reclaimed frame");
        let f4 = pool2.acquire(); // overflow again
        drop(f3);
        drop(f4);
        // pool size is capped at 1 – only f3 return is counted
        assert!(pool2.stats().release_count >= 1);
        drop(pool);
    }

    // 14. Threaded acquire/release
    #[test]
    fn test_threaded_acquire_release() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let pool = StdArc::new(FramePool::new(video_config(8, 8)));
        let mut handles = Vec::new();
        for _ in 0..4 {
            let p = StdArc::clone(&pool);
            handles.push(thread::spawn(move || {
                for _ in 0..10 {
                    if let Some(frame) = p.acquire() {
                        frame.with_mut(|f| f.data_mut()[0] = 42);
                    }
                }
            }));
        }
        for h in handles {
            h.join().expect("thread");
        }
        let stats = pool.stats();
        assert!(stats.acquire_count > 0);
    }
}
