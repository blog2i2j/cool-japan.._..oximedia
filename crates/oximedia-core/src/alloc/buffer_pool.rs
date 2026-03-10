//! Buffer pool for zero-copy memory management.
//!
//! This module provides a [`BufferPool`] for efficient buffer reuse,
//! avoiding allocation overhead in performance-critical paths.

use std::sync::{Arc, RwLock};

/// A pool of reusable buffers for zero-copy operations.
///
/// `BufferPool` manages a collection of fixed-size buffers that can be
/// acquired and released. This helps reduce allocation overhead in
/// hot paths like frame decoding.
///
/// # Thread Safety
///
/// `BufferPool` is thread-safe and can be shared across threads.
/// Acquired buffers are wrapped in `Arc<RwLock<_>>` for safe concurrent access.
///
/// # Examples
///
/// ```
/// use oximedia_core::alloc::BufferPool;
///
/// // Create a pool with 4 buffers of 1MB each
/// let pool = BufferPool::new(4, 1024 * 1024);
///
/// // Acquire a buffer
/// let buffer = pool.acquire();
/// assert!(buffer.is_some());
///
/// // Write to the buffer
/// {
///     let mut guard = buffer.as_ref()?.write()?;
///     guard[0] = 42;
/// }
///
/// // Release it back to the pool
/// pool.release(buffer?);
/// ```
#[derive(Debug)]
pub struct BufferPool {
    /// Available buffers in the pool.
    buffers: RwLock<Vec<Arc<RwLock<Vec<u8>>>>>,
    /// Size of each buffer in bytes.
    buffer_size: usize,
    /// Maximum number of buffers allowed in the pool.
    max_buffers: usize,
}

impl BufferPool {
    /// Creates a new buffer pool.
    ///
    /// # Arguments
    ///
    /// * `count` - Initial number of buffers to allocate
    /// * `buffer_size` - Size of each buffer in bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(8, 4096);
    /// ```
    #[must_use]
    pub fn new(count: usize, buffer_size: usize) -> Self {
        let buffers: Vec<_> = (0..count)
            .map(|_| Arc::new(RwLock::new(vec![0u8; buffer_size])))
            .collect();

        Self {
            buffers: RwLock::new(buffers),
            buffer_size,
            max_buffers: count,
        }
    }

    /// Creates a new buffer pool with a specified maximum capacity.
    ///
    /// The pool starts empty and allocates buffers on demand up to `max_buffers`.
    ///
    /// # Arguments
    ///
    /// * `max_buffers` - Maximum number of buffers the pool can hold
    /// * `buffer_size` - Size of each buffer in bytes
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::with_capacity(16, 8192);
    /// ```
    #[must_use]
    pub fn with_capacity(max_buffers: usize, buffer_size: usize) -> Self {
        Self {
            buffers: RwLock::new(Vec::with_capacity(max_buffers)),
            buffer_size,
            max_buffers,
        }
    }

    /// Acquires a buffer from the pool.
    ///
    /// Returns `None` if no buffers are available. Use [`acquire_or_alloc`](Self::acquire_or_alloc)
    /// if you want to allocate a new buffer when the pool is empty.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(2, 1024);
    /// let buf1 = pool.acquire();
    /// let buf2 = pool.acquire();
    /// let buf3 = pool.acquire(); // Returns None, pool exhausted
    /// assert!(buf1.is_some());
    /// assert!(buf2.is_some());
    /// assert!(buf3.is_none());
    /// ```
    #[must_use]
    pub fn acquire(&self) -> Option<Arc<RwLock<Vec<u8>>>> {
        self.buffers.write().ok()?.pop()
    }

    /// Acquires a buffer from the pool, allocating a new one if necessary.
    ///
    /// If the pool is empty, allocates a new buffer. This is useful when
    /// you need a buffer regardless of pool state.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(0, 1024); // Empty pool
    /// let buffer = pool.acquire_or_alloc();
    /// assert_eq!(buffer.read()?.len(), 1024);
    /// ```
    #[must_use]
    pub fn acquire_or_alloc(&self) -> Arc<RwLock<Vec<u8>>> {
        self.acquire()
            .unwrap_or_else(|| Arc::new(RwLock::new(vec![0u8; self.buffer_size])))
    }

    /// Releases a buffer back to the pool.
    ///
    /// The buffer should have been previously acquired from this pool.
    /// If the pool is at capacity, the buffer is dropped.
    ///
    /// # Arguments
    ///
    /// * `buffer` - The buffer to return to the pool
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(2, 1024);
    /// let buffer = pool.acquire()?;
    /// // Use the buffer...
    /// pool.release(buffer);
    /// ```
    pub fn release(&self, buffer: Arc<RwLock<Vec<u8>>>) {
        if let Ok(mut buffers) = self.buffers.write() {
            if buffers.len() < self.max_buffers {
                // Clear the buffer for security and consistency
                if let Ok(mut guard) = buffer.write() {
                    guard.fill(0);
                }
                buffers.push(buffer);
            }
            // If at capacity, the buffer is simply dropped
        }
    }

    /// Returns the number of buffers currently available in the pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(4, 1024);
    /// assert_eq!(pool.available(), 4);
    /// let _buf = pool.acquire();
    /// assert_eq!(pool.available(), 3);
    /// ```
    #[must_use]
    pub fn available(&self) -> usize {
        self.buffers.read().map(|b| b.len()).unwrap_or(0)
    }

    /// Returns the size of each buffer in the pool.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(2, 4096);
    /// assert_eq!(pool.buffer_size(), 4096);
    /// ```
    #[must_use]
    pub fn buffer_size(&self) -> usize {
        self.buffer_size
    }

    /// Returns the maximum number of buffers the pool can hold.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_core::alloc::BufferPool;
    ///
    /// let pool = BufferPool::new(8, 1024);
    /// assert_eq!(pool.max_buffers(), 8);
    /// ```
    #[must_use]
    pub fn max_buffers(&self) -> usize {
        self.max_buffers
    }
}

impl Default for BufferPool {
    fn default() -> Self {
        Self::new(4, 4096)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new() {
        let pool = BufferPool::new(4, 1024);
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.buffer_size(), 1024);
        assert_eq!(pool.max_buffers(), 4);
    }

    #[test]
    fn test_with_capacity() {
        let pool = BufferPool::with_capacity(8, 2048);
        assert_eq!(pool.available(), 0);
        assert_eq!(pool.buffer_size(), 2048);
        assert_eq!(pool.max_buffers(), 8);
    }

    #[test]
    fn test_acquire_release() {
        let pool = BufferPool::new(2, 1024);
        assert_eq!(pool.available(), 2);

        let buf1 = pool.acquire().expect("acquire should succeed");
        assert_eq!(pool.available(), 1);

        let buf2 = pool.acquire().expect("acquire should succeed");
        assert_eq!(pool.available(), 0);

        assert!(pool.acquire().is_none());

        pool.release(buf1);
        assert_eq!(pool.available(), 1);

        pool.release(buf2);
        assert_eq!(pool.available(), 2);
    }

    #[test]
    fn test_acquire_or_alloc() {
        let pool = BufferPool::new(0, 1024);
        assert_eq!(pool.available(), 0);

        let buffer = pool.acquire_or_alloc();
        assert_eq!(buffer.read().expect("read lock should succeed").len(), 1024);
    }

    #[test]
    fn test_buffer_contents() {
        let pool = BufferPool::new(1, 64);
        let buffer = pool.acquire().expect("acquire should succeed");

        // Write to buffer
        {
            let mut guard = buffer.write().expect("write lock should succeed");
            guard[0] = 42;
            guard[63] = 255;
        }

        // Read from buffer
        {
            let guard = buffer.read().expect("read lock should succeed");
            assert_eq!(guard[0], 42);
            assert_eq!(guard[63], 255);
        }

        // Release and reacquire - buffer should be zeroed
        pool.release(buffer);
        let buffer = pool.acquire().expect("acquire should succeed");
        {
            let guard = buffer.read().expect("read lock should succeed");
            assert_eq!(guard[0], 0);
            assert_eq!(guard[63], 0);
        }
    }

    #[test]
    fn test_default() {
        let pool = BufferPool::default();
        assert_eq!(pool.available(), 4);
        assert_eq!(pool.buffer_size(), 4096);
    }

    #[test]
    fn test_release_at_capacity() {
        let pool = BufferPool::new(2, 1024);
        let extra_buffer = Arc::new(RwLock::new(vec![0u8; 1024]));

        // Pool is full, releasing should not add more buffers
        pool.release(extra_buffer);
        assert_eq!(pool.available(), 2); // Still 2, not 3
    }
}
