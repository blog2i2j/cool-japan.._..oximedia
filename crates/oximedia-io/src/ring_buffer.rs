//! Lock-free style ring buffer for streaming I/O.
//!
//! Provides a generic circular buffer and a byte-specialised variant
//! with slice-oriented push/pop helpers for use in streaming pipelines.

#![allow(dead_code)]

/// A circular (ring) buffer with fixed capacity.
///
/// Items are stored in a pre-allocated `Vec<Option<T>>`.  `push` fails when
/// the buffer is full; `pop` removes and returns the oldest item.
pub struct RingBuffer<T> {
    data: Vec<Option<T>>,
    head: usize,
    tail: usize,
    capacity: usize,
    len: usize,
}

impl<T> RingBuffer<T> {
    /// Create a new ring buffer with the given capacity.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "RingBuffer capacity must be > 0");
        let mut data = Vec::with_capacity(capacity);
        for _ in 0..capacity {
            data.push(None);
        }
        Self {
            data,
            head: 0,
            tail: 0,
            capacity,
            len: 0,
        }
    }

    /// Push an item onto the back of the buffer.
    ///
    /// Returns `false` if the buffer is full and the item was not inserted.
    pub fn push(&mut self, item: T) -> bool {
        if self.is_full() {
            return false;
        }
        self.data[self.tail] = Some(item);
        self.tail = (self.tail + 1) % self.capacity;
        self.len += 1;
        true
    }

    /// Remove and return the oldest item from the front of the buffer.
    pub fn pop(&mut self) -> Option<T> {
        if self.is_empty() {
            return None;
        }
        let item = self.data[self.head].take();
        self.head = (self.head + 1) % self.capacity;
        self.len -= 1;
        item
    }

    /// Peek at the next item without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        if self.is_empty() {
            return None;
        }
        self.data[self.head].as_ref()
    }

    /// Number of items currently in the buffer
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the buffer contains no items
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `true` if the buffer is at capacity
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == self.capacity
    }

    /// Maximum number of items the buffer can hold
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Remove all items from the buffer
    pub fn clear(&mut self) {
        for slot in &mut self.data {
            *slot = None;
        }
        self.head = 0;
        self.tail = 0;
        self.len = 0;
    }
}

impl<T: Clone> RingBuffer<T> {
    /// Collect all items into a `Vec` without removing them (front-to-back order)
    #[must_use]
    pub fn to_vec(&self) -> Vec<T> {
        let mut result = Vec::with_capacity(self.len);
        let mut idx = self.head;
        for _ in 0..self.len {
            if let Some(ref item) = self.data[idx] {
                result.push(item.clone());
            }
            idx = (idx + 1) % self.capacity;
        }
        result
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Byte-specialised ring buffer
// ──────────────────────────────────────────────────────────────────────────────

/// A ring buffer specialised for `u8` data with slice-oriented helpers.
pub struct ByteRingBuffer {
    inner: RingBuffer<u8>,
}

impl ByteRingBuffer {
    /// Create a new byte ring buffer with the given capacity in bytes
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: RingBuffer::new(capacity),
        }
    }

    /// Push as many bytes from `data` as will fit.
    ///
    /// Returns the number of bytes actually pushed.
    pub fn push_slice(&mut self, data: &[u8]) -> usize {
        let mut pushed = 0;
        for &byte in data {
            if !self.inner.push(byte) {
                break;
            }
            pushed += 1;
        }
        pushed
    }

    /// Pop exactly `n` bytes, returning `None` if fewer than `n` are available.
    pub fn pop_exact(&mut self, n: usize) -> Option<Vec<u8>> {
        if self.inner.len() < n {
            return None;
        }
        let mut result = Vec::with_capacity(n);
        for _ in 0..n {
            if let Some(b) = self.inner.pop() {
                result.push(b);
            }
        }
        Some(result)
    }

    /// Push a single byte; returns `false` if the buffer is full
    pub fn push(&mut self, byte: u8) -> bool {
        self.inner.push(byte)
    }

    /// Pop a single byte
    pub fn pop(&mut self) -> Option<u8> {
        self.inner.pop()
    }

    /// Peek at the next byte without removing it
    #[must_use]
    pub fn peek(&self) -> Option<&u8> {
        self.inner.peek()
    }

    /// Number of bytes currently stored
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if no bytes are stored
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns `true` if the buffer is at capacity
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.inner.is_full()
    }

    /// Maximum byte capacity
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.inner.capacity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── RingBuffer<u32> ───────────────────────────────────────────────────────

    #[test]
    fn test_ring_new_empty() {
        let rb: RingBuffer<u32> = RingBuffer::new(4);
        assert!(rb.is_empty());
        assert!(!rb.is_full());
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), 4);
    }

    #[test]
    fn test_ring_push_and_pop_fifo() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(4);
        assert!(rb.push(1));
        assert!(rb.push(2));
        assert!(rb.push(3));
        assert_eq!(rb.pop(), Some(1));
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), None);
    }

    #[test]
    fn test_ring_full_returns_false() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(2);
        assert!(rb.push(10));
        assert!(rb.push(20));
        assert!(rb.is_full());
        assert!(!rb.push(30)); // must fail
    }

    #[test]
    fn test_ring_peek_does_not_remove() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(4);
        rb.push(42);
        assert_eq!(rb.peek(), Some(&42));
        assert_eq!(rb.len(), 1);
        assert_eq!(rb.pop(), Some(42));
    }

    #[test]
    fn test_ring_wrap_around() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(3);
        rb.push(1);
        rb.push(2);
        rb.push(3);
        rb.pop(); // remove 1
        rb.push(4); // wrap
        assert_eq!(rb.pop(), Some(2));
        assert_eq!(rb.pop(), Some(3));
        assert_eq!(rb.pop(), Some(4));
    }

    #[test]
    fn test_ring_clear() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(4);
        rb.push(1);
        rb.push(2);
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn test_ring_to_vec() {
        let mut rb: RingBuffer<u32> = RingBuffer::new(4);
        rb.push(10);
        rb.push(20);
        rb.push(30);
        assert_eq!(rb.to_vec(), vec![10, 20, 30]);
    }

    // ── ByteRingBuffer ────────────────────────────────────────────────────────

    #[test]
    fn test_byte_ring_push_slice_full() {
        let mut brb = ByteRingBuffer::new(4);
        let pushed = brb.push_slice(&[1, 2, 3, 4, 5]);
        assert_eq!(pushed, 4); // only 4 fit
        assert!(brb.is_full());
    }

    #[test]
    fn test_byte_ring_pop_exact_success() {
        let mut brb = ByteRingBuffer::new(8);
        brb.push_slice(&[10, 20, 30, 40]);
        let out = brb.pop_exact(3).expect("pop_exact should succeed");
        assert_eq!(out, vec![10, 20, 30]);
        assert_eq!(brb.len(), 1);
    }

    #[test]
    fn test_byte_ring_pop_exact_insufficient() {
        let mut brb = ByteRingBuffer::new(8);
        brb.push_slice(&[1, 2]);
        assert!(brb.pop_exact(5).is_none());
        // Data should still be there
        assert_eq!(brb.len(), 2);
    }

    #[test]
    fn test_byte_ring_peek() {
        let mut brb = ByteRingBuffer::new(8);
        brb.push(0xAB);
        assert_eq!(brb.peek(), Some(&0xAB));
        assert_eq!(brb.len(), 1);
    }

    #[test]
    fn test_byte_ring_wrap_around() {
        let mut brb = ByteRingBuffer::new(4);
        brb.push_slice(&[1, 2, 3, 4]);
        brb.pop();
        brb.pop();
        let pushed = brb.push_slice(&[5, 6]);
        assert_eq!(pushed, 2);
        assert_eq!(brb.pop(), Some(3));
        assert_eq!(brb.pop(), Some(4));
        assert_eq!(brb.pop(), Some(5));
        assert_eq!(brb.pop(), Some(6));
    }
}
