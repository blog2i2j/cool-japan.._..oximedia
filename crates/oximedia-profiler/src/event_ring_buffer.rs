//! Zero-allocation fixed-capacity ring buffer for profiling event streams.
//!
//! `EventRingBuffer<T, N>` stores up to `N` events of type `T` in a
//! stack-allocated array.  When the buffer is full the oldest event is
//! silently overwritten (FIFO / circular semantics).
//!
//! The const-generic capacity `N` must be > 0; a capacity of 0 is caught at
//! compile time via a trait bound.
//!
//! # Example
//!
//! ```
//! use oximedia_profiler::event_ring_buffer::EventRingBuffer;
//!
//! let mut rb: EventRingBuffer<u64, 4> = EventRingBuffer::new();
//! rb.push(1);
//! rb.push(2);
//! rb.push(3);
//! rb.push(4);
//! rb.push(5); // overwrites 1
//! assert_eq!(rb.len(), 4);
//! let items: Vec<u64> = rb.iter().collect();
//! assert_eq!(items, vec![2, 3, 4, 5]);
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// EventRingBuffer
// ---------------------------------------------------------------------------

/// Zero-allocation fixed-capacity circular buffer for `Copy` event types.
///
/// The buffer is backed by a const-generic array of size `N`, so no heap
/// allocation is performed.  When `N` slots are filled each subsequent
/// [`push`](Self::push) overwrites the oldest element.
///
/// # Type Parameters
///
/// * `T` — element type; must be `Copy` and have a sensible default.
/// * `N` — compile-time capacity; must be ≥ 1.
#[derive(Debug, Clone)]
pub struct EventRingBuffer<T: Copy + Default, const N: usize> {
    buf: [T; N],
    /// Write head (index where the next push will land).
    head: usize,
    /// Current number of valid elements (≤ N).
    len: usize,
}

impl<T: Copy + Default, const N: usize> EventRingBuffer<T, N> {
    /// Create an empty ring buffer.
    ///
    /// # Panics
    ///
    /// Panics at compile time (monomorphisation) if `N == 0`.
    #[must_use]
    pub fn new() -> Self {
        // Statically assert N > 0 — the const expression evaluates at monomorphisation.
        const { assert!(N > 0, "EventRingBuffer capacity N must be > 0") }
        Self {
            buf: [T::default(); N],
            head: 0,
            len: 0,
        }
    }

    /// Push an event into the buffer.
    ///
    /// If the buffer is already full the oldest element is overwritten.
    pub fn push(&mut self, item: T) {
        self.buf[self.head] = item;
        self.head = (self.head + 1) % N;
        if self.len < N {
            self.len += 1;
        }
    }

    /// Return the number of valid events currently stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Return `true` if no events have been recorded yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Return `true` if the buffer has reached its capacity.
    #[must_use]
    pub fn is_full(&self) -> bool {
        self.len == N
    }

    /// Return the fixed capacity of this buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        N
    }

    /// Clear all events (the underlying array is NOT zeroed, but `len` resets).
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    /// Peek at the most-recently pushed element, if any.
    #[must_use]
    pub fn last(&self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        let idx = if self.head == 0 { N - 1 } else { self.head - 1 };
        Some(self.buf[idx])
    }

    /// Peek at the oldest (least recently pushed) element, if any.
    #[must_use]
    pub fn first(&self) -> Option<T> {
        if self.len == 0 {
            return None;
        }
        // When not yet full the oldest element is at index 0.
        // When full the oldest element is at `head` (the slot just overwritten).
        let idx = if self.len < N { 0 } else { self.head };
        Some(self.buf[idx])
    }

    /// Return an iterator that yields elements in insertion order (oldest first).
    pub fn iter(&self) -> RingIter<'_, T, N> {
        RingIter {
            buf: &self.buf,
            start: if self.len < N { 0 } else { self.head },
            len: self.len,
            emitted: 0,
            capacity: N,
        }
    }
}

impl<T: Copy + Default, const N: usize> Default for EventRingBuffer<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Iterator
// ---------------------------------------------------------------------------

/// Iterator over an [`EventRingBuffer`] in insertion order.
pub struct RingIter<'a, T: Copy, const N: usize> {
    buf: &'a [T; N],
    start: usize,
    len: usize,
    emitted: usize,
    capacity: usize,
}

impl<T: Copy, const N: usize> Iterator for RingIter<'_, T, N> {
    type Item = T;

    fn next(&mut self) -> Option<T> {
        if self.emitted >= self.len {
            return None;
        }
        let idx = (self.start + self.emitted) % self.capacity;
        self.emitted += 1;
        Some(self.buf[idx])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.len - self.emitted;
        (remaining, Some(remaining))
    }
}

impl<T: Copy, const N: usize> ExactSizeIterator for RingIter<'_, T, N> {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_empty() {
        let rb: EventRingBuffer<u32, 8> = EventRingBuffer::new();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
        assert_eq!(rb.capacity(), 8);
    }

    #[test]
    fn test_push_below_capacity() {
        let mut rb: EventRingBuffer<u32, 8> = EventRingBuffer::new();
        rb.push(10);
        rb.push(20);
        assert_eq!(rb.len(), 2);
        assert!(!rb.is_full());
    }

    #[test]
    fn test_push_fills_capacity() {
        let mut rb: EventRingBuffer<u32, 4> = EventRingBuffer::new();
        for i in 0..4 {
            rb.push(i);
        }
        assert_eq!(rb.len(), 4);
        assert!(rb.is_full());
    }

    #[test]
    fn test_overwrite_oldest() {
        let mut rb: EventRingBuffer<u32, 4> = EventRingBuffer::new();
        for i in 1..=5 {
            rb.push(i);
        }
        assert_eq!(rb.len(), 4);
        let items: Vec<u32> = rb.iter().collect();
        assert_eq!(items, vec![2, 3, 4, 5]);
    }

    #[test]
    fn test_iter_insertion_order() {
        let mut rb: EventRingBuffer<u8, 6> = EventRingBuffer::new();
        for i in 0..6u8 {
            rb.push(i);
        }
        let items: Vec<u8> = rb.iter().collect();
        assert_eq!(items, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_iter_after_wrap() {
        let mut rb: EventRingBuffer<u8, 4> = EventRingBuffer::new();
        for i in 0..7u8 {
            rb.push(i);
        }
        // After 7 pushes into capacity-4 buffer: oldest is 3, newest is 6.
        let items: Vec<u8> = rb.iter().collect();
        assert_eq!(items, vec![3, 4, 5, 6]);
    }

    #[test]
    fn test_last_element() {
        let mut rb: EventRingBuffer<i32, 4> = EventRingBuffer::new();
        assert!(rb.last().is_none());
        rb.push(100);
        rb.push(200);
        assert_eq!(rb.last(), Some(200));
    }

    #[test]
    fn test_first_element() {
        let mut rb: EventRingBuffer<i32, 4> = EventRingBuffer::new();
        assert!(rb.first().is_none());
        rb.push(10);
        rb.push(20);
        assert_eq!(rb.first(), Some(10));
    }

    #[test]
    fn test_first_after_wrap() {
        let mut rb: EventRingBuffer<i32, 3> = EventRingBuffer::new();
        rb.push(1);
        rb.push(2);
        rb.push(3);
        rb.push(4); // wraps; oldest is now 2
        assert_eq!(rb.first(), Some(2));
    }

    #[test]
    fn test_clear() {
        let mut rb: EventRingBuffer<u64, 4> = EventRingBuffer::new();
        rb.push(1);
        rb.push(2);
        rb.clear();
        assert!(rb.is_empty());
        assert_eq!(rb.len(), 0);
    }

    #[test]
    fn test_exact_size_iterator() {
        let mut rb: EventRingBuffer<u8, 4> = EventRingBuffer::new();
        for i in 0..4 {
            rb.push(i);
        }
        let mut it = rb.iter();
        assert_eq!(it.len(), 4);
        it.next();
        assert_eq!(it.len(), 3);
    }

    #[test]
    fn test_capacity_1() {
        let mut rb: EventRingBuffer<u32, 1> = EventRingBuffer::new();
        rb.push(42);
        rb.push(99);
        assert_eq!(rb.len(), 1);
        assert_eq!(rb.last(), Some(99));
        let items: Vec<u32> = rb.iter().collect();
        assert_eq!(items, vec![99]);
    }

    #[test]
    fn test_default_trait() {
        let rb: EventRingBuffer<u32, 8> = EventRingBuffer::default();
        assert!(rb.is_empty());
    }
}
