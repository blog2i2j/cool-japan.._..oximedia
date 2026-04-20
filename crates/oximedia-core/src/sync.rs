//! Core synchronisation primitives for `OxiMedia`.
//!
//! This module offers both lightweight single-threaded simulation types and
//! a fully thread-safe bounded channel with backpressure:
//!
//! - [`AtomicCounter`] – simple monotonic u64 counter (single-threaded)
//! - [`Semaphore`] – counting semaphore (single-threaded)
//! - [`SimRwLock`] – readers-writer lock simulation (single-threaded)
//! - [`Barrier`] – cyclic barrier (single-threaded)
//! - [`BoundedChannel`] – bounded MPSC channel with backpressure (thread-safe)
//!
//! # Note
//!
//! The simulation types ([`AtomicCounter`], [`Semaphore`], [`SimRwLock`],
//! [`Barrier`]) are **logical** (non-OS) implementations intended for testing
//! and single-threaded pipeline coordination.  For real multi-threaded use,
//! prefer `std::sync` or [`BoundedChannel`].
//!
//! # Example
//!
//! ```
//! use oximedia_core::sync::{AtomicCounter, Semaphore, Barrier};
//!
//! let mut counter = AtomicCounter::new(0);
//! counter.increment();
//! assert_eq!(counter.get(), 1);
//!
//! let mut sem = Semaphore::new(3);
//! assert!(sem.acquire());
//! assert_eq!(sem.available(), 2);
//!
//! let mut barrier = Barrier::new(2);
//! assert!(!barrier.wait()); // first thread
//! assert!(barrier.wait());  // last thread – barrier released
//! ```

#![allow(dead_code)]

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

/// A simple, non-atomic counter backed by a plain `u64`.
///
/// Useful as a logical counter in single-threaded or test contexts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AtomicCounter {
    value: u64,
}

impl AtomicCounter {
    /// Creates a new counter initialised to `v`.
    #[must_use]
    pub fn new(v: u64) -> Self {
        Self { value: v }
    }

    /// Increments the counter by 1 and returns the **new** value.
    pub fn increment(&mut self) -> u64 {
        self.value = self.value.saturating_add(1);
        self.value
    }

    /// Decrements the counter by 1 (saturating at 0) and returns the **new** value.
    pub fn decrement(&mut self) -> u64 {
        self.value = self.value.saturating_sub(1);
        self.value
    }

    /// Returns the current value without modifying it.
    #[must_use]
    pub fn get(&self) -> u64 {
        self.value
    }

    /// Overwrites the counter value.
    pub fn set(&mut self, v: u64) {
        self.value = v;
    }
}

impl Default for AtomicCounter {
    fn default() -> Self {
        Self::new(0)
    }
}

/// A counting semaphore that tracks the number of available permits.
#[derive(Debug, Clone)]
pub struct Semaphore {
    count: i64,
    max: i64,
}

impl Semaphore {
    /// Creates a semaphore with `max` permits, all initially available.
    #[must_use]
    pub fn new(max: i64) -> Self {
        Self { count: max, max }
    }

    /// Attempts to acquire one permit.
    ///
    /// Returns `true` on success, `false` when no permits are available.
    pub fn acquire(&mut self) -> bool {
        if self.count > 0 {
            self.count -= 1;
            true
        } else {
            false
        }
    }

    /// Releases one permit back to the semaphore (up to `max`).
    ///
    /// Returns `true` if the permit was accepted, `false` when already full.
    pub fn release(&mut self) -> bool {
        if self.count < self.max {
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// Number of permits currently available.
    #[must_use]
    pub fn available(&self) -> i64 {
        self.count
    }

    /// Maximum capacity of the semaphore.
    #[must_use]
    pub fn max(&self) -> i64 {
        self.max
    }
}

/// A simple readers-writer lock simulation.
///
/// Multiple concurrent readers are permitted; at most one writer is allowed,
/// and only when there are no active readers.
///
/// This is a **single-threaded** simulation (no OS primitives).
#[derive(Debug)]
pub struct SimRwLock<T> {
    data: T,
    readers: u32,
    writer: bool,
}

impl<T> SimRwLock<T> {
    /// Wraps `data` in a new `SimRwLock`.
    #[must_use]
    pub fn new(data: T) -> Self {
        Self {
            data,
            readers: 0,
            writer: false,
        }
    }

    /// Acquires a shared read reference.
    ///
    /// Returns `None` when a writer is active.
    pub fn read(&mut self) -> Option<&T> {
        if self.writer {
            return None;
        }
        self.readers += 1;
        Some(&self.data)
    }

    /// Releases one reader.
    pub fn release_read(&mut self) {
        if self.readers > 0 {
            self.readers -= 1;
        }
    }

    /// Acquires an exclusive write reference.
    ///
    /// Returns `None` when readers are active or a writer already holds the lock.
    pub fn write(&mut self) -> Option<&mut T> {
        if self.writer || self.readers > 0 {
            return None;
        }
        self.writer = true;
        Some(&mut self.data)
    }

    /// Alias for [`write`](Self::write) – tries to acquire write access without blocking.
    pub fn try_write(&mut self) -> Option<&mut T> {
        self.write()
    }

    /// Releases the write lock.
    pub fn release_write(&mut self) {
        self.writer = false;
    }

    /// Returns the number of active readers.
    #[must_use]
    pub fn reader_count(&self) -> u32 {
        self.readers
    }

    /// Returns `true` when a writer holds the lock.
    #[must_use]
    pub fn is_writing(&self) -> bool {
        self.writer
    }
}

/// A cyclic barrier that releases all waiters once `count` threads have called
/// [`wait`](Barrier::wait).
///
/// After the barrier fires, it automatically resets for the next cycle.
#[derive(Debug, Clone)]
pub struct Barrier {
    count: usize,
    waiting: usize,
}

impl Barrier {
    /// Creates a new barrier that triggers after `count` calls to [`wait`](Self::wait).
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self { count, waiting: 0 }
    }

    /// Records one arrival at the barrier.
    ///
    /// Returns `true` for the **last** thread to arrive (the one that
    /// triggers the release), and `false` for all earlier arrivals.
    /// After the last arrival, the barrier is automatically reset.
    pub fn wait(&mut self) -> bool {
        self.waiting += 1;
        if self.waiting >= self.count {
            self.reset();
            true
        } else {
            false
        }
    }

    /// Resets the barrier so it can be used for the next cycle.
    pub fn reset(&mut self) {
        self.waiting = 0;
    }

    /// Number of threads currently waiting.
    #[must_use]
    pub fn waiting_count(&self) -> usize {
        self.waiting
    }

    /// The total count required to release the barrier.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.count
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// BoundedChannel – thread-safe bounded MPSC channel with backpressure
// ─────────────────────────────────────────────────────────────────────────────

/// Error returned by [`BoundedSender`] and [`BoundedReceiver`] operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ChannelError {
    /// The channel has been closed (all senders or all receivers dropped).
    Disconnected,
    /// The channel is currently full (only from [`BoundedSender::try_send`]).
    Full,
    /// The channel is currently empty (only from [`BoundedReceiver::try_recv`]).
    Empty,
}

impl std::fmt::Display for ChannelError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => write!(f, "channel disconnected"),
            Self::Full => write!(f, "channel is full"),
            Self::Empty => write!(f, "channel is empty"),
        }
    }
}

impl std::error::Error for ChannelError {}

/// Shared state hidden behind an `Arc`.
struct ChannelInner<T> {
    queue: Mutex<ChannelState<T>>,
    not_full: Condvar,
    not_empty: Condvar,
}

struct ChannelState<T> {
    buf: VecDeque<T>,
    capacity: usize,
    /// Number of live [`BoundedSender`] handles.
    sender_count: usize,
    /// Number of live [`BoundedReceiver`] handles.
    receiver_count: usize,
}

impl<T> ChannelState<T> {
    fn is_closed_for_send(&self) -> bool {
        self.receiver_count == 0
    }

    fn is_closed_for_recv(&self) -> bool {
        self.sender_count == 0 && self.buf.is_empty()
    }
}

/// The sending half of a [`BoundedChannel`].
///
/// Cloneable – each clone increments the sender reference count so the channel
/// remains open until all senders are dropped.
///
/// [`send`](BoundedSender::send) **blocks** when the channel is full,
/// providing natural backpressure to the producer.
pub struct BoundedSender<T> {
    inner: Arc<ChannelInner<T>>,
}

impl<T> Clone for BoundedSender<T> {
    fn clone(&self) -> Self {
        if let Ok(mut state) = self.inner.queue.lock() {
            state.sender_count += 1;
        }
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Drop for BoundedSender<T> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.inner.queue.lock() {
            state.sender_count -= 1;
            let was_last = state.sender_count == 0;
            drop(state);
            if was_last {
                // Wake any blocked receivers so they can observe the closed state.
                self.inner.not_empty.notify_all();
            }
        }
    }
}

impl<T: Send> BoundedSender<T> {
    /// Sends `item`, **blocking** until space is available or the channel closes.
    ///
    /// Returns `Ok(())` on success, [`ChannelError::Disconnected`] when all
    /// receivers have been dropped.
    pub fn send(&self, item: T) -> Result<(), ChannelError> {
        let mut state = self
            .inner
            .queue
            .lock()
            .map_err(|_| ChannelError::Disconnected)?;
        loop {
            if state.is_closed_for_send() {
                return Err(ChannelError::Disconnected);
            }
            if state.buf.len() < state.capacity {
                state.buf.push_back(item);
                drop(state);
                self.inner.not_empty.notify_one();
                return Ok(());
            }
            // Block until a slot opens (backpressure).
            state = self
                .inner
                .not_full
                .wait(state)
                .map_err(|_| ChannelError::Disconnected)?;
        }
    }

    /// Non-blocking variant; returns [`ChannelError::Full`] immediately if
    /// the channel is at capacity.
    pub fn try_send(&self, item: T) -> Result<(), ChannelError> {
        let mut state = self
            .inner
            .queue
            .lock()
            .map_err(|_| ChannelError::Disconnected)?;
        if state.is_closed_for_send() {
            return Err(ChannelError::Disconnected);
        }
        if state.buf.len() >= state.capacity {
            return Err(ChannelError::Full);
        }
        state.buf.push_back(item);
        drop(state);
        self.inner.not_empty.notify_one();
        Ok(())
    }

    /// Returns the number of items currently in the channel buffer.
    pub fn len(&self) -> usize {
        self.inner.queue.lock().map(|g| g.buf.len()).unwrap_or(0)
    }

    /// Returns `true` when the channel buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the channel capacity.
    pub fn capacity(&self) -> usize {
        self.inner.queue.lock().map(|g| g.capacity).unwrap_or(0)
    }
}

/// The receiving half of a [`BoundedChannel`].
pub struct BoundedReceiver<T> {
    inner: Arc<ChannelInner<T>>,
}

impl<T> Clone for BoundedReceiver<T> {
    fn clone(&self) -> Self {
        if let Ok(mut state) = self.inner.queue.lock() {
            state.receiver_count += 1;
        }
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T> Drop for BoundedReceiver<T> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.inner.queue.lock() {
            state.receiver_count -= 1;
            let was_last = state.receiver_count == 0;
            drop(state);
            if was_last {
                // Wake blocked senders so they can observe the closed state.
                self.inner.not_full.notify_all();
            }
        }
    }
}

impl<T: Send> BoundedReceiver<T> {
    /// Receives an item, **blocking** until one is available or the channel closes.
    ///
    /// Returns `Ok(item)` on success, [`ChannelError::Disconnected`] when all
    /// senders have been dropped **and** the buffer is empty.
    pub fn recv(&self) -> Result<T, ChannelError> {
        let mut state = self
            .inner
            .queue
            .lock()
            .map_err(|_| ChannelError::Disconnected)?;
        loop {
            if let Some(item) = state.buf.pop_front() {
                drop(state);
                self.inner.not_full.notify_one();
                return Ok(item);
            }
            if state.is_closed_for_recv() {
                return Err(ChannelError::Disconnected);
            }
            state = self
                .inner
                .not_empty
                .wait(state)
                .map_err(|_| ChannelError::Disconnected)?;
        }
    }

    /// Non-blocking variant; returns [`ChannelError::Empty`] when no item is
    /// immediately available.
    pub fn try_recv(&self) -> Result<T, ChannelError> {
        let mut state = self
            .inner
            .queue
            .lock()
            .map_err(|_| ChannelError::Disconnected)?;
        if let Some(item) = state.buf.pop_front() {
            drop(state);
            self.inner.not_full.notify_one();
            return Ok(item);
        }
        if state.is_closed_for_recv() {
            Err(ChannelError::Disconnected)
        } else {
            Err(ChannelError::Empty)
        }
    }

    /// Returns the number of items currently in the channel buffer.
    pub fn len(&self) -> usize {
        self.inner.queue.lock().map(|g| g.buf.len()).unwrap_or(0)
    }

    /// Returns `true` when the channel buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A bounded MPSC channel with configurable backpressure.
///
/// Create with [`BoundedChannel::new`]; split into sender/receiver halves with
/// [`into_split`](BoundedChannel::into_split).
///
/// # Example
///
/// ```
/// use oximedia_core::sync::BoundedChannel;
///
/// let (tx, rx) = BoundedChannel::<u32>::new(4).into_split();
/// tx.send(1).expect("send ok");
/// tx.send(2).expect("send ok");
/// assert_eq!(rx.recv().expect("recv ok"), 1);
/// assert_eq!(rx.recv().expect("recv ok"), 2);
/// ```
pub struct BoundedChannel<T> {
    inner: Arc<ChannelInner<T>>,
}

impl<T: Send> BoundedChannel<T> {
    /// Creates a new `BoundedChannel` with the given `capacity`.
    ///
    /// # Panics
    ///
    /// Panics if `capacity` is zero.
    #[must_use]
    pub fn new(capacity: usize) -> Self {
        assert!(capacity > 0, "BoundedChannel capacity must be non-zero");
        let state = ChannelState {
            buf: VecDeque::with_capacity(capacity),
            capacity,
            sender_count: 1,
            receiver_count: 1,
        };
        Self {
            inner: Arc::new(ChannelInner {
                queue: Mutex::new(state),
                not_full: Condvar::new(),
                not_empty: Condvar::new(),
            }),
        }
    }

    /// Splits the channel into a `(BoundedSender<T>, BoundedReceiver<T>)` pair.
    ///
    /// Consumes `self`; the initial reference counts (set to 1 by
    /// [`new`](Self::new)) are transferred to the returned halves.
    pub fn into_split(self) -> (BoundedSender<T>, BoundedReceiver<T>) {
        // Clone each half's Arc (refcount: 1 -> 3 total across inner + tx + rx).
        let tx = BoundedSender {
            inner: Arc::clone(&self.inner),
        };
        let rx = BoundedReceiver {
            inner: Arc::clone(&self.inner),
        };
        // Forget `self` so its Drop does not run, leaving refcount at 2 (tx + rx).
        // The sender_count / receiver_count remain at 1 each, which is correct.
        std::mem::forget(self);
        (tx, rx)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. AtomicCounter::increment
    #[test]
    fn test_counter_increment() {
        let mut c = AtomicCounter::new(0);
        assert_eq!(c.increment(), 1);
        assert_eq!(c.increment(), 2);
    }

    // 2. AtomicCounter::decrement
    #[test]
    fn test_counter_decrement() {
        let mut c = AtomicCounter::new(5);
        assert_eq!(c.decrement(), 4);
        assert_eq!(c.decrement(), 3);
    }

    // 3. AtomicCounter::decrement saturates at 0
    #[test]
    fn test_counter_decrement_saturates() {
        let mut c = AtomicCounter::new(0);
        assert_eq!(c.decrement(), 0);
    }

    // 4. AtomicCounter::get and set
    #[test]
    fn test_counter_get_set() {
        let mut c = AtomicCounter::new(10);
        assert_eq!(c.get(), 10);
        c.set(42);
        assert_eq!(c.get(), 42);
    }

    // 5. Semaphore::acquire / release
    #[test]
    fn test_semaphore_acquire_release() {
        let mut sem = Semaphore::new(3);
        assert_eq!(sem.available(), 3);
        assert!(sem.acquire());
        assert_eq!(sem.available(), 2);
        assert!(sem.release());
        assert_eq!(sem.available(), 3);
    }

    // 6. Semaphore::acquire – blocks when exhausted
    #[test]
    fn test_semaphore_exhausted() {
        let mut sem = Semaphore::new(1);
        assert!(sem.acquire());
        assert!(!sem.acquire()); // No permits left
    }

    // 7. Semaphore::release – does not exceed max
    #[test]
    fn test_semaphore_release_at_max() {
        let mut sem = Semaphore::new(2);
        assert!(!sem.release()); // Already at max
        assert_eq!(sem.available(), 2);
    }

    // 8. Semaphore::max
    #[test]
    fn test_semaphore_max() {
        let sem = Semaphore::new(5);
        assert_eq!(sem.max(), 5);
    }

    // 9. SimRwLock – read while no writer
    #[test]
    fn test_rwlock_read_success() {
        let mut lock = SimRwLock::new(42u32);
        let val = lock.read().copied();
        assert_eq!(val, Some(42));
        assert_eq!(lock.reader_count(), 1);
        lock.release_read();
        assert_eq!(lock.reader_count(), 0);
    }

    // 10. SimRwLock – write while readers blocked
    #[test]
    fn test_rwlock_write_blocked_by_readers() {
        let mut lock = SimRwLock::new(0u32);
        let _ = lock.read();
        assert!(lock.write().is_none());
        lock.release_read();
        assert!(lock.write().is_some());
    }

    // 11. SimRwLock – read blocked by writer
    #[test]
    fn test_rwlock_read_blocked_by_writer() {
        let mut lock = SimRwLock::new(0u32);
        let _ = lock.write();
        assert!(lock.read().is_none());
        lock.release_write();
        assert!(lock.read().is_some());
    }

    // 12. SimRwLock – try_write
    #[test]
    fn test_rwlock_try_write() {
        let mut lock = SimRwLock::new(99u32);
        {
            let w = lock.try_write().expect("try_write should succeed");
            *w = 100;
        }
        lock.release_write();
        let v = lock.read().copied();
        assert_eq!(v, Some(100));
    }

    // 13. Barrier – fires on last thread
    #[test]
    fn test_barrier_fires() {
        let mut b = Barrier::new(3);
        assert!(!b.wait()); // 1st
        assert!(!b.wait()); // 2nd
        assert!(b.wait()); // 3rd – fires
                           // After firing, barrier is reset
        assert_eq!(b.waiting_count(), 0);
    }

    // 14. Barrier – cyclic reuse
    #[test]
    fn test_barrier_cyclic() {
        let mut b = Barrier::new(2);
        assert!(!b.wait());
        assert!(b.wait()); // fires & resets
                           // Second cycle
        assert!(!b.wait());
        assert!(b.wait()); // fires again
    }

    // 15. Barrier – target_count
    #[test]
    fn test_barrier_target_count() {
        let b = Barrier::new(7);
        assert_eq!(b.target_count(), 7);
    }

    // ── BoundedChannel tests ─────────────────────────────────────────────────

    // 16. Basic send/recv round-trip
    #[test]
    fn test_bounded_channel_basic() {
        let (tx, rx) = BoundedChannel::<i32>::new(4).into_split();
        tx.send(10).expect("send");
        tx.send(20).expect("send");
        assert_eq!(rx.recv().expect("recv"), 10);
        assert_eq!(rx.recv().expect("recv"), 20);
    }

    // 17. try_send returns Full when at capacity
    #[test]
    fn test_bounded_channel_try_send_full() {
        let (tx, rx) = BoundedChannel::<i32>::new(2).into_split();
        assert!(tx.try_send(1).is_ok());
        assert!(tx.try_send(2).is_ok());
        assert_eq!(tx.try_send(3), Err(ChannelError::Full));
        // drain so the channel can close cleanly
        rx.try_recv().expect("recv 1");
        rx.try_recv().expect("recv 2");
    }

    // 18. try_recv returns Empty when buffer is empty
    #[test]
    fn test_bounded_channel_try_recv_empty() {
        let (_tx, rx) = BoundedChannel::<i32>::new(2).into_split();
        assert_eq!(rx.try_recv(), Err(ChannelError::Empty));
    }

    // 19. Disconnected when receiver is dropped
    #[test]
    fn test_bounded_channel_disconnected_on_recv_drop() {
        let (tx, rx) = BoundedChannel::<i32>::new(2).into_split();
        drop(rx);
        assert_eq!(tx.try_send(1), Err(ChannelError::Disconnected));
    }

    // 20. Disconnected when sender is dropped and buffer is empty
    #[test]
    fn test_bounded_channel_disconnected_on_send_drop() {
        let (tx, rx) = BoundedChannel::<i32>::new(2).into_split();
        drop(tx);
        assert_eq!(rx.try_recv(), Err(ChannelError::Disconnected));
    }

    // 21. After sender drop, buffered items are still readable
    #[test]
    fn test_bounded_channel_drain_after_sender_drop() {
        let (tx, rx) = BoundedChannel::<i32>::new(4).into_split();
        tx.send(7).expect("send");
        tx.send(8).expect("send");
        drop(tx);
        assert_eq!(rx.recv().expect("recv"), 7);
        assert_eq!(rx.recv().expect("recv"), 8);
        assert_eq!(rx.recv(), Err(ChannelError::Disconnected));
    }

    // 22. len/is_empty/capacity accessors
    #[test]
    fn test_bounded_channel_accessors() {
        let (tx, rx) = BoundedChannel::<u8>::new(8).into_split();
        assert_eq!(tx.capacity(), 8);
        assert!(tx.is_empty());
        tx.send(1).expect("send");
        tx.send(2).expect("send");
        assert_eq!(tx.len(), 2);
        assert_eq!(rx.len(), 2);
    }

    // 23. Multi-threaded producer/consumer with backpressure
    #[test]
    fn test_bounded_channel_threaded() {
        use std::thread;
        let (tx, rx) = BoundedChannel::<u32>::new(4).into_split();
        let producer = thread::spawn(move || {
            for i in 0..16_u32 {
                tx.send(i).expect("send");
            }
        });
        let consumer = thread::spawn(move || {
            let mut out = Vec::with_capacity(16);
            for _ in 0..16 {
                out.push(rx.recv().expect("recv"));
            }
            out
        });
        producer.join().expect("producer");
        let result = consumer.join().expect("consumer");
        assert_eq!(result, (0..16).collect::<Vec<_>>());
    }

    // 24. Cloned sender increments reference count correctly
    #[test]
    fn test_bounded_channel_clone_sender() {
        let (tx, rx) = BoundedChannel::<i32>::new(4).into_split();
        let tx2 = tx.clone();
        tx.send(1).expect("send");
        tx2.send(2).expect("send");
        drop(tx);
        drop(tx2);
        // Both senders dropped – receiver should see Disconnected after draining
        assert_eq!(rx.recv().expect("recv 1"), 1);
        assert_eq!(rx.recv().expect("recv 2"), 2);
        assert_eq!(rx.recv(), Err(ChannelError::Disconnected));
    }
}
