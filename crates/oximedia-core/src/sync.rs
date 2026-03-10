//! Core synchronisation primitives for `OxiMedia`.
//!
//! This module offers lightweight, single-threaded simulation types for
//! synchronisation concepts, useful in non-`async` multimedia pipelines:
//!
//! - [`AtomicCounter`] – simple monotonic u64 counter
//! - [`Semaphore`] – counting semaphore
//! - [`SimRwLock`] – readers-writer lock simulation
//! - [`Barrier`] – cyclic barrier
//!
//! # Note
//!
//! These are **logical** (non-OS) implementations intended for testing and
//! single-threaded pipeline coordination.  For real multi-threaded use,
//! prefer `std::sync`.
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
}
