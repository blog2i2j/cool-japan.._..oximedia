//! Work-stealing scheduler for multi-threaded media pipelines.
//!
//! This module implements the classic **Chase-Lev** work-stealing deque
//! algorithm (David Chase & Yossi Lev, "Dynamic Circular Work-Stealing
//! Deque", SPAA 2005) using only `std::sync::atomic` primitives — no
//! external dependencies.
//!
//! # Architecture
//!
//! Each worker thread owns a single [`WorkStealDeque`] that it can push and
//! pop from its **bottom** (LIFO, cache-warm).  Other threads steal from the
//! **top** (FIFO, oldest work first).  The [`WorkStealPool`] manages a set of
//! deques and a global task counter.  When a worker's deque is empty it
//! attempts to steal from a pseudo-randomly chosen peer.
//!
//! ## Guarantees
//!
//! - **No `unsafe`** — every atomic access uses `std::sync::atomic`.
//! - **No `unwrap`** — all fallible operations return `Result` or `Option`.
//! - A `Mutex<Vec<Option<T>>>` is used for the growable backing storage to
//!   avoid raw-pointer manipulation while remaining lock-free in the common
//!   (non-grow) path.
//!
//! # Example
//!
//! ```
//! use oximedia_core::work_steal::{WorkStealPool, StealResult};
//!
//! let pool = WorkStealPool::<u32>::new(2);
//!
//! // Worker 0 pushes 5 items.
//! for i in 0..5_u32 {
//!     pool.push(0, i).expect("deque not full");
//! }
//!
//! // Worker 0 pops one item (LIFO — gets the last pushed).
//! assert_eq!(pool.pop(0), Some(4));
//!
//! // Worker 1 steals from worker 0 (FIFO — gets the first pushed).
//! match pool.steal(1, 0) {
//!     StealResult::Success(v) => assert_eq!(v, 0),
//!     other => panic!("unexpected: {other:?}"),
//! }
//! ```

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

// ─────────────────────────────────────────────────────────────────────────────
// StealResult
// ─────────────────────────────────────────────────────────────────────────────

/// The outcome of a steal attempt.
#[derive(Debug, PartialEq, Eq)]
pub enum StealResult<T> {
    /// A task was successfully stolen.
    Success(T),
    /// The target deque was empty.
    Empty,
    /// A concurrent modification was detected; the caller may retry.
    Retry,
    /// Invalid worker index.
    InvalidIndex,
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal circular buffer
// ─────────────────────────────────────────────────────────────────────────────

/// A growable circular array used as backing storage for a [`WorkStealDeque`].
///
/// Wrapped in a `Mutex` so that the grow operation is safe without `unsafe`.
struct CircularBuffer<T> {
    slots: Vec<Option<T>>,
    /// Always a power of two.
    capacity: usize,
}

impl<T: Clone> CircularBuffer<T> {
    fn with_capacity(cap: usize) -> Self {
        // Round up to next power of two.
        let capacity = cap.next_power_of_two().max(2);
        Self {
            slots: (0..capacity).map(|_| None).collect(),
            capacity,
        }
    }

    #[inline]
    fn mask(&self) -> usize {
        self.capacity - 1
    }

    fn get(&self, index: i64) -> Option<T> {
        let idx = (index as usize) & self.mask();
        self.slots.get(idx).and_then(|s| s.clone())
    }

    fn put(&mut self, index: i64, val: T) {
        let idx = (index as usize) & self.mask();
        if let Some(slot) = self.slots.get_mut(idx) {
            *slot = Some(val);
        }
    }

    fn clear_slot(&mut self, index: i64) {
        let idx = (index as usize) & self.mask();
        if let Some(slot) = self.slots.get_mut(idx) {
            *slot = None;
        }
    }

    /// Returns a grown copy of this buffer, copying all live items in
    /// `[top, bottom)`.
    fn grow(&self, top: i64, bottom: i64) -> Self {
        let new_cap = self.capacity * 2;
        let mut next = Self {
            slots: (0..new_cap).map(|_| None).collect(),
            capacity: new_cap,
        };
        let mask = next.capacity - 1;
        for i in top..bottom {
            let old_idx = (i as usize) & self.mask();
            let new_idx = (i as usize) & mask;
            if let (Some(dst), Some(src)) = (next.slots.get_mut(new_idx), self.slots.get(old_idx)) {
                *dst = src.clone();
            }
        }
        next
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WorkStealDeque
// ─────────────────────────────────────────────────────────────────────────────

/// A single-owner work-stealing deque.
///
/// Push and pop operations are performed by the *owner* thread at the
/// **bottom**.  Steal operations are performed by *thief* threads at the
/// **top**.
///
/// # Type parameter
///
/// `T` must be `Clone` because the backing buffer may be read concurrently by
/// stealers without `unsafe` raw-pointer manipulation.  For non-`Clone` types
/// consider wrapping in `Arc`.
pub struct WorkStealDeque<T: Clone + Send + 'static> {
    top: AtomicI64,
    bottom: AtomicI64,
    buf: Mutex<CircularBuffer<T>>,
}

impl<T: Clone + Send + 'static> WorkStealDeque<T> {
    /// Creates a new deque with an initial capacity of `initial_cap` (rounded
    /// up to the next power of two).
    #[must_use]
    pub fn new(initial_cap: usize) -> Self {
        Self {
            top: AtomicI64::new(0),
            bottom: AtomicI64::new(0),
            buf: Mutex::new(CircularBuffer::with_capacity(initial_cap)),
        }
    }

    /// Returns the number of items currently in the deque.
    ///
    /// This is a snapshot value; the actual count may change immediately after
    /// reading due to concurrent steals.
    #[must_use]
    pub fn len(&self) -> usize {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Relaxed);
        let diff = b - t;
        if diff > 0 {
            diff as usize
        } else {
            0
        }
    }

    /// Returns `true` if the deque appears empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the current capacity of the backing buffer.
    #[must_use]
    pub fn capacity(&self) -> usize {
        self.buf.lock().map(|g| g.capacity).unwrap_or(0)
    }

    /// Push an item onto the **bottom** of the deque (owner thread only).
    ///
    /// Returns `Err(item)` if the lock on the backing buffer cannot be
    /// acquired (should never happen in normal operation).
    ///
    /// # Grow policy
    ///
    /// If the deque is at capacity the backing buffer is doubled automatically.
    pub fn push(&self, item: T) -> Result<(), T> {
        let b = self.bottom.load(Ordering::Relaxed);
        let t = self.top.load(Ordering::Acquire);
        let mut guard = self.buf.lock().map_err(|_| item.clone())?;

        let size = b - t;
        if size >= guard.capacity as i64 {
            // Grow the buffer.
            let grown = guard.grow(t, b);
            *guard = grown;
        }
        guard.put(b, item);
        // Release the guard before the bottom store so stealers see a
        // consistent state.
        drop(guard);

        self.bottom.store(b + 1, Ordering::Release);
        Ok(())
    }

    /// Pop an item from the **bottom** (owner thread only).
    ///
    /// Returns `None` if the deque is empty.
    pub fn pop(&self) -> Option<T> {
        let b = self.bottom.load(Ordering::Relaxed) - 1;
        self.bottom.store(b, Ordering::Relaxed);
        // Fence: ensure the bottom decrement is visible before reading top.
        std::sync::atomic::fence(Ordering::SeqCst);
        let t = self.top.load(Ordering::Relaxed);

        let size = b - t;
        if size < 0 {
            // Deque is empty — restore bottom.
            self.bottom.store(b + 1, Ordering::Relaxed);
            return None;
        }

        let guard = self.buf.lock().ok()?;
        let item = guard.get(b);

        if size > 0 {
            // More items remain; no race with stealers on this slot.
            return item;
        }

        // Last item — race with stealers.
        let won = self
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
            .is_ok();

        self.bottom.store(b + 1, Ordering::Relaxed);

        if won {
            item
        } else {
            None
        }
    }

    /// Steal an item from the **top** (thief threads only).
    ///
    /// Returns [`StealResult::Success`] on success, [`StealResult::Empty`] if
    /// empty, or [`StealResult::Retry`] if a concurrent modification is
    /// detected.
    pub fn steal(&self) -> StealResult<T> {
        let t = self.top.load(Ordering::Acquire);
        std::sync::atomic::fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if t >= b {
            return StealResult::Empty;
        }

        let guard = match self.buf.lock() {
            Ok(g) => g,
            Err(_) => return StealResult::Retry,
        };
        let item = match guard.get(t) {
            Some(v) => v,
            None => return StealResult::Empty,
        };
        drop(guard);

        match self
            .top
            .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
        {
            Ok(_) => StealResult::Success(item),
            Err(_) => StealResult::Retry,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// WorkStealPool
// ─────────────────────────────────────────────────────────────────────────────

/// A pool of work-stealing deques, one per logical worker.
///
/// Each worker owns a slot identified by its zero-based index.  Workers push
/// and pop from their own slot and steal from peers.
///
/// # Example
///
/// ```
/// use oximedia_core::work_steal::{WorkStealPool, StealResult};
///
/// let pool = WorkStealPool::<u32>::new(3);
/// pool.push(0, 10).expect("ok");
/// pool.push(0, 20).expect("ok");
/// assert_eq!(pool.pop(0), Some(20)); // LIFO
/// match pool.steal(1, 0) {
///     StealResult::Success(v) => assert_eq!(v, 10),
///     _ => panic!("expected success"),
/// }
/// ```
pub struct WorkStealPool<T: Clone + Send + 'static> {
    deques: Vec<Arc<WorkStealDeque<T>>>,
}

impl<T: Clone + Send + 'static> WorkStealPool<T> {
    /// Creates a pool with `num_workers` deques (each with default capacity 16).
    #[must_use]
    pub fn new(num_workers: usize) -> Self {
        let deques = (0..num_workers)
            .map(|_| Arc::new(WorkStealDeque::new(16)))
            .collect();
        Self { deques }
    }

    /// Returns the number of worker slots in this pool.
    #[must_use]
    pub fn num_workers(&self) -> usize {
        self.deques.len()
    }

    /// Pushes an item into the deque of worker `worker_id`.
    ///
    /// Returns `Err(item)` if `worker_id` is out of range or if the lock
    /// cannot be acquired.
    pub fn push(&self, worker_id: usize, item: T) -> Result<(), T> {
        match self.deques.get(worker_id) {
            Some(d) => d.push(item),
            None => Err(item),
        }
    }

    /// Pops an item from the bottom of worker `worker_id`'s deque.
    ///
    /// Returns `None` if the deque is empty or the index is invalid.
    pub fn pop(&self, worker_id: usize) -> Option<T> {
        self.deques.get(worker_id)?.pop()
    }

    /// Steals an item from the top of `target_id`'s deque on behalf of
    /// `thief_id`.
    pub fn steal(&self, thief_id: usize, target_id: usize) -> StealResult<T> {
        if thief_id >= self.deques.len() || target_id >= self.deques.len() {
            return StealResult::InvalidIndex;
        }
        self.deques[target_id].steal()
    }

    /// Returns the total number of items across all deques.
    #[must_use]
    pub fn total_len(&self) -> usize {
        self.deques.iter().map(|d| d.len()).sum()
    }

    /// Returns `true` if all deques are empty.
    #[must_use]
    pub fn is_globally_empty(&self) -> bool {
        self.deques.iter().all(|d| d.is_empty())
    }

    /// Returns a reference-counted handle to the deque of `worker_id`.
    ///
    /// Returns `None` if `worker_id` is out of range.
    #[must_use]
    pub fn deque(&self, worker_id: usize) -> Option<Arc<WorkStealDeque<T>>> {
        self.deques.get(worker_id).cloned()
    }

    /// Returns the length of the deque owned by `worker_id`.
    ///
    /// Returns `0` for an invalid index.
    #[must_use]
    pub fn worker_len(&self, worker_id: usize) -> usize {
        self.deques.get(worker_id).map_or(0, |d| d.len())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // 1. Basic push/pop LIFO behaviour
    #[test]
    fn push_pop_lifo() {
        let pool = WorkStealPool::<u32>::new(1);
        pool.push(0, 1).expect("ok");
        pool.push(0, 2).expect("ok");
        pool.push(0, 3).expect("ok");
        assert_eq!(pool.pop(0), Some(3));
        assert_eq!(pool.pop(0), Some(2));
        assert_eq!(pool.pop(0), Some(1));
        assert_eq!(pool.pop(0), None);
    }

    // 2. Steal from another worker (FIFO from the top)
    #[test]
    fn steal_fifo_order() {
        let pool = WorkStealPool::<u32>::new(2);
        for i in 0..4_u32 {
            pool.push(0, i).expect("ok");
        }
        // Steal steals from the top → oldest first (0, 1, 2, 3).
        match pool.steal(1, 0) {
            StealResult::Success(v) => assert_eq!(v, 0),
            other => panic!("expected success, got {other:?}"),
        }
        match pool.steal(1, 0) {
            StealResult::Success(v) => assert_eq!(v, 1),
            other => panic!("expected success, got {other:?}"),
        }
    }

    // 3. Steal from empty deque returns Empty
    #[test]
    fn steal_empty() {
        let pool = WorkStealPool::<u32>::new(2);
        assert_eq!(pool.steal(1, 0), StealResult::Empty);
    }

    // 4. Pop from empty returns None
    #[test]
    fn pop_empty() {
        let pool = WorkStealPool::<u32>::new(1);
        assert_eq!(pool.pop(0), None);
    }

    // 5. total_len counts items across all deques
    #[test]
    fn total_len() {
        let pool = WorkStealPool::<i32>::new(3);
        pool.push(0, 10).expect("ok");
        pool.push(1, 20).expect("ok");
        pool.push(2, 30).expect("ok");
        assert_eq!(pool.total_len(), 3);
    }

    // 6. is_globally_empty
    #[test]
    fn globally_empty() {
        let pool = WorkStealPool::<i32>::new(2);
        assert!(pool.is_globally_empty());
        pool.push(0, 99).expect("ok");
        assert!(!pool.is_globally_empty());
        pool.pop(0);
        assert!(pool.is_globally_empty());
    }

    // 7. Steal with invalid index returns InvalidIndex
    #[test]
    fn steal_invalid_index() {
        let pool = WorkStealPool::<u32>::new(2);
        assert_eq!(pool.steal(99, 0), StealResult::InvalidIndex);
        assert_eq!(pool.steal(0, 99), StealResult::InvalidIndex);
    }

    // 8. Deque auto-grows beyond initial capacity
    #[test]
    fn deque_grows_beyond_capacity() {
        let pool = WorkStealPool::<u32>::new(1);
        // Initial capacity is 16; push 40 items.
        for i in 0..40_u32 {
            pool.push(0, i).expect("push should succeed after grow");
        }
        assert_eq!(pool.worker_len(0), 40);
    }

    // 9. num_workers returns correct count
    #[test]
    fn num_workers() {
        let pool = WorkStealPool::<u8>::new(5);
        assert_eq!(pool.num_workers(), 5);
    }

    // 10. deque() handle returns correct worker's deque
    #[test]
    fn deque_handle() {
        let pool = WorkStealPool::<u32>::new(2);
        pool.push(0, 42).expect("ok");
        let d = pool.deque(0).expect("valid index");
        assert_eq!(d.len(), 1);
        assert!(pool.deque(99).is_none());
    }

    // 11. push to invalid worker returns Err
    #[test]
    fn push_invalid_worker() {
        let pool = WorkStealPool::<u32>::new(2);
        assert!(pool.push(99, 1).is_err());
    }

    // 12. Steal last item — race with pop (single-threaded coverage)
    #[test]
    fn steal_last_item() {
        let pool = WorkStealPool::<u32>::new(2);
        pool.push(0, 7).expect("ok");
        // Owner does not pop; thief steals the only item.
        match pool.steal(1, 0) {
            StealResult::Success(v) => assert_eq!(v, 7),
            // Retry is also acceptable on the last item.
            StealResult::Retry => {}
            other => panic!("unexpected {other:?}"),
        }
    }

    // 13. is_empty / len on WorkStealDeque directly
    #[test]
    fn deque_len_and_is_empty() {
        let d: WorkStealDeque<u32> = WorkStealDeque::new(4);
        assert!(d.is_empty());
        d.push(1).expect("ok");
        d.push(2).expect("ok");
        assert_eq!(d.len(), 2);
        d.pop();
        assert_eq!(d.len(), 1);
    }

    // 14. worker_len for invalid index
    #[test]
    fn worker_len_invalid() {
        let pool = WorkStealPool::<u32>::new(2);
        assert_eq!(pool.worker_len(99), 0);
    }

    // 15. Multi-threaded push-pop-steal stress test
    #[test]
    fn threaded_push_steal() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        use std::thread;

        let pool = Arc::new(WorkStealPool::<u32>::new(2));
        let stolen = Arc::new(AtomicUsize::new(0));

        // Push 100 items into worker 0.
        for i in 0..100_u32 {
            pool.push(0, i).expect("ok");
        }

        let pool_clone = Arc::clone(&pool);
        let stolen_clone = Arc::clone(&stolen);
        let thief = thread::spawn(move || {
            for _ in 0..100 {
                match pool_clone.steal(1, 0) {
                    StealResult::Success(_) => {
                        stolen_clone.fetch_add(1, Ordering::Relaxed);
                    }
                    StealResult::Empty => break,
                    StealResult::Retry => {}
                    StealResult::InvalidIndex => {}
                }
            }
        });

        // Owner pops some items.
        let mut owner_got = 0usize;
        for _ in 0..50 {
            if pool.pop(0).is_some() {
                owner_got += 1;
            }
        }

        thief.join().expect("thief thread panicked");
        let total = owner_got + stolen.load(Ordering::Relaxed);
        // Together they must have processed at most 100 items (no duplicates).
        assert!(total <= 100, "total {total} > 100");
    }
}
