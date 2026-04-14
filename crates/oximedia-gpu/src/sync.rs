//! Synchronization primitives for GPU operations
//!
//! This module provides abstractions for synchronizing GPU operations,
//! including fences, barriers, and event synchronization.

use crate::GpuDevice;
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Fence for CPU-GPU synchronization
///
/// A fence allows the CPU to wait for GPU operations to complete.
pub struct Fence {
    device: Arc<wgpu::Device>,
    signaled: Arc<AtomicBool>,
    timestamp: Arc<AtomicU64>,
}

impl Fence {
    /// Create a new fence
    #[must_use]
    pub fn new(device: &GpuDevice) -> Self {
        Self {
            device: Arc::clone(device.device()),
            signaled: Arc::new(AtomicBool::new(false)),
            timestamp: Arc::new(AtomicU64::new(0)),
        }
    }

    /// Signal the fence
    ///
    /// This should be called after submitting GPU commands that you want to wait for.
    pub fn signal(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
        self.signaled.store(true, Ordering::Release);
        self.timestamp.store(
            Instant::now().elapsed().as_nanos() as u64,
            Ordering::Relaxed,
        );
    }

    /// Wait for the fence to be signaled
    ///
    /// This will block the current thread until the GPU has completed the operations
    /// that were submitted before the fence was signaled.
    pub fn wait(&self) {
        while !self.signaled.load(Ordering::Acquire) {
            std::thread::yield_now();
        }
    }

    /// Wait for the fence with a timeout
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait
    ///
    /// # Returns
    ///
    /// True if the fence was signaled within the timeout, false otherwise.
    #[must_use]
    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let start = Instant::now();
        while !self.signaled.load(Ordering::Acquire) {
            if start.elapsed() > timeout {
                return false;
            }
            std::thread::yield_now();
        }
        true
    }

    /// Check if the fence is signaled without blocking
    #[must_use]
    pub fn is_signaled(&self) -> bool {
        self.signaled.load(Ordering::Acquire)
    }

    /// Reset the fence to unsignaled state
    pub fn reset(&self) {
        self.signaled.store(false, Ordering::Release);
    }

    /// Get the timestamp when the fence was signaled (in nanoseconds)
    #[must_use]
    pub fn timestamp(&self) -> Option<u64> {
        if self.is_signaled() {
            Some(self.timestamp.load(Ordering::Relaxed))
        } else {
            None
        }
    }
}

impl Clone for Fence {
    fn clone(&self) -> Self {
        Self {
            device: Arc::clone(&self.device),
            signaled: Arc::clone(&self.signaled),
            timestamp: Arc::clone(&self.timestamp),
        }
    }
}

/// Semaphore for GPU-GPU synchronization
///
/// Semaphores are used to synchronize operations between different command queues.
pub struct Semaphore {
    value: Arc<AtomicU64>,
    condvar: Arc<Condvar>,
    mutex: Arc<Mutex<()>>,
}

impl Semaphore {
    /// Create a new semaphore with an initial value
    #[must_use]
    pub fn new(initial_value: u64) -> Self {
        Self {
            value: Arc::new(AtomicU64::new(initial_value)),
            condvar: Arc::new(Condvar::new()),
            mutex: Arc::new(Mutex::new(())),
        }
    }

    /// Signal the semaphore (increment value)
    pub fn signal(&self) {
        self.value.fetch_add(1, Ordering::Release);
        self.condvar.notify_all();
    }

    /// Wait for the semaphore (decrement value, block if zero)
    pub fn wait(&self) {
        loop {
            let current = self.value.load(Ordering::Acquire);
            if current > 0 {
                if self
                    .value
                    .compare_exchange_weak(
                        current,
                        current - 1,
                        Ordering::AcqRel,
                        Ordering::Acquire,
                    )
                    .is_ok()
                {
                    return;
                }
            } else {
                let mut guard = self.mutex.lock();
                self.condvar.wait(&mut guard);
            }
        }
    }

    /// Try to wait for the semaphore without blocking
    ///
    /// # Returns
    ///
    /// True if the semaphore was successfully acquired, false otherwise.
    #[must_use]
    pub fn try_wait(&self) -> bool {
        loop {
            let current = self.value.load(Ordering::Acquire);
            if current > 0 {
                match self.value.compare_exchange_weak(
                    current,
                    current - 1,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => return true,
                    Err(_) => continue,
                }
            }
            return false;
        }
    }

    /// Get the current semaphore value
    #[must_use]
    pub fn value(&self) -> u64 {
        self.value.load(Ordering::Acquire)
    }

    /// Reset the semaphore to a specific value
    pub fn reset(&self, value: u64) {
        self.value.store(value, Ordering::Release);
        self.condvar.notify_all();
    }
}

impl Clone for Semaphore {
    fn clone(&self) -> Self {
        Self {
            value: Arc::clone(&self.value),
            condvar: Arc::clone(&self.condvar),
            mutex: Arc::clone(&self.mutex),
        }
    }
}

/// Event for signaling completion of operations
pub struct Event {
    signaled: Arc<AtomicBool>,
    condvar: Arc<Condvar>,
    mutex: Arc<Mutex<()>>,
}

impl Event {
    /// Create a new event
    #[must_use]
    pub fn new() -> Self {
        Self {
            signaled: Arc::new(AtomicBool::new(false)),
            condvar: Arc::new(Condvar::new()),
            mutex: Arc::new(Mutex::new(())),
        }
    }

    /// Signal the event
    pub fn signal(&self) {
        self.signaled.store(true, Ordering::Release);
        self.condvar.notify_all();
    }

    /// Wait for the event to be signaled
    pub fn wait(&self) {
        while !self.signaled.load(Ordering::Acquire) {
            let mut guard = self.mutex.lock();
            self.condvar.wait(&mut guard);
        }
    }

    /// Wait for the event with a timeout
    ///
    /// # Arguments
    ///
    /// * `timeout` - Maximum time to wait
    ///
    /// # Returns
    ///
    /// True if the event was signaled within the timeout, false otherwise.
    #[must_use]
    pub fn wait_timeout(&self, timeout: Duration) -> bool {
        let start = Instant::now();
        while !self.signaled.load(Ordering::Acquire) {
            if start.elapsed() > timeout {
                return false;
            }
            let mut guard = self.mutex.lock();
            let remaining = timeout.saturating_sub(start.elapsed());
            if remaining.is_zero() {
                return false;
            }
            self.condvar.wait_for(&mut guard, remaining);
        }
        true
    }

    /// Check if the event is signaled
    #[must_use]
    pub fn is_signaled(&self) -> bool {
        self.signaled.load(Ordering::Acquire)
    }

    /// Reset the event to unsignaled state
    pub fn reset(&self) {
        self.signaled.store(false, Ordering::Release);
    }
}

impl Default for Event {
    fn default() -> Self {
        Self::new()
    }
}

impl Clone for Event {
    fn clone(&self) -> Self {
        Self {
            signaled: Arc::clone(&self.signaled),
            condvar: Arc::clone(&self.condvar),
            mutex: Arc::clone(&self.mutex),
        }
    }
}

/// Barrier for synchronizing multiple operations
pub struct Barrier {
    total_count: usize,
    current_count: Arc<AtomicU64>,
    generation: Arc<AtomicU64>,
    condvar: Arc<Condvar>,
    mutex: Arc<Mutex<()>>,
}

impl Barrier {
    /// Create a new barrier
    ///
    /// # Arguments
    ///
    /// * `count` - Number of threads/operations that must reach the barrier
    #[must_use]
    pub fn new(count: usize) -> Self {
        Self {
            total_count: count,
            current_count: Arc::new(AtomicU64::new(0)),
            generation: Arc::new(AtomicU64::new(0)),
            condvar: Arc::new(Condvar::new()),
            mutex: Arc::new(Mutex::new(())),
        }
    }

    /// Wait at the barrier
    ///
    /// This will block until all threads/operations have reached the barrier.
    pub fn wait(&self) {
        let gen = self.generation.load(Ordering::Acquire);
        let count = self.current_count.fetch_add(1, Ordering::AcqRel) + 1;

        if count >= self.total_count as u64 {
            // Last thread resets the barrier
            self.current_count.store(0, Ordering::Release);
            self.generation.fetch_add(1, Ordering::Release);
            self.condvar.notify_all();
        } else {
            // Wait for all threads
            let mut guard = self.mutex.lock();
            while gen == self.generation.load(Ordering::Acquire) {
                self.condvar.wait(&mut guard);
            }
        }
    }

    /// Get the total count required for the barrier
    #[must_use]
    pub fn count(&self) -> usize {
        self.total_count
    }

    /// Get the current number of waiting threads
    #[must_use]
    pub fn waiting(&self) -> u64 {
        self.current_count.load(Ordering::Acquire)
    }
}

impl Clone for Barrier {
    fn clone(&self) -> Self {
        Self {
            total_count: self.total_count,
            current_count: Arc::clone(&self.current_count),
            generation: Arc::clone(&self.generation),
            condvar: Arc::clone(&self.condvar),
            mutex: Arc::clone(&self.mutex),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_semaphore() {
        let sem = Semaphore::new(1);
        assert_eq!(sem.value(), 1);

        assert!(sem.try_wait());
        assert_eq!(sem.value(), 0);

        assert!(!sem.try_wait());
        assert_eq!(sem.value(), 0);

        sem.signal();
        assert_eq!(sem.value(), 1);

        assert!(sem.try_wait());
        assert_eq!(sem.value(), 0);
    }

    #[test]
    fn test_event() {
        let event = Event::new();
        assert!(!event.is_signaled());

        event.signal();
        assert!(event.is_signaled());

        event.reset();
        assert!(!event.is_signaled());
    }

    #[test]
    fn test_barrier() {
        let barrier = Barrier::new(3);
        assert_eq!(barrier.count(), 3);
        assert_eq!(barrier.waiting(), 0);
    }
}
