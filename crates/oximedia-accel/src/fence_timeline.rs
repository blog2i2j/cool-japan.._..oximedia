//! GPU fence / timeline synchronisation primitives.
//!
//! Models Vulkan-style timeline semaphores and binary fences in pure Rust.
//! Useful for ordering GPU command buffer submissions and CPU/GPU
//! synchronisation without actual Vulkan dependencies in unit tests.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt;
use std::time::{Duration, Instant};

/// Unique identifier for a fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FenceId(u64);

impl fmt::Display for FenceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "fence#{}", self.0)
    }
}

/// State of a binary fence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FenceState {
    /// Fence has not yet been signalled.
    Unsignalled,
    /// Fence has been signalled (work completed).
    Signalled,
}

/// A binary fence that transitions from unsignalled to signalled once.
#[derive(Debug)]
pub struct Fence {
    /// Unique identifier.
    id: FenceId,
    /// Current state.
    state: FenceState,
    /// When the fence was created.
    created_at: Instant,
    /// When the fence was signalled (if ever).
    signalled_at: Option<Instant>,
}

impl Fence {
    /// Create a new unsignalled fence.
    #[must_use]
    fn new(id: FenceId) -> Self {
        Self {
            id,
            state: FenceState::Unsignalled,
            created_at: Instant::now(),
            signalled_at: None,
        }
    }

    /// Fence identifier.
    #[must_use]
    pub fn id(&self) -> FenceId {
        self.id
    }

    /// Current state.
    #[must_use]
    pub fn state(&self) -> FenceState {
        self.state
    }

    /// Whether the fence has been signalled.
    #[must_use]
    pub fn is_signalled(&self) -> bool {
        self.state == FenceState::Signalled
    }

    /// Signal the fence.
    pub fn signal(&mut self) {
        if self.state == FenceState::Unsignalled {
            self.state = FenceState::Signalled;
            self.signalled_at = Some(Instant::now());
        }
    }

    /// Reset the fence back to unsignalled.
    pub fn reset(&mut self) {
        self.state = FenceState::Unsignalled;
        self.signalled_at = None;
    }

    /// Duration between creation and signal, if signalled.
    #[must_use]
    pub fn signal_latency(&self) -> Option<Duration> {
        self.signalled_at.map(|s| s.duration_since(self.created_at))
    }
}

/// A timeline semaphore whose value monotonically increases.
///
/// Work is ordered by associating a timeline value with each submission.
/// Waiting for value N means waiting until the timeline reaches at least N.
#[derive(Debug)]
pub struct TimelineSemaphore {
    /// Current signalled value.
    value: u64,
    /// Record of when each value was signalled.
    history: BTreeMap<u64, Instant>,
    /// Maximum history entries to keep.
    max_history: usize,
}

impl TimelineSemaphore {
    /// Create a new timeline semaphore starting at the given value.
    #[must_use]
    pub fn new(initial_value: u64) -> Self {
        Self {
            value: initial_value,
            history: BTreeMap::new(),
            max_history: 1024,
        }
    }

    /// Current signalled value.
    #[must_use]
    pub fn value(&self) -> u64 {
        self.value
    }

    /// Signal the timeline to a new value.
    ///
    /// The new value must be greater than the current value;
    /// non-monotonic signals are ignored.
    pub fn signal(&mut self, new_value: u64) {
        if new_value > self.value {
            self.value = new_value;
            self.history.insert(new_value, Instant::now());
            self.trim_history();
        }
    }

    /// Increment the timeline by 1 and return the new value.
    pub fn increment(&mut self) -> u64 {
        let next = self.value + 1;
        self.signal(next);
        next
    }

    /// Whether the timeline has reached at least `target`.
    #[must_use]
    pub fn has_reached(&self, target: u64) -> bool {
        self.value >= target
    }

    /// Number of history entries stored.
    #[must_use]
    pub fn history_len(&self) -> usize {
        self.history.len()
    }

    /// The instant at which a given timeline value was signalled, if recorded.
    #[must_use]
    pub fn signalled_at(&self, value: u64) -> Option<Instant> {
        self.history.get(&value).copied()
    }

    /// Trim oldest history entries to stay within budget.
    fn trim_history(&mut self) {
        while self.history.len() > self.max_history {
            if let Some((&oldest_key, _)) = self.history.iter().next() {
                self.history.remove(&oldest_key);
            }
        }
    }
}

/// Pool of fences for reuse (avoids repeated allocation).
#[derive(Debug)]
pub struct FencePool {
    /// Available (recycled) fences.
    free_list: Vec<Fence>,
    /// Next fence id to allocate.
    next_id: u64,
    /// Total fences ever created.
    total_created: u64,
}

impl FencePool {
    /// Create a new empty fence pool.
    #[must_use]
    pub fn new() -> Self {
        Self {
            free_list: Vec::new(),
            next_id: 0,
            total_created: 0,
        }
    }

    /// Acquire a fence from the pool (or create a new one).
    pub fn acquire(&mut self) -> Fence {
        if let Some(mut fence) = self.free_list.pop() {
            fence.reset();
            fence
        } else {
            let id = FenceId(self.next_id);
            self.next_id += 1;
            self.total_created += 1;
            Fence::new(id)
        }
    }

    /// Return a fence to the pool for later reuse.
    pub fn release(&mut self, fence: Fence) {
        self.free_list.push(fence);
    }

    /// Number of fences currently available in the pool.
    #[must_use]
    pub fn available(&self) -> usize {
        self.free_list.len()
    }

    /// Total fences ever created by this pool.
    #[must_use]
    pub fn total_created(&self) -> u64 {
        self.total_created
    }
}

impl Default for FencePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Summary of synchronization state across multiple timelines.
#[derive(Debug, Clone)]
pub struct SyncStatus {
    /// Number of unsignalled fences.
    pub pending_fences: usize,
    /// Highest timeline value across all tracked semaphores.
    pub max_timeline_value: u64,
    /// Number of timelines tracked.
    pub timeline_count: usize,
}

/// A collection of timeline semaphores keyed by name.
#[derive(Debug)]
pub struct TimelineRegistry {
    /// Named timelines.
    timelines: BTreeMap<String, TimelineSemaphore>,
}

impl TimelineRegistry {
    /// Create a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            timelines: BTreeMap::new(),
        }
    }

    /// Get or create a timeline with the given name.
    pub fn get_or_create(&mut self, name: &str) -> &mut TimelineSemaphore {
        self.timelines
            .entry(name.to_owned())
            .or_insert_with(|| TimelineSemaphore::new(0))
    }

    /// Get a timeline by name.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&TimelineSemaphore> {
        self.timelines.get(name)
    }

    /// Number of registered timelines.
    #[must_use]
    pub fn len(&self) -> usize {
        self.timelines.len()
    }

    /// Whether the registry is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.timelines.is_empty()
    }

    /// Compute an overall sync status.
    #[must_use]
    pub fn status(&self) -> SyncStatus {
        let max_val = self
            .timelines
            .values()
            .map(TimelineSemaphore::value)
            .max()
            .unwrap_or(0);
        SyncStatus {
            pending_fences: 0,
            max_timeline_value: max_val,
            timeline_count: self.timelines.len(),
        }
    }
}

impl Default for TimelineRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fence_id_display() {
        let id = FenceId(7);
        assert_eq!(id.to_string(), "fence#7");
    }

    #[test]
    fn test_fence_lifecycle() {
        let mut fence = Fence::new(FenceId(0));
        assert!(!fence.is_signalled());
        fence.signal();
        assert!(fence.is_signalled());
        fence.reset();
        assert!(!fence.is_signalled());
    }

    #[test]
    fn test_fence_double_signal() {
        let mut fence = Fence::new(FenceId(0));
        fence.signal();
        fence.signal(); // should be idempotent
        assert!(fence.is_signalled());
    }

    #[test]
    fn test_timeline_initial_value() {
        let ts = TimelineSemaphore::new(42);
        assert_eq!(ts.value(), 42);
        assert!(ts.has_reached(42));
        assert!(!ts.has_reached(43));
    }

    #[test]
    fn test_timeline_signal() {
        let mut ts = TimelineSemaphore::new(0);
        ts.signal(5);
        assert_eq!(ts.value(), 5);
        assert!(ts.has_reached(5));
    }

    #[test]
    fn test_timeline_non_monotonic_ignored() {
        let mut ts = TimelineSemaphore::new(10);
        ts.signal(5); // lower, should be ignored
        assert_eq!(ts.value(), 10);
    }

    #[test]
    fn test_timeline_increment() {
        let mut ts = TimelineSemaphore::new(0);
        let v1 = ts.increment();
        assert_eq!(v1, 1);
        let v2 = ts.increment();
        assert_eq!(v2, 2);
        assert_eq!(ts.value(), 2);
    }

    #[test]
    fn test_timeline_history() {
        let mut ts = TimelineSemaphore::new(0);
        ts.signal(1);
        ts.signal(2);
        assert_eq!(ts.history_len(), 2);
        assert!(ts.signalled_at(1).is_some());
        assert!(ts.signalled_at(0).is_none());
    }

    #[test]
    fn test_fence_pool_acquire_release() {
        let mut pool = FencePool::new();
        assert_eq!(pool.available(), 0);
        let f = pool.acquire();
        assert_eq!(pool.total_created(), 1);
        pool.release(f);
        assert_eq!(pool.available(), 1);
        let f2 = pool.acquire();
        assert_eq!(pool.total_created(), 1); // reused, not created
        assert!(!f2.is_signalled());
    }

    #[test]
    fn test_fence_pool_default() {
        let pool = FencePool::default();
        assert_eq!(pool.available(), 0);
    }

    #[test]
    fn test_timeline_registry() {
        let mut reg = TimelineRegistry::new();
        assert!(reg.is_empty());
        reg.get_or_create("render").signal(10);
        reg.get_or_create("compute").signal(20);
        assert_eq!(reg.len(), 2);
        let status = reg.status();
        assert_eq!(status.max_timeline_value, 20);
        assert_eq!(status.timeline_count, 2);
    }

    #[test]
    fn test_timeline_registry_get() {
        let mut reg = TimelineRegistry::new();
        reg.get_or_create("a").signal(5);
        assert!(reg.get("a").is_some());
        assert!(reg.get("b").is_none());
    }

    #[test]
    fn test_fence_signal_latency() {
        let mut fence = Fence::new(FenceId(0));
        assert!(fence.signal_latency().is_none());
        fence.signal();
        // Latency should be non-negative
        let latency = fence.signal_latency().expect("latency should be valid");
        assert!(latency.as_nanos() < 1_000_000_000); // less than 1 second
    }
}
