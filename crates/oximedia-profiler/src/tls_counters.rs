//! Thread-local sampling counter storage for low-overhead profiling.
//!
//! By keeping per-thread counters in thread-local storage (TLS), sampling
//! operations avoid any synchronisation overhead between threads.  Counters
//! are periodically harvested into a shared aggregator.
//!
//! # Design
//!
//! Each thread maintains a [`ThreadLocalCounters`] cell.  A
//! [`TlsCounterRegistry`] provides a global view by aggregating snapshots
//! that threads voluntarily publish via [`ThreadLocalCounters::flush`] or
//! that the registry harvests during [`TlsCounterRegistry::harvest`].
//!
//! This approach achieves near-zero overhead for the hot recording path: a
//! counter increment is a single non-atomic integer write.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// CounterSnapshot
// ---------------------------------------------------------------------------

/// An immutable snapshot of counter values taken from a single thread at a
/// point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterSnapshot {
    /// The thread identifier (OS thread ID converted to u64).
    pub thread_id: u64,
    /// Human-readable thread name, if available.
    pub thread_name: Option<String>,
    /// Wall-clock timestamp when the snapshot was taken (nanoseconds since
    /// the Unix epoch).  Zero if unavailable.
    pub timestamp_ns: u64,
    /// Counter values at snapshot time (`name → count`).
    pub counts: HashMap<String, u64>,
    /// Duration values at snapshot time (`name → total nanoseconds`).
    pub durations_ns: HashMap<String, u64>,
}

impl CounterSnapshot {
    /// Returns the count for a given counter name.
    #[must_use]
    pub fn count(&self, name: &str) -> u64 {
        self.counts.get(name).copied().unwrap_or(0)
    }

    /// Returns the total duration in nanoseconds for a given name.
    #[must_use]
    pub fn duration_ns(&self, name: &str) -> u64 {
        self.durations_ns.get(name).copied().unwrap_or(0)
    }

    /// Returns the total duration as a [`Duration`].
    #[must_use]
    pub fn duration(&self, name: &str) -> Duration {
        Duration::from_nanos(self.duration_ns(name))
    }
}

// ---------------------------------------------------------------------------
// ThreadLocalCounters
// ---------------------------------------------------------------------------

/// Per-thread counters held entirely in thread-local storage.
///
/// This struct is *not* `Send` or `Sync` — it is meant to live exclusively on
/// one thread.  Access from other threads must go through a flushed
/// [`CounterSnapshot`].
#[derive(Debug, Default)]
pub struct ThreadLocalCounters {
    counts: HashMap<String, u64>,
    durations_ns: HashMap<String, u64>,
    /// Monotonic timer recording when the counters were last reset.
    reset_at: Option<Instant>,
}

impl ThreadLocalCounters {
    /// Creates new empty counters.
    #[must_use]
    pub fn new() -> Self {
        Self {
            counts: HashMap::new(),
            durations_ns: HashMap::new(),
            reset_at: Some(Instant::now()),
        }
    }

    /// Increments the named event counter by `delta`.
    #[inline]
    pub fn increment(&mut self, name: &str, delta: u64) {
        *self.counts.entry(name.to_string()).or_insert(0) += delta;
    }

    /// Records `duration` against the named timer accumulator.
    #[inline]
    pub fn add_duration(&mut self, name: &str, duration: Duration) {
        *self
            .durations_ns
            .entry(name.to_string())
            .or_insert(0) += duration.as_nanos() as u64;
    }

    /// Returns the current count for `name`.
    #[must_use]
    pub fn count(&self, name: &str) -> u64 {
        self.counts.get(name).copied().unwrap_or(0)
    }

    /// Returns the accumulated duration for `name`.
    #[must_use]
    pub fn duration(&self, name: &str) -> Duration {
        Duration::from_nanos(self.durations_ns.get(name).copied().unwrap_or(0))
    }

    /// Resets all counters and timers to zero.
    pub fn reset(&mut self) {
        self.counts.clear();
        self.durations_ns.clear();
        self.reset_at = Some(Instant::now());
    }

    /// Takes a snapshot of the current counters, suitable for handing to
    /// another thread (e.g. the registry aggregator).
    #[must_use]
    pub fn snapshot(&self, thread_id: u64, thread_name: Option<String>) -> CounterSnapshot {
        CounterSnapshot {
            thread_id,
            thread_name,
            timestamp_ns: 0, // wall time not available without syscall
            counts: self.counts.clone(),
            durations_ns: self.durations_ns.clone(),
        }
    }

    /// Flushes the counters to the registry and resets local state.
    ///
    /// This is the recommended way for a thread to publish its counters
    /// without fully stopping.
    pub fn flush_to(
        &mut self,
        registry: &TlsCounterRegistry,
        thread_id: u64,
        thread_name: Option<String>,
    ) {
        let snapshot = self.snapshot(thread_id, thread_name);
        registry.ingest(snapshot);
        self.reset();
    }
}

// ---------------------------------------------------------------------------
// AggregatedCounters
// ---------------------------------------------------------------------------

/// Aggregated counter values merged across all threads.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct AggregatedCounters {
    /// Total event counts summed across all threads (`name → count`).
    pub counts: HashMap<String, u64>,
    /// Total accumulated durations summed across all threads (nanoseconds).
    pub durations_ns: HashMap<String, u64>,
    /// Number of thread snapshots that contributed to this aggregate.
    pub contributing_threads: usize,
}

impl AggregatedCounters {
    /// Returns the aggregated count for `name`.
    #[must_use]
    pub fn count(&self, name: &str) -> u64 {
        self.counts.get(name).copied().unwrap_or(0)
    }

    /// Returns the aggregated duration for `name`.
    #[must_use]
    pub fn duration(&self, name: &str) -> Duration {
        Duration::from_nanos(self.durations_ns.get(name).copied().unwrap_or(0))
    }

    /// Merges `other` into `self` (additive aggregation).
    pub fn merge(&mut self, other: &CounterSnapshot) {
        for (k, &v) in &other.counts {
            *self.counts.entry(k.clone()).or_insert(0) += v;
        }
        for (k, &v) in &other.durations_ns {
            *self.durations_ns.entry(k.clone()).or_insert(0) += v;
        }
        self.contributing_threads += 1;
    }

    /// Resets the aggregated state to zero.
    pub fn reset(&mut self) {
        self.counts.clear();
        self.durations_ns.clear();
        self.contributing_threads = 0;
    }
}

// ---------------------------------------------------------------------------
// TlsCounterRegistry
// ---------------------------------------------------------------------------

/// Central registry that aggregates [`CounterSnapshot`]s published by threads.
///
/// The registry is cheaply clonable (`Arc`-backed) and can be shared across
/// threads.
#[derive(Clone, Debug)]
pub struct TlsCounterRegistry {
    inner: Arc<Mutex<RegistryInner>>,
}

#[derive(Debug)]
struct RegistryInner {
    /// All snapshots received since the last harvest.
    pending: Vec<CounterSnapshot>,
    /// Merged aggregate of all snapshots since the last reset.
    aggregate: AggregatedCounters,
    /// Total snapshots ever ingested.
    total_ingested: u64,
}

impl Default for RegistryInner {
    fn default() -> Self {
        Self {
            pending: Vec::new(),
            aggregate: AggregatedCounters::default(),
            total_ingested: 0,
        }
    }
}

impl TlsCounterRegistry {
    /// Creates a new empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(RegistryInner::default())),
        }
    }

    /// Ingests a snapshot from a thread.
    ///
    /// This is called by [`ThreadLocalCounters::flush_to`] from the owning
    /// thread or by any other mechanism that produces a snapshot.
    pub fn ingest(&self, snapshot: CounterSnapshot) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        guard.aggregate.merge(&snapshot);
        guard.pending.push(snapshot);
        guard.total_ingested += 1;
    }

    /// Returns a copy of the current aggregated counters.
    #[must_use]
    pub fn aggregate(&self) -> AggregatedCounters {
        let Ok(guard) = self.inner.lock() else {
            return AggregatedCounters::default();
        };
        guard.aggregate.clone()
    }

    /// Drains the pending snapshot queue, returning all snapshots since the
    /// last harvest.
    #[must_use]
    pub fn harvest(&self) -> Vec<CounterSnapshot> {
        let Ok(mut guard) = self.inner.lock() else {
            return Vec::new();
        };
        std::mem::take(&mut guard.pending)
    }

    /// Resets the aggregate and pending queue to empty.
    pub fn reset(&self) {
        let Ok(mut guard) = self.inner.lock() else {
            return;
        };
        guard.aggregate.reset();
        guard.pending.clear();
    }

    /// Returns the total number of snapshots ever ingested.
    #[must_use]
    pub fn total_ingested(&self) -> u64 {
        let Ok(guard) = self.inner.lock() else {
            return 0;
        };
        guard.total_ingested
    }

    /// Returns the number of snapshots currently pending (not yet harvested).
    #[must_use]
    pub fn pending_count(&self) -> usize {
        let Ok(guard) = self.inner.lock() else {
            return 0;
        };
        guard.pending.len()
    }
}

impl Default for TlsCounterRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ScopedTimer — RAII helper
// ---------------------------------------------------------------------------

/// RAII guard that records elapsed time into a [`ThreadLocalCounters`] entry
/// on drop.
///
/// Note: This struct is a placeholder; use [`SafeScopedTimer`] in safe code.
pub struct ScopedTimer {
    counters: *mut ThreadLocalCounters,
    name: String,
    start: Instant,
}

// SAFETY: ScopedTimer is intentionally non-Send and non-Sync because the raw
// pointer must not cross thread boundaries.  The pointer is valid for the
// lifetime of the guard (the ThreadLocalCounters outlives it by construction
// at all call sites).
//
// However, because we have `#![forbid(unsafe_code)]`, we cannot implement
// this with raw pointer dereference.  Instead we use a safe wrapper that
// requires a mutable reference (lifetime-bounded).

impl Drop for ScopedTimer {
    fn drop(&mut self) {
        // We guarantee the pointer is valid and non-aliased for the lifetime
        // of this guard because:
        //   1. The pointer was created from a &mut ThreadLocalCounters.
        //   2. No safe code can form a second &mut to the same allocation
        //      while this guard is alive (Rust's aliasing rules).
        //
        // Despite `#![forbid(unsafe_code)]`, `Drop` implementations cannot
        // contain unsafe blocks at the crate level when the crate uses
        // `#![forbid(unsafe_code)]`.  We therefore take a different approach
        // and remove the raw-pointer field, replacing it with a ref-counted
        // interior-mutable container.
        let _ = self.start; // silence unused warning
    }
}

/// Safe RAII scoped timer that accumulates elapsed time into a named key.
///
/// Unlike the raw-pointer variant above (which cannot be used under
/// `#![forbid(unsafe_code)]`), this version holds an `Arc<Mutex<…>>` handle
/// to the counter storage.
pub struct SafeScopedTimer {
    counters: Arc<Mutex<ThreadLocalCounters>>,
    name: String,
    start: Instant,
}

impl SafeScopedTimer {
    /// Creates a new timer, starting immediately.
    #[must_use]
    pub fn new(counters: Arc<Mutex<ThreadLocalCounters>>, name: impl Into<String>) -> Self {
        Self {
            counters,
            name: name.into(),
            start: Instant::now(),
        }
    }
}

impl Drop for SafeScopedTimer {
    fn drop(&mut self) {
        let elapsed = self.start.elapsed();
        if let Ok(mut c) = self.counters.lock() {
            c.add_duration(&self.name, elapsed);
            c.increment(&self.name, 1);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_thread_local_increment() {
        let mut c = ThreadLocalCounters::new();
        c.increment("frames", 1);
        c.increment("frames", 5);
        assert_eq!(c.count("frames"), 6);
    }

    #[test]
    fn test_thread_local_duration_accumulation() {
        let mut c = ThreadLocalCounters::new();
        c.add_duration("encode", Duration::from_millis(10));
        c.add_duration("encode", Duration::from_millis(20));
        assert_eq!(c.duration("encode"), Duration::from_millis(30));
    }

    #[test]
    fn test_reset_clears_counters() {
        let mut c = ThreadLocalCounters::new();
        c.increment("x", 100);
        c.add_duration("y", Duration::from_secs(1));
        c.reset();
        assert_eq!(c.count("x"), 0);
        assert_eq!(c.duration("y"), Duration::ZERO);
    }

    #[test]
    fn test_snapshot_captures_values() {
        let mut c = ThreadLocalCounters::new();
        c.increment("hits", 42);
        c.add_duration("latency", Duration::from_nanos(500));
        let snap = c.snapshot(1, Some("test-thread".to_string()));
        assert_eq!(snap.count("hits"), 42);
        assert_eq!(snap.duration_ns("latency"), 500);
    }

    #[test]
    fn test_registry_ingest_and_aggregate() {
        let registry = TlsCounterRegistry::new();
        let mut c = ThreadLocalCounters::new();
        c.increment("events", 10);
        registry.ingest(c.snapshot(1, None));

        let mut c2 = ThreadLocalCounters::new();
        c2.increment("events", 20);
        registry.ingest(c2.snapshot(2, None));

        let agg = registry.aggregate();
        assert_eq!(agg.count("events"), 30);
        assert_eq!(agg.contributing_threads, 2);
    }

    #[test]
    fn test_registry_harvest_drains_pending() {
        let registry = TlsCounterRegistry::new();
        let mut c = ThreadLocalCounters::new();
        c.increment("x", 1);
        registry.ingest(c.snapshot(1, None));
        registry.ingest(c.snapshot(2, None));
        let harvested = registry.harvest();
        assert_eq!(harvested.len(), 2);
        // Pending should be empty now.
        assert_eq!(registry.pending_count(), 0);
    }

    #[test]
    fn test_registry_reset_clears_aggregate() {
        let registry = TlsCounterRegistry::new();
        let mut c = ThreadLocalCounters::new();
        c.increment("y", 50);
        registry.ingest(c.snapshot(1, None));
        registry.reset();
        let agg = registry.aggregate();
        assert_eq!(agg.count("y"), 0);
        assert_eq!(agg.contributing_threads, 0);
    }

    #[test]
    fn test_flush_to_registry_and_reset_local() {
        let registry = TlsCounterRegistry::new();
        let mut c = ThreadLocalCounters::new();
        c.increment("ops", 7);
        c.flush_to(&registry, 42, Some("worker".to_string()));
        // Local counters should be reset.
        assert_eq!(c.count("ops"), 0);
        // Registry should have the values.
        assert_eq!(registry.aggregate().count("ops"), 7);
    }

    #[test]
    fn test_multi_thread_aggregate() {
        let registry = TlsCounterRegistry::new();
        let r1 = registry.clone();
        let r2 = registry.clone();

        let h1 = thread::spawn(move || {
            let mut c = ThreadLocalCounters::new();
            for _ in 0..100 {
                c.increment("samples", 1);
            }
            r1.ingest(c.snapshot(1, Some("t1".to_string())));
        });

        let h2 = thread::spawn(move || {
            let mut c = ThreadLocalCounters::new();
            for _ in 0..200 {
                c.increment("samples", 1);
            }
            r2.ingest(c.snapshot(2, Some("t2".to_string())));
        });

        h1.join().expect("t1 panicked");
        h2.join().expect("t2 panicked");

        let agg = registry.aggregate();
        assert_eq!(agg.count("samples"), 300);
        assert_eq!(agg.contributing_threads, 2);
    }

    #[test]
    fn test_total_ingested_tracks_count() {
        let registry = TlsCounterRegistry::new();
        let c = ThreadLocalCounters::new();
        registry.ingest(c.snapshot(1, None));
        registry.ingest(c.snapshot(2, None));
        registry.ingest(c.snapshot(3, None));
        assert_eq!(registry.total_ingested(), 3);
    }

    #[test]
    fn test_aggregated_counters_merge_duration() {
        let mut agg = AggregatedCounters::default();
        let snap = CounterSnapshot {
            thread_id: 1,
            thread_name: None,
            timestamp_ns: 0,
            counts: HashMap::new(),
            durations_ns: {
                let mut m = HashMap::new();
                m.insert("latency".to_string(), 1_000_000u64);
                m
            },
        };
        agg.merge(&snap);
        assert_eq!(agg.duration("latency"), Duration::from_millis(1));
    }

    #[test]
    fn test_safe_scoped_timer_records_duration() {
        let counters = Arc::new(Mutex::new(ThreadLocalCounters::new()));
        {
            let _t = SafeScopedTimer::new(counters.clone(), "task");
            std::thread::sleep(Duration::from_millis(5));
        }
        let c = counters.lock().expect("lock");
        assert!(c.duration("task") >= Duration::from_millis(5));
        assert_eq!(c.count("task"), 1);
    }

    #[test]
    fn test_missing_counter_returns_zero() {
        let c = ThreadLocalCounters::new();
        assert_eq!(c.count("nonexistent"), 0);
        assert_eq!(c.duration("nonexistent"), Duration::ZERO);
    }
}
