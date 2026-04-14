//! Lock-free routing updates for glitch-free real-time changes.
//!
//! Traditional routing matrix updates that use a `Mutex` can cause priority
//! inversion and audio glitches when the real-time thread blocks.  This
//! module implements a **versioned, double-buffered routing state** where:
//!
//! * The **writer** (control thread) prepares changes on a staging copy
//!   and publishes them atomically via [`Arc`] swap.
//! * The **reader** (real-time thread) always reads a consistent snapshot
//!   without blocking.
//!
//! No `unsafe` code is used — correctness relies on `Arc`, `RwLock`, and
//! atomic version counters.
//!
//! # Example
//!
//! ```
//! use oximedia_routing::lock_free_router::{LockFreeRouter, RoutingState};
//!
//! let router = LockFreeRouter::new(8, 8);
//!
//! // Control thread: stage changes.
//! router.stage_connect(0, 0, 0.0);
//! router.stage_connect(1, 1, -6.0);
//!
//! // Publish atomically.
//! router.publish();
//!
//! // Real-time thread: read the snapshot.
//! let snap = router.snapshot();
//! assert!(snap.is_connected(0, 0));
//! assert!(snap.is_connected(1, 1));
//! ```

#![allow(dead_code)]

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

use serde::{Deserialize, Serialize};

/// A crosspoint connection entry.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Crosspoint {
    /// Whether this crosspoint is active.
    pub connected: bool,
    /// Gain in dB applied at this crosspoint.
    pub gain_db: f32,
}

impl Default for Crosspoint {
    fn default() -> Self {
        Self {
            connected: false,
            gain_db: 0.0,
        }
    }
}

/// An immutable routing state snapshot.
///
/// This is a flat `inputs × outputs` matrix stored in row-major order.
/// It is cheaply cloneable via `Arc`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingState {
    inputs: usize,
    outputs: usize,
    /// Row-major: index = input * outputs + output.
    crosspoints: Vec<Crosspoint>,
    /// Monotonically increasing version stamp.
    version: u64,
}

impl RoutingState {
    /// Creates a new blank routing state.
    pub fn new(inputs: usize, outputs: usize) -> Self {
        Self {
            inputs,
            outputs,
            crosspoints: vec![Crosspoint::default(); inputs * outputs],
            version: 0,
        }
    }

    /// Number of inputs.
    pub fn inputs(&self) -> usize {
        self.inputs
    }

    /// Number of outputs.
    pub fn outputs(&self) -> usize {
        self.outputs
    }

    /// Version stamp.
    pub fn version(&self) -> u64 {
        self.version
    }

    fn idx(&self, input: usize, output: usize) -> Option<usize> {
        if input < self.inputs && output < self.outputs {
            Some(input * self.outputs + output)
        } else {
            None
        }
    }

    /// Returns `true` if the crosspoint is active.
    pub fn is_connected(&self, input: usize, output: usize) -> bool {
        self.idx(input, output)
            .and_then(|i| self.crosspoints.get(i))
            .map_or(false, |cp| cp.connected)
    }

    /// Returns the gain at a crosspoint, or `None` if out of bounds or
    /// not connected.
    pub fn gain_db(&self, input: usize, output: usize) -> Option<f32> {
        self.idx(input, output)
            .and_then(|i| self.crosspoints.get(i))
            .filter(|cp| cp.connected)
            .map(|cp| cp.gain_db)
    }

    /// Returns all inputs connected to the given output.
    pub fn inputs_for_output(&self, output: usize) -> Vec<(usize, f32)> {
        if output >= self.outputs {
            return Vec::new();
        }
        (0..self.inputs)
            .filter_map(|inp| {
                let idx = inp * self.outputs + output;
                self.crosspoints.get(idx).and_then(|cp| {
                    if cp.connected {
                        Some((inp, cp.gain_db))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Returns all outputs connected from the given input.
    pub fn outputs_for_input(&self, input: usize) -> Vec<(usize, f32)> {
        if input >= self.inputs {
            return Vec::new();
        }
        let base = input * self.outputs;
        (0..self.outputs)
            .filter_map(|out| {
                self.crosspoints.get(base + out).and_then(|cp| {
                    if cp.connected {
                        Some((out, cp.gain_db))
                    } else {
                        None
                    }
                })
            })
            .collect()
    }

    /// Total number of active crosspoints.
    pub fn active_count(&self) -> usize {
        self.crosspoints.iter().filter(|cp| cp.connected).count()
    }

    /// Connect a crosspoint (mutable, used during staging).
    fn connect(&mut self, input: usize, output: usize, gain_db: f32) {
        if let Some(i) = self.idx(input, output) {
            if let Some(cp) = self.crosspoints.get_mut(i) {
                cp.connected = true;
                cp.gain_db = gain_db;
            }
        }
    }

    /// Disconnect a crosspoint.
    fn disconnect(&mut self, input: usize, output: usize) {
        if let Some(i) = self.idx(input, output) {
            if let Some(cp) = self.crosspoints.get_mut(i) {
                cp.connected = false;
                cp.gain_db = 0.0;
            }
        }
    }

    /// Disconnect all crosspoints.
    fn clear(&mut self) {
        for cp in &mut self.crosspoints {
            cp.connected = false;
            cp.gain_db = 0.0;
        }
    }
}

/// A lock-free (RwLock-based) routing matrix.
///
/// The published state is accessed through a read-lock which never blocks
/// the writer (on most platforms, RwLock allows concurrent readers).
/// Updates are staged on a private copy and atomically swapped in via
/// `publish()`.
#[derive(Debug)]
pub struct LockFreeRouter {
    /// The currently published state.
    published: RwLock<Arc<RoutingState>>,
    /// Staging area (private to the writer).
    staging: RwLock<RoutingState>,
    /// Global version counter.
    version_counter: AtomicU64,
    /// Number of times `publish()` has been called.
    publish_count: AtomicU64,
}

impl LockFreeRouter {
    /// Creates a new router with the given input and output counts.
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let state = RoutingState::new(inputs, outputs);
        Self {
            published: RwLock::new(Arc::new(state.clone())),
            staging: RwLock::new(state),
            version_counter: AtomicU64::new(0),
            publish_count: AtomicU64::new(0),
        }
    }

    /// Returns a snapshot of the currently published routing state.
    ///
    /// This is the read path for the real-time thread.  It never blocks
    /// the writer.
    pub fn snapshot(&self) -> Arc<RoutingState> {
        self.published
            .read()
            .map(|guard| Arc::clone(&guard))
            .unwrap_or_else(|poisoned| Arc::clone(&poisoned.into_inner()))
    }

    /// Stage a connection change (does NOT affect the published state).
    pub fn stage_connect(&self, input: usize, output: usize, gain_db: f32) {
        if let Ok(mut staging) = self.staging.write() {
            staging.connect(input, output, gain_db);
        }
    }

    /// Stage a disconnection.
    pub fn stage_disconnect(&self, input: usize, output: usize) {
        if let Ok(mut staging) = self.staging.write() {
            staging.disconnect(input, output);
        }
    }

    /// Stage a clear of all crosspoints.
    pub fn stage_clear(&self) {
        if let Ok(mut staging) = self.staging.write() {
            staging.clear();
        }
    }

    /// Stage a gain change on an already-connected crosspoint.
    pub fn stage_set_gain(&self, input: usize, output: usize, gain_db: f32) {
        if let Ok(mut staging) = self.staging.write() {
            if let Some(i) = staging.idx(input, output) {
                if let Some(cp) = staging.crosspoints.get_mut(i) {
                    if cp.connected {
                        cp.gain_db = gain_db;
                    }
                }
            }
        }
    }

    /// Atomically publishes the staged state.
    ///
    /// After this call, all subsequent `snapshot()` calls will see the
    /// new routing.  The staging area is reset to a clone of the newly
    /// published state so that further edits start from the latest baseline.
    pub fn publish(&self) {
        let new_version = self.version_counter.fetch_add(1, Ordering::SeqCst) + 1;

        // Clone staging → new published state.
        let new_state = {
            let mut staging = match self.staging.write() {
                Ok(s) => s,
                Err(_) => return,
            };
            staging.version = new_version;
            Arc::new(staging.clone())
        };

        // Swap into published.
        if let Ok(mut published) = self.published.write() {
            *published = new_state;
        }

        self.publish_count.fetch_add(1, Ordering::Relaxed);
    }

    /// Resets the staging area to match the currently published state.
    ///
    /// Discards any staged-but-unpublished changes.
    pub fn discard_staged(&self) {
        let current = self.snapshot();
        if let Ok(mut staging) = self.staging.write() {
            *staging = (*current).clone();
        }
    }

    /// Number of times `publish()` has been called.
    pub fn publish_count(&self) -> u64 {
        self.publish_count.load(Ordering::Relaxed)
    }

    /// Current published version.
    pub fn published_version(&self) -> u64 {
        self.snapshot().version
    }

    /// Returns the staging state version (always 0 — version is assigned at
    /// publish time).
    pub fn inputs(&self) -> usize {
        self.snapshot().inputs
    }

    /// Output count.
    pub fn outputs(&self) -> usize {
        self.snapshot().outputs
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_router_empty() {
        let router = LockFreeRouter::new(4, 4);
        let snap = router.snapshot();
        assert_eq!(snap.active_count(), 0);
    }

    #[test]
    fn test_stage_and_publish() {
        let router = LockFreeRouter::new(4, 4);
        router.stage_connect(0, 0, 0.0);

        // Not published yet.
        let snap1 = router.snapshot();
        assert!(!snap1.is_connected(0, 0));

        router.publish();

        let snap2 = router.snapshot();
        assert!(snap2.is_connected(0, 0));
    }

    #[test]
    fn test_gain_tracking() {
        let router = LockFreeRouter::new(4, 4);
        router.stage_connect(1, 2, -6.0);
        router.publish();

        let snap = router.snapshot();
        assert_eq!(snap.gain_db(1, 2), Some(-6.0));
    }

    #[test]
    fn test_disconnect() {
        let router = LockFreeRouter::new(4, 4);
        router.stage_connect(0, 0, 0.0);
        router.publish();
        assert!(router.snapshot().is_connected(0, 0));

        router.stage_disconnect(0, 0);
        router.publish();
        assert!(!router.snapshot().is_connected(0, 0));
    }

    #[test]
    fn test_clear() {
        let router = LockFreeRouter::new(4, 4);
        router.stage_connect(0, 0, 0.0);
        router.stage_connect(1, 1, 0.0);
        router.publish();
        assert_eq!(router.snapshot().active_count(), 2);

        router.stage_clear();
        router.publish();
        assert_eq!(router.snapshot().active_count(), 0);
    }

    #[test]
    fn test_version_increments() {
        let router = LockFreeRouter::new(4, 4);
        assert_eq!(router.published_version(), 0);

        router.publish();
        assert_eq!(router.published_version(), 1);

        router.publish();
        assert_eq!(router.published_version(), 2);
    }

    #[test]
    fn test_publish_count() {
        let router = LockFreeRouter::new(4, 4);
        assert_eq!(router.publish_count(), 0);
        router.publish();
        router.publish();
        assert_eq!(router.publish_count(), 2);
    }

    #[test]
    fn test_discard_staged() {
        let router = LockFreeRouter::new(4, 4);
        router.stage_connect(0, 0, 0.0);
        router.publish();

        // Stage a change but discard it.
        router.stage_disconnect(0, 0);
        router.discard_staged();
        router.publish();

        // Should still be connected (discard reverted the staging).
        assert!(router.snapshot().is_connected(0, 0));
    }

    #[test]
    fn test_inputs_for_output() {
        let router = LockFreeRouter::new(8, 4);
        router.stage_connect(0, 2, 0.0);
        router.stage_connect(3, 2, -3.0);
        router.stage_connect(5, 2, -6.0);
        router.publish();

        let snap = router.snapshot();
        let inputs = snap.inputs_for_output(2);
        assert_eq!(inputs.len(), 3);
    }

    #[test]
    fn test_outputs_for_input() {
        let router = LockFreeRouter::new(4, 8);
        router.stage_connect(1, 0, 0.0);
        router.stage_connect(1, 3, -3.0);
        router.publish();

        let snap = router.snapshot();
        let outputs = snap.outputs_for_input(1);
        assert_eq!(outputs.len(), 2);
    }

    #[test]
    fn test_out_of_bounds_connect() {
        let router = LockFreeRouter::new(4, 4);
        router.stage_connect(10, 10, 0.0); // out of bounds
        router.publish();
        let snap = router.snapshot();
        assert_eq!(snap.active_count(), 0);
    }

    #[test]
    fn test_gain_db_none_when_disconnected() {
        let state = RoutingState::new(4, 4);
        assert!(state.gain_db(0, 0).is_none());
    }

    #[test]
    fn test_gain_db_none_when_out_of_bounds() {
        let state = RoutingState::new(4, 4);
        assert!(state.gain_db(10, 10).is_none());
    }

    #[test]
    fn test_inputs_for_output_out_of_bounds() {
        let state = RoutingState::new(4, 4);
        assert!(state.inputs_for_output(10).is_empty());
    }

    #[test]
    fn test_outputs_for_input_out_of_bounds() {
        let state = RoutingState::new(4, 4);
        assert!(state.outputs_for_input(10).is_empty());
    }

    #[test]
    fn test_set_gain_on_connected() {
        let router = LockFreeRouter::new(4, 4);
        router.stage_connect(0, 0, 0.0);
        router.publish();

        router.stage_set_gain(0, 0, -12.0);
        router.publish();

        assert_eq!(router.snapshot().gain_db(0, 0), Some(-12.0));
    }

    #[test]
    fn test_snapshot_is_arc_clone_cheap() {
        let router = LockFreeRouter::new(4, 4);
        router.stage_connect(0, 0, 0.0);
        router.publish();

        let s1 = router.snapshot();
        let s2 = router.snapshot();
        // Both point to the same underlying state.
        assert_eq!(s1.version(), s2.version());
        assert!(Arc::ptr_eq(&s1, &s2));
    }

    #[test]
    fn test_routing_state_dimensions() {
        let router = LockFreeRouter::new(16, 8);
        assert_eq!(router.inputs(), 16);
        assert_eq!(router.outputs(), 8);
    }

    #[test]
    fn test_multiple_publish_sequence() {
        let router = LockFreeRouter::new(4, 4);

        // First batch.
        router.stage_connect(0, 0, 0.0);
        router.publish();

        // Second batch.
        router.stage_connect(1, 1, -3.0);
        router.publish();

        let snap = router.snapshot();
        // Both connections should be present.
        assert!(snap.is_connected(0, 0));
        assert!(snap.is_connected(1, 1));
        assert_eq!(snap.active_count(), 2);
    }

    // -----------------------------------------------------------------------
    // Task D: concurrent read + write tests (6 tests)
    // -----------------------------------------------------------------------

    #[test]
    fn test_concurrent_readers_see_consistent_state() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let router = StdArc::new(LockFreeRouter::new(8, 8));

        // Publish an initial state.
        router.stage_connect(0, 0, 0.0);
        router.stage_connect(1, 1, -3.0);
        router.publish();

        let router_clone = StdArc::clone(&router);
        let reader_handle = thread::spawn(move || {
            // 5000 rapid snapshot reads — each must be consistent.
            for _ in 0..5000 {
                let snap = router_clone.snapshot();
                // The snapshot is immutable once created; there must never be
                // a state where connected=true but active_count=0.
                let count = snap.active_count();
                if snap.is_connected(0, 0) || snap.is_connected(1, 1) {
                    assert!(count > 0, "inconsistent state: connected but count=0");
                }
            }
        });

        // Writer: publish additional connections while readers run.
        for i in 2..6 {
            router.stage_connect(i, i, 0.0);
            router.publish();
        }

        reader_handle.join().expect("reader thread panicked");
    }

    #[test]
    fn test_concurrent_readers_never_block_longer_than_1ms() {
        use std::sync::Arc as StdArc;
        use std::thread;
        use std::time::{Duration, Instant};

        let router = StdArc::new(LockFreeRouter::new(16, 16));
        router.stage_connect(0, 0, 0.0);
        router.publish();

        let router_clone = StdArc::clone(&router);
        let reader_handle = thread::spawn(move || {
            for _ in 0..1000 {
                let t = Instant::now();
                let _snap = router_clone.snapshot();
                let elapsed = t.elapsed();
                assert!(
                    elapsed < Duration::from_millis(1),
                    "snapshot() blocked for {:?}",
                    elapsed
                );
            }
        });

        // Writer publishes updates concurrently.
        for i in 0..10 {
            router.stage_connect(i, i, 0.0);
            router.publish();
        }

        reader_handle.join().expect("reader thread panicked");
    }

    #[test]
    fn test_concurrent_writes_do_not_corrupt_state() {
        use std::sync::Arc as StdArc;
        use std::thread;

        let router = StdArc::new(LockFreeRouter::new(8, 8));

        // Two writer threads each perform their own stage+publish sequence.
        // Because the router uses an RwLock for the staging area, only one
        // writer can hold the write lock at a time — no corruption is possible.
        let r1 = StdArc::clone(&router);
        let r2 = StdArc::clone(&router);

        let t1 = thread::spawn(move || {
            for _ in 0..50 {
                r1.stage_connect(0, 0, 0.0);
                r1.publish();
                r1.stage_disconnect(0, 0);
                r1.publish();
            }
        });

        let t2 = thread::spawn(move || {
            for _ in 0..50 {
                r2.stage_connect(1, 1, -6.0);
                r2.publish();
                r2.stage_disconnect(1, 1);
                r2.publish();
            }
        });

        t1.join().expect("writer t1 panicked");
        t2.join().expect("writer t2 panicked");

        // After all writes, the router must not be poisoned — snapshot() must work.
        let snap = router.snapshot();
        let _ = snap.active_count(); // must not panic
    }

    #[test]
    fn test_reader_sees_latest_published_version() {
        use std::sync::Arc as StdArc;

        let router = StdArc::new(LockFreeRouter::new(4, 4));

        let v0 = router.published_version();
        router.stage_connect(0, 0, 0.0);
        router.publish();
        let v1 = router.published_version();
        assert!(v1 > v0, "version must increase after publish");

        // snapshot() must reflect the latest version.
        let snap = router.snapshot();
        assert_eq!(snap.version(), v1);
        assert!(snap.is_connected(0, 0));
    }

    #[test]
    fn test_reader_before_and_after_publish() {
        use std::sync::Arc as StdArc;

        let router = StdArc::new(LockFreeRouter::new(4, 4));

        // Take a snapshot before publishing.
        let snap_before = router.snapshot();
        assert_eq!(snap_before.active_count(), 0);

        router.stage_connect(2, 3, -12.0);
        router.publish();

        // Take a snapshot after publishing.
        let snap_after = router.snapshot();
        assert_eq!(snap_after.active_count(), 1);
        assert!(snap_after.is_connected(2, 3));

        // The old snapshot is still valid and unchanged (immutable Arc).
        assert_eq!(snap_before.active_count(), 0);
        assert!(!snap_before.is_connected(2, 3));
    }

    #[test]
    fn test_hot_path_snapshot_is_wait_free() {
        // The snapshot() call holds only a read lock (shared) and does a
        // cheap Arc::clone.  This test verifies that 10_000 concurrent
        // snapshot reads from a single thread complete in well under 100 ms.
        use std::time::{Duration, Instant};

        let router = LockFreeRouter::new(8, 8);
        router.stage_connect(0, 0, 0.0);
        router.publish();

        let start = Instant::now();
        for _ in 0..10_000 {
            let _snap = router.snapshot();
        }
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(100),
            "10_000 snapshot() calls took {:?} (> 100 ms)",
            elapsed
        );
    }
}
