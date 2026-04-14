//! Graceful shutdown coordination for the encoding farm.
//!
//! A [`ShutdownCoordinator`] tracks in-flight tasks and drives a two-phase
//! shutdown: first stop accepting new work (drain), then wait for all existing
//! work to finish before hard-stopping.

use parking_lot::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Lifecycle phase of a graceful shutdown.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ShutdownPhase {
    /// Normal operation — new jobs are accepted.
    Running,
    /// Drain requested — no new jobs accepted; existing work continues.
    DrainRequested,
    /// Waiting for all in-flight tasks to finish.
    WaitingForCompletion,
    /// All work finished; the process may exit.
    Terminated,
}

/// Coordinates graceful shutdown across the coordinator and workers.
///
/// Workers must call [`ShutdownCoordinator::register_task`] before starting
/// each task and [`ShutdownCoordinator::complete_task`] when done.  When the
/// operator calls [`ShutdownCoordinator::begin_drain`], a `watch` channel
/// notifies all workers to stop pulling new work.
pub struct ShutdownCoordinator {
    phase: Mutex<ShutdownPhase>,
    in_flight: AtomicU64,
    drain_tx: Option<tokio::sync::watch::Sender<bool>>,
}

impl ShutdownCoordinator {
    /// Create a new coordinator in `Running` state.
    ///
    /// Returns the coordinator and a `watch::Receiver<bool>` that transitions
    /// to `true` when draining begins.
    #[must_use]
    pub fn new() -> (Self, tokio::sync::watch::Receiver<bool>) {
        let (tx, rx) = tokio::sync::watch::channel(false);
        let sc = Self {
            phase: Mutex::new(ShutdownPhase::Running),
            in_flight: AtomicU64::new(0),
            drain_tx: Some(tx),
        };
        (sc, rx)
    }

    /// Transition to `DrainRequested` and notify all watch receivers.
    ///
    /// After this call no new jobs should be accepted; existing in-flight tasks
    /// are allowed to finish naturally.
    pub fn begin_drain(&self) {
        *self.phase.lock() = ShutdownPhase::DrainRequested;
        if let Some(tx) = &self.drain_tx {
            let _ = tx.send(true);
        }
    }

    /// Transition to `WaitingForCompletion`.
    ///
    /// Call this after `begin_drain` once the job-acceptance layer has stopped.
    pub fn begin_waiting(&self) {
        *self.phase.lock() = ShutdownPhase::WaitingForCompletion;
    }

    /// Transition to `Terminated`.
    ///
    /// Call this once `wait_for_drain` returns `true`.
    pub fn terminate(&self) {
        *self.phase.lock() = ShutdownPhase::Terminated;
    }

    /// Increment the in-flight task counter.
    ///
    /// Must be called before a task starts executing.
    pub fn register_task(&self) {
        self.in_flight.fetch_add(1, Ordering::AcqRel);
    }

    /// Decrement the in-flight task counter.
    ///
    /// Must be called exactly once when a task finishes (success or failure).
    pub fn complete_task(&self) {
        self.in_flight.fetch_sub(1, Ordering::AcqRel);
    }

    /// Current number of tasks that have been registered but not completed.
    #[must_use]
    pub fn in_flight_count(&self) -> u64 {
        self.in_flight.load(Ordering::Acquire)
    }

    /// Current shutdown phase.
    #[must_use]
    pub fn phase(&self) -> ShutdownPhase {
        self.phase.lock().clone()
    }

    /// Whether new jobs are still being accepted.
    #[must_use]
    pub fn is_accepting_jobs(&self) -> bool {
        matches!(*self.phase.lock(), ShutdownPhase::Running)
    }

    /// Wait until all in-flight tasks complete or `timeout` elapses.
    ///
    /// Returns `true` if all tasks drained within the timeout, `false`
    /// otherwise (forced-stop scenario).
    pub async fn wait_for_drain(&self, timeout: Duration) -> bool {
        let deadline = tokio::time::Instant::now() + timeout;
        loop {
            if self.in_flight_count() == 0 {
                return true;
            }
            if tokio::time::Instant::now() >= deadline {
                return false;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    }
}

impl Default for ShutdownCoordinator {
    fn default() -> Self {
        let (sc, _) = Self::new();
        sc
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ── Phase transitions ─────────────────────────────────────────────────────

    #[test]
    fn initial_phase_is_running() {
        let (sc, _rx) = ShutdownCoordinator::new();
        assert_eq!(sc.phase(), ShutdownPhase::Running);
    }

    #[test]
    fn begin_drain_changes_phase_to_drain_requested() {
        let (sc, _rx) = ShutdownCoordinator::new();
        sc.begin_drain();
        assert_eq!(sc.phase(), ShutdownPhase::DrainRequested);
    }

    #[test]
    fn begin_waiting_changes_phase() {
        let (sc, _rx) = ShutdownCoordinator::new();
        sc.begin_drain();
        sc.begin_waiting();
        assert_eq!(sc.phase(), ShutdownPhase::WaitingForCompletion);
    }

    #[test]
    fn terminate_changes_phase_to_terminated() {
        let (sc, _rx) = ShutdownCoordinator::new();
        sc.begin_drain();
        sc.begin_waiting();
        sc.terminate();
        assert_eq!(sc.phase(), ShutdownPhase::Terminated);
    }

    // ── In-flight counter ─────────────────────────────────────────────────────

    #[test]
    fn register_and_complete_task_adjust_counter() {
        let (sc, _rx) = ShutdownCoordinator::new();
        assert_eq!(sc.in_flight_count(), 0);
        sc.register_task();
        sc.register_task();
        assert_eq!(sc.in_flight_count(), 2);
        sc.complete_task();
        assert_eq!(sc.in_flight_count(), 1);
        sc.complete_task();
        assert_eq!(sc.in_flight_count(), 0);
    }

    // ── wait_for_drain ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn wait_for_drain_returns_true_when_already_drained() {
        let (sc, _rx) = ShutdownCoordinator::new();
        // No tasks registered — counter is already 0
        let result = sc.wait_for_drain(Duration::from_millis(100)).await;
        assert!(result, "should return true when no tasks are in-flight");
    }

    #[tokio::test]
    async fn wait_for_drain_returns_true_after_tasks_complete() {
        let (sc, _rx) = ShutdownCoordinator::new();
        let sc = Arc::new(sc);

        sc.register_task();
        sc.register_task();

        // Complete tasks after a short delay in a background task
        let sc2 = sc.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(30)).await;
            sc2.complete_task();
            sc2.complete_task();
        });

        let result = sc.wait_for_drain(Duration::from_millis(500)).await;
        assert!(result, "should drain before timeout");
    }

    #[tokio::test]
    async fn wait_for_drain_returns_false_on_timeout() {
        let (sc, _rx) = ShutdownCoordinator::new();
        // Register a task that is never completed
        sc.register_task();

        // Very short timeout — should expire
        let result = sc.wait_for_drain(Duration::from_millis(50)).await;
        assert!(!result, "should return false when timeout elapses");
    }

    // ── Watch channel notification ────────────────────────────────────────────

    #[tokio::test]
    async fn begin_drain_notifies_watch_receivers() {
        let (sc, mut rx) = ShutdownCoordinator::new();

        // Initially false
        assert!(!*rx.borrow());

        sc.begin_drain();

        // Wait for the notification to propagate
        rx.changed().await.unwrap();
        assert!(*rx.borrow(), "watch should flip to true after begin_drain");
    }

    #[test]
    fn is_accepting_jobs_false_after_drain() {
        let (sc, _rx) = ShutdownCoordinator::new();
        assert!(sc.is_accepting_jobs());
        sc.begin_drain();
        assert!(!sc.is_accepting_jobs());
    }
}
