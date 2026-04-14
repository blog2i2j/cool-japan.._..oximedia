//! Graceful shutdown for `BatchEngine`.
//!
//! Provides [`GracefulShutdown`] — a controller that coordinates an orderly
//! shutdown sequence with configurable drain timeout and force-kill escalation.
//! The progression is tracked through [`ShutdownPhase`] and observable at any
//! time via `phase()`.

#![allow(dead_code)]

use std::collections::HashSet;
use std::sync::atomic::{AtomicU8, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;

use crate::error::{BatchError, Result};
use crate::types::JobId;

// ---------------------------------------------------------------------------
// ShutdownPhase
// ---------------------------------------------------------------------------

/// The lifecycle phases of a graceful shutdown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum ShutdownPhase {
    /// The engine is running normally and accepting new jobs.
    Running = 0,
    /// Drain has started — no new jobs accepted, in-progress jobs are
    /// allowed to finish within the drain timeout window.
    Draining = 1,
    /// The drain timeout has elapsed without all jobs completing; remaining
    /// jobs are being forcibly stopped.
    ForceStop = 2,
    /// All jobs have either completed or been force-stopped.  The engine is
    /// fully shut down.
    Stopped = 3,
}

impl ShutdownPhase {
    fn from_u8(v: u8) -> Self {
        match v {
            0 => Self::Running,
            1 => Self::Draining,
            2 => Self::ForceStop,
            3 => Self::Stopped,
            _ => Self::Stopped,
        }
    }

    /// Returns `true` when the phase indicates the engine is no longer
    /// accepting new work.
    #[must_use]
    pub fn is_shutting_down(self) -> bool {
        !matches!(self, Self::Running)
    }

    /// Returns `true` when the shutdown sequence is fully complete.
    #[must_use]
    pub fn is_stopped(self) -> bool {
        matches!(self, Self::Stopped)
    }
}

impl std::fmt::Display for ShutdownPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Running => write!(f, "Running"),
            Self::Draining => write!(f, "Draining"),
            Self::ForceStop => write!(f, "ForceStop"),
            Self::Stopped => write!(f, "Stopped"),
        }
    }
}

// ---------------------------------------------------------------------------
// GracefulShutdownConfig
// ---------------------------------------------------------------------------

/// Configuration for graceful shutdown behaviour.
#[derive(Debug, Clone)]
pub struct GracefulShutdownConfig {
    /// Maximum time (in milliseconds) to wait for in-progress jobs to
    /// complete before escalating to `ForceStop`.
    pub drain_timeout_ms: u64,
    /// If `true`, the controller will transition to `ForceStop` and mark
    /// remaining jobs as force-killed after the drain timeout elapses.
    /// If `false`, the shutdown will simply transition to `Stopped` after
    /// the drain timeout, leaving stragglers in whatever state they are in.
    pub enable_force_kill: bool,
}

impl Default for GracefulShutdownConfig {
    fn default() -> Self {
        Self {
            drain_timeout_ms: 10_000,
            enable_force_kill: true,
        }
    }
}

// ---------------------------------------------------------------------------
// ShutdownResult
// ---------------------------------------------------------------------------

/// Outcome of a shutdown sequence.
#[derive(Debug, Clone)]
pub struct ShutdownResult {
    /// Final phase (always `Stopped`).
    pub final_phase: ShutdownPhase,
    /// Job IDs that completed normally during drain.
    pub drained_jobs: Vec<String>,
    /// Job IDs that were force-killed.
    pub force_killed_jobs: Vec<String>,
    /// Wall-clock duration of the shutdown sequence.
    pub elapsed: Duration,
}

// ---------------------------------------------------------------------------
// GracefulShutdown
// ---------------------------------------------------------------------------

/// Controller for a graceful shutdown lifecycle.
///
/// # Thread safety
///
/// The current phase is stored as an [`AtomicU8`] for lock-free reads.
/// The set of in-progress job IDs is protected by a [`RwLock`].
pub struct GracefulShutdown {
    phase: AtomicU8,
    config: GracefulShutdownConfig,
    in_progress: RwLock<HashSet<String>>,
    drained: RwLock<Vec<String>>,
    force_killed: RwLock<Vec<String>>,
    started_at: RwLock<Option<Instant>>,
}

impl GracefulShutdown {
    /// Create a new controller in the `Running` phase.
    #[must_use]
    pub fn new(config: GracefulShutdownConfig) -> Self {
        Self {
            phase: AtomicU8::new(ShutdownPhase::Running as u8),
            config,
            in_progress: RwLock::new(HashSet::new()),
            drained: RwLock::new(Vec::new()),
            force_killed: RwLock::new(Vec::new()),
            started_at: RwLock::new(None),
        }
    }

    /// Create a controller with the default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(GracefulShutdownConfig::default())
    }

    // -- Observation --------------------------------------------------------

    /// Current shutdown phase.
    #[must_use]
    pub fn phase(&self) -> ShutdownPhase {
        ShutdownPhase::from_u8(self.phase.load(Ordering::Acquire))
    }

    /// Number of jobs currently tracked as in-progress.
    #[must_use]
    pub fn in_progress_count(&self) -> usize {
        self.in_progress.read().len()
    }

    /// Returns `true` when new job submissions should be rejected.
    #[must_use]
    pub fn should_reject_new_jobs(&self) -> bool {
        self.phase().is_shutting_down()
    }

    /// The configured drain timeout.
    #[must_use]
    pub fn drain_timeout(&self) -> Duration {
        Duration::from_millis(self.config.drain_timeout_ms)
    }

    /// Returns elapsed time since drain started, or `None` if not draining.
    #[must_use]
    pub fn elapsed_since_drain(&self) -> Option<Duration> {
        self.started_at.read().map(|t| t.elapsed())
    }

    // -- Job tracking -------------------------------------------------------

    /// Register a job as in-progress.
    ///
    /// # Errors
    ///
    /// Returns [`BatchError::Cancelled`] if the engine is shutting down and
    /// should not accept new work.
    pub fn register_job(&self, job_id: &JobId) -> Result<()> {
        if self.should_reject_new_jobs() {
            return Err(BatchError::Cancelled(format!(
                "Cannot register job {} — engine is in {} phase",
                job_id.as_str(),
                self.phase()
            )));
        }
        self.in_progress
            .write()
            .insert(job_id.as_str().to_string());
        Ok(())
    }

    /// Mark a job as completed (removing it from the in-progress set).
    /// During the drain phase, the job ID is recorded in `drained_jobs`.
    pub fn complete_job(&self, job_id: &JobId) {
        let removed = self
            .in_progress
            .write()
            .remove(job_id.as_str());
        if removed && self.phase() == ShutdownPhase::Draining {
            self.drained.write().push(job_id.as_str().to_string());
        }
    }

    // -- Shutdown sequence --------------------------------------------------

    /// Initiate the shutdown sequence.
    ///
    /// Transitions from `Running` -> `Draining`.  This is a no-op if the
    /// engine is already shutting down.
    pub fn initiate_shutdown(&self) {
        let current = self.phase();
        if current == ShutdownPhase::Running {
            *self.started_at.write() = Some(Instant::now());
            self.set_phase(ShutdownPhase::Draining);
        }
    }

    /// Drive the shutdown state machine forward.
    ///
    /// Call this periodically (or in a loop) after `initiate_shutdown()`.
    /// It will:
    /// - stay in `Draining` while there are still in-progress jobs and the
    ///   drain timeout has not elapsed;
    /// - escalate to `ForceStop` (if enabled) when the drain timeout fires
    ///   and there are remaining jobs;
    /// - transition to `Stopped` when all jobs have been handled.
    ///
    /// Returns the current phase after the tick.
    pub fn tick(&self) -> ShutdownPhase {
        let current = self.phase();
        match current {
            ShutdownPhase::Running | ShutdownPhase::Stopped => current,
            ShutdownPhase::Draining => {
                if self.in_progress.read().is_empty() {
                    self.set_phase(ShutdownPhase::Stopped);
                    return ShutdownPhase::Stopped;
                }
                if self.drain_timeout_exceeded() {
                    if self.config.enable_force_kill {
                        self.force_kill_remaining();
                        self.set_phase(ShutdownPhase::ForceStop);
                        // ForceStop immediately transitions to Stopped.
                        self.set_phase(ShutdownPhase::Stopped);
                        ShutdownPhase::Stopped
                    } else {
                        self.set_phase(ShutdownPhase::Stopped);
                        ShutdownPhase::Stopped
                    }
                } else {
                    ShutdownPhase::Draining
                }
            }
            ShutdownPhase::ForceStop => {
                // ForceStop is transient; we should already be Stopped.
                self.set_phase(ShutdownPhase::Stopped);
                ShutdownPhase::Stopped
            }
        }
    }

    /// Run the full shutdown sequence synchronously, blocking until the
    /// engine reaches `Stopped`.
    ///
    /// # Returns
    ///
    /// A [`ShutdownResult`] summarising what happened.
    pub fn run_shutdown(&self) -> ShutdownResult {
        self.initiate_shutdown();
        let start = Instant::now();

        // Spin-tick until Stopped. We use a 1ms yield to avoid busy-waiting.
        loop {
            let phase = self.tick();
            if phase == ShutdownPhase::Stopped {
                break;
            }
            std::thread::sleep(Duration::from_millis(1));
        }

        ShutdownResult {
            final_phase: ShutdownPhase::Stopped,
            drained_jobs: self.drained.read().clone(),
            force_killed_jobs: self.force_killed.read().clone(),
            elapsed: start.elapsed(),
        }
    }

    // -- Internal helpers ---------------------------------------------------

    fn set_phase(&self, phase: ShutdownPhase) {
        self.phase.store(phase as u8, Ordering::Release);
    }

    fn drain_timeout_exceeded(&self) -> bool {
        self.started_at
            .read()
            .map_or(false, |t| t.elapsed() >= self.drain_timeout())
    }

    fn force_kill_remaining(&self) {
        let remaining: Vec<String> = self.in_progress.write().drain().collect();
        self.force_killed.write().extend(remaining);
    }
}

impl std::fmt::Debug for GracefulShutdown {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GracefulShutdown")
            .field("phase", &self.phase())
            .field("in_progress", &self.in_progress_count())
            .field("config", &self.config)
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Wrap in Arc for convenient sharing
// ---------------------------------------------------------------------------

/// A thread-safe handle to a [`GracefulShutdown`] controller.
pub type SharedGracefulShutdown = Arc<GracefulShutdown>;

/// Create a new [`SharedGracefulShutdown`].
#[must_use]
pub fn shared_graceful_shutdown(config: GracefulShutdownConfig) -> SharedGracefulShutdown {
    Arc::new(GracefulShutdown::new(config))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn jid(s: &str) -> JobId {
        JobId::from(s)
    }

    #[test]
    fn test_initial_phase_is_running() {
        let gs = GracefulShutdown::with_defaults();
        assert_eq!(gs.phase(), ShutdownPhase::Running);
        assert!(!gs.should_reject_new_jobs());
    }

    #[test]
    fn test_initiate_shutdown_transitions_to_draining() {
        let gs = GracefulShutdown::with_defaults();
        gs.initiate_shutdown();
        assert_eq!(gs.phase(), ShutdownPhase::Draining);
        assert!(gs.should_reject_new_jobs());
    }

    #[test]
    fn test_register_job_rejected_during_shutdown() {
        let gs = GracefulShutdown::with_defaults();
        gs.initiate_shutdown();
        let result = gs.register_job(&jid("late-job"));
        assert!(result.is_err());
    }

    #[test]
    fn test_register_and_complete_job() {
        let gs = GracefulShutdown::with_defaults();
        gs.register_job(&jid("job-1")).expect("should register");
        assert_eq!(gs.in_progress_count(), 1);
        gs.complete_job(&jid("job-1"));
        assert_eq!(gs.in_progress_count(), 0);
    }

    #[test]
    fn test_drain_completes_when_all_jobs_finish() {
        let gs = GracefulShutdown::new(GracefulShutdownConfig {
            drain_timeout_ms: 5_000,
            enable_force_kill: false,
        });
        gs.register_job(&jid("a")).expect("register a");
        gs.initiate_shutdown();
        // Still draining because job "a" is in progress.
        assert_eq!(gs.tick(), ShutdownPhase::Draining);
        // Complete the job.
        gs.complete_job(&jid("a"));
        // Now tick should transition to Stopped.
        assert_eq!(gs.tick(), ShutdownPhase::Stopped);
    }

    #[test]
    fn test_force_kill_after_drain_timeout() {
        let gs = GracefulShutdown::new(GracefulShutdownConfig {
            drain_timeout_ms: 0, // immediate timeout
            enable_force_kill: true,
        });
        gs.register_job(&jid("slow")).expect("register");
        gs.initiate_shutdown();
        // Because drain_timeout_ms is 0, tick should force-kill and stop.
        let phase = gs.tick();
        assert_eq!(phase, ShutdownPhase::Stopped);
        assert_eq!(gs.in_progress_count(), 0);
        assert!(gs.force_killed.read().contains(&"slow".to_string()));
    }

    #[test]
    fn test_shutdown_without_force_kill_on_timeout() {
        let gs = GracefulShutdown::new(GracefulShutdownConfig {
            drain_timeout_ms: 0,
            enable_force_kill: false,
        });
        gs.register_job(&jid("lingering")).expect("register");
        gs.initiate_shutdown();
        let phase = gs.tick();
        assert_eq!(phase, ShutdownPhase::Stopped);
        // Job was NOT moved to force_killed since force_kill is disabled.
        assert!(gs.force_killed.read().is_empty());
    }

    #[test]
    fn test_run_shutdown_no_jobs() {
        let gs = GracefulShutdown::new(GracefulShutdownConfig {
            drain_timeout_ms: 100,
            enable_force_kill: true,
        });
        let result = gs.run_shutdown();
        assert_eq!(result.final_phase, ShutdownPhase::Stopped);
        assert!(result.drained_jobs.is_empty());
        assert!(result.force_killed_jobs.is_empty());
    }

    #[test]
    fn test_shutdown_phase_display() {
        assert_eq!(ShutdownPhase::Running.to_string(), "Running");
        assert_eq!(ShutdownPhase::Draining.to_string(), "Draining");
        assert_eq!(ShutdownPhase::ForceStop.to_string(), "ForceStop");
        assert_eq!(ShutdownPhase::Stopped.to_string(), "Stopped");
    }

    #[test]
    fn test_shutdown_phase_from_u8_boundary() {
        assert_eq!(ShutdownPhase::from_u8(0), ShutdownPhase::Running);
        assert_eq!(ShutdownPhase::from_u8(3), ShutdownPhase::Stopped);
        assert_eq!(ShutdownPhase::from_u8(255), ShutdownPhase::Stopped);
    }

    #[test]
    fn test_multiple_initiate_shutdown_idempotent() {
        let gs = GracefulShutdown::with_defaults();
        gs.initiate_shutdown();
        gs.initiate_shutdown(); // should be no-op
        assert_eq!(gs.phase(), ShutdownPhase::Draining);
    }

    #[test]
    fn test_shared_graceful_shutdown() {
        let gs = shared_graceful_shutdown(GracefulShutdownConfig::default());
        let gs2 = Arc::clone(&gs);
        gs.register_job(&jid("shared-job")).expect("register");
        assert_eq!(gs2.in_progress_count(), 1);
        gs2.initiate_shutdown();
        assert!(gs.should_reject_new_jobs());
    }

    #[test]
    fn test_drained_jobs_tracked_during_drain_phase() {
        let gs = GracefulShutdown::new(GracefulShutdownConfig {
            drain_timeout_ms: 5_000,
            enable_force_kill: false,
        });
        gs.register_job(&jid("d1")).expect("register d1");
        gs.register_job(&jid("d2")).expect("register d2");
        gs.initiate_shutdown();
        gs.complete_job(&jid("d1"));
        assert!(gs.drained.read().contains(&"d1".to_string()));
        assert!(!gs.drained.read().contains(&"d2".to_string()));
    }

    #[test]
    fn test_elapsed_since_drain_none_before_shutdown() {
        let gs = GracefulShutdown::with_defaults();
        assert!(gs.elapsed_since_drain().is_none());
    }

    #[test]
    fn test_elapsed_since_drain_some_after_shutdown() {
        let gs = GracefulShutdown::with_defaults();
        gs.initiate_shutdown();
        let elapsed = gs.elapsed_since_drain();
        assert!(elapsed.is_some());
    }

    #[test]
    fn test_debug_impl() {
        let gs = GracefulShutdown::with_defaults();
        let dbg = format!("{gs:?}");
        assert!(dbg.contains("GracefulShutdown"));
        assert!(dbg.contains("Running"));
    }
}
