//! Chaos testing utilities for the encoding farm.
//!
//! `ChaosTester` injects random worker disconnects during simulated job
//! processing to verify that the coordinator correctly handles transient
//! network failures and worker restarts.

use parking_lot::Mutex;
use rand::{rngs::SmallRng, RngExt, SeedableRng};
use std::sync::atomic::{AtomicU64, Ordering};

/// Configuration for the chaos injector.
#[derive(Debug, Clone)]
pub struct ChaosConfig {
    /// Probability (in [0.0, 1.0]) that any given job triggers a disconnect.
    pub disconnect_probability: f32,
    /// Milliseconds to wait before reconnecting after a simulated disconnect.
    pub reconnect_delay_ms: u64,
    /// RNG seed for deterministic replay.
    pub seed: u64,
}

impl Default for ChaosConfig {
    fn default() -> Self {
        Self {
            disconnect_probability: 0.1,
            reconnect_delay_ms: 100,
            seed: 42,
        }
    }
}

/// Events emitted during a simulated worker lifecycle.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerEvent {
    /// The worker began executing a job.
    JobStarted { worker_id: u64 },
    /// The worker completed a job without interruption.
    JobCompleted { worker_id: u64 },
    /// The worker lost network connectivity mid-job.
    Disconnected { worker_id: u64 },
    /// The worker re-established connectivity and resumed.
    Reconnected { worker_id: u64 },
}

/// Chaos injector that randomly disconnects workers during simulated jobs.
///
/// All randomness is seeded so tests are fully deterministic when given the
/// same [`ChaosConfig::seed`].
pub struct ChaosTester {
    config: ChaosConfig,
    rng: Mutex<SmallRng>,
    disconnect_count: AtomicU64,
    reconnect_count: AtomicU64,
}

impl ChaosTester {
    /// Create a new `ChaosTester` with the given configuration.
    #[must_use]
    pub fn new(config: ChaosConfig) -> Self {
        Self {
            rng: Mutex::new(SmallRng::seed_from_u64(config.seed)),
            config,
            disconnect_count: AtomicU64::new(0),
            reconnect_count: AtomicU64::new(0),
        }
    }

    /// Decide whether the current event should trigger a disconnect.
    ///
    /// Increments `disconnect_count` when returning `true`.
    pub fn should_disconnect(&self) -> bool {
        let roll: f32 = self.rng.lock().random();
        if roll < self.config.disconnect_probability {
            self.disconnect_count.fetch_add(1, Ordering::Relaxed);
            true
        } else {
            false
        }
    }

    /// Simulate `job_count` jobs for a single worker and return the event log.
    ///
    /// For each job the worker either:
    /// - completes normally (`JobStarted` → `JobCompleted`), or
    /// - gets disconnected and reconnects (`JobStarted` → `Disconnected` → `Reconnected`).
    pub fn simulate_worker_lifecycle(&self, worker_id: u64, job_count: u32) -> Vec<WorkerEvent> {
        let mut events = Vec::with_capacity((job_count * 2) as usize);
        for _ in 0..job_count {
            events.push(WorkerEvent::JobStarted { worker_id });
            if self.should_disconnect() {
                events.push(WorkerEvent::Disconnected { worker_id });
                events.push(WorkerEvent::Reconnected { worker_id });
                self.reconnect_count.fetch_add(1, Ordering::Relaxed);
            } else {
                events.push(WorkerEvent::JobCompleted { worker_id });
            }
        }
        events
    }

    /// Total number of disconnects that have been injected so far.
    #[must_use]
    pub fn disconnect_count(&self) -> u64 {
        self.disconnect_count.load(Ordering::Relaxed)
    }

    /// Total number of reconnects that have been recorded so far.
    #[must_use]
    pub fn reconnect_count(&self) -> u64 {
        self.reconnect_count.load(Ordering::Relaxed)
    }

    /// Probability configured for disconnects.
    #[must_use]
    pub fn disconnect_probability(&self) -> f32 {
        self.config.disconnect_probability
    }

    /// Configured reconnect delay.
    #[must_use]
    pub fn reconnect_delay_ms(&self) -> u64 {
        self.config.reconnect_delay_ms
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Probability extremes ─────────────────────────────────────────────────

    #[test]
    fn probability_one_every_job_disconnects() {
        let tester = ChaosTester::new(ChaosConfig {
            disconnect_probability: 1.0,
            reconnect_delay_ms: 0,
            seed: 0,
        });
        let events = tester.simulate_worker_lifecycle(1, 10);
        // With p=1.0 every job generates Disconnected + Reconnected
        let disconnects = events
            .iter()
            .filter(|e| matches!(e, WorkerEvent::Disconnected { .. }))
            .count();
        assert_eq!(disconnects, 10, "all 10 jobs should disconnect");
    }

    #[test]
    fn probability_zero_no_disconnects() {
        let tester = ChaosTester::new(ChaosConfig {
            disconnect_probability: 0.0,
            reconnect_delay_ms: 0,
            seed: 0,
        });
        let events = tester.simulate_worker_lifecycle(2, 20);
        let disconnects = events
            .iter()
            .filter(|e| matches!(e, WorkerEvent::Disconnected { .. }))
            .count();
        assert_eq!(disconnects, 0, "no disconnects expected with p=0");
    }

    #[test]
    fn seeded_rng_is_deterministic() {
        let config = ChaosConfig {
            disconnect_probability: 0.5,
            reconnect_delay_ms: 50,
            seed: 12345,
        };
        let tester1 = ChaosTester::new(config.clone());
        let tester2 = ChaosTester::new(config);
        let events1 = tester1.simulate_worker_lifecycle(1, 20);
        let events2 = tester2.simulate_worker_lifecycle(1, 20);
        assert_eq!(events1, events2, "same seed must produce identical events");
    }

    #[test]
    fn simulate_returns_correct_event_count() {
        let tester = ChaosTester::new(ChaosConfig {
            disconnect_probability: 0.0,
            ..Default::default()
        });
        // p=0 → exactly 2 events per job (Started + Completed)
        let events = tester.simulate_worker_lifecycle(7, 5);
        assert_eq!(events.len(), 10);
    }

    #[test]
    fn simulate_with_all_disconnects_event_count() {
        let tester = ChaosTester::new(ChaosConfig {
            disconnect_probability: 1.0,
            ..Default::default()
        });
        // p=1 → 3 events per job (Started + Disconnected + Reconnected)
        let events = tester.simulate_worker_lifecycle(7, 5);
        assert_eq!(events.len(), 15);
    }

    #[test]
    fn ten_workers_hundred_jobs_disconnect_count_within_range() {
        // With seed=42 and p=0.1, 10 workers × 100 jobs = 1000 attempts.
        // Expected disconnects ≈ 100; accept within ±3σ (σ=√90 ≈ 9.5 → 3σ ≈ 28.5).
        let tester = ChaosTester::new(ChaosConfig {
            disconnect_probability: 0.1,
            reconnect_delay_ms: 0,
            seed: 42,
        });
        for w in 0..10u64 {
            tester.simulate_worker_lifecycle(w, 100);
        }
        let dc = tester.disconnect_count();
        assert!(
            dc <= 130,
            "disconnect count {dc} exceeds upper bound 130 (expected ≈100)"
        );
        assert!(
            dc >= 70,
            "disconnect count {dc} below lower bound 70 (expected ≈100)"
        );
    }

    #[test]
    fn reconnect_count_equals_disconnect_count() {
        let tester = ChaosTester::new(ChaosConfig {
            disconnect_probability: 0.5,
            reconnect_delay_ms: 0,
            seed: 99,
        });
        tester.simulate_worker_lifecycle(1, 100);
        assert_eq!(
            tester.disconnect_count(),
            tester.reconnect_count(),
            "every disconnect must be followed by a reconnect"
        );
    }

    #[test]
    fn job_started_always_first_per_job() {
        let tester = ChaosTester::new(ChaosConfig {
            disconnect_probability: 0.5,
            reconnect_delay_ms: 0,
            seed: 777,
        });
        let events = tester.simulate_worker_lifecycle(3, 50);
        let mut idx = 0usize;
        while idx < events.len() {
            assert!(
                matches!(events[idx], WorkerEvent::JobStarted { .. }),
                "event at position {idx} should be JobStarted, got {:?}",
                events[idx]
            );
            // Advance past the job's events (either 2 or 3 events per job)
            idx += 1; // skip JobStarted
            match &events[idx] {
                WorkerEvent::JobCompleted { .. } => idx += 1,
                WorkerEvent::Disconnected { .. } => {
                    idx += 1; // skip Disconnected
                    assert!(matches!(events[idx], WorkerEvent::Reconnected { .. }));
                    idx += 1; // skip Reconnected
                }
                other => panic!("unexpected event after JobStarted: {other:?}"),
            }
        }
    }

    #[test]
    fn worker_event_is_clone_and_partialeq() {
        let e1 = WorkerEvent::JobStarted { worker_id: 1 };
        let e2 = e1.clone();
        assert_eq!(e1, e2);

        let e3 = WorkerEvent::Disconnected { worker_id: 2 };
        assert_ne!(e1, e3);
    }
}
