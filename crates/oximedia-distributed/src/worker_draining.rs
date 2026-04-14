// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Worker draining — graceful shutdown for distributed encoding workers.
//!
//! Draining allows a worker to finish its current in-flight tasks without
//! accepting new ones, then transition to an `Idle` or `Offline` state.
//! This is essential for rolling upgrades and planned maintenance.

use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors produced by the draining subsystem.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DrainError {
    /// The worker ID is not registered.
    #[error("worker '{0}' not found")]
    WorkerNotFound(String),

    /// An operation was attempted on a worker that is already draining.
    #[error("worker '{0}' is already draining")]
    AlreadyDraining(String),

    /// An operation was attempted on a worker that has not started draining.
    #[error("worker '{0}' is not in draining state")]
    NotDraining(String),

    /// A task was submitted to a worker that is draining.
    #[error("worker '{0}' is draining and cannot accept new tasks")]
    WorkerDraining(String),
}

/// The operational state of a managed worker.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WorkerState {
    /// Worker is healthy and accepting tasks.
    Active,
    /// Worker is finishing in-flight tasks; no new tasks accepted.
    Draining,
    /// Worker has finished all tasks and is ready to be removed.
    Drained,
    /// Worker is offline / shut down.
    Offline,
}

/// Internal record kept per worker.
#[derive(Debug)]
struct WorkerRecord {
    state: WorkerState,
    in_flight: HashSet<String>,
    drain_started: Option<Instant>,
}

/// Manages graceful draining of workers in a distributed cluster.
///
/// Tasks are tracked per worker. When draining is requested the worker
/// stops accepting new tasks and a [`WorkerState::Drained`] transition
/// is triggered automatically once all in-flight tasks complete.
#[derive(Debug, Default)]
pub struct DrainManager {
    workers: HashMap<String, WorkerRecord>,
}

impl DrainManager {
    /// Create a new drain manager with no workers.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new worker in the `Active` state.
    pub fn register(&mut self, worker_id: impl Into<String>) {
        self.workers.insert(
            worker_id.into(),
            WorkerRecord {
                state: WorkerState::Active,
                in_flight: HashSet::new(),
                drain_started: None,
            },
        );
    }

    /// Deregister a worker.  Returns `true` if it was present.
    pub fn deregister(&mut self, worker_id: &str) -> bool {
        self.workers.remove(worker_id).is_some()
    }

    /// Assign a task to a worker.
    ///
    /// Fails with [`DrainError::WorkerDraining`] if the worker is draining or
    /// drained, and with [`DrainError::WorkerNotFound`] if unknown.
    pub fn assign_task(
        &mut self,
        worker_id: &str,
        task_id: impl Into<String>,
    ) -> Result<(), DrainError> {
        let record = self
            .workers
            .get_mut(worker_id)
            .ok_or_else(|| DrainError::WorkerNotFound(worker_id.to_owned()))?;

        if record.state != WorkerState::Active {
            return Err(DrainError::WorkerDraining(worker_id.to_owned()));
        }

        record.in_flight.insert(task_id.into());
        Ok(())
    }

    /// Mark a task as complete on a worker.
    ///
    /// If the worker is draining and has no remaining in-flight tasks,
    /// it transitions to [`WorkerState::Drained`] automatically.
    pub fn complete_task(&mut self, worker_id: &str, task_id: &str) -> Result<(), DrainError> {
        let record = self
            .workers
            .get_mut(worker_id)
            .ok_or_else(|| DrainError::WorkerNotFound(worker_id.to_owned()))?;

        record.in_flight.remove(task_id);

        // Auto-transition to Drained when drain is in progress and no tasks remain.
        if record.state == WorkerState::Draining && record.in_flight.is_empty() {
            record.state = WorkerState::Drained;
        }

        Ok(())
    }

    /// Initiate graceful draining of a worker.
    ///
    /// Returns [`DrainError::AlreadyDraining`] if already in progress.
    /// If the worker has no in-flight tasks it immediately becomes `Drained`.
    pub fn start_drain(&mut self, worker_id: &str) -> Result<(), DrainError> {
        let record = self
            .workers
            .get_mut(worker_id)
            .ok_or_else(|| DrainError::WorkerNotFound(worker_id.to_owned()))?;

        if record.state == WorkerState::Draining || record.state == WorkerState::Drained {
            return Err(DrainError::AlreadyDraining(worker_id.to_owned()));
        }

        record.drain_started = Some(Instant::now());

        if record.in_flight.is_empty() {
            record.state = WorkerState::Drained;
        } else {
            record.state = WorkerState::Draining;
        }

        Ok(())
    }

    /// Force-complete draining (e.g., after a timeout), discarding in-flight tasks.
    pub fn force_drain(&mut self, worker_id: &str) -> Result<Vec<String>, DrainError> {
        let record = self
            .workers
            .get_mut(worker_id)
            .ok_or_else(|| DrainError::WorkerNotFound(worker_id.to_owned()))?;

        let discarded: Vec<String> = record.in_flight.drain().collect();
        record.state = WorkerState::Drained;
        Ok(discarded)
    }

    /// Return the current state of a worker.
    pub fn state(&self, worker_id: &str) -> Option<&WorkerState> {
        self.workers.get(worker_id).map(|r| &r.state)
    }

    /// Return the number of in-flight tasks on a worker.
    pub fn in_flight_count(&self, worker_id: &str) -> Option<usize> {
        self.workers.get(worker_id).map(|r| r.in_flight.len())
    }

    /// Return the time elapsed since draining started, if applicable.
    pub fn drain_elapsed(&self, worker_id: &str) -> Option<Duration> {
        self.workers
            .get(worker_id)
            .and_then(|r| r.drain_started.map(|t| t.elapsed()))
    }

    /// Collect all workers whose drain has exceeded `timeout`.
    ///
    /// These workers can then be force-drained to unblock rolling upgrades.
    pub fn timed_out_drains(&self, timeout: Duration) -> Vec<String> {
        self.workers
            .iter()
            .filter_map(|(id, r)| {
                if r.state == WorkerState::Draining {
                    r.drain_started
                        .filter(|t| t.elapsed() >= timeout)
                        .map(|_| id.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return ids of all `Active` workers (candidates to receive new tasks).
    pub fn active_workers(&self) -> Vec<String> {
        let mut ids: Vec<String> = self
            .workers
            .iter()
            .filter(|(_, r)| r.state == WorkerState::Active)
            .map(|(id, _)| id.clone())
            .collect();
        ids.sort();
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manager_with_two_workers() -> DrainManager {
        let mut m = DrainManager::new();
        m.register("w1");
        m.register("w2");
        m
    }

    #[test]
    fn test_register_and_active_state() {
        let m = manager_with_two_workers();
        assert_eq!(m.state("w1"), Some(&WorkerState::Active));
    }

    #[test]
    fn test_assign_task_success() {
        let mut m = manager_with_two_workers();
        m.assign_task("w1", "t1").unwrap();
        assert_eq!(m.in_flight_count("w1"), Some(1));
    }

    #[test]
    fn test_assign_task_to_draining_fails() {
        let mut m = manager_with_two_workers();
        m.assign_task("w1", "t1").unwrap();
        m.start_drain("w1").unwrap();
        assert!(matches!(
            m.assign_task("w1", "t2").unwrap_err(),
            DrainError::WorkerDraining(_)
        ));
    }

    #[test]
    fn test_drain_with_no_tasks_immediately_drained() {
        let mut m = manager_with_two_workers();
        m.start_drain("w1").unwrap();
        assert_eq!(m.state("w1"), Some(&WorkerState::Drained));
    }

    #[test]
    fn test_drain_completes_when_tasks_finish() {
        let mut m = manager_with_two_workers();
        m.assign_task("w1", "t1").unwrap();
        m.assign_task("w1", "t2").unwrap();
        m.start_drain("w1").unwrap();
        assert_eq!(m.state("w1"), Some(&WorkerState::Draining));
        m.complete_task("w1", "t1").unwrap();
        assert_eq!(m.state("w1"), Some(&WorkerState::Draining));
        m.complete_task("w1", "t2").unwrap();
        assert_eq!(m.state("w1"), Some(&WorkerState::Drained));
    }

    #[test]
    fn test_already_draining_error() {
        let mut m = manager_with_two_workers();
        m.assign_task("w1", "t1").unwrap();
        m.start_drain("w1").unwrap();
        assert!(matches!(
            m.start_drain("w1").unwrap_err(),
            DrainError::AlreadyDraining(_)
        ));
    }

    #[test]
    fn test_force_drain_discards_tasks() {
        let mut m = manager_with_two_workers();
        m.assign_task("w1", "t1").unwrap();
        m.assign_task("w1", "t2").unwrap();
        m.start_drain("w1").unwrap();
        let discarded = m.force_drain("w1").unwrap();
        assert_eq!(discarded.len(), 2);
        assert_eq!(m.state("w1"), Some(&WorkerState::Drained));
        assert_eq!(m.in_flight_count("w1"), Some(0));
    }

    #[test]
    fn test_deregister() {
        let mut m = manager_with_two_workers();
        assert!(m.deregister("w1"));
        assert!(m.state("w1").is_none());
        assert!(!m.deregister("ghost"));
    }

    #[test]
    fn test_active_workers_excludes_draining() {
        let mut m = manager_with_two_workers();
        m.assign_task("w1", "t1").unwrap();
        m.start_drain("w1").unwrap();
        let active = m.active_workers();
        assert_eq!(active, vec!["w2"]);
    }

    #[test]
    fn test_unknown_worker_errors() {
        let mut m = manager_with_two_workers();
        assert!(matches!(
            m.assign_task("ghost", "t").unwrap_err(),
            DrainError::WorkerNotFound(_)
        ));
        assert!(matches!(
            m.start_drain("ghost").unwrap_err(),
            DrainError::WorkerNotFound(_)
        ));
    }
}
