// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! `JobId`, `JobRegistry`, and `PriorityQueue<T>` for the jobs system.

#![allow(dead_code)]
#![allow(clippy::missing_panics_doc)]

use crate::job::{Job, JobStatus};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::{Arc, Mutex};
use uuid::Uuid;

// ---------------------------------------------------------------------------
// JobId
// ---------------------------------------------------------------------------

/// A newtype wrapper around [`Uuid`] that uniquely identifies a job.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct JobId(Uuid);

impl JobId {
    /// Create a new random job ID.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Wrap an existing UUID.
    #[must_use]
    pub const fn from_uuid(uuid: Uuid) -> Self {
        Self(uuid)
    }

    /// Get the underlying UUID.
    #[must_use]
    pub const fn as_uuid(&self) -> &Uuid {
        &self.0
    }

    /// Get the inner UUID by value.
    #[must_use]
    pub const fn into_uuid(self) -> Uuid {
        self.0
    }
}

impl Default for JobId {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Display for JobId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<Uuid> for JobId {
    fn from(uuid: Uuid) -> Self {
        Self(uuid)
    }
}

impl From<JobId> for Uuid {
    fn from(id: JobId) -> Self {
        id.0
    }
}

// ---------------------------------------------------------------------------
// JobRegistry
// ---------------------------------------------------------------------------

/// Thread-safe, in-memory registry of [`Job`]s.
///
/// Uses `Arc<Mutex<_>>` internally so it can be cloned cheaply and shared
/// across threads.
#[derive(Debug, Clone, Default)]
pub struct JobRegistry {
    inner: Arc<Mutex<HashMap<Uuid, Job>>>,
}

impl JobRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a job.  Returns the job's ID.
    #[must_use]
    pub fn register(&self, job: Job) -> Uuid {
        let id = job.id;
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(id, job);
        id
    }

    /// Look up a job by ID.
    #[must_use]
    pub fn get(&self, id: Uuid) -> Option<Job> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .get(&id)
            .cloned()
    }

    /// Remove a job. Returns the removed job if it existed.
    #[must_use]
    pub fn remove(&self, id: Uuid) -> Option<Job> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .remove(&id)
    }

    /// Update a job (replaces existing entry). Returns `false` if not found.
    #[must_use]
    pub fn update(&self, job: Job) -> bool {
        let mut guard = self.inner.lock().unwrap_or_else(|e| e.into_inner());
        if let std::collections::hash_map::Entry::Occupied(mut e) = guard.entry(job.id) {
            e.insert(job);
            true
        } else {
            false
        }
    }

    /// Return all jobs with the given status.
    #[must_use]
    pub fn by_status(&self, status: JobStatus) -> Vec<Job> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .filter(|j| j.status == status)
            .cloned()
            .collect()
    }

    /// Total number of registered jobs.
    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.lock().unwrap_or_else(|e| e.into_inner()).len()
    }

    /// Returns `true` if no jobs are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return all registered jobs.
    #[must_use]
    pub fn all(&self) -> Vec<Job> {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .values()
            .cloned()
            .collect()
    }

    /// Return `true` if the job with the given ID is registered.
    #[must_use]
    pub fn contains(&self, id: Uuid) -> bool {
        self.inner
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .contains_key(&id)
    }
}

// ---------------------------------------------------------------------------
// PriorityQueue<T>
// ---------------------------------------------------------------------------

/// An item in the priority queue with an associated numeric priority.
///
/// Lower `priority` value means **higher** urgency (min-heap semantics).
#[derive(Debug, Clone)]
struct PqItem<T> {
    priority: i64,
    seq: u64, // tie-breaker: earlier insertion wins
    value: T,
}

impl<T> PartialEq for PqItem<T> {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.seq == other.seq
    }
}

impl<T> Eq for PqItem<T> {}

impl<T> PartialOrd for PqItem<T> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Reversed so that `BinaryHeap` (max-heap) behaves like a min-heap.
impl<T> Ord for PqItem<T> {
    fn cmp(&self, other: &Self) -> Ordering {
        // Flip the comparison on priority (lower priority int = higher urgency).
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.seq.cmp(&self.seq))
    }
}

/// A min-heap priority queue.
///
/// `push` accepts a numeric priority; lower values dequeue first.
pub struct PriorityQueue<T> {
    heap: BinaryHeap<PqItem<T>>,
    counter: u64,
}

impl<T> PriorityQueue<T> {
    /// Create an empty priority queue.
    #[must_use]
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
            counter: 0,
        }
    }

    /// Push an item with a given numeric priority.
    ///
    /// Lower `priority` values dequeue first.
    pub fn push(&mut self, value: T, priority: i64) {
        let seq = self.counter;
        self.counter += 1;
        self.heap.push(PqItem {
            priority,
            seq,
            value,
        });
    }

    /// Pop the highest-priority (lowest numeric) item.
    pub fn pop(&mut self) -> Option<T> {
        self.heap.pop().map(|item| item.value)
    }

    /// Peek at the highest-priority item without removing it.
    #[must_use]
    pub fn peek(&self) -> Option<&T> {
        self.heap.peek().map(|item| &item.value)
    }

    /// Return the number of items in the queue.
    #[must_use]
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Returns `true` if the queue is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

impl<T> Default for PriorityQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::job::{JobPayload, Priority, TranscodeParams};

    fn make_job(name: &str) -> Job {
        let params = TranscodeParams {
            input: "in.mp4".to_string(),
            output: "out.mp4".to_string(),
            video_codec: "h264".to_string(),
            audio_codec: "aac".to_string(),
            video_bitrate: 4_000_000,
            audio_bitrate: 128_000,
            resolution: None,
            framerate: None,
            preset: "fast".to_string(),
            hw_accel: None,
        };
        Job::new(
            name.to_string(),
            Priority::Normal,
            JobPayload::Transcode(params),
        )
    }

    // --- JobId ---

    #[test]
    fn test_job_id_uniqueness() {
        let a = JobId::new();
        let b = JobId::new();
        assert_ne!(a, b);
    }

    #[test]
    fn test_job_id_from_uuid() {
        let uuid = Uuid::new_v4();
        let id = JobId::from_uuid(uuid);
        assert_eq!(*id.as_uuid(), uuid);
    }

    #[test]
    fn test_job_id_into_uuid() {
        let id = JobId::new();
        let uuid: Uuid = id.into();
        assert_eq!(uuid, *id.as_uuid()); // id still valid after copy
    }

    #[test]
    fn test_job_id_display() {
        let id = JobId::new();
        let s = id.to_string();
        assert_eq!(s.len(), 36); // canonical UUID format
    }

    // --- JobRegistry ---

    #[test]
    fn test_registry_register_and_get() {
        let reg = JobRegistry::new();
        let job = make_job("job-1");
        let id = job.id;

        let _ = reg.register(job);
        assert!(reg.get(id).is_some());
        assert_eq!(reg.len(), 1);
    }

    #[test]
    fn test_registry_remove() {
        let reg = JobRegistry::new();
        let job = make_job("job-2");
        let id = job.id;
        let _ = reg.register(job);

        let removed = reg.remove(id);
        assert!(removed.is_some());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn test_registry_update() {
        let reg = JobRegistry::new();
        let mut job = make_job("job-3");
        let id = job.id;
        let _ = reg.register(job.clone());

        job.status = JobStatus::Running;
        assert!(reg.update(job));
        assert_eq!(
            reg.get(id).expect("get should succeed").status,
            JobStatus::Running
        );
    }

    #[test]
    fn test_registry_by_status() {
        let reg = JobRegistry::new();
        let job1 = make_job("j1");
        let mut job2 = make_job("j2");
        job2.status = JobStatus::Completed;

        let _ = reg.register(job1);
        let _ = reg.register(job2);

        let pending = reg.by_status(JobStatus::Pending);
        assert_eq!(pending.len(), 1);
        let completed = reg.by_status(JobStatus::Completed);
        assert_eq!(completed.len(), 1);
    }

    #[test]
    fn test_registry_contains() {
        let reg = JobRegistry::new();
        let job = make_job("j-check");
        let id = job.id;
        let _ = reg.register(job);
        assert!(reg.contains(id));
        assert!(!reg.contains(Uuid::new_v4()));
    }

    #[test]
    fn test_registry_clone_shared() {
        let reg = JobRegistry::new();
        let reg2 = reg.clone();

        let job = make_job("shared");
        let id = job.id;
        let _ = reg.register(job);

        // Both handles see the same data.
        assert!(reg2.get(id).is_some());
    }

    // --- PriorityQueue ---

    #[test]
    fn test_pq_min_heap_ordering() {
        let mut pq: PriorityQueue<&str> = PriorityQueue::new();
        pq.push("low", 10);
        pq.push("high", 1);
        pq.push("medium", 5);

        assert_eq!(pq.pop(), Some("high"));
        assert_eq!(pq.pop(), Some("medium"));
        assert_eq!(pq.pop(), Some("low"));
    }

    #[test]
    fn test_pq_fifo_tie_breaking() {
        let mut pq: PriorityQueue<u32> = PriorityQueue::new();
        pq.push(1u32, 0);
        pq.push(2u32, 0);
        pq.push(3u32, 0);

        // Same priority -> FIFO order.
        assert_eq!(pq.pop(), Some(1));
        assert_eq!(pq.pop(), Some(2));
        assert_eq!(pq.pop(), Some(3));
    }

    #[test]
    fn test_pq_peek() {
        let mut pq: PriorityQueue<i32> = PriorityQueue::new();
        pq.push(42, 5);
        pq.push(7, 1);

        assert_eq!(pq.peek(), Some(&7));
        assert_eq!(pq.len(), 2); // peek does not remove
    }

    #[test]
    fn test_pq_empty() {
        let mut pq: PriorityQueue<String> = PriorityQueue::new();
        assert!(pq.is_empty());
        assert_eq!(pq.pop(), None);
        assert_eq!(pq.peek(), None);
    }

    #[test]
    fn test_pq_negative_priority() {
        let mut pq: PriorityQueue<&str> = PriorityQueue::new();
        pq.push("urgent", -100);
        pq.push("normal", 0);
        pq.push("low", 50);

        assert_eq!(pq.pop(), Some("urgent"));
    }
}
