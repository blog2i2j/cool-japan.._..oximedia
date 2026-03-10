//! Async I/O abstractions: completion queue, polling, and cancellation tokens.
//!
//! Provides lightweight building blocks for async I/O patterns without
//! requiring additional external dependencies beyond `tokio`.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use std::collections::VecDeque;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

// ---------------------------------------------------------------------------
// Cancellation token
// ---------------------------------------------------------------------------

/// A lightweight cancellation token that can be shared across tasks.
#[derive(Debug, Clone)]
pub struct CancellationToken {
    cancelled: Arc<AtomicBool>,
}

impl CancellationToken {
    #[must_use]
    pub fn new() -> Self {
        Self {
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    /// Signal cancellation.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::SeqCst);
    }

    /// Returns `true` if cancellation has been signalled.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::SeqCst)
    }

    /// Create a child token that shares the same cancel flag.
    #[must_use]
    pub fn child(&self) -> Self {
        Self {
            cancelled: Arc::clone(&self.cancelled),
        }
    }
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// I/O operation descriptor
// ---------------------------------------------------------------------------

/// Unique identifier for an async I/O operation.
pub type OpId = u64;

/// The type of an async I/O operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpKind {
    Read,
    Write,
    Flush,
    Seek,
}

/// Status of a queued I/O operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OpStatus {
    Pending,
    InFlight,
    Completed { bytes: usize },
    Failed { reason: String },
    Cancelled,
}

/// Descriptor for a single async I/O operation.
#[derive(Debug, Clone)]
pub struct IoOp {
    pub id: OpId,
    pub kind: OpKind,
    pub offset: u64,
    pub length: usize,
    pub status: OpStatus,
    pub cancel: CancellationToken,
}

impl IoOp {
    #[must_use]
    pub fn new_read(id: OpId, offset: u64, length: usize) -> Self {
        Self {
            id,
            kind: OpKind::Read,
            offset,
            length,
            status: OpStatus::Pending,
            cancel: CancellationToken::new(),
        }
    }

    #[must_use]
    pub fn new_write(id: OpId, offset: u64, length: usize) -> Self {
        Self {
            id,
            kind: OpKind::Write,
            offset,
            length,
            status: OpStatus::Pending,
            cancel: CancellationToken::new(),
        }
    }

    #[must_use]
    pub fn is_pending(&self) -> bool {
        self.status == OpStatus::Pending
    }

    #[must_use]
    pub fn is_complete(&self) -> bool {
        matches!(self.status, OpStatus::Completed { .. })
    }

    #[must_use]
    pub fn is_failed(&self) -> bool {
        matches!(self.status, OpStatus::Failed { .. })
    }

    pub fn cancel(&mut self) {
        self.cancel.cancel();
        if self.status == OpStatus::Pending || self.status == OpStatus::InFlight {
            self.status = OpStatus::Cancelled;
        }
    }
}

// ---------------------------------------------------------------------------
// Completion queue
// ---------------------------------------------------------------------------

/// A simple synchronous completion queue for I/O operations.
/// In production this would integrate with an OS completion mechanism
/// (`io_uring`, IOCP, kqueue).  Here we model the data structures.
#[derive(Debug, Default)]
pub struct CompletionQueue {
    pending: VecDeque<IoOp>,
    completed: VecDeque<IoOp>,
    next_id: OpId,
}

impl CompletionQueue {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn next_id(&mut self) -> OpId {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Submit an operation to the queue.
    pub fn submit(&mut self, op: IoOp) {
        self.pending.push_back(op);
    }

    /// Simulate completion: move all pending ops to completed, marking them
    /// as succeeded with `bytes_per_op` bytes transferred.
    pub fn drain_pending(&mut self, bytes_per_op: usize) {
        while let Some(mut op) = self.pending.pop_front() {
            if op.cancel.is_cancelled() {
                op.status = OpStatus::Cancelled;
            } else {
                op.status = OpStatus::Completed {
                    bytes: bytes_per_op,
                };
            }
            self.completed.push_back(op);
        }
    }

    /// Pop the next completed operation.
    pub fn pop_completed(&mut self) -> Option<IoOp> {
        self.completed.pop_front()
    }

    #[must_use]
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    #[must_use]
    pub fn completed_count(&self) -> usize {
        self.completed.len()
    }
}

// ---------------------------------------------------------------------------
// Polling state machine
// ---------------------------------------------------------------------------

/// State of a polling async operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PollState {
    /// Operation not yet started.
    Idle,
    /// Waiting for I/O to become ready.
    Waiting,
    /// Data is available; the operation can proceed.
    Ready,
    /// Operation is complete.
    Done,
    /// Operation failed.
    Error(String),
}

/// Simulates an async poller over a byte buffer.
#[derive(Debug)]
pub struct BytePoller {
    data: Vec<u8>,
    pos: usize,
    state: PollState,
    poll_count: u32,
    /// After this many polls, the poller becomes Ready.
    ready_after: u32,
}

impl BytePoller {
    #[must_use]
    pub fn new(data: Vec<u8>, ready_after: u32) -> Self {
        Self {
            data,
            pos: 0,
            state: PollState::Idle,
            poll_count: 0,
            ready_after,
        }
    }

    /// Advance the state machine one step.
    pub fn poll(&mut self) -> &PollState {
        self.poll_count += 1;
        match &self.state {
            PollState::Idle => {
                self.state = PollState::Waiting;
            }
            PollState::Waiting => {
                if self.poll_count >= self.ready_after {
                    self.state = PollState::Ready;
                }
            }
            PollState::Ready => {
                self.state = PollState::Done;
            }
            PollState::Done | PollState::Error(_) => {}
        }
        &self.state
    }

    /// Read bytes from the underlying data (only valid in Ready/Done state).
    pub fn read_chunk(&mut self, n: usize) -> Option<&[u8]> {
        if !matches!(self.state, PollState::Ready | PollState::Done) {
            return None;
        }
        let end = (self.pos + n).min(self.data.len());
        let slice = &self.data[self.pos..end];
        self.pos = end;
        Some(slice)
    }

    #[must_use]
    pub fn state(&self) -> &PollState {
        &self.state
    }

    #[must_use]
    pub fn poll_count(&self) -> u32 {
        self.poll_count
    }
}

// ---------------------------------------------------------------------------
// Timeout helper (synchronous mock)
// ---------------------------------------------------------------------------

/// A simple deadline tracker (synchronous; does not use `tokio` timers).
#[derive(Debug, Clone)]
pub struct Deadline {
    pub timeout: Duration,
    elapsed: Duration,
}

impl Deadline {
    #[must_use]
    pub fn new(timeout: Duration) -> Self {
        Self {
            timeout,
            elapsed: Duration::ZERO,
        }
    }

    /// Advance the simulated clock by `step`.  Returns `true` if the deadline
    /// has NOT yet been exceeded.
    pub fn tick(&mut self, step: Duration) -> bool {
        self.elapsed += step;
        self.elapsed < self.timeout
    }

    #[must_use]
    pub fn is_expired(&self) -> bool {
        self.elapsed >= self.timeout
    }

    #[must_use]
    pub fn remaining(&self) -> Duration {
        self.timeout.saturating_sub(self.elapsed)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cancellation_token_default_not_cancelled() {
        let t = CancellationToken::new();
        assert!(!t.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_cancel() {
        let t = CancellationToken::new();
        t.cancel();
        assert!(t.is_cancelled());
    }

    #[test]
    fn test_cancellation_token_child_shares_state() {
        let parent = CancellationToken::new();
        let child = parent.child();
        parent.cancel();
        assert!(child.is_cancelled());
    }

    #[test]
    fn test_io_op_initial_state() {
        let op = IoOp::new_read(1, 0, 4096);
        assert!(op.is_pending());
        assert!(!op.is_complete());
        assert_eq!(op.kind, OpKind::Read);
    }

    #[test]
    fn test_io_op_cancel() {
        let mut op = IoOp::new_write(2, 0, 1024);
        op.cancel();
        assert_eq!(op.status, OpStatus::Cancelled);
    }

    #[test]
    fn test_completion_queue_submit_and_drain() {
        let mut cq = CompletionQueue::new();
        let id = cq.next_id();
        cq.submit(IoOp::new_read(id, 0, 512));
        assert_eq!(cq.pending_count(), 1);
        cq.drain_pending(512);
        assert_eq!(cq.pending_count(), 0);
        assert_eq!(cq.completed_count(), 1);
    }

    #[test]
    fn test_completion_queue_cancelled_op_stays_cancelled() {
        let mut cq = CompletionQueue::new();
        let mut op = IoOp::new_read(10, 0, 256);
        op.cancel();
        cq.submit(op);
        cq.drain_pending(256);
        let done = cq
            .pop_completed()
            .expect("pop_completed should return item");
        assert_eq!(done.status, OpStatus::Cancelled);
    }

    #[test]
    fn test_completion_queue_pop_order_fifo() {
        let mut cq = CompletionQueue::new();
        cq.submit(IoOp::new_read(1, 0, 1));
        cq.submit(IoOp::new_read(2, 1, 1));
        cq.drain_pending(1);
        assert_eq!(
            cq.pop_completed()
                .expect("pop_completed should return item")
                .id,
            1
        );
        assert_eq!(
            cq.pop_completed()
                .expect("pop_completed should return item")
                .id,
            2
        );
    }

    #[test]
    fn test_byte_poller_reaches_ready_state() {
        let mut p = BytePoller::new(vec![1, 2, 3], 3);
        p.poll(); // Idle → Waiting
        p.poll(); // stays Waiting (count=2 < 3)
        p.poll(); // count==3 → Ready
        assert_eq!(*p.state(), PollState::Ready);
    }

    #[test]
    fn test_byte_poller_read_chunk() {
        let mut p = BytePoller::new(vec![10, 20, 30], 1);
        p.poll(); // Idle → Waiting
        p.poll(); // Waiting → Ready (count=2 >= 1)
        let chunk = p.read_chunk(2).expect("read_chunk should succeed");
        assert_eq!(chunk, &[10, 20]);
    }

    #[test]
    fn test_byte_poller_no_read_when_waiting() {
        let mut p = BytePoller::new(vec![1], 99);
        p.poll();
        assert!(p.read_chunk(1).is_none());
    }

    #[test]
    fn test_deadline_not_expired_initially() {
        let d = Deadline::new(Duration::from_secs(5));
        assert!(!d.is_expired());
    }

    #[test]
    fn test_deadline_expires_after_ticks() {
        let mut d = Deadline::new(Duration::from_secs(2));
        assert!(d.tick(Duration::from_secs(1)));
        assert!(!d.tick(Duration::from_secs(1))); // now elapsed == timeout
        assert!(d.is_expired());
    }

    #[test]
    fn test_deadline_remaining() {
        let mut d = Deadline::new(Duration::from_secs(10));
        d.tick(Duration::from_secs(3));
        assert_eq!(d.remaining(), Duration::from_secs(7));
    }
}
