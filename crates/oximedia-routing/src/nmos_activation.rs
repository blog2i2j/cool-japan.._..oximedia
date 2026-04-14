//! NMOS IS-05 activation scheduling and management.
//!
//! NMOS IS-05 Connection Management defines three activation modes for
//! making routing changes:
//!
//! - **Immediate**: The connection activates as soon as the request is received.
//! - **Scheduled (absolute)**: The connection activates at a specific TAI
//!   timestamp (PTP-synchronized).
//! - **Scheduled (relative)**: The connection activates after a specified
//!   offset from the time the request was received.
//!
//! [`ActivationScheduler`] maintains a queue of pending activations and
//! fires them when their scheduled time arrives (caller polls
//! [`tick`][ActivationScheduler::tick] with a monotonic timestamp in
//! nanoseconds).
//!
//! ## Example
//!
//! ```rust
//! use oximedia_routing::nmos_activation::{
//!     ActivationMode, ActivationRequest, ActivationScheduler,
//! };
//!
//! let mut sched = ActivationScheduler::new();
//!
//! // Schedule a connection 500 ms from "now" (t=0)
//! let req = ActivationRequest::relative("sender-1", "receiver-1", 500_000_000);
//! let id = sched.schedule(req, 0).expect("scheduled");
//!
//! // Tick at t = 300 ms — not fired yet
//! let fired = sched.tick(300_000_000);
//! assert_eq!(fired.len(), 0);
//!
//! // Tick at t = 600 ms — should fire
//! let fired = sched.tick(600_000_000);
//! assert_eq!(fired.len(), 1);
//! assert_eq!(fired[0].sender_id, "sender-1");
//! ```

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from NMOS activation scheduling.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ActivationError {
    /// An activation with the given ID does not exist.
    #[error("activation not found: {0}")]
    NotFound(u64),
    /// An activation has already been cancelled.
    #[error("activation {0} is already cancelled")]
    AlreadyCancelled(u64),
    /// An activation has already fired and cannot be modified.
    #[error("activation {0} has already fired")]
    AlreadyFired(u64),
    /// The scheduled timestamp is in the past.
    #[error("scheduled time {0} ns is in the past (current: {1} ns)")]
    ScheduledInPast(u64, u64),
    /// The relative offset is zero, which is ambiguous.
    #[error("relative offset must be greater than zero")]
    ZeroOffset,
    /// Maximum pending activations exceeded.
    #[error("pending activation limit ({0}) exceeded")]
    LimitExceeded(usize),
}

// ---------------------------------------------------------------------------
// Activation mode
// ---------------------------------------------------------------------------

/// The timing mode for an NMOS IS-05 connection activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationMode {
    /// Activate immediately upon receipt.
    Immediate,
    /// Activate at an absolute TAI timestamp (nanoseconds since epoch).
    ScheduledAbsolute,
    /// Activate after an offset (nanoseconds) from the request receipt time.
    ScheduledRelative,
}

// ---------------------------------------------------------------------------
// Activation state
// ---------------------------------------------------------------------------

/// State of a scheduled activation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActivationState {
    /// Pending — not yet fired.
    Pending,
    /// Successfully activated.
    Activated,
    /// Cancelled before firing.
    Cancelled,
    /// Failed during activation.
    Failed,
}

// ---------------------------------------------------------------------------
// ActivationRequest
// ---------------------------------------------------------------------------

/// A request to activate an NMOS IS-05 connection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActivationRequest {
    /// NMOS sender ID (UUID string).
    pub sender_id: String,
    /// NMOS receiver ID (UUID string).
    pub receiver_id: String,
    /// Activation mode.
    pub mode: ActivationMode,
    /// Requested activation time (absolute ns for `ScheduledAbsolute`; offset ns for `ScheduledRelative`; 0 for `Immediate`).
    pub requested_time_ns: u64,
    /// Optional human-readable label for this activation.
    pub label: Option<String>,
}

impl ActivationRequest {
    /// Creates an immediate activation request.
    pub fn immediate(sender_id: impl Into<String>, receiver_id: impl Into<String>) -> Self {
        Self {
            sender_id: sender_id.into(),
            receiver_id: receiver_id.into(),
            mode: ActivationMode::Immediate,
            requested_time_ns: 0,
            label: None,
        }
    }

    /// Creates a scheduled activation at an absolute timestamp.
    pub fn absolute(
        sender_id: impl Into<String>,
        receiver_id: impl Into<String>,
        time_ns: u64,
    ) -> Self {
        Self {
            sender_id: sender_id.into(),
            receiver_id: receiver_id.into(),
            mode: ActivationMode::ScheduledAbsolute,
            requested_time_ns: time_ns,
            label: None,
        }
    }

    /// Creates a scheduled activation with a relative offset from receipt time.
    pub fn relative(
        sender_id: impl Into<String>,
        receiver_id: impl Into<String>,
        offset_ns: u64,
    ) -> Self {
        Self {
            sender_id: sender_id.into(),
            receiver_id: receiver_id.into(),
            mode: ActivationMode::ScheduledRelative,
            requested_time_ns: offset_ns,
            label: None,
        }
    }

    /// Attaches a human-readable label.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }
}

// ---------------------------------------------------------------------------
// PendingActivation — internal record
// ---------------------------------------------------------------------------

/// Internal record for a pending or completed activation.
#[derive(Debug, Clone)]
struct PendingActivation {
    /// Unique ID.
    id: u64,
    /// Original request.
    request: ActivationRequest,
    /// Absolute fire time in nanoseconds (computed at schedule time).
    fire_at_ns: u64,
    /// Current state.
    state: ActivationState,
}

// ---------------------------------------------------------------------------
// FiredActivation — returned by tick()
// ---------------------------------------------------------------------------

/// Information about an activation that has been fired.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FiredActivation {
    /// Scheduler-assigned activation ID.
    pub id: u64,
    /// Sender ID from the original request.
    pub sender_id: String,
    /// Receiver ID from the original request.
    pub receiver_id: String,
    /// The absolute timestamp at which the activation was scheduled to fire.
    pub scheduled_ns: u64,
    /// The timestamp at which `tick()` fired this activation.
    pub actual_ns: u64,
    /// Optional label.
    pub label: Option<String>,
}

// ---------------------------------------------------------------------------
// ActivationScheduler
// ---------------------------------------------------------------------------

/// Maximum number of concurrent pending activations.
const MAX_PENDING: usize = 1024;

/// Schedules and fires NMOS IS-05 connection activations.
pub struct ActivationScheduler {
    /// All activations (pending, fired, cancelled).
    activations: HashMap<u64, PendingActivation>,
    /// Next ID to assign.
    next_id: u64,
    /// History of fired activations (in order).
    history: Vec<FiredActivation>,
    /// Maximum history length before old entries are dropped.
    max_history: usize,
}

impl ActivationScheduler {
    /// Creates a new, empty scheduler.
    pub fn new() -> Self {
        Self {
            activations: HashMap::new(),
            next_id: 1,
            history: Vec::new(),
            max_history: 1000,
        }
    }

    /// Creates a scheduler with a custom history limit.
    pub fn with_history_limit(max_history: usize) -> Self {
        Self {
            max_history,
            ..Self::new()
        }
    }

    /// Schedules an activation request, returning its assigned ID.
    ///
    /// `now_ns` is the current monotonic timestamp in nanoseconds.
    pub fn schedule(
        &mut self,
        request: ActivationRequest,
        now_ns: u64,
    ) -> Result<u64, ActivationError> {
        let pending_count = self
            .activations
            .values()
            .filter(|a| a.state == ActivationState::Pending)
            .count();
        if pending_count >= MAX_PENDING {
            return Err(ActivationError::LimitExceeded(MAX_PENDING));
        }

        let fire_at_ns = match request.mode {
            ActivationMode::Immediate => now_ns,
            ActivationMode::ScheduledAbsolute => {
                if request.requested_time_ns < now_ns {
                    return Err(ActivationError::ScheduledInPast(
                        request.requested_time_ns,
                        now_ns,
                    ));
                }
                request.requested_time_ns
            }
            ActivationMode::ScheduledRelative => {
                if request.requested_time_ns == 0 {
                    return Err(ActivationError::ZeroOffset);
                }
                now_ns.saturating_add(request.requested_time_ns)
            }
        };

        let id = self.next_id;
        self.next_id += 1;

        self.activations.insert(
            id,
            PendingActivation {
                id,
                request,
                fire_at_ns,
                state: ActivationState::Pending,
            },
        );

        Ok(id)
    }

    /// Cancels a pending activation.
    pub fn cancel(&mut self, id: u64) -> Result<(), ActivationError> {
        let entry = self
            .activations
            .get_mut(&id)
            .ok_or(ActivationError::NotFound(id))?;

        match entry.state {
            ActivationState::Cancelled => Err(ActivationError::AlreadyCancelled(id)),
            ActivationState::Activated | ActivationState::Failed => {
                Err(ActivationError::AlreadyFired(id))
            }
            ActivationState::Pending => {
                entry.state = ActivationState::Cancelled;
                Ok(())
            }
        }
    }

    /// Advances the scheduler clock to `now_ns`, firing all due activations.
    ///
    /// Returns the list of activations that fired during this tick.
    pub fn tick(&mut self, now_ns: u64) -> Vec<FiredActivation> {
        let mut fired = Vec::new();

        // Collect IDs of pending activations whose fire time has arrived
        let due_ids: Vec<u64> = self
            .activations
            .values()
            .filter(|a| a.state == ActivationState::Pending && a.fire_at_ns <= now_ns)
            .map(|a| a.id)
            .collect();

        for id in due_ids {
            if let Some(entry) = self.activations.get_mut(&id) {
                entry.state = ActivationState::Activated;
                let fa = FiredActivation {
                    id: entry.id,
                    sender_id: entry.request.sender_id.clone(),
                    receiver_id: entry.request.receiver_id.clone(),
                    scheduled_ns: entry.fire_at_ns,
                    actual_ns: now_ns,
                    label: entry.request.label.clone(),
                };
                fired.push(fa.clone());
                if self.history.len() >= self.max_history {
                    self.history.remove(0);
                }
                self.history.push(fa);
            }
        }

        // Sort by scheduled time for deterministic ordering
        fired.sort_by_key(|f| f.scheduled_ns);
        fired
    }

    /// Returns the current state of an activation.
    pub fn state(&self, id: u64) -> Option<ActivationState> {
        self.activations.get(&id).map(|a| a.state)
    }

    /// Returns the number of pending activations.
    pub fn pending_count(&self) -> usize {
        self.activations
            .values()
            .filter(|a| a.state == ActivationState::Pending)
            .count()
    }

    /// Returns the total number of activations (all states).
    pub fn total_count(&self) -> usize {
        self.activations.len()
    }

    /// Returns the firing history.
    pub fn history(&self) -> &[FiredActivation] {
        &self.history
    }

    /// Clears fired and cancelled activations from memory, retaining only pending ones.
    pub fn prune(&mut self) {
        self.activations
            .retain(|_, a| a.state == ActivationState::Pending);
    }

    /// Returns the scheduled fire time (ns) for a pending activation.
    pub fn fire_time(&self, id: u64) -> Option<u64> {
        self.activations
            .get(&id)
            .filter(|a| a.state == ActivationState::Pending)
            .map(|a| a.fire_at_ns)
    }
}

impl Default for ActivationScheduler {
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

    #[test]
    fn test_immediate_activation_fires_at_now() {
        let mut sched = ActivationScheduler::new();
        let req = ActivationRequest::immediate("s1", "r1");
        let id = sched.schedule(req, 1000).expect("scheduled");
        let fired = sched.tick(1000);
        assert_eq!(fired.len(), 1);
        assert_eq!(fired[0].id, id);
        assert_eq!(fired[0].sender_id, "s1");
        assert_eq!(fired[0].receiver_id, "r1");
    }

    #[test]
    fn test_absolute_fires_at_scheduled_time() {
        let mut sched = ActivationScheduler::new();
        let req = ActivationRequest::absolute("s2", "r2", 5_000_000_000);
        let _id = sched.schedule(req, 0).expect("scheduled");

        let before = sched.tick(4_999_999_999);
        assert_eq!(before.len(), 0);

        let at_time = sched.tick(5_000_000_000);
        assert_eq!(at_time.len(), 1);
        assert_eq!(at_time[0].sender_id, "s2");
    }

    #[test]
    fn test_relative_fires_after_offset() {
        let mut sched = ActivationScheduler::new();
        let req = ActivationRequest::relative("s3", "r3", 500_000_000);
        let _id = sched.schedule(req, 0).expect("scheduled");

        let before = sched.tick(300_000_000);
        assert_eq!(before.len(), 0);

        let after = sched.tick(600_000_000);
        assert_eq!(after.len(), 1);
        assert_eq!(after[0].sender_id, "s3");
    }

    #[test]
    fn test_cancel_pending() {
        let mut sched = ActivationScheduler::new();
        let req = ActivationRequest::absolute("s4", "r4", 10_000);
        let id = sched.schedule(req, 0).expect("scheduled");

        sched.cancel(id).expect("cancel ok");
        assert_eq!(sched.state(id), Some(ActivationState::Cancelled));

        let fired = sched.tick(10_001);
        assert_eq!(fired.len(), 0);
    }

    #[test]
    fn test_cancel_already_fired_returns_error() {
        let mut sched = ActivationScheduler::new();
        let req = ActivationRequest::immediate("s5", "r5");
        let id = sched.schedule(req, 0).expect("scheduled");
        sched.tick(0);

        let result = sched.cancel(id);
        assert!(matches!(result, Err(ActivationError::AlreadyFired(_))));
    }

    #[test]
    fn test_scheduled_in_past_error() {
        let mut sched = ActivationScheduler::new();
        let req = ActivationRequest::absolute("s6", "r6", 100);
        let result = sched.schedule(req, 5000);
        assert!(matches!(result, Err(ActivationError::ScheduledInPast(..))));
    }

    #[test]
    fn test_zero_relative_offset_error() {
        let mut sched = ActivationScheduler::new();
        let req = ActivationRequest::relative("s7", "r7", 0);
        let result = sched.schedule(req, 0);
        assert!(matches!(result, Err(ActivationError::ZeroOffset)));
    }

    #[test]
    fn test_pending_count() {
        let mut sched = ActivationScheduler::new();
        sched
            .schedule(ActivationRequest::absolute("a", "b", 100), 0)
            .expect("ok");
        sched
            .schedule(ActivationRequest::absolute("c", "d", 200), 0)
            .expect("ok");
        assert_eq!(sched.pending_count(), 2);

        sched.tick(150);
        assert_eq!(sched.pending_count(), 1);
    }

    #[test]
    fn test_history_recording() {
        let mut sched = ActivationScheduler::new();
        sched
            .schedule(ActivationRequest::immediate("x", "y"), 0)
            .expect("ok");
        sched.tick(0);
        assert_eq!(sched.history().len(), 1);
    }

    #[test]
    fn test_prune_removes_non_pending() {
        let mut sched = ActivationScheduler::new();
        let id = sched
            .schedule(ActivationRequest::immediate("p", "q"), 0)
            .expect("ok");
        sched.tick(0);
        assert_eq!(sched.total_count(), 1);
        sched.prune();
        assert_eq!(sched.total_count(), 0);
        assert!(sched.state(id).is_none());
    }

    #[test]
    fn test_fire_time_query() {
        let mut sched = ActivationScheduler::new();
        let req = ActivationRequest::absolute("t1", "t2", 9999);
        let id = sched.schedule(req, 0).expect("ok");
        assert_eq!(sched.fire_time(id), Some(9999));
    }

    #[test]
    fn test_label_propagates_to_fired() {
        let mut sched = ActivationScheduler::new();
        let req = ActivationRequest::immediate("lbl_s", "lbl_r").with_label("Test Activation");
        let _id = sched.schedule(req, 0).expect("ok");
        let fired = sched.tick(0);
        assert_eq!(fired[0].label.as_deref(), Some("Test Activation"));
    }

    #[test]
    fn test_multiple_activations_in_one_tick() {
        let mut sched = ActivationScheduler::new();
        sched
            .schedule(ActivationRequest::absolute("m1", "n1", 100), 0)
            .expect("ok");
        sched
            .schedule(ActivationRequest::absolute("m2", "n2", 100), 0)
            .expect("ok");
        sched
            .schedule(ActivationRequest::absolute("m3", "n3", 200), 0)
            .expect("ok");

        let fired = sched.tick(150);
        assert_eq!(fired.len(), 2);
    }

    #[test]
    fn test_default_constructor() {
        let sched = ActivationScheduler::default();
        assert_eq!(sched.pending_count(), 0);
        assert!(sched.history().is_empty());
    }

    #[test]
    fn test_not_found_cancel() {
        let mut sched = ActivationScheduler::new();
        let result = sched.cancel(9999);
        assert!(matches!(result, Err(ActivationError::NotFound(9999))));
    }
}
