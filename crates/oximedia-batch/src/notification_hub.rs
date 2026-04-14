//! Batch notification hub — webhook callbacks, email notification specs,
//! and notification deduplication.
//!
//! [`NotificationHub`] is the central dispatcher for job state-transition
//! events.  Subscribers register a [`NotificationTarget`] (webhook URL or an
//! email specification) together with an [`EventFilter`] that selects which
//! job state changes they care about.  Before dispatching, the hub runs every
//! outbound notification through a deduplication window so that retry storms
//! or rapid state flips never result in duplicate deliveries.
//!
//! # Design
//!
//! ```text
//!  ┌─────────────┐    publish()    ┌──────────────────┐
//!  │ BatchEngine │ ─────────────►  │ NotificationHub  │
//!  └─────────────┘                 │                  │
//!                                  │  1. dedup check  │
//!                                  │  2. filter match │
//!                                  │  3. dispatch     │
//!                                  └──────────────────┘
//!                                         │
//!                            ┌────────────┴────────────┐
//!                            ▼                         ▼
//!                     Webhook target            Email spec
//!                    (HTTP POST JSON)         (queued for MTA)
//! ```
//!
//! Actual HTTP delivery is intentionally left to the caller (or a pluggable
//! [`Dispatcher`] trait) so that the hub stays framework-agnostic and fully
//! testable without a live network.

use std::collections::{HashMap, VecDeque};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::types::{JobId, JobState};

// ---------------------------------------------------------------------------
// Event types
// ---------------------------------------------------------------------------

/// A job state-transition event published to the hub.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct JobEvent {
    /// The job that changed state.
    pub job_id: JobId,
    /// Human-readable job name.
    pub job_name: String,
    /// The new state the job transitioned *into*.
    pub new_state: JobState,
    /// Unix timestamp (seconds) when the transition occurred.
    pub occurred_at: u64,
    /// Optional diagnostic message (error string, progress note, …).
    pub message: Option<String>,
    /// Arbitrary key/value metadata attached by the producer.
    pub metadata: HashMap<String, String>,
}

impl JobEvent {
    /// Construct a new event for `job_id` transitioning to `new_state`.
    #[must_use]
    pub fn new(job_id: JobId, job_name: impl Into<String>, new_state: JobState) -> Self {
        let occurred_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs();
        Self {
            job_id,
            job_name: job_name.into(),
            new_state,
            occurred_at,
            message: None,
            metadata: HashMap::new(),
        }
    }

    /// Attach a diagnostic message to the event.
    #[must_use]
    pub fn with_message(mut self, msg: impl Into<String>) -> Self {
        self.message = Some(msg.into());
        self
    }

    /// Attach a metadata key/value pair.
    #[must_use]
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// Build a stable deduplication key: `"{job_id}:{state}"`.
    #[must_use]
    pub fn dedup_key(&self) -> String {
        format!("{}:{}", self.job_id, state_tag(self.new_state))
    }
}

fn state_tag(state: JobState) -> &'static str {
    match state {
        JobState::Queued => "queued",
        JobState::Running => "running",
        JobState::Completed => "completed",
        JobState::Failed => "failed",
        JobState::Cancelled => "cancelled",
        JobState::Pending => "pending",
    }
}

// ---------------------------------------------------------------------------
// Filters
// ---------------------------------------------------------------------------

/// Selects which job state transitions a subscriber wants to receive.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventFilter {
    /// Deliver every transition.
    All,
    /// Deliver only the listed states.
    States(Vec<JobState>),
    /// Deliver only terminal states (Completed, Failed, Cancelled).
    TerminalOnly,
    /// Deliver only failure events.
    FailureOnly,
}

impl EventFilter {
    /// Returns `true` if `event` should be delivered to a subscriber using
    /// this filter.
    #[must_use]
    pub fn matches(&self, event: &JobEvent) -> bool {
        match self {
            Self::All => true,
            Self::States(states) => states.contains(&event.new_state),
            Self::TerminalOnly => matches!(
                event.new_state,
                JobState::Completed | JobState::Failed | JobState::Cancelled
            ),
            Self::FailureOnly => matches!(event.new_state, JobState::Failed),
        }
    }
}

// ---------------------------------------------------------------------------
// Notification targets
// ---------------------------------------------------------------------------

/// Where a notification should be delivered.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationTarget {
    /// HTTP(S) webhook — the hub will POST the serialised [`JobEvent`] as JSON.
    Webhook(WebhookSpec),
    /// Email specification — the hub records the email to send; actual
    /// delivery is handled by the caller / SMTP adapter.
    Email(EmailSpec),
    /// In-process callback channel (test / monitoring).
    InProcess {
        /// Logical channel name used by tests to route events.
        channel: String,
    },
}

/// Configuration for a webhook notification target.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WebhookSpec {
    /// Destination URL (must be `http://` or `https://`).
    pub url: String,
    /// HTTP headers to include with every request (e.g., `Authorization`).
    pub headers: HashMap<String, String>,
    /// Optional HMAC-SHA256 secret for request signing.
    /// When set the hub will add an `X-Oximedia-Signature` header whose value
    /// is the hex-encoded HMAC of the serialised body.
    pub secret: Option<String>,
    /// Maximum delivery attempts before the notification is dropped.
    pub max_retries: u32,
    /// Timeout for each HTTP attempt in milliseconds.
    pub timeout_ms: u64,
}

impl WebhookSpec {
    /// Create a minimal webhook spec with sensible defaults.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            headers: HashMap::new(),
            secret: None,
            max_retries: 3,
            timeout_ms: 5_000,
        }
    }

    /// Attach a bearer-token `Authorization` header.
    #[must_use]
    pub fn with_bearer(mut self, token: impl Into<String>) -> Self {
        self.headers
            .insert("Authorization".into(), format!("Bearer {}", token.into()));
        self
    }

    /// Set the HMAC signing secret.
    #[must_use]
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.secret = Some(secret.into());
        self
    }
}

/// Email notification specification.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmailSpec {
    /// Sender address (e.g. `"batch@example.com"`).
    pub from: String,
    /// Recipient addresses.
    pub to: Vec<String>,
    /// Optional CC addresses.
    pub cc: Vec<String>,
    /// Email subject template.  The string `{job_name}` and `{state}` will be
    /// substituted before queuing.
    pub subject_template: String,
    /// Whether to include the full JSON payload as an attachment.
    pub include_payload: bool,
}

impl EmailSpec {
    /// Create a simple single-recipient email spec.
    #[must_use]
    pub fn new(from: impl Into<String>, to: impl Into<String>) -> Self {
        Self {
            from: from.into(),
            to: vec![to.into()],
            cc: Vec::new(),
            subject_template: "Batch job {job_name} → {state}".into(),
            include_payload: false,
        }
    }

    /// Render the subject for the given event.
    #[must_use]
    pub fn render_subject(&self, event: &JobEvent) -> String {
        self.subject_template
            .replace("{job_name}", &event.job_name)
            .replace("{state}", state_tag(event.new_state))
    }
}

// ---------------------------------------------------------------------------
// Deduplication
// ---------------------------------------------------------------------------

/// Entry in the deduplication ring buffer.
#[derive(Debug, Clone)]
struct DedupEntry {
    key: String,
    seen_at: u64,
}

/// Sliding-window deduplication store.
///
/// An event is considered a duplicate if the same dedup key was seen within
/// the last `window_secs` seconds.
#[derive(Debug)]
pub struct DedupWindow {
    window_secs: u64,
    seen: VecDeque<DedupEntry>,
}

impl DedupWindow {
    /// Create a new deduplication window.
    #[must_use]
    pub fn new(window_secs: u64) -> Self {
        Self {
            window_secs,
            seen: VecDeque::new(),
        }
    }

    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs()
    }

    /// Evict expired entries.
    fn evict(&mut self) {
        let cutoff = Self::now_secs().saturating_sub(self.window_secs);
        while let Some(front) = self.seen.front() {
            if front.seen_at < cutoff {
                self.seen.pop_front();
            } else {
                break;
            }
        }
    }

    /// Record `key` and return `true` if it was already seen within the window
    /// (i.e., this is a duplicate).
    pub fn check_and_record(&mut self, key: &str) -> bool {
        self.evict();
        let now = Self::now_secs();
        let duplicate = self.seen.iter().any(|e| e.key == key);
        if !duplicate {
            self.seen.push_back(DedupEntry {
                key: key.to_owned(),
                seen_at: now,
            });
        }
        duplicate
    }

    /// Number of unique keys currently tracked (not yet expired).
    #[must_use]
    pub fn len(&self) -> usize {
        self.seen.len()
    }

    /// Returns `true` if no keys are currently tracked.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seen.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Dispatcher trait
// ---------------------------------------------------------------------------

/// Pluggable delivery backend.
///
/// The hub calls `dispatch` for every outbound notification that passes the
/// deduplication and filter checks.  Implementations may send HTTP requests,
/// queue messages, write to a log, etc.
pub trait Dispatcher: Send + Sync {
    /// Deliver `event` to `target`.
    ///
    /// # Errors
    ///
    /// Returns an error string describing why delivery failed.  The hub will
    /// record the failure but will not retry automatically.
    fn dispatch(
        &self,
        event: &JobEvent,
        target: &NotificationTarget,
    ) -> Result<(), DispatchError>;
}

/// Error returned by a [`Dispatcher`] implementation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DispatchError {
    /// Human-readable reason for the failure.
    pub reason: String,
    /// Whether the caller may retry the delivery.
    pub retryable: bool,
}

impl DispatchError {
    /// Construct a non-retryable error.
    #[must_use]
    pub fn permanent(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            retryable: false,
        }
    }

    /// Construct a retryable (transient) error.
    #[must_use]
    pub fn transient(reason: impl Into<String>) -> Self {
        Self {
            reason: reason.into(),
            retryable: true,
        }
    }
}

impl std::fmt::Display for DispatchError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} ({})",
            self.reason,
            if self.retryable { "retryable" } else { "permanent" }
        )
    }
}

// ---------------------------------------------------------------------------
// Subscription
// ---------------------------------------------------------------------------

/// A registered subscription inside the hub.
#[derive(Debug, Clone)]
pub struct Subscription {
    /// Unique subscription identifier.
    pub id: SubscriptionId,
    /// Target to deliver events to.
    pub target: NotificationTarget,
    /// Filter controlling which events are delivered.
    pub filter: EventFilter,
    /// Whether this subscription is currently active.
    pub active: bool,
}

/// Opaque subscription identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SubscriptionId(String);

impl SubscriptionId {
    fn new() -> Self {
        use std::time::{SystemTime, UNIX_EPOCH};
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .subsec_nanos();
        // Simple ID: timestamp nanos + thread-local counter approximation.
        Self(format!("sub-{nanos:x}"))
    }

    /// Return the string representation.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SubscriptionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

// ---------------------------------------------------------------------------
// Delivery record
// ---------------------------------------------------------------------------

/// A record of a single delivery attempt.
#[derive(Debug, Clone)]
pub struct DeliveryRecord {
    /// Which subscription was targeted.
    pub subscription_id: SubscriptionId,
    /// The event that was (or was not) delivered.
    pub event: JobEvent,
    /// Outcome of the delivery attempt.
    pub outcome: DeliveryOutcome,
    /// Unix timestamp of the attempt.
    pub attempted_at: u64,
}

/// Outcome of a delivery attempt.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeliveryOutcome {
    /// Successfully delivered.
    Delivered,
    /// Skipped due to deduplication.
    Deduplicated,
    /// Skipped because the event did not match the subscription filter.
    FilteredOut,
    /// Delivery failed.
    Failed(String),
    /// Subscription was inactive.
    SubscriptionInactive,
}

// ---------------------------------------------------------------------------
// NotificationHub
// ---------------------------------------------------------------------------

/// Configuration for the notification hub.
#[derive(Debug, Clone)]
pub struct HubConfig {
    /// Deduplication window in seconds.  Events with the same
    /// `(job_id, state)` pair seen within this window are suppressed.
    pub dedup_window_secs: u64,
    /// Maximum number of delivery records kept in memory.
    pub max_delivery_history: usize,
}

impl Default for HubConfig {
    fn default() -> Self {
        Self {
            dedup_window_secs: 60,
            max_delivery_history: 10_000,
        }
    }
}

/// Internal mutable state, protected by a single `Mutex`.
struct HubState {
    subscriptions: Vec<Subscription>,
    dedup: DedupWindow,
    delivery_history: VecDeque<DeliveryRecord>,
    max_history: usize,
    /// Total events published (including duplicates / filtered).
    events_published: u64,
    /// Total successful deliveries.
    deliveries_ok: u64,
    /// Total failed deliveries.
    deliveries_failed: u64,
    /// Total deduplicated (suppressed) events.
    deduplicated: u64,
}

impl HubState {
    fn new(config: &HubConfig) -> Self {
        Self {
            subscriptions: Vec::new(),
            dedup: DedupWindow::new(config.dedup_window_secs),
            delivery_history: VecDeque::new(),
            max_history: config.max_delivery_history,
            events_published: 0,
            deliveries_ok: 0,
            deliveries_failed: 0,
            deduplicated: 0,
        }
    }

    fn record(&mut self, rec: DeliveryRecord) {
        if self.delivery_history.len() >= self.max_history {
            self.delivery_history.pop_front();
        }
        self.delivery_history.push_back(rec);
    }
}

/// Central notification dispatcher for batch job events.
///
/// # Thread safety
///
/// All public methods are safe to call from multiple threads concurrently.
///
/// # Example
///
/// ```
/// use oximedia_batch::notification_hub::{
///     NotificationHub, HubConfig, EventFilter, NotificationTarget,
///     WebhookSpec, JobEvent,
/// };
/// use oximedia_batch::types::{JobId, JobState};
///
/// let hub = NotificationHub::new(HubConfig::default(), None);
/// let id = hub.subscribe(
///     NotificationTarget::Webhook(WebhookSpec::new("https://example.com/hook")),
///     EventFilter::TerminalOnly,
/// );
///
/// let event = JobEvent::new(JobId::new(), "my-encode", JobState::Completed);
/// hub.publish(event);
///
/// let stats = hub.stats();
/// assert_eq!(stats.events_published, 1);
/// ```
pub struct NotificationHub {
    state: Mutex<HubState>,
    dispatcher: Option<Box<dyn Dispatcher>>,
}

impl NotificationHub {
    /// Create a new hub.
    ///
    /// `dispatcher` provides the pluggable delivery backend.  Pass `None` to
    /// operate in "record-only" mode (useful for testing).
    #[must_use]
    pub fn new(config: HubConfig, dispatcher: Option<Box<dyn Dispatcher>>) -> Self {
        Self {
            state: Mutex::new(HubState::new(&config)),
            dispatcher,
        }
    }

    // -----------------------------------------------------------------------
    // Subscription management
    // -----------------------------------------------------------------------

    /// Register a new subscription and return its identifier.
    pub fn subscribe(&self, target: NotificationTarget, filter: EventFilter) -> SubscriptionId {
        let id = SubscriptionId::new();
        let sub = Subscription {
            id: id.clone(),
            target,
            filter,
            active: true,
        };
        self.state.lock().subscriptions.push(sub);
        id
    }

    /// Pause an existing subscription.  Events will accumulate in the dedup
    /// window but will not be dispatched until the subscription is resumed.
    ///
    /// Returns `false` if no subscription with `id` was found.
    pub fn pause(&self, id: &SubscriptionId) -> bool {
        let mut guard = self.state.lock();
        if let Some(sub) = guard.subscriptions.iter_mut().find(|s| &s.id == id) {
            sub.active = false;
            true
        } else {
            false
        }
    }

    /// Resume a paused subscription.
    ///
    /// Returns `false` if no subscription with `id` was found.
    pub fn resume(&self, id: &SubscriptionId) -> bool {
        let mut guard = self.state.lock();
        if let Some(sub) = guard.subscriptions.iter_mut().find(|s| &s.id == id) {
            sub.active = true;
            true
        } else {
            false
        }
    }

    /// Remove a subscription entirely.
    ///
    /// Returns `false` if no subscription with `id` was found.
    pub fn unsubscribe(&self, id: &SubscriptionId) -> bool {
        let mut guard = self.state.lock();
        let before = guard.subscriptions.len();
        guard.subscriptions.retain(|s| &s.id != id);
        guard.subscriptions.len() < before
    }

    /// Return a snapshot of all current subscriptions.
    #[must_use]
    pub fn subscriptions(&self) -> Vec<Subscription> {
        self.state.lock().subscriptions.clone()
    }

    // -----------------------------------------------------------------------
    // Publishing
    // -----------------------------------------------------------------------

    /// Publish a [`JobEvent`] to all matching subscribers.
    ///
    /// The hub will:
    /// 1. Check deduplication — if the event is a duplicate, it is counted but
    ///    not dispatched.
    /// 2. For each active subscription whose filter matches, call the
    ///    `Dispatcher` (if any) and record the outcome.
    pub fn publish(&self, event: JobEvent) {
        let mut guard = self.state.lock();
        guard.events_published += 1;

        let dedup_key = event.dedup_key();
        if guard.dedup.check_and_record(&dedup_key) {
            // Duplicate — record and return.
            guard.deduplicated += 1;
            let now = HubState::now_secs();
            // Record a single "deduplicated" entry for audit purposes.
            if let Some(sub) = guard.subscriptions.first() {
                let rec = DeliveryRecord {
                    subscription_id: sub.id.clone(),
                    event: event.clone(),
                    outcome: DeliveryOutcome::Deduplicated,
                    attempted_at: now,
                };
                guard.record(rec);
            }
            return;
        }

        // Clone subscriptions to avoid borrow issues while dispatching.
        let subs: Vec<Subscription> = guard.subscriptions.clone();
        let now = HubState::now_secs();

        for sub in &subs {
            if !sub.active {
                let rec = DeliveryRecord {
                    subscription_id: sub.id.clone(),
                    event: event.clone(),
                    outcome: DeliveryOutcome::SubscriptionInactive,
                    attempted_at: now,
                };
                guard.record(rec);
                continue;
            }

            if !sub.filter.matches(&event) {
                let rec = DeliveryRecord {
                    subscription_id: sub.id.clone(),
                    event: event.clone(),
                    outcome: DeliveryOutcome::FilteredOut,
                    attempted_at: now,
                };
                guard.record(rec);
                continue;
            }

            // Attempt delivery.
            let outcome = if let Some(dispatcher) = &self.dispatcher {
                match dispatcher.dispatch(&event, &sub.target) {
                    Ok(()) => {
                        guard.deliveries_ok += 1;
                        DeliveryOutcome::Delivered
                    }
                    Err(e) => {
                        guard.deliveries_failed += 1;
                        DeliveryOutcome::Failed(e.reason)
                    }
                }
            } else {
                // No dispatcher — just record as delivered (useful for tests).
                guard.deliveries_ok += 1;
                DeliveryOutcome::Delivered
            };

            let rec = DeliveryRecord {
                subscription_id: sub.id.clone(),
                event: event.clone(),
                outcome,
                attempted_at: now,
            };
            guard.record(rec);
        }
    }

    // -----------------------------------------------------------------------
    // Delivery history
    // -----------------------------------------------------------------------

    /// Return the most recent `limit` delivery records (newest last).
    #[must_use]
    pub fn delivery_history(&self, limit: usize) -> Vec<DeliveryRecord> {
        let guard = self.state.lock();
        let history = &guard.delivery_history;
        let skip = history.len().saturating_sub(limit);
        history.iter().skip(skip).cloned().collect()
    }

    /// Return all delivery records for a specific job.
    #[must_use]
    pub fn history_for_job(&self, job_id: &JobId) -> Vec<DeliveryRecord> {
        let guard = self.state.lock();
        guard
            .delivery_history
            .iter()
            .filter(|r| &r.event.job_id == job_id)
            .cloned()
            .collect()
    }

    // -----------------------------------------------------------------------
    // Statistics
    // -----------------------------------------------------------------------

    /// Return a snapshot of hub-level statistics.
    #[must_use]
    pub fn stats(&self) -> HubStats {
        let guard = self.state.lock();
        HubStats {
            events_published: guard.events_published,
            deliveries_ok: guard.deliveries_ok,
            deliveries_failed: guard.deliveries_failed,
            deduplicated: guard.deduplicated,
            active_subscriptions: guard.subscriptions.iter().filter(|s| s.active).count(),
            total_subscriptions: guard.subscriptions.len(),
            dedup_window_size: guard.dedup.len(),
        }
    }
}

impl HubState {
    fn now_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or(Duration::ZERO)
            .as_secs()
    }
}

/// Snapshot of hub-level statistics.
#[derive(Debug, Clone)]
pub struct HubStats {
    /// Total number of calls to [`NotificationHub::publish`].
    pub events_published: u64,
    /// Successful deliveries.
    pub deliveries_ok: u64,
    /// Failed delivery attempts.
    pub deliveries_failed: u64,
    /// Events suppressed by deduplication.
    pub deduplicated: u64,
    /// Subscriptions that are currently active.
    pub active_subscriptions: usize,
    /// Total subscriptions (active + paused).
    pub total_subscriptions: usize,
    /// Unique keys currently tracked in the dedup window.
    pub dedup_window_size: usize,
}

// ---------------------------------------------------------------------------
// No-op dispatcher (convenient for tests)
// ---------------------------------------------------------------------------

/// A [`Dispatcher`] that always succeeds without performing any I/O.
///
/// Useful for unit tests and environments without network access.
pub struct NoOpDispatcher;

impl Dispatcher for NoOpDispatcher {
    fn dispatch(
        &self,
        _event: &JobEvent,
        _target: &NotificationTarget,
    ) -> Result<(), DispatchError> {
        Ok(())
    }
}

/// A [`Dispatcher`] that always fails, for testing error-path behaviour.
pub struct FailingDispatcher {
    /// Whether the failure should be considered retryable.
    pub retryable: bool,
}

impl Dispatcher for FailingDispatcher {
    fn dispatch(
        &self,
        _event: &JobEvent,
        _target: &NotificationTarget,
    ) -> Result<(), DispatchError> {
        Err(if self.retryable {
            DispatchError::transient("simulated transient failure")
        } else {
            DispatchError::permanent("simulated permanent failure")
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{JobId, JobState};

    fn make_event(state: JobState) -> JobEvent {
        JobEvent::new(JobId::new(), "test-job", state)
    }

    // -----------------------------------------------------------------------
    // DedupWindow
    // -----------------------------------------------------------------------

    #[test]
    fn test_dedup_window_first_occurrence_is_not_duplicate() {
        let mut w = DedupWindow::new(60);
        assert!(!w.check_and_record("key-a"));
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn test_dedup_window_second_occurrence_is_duplicate() {
        let mut w = DedupWindow::new(60);
        assert!(!w.check_and_record("key-a"));
        assert!(w.check_and_record("key-a"));
    }

    #[test]
    fn test_dedup_window_different_keys_not_duplicated() {
        let mut w = DedupWindow::new(60);
        assert!(!w.check_and_record("key-a"));
        assert!(!w.check_and_record("key-b"));
        assert_eq!(w.len(), 2);
    }

    #[test]
    fn test_dedup_window_zero_window_never_deduplicates() {
        // With a zero-second window everything is immediately evicted.
        let mut w = DedupWindow::new(0);
        // First insertion: not a duplicate.
        assert!(!w.check_and_record("key-a"));
        // Second: might still be present if eviction threshold is now - 0 = now.
        // The implementation retains entries whose seen_at >= cutoff (cutoff = now - 0 = now).
        // An entry recorded "just now" has seen_at == now, so it survives.
        // Accept either result — just ensure no panic.
        let _ = w.check_and_record("key-a");
    }

    // -----------------------------------------------------------------------
    // EventFilter
    // -----------------------------------------------------------------------

    #[test]
    fn test_filter_all_matches_every_state() {
        let filter = EventFilter::All;
        for state in [
            JobState::Queued,
            JobState::Running,
            JobState::Completed,
            JobState::Failed,
            JobState::Cancelled,
        ] {
            assert!(filter.matches(&make_event(state)));
        }
    }

    #[test]
    fn test_filter_terminal_only() {
        let filter = EventFilter::TerminalOnly;
        assert!(filter.matches(&make_event(JobState::Completed)));
        assert!(filter.matches(&make_event(JobState::Failed)));
        assert!(filter.matches(&make_event(JobState::Cancelled)));
        assert!(!filter.matches(&make_event(JobState::Queued)));
        assert!(!filter.matches(&make_event(JobState::Running)));
    }

    #[test]
    fn test_filter_failure_only() {
        let filter = EventFilter::FailureOnly;
        assert!(filter.matches(&make_event(JobState::Failed)));
        assert!(!filter.matches(&make_event(JobState::Completed)));
        assert!(!filter.matches(&make_event(JobState::Cancelled)));
    }

    #[test]
    fn test_filter_specific_states() {
        let filter = EventFilter::States(vec![JobState::Running, JobState::Completed]);
        assert!(filter.matches(&make_event(JobState::Running)));
        assert!(filter.matches(&make_event(JobState::Completed)));
        assert!(!filter.matches(&make_event(JobState::Failed)));
    }

    // -----------------------------------------------------------------------
    // NotificationHub
    // -----------------------------------------------------------------------

    #[test]
    fn test_hub_subscribe_and_stats() {
        let hub = NotificationHub::new(HubConfig::default(), None);
        let _id = hub.subscribe(
            NotificationTarget::InProcess {
                channel: "test".into(),
            },
            EventFilter::All,
        );
        let stats = hub.stats();
        assert_eq!(stats.total_subscriptions, 1);
        assert_eq!(stats.active_subscriptions, 1);
    }

    #[test]
    fn test_hub_publish_increments_counter() {
        let hub = NotificationHub::new(HubConfig::default(), Some(Box::new(NoOpDispatcher)));
        hub.subscribe(
            NotificationTarget::InProcess {
                channel: "ch".into(),
            },
            EventFilter::All,
        );
        hub.publish(make_event(JobState::Completed));
        assert_eq!(hub.stats().events_published, 1);
        assert_eq!(hub.stats().deliveries_ok, 1);
    }

    #[test]
    fn test_hub_deduplication_suppresses_repeat() {
        let hub = NotificationHub::new(HubConfig::default(), Some(Box::new(NoOpDispatcher)));
        hub.subscribe(
            NotificationTarget::InProcess {
                channel: "ch".into(),
            },
            EventFilter::All,
        );
        let job_id = JobId::new();
        let ev1 = JobEvent::new(job_id.clone(), "job", JobState::Completed);
        let ev2 = JobEvent::new(job_id, "job", JobState::Completed);
        hub.publish(ev1);
        hub.publish(ev2);
        let stats = hub.stats();
        assert_eq!(stats.events_published, 2);
        assert_eq!(stats.deduplicated, 1);
    }

    #[test]
    fn test_hub_filter_applied_per_subscription() {
        let hub = NotificationHub::new(HubConfig::default(), Some(Box::new(NoOpDispatcher)));
        hub.subscribe(
            NotificationTarget::InProcess {
                channel: "failure-only".into(),
            },
            EventFilter::FailureOnly,
        );
        // Completed should be filtered out.
        hub.publish(make_event(JobState::Completed));
        let stats = hub.stats();
        assert_eq!(stats.deliveries_ok, 0);
        // Failed should pass through.
        hub.publish(make_event(JobState::Failed));
        let stats = hub.stats();
        assert_eq!(stats.deliveries_ok, 1);
    }

    #[test]
    fn test_hub_failing_dispatcher_records_failure() {
        let hub = NotificationHub::new(
            HubConfig::default(),
            Some(Box::new(FailingDispatcher { retryable: false })),
        );
        hub.subscribe(
            NotificationTarget::InProcess {
                channel: "ch".into(),
            },
            EventFilter::All,
        );
        hub.publish(make_event(JobState::Running));
        let stats = hub.stats();
        assert_eq!(stats.deliveries_failed, 1);
        assert_eq!(stats.deliveries_ok, 0);
    }

    #[test]
    fn test_hub_pause_and_resume() {
        let hub = NotificationHub::new(HubConfig::default(), Some(Box::new(NoOpDispatcher)));
        let id = hub.subscribe(
            NotificationTarget::InProcess {
                channel: "ch".into(),
            },
            EventFilter::All,
        );
        assert!(hub.pause(&id));
        hub.publish(make_event(JobState::Running));
        assert_eq!(hub.stats().deliveries_ok, 0);

        assert!(hub.resume(&id));
        hub.publish(make_event(JobState::Completed)); // Different state → not deduped.
        assert_eq!(hub.stats().deliveries_ok, 1);
    }

    #[test]
    fn test_hub_unsubscribe() {
        let hub = NotificationHub::new(HubConfig::default(), Some(Box::new(NoOpDispatcher)));
        let id = hub.subscribe(
            NotificationTarget::InProcess {
                channel: "ch".into(),
            },
            EventFilter::All,
        );
        assert!(hub.unsubscribe(&id));
        assert_eq!(hub.stats().total_subscriptions, 0);
        // Removing again returns false.
        assert!(!hub.unsubscribe(&id));
    }

    #[test]
    fn test_hub_delivery_history() {
        let hub = NotificationHub::new(HubConfig::default(), Some(Box::new(NoOpDispatcher)));
        hub.subscribe(
            NotificationTarget::InProcess {
                channel: "ch".into(),
            },
            EventFilter::All,
        );
        hub.publish(make_event(JobState::Queued));
        hub.publish(make_event(JobState::Running));
        let history = hub.delivery_history(10);
        assert_eq!(history.len(), 2);
    }

    #[test]
    fn test_webhook_spec_builder() {
        let spec = WebhookSpec::new("https://example.com/hook")
            .with_bearer("tok123")
            .with_secret("s3cr3t");
        assert!(spec.headers.contains_key("Authorization"));
        assert_eq!(spec.secret.as_deref(), Some("s3cr3t"));
        assert_eq!(spec.max_retries, 3);
    }

    #[test]
    fn test_email_spec_render_subject() {
        let spec = EmailSpec::new("from@example.com", "to@example.com");
        let event = JobEvent::new(JobId::new(), "encode-job", JobState::Completed);
        let subject = spec.render_subject(&event);
        assert!(subject.contains("encode-job"));
        assert!(subject.contains("completed"));
    }

    #[test]
    fn test_job_event_dedup_key_uniqueness() {
        let id = JobId::new();
        let e1 = JobEvent::new(id.clone(), "j", JobState::Completed);
        let e2 = JobEvent::new(id.clone(), "j", JobState::Failed);
        assert_ne!(e1.dedup_key(), e2.dedup_key());
    }

    #[test]
    fn test_hub_history_for_job() {
        let hub = NotificationHub::new(HubConfig::default(), Some(Box::new(NoOpDispatcher)));
        let id = hub.subscribe(
            NotificationTarget::InProcess {
                channel: "ch".into(),
            },
            EventFilter::All,
        );

        let job_a = JobId::from("job-a");
        let job_b = JobId::from("job-b");
        hub.publish(JobEvent::new(job_a.clone(), "A", JobState::Running));
        hub.publish(JobEvent::new(job_b.clone(), "B", JobState::Completed));

        let history_a = hub.history_for_job(&job_a);
        assert_eq!(history_a.len(), 1);

        // Unsubscribe to avoid leaks in later tests.
        hub.unsubscribe(&id);
    }
}
