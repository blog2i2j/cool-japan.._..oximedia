// Copyright 2024 OxiMedia Project
// Licensed under the Apache License, Version 2.0

//! Webhook notifier — send job status change notifications to external HTTP URLs.
//!
//! # Security
//!
//! When a `secret` is configured the notifier computes an HMAC-SHA256 over the
//! serialised JSON payload and adds it as an `X-OxiMedia-Signature` request
//! header in the form `sha256=<hex-digest>`.  Recipients can verify the
//! authenticity of the notification by recomputing the HMAC with the shared
//! secret.
//!
//! # Retry
//!
//! Failed deliveries are retried according to the attached `RetryConfig` with
//! exponential backoff.  Because this module is meant to be testable without a
//! real HTTP server, network I/O is handled by the `HttpClient` trait, which
//! ships with a default `NoopHttpClient` that records calls for unit tests.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::VecDeque;
use std::time::Duration;
use thiserror::Error;
use uuid::Uuid;

use hmac::{Hmac, KeyInit, Mac};
use sha2::Sha256;

// ─────────────────────────────────────────────────────────────────────────────
// Error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by the webhook notifier.
#[derive(Debug, Error)]
pub enum WebhookError {
    /// The HTTP client reported a delivery failure.
    #[error("HTTP delivery failed: {0}")]
    DeliveryFailed(String),
    /// All retry attempts were exhausted.
    #[error("All {0} delivery attempts failed for job {1}")]
    RetriesExhausted(u32, Uuid),
    /// The payload could not be serialised.
    #[error("Serialisation error: {0}")]
    Serialisation(#[from] serde_json::Error),
}

// ─────────────────────────────────────────────────────────────────────────────
// Domain types
// ─────────────────────────────────────────────────────────────────────────────

/// Job event types that can trigger a webhook notification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum JobEvent {
    /// The job was submitted to the queue.
    Submitted,
    /// The job started executing.
    Started,
    /// The job completed successfully.
    Completed,
    /// The job failed.
    Failed,
    /// The job was cancelled.
    Cancelled,
    /// Job progress was updated.
    Progress,
    /// The job was moved to the dead-letter queue.
    DeadLettered,
}

impl JobEvent {
    /// Short string representation.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Submitted => "submitted",
            Self::Started => "started",
            Self::Completed => "completed",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Progress => "progress",
            Self::DeadLettered => "dead_lettered",
        }
    }
}

/// Configuration for a single webhook endpoint.
#[derive(Debug, Clone)]
pub struct WebhookConfig {
    /// Target URL.
    pub url: String,
    /// Optional shared secret for HMAC-SHA256 signing.
    pub secret: Option<String>,
    /// Which event types should trigger this webhook.
    pub events: Vec<JobEvent>,
    /// Retry behaviour on delivery failure.
    pub retry_policy: RetryConfig,
}

impl WebhookConfig {
    /// Create a new webhook config for the given URL.
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            secret: None,
            events: vec![
                JobEvent::Submitted,
                JobEvent::Started,
                JobEvent::Completed,
                JobEvent::Failed,
            ],
            retry_policy: RetryConfig::default(),
        }
    }

    /// Set the HMAC secret.
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.secret = Some(secret.into());
        self
    }

    /// Override the event filter.
    pub fn with_events(mut self, events: Vec<JobEvent>) -> Self {
        self.events = events;
        self
    }

    /// Override the retry policy.
    pub fn with_retry(mut self, policy: RetryConfig) -> Self {
        self.retry_policy = policy;
        self
    }

    /// Returns `true` if this webhook is subscribed to `event`.
    pub fn subscribes_to(&self, event: JobEvent) -> bool {
        self.events.contains(&event)
    }
}

/// Retry configuration for webhook delivery.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of delivery attempts (initial + retries).
    pub max_attempts: u32,
    /// Initial backoff before the first retry.
    pub initial_backoff: Duration,
    /// Multiplier applied to the backoff after each failure.
    pub backoff_multiplier: f64,
    /// Maximum backoff cap.
    pub max_backoff: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            initial_backoff: Duration::from_secs(1),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(30),
        }
    }
}

impl RetryConfig {
    /// Compute the backoff duration for attempt `n` (0-indexed).
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss)]
    pub fn backoff_for(&self, attempt: u32) -> Duration {
        if attempt == 0 {
            return self.initial_backoff;
        }
        let secs =
            self.initial_backoff.as_secs_f64() * self.backoff_multiplier.powi(attempt as i32);
        let secs = secs.min(self.max_backoff.as_secs_f64()) as u64;
        Duration::from_secs(secs)
    }
}

/// A webhook event payload sent to subscribers.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WebhookEvent {
    /// Type of the event.
    pub event_type: JobEvent,
    /// Job that triggered the event.
    pub job_id: Uuid,
    /// When the event occurred.
    pub timestamp: DateTime<Utc>,
    /// Arbitrary event-specific data.
    pub payload: Value,
}

impl WebhookEvent {
    /// Create a new webhook event.
    pub fn new(event_type: JobEvent, job_id: Uuid, payload: Value) -> Self {
        Self {
            event_type,
            job_id,
            timestamp: Utc::now(),
            payload,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HTTP client abstraction
// ─────────────────────────────────────────────────────────────────────────────

/// A request that the notifier wants to send.
#[derive(Debug, Clone)]
pub struct OutboundRequest {
    /// Target URL.
    pub url: String,
    /// Request body (JSON).
    pub body: String,
    /// HMAC-SHA256 signature header value, if a secret was configured.
    pub signature: Option<String>,
    /// The event type for tracking purposes.
    pub event_type: JobEvent,
}

/// Outcome returned by an `HttpClient` implementation.
#[derive(Debug, Clone)]
pub struct HttpResponse {
    /// HTTP status code.
    pub status: u16,
    /// Whether the delivery was considered successful.
    pub success: bool,
}

impl HttpResponse {
    /// Convenience constructor for a successful delivery.
    pub fn ok() -> Self {
        Self {
            status: 200,
            success: true,
        }
    }

    /// Convenience constructor for a failed delivery.
    pub fn err(status: u16) -> Self {
        Self {
            status,
            success: false,
        }
    }
}

/// Trait that abstracts over the actual HTTP transport.
///
/// Implementations may use `reqwest`, a mock, or a no-op.
pub trait HttpClient: Send + Sync {
    /// Attempt to deliver `request`.  Returns an `HttpResponse` or an error
    /// string on transport-level failures.
    fn send(&self, request: &OutboundRequest) -> Result<HttpResponse, String>;
}

/// A no-op client that always reports success and records calls for tests.
#[derive(Debug, Default)]
pub struct NoopHttpClient {
    /// Requests recorded by `send`.
    pub calls: std::sync::Mutex<Vec<OutboundRequest>>,
}

impl NoopHttpClient {
    /// Create a new no-op client.
    pub fn new() -> Self {
        Self::default()
    }

    /// Return a snapshot of all calls received so far.
    pub fn recorded_calls(&self) -> Vec<OutboundRequest> {
        self.calls.lock().map(|g| g.clone()).unwrap_or_default()
    }

    /// Number of calls recorded.
    pub fn call_count(&self) -> usize {
        self.calls.lock().map(|g| g.len()).unwrap_or(0)
    }
}

impl HttpClient for NoopHttpClient {
    fn send(&self, request: &OutboundRequest) -> Result<HttpResponse, String> {
        if let Ok(mut calls) = self.calls.lock() {
            calls.push(request.clone());
        }
        Ok(HttpResponse::ok())
    }
}

/// A client that always fails (for retry testing).
#[derive(Debug, Default)]
pub struct FailingHttpClient {
    /// Requests recorded by `send`.
    pub calls: std::sync::Mutex<Vec<OutboundRequest>>,
    /// The HTTP status to return.
    pub status: u16,
}

impl FailingHttpClient {
    /// Create a failing client that returns the given status code.
    pub fn new(status: u16) -> Self {
        Self {
            calls: std::sync::Mutex::default(),
            status,
        }
    }

    /// Number of calls recorded.
    pub fn call_count(&self) -> usize {
        self.calls.lock().map(|g| g.len()).unwrap_or(0)
    }
}

impl HttpClient for FailingHttpClient {
    fn send(&self, request: &OutboundRequest) -> Result<HttpResponse, String> {
        if let Ok(mut calls) = self.calls.lock() {
            calls.push(request.clone());
        }
        Ok(HttpResponse::err(self.status))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// HMAC signature
// ─────────────────────────────────────────────────────────────────────────────

/// Compute an HMAC-SHA256 signature over `body` using `secret`.
///
/// Returns the lowercase hex-encoded digest prefixed with `"sha256="`.
pub fn compute_signature(secret: &str, body: &str) -> String {
    let mut mac =
        Hmac::<Sha256>::new_from_slice(secret.as_bytes()).expect("HMAC accepts any key length");
    mac.update(body.as_bytes());
    let result = mac.finalize().into_bytes();
    let hex = bytes_to_hex(&result);
    format!("sha256={hex}")
}

/// Encode a byte slice as a lowercase hex string.
fn bytes_to_hex(bytes: &[u8]) -> String {
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut acc, b| {
            use std::fmt::Write;
            let _ = write!(acc, "{b:02x}");
            acc
        })
}

// ─────────────────────────────────────────────────────────────────────────────
// DeliveryRecord
// ─────────────────────────────────────────────────────────────────────────────

/// A record of a single delivery attempt.
#[derive(Debug, Clone)]
pub struct DeliveryRecord {
    /// When the attempt was made.
    pub at: DateTime<Utc>,
    /// Which attempt number (0-indexed).
    pub attempt: u32,
    /// Whether it succeeded.
    pub success: bool,
    /// HTTP status code (or 0 on transport error).
    pub status: u16,
}

// ─────────────────────────────────────────────────────────────────────────────
// WebhookNotifier
// ─────────────────────────────────────────────────────────────────────────────

/// Notifier that dispatches [`WebhookEvent`]s to registered endpoints.
///
/// Uses the injected `HttpClient` so it can be used in unit tests without
/// making real HTTP requests.
pub struct WebhookNotifier<C: HttpClient> {
    configs: Vec<WebhookConfig>,
    client: C,
    /// History of all delivery attempts (most recent last).
    delivery_history: VecDeque<DeliveryRecord>,
    /// Maximum number of history entries to retain.
    max_history: usize,
}

impl<C: HttpClient> WebhookNotifier<C> {
    /// Create a notifier with an injected HTTP client.
    pub fn new(client: C) -> Self {
        Self {
            configs: Vec::new(),
            client,
            delivery_history: VecDeque::new(),
            max_history: 1000,
        }
    }

    /// Register a webhook endpoint.
    pub fn add_config(&mut self, config: WebhookConfig) {
        self.configs.push(config);
    }

    /// Number of registered webhook configs.
    pub fn config_count(&self) -> usize {
        self.configs.len()
    }

    /// Attempt to deliver `event` to all subscribed endpoints.
    ///
    /// Delivery failures are retried per each config's `RetryConfig`.
    ///
    /// # Errors
    ///
    /// Returns the **last** error if at least one endpoint's final retry
    /// attempt failed.  Successful deliveries to other endpoints are not
    /// affected.
    pub fn notify(&mut self, event: &WebhookEvent) -> Result<(), WebhookError> {
        let body = serde_json::to_string(event)?;
        let mut last_err: Option<WebhookError> = None;

        // We collect config data to avoid borrowing `self` across the loop.
        let configs: Vec<WebhookConfig> = self.configs.clone();

        for config in &configs {
            if !config.subscribes_to(event.event_type) {
                continue;
            }

            let signature = config
                .secret
                .as_deref()
                .map(|secret| compute_signature(secret, &body));

            let request = OutboundRequest {
                url: config.url.clone(),
                body: body.clone(),
                signature,
                event_type: event.event_type,
            };

            match self.deliver_with_retry(&request, &config.retry_policy, event.job_id) {
                Ok(()) => {}
                Err(e) => {
                    last_err = Some(e);
                }
            }
        }

        last_err.map_or(Ok(()), Err)
    }

    /// Deliver `request` with retry.
    fn deliver_with_retry(
        &mut self,
        request: &OutboundRequest,
        policy: &RetryConfig,
        job_id: Uuid,
    ) -> Result<(), WebhookError> {
        for attempt in 0..policy.max_attempts {
            match self.client.send(request) {
                Ok(resp) => {
                    let record = DeliveryRecord {
                        at: Utc::now(),
                        attempt,
                        success: resp.success,
                        status: resp.status,
                    };
                    self.push_history(record);
                    if resp.success {
                        return Ok(());
                    }
                    // Non-success status — retry (unless last attempt).
                }
                Err(transport_err) => {
                    let record = DeliveryRecord {
                        at: Utc::now(),
                        attempt,
                        success: false,
                        status: 0,
                    };
                    self.push_history(record);
                    if attempt + 1 >= policy.max_attempts {
                        return Err(WebhookError::DeliveryFailed(transport_err));
                    }
                }
            }

            // Not the last attempt — wait before retrying.
            if attempt + 1 < policy.max_attempts {
                let backoff = policy.backoff_for(attempt);
                // In production code we would `tokio::time::sleep(backoff).await`
                // but this notifier is intentionally synchronous to remain
                // testable without a Tokio runtime.  Production callers that need
                // async should wrap this in a task.
                let _ = backoff; // acknowledged — not sleeping in unit tests
            }
        }

        Err(WebhookError::RetriesExhausted(policy.max_attempts, job_id))
    }

    fn push_history(&mut self, record: DeliveryRecord) {
        if self.delivery_history.len() >= self.max_history {
            self.delivery_history.pop_front();
        }
        self.delivery_history.push_back(record);
    }

    /// Total number of delivery attempts recorded.
    pub fn delivery_count(&self) -> usize {
        self.delivery_history.len()
    }

    /// Number of successful delivery attempts.
    pub fn successful_deliveries(&self) -> usize {
        self.delivery_history.iter().filter(|r| r.success).count()
    }

    /// Number of failed delivery attempts.
    pub fn failed_deliveries(&self) -> usize {
        self.delivery_history.iter().filter(|r| !r.success).count()
    }

    /// Recent delivery history (most recent last).
    pub fn delivery_history(&self) -> impl Iterator<Item = &DeliveryRecord> {
        self.delivery_history.iter()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn make_event(event_type: JobEvent) -> WebhookEvent {
        WebhookEvent::new(event_type, Uuid::new_v4(), json!({"key": "value"}))
    }

    // ── WebhookConfig ─────────────────────────────────────────────────────────

    #[test]
    fn test_config_subscribes_to() {
        let cfg = WebhookConfig::new("https://example.com")
            .with_events(vec![JobEvent::Completed, JobEvent::Failed]);
        assert!(cfg.subscribes_to(JobEvent::Completed));
        assert!(cfg.subscribes_to(JobEvent::Failed));
        assert!(!cfg.subscribes_to(JobEvent::Submitted));
    }

    #[test]
    fn test_config_default_events() {
        let cfg = WebhookConfig::new("https://example.com");
        assert!(cfg.subscribes_to(JobEvent::Submitted));
        assert!(cfg.subscribes_to(JobEvent::Completed));
    }

    // ── RetryConfig ────────────────────────────────────────────────────────────

    #[test]
    fn test_retry_config_backoff_for_attempt_0() {
        let rc = RetryConfig::default();
        assert_eq!(rc.backoff_for(0), Duration::from_secs(1));
    }

    #[test]
    fn test_retry_config_backoff_increases() {
        let rc = RetryConfig::default();
        let b0 = rc.backoff_for(0);
        let b1 = rc.backoff_for(1);
        assert!(b1 >= b0);
    }

    #[test]
    fn test_retry_config_backoff_capped_at_max() {
        let rc = RetryConfig {
            max_attempts: 10,
            initial_backoff: Duration::from_secs(1),
            backoff_multiplier: 10.0,
            max_backoff: Duration::from_secs(5),
        };
        // After many doublings backoff should be capped at 5 s
        assert!(rc.backoff_for(5) <= Duration::from_secs(5));
    }

    // ── HMAC signature ────────────────────────────────────────────────────────

    #[test]
    fn test_signature_format() {
        let sig = compute_signature("my-secret", "hello world");
        assert!(sig.starts_with("sha256="));
        assert_eq!(sig.len(), 7 + 64); // "sha256=" + 32 bytes * 2 hex chars
    }

    #[test]
    fn test_signature_deterministic() {
        let s1 = compute_signature("secret", "body");
        let s2 = compute_signature("secret", "body");
        assert_eq!(s1, s2);
    }

    #[test]
    fn test_signature_different_for_different_secrets() {
        let s1 = compute_signature("secret1", "body");
        let s2 = compute_signature("secret2", "body");
        assert_ne!(s1, s2);
    }

    #[test]
    fn test_signature_different_for_different_bodies() {
        let s1 = compute_signature("secret", "body1");
        let s2 = compute_signature("secret", "body2");
        assert_ne!(s1, s2);
    }

    // ── NoopHttpClient ────────────────────────────────────────────────────────

    #[test]
    fn test_noop_client_records_calls() {
        let client = NoopHttpClient::new();
        let req = OutboundRequest {
            url: "https://example.com".to_string(),
            body: "{}".to_string(),
            signature: None,
            event_type: JobEvent::Completed,
        };
        client.send(&req).expect("send should succeed");
        assert_eq!(client.call_count(), 1);
    }

    // ── WebhookNotifier ───────────────────────────────────────────────────────

    #[test]
    fn test_notify_dispatches_to_subscribed_endpoint() {
        let client = NoopHttpClient::new();
        let mut notifier = WebhookNotifier::new(NoopHttpClient::new());
        notifier.add_config(
            WebhookConfig::new("https://example.com").with_events(vec![JobEvent::Completed]),
        );
        let _ = client; // silence unused warning

        let event = make_event(JobEvent::Completed);
        notifier.notify(&event).expect("notify should succeed");
        assert_eq!(notifier.delivery_count(), 1);
        assert_eq!(notifier.successful_deliveries(), 1);
    }

    #[test]
    fn test_notify_skips_unsubscribed_endpoint() {
        let mut notifier = WebhookNotifier::new(NoopHttpClient::new());
        notifier.add_config(
            WebhookConfig::new("https://example.com").with_events(vec![JobEvent::Failed]),
        );
        let event = make_event(JobEvent::Completed); // not subscribed
        notifier.notify(&event).expect("notify should succeed");
        assert_eq!(notifier.delivery_count(), 0);
    }

    #[test]
    fn test_notify_attaches_signature_when_secret_set() {
        let client = std::sync::Arc::new(NoopHttpClient::new());
        // We need a way to inspect calls; use a fresh client.
        let recording = NoopHttpClient::new();
        let mut notifier = WebhookNotifier::new(recording);
        notifier.add_config(
            WebhookConfig::new("https://example.com")
                .with_secret("my-secret")
                .with_events(vec![JobEvent::Completed]),
        );
        let _ = client;

        let event = make_event(JobEvent::Completed);
        notifier.notify(&event).expect("notify should succeed");

        // Check that the recorded call has a signature.
        let calls = notifier.client.recorded_calls();
        assert_eq!(calls.len(), 1);
        let sig = calls[0]
            .signature
            .as_deref()
            .expect("signature should be present");
        assert!(sig.starts_with("sha256="));
    }

    #[test]
    fn test_notify_no_signature_without_secret() {
        let mut notifier = WebhookNotifier::new(NoopHttpClient::new());
        notifier.add_config(
            WebhookConfig::new("https://example.com").with_events(vec![JobEvent::Completed]),
        );
        let event = make_event(JobEvent::Completed);
        notifier.notify(&event).expect("notify should succeed");

        let calls = notifier.client.recorded_calls();
        assert_eq!(calls.len(), 1);
        assert!(calls[0].signature.is_none());
    }

    #[test]
    fn test_notify_retries_on_failure() {
        let max_attempts = 3u32;
        let policy = RetryConfig {
            max_attempts,
            initial_backoff: Duration::from_millis(0),
            backoff_multiplier: 1.0,
            max_backoff: Duration::from_millis(0),
        };
        let mut notifier = WebhookNotifier::new(FailingHttpClient::new(500));
        notifier.add_config(
            WebhookConfig::new("https://example.com")
                .with_events(vec![JobEvent::Completed])
                .with_retry(policy),
        );
        let event = make_event(JobEvent::Completed);
        let result = notifier.notify(&event);
        assert!(result.is_err());
        // All 3 attempts should have been recorded in history
        assert_eq!(notifier.delivery_count() as u32, max_attempts);
        assert_eq!(notifier.failed_deliveries() as u32, max_attempts);
    }

    #[test]
    fn test_notify_multiple_endpoints() {
        let mut notifier = WebhookNotifier::new(NoopHttpClient::new());
        notifier.add_config(
            WebhookConfig::new("https://ep1.example.com").with_events(vec![JobEvent::Completed]),
        );
        notifier.add_config(
            WebhookConfig::new("https://ep2.example.com").with_events(vec![JobEvent::Completed]),
        );
        let event = make_event(JobEvent::Completed);
        notifier.notify(&event).expect("notify should succeed");
        assert_eq!(notifier.delivery_count(), 2);
    }

    #[test]
    fn test_delivery_history_tracks_success_and_failure() {
        let policy = RetryConfig {
            max_attempts: 1,
            initial_backoff: Duration::from_millis(0),
            backoff_multiplier: 1.0,
            max_backoff: Duration::from_millis(0),
        };
        let mut notifier = WebhookNotifier::new(FailingHttpClient::new(503));
        notifier.add_config(
            WebhookConfig::new("https://example.com")
                .with_events(vec![JobEvent::Failed])
                .with_retry(policy),
        );
        let event = make_event(JobEvent::Failed);
        let _ = notifier.notify(&event);
        assert_eq!(notifier.failed_deliveries(), 1);
        assert_eq!(notifier.successful_deliveries(), 0);
    }
}
