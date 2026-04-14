//! HTTP webhook trigger system for OxiMedia workflows.
//!
//! This module provides a self-contained webhook trigger registry that maps
//! incoming HTTP requests to workflow IDs. It is transport-agnostic: callers
//! construct a [`WebhookPayload`] from whatever HTTP framework they use and
//! pass it to [`WebhookTriggerRegistry::match_trigger`].
//!
//! # Signature verification
//!
//! When a [`WebhookTrigger`] is configured with a `secret_token`, the registry
//! validates the `X-Signature` request header against a deterministic XOR-based
//! MAC of the raw body. The MAC is **not** cryptographically secure (it is a
//! lightweight stub for integration testing); production deployments should use
//! the full HMAC-SHA256 implementation in [`crate::triggers`].
//!
//! # Example
//!
//! ```rust
//! use oximedia_workflow::webhook_trigger::{
//!     WebhookTrigger, WebhookTriggerRegistry, WebhookPayload,
//! };
//!
//! let mut registry = WebhookTriggerRegistry::new();
//! registry.add_trigger(WebhookTrigger {
//!     id: "t1".to_string(),
//!     path: "/webhooks/ingest".to_string(),
//!     workflow_id: "wf-ingest".to_string(),
//!     secret_token: None,
//! });
//!
//! let payload = WebhookPayload {
//!     method: "POST".to_string(),
//!     path: "/webhooks/ingest".to_string(),
//!     body: "{}".to_string(),
//!     headers: vec![],
//! };
//!
//! let trigger = registry.match_trigger(&payload);
//! assert!(trigger.is_some());
//! assert_eq!(trigger.unwrap().workflow_id, "wf-ingest");
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// WebhookTrigger
// ---------------------------------------------------------------------------

/// A registered webhook trigger that binds an HTTP path to a workflow.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebhookTrigger {
    /// Unique identifier for this trigger.
    pub id: String,
    /// HTTP path that activates the trigger (e.g. `"/webhooks/ingest-ready"`).
    pub path: String,
    /// ID of the workflow to start when this trigger fires.
    pub workflow_id: String,
    /// Optional shared secret used to validate `X-Signature` headers.
    ///
    /// When set the header value must match `sha256=<xor_mac(body, secret)>`.
    pub secret_token: Option<String>,
}

impl WebhookTrigger {
    /// Create a new trigger without a secret.
    #[must_use]
    pub fn new(
        id: impl Into<String>,
        path: impl Into<String>,
        workflow_id: impl Into<String>,
    ) -> Self {
        Self {
            id: id.into(),
            path: path.into(),
            workflow_id: workflow_id.into(),
            secret_token: None,
        }
    }

    /// Attach a shared secret to this trigger.
    #[must_use]
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.secret_token = Some(secret.into());
        self
    }
}

// ---------------------------------------------------------------------------
// WebhookPayload
// ---------------------------------------------------------------------------

/// Represents an incoming HTTP request delivered to a webhook endpoint.
#[derive(Debug, Clone)]
pub struct WebhookPayload {
    /// HTTP method (e.g. `"POST"`).
    pub method: String,
    /// URL path of the request.
    pub path: String,
    /// Raw request body as a UTF-8 string.
    pub body: String,
    /// Request headers as `(name, value)` pairs.  Header names should be
    /// lower-cased for case-insensitive comparison.
    pub headers: Vec<(String, String)>,
}

impl WebhookPayload {
    /// Look up a header value by name (case-insensitive).
    #[must_use]
    pub fn header(&self, name: &str) -> Option<&str> {
        let lower = name.to_lowercase();
        self.headers
            .iter()
            .find(|(k, _)| k.to_lowercase() == lower)
            .map(|(_, v)| v.as_str())
    }
}

// ---------------------------------------------------------------------------
// WebhookTriggerRegistry
// ---------------------------------------------------------------------------

/// In-process registry that matches incoming [`WebhookPayload`]s to registered
/// [`WebhookTrigger`]s.
#[derive(Debug, Default)]
pub struct WebhookTriggerRegistry {
    /// All registered triggers, in insertion order.
    pub triggers: Vec<WebhookTrigger>,
}

impl WebhookTriggerRegistry {
    /// Create an empty registry.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a new trigger.  If a trigger with the same `id` already exists
    /// it is replaced.
    pub fn add_trigger(&mut self, trigger: WebhookTrigger) {
        if let Some(pos) = self.triggers.iter().position(|t| t.id == trigger.id) {
            self.triggers[pos] = trigger;
        } else {
            self.triggers.push(trigger);
        }
    }

    /// Remove the trigger with the given id, returning it if present.
    pub fn remove_trigger(&mut self, id: &str) -> Option<WebhookTrigger> {
        if let Some(pos) = self.triggers.iter().position(|t| t.id == id) {
            Some(self.triggers.remove(pos))
        } else {
            None
        }
    }

    /// Return the first trigger whose `path` exactly matches
    /// `payload.path`.
    ///
    /// If the matching trigger has a `secret_token`, the `X-Signature` header
    /// of the payload is also validated via [`validate_signature`].  A trigger
    /// whose signature check **fails** is skipped (the next matching trigger is
    /// tried instead).
    #[must_use]
    pub fn match_trigger(&self, payload: &WebhookPayload) -> Option<&WebhookTrigger> {
        for trigger in &self.triggers {
            if trigger.path != payload.path {
                continue;
            }
            // If a secret is configured, validate the signature header.
            if let Some(ref secret) = trigger.secret_token {
                if !validate_signature(payload, secret) {
                    continue;
                }
            }
            return Some(trigger);
        }
        None
    }

    /// Return the number of registered triggers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.triggers.len()
    }

    /// Return `true` when no triggers are registered.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.triggers.is_empty()
    }
}

// ---------------------------------------------------------------------------
// XOR-based MAC (stub — not cryptographically secure)
// ---------------------------------------------------------------------------

/// Compute a lightweight 64-bit XOR-based MAC of `body` keyed with `secret`.
///
/// The algorithm folds each byte of `body` into a 64-bit accumulator using
/// XOR and a FNV-inspired mixing step, then XORs in bytes of the key cyclically.
/// This is **not** a secure MAC; it is a deterministic stub for use in tests
/// and local development.
#[must_use]
pub fn xor_mac(body: &str, secret: &str) -> u64 {
    let body_bytes = body.as_bytes();
    let key_bytes = secret.as_bytes();

    if body_bytes.is_empty() || key_bytes.is_empty() {
        return 0;
    }

    let mut acc: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis

    for (i, &b) in body_bytes.iter().enumerate() {
        let key_byte = key_bytes[i % key_bytes.len()];
        acc ^= u64::from(b) ^ u64::from(key_byte);
        // FNV-like mixing
        acc = acc.wrapping_mul(0x0000_0100_0000_01b3);
    }
    acc
}

/// Validate the `X-Signature` header of a [`WebhookPayload`] against a secret.
///
/// The expected header format is `sha256=<hex-encoded xor_mac>`.
///
/// Returns `true` when:
/// - The header is present and well-formed, **and**
/// - The computed MAC of `payload.body` with `secret` matches the header value.
///
/// Returns `false` in all other cases (missing header, bad format, wrong MAC).
#[must_use]
pub fn validate_signature(payload: &WebhookPayload, secret: &str) -> bool {
    let Some(sig_header) = payload.header("x-signature") else {
        return false;
    };

    let mac_hex = if let Some(hex) = sig_header.strip_prefix("sha256=") {
        hex
    } else {
        sig_header
    };

    let expected = xor_mac(&payload.body, secret);
    let expected_hex = format!("{expected:016x}");

    // Constant-time comparison (lengths equal → byte XOR fold).
    let a = expected_hex.as_bytes();
    let b = mac_hex.as_bytes();
    if a.len() != b.len() {
        return false;
    }
    let diff = a
        .iter()
        .zip(b.iter())
        .fold(0u8, |acc, (x, y)| acc | (x ^ y));
    diff == 0
}

// ===========================================================================
// SimpleWebhookTrigger — URL-centric fire-and-report API
// ===========================================================================

/// A lightweight webhook trigger keyed only by destination URL.
///
/// Unlike [`WebhookTrigger`] (which is path-routed), `SimpleWebhookTrigger`
/// holds the full URL and exposes a single [`fire`](Self::fire) method that
/// returns a human-readable report string describing the event it would
/// dispatch.  No actual HTTP call is made; the implementation is deliberately
/// synchronous and allocation-free for use in tests and offline pipelines.
///
/// # Example
///
/// ```rust
/// use oximedia_workflow::webhook_trigger::SimpleWebhookTrigger;
///
/// let trigger = SimpleWebhookTrigger::new("https://example.com/hooks/ingest");
/// let report = trigger.fire("job.complete");
/// assert!(report.contains("job.complete"));
/// assert!(report.contains("example.com"));
/// ```
#[derive(Debug, Clone)]
pub struct SimpleWebhookTrigger {
    /// Destination URL for the webhook.
    pub url: String,
}

impl SimpleWebhookTrigger {
    /// Create a new simple webhook trigger for the given URL.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self { url: url.into() }
    }

    /// Simulate firing a webhook event.
    ///
    /// Returns a report string: `"WEBHOOK {url} event={event}"`.
    /// No real HTTP request is made.
    #[must_use]
    pub fn fire(&self, event: &str) -> String {
        format!("WEBHOOK {} event={event}", self.url)
    }

    /// Return the target URL.
    #[must_use]
    pub fn url(&self) -> &str {
        &self.url
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // xor_mac
    // ------------------------------------------------------------------

    #[test]
    fn xor_mac_empty_body_returns_zero() {
        assert_eq!(xor_mac("", "secret"), 0);
    }

    #[test]
    fn xor_mac_empty_secret_returns_zero() {
        assert_eq!(xor_mac("hello", ""), 0);
    }

    #[test]
    fn xor_mac_deterministic() {
        let a = xor_mac("payload body", "my-secret");
        let b = xor_mac("payload body", "my-secret");
        assert_eq!(a, b);
    }

    #[test]
    fn xor_mac_different_secrets_differ() {
        let a = xor_mac("same body", "secret-a");
        let b = xor_mac("same body", "secret-b");
        assert_ne!(a, b);
    }

    #[test]
    fn xor_mac_different_bodies_differ() {
        let a = xor_mac("body-a", "secret");
        let b = xor_mac("body-b", "secret");
        assert_ne!(a, b);
    }

    // ------------------------------------------------------------------
    // validate_signature
    // ------------------------------------------------------------------

    fn make_payload_with_sig(body: &str, secret: &str) -> WebhookPayload {
        let mac = xor_mac(body, secret);
        let header_val = format!("sha256={mac:016x}");
        WebhookPayload {
            method: "POST".to_string(),
            path: "/test".to_string(),
            body: body.to_string(),
            headers: vec![("x-signature".to_string(), header_val)],
        }
    }

    #[test]
    fn validate_signature_valid() {
        let payload = make_payload_with_sig("hello world", "my-secret");
        assert!(validate_signature(&payload, "my-secret"));
    }

    #[test]
    fn validate_signature_wrong_secret() {
        let payload = make_payload_with_sig("hello world", "correct-secret");
        assert!(!validate_signature(&payload, "wrong-secret"));
    }

    #[test]
    fn validate_signature_missing_header() {
        let payload = WebhookPayload {
            method: "POST".to_string(),
            path: "/test".to_string(),
            body: "hello".to_string(),
            headers: vec![],
        };
        assert!(!validate_signature(&payload, "secret"));
    }

    #[test]
    fn validate_signature_malformed_header() {
        let payload = WebhookPayload {
            method: "POST".to_string(),
            path: "/test".to_string(),
            body: "hello".to_string(),
            headers: vec![("x-signature".to_string(), "notahex!!".to_string())],
        };
        assert!(!validate_signature(&payload, "secret"));
    }

    // ------------------------------------------------------------------
    // WebhookPayload::header
    // ------------------------------------------------------------------

    #[test]
    fn header_lookup_case_insensitive() {
        let p = WebhookPayload {
            method: "POST".to_string(),
            path: "/".to_string(),
            body: String::new(),
            headers: vec![("Content-Type".to_string(), "application/json".to_string())],
        };
        assert_eq!(p.header("content-type"), Some("application/json"));
        assert_eq!(p.header("CONTENT-TYPE"), Some("application/json"));
        assert!(p.header("x-missing").is_none());
    }

    // ------------------------------------------------------------------
    // WebhookTriggerRegistry
    // ------------------------------------------------------------------

    #[test]
    fn registry_match_by_path() {
        let mut reg = WebhookTriggerRegistry::new();
        reg.add_trigger(WebhookTrigger::new("t1", "/hook/ingest", "wf-001"));
        reg.add_trigger(WebhookTrigger::new("t2", "/hook/export", "wf-002"));

        let payload = WebhookPayload {
            method: "POST".to_string(),
            path: "/hook/ingest".to_string(),
            body: "{}".to_string(),
            headers: vec![],
        };
        let m = reg.match_trigger(&payload);
        assert!(m.is_some());
        assert_eq!(m.unwrap().workflow_id, "wf-001");
    }

    #[test]
    fn registry_no_match_returns_none() {
        let mut reg = WebhookTriggerRegistry::new();
        reg.add_trigger(WebhookTrigger::new("t1", "/hook/known", "wf-001"));

        let payload = WebhookPayload {
            method: "POST".to_string(),
            path: "/hook/unknown".to_string(),
            body: "{}".to_string(),
            headers: vec![],
        };
        assert!(reg.match_trigger(&payload).is_none());
    }

    #[test]
    fn registry_secret_match_with_valid_signature() {
        let secret = "top-secret";
        let body = r#"{"event":"ingest_done"}"#;
        let mac = xor_mac(body, secret);
        let sig_header = format!("sha256={mac:016x}");

        let mut reg = WebhookTriggerRegistry::new();
        reg.add_trigger(WebhookTrigger::new("t1", "/secure", "wf-secure").with_secret(secret));

        let payload = WebhookPayload {
            method: "POST".to_string(),
            path: "/secure".to_string(),
            body: body.to_string(),
            headers: vec![("x-signature".to_string(), sig_header)],
        };

        let m = reg.match_trigger(&payload);
        assert!(m.is_some(), "should match with valid signature");
        assert_eq!(m.unwrap().id, "t1");
    }

    #[test]
    fn registry_secret_match_fails_with_wrong_signature() {
        let mut reg = WebhookTriggerRegistry::new();
        reg.add_trigger(
            WebhookTrigger::new("t1", "/secure", "wf-secure").with_secret("correct-secret"),
        );

        let payload = WebhookPayload {
            method: "POST".to_string(),
            path: "/secure".to_string(),
            body: "{}".to_string(),
            headers: vec![(
                "x-signature".to_string(),
                "sha256=deadbeef00000000".to_string(),
            )],
        };

        assert!(
            reg.match_trigger(&payload).is_none(),
            "should not match with wrong signature"
        );
    }

    #[test]
    fn registry_add_replaces_existing_id() {
        let mut reg = WebhookTriggerRegistry::new();
        reg.add_trigger(WebhookTrigger::new("t1", "/old-path", "wf-old"));
        reg.add_trigger(WebhookTrigger::new("t1", "/new-path", "wf-new"));

        assert_eq!(reg.len(), 1);
        assert_eq!(reg.triggers[0].path, "/new-path");
    }

    #[test]
    fn registry_remove_trigger() {
        let mut reg = WebhookTriggerRegistry::new();
        reg.add_trigger(WebhookTrigger::new("t1", "/hook", "wf-001"));
        reg.add_trigger(WebhookTrigger::new("t2", "/hook2", "wf-002"));
        assert_eq!(reg.len(), 2);

        let removed = reg.remove_trigger("t1");
        assert!(removed.is_some());
        assert_eq!(reg.len(), 1);

        let not_found = reg.remove_trigger("t999");
        assert!(not_found.is_none());
    }

    #[test]
    fn registry_is_empty() {
        let mut reg = WebhookTriggerRegistry::new();
        assert!(reg.is_empty());
        reg.add_trigger(WebhookTrigger::new("t1", "/x", "wf-x"));
        assert!(!reg.is_empty());
    }
}
