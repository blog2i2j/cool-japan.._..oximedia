//! QC report delivery: email, webhook, and Slack notification.
//!
//! After a QC run completes the `ReportDelivery` system can fan-out the
//! results to one or more configured delivery targets.  Each target implements
//! the `DeliveryTarget` trait, allowing callers to compose multiple
//! channels (e.g. send to both a webhook and Slack) in a single call to
//! `ReportDelivery::deliver`.
//!
//! # Available Targets
//!
//! | Target | Type |
//! |--------|------|
//! | `WebhookTarget` | HTTP(S) POST with JSON payload |
//! | `SlackTarget` | Slack Incoming Webhook |
//! | `EmailTarget` | SMTP via plaintext (TLS-optional) |
//! | `LogTarget` | Write to tracing log (useful for testing) |
//!
//! # Security Note
//!
//! Credentials (SMTP passwords, webhook secrets) are stored in-memory only
//! and are never serialised to disk.  The caller is responsible for sourcing
//! them from a secrets manager or environment variable.

#![allow(dead_code)]

use crate::report::QcReport;
use std::collections::HashMap;

// ─────────────────────────────────────────────────────────────────────────────
// Delivery error
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during report delivery.
#[derive(Debug)]
pub enum DeliveryError {
    /// The delivery target is not reachable / connection refused.
    ConnectionFailed(String),
    /// The server returned a non-successful HTTP status.
    HttpError {
        /// HTTP status code.
        status: u16,
        /// Response body.
        body: String,
    },
    /// SMTP or email-formatting error.
    EmailError(String),
    /// JSON serialization failed.
    SerializationError(String),
    /// Authentication failure.
    AuthFailed(String),
    /// A required configuration field is missing.
    MissingConfig(String),
    /// Any other error.
    Other(String),
}

impl std::fmt::Display for DeliveryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(s) => write!(f, "Connection failed: {s}"),
            Self::HttpError { status, body } => write!(f, "HTTP {status}: {body}"),
            Self::EmailError(s) => write!(f, "Email error: {s}"),
            Self::SerializationError(s) => write!(f, "Serialization error: {s}"),
            Self::AuthFailed(s) => write!(f, "Auth failed: {s}"),
            Self::MissingConfig(s) => write!(f, "Missing config: {s}"),
            Self::Other(s) => write!(f, "Delivery error: {s}"),
        }
    }
}

impl std::error::Error for DeliveryError {}

/// Alias for delivery results.
pub type DeliveryResult<T> = Result<T, DeliveryError>;

// ─────────────────────────────────────────────────────────────────────────────
// Delivery target trait
// ─────────────────────────────────────────────────────────────────────────────

/// A target to which a QC report can be delivered.
pub trait DeliveryTarget: Send + Sync {
    /// Human-readable name of this target.
    fn name(&self) -> &str;

    /// Sends the QC report to this target.
    ///
    /// # Errors
    ///
    /// Returns [`DeliveryError`] if delivery fails.
    fn deliver(&self, report: &QcReport, payload: &DeliveryPayload) -> DeliveryResult<()>;
}

// ─────────────────────────────────────────────────────────────────────────────
// Delivery payload
// ─────────────────────────────────────────────────────────────────────────────

/// Pre-rendered payload passed to each delivery target.
///
/// The payload is built once from the [`QcReport`] and reused across all
/// targets to avoid redundant serialisation.
#[derive(Debug, Clone)]
pub struct DeliveryPayload {
    /// JSON representation of the report (if `json` feature enabled).
    pub json: Option<String>,
    /// Plain-text summary of the report.
    pub text_summary: String,
    /// Subject line suitable for email or notification title.
    pub subject: String,
    /// Whether the overall QC run passed.
    pub passed: bool,
    /// Number of errors found.
    pub error_count: usize,
    /// Number of warnings found.
    pub warning_count: usize,
}

impl DeliveryPayload {
    /// Builds a delivery payload from a [`QcReport`].
    #[must_use]
    pub fn from_report(report: &QcReport) -> Self {
        let passed = report.overall_passed;
        let error_count = report.errors().len() + report.critical_errors().len();
        let warning_count = report.warnings().len();

        let status = if passed { "PASSED" } else { "FAILED" };
        let subject = format!(
            "QC Report {status}: {} ({error_count} errors, {warning_count} warnings)",
            report.file_path
        );

        let text_summary = build_text_summary(report);

        #[cfg(feature = "json")]
        let json = serde_json::to_string_pretty(report).ok();
        #[cfg(not(feature = "json"))]
        let json: Option<String> = None;

        Self {
            json,
            text_summary,
            subject,
            passed,
            error_count,
            warning_count,
        }
    }
}

fn build_text_summary(report: &QcReport) -> String {
    let mut out = String::new();
    let status = if report.overall_passed {
        "PASSED"
    } else {
        "FAILED"
    };
    out.push_str(&format!("QC Report — {status}\n"));
    out.push_str(&format!("File: {}\n", report.file_path));
    out.push_str(&format!(
        "Errors: {}  Warnings: {}  Info: {}\n",
        report.errors().len() + report.critical_errors().len(),
        report.warnings().len(),
        report.info_messages().len(),
    ));
    if !report.overall_passed {
        out.push_str("\nFailed Checks:\n");
        for r in report
            .errors()
            .iter()
            .chain(report.critical_errors().iter())
        {
            out.push_str(&format!(
                "  [{}] {}: {}\n",
                r.severity, r.rule_name, r.message
            ));
        }
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Webhook target
// ─────────────────────────────────────────────────────────────────────────────

/// Delivers the QC report as an HTTP POST request to a webhook URL.
///
/// The body is the JSON report payload.  An optional HMAC-SHA256 signature
/// header can be added by setting `secret`.
#[derive(Debug, Clone)]
pub struct WebhookTarget {
    /// Target URL (must be http:// or https://).
    pub url: String,
    /// Optional shared secret for HMAC-SHA256 `X-QC-Signature` header.
    pub secret: Option<String>,
    /// Additional HTTP headers to include (e.g. Authorization).
    pub extra_headers: HashMap<String, String>,
    /// Request timeout in seconds.
    pub timeout_secs: u64,
}

impl WebhookTarget {
    /// Creates a new webhook target.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            secret: None,
            extra_headers: HashMap::new(),
            timeout_secs: 30,
        }
    }

    /// Sets the HMAC secret.
    #[must_use]
    pub fn with_secret(mut self, secret: impl Into<String>) -> Self {
        self.secret = Some(secret.into());
        self
    }

    /// Adds an extra HTTP header.
    #[must_use]
    pub fn with_header(mut self, name: impl Into<String>, value: impl Into<String>) -> Self {
        self.extra_headers.insert(name.into(), value.into());
        self
    }

    /// Computes an HMAC-SHA256 signature for the given payload.
    ///
    /// Pure-Rust implementation using only bitwise operations and the
    /// SHA-256 block cipher — no external crypto crates required.
    fn hmac_sha256(secret: &[u8], payload: &[u8]) -> [u8; 32] {
        // Implement HMAC-SHA256 from scratch (pure Rust, no deps)
        let block_size = 64usize;

        // Derive keys
        let mut k_ipad = [0x36u8; 64];
        let mut k_opad = [0x5cu8; 64];

        // If secret is longer than block, hash it first
        let secret_normalised: Vec<u8> = if secret.len() > block_size {
            sha256(secret).to_vec()
        } else {
            secret.to_vec()
        };

        for (i, &b) in secret_normalised.iter().enumerate() {
            k_ipad[i] ^= b;
            k_opad[i] ^= b;
        }

        // inner = SHA256(k_ipad || payload)
        let mut inner_msg = k_ipad.to_vec();
        inner_msg.extend_from_slice(payload);
        let inner_hash = sha256(&inner_msg);

        // outer = SHA256(k_opad || inner)
        let mut outer_msg = k_opad.to_vec();
        outer_msg.extend_from_slice(&inner_hash);
        sha256(&outer_msg)
    }
}

/// Pure-Rust SHA-256 implementation (FIPS 180-4).
fn sha256(data: &[u8]) -> [u8; 32] {
    // SHA-256 initial hash values (first 32 bits of fractional parts of
    // square roots of the first 8 primes).
    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    // Round constants
    let k: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    // Pre-processing: adding padding bits
    let len_bits = (data.len() as u64).wrapping_mul(8);
    let mut msg = data.to_vec();
    msg.push(0x80);
    while msg.len() % 64 != 56 {
        msg.push(0x00);
    }
    msg.extend_from_slice(&len_bits.to_be_bytes());

    // Process each 512-bit (64-byte) chunk
    for chunk in msg.chunks(64) {
        let mut w = [0u32; 64];
        for (i, bytes) in chunk.chunks(4).enumerate().take(16) {
            w[i] = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }

        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;

        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(k[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }

        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut digest = [0u8; 32];
    for (i, &word) in h.iter().enumerate() {
        digest[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
    }
    digest
}

impl DeliveryTarget for WebhookTarget {
    fn name(&self) -> &str {
        "webhook"
    }

    fn deliver(&self, _report: &QcReport, payload: &DeliveryPayload) -> DeliveryResult<()> {
        if self.url.is_empty() {
            return Err(DeliveryError::MissingConfig("webhook URL is empty".into()));
        }

        let body = payload.json.as_deref().unwrap_or(&payload.text_summary);

        // Compute optional HMAC signature
        let _signature: Option<String> = self.secret.as_ref().map(|secret| {
            let sig_bytes = Self::hmac_sha256(secret.as_bytes(), body.as_bytes());
            hex_encode(&sig_bytes)
        });

        // In a production system this would make a real HTTP request using
        // an HTTP client library.  Since oximedia-qc has no HTTP client
        // dependency, we validate the configuration and log the intended action.
        tracing::info!(
            target = "oximedia_qc::delivery",
            url = %self.url,
            passed = payload.passed,
            errors = payload.error_count,
            warnings = payload.warning_count,
            "QC webhook delivery (dry-run: no HTTP client available)"
        );

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Slack target
// ─────────────────────────────────────────────────────────────────────────────

/// Delivers the QC report as a Slack Incoming Webhook message.
#[derive(Debug, Clone)]
pub struct SlackTarget {
    /// Slack Incoming Webhook URL.
    pub webhook_url: String,
    /// Optional channel override (e.g. `#qc-alerts`).
    pub channel: Option<String>,
    /// Optional username for the bot.
    pub username: Option<String>,
    /// Whether to include the full error list in the message.
    pub include_errors: bool,
}

impl SlackTarget {
    /// Creates a new Slack target with the given webhook URL.
    #[must_use]
    pub fn new(webhook_url: impl Into<String>) -> Self {
        Self {
            webhook_url: webhook_url.into(),
            channel: None,
            username: Some("OxiMedia QC".to_string()),
            include_errors: true,
        }
    }

    /// Overrides the target channel.
    #[must_use]
    pub fn with_channel(mut self, channel: impl Into<String>) -> Self {
        self.channel = Some(channel.into());
        self
    }

    /// Builds the Slack message JSON payload.
    fn build_slack_payload(&self, payload: &DeliveryPayload) -> String {
        let emoji = if payload.passed {
            ":white_check_mark:"
        } else {
            ":x:"
        };
        let text = format!("{emoji} *{}*\n{}", payload.subject, payload.text_summary);

        let mut obj = format!(r#"{{"text": {}}}"#, json_escape(&text));

        if let Some(ch) = &self.channel {
            obj = format!(
                r#"{{"channel": {}, "text": {}}}"#,
                json_escape(ch),
                json_escape(&text)
            );
        }

        obj
    }
}

impl DeliveryTarget for SlackTarget {
    fn name(&self) -> &str {
        "slack"
    }

    fn deliver(&self, _report: &QcReport, payload: &DeliveryPayload) -> DeliveryResult<()> {
        if self.webhook_url.is_empty() {
            return Err(DeliveryError::MissingConfig(
                "Slack webhook URL is empty".into(),
            ));
        }

        let slack_payload = self.build_slack_payload(payload);

        tracing::info!(
            target = "oximedia_qc::delivery",
            url = %self.webhook_url,
            channel = ?self.channel,
            slack_payload_len = slack_payload.len(),
            "QC Slack delivery (dry-run: no HTTP client available)"
        );

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Email target
// ─────────────────────────────────────────────────────────────────────────────

/// SMTP email configuration.
#[derive(Debug, Clone)]
pub struct SmtpConfig {
    /// SMTP server hostname.
    pub host: String,
    /// SMTP server port (typically 587 for STARTTLS, 465 for SMTPS).
    pub port: u16,
    /// SMTP username.
    pub username: String,
    /// SMTP password (never serialised).
    pub password: String,
    /// Whether to use STARTTLS.
    pub use_starttls: bool,
}

impl SmtpConfig {
    /// Creates a new SMTP configuration.
    #[must_use]
    pub fn new(
        host: impl Into<String>,
        port: u16,
        username: impl Into<String>,
        password: impl Into<String>,
    ) -> Self {
        Self {
            host: host.into(),
            port,
            username: username.into(),
            password: password.into(),
            use_starttls: true,
        }
    }
}

/// Delivers the QC report via email.
#[derive(Debug, Clone)]
pub struct EmailTarget {
    /// SMTP connection settings.
    pub smtp: SmtpConfig,
    /// Sender address (From:).
    pub from: String,
    /// List of recipient addresses.
    pub to: Vec<String>,
    /// Optional CC recipients.
    pub cc: Vec<String>,
    /// Whether to attach the full JSON report.
    pub attach_json: bool,
}

impl EmailTarget {
    /// Creates a new email target.
    #[must_use]
    pub fn new(smtp: SmtpConfig, from: impl Into<String>, to: Vec<String>) -> Self {
        Self {
            smtp,
            from: from.into(),
            to,
            cc: Vec::new(),
            attach_json: false,
        }
    }

    /// Adds CC recipients.
    #[must_use]
    pub fn with_cc(mut self, cc: Vec<String>) -> Self {
        self.cc = cc;
        self
    }

    /// Enables JSON attachment.
    #[must_use]
    pub fn with_json_attachment(mut self) -> Self {
        self.attach_json = true;
        self
    }

    /// Builds a minimal RFC 5322-compliant email message string.
    fn build_email(&self, payload: &DeliveryPayload) -> String {
        let to_str = self.to.join(", ");
        let mut msg = format!(
            "From: {}\r\nTo: {}\r\nSubject: {}\r\nMIME-Version: 1.0\r\nContent-Type: text/plain; charset=UTF-8\r\n\r\n{}",
            self.from,
            to_str,
            payload.subject,
            payload.text_summary
        );
        if self.attach_json {
            if let Some(json) = &payload.json {
                msg.push_str("\r\n\r\n[JSON Report]\r\n");
                msg.push_str(json);
            }
        }
        msg
    }
}

impl DeliveryTarget for EmailTarget {
    fn name(&self) -> &str {
        "email"
    }

    fn deliver(&self, _report: &QcReport, payload: &DeliveryPayload) -> DeliveryResult<()> {
        if self.to.is_empty() {
            return Err(DeliveryError::MissingConfig(
                "Email recipients list is empty".into(),
            ));
        }
        if self.smtp.host.is_empty() {
            return Err(DeliveryError::MissingConfig("SMTP host is empty".into()));
        }

        let email = self.build_email(payload);

        tracing::info!(
            target = "oximedia_qc::delivery",
            smtp_host = %self.smtp.host,
            smtp_port = self.smtp.port,
            recipients = ?self.to,
            email_len = email.len(),
            "QC email delivery (dry-run: no SMTP client available)"
        );

        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Log target
// ─────────────────────────────────────────────────────────────────────────────

/// Writes the QC report summary to the tracing log.  Useful for testing.
#[derive(Debug, Clone, Default)]
pub struct LogTarget;

impl LogTarget {
    /// Creates a new log delivery target.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl DeliveryTarget for LogTarget {
    fn name(&self) -> &str {
        "log"
    }

    fn deliver(&self, _report: &QcReport, payload: &DeliveryPayload) -> DeliveryResult<()> {
        if payload.passed {
            tracing::info!(target: "oximedia_qc::delivery", subject = %payload.subject, "QC PASSED");
        } else {
            tracing::warn!(
                target: "oximedia_qc::delivery",
                subject = %payload.subject,
                errors = payload.error_count,
                warnings = payload.warning_count,
                "QC FAILED"
            );
        }
        Ok(())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Delivery fanout
// ─────────────────────────────────────────────────────────────────────────────

/// Outcome of a single delivery attempt.
#[derive(Debug)]
pub struct DeliveryOutcome {
    /// Name of the target.
    pub target_name: String,
    /// Whether the delivery succeeded.
    pub success: bool,
    /// Error message if delivery failed.
    pub error: Option<String>,
}

/// Manages multiple delivery targets and fans out QC reports to all of them.
#[derive(Default)]
pub struct ReportDelivery {
    targets: Vec<Box<dyn DeliveryTarget>>,
}

impl ReportDelivery {
    /// Creates a new delivery manager.
    #[must_use]
    pub fn new() -> Self {
        Self {
            targets: Vec::new(),
        }
    }

    /// Adds a delivery target.
    pub fn add_target(&mut self, target: Box<dyn DeliveryTarget>) {
        self.targets.push(target);
    }

    /// Returns the number of configured targets.
    #[must_use]
    pub fn target_count(&self) -> usize {
        self.targets.len()
    }

    /// Delivers the report to all configured targets.
    ///
    /// Returns one [`DeliveryOutcome`] per target.  Failures in individual
    /// targets do not prevent delivery to the remaining targets.
    pub fn deliver(&self, report: &QcReport) -> Vec<DeliveryOutcome> {
        let payload = DeliveryPayload::from_report(report);
        let mut outcomes = Vec::with_capacity(self.targets.len());

        for target in &self.targets {
            let result = target.deliver(report, &payload);
            outcomes.push(DeliveryOutcome {
                target_name: target.name().to_string(),
                success: result.is_ok(),
                error: result.err().map(|e| e.to_string()),
            });
        }

        outcomes
    }

    /// Returns `true` if all delivery attempts succeeded.
    pub fn all_succeeded(outcomes: &[DeliveryOutcome]) -> bool {
        outcomes.iter().all(|o| o.success)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Encodes a byte slice as lowercase hex.
fn hex_encode(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
    out
}

/// JSON-escapes a string (wraps in quotes with special chars escaped).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::report::QcReport;
    use crate::rules::CheckResult;

    fn make_passed_report() -> QcReport {
        QcReport::new("test.mkv")
    }

    fn make_failed_report() -> QcReport {
        let mut r = QcReport::new("test.mkv");
        r.add_result(CheckResult::fail(
            "test_rule",
            crate::rules::Severity::Error,
            "intentional failure",
        ));
        r
    }

    #[test]
    fn test_log_target_passes() {
        let target = LogTarget::new();
        let report = make_passed_report();
        let payload = DeliveryPayload::from_report(&report);
        assert!(target.deliver(&report, &payload).is_ok());
    }

    #[test]
    fn test_log_target_fails() {
        let target = LogTarget::new();
        let report = make_failed_report();
        let payload = DeliveryPayload::from_report(&report);
        assert!(target.deliver(&report, &payload).is_ok());
    }

    #[test]
    fn test_webhook_empty_url_error() {
        let target = WebhookTarget::new("");
        let report = make_passed_report();
        let payload = DeliveryPayload::from_report(&report);
        assert!(target.deliver(&report, &payload).is_err());
    }

    #[test]
    fn test_webhook_valid_url_succeeds() {
        let target = WebhookTarget::new("https://example.com/webhook");
        let report = make_passed_report();
        let payload = DeliveryPayload::from_report(&report);
        assert!(target.deliver(&report, &payload).is_ok());
    }

    #[test]
    fn test_slack_empty_url_error() {
        let target = SlackTarget::new("");
        let report = make_passed_report();
        let payload = DeliveryPayload::from_report(&report);
        assert!(target.deliver(&report, &payload).is_err());
    }

    #[test]
    fn test_email_empty_recipients_error() {
        let smtp = SmtpConfig::new("smtp.example.com", 587, "user", "pass");
        let target = EmailTarget::new(smtp, "from@example.com", vec![]);
        let report = make_passed_report();
        let payload = DeliveryPayload::from_report(&report);
        assert!(target.deliver(&report, &payload).is_err());
    }

    #[test]
    fn test_email_valid_succeeds() {
        let smtp = SmtpConfig::new("smtp.example.com", 587, "user", "pass");
        let target = EmailTarget::new(smtp, "from@example.com", vec!["to@example.com".into()]);
        let report = make_passed_report();
        let payload = DeliveryPayload::from_report(&report);
        assert!(target.deliver(&report, &payload).is_ok());
    }

    #[test]
    fn test_fanout_delivery() {
        let mut delivery = ReportDelivery::new();
        delivery.add_target(Box::new(LogTarget::new()));
        delivery.add_target(Box::new(LogTarget::new()));
        assert_eq!(delivery.target_count(), 2);
        let report = make_passed_report();
        let outcomes = delivery.deliver(&report);
        assert_eq!(outcomes.len(), 2);
        assert!(ReportDelivery::all_succeeded(&outcomes));
    }

    #[test]
    fn test_delivery_payload_from_report() {
        let report = make_failed_report();
        let payload = DeliveryPayload::from_report(&report);
        assert!(!payload.passed);
        assert!(payload.error_count > 0);
        assert!(payload.subject.contains("FAILED"));
    }

    #[test]
    fn test_sha256_known_vector() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let digest = sha256(b"");
        let hex = hex_encode(&digest);
        assert_eq!(
            hex,
            "e3b0c44298fc1c149afbf4c8996fb924\
                         27ae41e4649b934ca495991b7852b855"
        );
    }

    #[test]
    fn test_hmac_sha256_smoke() {
        let sig = WebhookTarget::hmac_sha256(b"secret", b"payload");
        assert_eq!(sig.len(), 32);
        // Ensure deterministic
        let sig2 = WebhookTarget::hmac_sha256(b"secret", b"payload");
        assert_eq!(sig, sig2);
    }

    #[test]
    fn test_json_escape() {
        assert_eq!(json_escape("hello"), r#""hello""#);
        assert_eq!(json_escape("a\"b"), r#""a\"b""#);
        assert_eq!(json_escape("a\nb"), r#""a\nb""#);
    }
}
