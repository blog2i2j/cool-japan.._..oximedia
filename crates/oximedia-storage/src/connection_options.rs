//! Connection keep-alive and HTTP/2 multiplexing configuration for provider clients.
//!
//! Provider HTTP clients (S3, Azure Blob, GCS) benefit enormously from:
//! - **TCP keep-alive**: eliminates per-request handshake RTT (~1-3 ms saved per request)
//! - **HTTP/2 multiplexing**: multiple in-flight requests share one TLS connection,
//!   giving ~3-5x throughput on object-heavy workloads (e.g. listing + downloading)
//!
//! `ConnectionOptions` models these settings in a provider-agnostic way.  Each
//! provider adapter reads these options and maps them to its underlying HTTP
//! client configuration (reqwest, hyper, azure_core, etc.).
//!
//! # Example
//!
//! ```rust
//! use oximedia_storage::connection_options::ConnectionOptions;
//!
//! let opts = ConnectionOptions::default()
//!     .with_keep_alive(true)
//!     .with_http2(true)
//!     .with_max_concurrent_streams(200);
//!
//! // HTTP/2 with 200 streams → ~4x throughput multiplier
//! assert!(opts.estimated_throughput_multiplier() >= 4.0);
//! ```

use serde::{Deserialize, Serialize};

// ─── ConnectionOptions ───────────────────────────────────────────────────────

/// Tunable connection parameters for cloud storage provider HTTP clients.
///
/// These are **hints** — the underlying HTTP client applies them on a
/// best-effort basis.  Not all providers support every option.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConnectionOptions {
    /// Enable TCP keep-alive probes to detect broken connections early
    /// and avoid waiting for OS-level timeout (default: `true`).
    pub keep_alive: bool,

    /// Interval between TCP keep-alive probes in seconds (default: `30`).
    ///
    /// Only meaningful when `keep_alive` is `true`.
    pub keep_alive_interval_secs: u64,

    /// Enable HTTP/2 multiplexing: multiple requests share a single TLS
    /// connection, eliminating per-request handshake overhead (default: `true`).
    pub http2_multiplexing: bool,

    /// Maximum number of concurrent streams on a single HTTP/2 connection
    /// (default: `100`).  Capped by the server's `SETTINGS_MAX_CONCURRENT_STREAMS`.
    pub max_concurrent_streams: u32,

    /// Timeout for establishing a new TCP+TLS connection in seconds (default: `10`).
    pub connect_timeout_secs: u64,

    /// Timeout for a complete request (including read) in seconds (default: `30`).
    pub request_timeout_secs: u64,

    /// Disable Nagle's algorithm — send small packets immediately (default: `true`).
    ///
    /// Reduces latency for metadata-heavy workloads at the cost of slightly
    /// higher network overhead.
    pub tcp_nodelay: bool,

    /// Idle connection pool keep-alive interval in seconds (default: `60`).
    ///
    /// HTTP/2 PING frames are sent at this interval to keep pooled connections
    /// alive through NAT gateways that close idle sessions.
    pub pool_idle_timeout_secs: u64,
}

impl Default for ConnectionOptions {
    fn default() -> Self {
        Self {
            keep_alive: true,
            keep_alive_interval_secs: 30,
            http2_multiplexing: true,
            max_concurrent_streams: 100,
            connect_timeout_secs: 10,
            request_timeout_secs: 30,
            tcp_nodelay: true,
            pool_idle_timeout_secs: 60,
        }
    }
}

impl ConnectionOptions {
    /// Create a new `ConnectionOptions` with all defaults.
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable or disable TCP keep-alive (builder pattern).
    #[must_use]
    pub fn with_keep_alive(mut self, enabled: bool) -> Self {
        self.keep_alive = enabled;
        self
    }

    /// Set the TCP keep-alive probe interval in seconds (builder pattern).
    #[must_use]
    pub fn with_keep_alive_interval(mut self, interval_secs: u64) -> Self {
        self.keep_alive_interval_secs = interval_secs;
        self
    }

    /// Enable or disable HTTP/2 multiplexing (builder pattern).
    #[must_use]
    pub fn with_http2(mut self, enabled: bool) -> Self {
        self.http2_multiplexing = enabled;
        self
    }

    /// Set the maximum number of concurrent HTTP/2 streams (builder pattern).
    #[must_use]
    pub fn with_max_concurrent_streams(mut self, streams: u32) -> Self {
        self.max_concurrent_streams = streams;
        self
    }

    /// Set the connection establishment timeout in seconds (builder pattern).
    #[must_use]
    pub fn with_connect_timeout(mut self, timeout_secs: u64) -> Self {
        self.connect_timeout_secs = timeout_secs;
        self
    }

    /// Set the request timeout in seconds (builder pattern).
    #[must_use]
    pub fn with_request_timeout(mut self, timeout_secs: u64) -> Self {
        self.request_timeout_secs = timeout_secs;
        self
    }

    /// Enable or disable `TCP_NODELAY` (builder pattern).
    #[must_use]
    pub fn with_tcp_nodelay(mut self, enabled: bool) -> Self {
        self.tcp_nodelay = enabled;
        self
    }

    /// Set the idle pool keep-alive interval in seconds (builder pattern).
    #[must_use]
    pub fn with_pool_idle_timeout(mut self, timeout_secs: u64) -> Self {
        self.pool_idle_timeout_secs = timeout_secs;
        self
    }

    /// Estimated throughput multiplier relative to a naive HTTP/1.1 connection
    /// per request (i.e., no keep-alive, no multiplexing).
    ///
    /// | Mode                   | Multiplier |
    /// |------------------------|------------|
    /// | No keep-alive           | ×1.0       |
    /// | Keep-alive only        | ×1.5       |
    /// | HTTP/2, <100 streams   | ×2.0       |
    /// | HTTP/2, ≥100 streams   | ×4.0       |
    ///
    /// These values are approximate rule-of-thumb figures based on commonly
    /// published cloud SDK benchmarks; actual gains depend on request size,
    /// region latency, and server capacity.
    #[must_use]
    pub fn estimated_throughput_multiplier(&self) -> f64 {
        if self.http2_multiplexing {
            if self.max_concurrent_streams >= 100 {
                4.0
            } else {
                2.0
            }
        } else if self.keep_alive {
            1.5
        } else {
            1.0
        }
    }

    /// Return `true` if the configuration uses any form of connection reuse.
    #[must_use]
    pub fn has_connection_reuse(&self) -> bool {
        self.keep_alive || self.http2_multiplexing
    }

    /// Return `true` if this configuration is appropriate for high-throughput
    /// workloads (HTTP/2 with ≥100 concurrent streams).
    #[must_use]
    pub fn is_high_throughput(&self) -> bool {
        self.http2_multiplexing && self.max_concurrent_streams >= 100
    }

    /// Build a human-readable description of the active transport optimisations.
    pub fn describe(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if self.keep_alive {
            parts.push("keep-alive");
        }
        if self.http2_multiplexing {
            parts.push("HTTP/2");
        }
        if self.tcp_nodelay {
            parts.push("TCP_NODELAY");
        }
        if parts.is_empty() {
            "no optimisations".to_string()
        } else {
            parts.join(", ")
        }
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Default values ────────────────────────────────────────────────────────

    #[test]
    fn test_default_keep_alive_enabled() {
        let opts = ConnectionOptions::default();
        assert!(opts.keep_alive);
    }

    #[test]
    fn test_default_http2_multiplexing_enabled() {
        let opts = ConnectionOptions::default();
        assert!(opts.http2_multiplexing);
    }

    #[test]
    fn test_default_keep_alive_interval() {
        let opts = ConnectionOptions::default();
        assert_eq!(opts.keep_alive_interval_secs, 30);
    }

    #[test]
    fn test_default_max_concurrent_streams() {
        let opts = ConnectionOptions::default();
        assert_eq!(opts.max_concurrent_streams, 100);
    }

    #[test]
    fn test_default_connect_timeout() {
        let opts = ConnectionOptions::default();
        assert_eq!(opts.connect_timeout_secs, 10);
    }

    #[test]
    fn test_default_request_timeout() {
        let opts = ConnectionOptions::default();
        assert_eq!(opts.request_timeout_secs, 30);
    }

    #[test]
    fn test_default_tcp_nodelay_enabled() {
        let opts = ConnectionOptions::default();
        assert!(opts.tcp_nodelay);
    }

    // ── Builder pattern ───────────────────────────────────────────────────────

    #[test]
    fn test_builder_with_keep_alive_false() {
        let opts = ConnectionOptions::default().with_keep_alive(false);
        assert!(!opts.keep_alive);
        // Other fields should remain unchanged
        assert!(opts.http2_multiplexing);
    }

    #[test]
    fn test_builder_with_http2_false() {
        let opts = ConnectionOptions::default().with_http2(false);
        assert!(!opts.http2_multiplexing);
    }

    #[test]
    fn test_builder_with_max_concurrent_streams() {
        let opts = ConnectionOptions::default().with_max_concurrent_streams(50);
        assert_eq!(opts.max_concurrent_streams, 50);
    }

    #[test]
    fn test_builder_with_connect_timeout() {
        let opts = ConnectionOptions::default().with_connect_timeout(5);
        assert_eq!(opts.connect_timeout_secs, 5);
    }

    #[test]
    fn test_builder_with_request_timeout() {
        let opts = ConnectionOptions::default().with_request_timeout(120);
        assert_eq!(opts.request_timeout_secs, 120);
    }

    #[test]
    fn test_builder_chaining() {
        let opts = ConnectionOptions::new()
            .with_keep_alive(false)
            .with_http2(false)
            .with_max_concurrent_streams(10)
            .with_connect_timeout(3)
            .with_request_timeout(60)
            .with_tcp_nodelay(false)
            .with_keep_alive_interval(45)
            .with_pool_idle_timeout(120);

        assert!(!opts.keep_alive);
        assert!(!opts.http2_multiplexing);
        assert_eq!(opts.max_concurrent_streams, 10);
        assert_eq!(opts.connect_timeout_secs, 3);
        assert_eq!(opts.request_timeout_secs, 60);
        assert!(!opts.tcp_nodelay);
        assert_eq!(opts.keep_alive_interval_secs, 45);
        assert_eq!(opts.pool_idle_timeout_secs, 120);
    }

    // ── Throughput multiplier ─────────────────────────────────────────────────

    #[test]
    fn test_throughput_multiplier_no_optimisations() {
        let opts = ConnectionOptions::default()
            .with_keep_alive(false)
            .with_http2(false);
        let mult = opts.estimated_throughput_multiplier();
        assert!(
            (mult - 1.0).abs() < f64::EPSILON,
            "expected 1.0, got {mult}"
        );
    }

    #[test]
    fn test_throughput_multiplier_keep_alive_only() {
        let opts = ConnectionOptions::default()
            .with_keep_alive(true)
            .with_http2(false);
        let mult = opts.estimated_throughput_multiplier();
        assert!(
            (mult - 1.5).abs() < f64::EPSILON,
            "expected 1.5, got {mult}"
        );
    }

    #[test]
    fn test_throughput_multiplier_http2_low_streams() {
        let opts = ConnectionOptions::default()
            .with_http2(true)
            .with_max_concurrent_streams(10);
        let mult = opts.estimated_throughput_multiplier();
        assert!(
            (mult - 2.0).abs() < f64::EPSILON,
            "expected 2.0, got {mult}"
        );
    }

    #[test]
    fn test_throughput_multiplier_http2_high_streams() {
        let opts = ConnectionOptions::default()
            .with_http2(true)
            .with_max_concurrent_streams(100);
        let mult = opts.estimated_throughput_multiplier();
        assert!(
            (mult - 4.0).abs() < f64::EPSILON,
            "expected 4.0, got {mult}"
        );
    }

    #[test]
    fn test_throughput_multiplier_http2_very_high_streams() {
        let opts = ConnectionOptions::default()
            .with_http2(true)
            .with_max_concurrent_streams(500);
        let mult = opts.estimated_throughput_multiplier();
        assert!(
            (mult - 4.0).abs() < f64::EPSILON,
            "expected 4.0, got {mult}"
        );
    }

    // ── Serde round-trip ──────────────────────────────────────────────────────

    #[test]
    fn test_serde_round_trip() {
        let original = ConnectionOptions::new()
            .with_keep_alive(true)
            .with_http2(true)
            .with_max_concurrent_streams(200)
            .with_connect_timeout(15)
            .with_request_timeout(45);

        let json = serde_json::to_string(&original).expect("serialize ConnectionOptions");
        let restored: ConnectionOptions =
            serde_json::from_str(&json).expect("deserialize ConnectionOptions");

        assert_eq!(original, restored);
    }

    // ── Helper predicates ─────────────────────────────────────────────────────

    #[test]
    fn test_has_connection_reuse_with_keep_alive() {
        let opts = ConnectionOptions::default()
            .with_keep_alive(true)
            .with_http2(false);
        assert!(opts.has_connection_reuse());
    }

    #[test]
    fn test_has_connection_reuse_with_http2() {
        let opts = ConnectionOptions::default()
            .with_keep_alive(false)
            .with_http2(true);
        assert!(opts.has_connection_reuse());
    }

    #[test]
    fn test_has_no_connection_reuse_when_both_disabled() {
        let opts = ConnectionOptions::default()
            .with_keep_alive(false)
            .with_http2(false);
        assert!(!opts.has_connection_reuse());
    }

    #[test]
    fn test_is_high_throughput_true() {
        let opts = ConnectionOptions::default()
            .with_http2(true)
            .with_max_concurrent_streams(100);
        assert!(opts.is_high_throughput());
    }

    #[test]
    fn test_is_high_throughput_false_low_streams() {
        let opts = ConnectionOptions::default()
            .with_http2(true)
            .with_max_concurrent_streams(50);
        assert!(!opts.is_high_throughput());
    }

    #[test]
    fn test_is_high_throughput_false_no_http2() {
        let opts = ConnectionOptions::default()
            .with_http2(false)
            .with_max_concurrent_streams(200);
        assert!(!opts.is_high_throughput());
    }

    // ── describe() ───────────────────────────────────────────────────────────

    #[test]
    fn test_describe_all_enabled() {
        let opts = ConnectionOptions::default();
        let desc = opts.describe();
        assert!(desc.contains("keep-alive"));
        assert!(desc.contains("HTTP/2"));
        assert!(desc.contains("TCP_NODELAY"));
    }

    #[test]
    fn test_describe_none() {
        let opts = ConnectionOptions::default()
            .with_keep_alive(false)
            .with_http2(false)
            .with_tcp_nodelay(false);
        assert_eq!(opts.describe(), "no optimisations");
    }

    // ── Clone ────────────────────────────────────────────────────────────────

    #[test]
    fn test_clone() {
        let opts = ConnectionOptions::default().with_max_concurrent_streams(42);
        let cloned = opts.clone();
        assert_eq!(cloned.max_concurrent_streams, 42);
    }
}
