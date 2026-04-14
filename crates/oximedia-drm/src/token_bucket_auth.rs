//! Token-bucket–based authentication gate for DRM license requests.
//!
//! This module extends the basic rate-limiting primitives in [`crate::rate_limit`]
//! with a full *authentication* layer: every device is issued a named bucket whose
//! tokens represent "license-request credits".  A request goes through three
//! concentric checks before a license is granted:
//!
//! 1. **Global gate** – a single shared bucket for the entire service.
//! 2. **Per-device gate** – each device has its own bucket.
//! 3. **Burst guard** – a secondary hard cap on requests within a short window.
//!
//! The [`TokenBucketAuthGate`] is the top-level entry point.  It is deliberately
//! `!Send`/`!Sync` so it can be wrapped in whatever concurrency primitive (e.g.
//! `Mutex`) the caller chooses.
//!
//! ## Design notes
//! * All timestamps are caller-supplied Unix seconds (`u64`).  This makes the
//!   code fully deterministic and easy to unit-test without mocking the clock.
//! * No `unsafe`, no `unwrap()` in library code.
//! * Every public API returns `Result<_, AuthError>` so the caller always gets
//!   a typed reason on failure.

use std::collections::HashMap;
use thiserror::Error;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// Errors returned by the token-bucket auth gate.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum AuthError {
    /// The global service-level bucket is empty.
    #[error("global rate limit exceeded (retry after {retry_after_secs}s)")]
    GlobalLimitExceeded { retry_after_secs: u64 },

    /// The per-device bucket is empty.
    #[error("device '{device_id}' rate limit exceeded (retry after {retry_after_secs}s)")]
    DeviceLimitExceeded {
        device_id: String,
        retry_after_secs: u64,
    },

    /// The short-window burst guard fired.
    #[error("device '{device_id}' burst limit exceeded (retry after {retry_after_secs}s)")]
    BurstLimitExceeded {
        device_id: String,
        retry_after_secs: u64,
    },

    /// The device has been explicitly blocked.
    #[error("device '{device_id}' is blocked until {unblocked_at_secs}")]
    DeviceBlocked {
        device_id: String,
        unblocked_at_secs: u64,
    },

    /// The device is unknown and auto-registration is disabled.
    #[error("device '{device_id}' is not registered")]
    UnknownDevice { device_id: String },
}

// ---------------------------------------------------------------------------
// Internal token bucket
// ---------------------------------------------------------------------------

/// Minimal token bucket (continuous refill model).
///
/// Exposed as `pub(crate)` so the sibling `rate_limit` module can share the
/// concept without duplicating code.  All operations accept an explicit
/// `now_secs` timestamp for deterministic testing.
#[derive(Debug, Clone)]
pub struct DeviceBucket {
    /// Maximum capacity (tokens).
    capacity: f64,
    /// Current token count.
    tokens: f64,
    /// Tokens added per second.
    refill_rate: f64,
    /// Last timestamp at which a refill was performed.
    last_refill_secs: u64,
    /// Total requests granted.
    grants: u64,
    /// Total requests denied.
    denials: u64,
}

impl DeviceBucket {
    /// Create a new bucket that starts full.
    pub fn new(capacity: f64, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill_secs: 0,
            grants: 0,
            denials: 0,
        }
    }

    /// Attempt to consume one token.  Returns `true` when allowed.
    pub fn try_acquire(&mut self, now_secs: u64) -> bool {
        self.refill(now_secs);
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            self.grants += 1;
            true
        } else {
            self.denials += 1;
            false
        }
    }

    /// How many seconds until the next token is available (0 if already ≥ 1).
    pub fn secs_until_next_token(&self) -> u64 {
        if self.tokens >= 1.0 {
            0
        } else {
            let needed = 1.0 - self.tokens;
            (needed / self.refill_rate).ceil() as u64
        }
    }

    /// Available tokens (floored to the nearest whole token for display).
    pub fn available(&self) -> u64 {
        self.tokens.floor() as u64
    }

    /// Total granted requests.
    pub fn grants(&self) -> u64 {
        self.grants
    }

    /// Total denied requests.
    pub fn denials(&self) -> u64 {
        self.denials
    }

    /// Refill based on elapsed time.
    fn refill(&mut self, now_secs: u64) {
        if now_secs > self.last_refill_secs {
            let elapsed = (now_secs - self.last_refill_secs) as f64;
            self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
            self.last_refill_secs = now_secs;
        }
    }
}

// ---------------------------------------------------------------------------
// Burst guard — sliding fixed-window
// ---------------------------------------------------------------------------

/// Fixed-window burst guard.
///
/// Counts requests within a short window (`window_secs`) and rejects once the
/// hard cap (`max_requests`) is exceeded.  The window resets atomically when
/// the current window period expires.
#[derive(Debug, Clone)]
struct BurstGuard {
    max_requests: u32,
    window_secs: u64,
    /// Start timestamp of the current window (`None` before the first request).
    window_start: Option<u64>,
    /// Count within the current window.
    count: u32,
}

impl BurstGuard {
    fn new(max_requests: u32, window_secs: u64) -> Self {
        Self {
            max_requests,
            window_secs,
            window_start: None,
            count: 0,
        }
    }

    /// Returns `true` if the request is within the burst limit.
    fn try_record(&mut self, now_secs: u64) -> bool {
        // Initialise or reset the window when needed.
        match self.window_start {
            None => {
                self.window_start = Some(now_secs);
                self.count = 0;
            }
            Some(start) if now_secs >= start + self.window_secs => {
                self.window_start = Some(now_secs);
                self.count = 0;
            }
            _ => {}
        }
        if self.count < self.max_requests {
            self.count += 1;
            true
        } else {
            false
        }
    }

    /// Seconds until the current window expires (i.e. the burst guard resets).
    fn secs_until_reset(&self, now_secs: u64) -> u64 {
        match self.window_start {
            None => 0,
            Some(start) => {
                let window_end = start + self.window_secs;
                window_end.saturating_sub(now_secs)
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-device state
// ---------------------------------------------------------------------------

/// All mutable state tracked for a single device.
#[derive(Debug, Clone)]
struct DeviceState {
    bucket: DeviceBucket,
    burst: BurstGuard,
    /// If `Some(t)`, the device is blocked until Unix timestamp `t`.
    blocked_until: Option<u64>,
}

impl DeviceState {
    fn new(bucket_capacity: f64, refill_rate: f64, burst_max: u32, burst_window_secs: u64) -> Self {
        Self {
            bucket: DeviceBucket::new(bucket_capacity, refill_rate),
            burst: BurstGuard::new(burst_max, burst_window_secs),
            blocked_until: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the [`TokenBucketAuthGate`].
#[derive(Debug, Clone)]
pub struct AuthGateConfig {
    /// Global bucket capacity (requests).
    pub global_capacity: f64,
    /// Global bucket refill rate (requests per second).
    pub global_refill_rate: f64,
    /// Per-device bucket capacity.
    pub device_capacity: f64,
    /// Per-device refill rate (requests per second).
    pub device_refill_rate: f64,
    /// Maximum requests in the burst window per device.
    pub burst_max: u32,
    /// Duration of the burst window in seconds.
    pub burst_window_secs: u64,
    /// Whether to auto-register unknown devices (if `false`, returns `UnknownDevice`).
    pub auto_register: bool,
}

impl Default for AuthGateConfig {
    fn default() -> Self {
        Self {
            global_capacity: 1000.0,
            global_refill_rate: 100.0, // 100 req/s globally
            device_capacity: 10.0,
            device_refill_rate: 0.5,  // 1 req per 2s per device
            burst_max: 5,
            burst_window_secs: 10,
            auto_register: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Per-device statistics snapshot.
#[derive(Debug, Clone)]
pub struct DeviceStats {
    pub device_id: String,
    pub available_tokens: u64,
    pub grants: u64,
    pub denials: u64,
    pub is_blocked: bool,
}

/// Service-wide statistics snapshot.
#[derive(Debug, Clone)]
pub struct GateStats {
    pub global_available: u64,
    pub total_devices: usize,
    pub blocked_devices: usize,
    pub total_grants: u64,
    pub total_denials: u64,
}

// ---------------------------------------------------------------------------
// Main gate
// ---------------------------------------------------------------------------

/// Three-layer token-bucket authentication gate for DRM license requests.
///
/// # Example
///
/// ```
/// use oximedia_drm::token_bucket_auth::{TokenBucketAuthGate, AuthGateConfig};
///
/// let mut gate = TokenBucketAuthGate::new(AuthGateConfig::default());
/// // Grant a license request for device "device-001" at t=0
/// assert!(gate.request("device-001", 0).is_ok());
/// ```
#[derive(Debug)]
pub struct TokenBucketAuthGate {
    config: AuthGateConfig,
    global: DeviceBucket,
    devices: HashMap<String, DeviceState>,
}

impl TokenBucketAuthGate {
    /// Create a new gate with the given configuration.
    pub fn new(config: AuthGateConfig) -> Self {
        let global = DeviceBucket::new(config.global_capacity, config.global_refill_rate);
        Self {
            config,
            global,
            devices: HashMap::new(),
        }
    }

    /// Attempt to grant a license request for `device_id` at timestamp `now_secs`.
    ///
    /// Runs three checks in order: global → per-device → burst guard.
    /// The first failing check returns the corresponding [`AuthError`].
    pub fn request(&mut self, device_id: &str, now_secs: u64) -> Result<(), AuthError> {
        // 1. Global gate
        if !self.global.try_acquire(now_secs) {
            return Err(AuthError::GlobalLimitExceeded {
                retry_after_secs: self.global.secs_until_next_token(),
            });
        }

        // 2. Ensure the device exists (or auto-register)
        if !self.devices.contains_key(device_id) {
            if !self.config.auto_register {
                // Refund the global token we consumed
                self.global.tokens = (self.global.tokens + 1.0).min(self.global.capacity);
                return Err(AuthError::UnknownDevice {
                    device_id: device_id.to_string(),
                });
            }
            self.devices.insert(
                device_id.to_string(),
                DeviceState::new(
                    self.config.device_capacity,
                    self.config.device_refill_rate,
                    self.config.burst_max,
                    self.config.burst_window_secs,
                ),
            );
        }

        let state = self
            .devices
            .get_mut(device_id)
            .ok_or_else(|| AuthError::UnknownDevice {
                device_id: device_id.to_string(),
            })?;

        // 3. Block check
        if let Some(until) = state.blocked_until {
            if now_secs < until {
                return Err(AuthError::DeviceBlocked {
                    device_id: device_id.to_string(),
                    unblocked_at_secs: until,
                });
            }
            // Block expired
            state.blocked_until = None;
        }

        // 4. Per-device token bucket
        if !state.bucket.try_acquire(now_secs) {
            return Err(AuthError::DeviceLimitExceeded {
                device_id: device_id.to_string(),
                retry_after_secs: state.bucket.secs_until_next_token(),
            });
        }

        // 5. Burst guard
        if !state.burst.try_record(now_secs) {
            // Refund the device token (burst guard failure should not penalise the slow-moving bucket)
            state.bucket.tokens = (state.bucket.tokens + 1.0).min(state.bucket.capacity);
            state.bucket.grants -= 1;
            return Err(AuthError::BurstLimitExceeded {
                device_id: device_id.to_string(),
                retry_after_secs: state.burst.secs_until_reset(now_secs).max(1),
            });
        }

        Ok(())
    }

    /// Manually register a device without issuing a request.
    pub fn register_device(&mut self, device_id: &str) {
        self.devices.entry(device_id.to_string()).or_insert_with(|| {
            DeviceState::new(
                self.config.device_capacity,
                self.config.device_refill_rate,
                self.config.burst_max,
                self.config.burst_window_secs,
            )
        });
    }

    /// Block a device until `unblocked_at_secs`.
    ///
    /// Returns `false` if the device is not registered.
    pub fn block_device(&mut self, device_id: &str, unblocked_at_secs: u64) -> bool {
        if let Some(state) = self.devices.get_mut(device_id) {
            state.blocked_until = Some(unblocked_at_secs);
            true
        } else {
            false
        }
    }

    /// Unblock a device immediately.
    ///
    /// Returns `false` if the device is not registered.
    pub fn unblock_device(&mut self, device_id: &str) -> bool {
        if let Some(state) = self.devices.get_mut(device_id) {
            state.blocked_until = None;
            true
        } else {
            false
        }
    }

    /// Check whether a device is currently blocked at `now_secs`.
    pub fn is_blocked(&self, device_id: &str, now_secs: u64) -> bool {
        self.devices
            .get(device_id)
            .and_then(|s| s.blocked_until)
            .map_or(false, |until| now_secs < until)
    }

    /// Retrieve per-device statistics.
    pub fn device_stats(&self, device_id: &str, now_secs: u64) -> Option<DeviceStats> {
        self.devices.get(device_id).map(|s| DeviceStats {
            device_id: device_id.to_string(),
            available_tokens: s.bucket.available(),
            grants: s.bucket.grants(),
            denials: s.bucket.denials(),
            is_blocked: s.blocked_until.map_or(false, |u| now_secs < u),
        })
    }

    /// Retrieve service-wide statistics.
    pub fn gate_stats(&self, now_secs: u64) -> GateStats {
        let blocked = self
            .devices
            .values()
            .filter(|s| s.blocked_until.map_or(false, |u| now_secs < u))
            .count();
        let total_grants: u64 = self.devices.values().map(|s| s.bucket.grants()).sum();
        let total_denials: u64 = self.devices.values().map(|s| s.bucket.denials()).sum();
        GateStats {
            global_available: self.global.available(),
            total_devices: self.devices.len(),
            blocked_devices: blocked,
            total_grants,
            total_denials,
        }
    }

    /// Remove stale device entries that have not been used for `idle_secs` seconds.
    pub fn evict_idle_devices(&mut self, now_secs: u64, idle_secs: u64) {
        self.devices.retain(|_, state| {
            now_secs.saturating_sub(state.bucket.last_refill_secs) < idle_secs
        });
    }

    /// Number of registered devices.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn default_gate() -> TokenBucketAuthGate {
        TokenBucketAuthGate::new(AuthGateConfig {
            global_capacity: 1000.0,
            global_refill_rate: 100.0,
            device_capacity: 5.0,
            device_refill_rate: 1.0,
            burst_max: 3,
            burst_window_secs: 5,
            auto_register: true,
        })
    }

    #[test]
    fn test_basic_grant() {
        let mut gate = default_gate();
        assert!(gate.request("dev-1", 0).is_ok());
    }

    #[test]
    fn test_device_bucket_exhaustion() {
        // Use a large burst limit so only the token bucket limits requests.
        let mut gate = TokenBucketAuthGate::new(AuthGateConfig {
            global_capacity: 1000.0,
            global_refill_rate: 100.0,
            device_capacity: 3.0,
            device_refill_rate: 1.0,
            burst_max: 100,          // large burst allowance
            burst_window_secs: 1000,
            auto_register: true,
        });
        // Exhaust the 3-token device bucket
        assert!(gate.request("dev-a", 0).is_ok());
        assert!(gate.request("dev-a", 0).is_ok());
        assert!(gate.request("dev-a", 0).is_ok());
        // 4th request is blocked by the device token bucket
        let result = gate.request("dev-a", 0);
        assert!(matches!(result, Err(AuthError::DeviceLimitExceeded { .. })));
        // After 3 seconds the bucket refills by 3 tokens → ok again
        assert!(gate.request("dev-a", 3).is_ok());
    }

    #[test]
    fn test_burst_guard_fires() {
        let mut gate = default_gate();
        // burst_max=3, window=5s
        assert!(gate.request("dev-b", 0).is_ok());
        assert!(gate.request("dev-b", 0).is_ok());
        assert!(gate.request("dev-b", 0).is_ok());
        // 4th request in the same window should fail burst guard
        let result = gate.request("dev-b", 0);
        assert!(matches!(result, Err(AuthError::BurstLimitExceeded { .. })));
    }

    #[test]
    fn test_burst_guard_resets_after_window() {
        let mut gate = default_gate();
        for _ in 0..3 {
            gate.request("dev-c", 0).expect("should succeed");
        }
        // Next request in same window is blocked
        assert!(gate.request("dev-c", 0).is_err());
        // After window expires (5 s) it resets
        assert!(gate.request("dev-c", 6).is_ok());
    }

    #[test]
    fn test_block_device() {
        let mut gate = default_gate();
        gate.register_device("dev-d");
        gate.block_device("dev-d", 9999);
        let result = gate.request("dev-d", 1000);
        assert!(matches!(result, Err(AuthError::DeviceBlocked { .. })));
    }

    #[test]
    fn test_block_expires() {
        let mut gate = default_gate();
        gate.register_device("dev-e");
        gate.block_device("dev-e", 50);
        // At t=50 the block expires
        assert!(gate.request("dev-e", 60).is_ok());
        assert!(!gate.is_blocked("dev-e", 60));
    }

    #[test]
    fn test_unblock_device() {
        let mut gate = default_gate();
        gate.register_device("dev-f");
        gate.block_device("dev-f", 9999);
        assert!(gate.is_blocked("dev-f", 0));
        gate.unblock_device("dev-f");
        assert!(!gate.is_blocked("dev-f", 0));
    }

    #[test]
    fn test_unknown_device_no_auto_register() {
        let mut gate = TokenBucketAuthGate::new(AuthGateConfig {
            auto_register: false,
            ..AuthGateConfig::default()
        });
        let result = gate.request("unknown-dev", 0);
        assert!(matches!(result, Err(AuthError::UnknownDevice { .. })));
    }

    #[test]
    fn test_unknown_device_auto_register() {
        let mut gate = TokenBucketAuthGate::new(AuthGateConfig::default());
        assert!(gate.request("new-dev", 0).is_ok());
        assert_eq!(gate.device_count(), 1);
    }

    #[test]
    fn test_device_stats() {
        let mut gate = default_gate();
        gate.request("dev-g", 0).expect("should succeed");
        let stats = gate.device_stats("dev-g", 0).expect("stats should exist");
        assert_eq!(stats.grants, 1);
        assert_eq!(stats.denials, 0);
        assert!(!stats.is_blocked);
    }

    #[test]
    fn test_gate_stats() {
        let mut gate = default_gate();
        gate.request("d1", 0).expect("ok");
        gate.request("d2", 0).expect("ok");
        gate.register_device("d3");
        gate.block_device("d3", 9999);
        let stats = gate.gate_stats(0);
        assert_eq!(stats.total_devices, 3);
        assert_eq!(stats.blocked_devices, 1);
        assert_eq!(stats.total_grants, 2);
    }

    #[test]
    fn test_evict_idle_devices() {
        let mut gate = default_gate();
        gate.request("stale", 0).expect("ok");
        gate.request("active", 100).expect("ok");
        gate.evict_idle_devices(100, 50);
        assert_eq!(gate.device_count(), 1);
        assert!(gate.device_stats("active", 100).is_some());
        assert!(gate.device_stats("stale", 100).is_none());
    }

    #[test]
    fn test_multi_device_isolation() {
        let mut gate = default_gate();
        // Exhaust burst for device-x
        for _ in 0..3 {
            gate.request("device-x", 0).expect("ok");
        }
        // device-y is independent
        assert!(gate.request("device-y", 0).is_ok());
    }

    #[test]
    fn test_device_bucket_refills_over_time() {
        let mut gate = TokenBucketAuthGate::new(AuthGateConfig {
            global_capacity: 1000.0,
            global_refill_rate: 100.0,
            device_capacity: 2.0,
            device_refill_rate: 1.0,
            burst_max: 100,
            burst_window_secs: 1000,
            auto_register: true,
        });
        // Exhaust bucket
        gate.request("refill-dev", 0).expect("ok");
        gate.request("refill-dev", 0).expect("ok");
        assert!(matches!(
            gate.request("refill-dev", 0),
            Err(AuthError::DeviceLimitExceeded { .. })
        ));
        // After 2 seconds bucket refills by 2 tokens
        assert!(gate.request("refill-dev", 2).is_ok());
    }

    #[test]
    fn test_block_nonexistent_device_returns_false() {
        let mut gate = default_gate();
        assert!(!gate.block_device("ghost", 9999));
    }

    #[test]
    fn test_auth_error_messages() {
        let e = AuthError::GlobalLimitExceeded { retry_after_secs: 3 };
        assert!(format!("{e}").contains("global"));

        let e2 = AuthError::DeviceBlocked {
            device_id: "d".to_string(),
            unblocked_at_secs: 100,
        };
        assert!(format!("{e2}").contains("blocked"));
    }
}
