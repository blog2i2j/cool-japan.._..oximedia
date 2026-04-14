//! Plugin telemetry collection — anonymous usage statistics for plugin authors.
//!
//! This module records fine-grained codec usage events inside the running
//! process.  All data is kept **in-memory only** — nothing is written to disk
//! or transmitted over a network without an explicit call from the host
//! application.  The design therefore satisfies a privacy-by-default model
//! while still giving plugin authors actionable information.
//!
//! # Architecture
//!
//! ```text
//! PluginTelemetry
//!   ├── per-plugin aggregated counters  (AtomicU64 maps)
//!   └── per-plugin event ring buffer    (fixed capacity, oldest dropped)
//! ```
//!
//! - [`TelemetryEvent`] — individual timestamped observation.
//! - [`TelemetryKind`] — the type of event (codec operation, error, etc.).
//! - [`PluginStats`] — aggregated view derived from raw events.
//! - [`PluginTelemetry`] — thread-safe store; all public methods take `&self`.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_plugin::plugin_telemetry::{PluginTelemetry, TelemetryKind, TelemetryConfig};
//! use std::time::Duration;
//!
//! let tel = PluginTelemetry::new(TelemetryConfig::default());
//!
//! tel.record("my-codec-plugin", TelemetryKind::DecodeStart { codec: "vp9".into() });
//! tel.record("my-codec-plugin", TelemetryKind::DecodeEnd {
//!     codec: "vp9".into(),
//!     duration: Duration::from_millis(12),
//!     success: true,
//! });
//!
//! let stats = tel.stats("my-codec-plugin").expect("plugin found");
//! assert!(stats.decode_calls > 0);
//! ```

use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant, SystemTime};

// ── TelemetryKind ─────────────────────────────────────────────────────────────

/// The kind of telemetry event being recorded.
#[derive(Debug, Clone, PartialEq)]
pub enum TelemetryKind {
    /// A decode operation started.
    DecodeStart {
        /// Name of the codec being decoded.
        codec: String,
    },
    /// A decode operation finished.
    DecodeEnd {
        /// Name of the codec that was decoded.
        codec: String,
        /// Wall-clock duration of the decode operation.
        duration: Duration,
        /// `true` if the operation succeeded.
        success: bool,
    },
    /// An encode operation started.
    EncodeStart {
        /// Name of the codec being encoded.
        codec: String,
    },
    /// An encode operation finished.
    EncodeEnd {
        /// Name of the codec that was encoded.
        codec: String,
        /// Wall-clock duration of the encode operation.
        duration: Duration,
        /// `true` if the operation succeeded.
        success: bool,
    },
    /// The plugin was loaded into the registry.
    PluginLoaded,
    /// The plugin was unloaded from the registry.
    PluginUnloaded,
    /// A generic error occurred inside the plugin.
    Error {
        /// Short error category (e.g. `"io"`, `"corrupt-frame"`).
        category: String,
        /// Optional human-readable detail (kept brief to avoid PII).
        detail: Option<String>,
    },
    /// A custom event specific to the plugin implementation.
    Custom {
        /// Short event name chosen by the plugin author.
        name: String,
        /// Optional payload serialised as a string.
        payload: Option<String>,
    },
}

// ── TelemetryEvent ────────────────────────────────────────────────────────────

/// A single recorded telemetry event with its capture time.
#[derive(Debug, Clone)]
pub struct TelemetryEvent {
    /// Name of the plugin that generated this event.
    pub plugin_name: String,
    /// Event kind.
    pub kind: TelemetryKind,
    /// Wall-clock time at which the event was recorded (seconds since
    /// `UNIX_EPOCH`; `None` if `SystemTime` is not available).
    pub timestamp_secs: Option<u64>,
    /// Monotonic instant at which the event was recorded (useful for
    /// relative ordering and duration measurement without system clock skew).
    pub instant: Instant,
}

impl TelemetryEvent {
    fn new(plugin_name: impl Into<String>, kind: TelemetryKind) -> Self {
        let timestamp_secs = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs());
        Self {
            plugin_name: plugin_name.into(),
            kind,
            timestamp_secs,
            instant: Instant::now(),
        }
    }
}

// ── PluginStats ───────────────────────────────────────────────────────────────

/// Aggregated usage statistics derived from the recorded events of one plugin.
#[derive(Debug, Clone, Default)]
pub struct PluginStats {
    /// Total number of successfully completed decode calls.
    pub decode_calls: u64,
    /// Total number of failed decode calls.
    pub decode_errors: u64,
    /// Total number of successfully completed encode calls.
    pub encode_calls: u64,
    /// Total number of failed encode calls.
    pub encode_errors: u64,
    /// Cumulative wall-clock time spent in decode operations (successful only).
    pub total_decode_duration: Duration,
    /// Cumulative wall-clock time spent in encode operations (successful only).
    pub total_encode_duration: Duration,
    /// Number of times the plugin has been loaded.
    pub load_count: u64,
    /// Number of times the plugin has been unloaded.
    pub unload_count: u64,
    /// Number of generic errors recorded.
    pub error_count: u64,
    /// Number of custom events recorded.
    pub custom_event_count: u64,
    /// Per-codec decode call counts.
    pub decode_per_codec: HashMap<String, u64>,
    /// Per-codec encode call counts.
    pub encode_per_codec: HashMap<String, u64>,
    /// Total number of raw events stored in the ring buffer for this plugin.
    pub buffered_event_count: usize,
}

impl PluginStats {
    /// Average decode duration across successful decode operations.
    ///
    /// Returns `None` if no successful decode has been recorded.
    pub fn avg_decode_duration(&self) -> Option<Duration> {
        if self.decode_calls == 0 {
            None
        } else {
            Some(self.total_decode_duration / self.decode_calls as u32)
        }
    }

    /// Average encode duration across successful encode operations.
    ///
    /// Returns `None` if no successful encode has been recorded.
    pub fn avg_encode_duration(&self) -> Option<Duration> {
        if self.encode_calls == 0 {
            None
        } else {
            Some(self.total_encode_duration / self.encode_calls as u32)
        }
    }

    /// Decode error rate in [0.0, 1.0].
    ///
    /// Returns `0.0` if no decode operation has been recorded.
    pub fn decode_error_rate(&self) -> f64 {
        let total = self.decode_calls + self.decode_errors;
        if total == 0 {
            0.0
        } else {
            self.decode_errors as f64 / total as f64
        }
    }

    /// Encode error rate in [0.0, 1.0].
    ///
    /// Returns `0.0` if no encode operation has been recorded.
    pub fn encode_error_rate(&self) -> f64 {
        let total = self.encode_calls + self.encode_errors;
        if total == 0 {
            0.0
        } else {
            self.encode_errors as f64 / total as f64
        }
    }
}

// ── TelemetryConfig ───────────────────────────────────────────────────────────

/// Configuration for a [`PluginTelemetry`] store.
#[derive(Debug, Clone)]
pub struct TelemetryConfig {
    /// Maximum number of raw events to keep per plugin in the ring buffer.
    ///
    /// When the buffer is full, the oldest event is dropped.  Use `0` to
    /// disable raw event buffering (only aggregate counters are maintained).
    pub max_events_per_plugin: usize,

    /// When `true`, telemetry collection is globally enabled.
    ///
    /// When `false`, [`PluginTelemetry::record`] returns immediately without
    /// storing anything.  Useful for disabling telemetry at runtime without
    /// re-initialising the object.
    pub enabled: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            max_events_per_plugin: 256,
            enabled: true,
        }
    }
}

impl TelemetryConfig {
    /// Create a disabled telemetry config (no events are stored).
    pub fn disabled() -> Self {
        Self {
            max_events_per_plugin: 0,
            enabled: false,
        }
    }
}

// ── PluginTelemetryInner ──────────────────────────────────────────────────────

/// Per-plugin state managed inside the `RwLock`.
struct PluginTelemetryState {
    /// Bounded ring buffer of raw events.
    events: std::collections::VecDeque<TelemetryEvent>,
    /// Derived aggregate counters rebuilt incrementally on each `record`.
    stats: PluginStats,
}

impl PluginTelemetryState {
    fn new() -> Self {
        Self {
            events: std::collections::VecDeque::new(),
            stats: PluginStats::default(),
        }
    }

    /// Push a new event, evicting the oldest if the buffer is full.
    fn push_event(
        &mut self,
        event: TelemetryEvent,
        max_events: usize,
        config_enabled: bool,
    ) {
        // Update aggregate stats regardless of buffering.
        match &event.kind {
            TelemetryKind::DecodeEnd { codec, duration, success } => {
                if *success {
                    self.stats.decode_calls += 1;
                    self.stats.total_decode_duration += *duration;
                    *self.stats.decode_per_codec.entry(codec.clone()).or_insert(0) += 1;
                } else {
                    self.stats.decode_errors += 1;
                }
            }
            TelemetryKind::EncodeEnd { codec, duration, success } => {
                if *success {
                    self.stats.encode_calls += 1;
                    self.stats.total_encode_duration += *duration;
                    *self.stats.encode_per_codec.entry(codec.clone()).or_insert(0) += 1;
                } else {
                    self.stats.encode_errors += 1;
                }
            }
            TelemetryKind::PluginLoaded => self.stats.load_count += 1,
            TelemetryKind::PluginUnloaded => self.stats.unload_count += 1,
            TelemetryKind::Error { .. } => self.stats.error_count += 1,
            TelemetryKind::Custom { .. } => self.stats.custom_event_count += 1,
            // Start events don't contribute to aggregates.
            TelemetryKind::DecodeStart { .. } | TelemetryKind::EncodeStart { .. } => {}
        }

        // Buffer the raw event only if buffering is enabled.
        if config_enabled && max_events > 0 {
            if self.events.len() >= max_events {
                self.events.pop_front();
            }
            self.events.push_back(event);
            self.stats.buffered_event_count = self.events.len();
        }
    }
}

// ── PluginTelemetry ───────────────────────────────────────────────────────────

/// Thread-safe in-process telemetry store for all registered plugins.
///
/// Each plugin gets its own event ring buffer and aggregate counters.
/// The store is safe to share across threads via [`Arc`].
///
/// # Example
///
/// ```rust
/// use oximedia_plugin::plugin_telemetry::{PluginTelemetry, TelemetryKind, TelemetryConfig};
/// use std::sync::Arc;
/// use std::time::Duration;
///
/// let tel = Arc::new(PluginTelemetry::new(TelemetryConfig::default()));
///
/// tel.record("codec-x", TelemetryKind::PluginLoaded);
/// tel.record("codec-x", TelemetryKind::DecodeEnd {
///     codec: "av1".into(),
///     duration: Duration::from_millis(8),
///     success: true,
/// });
///
/// let stats = tel.stats("codec-x").unwrap();
/// assert_eq!(stats.decode_calls, 1);
/// assert_eq!(stats.load_count, 1);
/// ```
pub struct PluginTelemetry {
    config: TelemetryConfig,
    state: RwLock<HashMap<String, PluginTelemetryState>>,
}

impl PluginTelemetry {
    /// Create a new telemetry store with the given configuration.
    pub fn new(config: TelemetryConfig) -> Self {
        Self {
            config,
            state: RwLock::new(HashMap::new()),
        }
    }

    /// Create a new telemetry store wrapped in an `Arc` for shared ownership.
    pub fn new_shared(config: TelemetryConfig) -> Arc<Self> {
        Arc::new(Self::new(config))
    }

    /// Record a telemetry event for `plugin_name`.
    ///
    /// This is a no-op when telemetry is disabled (`config.enabled == false`).
    pub fn record(&self, plugin_name: impl Into<String>, kind: TelemetryKind) {
        if !self.config.enabled {
            return;
        }
        let name = plugin_name.into();
        let event = TelemetryEvent::new(name.clone(), kind);
        let mut guard = match self.state.write() {
            Ok(g) => g,
            Err(_) => return, // poisoned lock — swallow silently
        };
        let plugin_state = guard.entry(name).or_insert_with(PluginTelemetryState::new);
        plugin_state.push_event(event, self.config.max_events_per_plugin, self.config.enabled);
    }

    /// Return aggregated stats for `plugin_name`, or `None` if the plugin
    /// has never recorded any events.
    pub fn stats(&self, plugin_name: &str) -> Option<PluginStats> {
        let guard = self.state.read().ok()?;
        guard.get(plugin_name).map(|s| s.stats.clone())
    }

    /// Drain and return all buffered raw events for `plugin_name`.
    ///
    /// After this call the plugin's event buffer is empty.  Aggregate counters
    /// are **not** reset.
    pub fn drain_events(&self, plugin_name: &str) -> Vec<TelemetryEvent> {
        let mut guard = match self.state.write() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        match guard.get_mut(plugin_name) {
            Some(s) => {
                let events: Vec<TelemetryEvent> = s.events.drain(..).collect();
                s.stats.buffered_event_count = 0;
                events
            }
            None => Vec::new(),
        }
    }

    /// Peek at the buffered raw events for `plugin_name` without removing them.
    pub fn peek_events(&self, plugin_name: &str) -> Vec<TelemetryEvent> {
        let guard = match self.state.read() {
            Ok(g) => g,
            Err(_) => return Vec::new(),
        };
        guard
            .get(plugin_name)
            .map(|s| s.events.iter().cloned().collect())
            .unwrap_or_default()
    }

    /// Return a list of all plugin names that have recorded at least one event.
    pub fn known_plugins(&self) -> Vec<String> {
        match self.state.read() {
            Ok(g) => g.keys().cloned().collect(),
            Err(_) => Vec::new(),
        }
    }

    /// Reset the aggregate stats **and** event buffer for `plugin_name`.
    ///
    /// Returns `true` if the plugin existed and was reset; `false` otherwise.
    pub fn reset_plugin(&self, plugin_name: &str) -> bool {
        let mut guard = match self.state.write() {
            Ok(g) => g,
            Err(_) => return false,
        };
        if let Some(state) = guard.get_mut(plugin_name) {
            *state = PluginTelemetryState::new();
            true
        } else {
            false
        }
    }

    /// Clear all telemetry data across all plugins.
    pub fn clear_all(&self) {
        if let Ok(mut guard) = self.state.write() {
            guard.clear();
        }
    }

    /// Return the total number of events currently buffered across all plugins.
    pub fn total_buffered_events(&self) -> usize {
        match self.state.read() {
            Ok(g) => g.values().map(|s| s.events.len()).sum(),
            Err(_) => 0,
        }
    }

    /// Whether telemetry collection is currently enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tel() -> PluginTelemetry {
        PluginTelemetry::new(TelemetryConfig::default())
    }

    // 1. Disabled telemetry records nothing
    #[test]
    fn test_disabled_telemetry_records_nothing() {
        let t = PluginTelemetry::new(TelemetryConfig::disabled());
        t.record("plug", TelemetryKind::PluginLoaded);
        assert!(t.stats("plug").is_none());
        assert!(!t.is_enabled());
    }

    // 2. Plugin loaded event increments load_count
    #[test]
    fn test_plugin_loaded_increments_load_count() {
        let t = tel();
        t.record("plug", TelemetryKind::PluginLoaded);
        t.record("plug", TelemetryKind::PluginLoaded);
        let stats = t.stats("plug").expect("stats");
        assert_eq!(stats.load_count, 2);
    }

    // 3. Plugin unloaded event increments unload_count
    #[test]
    fn test_plugin_unloaded_increments_unload_count() {
        let t = tel();
        t.record("plug", TelemetryKind::PluginUnloaded);
        let stats = t.stats("plug").expect("stats");
        assert_eq!(stats.unload_count, 1);
    }

    // 4. Successful decode increments decode_calls and accumulates duration
    #[test]
    fn test_decode_end_success() {
        let t = tel();
        t.record(
            "plug",
            TelemetryKind::DecodeEnd {
                codec: "vp9".into(),
                duration: Duration::from_millis(10),
                success: true,
            },
        );
        let stats = t.stats("plug").expect("stats");
        assert_eq!(stats.decode_calls, 1);
        assert_eq!(stats.decode_errors, 0);
        assert_eq!(stats.total_decode_duration, Duration::from_millis(10));
        assert_eq!(*stats.decode_per_codec.get("vp9").expect("vp9"), 1);
    }

    // 5. Failed decode increments decode_errors, not decode_calls
    #[test]
    fn test_decode_end_failure() {
        let t = tel();
        t.record(
            "plug",
            TelemetryKind::DecodeEnd {
                codec: "av1".into(),
                duration: Duration::from_millis(5),
                success: false,
            },
        );
        let stats = t.stats("plug").expect("stats");
        assert_eq!(stats.decode_calls, 0);
        assert_eq!(stats.decode_errors, 1);
    }

    // 6. decode_error_rate calculation
    #[test]
    fn test_decode_error_rate() {
        let t = tel();
        // 3 successes
        for _ in 0..3 {
            t.record(
                "p",
                TelemetryKind::DecodeEnd {
                    codec: "h264".into(),
                    duration: Duration::from_millis(1),
                    success: true,
                },
            );
        }
        // 1 error
        t.record(
            "p",
            TelemetryKind::DecodeEnd {
                codec: "h264".into(),
                duration: Duration::from_millis(1),
                success: false,
            },
        );
        let stats = t.stats("p").expect("stats");
        let rate = stats.decode_error_rate();
        assert!((rate - 0.25).abs() < 1e-9);
    }

    // 7. Successful encode increments encode_calls
    #[test]
    fn test_encode_end_success() {
        let t = tel();
        t.record(
            "enc",
            TelemetryKind::EncodeEnd {
                codec: "av1".into(),
                duration: Duration::from_millis(20),
                success: true,
            },
        );
        let stats = t.stats("enc").expect("stats");
        assert_eq!(stats.encode_calls, 1);
        assert_eq!(stats.encode_errors, 0);
        assert_eq!(*stats.encode_per_codec.get("av1").expect("av1"), 1);
    }

    // 8. Error event increments error_count
    #[test]
    fn test_error_event() {
        let t = tel();
        t.record(
            "plug",
            TelemetryKind::Error {
                category: "io".into(),
                detail: Some("read timeout".into()),
            },
        );
        let stats = t.stats("plug").expect("stats");
        assert_eq!(stats.error_count, 1);
    }

    // 9. Custom event increments custom_event_count
    #[test]
    fn test_custom_event() {
        let t = tel();
        t.record(
            "plug",
            TelemetryKind::Custom {
                name: "frame_drop".into(),
                payload: None,
            },
        );
        let stats = t.stats("plug").expect("stats");
        assert_eq!(stats.custom_event_count, 1);
    }

    // 10. Ring buffer evicts oldest event when full
    #[test]
    fn test_ring_buffer_eviction() {
        let config = TelemetryConfig {
            max_events_per_plugin: 3,
            enabled: true,
        };
        let t = PluginTelemetry::new(config);
        for i in 0u64..5 {
            t.record(
                "p",
                TelemetryKind::Custom {
                    name: format!("event-{i}"),
                    payload: None,
                },
            );
        }
        // Only 3 most-recent events should remain.
        let events = t.peek_events("p");
        assert_eq!(events.len(), 3);
        // The newest should be "event-4"
        let last = events.last().expect("last");
        assert!(
            matches!(&last.kind, TelemetryKind::Custom { name, .. } if name == "event-4")
        );
    }

    // 11. drain_events clears the buffer
    #[test]
    fn test_drain_events() {
        let t = tel();
        t.record("p", TelemetryKind::PluginLoaded);
        t.record("p", TelemetryKind::PluginLoaded);
        let drained = t.drain_events("p");
        assert_eq!(drained.len(), 2);
        // Buffer should now be empty.
        assert_eq!(t.peek_events("p").len(), 0);
        // Aggregate stats still intact.
        assert_eq!(t.stats("p").expect("stats").load_count, 2);
    }

    // 12. known_plugins returns names of plugins that recorded events
    #[test]
    fn test_known_plugins() {
        let t = tel();
        t.record("alpha", TelemetryKind::PluginLoaded);
        t.record("beta", TelemetryKind::PluginLoaded);
        let mut names = t.known_plugins();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    // 13. reset_plugin clears stats and buffer
    #[test]
    fn test_reset_plugin() {
        let t = tel();
        t.record("p", TelemetryKind::PluginLoaded);
        t.record("p", TelemetryKind::PluginLoaded);
        assert!(t.reset_plugin("p"));
        let stats = t.stats("p").expect("still known");
        assert_eq!(stats.load_count, 0);
        assert_eq!(t.peek_events("p").len(), 0);
    }

    // 14. reset_plugin returns false for unknown plugin
    #[test]
    fn test_reset_unknown_plugin() {
        let t = tel();
        assert!(!t.reset_plugin("ghost"));
    }

    // 15. clear_all empties all data
    #[test]
    fn test_clear_all() {
        let t = tel();
        t.record("a", TelemetryKind::PluginLoaded);
        t.record("b", TelemetryKind::PluginLoaded);
        t.clear_all();
        assert!(t.known_plugins().is_empty());
        assert!(t.stats("a").is_none());
    }

    // 16. avg_decode_duration returns None for zero calls
    #[test]
    fn test_avg_decode_duration_none() {
        let stats = PluginStats::default();
        assert!(stats.avg_decode_duration().is_none());
    }

    // 17. avg_decode_duration correct average
    #[test]
    fn test_avg_decode_duration() {
        let t = tel();
        for ms in [10u64, 20, 30] {
            t.record(
                "p",
                TelemetryKind::DecodeEnd {
                    codec: "vp8".into(),
                    duration: Duration::from_millis(ms),
                    success: true,
                },
            );
        }
        let stats = t.stats("p").expect("stats");
        let avg = stats.avg_decode_duration().expect("avg");
        assert_eq!(avg, Duration::from_millis(20));
    }

    // 18. total_buffered_events counts across all plugins
    #[test]
    fn test_total_buffered_events() {
        let t = tel();
        t.record("a", TelemetryKind::PluginLoaded);
        t.record("a", TelemetryKind::PluginLoaded);
        t.record("b", TelemetryKind::PluginLoaded);
        assert_eq!(t.total_buffered_events(), 3);
    }

    // 19. Start events do not affect aggregate counters
    #[test]
    fn test_start_events_no_aggregate_effect() {
        let t = tel();
        t.record("p", TelemetryKind::DecodeStart { codec: "av1".into() });
        t.record("p", TelemetryKind::EncodeStart { codec: "av1".into() });
        let stats = t.stats("p").expect("stats");
        assert_eq!(stats.decode_calls, 0);
        assert_eq!(stats.encode_calls, 0);
        assert_eq!(stats.decode_errors, 0);
        assert_eq!(stats.encode_errors, 0);
    }

    // 20. encode_error_rate zero when no encodes recorded
    #[test]
    fn test_encode_error_rate_zero() {
        let stats = PluginStats::default();
        assert_eq!(stats.encode_error_rate(), 0.0);
    }
}
