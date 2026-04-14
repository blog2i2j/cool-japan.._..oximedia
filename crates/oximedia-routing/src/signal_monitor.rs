#![allow(dead_code)]
//! Signal presence and health monitoring for routed signals.
//!
//! Provides [`SignalStatus`], [`SignalMonitor`], and [`MonitorReport`] for
//! tracking whether routed signals are active and within spec.

use std::collections::HashMap;
use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// Signal status
// ---------------------------------------------------------------------------

/// The health status of a monitored signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalStatus {
    /// Signal is present and within specification.
    Ok,
    /// Signal is present but a metric is out of tolerance.
    Warning,
    /// Signal loss detected.
    Lost,
    /// Monitoring has not yet produced a result for this signal.
    Unknown,
}

impl SignalStatus {
    /// Returns a short human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Ok => "ok",
            Self::Warning => "warning",
            Self::Lost => "lost",
            Self::Unknown => "unknown",
        }
    }

    /// Returns `true` if the signal requires operator attention.
    pub fn needs_attention(&self) -> bool {
        matches!(self, Self::Warning | Self::Lost)
    }

    /// Returns `true` if the signal is operational.
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Ok)
    }
}

// ---------------------------------------------------------------------------
// Signal sample
// ---------------------------------------------------------------------------

/// A single measurement sample for a monitored signal.
#[derive(Debug, Clone)]
pub struct SignalSample {
    /// Signal level in dBFS (audio) or dBmV (video).
    pub level_db: f32,
    /// Signal-to-noise ratio in dB.
    pub snr_db: f32,
    /// Timestamp of the sample.
    pub timestamp: Instant,
}

impl SignalSample {
    /// Creates a new sample taken at the current instant.
    pub fn now(level_db: f32, snr_db: f32) -> Self {
        Self {
            level_db,
            snr_db,
            timestamp: Instant::now(),
        }
    }

    /// Returns the age of this sample.
    pub fn age(&self) -> Duration {
        self.timestamp.elapsed()
    }
}

// ---------------------------------------------------------------------------
// Monitor entry
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct MonitorEntry {
    port_name: String,
    status: SignalStatus,
    last_sample: Option<SignalSample>,
    fault_count: u32,
    min_level_db: f32,
    max_snr_db: f32,
}

impl MonitorEntry {
    fn new(port_name: impl Into<String>, min_level_db: f32, max_snr_db: f32) -> Self {
        Self {
            port_name: port_name.into(),
            status: SignalStatus::Unknown,
            last_sample: None,
            fault_count: 0,
            min_level_db,
            max_snr_db,
        }
    }

    fn update(&mut self, sample: SignalSample) {
        let level_ok = sample.level_db >= self.min_level_db;
        let snr_ok = sample.snr_db <= self.max_snr_db;

        self.status = match (level_ok, snr_ok) {
            (true, true) => SignalStatus::Ok,
            (false, _) => {
                self.fault_count += 1;
                SignalStatus::Lost
            }
            (true, false) => SignalStatus::Warning,
        };
        self.last_sample = Some(sample);
    }
}

// ---------------------------------------------------------------------------
// SignalMonitor
// ---------------------------------------------------------------------------

/// Monitors the health of a set of named signal ports.
#[derive(Debug, Default)]
pub struct SignalMonitor {
    entries: HashMap<String, MonitorEntry>,
}

impl SignalMonitor {
    /// Creates an empty monitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a port for monitoring.
    ///
    /// `min_level_db` — minimum acceptable level (e.g., -60 dBFS).
    /// `max_snr_threshold_db` — maximum SNR value considered a warning
    ///   (i.e., SNR above this value is considered noisy/reversed — set
    ///   very high to effectively disable, e.g., 100.0).
    pub fn register_port(
        &mut self,
        port_name: impl Into<String>,
        min_level_db: f32,
        max_snr_threshold_db: f32,
    ) {
        let name = port_name.into();
        self.entries.insert(
            name.clone(),
            MonitorEntry::new(name, min_level_db, max_snr_threshold_db),
        );
    }

    /// Submits a measurement sample for the named port.
    ///
    /// Returns `None` if the port is not registered.
    pub fn submit(&mut self, port_name: &str, sample: SignalSample) -> Option<SignalStatus> {
        let entry = self.entries.get_mut(port_name)?;
        entry.update(sample);
        Some(entry.status)
    }

    /// Returns the current status of the named port.
    pub fn status(&self, port_name: &str) -> SignalStatus {
        self.entries
            .get(port_name)
            .map(|e| e.status)
            .unwrap_or(SignalStatus::Unknown)
    }

    /// Returns the number of faults recorded for a port.
    pub fn fault_count(&self, port_name: &str) -> u32 {
        self.entries
            .get(port_name)
            .map(|e| e.fault_count)
            .unwrap_or(0)
    }

    /// Returns the last sample for a port, if any.
    pub fn last_sample(&self, port_name: &str) -> Option<&SignalSample> {
        self.entries.get(port_name)?.last_sample.as_ref()
    }

    /// Returns the number of registered ports.
    pub fn port_count(&self) -> usize {
        self.entries.len()
    }

    /// Generates a summary report.
    pub fn report(&self) -> MonitorReport {
        let mut ok = 0usize;
        let mut warning = 0usize;
        let mut lost = 0usize;
        let mut unknown = 0usize;

        for entry in self.entries.values() {
            match entry.status {
                SignalStatus::Ok => ok += 1,
                SignalStatus::Warning => warning += 1,
                SignalStatus::Lost => lost += 1,
                SignalStatus::Unknown => unknown += 1,
            }
        }

        MonitorReport {
            total: self.entries.len(),
            ok,
            warning,
            lost,
            unknown,
        }
    }
}

// ---------------------------------------------------------------------------
// MonitorReport
// ---------------------------------------------------------------------------

/// Summary report from a [`SignalMonitor`] poll.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MonitorReport {
    /// Total number of monitored ports.
    pub total: usize,
    /// Ports with `Ok` status.
    pub ok: usize,
    /// Ports with `Warning` status.
    pub warning: usize,
    /// Ports with `Lost` status.
    pub lost: usize,
    /// Ports with `Unknown` status.
    pub unknown: usize,
}

impl MonitorReport {
    /// Returns `true` if all ports are `Ok`.
    pub fn all_ok(&self) -> bool {
        self.total > 0 && self.ok == self.total
    }

    /// Returns `true` if any port needs operator attention.
    pub fn has_alerts(&self) -> bool {
        self.warning > 0 || self.lost > 0
    }
}

// ---------------------------------------------------------------------------
// Alert severity
// ---------------------------------------------------------------------------

/// Severity level for a signal alert.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational — no action needed.
    Info,
    /// Warning — may need attention soon.
    Warning,
    /// Critical — immediate attention required.
    Critical,
}

impl AlertSeverity {
    /// Returns a short label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Info => "info",
            Self::Warning => "warning",
            Self::Critical => "critical",
        }
    }
}

// ---------------------------------------------------------------------------
// Alert kind
// ---------------------------------------------------------------------------

/// The kind of alert raised by the threshold engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlertKind {
    /// Signal was absent but is now present.
    SignalPresent,
    /// Signal was present but has been lost.
    SignalLost,
    /// Signal is above the overmodulation threshold (potential clipping).
    Overmodulation,
    /// Signal level is below the noise floor.
    BelowNoiseFloor,
    /// Signal phase inversion detected (reserved for future use).
    PhaseInversion,
}

impl AlertKind {
    /// Returns a human-readable label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::SignalPresent => "signal_present",
            Self::SignalLost => "signal_lost",
            Self::Overmodulation => "overmodulation",
            Self::BelowNoiseFloor => "below_noise_floor",
            Self::PhaseInversion => "phase_inversion",
        }
    }
}

// ---------------------------------------------------------------------------
// Alert
// ---------------------------------------------------------------------------

/// A concrete alert emitted by the [`ThresholdAlertEngine`].
#[derive(Debug, Clone)]
pub struct SignalAlert {
    /// Name of the port that triggered this alert.
    pub port_name: String,
    /// Kind of alert.
    pub kind: AlertKind,
    /// Severity of the alert.
    pub severity: AlertSeverity,
    /// The sample level (dBFS) that triggered the alert.
    pub level_db: f32,
    /// Descriptive message.
    pub message: String,
}

// ---------------------------------------------------------------------------
// ThresholdConfig
// ---------------------------------------------------------------------------

/// Per-port threshold configuration for the alert engine.
#[derive(Debug, Clone)]
pub struct ThresholdConfig {
    /// Level in dBFS above which overmodulation is flagged. Typically -1.0 or -0.5.
    pub overmodulation_db: f32,
    /// Level in dBFS below which the signal is considered lost. Typically -60.0.
    pub signal_lost_db: f32,
    /// Level in dBFS above which a previously-lost signal is considered present.
    /// Should be higher than `signal_lost_db` to provide hysteresis.
    pub signal_present_db: f32,
    /// Number of consecutive samples that must breach a threshold before an
    /// alert is raised (debounce / hold-off).
    pub hold_off_count: u32,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            overmodulation_db: -1.0,
            signal_lost_db: -60.0,
            signal_present_db: -50.0,
            hold_off_count: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// Port alert state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct PortAlertState {
    config: ThresholdConfig,
    /// Whether the signal is currently considered present.
    signal_present: bool,
    /// Counter for consecutive lost samples.
    lost_streak: u32,
    /// Counter for consecutive present samples.
    present_streak: u32,
    /// Counter for consecutive overmodulation samples.
    overmod_streak: u32,
    /// Total number of alerts raised for this port.
    alert_count: u32,
}

impl PortAlertState {
    fn new(config: ThresholdConfig) -> Self {
        Self {
            config,
            signal_present: false,
            lost_streak: 0,
            present_streak: 0,
            overmod_streak: 0,
            alert_count: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// ThresholdAlertEngine
// ---------------------------------------------------------------------------

/// Engine that evaluates signal levels against configurable thresholds and
/// emits [`SignalAlert`]s for overmodulation, signal present, and signal lost
/// conditions with debounce hold-off.
#[derive(Debug, Clone, Default)]
pub struct ThresholdAlertEngine {
    ports: HashMap<String, PortAlertState>,
}

impl ThresholdAlertEngine {
    /// Creates a new, empty engine.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a port with the given threshold configuration.
    pub fn register(&mut self, port_name: impl Into<String>, config: ThresholdConfig) {
        let name = port_name.into();
        self.ports.insert(name, PortAlertState::new(config));
    }

    /// Submits a level measurement for a port and returns any alerts triggered.
    ///
    /// Returns an empty `Vec` if the port is not registered or no threshold
    /// is breached.
    pub fn evaluate(&mut self, port_name: &str, level_db: f32) -> Vec<SignalAlert> {
        let state = match self.ports.get_mut(port_name) {
            Some(s) => s,
            None => return Vec::new(),
        };
        let mut alerts = Vec::new();

        // --- Overmodulation ---
        if level_db >= state.config.overmodulation_db {
            state.overmod_streak += 1;
            if state.overmod_streak >= state.config.hold_off_count {
                state.alert_count += 1;
                alerts.push(SignalAlert {
                    port_name: port_name.to_string(),
                    kind: AlertKind::Overmodulation,
                    severity: AlertSeverity::Critical,
                    level_db,
                    message: format!(
                        "Port '{}' overmodulated at {:.1} dBFS (threshold {:.1})",
                        port_name, level_db, state.config.overmodulation_db
                    ),
                });
            }
        } else {
            state.overmod_streak = 0;
        }

        // --- Signal lost ---
        if level_db < state.config.signal_lost_db {
            state.lost_streak += 1;
            state.present_streak = 0;

            if state.signal_present && state.lost_streak >= state.config.hold_off_count {
                state.signal_present = false;
                state.alert_count += 1;
                alerts.push(SignalAlert {
                    port_name: port_name.to_string(),
                    kind: AlertKind::SignalLost,
                    severity: AlertSeverity::Critical,
                    level_db,
                    message: format!("Port '{}' signal lost at {:.1} dBFS", port_name, level_db),
                });
            }
        }
        // --- Signal present ---
        else if level_db >= state.config.signal_present_db {
            state.present_streak += 1;
            state.lost_streak = 0;

            if !state.signal_present && state.present_streak >= state.config.hold_off_count {
                state.signal_present = true;
                state.alert_count += 1;
                alerts.push(SignalAlert {
                    port_name: port_name.to_string(),
                    kind: AlertKind::SignalPresent,
                    severity: AlertSeverity::Info,
                    level_db,
                    message: format!(
                        "Port '{}' signal present at {:.1} dBFS",
                        port_name, level_db
                    ),
                });
            }
        } else {
            // Between signal_lost_db and signal_present_db — hysteresis zone.
            // Don't change state, reset streaks.
            state.lost_streak = 0;
            state.present_streak = 0;
        }

        alerts
    }

    /// Returns the total number of alerts raised for a port.
    pub fn alert_count(&self, port_name: &str) -> u32 {
        self.ports
            .get(port_name)
            .map(|s| s.alert_count)
            .unwrap_or(0)
    }

    /// Returns whether a port currently has signal present.
    pub fn is_signal_present(&self, port_name: &str) -> bool {
        self.ports
            .get(port_name)
            .map(|s| s.signal_present)
            .unwrap_or(false)
    }

    /// Returns the number of registered ports.
    pub fn port_count(&self) -> usize {
        self.ports.len()
    }

    /// Resets all streak counters and alert counts for a port.
    pub fn reset(&mut self, port_name: &str) {
        if let Some(state) = self.ports.get_mut(port_name) {
            state.lost_streak = 0;
            state.present_streak = 0;
            state.overmod_streak = 0;
            state.alert_count = 0;
            state.signal_present = false;
        }
    }
}

// ---------------------------------------------------------------------------
// SignalEvent
// ---------------------------------------------------------------------------

/// An event emitted by [`SignalPresenceDetector`] when the signal presence
/// state transitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignalEvent {
    /// A signal that was absent has become present.
    Present,
    /// A signal that was present has become absent.
    Absent,
}

// ---------------------------------------------------------------------------
// SignalPresenceDetector
// ---------------------------------------------------------------------------

/// Per-crosspoint signal presence detector with configurable threshold and
/// hold-off debounce.
///
/// The detector tracks whether a signal level is above `threshold_dbfs`.  A
/// state transition (absent → present or present → absent) is only emitted
/// after `holdoff_samples` consecutive samples agree with the new state, where
/// `holdoff_samples = (holdoff_ms * sample_rate_hz) / 1000`.
#[derive(Debug, Clone)]
pub struct SignalPresenceDetector {
    threshold_dbfs_val: f32,
    holdoff_ms: u32,
    currently_present: bool,
    consecutive_present: u32,
    consecutive_absent: u32,
}

impl SignalPresenceDetector {
    /// Creates a new detector.
    ///
    /// * `threshold_dbfs` — level above which signal is considered present.
    /// * `holdoff_ms` — debounce hold-off in milliseconds.
    pub fn new(threshold_dbfs: f32, holdoff_ms: u32) -> Self {
        Self {
            threshold_dbfs_val: threshold_dbfs,
            holdoff_ms,
            currently_present: false,
            consecutive_present: 0,
            consecutive_absent: 0,
        }
    }

    /// Returns the configured threshold in dBFS.
    pub fn threshold_dbfs(&self) -> f32 {
        self.threshold_dbfs_val
    }

    /// Returns `true` if the signal is currently considered present.
    pub fn is_present(&self) -> bool {
        self.currently_present
    }

    /// Resets the detector to its initial (absent) state.
    pub fn reset(&mut self) {
        self.currently_present = false;
        self.consecutive_present = 0;
        self.consecutive_absent = 0;
    }

    /// Processes a single level sample.
    ///
    /// Returns `Some(SignalEvent::Present)` or `Some(SignalEvent::Absent)` on a
    /// state transition, or `None` if the state did not change.
    ///
    /// `sample_rate_hz` is used to convert `holdoff_ms` into a sample count.
    pub fn process(&mut self, level_dbfs: f32, sample_rate_hz: u32) -> Option<SignalEvent> {
        let holdoff_samples = (self.holdoff_ms as u64 * sample_rate_hz as u64) / 1000;
        let holdoff_samples = holdoff_samples as u32;

        if level_dbfs > self.threshold_dbfs_val {
            self.consecutive_present = self.consecutive_present.saturating_add(1);
            self.consecutive_absent = 0;

            if !self.currently_present && self.consecutive_present >= holdoff_samples {
                self.currently_present = true;
                return Some(SignalEvent::Present);
            }
        } else {
            self.consecutive_absent = self.consecutive_absent.saturating_add(1);
            self.consecutive_present = 0;

            if self.currently_present && self.consecutive_absent >= holdoff_samples {
                self.currently_present = false;
                return Some(SignalEvent::Absent);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// CrosspointSignalMonitor
// ---------------------------------------------------------------------------

/// Tracks per-crosspoint signal presence using a [`SignalPresenceDetector`]
/// on each registered `(input, output)` pair.
#[derive(Debug, Clone, Default)]
pub struct CrosspointSignalMonitor {
    detectors: HashMap<(usize, usize), SignalPresenceDetector>,
}

impl CrosspointSignalMonitor {
    /// Creates a new, empty monitor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Registers a crosspoint for monitoring.
    pub fn register_crosspoint(
        &mut self,
        input: usize,
        output: usize,
        threshold_dbfs: f32,
        holdoff_ms: u32,
    ) {
        self.detectors.insert(
            (input, output),
            SignalPresenceDetector::new(threshold_dbfs, holdoff_ms),
        );
    }

    /// Processes a level sample for the given crosspoint.
    ///
    /// Returns `None` if the crosspoint is not registered or no state
    /// transition occurred.
    pub fn process_crosspoint(
        &mut self,
        input: usize,
        output: usize,
        level_dbfs: f32,
        sample_rate_hz: u32,
    ) -> Option<SignalEvent> {
        let det = self.detectors.get_mut(&(input, output))?;
        det.process(level_dbfs, sample_rate_hz)
    }

    /// Returns `true` if the given crosspoint currently has signal present.
    pub fn is_crosspoint_present(&self, input: usize, output: usize) -> bool {
        self.detectors
            .get(&(input, output))
            .map(|d| d.is_present())
            .unwrap_or(false)
    }

    /// Returns the number of crosspoints with signal currently present.
    pub fn active_crosspoints(&self) -> usize {
        self.detectors.values().filter(|d| d.is_present()).count()
    }

    /// Returns the total number of registered crosspoints.
    pub fn registered_count(&self) -> usize {
        self.detectors.len()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_monitor() -> SignalMonitor {
        let mut m = SignalMonitor::new();
        // min_level = -60 dBFS, max_snr = 100.0 (effectively disabled)
        m.register_port("cam1", -60.0, 100.0);
        m.register_port("cam2", -60.0, 100.0);
        m
    }

    #[test]
    fn test_initial_status_unknown() {
        let m = make_monitor();
        assert_eq!(m.status("cam1"), SignalStatus::Unknown);
    }

    #[test]
    fn test_unregistered_port_unknown() {
        let m = make_monitor();
        assert_eq!(m.status("no_such_port"), SignalStatus::Unknown);
    }

    #[test]
    fn test_submit_ok_sample() {
        let mut m = make_monitor();
        let s = m.submit("cam1", SignalSample::now(-20.0, 50.0));
        assert_eq!(s, Some(SignalStatus::Ok));
        assert_eq!(m.status("cam1"), SignalStatus::Ok);
    }

    #[test]
    fn test_submit_lost_below_min_level() {
        let mut m = make_monitor();
        // Level below -60 dBFS → Lost
        let s = m.submit("cam1", SignalSample::now(-80.0, 50.0));
        assert_eq!(s, Some(SignalStatus::Lost));
    }

    #[test]
    fn test_submit_warning_snr_exceeded() {
        let mut m = SignalMonitor::new();
        // max_snr = 30 → any SNR above 30 triggers Warning
        m.register_port("mic", -60.0, 30.0);
        let s = m.submit("mic", SignalSample::now(-10.0, 35.0));
        assert_eq!(s, Some(SignalStatus::Warning));
    }

    #[test]
    fn test_submit_unregistered_returns_none() {
        let mut m = SignalMonitor::new();
        let result = m.submit("ghost", SignalSample::now(0.0, 60.0));
        assert!(result.is_none());
    }

    #[test]
    fn test_fault_count_increments_on_loss() {
        let mut m = make_monitor();
        m.submit("cam1", SignalSample::now(-80.0, 50.0));
        m.submit("cam1", SignalSample::now(-90.0, 50.0));
        assert_eq!(m.fault_count("cam1"), 2);
    }

    #[test]
    fn test_fault_count_zero_on_ok() {
        let mut m = make_monitor();
        m.submit("cam1", SignalSample::now(-10.0, 50.0));
        assert_eq!(m.fault_count("cam1"), 0);
    }

    #[test]
    fn test_last_sample_stored() {
        let mut m = make_monitor();
        m.submit("cam1", SignalSample::now(-15.0, 40.0));
        let sample = m.last_sample("cam1");
        assert!(sample.is_some());
        assert!((sample.expect("should succeed in test").level_db - (-15.0)).abs() < 0.001);
    }

    #[test]
    fn test_port_count() {
        let m = make_monitor();
        assert_eq!(m.port_count(), 2);
    }

    #[test]
    fn test_report_all_ok() {
        let mut m = make_monitor();
        m.submit("cam1", SignalSample::now(-10.0, 40.0));
        m.submit("cam2", SignalSample::now(-5.0, 35.0));
        let report = m.report();
        assert!(report.all_ok());
        assert!(!report.has_alerts());
    }

    #[test]
    fn test_report_has_alerts_on_loss() {
        let mut m = make_monitor();
        m.submit("cam1", SignalSample::now(-80.0, 40.0));
        let report = m.report();
        assert!(report.has_alerts());
        assert_eq!(report.lost, 1);
    }

    #[test]
    fn test_signal_status_label() {
        assert_eq!(SignalStatus::Ok.label(), "ok");
        assert_eq!(SignalStatus::Warning.label(), "warning");
        assert_eq!(SignalStatus::Lost.label(), "lost");
        assert_eq!(SignalStatus::Unknown.label(), "unknown");
    }

    #[test]
    fn test_signal_status_needs_attention() {
        assert!(!SignalStatus::Ok.needs_attention());
        assert!(SignalStatus::Warning.needs_attention());
        assert!(SignalStatus::Lost.needs_attention());
        assert!(!SignalStatus::Unknown.needs_attention());
    }

    #[test]
    fn test_signal_status_is_operational() {
        assert!(SignalStatus::Ok.is_operational());
        assert!(!SignalStatus::Warning.is_operational());
        assert!(!SignalStatus::Lost.is_operational());
    }

    #[test]
    fn test_report_unknown_before_submission() {
        let m = make_monitor();
        let report = m.report();
        assert_eq!(report.unknown, 2);
        assert!(!report.all_ok());
    }

    // -----------------------------------------------------------------------
    // SignalPresenceDetector tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_signal_event_enum_variants() {
        let present = SignalEvent::Present;
        let absent = SignalEvent::Absent;
        assert_eq!(present, SignalEvent::Present);
        assert_eq!(absent, SignalEvent::Absent);
        assert_ne!(present, absent);
    }

    #[test]
    fn test_signal_presence_detector_new() {
        let d = SignalPresenceDetector::new(-40.0, 100);
        assert!((d.threshold_dbfs() - (-40.0)).abs() < 1e-6);
        assert!(!d.is_present());
    }

    #[test]
    fn test_detector_is_present_initial_false() {
        let d = SignalPresenceDetector::new(-40.0, 50);
        assert!(!d.is_present());
    }

    #[test]
    fn test_detector_threshold_getter() {
        let d = SignalPresenceDetector::new(-20.0, 10);
        assert!((d.threshold_dbfs() - (-20.0)).abs() < 1e-6);
    }

    #[test]
    fn test_detector_no_event_below_holdoff() {
        // holdoff_ms=100, sample_rate=1000 → 100 samples needed
        let mut d = SignalPresenceDetector::new(-40.0, 100);
        // Feed 99 above-threshold samples; should not yet emit Present
        for _ in 0..99 {
            let ev = d.process(-20.0, 1000);
            assert!(ev.is_none(), "should not emit before holdoff reached");
        }
        assert!(!d.is_present());
    }

    #[test]
    fn test_detector_present_after_holdoff() {
        // holdoff_ms=3, sample_rate=1000 → 3 samples
        let mut d = SignalPresenceDetector::new(-40.0, 3);
        let ev1 = d.process(-20.0, 1000);
        assert!(ev1.is_none());
        let ev2 = d.process(-20.0, 1000);
        assert!(ev2.is_none());
        let ev3 = d.process(-20.0, 1000);
        assert_eq!(ev3, Some(SignalEvent::Present));
        assert!(d.is_present());
    }

    #[test]
    fn test_detector_absent_after_holdoff() {
        // First go present, then go absent
        let mut d = SignalPresenceDetector::new(-40.0, 2);
        // Feed 2 above-threshold → Present
        d.process(-20.0, 1000);
        d.process(-20.0, 1000);
        assert!(d.is_present());
        // Now go below threshold for 2 samples → Absent
        let ev1 = d.process(-60.0, 1000);
        assert!(ev1.is_none());
        let ev2 = d.process(-60.0, 1000);
        assert_eq!(ev2, Some(SignalEvent::Absent));
        assert!(!d.is_present());
    }

    #[test]
    fn test_detector_hysteresis() {
        // Signal goes present, then dips below threshold; must wait holdoff_count
        let mut d = SignalPresenceDetector::new(-40.0, 3);
        // Go present
        for _ in 0..3 {
            d.process(-20.0, 1000);
        }
        assert!(d.is_present());
        // Only 2 below-threshold samples — not enough to go absent
        d.process(-60.0, 1000);
        d.process(-60.0, 1000);
        assert!(
            d.is_present(),
            "should still be present after only 2 absent samples"
        );
    }

    #[test]
    fn test_detector_reset() {
        let mut d = SignalPresenceDetector::new(-40.0, 2);
        d.process(-20.0, 1000);
        d.process(-20.0, 1000);
        assert!(d.is_present());
        d.reset();
        assert!(!d.is_present());
    }

    // -----------------------------------------------------------------------
    // CrosspointSignalMonitor tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_crosspoint_monitor_new() {
        let m = CrosspointSignalMonitor::new();
        assert_eq!(m.registered_count(), 0);
        assert_eq!(m.active_crosspoints(), 0);
    }

    #[test]
    fn test_crosspoint_register_and_process() {
        let mut m = CrosspointSignalMonitor::new();
        m.register_crosspoint(0, 0, -40.0, 2);
        // holdoff = 2 samples at 1000 Hz
        let ev1 = m.process_crosspoint(0, 0, -20.0, 1000);
        assert!(ev1.is_none());
        let ev2 = m.process_crosspoint(0, 0, -20.0, 1000);
        assert_eq!(ev2, Some(SignalEvent::Present));
        assert!(m.is_crosspoint_present(0, 0));
    }

    #[test]
    fn test_crosspoint_active_count() {
        let mut m = CrosspointSignalMonitor::new();
        m.register_crosspoint(0, 0, -40.0, 1);
        m.register_crosspoint(1, 1, -40.0, 1);
        // Make both present (holdoff=1 → 1 sample)
        m.process_crosspoint(0, 0, -20.0, 1000);
        m.process_crosspoint(1, 1, -20.0, 1000);
        assert_eq!(m.active_crosspoints(), 2);
    }

    #[test]
    fn test_crosspoint_registered_count() {
        let mut m = CrosspointSignalMonitor::new();
        m.register_crosspoint(0, 0, -40.0, 10);
        m.register_crosspoint(1, 2, -40.0, 10);
        m.register_crosspoint(3, 5, -40.0, 10);
        assert_eq!(m.registered_count(), 3);
    }

    #[test]
    fn test_crosspoint_unregistered_returns_none() {
        let mut m = CrosspointSignalMonitor::new();
        // Process without registering
        let ev = m.process_crosspoint(99, 99, -20.0, 1000);
        assert!(ev.is_none());
    }

    // -----------------------------------------------------------------------
    // ThresholdAlertEngine tests
    // -----------------------------------------------------------------------

    fn make_engine() -> ThresholdAlertEngine {
        let mut engine = ThresholdAlertEngine::new();
        engine.register(
            "ch1",
            ThresholdConfig {
                overmodulation_db: -1.0,
                signal_lost_db: -60.0,
                signal_present_db: -50.0,
                hold_off_count: 3,
            },
        );
        engine
    }

    #[test]
    fn test_engine_new_empty() {
        let engine = ThresholdAlertEngine::new();
        assert_eq!(engine.port_count(), 0);
    }

    #[test]
    fn test_engine_register_port() {
        let engine = make_engine();
        assert_eq!(engine.port_count(), 1);
    }

    #[test]
    fn test_engine_evaluate_unregistered_returns_empty() {
        let mut engine = ThresholdAlertEngine::new();
        let alerts = engine.evaluate("ghost", -20.0);
        assert!(alerts.is_empty());
    }

    #[test]
    fn test_engine_no_alert_before_holdoff() {
        let mut engine = make_engine();
        // Feed 2 above-threshold samples (hold_off_count = 3) → no alert yet
        let a1 = engine.evaluate("ch1", -45.0);
        let a2 = engine.evaluate("ch1", -45.0);
        assert!(a1.is_empty());
        assert!(a2.is_empty());
        assert!(!engine.is_signal_present("ch1"));
    }

    #[test]
    fn test_engine_signal_present_after_holdoff() {
        let mut engine = make_engine();
        // hold_off_count = 3 → need 3 consecutive samples >= signal_present_db (-50)
        engine.evaluate("ch1", -45.0);
        engine.evaluate("ch1", -45.0);
        let alerts = engine.evaluate("ch1", -45.0);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].kind, AlertKind::SignalPresent);
        assert_eq!(alerts[0].severity, AlertSeverity::Info);
        assert!(engine.is_signal_present("ch1"));
    }

    #[test]
    fn test_engine_signal_lost_after_holdoff() {
        let mut engine = make_engine();
        // First go present (3 samples)
        engine.evaluate("ch1", -45.0);
        engine.evaluate("ch1", -45.0);
        engine.evaluate("ch1", -45.0);
        assert!(engine.is_signal_present("ch1"));
        // Now go lost: 3 consecutive samples below -60 dBFS
        engine.evaluate("ch1", -70.0);
        engine.evaluate("ch1", -70.0);
        let alerts = engine.evaluate("ch1", -70.0);
        assert_eq!(alerts.len(), 1);
        assert_eq!(alerts[0].kind, AlertKind::SignalLost);
        assert_eq!(alerts[0].severity, AlertSeverity::Critical);
        assert!(!engine.is_signal_present("ch1"));
    }

    #[test]
    fn test_engine_overmodulation_alert() {
        let mut engine = make_engine();
        // Feed 3 overmodulated samples (>= -1 dBFS)
        engine.evaluate("ch1", -0.5);
        engine.evaluate("ch1", -0.5);
        let alerts = engine.evaluate("ch1", -0.5);
        assert!(!alerts.is_empty());
        let overmod = alerts.iter().find(|a| a.kind == AlertKind::Overmodulation);
        assert!(overmod.is_some());
        assert_eq!(
            overmod.expect("overmod alert").severity,
            AlertSeverity::Critical
        );
    }

    #[test]
    fn test_engine_overmod_resets_after_normal_level() {
        let mut engine = make_engine();
        // 2 overmod samples (less than hold_off_count=3)
        engine.evaluate("ch1", -0.5);
        engine.evaluate("ch1", -0.5);
        // Back to normal → streak resets
        engine.evaluate("ch1", -20.0);
        // Only 1 overmod sample after reset — should not fire
        let alerts = engine.evaluate("ch1", -0.5);
        assert!(alerts.iter().all(|a| a.kind != AlertKind::Overmodulation));
    }

    #[test]
    fn test_engine_alert_count_increments() {
        let mut engine = make_engine();
        engine.evaluate("ch1", -45.0);
        engine.evaluate("ch1", -45.0);
        engine.evaluate("ch1", -45.0);
        assert!(engine.alert_count("ch1") >= 1);
    }

    #[test]
    fn test_engine_alert_count_zero_for_unregistered() {
        let engine = ThresholdAlertEngine::new();
        assert_eq!(engine.alert_count("nobody"), 0);
    }

    #[test]
    fn test_engine_reset_clears_state() {
        let mut engine = make_engine();
        // Go present
        engine.evaluate("ch1", -45.0);
        engine.evaluate("ch1", -45.0);
        engine.evaluate("ch1", -45.0);
        assert!(engine.is_signal_present("ch1"));
        engine.reset("ch1");
        assert!(!engine.is_signal_present("ch1"));
        assert_eq!(engine.alert_count("ch1"), 0);
    }

    #[test]
    fn test_engine_hysteresis_zone_no_state_change() {
        let mut engine = make_engine();
        // signal_lost_db = -60, signal_present_db = -50
        // Hysteresis zone: (-60, -50) — no state change expected
        for _ in 0..10 {
            let alerts = engine.evaluate("ch1", -55.0);
            assert!(alerts.is_empty());
        }
        assert!(!engine.is_signal_present("ch1"));
    }

    #[test]
    fn test_engine_alert_severity_ordering() {
        assert!(AlertSeverity::Info < AlertSeverity::Warning);
        assert!(AlertSeverity::Warning < AlertSeverity::Critical);
    }

    #[test]
    fn test_alert_kind_labels() {
        assert_eq!(AlertKind::SignalPresent.label(), "signal_present");
        assert_eq!(AlertKind::SignalLost.label(), "signal_lost");
        assert_eq!(AlertKind::Overmodulation.label(), "overmodulation");
        assert_eq!(AlertKind::BelowNoiseFloor.label(), "below_noise_floor");
        assert_eq!(AlertKind::PhaseInversion.label(), "phase_inversion");
    }

    #[test]
    fn test_alert_severity_labels() {
        assert_eq!(AlertSeverity::Info.label(), "info");
        assert_eq!(AlertSeverity::Warning.label(), "warning");
        assert_eq!(AlertSeverity::Critical.label(), "critical");
    }

    #[test]
    fn test_threshold_config_default() {
        let cfg = ThresholdConfig::default();
        assert!((cfg.overmodulation_db - (-1.0)).abs() < f32::EPSILON);
        assert!((cfg.signal_lost_db - (-60.0)).abs() < f32::EPSILON);
        assert!((cfg.signal_present_db - (-50.0)).abs() < f32::EPSILON);
        assert_eq!(cfg.hold_off_count, 3);
    }

    #[test]
    fn test_engine_two_ports_independent() {
        let mut engine = ThresholdAlertEngine::new();
        let cfg = ThresholdConfig {
            hold_off_count: 1,
            ..ThresholdConfig::default()
        };
        engine.register("a", cfg.clone());
        engine.register("b", cfg);
        // Port a goes present
        let alerts_a = engine.evaluate("a", -45.0);
        // Port b stays below signal_lost → no state to transition from
        let alerts_b = engine.evaluate("b", -80.0);
        assert_eq!(alerts_a.len(), 1);
        assert_eq!(alerts_a[0].kind, AlertKind::SignalPresent);
        assert!(alerts_b.iter().all(|a| a.kind != AlertKind::SignalLost));
    }

    #[test]
    fn test_engine_signal_present_port_name_in_alert() {
        let mut engine = make_engine();
        engine.evaluate("ch1", -45.0);
        engine.evaluate("ch1", -45.0);
        let alerts = engine.evaluate("ch1", -45.0);
        assert!(!alerts.is_empty());
        assert!(alerts[0].port_name.contains("ch1"));
        assert!(!alerts[0].message.is_empty());
    }
}
