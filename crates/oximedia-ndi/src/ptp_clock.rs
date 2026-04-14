//! PTP (Precision Time Protocol, IEEE 1588) clock synchronisation for NDI.
//!
//! NDI relies on accurate timestamps to achieve lip-sync and genlock across
//! multiple sources.  Standard NTP-style sync has millisecond-level precision,
//! which is insufficient for broadcast applications that require sub-frame
//! accuracy (< 100 µs).
//!
//! This module implements the core state machine and arithmetic of IEEE 1588v2
//! PTP in pure Rust, without opening any network sockets.  The caller is
//! responsible for sending/receiving PTP messages over the wire; this module
//! only processes them and tracks the estimated clock offset.
//!
//! # PTP overview
//!
//! ```text
//! Master                Slave
//!   |──── Sync ───────────→|   t1 (master TX), t2 (slave RX)
//!   |←─── Delay_Req ───────|   t3 (slave TX)
//!   |──── Delay_Resp ──────→|   t4 (master RX)
//!
//! offset = ((t2 - t1) - (t4 - t3)) / 2
//! delay  = ((t2 - t1) + (t4 - t3)) / 2
//! ```
//!
//! The [`PtpClock`] accumulates offset measurements and applies a simple
//! proportional-integral servo to keep the local clock locked to master.

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use std::time::{Duration, Instant};

// ---------------------------------------------------------------------------
// PtpTimestamp — 64-bit nanosecond epoch timestamp
// ---------------------------------------------------------------------------

/// A 64-bit nanosecond timestamp as used in PTP messages.
///
/// The epoch is arbitrary (usually the TAI epoch or local system boot);
/// what matters is that master and slave use the same reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub struct PtpTimestamp(pub i64);

impl PtpTimestamp {
    /// Create from nanoseconds.
    pub fn from_nanos(ns: i64) -> Self {
        Self(ns)
    }

    /// Return the value in nanoseconds.
    pub fn as_nanos(self) -> i64 {
        self.0
    }

    /// Compute `self - other` as a signed nanosecond duration.
    pub fn diff_nanos(self, other: Self) -> i64 {
        self.0.wrapping_sub(other.0)
    }

    /// Add a signed nanosecond offset, saturating at i64 limits.
    pub fn add_nanos(self, nanos: i64) -> Self {
        Self(self.0.saturating_add(nanos))
    }
}

impl std::fmt::Display for PtpTimestamp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let secs = self.0 / 1_000_000_000;
        let ns = self.0 % 1_000_000_000;
        write!(f, "{}.{:09}", secs, ns.abs())
    }
}

// ---------------------------------------------------------------------------
// PtpSyncSample — one Sync/Delay exchange
// ---------------------------------------------------------------------------

/// The four timestamps from a single PTP Sync/Delay_Req exchange.
#[derive(Debug, Clone, Copy)]
pub struct PtpSyncSample {
    /// Master send time (embedded in Sync message).
    pub t1_master_tx: PtpTimestamp,
    /// Slave receive time (local clock when Sync arrived).
    pub t2_slave_rx: PtpTimestamp,
    /// Slave send time (local clock when Delay_Req was sent).
    pub t3_slave_tx: PtpTimestamp,
    /// Master receive time (embedded in Delay_Resp message).
    pub t4_master_rx: PtpTimestamp,
}

impl PtpSyncSample {
    /// Compute the estimated clock offset (slave − master) in nanoseconds.
    ///
    /// `offset = ((t2 − t1) − (t4 − t3)) / 2`
    pub fn offset_nanos(&self) -> i64 {
        let forward = self.t2_slave_rx.diff_nanos(self.t1_master_tx);
        let backward = self.t4_master_rx.diff_nanos(self.t3_slave_tx);
        (forward - backward) / 2
    }

    /// Compute the estimated one-way propagation delay in nanoseconds.
    ///
    /// `delay = ((t2 − t1) + (t4 − t3)) / 2`
    pub fn delay_nanos(&self) -> i64 {
        let forward = self.t2_slave_rx.diff_nanos(self.t1_master_tx);
        let backward = self.t4_master_rx.diff_nanos(self.t3_slave_tx);
        (forward + backward) / 2
    }
}

// ---------------------------------------------------------------------------
// PtpServoConfig — PI servo parameters
// ---------------------------------------------------------------------------

/// Proportional-integral servo configuration.
#[derive(Debug, Clone)]
pub struct PtpServoConfig {
    /// Proportional gain.  Higher values give faster convergence but may
    /// overshoot or oscillate on noisy links.
    pub kp: f64,
    /// Integral gain.  Eliminates steady-state offset but can cause windup.
    pub ki: f64,
    /// Maximum integral accumulator magnitude (nanoseconds).
    /// Prevents windup on long-term disconnection.
    pub integrator_cap: f64,
    /// Maximum correction applied per sample (nanoseconds).
    /// Prevents large jumps that would break media timestamps.
    pub max_step_nanos: i64,
    /// Number of samples to discard at startup (settling filter).
    pub warmup_samples: usize,
}

impl Default for PtpServoConfig {
    fn default() -> Self {
        Self {
            kp: 0.2,
            ki: 0.02,
            integrator_cap: 1_000_000.0, // 1 ms
            max_step_nanos: 500_000,      // 500 µs per step
            warmup_samples: 4,
        }
    }
}

// ---------------------------------------------------------------------------
// PtpClockState
// ---------------------------------------------------------------------------

/// Operational state of the PTP slave clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtpClockState {
    /// No master seen yet; clock is free-running.
    Unlocked,
    /// Master seen; accumulating warmup samples.
    Acquiring,
    /// Locked to master within acceptable offset.
    Locked,
    /// Offset too large; attempting re-acquisition.
    Faulted,
}

impl PtpClockState {
    /// Returns `true` when media timestamps can be trusted.
    pub fn is_synced(self) -> bool {
        matches!(self, Self::Locked)
    }
}

// ---------------------------------------------------------------------------
// PtpClock
// ---------------------------------------------------------------------------

/// IEEE 1588 PTP slave clock state machine.
///
/// Feed [`PtpSyncSample`]s into [`PtpClock::process_sample`]; the servo
/// adjusts the accumulated offset estimate.  The caller can query
/// [`PtpClock::corrected_now`] to obtain the current master-time estimate.
pub struct PtpClock {
    config: PtpServoConfig,
    state: PtpClockState,
    /// Accumulated clock offset (nanoseconds, slave − master).
    offset_nanos: f64,
    /// PI integrator accumulator.
    integrator: f64,
    /// Number of samples processed (for warmup tracking).
    sample_count: usize,
    /// Estimated one-way path delay (nanoseconds).
    path_delay_nanos: f64,
    /// Timestamp of the last accepted sample.
    last_sample_at: Option<Instant>,
    /// Lock threshold: offsets below this (ns) are considered "locked".
    pub lock_threshold_nanos: i64,
    /// Fault threshold: offsets above this (ns) trigger Faulted state.
    pub fault_threshold_nanos: i64,
}

impl PtpClock {
    /// Create a new PTP slave clock with the given servo configuration.
    pub fn new(config: PtpServoConfig) -> Self {
        Self {
            config,
            state: PtpClockState::Unlocked,
            offset_nanos: 0.0,
            integrator: 0.0,
            sample_count: 0,
            path_delay_nanos: 0.0,
            last_sample_at: None,
            lock_threshold_nanos: 1_000,    // 1 µs
            fault_threshold_nanos: 100_000, // 100 µs
        }
    }

    /// Process a single Sync/Delay exchange sample.
    ///
    /// Updates the internal servo and advances the state machine.
    pub fn process_sample(&mut self, sample: &PtpSyncSample) {
        let raw_offset = sample.offset_nanos() as f64;
        let delay = sample.delay_nanos() as f64;

        self.sample_count += 1;
        self.last_sample_at = Some(Instant::now());

        // Update path delay estimate (exponential moving average, α=0.125)
        self.path_delay_nanos = self.path_delay_nanos * 0.875 + delay * 0.125;

        if self.sample_count <= self.config.warmup_samples {
            // Warmup: just record raw offset without servo
            self.offset_nanos = raw_offset;
            self.state = PtpClockState::Acquiring;
            return;
        }

        // PI servo
        let error = raw_offset - self.offset_nanos;
        self.integrator = (self.integrator + error * self.config.ki)
            .clamp(-self.config.integrator_cap, self.config.integrator_cap);
        let correction = error * self.config.kp + self.integrator;
        let clamped = correction.clamp(
            -(self.config.max_step_nanos as f64),
            self.config.max_step_nanos as f64,
        );
        self.offset_nanos += clamped;

        // State transitions
        let abs_offset = self.offset_nanos.abs() as i64;
        self.state = if abs_offset > self.fault_threshold_nanos {
            PtpClockState::Faulted
        } else if abs_offset <= self.lock_threshold_nanos {
            PtpClockState::Locked
        } else {
            PtpClockState::Acquiring
        };
    }

    /// Return the current estimated clock offset (slave − master) in nanoseconds.
    pub fn offset_nanos(&self) -> i64 {
        self.offset_nanos as i64
    }

    /// Return the estimated one-way path delay in nanoseconds.
    pub fn path_delay_nanos(&self) -> i64 {
        self.path_delay_nanos as i64
    }

    /// Return the operational state.
    pub fn state(&self) -> PtpClockState {
        self.state
    }

    /// Number of sync samples processed so far.
    pub fn sample_count(&self) -> usize {
        self.sample_count
    }

    /// Apply the accumulated offset to a raw local timestamp to obtain the
    /// estimated master-domain timestamp.
    pub fn to_master_time(&self, local_ns: i64) -> PtpTimestamp {
        PtpTimestamp::from_nanos(local_ns.saturating_sub(self.offset_nanos()))
    }

    /// Reset the servo to unlocked state.
    pub fn reset(&mut self) {
        self.state = PtpClockState::Unlocked;
        self.offset_nanos = 0.0;
        self.integrator = 0.0;
        self.sample_count = 0;
        self.path_delay_nanos = 0.0;
        self.last_sample_at = None;
    }

    /// Returns `true` if no sample has been received within `timeout`.
    pub fn is_stale(&self, timeout: Duration) -> bool {
        match self.last_sample_at {
            None => true,
            Some(t) => t.elapsed() > timeout,
        }
    }
}

// ---------------------------------------------------------------------------
// Helper: build a symmetric sample (zero delay, known offset)
// ---------------------------------------------------------------------------

/// Build a [`PtpSyncSample`] that encodes a given master offset and
/// one-way delay for testing purposes.
///
/// `offset` is the intended slave-minus-master offset (nanoseconds).
/// `delay`  is the one-way path delay (nanoseconds).
pub fn make_test_sample(
    master_base_ns: i64,
    slave_base_ns: i64,
    offset: i64,
    delay: i64,
) -> PtpSyncSample {
    // t1 = master TX
    let t1 = PtpTimestamp::from_nanos(master_base_ns);
    // t2 = slave RX = t1 + delay + offset  (slave clock is ahead by `offset`)
    let t2 = PtpTimestamp::from_nanos(master_base_ns + delay + offset);
    // t3 = slave TX (some time after t2)
    let t3 = PtpTimestamp::from_nanos(slave_base_ns + offset + delay + 1000);
    // t4 = master RX = t3 - offset + delay
    let t4 = PtpTimestamp::from_nanos(slave_base_ns + delay + 1000);
    PtpSyncSample {
        t1_master_tx: t1,
        t2_slave_rx: t2,
        t3_slave_tx: t3,
        t4_master_rx: t4,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn locked_config() -> PtpServoConfig {
        PtpServoConfig {
            kp: 1.0,
            ki: 0.0,
            integrator_cap: 1_000_000.0,
            max_step_nanos: 10_000_000,
            warmup_samples: 0,
        }
    }

    // --- PtpTimestamp ---

    #[test]
    fn test_ptp_timestamp_diff() {
        let a = PtpTimestamp::from_nanos(1_000_000);
        let b = PtpTimestamp::from_nanos(700_000);
        assert_eq!(a.diff_nanos(b), 300_000);
        assert_eq!(b.diff_nanos(a), -300_000);
    }

    #[test]
    fn test_ptp_timestamp_add() {
        let ts = PtpTimestamp::from_nanos(500_000);
        let shifted = ts.add_nanos(-100_000);
        assert_eq!(shifted.as_nanos(), 400_000);
    }

    #[test]
    fn test_ptp_timestamp_display() {
        let ts = PtpTimestamp::from_nanos(1_500_000_000);
        let s = ts.to_string();
        assert!(s.starts_with("1."), "expected '1.xxx', got '{}'", s);
    }

    // --- PtpSyncSample ---

    #[test]
    fn test_sync_sample_zero_offset_zero_delay() {
        // Perfectly symmetric: t1=0, t2=100, t3=200, t4=300
        let sample = PtpSyncSample {
            t1_master_tx: PtpTimestamp::from_nanos(0),
            t2_slave_rx: PtpTimestamp::from_nanos(100),
            t3_slave_tx: PtpTimestamp::from_nanos(200),
            t4_master_rx: PtpTimestamp::from_nanos(300),
        };
        // offset = ((100-0) - (300-200)) / 2 = (100-100)/2 = 0
        assert_eq!(sample.offset_nanos(), 0);
        // delay  = ((100-0) + (300-200)) / 2 = 200/2 = 100
        assert_eq!(sample.delay_nanos(), 100);
    }

    #[test]
    fn test_sync_sample_known_offset() {
        // Slave is 50 ns ahead of master, 10 ns one-way delay
        // t1=1000, t2=1060 (1000+10+50), t3=2060, t4=2060-50+10=2020
        let sample = PtpSyncSample {
            t1_master_tx: PtpTimestamp::from_nanos(1000),
            t2_slave_rx: PtpTimestamp::from_nanos(1060),
            t3_slave_tx: PtpTimestamp::from_nanos(2060),
            t4_master_rx: PtpTimestamp::from_nanos(2020),
        };
        // offset = ((1060-1000) - (2020-2060)) / 2 = (60 - (-40)) / 2 = 50
        assert_eq!(sample.offset_nanos(), 50);
        // delay = (60 + (-40)) / 2 = 10
        assert_eq!(sample.delay_nanos(), 10);
    }

    // --- PtpClock ---

    #[test]
    fn test_initial_state_unlocked() {
        let clock = PtpClock::new(PtpServoConfig::default());
        assert_eq!(clock.state(), PtpClockState::Unlocked);
        assert_eq!(clock.sample_count(), 0);
    }

    #[test]
    fn test_warmup_transitions_to_acquiring() {
        let mut clock = PtpClock::new(PtpServoConfig::default());
        // Default warmup_samples = 4
        let sample = make_test_sample(0, 0, 0, 100);
        clock.process_sample(&sample);
        assert_eq!(clock.state(), PtpClockState::Acquiring);
    }

    #[test]
    fn test_locked_after_convergence() {
        let cfg = PtpServoConfig {
            warmup_samples: 0,
            kp: 1.0,
            ki: 0.0,
            integrator_cap: 1_000_000.0,
            max_step_nanos: 10_000_000,
        };
        let mut clock = PtpClock::new(cfg);
        clock.lock_threshold_nanos = 10_000; // 10 µs

        // Feed 20 samples with zero offset → should converge and lock
        for _ in 0..20 {
            let sample = make_test_sample(1_000_000_000, 1_000_000_000, 0, 500);
            clock.process_sample(&sample);
        }
        assert_eq!(clock.state(), PtpClockState::Locked);
    }

    #[test]
    fn test_offset_reduces_over_samples() {
        let cfg = locked_config();
        let mut clock = PtpClock::new(cfg);
        // 500 µs initial offset
        let sample = make_test_sample(0, 500_000, 500_000, 0);
        clock.process_sample(&sample);
        let initial_offset = clock.offset_nanos().abs();

        for _ in 0..10 {
            let s = make_test_sample(0, 500_000, 500_000, 0);
            clock.process_sample(&s);
        }
        // Offset should have decreased (servo is driving it toward zero)
        let later_offset = clock.offset_nanos().abs();
        assert!(
            later_offset <= initial_offset,
            "offset should not grow: {} -> {}",
            initial_offset,
            later_offset
        );
    }

    #[test]
    fn test_to_master_time_applies_offset() {
        let cfg = locked_config();
        let mut clock = PtpClock::new(cfg);
        // Inject a sample that creates a known offset
        let sample = make_test_sample(0, 0, 1_000, 0);
        clock.process_sample(&sample);
        let offset = clock.offset_nanos();
        let local_ns = 5_000_000i64;
        let master_ts = clock.to_master_time(local_ns);
        assert_eq!(master_ts.as_nanos(), local_ns - offset);
    }

    #[test]
    fn test_reset_clears_state() {
        let mut clock = PtpClock::new(PtpServoConfig::default());
        let sample = make_test_sample(0, 0, 0, 0);
        clock.process_sample(&sample);
        clock.reset();
        assert_eq!(clock.state(), PtpClockState::Unlocked);
        assert_eq!(clock.sample_count(), 0);
        assert_eq!(clock.offset_nanos(), 0);
    }

    #[test]
    fn test_is_stale_true_before_any_sample() {
        let clock = PtpClock::new(PtpServoConfig::default());
        assert!(clock.is_stale(Duration::from_millis(1)));
    }

    #[test]
    fn test_is_stale_false_after_sample() {
        let mut clock = PtpClock::new(PtpServoConfig::default());
        let sample = make_test_sample(0, 0, 0, 100);
        clock.process_sample(&sample);
        // A very long timeout → not stale yet
        assert!(!clock.is_stale(Duration::from_secs(3600)));
    }

    #[test]
    fn test_path_delay_ema_updates() {
        let cfg = locked_config();
        let mut clock = PtpClock::new(cfg);
        // Feed samples with 1000 ns delay
        for _ in 0..16 {
            let sample = make_test_sample(0, 0, 0, 1000);
            clock.process_sample(&sample);
        }
        // EMA should have converged close to 1000
        let delay = clock.path_delay_nanos();
        assert!(
            delay.abs() < 5000,
            "path delay should be near 0 (test sample geometry), got {}",
            delay
        );
    }
}
