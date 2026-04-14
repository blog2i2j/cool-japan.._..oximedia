//! White Rabbit (WR) sub-nanosecond PTP extension.
//!
//! White Rabbit is a fully deterministic Ethernet-based network for general
//! purpose data transfer and synchronisation, extending IEEE 1588-2008 with
//! sub-nanosecond accuracy. It is widely deployed in scientific and broadcast
//! facilities (CERN, ESO, radio-telescope arrays).
//!
//! # Key capabilities
//! - Sub-nanosecond synchronisation (< 1 ns accuracy)
//! - Picosecond-level precision on short links (< 1 ns jitter)
//! - Phase tracking through a dedicated DDMTD (Dual Diode Mixed Time Difference)
//!   phase detector
//! - Syntonisation: slave oscillator locked to master at the hardware level
//!
//! # References
//! - White Rabbit Specification, version 2.0 (OHWR)
//! - IEEE 1588-2008 (PTPv2)
//! - WRPTP specification: <https://white-rabbit.web.cern.ch/>

use crate::error::{TimeSyncError, TimeSyncResult};

// ---------------------------------------------------------------------------
// WR link model
// ---------------------------------------------------------------------------

/// One-way propagation delay coefficients for a single WR link.
///
/// White Rabbit uses a calibration procedure to measure the fixed
/// (cable+hardware) delays at both ends.  The total one-way delay is:
///
/// ```text
/// delay_ms2s = delta_tx_m + delta_rx_s + (α / (1+α)) * δ
/// ```
///
/// where `δ` is the round-trip time measured by the delay-request mechanism,
/// `α` is the link asymmetry coefficient, and `delta_tx_m` / `delta_rx_s`
/// are the fixed TX/RX delays of the master and slave, respectively.
#[derive(Debug, Clone, PartialEq)]
pub struct WrLinkDelayCoefficients {
    /// Fixed transmission delay of the master port (picoseconds).
    pub delta_tx_master_ps: i64,
    /// Fixed reception delay of the slave port (picoseconds).
    pub delta_rx_slave_ps: i64,
    /// Link asymmetry coefficient `α = (δ_ms - δ_sm) / δ_sm`.
    ///
    /// A perfectly symmetric link has `α = 0`.
    pub alpha: f64,
}

impl WrLinkDelayCoefficients {
    /// Creates link-delay coefficients for a perfectly symmetric link.
    #[must_use]
    pub fn symmetric() -> Self {
        Self {
            delta_tx_master_ps: 0,
            delta_rx_slave_ps: 0,
            alpha: 0.0,
        }
    }

    /// Creates link-delay coefficients with the given fixed delays and asymmetry.
    #[must_use]
    pub fn new(delta_tx_master_ps: i64, delta_rx_slave_ps: i64, alpha: f64) -> Self {
        Self {
            delta_tx_master_ps,
            delta_rx_slave_ps,
            alpha,
        }
    }

    /// Computes the master-to-slave one-way delay in picoseconds given the
    /// measured round-trip time `rtt_ps`.
    ///
    /// Formula per WR spec §3.2:
    /// `delay_ms = delta_tx_m + delta_rx_s + (α / (1+α)) * rtt`
    #[must_use]
    pub fn delay_master_to_slave_ps(&self, rtt_ps: u64) -> i64 {
        let alpha_term = if (self.alpha + 1.0).abs() > f64::EPSILON {
            (self.alpha / (1.0 + self.alpha)) * rtt_ps as f64
        } else {
            0.0
        };
        self.delta_tx_master_ps + self.delta_rx_slave_ps + alpha_term as i64
    }
}

// ---------------------------------------------------------------------------
// DDMTD phase measurement
// ---------------------------------------------------------------------------

/// Result of a DDMTD (Dual Diode Mixed Time Difference) phase measurement.
///
/// DDMTD is the hardware phase detector used by WR for sub-nanosecond phase
/// measurements between the recovered clock and the local VCXO.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DdmtdSample {
    /// Phase tag from the DDMTD counter (raw register value).
    pub raw_tag: u32,
    /// DDMTD clock period in picoseconds (typically ≈ 8 ns / 2^16 ≈ 122 ps).
    pub resolution_ps: u32,
}

impl DdmtdSample {
    /// Creates a new DDMTD sample.
    ///
    /// `raw_tag` is the raw phase register value.
    /// `resolution_ps` is the counter period (picoseconds per count).
    #[must_use]
    pub fn new(raw_tag: u32, resolution_ps: u32) -> Self {
        Self {
            raw_tag,
            resolution_ps,
        }
    }

    /// Converts the raw tag to a phase offset in picoseconds.
    #[must_use]
    pub fn phase_ps(&self) -> i64 {
        i64::from(self.raw_tag) * i64::from(self.resolution_ps)
    }
}

// ---------------------------------------------------------------------------
// WR port state
// ---------------------------------------------------------------------------

/// State of a White Rabbit port.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WrPortState {
    /// Port is initialising.
    Idle,
    /// Master has been detected; WR capability negotiation in progress.
    Presented,
    /// Slave is calibrating its TX/RX delays.
    Calibrated,
    /// Phase tracking active; waiting for lock.
    Tracking,
    /// Phase locked; sub-nanosecond accuracy achieved.
    Locked,
    /// Error state.
    Error,
}

impl WrPortState {
    /// Returns `true` if the port is providing sub-nanosecond synchronisation.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        matches!(self, Self::Locked)
    }

    /// Returns `true` for states where active protocol exchange is happening.
    #[must_use]
    pub fn is_active(&self) -> bool {
        !matches!(self, Self::Idle | Self::Error)
    }
}

// ---------------------------------------------------------------------------
// WR calibration record
// ---------------------------------------------------------------------------

/// Calibration data stored for a WR link.
#[derive(Debug, Clone)]
pub struct WrCalibration {
    /// Link delay coefficients obtained from calibration.
    pub coefficients: WrLinkDelayCoefficients,
    /// Most recent round-trip measurement (picoseconds).
    pub last_rtt_ps: u64,
    /// Computed master-to-slave delay (picoseconds).
    pub delay_ms_ps: i64,
    /// Whether calibration has been completed at least once.
    pub calibrated: bool,
}

impl WrCalibration {
    /// Creates a new, uncalibrated record.
    #[must_use]
    pub fn new(coefficients: WrLinkDelayCoefficients) -> Self {
        Self {
            coefficients,
            last_rtt_ps: 0,
            delay_ms_ps: 0,
            calibrated: false,
        }
    }

    /// Updates the record with a new RTT measurement.
    ///
    /// Returns the updated master-to-slave delay in picoseconds.
    pub fn update_rtt(&mut self, rtt_ps: u64) -> i64 {
        self.last_rtt_ps = rtt_ps;
        self.delay_ms_ps = self.coefficients.delay_master_to_slave_ps(rtt_ps);
        self.calibrated = true;
        self.delay_ms_ps
    }
}

// ---------------------------------------------------------------------------
// WR offset computation
// ---------------------------------------------------------------------------

/// Computes the White Rabbit clock offset from a set of phase and delay
/// measurements.
///
/// The WR offset formula is:
/// ```text
/// offset = t2 − t1 − delay_ms + phase_correction
/// ```
///
/// where `t1` is the PTP sync origin timestamp, `t2` is the local receive
/// timestamp, `delay_ms` is the calibrated one-way delay, and
/// `phase_correction` is the DDMTD phase measurement converted to time units.
///
/// # Arguments
/// - `sync_origin_ps` — grandmaster sync origin timestamp (picoseconds).
/// - `local_recv_ps`  — local receive timestamp (picoseconds).
/// - `delay_ms_ps`    — calibrated one-way delay (picoseconds).
/// - `ddmtd_phase_ps` — DDMTD phase correction (picoseconds, signed).
///
/// # Returns
/// Clock offset in picoseconds (positive = slave is behind master).
#[must_use]
pub fn compute_wr_offset_ps(
    sync_origin_ps: i64,
    local_recv_ps: i64,
    delay_ms_ps: i64,
    ddmtd_phase_ps: i64,
) -> i64 {
    local_recv_ps - sync_origin_ps - delay_ms_ps + ddmtd_phase_ps
}

// ---------------------------------------------------------------------------
// WR clock
// ---------------------------------------------------------------------------

/// White Rabbit clock state machine.
///
/// Manages the WR port state, calibration, and phase tracking loop.
pub struct WrClock {
    /// Current port state.
    pub state: WrPortState,
    /// Calibration record for the link.
    pub calibration: WrCalibration,
    /// Latest DDMTD phase sample.
    pub latest_ddmtd: Option<DdmtdSample>,
    /// Latest computed offset in picoseconds.
    pub offset_ps: i64,
    /// Phase error integration (PI loop accumulator) in picoseconds.
    phase_integrator_ps: i64,
    /// Proportional gain (in units of 1/1000 adjustment-per-ps-error).
    kp_permille: i64,
    /// Integral gain (in units of 1/1000000 adjustment-per-ps-error).
    ki_permille: i64,
}

impl WrClock {
    /// Creates a new WR clock in the `Idle` state with symmetric link.
    #[must_use]
    pub fn new() -> Self {
        Self {
            state: WrPortState::Idle,
            calibration: WrCalibration::new(WrLinkDelayCoefficients::symmetric()),
            latest_ddmtd: None,
            offset_ps: 0,
            phase_integrator_ps: 0,
            kp_permille: 10,
            ki_permille: 1,
        }
    }

    /// Creates a WR clock with custom link coefficients.
    #[must_use]
    pub fn with_coefficients(coefficients: WrLinkDelayCoefficients) -> Self {
        Self {
            calibration: WrCalibration::new(coefficients),
            ..Self::new()
        }
    }

    /// Sets PI controller gains.
    ///
    /// `kp_permille`: proportional gain × 1000.
    /// `ki_permille`: integral gain × 1000.
    pub fn set_gains(&mut self, kp_permille: i64, ki_permille: i64) {
        self.kp_permille = kp_permille;
        self.ki_permille = ki_permille;
    }

    /// Transitions the port to a new state.
    ///
    /// Returns an error if the transition is not allowed from the current state.
    pub fn transition(&mut self, new_state: WrPortState) -> TimeSyncResult<()> {
        let allowed = match (self.state, new_state) {
            (WrPortState::Idle, WrPortState::Presented) => true,
            (WrPortState::Presented, WrPortState::Calibrated) => true,
            (WrPortState::Presented, WrPortState::Error) => true,
            (WrPortState::Calibrated, WrPortState::Tracking) => true,
            (WrPortState::Calibrated, WrPortState::Error) => true,
            (WrPortState::Tracking, WrPortState::Locked) => true,
            (WrPortState::Tracking, WrPortState::Calibrated) => true,
            (WrPortState::Tracking, WrPortState::Error) => true,
            (WrPortState::Locked, WrPortState::Tracking) => true,
            (WrPortState::Locked, WrPortState::Error) => true,
            (WrPortState::Error, WrPortState::Idle) => true,
            (s, t) if s == t => true,
            _ => false,
        };

        if allowed {
            self.state = new_state;
            Ok(())
        } else {
            Err(TimeSyncError::InvalidConfig(format!(
                "WR state transition {:?} → {:?} is not allowed",
                self.state, new_state
            )))
        }
    }

    /// Updates the calibration with a new round-trip measurement.
    ///
    /// Returns the updated master-to-slave delay in picoseconds.
    pub fn update_calibration(&mut self, rtt_ps: u64) -> i64 {
        self.calibration.update_rtt(rtt_ps)
    }

    /// Processes a new phase measurement and sync timestamp pair.
    ///
    /// Updates the internal offset estimate and PI phase loop.
    ///
    /// Returns the new offset in picoseconds.
    pub fn update_phase(
        &mut self,
        sync_origin_ps: i64,
        local_recv_ps: i64,
        ddmtd: DdmtdSample,
    ) -> i64 {
        self.latest_ddmtd = Some(ddmtd);
        let delay_ms = self.calibration.delay_ms_ps;
        let offset =
            compute_wr_offset_ps(sync_origin_ps, local_recv_ps, delay_ms, ddmtd.phase_ps());
        self.offset_ps = offset;

        // PI phase correction (no actual VCXO control here — we return the
        // adjustment that should be applied to the local oscillator).
        let prop = (offset * self.kp_permille) / 1000;
        self.phase_integrator_ps = self
            .phase_integrator_ps
            .saturating_add((offset * self.ki_permille) / 1_000_000);
        let _adjustment = prop + self.phase_integrator_ps;

        offset
    }

    /// Returns `true` when sub-nanosecond lock is achieved.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.state.is_locked()
    }

    /// Returns the latest offset estimate in picoseconds.
    #[must_use]
    pub fn offset_ps(&self) -> i64 {
        self.offset_ps
    }
}

impl Default for WrClock {
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
    fn test_link_delay_symmetric() {
        let coeff = WrLinkDelayCoefficients::symmetric();
        // No fixed delays, no asymmetry: delay_ms = rtt * 0 = 0
        assert_eq!(coeff.delay_master_to_slave_ps(1_000_000), 0);
    }

    #[test]
    fn test_link_delay_with_fixed_delays() {
        // delta_tx_m = 100 ps, delta_rx_s = 200 ps, symmetric (α=0)
        // delay_ms = 100 + 200 + 0 * rtt = 300 ps (regardless of rtt when α=0)
        let coeff = WrLinkDelayCoefficients::new(100, 200, 0.0);
        assert_eq!(coeff.delay_master_to_slave_ps(5_000_000), 300);
    }

    #[test]
    fn test_link_delay_asymmetric() {
        // delta_tx_m = 0, delta_rx_s = 0, α = 0.1, rtt = 1_000_000 ps
        // delay_ms = 0 + 0 + (0.1 / 1.1) * 1_000_000 ≈ 90_909 ps
        let coeff = WrLinkDelayCoefficients::new(0, 0, 0.1);
        let delay = coeff.delay_master_to_slave_ps(1_000_000);
        // Allow ±1 ps for floating-point rounding
        assert!((delay - 90_909).abs() <= 1, "delay={delay}");
    }

    #[test]
    fn test_ddmtd_sample_phase_ps() {
        let sample = DdmtdSample::new(100, 122);
        assert_eq!(sample.phase_ps(), 12_200);
    }

    #[test]
    fn test_wr_port_state_is_locked() {
        assert!(WrPortState::Locked.is_locked());
        assert!(!WrPortState::Tracking.is_locked());
        assert!(!WrPortState::Idle.is_locked());
    }

    #[test]
    fn test_wr_port_state_is_active() {
        assert!(WrPortState::Tracking.is_active());
        assert!(WrPortState::Locked.is_active());
        assert!(!WrPortState::Idle.is_active());
        assert!(!WrPortState::Error.is_active());
    }

    #[test]
    fn test_compute_wr_offset_zero() {
        // Symmetric: recv = origin + delay, phase = 0
        let offset = compute_wr_offset_ps(1_000_000, 1_500_000, 500_000, 0);
        assert_eq!(offset, 0);
    }

    #[test]
    fn test_compute_wr_offset_nonzero() {
        // origin=1_000_000, recv=1_600_000, delay=500_000, phase=0
        // offset = 1_600_000 - 1_000_000 - 500_000 + 0 = 100_000 ps
        let offset = compute_wr_offset_ps(1_000_000, 1_600_000, 500_000, 0);
        assert_eq!(offset, 100_000);
    }

    #[test]
    fn test_wr_clock_creation() {
        let clock = WrClock::new();
        assert_eq!(clock.state, WrPortState::Idle);
        assert!(!clock.is_locked());
        assert_eq!(clock.offset_ps(), 0);
    }

    #[test]
    fn test_wr_clock_transition_valid() {
        let mut clock = WrClock::new();
        assert!(clock.transition(WrPortState::Presented).is_ok());
        assert_eq!(clock.state, WrPortState::Presented);
        assert!(clock.transition(WrPortState::Calibrated).is_ok());
        assert!(clock.transition(WrPortState::Tracking).is_ok());
        assert!(clock.transition(WrPortState::Locked).is_ok());
        assert!(clock.is_locked());
    }

    #[test]
    fn test_wr_clock_transition_invalid() {
        let mut clock = WrClock::new();
        // Cannot jump directly from Idle to Locked
        assert!(clock.transition(WrPortState::Locked).is_err());
        assert_eq!(
            clock.state,
            WrPortState::Idle,
            "state must not change on error"
        );
    }

    #[test]
    fn test_wr_clock_update_calibration() {
        let mut clock = WrClock::new();
        let delay = clock.update_calibration(1_000_000);
        // Symmetric link: delay_ms = 0
        assert_eq!(delay, 0);
        assert!(clock.calibration.calibrated);
    }

    #[test]
    fn test_wr_clock_update_phase() {
        let mut clock = WrClock::new();
        clock.update_calibration(1_000_000); // rtt=1ms, symmetric → delay_ms=0
        let ddmtd = DdmtdSample::new(0, 122); // zero phase correction
                                              // origin = 1_000_000 ps, recv = 1_000_500 ps, delay = 0, phase = 0
                                              // offset = 1_000_500 - 1_000_000 - 0 + 0 = 500 ps
        let offset = clock.update_phase(1_000_000, 1_000_500, ddmtd);
        assert_eq!(offset, 500);
    }

    #[test]
    fn test_wr_calibration_update_rtt() {
        let coeff = WrLinkDelayCoefficients::new(50, 50, 0.0);
        let mut cal = WrCalibration::new(coeff);
        assert!(!cal.calibrated);
        let delay = cal.update_rtt(1_000_000);
        assert_eq!(delay, 100); // 50 + 50 + 0 = 100 ps
        assert!(cal.calibrated);
    }
}
