//! RAVENNA Audio over IP clock synchronisation profile.
//!
//! RAVENNA is a professional AoIP (Audio over IP) technology standard used in
//! broadcast and professional audio.  It uses PTP (IEEE 1588) for network-wide
//! clock synchronisation and RTP for media transport.
//!
//! # Key characteristics vs AES67
//! - RAVENNA mandates IEEE 1588-2008 (PTPv2) with domain 0 by default.
//! - Source-specific multicast (SSM) for clock distribution.
//! - Tightly specified jitter: ≤ 500 ns end-to-end.
//! - Supports both 48 kHz and 96 kHz sample rates.
//! - Packet times: 125 µs (preferred), 250 µs, 1 ms.
//!
//! # References
//! - RAVENNA Technology Document, v3.0 (ALC NetworX GmbH)
//! - AES67-2015 interoperability profile
//! - IEEE 1588-2008 / IEEE 802.1AS

use crate::error::{TimeSyncError, TimeSyncResult};
use std::fmt;

// ---------------------------------------------------------------------------
// RAVENNA stream descriptor
// ---------------------------------------------------------------------------

/// Packet time options supported by RAVENNA.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RavennaPtime {
    /// 125 microseconds (preferred for low-latency broadcast).
    Us125,
    /// 250 microseconds.
    Us250,
    /// 333 microseconds (1/3 ms, non-standard but seen in practice).
    Us333,
    /// 1000 microseconds (1 ms).
    Us1000,
}

impl RavennaPtime {
    /// Returns the packet time in microseconds.
    #[must_use]
    pub fn microseconds(&self) -> u32 {
        match self {
            Self::Us125 => 125,
            Self::Us250 => 250,
            Self::Us333 => 333,
            Self::Us1000 => 1000,
        }
    }

    /// Returns the SDP `a=ptime:` value as a string (milliseconds, possibly fractional).
    #[must_use]
    pub fn sdp_string(&self) -> &'static str {
        match self {
            Self::Us125 => "0.125",
            Self::Us250 => "0.25",
            Self::Us333 => "0.333",
            Self::Us1000 => "1",
        }
    }

    /// Computes the number of samples per packet at `sample_rate` Hz.
    #[must_use]
    pub fn samples_per_packet(&self, sample_rate: u32) -> u32 {
        (sample_rate as u64 * self.microseconds() as u64 / 1_000_000) as u32
    }
}

impl fmt::Display for RavennaPtime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} µs", self.microseconds())
    }
}

/// RAVENNA stream configuration.
#[derive(Debug, Clone)]
pub struct RavennaStreamConfig {
    /// Sample rate (48_000 or 96_000 Hz).
    pub sample_rate: u32,
    /// Channel count.
    pub channels: u8,
    /// Bit depth (16, 24, or 32).
    pub bit_depth: u8,
    /// Packet time.
    pub ptime: RavennaPtime,
    /// PTP domain number (0 by default per RAVENNA spec).
    pub ptp_domain: u8,
}

impl RavennaStreamConfig {
    /// Creates the default RAVENNA broadcast configuration:
    /// 48 kHz, 2 channels, 24-bit, 1 ms packets, domain 0.
    #[must_use]
    pub fn default_broadcast() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            ptime: RavennaPtime::Us1000,
            ptp_domain: 0,
        }
    }

    /// Creates a low-latency RAVENNA configuration:
    /// 48 kHz, 2 channels, 24-bit, 125 µs packets, domain 0.
    #[must_use]
    pub fn low_latency() -> Self {
        Self {
            sample_rate: 48_000,
            channels: 2,
            bit_depth: 24,
            ptime: RavennaPtime::Us125,
            ptp_domain: 0,
        }
    }

    /// Validates the configuration against the RAVENNA specification.
    ///
    /// Returns `Ok(())` if the configuration is compliant, or an error
    /// describing the first violation.
    pub fn validate(&self) -> TimeSyncResult<()> {
        if self.sample_rate != 48_000 && self.sample_rate != 96_000 {
            return Err(TimeSyncError::InvalidConfig(format!(
                "RAVENNA: sample_rate must be 48000 or 96000, got {}",
                self.sample_rate
            )));
        }
        if self.channels == 0 || self.channels > 64 {
            return Err(TimeSyncError::InvalidConfig(format!(
                "RAVENNA: channels must be 1–64, got {}",
                self.channels
            )));
        }
        if !matches!(self.bit_depth, 16 | 24 | 32) {
            return Err(TimeSyncError::InvalidConfig(format!(
                "RAVENNA: bit_depth must be 16, 24, or 32, got {}",
                self.bit_depth
            )));
        }
        if self.ptp_domain > 127 {
            return Err(TimeSyncError::InvalidConfig(format!(
                "RAVENNA: ptp_domain must be 0–127, got {}",
                self.ptp_domain
            )));
        }
        Ok(())
    }

    /// Returns the number of samples per packet.
    #[must_use]
    pub fn samples_per_packet(&self) -> u32 {
        self.ptime.samples_per_packet(self.sample_rate)
    }

    /// Returns the packet payload size in bytes (audio data only, excluding
    /// RTP header).
    #[must_use]
    pub fn payload_bytes(&self) -> u32 {
        self.samples_per_packet() * u32::from(self.channels) * u32::from(self.bit_depth / 8)
    }
}

impl Default for RavennaStreamConfig {
    fn default() -> Self {
        Self::default_broadcast()
    }
}

// ---------------------------------------------------------------------------
// RAVENNA PTP profile compliance
// ---------------------------------------------------------------------------

/// Parameters of the PTP clock as reported in RAVENNA's RTSP SDP.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RavennaClockParams {
    /// PTP domain number.
    pub domain: u8,
    /// PTP grandmaster clock identity (8 bytes, EUI-64).
    pub gm_identity: [u8; 8],
    /// PTP grandmaster clock variant string as in SDP
    /// (e.g. "IEEE1588-2008", "IEEE802.1AS-2011").
    pub variant: String,
}

impl RavennaClockParams {
    /// Creates clock parameters for IEEE 1588-2008 (the RAVENNA standard).
    #[must_use]
    pub fn ieee1588_2008(domain: u8, gm_identity: [u8; 8]) -> Self {
        Self {
            domain,
            gm_identity,
            variant: "IEEE1588-2008".to_string(),
        }
    }

    /// Verifies that the clock parameters are RAVENNA-compliant.
    ///
    /// RAVENNA mandates IEEE 1588-2008 and a domain in the range 0–127.
    pub fn verify_compliance(&self) -> TimeSyncResult<()> {
        if self.domain > 127 {
            return Err(TimeSyncError::InvalidConfig(format!(
                "RAVENNA: PTP domain {} is out of range 0–127",
                self.domain
            )));
        }
        if self.variant != "IEEE1588-2008" && self.variant != "IEEE802.1AS-2011" {
            return Err(TimeSyncError::InvalidConfig(format!(
                "RAVENNA: PTP variant '{}' is not supported; expected IEEE1588-2008 or IEEE802.1AS-2011",
                self.variant
            )));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RAVENNA clock domain
// ---------------------------------------------------------------------------

/// Tracks a RAVENNA PTP clock domain, monitoring sources and jitter.
pub struct RavennaClockDomain {
    /// Domain number.
    pub domain: u8,
    /// Whether a grandmaster has been detected.
    pub gm_present: bool,
    /// Grandmaster identity (if known).
    pub gm_identity: Option<[u8; 8]>,
    /// Current offset from GM in nanoseconds.
    pub offset_ns: i64,
    /// Estimated end-to-end jitter in nanoseconds.
    pub jitter_ns: f64,
    /// Number of sync messages received.
    pub sync_count: u64,
    /// Maximum observed jitter (nanoseconds) — RAVENNA limit is 500 ns.
    max_jitter_ns: f64,
}

impl RavennaClockDomain {
    /// RAVENNA jitter budget: 500 ns end-to-end.
    pub const JITTER_LIMIT_NS: f64 = 500.0;

    /// Creates a new RAVENNA clock domain tracker for the given domain.
    #[must_use]
    pub fn new(domain: u8) -> Self {
        Self {
            domain,
            gm_present: false,
            gm_identity: None,
            offset_ns: 0,
            jitter_ns: 0.0,
            sync_count: 0,
            max_jitter_ns: 0.0,
        }
    }

    /// Records a sync message with the given offset measurement.
    ///
    /// Updates the jitter estimate using an EWMA (exponential weighted moving
    /// average).
    pub fn record_sync(&mut self, gm_identity: [u8; 8], offset_ns: i64) {
        let prev_offset = self.offset_ns;
        self.offset_ns = offset_ns;
        self.gm_identity = Some(gm_identity);
        self.gm_present = true;
        self.sync_count += 1;

        // Jitter estimate: EWMA of |Δoffset|
        let delta = (offset_ns - prev_offset).unsigned_abs() as f64;
        if self.sync_count == 1 {
            self.jitter_ns = delta;
        } else {
            self.jitter_ns = 0.875 * self.jitter_ns + 0.125 * delta;
        }

        if delta > self.max_jitter_ns {
            self.max_jitter_ns = delta;
        }
    }

    /// Returns `true` if the current jitter is within the RAVENNA budget.
    #[must_use]
    pub fn is_within_jitter_budget(&self) -> bool {
        self.jitter_ns <= Self::JITTER_LIMIT_NS
    }

    /// Returns the maximum observed jitter.
    #[must_use]
    pub fn max_jitter_ns(&self) -> f64 {
        self.max_jitter_ns
    }

    /// Returns `true` if the domain has an active grandmaster.
    #[must_use]
    pub fn has_grandmaster(&self) -> bool {
        self.gm_present && self.gm_identity.is_some()
    }

    /// Marks the grandmaster as absent (e.g. after announce timeout).
    pub fn mark_gm_absent(&mut self) {
        self.gm_present = false;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ravenna_ptime_us125_samples() {
        let ptime = RavennaPtime::Us125;
        // 48000 * 125 / 1_000_000 = 6 samples
        assert_eq!(ptime.samples_per_packet(48_000), 6);
    }

    #[test]
    fn test_ravenna_ptime_us1000_samples() {
        let ptime = RavennaPtime::Us1000;
        // 48000 * 1000 / 1_000_000 = 48 samples
        assert_eq!(ptime.samples_per_packet(48_000), 48);
    }

    #[test]
    fn test_ravenna_ptime_display() {
        assert_eq!(format!("{}", RavennaPtime::Us125), "125 µs");
        assert_eq!(format!("{}", RavennaPtime::Us1000), "1000 µs");
    }

    #[test]
    fn test_stream_config_validate_ok() {
        let cfg = RavennaStreamConfig::default_broadcast();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_stream_config_validate_bad_sample_rate() {
        let cfg = RavennaStreamConfig {
            sample_rate: 44_100,
            ..RavennaStreamConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_stream_config_validate_bad_bit_depth() {
        let cfg = RavennaStreamConfig {
            bit_depth: 20,
            ..RavennaStreamConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_stream_config_payload_bytes() {
        // 48 samples, 2 channels, 3 bytes = 288 bytes
        let cfg = RavennaStreamConfig::default_broadcast();
        assert_eq!(cfg.payload_bytes(), 48 * 2 * 3);
    }

    #[test]
    fn test_ravenna_clock_params_compliance_ok() {
        let params = RavennaClockParams::ieee1588_2008(0, [0u8; 8]);
        assert!(params.verify_compliance().is_ok());
    }

    #[test]
    fn test_ravenna_clock_params_bad_variant() {
        let params = RavennaClockParams {
            domain: 0,
            gm_identity: [0u8; 8],
            variant: "NTPv4".to_string(),
        };
        assert!(params.verify_compliance().is_err());
    }

    #[test]
    fn test_ravenna_clock_domain_new() {
        let domain = RavennaClockDomain::new(0);
        assert!(!domain.has_grandmaster());
        assert_eq!(domain.sync_count, 0);
        assert!(
            domain.is_within_jitter_budget(),
            "0 jitter should be within budget"
        );
    }

    #[test]
    fn test_ravenna_clock_domain_record_sync() {
        let mut domain = RavennaClockDomain::new(0);
        let gm = [1u8; 8];
        domain.record_sync(gm, 100);
        assert!(domain.has_grandmaster());
        assert_eq!(domain.sync_count, 1);
        assert_eq!(domain.offset_ns, 100);
    }

    #[test]
    fn test_ravenna_clock_domain_jitter_within_budget() {
        let mut domain = RavennaClockDomain::new(0);
        let gm = [1u8; 8];
        domain.record_sync(gm, 0);
        domain.record_sync(gm, 100); // jitter < 500 ns
        assert!(domain.is_within_jitter_budget());
    }

    #[test]
    fn test_ravenna_clock_domain_jitter_exceeds_budget() {
        let mut domain = RavennaClockDomain::new(0);
        let gm = [1u8; 8];
        domain.record_sync(gm, 0);
        domain.record_sync(gm, 10_000); // 10 µs jitter > 500 ns
                                        // EWMA: 0.875 * 0 + 0.125 * 10000 = 1250 ns
        assert!(!domain.is_within_jitter_budget());
    }

    #[test]
    fn test_ravenna_clock_domain_mark_gm_absent() {
        let mut domain = RavennaClockDomain::new(0);
        domain.record_sync([1u8; 8], 0);
        assert!(domain.has_grandmaster());
        domain.mark_gm_absent();
        assert!(!domain.gm_present);
    }
}
