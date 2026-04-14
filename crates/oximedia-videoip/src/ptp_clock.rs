//! IEEE 1588 PTP Clock Model — Announce, Sync, and Follow-Up Messages.
//!
//! This module provides the wire-level message structures for the PTP
//! (Precision Time Protocol) ordinary clock as defined in IEEE 1588-2019,
//! plus a simple clock servo and offset estimator.
//!
//! It complements [`crate::ptp`] (which focuses on the BMCA engine and
//! data-set comparison) by providing:
//!
//! - Full wire-format message structs: [`AnnounceMessage`], [`SyncMessage`],
//!   [`FollowUpMessage`], [`DelayReqMessage`], [`DelayRespMessage`].
//! - [`MessageHeader`] — the common 34-byte PTP header.
//! - [`ClockServo`] — a simple PI servo that converts raw offset samples into
//!   a local-clock frequency correction (ppb).
//! - [`OffsetEstimator`] — exponential moving average for smoothing noisy
//!   offset measurements.
//!
//! # Message Flow
//!
//! ```text
//! Master                  Slave
//!   │                       │
//!   │─── Sync (T1) ────────>│  T2 = slave rx time
//!   │─── Follow_Up (T1) ───>│  (two-step clock only)
//!   │                       │
//!   │<── Delay_Req (T3) ────│
//!   │─── Delay_Resp (T4) ──>│
//!   │                       │
//!   │          offset = (T2 - T1) - path_delay
//!   │          path_delay = ((T2 - T1) + (T4 - T3)) / 2
//! ```

use std::time::Duration;

// ─── PTP Timestamp ────────────────────────────────────────────────────────────

/// IEEE 1588 Timestamp: 48-bit TAI seconds + 32-bit nanoseconds.
///
/// Wire format: 6 bytes seconds (big-endian) + 4 bytes nanoseconds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PtpTimestamp {
    /// TAI seconds since the PTP epoch (1970-01-01 00:00:00 TAI).
    pub seconds: u64,
    /// Sub-second nanoseconds (0 … 999_999_999).
    pub nanoseconds: u32,
}

impl PtpTimestamp {
    /// Create a new timestamp.
    #[must_use]
    pub const fn new(seconds: u64, nanoseconds: u32) -> Self {
        Self { seconds, nanoseconds }
    }

    /// Total nanoseconds from the epoch (saturating).
    #[must_use]
    pub fn to_nanos(self) -> u64 {
        self.seconds
            .saturating_mul(1_000_000_000)
            .saturating_add(u64::from(self.nanoseconds))
    }

    /// Signed difference `self − other` in nanoseconds.
    #[must_use]
    pub fn diff_nanos(self, other: Self) -> i64 {
        let a = self.to_nanos();
        let b = other.to_nanos();
        (a as i64).wrapping_sub(b as i64)
    }

    /// Add a nanosecond offset to this timestamp (saturating).
    #[must_use]
    pub fn add_nanos(self, nanos: i64) -> Self {
        let total = (self.to_nanos() as i64).saturating_add(nanos);
        if total < 0 {
            return Self::default();
        }
        let total_u = total as u64;
        Self {
            seconds: total_u / 1_000_000_000,
            nanoseconds: (total_u % 1_000_000_000) as u32,
        }
    }
}

// ─── Port Identity ────────────────────────────────────────────────────────────

/// PTP Port Identity: clock identity (EUI-64) + port number.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PortIdentity {
    /// EUI-64 clock identity.
    pub clock_identity: [u8; 8],
    /// Port number (1-indexed).
    pub port_number: u16,
}

impl PortIdentity {
    /// Create a PortIdentity from a 64-bit integer and port number.
    #[must_use]
    pub fn from_u64(id: u64, port: u16) -> Self {
        Self {
            clock_identity: id.to_be_bytes(),
            port_number: port,
        }
    }
}

// ─── PTP Message Type ─────────────────────────────────────────────────────────

/// PTP message type nibble (bits 3–0 of the first byte of the header).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MessageType {
    /// Sync message (event).
    Sync = 0x0,
    /// Delay_Req message (event).
    DelayReq = 0x1,
    /// Follow_Up message (general).
    FollowUp = 0x8,
    /// Delay_Resp message (general).
    DelayResp = 0x9,
    /// Announce message (general).
    Announce = 0xB,
    /// Signaling message (general).
    Signaling = 0xC,
    /// Management message (general).
    Management = 0xD,
}

impl MessageType {
    /// Parse from a raw nibble value.
    #[must_use]
    pub fn from_nibble(v: u8) -> Option<Self> {
        match v & 0x0F {
            0x0 => Some(Self::Sync),
            0x1 => Some(Self::DelayReq),
            0x8 => Some(Self::FollowUp),
            0x9 => Some(Self::DelayResp),
            0xB => Some(Self::Announce),
            0xC => Some(Self::Signaling),
            0xD => Some(Self::Management),
            _ => None,
        }
    }

    /// Returns `true` for event messages that require hardware timestamping.
    #[must_use]
    pub fn is_event(self) -> bool {
        matches!(self, Self::Sync | Self::DelayReq)
    }
}

impl std::fmt::Display for MessageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            Self::Sync => "Sync",
            Self::DelayReq => "Delay_Req",
            Self::FollowUp => "Follow_Up",
            Self::DelayResp => "Delay_Resp",
            Self::Announce => "Announce",
            Self::Signaling => "Signaling",
            Self::Management => "Management",
        };
        write!(f, "{s}")
    }
}

// ─── Message Header ───────────────────────────────────────────────────────────

/// Common 34-byte PTP message header (IEEE 1588-2019 §13.3.1).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MessageHeader {
    /// Transport-specific nibble (upper 4 bits) | message type (lower 4 bits).
    pub transport_and_type: u8,
    /// PTP version (currently 2).
    pub ptp_version: u8,
    /// Total message length in bytes including the header.
    pub message_length: u16,
    /// PTP domain number (0–127).
    pub domain_number: u8,
    /// Minor SdoId (sub-domain id, usually 0).
    pub minor_sdo_id: u8,
    /// Flags (bit field; see IEEE 1588-2019 §13.3.2.8).
    pub flags: u16,
    /// Correction field (scaled nanoseconds, fixed-point 64-bit).
    pub correction_ns: i64,
    /// Message type-specific field (4 bytes).
    pub type_specific: u32,
    /// Source port identity.
    pub source_port_identity: PortIdentity,
    /// Sequence ID (incremented for each message of the same type).
    pub sequence_id: u16,
    /// Control field (legacy: 0=Sync, 1=Delay_Req, 2=Follow_Up, 3=Delay_Resp, 5=Announce, 4=other).
    pub control: u8,
    /// Log message interval (log₂ of the message period in seconds).
    pub log_message_interval: i8,
}

impl MessageHeader {
    /// Flag bit: two-step clock (Sync carries only origin-ts; Follow_Up follows).
    pub const FLAG_TWO_STEP: u16 = 1 << 9;
    /// Flag bit: UTC offset valid.
    pub const FLAG_UTC_REASONABLE: u16 = 1 << 2;
    /// Flag bit: PTP timescale (TAI).
    pub const FLAG_PTP_TIMESCALE: u16 = 1 << 3;

    /// Construct a new header for the given message type.
    #[must_use]
    pub fn new(
        msg_type: MessageType,
        source: PortIdentity,
        domain: u8,
        sequence_id: u16,
        message_length: u16,
    ) -> Self {
        let control = match msg_type {
            MessageType::Sync => 0,
            MessageType::DelayReq => 1,
            MessageType::FollowUp => 2,
            MessageType::DelayResp => 3,
            MessageType::Announce => 5,
            _ => 4,
        };
        Self {
            transport_and_type: msg_type as u8,
            ptp_version: 2,
            message_length,
            domain_number: domain,
            minor_sdo_id: 0,
            flags: 0,
            correction_ns: 0,
            type_specific: 0,
            source_port_identity: source,
            sequence_id,
            control,
            log_message_interval: 0,
        }
    }

    /// Extract the message type from the header.
    #[must_use]
    pub fn message_type(&self) -> Option<MessageType> {
        MessageType::from_nibble(self.transport_and_type)
    }

    /// Check the two-step flag.
    #[must_use]
    pub fn is_two_step(&self) -> bool {
        self.flags & Self::FLAG_TWO_STEP != 0
    }
}

// ─── Announce Message ─────────────────────────────────────────────────────────

/// PTP Announce message (IEEE 1588-2019 §13.5).
///
/// Carries grandmaster clock quality information used by the BMCA.
#[derive(Debug, Clone)]
pub struct AnnounceMessage {
    /// Common header.
    pub header: MessageHeader,
    /// Time of origin (master's current TAI time).
    pub origin_timestamp: PtpTimestamp,
    /// Current UTC offset in seconds (TAI − UTC).
    pub current_utc_offset: i16,
    /// Reserved byte.
    pub reserved: u8,
    /// Grandmaster priority 1.
    pub grandmaster_priority1: u8,
    /// Grandmaster clock class.
    pub grandmaster_clock_class: u8,
    /// Grandmaster clock accuracy.
    pub grandmaster_clock_accuracy: u8,
    /// Grandmaster clock variance (offset scaled log variance).
    pub grandmaster_clock_variance: u16,
    /// Grandmaster priority 2.
    pub grandmaster_priority2: u8,
    /// Grandmaster clock identity (EUI-64).
    pub grandmaster_identity: [u8; 8],
    /// Steps removed (number of boundary clocks between this port and the GM).
    pub steps_removed: u16,
    /// Time source (GNSS=0x20, NTP=0x30, etc.).
    pub time_source: u8,
}

impl AnnounceMessage {
    /// Construct a new Announce message.
    ///
    /// The `announce_interval_log` should be 0 (1 s) for ordinary clocks.
    #[must_use]
    pub fn new(
        source: PortIdentity,
        domain: u8,
        sequence_id: u16,
        origin_timestamp: PtpTimestamp,
        grandmaster_identity: [u8; 8],
        priority1: u8,
        priority2: u8,
        clock_class: u8,
        steps_removed: u16,
        announce_interval_log: i8,
    ) -> Self {
        let mut hdr = MessageHeader::new(
            MessageType::Announce,
            source,
            domain,
            sequence_id,
            // Announce wire length = 34 (header) + 30 (body) = 64 bytes
            64,
        );
        hdr.log_message_interval = announce_interval_log;
        Self {
            header: hdr,
            origin_timestamp,
            current_utc_offset: 37, // TAI − UTC as of 2017
            reserved: 0,
            grandmaster_priority1: priority1,
            grandmaster_clock_class: clock_class,
            grandmaster_clock_accuracy: 0x21, // < 100 ns
            grandmaster_clock_variance: 0x4E5D,
            grandmaster_priority2: priority2,
            grandmaster_identity,
            steps_removed,
            time_source: 0x20, // GNSS
        }
    }

    /// Validate basic announce message invariants.
    pub fn validate(&self) -> Result<(), String> {
        if self.header.ptp_version != 2 {
            return Err(format!(
                "unsupported PTP version {}",
                self.header.ptp_version
            ));
        }
        if self.steps_removed > 255 {
            return Err(format!(
                "steps_removed {} exceeds maximum 255",
                self.steps_removed
            ));
        }
        Ok(())
    }
}

// ─── Sync Message ─────────────────────────────────────────────────────────────

/// PTP Sync message (IEEE 1588-2019 §13.6).
///
/// In a **two-step** clock the `origin_timestamp` is zero; the precise
/// transmit timestamp is sent in a subsequent [`FollowUpMessage`].
#[derive(Debug, Clone)]
pub struct SyncMessage {
    /// Common header.
    pub header: MessageHeader,
    /// Origin timestamp (precise only for one-step clocks).
    pub origin_timestamp: PtpTimestamp,
}

impl SyncMessage {
    /// Construct a one-step Sync message carrying the origin timestamp.
    #[must_use]
    pub fn one_step(
        source: PortIdentity,
        domain: u8,
        sequence_id: u16,
        origin_timestamp: PtpTimestamp,
        sync_interval_log: i8,
    ) -> Self {
        let mut hdr = MessageHeader::new(
            MessageType::Sync,
            source,
            domain,
            sequence_id,
            // Wire length = 34 (header) + 10 (timestamp) = 44 bytes
            44,
        );
        hdr.log_message_interval = sync_interval_log;
        Self { header: hdr, origin_timestamp }
    }

    /// Construct a two-step Sync message (origin_timestamp = 0, follow-up to follow).
    #[must_use]
    pub fn two_step(
        source: PortIdentity,
        domain: u8,
        sequence_id: u16,
        sync_interval_log: i8,
    ) -> Self {
        let mut hdr = MessageHeader::new(
            MessageType::Sync,
            source,
            domain,
            sequence_id,
            44,
        );
        hdr.log_message_interval = sync_interval_log;
        hdr.flags |= MessageHeader::FLAG_TWO_STEP;
        Self {
            header: hdr,
            origin_timestamp: PtpTimestamp::default(),
        }
    }
}

// ─── Follow-Up Message ────────────────────────────────────────────────────────

/// PTP Follow_Up message (IEEE 1588-2019 §13.7).
///
/// Carries the precise transmit timestamp for the preceding two-step Sync.
/// The `sequence_id` must match the Sync it follows.
#[derive(Debug, Clone)]
pub struct FollowUpMessage {
    /// Common header.
    pub header: MessageHeader,
    /// Precise origin timestamp (T1 in the two-step model).
    pub precise_origin_timestamp: PtpTimestamp,
}

impl FollowUpMessage {
    /// Construct a Follow_Up message for the Sync with `sync_sequence_id`.
    #[must_use]
    pub fn new(
        source: PortIdentity,
        domain: u8,
        sync_sequence_id: u16,
        precise_origin_timestamp: PtpTimestamp,
    ) -> Self {
        let hdr = MessageHeader::new(
            MessageType::FollowUp,
            source,
            domain,
            sync_sequence_id,
            // Wire length = 34 + 10 = 44 bytes
            44,
        );
        Self { header: hdr, precise_origin_timestamp }
    }
}

// ─── Delay-Req / Delay-Resp Messages ─────────────────────────────────────────

/// PTP Delay_Req message (IEEE 1588-2019 §13.9).
#[derive(Debug, Clone)]
pub struct DelayReqMessage {
    /// Common header.
    pub header: MessageHeader,
    /// Origin timestamp (T3 — slave's transmit time; set after sending).
    pub origin_timestamp: PtpTimestamp,
}

impl DelayReqMessage {
    /// Construct a Delay_Req with the given origin timestamp (T3).
    #[must_use]
    pub fn new(
        source: PortIdentity,
        domain: u8,
        sequence_id: u16,
        origin_timestamp: PtpTimestamp,
    ) -> Self {
        let hdr = MessageHeader::new(
            MessageType::DelayReq,
            source,
            domain,
            sequence_id,
            44,
        );
        Self { header: hdr, origin_timestamp }
    }
}

/// PTP Delay_Resp message (IEEE 1588-2019 §13.10).
#[derive(Debug, Clone)]
pub struct DelayRespMessage {
    /// Common header.
    pub header: MessageHeader,
    /// Receive timestamp (T4 — master's rx time of the Delay_Req).
    pub receive_timestamp: PtpTimestamp,
    /// Port identity of the requesting slave.
    pub requesting_port_identity: PortIdentity,
}

impl DelayRespMessage {
    /// Construct a Delay_Resp message.
    #[must_use]
    pub fn new(
        source: PortIdentity,
        domain: u8,
        sequence_id: u16,
        receive_timestamp: PtpTimestamp,
        requesting_port_identity: PortIdentity,
    ) -> Self {
        let hdr = MessageHeader::new(
            MessageType::DelayResp,
            source,
            domain,
            sequence_id,
            // Wire length = 34 + 10 + 10 = 54 bytes
            54,
        );
        Self {
            header: hdr,
            receive_timestamp,
            requesting_port_identity,
        }
    }
}

// ─── Offset Estimator ─────────────────────────────────────────────────────────

/// Exponential moving average (EMA) offset estimator.
///
/// Smooths noisy `offset_from_master` samples to reduce jitter in the
/// servo's correction signal.
#[derive(Debug, Clone)]
pub struct OffsetEstimator {
    /// Smoothing factor α ∈ (0, 1].  Larger = more responsive, noisier.
    alpha: f64,
    /// Current EMA estimate (nanoseconds).
    estimate_ns: f64,
    /// Number of samples processed.
    sample_count: u64,
    /// Whether the estimator has been initialised (first sample sets directly).
    initialized: bool,
}

impl OffsetEstimator {
    /// Create a new estimator with smoothing factor `alpha`.
    ///
    /// Typical values: 0.1 for slow networks, 0.3 for LAN.
    ///
    /// Returns an error if `alpha` is outside `(0, 1]`.
    pub fn new(alpha: f64) -> Result<Self, String> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(format!("alpha {alpha} must be in (0, 1]"));
        }
        Ok(Self {
            alpha,
            estimate_ns: 0.0,
            sample_count: 0,
            initialized: false,
        })
    }

    /// Feed a new offset sample (nanoseconds).
    pub fn update(&mut self, offset_ns: i64) {
        let x = offset_ns as f64;
        if !self.initialized {
            self.estimate_ns = x;
            self.initialized = true;
        } else {
            self.estimate_ns = self.alpha * x + (1.0 - self.alpha) * self.estimate_ns;
        }
        self.sample_count += 1;
    }

    /// Current smoothed offset estimate (nanoseconds, rounded).
    #[must_use]
    pub fn estimate_ns(&self) -> i64 {
        self.estimate_ns.round() as i64
    }

    /// Number of samples processed.
    #[must_use]
    pub fn sample_count(&self) -> u64 {
        self.sample_count
    }

    /// Reset the estimator.
    pub fn reset(&mut self) {
        self.estimate_ns = 0.0;
        self.sample_count = 0;
        self.initialized = false;
    }
}

// ─── PI Clock Servo ───────────────────────────────────────────────────────────

/// Proportional-Integral (PI) clock servo.
///
/// Converts smoothed offset measurements into a **frequency correction** in
/// parts-per-billion (ppb) to be applied to a local hardware clock.
///
/// The servo uses the standard PI formulation:
///
/// ```text
/// correction_ppb = Kp × offset_ns + Ki × integral_ns
/// ```
///
/// with an integral windup clamp to prevent unbounded growth.
#[derive(Debug, Clone)]
pub struct ClockServo {
    /// Proportional gain (ppb / ns).
    pub kp: f64,
    /// Integral gain (ppb / ns·sample).
    pub ki: f64,
    /// Integral accumulator (nanoseconds).
    integral_ns: f64,
    /// Anti-windup clamp (maximum absolute integral value).
    pub integral_clamp_ns: f64,
    /// Last applied correction (ppb).
    pub last_correction_ppb: f64,
    /// Lock counter: incremented when |offset| < lock_threshold.
    lock_count: u32,
    /// Threshold for declaring lock (nanoseconds).
    pub lock_threshold_ns: i64,
    /// Number of consecutive in-threshold samples required to declare lock.
    pub lock_count_threshold: u32,
    /// Whether the servo is currently locked.
    locked: bool,
}

impl ClockServo {
    /// Construct a new PI servo with the given gains.
    ///
    /// Typical values for a LAN:
    /// - `kp = 0.7` (ppb/ns)
    /// - `ki = 0.3` (ppb/ns)
    /// - `integral_clamp_ns = 200_000` (200 µs)
    /// - `lock_threshold_ns = 100` (100 ns)
    #[must_use]
    pub fn new(kp: f64, ki: f64, integral_clamp_ns: f64) -> Self {
        Self {
            kp,
            ki,
            integral_ns: 0.0,
            integral_clamp_ns: integral_clamp_ns.abs(),
            last_correction_ppb: 0.0,
            lock_count: 0,
            lock_threshold_ns: 100,
            lock_count_threshold: 4,
            locked: false,
        }
    }

    /// Process a new offset sample and return the frequency correction (ppb).
    ///
    /// Positive correction means the local clock is running too slow and
    /// should speed up.
    pub fn update(&mut self, offset_ns: i64) -> f64 {
        let e = offset_ns as f64;

        // Integrate with anti-windup.
        self.integral_ns = (self.integral_ns + e)
            .clamp(-self.integral_clamp_ns, self.integral_clamp_ns);

        let correction = self.kp * e + self.ki * self.integral_ns;
        self.last_correction_ppb = correction;

        // Update lock state.
        if offset_ns.abs() <= self.lock_threshold_ns {
            self.lock_count = self.lock_count.saturating_add(1);
            if self.lock_count >= self.lock_count_threshold {
                self.locked = true;
            }
        } else {
            self.lock_count = 0;
            self.locked = false;
        }

        correction
    }

    /// Returns `true` when the servo has been locked for
    /// `lock_count_threshold` consecutive samples.
    #[must_use]
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Reset the servo (clear integral and lock state).
    pub fn reset(&mut self) {
        self.integral_ns = 0.0;
        self.last_correction_ppb = 0.0;
        self.lock_count = 0;
        self.locked = false;
    }
}

// ─── Path Delay Calculator ────────────────────────────────────────────────────

/// E2E path delay computation from the four PTP timestamps.
///
/// ```text
/// path_delay = ((T2 - T1) + (T4 - T3)) / 2
/// offset     = (T2 - T1) - path_delay
/// ```
#[derive(Debug, Clone, Copy)]
pub struct PathDelay {
    /// T1: master sends Sync (or Follow_Up precise origin timestamp).
    pub t1: PtpTimestamp,
    /// T2: slave receives Sync.
    pub t2: PtpTimestamp,
    /// T3: slave sends Delay_Req.
    pub t3: PtpTimestamp,
    /// T4: master receives Delay_Req.
    pub t4: PtpTimestamp,
}

impl PathDelay {
    /// Compute mean path delay in nanoseconds.
    ///
    /// Returns `None` if the result would be negative (network asymmetry
    /// exceeds the algorithm's range).
    #[must_use]
    pub fn mean_delay_nanos(&self) -> Option<u64> {
        let fwd = self.t2.diff_nanos(self.t1); // T2 − T1
        let rev = self.t4.diff_nanos(self.t3); // T4 − T3
        let sum = fwd.checked_add(rev)?;
        if sum < 0 {
            return None;
        }
        Some((sum as u64) / 2)
    }

    /// Compute the offset from master in nanoseconds.
    ///
    /// Returns `None` if the path delay is undefined.
    #[must_use]
    pub fn offset_nanos(&self) -> Option<i64> {
        let delay = self.mean_delay_nanos()? as i64;
        let fwd = self.t2.diff_nanos(self.t1);
        Some(fwd - delay)
    }

    /// Convert the mean delay to a [`Duration`].
    #[must_use]
    pub fn mean_delay_duration(&self) -> Option<Duration> {
        let ns = self.mean_delay_nanos()?;
        Some(Duration::from_nanos(ns))
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ts(s: u64, ns: u32) -> PtpTimestamp {
        PtpTimestamp::new(s, ns)
    }

    fn port(id: u64, port: u16) -> PortIdentity {
        PortIdentity::from_u64(id, port)
    }

    // ── PtpTimestamp ──────────────────────────────────────────────────────────

    #[test]
    fn test_ptp_timestamp_to_nanos() {
        let t = ts(1, 500_000_000);
        assert_eq!(t.to_nanos(), 1_500_000_000);
    }

    #[test]
    fn test_ptp_timestamp_diff_positive() {
        let a = ts(2, 0);
        let b = ts(1, 0);
        assert_eq!(a.diff_nanos(b), 1_000_000_000);
    }

    #[test]
    fn test_ptp_timestamp_add_nanos() {
        let t = ts(1, 0);
        let t2 = t.add_nanos(500);
        assert_eq!(t2.seconds, 1);
        assert_eq!(t2.nanoseconds, 500);
    }

    #[test]
    fn test_ptp_timestamp_add_nanos_wrap() {
        // 999_999_900 + 200 wraps into the next second.
        let t = ts(0, 999_999_900);
        let t2 = t.add_nanos(200);
        assert_eq!(t2.seconds, 1);
        assert_eq!(t2.nanoseconds, 100);
    }

    // ── MessageType ───────────────────────────────────────────────────────────

    #[test]
    fn test_message_type_from_nibble_roundtrip() {
        let types = [
            MessageType::Sync,
            MessageType::DelayReq,
            MessageType::FollowUp,
            MessageType::DelayResp,
            MessageType::Announce,
        ];
        for mt in types {
            let nibble = mt as u8;
            let decoded = MessageType::from_nibble(nibble).expect("should decode");
            assert_eq!(mt, decoded);
        }
    }

    #[test]
    fn test_sync_is_event_message() {
        assert!(MessageType::Sync.is_event());
        assert!(MessageType::DelayReq.is_event());
        assert!(!MessageType::FollowUp.is_event());
        assert!(!MessageType::Announce.is_event());
    }

    // ── MessageHeader ─────────────────────────────────────────────────────────

    #[test]
    fn test_message_header_construction() {
        let src = port(0xAABBCC, 1);
        let hdr = MessageHeader::new(MessageType::Sync, src, 0, 42, 44);
        assert_eq!(hdr.ptp_version, 2);
        assert_eq!(hdr.sequence_id, 42);
        assert_eq!(hdr.message_length, 44);
        assert_eq!(hdr.message_type(), Some(MessageType::Sync));
    }

    #[test]
    fn test_two_step_flag() {
        let src = port(0x01, 1);
        let sync = SyncMessage::two_step(src, 0, 1, 0);
        assert!(sync.header.is_two_step());
    }

    #[test]
    fn test_one_step_not_two_step() {
        let src = port(0x01, 1);
        let sync = SyncMessage::one_step(src, 0, 1, ts(100, 0), 0);
        assert!(!sync.header.is_two_step());
    }

    // ── AnnounceMessage ───────────────────────────────────────────────────────

    #[test]
    fn test_announce_message_validates_ok() {
        let src = port(0x1234_5678_9ABC_DEF0, 1);
        let msg = AnnounceMessage::new(
            src, 0, 1, ts(1_000_000, 0),
            [0u8; 8],
            128, 128, 135, 0, 0,
        );
        assert!(msg.validate().is_ok());
    }

    #[test]
    fn test_announce_message_wrong_version_fails() {
        let src = port(0x01, 1);
        let mut msg = AnnounceMessage::new(src, 0, 1, ts(0, 0), [0u8; 8], 128, 128, 248, 0, 0);
        msg.header.ptp_version = 1;
        assert!(msg.validate().is_err());
    }

    // ── FollowUpMessage ───────────────────────────────────────────────────────

    #[test]
    fn test_follow_up_sequence_matches_sync() {
        let src = port(0x01, 1);
        let seq = 77u16;
        let fu = FollowUpMessage::new(src, 0, seq, ts(1, 500_000));
        assert_eq!(fu.header.sequence_id, seq);
        assert_eq!(fu.precise_origin_timestamp, ts(1, 500_000));
    }

    // ── DelayReqMessage / DelayRespMessage ────────────────────────────────────

    #[test]
    fn test_delay_req_resp_pair() {
        let master = port(0xAA, 1);
        let slave = port(0xBB, 1);

        // T3: slave sends delay req
        let t3 = ts(100, 0);
        let req = DelayReqMessage::new(slave, 0, 5, t3);
        assert_eq!(req.header.sequence_id, 5);

        // T4: master receives it and responds
        let t4 = ts(100, 500);
        let resp = DelayRespMessage::new(master, 0, 5, t4, slave);
        assert_eq!(resp.receive_timestamp, t4);
        assert_eq!(resp.requesting_port_identity, slave);
        assert_eq!(resp.header.sequence_id, 5);
    }

    // ── PathDelay ─────────────────────────────────────────────────────────────

    #[test]
    fn test_path_delay_symmetric() {
        // T1=0, T2=500ns, T3=600ns, T4=1100ns → delay=500ns, offset=0
        let pd = PathDelay {
            t1: ts(0, 0),
            t2: ts(0, 500),
            t3: ts(0, 600),
            t4: ts(0, 1_100),
        };
        assert_eq!(pd.mean_delay_nanos(), Some(500));
        assert_eq!(pd.offset_nanos(), Some(0));
    }

    #[test]
    fn test_path_delay_asymmetric() {
        // T1=0, T2=600ns (slave slightly after), T3=700ns, T4=1100ns
        // fwd=600, rev=400 → delay=(600+400)/2=500, offset=600-500=100ns
        let pd = PathDelay {
            t1: ts(0, 0),
            t2: ts(0, 600),
            t3: ts(0, 700),
            t4: ts(0, 1_100),
        };
        assert_eq!(pd.mean_delay_nanos(), Some(500));
        assert_eq!(pd.offset_nanos(), Some(100));
    }

    #[test]
    fn test_path_delay_duration_conversion() {
        let pd = PathDelay {
            t1: ts(0, 0),
            t2: ts(0, 2_000),
            t3: ts(0, 3_000),
            t4: ts(0, 5_000),
        };
        // fwd=2000, rev=2000 → delay=2000ns
        let dur = pd.mean_delay_duration().expect("should compute duration");
        assert_eq!(dur.as_nanos(), 2000);
    }

    // ── OffsetEstimator ───────────────────────────────────────────────────────

    #[test]
    fn test_offset_estimator_first_sample_exact() {
        let mut est = OffsetEstimator::new(0.5).expect("valid alpha");
        est.update(1000);
        assert_eq!(est.estimate_ns(), 1000);
    }

    #[test]
    fn test_offset_estimator_smoothing() {
        let mut est = OffsetEstimator::new(0.5).expect("valid alpha");
        est.update(0);
        est.update(1000);
        // After one update from 0 → α=0.5: estimate = 0.5×1000 + 0.5×0 = 500
        assert_eq!(est.estimate_ns(), 500);
    }

    #[test]
    fn test_offset_estimator_invalid_alpha() {
        assert!(OffsetEstimator::new(0.0).is_err());
        assert!(OffsetEstimator::new(1.5).is_err());
    }

    #[test]
    fn test_offset_estimator_reset() {
        let mut est = OffsetEstimator::new(0.3).expect("valid");
        est.update(5000);
        est.update(4000);
        est.reset();
        assert_eq!(est.sample_count(), 0);
        est.update(100);
        assert_eq!(est.estimate_ns(), 100);
    }

    // ── ClockServo ────────────────────────────────────────────────────────────

    #[test]
    fn test_servo_zero_offset_zero_correction() {
        let mut servo = ClockServo::new(0.7, 0.3, 200_000.0);
        let correction = servo.update(0);
        assert!((correction).abs() < 1e-6);
    }

    #[test]
    fn test_servo_positive_offset_positive_correction() {
        let mut servo = ClockServo::new(0.7, 0.3, 200_000.0);
        let correction = servo.update(100);
        assert!(
            correction > 0.0,
            "positive offset should yield positive correction"
        );
    }

    #[test]
    fn test_servo_lock_after_sustained_small_offset() {
        let mut servo = ClockServo::new(0.7, 0.3, 200_000.0);
        servo.lock_threshold_ns = 200; // generous for test
        servo.lock_count_threshold = 3;
        assert!(!servo.is_locked());
        for _ in 0..4 {
            servo.update(50); // within threshold
        }
        assert!(servo.is_locked());
    }

    #[test]
    fn test_servo_loses_lock_on_large_offset() {
        let mut servo = ClockServo::new(0.7, 0.3, 200_000.0);
        servo.lock_threshold_ns = 200;
        servo.lock_count_threshold = 3;
        for _ in 0..4 {
            servo.update(50);
        }
        assert!(servo.is_locked());
        servo.update(50_000); // large jump
        assert!(!servo.is_locked());
    }

    #[test]
    fn test_servo_reset_clears_lock() {
        let mut servo = ClockServo::new(0.7, 0.3, 200_000.0);
        servo.lock_threshold_ns = 200;
        servo.lock_count_threshold = 3;
        for _ in 0..4 {
            servo.update(50);
        }
        servo.reset();
        assert!(!servo.is_locked());
    }

    #[test]
    fn test_servo_integral_clamp() {
        let mut servo = ClockServo::new(0.0, 1.0, 100.0); // kp=0, ki=1, clamp=100
        // After many large offsets, integral should be clamped.
        for _ in 0..1000 {
            servo.update(10_000);
        }
        // Correction should be ki × clamp = 1.0 × 100.0 = 100.0
        assert!(
            servo.last_correction_ppb.abs() <= 101.0,
            "integral clamp should limit correction: {}",
            servo.last_correction_ppb
        );
    }
}
