//! IEEE 1588-2019 Precision Time Protocol (PTP) implementation.
//!
//! This module implements PTP v2 with support for:
//! - Ordinary Clock (OC) and Boundary Clock (BC)
//! - Best Master Clock Algorithm (BMCA)
//! - Transparent Clock support
//! - Unicast and multicast modes

pub mod bmca;
#[cfg(not(target_arch = "wasm32"))]
pub mod clock;
pub mod crc_validation;
pub mod dataset;
pub mod message;
pub mod port;
pub mod slave;
pub mod transparent;

use crate::error::TimeSyncResult;
use std::net::SocketAddr;

/// PTP domain number (0-127).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Domain(pub u8);

impl Domain {
    /// Default PTP domain
    pub const DEFAULT: Self = Self(0);

    /// Create a new domain
    pub fn new(domain: u8) -> TimeSyncResult<Self> {
        if domain > 127 {
            return Err(crate::error::TimeSyncError::InvalidConfig(
                "Domain must be 0-127".to_string(),
            ));
        }
        Ok(Self(domain))
    }
}

/// PTP clock identity (64-bit EUI-64).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ClockIdentity(pub [u8; 8]);

impl ClockIdentity {
    /// Generate a clock identity from MAC address
    #[must_use]
    pub fn from_mac(mac: [u8; 6]) -> Self {
        let mut id = [0u8; 8];
        id[0] = mac[0];
        id[1] = mac[1];
        id[2] = mac[2];
        id[3] = 0xFF;
        id[4] = 0xFE;
        id[5] = mac[3];
        id[6] = mac[4];
        id[7] = mac[5];
        Self(id)
    }

    /// Generate a random clock identity
    #[must_use]
    pub fn random() -> Self {
        use std::time::SystemTime;
        let nanos = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        let mut id = [0u8; 8];
        id.copy_from_slice(&nanos.to_be_bytes());
        Self(id)
    }
}

/// PTP port identity (clock identity + port number).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PortIdentity {
    /// Clock identity
    pub clock_identity: ClockIdentity,
    /// Port number (1-based)
    pub port_number: u16,
}

impl PortIdentity {
    /// Create a new port identity
    #[must_use]
    pub fn new(clock_identity: ClockIdentity, port_number: u16) -> Self {
        Self {
            clock_identity,
            port_number,
        }
    }
}

/// PTP timestamp representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtpTimestamp {
    /// Seconds since epoch (January 1, 1970 00:00:00 UTC)
    pub seconds: u64,
    /// Nanoseconds
    pub nanoseconds: u32,
}

impl PtpTimestamp {
    /// Create a new PTP timestamp
    pub fn new(seconds: u64, nanoseconds: u32) -> TimeSyncResult<Self> {
        if nanoseconds >= 1_000_000_000 {
            return Err(crate::error::TimeSyncError::InvalidTimestamp);
        }
        Ok(Self {
            seconds,
            nanoseconds,
        })
    }

    /// Get current system time as PTP timestamp
    #[must_use]
    pub fn now() -> Self {
        use std::time::SystemTime;
        let duration = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default();
        Self {
            seconds: duration.as_secs(),
            nanoseconds: duration.subsec_nanos(),
        }
    }

    /// Convert to nanoseconds since epoch
    #[must_use]
    pub fn to_nanos(&self) -> u128 {
        u128::from(self.seconds) * 1_000_000_000 + u128::from(self.nanoseconds)
    }

    /// Convert from nanoseconds since epoch
    pub fn from_nanos(nanos: u128) -> TimeSyncResult<Self> {
        let seconds = (nanos / 1_000_000_000) as u64;
        let nanoseconds = (nanos % 1_000_000_000) as u32;
        Self::new(seconds, nanoseconds)
    }

    /// Calculate the difference between two timestamps in nanoseconds
    #[must_use]
    pub fn diff(&self, other: &Self) -> i64 {
        let self_nanos = self.to_nanos() as i128;
        let other_nanos = other.to_nanos() as i128;
        (self_nanos - other_nanos) as i64
    }

    /// Add nanoseconds to this timestamp
    pub fn add_nanos(&self, nanos: i64) -> TimeSyncResult<Self> {
        let current_nanos = self.to_nanos() as i128;
        let new_nanos = current_nanos + i128::from(nanos);
        if new_nanos < 0 {
            return Err(crate::error::TimeSyncError::Overflow);
        }
        Self::from_nanos(new_nanos as u128)
    }
}

/// PTP communication mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CommunicationMode {
    /// Multicast (default)
    Multicast,
    /// Unicast with specific master
    Unicast(SocketAddr),
}

/// PTP delay mechanism.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DelayMechanism {
    /// End-to-end delay (E2E)
    E2E,
    /// Peer-to-peer delay (P2P)
    P2P,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_creation() {
        assert!(Domain::new(0).is_ok());
        assert!(Domain::new(127).is_ok());
        assert!(Domain::new(128).is_err());
    }

    #[test]
    fn test_clock_identity_from_mac() {
        let mac = [0x00, 0x11, 0x22, 0x33, 0x44, 0x55];
        let id = ClockIdentity::from_mac(mac);
        assert_eq!(id.0[0], 0x00);
        assert_eq!(id.0[1], 0x11);
        assert_eq!(id.0[2], 0x22);
        assert_eq!(id.0[3], 0xFF);
        assert_eq!(id.0[4], 0xFE);
        assert_eq!(id.0[5], 0x33);
        assert_eq!(id.0[6], 0x44);
        assert_eq!(id.0[7], 0x55);
    }

    #[test]
    fn test_ptp_timestamp() {
        let ts = PtpTimestamp::new(1000, 500_000_000).expect("should succeed in test");
        assert_eq!(ts.seconds, 1000);
        assert_eq!(ts.nanoseconds, 500_000_000);
        assert_eq!(ts.to_nanos(), 1_000_500_000_000);

        assert!(PtpTimestamp::new(1000, 1_000_000_000).is_err());
    }

    #[test]
    fn test_timestamp_diff() {
        let ts1 = PtpTimestamp::new(1000, 500_000_000).expect("should succeed in test");
        let ts2 = PtpTimestamp::new(1000, 600_000_000).expect("should succeed in test");
        assert_eq!(ts2.diff(&ts1), 100_000_000);
        assert_eq!(ts1.diff(&ts2), -100_000_000);
    }

    #[test]
    fn test_timestamp_add_nanos() {
        let ts = PtpTimestamp::new(1000, 500_000_000).expect("should succeed in test");
        let ts2 = ts.add_nanos(100_000_000).expect("should succeed in test");
        assert_eq!(ts2.seconds, 1000);
        assert_eq!(ts2.nanoseconds, 600_000_000);

        let ts3 = ts.add_nanos(600_000_000).expect("should succeed in test");
        assert_eq!(ts3.seconds, 1001);
        assert_eq!(ts3.nanoseconds, 100_000_000);
    }
}
