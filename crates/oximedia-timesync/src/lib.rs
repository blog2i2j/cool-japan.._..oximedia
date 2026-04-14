//! `OxiMedia` Time Synchronization - Precision Time Protocol and Clock Discipline
//!
//! `oximedia-timesync` provides comprehensive time synchronization capabilities for
//! professional media production, including:
//!
//! - **PTP (IEEE 1588-2019)**: Precision Time Protocol with sub-microsecond accuracy
//!   - Ordinary Clock (OC) and Boundary Clock (BC) support
//!   - Best Master Clock Algorithm (BMCA)
//!   - Transparent Clock support
//!   - Unicast and multicast modes
//!
//! - **NTP (RFC 5905)**: Network Time Protocol with millisecond accuracy
//!   - NTP v4 client implementation
//!   - Server pool management with automatic failover
//!   - Stratum hierarchy support
//!
//! - **Timecode Synchronization**: Frame-accurate timecode support
//!   - LTC (Linear Timecode) generation and reading
//!   - SMPTE 12M timecode
//!   - MTC (MIDI Time Code)
//!   - Jam sync with holdover
//!
//! - **Clock Discipline**: Advanced clock control algorithms
//!   - PID controller for smooth adjustments
//!   - Drift compensation with prediction
//!   - Holdover mode for maintaining accuracy without reference
//!   - Multi-source clock selection
//!
//! - **Media Synchronization**:
//!   - Genlock reference generation
//!   - Frame-accurate video sync
//!   - Sample-accurate audio sync
//!
//! - **Integration**: Low-latency access and integration
//!   - Unix domain socket IPC
//!   - Shared memory for microsecond-level access
//!   - Integration with oximedia-core Timestamp type
//!   - System clock adjustment (with appropriate privileges)
//!
//! # Examples
//!
//! ## PTP Synchronization
//!
//! ```rust,no_run
//! use oximedia_timesync::{ClockIdentity, Domain};
//! use oximedia_timesync::ptp::clock::OrdinaryClock;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create a PTP ordinary clock
//! let clock_id = ClockIdentity::random();
//! let mut clock = OrdinaryClock::new(clock_id, Domain::DEFAULT);
//!
//! // Bind to network
//! clock.bind("0.0.0.0:319".parse()?).await?;
//!
//! // Get current offset
//! let offset = clock.offset_from_master();
//! println!("Offset from master: {} ns", offset);
//! # Ok(())
//! # }
//! ```
//!
//! ## NTP Synchronization
//!
//! ```rust,no_run
//! use oximedia_timesync::ntp::NtpClient;
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! // Create NTP client with default servers
//! let mut client = NtpClient::new();
//!
//! // Synchronize with NTP servers
//! let result = client.synchronize().await?;
//!
//! println!("NTP offset: {:.6} s", result.offset);
//! println!("Round-trip delay: {:.6} s", result.delay);
//! # Ok(())
//! # }
//! ```
//!
//! ## Clock Discipline
//!
//! ```rust
//! use oximedia_timesync::{ClockDiscipline, ClockSource};
//!
//! let mut discipline = ClockDiscipline::new();
//!
//! // Update with measured offset
//! let adjustment = discipline.update(10_000, ClockSource::Ptp).expect("should succeed in test");
//! println!("Recommended adjustment: {:?}", adjustment);
//! ```
//!
//! ## Timecode Synchronization
//!
//! ```rust
//! use oximedia_timesync::timecode::{TimecodeState, TimecodeSource};
//! use oximedia_timecode::{Timecode, FrameRate};
//!
//! let mut state = TimecodeState::new(FrameRate::Fps25);
//! let tc = Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("should succeed in test");
//! state.update(tc, TimecodeSource::Ltc);
//!
//! println!("Locked to LTC: {}", state.locked);
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]

pub mod aes67;
pub mod boundary_clock;
pub mod clock;
pub mod clock_discipline;
pub mod clock_domain;
pub mod clock_ensemble;
pub mod clock_error;
pub mod clock_graph;
pub mod clock_quality;
pub mod clock_quality_monitor;
pub mod clock_recovery;
pub mod clock_steering;
pub mod dante_clock;
pub mod drift_monitor;
pub mod error;
#[cfg(not(target_arch = "wasm32"))]
pub mod ffi;
pub mod frequency_estimator;
pub mod frequency_sync;
pub mod gps_reference;
pub mod gptp;
pub mod holdover_estimator;
pub mod integration;
pub mod ipc;
pub mod jitter_buffer;
pub mod leap_second;
pub mod ntp;
pub mod offset_correction;
pub mod offset_filter;
pub mod phase_lock;
pub mod ptp;
pub mod ptp_management;
pub mod ravenna;
pub mod reference_clock;
pub mod smpte_2059;
pub mod sync;
pub mod sync_audit;
pub mod sync_chain;
pub mod sync_metrics;
pub mod sync_monitor;
pub mod sync_protocol;
pub mod sync_stats;
pub mod sync_status;
pub mod sync_test;
pub mod sync_window;
pub mod time_reference;
pub mod time_scale;
pub mod timecode;
pub mod white_rabbit;

// Re-export commonly used types
pub use error::{TimeSyncError, TimeSyncResult};

pub use ptp::{
    ClockIdentity, CommunicationMode, DelayMechanism, Domain, PortIdentity, PtpTimestamp,
};

#[cfg(not(target_arch = "wasm32"))]
pub use ntp::NtpClient;
pub use ntp::{NtpPacket, NtpTimestamp, ServerPool, Stratum};

pub use clock::{
    discipline::ClockDiscipline, drift::DriftEstimator, holdover::HoldoverManager,
    offset::OffsetFilter, selection::SourceSelector, ClockSource, ClockStats, SyncState,
};

pub use timecode::{TimecodeSource, TimecodeState};

pub use sync::{
    audio::AudioSync,
    genlock::{GenlockFrameRate, GenlockGenerator},
    video::{FrameAccurateSync, VideoSync},
    SyncMode,
};

pub use integration::{
    adjust_timestamp, system_time_to_timestamp, timestamp_to_system_time, TimestampSync,
};

pub use ipc::{StateInfo, TimeSyncMessage};

/// Library version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_exports() {
        // Ensure all main types are accessible
        let _domain = Domain::DEFAULT;
        let _source = ClockSource::Ptp;
        let _state = SyncState::Unsync;
    }
}
