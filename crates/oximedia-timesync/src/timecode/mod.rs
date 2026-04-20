//! Timecode synchronization (LTC, SMPTE, MTC).

pub mod jam_sync;
pub mod ltc;
pub mod mtc;
pub mod smpte;

use oximedia_timecode::{FrameRate, Timecode, TimecodeError};
use std::time::Duration;

/// Timecode source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimecodeSource {
    /// Linear Timecode (LTC)
    Ltc,
    /// MIDI Time Code (MTC)
    Mtc,
    /// SMPTE timecode
    Smpte,
    /// Internal generator
    Internal,
}

/// Timecode synchronization state.
#[derive(Debug, Clone)]
pub struct TimecodeState {
    /// Current timecode
    pub current: Timecode,
    /// Source of timecode
    pub source: TimecodeSource,
    /// Last update timestamp
    pub last_update: std::time::Instant,
    /// Frame rate
    pub frame_rate: FrameRate,
    /// Whether locked to external source
    pub locked: bool,
}

impl TimecodeState {
    /// Create a new timecode state.
    ///
    /// # Errors
    ///
    /// Returns an error if the initial zero timecode cannot be created for the
    /// given frame rate (this should never occur in practice since frame 0 is
    /// always valid).
    pub fn new(frame_rate: FrameRate) -> Result<Self, TimecodeError> {
        let current = Timecode::new(0, 0, 0, 0, frame_rate)?;
        Ok(Self {
            current,
            source: TimecodeSource::Internal,
            last_update: std::time::Instant::now(),
            frame_rate,
            locked: false,
        })
    }

    /// Update with new timecode.
    pub fn update(&mut self, timecode: Timecode, source: TimecodeSource) {
        self.current = timecode;
        self.source = source;
        self.last_update = std::time::Instant::now();
        self.locked = source != TimecodeSource::Internal;
    }

    /// Check if timecode is stale (not updated recently).
    #[must_use]
    pub fn is_stale(&self, threshold: Duration) -> bool {
        self.last_update.elapsed() > threshold
    }

    /// Increment timecode by one frame.
    pub fn increment(&mut self) -> Result<(), oximedia_timecode::TimecodeError> {
        self.current.increment()?;
        self.last_update = std::time::Instant::now();
        if self.locked {
            self.locked = false; // Lost lock
            self.source = TimecodeSource::Internal;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_timecode_state_creation() {
        let state = TimecodeState::new(FrameRate::Fps25).expect("should succeed in test");
        assert_eq!(state.source, TimecodeSource::Internal);
        assert!(!state.locked);
    }

    #[test]
    fn test_timecode_update() {
        let mut state = TimecodeState::new(FrameRate::Fps25).expect("should succeed in test");
        let tc = Timecode::new(1, 2, 3, 4, FrameRate::Fps25).expect("should succeed in test");

        state.update(tc, TimecodeSource::Ltc);
        assert_eq!(state.source, TimecodeSource::Ltc);
        assert!(state.locked);
    }

    #[test]
    fn test_timecode_increment() {
        let mut state = TimecodeState::new(FrameRate::Fps25).expect("should succeed in test");
        state.locked = true;

        state.increment().expect("should succeed in test");
        assert_eq!(state.current.frames, 1);
        assert!(!state.locked); // Lost lock after manual increment
    }
}
