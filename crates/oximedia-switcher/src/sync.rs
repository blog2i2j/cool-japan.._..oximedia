//! Frame synchronization and genlock support for professional video switchers.
//!
//! This module provides frame-accurate synchronization across multiple video sources,
//! genlock support, and timecode management for broadcast-quality video production.

use oximedia_timecode::Timecode;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use thiserror::Error;

/// Errors that can occur during synchronization operations.
#[derive(Error, Debug, Clone)]
pub enum SyncError {
    #[error("Frame rate mismatch: expected {expected} fps, got {actual} fps")]
    FrameRateMismatch { expected: f64, actual: f64 },

    #[error("Genlock signal lost")]
    GenlockLost,

    #[error("Source not synchronized")]
    SourceNotSynchronized,

    #[error("Timing drift exceeded threshold: {drift_ms} ms")]
    TimingDrift { drift_ms: f64 },

    #[error("Invalid frame rate: {0}")]
    InvalidFrameRate(f64),
}

/// Video frame rates supported by the switcher.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FrameRate {
    /// 23.976 fps (film standard)
    Fps23_976,
    /// 24 fps (film)
    Fps24,
    /// 25 fps (PAL)
    Fps25,
    /// 29.97 fps (NTSC)
    Fps29_97,
    /// 30 fps
    Fps30,
    /// 50 fps (PAL progressive)
    Fps50,
    /// 59.94 fps (NTSC progressive)
    Fps59_94,
    /// 60 fps
    Fps60,
    /// 120 fps (high frame rate)
    Fps120,
}

impl FrameRate {
    /// Get the numeric frame rate value.
    pub fn value(&self) -> f64 {
        match self {
            FrameRate::Fps23_976 => 23.976,
            FrameRate::Fps24 => 24.0,
            FrameRate::Fps25 => 25.0,
            FrameRate::Fps29_97 => 29.97,
            FrameRate::Fps30 => 30.0,
            FrameRate::Fps50 => 50.0,
            FrameRate::Fps59_94 => 59.94,
            FrameRate::Fps60 => 60.0,
            FrameRate::Fps120 => 120.0,
        }
    }

    /// Get the frame duration.
    pub fn frame_duration(&self) -> Duration {
        Duration::from_secs_f64(1.0 / self.value())
    }

    /// Check if this is a drop-frame rate.
    pub fn is_drop_frame(&self) -> bool {
        matches!(self, FrameRate::Fps29_97 | FrameRate::Fps59_94)
    }
}

/// Genlock reference source.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GenlockSource {
    /// Internal free-running clock
    Internal,
    /// External black burst/tri-level sync
    External,
    /// Specific input source
    Input(usize),
}

/// Synchronization state for a video source.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SyncState {
    /// Whether the source is currently synchronized
    pub synchronized: bool,
    /// Frame number since synchronization started
    pub frame_count: u64,
    /// Current timecode
    pub timecode: Option<Timecode>,
    /// Genlock status
    pub genlock_locked: bool,
    /// Timing offset from reference (in microseconds)
    pub timing_offset_us: i64,
}

/// Frame synchronizer manages timing and genlock for all video sources.
pub struct FrameSynchronizer {
    frame_rate: FrameRate,
    genlock_source: GenlockSource,
    reference_time: Instant,
    frame_count: u64,
    max_drift_us: i64,
    sources: Vec<SyncState>,
}

impl FrameSynchronizer {
    /// Create a new frame synchronizer.
    pub fn new(frame_rate: FrameRate, num_sources: usize) -> Self {
        Self {
            frame_rate,
            genlock_source: GenlockSource::Internal,
            reference_time: Instant::now(),
            frame_count: 0,
            max_drift_us: 1000, // 1ms maximum drift
            sources: vec![SyncState::default(); num_sources],
        }
    }

    /// Get the current frame rate.
    pub fn frame_rate(&self) -> FrameRate {
        self.frame_rate
    }

    /// Set the frame rate.
    pub fn set_frame_rate(&mut self, frame_rate: FrameRate) -> Result<(), SyncError> {
        self.frame_rate = frame_rate;
        self.reset();
        Ok(())
    }

    /// Get the genlock source.
    pub fn genlock_source(&self) -> GenlockSource {
        self.genlock_source
    }

    /// Set the genlock source.
    pub fn set_genlock_source(&mut self, source: GenlockSource) {
        self.genlock_source = source;
        self.reset();
    }

    /// Reset synchronization.
    pub fn reset(&mut self) {
        self.reference_time = Instant::now();
        self.frame_count = 0;
        for state in &mut self.sources {
            state.synchronized = false;
            state.frame_count = 0;
            state.timing_offset_us = 0;
        }
    }

    /// Get the current frame number.
    pub fn current_frame(&self) -> u64 {
        self.frame_count
    }

    /// Get the reference time.
    pub fn reference_time(&self) -> Instant {
        self.reference_time
    }

    /// Advance to the next frame.
    pub fn advance_frame(&mut self) {
        self.frame_count += 1;
    }

    /// Get the time until the next frame boundary.
    pub fn time_until_next_frame(&self) -> Duration {
        let frame_duration = self.frame_rate.frame_duration();
        let elapsed = self.reference_time.elapsed();
        let next_frame_time = frame_duration * (self.frame_count as u32 + 1);

        if next_frame_time > elapsed {
            next_frame_time
                .checked_sub(elapsed)
                .unwrap_or(Duration::ZERO)
        } else {
            Duration::ZERO
        }
    }

    /// Check if it's time for the next frame.
    pub fn is_next_frame_ready(&self) -> bool {
        let frame_duration = self.frame_rate.frame_duration();
        let elapsed = self.reference_time.elapsed();
        let next_frame_time = frame_duration * (self.frame_count as u32 + 1);

        elapsed >= next_frame_time
    }

    /// Synchronize a specific source.
    pub fn sync_source(
        &mut self,
        source_id: usize,
        timecode: Option<Timecode>,
    ) -> Result<(), SyncError> {
        if source_id >= self.sources.len() {
            return Ok(());
        }

        let state = &mut self.sources[source_id];
        state.synchronized = true;
        state.frame_count = self.frame_count;
        state.timecode = timecode;
        state.genlock_locked = matches!(
            self.genlock_source,
            GenlockSource::External | GenlockSource::Input(_)
        );

        Ok(())
    }

    /// Update source timing information.
    pub fn update_source_timing(
        &mut self,
        source_id: usize,
        offset_us: i64,
    ) -> Result<(), SyncError> {
        if source_id >= self.sources.len() {
            return Ok(());
        }

        let state = &mut self.sources[source_id];
        state.timing_offset_us = offset_us;

        // Check for excessive drift
        if offset_us.abs() > self.max_drift_us {
            return Err(SyncError::TimingDrift {
                drift_ms: offset_us as f64 / 1000.0,
            });
        }

        Ok(())
    }

    /// Get the sync state for a source.
    pub fn get_source_state(&self, source_id: usize) -> Option<&SyncState> {
        self.sources.get(source_id)
    }

    /// Check if a source is synchronized.
    pub fn is_source_synchronized(&self, source_id: usize) -> bool {
        self.sources.get(source_id).is_some_and(|s| s.synchronized)
    }

    /// Check if all sources are synchronized.
    pub fn all_sources_synchronized(&self) -> bool {
        self.sources.iter().all(|s| s.synchronized)
    }

    /// Get the number of synchronized sources.
    pub fn synchronized_count(&self) -> usize {
        self.sources.iter().filter(|s| s.synchronized).count()
    }

    /// Set maximum allowed timing drift in microseconds.
    pub fn set_max_drift_us(&mut self, max_drift_us: i64) {
        self.max_drift_us = max_drift_us;
    }

    /// Get the maximum allowed timing drift.
    pub fn max_drift_us(&self) -> i64 {
        self.max_drift_us
    }
}

/// Genlock controller for external synchronization.
pub struct GenlockController {
    enabled: bool,
    locked: bool,
    reference_frame_rate: FrameRate,
    last_pulse_time: Option<Instant>,
    pulse_count: u64,
}

impl GenlockController {
    /// Create a new genlock controller.
    pub fn new(frame_rate: FrameRate) -> Self {
        Self {
            enabled: false,
            locked: false,
            reference_frame_rate: frame_rate,
            last_pulse_time: None,
            pulse_count: 0,
        }
    }

    /// Enable genlock.
    pub fn enable(&mut self) {
        self.enabled = true;
    }

    /// Disable genlock.
    pub fn disable(&mut self) {
        self.enabled = false;
        self.locked = false;
    }

    /// Check if genlock is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }

    /// Check if genlock is locked.
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Process a genlock pulse.
    pub fn process_pulse(&mut self, pulse_time: Instant) -> Result<(), SyncError> {
        if !self.enabled {
            return Ok(());
        }

        if let Some(last_time) = self.last_pulse_time {
            let duration = pulse_time.duration_since(last_time);
            let expected_duration = self.reference_frame_rate.frame_duration();

            // Check if pulse timing is within tolerance (±5%)
            let tolerance = expected_duration.as_secs_f64() * 0.05;
            let diff = (duration.as_secs_f64() - expected_duration.as_secs_f64()).abs();

            if diff <= tolerance {
                self.locked = true;
                self.pulse_count += 1;
            } else {
                self.locked = false;
                return Err(SyncError::GenlockLost);
            }
        }

        self.last_pulse_time = Some(pulse_time);
        Ok(())
    }

    /// Get the pulse count.
    pub fn pulse_count(&self) -> u64 {
        self.pulse_count
    }

    /// Reset the genlock controller.
    pub fn reset(&mut self) {
        self.locked = false;
        self.last_pulse_time = None;
        self.pulse_count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_rate_values() {
        assert_eq!(FrameRate::Fps25.value(), 25.0);
        assert_eq!(FrameRate::Fps29_97.value(), 29.97);
        assert_eq!(FrameRate::Fps60.value(), 60.0);
    }

    #[test]
    fn test_drop_frame_detection() {
        assert!(FrameRate::Fps29_97.is_drop_frame());
        assert!(FrameRate::Fps59_94.is_drop_frame());
        assert!(!FrameRate::Fps25.is_drop_frame());
        assert!(!FrameRate::Fps30.is_drop_frame());
    }

    #[test]
    fn test_frame_synchronizer_creation() {
        let sync = FrameSynchronizer::new(FrameRate::Fps25, 4);
        assert_eq!(sync.frame_rate(), FrameRate::Fps25);
        assert_eq!(sync.current_frame(), 0);
        assert!(!sync.all_sources_synchronized());
    }

    #[test]
    fn test_frame_advance() {
        let mut sync = FrameSynchronizer::new(FrameRate::Fps25, 4);
        assert_eq!(sync.current_frame(), 0);

        sync.advance_frame();
        assert_eq!(sync.current_frame(), 1);

        sync.advance_frame();
        assert_eq!(sync.current_frame(), 2);
    }

    #[test]
    fn test_source_synchronization() {
        let mut sync = FrameSynchronizer::new(FrameRate::Fps25, 4);

        assert!(!sync.is_source_synchronized(0));
        assert_eq!(sync.synchronized_count(), 0);

        sync.sync_source(0, None).expect("should succeed in test");
        assert!(sync.is_source_synchronized(0));
        assert_eq!(sync.synchronized_count(), 1);

        sync.sync_source(1, None).expect("should succeed in test");
        assert_eq!(sync.synchronized_count(), 2);
    }

    #[test]
    fn test_timing_drift_detection() {
        let mut sync = FrameSynchronizer::new(FrameRate::Fps25, 4);
        sync.set_max_drift_us(1000);

        // Small drift should be ok
        assert!(sync.update_source_timing(0, 500).is_ok());

        // Large drift should error
        assert!(sync.update_source_timing(0, 2000).is_err());
    }

    #[test]
    fn test_genlock_controller() {
        let mut genlock = GenlockController::new(FrameRate::Fps25);
        assert!(!genlock.is_enabled());
        assert!(!genlock.is_locked());

        genlock.enable();
        assert!(genlock.is_enabled());

        let first_pulse = Instant::now();
        genlock
            .process_pulse(first_pulse)
            .expect("should succeed in test");
        assert_eq!(genlock.pulse_count(), 0);

        // Simulate proper frame-rate pulses with more robust timing
        let frame_duration = FrameRate::Fps25.frame_duration();
        std::thread::sleep(frame_duration);
        // Capture time immediately after waking up
        let second_pulse = first_pulse + frame_duration + std::time::Duration::from_millis(1);
        genlock
            .process_pulse(second_pulse)
            .expect("should succeed in test");
        assert!(genlock.is_locked());
        assert_eq!(genlock.pulse_count(), 1);
    }

    #[test]
    fn test_genlock_source_types() {
        let mut sync = FrameSynchronizer::new(FrameRate::Fps25, 4);

        assert_eq!(sync.genlock_source(), GenlockSource::Internal);

        sync.set_genlock_source(GenlockSource::External);
        assert_eq!(sync.genlock_source(), GenlockSource::External);

        sync.set_genlock_source(GenlockSource::Input(2));
        assert_eq!(sync.genlock_source(), GenlockSource::Input(2));
    }
}
