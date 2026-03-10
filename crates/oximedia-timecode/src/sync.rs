//! Timecode Synchronization
//!
//! This module provides synchronization between multiple timecode sources:
//! - LTC and VITC cross-checking
//! - Jam sync for timecode generation
//! - Clock drift correction
//! - Multi-source reconciliation
//! - Genlock synchronization

use crate::{FrameRate, Timecode, TimecodeError};

/// Timecode synchronizer
pub struct TimecodeSynchronizer {
    /// Reference frame rate
    frame_rate: FrameRate,
    /// Current synchronized timecode
    current_timecode: Option<Timecode>,
    /// LTC source state
    ltc_state: SourceState,
    /// VITC source state
    vitc_state: SourceState,
    /// Jam sync state
    jam_sync: Option<JamSyncState>,
    /// Drift corrector
    drift_corrector: DriftCorrector,
    /// Reconciliation strategy
    strategy: ReconciliationStrategy,
}

impl TimecodeSynchronizer {
    /// Create a new synchronizer
    pub fn new(frame_rate: FrameRate, strategy: ReconciliationStrategy) -> Self {
        TimecodeSynchronizer {
            frame_rate,
            current_timecode: None,
            ltc_state: SourceState::new("LTC"),
            vitc_state: SourceState::new("VITC"),
            jam_sync: None,
            drift_corrector: DriftCorrector::new(frame_rate),
            strategy,
        }
    }

    /// Update with LTC timecode
    pub fn update_ltc(&mut self, timecode: Timecode) -> Result<(), TimecodeError> {
        self.ltc_state.update(timecode);
        self.reconcile()
    }

    /// Update with VITC timecode
    pub fn update_vitc(&mut self, timecode: Timecode) -> Result<(), TimecodeError> {
        self.vitc_state.update(timecode);
        self.reconcile()
    }

    /// Get the current synchronized timecode
    pub fn get_timecode(&self) -> Option<Timecode> {
        self.current_timecode
    }

    /// Enable jam sync from a reference timecode
    pub fn jam_sync(&mut self, reference: Timecode) {
        self.jam_sync = Some(JamSyncState::new(reference, self.frame_rate));
        self.current_timecode = Some(reference);
    }

    /// Disable jam sync
    pub fn disable_jam_sync(&mut self) {
        self.jam_sync = None;
    }

    /// Check if jam sync is active
    pub fn is_jam_synced(&self) -> bool {
        self.jam_sync.is_some()
    }

    /// Reconcile timecodes from multiple sources
    fn reconcile(&mut self) -> Result<(), TimecodeError> {
        let ltc_tc = self.ltc_state.last_timecode();
        let vitc_tc = self.vitc_state.last_timecode();

        let new_timecode = match self.strategy {
            ReconciliationStrategy::PreferLtc => ltc_tc.or(vitc_tc),
            ReconciliationStrategy::PreferVitc => vitc_tc.or(ltc_tc),
            ReconciliationStrategy::CrossCheck => self.cross_check_timecodes(ltc_tc, vitc_tc),
            ReconciliationStrategy::MostRecent => self.select_most_recent(ltc_tc, vitc_tc),
        };

        if let Some(tc) = new_timecode {
            // Apply drift correction
            let corrected = self.drift_corrector.correct(tc)?;
            self.current_timecode = Some(corrected);
        }

        Ok(())
    }

    /// Cross-check timecodes and select the most reliable
    fn cross_check_timecodes(
        &self,
        ltc: Option<Timecode>,
        vitc: Option<Timecode>,
    ) -> Option<Timecode> {
        match (ltc, vitc) {
            (Some(ltc_tc), Some(vitc_tc)) => {
                // Check if timecodes match (within tolerance)
                let ltc_frames = ltc_tc.to_frames();
                let vitc_frames = vitc_tc.to_frames();
                let diff = (ltc_frames as i64 - vitc_frames as i64).abs();

                if diff <= 2 {
                    // Timecodes match - prefer LTC for continuous playback
                    Some(ltc_tc)
                } else {
                    // Mismatch - prefer VITC at low speeds, LTC at normal speed
                    if self.ltc_state.is_reliable() {
                        Some(ltc_tc)
                    } else {
                        Some(vitc_tc)
                    }
                }
            }
            (Some(tc), None) | (None, Some(tc)) => Some(tc),
            (None, None) => None,
        }
    }

    /// Select the most recently updated timecode
    fn select_most_recent(
        &self,
        ltc: Option<Timecode>,
        vitc: Option<Timecode>,
    ) -> Option<Timecode> {
        let ltc_age = self.ltc_state.age_ms();
        let vitc_age = self.vitc_state.age_ms();

        match (ltc, vitc) {
            (Some(ltc_tc), Some(_vitc_tc)) => {
                if ltc_age < vitc_age {
                    Some(ltc_tc)
                } else {
                    Some(_vitc_tc)
                }
            }
            (Some(tc), None) | (None, Some(tc)) => Some(tc),
            (None, None) => None,
        }
    }

    /// Reset synchronizer state
    pub fn reset(&mut self) {
        self.current_timecode = None;
        self.ltc_state.reset();
        self.vitc_state.reset();
        self.jam_sync = None;
        self.drift_corrector.reset();
    }

    /// Get synchronization status
    pub fn status(&self) -> SyncStatus {
        SyncStatus {
            is_synchronized: self.current_timecode.is_some(),
            ltc_available: self.ltc_state.is_available(),
            vitc_available: self.vitc_state.is_available(),
            jam_sync_active: self.jam_sync.is_some(),
            drift_ppm: self.drift_corrector.drift_ppm(),
        }
    }
}

/// Source state tracking
struct SourceState {
    /// Source name
    #[allow(dead_code)]
    name: String,
    /// Last timecode received
    last_timecode: Option<Timecode>,
    /// Timestamp of last update (in milliseconds)
    last_update_ms: u64,
    /// Reliability score (0.0 to 1.0)
    reliability: f32,
    /// Error count
    #[allow(dead_code)]
    error_count: u32,
}

impl SourceState {
    fn new(name: &str) -> Self {
        SourceState {
            name: name.to_string(),
            last_timecode: None,
            last_update_ms: 0,
            reliability: 0.0,
            error_count: 0,
        }
    }

    fn update(&mut self, timecode: Timecode) {
        // Validate timecode continuity
        if let Some(last) = self.last_timecode {
            let expected_frames = last.to_frames() + 1;
            let actual_frames = timecode.to_frames();

            if (expected_frames as i64 - actual_frames as i64).abs() > 5 {
                self.error_count += 1;
                self.reliability = (self.reliability - 0.1).max(0.0);
            } else {
                self.reliability = (self.reliability + 0.1).min(1.0);
            }
        }

        self.last_timecode = Some(timecode);
        self.last_update_ms = current_time_ms();
    }

    fn last_timecode(&self) -> Option<Timecode> {
        self.last_timecode
    }

    fn is_available(&self) -> bool {
        self.last_timecode.is_some() && self.age_ms() < 1000
    }

    fn is_reliable(&self) -> bool {
        self.reliability > 0.7
    }

    fn age_ms(&self) -> u64 {
        current_time_ms().saturating_sub(self.last_update_ms)
    }

    fn reset(&mut self) {
        self.last_timecode = None;
        self.last_update_ms = 0;
        self.reliability = 0.0;
        self.error_count = 0;
    }
}

/// Jam sync state
#[allow(dead_code)]
struct JamSyncState {
    /// Reference timecode
    reference: Timecode,
    /// Frame rate
    frame_rate: FrameRate,
    /// Start time (in milliseconds)
    start_time_ms: u64,
    /// Accumulated frames since start
    accumulated_frames: u64,
}

impl JamSyncState {
    fn new(reference: Timecode, frame_rate: FrameRate) -> Self {
        JamSyncState {
            reference,
            frame_rate,
            start_time_ms: current_time_ms(),
            accumulated_frames: 0,
        }
    }

    /// Generate current timecode based on elapsed time
    #[allow(dead_code)]
    fn generate_current(&mut self) -> Result<Timecode, TimecodeError> {
        let elapsed_ms = current_time_ms().saturating_sub(self.start_time_ms);
        let fps = self.frame_rate.as_float();
        let frames = ((elapsed_ms as f64 / 1000.0) * fps) as u64;

        let total_frames = self.reference.to_frames() + frames;
        Timecode::from_frames(total_frames, self.frame_rate)
    }
}

/// Drift corrector
struct DriftCorrector {
    /// Frame rate
    frame_rate: FrameRate,
    /// Reference clock (in frames)
    reference_frames: u64,
    /// Actual frames received
    actual_frames: u64,
    /// Drift in PPM (parts per million)
    drift_ppm: f32,
    /// History for drift calculation
    history: Vec<(u64, u64)>, // (timestamp_ms, frames)
}

impl DriftCorrector {
    fn new(frame_rate: FrameRate) -> Self {
        DriftCorrector {
            frame_rate,
            reference_frames: 0,
            actual_frames: 0,
            drift_ppm: 0.0,
            history: Vec::new(),
        }
    }

    /// Correct timecode for drift
    fn correct(&mut self, timecode: Timecode) -> Result<Timecode, TimecodeError> {
        let frames = timecode.to_frames();
        let timestamp = current_time_ms();

        // Add to history
        self.history.push((timestamp, frames));
        if self.history.len() > 100 {
            self.history.remove(0);
        }

        // Calculate drift if we have enough history
        if self.history.len() >= 10 {
            self.calculate_drift();
        }

        // Apply correction if drift is significant
        if self.drift_ppm.abs() > 100.0 {
            let correction_frames = (frames as f32 * self.drift_ppm / 1_000_000.0) as i64;
            let corrected_frames = (frames as i64 + correction_frames).max(0) as u64;
            Timecode::from_frames(corrected_frames, self.frame_rate)
        } else {
            Ok(timecode)
        }
    }

    /// Calculate drift from history
    fn calculate_drift(&mut self) {
        if self.history.len() < 2 {
            return;
        }

        let (first_time, first_frames) = self.history[0];
        let (last_time, last_frames) = match self.history.last() {
            Some(v) => *v,
            None => return,
        };

        let time_diff_ms = last_time.saturating_sub(first_time);
        let frame_diff = last_frames.saturating_sub(first_frames);

        if time_diff_ms > 0 {
            let expected_frames = (time_diff_ms as f64 / 1000.0) * self.frame_rate.as_float();
            let drift = (frame_diff as f64 - expected_frames) / expected_frames;
            self.drift_ppm = (drift * 1_000_000.0) as f32;
        }
    }

    fn drift_ppm(&self) -> f32 {
        self.drift_ppm
    }

    fn reset(&mut self) {
        self.reference_frames = 0;
        self.actual_frames = 0;
        self.drift_ppm = 0.0;
        self.history.clear();
    }
}

/// Reconciliation strategy for multiple sources
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconciliationStrategy {
    /// Prefer LTC, fall back to VITC
    PreferLtc,
    /// Prefer VITC, fall back to LTC
    PreferVitc,
    /// Cross-check both sources
    CrossCheck,
    /// Use most recently updated source
    MostRecent,
}

/// Synchronization status
#[derive(Debug, Clone)]
pub struct SyncStatus {
    /// Is timecode synchronized
    pub is_synchronized: bool,
    /// Is LTC available
    pub ltc_available: bool,
    /// Is VITC available
    pub vitc_available: bool,
    /// Is jam sync active
    pub jam_sync_active: bool,
    /// Clock drift in PPM
    pub drift_ppm: f32,
}

/// Genlock synchronizer
pub struct GenlockSynchronizer {
    /// Frame rate
    frame_rate: FrameRate,
    /// Reference phase
    reference_phase: f32,
    /// Current phase
    current_phase: f32,
    /// Phase error
    phase_error: f32,
    /// Locked status
    locked: bool,
}

impl GenlockSynchronizer {
    /// Create a new genlock synchronizer
    pub fn new(frame_rate: FrameRate) -> Self {
        GenlockSynchronizer {
            frame_rate,
            reference_phase: 0.0,
            current_phase: 0.0,
            phase_error: 0.0,
            locked: false,
        }
    }

    /// Update with reference sync pulse
    pub fn update_reference(&mut self, phase: f32) {
        self.reference_phase = phase;
        self.calculate_phase_error();
    }

    /// Update with timecode sync
    pub fn update_timecode(&mut self, timecode: &Timecode) {
        let frames = timecode.frames as f32;
        let fps = self.frame_rate.frames_per_second() as f32;
        self.current_phase = frames / fps;
        self.calculate_phase_error();
    }

    /// Calculate phase error
    fn calculate_phase_error(&mut self) {
        self.phase_error = self.current_phase - self.reference_phase;

        // Wrap phase error to [-0.5, 0.5]
        while self.phase_error > 0.5 {
            self.phase_error -= 1.0;
        }
        while self.phase_error < -0.5 {
            self.phase_error += 1.0;
        }

        // Update locked status
        self.locked = self.phase_error.abs() < 0.01;
    }

    /// Check if locked to reference
    pub fn is_locked(&self) -> bool {
        self.locked
    }

    /// Get phase error
    pub fn phase_error(&self) -> f32 {
        self.phase_error
    }

    /// Get correction in frames
    pub fn correction_frames(&self) -> i32 {
        let fps = self.frame_rate.frames_per_second() as f32;
        (self.phase_error * fps) as i32
    }

    /// Reset genlock state
    pub fn reset(&mut self) {
        self.reference_phase = 0.0;
        self.current_phase = 0.0;
        self.phase_error = 0.0;
        self.locked = false;
    }
}

/// Multi-source timecode aggregator
pub struct TimecodeAggregator {
    /// Sources
    sources: Vec<TimecodeSource>,
    /// Voting strategy
    strategy: VotingStrategy,
}

impl TimecodeAggregator {
    /// Create a new aggregator
    pub fn new(strategy: VotingStrategy) -> Self {
        TimecodeAggregator {
            sources: Vec::new(),
            strategy,
        }
    }

    /// Add a timecode source
    pub fn add_source(&mut self, name: String, timecode: Timecode, confidence: f32) {
        self.sources.push(TimecodeSource {
            name,
            timecode,
            confidence,
        });
    }

    /// Clear all sources
    pub fn clear_sources(&mut self) {
        self.sources.clear();
    }

    /// Get aggregated timecode
    pub fn aggregate(&self) -> Option<Timecode> {
        if self.sources.is_empty() {
            return None;
        }

        match self.strategy {
            VotingStrategy::Unanimous => self.unanimous_vote(),
            VotingStrategy::Majority => self.majority_vote(),
            VotingStrategy::HighestConfidence => self.highest_confidence(),
            VotingStrategy::WeightedAverage => self.weighted_average(),
        }
    }

    /// Unanimous vote - all sources must agree
    fn unanimous_vote(&self) -> Option<Timecode> {
        if self.sources.is_empty() {
            return None;
        }

        let first = &self.sources[0].timecode;
        for source in &self.sources[1..] {
            if source.timecode.to_frames() != first.to_frames() {
                return None;
            }
        }

        Some(*first)
    }

    /// Majority vote
    fn majority_vote(&self) -> Option<Timecode> {
        if self.sources.is_empty() {
            return None;
        }

        // Count occurrences of each timecode
        let mut counts: Vec<(u64, usize)> = Vec::new();
        for source in &self.sources {
            let frames = source.timecode.to_frames();
            if let Some(entry) = counts.iter_mut().find(|(f, _)| *f == frames) {
                entry.1 += 1;
            } else {
                counts.push((frames, 1));
            }
        }

        // Find majority
        counts.sort_by(|a, b| b.1.cmp(&a.1));
        if let Some((frames, _)) = counts.first() {
            // Return the timecode with the most votes
            self.sources
                .iter()
                .find(|s| s.timecode.to_frames() == *frames)
                .map(|s| s.timecode)
        } else {
            None
        }
    }

    /// Highest confidence
    fn highest_confidence(&self) -> Option<Timecode> {
        self.sources
            .iter()
            .max_by(|a, b| {
                a.confidence
                    .partial_cmp(&b.confidence)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|s| s.timecode)
    }

    /// Weighted average
    fn weighted_average(&self) -> Option<Timecode> {
        if self.sources.is_empty() {
            return None;
        }

        let total_weight: f32 = self.sources.iter().map(|s| s.confidence).sum();
        if total_weight == 0.0 {
            return None;
        }

        let weighted_frames: f64 = self
            .sources
            .iter()
            .map(|s| s.timecode.to_frames() as f64 * s.confidence as f64)
            .sum();

        let avg_frames = (weighted_frames / total_weight as f64) as u64;

        // Use frame rate from first source
        Timecode::from_frames(avg_frames, FrameRate::Fps25).ok()
    }
}

/// Timecode source
#[derive(Debug, Clone)]
struct TimecodeSource {
    #[allow(dead_code)]
    name: String,
    timecode: Timecode,
    confidence: f32,
}

/// Voting strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VotingStrategy {
    /// All sources must agree
    Unanimous,
    /// Majority wins
    Majority,
    /// Highest confidence wins
    HighestConfidence,
    /// Weighted average
    WeightedAverage,
}

/// Get current time in milliseconds since the Unix epoch.
fn current_time_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synchronizer_creation() {
        let sync = TimecodeSynchronizer::new(FrameRate::Fps25, ReconciliationStrategy::PreferLtc);
        assert!(sync.get_timecode().is_none());
    }

    #[test]
    fn test_jam_sync() {
        let mut sync =
            TimecodeSynchronizer::new(FrameRate::Fps25, ReconciliationStrategy::PreferLtc);
        let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");

        sync.jam_sync(tc);
        assert!(sync.is_jam_synced());
    }

    #[test]
    fn test_genlock() {
        let mut genlock = GenlockSynchronizer::new(FrameRate::Fps25);
        genlock.update_reference(0.0);

        let tc = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        genlock.update_timecode(&tc);

        // Phase for frame 0 is 0.0, which matches reference 0.0, so it should be locked
        assert!(genlock.is_locked());

        // Test with different phase
        genlock.update_reference(0.5);
        let tc2 = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        genlock.update_timecode(&tc2);
        // Phase error is now 0.5, so it should not be locked
        assert!(!genlock.is_locked());
    }

    #[test]
    fn test_aggregator() {
        let mut agg = TimecodeAggregator::new(VotingStrategy::HighestConfidence);

        let tc1 = Timecode::new(1, 0, 0, 0, FrameRate::Fps25).expect("valid timecode");
        let tc2 = Timecode::new(1, 0, 0, 1, FrameRate::Fps25).expect("valid timecode");

        agg.add_source("LTC".to_string(), tc1, 0.8);
        agg.add_source("VITC".to_string(), tc2, 0.9);

        let result = agg.aggregate();
        assert_eq!(result, Some(tc2));
    }
}
