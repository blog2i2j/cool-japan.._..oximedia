//! Sync drift detection and correction for multi-camera production.
//!
//! This module detects and corrects temporal drift between camera angles.

use super::{SyncOffset, SyncResult};
use crate::{AngleId, FrameNumber, Result};

/// Drift detector
#[derive(Debug)]
pub struct DriftDetector {
    /// Initial sync offsets
    initial_offsets: Vec<SyncOffset>,
    /// Current sync offsets
    current_offsets: Vec<SyncOffset>,
    /// Drift tolerance (frames)
    tolerance: f64,
    /// Frame rate
    frame_rate: f64,
}

/// Drift measurement
#[derive(Debug, Clone, Copy)]
pub struct DriftMeasurement {
    /// Angle identifier
    pub angle: AngleId,
    /// Drift amount in frames
    pub drift_frames: f64,
    /// Drift rate (frames per second)
    pub drift_rate: f64,
    /// Whether drift exceeds tolerance
    pub exceeds_tolerance: bool,
}

impl DriftDetector {
    /// Create a new drift detector
    #[must_use]
    pub fn new(initial_sync: &SyncResult, tolerance: f64, frame_rate: f64) -> Self {
        Self {
            initial_offsets: initial_sync.offsets.clone(),
            current_offsets: initial_sync.offsets.clone(),
            tolerance,
            frame_rate,
        }
    }

    /// Update current sync offsets
    pub fn update(&mut self, current_sync: &SyncResult) {
        self.current_offsets = current_sync.offsets.clone();
    }

    /// Detect drift for specific angle
    #[must_use]
    pub fn detect_drift(&self, angle: AngleId) -> Option<DriftMeasurement> {
        let initial = self.initial_offsets.iter().find(|o| o.angle == angle)?;
        let current = self.current_offsets.iter().find(|o| o.angle == angle)?;

        let drift_frames = current.total_frames() - initial.total_frames();
        let drift_rate = drift_frames * self.frame_rate;
        let exceeds_tolerance = drift_frames.abs() > self.tolerance;

        Some(DriftMeasurement {
            angle,
            drift_frames,
            drift_rate,
            exceeds_tolerance,
        })
    }

    /// Detect drift for all angles
    #[must_use]
    pub fn detect_all_drift(&self) -> Vec<DriftMeasurement> {
        (0..self.initial_offsets.len())
            .filter_map(|angle| self.detect_drift(angle))
            .collect()
    }

    /// Check if any angle has drifted beyond tolerance
    #[must_use]
    pub fn has_drift(&self) -> bool {
        self.detect_all_drift().iter().any(|m| m.exceeds_tolerance)
    }

    /// Get maximum drift across all angles
    #[must_use]
    pub fn max_drift(&self) -> f64 {
        self.detect_all_drift()
            .iter()
            .map(|m| m.drift_frames.abs())
            .fold(0.0f64, f64::max)
    }

    /// Correct drift by adjusting offsets
    pub fn correct_drift(&mut self, angle: AngleId, correction_frames: f64) {
        if let Some(offset) = self.current_offsets.iter_mut().find(|o| o.angle == angle) {
            let total = offset.total_frames() + correction_frames;
            offset.frames = total.floor() as i64;
            offset.sub_frame = total - offset.frames as f64;
        }
    }

    /// Auto-correct drift for all angles
    pub fn auto_correct(&mut self) {
        for angle in 0..self.initial_offsets.len() {
            if let Some(measurement) = self.detect_drift(angle) {
                if measurement.exceeds_tolerance {
                    self.correct_drift(angle, -measurement.drift_frames);
                }
            }
        }
    }

    /// Predict drift at future frame
    #[must_use]
    pub fn predict_drift(
        &self,
        angle: AngleId,
        future_frame: FrameNumber,
        current_frame: FrameNumber,
    ) -> f64 {
        if let Some(measurement) = self.detect_drift(angle) {
            let elapsed = (future_frame - current_frame) as f64;
            let drift_per_frame = measurement.drift_rate / self.frame_rate;
            measurement.drift_frames + (drift_per_frame * elapsed)
        } else {
            0.0
        }
    }

    /// Calculate drift velocity (frames per frame)
    #[must_use]
    pub fn drift_velocity(&self, angle: AngleId, time_elapsed: f64) -> f64 {
        if let Some(measurement) = self.detect_drift(angle) {
            if time_elapsed > 0.0 {
                measurement.drift_frames / time_elapsed
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Set tolerance
    pub fn set_tolerance(&mut self, tolerance: f64) {
        self.tolerance = tolerance;
    }

    /// Get tolerance
    #[must_use]
    pub fn tolerance(&self) -> f64 {
        self.tolerance
    }

    /// Reset drift detection to current state
    pub fn reset(&mut self) {
        self.initial_offsets = self.current_offsets.clone();
    }
}

/// Drift correction strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DriftCorrectionStrategy {
    /// No correction
    None,
    /// Immediate correction when drift detected
    Immediate,
    /// Gradual correction over time
    Gradual,
    /// Correct only at scene boundaries
    SceneBoundary,
}

/// Drift corrector
#[derive(Debug)]
pub struct DriftCorrector {
    /// Strategy for correction
    strategy: DriftCorrectionStrategy,
    /// Correction rate for gradual strategy (frames per frame)
    correction_rate: f64,
    /// Pending corrections
    pending_corrections: Vec<(AngleId, f64)>,
}

impl DriftCorrector {
    /// Create a new drift corrector
    #[must_use]
    pub fn new(strategy: DriftCorrectionStrategy) -> Self {
        Self {
            strategy,
            correction_rate: 0.1, // Correct 10% per frame
            pending_corrections: Vec::new(),
        }
    }

    /// Set correction strategy
    pub fn set_strategy(&mut self, strategy: DriftCorrectionStrategy) {
        self.strategy = strategy;
    }

    /// Set correction rate for gradual strategy
    pub fn set_correction_rate(&mut self, rate: f64) {
        self.correction_rate = rate.clamp(0.0, 1.0);
    }

    /// Apply correction based on drift measurement
    ///
    /// # Errors
    ///
    /// Returns an error if correction fails
    pub fn apply_correction(&mut self, measurement: &DriftMeasurement) -> Result<f64> {
        match self.strategy {
            DriftCorrectionStrategy::None => Ok(0.0),
            DriftCorrectionStrategy::Immediate => Ok(-measurement.drift_frames),
            DriftCorrectionStrategy::Gradual => {
                let correction = -measurement.drift_frames * self.correction_rate;
                Ok(correction)
            }
            DriftCorrectionStrategy::SceneBoundary => {
                // Queue correction for next scene boundary
                self.pending_corrections
                    .push((measurement.angle, -measurement.drift_frames));
                Ok(0.0)
            }
        }
    }

    /// Apply pending corrections at scene boundary
    pub fn apply_pending(&mut self, angle: AngleId) -> f64 {
        let mut total_correction = 0.0;
        self.pending_corrections.retain(|(a, correction)| {
            if *a == angle {
                total_correction += *correction;
                false
            } else {
                true
            }
        });
        total_correction
    }

    /// Clear pending corrections
    pub fn clear_pending(&mut self) {
        self.pending_corrections.clear();
    }

    /// Get pending correction for angle
    #[must_use]
    pub fn get_pending(&self, angle: AngleId) -> f64 {
        self.pending_corrections
            .iter()
            .filter(|(a, _)| *a == angle)
            .map(|(_, c)| *c)
            .sum()
    }
}

/// Drift monitor for continuous monitoring
#[derive(Debug)]
pub struct DriftMonitor {
    /// Detector
    detector: DriftDetector,
    /// Corrector
    corrector: DriftCorrector,
    /// Monitoring interval (frames)
    monitor_interval: u64,
    /// Last monitoring frame
    last_monitor_frame: u64,
}

impl DriftMonitor {
    /// Create a new drift monitor
    #[must_use]
    pub fn new(
        initial_sync: &SyncResult,
        tolerance: f64,
        frame_rate: f64,
        strategy: DriftCorrectionStrategy,
        monitor_interval: u64,
    ) -> Self {
        Self {
            detector: DriftDetector::new(initial_sync, tolerance, frame_rate),
            corrector: DriftCorrector::new(strategy),
            monitor_interval,
            last_monitor_frame: 0,
        }
    }

    /// Check if monitoring is due at current frame
    #[must_use]
    pub fn should_monitor(&self, current_frame: u64) -> bool {
        current_frame >= self.last_monitor_frame + self.monitor_interval
    }

    /// Monitor and correct drift
    ///
    /// # Errors
    ///
    /// Returns an error if monitoring fails
    pub fn monitor(
        &mut self,
        current_frame: u64,
        current_sync: &SyncResult,
    ) -> Result<Vec<(AngleId, f64)>> {
        if !self.should_monitor(current_frame) {
            return Ok(Vec::new());
        }

        self.last_monitor_frame = current_frame;
        self.detector.update(current_sync);

        let measurements = self.detector.detect_all_drift();
        let mut corrections = Vec::new();

        for measurement in measurements {
            if measurement.exceeds_tolerance {
                let correction = self.corrector.apply_correction(&measurement)?;
                if correction.abs() > 0.0 {
                    corrections.push((measurement.angle, correction));
                }
            }
        }

        Ok(corrections)
    }

    /// Get detector
    #[must_use]
    pub fn detector(&self) -> &DriftDetector {
        &self.detector
    }

    /// Get corrector
    #[must_use]
    pub fn corrector(&self) -> &DriftCorrector {
        &self.corrector
    }

    /// Get mutable detector
    pub fn detector_mut(&mut self) -> &mut DriftDetector {
        &mut self.detector
    }

    /// Get mutable corrector
    pub fn corrector_mut(&mut self) -> &mut DriftCorrector {
        &mut self.corrector
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::{SyncMethod, SyncOffset, SyncResult};

    #[test]
    fn test_drift_detector_creation() {
        let sync_result = SyncResult {
            reference_angle: 0,
            offsets: vec![
                SyncOffset::new(0, 0, 0.0, 1.0),
                SyncOffset::new(1, 10, 0.0, 0.9),
            ],
            confidence: 0.95,
            method: SyncMethod::Audio,
        };

        let detector = DriftDetector::new(&sync_result, 2.0, 25.0);
        assert_eq!(detector.initial_offsets.len(), 2);
        assert_eq!(detector.tolerance, 2.0);
    }

    #[test]
    fn test_drift_detection() {
        let initial_sync = SyncResult {
            reference_angle: 0,
            offsets: vec![
                SyncOffset::new(0, 0, 0.0, 1.0),
                SyncOffset::new(1, 10, 0.0, 0.9),
            ],
            confidence: 0.95,
            method: SyncMethod::Audio,
        };

        let mut detector = DriftDetector::new(&initial_sync, 2.0, 25.0);

        // Simulate drift
        let drifted_sync = SyncResult {
            reference_angle: 0,
            offsets: vec![
                SyncOffset::new(0, 0, 0.0, 1.0),
                SyncOffset::new(1, 13, 0.0, 0.9), // Drifted by 3 frames
            ],
            confidence: 0.95,
            method: SyncMethod::Audio,
        };

        detector.update(&drifted_sync);

        let measurement = detector
            .detect_drift(1)
            .expect("multicam test operation should succeed");
        assert!((measurement.drift_frames - 3.0).abs() < 0.01);
        assert!(measurement.exceeds_tolerance);
    }

    #[test]
    fn test_drift_correction() {
        let sync_result = SyncResult {
            reference_angle: 0,
            offsets: vec![
                SyncOffset::new(0, 0, 0.0, 1.0),
                SyncOffset::new(1, 10, 0.0, 0.9),
            ],
            confidence: 0.95,
            method: SyncMethod::Audio,
        };

        let mut detector = DriftDetector::new(&sync_result, 2.0, 25.0);
        detector.correct_drift(1, 5.0);

        let offset = detector
            .current_offsets
            .iter()
            .find(|o| o.angle == 1)
            .expect("multicam test operation should succeed");
        assert!((offset.total_frames() - 15.0).abs() < 0.01);
    }

    #[test]
    fn test_drift_corrector_immediate() {
        let mut corrector = DriftCorrector::new(DriftCorrectionStrategy::Immediate);
        let measurement = DriftMeasurement {
            angle: 1,
            drift_frames: 5.0,
            drift_rate: 125.0,
            exceeds_tolerance: true,
        };

        let correction = corrector
            .apply_correction(&measurement)
            .expect("multicam test operation should succeed");
        assert_eq!(correction, -5.0);
    }

    #[test]
    fn test_drift_corrector_gradual() {
        let mut corrector = DriftCorrector::new(DriftCorrectionStrategy::Gradual);
        corrector.set_correction_rate(0.5);

        let measurement = DriftMeasurement {
            angle: 1,
            drift_frames: 10.0,
            drift_rate: 250.0,
            exceeds_tolerance: true,
        };

        let correction = corrector
            .apply_correction(&measurement)
            .expect("multicam test operation should succeed");
        assert_eq!(correction, -5.0); // 50% of 10 frames
    }

    #[test]
    fn test_drift_monitor() {
        let sync_result = SyncResult {
            reference_angle: 0,
            offsets: vec![
                SyncOffset::new(0, 0, 0.0, 1.0),
                SyncOffset::new(1, 10, 0.0, 0.9),
            ],
            confidence: 0.95,
            method: SyncMethod::Audio,
        };

        let monitor = DriftMonitor::new(
            &sync_result,
            2.0,
            25.0,
            DriftCorrectionStrategy::Immediate,
            100,
        );

        assert!(monitor.should_monitor(100));
        assert!(!monitor.should_monitor(50));
    }
}
