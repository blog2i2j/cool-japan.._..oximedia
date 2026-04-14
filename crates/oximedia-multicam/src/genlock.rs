//! Genlock drift correction for long-form multi-camera recording.
//!
//! Tracks the phase relationship between camera clocks and a reference,
//! computes drift in microseconds, and recommends corrections. Designed
//! for sessions exceeding one hour where crystal-oscillator drift
//! accumulates to visible sync errors.

use std::collections::HashMap;
use std::fmt;

use crate::{MultiCamError, Result};

/// Correction mode applied when drift exceeds tolerance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CorrectionMode {
    /// Drop or duplicate a single frame to resync.
    FrameDropDuplicate,
    /// Smoothly speed-ramp playback rate over a window.
    SpeedRamp,
    /// Re-timestamp packets without altering media.
    RetimestampOnly,
    /// Do not correct automatically; report only.
    ReportOnly,
}

impl fmt::Display for CorrectionMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::FrameDropDuplicate => write!(f, "FrameDropDuplicate"),
            Self::SpeedRamp => write!(f, "SpeedRamp"),
            Self::RetimestampOnly => write!(f, "RetimestampOnly"),
            Self::ReportOnly => write!(f, "ReportOnly"),
        }
    }
}

/// Genlock drift-correction configuration.
#[derive(Debug, Clone)]
pub struct GenlockConfig {
    /// Reference frame rate (e.g. 25.0, 29.97, 59.94).
    pub reference_fps: f64,
    /// Maximum acceptable drift in microseconds before correction triggers.
    pub tolerance_us: f64,
    /// Correction strategy.
    pub correction_mode: CorrectionMode,
    /// Window size (in samples) for the moving-average filter on drift.
    pub averaging_window: usize,
    /// Number of consecutive over-tolerance samples before status becomes `Lost`.
    pub lost_threshold: usize,
}

impl Default for GenlockConfig {
    fn default() -> Self {
        Self {
            reference_fps: 25.0,
            tolerance_us: 500.0,
            correction_mode: CorrectionMode::SpeedRamp,
            averaging_window: 16,
            lost_threshold: 30,
        }
    }
}

impl GenlockConfig {
    /// Validate the configuration, returning an error for invalid values.
    pub fn validate(&self) -> Result<()> {
        if self.reference_fps <= 0.0 {
            return Err(MultiCamError::ConfigError(
                "reference_fps must be positive".into(),
            ));
        }
        if self.tolerance_us <= 0.0 {
            return Err(MultiCamError::ConfigError(
                "tolerance_us must be positive".into(),
            ));
        }
        if self.averaging_window == 0 {
            return Err(MultiCamError::ConfigError(
                "averaging_window must be >= 1".into(),
            ));
        }
        if self.lost_threshold == 0 {
            return Err(MultiCamError::ConfigError(
                "lost_threshold must be >= 1".into(),
            ));
        }
        Ok(())
    }

    /// Frame interval in microseconds derived from `reference_fps`.
    #[must_use]
    pub fn frame_interval_us(&self) -> f64 {
        if self.reference_fps <= 0.0 {
            return 0.0;
        }
        1_000_000.0 / self.reference_fps
    }
}

/// Status of genlock synchronization for a single camera.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenlockStatus {
    /// Phase-locked within tolerance.
    Locked,
    /// Drift detected but within recovery range.
    Drifting,
    /// Lock has been lost; manual intervention may be needed.
    Lost,
}

impl fmt::Display for GenlockStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Locked => write!(f, "LOCKED"),
            Self::Drifting => write!(f, "DRIFTING"),
            Self::Lost => write!(f, "LOST"),
        }
    }
}

/// A recommended correction to bring a camera back into sync.
#[derive(Debug, Clone)]
pub struct CorrectionRecommendation {
    /// Camera identifier.
    pub camera_id: u32,
    /// Current drift in microseconds (positive = camera leads reference).
    pub drift_us: f64,
    /// Which correction mode to apply.
    pub mode: CorrectionMode,
    /// For `SpeedRamp`: playback rate multiplier (e.g. 1.001).
    /// For `FrameDropDuplicate`: number of frames to drop (negative) or duplicate (positive).
    /// For others: 0.0.
    pub correction_value: f64,
    /// Human-readable description of the recommendation.
    pub description: String,
}

/// Per-camera tracking state inside the monitor.
#[derive(Debug, Clone)]
struct CameraState {
    /// Camera label.
    _name: String,
    /// Ring buffer of recent drift measurements in microseconds.
    drift_history: Vec<f64>,
    /// Write position in the ring buffer.
    history_pos: usize,
    /// Whether the ring buffer has wrapped at least once.
    history_full: bool,
    /// Current status.
    status: GenlockStatus,
    /// Consecutive samples exceeding tolerance.
    over_tolerance_count: usize,
    /// Total samples received.
    total_samples: u64,
}

impl CameraState {
    fn new(name: &str, window: usize) -> Self {
        Self {
            _name: name.to_owned(),
            drift_history: vec![0.0; window],
            history_pos: 0,
            history_full: false,
            status: GenlockStatus::Locked,
            over_tolerance_count: 0,
            total_samples: 0,
        }
    }

    fn push_drift(&mut self, drift_us: f64) {
        let len = self.drift_history.len();
        if len > 0 {
            self.drift_history[self.history_pos % len] = drift_us;
            self.history_pos += 1;
            if self.history_pos >= len {
                self.history_full = true;
                self.history_pos %= len;
            }
        }
        self.total_samples += 1;
    }

    fn averaged_drift(&self) -> f64 {
        let count = if self.history_full {
            self.drift_history.len()
        } else {
            self.history_pos
        };
        if count == 0 {
            return 0.0;
        }
        let sum: f64 = self.drift_history[..count].iter().sum();
        sum / count as f64
    }
}

/// Genlock monitor that tracks the phase relationship between multiple
/// cameras and a reference clock, computes drift, and recommends
/// corrections.
#[derive(Debug)]
pub struct GenlockMonitor {
    config: GenlockConfig,
    cameras: HashMap<u32, CameraState>,
    next_id: u32,
}

impl GenlockMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: GenlockConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            config,
            cameras: HashMap::new(),
            next_id: 0,
        })
    }

    /// Create a monitor with default configuration.
    pub fn with_defaults() -> Result<Self> {
        Self::new(GenlockConfig::default())
    }

    /// Register a camera. Returns its identifier.
    pub fn add_camera(&mut self, name: &str) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.cameras.insert(
            id,
            CameraState::new(name, self.config.averaging_window),
        );
        id
    }

    /// Remove a camera by identifier.
    pub fn remove_camera(&mut self, id: u32) -> bool {
        self.cameras.remove(&id).is_some()
    }

    /// Number of registered cameras.
    #[must_use]
    pub fn camera_count(&self) -> usize {
        self.cameras.len()
    }

    /// Report a phase measurement for a camera.
    ///
    /// `drift_us` is the signed drift in microseconds (positive means
    /// the camera clock leads the reference).
    pub fn report_drift(&mut self, camera_id: u32, drift_us: f64) -> Result<GenlockStatus> {
        let state = self
            .cameras
            .get_mut(&camera_id)
            .ok_or(MultiCamError::AngleNotFound(camera_id as usize))?;

        state.push_drift(drift_us);

        let avg = state.averaged_drift();
        if avg.abs() <= self.config.tolerance_us {
            state.over_tolerance_count = 0;
            state.status = GenlockStatus::Locked;
        } else if state.over_tolerance_count >= self.config.lost_threshold {
            state.status = GenlockStatus::Lost;
        } else {
            state.over_tolerance_count += 1;
            state.status = GenlockStatus::Drifting;
        }
        Ok(state.status)
    }

    /// Get the current status of a camera.
    pub fn status(&self, camera_id: u32) -> Result<GenlockStatus> {
        self.cameras
            .get(&camera_id)
            .map(|s| s.status)
            .ok_or(MultiCamError::AngleNotFound(camera_id as usize))
    }

    /// Get the averaged drift for a camera in microseconds.
    pub fn averaged_drift_us(&self, camera_id: u32) -> Result<f64> {
        self.cameras
            .get(&camera_id)
            .map(|s| s.averaged_drift())
            .ok_or(MultiCamError::AngleNotFound(camera_id as usize))
    }

    /// Compute a correction recommendation for a camera.
    pub fn recommend_correction(&self, camera_id: u32) -> Result<CorrectionRecommendation> {
        let state = self
            .cameras
            .get(&camera_id)
            .ok_or(MultiCamError::AngleNotFound(camera_id as usize))?;

        let drift = state.averaged_drift();
        let mode = self.config.correction_mode;
        let frame_interval = self.config.frame_interval_us();

        let (correction_value, description) = match mode {
            CorrectionMode::FrameDropDuplicate => {
                // How many frames worth of drift?
                let frames = if frame_interval > 0.0 {
                    drift / frame_interval
                } else {
                    0.0
                };
                let desc = if drift > 0.0 {
                    format!("Drop {:.1} frame(s) to compensate +{:.1}us drift", frames.abs(), drift)
                } else {
                    format!(
                        "Duplicate {:.1} frame(s) to compensate {:.1}us drift",
                        frames.abs(),
                        drift
                    )
                };
                (-frames, desc)
            }
            CorrectionMode::SpeedRamp => {
                // Adjust playback rate to consume the drift over the next second.
                let rate = if frame_interval > 0.0 {
                    1.0 - drift / 1_000_000.0
                } else {
                    1.0
                };
                let desc = format!(
                    "Apply playback rate {:.6}x to correct {:.1}us drift",
                    rate, drift
                );
                (rate, desc)
            }
            CorrectionMode::RetimestampOnly => {
                let desc = format!("Re-timestamp by {:.1}us", -drift);
                (-drift, desc)
            }
            CorrectionMode::ReportOnly => {
                let desc = format!("Drift is {:.1}us (report only, no correction)", drift);
                (0.0, desc)
            }
        };

        Ok(CorrectionRecommendation {
            camera_id,
            drift_us: drift,
            mode,
            correction_value,
            description,
        })
    }

    /// Get a snapshot of all camera statuses.
    #[must_use]
    pub fn all_statuses(&self) -> Vec<(u32, GenlockStatus, f64)> {
        let mut result: Vec<_> = self
            .cameras
            .iter()
            .map(|(&id, s)| (id, s.status, s.averaged_drift()))
            .collect();
        result.sort_by_key(|(id, _, _)| *id);
        result
    }

    /// Check if all cameras are locked.
    #[must_use]
    pub fn all_locked(&self) -> bool {
        !self.cameras.is_empty()
            && self
                .cameras
                .values()
                .all(|s| s.status == GenlockStatus::Locked)
    }

    /// Reset the drift history for a camera (e.g. after manual re-sync).
    pub fn reset_camera(&mut self, camera_id: u32) -> Result<()> {
        let state = self
            .cameras
            .get_mut(&camera_id)
            .ok_or(MultiCamError::AngleNotFound(camera_id as usize))?;
        state.drift_history.fill(0.0);
        state.history_pos = 0;
        state.history_full = false;
        state.over_tolerance_count = 0;
        state.status = GenlockStatus::Locked;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let cfg = GenlockConfig::default();
        assert_eq!(cfg.reference_fps, 25.0);
        assert_eq!(cfg.tolerance_us, 500.0);
        assert_eq!(cfg.correction_mode, CorrectionMode::SpeedRamp);
        assert_eq!(cfg.averaging_window, 16);
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = GenlockConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_config_validate_bad_fps() {
        let mut cfg = GenlockConfig::default();
        cfg.reference_fps = 0.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_config_validate_bad_tolerance() {
        let mut cfg = GenlockConfig::default();
        cfg.tolerance_us = -1.0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_frame_interval_us() {
        let cfg = GenlockConfig::default(); // 25fps
        let interval = cfg.frame_interval_us();
        assert!((interval - 40_000.0).abs() < 1.0);
    }

    #[test]
    fn test_add_camera_and_status() {
        let mut mon = GenlockMonitor::with_defaults().expect("default config");
        let id = mon.add_camera("Cam1");
        assert_eq!(mon.camera_count(), 1);
        assert_eq!(mon.status(id).expect("status"), GenlockStatus::Locked);
    }

    #[test]
    fn test_remove_camera() {
        let mut mon = GenlockMonitor::with_defaults().expect("default config");
        let id = mon.add_camera("Cam1");
        assert!(mon.remove_camera(id));
        assert!(!mon.remove_camera(id)); // already removed
        assert_eq!(mon.camera_count(), 0);
    }

    #[test]
    fn test_locked_within_tolerance() {
        let mut mon = GenlockMonitor::with_defaults().expect("default config");
        let id = mon.add_camera("Cam1");
        let status = mon.report_drift(id, 100.0).expect("report");
        assert_eq!(status, GenlockStatus::Locked);
    }

    #[test]
    fn test_drifting_exceeds_tolerance() {
        let mut cfg = GenlockConfig::default();
        cfg.tolerance_us = 50.0;
        cfg.lost_threshold = 5;
        let mut mon = GenlockMonitor::new(cfg).expect("config");
        let id = mon.add_camera("Cam1");
        // Report drift exceeding tolerance
        let status = mon.report_drift(id, 200.0).expect("report");
        assert_eq!(status, GenlockStatus::Drifting);
    }

    #[test]
    fn test_lost_after_threshold() {
        let mut cfg = GenlockConfig::default();
        cfg.tolerance_us = 10.0;
        cfg.lost_threshold = 3;
        cfg.averaging_window = 1;
        let mut mon = GenlockMonitor::new(cfg).expect("config");
        let id = mon.add_camera("Cam1");

        // Push over-tolerance samples until lost
        for _ in 0..4 {
            let _ = mon.report_drift(id, 1000.0).expect("report");
        }
        assert_eq!(mon.status(id).expect("status"), GenlockStatus::Lost);
    }

    #[test]
    fn test_averaged_drift() {
        let mut cfg = GenlockConfig::default();
        cfg.averaging_window = 4;
        cfg.tolerance_us = 10000.0; // keep locked for this test
        let mut mon = GenlockMonitor::new(cfg).expect("config");
        let id = mon.add_camera("Cam1");

        let _ = mon.report_drift(id, 100.0);
        let _ = mon.report_drift(id, 200.0);
        let _ = mon.report_drift(id, 300.0);
        let avg = mon.averaged_drift_us(id).expect("avg");
        // (100+200+300)/3 = 200
        assert!((avg - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_recommend_correction_speed_ramp() {
        let mut cfg = GenlockConfig::default();
        cfg.correction_mode = CorrectionMode::SpeedRamp;
        cfg.tolerance_us = 100000.0; // keep locked
        cfg.averaging_window = 1;
        let mut mon = GenlockMonitor::new(cfg).expect("config");
        let id = mon.add_camera("Cam1");
        let _ = mon.report_drift(id, 500.0);

        let rec = mon.recommend_correction(id).expect("rec");
        assert_eq!(rec.mode, CorrectionMode::SpeedRamp);
        // rate = 1.0 - 500/1_000_000 = 0.9995
        assert!((rec.correction_value - 0.9995).abs() < 1e-6);
    }

    #[test]
    fn test_recommend_correction_frame_drop() {
        let mut cfg = GenlockConfig::default();
        cfg.reference_fps = 25.0;
        cfg.correction_mode = CorrectionMode::FrameDropDuplicate;
        cfg.tolerance_us = 100000.0;
        cfg.averaging_window = 1;
        let mut mon = GenlockMonitor::new(cfg).expect("config");
        let id = mon.add_camera("Cam1");
        // 40000us = 1 frame at 25fps
        let _ = mon.report_drift(id, 40000.0);

        let rec = mon.recommend_correction(id).expect("rec");
        assert_eq!(rec.mode, CorrectionMode::FrameDropDuplicate);
        // -frames: drift/interval = 40000/40000 = 1.0 => correction_value = -1.0
        assert!((rec.correction_value - (-1.0)).abs() < 0.01);
    }

    #[test]
    fn test_all_locked() {
        let mut mon = GenlockMonitor::with_defaults().expect("config");
        let a = mon.add_camera("A");
        let b = mon.add_camera("B");
        assert!(mon.all_locked()); // initial state is Locked

        let _ = mon.report_drift(a, 0.0);
        let _ = mon.report_drift(b, 0.0);
        assert!(mon.all_locked());
    }

    #[test]
    fn test_reset_camera() {
        let mut cfg = GenlockConfig::default();
        cfg.tolerance_us = 10.0;
        cfg.averaging_window = 1;
        let mut mon = GenlockMonitor::new(cfg).expect("config");
        let id = mon.add_camera("Cam1");
        let _ = mon.report_drift(id, 5000.0);
        assert_eq!(mon.status(id).expect("s"), GenlockStatus::Drifting);

        mon.reset_camera(id).expect("reset");
        assert_eq!(mon.status(id).expect("s"), GenlockStatus::Locked);
        assert!((mon.averaged_drift_us(id).expect("avg")).abs() < 0.01);
    }

    #[test]
    fn test_report_drift_unknown_camera() {
        let mut mon = GenlockMonitor::with_defaults().expect("config");
        assert!(mon.report_drift(999, 100.0).is_err());
    }

    #[test]
    fn test_correction_mode_display() {
        assert_eq!(format!("{}", CorrectionMode::SpeedRamp), "SpeedRamp");
        assert_eq!(
            format!("{}", CorrectionMode::FrameDropDuplicate),
            "FrameDropDuplicate"
        );
    }

    #[test]
    fn test_genlock_status_display() {
        assert_eq!(format!("{}", GenlockStatus::Locked), "LOCKED");
        assert_eq!(format!("{}", GenlockStatus::Drifting), "DRIFTING");
        assert_eq!(format!("{}", GenlockStatus::Lost), "LOST");
    }

    #[test]
    fn test_all_statuses() {
        let mut mon = GenlockMonitor::with_defaults().expect("config");
        mon.add_camera("A");
        mon.add_camera("B");
        let statuses = mon.all_statuses();
        assert_eq!(statuses.len(), 2);
        assert_eq!(statuses[0].0, 0);
        assert_eq!(statuses[1].0, 1);
    }
}
