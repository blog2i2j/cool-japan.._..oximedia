//! Technical video analysis.
//!
//! This module provides technical analysis of video signals including levels,
//! range checking, gamut violations, cadence detection, and sync monitoring.

pub mod levels;
pub mod range;
pub mod gamut;
pub mod cadence;
pub mod sync;

use crate::{MonitorError, MonitorResult};
use serde::{Deserialize, Serialize};

pub use levels::{LevelAnalyzer, LevelMetrics};
pub use range::{RangeChecker, RangeMetrics};
pub use gamut::{GamutChecker, GamutMetrics};
pub use cadence::{CadenceDetector, CadenceMetrics};
pub use sync::{SyncMonitor, SyncMetrics};

/// Video quality metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct VideoQualityMetrics {
    /// Black level (0.0-1.0).
    pub black_level: f32,

    /// White level (0.0-1.0).
    pub white_level: f32,

    /// Brightness (0.0-1.0).
    pub brightness: f32,

    /// Contrast ratio.
    pub contrast: f32,

    /// Color saturation (0.0-1.0).
    pub saturation: f32,

    /// Sharpness score (0.0-1.0).
    pub sharpness: f32,
}

/// Technical video analysis metrics.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TechnicalMetrics {
    /// Level analysis metrics.
    pub level_metrics: LevelMetrics,

    /// Range checking metrics.
    pub range_metrics: RangeMetrics,

    /// Gamut violation metrics.
    pub gamut_metrics: GamutMetrics,

    /// Cadence detection metrics.
    pub cadence_metrics: CadenceMetrics,

    /// Sync monitoring metrics.
    pub sync_metrics: SyncMetrics,

    /// Overall video quality metrics.
    pub quality_metrics: VideoQualityMetrics,
}

/// Technical analyzer combining all analysis features.
pub struct TechnicalAnalyzer {
    level_analyzer: LevelAnalyzer,
    range_checker: RangeChecker,
    gamut_checker: GamutChecker,
    cadence_detector: CadenceDetector,
    sync_monitor: SyncMonitor,
    metrics: TechnicalMetrics,
}

impl TechnicalAnalyzer {
    /// Create a new technical analyzer.
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails.
    pub fn new() -> MonitorResult<Self> {
        Ok(Self {
            level_analyzer: LevelAnalyzer::new(),
            range_checker: RangeChecker::new(),
            gamut_checker: GamutChecker::new(),
            cadence_detector: CadenceDetector::new(),
            sync_monitor: SyncMonitor::new(),
            metrics: TechnicalMetrics::default(),
        })
    }

    /// Analyze a video frame.
    ///
    /// # Errors
    ///
    /// Returns an error if analysis fails.
    pub fn analyze_frame(&mut self, frame: &[u8], width: u32, height: u32) -> MonitorResult<()> {
        self.level_analyzer.analyze(frame, width, height)?;
        self.range_checker.check(frame, width, height)?;
        self.gamut_checker.check(frame, width, height)?;
        self.cadence_detector.process_frame();
        self.sync_monitor.update();

        self.update_metrics();

        Ok(())
    }

    /// Get current metrics.
    #[must_use]
    pub const fn metrics(&self) -> &TechnicalMetrics {
        &self.metrics
    }

    /// Reset analyzer.
    pub fn reset(&mut self) {
        self.level_analyzer.reset();
        self.range_checker.reset();
        self.gamut_checker.reset();
        self.cadence_detector.reset();
        self.sync_monitor.reset();
        self.metrics = TechnicalMetrics::default();
    }

    fn update_metrics(&mut self) {
        self.metrics.level_metrics = self.level_analyzer.metrics().clone();
        self.metrics.range_metrics = self.range_checker.metrics().clone();
        self.metrics.gamut_metrics = self.gamut_checker.metrics().clone();
        self.metrics.cadence_metrics = self.cadence_detector.metrics().clone();
        self.metrics.sync_metrics = self.sync_monitor.metrics().clone();

        // Update quality metrics
        self.metrics.quality_metrics.black_level = self.metrics.level_metrics.black_level;
        self.metrics.quality_metrics.white_level = self.metrics.level_metrics.white_level;
        self.metrics.quality_metrics.brightness = self.metrics.level_metrics.avg_luma;
        self.metrics.quality_metrics.contrast = self.metrics.level_metrics.contrast_ratio;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_technical_analyzer() {
        let result = TechnicalAnalyzer::new();
        assert!(result.is_ok());
    }

    #[test]
    fn test_analyze_frame() {
        let mut analyzer = TechnicalAnalyzer::new().expect("failed to create");

        let frame = vec![128u8; 1920 * 1080 * 3];
        assert!(analyzer.analyze_frame(&frame, 1920, 1080).is_ok());

        let metrics = analyzer.metrics();
        assert!(metrics.quality_metrics.brightness > 0.0);
    }
}
