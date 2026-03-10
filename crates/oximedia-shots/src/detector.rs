//! High-level shot detector with batch processing capabilities.

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::FrameBuffer;
use crate::types::{Scene, Shot, ShotStatistics};
use crate::{ShotDetector, ShotDetectorConfig};
use rayon::prelude::*;

/// Batch shot detector for processing large video files.
pub struct BatchShotDetector {
    detector: ShotDetector,
    batch_size: usize,
    parallel: bool,
}

impl BatchShotDetector {
    /// Create a new batch shot detector.
    #[must_use]
    pub fn new(config: ShotDetectorConfig, batch_size: usize) -> Self {
        Self {
            detector: ShotDetector::new(config),
            batch_size,
            parallel: true,
        }
    }

    /// Enable or disable parallel processing.
    pub fn set_parallel(&mut self, parallel: bool) {
        self.parallel = parallel;
    }

    /// Process frames in batches and return all detected shots.
    ///
    /// # Errors
    ///
    /// Returns error if frame processing fails.
    pub fn process_batches(&self, frames: &[FrameBuffer]) -> ShotResult<Vec<Shot>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let chunks: Vec<_> = frames.chunks(self.batch_size).collect();

        let batch_results: Result<Vec<Vec<Shot>>, ShotError> = if self.parallel {
            chunks
                .par_iter()
                .enumerate()
                .map(|(batch_idx, chunk)| {
                    let mut batch_shots = self.detector.detect_shots(chunk)?;

                    // Adjust shot IDs and timestamps for batch offset
                    let frame_offset = batch_idx * self.batch_size;
                    for shot in &mut batch_shots {
                        shot.id += (batch_idx * 1000) as u64;
                        shot.start.pts += frame_offset as i64;
                        shot.end.pts += frame_offset as i64;
                    }

                    Ok(batch_shots)
                })
                .collect()
        } else {
            chunks
                .iter()
                .enumerate()
                .map(|(batch_idx, chunk)| {
                    let mut batch_shots = self.detector.detect_shots(chunk)?;

                    let frame_offset = batch_idx * self.batch_size;
                    for shot in &mut batch_shots {
                        shot.id += (batch_idx * 1000) as u64;
                        shot.start.pts += frame_offset as i64;
                        shot.end.pts += frame_offset as i64;
                    }

                    Ok(batch_shots)
                })
                .collect()
        };

        let shots: Vec<Shot> = batch_results?.into_iter().flatten().collect();

        Ok(shots)
    }

    /// Process and analyze shots, returning comprehensive results.
    ///
    /// # Errors
    ///
    /// Returns error if processing fails.
    pub fn process_and_analyze(&self, frames: &[FrameBuffer]) -> ShotResult<ProcessingResults> {
        let shots = self.process_batches(frames)?;
        let statistics = self.detector.analyze_shots(&shots);
        let scenes = self.detector.detect_scenes(&shots);
        let continuity_issues = self.detector.check_continuity(&shots);
        let patterns = self.detector.analyze_patterns(&shots);
        let rhythm = self.detector.analyze_rhythm(&shots);

        Ok(ProcessingResults {
            shots,
            statistics,
            scenes,
            continuity_issues,
            patterns,
            rhythm,
        })
    }
}

/// Complete processing results.
#[derive(Debug, Clone)]
pub struct ProcessingResults {
    /// Detected shots.
    pub shots: Vec<Shot>,
    /// Shot statistics.
    pub statistics: ShotStatistics,
    /// Detected scenes.
    pub scenes: Vec<Scene>,
    /// Continuity issues.
    pub continuity_issues: Vec<crate::continuity::ContinuityIssue>,
    /// Pattern analysis.
    pub patterns: crate::pattern::PatternAnalysis,
    /// Rhythm analysis.
    pub rhythm: crate::pattern::RhythmAnalysis,
}

impl ProcessingResults {
    /// Generate a comprehensive text report.
    #[must_use]
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("SHOT DETECTION REPORT\n");
        report.push_str("====================\n\n");

        report.push_str(&format!("Total Shots: {}\n", self.statistics.total_shots));
        report.push_str(&format!("Total Scenes: {}\n", self.statistics.total_scenes));
        report.push_str(&format!(
            "Average Shot Duration: {:.2}s\n",
            self.statistics.average_shot_duration
        ));
        report.push_str(&format!(
            "Median Shot Duration: {:.2}s\n",
            self.statistics.median_shot_duration
        ));
        report.push_str(&format!(
            "Min Shot Duration: {:.2}s\n",
            self.statistics.min_shot_duration
        ));
        report.push_str(&format!(
            "Max Shot Duration: {:.2}s\n\n",
            self.statistics.max_shot_duration
        ));

        report.push_str("EDITING RHYTHM\n");
        report.push_str("--------------\n");
        report.push_str(&format!("Beat: {:.2} cuts/second\n", self.rhythm.beat));
        report.push_str(&format!("Regularity: {:.2}\n", self.rhythm.regularity));
        report.push_str(&format!("Accelerations: {}\n", self.rhythm.accelerations));
        report.push_str(&format!("Decelerations: {}\n\n", self.rhythm.decelerations));

        report.push_str("PATTERNS\n");
        report.push_str("--------\n");
        report.push_str(&format!(
            "Shot-Reverse-Shot: {}\n",
            self.patterns.shot_reverse_shot_count
        ));
        report.push_str(&format!(
            "Montage Sequences: {}\n",
            self.patterns.montage_sequences
        ));
        report.push_str(&format!(
            "Coverage Pattern: {}\n\n",
            self.patterns.coverage_pattern
        ));

        report.push_str("CONTINUITY ISSUES\n");
        report.push_str("----------------\n");
        report.push_str(&format!("Total Issues: {}\n", self.continuity_issues.len()));

        for issue in &self.continuity_issues {
            report.push_str(&format!(
                "  - Shot {} [{:?}]: {}\n",
                issue.shot_id, issue.severity, issue.description
            ));
        }

        report
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_batch_detector_creation() {
        let config = ShotDetectorConfig::default();
        let _detector = BatchShotDetector::new(config, 100);
    }

    #[test]
    fn test_process_empty() {
        let config = ShotDetectorConfig::default();
        let detector = BatchShotDetector::new(config, 100);
        let result = detector.process_batches(&[]);
        assert!(result.is_ok());
        assert!(result.expect("should succeed in test").is_empty());
    }

    #[test]
    fn test_parallel_toggle() {
        let config = ShotDetectorConfig::default();
        let mut detector = BatchShotDetector::new(config, 100);
        detector.set_parallel(false);
        assert!(!detector.parallel);
        detector.set_parallel(true);
        assert!(detector.parallel);
    }
}
