//! Advanced shot detection and classification engine for `OxiMedia`.
//!
//! `oximedia-shots` provides comprehensive shot detection, classification, and analysis
//! capabilities for the `OxiMedia` multimedia framework. This includes:
//!
//! - **Shot Detection**: Hard cuts, dissolves, fades, wipes
//! - **Shot Classification**: Shot types (ECU, CU, MCU, MS, MLS, LS, ELS)
//! - **Camera Angle**: High, eye-level, low, bird's eye, Dutch
//! - **Camera Movement**: Pan, tilt, zoom, dolly, handheld detection
//! - **Composition Analysis**: Rule of thirds, symmetry, balance, leading lines, depth
//! - **Scene Detection**: Automatic scene boundary detection
//! - **Coverage Analysis**: Master, single, two-shot, over-the-shoulder
//! - **Continuity Checking**: Jump cuts, crossing the line, screen direction
//! - **Pattern Analysis**: Shot-reverse-shot, montage, editing rhythm
//! - **Export**: Shot lists (CSV, JSON), EDL with metadata
//!
//! # Shot Types
//!
//! The following shot types are classified:
//!
//! - **ECU (Extreme Close-up)**: Face details, eyes, lips
//! - **CU (Close-up)**: Head and shoulders
//! - **MCU (Medium Close-up)**: Waist up
//! - **MS (Medium Shot)**: Knees up
//! - **MLS (Medium Long Shot)**: Full body with space
//! - **LS (Long Shot)**: Full body in environment
//! - **ELS (Extreme Long Shot)**: Establishing shot
//!
//! # Camera Movements
//!
//! The following camera movements are detected:
//!
//! - **Pan**: Left/right horizontal movement
//! - **Tilt**: Up/down vertical movement
//! - **Zoom**: Lens zoom in/out
//! - **Dolly**: Camera movement toward/away from subject
//! - **Track**: Lateral camera movement
//! - **Handheld**: Unstabilized camera shake
//!
//! # Example
//!
//! ```
//! use oximedia_shots::{ShotDetector, ShotDetectorConfig};
//! use oximedia_shots::types::ShotType;
//!
//! // Create a shot detector with default configuration
//! let config = ShotDetectorConfig::default();
//! let detector = ShotDetector::new(config);
//!
//! // Shot detection would typically happen here with actual video frames
//! // let shots = detector.detect_shots(&frames)?;
//! ```

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::many_single_char_names)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::unused_self)]
#![allow(clippy::items_after_statements)]
#![allow(clippy::manual_memcpy)]
#![allow(clippy::unnecessary_wraps)]
#![allow(clippy::trivially_copy_pass_by_ref)]
#![allow(clippy::single_match_else)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::manual_swap)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::struct_excessive_bools)]
#![allow(clippy::too_many_arguments)]
#![allow(dead_code)]

use rayon::prelude::*;

pub mod adaptive_threshold;
pub mod analysis;
pub mod audio_scene_boundary;
pub mod boundary;
pub mod camera;
pub mod camera_movement;
pub mod classification;
pub mod classify;
pub mod color_continuity;
pub mod color_histogram;
pub mod composition;
pub mod confidence_calibration;
pub mod continuity;
pub mod coverage;
pub mod coverage_map;
pub mod depth_of_field;
pub mod detect;
pub mod detector;
pub mod duration;
pub mod error;
pub mod export;
pub mod flash_detection;
pub mod frame_buffer;
pub mod framing;
pub mod framing_guide;
pub mod golden_ratio;
pub mod insert_cutaway;
pub mod log;
pub mod metrics;
pub mod ml_boundary;
pub mod pacing;
pub mod pattern;
pub mod rating;
pub mod realtime;
pub mod scene;
pub mod scene_graph;
pub mod shot_annotation;
pub mod shot_classifier;
pub mod shot_density;
pub mod shot_grouping;
pub mod shot_matching;
pub mod shot_metadata;
pub mod shot_palette;
pub mod shot_report;
pub mod shot_rhythm;
pub mod shot_similarity;
pub mod shot_stats;
pub mod shot_tempo;
pub mod shot_transition;
pub mod shot_type;
pub mod storyboard;
pub mod temporal_fingerprint;
pub mod transition_analysis;
pub mod types;
pub mod visualize;
pub mod wipe_direction;
pub mod wipe_patterns;

// Re-export commonly used items at crate root
pub use error::{ShotError, ShotResult};
pub use frame_buffer::{FloatImage, FrameBuffer, GrayImage};
pub use types::{
    CameraAngle, CameraMovement, CompositionAnalysis, CoverageType, MovementType, Scene, Shot,
    ShotStatistics, ShotType, TransitionType,
};

/// Shot detector configuration.
#[derive(Debug, Clone)]
pub struct ShotDetectorConfig {
    /// Enable cut detection.
    pub enable_cut_detection: bool,
    /// Enable dissolve detection.
    pub enable_dissolve_detection: bool,
    /// Enable fade detection.
    pub enable_fade_detection: bool,
    /// Enable wipe detection.
    pub enable_wipe_detection: bool,
    /// Enable shot classification.
    pub enable_classification: bool,
    /// Enable camera movement detection.
    pub enable_movement_detection: bool,
    /// Enable composition analysis.
    pub enable_composition_analysis: bool,
    /// Cut detection threshold.
    pub cut_threshold: f32,
    /// Dissolve detection threshold.
    pub dissolve_threshold: f32,
    /// Fade detection threshold.
    pub fade_threshold: f32,
}

impl Default for ShotDetectorConfig {
    fn default() -> Self {
        Self {
            enable_cut_detection: true,
            enable_dissolve_detection: true,
            enable_fade_detection: true,
            enable_wipe_detection: true,
            enable_classification: true,
            enable_movement_detection: true,
            enable_composition_analysis: true,
            cut_threshold: 0.3,
            dissolve_threshold: 0.15,
            fade_threshold: 0.2,
        }
    }
}

/// Main shot detector.
pub struct ShotDetector {
    config: ShotDetectorConfig,
    cut_detector: detect::CutDetector,
    dissolve_detector: detect::DissolveDetector,
    fade_detector: detect::FadeDetector,
    wipe_detector: detect::WipeDetector,
    shot_classifier: classify::ShotTypeClassifier,
    angle_classifier: classify::AngleClassifier,
    composition_analyzer: classify::CompositionAnalyzer,
    movement_detector: camera::MovementDetector,
}

impl ShotDetector {
    /// Create a new shot detector with the given configuration.
    #[must_use]
    pub fn new(config: ShotDetectorConfig) -> Self {
        Self {
            cut_detector: detect::CutDetector::with_params(config.cut_threshold, 0.4, 5),
            dissolve_detector: detect::DissolveDetector::new(),
            fade_detector: detect::FadeDetector::new(),
            wipe_detector: detect::WipeDetector::new(),
            shot_classifier: classify::ShotTypeClassifier::new(),
            angle_classifier: classify::AngleClassifier::new(),
            composition_analyzer: classify::CompositionAnalyzer::new(),
            movement_detector: camera::MovementDetector::new(),
            config,
        }
    }

    /// Detect shots in a sequence of frames.
    ///
    /// The boundary detection phase runs in parallel using rayon: adjacent frame
    /// pairs are evaluated concurrently, and the results are collected in order.
    /// Classification and composition analysis per detected shot are then applied
    /// sequentially because those classifiers hold interior-mutability caches that
    /// are not `Sync`.
    ///
    /// # Errors
    ///
    /// Returns error if frame processing fails.
    pub fn detect_shots(&self, frames: &[FrameBuffer]) -> ShotResult<Vec<Shot>> {
        if frames.is_empty() {
            return Ok(Vec::new());
        }

        let mut shot_boundaries = vec![0usize]; // Start of first shot

        // ── Parallel boundary detection ────────────────────────────────────────
        // Each adjacent pair (i-1, i) is independent, so we can evaluate all
        // pairs concurrently.  We collect a Vec<bool> (is_cut at position i)
        // then filter the indices.
        //
        // Note: `CutDetector` has a `RefCell` inside and is therefore not `Sync`.
        // We work around this by producing a *fresh* `CutDetector` per rayon
        // task using the current (non-adaptive) threshold configuration.
        if self.config.enable_cut_detection && frames.len() > 1 {
            let cut_threshold = self.config.cut_threshold;
            let is_cut_flags: Result<Vec<bool>, ShotError> = (1..frames.len())
                .into_par_iter()
                .map(|i| {
                    // Create a thread-local detector with the same thresholds.
                    let detector = detect::CutDetector::with_params(cut_threshold, 0.4, 5);
                    let (is_cut, _score) = detector.detect_cut(&frames[i - 1], &frames[i])?;
                    Ok(is_cut)
                })
                .collect();

            let flags = is_cut_flags?;
            for (idx, &cut) in flags.iter().enumerate() {
                if cut {
                    shot_boundaries.push(idx + 1);
                }
            }
        }

        shot_boundaries.push(frames.len()); // End of last shot

        // ── Sequential classification and composition analysis ─────────────────
        let mut shots = Vec::with_capacity(shot_boundaries.len().saturating_sub(1));

        for i in 0..shot_boundaries.len().saturating_sub(1) {
            let start_frame = shot_boundaries[i];
            let end_frame = shot_boundaries[i + 1];

            let mut shot = Shot::new(
                i as u64,
                oximedia_core::types::Timestamp::new(
                    start_frame as i64,
                    oximedia_core::types::Rational::new(1, 30),
                ),
                oximedia_core::types::Timestamp::new(
                    end_frame as i64,
                    oximedia_core::types::Rational::new(1, 30),
                ),
            );

            // Classify shot type
            if self.config.enable_classification && start_frame < frames.len() {
                let (shot_type, confidence) =
                    self.shot_classifier.classify(&frames[start_frame])?;
                shot.shot_type = shot_type;
                shot.confidence = confidence;

                // Classify camera angle
                let (angle, _) = self.angle_classifier.classify(&frames[start_frame])?;
                shot.angle = angle;
            }

            // Analyze composition
            if self.config.enable_composition_analysis && start_frame < frames.len() {
                shot.composition = self.composition_analyzer.analyze(&frames[start_frame])?;
            }

            // Detect camera movements
            if self.config.enable_movement_detection {
                let shot_frames = &frames[start_frame..end_frame.min(frames.len())];
                if !shot_frames.is_empty() {
                    shot.movements = self.movement_detector.detect_movements(shot_frames)?;
                }
            }

            shots.push(shot);
        }

        Ok(shots)
    }

    /// Analyze shots and generate statistics.
    #[must_use]
    pub fn analyze_shots(&self, shots: &[Shot]) -> ShotStatistics {
        let analyzer = duration::DurationAnalyzer::new();
        analyzer.analyze(shots)
    }

    /// Detect scenes from shots.
    #[must_use]
    pub fn detect_scenes(&self, shots: &[Shot]) -> Vec<Scene> {
        let detector = scene::SceneDetector::new();
        detector.detect_scenes(shots)
    }

    /// Check continuity in shots.
    #[must_use]
    pub fn check_continuity(&self, shots: &[Shot]) -> Vec<continuity::ContinuityIssue> {
        let checker = continuity::ContinuityChecker::new();
        checker.check_continuity(shots)
    }

    /// Analyze editing patterns.
    #[must_use]
    pub fn analyze_patterns(&self, shots: &[Shot]) -> pattern::PatternAnalysis {
        let analyzer = pattern::PatternAnalyzer::new();
        analyzer.analyze(shots)
    }

    /// Analyze editing rhythm.
    #[must_use]
    pub fn analyze_rhythm(&self, shots: &[Shot]) -> pattern::RhythmAnalysis {
        let analyzer = pattern::RhythmAnalyzer::new();
        analyzer.analyze(shots)
    }
}

impl Default for ShotDetector {
    fn default() -> Self {
        Self::new(ShotDetectorConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_default() {
        let config = ShotDetectorConfig::default();
        assert!(config.enable_cut_detection);
        assert!(config.enable_classification);
    }

    #[test]
    fn test_detector_creation() {
        let config = ShotDetectorConfig::default();
        let _detector = ShotDetector::new(config);
    }

    #[test]
    fn test_detect_shots_empty() {
        let detector = ShotDetector::default();
        let result = detector.detect_shots(&[]);
        assert!(result.is_ok());
        if let Ok(shots) = result {
            assert!(shots.is_empty());
        }
    }

    #[test]
    fn test_detect_shots_single_frame() {
        let detector = ShotDetector::default();
        let frames = vec![FrameBuffer::zeros(100, 100, 3)];
        let result = detector.detect_shots(&frames);
        assert!(result.is_ok());
        if let Ok(shots) = result {
            assert_eq!(shots.len(), 1);
        }
    }

    #[test]
    fn test_analyze_shots_empty() {
        let detector = ShotDetector::default();
        let stats = detector.analyze_shots(&[]);
        assert_eq!(stats.total_shots, 0);
    }

    #[test]
    fn test_detect_scenes_empty() {
        let detector = ShotDetector::default();
        let scenes = detector.detect_scenes(&[]);
        assert!(scenes.is_empty());
    }

    #[test]
    fn test_check_continuity_empty() {
        let detector = ShotDetector::default();
        let issues = detector.check_continuity(&[]);
        assert!(issues.is_empty());
    }
}
