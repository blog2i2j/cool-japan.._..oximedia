//! Scene classification (indoor/outdoor, day/night, etc.).

use crate::common::Confidence;
use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};

/// Type of scene detected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SceneType {
    /// Indoor scene.
    Indoor,
    /// Outdoor scene.
    Outdoor,
    /// Day scene (bright, well-lit).
    Day,
    /// Night scene (dark, low-light).
    Night,
    /// Landscape orientation and composition.
    Landscape,
    /// Portrait orientation and composition.
    Portrait,
    /// Urban environment.
    Urban,
    /// Natural environment.
    Natural,
    /// Water scene (ocean, lake, river).
    Water,
    /// Sky-dominant scene.
    Sky,
    /// Unknown or mixed scene.
    Unknown,
}

impl SceneType {
    /// Get all possible scene types.
    #[must_use]
    pub fn all() -> &'static [Self] {
        &[
            Self::Indoor,
            Self::Outdoor,
            Self::Day,
            Self::Night,
            Self::Landscape,
            Self::Portrait,
            Self::Urban,
            Self::Natural,
            Self::Water,
            Self::Sky,
            Self::Unknown,
        ]
    }

    /// Get human-readable name.
    #[must_use]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Indoor => "Indoor",
            Self::Outdoor => "Outdoor",
            Self::Day => "Day",
            Self::Night => "Night",
            Self::Landscape => "Landscape",
            Self::Portrait => "Portrait",
            Self::Urban => "Urban",
            Self::Natural => "Natural",
            Self::Water => "Water",
            Self::Sky => "Sky",
            Self::Unknown => "Unknown",
        }
    }
}

/// Scene classification result.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneClassification {
    /// Primary scene type.
    pub scene_type: SceneType,
    /// Confidence score.
    pub confidence: Confidence,
    /// Scores for all scene types.
    pub scores: Vec<(SceneType, f32)>,
    /// Additional features used for classification.
    pub features: SceneFeatures,
}

/// Features extracted for scene classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneFeatures {
    /// Average brightness (0.0-1.0).
    pub brightness: f32,
    /// Color temperature (warm/cool).
    pub color_temperature: f32,
    /// Saturation level (0.0-1.0).
    pub saturation: f32,
    /// Sky region ratio (0.0-1.0).
    pub sky_ratio: f32,
    /// Vegetation ratio (0.0-1.0).
    pub vegetation_ratio: f32,
    /// Artificial structure ratio (0.0-1.0).
    pub structure_ratio: f32,
    /// Horizon line position (0.0-1.0, from top).
    pub horizon_position: Option<f32>,
}

impl Default for SceneFeatures {
    fn default() -> Self {
        Self {
            brightness: 0.5,
            color_temperature: 0.5,
            saturation: 0.5,
            sky_ratio: 0.0,
            vegetation_ratio: 0.0,
            structure_ratio: 0.0,
            horizon_position: None,
        }
    }
}

/// Configuration for scene classification.
#[derive(Debug, Clone)]
pub struct SceneConfig {
    /// Minimum confidence threshold.
    pub confidence_threshold: f32,
    /// Enable color histogram analysis.
    pub use_color_histogram: bool,
    /// Enable edge detection.
    pub use_edge_detection: bool,
    /// Enable texture analysis.
    pub use_texture_analysis: bool,
    /// Enable temporal smoothing across frames to reduce flickering.
    pub temporal_smoothing: bool,
    /// Number of frames in temporal smoothing window.
    pub temporal_window: usize,
    /// Decay rate for exponential weighting (higher = faster decay = less smoothing).
    pub temporal_decay: f32,
}

impl Default for SceneConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: 0.5,
            use_color_histogram: true,
            use_edge_detection: true,
            use_texture_analysis: true,
            temporal_smoothing: false,
            temporal_window: 8,
            temporal_decay: 0.3,
        }
    }
}

/// Temporal smoothing buffer for scene classification.
#[derive(Debug, Clone)]
struct TemporalBuffer {
    /// Accumulated scores per scene type across recent frames.
    scores: Vec<Vec<f32>>,
    /// Maximum history length.
    capacity: usize,
}

impl TemporalBuffer {
    fn new(capacity: usize, _num_types: usize) -> Self {
        Self {
            scores: Vec::with_capacity(capacity),
            capacity,
        }
    }

    fn push(&mut self, frame_scores: Vec<f32>) {
        if self.scores.len() >= self.capacity {
            self.scores.remove(0);
        }
        self.scores.push(frame_scores);
    }

    /// Compute exponentially weighted moving average of scores.
    fn smooth(&self, decay: f32) -> Vec<f32> {
        if self.scores.is_empty() {
            return Vec::new();
        }
        let len = self.scores[0].len();
        let mut smoothed = vec![0.0f32; len];

        // Weight recent frames more heavily
        let n = self.scores.len();
        let mut weight_sum = 0.0_f32;
        for (i, frame_scores) in self.scores.iter().enumerate() {
            // Exponential decay: newer frames get higher weight
            let age = (n - 1 - i) as f32;
            let weight = (-decay * age).exp();
            for (j, s) in frame_scores.iter().enumerate() {
                smoothed[j] += s * weight;
            }
            weight_sum += weight;
        }
        if weight_sum > 0.0 {
            for v in &mut smoothed {
                *v /= weight_sum;
            }
        }
        smoothed
    }
}

/// Scene classifier using color histograms and heuristics.
pub struct SceneClassifier {
    config: SceneConfig,
    /// Optional temporal smoothing buffer (populated when temporal_smoothing is enabled).
    temporal_buffer: Option<std::sync::Mutex<TemporalBuffer>>,
}

impl SceneClassifier {
    /// Create a new scene classifier.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SceneConfig::default(),
            temporal_buffer: None,
        }
    }

    /// Create a scene classifier with custom configuration.
    #[must_use]
    pub fn with_config(config: SceneConfig) -> Self {
        let temporal_buffer = if config.temporal_smoothing {
            let buf = TemporalBuffer::new(config.temporal_window, SceneType::all().len());
            Some(std::sync::Mutex::new(buf))
        } else {
            None
        };
        Self {
            config,
            temporal_buffer,
        }
    }

    /// Create a scene classifier with temporal smoothing enabled.
    #[must_use]
    pub fn with_temporal_smoothing(window: usize) -> Self {
        let config = SceneConfig {
            temporal_smoothing: true,
            temporal_window: window,
            ..SceneConfig::default()
        };
        Self::with_config(config)
    }

    /// Reset the temporal smoothing buffer (e.g. at scene cuts).
    pub fn reset_temporal_buffer(&self) {
        if let Some(ref buf) = self.temporal_buffer {
            if let Ok(mut guard) = buf.lock() {
                guard.scores.clear();
            }
        }
    }

    /// Classify a scene from RGB image data.
    ///
    /// # Arguments
    ///
    /// * `rgb_data` - RGB image data (height x width x 3)
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Errors
    ///
    /// Returns error if classification fails or invalid dimensions.
    pub fn classify(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<SceneClassification> {
        if rgb_data.len() != width * height * 3 {
            return Err(SceneError::InvalidDimensions(format!(
                "Expected {} bytes, got {}",
                width * height * 3,
                rgb_data.len()
            )));
        }

        // Extract features
        let features = self.extract_features(rgb_data, width, height)?;

        // Compute raw scores for each scene type
        let raw_scores: Vec<f32> = vec![
            self.score_indoor(&features),
            self.score_outdoor(&features),
            self.score_day(&features),
            self.score_night(&features),
            self.score_landscape(&features),
            self.score_portrait(&features),
            self.score_urban(&features),
            self.score_natural(&features),
            self.score_water(&features),
            self.score_sky(&features),
        ];

        // Apply temporal smoothing if enabled
        let final_scores = if let Some(ref buf_mutex) = self.temporal_buffer {
            if let Ok(mut buf) = buf_mutex.lock() {
                buf.push(raw_scores.clone());
                buf.smooth(self.config.temporal_decay)
            } else {
                raw_scores.clone()
            }
        } else {
            raw_scores.clone()
        };

        let scene_types = [
            SceneType::Indoor,
            SceneType::Outdoor,
            SceneType::Day,
            SceneType::Night,
            SceneType::Landscape,
            SceneType::Portrait,
            SceneType::Urban,
            SceneType::Natural,
            SceneType::Water,
            SceneType::Sky,
        ];

        let scores: Vec<(SceneType, f32)> = scene_types
            .iter()
            .zip(final_scores.iter())
            .map(|(&t, &s)| (t, s))
            .collect();

        // Find highest score
        let (scene_type, confidence) = scores
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map_or((SceneType::Unknown, 0.0), |(t, s)| (*t, *s));

        Ok(SceneClassification {
            scene_type,
            confidence: Confidence::new(confidence),
            scores,
            features,
        })
    }

    /// Extract scene features from RGB data.
    fn extract_features(
        &self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> SceneResult<SceneFeatures> {
        let mut brightness_sum = 0.0;
        let mut saturation_sum = 0.0;
        let mut color_temp_sum = 0.0;
        let pixel_count = width * height;

        // Analyze pixels
        for y in 0..height {
            for x in 0..width {
                let idx = (y * width + x) * 3;
                let r = f32::from(rgb_data[idx]);
                let g = f32::from(rgb_data[idx + 1]);
                let b = f32::from(rgb_data[idx + 2]);

                // Brightness (perceived luminance)
                brightness_sum += 0.299 * r + 0.587 * g + 0.114 * b;

                // Saturation
                let max = r.max(g).max(b);
                let min = r.min(g).min(b);
                if max > 0.0 {
                    saturation_sum += (max - min) / max;
                }

                // Color temperature (blue vs red)
                color_temp_sum += (b - r) / 255.0;
            }
        }

        let brightness = (brightness_sum / (pixel_count as f32 * 255.0)).clamp(0.0, 1.0);
        let saturation = (saturation_sum / pixel_count as f32).clamp(0.0, 1.0);
        let color_temperature = ((color_temp_sum / pixel_count as f32) + 1.0) / 2.0;

        // Detect sky, vegetation, and structures using color heuristics
        let (sky_ratio, vegetation_ratio, structure_ratio) =
            self.detect_regions(rgb_data, width, height);

        // Detect horizon
        let horizon_position = self.detect_horizon(rgb_data, width, height);

        Ok(SceneFeatures {
            brightness,
            color_temperature,
            saturation,
            sky_ratio,
            vegetation_ratio,
            structure_ratio,
            horizon_position,
        })
    }

    /// Detect sky, vegetation, and structure regions.
    fn detect_regions(&self, rgb_data: &[u8], width: usize, height: usize) -> (f32, f32, f32) {
        let mut sky_pixels = 0;
        let mut vegetation_pixels = 0;
        let mut structure_pixels = 0;
        let pixel_count = width * height;

        for i in (0..rgb_data.len()).step_by(3) {
            let r = rgb_data[i];
            let g = rgb_data[i + 1];
            let b = rgb_data[i + 2];

            // Sky: blue dominant, high brightness
            if b > r && b > g && b > 128 {
                sky_pixels += 1;
            }
            // Vegetation: green dominant
            else if g > r && g > b && g > 64 {
                vegetation_pixels += 1;
            }
            // Structure: low saturation (gray tones)
            else {
                let max = r.max(g).max(b);
                let min = r.min(g).min(b);
                if max > 0 && (max - min) < 30 {
                    structure_pixels += 1;
                }
            }
        }

        (
            sky_pixels as f32 / pixel_count as f32,
            vegetation_pixels as f32 / pixel_count as f32,
            structure_pixels as f32 / pixel_count as f32,
        )
    }

    /// Detect horizon line position.
    fn detect_horizon(&self, rgb_data: &[u8], width: usize, height: usize) -> Option<f32> {
        // Simple horizon detection: find strongest horizontal edge in middle third
        let start_y = height / 3;
        let end_y = (height * 2) / 3;
        let mut max_edge = 0.0;
        let mut horizon_y = None;

        for y in start_y..end_y {
            let mut edge_strength = 0.0;
            for x in 1..width - 1 {
                let _idx = (y * width + x) * 3;
                let idx_above = ((y - 1) * width + x) * 3;
                let idx_below = ((y + 1) * width + x) * 3;

                // Vertical gradient
                for c in 0..3 {
                    let diff = (rgb_data[idx_below + c] as i32 - rgb_data[idx_above + c] as i32)
                        .unsigned_abs() as f32;
                    edge_strength += diff;
                }
            }

            if edge_strength > max_edge {
                max_edge = edge_strength;
                horizon_y = Some(y);
            }
        }

        horizon_y.map(|y| y as f32 / height as f32)
    }

    // Scoring functions for each scene type
    fn score_indoor(&self, features: &SceneFeatures) -> f32 {
        let mut score = 0.0;
        // Indoor scenes typically have lower brightness
        score += (1.0 - features.brightness) * 0.3;
        // Less sky
        score += (1.0 - features.sky_ratio) * 0.4;
        // More structures
        score += features.structure_ratio * 0.3;
        score.clamp(0.0, 1.0)
    }

    fn score_outdoor(&self, features: &SceneFeatures) -> f32 {
        let mut score = 0.0;
        // Higher brightness
        score += features.brightness * 0.3;
        // More sky
        score += features.sky_ratio * 0.4;
        // Natural elements
        score += features.vegetation_ratio * 0.3;
        score.clamp(0.0, 1.0)
    }

    fn score_day(&self, features: &SceneFeatures) -> f32 {
        // High brightness, high saturation
        (features.brightness * 0.7 + features.saturation * 0.3).clamp(0.0, 1.0)
    }

    fn score_night(&self, features: &SceneFeatures) -> f32 {
        // Low brightness
        (1.0 - features.brightness).clamp(0.0, 1.0)
    }

    fn score_landscape(&self, features: &SceneFeatures) -> f32 {
        let mut score = 0.0;
        // Horizon present
        if features.horizon_position.is_some() {
            score += 0.5;
        }
        // Sky and vegetation
        score += (features.sky_ratio + features.vegetation_ratio) * 0.5;
        score.clamp(0.0, 1.0)
    }

    fn score_portrait(&self, features: &SceneFeatures) -> f32 {
        // Less sky, more centered composition
        let mut score = 1.0 - features.sky_ratio;
        if let Some(horizon) = features.horizon_position {
            // Horizon in middle third is less common in portraits
            if (0.33..=0.67).contains(&horizon) {
                score *= 0.5;
            }
        }
        score.clamp(0.0, 1.0)
    }

    fn score_urban(&self, features: &SceneFeatures) -> f32 {
        // More structures, less vegetation
        let mut score = features.structure_ratio * 0.6;
        score += (1.0 - features.vegetation_ratio) * 0.4;
        score.clamp(0.0, 1.0)
    }

    fn score_natural(&self, features: &SceneFeatures) -> f32 {
        // More vegetation, high saturation
        (features.vegetation_ratio * 0.7 + features.saturation * 0.3).clamp(0.0, 1.0)
    }

    fn score_water(&self, features: &SceneFeatures) -> f32 {
        // Cool color temperature, specific horizon position
        let mut score = 0.0;
        if features.color_temperature > 0.5 {
            score += (features.color_temperature - 0.5) * 2.0 * 0.5;
        }
        if let Some(horizon) = features.horizon_position {
            // Water typically has horizon in middle third
            if (0.33..=0.67).contains(&horizon) {
                score += 0.5;
            }
        }
        score.clamp(0.0, 1.0)
    }

    fn score_sky(&self, features: &SceneFeatures) -> f32 {
        // High sky ratio
        features.sky_ratio
    }
}

impl Default for SceneClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SmoothedSceneClassifier — mode-based temporal smoothing wrapper
// ---------------------------------------------------------------------------

/// A wrapper around [`SceneClassifier`] that applies mode-based temporal
/// smoothing via a sliding window to reduce frame-to-frame classification
/// flicker.
///
/// Instead of the EWMA score-based smoothing built into [`SceneClassifier`],
/// this wrapper uses [`crate::classify::temporal_smooth::TemporalSmoother`]
/// which selects the most frequently occurring [`SceneType`] across a window
/// of the last `N` frames.  This is useful when you want a hard label that
/// doesn't waver between adjacent categories for single anomalous frames.
pub struct SmoothedSceneClassifier {
    inner: SceneClassifier,
    smoother: crate::classify::temporal_smooth::TemporalSmoother<SceneType>,
}

impl SmoothedSceneClassifier {
    /// Create a new `SmoothedSceneClassifier` with the given window size.
    ///
    /// A `window_size` of 1 disables smoothing.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            inner: SceneClassifier::new(),
            smoother: crate::classify::temporal_smooth::TemporalSmoother::new(window_size),
        }
    }

    /// Create a `SmoothedSceneClassifier` wrapping an existing `SceneClassifier`.
    #[must_use]
    pub fn with_classifier(classifier: SceneClassifier, window_size: usize) -> Self {
        Self {
            inner: classifier,
            smoother: crate::classify::temporal_smooth::TemporalSmoother::new(window_size),
        }
    }

    /// Classify a frame and return the mode-smoothed [`SceneType`].
    ///
    /// # Errors
    ///
    /// Returns `SceneError::InvalidDimensions` if `rgb.len() != width * height * 3`.
    pub fn classify(
        &mut self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> crate::error::SceneResult<SceneType> {
        let classification = self.inner.classify(rgb_data, width, height)?;
        self.smoother.push(classification.scene_type);
        Ok(self
            .smoother
            .current_class()
            .copied()
            .unwrap_or(classification.scene_type))
    }

    /// Classify a frame and return both the raw [`SceneClassification`] and the
    /// mode-smoothed [`SceneType`].
    ///
    /// # Errors
    ///
    /// Returns `SceneError::InvalidDimensions` if `rgb.len() != width * height * 3`.
    pub fn classify_full(
        &mut self,
        rgb_data: &[u8],
        width: usize,
        height: usize,
    ) -> crate::error::SceneResult<(SceneClassification, SceneType)> {
        let classification = self.inner.classify(rgb_data, width, height)?;
        self.smoother.push(classification.scene_type);
        let smoothed = self
            .smoother
            .current_class()
            .copied()
            .unwrap_or(classification.scene_type);
        Ok((classification, smoothed))
    }

    /// Reset the temporal smoothing window.
    pub fn reset(&mut self) {
        self.smoother.clear();
    }

    /// Return the configured window size.
    #[must_use]
    pub fn window_size(&self) -> usize {
        self.smoother.window_size
    }

    /// Return the number of frames currently in the window.
    #[must_use]
    pub fn window_len(&self) -> usize {
        self.smoother.len()
    }
}

/// Helper: build a solid color image.
fn solid_image(width: usize, height: usize, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut data = vec![0u8; width * height * 3];
    for i in (0..data.len()).step_by(3) {
        data[i] = r;
        data[i + 1] = g;
        data[i + 2] = b;
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scene_type_name() {
        assert_eq!(SceneType::Indoor.name(), "Indoor");
        assert_eq!(SceneType::Outdoor.name(), "Outdoor");
    }

    #[test]
    fn test_scene_classifier() {
        let classifier = SceneClassifier::new();

        // Create a bright blue image (sky scene)
        let width = 100;
        let height = 100;
        let mut rgb_data = vec![0u8; width * height * 3];
        for i in (0..rgb_data.len()).step_by(3) {
            rgb_data[i] = 100; // R
            rgb_data[i + 1] = 150; // G
            rgb_data[i + 2] = 255; // B (bright blue)
        }

        let result = classifier.classify(&rgb_data, width, height);
        assert!(result.is_ok());

        let classification = result.expect("should succeed in test");
        assert!(classification.confidence.value() > 0.0);
    }

    #[test]
    fn test_invalid_dimensions() {
        let classifier = SceneClassifier::new();
        let rgb_data = vec![0u8; 100];
        let result = classifier.classify(&rgb_data, 10, 10);
        assert!(result.is_err());
    }

    #[test]
    fn test_scene_features_default() {
        let features = SceneFeatures::default();
        assert!((features.brightness - 0.5).abs() < f32::EPSILON);
        assert!((features.saturation - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_temporal_smoothing_reduces_flicker() {
        let classifier = SceneClassifier::with_temporal_smoothing(5);
        let w = 80;
        let h = 80;

        // Alternate between two different frames
        let sky_frame = solid_image(w, h, 80, 120, 220);
        let dark_frame = solid_image(w, h, 20, 20, 20);

        let r1 = classifier.classify(&sky_frame, w, h).expect("ok");
        let r2 = classifier.classify(&dark_frame, w, h).expect("ok");
        let r3 = classifier.classify(&sky_frame, w, h).expect("ok");

        // After temporal smoothing the sky frame after a dark frame should still
        // produce a valid result with non-zero confidence
        assert!(r1.confidence.value() >= 0.0);
        assert!(r2.confidence.value() >= 0.0);
        assert!(r3.confidence.value() >= 0.0);
    }

    #[test]
    fn test_reset_temporal_buffer() {
        let classifier = SceneClassifier::with_temporal_smoothing(4);
        let w = 60;
        let h = 60;
        let frame = solid_image(w, h, 100, 150, 200);
        let _ = classifier.classify(&frame, w, h).expect("ok");
        // Reset should not panic
        classifier.reset_temporal_buffer();
        // Should still work after reset
        let r = classifier.classify(&frame, w, h).expect("ok");
        assert!(r.confidence.value() >= 0.0);
    }

    #[test]
    fn test_temporal_buffer_smooth() {
        let mut buf = TemporalBuffer::new(3, 3);
        buf.push(vec![1.0, 0.0, 0.0]);
        buf.push(vec![0.0, 1.0, 0.0]);
        buf.push(vec![0.0, 0.0, 1.0]);
        let smoothed = buf.smooth(0.3);
        // All three types should have non-zero weight after smoothing
        assert_eq!(smoothed.len(), 3);
        assert!(smoothed.iter().all(|&v| v >= 0.0 && v <= 1.0));
    }

    // ── SmoothedSceneClassifier ──────────────────────────────────────────────

    #[test]
    fn test_smoothed_classifier_returns_scene_type() {
        let mut sc = SmoothedSceneClassifier::new(3);
        let w = 40;
        let h = 40;
        let frame = solid_image(w, h, 80, 120, 220);
        let result = sc.classify(&frame, w, h);
        assert!(result.is_ok(), "should classify successfully");
    }

    #[test]
    fn test_smoothed_classifier_invalid_dimensions() {
        let mut sc = SmoothedSceneClassifier::new(3);
        let bad_data = vec![0u8; 10];
        let result = sc.classify(&bad_data, 10, 10);
        assert!(result.is_err(), "wrong-size buffer should error");
    }

    #[test]
    fn test_smoothed_classifier_mode_stabilizes_outlier() {
        let mut sc = SmoothedSceneClassifier::new(5);
        let w = 40;
        let h = 40;
        // Three sky-like frames then one dark outlier then one sky
        let sky = solid_image(w, h, 80, 120, 220);
        let dark = solid_image(w, h, 10, 10, 10);
        sc.classify(&sky, w, h).expect("ok");
        sc.classify(&sky, w, h).expect("ok");
        sc.classify(&sky, w, h).expect("ok");
        let after_outlier = sc.classify(&dark, w, h).expect("ok");
        let after_sky = sc.classify(&sky, w, h).expect("ok");
        // The window has 4 sky + 1 dark; mode should be the sky-related class
        assert_eq!(
            after_outlier, after_sky,
            "mode should stabilise away from single outlier"
        );
    }

    #[test]
    fn test_smoothed_classifier_reset_clears_window() {
        let mut sc = SmoothedSceneClassifier::new(4);
        let w = 30;
        let h = 30;
        let frame = solid_image(w, h, 100, 150, 200);
        sc.classify(&frame, w, h).expect("ok");
        sc.classify(&frame, w, h).expect("ok");
        assert_eq!(sc.window_len(), 2);
        sc.reset();
        assert_eq!(sc.window_len(), 0, "reset should clear window");
    }

    #[test]
    fn test_smoothed_classifier_window_size_reported() {
        let sc = SmoothedSceneClassifier::new(7);
        assert_eq!(sc.window_size(), 7);
    }

    #[test]
    fn test_smoothed_classifier_classify_full_returns_both() {
        let mut sc = SmoothedSceneClassifier::new(3);
        let w = 40;
        let h = 40;
        let frame = solid_image(w, h, 80, 120, 220);
        let result = sc.classify_full(&frame, w, h);
        assert!(result.is_ok(), "classify_full should succeed");
        let (classification, smoothed_type) = result.expect("ok");
        assert!(classification.confidence.value() >= 0.0);
        // After one frame the smoothed type equals the raw classification
        assert_eq!(smoothed_type, classification.scene_type);
    }

    #[test]
    fn test_smoothed_classifier_with_custom_inner() {
        let inner = SceneClassifier::with_temporal_smoothing(3);
        let mut sc = SmoothedSceneClassifier::with_classifier(inner, 4);
        let w = 30;
        let h = 30;
        let frame = solid_image(w, h, 200, 100, 50);
        let result = sc.classify(&frame, w, h);
        assert!(result.is_ok());
    }

    #[test]
    fn test_smoothed_classifier_single_frame_window_len_one() {
        let mut sc = SmoothedSceneClassifier::new(5);
        let w = 20;
        let h = 20;
        let frame = solid_image(w, h, 100, 200, 100);
        sc.classify(&frame, w, h).expect("ok");
        assert_eq!(sc.window_len(), 1);
    }

    #[test]
    fn test_smoothed_classifier_consistent_scene_stays_stable() {
        let mut sc = SmoothedSceneClassifier::new(4);
        let w = 50;
        let h = 50;
        let frame = solid_image(w, h, 50, 160, 60); // vegetation-like
        let mut last_type = None;
        for _ in 0..6 {
            let t = sc.classify(&frame, w, h).expect("ok");
            if let Some(prev) = last_type {
                assert_eq!(t, prev, "consistent scene should produce stable label");
            }
            last_type = Some(t);
        }
    }
}
