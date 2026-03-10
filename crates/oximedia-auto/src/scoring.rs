//! Scene scoring and importance analysis.
//!
//! This module provides algorithms for scoring video scenes based on various
//! features and generating interest curves for content analysis.
//!
//! # Features
//!
//! - **Scene importance scoring**: Combine multiple metrics
//! - **Interest curve generation**: Temporal engagement analysis
//! - **Content classification**: Categorize scene types
//! - **Sentiment analysis**: Visual emotion detection
//! - **Auto-titling suggestions**: Generate descriptive titles
//!
//! # Example
//!
//! ```
//! use oximedia_auto::scoring::{SceneScorer, ScoringConfig};
//!
//! let config = ScoringConfig::default();
//! let scorer = SceneScorer::new(config);
//! ```

use crate::error::{AutoError, AutoResult};
use oximedia_core::Timestamp;
use std::collections::HashMap;

/// Scene importance score (0.0 to 1.0).
pub type ImportanceScore = f64;

/// Type of content detected in a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContentType {
    /// Action scene with high motion.
    Action,
    /// Dialogue or conversation.
    Dialogue,
    /// Static or slow-moving scene.
    Static,
    /// Establishing shot or landscape.
    Establishing,
    /// Close-up shot.
    CloseUp,
    /// Group shot with multiple subjects.
    Group,
    /// Transition or filler content.
    Transition,
    /// Unknown or unclassified content.
    Unknown,
}

impl ContentType {
    /// Get the base importance weight for this content type.
    #[must_use]
    pub const fn base_importance(&self) -> f64 {
        match self {
            Self::Action => 0.85,
            Self::Dialogue => 0.75,
            Self::CloseUp => 0.70,
            Self::Group => 0.65,
            Self::Establishing => 0.55,
            Self::Static => 0.40,
            Self::Transition => 0.20,
            Self::Unknown => 0.50,
        }
    }
}

/// Visual sentiment detected in a scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sentiment {
    /// Positive or uplifting.
    Positive,
    /// Neutral.
    Neutral,
    /// Negative or somber.
    Negative,
    /// Tense or suspenseful.
    Tense,
    /// Calm or peaceful.
    Calm,
}

impl Sentiment {
    /// Get the emotional intensity multiplier.
    #[must_use]
    pub const fn intensity_multiplier(&self) -> f64 {
        match self {
            Self::Tense => 1.3,
            Self::Positive => 1.2,
            Self::Negative => 1.1,
            Self::Neutral => 1.0,
            Self::Calm => 0.9,
        }
    }
}

/// Scene feature metrics used for scoring.
#[derive(Debug, Clone, Default)]
pub struct SceneFeatures {
    /// Motion intensity (0.0 to 1.0).
    pub motion_intensity: f64,
    /// Face count in the scene.
    pub face_count: usize,
    /// Face coverage ratio (0.0 to 1.0).
    pub face_coverage: f64,
    /// Audio peak level (0.0 to 1.0).
    pub audio_peak: f64,
    /// Audio energy (0.0 to 1.0).
    pub audio_energy: f64,
    /// Color diversity score (0.0 to 1.0).
    pub color_diversity: f64,
    /// Edge density (0.0 to 1.0).
    pub edge_density: f64,
    /// Brightness mean (0.0 to 1.0).
    pub brightness_mean: f64,
    /// Contrast level (0.0 to 1.0).
    pub contrast: f64,
    /// Sharpness metric (0.0 to 1.0).
    pub sharpness: f64,
    /// Object count.
    pub object_count: usize,
    /// Object diversity score (0.0 to 1.0).
    pub object_diversity: f64,
    /// Temporal stability (0.0 to 1.0).
    pub temporal_stability: f64,
}

impl SceneFeatures {
    /// Create new scene features with all values set to zero.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Compute a composite feature score.
    #[must_use]
    pub fn composite_score(&self, weights: &FeatureWeights) -> f64 {
        let mut score = 0.0;
        let mut total_weight = 0.0;

        score += self.motion_intensity * weights.motion;
        total_weight += weights.motion;

        score += self.face_coverage * weights.face;
        total_weight += weights.face;

        score += self.audio_peak * weights.audio_peak;
        total_weight += weights.audio_peak;

        score += self.audio_energy * weights.audio_energy;
        total_weight += weights.audio_energy;

        score += self.color_diversity * weights.color;
        total_weight += weights.color;

        score += self.edge_density * weights.edge;
        total_weight += weights.edge;

        score += self.contrast * weights.contrast;
        total_weight += weights.contrast;

        score += self.sharpness * weights.sharpness;
        total_weight += weights.sharpness;

        score += self.object_diversity * weights.object;
        total_weight += weights.object;

        if total_weight > 0.0 {
            score / total_weight
        } else {
            0.0
        }
    }
}

/// Weights for different feature components.
#[derive(Debug, Clone)]
pub struct FeatureWeights {
    /// Motion importance weight.
    pub motion: f64,
    /// Face detection weight.
    pub face: f64,
    /// Audio peak weight.
    pub audio_peak: f64,
    /// Audio energy weight.
    pub audio_energy: f64,
    /// Color diversity weight.
    pub color: f64,
    /// Edge density weight.
    pub edge: f64,
    /// Contrast weight.
    pub contrast: f64,
    /// Sharpness weight.
    pub sharpness: f64,
    /// Object detection weight.
    pub object: f64,
}

impl Default for FeatureWeights {
    fn default() -> Self {
        Self {
            motion: 1.5,
            face: 1.2,
            audio_peak: 1.3,
            audio_energy: 1.0,
            color: 0.8,
            edge: 0.7,
            contrast: 0.6,
            sharpness: 0.5,
            object: 1.1,
        }
    }
}

/// A scored scene segment.
#[derive(Debug, Clone)]
pub struct ScoredScene {
    /// Scene start timestamp.
    pub start: Timestamp,
    /// Scene end timestamp.
    pub end: Timestamp,
    /// Importance score (0.0 to 1.0).
    pub score: ImportanceScore,
    /// Content type classification.
    pub content_type: ContentType,
    /// Detected sentiment.
    pub sentiment: Sentiment,
    /// Scene features used for scoring.
    pub features: SceneFeatures,
    /// Suggested title or description.
    pub suggested_title: Option<String>,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl ScoredScene {
    /// Create a new scored scene.
    #[must_use]
    pub fn new(
        start: Timestamp,
        end: Timestamp,
        score: ImportanceScore,
        content_type: ContentType,
        sentiment: Sentiment,
    ) -> Self {
        Self {
            start,
            end,
            score: score.clamp(0.0, 1.0),
            content_type,
            sentiment,
            features: SceneFeatures::default(),
            suggested_title: None,
            metadata: HashMap::new(),
        }
    }

    /// Get the duration of this scene.
    #[must_use]
    pub fn duration(&self) -> i64 {
        (self.end.pts - self.start.pts).max(0)
    }

    /// Check if this scene meets a minimum score threshold.
    #[must_use]
    pub fn meets_threshold(&self, threshold: ImportanceScore) -> bool {
        self.score >= threshold
    }

    /// Compute the final adjusted score with sentiment multiplier.
    #[must_use]
    pub fn adjusted_score(&self) -> ImportanceScore {
        (self.score * self.sentiment.intensity_multiplier()).clamp(0.0, 1.0)
    }
}

/// Interest curve for temporal engagement analysis.
#[derive(Debug, Clone)]
pub struct InterestCurve {
    /// Timestamp and score pairs.
    pub points: Vec<(Timestamp, ImportanceScore)>,
    /// Smoothing window size.
    pub window_size: usize,
}

impl InterestCurve {
    /// Create a new interest curve.
    #[must_use]
    pub fn new(window_size: usize) -> Self {
        Self {
            points: Vec::new(),
            window_size,
        }
    }

    /// Add a point to the curve.
    pub fn add_point(&mut self, timestamp: Timestamp, score: ImportanceScore) {
        self.points.push((timestamp, score.clamp(0.0, 1.0)));
    }

    /// Get the score at a specific timestamp (interpolated).
    #[must_use]
    pub fn score_at(&self, timestamp: Timestamp) -> ImportanceScore {
        if self.points.is_empty() {
            return 0.0;
        }

        // Find surrounding points
        let mut before = None;
        let mut after = None;

        for (i, (ts, score)) in self.points.iter().enumerate() {
            if *ts <= timestamp {
                before = Some((i, *ts, *score));
            }
            if *ts >= timestamp && after.is_none() {
                after = Some((i, *ts, *score));
                break;
            }
        }

        match (before, after) {
            (Some((_, _, score)), None) => score,
            (None, Some((_, _, score))) => score,
            (Some((_, t1, s1)), Some((_, t2, s2))) => {
                if t1 == t2 {
                    s1
                } else {
                    let ratio = (timestamp.pts - t1.pts) as f64 / (t2.pts - t1.pts) as f64;
                    s1 + (s2 - s1) * ratio
                }
            }
            (None, None) => 0.0,
        }
    }

    /// Get smoothed curve using moving average.
    #[must_use]
    pub fn smoothed(&self) -> Self {
        if self.points.len() < self.window_size {
            return self.clone();
        }

        let mut smoothed = Self::new(self.window_size);

        for i in 0..self.points.len() {
            let start = i.saturating_sub(self.window_size / 2);
            let end = (i + self.window_size / 2 + 1).min(self.points.len());

            let avg_score: f64 =
                self.points[start..end].iter().map(|(_, s)| s).sum::<f64>() / (end - start) as f64;

            smoothed.add_point(self.points[i].0, avg_score);
        }

        smoothed
    }

    /// Find peaks in the interest curve.
    #[must_use]
    pub fn find_peaks(&self, threshold: ImportanceScore) -> Vec<(Timestamp, ImportanceScore)> {
        let smoothed = self.smoothed();
        let mut peaks = Vec::new();

        for i in 1..smoothed.points.len().saturating_sub(1) {
            let (ts, score) = smoothed.points[i];
            let prev_score = smoothed.points[i - 1].1;
            let next_score = smoothed.points[i + 1].1;

            if score >= threshold && score >= prev_score && score >= next_score {
                peaks.push((ts, score));
            }
        }

        peaks
    }

    /// Find valleys in the interest curve.
    #[must_use]
    pub fn find_valleys(&self, threshold: ImportanceScore) -> Vec<(Timestamp, ImportanceScore)> {
        let smoothed = self.smoothed();
        let mut valleys = Vec::new();

        for i in 1..smoothed.points.len().saturating_sub(1) {
            let (ts, score) = smoothed.points[i];
            let prev_score = smoothed.points[i - 1].1;
            let next_score = smoothed.points[i + 1].1;

            if score <= threshold && score <= prev_score && score <= next_score {
                valleys.push((ts, score));
            }
        }

        valleys
    }
}

/// Configuration for scene scoring.
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    /// Feature weights.
    pub feature_weights: FeatureWeights,
    /// Minimum scene duration in milliseconds.
    pub min_scene_duration_ms: i64,
    /// Interest curve smoothing window size.
    pub curve_window_size: usize,
    /// Enable content classification.
    pub enable_classification: bool,
    /// Enable sentiment analysis.
    pub enable_sentiment: bool,
    /// Enable auto-titling.
    pub enable_auto_titling: bool,
    /// Peak detection threshold.
    pub peak_threshold: f64,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            feature_weights: FeatureWeights::default(),
            min_scene_duration_ms: 500,
            curve_window_size: 5,
            enable_classification: true,
            enable_sentiment: true,
            enable_auto_titling: true,
            peak_threshold: 0.65,
        }
    }
}

impl ScoringConfig {
    /// Create a new scoring configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the feature weights.
    #[must_use]
    pub fn with_feature_weights(mut self, weights: FeatureWeights) -> Self {
        self.feature_weights = weights;
        self
    }

    /// Set the minimum scene duration.
    #[must_use]
    pub const fn with_min_scene_duration_ms(mut self, duration_ms: i64) -> Self {
        self.min_scene_duration_ms = duration_ms;
        self
    }

    /// Set the curve window size.
    #[must_use]
    pub const fn with_curve_window_size(mut self, window_size: usize) -> Self {
        self.curve_window_size = window_size;
        self
    }

    /// Enable or disable content classification.
    #[must_use]
    pub const fn with_classification(mut self, enable: bool) -> Self {
        self.enable_classification = enable;
        self
    }

    /// Enable or disable sentiment analysis.
    #[must_use]
    pub const fn with_sentiment(mut self, enable: bool) -> Self {
        self.enable_sentiment = enable;
        self
    }

    /// Enable or disable auto-titling.
    #[must_use]
    pub const fn with_auto_titling(mut self, enable: bool) -> Self {
        self.enable_auto_titling = enable;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if self.min_scene_duration_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.min_scene_duration_ms,
            });
        }

        if self.curve_window_size == 0 {
            return Err(AutoError::invalid_parameter(
                "curve_window_size",
                "must be greater than 0",
            ));
        }

        if !(0.0..=1.0).contains(&self.peak_threshold) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.peak_threshold,
                min: 0.0,
                max: 1.0,
            });
        }

        Ok(())
    }
}

/// Scene scorer for importance analysis.
pub struct SceneScorer {
    /// Configuration.
    config: ScoringConfig,
}

impl SceneScorer {
    /// Create a new scene scorer.
    #[must_use]
    pub fn new(config: ScoringConfig) -> Self {
        Self { config }
    }

    /// Create a scene scorer with default configuration.
    #[must_use]
    pub fn default_scorer() -> Self {
        Self::new(ScoringConfig::default())
    }

    /// Score a scene based on its features.
    pub fn score_scene(
        &self,
        start: Timestamp,
        end: Timestamp,
        features: SceneFeatures,
    ) -> AutoResult<ScoredScene> {
        self.config.validate()?;

        // Compute base score from features
        let base_score = features.composite_score(&self.config.feature_weights);

        // Classify content type
        let content_type = if self.config.enable_classification {
            self.classify_content(&features)
        } else {
            ContentType::Unknown
        };

        // Apply content type base importance
        let type_adjusted_score = base_score * 0.7 + content_type.base_importance() * 0.3;

        // Detect sentiment
        let sentiment = if self.config.enable_sentiment {
            self.detect_sentiment(&features)
        } else {
            Sentiment::Neutral
        };

        // Generate title suggestion
        let suggested_title = if self.config.enable_auto_titling {
            Some(self.generate_title(content_type, sentiment, &features))
        } else {
            None
        };

        let mut scene = ScoredScene::new(start, end, type_adjusted_score, content_type, sentiment);
        scene.features = features;
        scene.suggested_title = suggested_title;

        Ok(scene)
    }

    /// Classify content type based on features.
    #[must_use]
    fn classify_content(&self, features: &SceneFeatures) -> ContentType {
        // High motion indicates action
        if features.motion_intensity > 0.7 {
            return ContentType::Action;
        }

        // Multiple faces with low motion suggests dialogue
        if features.face_count >= 2 && features.motion_intensity < 0.3 {
            return ContentType::Dialogue;
        }

        // Single face with high coverage is close-up
        if features.face_count == 1 && features.face_coverage > 0.4 {
            return ContentType::CloseUp;
        }

        // Multiple faces is group shot
        if features.face_count > 2 {
            return ContentType::Group;
        }

        // Low motion and high temporal stability is static
        if features.motion_intensity < 0.2 && features.temporal_stability > 0.7 {
            return ContentType::Static;
        }

        // High edge density with low faces suggests establishing shot
        if features.edge_density > 0.6 && features.face_count == 0 {
            return ContentType::Establishing;
        }

        ContentType::Unknown
    }

    /// Detect visual sentiment based on features.
    #[must_use]
    fn detect_sentiment(&self, features: &SceneFeatures) -> Sentiment {
        // High motion and energy suggests tense or action-oriented
        if features.motion_intensity > 0.6 && features.audio_energy > 0.6 {
            return Sentiment::Tense;
        }

        // Bright with high color diversity suggests positive
        if features.brightness_mean > 0.6 && features.color_diversity > 0.5 {
            return Sentiment::Positive;
        }

        // Low brightness and low color suggests negative
        if features.brightness_mean < 0.4 && features.color_diversity < 0.4 {
            return Sentiment::Negative;
        }

        // Low motion and high stability suggests calm
        if features.motion_intensity < 0.3 && features.temporal_stability > 0.7 {
            return Sentiment::Calm;
        }

        Sentiment::Neutral
    }

    /// Generate a suggested title for the scene.
    #[must_use]
    fn generate_title(
        &self,
        content_type: ContentType,
        sentiment: Sentiment,
        features: &SceneFeatures,
    ) -> String {
        let type_str = match content_type {
            ContentType::Action => "Action Sequence",
            ContentType::Dialogue => "Conversation",
            ContentType::CloseUp => "Close-up Shot",
            ContentType::Group => "Group Scene",
            ContentType::Establishing => "Establishing Shot",
            ContentType::Static => "Static Scene",
            ContentType::Transition => "Transition",
            ContentType::Unknown => "Scene",
        };

        let sentiment_modifier = match sentiment {
            Sentiment::Positive => " (Uplifting)",
            Sentiment::Negative => " (Somber)",
            Sentiment::Tense => " (Intense)",
            Sentiment::Calm => " (Peaceful)",
            Sentiment::Neutral => "",
        };

        // Add face count if relevant
        let face_note = if features.face_count > 0 {
            format!(
                " - {} {}",
                features.face_count,
                if features.face_count == 1 {
                    "person"
                } else {
                    "people"
                }
            )
        } else {
            String::new()
        };

        format!("{type_str}{sentiment_modifier}{face_note}")
    }

    /// Generate an interest curve from scored scenes.
    #[must_use]
    pub fn generate_interest_curve(&self, scenes: &[ScoredScene]) -> InterestCurve {
        let mut curve = InterestCurve::new(self.config.curve_window_size);

        for scene in scenes {
            // Add points at start and end of each scene
            curve.add_point(scene.start, scene.adjusted_score());
            curve.add_point(scene.end, scene.adjusted_score());
        }

        curve
    }

    /// Find highlight moments using interest curve peaks.
    #[must_use]
    pub fn find_highlights(&self, curve: &InterestCurve) -> Vec<(Timestamp, ImportanceScore)> {
        curve.find_peaks(self.config.peak_threshold)
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &ScoringConfig {
        &self.config
    }
}

impl Default for SceneScorer {
    fn default() -> Self {
        Self::default_scorer()
    }
}

/// Batch score multiple scenes.
#[allow(dead_code)]
pub fn batch_score_scenes(
    scorer: &SceneScorer,
    scene_data: &[(Timestamp, Timestamp, SceneFeatures)],
) -> AutoResult<Vec<ScoredScene>> {
    scene_data
        .iter()
        .map(|(start, end, features)| scorer.score_scene(*start, *end, features.clone()))
        .collect()
}

/// Compute normalized importance scores across all scenes.
#[allow(dead_code)]
pub fn normalize_scores(scenes: &mut [ScoredScene]) {
    if scenes.is_empty() {
        return;
    }

    let max_score = scenes
        .iter()
        .map(ScoredScene::adjusted_score)
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or(1.0);

    if max_score > 0.0 {
        for scene in scenes {
            scene.score = (scene.adjusted_score() / max_score).clamp(0.0, 1.0);
        }
    }
}
