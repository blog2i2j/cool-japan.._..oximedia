//! Auto-generated image alt-text from visual features.
//!
//! Analyses low-level visual features of an image (dominant colour, edge density,
//! brightness, aspect ratio, texture complexity) and composes a descriptive
//! alt-text string suitable for screen readers.
//!
//! This module does **not** rely on a neural network — instead it uses
//! deterministic heuristics that run in pure Rust with no external dependencies.
//! The resulting descriptions are intentionally factual and concise, following
//! WCAG 1.1.1 guidance for non-text content.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_access::auto_alt_text::{AltTextGenerator, ImageFeatures, DominantColor};
//!
//! let features = ImageFeatures {
//!     width: 1920,
//!     height: 1080,
//!     dominant_color: DominantColor::Blue,
//!     brightness: 0.65,
//!     edge_density: 0.3,
//!     texture_complexity: 0.4,
//!     face_count: 2,
//!     text_present: false,
//! };
//!
//! let gen = AltTextGenerator::default();
//! let alt = gen.generate(&features);
//! assert!(!alt.text.is_empty());
//! ```

use serde::{Deserialize, Serialize};

// ── Visual feature types ─────────────────────────────────────────────────────

/// Dominant colour category detected in an image.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DominantColor {
    /// Predominantly red hues.
    Red,
    /// Predominantly orange hues.
    Orange,
    /// Predominantly yellow hues.
    Yellow,
    /// Predominantly green hues.
    Green,
    /// Predominantly blue hues.
    Blue,
    /// Predominantly purple / violet hues.
    Purple,
    /// Predominantly brown / earth tones.
    Brown,
    /// Predominantly grey / neutral tones.
    Grey,
    /// Predominantly black / very dark.
    Black,
    /// Predominantly white / very bright.
    White,
    /// Multi-coloured with no single dominant hue.
    Multicolored,
}

impl DominantColor {
    /// Human-readable colour name for alt-text descriptions.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Red => "red",
            Self::Orange => "orange",
            Self::Yellow => "yellow",
            Self::Green => "green",
            Self::Blue => "blue",
            Self::Purple => "purple",
            Self::Brown => "brown",
            Self::Grey => "grey",
            Self::Black => "black",
            Self::White => "white",
            Self::Multicolored => "multicolored",
        }
    }

    /// Warm / cool classification for richer descriptions.
    #[must_use]
    pub fn temperature(&self) -> ColorTemperature {
        match self {
            Self::Red | Self::Orange | Self::Yellow | Self::Brown => ColorTemperature::Warm,
            Self::Blue | Self::Purple | Self::Grey => ColorTemperature::Cool,
            Self::Green => ColorTemperature::Neutral,
            Self::Black | Self::White | Self::Multicolored => ColorTemperature::Neutral,
        }
    }
}

/// Warm / cool / neutral colour classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorTemperature {
    /// Warm tones (reds, oranges, yellows).
    Warm,
    /// Cool tones (blues, purples).
    Cool,
    /// Neutral tones.
    Neutral,
}

// ── Image features ───────────────────────────────────────────────────────────

/// Low-level visual features extracted from an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageFeatures {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// The most dominant colour category.
    pub dominant_color: DominantColor,
    /// Mean brightness (0.0 = fully dark, 1.0 = fully bright).
    pub brightness: f32,
    /// Edge density (0.0 = no edges, 1.0 = maximum edge content).
    pub edge_density: f32,
    /// Texture complexity (0.0 = flat/smooth, 1.0 = very detailed).
    pub texture_complexity: f32,
    /// Estimated number of human faces detected (0 if none).
    pub face_count: u32,
    /// Whether text / characters were detected in the image.
    pub text_present: bool,
}

impl ImageFeatures {
    /// Aspect ratio as width / height.
    #[must_use]
    pub fn aspect_ratio(&self) -> f32 {
        if self.height == 0 {
            return 0.0;
        }
        self.width as f32 / self.height as f32
    }

    /// Classify the aspect ratio.
    #[must_use]
    pub fn aspect_class(&self) -> AspectClass {
        let ratio = self.aspect_ratio();
        if ratio < 0.01 {
            AspectClass::Unknown
        } else if ratio < 0.8 {
            AspectClass::Portrait
        } else if ratio <= 1.2 {
            AspectClass::Square
        } else if ratio <= 2.0 {
            AspectClass::Landscape
        } else {
            AspectClass::Panoramic
        }
    }

    /// Classify brightness.
    #[must_use]
    pub fn brightness_class(&self) -> BrightnessClass {
        if self.brightness < 0.2 {
            BrightnessClass::Dark
        } else if self.brightness < 0.45 {
            BrightnessClass::Dim
        } else if self.brightness < 0.65 {
            BrightnessClass::Medium
        } else if self.brightness < 0.85 {
            BrightnessClass::Bright
        } else {
            BrightnessClass::VeryBright
        }
    }

    /// Classify scene complexity from edge density + texture.
    #[must_use]
    pub fn scene_complexity(&self) -> SceneComplexity {
        let combined = (self.edge_density + self.texture_complexity) / 2.0;
        if combined < 0.15 {
            SceneComplexity::Minimal
        } else if combined < 0.35 {
            SceneComplexity::Simple
        } else if combined < 0.6 {
            SceneComplexity::Moderate
        } else {
            SceneComplexity::Complex
        }
    }

    /// Classify the likely content type based on features.
    #[must_use]
    pub fn content_type_hint(&self) -> ContentTypeHint {
        if self.text_present && self.edge_density > 0.4 {
            return ContentTypeHint::Graphic;
        }
        if self.face_count > 0 {
            return ContentTypeHint::People;
        }
        if self.edge_density < 0.1 && self.texture_complexity < 0.1 {
            return ContentTypeHint::SolidOrGradient;
        }
        if self.edge_density > 0.5 {
            return ContentTypeHint::DetailedScene;
        }
        ContentTypeHint::Photograph
    }
}

/// Aspect ratio classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AspectClass {
    /// Taller than wide (ratio < 0.8).
    Portrait,
    /// Roughly equal sides (0.8 – 1.2).
    Square,
    /// Wider than tall (1.2 – 2.0).
    Landscape,
    /// Very wide (ratio > 2.0).
    Panoramic,
    /// Could not determine (degenerate dimensions).
    Unknown,
}

impl AspectClass {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::Portrait => "portrait",
            Self::Square => "square",
            Self::Landscape => "landscape",
            Self::Panoramic => "panoramic",
            Self::Unknown => "unknown-format",
        }
    }
}

/// Brightness classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BrightnessClass {
    /// Very dark / low-key image.
    Dark,
    /// Dim image.
    Dim,
    /// Medium brightness.
    Medium,
    /// Bright image.
    Bright,
    /// Very bright / high-key image.
    VeryBright,
}

impl BrightnessClass {
    /// Adjective for alt-text use.
    #[must_use]
    pub fn adjective(&self) -> &'static str {
        match self {
            Self::Dark => "dark",
            Self::Dim => "dimly lit",
            Self::Medium => "moderately lit",
            Self::Bright => "bright",
            Self::VeryBright => "very bright",
        }
    }
}

/// Scene complexity classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SceneComplexity {
    /// Almost no visual detail.
    Minimal,
    /// Simple composition.
    Simple,
    /// Moderate detail.
    Moderate,
    /// Highly detailed / busy scene.
    Complex,
}

impl SceneComplexity {
    /// Descriptor for alt-text.
    #[must_use]
    pub fn descriptor(&self) -> &'static str {
        match self {
            Self::Minimal => "minimalist",
            Self::Simple => "simple",
            Self::Moderate => "moderately detailed",
            Self::Complex => "highly detailed",
        }
    }
}

/// Content type hint derived from visual features.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ContentTypeHint {
    /// Photograph or natural scene.
    Photograph,
    /// Image containing people / faces.
    People,
    /// Graphic, chart, or text-heavy image.
    Graphic,
    /// Solid colour or gradient.
    SolidOrGradient,
    /// Very detailed scene (architecture, crowds, etc.).
    DetailedScene,
}

impl ContentTypeHint {
    /// Noun phrase for alt-text use.
    #[must_use]
    pub fn noun_phrase(&self) -> &'static str {
        match self {
            Self::Photograph => "photograph",
            Self::People => "image of people",
            Self::Graphic => "graphic or chart",
            Self::SolidOrGradient => "solid or gradient background",
            Self::DetailedScene => "detailed scene",
        }
    }
}

// ── Generated alt-text ───────────────────────────────────────────────────────

/// The result of alt-text generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedAltText {
    /// The composed alt-text string.
    pub text: String,
    /// Confidence in the description accuracy (0.0–1.0).
    pub confidence: f32,
    /// The content type that was inferred.
    pub content_type: ContentTypeHint,
    /// Number of features that contributed to the description.
    pub feature_count: usize,
}

// ── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the alt-text generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AltTextConfig {
    /// Maximum character length for generated alt text.
    pub max_length: usize,
    /// Whether to include resolution information.
    pub include_resolution: bool,
    /// Whether to include brightness description.
    pub include_brightness: bool,
    /// Whether to include colour description.
    pub include_color: bool,
    /// Whether to include face count.
    pub include_faces: bool,
    /// Whether to include scene complexity.
    pub include_complexity: bool,
    /// Minimum confidence threshold (descriptions below this are marked low-confidence).
    pub confidence_threshold: f32,
}

impl Default for AltTextConfig {
    fn default() -> Self {
        Self {
            max_length: 200,
            include_resolution: false,
            include_brightness: true,
            include_color: true,
            include_faces: true,
            include_complexity: true,
            confidence_threshold: 0.5,
        }
    }
}

// ── Generator ────────────────────────────────────────────────────────────────

/// Generates alt-text from extracted [`ImageFeatures`].
pub struct AltTextGenerator {
    config: AltTextConfig,
}

impl AltTextGenerator {
    /// Create with custom configuration.
    #[must_use]
    pub fn new(config: AltTextConfig) -> Self {
        Self { config }
    }

    /// Generate alt-text from visual features.
    #[must_use]
    pub fn generate(&self, features: &ImageFeatures) -> GeneratedAltText {
        let mut parts: Vec<String> = Vec::new();
        let mut feature_count = 0_usize;

        // 1. Content type hint
        let content = features.content_type_hint();
        let aspect = features.aspect_class();
        parts.push(format!(
            "A {} {} {}",
            features.brightness_class().adjective(),
            aspect.label(),
            content.noun_phrase(),
        ));
        feature_count += 1;

        // 2. Colour
        if self.config.include_color {
            let color = features.dominant_color;
            let temp = color.temperature();
            let temp_adj = match temp {
                ColorTemperature::Warm => "warm",
                ColorTemperature::Cool => "cool",
                ColorTemperature::Neutral => "neutral",
            };
            parts.push(format!(
                "with predominantly {} ({}) tones",
                color.label(),
                temp_adj,
            ));
            feature_count += 1;
        }

        // 3. Scene complexity
        if self.config.include_complexity {
            parts.push(format!(
                "featuring {} composition",
                features.scene_complexity().descriptor(),
            ));
            feature_count += 1;
        }

        // 4. Faces
        if self.config.include_faces && features.face_count > 0 {
            let face_desc = match features.face_count {
                1 => "one person".to_string(),
                2 => "two people".to_string(),
                n => format!("{n} people"),
            };
            parts.push(format!("showing {face_desc}"));
            feature_count += 1;
        }

        // 5. Text presence
        if features.text_present {
            parts.push("containing visible text".to_string());
            feature_count += 1;
        }

        // 6. Resolution
        if self.config.include_resolution {
            parts.push(format!("({}x{})", features.width, features.height));
            feature_count += 1;
        }

        // Compose and truncate
        let mut text = parts.join(", ");
        if !text.ends_with('.') {
            text.push('.');
        }
        if text.len() > self.config.max_length {
            text.truncate(self.config.max_length.saturating_sub(3));
            text.push_str("...");
        }

        // Confidence heuristic: more features → higher confidence
        let base_confidence = (feature_count as f32 / 6.0).clamp(0.0, 1.0);
        // Penalise degenerate images
        let penalty = if features.width == 0 || features.height == 0 {
            0.3
        } else {
            0.0
        };
        let confidence = (base_confidence - penalty).clamp(0.0, 1.0);

        GeneratedAltText {
            text,
            confidence,
            content_type: content,
            feature_count,
        }
    }

    /// Generate alt-text and check it against the confidence threshold.
    /// Returns `None` if confidence is below the configured threshold.
    #[must_use]
    pub fn generate_if_confident(&self, features: &ImageFeatures) -> Option<GeneratedAltText> {
        let result = self.generate(features);
        if result.confidence >= self.config.confidence_threshold {
            Some(result)
        } else {
            None
        }
    }

    /// Generate alt-text for multiple images.
    #[must_use]
    pub fn generate_batch(&self, images: &[ImageFeatures]) -> Vec<GeneratedAltText> {
        images.iter().map(|f| self.generate(f)).collect()
    }

    /// Get configuration.
    #[must_use]
    pub fn config(&self) -> &AltTextConfig {
        &self.config
    }
}

impl Default for AltTextGenerator {
    fn default() -> Self {
        Self::new(AltTextConfig::default())
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_features() -> ImageFeatures {
        ImageFeatures {
            width: 1920,
            height: 1080,
            dominant_color: DominantColor::Blue,
            brightness: 0.65,
            edge_density: 0.3,
            texture_complexity: 0.4,
            face_count: 2,
            text_present: false,
        }
    }

    #[test]
    fn test_basic_generation() {
        let gen = AltTextGenerator::default();
        let alt = gen.generate(&sample_features());
        assert!(!alt.text.is_empty());
        assert!(alt.confidence > 0.0);
        assert!(alt.feature_count >= 3);
    }

    #[test]
    fn test_alt_text_contains_color() {
        let gen = AltTextGenerator::default();
        let alt = gen.generate(&sample_features());
        assert!(
            alt.text.contains("blue"),
            "alt text should mention dominant colour: {}",
            alt.text
        );
    }

    #[test]
    fn test_alt_text_contains_people() {
        let gen = AltTextGenerator::default();
        let alt = gen.generate(&sample_features());
        assert!(
            alt.text.contains("two people"),
            "alt text should describe face count: {}",
            alt.text
        );
    }

    #[test]
    fn test_no_faces_omits_people() {
        let mut features = sample_features();
        features.face_count = 0;
        let gen = AltTextGenerator::default();
        let alt = gen.generate(&features);
        assert!(
            !alt.text.contains("people") && !alt.text.contains("person"),
            "should not mention people when no faces: {}",
            alt.text
        );
    }

    #[test]
    fn test_text_present_mentioned() {
        let mut features = sample_features();
        features.text_present = true;
        features.edge_density = 0.5; // triggers Graphic content type
        let gen = AltTextGenerator::default();
        let alt = gen.generate(&features);
        assert!(
            alt.text.contains("text"),
            "should mention visible text: {}",
            alt.text
        );
    }

    #[test]
    fn test_aspect_classification() {
        let portrait = ImageFeatures {
            width: 600,
            height: 1200,
            ..sample_features()
        };
        assert_eq!(portrait.aspect_class(), AspectClass::Portrait);

        let landscape = ImageFeatures {
            width: 1920,
            height: 1080,
            ..sample_features()
        };
        assert_eq!(landscape.aspect_class(), AspectClass::Landscape);

        let square = ImageFeatures {
            width: 1000,
            height: 1000,
            ..sample_features()
        };
        assert_eq!(square.aspect_class(), AspectClass::Square);

        let pano = ImageFeatures {
            width: 4000,
            height: 1000,
            ..sample_features()
        };
        assert_eq!(pano.aspect_class(), AspectClass::Panoramic);
    }

    #[test]
    fn test_brightness_classification() {
        let dark = ImageFeatures {
            brightness: 0.1,
            ..sample_features()
        };
        assert_eq!(dark.brightness_class(), BrightnessClass::Dark);

        let bright = ImageFeatures {
            brightness: 0.9,
            ..sample_features()
        };
        assert_eq!(bright.brightness_class(), BrightnessClass::VeryBright);
    }

    #[test]
    fn test_scene_complexity() {
        let minimal = ImageFeatures {
            edge_density: 0.05,
            texture_complexity: 0.05,
            ..sample_features()
        };
        assert_eq!(minimal.scene_complexity(), SceneComplexity::Minimal);

        let complex = ImageFeatures {
            edge_density: 0.8,
            texture_complexity: 0.9,
            ..sample_features()
        };
        assert_eq!(complex.scene_complexity(), SceneComplexity::Complex);
    }

    #[test]
    fn test_confidence_threshold_filtering() {
        let config = AltTextConfig {
            confidence_threshold: 0.99,
            ..AltTextConfig::default()
        };
        let gen = AltTextGenerator::new(config);
        let result = gen.generate_if_confident(&sample_features());
        // Feature count is 4 out of 6, confidence ~0.67, should be filtered
        assert!(result.is_none());
    }

    #[test]
    fn test_max_length_truncation() {
        let config = AltTextConfig {
            max_length: 30,
            ..AltTextConfig::default()
        };
        let gen = AltTextGenerator::new(config);
        let alt = gen.generate(&sample_features());
        assert!(
            alt.text.len() <= 33, // 30 + "..."
            "text should be truncated: {} (len={})",
            alt.text,
            alt.text.len()
        );
    }

    #[test]
    fn test_batch_generation() {
        let gen = AltTextGenerator::default();
        let images = vec![sample_features(), sample_features(), sample_features()];
        let results = gen.generate_batch(&images);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(!r.text.is_empty());
        }
    }

    #[test]
    fn test_content_type_solid_gradient() {
        let features = ImageFeatures {
            edge_density: 0.05,
            texture_complexity: 0.05,
            face_count: 0,
            text_present: false,
            ..sample_features()
        };
        assert_eq!(features.content_type_hint(), ContentTypeHint::SolidOrGradient);
    }

    #[test]
    fn test_dominant_color_temperature() {
        assert_eq!(DominantColor::Red.temperature(), ColorTemperature::Warm);
        assert_eq!(DominantColor::Blue.temperature(), ColorTemperature::Cool);
        assert_eq!(DominantColor::Green.temperature(), ColorTemperature::Neutral);
    }

    #[test]
    fn test_degenerate_dimensions_low_confidence() {
        let features = ImageFeatures {
            width: 0,
            height: 0,
            ..sample_features()
        };
        let gen = AltTextGenerator::default();
        let alt = gen.generate(&features);
        assert!(
            alt.confidence < 0.5,
            "degenerate dimensions should lower confidence"
        );
    }

    #[test]
    fn test_one_person_wording() {
        let mut features = sample_features();
        features.face_count = 1;
        let gen = AltTextGenerator::default();
        let alt = gen.generate(&features);
        assert!(
            alt.text.contains("one person"),
            "single face should say 'one person': {}",
            alt.text
        );
    }

    #[test]
    fn test_resolution_option() {
        let config = AltTextConfig {
            include_resolution: true,
            ..AltTextConfig::default()
        };
        let gen = AltTextGenerator::new(config);
        let alt = gen.generate(&sample_features());
        assert!(
            alt.text.contains("1920x1080"),
            "should include resolution when configured: {}",
            alt.text
        );
    }
}
