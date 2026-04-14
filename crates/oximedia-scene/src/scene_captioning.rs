//! Scene captioning module for generating natural-language descriptions from scene features.
//!
//! Produces structured, template-driven captions for video scenes by combining
//! scene classification, object detection results, composition analysis, and motion
//! energy into human-readable descriptions. All generation is rule-based and
//! patent-free (no neural language models required).

use crate::error::{SceneError, SceneResult};
use serde::{Deserialize, Serialize};
use std::fmt;

/// Granularity level for caption generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CaptionGranularity {
    /// Single-word or short phrase label.
    Brief,
    /// One-sentence description.
    Sentence,
    /// Multi-sentence paragraph.
    Paragraph,
}

/// Scene environment context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SceneEnvironment {
    /// Outdoor daylight scene.
    OutdoorDay,
    /// Outdoor nighttime scene.
    OutdoorNight,
    /// Indoor scene.
    Indoor,
    /// Studio or controlled environment.
    Studio,
    /// Unknown or indeterminate environment.
    Unknown,
}

impl SceneEnvironment {
    /// Human-readable description.
    #[must_use]
    pub const fn description(&self) -> &'static str {
        match self {
            Self::OutdoorDay => "an outdoor daylight setting",
            Self::OutdoorNight => "an outdoor nighttime setting",
            Self::Indoor => "an indoor environment",
            Self::Studio => "a studio or controlled environment",
            Self::Unknown => "an unspecified environment",
        }
    }
}

/// Motion descriptor for captions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionDescriptor {
    /// No significant motion.
    Static,
    /// Slow, gentle motion.
    Slow,
    /// Moderate motion.
    Moderate,
    /// Fast, dynamic motion.
    Fast,
    /// Rapid, chaotic motion.
    Rapid,
}

impl MotionDescriptor {
    /// Build from normalised energy (0.0–1.0).
    #[must_use]
    pub fn from_energy(energy: f32) -> Self {
        if energy < 0.05 {
            Self::Static
        } else if energy < 0.2 {
            Self::Slow
        } else if energy < 0.45 {
            Self::Moderate
        } else if energy < 0.7 {
            Self::Fast
        } else {
            Self::Rapid
        }
    }

    /// Motion adverb for use in captions.
    #[must_use]
    pub const fn adverb(&self) -> &'static str {
        match self {
            Self::Static => "still",
            Self::Slow => "slowly",
            Self::Moderate => "steadily",
            Self::Fast => "quickly",
            Self::Rapid => "rapidly",
        }
    }

    /// Motion adjective for use in captions.
    #[must_use]
    pub const fn adjective(&self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Slow => "slow-paced",
            Self::Moderate => "moderate",
            Self::Fast => "fast-paced",
            Self::Rapid => "high-energy",
        }
    }
}

/// A detected object entry for caption generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionObject {
    /// Object category label.
    pub label: String,
    /// Confidence (0.0–1.0).
    pub confidence: f32,
    /// Approximate count of this object in frame.
    pub count: u32,
}

impl CaptionObject {
    /// Create a new caption object.
    pub fn new(label: impl Into<String>, confidence: f32, count: u32) -> Self {
        Self {
            label: label.into(),
            confidence: confidence.clamp(0.0, 1.0),
            count,
        }
    }
}

/// Composition style for caption context.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionStyle {
    /// Subject centred in frame.
    Centred,
    /// Rule-of-thirds placement.
    RuleOfThirds,
    /// Strongly symmetrical composition.
    Symmetrical,
    /// Leading lines drawing eye.
    LeadingLines,
    /// Cluttered or complex composition.
    Complex,
    /// No dominant composition pattern.
    Undefined,
}

impl CompositionStyle {
    /// Phrase fragment describing the composition.
    #[must_use]
    pub const fn phrase(&self) -> &'static str {
        match self {
            Self::Centred => "with a centred subject",
            Self::RuleOfThirds => "composed using the rule of thirds",
            Self::Symmetrical => "with strong symmetry",
            Self::LeadingLines => "with leading lines guiding the eye",
            Self::Complex => "with a complex, layered composition",
            Self::Undefined => "",
        }
    }
}

/// Input descriptor for the scene captioner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionInput {
    /// Scene environment context.
    pub environment: SceneEnvironment,
    /// Dominant subject/objects detected.
    pub objects: Vec<CaptionObject>,
    /// Normalised motion energy (0.0–1.0).
    pub motion_energy: f32,
    /// Composition style hint.
    pub composition: CompositionStyle,
    /// Optional mood descriptor (e.g. "tense", "serene", "joyful").
    pub mood: Option<String>,
    /// Optional activity description (e.g. "running", "conversation").
    pub activity: Option<String>,
    /// Number of visible people (-1 = unknown).
    pub person_count: i32,
}

impl CaptionInput {
    /// Create a minimal caption input.
    #[must_use]
    pub fn new(environment: SceneEnvironment) -> Self {
        Self {
            environment,
            objects: Vec::new(),
            motion_energy: 0.0,
            composition: CompositionStyle::Undefined,
            mood: None,
            activity: None,
            person_count: -1,
        }
    }

    /// Add a detected object.
    #[must_use]
    pub fn with_object(mut self, obj: CaptionObject) -> Self {
        self.objects.push(obj);
        self
    }

    /// Set normalised motion energy.
    #[must_use]
    pub fn with_motion_energy(mut self, energy: f32) -> Self {
        self.motion_energy = energy.clamp(0.0, 1.0);
        self
    }

    /// Set composition style.
    #[must_use]
    pub fn with_composition(mut self, style: CompositionStyle) -> Self {
        self.composition = style;
        self
    }

    /// Set mood hint.
    #[must_use]
    pub fn with_mood(mut self, mood: impl Into<String>) -> Self {
        self.mood = Some(mood.into());
        self
    }

    /// Set activity hint.
    #[must_use]
    pub fn with_activity(mut self, activity: impl Into<String>) -> Self {
        self.activity = Some(activity.into());
        self
    }

    /// Set person count.
    #[must_use]
    pub fn with_person_count(mut self, count: i32) -> Self {
        self.person_count = count;
        self
    }
}

/// A generated caption at a specified granularity.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Caption {
    /// Caption text.
    pub text: String,
    /// Granularity used.
    pub granularity: CaptionGranularity,
    /// Estimated confidence in caption quality (0.0–1.0).
    pub confidence: f32,
}

impl Caption {
    fn new(text: String, granularity: CaptionGranularity, confidence: f32) -> Self {
        Self {
            text,
            granularity,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }
}

impl fmt::Display for Caption {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

/// Configuration for the scene captioner.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptionerConfig {
    /// Minimum object confidence to include in caption.
    pub min_object_confidence: f32,
    /// Maximum number of distinct objects to mention.
    pub max_objects_mentioned: usize,
    /// Whether to include composition details in sentence/paragraph modes.
    pub include_composition: bool,
    /// Whether to include mood/atmosphere details.
    pub include_mood: bool,
}

impl Default for CaptionerConfig {
    fn default() -> Self {
        Self {
            min_object_confidence: 0.4,
            max_objects_mentioned: 4,
            include_composition: true,
            include_mood: true,
        }
    }
}

impl CaptionerConfig {
    /// Validate configuration parameters.
    pub fn validate(&self) -> SceneResult<()> {
        if self.min_object_confidence < 0.0 || self.min_object_confidence > 1.0 {
            return Err(SceneError::InvalidParameter(
                "min_object_confidence must be in [0.0, 1.0]".into(),
            ));
        }
        if self.max_objects_mentioned == 0 {
            return Err(SceneError::InvalidParameter(
                "max_objects_mentioned must be >= 1".into(),
            ));
        }
        Ok(())
    }
}

/// Scene captioner: generates natural-language captions from structured scene descriptors.
///
/// All text generation is template-driven and deterministic — no neural LM required.
pub struct SceneCaptioner {
    config: CaptionerConfig,
}

impl Default for SceneCaptioner {
    fn default() -> Self {
        Self::new()
    }
}

impl SceneCaptioner {
    /// Create with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: CaptionerConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: CaptionerConfig) -> SceneResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Generate a caption at the requested granularity.
    pub fn caption(&self, input: &CaptionInput, granularity: CaptionGranularity) -> Caption {
        match granularity {
            CaptionGranularity::Brief => self.generate_brief(input),
            CaptionGranularity::Sentence => self.generate_sentence(input),
            CaptionGranularity::Paragraph => self.generate_paragraph(input),
        }
    }

    /// Generate all granularity levels at once.
    pub fn caption_all(&self, input: &CaptionInput) -> [Caption; 3] {
        [
            self.caption(input, CaptionGranularity::Brief),
            self.caption(input, CaptionGranularity::Sentence),
            self.caption(input, CaptionGranularity::Paragraph),
        ]
    }

    // ── internal generators ──────────────────────────────────────────────────

    fn confident_objects<'a>(&self, input: &'a CaptionInput) -> Vec<&'a CaptionObject> {
        let mut objs: Vec<&CaptionObject> = input
            .objects
            .iter()
            .filter(|o| o.confidence >= self.config.min_object_confidence)
            .collect();
        // Sort by confidence descending, then trim.
        objs.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        objs.truncate(self.config.max_objects_mentioned);
        objs
    }

    fn generate_brief(&self, input: &CaptionInput) -> Caption {
        let motion = MotionDescriptor::from_energy(input.motion_energy);
        let text = if let Some(activity) = &input.activity {
            activity.clone()
        } else {
            let env = match input.environment {
                SceneEnvironment::OutdoorDay => "outdoor scene",
                SceneEnvironment::OutdoorNight => "night scene",
                SceneEnvironment::Indoor => "indoor scene",
                SceneEnvironment::Studio => "studio shot",
                SceneEnvironment::Unknown => "scene",
            };
            format!("{} {}", motion.adjective(), env)
        };
        Caption::new(text, CaptionGranularity::Brief, 0.7)
    }

    fn generate_sentence(&self, input: &CaptionInput) -> Caption {
        let motion = MotionDescriptor::from_energy(input.motion_energy);
        let objs = self.confident_objects(input);

        // Subject fragment
        let subject = self.build_subject_fragment(input, &objs);

        // Action / motion fragment
        let action = if let Some(act) = &input.activity {
            format!(", {}", act)
        } else if motion != MotionDescriptor::Static {
            format!(", moving {}", motion.adverb())
        } else {
            String::new()
        };

        // Environment
        let env_frag = format!(" in {}", input.environment.description());

        // Composition
        let comp_frag = if self.config.include_composition {
            let phrase = input.composition.phrase();
            if phrase.is_empty() {
                String::new()
            } else {
                format!(", {}", phrase)
            }
        } else {
            String::new()
        };

        let text = format!("{}{}{}{}", subject, action, env_frag, comp_frag);
        // Capitalise first letter
        let text = capitalise_first(&text);
        Caption::new(text, CaptionGranularity::Sentence, 0.75)
    }

    fn generate_paragraph(&self, input: &CaptionInput) -> Caption {
        let sentence_caption = self.generate_sentence(input);
        let mut parts: Vec<String> = vec![sentence_caption.text.clone()];

        let objs = self.confident_objects(input);

        // Object detail sentence
        if objs.len() > 1 {
            let obj_labels: Vec<String> = objs
                .iter()
                .map(|o| {
                    if o.count > 1 {
                        format!("{} {}s", o.count, o.label)
                    } else {
                        o.label.clone()
                    }
                })
                .collect();
            let obj_text = join_with_oxford_comma(&obj_labels);
            parts.push(format!("The scene contains {}.", obj_text));
        }

        // Motion detail sentence
        let motion = MotionDescriptor::from_energy(input.motion_energy);
        let motion_sentence = match motion {
            MotionDescriptor::Static => {
                "The frame is largely static with minimal movement.".to_string()
            }
            MotionDescriptor::Slow => "Motion is gentle and unhurried.".to_string(),
            MotionDescriptor::Moderate => {
                "There is moderate movement throughout the frame.".to_string()
            }
            MotionDescriptor::Fast => "The scene features fast, dynamic motion.".to_string(),
            MotionDescriptor::Rapid => {
                "Rapid, high-energy movement dominates the frame.".to_string()
            }
        };
        parts.push(motion_sentence);

        // Mood sentence
        if self.config.include_mood {
            if let Some(mood) = &input.mood {
                parts.push(format!("The overall atmosphere is {}.", mood));
            }
        }

        let text = parts.join(" ");
        Caption::new(text, CaptionGranularity::Paragraph, 0.8)
    }

    fn build_subject_fragment<'a>(
        &self,
        input: &CaptionInput,
        objs: &[&'a CaptionObject],
    ) -> String {
        if input.person_count > 0 {
            let person_label = if input.person_count == 1 {
                "A person".to_string()
            } else {
                format!("{} people", input.person_count)
            };
            return person_label;
        }
        if let Some(first_obj) = objs.first() {
            let article = if starts_with_vowel(&first_obj.label) {
                "An"
            } else {
                "A"
            };
            return format!("{} {}", article, first_obj.label);
        }
        // Fallback to environment
        capitalise_first(input.environment.description())
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn capitalise_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        None => String::new(),
        Some(c) => c.to_uppercase().to_string() + chars.as_str(),
    }
}

fn starts_with_vowel(s: &str) -> bool {
    matches!(
        s.chars().next().map(|c| c.to_ascii_lowercase()),
        Some('a' | 'e' | 'i' | 'o' | 'u')
    )
}

fn join_with_oxford_comma(items: &[String]) -> String {
    match items.len() {
        0 => String::new(),
        1 => items[0].clone(),
        2 => format!("{} and {}", items[0], items[1]),
        _ => {
            let all_but_last = items[..items.len() - 1].join(", ");
            format!("{}, and {}", all_but_last, items[items.len() - 1])
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input() -> CaptionInput {
        CaptionInput::new(SceneEnvironment::OutdoorDay)
            .with_motion_energy(0.3)
            .with_composition(CompositionStyle::RuleOfThirds)
            .with_mood("serene")
            .with_activity("a group of people walking")
            .with_person_count(3)
            .with_object(CaptionObject::new("dog", 0.85, 1))
            .with_object(CaptionObject::new("tree", 0.72, 2))
    }

    #[test]
    fn test_brief_caption_with_activity() {
        let captioner = SceneCaptioner::new();
        let input = make_input();
        let cap = captioner.caption(&input, CaptionGranularity::Brief);
        assert!(!cap.text.is_empty());
        assert_eq!(cap.granularity, CaptionGranularity::Brief);
    }

    #[test]
    fn test_brief_caption_without_activity() {
        let captioner = SceneCaptioner::new();
        let input = CaptionInput::new(SceneEnvironment::Indoor).with_motion_energy(0.0);
        let cap = captioner.caption(&input, CaptionGranularity::Brief);
        assert!(cap.text.contains("static") || cap.text.contains("indoor"));
    }

    #[test]
    fn test_sentence_caption_contains_environment() {
        let captioner = SceneCaptioner::new();
        let input = CaptionInput::new(SceneEnvironment::OutdoorNight).with_person_count(1);
        let cap = captioner.caption(&input, CaptionGranularity::Sentence);
        assert!(
            cap.text.to_lowercase().contains("night")
                || cap.text.to_lowercase().contains("outdoor")
        );
    }

    #[test]
    fn test_paragraph_caption_multi_sentence() {
        let captioner = SceneCaptioner::new();
        let input = make_input();
        let cap = captioner.caption(&input, CaptionGranularity::Paragraph);
        // Paragraph should contain at least two sentences
        let sentence_count = cap.text.matches('.').count();
        assert!(
            sentence_count >= 2,
            "Expected >= 2 sentences, got {sentence_count}"
        );
    }

    #[test]
    fn test_caption_all_returns_three() {
        let captioner = SceneCaptioner::new();
        let input = make_input();
        let caps = captioner.caption_all(&input);
        assert_eq!(caps.len(), 3);
        assert_eq!(caps[0].granularity, CaptionGranularity::Brief);
        assert_eq!(caps[1].granularity, CaptionGranularity::Sentence);
        assert_eq!(caps[2].granularity, CaptionGranularity::Paragraph);
    }

    #[test]
    fn test_motion_descriptor_from_energy() {
        assert_eq!(MotionDescriptor::from_energy(0.0), MotionDescriptor::Static);
        assert_eq!(MotionDescriptor::from_energy(0.1), MotionDescriptor::Slow);
        assert_eq!(
            MotionDescriptor::from_energy(0.3),
            MotionDescriptor::Moderate
        );
        assert_eq!(MotionDescriptor::from_energy(0.6), MotionDescriptor::Fast);
        assert_eq!(MotionDescriptor::from_energy(0.95), MotionDescriptor::Rapid);
    }

    #[test]
    fn test_config_validation_error() {
        let bad_config = CaptionerConfig {
            min_object_confidence: 1.5,
            ..Default::default()
        };
        assert!(bad_config.validate().is_err());

        let zero_obj_config = CaptionerConfig {
            max_objects_mentioned: 0,
            ..Default::default()
        };
        assert!(zero_obj_config.validate().is_err());
    }

    #[test]
    fn test_oxford_comma() {
        assert_eq!(join_with_oxford_comma(&[]), "");
        assert_eq!(join_with_oxford_comma(&["a".to_string()]), "a");
        assert_eq!(
            join_with_oxford_comma(&["a".to_string(), "b".to_string()]),
            "a and b"
        );
        assert_eq!(
            join_with_oxford_comma(&["a".to_string(), "b".to_string(), "c".to_string()]),
            "a, b, and c"
        );
    }

    #[test]
    fn test_object_confidence_filter() {
        let config = CaptionerConfig {
            min_object_confidence: 0.8,
            ..Default::default()
        };
        let captioner = SceneCaptioner::with_config(config).expect("valid config");
        let input = CaptionInput::new(SceneEnvironment::Studio)
            .with_object(CaptionObject::new("chair", 0.9, 1))
            .with_object(CaptionObject::new("table", 0.5, 1)); // below threshold
        let cap = captioner.caption(&input, CaptionGranularity::Paragraph);
        // "table" should not appear (below threshold)
        assert!(!cap.text.to_lowercase().contains("table"));
    }

    #[test]
    fn test_caption_display() {
        let cap = Caption::new(
            "A sunny scene.".to_string(),
            CaptionGranularity::Sentence,
            0.9,
        );
        assert_eq!(format!("{}", cap), "A sunny scene.");
    }
}
