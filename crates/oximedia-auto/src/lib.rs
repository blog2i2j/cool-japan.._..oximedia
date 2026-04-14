//! Automated video editing for `OxiMedia`.
//!
//! `oximedia-auto` provides a comprehensive automated video editing system with:
//!
//! - **Highlight Detection**: Automatically identify exciting moments
//! - **Smart Cutting**: Intelligent shot boundary and transition detection
//! - **Auto-Assembly**: Generate highlight reels, trailers, and social clips
//! - **Rules Engine**: Configurable editing constraints and preferences
//! - **Scene Scoring**: AI-powered importance analysis
//!
//! # Architecture
//!
//! The automated editing system is built around these core components:
//!
//! ## Highlight Detection
//!
//! The [`highlights`] module detects important moments using:
//! - Motion intensity analysis
//! - Face detection and tracking
//! - Audio peak detection (cheers, applause)
//! - Object detection integration
//! - Multi-factor scoring
//!
//! ## Smart Cutting
//!
//! The [`cuts`] module provides intelligent cutting:
//! - Shot boundary detection
//! - Transition recommendations
//! - Beat detection for music sync
//! - Dialogue-aware cutting
//! - Jump cut removal
//!
//! ## Auto-Assembly
//!
//! The [`assembly`] module assembles final edits:
//! - Highlight reel generation
//! - Trailer creation
//! - Social media clips (15s, 30s, 60s)
//! - Best moments extraction
//! - Automatic pacing and dramatic arc
//!
//! ## Rules Engine
//!
//! The [`rules`] module enforces editing rules:
//! - Shot duration constraints
//! - Transition preferences
//! - Music synchronization
//! - Aspect ratio adaptation
//! - Pacing presets
//!
//! ## Scene Scoring
//!
//! The [`scoring`] module analyzes content:
//! - Multi-feature scoring
//! - Content classification
//! - Sentiment analysis
//! - Interest curve generation
//! - Auto-titling suggestions
//!
//! # Example Usage
//!
//! ```no_run
//! use oximedia_auto::{AutoEditor, AutoEditorConfig};
//! use oximedia_auto::assembly::AssemblyType;
//! use oximedia_auto::rules::PacingPreset;
//!
//! // Create an auto editor for highlight reels
//! let config = AutoEditorConfig::default()
//!     .with_assembly_type(AssemblyType::HighlightReel)
//!     .with_target_duration_ms(60_000)  // 60 seconds
//!     .with_pacing(PacingPreset::Fast);
//!
//! let editor = AutoEditor::new(config);
//!
//! // Process video to generate a highlight reel
//! // let highlights = editor.process_video(&frames, &audio)?;
//! ```
//!
//! # Social Media Clips
//!
//! Generate platform-optimized short clips:
//!
//! ```no_run
//! use oximedia_auto::{AutoEditor, AutoEditorConfig};
//! use oximedia_auto::assembly::AssemblyType;
//! use oximedia_auto::rules::AspectRatio;
//!
//! // Configure for TikTok/Reels (vertical 9:16)
//! let config = AutoEditorConfig::default()
//!     .with_assembly_type(AssemblyType::SocialClip)
//!     .with_target_duration_ms(30_000)  // 30 seconds
//!     .with_aspect_ratio(AspectRatio::Vertical9x16);
//!
//! let editor = AutoEditor::new(config);
//! ```
//!
//! # Trailers
//!
//! Create compelling video trailers:
//!
//! ```no_run
//! use oximedia_auto::{AutoEditor, AutoEditorConfig};
//! use oximedia_auto::assembly::AssemblyType;
//!
//! let config = AutoEditorConfig::default()
//!     .with_assembly_type(AssemblyType::Trailer)
//!     .with_target_duration_ms(90_000)   // 90 seconds
//!     .with_dramatic_arc(true);
//!
//! let editor = AutoEditor::new(config);
//! ```
//!
//! # Green List Only
//!
//! Like all `OxiMedia` components, `oximedia-auto` only supports patent-free
//! codecs (AV1, VP9, VP8, Opus, Vorbis, FLAC). Attempting to use patent-
//! encumbered codecs will result in errors.

#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::must_use_candidate)]
#![allow(dead_code)]

pub mod a_b_roll;
pub mod assembly;
pub mod audio_sync_align;
pub mod auto_caption;
pub mod auto_chaptering;
pub mod auto_grade;
pub mod auto_thumbnail;
pub mod av_sync;
pub mod b_roll_selector;
pub mod batch_auto;
pub mod chapter_generator;
pub mod color_continuity;
pub mod color_grade_suggest;
pub mod color_match;
pub mod color_suggest;
pub mod content_index;
pub mod content_indexer;
pub mod content_warning;
pub mod cuts;
pub mod engagement_predictor;
pub mod error;
pub mod highlight_reel;
pub mod highlights;
pub mod montage_builder;
pub mod multi_pass_highlight;
pub mod music_sync;
pub mod music_video_sync;
pub mod narrative;
pub mod narrative_structure;
pub mod pacing;
pub mod pacing_curve;
pub mod pacing_editor;
pub mod quality_gate;
pub mod reframe_analyzer;
pub mod rhythm_cutter;
pub mod rules;
pub mod scene_classifier;
pub mod scene_description;
pub mod scene_detect_auto;
pub mod scene_reorder;
pub mod scoring;
pub mod segment_merge;
pub mod silence_detector;
pub mod smart_crop;
pub mod smart_reframe;
pub mod smart_trim;
pub mod social_clip_formatter;
pub mod subtitle_sync;
pub mod tag_suggest;
pub mod tempo_detect;
pub mod temporal_scorer;
pub mod title_card_generator;
pub mod transition_suggest;
pub mod visual_theme;
pub mod workflow_auto;

// Re-export commonly used items
pub use assembly::{AssembledClip, AssemblyConfig, AssemblyType, AutoAssembler};
pub use cuts::{
    Beat, CutConfig, CutDetector, CutPoint, CutType, DialogueSegment, JumpCutConfig, ShotConfig,
};
pub use error::{AutoError, AutoResult};
pub use highlights::{
    AudioPeakConfig, FaceConfig, Highlight, HighlightConfig, HighlightDetector, HighlightType,
    MotionConfig, ObjectConfig,
};
pub use rules::{
    AspectRatio, EditRules, MusicSyncMode, PacingPreset, RulesEngine, ShotConstraints,
    TransitionPreferences,
};
pub use scoring::{
    ContentType, FeatureWeights, ImportanceScore, InterestCurve, SceneFeatures, SceneScorer,
    ScoredScene, ScoringConfig, Sentiment,
};

use oximedia_codec::VideoFrame;
use oximedia_core::Timestamp;

/// Version information.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Complete configuration for automated video editing.
#[derive(Debug, Clone, Default)]
pub struct AutoEditorConfig {
    /// Highlight detection configuration.
    pub highlight_config: HighlightConfig,
    /// Cut detection configuration.
    pub cut_config: CutConfig,
    /// Assembly configuration.
    pub assembly_config: AssemblyConfig,
    /// Editing rules.
    pub rules: EditRules,
    /// Scoring configuration.
    pub scoring_config: ScoringConfig,
}

impl AutoEditorConfig {
    /// Create a new auto editor configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the assembly type.
    #[must_use]
    pub fn with_assembly_type(mut self, assembly_type: AssemblyType) -> Self {
        self.assembly_config.assembly_type = assembly_type;
        self.assembly_config.target_aspect_ratio = assembly_type.recommended_aspect_ratio();
        let (min, max) = assembly_type.typical_duration_range_ms();
        self.assembly_config.target_duration_ms = (min + max) / 2;
        self
    }

    /// Set the target duration in milliseconds.
    #[must_use]
    pub const fn with_target_duration_ms(mut self, duration_ms: i64) -> Self {
        self.assembly_config.target_duration_ms = duration_ms;
        self
    }

    /// Set the aspect ratio.
    #[must_use]
    pub const fn with_aspect_ratio(mut self, ratio: AspectRatio) -> Self {
        self.assembly_config.target_aspect_ratio = ratio;
        self.rules.target_aspect_ratio = ratio;
        self
    }

    /// Set the pacing preset.
    #[must_use]
    pub fn with_pacing(mut self, pacing: PacingPreset) -> Self {
        self.rules = self.rules.with_pacing(pacing);
        self
    }

    /// Enable or disable dramatic arc.
    #[must_use]
    pub const fn with_dramatic_arc(mut self, enabled: bool) -> Self {
        self.assembly_config.use_dramatic_arc = enabled;
        self
    }

    /// Set music synchronization mode.
    #[must_use]
    pub fn with_music_sync(mut self, sync_mode: MusicSyncMode) -> Self {
        self.rules = self.rules.with_music_sync(sync_mode);
        self.cut_config.beat.enabled = !matches!(sync_mode, MusicSyncMode::None);
        self.cut_config.prefer_beat_cuts = !matches!(sync_mode, MusicSyncMode::None);
        self
    }

    /// Create a configuration for a specific use case.
    #[must_use]
    pub fn for_use_case(use_case: &str) -> Self {
        let rules = EditRules::for_use_case(use_case);
        let mut config = Self::default();
        config.rules = rules;

        match use_case.to_lowercase().as_str() {
            "trailer" => {
                config.assembly_config = AssemblyConfig::for_type(AssemblyType::Trailer);
            }
            "highlights" => {
                config.assembly_config = AssemblyConfig::for_type(AssemblyType::HighlightReel);
            }
            "social" => {
                config.assembly_config = AssemblyConfig::for_type(AssemblyType::SocialClip);
            }
            _ => {}
        }

        config
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        self.highlight_config.validate()?;
        self.cut_config.validate()?;
        self.assembly_config.validate()?;
        self.rules.validate()?;
        self.scoring_config.validate()?;
        Ok(())
    }
}

/// The main automated video editor.
///
/// This combines all the automated editing components into a single high-level API.
pub struct AutoEditor {
    /// Configuration.
    config: AutoEditorConfig,
    /// Highlight detector.
    highlight_detector: HighlightDetector,
    /// Cut detector.
    cut_detector: CutDetector,
    /// Auto assembler.
    assembler: AutoAssembler,
    /// Rules engine.
    rules_engine: RulesEngine,
    /// Scene scorer.
    scorer: SceneScorer,
}

impl AutoEditor {
    /// Create a new auto editor with the given configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_auto::{AutoEditor, AutoEditorConfig};
    ///
    /// let config = AutoEditorConfig::default();
    /// let editor = AutoEditor::new(config);
    /// ```
    #[must_use]
    pub fn new(config: AutoEditorConfig) -> Self {
        Self {
            highlight_detector: HighlightDetector::new(config.highlight_config.clone()),
            cut_detector: CutDetector::new(config.cut_config.clone()),
            assembler: AutoAssembler::new(config.assembly_config.clone()),
            rules_engine: RulesEngine::new(config.rules.clone()),
            scorer: SceneScorer::new(config.scoring_config.clone()),
            config,
        }
    }

    /// Create an auto editor with default configuration.
    #[must_use]
    pub fn default_editor() -> Self {
        Self::new(AutoEditorConfig::default())
    }

    /// Create an auto editor for a specific use case.
    ///
    /// Supported use cases:
    /// - "trailer" - Movie trailer preset
    /// - "highlights" - Highlight reel preset
    /// - "social" - Social media preset
    /// - "documentary" - Documentary preset
    /// - "`music_video`" - Music video preset
    #[must_use]
    pub fn for_use_case(use_case: &str) -> Self {
        Self::new(AutoEditorConfig::for_use_case(use_case))
    }

    /// Process video frames to generate highlights.
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails or configuration is invalid.
    pub fn detect_highlights(&self, frames: &[VideoFrame]) -> AutoResult<Vec<Highlight>> {
        self.config.validate()?;
        self.highlight_detector.detect_highlights(frames)
    }

    /// Detect audio highlights from audio samples.
    ///
    /// # Errors
    ///
    /// Returns an error if detection fails.
    pub fn detect_audio_highlights(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
    ) -> AutoResult<Vec<Highlight>> {
        self.highlight_detector
            .detect_audio_highlights(audio_samples, sample_rate)
    }

    /// Score scenes for importance.
    ///
    /// # Errors
    ///
    /// Returns an error if scoring fails.
    pub fn score_scenes(
        &self,
        scene_data: &[(Timestamp, Timestamp, SceneFeatures)],
    ) -> AutoResult<Vec<ScoredScene>> {
        scoring::batch_score_scenes(&self.scorer, scene_data)
    }

    /// Generate an interest curve from scored scenes.
    #[must_use]
    pub fn generate_interest_curve(&self, scenes: &[ScoredScene]) -> InterestCurve {
        self.scorer.generate_interest_curve(scenes)
    }

    /// Detect cut points using scene changes.
    ///
    /// # Errors
    ///
    /// Returns an error if cut detection fails.
    pub fn detect_cuts(
        &self,
        scene_changes: &[oximedia_cv::scene::SceneChange],
        beats: Option<&[Beat]>,
        dialogue: Option<&[DialogueSegment]>,
    ) -> AutoResult<Vec<CutPoint>> {
        self.cut_detector
            .detect_cuts(scene_changes, beats, dialogue)
    }

    /// Detect beats in audio.
    ///
    /// # Errors
    ///
    /// Returns an error if beat detection fails.
    pub fn detect_beats(&self, audio_samples: &[f32], sample_rate: u32) -> AutoResult<Vec<Beat>> {
        self.cut_detector.detect_beats(audio_samples, sample_rate)
    }

    /// Detect dialogue segments in audio.
    ///
    /// # Errors
    ///
    /// Returns an error if dialogue detection fails.
    pub fn detect_dialogue(
        &self,
        audio_samples: &[f32],
        sample_rate: u32,
    ) -> AutoResult<Vec<DialogueSegment>> {
        self.cut_detector
            .detect_dialogue(audio_samples, sample_rate)
    }

    /// Assemble a video from scored scenes.
    ///
    /// # Errors
    ///
    /// Returns an error if assembly fails.
    pub fn assemble_from_scenes(&self, scenes: &[ScoredScene]) -> AutoResult<Vec<AssembledClip>> {
        self.assembler.assemble_from_scenes(scenes)
    }

    /// Assemble a video from highlights.
    ///
    /// # Errors
    ///
    /// Returns an error if assembly fails.
    pub fn assemble_from_highlights(
        &self,
        highlights: &[Highlight],
    ) -> AutoResult<Vec<AssembledClip>> {
        self.assembler.assemble_from_highlights(highlights)
    }

    /// Generate a social media clip of specified duration.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn generate_social_clip(
        &self,
        scenes: &[ScoredScene],
        duration_ms: i64,
    ) -> AutoResult<Vec<AssembledClip>> {
        self.assembler.generate_social_clip(scenes, duration_ms)
    }

    /// Generate a trailer.
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn generate_trailer(
        &self,
        scenes: &[ScoredScene],
        cuts: &[CutPoint],
    ) -> AutoResult<Vec<AssembledClip>> {
        self.assembler.generate_trailer(scenes, cuts)
    }

    /// Apply editing rules to cut points.
    ///
    /// # Errors
    ///
    /// Returns an error if rule application fails.
    pub fn apply_rules(&self, cuts: &mut Vec<CutPoint>) -> AutoResult<()> {
        self.rules_engine.apply_rules(cuts)
    }

    /// Complete automated workflow: detect, score, cut, and assemble.
    ///
    /// # Errors
    ///
    /// Returns an error if any step fails.
    #[allow(clippy::too_many_arguments)]
    pub fn auto_edit(
        &self,
        frames: &[VideoFrame],
        audio_samples: Option<&[f32]>,
        sample_rate: Option<u32>,
        scene_changes: &[oximedia_cv::scene::SceneChange],
    ) -> AutoResult<AutoEditResult> {
        self.config.validate()?;

        // Step 1: Detect highlights from video
        let video_highlights = self.detect_highlights(frames)?;

        // Step 2: Detect audio highlights if audio is provided
        let audio_highlights = if let (Some(audio), Some(sr)) = (audio_samples, sample_rate) {
            self.detect_audio_highlights(audio, sr)?
        } else {
            Vec::new()
        };

        // Step 3: Combine all highlights
        let mut all_highlights = video_highlights;
        all_highlights.extend(audio_highlights);

        // Step 4: Convert highlights to scored scenes
        let scenes: Vec<ScoredScene> = all_highlights
            .iter()
            .map(|h| {
                let mut scene = ScoredScene::new(
                    h.start,
                    h.end,
                    h.weighted_score(),
                    ContentType::Unknown,
                    Sentiment::Neutral,
                );
                scene.features = h.features.clone();
                scene.suggested_title = Some(h.description.clone());
                scene
            })
            .collect();

        // Step 5: Generate interest curve
        let interest_curve = self.generate_interest_curve(&scenes);

        // Step 6: Detect beats if audio is provided
        let beats = if let (Some(audio), Some(sr)) = (audio_samples, sample_rate) {
            self.detect_beats(audio, sr).ok()
        } else {
            None
        };

        // Step 7: Detect dialogue if audio is provided
        let dialogue = if let (Some(audio), Some(sr)) = (audio_samples, sample_rate) {
            self.detect_dialogue(audio, sr).ok()
        } else {
            None
        };

        // Step 8: Detect cuts
        let mut cuts = self.detect_cuts(scene_changes, beats.as_deref(), dialogue.as_deref())?;

        // Step 9: Apply rules
        self.apply_rules(&mut cuts)?;

        // Step 10: Assemble final edit
        let assembled = self.assemble_from_scenes(&scenes)?;

        Ok(AutoEditResult {
            highlights: all_highlights,
            scenes,
            interest_curve,
            beats: beats.unwrap_or_default(),
            dialogue: dialogue.unwrap_or_default(),
            cuts,
            assembled,
        })
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &AutoEditorConfig {
        &self.config
    }

    /// Update the configuration.
    pub fn set_config(&mut self, config: AutoEditorConfig) {
        self.highlight_detector = HighlightDetector::new(config.highlight_config.clone());
        self.cut_detector = CutDetector::new(config.cut_config.clone());
        self.assembler = AutoAssembler::new(config.assembly_config.clone());
        self.rules_engine = RulesEngine::new(config.rules.clone());
        self.scorer = SceneScorer::new(config.scoring_config.clone());
        self.config = config;
    }
}

impl Default for AutoEditor {
    fn default() -> Self {
        Self::default_editor()
    }
}

/// Result of automated editing workflow.
#[derive(Debug, Clone)]
pub struct AutoEditResult {
    /// Detected highlights.
    pub highlights: Vec<Highlight>,
    /// Scored scenes.
    pub scenes: Vec<ScoredScene>,
    /// Interest curve.
    pub interest_curve: InterestCurve,
    /// Detected beats.
    pub beats: Vec<Beat>,
    /// Detected dialogue segments.
    pub dialogue: Vec<DialogueSegment>,
    /// Cut points.
    pub cuts: Vec<CutPoint>,
    /// Assembled clips.
    pub assembled: Vec<AssembledClip>,
}

impl AutoEditResult {
    /// Get the total duration of the assembled edit.
    #[must_use]
    pub fn total_duration_ms(&self) -> i64 {
        self.assembled.last().map_or(0, |clip| clip.output_end.pts)
    }

    /// Get the number of clips in the assembly.
    #[must_use]
    pub fn clip_count(&self) -> usize {
        self.assembled.len()
    }

    /// Get the number of detected highlights.
    #[must_use]
    pub fn highlight_count(&self) -> usize {
        self.highlights.len()
    }

    /// Get the average importance score.
    #[must_use]
    pub fn average_importance(&self) -> f64 {
        if self.scenes.is_empty() {
            return 0.0;
        }

        let total: f64 = self
            .scenes
            .iter()
            .map(scoring::ScoredScene::adjusted_score)
            .sum();
        total / self.scenes.len() as f64
    }
}

/// Prelude module for convenient imports.
pub mod prelude {
    //! Prelude module for convenient imports.

    pub use crate::assembly::{AssembledClip, AssemblyConfig, AssemblyType, AutoAssembler};
    pub use crate::cuts::{CutConfig, CutDetector, CutPoint, CutType};
    pub use crate::error::{AutoError, AutoResult};
    pub use crate::highlights::{Highlight, HighlightConfig, HighlightDetector, HighlightType};
    pub use crate::rules::{AspectRatio, EditRules, PacingPreset, RulesEngine};
    pub use crate::scoring::{
        ContentType, InterestCurve, SceneFeatures, SceneScorer, ScoredScene, ScoringConfig,
    };
    pub use crate::{AutoEditResult, AutoEditor, AutoEditorConfig};
}
