//! Auto-assembly for automated video editing.
//!
//! This module provides algorithms for automatically assembling video edits:
//!
//! - **Highlight reel generation**: Create exciting compilation videos
//! - **Trailer creation**: Assemble compelling previews
//! - **Social media clips**: Generate platform-optimized shorts
//! - **Best moments extraction**: Find and compile peak moments
//! - **Automatic pacing**: Intelligent timing and rhythm
//!
//! # Example
//!
//! ```
//! use oximedia_auto::assembly::{AutoAssembler, AssemblyConfig};
//!
//! let config = AssemblyConfig::default();
//! let assembler = AutoAssembler::new(config);
//! ```

use crate::cuts::CutPoint;
use crate::error::{AutoError, AutoResult};
use crate::highlights::Highlight;
use crate::rules::AspectRatio;
use crate::scoring::ScoredScene;
use oximedia_core::Timestamp;
use std::collections::HashMap;

/// Type of assembly output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AssemblyType {
    /// Full highlight reel.
    HighlightReel,
    /// Movie trailer.
    Trailer,
    /// Social media clip (short form).
    SocialClip,
    /// Best moments compilation.
    BestMoments,
    /// Recap or summary.
    Recap,
    /// Custom assembly.
    Custom,
}

impl AssemblyType {
    /// Get the typical target duration range for this assembly type.
    #[must_use]
    pub const fn typical_duration_range_ms(&self) -> (i64, i64) {
        match self {
            Self::HighlightReel => (30_000, 120_000), // 30s - 2min
            Self::Trailer => (60_000, 150_000),       // 1min - 2.5min
            Self::SocialClip => (15_000, 60_000),     // 15s - 1min
            Self::BestMoments => (45_000, 180_000),   // 45s - 3min
            Self::Recap => (30_000, 90_000),          // 30s - 1.5min
            Self::Custom => (15_000, 300_000),        // 15s - 5min
        }
    }

    /// Get the recommended aspect ratio for this assembly type.
    #[must_use]
    pub const fn recommended_aspect_ratio(&self) -> AspectRatio {
        match self {
            Self::SocialClip => AspectRatio::Vertical9x16,
            Self::Trailer => AspectRatio::Landscape16x9,
            Self::HighlightReel => AspectRatio::Landscape16x9,
            Self::BestMoments => AspectRatio::Landscape16x9,
            Self::Recap => AspectRatio::Landscape16x9,
            Self::Custom => AspectRatio::Landscape16x9,
        }
    }
}

/// An assembled clip segment.
#[derive(Debug, Clone)]
pub struct AssembledClip {
    /// Start timestamp in source video.
    pub source_start: Timestamp,
    /// End timestamp in source video.
    pub source_end: Timestamp,
    /// Start timestamp in assembled output.
    pub output_start: Timestamp,
    /// End timestamp in assembled output.
    pub output_end: Timestamp,
    /// Importance score of this clip.
    pub importance: f64,
    /// Speed multiplier (1.0 = normal).
    pub speed: f64,
    /// Transition in.
    pub transition_in: Option<crate::cuts::CutType>,
    /// Transition out.
    pub transition_out: Option<crate::cuts::CutType>,
    /// Transition in duration (ms).
    pub transition_in_duration_ms: i64,
    /// Transition out duration (ms).
    pub transition_out_duration_ms: i64,
    /// Description of this clip.
    pub description: String,
    /// Additional metadata.
    pub metadata: HashMap<String, String>,
}

impl AssembledClip {
    /// Create a new assembled clip.
    #[must_use]
    pub fn new(
        source_start: Timestamp,
        source_end: Timestamp,
        output_start: Timestamp,
        output_end: Timestamp,
    ) -> Self {
        Self {
            source_start,
            source_end,
            output_start,
            output_end,
            importance: 0.5,
            speed: 1.0,
            transition_in: None,
            transition_out: None,
            transition_in_duration_ms: 0,
            transition_out_duration_ms: 0,
            description: String::new(),
            metadata: HashMap::new(),
        }
    }

    /// Get the source duration in milliseconds.
    #[must_use]
    pub fn source_duration_ms(&self) -> i64 {
        (self.source_end.pts - self.source_start.pts).max(0)
    }

    /// Get the output duration in milliseconds.
    #[must_use]
    pub fn output_duration_ms(&self) -> i64 {
        (self.output_end.pts - self.output_start.pts).max(0)
    }

    /// Set the speed multiplier.
    #[must_use]
    pub fn with_speed(mut self, speed: f64) -> Self {
        self.speed = speed.max(0.1).min(10.0);
        self
    }

    /// Set the importance score.
    #[must_use]
    pub fn with_importance(mut self, importance: f64) -> Self {
        self.importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Set the description.
    #[must_use]
    pub fn with_description<S: Into<String>>(mut self, description: S) -> Self {
        self.description = description.into();
        self
    }
}

/// Configuration for assembly.
#[derive(Debug, Clone)]
pub struct AssemblyConfig {
    /// Type of assembly to create.
    pub assembly_type: AssemblyType,
    /// Target duration in milliseconds.
    pub target_duration_ms: i64,
    /// Target aspect ratio.
    pub target_aspect_ratio: AspectRatio,
    /// Minimum clip duration in milliseconds.
    pub min_clip_duration_ms: i64,
    /// Maximum clip duration in milliseconds.
    pub max_clip_duration_ms: i64,
    /// Minimum importance threshold for clips.
    pub min_importance: f64,
    /// Number of clips to include (0 = auto).
    pub target_clip_count: usize,
    /// Allow speed ramping (slow-mo/time-lapse).
    pub allow_speed_changes: bool,
    /// Add intro/outro.
    pub add_bookends: bool,
    /// Intro duration in milliseconds.
    pub intro_duration_ms: i64,
    /// Outro duration in milliseconds.
    pub outro_duration_ms: i64,
    /// Sort clips by timestamp (true) or importance (false).
    pub chronological_order: bool,
    /// Build up to a climax.
    pub use_dramatic_arc: bool,
}

impl Default for AssemblyConfig {
    fn default() -> Self {
        Self {
            assembly_type: AssemblyType::HighlightReel,
            target_duration_ms: 60_000, // 1 minute
            target_aspect_ratio: AspectRatio::Landscape16x9,
            min_clip_duration_ms: 1000,   // 1 second
            max_clip_duration_ms: 10_000, // 10 seconds
            min_importance: 0.5,
            target_clip_count: 0, // Auto-determine
            allow_speed_changes: false,
            add_bookends: false,
            intro_duration_ms: 0,
            outro_duration_ms: 0,
            chronological_order: true,
            use_dramatic_arc: true,
        }
    }
}

impl AssemblyConfig {
    /// Create a new assembly configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a configuration for a specific assembly type.
    #[must_use]
    pub fn for_type(assembly_type: AssemblyType) -> Self {
        let (min_dur, max_dur) = assembly_type.typical_duration_range_ms();
        let target = (min_dur + max_dur) / 2;

        Self {
            assembly_type,
            target_duration_ms: target,
            target_aspect_ratio: assembly_type.recommended_aspect_ratio(),
            ..Self::default()
        }
    }

    /// Set the target duration.
    #[must_use]
    pub const fn with_duration(mut self, duration_ms: i64) -> Self {
        self.target_duration_ms = duration_ms;
        self
    }

    /// Set the target clip count.
    #[must_use]
    pub const fn with_clip_count(mut self, count: usize) -> Self {
        self.target_clip_count = count;
        self
    }

    /// Set the minimum importance threshold.
    #[must_use]
    pub fn with_min_importance(mut self, importance: f64) -> Self {
        self.min_importance = importance.clamp(0.0, 1.0);
        self
    }

    /// Enable or disable chronological order.
    #[must_use]
    pub const fn with_chronological(mut self, chronological: bool) -> Self {
        self.chronological_order = chronological;
        self
    }

    /// Enable or disable dramatic arc.
    #[must_use]
    pub const fn with_dramatic_arc(mut self, use_arc: bool) -> Self {
        self.use_dramatic_arc = use_arc;
        self
    }

    /// Validate the configuration.
    pub fn validate(&self) -> AutoResult<()> {
        if self.target_duration_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.target_duration_ms,
            });
        }

        if self.min_clip_duration_ms <= 0 {
            return Err(AutoError::InvalidDuration {
                duration_ms: self.min_clip_duration_ms,
            });
        }

        if self.max_clip_duration_ms <= self.min_clip_duration_ms {
            return Err(AutoError::invalid_parameter(
                "max_clip_duration",
                "must be greater than min_clip_duration",
            ));
        }

        if !(0.0..=1.0).contains(&self.min_importance) {
            return Err(AutoError::InvalidThreshold {
                threshold: self.min_importance,
                min: 0.0,
                max: 1.0,
            });
        }

        Ok(())
    }
}

/// Automatic video assembler.
pub struct AutoAssembler {
    /// Configuration.
    config: AssemblyConfig,
}

impl AutoAssembler {
    /// Create a new auto assembler.
    #[must_use]
    pub fn new(config: AssemblyConfig) -> Self {
        Self { config }
    }

    /// Create an assembler with default configuration.
    #[must_use]
    pub fn default_assembler() -> Self {
        Self::new(AssemblyConfig::default())
    }

    /// Assemble a highlight reel from scored scenes.
    ///
    /// # Errors
    ///
    /// Returns an error if assembly fails or configuration is invalid.
    pub fn assemble_from_scenes(&self, scenes: &[ScoredScene]) -> AutoResult<Vec<AssembledClip>> {
        self.config.validate()?;

        if scenes.is_empty() {
            return Err(AutoError::insufficient_data("No scenes provided"));
        }

        // Filter scenes by minimum importance
        let mut filtered: Vec<_> = scenes
            .iter()
            .filter(|s| s.adjusted_score() >= self.config.min_importance)
            .collect();

        if filtered.is_empty() {
            return Err(AutoError::insufficient_data(
                "No scenes meet importance threshold",
            ));
        }

        // Sort by importance (descending)
        filtered.sort_by(|a, b| {
            b.adjusted_score()
                .partial_cmp(&a.adjusted_score())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Select clips to fit target duration
        let selected = self.select_clips_for_duration(&filtered)?;

        // Create assembled clips
        let mut assembled = Vec::new();
        let mut current_output_time = if self.config.add_bookends {
            self.config.intro_duration_ms
        } else {
            0
        };

        let timebase = oximedia_core::Rational::new(1, 1000);

        for scene in &selected {
            let source_duration = scene.duration();
            let clip_duration = source_duration.min(self.config.max_clip_duration_ms);

            let mut clip = AssembledClip::new(
                scene.start,
                Timestamp::new(scene.start.pts + clip_duration, scene.start.timebase),
                Timestamp::new(current_output_time, timebase),
                Timestamp::new(current_output_time + clip_duration, timebase),
            );

            clip = clip
                .with_importance(scene.adjusted_score())
                .with_description(
                    scene
                        .suggested_title
                        .clone()
                        .unwrap_or_else(|| "Highlight".to_string()),
                );

            assembled.push(clip);
            current_output_time += clip_duration;
        }

        // Sort by chronological order if configured
        if self.config.chronological_order {
            assembled.sort_by_key(|c| c.source_start.pts);

            // Recalculate output times
            let mut output_time = if self.config.add_bookends {
                self.config.intro_duration_ms
            } else {
                0
            };

            for clip in &mut assembled {
                let duration = clip.output_duration_ms();
                clip.output_start = Timestamp::new(output_time, timebase);
                clip.output_end = Timestamp::new(output_time + duration, timebase);
                output_time += duration;
            }
        }

        // Apply dramatic arc if configured
        if self.config.use_dramatic_arc {
            self.apply_dramatic_arc(&mut assembled);
        }

        Ok(assembled)
    }

    /// Assemble from highlights.
    ///
    /// # Errors
    ///
    /// Returns an error if assembly fails or configuration is invalid.
    pub fn assemble_from_highlights(
        &self,
        highlights: &[Highlight],
    ) -> AutoResult<Vec<AssembledClip>> {
        self.config.validate()?;

        if highlights.is_empty() {
            return Err(AutoError::insufficient_data("No highlights provided"));
        }

        // Convert highlights to scored scenes
        let scenes: Vec<ScoredScene> = highlights
            .iter()
            .map(|h| {
                let mut scene = ScoredScene::new(
                    h.start,
                    h.end,
                    h.weighted_score(),
                    crate::scoring::ContentType::Unknown,
                    crate::scoring::Sentiment::Neutral,
                );
                scene.features = h.features.clone();
                scene.suggested_title = Some(h.description.clone());
                scene
            })
            .collect();

        self.assemble_from_scenes(&scenes)
    }

    /// Select clips to fit the target duration.
    fn select_clips_for_duration<'a>(
        &self,
        candidates: &[&'a ScoredScene],
    ) -> AutoResult<Vec<&'a ScoredScene>> {
        let mut selected = Vec::new();
        let mut total_duration = 0i64;

        let available_duration = self.config.target_duration_ms
            - if self.config.add_bookends {
                self.config.intro_duration_ms + self.config.outro_duration_ms
            } else {
                0
            };

        // Determine target clip count
        let target_count = if self.config.target_clip_count > 0 {
            self.config.target_clip_count
        } else {
            // Auto-determine based on average clip duration
            let avg_clip_duration =
                (self.config.min_clip_duration_ms + self.config.max_clip_duration_ms) / 2;
            (available_duration / avg_clip_duration).max(1) as usize
        };

        // Select top clips
        for &scene in candidates.iter().take(target_count) {
            let clip_duration = scene
                .duration()
                .min(self.config.max_clip_duration_ms)
                .max(self.config.min_clip_duration_ms);

            if total_duration + clip_duration <= available_duration {
                selected.push(scene);
                total_duration += clip_duration;
            }

            if total_duration >= available_duration {
                break;
            }
        }

        if selected.is_empty() {
            return Err(AutoError::insufficient_data(
                "Could not select clips within duration constraints",
            ));
        }

        Ok(selected)
    }

    /// Apply dramatic arc by reordering clips for maximum impact.
    fn apply_dramatic_arc(&self, clips: &mut [AssembledClip]) {
        if clips.len() < 3 {
            return; // Need at least 3 clips for an arc
        }

        // Sort clips by importance
        let mut sorted_by_importance: Vec<_> = clips.iter().enumerate().collect();
        sorted_by_importance.sort_by(|a, b| {
            b.1.importance
                .partial_cmp(&a.1.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Create arc: moderate -> build -> climax -> moderate
        let climax_pos = clips.len() * 3 / 4; // Climax at 75% through

        // Place highest importance clip at climax
        if let Some((idx, _)) = sorted_by_importance.first() {
            if *idx != climax_pos && climax_pos < clips.len() {
                clips.swap(*idx, climax_pos);
            }
        }

        // Distribute remaining clips for smooth build-up
        // This is a simplified version - real implementation would be more sophisticated
    }

    /// Generate a social media clip (15s, 30s, or 60s).
    ///
    /// # Errors
    ///
    /// Returns an error if generation fails.
    pub fn generate_social_clip(
        &self,
        scenes: &[ScoredScene],
        duration_ms: i64,
    ) -> AutoResult<Vec<AssembledClip>> {
        let mut config = self.config.clone();
        config.assembly_type = AssemblyType::SocialClip;
        config.target_duration_ms = duration_ms;
        config.target_aspect_ratio = AspectRatio::Vertical9x16;
        config.use_dramatic_arc = true;
        config.chronological_order = false;

        let assembler = Self::new(config);
        assembler.assemble_from_scenes(scenes)
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
        let mut config = self.config.clone();
        config.assembly_type = AssemblyType::Trailer;
        config.target_duration_ms = 90_000; // 90 seconds
        config.use_dramatic_arc = true;
        config.add_bookends = true;
        config.intro_duration_ms = 3000; // 3 second fade in
        config.outro_duration_ms = 2000; // 2 second fade out

        let mut assembler = Self::new(config);

        // Use cut points to refine selection
        if !cuts.is_empty() {
            assembler = self.refine_with_cuts(assembler, cuts);
        }

        assembler.assemble_from_scenes(scenes)
    }

    /// Refine assembler with cut points.
    fn refine_with_cuts(&self, mut assembler: AutoAssembler, _cuts: &[CutPoint]) -> AutoAssembler {
        // In a real implementation, this would use cut points to inform clip selection
        assembler.config.min_clip_duration_ms = 2000; // Longer clips for trailers
        assembler
    }

    /// Get the current configuration.
    #[must_use]
    pub const fn config(&self) -> &AssemblyConfig {
        &self.config
    }
}

impl Default for AutoAssembler {
    fn default() -> Self {
        Self::default_assembler()
    }
}

/// Generate multiple social media clips of different durations.
#[allow(dead_code)]
pub fn generate_social_suite(
    scenes: &[ScoredScene],
) -> AutoResult<HashMap<String, Vec<AssembledClip>>> {
    let mut suite = HashMap::new();

    let config = AssemblyConfig::for_type(AssemblyType::SocialClip);
    let assembler = AutoAssembler::new(config);

    // 15 second clip
    let clip_15s = assembler.generate_social_clip(scenes, 15_000)?;
    suite.insert("15s".to_string(), clip_15s);

    // 30 second clip
    let clip_30s = assembler.generate_social_clip(scenes, 30_000)?;
    suite.insert("30s".to_string(), clip_30s);

    // 60 second clip
    let clip_60s = assembler.generate_social_clip(scenes, 60_000)?;
    suite.insert("60s".to_string(), clip_60s);

    Ok(suite)
}

/// Extract best moments with specific duration.
#[allow(dead_code)]
pub fn extract_best_moments(
    scenes: &[ScoredScene],
    count: usize,
    min_score: f64,
) -> Vec<ScoredScene> {
    let mut filtered: Vec<_> = scenes
        .iter()
        .filter(|s| s.adjusted_score() >= min_score)
        .collect();

    filtered.sort_by(|a, b| {
        b.adjusted_score()
            .partial_cmp(&a.adjusted_score())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    filtered.into_iter().take(count).cloned().collect()
}

/// Calculate optimal clip pacing for an assembly.
#[allow(dead_code)]
pub fn calculate_optimal_pacing(
    total_duration_ms: i64,
    num_clips: usize,
    assembly_type: AssemblyType,
) -> (i64, i64) {
    let (min_range, max_range) = assembly_type.typical_duration_range_ms();

    if num_clips == 0 {
        return (min_range / 10, max_range / 10);
    }

    let avg_duration = total_duration_ms / num_clips as i64;
    let min_duration = avg_duration / 2;
    let max_duration = avg_duration * 2;

    (
        min_duration.max(1000),   // At least 1 second
        max_duration.min(15_000), // At most 15 seconds
    )
}
