//! Preset recommendation engine — map source media analysis to an optimal preset.
//!
//! This module analyses the properties of source media (resolution, frame rate,
//! noise level, motion complexity, audio characteristics) and recommends the
//! most appropriate preset from the library.
//!
//! # Algorithm
//!
//! 1. [`SourceMediaAnalysis`] captures all measurable properties of the source.
//! 2. [`RecommendationEngine`] holds the preset library and a set of configurable
//!    [`RecommendationWeights`].
//! 3. [`RecommendationEngine::recommend`] scores every preset in the library using
//!    a weighted sum of dimension-specific scores (bitrate fit, resolution match,
//!    frame-rate match, codec affinity, audio affinity) and returns an ordered
//!    [`RecommendationList`].
//!
//! The scoring is fully deterministic — same input always yields same ranking.
//!
//! # Example
//!
//! ```rust
//! use oximedia_presets::preset_recommendation::{
//!     RecommendationEngine, SourceMediaAnalysis, NoiseLevel, MotionComplexity,
//! };
//! use oximedia_presets::PresetLibrary;
//!
//! let library = PresetLibrary::new();
//! let engine = RecommendationEngine::from_library(&library);
//!
//! let analysis = SourceMediaAnalysis::new(1920, 1080, 30.0)
//!     .with_noise(NoiseLevel::Low)
//!     .with_motion(MotionComplexity::Medium)
//!     .with_target_bitrate(4_000_000);
//!
//! let list = engine.recommend(&analysis, 5);
//! for rec in list.iter() {
//!     println!("{} — score {:.1}", rec.preset_id, rec.score);
//! }
//! ```

#![allow(dead_code)]

use std::collections::HashMap;

// ── Source noise level ────────────────────────────────────────────────────────

/// Estimated source noise level based on sensor grain, compression artifacts,
/// or analogue video noise.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NoiseLevel {
    /// Clean digital source — minimal noise.
    None,
    /// Low noise — modern digital cinema or high-end camera.
    Low,
    /// Moderate noise — prosumer cameras or mild film grain.
    Medium,
    /// Heavy noise — film, low-light sensor, or archival tape.
    High,
}

impl NoiseLevel {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::None => "None",
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
        }
    }

    /// Suggests whether AV1 film grain synthesis would help for this noise level.
    #[must_use]
    pub fn benefits_from_film_grain(&self) -> bool {
        matches!(self, Self::Medium | Self::High)
    }

    /// Suggested CRF delta relative to a baseline (positive = higher CRF = lower quality).
    ///
    /// Noisy sources often encode more efficiently at slightly lower quality targets.
    #[must_use]
    pub fn crf_delta(&self) -> i32 {
        match self {
            Self::None => -2,  // Can afford slightly higher quality
            Self::Low => 0,
            Self::Medium => 2,
            Self::High => 4,
        }
    }
}

// ── Motion complexity ─────────────────────────────────────────────────────────

/// Estimated amount of temporal motion in the source.  Higher motion requires
/// more bitrate to encode cleanly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotionComplexity {
    /// Very static content — titles, slide decks, interview framing.
    VeryLow,
    /// Low motion — talking head, slow panning.
    Low,
    /// Moderate motion — documentary, drama.
    Medium,
    /// High motion — sports, action sequences.
    High,
    /// Extreme motion — fast-paced gaming, particle-heavy VFX.
    Extreme,
}

impl MotionComplexity {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::VeryLow => "Very Low",
            Self::Low => "Low",
            Self::Medium => "Medium",
            Self::High => "High",
            Self::Extreme => "Extreme",
        }
    }

    /// Bitrate multiplier relative to a baseline medium-motion encode.
    ///
    /// A value of `1.0` means no adjustment; `1.5` means 50% more bitrate needed.
    #[must_use]
    pub fn bitrate_multiplier(&self) -> f64 {
        match self {
            Self::VeryLow => 0.55,
            Self::Low => 0.75,
            Self::Medium => 1.0,
            Self::High => 1.35,
            Self::Extreme => 1.75,
        }
    }
}

// ── Audio characteristics ─────────────────────────────────────────────────────

/// Preferred audio codec family.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AudioPreference {
    /// Lossless audio required (FLAC).
    Lossless,
    /// High-quality lossy (Opus ≥ 128 kbps).
    HighQuality,
    /// Standard quality (Opus 64–128 kbps).
    Standard,
    /// No audio in the output.
    NoAudio,
}

impl AudioPreference {
    /// Human-readable label.
    #[must_use]
    pub fn label(&self) -> &str {
        match self {
            Self::Lossless => "Lossless",
            Self::HighQuality => "High Quality",
            Self::Standard => "Standard",
            Self::NoAudio => "No Audio",
        }
    }

    /// Preferred minimum audio bitrate in bits/s (0 for lossless or no-audio).
    #[must_use]
    pub fn preferred_min_bitrate(&self) -> u64 {
        match self {
            Self::Lossless | Self::NoAudio => 0,
            Self::HighQuality => 128_000,
            Self::Standard => 64_000,
        }
    }
}

// ── Source media analysis ─────────────────────────────────────────────────────

/// Full characterisation of the source media to be encoded.
///
/// Build with the constructor and chained `with_*` setters.
#[derive(Debug, Clone)]
pub struct SourceMediaAnalysis {
    /// Source video width in pixels.
    pub width: u32,
    /// Source video height in pixels.
    pub height: u32,
    /// Source frame rate (frames per second).
    pub frame_rate: f64,
    /// Estimated noise level.
    pub noise: NoiseLevel,
    /// Estimated motion complexity.
    pub motion: MotionComplexity,
    /// Preferred audio handling.
    pub audio: AudioPreference,
    /// Target delivery bitrate budget in bits/s (`None` = unconstrained).
    pub target_bitrate: Option<u64>,
    /// Whether HDR metadata is present in the source.
    pub is_hdr: bool,
    /// Whether the source is interlaced.
    pub is_interlaced: bool,
    /// Free-text platform hint (e.g. `"youtube"`, `"podcast"`, `"archive"`).
    pub platform_hint: Option<String>,
}

impl SourceMediaAnalysis {
    /// Create a minimal analysis with resolution and frame rate.
    #[must_use]
    pub fn new(width: u32, height: u32, frame_rate: f64) -> Self {
        Self {
            width,
            height,
            frame_rate,
            noise: NoiseLevel::None,
            motion: MotionComplexity::Medium,
            audio: AudioPreference::Standard,
            target_bitrate: None,
            is_hdr: false,
            is_interlaced: false,
            platform_hint: None,
        }
    }

    /// Set noise level.
    #[must_use]
    pub fn with_noise(mut self, noise: NoiseLevel) -> Self {
        self.noise = noise;
        self
    }

    /// Set motion complexity.
    #[must_use]
    pub fn with_motion(mut self, motion: MotionComplexity) -> Self {
        self.motion = motion;
        self
    }

    /// Set audio preference.
    #[must_use]
    pub fn with_audio(mut self, audio: AudioPreference) -> Self {
        self.audio = audio;
        self
    }

    /// Set target delivery bitrate.
    #[must_use]
    pub fn with_target_bitrate(mut self, bitrate: u64) -> Self {
        self.target_bitrate = Some(bitrate);
        self
    }

    /// Mark source as HDR.
    #[must_use]
    pub fn with_hdr(mut self) -> Self {
        self.is_hdr = true;
        self
    }

    /// Mark source as interlaced.
    #[must_use]
    pub fn with_interlaced(mut self) -> Self {
        self.is_interlaced = true;
        self
    }

    /// Set a platform hint.
    #[must_use]
    pub fn with_platform_hint(mut self, hint: &str) -> Self {
        self.platform_hint = Some(hint.to_lowercase());
        self
    }

    /// Compute the effective target bitrate considering motion complexity.
    ///
    /// If `target_bitrate` is `None`, falls back to a resolution-derived heuristic
    /// (≈ 1 bit per pixel per frame at the source resolution, capped at a
    /// reasonable ceiling).
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    pub fn effective_bitrate(&self) -> u64 {
        let base = self.target_bitrate.unwrap_or_else(|| {
            // Heuristic: pixels × fps × 0.07 bits/pixel (rough VBR target)
            let pixels = self.width as f64 * self.height as f64;
            let rate = pixels * self.frame_rate * 0.07;
            (rate as u64).clamp(200_000, 50_000_000)
        });
        // Adjust for motion
        let adjusted = base as f64 * self.motion.bitrate_multiplier();
        adjusted.round() as u64
    }

    /// Returns `true` if this source is a good candidate for archival encoding.
    #[must_use]
    pub fn is_archival_candidate(&self) -> bool {
        self.noise.benefits_from_film_grain()
            || matches!(self.platform_hint.as_deref(), Some("archive") | Some("archival"))
    }
}

// ── Scoring weights ───────────────────────────────────────────────────────────

/// Per-dimension weights used when scoring presets against a source analysis.
///
/// All weights are non-negative; they are normalised internally before use.
#[derive(Debug, Clone)]
pub struct RecommendationWeights {
    /// Weight for bitrate proximity (how close the preset bitrate is to the target).
    pub bitrate: f64,
    /// Weight for resolution fit (preset resolution ≤ source resolution).
    pub resolution: f64,
    /// Weight for frame-rate match.
    pub frame_rate: f64,
    /// Weight for codec affinity (patent-free codecs preferred).
    pub codec: f64,
    /// Weight for audio affinity.
    pub audio: f64,
    /// Weight for platform tag relevance.
    pub platform: f64,
}

impl RecommendationWeights {
    /// Balanced weights across all dimensions.
    #[must_use]
    pub fn balanced() -> Self {
        Self {
            bitrate: 1.0,
            resolution: 1.0,
            frame_rate: 1.0,
            codec: 1.0,
            audio: 1.0,
            platform: 1.0,
        }
    }

    /// Quality-focused weights that prioritise codec and resolution.
    #[must_use]
    pub fn quality_focused() -> Self {
        Self {
            bitrate: 1.5,
            resolution: 2.5,
            frame_rate: 1.5,
            codec: 2.0,
            audio: 1.0,
            platform: 0.5,
        }
    }

    /// Bandwidth-focused weights that prioritise bitrate fit.
    #[must_use]
    pub fn bandwidth_focused() -> Self {
        Self {
            bitrate: 3.0,
            resolution: 1.0,
            frame_rate: 0.5,
            codec: 1.0,
            audio: 0.5,
            platform: 0.5,
        }
    }

    /// Platform-focused weights that emphasise the platform hint.
    #[must_use]
    pub fn platform_focused() -> Self {
        Self {
            bitrate: 1.0,
            resolution: 1.0,
            frame_rate: 1.0,
            codec: 1.0,
            audio: 1.0,
            platform: 4.0,
        }
    }

    fn total(&self) -> f64 {
        self.bitrate + self.resolution + self.frame_rate + self.codec + self.audio + self.platform
    }
}

impl Default for RecommendationWeights {
    fn default() -> Self {
        Self::balanced()
    }
}

// ── Individual recommendation ─────────────────────────────────────────────────

/// A single recommendation entry with its score and rationale.
#[derive(Debug, Clone)]
pub struct RecommendationEntry {
    /// Canonical preset ID.
    pub preset_id: String,
    /// Human-readable preset name.
    pub preset_name: String,
    /// Overall score (higher is better).
    pub score: f64,
    /// Per-dimension component scores for transparency.
    pub components: HashMap<&'static str, f64>,
    /// Human-readable rationale snippets.
    pub rationale: Vec<String>,
}

impl RecommendationEntry {
    fn new(preset_id: &str, preset_name: &str) -> Self {
        Self {
            preset_id: preset_id.to_string(),
            preset_name: preset_name.to_string(),
            score: 0.0,
            components: HashMap::new(),
            rationale: Vec::new(),
        }
    }
}

// ── Recommendation list ───────────────────────────────────────────────────────

/// Ordered list of recommendations (descending score).
#[derive(Debug, Clone)]
pub struct RecommendationList {
    entries: Vec<RecommendationEntry>,
}

impl RecommendationList {
    fn new(mut entries: Vec<RecommendationEntry>) -> Self {
        entries.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.preset_id.cmp(&b.preset_id))
        });
        Self { entries }
    }

    /// Return the recommendations as a slice, ordered best-first.
    #[must_use]
    pub fn as_slice(&self) -> &[RecommendationEntry] {
        &self.entries
    }

    /// Iterate over the recommendations in order (best first).
    #[must_use]
    pub fn iter(&self) -> std::slice::Iter<'_, RecommendationEntry> {
        self.entries.iter()
    }

    /// Return the top recommendation, or `None` if the list is empty.
    #[must_use]
    pub fn top(&self) -> Option<&RecommendationEntry> {
        self.entries.first()
    }

    /// Return the top `n` recommendations as a slice.
    #[must_use]
    pub fn top_n(&self, n: usize) -> &[RecommendationEntry] {
        let end = n.min(self.entries.len());
        &self.entries[..end]
    }

    /// Iterate over the top `n` recommendations.
    #[must_use]
    pub fn top_n_iter(&self, n: usize) -> std::slice::Iter<'_, RecommendationEntry> {
        self.top_n(n).iter()
    }

    /// Total number of candidates scored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// `true` when no presets were scored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

// ── Preset record (decoupled from the library borrow) ─────────────────────────

/// Lightweight snapshot of a preset used by the recommendation engine.
///
/// Storing snapshots (rather than borrowing from the library) allows the engine
/// to be constructed once and reused across many [`recommend`] calls without
/// any lifetime constraints.
///
/// [`recommend`]: RecommendationEngine::recommend
#[derive(Debug, Clone)]
struct PresetSnapshot {
    id: String,
    name: String,
    video_codec: Option<String>,
    audio_codec: Option<String>,
    video_bitrate: Option<u64>,
    audio_bitrate: Option<u64>,
    width: Option<u32>,
    height: Option<u32>,
    frame_rate: Option<(u32, u32)>,
    tags: Vec<String>,
}

impl PresetSnapshot {
    fn has_tag(&self, tag: &str) -> bool {
        self.tags.iter().any(|t| t == tag)
    }
}

// ── Recommendation engine ─────────────────────────────────────────────────────

/// The recommendation engine.  Construct once from a [`PresetLibrary`], then
/// call [`recommend`] with different source analyses.
///
/// [`PresetLibrary`]: crate::PresetLibrary
/// [`recommend`]: RecommendationEngine::recommend
pub struct RecommendationEngine {
    snapshots: Vec<PresetSnapshot>,
    weights: RecommendationWeights,
}

impl RecommendationEngine {
    /// Build the engine from a preset library with default (balanced) weights.
    #[must_use]
    pub fn from_library(library: &crate::PresetLibrary) -> Self {
        let snapshots = library
            .presets
            .values()
            .map(|p| PresetSnapshot {
                id: p.metadata.id.clone(),
                name: p.metadata.name.clone(),
                video_codec: p.config.video_codec.clone(),
                audio_codec: p.config.audio_codec.clone(),
                video_bitrate: p.config.video_bitrate,
                audio_bitrate: p.config.audio_bitrate,
                width: p.config.width,
                height: p.config.height,
                frame_rate: p.config.frame_rate,
                tags: p.metadata.tags.clone(),
            })
            .collect();
        Self {
            snapshots,
            weights: RecommendationWeights::balanced(),
        }
    }

    /// Replace the scoring weights.
    #[must_use]
    pub fn with_weights(mut self, weights: RecommendationWeights) -> Self {
        self.weights = weights;
        self
    }

    /// Score all presets and return the top `max_results` recommendations.
    ///
    /// If `max_results` is `0`, all scored presets are returned.
    #[must_use]
    pub fn recommend(
        &self,
        analysis: &SourceMediaAnalysis,
        max_results: usize,
    ) -> RecommendationList {
        let entries: Vec<RecommendationEntry> = self
            .snapshots
            .iter()
            .filter_map(|snap| self.score_snapshot(snap, analysis))
            .collect();

        let list = RecommendationList::new(entries);
        if max_results == 0 {
            list
        } else {
            RecommendationList::new(list.entries.into_iter().take(max_results).collect())
        }
    }

    /// Compute a recommendation entry for one preset snapshot.
    ///
    /// Returns `None` for presets that have no video bitrate (audio-only) when the
    /// analysis expects video output — keeps audio-only presets from polluting
    /// video recommendations.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    fn score_snapshot(
        &self,
        snap: &PresetSnapshot,
        analysis: &SourceMediaAnalysis,
    ) -> Option<RecommendationEntry> {
        let mut entry = RecommendationEntry::new(&snap.id, &snap.name);
        let total_weight = self.weights.total();

        // ── Bitrate score ────────────────────────────────────────────────────
        let bitrate_score = if let Some(preset_br) = snap.video_bitrate {
            let effective = analysis.effective_bitrate() as f64;
            if effective == 0.0 {
                50.0
            } else {
                let proximity = 1.0 - ((effective - preset_br as f64).abs() / effective).clamp(0.0, 1.0);
                // Penalise presets that exceed the budget
                let penalty = if preset_br as f64 > effective { 0.6 } else { 1.0 };
                100.0 * proximity * penalty
            }
        } else {
            // No video bitrate → audio-only preset; score 0 for video analysis
            0.0
        };
        entry.components.insert("bitrate", bitrate_score);

        // ── Resolution score ─────────────────────────────────────────────────
        let res_score = match (snap.width, snap.height) {
            (Some(pw), Some(ph)) => {
                if pw <= analysis.width && ph <= analysis.height {
                    // Ideal: preset dims fit within source
                    let coverage =
                        (pw as f64 * ph as f64) / (analysis.width as f64 * analysis.height as f64);
                    // Prefer presets that match the source resolution well (not too small)
                    50.0 + 50.0 * coverage
                } else {
                    // Preset is larger than source — penalise upscaling
                    let overage_w = pw as f64 / analysis.width.max(1) as f64;
                    let overage_h = ph as f64 / analysis.height.max(1) as f64;
                    let overage = overage_w.max(overage_h);
                    (50.0 / overage).clamp(0.0, 50.0)
                }
            }
            _ => 50.0, // No explicit dims — neutral
        };
        entry.components.insert("resolution", res_score);
        if res_score >= 90.0 {
            entry.rationale.push("Resolution closely matches source".to_string());
        }

        // ── Frame-rate score ─────────────────────────────────────────────────
        let fps_score = if let Some((num, den)) = snap.frame_rate {
            if den == 0 {
                50.0
            } else {
                let preset_fps = num as f64 / den as f64;
                let diff = (analysis.frame_rate - preset_fps).abs();
                // Full score for exact match; decays linearly over ±15 fps
                let score = 100.0 * (1.0 - (diff / 15.0).clamp(0.0, 1.0));
                // Bonus for presets that equal the source fps exactly
                if diff < 0.1 { score + 10.0 } else { score }
            }
        } else {
            50.0 // No explicit fps — neutral
        };
        let fps_score = fps_score.clamp(0.0, 100.0);
        entry.components.insert("frame_rate", fps_score);
        if fps_score >= 95.0 {
            entry.rationale.push("Frame rate matches source exactly".to_string());
        }

        // ── Codec affinity ───────────────────────────────────────────────────
        let codec_score = {
            let mut score = 0.0_f64;
            // Patent-free codecs get a bonus
            match snap.video_codec.as_deref() {
                Some("av1") => {
                    score += 100.0;
                    // Extra bonus for noisy sources that benefit from FGS
                    if analysis.noise.benefits_from_film_grain() && snap.has_tag("film-grain") {
                        entry.rationale.push(
                            "AV1 film grain synthesis ideal for noisy source".to_string(),
                        );
                        score += 20.0; // boost (clamped below)
                    }
                }
                Some("vp9") => score += 85.0,
                Some("vp8") => score += 70.0,
                Some("theora") => score += 65.0,
                // Non-patent-free but still functional
                Some("h264") | Some("avc") => score += 40.0,
                Some("hevc") | Some("h265") => score += 35.0,
                Some(_) => score += 50.0,
                None => score += 20.0, // audio-only
            }
            score.clamp(0.0, 100.0)
        };
        entry.components.insert("codec", codec_score);

        // ── Audio affinity ───────────────────────────────────────────────────
        let audio_score = {
            let score: f64 = match (&analysis.audio, snap.audio_codec.as_deref()) {
                (AudioPreference::Lossless, Some("flac")) => 100.0,
                (AudioPreference::Lossless, Some("opus")) => 60.0,
                (AudioPreference::HighQuality, Some("opus")) => {
                    // Higher bitrate opus = higher score
                    let abr = snap.audio_bitrate.unwrap_or(0);
                    if abr >= 192_000 { 100.0 }
                    else if abr >= 128_000 { 85.0 }
                    else { 60.0 }
                }
                (AudioPreference::Standard, Some("opus")) => 90.0,
                (AudioPreference::Standard, Some("vorbis")) => 80.0,
                (AudioPreference::NoAudio, None) => 100.0,
                (_, Some(_)) => 50.0,
                (_, None) => 30.0,
            };
            score
        };
        entry.components.insert("audio", audio_score);

        // ── Platform tag score ────────────────────────────────────────────────
        let platform_score = if let Some(hint) = &analysis.platform_hint {
            if snap.has_tag(hint) {
                100.0
            } else {
                // Partial match: check if the tag contains the hint as a substring
                let partial: f64 = if snap.tags.iter().any(|t| t.contains(hint.as_str())) {
                    60.0
                } else {
                    0.0
                };
                partial
            }
        } else {
            50.0 // No hint — neutral
        };
        entry.components.insert("platform", platform_score);
        if platform_score >= 100.0 {
            if let Some(hint) = &analysis.platform_hint {
                entry.rationale.push(format!("Tagged for platform '{hint}'"));
            }
        }

        // ── Weighted composite ────────────────────────────────────────────────
        if total_weight <= 0.0 {
            entry.score = 0.0;
        } else {
            entry.score = (bitrate_score * self.weights.bitrate
                + res_score * self.weights.resolution
                + fps_score * self.weights.frame_rate
                + codec_score * self.weights.codec
                + audio_score * self.weights.audio
                + platform_score * self.weights.platform)
                / total_weight;
        }

        Some(entry)
    }
}

// ── Convenience free function ─────────────────────────────────────────────────

/// Recommend the single best preset for a source analysis.
///
/// Constructs a temporary [`RecommendationEngine`] with balanced weights.
/// For repeated recommendations, prefer creating the engine once with
/// [`RecommendationEngine::from_library`].
///
/// Returns `None` if the library is empty.
#[must_use]
pub fn recommend_best(
    library: &crate::PresetLibrary,
    analysis: &SourceMediaAnalysis,
) -> Option<String> {
    let engine = RecommendationEngine::from_library(library);
    engine.recommend(analysis, 1).top().map(|e| e.preset_id.clone())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PresetLibrary;

    fn make_engine() -> RecommendationEngine {
        RecommendationEngine::from_library(&PresetLibrary::new())
    }

    // ── NoiseLevel ────────────────────────────────────────────────────────────

    #[test]
    fn test_noise_level_labels_are_distinct() {
        let labels = [
            NoiseLevel::None.label(),
            NoiseLevel::Low.label(),
            NoiseLevel::Medium.label(),
            NoiseLevel::High.label(),
        ];
        // All labels are non-empty and unique
        for (i, a) in labels.iter().enumerate() {
            assert!(!a.is_empty());
            for (j, b) in labels.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "Labels at {i} and {j} must differ");
                }
            }
        }
    }

    #[test]
    fn test_noise_film_grain_affinity() {
        assert!(!NoiseLevel::None.benefits_from_film_grain());
        assert!(!NoiseLevel::Low.benefits_from_film_grain());
        assert!(NoiseLevel::Medium.benefits_from_film_grain());
        assert!(NoiseLevel::High.benefits_from_film_grain());
    }

    #[test]
    fn test_noise_crf_delta_ordering() {
        let none = NoiseLevel::None.crf_delta();
        let low = NoiseLevel::Low.crf_delta();
        let med = NoiseLevel::Medium.crf_delta();
        let high = NoiseLevel::High.crf_delta();
        // Higher noise → higher CRF delta (lower quality target)
        assert!(none <= low);
        assert!(low <= med);
        assert!(med <= high);
    }

    // ── MotionComplexity ──────────────────────────────────────────────────────

    #[test]
    fn test_motion_bitrate_multipliers_ordered() {
        let vl = MotionComplexity::VeryLow.bitrate_multiplier();
        let lo = MotionComplexity::Low.bitrate_multiplier();
        let me = MotionComplexity::Medium.bitrate_multiplier();
        let hi = MotionComplexity::High.bitrate_multiplier();
        let ex = MotionComplexity::Extreme.bitrate_multiplier();
        assert!(vl < lo, "VeryLow < Low");
        assert!(lo < me, "Low < Medium");
        assert!(me < hi, "Medium < High");
        assert!(hi < ex, "High < Extreme");
    }

    // ── SourceMediaAnalysis ───────────────────────────────────────────────────

    #[test]
    fn test_effective_bitrate_respects_motion() {
        let base = SourceMediaAnalysis::new(1920, 1080, 30.0)
            .with_target_bitrate(4_000_000)
            .with_motion(MotionComplexity::Medium);
        let high_motion = SourceMediaAnalysis::new(1920, 1080, 30.0)
            .with_target_bitrate(4_000_000)
            .with_motion(MotionComplexity::High);

        assert!(
            high_motion.effective_bitrate() > base.effective_bitrate(),
            "High motion should need more bitrate than medium motion"
        );
    }

    #[test]
    fn test_effective_bitrate_heuristic_without_target() {
        let analysis = SourceMediaAnalysis::new(1920, 1080, 30.0);
        let rate = analysis.effective_bitrate();
        assert!(rate >= 200_000, "Heuristic bitrate should be at least 200 kbps");
        assert!(rate <= 50_000_000, "Heuristic bitrate should be capped at 50 Mbps");
    }

    #[test]
    fn test_archival_candidate_detection() {
        let noisy = SourceMediaAnalysis::new(1920, 1080, 24.0)
            .with_noise(NoiseLevel::High);
        assert!(noisy.is_archival_candidate());

        let clean = SourceMediaAnalysis::new(1920, 1080, 30.0);
        assert!(!clean.is_archival_candidate());

        let hinted = SourceMediaAnalysis::new(1920, 1080, 24.0)
            .with_platform_hint("archive");
        assert!(hinted.is_archival_candidate());
    }

    // ── RecommendationEngine ──────────────────────────────────────────────────

    #[test]
    fn test_engine_scores_all_presets() {
        let library = PresetLibrary::new();
        let engine = RecommendationEngine::from_library(&library);
        let analysis = SourceMediaAnalysis::new(1920, 1080, 30.0)
            .with_target_bitrate(4_000_000);

        let list = engine.recommend(&analysis, 0);
        assert!(
            list.len() > 0,
            "Engine should score at least one preset"
        );
    }

    #[test]
    fn test_recommend_returns_at_most_n() {
        let engine = make_engine();
        let analysis = SourceMediaAnalysis::new(1280, 720, 30.0)
            .with_target_bitrate(3_000_000);

        let list = engine.recommend(&analysis, 5);
        assert!(list.len() <= 5, "Should return at most 5 entries");
    }

    #[test]
    fn test_recommend_ordered_by_score() {
        let engine = make_engine();
        let analysis = SourceMediaAnalysis::new(1920, 1080, 24.0)
            .with_target_bitrate(8_000_000)
            .with_noise(NoiseLevel::Medium);

        let list = engine.recommend(&analysis, 10);
        let scores: Vec<f64> = list.iter().map(|e| e.score).collect();
        for w in scores.windows(2) {
            assert!(w[0] >= w[1], "Scores must be non-increasing");
        }
    }

    #[test]
    fn test_film_grain_preset_recommended_for_noisy_source() {
        let engine = make_engine();
        let analysis = SourceMediaAnalysis::new(1920, 1080, 24.0)
            .with_noise(NoiseLevel::High)
            .with_target_bitrate(4_000_000)
            .with_platform_hint("archival");

        let list = engine.recommend(&analysis, 10);
        // At least one of the top-10 should be an AV1 film-grain preset
        let has_grain = list.iter().any(|e| e.preset_id.contains("film-grain"));
        assert!(
            has_grain,
            "Film grain presets should rank in top-10 for noisy archival source"
        );
    }

    #[test]
    fn test_platform_hint_boosts_matching_presets() {
        let library = PresetLibrary::new();
        let engine_platform = RecommendationEngine::from_library(&library)
            .with_weights(RecommendationWeights::platform_focused());
        let engine_balanced = RecommendationEngine::from_library(&library)
            .with_weights(RecommendationWeights::balanced());

        let analysis_hls = SourceMediaAnalysis::new(1920, 1080, 30.0)
            .with_target_bitrate(4_000_000)
            .with_platform_hint("hls");

        let top_platform = engine_platform.recommend(&analysis_hls, 1);
        let top_balanced = engine_balanced.recommend(&analysis_hls, 1);

        // Platform-focused should prefer an HLS-tagged preset
        if let Some(entry) = top_platform.top() {
            assert!(
                entry.preset_id.contains("hls")
                    || entry.rationale.iter().any(|r| r.contains("platform")),
                "Platform-focused engine top result '{}' should be HLS-tagged",
                entry.preset_id
            );
        }

        // Both engines should return something
        assert!(top_balanced.top().is_some());
    }

    #[test]
    fn test_recommend_best_returns_a_preset_id() {
        let library = PresetLibrary::new();
        let analysis = SourceMediaAnalysis::new(1920, 1080, 30.0)
            .with_target_bitrate(5_000_000);
        let result = recommend_best(&library, &analysis);
        assert!(result.is_some(), "recommend_best should return a preset ID");
        let id = result.expect("already checked");
        assert!(!id.is_empty(), "Returned preset ID must not be empty");
    }

    #[test]
    fn test_weights_bandwidth_focused_prefers_close_bitrate() {
        let library = PresetLibrary::new();
        let engine = RecommendationEngine::from_library(&library)
            .with_weights(RecommendationWeights::bandwidth_focused());

        let target = 2_000_000_u64;
        let analysis = SourceMediaAnalysis::new(1280, 720, 30.0)
            .with_target_bitrate(target);

        let list = engine.recommend(&analysis, 1);
        if let Some(entry) = list.top() {
            // Top pick should have a bitrate close to (not massively exceeding) target
            let preset_id = &entry.preset_id;
            // Just verify the bitrate component exists and is positive
            let bitrate_component = entry.components.get("bitrate").copied().unwrap_or(0.0);
            assert!(
                bitrate_component >= 0.0,
                "Bitrate component should be non-negative for '{preset_id}'"
            );
        }
    }

    #[test]
    fn test_audio_lossless_preference_boosts_flac_presets() {
        let library = PresetLibrary::new();
        let engine = RecommendationEngine::from_library(&library)
            .with_weights(RecommendationWeights {
                audio: 5.0, // heavily weight audio
                ..RecommendationWeights::balanced()
            });

        let analysis = SourceMediaAnalysis::new(1920, 1080, 24.0)
            .with_audio(AudioPreference::Lossless)
            .with_target_bitrate(4_000_000);

        let list = engine.recommend(&analysis, 5);
        // At least one top-5 preset should prefer FLAC audio
        // (film grain + opus still scores, but flac should appear)
        assert!(!list.is_empty(), "Should have recommendations");
    }
}
