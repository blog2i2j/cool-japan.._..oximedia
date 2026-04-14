//! Enhanced optimal preset selection with resolution, frame rate, and use-case awareness.
//!
//! Provides [`SelectionCriteria`], [`UseCase`], [`ScoredPreset`], and the
//! `OptimalPresetSelector::select_scored` method that scores every preset in
//! the library against multi-dimensional criteria and returns a ranked list
//! with human-readable explanations.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

use crate::{Preset, PresetLibrary};
use std::sync::Arc;

// ─────────────────────────────────────────────────────────────────────────────
// UseCase
// ─────────────────────────────────────────────────────────────────────────────

/// Intended use case for the encoded output.
///
/// Influences which preset attributes receive bonus scores during selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum UseCase {
    /// Live or VOD streaming platforms (YouTube, Twitch, HLS/DASH).
    Streaming,
    /// Long-term archival; prioritises lossless or near-lossless formats.
    Archive,
    /// Short-form social-media delivery (Instagram, TikTok, Twitter).
    Social,
    /// Professional broadcast (ATSC, DVB-T, ISDB) with strict compliance.
    Broadcast,
    /// Mobile delivery optimised for small screens and limited bandwidth.
    Mobile,
}

impl UseCase {
    /// Return the tag strings associated with this use case for tag-bonus scoring.
    pub fn associated_tags(self) -> &'static [&'static str] {
        match self {
            Self::Streaming => &["streaming", "hls", "dash", "rtmp", "srt", "vod"],
            Self::Archive => &["archive", "lossless", "mezzanine", "ffv1", "flac"],
            Self::Social => &[
                "social",
                "instagram",
                "tiktok",
                "twitter",
                "facebook",
                "reels",
                "stories",
            ],
            Self::Broadcast => &["broadcast", "atsc", "dvb", "isdb", "hd-sdi"],
            Self::Mobile => &["mobile", "ios", "android", "cellular", "adaptive"],
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SelectionCriteria
// ─────────────────────────────────────────────────────────────────────────────

/// Multi-dimensional criteria used to score presets in `OptimalPresetSelector::select_scored`.
#[derive(Debug, Clone)]
pub struct SelectionCriteria {
    /// Target video bitrate in kbps (0 = not considered).
    pub target_bitrate_kbps: u64,
    /// Target output width in pixels (0 = not considered).
    pub width: u32,
    /// Target output height in pixels (0 = not considered).
    pub height: u32,
    /// Target frame rate in frames per second (0.0 = not considered).
    pub frame_rate: f32,
    /// Intended use case (applies tag-based bonus scoring).
    pub use_case: UseCase,
}

impl SelectionCriteria {
    /// Create criteria targeting 1080p30 streaming at 4 Mbps.
    pub fn streaming_1080p() -> Self {
        Self {
            target_bitrate_kbps: 4_000,
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            use_case: UseCase::Streaming,
        }
    }

    /// Create criteria targeting 4K (2160p) archival.
    pub fn archive_4k() -> Self {
        Self {
            target_bitrate_kbps: 0,
            width: 3840,
            height: 2160,
            frame_rate: 24.0,
            use_case: UseCase::Archive,
        }
    }

    /// Create criteria targeting mobile delivery (720p, 1 Mbps).
    pub fn mobile_720p() -> Self {
        Self {
            target_bitrate_kbps: 1_000,
            width: 1280,
            height: 720,
            frame_rate: 30.0,
            use_case: UseCase::Mobile,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ScoredPreset
// ─────────────────────────────────────────────────────────────────────────────

/// A preset together with its selection score and explanatory reasons.
#[derive(Debug, Clone)]
pub struct ScoredPreset {
    /// The matched preset (cheaply shared via Arc).
    pub preset: Arc<Preset>,
    /// Overall score in 0.0 – 100.0; higher is a better match.
    pub score: f32,
    /// Human-readable reasons explaining the score components.
    pub reasons: Vec<String>,
}

// ─────────────────────────────────────────────────────────────────────────────
// Scoring helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Score (0–40) based on how closely the preset's configured resolution matches
/// the target dimensions.
///
/// Uses `PresetConfig.width`/`height` when available (precise); falls back to
/// name/description token scanning for presets without explicit dimensions set.
/// Presets that match exactly score 40; those that slightly exceed the target
/// score ~30; wildly different resolutions score 0.
fn score_resolution(preset: &Preset, criteria: &SelectionCriteria) -> (f32, Option<String>) {
    if criteria.width == 0 || criteria.height == 0 {
        return (0.0, None);
    }

    let target_px = criteria.width as f32 * criteria.height as f32;

    // Prefer using the actual configured dimensions when present.
    let preset_px: Option<f32> = if let (Some(w), Some(h)) =
        (preset.config.width, preset.config.height)
    {
        Some(w as f32 * h as f32)
    } else {
        // Fallback: parse resolution tokens from name / description.
        let name_lower = preset.metadata.name.to_lowercase();
        let desc_lower = preset.metadata.description.to_lowercase();
        if name_lower.contains("4k") || name_lower.contains("2160") || desc_lower.contains("2160") {
            Some(3840.0 * 2160.0)
        } else if name_lower.contains("1440") || desc_lower.contains("1440") {
            Some(2560.0 * 1440.0)
        } else if name_lower.contains("1080") || desc_lower.contains("1080") {
            Some(1920.0 * 1080.0)
        } else if name_lower.contains("720") || desc_lower.contains("720") {
            Some(1280.0 * 720.0)
        } else if name_lower.contains("480") || desc_lower.contains("480") {
            Some(854.0 * 480.0)
        } else if name_lower.contains("360") || desc_lower.contains("360") {
            Some(640.0 * 360.0)
        } else if name_lower.contains("lossless") || desc_lower.contains("lossless") {
            // Lossless presets are resolution-agnostic — give neutral score.
            Some(target_px)
        } else {
            None
        }
    };

    if let Some(ppx) = preset_px {
        let ratio = if ppx >= target_px {
            ppx / target_px
        } else {
            target_px / ppx
        };
        // Perfect match → 40; 2× off → ~20; 4× off → ~10; beyond → 0.
        let raw = (40.0 / ratio).min(40.0).max(0.0);
        let label = if (ratio - 1.0).abs() < 0.01 {
            "exact resolution match".to_string()
        } else if ppx > target_px {
            format!(
                "resolution exceeds target ({}×{} vs {}×{})",
                preset.config.width.unwrap_or(0),
                preset.config.height.unwrap_or(0),
                criteria.width,
                criteria.height
            )
        } else {
            "resolution below target".to_string()
        };
        (raw, Some(label))
    } else {
        (0.0, None)
    }
}

/// Score (0–20) based on frame-rate match.
fn score_frame_rate(preset: &Preset, criteria: &SelectionCriteria) -> (f32, Option<String>) {
    if criteria.frame_rate <= 0.0 {
        return (0.0, None);
    }

    let target = criteria.frame_rate;

    // Check precise config.frame_rate first.
    if let Some((num, den)) = preset.config.frame_rate {
        let actual_fps = if den == 0 {
            0.0
        } else {
            num as f32 / den as f32
        };
        if actual_fps > 0.0 {
            let diff = (actual_fps - target).abs();
            let raw = if diff < 0.5 {
                20.0 // exact match (includes 29.97 vs 30)
            } else if diff < 5.0 {
                12.0 // close match
            } else {
                4.0 // significant mismatch
            };
            let reason = format!("{actual_fps:.2} fps vs target {target:.2} fps");
            return (raw, Some(reason));
        }
    }

    // Fallback: scan name / description tokens.
    let name_lower = preset.metadata.name.to_lowercase();
    let desc_lower = preset.metadata.description.to_lowercase();

    let supports_60 = name_lower.contains("60fps")
        || name_lower.contains("60 fps")
        || desc_lower.contains("60fps")
        || desc_lower.contains("60 fps");
    let supports_30 = name_lower.contains("30fps")
        || name_lower.contains("30 fps")
        || desc_lower.contains("30fps")
        || desc_lower.contains("30 fps");
    let supports_24 = name_lower.contains("24fps")
        || name_lower.contains("24 fps")
        || desc_lower.contains("24fps")
        || desc_lower.contains("24 fps");

    let (raw, reason) = if (target - 60.0).abs() < 1.0 && supports_60 {
        (20.0, "60 fps preset matches target")
    } else if (target - 30.0).abs() < 1.0 && supports_30 {
        (20.0, "30 fps preset matches target")
    } else if (target - 24.0).abs() < 1.0 && supports_24 {
        (20.0, "24 fps preset matches target")
    } else if (target - 60.0).abs() < 1.0 && !supports_60 {
        (5.0, "preset does not advertise 60 fps support")
    } else {
        (10.0, "frame rate neutral")
    };

    (raw, Some(reason.to_string()))
}

/// Score (0–25) based on bitrate proximity.
fn score_bitrate(preset: &Preset, criteria: &SelectionCriteria) -> (f32, Option<String>) {
    if criteria.target_bitrate_kbps == 0 {
        return (0.0, None);
    }

    if let Some(br_bps) = preset.config.video_bitrate {
        let target_bps = criteria.target_bitrate_kbps * 1000;
        let diff = if br_bps > target_bps {
            br_bps - target_bps
        } else {
            target_bps - br_bps
        };
        // Score: 25 for exact match, decays with difference.
        let ratio = diff as f32 / target_bps as f32;
        let raw = (25.0 * (1.0 - ratio.min(1.0))).max(0.0);
        let reason = format!(
            "bitrate {}kbps vs target {}kbps",
            br_bps / 1000,
            criteria.target_bitrate_kbps
        );
        (raw, Some(reason))
    } else {
        (0.0, None)
    }
}

/// Score (0–15) based on use-case tag matching.
fn score_use_case(preset: &Preset, criteria: &SelectionCriteria) -> (f32, Option<String>) {
    let tags = criteria.use_case.associated_tags();
    let matched: Vec<&str> = tags
        .iter()
        .filter(|&&t| preset.has_tag(t))
        .copied()
        .collect();

    if matched.is_empty() {
        (0.0, None)
    } else {
        let raw = ((matched.len() as f32 / tags.len() as f32) * 15.0).min(15.0);
        let reason = format!("matched use-case tags: {}", matched.join(", "));
        (raw, Some(reason))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// OptimalPreset enhancement
// ─────────────────────────────────────────────────────────────────────────────

/// Enhanced optimal-preset selector.
///
/// The existing [`crate::OptimalPreset`] already provides bitrate-only
/// selection.  This module adds multi-criteria scored selection.
pub struct OptimalPresetSelector;

impl OptimalPresetSelector {
    /// Score every preset in `library` against `criteria` and return them
    /// sorted by descending score.
    ///
    /// Only presets with a non-zero composite score (≥ 0.1) are included
    /// unless the library would otherwise return nothing, in which case all
    /// presets are returned with score 0.
    #[must_use]
    pub fn select(criteria: &SelectionCriteria, library: &PresetLibrary) -> Vec<ScoredPreset> {
        let mut results: Vec<ScoredPreset> = library
            .presets_iter()
            .map(|preset| {
                let mut total_score = 0.0_f32;
                let mut reasons: Vec<String> = Vec::new();

                // Resolution component (max 40 pts).
                let (res_score, res_reason) = score_resolution(preset, criteria);
                total_score += res_score;
                if let Some(r) = res_reason {
                    reasons.push(format!("resolution: {r} (+{res_score:.1})"));
                }

                // Frame rate component (max 20 pts).
                let (fps_score, fps_reason) = score_frame_rate(preset, criteria);
                total_score += fps_score;
                if let Some(r) = fps_reason {
                    reasons.push(format!("frame-rate: {r} (+{fps_score:.1})"));
                }

                // Bitrate component (max 25 pts).
                let (br_score, br_reason) = score_bitrate(preset, criteria);
                total_score += br_score;
                if let Some(r) = br_reason {
                    reasons.push(format!("bitrate: {r} (+{br_score:.1})"));
                }

                // Use-case tag component (max 15 pts).
                let (uc_score, uc_reason) = score_use_case(preset, criteria);
                total_score += uc_score;
                if let Some(r) = uc_reason {
                    reasons.push(format!("use-case: {r} (+{uc_score:.1})"));
                }

                ScoredPreset {
                    preset: Arc::new(preset.clone()),
                    score: total_score,
                    reasons,
                }
            })
            .collect();

        results.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Filter to presets with meaningful scores when possible.
        let any_nonzero = results.iter().any(|r| r.score > 0.1);
        if any_nonzero {
            results.retain(|r| r.score > 0.1);
        }

        results
    }

    /// Return only the top-N results.
    #[must_use]
    pub fn top_n(
        criteria: &SelectionCriteria,
        library: &PresetLibrary,
        n: usize,
    ) -> Vec<ScoredPreset> {
        let all = Self::select(criteria, library);
        all.into_iter().take(n).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn library() -> PresetLibrary {
        PresetLibrary::new()
    }

    // ── UseCase helpers ───────────────────────────────────────────────────────

    #[test]
    fn test_use_case_tags_non_empty() {
        for uc in [
            UseCase::Streaming,
            UseCase::Archive,
            UseCase::Social,
            UseCase::Broadcast,
            UseCase::Mobile,
        ] {
            assert!(
                !uc.associated_tags().is_empty(),
                "UseCase {uc:?} should have tags"
            );
        }
    }

    // ── SelectionCriteria constructors ────────────────────────────────────────

    #[test]
    fn test_criteria_streaming_1080p() {
        let c = SelectionCriteria::streaming_1080p();
        assert_eq!(c.height, 1080);
        assert_eq!(c.use_case, UseCase::Streaming);
    }

    #[test]
    fn test_criteria_archive_4k() {
        let c = SelectionCriteria::archive_4k();
        assert_eq!(c.height, 2160);
        assert_eq!(c.use_case, UseCase::Archive);
    }

    #[test]
    fn test_criteria_mobile_720p() {
        let c = SelectionCriteria::mobile_720p();
        assert_eq!(c.height, 720);
        assert_eq!(c.use_case, UseCase::Mobile);
    }

    // ── Core selection tests ──────────────────────────────────────────────────

    #[test]
    fn test_select_returns_results() {
        let lib = library();
        let criteria = SelectionCriteria::streaming_1080p();
        let results = OptimalPresetSelector::select(&criteria, &lib);
        assert!(!results.is_empty(), "Should return at least one result");
    }

    #[test]
    fn test_select_sorted_descending() {
        let lib = library();
        let criteria = SelectionCriteria::streaming_1080p();
        let results = OptimalPresetSelector::select(&criteria, &lib);
        for w in results.windows(2) {
            assert!(
                w[0].score >= w[1].score,
                "Results must be sorted by descending score"
            );
        }
    }

    #[test]
    fn test_scored_preset_has_reasons() {
        let lib = library();
        let criteria = SelectionCriteria::streaming_1080p();
        let results = OptimalPresetSelector::select(&criteria, &lib);
        let top = results.first().expect("Should have at least one result");
        // Top scored preset must explain itself.
        assert!(
            !top.reasons.is_empty(),
            "Top result should have scoring reasons"
        );
    }

    #[test]
    fn test_1080p30_streaming_prefers_streaming_presets() {
        let lib = library();
        let criteria = SelectionCriteria::streaming_1080p();
        // Use top 20 — many 1080p presets exist across categories; we assert the
        // selection system at least surfaces streaming presets in the top 20.
        let results = OptimalPresetSelector::top_n(&criteria, &lib, 20);
        let any_streaming = results.iter().any(|r| {
            r.preset.has_tag("hls")
                || r.preset.has_tag("streaming")
                || r.preset.has_tag("rtmp")
                || r.preset.has_tag("dash")
                || r.preset.has_tag("srt")
        });
        assert!(
            any_streaming,
            "Top 20 for streaming 1080p should include a streaming preset"
        );
    }

    #[test]
    fn test_mobile_criteria_prefers_mobile_presets() {
        let lib = library();
        let criteria = SelectionCriteria::mobile_720p();
        // Use top 20 — mobile presets need to be within the top 20 by score.
        let results = OptimalPresetSelector::top_n(&criteria, &lib, 20);
        let any_mobile = results.iter().any(|r| {
            r.preset.has_tag("mobile") || r.preset.has_tag("ios") || r.preset.has_tag("android")
        });
        assert!(
            any_mobile,
            "Top 20 for mobile should include a mobile-tagged preset"
        );
    }

    #[test]
    fn test_archive_criteria_prefers_archive_presets() {
        let lib = library();
        let criteria = SelectionCriteria::archive_4k();
        let results = OptimalPresetSelector::top_n(&criteria, &lib, 10);
        let any_archive = results.iter().any(|r| {
            r.preset.has_tag("archive")
                || r.preset.has_tag("lossless")
                || r.preset.has_tag("mezzanine")
        });
        assert!(
            any_archive,
            "Top 10 for archive should include an archive-tagged preset"
        );
    }

    #[test]
    fn test_top_n_respects_limit() {
        let lib = library();
        let criteria = SelectionCriteria::streaming_1080p();
        let results = OptimalPresetSelector::top_n(&criteria, &lib, 3);
        assert!(results.len() <= 3, "top_n(3) must return at most 3 results");
    }

    #[test]
    fn test_score_bounded_0_100() {
        let lib = library();
        let criteria = SelectionCriteria::streaming_1080p();
        let results = OptimalPresetSelector::select(&criteria, &lib);
        for r in &results {
            assert!(
                r.score >= 0.0 && r.score <= 100.0,
                "Score {} out of [0, 100] for preset {}",
                r.score,
                r.preset.metadata.id
            );
        }
    }

    #[test]
    fn test_scored_preset_arc_cheap_clone() {
        let lib = library();
        let criteria = SelectionCriteria::streaming_1080p();
        let results = OptimalPresetSelector::select(&criteria, &lib);
        if let Some(first) = results.first() {
            // Cloning an Arc<Preset> should not deep-clone the preset data.
            let cloned = Arc::clone(&first.preset);
            assert!(
                Arc::ptr_eq(&first.preset, &cloned),
                "Arc clone should point to same allocation"
            );
        }
    }

    #[test]
    fn test_zero_bitrate_criteria_still_scores() {
        let lib = library();
        let criteria = SelectionCriteria {
            target_bitrate_kbps: 0,
            width: 1920,
            height: 1080,
            frame_rate: 30.0,
            use_case: UseCase::Streaming,
        };
        let results = OptimalPresetSelector::select(&criteria, &lib);
        assert!(
            !results.is_empty(),
            "Zero bitrate criteria should still return presets"
        );
    }

    #[test]
    fn test_zero_resolution_criteria_still_scores() {
        let lib = library();
        let criteria = SelectionCriteria {
            target_bitrate_kbps: 4_000,
            width: 0,
            height: 0,
            frame_rate: 0.0,
            use_case: UseCase::Streaming,
        };
        let results = OptimalPresetSelector::select(&criteria, &lib);
        // Bitrate score alone should differentiate presets.
        assert!(!results.is_empty());
    }

    #[test]
    fn test_select_all_use_cases_return_results() {
        let lib = library();
        for uc in [
            UseCase::Streaming,
            UseCase::Archive,
            UseCase::Social,
            UseCase::Broadcast,
            UseCase::Mobile,
        ] {
            let criteria = SelectionCriteria {
                target_bitrate_kbps: 5_000,
                width: 1920,
                height: 1080,
                frame_rate: 30.0,
                use_case: uc,
            };
            let results = OptimalPresetSelector::select(&criteria, &lib);
            assert!(!results.is_empty(), "UseCase {uc:?} should return results");
        }
    }

    #[test]
    fn test_broadcast_criteria_prefers_broadcast_presets() {
        let lib = library();
        let criteria = SelectionCriteria {
            target_bitrate_kbps: 15_000,
            width: 1920,
            height: 1080,
            frame_rate: 29.97,
            use_case: UseCase::Broadcast,
        };
        let results = OptimalPresetSelector::top_n(&criteria, &lib, 10);
        let any_broadcast = results.iter().any(|r| {
            r.preset.has_tag("broadcast") || r.preset.has_tag("atsc") || r.preset.has_tag("dvb")
        });
        assert!(
            any_broadcast,
            "Top 10 for broadcast should include a broadcast-tagged preset"
        );
    }

    #[test]
    fn test_social_criteria_prefers_social_presets() {
        let lib = library();
        let criteria = SelectionCriteria {
            target_bitrate_kbps: 3_500,
            width: 1080,
            height: 1920,
            frame_rate: 30.0,
            use_case: UseCase::Social,
        };
        let results = OptimalPresetSelector::top_n(&criteria, &lib, 10);
        let any_social = results.iter().any(|r| {
            r.preset.has_tag("social")
                || r.preset.has_tag("instagram")
                || r.preset.has_tag("tiktok")
                || r.preset.has_tag("reels")
        });
        assert!(
            any_social,
            "Top 10 for social should include a social-tagged preset"
        );
    }
}
