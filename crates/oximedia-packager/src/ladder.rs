//! Bitrate ladder generation for adaptive streaming.

use crate::config::{BitrateEntry, BitrateLadder};
use crate::error::{PackagerError, PackagerResult};
use tracing::{debug, info};

/// Video quality preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualityPreset {
    /// Low quality (mobile).
    Low,
    /// Medium quality (SD).
    Medium,
    /// High quality (HD).
    High,
    /// Very high quality (Full HD).
    VeryHigh,
    /// Ultra quality (4K).
    Ultra,
}

/// Source video information.
#[derive(Debug, Clone)]
pub struct SourceInfo {
    /// Source width.
    pub width: u32,
    /// Source height.
    pub height: u32,
    /// Source bitrate (if known).
    pub bitrate: Option<u32>,
    /// Source frame rate.
    pub framerate: f64,
    /// Source codec.
    pub codec: String,
}

impl SourceInfo {
    /// Create new source info.
    #[must_use]
    pub fn new(width: u32, height: u32, framerate: f64, codec: String) -> Self {
        Self {
            width,
            height,
            bitrate: None,
            framerate,
            codec,
        }
    }

    /// Set the source bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bitrate: u32) -> Self {
        self.bitrate = Some(bitrate);
        self
    }

    /// Get the aspect ratio.
    #[must_use]
    pub fn aspect_ratio(&self) -> f64 {
        f64::from(self.width) / f64::from(self.height)
    }

    /// Check if source is 4K or higher.
    #[must_use]
    pub fn is_4k_or_higher(&self) -> bool {
        self.width >= 3840 || self.height >= 2160
    }

    /// Check if source is 1080p or higher.
    #[must_use]
    pub fn is_1080p_or_higher(&self) -> bool {
        self.width >= 1920 || self.height >= 1080
    }

    /// Check if source is 720p or higher.
    #[must_use]
    pub fn is_720p_or_higher(&self) -> bool {
        self.width >= 1280 || self.height >= 720
    }
}

/// Bitrate ladder generator.
pub struct LadderGenerator {
    source: SourceInfo,
    codec: String,
    min_bitrate: u32,
    max_bitrate: Option<u32>,
}

impl LadderGenerator {
    /// Create a new ladder generator.
    #[must_use]
    pub fn new(source: SourceInfo) -> Self {
        Self {
            source,
            codec: "av1".to_string(),
            min_bitrate: 250_000, // 250 kbps
            max_bitrate: None,
        }
    }

    /// Set the target codec.
    #[must_use]
    pub fn with_codec(mut self, codec: &str) -> Self {
        self.codec = codec.to_string();
        self
    }

    /// Set the minimum bitrate.
    #[must_use]
    pub fn with_min_bitrate(mut self, bitrate: u32) -> Self {
        self.min_bitrate = bitrate;
        self
    }

    /// Set the maximum bitrate.
    #[must_use]
    pub fn with_max_bitrate(mut self, bitrate: u32) -> Self {
        self.max_bitrate = Some(bitrate);
        self
    }

    /// Generate a bitrate ladder.
    pub fn generate(&self) -> PackagerResult<BitrateLadder> {
        info!(
            "Generating bitrate ladder for {}x{} source",
            self.source.width, self.source.height
        );

        let mut ladder = BitrateLadder::new();
        let entries = self.generate_entries()?;

        for entry in entries {
            debug!(
                "Adding ladder entry: {}x{} @ {} bps",
                entry.width, entry.height, entry.bitrate
            );
            ladder.add_entry(entry);
        }

        ladder.auto_generate = false;
        Ok(ladder)
    }

    /// Generate ladder entries based on source.
    fn generate_entries(&self) -> PackagerResult<Vec<BitrateEntry>> {
        let mut entries = Vec::new();

        // Determine which resolutions to include based on source
        if self.source.is_4k_or_higher() {
            // 4K source: generate 4K, 1080p, 720p, 480p, 360p
            entries.extend(self.create_4k_ladder()?);
        } else if self.source.is_1080p_or_higher() {
            // 1080p source: generate 1080p, 720p, 480p, 360p
            entries.extend(self.create_1080p_ladder()?);
        } else if self.source.is_720p_or_higher() {
            // 720p source: generate 720p, 480p, 360p
            entries.extend(self.create_720p_ladder()?);
        } else {
            // SD source: generate source resolution and lower
            entries.extend(self.create_sd_ladder()?);
        }

        // Filter out entries that exceed max bitrate or are below min bitrate
        entries.retain(|e| {
            e.bitrate >= self.min_bitrate && self.max_bitrate.map_or(true, |max| e.bitrate <= max)
        });

        if entries.is_empty() {
            return Err(PackagerError::InvalidLadder(
                "No valid bitrate entries generated".to_string(),
            ));
        }

        Ok(entries)
    }

    /// Create 4K bitrate ladder.
    fn create_4k_ladder(&self) -> PackagerResult<Vec<BitrateEntry>> {
        let ar = self.source.aspect_ratio();
        let mut entries = Vec::new();

        // 4K (3840x2160 or adjusted for aspect ratio)
        let (width_4k, height_4k) = self.adjust_resolution(3840, 2160, ar);
        entries.push(
            BitrateEntry::new(
                self.calculate_bitrate(width_4k, height_4k),
                width_4k,
                height_4k,
                &self.codec,
            )
            .with_framerate(self.source.framerate),
        );

        // 1080p
        let (width_1080, height_1080) = self.adjust_resolution(1920, 1080, ar);
        entries.push(
            BitrateEntry::new(
                self.calculate_bitrate(width_1080, height_1080),
                width_1080,
                height_1080,
                &self.codec,
            )
            .with_framerate(self.source.framerate),
        );

        // 720p
        let (width_720, height_720) = self.adjust_resolution(1280, 720, ar);
        entries.push(
            BitrateEntry::new(
                self.calculate_bitrate(width_720, height_720),
                width_720,
                height_720,
                &self.codec,
            )
            .with_framerate(self.source.framerate),
        );

        // 480p
        entries.push(self.create_480p_entry(ar)?);

        // 360p
        entries.push(self.create_360p_entry(ar)?);

        Ok(entries)
    }

    /// Create 1080p bitrate ladder.
    fn create_1080p_ladder(&self) -> PackagerResult<Vec<BitrateEntry>> {
        let ar = self.source.aspect_ratio();
        let mut entries = Vec::new();

        // 1080p
        let (width_1080, height_1080) = self.adjust_resolution(1920, 1080, ar);
        entries.push(
            BitrateEntry::new(
                self.calculate_bitrate(width_1080, height_1080),
                width_1080,
                height_1080,
                &self.codec,
            )
            .with_framerate(self.source.framerate),
        );

        // 720p
        let (width_720, height_720) = self.adjust_resolution(1280, 720, ar);
        entries.push(
            BitrateEntry::new(
                self.calculate_bitrate(width_720, height_720),
                width_720,
                height_720,
                &self.codec,
            )
            .with_framerate(self.source.framerate),
        );

        // 480p
        entries.push(self.create_480p_entry(ar)?);

        // 360p
        entries.push(self.create_360p_entry(ar)?);

        Ok(entries)
    }

    /// Create 720p bitrate ladder.
    fn create_720p_ladder(&self) -> PackagerResult<Vec<BitrateEntry>> {
        let ar = self.source.aspect_ratio();
        let mut entries = Vec::new();

        // 720p
        let (width_720, height_720) = self.adjust_resolution(1280, 720, ar);
        entries.push(
            BitrateEntry::new(
                self.calculate_bitrate(width_720, height_720),
                width_720,
                height_720,
                &self.codec,
            )
            .with_framerate(self.source.framerate),
        );

        // 480p
        entries.push(self.create_480p_entry(ar)?);

        // 360p
        entries.push(self.create_360p_entry(ar)?);

        Ok(entries)
    }

    /// Create SD bitrate ladder.
    fn create_sd_ladder(&self) -> PackagerResult<Vec<BitrateEntry>> {
        let ar = self.source.aspect_ratio();
        let mut entries = Vec::new();

        // Source resolution
        entries.push(
            BitrateEntry::new(
                self.calculate_bitrate(self.source.width, self.source.height),
                self.source.width,
                self.source.height,
                &self.codec,
            )
            .with_framerate(self.source.framerate),
        );

        // 360p if source is larger
        if self.source.height > 360 {
            entries.push(self.create_360p_entry(ar)?);
        }

        // 240p for mobile
        if self.source.height > 240 {
            let (width_240, height_240) = self.adjust_resolution(426, 240, ar);
            entries.push(
                BitrateEntry::new(
                    self.calculate_bitrate(width_240, height_240),
                    width_240,
                    height_240,
                    &self.codec,
                )
                .with_framerate(self.source.framerate),
            );
        }

        Ok(entries)
    }

    /// Create 480p entry.
    fn create_480p_entry(&self, aspect_ratio: f64) -> PackagerResult<BitrateEntry> {
        let (width, height) = self.adjust_resolution(854, 480, aspect_ratio);
        Ok(BitrateEntry::new(
            self.calculate_bitrate(width, height),
            width,
            height,
            &self.codec,
        )
        .with_framerate(self.source.framerate))
    }

    /// Create 360p entry.
    fn create_360p_entry(&self, aspect_ratio: f64) -> PackagerResult<BitrateEntry> {
        let (width, height) = self.adjust_resolution(640, 360, aspect_ratio);
        Ok(BitrateEntry::new(
            self.calculate_bitrate(width, height),
            width,
            height,
            &self.codec,
        )
        .with_framerate(self.source.framerate))
    }

    /// Adjust resolution to match aspect ratio.
    fn adjust_resolution(&self, width: u32, height: u32, target_ar: f64) -> (u32, u32) {
        let current_ar = f64::from(width) / f64::from(height);

        if (current_ar - target_ar).abs() < 0.01 {
            return (width, height);
        }

        // Adjust width to match target aspect ratio
        let adjusted_width = (f64::from(height) * target_ar).round() as u32;
        // Ensure even dimensions for video encoding
        let adjusted_width = (adjusted_width / 2) * 2;

        (adjusted_width, height)
    }

    /// Calculate bitrate based on resolution and codec.
    fn calculate_bitrate(&self, width: u32, height: u32) -> u32 {
        let pixels = u64::from(width) * u64::from(height);
        let fps = self.source.framerate;

        // Base bitrate calculation (bits per pixel)
        let base_bpp = match self.codec.as_str() {
            "av1" => 0.05, // AV1 is most efficient
            "vp9" => 0.08, // VP9 is efficient
            "vp8" => 0.12, // VP8 is less efficient
            _ => 0.08,     // Default to VP9 efficiency
        };

        // Adjust for frame rate
        let fps_factor = if fps > 30.0 {
            1.0 + ((fps - 30.0) / 30.0) * 0.3
        } else {
            fps / 30.0
        };

        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let bitrate = (pixels as f64 * base_bpp * fps_factor) as u32;

        // Clamp to reasonable range
        bitrate.max(250_000).min(50_000_000)
    }
}

/// Pre-defined bitrate ladder presets.
pub struct LadderPresets;

impl LadderPresets {
    /// Get a standard HLS ladder for 1080p content.
    #[must_use]
    pub fn hls_1080p() -> BitrateLadder {
        let mut ladder = BitrateLadder::new();

        ladder.add_entry(BitrateEntry::new(5_000_000, 1920, 1080, "av1"));
        ladder.add_entry(BitrateEntry::new(3_000_000, 1280, 720, "av1"));
        ladder.add_entry(BitrateEntry::new(1_500_000, 854, 480, "av1"));
        ladder.add_entry(BitrateEntry::new(800_000, 640, 360, "av1"));

        ladder.auto_generate = false;
        ladder
    }

    /// Get a standard DASH ladder for 4K content.
    #[must_use]
    pub fn dash_4k() -> BitrateLadder {
        let mut ladder = BitrateLadder::new();

        ladder.add_entry(BitrateEntry::new(15_000_000, 3840, 2160, "av1"));
        ladder.add_entry(BitrateEntry::new(8_000_000, 2560, 1440, "av1"));
        ladder.add_entry(BitrateEntry::new(5_000_000, 1920, 1080, "av1"));
        ladder.add_entry(BitrateEntry::new(3_000_000, 1280, 720, "av1"));
        ladder.add_entry(BitrateEntry::new(1_500_000, 854, 480, "av1"));

        ladder.auto_generate = false;
        ladder
    }

    /// Get a mobile-optimized ladder.
    #[must_use]
    pub fn mobile_optimized() -> BitrateLadder {
        let mut ladder = BitrateLadder::new();

        ladder.add_entry(BitrateEntry::new(1_500_000, 854, 480, "av1"));
        ladder.add_entry(BitrateEntry::new(800_000, 640, 360, "av1"));
        ladder.add_entry(BitrateEntry::new(400_000, 426, 240, "av1"));

        ladder.auto_generate = false;
        ladder
    }
}

// ---------------------------------------------------------------------------
// SourceAnalysis / LadderRung / BitrateLadderGenerator
// ---------------------------------------------------------------------------

/// Per-title source analysis used for complexity-aware ladder generation.
#[derive(Debug, Clone)]
pub struct SourceAnalysis {
    /// Source frame width in pixels.
    pub width: u32,
    /// Source frame height in pixels.
    pub height: u32,
    /// Source bitrate in bits/s, if known (used as an upper-bound cap).
    pub bitrate_bps: Option<u64>,
    /// Complexity score in `[0.0, 1.0]`.
    ///
    /// `0.0` = low complexity (static/talking-head); `1.0` = high complexity
    /// (fast motion, film grain, animation with detail).
    pub complexity_score: f64,
}

impl SourceAnalysis {
    /// Create a new source analysis.
    #[must_use]
    pub fn new(width: u32, height: u32, complexity_score: f64) -> Self {
        Self {
            width,
            height,
            bitrate_bps: None,
            complexity_score,
        }
    }

    /// Attach a known source bitrate.
    #[must_use]
    pub fn with_bitrate(mut self, bitrate_bps: u64) -> Self {
        self.bitrate_bps = Some(bitrate_bps);
        self
    }

    /// Total pixel count.
    #[must_use]
    pub fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// A single rung in an automatically generated encoding ladder.
#[derive(Debug, Clone)]
pub struct LadderRung {
    /// Target encode width in pixels.
    pub width: u32,
    /// Target encode height in pixels.
    pub height: u32,
    /// Target encode bitrate in bits/s.
    pub target_bitrate_bps: u64,
    /// Target codec identifier (default: `"av1"`).
    pub codec: String,
}

impl LadderRung {
    /// Pixel count for this rung.
    #[must_use]
    pub fn pixels(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

/// Complexity-aware, per-title bitrate ladder generator.
///
/// Given a [`SourceAnalysis`] describing resolution, optional source bitrate,
/// and a complexity score, `BitrateLadderGenerator` produces an ordered list
/// of [`LadderRung`] encoding targets (highest resolution first).
///
/// # Resolution → rung count mapping
///
/// | Source resolution | Rungs generated              |
/// |-------------------|------------------------------|
/// | 4K (≥ 3840×2160)  | 2160p, 1080p, 720p, 480p     |
/// | 1080p (≥ 1920×1080) | 1080p, 720p, 480p           |
/// | 720p (≥ 1280×720) | 720p, 480p                   |
/// | Below 720p        | Source resolution             |
///
/// # Bitrate formula
///
/// ```text
/// complexity_multiplier = 0.7 + 0.6 * complexity_score   // range [0.7, 1.3]
/// target_bitrate        = base_bitrate * complexity_multiplier
/// ```
///
/// If the source bitrate is known, each rung is additionally capped at
/// `source_bitrate * (rung_pixels / source_pixels)` to avoid inflating the
/// bitrate beyond what the source actually carries.
///
/// All rungs are guaranteed to have `target_bitrate_bps >= 200_000`.
///
/// # Example
///
/// ```
/// use oximedia_packager::ladder::{BitrateLadderGenerator, SourceAnalysis};
///
/// let analysis = SourceAnalysis::new(1920, 1080, 0.5);
/// let generator = BitrateLadderGenerator::new(analysis);
/// let rungs = generator.generate().expect("should succeed");
/// assert_eq!(rungs.len(), 3); // 1080p source → 3 rungs
/// ```
pub struct BitrateLadderGenerator {
    analysis: SourceAnalysis,
    codec: String,
}

/// Base bitrates (in bits/s) for AV1 at 30 fps per canonical resolution.
const BASE_2160P: u64 = 8_000_000;
const BASE_1080P: u64 = 3_500_000;
const BASE_720P: u64 = 2_000_000;
const BASE_480P: u64 = 1_000_000;
const MIN_BITRATE: u64 = 200_000;

impl BitrateLadderGenerator {
    /// Create a new generator.
    #[must_use]
    pub fn new(analysis: SourceAnalysis) -> Self {
        Self {
            analysis,
            codec: "av1".to_string(),
        }
    }

    /// Override the target codec (default: `"av1"`).
    #[must_use]
    pub fn with_codec(mut self, codec: impl Into<String>) -> Self {
        self.codec = codec.into();
        self
    }

    /// Generate the bitrate ladder rungs (highest resolution first).
    ///
    /// # Errors
    ///
    /// Returns [`PackagerError::InvalidConfig`] if `complexity_score` is
    /// outside `[0.0, 1.0]`.
    pub fn generate(&self) -> PackagerResult<Vec<LadderRung>> {
        if !(0.0..=1.0).contains(&self.analysis.complexity_score) {
            return Err(PackagerError::InvalidConfig(format!(
                "complexity_score must be in [0.0, 1.0], got {}",
                self.analysis.complexity_score
            )));
        }

        let multiplier = 0.7 + 0.6 * self.analysis.complexity_score;
        let src_w = self.analysis.width;
        let src_h = self.analysis.height;

        // Determine canonical rung resolutions based on source dimensions.
        let canonical: Vec<(u32, u32, u64)> = if src_w >= 3840 || src_h >= 2160 {
            vec![
                (3840, 2160, BASE_2160P),
                (1920, 1080, BASE_1080P),
                (1280, 720, BASE_720P),
                (854, 480, BASE_480P),
            ]
        } else if src_w >= 1920 || src_h >= 1080 {
            vec![
                (1920, 1080, BASE_1080P),
                (1280, 720, BASE_720P),
                (854, 480, BASE_480P),
            ]
        } else if src_w >= 1280 || src_h >= 720 {
            vec![(1280, 720, BASE_720P), (854, 480, BASE_480P)]
        } else {
            // Sub-720p: single rung at source resolution
            let base = self.estimate_base_bitrate_for_resolution(src_w, src_h);
            vec![(src_w, src_h, base)]
        };

        let src_pixels = self.analysis.pixels();

        let rungs: Vec<LadderRung> = canonical
            .into_iter()
            .map(|(w, h, base)| {
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let mut bitrate = (base as f64 * multiplier) as u64;

                // Apply source-bitrate cap if available.
                if let Some(src_bps) = self.analysis.bitrate_bps {
                    let rung_pixels = u64::from(w) * u64::from(h);
                    if src_pixels > 0 {
                        let cap = src_bps
                            .saturating_mul(rung_pixels)
                            .saturating_div(src_pixels);
                        bitrate = bitrate.min(cap);
                    }
                }

                // Enforce minimum.
                bitrate = bitrate.max(MIN_BITRATE);

                LadderRung {
                    width: w,
                    height: h,
                    target_bitrate_bps: bitrate,
                    codec: self.codec.clone(),
                }
            })
            .collect();

        Ok(rungs)
    }

    /// Estimate a reasonable base bitrate for an arbitrary source resolution.
    fn estimate_base_bitrate_for_resolution(&self, width: u32, height: u32) -> u64 {
        let pixels = u64::from(width) * u64::from(height);
        // Scale from 1080p base proportionally by pixel count.
        let ref_pixels = 1920u64 * 1080;
        if ref_pixels == 0 {
            return BASE_1080P;
        }
        BASE_1080P
            .saturating_mul(pixels)
            .saturating_div(ref_pixels)
            .max(MIN_BITRATE)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_source_info_aspect_ratio() {
        let source = SourceInfo::new(1920, 1080, 30.0, "av1".to_string());
        assert!((source.aspect_ratio() - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_ladder_generation_1080p() {
        let source = SourceInfo::new(1920, 1080, 30.0, "av1".to_string());
        let generator = LadderGenerator::new(source);
        let ladder = generator.generate().expect("should succeed in test");

        assert!(!ladder.entries.is_empty());
        assert!(ladder.entries.iter().any(|e| e.height == 1080));
        assert!(ladder.entries.iter().any(|e| e.height == 720));
        assert!(ladder.entries.iter().any(|e| e.height == 360));
    }

    #[test]
    fn test_bitrate_calculation() {
        let source = SourceInfo::new(1920, 1080, 30.0, "av1".to_string());
        let generator = LadderGenerator::new(source);

        let bitrate_1080 = generator.calculate_bitrate(3840, 2160);
        let bitrate_720 = generator.calculate_bitrate(1920, 1080);

        assert!(bitrate_1080 > bitrate_720);
    }

    // --- BitrateLadderGenerator tests ---------------------------------------

    #[test]
    fn test_bitrate_ladder_gen_4k_produces_4_rungs() {
        let analysis = SourceAnalysis::new(3840, 2160, 0.5);
        let gen = BitrateLadderGenerator::new(analysis);
        let rungs = gen.generate().expect("should succeed");
        assert_eq!(rungs.len(), 4, "4K source should produce 4 rungs");
    }

    #[test]
    fn test_bitrate_ladder_gen_1080p_produces_3_rungs() {
        let analysis = SourceAnalysis::new(1920, 1080, 0.5);
        let gen = BitrateLadderGenerator::new(analysis);
        let rungs = gen.generate().expect("should succeed");
        assert_eq!(rungs.len(), 3, "1080p source should produce 3 rungs");
    }

    #[test]
    fn test_bitrate_ladder_gen_720p_produces_2_rungs() {
        let analysis = SourceAnalysis::new(1280, 720, 0.5);
        let gen = BitrateLadderGenerator::new(analysis);
        let rungs = gen.generate().expect("should succeed");
        assert_eq!(rungs.len(), 2, "720p source should produce 2 rungs");
    }

    #[test]
    fn test_bitrate_ladder_gen_sub720p_produces_1_rung() {
        let analysis = SourceAnalysis::new(640, 480, 0.5);
        let gen = BitrateLadderGenerator::new(analysis);
        let rungs = gen.generate().expect("should succeed");
        assert_eq!(rungs.len(), 1, "sub-720p source should produce 1 rung");
    }

    #[test]
    fn test_bitrate_ladder_gen_high_complexity_raises_bitrate() {
        let lo = BitrateLadderGenerator::new(SourceAnalysis::new(1920, 1080, 0.0))
            .generate()
            .expect("should succeed");
        let hi = BitrateLadderGenerator::new(SourceAnalysis::new(1920, 1080, 1.0))
            .generate()
            .expect("should succeed");

        for (l, h) in lo.iter().zip(hi.iter()) {
            assert!(
                h.target_bitrate_bps > l.target_bitrate_bps,
                "high complexity should produce higher bitrate"
            );
        }
    }

    #[test]
    fn test_bitrate_ladder_gen_invalid_complexity_error() {
        let gen = BitrateLadderGenerator::new(SourceAnalysis::new(1920, 1080, 1.5));
        assert!(gen.generate().is_err(), "complexity > 1.0 should be error");

        let gen2 = BitrateLadderGenerator::new(SourceAnalysis::new(1920, 1080, -0.1));
        assert!(gen2.generate().is_err(), "complexity < 0.0 should be error");
    }

    #[test]
    fn test_bitrate_ladder_gen_source_bitrate_cap() {
        // Very low source bitrate should cap rung bitrates.
        let analysis =
            SourceAnalysis::new(1920, 1080, 0.5).with_bitrate(500_000 /* very low */);
        let gen = BitrateLadderGenerator::new(analysis);
        let rungs = gen.generate().expect("should succeed");

        // The 1080p rung should be capped at/below 500_000 (or at min floor).
        let top_rung = &rungs[0];
        // The cap for 1080p rung = 500_000 * (1920*1080) / (1920*1080) = 500_000
        assert!(
            top_rung.target_bitrate_bps <= 500_000,
            "rung bitrate should be capped at source bitrate, got {}",
            top_rung.target_bitrate_bps
        );
    }

    #[test]
    fn test_bitrate_ladder_gen_all_rungs_above_minimum() {
        let analysis = SourceAnalysis::new(3840, 2160, 0.0); // low complexity
        let gen = BitrateLadderGenerator::new(analysis);
        let rungs = gen.generate().expect("should succeed");

        for rung in &rungs {
            assert!(
                rung.target_bitrate_bps >= MIN_BITRATE,
                "rung bitrate {} below minimum {}",
                rung.target_bitrate_bps,
                MIN_BITRATE
            );
        }
    }

    #[test]
    fn test_bitrate_ladder_gen_rungs_ordered_highest_first() {
        let gen = BitrateLadderGenerator::new(SourceAnalysis::new(3840, 2160, 0.5));
        let rungs = gen.generate().expect("should succeed");

        // Resolution (pixels) should be strictly descending
        for w in rungs.windows(2) {
            assert!(
                w[0].pixels() >= w[1].pixels(),
                "rungs should be highest resolution first"
            );
        }
    }

    #[test]
    fn test_bitrate_ladder_gen_low_complexity_base_bitrates() {
        // complexity=0.0 → multiplier=0.7
        let gen = BitrateLadderGenerator::new(SourceAnalysis::new(1920, 1080, 0.0));
        let rungs = gen.generate().expect("should succeed");

        // 1080p rung base is 3_500_000; * 0.7 = 2_450_000
        let expected_1080p = (3_500_000_f64 * 0.7) as u64;
        assert_eq!(
            rungs[0].target_bitrate_bps, expected_1080p,
            "1080p rung at complexity=0 should be base * 0.7"
        );
    }

    #[test]
    fn test_bitrate_ladder_gen_codec_field() {
        let gen =
            BitrateLadderGenerator::new(SourceAnalysis::new(1280, 720, 0.5)).with_codec("vp9");
        let rungs = gen.generate().expect("should succeed");
        for rung in &rungs {
            assert_eq!(rung.codec, "vp9");
        }
    }
}
