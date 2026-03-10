#![allow(dead_code)]
//! Quality ladder generation for adaptive bitrate streaming.
//!
//! This module generates optimal sets of encoding renditions (quality "rungs")
//! for ABR delivery. It analyzes source content complexity to determine the
//! best resolution-bitrate pairs, avoiding wasteful over-encoding of simple
//! content and under-encoding of complex content.

use std::fmt;

/// Resolution descriptor for a quality rung.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct LadderResolution {
    /// Width in pixels.
    pub width: u32,
    /// Height in pixels.
    pub height: u32,
}

impl LadderResolution {
    /// Creates a new resolution descriptor.
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }

    /// Returns the total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Returns the aspect ratio as a floating point value.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn aspect_ratio(&self) -> f64 {
        if self.height == 0 {
            return 0.0;
        }
        self.width as f64 / self.height as f64
    }
}

impl fmt::Display for LadderResolution {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}x{}", self.width, self.height)
    }
}

/// A single rung in the quality ladder.
#[derive(Debug, Clone)]
pub struct QualityRung {
    /// Resolution for this rung.
    pub resolution: LadderResolution,
    /// Target bitrate in bits per second.
    pub bitrate_bps: u64,
    /// Target frame rate.
    pub frame_rate: f64,
    /// CRF or QP value (if applicable).
    pub crf: Option<f64>,
    /// Estimated VMAF score for this rung.
    pub estimated_vmaf: Option<f64>,
    /// Label for this rung (e.g., "1080p", "720p").
    pub label: String,
}

impl QualityRung {
    /// Creates a new quality rung.
    #[must_use]
    pub fn new(resolution: LadderResolution, bitrate_bps: u64, frame_rate: f64) -> Self {
        Self {
            resolution,
            bitrate_bps,
            frame_rate,
            crf: None,
            estimated_vmaf: None,
            label: format!("{}p", resolution.height),
        }
    }

    /// Sets the CRF value.
    #[must_use]
    pub fn with_crf(mut self, crf: f64) -> Self {
        self.crf = Some(crf);
        self
    }

    /// Sets the estimated VMAF.
    #[must_use]
    pub fn with_vmaf(mut self, vmaf: f64) -> Self {
        self.estimated_vmaf = Some(vmaf);
        self
    }

    /// Sets a custom label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = label.into();
        self
    }

    /// Returns bitrate in kilobits per second.
    #[must_use]
    pub fn bitrate_kbps(&self) -> u64 {
        self.bitrate_bps / 1000
    }

    /// Returns bitrate in megabits per second.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn bitrate_mbps(&self) -> f64 {
        self.bitrate_bps as f64 / 1_000_000.0
    }

    /// Returns bits per pixel per frame.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn bits_per_pixel(&self) -> f64 {
        let pixels = self.resolution.pixel_count();
        if pixels == 0 || self.frame_rate <= 0.0 {
            return 0.0;
        }
        self.bitrate_bps as f64 / (pixels as f64 * self.frame_rate)
    }
}

/// Content complexity classification for ladder generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContentComplexity {
    /// Very simple content (static slides, simple animation).
    VeryLow,
    /// Low complexity (talking head, slow motion).
    Low,
    /// Medium complexity (typical video content).
    Medium,
    /// High complexity (sports, action scenes).
    High,
    /// Very high complexity (dense detail, fast motion).
    VeryHigh,
}

/// Configuration for quality ladder generation.
#[derive(Debug, Clone)]
pub struct LadderConfig {
    /// Maximum resolution to include.
    pub max_resolution: LadderResolution,
    /// Minimum resolution to include.
    pub min_resolution: LadderResolution,
    /// Maximum bitrate in bps.
    pub max_bitrate_bps: u64,
    /// Minimum bitrate in bps.
    pub min_bitrate_bps: u64,
    /// Maximum number of rungs.
    pub max_rungs: usize,
    /// Target frame rate.
    pub frame_rate: f64,
    /// Minimum VMAF threshold (skip rungs below this).
    pub min_vmaf: f64,
    /// Whether to include audio-only rung.
    pub include_audio_only: bool,
    /// Content complexity hint.
    pub complexity: ContentComplexity,
}

impl Default for LadderConfig {
    fn default() -> Self {
        Self {
            max_resolution: LadderResolution::new(1920, 1080),
            min_resolution: LadderResolution::new(426, 240),
            max_bitrate_bps: 8_000_000,
            min_bitrate_bps: 200_000,
            max_rungs: 8,
            frame_rate: 30.0,
            min_vmaf: 40.0,
            include_audio_only: false,
            complexity: ContentComplexity::Medium,
        }
    }
}

/// Standard resolution presets.
#[derive(Debug, Clone)]
pub struct StandardResolutions;

impl StandardResolutions {
    /// 4K UHD resolution.
    #[must_use]
    pub fn uhd_4k() -> LadderResolution {
        LadderResolution::new(3840, 2160)
    }

    /// 1080p Full HD resolution.
    #[must_use]
    pub fn fhd_1080p() -> LadderResolution {
        LadderResolution::new(1920, 1080)
    }

    /// 720p HD resolution.
    #[must_use]
    pub fn hd_720p() -> LadderResolution {
        LadderResolution::new(1280, 720)
    }

    /// 480p SD resolution.
    #[must_use]
    pub fn sd_480p() -> LadderResolution {
        LadderResolution::new(854, 480)
    }

    /// 360p low resolution.
    #[must_use]
    pub fn low_360p() -> LadderResolution {
        LadderResolution::new(640, 360)
    }

    /// 240p minimum resolution.
    #[must_use]
    pub fn min_240p() -> LadderResolution {
        LadderResolution::new(426, 240)
    }
}

/// Quality ladder generator.
#[derive(Debug, Clone)]
pub struct QualityLadderGenerator {
    /// Configuration for ladder generation.
    config: LadderConfig,
}

impl QualityLadderGenerator {
    /// Creates a new generator with the given configuration.
    #[must_use]
    pub fn new(config: LadderConfig) -> Self {
        Self { config }
    }

    /// Creates a generator with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(LadderConfig::default())
    }

    /// Returns the configuration.
    #[must_use]
    pub fn config(&self) -> &LadderConfig {
        &self.config
    }

    /// Generates a quality ladder based on configuration and complexity.
    #[must_use]
    pub fn generate(&self) -> QualityLadder {
        let resolutions = self.select_resolutions();
        let bitrates = self.compute_bitrates(&resolutions);

        let mut rungs = Vec::new();
        for (res, bitrate) in resolutions.iter().zip(bitrates.iter()) {
            let rung = QualityRung::new(*res, *bitrate, self.config.frame_rate);
            rungs.push(rung);
        }

        QualityLadder { rungs }
    }

    /// Selects resolutions based on max/min and max rungs.
    fn select_resolutions(&self) -> Vec<LadderResolution> {
        let standard = [
            StandardResolutions::uhd_4k(),
            StandardResolutions::fhd_1080p(),
            StandardResolutions::hd_720p(),
            StandardResolutions::sd_480p(),
            StandardResolutions::low_360p(),
            StandardResolutions::min_240p(),
        ];

        let max_pixels = self.config.max_resolution.pixel_count();
        let min_pixels = self.config.min_resolution.pixel_count();

        let filtered: Vec<LadderResolution> = standard
            .iter()
            .filter(|r| {
                let px = r.pixel_count();
                px >= min_pixels && px <= max_pixels
            })
            .copied()
            .collect();

        if filtered.len() <= self.config.max_rungs {
            filtered
        } else {
            // Evenly sample from the filtered list
            let step = filtered.len() as f64 / self.config.max_rungs as f64;
            let mut result = Vec::new();
            let mut idx = 0.0_f64;
            while result.len() < self.config.max_rungs && (idx as usize) < filtered.len() {
                result.push(filtered[idx as usize]);
                idx += step;
            }
            result
        }
    }

    /// Computes target bitrates for each resolution based on complexity.
    #[allow(clippy::cast_precision_loss)]
    fn compute_bitrates(&self, resolutions: &[LadderResolution]) -> Vec<u64> {
        let complexity_multiplier = match self.config.complexity {
            ContentComplexity::VeryLow => 0.5,
            ContentComplexity::Low => 0.7,
            ContentComplexity::Medium => 1.0,
            ContentComplexity::High => 1.4,
            ContentComplexity::VeryHigh => 1.8,
        };

        let max_pixels = self.config.max_resolution.pixel_count() as f64;

        resolutions
            .iter()
            .map(|res| {
                let pixel_ratio = res.pixel_count() as f64 / max_pixels;
                // Use power-law scaling (bitrate ~ pixels^0.75)
                let scale = pixel_ratio.powf(0.75);
                let bitrate = self.config.max_bitrate_bps as f64 * scale * complexity_multiplier;
                let bitrate = bitrate
                    .max(self.config.min_bitrate_bps as f64)
                    .min(self.config.max_bitrate_bps as f64);
                bitrate as u64
            })
            .collect()
    }
}

/// A complete quality ladder with all rungs.
#[derive(Debug, Clone)]
pub struct QualityLadder {
    /// Ordered rungs from highest to lowest quality.
    pub rungs: Vec<QualityRung>,
}

impl QualityLadder {
    /// Creates a new empty quality ladder.
    #[must_use]
    pub fn new() -> Self {
        Self { rungs: Vec::new() }
    }

    /// Adds a rung to the ladder.
    pub fn add_rung(&mut self, rung: QualityRung) {
        self.rungs.push(rung);
    }

    /// Returns the number of rungs.
    #[must_use]
    pub fn rung_count(&self) -> usize {
        self.rungs.len()
    }

    /// Returns the highest bitrate rung.
    #[must_use]
    pub fn top_rung(&self) -> Option<&QualityRung> {
        self.rungs.iter().max_by_key(|r| r.bitrate_bps)
    }

    /// Returns the lowest bitrate rung.
    #[must_use]
    pub fn bottom_rung(&self) -> Option<&QualityRung> {
        self.rungs.iter().min_by_key(|r| r.bitrate_bps)
    }

    /// Returns total bandwidth required if all rungs are encoded.
    #[must_use]
    pub fn total_bandwidth_bps(&self) -> u64 {
        self.rungs.iter().map(|r| r.bitrate_bps).sum()
    }

    /// Returns a summary string of the ladder.
    #[must_use]
    pub fn summary(&self) -> String {
        let mut lines = Vec::new();
        for rung in &self.rungs {
            lines.push(format!(
                "{}: {} @ {:.1} Mbps",
                rung.label,
                rung.resolution,
                rung.bitrate_mbps()
            ));
        }
        lines.join("\n")
    }
}

impl Default for QualityLadder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resolution_pixel_count() {
        let res = LadderResolution::new(1920, 1080);
        assert_eq!(res.pixel_count(), 2_073_600);
    }

    #[test]
    fn test_resolution_aspect_ratio() {
        let res = LadderResolution::new(1920, 1080);
        let ar = res.aspect_ratio();
        assert!((ar - 16.0 / 9.0).abs() < 0.01);
    }

    #[test]
    fn test_resolution_aspect_ratio_zero_height() {
        let res = LadderResolution::new(1920, 0);
        assert!((res.aspect_ratio() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_resolution_display() {
        let res = LadderResolution::new(1280, 720);
        assert_eq!(format!("{res}"), "1280x720");
    }

    #[test]
    fn test_quality_rung_new() {
        let res = LadderResolution::new(1920, 1080);
        let rung = QualityRung::new(res, 5_000_000, 30.0);
        assert_eq!(rung.label, "1080p");
        assert_eq!(rung.bitrate_kbps(), 5000);
    }

    #[test]
    fn test_quality_rung_bitrate_mbps() {
        let res = LadderResolution::new(1280, 720);
        let rung = QualityRung::new(res, 3_500_000, 30.0);
        assert!((rung.bitrate_mbps() - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_quality_rung_bits_per_pixel() {
        let res = LadderResolution::new(1920, 1080);
        let rung = QualityRung::new(res, 5_000_000, 30.0);
        let bpp = rung.bits_per_pixel();
        // 5_000_000 / (1920*1080*30) = ~0.0803
        assert!(bpp > 0.05 && bpp < 0.15);
    }

    #[test]
    fn test_quality_rung_builder() {
        let res = LadderResolution::new(1920, 1080);
        let rung = QualityRung::new(res, 5_000_000, 30.0)
            .with_crf(23.0)
            .with_vmaf(92.0)
            .with_label("Full HD");
        assert_eq!(rung.crf, Some(23.0));
        assert_eq!(rung.estimated_vmaf, Some(92.0));
        assert_eq!(rung.label, "Full HD");
    }

    #[test]
    fn test_standard_resolutions() {
        assert_eq!(StandardResolutions::uhd_4k().pixel_count(), 3840 * 2160);
        assert_eq!(StandardResolutions::fhd_1080p().pixel_count(), 1920 * 1080);
        assert_eq!(StandardResolutions::hd_720p().pixel_count(), 1280 * 720);
    }

    #[test]
    fn test_ladder_config_default() {
        let cfg = LadderConfig::default();
        assert_eq!(cfg.max_resolution.width, 1920);
        assert_eq!(cfg.max_rungs, 8);
        assert!((cfg.frame_rate - 30.0).abs() < 1e-6);
    }

    #[test]
    fn test_generator_generate_default() {
        let gen = QualityLadderGenerator::with_defaults();
        let ladder = gen.generate();
        assert!(!ladder.rungs.is_empty());
        assert!(ladder.rung_count() <= 8);
    }

    #[test]
    fn test_ladder_top_and_bottom() {
        let gen = QualityLadderGenerator::with_defaults();
        let ladder = gen.generate();
        let top = ladder.top_rung().expect("top rung should exist");
        let bottom = ladder.bottom_rung().expect("bottom rung should exist");
        assert!(top.bitrate_bps >= bottom.bitrate_bps);
    }

    #[test]
    fn test_ladder_total_bandwidth() {
        let mut ladder = QualityLadder::new();
        let res1 = LadderResolution::new(1920, 1080);
        let res2 = LadderResolution::new(1280, 720);
        ladder.add_rung(QualityRung::new(res1, 5_000_000, 30.0));
        ladder.add_rung(QualityRung::new(res2, 3_000_000, 30.0));
        assert_eq!(ladder.total_bandwidth_bps(), 8_000_000);
    }

    #[test]
    fn test_ladder_summary() {
        let gen = QualityLadderGenerator::with_defaults();
        let ladder = gen.generate();
        let summary = ladder.summary();
        assert!(!summary.is_empty());
        // Should contain "p" labels like "1080p"
        assert!(summary.contains('p'));
    }

    #[test]
    fn test_complexity_affects_bitrates() {
        let low_config = LadderConfig {
            complexity: ContentComplexity::VeryLow,
            ..LadderConfig::default()
        };
        let high_config = LadderConfig {
            complexity: ContentComplexity::VeryHigh,
            ..LadderConfig::default()
        };
        let low_ladder = QualityLadderGenerator::new(low_config).generate();
        let high_ladder = QualityLadderGenerator::new(high_config).generate();

        // For matching resolutions, high complexity should have >= bitrate
        if let (Some(low_top), Some(high_top)) = (low_ladder.top_rung(), high_ladder.top_rung()) {
            assert!(high_top.bitrate_bps >= low_top.bitrate_bps);
        }
    }
}
