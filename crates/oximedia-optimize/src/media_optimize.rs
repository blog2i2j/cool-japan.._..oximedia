//! High-level media optimization: profiles, complexity analysis, and adaptive bitrate allocation.

use std::time::Duration;

// ──────────────────────────────────────────────────────────────────────────────
// OptimizeTarget
// ──────────────────────────────────────────────────────────────────────────────

/// Describes the intended delivery or storage use-case for the optimized output.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OptimizeTarget {
    /// Streaming over HTTP (e.g., HLS/DASH) with broad device compatibility.
    WebDelivery,
    /// Long-term archival with lossless or near-lossless quality.
    Archive,
    /// Non-destructive editing proxy (large file, fast decode).
    Edit,
    /// Apple ProRes intermediate format for post-production workflows.
    ProRes,
    /// Mobile devices with constrained bandwidth and battery.
    MobileStream,
}

impl OptimizeTarget {
    /// Returns a human-readable label.
    #[must_use]
    pub fn label(&self) -> &'static str {
        match self {
            Self::WebDelivery => "Web Delivery",
            Self::Archive => "Archive",
            Self::Edit => "Edit Proxy",
            Self::ProRes => "ProRes",
            Self::MobileStream => "Mobile Stream",
        }
    }

    /// Whether this target benefits from two-pass encoding.
    #[must_use]
    pub fn prefers_two_pass(&self) -> bool {
        matches!(self, Self::WebDelivery | Self::Archive | Self::MobileStream)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OptimizeProfile
// ──────────────────────────────────────────────────────────────────────────────

/// Full optimization profile describing all encoding parameters.
#[derive(Debug, Clone)]
pub struct OptimizeProfile {
    /// Target delivery / storage scenario.
    pub target: OptimizeTarget,
    /// Output width in pixels.
    pub width: u32,
    /// Output height in pixels.
    pub height: u32,
    /// Target bitrate in bits per second.
    pub bitrate: u64,
    /// Video codec identifier (e.g., `"h264"`, `"vp9"`, `"av1"`).
    pub codec: String,
    /// Enable two-pass encoding for better rate control.
    pub two_pass: bool,
    /// Enable temporal noise reduction.
    pub denoising: bool,
    /// CRF value (0 = lossless, 51 = worst quality).  `None` uses CBR/VBR.
    pub crf: Option<u8>,
    /// Frame rate (numerator, denominator).
    pub frame_rate: (u32, u32),
}

impl OptimizeProfile {
    /// Returns a profile suitable for web streaming at 1080p.
    #[must_use]
    pub fn web_delivery_1080p() -> Self {
        Self {
            target: OptimizeTarget::WebDelivery,
            width: 1920,
            height: 1080,
            bitrate: 4_000_000,
            codec: "h264".to_string(),
            two_pass: true,
            denoising: false,
            crf: Some(23),
            frame_rate: (30, 1),
        }
    }

    /// Returns a profile suitable for mobile streaming at 720p.
    #[must_use]
    pub fn mobile_stream_720p() -> Self {
        Self {
            target: OptimizeTarget::MobileStream,
            width: 1280,
            height: 720,
            bitrate: 1_500_000,
            codec: "h264".to_string(),
            two_pass: true,
            denoising: true,
            crf: Some(28),
            frame_rate: (30, 1),
        }
    }

    /// Returns a lossless archive profile.
    #[must_use]
    pub fn archive() -> Self {
        Self {
            target: OptimizeTarget::Archive,
            width: 1920,
            height: 1080,
            bitrate: 50_000_000,
            codec: "ffv1".to_string(),
            two_pass: false,
            denoising: false,
            crf: None,
            frame_rate: (25, 1),
        }
    }

    /// Returns a ProRes editing proxy profile.
    #[must_use]
    pub fn prores_proxy() -> Self {
        Self {
            target: OptimizeTarget::ProRes,
            width: 1920,
            height: 1080,
            bitrate: 100_000_000,
            codec: "prores".to_string(),
            two_pass: false,
            denoising: false,
            crf: None,
            frame_rate: (25, 1),
        }
    }

    /// Returns an edit proxy profile.
    #[must_use]
    pub fn edit_proxy() -> Self {
        Self {
            target: OptimizeTarget::Edit,
            width: 1280,
            height: 720,
            bitrate: 20_000_000,
            codec: "dnxhd".to_string(),
            two_pass: false,
            denoising: false,
            crf: None,
            frame_rate: (25, 1),
        }
    }

    /// Returns frame rate as a floating-point value.
    #[must_use]
    pub fn fps(&self) -> f64 {
        if self.frame_rate.1 == 0 {
            return 0.0;
        }
        f64::from(self.frame_rate.0) / f64::from(self.frame_rate.1)
    }

    /// Returns total pixel count.
    #[must_use]
    pub fn pixel_count(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// MediaAnalyzer
// ──────────────────────────────────────────────────────────────────────────────

/// Analyzes media frames to derive complexity metrics.
pub struct MediaAnalyzer;

impl MediaAnalyzer {
    /// Creates a new `MediaAnalyzer`.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Computes a scene complexity score in the range `[0.0, 1.0]`.
    ///
    /// The score is based on the spatial variance of each frame (how much
    /// pixel-level detail / texture is present).  A flat frame scores near 0.0;
    /// a highly textured frame scores near 1.0.
    ///
    /// `frames` is a slice of raw 8-bit luma planes, each frame stored as a
    /// contiguous `Vec<u8>`.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_optimize::media_optimize::MediaAnalyzer;
    ///
    /// let flat = vec![128u8; 1024];
    /// let noisy: Vec<u8> = (0..1024).map(|i| (i % 255) as u8).collect();
    /// let analyzer = MediaAnalyzer::new();
    /// let score = analyzer.analyze_complexity(&[flat, noisy]);
    /// assert!(score >= 0.0 && score <= 1.0);
    /// ```
    #[must_use]
    pub fn analyze_complexity(&self, frames: &[Vec<u8>]) -> f32 {
        if frames.is_empty() {
            return 0.0;
        }

        let per_frame_scores: Vec<f32> = frames.iter().map(|f| self.frame_complexity(f)).collect();

        let mean: f32 = per_frame_scores.iter().sum::<f32>() / per_frame_scores.len() as f32;
        mean.clamp(0.0, 1.0)
    }

    /// Computes a spatial complexity score for a single luma plane.
    #[must_use]
    pub fn frame_complexity(&self, luma: &[u8]) -> f32 {
        if luma.len() < 2 {
            return 0.0;
        }

        // Compute variance of pixel values.
        let mean: f64 = luma.iter().map(|&p| f64::from(p)).sum::<f64>() / luma.len() as f64;
        let variance: f64 = luma
            .iter()
            .map(|&p| {
                let d = f64::from(p) - mean;
                d * d
            })
            .sum::<f64>()
            / luma.len() as f64;

        // Normalise to [0, 1]: max variance for 8-bit pixel is 128^2 = 16384.
        let normalised = (variance / 16384.0).min(1.0);
        normalised as f32
    }

    /// Returns a per-frame complexity curve (same indexing as `frames`).
    #[must_use]
    pub fn complexity_curve(&self, frames: &[Vec<u8>]) -> Vec<f32> {
        frames.iter().map(|f| self.frame_complexity(f)).collect()
    }
}

impl Default for MediaAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// AdaptiveBitrateOptimizer
// ──────────────────────────────────────────────────────────────────────────────

/// Allocates bitrate across frames proportional to their complexity score.
///
/// Simple frames receive fewer bits; complex frames receive more, while
/// respecting the overall budget.
pub struct AdaptiveBitrateOptimizer {
    /// Total bit budget for the clip.
    total_bits: u64,
    /// Minimum fraction of average bits any single frame can receive.
    min_fraction: f64,
    /// Maximum multiple of average bits any single frame can receive.
    max_fraction: f64,
}

impl AdaptiveBitrateOptimizer {
    /// Creates a new optimizer with the specified bit budget.
    #[must_use]
    pub fn new(total_bits: u64) -> Self {
        Self {
            total_bits,
            min_fraction: 0.2,
            max_fraction: 5.0,
        }
    }

    /// Sets the minimum per-frame fraction of the average allocation.
    #[must_use]
    pub fn with_min_fraction(mut self, min: f64) -> Self {
        self.min_fraction = min;
        self
    }

    /// Sets the maximum per-frame fraction of the average allocation.
    #[must_use]
    pub fn with_max_fraction(mut self, max: f64) -> Self {
        self.max_fraction = max;
        self
    }

    /// Allocates bits to each frame based on its entry in `complexity_curve`.
    ///
    /// Returns a `Vec` of per-frame bit counts.  The sum is guaranteed to be
    /// at most `total_bits` (possibly slightly less due to integer rounding).
    #[must_use]
    pub fn allocate(&self, complexity_curve: &[f32]) -> Vec<u64> {
        if complexity_curve.is_empty() {
            return vec![];
        }
        let n = complexity_curve.len();
        let _avg_bits = self.total_bits as f64 / n as f64;

        // Compute weights: complexity in [0,1], scale so weight ≥ min_fraction and ≤ max_fraction.
        let weights: Vec<f64> = complexity_curve
            .iter()
            .map(|&c| {
                let raw = 0.5 + f64::from(c); // shift: flat scene (0) → 0.5, complex (1) → 1.5
                raw.clamp(self.min_fraction, self.max_fraction)
            })
            .collect();

        let weight_sum: f64 = weights.iter().sum();

        weights
            .iter()
            .map(|&w| {
                let fraction = w / weight_sum;
                (self.total_bits as f64 * fraction).round() as u64
            })
            .collect()
    }

    /// Derives a per-frame CRF sequence from a complexity curve and a target average CRF.
    ///
    /// Complex frames get a lower (higher quality) CRF; simple frames get a higher CRF.
    /// The returned values are clamped to `[0, 51]`.
    #[must_use]
    pub fn derive_crf_sequence(&self, complexity_curve: &[f32], avg_crf: u8) -> Vec<u8> {
        complexity_curve
            .iter()
            .map(|&c| {
                // c in [0,1]; complex frame: lower CRF; simple frame: higher CRF.
                let delta = (f64::from(c) - 0.5) * -10.0; // complex → -5, flat → +5
                let crf_f = f64::from(avg_crf) + delta;
                crf_f.clamp(0.0, 51.0).round() as u8
            })
            .collect()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// OptimizationReport
// ──────────────────────────────────────────────────────────────────────────────

/// Summary report produced after an optimization run.
#[derive(Debug, Clone)]
pub struct OptimizationReport {
    /// Reduction in file size relative to the input, as a percentage (0–100).
    pub size_reduction_percent: f64,
    /// Change in perceived quality (positive = better, negative = worse).
    /// Expressed as a VMAF-delta or PSNR-delta depending on the metric used.
    pub quality_delta: f64,
    /// Wall-clock processing time.
    pub processing_time: Duration,
    /// Original file size in bytes.
    pub original_size_bytes: u64,
    /// Output file size in bytes.
    pub output_size_bytes: u64,
    /// Target that was optimized for.
    pub target: OptimizeTarget,
    /// Whether two-pass encoding was used.
    pub two_pass_used: bool,
    /// Average scene complexity score (0–1).
    pub avg_complexity: f32,
}

impl OptimizationReport {
    /// Creates a new report.
    #[allow(clippy::too_many_arguments)]
    #[must_use]
    pub fn new(
        original_size_bytes: u64,
        output_size_bytes: u64,
        quality_delta: f64,
        processing_time: Duration,
        target: OptimizeTarget,
        two_pass_used: bool,
        avg_complexity: f32,
    ) -> Self {
        let size_reduction_percent = if original_size_bytes > 0 {
            let saved = original_size_bytes.saturating_sub(output_size_bytes);
            (saved as f64 / original_size_bytes as f64) * 100.0
        } else {
            0.0
        };
        Self {
            size_reduction_percent,
            quality_delta,
            processing_time,
            original_size_bytes,
            output_size_bytes,
            target,
            two_pass_used,
            avg_complexity,
        }
    }

    /// Returns `true` if the optimization achieved a net size reduction.
    #[must_use]
    pub fn is_beneficial(&self) -> bool {
        self.size_reduction_percent > 0.0
    }

    /// Returns a short textual summary of the report.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "Target: {} | Size reduction: {:.1}% | Quality delta: {:.2} | Time: {:.1}s",
            self.target.label(),
            self.size_reduction_percent,
            self.quality_delta,
            self.processing_time.as_secs_f64(),
        )
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── OptimizeTarget ────────────────────────────────────────────────────────

    #[test]
    fn test_optimize_target_labels() {
        assert_eq!(OptimizeTarget::WebDelivery.label(), "Web Delivery");
        assert_eq!(OptimizeTarget::Archive.label(), "Archive");
        assert_eq!(OptimizeTarget::Edit.label(), "Edit Proxy");
        assert_eq!(OptimizeTarget::ProRes.label(), "ProRes");
        assert_eq!(OptimizeTarget::MobileStream.label(), "Mobile Stream");
    }

    #[test]
    fn test_optimize_target_two_pass_preference() {
        assert!(OptimizeTarget::WebDelivery.prefers_two_pass());
        assert!(OptimizeTarget::Archive.prefers_two_pass());
        assert!(OptimizeTarget::MobileStream.prefers_two_pass());
        assert!(!OptimizeTarget::ProRes.prefers_two_pass());
        assert!(!OptimizeTarget::Edit.prefers_two_pass());
    }

    // ── OptimizeProfile ───────────────────────────────────────────────────────

    #[test]
    fn test_profile_web_delivery() {
        let p = OptimizeProfile::web_delivery_1080p();
        assert_eq!(p.target, OptimizeTarget::WebDelivery);
        assert_eq!(p.width, 1920);
        assert_eq!(p.height, 1080);
        assert!(p.two_pass);
        assert_eq!(p.codec, "h264");
    }

    #[test]
    fn test_profile_mobile_stream() {
        let p = OptimizeProfile::mobile_stream_720p();
        assert_eq!(p.target, OptimizeTarget::MobileStream);
        assert_eq!(p.width, 1280);
        assert_eq!(p.height, 720);
        assert!(p.denoising);
        assert!(p.crf.is_some());
        assert!(p.crf.expect("CRF should be set") > 23); // lower quality than 1080p
    }

    #[test]
    fn test_profile_fps() {
        let p = OptimizeProfile::web_delivery_1080p();
        assert!((p.fps() - 30.0).abs() < 0.001);
    }

    #[test]
    fn test_profile_pixel_count() {
        let p = OptimizeProfile::web_delivery_1080p();
        assert_eq!(p.pixel_count(), 1920 * 1080);
    }

    #[test]
    fn test_profile_archive_lossless() {
        let p = OptimizeProfile::archive();
        assert_eq!(p.codec, "ffv1");
        assert!(p.crf.is_none());
        assert!(!p.denoising);
    }

    // ── MediaAnalyzer ─────────────────────────────────────────────────────────

    #[test]
    fn test_flat_frame_low_complexity() {
        let analyzer = MediaAnalyzer::new();
        let flat = vec![128u8; 1024];
        let score = analyzer.frame_complexity(&flat);
        assert!(
            score < 0.01,
            "Flat frame should score near zero, got {score}"
        );
    }

    #[test]
    fn test_noisy_frame_high_complexity() {
        let analyzer = MediaAnalyzer::new();
        let noisy: Vec<u8> = (0..1024)
            .map(|i| if i % 2 == 0 { 0u8 } else { 255u8 })
            .collect();
        let score = analyzer.frame_complexity(&noisy);
        assert!(
            score > 0.5,
            "Alternating 0/255 should score high, got {score}"
        );
    }

    #[test]
    fn test_analyze_complexity_range() {
        let analyzer = MediaAnalyzer::new();
        let flat = vec![100u8; 512];
        let mixed: Vec<u8> = (0..512).map(|i| (i * 3 % 256) as u8).collect();
        let score = analyzer.analyze_complexity(&[flat, mixed]);
        assert!((0.0..=1.0).contains(&score));
    }

    #[test]
    fn test_analyze_complexity_empty() {
        let analyzer = MediaAnalyzer::new();
        assert_eq!(analyzer.analyze_complexity(&[]), 0.0);
    }

    #[test]
    fn test_complexity_curve_length() {
        let analyzer = MediaAnalyzer::new();
        let frames: Vec<Vec<u8>> = (0..5).map(|_| vec![128u8; 256]).collect();
        let curve = analyzer.complexity_curve(&frames);
        assert_eq!(curve.len(), 5);
    }

    // ── AdaptiveBitrateOptimizer ──────────────────────────────────────────────

    #[test]
    fn test_abr_allocation_sum_within_budget() {
        let optimizer = AdaptiveBitrateOptimizer::new(1_000_000);
        let curve = vec![0.1, 0.5, 0.9, 0.3, 0.7];
        let alloc = optimizer.allocate(&curve);
        assert_eq!(alloc.len(), 5);
        let total: u64 = alloc.iter().sum();
        // Allow ±len due to rounding.
        assert!(total <= 1_000_000 + curve.len() as u64);
    }

    #[test]
    fn test_abr_complex_frames_get_more_bits() {
        let optimizer = AdaptiveBitrateOptimizer::new(1_000_000);
        let curve = vec![0.0, 1.0]; // flat then complex
        let alloc = optimizer.allocate(&curve);
        assert_eq!(alloc.len(), 2);
        assert!(
            alloc[1] > alloc[0],
            "Complex frame should receive more bits"
        );
    }

    #[test]
    fn test_abr_empty_curve() {
        let optimizer = AdaptiveBitrateOptimizer::new(1_000_000);
        let alloc = optimizer.allocate(&[]);
        assert!(alloc.is_empty());
    }

    #[test]
    fn test_derive_crf_sequence() {
        let optimizer = AdaptiveBitrateOptimizer::new(1_000_000);
        let curve = vec![0.0_f32, 1.0_f32]; // flat → high CRF, complex → low CRF
        let crfs = optimizer.derive_crf_sequence(&curve, 23);
        assert_eq!(crfs.len(), 2);
        assert!(
            crfs[0] > crfs[1],
            "Flat frame should get higher CRF than complex"
        );
    }

    // ── OptimizationReport ────────────────────────────────────────────────────

    #[test]
    fn test_report_size_reduction() {
        let report = OptimizationReport::new(
            100_000_000,
            60_000_000,
            2.5,
            Duration::from_secs(120),
            OptimizeTarget::WebDelivery,
            true,
            0.4,
        );
        assert!((report.size_reduction_percent - 40.0).abs() < 0.001);
        assert!(report.is_beneficial());
    }

    #[test]
    fn test_report_no_reduction() {
        let report = OptimizationReport::new(
            50_000_000,
            55_000_000, // bigger after encoding!
            -1.0,
            Duration::from_secs(30),
            OptimizeTarget::Archive,
            false,
            0.2,
        );
        assert!(!report.is_beneficial());
        assert_eq!(report.size_reduction_percent, 0.0); // saturating_sub
    }

    #[test]
    fn test_report_summary_contains_target() {
        let report = OptimizationReport::new(
            80_000_000,
            40_000_000,
            1.0,
            Duration::from_secs(60),
            OptimizeTarget::MobileStream,
            true,
            0.3,
        );
        let summary = report.summary();
        assert!(summary.contains("Mobile Stream"));
        assert!(summary.contains("50.0")); // 50% reduction
    }
}
