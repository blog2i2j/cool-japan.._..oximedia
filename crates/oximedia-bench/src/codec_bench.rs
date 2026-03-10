//! Codec-specific benchmarking for `OxiMedia`.
//!
//! Provides a suite for measuring encode/decode performance and quality
//! metrics for individual codec configurations.

#![allow(dead_code)]

use serde::{Deserialize, Serialize};

/// Configuration parameters for a codec benchmark run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodecBenchConfig {
    /// Human-readable codec name (e.g. "H.264", "HEVC", "AV1").
    pub codec_name: String,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Target frame rate.
    pub fps: f64,
    /// Target bitrate in kilobits per second.
    pub bitrate_kbps: u32,
    /// Group-of-pictures size.
    pub gop_size: u32,
    /// Number of encoding passes.
    pub passes: u32,
}

impl CodecBenchConfig {
    /// Standard H.264 1080p configuration.
    #[must_use]
    pub fn h264_1080p() -> Self {
        Self {
            codec_name: "H.264".to_string(),
            width: 1920,
            height: 1080,
            fps: 30.0,
            bitrate_kbps: 4000,
            gop_size: 60,
            passes: 1,
        }
    }

    /// HEVC 4K configuration.
    #[must_use]
    pub fn hevc_4k() -> Self {
        Self {
            codec_name: "HEVC".to_string(),
            width: 3840,
            height: 2160,
            fps: 30.0,
            bitrate_kbps: 15000,
            gop_size: 120,
            passes: 2,
        }
    }

    /// AV1 1080p configuration.
    #[must_use]
    pub fn av1_1080p() -> Self {
        Self {
            codec_name: "AV1".to_string(),
            width: 1920,
            height: 1080,
            fps: 24.0,
            bitrate_kbps: 3000,
            gop_size: 240,
            passes: 2,
        }
    }

    /// Total pixel count per frame.
    #[must_use]
    pub fn pixels_per_frame(&self) -> u64 {
        u64::from(self.width) * u64::from(self.height)
    }

    /// Bits per pixel at the given bitrate and frame rate.
    #[must_use]
    pub fn theoretical_bpp(&self) -> f64 {
        if self.fps == 0.0 || self.pixels_per_frame() == 0 {
            return 0.0;
        }
        (f64::from(self.bitrate_kbps) * 1000.0) / (self.fps * self.pixels_per_frame() as f64)
    }
}

/// Results produced by encoding/decoding a single codec configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodeBenchResult {
    /// Configuration used for this benchmark.
    pub config: CodecBenchConfig,
    /// Encoding throughput in frames per second.
    pub encode_fps: f64,
    /// Decoding throughput in frames per second.
    pub decode_fps: f64,
    /// Peak signal-to-noise ratio in decibels.
    pub psnr_db: f64,
    /// Structural similarity index (0.0–1.0).
    pub ssim: f64,
    /// Actual bits per pixel in the encoded stream.
    pub bits_per_pixel: f64,
    /// Total elapsed wall-clock time in milliseconds.
    pub elapsed_ms: u64,
}

impl EncodeBenchResult {
    /// Ratio of PSNR (quality) to actual bitrate in kbps.
    ///
    /// Higher is better: more quality per kilobit.
    #[must_use]
    pub fn quality_to_bitrate_ratio(&self) -> f64 {
        if self.config.bitrate_kbps == 0 {
            return 0.0;
        }
        self.psnr_db / f64::from(self.config.bitrate_kbps)
    }

    /// Composite efficiency score combining quality, speed, and bit usage.
    ///
    /// Score is normalised so that higher values indicate a more efficient codec.
    #[must_use]
    pub fn efficiency_score(&self) -> f64 {
        // Weighted combination: quality (60%), speed (30%), low bpp penalty (10%)
        let quality_component = self.psnr_db / 50.0; // PSNR rarely exceeds 50 dB
        let speed_component = (self.encode_fps / 60.0).min(1.0); // normalise to 60 fps
        let bpp_penalty = if self.bits_per_pixel > 0.0 {
            (1.0 / self.bits_per_pixel).min(1.0)
        } else {
            0.0
        };
        0.6 * quality_component + 0.3 * speed_component + 0.1 * bpp_penalty
    }
}

/// A collection of encode benchmark results for comparison.
#[derive(Debug, Clone, Default)]
pub struct CodecBenchSuite {
    /// Individual benchmark results.
    pub results: Vec<EncodeBenchResult>,
}

impl CodecBenchSuite {
    /// Create an empty benchmark suite.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a result to the suite.
    pub fn add(&mut self, r: EncodeBenchResult) {
        self.results.push(r);
    }

    /// Return the result with the highest PSNR (best quality).
    #[must_use]
    pub fn best_quality(&self) -> Option<&EncodeBenchResult> {
        self.results.iter().max_by(|a, b| {
            a.psnr_db
                .partial_cmp(&b.psnr_db)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Return the result with the highest encoding FPS (fastest).
    #[must_use]
    pub fn best_speed(&self) -> Option<&EncodeBenchResult> {
        self.results.iter().max_by(|a, b| {
            a.encode_fps
                .partial_cmp(&b.encode_fps)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Compute the Pareto front: results that are not dominated in both quality and speed.
    ///
    /// A result `a` dominates result `b` when `a.psnr_db >= b.psnr_db` AND
    /// `a.encode_fps >= b.encode_fps` (with at least one strict inequality).
    #[must_use]
    pub fn pareto_front(&self) -> Vec<&EncodeBenchResult> {
        let mut front: Vec<&EncodeBenchResult> = Vec::new();
        for candidate in &self.results {
            let dominated = self.results.iter().any(|other| {
                // `other` dominates `candidate`
                let other_ptr = other as *const EncodeBenchResult;
                let cand_ptr = candidate as *const EncodeBenchResult;
                if other_ptr == cand_ptr {
                    return false;
                }
                other.psnr_db >= candidate.psnr_db
                    && other.encode_fps >= candidate.encode_fps
                    && (other.psnr_db > candidate.psnr_db
                        || other.encode_fps > candidate.encode_fps)
            });
            if !dominated {
                front.push(candidate);
            }
        }
        front
    }

    /// Return a summary string for the suite.
    #[must_use]
    pub fn summary(&self) -> String {
        format!(
            "CodecBenchSuite: {} results, best quality: {:.1} dB, best speed: {:.1} fps",
            self.results.len(),
            self.best_quality().map_or(0.0, |r| r.psnr_db),
            self.best_speed().map_or(0.0, |r| r.encode_fps),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(psnr: f64, fps: f64, bitrate_kbps: u32) -> EncodeBenchResult {
        EncodeBenchResult {
            config: CodecBenchConfig {
                codec_name: "TestCodec".to_string(),
                width: 1920,
                height: 1080,
                fps: 30.0,
                bitrate_kbps,
                gop_size: 60,
                passes: 1,
            },
            encode_fps: fps,
            decode_fps: fps * 4.0,
            psnr_db: psnr,
            ssim: 0.95,
            bits_per_pixel: 0.1,
            elapsed_ms: 1000,
        }
    }

    #[test]
    fn test_h264_1080p_config() {
        let c = CodecBenchConfig::h264_1080p();
        assert_eq!(c.width, 1920);
        assert_eq!(c.height, 1080);
        assert_eq!(c.fps, 30.0);
        assert_eq!(c.bitrate_kbps, 4000);
    }

    #[test]
    fn test_hevc_4k_config() {
        let c = CodecBenchConfig::hevc_4k();
        assert_eq!(c.width, 3840);
        assert_eq!(c.height, 2160);
        assert_eq!(c.passes, 2);
    }

    #[test]
    fn test_av1_1080p_config() {
        let c = CodecBenchConfig::av1_1080p();
        assert_eq!(c.codec_name, "AV1");
        assert_eq!(c.fps, 24.0);
    }

    #[test]
    fn test_pixels_per_frame() {
        let c = CodecBenchConfig::h264_1080p();
        assert_eq!(c.pixels_per_frame(), 1920 * 1080);
    }

    #[test]
    fn test_theoretical_bpp_nonzero() {
        let c = CodecBenchConfig::h264_1080p();
        assert!(c.theoretical_bpp() > 0.0);
    }

    #[test]
    fn test_theoretical_bpp_zero_fps() {
        let mut c = CodecBenchConfig::h264_1080p();
        c.fps = 0.0;
        assert_eq!(c.theoretical_bpp(), 0.0);
    }

    #[test]
    fn test_quality_to_bitrate_ratio() {
        let r = make_result(40.0, 30.0, 4000);
        let ratio = r.quality_to_bitrate_ratio();
        assert!((ratio - 40.0 / 4000.0).abs() < 1e-9);
    }

    #[test]
    fn test_quality_to_bitrate_ratio_zero_bitrate() {
        let r = make_result(40.0, 30.0, 0);
        assert_eq!(r.quality_to_bitrate_ratio(), 0.0);
    }

    #[test]
    fn test_efficiency_score_range() {
        let r = make_result(38.0, 25.0, 4000);
        let score = r.efficiency_score();
        assert!(score >= 0.0 && score <= 1.5, "score={score}");
    }

    #[test]
    fn test_suite_best_quality() {
        let mut suite = CodecBenchSuite::new();
        suite.add(make_result(35.0, 60.0, 2000));
        suite.add(make_result(42.0, 20.0, 4000));
        suite.add(make_result(38.0, 40.0, 3000));
        let best = suite.best_quality().expect("best should be valid");
        assert_eq!(best.psnr_db, 42.0);
    }

    #[test]
    fn test_suite_best_speed() {
        let mut suite = CodecBenchSuite::new();
        suite.add(make_result(35.0, 60.0, 2000));
        suite.add(make_result(42.0, 20.0, 4000));
        let best = suite.best_speed().expect("best should be valid");
        assert_eq!(best.encode_fps, 60.0);
    }

    #[test]
    fn test_suite_empty() {
        let suite = CodecBenchSuite::new();
        assert!(suite.best_quality().is_none());
        assert!(suite.best_speed().is_none());
        assert!(suite.pareto_front().is_empty());
    }

    #[test]
    fn test_pareto_front_no_dominance() {
        let mut suite = CodecBenchSuite::new();
        // result A: better quality, slower speed — not dominated by B
        suite.add(make_result(42.0, 10.0, 4000));
        // result B: lower quality, faster speed — not dominated by A
        suite.add(make_result(30.0, 60.0, 2000));
        let front = suite.pareto_front();
        assert_eq!(front.len(), 2);
    }

    #[test]
    fn test_pareto_front_with_dominance() {
        let mut suite = CodecBenchSuite::new();
        // A dominates B: same speed but better quality
        suite.add(make_result(42.0, 30.0, 4000)); // A
        suite.add(make_result(35.0, 30.0, 4000)); // B — dominated by A
        let front = suite.pareto_front();
        assert_eq!(front.len(), 1);
        assert_eq!(front[0].psnr_db, 42.0);
    }

    #[test]
    fn test_suite_summary_non_empty() {
        let mut suite = CodecBenchSuite::new();
        suite.add(make_result(40.0, 30.0, 4000));
        let s = suite.summary();
        assert!(s.contains("1 results"));
    }
}
