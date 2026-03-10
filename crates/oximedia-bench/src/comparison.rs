//! Cross-codec comparison tools for analyzing benchmark results.

use crate::CodecBenchmarkResult;
use oximedia_core::types::CodecId;
use serde::{Deserialize, Serialize};

/// Result of comparing two codecs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComparisonResult {
    /// Speed ratio (codec_a / codec_b)
    pub encoding_speed_ratio: f64,
    /// Decoding speed ratio
    pub decoding_speed_ratio: f64,
    /// Quality difference (PSNR)
    pub psnr_difference: Option<f64>,
    /// SSIM difference
    pub ssim_difference: Option<f64>,
    /// File size ratio
    pub file_size_ratio: f64,
    /// Efficiency score (quality per bit)
    pub efficiency_score: Option<f64>,
}

/// Codec comparison utilities.
pub struct CodecComparison;

impl CodecComparison {
    /// Compare two codec results.
    #[must_use]
    pub fn compare(
        codec_a: Vec<&CodecBenchmarkResult>,
        codec_b: Vec<&CodecBenchmarkResult>,
    ) -> ComparisonResult {
        let avg_a = Self::average_results(&codec_a);
        let avg_b = Self::average_results(&codec_b);

        let encoding_speed_ratio = avg_a.encoding_fps / avg_b.encoding_fps;
        let decoding_speed_ratio = avg_a.decoding_fps / avg_b.decoding_fps;
        let file_size_ratio = avg_a.file_size as f64 / avg_b.file_size as f64;

        let psnr_difference = match (avg_a.psnr, avg_b.psnr) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };

        let ssim_difference = match (avg_a.ssim, avg_b.ssim) {
            (Some(a), Some(b)) => Some(a - b),
            _ => None,
        };

        let efficiency_score = if let (Some(psnr_a), Some(psnr_b)) = (avg_a.psnr, avg_b.psnr) {
            let eff_a = psnr_a / (avg_a.file_size as f64 / 1_000_000.0);
            let eff_b = psnr_b / (avg_b.file_size as f64 / 1_000_000.0);
            Some(eff_a / eff_b)
        } else {
            None
        };

        ComparisonResult {
            encoding_speed_ratio,
            decoding_speed_ratio,
            psnr_difference,
            ssim_difference,
            file_size_ratio,
            efficiency_score,
        }
    }

    fn average_results(results: &[&CodecBenchmarkResult]) -> AverageMetrics {
        let mut total_encoding_fps = 0.0;
        let mut total_decoding_fps = 0.0;
        let mut total_file_size = 0u64;
        let mut total_psnr = 0.0;
        let mut total_ssim = 0.0;
        let mut psnr_count = 0;
        let mut ssim_count = 0;
        let mut count = 0;

        for result in results {
            for seq in &result.sequence_results {
                total_encoding_fps += seq.encoding_fps;
                total_decoding_fps += seq.decoding_fps;
                total_file_size += seq.file_size_bytes;

                if let Some(psnr) = seq.metrics.psnr {
                    total_psnr += psnr;
                    psnr_count += 1;
                }

                if let Some(ssim) = seq.metrics.ssim {
                    total_ssim += ssim;
                    ssim_count += 1;
                }

                count += 1;
            }
        }

        if count == 0 {
            return AverageMetrics::default();
        }

        AverageMetrics {
            encoding_fps: total_encoding_fps / count as f64,
            decoding_fps: total_decoding_fps / count as f64,
            file_size: total_file_size / count as u64,
            psnr: if psnr_count > 0 {
                Some(total_psnr / psnr_count as f64)
            } else {
                None
            },
            ssim: if ssim_count > 0 {
                Some(total_ssim / ssim_count as f64)
            } else {
                None
            },
        }
    }
}

#[derive(Debug, Default)]
struct AverageMetrics {
    encoding_fps: f64,
    decoding_fps: f64,
    file_size: u64,
    psnr: Option<f64>,
    ssim: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CodecBenchmarkResult, QualityMetrics, SequenceResult, Statistics};
    use oximedia_core::types::CodecId;
    use std::time::Duration;

    fn make_sequence_result(
        name: &str,
        encoding_fps: f64,
        decoding_fps: f64,
        file_size: u64,
        psnr: Option<f64>,
    ) -> SequenceResult {
        SequenceResult {
            sequence_name: name.to_string(),
            frames_processed: 100,
            encoding_fps,
            decoding_fps,
            file_size_bytes: file_size,
            metrics: QualityMetrics {
                psnr,
                ..Default::default()
            },
            encoding_duration: Duration::from_secs(1),
            decoding_duration: Duration::from_secs(1),
        }
    }

    #[test]
    fn test_comparison_result() {
        let result_a = CodecBenchmarkResult {
            codec_id: CodecId::Av1,
            preset: None,
            bitrate_kbps: None,
            cq_level: None,
            sequence_results: vec![make_sequence_result(
                "seq1",
                30.0,
                60.0,
                1_000_000,
                Some(40.0),
            )],
            statistics: Statistics::default(),
        };

        let result_b = CodecBenchmarkResult {
            codec_id: CodecId::Vp9,
            preset: None,
            bitrate_kbps: None,
            cq_level: None,
            sequence_results: vec![make_sequence_result(
                "seq1",
                60.0,
                120.0,
                1_200_000,
                Some(38.0),
            )],
            statistics: Statistics::default(),
        };

        let comparison = CodecComparison::compare(vec![&result_a], vec![&result_b]);

        assert_eq!(comparison.encoding_speed_ratio, 0.5);
        assert_eq!(comparison.decoding_speed_ratio, 0.5);
        assert_eq!(comparison.psnr_difference, Some(2.0));
    }
}

/// Detailed codec comparison with statistical analysis.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedComparison {
    /// Basic comparison result
    pub basic: ComparisonResult,
    /// Speed comparison details
    pub speed: SpeedComparison,
    /// Quality comparison details
    pub quality: QualityComparison,
    /// Efficiency comparison details
    pub efficiency: EfficiencyComparison,
}

/// Speed comparison details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedComparison {
    /// Encoding speed statistics
    pub encoding: SpeedStats,
    /// Decoding speed statistics
    pub decoding: SpeedStats,
}

/// Speed statistics for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeedStats {
    /// Mean FPS for codec A
    pub mean_a: f64,
    /// Mean FPS for codec B
    pub mean_b: f64,
    /// Median FPS for codec A
    pub median_a: f64,
    /// Median FPS for codec B
    pub median_b: f64,
    /// Standard deviation for codec A
    pub std_dev_a: f64,
    /// Standard deviation for codec B
    pub std_dev_b: f64,
    /// Speed ratio (A/B)
    pub ratio: f64,
    /// Statistical significance p-value
    pub p_value: Option<f64>,
}

/// Quality comparison details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityComparison {
    /// PSNR comparison
    pub psnr: Option<QualityStats>,
    /// SSIM comparison
    pub ssim: Option<QualityStats>,
    /// VMAF comparison
    pub vmaf: Option<QualityStats>,
}

/// Quality statistics for comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStats {
    /// Mean quality for codec A
    pub mean_a: f64,
    /// Mean quality for codec B
    pub mean_b: f64,
    /// Quality difference
    pub difference: f64,
    /// Relative improvement percentage
    pub relative_improvement: f64,
}

/// Efficiency comparison details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyComparison {
    /// Bits per pixel
    pub bpp: BppComparison,
    /// Quality per bit
    pub quality_per_bit: Option<f64>,
    /// Compression ratio
    pub compression_ratio: CompressionRatio,
}

/// Bits per pixel comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BppComparison {
    /// BPP for codec A
    pub bpp_a: f64,
    /// BPP for codec B
    pub bpp_b: f64,
    /// BPP ratio
    pub ratio: f64,
}

/// Compression ratio comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionRatio {
    /// Compression ratio for codec A
    pub ratio_a: f64,
    /// Compression ratio for codec B
    pub ratio_b: f64,
    /// Relative efficiency
    pub relative_efficiency: f64,
}

/// Preset comparison for analyzing different encoding presets.
#[derive(Debug, Clone)]
pub struct PresetComparison {
    codec_id: CodecId,
    presets: Vec<String>,
}

impl PresetComparison {
    /// Create a new preset comparison.
    #[must_use]
    pub fn new(codec_id: CodecId) -> Self {
        Self {
            codec_id,
            presets: Vec::new(),
        }
    }

    /// Add a preset to compare.
    pub fn add_preset(&mut self, preset: impl Into<String>) {
        self.presets.push(preset.into());
    }

    /// Compare presets.
    #[must_use]
    pub fn compare(&self, _results: &[&CodecBenchmarkResult]) -> PresetComparisonResult {
        // Placeholder for preset comparison
        PresetComparisonResult {
            codec_id: self.codec_id,
            fastest_preset: String::new(),
            highest_quality_preset: String::new(),
            best_balanced_preset: String::new(),
        }
    }
}

/// Result of preset comparison.
#[derive(Debug, Clone)]
pub struct PresetComparisonResult {
    /// Codec being compared
    pub codec_id: CodecId,
    /// Fastest encoding preset
    pub fastest_preset: String,
    /// Highest quality preset
    pub highest_quality_preset: String,
    /// Best balanced preset (speed vs quality)
    pub best_balanced_preset: String,
}

/// Rate-distortion curve comparison.
#[derive(Debug, Clone)]
pub struct RdCurveComparison {
    points_a: Vec<RdPoint>,
    points_b: Vec<RdPoint>,
}

/// Rate-distortion point.
#[derive(Debug, Clone, Copy)]
pub struct RdPoint {
    /// Bitrate in kbps
    pub bitrate: f64,
    /// Quality metric (PSNR or VMAF)
    pub quality: f64,
}

impl RdCurveComparison {
    /// Create a new RD curve comparison.
    #[must_use]
    pub fn new() -> Self {
        Self {
            points_a: Vec::new(),
            points_b: Vec::new(),
        }
    }

    /// Add a point for codec A.
    pub fn add_point_a(&mut self, bitrate: f64, quality: f64) {
        self.points_a.push(RdPoint { bitrate, quality });
    }

    /// Add a point for codec B.
    pub fn add_point_b(&mut self, bitrate: f64, quality: f64) {
        self.points_b.push(RdPoint { bitrate, quality });
    }

    /// Calculate BD-Rate (Bjøntegaard Delta Rate).
    #[must_use]
    pub fn calculate_bd_rate(&self) -> Option<f64> {
        if self.points_a.len() < 2 || self.points_b.len() < 2 {
            return None;
        }

        // Simplified BD-Rate calculation (placeholder)
        // Real implementation would use polynomial fitting
        Some(0.0)
    }

    /// Calculate BD-PSNR (Bjøntegaard Delta PSNR).
    #[must_use]
    pub fn calculate_bd_psnr(&self) -> Option<f64> {
        if self.points_a.len() < 2 || self.points_b.len() < 2 {
            return None;
        }

        // Simplified BD-PSNR calculation (placeholder)
        Some(0.0)
    }
}

impl Default for RdCurveComparison {
    fn default() -> Self {
        Self::new()
    }
}

/// Codec ranking based on multiple criteria.
#[derive(Debug, Clone)]
pub struct CodecRanking {
    rankings: Vec<RankingEntry>,
}

/// Ranking entry for a codec.
#[derive(Debug, Clone)]
pub struct RankingEntry {
    /// Codec identifier
    pub codec_id: CodecId,
    /// Overall score (0-100)
    pub overall_score: f64,
    /// Speed score (0-100)
    pub speed_score: f64,
    /// Quality score (0-100)
    pub quality_score: f64,
    /// Efficiency score (0-100)
    pub efficiency_score: f64,
}

impl CodecRanking {
    /// Create a new codec ranking.
    #[must_use]
    pub fn new() -> Self {
        Self {
            rankings: Vec::new(),
        }
    }

    /// Add a codec to the ranking.
    pub fn add(&mut self, entry: RankingEntry) {
        self.rankings.push(entry);
    }

    /// Sort rankings by overall score.
    pub fn sort_by_overall(&mut self) {
        self.rankings.sort_by(|a, b| {
            b.overall_score
                .partial_cmp(&a.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
    }

    /// Get the top codec.
    #[must_use]
    pub fn top_codec(&self) -> Option<&RankingEntry> {
        self.rankings.first()
    }

    /// Get rankings.
    #[must_use]
    pub fn rankings(&self) -> &[RankingEntry] {
        &self.rankings
    }
}

impl Default for CodecRanking {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod extended_tests {
    use super::*;

    #[test]
    fn test_speed_stats() {
        let stats = SpeedStats {
            mean_a: 30.0,
            mean_b: 60.0,
            median_a: 29.0,
            median_b: 59.0,
            std_dev_a: 2.0,
            std_dev_b: 3.0,
            ratio: 0.5,
            p_value: Some(0.01),
        };

        assert_eq!(stats.ratio, 0.5);
    }

    #[test]
    fn test_quality_stats() {
        let stats = QualityStats {
            mean_a: 38.0,
            mean_b: 40.0,
            difference: 2.0,
            relative_improvement: 5.26,
        };

        assert_eq!(stats.difference, 2.0);
    }

    #[test]
    fn test_rd_curve() {
        let mut curve = RdCurveComparison::new();
        curve.add_point_a(1000.0, 35.0);
        curve.add_point_a(2000.0, 38.0);
        curve.add_point_b(1000.0, 34.0);
        curve.add_point_b(2000.0, 37.5);

        assert_eq!(curve.points_a.len(), 2);
        assert_eq!(curve.points_b.len(), 2);
    }

    #[test]
    fn test_codec_ranking() {
        let mut ranking = CodecRanking::new();

        ranking.add(RankingEntry {
            codec_id: CodecId::Av1,
            overall_score: 85.0,
            speed_score: 70.0,
            quality_score: 95.0,
            efficiency_score: 90.0,
        });

        ranking.add(RankingEntry {
            codec_id: CodecId::Vp9,
            overall_score: 80.0,
            speed_score: 75.0,
            quality_score: 90.0,
            efficiency_score: 75.0,
        });

        ranking.sort_by_overall();

        let top = ranking.top_codec().expect("top should be valid");
        assert_eq!(top.codec_id, CodecId::Av1);
        assert_eq!(top.overall_score, 85.0);
    }

    #[test]
    fn test_preset_comparison() {
        let mut comp = PresetComparison::new(CodecId::Av1);
        comp.add_preset("fast");
        comp.add_preset("medium");
        comp.add_preset("slow");

        assert_eq!(comp.presets.len(), 3);
    }

    #[test]
    fn test_bpp_comparison() {
        let bpp = BppComparison {
            bpp_a: 0.5,
            bpp_b: 0.6,
            ratio: 0.833,
        };

        assert_eq!(bpp.bpp_a, 0.5);
        assert_eq!(bpp.bpp_b, 0.6);
    }

    #[test]
    fn test_compression_ratio() {
        let ratio = CompressionRatio {
            ratio_a: 50.0,
            ratio_b: 40.0,
            relative_efficiency: 1.25,
        };

        assert_eq!(ratio.ratio_a, 50.0);
        assert_eq!(ratio.relative_efficiency, 1.25);
    }
}
