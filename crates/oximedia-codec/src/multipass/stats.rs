//! Statistics collection and storage for multi-pass encoding.
//!
//! This module handles the collection, serialization, and storage of encoding
//! statistics during the first pass, which are then used in the second pass
//! for optimal bitrate allocation.

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use crate::frame::FrameType;
use crate::multipass::complexity::FrameComplexity;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::Path;

/// Statistics for a single encoded frame.
#[derive(Clone, Debug)]
pub struct FrameStatistics {
    /// Frame index in the stream.
    pub frame_index: u64,
    /// Frame type.
    pub frame_type: FrameType,
    /// Quantization parameter used.
    pub qp: f64,
    /// Actual bits used to encode this frame.
    pub bits: u64,
    /// Frame complexity metrics.
    pub complexity: FrameComplexity,
    /// Motion estimation data (average motion vector magnitude).
    pub avg_motion: f64,
    /// Peak Signal-to-Noise Ratio (if available).
    pub psnr: Option<f64>,
    /// Structural Similarity Index (if available).
    pub ssim: Option<f64>,
}

impl FrameStatistics {
    /// Create new frame statistics.
    #[must_use]
    pub fn new(
        frame_index: u64,
        frame_type: FrameType,
        qp: f64,
        bits: u64,
        complexity: FrameComplexity,
    ) -> Self {
        Self {
            frame_index,
            frame_type,
            qp,
            bits,
            complexity,
            avg_motion: 0.0,
            psnr: None,
            ssim: None,
        }
    }

    /// Set motion estimation data.
    pub fn set_motion(&mut self, avg_motion: f64) {
        self.avg_motion = avg_motion;
    }

    /// Set quality metrics.
    pub fn set_quality_metrics(&mut self, psnr: f64, ssim: f64) {
        self.psnr = Some(psnr);
        self.ssim = Some(ssim);
    }

    /// Get bits per pixel.
    #[must_use]
    pub fn bits_per_pixel(&self, width: u32, height: u32) -> f64 {
        let pixels = (width as u64) * (height as u64);
        if pixels == 0 {
            return 0.0;
        }
        self.bits as f64 / pixels as f64
    }
}

/// First-pass statistics collection.
#[derive(Clone, Debug)]
pub struct PassStatistics {
    /// All frame statistics.
    pub frames: Vec<FrameStatistics>,
    /// Total frames encoded.
    pub total_frames: u64,
    /// Total bits used.
    pub total_bits: u64,
    /// Average QP across all frames.
    pub avg_qp: f64,
    /// Average frame size in bits.
    pub avg_frame_bits: f64,
    /// Video width.
    pub width: u32,
    /// Video height.
    pub height: u32,
    /// Frame rate numerator.
    pub framerate_num: u32,
    /// Frame rate denominator.
    pub framerate_den: u32,
}

impl PassStatistics {
    /// Create a new pass statistics collector.
    #[must_use]
    pub fn new(width: u32, height: u32, framerate_num: u32, framerate_den: u32) -> Self {
        Self {
            frames: Vec::new(),
            total_frames: 0,
            total_bits: 0,
            avg_qp: 0.0,
            avg_frame_bits: 0.0,
            width,
            height,
            framerate_num,
            framerate_den,
        }
    }

    /// Add a frame's statistics.
    pub fn add_frame(&mut self, stats: FrameStatistics) {
        self.total_bits += stats.bits;
        self.total_frames += 1;
        self.frames.push(stats);
        self.update_averages();
    }

    /// Update average statistics.
    fn update_averages(&mut self) {
        if self.total_frames == 0 {
            return;
        }

        self.avg_frame_bits = self.total_bits as f64 / self.total_frames as f64;

        let total_qp: f64 = self.frames.iter().map(|f| f.qp).sum();
        self.avg_qp = total_qp / self.total_frames as f64;
    }

    /// Get statistics for a specific frame.
    #[must_use]
    pub fn get_frame(&self, index: u64) -> Option<&FrameStatistics> {
        self.frames.iter().find(|f| f.frame_index == index)
    }

    /// Get average bitrate in bits per second.
    #[must_use]
    pub fn average_bitrate(&self) -> u64 {
        if self.total_frames == 0 {
            return 0;
        }

        let fps = self.framerate_num as f64 / self.framerate_den as f64;
        (self.avg_frame_bits * fps) as u64
    }

    /// Get peak bitrate (highest frame size * framerate).
    #[must_use]
    pub fn peak_bitrate(&self) -> u64 {
        let max_frame_bits = self.frames.iter().map(|f| f.bits).max().unwrap_or(0);

        let fps = self.framerate_num as f64 / self.framerate_den as f64;
        (max_frame_bits as f64 * fps) as u64
    }

    /// Calculate complexity distribution across frames.
    #[must_use]
    pub fn complexity_distribution(&self) -> ComplexityStats {
        if self.frames.is_empty() {
            return ComplexityStats::default();
        }

        let mut complexities: Vec<f64> = self
            .frames
            .iter()
            .map(|f| f.complexity.combined_complexity)
            .collect();

        complexities.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let sum: f64 = complexities.iter().sum();
        let mean = sum / complexities.len() as f64;

        let variance: f64 = complexities.iter().map(|c| (c - mean).powi(2)).sum::<f64>()
            / complexities.len() as f64;
        let std_dev = variance.sqrt();

        let median = if complexities.len() % 2 == 0 {
            let mid = complexities.len() / 2;
            (complexities[mid - 1] + complexities[mid]) / 2.0
        } else {
            complexities[complexities.len() / 2]
        };

        ComplexityStats {
            mean,
            std_dev,
            median,
            min: complexities.first().copied().unwrap_or(0.0),
            max: complexities.last().copied().unwrap_or(0.0),
        }
    }

    /// Save statistics to a file.
    pub fn save_to_file<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write header
        writeln!(writer, "# OxiMedia First Pass Statistics v1.0")?;
        writeln!(writer, "width={}", self.width)?;
        writeln!(writer, "height={}", self.height)?;
        writeln!(
            writer,
            "framerate={}/{}",
            self.framerate_num, self.framerate_den
        )?;
        writeln!(writer, "total_frames={}", self.total_frames)?;
        writeln!(writer, "total_bits={}", self.total_bits)?;
        writeln!(writer, "avg_qp={:.2}", self.avg_qp)?;
        writeln!(writer, "avg_frame_bits={:.2}", self.avg_frame_bits)?;
        writeln!(writer)?;

        // Write per-frame data
        writeln!(
            writer,
            "# frame_idx,frame_type,qp,bits,spatial,temporal,combined,sad,variance,difficulty,scene_change,avg_motion,psnr,ssim"
        )?;

        for stats in &self.frames {
            let frame_type_str = match stats.frame_type {
                FrameType::Key => "I",
                FrameType::Inter => "P",
                FrameType::BiDir => "B",
                FrameType::Switch => "S",
            };

            write!(
                writer,
                "{},{},{:.2},{},{:.6},{:.6},{:.6},{},{:.2},{:.6},{},{:.6}",
                stats.frame_index,
                frame_type_str,
                stats.qp,
                stats.bits,
                stats.complexity.spatial_complexity,
                stats.complexity.temporal_complexity,
                stats.complexity.combined_complexity,
                stats.complexity.sad,
                stats.complexity.variance,
                stats.complexity.encoding_difficulty,
                if stats.complexity.is_scene_change {
                    1
                } else {
                    0
                },
                stats.avg_motion,
            )?;

            if let Some(psnr) = stats.psnr {
                write!(writer, ",{:.2}", psnr)?;
            } else {
                write!(writer, ",")?;
            }

            if let Some(ssim) = stats.ssim {
                write!(writer, ",{:.4}", ssim)?;
            } else {
                write!(writer, ",")?;
            }

            writeln!(writer)?;
        }

        writer.flush()?;
        Ok(())
    }

    /// Load statistics from a file.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);

        let mut width = 0u32;
        let mut height = 0u32;
        let mut framerate_num = 30u32;
        let mut framerate_den = 1u32;
        let mut frames = Vec::new();

        for line in reader.lines() {
            let line = line?;
            let line = line.trim();

            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse header fields
            if line.starts_with("width=") {
                width = line[6..].parse().unwrap_or(0);
                continue;
            }
            if line.starts_with("height=") {
                height = line[7..].parse().unwrap_or(0);
                continue;
            }
            if line.starts_with("framerate=") {
                let parts: Vec<&str> = line[10..].split('/').collect();
                if parts.len() == 2 {
                    framerate_num = parts[0].parse().unwrap_or(30);
                    framerate_den = parts[1].parse().unwrap_or(1);
                }
                continue;
            }

            // Skip other header fields
            if line.contains('=') {
                continue;
            }

            // Parse frame data
            let parts: Vec<&str> = line.split(',').collect();
            if parts.len() < 12 {
                continue;
            }

            let frame_index: u64 = parts[0].parse().unwrap_or(0);
            let frame_type = match parts[1] {
                "I" => FrameType::Key,
                "P" => FrameType::Inter,
                "B" => FrameType::BiDir,
                "S" => FrameType::Switch,
                _ => FrameType::Inter,
            };
            let qp: f64 = parts[2].parse().unwrap_or(28.0);
            let bits: u64 = parts[3].parse().unwrap_or(0);

            let mut complexity = FrameComplexity::new(frame_index, frame_type);
            complexity.spatial_complexity = parts[4].parse().unwrap_or(0.5);
            complexity.temporal_complexity = parts[5].parse().unwrap_or(0.5);
            complexity.combined_complexity = parts[6].parse().unwrap_or(0.5);
            complexity.sad = parts[7].parse().unwrap_or(0);
            complexity.variance = parts[8].parse().unwrap_or(0.0);
            complexity.encoding_difficulty = parts[9].parse().unwrap_or(1.0);
            complexity.is_scene_change = parts[10] == "1";

            let mut stats = FrameStatistics::new(frame_index, frame_type, qp, bits, complexity);
            stats.avg_motion = parts[11].parse().unwrap_or(0.0);

            if parts.len() > 12 && !parts[12].is_empty() {
                if let Ok(psnr) = parts[12].parse::<f64>() {
                    stats.psnr = Some(psnr);
                }
            }

            if parts.len() > 13 && !parts[13].is_empty() {
                if let Ok(ssim) = parts[13].parse::<f64>() {
                    stats.ssim = Some(ssim);
                }
            }

            frames.push(stats);
        }

        let mut stats = Self::new(width, height, framerate_num, framerate_den);
        stats.frames = frames;
        stats.total_frames = stats.frames.len() as u64;
        stats.total_bits = stats.frames.iter().map(|f| f.bits).sum();
        stats.update_averages();

        Ok(stats)
    }
}

/// Complexity statistics summary.
#[derive(Clone, Debug, Default)]
pub struct ComplexityStats {
    /// Mean complexity.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Median complexity.
    pub median: f64,
    /// Minimum complexity.
    pub min: f64,
    /// Maximum complexity.
    pub max: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_complexity(frame_index: u64) -> FrameComplexity {
        FrameComplexity::new(frame_index, FrameType::Inter)
    }

    #[test]
    fn test_frame_statistics_new() {
        let complexity = create_test_complexity(0);
        let stats = FrameStatistics::new(0, FrameType::Key, 28.0, 10000, complexity);

        assert_eq!(stats.frame_index, 0);
        assert_eq!(stats.qp, 28.0);
        assert_eq!(stats.bits, 10000);
    }

    #[test]
    fn test_pass_statistics_add_frame() {
        let mut pass_stats = PassStatistics::new(1920, 1080, 30, 1);

        let complexity = create_test_complexity(0);
        let frame_stats = FrameStatistics::new(0, FrameType::Key, 28.0, 10000, complexity);

        pass_stats.add_frame(frame_stats);

        assert_eq!(pass_stats.total_frames, 1);
        assert_eq!(pass_stats.total_bits, 10000);
        assert_eq!(pass_stats.avg_qp, 28.0);
    }

    #[test]
    fn test_average_bitrate() {
        let mut pass_stats = PassStatistics::new(1920, 1080, 30, 1);

        for i in 0..30 {
            let complexity = create_test_complexity(i);
            let frame_stats = FrameStatistics::new(i, FrameType::Inter, 28.0, 5000, complexity);
            pass_stats.add_frame(frame_stats);
        }

        let avg_bitrate = pass_stats.average_bitrate();
        assert_eq!(avg_bitrate, 5000 * 30); // 5000 bits/frame * 30 fps
    }

    #[test]
    fn test_complexity_distribution() {
        let mut pass_stats = PassStatistics::new(1920, 1080, 30, 1);

        for i in 0..10 {
            let mut complexity = create_test_complexity(i);
            complexity.combined_complexity = i as f64 / 10.0;
            let frame_stats = FrameStatistics::new(i, FrameType::Inter, 28.0, 5000, complexity);
            pass_stats.add_frame(frame_stats);
        }

        let dist = pass_stats.complexity_distribution();
        assert!(dist.mean > 0.0);
        assert!(dist.std_dev >= 0.0);
    }

    #[test]
    fn test_save_and_load() -> std::io::Result<()> {
        let mut pass_stats = PassStatistics::new(1920, 1080, 30, 1);

        for i in 0..5 {
            let complexity = create_test_complexity(i);
            let frame_stats = FrameStatistics::new(i, FrameType::Inter, 28.0, 5000, complexity);
            pass_stats.add_frame(frame_stats);
        }

        let temp_file = std::env::temp_dir()
            .join("oximedia-codec-multipass-stats.txt")
            .to_string_lossy()
            .into_owned();
        pass_stats.save_to_file(&temp_file)?;

        let loaded = PassStatistics::load_from_file(&temp_file)?;
        assert_eq!(loaded.width, 1920);
        assert_eq!(loaded.height, 1080);
        assert_eq!(loaded.total_frames, 5);

        std::fs::remove_file(&temp_file)?;
        Ok(())
    }
}
