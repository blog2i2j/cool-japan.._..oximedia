#![allow(dead_code)]
//! Encoding statistics collection and analysis for multi-pass optimization.
//!
//! This module tracks per-frame, per-GOP, and per-segment encoding statistics
//! to enable data-driven rate control and quality optimization decisions.
//! Statistics can be aggregated, compared across runs, and exported for
//! offline analysis.

use std::collections::HashMap;

/// Codec type for statistics categorization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatsCodec {
    /// H.264 / AVC codec.
    H264,
    /// H.265 / HEVC codec.
    H265,
    /// AV1 codec.
    Av1,
    /// VP9 codec.
    Vp9,
}

/// Frame type classification used in statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StatsFrameType {
    /// Intra-coded frame (I-frame).
    Intra,
    /// Predicted frame (P-frame).
    Predicted,
    /// Bi-predicted frame (B-frame).
    BiPredicted,
    /// Key frame with scene change.
    SceneChange,
}

/// Per-frame encoding statistics.
#[derive(Debug, Clone)]
pub struct FrameEncodeStats {
    /// Frame index in the stream.
    pub frame_index: u64,
    /// Frame type.
    pub frame_type: StatsFrameType,
    /// Encoded size in bytes.
    pub encoded_bytes: u64,
    /// Quantization parameter used.
    pub qp: f64,
    /// PSNR value (dB) if measured.
    pub psnr: Option<f64>,
    /// SSIM value if measured.
    pub ssim: Option<f64>,
    /// Encoding time in microseconds.
    pub encode_time_us: u64,
    /// Number of intra-coded blocks.
    pub intra_blocks: u32,
    /// Number of inter-coded blocks.
    pub inter_blocks: u32,
    /// Number of skip blocks.
    pub skip_blocks: u32,
    /// Average motion vector magnitude.
    pub avg_mv_magnitude: f64,
}

impl FrameEncodeStats {
    /// Creates new frame statistics with the given index and type.
    #[must_use]
    pub fn new(frame_index: u64, frame_type: StatsFrameType) -> Self {
        Self {
            frame_index,
            frame_type,
            encoded_bytes: 0,
            qp: 0.0,
            psnr: None,
            ssim: None,
            encode_time_us: 0,
            intra_blocks: 0,
            inter_blocks: 0,
            skip_blocks: 0,
            avg_mv_magnitude: 0.0,
        }
    }

    /// Returns the total number of coded blocks.
    #[must_use]
    pub fn total_blocks(&self) -> u32 {
        self.intra_blocks + self.inter_blocks + self.skip_blocks
    }

    /// Returns the skip ratio (fraction of blocks that are skip).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn skip_ratio(&self) -> f64 {
        let total = self.total_blocks();
        if total == 0 {
            return 0.0;
        }
        self.skip_blocks as f64 / total as f64
    }

    /// Returns the intra ratio (fraction of blocks that are intra).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn intra_ratio(&self) -> f64 {
        let total = self.total_blocks();
        if total == 0 {
            return 0.0;
        }
        self.intra_blocks as f64 / total as f64
    }

    /// Returns bits per pixel if resolution info is available.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn bits_per_block(&self) -> f64 {
        let total = self.total_blocks();
        if total == 0 {
            return 0.0;
        }
        (self.encoded_bytes * 8) as f64 / total as f64
    }
}

/// GOP-level aggregate statistics.
#[derive(Debug, Clone)]
pub struct GopEncodeStats {
    /// GOP index.
    pub gop_index: u32,
    /// Number of frames in this GOP.
    pub frame_count: u32,
    /// Total encoded bytes for the GOP.
    pub total_bytes: u64,
    /// Average QP across all frames.
    pub avg_qp: f64,
    /// Average PSNR across frames that have it.
    pub avg_psnr: Option<f64>,
    /// Average SSIM across frames that have it.
    pub avg_ssim: Option<f64>,
    /// Total encoding time in microseconds.
    pub total_encode_time_us: u64,
    /// Peak encoded frame size in bytes.
    pub peak_frame_bytes: u64,
    /// Minimum encoded frame size in bytes.
    pub min_frame_bytes: u64,
}

impl GopEncodeStats {
    /// Creates a new empty GOP stats entry.
    #[must_use]
    pub fn new(gop_index: u32) -> Self {
        Self {
            gop_index,
            frame_count: 0,
            total_bytes: 0,
            avg_qp: 0.0,
            avg_psnr: None,
            avg_ssim: None,
            total_encode_time_us: 0,
            peak_frame_bytes: 0,
            min_frame_bytes: u64::MAX,
        }
    }

    /// Adds a frame's statistics to this GOP aggregate.
    pub fn add_frame(&mut self, frame: &FrameEncodeStats) {
        let n = self.frame_count;
        self.frame_count += 1;
        self.total_bytes += frame.encoded_bytes;

        // Running average for QP
        #[allow(clippy::cast_precision_loss)]
        {
            self.avg_qp = (self.avg_qp * n as f64 + frame.qp) / self.frame_count as f64;
        }

        // PSNR average
        if let Some(psnr) = frame.psnr {
            #[allow(clippy::cast_precision_loss)]
            {
                let current = self.avg_psnr.unwrap_or(0.0);
                self.avg_psnr = Some((current * n as f64 + psnr) / self.frame_count as f64);
            }
        }

        // SSIM average
        if let Some(ssim) = frame.ssim {
            #[allow(clippy::cast_precision_loss)]
            {
                let current = self.avg_ssim.unwrap_or(0.0);
                self.avg_ssim = Some((current * n as f64 + ssim) / self.frame_count as f64);
            }
        }

        self.total_encode_time_us += frame.encode_time_us;
        self.peak_frame_bytes = self.peak_frame_bytes.max(frame.encoded_bytes);
        self.min_frame_bytes = self.min_frame_bytes.min(frame.encoded_bytes);
    }

    /// Returns the average bitrate in bits per second for a given frame rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn avg_bitrate_bps(&self, fps: f64) -> f64 {
        if self.frame_count == 0 || fps <= 0.0 {
            return 0.0;
        }
        let duration_s = self.frame_count as f64 / fps;
        (self.total_bytes * 8) as f64 / duration_s
    }

    /// Returns the peak-to-average frame size ratio.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn peak_to_avg_ratio(&self) -> f64 {
        if self.frame_count == 0 || self.total_bytes == 0 {
            return 0.0;
        }
        let avg = self.total_bytes as f64 / self.frame_count as f64;
        if avg <= 0.0 {
            return 0.0;
        }
        self.peak_frame_bytes as f64 / avg
    }
}

/// Histogram bucket for QP distribution analysis.
#[derive(Debug, Clone)]
pub struct QpBucket {
    /// QP value for this bucket.
    pub qp: u32,
    /// Number of frames or blocks at this QP.
    pub count: u64,
    /// Total bytes encoded at this QP.
    pub total_bytes: u64,
}

/// Full encoding session statistics collector.
#[derive(Debug, Clone)]
pub struct EncodeStatsCollector {
    /// Codec being used.
    pub codec: StatsCodec,
    /// Width of the encoded video.
    pub width: u32,
    /// Height of the encoded video.
    pub height: u32,
    /// Frame rate.
    pub fps: f64,
    /// Per-frame statistics stored in order.
    frames: Vec<FrameEncodeStats>,
    /// Per-GOP aggregated statistics.
    gops: Vec<GopEncodeStats>,
    /// QP distribution histogram.
    qp_histogram: HashMap<u32, QpBucket>,
    /// Current GOP being accumulated.
    current_gop: GopEncodeStats,
    /// Next GOP index.
    next_gop_index: u32,
}

impl EncodeStatsCollector {
    /// Creates a new stats collector for the given codec and resolution.
    #[must_use]
    pub fn new(codec: StatsCodec, width: u32, height: u32, fps: f64) -> Self {
        Self {
            codec,
            width,
            height,
            fps,
            frames: Vec::new(),
            gops: Vec::new(),
            qp_histogram: HashMap::new(),
            current_gop: GopEncodeStats::new(0),
            next_gop_index: 1,
        }
    }

    /// Records a frame's encoding statistics.
    pub fn record_frame(&mut self, stats: FrameEncodeStats) {
        // If this is an intra or scene-change frame, finalize previous GOP
        if (stats.frame_type == StatsFrameType::Intra
            || stats.frame_type == StatsFrameType::SceneChange)
            && self.current_gop.frame_count > 0
        {
            let finished_gop = std::mem::replace(
                &mut self.current_gop,
                GopEncodeStats::new(self.next_gop_index),
            );
            self.gops.push(finished_gop);
            self.next_gop_index += 1;
        }

        // Update QP histogram
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        let qp_key = stats.qp.round() as u32;
        let bucket = self.qp_histogram.entry(qp_key).or_insert_with(|| QpBucket {
            qp: qp_key,
            count: 0,
            total_bytes: 0,
        });
        bucket.count += 1;
        bucket.total_bytes += stats.encoded_bytes;

        self.current_gop.add_frame(&stats);
        self.frames.push(stats);
    }

    /// Returns the total number of recorded frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Returns all recorded frame statistics.
    #[must_use]
    pub fn frames(&self) -> &[FrameEncodeStats] {
        &self.frames
    }

    /// Returns all finalized GOP statistics.
    #[must_use]
    pub fn gops(&self) -> &[GopEncodeStats] {
        &self.gops
    }

    /// Finalizes collection and returns a summary.
    pub fn finalize(&mut self) -> EncodeStatsSummary {
        // Finalize current GOP if it has frames
        if self.current_gop.frame_count > 0 {
            let finished = std::mem::replace(
                &mut self.current_gop,
                GopEncodeStats::new(self.next_gop_index),
            );
            self.gops.push(finished);
            self.next_gop_index += 1;
        }

        let total_bytes: u64 = self.frames.iter().map(|f| f.encoded_bytes).sum();
        let total_time_us: u64 = self.frames.iter().map(|f| f.encode_time_us).sum();

        #[allow(clippy::cast_precision_loss)]
        let avg_qp = if self.frames.is_empty() {
            0.0
        } else {
            self.frames.iter().map(|f| f.qp).sum::<f64>() / self.frames.len() as f64
        };

        let psnr_values: Vec<f64> = self.frames.iter().filter_map(|f| f.psnr).collect();
        #[allow(clippy::cast_precision_loss)]
        let avg_psnr = if psnr_values.is_empty() {
            None
        } else {
            Some(psnr_values.iter().sum::<f64>() / psnr_values.len() as f64)
        };

        let ssim_values: Vec<f64> = self.frames.iter().filter_map(|f| f.ssim).collect();
        #[allow(clippy::cast_precision_loss)]
        let avg_ssim = if ssim_values.is_empty() {
            None
        } else {
            Some(ssim_values.iter().sum::<f64>() / ssim_values.len() as f64)
        };

        #[allow(clippy::cast_precision_loss)]
        let duration_s = if self.fps > 0.0 {
            self.frames.len() as f64 / self.fps
        } else {
            0.0
        };

        #[allow(clippy::cast_precision_loss)]
        let avg_bitrate_bps = if duration_s > 0.0 {
            (total_bytes * 8) as f64 / duration_s
        } else {
            0.0
        };

        let frame_type_counts = self.count_frame_types();

        EncodeStatsSummary {
            codec: self.codec,
            total_frames: self.frames.len(),
            total_bytes,
            total_encode_time_us: total_time_us,
            avg_qp,
            avg_psnr,
            avg_ssim,
            avg_bitrate_bps,
            duration_s,
            gop_count: self.gops.len(),
            frame_type_counts,
        }
    }

    /// Counts frames by type.
    fn count_frame_types(&self) -> HashMap<StatsFrameType, usize> {
        let mut counts = HashMap::new();
        for f in &self.frames {
            *counts.entry(f.frame_type).or_insert(0) += 1;
        }
        counts
    }

    /// Returns the QP histogram.
    #[must_use]
    pub fn qp_histogram(&self) -> &HashMap<u32, QpBucket> {
        &self.qp_histogram
    }

    /// Returns the QP standard deviation across all frames.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn qp_stddev(&self) -> f64 {
        if self.frames.len() < 2 {
            return 0.0;
        }
        let mean = self.frames.iter().map(|f| f.qp).sum::<f64>() / self.frames.len() as f64;
        let variance = self
            .frames
            .iter()
            .map(|f| (f.qp - mean).powi(2))
            .sum::<f64>()
            / (self.frames.len() - 1) as f64;
        variance.sqrt()
    }
}

/// Summary of an entire encoding session.
#[derive(Debug, Clone)]
pub struct EncodeStatsSummary {
    /// Codec used.
    pub codec: StatsCodec,
    /// Total number of frames.
    pub total_frames: usize,
    /// Total encoded bytes.
    pub total_bytes: u64,
    /// Total encoding time in microseconds.
    pub total_encode_time_us: u64,
    /// Average QP.
    pub avg_qp: f64,
    /// Average PSNR.
    pub avg_psnr: Option<f64>,
    /// Average SSIM.
    pub avg_ssim: Option<f64>,
    /// Average bitrate in bits per second.
    pub avg_bitrate_bps: f64,
    /// Total duration in seconds.
    pub duration_s: f64,
    /// Number of GOPs.
    pub gop_count: usize,
    /// Frame count by type.
    pub frame_type_counts: HashMap<StatsFrameType, usize>,
}

impl EncodeStatsSummary {
    /// Returns encoding speed as a multiple of real-time.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn realtime_speed(&self) -> f64 {
        if self.duration_s <= 0.0 || self.total_encode_time_us == 0 {
            return 0.0;
        }
        let encode_s = self.total_encode_time_us as f64 / 1_000_000.0;
        self.duration_s / encode_s
    }

    /// Returns the compression ratio (uncompressed / compressed).
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn compression_ratio(&self, bits_per_pixel: u32, width: u32, height: u32) -> f64 {
        if self.total_bytes == 0 || self.total_frames == 0 {
            return 0.0;
        }
        let uncompressed =
            self.total_frames as f64 * width as f64 * height as f64 * bits_per_pixel as f64 / 8.0;
        uncompressed / self.total_bytes as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_frame(index: u64, ft: StatsFrameType, bytes: u64, qp: f64) -> FrameEncodeStats {
        FrameEncodeStats {
            frame_index: index,
            frame_type: ft,
            encoded_bytes: bytes,
            qp,
            psnr: Some(38.0 + qp * 0.1),
            ssim: Some(0.95),
            encode_time_us: 1000,
            intra_blocks: 10,
            inter_blocks: 50,
            skip_blocks: 40,
            avg_mv_magnitude: 2.5,
        }
    }

    #[test]
    fn test_frame_stats_new() {
        let fs = FrameEncodeStats::new(0, StatsFrameType::Intra);
        assert_eq!(fs.frame_index, 0);
        assert_eq!(fs.frame_type, StatsFrameType::Intra);
        assert_eq!(fs.encoded_bytes, 0);
    }

    #[test]
    fn test_total_blocks() {
        let fs = make_frame(0, StatsFrameType::Predicted, 500, 28.0);
        assert_eq!(fs.total_blocks(), 100);
    }

    #[test]
    fn test_skip_ratio() {
        let fs = make_frame(0, StatsFrameType::Predicted, 500, 28.0);
        let ratio = fs.skip_ratio();
        assert!((ratio - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_intra_ratio() {
        let fs = make_frame(0, StatsFrameType::Predicted, 500, 28.0);
        let ratio = fs.intra_ratio();
        assert!((ratio - 0.1).abs() < 1e-9);
    }

    #[test]
    fn test_skip_ratio_zero_blocks() {
        let mut fs = FrameEncodeStats::new(0, StatsFrameType::Intra);
        fs.intra_blocks = 0;
        fs.inter_blocks = 0;
        fs.skip_blocks = 0;
        assert!((fs.skip_ratio() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_bits_per_block() {
        let fs = make_frame(0, StatsFrameType::Predicted, 500, 28.0);
        let bpb = fs.bits_per_block();
        // 500 * 8 / 100 = 40.0
        assert!((bpb - 40.0).abs() < 1e-9);
    }

    #[test]
    fn test_gop_stats_add_frame() {
        let mut gop = GopEncodeStats::new(0);
        let f1 = make_frame(0, StatsFrameType::Intra, 1000, 22.0);
        let f2 = make_frame(1, StatsFrameType::Predicted, 200, 26.0);
        gop.add_frame(&f1);
        gop.add_frame(&f2);
        assert_eq!(gop.frame_count, 2);
        assert_eq!(gop.total_bytes, 1200);
        assert_eq!(gop.peak_frame_bytes, 1000);
        assert_eq!(gop.min_frame_bytes, 200);
    }

    #[test]
    fn test_gop_avg_bitrate() {
        let mut gop = GopEncodeStats::new(0);
        let f1 = make_frame(0, StatsFrameType::Intra, 1000, 22.0);
        let f2 = make_frame(1, StatsFrameType::Predicted, 1000, 24.0);
        gop.add_frame(&f1);
        gop.add_frame(&f2);
        let bps = gop.avg_bitrate_bps(30.0);
        // 2 frames / 30fps = 1/15 s; 2000 * 8 = 16000 bits; 16000 / (1/15) = 240000
        assert!((bps - 240_000.0).abs() < 1.0);
    }

    #[test]
    fn test_gop_peak_to_avg() {
        let mut gop = GopEncodeStats::new(0);
        let f1 = make_frame(0, StatsFrameType::Intra, 1000, 22.0);
        let f2 = make_frame(1, StatsFrameType::Predicted, 200, 26.0);
        gop.add_frame(&f1);
        gop.add_frame(&f2);
        let ratio = gop.peak_to_avg_ratio();
        // avg = 600, peak = 1000, ratio ~= 1.666
        assert!((ratio - 1000.0 / 600.0).abs() < 1e-6);
    }

    #[test]
    fn test_collector_record_and_finalize() {
        let mut collector = EncodeStatsCollector::new(StatsCodec::H265, 1920, 1080, 30.0);
        for i in 0..10 {
            let ft = if i % 5 == 0 {
                StatsFrameType::Intra
            } else {
                StatsFrameType::Predicted
            };
            let frame = make_frame(i, ft, 500 + i * 10, 24.0 + i as f64 * 0.5);
            collector.record_frame(frame);
        }
        assert_eq!(collector.frame_count(), 10);
        let summary = collector.finalize();
        assert_eq!(summary.total_frames, 10);
        assert!(summary.avg_bitrate_bps > 0.0);
        assert!(summary.avg_psnr.is_some());
    }

    #[test]
    fn test_collector_gop_splitting() {
        let mut collector = EncodeStatsCollector::new(StatsCodec::Av1, 1280, 720, 24.0);
        // Frame 0: Intra (starts GOP 0)
        collector.record_frame(make_frame(0, StatsFrameType::Intra, 2000, 20.0));
        // Frames 1-4: Predicted
        for i in 1..5 {
            collector.record_frame(make_frame(i, StatsFrameType::Predicted, 300, 26.0));
        }
        // Frame 5: Intra (starts GOP 1, finalizes GOP 0)
        collector.record_frame(make_frame(5, StatsFrameType::Intra, 1800, 21.0));

        let summary = collector.finalize();
        // Should have 2 GOPs (one finalized by frame 5, one finalized by finalize())
        assert_eq!(summary.gop_count, 2);
    }

    #[test]
    fn test_qp_stddev() {
        let mut collector = EncodeStatsCollector::new(StatsCodec::H264, 1920, 1080, 30.0);
        // All same QP => stddev 0
        for i in 0..5 {
            collector.record_frame(make_frame(i, StatsFrameType::Predicted, 500, 25.0));
        }
        assert!((collector.qp_stddev() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_qp_histogram() {
        let mut collector = EncodeStatsCollector::new(StatsCodec::H264, 1920, 1080, 30.0);
        collector.record_frame(make_frame(0, StatsFrameType::Intra, 1000, 22.0));
        collector.record_frame(make_frame(1, StatsFrameType::Predicted, 300, 22.0));
        collector.record_frame(make_frame(2, StatsFrameType::Predicted, 400, 26.0));
        let hist = collector.qp_histogram();
        assert_eq!(hist.get(&22).expect("entry should exist").count, 2);
        assert_eq!(hist.get(&26).expect("entry should exist").count, 1);
    }

    #[test]
    fn test_summary_realtime_speed() {
        let summary = EncodeStatsSummary {
            codec: StatsCodec::H265,
            total_frames: 300,
            total_bytes: 1_000_000,
            total_encode_time_us: 5_000_000, // 5 seconds
            avg_qp: 25.0,
            avg_psnr: Some(40.0),
            avg_ssim: Some(0.96),
            avg_bitrate_bps: 8_000_000.0,
            duration_s: 10.0, // 10 seconds of video
            gop_count: 5,
            frame_type_counts: HashMap::new(),
        };
        // Realtime speed = 10 / 5 = 2.0x
        assert!((summary.realtime_speed() - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_summary_compression_ratio() {
        let summary = EncodeStatsSummary {
            codec: StatsCodec::H264,
            total_frames: 30,
            total_bytes: 100_000,
            total_encode_time_us: 1_000_000,
            avg_qp: 28.0,
            avg_psnr: None,
            avg_ssim: None,
            avg_bitrate_bps: 800_000.0,
            duration_s: 1.0,
            gop_count: 1,
            frame_type_counts: HashMap::new(),
        };
        let ratio = summary.compression_ratio(24, 1920, 1080);
        // uncompressed = 30 * 1920 * 1080 * 24 / 8 = 30 * 1920 * 1080 * 3 = 186,624,000
        // ratio = 186_624_000 / 100_000 = 1866.24
        assert!(ratio > 1000.0);
    }
}
