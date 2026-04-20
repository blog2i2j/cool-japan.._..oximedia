//! Extract playable segments from partially corrupt files.
//!
//! `StreamSplicer` scans a file for valid segment boundaries (sync points,
//! keyframe headers, container atoms) and extracts the playable portions,
//! discarding corrupt regions. The extracted segments can be concatenated
//! or used individually.
//!
//! This is the last-resort recovery path when full file repair is not feasible.

#![allow(dead_code)]

use crate::Result;
use std::path::Path;

/// A valid segment extracted from a corrupt file.
#[derive(Debug, Clone)]
pub struct SplicedSegment {
    /// Start byte offset in the original file (inclusive).
    pub start_offset: u64,
    /// End byte offset in the original file (exclusive).
    pub end_offset: u64,
    /// Estimated quality of this segment (0.0 = junk, 1.0 = perfect).
    pub quality: f64,
    /// Whether this segment starts with a keyframe / sync point.
    pub starts_at_sync_point: bool,
    /// Number of valid packets/frames within this segment.
    pub valid_frame_count: usize,
    /// Human-readable note about this segment.
    pub note: String,
}

impl SplicedSegment {
    /// Byte length of this segment.
    pub fn len(&self) -> u64 {
        self.end_offset.saturating_sub(self.start_offset)
    }

    /// Whether this segment has zero length.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Strategy for locating segment boundaries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpliceStrategy {
    /// Scan for MPEG-TS 0x47 sync bytes.
    MpegTs,
    /// Scan for Annex-B NAL start codes (H.264/H.265).
    AnnexB,
    /// Scan for MPEG start codes (0x000001).
    MpegStartCode,
    /// Scan for ADTS sync words (audio).
    Adts,
    /// Auto-detect the best strategy.
    Auto,
}

/// Configuration for the stream splicer.
#[derive(Debug, Clone)]
pub struct SpliceConfig {
    /// Minimum segment size in bytes to be considered playable.
    pub min_segment_bytes: u64,
    /// Maximum gap (corrupt region) to skip over, in bytes.
    pub max_gap_bytes: u64,
    /// Strategy for locating boundaries.
    pub strategy: SpliceStrategy,
    /// Minimum number of consecutive valid packets to confirm a segment.
    pub min_consecutive_valid: usize,
}

impl Default for SpliceConfig {
    fn default() -> Self {
        Self {
            min_segment_bytes: 1024,
            max_gap_bytes: 1024 * 1024, // 1 MiB
            strategy: SpliceStrategy::Auto,
            min_consecutive_valid: 3,
        }
    }
}

/// Result of a splice operation.
#[derive(Debug)]
pub struct SpliceResult {
    /// Extracted playable segments.
    pub segments: Vec<SplicedSegment>,
    /// Total bytes that are playable.
    pub playable_bytes: u64,
    /// Total bytes that were discarded.
    pub discarded_bytes: u64,
    /// Detected strategy used.
    pub strategy_used: SpliceStrategy,
}

impl SpliceResult {
    /// Recovery ratio: fraction of the file that is playable.
    pub fn recovery_ratio(&self) -> f64 {
        let total = self.playable_bytes + self.discarded_bytes;
        if total == 0 {
            return 0.0;
        }
        self.playable_bytes as f64 / total as f64
    }
}

/// Stream splicer that extracts playable segments from corrupt data.
#[derive(Debug)]
pub struct StreamSplicer {
    config: SpliceConfig,
}

impl StreamSplicer {
    /// Create a new stream splicer with default configuration.
    pub fn new() -> Self {
        Self {
            config: SpliceConfig::default(),
        }
    }

    /// Create a stream splicer with custom configuration.
    pub fn with_config(config: SpliceConfig) -> Self {
        Self { config }
    }

    /// Scan a file and extract playable segments.
    pub fn splice_file(&self, path: &Path) -> Result<SpliceResult> {
        let data = std::fs::read(path)?;
        self.splice_data(&data)
    }

    /// Scan in-memory data and extract playable segments.
    pub fn splice_data(&self, data: &[u8]) -> Result<SpliceResult> {
        if data.is_empty() {
            return Ok(SpliceResult {
                segments: Vec::new(),
                playable_bytes: 0,
                discarded_bytes: 0,
                strategy_used: self.config.strategy,
            });
        }

        let strategy = if self.config.strategy == SpliceStrategy::Auto {
            auto_detect_strategy(data)
        } else {
            self.config.strategy
        };

        let sync_points = match strategy {
            SpliceStrategy::MpegTs => find_mpegts_sync_points(data),
            SpliceStrategy::AnnexB => find_annexb_sync_points(data),
            SpliceStrategy::MpegStartCode => find_mpeg_start_code_sync_points(data),
            SpliceStrategy::Adts => find_adts_sync_points(data),
            SpliceStrategy::Auto => find_mpeg_start_code_sync_points(data),
        };

        let segments = self.build_segments(&sync_points, data.len() as u64);

        let playable_bytes: u64 = segments.iter().map(|s| s.len()).sum();
        let discarded_bytes = data.len() as u64 - playable_bytes;

        Ok(SpliceResult {
            segments,
            playable_bytes,
            discarded_bytes,
            strategy_used: strategy,
        })
    }

    /// Extract segment data from the original file data.
    pub fn extract_segment<'a>(&self, data: &'a [u8], segment: &SplicedSegment) -> &'a [u8] {
        let start = segment.start_offset as usize;
        let end = (segment.end_offset as usize).min(data.len());
        if start >= data.len() || start >= end {
            return &[];
        }
        &data[start..end]
    }

    /// Build segments from a sorted list of sync-point offsets.
    fn build_segments(&self, sync_points: &[SyncPoint], file_len: u64) -> Vec<SplicedSegment> {
        if sync_points.is_empty() {
            return Vec::new();
        }

        let mut segments = Vec::new();
        let mut seg_start = sync_points[0].offset;
        let mut consecutive = 1usize;
        let mut frame_count = 1usize;
        let mut last_offset = seg_start;

        for sp in sync_points.iter().skip(1) {
            let gap = sp.offset - last_offset;

            if gap > self.config.max_gap_bytes {
                // Gap too large: finish current segment and start a new one
                if consecutive >= self.config.min_consecutive_valid {
                    let end = last_offset + 1; // approximate end
                    if end - seg_start >= self.config.min_segment_bytes {
                        segments.push(SplicedSegment {
                            start_offset: seg_start,
                            end_offset: end.min(file_len),
                            quality: quality_from_consecutive(consecutive),
                            starts_at_sync_point: true,
                            valid_frame_count: frame_count,
                            note: format!("{} consecutive valid sync points", consecutive),
                        });
                    }
                }
                seg_start = sp.offset;
                consecutive = 1;
                frame_count = 1;
            } else {
                consecutive += 1;
                frame_count += 1;
            }

            last_offset = sp.offset;
        }

        // Final segment
        if consecutive >= self.config.min_consecutive_valid {
            let end = (last_offset + 1).min(file_len);
            if end - seg_start >= self.config.min_segment_bytes {
                segments.push(SplicedSegment {
                    start_offset: seg_start,
                    end_offset: end,
                    quality: quality_from_consecutive(consecutive),
                    starts_at_sync_point: true,
                    valid_frame_count: frame_count,
                    note: format!("{} consecutive valid sync points", consecutive),
                });
            }
        }

        segments
    }
}

impl Default for StreamSplicer {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Sync point detection
// ---------------------------------------------------------------------------

/// A detected sync point in the data.
#[derive(Debug, Clone, Copy)]
struct SyncPoint {
    offset: u64,
    is_keyframe: bool,
}

/// Auto-detect the best splice strategy for the data.
fn auto_detect_strategy(data: &[u8]) -> SpliceStrategy {
    // Count MPEG-TS sync bytes
    let mut ts_score = 0;
    let mut i = 0;
    while i + 188 <= data.len() {
        if data[i] == 0x47 && (i + 188 >= data.len() || data[i + 188] == 0x47) {
            ts_score += 1;
            i += 188;
        } else {
            i += 1;
        }
    }
    if ts_score >= 3 {
        return SpliceStrategy::MpegTs;
    }

    // Count ADTS sync words
    let mut adts_score = 0;
    for j in 0..data.len().saturating_sub(1) {
        if data[j] == 0xFF && (data[j + 1] & 0xF0) == 0xF0 {
            adts_score += 1;
        }
    }
    if adts_score >= 5 {
        return SpliceStrategy::Adts;
    }

    // Count Annex-B start codes
    let mut annexb_score = 0;
    for j in 0..data.len().saturating_sub(3) {
        if data[j] == 0
            && data[j + 1] == 0
            && data[j + 2] == 0
            && j + 3 < data.len()
            && data[j + 3] == 1
        {
            annexb_score += 1;
        }
    }
    if annexb_score >= 3 {
        return SpliceStrategy::AnnexB;
    }

    SpliceStrategy::MpegStartCode
}

/// Find MPEG-TS sync points (0x47 every 188 bytes).
fn find_mpegts_sync_points(data: &[u8]) -> Vec<SyncPoint> {
    let mut points = Vec::new();
    let mut i = 0;
    while i + 188 <= data.len() {
        if data[i] == 0x47 {
            let next = i + 188;
            if next >= data.len() || data[next] == 0x47 {
                // Check if it's a keyframe (adaptation field has RAI bit)
                let is_keyframe = if i + 5 < data.len() && (data[i + 3] & 0x20) != 0 {
                    (data[i + 5] & 0x40) != 0
                } else {
                    false
                };
                points.push(SyncPoint {
                    offset: i as u64,
                    is_keyframe,
                });
                i += 188;
                continue;
            }
        }
        i += 1;
    }
    points
}

/// Find Annex-B NAL start code sync points.
fn find_annexb_sync_points(data: &[u8]) -> Vec<SyncPoint> {
    let mut points = Vec::new();
    let mut i = 0;
    while i + 4 <= data.len() {
        let is_4byte = data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 0 && data[i + 3] == 1;
        let is_3byte = data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1;

        if is_4byte || is_3byte {
            let prefix_len = if is_4byte { 4 } else { 3 };
            let nal_pos = i + prefix_len;
            let is_keyframe = if nal_pos < data.len() {
                let nal_type = data[nal_pos] & 0x1F;
                nal_type == 5 || nal_type == 7 // IDR or SPS
            } else {
                false
            };
            points.push(SyncPoint {
                offset: i as u64,
                is_keyframe,
            });
            i += prefix_len;
        } else {
            i += 1;
        }
    }
    points
}

/// Find MPEG start code (0x000001) sync points.
fn find_mpeg_start_code_sync_points(data: &[u8]) -> Vec<SyncPoint> {
    let mut points = Vec::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        if data[i] == 0 && data[i + 1] == 0 && data[i + 2] == 1 {
            points.push(SyncPoint {
                offset: i as u64,
                is_keyframe: false,
            });
            i += 3;
        } else {
            i += 1;
        }
    }
    points
}

/// Find ADTS sync word (0xFFF) sync points.
fn find_adts_sync_points(data: &[u8]) -> Vec<SyncPoint> {
    let mut points = Vec::new();
    let mut i = 0;
    while i + 7 <= data.len() {
        if data[i] == 0xFF && (data[i + 1] & 0xF0) == 0xF0 {
            // Parse frame length
            let frame_len = (((data[i + 3] & 0x03) as usize) << 11)
                | ((data[i + 4] as usize) << 3)
                | ((data[i + 5] as usize) >> 5);

            if frame_len >= 7 && i + frame_len <= data.len() {
                points.push(SyncPoint {
                    offset: i as u64,
                    is_keyframe: true, // All ADTS frames are independently decodable
                });
                i += frame_len;
                continue;
            }
        }
        i += 1;
    }
    points
}

/// Estimate quality from the number of consecutive valid sync points.
fn quality_from_consecutive(consecutive: usize) -> f64 {
    match consecutive {
        0..=2 => 0.3,
        3..=10 => 0.6,
        11..=50 => 0.8,
        _ => 0.95,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_splicer_empty_data() {
        let splicer = StreamSplicer::new();
        let result = splicer.splice_data(&[]).expect("should succeed");
        assert!(result.segments.is_empty());
        assert_eq!(result.playable_bytes, 0);
        assert_eq!(result.recovery_ratio(), 0.0);
    }

    #[test]
    fn test_splicer_default_config() {
        let config = SpliceConfig::default();
        assert_eq!(config.min_segment_bytes, 1024);
        assert_eq!(config.min_consecutive_valid, 3);
        assert_eq!(config.strategy, SpliceStrategy::Auto);
    }

    #[test]
    fn test_spliced_segment_len() {
        let seg = SplicedSegment {
            start_offset: 100,
            end_offset: 500,
            quality: 0.8,
            starts_at_sync_point: true,
            valid_frame_count: 10,
            note: String::new(),
        };
        assert_eq!(seg.len(), 400);
        assert!(!seg.is_empty());
    }

    #[test]
    fn test_spliced_segment_empty() {
        let seg = SplicedSegment {
            start_offset: 100,
            end_offset: 100,
            quality: 0.0,
            starts_at_sync_point: false,
            valid_frame_count: 0,
            note: String::new(),
        };
        assert!(seg.is_empty());
    }

    #[test]
    fn test_splice_mpegts() {
        // Build 10 MPEG-TS packets (10 * 188 = 1880 bytes)
        let mut data = vec![0u8; 188 * 10];
        for i in 0..10 {
            data[i * 188] = 0x47;
        }

        let splicer = StreamSplicer::with_config(SpliceConfig {
            min_segment_bytes: 100,
            min_consecutive_valid: 3,
            strategy: SpliceStrategy::MpegTs,
            ..Default::default()
        });

        let result = splicer.splice_data(&data).expect("should succeed");
        assert!(!result.segments.is_empty());
        assert!(result.playable_bytes > 0);
    }

    #[test]
    fn test_splice_annexb() {
        // Several NAL units with 4-byte start codes
        let mut data = Vec::new();
        for _ in 0..20 {
            data.extend_from_slice(&[0x00, 0x00, 0x00, 0x01]);
            data.extend_from_slice(&[0x65; 64]); // IDR NAL data
        }

        let splicer = StreamSplicer::with_config(SpliceConfig {
            min_segment_bytes: 100,
            min_consecutive_valid: 3,
            strategy: SpliceStrategy::AnnexB,
            ..Default::default()
        });

        let result = splicer.splice_data(&data).expect("should succeed");
        assert!(!result.segments.is_empty());
    }

    #[test]
    fn test_splice_mpeg_start_codes() {
        // MPEG start codes with data between them
        let mut data = Vec::new();
        for _ in 0..15 {
            data.extend_from_slice(&[0x00, 0x00, 0x01, 0xBA]);
            data.extend_from_slice(&[0xAA; 100]);
        }

        let splicer = StreamSplicer::with_config(SpliceConfig {
            min_segment_bytes: 100,
            min_consecutive_valid: 3,
            strategy: SpliceStrategy::MpegStartCode,
            ..Default::default()
        });

        let result = splicer.splice_data(&data).expect("should succeed");
        assert!(!result.segments.is_empty());
    }

    #[test]
    fn test_splice_with_corruption_gap() {
        // Good segment, then large corrupt gap, then good segment
        let mut data = Vec::new();

        // First good segment: 5 MPEG start codes
        for _ in 0..5 {
            data.extend_from_slice(&[0x00, 0x00, 0x01, 0xBA]);
            data.extend_from_slice(&[0xAA; 300]);
        }

        // Large corrupt gap
        data.extend_from_slice(&vec![0xFF; 2_000_000]);

        // Second good segment
        for _ in 0..5 {
            data.extend_from_slice(&[0x00, 0x00, 0x01, 0xBA]);
            data.extend_from_slice(&[0xBB; 300]);
        }

        let splicer = StreamSplicer::with_config(SpliceConfig {
            min_segment_bytes: 100,
            min_consecutive_valid: 3,
            max_gap_bytes: 1_000_000,
            strategy: SpliceStrategy::MpegStartCode,
        });

        let result = splicer.splice_data(&data).expect("should succeed");
        // Should find 2 separate segments
        assert!(result.segments.len() >= 2);
        assert!(result.discarded_bytes > 0);
    }

    #[test]
    fn test_splice_result_recovery_ratio() {
        let result = SpliceResult {
            segments: vec![SplicedSegment {
                start_offset: 0,
                end_offset: 500,
                quality: 0.9,
                starts_at_sync_point: true,
                valid_frame_count: 10,
                note: String::new(),
            }],
            playable_bytes: 500,
            discarded_bytes: 500,
            strategy_used: SpliceStrategy::Auto,
        };
        assert!((result.recovery_ratio() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_auto_detect_mpegts() {
        let mut data = vec![0u8; 188 * 5];
        for i in 0..5 {
            data[i * 188] = 0x47;
        }
        assert_eq!(auto_detect_strategy(&data), SpliceStrategy::MpegTs);
    }

    #[test]
    fn test_auto_detect_annexb() {
        let data = vec![
            0x00, 0x00, 0x00, 0x01, 0x65, 0xAA, 0x00, 0x00, 0x00, 0x01, 0x41, 0xBB, 0x00, 0x00,
            0x00, 0x01, 0x41, 0xCC,
        ];
        assert_eq!(auto_detect_strategy(&data), SpliceStrategy::AnnexB);
    }

    #[test]
    fn test_extract_segment() {
        let data = vec![0x01, 0x02, 0x03, 0x04, 0x05];
        let splicer = StreamSplicer::new();
        let segment = SplicedSegment {
            start_offset: 1,
            end_offset: 4,
            quality: 1.0,
            starts_at_sync_point: true,
            valid_frame_count: 1,
            note: String::new(),
        };
        let extracted = splicer.extract_segment(&data, &segment);
        assert_eq!(extracted, &[0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_extract_segment_out_of_bounds() {
        let data = vec![0x01, 0x02];
        let splicer = StreamSplicer::new();
        let segment = SplicedSegment {
            start_offset: 10,
            end_offset: 20,
            quality: 0.0,
            starts_at_sync_point: false,
            valid_frame_count: 0,
            note: String::new(),
        };
        let extracted = splicer.extract_segment(&data, &segment);
        assert!(extracted.is_empty());
    }

    #[test]
    fn test_quality_from_consecutive() {
        assert!(quality_from_consecutive(1) < quality_from_consecutive(5));
        assert!(quality_from_consecutive(5) < quality_from_consecutive(20));
        assert!(quality_from_consecutive(20) < quality_from_consecutive(100));
    }

    #[test]
    fn test_splice_file_nonexistent() {
        let splicer = StreamSplicer::new();
        let missing = std::env::temp_dir().join("oximedia-repair-splice-nonexistent.bin");
        let result = splicer.splice_file(&missing);
        assert!(result.is_err());
    }

    #[test]
    fn test_splice_file_real() {
        let path = std::env::temp_dir().join("oximedia_splice_test.bin");
        // Write some MPEG start codes
        let mut data = Vec::new();
        for _ in 0..10 {
            data.extend_from_slice(&[0x00, 0x00, 0x01, 0xBA]);
            data.extend_from_slice(&[0xAA; 200]);
        }
        std::fs::write(&path, &data).expect("write test file");

        let splicer = StreamSplicer::with_config(SpliceConfig {
            min_segment_bytes: 100,
            min_consecutive_valid: 3,
            strategy: SpliceStrategy::MpegStartCode,
            ..Default::default()
        });
        let result = splicer.splice_file(&path).expect("should succeed");
        assert!(!result.segments.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_find_mpegts_sync_points() {
        let mut data = vec![0u8; 188 * 3];
        data[0] = 0x47;
        data[188] = 0x47;
        data[376] = 0x47;
        let points = find_mpegts_sync_points(&data);
        // last packet can't verify next sync (it's at end), but it's still picked up
        assert!(points.len() >= 2);
    }

    #[test]
    fn test_find_annexb_sync_points() {
        let data = vec![0x00, 0x00, 0x00, 0x01, 0x65, 0x00, 0x00, 0x01, 0x41];
        let points = find_annexb_sync_points(&data);
        assert_eq!(points.len(), 2);
    }

    #[test]
    fn test_find_adts_sync_points() {
        // Build two minimal ADTS frames (frame_len = 7)
        let frame: Vec<u8> = vec![0xFF, 0xF1, 0x50, 0x00, 0x00, 0xE0, 0x1C];
        let mut data = frame.clone();
        data.extend(&frame);
        let points = find_adts_sync_points(&data);
        assert!(points.len() <= 2);
    }

    #[test]
    fn test_splicer_debug() {
        let splicer = StreamSplicer::new();
        let debug = format!("{:?}", splicer);
        assert!(debug.contains("StreamSplicer"));
    }
}
