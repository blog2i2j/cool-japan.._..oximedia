#![allow(dead_code)]

//! Elementary stream recovery from damaged container files.
//!
//! When a container (MP4, MKV, AVI, etc.) is damaged beyond repair at the
//! container level, this module can extract and reconstruct individual
//! elementary streams (raw H.264, AAC, etc.) by scanning for codec-specific
//! sync patterns in the raw byte data.

use std::collections::HashMap;

/// Elementary stream type that can be recovered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamType {
    /// H.264 / AVC video.
    H264,
    /// H.265 / HEVC video.
    H265,
    /// MPEG-2 video.
    Mpeg2Video,
    /// AAC audio.
    Aac,
    /// MP3 audio.
    Mp3,
    /// AC-3 (Dolby Digital) audio.
    Ac3,
    /// Opus audio.
    Opus,
    /// FLAC audio.
    Flac,
    /// Raw PCM audio.
    Pcm,
}

/// Describes the sync pattern used to find stream data in raw bytes.
#[derive(Debug, Clone)]
pub struct SyncPattern {
    /// The byte pattern that marks a frame start.
    pub pattern: Vec<u8>,
    /// Bitmask for partial matching (0xFF = exact, other = masked).
    pub mask: Vec<u8>,
    /// Minimum distance between consecutive sync points.
    pub min_frame_distance: usize,
    /// Maximum distance between consecutive sync points.
    pub max_frame_distance: usize,
}

/// A recovered frame from the raw byte stream.
#[derive(Debug, Clone)]
pub struct RecoveredFrame {
    /// Stream this frame belongs to.
    pub stream_type: StreamType,
    /// Byte offset in the source file.
    pub offset: u64,
    /// Frame data.
    pub data: Vec<u8>,
    /// Sequence number (order recovered).
    pub sequence: u64,
    /// Whether this is a keyframe / sync sample.
    pub keyframe: bool,
    /// Estimated presentation timestamp in milliseconds.
    pub pts_ms: Option<f64>,
}

/// Confidence level for a recovered stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Confidence {
    /// Very low confidence - might be false positives.
    VeryLow,
    /// Low confidence - some frames found but gaps exist.
    Low,
    /// Medium confidence - most frames recovered.
    Medium,
    /// High confidence - stream looks complete.
    High,
}

/// Result of a stream recovery pass.
#[derive(Debug, Clone)]
pub struct RecoveryResult {
    /// Stream type that was recovered.
    pub stream_type: StreamType,
    /// Number of frames recovered.
    pub frames_recovered: usize,
    /// Number of gaps detected (missing frames).
    pub gaps_detected: usize,
    /// Confidence in the recovered data.
    pub confidence: Confidence,
    /// Total bytes recovered.
    pub bytes_recovered: u64,
    /// Estimated duration in seconds.
    pub estimated_duration_secs: f64,
}

/// Options controlling stream recovery behavior.
#[derive(Debug, Clone)]
pub struct RecoveryOptions {
    /// Stream types to search for (empty = search all).
    pub target_streams: Vec<StreamType>,
    /// Maximum bytes to scan (None = entire file).
    pub max_scan_bytes: Option<u64>,
    /// Minimum number of consecutive sync points to consider valid.
    pub min_sync_run: usize,
    /// Whether to attempt timestamp reconstruction.
    pub reconstruct_timestamps: bool,
    /// Expected frame rate for video (used in timestamp reconstruction).
    pub expected_fps: Option<f64>,
    /// Expected audio sample rate (used in timestamp reconstruction).
    pub expected_sample_rate: Option<u32>,
}

impl Default for RecoveryOptions {
    fn default() -> Self {
        Self {
            target_streams: Vec::new(),
            max_scan_bytes: None,
            min_sync_run: 3,
            reconstruct_timestamps: true,
            expected_fps: None,
            expected_sample_rate: None,
        }
    }
}

/// Pattern library providing sync patterns for known codecs.
#[derive(Debug)]
pub struct PatternLibrary {
    patterns: HashMap<StreamType, SyncPattern>,
}

impl PatternLibrary {
    /// Create a library with all built-in patterns.
    pub fn new() -> Self {
        let mut patterns = HashMap::new();

        // H.264: NAL start code 0x00 0x00 0x01 or 0x00 0x00 0x00 0x01
        patterns.insert(
            StreamType::H264,
            SyncPattern {
                pattern: vec![0x00, 0x00, 0x01],
                mask: vec![0xFF, 0xFF, 0xFF],
                min_frame_distance: 64,
                max_frame_distance: 2_000_000,
            },
        );

        // H.265: NAL start code same prefix, different NAL unit type byte
        patterns.insert(
            StreamType::H265,
            SyncPattern {
                pattern: vec![0x00, 0x00, 0x01],
                mask: vec![0xFF, 0xFF, 0xFF],
                min_frame_distance: 64,
                max_frame_distance: 4_000_000,
            },
        );

        // AAC ADTS: 0xFFF (12 bits sync word)
        patterns.insert(
            StreamType::Aac,
            SyncPattern {
                pattern: vec![0xFF, 0xF0],
                mask: vec![0xFF, 0xF0],
                min_frame_distance: 7,
                max_frame_distance: 8192,
            },
        );

        // MP3: 0xFFE0 (11 bits sync word)
        patterns.insert(
            StreamType::Mp3,
            SyncPattern {
                pattern: vec![0xFF, 0xE0],
                mask: vec![0xFF, 0xE0],
                min_frame_distance: 96,
                max_frame_distance: 4608,
            },
        );

        // AC-3: 0x0B77
        patterns.insert(
            StreamType::Ac3,
            SyncPattern {
                pattern: vec![0x0B, 0x77],
                mask: vec![0xFF, 0xFF],
                min_frame_distance: 128,
                max_frame_distance: 4096,
            },
        );

        // FLAC: "fLaC" magic
        patterns.insert(
            StreamType::Flac,
            SyncPattern {
                pattern: vec![0x66, 0x4C, 0x61, 0x43],
                mask: vec![0xFF, 0xFF, 0xFF, 0xFF],
                min_frame_distance: 1024,
                max_frame_distance: 65536,
            },
        );

        Self { patterns }
    }

    /// Look up the sync pattern for a given stream type.
    pub fn get_pattern(&self, stream_type: StreamType) -> Option<&SyncPattern> {
        self.patterns.get(&stream_type)
    }

    /// Return all supported stream types.
    pub fn supported_types(&self) -> Vec<StreamType> {
        self.patterns.keys().copied().collect()
    }
}

impl Default for PatternLibrary {
    fn default() -> Self {
        Self::new()
    }
}

/// The main stream recovery engine.
#[derive(Debug)]
pub struct StreamRecoveryEngine {
    library: PatternLibrary,
    options: RecoveryOptions,
}

impl StreamRecoveryEngine {
    /// Create a new engine with default options.
    pub fn new() -> Self {
        Self {
            library: PatternLibrary::new(),
            options: RecoveryOptions::default(),
        }
    }

    /// Create with specific options.
    pub fn with_options(options: RecoveryOptions) -> Self {
        Self {
            library: PatternLibrary::new(),
            options,
        }
    }

    /// Search `data` for occurrences of `pattern` (with mask), return byte offsets.
    #[allow(clippy::cast_precision_loss)]
    pub fn find_sync_points(&self, data: &[u8], pattern: &SyncPattern) -> Vec<u64> {
        let pat_len = pattern.pattern.len();
        if data.len() < pat_len {
            return Vec::new();
        }
        let mut offsets = Vec::new();
        let end = data.len() - pat_len;
        let mut i = 0;
        while i <= end {
            let matched = pattern
                .pattern
                .iter()
                .enumerate()
                .all(|(j, &p)| (data[i + j] & pattern.mask[j]) == (p & pattern.mask[j]));
            if matched {
                // Check min distance from previous
                if let Some(&last) = offsets.last() {
                    let dist = (i as u64) - last;
                    if (dist as usize) < pattern.min_frame_distance {
                        i += 1;
                        continue;
                    }
                }
                offsets.push(i as u64);
            }
            i += 1;
        }
        offsets
    }

    /// Validate a run of sync points to filter false positives.
    pub fn validate_sync_run(&self, offsets: &[u64], pattern: &SyncPattern) -> Vec<u64> {
        if offsets.len() < self.options.min_sync_run {
            return Vec::new();
        }
        let mut valid = Vec::new();
        let mut run_start = 0;
        let mut run_len = 1usize;

        for i in 1..offsets.len() {
            let dist = (offsets[i] - offsets[i - 1]) as usize;
            if dist <= pattern.max_frame_distance {
                run_len += 1;
            } else {
                if run_len >= self.options.min_sync_run {
                    for j in run_start..i {
                        valid.push(offsets[j]);
                    }
                }
                run_start = i;
                run_len = 1;
            }
        }
        // Flush last run
        if run_len >= self.options.min_sync_run {
            for j in run_start..offsets.len() {
                valid.push(offsets[j]);
            }
        }
        valid
    }

    /// Estimate frame timestamps given offsets and a frame rate.
    #[allow(clippy::cast_precision_loss)]
    pub fn reconstruct_timestamps(offsets: &[u64], fps: f64) -> Vec<f64> {
        if fps <= 0.0 {
            return vec![0.0; offsets.len()];
        }
        let frame_duration_ms = 1000.0 / fps;
        (0..offsets.len())
            .map(|i| i as f64 * frame_duration_ms)
            .collect()
    }

    /// Determine confidence based on gap analysis.
    #[allow(clippy::cast_precision_loss)]
    pub fn assess_confidence(&self, total_offsets: usize, valid_offsets: usize) -> Confidence {
        if total_offsets == 0 {
            return Confidence::VeryLow;
        }
        let ratio = valid_offsets as f64 / total_offsets as f64;
        if ratio > 0.9 {
            Confidence::High
        } else if ratio > 0.7 {
            Confidence::Medium
        } else if ratio > 0.4 {
            Confidence::Low
        } else {
            Confidence::VeryLow
        }
    }

    /// Count gaps in a sorted offset list using expected max distance.
    pub fn count_gaps(offsets: &[u64], max_distance: usize) -> usize {
        let mut gaps = 0;
        for i in 1..offsets.len() {
            let dist = (offsets[i] - offsets[i - 1]) as usize;
            if dist > max_distance {
                gaps += 1;
            }
        }
        gaps
    }

    /// Build a recovery result from analyzed data.
    #[allow(clippy::cast_precision_loss)]
    pub fn build_result(
        &self,
        stream_type: StreamType,
        valid_offsets: &[u64],
        total_offsets: usize,
        max_distance: usize,
        fps: Option<f64>,
    ) -> RecoveryResult {
        let gaps = Self::count_gaps(valid_offsets, max_distance);
        let confidence = self.assess_confidence(total_offsets, valid_offsets.len());
        let estimated_duration = if let Some(f) = fps {
            if f > 0.0 {
                valid_offsets.len() as f64 / f
            } else {
                0.0
            }
        } else {
            0.0
        };
        // Rough bytes estimate
        let bytes_recovered = if valid_offsets.len() > 1 {
            valid_offsets.last().unwrap_or(&0) - valid_offsets.first().unwrap_or(&0)
        } else {
            0
        };
        RecoveryResult {
            stream_type,
            frames_recovered: valid_offsets.len(),
            gaps_detected: gaps,
            confidence,
            bytes_recovered,
            estimated_duration_secs: estimated_duration,
        }
    }
}

impl Default for StreamRecoveryEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stream_type_equality() {
        assert_eq!(StreamType::H264, StreamType::H264);
        assert_ne!(StreamType::Aac, StreamType::Mp3);
    }

    #[test]
    fn test_confidence_ordering() {
        assert!(Confidence::VeryLow < Confidence::Low);
        assert!(Confidence::Low < Confidence::Medium);
        assert!(Confidence::Medium < Confidence::High);
    }

    #[test]
    fn test_pattern_library_new() {
        let lib = PatternLibrary::new();
        assert!(lib.get_pattern(StreamType::H264).is_some());
        assert!(lib.get_pattern(StreamType::Aac).is_some());
        assert!(lib.get_pattern(StreamType::Mp3).is_some());
        assert!(lib.get_pattern(StreamType::Ac3).is_some());
    }

    #[test]
    fn test_pattern_library_supported_types() {
        let lib = PatternLibrary::new();
        let types = lib.supported_types();
        assert!(types.contains(&StreamType::H264));
        assert!(types.contains(&StreamType::Aac));
    }

    #[test]
    fn test_pattern_library_unsupported() {
        let lib = PatternLibrary::new();
        assert!(lib.get_pattern(StreamType::Pcm).is_none());
    }

    #[test]
    fn test_find_sync_points_mp3() {
        let engine = StreamRecoveryEngine::new();
        // Build data with two MP3 sync words spaced 200 bytes apart
        let mut data = vec![0u8; 400];
        data[0] = 0xFF;
        data[1] = 0xFB;
        data[200] = 0xFF;
        data[201] = 0xFB;
        let pat = engine
            .library
            .get_pattern(StreamType::Mp3)
            .expect("unexpected None/Err");
        let offsets = engine.find_sync_points(&data, pat);
        assert_eq!(offsets.len(), 2);
        assert_eq!(offsets[0], 0);
        assert_eq!(offsets[1], 200);
    }

    #[test]
    fn test_find_sync_points_empty() {
        let engine = StreamRecoveryEngine::new();
        let data = vec![0u8; 100];
        let pat = engine
            .library
            .get_pattern(StreamType::Ac3)
            .expect("unexpected None/Err");
        let offsets = engine.find_sync_points(&data, pat);
        assert!(offsets.is_empty());
    }

    #[test]
    fn test_find_sync_points_short_data() {
        let engine = StreamRecoveryEngine::new();
        let data = vec![0u8; 1];
        let pat = engine
            .library
            .get_pattern(StreamType::Mp3)
            .expect("unexpected None/Err");
        let offsets = engine.find_sync_points(&data, pat);
        assert!(offsets.is_empty());
    }

    #[test]
    fn test_validate_sync_run_filters_short() {
        let engine = StreamRecoveryEngine::with_options(RecoveryOptions {
            min_sync_run: 3,
            ..Default::default()
        });
        let pat = SyncPattern {
            pattern: vec![0xFF],
            mask: vec![0xFF],
            min_frame_distance: 1,
            max_frame_distance: 100,
        };
        // Only 2 offsets: below min_sync_run of 3
        let offsets = vec![0, 50];
        let valid = engine.validate_sync_run(&offsets, &pat);
        assert!(valid.is_empty());
    }

    #[test]
    fn test_validate_sync_run_keeps_long() {
        let engine = StreamRecoveryEngine::with_options(RecoveryOptions {
            min_sync_run: 3,
            ..Default::default()
        });
        let pat = SyncPattern {
            pattern: vec![0xFF],
            mask: vec![0xFF],
            min_frame_distance: 1,
            max_frame_distance: 200,
        };
        let offsets = vec![0, 100, 200, 300];
        let valid = engine.validate_sync_run(&offsets, &pat);
        assert_eq!(valid.len(), 4);
    }

    #[test]
    fn test_reconstruct_timestamps() {
        let offsets = vec![0, 1000, 2000, 3000, 4000];
        let ts = StreamRecoveryEngine::reconstruct_timestamps(&offsets, 25.0);
        assert_eq!(ts.len(), 5);
        assert!((ts[0] - 0.0).abs() < 0.001);
        assert!((ts[1] - 40.0).abs() < 0.001);
    }

    #[test]
    fn test_reconstruct_timestamps_zero_fps() {
        let offsets = vec![0, 1000];
        let ts = StreamRecoveryEngine::reconstruct_timestamps(&offsets, 0.0);
        assert_eq!(ts, vec![0.0, 0.0]);
    }

    #[test]
    fn test_assess_confidence_high() {
        let engine = StreamRecoveryEngine::new();
        assert_eq!(engine.assess_confidence(100, 95), Confidence::High);
    }

    #[test]
    fn test_assess_confidence_low() {
        let engine = StreamRecoveryEngine::new();
        assert_eq!(engine.assess_confidence(100, 50), Confidence::Low);
    }

    #[test]
    fn test_assess_confidence_zero() {
        let engine = StreamRecoveryEngine::new();
        assert_eq!(engine.assess_confidence(0, 0), Confidence::VeryLow);
    }

    #[test]
    fn test_count_gaps() {
        let offsets = vec![0, 100, 200, 5000, 5100];
        let gaps = StreamRecoveryEngine::count_gaps(&offsets, 300);
        assert_eq!(gaps, 1);
    }

    #[test]
    fn test_build_result() {
        let engine = StreamRecoveryEngine::new();
        let offsets = vec![0, 100, 200, 300, 400];
        let result = engine.build_result(StreamType::H264, &offsets, 5, 200, Some(25.0));
        assert_eq!(result.stream_type, StreamType::H264);
        assert_eq!(result.frames_recovered, 5);
        assert_eq!(result.gaps_detected, 0);
        assert_eq!(result.confidence, Confidence::High);
        assert!((result.estimated_duration_secs - 0.2).abs() < 0.01);
    }

    #[test]
    fn test_recovery_options_default() {
        let opts = RecoveryOptions::default();
        assert!(opts.target_streams.is_empty());
        assert!(opts.max_scan_bytes.is_none());
        assert_eq!(opts.min_sync_run, 3);
        assert!(opts.reconstruct_timestamps);
    }
}
