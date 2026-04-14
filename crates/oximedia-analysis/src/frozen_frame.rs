//! Duplicate / frozen frame detection.
//!
//! Detects segments of video where the same frame is repeated (frozen video),
//! commonly caused by encoder errors, network drops, or intentional freeze-frames.
//!
//! # Algorithm
//!
//! Each frame is compared to the previous one via a [`FrameHash`].  When
//! consecutive frames have identical (or near-identical) hashes for more than
//! a configurable run length, the segment is reported as a [`FrozenRange`].

/// A compact frame fingerprint used for duplicate detection.
///
/// Use a simple 64-bit perceptual hash (or a direct pixel checksum) and store
/// it here.  Two frames are considered identical when their hashes match within
/// `threshold` Hamming distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameHash {
    /// 64-bit hash value (e.g. dHash, pHash, or CRC64 of luma).
    pub value: u64,
    /// Frame index this hash was computed from.
    pub frame_idx: usize,
}

impl FrameHash {
    /// Create a new `FrameHash`.
    #[must_use]
    pub fn new(frame_idx: usize, value: u64) -> Self {
        Self { value, frame_idx }
    }

    /// Compute the 64-bit CRC-like hash of a luma plane using a fast XOR-fold.
    ///
    /// This is **not** a perceptual hash — it is a content fingerprint suitable
    /// for detecting exact duplicate frames.
    #[must_use]
    pub fn from_luma(frame_idx: usize, luma: &[u8]) -> Self {
        let value = fast_hash(luma);
        Self { value, frame_idx }
    }

    /// Hamming distance (number of differing bits) between two hashes.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.value ^ other.value).count_ones()
    }
}

/// A range of consecutive frozen (duplicate) frames.
#[derive(Debug, Clone)]
pub struct FrozenRange {
    /// Index of the first frozen frame in the run.
    pub start_frame: usize,
    /// Index of the last frozen frame in the run (inclusive).
    pub end_frame: usize,
    /// Number of frames in this frozen segment.
    pub length: usize,
}

impl FrozenRange {
    /// Compute duration in seconds given a frame rate.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn duration_seconds(&self, fps: f32) -> f32 {
        self.length as f32 / fps.max(1.0)
    }
}

/// Frozen frame detector.
///
/// Compares consecutive [`FrameHash`] values and reports runs of identical
/// (or near-identical) frames that exceed the configured minimum run length.
pub struct FrozenFrameDetector {
    /// Maximum Hamming distance below which two hashes are considered identical.
    pub threshold: f32,
    /// Minimum run length (in frames) to qualify as a frozen segment.
    pub min_run_length: usize,
}

impl FrozenFrameDetector {
    /// Create a new detector.
    ///
    /// * `threshold` – maximum Hamming distance (0.0–64.0) for two hashes to
    ///   be considered equal.  Use 0.0 for exact-match-only detection.
    /// * `min_run_length` – minimum number of consecutive identical frames to
    ///   report as a frozen segment.
    #[must_use]
    pub fn new(threshold: f32, min_run_length: usize) -> Self {
        Self {
            threshold: threshold.clamp(0.0, 64.0),
            min_run_length: min_run_length.max(2),
        }
    }

    /// Create a detector with default parameters:
    /// * `threshold = 4.0` (4-bit Hamming tolerance)
    /// * `min_run_length = 3` frames
    #[must_use]
    pub fn default_params() -> Self {
        Self::new(4.0, 3)
    }

    /// Detect frozen frame segments in a sequence of frame hashes.
    ///
    /// `frames` must be ordered by frame index.
    ///
    /// Returns a list of [`FrozenRange`] segments.
    #[must_use]
    pub fn detect(&self, frames: &[FrameHash]) -> Vec<FrozenRange> {
        if frames.len() < 2 {
            return Vec::new();
        }

        let mut ranges = Vec::new();
        let mut run_start: Option<usize> = None;
        let mut run_len = 1usize;

        for i in 1..frames.len() {
            let dist = frames[i].hamming_distance(&frames[i - 1]);
            let identical = dist as f32 <= self.threshold;

            if identical {
                if run_start.is_none() {
                    run_start = Some(i - 1);
                }
                run_len += 1;
            } else {
                if let Some(start) = run_start.take() {
                    if run_len >= self.min_run_length {
                        ranges.push(FrozenRange {
                            start_frame: frames[start].frame_idx,
                            end_frame: frames[i - 1].frame_idx,
                            length: run_len,
                        });
                    }
                }
                run_len = 1;
            }
        }

        // Handle a frozen run that extends to the end
        if let Some(start) = run_start {
            if run_len >= self.min_run_length {
                let last = frames.len() - 1;
                ranges.push(FrozenRange {
                    start_frame: frames[start].frame_idx,
                    end_frame: frames[last].frame_idx,
                    length: run_len,
                });
            }
        }

        ranges
    }
}

impl Default for FrozenFrameDetector {
    fn default() -> Self {
        Self::default_params()
    }
}

/// Fast 64-bit XOR-fold hash for exact duplicate detection.
fn fast_hash(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for chunk in data.chunks(8) {
        let mut word = 0u64;
        for (i, &b) in chunk.iter().enumerate() {
            word |= (b as u64) << (i * 8);
        }
        h ^= word;
        h = h.wrapping_mul(0x0000_0100_0000_01B3); // FNV prime
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_frozen_on_empty() {
        let det = FrozenFrameDetector::default();
        assert!(det.detect(&[]).is_empty());
    }

    #[test]
    fn test_no_frozen_single_frame() {
        let det = FrozenFrameDetector::default();
        let frames = vec![FrameHash::new(0, 0xDEAD)];
        assert!(det.detect(&frames).is_empty());
    }

    #[test]
    fn test_three_identical_frames_detected() {
        let det = FrozenFrameDetector::new(0.0, 3);
        let hash_val = 0xABCD_1234u64;
        let frames: Vec<FrameHash> = (0..3).map(|i| FrameHash::new(i, hash_val)).collect();
        let ranges = det.detect(&frames);
        assert_eq!(ranges.len(), 1, "should detect one frozen range");
        assert_eq!(ranges[0].length, 3);
    }

    #[test]
    fn test_different_frames_no_frozen() {
        let det = FrozenFrameDetector::new(0.0, 3);
        let frames: Vec<FrameHash> = (0..5).map(|i| FrameHash::new(i, i as u64 * 0x1234)).collect();
        assert!(det.detect(&frames).is_empty());
    }

    #[test]
    fn test_frozen_range_duration() {
        let r = FrozenRange {
            start_frame: 0,
            end_frame: 29,
            length: 30,
        };
        assert!((r.duration_seconds(30.0) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_frame_hash_hamming_distance_identical() {
        let a = FrameHash::new(0, 0xFF);
        let b = FrameHash::new(1, 0xFF);
        assert_eq!(a.hamming_distance(&b), 0);
    }

    #[test]
    fn test_frame_hash_hamming_distance_one_bit() {
        let a = FrameHash::new(0, 0b0000);
        let b = FrameHash::new(1, 0b0001);
        assert_eq!(a.hamming_distance(&b), 1);
    }

    #[test]
    fn test_frame_hash_from_luma_deterministic() {
        let luma = vec![128u8; 64 * 64];
        let h1 = FrameHash::from_luma(0, &luma);
        let h2 = FrameHash::from_luma(0, &luma);
        assert_eq!(h1.value, h2.value);
    }

    #[test]
    fn test_frame_hash_from_luma_different_data() {
        let luma_a = vec![100u8; 64];
        let luma_b = vec![200u8; 64];
        let ha = FrameHash::from_luma(0, &luma_a);
        let hb = FrameHash::from_luma(0, &luma_b);
        assert_ne!(ha.value, hb.value);
    }

    #[test]
    fn test_min_run_length_enforced() {
        let det = FrozenFrameDetector::new(0.0, 5);
        // Only 3 identical frames → below min_run_length=5 → no detection
        let hash_val = 0x9999u64;
        let frames: Vec<FrameHash> = (0..3).map(|i| FrameHash::new(i, hash_val)).collect();
        assert!(det.detect(&frames).is_empty());
    }
}
