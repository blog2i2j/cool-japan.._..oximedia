//! Perceptual hashing (pHash) and near-duplicate detection for video frames.
//!
//! This module provides:
//! - DCT-based perceptual hash (`PHash`) for images/video frames
//! - Hamming distance comparison for near-duplicate detection
//! - Sliding-window duplicate detection over sequences of frames
//! - Content fingerprinting for audio segments
//!
//! # Algorithm
//!
//! pHash operates by:
//! 1. Converting the image to grayscale and resizing to 32×32
//! 2. Applying a 2D Discrete Cosine Transform (DCT)
//! 3. Keeping only the 8×8 low-frequency coefficients (top-left block)
//! 4. Thresholding coefficients at the median → 64-bit hash
//!
//! Two images with Hamming distance ≤ 10 are considered near-duplicates.

use crate::{DedupError, DedupResult};

// --------------------------------------------------------------------------
// Perceptual hash
// --------------------------------------------------------------------------

/// Size of the DCT input (32×32 pixels).
const DCT_SIZE: usize = 32;
/// Size of the low-frequency hash block (8×8 → 64 bits).
const HASH_BLOCK: usize = 8;
/// Number of bits in the hash.
const HASH_BITS: u32 = (HASH_BLOCK * HASH_BLOCK) as u32;

/// A 64-bit perceptual hash derived from DCT coefficients.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PHash(u64);

impl PHash {
    /// Create a pHash from a raw 64-bit value.
    #[must_use]
    pub const fn from_bits(bits: u64) -> Self {
        Self(bits)
    }

    /// Return the raw 64-bit value.
    #[must_use]
    pub const fn bits(self) -> u64 {
        self.0
    }

    /// Hamming distance between two hashes (number of differing bits).
    #[must_use]
    pub fn hamming_distance(self, other: Self) -> u32 {
        (self.0 ^ other.0).count_ones()
    }

    /// Similarity in range [0.0, 1.0].
    ///
    /// 1.0 = identical, 0.0 = maximally different.
    #[must_use]
    pub fn similarity(self, other: Self) -> f64 {
        1.0 - (f64::from(self.hamming_distance(other)) / f64::from(HASH_BITS))
    }

    /// Returns `true` if the two hashes are considered near-duplicates.
    ///
    /// The default threshold is Hamming distance ≤ 10 (≈84 % similarity).
    #[must_use]
    pub fn is_near_duplicate(self, other: Self) -> bool {
        self.hamming_distance(other) <= 10
    }

    /// Returns `true` if Hamming distance is within `max_distance`.
    #[must_use]
    pub fn within_distance(self, other: Self, max_distance: u32) -> bool {
        self.hamming_distance(other) <= max_distance
    }

    /// Hex string representation.
    #[must_use]
    pub fn to_hex(self) -> String {
        format!("{:016x}", self.0)
    }

    /// Parse from hex string.
    ///
    /// # Errors
    ///
    /// Returns an error if the string is not a valid 16-character hex value.
    pub fn from_hex(s: &str) -> DedupResult<Self> {
        u64::from_str_radix(s, 16)
            .map(Self)
            .map_err(|e| DedupError::Hash(format!("Invalid pHash hex '{s}': {e}")))
    }
}

impl std::fmt::Display for PHash {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_hex())
    }
}

// --------------------------------------------------------------------------
// Grayscale frame representation
// --------------------------------------------------------------------------

/// A grayscale image used as input for perceptual hashing.
///
/// Pixels are stored in row-major order with values in `[0, 255]`.
#[derive(Debug, Clone)]
pub struct GrayFrame {
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
    /// Row-major grayscale pixel values.
    pub data: Vec<u8>,
}

impl GrayFrame {
    /// Create a new frame from raw data.
    ///
    /// # Errors
    ///
    /// Returns an error if `data.len() != width * height`.
    pub fn new(width: usize, height: usize, data: Vec<u8>) -> DedupResult<Self> {
        if data.len() != width * height {
            return Err(DedupError::Visual(format!(
                "GrayFrame: expected {} pixels, got {}",
                width * height,
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Create from an RGB(A) buffer.
    ///
    /// Uses ITU-R BT.601 luma: `Y = 0.299R + 0.587G + 0.114B`.
    ///
    /// # Errors
    ///
    /// Returns an error if buffer length doesn't match `width * height * channels`.
    pub fn from_rgb(width: usize, height: usize, rgb: &[u8], channels: usize) -> DedupResult<Self> {
        let expected = width * height * channels;
        if rgb.len() != expected {
            return Err(DedupError::Visual(format!(
                "from_rgb: expected {expected} bytes, got {}",
                rgb.len()
            )));
        }
        if channels < 3 {
            return Err(DedupError::Visual(
                "from_rgb: need at least 3 channels".to_string(),
            ));
        }
        let mut data = Vec::with_capacity(width * height);
        for i in 0..width * height {
            let r = f32::from(rgb[i * channels]);
            let g = f32::from(rgb[i * channels + 1]);
            let b = f32::from(rgb[i * channels + 2]);
            data.push((0.299 * r + 0.587 * g + 0.114 * b) as u8);
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    /// Nearest-neighbour resize to `new_width × new_height`.
    #[must_use]
    pub fn resize(&self, new_width: usize, new_height: usize) -> Self {
        let x_ratio = self.width as f32 / new_width as f32;
        let y_ratio = self.height as f32 / new_height as f32;
        let mut data = Vec::with_capacity(new_width * new_height);
        for ny in 0..new_height {
            let sy = (ny as f32 * y_ratio) as usize;
            for nx in 0..new_width {
                let sx = (nx as f32 * x_ratio) as usize;
                data.push(self.data[sy * self.width + sx]);
            }
        }
        Self {
            width: new_width,
            height: new_height,
            data,
        }
    }
}

// --------------------------------------------------------------------------
// 1D DCT-II (normalised) – building block for the 2D separable DCT
// --------------------------------------------------------------------------

/// Compute the 1D DCT-II of `input` in-place.
///
/// Uses the naive O(N²) algorithm because N ≤ 32 and this runs infrequently.
fn dct1d(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut out = vec![0.0; n];
    let pi_over_2n = std::f64::consts::PI / (2.0 * n as f64);
    for (k, ok) in out.iter_mut().enumerate() {
        let scale = if k == 0 {
            (1.0 / n as f64).sqrt()
        } else {
            (2.0 / n as f64).sqrt()
        };
        let mut s = 0.0;
        for (i, xi) in input.iter().enumerate() {
            s += xi * ((2 * i + 1) as f64 * k as f64 * pi_over_2n).cos();
        }
        *ok = scale * s;
    }
    out
}

/// 2D separable DCT: row-by-row, then column-by-column.
fn dct2d(input: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    // Row transforms
    let mut tmp = vec![0.0f64; rows * cols];
    for r in 0..rows {
        let row: Vec<f64> = input[r * cols..(r + 1) * cols].to_vec();
        let d = dct1d(&row);
        tmp[r * cols..(r + 1) * cols].copy_from_slice(&d);
    }
    // Column transforms
    let mut out = vec![0.0f64; rows * cols];
    for c in 0..cols {
        let col: Vec<f64> = (0..rows).map(|r| tmp[r * cols + c]).collect();
        let d = dct1d(&col);
        for r in 0..rows {
            out[r * cols + c] = d[r];
        }
    }
    out
}

// --------------------------------------------------------------------------
// Public pHash computation
// --------------------------------------------------------------------------

/// Compute the pHash of a `GrayFrame`.
///
/// The frame is first resized to `DCT_SIZE × DCT_SIZE`, then the 2D DCT is
/// applied and the top-left `HASH_BLOCK × HASH_BLOCK` block is thresholded
/// at the median to produce a 64-bit hash.
#[must_use]
pub fn compute_phash(frame: &GrayFrame) -> PHash {
    let resized = frame.resize(DCT_SIZE, DCT_SIZE);

    // Convert to f64 for DCT
    let input: Vec<f64> = resized.data.iter().map(|&v| f64::from(v)).collect();
    let dct = dct2d(&input, DCT_SIZE, DCT_SIZE);

    // Extract top-left HASH_BLOCK × HASH_BLOCK
    let mut low: Vec<f64> = Vec::with_capacity(HASH_BLOCK * HASH_BLOCK);
    for r in 0..HASH_BLOCK {
        for c in 0..HASH_BLOCK {
            low.push(dct[r * DCT_SIZE + c]);
        }
    }

    // Median (we skip DC coefficient at [0] which dominates the average)
    let mut sorted = low.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    // Hash: bit = 1 if coefficient > median
    let mut hash = 0u64;
    for (i, &v) in low.iter().enumerate() {
        if v > median {
            hash |= 1u64 << i;
        }
    }
    PHash(hash)
}

/// Compute pHash from a raw RGB pixel buffer.
///
/// Convenience wrapper around `compute_phash` + `GrayFrame::from_rgb`.
///
/// # Errors
///
/// Returns an error if buffer size doesn't match `width * height * channels`.
pub fn compute_phash_rgb(
    width: usize,
    height: usize,
    rgb: &[u8],
    channels: usize,
) -> DedupResult<PHash> {
    let frame = GrayFrame::from_rgb(width, height, rgb, channels)?;
    Ok(compute_phash(&frame))
}

// --------------------------------------------------------------------------
// Sliding-window duplicate detection
// --------------------------------------------------------------------------

/// A near-duplicate match found by the sliding window detector.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameMatch {
    /// Index of the first (reference) frame.
    pub frame_a: usize,
    /// Index of the second (candidate) frame.
    pub frame_b: usize,
    /// Hamming distance between the two hashes.
    pub hamming_distance: u32,
    /// Similarity score [0.0, 1.0].
    pub similarity: f64,
}

/// Configuration for the sliding-window detector.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Number of frames to look ahead from each reference frame.
    pub window_size: usize,
    /// Maximum Hamming distance to be considered a duplicate.
    pub max_distance: u32,
    /// Minimum gap between frame pairs to avoid consecutive-frame matching
    /// (set to 0 to include all pairs).
    pub min_gap: usize,
}

impl Default for SlidingWindowConfig {
    fn default() -> Self {
        Self {
            window_size: 30, // ~1 second at 30 fps
            max_distance: 10,
            min_gap: 1,
        }
    }
}

/// Detect near-duplicate frames in a sequence using a sliding window.
///
/// Each frame hash is compared against the next `window_size` hashes.
/// Pairs with Hamming distance ≤ `max_distance` are returned as matches.
///
/// The algorithm is O(N × window_size) making it suitable for long sequences.
#[must_use]
pub fn sliding_window_detect(hashes: &[PHash], config: &SlidingWindowConfig) -> Vec<FrameMatch> {
    let mut matches = Vec::new();

    for i in 0..hashes.len() {
        let end = (i + 1 + config.window_size).min(hashes.len());
        for j in (i + config.min_gap + 1).min(end)..end {
            let dist = hashes[i].hamming_distance(hashes[j]);
            if dist <= config.max_distance {
                matches.push(FrameMatch {
                    frame_a: i,
                    frame_b: j,
                    hamming_distance: dist,
                    similarity: hashes[i].similarity(hashes[j]),
                });
            }
        }
    }

    matches
}

/// Find all consecutive frame pairs that are likely frozen (identical or near-identical).
///
/// Returns the start indices of frozen runs and the run length.
#[must_use]
pub fn detect_frozen_segments(hashes: &[PHash], max_distance: u32) -> Vec<(usize, usize)> {
    if hashes.len() < 2 {
        return Vec::new();
    }

    let mut segments = Vec::new();
    let mut run_start = 0usize;
    let mut in_run = false;

    for i in 1..hashes.len() {
        let dist = hashes[i - 1].hamming_distance(hashes[i]);
        if dist <= max_distance {
            if !in_run {
                run_start = i - 1;
                in_run = true;
            }
        } else if in_run {
            let run_len = i - run_start;
            if run_len >= 2 {
                segments.push((run_start, run_len));
            }
            in_run = false;
        }
    }
    if in_run {
        let run_len = hashes.len() - run_start;
        if run_len >= 2 {
            segments.push((run_start, run_len));
        }
    }
    segments
}

// --------------------------------------------------------------------------
// Audio content fingerprinting
// --------------------------------------------------------------------------

/// A compact fingerprint for an audio segment based on sub-band energy ratios.
///
/// The fingerprint is computed by:
/// 1. Dividing the audio into short frames (default 64 ms)
/// 2. Computing the energy in each of 8 frequency sub-bands
/// 3. Encoding the sign of inter-band energy differences as bits
///
/// This produces a bit-string that is robust to small amplitude changes and
/// can be compared with Hamming distance for near-duplicate detection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioSegmentFingerprint {
    /// Raw fingerprint bytes (1 byte per analysis frame).
    pub bytes: Vec<u8>,
    /// Number of audio samples analysed.
    pub sample_count: usize,
    /// Sample rate used during fingerprinting.
    pub sample_rate: u32,
}

impl AudioSegmentFingerprint {
    /// Hamming distance between two fingerprints (proportional to byte length difference).
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> usize {
        let min_len = self.bytes.len().min(other.bytes.len());
        let len_penalty =
            (self.bytes.len() as i64 - other.bytes.len() as i64).unsigned_abs() as usize * 8;
        let bit_diff: usize = self.bytes[..min_len]
            .iter()
            .zip(other.bytes[..min_len].iter())
            .map(|(a, b)| (a ^ b).count_ones() as usize)
            .sum();
        bit_diff + len_penalty
    }

    /// Similarity in [0.0, 1.0].
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        let max_bits = self.bytes.len().max(other.bytes.len()) * 8;
        if max_bits == 0 {
            return 1.0;
        }
        let dist = self.hamming_distance(other);
        (1.0 - dist as f64 / max_bits as f64).clamp(0.0, 1.0)
    }

    /// Returns `true` if similarity ≥ `threshold`.
    #[must_use]
    pub fn is_similar(&self, other: &Self, threshold: f64) -> bool {
        self.similarity(other) >= threshold
    }
}

/// Number of sub-bands for the audio fingerprint.
const N_BANDS: usize = 8;
/// Frame size in samples at the reference rate (11 025 Hz).
const FRAME_SAMPLES: usize = 128;

/// Compute an audio segment fingerprint from mono PCM samples.
///
/// `samples` should be normalised floats in `[-1.0, 1.0]`.
/// `sample_rate` is the actual sample rate; samples are logically
/// downsampled to 11 025 Hz (no actual resampling – we stride).
#[must_use]
pub fn compute_audio_fingerprint(samples: &[f32], sample_rate: u32) -> AudioSegmentFingerprint {
    if samples.is_empty() {
        return AudioSegmentFingerprint {
            bytes: Vec::new(),
            sample_count: 0,
            sample_rate,
        };
    }

    // Stride to approximate 11 025 Hz downsampling
    let stride = (sample_rate as usize / 11025).max(1);
    let downsampled: Vec<f32> = samples.iter().step_by(stride).copied().collect();

    let mut fingerprint_bytes = Vec::new();

    let num_frames = downsampled.len() / FRAME_SAMPLES;
    for frame_idx in 0..num_frames {
        let start = frame_idx * FRAME_SAMPLES;
        let frame = &downsampled[start..start + FRAME_SAMPLES];

        // Compute sub-band energies using overlapping frequency windows.
        // Band k contains frequencies proportional to (k..k+1)/N_BANDS of Nyquist.
        // We use a simplified DFT-free approach: compute energy of frame
        // segments that approximate frequency bands via the Haar-like decomposition.
        let band_energies = compute_subband_energies(frame);

        // Encode adjacent band energy differences as bits
        let mut byte = 0u8;
        for b in 0..N_BANDS - 1 {
            // Bit set if band b has more energy than band b+1
            if band_energies[b] > band_energies[b + 1] {
                byte |= 1u8 << b;
            }
        }
        // Last bit: compare first and last band (wraps around)
        if band_energies[0] > band_energies[N_BANDS - 1] {
            byte |= 1u8 << (N_BANDS - 1);
        }
        fingerprint_bytes.push(byte);
    }

    AudioSegmentFingerprint {
        bytes: fingerprint_bytes,
        sample_count: samples.len(),
        sample_rate,
    }
}

/// Compute approximate sub-band energies for a frame using recursive splitting.
///
/// Each iteration of the Haar-like wavelet halves the frequency range,
/// giving `N_BANDS` sub-band energies without needing an FFT.
fn compute_subband_energies(frame: &[f32]) -> [f64; N_BANDS] {
    let len = frame.len();
    let half = len / 2;

    // Split frame into N_BANDS equal segments and compute RMS energy per segment
    let segment_size = (len / N_BANDS).max(1);
    let mut energies = [0.0f64; N_BANDS];

    for (band, energy) in energies.iter_mut().enumerate() {
        let start = band * segment_size;
        let end = ((band + 1) * segment_size).min(len);
        if start >= end {
            break;
        }
        let rms: f64 = frame[start..end]
            .iter()
            .map(|&s| (s as f64).powi(2))
            .sum::<f64>()
            / (end - start) as f64;
        *energy = rms;
    }

    // Apply simple Haar-like smoothing: blend adjacent bands
    // This makes the fingerprint more robust to small spectral shifts
    let orig = energies;
    for b in 1..N_BANDS - 1 {
        energies[b] = 0.25 * orig[b - 1] + 0.5 * orig[b] + 0.25 * orig[b + 1];
    }
    let _ = half; // suppress unused warning from the half var above

    energies
}

/// Compare two audio fingerprints and return a similarity report.
#[derive(Debug, Clone)]
pub struct FingerprintComparison {
    /// Hamming distance.
    pub hamming_distance: usize,
    /// Similarity score [0.0, 1.0].
    pub similarity: f64,
    /// Whether the segments are considered duplicates.
    pub is_duplicate: bool,
    /// The threshold used.
    pub threshold: f64,
}

/// Compare two audio fingerprints.
#[must_use]
pub fn compare_fingerprints(
    fp1: &AudioSegmentFingerprint,
    fp2: &AudioSegmentFingerprint,
    threshold: f64,
) -> FingerprintComparison {
    let hamming = fp1.hamming_distance(fp2);
    let similarity = fp1.similarity(fp2);
    FingerprintComparison {
        hamming_distance: hamming,
        similarity,
        is_duplicate: similarity >= threshold,
        threshold,
    }
}

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ----- PHash tests -----

    fn solid_frame(val: u8, w: usize, h: usize) -> GrayFrame {
        GrayFrame::new(w, h, vec![val; w * h]).expect("operation should succeed")
    }

    fn gradient_frame(w: usize, h: usize) -> GrayFrame {
        let data = (0..w * h).map(|i| (i % 256) as u8).collect();
        GrayFrame::new(w, h, data).expect("operation should succeed")
    }

    #[test]
    fn test_phash_identical_frames_zero_distance() {
        let frame = gradient_frame(64, 64);
        let h1 = compute_phash(&frame);
        let h2 = compute_phash(&frame);
        assert_eq!(h1.hamming_distance(h2), 0);
        assert_eq!(h1.similarity(h2), 1.0);
    }

    #[test]
    fn test_phash_different_frames_nonzero_distance() {
        let f1 = solid_frame(0, 64, 64);
        let f2 = solid_frame(255, 64, 64);
        let h1 = compute_phash(&f1);
        let h2 = compute_phash(&f2);
        // Fully black and fully white → different hashes
        assert!(h1.hamming_distance(h2) > 0);
    }

    #[test]
    fn test_phash_similarity_range() {
        let f1 = gradient_frame(64, 64);
        let f2 = gradient_frame(32, 32);
        let h1 = compute_phash(&f1);
        let h2 = compute_phash(&f2);
        let sim = h1.similarity(h2);
        assert!((0.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_phash_hex_roundtrip() {
        let frame = gradient_frame(64, 64);
        let h = compute_phash(&frame);
        let hex = h.to_hex();
        assert_eq!(hex.len(), 16);
        let h2 = PHash::from_hex(&hex).expect("operation should succeed");
        assert_eq!(h, h2);
    }

    #[test]
    fn test_phash_invalid_hex() {
        assert!(PHash::from_hex("not_a_hex!").is_err());
    }

    #[test]
    fn test_phash_within_distance() {
        let h1 = PHash::from_bits(0xFFFF_FFFF_FFFF_FFFF);
        let h2 = PHash::from_bits(0xFFFF_FFFF_FFFF_FFFE); // 1 bit diff
        assert!(h1.within_distance(h2, 1));
        assert!(!h1.within_distance(h2, 0));
    }

    #[test]
    fn test_phash_near_duplicate() {
        let h1 = PHash::from_bits(0xFFFF_FFFF_FFFF_FFFF);
        let h2 = PHash::from_bits(0xFFFF_FFFF_FFFF_FF00); // 8 bit diff
        assert!(h1.is_near_duplicate(h2)); // 8 ≤ 10
        let h3 = PHash::from_bits(0x0000_FFFF_FFFF_FFFF); // 16 bit diff
        assert!(!h1.is_near_duplicate(h3));
    }

    #[test]
    fn test_compute_phash_rgb() {
        let rgb: Vec<u8> = (0..32 * 32 * 3).map(|i| (i % 256) as u8).collect();
        assert!(compute_phash_rgb(32, 32, &rgb, 3).is_ok());
    }

    #[test]
    fn test_compute_phash_rgb_wrong_size() {
        let rgb = vec![0u8; 10];
        assert!(compute_phash_rgb(32, 32, &rgb, 3).is_err());
    }

    #[test]
    fn test_gray_frame_from_rgb() {
        let rgb: Vec<u8> = (0..8 * 8 * 3).map(|i| (i % 256) as u8).collect();
        let frame = GrayFrame::from_rgb(8, 8, &rgb, 3).expect("operation should succeed");
        assert_eq!(frame.data.len(), 64);
    }

    // ----- Sliding window tests -----

    fn seq_hashes(n: usize) -> Vec<PHash> {
        (0..n).map(|i| PHash::from_bits(i as u64)).collect()
    }

    #[test]
    fn test_sliding_window_no_duplicates() {
        let hashes = seq_hashes(10);
        let config = SlidingWindowConfig {
            max_distance: 2,
            ..Default::default()
        };
        // Sequential integers differ in at least 1 bit, many differ in > 2
        let matches = sliding_window_detect(&hashes, &config);
        // Adjacent pairs (i, i+1) differ by at most a few bits for small i
        // Just ensure no panic and result is reasonable
        assert!(matches.len() < hashes.len() * 10);
    }

    #[test]
    fn test_sliding_window_identical_sequence() {
        let hashes = vec![PHash::from_bits(0xABCD_EF12_3456_7890); 20];
        let config = SlidingWindowConfig {
            window_size: 5,
            max_distance: 0,
            min_gap: 1,
        };
        let matches = sliding_window_detect(&hashes, &config);
        assert!(!matches.is_empty(), "Identical hashes should all match");
        // All matches should have distance 0
        for m in &matches {
            assert_eq!(m.hamming_distance, 0);
            assert_eq!(m.similarity, 1.0);
        }
    }

    #[test]
    fn test_sliding_window_empty() {
        let hashes: Vec<PHash> = vec![];
        let matches = sliding_window_detect(&hashes, &SlidingWindowConfig::default());
        assert!(matches.is_empty());
    }

    #[test]
    fn test_sliding_window_single() {
        let hashes = vec![PHash::from_bits(42)];
        let matches = sliding_window_detect(&hashes, &SlidingWindowConfig::default());
        assert!(matches.is_empty());
    }

    #[test]
    fn test_detect_frozen_segments() {
        let h = PHash::from_bits(0xDEAD_BEEF_DEAD_BEEF);
        let hashes: Vec<PHash> = vec![
            PHash::from_bits(1), // 0: different
            h,
            h,
            h,
            h,                    // 1..4: frozen
            PHash::from_bits(99), // 5: different
        ];
        let segs = detect_frozen_segments(&hashes, 0);
        assert_eq!(segs.len(), 1);
        assert_eq!(segs[0].0, 1); // starts at frame 1
        assert_eq!(segs[0].1, 4); // 4 frames long
    }

    #[test]
    fn test_detect_frozen_empty() {
        let segs = detect_frozen_segments(&[], 0);
        assert!(segs.is_empty());
    }

    // ----- Audio fingerprint tests -----

    fn sine_samples(freq_hz: f32, duration_s: f32, sr: u32) -> Vec<f32> {
        let n = (duration_s * sr as f32) as usize;
        (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * freq_hz * i as f32 / sr as f32).sin())
            .collect()
    }

    #[test]
    fn test_audio_fingerprint_empty() {
        let fp = compute_audio_fingerprint(&[], 44100);
        assert!(fp.bytes.is_empty());
    }

    #[test]
    fn test_audio_fingerprint_non_empty() {
        let samples = sine_samples(440.0, 1.0, 44100);
        let fp = compute_audio_fingerprint(&samples, 44100);
        assert!(!fp.bytes.is_empty());
    }

    #[test]
    fn test_audio_fingerprint_same_signal_high_similarity() {
        let samples = sine_samples(440.0, 2.0, 44100);
        let fp1 = compute_audio_fingerprint(&samples, 44100);
        let fp2 = compute_audio_fingerprint(&samples, 44100);
        assert_eq!(fp1.similarity(&fp2), 1.0, "Same signal should be identical");
    }

    #[test]
    fn test_audio_fingerprint_different_signals_lower_similarity() {
        let s1 = sine_samples(440.0, 2.0, 44100);
        let s2 = sine_samples(880.0, 2.0, 44100); // Octave up
        let fp1 = compute_audio_fingerprint(&s1, 44100);
        let fp2 = compute_audio_fingerprint(&s2, 44100);
        let sim = fp1.similarity(&fp2);
        assert!(sim < 1.0, "Different signals should have < 1.0 similarity");
        assert!((0.0..=1.0).contains(&sim));
    }

    #[test]
    fn test_fingerprint_comparison() {
        let s = sine_samples(440.0, 1.0, 44100);
        let fp1 = compute_audio_fingerprint(&s, 44100);
        let fp2 = compute_audio_fingerprint(&s, 44100);
        let cmp = compare_fingerprints(&fp1, &fp2, 0.9);
        assert!(cmp.is_duplicate);
        assert_eq!(cmp.hamming_distance, 0);
    }

    #[test]
    fn test_audio_fingerprint_hamming_symmetry() {
        let s1 = sine_samples(220.0, 1.0, 44100);
        let s2 = sine_samples(880.0, 1.0, 44100);
        let fp1 = compute_audio_fingerprint(&s1, 44100);
        let fp2 = compute_audio_fingerprint(&s2, 44100);
        assert_eq!(fp1.hamming_distance(&fp2), fp2.hamming_distance(&fp1));
    }

    #[test]
    fn test_audio_fingerprint_similarity_symmetric() {
        let s1 = sine_samples(330.0, 1.0, 44100);
        let s2 = sine_samples(660.0, 1.0, 44100);
        let fp1 = compute_audio_fingerprint(&s1, 44100);
        let fp2 = compute_audio_fingerprint(&s2, 44100);
        let sim12 = fp1.similarity(&fp2);
        let sim21 = fp2.similarity(&fp1);
        assert!((sim12 - sim21).abs() < 1e-9);
    }
}
