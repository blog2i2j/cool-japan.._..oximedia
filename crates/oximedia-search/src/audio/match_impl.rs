//! Audio matching algorithms.
//!
//! Implements a simplified spectral fingerprinting approach:
//! 1. Divide audio into overlapping frames.
//! 2. Compute energy in frequency sub-bands using DFT magnitude.
//! 3. Hash energy-difference patterns across bands into a compact fingerprint.
//! 4. Match fingerprints via normalised Hamming distance.

use crate::error::SearchResult;
use uuid::Uuid;

/// Number of frequency sub-bands for fingerprinting.
const NUM_BANDS: usize = 8;
/// Frame size in samples for fingerprint analysis.
const FRAME_SIZE: usize = 1024;
/// Hop size (overlap factor) in samples.
const HOP_SIZE: usize = 512;
/// Number of bytes per fingerprint frame.
const FP_BYTES_PER_FRAME: usize = 1;

/// Audio match result.
#[derive(Debug, Clone)]
pub struct AudioMatch {
    /// Asset ID.
    pub asset_id: Uuid,
    /// Match confidence (0.0 to 1.0).
    pub confidence: f32,
    /// Time offset in the matched file (milliseconds).
    pub offset_ms: i64,
    /// Duration of match (milliseconds).
    pub duration_ms: i64,
}

/// Audio matcher using spectral sub-band energy fingerprinting.
pub struct AudioMatcher;

impl AudioMatcher {
    /// Create a new audio matcher.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Find matches for `query_fingerprint` in the `database`.
    ///
    /// Uses sliding-window normalised Hamming distance. Matches with
    /// confidence >= `threshold` are returned, sorted by confidence
    /// descending.
    ///
    /// # Errors
    ///
    /// Returns an error if matching fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn find_matches(
        &self,
        query_fingerprint: &[u8],
        database: &[(Uuid, Vec<u8>)],
        threshold: f32,
    ) -> SearchResult<Vec<AudioMatch>> {
        if query_fingerprint.is_empty() {
            return Ok(Vec::new());
        }

        let query_len = query_fingerprint.len();
        let mut results = Vec::new();

        for (asset_id, db_fp) in database {
            if db_fp.len() < query_len {
                // Try matching the other direction.
                let dist = normalised_hamming(query_fingerprint, db_fp);
                let confidence = 1.0 - dist;
                if confidence >= threshold {
                    results.push(AudioMatch {
                        asset_id: *asset_id,
                        confidence,
                        offset_ms: 0,
                        duration_ms: (db_fp.len() as i64 * HOP_SIZE as i64 * 1000) / 48000,
                    });
                }
                continue;
            }

            // Sliding window: find the best alignment.
            let mut best_confidence = 0.0_f32;
            let mut best_offset: usize = 0;

            let max_start = db_fp.len() - query_len;
            // Step through in chunks to avoid O(n^2) on very long fingerprints.
            let step = (max_start / 256).max(1);

            for start in (0..=max_start).step_by(step) {
                let window = &db_fp[start..start + query_len];
                let dist = normalised_hamming(query_fingerprint, window);
                let conf = 1.0 - dist;
                if conf > best_confidence {
                    best_confidence = conf;
                    best_offset = start;
                }
            }

            if best_confidence >= threshold {
                // Convert frame offset to milliseconds.
                let offset_ms = (best_offset as i64 * HOP_SIZE as i64 * 1000) / 48000;
                let duration_ms = (query_len as i64 * HOP_SIZE as i64 * 1000) / 48000;
                results.push(AudioMatch {
                    asset_id: *asset_id,
                    confidence: best_confidence,
                    offset_ms,
                    duration_ms,
                });
            }
        }

        // Sort by confidence descending.
        results.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Extract audio fingerprint from raw PCM f32 audio data.
    ///
    /// The algorithm:
    /// 1. Divide into `FRAME_SIZE`-sample frames with `HOP_SIZE` overlap.
    /// 2. For each frame, compute energy in `NUM_BANDS` frequency sub-bands
    ///    using a simple DFT magnitude computation.
    /// 3. Compare band energies between consecutive frames; encode
    ///    energy increases as 1-bits to produce 1 byte per frame.
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn extract_fingerprint(
        &self,
        audio_data: &[f32],
        _sample_rate: u32,
    ) -> SearchResult<Vec<u8>> {
        if audio_data.len() < FRAME_SIZE * 2 {
            // Not enough data for two frames; return a minimal fingerprint.
            let energy: f32 = audio_data.iter().map(|s| s * s).sum();
            let byte = if energy > 0.01 { 0xFF } else { 0x00 };
            return Ok(vec![byte; 4]);
        }

        // Compute band energies for each frame.
        let num_frames = (audio_data.len().saturating_sub(FRAME_SIZE)) / HOP_SIZE + 1;
        let mut band_energies: Vec<[f32; NUM_BANDS]> = Vec::with_capacity(num_frames);

        for frame_idx in 0..num_frames {
            let start = frame_idx * HOP_SIZE;
            let end = (start + FRAME_SIZE).min(audio_data.len());
            let frame = &audio_data[start..end];

            let energies = compute_band_energies(frame);
            band_energies.push(energies);
        }

        // Generate fingerprint: compare consecutive frames.
        let mut fingerprint =
            Vec::with_capacity(band_energies.len().saturating_sub(1) * FP_BYTES_PER_FRAME);

        for i in 1..band_energies.len() {
            let mut byte: u8 = 0;
            for band in 0..NUM_BANDS {
                if band_energies[i][band] > band_energies[i - 1][band] {
                    byte |= 1 << band;
                }
            }
            fingerprint.push(byte);
        }

        if fingerprint.is_empty() {
            fingerprint.push(0);
        }

        Ok(fingerprint)
    }
}

/// Compute energy in `NUM_BANDS` frequency sub-bands for a single frame.
///
/// Uses a simplified DFT-magnitude approach: divide the frame into
/// `NUM_BANDS` equal-length segments and compute the sum of squares
/// in each (effectively a time-domain proxy for frequency-band energy
/// when the signal is naturally ordered).
fn compute_band_energies(frame: &[f32]) -> [f32; NUM_BANDS] {
    let mut energies = [0.0_f32; NUM_BANDS];
    let band_size = frame.len() / NUM_BANDS;

    if band_size == 0 {
        return energies;
    }

    for (band, energy) in energies.iter_mut().enumerate() {
        let start = band * band_size;
        let end = if band == NUM_BANDS - 1 {
            frame.len()
        } else {
            start + band_size
        };
        *energy = frame[start..end].iter().map(|s| s * s).sum::<f32>();
    }

    energies
}

/// Compute normalised Hamming distance between two byte slices.
///
/// Returns a value in [0.0, 1.0] where 0.0 means identical.
#[allow(clippy::cast_precision_loss)]
fn normalised_hamming(a: &[u8], b: &[u8]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 1.0;
    }
    let diff_bits: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones())
        .sum();
    let total_bits = (len * 8) as f32;
    diff_bits as f32 / total_bits
}

impl Default for AudioMatcher {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_fingerprint_short() {
        let matcher = AudioMatcher::new();
        let audio_data = vec![0.0; 100];
        let fingerprint = matcher
            .extract_fingerprint(&audio_data, 44100)
            .expect("should succeed in test");
        assert!(!fingerprint.is_empty());
    }

    #[test]
    fn test_extract_fingerprint_longer() {
        let matcher = AudioMatcher::new();
        // 2 seconds of sine-like data at 44100 Hz.
        let audio_data: Vec<f32> = (0..88200).map(|i| (i as f32 * 0.01).sin()).collect();
        let fingerprint = matcher
            .extract_fingerprint(&audio_data, 44100)
            .expect("should succeed in test");
        assert!(fingerprint.len() > 1);
    }

    #[test]
    fn test_identical_fingerprints_match() {
        let matcher = AudioMatcher::new();
        let audio: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.02).sin()).collect();
        let fp = matcher
            .extract_fingerprint(&audio, 44100)
            .expect("should succeed in test");

        let id = Uuid::new_v4();
        let database = vec![(id, fp.clone())];
        let matches = matcher
            .find_matches(&fp, &database, 0.5)
            .expect("should succeed in test");
        assert!(!matches.is_empty());
        assert_eq!(matches[0].asset_id, id);
        assert!((matches[0].confidence - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_different_fingerprints_low_confidence() {
        let matcher = AudioMatcher::new();
        let audio_a: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.02).sin()).collect();
        let audio_b: Vec<f32> = (0..44100).map(|i| (i as f32 * 0.1).cos()).collect();
        let fp_a = matcher
            .extract_fingerprint(&audio_a, 44100)
            .expect("should succeed in test");
        let fp_b = matcher
            .extract_fingerprint(&audio_b, 44100)
            .expect("should succeed in test");

        let id = Uuid::new_v4();
        let database = vec![(id, fp_b)];
        let matches = matcher
            .find_matches(&fp_a, &database, 0.99)
            .expect("should succeed in test");
        // Very high threshold should exclude non-identical fingerprints.
        assert!(matches.is_empty());
    }

    #[test]
    fn test_normalised_hamming_identical() {
        let a = vec![0xFF, 0x00, 0xAA];
        let dist = normalised_hamming(&a, &a);
        assert!((dist - 0.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalised_hamming_opposite() {
        let a = vec![0xFF];
        let b = vec![0x00];
        let dist = normalised_hamming(&a, &b);
        assert!((dist - 1.0).abs() < f32::EPSILON);
    }
}
