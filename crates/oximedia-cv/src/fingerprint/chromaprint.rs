//! Audio fingerprinting using Chromaprint-like algorithm.
//!
//! This module implements an audio fingerprinting system similar to Chromaprint/AcoustID:
//!
//! - **FFT-based analysis**: Spectral feature extraction using FFT
//! - **Robust fingerprints**: Resistant to noise, compression, and pitch changes
//! - **Fast matching**: Efficient lookup in large databases
//! - **Offset detection**: Identifies where in a track a sample comes from
//! - **Quantization**: Compact binary fingerprints

use crate::error::{CvError, CvResult};
use rayon::prelude::*;
use rustfft::{num_complex::Complex, FftPlanner};

/// Audio fingerprint configuration.
#[derive(Debug, Clone)]
pub struct AudioFingerprintConfig {
    /// Sample rate (Hz).
    pub sample_rate: u32,
    /// FFT frame size (samples).
    pub frame_size: usize,
    /// Hop size between frames (samples).
    pub hop_size: usize,
    /// Number of frequency bands.
    pub num_bands: usize,
    /// Minimum frequency (Hz).
    pub min_freq: f32,
    /// Maximum frequency (Hz).
    pub max_freq: f32,
}

impl Default for AudioFingerprintConfig {
    fn default() -> Self {
        Self {
            sample_rate: 11025,
            frame_size: 4096,
            hop_size: 1024,
            num_bands: 12,
            min_freq: 300.0,
            max_freq: 5000.0,
        }
    }
}

/// Audio fingerprint.
#[derive(Debug, Clone)]
pub struct AudioFingerprint {
    /// Raw fingerprint data (32-bit integers).
    pub fingerprint: Vec<u32>,
    /// Duration in seconds.
    pub duration: f64,
    /// Sample rate used.
    pub sample_rate: u32,
}

impl AudioFingerprint {
    /// Creates a new audio fingerprint.
    #[must_use]
    pub fn new(fingerprint: Vec<u32>, duration: f64, sample_rate: u32) -> Self {
        Self {
            fingerprint,
            duration,
            sample_rate,
        }
    }

    /// Returns the size in bytes.
    #[must_use]
    pub fn size_bytes(&self) -> usize {
        self.fingerprint.len() * std::mem::size_of::<u32>()
    }

    /// Compares with another fingerprint.
    ///
    /// Returns similarity score in [0.0, 1.0].
    #[must_use]
    pub fn compare(&self, other: &Self) -> f64 {
        compare_fingerprints(&self.fingerprint, &other.fingerprint)
    }
}

/// Extracts audio fingerprint from samples.
///
/// # Arguments
///
/// * `samples` - Audio samples (mono, f32, normalized to [-1.0, 1.0])
/// * `sample_rate` - Sample rate in Hz
/// * `frame_size` - FFT frame size (must be power of 2)
/// * `hop_size` - Hop size between frames
///
/// # Errors
///
/// Returns an error if fingerprinting fails.
pub fn extract_fingerprint(
    samples: &[f32],
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
) -> CvResult<Vec<u32>> {
    if samples.is_empty() {
        return Err(CvError::invalid_parameter("samples", "empty"));
    }

    if frame_size == 0 || (frame_size & (frame_size - 1)) != 0 {
        return Err(CvError::invalid_parameter(
            "frame_size",
            format!("{frame_size} (must be power of 2)"),
        ));
    }

    if hop_size == 0 || hop_size > frame_size {
        return Err(CvError::invalid_parameter(
            "hop_size",
            format!("{hop_size} (must be > 0 and <= frame_size)"),
        ));
    }

    let config = AudioFingerprintConfig {
        sample_rate,
        frame_size,
        hop_size,
        ..Default::default()
    };

    // Compute spectrogram
    let spectrogram = compute_spectrogram(samples, &config)?;

    // Extract chromagram features
    let chroma = compute_chromagram(&spectrogram, &config)?;

    // Quantize to binary fingerprint
    let fingerprint = quantize_features(&chroma);

    Ok(fingerprint)
}

/// Computes spectrogram using FFT.
fn compute_spectrogram(
    samples: &[f32],
    config: &AudioFingerprintConfig,
) -> CvResult<Vec<Vec<f32>>> {
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(config.frame_size);

    let num_frames = (samples.len() - config.frame_size) / config.hop_size + 1;
    let mut spectrogram = Vec::with_capacity(num_frames);

    // Hanning window
    let window = create_hanning_window(config.frame_size);

    for frame_idx in 0..num_frames {
        let start = frame_idx * config.hop_size;
        let end = start + config.frame_size;

        if end > samples.len() {
            break;
        }

        // Apply window and create complex buffer
        let mut buffer: Vec<Complex<f32>> = samples[start..end]
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| Complex::new(s * w, 0.0))
            .collect();

        // Perform FFT
        fft.process(&mut buffer);

        // Compute magnitude spectrum
        let magnitudes: Vec<f32> = buffer[..config.frame_size / 2]
            .iter()
            .map(|c| c.norm())
            .collect();

        spectrogram.push(magnitudes);
    }

    Ok(spectrogram)
}

/// Creates a Hanning window.
fn create_hanning_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            let phase = 2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32;
            0.5 * (1.0 - phase.cos())
        })
        .collect()
}

/// Computes chromagram (pitch class profile).
fn compute_chromagram(
    spectrogram: &[Vec<f32>],
    config: &AudioFingerprintConfig,
) -> CvResult<Vec<Vec<f32>>> {
    let mut chroma = Vec::with_capacity(spectrogram.len());

    // Create frequency to band mapping
    let freq_to_band = create_frequency_bands(config);

    for frame in spectrogram {
        let mut chroma_frame = vec![0.0; config.num_bands];

        for (bin_idx, &magnitude) in frame.iter().enumerate() {
            let freq = bin_idx as f32 * config.sample_rate as f32 / config.frame_size as f32;

            if freq < config.min_freq || freq > config.max_freq {
                continue;
            }

            // Find corresponding chroma band
            if let Some(band) = freq_to_band.get(&(freq as u32)) {
                chroma_frame[*band] += magnitude;
            }
        }

        // Normalize
        let sum: f32 = chroma_frame.iter().sum();
        if sum > 0.0 {
            for val in &mut chroma_frame {
                *val /= sum;
            }
        }

        chroma.push(chroma_frame);
    }

    Ok(chroma)
}

/// Creates frequency to band mapping.
fn create_frequency_bands(
    config: &AudioFingerprintConfig,
) -> std::collections::HashMap<u32, usize> {
    let mut mapping = std::collections::HashMap::new();

    let freq_range = config.max_freq - config.min_freq;
    let band_width = freq_range / config.num_bands as f32;

    for freq in (config.min_freq as u32)..=(config.max_freq as u32) {
        let band = ((freq as f32 - config.min_freq) / band_width).floor() as usize;
        let band = band.min(config.num_bands - 1);
        mapping.insert(freq, band);
    }

    mapping
}

/// Quantizes chromagram features to binary fingerprint.
fn quantize_features(chroma: &[Vec<f32>]) -> Vec<u32> {
    let mut fingerprint = Vec::new();

    if chroma.len() < 2 {
        return fingerprint;
    }

    // Compare consecutive frames
    for i in 0..chroma.len() - 1 {
        let mut hash: u32 = 0;

        for band in 0..chroma[i].len().min(32) {
            let diff = chroma[i + 1][band] - chroma[i][band];
            if diff > 0.0 {
                hash |= 1u32 << band;
            }
        }

        fingerprint.push(hash);
    }

    fingerprint
}

/// Compares two fingerprints and returns similarity score.
///
/// Returns a value in [0.0, 1.0] where 1.0 means identical.
#[must_use]
pub fn compare_fingerprints(fp1: &[u32], fp2: &[u32]) -> f64 {
    if fp1.is_empty() || fp2.is_empty() {
        return 0.0;
    }

    // Use cross-correlation to handle offsets
    let max_offset = 100.min(fp1.len().min(fp2.len()) / 2);
    let mut best_score = 0.0;

    for offset in 0..max_offset {
        let score = compare_at_offset(fp1, fp2, offset);
        if score > best_score {
            best_score = score;
        }
    }

    best_score
}

/// Compares fingerprints at a specific offset.
fn compare_at_offset(fp1: &[u32], fp2: &[u32], offset: usize) -> f64 {
    let len = (fp1.len() - offset).min(fp2.len());
    if len == 0 {
        return 0.0;
    }

    let mut matching_bits = 0;
    let mut total_bits = 0;

    for i in 0..len {
        let bits1 = fp1[i + offset];
        let bits2 = fp2[i];

        // Count matching bits
        let matching = (bits1 ^ bits2).count_zeros();
        matching_bits += matching;
        total_bits += 32;
    }

    matching_bits as f64 / total_bits as f64
}

/// Finds the offset where the fingerprints match best.
///
/// Returns `(offset, similarity_score)` or `None` if no good match.
#[must_use]
pub fn find_best_offset(fp1: &[u32], fp2: &[u32], threshold: f64) -> Option<(usize, f64)> {
    if fp1.is_empty() || fp2.is_empty() {
        return None;
    }

    let max_offset = 200.min(fp1.len().min(fp2.len()) / 2);
    let mut best_offset = 0;
    let mut best_score = 0.0;

    for offset in 0..max_offset {
        let score = compare_at_offset(fp1, fp2, offset);
        if score > best_score {
            best_score = score;
            best_offset = offset;
        }
    }

    if best_score >= threshold {
        Some((best_offset, best_score))
    } else {
        None
    }
}

/// Extracts fingerprint from audio file (parallel version).
///
/// # Errors
///
/// Returns an error if extraction fails.
pub fn extract_fingerprint_parallel(
    samples: &[f32],
    sample_rate: u32,
    frame_size: usize,
    hop_size: usize,
) -> CvResult<Vec<u32>> {
    if samples.is_empty() {
        return Err(CvError::invalid_parameter("samples", "empty"));
    }

    let config = AudioFingerprintConfig {
        sample_rate,
        frame_size,
        hop_size,
        ..Default::default()
    };

    // Split samples into chunks for parallel processing
    let num_frames = (samples.len() - frame_size) / hop_size + 1;
    let chunk_size = 1000; // Process 1000 frames at a time

    let fingerprints: Vec<Vec<u32>> = (0..num_frames)
        .step_by(chunk_size)
        .collect::<Vec<_>>()
        .par_iter()
        .filter_map(|&chunk_start| {
            let chunk_end = (chunk_start + chunk_size).min(num_frames);
            let sample_start = chunk_start * hop_size;
            let sample_end = (chunk_end * hop_size + frame_size).min(samples.len());

            if sample_end <= sample_start {
                return None;
            }

            let chunk_samples = &samples[sample_start..sample_end];
            extract_fingerprint(chunk_samples, sample_rate, frame_size, hop_size).ok()
        })
        .collect();

    // Merge results
    let mut result = Vec::new();
    for fp in fingerprints {
        result.extend(fp);
    }

    Ok(result)
}

/// Searches for a query fingerprint in a database.
///
/// Returns matching indices and scores.
#[must_use]
pub fn search_database(
    query: &[u32],
    database: &[Vec<u32>],
    threshold: f64,
) -> Vec<(usize, f64, usize)> {
    database
        .iter()
        .enumerate()
        .filter_map(|(idx, db_fp)| {
            if let Some((offset, score)) = find_best_offset(query, db_fp, threshold) {
                Some((idx, score, offset))
            } else {
                None
            }
        })
        .collect()
}

/// Computes hamming distance between two fingerprints.
#[must_use]
pub fn hamming_distance(fp1: &[u32], fp2: &[u32]) -> usize {
    let len = fp1.len().min(fp2.len());
    let mut distance = 0;

    for i in 0..len {
        distance += (fp1[i] ^ fp2[i]).count_ones() as usize;
    }

    // Add penalty for length difference
    distance += (fp1.len().abs_diff(fp2.len())) * 32;

    distance
}

/// Normalizes audio samples to [-1.0, 1.0].
#[must_use]
pub fn normalize_audio(samples: &[f32]) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }

    let max_val = samples.iter().map(|&x| x.abs()).fold(0.0f32, f32::max);

    if max_val == 0.0 {
        return samples.to_vec();
    }

    samples.iter().map(|&x| x / max_val).collect()
}

/// Converts stereo to mono by averaging channels.
#[must_use]
pub fn stereo_to_mono(stereo: &[f32]) -> Vec<f32> {
    let mut mono = Vec::with_capacity(stereo.len() / 2);

    for chunk in stereo.chunks_exact(2) {
        mono.push((chunk[0] + chunk[1]) / 2.0);
    }

    mono
}

/// Downsamples audio to target sample rate.
#[must_use]
pub fn downsample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f32 / to_rate as f32;
    let output_len = (samples.len() as f32 / ratio) as usize;
    let mut output = Vec::with_capacity(output_len);

    for i in 0..output_len {
        let src_idx = (i as f32 * ratio) as usize;
        if src_idx < samples.len() {
            output.push(samples[src_idx]);
        }
    }

    output
}

/// Applies high-pass filter to remove DC offset and low frequencies.
#[must_use]
pub fn high_pass_filter(samples: &[f32], cutoff_freq: f32, sample_rate: u32) -> Vec<f32> {
    let rc = 1.0 / (2.0 * std::f32::consts::PI * cutoff_freq);
    let dt = 1.0 / sample_rate as f32;
    let alpha = rc / (rc + dt);

    let mut output = Vec::with_capacity(samples.len());
    let mut prev_input = 0.0;
    let mut prev_output = 0.0;

    for &sample in samples {
        let filtered = alpha * (prev_output + sample - prev_input);
        output.push(filtered);
        prev_input = sample;
        prev_output = filtered;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_audio(duration_secs: f32, sample_rate: u32, freq: f32) -> Vec<f32> {
        let num_samples = (duration_secs * sample_rate as f32) as usize;
        (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * freq * t).sin()
            })
            .collect()
    }

    #[test]
    fn test_extract_fingerprint() {
        let samples = create_test_audio(1.0, 11025, 440.0);
        let fp = extract_fingerprint(&samples, 11025, 4096, 1024)
            .expect("extract_fingerprint should succeed");
        assert!(!fp.is_empty());
    }

    #[test]
    fn test_config_default() {
        let config = AudioFingerprintConfig::default();
        assert_eq!(config.sample_rate, 11025);
        assert_eq!(config.frame_size, 4096);
        assert_eq!(config.hop_size, 1024);
    }

    #[test]
    fn test_hanning_window() {
        let window = create_hanning_window(512);
        assert_eq!(window.len(), 512);
        assert!(window[0] < 0.1);
        assert!(window[256] > 0.9);
    }

    #[test]
    fn test_compare_fingerprints() {
        let fp1 = vec![0x12345678, 0x9ABCDEF0];
        let fp2 = vec![0x12345678, 0x9ABCDEF0];
        let score = compare_fingerprints(&fp1, &fp2);
        assert!(score > 0.95);
    }

    #[test]
    fn test_compare_different_fingerprints() {
        let fp1 = vec![0xFFFFFFFF; 10];
        let fp2 = vec![0x00000000; 10];
        let score = compare_fingerprints(&fp1, &fp2);
        assert!(score < 0.1);
    }

    #[test]
    fn test_find_best_offset() {
        let fp1 = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let fp2 = vec![0, 0, 1, 2, 3, 4, 5, 6];

        let result = find_best_offset(&fp1, &fp2, 0.5);
        assert!(result.is_some());
    }

    #[test]
    fn test_hamming_distance() {
        let fp1 = vec![0, 0, 0];
        let fp2 = vec![0, 0, 0];
        assert_eq!(hamming_distance(&fp1, &fp2), 0);

        let fp3 = vec![0xFFFFFFFF; 3];
        let distance = hamming_distance(&fp1, &fp3);
        assert_eq!(distance, 96); // 3 * 32 bits
    }

    #[test]
    fn test_normalize_audio() {
        let samples = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let normalized = normalize_audio(&samples);
        assert_eq!(normalized.len(), 5);
        assert!(normalized.iter().all(|&x| x >= -1.0 && x <= 1.0));
    }

    #[test]
    fn test_stereo_to_mono() {
        let stereo = vec![1.0, 0.0, 0.5, 0.5, -1.0, 1.0];
        let mono = stereo_to_mono(&stereo);
        assert_eq!(mono.len(), 3);
        assert_eq!(mono[0], 0.5);
        assert_eq!(mono[1], 0.5);
        assert_eq!(mono[2], 0.0);
    }

    #[test]
    fn test_downsample() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let downsampled = downsample(&samples, 8000, 4000);
        assert!(downsampled.len() >= 3 && downsampled.len() <= 5);
    }

    #[test]
    fn test_high_pass_filter() {
        let samples = create_test_audio(0.1, 11025, 440.0);
        let filtered = high_pass_filter(&samples, 100.0, 11025);
        assert_eq!(filtered.len(), samples.len());
    }

    #[test]
    fn test_empty_samples() {
        let samples: Vec<f32> = Vec::new();
        assert!(extract_fingerprint(&samples, 11025, 4096, 1024).is_err());
    }

    #[test]
    fn test_invalid_frame_size() {
        let samples = create_test_audio(1.0, 11025, 440.0);
        assert!(extract_fingerprint(&samples, 11025, 100, 50).is_err());
    }

    #[test]
    fn test_search_database() {
        let query = vec![1, 2, 3, 4, 5];
        let database = vec![
            vec![1, 2, 3, 4, 5],
            vec![6, 7, 8, 9, 10],
            vec![1, 2, 3, 4, 6],
        ];

        let results = search_database(&query, &database, 0.8);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_quantize_features() {
        let chroma = vec![
            vec![0.1, 0.2, 0.3],
            vec![0.2, 0.3, 0.4],
            vec![0.3, 0.4, 0.5],
        ];

        let fp = quantize_features(&chroma);
        assert_eq!(fp.len(), 2); // n-1 for n frames
    }

    #[test]
    fn test_audio_fingerprint() {
        let fp = AudioFingerprint::new(vec![1, 2, 3], 10.0, 11025);
        assert_eq!(fp.fingerprint.len(), 3);
        assert_eq!(fp.duration, 10.0);
        assert_eq!(fp.sample_rate, 11025);
        assert_eq!(fp.size_bytes(), 12);
    }
}
