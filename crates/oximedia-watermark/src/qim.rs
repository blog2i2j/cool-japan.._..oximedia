//! Quantization Index Modulation (QIM) watermarking.
//!
//! This module implements QIM-based watermarking, which provides
//! excellent robustness through quantization-based embedding.

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{pack_bits, unpack_bits, PayloadCodec};
use oxifft::Complex;

/// QIM configuration.
#[derive(Debug, Clone)]
pub struct QimConfig {
    /// Quantization step size.
    pub step_size: f32,
    /// Use dither modulation.
    pub dither: bool,
    /// Frequency domain embedding.
    pub frequency_domain: bool,
    /// Frame size for frequency domain.
    pub frame_size: usize,
    /// Start bin for frequency embedding.
    pub start_bin: usize,
    /// End bin for frequency embedding.
    pub end_bin: usize,
}

impl Default for QimConfig {
    fn default() -> Self {
        Self {
            step_size: 0.01,
            dither: true,
            frequency_domain: true,
            frame_size: 2048,
            start_bin: 50,
            end_bin: 500,
        }
    }
}

/// QIM watermark embedder.
pub struct QimEmbedder {
    config: QimConfig,
    codec: PayloadCodec,
}

impl QimEmbedder {
    /// Create a new QIM embedder.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: QimConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Embed watermark using QIM.
    ///
    /// # Errors
    ///
    /// Returns error if audio is too short or encoding fails.
    pub fn embed(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        // Encode payload
        let encoded = self.codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        if self.config.frequency_domain {
            self.embed_frequency(samples, &bits)
        } else {
            self.embed_time(samples, &bits)
        }
    }

    /// Embed in time domain.
    fn embed_time(&self, samples: &[f32], bits: &[bool]) -> WatermarkResult<Vec<f32>> {
        let samples_per_bit = 100;
        let required_samples = bits.len() * samples_per_bit;

        if samples.len() < required_samples {
            return Err(WatermarkError::InsufficientCapacity {
                needed: required_samples,
                have: samples.len(),
            });
        }

        let mut watermarked = samples.to_vec();

        for (bit_idx, &bit) in bits.iter().enumerate() {
            let start = bit_idx * samples_per_bit;
            let end = (start + samples_per_bit).min(watermarked.len());

            for sample in &mut watermarked[start..end] {
                *sample = self.quantize(*sample, bit);
            }
        }

        Ok(watermarked)
    }

    /// Embed in frequency domain.
    fn embed_frequency(&self, samples: &[f32], bits: &[bool]) -> WatermarkResult<Vec<f32>> {
        // Use non-overlapping frames so that writing IFFT output back to the
        // signal buffer does not corrupt previously embedded frames.
        let hop_size = self.config.frame_size;
        let bins_per_bit = 10;
        let bits_per_frame = (self.config.end_bin - self.config.start_bin) / bins_per_bit;

        let required_frames = bits.len().div_ceil(bits_per_frame);
        let required_samples = required_frames * hop_size;

        if samples.len() < required_samples {
            return Err(WatermarkError::InsufficientCapacity {
                needed: required_samples,
                have: samples.len(),
            });
        }

        let mut watermarked = samples.to_vec();

        let mut bit_idx = 0;

        for frame_idx in 0..required_frames {
            if bit_idx >= bits.len() {
                break;
            }

            let frame_start = frame_idx * hop_size; // non-overlapping
            if frame_start + self.config.frame_size > samples.len() {
                break;
            }

            let frame = &samples[frame_start..frame_start + self.config.frame_size];

            // FFT
            let freq_input: Vec<Complex<f32>> =
                frame.iter().map(|&s| Complex::new(s, 0.0)).collect();
            let mut freq_data = oxifft::fft(&freq_input);

            // Embed bits in magnitude
            for _ in 0..bits_per_frame {
                if bit_idx >= bits.len() {
                    break;
                }

                let bit = bits[bit_idx];
                let bin_start = self.config.start_bin + (bit_idx % bits_per_frame) * bins_per_bit;

                for offset in 0..bins_per_bit {
                    let bin = bin_start + offset;
                    if bin >= self.config.end_bin || bin >= freq_data.len() / 2 {
                        break;
                    }

                    let mag = freq_data[bin].norm();
                    let phase = freq_data[bin].arg();

                    // Quantize magnitude
                    let new_mag = self.quantize(mag, bit);
                    freq_data[bin] = Complex::from_polar(new_mag, phase);

                    // Conjugate symmetry
                    let mirror = self.config.frame_size - bin;
                    if mirror < freq_data.len() {
                        freq_data[mirror] = freq_data[bin].conj();
                    }
                }

                bit_idx += 1;
            }

            // IFFT
            // oxifft::ifft already normalises by 1/N, so write back directly.
            let ifft_result = oxifft::ifft(&freq_data);

            for (i, c) in ifft_result.iter().enumerate() {
                let idx = frame_start + i;
                if idx < watermarked.len() {
                    watermarked[idx] = c.re;
                }
            }
        }

        Ok(watermarked)
    }

    /// Quantize value based on bit.
    fn quantize(&self, value: f32, bit: bool) -> f32 {
        let delta = self.config.step_size;

        // Two quantizers: Q0 for bit=0, Q1 for bit=1
        // Q1 is offset by delta/2 from Q0
        let offset = if bit { delta / 2.0 } else { 0.0 };

        let quantized = ((value - offset) / delta).round() * delta + offset;

        if self.config.dither {
            // Add small dither using deterministic seed based on value bits
            let seed = u64::from(quantized.to_bits());
            let mut rng = scirs2_core::random::Random::seed(seed);
            quantized + (rng.random_f64() as f32 - 0.5) * delta * 0.1
        } else {
            quantized
        }
    }

    /// Detect quantized bit.
    #[allow(dead_code)]
    fn detect_bit(&self, value: f32) -> bool {
        let delta = self.config.step_size;

        // Calculate distance to both quantizers
        let dist_0 = (value - (value / delta).round() * delta).abs();
        let dist_1 = (value - delta / 2.0 - ((value - delta / 2.0) / delta).round() * delta).abs();

        // Choose quantizer with smaller distance
        dist_1 < dist_0
    }

    /// Calculate capacity in bits.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        if self.config.frequency_domain {
            // Non-overlapping frames
            let hop_size = self.config.frame_size;
            let bins_per_bit = 10;
            let bits_per_frame = (self.config.end_bin - self.config.start_bin) / bins_per_bit;
            (sample_count / hop_size) * bits_per_frame
        } else {
            sample_count / 100
        }
    }
}

/// QIM watermark detector.
pub struct QimDetector {
    config: QimConfig,
    codec: PayloadCodec,
}

impl QimDetector {
    /// Create a new QIM detector.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: QimConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Detect and extract watermark.
    ///
    /// # Errors
    ///
    /// Returns error if decoding fails.
    pub fn detect(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        let bits = if self.config.frequency_domain {
            self.detect_frequency(samples, expected_bits)?
        } else {
            self.detect_time(samples, expected_bits)?
        };

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }

    /// Detect in time domain.
    fn detect_time(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<bool>> {
        let samples_per_bit = 100;
        let mut bits = Vec::new();

        for bit_idx in 0..expected_bits {
            let start = bit_idx * samples_per_bit;
            let end = (start + samples_per_bit).min(samples.len());

            if end <= start {
                break;
            }

            // Majority vote
            let mut count_1 = 0;
            for &sample in &samples[start..end] {
                if self.detect_bit(sample) {
                    count_1 += 1;
                }
            }

            bits.push(count_1 > (end - start) / 2);
        }

        Ok(bits)
    }

    /// Detect in frequency domain.
    fn detect_frequency(
        &self,
        samples: &[f32],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<bool>> {
        let mut bits = Vec::new();

        // Use non-overlapping frames to match the embedder's frame layout.
        let hop_size = self.config.frame_size;
        let bins_per_bit = 10;
        let bits_per_frame = (self.config.end_bin - self.config.start_bin) / bins_per_bit;

        for frame_start in (0..samples.len()).step_by(hop_size) {
            if bits.len() >= expected_bits {
                break;
            }

            if frame_start + self.config.frame_size > samples.len() {
                break;
            }

            let frame = &samples[frame_start..frame_start + self.config.frame_size];

            // FFT
            let freq_input: Vec<Complex<f32>> =
                frame.iter().map(|&s| Complex::new(s, 0.0)).collect();
            let freq_data = oxifft::fft(&freq_input);

            // Detect bits from magnitude
            for _ in 0..bits_per_frame {
                if bits.len() >= expected_bits {
                    break;
                }

                let bit_idx = bits.len();
                let bin_start = self.config.start_bin + (bit_idx % bits_per_frame) * bins_per_bit;

                let mut count_1 = 0;
                let mut total = 0;

                for offset in 0..bins_per_bit {
                    let bin = bin_start + offset;
                    if bin >= self.config.end_bin || bin >= freq_data.len() / 2 {
                        break;
                    }

                    let mag = freq_data[bin].norm();
                    if self.detect_bit(mag) {
                        count_1 += 1;
                    }
                    total += 1;
                }

                if total > 0 {
                    bits.push(count_1 > total / 2);
                }
            }
        }

        Ok(bits)
    }

    /// Detect quantized bit.
    fn detect_bit(&self, value: f32) -> bool {
        let delta = self.config.step_size;
        let dist_0 = (value - (value / delta).round() * delta).abs();
        let dist_1 = (value - delta / 2.0 - ((value - delta / 2.0) / delta).round() * delta).abs();
        dist_1 < dist_0
    }
}

/// Distortion-compensated QIM (DC-QIM).
pub struct DcQimEmbedder {
    step_size: f32,
    alpha: f32, // Compensation factor
}

impl DcQimEmbedder {
    /// Create a new DC-QIM embedder.
    #[must_use]
    pub fn new(step_size: f32, alpha: f32) -> Self {
        Self { step_size, alpha }
    }

    /// Embed with distortion compensation.
    #[must_use]
    pub fn embed(&self, value: f32, bit: bool) -> f32 {
        let delta = self.step_size;
        let offset = if bit { delta / 2.0 } else { 0.0 };

        let quantized = ((value - offset) / delta).round() * delta + offset;

        // Apply distortion compensation
        value + self.alpha * (quantized - value)
    }

    /// Detect bit.
    #[must_use]
    pub fn detect(&self, value: f32) -> bool {
        let delta = self.step_size;
        let dist_0 = (value - (value / delta).round() * delta).abs();
        let dist_1 = (value - delta / 2.0 - ((value - delta / 2.0) / delta).round() * delta).abs();
        dist_1 < dist_0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qim_time_domain() {
        let config = QimConfig {
            step_size: 0.02,
            dither: false,
            frequency_domain: false,
            ..Default::default()
        };

        let embedder = QimEmbedder::new(config.clone()).unwrap();
        let detector = QimDetector::new(config).unwrap();

        let samples: Vec<f32> = vec![0.1; 50000];
        let payload = b"QIM";

        let watermarked = embedder
            .embed(&samples, payload)
            .expect("should succeed in test");

        let encoded = embedder
            .codec
            .encode(payload)
            .expect("should succeed in test");
        let expected_bits = encoded.len() * 8;

        let extracted = detector
            .detect(&watermarked, expected_bits)
            .expect("should succeed in test");
        assert_eq!(payload.as_slice(), extracted.as_slice());
    }

    #[test]
    fn test_qim_frequency_domain() {
        // Disable dithering for deterministic test results; dithering adds
        // randomness that can flip quantization decisions during detection.
        let config = QimConfig {
            step_size: 0.05,
            dither: false,
            frequency_domain: true,
            ..Default::default()
        };

        let embedder = QimEmbedder::new(config.clone()).unwrap();
        let detector = QimDetector::new(config).unwrap();

        // Use pseudo-random noise for broadband FFT energy across all bins
        // (especially bins 50-499 used for QIM embedding with this config).
        let mut rng = scirs2_core::random::Random::seed(77);
        let samples: Vec<f32> = (0..50000).map(|_| rng.random_f64() as f32 - 0.5).collect();
        let payload = b"FQ";

        let watermarked = embedder
            .embed(&samples, payload)
            .expect("should succeed in test");

        let encoded = embedder
            .codec
            .encode(payload)
            .expect("should succeed in test");
        let expected_bits = encoded.len() * 8;

        let extracted = detector
            .detect(&watermarked, expected_bits)
            .expect("should succeed in test");
        assert_eq!(payload.as_slice(), extracted.as_slice());
    }

    #[test]
    fn test_quantization() {
        let config = QimConfig {
            step_size: 0.1,
            dither: false,
            ..Default::default()
        };

        let embedder = QimEmbedder::new(config).unwrap();

        let value = 0.55f32;
        let q0 = embedder.quantize(value, false);
        let q1 = embedder.quantize(value, true);

        assert_ne!(q0, q1);
        assert!(embedder.detect_bit(q0) == false);
        assert!(embedder.detect_bit(q1) == true);
    }

    #[test]
    fn test_dc_qim() {
        let embedder = DcQimEmbedder::new(0.1, 0.5);

        let value = 0.55f32;
        let wm_0 = embedder.embed(value, false);
        let wm_1 = embedder.embed(value, true);

        assert!(embedder.detect(wm_0) == false);
        assert!(embedder.detect(wm_1) == true);
    }

    #[test]
    fn test_capacity() {
        let config = QimConfig::default();
        let embedder = QimEmbedder::new(config).unwrap();

        let capacity = embedder.capacity(44100);
        assert!(capacity > 0);
    }
}
