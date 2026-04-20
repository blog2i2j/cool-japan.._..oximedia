//! Direct Sequence Spread Spectrum (DSSS) watermarking.
//!
//! This module implements spread spectrum watermarking, which embeds
//! watermark bits by spreading them using pseudorandom sequences.

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{generate_pn_sequence, pack_bits, unpack_bits, PayloadCodec};
use crate::psychoacoustic::PsychoacousticModel;
use oxifft::Complex;

/// Spread spectrum watermarking configuration.
#[derive(Debug, Clone)]
pub struct SpreadSpectrumConfig {
    /// Embedding strength (0.0 to 1.0).
    pub strength: f32,
    /// Chip rate (spreading factor).
    pub chip_rate: usize,
    /// Use frequency domain embedding.
    pub frequency_domain: bool,
    /// Psychoacoustic masking enabled.
    pub psychoacoustic: bool,
    /// Secret key for PN sequence generation.
    pub key: u64,
}

impl Default for SpreadSpectrumConfig {
    fn default() -> Self {
        Self {
            strength: 0.1,
            chip_rate: 64,
            frequency_domain: true,
            psychoacoustic: true,
            key: 0,
        }
    }
}

/// Spread spectrum watermark embedder.
pub struct SpreadSpectrumEmbedder {
    config: SpreadSpectrumConfig,
    codec: PayloadCodec,
    psycho_model: Option<PsychoacousticModel>,
}

impl SpreadSpectrumEmbedder {
    /// Create a new spread spectrum embedder.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(
        config: SpreadSpectrumConfig,
        sample_rate: u32,
        frame_size: usize,
    ) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        let psycho_model = if config.psychoacoustic {
            Some(PsychoacousticModel::new(sample_rate, frame_size))
        } else {
            None
        };

        Ok(Self {
            config,
            codec,
            psycho_model,
        })
    }

    /// Embed watermark in audio samples.
    ///
    /// # Errors
    ///
    /// Returns error if audio is too short or encoding fails.
    pub fn embed(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        // Encode payload with error correction
        let encoded = self.codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        if self.config.frequency_domain {
            self.embed_frequency_domain(samples, &bits)
        } else {
            self.embed_time_domain(samples, &bits)
        }
    }

    /// Embed in time domain.
    fn embed_time_domain(&self, samples: &[f32], bits: &[bool]) -> WatermarkResult<Vec<f32>> {
        let required_samples = bits.len() * self.config.chip_rate;
        if samples.len() < required_samples {
            return Err(WatermarkError::InsufficientCapacity {
                needed: required_samples,
                have: samples.len(),
            });
        }

        let mut watermarked = samples.to_vec();

        for (bit_idx, &bit) in bits.iter().enumerate() {
            let pn_seq =
                generate_pn_sequence(self.config.chip_rate, self.config.key + bit_idx as u64);
            let bit_value = if bit { 1.0f32 } else { -1.0f32 };

            let start = bit_idx * self.config.chip_rate;
            for (i, &pn) in pn_seq.iter().enumerate() {
                if start + i >= watermarked.len() {
                    break;
                }
                watermarked[start + i] += self.config.strength * bit_value * f32::from(pn);
            }
        }

        Ok(watermarked)
    }

    /// Embed in frequency domain.
    fn embed_frequency_domain(&self, samples: &[f32], bits: &[bool]) -> WatermarkResult<Vec<f32>> {
        let frame_size = 2048;
        // Use non-overlapping frames (hop = frame_size) to avoid overwrite conflicts
        // when writing IFFT output back to the signal buffer.
        let hop_size = frame_size;
        let required_frames = bits.len().div_ceil(8); // 8 bits per frame

        if samples.len() < required_frames * hop_size {
            return Err(WatermarkError::InsufficientCapacity {
                needed: required_frames * hop_size,
                have: samples.len(),
            });
        }

        let mut watermarked = samples.to_vec();

        // Calculate masking threshold using the first frame only.
        // The psychoacoustic model requires exactly frame_size samples.
        let masking = if let Some(ref model) = self.psycho_model {
            let first_frame = &samples[..frame_size.min(samples.len())];
            if first_frame.len() == frame_size {
                Some(model.calculate_masking_threshold(first_frame))
            } else {
                None
            }
        } else {
            None
        };

        let mut bit_idx = 0;

        for frame_idx in 0..required_frames {
            if bit_idx >= bits.len() {
                break;
            }

            let frame_start = frame_idx * hop_size;
            if frame_start + frame_size > samples.len() {
                break;
            }

            // Extract frame
            let frame = &samples[frame_start..frame_start + frame_size];

            // FFT
            let freq_input: Vec<Complex<f32>> =
                frame.iter().map(|&s| Complex::new(s, 0.0)).collect();
            let mut freq_data = oxifft::fft(&freq_input);

            // Embed bits in frequency domain
            for _ in 0..8 {
                if bit_idx >= bits.len() {
                    break;
                }

                let bit = bits[bit_idx];
                let pn_seq =
                    generate_pn_sequence(self.config.chip_rate, self.config.key + bit_idx as u64);

                let bit_value = if bit { 1.0f32 } else { -1.0f32 };

                // Embed in mid-frequency range (more robust)
                let start_bin = frame_size / 8;
                let end_bin = start_bin + self.config.chip_rate;

                for (i, &pn) in pn_seq.iter().enumerate().take(self.config.chip_rate) {
                    let bin = start_bin + i;
                    if bin >= end_bin || bin >= freq_data.len() / 2 {
                        break;
                    }

                    // Calculate embedding strength based on masking
                    let strength = if let Some(ref mask) = masking {
                        let mask_val = mask.get(bin).copied().unwrap_or(-60.0);
                        // Scale strength based on masking threshold
                        self.config.strength * 10.0f32.powf(mask_val / 20.0)
                    } else {
                        self.config.strength
                    };

                    let watermark_val = strength * bit_value * f32::from(pn);
                    freq_data[bin] += Complex::new(watermark_val, 0.0);

                    // Mirror for conjugate symmetry
                    let mirror_bin = frame_size - bin;
                    if mirror_bin < freq_data.len() {
                        freq_data[mirror_bin] += Complex::new(watermark_val, 0.0);
                    }
                }

                bit_idx += 1;
            }

            // IFFT
            let ifft_result = oxifft::ifft(&freq_data);

            // Overlap-add
            #[allow(clippy::cast_precision_loss)]
            let scale = 1.0 / frame_size as f32;
            for (i, c) in ifft_result.iter().enumerate().take(frame_size) {
                let idx = frame_start + i;
                if idx < watermarked.len() {
                    watermarked[idx] = c.re * scale;
                }
            }
        }

        Ok(watermarked)
    }

    /// Calculate capacity in bits for given audio length.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        if self.config.frequency_domain {
            let frame_size = 2048;
            // Non-overlapping frames
            let hop_size = frame_size;
            let frame_count = sample_count / hop_size;
            frame_count * 8 // 8 bits per frame
        } else {
            sample_count / self.config.chip_rate
        }
    }
}

/// Spread spectrum watermark detector.
pub struct SpreadSpectrumDetector {
    config: SpreadSpectrumConfig,
    codec: PayloadCodec,
}

impl SpreadSpectrumDetector {
    /// Create a new spread spectrum detector.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: SpreadSpectrumConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Detect and extract watermark from audio samples.
    ///
    /// # Errors
    ///
    /// Returns error if watermark not detected or decoding fails.
    pub fn detect(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        let bits = if self.config.frequency_domain {
            self.detect_frequency_domain(samples, expected_bits)?
        } else {
            self.detect_time_domain(samples, expected_bits)?
        };

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }

    /// Detect in time domain.
    fn detect_time_domain(
        &self,
        samples: &[f32],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<bool>> {
        let mut bits = Vec::new();

        for bit_idx in 0..expected_bits {
            let pn_seq =
                generate_pn_sequence(self.config.chip_rate, self.config.key + bit_idx as u64);
            let start = bit_idx * self.config.chip_rate;

            if start + self.config.chip_rate > samples.len() {
                break;
            }

            // Correlate with PN sequence
            let mut corr = 0.0f32;
            for (i, &pn) in pn_seq.iter().enumerate() {
                corr += samples[start + i] * f32::from(pn);
            }

            bits.push(corr > 0.0);
        }

        Ok(bits)
    }

    /// Detect in frequency domain.
    fn detect_frequency_domain(
        &self,
        samples: &[f32],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<bool>> {
        let frame_size = 2048;
        // Use non-overlapping frames to match the embedder's frame layout.
        let hop_size = frame_size;
        let required_frames = expected_bits.div_ceil(8);

        let mut bits = Vec::new();

        for frame_idx in 0..required_frames {
            let frame_start = frame_idx * hop_size;
            if frame_start + frame_size > samples.len() {
                break;
            }

            let frame = &samples[frame_start..frame_start + frame_size];

            // FFT
            let freq_input: Vec<Complex<f32>> =
                frame.iter().map(|&s| Complex::new(s, 0.0)).collect();
            let freq_data = oxifft::fft(&freq_input);

            // Extract bits
            for _ in 0..8 {
                if bits.len() >= expected_bits {
                    break;
                }

                let bit_idx = bits.len();
                let pn_seq =
                    generate_pn_sequence(self.config.chip_rate, self.config.key + bit_idx as u64);

                let start_bin = frame_size / 8;
                let mut corr = 0.0f32;

                for (i, &pn) in pn_seq.iter().enumerate().take(self.config.chip_rate) {
                    let bin = start_bin + i;
                    if bin >= freq_data.len() / 2 {
                        break;
                    }

                    corr += freq_data[bin].re * f32::from(pn);
                }

                bits.push(corr > 0.0);
            }
        }

        Ok(bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_spectrum_time_domain() {
        let config = SpreadSpectrumConfig {
            strength: 0.05,
            chip_rate: 32,
            frequency_domain: false,
            psychoacoustic: false,
            key: 12345,
        };

        let embedder = SpreadSpectrumEmbedder::new(config.clone(), 44100, 2048).unwrap();
        let detector = SpreadSpectrumDetector::new(config).unwrap();

        let samples: Vec<f32> = vec![0.0; 10000];
        let payload = b"Test";

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
    fn test_spread_spectrum_frequency_domain() {
        let config = SpreadSpectrumConfig {
            strength: 0.1,
            chip_rate: 64,
            frequency_domain: true,
            psychoacoustic: false,
            key: 54321,
        };

        let embedder = SpreadSpectrumEmbedder::new(config.clone(), 44100, 2048).unwrap();
        let detector = SpreadSpectrumDetector::new(config).unwrap();

        // "WM" encodes to 280 bits, requiring 35 frames * 2048 frame_size = 71680 samples
        // with non-overlapping frames. Use 73728 (36 * 2048) for headroom.
        let samples: Vec<f32> = vec![0.0; 73728];
        let payload = b"WM";

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
    fn test_capacity_calculation() {
        let config = SpreadSpectrumConfig::default();
        let embedder = SpreadSpectrumEmbedder::new(config, 44100, 2048).unwrap();

        let capacity = embedder.capacity(44100); // 1 second
        assert!(capacity > 0);
    }
}
