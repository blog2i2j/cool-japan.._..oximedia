//! Phase coding watermarking.
//!
//! This module implements phase-based watermarking in the frequency domain.
//! Watermark bits are embedded by modifying the phase of DFT coefficients.

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{pack_bits, unpack_bits, PayloadCodec};
use oxifft::Complex;
use std::f32::consts::PI;

/// Phase coding configuration.
#[derive(Debug, Clone)]
pub struct PhaseConfig {
    /// Frame size for DFT.
    pub frame_size: usize,
    /// Phase shift for bit 0 (radians).
    pub phase_0: f32,
    /// Phase shift for bit 1 (radians).
    pub phase_1: f32,
    /// Start frequency bin for embedding.
    pub start_bin: usize,
    /// End frequency bin for embedding.
    pub end_bin: usize,
    /// Number of bins per bit.
    pub bins_per_bit: usize,
}

impl Default for PhaseConfig {
    fn default() -> Self {
        Self {
            frame_size: 2048,
            phase_0: -PI / 4.0,
            phase_1: PI / 4.0,
            start_bin: 10,
            end_bin: 500,
            bins_per_bit: 5,
        }
    }
}

/// Phase coding watermark embedder.
pub struct PhaseEmbedder {
    config: PhaseConfig,
    codec: PayloadCodec,
}

impl PhaseEmbedder {
    /// Create a new phase embedder.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: PhaseConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Embed watermark by modifying phase.
    ///
    /// # Errors
    ///
    /// Returns error if audio is too short or encoding fails.
    pub fn embed(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        // Encode payload
        let encoded = self.codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        // Check capacity
        let capacity = self.capacity(samples.len());
        if bits.len() > capacity {
            return Err(WatermarkError::InsufficientCapacity {
                needed: bits.len(),
                have: capacity,
            });
        }

        let mut watermarked = samples.to_vec();

        // Use non-overlapping frames so that writing IFFT output back to the
        // signal buffer does not corrupt previously embedded frames.
        let hop_size = self.config.frame_size;
        let mut bit_idx = 0;

        // Process each frame
        for frame_start in (0..samples.len()).step_by(hop_size) {
            if bit_idx >= bits.len() {
                break;
            }

            if frame_start + self.config.frame_size > samples.len() {
                break;
            }

            let frame = &samples[frame_start..frame_start + self.config.frame_size];

            // Apply window
            let windowed = self.apply_window(frame);

            // FFT
            let freq_input: Vec<Complex<f32>> =
                windowed.iter().map(|&s| Complex::new(s, 0.0)).collect();
            let mut freq_data = oxifft::fft(&freq_input);

            // Store original magnitudes
            let magnitudes: Vec<f32> = freq_data.iter().map(|c| c.norm()).collect();

            // Modify phases for watermark bits
            let bits_per_frame =
                (self.config.end_bin - self.config.start_bin) / self.config.bins_per_bit;

            for frame_bit in 0..bits_per_frame {
                if bit_idx >= bits.len() {
                    break;
                }

                let bit = bits[bit_idx];
                let target_phase = if bit {
                    self.config.phase_1
                } else {
                    self.config.phase_0
                };

                // Modify phase of bins for this bit
                let bin_start = self.config.start_bin + frame_bit * self.config.bins_per_bit;
                for bin_offset in 0..self.config.bins_per_bit {
                    let bin = bin_start + bin_offset;
                    if bin >= self.config.end_bin || bin >= freq_data.len() / 2 {
                        break;
                    }

                    // Set phase while preserving magnitude
                    let mag = magnitudes[bin];
                    freq_data[bin] = Complex::from_polar(mag, target_phase);

                    // Maintain conjugate symmetry
                    let mirror_bin = self.config.frame_size - bin;
                    if mirror_bin < freq_data.len() {
                        freq_data[mirror_bin] = freq_data[bin].conj();
                    }
                }

                bit_idx += 1;
            }

            // IFFT
            let ifft_result = oxifft::ifft(&freq_data);

            // Write back without re-applying the window.  oxifft::ifft already
            // normalises its output by 1/N, so no additional scaling is required.
            for (i, c) in ifft_result.iter().enumerate() {
                let idx = frame_start + i;
                if idx < watermarked.len() {
                    watermarked[idx] = c.re;
                }
            }
        }

        Ok(watermarked)
    }

    /// Apply window function.
    fn apply_window(&self, samples: &[f32]) -> Vec<f32> {
        let window = self.create_window();
        samples
            .iter()
            .zip(window.iter())
            .map(|(&s, &w)| s * w)
            .collect()
    }

    /// Create Hann window.
    fn create_window(&self) -> Vec<f32> {
        (0..self.config.frame_size)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                let t = i as f32 / self.config.frame_size as f32;
                0.5 * (1.0 - (2.0 * PI * t).cos())
            })
            .collect()
    }

    /// Calculate capacity in bits.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        // Non-overlapping frames
        let hop_size = self.config.frame_size;
        let frame_count = sample_count / hop_size;
        let bits_per_frame =
            (self.config.end_bin - self.config.start_bin) / self.config.bins_per_bit;
        frame_count * bits_per_frame
    }
}

/// Phase coding watermark detector.
pub struct PhaseDetector {
    config: PhaseConfig,
    codec: PayloadCodec,
}

impl PhaseDetector {
    /// Create a new phase detector.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: PhaseConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Detect and extract watermark.
    ///
    /// # Errors
    ///
    /// Returns error if watermark not detected or decoding fails.
    pub fn detect(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        let mut bits = Vec::new();

        // Use non-overlapping frames to match the embedder's frame layout.
        let hop_size = self.config.frame_size;

        for frame_start in (0..samples.len()).step_by(hop_size) {
            if bits.len() >= expected_bits {
                break;
            }

            if frame_start + self.config.frame_size > samples.len() {
                break;
            }

            let frame = &samples[frame_start..frame_start + self.config.frame_size];

            // No windowing: the embedder applies analysis window and writes back
            // IFFT output without synthesis window, so FFT(frame) recovers the
            // embedded phases exactly.
            let freq_input: Vec<Complex<f32>> =
                frame.iter().map(|&s| Complex::new(s, 0.0)).collect();
            let freq_data = oxifft::fft(&freq_input);

            // Extract bits from phases
            let bits_per_frame =
                (self.config.end_bin - self.config.start_bin) / self.config.bins_per_bit;

            for frame_bit in 0..bits_per_frame {
                if bits.len() >= expected_bits {
                    break;
                }

                // Average phase over bins for this bit
                let bin_start = self.config.start_bin + frame_bit * self.config.bins_per_bit;
                let mut avg_phase = 0.0f32;
                let mut count = 0;

                for bin_offset in 0..self.config.bins_per_bit {
                    let bin = bin_start + bin_offset;
                    if bin >= self.config.end_bin || bin >= freq_data.len() / 2 {
                        break;
                    }

                    let phase = freq_data[bin].arg();
                    avg_phase += phase;
                    count += 1;
                }

                if count > 0 {
                    #[allow(clippy::cast_precision_loss)]
                    let count_f32 = count as f32;
                    avg_phase /= count_f32;

                    // Determine bit based on phase
                    let phase_diff_0 = (avg_phase - self.config.phase_0).abs();
                    let phase_diff_1 = (avg_phase - self.config.phase_1).abs();

                    bits.push(phase_diff_1 < phase_diff_0);
                }
            }
        }

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }
}

/// Relative phase shift watermarking.
pub struct RelativePhaseEmbedder {
    frame_size: usize,
    phase_delta: f32,
}

impl RelativePhaseEmbedder {
    /// Create a new relative phase embedder.
    #[must_use]
    pub fn new(frame_size: usize, phase_delta: f32) -> Self {
        Self {
            frame_size,
            phase_delta,
        }
    }

    /// Embed bit by relative phase shift.
    #[must_use]
    pub fn embed_bit(&self, samples: &[f32], bit: bool) -> Vec<f32> {
        let freq_input: Vec<Complex<f32>> = samples.iter().map(|&s| Complex::new(s, 0.0)).collect();

        let mut freq_data = oxifft::fft(&freq_input);

        // Apply relative phase shift
        let delta = if bit {
            self.phase_delta
        } else {
            -self.phase_delta
        };

        for i in 1..freq_data.len() / 2 {
            let mag = freq_data[i].norm();
            let phase = freq_data[i].arg();
            freq_data[i] = Complex::from_polar(mag, phase + delta);

            // Conjugate symmetry
            let mirror = self.frame_size - i;
            freq_data[mirror] = freq_data[i].conj();
        }

        let ifft_result = oxifft::ifft(&freq_data);

        #[allow(clippy::cast_precision_loss)]
        ifft_result
            .iter()
            .map(|c| c.re / self.frame_size as f32)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_phase_embedding() {
        let config = PhaseConfig::default();
        let embedder = PhaseEmbedder::new(config.clone()).unwrap();
        let detector = PhaseDetector::new(config).unwrap();

        // Phase coding requires broadband signal energy across all embedding bins
        // (bins 10-499 with this config).  A pure sine wave concentrates energy at
        // a single bin, leaving most embedding bins with near-zero magnitude and
        // unreliable phase detection.  Use pseudo-random noise instead.
        // With frame_size=2048 (non-overlapping frames) and 50000 samples we get
        // 24 frames × 98 bits/frame = 2352 bits of capacity, well above the
        // encoded payload size.
        let mut rng = scirs2_core::random::Random::seed(42);
        let samples: Vec<f32> = (0..50000).map(|_| rng.random_f64() as f32 - 0.5).collect();
        let payload = b"Phase";

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
    fn test_capacity() {
        let config = PhaseConfig::default();
        let embedder = PhaseEmbedder::new(config).unwrap();

        let capacity = embedder.capacity(44100);
        assert!(capacity > 0);
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_relative_phase() {
        let embedder = RelativePhaseEmbedder::new(1024, PI / 8.0);

        // Use a varying signal (sine wave) so phase modifications produce
        // distinct frequency-domain content and yield different time-domain output.
        let samples: Vec<f32> = (0..1024)
            .map(|i| (2.0 * PI * 440.0 * i as f32 / 44100.0).sin() * 0.5)
            .collect();

        let wm_0 = embedder.embed_bit(&samples, false);
        let wm_1 = embedder.embed_bit(&samples, true);

        assert_ne!(wm_0, wm_1);
        assert_eq!(wm_0.len(), 1024);
    }
}
