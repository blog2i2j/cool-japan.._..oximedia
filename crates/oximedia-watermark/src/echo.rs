//! Echo hiding watermarking.
//!
//! This module implements echo-based watermarking techniques:
//! - Single echo
//! - Double echo (binary encoding)
//! - Triple echo (ternary encoding)

use crate::error::{WatermarkError, WatermarkResult};
use crate::payload::{pack_bits, unpack_bits, PayloadCodec};

/// Echo hiding configuration.
#[derive(Debug, Clone)]
pub struct EchoConfig {
    /// Delay for bit 0 (in samples).
    pub delay_0: usize,
    /// Delay for bit 1 (in samples).
    pub delay_1: usize,
    /// Echo amplitude (0.0 to 1.0).
    pub amplitude: f32,
    /// Decay rate for echo.
    pub decay: f32,
    /// Kernel size for echo detection.
    pub kernel_size: usize,
}

impl Default for EchoConfig {
    fn default() -> Self {
        Self {
            delay_0: 50,  // ~1.1 ms at 44.1kHz
            delay_1: 100, // ~2.3 ms at 44.1kHz
            amplitude: 0.5,
            decay: 0.8,
            kernel_size: 512,
        }
    }
}

/// Echo hiding watermark embedder.
pub struct EchoEmbedder {
    config: EchoConfig,
    codec: PayloadCodec,
}

impl EchoEmbedder {
    /// Create a new echo embedder.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: EchoConfig) -> WatermarkResult<Self> {
        let codec = PayloadCodec::new(16, 8)?;
        Ok(Self { config, codec })
    }

    /// Embed watermark using echo hiding.
    ///
    /// # Errors
    ///
    /// Returns error if audio is too short or encoding fails.
    pub fn embed(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        // Encode payload
        let encoded = self.codec.encode(payload)?;
        let bits = unpack_bits(&encoded, encoded.len() * 8);

        // Check capacity
        let required_samples = bits.len() * self.config.kernel_size;
        if samples.len() < required_samples {
            return Err(WatermarkError::InsufficientCapacity {
                needed: required_samples,
                have: samples.len(),
            });
        }

        let mut watermarked = samples.to_vec();

        // Embed each bit
        for (bit_idx, &bit) in bits.iter().enumerate() {
            let delay = if bit {
                self.config.delay_1
            } else {
                self.config.delay_0
            };

            let start = bit_idx * self.config.kernel_size;
            let end = (start + self.config.kernel_size).min(watermarked.len());

            // Add echo with decay, using the ORIGINAL signal for echo sources to
            // avoid cascade echo-of-echo artifacts from iterative in-place modification.
            // Constrain to the current block boundary to prevent inter-block interference.
            for i in start..end {
                let echo_idx = i + delay;
                if echo_idx < end && echo_idx < watermarked.len() {
                    let echo_amplitude = self.config.amplitude * self.config.decay;
                    watermarked[echo_idx] += samples[i] * echo_amplitude;
                }

                // Multiple echoes for more robustness
                let echo_idx2 = i + delay * 2;
                if echo_idx2 < end && echo_idx2 < watermarked.len() {
                    let echo_amplitude2 = self.config.amplitude * self.config.decay.powi(2);
                    watermarked[echo_idx2] += samples[i] * echo_amplitude2;
                }
            }
        }

        Ok(watermarked)
    }

    /// Calculate capacity in bits.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        sample_count / self.config.kernel_size
    }
}

/// Echo hiding watermark detector.
pub struct EchoDetector {
    config: EchoConfig,
    codec: PayloadCodec,
}

impl EchoDetector {
    /// Create a new echo detector.
    ///
    /// # Errors
    ///
    /// Returns an error if the Reed-Solomon codec cannot be initialised.
    pub fn new(config: EchoConfig) -> WatermarkResult<Self> {
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

        for bit_idx in 0..expected_bits {
            let start = bit_idx * self.config.kernel_size;
            let end = (start + self.config.kernel_size).min(samples.len());

            if end <= start {
                break;
            }

            let segment = &samples[start..end];

            // Calculate autocorrelation at both delays
            let corr_0 = self.autocorrelation(segment, self.config.delay_0);
            let corr_1 = self.autocorrelation(segment, self.config.delay_1);

            // Choose delay with higher correlation
            bits.push(corr_1 > corr_0);
        }

        let bytes = pack_bits(&bits);
        self.codec.decode(&bytes)
    }

    /// Calculate autocorrelation at given delay.
    fn autocorrelation(&self, samples: &[f32], delay: usize) -> f32 {
        if samples.len() <= delay {
            return 0.0;
        }

        let mut sum = 0.0f32;
        let mut energy = 0.0f32;

        for i in 0..(samples.len() - delay) {
            sum += samples[i] * samples[i + delay];
            energy += samples[i] * samples[i];
        }

        if energy > 1e-10 {
            sum / energy
        } else {
            0.0
        }
    }

    /// Calculate cepstrum for echo detection.
    #[allow(dead_code)]
    fn cepstrum(&self, samples: &[f32]) -> Vec<f32> {
        use oxifft::Complex;

        let fft_size = samples.len().next_power_of_two();

        // FFT
        let freq_input: Vec<Complex<f32>> = samples
            .iter()
            .map(|&s| Complex::new(s, 0.0))
            .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
            .take(fft_size)
            .collect();

        let fft_result = oxifft::fft(&freq_input);

        // Log magnitude
        let log_mag: Vec<Complex<f32>> = fft_result
            .iter()
            .map(|c| {
                let mag = c.norm().max(1e-10);
                Complex::new(mag.ln(), 0.0)
            })
            .collect();

        // IFFT
        let ifft_result = oxifft::ifft(&log_mag);

        // Return real part
        #[allow(clippy::cast_precision_loss)]
        ifft_result.iter().map(|c| c.re / fft_size as f32).collect()
    }
}

/// Triple echo watermarking for ternary encoding.
pub struct TripleEchoEmbedder {
    delay_0: usize,
    delay_1: usize,
    delay_2: usize,
    amplitude: f32,
    #[allow(dead_code)]
    kernel_size: usize,
}

impl TripleEchoEmbedder {
    /// Create a new triple echo embedder.
    #[must_use]
    pub fn new(delay_0: usize, delay_1: usize, delay_2: usize, amplitude: f32) -> Self {
        Self {
            delay_0,
            delay_1,
            delay_2,
            amplitude,
            kernel_size: 512,
        }
    }

    /// Embed ternary symbol (0, 1, or 2).
    #[must_use]
    pub fn embed_symbol(&self, samples: &[f32], symbol: u8) -> Vec<f32> {
        let delay = match symbol {
            0 => self.delay_0,
            1 => self.delay_1,
            _ => self.delay_2,
        };

        let mut watermarked = samples.to_vec();

        for i in 0..samples.len() {
            let echo_idx = i + delay;
            if echo_idx < watermarked.len() {
                watermarked[echo_idx] += samples[i] * self.amplitude;
            }
        }

        watermarked
    }

    /// Detect ternary symbol.
    #[must_use]
    pub fn detect_symbol(&self, samples: &[f32]) -> u8 {
        let corr_0 = self.autocorr(samples, self.delay_0);
        let corr_1 = self.autocorr(samples, self.delay_1);
        let corr_2 = self.autocorr(samples, self.delay_2);

        if corr_0 >= corr_1 && corr_0 >= corr_2 {
            0
        } else if corr_1 >= corr_2 {
            1
        } else {
            2
        }
    }

    /// Calculate autocorrelation.
    fn autocorr(&self, samples: &[f32], delay: usize) -> f32 {
        if samples.len() <= delay {
            return 0.0;
        }

        let mut sum = 0.0f32;
        for i in 0..(samples.len() - delay) {
            sum += samples[i] * samples[i + delay];
        }
        sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_echo_embedding() {
        let config = EchoConfig::default();
        let embedder = EchoEmbedder::new(config.clone()).unwrap();
        let detector = EchoDetector::new(config.clone()).unwrap();

        // Payload "Echo Test" (9 bytes) encodes to ~280 bits with PayloadCodec(16,8).
        // Each bit needs kernel_size=512 samples, so we need at least 280*512=143360.
        // Use a signal with a single impulse at the start of each block: this gives
        // zero base autocorrelation at all non-zero lags, so only the embedded echo
        // creates correlation peaks for reliable detection.
        let kernel_size = config.kernel_size;
        let n_blocks = 300; // 300 * 512 = 153600 > 143360 needed
        let mut samples = vec![0.0f32; n_blocks * kernel_size];
        for block in 0..n_blocks {
            samples[block * kernel_size] = 1.0;
        }
        let payload = b"Echo Test";

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
    fn test_autocorrelation() {
        let config = EchoConfig::default();
        let detector = EchoDetector::new(config.clone()).unwrap();

        let samples: Vec<f32> = (0..1000)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)]
                (i as f32 * 0.1).sin()
            })
            .collect();

        let corr = detector.autocorrelation(&samples, 10);
        assert!(corr.abs() <= 1.0);
    }

    #[test]
    fn test_triple_echo() {
        let embedder = TripleEchoEmbedder::new(30, 60, 90, 0.5);

        // Use pseudo-random noise with a long signal (N=10000).  For white noise,
        // the raw autocorrelation at non-zero lags is O(sqrt(N)) while the echo
        // contribution is O(N * amplitude), so the SNR is O(amplitude * sqrt(N)).
        // With amplitude=0.5 and N=10000, SNR ≈ 50 → reliable detection.
        let mut rng = scirs2_core::random::Random::seed(42);
        let samples: Vec<f32> = (0..10000).map(|_| rng.random_f64() as f32 - 0.5).collect();

        for symbol in 0..3 {
            let watermarked = embedder.embed_symbol(&samples, symbol);
            let detected = embedder.detect_symbol(&watermarked);
            assert_eq!(symbol, detected);
        }
    }

    #[test]
    fn test_capacity() {
        let config = EchoConfig::default();
        let embedder = EchoEmbedder::new(config).unwrap();

        let capacity = embedder.capacity(44100); // 1 second at 44.1kHz
        assert!(capacity > 0);
    }
}
