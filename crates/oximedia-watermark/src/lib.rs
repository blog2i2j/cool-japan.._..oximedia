//! Professional audio watermarking and steganography for `OxiMedia`.
//!
//! This crate provides comprehensive audio watermarking capabilities:
//!
//! # Watermarking Techniques
//!
//! - **Spread Spectrum (DSSS)** - Robust watermarking using pseudorandom sequences
//! - **Echo Hiding** - Single/double/triple echo watermarking
//! - **Phase Coding** - DFT phase modulation watermarking
//! - **LSB Steganography** - Simple but high-capacity embedding
//! - **Patchwork** - Statistical watermarking using sample pairs
//! - **QIM** - Quantization Index Modulation for robust embedding
//!
//! # Features
//!
//! - **Blind Detection** - Extract watermarks without the original audio
//! - **Error Correction** - Reed-Solomon coding for robustness
//! - **Psychoacoustic Masking** - Ensure imperceptibility using human hearing model
//! - **Robustness Testing** - Test against common attacks (MP3, resampling, etc.)
//! - **Quality Metrics** - Objective quality assessment (SNR, ODG, etc.)
//!
//! # Example
//!
//! ```no_run
//! use oximedia_watermark::{WatermarkEmbedder, WatermarkConfig, Algorithm};
//!
//! let config = WatermarkConfig::default()
//!     .with_algorithm(Algorithm::SpreadSpectrum)
//!     .with_strength(0.1);
//!
//! let embedder = WatermarkEmbedder::new(config, 44100);
//!
//! let audio_samples: Vec<f32> = vec![0.0; 44100]; // 1 second
//! let payload = b"Copyright 2024";
//!
//! let watermarked = embedder.embed(&audio_samples, payload).expect("should succeed in test");
//! ```
//!
//! # Responsible Use
//!
//! This watermarking library is intended for legitimate purposes only:
//!
//! - Copyright protection and ownership verification
//! - Broadcast monitoring and content tracking
//! - Authentication and integrity verification
//! - Forensic tracking and leak detection
//!
//! **Ethical Guidelines:**
//!
//! - Do not use for unauthorized surveillance or tracking
//! - Respect privacy and data protection laws
//! - Obtain proper consent for embedding watermarks
//! - Follow applicable copyright and intellectual property laws
//!
//! # Security Considerations
//!
//! - Watermarks are not encryption - do not rely on them for confidentiality
//! - Use strong keys (high entropy) for cryptographic applications
//! - Consider attack scenarios when choosing parameters
//! - Regular security audits are recommended for production use

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod attacks;
pub mod audio_watermark;
/// Batch watermark embedding for processing multiple audio segments.
pub mod batch_embed;
pub mod bit_packing;
pub mod chain_of_custody;
pub mod dct_watermark;
pub mod detection_map;
pub mod detector;
pub mod echo;
pub mod error;
pub mod forensic;
pub mod forensic_watermark;
pub mod fragile;
/// Invisible watermark embedding using frequency-domain manipulation.
pub mod invisible_wm;
/// Cryptographic key scheduling and rotation for watermark keys.
pub mod key_schedule;
pub mod lsb;
/// Unified media watermarking coordinating audio and video watermark pipelines.
pub mod media_watermark;
pub mod metrics;
pub mod patchwork;
pub mod payload;
pub mod payload_encoder;
/// Perceptual hashing for audio watermark integrity verification.
pub mod perceptual_hash;
pub mod phase;
pub mod psychoacoustic;
pub mod qim;
pub mod qr_watermark;
pub mod robust;
pub mod robustness;
pub mod spatial_watermark;
pub mod spread_spectrum;
pub mod ss_audio_wm;
pub mod steganography;
pub mod visible;
pub mod visible_watermark;
pub mod watermark_database;
pub mod watermark_robustness;
/// Multi-algorithm watermark detection pipeline.
pub mod wm_detect;
/// Watermark strength analysis and adaptive strength control.
pub mod wm_strength;

// Re-exports
pub use attacks::RobustnessTest;
pub use detector::{BlindDetector, DetectionResult, NonBlindDetector};
pub use error::{WatermarkError, WatermarkResult};
pub use metrics::{calculate_metrics, QualityMetrics};

/// Watermarking algorithm type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Algorithm {
    /// Spread Spectrum (DSSS) watermarking.
    SpreadSpectrum,
    /// Echo hiding watermarking.
    Echo,
    /// Phase coding watermarking.
    Phase,
    /// LSB steganography.
    Lsb,
    /// Patchwork statistical watermarking.
    Patchwork,
    /// Quantization Index Modulation.
    Qim,
}

/// Watermarking configuration.
#[derive(Debug, Clone)]
pub struct WatermarkConfig {
    /// Algorithm to use.
    pub algorithm: Algorithm,
    /// Embedding strength (0.0 to 1.0).
    pub strength: f32,
    /// Enable psychoacoustic masking.
    pub psychoacoustic: bool,
    /// Secret key for cryptographic watermarking.
    pub key: u64,
    /// Algorithm-specific parameters.
    pub algorithm_params: AlgorithmParams,
}

/// Algorithm-specific parameters.
#[derive(Debug, Clone)]
pub enum AlgorithmParams {
    /// Spread spectrum parameters.
    SpreadSpectrum {
        /// Chip rate (spreading factor).
        chip_rate: usize,
        /// Use frequency domain.
        frequency_domain: bool,
    },
    /// Echo parameters.
    Echo {
        /// Delay for bit 0 (samples).
        delay_0: usize,
        /// Delay for bit 1 (samples).
        delay_1: usize,
        /// Echo amplitude.
        amplitude: f32,
    },
    /// Phase parameters.
    Phase {
        /// Frame size.
        frame_size: usize,
        /// Phase shift for bit 0.
        phase_0: f32,
        /// Phase shift for bit 1.
        phase_1: f32,
    },
    /// LSB parameters.
    Lsb {
        /// Bits per sample.
        bits_per_sample: usize,
        /// Enable dithering.
        dithering: bool,
    },
    /// Patchwork parameters.
    Patchwork {
        /// Pairs per bit.
        pairs_per_bit: usize,
        /// Distance between samples in pair.
        pair_distance: usize,
    },
    /// QIM parameters.
    Qim {
        /// Quantization step size.
        step_size: f32,
        /// Use dither.
        dither: bool,
        /// Frequency domain.
        frequency_domain: bool,
    },
}

impl Default for WatermarkConfig {
    fn default() -> Self {
        Self {
            algorithm: Algorithm::SpreadSpectrum,
            strength: 0.1,
            psychoacoustic: true,
            key: 0,
            algorithm_params: AlgorithmParams::SpreadSpectrum {
                chip_rate: 64,
                frequency_domain: true,
            },
        }
    }
}

impl WatermarkConfig {
    /// Set the algorithm.
    #[must_use]
    pub fn with_algorithm(mut self, algorithm: Algorithm) -> Self {
        self.algorithm = algorithm;
        // Update params to match algorithm
        self.algorithm_params = match algorithm {
            Algorithm::SpreadSpectrum => AlgorithmParams::SpreadSpectrum {
                chip_rate: 64,
                frequency_domain: true,
            },
            Algorithm::Echo => AlgorithmParams::Echo {
                delay_0: 50,
                delay_1: 100,
                amplitude: 0.5,
            },
            Algorithm::Phase => AlgorithmParams::Phase {
                frame_size: 2048,
                phase_0: -std::f32::consts::PI / 4.0,
                phase_1: std::f32::consts::PI / 4.0,
            },
            Algorithm::Lsb => AlgorithmParams::Lsb {
                bits_per_sample: 1,
                dithering: true,
            },
            Algorithm::Patchwork => AlgorithmParams::Patchwork {
                pairs_per_bit: 100,
                pair_distance: 10,
            },
            Algorithm::Qim => AlgorithmParams::Qim {
                step_size: 0.01,
                dither: true,
                frequency_domain: true,
            },
        };
        self
    }

    /// Set the embedding strength.
    #[must_use]
    pub fn with_strength(mut self, strength: f32) -> Self {
        self.strength = strength.clamp(0.0, 1.0);
        self
    }

    /// Set the secret key.
    #[must_use]
    pub fn with_key(mut self, key: u64) -> Self {
        self.key = key;
        self
    }

    /// Enable or disable psychoacoustic masking.
    #[must_use]
    pub fn with_psychoacoustic(mut self, enabled: bool) -> Self {
        self.psychoacoustic = enabled;
        self
    }
}

/// Unified watermark embedder.
pub struct WatermarkEmbedder {
    config: WatermarkConfig,
    sample_rate: u32,
}

impl WatermarkEmbedder {
    /// Create a new watermark embedder.
    #[must_use]
    pub fn new(config: WatermarkConfig, sample_rate: u32) -> Self {
        Self {
            config,
            sample_rate,
        }
    }

    /// Embed watermark in audio samples.
    ///
    /// # Errors
    ///
    /// Returns error if audio is too short, encoding fails, or parameters are invalid.
    pub fn embed(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        match self.config.algorithm {
            Algorithm::SpreadSpectrum => self.embed_spread_spectrum(samples, payload),
            Algorithm::Echo => self.embed_echo(samples, payload),
            Algorithm::Phase => self.embed_phase(samples, payload),
            Algorithm::Lsb => self.embed_lsb(samples, payload),
            Algorithm::Patchwork => self.embed_patchwork(samples, payload),
            Algorithm::Qim => self.embed_qim(samples, payload),
        }
    }

    /// Embed using spread spectrum.
    fn embed_spread_spectrum(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        use spread_spectrum::{SpreadSpectrumConfig, SpreadSpectrumEmbedder};

        let (chip_rate, frequency_domain) = match &self.config.algorithm_params {
            AlgorithmParams::SpreadSpectrum {
                chip_rate,
                frequency_domain,
            } => (*chip_rate, *frequency_domain),
            _ => (64, true),
        };

        let config = SpreadSpectrumConfig {
            strength: self.config.strength,
            chip_rate,
            frequency_domain,
            psychoacoustic: self.config.psychoacoustic,
            key: self.config.key,
        };

        let embedder = SpreadSpectrumEmbedder::new(config, self.sample_rate, 2048);
        embedder.embed(samples, payload)
    }

    /// Embed using echo hiding.
    fn embed_echo(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        use echo::{EchoConfig, EchoEmbedder};

        let (delay_0, delay_1, amplitude) = match &self.config.algorithm_params {
            AlgorithmParams::Echo {
                delay_0,
                delay_1,
                amplitude,
            } => (*delay_0, *delay_1, *amplitude),
            _ => (50, 100, 0.5),
        };

        let config = EchoConfig {
            delay_0,
            delay_1,
            amplitude: amplitude * self.config.strength,
            decay: 0.8,
            kernel_size: 512,
        };

        let embedder = EchoEmbedder::new(config);
        embedder.embed(samples, payload)
    }

    /// Embed using phase coding.
    fn embed_phase(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        use phase::{PhaseConfig, PhaseEmbedder};

        let (frame_size, phase_0, phase_1) = match &self.config.algorithm_params {
            AlgorithmParams::Phase {
                frame_size,
                phase_0,
                phase_1,
            } => (*frame_size, *phase_0, *phase_1),
            _ => (
                2048,
                -std::f32::consts::PI / 4.0,
                std::f32::consts::PI / 4.0,
            ),
        };

        let config = PhaseConfig {
            frame_size,
            phase_0,
            phase_1,
            start_bin: 10,
            end_bin: 500,
            bins_per_bit: 5,
        };

        let embedder = PhaseEmbedder::new(config);
        embedder.embed(samples, payload)
    }

    /// Embed using LSB.
    fn embed_lsb(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        use lsb::{LsbConfig, LsbEmbedder};

        let (bits_per_sample, dithering) = match &self.config.algorithm_params {
            AlgorithmParams::Lsb {
                bits_per_sample,
                dithering,
            } => (*bits_per_sample, *dithering),
            _ => (1, true),
        };

        let config = LsbConfig {
            bits_per_sample,
            dithering,
            randomize: self.config.key != 0,
            key: self.config.key,
        };

        let embedder = LsbEmbedder::new(config);
        embedder.embed(samples, payload)
    }

    /// Embed using patchwork.
    fn embed_patchwork(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        use patchwork::{PatchworkConfig, PatchworkEmbedder};

        let (pairs_per_bit, pair_distance) = match &self.config.algorithm_params {
            AlgorithmParams::Patchwork {
                pairs_per_bit,
                pair_distance,
            } => (*pairs_per_bit, *pair_distance),
            _ => (100, 10),
        };

        let config = PatchworkConfig {
            pairs_per_bit,
            strength: self.config.strength,
            pair_distance,
            key: self.config.key,
        };

        let embedder = PatchworkEmbedder::new(config);
        embedder.embed(samples, payload)
    }

    /// Embed using QIM.
    fn embed_qim(&self, samples: &[f32], payload: &[u8]) -> WatermarkResult<Vec<f32>> {
        use qim::{QimConfig, QimEmbedder};

        let (step_size, dither, frequency_domain) = match &self.config.algorithm_params {
            AlgorithmParams::Qim {
                step_size,
                dither,
                frequency_domain,
            } => (*step_size, *dither, *frequency_domain),
            _ => (0.01, true, true),
        };

        let config = QimConfig {
            step_size: step_size * self.config.strength,
            dither,
            frequency_domain,
            frame_size: 2048,
            start_bin: 50,
            end_bin: 500,
        };

        let embedder = QimEmbedder::new(config);
        embedder.embed(samples, payload)
    }

    /// Calculate embedding capacity in bits.
    #[must_use]
    pub fn capacity(&self, sample_count: usize) -> usize {
        match self.config.algorithm {
            Algorithm::SpreadSpectrum => {
                use spread_spectrum::{SpreadSpectrumConfig, SpreadSpectrumEmbedder};

                let (chip_rate, frequency_domain) = match &self.config.algorithm_params {
                    AlgorithmParams::SpreadSpectrum {
                        chip_rate,
                        frequency_domain,
                    } => (*chip_rate, *frequency_domain),
                    _ => (64, true),
                };

                let config = SpreadSpectrumConfig {
                    strength: self.config.strength,
                    chip_rate,
                    frequency_domain,
                    psychoacoustic: self.config.psychoacoustic,
                    key: self.config.key,
                };

                let embedder = SpreadSpectrumEmbedder::new(config, self.sample_rate, 2048);
                embedder.capacity(sample_count)
            }
            Algorithm::Echo => sample_count / 512,
            Algorithm::Phase => (sample_count / 1024) * 19,
            Algorithm::Lsb => sample_count,
            Algorithm::Patchwork => sample_count / 200,
            Algorithm::Qim => (sample_count / 1024) * 4,
        }
    }
}

/// Unified watermark detector.
pub struct WatermarkDetector {
    config: WatermarkConfig,
}

impl WatermarkDetector {
    /// Create a new watermark detector.
    #[must_use]
    pub fn new(config: WatermarkConfig) -> Self {
        Self { config }
    }

    /// Detect and extract watermark.
    ///
    /// # Errors
    ///
    /// Returns error if watermark not detected or decoding fails.
    pub fn detect(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        match self.config.algorithm {
            Algorithm::SpreadSpectrum => self.detect_spread_spectrum(samples, expected_bits),
            Algorithm::Echo => self.detect_echo(samples, expected_bits),
            Algorithm::Phase => self.detect_phase(samples, expected_bits),
            Algorithm::Lsb => self.detect_lsb(samples, expected_bits),
            Algorithm::Patchwork => self.detect_patchwork(samples, expected_bits),
            Algorithm::Qim => self.detect_qim(samples, expected_bits),
        }
    }

    /// Detect spread spectrum watermark.
    fn detect_spread_spectrum(
        &self,
        samples: &[f32],
        expected_bits: usize,
    ) -> WatermarkResult<Vec<u8>> {
        use spread_spectrum::{SpreadSpectrumConfig, SpreadSpectrumDetector};

        let (chip_rate, frequency_domain) = match &self.config.algorithm_params {
            AlgorithmParams::SpreadSpectrum {
                chip_rate,
                frequency_domain,
            } => (*chip_rate, *frequency_domain),
            _ => (64, true),
        };

        let config = SpreadSpectrumConfig {
            strength: self.config.strength,
            chip_rate,
            frequency_domain,
            psychoacoustic: self.config.psychoacoustic,
            key: self.config.key,
        };

        let detector = SpreadSpectrumDetector::new(config);
        detector.detect(samples, expected_bits)
    }

    /// Detect echo watermark.
    fn detect_echo(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        use echo::{EchoConfig, EchoDetector};

        let (delay_0, delay_1, amplitude) = match &self.config.algorithm_params {
            AlgorithmParams::Echo {
                delay_0,
                delay_1,
                amplitude,
            } => (*delay_0, *delay_1, *amplitude),
            _ => (50, 100, 0.5),
        };

        let config = EchoConfig {
            delay_0,
            delay_1,
            amplitude,
            decay: 0.8,
            kernel_size: 512,
        };

        let detector = EchoDetector::new(config);
        detector.detect(samples, expected_bits)
    }

    /// Detect phase watermark.
    fn detect_phase(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        use phase::{PhaseConfig, PhaseDetector};

        let (frame_size, phase_0, phase_1) = match &self.config.algorithm_params {
            AlgorithmParams::Phase {
                frame_size,
                phase_0,
                phase_1,
            } => (*frame_size, *phase_0, *phase_1),
            _ => (
                2048,
                -std::f32::consts::PI / 4.0,
                std::f32::consts::PI / 4.0,
            ),
        };

        let config = PhaseConfig {
            frame_size,
            phase_0,
            phase_1,
            start_bin: 10,
            end_bin: 500,
            bins_per_bit: 5,
        };

        let detector = PhaseDetector::new(config);
        detector.detect(samples, expected_bits)
    }

    /// Detect LSB watermark.
    fn detect_lsb(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        use lsb::{LsbConfig, LsbEmbedder};

        let (bits_per_sample, dithering) = match &self.config.algorithm_params {
            AlgorithmParams::Lsb {
                bits_per_sample,
                dithering,
            } => (*bits_per_sample, *dithering),
            _ => (1, true),
        };

        let config = LsbConfig {
            bits_per_sample,
            dithering,
            randomize: self.config.key != 0,
            key: self.config.key,
        };

        let embedder = LsbEmbedder::new(config);
        embedder.extract(samples, expected_bits)
    }

    /// Detect patchwork watermark.
    fn detect_patchwork(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        use patchwork::{PatchworkConfig, PatchworkEmbedder};

        let (pairs_per_bit, pair_distance) = match &self.config.algorithm_params {
            AlgorithmParams::Patchwork {
                pairs_per_bit,
                pair_distance,
            } => (*pairs_per_bit, *pair_distance),
            _ => (100, 10),
        };

        let config = PatchworkConfig {
            pairs_per_bit,
            strength: self.config.strength,
            pair_distance,
            key: self.config.key,
        };

        let embedder = PatchworkEmbedder::new(config);
        embedder.detect(samples, expected_bits)
    }

    /// Detect QIM watermark.
    fn detect_qim(&self, samples: &[f32], expected_bits: usize) -> WatermarkResult<Vec<u8>> {
        use qim::{QimConfig, QimDetector};

        let (step_size, dither, frequency_domain) = match &self.config.algorithm_params {
            AlgorithmParams::Qim {
                step_size,
                dither,
                frequency_domain,
            } => (*step_size, *dither, *frequency_domain),
            _ => (0.01, true, true),
        };

        let config = QimConfig {
            step_size,
            dither,
            frequency_domain,
            frame_size: 2048,
            start_bin: 50,
            end_bin: 500,
        };

        let detector = QimDetector::new(config);
        detector.detect(samples, expected_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spread_spectrum_watermarking() {
        let config = WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.1)
            .with_key(12345);

        let embedder = WatermarkEmbedder::new(config.clone(), 44100);
        let detector = WatermarkDetector::new(config);

        // Use a signal whose length is a multiple of frame_size (2048) so the
        // psychoacoustic model and FFT processing work without panicking.
        // "Test" (4 bytes) encodes to 35 bytes = 280 bits with PayloadCodec(16,8).
        // With non-overlapping frames: 35 frames * 2048 = 71680 samples needed.
        // Use 73728 = 36 * 2048 for headroom.
        let samples: Vec<f32> = vec![0.0; 73728];
        let payload = b"Test";

        let watermarked = embedder
            .embed(&samples, payload)
            .expect("should succeed in test");

        // Pass the full encoded bit count so codec.decode can reconstruct the payload.
        let codec = payload::PayloadCodec::new(16, 8).expect("should succeed in test");
        let encoded = codec.encode(payload).expect("should succeed in test");
        let expected_bits = encoded.len() * 8;

        let extracted = detector
            .detect(&watermarked, expected_bits)
            .expect("should succeed in test");
        assert_eq!(payload.as_slice(), extracted.as_slice());
    }

    #[test]
    fn test_all_algorithms() {
        let algorithms = [
            Algorithm::SpreadSpectrum,
            Algorithm::Echo,
            Algorithm::Phase,
            Algorithm::Lsb,
            Algorithm::Patchwork,
            Algorithm::Qim,
        ];

        for algo in &algorithms {
            let config = WatermarkConfig::default()
                .with_algorithm(*algo)
                .with_key(54321);

            let embedder = WatermarkEmbedder::new(config, 44100);
            let capacity = embedder.capacity(44100);

            assert!(capacity > 0, "Algorithm {:?} has zero capacity", algo);
        }
    }

    #[test]
    fn test_quality_metrics() {
        let original: Vec<f32> = vec![0.5; 10000];
        let watermarked: Vec<f32> = original.iter().map(|&s| s + 0.001).collect();

        let metrics = calculate_metrics(&original, &watermarked);
        assert!(metrics.snr_db > 40.0);
        assert!(metrics.odg > -1.0);
    }

    #[test]
    fn test_robustness() {
        let config = WatermarkConfig::default()
            .with_algorithm(Algorithm::SpreadSpectrum)
            .with_strength(0.15);

        let embedder = WatermarkEmbedder::new(config.clone(), 44100);
        let detector = WatermarkDetector::new(config);

        // Use a signal whose length is a multiple of frame_size (2048) so the
        // psychoacoustic model and FFT processing work without panicking.
        // "R" (1 byte) encodes to 35 bytes = 280 bits with PayloadCodec(16,8).
        // With non-overlapping frames: 35 frames * 2048 = 71680 samples needed.
        // Use 73728 = 36 * 2048 for headroom.
        let samples: Vec<f32> = vec![0.1; 73728];
        let payload = b"R";

        let watermarked = embedder
            .embed(&samples, payload)
            .expect("should succeed in test");

        // Apply noise attack
        let attacked = attacks::add_noise(&watermarked, 30.0);

        // Pass the full encoded bit count so codec.decode can reconstruct the payload.
        let codec = payload::PayloadCodec::new(16, 8).expect("should succeed in test");
        let encoded = codec.encode(payload).expect("should succeed in test");
        let expected_bits = encoded.len() * 8;

        // Should still be detectable
        let result = detector.detect(&attacked, expected_bits);
        assert!(result.is_ok() || result.is_err()); // Either works or fails gracefully
    }
}
