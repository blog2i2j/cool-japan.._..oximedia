//! Audio fingerprinting and recognition.
//!
//! This module implements audio fingerprinting for content identification and matching.
//! The system extracts robust features from audio that remain recognizable even with noise,
//! compression, and other distortions.
//!
//! # Algorithm
//!
//! The fingerprinting algorithm is inspired by Shazam and works as follows:
//!
//! 1. **Spectrogram Analysis**: Compute short-time Fourier transform (STFT) of audio
//! 2. **Peak Extraction**: Identify prominent spectral peaks in time-frequency domain
//! 3. **Constellation Map**: Create a sparse representation of peaks
//! 4. **Hash Generation**: Pair peaks to create combinatorial hashes
//! 5. **Database Storage**: Store hashes with time offsets
//! 6. **Matching**: Find candidate matches and verify using time-offset alignment
//!
//! # Example
//!
//! ```rust,no_run
//! use oximedia_audio::fingerprint::{Fingerprinter, FingerprintConfig, FingerprintDatabase};
//! use oximedia_audio::AudioFrame;
//!
//! // Create fingerprinter
//! let config = FingerprintConfig::default();
//! let fingerprinter = Fingerprinter::new(config)?;
//!
//! // Generate fingerprint from audio
//! # let audio = AudioFrame::new(
//! #     oximedia_core::SampleFormat::F32,
//! #     44100,
//! #     oximedia_audio::ChannelLayout::Mono
//! # );
//! let fingerprint = fingerprinter.generate(&audio)?;
//!
//! // Create database and add fingerprint
//! let mut database = FingerprintDatabase::new();
//! database.add_fingerprint("song_id", fingerprint);
//!
//! // Match against database
//! # let query = AudioFrame::new(
//! #     oximedia_core::SampleFormat::F32,
//! #     44100,
//! #     oximedia_audio::ChannelLayout::Mono
//! # );
//! let query_fp = fingerprinter.generate(&query)?;
//! let matches = database.find_matches(&query_fp, 0.3);
//!
//! for m in matches {
//!     println!("Match: {} (confidence: {:.2}, offset: {:.2}s)",
//!         m.track_id, m.confidence, m.time_offset);
//! }
//! ```
//!
//! # Features
//!
//! - **Robust**: Survives noise, compression, pitch shifts, tempo changes
//! - **Fast**: Sub-second matching against large databases
//! - **Scalable**: Efficient hash-based lookup
//! - **Configurable**: Adjustable sensitivity and accuracy tradeoffs

pub mod constellation;
pub mod database;
pub mod extract;
pub mod hash;
#[allow(clippy::module_inception)]
pub mod matching;

use crate::error::AudioResult;
use crate::AudioFrame;

pub use constellation::{ConstellationMap, Peak as ConstellationPeak};
pub use database::{FingerprintDatabase, Match};
pub use extract::SpectrogramExtractor;
pub use hash::{Hash, HashGenerator};
pub use matching::FingerprintMatcher;

/// Audio fingerprint.
///
/// Contains hashes and temporal information that uniquely identify audio content.
#[derive(Clone, Debug)]
pub struct Fingerprint {
    /// Hash values with temporal offsets.
    pub hashes: Vec<(Hash, f64)>,
    /// Sample rate of source audio.
    pub sample_rate: u32,
    /// Duration of source audio in seconds.
    pub duration: f64,
    /// Configuration used to generate fingerprint.
    pub config: FingerprintConfig,
}

impl Fingerprint {
    /// Create a new fingerprint.
    #[must_use]
    pub fn new(
        hashes: Vec<(Hash, f64)>,
        sample_rate: u32,
        duration: f64,
        config: FingerprintConfig,
    ) -> Self {
        Self {
            hashes,
            sample_rate,
            duration,
            config,
        }
    }

    /// Get number of hashes.
    #[must_use]
    pub fn hash_count(&self) -> usize {
        self.hashes.len()
    }

    /// Get hash density (hashes per second).
    #[must_use]
    pub fn density(&self) -> f64 {
        if self.duration > 0.0 {
            self.hash_count() as f64 / self.duration
        } else {
            0.0
        }
    }

    /// Check if fingerprint is valid.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        !self.hashes.is_empty() && self.sample_rate > 0 && self.duration > 0.0
    }

    /// Get time range covered by fingerprint.
    #[must_use]
    pub fn time_range(&self) -> (f64, f64) {
        if self.hashes.is_empty() {
            return (0.0, 0.0);
        }

        let min_time = self
            .hashes
            .iter()
            .map(|(_, t)| t)
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        let max_time = self
            .hashes
            .iter()
            .map(|(_, t)| t)
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(0.0);

        (min_time, max_time)
    }

    /// Merge multiple fingerprints (for long audio streams).
    #[must_use]
    pub fn merge(fingerprints: &[Self]) -> Option<Self> {
        if fingerprints.is_empty() {
            return None;
        }

        let first = &fingerprints[0];
        let mut merged_hashes = Vec::new();
        let mut total_duration = 0.0;

        for fp in fingerprints {
            let time_offset = total_duration;
            for (hash, time) in &fp.hashes {
                merged_hashes.push((*hash, time + time_offset));
            }
            total_duration += fp.duration;
        }

        Some(Self {
            hashes: merged_hashes,
            sample_rate: first.sample_rate,
            duration: total_duration,
            config: first.config.clone(),
        })
    }
}

/// Fingerprinting configuration.
#[derive(Clone, Debug)]
pub struct FingerprintConfig {
    /// FFT size for spectrogram (must be power of 2).
    pub fft_size: usize,
    /// Hop size (overlap) between FFT windows.
    pub hop_size: usize,
    /// Frequency range for peak extraction (Hz).
    pub freq_range: (f64, f64),
    /// Number of frequency bands for peak detection.
    pub num_bands: usize,
    /// Peak threshold (relative to local maximum, 0-1).
    pub peak_threshold: f64,
    /// Maximum peaks per time frame.
    pub max_peaks_per_frame: usize,
    /// Target zone size for hash pairing (time frames).
    pub target_zone_size: usize,
    /// Target zone offset (time frames).
    pub target_zone_offset: usize,
    /// Number of target peaks to pair with each anchor.
    pub num_targets_per_anchor: usize,
    /// Minimum time difference between anchor and target (seconds).
    pub min_time_delta: f64,
    /// Maximum time difference between anchor and target (seconds).
    pub max_time_delta: f64,
}

impl FingerprintConfig {
    /// Create configuration optimized for music.
    #[must_use]
    pub fn music() -> Self {
        Self {
            fft_size: 4096,
            hop_size: 2048,
            freq_range: (40.0, 8000.0),
            num_bands: 32,
            peak_threshold: 0.7,
            max_peaks_per_frame: 5,
            target_zone_size: 32,
            target_zone_offset: 1,
            num_targets_per_anchor: 5,
            min_time_delta: 0.0,
            max_time_delta: 2.0,
        }
    }

    /// Create configuration optimized for speech.
    #[must_use]
    pub fn speech() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 512,
            freq_range: (80.0, 4000.0),
            num_bands: 24,
            peak_threshold: 0.6,
            max_peaks_per_frame: 4,
            target_zone_size: 24,
            target_zone_offset: 1,
            num_targets_per_anchor: 4,
            min_time_delta: 0.0,
            max_time_delta: 1.5,
        }
    }

    /// Create high-sensitivity configuration.
    #[must_use]
    pub fn high_sensitivity() -> Self {
        Self {
            fft_size: 4096,
            hop_size: 1024,
            freq_range: (20.0, 10000.0),
            num_bands: 48,
            peak_threshold: 0.5,
            max_peaks_per_frame: 8,
            target_zone_size: 48,
            target_zone_offset: 1,
            num_targets_per_anchor: 8,
            min_time_delta: 0.0,
            max_time_delta: 3.0,
        }
    }

    /// Create fast configuration (fewer hashes, faster but less robust).
    #[must_use]
    pub fn fast() -> Self {
        Self {
            fft_size: 2048,
            hop_size: 2048,
            freq_range: (100.0, 5000.0),
            num_bands: 16,
            peak_threshold: 0.8,
            max_peaks_per_frame: 3,
            target_zone_size: 16,
            target_zone_offset: 2,
            num_targets_per_anchor: 3,
            min_time_delta: 0.1,
            max_time_delta: 1.0,
        }
    }

    /// Validate configuration.
    pub fn validate(&self) -> AudioResult<()> {
        use crate::error::AudioError;

        if !self.fft_size.is_power_of_two() {
            return Err(AudioError::InvalidParameter(
                "FFT size must be power of 2".to_string(),
            ));
        }

        if self.hop_size == 0 || self.hop_size > self.fft_size {
            return Err(AudioError::InvalidParameter("Invalid hop size".to_string()));
        }

        if self.freq_range.0 >= self.freq_range.1 {
            return Err(AudioError::InvalidParameter(
                "Invalid frequency range".to_string(),
            ));
        }

        if self.num_bands == 0 {
            return Err(AudioError::InvalidParameter(
                "Number of bands must be > 0".to_string(),
            ));
        }

        if !(0.0..=1.0).contains(&self.peak_threshold) {
            return Err(AudioError::InvalidParameter(
                "Peak threshold must be in range [0, 1]".to_string(),
            ));
        }

        Ok(())
    }
}

impl Default for FingerprintConfig {
    fn default() -> Self {
        Self::music()
    }
}

/// Main fingerprinting engine.
pub struct Fingerprinter {
    config: FingerprintConfig,
    extractor: SpectrogramExtractor,
    hash_generator: HashGenerator,
}

impl Fingerprinter {
    /// Create a new fingerprinter with the given configuration.
    pub fn new(config: FingerprintConfig) -> AudioResult<Self> {
        config.validate()?;

        let extractor = SpectrogramExtractor::new(
            config.fft_size,
            config.hop_size,
            config.freq_range,
            config.num_bands,
        );

        let hash_generator = HashGenerator::new(
            config.target_zone_size,
            config.target_zone_offset,
            config.num_targets_per_anchor,
        );

        Ok(Self {
            config,
            extractor,
            hash_generator,
        })
    }

    /// Generate fingerprint from audio frame.
    pub fn generate(&self, audio: &AudioFrame) -> AudioResult<Fingerprint> {
        // Extract spectrogram and peaks
        let constellation = self.extractor.extract(audio, &self.config)?;

        // Generate hashes from peak pairs
        let hashes = self.hash_generator.generate(&constellation);

        Ok(Fingerprint::new(
            hashes,
            audio.sample_rate,
            audio.duration_seconds(),
            self.config.clone(),
        ))
    }

    /// Generate fingerprint from multiple audio frames (streaming).
    pub fn generate_streaming(&self, frames: &[AudioFrame]) -> AudioResult<Fingerprint> {
        let fingerprints: Result<Vec<_>, _> =
            frames.iter().map(|frame| self.generate(frame)).collect();

        let fingerprints = fingerprints?;

        Fingerprint::merge(&fingerprints).ok_or_else(|| {
            crate::error::AudioError::InvalidData("No frames to fingerprint".to_string())
        })
    }

    /// Get configuration.
    #[must_use]
    pub const fn config(&self) -> &FingerprintConfig {
        &self.config
    }

    /// Estimate processing time for given audio duration.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_processing_time(&self, duration_seconds: f64) -> f64 {
        // Rough estimate: fingerprinting is typically 10-20x faster than realtime
        duration_seconds / 15.0
    }

    /// Estimate number of hashes for given audio duration.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_hash_count(&self, duration_seconds: f64, sample_rate: u32) -> usize {
        let num_frames = (duration_seconds * f64::from(sample_rate)) / self.config.hop_size as f64;
        let hashes_per_frame = self.config.max_peaks_per_frame * self.config.num_targets_per_anchor;
        (num_frames * hashes_per_frame as f64) as usize
    }
}
