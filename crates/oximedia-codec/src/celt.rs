//! Standalone CELT frame decoding types and scaffolding.
//!
//! CELT (Constrained Energy Lapped Transform) is the music/wideband codec
//! used within Opus. This module provides lightweight frame-level types for
//! energy bookkeeping and the MDCT-IV inverse transform, independent of the
//! full Opus decoder pipeline found in `crate::opus::celt`.

use std::f32::consts::PI;

/// CELT band-edge positions in MDCT bins for a 48 kHz, 20 ms (960-sample) frame.
///
/// There are 21 bands bounded by 22 edges. The first edge is bin 0 and the
/// last is bin 100 (equivalent to 10 kHz at 48 kHz).
pub const CELT_BANDS: [usize; 22] = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100,
];

/// Number of CELT frequency bands.
const NUM_BANDS: usize = 21;

/// Configuration for a CELT frame decoder.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CeltFrameConfig {
    /// Frame size in samples (120, 240, 480, 960, or 1920).
    pub frame_size: usize,
    /// Number of audio channels (1 or 2).
    pub channels: u8,
    /// First band index (for hybrid Opus mode).
    pub start_band: u8,
    /// One-past-last band index.
    pub end_band: u8,
}

impl Default for CeltFrameConfig {
    /// Default configuration: 960-sample stereo frame covering all 21 bands.
    fn default() -> Self {
        Self {
            frame_size: 960,
            channels: 2,
            start_band: 0,
            end_band: NUM_BANDS as u8,
        }
    }
}

impl CeltFrameConfig {
    /// Sets the frame size and returns `self` for chaining.
    pub fn with_frame_size(mut self, frame_size: usize) -> Self {
        self.frame_size = frame_size;
        self
    }

    /// Sets the channel count and returns `self` for chaining.
    pub fn with_channels(mut self, channels: u8) -> Self {
        self.channels = channels;
        self
    }

    /// Sets the start band and returns `self` for chaining.
    pub fn with_start_band(mut self, start_band: u8) -> Self {
        self.start_band = start_band;
        self
    }

    /// Sets the end band and returns `self` for chaining.
    pub fn with_end_band(mut self, end_band: u8) -> Self {
        self.end_band = end_band;
        self
    }
}

/// Per-band log-domain energy for a CELT frame.
///
/// All 21 CELT bands are tracked. Energy values are in the log domain
/// (natural logarithm of linear energy) as used internally by CELT.
#[derive(Debug, Clone)]
pub struct CeltEnergy {
    /// Per-band energy values (log domain), indexed 0..21.
    pub bands: [f32; NUM_BANDS],
}

impl CeltEnergy {
    /// Creates a `CeltEnergy` with all bands initialised to zero.
    pub fn new() -> Self {
        Self {
            bands: [0.0f32; NUM_BANDS],
        }
    }

    /// Returns the energy for the given band index.
    ///
    /// Returns `0.0` for out-of-range indices.
    pub fn energy(&self, band: usize) -> f32 {
        if band < NUM_BANDS {
            self.bands[band]
        } else {
            0.0
        }
    }

    /// Sets the energy for the given band index.
    ///
    /// Out-of-range indices are silently ignored.
    pub fn set_energy(&mut self, band: usize, val: f32) {
        if band < NUM_BANDS {
            self.bands[band] = val;
        }
    }
}

/// A decoded CELT frame.
#[derive(Debug, Clone)]
pub struct CeltFrame {
    /// Frame configuration.
    pub config: CeltFrameConfig,
    /// Per-band energy decoded from the bitstream.
    pub energy: CeltEnergy,
    /// Bitmask of collapsed (zeroed) bands.
    pub collapsed_mask: u32,
    /// Decoded output samples.
    pub samples: Vec<f32>,
}

impl CeltFrame {
    /// Creates a new zeroed `CeltFrame` for the given configuration.
    pub fn new(config: CeltFrameConfig) -> Self {
        let sample_count = config.frame_size;
        Self {
            config,
            energy: CeltEnergy::new(),
            collapsed_mask: 0,
            samples: vec![0.0f32; sample_count],
        }
    }

    /// Returns the number of samples in this frame (per channel).
    pub fn sample_count(&self) -> usize {
        self.config.frame_size
    }
}

/// CELT frame decoder scaffold.
///
/// Provides energy decoding and the MDCT-IV inverse transform. Full
/// entropy-coded CELT decoding (PVQ, fine energy, band prediction …) is
/// provided by `crate::opus::celt`; this type is intentionally lightweight
/// and intended for testing and scaffolding.
#[derive(Debug)]
pub struct CeltDecoder {
    /// Frame configuration.
    pub config: CeltFrameConfig,
    /// Per-band energy state carried across frames.
    pub prev_energy: CeltEnergy,
}

impl CeltDecoder {
    /// Creates a new `CeltDecoder` for the given configuration.
    pub fn new(config: CeltFrameConfig) -> Self {
        Self {
            config,
            prev_energy: CeltEnergy::new(),
        }
    }

    /// Parses energy values from the first bytes of `data` and returns a
    /// `CeltFrame` with zeroed samples.
    ///
    /// Each active band contributes one byte to the energy encoding: the byte
    /// is interpreted as a signed i8 and scaled by 1/16 to produce a
    /// log-domain energy value. Bands beyond the end of `data` default to
    /// `0.0`.
    ///
    /// Full PVQ coefficient decoding is not performed; this method is a
    /// scaffold.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<CeltFrame, String> {
        let mut frame = CeltFrame::new(self.config.clone());

        let start = self.config.start_band as usize;
        let end = self.config.end_band as usize;

        for band in start..end {
            let band_idx = band - start;
            let energy = if band_idx < data.len() {
                let raw = data[band_idx] as i8;
                raw as f32 / 16.0
            } else {
                0.0
            };
            // Delta from previous frame (simple inter-frame prediction).
            let predicted = self.prev_energy.energy(band);
            let new_energy = predicted + energy;
            frame.energy.set_energy(band, new_energy);
            self.prev_energy.set_energy(band, new_energy);
        }

        Ok(frame)
    }

    /// Computes the Type-IV MDCT inverse transform (IMDCT-IV).
    ///
    /// Given `N = coeffs.len()` spectral coefficients `X[k]`, the output
    /// time-domain samples are:
    ///
    /// ```text
    /// x[n] = sqrt(2/N) * sum_{k=0}^{N-1} X[k] * cos(π/N * (n + 0.5 + N/2) * (k + 0.5))
    /// ```
    ///
    /// for `n = 0 .. N-1`.
    ///
    /// Returns an empty vector if `coeffs` is empty.
    pub fn apply_mdct_inverse(&self, coeffs: &[f32]) -> Vec<f32> {
        let n = coeffs.len();
        if n == 0 {
            return Vec::new();
        }

        let scale = (2.0f32 / n as f32).sqrt();
        let mut output = vec![0.0f32; n];

        for (idx, out) in output.iter_mut().enumerate() {
            let n_f = n as f32;
            let nn = idx as f32 + 0.5 + n_f / 2.0;
            let mut acc = 0.0f32;
            for (k, &coeff) in coeffs.iter().enumerate() {
                let kk = k as f32 + 0.5;
                acc += coeff * (PI / n_f * nn * kk).cos();
            }
            *out = scale * acc;
        }

        output
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_celt_bands_has_22_elements() {
        assert_eq!(CELT_BANDS.len(), 22, "21 bands require 22 edges");
    }

    #[test]
    fn test_celt_bands_starts_at_zero() {
        assert_eq!(CELT_BANDS[0], 0);
    }

    #[test]
    fn test_celt_bands_monotonically_increasing() {
        for i in 1..CELT_BANDS.len() {
            assert!(
                CELT_BANDS[i] > CELT_BANDS[i - 1],
                "CELT_BANDS[{}] = {} should be > CELT_BANDS[{}] = {}",
                i,
                CELT_BANDS[i],
                i - 1,
                CELT_BANDS[i - 1]
            );
        }
    }

    #[test]
    fn test_celt_frame_config_default() {
        let cfg = CeltFrameConfig::default();
        assert_eq!(cfg.frame_size, 960);
        assert_eq!(cfg.channels, 2);
        assert_eq!(cfg.start_band, 0);
        assert_eq!(cfg.end_band, 21);
    }

    #[test]
    fn test_celt_frame_config_builder() {
        let cfg = CeltFrameConfig::default()
            .with_frame_size(480)
            .with_channels(1)
            .with_start_band(2)
            .with_end_band(18);
        assert_eq!(cfg.frame_size, 480);
        assert_eq!(cfg.channels, 1);
        assert_eq!(cfg.start_band, 2);
        assert_eq!(cfg.end_band, 18);
    }

    #[test]
    fn test_celt_energy_new_all_zero() {
        let energy = CeltEnergy::new();
        for band in 0..NUM_BANDS {
            assert_eq!(energy.energy(band), 0.0);
        }
    }

    #[test]
    fn test_celt_energy_set_and_get() {
        let mut energy = CeltEnergy::new();
        energy.set_energy(5, 3.14);
        assert!((energy.energy(5) - 3.14).abs() < 1e-6);
    }

    #[test]
    fn test_celt_energy_out_of_range() {
        let mut energy = CeltEnergy::new();
        // Out-of-range set should not panic.
        energy.set_energy(100, 99.0);
        // Out-of-range get should return 0.
        assert_eq!(energy.energy(100), 0.0);
    }

    #[test]
    fn test_celt_frame_sample_count() {
        let cfg = CeltFrameConfig::default().with_frame_size(480);
        let frame = CeltFrame::new(cfg);
        assert_eq!(frame.sample_count(), 480);
    }

    #[test]
    fn test_celt_frame_sample_count_960() {
        let cfg = CeltFrameConfig::default();
        let frame = CeltFrame::new(cfg);
        assert_eq!(frame.sample_count(), 960);
    }

    #[test]
    fn test_celt_decoder_new() {
        let cfg = CeltFrameConfig::default();
        let dec = CeltDecoder::new(cfg.clone());
        assert_eq!(dec.config, cfg);
    }

    #[test]
    fn test_celt_decoder_decode_frame_returns_correct_size() {
        let cfg = CeltFrameConfig::default().with_frame_size(480);
        let mut dec = CeltDecoder::new(cfg);
        let data = vec![0u8; 21];
        let frame = dec.decode_frame(&data).expect("should succeed");
        assert_eq!(frame.sample_count(), 480);
    }

    #[test]
    fn test_celt_decoder_decode_frame_zero_data_zero_energy() {
        let cfg = CeltFrameConfig::default();
        let mut dec = CeltDecoder::new(cfg);
        let data = vec![0u8; 21];
        let frame = dec.decode_frame(&data).expect("should succeed");
        for band in 0..NUM_BANDS {
            assert_eq!(frame.energy.energy(band), 0.0);
        }
    }

    #[test]
    fn test_celt_decoder_apply_mdct_inverse_all_zero_input() {
        let cfg = CeltFrameConfig::default();
        let dec = CeltDecoder::new(cfg);
        // All-zero coefficients must produce all-zero output.
        let coeffs = vec![0.0f32; 16];
        let output = dec.apply_mdct_inverse(&coeffs);
        assert_eq!(output.len(), 16);
        for &sample in &output {
            assert!(sample.abs() < 1e-6, "expected zero, got {}", sample);
        }
    }

    #[test]
    fn test_celt_decoder_apply_mdct_inverse_empty_input() {
        let cfg = CeltFrameConfig::default();
        let dec = CeltDecoder::new(cfg);
        let output = dec.apply_mdct_inverse(&[]);
        assert!(output.is_empty());
    }

    #[test]
    fn test_celt_decoder_apply_mdct_inverse_nonzero() {
        let cfg = CeltFrameConfig::default();
        let dec = CeltDecoder::new(cfg);
        // Single non-zero DC coefficient should yield non-zero output.
        let mut coeffs = vec![0.0f32; 8];
        coeffs[0] = 1.0;
        let output = dec.apply_mdct_inverse(&coeffs);
        assert_eq!(output.len(), 8);
        let any_nonzero = output.iter().any(|&s| s.abs() > 1e-6);
        assert!(any_nonzero, "IMDCT of non-zero input must not be all-zero");
    }
}
