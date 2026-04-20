//! Short-Time Fourier Transform (STFT) and Inverse STFT (ISTFT).
//!
//! The STFT slices a signal into overlapping frames, applies a window function,
//! and transforms each frame to the frequency domain via FFT. The ISTFT
//! reconstructs the time-domain signal from a sequence of complex spectra using
//! the overlap-add (OLA) method with optional window normalization.
//!
//! # Features
//!
//! - Configurable FFT size, hop size, and window type
//! - Built-in window functions: Hann, Hamming, Blackman, Rectangular
//! - Per-frame magnitude and phase spectra
//! - ISTFT with OLA and normalization
//! - Magnitude spectrogram helper
//!
//! # Example
//!
//! ```rust
//! use oximedia_audio::stft::{Stft, StftConfig, WindowType};
//!
//! let config = StftConfig::new(512, 128, WindowType::Hann);
//! let mut stft = Stft::new(config);
//!
//! let signal: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.01).sin()).collect();
//! let frames = stft.forward(&signal);
//! assert!(!frames.is_empty());
//!
//! // Each frame has `fft_size / 2 + 1` complex bins.
//! assert_eq!(frames[0].len(), 257);
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::f32::consts::PI;

use oxifft::api::{Direction, Flags, Plan};
use oxifft::Complex;

use crate::error::{AudioError, AudioResult};

// ── Window functions ──────────────────────────────────────────────────────────

/// Window function applied to each STFT frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WindowType {
    /// Rectangular (no windowing) — best frequency resolution, high spectral leakage.
    Rectangular,
    /// Hann (raised-cosine) — low spectral leakage, moderate frequency resolution.
    Hann,
    /// Hamming window — slightly lower sidelobe level than Hann.
    Hamming,
    /// Blackman window — very low sidelobe level, lower frequency resolution.
    Blackman,
}

/// Generate a window vector of length `n` for the chosen type.
#[must_use]
fn make_window(kind: WindowType, n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    match kind {
        WindowType::Rectangular => vec![1.0_f32; n],
        WindowType::Hann => (0..n)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / n as f32).cos()))
            .collect(),
        WindowType::Hamming => (0..n)
            .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / n as f32).cos())
            .collect(),
        WindowType::Blackman => (0..n)
            .map(|i| {
                let x = 2.0 * PI * i as f32 / n as f32;
                0.42 - 0.5 * x.cos() + 0.08 * (2.0 * x).cos()
            })
            .collect(),
    }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration parameters for [`Stft`].
#[derive(Debug, Clone)]
pub struct StftConfig {
    /// FFT size (must be a power of two for best performance).
    pub fft_size: usize,
    /// Hop size (step between successive frames), in samples.
    pub hop_size: usize,
    /// Window function applied to each frame before the FFT.
    pub window: WindowType,
}

impl StftConfig {
    /// Create a new STFT configuration.
    ///
    /// # Panics
    ///
    /// Panics if `fft_size` or `hop_size` are zero, or if `hop_size > fft_size`.
    #[must_use]
    pub fn new(fft_size: usize, hop_size: usize, window: WindowType) -> Self {
        assert!(fft_size > 0, "fft_size must be non-zero");
        assert!(hop_size > 0, "hop_size must be non-zero");
        assert!(
            hop_size <= fft_size,
            "hop_size ({hop_size}) must not exceed fft_size ({fft_size})"
        );
        Self { fft_size, hop_size, window }
    }

    /// Number of complex output bins per frame: `fft_size / 2 + 1`.
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }
}

impl Default for StftConfig {
    fn default() -> Self {
        Self::new(1024, 256, WindowType::Hann)
    }
}

// ── STFT ──────────────────────────────────────────────────────────────────────

/// A complex spectral frame: `fft_size / 2 + 1` complex values (f32).
pub type SpectralFrame = Vec<Complex<f64>>;

/// Short-Time Fourier Transform processor.
///
/// Holds a pre-computed window and an FFT plan so that repeated calls to
/// [`Stft::forward`] reuse allocations.
pub struct Stft {
    config: StftConfig,
    window: Vec<f32>,
    fft_plan: Plan<f64>,
}

impl Stft {
    /// Create a new STFT processor with the given configuration.
    ///
    /// Returns an error if the FFT plan cannot be created (e.g., `fft_size` is zero).
    /// Under normal usage `StftConfig::new` already enforces `fft_size > 0`.
    pub fn try_new(config: StftConfig) -> AudioResult<Self> {
        let window = make_window(config.window, config.fft_size);
        let fft_plan =
            Plan::<f64>::dft_1d(config.fft_size, Direction::Forward, Flags::MEASURE)
                .or_else(|| Plan::<f64>::dft_1d(config.fft_size, Direction::Forward, Flags::ESTIMATE))
                .ok_or_else(|| AudioError::InvalidParameter(format!(
                    "Failed to create FFT plan for fft_size={}",
                    config.fft_size
                )))?;
        Ok(Self { config, window, fft_plan })
    }

    /// Create a new STFT processor with the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if the FFT plan cannot be created. This only happens when
    /// `fft_size` is zero, which `StftConfig::new` already rejects via assert.
    #[must_use]
    pub fn new(config: StftConfig) -> Self {
        Self::try_new(config).unwrap_or_else(|e| {
            panic!("Stft::new failed (use try_new for recoverable error handling): {e}")
        })
    }

    /// Returns a reference to the active configuration.
    #[must_use]
    pub fn config(&self) -> &StftConfig {
        &self.config
    }

    /// Compute the forward STFT of `signal`.
    ///
    /// Returns a `Vec` of spectral frames.  Each frame contains
    /// `config.num_bins()` complex values covering the non-negative
    /// frequency bins `[0, fft_size/2]`.
    ///
    /// The last frame is produced with zero-padding if the signal does not
    /// align perfectly to a whole number of hops.
    #[must_use]
    pub fn forward(&mut self, signal: &[f32]) -> Vec<SpectralFrame> {
        if signal.is_empty() {
            return Vec::new();
        }

        let n = self.config.fft_size;
        let hop = self.config.hop_size;
        let num_bins = self.config.num_bins();

        // Allocate working buffers once.
        let mut frame_buf = vec![Complex::new(0.0_f64, 0.0); n];
        let mut out_buf = vec![Complex::new(0.0_f64, 0.0); n];
        let mut frames = Vec::new();

        let mut start = 0usize;
        // Produce at least one frame even for very short signals.
        loop {
            // Fill frame with windowed samples (zero-pad if past end of signal).
            for (k, dst) in frame_buf.iter_mut().enumerate() {
                let sample_idx = start + k;
                let sample = signal.get(sample_idx).copied().unwrap_or(0.0) as f64;
                let win = self.window.get(k).copied().unwrap_or(1.0) as f64;
                *dst = Complex::new(sample * win, 0.0);
            }

            // Execute FFT.
            self.fft_plan.execute(&mut frame_buf, &mut out_buf);

            // Keep only the non-negative frequency bins.
            let spectral: SpectralFrame = out_buf[..num_bins].to_vec();
            frames.push(spectral);

            start += hop;
            if start >= signal.len() {
                break;
            }
        }

        frames
    }

    /// Compute the magnitude spectrogram.
    ///
    /// Returns a 2-D matrix `[time_frames][num_bins]` of magnitudes in linear
    /// scale.
    #[must_use]
    pub fn magnitude_spectrogram(&mut self, signal: &[f32]) -> Vec<Vec<f32>> {
        self.forward(signal)
            .into_iter()
            .map(|frame| frame.iter().map(|c| c.norm() as f32).collect())
            .collect()
    }

    /// Compute the power spectrogram (magnitude squared).
    #[must_use]
    pub fn power_spectrogram(&mut self, signal: &[f32]) -> Vec<Vec<f32>> {
        self.forward(signal)
            .into_iter()
            .map(|frame| frame.iter().map(|c| c.norm_sqr() as f32).collect())
            .collect()
    }
}

// ── ISTFT ─────────────────────────────────────────────────────────────────────

/// Inverse Short-Time Fourier Transform using overlap-add.
///
/// Reconstructs a time-domain signal from a sequence of complex spectral
/// frames produced by [`Stft::forward`].  The output length is determined by
/// the number of frames and the hop size.
pub struct Istft {
    config: StftConfig,
    window: Vec<f32>,
    ifft_plan: Plan<f64>,
}

impl Istft {
    /// Create a new ISTFT processor matching the given configuration.
    ///
    /// Returns an error if the inverse FFT plan cannot be created (e.g., `fft_size` is zero).
    pub fn try_new(config: StftConfig) -> AudioResult<Self> {
        let window = make_window(config.window, config.fft_size);
        let ifft_plan =
            Plan::<f64>::dft_1d(config.fft_size, Direction::Backward, Flags::MEASURE)
                .or_else(|| Plan::<f64>::dft_1d(config.fft_size, Direction::Backward, Flags::ESTIMATE))
                .ok_or_else(|| AudioError::InvalidParameter(format!(
                    "Failed to create IFFT plan for fft_size={}",
                    config.fft_size
                )))?;
        Ok(Self { config, window, ifft_plan })
    }

    /// Create a new ISTFT processor matching the given configuration.
    ///
    /// # Panics
    ///
    /// Panics if the inverse FFT plan cannot be created. This only happens when
    /// `fft_size` is zero, which `StftConfig::new` already rejects via assert.
    #[must_use]
    pub fn new(config: StftConfig) -> Self {
        Self::try_new(config).unwrap_or_else(|e| {
            panic!("Istft::new failed (use try_new for recoverable error handling): {e}")
        })
    }

    /// Reconstruct the time-domain signal from spectral frames.
    ///
    /// Uses the overlap-add method.  Normalizes by the sum-of-squares of the
    /// window function to compensate for repeated windowing.
    ///
    /// `frames` must contain the same number of bins as `config.num_bins()`.
    #[must_use]
    pub fn inverse(&mut self, frames: &[SpectralFrame]) -> Vec<f32> {
        if frames.is_empty() {
            return Vec::new();
        }

        let n = self.config.fft_size;
        let hop = self.config.hop_size;
        let num_bins = self.config.num_bins();

        // Determine output length.
        let out_len = (frames.len() - 1) * hop + n;
        let mut output = vec![0.0_f64; out_len];
        let mut norm = vec![0.0_f64; out_len];

        // Reconstruction buffer.
        let mut complex_in = vec![Complex::new(0.0_f64, 0.0); n];
        let mut complex_out = vec![Complex::new(0.0_f64, 0.0); n];

        for (frame_idx, frame) in frames.iter().enumerate() {
            // Reconstruct the full spectrum from the one-sided representation.
            // Bin 0 and the Nyquist bin are their own conjugates; all others
            // are mirrored.
            for k in 0..num_bins.min(frame.len()) {
                complex_in[k] = frame[k];
            }
            for k in num_bins..n {
                let mirror = n - k;
                complex_in[k] = if mirror < num_bins && mirror < frame.len() {
                    let c = frame[mirror];
                    Complex::new(c.re, -c.im)
                } else {
                    Complex::new(0.0, 0.0)
                };
            }

            // Inverse FFT.
            self.ifft_plan.execute(&mut complex_in, &mut complex_out);

            // Overlap-add with window and scale by 1/N (FFTW convention).
            let start = frame_idx * hop;
            for k in 0..n {
                let win = self.window.get(k).copied().unwrap_or(1.0) as f64;
                let sample = complex_out[k].re / n as f64;
                output[start + k] += sample * win;
                norm[start + k] += win * win;
            }
        }

        // Normalize by the window envelope.
        output
            .iter()
            .zip(norm.iter())
            .map(|(&s, &w)| {
                if w > 1e-12 {
                    (s / w) as f32
                } else {
                    0.0
                }
            })
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sine_signal(freq_hz: f32, sample_rate: u32, num_samples: usize) -> Vec<f32> {
        let sr = sample_rate as f32;
        (0..num_samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn test_forward_produces_frames() {
        let cfg = StftConfig::new(512, 128, WindowType::Hann);
        let mut stft = Stft::new(cfg);
        let signal = sine_signal(440.0, 44_100, 2048);
        let frames = stft.forward(&signal);
        assert!(!frames.is_empty(), "should produce at least one frame");
    }

    #[test]
    fn test_frame_bin_count() {
        let fft_size = 1024usize;
        let cfg = StftConfig::new(fft_size, 256, WindowType::Hann);
        let mut stft = Stft::new(cfg);
        let signal = sine_signal(1000.0, 48_000, 4096);
        let frames = stft.forward(&signal);
        for frame in &frames {
            assert_eq!(
                frame.len(),
                fft_size / 2 + 1,
                "each frame must have fft_size/2+1 bins"
            );
        }
    }

    #[test]
    fn test_window_types_all_produce_output() {
        let signal: Vec<f32> = (0..2048).map(|i| (i as f32 * 0.05).sin()).collect();
        for wt in [
            WindowType::Rectangular,
            WindowType::Hann,
            WindowType::Hamming,
            WindowType::Blackman,
        ] {
            let cfg = StftConfig::new(512, 256, wt);
            let mut stft = Stft::new(cfg);
            let frames = stft.forward(&signal);
            assert!(!frames.is_empty(), "{wt:?} should produce frames");
        }
    }

    #[test]
    fn test_empty_signal_returns_empty_frames() {
        let cfg = StftConfig::new(512, 128, WindowType::Hann);
        let mut stft = Stft::new(cfg);
        let frames = stft.forward(&[]);
        assert!(frames.is_empty(), "empty signal should yield no frames");
    }

    #[test]
    fn test_magnitude_spectrogram_no_negatives() {
        let cfg = StftConfig::new(512, 256, WindowType::Hann);
        let mut stft = Stft::new(cfg);
        let signal = sine_signal(440.0, 44_100, 4096);
        let mag = stft.magnitude_spectrogram(&signal);
        for row in &mag {
            for &v in row {
                assert!(v >= 0.0, "magnitudes must be non-negative");
                assert!(v.is_finite(), "magnitudes must be finite");
            }
        }
    }

    #[test]
    fn test_power_spectrogram_is_magnitude_squared() {
        let cfg = StftConfig::new(256, 64, WindowType::Hann);
        let mut stft1 = Stft::new(cfg.clone());
        let mut stft2 = Stft::new(cfg);
        let signal = sine_signal(220.0, 44_100, 1024);

        let mag = stft1.magnitude_spectrogram(&signal);
        let power = stft2.power_spectrogram(&signal);

        assert_eq!(mag.len(), power.len());
        for (m_row, p_row) in mag.iter().zip(power.iter()) {
            assert_eq!(m_row.len(), p_row.len());
            for (&m, &p) in m_row.iter().zip(p_row.iter()) {
                let diff = (m * m - p).abs();
                assert!(diff < 1e-3, "power should equal magnitude squared; diff={diff}");
            }
        }
    }

    #[test]
    fn test_stft_config_num_bins() {
        let cfg = StftConfig::new(1024, 256, WindowType::Hann);
        assert_eq!(cfg.num_bins(), 513);

        let cfg2 = StftConfig::new(512, 128, WindowType::Hamming);
        assert_eq!(cfg2.num_bins(), 257);
    }

    #[test]
    fn test_stft_default_config() {
        let cfg = StftConfig::default();
        assert_eq!(cfg.fft_size, 1024);
        assert_eq!(cfg.hop_size, 256);
        assert_eq!(cfg.window, WindowType::Hann);
    }

    #[test]
    fn test_istft_round_trip_approximation() {
        // ISTFT should produce a signal that is at least the correct length.
        let cfg = StftConfig::new(512, 128, WindowType::Hann);
        let signal = sine_signal(440.0, 44_100, 1024);

        let mut stft = Stft::new(cfg.clone());
        let frames = stft.forward(&signal);
        assert!(!frames.is_empty());

        let mut istft = Istft::new(cfg);
        let reconstructed = istft.inverse(&frames);

        // Reconstructed signal must cover at least the original length.
        assert!(reconstructed.len() >= signal.len());

        // Check that reconstructed samples are finite.
        for &s in &reconstructed {
            assert!(s.is_finite(), "reconstructed sample must be finite");
        }
    }

    #[test]
    fn test_istft_empty_frames() {
        let cfg = StftConfig::new(512, 128, WindowType::Hann);
        let mut istft = Istft::new(cfg);
        let out = istft.inverse(&[]);
        assert!(out.is_empty(), "empty frames should yield empty output");
    }

    #[test]
    fn test_short_signal_at_least_one_frame() {
        // Even a 1-sample signal should produce exactly one frame.
        let cfg = StftConfig::new(512, 256, WindowType::Rectangular);
        let mut stft = Stft::new(cfg);
        let frames = stft.forward(&[0.5]);
        assert_eq!(frames.len(), 1);
    }
}
