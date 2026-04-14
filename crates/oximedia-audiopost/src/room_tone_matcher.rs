#![allow(dead_code)]
//! Room tone matching for audio post-production.
//!
//! Matches the ambient noise ("room tone") of a recorded segment so that gaps,
//! edits, and ADR inserts blend seamlessly with the production track.
//!
//! # Workflow
//!
//! 1. [`RoomToneAnalyzer`] measures a reference room-tone sample to build a
//!    statistical spectral profile (mean + variance per bin).
//! 2. [`RoomToneSynthesizer`] generates new room-tone samples whose long-term
//!    spectral envelope matches the reference profile.
//! 3. [`RoomToneMatcher`] wraps both and exposes the end-to-end API:
//!    - `analyze()` — ingest reference samples
//!    - `synthesize()` — produce fill material of any requested length
//!    - `apply_fade()` — crossfade synthesized fill into a gap

use crate::error::{AudioPostError, AudioPostResult};
use oxifft::{fft, ifft, Complex};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DEFAULT_FFT_SIZE: usize = 2048;
const OVERLAP_FACTOR: usize = 4; // 75 % overlap for overlap-add synthesis
const LCG_A: u64 = 6_364_136_223_846_793_005;
const LCG_C: u64 = 1_442_695_040_888_963_407;

// ---------------------------------------------------------------------------
// Minimal LCG pseudo-random generator (no external rand dependency)
// ---------------------------------------------------------------------------

/// Simple 64-bit LCG PRNG used internally for phase randomisation.
#[derive(Debug, Clone)]
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed ^ 0xDEAD_BEEF_CAFE_1234 }
    }

    /// Returns the next value in [0, 1).
    fn next_f32(&mut self) -> f32 {
        self.state = self.state.wrapping_mul(LCG_A).wrapping_add(LCG_C);
        // Use upper 23 bits for mantissa
        let bits = (self.state >> 41) as u32;
        bits as f32 / (1u32 << 23) as f32
    }

    /// Returns the next value in [-π, π).
    fn next_phase(&mut self) -> f32 {
        (self.next_f32() * 2.0 - 1.0) * std::f32::consts::PI
    }
}

// ---------------------------------------------------------------------------
// Spectral profile
// ---------------------------------------------------------------------------

/// Per-bin magnitude statistics measured from a room-tone reference.
#[derive(Debug, Clone)]
pub struct SpectralProfile {
    /// Mean magnitude per FFT bin (half-spectrum, length `fft_size / 2 + 1`).
    pub mean_magnitude: Vec<f32>,
    /// Variance of magnitude per FFT bin.
    pub variance_magnitude: Vec<f32>,
    /// FFT size used during analysis.
    pub fft_size: usize,
    /// Number of frames that contributed to the profile.
    pub frame_count: usize,
    /// RMS level of the reference material (linear).
    pub rms_level: f32,
}

impl SpectralProfile {
    fn new(fft_size: usize) -> Self {
        let bins = fft_size / 2 + 1;
        Self {
            mean_magnitude: vec![0.0; bins],
            variance_magnitude: vec![0.0; bins],
            fft_size,
            frame_count: 0,
            rms_level: 0.0,
        }
    }

    /// Returns the number of spectral bins.
    #[must_use]
    pub fn bin_count(&self) -> usize {
        self.mean_magnitude.len()
    }

    /// Return the dominant noise floor level in dBFS (−inf for silence).
    #[must_use]
    pub fn noise_floor_dbfs(&self) -> f32 {
        if self.rms_level <= 0.0 {
            return f32::NEG_INFINITY;
        }
        20.0 * self.rms_level.log10()
    }
}

// ---------------------------------------------------------------------------
// RoomToneAnalyzer
// ---------------------------------------------------------------------------

/// Analyses a room-tone reference to build a [`SpectralProfile`].
#[derive(Debug)]
pub struct RoomToneAnalyzer {
    fft_size: usize,
    hop_size: usize,
    /// Accumulator for the online mean calculation.
    mean_acc: Vec<f64>,
    /// Accumulator for the Welford online variance (M2).
    m2_acc: Vec<f64>,
    frame_count: usize,
    /// Hann window for each frame.
    window: Vec<f32>,
    /// Running sum of squared samples for RMS.
    rms_sum_sq: f64,
    rms_sample_count: usize,
}

impl RoomToneAnalyzer {
    /// Create a new analyser.
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero or `fft_size` is not a power of two ≥ 64.
    pub fn new(sample_rate: u32, fft_size: usize) -> AudioPostResult<Self> {
        if sample_rate == 0 {
            return Err(AudioPostError::InvalidSampleRate(sample_rate));
        }
        if !fft_size.is_power_of_two() || fft_size < 64 {
            return Err(AudioPostError::InvalidBufferSize(fft_size));
        }
        let hop_size = fft_size / OVERLAP_FACTOR;
        let window = hann_window(fft_size);
        let bins = fft_size / 2 + 1;
        Ok(Self {
            fft_size,
            hop_size,
            mean_acc: vec![0.0; bins],
            m2_acc: vec![0.0; bins],
            frame_count: 0,
            window,
            rms_sum_sq: 0.0,
            rms_sample_count: 0,
        })
    }

    /// Ingest reference room-tone samples.  Can be called multiple times to
    /// accumulate statistics over multiple sections.
    pub fn ingest(&mut self, samples: &[f32]) {
        // Update RMS accumulators
        for &s in samples {
            self.rms_sum_sq += (s as f64) * (s as f64);
        }
        self.rms_sample_count += samples.len();

        // Slide over the input with hop_size steps
        let mut pos = 0usize;
        while pos + self.fft_size <= samples.len() {
            let frame = &samples[pos..pos + self.fft_size];
            let complex_frame: Vec<Complex<f32>> = frame
                .iter()
                .zip(self.window.iter())
                .map(|(&s, &w)| Complex::new(s * w, 0.0))
                .collect();

            let spectrum = fft(&complex_frame);

            // Welford online update for mean and variance
            for (i, item) in spectrum.iter().take(self.fft_size / 2 + 1).enumerate() {
                let mag = item.norm() as f64;
                let n = self.frame_count as f64 + 1.0;
                let delta = mag - self.mean_acc[i];
                self.mean_acc[i] += delta / n;
                let delta2 = mag - self.mean_acc[i];
                self.m2_acc[i] += delta * delta2;
            }

            self.frame_count += 1;
            pos += self.hop_size;
        }
    }

    /// Finalise and return the computed [`SpectralProfile`].
    ///
    /// # Errors
    ///
    /// Returns an error if no frames have been analysed yet.
    pub fn finish(&self) -> AudioPostResult<SpectralProfile> {
        if self.frame_count == 0 {
            return Err(AudioPostError::Generic(
                "No frames analysed — call ingest() with sufficient audio first".to_string(),
            ));
        }
        let bins = self.fft_size / 2 + 1;
        let mut profile = SpectralProfile::new(self.fft_size);
        profile.frame_count = self.frame_count;

        for i in 0..bins {
            profile.mean_magnitude[i] = self.mean_acc[i] as f32;
            profile.variance_magnitude[i] = if self.frame_count > 1 {
                (self.m2_acc[i] / (self.frame_count - 1) as f64) as f32
            } else {
                0.0
            };
        }

        profile.rms_level = if self.rms_sample_count > 0 {
            ((self.rms_sum_sq / self.rms_sample_count as f64).sqrt()) as f32
        } else {
            0.0
        };

        Ok(profile)
    }

    /// Reset all accumulated state.
    pub fn reset(&mut self) {
        for v in self.mean_acc.iter_mut() {
            *v = 0.0;
        }
        for v in self.m2_acc.iter_mut() {
            *v = 0.0;
        }
        self.frame_count = 0;
        self.rms_sum_sq = 0.0;
        self.rms_sample_count = 0;
    }
}

// ---------------------------------------------------------------------------
// RoomToneSynthesizer
// ---------------------------------------------------------------------------

/// Synthesizes room-tone fill material from a [`SpectralProfile`].
///
/// Uses an overlap-add (OLA) method: for each output hop we synthesize a
/// spectral frame whose magnitude matches the profile and whose phase is
/// randomised, then IFFT and OLA-accumulate into the output buffer.
#[derive(Debug)]
pub struct RoomToneSynthesizer {
    fft_size: usize,
    hop_size: usize,
    window: Vec<f32>,
    /// OLA accumulation buffer (length `fft_size`).
    ola_buf: Vec<f32>,
    /// Write cursor into ola_buf (in hops).
    rng: Lcg,
}

impl RoomToneSynthesizer {
    /// Create a synthesizer from a spectral profile.
    ///
    /// # Errors
    ///
    /// Returns an error if the profile's `fft_size` is not a power of two ≥ 64.
    pub fn new(profile: &SpectralProfile, seed: u64) -> AudioPostResult<Self> {
        let fft_size = profile.fft_size;
        if !fft_size.is_power_of_two() || fft_size < 64 {
            return Err(AudioPostError::InvalidBufferSize(fft_size));
        }
        let hop_size = fft_size / OVERLAP_FACTOR;
        let window = hann_window(fft_size);
        let ola_buf = vec![0.0f32; fft_size + hop_size];
        Ok(Self {
            fft_size,
            hop_size,
            window,
            ola_buf,
            rng: Lcg::new(seed),
        })
    }

    /// Synthesize exactly `n_samples` of room-tone fill matching `profile`.
    ///
    /// The returned buffer will have exactly `n_samples` elements.
    pub fn synthesize(&mut self, profile: &SpectralProfile, n_samples: usize) -> Vec<f32> {
        if n_samples == 0 {
            return Vec::new();
        }

        // OLA normalisation factor (Hann window, 75% overlap → sum of windows² = 1.5 per sample)
        // Pre-computed normalisation: for Hann OLA with 75 % overlap the gain is 1.5.
        let ola_gain = 1.5_f32;

        let mut output = vec![0.0f32; n_samples];
        let bins = self.fft_size / 2 + 1;
        let mut written = 0usize;

        while written < n_samples {
            // Build a complex spectrum: magnitude from profile, random phase
            let mut spectrum = vec![Complex::new(0.0f32, 0.0); self.fft_size];
            for i in 0..bins {
                let mag = profile.mean_magnitude[i].max(0.0);
                let phase = self.rng.next_phase();
                spectrum[i] = Complex::new(mag * phase.cos(), mag * phase.sin());
                // Mirror for real IFFT symmetry (skip DC and Nyquist)
                if i > 0 && i < bins - 1 {
                    spectrum[self.fft_size - i] =
                        Complex::new(mag * phase.cos(), -mag * phase.sin());
                }
            }

            let time_frame = ifft(&spectrum);
            // Apply synthesis window and OLA-add into output
            let frames_needed = (n_samples - written + self.hop_size - 1) / self.hop_size;
            let _ = frames_needed; // used implicitly below

            let copy_start = written;
            let copy_end = (written + self.fft_size).min(n_samples);
            for (j, out_sample) in output[copy_start..copy_end].iter_mut().enumerate() {
                let s = time_frame[j].re * self.window[j] / ola_gain;
                *out_sample += s;
            }

            written += self.hop_size;
        }

        // Normalise to profile RMS level
        if profile.rms_level > 0.0 {
            let cur_rms = {
                let sq: f32 = output.iter().map(|&x| x * x).sum();
                if output.is_empty() {
                    0.0
                } else {
                    (sq / output.len() as f32).sqrt()
                }
            };
            if cur_rms > 1e-12 {
                let scale = profile.rms_level / cur_rms;
                for s in output.iter_mut() {
                    *s *= scale;
                }
            }
        }

        output
    }
}

// ---------------------------------------------------------------------------
// RoomToneMatcher
// ---------------------------------------------------------------------------

/// End-to-end room-tone match processor.
///
/// Analyses a reference, then synthesizes fill of arbitrary length.
#[derive(Debug)]
pub struct RoomToneMatcher {
    analyzer: RoomToneAnalyzer,
    profile: Option<SpectralProfile>,
    sample_rate: u32,
    fft_size: usize,
}

impl RoomToneMatcher {
    /// Create a new matcher.
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero.
    pub fn new(sample_rate: u32) -> AudioPostResult<Self> {
        Self::with_fft_size(sample_rate, DEFAULT_FFT_SIZE)
    }

    /// Create a new matcher with a custom FFT size.
    ///
    /// # Errors
    ///
    /// Returns an error if `sample_rate` is zero or `fft_size` is invalid.
    pub fn with_fft_size(sample_rate: u32, fft_size: usize) -> AudioPostResult<Self> {
        let analyzer = RoomToneAnalyzer::new(sample_rate, fft_size)?;
        Ok(Self {
            analyzer,
            profile: None,
            sample_rate,
            fft_size,
        })
    }

    /// Ingest reference room-tone samples.
    pub fn analyze(&mut self, samples: &[f32]) {
        self.analyzer.ingest(samples);
    }

    /// Finalise the spectral profile from ingested reference audio.
    ///
    /// Must be called before [`synthesize`].
    ///
    /// # Errors
    ///
    /// Returns an error if no reference audio has been ingested.
    pub fn build_profile(&mut self) -> AudioPostResult<&SpectralProfile> {
        let profile = self.analyzer.finish()?;
        self.profile = Some(profile);
        Ok(self.profile.as_ref().expect("just set above"))
    }

    /// Return the current profile if available.
    #[must_use]
    pub fn profile(&self) -> Option<&SpectralProfile> {
        self.profile.as_ref()
    }

    /// Synthesize `n_samples` of room-tone fill.
    ///
    /// # Errors
    ///
    /// Returns an error if no profile has been built yet.
    pub fn synthesize(&self, n_samples: usize, seed: u64) -> AudioPostResult<Vec<f32>> {
        let profile = self
            .profile
            .as_ref()
            .ok_or_else(|| AudioPostError::Generic("build_profile() not called".to_string()))?;
        let mut synth = RoomToneSynthesizer::new(profile, seed)?;
        Ok(synth.synthesize(profile, n_samples))
    }

    /// Apply a linear crossfade between `main` and `fill` over `fade_samples`.
    ///
    /// This is useful for blending synthesised room tone into a gap.
    /// `main` is assumed to already be at the splice point — fade-out from the
    /// end of `main`, fade-in from the beginning of `fill`.
    ///
    /// # Errors
    ///
    /// Returns an error if `fade_samples` exceeds the length of either slice.
    pub fn apply_crossfade(
        main: &mut [f32],
        fill: &mut [f32],
        fade_samples: usize,
    ) -> AudioPostResult<()> {
        if fade_samples > main.len() || fade_samples > fill.len() {
            return Err(AudioPostError::InvalidBufferSize(fade_samples));
        }
        let start_main = main.len() - fade_samples;
        for i in 0..fade_samples {
            let t = i as f32 / fade_samples.max(1) as f32;
            main[start_main + i] *= 1.0 - t;
            fill[i] *= t;
        }
        Ok(())
    }

    /// Sample rate this matcher was configured with.
    #[must_use]
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }
}

// ---------------------------------------------------------------------------
// Hann window helper
// ---------------------------------------------------------------------------

fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| {
            0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / (size - 1) as f32).cos())
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn white_noise(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Lcg::new(seed);
        (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect()
    }

    #[test]
    fn test_analyzer_basic() {
        let mut a = RoomToneAnalyzer::new(48000, 2048).unwrap();
        let noise = white_noise(4096, 42);
        a.ingest(&noise);
        let profile = a.finish().unwrap();
        assert_eq!(profile.fft_size, 2048);
        assert!(profile.frame_count > 0);
        assert!(profile.rms_level > 0.0);
    }

    #[test]
    fn test_analyzer_invalid_sample_rate() {
        assert!(RoomToneAnalyzer::new(0, 2048).is_err());
    }

    #[test]
    fn test_analyzer_invalid_fft_size() {
        assert!(RoomToneAnalyzer::new(48000, 300).is_err()); // not power of two
    }

    #[test]
    fn test_analyzer_empty_error() {
        let a = RoomToneAnalyzer::new(48000, 2048).unwrap();
        assert!(a.finish().is_err());
    }

    #[test]
    fn test_synthesizer_length() {
        let mut a = RoomToneAnalyzer::new(48000, 2048).unwrap();
        let noise = white_noise(8192, 99);
        a.ingest(&noise);
        let profile = a.finish().unwrap();
        let mut synth = RoomToneSynthesizer::new(&profile, 7).unwrap();
        let output = synth.synthesize(&profile, 4800);
        assert_eq!(output.len(), 4800);
    }

    #[test]
    fn test_matcher_end_to_end() {
        let mut matcher = RoomToneMatcher::new(48000).unwrap();
        let noise = white_noise(8192, 1);
        matcher.analyze(&noise);
        matcher.build_profile().unwrap();
        let fill = matcher.synthesize(2400, 42).unwrap();
        assert_eq!(fill.len(), 2400);
        // Fill must not be all-zero
        let energy: f32 = fill.iter().map(|&x| x * x).sum();
        assert!(energy > 1e-6, "Synthesized fill should have non-zero energy");
    }

    #[test]
    fn test_matcher_no_profile_error() {
        let matcher = RoomToneMatcher::new(48000).unwrap();
        assert!(matcher.synthesize(1024, 0).is_err());
    }

    #[test]
    fn test_crossfade_modifies_edges() {
        let mut main = vec![1.0f32; 512];
        let mut fill = vec![1.0f32; 512];
        RoomToneMatcher::apply_crossfade(&mut main, &mut fill, 128).unwrap();
        // Last sample of fade-out region of main should be near 0
        assert!(main[511] < 0.1, "End of main should fade to near zero");
        // Last sample of fade-in region of fill should be near 1
        assert!(fill[127] > 0.9, "End of fade-in should be near 1");
    }

    #[test]
    fn test_crossfade_too_long_error() {
        let mut main = vec![0.0f32; 100];
        let mut fill = vec![0.0f32; 100];
        assert!(RoomToneMatcher::apply_crossfade(&mut main, &mut fill, 200).is_err());
    }

    #[test]
    fn test_noise_floor_silence() {
        let profile = SpectralProfile::new(2048);
        assert_eq!(profile.noise_floor_dbfs(), f32::NEG_INFINITY);
    }
}
