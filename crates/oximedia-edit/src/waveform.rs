//! Audio waveform generation for visual editing feedback.
//!
//! Produces compact peak/RMS data that a UI can render as a waveform overview
//! lane for an audio clip.  The generator down-samples a slice of PCM f32
//! samples to a fixed pixel width by splitting the input into equal-length
//! *buckets* and recording the minimum, maximum, and RMS value in each bucket.

/// Per-pixel waveform peak information.
///
/// `min` is in \[-1.0, 0.0\] and `max` is in \[0.0, 1.0\] (clamped from the
/// raw PCM values so that the pair can be drawn symmetrically).
#[derive(Debug, Clone, PartialEq)]
pub struct WaveformPeak {
    /// Minimum (most negative) sample in this pixel column.
    pub min: f32,
    /// Maximum (most positive) sample in this pixel column.
    pub max: f32,
    /// Root-mean-square level in this pixel column.
    pub rms: f32,
}

impl WaveformPeak {
    /// Construct a peak record.
    #[must_use]
    pub fn new(min: f32, max: f32, rms: f32) -> Self {
        Self { min, max, rms }
    }

    /// Returns `true` if the peak represents silence (all values zero).
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.min == 0.0 && self.max == 0.0 && self.rms == 0.0
    }
}

/// Complete waveform data ready for rendering.
#[derive(Debug, Clone)]
pub struct WaveformData {
    /// Number of source samples that map to a single visual pixel.
    pub samples_per_pixel: u32,
    /// One [`WaveformPeak`] per output pixel column.
    ///
    /// The length equals the `width_pixels` passed to the generator.
    ///
    /// Also compatible with the legacy `(min, max)` pair convention:
    /// `peaks[i]` exposes `.min` and `.max` directly.
    pub peaks: Vec<WaveformPeak>,
    /// Total number of source samples that were processed.
    pub total_samples: usize,
}

impl WaveformData {
    /// Return the number of pixel columns.
    #[must_use]
    pub fn width(&self) -> usize {
        self.peaks.len()
    }

    /// Overall peak amplitude across all pixel columns.
    #[must_use]
    pub fn peak_amplitude(&self) -> f32 {
        self.peaks
            .iter()
            .map(|p| p.max.abs().max(p.min.abs()))
            .fold(0.0_f32, f32::max)
    }

    /// Returns `true` if all pixel columns are silent.
    #[must_use]
    pub fn is_silent(&self) -> bool {
        self.peaks.iter().all(WaveformPeak::is_silent)
    }
}

/// Generates [`WaveformData`] from a slice of interleaved or mono f32 PCM.
///
/// # Example
///
/// ```rust
/// use oximedia_edit::waveform::WaveformGenerator;
///
/// let samples: Vec<f32> = (0..4800).map(|i| (i as f32 / 100.0).sin() * 0.5).collect();
/// let data = WaveformGenerator::new(1).generate(&samples, 120);
/// assert_eq!(data.width(), 120);
/// ```
#[derive(Debug, Clone)]
pub struct WaveformGenerator {
    /// Number of audio channels (used to de-interleave before analysis).
    ///
    /// Set to 1 for already-mono data.
    pub channels: u32,
}

impl WaveformGenerator {
    /// Create a generator for the given channel count.
    ///
    /// `channels` must be ≥ 1; it is clamped to 1 if zero is supplied.
    #[must_use]
    pub fn new(channels: u32) -> Self {
        Self {
            channels: channels.max(1),
        }
    }

    /// Generate waveform data for `width_pixels` pixel columns.
    ///
    /// If `audio` is empty or `width_pixels` is zero, returns a silent waveform
    /// of the requested width (or width 1 if width is zero).
    ///
    /// When `channels > 1` the generator mixes down to mono by averaging all
    /// channels before computing peaks.
    #[must_use]
    #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
    pub fn generate(&self, audio: &[f32], width_pixels: u32) -> WaveformData {
        let width = (width_pixels as usize).max(1);

        if audio.is_empty() {
            return WaveformData {
                samples_per_pixel: 0,
                peaks: vec![WaveformPeak::new(0.0, 0.0, 0.0); width],
                total_samples: 0,
            };
        }

        // Mix down to mono if multi-channel.
        let mono: Vec<f32> = if self.channels <= 1 {
            audio.to_vec()
        } else {
            let ch = self.channels as usize;
            let frames = audio.len() / ch;
            (0..frames)
                .map(|f| {
                    let sum: f32 = (0..ch).map(|c| audio[f * ch + c]).sum();
                    sum / ch as f32
                })
                .collect()
        };

        let total_samples = mono.len();
        let samples_per_pixel = ((total_samples as f64) / (width as f64)).ceil() as usize;
        let samples_per_pixel = samples_per_pixel.max(1);

        let mut peaks = Vec::with_capacity(width);

        for col in 0..width {
            let start = col * samples_per_pixel;
            if start >= total_samples {
                peaks.push(WaveformPeak::new(0.0, 0.0, 0.0));
                continue;
            }
            let end = (start + samples_per_pixel).min(total_samples);
            let bucket = &mono[start..end];

            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            let mut sum_sq = 0.0_f64;

            for &s in bucket {
                if s < min_val {
                    min_val = s;
                }
                if s > max_val {
                    max_val = s;
                }
                sum_sq += (s as f64) * (s as f64);
            }

            let rms = ((sum_sq / bucket.len() as f64).sqrt()) as f32;

            // Clamp to [-1, 1] for display sanity
            let min_val = min_val.clamp(-1.0, 0.0);
            let max_val = max_val.clamp(0.0, 1.0);

            peaks.push(WaveformPeak::new(min_val, max_val, rms));
        }

        WaveformData {
            samples_per_pixel: samples_per_pixel as u32,
            peaks,
            total_samples,
        }
    }

    /// Convenience: generate a normalised waveform where `peak_amplitude == 1.0`.
    ///
    /// If the audio is silent the data is returned as-is (no division by zero).
    #[must_use]
    pub fn generate_normalised(&self, audio: &[f32], width_pixels: u32) -> WaveformData {
        let mut data = self.generate(audio, width_pixels);
        let peak = data.peak_amplitude();
        if peak > 1e-12 {
            for p in &mut data.peaks {
                p.min /= peak;
                p.max /= peak;
                p.rms /= peak;
            }
        }
        data
    }
}

impl Default for WaveformGenerator {
    fn default() -> Self {
        Self::new(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(samples: usize, freq_hz: f32, sample_rate: f32) -> Vec<f32> {
        (0..samples)
            .map(|i| (2.0 * PI * freq_hz * i as f32 / sample_rate).sin())
            .collect()
    }

    #[test]
    fn test_generate_width_matches_request() {
        let audio = sine_wave(4800, 440.0, 48000.0);
        let data = WaveformGenerator::new(1).generate(&audio, 120);
        assert_eq!(data.width(), 120);
        assert_eq!(data.peaks.len(), 120);
    }

    #[test]
    fn test_generate_empty_audio_returns_silent() {
        let data = WaveformGenerator::new(1).generate(&[], 80);
        assert!(data.is_silent());
        assert_eq!(data.width(), 80);
    }

    #[test]
    fn test_generate_zero_width_clamps_to_one() {
        let audio = vec![0.5_f32; 100];
        let data = WaveformGenerator::new(1).generate(&audio, 0);
        assert_eq!(data.width(), 1);
    }

    #[test]
    fn test_peak_amplitude_sine() {
        // Full-scale sine wave: peak should be ≈ 1.0
        let audio = sine_wave(48000, 440.0, 48000.0);
        let data = WaveformGenerator::new(1).generate(&audio, 100);
        let peak = data.peak_amplitude();
        assert!(peak > 0.9, "peak too low: {peak}");
        assert!(peak <= 1.0, "peak exceeds 1.0: {peak}");
    }

    #[test]
    fn test_silence_detection() {
        let audio = vec![0.0_f32; 1000];
        let data = WaveformGenerator::new(1).generate(&audio, 50);
        assert!(data.is_silent());
    }

    #[test]
    fn test_stereo_mixdown() {
        // Left = 1.0, right = -1.0 → mixed = 0.0 (silent)
        let audio: Vec<f32> = (0..200)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        let data = WaveformGenerator::new(2).generate(&audio, 10);
        // After mix-down each frame is (1 + -1)/2 = 0 → silent
        assert!(data.is_silent());
    }

    #[test]
    fn test_normalised_peak_is_one() {
        let audio = sine_wave(4800, 440.0, 48000.0);
        let data = WaveformGenerator::new(1).generate_normalised(&audio, 60);
        let peak = data.peak_amplitude();
        // After normalisation the peak should be exactly 1.0 (within float precision)
        assert!((peak - 1.0).abs() < 1e-5, "normalised peak = {peak}");
    }

    #[test]
    fn test_normalised_silent_does_not_divide_by_zero() {
        let audio = vec![0.0_f32; 100];
        let data = WaveformGenerator::new(1).generate_normalised(&audio, 20);
        assert!(data.is_silent());
    }

    #[test]
    fn test_samples_per_pixel_recorded() {
        let audio = vec![0.5_f32; 480];
        let data = WaveformGenerator::new(1).generate(&audio, 48);
        // 480 / 48 = 10 samples per pixel
        assert_eq!(data.samples_per_pixel, 10);
        assert_eq!(data.total_samples, 480);
    }

    #[test]
    fn test_peaks_min_max_sign_convention() {
        // DC offset 0.8 → every sample positive → min clamped to 0.0, max ≈ 0.8
        let audio = vec![0.8_f32; 300];
        let data = WaveformGenerator::new(1).generate(&audio, 10);
        for p in &data.peaks {
            assert_eq!(p.min, 0.0, "min should be clamped to 0 for positive DC");
            assert!((p.max - 0.8).abs() < 1e-5);
        }
    }

    #[test]
    fn test_peaks_negative_dc() {
        // DC offset -0.6 → every sample negative → max clamped to 0.0, min ≈ -0.6
        let audio = vec![-0.6_f32; 300];
        let data = WaveformGenerator::new(1).generate(&audio, 10);
        for p in &data.peaks {
            assert_eq!(p.max, 0.0, "max should be clamped to 0 for negative DC");
            assert!((p.min - (-0.6)).abs() < 1e-5);
        }
    }

    #[test]
    fn test_rms_positive_for_non_silent() {
        let audio = sine_wave(4800, 220.0, 48000.0);
        let data = WaveformGenerator::new(1).generate(&audio, 48);
        let all_positive_rms = data.peaks.iter().all(|p| p.rms > 0.0);
        assert!(all_positive_rms, "RMS should be > 0 for non-silent audio");
    }

    // ── Additional comprehensive tests ──────────────────────────────────────

    #[test]
    fn test_waveform_peak_is_silent_true() {
        let p = WaveformPeak::new(0.0, 0.0, 0.0);
        assert!(p.is_silent());
    }

    #[test]
    fn test_waveform_peak_is_silent_false_nonzero_max() {
        let p = WaveformPeak::new(0.0, 0.1, 0.0);
        assert!(!p.is_silent());
    }

    #[test]
    fn test_waveform_peak_is_silent_false_nonzero_rms() {
        let p = WaveformPeak::new(0.0, 0.0, 0.05);
        assert!(!p.is_silent());
    }

    #[test]
    fn test_waveform_data_width_matches_peaks_len() {
        let gen = WaveformGenerator::default();
        let audio = vec![0.3_f32; 960];
        let data = gen.generate(&audio, 60);
        assert_eq!(data.width(), data.peaks.len());
        assert_eq!(data.width(), 60);
    }

    #[test]
    fn test_waveform_data_peak_amplitude_silent() {
        let gen = WaveformGenerator::new(1);
        let data = gen.generate(&[], 10);
        assert_eq!(data.peak_amplitude(), 0.0);
    }

    #[test]
    fn test_generate_single_pixel_output() {
        let audio = vec![0.5_f32; 100];
        let data = WaveformGenerator::new(1).generate(&audio, 1);
        assert_eq!(data.width(), 1);
        assert!((data.peaks[0].max - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_multichannel_four_channels() {
        // 4-channel audio where all channels sum to zero → silent after mixdown
        let n_frames = 100;
        let audio: Vec<f32> = (0..n_frames * 4)
            .map(|i| match i % 4 {
                0 => 1.0,
                1 => -1.0,
                2 => 0.5,
                3 => -0.5,
                _ => 0.0,
            })
            .collect();
        let data = WaveformGenerator::new(4).generate(&audio, 10);
        // Each frame: (1 + -1 + 0.5 + -0.5) / 4 = 0 → silent
        assert!(data.is_silent(), "4-channel mix should be silent");
    }

    #[test]
    fn test_normalised_clamps_rms_proportionally() {
        // Constant +0.5 signal: max = 0.5, rms = 0.5
        let audio = vec![0.5_f32; 1000];
        let data = WaveformGenerator::new(1).generate_normalised(&audio, 20);
        // After normalisation by peak (0.5), max should be 1.0
        for p in &data.peaks {
            assert!((p.max - 1.0).abs() < 1e-5, "Normalised max should be 1.0");
        }
    }

    #[test]
    fn test_total_samples_matches_input_length_mono() {
        let audio: Vec<f32> = (0..777).map(|i| (i as f32) * 0.001).collect();
        let data = WaveformGenerator::new(1).generate(&audio, 20);
        assert_eq!(data.total_samples, 777);
    }

    #[test]
    fn test_total_samples_after_stereo_mixdown() {
        // Stereo: 200 interleaved samples → 100 frames
        let audio = vec![0.3_f32; 200];
        let data = WaveformGenerator::new(2).generate(&audio, 10);
        // After mixdown we have 100 mono frames
        assert_eq!(data.total_samples, 100);
    }

    #[test]
    fn test_generator_default_is_mono() {
        let gen = WaveformGenerator::default();
        assert_eq!(gen.channels, 1);
    }

    #[test]
    fn test_generator_zero_channels_clamped_to_one() {
        let gen = WaveformGenerator::new(0);
        assert_eq!(gen.channels, 1);
    }
}
