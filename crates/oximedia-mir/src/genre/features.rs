//! Genre-specific feature extraction.

use crate::utils::{mean, std_dev, stft};
use crate::MirResult;

/// Features for genre classification.
#[derive(Debug, Clone)]
pub struct GenreFeatureVector {
    /// Average spectral centroid.
    pub spectral_centroid: f32,
    /// Spectral bandwidth.
    pub spectral_bandwidth: f32,
    /// Zero crossing rate.
    pub zero_crossing_rate: f32,
    /// Overall energy.
    pub energy: f32,
    /// Energy variance.
    pub energy_variance: f32,
    /// Estimated tempo.
    pub tempo: f32,
    /// Beat strength.
    pub beat_strength: f32,
    /// Harmonic complexity.
    pub harmonic_complexity: f32,
}

/// Genre feature extractor.
pub struct GenreFeatures {
    #[allow(dead_code)]
    sample_rate: f32,
}

impl GenreFeatures {
    /// Create a new genre feature extractor.
    #[must_use]
    pub fn new(sample_rate: f32) -> Self {
        Self { sample_rate }
    }

    /// Extract features from audio signal.
    ///
    /// # Errors
    ///
    /// Returns error if feature extraction fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn extract(&self, signal: &[f32]) -> MirResult<GenreFeatureVector> {
        let window_size = 2048;
        let hop_size = 512;

        let frames = stft(signal, window_size, hop_size)?;

        let mut spectral_centroids = Vec::new();
        let mut spectral_bandwidths = Vec::new();
        let mut energies = Vec::new();

        for frame in &frames {
            let mag = crate::utils::magnitude_spectrum(frame);

            let centroid = self.compute_spectral_centroid(&mag);
            let bandwidth = self.compute_spectral_bandwidth(&mag, centroid);
            let energy = mag.iter().map(|m| m * m).sum::<f32>();

            spectral_centroids.push(centroid);
            spectral_bandwidths.push(bandwidth);
            energies.push(energy);
        }

        let spectral_centroid = mean(&spectral_centroids);
        let spectral_bandwidth = mean(&spectral_bandwidths);
        let energy = mean(&energies);
        let energy_variance = std_dev(&energies);

        let zero_crossing_rate = self.compute_zero_crossing_rate(signal);

        // Compute actual tempo via autocorrelation of the onset-strength envelope
        let tempo = self.estimate_tempo(&energies);

        // Compute beat strength from onset envelope peak prominence
        let beat_strength = self.estimate_beat_strength(&energies);

        let harmonic_complexity = spectral_bandwidth / (spectral_centroid + 1.0);

        Ok(GenreFeatureVector {
            spectral_centroid,
            spectral_bandwidth,
            zero_crossing_rate,
            energy,
            energy_variance,
            tempo,
            beat_strength,
            harmonic_complexity,
        })
    }

    /// Compute spectral centroid.
    #[allow(clippy::cast_precision_loss)]
    fn compute_spectral_centroid(&self, spectrum: &[f32]) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total = 0.0;

        for (i, &mag) in spectrum.iter().enumerate() {
            weighted_sum += i as f32 * mag;
            total += mag;
        }

        if total > 0.0 {
            weighted_sum / total
        } else {
            0.0
        }
    }

    /// Compute spectral bandwidth.
    #[allow(clippy::cast_precision_loss)]
    fn compute_spectral_bandwidth(&self, spectrum: &[f32], centroid: f32) -> f32 {
        let mut weighted_sum = 0.0;
        let mut total = 0.0;

        for (i, &mag) in spectrum.iter().enumerate() {
            let diff = i as f32 - centroid;
            weighted_sum += diff * diff * mag;
            total += mag;
        }

        if total > 0.0 {
            (weighted_sum / total).sqrt()
        } else {
            0.0
        }
    }

    /// Compute zero crossing rate.
    #[allow(clippy::cast_precision_loss)]
    fn compute_zero_crossing_rate(&self, signal: &[f32]) -> f32 {
        if signal.len() < 2 {
            return 0.0;
        }
        let mut crossings = 0;

        for i in 1..signal.len() {
            if (signal[i] >= 0.0 && signal[i - 1] < 0.0)
                || (signal[i] < 0.0 && signal[i - 1] >= 0.0)
            {
                crossings += 1;
            }
        }

        crossings as f32 / signal.len() as f32
    }

    /// Estimate tempo from energy envelope using autocorrelation.
    ///
    /// Computes the autocorrelation of the frame energy envelope, then searches
    /// for the strongest peak in the lag range corresponding to 60-200 BPM.
    /// The hop rate is `sample_rate / hop_size` frames per second.
    #[allow(clippy::cast_precision_loss)]
    fn estimate_tempo(&self, energies: &[f32]) -> f32 {
        if energies.len() < 4 {
            return 120.0; // Fallback when too few frames
        }

        let hop_size: f32 = 512.0;
        let frames_per_second = self.sample_rate / hop_size;

        // Lag range: 60 BPM -> 1 beat/sec -> frames_per_second frames/beat
        //            200 BPM -> 3.33 beats/sec -> frames_per_second/3.33 frames/beat
        let min_lag = (frames_per_second * 60.0 / 200.0).ceil() as usize;
        let max_lag = (frames_per_second * 60.0 / 60.0).floor() as usize;
        let max_lag = max_lag.min(energies.len() - 1);

        if min_lag >= max_lag {
            return 120.0;
        }

        // Compute autocorrelation for the lag range
        let n = energies.len();
        let e_mean = mean(energies);

        let mut best_lag = min_lag;
        let mut best_corr = f32::NEG_INFINITY;

        for lag in min_lag..=max_lag {
            let mut num = 0.0_f32;
            let mut denom_a = 0.0_f32;
            let mut denom_b = 0.0_f32;
            for i in 0..(n - lag) {
                let a = energies[i] - e_mean;
                let b = energies[i + lag] - e_mean;
                num += a * b;
                denom_a += a * a;
                denom_b += b * b;
            }
            let denom = (denom_a * denom_b).sqrt();
            let corr = if denom > 1e-12 { num / denom } else { 0.0 };

            if corr > best_corr {
                best_corr = corr;
                best_lag = lag;
            }
        }

        // Convert lag back to BPM
        let bpm = frames_per_second * 60.0 / best_lag as f32;
        bpm.clamp(60.0, 200.0)
    }

    /// Estimate beat strength from energy envelope peak prominence.
    ///
    /// Computes the ratio of energy peaks above the local mean to the overall
    /// energy mean, giving a measure of how "punchy" the beats are.
    fn estimate_beat_strength(&self, energies: &[f32]) -> f32 {
        if energies.len() < 3 {
            return 0.5;
        }

        let overall_mean = mean(energies);
        if overall_mean < 1e-12 {
            return 0.0;
        }

        // Count frames that are local peaks and significantly above mean
        let mut peak_energy_sum = 0.0_f32;
        let mut peak_count = 0_u32;

        for i in 1..(energies.len() - 1) {
            if energies[i] > energies[i - 1] && energies[i] > energies[i + 1] {
                // It is a local peak
                if energies[i] > overall_mean * 1.2 {
                    peak_energy_sum += energies[i];
                    peak_count += 1;
                }
            }
        }

        if peak_count == 0 {
            return 0.1;
        }

        let avg_peak = peak_energy_sum / peak_count as f32;
        // Normalize: ratio of average peak energy to overall mean, clamped 0-1
        let ratio = (avg_peak / overall_mean - 1.0) / 3.0; // Scale so ratio of 4x -> 1.0
        ratio.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genre_features_creation() {
        let features = GenreFeatures::new(44100.0);
        assert_eq!(features.sample_rate, 44100.0);
    }

    #[test]
    fn test_zero_crossing_rate() {
        let features = GenreFeatures::new(44100.0);
        let signal = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let zcr = features.compute_zero_crossing_rate(&signal);
        assert!(zcr > 0.5);
    }
}
