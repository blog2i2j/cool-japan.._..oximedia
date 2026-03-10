//! Total harmonic distortion (THD) measurement.

use rustfft::{num_complex::Complex, FftPlanner};

/// Compute total harmonic distortion (THD).
///
/// THD is the ratio of the sum of harmonic powers to the fundamental power.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `sample_rate` - Sample rate in Hz
///
/// # Returns
/// THD value (0-1, lower is better)
#[must_use]
pub fn total_harmonic_distortion(samples: &[f32], _sample_rate: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let fft_size = samples.len().next_power_of_two();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    // Prepare FFT input
    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    fft.process(&mut buffer);

    // Compute magnitude spectrum
    let magnitude: Vec<f32> = buffer[..fft_size / 2].iter().map(|c| c.norm()).collect();

    // Find fundamental frequency peak
    let fundamental_bin = magnitude
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    if fundamental_bin == 0 {
        return 0.0;
    }

    let fundamental_power = magnitude[fundamental_bin].powi(2);

    // Sum harmonic powers (2f, 3f, 4f, 5f)
    let mut harmonic_power = 0.0;
    for harmonic in 2..=5 {
        let harmonic_bin = fundamental_bin * harmonic;
        if harmonic_bin < magnitude.len() {
            harmonic_power += magnitude[harmonic_bin].powi(2);
        }
    }

    if fundamental_power > 0.0 {
        (harmonic_power / fundamental_power).sqrt()
    } else {
        0.0
    }
}

/// Compute THD+N (THD plus noise).
#[must_use]
pub fn thd_plus_noise(samples: &[f32], _sample_rate: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let fft_size = samples.len().next_power_of_two();
    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(fft_size);

    let mut buffer: Vec<Complex<f32>> = samples
        .iter()
        .map(|&s| Complex::new(s, 0.0))
        .chain(std::iter::repeat(Complex::new(0.0, 0.0)))
        .take(fft_size)
        .collect();

    fft.process(&mut buffer);

    let magnitude: Vec<f32> = buffer[..fft_size / 2].iter().map(|c| c.norm()).collect();

    let fundamental_bin = magnitude
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    if fundamental_bin == 0 {
        return 0.0;
    }

    let fundamental_power = magnitude[fundamental_bin].powi(2);
    let total_power: f32 = magnitude.iter().map(|&m| m.powi(2)).sum();

    let noise_plus_harmonics = total_power - fundamental_power;

    if fundamental_power > 0.0 {
        (noise_plus_harmonics / fundamental_power).sqrt()
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thd_clean_signal() {
        // Pure sine wave should have very low THD
        let sample_rate = 44100.0;
        let samples: Vec<f32> = (0..4096)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin())
            .collect();

        let thd = total_harmonic_distortion(&samples, sample_rate);
        assert!(thd < 0.1);
    }

    #[test]
    fn test_thd_distorted_signal() {
        // Clipped sine wave should have higher THD
        let sample_rate = 44100.0;
        let samples: Vec<f32> = (0..4096)
            .map(|i| {
                let x = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate).sin();
                x.clamp(-0.5, 0.5) // Clipping
            })
            .collect();

        let thd = total_harmonic_distortion(&samples, sample_rate);
        assert!(thd > 0.05);
    }
}
