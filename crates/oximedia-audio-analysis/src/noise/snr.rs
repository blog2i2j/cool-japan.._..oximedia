//! Signal-to-noise ratio (SNR) computation.

use crate::compute_rms;

/// Compute signal-to-noise ratio.
///
/// # Arguments
/// * `signal` - Clean signal samples
/// * `noise` - Noise samples
///
/// # Returns
/// SNR in linear scale (signal power / noise power)
#[must_use]
pub fn signal_to_noise_ratio(signal: &[f32], noise: &[f32]) -> f32 {
    if signal.is_empty() || noise.is_empty() {
        return 0.0;
    }

    let signal_power = compute_rms(signal).powi(2);
    let noise_power = compute_rms(noise).powi(2);

    if noise_power > 0.0 {
        signal_power / noise_power
    } else {
        f32::INFINITY
    }
}

/// Compute SNR in decibels.
#[must_use]
pub fn compute_snr_db(signal: &[f32], noise: &[f32]) -> f32 {
    let snr = signal_to_noise_ratio(signal, noise);

    if snr > 0.0 && snr.is_finite() {
        10.0 * snr.log10()
    } else if snr.is_infinite() {
        100.0 // Cap at 100 dB
    } else {
        -100.0 // Floor at -100 dB
    }
}

/// Estimate SNR from a single signal by separating signal and noise.
///
/// Uses a simple approach: assumes noise is the low-amplitude portions.
#[must_use]
pub fn estimate_snr(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    // Sort by amplitude to separate signal from noise
    let mut sorted: Vec<f32> = samples.iter().map(|&x| x.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Bottom 25% is noise
    let noise_cutoff = sorted.len() / 4;
    let noise_samples = &sorted[..noise_cutoff];

    // Top 75% is signal + noise
    let signal_samples = &sorted[noise_cutoff..];

    signal_to_noise_ratio(signal_samples, noise_samples)
}

/// Estimate SNR in decibels.
#[must_use]
pub fn estimate_snr_db(samples: &[f32]) -> f32 {
    let snr = estimate_snr(samples);

    if snr > 0.0 && snr.is_finite() {
        10.0 * snr.log10()
    } else {
        -100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_snr_computation() {
        // Clean signal
        let signal = vec![1.0; 100];
        // Low noise
        let noise = vec![0.1; 100];

        let snr = signal_to_noise_ratio(&signal, &noise);
        assert!(snr > 10.0); // Should have high SNR

        let snr_db = compute_snr_db(&signal, &noise);
        assert!(snr_db > 10.0);
    }

    #[test]
    fn test_snr_estimation() {
        // Signal with some noise
        let mut samples = vec![0.01; 100]; // Noise floor
        samples.extend(vec![1.0; 100]); // Signal

        let snr_db = estimate_snr_db(&samples);
        assert!(snr_db > 0.0);
    }
}
