//! RMS level computation.

use crate::compute_rms;

/// Compute RMS level (same as in lib.rs, re-exported for convenience).
#[must_use]
pub fn rms_level(samples: &[f32]) -> f32 {
    compute_rms(samples)
}

/// Compute RMS level over time with specified hop size.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `hop_size` - Number of samples between RMS computations
///
/// # Returns
/// Vector of RMS values over time
#[must_use]
pub fn rms_over_time(samples: &[f32], hop_size: usize) -> Vec<f32> {
    if hop_size == 0 || samples.is_empty() {
        return vec![];
    }

    let window_size = hop_size * 2; // Use 2x hop size for window

    if samples.len() < window_size {
        return vec![rms_level(samples)];
    }

    let num_frames = (samples.len() - window_size) / hop_size + 1;
    let mut rms_values = Vec::with_capacity(num_frames);

    for i in 0..num_frames {
        let start = i * hop_size;
        let end = (start + window_size).min(samples.len());
        let rms = rms_level(&samples[start..end]);
        rms_values.push(rms);
    }

    rms_values
}

/// Compute RMS level in decibels.
#[must_use]
pub fn rms_db(samples: &[f32]) -> f32 {
    let rms = rms_level(samples);
    if rms > 0.0 {
        20.0 * rms.log10()
    } else {
        -100.0
    }
}

/// Compute short-term and long-term RMS levels.
#[must_use]
pub fn short_long_term_rms(samples: &[f32], short_window: usize, long_window: usize) -> (f32, f32) {
    let short_rms = if samples.len() >= short_window {
        rms_level(&samples[samples.len() - short_window..])
    } else {
        rms_level(samples)
    };

    let long_rms = if samples.len() >= long_window {
        rms_level(&samples[samples.len() - long_window..])
    } else {
        rms_level(samples)
    };

    (short_rms, long_rms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rms_over_time() {
        // Generate signal with increasing amplitude
        let mut samples = Vec::new();
        for i in 0..10000 {
            let amp = i as f32 / 10000.0;
            samples.push(amp);
        }

        let rms_track = rms_over_time(&samples, 1000);
        assert!(!rms_track.is_empty());

        // RMS should generally increase
        assert!(
            rms_track.last().expect("expected last to be available")
                > rms_track.first().expect("expected last to be available")
        );
    }

    #[test]
    fn test_rms_db() {
        let samples = vec![1.0; 100];
        let db = rms_db(&samples);
        assert!((db - 0.0).abs() < 0.1); // RMS of 1.0 = 0 dB
    }

    #[test]
    fn test_short_long_term() {
        let samples = vec![0.5; 10000];
        let (short, long) = short_long_term_rms(&samples, 1000, 5000);

        // Should be similar for constant signal
        assert!((short - long).abs() < 0.01);
    }
}
