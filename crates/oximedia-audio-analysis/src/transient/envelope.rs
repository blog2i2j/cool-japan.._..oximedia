//! Envelope analysis and ADSR detection.

/// Envelope characteristics (ADSR parameters).
#[derive(Debug, Clone)]
pub struct EnvelopeCharacteristics {
    /// Attack time in seconds
    pub attack_time: f32,
    /// Decay time in seconds
    pub decay_time: f32,
    /// Sustain level (0-1)
    pub sustain_level: f32,
    /// Release time in seconds
    pub release_time: f32,
    /// Peak amplitude
    pub peak_amplitude: f32,
}

/// Compute amplitude envelope of audio signal.
///
/// # Arguments
/// * `samples` - Audio samples
/// * `window_size` - Window size for envelope detection
///
/// # Returns
/// Amplitude envelope
pub fn compute_envelope(samples: &[f32], window_size: usize) -> Vec<f32> {
    if samples.is_empty() || window_size == 0 {
        return vec![];
    }

    let mut envelope = Vec::with_capacity(samples.len() / window_size + 1);

    for chunk in samples.chunks(window_size) {
        let peak = chunk.iter().map(|&x| x.abs()).fold(0.0_f32, f32::max);
        envelope.push(peak);
    }

    // Smooth envelope
    smooth_envelope(&mut envelope, 3);

    envelope
}

/// Smooth envelope using moving average.
fn smooth_envelope(envelope: &mut [f32], window: usize) {
    if envelope.len() < window || window == 0 {
        return;
    }

    let mut smoothed = Vec::with_capacity(envelope.len());

    for i in 0..envelope.len() {
        let start = i.saturating_sub(window / 2);
        let end = (i + window / 2 + 1).min(envelope.len());

        let sum: f32 = envelope[start..end].iter().sum();
        let avg = sum / (end - start) as f32;
        smoothed.push(avg);
    }

    envelope.copy_from_slice(&smoothed);
}

/// Extract ADSR characteristics from envelope.
#[must_use]
pub fn extract_adsr(
    envelope: &[f32],
    sample_rate: f32,
    window_size: usize,
) -> EnvelopeCharacteristics {
    if envelope.is_empty() {
        return EnvelopeCharacteristics {
            attack_time: 0.0,
            decay_time: 0.0,
            sustain_level: 0.0,
            release_time: 0.0,
            peak_amplitude: 0.0,
        };
    }

    let time_per_sample = window_size as f32 / sample_rate;

    // Find peak
    let peak_idx = envelope
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map_or(0, |(i, _)| i);

    let peak_amplitude = envelope[peak_idx];

    // Attack time: from start to peak
    let attack_time = peak_idx as f32 * time_per_sample;

    // Find sustain level (average of middle 50% after peak)
    let sustain_start = peak_idx + (envelope.len() - peak_idx) / 4;
    let sustain_end = peak_idx + 3 * (envelope.len() - peak_idx) / 4;

    let sustain_level = if sustain_end > sustain_start {
        let sum: f32 = envelope[sustain_start..sustain_end].iter().sum();
        sum / (sustain_end - sustain_start) as f32
    } else {
        envelope.last().copied().unwrap_or(0.0)
    };

    // Decay time: from peak to sustain level
    let sustain_threshold = (peak_amplitude + sustain_level) / 2.0;
    let mut decay_idx = peak_idx;

    #[allow(clippy::needless_range_loop)]
    for i in peak_idx..envelope.len() {
        if envelope[i] <= sustain_threshold {
            decay_idx = i;
            break;
        }
    }

    let decay_time = (decay_idx - peak_idx) as f32 * time_per_sample;

    // Release time: from end of sustain to end
    let release_time = (envelope.len() - sustain_end) as f32 * time_per_sample;

    EnvelopeCharacteristics {
        attack_time,
        decay_time,
        sustain_level,
        release_time,
        peak_amplitude,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_envelope() {
        // Generate signal with clear envelope
        let mut samples = Vec::new();

        // Attack
        for i in 0..1000 {
            samples.push(i as f32 / 1000.0);
        }

        // Sustain
        for _ in 0..2000 {
            samples.push(0.8);
        }

        // Release
        for i in 0..1000 {
            samples.push(0.8 * (1.0 - i as f32 / 1000.0));
        }

        let envelope = compute_envelope(&samples, 100);
        assert!(!envelope.is_empty());

        // Envelope should have peak
        let max = envelope.iter().copied().fold(0.0_f32, f32::max);
        assert!(max > 0.5);
    }

    #[test]
    fn test_extract_adsr() {
        let envelope = vec![
            0.0, 0.2, 0.4, 0.6, 0.8, 1.0, // Attack
            0.9, 0.8, 0.7, 0.7, 0.7, 0.7, // Decay and sustain
            0.6, 0.4, 0.2, 0.0, // Release
        ];

        let adsr = extract_adsr(&envelope, 44100.0, 512);

        assert!(adsr.peak_amplitude > 0.9);
        assert!(adsr.attack_time >= 0.0);
        assert!(adsr.sustain_level > 0.0);
    }
}
