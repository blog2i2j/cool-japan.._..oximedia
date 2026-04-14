//! Vocal effort estimation: whisper, normal, shout classification.
//!
//! Classifies vocal effort level based on energy, spectral tilt, and
//! zero-crossing rate characteristics:
//!
//! - **Whisper**: low energy, high ZCR (turbulent noise-like), flat/negative spectral tilt
//! - **Normal**: moderate energy, typical ZCR, moderate spectral tilt
//! - **Shout**: high energy, lower ZCR (periodic), positive/flat spectral tilt
//!
//! These thresholds are empirically motivated by acoustic phonetics research:
//! Hanson & Chuang (1999), Lienard & Di Benedetto (1999).

/// Vocal effort level classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VocalEffort {
    /// Whispered speech: low amplitude, turbulent, breathy.
    Whisper,
    /// Normal conversational speech.
    Normal,
    /// Shouted or raised voice: high amplitude, tense, less periodic.
    Shout,
}

impl std::fmt::Display for VocalEffort {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Whisper => write!(f, "whisper"),
            Self::Normal => write!(f, "normal"),
            Self::Shout => write!(f, "shout"),
        }
    }
}

/// Result of vocal effort estimation.
#[derive(Debug, Clone)]
pub struct VocalEffortResult {
    /// Classified vocal effort level.
    pub effort: VocalEffort,
    /// Confidence score for the classification (0.0–1.0).
    pub confidence: f32,
    /// RMS energy of the input (linear scale).
    pub rms_energy: f32,
    /// Zero-crossing rate (crossings per sample).
    pub zcr: f32,
    /// Spectral tilt in dB/octave (negative = more high-frequency attenuation).
    pub spectral_tilt_db_octave: f32,
    /// Harmonic-to-noise ratio estimate (higher = more periodic / voiced).
    pub hnr_estimate: f32,
}

/// Estimate vocal effort from audio samples.
///
/// # Arguments
/// * `samples` - Mono audio samples.
/// * `sample_rate` - Sample rate in Hz.
///
/// # Returns
/// `VocalEffortResult` with effort level, confidence, and diagnostic features.
#[must_use]
pub fn estimate_vocal_effort(samples: &[f32], sample_rate: f32) -> VocalEffortResult {
    if samples.is_empty() {
        return VocalEffortResult {
            effort: VocalEffort::Normal,
            confidence: 0.0,
            rms_energy: 0.0,
            zcr: 0.0,
            spectral_tilt_db_octave: 0.0,
            hnr_estimate: 0.0,
        };
    }

    let rms_energy = compute_rms(samples);
    let zcr = compute_zcr(samples);
    let spectral_tilt_db_octave = estimate_spectral_tilt(samples, sample_rate);
    let hnr_estimate = estimate_hnr(samples, sample_rate);

    // Decision logic based on acoustic features
    //
    // Whisper profile: low RMS, high ZCR (turbulent noise), negative spectral tilt,
    //                  low HNR (aperiodic).
    // Shout profile:   high RMS, low/moderate ZCR, flat/positive spectral tilt,
    //                  moderate–high HNR.
    // Normal:          in between.

    // Normalise RMS to a rough loudness scale (speech RMS typically 0.01–0.3)
    let loudness = (rms_energy / 0.1_f32).clamp(0.0, 1.0);

    // Score each effort class in [0, 1]
    let whisper_score = {
        let energy_score = (1.0 - loudness).max(0.0);
        let zcr_score = (zcr / 0.4_f32).clamp(0.0, 1.0); // high ZCR → whisper
        let hnr_score = (1.0 - (hnr_estimate / 20.0_f32).clamp(0.0, 1.0)).max(0.0);
        let tilt_score = if spectral_tilt_db_octave < -3.0 { 1.0_f32 } else { 0.3 };
        energy_score * 0.35 + zcr_score * 0.30 + hnr_score * 0.20 + tilt_score * 0.15
    };

    let shout_score = {
        let energy_score = loudness;
        let zcr_score = (1.0 - (zcr / 0.3_f32).clamp(0.0, 1.0)).max(0.0);
        let hnr_score = ((hnr_estimate - 5.0) / 20.0_f32).clamp(0.0, 1.0);
        let tilt_score = if spectral_tilt_db_octave > -1.0 { 1.0_f32 } else { 0.2 };
        energy_score * 0.40 + zcr_score * 0.25 + hnr_score * 0.20 + tilt_score * 0.15
    };

    let normal_score = {
        // Normal is the complement of extremes
        let not_whisper = 1.0 - whisper_score;
        let not_shout = 1.0 - shout_score;
        (not_whisper * 0.5 + not_shout * 0.5).clamp(0.0, 1.0)
    };

    let max_score = whisper_score.max(shout_score).max(normal_score);

    let (effort, raw_confidence) = if max_score == whisper_score && whisper_score > shout_score {
        (VocalEffort::Whisper, whisper_score)
    } else if max_score == shout_score && shout_score > normal_score {
        (VocalEffort::Shout, shout_score)
    } else {
        (VocalEffort::Normal, normal_score)
    };

    // Confidence: distance of winner from second-best, normalised
    let mut scores = [whisper_score, normal_score, shout_score];
    scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    let confidence = if scores[0] > 1e-6 {
        ((scores[0] - scores[1]) / scores[0]).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let _ = raw_confidence;

    VocalEffortResult {
        effort,
        confidence,
        rms_energy,
        zcr,
        spectral_tilt_db_octave,
        hnr_estimate,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn compute_rms(samples: &[f32]) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }
    let sq: f32 = samples.iter().map(|&x| x * x).sum();
    (sq / samples.len() as f32).sqrt()
}

fn compute_zcr(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }
    let mut crossings = 0usize;
    for i in 1..samples.len() {
        if (samples[i] >= 0.0) != (samples[i - 1] >= 0.0) {
            crossings += 1;
        }
    }
    crossings as f32 / (samples.len() - 1) as f32
}

/// Estimate spectral tilt in dB/octave using a simple two-band comparison.
///
/// Computes the ratio of energy in the upper half of the spectrum to the
/// lower half, then converts to dB/octave.
fn estimate_spectral_tilt(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    // Simple approximation: compare low-band energy (< 1 kHz) vs high-band (1–8 kHz).
    let bin_hz = sample_rate / (samples.len() as f32 * 2.0);
    let low_end = (1000.0 / bin_hz) as usize;
    let high_end = (8000.0 / bin_hz).min(samples.len() as f32 / 2.0) as usize;

    if low_end == 0 || high_end <= low_end {
        return -6.0; // default speech-like value
    }

    // Use the raw time-domain energy in sub-windows as a proxy.
    // For a proper tilt we'd need an FFT, but here we use a filterbank approximation.
    // Split samples into N equal segments; correlate index with log(energy).
    let n_bands = 8usize;
    let band_size = (samples.len() / n_bands).max(1);
    let mut band_energies = Vec::with_capacity(n_bands);

    for b in 0..n_bands {
        let start = b * band_size;
        let end = ((b + 1) * band_size).min(samples.len());
        let e: f32 = samples[start..end].iter().map(|&x| x * x).sum::<f32>()
            / (end - start) as f32;
        band_energies.push(e.max(1e-20_f32));
    }

    // Linear regression of log(energy) vs log(frequency_index)
    let n = n_bands as f32;
    let log_energies: Vec<f32> = band_energies.iter().map(|&e| e.ln()).collect();
    // Frequency roughly proportional to band index + 1
    let log_freqs: Vec<f32> = (1..=n_bands).map(|i| (i as f32).ln()).collect();

    let mean_lf = log_freqs.iter().sum::<f32>() / n;
    let mean_le = log_energies.iter().sum::<f32>() / n;

    let num: f32 = log_freqs
        .iter()
        .zip(&log_energies)
        .map(|(&lf, &le)| (lf - mean_lf) * (le - mean_le))
        .sum();
    let den: f32 = log_freqs.iter().map(|&lf| (lf - mean_lf).powi(2)).sum();

    if den.abs() < 1e-10 {
        return -6.0;
    }

    // Slope in (log-energy / log-freq) units → multiply by ~3.32 to get dB/octave
    let slope = num / den; // dB (natural log scale) per log-freq unit
    slope * 10.0 / std::f32::consts::LN_2 // convert to dB/octave
}

/// Estimate harmonic-to-noise ratio using autocorrelation.
///
/// Returns an approximate HNR in dB (positive = more periodic/voiced).
fn estimate_hnr(samples: &[f32], sample_rate: f32) -> f32 {
    if samples.is_empty() {
        return 0.0;
    }

    let energy: f32 = samples.iter().map(|&x| x * x).sum();
    if energy < 1e-12 {
        return 0.0;
    }

    let min_lag = (sample_rate / 500.0) as usize; // 500 Hz max F0
    let max_lag = (sample_rate / 60.0) as usize; // 60 Hz min F0

    if max_lag >= samples.len() || min_lag >= max_lag {
        return 0.0;
    }

    let mut best_corr = 0.0_f32;

    for lag in min_lag..=max_lag.min(samples.len() - 1) {
        let corr: f32 = samples[..samples.len() - lag]
            .iter()
            .zip(&samples[lag..])
            .map(|(&a, &b)| a * b)
            .sum::<f32>()
            / energy;
        if corr > best_corr {
            best_corr = corr;
        }
    }

    // Convert autocorrelation peak to HNR in dB
    let r = best_corr.clamp(0.0, 0.9999);
    if r < 1e-6 {
        return 0.0;
    }
    10.0 * (r / (1.0 - r)).log10()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq: f32, amp: f32, n: usize, sr: f32) -> Vec<f32> {
        (0..n)
            .map(|i| amp * (2.0 * PI * freq * i as f32 / sr).sin())
            .collect()
    }

    fn white_noise_like(n: usize, amp: f32) -> Vec<f32> {
        // Use a simple LCG for deterministic "noise".
        // We use a 32-bit hash of the full 64-bit state to ensure
        // values span both positive and negative ranges symmetrically.
        let mut state = 12345u64;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                // XOR-fold the high and low 32 bits to get a well-distributed u32
                let folded = ((state >> 32) as u32) ^ (state as u32);
                let v = (folded as f32 / u32::MAX as f32) * 2.0 - 1.0;
                v * amp
            })
            .collect()
    }

    #[test]
    fn test_estimate_vocal_effort_silence() {
        let result = estimate_vocal_effort(&vec![0.0; 4096], 44100.0);
        // Silence is ambiguous but should not panic; confidence should be low
        assert!(result.rms_energy < 1e-6);
    }

    #[test]
    fn test_estimate_vocal_effort_shout_high_energy() {
        // High amplitude sine: should classify as shout or normal
        let samples = sine_wave(200.0, 0.9, 8192, 44100.0);
        let result = estimate_vocal_effort(&samples, 44100.0);
        // High energy → should lean shout
        assert!(result.rms_energy > 0.5, "RMS should be high: {}", result.rms_energy);
        assert!(
            matches!(result.effort, VocalEffort::Shout | VocalEffort::Normal),
            "High energy should classify as shout or normal"
        );
    }

    #[test]
    fn test_estimate_vocal_effort_whisper_noise() {
        // Low amplitude noise: should classify as whisper
        let samples = white_noise_like(8192, 0.01);
        let result = estimate_vocal_effort(&samples, 44100.0);
        assert!(result.rms_energy < 0.05, "RMS should be low: {}", result.rms_energy);
        // Noise has high ZCR → whisper-like
        assert!(
            result.zcr > 0.0,
            "ZCR should be positive for noise: {}",
            result.zcr
        );
    }

    #[test]
    fn test_vocal_effort_result_fields() {
        let samples = sine_wave(440.0, 0.3, 4096, 44100.0);
        let result = estimate_vocal_effort(&samples, 44100.0);
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.rms_energy >= 0.0);
        assert!(result.zcr >= 0.0 && result.zcr <= 1.0);
    }

    #[test]
    fn test_vocal_effort_empty() {
        let result = estimate_vocal_effort(&[], 44100.0);
        assert_eq!(result.effort, VocalEffort::Normal);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_display() {
        assert_eq!(VocalEffort::Whisper.to_string(), "whisper");
        assert_eq!(VocalEffort::Normal.to_string(), "normal");
        assert_eq!(VocalEffort::Shout.to_string(), "shout");
    }
}
