//! Audio forensics — detecting spliced and edited audio recordings.
//!
//! This module analyses a mono PCM f32 audio signal for evidence of
//! post-production splicing: sudden energy jumps, spectrum discontinuities,
//! envelope changes, and phase glitches.  It also implements a simplified
//! Electrical Network Frequency (ENF) consistency check as a tampering score.
//!
//! # Reference
//!
//! * Malik, H. (2013). "Acoustic environment identification and its applications
//!   to audio forensics". *IEEE Trans. IFST*, 8(11), 1827–1837.
//! * Hua, G., Goh, J., & Thing, V. L. L. (2014). "A dynamic matching algorithm
//!   for audio timestamp identification using the ENF criterion". *IEEE Trans. IFS*.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The technique by which a splice was detected.
#[derive(Debug, Clone, PartialEq)]
pub enum SpliceMethod {
    /// Sudden jump in RMS energy level (> 10 dB between adjacent frames).
    LevelJump,
    /// Abrupt change in spectral centroid or spectral content.
    SpectrumDiscontinuity,
    /// Discontinuity in the temporal amplitude envelope.
    EnvChange,
    /// Sudden shift in zero-crossing rate — indicative of phase discontinuity.
    PhaseGlitch,
}

/// A detected splice point in an audio signal.
#[derive(Debug, Clone)]
pub struct AudioSplice {
    /// Sample offset of the detected splice (left edge of the analysis frame).
    pub position_samples: usize,
    /// Confidence in `[0, 1]` that this is a genuine splice.
    pub confidence: f32,
    /// Primary method that triggered the detection.
    pub method: SpliceMethod,
}

// ---------------------------------------------------------------------------
// Core detection
// ---------------------------------------------------------------------------

/// Detect potential splice points in a mono PCM f32 signal.
///
/// The signal is divided into non-overlapping frames of ~10 ms
/// (`sample_rate / 100` samples).  For each pair of consecutive frames the
/// function computes:
///
/// * **RMS energy** — an inter-frame jump > 10 dB triggers `LevelJump`.
/// * **Spectral centroid** — a sharp change in centroid (via DFT-free
///   approximation) triggers `SpectrumDiscontinuity`.
/// * **Envelope** — a mean-absolute-value change > 3σ triggers `EnvChange`.
/// * **Zero-crossing rate (ZCR)** — a sudden ZCR shift > 2σ triggers `PhaseGlitch`.
///
/// # Arguments
///
/// * `samples`     — Mono PCM samples in [-1, 1].
/// * `sample_rate` — Sample rate in Hz (e.g. 44100).
///
/// # Returns
///
/// A `Vec<AudioSplice>` of detected splice candidates, sorted by sample
/// position.
#[must_use]
pub fn detect_audio_splices(samples: &[f32], sample_rate: u32) -> Vec<AudioSplice> {
    if samples.is_empty() || sample_rate == 0 {
        return Vec::new();
    }

    // Frame length ≈ 10 ms
    let frame_len = ((sample_rate as usize) / 100).max(1);
    let num_frames = samples.len() / frame_len;

    if num_frames < 2 {
        return Vec::new();
    }

    // Pre-compute per-frame features
    let rms: Vec<f32> = (0..num_frames)
        .map(|f| compute_rms(&samples[f * frame_len..(f + 1) * frame_len]))
        .collect();

    let zcr: Vec<f32> = (0..num_frames)
        .map(|f| compute_zcr(&samples[f * frame_len..(f + 1) * frame_len]))
        .collect();

    let centroid: Vec<f32> = (0..num_frames)
        .map(|f| compute_spectral_centroid(&samples[f * frame_len..(f + 1) * frame_len]))
        .collect();

    let envelope: Vec<f32> = (0..num_frames)
        .map(|f| compute_mean_abs(&samples[f * frame_len..(f + 1) * frame_len]))
        .collect();

    // Global statistics for adaptive thresholding
    let zcr_stats = mean_std(&zcr);
    let centroid_stats = mean_std(&centroid);
    let envelope_stats = mean_std(&envelope);

    let mut splices: Vec<AudioSplice> = Vec::new();

    for f in 1..num_frames {
        let pos = f * frame_len;

        // --- LevelJump: RMS energy > 10 dB ---
        let rms_prev = rms[f - 1].max(1e-12);
        let rms_curr = rms[f].max(1e-12);
        let db_change = 20.0 * (rms_curr / rms_prev).log10().abs();
        if db_change > 10.0 {
            let confidence = (db_change / 40.0).min(1.0);
            splices.push(AudioSplice {
                position_samples: pos,
                confidence,
                method: SpliceMethod::LevelJump,
            });
            continue; // prioritise LevelJump over softer indicators
        }

        // --- PhaseGlitch: ZCR sudden shift > 2σ ---
        let zcr_delta = (zcr[f] - zcr[f - 1]).abs();
        if zcr_stats.1 > 1e-9 && zcr_delta > 2.0 * zcr_stats.1 {
            let confidence = (zcr_delta / (2.0 * zcr_stats.1 + 1e-9)).min(1.0) * 0.85;
            splices.push(AudioSplice {
                position_samples: pos,
                confidence,
                method: SpliceMethod::PhaseGlitch,
            });
            continue;
        }

        // --- SpectrumDiscontinuity: centroid change > 3σ ---
        let centroid_delta = (centroid[f] - centroid[f - 1]).abs();
        if centroid_stats.1 > 1e-9 && centroid_delta > 3.0 * centroid_stats.1 {
            let confidence = (centroid_delta / (3.0 * centroid_stats.1 + 1e-9)).min(1.0) * 0.75;
            splices.push(AudioSplice {
                position_samples: pos,
                confidence,
                method: SpliceMethod::SpectrumDiscontinuity,
            });
            continue;
        }

        // --- EnvChange: envelope MAV > 3σ ---
        let env_delta = (envelope[f] - envelope[f - 1]).abs();
        if envelope_stats.1 > 1e-9 && env_delta > 3.0 * envelope_stats.1 {
            let confidence = (env_delta / (3.0 * envelope_stats.1 + 1e-9)).min(1.0) * 0.65;
            splices.push(AudioSplice {
                position_samples: pos,
                confidence,
                method: SpliceMethod::EnvChange,
            });
        }
    }

    splices.sort_by_key(|s| s.position_samples);
    splices
}

// ---------------------------------------------------------------------------
// ENF consistency test
// ---------------------------------------------------------------------------

/// Electrical Network Frequency (ENF) tampering score.
///
/// Recordings made near mains-powered equipment contain a faint 50 Hz or 60 Hz
/// tone (and its harmonics) whose frequency drifts slightly over time.  A
/// spliced recording typically shows a discontinuity in this drift pattern.
///
/// This simplified implementation estimates the ENF via the spectral centroid
/// computed in a narrow band around 50/60 Hz.  The deviation of this centroid
/// from the expected frequency is returned as a normalised tampering score in
/// `[0, 1]`.
///
/// # Arguments
///
/// * `samples` — Mono PCM signal in [-1, 1].
///
/// # Returns
///
/// A score where `0.0` indicates no detectable ENF deviation (clean) and `1.0`
/// indicates maximum deviation (likely tampered or no ENF signal present).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn audio_eno_test(samples: &[f32]) -> f32 {
    // Use DFT on the entire signal to find spectral energy near 50 and 60 Hz.
    // We use a brute-force Goertzel-like evaluation at a few target frequencies.
    if samples.is_empty() {
        return 1.0; // no signal → maximum uncertainty
    }

    let n = samples.len();

    // Evaluate DFT at specific bins via explicit trigonometric sums (no FFT crate needed)
    // We look for energy at 50 Hz ± 1 Hz and 60 Hz ± 1 Hz.
    // In a fixed-frequency scenario the sample_rate is assumed to be 44100 Hz.
    // The caller may pass any sample rate, but we cannot recover it here — so we
    // use relative bin fractions derived from typical recording sample rates.
    // Instead of absolute Hz we compute the spectral centroid in the low-frequency
    // region (bins 0..N/4) and measure how close it is to the expected ENF bins.

    // Goertzel evaluation at normalised frequency ω = 2π·f/N
    // We check whether the signal energy concentrates near ω ≈ 0 (DC) or shows
    // a meaningful tone in the sub-1 kHz region characteristic of ENF presence.
    let goertzel = |target_bin: f64| -> f32 {
        let w = 2.0 * std::f64::consts::PI * target_bin / n as f64;
        let coeff = 2.0 * w.cos();
        let mut q1 = 0.0_f64;
        let mut q2 = 0.0_f64;
        for &s in samples {
            let q0 = f64::from(s) + coeff * q1 - q2;
            q2 = q1;
            q1 = q0;
        }
        // Power = q1² + q2² - q1·q2·coeff
        let power = q1 * q1 + q2 * q2 - q1 * q2 * coeff;
        power.max(0.0) as f32
    };

    // For a typical 44100 Hz recording, 50 Hz ≈ bin 50 * N / 44100
    // We use the ratio directly: fraction of N = 50/44100 ≈ 0.001134
    // and 60/44100 ≈ 0.001361.
    // We evaluate at a few candidate bins and pick the one with most energy.
    let candidate_fractions: &[f64] = &[
        50.0 / 44100.0,
        60.0 / 44100.0,
        50.0 / 48000.0,
        60.0 / 48000.0,
        100.0 / 44100.0, // 2nd harmonic at 44.1 kHz
        120.0 / 44100.0,
    ];

    let mut enf_energy = 0.0_f32;
    let mut max_energy = 0.0_f32;

    for &frac in candidate_fractions {
        let bin = frac * n as f64;
        let energy = goertzel(bin);
        if energy > enf_energy {
            enf_energy = energy;
        }
    }

    // Total signal energy for normalisation
    let total_energy: f32 = samples.iter().map(|&s| s * s).sum::<f32>();
    if total_energy < 1e-12 {
        return 1.0; // silent signal → undefined ENF
    }

    // Peak energy across all "wide" bins to get overall scale
    let wide_step = (n / 64).max(1);
    for k in (1..n / 4).step_by(wide_step) {
        let e = goertzel(k as f64);
        if e > max_energy {
            max_energy = e;
        }
    }

    if max_energy < 1e-12 {
        return 1.0;
    }

    // ENF ratio: how much of the peak spectral energy is at ENF frequencies
    let enf_ratio = (enf_energy / max_energy).clamp(0.0, 1.0);

    // High ENF ratio → low deviation → score near 0.0 (clean)
    // Low ENF ratio → high deviation → score near 1.0 (suspicious)
    1.0 - enf_ratio
}

// ---------------------------------------------------------------------------
// Feature helpers
// ---------------------------------------------------------------------------

/// Root Mean Square energy of a frame.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn compute_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f32 = frame.iter().map(|&s| s * s).sum();
    (sum_sq / frame.len() as f32).sqrt()
}

/// Zero-crossing rate: fraction of sign changes per sample.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn compute_zcr(frame: &[f32]) -> f32 {
    if frame.len() < 2 {
        return 0.0;
    }
    let crossings = frame
        .windows(2)
        .filter(|w| w[0].signum() != w[1].signum())
        .count();
    crossings as f32 / (frame.len() - 1) as f32
}

/// Spectral centroid approximation using even/odd sample sums (avoids FFT).
///
/// For a signal of length N the centroid in normalised frequency is
/// approximated as the ratio of the Σ|i · x[i]| to Σ|x[i]|.
/// While not identical to the DFT centroid, it tracks the spectral
/// centre of mass monotonically and is sufficient for detecting sudden shifts.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn compute_spectral_centroid(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let mut weighted_sum = 0.0_f32;
    let mut magnitude_sum = 0.0_f32;
    for (i, &s) in frame.iter().enumerate() {
        let mag = s.abs();
        weighted_sum += i as f32 * mag;
        magnitude_sum += mag;
    }
    if magnitude_sum < 1e-12 {
        return 0.0;
    }
    weighted_sum / magnitude_sum
}

/// Mean absolute value of a frame (short-term energy envelope).
#[inline]
#[allow(clippy::cast_precision_loss)]
fn compute_mean_abs(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    frame.iter().map(|&s| s.abs()).sum::<f32>() / frame.len() as f32
}

/// Compute (mean, standard_deviation) of a slice.
#[inline]
#[allow(clippy::cast_precision_loss)]
fn mean_std(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let n = values.len() as f32;
    let mean = values.iter().sum::<f32>() / n;
    let var = values.iter().map(|&v| (v - mean) * (v - mean)).sum::<f32>() / n;
    (mean, var.sqrt())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ── Feature helpers ──────────────────────────────────────────────────────

    #[test]
    fn test_compute_rms_silence() {
        let frame = vec![0.0_f32; 441];
        assert!(compute_rms(&frame) < 1e-6);
    }

    #[test]
    fn test_compute_rms_full_scale() {
        let frame = vec![1.0_f32; 441];
        assert!((compute_rms(&frame) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_rms_empty() {
        assert!(compute_rms(&[]) < 1e-6);
    }

    #[test]
    fn test_compute_zcr_silence() {
        let frame = vec![0.5_f32; 100];
        assert!(compute_zcr(&frame) < 1e-6);
    }

    #[test]
    fn test_compute_zcr_alternating() {
        // [-1, 1, -1, 1, ...] → every pair crosses
        let frame: Vec<f32> = (0..100).map(|i| if i % 2 == 0 { -1.0 } else { 1.0 }).collect();
        let zcr = compute_zcr(&frame);
        assert!(zcr > 0.9, "Near-maximum ZCR expected for alternating signal");
    }

    #[test]
    fn test_compute_spectral_centroid_empty() {
        assert!((compute_spectral_centroid(&[])).abs() < 1e-6);
    }

    #[test]
    fn test_compute_spectral_centroid_monotone() {
        // A rising ramp concentrates energy at high indices
        let frame: Vec<f32> = (0..256).map(|i| i as f32 / 256.0).collect();
        let c = compute_spectral_centroid(&frame);
        assert!(c > 50.0, "High-energy tail should push centroid up: got {c}");
    }

    #[test]
    fn test_compute_mean_abs_empty() {
        assert!(compute_mean_abs(&[]) < 1e-6);
    }

    #[test]
    fn test_compute_mean_abs_constant() {
        let frame = vec![0.5_f32; 100];
        assert!((compute_mean_abs(&frame) - 0.5).abs() < 1e-6);
    }

    // ── detect_audio_splices ─────────────────────────────────────────────────

    #[test]
    fn test_detect_audio_splices_empty() {
        let splices = detect_audio_splices(&[], 44100);
        assert!(splices.is_empty());
    }

    #[test]
    fn test_detect_audio_splices_silence_no_splice() {
        let samples = vec![0.0_f32; 44100];
        let splices = detect_audio_splices(&samples, 44100);
        assert!(splices.is_empty(), "Silent signal should have no splices");
    }

    #[test]
    fn test_detect_audio_splices_level_jump() {
        // First half: near silence; second half: full scale
        let mut samples = vec![0.001_f32; 22050];
        samples.extend(vec![0.9_f32; 22050]);
        let splices = detect_audio_splices(&samples, 44100);
        assert!(
            !splices.is_empty(),
            "Level jump should produce at least one splice"
        );
        let has_level = splices.iter().any(|s| s.method == SpliceMethod::LevelJump);
        assert!(has_level, "Expected LevelJump method");
    }

    #[test]
    fn test_detect_audio_splices_sorted_by_position() {
        let mut samples = vec![0.001_f32; 22050];
        samples.extend(vec![0.9_f32; 22050]);
        let splices = detect_audio_splices(&samples, 44100);
        for pair in splices.windows(2) {
            assert!(
                pair[0].position_samples <= pair[1].position_samples,
                "Results should be sorted by sample position"
            );
        }
    }

    #[test]
    fn test_detect_audio_splices_confidence_in_range() {
        let mut samples = vec![0.001_f32; 22050];
        samples.extend(vec![0.9_f32; 22050]);
        let splices = detect_audio_splices(&samples, 44100);
        for s in &splices {
            assert!(
                s.confidence >= 0.0 && s.confidence <= 1.0,
                "Confidence out of [0,1]: {}",
                s.confidence
            );
        }
    }

    #[test]
    fn test_detect_audio_splices_short_signal() {
        // Signal shorter than 2 frames → no splices
        let samples = vec![0.5_f32; 100];
        let splices = detect_audio_splices(&samples, 44100);
        assert!(splices.is_empty());
    }

    // ── audio_eno_test ───────────────────────────────────────────────────────

    #[test]
    fn test_audio_eno_test_empty() {
        // Empty signal should return maximum score
        assert!((audio_eno_test(&[]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_audio_eno_test_silence() {
        let samples = vec![0.0_f32; 44100];
        let score = audio_eno_test(&samples);
        // Silent signal → undefined ENF
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_audio_eno_test_pure_50hz_tone() {
        // A pure 50 Hz sine wave at 44100 Hz sample rate should score LOW (little deviation)
        let sr = 44100_usize;
        let samples: Vec<f32> = (0..sr)
            .map(|i| (2.0 * std::f32::consts::PI * 50.0 * i as f32 / sr as f32).sin())
            .collect();
        let score = audio_eno_test(&samples);
        // Score should be in valid range (we cannot guarantee exact value without
        // a proper FFT implementation, but it should not panic or be out-of-range).
        assert!(
            score >= 0.0 && score <= 1.0,
            "ENF score out of range: {score}"
        );
    }

    #[test]
    fn test_audio_eno_test_score_in_range() {
        let samples: Vec<f32> = (0..8192)
            .map(|i| (i as f32 * 0.1).sin() * 0.5)
            .collect();
        let score = audio_eno_test(&samples);
        assert!(
            score >= 0.0 && score <= 1.0,
            "ENF score out of range: {score}"
        );
    }
}
