//! Zero crossing rate (ZCR) with frame-based computation.
//!
//! The zero crossing rate is the rate at which the signal changes sign.
//! It is commonly used as a simple voiced/unvoiced discriminator, a
//! percussion detector, and a noisiness measure.

/// Compute the zero crossing rate of a single audio frame.
///
/// The ZCR is the number of sign changes divided by the number of samples
/// minus one (i.e., the number of adjacent pairs).
///
/// # Arguments
/// * `samples` - Audio samples
///
/// # Returns
/// ZCR in crossings-per-sample (range 0.0–1.0).  Returns 0.0 for slices
/// shorter than 2 samples.
#[must_use]
pub fn zero_crossing_rate(samples: &[f32]) -> f32 {
    if samples.len() < 2 {
        return 0.0;
    }

    let mut crossings = 0usize;
    for i in 1..samples.len() {
        // A crossing occurs when adjacent samples have different signs.
        // Treat 0.0 as positive to avoid counting boundary cases twice.
        let sign_a = samples[i - 1] >= 0.0;
        let sign_b = samples[i] >= 0.0;
        if sign_a != sign_b {
            crossings += 1;
        }
    }

    crossings as f32 / (samples.len() - 1) as f32
}

/// Compute the zero crossing count (absolute number of crossings).
#[must_use]
pub fn zero_crossing_count(samples: &[f32]) -> usize {
    if samples.len() < 2 {
        return 0;
    }

    let mut crossings = 0usize;
    for i in 1..samples.len() {
        let sign_a = samples[i - 1] >= 0.0;
        let sign_b = samples[i] >= 0.0;
        if sign_a != sign_b {
            crossings += 1;
        }
    }
    crossings
}

/// Frame-based ZCR computation.
///
/// Splits the input signal into overlapping frames and computes the ZCR for
/// each frame.  Frames that are shorter than `frame_size` at the end of the
/// signal are discarded.
///
/// # Arguments
/// * `samples`    - Full audio signal
/// * `frame_size` - Number of samples per frame (must be ≥ 2)
/// * `hop_size`   - Number of samples between frame starts (must be ≥ 1)
///
/// # Returns
/// Vector of per-frame ZCR values.  Empty if `samples` has fewer samples
/// than `frame_size`, or if `frame_size` < 2 or `hop_size` < 1.
#[must_use]
pub fn zero_crossing_rate_framed(samples: &[f32], frame_size: usize, hop_size: usize) -> Vec<f32> {
    if frame_size < 2 || hop_size < 1 || samples.len() < frame_size {
        return Vec::new();
    }

    let n_frames = (samples.len() - frame_size) / hop_size + 1;
    let mut zcr_track = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * hop_size;
        let end = start + frame_size;
        if end > samples.len() {
            break;
        }
        zcr_track.push(zero_crossing_rate(&samples[start..end]));
    }

    zcr_track
}

/// Compute the mean ZCR over a framed signal.
///
/// Returns 0.0 if no frames are produced.
#[must_use]
pub fn mean_zcr(samples: &[f32], frame_size: usize, hop_size: usize) -> f32 {
    let track = zero_crossing_rate_framed(samples, frame_size, hop_size);
    if track.is_empty() {
        return 0.0;
    }
    track.iter().sum::<f32>() / track.len() as f32
}

/// Classify a frame as voiced or unvoiced based on its ZCR.
///
/// A ZCR above `threshold` suggests an unvoiced (noisy) segment, which tends
/// to have many high-frequency zero crossings.  A typical threshold is 0.1.
#[must_use]
pub fn is_unvoiced(zcr: f32, threshold: f32) -> bool {
    zcr > threshold
}

/// ZCR-based voiced/unvoiced decision for each frame.
///
/// Returns a boolean vector: `true` = unvoiced, `false` = voiced.
#[must_use]
pub fn voiced_unvoiced_frames(
    samples: &[f32],
    frame_size: usize,
    hop_size: usize,
    threshold: f32,
) -> Vec<bool> {
    zero_crossing_rate_framed(samples, frame_size, hop_size)
        .into_iter()
        .map(|zcr| is_unvoiced(zcr, threshold))
        .collect()
}

/// Statistics over a ZCR track.
#[derive(Debug, Clone)]
pub struct ZcrStats {
    /// Mean ZCR across all frames.
    pub mean: f32,
    /// Standard deviation of ZCR.
    pub std: f32,
    /// Minimum ZCR.
    pub min: f32,
    /// Maximum ZCR.
    pub max: f32,
    /// Fraction of frames classified as unvoiced using a 0.1 threshold.
    pub unvoiced_fraction: f32,
    /// Number of frames analysed.
    pub frame_count: usize,
}

impl ZcrStats {
    /// Returns `true` if the majority of frames are classified as unvoiced.
    #[must_use]
    pub fn is_mostly_unvoiced(&self) -> bool {
        self.unvoiced_fraction > 0.5
    }
}

/// Compute ZCR statistics over a framed signal.
///
/// Returns `None` if no frames are produced.
#[must_use]
pub fn zcr_statistics(samples: &[f32], frame_size: usize, hop_size: usize) -> Option<ZcrStats> {
    let track = zero_crossing_rate_framed(samples, frame_size, hop_size);
    if track.is_empty() {
        return None;
    }

    let n = track.len() as f32;
    let mean = track.iter().sum::<f32>() / n;
    let variance = track.iter().map(|&z| (z - mean) * (z - mean)).sum::<f32>() / n;
    let std = variance.sqrt();
    let min = track.iter().copied().fold(f32::INFINITY, f32::min);
    let max = track.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let unvoiced_count = track.iter().filter(|&&z| is_unvoiced(z, 0.1)).count();
    let unvoiced_fraction = unvoiced_count as f32 / n;

    Some(ZcrStats {
        mean,
        std,
        min: min.max(0.0),
        max: max.max(0.0),
        unvoiced_fraction,
        frame_count: track.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn sine_wave(freq: f32, sample_rate: f32, n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (2.0 * PI * freq * i as f32 / sample_rate).sin())
            .collect()
    }

    fn noise(n: usize) -> Vec<f32> {
        // Deterministic "noise": alternating +/- values
        (0..n)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect()
    }

    // ── zero_crossing_rate ────────────────────────────────────────────────────

    #[test]
    fn test_zcr_alternating_signal() {
        // Perfect alternating signal: every sample crosses zero.
        let samples = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        let zcr = zero_crossing_rate(&samples);
        assert!((zcr - 1.0).abs() < 1e-6, "Expected ZCR=1.0, got {zcr}");
    }

    #[test]
    fn test_zcr_constant_positive() {
        let samples = vec![1.0; 10];
        assert_eq!(zero_crossing_rate(&samples), 0.0);
    }

    #[test]
    fn test_zcr_constant_negative() {
        let samples = vec![-1.0; 10];
        assert_eq!(zero_crossing_rate(&samples), 0.0);
    }

    #[test]
    fn test_zcr_single_sample() {
        let samples = vec![1.0];
        assert_eq!(zero_crossing_rate(&samples), 0.0);
    }

    #[test]
    fn test_zcr_empty() {
        let samples: Vec<f32> = vec![];
        assert_eq!(zero_crossing_rate(&samples), 0.0);
    }

    #[test]
    fn test_zcr_range() {
        // ZCR must be in [0, 1].
        let samples = noise(64);
        let zcr = zero_crossing_rate(&samples);
        assert!(zcr >= 0.0 && zcr <= 1.0);
    }

    #[test]
    fn test_zcr_sine_wave() {
        // A 440 Hz sine at 44100 samples/sec crosses zero ~2*440 times per second.
        let sr = 44100.0_f32;
        let n = 44100;
        let samples = sine_wave(440.0, sr, n);
        let zcr = zero_crossing_rate(&samples);
        // Expected: 2 * freq / sample_rate = 2 * 440 / 44100 ≈ 0.01995
        let expected = 2.0 * 440.0 / sr;
        assert!(
            (zcr - expected).abs() < 0.002,
            "ZCR={zcr}, expected~{expected}"
        );
    }

    // ── zero_crossing_count ───────────────────────────────────────────────────

    #[test]
    fn test_zcr_count_alternating() {
        let samples = vec![1.0, -1.0, 1.0, -1.0];
        assert_eq!(zero_crossing_count(&samples), 3);
    }

    #[test]
    fn test_zcr_count_empty() {
        let empty: Vec<f32> = vec![];
        assert_eq!(zero_crossing_count(&empty), 0);
    }

    // ── zero_crossing_rate_framed ─────────────────────────────────────────────

    #[test]
    fn test_zcr_framed_basic_length() {
        let samples = vec![1.0; 100];
        let track = zero_crossing_rate_framed(&samples, 10, 5);
        // (100 - 10) / 5 + 1 = 19 frames
        assert_eq!(track.len(), 19);
    }

    #[test]
    fn test_zcr_framed_too_short() {
        let samples = vec![1.0; 5];
        let track = zero_crossing_rate_framed(&samples, 10, 5);
        assert!(track.is_empty());
    }

    #[test]
    fn test_zcr_framed_invalid_frame_size() {
        let samples = vec![1.0; 100];
        let track = zero_crossing_rate_framed(&samples, 1, 5);
        assert!(track.is_empty());
    }

    #[test]
    fn test_zcr_framed_invalid_hop_size() {
        let samples = vec![1.0; 100];
        let track = zero_crossing_rate_framed(&samples, 10, 0);
        assert!(track.is_empty());
    }

    #[test]
    fn test_zcr_framed_noise_higher_than_sine() {
        let sr = 44100.0;
        let n = 4096;
        let sine = sine_wave(220.0, sr, n);
        let noisy = noise(n);
        let sine_zcr = mean_zcr(&sine, 512, 256);
        let noise_zcr = mean_zcr(&noisy, 512, 256);
        assert!(
            noise_zcr > sine_zcr,
            "Noise ZCR ({noise_zcr}) should be > sine ZCR ({sine_zcr})"
        );
    }

    #[test]
    fn test_zcr_framed_all_values_in_range() {
        let samples = noise(1024);
        let track = zero_crossing_rate_framed(&samples, 64, 32);
        for &z in &track {
            assert!(z >= 0.0 && z <= 1.0, "ZCR {z} out of range");
        }
    }

    #[test]
    fn test_zcr_framed_constant_signal() {
        let samples = vec![0.5_f32; 256];
        let track = zero_crossing_rate_framed(&samples, 64, 32);
        for &z in &track {
            assert_eq!(z, 0.0);
        }
    }

    // ── mean_zcr ───────────────────────────────────────────────────────────────

    #[test]
    fn test_mean_zcr_constant() {
        let samples = vec![1.0; 100];
        assert_eq!(mean_zcr(&samples, 10, 5), 0.0);
    }

    #[test]
    fn test_mean_zcr_empty() {
        assert_eq!(mean_zcr(&[], 10, 5), 0.0);
    }

    // ── voiced_unvoiced ────────────────────────────────────────────────────────

    #[test]
    fn test_voiced_unvoiced_frames_sine_mostly_voiced() {
        let sr = 44100.0;
        let samples = sine_wave(220.0, sr, 4096);
        let decisions = voiced_unvoiced_frames(&samples, 512, 256, 0.1);
        let voiced_count = decisions.iter().filter(|&&v| !v).count();
        assert!(
            voiced_count > decisions.len() / 2,
            "Sine should be mostly voiced"
        );
    }

    #[test]
    fn test_voiced_unvoiced_frames_noise_mostly_unvoiced() {
        let samples = noise(4096);
        let decisions = voiced_unvoiced_frames(&samples, 512, 256, 0.1);
        let unvoiced_count = decisions.iter().filter(|&&v| v).count();
        assert!(
            unvoiced_count > decisions.len() / 2,
            "Noise should be mostly unvoiced"
        );
    }

    // ── zcr_statistics ────────────────────────────────────────────────────────

    #[test]
    fn test_zcr_stats_none_when_no_frames() {
        let empty: Vec<f32> = vec![];
        assert!(zcr_statistics(&empty, 64, 32).is_none());
    }

    #[test]
    fn test_zcr_stats_constant_zero() {
        let samples = vec![1.0_f32; 256];
        let stats = zcr_statistics(&samples, 64, 32).expect("ZCR statistics should succeed");
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.min, 0.0);
        assert_eq!(stats.max, 0.0);
        assert!(!stats.is_mostly_unvoiced());
    }

    #[test]
    fn test_zcr_stats_noise() {
        let samples = noise(4096);
        let stats = zcr_statistics(&samples, 512, 256).expect("ZCR statistics should succeed");
        assert!(stats.mean > 0.5, "Noise ZCR mean should be high");
        assert!(stats.is_mostly_unvoiced());
        assert!(stats.frame_count > 0);
    }

    #[test]
    fn test_zcr_stats_range_validity() {
        let sr = 44100.0;
        let samples = sine_wave(440.0, sr, 4096);
        let stats = zcr_statistics(&samples, 512, 256).expect("ZCR statistics should succeed");
        assert!(stats.min >= 0.0);
        assert!(stats.max <= 1.0);
        assert!(stats.std >= 0.0);
        assert!(stats.unvoiced_fraction >= 0.0 && stats.unvoiced_fraction <= 1.0);
    }
}
