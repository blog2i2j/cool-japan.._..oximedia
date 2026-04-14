//! Quality metrics for codec evaluation.
//!
//! Provides PSNR, SNR, and MSE calculations for comparing original and
//! reconstructed media data. These metrics are used to validate codec
//! round-trip fidelity and to verify that encoding quality meets minimum
//! thresholds.
//!
//! # Metrics
//!
//! - **MSE** (Mean Squared Error): Average squared difference between samples.
//! - **PSNR** (Peak Signal-to-Noise Ratio): Logarithmic quality measure in dB.
//! - **SNR** (Signal-to-Noise Ratio): Ratio of signal power to noise power.
//!
//! # Usage
//!
//! ```rust
//! use oximedia_codec::quality_metrics::{compute_psnr_u8, compute_mse_f32, compute_snr_f32};
//!
//! let original = [100u8, 150, 200, 50, 30];
//! let decoded  = [101u8, 149, 201, 49, 31];
//! let psnr = compute_psnr_u8(&original, &decoded);
//! assert!(psnr > 40.0, "near-identical samples should have high PSNR");
//! ```

/// Compute Mean Squared Error between two `u8` slices.
///
/// Returns `0.0` if both slices are empty. Only the overlapping portion
/// (min length) is compared.
#[must_use]
pub fn compute_mse_u8(original: &[u8], decoded: &[u8]) -> f64 {
    let len = original.len().min(decoded.len());
    if len == 0 {
        return 0.0;
    }
    let sum_sq: f64 = original[..len]
        .iter()
        .zip(decoded[..len].iter())
        .map(|(&a, &b)| {
            let diff = f64::from(a) - f64::from(b);
            diff * diff
        })
        .sum();
    sum_sq / len as f64
}

/// Compute PSNR (Peak Signal-to-Noise Ratio) for `u8` data.
///
/// Peak value is 255. Returns `f64::INFINITY` when the signals are identical
/// (MSE == 0).
#[must_use]
pub fn compute_psnr_u8(original: &[u8], decoded: &[u8]) -> f64 {
    let mse = compute_mse_u8(original, decoded);
    if mse < f64::EPSILON {
        return f64::INFINITY;
    }
    let peak = 255.0_f64;
    10.0 * (peak * peak / mse).log10()
}

/// Compute Mean Squared Error between two `f32` slices.
///
/// Compares only the overlapping portion.
#[must_use]
pub fn compute_mse_f32(original: &[f32], decoded: &[f32]) -> f64 {
    let len = original.len().min(decoded.len());
    if len == 0 {
        return 0.0;
    }
    let sum_sq: f64 = original[..len]
        .iter()
        .zip(decoded[..len].iter())
        .map(|(&a, &b)| {
            let diff = f64::from(a) - f64::from(b);
            diff * diff
        })
        .sum();
    sum_sq / len as f64
}

/// Compute PSNR for normalised `f32` audio data (peak = 1.0).
///
/// Returns `f64::INFINITY` when signals are identical.
#[must_use]
pub fn compute_psnr_f32(original: &[f32], decoded: &[f32]) -> f64 {
    let mse = compute_mse_f32(original, decoded);
    if mse < f64::EPSILON {
        return f64::INFINITY;
    }
    let peak = 1.0_f64;
    10.0 * (peak * peak / mse).log10()
}

/// Compute Signal-to-Noise Ratio for `f32` data.
///
/// SNR = 10 * log10(signal_power / noise_power).
/// Returns `f64::INFINITY` when noise is zero.
#[must_use]
pub fn compute_snr_f32(original: &[f32], decoded: &[f32]) -> f64 {
    let len = original.len().min(decoded.len());
    if len == 0 {
        return 0.0;
    }
    let signal_power: f64 = original[..len]
        .iter()
        .map(|&s| f64::from(s) * f64::from(s))
        .sum::<f64>()
        / len as f64;
    let noise_power: f64 = original[..len]
        .iter()
        .zip(decoded[..len].iter())
        .map(|(&a, &b)| {
            let d = f64::from(a) - f64::from(b);
            d * d
        })
        .sum::<f64>()
        / len as f64;

    if noise_power < f64::EPSILON {
        return f64::INFINITY;
    }
    if signal_power < f64::EPSILON {
        return 0.0;
    }
    10.0 * (signal_power / noise_power).log10()
}

/// Compute PSNR for `u16` data (useful for 10-bit or 12-bit video).
///
/// `bit_depth` determines the peak value (2^bit_depth - 1).
#[must_use]
pub fn compute_psnr_u16(original: &[u16], decoded: &[u16], bit_depth: u8) -> f64 {
    let len = original.len().min(decoded.len());
    if len == 0 {
        return 0.0;
    }
    let sum_sq: f64 = original[..len]
        .iter()
        .zip(decoded[..len].iter())
        .map(|(&a, &b)| {
            let diff = f64::from(a) - f64::from(b);
            diff * diff
        })
        .sum();
    let mse = sum_sq / len as f64;
    if mse < f64::EPSILON {
        return f64::INFINITY;
    }
    let peak = (1u32 << bit_depth) as f64 - 1.0;
    10.0 * (peak * peak / mse).log10()
}

/// Compute the Structural Similarity Index (simplified) for `u8` data.
///
/// This is a simplified version using global statistics rather than the
/// standard 11x11 Gaussian windowed approach, suitable for quick quality
/// checks in codec testing.
///
/// Returns a value in `[0.0, 1.0]` where 1.0 indicates identical signals.
#[must_use]
pub fn compute_ssim_simplified_u8(original: &[u8], decoded: &[u8]) -> f64 {
    let len = original.len().min(decoded.len());
    if len == 0 {
        return 1.0;
    }

    let n = len as f64;

    // Mean
    let mu_x: f64 = original[..len].iter().map(|&v| f64::from(v)).sum::<f64>() / n;
    let mu_y: f64 = decoded[..len].iter().map(|&v| f64::from(v)).sum::<f64>() / n;

    // Variance and covariance
    let mut var_x = 0.0_f64;
    let mut var_y = 0.0_f64;
    let mut cov_xy = 0.0_f64;

    for i in 0..len {
        let dx = f64::from(original[i]) - mu_x;
        let dy = f64::from(decoded[i]) - mu_y;
        var_x += dx * dx;
        var_y += dy * dy;
        cov_xy += dx * dy;
    }
    var_x /= n;
    var_y /= n;
    cov_xy /= n;

    // Constants (for 8-bit dynamic range: L=255)
    let c1 = (0.01 * 255.0) * (0.01 * 255.0); // 6.5025
    let c2 = (0.03 * 255.0) * (0.03 * 255.0); // 58.5225

    let numerator = (2.0 * mu_x * mu_y + c1) * (2.0 * cov_xy + c2);
    let denominator = (mu_x * mu_x + mu_y * mu_y + c1) * (var_x + var_y + c2);

    if denominator < f64::EPSILON {
        return 1.0;
    }

    (numerator / denominator).clamp(0.0, 1.0)
}

/// Quality assessment result for a codec round-trip test.
#[derive(Debug, Clone)]
pub struct QualityAssessment {
    /// PSNR in dB.
    pub psnr_db: f64,
    /// Mean Squared Error.
    pub mse: f64,
    /// SNR in dB.
    pub snr_db: f64,
    /// Whether the quality meets the minimum threshold.
    pub passes_threshold: bool,
    /// Minimum PSNR threshold that was applied.
    pub threshold_db: f64,
}

/// Perform a full quality assessment on `u8` data.
///
/// `min_psnr_db` is the threshold: if the measured PSNR is at least this
/// value, `passes_threshold` is set to `true`.
#[must_use]
pub fn assess_quality_u8(original: &[u8], decoded: &[u8], min_psnr_db: f64) -> QualityAssessment {
    let mse = compute_mse_u8(original, decoded);
    let psnr_db = compute_psnr_u8(original, decoded);

    // Convert u8 to f32 for SNR
    let orig_f32: Vec<f32> = original.iter().map(|&v| v as f32 / 255.0).collect();
    let dec_f32: Vec<f32> = decoded.iter().map(|&v| v as f32 / 255.0).collect();
    let snr_db = compute_snr_f32(&orig_f32, &dec_f32);

    QualityAssessment {
        psnr_db,
        mse,
        snr_db,
        passes_threshold: psnr_db >= min_psnr_db,
        threshold_db: min_psnr_db,
    }
}

/// Perform a full quality assessment on `f32` data.
#[must_use]
pub fn assess_quality_f32(
    original: &[f32],
    decoded: &[f32],
    min_psnr_db: f64,
) -> QualityAssessment {
    let mse = compute_mse_f32(original, decoded);
    let psnr_db = compute_psnr_f32(original, decoded);
    let snr_db = compute_snr_f32(original, decoded);

    QualityAssessment {
        psnr_db,
        mse,
        snr_db,
        passes_threshold: psnr_db >= min_psnr_db,
        threshold_db: min_psnr_db,
    }
}

// =============================================================================
// Tests — Round-trip encode/decode quality (PSNR > threshold)
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ── MSE tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_mse_u8_identical() {
        let data = [10u8, 20, 30, 40, 50];
        let mse = compute_mse_u8(&data, &data);
        assert!(mse < f64::EPSILON, "MSE of identical signals should be 0");
    }

    #[test]
    fn test_mse_u8_known_difference() {
        let a = [100u8, 200, 150];
        let b = [101u8, 199, 151];
        // Each diff is ±1 → sum_sq = 1+1+1 = 3 → MSE = 1.0
        let mse = compute_mse_u8(&a, &b);
        assert!((mse - 1.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mse_u8_empty() {
        let mse = compute_mse_u8(&[], &[]);
        assert!(mse < f64::EPSILON);
    }

    #[test]
    fn test_mse_f32_identical() {
        let data = [0.1f32, 0.5, -0.3, 0.9, -0.8];
        let mse = compute_mse_f32(&data, &data);
        assert!(mse < f64::EPSILON);
    }

    #[test]
    fn test_mse_f32_known_difference() {
        let a = [1.0f32, 0.0, 0.0];
        let b = [0.0f32, 0.0, 0.0];
        // diff = 1 → sum_sq = 1 → MSE = 1/3
        let mse = compute_mse_f32(&a, &b);
        assert!((mse - 1.0 / 3.0).abs() < 1e-10);
    }

    // ── PSNR tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_psnr_u8_identical() {
        let data = [128u8; 100];
        let psnr = compute_psnr_u8(&data, &data);
        assert!(
            psnr.is_infinite(),
            "identical signals should have infinite PSNR"
        );
    }

    #[test]
    fn test_psnr_u8_small_error() {
        let original: Vec<u8> = (0..=255).collect();
        let mut decoded = original.clone();
        // Introduce ±1 error on every other sample
        for i in (0..decoded.len()).step_by(2) {
            decoded[i] = decoded[i].saturating_add(1);
        }
        let psnr = compute_psnr_u8(&original, &decoded);
        // ±1 error on half the samples → MSE = 0.5 → PSNR ≈ 51.1 dB
        assert!(psnr > 40.0, "±1 error should yield high PSNR, got {psnr}");
    }

    #[test]
    fn test_psnr_u8_large_error() {
        let original = vec![128u8; 100];
        let decoded = vec![0u8; 100];
        let psnr = compute_psnr_u8(&original, &decoded);
        // MSE = 128² = 16384 → PSNR ≈ 6.0 dB
        assert!(psnr < 10.0, "large error should yield low PSNR, got {psnr}");
    }

    #[test]
    fn test_psnr_f32_identical() {
        let data = [0.5f32; 50];
        let psnr = compute_psnr_f32(&data, &data);
        assert!(psnr.is_infinite());
    }

    #[test]
    fn test_psnr_f32_small_error() {
        let original: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let decoded: Vec<f32> = original.iter().map(|&s| s + 0.001).collect();
        let psnr = compute_psnr_f32(&original, &decoded);
        assert!(psnr > 50.0, "tiny error should yield high PSNR, got {psnr}");
    }

    // ── SNR tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_snr_f32_identical() {
        let data = [0.5f32, -0.3, 0.8, -0.1, 0.9];
        let snr = compute_snr_f32(&data, &data);
        assert!(snr.is_infinite());
    }

    #[test]
    fn test_snr_f32_noisy() {
        let original: Vec<f32> = (0..100).map(|i| (i as f32 / 50.0) - 1.0).collect();
        let decoded: Vec<f32> = original.iter().map(|&s| s + 0.1).collect();
        let snr = compute_snr_f32(&original, &decoded);
        assert!(
            snr > 0.0,
            "signal with small noise should have positive SNR"
        );
    }

    #[test]
    fn test_snr_f32_zero_signal() {
        let original = [0.0f32; 100];
        let decoded = [0.1f32; 100];
        let snr = compute_snr_f32(&original, &decoded);
        assert!(
            snr <= 0.0,
            "zero signal with noise should have non-positive SNR"
        );
    }

    // ── PSNR u16 tests ──────────────────────────────────────────────────────

    #[test]
    fn test_psnr_u16_identical() {
        let data: Vec<u16> = (0..100).map(|i| (i * 10) as u16).collect();
        let psnr = compute_psnr_u16(&data, &data, 10);
        assert!(psnr.is_infinite());
    }

    #[test]
    fn test_psnr_u16_10bit_small_error() {
        let original: Vec<u16> = (0..1024).collect();
        let mut decoded = original.clone();
        for i in (0..decoded.len()).step_by(2) {
            decoded[i] = decoded[i].saturating_add(1);
        }
        let psnr = compute_psnr_u16(&original, &decoded, 10);
        // 10-bit peak = 1023
        assert!(
            psnr > 50.0,
            "±1 error on 10-bit should yield high PSNR, got {psnr}"
        );
    }

    // ── SSIM tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_ssim_identical() {
        let data: Vec<u8> = (0..=255).collect();
        let ssim = compute_ssim_simplified_u8(&data, &data);
        assert!(
            (ssim - 1.0).abs() < 1e-6,
            "identical signals should have SSIM=1, got {ssim}"
        );
    }

    #[test]
    fn test_ssim_similar() {
        let original: Vec<u8> = (0..=255).collect();
        let mut decoded = original.clone();
        for i in (0..decoded.len()).step_by(3) {
            decoded[i] = decoded[i].saturating_add(1);
        }
        let ssim = compute_ssim_simplified_u8(&original, &decoded);
        assert!(
            ssim > 0.99,
            "nearly identical signals should have high SSIM, got {ssim}"
        );
    }

    #[test]
    fn test_ssim_empty() {
        let ssim = compute_ssim_simplified_u8(&[], &[]);
        assert!((ssim - 1.0).abs() < 1e-6);
    }

    // ── Quality assessment tests ────────────────────────────────────────────

    #[test]
    fn test_assess_quality_u8_passes() {
        let original: Vec<u8> = (0..=255).collect();
        let decoded = original.clone();
        let result = assess_quality_u8(&original, &decoded, 30.0);
        assert!(
            result.passes_threshold,
            "identical data should pass any threshold"
        );
        assert!(result.psnr_db.is_infinite());
        assert!(result.mse < f64::EPSILON);
    }

    #[test]
    fn test_assess_quality_u8_fails() {
        let original = vec![128u8; 100];
        let decoded = vec![0u8; 100];
        let result = assess_quality_u8(&original, &decoded, 30.0);
        assert!(
            !result.passes_threshold,
            "large error should fail threshold"
        );
        assert!(result.psnr_db < 30.0);
    }

    #[test]
    fn test_assess_quality_f32_passes() {
        let original: Vec<f32> = (0..100).map(|i| i as f32 / 100.0).collect();
        let decoded: Vec<f32> = original.iter().map(|&s| s + 0.0001).collect();
        let result = assess_quality_f32(&original, &decoded, 60.0);
        assert!(
            result.passes_threshold,
            "tiny error should pass 60 dB threshold, got {} dB",
            result.psnr_db
        );
    }

    // ── PCM codec round-trip PSNR tests ─────────────────────────────────────

    #[test]
    fn test_pcm_f32_roundtrip_psnr() {
        use crate::audio::{AudioFrame, SampleFormat};
        use crate::pcm::{ByteOrder, PcmConfig, PcmDecoder, PcmEncoder, PcmFormat};

        let config = PcmConfig {
            format: PcmFormat::F32,
            byte_order: ByteOrder::Little,
            sample_rate: 48000,
            channels: 1,
        };
        let enc = PcmEncoder::new(config.clone());
        let dec = PcmDecoder::new(config);

        // Generate a 440 Hz sine wave
        let samples: Vec<f32> = (0..4800)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 48000.0).sin())
            .collect();

        let raw_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame = AudioFrame::new(raw_bytes, 4800, 48000, 1, SampleFormat::F32);
        let encoded = enc.encode_frame(&frame).expect("encode");
        let decoded_frame = dec.decode_bytes(&encoded).expect("decode");

        // Extract decoded samples
        let decoded_samples: Vec<f32> = decoded_frame
            .samples
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let psnr = compute_psnr_f32(&samples, &decoded_samples);
        assert!(
            psnr.is_infinite(),
            "PCM F32 round-trip should be lossless (infinite PSNR), got {psnr}"
        );
    }

    #[test]
    fn test_pcm_i16_roundtrip_psnr_above_threshold() {
        use crate::audio::{AudioFrame, SampleFormat};
        use crate::pcm::{ByteOrder, PcmConfig, PcmDecoder, PcmEncoder, PcmFormat};

        let config = PcmConfig {
            format: PcmFormat::I16,
            byte_order: ByteOrder::Little,
            sample_rate: 44100,
            channels: 2,
        };
        let enc = PcmEncoder::new(config.clone());
        let dec = PcmDecoder::new(config);

        // Generate stereo signal
        let samples: Vec<f32> = (0..8820)
            .map(|i| {
                let t = i as f32 / 44100.0;
                let ch = i % 2;
                let freq = if ch == 0 { 440.0 } else { 880.0 };
                (2.0 * std::f32::consts::PI * freq * t).sin() * 0.8
            })
            .collect();

        let raw_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame = AudioFrame::new(raw_bytes, 4410, 44100, 2, SampleFormat::F32);
        let encoded = enc.encode_frame(&frame).expect("encode");
        let decoded_frame = dec.decode_bytes(&encoded).expect("decode");

        let decoded_samples: Vec<f32> = decoded_frame
            .samples
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // I16 quantization introduces ≈ 96 dB SNR → PSNR should be very high
        let psnr = compute_psnr_f32(&samples, &decoded_samples);
        assert!(
            psnr > 80.0,
            "PCM I16 round-trip should have PSNR > 80 dB, got {psnr}"
        );
    }

    #[test]
    fn test_pcm_u8_roundtrip_psnr_above_threshold() {
        use crate::audio::{AudioFrame, SampleFormat};
        use crate::pcm::{ByteOrder, PcmConfig, PcmDecoder, PcmEncoder, PcmFormat};

        let config = PcmConfig {
            format: PcmFormat::U8,
            byte_order: ByteOrder::Little,
            sample_rate: 22050,
            channels: 1,
        };
        let enc = PcmEncoder::new(config.clone());
        let dec = PcmDecoder::new(config);

        let samples: Vec<f32> = (0..2205)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 22050.0).sin() * 0.5)
            .collect();

        let raw_bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
        let frame = AudioFrame::new(raw_bytes, 2205, 22050, 1, SampleFormat::F32);
        let encoded = enc.encode_frame(&frame).expect("encode");
        let decoded_frame = dec.decode_bytes(&encoded).expect("decode");

        let decoded_samples: Vec<f32> = decoded_frame
            .samples
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        // U8 has only 8-bit resolution → ~48 dB SNR
        let psnr = compute_psnr_f32(&samples, &decoded_samples);
        assert!(
            psnr > 30.0,
            "PCM U8 round-trip should have PSNR > 30 dB, got {psnr}"
        );
    }
}
