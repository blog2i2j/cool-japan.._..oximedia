//! Simplified FLAC codec using fixed linear prediction + Rice coding.
//!
//! This module implements a lightweight FLAC-style encoder/decoder based on
//! fixed predictors (order 0-4) rather than full LPC analysis. It is useful
//! for educational purposes and lightweight lossless audio compression.
//!
//! ## Fixed predictor formulas
//!
//! | Order | Residual formula                                           |
//! |-------|------------------------------------------------------------|
//! | 0     | `r[i] = s[i]`                                              |
//! | 1     | `r[i] = s[i] - s[i-1]`                                     |
//! | 2     | `r[i] = s[i] - 2*s[i-1] + s[i-2]`                         |
//! | 3     | `r[i] = s[i] - 3*s[i-1] + 3*s[i-2] - s[i-3]`             |
//! | 4     | `r[i] = s[i] - 4*s[i-1] + 6*s[i-2] - 4*s[i-3] + s[i-4]` |
//!
//! ## Rice coding
//!
//! Residuals are zigzag-folded (signed → unsigned) then split into a unary
//! quotient and binary remainder using a Rice parameter `k`.

#![forbid(unsafe_code)]

// =============================================================================
// Stream info & frame header
// =============================================================================

/// FLAC stream info metadata.
#[derive(Debug, Clone)]
pub struct FlacStreamInfo {
    /// Minimum block size in samples.
    pub min_block_size: u16,
    /// Maximum block size in samples.
    pub max_block_size: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u8,
    /// Bits per sample (e.g. 16).
    pub bits_per_sample: u8,
    /// Total samples in stream (0 if unknown).
    pub total_samples: u64,
}

/// FLAC frame header.
#[derive(Debug, Clone)]
pub struct FlacFrameHeader {
    /// Block size (samples per channel) in this frame.
    pub block_size: u16,
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of channels.
    pub channels: u8,
    /// Bits per sample.
    pub bits_per_sample: u8,
    /// Frame number (sequential).
    pub frame_number: u32,
}

// =============================================================================
// Encoder configuration
// =============================================================================

/// FLAC encoder configuration.
#[derive(Debug, Clone)]
pub struct FlacEncoderConfig {
    /// Sample rate in Hz.
    pub sample_rate: u32,
    /// Number of audio channels.
    pub channels: u8,
    /// Bits per sample (8, 16, 24).
    pub bits_per_sample: u8,
    /// Block size in samples per channel (default 4096).
    pub block_size: u16,
    /// Compression level 0-8 (higher = try more predictor orders). Default 5.
    pub compression_level: u8,
}

impl Default for FlacEncoderConfig {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            channels: 2,
            bits_per_sample: 16,
            block_size: 4096,
            compression_level: 5,
        }
    }
}

// =============================================================================
// Rice coding
// =============================================================================

/// Zigzag-encode a signed value to unsigned: `n >= 0 → 2n`, `n < 0 → 2|n|-1`.
#[inline]
fn zigzag_encode(v: i32) -> u32 {
    if v >= 0 {
        (v as u32) << 1
    } else {
        ((-v - 1) as u32) << 1 | 1
    }
}

/// Zigzag-decode an unsigned value back to signed.
#[inline]
fn zigzag_decode(u: u32) -> i32 {
    if u & 1 == 0 {
        (u >> 1) as i32
    } else {
        -((u >> 1) as i32) - 1
    }
}

/// Rice-encode a sequence of residuals with parameter `k`.
///
/// Each residual is zigzag-folded, then the quotient (`zigzag >> k`) is
/// unary-coded (q zeros followed by a 1) and the remainder is binary-coded
/// in `k` bits. Bits are packed MSB-first into bytes.
fn rice_encode(residuals: &[i32], rice_param: u8) -> Vec<u8> {
    let k = rice_param;
    let mut bits: Vec<bool> = Vec::new();

    for &r in residuals {
        let u = zigzag_encode(r);
        let quotient = u >> k;
        let remainder = u & ((1u32 << k).wrapping_sub(1));

        // Unary: quotient zeros then a 1
        for _ in 0..quotient {
            bits.push(false);
        }
        bits.push(true);

        // Binary remainder (MSB first)
        for bit_idx in (0..k).rev() {
            bits.push((remainder >> bit_idx) & 1 != 0);
        }
    }

    // Pack into bytes
    let mut out = Vec::with_capacity((bits.len() + 7) / 8);
    let mut byte = 0u8;
    let mut fill = 0u8;
    for bit in bits {
        byte = (byte << 1) | u8::from(bit);
        fill += 1;
        if fill == 8 {
            out.push(byte);
            byte = 0;
            fill = 0;
        }
    }
    if fill > 0 {
        out.push(byte << (8 - fill));
    }
    out
}

/// Rice-decode `count` residuals from `data` with parameter `k`.
fn rice_decode(data: &[u8], count: usize, rice_param: u8) -> Result<Vec<i32>, String> {
    let k = rice_param;
    let mut byte_pos = 0usize;
    let mut bit_pos = 0u8;

    let read_bit = |bp: &mut usize, bi: &mut u8| -> Result<bool, String> {
        if *bp >= data.len() {
            return Err("Rice decode: unexpected end of data".to_string());
        }
        let bit = (data[*bp] >> (7 - *bi)) & 1 != 0;
        *bi += 1;
        if *bi == 8 {
            *bp += 1;
            *bi = 0;
        }
        Ok(bit)
    };

    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        // Read unary quotient: count zeros until a 1
        let mut quotient = 0u32;
        loop {
            let bit = read_bit(&mut byte_pos, &mut bit_pos)?;
            if bit {
                break;
            }
            quotient += 1;
            if quotient > 1_048_576 {
                return Err("Rice decode: quotient overflow (corrupt data)".to_string());
            }
        }

        // Read k-bit remainder
        let mut remainder = 0u32;
        for _ in 0..k {
            let bit = read_bit(&mut byte_pos, &mut bit_pos)?;
            remainder = (remainder << 1) | u32::from(bit);
        }

        let u = (quotient << k) | remainder;
        out.push(zigzag_decode(u));
    }

    Ok(out)
}

// =============================================================================
// Fixed linear prediction
// =============================================================================

/// Compute the optimal fixed predictor order (0-4) for a block of samples.
///
/// Tries each order, picks the one yielding the smallest sum-of-absolute
/// residuals (SAR). `compression_level` limits the maximum order tested
/// (level 0-1 → max order 1, 2-4 → max order 2, 5-8 → max order 4).
fn optimal_predictor_order(samples: &[i16], compression_level: u8) -> u8 {
    if samples.is_empty() {
        return 0;
    }
    let max_order = match compression_level {
        0..=1 => 1u8,
        2..=4 => 2,
        _ => 4,
    };
    let max_order = max_order.min(samples.len().saturating_sub(1) as u8).min(4);

    let mut best_order = 0u8;
    let mut best_cost = u64::MAX;

    for order in 0..=max_order {
        let residuals = fixed_predict(samples, order);
        let cost: u64 = residuals.iter().map(|r| r.unsigned_abs() as u64).sum();
        if cost < best_cost {
            best_cost = cost;
            best_order = order;
        }
    }
    best_order
}

/// Apply fixed linear prediction of the given order, returning residuals.
///
/// The first `order` samples are warmup and not included in residuals.
fn fixed_predict(samples: &[i16], order: u8) -> Vec<i32> {
    let n = samples.len();
    let o = order as usize;
    if n <= o {
        return Vec::new();
    }
    let s: Vec<i32> = samples.iter().map(|&v| v as i32).collect();

    let mut residuals = Vec::with_capacity(n - o);
    for i in o..n {
        let r = match order {
            0 => s[i],
            1 => s[i] - s[i - 1],
            2 => s[i] - 2 * s[i - 1] + s[i - 2],
            3 => s[i] - 3 * s[i - 1] + 3 * s[i - 2] - s[i - 3],
            4 => s[i] - 4 * s[i - 1] + 6 * s[i - 2] - 4 * s[i - 3] + s[i - 4],
            _ => s[i],
        };
        residuals.push(r);
    }
    residuals
}

/// Undo fixed linear prediction — reconstruct samples from residuals + warmup.
fn fixed_restore(residuals: &[i32], order: u8, warmup: &[i16]) -> Vec<i16> {
    let o = order as usize;
    let mut out: Vec<i32> = warmup.iter().map(|&v| v as i32).collect();

    for &r in residuals {
        let n = out.len();
        let sample = match order {
            0 => r,
            1 => r + out[n - 1],
            2 => r + 2 * out[n - 1] - out[n - 2],
            3 => r + 3 * out[n - 1] - 3 * out[n - 2] + out[n - 3],
            4 => r + 4 * out[n - 1] - 6 * out[n - 2] + 4 * out[n - 3] - out[n - 4],
            _ => r,
        };
        out.push(sample);
    }

    out.iter().map(|&v| v as i16).collect()
}

/// Select optimal Rice parameter for a set of residuals.
fn optimal_rice_param(residuals: &[i32]) -> u8 {
    if residuals.is_empty() {
        return 0;
    }
    let mut best_k = 0u8;
    let mut best_cost = u64::MAX;
    for k in 0..=14u8 {
        let cost: u64 = residuals
            .iter()
            .map(|&r| {
                let u = zigzag_encode(r);
                1u64 + u64::from(k) + u64::from(u >> k)
            })
            .sum();
        if cost < best_cost {
            best_cost = cost;
            best_k = k;
        }
    }
    best_k
}

// =============================================================================
// Frame-level encode / decode
// =============================================================================

/// Encode i16 samples into a single FLAC-style frame.
///
/// Uses fixed linear prediction (order 0-4) + Rice coding.
///
/// Binary layout:
/// ```text
/// [1B order] [1B rice_param] [2B warmup_count]
/// [warmup_count * 2B warmup samples (big-endian i16)]
/// [4B residual_count (big-endian u32)]
/// [4B rice_byte_len (big-endian u32)]
/// [rice_byte_len bytes of Rice-coded residuals]
/// ```
pub fn encode_flac_frame(samples: &[i16], config: &FlacEncoderConfig) -> Vec<u8> {
    if samples.is_empty() {
        // Minimal empty frame: order=0, k=0, 0 warmup, 0 residuals, 0 rice bytes
        return vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
    }

    let order = optimal_predictor_order(samples, config.compression_level);
    let residuals = fixed_predict(samples, order);
    let k = optimal_rice_param(&residuals);
    let rice_bytes = rice_encode(&residuals, k);

    let warmup_count = (order as usize).min(samples.len());
    let mut out = Vec::new();

    // Header
    out.push(order);
    out.push(k);
    let wc = warmup_count as u16;
    out.extend_from_slice(&wc.to_be_bytes());

    // Warmup samples
    for &s in &samples[..warmup_count] {
        out.extend_from_slice(&s.to_be_bytes());
    }

    // Residual count
    let rc = residuals.len() as u32;
    out.extend_from_slice(&rc.to_be_bytes());

    // Rice data length + data
    let rl = rice_bytes.len() as u32;
    out.extend_from_slice(&rl.to_be_bytes());
    out.extend_from_slice(&rice_bytes);

    out
}

/// Decode a FLAC-style frame back to i16 samples.
pub fn decode_flac_frame(data: &[u8], _info: &FlacStreamInfo) -> Result<Vec<i16>, String> {
    if data.len() < 12 {
        return Err("FLAC frame too short".to_string());
    }

    let order = data[0];
    if order > 4 {
        return Err(format!("Invalid predictor order: {order}"));
    }
    let k = data[1];
    if k > 30 {
        return Err(format!("Invalid Rice parameter: {k}"));
    }
    let warmup_count = u16::from_be_bytes([data[2], data[3]]) as usize;

    let mut pos = 4;

    // Read warmup
    if pos + warmup_count * 2 > data.len() {
        return Err("FLAC frame: warmup overruns data".to_string());
    }
    let mut warmup = Vec::with_capacity(warmup_count);
    for _ in 0..warmup_count {
        let s = i16::from_be_bytes([data[pos], data[pos + 1]]);
        warmup.push(s);
        pos += 2;
    }

    // Residual count
    if pos + 4 > data.len() {
        return Err("FLAC frame: missing residual count".to_string());
    }
    let residual_count =
        u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
    pos += 4;

    // Rice data length
    if pos + 4 > data.len() {
        return Err("FLAC frame: missing rice length".to_string());
    }
    let rice_len =
        u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
    pos += 4;

    if pos + rice_len > data.len() {
        return Err("FLAC frame: rice data overruns frame".to_string());
    }
    let rice_data = &data[pos..pos + rice_len];

    // Decode residuals
    let residuals = if residual_count == 0 {
        Vec::new()
    } else {
        rice_decode(rice_data, residual_count, k)?
    };

    // Restore signal
    let samples = fixed_restore(&residuals, order, &warmup);
    Ok(samples)
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_predict_order0() {
        let samples: Vec<i16> = vec![10, 20, 30, 40, 50];
        let residuals = fixed_predict(&samples, 0);
        // Order 0: residual = sample itself
        let expected: Vec<i32> = vec![10, 20, 30, 40, 50];
        assert_eq!(residuals, expected, "Order 0 should be identity");
    }

    #[test]
    fn test_fixed_predict_order1_constant() {
        let samples: Vec<i16> = vec![100, 100, 100, 100];
        let residuals = fixed_predict(&samples, 1);
        // s[i] - s[i-1] = 0 for constant
        assert!(
            residuals.iter().all(|&r| r == 0),
            "Constant signal should produce all-zero order-1 residuals: {:?}",
            residuals
        );
    }

    #[test]
    fn test_fixed_predict_restore_roundtrip() {
        let samples: Vec<i16> = vec![10, -5, 300, -200, 0, 127, -128, 500];
        for order in 0..=4u8 {
            if samples.len() <= order as usize {
                continue;
            }
            let residuals = fixed_predict(&samples, order);
            let warmup = &samples[..order as usize];
            let restored = fixed_restore(&residuals, order, warmup);
            assert_eq!(
                restored, samples,
                "Order {order} predict-restore roundtrip must be lossless"
            );
        }
    }

    #[test]
    fn test_rice_encode_decode_roundtrip() {
        let residuals = vec![0i32, 1, -1, 5, -5, 100, -100, 0];
        for k in 0..=6u8 {
            let encoded = rice_encode(&residuals, k);
            let decoded =
                rice_decode(&encoded, residuals.len(), k).expect("rice decode should succeed");
            assert_eq!(decoded, residuals, "Rice roundtrip failed for k={k}");
        }
    }

    #[test]
    fn test_rice_encode_zeros() {
        let zeros = vec![0i32; 64];
        let k = optimal_rice_param(&zeros);
        assert_eq!(k, 0, "All zeros should use k=0");
        let encoded = rice_encode(&zeros, k);
        // k=0: each zero is zigzag(0)=0, unary 0 = just a '1' bit → 1 bit each
        // 64 bits = 8 bytes
        assert_eq!(
            encoded.len(),
            8,
            "64 zero residuals at k=0 should be 8 bytes"
        );
    }

    #[test]
    fn test_optimal_predictor_silence() {
        let silence: Vec<i16> = vec![0; 128];
        let order = optimal_predictor_order(&silence, 5);
        assert_eq!(order, 0, "Silence should pick order 0");
    }

    #[test]
    fn test_optimal_predictor_linear_ramp() {
        let ramp: Vec<i16> = (0..128).map(|i| i as i16).collect();
        let order = optimal_predictor_order(&ramp, 5);
        // A linear ramp has zero residuals at order >= 1; order 1 should win
        assert!(
            order >= 1,
            "Linear ramp should pick order >= 1, got {order}"
        );
    }

    #[test]
    fn test_encode_decode_frame_roundtrip() {
        let config = FlacEncoderConfig::default();
        let info = FlacStreamInfo {
            min_block_size: 4096,
            max_block_size: 4096,
            sample_rate: 44100,
            channels: 2,
            bits_per_sample: 16,
            total_samples: 0,
        };

        // Generate a test signal: ramp + sine
        let samples: Vec<i16> = (0..512)
            .map(|i| {
                let ramp = (i as f64 / 512.0 * 1000.0) as i16;
                let sine = (100.0 * (i as f64 * 0.1).sin()) as i16;
                ramp.saturating_add(sine)
            })
            .collect();

        let encoded = encode_flac_frame(&samples, &config);
        let decoded = decode_flac_frame(&encoded, &info).expect("decode should succeed");
        assert_eq!(
            decoded, samples,
            "Frame encode-decode roundtrip must be lossless"
        );
    }

    #[test]
    fn test_flac_config_default() {
        let config = FlacEncoderConfig::default();
        assert_eq!(config.sample_rate, 44100);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bits_per_sample, 16);
        assert_eq!(config.block_size, 4096);
        assert_eq!(config.compression_level, 5);
    }

    #[test]
    fn test_encode_empty_block() {
        let config = FlacEncoderConfig::default();
        let info = FlacStreamInfo {
            min_block_size: 0,
            max_block_size: 0,
            sample_rate: 44100,
            channels: 1,
            bits_per_sample: 16,
            total_samples: 0,
        };

        let encoded = encode_flac_frame(&[], &config);
        assert!(
            !encoded.is_empty(),
            "Empty input should still produce a frame header"
        );
        let decoded = decode_flac_frame(&encoded, &info).expect("decode empty should succeed");
        assert!(
            decoded.is_empty(),
            "Decoded empty frame should have no samples"
        );
    }

    #[test]
    fn test_zigzag_roundtrip() {
        for v in [-1000i32, -1, 0, 1, 1000, i16::MIN as i32, i16::MAX as i32] {
            let u = zigzag_encode(v);
            let back = zigzag_decode(u);
            assert_eq!(back, v, "zigzag roundtrip failed for {v}");
        }
    }

    #[test]
    fn test_fixed_predict_order2_quadratic() {
        // Quadratic: s[i] = i^2 → second differences are constant
        let samples: Vec<i16> = (0..20).map(|i: i16| i * i).collect();
        let residuals = fixed_predict(&samples, 2);
        // After warmup of 2, all second-order residuals for a quadratic should be constant (= 2)
        let all_two = residuals.iter().all(|&r| r == 2);
        assert!(
            all_two,
            "Quadratic signal order-2 residuals should all be 2: {:?}",
            residuals
        );
    }

    #[test]
    fn test_encode_decode_large_block() {
        let config = FlacEncoderConfig {
            compression_level: 8,
            ..FlacEncoderConfig::default()
        };
        let info = FlacStreamInfo {
            min_block_size: 4096,
            max_block_size: 4096,
            sample_rate: 44100,
            channels: 1,
            bits_per_sample: 16,
            total_samples: 4096,
        };
        let samples: Vec<i16> = (0..4096)
            .map(|i| (1000.0 * (i as f64 * 0.05).sin()) as i16)
            .collect();
        let encoded = encode_flac_frame(&samples, &config);
        // Compressed should be smaller than raw (4096 * 2 = 8192 bytes)
        assert!(
            encoded.len() < 8192,
            "Compressed frame ({} bytes) should be smaller than raw (8192)",
            encoded.len()
        );
        let decoded = decode_flac_frame(&encoded, &info).expect("decode");
        assert_eq!(decoded, samples);
    }
}
