//! SIMD-optimized summing for mix bus computation.
//!
//! Provides vectorized sum-with-gain operations used by the matrix router.
//! On targets that do not support SIMD, falls back to scalar accumulation.
//!
//! All functions in this module are pure Rust (no intrinsics) but are
//! structured to auto-vectorize well with LLVM.

// ---------------------------------------------------------------------------
// Runtime CPU feature detection
// ---------------------------------------------------------------------------

/// Detected SIMD capabilities of the current CPU.
#[derive(Debug, Clone, Copy, Default)]
pub struct CpuSimdFeatures {
    /// AVX2 (256-bit float SIMD) is available.
    pub avx2: bool,
    /// SSE4.2 (128-bit float SIMD) is available.
    pub sse42: bool,
}

impl CpuSimdFeatures {
    /// Detect the SIMD capabilities at runtime.
    ///
    /// On non-x86_64 targets this always returns `false` for both flags.
    pub fn detect() -> Self {
        #[cfg(target_arch = "x86_64")]
        {
            Self {
                avx2: std::is_x86_feature_detected!("avx2"),
                sse42: std::is_x86_feature_detected!("sse4.2"),
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            Self {
                avx2: false,
                sse42: false,
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SIMD mix-bus: AVX2 (x86_64 only)
// ---------------------------------------------------------------------------

/// Mix-bus summation using AVX2 (256-bit FMA, 8 f32 lanes per iteration).
///
/// Computes `output[s] += input_buffers[i][s] * gains[i]` for all inputs `i`
/// and samples `s`.  Falls back to the scalar path at the tail.
///
/// # Safety
/// Must only be called when the CPU supports AVX2 (guarded by `target_feature`).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mix_bus_avx2_inner(input_buffers: &[&[f32]], gains: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let num_inputs = input_buffers.len().min(gains.len());
    let num_samples = output.len();

    // Zero the output buffer.
    for s in output.iter_mut() {
        *s = 0.0;
    }

    for i in 0..num_inputs {
        let gain = gains[i];
        let buf = input_buffers[i];
        let len = buf.len().min(num_samples);

        let gain_vec = _mm256_set1_ps(gain);
        let chunks = len / 8;

        // 8-wide FMA loop.
        for c in 0..chunks {
            let base = c * 8;
            // Safety: we checked `len = buf.len().min(num_samples)` and
            // `base + 8 <= chunks * 8 <= len`.
            let src = _mm256_loadu_ps(buf.as_ptr().add(base));
            let dst = _mm256_loadu_ps(output.as_ptr().add(base));
            // dst += src * gain  (FMA: a*b+c with c=dst)
            let result = _mm256_fmadd_ps(src, gain_vec, dst);
            _mm256_storeu_ps(output.as_mut_ptr().add(base), result);
        }

        // Scalar tail.
        let tail_start = chunks * 8;
        for s in tail_start..len {
            output[s] += buf[s] * gain;
        }
    }
}

/// AVX2-accelerated mix bus — public entry point.
///
/// Dispatches to the AVX2 inner function when available, otherwise falls
/// back to the pure scalar implementation.
pub fn mix_bus_avx2(input_buffers: &[&[f32]], gains: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
        // Safety: we just checked that AVX2 and FMA are supported.
        unsafe { mix_bus_avx2_inner(input_buffers, gains, output) };
        return;
    }
    // Fallback.
    mix_bus_sum(input_buffers, gains, output);
}

// ---------------------------------------------------------------------------
// SIMD mix-bus: SSE4.2 (x86_64 only)
// ---------------------------------------------------------------------------

/// Mix-bus summation using SSE4.2 (128-bit, 4 f32 lanes per iteration).
///
/// # Safety
/// Must only be called when the CPU supports SSE4.2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn mix_bus_sse42_inner(input_buffers: &[&[f32]], gains: &[f32], output: &mut [f32]) {
    use std::arch::x86_64::*;

    let num_inputs = input_buffers.len().min(gains.len());
    let num_samples = output.len();

    // Zero the output buffer.
    for s in output.iter_mut() {
        *s = 0.0;
    }

    for i in 0..num_inputs {
        let gain = gains[i];
        let buf = input_buffers[i];
        let len = buf.len().min(num_samples);

        let gain_vec = _mm_set1_ps(gain);
        let chunks = len / 4;

        // 4-wide multiply-add loop.
        for c in 0..chunks {
            let base = c * 4;
            let src = _mm_loadu_ps(buf.as_ptr().add(base));
            let dst = _mm_loadu_ps(output.as_ptr().add(base));
            let scaled = _mm_mul_ps(src, gain_vec);
            let result = _mm_add_ps(dst, scaled);
            _mm_storeu_ps(output.as_mut_ptr().add(base), result);
        }

        // Scalar tail.
        let tail_start = chunks * 4;
        for s in tail_start..len {
            output[s] += buf[s] * gain;
        }
    }
}

/// SSE4.2-accelerated mix bus — public entry point.
pub fn mix_bus_sse42(input_buffers: &[&[f32]], gains: &[f32], output: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if std::is_x86_feature_detected!("sse4.2") {
        // Safety: we just checked that SSE4.2 is supported.
        unsafe { mix_bus_sse42_inner(input_buffers, gains, output) };
        return;
    }
    // Fallback.
    mix_bus_sum(input_buffers, gains, output);
}

/// Runtime-dispatched mix bus: picks the fastest available path.
///
/// Order: AVX2+FMA > SSE4.2 > scalar.
pub fn mix_bus_dispatch(input_buffers: &[&[f32]], gains: &[f32], output: &mut [f32]) {
    let feat = CpuSimdFeatures::detect();
    if feat.avx2 {
        mix_bus_avx2(input_buffers, gains, output);
    } else if feat.sse42 {
        mix_bus_sse42(input_buffers, gains, output);
    } else {
        mix_bus_sum(input_buffers, gains, output);
    }
}

// ---------------------------------------------------------------------------
// Dense scalar summing helpers (original API)
// ---------------------------------------------------------------------------

/// Sums `samples[i] * gains[i]` for paired slices.
///
/// Returns the accumulated sum. The slices must be the same length;
/// if they differ, only the shorter length is used.
pub fn sum_with_gains(samples: &[f32], gains: &[f32]) -> f32 {
    let len = samples.len().min(gains.len());
    // Process in chunks of 8 for autovectorization.
    let chunks = len / 8;
    let remainder = len % 8;

    let mut acc = [0.0_f32; 8];

    for chunk in 0..chunks {
        let base = chunk * 8;
        for j in 0..8 {
            acc[j] += samples[base + j] * gains[base + j];
        }
    }

    let mut total: f32 = acc.iter().sum();

    // Handle remainder
    let tail_base = chunks * 8;
    for i in 0..remainder {
        total += samples[tail_base + i] * gains[tail_base + i];
    }

    total
}

/// Sums `samples[indices[i]] * gains[i]` for sparse routing lookups.
///
/// This is the common pattern in crosspoint matrices where only some
/// inputs contribute to an output.
pub fn sparse_sum_with_gains(samples: &[f32], indices: &[usize], gains: &[f32]) -> f32 {
    let len = indices.len().min(gains.len());
    let mut sum = 0.0_f32;
    for i in 0..len {
        let idx = indices[i];
        if idx < samples.len() {
            sum += samples[idx] * gains[i];
        }
    }
    sum
}

/// Converts dB values to linear gains in-place.
///
/// `10^(db / 20)` for each element.
pub fn db_to_linear_batch(db_values: &[f32], out: &mut [f32]) {
    let len = db_values.len().min(out.len());
    for i in 0..len {
        out[i] = 10.0_f32.powf(db_values[i] / 20.0);
    }
}

/// Sums multiple input buffers with per-input gain, writing the result
/// into an output buffer (mix bus).
///
/// `output[s] = sum(input_buffers[i][s] * gains[i])` for each sample `s`.
pub fn mix_bus_sum(input_buffers: &[&[f32]], gains: &[f32], output: &mut [f32]) {
    let num_inputs = input_buffers.len().min(gains.len());

    // Zero the output
    for s in output.iter_mut() {
        *s = 0.0;
    }

    for i in 0..num_inputs {
        let gain = gains[i];
        let buf = input_buffers[i];
        let len = buf.len().min(output.len());
        for j in 0..len {
            output[j] += buf[j] * gain;
        }
    }
}

/// Scales a buffer in-place by a linear gain factor.
pub fn scale_buffer(buffer: &mut [f32], gain: f32) {
    for s in buffer.iter_mut() {
        *s *= gain;
    }
}

/// Adds `src` into `dst` (element-wise accumulate).
pub fn accumulate(dst: &mut [f32], src: &[f32]) {
    let len = dst.len().min(src.len());
    for i in 0..len {
        dst[i] += src[i];
    }
}

/// Peak absolute value of a buffer.
pub fn peak_abs(buffer: &[f32]) -> f32 {
    let mut peak = 0.0_f32;
    for &s in buffer {
        let abs = s.abs();
        if abs > peak {
            peak = abs;
        }
    }
    peak
}

/// RMS of a buffer.
pub fn rms(buffer: &[f32]) -> f32 {
    if buffer.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = buffer.iter().map(|&s| (s as f64) * (s as f64)).sum();
    (sum_sq / buffer.len() as f64).sqrt() as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sum_with_gains_basic() {
        let samples = [1.0_f32, 2.0, 3.0, 4.0];
        let gains = [1.0_f32, 0.5, 0.25, 0.125];
        let result = sum_with_gains(&samples, &gains);
        // 1*1 + 2*0.5 + 3*0.25 + 4*0.125 = 1 + 1 + 0.75 + 0.5 = 3.25
        assert!((result - 3.25).abs() < 1e-5);
    }

    #[test]
    fn test_sum_with_gains_empty() {
        assert!(sum_with_gains(&[], &[]).abs() < 1e-10);
    }

    #[test]
    fn test_sum_with_gains_mismatched_lengths() {
        let samples = [1.0_f32, 2.0, 3.0];
        let gains = [1.0_f32, 0.5];
        let result = sum_with_gains(&samples, &gains);
        // Only first 2 used: 1*1 + 2*0.5 = 2.0
        assert!((result - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_sum_with_gains_large() {
        // Test with 100 elements to exercise the chunked loop
        let samples: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let gains: Vec<f32> = vec![1.0; 100];
        let result = sum_with_gains(&samples, &gains);
        let expected: f32 = samples.iter().sum();
        assert!((result - expected).abs() < 1e-3);
    }

    #[test]
    fn test_sparse_sum_basic() {
        let samples = [0.0_f32, 0.5, 0.0, 0.8];
        let indices = [1, 3];
        let gains = [1.0_f32, 0.5];
        let result = sparse_sum_with_gains(&samples, &indices, &gains);
        // 0.5*1.0 + 0.8*0.5 = 0.9
        assert!((result - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_sparse_sum_out_of_bounds_index() {
        let samples = [1.0_f32, 2.0];
        let indices = [0, 99]; // index 99 out of bounds
        let gains = [1.0, 1.0];
        let result = sparse_sum_with_gains(&samples, &indices, &gains);
        // Only index 0 valid: 1.0
        assert!((result - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_db_to_linear_batch() {
        let db = [0.0_f32, -20.0, -6.0];
        let mut out = [0.0_f32; 3];
        db_to_linear_batch(&db, &mut out);
        assert!((out[0] - 1.0).abs() < 1e-5);
        assert!((out[1] - 0.1).abs() < 0.01);
        assert!((out[2] - 0.5012).abs() < 0.01);
    }

    #[test]
    fn test_mix_bus_sum() {
        let buf_a = [1.0_f32, 0.5, 0.25];
        let buf_b = [0.5_f32, 0.25, 0.125];
        let gains = [1.0_f32, 0.5];
        let mut output = [0.0_f32; 3];
        mix_bus_sum(&[&buf_a, &buf_b], &gains, &mut output);
        // output[0] = 1.0*1.0 + 0.5*0.5 = 1.25
        assert!((output[0] - 1.25).abs() < 1e-5);
        // output[1] = 0.5*1.0 + 0.25*0.5 = 0.625
        assert!((output[1] - 0.625).abs() < 1e-5);
    }

    #[test]
    fn test_scale_buffer() {
        let mut buf = [1.0_f32, 2.0, 3.0];
        scale_buffer(&mut buf, 0.5);
        assert!((buf[0] - 0.5).abs() < 1e-5);
        assert!((buf[1] - 1.0).abs() < 1e-5);
        assert!((buf[2] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_accumulate() {
        let mut dst = [1.0_f32, 2.0, 3.0];
        let src = [0.5_f32, 0.5, 0.5];
        accumulate(&mut dst, &src);
        assert!((dst[0] - 1.5).abs() < 1e-5);
        assert!((dst[1] - 2.5).abs() < 1e-5);
        assert!((dst[2] - 3.5).abs() < 1e-5);
    }

    #[test]
    fn test_peak_abs() {
        let buf = [0.5_f32, -0.8, 0.3, -0.1];
        assert!((peak_abs(&buf) - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_peak_abs_empty() {
        assert!(peak_abs(&[]).abs() < 1e-10);
    }

    #[test]
    fn test_rms_constant() {
        let buf = [0.5_f32; 100];
        assert!((rms(&buf) - 0.5).abs() < 1e-4);
    }

    // -----------------------------------------------------------------------
    // SIMD mix bus tests (8 tests — correctness vs scalar baseline)
    // -----------------------------------------------------------------------

    fn reference_mix(input_buffers: &[&[f32]], gains: &[f32], output: &mut [f32]) {
        mix_bus_sum(input_buffers, gains, output);
    }

    fn make_test_buffers() -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let len = 64;
        let buf_a: Vec<f32> = (0..len).map(|i| i as f32 * 0.01).collect();
        let buf_b: Vec<f32> = (0..len).map(|i| (len - i) as f32 * 0.02).collect();
        let gains = vec![0.5_f32, 0.75];
        let out = vec![0.0_f32; len];
        (buf_a, buf_b, gains, out)
    }

    #[test]
    fn test_mix_bus_avx2_matches_scalar() {
        let (buf_a, buf_b, gains, mut out_simd) = make_test_buffers();
        let mut out_ref = out_simd.clone();
        let inputs: Vec<&[f32]> = vec![&buf_a, &buf_b];

        mix_bus_avx2(&inputs, &gains, &mut out_simd);
        reference_mix(&inputs, &gains, &mut out_ref);

        for (s, r) in out_simd.iter().zip(out_ref.iter()) {
            assert!(
                (s - r).abs() < 1e-5,
                "AVX2 mismatch: {s:.6} vs ref {r:.6}"
            );
        }
    }

    #[test]
    fn test_mix_bus_sse42_matches_scalar() {
        let (buf_a, buf_b, gains, mut out_simd) = make_test_buffers();
        let mut out_ref = out_simd.clone();
        let inputs: Vec<&[f32]> = vec![&buf_a, &buf_b];

        mix_bus_sse42(&inputs, &gains, &mut out_simd);
        reference_mix(&inputs, &gains, &mut out_ref);

        for (s, r) in out_simd.iter().zip(out_ref.iter()) {
            assert!(
                (s - r).abs() < 1e-5,
                "SSE4.2 mismatch: {s:.6} vs ref {r:.6}"
            );
        }
    }

    #[test]
    fn test_mix_bus_dispatch_matches_scalar() {
        let (buf_a, buf_b, gains, mut out_disp) = make_test_buffers();
        let mut out_ref = out_disp.clone();
        let inputs: Vec<&[f32]> = vec![&buf_a, &buf_b];

        mix_bus_dispatch(&inputs, &gains, &mut out_disp);
        reference_mix(&inputs, &gains, &mut out_ref);

        for (d, r) in out_disp.iter().zip(out_ref.iter()) {
            assert!(
                (d - r).abs() < 1e-5,
                "dispatch mismatch: {d:.6} vs ref {r:.6}"
            );
        }
    }

    #[test]
    fn test_mix_bus_avx2_single_input() {
        let buf = vec![1.0_f32; 16];
        let gains = vec![0.5_f32];
        let mut out = vec![0.0_f32; 16];
        mix_bus_avx2(&[&buf], &gains, &mut out);
        for &v in &out {
            assert!((v - 0.5).abs() < 1e-6, "expected 0.5, got {v}");
        }
    }

    #[test]
    fn test_mix_bus_sse42_single_input() {
        let buf = vec![1.0_f32; 8];
        let gains = vec![0.25_f32];
        let mut out = vec![0.0_f32; 8];
        mix_bus_sse42(&[&buf], &gains, &mut out);
        for &v in &out {
            assert!((v - 0.25).abs() < 1e-6, "expected 0.25, got {v}");
        }
    }

    #[test]
    fn test_mix_bus_avx2_zero_gain() {
        let buf = vec![1.0_f32; 32];
        let gains = vec![0.0_f32];
        let mut out = vec![9.9_f32; 32]; // non-zero initial
        mix_bus_avx2(&[&buf], &gains, &mut out);
        for &v in &out {
            assert!(v.abs() < 1e-6, "expected 0, got {v}");
        }
    }

    #[test]
    fn test_mix_bus_avx2_tail_remainder() {
        // 13 samples = 1 AVX2 chunk (8) + 5 tail samples
        let buf = vec![1.0_f32; 13];
        let gains = vec![2.0_f32];
        let mut out = vec![0.0_f32; 13];
        mix_bus_avx2(&[&buf], &gains, &mut out);
        for &v in &out {
            assert!((v - 2.0).abs() < 1e-6, "expected 2.0, got {v}");
        }
    }

    #[test]
    fn test_mix_bus_sse42_tail_remainder() {
        // 7 samples = 1 SSE chunk (4) + 3 tail samples
        let buf = vec![3.0_f32; 7];
        let gains = vec![0.5_f32];
        let mut out = vec![0.0_f32; 7];
        mix_bus_sse42(&[&buf], &gains, &mut out);
        for &v in &out {
            assert!((v - 1.5).abs() < 1e-6, "expected 1.5, got {v}");
        }
    }

    #[test]
    fn test_rms_empty() {
        assert!(rms(&[]).abs() < 1e-10);
    }

    #[test]
    fn test_rms_sine() {
        // RMS of a sine wave is amplitude / sqrt(2)
        let buf: Vec<f32> = (0..48000)
            .map(|i| {
                let t = i as f64 / 48000.0;
                (2.0 * std::f64::consts::PI * 1000.0 * t).sin() as f32
            })
            .collect();
        let r = rms(&buf);
        let expected = 1.0_f32 / 2.0_f32.sqrt();
        assert!((r - expected).abs() < 0.01);
    }
}
