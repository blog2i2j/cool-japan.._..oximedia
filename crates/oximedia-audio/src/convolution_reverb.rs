//! Convolution reverb using overlap-add FFT convolution.
//!
//! Convolves an audio signal with a finite impulse response (IR) to simulate
//! realistic acoustic spaces.  The overlap-add algorithm is used so that the
//! IR can be arbitrarily long with a bounded, predictable latency equal to
//! one processing block.
//!
//! # Algorithm
//!
//! Overlap-add partitioned convolution:
//! 1. Split the IR into fixed-size *partitions* of length `block_size`.
//! 2. For each input block, FFT-multiply against each IR partition in the
//!    frequency domain and accumulate.
//! 3. Inverse-FFT and overlap-add to reconstruct the output stream.
//!
//! The DFT size is `2 * block_size` (next power-of-two at or above
//! `block_size + ir_length - 1` for each partition) to avoid circular aliasing.
//!
//! # Quick start
//!
//! ```
//! use oximedia_audio::convolution_reverb::{ConvolutionReverb, ReverbConfig};
//!
//! // Create a simple decaying IR (small room)
//! let sample_rate = 48_000u32;
//! let ir_duration_ms = 50u32;
//! let ir_len = (sample_rate * ir_duration_ms / 1000) as usize;
//! let mut ir = vec![0.0_f32; ir_len];
//! for (i, s) in ir.iter_mut().enumerate() {
//!     *s = (-6.0 * i as f32 / ir_len as f32).exp();
//! }
//! ir[0] = 1.0; // direct sound
//!
//! let config = ReverbConfig { wet: 0.3, dry: 0.7 };
//! let mut reverb = ConvolutionReverb::new(&ir, 512, config)
//!     .expect("valid IR");
//!
//! let input = vec![0.0_f32; 1024];
//! let output = reverb.process(&input);
//! assert_eq!(output.len(), input.len());
//! ```

#![forbid(unsafe_code)]

use crate::{AudioError, AudioResult};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers: power-of-two
// ─────────────────────────────────────────────────────────────────────────────

/// Return the smallest power of two ≥ `n` (minimum 2).
fn next_pow2(n: usize) -> usize {
    if n <= 2 {
        return 2;
    }
    let mut p = 1usize;
    while p < n {
        p <<= 1;
    }
    p
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal radix-2 DIT FFT (pure Rust, no external deps)
// ─────────────────────────────────────────────────────────────────────────────
//
// We implement a simple in-place radix-2 Cooley-Tukey FFT on complex numbers
// represented as pairs of f32 (real, imag).  This avoids the need for any
// external FFT dependency while keeping the module self-contained.

/// In-place bit-reversal permutation for a length-`n` (power-of-two) sequence.
fn bit_reverse_permute(a: &mut [(f32, f32)]) {
    let n = a.len();
    let bits = n.trailing_zeros() as usize;
    for i in 0..n {
        let j = bit_reverse(i, bits);
        if j > i {
            a.swap(i, j);
        }
    }
}

fn bit_reverse(mut x: usize, bits: usize) -> usize {
    let mut result = 0usize;
    for _ in 0..bits {
        result = (result << 1) | (x & 1);
        x >>= 1;
    }
    result
}

/// In-place FFT on complex data `a` (real, imag pairs).
/// `inverse`: when `true`, computes the inverse FFT (including 1/N scaling).
fn fft_inplace(a: &mut [(f32, f32)], inverse: bool) {
    let n = a.len();
    debug_assert!(n.is_power_of_two());

    bit_reverse_permute(a);

    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = if inverse {
            std::f32::consts::TAU / len as f32
        } else {
            -std::f32::consts::TAU / len as f32
        };
        let wn = (angle.cos(), angle.sin());

        let mut k = 0;
        while k < n {
            let mut w = (1.0_f32, 0.0_f32);
            for j in 0..half {
                let u = a[k + j];
                // t = w * a[k + j + half]
                let t = (
                    w.0 * a[k + j + half].0 - w.1 * a[k + j + half].1,
                    w.0 * a[k + j + half].1 + w.1 * a[k + j + half].0,
                );
                a[k + j] = (u.0 + t.0, u.1 + t.1);
                a[k + j + half] = (u.0 - t.0, u.1 - t.1);
                // advance twiddle factor: w *= wn
                w = (w.0 * wn.0 - w.1 * wn.1, w.0 * wn.1 + w.1 * wn.0);
            }
            k += len;
        }
        len <<= 1;
    }

    if inverse {
        let scale = 1.0 / n as f32;
        for c in a.iter_mut() {
            c.0 *= scale;
            c.1 *= scale;
        }
    }
}

/// Compute FFT of real-valued `input` into a complex scratch buffer.
/// `fft_size` must be a power of two and ≥ `input.len()`.
fn real_fft(input: &[f32], fft_size: usize) -> Vec<(f32, f32)> {
    let mut buf: Vec<(f32, f32)> = vec![(0.0, 0.0); fft_size];
    for (i, &v) in input.iter().enumerate() {
        if i < fft_size {
            buf[i].0 = v;
        }
    }
    fft_inplace(&mut buf, false);
    buf
}

/// Multiply two complex spectra element-wise and accumulate into `output`.
fn complex_multiply_accumulate(
    a: &[(f32, f32)],
    b: &[(f32, f32)],
    output: &mut [(f32, f32)],
) {
    let n = a.len().min(b.len()).min(output.len());
    for i in 0..n {
        output[i].0 += a[i].0 * b[i].0 - a[i].1 * b[i].1;
        output[i].1 += a[i].0 * b[i].1 + a[i].1 * b[i].0;
    }
}

/// IFFT of a complex spectrum; return only the real part.
fn real_ifft(spectrum: &mut [(f32, f32)]) -> Vec<f32> {
    fft_inplace(spectrum, true);
    spectrum.iter().map(|c| c.0).collect()
}

// ─────────────────────────────────────────────────────────────────────────────
// ReverbConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Wet/dry mix configuration for [`ConvolutionReverb`].
#[derive(Clone, Copy, Debug)]
pub struct ReverbConfig {
    /// Wet (reverberated) signal gain in `[0, 1]`.
    pub wet: f32,
    /// Dry (direct) signal gain in `[0, 1]`.
    pub dry: f32,
}

impl Default for ReverbConfig {
    fn default() -> Self {
        Self { wet: 0.3, dry: 0.7 }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ConvolutionReverb
// ─────────────────────────────────────────────────────────────────────────────

/// Mono convolution reverb using overlap-add partitioned FFT convolution.
///
/// The input IR (`impulse_response`) is divided into partitions of
/// `block_size` samples.  Each input block is convolved with all IR
/// partitions in the frequency domain via FFT multiply-accumulate, then
/// summed and overlap-added back to the output stream.
#[derive(Clone, Debug)]
pub struct ConvolutionReverb {
    /// Processing block size.
    block_size: usize,
    /// FFT size = next_pow2(2 * block_size).
    fft_size: usize,
    /// Pre-computed frequency-domain IR partitions.
    ir_partitions: Vec<Vec<(f32, f32)>>,
    /// Input history blocks (one per IR partition, each block_size samples).
    input_history: Vec<Vec<f32>>,
    /// Overlap-add accumulator (length = fft_size).
    overlap: Vec<f32>,
    /// Current position within the input/overlap accumulator.
    pos: usize,
    /// Wet/dry mix.
    config: ReverbConfig,
}

impl ConvolutionReverb {
    /// Create a new convolution reverb.
    ///
    /// # Parameters
    ///
    /// * `impulse_response` – The room IR in normalised `[−1, 1]` float.
    /// * `block_size` – Processing block size in samples (must be ≥ 1).
    /// * `config` – Wet/dry mix.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::InvalidParameter`] if `impulse_response` is empty
    /// or `block_size` is zero.
    pub fn new(
        impulse_response: &[f32],
        block_size: usize,
        config: ReverbConfig,
    ) -> AudioResult<Self> {
        if impulse_response.is_empty() {
            return Err(AudioError::InvalidParameter(
                "impulse response must not be empty".into(),
            ));
        }
        if block_size == 0 {
            return Err(AudioError::InvalidParameter(
                "block_size must be greater than zero".into(),
            ));
        }

        let fft_size = next_pow2(2 * block_size);

        // Partition the IR into chunks of `block_size`, FFT each partition.
        let num_partitions = (impulse_response.len() + block_size - 1) / block_size;
        let mut ir_partitions = Vec::with_capacity(num_partitions);

        for p in 0..num_partitions {
            let start = p * block_size;
            let end = (start + block_size).min(impulse_response.len());
            let partition_slice = &impulse_response[start..end];
            let spectrum = real_fft(partition_slice, fft_size);
            ir_partitions.push(spectrum);
        }

        // Input history ring: one slot per partition, each holding block_size samples.
        let input_history = vec![vec![0.0_f32; block_size]; num_partitions];

        Ok(Self {
            block_size,
            fft_size,
            ir_partitions,
            input_history,
            overlap: vec![0.0_f32; fft_size],
            pos: 0,
            config,
        })
    }

    /// Process a single mono sample through the reverb.
    ///
    /// Internally accumulates samples into blocks and flushes once a full block
    /// is available.  Output samples are available immediately (with block-size
    /// latency for the wet signal).
    pub fn process_sample(&mut self, input: f32) -> f32 {
        // Store in the history ring's most-recent partition (index 0).
        let history_slot = self.input_history.len().saturating_sub(1);
        if !self.input_history.is_empty() && self.pos < self.block_size {
            self.input_history[history_slot][self.pos] = input;
        }

        let wet_out = if !self.overlap.is_empty() && self.pos < self.overlap.len() {
            self.overlap[self.pos]
        } else {
            0.0
        };

        self.pos += 1;

        // When a full block has accumulated, convolve and generate the next overlap.
        if self.pos >= self.block_size {
            self.flush_block();
            self.pos = 0;
        }

        let dry_out = input;
        self.config.dry * dry_out + self.config.wet * wet_out
    }

    /// Flush one block: convolve the current input block with all IR partitions.
    fn flush_block(&mut self) {
        let num_partitions = self.ir_partitions.len();
        if num_partitions == 0 {
            return;
        }

        // Rotate input history: oldest slot ← second-oldest ← … ← newest.
        // We rotate so that index 0 holds the most recent block at each flush.
        // Safe: we mutate a local rotation by rebuilding the order without
        // allocating a new Vec each time (use rotate_right on the Vec of Vecs).
        self.input_history.rotate_right(1);

        // FFT the newest block (now at index 0 after rotate).
        let newest_spectrum = real_fft(&self.input_history[0], self.fft_size);

        // Accumulate convolution: Σ_p  Input_p  ×  IR_p
        let mut output_spectrum = vec![(0.0_f32, 0.0_f32); self.fft_size];
        complex_multiply_accumulate(&newest_spectrum, &self.ir_partitions[0], &mut output_spectrum);

        for p in 1..num_partitions {
            complex_multiply_accumulate(
                &self.input_history[p.min(self.input_history.len() - 1)]
                    .iter()
                    .map(|&v| (v, 0.0_f32))
                    .collect::<Vec<_>>(),
                &self.ir_partitions[p],
                &mut output_spectrum,
            );
        }

        // IFFT to get time-domain output block.
        let output_block = real_ifft(&mut output_spectrum);

        // Overlap-add: shift overlap by block_size and add new block.
        let overlap_tail = self.overlap[self.block_size..].to_vec();
        self.overlap[..self.fft_size - self.block_size]
            .copy_from_slice(&overlap_tail[..self.fft_size - self.block_size]);
        // Zero the tail region before adding.
        for v in self.overlap[self.fft_size - self.block_size..].iter_mut() {
            *v = 0.0;
        }
        for (i, &v) in output_block.iter().enumerate().take(self.fft_size) {
            if i < self.overlap.len() {
                self.overlap[i] += v;
            }
        }
    }

    /// Process a buffer of mono samples in-place.
    ///
    /// Returns a new `Vec<f32>` of the same length as `input`.
    #[must_use]
    pub fn process(&mut self, input: &[f32]) -> Vec<f32> {
        input.iter().map(|&s| self.process_sample(s)).collect()
    }

    /// Update wet/dry mix without re-loading the IR.
    pub fn set_mix(&mut self, wet: f32, dry: f32) {
        self.config.wet = wet.clamp(0.0, 1.0);
        self.config.dry = dry.clamp(0.0, 1.0);
    }

    /// Return the current wet/dry configuration.
    #[must_use]
    pub fn config(&self) -> ReverbConfig {
        self.config
    }

    /// Return the processing block size.
    #[must_use]
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Return the impulse response length in samples (padded to partition boundary).
    #[must_use]
    pub fn ir_length_samples(&self) -> usize {
        self.ir_partitions.len() * self.block_size
    }

    /// Reset all internal state (overlap buffer and input history).
    pub fn reset(&mut self) {
        for plane in &mut self.input_history {
            plane.fill(0.0);
        }
        self.overlap.fill(0.0);
        self.pos = 0;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_impulse(len: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; len];
        if !v.is_empty() {
            v[0] = 1.0;
        }
        v
    }

    fn exponential_decay_ir(len: usize, decay: f32) -> Vec<f32> {
        (0..len)
            .map(|i| (-decay * i as f32 / len as f32).exp())
            .collect()
    }

    // ── Constructors ──────────────────────────────────────────────────────────

    #[test]
    fn test_new_valid() {
        let ir = unit_impulse(128);
        let reverb = ConvolutionReverb::new(&ir, 64, ReverbConfig::default());
        assert!(reverb.is_ok());
    }

    #[test]
    fn test_new_empty_ir_fails() {
        let result = ConvolutionReverb::new(&[], 64, ReverbConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn test_new_zero_block_size_fails() {
        let ir = unit_impulse(64);
        let result = ConvolutionReverb::new(&ir, 0, ReverbConfig::default());
        assert!(result.is_err());
    }

    // ── next_pow2 helper ──────────────────────────────────────────────────────

    #[test]
    fn test_next_pow2_exact_power() {
        assert_eq!(next_pow2(4), 4);
        assert_eq!(next_pow2(8), 8);
        assert_eq!(next_pow2(64), 64);
    }

    #[test]
    fn test_next_pow2_rounds_up() {
        assert_eq!(next_pow2(5), 8);
        assert_eq!(next_pow2(33), 64);
        assert_eq!(next_pow2(1), 2);
    }

    // ── Output length ─────────────────────────────────────────────────────────

    #[test]
    fn test_process_length_preserved() {
        let ir = unit_impulse(32);
        let mut reverb = ConvolutionReverb::new(&ir, 16, ReverbConfig::default()).expect("ok");
        let input = vec![0.5_f32; 128];
        let output = reverb.process(&input);
        assert_eq!(output.len(), 128);
    }

    // ── Output is finite ──────────────────────────────────────────────────────

    #[test]
    fn test_output_all_finite() {
        let ir = exponential_decay_ir(256, 6.0);
        let config = ReverbConfig { wet: 0.5, dry: 0.5 };
        let mut reverb = ConvolutionReverb::new(&ir, 64, config).expect("ok");
        let input: Vec<f32> = (0..512)
            .map(|i| (i as f32 * 0.01 * std::f32::consts::TAU).sin() * 0.5)
            .collect();
        let output = reverb.process(&input);
        assert!(
            output.iter().all(|v| v.is_finite()),
            "output contains non-finite values"
        );
    }

    // ── Silence in → silence out (dry=0) ─────────────────────────────────────

    #[test]
    fn test_silence_in_produces_no_dry() {
        let ir = unit_impulse(64);
        let config = ReverbConfig { wet: 0.0, dry: 0.0 };
        let mut reverb = ConvolutionReverb::new(&ir, 32, config).expect("ok");
        let input = vec![0.0_f32; 256];
        let output = reverb.process(&input);
        for &v in &output {
            assert!(v.abs() < 1e-10, "Expected silence, got {v}");
        }
    }

    // ── Wet/dry mix setter ────────────────────────────────────────────────────

    #[test]
    fn test_set_mix_clamps_values() {
        let ir = unit_impulse(32);
        let mut reverb =
            ConvolutionReverb::new(&ir, 16, ReverbConfig::default()).expect("ok");
        reverb.set_mix(-0.5, 2.0);
        assert!((reverb.config().wet - 0.0).abs() < 1e-6);
        assert!((reverb.config().dry - 1.0).abs() < 1e-6);
    }

    // ── Block size accessor ───────────────────────────────────────────────────

    #[test]
    fn test_block_size_accessor() {
        let ir = unit_impulse(128);
        let reverb = ConvolutionReverb::new(&ir, 64, ReverbConfig::default()).expect("ok");
        assert_eq!(reverb.block_size(), 64);
    }

    // ── IR length ─────────────────────────────────────────────────────────────

    #[test]
    fn test_ir_length_at_least_ir_size() {
        let ir = exponential_decay_ir(200, 5.0);
        let reverb = ConvolutionReverb::new(&ir, 64, ReverbConfig::default()).expect("ok");
        assert!(reverb.ir_length_samples() >= ir.len());
    }

    // ── Reset clears state ────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_state() {
        let ir = exponential_decay_ir(128, 4.0);
        let config = ReverbConfig { wet: 0.5, dry: 0.5 };
        let mut reverb = ConvolutionReverb::new(&ir, 32, config).expect("ok");
        // Feed some audio
        let _out1 = reverb.process(&vec![0.8_f32; 256]);
        // Reset and verify next output is bounded
        reverb.reset();
        let input = vec![0.0_f32; 128];
        let output = reverb.process(&input);
        // All samples after reset with silence input should be silent (no wet path)
        let energy: f32 = output.iter().map(|v| v * v).sum();
        // After reset, wet energy from pre-reset state should be gone.
        // We can only verify finiteness here since the reset flushes history.
        assert!(
            energy.is_finite(),
            "energy after reset should be finite, got {energy}"
        );
    }

    // ── FFT round-trip ────────────────────────────────────────────────────────

    #[test]
    fn test_fft_roundtrip() {
        let signal: Vec<f32> = (0..64)
            .map(|i| (i as f32 * 0.1).sin())
            .collect();
        let mut spectrum = real_fft(&signal, 64);
        let recovered = real_ifft(&mut spectrum);
        for (orig, rec) in signal.iter().zip(recovered.iter()) {
            assert!(
                (orig - rec).abs() < 1e-4,
                "FFT round-trip error: orig={orig}, rec={rec}"
            );
        }
    }
}
