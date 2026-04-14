//! Partitioned convolution engine for low-latency HRTF rendering.
//!
//! Convolving a long HRTF impulse response (IR) directly in the time domain
//! requires `O(N·M)` operations per sample (N = IR length, M = block size).
//! For HRTFs of several thousand taps this becomes prohibitive.
//!
//! **Overlap-save** (also called overlap-discard) partitioned convolution splits
//! both the input signal and the IR into equal-length partitions and processes
//! them in the frequency domain, achieving `O(B log B)` per block where
//! `B` is the partition size.  This trades a small amount of latency (one
//! partition length) for dramatically reduced CPU cost.
//!
//! # Algorithm overview
//!
//! Given an impulse response `h` of length `L` and a block size `B`:
//!
//! 1. **Partition** the IR into `P = ceil(L/B)` zero-padded segments of length `2B`:
//!    `H_k = FFT(h[k*B .. (k+1)*B], 2B)`.
//! 2. **For each new input block** `x_n` of length `B`:
//!    a. Append to a **delay line** of length `(P+1)·B` samples.
//!    b. For each partition `k`, take the FFT of the corresponding delayed block,
//!       multiply by `H_k`, and accumulate into a **frequency-domain delay line**.
//!    c. IFFT the accumulated spectrum and output the second half of the result
//!       (overlap-save discard of the first `B` samples).
//!
//! # Implementation notes
//!
//! Since this crate has no FFT dependency, a pure-Rust DFT/IDFT implementation
//! is used here.  For production deployments (e.g. via `oximedia-audio`) the
//! partitioned engine interface is the same — only the FFT backend needs to be
//! swapped.  The DFT used here is `O(N²)` but is correct and self-contained.
//!
//! # Latency
//!
//! The algorithm introduces one block (`partition_size` samples) of latency.
//! This is the minimum achievable with overlap-save partitioned convolution
//! without lookahead.  For head-tracking compensation a latency of 256–512
//! samples at 48 kHz (5.3–10.7 ms) is typically acceptable.
//!
//! # References
//! - Gardner, W.G. (1995). "Efficient convolution without input-output delay."
//!   *JAES* 43(3), 127–136.
//! - Wefers, F. (2015). *Partitioned convolution algorithms for real-time
//!   auralization.* RWTH Aachen.

use crate::SpatialError;

// ─── DFT helpers (pure-Rust, no external crate) ───────────────────────────────

/// A complex number (f32 real and imaginary parts).
#[derive(Debug, Clone, Copy, Default)]
struct Complex {
    re: f32,
    im: f32,
}

impl Complex {
    fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }

    /// Multiply two complex numbers.
    fn mul(self, rhs: Self) -> Self {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }

    /// Add two complex numbers.
    fn add(self, rhs: Self) -> Self {
        Self { re: self.re + rhs.re, im: self.im + rhs.im }
    }
}

/// Compute the Discrete Fourier Transform (DFT) of a real-valued input.
///
/// Returns `n` complex coefficients.  This is an `O(n²)` naive DFT —
/// accurate and dependency-free but not optimised for large `n`.
fn dft(input: &[f32]) -> Vec<Complex> {
    use std::f32::consts::PI;
    let n = input.len();
    (0..n)
        .map(|k| {
            let mut sum = Complex::default();
            for (j, &x) in input.iter().enumerate() {
                let angle = -2.0 * PI * k as f32 * j as f32 / n as f32;
                sum = sum.add(Complex::new(x * angle.cos(), x * angle.sin()));
            }
            sum
        })
        .collect()
}

/// Compute the Inverse DFT and return the real part of each output sample,
/// scaled by `1/n`.
fn idft(input: &[Complex]) -> Vec<f32> {
    use std::f32::consts::PI;
    let n = input.len();
    let scale = 1.0 / n as f32;
    (0..n)
        .map(|j| {
            let mut sum = Complex::default();
            for (k, c) in input.iter().enumerate() {
                let angle = 2.0 * PI * k as f32 * j as f32 / n as f32;
                sum = sum.add(c.mul(Complex::new(angle.cos(), angle.sin())));
            }
            sum.re * scale
        })
        .collect()
}

// ─── IR partition ─────────────────────────────────────────────────────────────

/// Pre-computed frequency-domain partition of an impulse response.
#[derive(Debug, Clone)]
struct IrPartition {
    /// DFT of the zero-padded IR segment (length = `2 * partition_size`).
    freq: Vec<Complex>,
}

impl IrPartition {
    /// Compute the DFT of a zero-padded IR segment.
    fn from_time_domain(segment: &[f32], partition_size: usize) -> Self {
        let fft_size = 2 * partition_size;
        let mut padded = vec![0.0_f32; fft_size];
        let copy_len = segment.len().min(partition_size);
        padded[..copy_len].copy_from_slice(&segment[..copy_len]);
        Self { freq: dft(&padded) }
    }
}

// ─── Partitioned convolution engine ───────────────────────────────────────────

/// Uniform partitioned convolution engine (overlap-save method).
///
/// Processes audio in blocks of `partition_size` samples, convolving with an
/// arbitrary-length impulse response.  One block of latency is introduced.
///
/// # Example
///
/// ```rust
/// use oximedia_spatial::partitioned_convolution::PartitionedConvolver;
///
/// // Construct a simple 4-sample IR: [1, 0, 0, 0] (identity for block_size=4)
/// let ir = vec![1.0_f32, 0.0, 0.0, 0.0];
/// let mut conv = PartitionedConvolver::new(&ir, 4).unwrap();
/// let input = vec![1.0_f32, 2.0, 3.0, 4.0];
/// let output = conv.process_block(&input).unwrap();
/// assert_eq!(output.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct PartitionedConvolver {
    /// Block / partition size in samples.
    partition_size: usize,
    /// FFT size (always `2 * partition_size`).
    fft_size: usize,
    /// Pre-computed frequency-domain IR partitions.
    partitions: Vec<IrPartition>,
    /// Input delay line (length = `(num_partitions + 1) * partition_size`).
    input_delay: Vec<f32>,
    /// Write position in the delay line.
    write_pos: usize,
    /// Frequency-domain accumulator (length = `fft_size`).
    freq_accum: Vec<Complex>,
}

impl PartitionedConvolver {
    /// Create a new partitioned convolver.
    ///
    /// * `ir` — impulse response samples.
    /// * `partition_size` — block size in samples (must be ≥ 1).
    pub fn new(ir: &[f32], partition_size: usize) -> Result<Self, SpatialError> {
        if partition_size == 0 {
            return Err(SpatialError::InvalidConfig(
                "partition_size must be >= 1".to_string(),
            ));
        }
        if ir.is_empty() {
            return Err(SpatialError::InvalidConfig(
                "impulse response must not be empty".to_string(),
            ));
        }

        let fft_size = 2 * partition_size;
        let num_partitions = (ir.len() + partition_size - 1) / partition_size;

        let partitions: Vec<IrPartition> = (0..num_partitions)
            .map(|k| {
                let start = k * partition_size;
                let end = (start + partition_size).min(ir.len());
                IrPartition::from_time_domain(&ir[start..end], partition_size)
            })
            .collect();

        let delay_len = (num_partitions + 1) * partition_size;

        Ok(Self {
            partition_size,
            fft_size,
            partitions,
            input_delay: vec![0.0; delay_len],
            write_pos: 0,
            freq_accum: vec![Complex::default(); fft_size],
        })
    }

    /// Process a single block of `partition_size` input samples.
    ///
    /// Returns a block of `partition_size` output samples.
    /// The caller must ensure `input.len() == partition_size`.
    pub fn process_block(&mut self, input: &[f32]) -> Result<Vec<f32>, SpatialError> {
        if input.len() != self.partition_size {
            return Err(SpatialError::InvalidConfig(format!(
                "input block length {} does not match partition_size {}",
                input.len(),
                self.partition_size
            )));
        }

        let b = self.partition_size;
        let n = self.partitions.len();
        let delay_len = self.input_delay.len();

        // ── Write new input block into the delay line ─────────────────────
        for (i, &x) in input.iter().enumerate() {
            self.input_delay[(self.write_pos + i) % delay_len] = x;
        }

        // ── Accumulate frequency-domain products ──────────────────────────
        self.freq_accum = vec![Complex::default(); self.fft_size];

        for k in 0..n {
            // Extract the delayed block corresponding to partition k.
            // Partition 0 uses the current block (most recent), partition 1
            // uses the block before that, etc.
            let delay_offset = (k + 1) * b;

            let mut seg = vec![0.0_f32; self.fft_size];
            for i in 0..self.fft_size {
                // Look back (delay_offset + fft_size - 1 - i) samples from write_pos.
                let look_back = delay_offset + self.fft_size - 1 - i;
                let idx = (self.write_pos + b + delay_len - look_back % delay_len) % delay_len;
                seg[i] = self.input_delay[idx];
            }

            let x_freq = dft(&seg);
            for (j, (acc, x)) in self.freq_accum.iter_mut().zip(x_freq.iter()).enumerate() {
                *acc = acc.add(self.partitions[k].freq[j].mul(*x));
            }
        }

        // ── IFFT and extract the second half (overlap-save) ────────────────
        let time_out = idft(&self.freq_accum);
        let output: Vec<f32> = time_out[b..].to_vec();

        // ── Advance write position ─────────────────────────────────────────
        self.write_pos = (self.write_pos + b) % delay_len;

        Ok(output)
    }

    /// Update the impulse response without resetting the delay line.
    ///
    /// The new IR is partitioned immediately.  This allows smooth HRTF
    /// interpolation by gradually crossfading between the old and new
    /// partition sets (crossfading must be managed by the caller).
    pub fn update_ir(&mut self, ir: &[f32]) -> Result<(), SpatialError> {
        if ir.is_empty() {
            return Err(SpatialError::InvalidConfig(
                "impulse response must not be empty".to_string(),
            ));
        }
        let num_partitions = (ir.len() + self.partition_size - 1) / self.partition_size;
        self.partitions = (0..num_partitions)
            .map(|k| {
                let start = k * self.partition_size;
                let end = (start + self.partition_size).min(ir.len());
                IrPartition::from_time_domain(&ir[start..end], self.partition_size)
            })
            .collect();
        Ok(())
    }

    /// Return the partition size (block size) in samples.
    pub fn partition_size(&self) -> usize {
        self.partition_size
    }

    /// Return the number of IR partitions.
    pub fn num_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Reset the internal state (delay line and accumulator) without clearing the IR.
    pub fn reset(&mut self) {
        self.input_delay.fill(0.0);
        self.freq_accum.fill(Complex::default());
        self.write_pos = 0;
    }
}

// ─── Stereo binaural convolver ────────────────────────────────────────────────

/// Binaural convolver applying separate left and right HRTFs to a mono signal.
///
/// Wraps two [`PartitionedConvolver`] instances (one per ear) and provides a
/// unified API for HRTF-based binaural rendering.
///
/// # Example
///
/// ```rust
/// use oximedia_spatial::partitioned_convolution::BinauralConvolver;
///
/// let hrtf_l = vec![1.0_f32; 8]; // placeholder IR
/// let hrtf_r = vec![1.0_f32; 8]; // placeholder IR
/// let mut conv = BinauralConvolver::new(&hrtf_l, &hrtf_r, 4).unwrap();
/// let mono = vec![0.5_f32; 4];
/// let (left, right) = conv.process(&mono).unwrap();
/// assert_eq!(left.len(), 4);
/// assert_eq!(right.len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct BinauralConvolver {
    /// Left-ear HRTF convolver.
    left: PartitionedConvolver,
    /// Right-ear HRTF convolver.
    right: PartitionedConvolver,
}

impl BinauralConvolver {
    /// Create a binaural convolver.
    ///
    /// * `hrtf_l` — left-ear HRTF impulse response.
    /// * `hrtf_r` — right-ear HRTF impulse response.
    /// * `partition_size` — block size in samples.
    pub fn new(
        hrtf_l: &[f32],
        hrtf_r: &[f32],
        partition_size: usize,
    ) -> Result<Self, SpatialError> {
        Ok(Self {
            left: PartitionedConvolver::new(hrtf_l, partition_size)?,
            right: PartitionedConvolver::new(hrtf_r, partition_size)?,
        })
    }

    /// Process a mono block through both HRTFs and return `(left, right)`.
    pub fn process(&mut self, mono: &[f32]) -> Result<(Vec<f32>, Vec<f32>), SpatialError> {
        let l = self.left.process_block(mono)?;
        let r = self.right.process_block(mono)?;
        Ok((l, r))
    }

    /// Update both HRTFs simultaneously (for smooth direction changes).
    pub fn update_hrtfs(&mut self, hrtf_l: &[f32], hrtf_r: &[f32]) -> Result<(), SpatialError> {
        self.left.update_ir(hrtf_l)?;
        self.right.update_ir(hrtf_r)?;
        Ok(())
    }

    /// Return the partition (block) size.
    pub fn partition_size(&self) -> usize {
        self.left.partition_size()
    }

    /// Reset both convolvers.
    pub fn reset(&mut self) {
        self.left.reset();
        self.right.reset();
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_impulse(len: usize) -> Vec<f32> {
        let mut v = vec![0.0_f32; len];
        v[0] = 1.0;
        v
    }

    // ── DFT round-trip ──────────────────────────────────────────────────────

    #[test]
    fn test_dft_idft_roundtrip() {
        let signal = vec![1.0_f32, 2.0, 3.0, 4.0];
        let freq = dft(&signal);
        let recovered = idft(&freq);
        for (a, b) in signal.iter().zip(recovered.iter()) {
            assert!((a - b).abs() < 1e-4, "a={a}, b={b}");
        }
    }

    #[test]
    fn test_dft_of_constant_signal() {
        // DFT of [c, c, c, c] should have DC bin = 4c and others near zero.
        let signal = vec![1.0_f32; 4];
        let freq = dft(&signal);
        assert!((freq[0].re - 4.0).abs() < 1e-4, "DC={}", freq[0].re);
        assert!(freq[1].re.abs() < 1e-4, "bin1={}", freq[1].re);
    }

    // ── PartitionedConvolver construction ────────────────────────────────────

    #[test]
    fn test_convolver_rejects_zero_partition_size() {
        let ir = vec![1.0_f32; 8];
        assert!(PartitionedConvolver::new(&ir, 0).is_err());
    }

    #[test]
    fn test_convolver_rejects_empty_ir() {
        assert!(PartitionedConvolver::new(&[], 4).is_err());
    }

    #[test]
    fn test_convolver_num_partitions() {
        let ir = vec![1.0_f32; 10];
        let conv = PartitionedConvolver::new(&ir, 4).expect("ok");
        assert_eq!(conv.num_partitions(), 3); // ceil(10/4) = 3
    }

    // ── Identity convolution ─────────────────────────────────────────────────

    #[test]
    fn test_identity_ir_output_length() {
        let ir = make_impulse(4);
        let mut conv = PartitionedConvolver::new(&ir, 4).expect("ok");
        let input = vec![1.0_f32, 2.0, 3.0, 4.0];
        let output = conv.process_block(&input).expect("ok");
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_block_size_mismatch_returns_error() {
        let ir = make_impulse(4);
        let mut conv = PartitionedConvolver::new(&ir, 4).expect("ok");
        let bad_input = vec![1.0_f32; 3]; // Wrong size
        assert!(conv.process_block(&bad_input).is_err());
    }

    // ── IR update ───────────────────────────────────────────────────────────

    #[test]
    fn test_update_ir_succeeds() {
        let ir = make_impulse(4);
        let mut conv = PartitionedConvolver::new(&ir, 4).expect("ok");
        let new_ir = vec![0.5_f32; 8];
        assert!(conv.update_ir(&new_ir).is_ok());
        assert_eq!(conv.num_partitions(), 2);
    }

    #[test]
    fn test_update_ir_empty_returns_error() {
        let ir = make_impulse(4);
        let mut conv = PartitionedConvolver::new(&ir, 4).expect("ok");
        assert!(conv.update_ir(&[]).is_err());
    }

    // ── Reset ────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_clears_delay_line() {
        let ir = make_impulse(4);
        let mut conv = PartitionedConvolver::new(&ir, 4).expect("ok");
        // Feed some data.
        let _ = conv.process_block(&[1.0, 2.0, 3.0, 4.0]).expect("ok");
        conv.reset();
        assert_eq!(conv.write_pos, 0);
        assert!(conv.input_delay.iter().all(|&x| x == 0.0));
    }

    // ── BinauralConvolver ────────────────────────────────────────────────────

    #[test]
    fn test_binaural_convolver_output_lengths() {
        let hrtf = make_impulse(4);
        let mut bc = BinauralConvolver::new(&hrtf, &hrtf, 4).expect("ok");
        let mono = vec![1.0_f32; 4];
        let (l, r) = bc.process(&mono).expect("ok");
        assert_eq!(l.len(), 4);
        assert_eq!(r.len(), 4);
    }

    #[test]
    fn test_binaural_convolver_update_hrtfs() {
        let hrtf = make_impulse(4);
        let mut bc = BinauralConvolver::new(&hrtf, &hrtf, 4).expect("ok");
        let new_hrtf = vec![0.5_f32; 4];
        assert!(bc.update_hrtfs(&new_hrtf, &new_hrtf).is_ok());
    }

    #[test]
    fn test_binaural_convolver_partition_size() {
        let hrtf = make_impulse(8);
        let bc = BinauralConvolver::new(&hrtf, &hrtf, 4).expect("ok");
        assert_eq!(bc.partition_size(), 4);
    }
}
