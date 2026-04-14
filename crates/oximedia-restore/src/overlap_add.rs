//! Overlap-add (OLA) block processor for real-time audio preview.
//!
//! The overlap-add method divides an input signal into overlapping blocks,
//! applies a processing function to each block, multiplies by a Hann window,
//! and sums the windowed blocks back together with the appropriate time offset.
//! This eliminates blocking artefacts at block boundaries and enables smooth
//! real-time processing.
//!
//! # Example
//!
//! ```
//! use oximedia_restore::overlap_add::{OverlapAddConfig, OverlapAddProcessor};
//!
//! let config = OverlapAddConfig { block_size: 1024, overlap_pct: 0.5 };
//! let mut processor = OverlapAddProcessor::new(config);
//!
//! let block: Vec<f32> = (0..1024).map(|i| (i as f32 * 0.001).sin()).collect();
//! let output = processor.process_block(&block).expect("block size matches config");
//! assert_eq!(output.len(), 1024);
//! ```

use crate::error::{RestoreError, RestoreResult};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for the overlap-add processor.
///
/// `overlap_pct` must be in `[0.0, 0.75]`.  Values above 0.75 are clamped.
/// Common choices are 0.25, 0.50, and 0.75.
#[derive(Debug, Clone, Copy)]
pub struct OverlapAddConfig {
    /// The analysis/synthesis block size in samples.
    ///
    /// Must be at least 2.  Larger blocks reduce the overhead of the Hann
    /// window at the cost of increased latency.
    pub block_size: usize,

    /// Fraction of `block_size` that consecutive blocks overlap.
    ///
    /// Clamped to `[0.0, 0.75]`.  Use `0.5` for a standard 50 % overlap
    /// (requires two frames to fill each output sample).
    pub overlap_pct: f32,
}

impl OverlapAddConfig {
    /// Return the hop size (number of new samples between successive blocks).
    ///
    /// `hop = block_size * (1 - overlap_pct)`.  Always at least 1.
    #[must_use]
    pub fn hop_size(&self) -> usize {
        let overlap = self.overlap_pct.clamp(0.0, 0.75);
        let hop = self.block_size as f32 * (1.0 - overlap);
        (hop.round() as usize).max(1)
    }
}

impl Default for OverlapAddConfig {
    fn default() -> Self {
        Self {
            block_size: 1024,
            overlap_pct: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Hann window
// ---------------------------------------------------------------------------

/// Compute a Hann analysis/synthesis window of length `n`.
///
/// `w[i] = 0.5 * (1 - cos(2π·i / n))`
#[must_use]
fn hann_window(n: usize) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    (0..n)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / n as f64;
            (0.5 * (1.0 - angle.cos())) as f32
        })
        .collect()
}

// ---------------------------------------------------------------------------
// OverlapAddProcessor
// ---------------------------------------------------------------------------

/// Stateful overlap-add processor.
///
/// Call [`process_block`][Self::process_block] with successive equal-sized
/// input blocks to obtain windowed, overlap-added output blocks.  The
/// processor stores a tail buffer so that output from one block can be
/// correctly summed with the overlap from the previous block.
///
/// # Block size contract
///
/// Every call to [`process_block`][Self::process_block] **must** supply
/// exactly `config.block_size` samples.  Calls with the wrong length return
/// an error.
#[derive(Debug)]
pub struct OverlapAddProcessor {
    config: OverlapAddConfig,
    /// Pre-computed Hann window coefficients.
    window: Vec<f32>,
    /// Overlap tail accumulated from the previous block (length = overlap samples).
    overlap_tail: Vec<f32>,
    /// The hop size (derived from config, cached for convenience).
    hop: usize,
    /// Number of samples that overlap between consecutive blocks.
    overlap_len: usize,
}

impl OverlapAddProcessor {
    /// Construct a new processor with the given configuration.
    ///
    /// The overlap tail is initialised to silence.
    #[must_use]
    pub fn new(config: OverlapAddConfig) -> Self {
        let block_size = config.block_size.max(2);
        let hop = config.hop_size();
        let overlap_len = block_size.saturating_sub(hop);
        let window = hann_window(block_size);
        let overlap_tail = vec![0.0f32; overlap_len];

        Self {
            config: OverlapAddConfig {
                block_size,
                ..config
            },
            window,
            overlap_tail,
            hop,
            overlap_len,
        }
    }

    /// Process one block of audio through the overlap-add framework.
    ///
    /// # Algorithm
    ///
    /// 1. **Pad** the input with the overlap tail from the previous call.
    /// 2. **Window** each sample: `windowed[i] = combined[i] * hann[i]`.
    /// 3. **Overlap-add**: add the windowed tail from the previous block to
    ///    the first `overlap_len` samples of the current output.
    /// 4. **Store** the tail for the next call.
    /// 5. **Return** the `hop`-length output segment (the non-overlapping
    ///    portion of the current block).
    ///
    /// # Errors
    ///
    /// Returns [`RestoreError::InvalidParameter`] when `input.len()` ≠
    /// `config.block_size`.
    pub fn process_block(&mut self, input: &[f32]) -> RestoreResult<Vec<f32>> {
        let block_size = self.config.block_size;

        if input.len() != block_size {
            return Err(RestoreError::InvalidParameter(format!(
                "OverlapAddProcessor: expected {block_size} samples, got {}",
                input.len()
            )));
        }

        // 1. Build the windowed block from the raw input
        let windowed: Vec<f32> = input
            .iter()
            .zip(self.window.iter())
            .map(|(&x, &w)| x * w)
            .collect();

        // 2. Overlap-add: combine the stored tail with the start of the new windowed block
        let mut output = vec![0.0f32; block_size];

        // Add the overlap tail from the previous block into the first overlap_len samples
        for i in 0..self.overlap_len {
            output[i] = self.overlap_tail[i] + windowed[i];
        }
        // Copy the non-overlapping portion of the current windowed block
        for i in self.overlap_len..block_size {
            output[i] = windowed[i];
        }

        // 3. Store the new overlap tail (last overlap_len samples of the output)
        let tail_start = self.hop;
        self.overlap_tail = output[tail_start..].to_vec();
        // Make sure overlap_tail is exactly overlap_len in length
        self.overlap_tail.resize(self.overlap_len, 0.0);

        // 4. Return the output block (full block_size; callers may trim to hop if needed)
        Ok(output)
    }

    /// Reset the processor state (clear the overlap tail).
    ///
    /// Call this before processing a new, unrelated audio stream to avoid
    /// contaminating the output with stale overlap data.
    pub fn reset(&mut self) {
        self.overlap_tail.fill(0.0);
    }

    /// Return the current configuration.
    #[must_use]
    pub fn config(&self) -> &OverlapAddConfig {
        &self.config
    }

    /// Return the hop size in samples.
    #[must_use]
    pub fn hop_size(&self) -> usize {
        self.hop
    }

    /// Return the overlap length in samples.
    #[must_use]
    pub fn overlap_len(&self) -> usize {
        self.overlap_len
    }
}

// ---------------------------------------------------------------------------
// OverlapAdd — simplified frame-based API
// ---------------------------------------------------------------------------

/// A simplified overlap-add processor using a fixed frame size and hop size.
///
/// Unlike [`OverlapAddProcessor`] (which requires every call to supply exactly
/// `block_size` samples), `OverlapAdd` accepts arbitrary-length frames and
/// accumulates output samples in an internal synthesis buffer.  Callers drain
/// the buffer by calling [`process`][OverlapAdd::process] which returns any
/// complete output samples.
///
/// # Example
///
/// ```
/// use oximedia_restore::overlap_add::OverlapAdd;
///
/// let mut ola = OverlapAdd::new(256, 128);
/// let frame: Vec<f32> = (0..256).map(|i| (i as f32 * 0.01).sin()).collect();
/// let output = ola.process(&frame);
/// assert_eq!(output.len(), 256);
/// ```
#[derive(Debug)]
pub struct OverlapAdd {
    frame_size: usize,
    hop_size: usize,
    /// Hann window coefficients.
    window: Vec<f32>,
    /// Synthesis accumulation buffer (length = frame_size).
    synthesis_buf: Vec<f32>,
    /// Position in the synthesis buffer where the next hop starts.
    write_pos: usize,
}

impl OverlapAdd {
    /// Create a new overlap-add processor.
    ///
    /// # Parameters
    ///
    /// * `frame_size` — analysis/synthesis block length in samples.  Must be ≥ 2.
    /// * `hop_size`   — number of samples advanced per frame.  Must be 1 ≤ `hop_size` ≤ `frame_size`.
    #[must_use]
    pub fn new(frame_size: usize, hop_size: usize) -> Self {
        let frame_size = frame_size.max(2);
        let hop_size = hop_size.clamp(1, frame_size);
        let window = (0..frame_size)
            .map(|i| {
                let angle = 2.0 * std::f64::consts::PI * i as f64 / frame_size as f64;
                (0.5 * (1.0 - angle.cos())) as f32
            })
            .collect::<Vec<f32>>();
        let synthesis_buf = vec![0.0f32; frame_size];
        Self {
            frame_size,
            hop_size,
            window,
            synthesis_buf,
            write_pos: 0,
        }
    }

    /// Process a single analysis frame through the overlap-add framework.
    ///
    /// The frame is windowed with a Hann window and accumulated into the
    /// internal synthesis buffer.  Returns the `frame_size` output samples
    /// from the current buffer position (the output may contain overlap from
    /// prior frames).
    ///
    /// If `frame.len()` differs from `frame_size` the input is silently
    /// zero-padded or truncated to `frame_size` to remain infallible.
    pub fn process(&mut self, frame: &[f32]) -> Vec<f32> {
        let fs = self.frame_size;
        let hs = self.hop_size;

        // Apply Hann window and accumulate into synthesis buffer.
        for i in 0..fs {
            let sample = if i < frame.len() { frame[i] } else { 0.0 };
            let idx = (self.write_pos + i) % fs;
            self.synthesis_buf[idx] += sample * self.window[i];
        }

        // Read `fs` samples starting from write_pos, wrapping around.
        let mut output = Vec::with_capacity(fs);
        for i in 0..fs {
            let idx = (self.write_pos + i) % fs;
            output.push(self.synthesis_buf[idx]);
        }

        // Zero-out the region we just consumed (the next hop_size samples).
        for i in 0..hs {
            let idx = (self.write_pos + i) % fs;
            self.synthesis_buf[idx] = 0.0;
        }

        // Advance write position by hop_size.
        self.write_pos = (self.write_pos + hs) % fs;

        output
    }

    /// Return the configured frame size.
    #[must_use]
    pub fn frame_size(&self) -> usize {
        self.frame_size
    }

    /// Return the configured hop size.
    #[must_use]
    pub fn hop_size(&self) -> usize {
        self.hop_size
    }

    /// Reset the synthesis buffer and position.
    pub fn reset(&mut self) {
        self.synthesis_buf.fill(0.0);
        self.write_pos = 0;
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // ------ OverlapAddConfig -------------------------------------------------

    #[test]
    fn test_config_hop_size_50_pct() {
        let config = OverlapAddConfig {
            block_size: 1024,
            overlap_pct: 0.5,
        };
        assert_eq!(
            config.hop_size(),
            512,
            "50 % overlap → hop = block_size / 2"
        );
    }

    #[test]
    fn test_config_hop_size_25_pct() {
        let config = OverlapAddConfig {
            block_size: 1024,
            overlap_pct: 0.25,
        };
        assert_eq!(
            config.hop_size(),
            768,
            "25 % overlap → hop = 3 * block_size / 4"
        );
    }

    #[test]
    fn test_config_hop_size_zero_overlap() {
        let config = OverlapAddConfig {
            block_size: 512,
            overlap_pct: 0.0,
        };
        assert_eq!(config.hop_size(), 512, "no overlap → hop = block_size");
    }

    #[test]
    fn test_config_hop_size_clamped_above_75pct() {
        let config = OverlapAddConfig {
            block_size: 1024,
            overlap_pct: 0.99, // clamped to 0.75
        };
        // 0.75 overlap → hop = 0.25 * 1024 = 256
        assert_eq!(config.hop_size(), 256);
    }

    // ------ Hann window ------------------------------------------------------

    #[test]
    fn test_hann_window_endpoints() {
        let w = hann_window(8);
        // w[0] = 0.5*(1-cos(0)) = 0.0
        assert!(w[0].abs() < 1e-5, "first sample should be ~0");
    }

    #[test]
    fn test_hann_window_midpoint() {
        let w = hann_window(8);
        // w[4] = 0.5*(1-cos(pi)) = 1.0
        assert!(
            (w[4] - 1.0).abs() < 1e-4,
            "midpoint of even-length Hann window ≈ 1.0"
        );
    }

    #[test]
    fn test_hann_window_length() {
        assert_eq!(hann_window(16).len(), 16);
        assert_eq!(hann_window(0).len(), 0);
    }

    // ------ OverlapAddProcessor construction --------------------------------

    #[test]
    fn test_processor_new_default_config() {
        let cfg = OverlapAddConfig::default();
        let proc = OverlapAddProcessor::new(cfg);
        assert_eq!(proc.hop_size(), 512, "default 50 % overlap → hop = 512");
        assert_eq!(proc.overlap_len(), 512);
    }

    #[test]
    fn test_processor_new_75pct_overlap() {
        let cfg = OverlapAddConfig {
            block_size: 512,
            overlap_pct: 0.75,
        };
        let proc = OverlapAddProcessor::new(cfg);
        assert_eq!(proc.hop_size(), 128);
        assert_eq!(proc.overlap_len(), 384);
    }

    // ------ process_block ---------------------------------------------------

    #[test]
    fn test_process_block_returns_correct_length() {
        let cfg = OverlapAddConfig {
            block_size: 256,
            overlap_pct: 0.5,
        };
        let mut proc = OverlapAddProcessor::new(cfg);
        let input = vec![0.5f32; 256];
        let output = proc.process_block(&input).expect("process ok");
        assert_eq!(output.len(), 256);
    }

    #[test]
    fn test_process_block_wrong_size_returns_err() {
        let cfg = OverlapAddConfig {
            block_size: 256,
            overlap_pct: 0.5,
        };
        let mut proc = OverlapAddProcessor::new(cfg);
        let bad_input = vec![0.0f32; 100]; // wrong size
        let result = proc.process_block(&bad_input);
        assert!(result.is_err(), "wrong block size must return Err");
    }

    #[test]
    fn test_process_block_silence_stays_zero() {
        let cfg = OverlapAddConfig {
            block_size: 256,
            overlap_pct: 0.0,
        };
        let mut proc = OverlapAddProcessor::new(cfg);
        let silent = vec![0.0f32; 256];
        let output = proc.process_block(&silent).expect("ok");
        for s in &output {
            assert!(s.abs() < 1e-7, "silence input → silence output");
        }
    }

    #[test]
    fn test_process_block_output_bounded() {
        // DC signal of 1.0; Hann window peak is 1.0 at midpoint; output should
        // not exceed 1.0 in the absence of accumulated overlap.
        let cfg = OverlapAddConfig {
            block_size: 128,
            overlap_pct: 0.0,
        };
        let mut proc = OverlapAddProcessor::new(cfg);
        let dc = vec![1.0f32; 128];
        let out = proc.process_block(&dc).expect("ok");
        for &s in &out {
            assert!(s >= -1.0 && s <= 1.0, "output sample {s} out of [-1, 1]");
        }
    }

    #[test]
    fn test_reset_clears_overlap_tail() {
        let cfg = OverlapAddConfig {
            block_size: 64,
            overlap_pct: 0.5,
        };
        let mut proc = OverlapAddProcessor::new(cfg);
        // Process a non-silent block to populate the tail
        let noise: Vec<f32> = (0..64).map(|i| (i as f32).sin()).collect();
        let _ = proc.process_block(&noise).expect("ok");

        // Reset and verify the tail is zeroed
        proc.reset();
        let silent = vec![0.0f32; 64];
        let out = proc.process_block(&silent).expect("ok");
        // After reset, the first overlap_len samples come from zero tail → should be zero
        for &s in out.iter().take(proc.overlap_len()) {
            assert!(s.abs() < 1e-7, "after reset, overlap tail should be zeroed");
        }
    }

    #[test]
    fn test_consecutive_blocks_accumulate_overlap() {
        // With 50 % overlap the second block's first half is influenced by the
        // first block's second half.  Verify that two non-silent blocks produce
        // different output in the overlap zone compared to a fresh processor.
        let cfg = OverlapAddConfig {
            block_size: 64,
            overlap_pct: 0.5,
        };
        let mut proc = OverlapAddProcessor::new(cfg);
        let block_a: Vec<f32> = vec![1.0f32; 64];
        let _ = proc.process_block(&block_a).expect("first block");
        let block_b: Vec<f32> = vec![1.0f32; 64];
        let out_b = proc.process_block(&block_b).expect("second block");

        // First half of output should be non-zero (overlap from block_a added in)
        let overlap_zone_nonzero = out_b.iter().take(32).any(|&s| s.abs() > 1e-5);
        assert!(
            overlap_zone_nonzero,
            "overlap zone should be non-zero after first block"
        );
    }

    // ─── OverlapAdd tests ──────────────────────────────────────────────────

    #[test]
    fn test_overlap_add_output_length() {
        let mut ola = OverlapAdd::new(256, 128);
        let frame = vec![0.5f32; 256];
        let out = ola.process(&frame);
        assert_eq!(out.len(), 256, "output must equal frame_size");
    }

    #[test]
    fn test_overlap_add_silence_stays_zero() {
        let mut ola = OverlapAdd::new(64, 32);
        let frame = vec![0.0f32; 64];
        let out = ola.process(&frame);
        for s in &out {
            assert!(s.abs() < 1e-7, "silence should produce silence");
        }
    }

    #[test]
    fn test_overlap_add_nonzero_input_nonzero_output() {
        let mut ola = OverlapAdd::new(64, 32);
        let frame = vec![1.0f32; 64];
        let out = ola.process(&frame);
        let any_nonzero = out.iter().any(|&s| s.abs() > 1e-6);
        assert!(
            any_nonzero,
            "non-silent input should produce non-silent output"
        );
    }

    #[test]
    fn test_overlap_add_reset_clears_state() {
        let mut ola = OverlapAdd::new(64, 32);
        let frame = vec![1.0f32; 64];
        let _ = ola.process(&frame);
        ola.reset();
        let silent = vec![0.0f32; 64];
        let out = ola.process(&silent);
        for s in &out {
            assert!(s.abs() < 1e-7, "after reset silence should give silence");
        }
    }

    #[test]
    fn test_overlap_add_frame_size_accessor() {
        let ola = OverlapAdd::new(512, 256);
        assert_eq!(ola.frame_size(), 512);
        assert_eq!(ola.hop_size(), 256);
    }

    #[test]
    fn test_overlap_add_short_frame_zero_padded() {
        // Providing fewer samples than frame_size should not panic
        let mut ola = OverlapAdd::new(64, 32);
        let short = vec![1.0f32; 16]; // less than 64
        let out = ola.process(&short);
        assert_eq!(out.len(), 64);
    }
}
