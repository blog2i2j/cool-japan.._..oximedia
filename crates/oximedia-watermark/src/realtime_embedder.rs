//! Real-time streaming watermark embedder with frame-by-frame processing and
//! full state persistence.
//!
//! This module provides [`RealtimeEmbedder`], which is designed for live/streaming
//! audio pipelines where the input arrives in small, variable-size chunks.  The
//! embedder maintains per-frame state so that a watermark can be spread across
//! an unbounded stream without buffering the entire signal.
//!
//! ## Architecture
//!
//! ```text
//! chunk_0 ──┐
//! chunk_1 ──┤  RealtimeEmbedder.process_chunk()  ──►  output_chunk_N
//! chunk_2 ──┘        (internal frame buffer)
//!            └── EmbedState (bit_index, PN sequence, overlap buffer)
//! ```
//!
//! Each call to [`RealtimeEmbedder::process_chunk`] appends input to an
//! internal ring buffer.  When a complete frame accumulates, the embedder
//! consumes it and forwards the watermarked samples downstream.  Any leftover
//! samples stay in the buffer for the next call.
//!
//! ## Persistence
//!
//! The full embedder state — including buffer contents, bit cursor, and payload
//! — can be serialised to and restored from a [`SnapshotState`].  This enables
//! crash recovery and hand-off between process boundaries.
//!
//! # Example
//!
//! ```no_run
//! use oximedia_watermark::realtime_embedder::{RealtimeEmbedder, RealtimeEmbedderConfig};
//!
//! let config = RealtimeEmbedderConfig::default();
//! let mut embedder = RealtimeEmbedder::new(config, 44100).expect("codec ok");
//!
//! // Feed the payload once; then keep calling process_chunk with live audio.
//! embedder.set_payload(b"LiveStream-001").expect("payload ok");
//!
//! // Simulate live audio arriving in 512-sample chunks.
//! let chunk: Vec<f32> = vec![0.0; 512];
//! let output = embedder.process_chunk(&chunk).expect("process ok");
//! assert_eq!(output.samples.len(), chunk.len());
//! ```

#![allow(dead_code)]

use std::collections::VecDeque;

use crate::{
    error::{WatermarkError, WatermarkResult},
    payload::{generate_pn_sequence, pack_bits, unpack_bits, PayloadCodec},
};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration for [`RealtimeEmbedder`].
#[derive(Debug, Clone)]
pub struct RealtimeEmbedderConfig {
    /// Embedding frame size (samples).  Must be a power of two for best FFT
    /// performance, but any positive value is accepted.
    pub frame_size: usize,
    /// Embedding strength α — scales the watermark delta added to each sample.
    pub strength: f32,
    /// Secret key used to generate the PN spreading sequence.
    pub key: u64,
    /// Chip rate (spreading factor) — number of samples per embedded bit.
    pub chip_rate: usize,
    /// If `true` the embedder loops the payload when all bits have been
    /// embedded.  If `false` it stops embedding after one pass and just
    /// passes samples through unmodified.
    pub loop_payload: bool,
    /// Number of bits for each Reed-Solomon "data shard" (see [`PayloadCodec`]).
    pub rs_data_shards: usize,
    /// Number of Reed-Solomon parity shards.
    pub rs_parity_shards: usize,
}

impl Default for RealtimeEmbedderConfig {
    fn default() -> Self {
        Self {
            frame_size: 2048,
            strength: 0.1,
            key: 0,
            chip_rate: 64,
            loop_payload: true,
            rs_data_shards: 16,
            rs_parity_shards: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// SnapshotState
// ---------------------------------------------------------------------------

/// A serialisable snapshot of the [`RealtimeEmbedder`] internal state.
///
/// Snapshot/restore allows:
/// - Crash recovery: restore after a process restart.
/// - Pipeline hand-off: pass the state to a different machine.
/// - Debugging: inspect the exact state at any point.
#[derive(Debug, Clone)]
pub struct SnapshotState {
    /// Current position in the bit stream (index into `encoded_bits`).
    pub bit_index: usize,
    /// Total number of bits in the encoded payload (including RS overhead).
    pub total_bits: usize,
    /// The encoded payload bits (including RS parity).
    pub encoded_bits: Vec<bool>,
    /// Samples currently buffered but not yet processed into a full frame.
    pub buffer: Vec<f32>,
    /// Total number of samples produced by this embedder so far.
    pub samples_produced: u64,
    /// Total number of frames processed by this embedder so far.
    pub frames_processed: u64,
    /// Embedder configuration at snapshot time.
    pub config: RealtimeEmbedderConfig,
}

// ---------------------------------------------------------------------------
// Embed statistics
// ---------------------------------------------------------------------------

/// Accumulated statistics for a running [`RealtimeEmbedder`].
#[derive(Debug, Clone, Default)]
pub struct EmbedStats {
    /// Number of complete frames processed so far.
    pub frames_processed: u64,
    /// Total samples produced (including flushed remainder).
    pub samples_produced: u64,
    /// Number of bits embedded so far (including wrap-arounds if `loop_payload`).
    pub bits_embedded: u64,
    /// Number of payload repeats (wrap-arounds) so far.
    pub payload_loops: u64,
}

// ---------------------------------------------------------------------------
// ProcessResult
// ---------------------------------------------------------------------------

/// Result of one [`RealtimeEmbedder::process_chunk`] call.
#[derive(Debug, Clone)]
pub struct ProcessResult {
    /// Output audio samples with watermark applied (same length as input).
    pub samples: Vec<f32>,
    /// Number of complete frames processed in this call.
    pub frames_processed: usize,
    /// Whether any embedding occurred (`false` if no payload is set or
    /// `loop_payload = false` and the payload was exhausted before this chunk).
    pub embedded: bool,
}

// ---------------------------------------------------------------------------
// RealtimeEmbedder
// ---------------------------------------------------------------------------

/// Frame-by-frame watermark embedder for live/streaming audio pipelines.
pub struct RealtimeEmbedder {
    config: RealtimeEmbedderConfig,
    /// Ring buffer holding incoming samples waiting to fill a frame.
    buffer: VecDeque<f32>,
    /// Encoded payload bits (RS-encoded + sync).
    encoded_bits: Vec<bool>,
    /// Current position in `encoded_bits`.
    bit_index: usize,
    /// PN spreading sequence (pre-generated, one chip per sample in a frame).
    pn_sequence: Vec<i8>,
    /// Accumulated statistics.
    stats: EmbedStats,
    /// Whether a payload has been configured.
    payload_set: bool,
    /// The raw payload (for snapshot/restore).
    raw_payload: Vec<u8>,
    /// Sample rate (for future psychoacoustic extension).
    sample_rate: u32,
    /// Reed-Solomon codec.
    codec: PayloadCodec,
}

impl RealtimeEmbedder {
    /// Create a new embedder.  No payload is set; call [`set_payload`][Self::set_payload]
    /// before processing to enable embedding.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if the RS codec cannot be initialised.
    pub fn new(config: RealtimeEmbedderConfig, sample_rate: u32) -> WatermarkResult<Self> {
        let codec =
            PayloadCodec::new(config.rs_data_shards, config.rs_parity_shards)?;

        // Pre-generate a PN sequence of `chip_rate` chips — one per sample of
        // a single bit's worth of audio.
        let pn_sequence = generate_pn_sequence(config.chip_rate, config.key);

        Ok(Self {
            config,
            buffer: VecDeque::new(),
            encoded_bits: Vec::new(),
            bit_index: 0,
            pn_sequence,
            stats: EmbedStats::default(),
            payload_set: false,
            raw_payload: Vec::new(),
            sample_rate,
            codec,
        })
    }

    /// Set (or replace) the payload to embed.
    ///
    /// This re-encodes the payload with Reed-Solomon and resets the bit cursor
    /// to the beginning of the new bit stream.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if encoding fails.
    pub fn set_payload(&mut self, payload: &[u8]) -> WatermarkResult<()> {
        let encoded_bytes = self.codec.encode(payload)?;
        self.encoded_bits = unpack_bits(&encoded_bytes, encoded_bytes.len() * 8);
        self.bit_index = 0;
        self.payload_set = true;
        self.raw_payload = payload.to_vec();
        Ok(())
    }

    /// Process one chunk of audio samples.
    ///
    /// Samples are appended to the internal buffer.  As complete frames
    /// accumulate they are watermarked and written to the output.  The output
    /// always has the **same length** as the input.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if internal processing fails.
    pub fn process_chunk(&mut self, input: &[f32]) -> WatermarkResult<ProcessResult> {
        // Append input to the ring buffer.
        self.buffer.extend(input.iter().copied());

        let mut output = Vec::with_capacity(input.len());
        let mut frames_in_this_call = 0usize;
        let mut any_embedded = false;

        let frame_size = self.config.frame_size;

        // Process as many complete frames as possible.
        while self.buffer.len() >= frame_size {
            let frame: Vec<f32> = self.buffer.drain(..frame_size).collect();
            let processed = self.process_frame(frame);
            if processed.embedded {
                any_embedded = true;
            }
            output.extend_from_slice(&processed.samples);
            frames_in_this_call += 1;
            self.stats.frames_processed += 1;
        }

        // Pass through any remaining buffered samples that don't fill a frame.
        // They are kept in the buffer but we also emit them as-is so the output
        // length matches the input length.
        let remaining_in_buffer = self.buffer.len();
        let needed_from_buffer = input.len().saturating_sub(output.len());
        let passthrough_count = needed_from_buffer.min(remaining_in_buffer);

        // Peek at the front of the buffer for the pass-through samples — we do
        // NOT drain them; they stay for the next full-frame attempt.
        for sample in self.buffer.iter().take(passthrough_count) {
            output.push(*sample);
        }

        // Pad to exact input length if we somehow have fewer samples.
        while output.len() < input.len() {
            output.push(0.0);
        }
        output.truncate(input.len());

        self.stats.samples_produced += input.len() as u64;

        Ok(ProcessResult {
            samples: output,
            frames_processed: frames_in_this_call,
            embedded: any_embedded,
        })
    }

    /// Flush any remaining buffered samples, applying a partial watermark frame
    /// if enough bits are available.  Returns the flushed samples (may be empty).
    ///
    /// Call this when the stream ends to drain the internal buffer.
    #[must_use]
    pub fn flush(&mut self) -> Vec<f32> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let remaining: Vec<f32> = self.buffer.drain(..).collect();
        // Apply watermark to the partial frame if payload is set.
        if self.payload_set && !self.encoded_bits.is_empty() {
            self.embed_partial_frame(remaining)
        } else {
            remaining
        }
    }

    /// Take a snapshot of the current state for persistence.
    #[must_use]
    pub fn snapshot(&self) -> SnapshotState {
        SnapshotState {
            bit_index: self.bit_index,
            total_bits: self.encoded_bits.len(),
            encoded_bits: self.encoded_bits.clone(),
            buffer: self.buffer.iter().copied().collect(),
            samples_produced: self.stats.samples_produced,
            frames_processed: self.stats.frames_processed,
            config: self.config.clone(),
        }
    }

    /// Restore state from a snapshot.
    ///
    /// # Errors
    ///
    /// Returns [`WatermarkError`] if the snapshot's RS parameters are invalid.
    pub fn restore(&mut self, snap: SnapshotState) -> WatermarkResult<()> {
        if snap.config.rs_data_shards != self.config.rs_data_shards
            || snap.config.rs_parity_shards != self.config.rs_parity_shards
        {
            return Err(WatermarkError::InvalidParameter(
                "Snapshot RS parameters don't match embedder configuration".to_string(),
            ));
        }
        self.bit_index = snap.bit_index;
        self.encoded_bits = snap.encoded_bits;
        self.buffer = snap.buffer.into_iter().collect();
        self.stats.samples_produced = snap.samples_produced;
        self.stats.frames_processed = snap.frames_processed;
        self.payload_set = !self.encoded_bits.is_empty();
        Ok(())
    }

    /// Return current embedding statistics.
    #[must_use]
    pub fn stats(&self) -> &EmbedStats {
        &self.stats
    }

    /// Whether a payload has been configured.
    #[must_use]
    pub fn payload_set(&self) -> bool {
        self.payload_set
    }

    /// Number of samples currently in the internal buffer awaiting a full frame.
    #[must_use]
    pub fn buffered_samples(&self) -> usize {
        self.buffer.len()
    }

    /// Number of bits in the encoded payload (including RS overhead).
    #[must_use]
    pub fn encoded_bit_count(&self) -> usize {
        self.encoded_bits.len()
    }

    /// Current bit-cursor position within the encoded payload.
    #[must_use]
    pub fn bit_index(&self) -> usize {
        self.bit_index
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Process one complete frame.
    fn process_frame(&mut self, mut frame: Vec<f32>) -> FrameResult {
        if !self.payload_set || self.encoded_bits.is_empty() {
            return FrameResult {
                samples: frame,
                embedded: false,
            };
        }

        // Determine how many bits can fit in this frame.
        let bits_per_frame = frame.len() / self.config.chip_rate;
        if bits_per_frame == 0 {
            return FrameResult {
                samples: frame,
                embedded: false,
            };
        }

        // Check whether we have bits to embed.
        let total = self.encoded_bits.len();
        if !self.config.loop_payload && self.bit_index >= total {
            return FrameResult {
                samples: frame,
                embedded: false,
            };
        }

        let mut embedded = false;
        for bit_slot in 0..bits_per_frame {
            let bit_idx = if self.config.loop_payload {
                self.bit_index % total
            } else {
                if self.bit_index >= total {
                    break;
                }
                self.bit_index
            };

            let bit = self.encoded_bits[bit_idx];
            let sign: f32 = if bit { 1.0 } else { -1.0 };

            // Embed one bit using spread-spectrum: add `strength * sign * pn[i]`
            // for each sample in the chip window.
            let sample_start = bit_slot * self.config.chip_rate;
            let sample_end = (sample_start + self.config.chip_rate).min(frame.len());
            let chip_len = self.pn_sequence.len().min(sample_end - sample_start);

            for (i, sample) in frame[sample_start..sample_end]
                .iter_mut()
                .enumerate()
                .take(chip_len)
            {
                *sample += self.config.strength * sign * f32::from(self.pn_sequence[i]);
            }

            self.bit_index += 1;
            if self.config.loop_payload && self.bit_index >= total {
                self.bit_index = 0;
                self.stats.payload_loops += 1;
            }
            self.stats.bits_embedded += 1;
            embedded = true;
        }

        FrameResult {
            samples: frame,
            embedded,
        }
    }

    /// Apply a partial watermark to a sub-frame (used during flush).
    #[allow(clippy::cast_precision_loss)]
    fn embed_partial_frame(&mut self, mut samples: Vec<f32>) -> Vec<f32> {
        if self.encoded_bits.is_empty() {
            return samples;
        }
        let total = self.encoded_bits.len();
        let bits_available = if self.config.loop_payload {
            total
        } else {
            total.saturating_sub(self.bit_index)
        };
        if bits_available == 0 {
            return samples;
        }

        let chip_rate = self.config.chip_rate;
        let bits_possible = samples.len() / chip_rate;
        let bits_to_embed = bits_possible.min(bits_available);

        for bit_slot in 0..bits_to_embed {
            let bit_idx = if self.config.loop_payload {
                self.bit_index % total
            } else {
                self.bit_index
            };
            let bit = self.encoded_bits[bit_idx];
            let sign: f32 = if bit { 1.0 } else { -1.0 };

            let sample_start = bit_slot * chip_rate;
            let sample_end = (sample_start + chip_rate).min(samples.len());
            let chip_len = self.pn_sequence.len().min(sample_end - sample_start);

            for (i, sample) in samples[sample_start..sample_end]
                .iter_mut()
                .enumerate()
                .take(chip_len)
            {
                *sample += self.config.strength * sign * f32::from(self.pn_sequence[i]);
            }

            self.bit_index += 1;
            if self.config.loop_payload && self.bit_index >= total {
                self.bit_index = 0;
                self.stats.payload_loops += 1;
            }
            self.stats.bits_embedded += 1;
        }
        samples
    }
}

// ---------------------------------------------------------------------------
// Private intermediate result
// ---------------------------------------------------------------------------

struct FrameResult {
    samples: Vec<f32>,
    embedded: bool,
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embedder() -> RealtimeEmbedder {
        let config = RealtimeEmbedderConfig {
            frame_size: 1024,
            strength: 0.1,
            key: 42,
            chip_rate: 32,
            loop_payload: true,
            rs_data_shards: 8,
            rs_parity_shards: 4,
        };
        RealtimeEmbedder::new(config, 44100).unwrap()
    }

    #[test]
    fn test_process_chunk_output_length_matches_input() {
        let mut emb = make_embedder();
        emb.set_payload(b"Hello").expect("set_payload ok");

        let chunk: Vec<f32> = vec![0.0; 512];
        let result = emb.process_chunk(&chunk).expect("process ok");
        assert_eq!(result.samples.len(), chunk.len());
    }

    #[test]
    fn test_process_large_chunk_output_length_matches_input() {
        let mut emb = make_embedder();
        emb.set_payload(b"Large").expect("set_payload ok");

        let chunk: Vec<f32> = vec![0.5; 4096];
        let result = emb.process_chunk(&chunk).expect("process ok");
        assert_eq!(result.samples.len(), chunk.len());
    }

    #[test]
    fn test_without_payload_passthrough() {
        let mut emb = make_embedder();
        // No payload set — should pass through unchanged.
        let chunk: Vec<f32> = (0..512).map(|i| i as f32 * 0.001).collect();
        let result = emb.process_chunk(&chunk).expect("process ok");
        assert_eq!(result.samples.len(), chunk.len());
        assert!(!result.embedded);
    }

    #[test]
    fn test_set_payload_enables_embedding() {
        let mut emb = make_embedder();
        // Feed a large chunk so at least one full frame is processed.
        emb.set_payload(b"Embed").expect("set_payload ok");
        let chunk: Vec<f32> = vec![0.0; 2048];
        let result = emb.process_chunk(&chunk).expect("process ok");
        assert!(
            result.embedded || result.frames_processed == 0,
            "if frames were processed, embedding should have occurred"
        );
    }

    #[test]
    fn test_embedding_modifies_samples() {
        let mut emb = make_embedder();
        emb.set_payload(b"Modify").expect("ok");

        let input: Vec<f32> = vec![0.0; 2048];
        let result = emb.process_chunk(&input).expect("ok");

        // At least some samples should differ from 0.0 due to embedding.
        let modified = result.samples.iter().any(|&s| s.abs() > 1e-9);
        assert!(modified, "embedding should modify at least some samples");
    }

    #[test]
    fn test_payload_can_be_replaced() {
        let mut emb = make_embedder();
        emb.set_payload(b"First").expect("ok");
        let bit_count_first = emb.encoded_bit_count();

        emb.set_payload(b"A completely different payload with more bytes")
            .expect("ok");
        let bit_count_second = emb.encoded_bit_count();

        // Different payloads → different encoded bit counts.
        // (The RS overhead means both are non-zero, but lengths differ.)
        assert!(bit_count_first > 0);
        assert!(bit_count_second > 0);
        // Bit index is reset to 0 after set_payload.
        assert_eq!(emb.bit_index(), 0);
    }

    #[test]
    fn test_stats_accumulate() {
        let mut emb = make_embedder();
        emb.set_payload(b"Stats").expect("ok");

        let chunk: Vec<f32> = vec![0.0; 4096];
        emb.process_chunk(&chunk).expect("ok");

        assert!(emb.stats().samples_produced > 0);
    }

    #[test]
    fn test_snapshot_restore_roundtrip() {
        let mut emb = make_embedder();
        emb.set_payload(b"Snapshot").expect("ok");

        // Process some data to advance state.
        let chunk: Vec<f32> = vec![0.1; 2048];
        emb.process_chunk(&chunk).expect("ok");

        let snap = emb.snapshot();
        let bit_idx_before = snap.bit_index;
        let samples_before = snap.samples_produced;

        // Process more data.
        let chunk2: Vec<f32> = vec![0.2; 1024];
        emb.process_chunk(&chunk2).expect("ok");

        // Restore from snapshot.
        emb.restore(snap).expect("restore ok");
        assert_eq!(emb.bit_index(), bit_idx_before);
        assert_eq!(emb.stats().samples_produced, samples_before);
    }

    #[test]
    fn test_flush_drains_buffer() {
        let mut emb = make_embedder();
        emb.set_payload(b"Flush").expect("ok");

        // Feed a chunk that doesn't fill a complete frame.
        let chunk: Vec<f32> = vec![0.0; 500]; // < frame_size (1024)
        emb.process_chunk(&chunk).expect("ok");

        let buffered = emb.buffered_samples();
        let flushed = emb.flush();
        // After flush the buffer should be empty.
        assert_eq!(emb.buffered_samples(), 0);
        assert_eq!(flushed.len(), buffered);
    }

    #[test]
    fn test_loop_payload_wraps_bit_index() {
        let config = RealtimeEmbedderConfig {
            frame_size: 128,
            chip_rate: 8,
            loop_payload: true,
            rs_data_shards: 4,
            rs_parity_shards: 2,
            ..Default::default()
        };
        let mut emb = RealtimeEmbedder::new(config, 44100).unwrap();
        emb.set_payload(b"W").expect("ok");
        let total_bits = emb.encoded_bit_count();

        // Process enough data to force multiple wraps.
        let big_chunk: Vec<f32> = vec![0.0; total_bits * 10 * 8]; // 10 full passes
        emb.process_chunk(&big_chunk).expect("ok");

        assert!(
            emb.stats().payload_loops > 0,
            "payload should have looped at least once"
        );
    }

    #[test]
    fn test_no_loop_stops_after_one_pass() {
        let config = RealtimeEmbedderConfig {
            frame_size: 128,
            chip_rate: 8,
            loop_payload: false,
            rs_data_shards: 4,
            rs_parity_shards: 2,
            ..Default::default()
        };
        let mut emb = RealtimeEmbedder::new(config, 44100).unwrap();
        emb.set_payload(b"NL").expect("ok");
        let total_bits = emb.encoded_bit_count();

        // Process enough data for 3x the payload.
        let big_chunk: Vec<f32> = vec![0.0; total_bits * 3 * 8];
        emb.process_chunk(&big_chunk).expect("ok");

        assert_eq!(emb.stats().payload_loops, 0, "no-loop mode must never wrap");
        assert!(
            emb.bit_index() >= total_bits,
            "bit_index should be at end of payload"
        );
    }

    #[test]
    fn test_process_multiple_small_chunks_total_output_length() {
        let mut emb = make_embedder();
        emb.set_payload(b"Multi").expect("ok");

        let chunk_sizes = [100, 200, 50, 300, 150];
        let mut total_out = 0usize;
        for &sz in &chunk_sizes {
            let chunk: Vec<f32> = vec![0.0; sz];
            let result = emb.process_chunk(&chunk).expect("ok");
            assert_eq!(result.samples.len(), sz);
            total_out += result.samples.len();
        }
        let total_in: usize = chunk_sizes.iter().sum();
        assert_eq!(total_out, total_in);
    }

    #[test]
    fn test_snapshot_mismatched_rs_params_returns_error() {
        let mut emb1 = make_embedder(); // rs_data=8, rs_parity=4
        let config2 = RealtimeEmbedderConfig {
            rs_data_shards: 16,
            rs_parity_shards: 8,
            ..Default::default()
        };
        let mut emb2 = RealtimeEmbedder::new(config2, 44100).unwrap();
        emb2.set_payload(b"X").expect("ok");

        let snap = emb2.snapshot();
        let result = emb1.restore(snap);
        assert!(
            result.is_err(),
            "restoring snapshot with mismatched RS params must fail"
        );
    }

    #[test]
    fn test_payload_set_flag() {
        let mut emb = make_embedder();
        assert!(!emb.payload_set());
        emb.set_payload(b"Flag").expect("ok");
        assert!(emb.payload_set());
    }
}
