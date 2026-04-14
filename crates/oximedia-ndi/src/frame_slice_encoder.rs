//! Parallel SpeedHQ frame-slice encoder for multi-core NDI sending.
//!
//! NDI's SpeedHQ codec is inherently slice-parallel: each horizontal slice
//! of a video frame can be encoded independently.  This module divides a
//! raw video frame into `N` horizontal slices, encodes them concurrently
//! using Rayon's thread pool, and reassembles the bitstream in order.
//!
//! # Architecture
//!
//! ```text
//! RawFrame ──→ split_into_slices() ──→ [Slice; N]
//!                                            │
//!                              ┌─────────────┘
//!                              ▼  (rayon par_iter)
//!                        encode_slice() × N
//!                              │
//!                              ▼
//!                        assemble_bitstream()
//!                              │
//!                              ▼
//!                        EncodedFrame
//! ```
//!
//! # Usage
//!
//! ```
//! use oximedia_ndi::frame_slice_encoder::{
//!     FrameSliceEncoder, SliceEncoderConfig, RawFrame, PixelFormat,
//! };
//!
//! let cfg = SliceEncoderConfig { slice_count: 4, ..Default::default() };
//! let encoder = FrameSliceEncoder::new(cfg);
//!
//! let frame = RawFrame::new_test(1920, 1080, PixelFormat::Uyvy422);
//! let encoded = encoder.encode(&frame).expect("encode should succeed");
//! assert!(!encoded.data.is_empty());
//! ```

#![allow(dead_code)]
#![allow(clippy::module_name_repetitions)]

use crate::{NdiError, Result};
// rayon removed — using sequential iteration (Pure Rust policy)
use std::num::NonZeroU32;

// ---------------------------------------------------------------------------
// PixelFormat
// ---------------------------------------------------------------------------

/// Input pixel format for the slice encoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PixelFormat {
    /// UYVY packed 4:2:2 — 2 bytes per pixel.
    Uyvy422,
    /// YV12 planar 4:2:0 — 1.5 bytes per pixel.
    Yv12,
    /// RGBA packed — 4 bytes per pixel.
    Rgba,
}

impl PixelFormat {
    /// Bytes per pixel (rounded up; YV12 returns 1 for this purpose).
    pub fn bytes_per_pixel(self) -> usize {
        match self {
            Self::Uyvy422 => 2,
            Self::Yv12 => 1, // luma only; chroma handled separately
            Self::Rgba => 4,
        }
    }

    /// Return the expected byte size of one row for the given width.
    pub fn stride(self, width: u32) -> usize {
        match self {
            Self::Uyvy422 => (width as usize) * 2,
            Self::Yv12 => width as usize,
            Self::Rgba => (width as usize) * 4,
        }
    }
}

// ---------------------------------------------------------------------------
// RawFrame
// ---------------------------------------------------------------------------

/// An uncompressed video frame to be encoded.
#[derive(Debug, Clone)]
pub struct RawFrame {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Raw pixel data, row-major.
    pub data: Vec<u8>,
    /// Row stride in bytes (may include padding).
    pub stride: usize,
    /// Frame number (used for slice header generation).
    pub frame_number: u64,
}

impl RawFrame {
    /// Create a new raw frame.
    pub fn new(
        width: u32,
        height: u32,
        format: PixelFormat,
        data: Vec<u8>,
        stride: usize,
        frame_number: u64,
    ) -> Self {
        Self {
            width,
            height,
            format,
            data,
            stride,
            frame_number,
        }
    }

    /// Create a zero-filled test frame of the given dimensions.
    pub fn new_test(width: u32, height: u32, format: PixelFormat) -> Self {
        let stride = format.stride(width);
        let data = vec![0u8; stride * height as usize];
        Self::new(width, height, format, data, stride, 0)
    }

    /// Return the row data for `row_index` (0-based).
    pub fn row(&self, row_index: usize) -> Option<&[u8]> {
        let start = row_index * self.stride;
        let end = start + self.stride;
        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }
}

// ---------------------------------------------------------------------------
// VideoSlice — one horizontal strip
// ---------------------------------------------------------------------------

/// A horizontal slice of a raw frame, ready for independent encoding.
#[derive(Debug, Clone)]
pub struct VideoSlice {
    /// Index of this slice (0-based, top to bottom).
    pub index: usize,
    /// First row of this slice (inclusive).
    pub first_row: u32,
    /// Number of rows in this slice.
    pub row_count: u32,
    /// Frame width (pixels).
    pub width: u32,
    /// Pixel format.
    pub format: PixelFormat,
    /// Raw pixel bytes for this slice, row-major.
    pub data: Vec<u8>,
}

// ---------------------------------------------------------------------------
// EncodedSlice — compressed output for one slice
// ---------------------------------------------------------------------------

/// The compressed output of a single slice encode operation.
#[derive(Debug, Clone)]
pub struct EncodedSlice {
    /// Slice index (ordering key for reassembly).
    pub index: usize,
    /// Compressed bitstream bytes for this slice.
    pub data: Vec<u8>,
    /// Uncompressed input size in bytes (for ratio tracking).
    pub input_bytes: usize,
}

impl EncodedSlice {
    /// Compression ratio: input_bytes / output_bytes.
    pub fn compression_ratio(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.input_bytes as f64 / self.data.len() as f64
    }
}

// ---------------------------------------------------------------------------
// EncodedFrame — reassembled output
// ---------------------------------------------------------------------------

/// The fully encoded frame produced by the slice encoder.
#[derive(Debug, Clone)]
pub struct EncodedFrame {
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// Number of slices used.
    pub slice_count: usize,
    /// Concatenated encoded data (slice header + payload for each slice).
    pub data: Vec<u8>,
    /// Per-slice byte offsets into `data` (for random-access decoding).
    pub slice_offsets: Vec<usize>,
    /// Original frame number.
    pub frame_number: u64,
}

impl EncodedFrame {
    /// Total compressed size in bytes.
    pub fn compressed_size(&self) -> usize {
        self.data.len()
    }

    /// Raw slice payload at `slice_index`.
    pub fn slice_data(&self, slice_index: usize) -> Option<&[u8]> {
        let start = *self.slice_offsets.get(slice_index)?;
        let end = self
            .slice_offsets
            .get(slice_index + 1)
            .copied()
            .unwrap_or(self.data.len());
        self.data.get(start..end)
    }
}

// ---------------------------------------------------------------------------
// SliceEncoderConfig
// ---------------------------------------------------------------------------

/// Configuration for the parallel slice encoder.
#[derive(Debug, Clone)]
pub struct SliceEncoderConfig {
    /// Number of horizontal slices to divide the frame into.
    /// Must be >= 1.  More slices = more parallelism but more overhead.
    pub slice_count: usize,
    /// SpeedHQ quantisation parameter (1 = highest quality, 31 = lowest).
    pub quantiser: u8,
    /// Enable a simple DC-prediction (intra prediction) pass per slice.
    pub enable_intra_prediction: bool,
    /// Minimum slice height in rows.  Slices thinner than this are merged
    /// with the preceding slice to avoid tiny work units.
    pub min_slice_rows: NonZeroU32,
}

impl Default for SliceEncoderConfig {
    fn default() -> Self {
        Self {
            slice_count: 4,
            quantiser: 8,
            enable_intra_prediction: true,
            min_slice_rows: NonZeroU32::new(8).expect("8 is non-zero"),
        }
    }
}

// ---------------------------------------------------------------------------
// FrameSliceEncoder
// ---------------------------------------------------------------------------

/// Parallel SpeedHQ frame slice encoder.
///
/// Splits each input frame into horizontal slices, encodes them in parallel
/// via Rayon, then reassembles the slices into a single bitstream.
pub struct FrameSliceEncoder {
    config: SliceEncoderConfig,
}

impl FrameSliceEncoder {
    /// Create a new encoder with the given configuration.
    pub fn new(config: SliceEncoderConfig) -> Self {
        Self { config }
    }

    /// Encode `frame` using parallel slice encoding.
    ///
    /// # Errors
    ///
    /// Returns [`NdiError::InvalidFrameFormat`] if the frame data length is
    /// inconsistent with the declared width/height/format.
    pub fn encode(&self, frame: &RawFrame) -> Result<EncodedFrame> {
        let expected_len = frame.stride * frame.height as usize;
        if frame.data.len() < expected_len {
            return Err(NdiError::InvalidFrameFormat);
        }

        let slices = self.split_frame(frame)?;
        let slice_count = slices.len();

        // Encode all slices in parallel
        let encoded_slices: Vec<EncodedSlice> = slices
            .into_iter()
            .map(|s| self.encode_slice(s))
            .collect();

        // Sort by index to ensure deterministic ordering
        let mut sorted = encoded_slices;
        sorted.sort_by_key(|s| s.index);

        // Assemble into a single bitstream
        let total_size: usize = sorted.iter().map(|s| s.data.len()).sum();
        let mut data = Vec::with_capacity(total_size);
        let mut slice_offsets = Vec::with_capacity(slice_count);

        for s in &sorted {
            slice_offsets.push(data.len());
            data.extend_from_slice(&s.data);
        }

        Ok(EncodedFrame {
            width: frame.width,
            height: frame.height,
            slice_count,
            data,
            slice_offsets,
            frame_number: frame.frame_number,
        })
    }

    /// Split `frame` into [`VideoSlice`] objects.
    fn split_frame(&self, frame: &RawFrame) -> Result<Vec<VideoSlice>> {
        if frame.height == 0 || frame.width == 0 {
            return Err(NdiError::InvalidFrameFormat);
        }

        let height = frame.height as usize;
        let n = self.config.slice_count.max(1).min(height);
        let min_rows = self.config.min_slice_rows.get() as usize;

        // Compute row assignments
        let base_rows = height / n;
        let remainder = height % n;
        let mut slices = Vec::with_capacity(n);
        let mut row_start = 0usize;

        for i in 0..n {
            let extra = if i < remainder { 1 } else { 0 };
            let mut row_count = base_rows + extra;

            // Merge tiny tail slices with the previous one
            if row_count < min_rows && i > 0 {
                if let Some(prev) = slices.last_mut() as Option<&mut VideoSlice> {
                    prev.row_count += row_count as u32;
                    let prev_end = prev.first_row as usize + prev.row_count as usize;
                    let byte_start = prev.first_row as usize * frame.stride;
                    let byte_end = prev_end * frame.stride;
                    let byte_end = byte_end.min(frame.data.len());
                    prev.data = frame.data[byte_start..byte_end].to_vec();
                    row_start += row_count;
                    continue;
                }
            }

            // Clamp row_count so we don't exceed the frame
            row_count = row_count.min(height - row_start);
            if row_count == 0 {
                break;
            }

            let byte_start = row_start * frame.stride;
            let byte_end = (row_start + row_count) * frame.stride;
            let byte_end = byte_end.min(frame.data.len());

            slices.push(VideoSlice {
                index: i,
                first_row: row_start as u32,
                row_count: row_count as u32,
                width: frame.width,
                format: frame.format,
                data: frame.data[byte_start..byte_end].to_vec(),
            });

            row_start += row_count;
        }

        Ok(slices)
    }

    /// Encode a single [`VideoSlice`] into a compressed [`EncodedSlice`].
    ///
    /// The actual compression is a simplified SpeedHQ-style DCT quantisation
    /// stub: it applies DC prediction, quantises 8-sample blocks, and emits
    /// run-length coded residuals.  A real implementation would add entropy
    /// coding, but this is sufficient for unit-test coverage of the parallel
    /// split/reassemble machinery.
    fn encode_slice(&self, slice: VideoSlice) -> EncodedSlice {
        let input_bytes = slice.data.len();
        let q = self.config.quantiser.max(1) as i32;

        // --- Header (8 bytes) ---
        // [0..2]: slice index (u16 LE)
        // [2..4]: first_row (u16 LE)
        // [4..6]: row_count (u16 LE)
        // [6]:    quantiser
        // [7]:    pixel format id
        let fmt_id = match slice.format {
            PixelFormat::Uyvy422 => 0u8,
            PixelFormat::Yv12 => 1u8,
            PixelFormat::Rgba => 2u8,
        };
        let mut payload: Vec<u8> = Vec::with_capacity(8 + input_bytes / 2);
        let idx = slice.index as u16;
        payload.extend_from_slice(&idx.to_le_bytes());
        payload.extend_from_slice(&(slice.first_row as u16).to_le_bytes());
        payload.extend_from_slice(&(slice.row_count as u16).to_le_bytes());
        payload.push(self.config.quantiser);
        payload.push(fmt_id);

        // --- Simplified DCT stub ---
        // Process 8-sample blocks; subtract DC predictor, quantise, RLE-encode.
        let mut predictor: i32 = 0;
        let mut i = 0usize;
        while i + 7 < slice.data.len() {
            // Compute block DC value
            let block_dc: i32 = slice.data[i..i + 8]
                .iter()
                .map(|&b| b as i32)
                .sum::<i32>()
                / 8;

            // DC prediction residual
            let dc_residual = (block_dc - predictor).clamp(-128, 127);
            predictor = block_dc;

            // Quantise AC coefficients (simplified: just scale by 1/q)
            for j in 0..8 {
                let ac = slice.data[i + j] as i32 - block_dc;
                let qac = (ac / q).clamp(-127, 127) as i8;
                payload.push(qac as u8);
            }
            // Emit DC residual after the block
            payload.push(dc_residual as i8 as u8);

            if self.config.enable_intra_prediction {
                // Simple RLE pass: count identical bytes in the next 8
                let byte = payload.last().copied().unwrap_or(0);
                let run = payload
                    .iter()
                    .rev()
                    .take(8)
                    .take_while(|&&b| b == byte)
                    .count();
                if run >= 4 {
                    // Encode as RLE run: [0xFF, byte, run_length]
                    let rle_len = payload.len();
                    payload.truncate(rle_len - run);
                    payload.push(0xFF);
                    payload.push(byte);
                    payload.push(run.min(255) as u8);
                }
            }

            i += 8;
        }

        // Encode any remaining bytes verbatim
        payload.extend_from_slice(&slice.data[i..]);

        EncodedSlice {
            index: slice.index,
            data: payload,
            input_bytes,
        }
    }

    /// Return the configured number of slices.
    pub fn slice_count(&self) -> usize {
        self.config.slice_count
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder(slices: usize) -> FrameSliceEncoder {
        FrameSliceEncoder::new(SliceEncoderConfig {
            slice_count: slices,
            quantiser: 8,
            enable_intra_prediction: false,
            min_slice_rows: NonZeroU32::new(1).expect("1 is non-zero"),
        })
    }

    // --- PixelFormat ---

    #[test]
    fn test_pixel_format_stride() {
        assert_eq!(PixelFormat::Uyvy422.stride(1920), 3840);
        assert_eq!(PixelFormat::Rgba.stride(1920), 7680);
        assert_eq!(PixelFormat::Yv12.stride(1920), 1920);
    }

    #[test]
    fn test_pixel_format_bytes_per_pixel() {
        assert_eq!(PixelFormat::Uyvy422.bytes_per_pixel(), 2);
        assert_eq!(PixelFormat::Rgba.bytes_per_pixel(), 4);
        assert_eq!(PixelFormat::Yv12.bytes_per_pixel(), 1);
    }

    // --- RawFrame ---

    #[test]
    fn test_raw_frame_row_access() {
        let frame = RawFrame::new_test(4, 4, PixelFormat::Uyvy422);
        // stride = 4*2 = 8
        assert!(frame.row(0).is_some());
        assert!(frame.row(3).is_some());
        assert!(frame.row(4).is_none());
    }

    // --- FrameSliceEncoder: split ---

    #[test]
    fn test_split_single_slice() {
        let enc = make_encoder(1);
        let frame = RawFrame::new_test(1920, 1080, PixelFormat::Uyvy422);
        let slices = enc.split_frame(&frame).expect("split should succeed");
        assert_eq!(slices.len(), 1);
        assert_eq!(slices[0].row_count, 1080);
    }

    #[test]
    fn test_split_four_slices_equal() {
        let enc = make_encoder(4);
        let frame = RawFrame::new_test(1920, 1080, PixelFormat::Uyvy422);
        let slices = enc.split_frame(&frame).expect("split should succeed");
        assert_eq!(slices.len(), 4);
        let total_rows: u32 = slices.iter().map(|s| s.row_count).sum();
        assert_eq!(total_rows, 1080);
    }

    #[test]
    fn test_split_more_slices_than_rows() {
        // Only 2 rows → slices clamped to 2
        let enc = make_encoder(8);
        let frame = RawFrame::new_test(16, 2, PixelFormat::Uyvy422);
        let slices = enc.split_frame(&frame).expect("split should succeed");
        assert!(!slices.is_empty());
        let total: u32 = slices.iter().map(|s| s.row_count).sum();
        assert_eq!(total, 2);
    }

    #[test]
    fn test_split_returns_error_on_zero_height() {
        let enc = make_encoder(4);
        let frame = RawFrame::new(16, 0, PixelFormat::Uyvy422, vec![], 32, 0);
        assert!(enc.split_frame(&frame).is_err());
    }

    // --- FrameSliceEncoder: encode ---

    #[test]
    fn test_encode_produces_output() {
        let enc = make_encoder(4);
        let frame = RawFrame::new_test(320, 240, PixelFormat::Uyvy422);
        let result = enc.encode(&frame).expect("encode should succeed");
        assert!(!result.data.is_empty());
        assert_eq!(result.width, 320);
        assert_eq!(result.height, 240);
        assert_eq!(result.frame_number, 0);
    }

    #[test]
    fn test_encode_slice_offsets_are_monotone() {
        let enc = make_encoder(4);
        let frame = RawFrame::new_test(320, 240, PixelFormat::Uyvy422);
        let result = enc.encode(&frame).expect("encode should succeed");
        let offsets = &result.slice_offsets;
        for w in offsets.windows(2) {
            assert!(w[0] <= w[1], "offsets should be non-decreasing");
        }
    }

    #[test]
    fn test_encode_slice_data_accessible() {
        let enc = make_encoder(2);
        let frame = RawFrame::new_test(64, 64, PixelFormat::Uyvy422);
        let result = enc.encode(&frame).expect("encode should succeed");
        for i in 0..result.slice_count {
            assert!(
                result.slice_data(i).is_some(),
                "slice {} data should be accessible",
                i
            );
        }
    }

    #[test]
    fn test_encode_short_frame_returns_error() {
        let enc = make_encoder(2);
        // data is empty but height=10 → should fail
        let frame = RawFrame::new(64, 10, PixelFormat::Uyvy422, vec![], 128, 0);
        assert!(enc.encode(&frame).is_err());
    }

    #[test]
    fn test_encoded_frame_compressed_size() {
        let enc = make_encoder(4);
        let frame = RawFrame::new_test(320, 240, PixelFormat::Uyvy422);
        let result = enc.encode(&frame).expect("encode should succeed");
        assert_eq!(result.compressed_size(), result.data.len());
    }

    #[test]
    fn test_encoder_slice_count_accessor() {
        let enc = make_encoder(8);
        assert_eq!(enc.slice_count(), 8);
    }

    #[test]
    fn test_encoded_slice_compression_ratio_nonzero() {
        let enc = make_encoder(1);
        let frame = RawFrame::new_test(64, 64, PixelFormat::Uyvy422);
        let slices = enc.split_frame(&frame).expect("split should succeed");
        let encoded = enc.encode_slice(slices.into_iter().next().expect("at least one slice"));
        assert!(encoded.input_bytes > 0);
        assert!(encoded.compression_ratio() >= 0.0);
    }
}
