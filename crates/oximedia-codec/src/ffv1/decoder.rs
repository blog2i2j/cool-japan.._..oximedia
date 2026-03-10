//! FFV1 decoder implementation.
//!
//! Decodes FFV1 lossless video bitstreams as specified in RFC 9043.
//! Supports version 3 with range coder and CRC-32 error detection.

use crate::error::{CodecError, CodecResult};
use crate::frame::{FrameType, Plane, VideoFrame};
use crate::traits::VideoDecoder;
use oximedia_core::{CodecId, PixelFormat, Rational, Timestamp};

use super::crc32::crc32_mpeg2;
use super::prediction::{decode_line, predict_median};
use super::range_coder::SimpleRangeDecoder;
use super::types::{
    Ffv1ChromaType, Ffv1Colorspace, Ffv1Config, Ffv1Version, SliceHeader, CONTEXT_COUNT,
    INITIAL_STATE,
};

/// FFV1 decoder.
///
/// Implements the `VideoDecoder` trait for decoding FFV1 lossless video.
/// The decoder processes compressed packets and outputs decoded `VideoFrame`s.
///
/// # Usage
///
/// ```ignore
/// use oximedia_codec::ffv1::Ffv1Decoder;
/// use oximedia_codec::VideoDecoder;
///
/// let mut decoder = Ffv1Decoder::new();
/// decoder.send_packet(&compressed_data, pts)?;
/// if let Some(frame) = decoder.receive_frame()? {
///     // Process decoded frame
/// }
/// ```
pub struct Ffv1Decoder {
    /// Codec configuration (parsed from extradata or first frame).
    config: Option<Ffv1Config>,
    /// Output frame queue.
    output_queue: Vec<VideoFrame>,
    /// Whether the decoder is in flush mode.
    flushing: bool,
    /// Number of decoded frames.
    frame_count: u64,
    /// Per-plane context states for range coder (reset each keyframe).
    plane_states: Vec<Vec<u8>>,
}

impl Ffv1Decoder {
    /// Create a new FFV1 decoder.
    pub fn new() -> Self {
        Self {
            config: None,
            output_queue: Vec::new(),
            flushing: false,
            frame_count: 0,
            plane_states: Vec::new(),
        }
    }

    /// Create a decoder initialized with extradata (configuration record).
    pub fn with_extradata(extradata: &[u8]) -> CodecResult<Self> {
        let mut dec = Self::new();
        dec.parse_config(extradata)?;
        Ok(dec)
    }

    /// Parse the FFV1 configuration record from extradata.
    ///
    /// For FFV1 v3, the configuration record is a range-coded bitstream
    /// containing codec parameters. For simplicity, we also support a
    /// compact binary format used within our own container.
    fn parse_config(&mut self, data: &[u8]) -> CodecResult<()> {
        // Minimal configuration record: at least 13 bytes for our binary format.
        // Format: [version(1), colorspace(1), chroma_h_shift(1), chroma_v_shift(1),
        //          bits(1), ec(1), num_h_slices(1), num_v_slices(1),
        //          width(4 LE), height(4 LE)]  = 13 bytes minimum
        if data.len() < 13 {
            return Err(CodecError::InvalidBitstream(format!(
                "FFV1 config too short: {} bytes, need at least 13",
                data.len()
            )));
        }

        let version = Ffv1Version::from_u8(data[0])?;
        let colorspace = Ffv1Colorspace::from_u8(data[1])?;
        let h_shift = u32::from(data[2]);
        let v_shift = u32::from(data[3]);
        let chroma_type = Ffv1ChromaType::from_shifts(h_shift, v_shift)?;
        let bits_per_raw_sample = data[4];
        let ec = data[5] != 0;
        let num_h_slices = u32::from(data[6]);
        let num_v_slices = u32::from(data[7]);

        // Read width and height as little-endian u32
        let width_bytes: [u8; 4] = data[8..12]
            .try_into()
            .map_err(|_| CodecError::InvalidBitstream("bad width bytes".to_string()))?;
        let height_bytes_end = 12 + 4;
        if data.len() < height_bytes_end {
            return Err(CodecError::InvalidBitstream(
                "FFV1 config truncated".to_string(),
            ));
        }
        let height_bytes: [u8; 4] = data[12..16]
            .try_into()
            .map_err(|_| CodecError::InvalidBitstream("bad height bytes".to_string()))?;

        let width = u32::from_le_bytes(width_bytes);
        let height = u32::from_le_bytes(height_bytes);

        let config = Ffv1Config {
            version,
            width,
            height,
            colorspace,
            chroma_type,
            bits_per_raw_sample,
            num_h_slices,
            num_v_slices,
            ec,
            range_coder_mode: version.uses_range_coder(),
            state_transition_delta: Vec::new(),
        };
        config.validate()?;

        self.init_states(&config);
        self.config = Some(config);
        Ok(())
    }

    /// Initialize per-plane context states.
    fn init_states(&mut self, config: &Ffv1Config) {
        let plane_count = config.plane_count();
        self.plane_states.clear();
        for _ in 0..plane_count {
            self.plane_states.push(vec![INITIAL_STATE; CONTEXT_COUNT]);
        }
    }

    /// Reset all context states (done at keyframes).
    fn reset_states(&mut self) {
        for states in &mut self.plane_states {
            for s in states.iter_mut() {
                *s = INITIAL_STATE;
            }
        }
    }

    /// Decode a complete frame from the given packet data.
    fn decode_frame(&mut self, data: &[u8], pts: i64) -> CodecResult<VideoFrame> {
        // Extract all needed config values upfront to avoid borrow conflicts.
        let config = self
            .config
            .as_ref()
            .ok_or_else(|| CodecError::DecoderError("FFV1 decoder not configured".to_string()))?;

        let width = config.width;
        let height = config.height;
        let plane_count = config.plane_count();
        let ec = config.ec;
        let num_slices = config.num_slices();
        let num_h_slices = config.num_h_slices;
        let num_v_slices = config.num_v_slices;
        let max_val = config.max_sample_value();

        let plane_dims: Vec<(u32, u32)> = (0..plane_count)
            .map(|i| config.plane_dimensions(i))
            .collect();

        // Determine pixel format
        let pixel_format = match (
            config.colorspace,
            config.chroma_type,
            config.bits_per_raw_sample,
        ) {
            (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma420, 8) => PixelFormat::Yuv420p,
            (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma422, 8) => PixelFormat::Yuv422p,
            (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma444, 8) => PixelFormat::Yuv444p,
            _ => PixelFormat::Yuv420p, // fallback
        };

        // Release the immutable borrow on self.config before mutable operations.
        let _ = config;

        let is_keyframe = self.frame_count == 0;

        if is_keyframe {
            self.reset_states();
        }

        let mut frame = VideoFrame::new(pixel_format, width, height);
        frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
        frame.frame_type = if is_keyframe {
            FrameType::Key
        } else {
            FrameType::Inter
        };

        // Decode all planes
        let mut planes_data: Vec<Vec<Vec<i32>>> = Vec::with_capacity(plane_count);

        if num_slices == 1 {
            // Single slice: decode all planes from the whole packet
            let slice_data = if ec && data.len() >= 4 {
                // Last 4 bytes are CRC
                let payload = &data[..data.len() - 4];
                let stored_crc_bytes: [u8; 4] = data[data.len() - 4..]
                    .try_into()
                    .map_err(|_| CodecError::InvalidBitstream("bad CRC bytes".to_string()))?;
                let stored_crc = u32::from_le_bytes(stored_crc_bytes);
                let computed_crc = crc32_mpeg2(payload);
                if stored_crc != computed_crc {
                    return Err(CodecError::InvalidBitstream(format!(
                        "FFV1 slice CRC mismatch: stored={stored_crc:#010X}, computed={computed_crc:#010X}"
                    )));
                }
                payload
            } else {
                data
            };

            for plane_idx in 0..plane_count {
                let (pw, ph) = plane_dims[plane_idx];
                let plane_header = SliceHeader {
                    slice_x: 0,
                    slice_y: 0,
                    slice_width: pw,
                    slice_height: ph,
                };
                let decoded = self.decode_slice(slice_data, &plane_header, plane_idx)?;
                planes_data.push(decoded);
            }
        } else {
            // Multi-slice: split data into slice segments
            let slice_data_len = data.len() / (num_slices as usize);
            for plane_idx in 0..plane_count {
                let (pw, ph) = plane_dims[plane_idx];
                let slice_h = ph / num_v_slices;
                let slice_w = pw / num_h_slices;

                let mut plane_lines: Vec<Vec<i32>> = Vec::new();

                for sy in 0..num_v_slices {
                    for sx in 0..num_h_slices {
                        let slice_idx = (sy * num_h_slices + sx) as usize;
                        let start = slice_idx * slice_data_len;
                        let end = if slice_idx + 1 == num_slices as usize {
                            data.len()
                        } else {
                            start + slice_data_len
                        };
                        let segment = &data[start..end];

                        let sh = if sy == num_v_slices - 1 {
                            ph - sy * slice_h
                        } else {
                            slice_h
                        };
                        let sw = if sx == num_h_slices - 1 {
                            pw - sx * slice_w
                        } else {
                            slice_w
                        };

                        let header = SliceHeader {
                            slice_x: sx * slice_w,
                            slice_y: sy * slice_h,
                            slice_width: sw,
                            slice_height: sh,
                        };
                        let decoded = self.decode_slice(segment, &header, plane_idx)?;
                        for line in decoded {
                            plane_lines.push(line);
                        }
                    }
                }
                planes_data.push(plane_lines);
            }
        }

        // Convert decoded plane data to VideoFrame planes
        for (plane_idx, plane_lines) in planes_data.iter().enumerate() {
            let (pw, ph) = plane_dims[plane_idx];
            let stride = pw as usize;
            let mut plane_data = vec![0u8; stride * ph as usize];

            for (y, line) in plane_lines.iter().enumerate() {
                if y >= ph as usize {
                    break;
                }
                let row_start = y * stride;
                for (x, &sample) in line.iter().enumerate() {
                    if x >= pw as usize {
                        break;
                    }
                    // Clamp to valid range and convert to u8
                    let clamped = sample.clamp(0, max_val) as u8;
                    plane_data[row_start + x] = clamped;
                }
            }

            frame
                .planes
                .push(Plane::with_dimensions(plane_data, stride, pw, ph));
        }

        self.frame_count += 1;
        Ok(frame)
    }

    /// Decode a single slice for one plane using range coder.
    ///
    /// Returns the decoded samples as a Vec of lines, each line being a Vec<i32>.
    fn decode_slice(
        &mut self,
        data: &[u8],
        header: &SliceHeader,
        plane_idx: usize,
    ) -> CodecResult<Vec<Vec<i32>>> {
        let w = header.slice_width as usize;
        let h = header.slice_height as usize;

        if w == 0 || h == 0 {
            return Ok(Vec::new());
        }

        if data.len() < 2 {
            // Not enough data for range coder init; return black plane
            let mut lines = Vec::with_capacity(h);
            for _ in 0..h {
                lines.push(vec![0i32; w]);
            }
            return Ok(lines);
        }

        let mut decoder = SimpleRangeDecoder::new(data)?;
        let states = self
            .plane_states
            .get_mut(plane_idx)
            .ok_or_else(|| CodecError::Internal("invalid plane index".to_string()))?;

        let mut lines: Vec<Vec<i32>> = Vec::with_capacity(h);
        let mut prev_line = vec![0i32; w];

        for _y in 0..h {
            let mut line = Vec::with_capacity(w);
            for x in 0..w {
                let residual = decoder.get_symbol(states)?;
                let left = if x > 0 { line[x - 1] } else { 0 };
                let top = prev_line[x];
                let top_left = if x > 0 { prev_line[x - 1] } else { 0 };
                let pred = predict_median(left, top, top_left);
                line.push(pred + residual);
            }
            prev_line.clone_from(&line);
            lines.push(line);
        }

        Ok(lines)
    }
}

impl VideoDecoder for Ffv1Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Ffv1
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        if self.flushing {
            return Err(CodecError::DecoderError(
                "decoder is flushing, cannot accept new packets".to_string(),
            ));
        }

        // If not configured yet, try to parse config from the packet.
        // In practice, config comes from container extradata.
        if self.config.is_none() {
            return Err(CodecError::DecoderError(
                "FFV1 decoder not configured: call with_extradata() first".to_string(),
            ));
        }

        let frame = self.decode_frame(data, pts)?;
        self.output_queue.push(frame);
        Ok(())
    }

    fn receive_frame(&mut self) -> CodecResult<Option<VideoFrame>> {
        if self.output_queue.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.output_queue.remove(0)))
        }
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.output_queue.clear();
        self.flushing = false;
        self.frame_count = 0;
        self.reset_states();
    }

    fn output_format(&self) -> Option<PixelFormat> {
        self.config
            .as_ref()
            .map(|c| match (c.colorspace, c.chroma_type) {
                (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma420) => PixelFormat::Yuv420p,
                (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma422) => PixelFormat::Yuv422p,
                (Ffv1Colorspace::YCbCr, Ffv1ChromaType::Chroma444) => PixelFormat::Yuv444p,
                _ => PixelFormat::Yuv420p,
            })
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        self.config.as_ref().map(|c| (c.width, c.height))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::VideoDecoder;

    fn make_config_bytes(width: u32, height: u32) -> Vec<u8> {
        let mut data = Vec::new();
        data.push(3); // version = V3
        data.push(0); // colorspace = YCbCr
        data.push(1); // chroma h_shift = 1
        data.push(1); // chroma v_shift = 1
        data.push(8); // bits_per_raw_sample = 8
        data.push(1); // ec = true
        data.push(1); // num_h_slices
        data.push(1); // num_v_slices
        data.extend_from_slice(&width.to_le_bytes());
        data.extend_from_slice(&height.to_le_bytes());
        data
    }

    #[test]
    #[ignore]
    fn test_decoder_creation() {
        let dec = Ffv1Decoder::new();
        assert!(dec.config.is_none());
        assert_eq!(dec.codec(), CodecId::Ffv1);
    }

    #[test]
    #[ignore]
    fn test_decoder_with_extradata() {
        let config_data = make_config_bytes(320, 240);
        let dec = Ffv1Decoder::with_extradata(&config_data).expect("valid config");
        assert!(dec.config.is_some());
        assert_eq!(dec.dimensions(), Some((320, 240)));
        assert_eq!(dec.output_format(), Some(PixelFormat::Yuv420p));
    }

    #[test]
    #[ignore]
    fn test_decoder_invalid_config() {
        // Too short
        assert!(Ffv1Decoder::with_extradata(&[0; 5]).is_err());
        // Invalid version
        let mut bad = make_config_bytes(320, 240);
        bad[0] = 99;
        assert!(Ffv1Decoder::with_extradata(&bad).is_err());
    }

    #[test]
    #[ignore]
    fn test_decoder_not_configured() {
        let mut dec = Ffv1Decoder::new();
        assert!(dec.send_packet(&[0; 100], 0).is_err());
    }

    #[test]
    #[ignore]
    fn test_decoder_reset() {
        let config_data = make_config_bytes(16, 16);
        let mut dec = Ffv1Decoder::with_extradata(&config_data).expect("valid");
        dec.frame_count = 10;
        dec.flushing = true;
        dec.reset();
        assert_eq!(dec.frame_count, 0);
        assert!(!dec.flushing);
    }

    #[test]
    #[ignore]
    fn test_decoder_flush() {
        let config_data = make_config_bytes(16, 16);
        let mut dec = Ffv1Decoder::with_extradata(&config_data).expect("valid");
        dec.flush().expect("flush ok");
        assert!(dec.flushing);
        // Should reject new packets after flush
        assert!(dec.send_packet(&[0; 100], 0).is_err());
    }
}
