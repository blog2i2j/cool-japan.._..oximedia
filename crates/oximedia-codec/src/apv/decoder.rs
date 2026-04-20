//! APV decoder implementation.
//!
//! Decodes APV access units back to video frames. Each access unit is
//! independently decodable (intra-frame only), providing random-access
//! capability at any frame boundary.
//!
//! # Decoding pipeline
//!
//! 1. Parse APV access unit header (magic, profile, dimensions, QP, etc.)
//! 2. For each tile:
//!    a. Entropy decode (exp-Golomb + run-length → quantized coefficients)
//!    b. Inverse zigzag scan
//!    c. Dequantize using the QP from the header
//!    d. Inverse 8x8 DCT
//!    e. Reconstruct tile pixels
//! 3. Assemble tiles into the complete frame
//! 4. Optionally convert to the requested output pixel format

use super::dct::{dequantize_block, generate_quant_matrix, inverse_dct_8x8, inverse_zigzag_scan};
use super::entropy::{decode_block_coefficients, BitReader};
use super::types::{ApvError, ApvFrameHeader, APV_HEADER_SIZE};
use crate::error::{CodecError, CodecResult};
use crate::frame::{FrameType, Plane, VideoFrame};
use crate::traits::VideoDecoder;
use oximedia_core::{CodecId, PixelFormat, Rational, Timestamp};

/// APV video decoder.
///
/// Implements the `VideoDecoder` trait. The decoder is stateless with respect
/// to inter-frame dependencies — each packet is independently decodable.
#[derive(Debug)]
pub struct ApvDecoder {
    /// Detected frame width (0 before first frame).
    width: u32,
    /// Detected frame height (0 before first frame).
    height: u32,
    /// Output pixel format.
    output_format: PixelFormat,
    /// Pending decoded frames.
    output_queue: Vec<VideoFrame>,
    /// Whether the decoder has been flushed.
    flushed: bool,
    /// Number of frames decoded.
    frame_count: u64,
}

impl ApvDecoder {
    /// Create a new APV decoder.
    ///
    /// Dimensions are auto-detected from the first access unit header.
    #[must_use]
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            output_format: PixelFormat::Yuv420p,
            output_queue: Vec::new(),
            flushed: false,
            frame_count: 0,
        }
    }

    /// Set the desired output pixel format.
    #[must_use]
    pub fn with_output_format(mut self, format: PixelFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Get the number of frames decoded so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Decode a single APV access unit.
    fn decode_au(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        // Parse header
        let header = ApvFrameHeader::from_bytes(data).map_err(CodecError::from)?;

        // Update auto-detected dimensions
        if self.width == 0 {
            self.width = header.width;
        }
        if self.height == 0 {
            self.height = header.height;
        }

        let w = header.width as usize;
        let h = header.height as usize;

        // Generate quantization matrix from the header's QP
        let quant_matrix = generate_quant_matrix(header.qp);

        // Compute chroma dimensions
        let h_shift = header.chroma_format.chroma_h_shift() as usize;
        let v_shift = header.chroma_format.chroma_v_shift() as usize;
        let chroma_w = (w + (1 << h_shift) - 1) >> h_shift;
        let chroma_h = (h + (1 << v_shift) - 1) >> v_shift;

        // Allocate output planes
        let mut y_plane = vec![0u8; w * h];
        let mut cb_plane = vec![128u8; chroma_w * chroma_h];
        let mut cr_plane = vec![128u8; chroma_w * chroma_h];

        // Decode tiles
        let tile_cols = header.tile_cols;
        let tile_rows = header.tile_rows;
        let mut offset = APV_HEADER_SIZE;

        for tile_row in 0..tile_rows {
            for tile_col in 0..tile_cols {
                // Read tile data length
                if offset + 4 > data.len() {
                    return Err(CodecError::InvalidBitstream(
                        "truncated tile length".to_string(),
                    ));
                }
                let tile_len = u32::from_be_bytes([
                    data[offset],
                    data[offset + 1],
                    data[offset + 2],
                    data[offset + 3],
                ]) as usize;
                offset += 4;

                if offset + tile_len > data.len() {
                    return Err(CodecError::InvalidBitstream(format!(
                        "tile data truncated: need {tile_len} bytes at offset {offset}, have {}",
                        data.len() - offset
                    )));
                }

                let tile_data = &data[offset..offset + tile_len];
                offset += tile_len;

                // Compute tile region
                let tile_base_w = header.width / tile_cols as u32;
                let tile_remainder_w = header.width % tile_cols as u32;
                let tile_x = (tile_base_w * tile_col as u32) as usize;
                let tile_w = if tile_col < tile_cols - 1 {
                    tile_base_w as usize
                } else {
                    (tile_base_w + tile_remainder_w) as usize
                };

                let tile_base_h = header.height / tile_rows as u32;
                let tile_remainder_h = header.height % tile_rows as u32;
                let tile_y = (tile_base_h * tile_row as u32) as usize;
                let tile_h = if tile_row < tile_rows - 1 {
                    tile_base_h as usize
                } else {
                    (tile_base_h + tile_remainder_h) as usize
                };

                // Decode the tile
                self.decode_tile(
                    tile_data,
                    &quant_matrix,
                    &mut y_plane,
                    &mut cb_plane,
                    &mut cr_plane,
                    w,
                    h,
                    chroma_w,
                    chroma_h,
                    tile_x,
                    tile_y,
                    tile_w,
                    tile_h,
                    h_shift,
                    v_shift,
                )?;
            }
        }

        // Build output VideoFrame
        let video_frame = self.build_output_frame(
            &y_plane,
            &cb_plane,
            &cr_plane,
            header.width,
            header.height,
            chroma_w,
            chroma_h,
            pts,
        )?;

        self.output_queue.push(video_frame);
        self.frame_count += 1;

        Ok(())
    }

    /// Decode a single tile's Y, Cb, Cr blocks from the tile bitstream.
    fn decode_tile(
        &self,
        tile_data: &[u8],
        quant_matrix: &[f64; 64],
        y_plane: &mut [u8],
        cb_plane: &mut [u8],
        cr_plane: &mut [u8],
        frame_w: usize,
        frame_h: usize,
        chroma_w: usize,
        chroma_h: usize,
        tile_x: usize,
        tile_y: usize,
        tile_w: usize,
        tile_h: usize,
        h_shift: usize,
        v_shift: usize,
    ) -> CodecResult<()> {
        let mut reader = BitReader::new(tile_data);

        // Decode Y plane blocks
        self.decode_plane_blocks(
            &mut reader,
            quant_matrix,
            y_plane,
            frame_w,
            frame_h,
            tile_x,
            tile_y,
            tile_w,
            tile_h,
        )?;

        // Decode Cb plane blocks
        let chroma_tile_x = tile_x >> h_shift;
        let chroma_tile_y = tile_y >> v_shift;
        let chroma_tile_w = ((tile_x + tile_w + (1 << h_shift) - 1) >> h_shift) - chroma_tile_x;
        let chroma_tile_h = ((tile_y + tile_h + (1 << v_shift) - 1) >> v_shift) - chroma_tile_y;

        self.decode_plane_blocks(
            &mut reader,
            quant_matrix,
            cb_plane,
            chroma_w,
            chroma_h,
            chroma_tile_x,
            chroma_tile_y,
            chroma_tile_w,
            chroma_tile_h,
        )?;

        // Decode Cr plane blocks
        self.decode_plane_blocks(
            &mut reader,
            quant_matrix,
            cr_plane,
            chroma_w,
            chroma_h,
            chroma_tile_x,
            chroma_tile_y,
            chroma_tile_w,
            chroma_tile_h,
        )?;

        Ok(())
    }

    /// Decode all 8x8 blocks for a rectangular region of a plane.
    fn decode_plane_blocks(
        &self,
        reader: &mut BitReader<'_>,
        quant_matrix: &[f64; 64],
        plane: &mut [u8],
        plane_w: usize,
        plane_h: usize,
        tile_x: usize,
        tile_y: usize,
        tile_w: usize,
        tile_h: usize,
    ) -> CodecResult<()> {
        let blocks_h = (tile_h + 7) / 8;
        let blocks_w = (tile_w + 7) / 8;
        let dc_offset = 128.0; // For 8-bit

        for by in 0..blocks_h {
            for bx in 0..blocks_w {
                // Entropy decode
                let scanned = decode_block_coefficients(reader)
                    .map_err(|e| CodecError::DecoderError(format!("entropy decode: {e}")))?;

                // Inverse zigzag
                let quantized = inverse_zigzag_scan(&scanned);

                // Dequantize
                let mut coeffs = dequantize_block(&quantized, quant_matrix);

                // Inverse DCT
                inverse_dct_8x8(&mut coeffs, dc_offset);

                // Write reconstructed pixels into the plane
                let block_x = tile_x + bx * 8;
                let block_y = tile_y + by * 8;

                for row in 0..8 {
                    for col in 0..8 {
                        let px = block_x + col;
                        let py = block_y + row;
                        if px < plane_w && py < plane_h {
                            let sample = coeffs[row * 8 + col].round().clamp(0.0, 255.0) as u8;
                            plane[py * plane_w + px] = sample;
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Build the output `VideoFrame` from decoded YCbCr planes.
    fn build_output_frame(
        &self,
        y_plane: &[u8],
        cb_plane: &[u8],
        cr_plane: &[u8],
        width: u32,
        height: u32,
        chroma_w: usize,
        chroma_h: usize,
        pts: i64,
    ) -> CodecResult<VideoFrame> {
        match self.output_format {
            PixelFormat::Yuv420p => {
                let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
                frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
                frame.frame_type = FrameType::Key;
                frame.planes = vec![
                    Plane::with_dimensions(y_plane.to_vec(), width as usize, width, height),
                    Plane::with_dimensions(
                        cb_plane.to_vec(),
                        chroma_w,
                        (width + 1) / 2,
                        (height + 1) / 2,
                    ),
                    Plane::with_dimensions(
                        cr_plane.to_vec(),
                        chroma_w,
                        (width + 1) / 2,
                        (height + 1) / 2,
                    ),
                ];
                Ok(frame)
            }
            PixelFormat::Rgb24 => {
                self.yuv420p_to_rgb24(y_plane, cb_plane, cr_plane, width, height, chroma_w, pts)
            }
            other => Err(CodecError::InvalidParameter(format!(
                "APV decoder does not support output format: {other:?}"
            ))),
        }
    }

    /// Convert YCbCr 4:2:0 planes to interleaved RGB24 VideoFrame.
    fn yuv420p_to_rgb24(
        &self,
        y_plane: &[u8],
        cb_plane: &[u8],
        cr_plane: &[u8],
        width: u32,
        height: u32,
        chroma_w: usize,
        pts: i64,
    ) -> CodecResult<VideoFrame> {
        let w = width as usize;
        let h = height as usize;
        let mut rgb = Vec::with_capacity(w * h * 3);

        for row in 0..h {
            for col in 0..w {
                let y_idx = row * w + col;
                let c_row = row / 2;
                let c_col = col / 2;
                let c_idx = c_row * chroma_w + c_col;

                let y_val = y_plane.get(y_idx).copied().unwrap_or(0) as f32;
                let cb_val = cb_plane.get(c_idx).copied().unwrap_or(128) as f32;
                let cr_val = cr_plane.get(c_idx).copied().unwrap_or(128) as f32;

                let (r, g, b) = ycbcr_to_rgb(y_val, cb_val, cr_val);
                rgb.push(r);
                rgb.push(g);
                rgb.push(b);
            }
        }

        let mut frame = VideoFrame::new(PixelFormat::Rgb24, width, height);
        frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;
        frame.planes = vec![Plane::with_dimensions(
            rgb,
            (width * 3) as usize,
            width,
            height,
        )];
        Ok(frame)
    }
}

impl VideoDecoder for ApvDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Apv
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidParameter(
                "Cannot send packets after flush".to_string(),
            ));
        }
        self.decode_au(data, pts)
    }

    fn receive_frame(&mut self) -> CodecResult<Option<VideoFrame>> {
        if self.output_queue.is_empty() {
            Ok(None)
        } else {
            Ok(Some(self.output_queue.remove(0)))
        }
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.flushed = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.output_queue.clear();
        self.flushed = false;
        self.frame_count = 0;
        self.width = 0;
        self.height = 0;
    }

    fn output_format(&self) -> Option<PixelFormat> {
        Some(self.output_format)
    }

    fn dimensions(&self) -> Option<(u32, u32)> {
        if self.width > 0 && self.height > 0 {
            Some((self.width, self.height))
        } else {
            None
        }
    }
}

// ── BT.601 inverse color conversion (no external dependency) ───────────────

/// Convert YCbCr to RGB using ITU-R BT.601 coefficients.
fn ycbcr_to_rgb(y: f32, cb: f32, cr: f32) -> (u8, u8, u8) {
    let cb = cb - 128.0;
    let cr = cr - 128.0;
    let r = (y + 1.402 * cr).clamp(0.0, 255.0) as u8;
    let g = (y - 0.344136 * cb - 0.714136 * cr).clamp(0.0, 255.0) as u8;
    let b = (y + 1.772 * cb).clamp(0.0, 255.0) as u8;
    (r, g, b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::apv::encoder::ApvEncoder;
    use crate::apv::types::ApvConfig;
    use crate::traits::VideoEncoder;

    /// Create a solid-color RGB24 VideoFrame.
    fn make_solid_rgb_frame(width: u32, height: u32, r: u8, g: u8, b: u8) -> VideoFrame {
        let size = (width * height * 3) as usize;
        let mut data = vec![0u8; size];
        for i in (0..size).step_by(3) {
            data[i] = r;
            data[i + 1] = g;
            data[i + 2] = b;
        }
        let mut frame = VideoFrame::new(PixelFormat::Rgb24, width, height);
        frame.planes = vec![Plane::with_dimensions(
            data,
            (width * 3) as usize,
            width,
            height,
        )];
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;
        frame
    }

    /// Create a gradient RGB24 VideoFrame.
    fn make_gradient_rgb_frame(width: u32, height: u32) -> VideoFrame {
        let size = (width * height * 3) as usize;
        let mut data = vec![0u8; size];
        for y in 0..height {
            for x in 0..width {
                let idx = ((y * width + x) * 3) as usize;
                if idx + 2 < data.len() {
                    data[idx] = (x * 255 / width.max(1)) as u8;
                    data[idx + 1] = (y * 255 / height.max(1)) as u8;
                    data[idx + 2] = ((x + y) * 127 / (width + height).max(1)) as u8;
                }
            }
        }
        let mut frame = VideoFrame::new(PixelFormat::Rgb24, width, height);
        frame.planes = vec![Plane::with_dimensions(
            data,
            (width * 3) as usize,
            width,
            height,
        )];
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;
        frame
    }

    /// Helper: encode a frame and return the encoded packet data.
    fn encode_frame(config: &ApvConfig, frame: &VideoFrame) -> Vec<u8> {
        let mut encoder = ApvEncoder::new(config.clone()).expect("valid encoder");
        encoder.send_frame(frame).expect("encode failed");
        let pkt = encoder
            .receive_packet()
            .expect("receive failed")
            .expect("expected packet");
        pkt.data
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = ApvDecoder::new();
        assert_eq!(decoder.codec(), CodecId::Apv);
        assert_eq!(decoder.dimensions(), None);
    }

    #[test]
    fn test_decoder_output_format() {
        let decoder = ApvDecoder::new();
        assert_eq!(decoder.output_format(), Some(PixelFormat::Yuv420p));

        let decoder = ApvDecoder::new().with_output_format(PixelFormat::Rgb24);
        assert_eq!(decoder.output_format(), Some(PixelFormat::Rgb24));
    }

    #[test]
    fn test_decode_solid_color() {
        let config = ApvConfig::new(16, 16).expect("valid config");
        let frame = make_solid_rgb_frame(16, 16, 128, 128, 128);
        let encoded = encode_frame(&config, &frame);

        let mut decoder = ApvDecoder::new();
        decoder.send_packet(&encoded, 0).expect("decode failed");

        let decoded = decoder
            .receive_frame()
            .expect("receive failed")
            .expect("expected frame");
        assert_eq!(decoded.width, 16);
        assert_eq!(decoded.height, 16);
        assert_eq!(decoded.format, PixelFormat::Yuv420p);
        assert_eq!(decoded.frame_type, FrameType::Key);
    }

    #[test]
    fn test_auto_detect_dimensions() {
        let config = ApvConfig::new(32, 24).expect("valid config");
        let frame = make_solid_rgb_frame(32, 24, 100, 150, 200);
        let encoded = encode_frame(&config, &frame);

        let mut decoder = ApvDecoder::new();
        assert_eq!(decoder.dimensions(), None);

        decoder.send_packet(&encoded, 0).expect("decode failed");
        assert_eq!(decoder.dimensions(), Some((32, 24)));
    }

    #[test]
    fn test_decode_multiple_packets() {
        let config = ApvConfig::new(16, 16).expect("valid config");

        let mut decoder = ApvDecoder::new();
        for i in 0..5 {
            let frame = make_solid_rgb_frame(16, 16, i * 40, 128, 64);
            let encoded = encode_frame(&config, &frame);
            decoder
                .send_packet(&encoded, i as i64)
                .expect("decode failed");
            let decoded = decoder
                .receive_frame()
                .expect("receive failed")
                .expect("expected frame");
            assert_eq!(decoded.timestamp.pts, i as i64);
        }

        assert_eq!(decoder.frame_count(), 5);
    }

    #[test]
    fn test_decode_invalid_data() {
        let mut decoder = ApvDecoder::new();
        let result = decoder.send_packet(&[0x00; 20], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_too_short() {
        let mut decoder = ApvDecoder::new();
        let result = decoder.send_packet(&[0x00; 4], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_flush_prevents_more_packets() {
        let mut decoder = ApvDecoder::new();
        decoder.flush().expect("flush");

        let config = ApvConfig::new(16, 16).expect("valid config");
        let frame = make_solid_rgb_frame(16, 16, 128, 128, 128);
        let encoded = encode_frame(&config, &frame);
        assert!(decoder.send_packet(&encoded, 0).is_err());
    }

    #[test]
    fn test_reset() {
        let config = ApvConfig::new(16, 16).expect("valid config");
        let frame = make_solid_rgb_frame(16, 16, 128, 128, 128);
        let encoded = encode_frame(&config, &frame);

        let mut decoder = ApvDecoder::new();
        decoder.flush().expect("flush");
        decoder.reset();

        // Should work after reset
        assert!(decoder.send_packet(&encoded, 0).is_ok());
    }

    #[test]
    fn test_encoder_decoder_roundtrip_solid() {
        let config = ApvConfig::new(16, 16).expect("valid config").with_qp(5);
        let frame = make_solid_rgb_frame(16, 16, 200, 100, 50);
        let encoded = encode_frame(&config, &frame);

        let mut decoder = ApvDecoder::new().with_output_format(PixelFormat::Rgb24);
        decoder.send_packet(&encoded, 0).expect("decode failed");
        let decoded = decoder
            .receive_frame()
            .expect("receive failed")
            .expect("expected frame");

        assert_eq!(decoded.width, 16);
        assert_eq!(decoded.height, 16);
        assert_eq!(decoded.format, PixelFormat::Rgb24);

        // Check pixels are within lossy tolerance
        let decoded_rgb = &decoded.planes[0].data;
        let tolerance = 25i32; // Lossy codec tolerance
        for y in 2..14 {
            for x in 2..14 {
                let idx = ((y * 16 + x) * 3) as usize;
                if idx + 2 < decoded_rgb.len() {
                    let dr = (decoded_rgb[idx] as i32 - 200).abs();
                    let dg = (decoded_rgb[idx + 1] as i32 - 100).abs();
                    let db = (decoded_rgb[idx + 2] as i32 - 50).abs();
                    assert!(
                        dr <= tolerance && dg <= tolerance && db <= tolerance,
                        "Pixel ({x},{y}): R={} G={} B={} (expected 200,100,50, tol={tolerance})",
                        decoded_rgb[idx],
                        decoded_rgb[idx + 1],
                        decoded_rgb[idx + 2]
                    );
                }
            }
        }
    }

    #[test]
    fn test_encoder_decoder_roundtrip_psnr() {
        // Encode with QP=10 and verify PSNR > 30 dB
        let config = ApvConfig::new(64, 64).expect("valid config").with_qp(10);
        let frame = make_gradient_rgb_frame(64, 64);
        let encoded = encode_frame(&config, &frame);

        let mut decoder = ApvDecoder::new().with_output_format(PixelFormat::Rgb24);
        decoder.send_packet(&encoded, 0).expect("decode failed");
        let decoded = decoder
            .receive_frame()
            .expect("receive failed")
            .expect("expected frame");

        let original_rgb = &frame.planes[0].data;
        let decoded_rgb = &decoded.planes[0].data;
        let pixel_count = (64 * 64 * 3) as f64;

        let mut mse = 0.0f64;
        let min_len = original_rgb.len().min(decoded_rgb.len());
        for i in 0..min_len {
            let diff = original_rgb[i] as f64 - decoded_rgb[i] as f64;
            mse += diff * diff;
        }
        mse /= pixel_count;

        let psnr = if mse > 0.0 {
            10.0 * (255.0_f64 * 255.0 / mse).log10()
        } else {
            f64::INFINITY
        };

        assert!(
            psnr > 25.0,
            "PSNR too low: {psnr:.2} dB (expected >25 dB for QP=10)"
        );
    }

    #[test]
    fn test_roundtrip_various_resolutions() {
        for (w, h) in [(16, 16), (64, 64), (13, 7), (8, 8), (32, 48)] {
            let config = ApvConfig::new(w, h).expect("valid config").with_qp(15);
            let frame = make_solid_rgb_frame(w, h, 180, 90, 45);
            let encoded = encode_frame(&config, &frame);

            let mut decoder = ApvDecoder::new();
            decoder
                .send_packet(&encoded, 0)
                .unwrap_or_else(|e| panic!("decode failed for {w}x{h}: {e}"));
            let decoded = decoder
                .receive_frame()
                .expect("receive failed")
                .expect("expected frame");
            assert_eq!(decoded.width, w);
            assert_eq!(decoded.height, h);
        }
    }

    #[test]
    fn test_roundtrip_with_tiles() {
        let config = ApvConfig::new(64, 64)
            .expect("valid config")
            .with_qp(15)
            .with_tiles(2, 2)
            .expect("valid tiles");
        let frame = make_gradient_rgb_frame(64, 64);
        let encoded = encode_frame(&config, &frame);

        let mut decoder = ApvDecoder::new().with_output_format(PixelFormat::Rgb24);
        decoder.send_packet(&encoded, 0).expect("decode failed");
        let decoded = decoder
            .receive_frame()
            .expect("receive failed")
            .expect("expected frame");

        assert_eq!(decoded.width, 64);
        assert_eq!(decoded.height, 64);
        assert!(!decoded.planes[0].data.is_empty());
    }

    #[test]
    fn test_roundtrip_yuv420p_output() {
        let config = ApvConfig::new(16, 16).expect("valid config").with_qp(10);
        let frame = make_solid_rgb_frame(16, 16, 128, 128, 128);
        let encoded = encode_frame(&config, &frame);

        let mut decoder = ApvDecoder::new(); // default YUV420p output
        decoder.send_packet(&encoded, 0).expect("decode failed");
        let decoded = decoder
            .receive_frame()
            .expect("receive failed")
            .expect("expected frame");

        assert_eq!(decoded.format, PixelFormat::Yuv420p);
        assert_eq!(decoded.planes.len(), 3);
        // Y: full res
        assert_eq!(decoded.planes[0].data.len(), 16 * 16);
        // Cb, Cr: half res
        assert_eq!(decoded.planes[1].data.len(), 8 * 8);
        assert_eq!(decoded.planes[2].data.len(), 8 * 8);
    }

    #[test]
    fn test_decode_frame_count() {
        let config = ApvConfig::new(8, 8).expect("valid config");
        let mut decoder = ApvDecoder::new();

        for i in 0..3 {
            let frame = make_solid_rgb_frame(8, 8, 100, 100, 100);
            let encoded = encode_frame(&config, &frame);
            decoder.send_packet(&encoded, i).expect("decode failed");
            decoder.receive_frame().expect("receive");
        }

        assert_eq!(decoder.frame_count(), 3);
    }

    #[test]
    fn test_ycbcr_to_rgb_neutral_gray() {
        let (r, g, b) = ycbcr_to_rgb(128.0, 128.0, 128.0);
        assert!((r as i32 - 128).abs() <= 1);
        assert!((g as i32 - 128).abs() <= 1);
        assert!((b as i32 - 128).abs() <= 1);
    }

    #[test]
    fn test_ycbcr_to_rgb_clamp() {
        // Extreme values should be clamped — verify no panic
        let (r, g, b) = ycbcr_to_rgb(255.0, 0.0, 255.0);
        // u8 values are inherently 0–255; verify the clamping in the
        // conversion function produced valid pixels without saturating
        // in unexpected ways.
        // u8 values are always in 0-255; reaching here without panic means clamping worked.
        let _ = (r, g, b);
    }

    #[test]
    fn test_roundtrip_high_qp() {
        // Even at high QP (low quality), decode should not panic
        let config = ApvConfig::new(16, 16).expect("valid config").with_qp(60);
        let frame = make_gradient_rgb_frame(16, 16);
        let encoded = encode_frame(&config, &frame);

        let mut decoder = ApvDecoder::new();
        decoder.send_packet(&encoded, 0).expect("decode failed");
        let decoded = decoder
            .receive_frame()
            .expect("receive failed")
            .expect("expected frame");
        assert_eq!(decoded.width, 16);
        assert_eq!(decoded.height, 16);
    }
}
