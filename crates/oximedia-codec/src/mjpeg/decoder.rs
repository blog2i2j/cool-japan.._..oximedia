//! MJPEG decoder implementation.
//!
//! Wraps the `oximedia-image` JPEG baseline decoder to decode
//! Motion JPEG video — each packet is an independently JPEG-encoded frame.

use super::types::MjpegFrameInfo;
use crate::error::{CodecError, CodecResult};
use crate::frame::{FrameType, Plane, VideoFrame};
use crate::traits::VideoDecoder;
use oximedia_core::{CodecId, PixelFormat, Rational, Timestamp};
use oximedia_image::jpeg::JpegDecoder;

/// MJPEG decoder.
///
/// Each incoming packet is an independent JPEG image. The decoder
/// wraps `oximedia-image`'s JPEG decoder and converts the output
/// to `VideoFrame` with planar YUV or packed RGB format.
///
/// # Architecture
///
/// - Receives compressed MJPEG packets (each is a standalone JPEG frame)
/// - Decodes via `oximedia-image::jpeg::JpegDecoder`
/// - Converts decoded RGB pixels to the requested output format
/// - Produces `VideoFrame` with correct timestamps
#[derive(Debug)]
pub struct MjpegDecoder {
    /// Expected frame width (0 = auto-detect from first frame).
    width: u32,
    /// Expected frame height (0 = auto-detect from first frame).
    height: u32,
    /// Output pixel format.
    output_format: PixelFormat,
    /// Internal JPEG decoder.
    jpeg_decoder: JpegDecoder,
    /// Pending decoded frames.
    output_queue: Vec<VideoFrame>,
    /// Whether the decoder has been flushed.
    flushed: bool,
    /// Number of frames decoded.
    frame_count: u64,
}

impl MjpegDecoder {
    /// Create a new MJPEG decoder.
    ///
    /// # Arguments
    ///
    /// * `width` - Expected frame width (0 for auto-detect)
    /// * `height` - Expected frame height (0 for auto-detect)
    #[must_use]
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            output_format: PixelFormat::Rgb24,
            jpeg_decoder: JpegDecoder::new(),
            output_queue: Vec::new(),
            flushed: false,
            frame_count: 0,
        }
    }

    /// Create a decoder with a specific output format.
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

    /// Probe an MJPEG packet to extract frame metadata without full decoding.
    ///
    /// # Errors
    ///
    /// Returns error if the data is not valid JPEG.
    pub fn probe_frame(data: &[u8]) -> CodecResult<MjpegFrameInfo> {
        if data.len() < 4 {
            return Err(CodecError::InvalidBitstream(
                "MJPEG data too short".to_string(),
            ));
        }

        // Check SOI
        if data[0] != 0xFF || data[1] != 0xD8 {
            return Err(CodecError::InvalidBitstream(
                "Not a JPEG frame (missing SOI)".to_string(),
            ));
        }

        // Check for AVI1 marker
        let marker_type = if Self::find_avi1_marker(data) {
            super::types::MjpegMarkerType::Avi1
        } else {
            super::types::MjpegMarkerType::Plain
        };

        // Find SOF0 to extract dimensions
        let (width, height) = Self::extract_dimensions(data)?;

        Ok(MjpegFrameInfo {
            width,
            height,
            marker_type,
            compressed_size: data.len(),
        })
    }

    /// Check if an AVI1 APP0 marker is present.
    fn find_avi1_marker(data: &[u8]) -> bool {
        // Search for "AVI1" string in APP0 segments
        data.windows(4).any(|w| w == b"AVI1")
    }

    /// Extract width and height from JPEG SOF0 segment.
    fn extract_dimensions(data: &[u8]) -> CodecResult<(u32, u32)> {
        let mut pos = 2; // Skip SOI

        while pos + 3 < data.len() {
            if data[pos] != 0xFF {
                pos += 1;
                continue;
            }

            let marker = data[pos + 1];
            pos += 2;

            match marker {
                0xC0 => {
                    // SOF0 — baseline DCT
                    if pos + 7 <= data.len() {
                        let _len = u16::from_be_bytes([data[pos], data[pos + 1]]);
                        let _precision = data[pos + 2];
                        let height = u16::from_be_bytes([data[pos + 3], data[pos + 4]]) as u32;
                        let width = u16::from_be_bytes([data[pos + 5], data[pos + 6]]) as u32;
                        return Ok((width, height));
                    }
                    return Err(CodecError::InvalidBitstream(
                        "SOF0 segment too short".to_string(),
                    ));
                }
                0xD9 => break, // EOI
                0xDA => break, // SOS — no SOF0 found before scan data
                0x00 | 0xFF => {
                    // Padding/stuffing
                    continue;
                }
                _ => {
                    // Skip segment
                    if pos + 1 < data.len() {
                        let seg_len = u16::from_be_bytes([data[pos], data[pos + 1]]) as usize;
                        pos += seg_len;
                    } else {
                        break;
                    }
                }
            }
        }

        Err(CodecError::InvalidBitstream(
            "No SOF0 marker found in JPEG data".to_string(),
        ))
    }

    /// Strip the AVI1 marker from MJPEG data to produce standard JPEG.
    ///
    /// Some JPEG decoders do not understand the AVI1 APP0 segment.
    /// This removes it while preserving the rest of the JPEG stream.
    fn strip_avi1_marker(data: &[u8]) -> Vec<u8> {
        let mut result = Vec::with_capacity(data.len());
        let mut pos = 0;

        // Copy SOI
        if data.len() >= 2 {
            result.extend_from_slice(&data[..2]);
            pos = 2;
        }

        while pos + 3 < data.len() {
            if data[pos] == 0xFF && data[pos + 1] == 0xE0 {
                // APP0 segment
                if pos + 3 < data.len() {
                    let seg_len = u16::from_be_bytes([data[pos + 2], data[pos + 3]]) as usize;
                    let seg_end = pos + 2 + seg_len;

                    // Check if this APP0 contains AVI1
                    let is_avi1 = pos + 6 < data.len() && data[pos + 4..].starts_with(b"AVI1");

                    if is_avi1 {
                        // Skip this segment entirely
                        pos = seg_end.min(data.len());
                        continue;
                    }
                }
            }

            // Not an AVI1 segment — copy the rest of the stream
            result.extend_from_slice(&data[pos..]);
            break;
        }

        if result.len() <= 2 && pos < data.len() {
            // Edge case: no APP0 segments found, copy everything after SOI
            result.extend_from_slice(&data[pos..]);
        }

        result
    }

    /// Decode a JPEG packet into a VideoFrame.
    fn decode_packet_inner(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        if data.len() < 4 {
            return Err(CodecError::InvalidBitstream(
                "MJPEG packet too short".to_string(),
            ));
        }

        // Strip AVI1 marker if present (decode as standard JPEG)
        let clean_data = Self::strip_avi1_marker(data);

        // Decode JPEG
        let jpeg_frame = self
            .jpeg_decoder
            .decode(&clean_data)
            .map_err(|e| CodecError::DecoderError(format!("MJPEG decode failed: {e}")))?;

        // Auto-detect dimensions from first frame
        if self.width == 0 {
            self.width = jpeg_frame.width;
        }
        if self.height == 0 {
            self.height = jpeg_frame.height;
        }

        // Convert JpegFrame (interleaved RGB) to VideoFrame
        let video_frame = self.jpeg_frame_to_video_frame(&jpeg_frame, pts)?;
        self.output_queue.push(video_frame);
        self.frame_count += 1;

        Ok(())
    }

    /// Convert a decoded JpegFrame to a VideoFrame in the requested output format.
    fn jpeg_frame_to_video_frame(
        &self,
        jpeg_frame: &oximedia_image::jpeg::JpegFrame,
        pts: i64,
    ) -> CodecResult<VideoFrame> {
        let width = jpeg_frame.width;
        let height = jpeg_frame.height;
        let pixels = &jpeg_frame.pixels;

        match self.output_format {
            PixelFormat::Rgb24 => {
                let mut frame = VideoFrame::new(PixelFormat::Rgb24, width, height);
                frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
                frame.frame_type = FrameType::Key;
                frame.planes = vec![Plane::with_dimensions(
                    pixels.clone(),
                    (width * 3) as usize,
                    width,
                    height,
                )];
                Ok(frame)
            }
            PixelFormat::Yuv420p => self.rgb_to_yuv420p(pixels, width, height, pts),
            PixelFormat::Yuv444p => self.rgb_to_yuv444p(pixels, width, height, pts),
            other => Err(CodecError::InvalidParameter(format!(
                "MJPEG decoder does not support output format: {other:?}"
            ))),
        }
    }

    /// Convert interleaved RGB to YUV 4:2:0 planar VideoFrame.
    fn rgb_to_yuv420p(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        pts: i64,
    ) -> CodecResult<VideoFrame> {
        let w = width as usize;
        let h = height as usize;
        let chroma_w = (w + 1) / 2;
        let chroma_h = (h + 1) / 2;

        let mut y_plane = vec![0u8; w * h];
        let mut u_plane = vec![128u8; chroma_w * chroma_h];
        let mut v_plane = vec![128u8; chroma_w * chroma_h];

        // First pass: compute all Y values
        for row in 0..h {
            for col in 0..w {
                let pix_idx = (row * w + col) * 3;
                if pix_idx + 2 < rgb.len() {
                    let r = rgb[pix_idx];
                    let g = rgb[pix_idx + 1];
                    let b = rgb[pix_idx + 2];
                    let (y, _cb, _cr) = oximedia_image::jpeg::rgb_to_ycbcr(r, g, b);
                    y_plane[row * w + col] = y.clamp(0.0, 255.0) as u8;
                }
            }
        }

        // Second pass: compute Cb/Cr for 2x2 blocks (average)
        for c_row in 0..chroma_h {
            for c_col in 0..chroma_w {
                let mut sum_cb = 0.0f32;
                let mut sum_cr = 0.0f32;
                let mut count = 0u32;

                for dy in 0..2 {
                    let row = c_row * 2 + dy;
                    if row >= h {
                        continue;
                    }
                    for dx in 0..2 {
                        let col = c_col * 2 + dx;
                        if col >= w {
                            continue;
                        }
                        let pix_idx = (row * w + col) * 3;
                        if pix_idx + 2 < rgb.len() {
                            let r = rgb[pix_idx];
                            let g = rgb[pix_idx + 1];
                            let b = rgb[pix_idx + 2];
                            let (_y, cb, cr) = oximedia_image::jpeg::rgb_to_ycbcr(r, g, b);
                            sum_cb += cb;
                            sum_cr += cr;
                            count += 1;
                        }
                    }
                }

                if count > 0 {
                    let c_idx = c_row * chroma_w + c_col;
                    u_plane[c_idx] = (sum_cb / count as f32).clamp(0.0, 255.0) as u8;
                    v_plane[c_idx] = (sum_cr / count as f32).clamp(0.0, 255.0) as u8;
                }
            }
        }

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;
        frame.planes = vec![
            Plane::with_dimensions(y_plane, w, width, height),
            Plane::with_dimensions(u_plane, chroma_w, (width + 1) / 2, (height + 1) / 2),
            Plane::with_dimensions(v_plane, chroma_w, (width + 1) / 2, (height + 1) / 2),
        ];
        Ok(frame)
    }

    /// Convert interleaved RGB to YUV 4:4:4 planar VideoFrame.
    fn rgb_to_yuv444p(
        &self,
        rgb: &[u8],
        width: u32,
        height: u32,
        pts: i64,
    ) -> CodecResult<VideoFrame> {
        let w = width as usize;
        let h = height as usize;
        let plane_size = w * h;

        let mut y_plane = vec![0u8; plane_size];
        let mut u_plane = vec![128u8; plane_size];
        let mut v_plane = vec![128u8; plane_size];

        for idx in 0..plane_size {
            let pix_idx = idx * 3;
            if pix_idx + 2 < rgb.len() {
                let r = rgb[pix_idx];
                let g = rgb[pix_idx + 1];
                let b = rgb[pix_idx + 2];
                let (y, cb, cr) = oximedia_image::jpeg::rgb_to_ycbcr(r, g, b);
                y_plane[idx] = y.clamp(0.0, 255.0) as u8;
                u_plane[idx] = cb.clamp(0.0, 255.0) as u8;
                v_plane[idx] = cr.clamp(0.0, 255.0) as u8;
            }
        }

        let mut frame = VideoFrame::new(PixelFormat::Yuv444p, width, height);
        frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;
        frame.planes = vec![
            Plane::with_dimensions(y_plane, w, width, height),
            Plane::with_dimensions(u_plane, w, width, height),
            Plane::with_dimensions(v_plane, w, width, height),
        ];
        Ok(frame)
    }
}

impl VideoDecoder for MjpegDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Mjpeg
    }

    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        if self.flushed {
            return Err(CodecError::InvalidParameter(
                "Cannot send packets after flush".to_string(),
            ));
        }
        self.decode_packet_inner(data, pts)
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
        // MJPEG has no buffered packets — each packet produces output immediately.
        Ok(())
    }

    fn reset(&mut self) {
        self.output_queue.clear();
        self.flushed = false;
        self.frame_count = 0;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::Plane;
    use crate::traits::VideoEncoder;

    /// Create a minimal valid JPEG for testing.
    /// We use the actual encoder to produce real JPEG data.
    fn make_test_jpeg(width: u32, height: u32) -> Vec<u8> {
        use super::super::encoder::MjpegEncoder;
        use super::super::types::MjpegConfig;

        let config = MjpegConfig::new(width, height)
            .expect("valid config")
            .with_quality(85)
            .with_pixel_format(PixelFormat::Rgb24);
        let mut encoder = MjpegEncoder::new(config).expect("valid encoder");

        let size = (width * height * 3) as usize;
        let data = vec![128u8; size];
        let mut frame = VideoFrame::new(PixelFormat::Rgb24, width, height);
        frame.planes = vec![Plane::with_dimensions(
            data,
            (width * 3) as usize,
            width,
            height,
        )];
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;

        encoder.send_frame(&frame).expect("encode failed");
        let pkt = encoder
            .receive_packet()
            .expect("receive failed")
            .expect("expected packet");
        pkt.data
    }

    #[test]
    fn test_decoder_creation() {
        let decoder = MjpegDecoder::new(640, 480);
        assert_eq!(decoder.codec(), CodecId::Mjpeg);
        assert_eq!(decoder.dimensions(), Some((640, 480)));
    }

    #[test]
    fn test_decoder_auto_detect_dimensions() {
        let decoder = MjpegDecoder::new(0, 0);
        assert_eq!(decoder.dimensions(), None);
    }

    #[test]
    fn test_decoder_output_format() {
        let decoder = MjpegDecoder::new(640, 480);
        assert_eq!(decoder.output_format(), Some(PixelFormat::Rgb24));

        let decoder = MjpegDecoder::new(640, 480).with_output_format(PixelFormat::Yuv420p);
        assert_eq!(decoder.output_format(), Some(PixelFormat::Yuv420p));
    }

    #[test]
    fn test_decode_jpeg_packet() {
        let jpeg_data = make_test_jpeg(16, 16);
        let mut decoder = MjpegDecoder::new(16, 16);

        let result = decoder.send_packet(&jpeg_data, 0);
        assert!(result.is_ok(), "decode failed: {result:?}");

        let frame = decoder
            .receive_frame()
            .expect("receive failed")
            .expect("expected frame");
        assert_eq!(frame.width, 16);
        assert_eq!(frame.height, 16);
        assert_eq!(frame.format, PixelFormat::Rgb24);
        assert_eq!(frame.frame_type, FrameType::Key);
        assert!(!frame.planes.is_empty());
    }

    #[test]
    fn test_decode_to_yuv420p() {
        let jpeg_data = make_test_jpeg(16, 16);
        let mut decoder = MjpegDecoder::new(16, 16).with_output_format(PixelFormat::Yuv420p);

        decoder.send_packet(&jpeg_data, 0).expect("decode failed");
        let frame = decoder
            .receive_frame()
            .expect("receive failed")
            .expect("expected frame");

        assert_eq!(frame.format, PixelFormat::Yuv420p);
        assert_eq!(frame.planes.len(), 3);
        // Y plane: full resolution
        assert_eq!(frame.planes[0].data.len(), 16 * 16);
        // Cb/Cr planes: half resolution
        assert_eq!(frame.planes[1].data.len(), 8 * 8);
        assert_eq!(frame.planes[2].data.len(), 8 * 8);
    }

    #[test]
    fn test_decode_multiple_packets() {
        let jpeg_data = make_test_jpeg(16, 16);
        let mut decoder = MjpegDecoder::new(16, 16);

        for i in 0..5 {
            decoder.send_packet(&jpeg_data, i).expect("decode failed");
            let frame = decoder
                .receive_frame()
                .expect("receive failed")
                .expect("expected frame");
            assert_eq!(frame.timestamp.pts, i);
        }

        assert_eq!(decoder.frame_count(), 5);
    }

    #[test]
    fn test_decode_invalid_data() {
        let mut decoder = MjpegDecoder::new(16, 16);
        let result = decoder.send_packet(&[0x00, 0x00, 0x00, 0x00], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_too_short() {
        let mut decoder = MjpegDecoder::new(16, 16);
        let result = decoder.send_packet(&[0xFF, 0xD8], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_flush_prevents_more_packets() {
        let mut decoder = MjpegDecoder::new(16, 16);
        decoder.flush().expect("flush should succeed");

        let jpeg_data = make_test_jpeg(16, 16);
        let result = decoder.send_packet(&jpeg_data, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_reset() {
        let mut decoder = MjpegDecoder::new(16, 16);
        decoder.flush().expect("flush");
        decoder.reset();

        // Should be able to decode again after reset
        let jpeg_data = make_test_jpeg(16, 16);
        let result = decoder.send_packet(&jpeg_data, 0);
        assert!(result.is_ok());
    }

    #[test]
    fn test_probe_frame() {
        let jpeg_data = make_test_jpeg(16, 16);
        let info = MjpegDecoder::probe_frame(&jpeg_data);
        assert!(info.is_ok(), "probe failed: {info:?}");

        let info = info.expect("probe result");
        assert_eq!(info.width, 16);
        assert_eq!(info.height, 16);
        assert_eq!(info.compressed_size, jpeg_data.len());
    }

    #[test]
    fn test_probe_invalid_data() {
        let result = MjpegDecoder::probe_frame(&[0x00, 0x01, 0x02, 0x03]);
        assert!(result.is_err());
    }

    #[test]
    fn test_probe_too_short() {
        let result = MjpegDecoder::probe_frame(&[0xFF]);
        assert!(result.is_err());
    }

    #[test]
    fn test_strip_avi1_marker() {
        // Build JPEG with AVI1
        let jpeg_data = make_test_jpeg(16, 16);
        // The encoder adds AVI1, so stripping should produce valid JPEG
        let stripped = MjpegDecoder::strip_avi1_marker(&jpeg_data);
        assert!(stripped.len() <= jpeg_data.len());
        // Should start with SOI
        assert_eq!(stripped[0], 0xFF);
        assert_eq!(stripped[1], 0xD8);
    }

    #[test]
    fn test_encoder_decoder_roundtrip() {
        use super::super::encoder::MjpegEncoder;
        use super::super::types::MjpegConfig;

        // Create encoder
        let config = MjpegConfig::new(16, 16)
            .expect("valid config")
            .with_quality(95) // High quality for better roundtrip fidelity
            .with_pixel_format(PixelFormat::Rgb24);
        let mut encoder = MjpegEncoder::new(config).expect("valid encoder");

        // Create a test frame with known pixel values
        let w = 16u32;
        let h = 16u32;
        let mut rgb_data = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                rgb_data[idx] = 200; // R
                rgb_data[idx + 1] = 100; // G
                rgb_data[idx + 2] = 50; // B
            }
        }

        let mut frame = VideoFrame::new(PixelFormat::Rgb24, w, h);
        frame.planes = vec![Plane::with_dimensions(
            rgb_data.clone(),
            (w * 3) as usize,
            w,
            h,
        )];
        frame.timestamp = Timestamp::new(42, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;

        // Encode
        encoder.send_frame(&frame).expect("encode failed");
        let pkt = encoder
            .receive_packet()
            .expect("receive failed")
            .expect("expected packet");

        // Decode
        let mut decoder = MjpegDecoder::new(w, h);
        decoder
            .send_packet(&pkt.data, pkt.pts)
            .expect("decode failed");
        let decoded = decoder
            .receive_frame()
            .expect("receive failed")
            .expect("expected frame");

        assert_eq!(decoded.width, w);
        assert_eq!(decoded.height, h);
        assert_eq!(decoded.format, PixelFormat::Rgb24);

        // JPEG is lossy, but uniform input should produce close output.
        // Check a few pixels are within tolerance (JPEG quantization error).
        let decoded_rgb = &decoded.planes[0].data;
        let tolerance = 15i32; // JPEG lossy tolerance
        for y in 2..14 {
            for x in 2..14 {
                let idx = ((y * w + x) * 3) as usize;
                if idx + 2 < decoded_rgb.len() {
                    let dr = (decoded_rgb[idx] as i32 - 200).abs();
                    let dg = (decoded_rgb[idx + 1] as i32 - 100).abs();
                    let db = (decoded_rgb[idx + 2] as i32 - 50).abs();
                    assert!(
                        dr <= tolerance && dg <= tolerance && db <= tolerance,
                        "Pixel ({x},{y}) too far: R={} G={} B={} (expected 200,100,50)",
                        decoded_rgb[idx],
                        decoded_rgb[idx + 1],
                        decoded_rgb[idx + 2]
                    );
                }
            }
        }
    }

    #[test]
    fn test_auto_detect_dimensions_on_decode() {
        let jpeg_data = make_test_jpeg(32, 24);
        let mut decoder = MjpegDecoder::new(0, 0);

        assert_eq!(decoder.dimensions(), None);
        decoder.send_packet(&jpeg_data, 0).expect("decode failed");
        // After decoding first frame, dimensions should be auto-detected
        assert_eq!(decoder.dimensions(), Some((32, 24)));
    }
}
