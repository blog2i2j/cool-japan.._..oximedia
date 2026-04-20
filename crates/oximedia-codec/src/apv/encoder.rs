//! APV encoder implementation.
//!
//! Encodes video frames using the APV-S (Simple Profile) intra-frame codec.
//! Each frame is independently encoded — every output packet is a keyframe
//! suitable for random access.
//!
//! # Encoding pipeline
//!
//! 1. Convert input pixel format to planar YCbCr (if not already)
//! 2. Divide frame into tiles (for parallel encoding potential)
//! 3. For each tile, for each 8x8 block:
//!    a. Forward 8x8 DCT (f64 precision)
//!    b. Quantize using QP-derived matrix
//!    c. Zigzag scan
//!    d. Run-length + exp-Golomb entropy encode
//! 4. Pack tiles into APV access unit with header

use super::dct::{forward_dct_8x8, generate_quant_matrix, quantize_block, zigzag_scan};
use super::entropy::{encode_block_coefficients, BitWriter};
use super::types::{ApvConfig, ApvError, ApvFrameHeader};
use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;
use crate::traits::{BitrateMode, EncodedPacket, EncoderConfig, EncoderPreset, VideoEncoder};
use oximedia_core::{CodecId, PixelFormat, Rational};

/// APV video encoder.
///
/// Implements the `VideoEncoder` trait for APV-S (Simple Profile) encoding.
/// Each input frame produces exactly one output packet (intra-only codec).
#[derive(Debug)]
pub struct ApvEncoder {
    /// APV-specific configuration.
    apv_config: ApvConfig,
    /// Unified encoder configuration (for `VideoEncoder::config()`).
    encoder_config: EncoderConfig,
    /// Pre-computed quantization matrix for the configured QP.
    quant_matrix: [f64; 64],
    /// Frame counter for PTS generation.
    frame_count: u64,
    /// Pending output packets.
    output_queue: Vec<EncodedPacket>,
    /// Whether the encoder is in flush mode.
    flushing: bool,
}

impl ApvEncoder {
    /// Create a new APV encoder with the given configuration.
    ///
    /// # Errors
    ///
    /// Returns `ApvError` if the configuration is invalid.
    pub fn new(config: ApvConfig) -> Result<Self, ApvError> {
        config.validate()?;

        let encoder_config = EncoderConfig {
            codec: CodecId::Apv,
            width: config.width,
            height: config.height,
            pixel_format: PixelFormat::Yuv420p,
            framerate: Rational::new(30, 1),
            bitrate: BitrateMode::Crf(config.qp as f32),
            preset: EncoderPreset::Medium,
            profile: Some(config.profile.name().to_string()),
            keyint: 1, // Every frame is a keyframe (intra-only)
            threads: 0,
            timebase: Rational::new(1, 1000),
        };

        let quant_matrix = generate_quant_matrix(config.qp);

        Ok(Self {
            apv_config: config,
            encoder_config,
            quant_matrix,
            frame_count: 0,
            output_queue: Vec::new(),
            flushing: false,
        })
    }

    /// Create an APV encoder with default settings for the given dimensions.
    ///
    /// # Errors
    ///
    /// Returns error if dimensions are invalid.
    pub fn with_dimensions(width: u32, height: u32) -> Result<Self, ApvError> {
        let config = ApvConfig::new(width, height)?;
        Self::new(config)
    }

    /// Get the current QP setting.
    #[must_use]
    pub fn qp(&self) -> u8 {
        self.apv_config.qp
    }

    /// Get the number of frames encoded so far.
    #[must_use]
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Get the APV-specific configuration.
    #[must_use]
    pub fn apv_config(&self) -> &ApvConfig {
        &self.apv_config
    }

    /// Encode a single frame into an APV access unit.
    fn encode_frame(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        // Extract planar Y, Cb, Cr data from the input frame
        let (y_plane, cb_plane, cr_plane, actual_w, actual_h) = self.extract_yuv_planes(frame)?;

        let w = actual_w as usize;
        let h = actual_h as usize;

        // Build the APV access unit
        let header = ApvFrameHeader::from_config(&self.apv_config);
        let header_bytes = header.to_bytes();

        // Estimate output size and create writer for tile data
        let estimated_size = w * h * 2; // Conservative estimate
        let mut tile_data_buf: Vec<u8> = Vec::with_capacity(estimated_size);

        // Encode all tiles
        let tile_cols = self.apv_config.tile_cols;
        let tile_rows = self.apv_config.tile_rows;

        for tile_row in 0..tile_rows {
            for tile_col in 0..tile_cols {
                let tile_x = self.apv_config.tile_x_offset(tile_col) as usize;
                let tile_y = self.apv_config.tile_y_offset(tile_row) as usize;
                let tile_w = self.apv_config.tile_width(tile_col) as usize;
                let tile_h = self.apv_config.tile_height(tile_row) as usize;

                let tile_bytes = self.encode_tile(
                    &y_plane, &cb_plane, &cr_plane, w, h, tile_x, tile_y, tile_w, tile_h,
                )?;

                // Write tile data length (4 bytes BE) followed by tile data
                let tile_len = tile_bytes.len() as u32;
                tile_data_buf.extend_from_slice(&tile_len.to_be_bytes());
                tile_data_buf.extend_from_slice(&tile_bytes);
            }
        }

        // Assemble the complete access unit: header + tile data
        let mut au = Vec::with_capacity(header_bytes.len() + tile_data_buf.len());
        au.extend_from_slice(&header_bytes);
        au.extend_from_slice(&tile_data_buf);

        let pts = self.frame_count as i64;
        self.output_queue.push(EncodedPacket {
            data: au,
            pts,
            dts: pts,
            keyframe: true, // APV is intra-only
            duration: Some(1),
        });

        self.frame_count += 1;
        Ok(())
    }

    /// Encode a single tile (all planes: Y, Cb, Cr).
    fn encode_tile(
        &self,
        y_plane: &[u8],
        cb_plane: &[u8],
        cr_plane: &[u8],
        frame_w: usize,
        frame_h: usize,
        tile_x: usize,
        tile_y: usize,
        tile_w: usize,
        tile_h: usize,
    ) -> CodecResult<Vec<u8>> {
        let mut writer = BitWriter::new(tile_w * tile_h * 2);

        // Encode Y plane blocks
        self.encode_plane_blocks(
            &mut writer,
            y_plane,
            frame_w,
            frame_h,
            tile_x,
            tile_y,
            tile_w,
            tile_h,
        )?;

        // Compute chroma tile dimensions based on chroma format
        let h_shift = self.apv_config.chroma_format.chroma_h_shift() as usize;
        let v_shift = self.apv_config.chroma_format.chroma_v_shift() as usize;

        let chroma_frame_w = (frame_w + (1 << h_shift) - 1) >> h_shift;
        let chroma_frame_h = (frame_h + (1 << v_shift) - 1) >> v_shift;
        let chroma_tile_x = tile_x >> h_shift;
        let chroma_tile_y = tile_y >> v_shift;
        let chroma_tile_w = ((tile_x + tile_w + (1 << h_shift) - 1) >> h_shift) - chroma_tile_x;
        let chroma_tile_h = ((tile_y + tile_h + (1 << v_shift) - 1) >> v_shift) - chroma_tile_y;

        // Encode Cb plane blocks
        self.encode_plane_blocks(
            &mut writer,
            cb_plane,
            chroma_frame_w,
            chroma_frame_h,
            chroma_tile_x,
            chroma_tile_y,
            chroma_tile_w,
            chroma_tile_h,
        )?;

        // Encode Cr plane blocks
        self.encode_plane_blocks(
            &mut writer,
            cr_plane,
            chroma_frame_w,
            chroma_frame_h,
            chroma_tile_x,
            chroma_tile_y,
            chroma_tile_w,
            chroma_tile_h,
        )?;

        Ok(writer.finish())
    }

    /// Encode all 8x8 blocks within a rectangular region of a single plane.
    fn encode_plane_blocks(
        &self,
        writer: &mut BitWriter,
        plane: &[u8],
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
                // Extract 8x8 block with edge replication for partial blocks
                let mut block = [0.0f64; 64];
                let block_x = tile_x + bx * 8;
                let block_y = tile_y + by * 8;

                for row in 0..8 {
                    for col in 0..8 {
                        let px = (block_x + col).min(plane_w.saturating_sub(1));
                        let py = (block_y + row).min(plane_h.saturating_sub(1));
                        let idx = py * plane_w + px;
                        block[row * 8 + col] = if idx < plane.len() {
                            plane[idx] as f64
                        } else {
                            dc_offset
                        };
                    }
                }

                // Forward DCT
                forward_dct_8x8(&mut block, dc_offset);

                // Quantize
                let quantized = quantize_block(&block, &self.quant_matrix);

                // Zigzag scan
                let scanned = zigzag_scan(&quantized);

                // Entropy encode
                encode_block_coefficients(writer, &scanned);
            }
        }

        Ok(())
    }

    /// Extract Y, Cb, Cr planes from the input VideoFrame.
    ///
    /// Handles conversion from RGB24, RGBA32, YUV420p, YUV422p, YUV444p.
    fn extract_yuv_planes(
        &self,
        frame: &VideoFrame,
    ) -> CodecResult<(Vec<u8>, Vec<u8>, Vec<u8>, u32, u32)> {
        let w = frame.width;
        let h = frame.height;

        match frame.format {
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
                if frame.planes.len() < 3 {
                    return Err(CodecError::InvalidParameter(format!(
                        "YUV frame requires 3 planes, got {}",
                        frame.planes.len()
                    )));
                }
                Ok((
                    frame.planes[0].data.clone(),
                    frame.planes[1].data.clone(),
                    frame.planes[2].data.clone(),
                    w,
                    h,
                ))
            }
            PixelFormat::Rgb24 => self.rgb24_to_yuv_planes(frame, w, h),
            PixelFormat::Rgba32 => self.rgba32_to_yuv_planes(frame, w, h),
            other => Err(CodecError::InvalidParameter(format!(
                "APV encoder does not support input format: {other:?}"
            ))),
        }
    }

    /// Convert RGB24 frame to YCbCr 4:2:0 planes.
    fn rgb24_to_yuv_planes(
        &self,
        frame: &VideoFrame,
        w: u32,
        h: u32,
    ) -> CodecResult<(Vec<u8>, Vec<u8>, Vec<u8>, u32, u32)> {
        if frame.planes.is_empty() {
            return Err(CodecError::InvalidParameter(
                "RGB24 frame has no planes".to_string(),
            ));
        }
        let rgb = &frame.planes[0].data;
        let wu = w as usize;
        let hu = h as usize;
        let chroma_w = (wu + 1) / 2;
        let chroma_h = (hu + 1) / 2;

        let mut y_plane = vec![0u8; wu * hu];
        let mut cb_plane = vec![128u8; chroma_w * chroma_h];
        let mut cr_plane = vec![128u8; chroma_w * chroma_h];

        // Y pass
        for row in 0..hu {
            for col in 0..wu {
                let pix = (row * wu + col) * 3;
                if pix + 2 < rgb.len() {
                    let (y, _cb, _cr) = rgb_to_ycbcr(rgb[pix], rgb[pix + 1], rgb[pix + 2]);
                    y_plane[row * wu + col] = y;
                }
            }
        }

        // Cb/Cr pass (4:2:0 averaging)
        for cr in 0..chroma_h {
            for cc in 0..chroma_w {
                let mut sum_cb = 0i32;
                let mut sum_cr = 0i32;
                let mut count = 0i32;
                for dy in 0..2 {
                    let row = cr * 2 + dy;
                    if row >= hu {
                        continue;
                    }
                    for dx in 0..2 {
                        let col = cc * 2 + dx;
                        if col >= wu {
                            continue;
                        }
                        let pix = (row * wu + col) * 3;
                        if pix + 2 < rgb.len() {
                            let (_y, cb, cri) = rgb_to_ycbcr(rgb[pix], rgb[pix + 1], rgb[pix + 2]);
                            sum_cb += cb as i32;
                            sum_cr += cri as i32;
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    let c_idx = cr * chroma_w + cc;
                    cb_plane[c_idx] = (sum_cb / count).clamp(0, 255) as u8;
                    cr_plane[c_idx] = (sum_cr / count).clamp(0, 255) as u8;
                }
            }
        }

        Ok((y_plane, cb_plane, cr_plane, w, h))
    }

    /// Convert RGBA32 frame to YCbCr 4:2:0 planes (alpha discarded).
    fn rgba32_to_yuv_planes(
        &self,
        frame: &VideoFrame,
        w: u32,
        h: u32,
    ) -> CodecResult<(Vec<u8>, Vec<u8>, Vec<u8>, u32, u32)> {
        if frame.planes.is_empty() {
            return Err(CodecError::InvalidParameter(
                "RGBA32 frame has no planes".to_string(),
            ));
        }
        let rgba = &frame.planes[0].data;
        let wu = w as usize;
        let hu = h as usize;
        let chroma_w = (wu + 1) / 2;
        let chroma_h = (hu + 1) / 2;

        let mut y_plane = vec![0u8; wu * hu];
        let mut cb_plane = vec![128u8; chroma_w * chroma_h];
        let mut cr_plane = vec![128u8; chroma_w * chroma_h];

        // Y pass
        for row in 0..hu {
            for col in 0..wu {
                let pix = (row * wu + col) * 4;
                if pix + 2 < rgba.len() {
                    let (y, _cb, _cr) = rgb_to_ycbcr(rgba[pix], rgba[pix + 1], rgba[pix + 2]);
                    y_plane[row * wu + col] = y;
                }
            }
        }

        // Cb/Cr pass (4:2:0 averaging)
        for cr in 0..chroma_h {
            for cc in 0..chroma_w {
                let mut sum_cb = 0i32;
                let mut sum_cr = 0i32;
                let mut count = 0i32;
                for dy in 0..2 {
                    let row = cr * 2 + dy;
                    if row >= hu {
                        continue;
                    }
                    for dx in 0..2 {
                        let col = cc * 2 + dx;
                        if col >= wu {
                            continue;
                        }
                        let pix = (row * wu + col) * 4;
                        if pix + 2 < rgba.len() {
                            let (_y, cb, cri) =
                                rgb_to_ycbcr(rgba[pix], rgba[pix + 1], rgba[pix + 2]);
                            sum_cb += cb as i32;
                            sum_cr += cri as i32;
                            count += 1;
                        }
                    }
                }
                if count > 0 {
                    let c_idx = cr * chroma_w + cc;
                    cb_plane[c_idx] = (sum_cb / count).clamp(0, 255) as u8;
                    cr_plane[c_idx] = (sum_cr / count).clamp(0, 255) as u8;
                }
            }
        }

        Ok((y_plane, cb_plane, cr_plane, w, h))
    }
}

impl VideoEncoder for ApvEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Apv
    }

    fn send_frame(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        if self.flushing {
            return Err(CodecError::InvalidParameter(
                "Cannot send frames after flush".to_string(),
            ));
        }
        self.encode_frame(frame)
    }

    fn receive_packet(&mut self) -> CodecResult<Option<EncodedPacket>> {
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

    fn config(&self) -> &EncoderConfig {
        &self.encoder_config
    }
}

// ── Inline BT.601 color conversion (no external dependency) ────────────────

/// Convert RGB to YCbCr using ITU-R BT.601 coefficients.
///
/// Returns (Y, Cb, Cr) each clamped to 0–255.
fn rgb_to_ycbcr(r: u8, g: u8, b: u8) -> (u8, u8, u8) {
    let rf = r as f32;
    let gf = g as f32;
    let bf = b as f32;
    let y = (0.299 * rf + 0.587 * gf + 0.114 * bf).clamp(0.0, 255.0) as u8;
    let cb = (-0.168736 * rf - 0.331264 * gf + 0.5 * bf + 128.0).clamp(0.0, 255.0) as u8;
    let cr = (0.5 * rf - 0.418688 * gf - 0.081312 * bf + 128.0).clamp(0.0, 255.0) as u8;
    (y, cb, cr)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::frame::{FrameType, Plane};
    use oximedia_core::Timestamp;

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

    /// Create a YUV420p VideoFrame.
    fn make_yuv420p_frame(width: u32, height: u32) -> VideoFrame {
        let y_size = (width * height) as usize;
        let chroma_w = ((width + 1) / 2) as usize;
        let chroma_h = ((height + 1) / 2) as usize;
        let c_size = chroma_w * chroma_h;

        let y_data: Vec<u8> = (0..y_size).map(|i| ((i % 235) + 16) as u8).collect();
        let u_data: Vec<u8> = (0..c_size).map(|i| ((i % 225) + 16) as u8).collect();
        let v_data: Vec<u8> = (0..c_size).map(|i| ((i % 225) + 16) as u8).collect();

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, width, height);
        frame.planes = vec![
            Plane::with_dimensions(y_data, width as usize, width, height),
            Plane::with_dimensions(u_data, chroma_w, (width + 1) / 2, (height + 1) / 2),
            Plane::with_dimensions(v_data, chroma_w, (width + 1) / 2, (height + 1) / 2),
        ];
        frame.timestamp = Timestamp::new(0, Rational::new(1, 1000));
        frame.frame_type = FrameType::Key;
        frame
    }

    #[test]
    fn test_encoder_creation() {
        let config = ApvConfig::new(320, 240).expect("valid config");
        let encoder = ApvEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_with_dimensions() {
        let encoder = ApvEncoder::with_dimensions(640, 480);
        assert!(encoder.is_ok());
        let enc = encoder.expect("valid encoder");
        assert_eq!(enc.qp(), 22);
        assert_eq!(enc.frame_count(), 0);
    }

    #[test]
    fn test_encoder_invalid_dimensions() {
        let result = ApvEncoder::with_dimensions(0, 480);
        assert!(result.is_err());
    }

    #[test]
    fn test_encoder_codec_id() {
        let enc = ApvEncoder::with_dimensions(16, 16).expect("valid encoder");
        assert_eq!(enc.codec(), CodecId::Apv);
    }

    #[test]
    fn test_encode_solid_color_frame() {
        let mut enc = ApvEncoder::with_dimensions(16, 16).expect("valid encoder");
        let frame = make_solid_rgb_frame(16, 16, 128, 128, 128);

        let result = enc.send_frame(&frame);
        assert!(result.is_ok(), "send_frame failed: {result:?}");

        let packet = enc.receive_packet().expect("receive_packet failed");
        assert!(packet.is_some(), "Expected a packet");

        let pkt = packet.expect("packet should be Some");
        assert!(pkt.keyframe, "APV frames are always keyframes");
        assert!(!pkt.data.is_empty(), "Encoded data should not be empty");
        assert_eq!(pkt.pts, 0);

        // Verify APV signature
        assert_eq!(&pkt.data[0..4], b"APV1");
    }

    #[test]
    fn test_encode_gradient_frame() {
        let mut enc = ApvEncoder::with_dimensions(64, 64).expect("valid encoder");
        let frame = make_gradient_rgb_frame(64, 64);

        enc.send_frame(&frame).expect("encode gradient frame");
        let pkt = enc
            .receive_packet()
            .expect("receive")
            .expect("expected packet");
        assert!(!pkt.data.is_empty());
        assert_eq!(&pkt.data[0..4], b"APV1");
    }

    #[test]
    fn test_encode_yuv420p_frame() {
        let config = ApvConfig::new(16, 16).expect("valid config");
        let mut enc = ApvEncoder::new(config).expect("valid encoder");
        let frame = make_yuv420p_frame(16, 16);

        let result = enc.send_frame(&frame);
        assert!(result.is_ok(), "YUV420p encode failed: {result:?}");

        let pkt = enc.receive_packet().expect("receive").expect("packet");
        assert!(!pkt.data.is_empty());
    }

    #[test]
    fn test_encode_various_resolutions() {
        for (w, h) in [(16, 16), (64, 64), (320, 240), (13, 7), (8, 8)] {
            let mut enc = ApvEncoder::with_dimensions(w, h).expect("valid encoder");
            let frame = make_solid_rgb_frame(w, h, 100, 150, 200);
            enc.send_frame(&frame)
                .unwrap_or_else(|e| panic!("failed to encode {w}x{h}: {e}"));
            let pkt = enc
                .receive_packet()
                .expect("receive")
                .expect("expected packet");
            assert!(!pkt.data.is_empty(), "empty output for {w}x{h}");
        }
    }

    #[test]
    fn test_encode_multiple_frames() {
        let mut enc = ApvEncoder::with_dimensions(16, 16).expect("valid encoder");

        for i in 0..5 {
            let frame = make_solid_rgb_frame(16, 16, i * 50, 128, 64);
            enc.send_frame(&frame).expect("send_frame failed");
        }

        assert_eq!(enc.frame_count(), 5);

        for i in 0..5 {
            let pkt = enc
                .receive_packet()
                .expect("receive")
                .expect("expected packet");
            assert_eq!(pkt.pts, i);
            assert!(pkt.keyframe);
        }

        let pkt = enc.receive_packet().expect("receive");
        assert!(pkt.is_none());
    }

    #[test]
    fn test_higher_qp_produces_smaller_output() {
        let frame = make_gradient_rgb_frame(64, 64);

        let config_low_qp = ApvConfig::new(64, 64).expect("valid").with_qp(5);
        let mut enc_low = ApvEncoder::new(config_low_qp).expect("valid encoder");
        enc_low.send_frame(&frame).expect("encode low QP");
        let pkt_low = enc_low.receive_packet().expect("receive").expect("packet");

        let config_high_qp = ApvConfig::new(64, 64).expect("valid").with_qp(50);
        let mut enc_high = ApvEncoder::new(config_high_qp).expect("valid encoder");
        enc_high.send_frame(&frame).expect("encode high QP");
        let pkt_high = enc_high.receive_packet().expect("receive").expect("packet");

        assert!(
            pkt_high.data.len() <= pkt_low.data.len(),
            "Higher QP ({}) should produce smaller or equal output: {} vs {}",
            50,
            pkt_high.data.len(),
            pkt_low.data.len()
        );
    }

    #[test]
    fn test_flush_prevents_more_frames() {
        let mut enc = ApvEncoder::with_dimensions(16, 16).expect("valid encoder");
        enc.flush().expect("flush should succeed");

        let frame = make_solid_rgb_frame(16, 16, 128, 128, 128);
        let result = enc.send_frame(&frame);
        assert!(result.is_err(), "Should not accept frames after flush");
    }

    #[test]
    fn test_config_reflects_apv() {
        let enc = ApvEncoder::with_dimensions(640, 480).expect("valid encoder");
        let config = enc.config();
        assert_eq!(config.codec, CodecId::Apv);
        assert_eq!(config.width, 640);
        assert_eq!(config.height, 480);
        assert_eq!(config.keyint, 1);
    }

    #[test]
    fn test_encode_non_multiple_of_8() {
        let mut enc = ApvEncoder::with_dimensions(13, 7).expect("valid encoder");
        let frame = make_solid_rgb_frame(13, 7, 200, 100, 50);
        enc.send_frame(&frame).expect("encode non-aligned frame");
        let pkt = enc.receive_packet().expect("receive").expect("packet");
        assert!(!pkt.data.is_empty());
        assert_eq!(&pkt.data[0..4], b"APV1");
    }

    #[test]
    fn test_encode_with_tiles() {
        let config = ApvConfig::new(64, 64)
            .expect("valid")
            .with_tiles(2, 2)
            .expect("valid tiles");
        let mut enc = ApvEncoder::new(config).expect("valid encoder");
        let frame = make_gradient_rgb_frame(64, 64);
        enc.send_frame(&frame).expect("encode tiled frame");
        let pkt = enc.receive_packet().expect("receive").expect("packet");
        assert!(!pkt.data.is_empty());
        assert_eq!(&pkt.data[0..4], b"APV1");
    }

    #[test]
    fn test_rgb_to_ycbcr_neutral_gray() {
        let (y, cb, cr) = rgb_to_ycbcr(128, 128, 128);
        // Neutral gray: Y≈128, Cb≈128, Cr≈128
        assert!((y as i32 - 128).abs() <= 1);
        assert!((cb as i32 - 128).abs() <= 1);
        assert!((cr as i32 - 128).abs() <= 1);
    }

    #[test]
    fn test_rgb_to_ycbcr_white() {
        let (y, _cb, _cr) = rgb_to_ycbcr(255, 255, 255);
        assert_eq!(y, 255); // White has maximum Y
    }

    #[test]
    fn test_rgb_to_ycbcr_black() {
        let (y, cb, cr) = rgb_to_ycbcr(0, 0, 0);
        assert_eq!(y, 0);
        assert_eq!(cb, 128); // Neutral chroma
        assert_eq!(cr, 128);
    }

    #[test]
    fn test_apv_config_access() {
        let config = ApvConfig::new(320, 240).expect("valid").with_qp(30);
        let enc = ApvEncoder::new(config).expect("valid encoder");
        assert_eq!(enc.apv_config().width, 320);
        assert_eq!(enc.apv_config().height, 240);
        assert_eq!(enc.apv_config().qp, 30);
    }
}
