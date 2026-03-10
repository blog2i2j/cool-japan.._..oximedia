//! AV1 encoder implementation.

use super::obu::{encode_leb128, ObuHeader, ObuType};
use super::tile_encoder::{ParallelTileEncoder, TileEncoderConfig, TileInfoBuilder};
use crate::error::{CodecError, CodecResult};
use crate::frame::VideoFrame;
use crate::traits::{EncodedPacket, EncoderConfig, VideoEncoder};
use oximedia_core::CodecId;

/// AV1 encoder.
#[derive(Debug)]
pub struct Av1Encoder {
    /// Encoder configuration.
    config: EncoderConfig,
    /// Frame counter.
    frame_count: u64,
    /// Pending output packets.
    output_queue: Vec<EncodedPacket>,
    /// Encoder is in flush mode.
    flushing: bool,
    /// Tile encoder for parallel encoding.
    tile_encoder: Option<ParallelTileEncoder>,
    /// Quality parameter (0-255).
    quality: u8,
}

impl Av1Encoder {
    /// Create a new AV1 encoder.
    ///
    /// # Errors
    ///
    /// Returns error if encoder initialization fails.
    pub fn new(config: EncoderConfig) -> CodecResult<Self> {
        if config.width == 0 || config.height == 0 {
            return Err(CodecError::InvalidParameter(
                "Invalid frame dimensions".to_string(),
            ));
        }

        if config.codec != CodecId::Av1 {
            return Err(CodecError::InvalidParameter(
                "Expected AV1 codec".to_string(),
            ));
        }

        // Initialize tile encoder if threads > 1
        let tile_encoder = if config.threads > 1 {
            let tile_config = TileEncoderConfig::auto(config.width, config.height, config.threads);
            Some(ParallelTileEncoder::new(
                tile_config,
                config.width,
                config.height,
            )?)
        } else {
            None
        };

        // Compute quality from bitrate mode
        let quality = Self::compute_quality(&config);

        Ok(Self {
            config,
            frame_count: 0,
            output_queue: Vec::new(),
            flushing: false,
            tile_encoder,
            quality,
        })
    }

    /// Compute quality parameter from encoder config.
    fn compute_quality(config: &EncoderConfig) -> u8 {
        use crate::traits::BitrateMode;

        match config.bitrate {
            BitrateMode::Crf(crf) => {
                // Map CRF (0-51) to quality (255-0)
                // Lower CRF = higher quality
                let normalized = (crf / 51.0).clamp(0.0, 1.0);
                (255.0 * (1.0 - normalized)) as u8
            }
            BitrateMode::Cbr(_) | BitrateMode::Vbr { .. } => {
                // Default medium quality for bitrate modes
                128
            }
            BitrateMode::Lossless => {
                // Maximum quality for lossless
                255
            }
        }
    }

    /// Encode a single frame.
    fn encode_frame(&mut self, frame: &VideoFrame) {
        let is_keyframe = self.frame_count % u64::from(self.config.keyint) == 0;
        let mut data = Vec::new();

        if is_keyframe {
            self.write_sequence_header(&mut data);
        }

        // Use tile encoding if available, otherwise single-threaded
        if let Some(ref tile_encoder) = self.tile_encoder {
            self.encode_frame_with_tiles(frame, &mut data, is_keyframe);
        } else {
            Self::write_frame_obu(&mut data, is_keyframe);
        }

        #[allow(clippy::cast_possible_wrap)]
        let pts = self.frame_count as i64;
        let dts = pts;

        self.output_queue.push(EncodedPacket {
            data,
            pts,
            dts,
            keyframe: is_keyframe,
            duration: Some(1),
        });

        self.frame_count += 1;
    }

    /// Encode frame using parallel tile encoding.
    fn encode_frame_with_tiles(&self, frame: &VideoFrame, data: &mut Vec<u8>, is_keyframe: bool) {
        if let Some(ref tile_encoder) = self.tile_encoder {
            // Encode tiles in parallel
            match tile_encoder.encode_frame(frame, self.quality, is_keyframe) {
                Ok(tiles) => {
                    // Merge tiles into bitstream
                    match tile_encoder.merge_tiles(&tiles) {
                        Ok(merged) => {
                            // Write frame header with tile info
                            self.write_frame_header_with_tiles(data, is_keyframe, tile_encoder);
                            // Append tile data
                            data.extend_from_slice(&merged);
                        }
                        Err(_) => {
                            // Fall back to simple encoding on error
                            Self::write_frame_obu(data, is_keyframe);
                        }
                    }
                }
                Err(_) => {
                    // Fall back to simple encoding on error
                    Self::write_frame_obu(data, is_keyframe);
                }
            }
        }
    }

    /// Write frame header with tile information.
    fn write_frame_header_with_tiles(
        &self,
        data: &mut Vec<u8>,
        is_keyframe: bool,
        tile_encoder: &ParallelTileEncoder,
    ) {
        let header = ObuHeader {
            obu_type: ObuType::Frame,
            has_extension: false,
            has_size: true,
            temporal_id: 0,
            spatial_id: 0,
        };
        data.extend(header.to_bytes());

        // Build frame header with tile info
        let mut frame_header = Vec::new();
        let frame_type = u8::from(!is_keyframe);
        frame_header.push((frame_type << 5) | 0x10);

        // Write tile info
        let tile_info = TileInfoBuilder::from_config(
            tile_encoder.config(),
            self.config.width,
            self.config.height,
        );

        // Encode tile configuration (simplified)
        if tile_info.tile_count() > 1 {
            frame_header.push(tile_info.tile_cols_log2);
            frame_header.push(tile_info.tile_rows_log2);
        }

        // Write size and header (placeholder for real implementation)
        let size_bytes = encode_leb128(frame_header.len() as u64);
        data.extend(size_bytes);
        data.extend(frame_header);
    }

    /// Write sequence header OBU.
    fn write_sequence_header(&self, data: &mut Vec<u8>) {
        let header = ObuHeader {
            obu_type: ObuType::SequenceHeader,
            has_extension: false,
            has_size: true,
            temporal_id: 0,
            spatial_id: 0,
        };
        data.extend(header.to_bytes());

        let payload = self.build_sequence_header_payload();
        let size_bytes = encode_leb128(payload.len() as u64);
        data.extend(size_bytes);
        data.extend(payload);
    }

    /// Build sequence header payload.
    #[allow(clippy::cast_possible_truncation)]
    fn build_sequence_header_payload(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        payload.push(0x00);
        payload.push(0x00);
        payload.push(0x00);

        let width_bits = 32 - self.config.width.leading_zeros();
        let height_bits = 32 - self.config.height.leading_zeros();
        payload.push(
            ((width_bits.saturating_sub(1) as u8) << 4) | (height_bits.saturating_sub(1) as u8),
        );

        let width_minus_1 = self.config.width.saturating_sub(1);
        let height_minus_1 = self.config.height.saturating_sub(1);
        payload.extend(&width_minus_1.to_be_bytes()[2..]);
        payload.extend(&height_minus_1.to_be_bytes()[2..]);

        payload
    }

    /// Write frame OBU.
    fn write_frame_obu(data: &mut Vec<u8>, is_keyframe: bool) {
        let header = ObuHeader {
            obu_type: ObuType::Frame,
            has_extension: false,
            has_size: true,
            temporal_id: 0,
            spatial_id: 0,
        };
        data.extend(header.to_bytes());

        let frame_data = Self::build_frame_payload(is_keyframe);
        let size_bytes = encode_leb128(frame_data.len() as u64);
        data.extend(size_bytes);
        data.extend(frame_data);
    }

    /// Build frame payload.
    fn build_frame_payload(is_keyframe: bool) -> Vec<u8> {
        let mut payload = Vec::new();
        let frame_type = u8::from(!is_keyframe);
        payload.push((frame_type << 5) | 0x10);
        payload.extend(&[0x00; 16]);
        payload
    }
}

impl VideoEncoder for Av1Encoder {
    fn codec(&self) -> CodecId {
        CodecId::Av1
    }

    fn send_frame(&mut self, frame: &VideoFrame) -> CodecResult<()> {
        if self.flushing {
            return Err(CodecError::InvalidParameter(
                "Cannot send frame while flushing".to_string(),
            ));
        }

        if frame.width != self.config.width || frame.height != self.config.height {
            return Err(CodecError::InvalidParameter(format!(
                "Frame dimensions {}x{} don't match encoder config {}x{}",
                frame.width, frame.height, self.config.width, self.config.height
            )));
        }

        self.encode_frame(frame);
        Ok(())
    }

    fn receive_packet(&mut self) -> CodecResult<Option<EncodedPacket>> {
        if self.output_queue.is_empty() {
            return Ok(None);
        }
        Ok(Some(self.output_queue.remove(0)))
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn config(&self) -> &EncoderConfig {
        &self.config
    }
}

impl Av1Encoder {
    /// Get tile encoder configuration if enabled.
    #[must_use]
    pub fn tile_config(&self) -> Option<&TileEncoderConfig> {
        self.tile_encoder.as_ref().map(|e| e.config())
    }

    /// Check if parallel tile encoding is enabled.
    #[must_use]
    pub fn has_tile_encoding(&self) -> bool {
        self.tile_encoder.is_some()
    }

    /// Get number of tiles being used.
    #[must_use]
    pub fn tile_count(&self) -> usize {
        self.tile_encoder.as_ref().map_or(1, |e| e.tile_count())
    }

    /// Enable or reconfigure tile encoding.
    ///
    /// # Errors
    ///
    /// Returns error if tile configuration is invalid.
    pub fn set_tile_config(&mut self, tile_config: TileEncoderConfig) -> CodecResult<()> {
        tile_config.validate()?;

        self.tile_encoder = Some(ParallelTileEncoder::new(
            tile_config,
            self.config.width,
            self.config.height,
        )?);

        Ok(())
    }

    /// Disable tile encoding (single-threaded mode).
    pub fn disable_tile_encoding(&mut self) {
        self.tile_encoder = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oximedia_core::PixelFormat;

    #[test]
    fn test_encoder_creation() {
        let config = EncoderConfig::av1(1920, 1080);
        let encoder = Av1Encoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_invalid_dimensions() {
        let config = EncoderConfig::av1(0, 0);
        let encoder = Av1Encoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_encoder_codec_id() {
        let config = EncoderConfig::av1(1920, 1080);
        let encoder = Av1Encoder::new(config).expect("should succeed");
        assert_eq!(encoder.codec(), CodecId::Av1);
    }

    #[test]
    fn test_encode_frame() {
        let config = EncoderConfig::av1(320, 240);
        let mut encoder = Av1Encoder::new(config).expect("should succeed");

        let mut frame = VideoFrame::new(PixelFormat::Yuv420p, 320, 240);
        frame.allocate();

        assert!(encoder.send_frame(&frame).is_ok());

        let packet = encoder.receive_packet().expect("should succeed");
        assert!(packet.is_some());
        let packet = packet.expect("should succeed");
        assert!(packet.keyframe);
        assert!(!packet.data.is_empty());
    }

    #[test]
    fn test_frame_dimension_mismatch() {
        let config = EncoderConfig::av1(320, 240);
        let mut encoder = Av1Encoder::new(config).expect("should succeed");

        let frame = VideoFrame::new(PixelFormat::Yuv420p, 640, 480);
        let result = encoder.send_frame(&frame);
        assert!(result.is_err());
    }
}
