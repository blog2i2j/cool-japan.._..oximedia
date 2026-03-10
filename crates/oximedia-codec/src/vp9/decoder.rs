//! VP9 decoder implementation.

#![forbid(unsafe_code)]
#![allow(clippy::match_same_arms)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::struct_excessive_bools)]
#![allow(dead_code)]

use crate::error::{CodecError, CodecResult};
use crate::frame::{FrameType, VideoFrame};
use crate::reconstruct::{
    ChromaSubsampling, DecoderPipeline, FrameBuffer, FrameContext as ReconFrameContext,
    PipelineConfig,
};
use crate::traits::{DecoderConfig, VideoDecoder};
use crate::vp9::coeff_decode::{CoeffContext, CoeffDecoder};
use crate::vp9::compressed::CompressedHeader;
use crate::vp9::intra::IntraMode;
use crate::vp9::partition::{BlockSize, Partition, TxSize};
use crate::vp9::probability::FrameContext;
use crate::vp9::segmentation::Segmentation;
use crate::vp9::superframe::Superframe;
use crate::vp9::symbols::SymbolDecoder;
use crate::vp9::transform::{apply_inverse_transform, CoeffBuffer, TxType};
use crate::vp9::uncompressed::UncompressedHeader;
use oximedia_core::{CodecId, PixelFormat, Rational, Timestamp};

/// VP9 decoder with full reconstruction pipeline.
#[derive(Debug)]
pub struct Vp9Decoder {
    #[allow(dead_code)]
    config: DecoderConfig,
    width: u32,
    height: u32,
    output_format: PixelFormat,
    output_queue: Vec<VideoFrame>,
    ref_frames: [Option<VideoFrame>; 8],
    flushing: bool,
    /// Reconstruction pipeline.
    pipeline: Option<DecoderPipeline>,
    /// Frame context for probability tables.
    frame_context: FrameContext,
    /// Compressed header state.
    compressed_header: CompressedHeader,
    /// Symbol decoder for entropy coding.
    symbol_decoder: SymbolDecoder,
    /// Coefficient decoder.
    coeff_decoder: CoeffDecoder,
    /// Frame counter.
    frame_count: u64,
}

impl Vp9Decoder {
    /// Creates a new VP9 decoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: DecoderConfig) -> CodecResult<Self> {
        Ok(Self {
            config,
            width: 0,
            height: 0,
            output_format: PixelFormat::Yuv420p,
            output_queue: Vec::new(),
            ref_frames: Default::default(),
            flushing: false,
            pipeline: None,
            frame_context: FrameContext::new(),
            compressed_header: CompressedHeader::new(),
            symbol_decoder: SymbolDecoder::new(),
            coeff_decoder: CoeffDecoder::new(),
            frame_count: 0,
        })
    }

    /// Initializes or reconfigures the reconstruction pipeline.
    fn init_pipeline(&mut self, width: u32, height: u32, bit_depth: u8) -> CodecResult<()> {
        let subsampling = match self.output_format {
            PixelFormat::Yuv420p | PixelFormat::Yuv420p10le | PixelFormat::Yuv420p12le => {
                ChromaSubsampling::Cs420
            }
            PixelFormat::Yuv422p => ChromaSubsampling::Cs422,
            PixelFormat::Yuv444p => ChromaSubsampling::Cs444,
            _ => ChromaSubsampling::Cs420,
        };

        let pipeline_config = PipelineConfig::new(width, height)
            .with_bit_depth(bit_depth)
            .with_subsampling(subsampling);

        if let Some(ref mut pipeline) = self.pipeline {
            pipeline.reconfigure(pipeline_config).map_err(|e| {
                CodecError::DecoderError(format!("Pipeline reconfigure failed: {e}"))
            })?;
        } else {
            self.pipeline =
                Some(DecoderPipeline::new(pipeline_config).map_err(|e| {
                    CodecError::DecoderError(format!("Pipeline creation failed: {e}"))
                })?);
        }

        Ok(())
    }

    /// Decodes a complete frame with full reconstruction.
    fn decode_frame(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        let header = UncompressedHeader::parse(data)?;

        if header.show_existing_frame {
            let idx = header.frame_to_show as usize;
            if let Some(ref frame) = self.ref_frames[idx] {
                let mut output = frame.clone();
                output.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
                self.output_queue.push(output);
            }
            return Ok(());
        }

        // Update dimensions and format
        if header.width > 0 && header.height > 0 {
            self.width = header.width;
            self.height = header.height;
        }

        self.output_format = match (header.bit_depth, header.subsampling_x, header.subsampling_y) {
            (8, true, true) => PixelFormat::Yuv420p,
            (8, true, false) => PixelFormat::Yuv422p,
            (8, false, false) => PixelFormat::Yuv444p,
            (10, true, true) => PixelFormat::Yuv420p10le,
            (12, true, true) => PixelFormat::Yuv420p12le,
            _ => PixelFormat::Yuv420p,
        };

        // Initialize pipeline if needed
        self.init_pipeline(self.width, self.height, header.bit_depth)?;

        // Setup reconstruction context
        let mut recon_context = ReconFrameContext::new(self.width, self.height);
        recon_context.bit_depth = header.bit_depth;
        recon_context.is_keyframe = header.is_keyframe();
        recon_context.show_frame = header.show_frame;
        recon_context.decode_order = self.frame_count;
        recon_context.display_order = self.frame_count; // Simplified

        // Process frame through pipeline
        let frame_buffer = if let Some(ref mut pipeline) = self.pipeline {
            pipeline
                .process_frame(data, &recon_context)
                .map_err(|e| CodecError::DecoderError(format!("Pipeline processing failed: {e}")))?
        } else {
            return Err(CodecError::DecoderError("Pipeline not initialized".into()));
        };

        // Convert FrameBuffer to VideoFrame
        let mut frame = self.frame_buffer_to_video_frame(frame_buffer)?;
        frame.timestamp = Timestamp::new(pts, Rational::new(1, 1000));
        frame.frame_type = if header.is_keyframe() {
            FrameType::Key
        } else {
            FrameType::Inter
        };

        // Update reference frames
        for i in 0..8 {
            if header.refresh_frame_flags & (1 << i) != 0 {
                self.ref_frames[i] = Some(frame.clone());
            }
        }

        // Add to output queue if showing
        if header.show_frame {
            self.output_queue.push(frame);
        }

        self.frame_count += 1;

        Ok(())
    }

    /// Converts FrameBuffer to VideoFrame.
    fn frame_buffer_to_video_frame(&self, buffer: FrameBuffer) -> CodecResult<VideoFrame> {
        use crate::frame::Plane;
        let mut frame = VideoFrame::new(self.output_format, buffer.width(), buffer.height());

        // Convert Y plane
        let y_data = buffer.y_plane().to_u8();
        let y_stride = buffer.y_plane().width() as usize;
        frame.planes.push(Plane::new(y_data, y_stride));

        // Convert U plane if present
        if let Some(u_plane) = buffer.u_plane() {
            let u_data = u_plane.to_u8();
            let u_stride = u_plane.width() as usize;
            frame.planes.push(Plane::new(u_data, u_stride));
        }

        // Convert V plane if present
        if let Some(v_plane) = buffer.v_plane() {
            let v_data = v_plane.to_u8();
            let v_stride = v_plane.width() as usize;
            frame.planes.push(Plane::new(v_data, v_stride));
        }

        Ok(frame)
    }

    /// Decodes a superblock (64x64).
    fn decode_superblock(
        &mut self,
        data: &[u8],
        x: usize,
        y: usize,
        output: &mut FrameBuffer,
    ) -> CodecResult<()> {
        // Decode partition for this superblock
        let partition = self
            .symbol_decoder
            .decode_partition(data, &self.frame_context, 0)?;

        match partition {
            Partition::None => {
                // Decode single 64x64 block
                self.decode_block(data, x, y, BlockSize::Block64x64, output)?;
            }
            Partition::Split => {
                // Recursively decode 4x 32x32 blocks
                for i in 0..4 {
                    let bx = x + ((i & 1) * 32);
                    let by = y + ((i >> 1) * 32);
                    self.decode_block(data, bx, by, BlockSize::Block32x32, output)?;
                }
            }
            Partition::Horz => {
                // Decode 2x horizontal blocks
                self.decode_block(data, x, y, BlockSize::Block64x32, output)?;
                self.decode_block(data, x, y + 32, BlockSize::Block64x32, output)?;
            }
            Partition::Vert => {
                // Decode 2x vertical blocks
                self.decode_block(data, x, y, BlockSize::Block32x64, output)?;
                self.decode_block(data, x + 32, y, BlockSize::Block32x64, output)?;
            }
        }

        Ok(())
    }

    /// Decodes a single block.
    fn decode_block(
        &mut self,
        data: &[u8],
        x: usize,
        y: usize,
        block_size: BlockSize,
        output: &mut FrameBuffer,
    ) -> CodecResult<()> {
        // Decode skip flag
        let skip = self
            .symbol_decoder
            .decode_skip(data, &self.frame_context, 0)?;

        if skip {
            // Skip block - copy from reference or fill with DC
            return Ok(());
        }

        // Decode intra vs inter
        let is_inter = self
            .symbol_decoder
            .decode_is_inter(data, &self.frame_context, 0)?;

        if is_inter {
            // Inter prediction
            self.decode_inter_block(data, x, y, block_size, output)?;
        } else {
            // Intra prediction
            self.decode_intra_block(data, x, y, block_size, output)?;
        }

        Ok(())
    }

    /// Decodes an intra-predicted block.
    fn decode_intra_block(
        &mut self,
        data: &[u8],
        x: usize,
        y: usize,
        block_size: BlockSize,
        output: &mut FrameBuffer,
    ) -> CodecResult<()> {
        // Decode intra mode
        let y_mode = self.symbol_decoder.decode_intra_y_mode_kf(
            data,
            &self.frame_context,
            0, // above mode
            0, // left mode
        )?;

        // Get transform size
        let tx_size = self.get_tx_size_for_block(block_size);

        // Decode coefficients and apply transform
        self.decode_and_transform(data, x, y, tx_size, output)?;

        // Apply intra prediction
        self.apply_intra_prediction(x, y, block_size, y_mode, output)?;

        Ok(())
    }

    /// Decodes an inter-predicted block.
    fn decode_inter_block(
        &mut self,
        data: &[u8],
        x: usize,
        y: usize,
        block_size: BlockSize,
        output: &mut FrameBuffer,
    ) -> CodecResult<()> {
        // Decode inter mode
        let inter_mode = self
            .symbol_decoder
            .decode_inter_mode(data, &self.frame_context, 0)?;

        // Decode motion vector if needed
        if inter_mode.requires_mv_delta() {
            let _mv = self
                .symbol_decoder
                .decode_mv(data, &self.frame_context, false)?;
            // Motion compensation would be applied here
        }

        // Get transform size
        let tx_size = self.get_tx_size_for_block(block_size);

        // Decode residual coefficients and apply transform
        self.decode_and_transform(data, x, y, tx_size, output)?;

        Ok(())
    }

    /// Decodes coefficients and applies inverse transform.
    fn decode_and_transform(
        &mut self,
        data: &[u8],
        x: usize,
        y: usize,
        tx_size: TxSize,
        output: &mut FrameBuffer,
    ) -> CodecResult<()> {
        let mut coeffs = CoeffBuffer::for_size(tx_size);
        let mut coeff_ctx = CoeffContext::new(0, tx_size);

        // Decode coefficients for Y plane
        self.coeff_decoder.decode_block(
            data,
            &mut coeffs,
            &self.frame_context,
            &mut coeff_ctx,
            &self.compressed_header.segmentation.segments[0],
            x,
            y,
        )?;

        // Apply inverse transform to output buffer
        let y_plane = output.y_plane_mut();
        let stride = y_plane.stride();
        let pixel_offset = y * stride + x;

        if pixel_offset < y_plane.data().len() {
            // Create mutable slice for transform output
            let data_mut = y_plane.data_mut();
            let mut temp_output = vec![0u8; tx_size.size() * tx_size.size()];

            // Apply transform
            apply_inverse_transform(
                &coeffs,
                &mut temp_output,
                tx_size.size(),
                tx_size,
                TxType::DctDct,
            );

            // Copy transformed data to output
            let size = tx_size.size();
            for row in 0..size {
                let src_start = row * size;
                let dst_start = pixel_offset + row * stride;
                if dst_start + size <= data_mut.len() {
                    for col in 0..size {
                        data_mut[dst_start + col] =
                            temp_output[src_start + col].clamp(0, 255) as i16;
                    }
                }
            }
        }

        Ok(())
    }

    /// Applies intra prediction to a block.
    fn apply_intra_prediction(
        &mut self,
        _x: usize,
        _y: usize,
        _block_size: BlockSize,
        _mode: IntraMode,
        _output: &mut FrameBuffer,
    ) -> CodecResult<()> {
        // Intra prediction implementation
        // This would use the mode to predict from neighboring pixels
        Ok(())
    }

    /// Gets transform size for a block size.
    fn get_tx_size_for_block(&self, block_size: BlockSize) -> TxSize {
        match block_size {
            BlockSize::Block4x4 | BlockSize::Block4x8 | BlockSize::Block8x4 => TxSize::Tx4x4,
            BlockSize::Block8x8 | BlockSize::Block8x16 | BlockSize::Block16x8 => TxSize::Tx8x8,
            BlockSize::Block16x16 | BlockSize::Block16x32 | BlockSize::Block32x16 => {
                TxSize::Tx16x16
            }
            _ => TxSize::Tx32x32,
        }
    }

    /// Returns the number of pending output frames.
    #[must_use]
    pub fn pending_frames(&self) -> usize {
        self.output_queue.len()
    }

    /// Returns true if the decoder has been flushed.
    #[must_use]
    pub fn is_flushing(&self) -> bool {
        self.flushing
    }
}

impl VideoDecoder for Vp9Decoder {
    fn codec(&self) -> CodecId {
        CodecId::Vp9
    }

    #[allow(clippy::cast_possible_wrap)]
    fn send_packet(&mut self, data: &[u8], pts: i64) -> CodecResult<()> {
        if self.flushing {
            return Err(CodecError::InvalidParameter(
                "Cannot send packet while flushing".into(),
            ));
        }

        let superframe = Superframe::parse(data)?;

        for (i, frame_data) in superframe.frames.iter().enumerate() {
            let frame_pts = pts + i as i64;
            self.decode_frame(frame_data, frame_pts)?;
        }

        Ok(())
    }

    fn receive_frame(&mut self) -> CodecResult<Option<VideoFrame>> {
        if self.output_queue.is_empty() {
            if self.flushing {
                return Err(CodecError::Eof);
            }
            return Ok(None);
        }

        Ok(Some(self.output_queue.remove(0)))
    }

    fn flush(&mut self) -> CodecResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.output_queue.clear();
        self.ref_frames = Default::default();
        self.flushing = false;
    }

    fn output_format(&self) -> Option<PixelFormat> {
        if self.width > 0 && self.height > 0 {
            Some(self.output_format)
        } else {
            None
        }
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

    #[test]
    fn test_vp9_decoder_new() {
        let config = DecoderConfig::default();
        let decoder = Vp9Decoder::new(config).expect("should succeed");
        assert_eq!(decoder.codec(), CodecId::Vp9);
        assert_eq!(decoder.pending_frames(), 0);
        assert!(!decoder.is_flushing());
    }

    #[test]
    fn test_decoder_initial_state() {
        let config = DecoderConfig::default();
        let decoder = Vp9Decoder::new(config).expect("should succeed");
        assert!(decoder.output_format().is_none());
        assert!(decoder.dimensions().is_none());
    }

    #[test]
    fn test_flush() {
        let config = DecoderConfig::default();
        let mut decoder = Vp9Decoder::new(config).expect("should succeed");
        assert!(!decoder.is_flushing());
        decoder.flush().expect("should succeed");
        assert!(decoder.is_flushing());
    }

    #[test]
    fn test_reset() {
        let config = DecoderConfig::default();
        let mut decoder = Vp9Decoder::new(config).expect("should succeed");
        decoder.flush().expect("should succeed");
        assert!(decoder.is_flushing());
        decoder.reset();
        assert!(!decoder.is_flushing());
    }

    #[test]
    fn test_receive_no_frame() {
        let config = DecoderConfig::default();
        let mut decoder = Vp9Decoder::new(config).expect("should succeed");
        let frame = decoder.receive_frame().expect("should succeed");
        assert!(frame.is_none());
    }

    #[test]
    fn test_send_while_flushing() {
        let config = DecoderConfig::default();
        let mut decoder = Vp9Decoder::new(config).expect("should succeed");
        decoder.flush().expect("should succeed");
        let result = decoder.send_packet(&[0x80], 0);
        assert!(result.is_err());
    }
}
