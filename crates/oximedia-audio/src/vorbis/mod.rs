//! Vorbis codec implementation.
//!
//! Vorbis is an open-source, royalty-free lossy audio codec.
//! It uses MDCT transforms and psychoacoustic modeling.
//!
//! # Modules
//!
//! - [`header`] - Header parsing
//! - [`codebook`] - Codebook structures
//! - [`floor`] - Floor types
//! - [`encoder`] - Vorbis encoder
//! - [`bitpack`] - Bitstream packing
//! - [`mdct`] - MDCT transform
//! - [`psycho`] - Psychoacoustic model
//! - [`residue`] - Residue encoding

#![forbid(unsafe_code)]

pub mod bitpack;
pub mod codebook;
pub mod encoder;
pub mod floor;
pub mod header;
pub mod mdct;
pub mod psycho;
pub mod residue;

use crate::{AudioDecoder, AudioDecoderConfig, AudioError, AudioFrame, AudioResult, ChannelLayout};
use oximedia_core::{CodecId, SampleFormat};

// Re-export submodule types
pub use codebook::{Codebook, CodebookEntry, HuffmanTree};
pub use encoder::{QualityMode, VorbisEncoder};
pub use floor::{Floor, FloorType0, FloorType1};
pub use header::{CommentHeader, IdentificationHeader, SetupHeader, VorbisHeader};

/// Vorbis decoder state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[allow(dead_code)]
enum DecoderState {
    /// Waiting for identification header.
    #[default]
    WaitingForIdentification,
    /// Waiting for comment header.
    WaitingForComment,
    /// Waiting for setup header.
    WaitingForSetup,
    /// Ready to decode audio.
    Ready,
    /// Decoder has been flushed.
    Flushed,
}

/// Vorbis decoder.
pub struct VorbisDecoder {
    #[allow(dead_code)]
    config: AudioDecoderConfig,
    sample_rate: u32,
    channels: u8,
    flushing: bool,
    #[allow(dead_code)]
    state: DecoderState,
    #[allow(dead_code)]
    id_header: Option<IdentificationHeader>,
    #[allow(dead_code)]
    comment_header: Option<CommentHeader>,
    #[allow(dead_code)]
    setup_header: Option<SetupHeader>,
}

impl VorbisDecoder {
    /// Create new Vorbis decoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: &AudioDecoderConfig) -> AudioResult<Self> {
        if config.codec != CodecId::Vorbis {
            return Err(AudioError::InvalidParameter("Expected Vorbis codec".into()));
        }
        Ok(Self {
            config: config.clone(),
            sample_rate: config.sample_rate,
            channels: config.channels,
            flushing: false,
            state: DecoderState::WaitingForIdentification,
            id_header: None,
            comment_header: None,
            setup_header: None,
        })
    }

    /// Parse Vorbis identification header.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse_identification_header(data: &[u8]) -> AudioResult<IdentificationHeader> {
        IdentificationHeader::parse(data)
    }

    /// Parse Vorbis comment header.
    ///
    /// # Errors
    ///
    /// Returns error if header is invalid.
    pub fn parse_comment_header(data: &[u8]) -> AudioResult<CommentHeader> {
        CommentHeader::parse(data)
    }

    /// Check if decoder is ready for audio packets.
    #[must_use]
    pub fn is_ready(&self) -> bool {
        self.state == DecoderState::Ready
    }
}

impl AudioDecoder for VorbisDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Vorbis
    }

    fn send_packet(&mut self, _data: &[u8], _pts: i64) -> AudioResult<()> {
        Ok(())
    }

    fn receive_frame(&mut self) -> AudioResult<Option<AudioFrame>> {
        Ok(None)
    }

    fn flush(&mut self) -> AudioResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.flushing = false;
        self.state = DecoderState::WaitingForIdentification;
        self.id_header = None;
        self.comment_header = None;
        self.setup_header = None;
    }

    fn output_format(&self) -> Option<SampleFormat> {
        Some(SampleFormat::F32)
    }

    fn sample_rate(&self) -> Option<u32> {
        Some(self.sample_rate)
    }

    fn channel_layout(&self) -> Option<ChannelLayout> {
        Some(ChannelLayout::from_count(usize::from(self.channels)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vorbis_decoder() {
        let config = AudioDecoderConfig {
            codec: CodecId::Vorbis,
            ..Default::default()
        };
        let decoder = VorbisDecoder::new(&config).expect("should succeed");
        assert_eq!(decoder.codec(), CodecId::Vorbis);
    }

    #[test]
    fn test_vorbis_decoder_wrong_codec() {
        let config = AudioDecoderConfig {
            codec: CodecId::Opus,
            ..Default::default()
        };
        assert!(VorbisDecoder::new(&config).is_err());
    }

    #[test]
    fn test_vorbis_decoder_reset() {
        let config = AudioDecoderConfig {
            codec: CodecId::Vorbis,
            ..Default::default()
        };
        let mut decoder = VorbisDecoder::new(&config).expect("should succeed");
        decoder.reset();
        assert!(!decoder.flushing);
        assert_eq!(decoder.state, DecoderState::WaitingForIdentification);
    }

    #[test]
    fn test_vorbis_decoder_not_ready() {
        let config = AudioDecoderConfig {
            codec: CodecId::Vorbis,
            ..Default::default()
        };
        let decoder = VorbisDecoder::new(&config).expect("should succeed");
        assert!(!decoder.is_ready());
    }
}
