//! PCM codec implementation.

use crate::{
    AudioDecoder, AudioDecoderConfig, AudioEncoder, AudioEncoderConfig, AudioError, AudioFrame,
    AudioResult, ChannelLayout, EncodedAudioPacket,
};
use oximedia_core::{CodecId, SampleFormat};

/// PCM decoder.
pub struct PcmDecoder {
    #[allow(dead_code)]
    config: AudioDecoderConfig,
    sample_rate: u32,
    channels: u8,
    format: SampleFormat,
    flushing: bool,
}

impl PcmDecoder {
    /// Create new PCM decoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: &AudioDecoderConfig) -> AudioResult<Self> {
        if config.codec != CodecId::Pcm {
            return Err(AudioError::InvalidParameter("Expected PCM codec".into()));
        }
        Ok(Self {
            config: config.clone(),
            sample_rate: config.sample_rate,
            channels: config.channels,
            format: SampleFormat::S16,
            flushing: false,
        })
    }

    /// Create new PCM decoder with specified format.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn with_format(config: &AudioDecoderConfig, format: SampleFormat) -> AudioResult<Self> {
        if config.codec != CodecId::Pcm {
            return Err(AudioError::InvalidParameter("Expected PCM codec".into()));
        }
        Ok(Self {
            config: config.clone(),
            sample_rate: config.sample_rate,
            channels: config.channels,
            format,
            flushing: false,
        })
    }
}

impl AudioDecoder for PcmDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Pcm
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
    }

    fn output_format(&self) -> Option<SampleFormat> {
        Some(self.format)
    }

    fn sample_rate(&self) -> Option<u32> {
        Some(self.sample_rate)
    }

    fn channel_layout(&self) -> Option<ChannelLayout> {
        Some(ChannelLayout::from_count(usize::from(self.channels)))
    }
}

/// PCM encoder.
pub struct PcmEncoder {
    config: AudioEncoderConfig,
    format: SampleFormat,
    flushing: bool,
}

impl PcmEncoder {
    /// Create new PCM encoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: AudioEncoderConfig) -> AudioResult<Self> {
        if config.codec != CodecId::Pcm {
            return Err(AudioError::InvalidParameter("Expected PCM codec".into()));
        }
        Ok(Self {
            config,
            format: SampleFormat::S16,
            flushing: false,
        })
    }

    /// Create new PCM encoder with specified format.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn with_format(config: AudioEncoderConfig, format: SampleFormat) -> AudioResult<Self> {
        if config.codec != CodecId::Pcm {
            return Err(AudioError::InvalidParameter("Expected PCM codec".into()));
        }
        Ok(Self {
            config,
            format,
            flushing: false,
        })
    }

    /// Get the sample format.
    #[must_use]
    pub fn format(&self) -> SampleFormat {
        self.format
    }
}

impl AudioEncoder for PcmEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Pcm
    }

    fn send_frame(&mut self, _frame: &AudioFrame) -> AudioResult<()> {
        Ok(())
    }

    fn receive_packet(&mut self) -> AudioResult<Option<EncodedAudioPacket>> {
        Ok(None)
    }

    fn flush(&mut self) -> AudioResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn config(&self) -> &AudioEncoderConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pcm_decoder() {
        let config = AudioDecoderConfig {
            codec: CodecId::Pcm,
            ..Default::default()
        };
        let decoder = PcmDecoder::new(&config).expect("should succeed");
        assert_eq!(decoder.codec(), CodecId::Pcm);
    }

    #[test]
    fn test_pcm_decoder_with_format() {
        let config = AudioDecoderConfig {
            codec: CodecId::Pcm,
            ..Default::default()
        };
        let decoder = PcmDecoder::with_format(&config, SampleFormat::F32).expect("should succeed");
        assert_eq!(decoder.output_format(), Some(SampleFormat::F32));
    }

    #[test]
    fn test_pcm_encoder() {
        let config = AudioEncoderConfig {
            codec: CodecId::Pcm,
            ..Default::default()
        };
        let encoder = PcmEncoder::new(config).expect("should succeed");
        assert_eq!(encoder.codec(), CodecId::Pcm);
    }

    #[test]
    fn test_pcm_encoder_with_format() {
        let config = AudioEncoderConfig {
            codec: CodecId::Pcm,
            ..Default::default()
        };
        let encoder = PcmEncoder::with_format(config, SampleFormat::F32).expect("should succeed");
        assert_eq!(encoder.format(), SampleFormat::F32);
    }
}
