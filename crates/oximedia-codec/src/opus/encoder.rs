//! Opus audio codec encoder.
//!
//! Opus is a modern, royalty-free audio codec designed for interactive
//! speech and music transmission over the Internet. This encoder implements
//! the core functionality according to RFC 6716.
//!
//! # Features
//!
//! - **SILK mode**: Optimized for speech (narrowband to wideband)
//! - **CELT mode**: Optimized for music (narrowband to fullband)
//! - **Hybrid mode**: Combines SILK and CELT for mixed content
//! - **Adaptive bandwidth**: 4 kHz (narrowband) to 20 kHz (fullband)
//! - **Configurable bitrate**: From 6 kbps to 510 kbps
//! - **Low latency**: Frame sizes from 2.5ms to 60ms
//!
//! # Example
//!
//! ```ignore
//! use oximedia_codec::opus::{OpusEncoder, OpusEncoderConfig};
//!
//! let config = OpusEncoderConfig::new(48000, 2, 64000);
//! let mut encoder = OpusEncoder::new(config)?;
//!
//! let packet = encoder.encode(&audio_samples)?;
//! ```
//!
//! # References
//!
//! - RFC 6716: Definition of the Opus Audio Codec
//! - <https://opus-codec.org/>

use crate::{CodecError, CodecResult, SampleFormat};

use super::celt::CeltEncoder;
use super::packet::{OpusBandwidth, OpusMode, TocInfo};
use super::silk::SilkEncoder;
use super::vad::{VadConfig, VadDecision, VoiceActivityDetector};

/// Opus encoder configuration.
#[derive(Debug, Clone)]
pub struct OpusEncoderConfig {
    /// Sample rate in Hz (8000, 12000, 16000, 24000, or 48000)
    pub sample_rate: u32,
    /// Number of channels (1 or 2)
    pub channels: usize,
    /// Target bitrate in bits per second
    pub bitrate: u32,
    /// Frame duration in milliseconds (2.5, 5, 10, 20, 40, or 60)
    pub frame_duration_ms: f32,
    /// Operating mode (auto-detect by default)
    pub mode: Option<OpusMode>,
    /// Bandwidth (auto-detect by default)
    pub bandwidth: Option<OpusBandwidth>,
    /// Complexity (0-10, higher = better quality but slower)
    pub complexity: u32,
    /// Enable variable bitrate
    pub vbr: bool,
    /// Constrained VBR mode
    pub cvbr: bool,
    /// Enable discontinuous transmission (DTX)
    pub dtx: bool,
    /// Input sample format
    pub sample_format: SampleFormat,
}

impl OpusEncoderConfig {
    /// Creates a new encoder configuration.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz (8000, 12000, 16000, 24000, or 48000)
    /// * `channels` - Number of channels (1 or 2)
    /// * `bitrate` - Target bitrate in bits per second
    pub fn new(sample_rate: u32, channels: usize, bitrate: u32) -> Self {
        Self {
            sample_rate,
            channels,
            bitrate,
            frame_duration_ms: 20.0,
            mode: None,
            bandwidth: None,
            complexity: 5,
            vbr: true,
            cvbr: false,
            dtx: false,
            sample_format: SampleFormat::F32,
        }
    }

    /// Sets the frame duration.
    #[must_use]
    pub fn with_frame_duration(mut self, duration_ms: f32) -> Self {
        self.frame_duration_ms = duration_ms;
        self
    }

    /// Sets the operating mode.
    #[must_use]
    pub fn with_mode(mut self, mode: OpusMode) -> Self {
        self.mode = Some(mode);
        self
    }

    /// Sets the bandwidth.
    #[must_use]
    pub fn with_bandwidth(mut self, bandwidth: OpusBandwidth) -> Self {
        self.bandwidth = Some(bandwidth);
        self
    }

    /// Sets the complexity level (0-10).
    #[must_use]
    pub fn with_complexity(mut self, complexity: u32) -> Self {
        self.complexity = complexity.min(10);
        self
    }

    /// Enables or disables VBR.
    #[must_use]
    pub fn with_vbr(mut self, vbr: bool) -> Self {
        self.vbr = vbr;
        self
    }

    /// Enables or disables DTX.
    #[must_use]
    pub fn with_dtx(mut self, dtx: bool) -> Self {
        self.dtx = dtx;
        self
    }
}

impl Default for OpusEncoderConfig {
    fn default() -> Self {
        Self::new(48000, 2, 64000)
    }
}

/// Opus audio encoder.
///
/// Encodes PCM audio samples to Opus-compressed packets.
pub struct OpusEncoder {
    /// Configuration
    config: OpusEncoderConfig,
    /// SILK encoder (for speech mode)
    silk: Option<SilkEncoder>,
    /// CELT encoder (for music mode)
    celt: Option<CeltEncoder>,
    /// Current operating mode
    current_mode: OpusMode,
    /// Frame size in samples (at encoder sample rate)
    frame_size: usize,
    /// Frame counter
    frame_count: u64,
    /// Input buffer for partial frames
    input_buffer: Vec<f32>,
    /// Number of samples in input buffer
    buffered_samples: usize,
    /// Voice Activity Detector for DTX / bandwidth switching support.
    vad: VoiceActivityDetector,
    /// VAD decision for the last encoded frame.
    last_vad_decision: VadDecision,
    /// Count of consecutive silence frames suppressed by DTX.
    dtx_silence_frames: u32,
}

impl OpusEncoder {
    /// Creates a new Opus encoder.
    ///
    /// # Arguments
    ///
    /// * `config` - Encoder configuration
    pub fn new(config: OpusEncoderConfig) -> CodecResult<Self> {
        // Validate sample rate
        if !matches!(config.sample_rate, 8000 | 12000 | 16000 | 24000 | 48000) {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid sample rate: {}",
                config.sample_rate
            )));
        }

        // Validate channels
        if config.channels == 0 || config.channels > 2 {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid channel count: {}",
                config.channels
            )));
        }

        // Validate bitrate
        if config.bitrate < 6000 || config.bitrate > 510000 {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid bitrate: {} (must be 6000-510000)",
                config.bitrate
            )));
        }

        // Calculate frame size
        let frame_size = Self::calculate_frame_size(config.sample_rate, config.frame_duration_ms)?;

        // Determine mode and bandwidth
        let mode = config.mode.unwrap_or_else(|| Self::select_mode(&config));
        let bandwidth = config
            .bandwidth
            .unwrap_or_else(|| Self::select_bandwidth(&config));

        // Build VAD configuration appropriate for this encoder.
        let vad_config = VadConfig::default();

        // Create encoder instance
        let mut encoder = Self {
            config,
            silk: None,
            celt: None,
            current_mode: mode,
            frame_size,
            frame_count: 0,
            input_buffer: Vec::new(),
            buffered_samples: 0,
            vad: VoiceActivityDetector::new(vad_config),
            last_vad_decision: VadDecision::Silence,
            dtx_silence_frames: 0,
        };

        // Initialize the appropriate encoder
        encoder.initialize_encoder(mode, bandwidth)?;

        Ok(encoder)
    }

    /// Calculates frame size from sample rate and duration.
    fn calculate_frame_size(sample_rate: u32, duration_ms: f32) -> CodecResult<usize> {
        let frame_size = (sample_rate as f32 * duration_ms / 1000.0) as usize;

        // Validate frame size
        let valid_sizes = match sample_rate {
            48000 => vec![120, 240, 480, 960, 1920, 2880],
            24000 => vec![60, 120, 240, 480, 960, 1440],
            16000 => vec![40, 80, 160, 320, 640, 960],
            12000 => vec![30, 60, 120, 240, 480, 720],
            8000 => vec![20, 40, 80, 160, 320, 480],
            _ => {
                return Err(CodecError::InvalidParameter(
                    "Invalid sample rate".to_string(),
                ))
            }
        };

        if !valid_sizes.contains(&frame_size) {
            return Err(CodecError::InvalidParameter(format!(
                "Invalid frame duration {duration_ms}ms for sample rate {sample_rate}Hz"
            )));
        }

        Ok(frame_size)
    }

    /// Selects the best encoding mode based on configuration.
    fn select_mode(config: &OpusEncoderConfig) -> OpusMode {
        // Use CELT for music (higher bitrates, higher sample rates)
        // Use SILK for speech (lower bitrates, lower sample rates)
        // Use Hybrid for mixed content

        if config.bitrate > 32000 || config.sample_rate >= 24000 {
            OpusMode::Celt
        } else if config.bitrate < 20000 || config.sample_rate <= 16000 {
            OpusMode::Silk
        } else {
            OpusMode::Hybrid
        }
    }

    /// Selects the best bandwidth based on configuration.
    fn select_bandwidth(config: &OpusEncoderConfig) -> OpusBandwidth {
        match config.sample_rate {
            8000 => OpusBandwidth::Narrowband,
            12000 => OpusBandwidth::Mediumband,
            16000 => OpusBandwidth::Wideband,
            24000 => OpusBandwidth::SuperWideband,
            _ => OpusBandwidth::Fullband,
        }
    }

    /// Initializes the appropriate encoder for the given mode.
    fn initialize_encoder(&mut self, mode: OpusMode, bandwidth: OpusBandwidth) -> CodecResult<()> {
        match mode {
            OpusMode::Silk => {
                self.silk = Some(SilkEncoder::new(
                    self.config.sample_rate,
                    self.config.channels,
                    bandwidth,
                ));
            }
            OpusMode::Celt => {
                self.celt = Some(CeltEncoder::new(
                    self.config.sample_rate,
                    self.config.channels,
                    bandwidth,
                    self.frame_size,
                ));
            }
            OpusMode::Hybrid => {
                // For hybrid mode, we need both encoders
                self.silk = Some(SilkEncoder::new(
                    self.config.sample_rate,
                    self.config.channels,
                    bandwidth,
                ));
                self.celt = Some(CeltEncoder::new(
                    self.config.sample_rate,
                    self.config.channels,
                    bandwidth,
                    self.frame_size,
                ));
            }
        }

        Ok(())
    }

    /// Encodes a frame of audio samples.
    ///
    /// # Arguments
    ///
    /// * `samples` - Input audio samples (f32 format, interleaved if multi-channel)
    ///
    /// # Returns
    ///
    /// Compressed Opus packet, or None if more samples are needed
    pub fn encode(&mut self, samples: &[f32]) -> CodecResult<Option<Vec<u8>>> {
        // Add samples to buffer
        self.input_buffer.extend_from_slice(samples);
        self.buffered_samples += samples.len() / self.config.channels;

        // Check if we have enough samples for a frame
        if self.buffered_samples < self.frame_size {
            return Ok(None);
        }

        // Extract one frame worth of samples
        let frame_samples = self.frame_size * self.config.channels;
        let frame_data: Vec<f32> = self.input_buffer.drain(..frame_samples).collect();
        self.buffered_samples -= self.frame_size;

        // Encode the frame
        let packet = self.encode_frame(&frame_data)?;
        self.frame_count += 1;

        Ok(Some(packet))
    }

    /// Encodes a single frame.
    fn encode_frame(&mut self, samples: &[f32]) -> CodecResult<Vec<u8>> {
        // ── Voice Activity Detection ─────────────────────────────────────────
        // Down-mix to mono for VAD analysis.
        let channels = self.config.channels.max(1);
        let mono: Vec<f32> = samples
            .chunks(channels)
            .map(|ch| ch.iter().copied().sum::<f32>() / channels as f32)
            .collect();
        self.last_vad_decision = self.vad.process_f32(&mono, self.config.sample_rate);

        // DTX: suppress consecutive silence frames after the first one.
        if self.config.dtx && self.last_vad_decision == VadDecision::Silence {
            self.dtx_silence_frames += 1;
            if self.dtx_silence_frames > 1 {
                return Ok(Vec::new());
            }
        } else {
            self.dtx_silence_frames = 0;
        }

        // Allocate output buffer (max packet size)
        let max_packet_size = 1275; // RFC 6716 maximum
        let mut packet = vec![0u8; max_packet_size];

        // Generate TOC byte
        let toc_byte = self.generate_toc_byte()?;
        packet[0] = toc_byte;

        // Encode frame data based on mode
        let frame_size = match self.current_mode {
            OpusMode::Silk => {
                if let Some(silk) = &mut self.silk {
                    let bytes = silk.encode(samples, &mut packet[1..], self.frame_size)?;
                    bytes + 1
                } else {
                    return Err(CodecError::Internal(
                        "SILK encoder not initialized".to_string(),
                    ));
                }
            }
            OpusMode::Celt => {
                if let Some(celt) = &mut self.celt {
                    let bytes = celt.encode(samples, &mut packet[1..], self.frame_size)?;
                    bytes + 1
                } else {
                    return Err(CodecError::Internal(
                        "CELT encoder not initialized".to_string(),
                    ));
                }
            }
            OpusMode::Hybrid => {
                // For hybrid mode, encode with both and combine
                // This is a simplified implementation
                if let (Some(silk), Some(celt)) = (&mut self.silk, &mut self.celt) {
                    let mut silk_data = vec![0u8; max_packet_size / 2];
                    let mut celt_data = vec![0u8; max_packet_size / 2];

                    let silk_bytes = silk.encode(samples, &mut silk_data, self.frame_size)?;
                    let celt_bytes = celt.encode(samples, &mut celt_data, self.frame_size)?;

                    // Combine SILK and CELT data
                    // In real implementation, this would be more sophisticated
                    let total_bytes = 1 + silk_bytes + celt_bytes;
                    if total_bytes > max_packet_size {
                        return Err(CodecError::BufferTooSmall {
                            needed: total_bytes,
                            have: max_packet_size,
                        });
                    }

                    packet[1..1 + silk_bytes].copy_from_slice(&silk_data[..silk_bytes]);
                    packet[1 + silk_bytes..1 + silk_bytes + celt_bytes]
                        .copy_from_slice(&celt_data[..celt_bytes]);

                    total_bytes
                } else {
                    return Err(CodecError::Internal(
                        "Hybrid encoders not initialized".to_string(),
                    ));
                }
            }
        };

        // Truncate to actual size
        packet.truncate(frame_size);

        Ok(packet)
    }

    /// Generates the TOC (Table of Contents) byte for the packet.
    fn generate_toc_byte(&self) -> CodecResult<u8> {
        // TOC byte format:
        // - Bits 7-3: Configuration (mode, bandwidth, frame size)
        // - Bit 2: Stereo flag
        // - Bits 1-0: Frame count code (0 = single frame)

        let config = self.encode_configuration()?;
        let stereo_flag = if self.config.channels == 2 {
            0x04
        } else {
            0x00
        };
        let frame_code = 0x00; // Single frame

        Ok((config << 3) | stereo_flag | frame_code)
    }

    /// Encodes mode, bandwidth, and frame size into configuration value.
    fn encode_configuration(&self) -> CodecResult<u8> {
        let bandwidth = self
            .config
            .bandwidth
            .unwrap_or_else(|| Self::select_bandwidth(&self.config));

        let config = match self.current_mode {
            OpusMode::Silk => {
                // SILK: config 0-11
                let bw_code = match bandwidth {
                    OpusBandwidth::Narrowband => 0,
                    OpusBandwidth::Mediumband => 1,
                    OpusBandwidth::Wideband => 2,
                    _ => 3,
                };

                let frame_code = match self.frame_size {
                    480 => 0,  // 10ms at 48kHz
                    960 => 1,  // 20ms at 48kHz
                    1920 => 2, // 40ms at 48kHz
                    2880 => 3, // 60ms at 48kHz
                    _ => 1,    // Default to 20ms
                };

                (bw_code << 2) | frame_code
            }
            OpusMode::Hybrid => {
                // Hybrid: config 12-15
                let bw_code = match bandwidth {
                    OpusBandwidth::SuperWideband => 0,
                    OpusBandwidth::Fullband => 1,
                    _ => 0,
                };
                12 + bw_code
            }
            OpusMode::Celt => {
                // CELT: config 16-31
                let bw_code = match bandwidth {
                    OpusBandwidth::Narrowband => 0,
                    OpusBandwidth::Mediumband => 1,
                    OpusBandwidth::Wideband => 2,
                    OpusBandwidth::SuperWideband => 3,
                    OpusBandwidth::Fullband => 4,
                };

                let frame_code = match self.frame_size {
                    120 => 0, // 2.5ms at 48kHz
                    240 => 1, // 5ms at 48kHz
                    480 => 2, // 10ms at 48kHz
                    960 => 3, // 20ms at 48kHz
                    _ => 2,   // Default to 10ms
                };

                (16 + (bw_code << 2)) | frame_code
            }
        };

        Ok(config)
    }

    /// Flushes any buffered samples.
    ///
    /// This will pad the last frame with silence if needed and encode it.
    pub fn flush(&mut self) -> CodecResult<Option<Vec<u8>>> {
        if self.buffered_samples == 0 {
            return Ok(None);
        }

        // Pad with silence to complete the frame
        let needed_samples = (self.frame_size - self.buffered_samples) * self.config.channels;
        self.input_buffer.extend(vec![0.0f32; needed_samples]);
        self.buffered_samples = self.frame_size;

        // Encode the final frame
        let frame_samples = self.frame_size * self.config.channels;
        let frame_data: Vec<f32> = self.input_buffer.drain(..frame_samples).collect();
        self.buffered_samples = 0;

        let packet = self.encode_frame(&frame_data)?;
        Ok(Some(packet))
    }

    /// Resets encoder state.
    pub fn reset(&mut self) {
        if let Some(silk) = &mut self.silk {
            silk.reset();
        }
        if let Some(celt) = &mut self.celt {
            celt.reset();
        }
        self.frame_count = 0;
        self.input_buffer.clear();
        self.buffered_samples = 0;
        self.vad.reset();
        self.last_vad_decision = VadDecision::Silence;
        self.dtx_silence_frames = 0;
    }

    /// Returns the VAD decision for the most recently encoded frame.
    #[must_use]
    pub const fn last_vad_decision(&self) -> VadDecision {
        self.last_vad_decision
    }

    /// Returns the number of consecutive silence frames suppressed by DTX.
    #[must_use]
    pub const fn dtx_silence_frames(&self) -> u32 {
        self.dtx_silence_frames
    }

    /// Returns the current configuration.
    #[must_use]
    pub const fn config(&self) -> &OpusEncoderConfig {
        &self.config
    }

    /// Returns the number of frames encoded.
    #[must_use]
    pub const fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Returns the current operating mode.
    #[must_use]
    pub const fn current_mode(&self) -> OpusMode {
        self.current_mode
    }

    /// Returns the frame size in samples.
    #[must_use]
    pub const fn frame_size(&self) -> usize {
        self.frame_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_creation() {
        let config = OpusEncoderConfig::new(48000, 2, 64000);
        let encoder = OpusEncoder::new(config);
        assert!(encoder.is_ok());
    }

    #[test]
    fn test_encoder_invalid_sample_rate() {
        let config = OpusEncoderConfig::new(44100, 2, 64000);
        let encoder = OpusEncoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_encoder_invalid_channels() {
        let config = OpusEncoderConfig::new(48000, 0, 64000);
        let encoder = OpusEncoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_encoder_invalid_bitrate() {
        let config = OpusEncoderConfig::new(48000, 2, 1000);
        let encoder = OpusEncoder::new(config);
        assert!(encoder.is_err());
    }

    #[test]
    fn test_config_builder() {
        let config = OpusEncoderConfig::new(48000, 2, 64000)
            .with_frame_duration(10.0)
            .with_complexity(8)
            .with_vbr(true);

        assert_eq!(config.sample_rate, 48000);
        assert_eq!(config.channels, 2);
        assert_eq!(config.bitrate, 64000);
        assert!((config.frame_duration_ms - 10.0).abs() < f32::EPSILON);
        assert_eq!(config.complexity, 8);
        assert!(config.vbr);
    }

    #[test]
    fn test_mode_selection() {
        let config_speech = OpusEncoderConfig::new(16000, 1, 16000);
        assert_eq!(OpusEncoder::select_mode(&config_speech), OpusMode::Silk);

        let config_music = OpusEncoderConfig::new(48000, 2, 64000);
        assert_eq!(OpusEncoder::select_mode(&config_music), OpusMode::Celt);
    }

    #[test]
    fn test_bandwidth_selection() {
        let config_nb = OpusEncoderConfig::new(8000, 1, 16000);
        assert_eq!(
            OpusEncoder::select_bandwidth(&config_nb),
            OpusBandwidth::Narrowband
        );

        let config_fb = OpusEncoderConfig::new(48000, 2, 64000);
        assert_eq!(
            OpusEncoder::select_bandwidth(&config_fb),
            OpusBandwidth::Fullband
        );
    }

    #[test]
    fn test_frame_size_calculation() {
        let size = OpusEncoder::calculate_frame_size(48000, 20.0);
        assert!(size.is_ok());
        assert_eq!(size.expect("should succeed"), 960);

        let size = OpusEncoder::calculate_frame_size(48000, 10.0);
        assert!(size.is_ok());
        assert_eq!(size.expect("should succeed"), 480);
    }

    #[test]
    fn test_encoder_reset() {
        let config = OpusEncoderConfig::new(48000, 2, 64000);
        let mut encoder = OpusEncoder::new(config).expect("should succeed");
        encoder.reset();
        assert_eq!(encoder.frame_count(), 0);
    }

    // =========================================================================
    // VAD integration tests
    // =========================================================================

    fn speech_frame_f32(len: usize, sample_rate: u32) -> Vec<f32> {
        (0..len)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                (2.0 * std::f32::consts::PI * 200.0 * t).sin() * 0.8
            })
            .collect()
    }

    #[test]
    fn test_vad_initial_decision_is_silence() {
        let config = OpusEncoderConfig::new(16000, 1, 16000);
        let encoder = OpusEncoder::new(config).expect("encoder creation");
        assert_eq!(encoder.last_vad_decision(), VadDecision::Silence);
    }

    #[test]
    fn test_vad_dtx_silence_frames_zero_initially() {
        let config = OpusEncoderConfig::new(16000, 1, 16000);
        let encoder = OpusEncoder::new(config).expect("encoder creation");
        assert_eq!(encoder.dtx_silence_frames(), 0);
    }

    #[test]
    fn test_vad_decision_updates_after_encode() {
        let config = OpusEncoderConfig::new(16000, 1, 16000);
        let mut encoder = OpusEncoder::new(config).expect("encoder creation");
        let frame_size = encoder.frame_size();
        for _ in 0..10 {
            let silence = vec![0.0f32; frame_size];
            let _ = encoder.encode(&silence);
        }
        let speech = speech_frame_f32(frame_size, 16000);
        let _ = encoder.encode(&speech);
        assert_eq!(
            encoder.last_vad_decision(),
            VadDecision::Voice,
            "Loud speech should produce Voice decision"
        );
    }

    #[test]
    fn test_vad_reset_clears_decision() {
        let config = OpusEncoderConfig::new(16000, 1, 16000);
        let mut encoder = OpusEncoder::new(config).expect("encoder creation");
        let frame_size = encoder.frame_size();
        for _ in 0..5 {
            let _ = encoder.encode(&vec![0.0f32; frame_size]);
        }
        encoder.reset();
        assert_eq!(encoder.last_vad_decision(), VadDecision::Silence);
        assert_eq!(encoder.dtx_silence_frames(), 0);
    }

    #[test]
    fn test_dtx_suppresses_continuous_silence() {
        let config = OpusEncoderConfig::new(16000, 1, 16000).with_dtx(true);
        let mut encoder = OpusEncoder::new(config).expect("encoder creation");
        let frame_size = encoder.frame_size();
        let silence = vec![0.0f32; frame_size];
        let mut suppressed = 0u32;
        for _ in 0..40 {
            if let Ok(Some(pkt)) = encoder.encode(&silence) {
                if pkt.is_empty() {
                    suppressed += 1;
                }
            }
        }
        assert!(
            suppressed > 0,
            "DTX must suppress at least one silence frame over 40 frames"
        );
    }

    #[test]
    fn test_dtx_does_not_suppress_speech() {
        let config = OpusEncoderConfig::new(16000, 1, 16000).with_dtx(true);
        let mut encoder = OpusEncoder::new(config).expect("encoder creation");
        let frame_size = encoder.frame_size();
        for _ in 0..10 {
            let _ = encoder.encode(&vec![0.0f32; frame_size]);
        }
        let speech = speech_frame_f32(frame_size, 16000);
        if let Ok(Some(pkt)) = encoder.encode(&speech) {
            assert!(!pkt.is_empty(), "DTX must NOT suppress speech frames");
        }
    }

    #[test]
    fn test_dtx_disabled_never_suppresses() {
        let config = OpusEncoderConfig::new(16000, 1, 16000);
        assert!(!config.dtx);
        let mut encoder = OpusEncoder::new(config).expect("encoder creation");
        let frame_size = encoder.frame_size();
        let silence = vec![0.0f32; frame_size];
        for _ in 0..30 {
            if let Ok(Some(pkt)) = encoder.encode(&silence) {
                assert!(
                    !pkt.is_empty(),
                    "Without DTX, no packets should be suppressed"
                );
            }
        }
    }

    #[test]
    fn test_dtx_silence_frame_counter_increases() {
        let config = OpusEncoderConfig::new(16000, 1, 16000).with_dtx(true);
        let mut encoder = OpusEncoder::new(config).expect("encoder creation");
        let frame_size = encoder.frame_size();
        let silence = vec![0.0f32; frame_size];
        for _ in 0..40 {
            let _ = encoder.encode(&silence);
        }
        assert!(
            encoder.dtx_silence_frames() > 0,
            "dtx_silence_frames must be > 0 after sustained silence with DTX"
        );
    }
}
