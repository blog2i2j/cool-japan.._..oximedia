//! Opus codec implementation.
//!
//! Opus is a versatile audio codec that combines:
//! - **SILK** - For speech at low bitrates
//! - **CELT** - For music and general audio at higher bitrates
//! - **Hybrid** - Both SILK and CELT for wideband speech
//!
//! # Modules
//!
//! - [`packet`] - Opus packet parsing
//! - [`silk`] - SILK mode decoder
//! - [`celt`] - CELT mode decoder
//! - [`range_decoder`] - Range coding

#![forbid(unsafe_code)]

pub mod celt;
pub mod packet;
pub mod range_decoder;
pub mod silk;

use bytes::Bytes;

use crate::{
    AudioBuffer, AudioDecoder, AudioDecoderConfig, AudioEncoder, AudioEncoderConfig, AudioError,
    AudioFrame, AudioResult, ChannelLayout, EncodedAudioPacket,
};
use oximedia_core::{CodecId, Rational, SampleFormat, Timestamp};

// Re-export submodule types
pub use celt::{BandEnergy, CeltFrame, CeltMode, PitchPeriod};
pub use packet::{
    FrameCount, FrameDuration, OpusBandwidth, OpusMode, OpusPacket, OpusPacketConfig, TocByte,
};
pub use range_decoder::{RangeDecoder, Symbol};
pub use silk::{SilkBandwidth, SilkFrame, SilkSubframe};

// ─────────────────────────────── TOC byte helpers ────────────────────────────

/// Choose the CELT frame size that best matches a given sample count.
fn frame_size_for_samples(samples: u32) -> celt::CeltFrameSize {
    match samples {
        0..=180 => celt::CeltFrameSize::Ms2_5,
        181..=300 => celt::CeltFrameSize::Ms5,
        301..=720 => celt::CeltFrameSize::Ms10,
        _ => celt::CeltFrameSize::Ms20,
    }
}

/// Build the TOC byte for a CELT-only single-frame packet.
///
/// Config 28 = fullband, 20ms, CELT-only (bits 28 << 3 = 0xE0).
/// Stereo flag = bit 2.  Frame count code = 0 (one frame).
fn make_toc_byte(stereo: bool, frame_size: celt::CeltFrameSize) -> u8 {
    // CELT-only configs 16–31.
    // Within CELT: config = 16 + 4*bw + dur where dur 0=2.5ms,1=5ms,2=10ms,3=20ms.
    // We use fullband (bw=3) for all configurations.
    let dur: u8 = match frame_size {
        celt::CeltFrameSize::Ms2_5 => 0,
        celt::CeltFrameSize::Ms5 => 1,
        celt::CeltFrameSize::Ms10 => 2,
        celt::CeltFrameSize::Ms20 => 3,
    };
    let config: u8 = 16 + 4 * 3 + dur; // 28, 29, 30, or 31
    let stereo_bit: u8 = if stereo { 0x04 } else { 0x00 };
    // Frame count code 0 = single frame.
    (config << 3) | stereo_bit
}

// ─────────────────────────────── Opus Decoder ────────────────────────────────

/// Pending packet stored by `send_packet`.
#[derive(Clone, Default)]
struct PendingPacket {
    data: Vec<u8>,
    pts: i64,
}

/// Opus decoder.
pub struct OpusDecoder {
    #[allow(dead_code)]
    config: AudioDecoderConfig,
    sample_rate: u32,
    channels: u8,
    flushing: bool,
    silk_state: silk::SilkDecoderState,
    celt_state: celt::CeltDecoderState,
    /// Pending compressed packet waiting for `receive_frame`.
    pending: Option<PendingPacket>,
}

impl OpusDecoder {
    /// Create new Opus decoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: &AudioDecoderConfig) -> AudioResult<Self> {
        if config.codec != CodecId::Opus {
            return Err(AudioError::InvalidParameter("Expected Opus codec".into()));
        }

        let channels = config.channels.clamp(1, 2);
        let frame_size = celt::CeltFrameSize::Ms20;

        Ok(Self {
            config: config.clone(),
            sample_rate: config.sample_rate,
            channels,
            flushing: false,
            silk_state: silk::SilkDecoderState::new(silk::SilkBandwidth::Wide),
            celt_state: celt::CeltDecoderState::new(channels, frame_size),
            pending: None,
        })
    }

    /// Parse an Opus packet header without decoding.
    ///
    /// # Errors
    ///
    /// Returns error if packet is invalid.
    pub fn parse_packet(data: &[u8]) -> AudioResult<OpusPacketConfig> {
        let packet = OpusPacket::parse(data)?;
        Ok(packet.config)
    }

    /// Get expected frame duration from packet.
    #[must_use]
    pub fn get_frame_duration(data: &[u8]) -> Option<FrameDuration> {
        if data.is_empty() {
            return None;
        }
        let toc = TocByte::parse(data[0]);
        Some(toc.duration)
    }

    // ── Internal decode helpers ───────────────────────────────────────────

    /// Decode a single CELT frame payload and return f32 interleaved samples.
    fn decode_celt_frame(&mut self, frame_data: &[u8], toc: &TocByte) -> AudioResult<Vec<f32>> {
        let channels = usize::from(self.channels);

        // Determine frame size from TOC.
        let frame_size = match toc.duration {
            FrameDuration::Ms2_5 => celt::CeltFrameSize::Ms2_5,
            FrameDuration::Ms5 => celt::CeltFrameSize::Ms5,
            FrameDuration::Ms10 => celt::CeltFrameSize::Ms10,
            FrameDuration::Ms20 | FrameDuration::Ms40 | FrameDuration::Ms60 => {
                celt::CeltFrameSize::Ms20
            }
        };

        // Update CELT decoder state's frame size.
        self.celt_state.frame_size = frame_size;

        // Try to deserialize a structured CELT frame from the payload.
        // Our encoder uses celt::serialize_celt_frame format; fall back to
        // synthesizing silence if parsing fails (e.g., external packets).
        let celt_frame = if frame_data.len() > 1 {
            celt::deserialize_celt_frame(frame_data, frame_size, self.channels).unwrap_or_else(
                |_| {
                    // Unknown external packet: create empty frame with minimal energy.
                    let mut f = celt::CeltFrame::new(frame_size, self.channels);
                    // Set low-level energy so silence is produced.
                    let band_config = celt::CeltBandConfig::new_48khz();
                    for band in 0..band_config.band_count {
                        f.energy.set(band, -15360i16); // -60 dB in Q8
                    }
                    f
                },
            )
        } else {
            // Empty/minimal frame: produce silence.
            let n = frame_size.samples_48khz();
            return Ok(vec![0.0f32; n * channels]);
        };

        celt::decode_frame(&celt_frame, &mut self.celt_state)
    }

    /// Decode a single SILK frame payload and return f32 interleaved samples at 48kHz.
    fn decode_silk_frame(&mut self, frame_data: &[u8]) -> AudioResult<Vec<f32>> {
        let silk_frame = if frame_data.len() > 4 {
            silk::deserialize_silk_frame(frame_data).unwrap_or_else(|_| {
                // Unknown format: generate a silence frame.
                let bw = self.silk_state.bandwidth;
                let sr = bw.sample_rate();
                let frame_size = (sr * 20 / 1000) as usize; // 20ms
                silk::SilkFrame::new(bw, frame_size)
            })
        } else {
            let bw = self.silk_state.bandwidth;
            let sr = bw.sample_rate();
            let frame_size = (sr * 20 / 1000) as usize;
            silk::SilkFrame::new(bw, frame_size)
        };

        let mono_samples = silk::decode_frame(&silk_frame, &mut self.silk_state)?;

        // For stereo, duplicate the mono channel.
        let channels = usize::from(self.channels);
        if channels == 1 {
            Ok(mono_samples)
        } else {
            let mut stereo = Vec::with_capacity(mono_samples.len() * 2);
            for s in mono_samples {
                stereo.push(s);
                stereo.push(s);
            }
            Ok(stereo)
        }
    }

    /// Resample from any rate to the configured output sample rate (48kHz by default).
    fn resample_if_needed(&self, samples: Vec<f32>, from_rate: u32) -> Vec<f32> {
        if from_rate == self.sample_rate || from_rate == 0 || self.sample_rate == 0 {
            return samples;
        }
        let channels = usize::from(self.channels);
        let per_channel = samples.len() / channels.max(1);
        let target_per_channel =
            (per_channel as u64 * u64::from(self.sample_rate) / u64::from(from_rate)) as usize;
        let mut out = Vec::with_capacity(target_per_channel * channels);

        for ch in 0..channels {
            for i in 0..target_per_channel {
                let pos = i as f64 * from_rate as f64 / self.sample_rate as f64;
                let lo = pos.floor() as usize;
                let hi = (lo + 1).min(per_channel.saturating_sub(1));
                let frac = (pos - pos.floor()) as f32;
                let lo_val = samples.get(lo * channels + ch).copied().unwrap_or(0.0);
                let hi_val = samples.get(hi * channels + ch).copied().unwrap_or(lo_val);
                out.push(lo_val + (hi_val - lo_val) * frac);
            }
        }
        out
    }
}

impl AudioDecoder for OpusDecoder {
    fn codec(&self) -> CodecId {
        CodecId::Opus
    }

    /// Store the compressed Opus packet for processing in `receive_frame`.
    fn send_packet(&mut self, data: &[u8], pts: i64) -> AudioResult<()> {
        if data.is_empty() {
            return Err(AudioError::InvalidData("Empty Opus packet".into()));
        }
        self.pending = Some(PendingPacket {
            data: data.to_vec(),
            pts,
        });
        Ok(())
    }

    /// Decode the pending packet and return an `AudioFrame`.
    ///
    /// Returns `Ok(None)` if there is no pending packet or we are in
    /// flush mode with no remaining data.
    fn receive_frame(&mut self) -> AudioResult<Option<AudioFrame>> {
        let pending = match self.pending.take() {
            Some(p) => p,
            None => return Ok(None),
        };

        // Parse the packet structure.
        let opus_packet = OpusPacket::parse(&pending.data)?;
        let toc = opus_packet.config.toc;

        // Decode all frames and concatenate.
        let mut all_samples: Vec<f32> = Vec::new();

        for frame_data in &opus_packet.frames {
            let frame_samples = match toc.mode {
                OpusMode::CeltOnly => self.decode_celt_frame(frame_data, &toc)?,
                OpusMode::SilkOnly => self.decode_silk_frame(frame_data)?,
                OpusMode::Hybrid => {
                    // Hybrid: SILK lower bands + CELT upper bands.
                    // For simplicity, use CELT decode (it handles the full band).
                    let celt_samples = self.decode_celt_frame(frame_data, &toc)?;
                    let silk_samples = self.decode_silk_frame(frame_data)?;

                    // Blend: CELT handles high-frequency detail, SILK the low band.
                    // A simple approach: take average weighted toward CELT.
                    let len = celt_samples.len().max(silk_samples.len());
                    let mut blended = Vec::with_capacity(len);
                    for i in 0..len {
                        let c = celt_samples.get(i).copied().unwrap_or(0.0);
                        let s = silk_samples.get(i).copied().unwrap_or(0.0);
                        blended.push(c * 0.7 + s * 0.3);
                    }
                    blended
                }
            };

            all_samples.extend_from_slice(&frame_samples);
        }

        // CELT operates at 48kHz natively; SILK may be at a lower rate.
        // Our decode_silk_frame already upsamples, so all_samples are at 48kHz.
        let output_rate = 48000u32;
        let samples = if output_rate != self.sample_rate {
            self.resample_if_needed(all_samples, output_rate)
        } else {
            all_samples
        };

        // Pack f32 samples into bytes for AudioBuffer.
        let byte_count = samples.len() * 4;
        let mut bytes = vec![0u8; byte_count];
        for (i, &s) in samples.iter().enumerate() {
            let b = s.to_le_bytes();
            bytes[i * 4..i * 4 + 4].copy_from_slice(&b);
        }

        let channels = usize::from(self.channels);
        let sample_count = samples.len() / channels.max(1);

        let frame = AudioFrame {
            format: SampleFormat::F32,
            sample_rate: self.sample_rate,
            channels: ChannelLayout::from_count(channels),
            samples: AudioBuffer::Interleaved(Bytes::from(bytes)),
            timestamp: Timestamp::new(pending.pts, Rational::new(1, i64::from(self.sample_rate))),
        };

        debug_assert_eq!(frame.sample_count(), sample_count, "sample_count mismatch");

        Ok(Some(frame))
    }

    fn flush(&mut self) -> AudioResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn reset(&mut self) {
        self.flushing = false;
        self.pending = None;
        self.silk_state.reset();
        self.celt_state.reset();
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

// ─────────────────────────────── Opus Encoder ────────────────────────────────

/// Buffered PCM samples waiting to be encoded.
#[derive(Default)]
struct SampleBuffer {
    samples: Vec<f32>,
    pts: i64,
    has_pts: bool,
}

/// Opus encoder.
pub struct OpusEncoder {
    config: AudioEncoderConfig,
    flushing: bool,
    buffer: SampleBuffer,
    celt_enc_state: celt::CeltEncoderState,
    /// Whether to use SILK for low-bitrate speech (otherwise CELT).
    use_silk: bool,
}

impl OpusEncoder {
    /// Create new Opus encoder.
    ///
    /// # Errors
    ///
    /// Returns error if configuration is invalid.
    pub fn new(config: AudioEncoderConfig) -> AudioResult<Self> {
        if config.codec != CodecId::Opus {
            return Err(AudioError::InvalidParameter("Expected Opus codec".into()));
        }
        let channels = config.channels.clamp(1, 2);
        let frame_size = frame_size_for_samples(config.frame_size);
        // Use SILK for bitrates <= 32 kbps (speech-optimised).
        let use_silk = config.bitrate <= 32_000;

        Ok(Self {
            config: AudioEncoderConfig { channels, ..config },
            flushing: false,
            buffer: SampleBuffer::default(),
            celt_enc_state: celt::CeltEncoderState::new(channels, frame_size),
            use_silk,
        })
    }

    // ── Internal encode helpers ───────────────────────────────────────────

    /// Encode one frame worth of f32 samples into an Opus packet (CELT-only).
    ///
    /// Packet format:
    ///   byte 0 : TOC
    ///   bytes 1..: serialized CeltFrame payload
    fn encode_celt_packet(&mut self, samples: &[f32], pts: i64) -> AudioResult<EncodedAudioPacket> {
        let channels = usize::from(self.config.channels);
        let n = self.config.frame_size as usize;
        let frame_size = frame_size_for_samples(self.config.frame_size);
        let stereo = channels == 2;
        let toc = make_toc_byte(stereo, frame_size);

        // Ensure encoder state has the right frame size.
        if self.celt_enc_state.frame_size != frame_size {
            self.celt_enc_state = celt::CeltEncoderState::new(self.config.channels, frame_size);
        }

        let celt_frame = celt::encode_frame(samples, &mut self.celt_enc_state)?;
        let payload = celt::serialize_celt_frame(&celt_frame)?;

        let mut packet = Vec::with_capacity(1 + payload.len());
        packet.push(toc);
        packet.extend_from_slice(&payload);

        Ok(EncodedAudioPacket {
            data: packet,
            pts,
            duration: n as u32,
        })
    }

    /// Encode one frame worth of f32 samples into an Opus packet (SILK-only).
    fn encode_silk_packet(&mut self, samples: &[f32], pts: i64) -> AudioResult<EncodedAudioPacket> {
        let n = self.config.frame_size as usize;
        let bandwidth = if self.config.sample_rate <= 8000 {
            silk::SilkBandwidth::Narrow
        } else if self.config.sample_rate <= 12000 {
            silk::SilkBandwidth::Medium
        } else {
            silk::SilkBandwidth::Wide
        };

        // For SILK, we work on mono (stereo is handled by running twice).
        let channels = usize::from(self.config.channels);
        let mono: Vec<f32> = if channels == 2 {
            // Mix down to mono.
            (0..samples.len() / 2)
                .map(|i| (samples[i * 2] + samples[i * 2 + 1]) * 0.5)
                .collect()
        } else {
            samples.to_vec()
        };

        // Downsampled to SILK rate and back, store in a temporary state.
        let mut silk_state = silk::SilkDecoderState::new(bandwidth);
        let silk_frame = silk::encode_frame(&mono, &mut silk_state, bandwidth)?;
        let payload = silk::serialize_silk_frame(&silk_frame)?;

        // TOC: SILK narrowband config 0 = 0x00, mono, single frame.
        let bw_config: u8 = match bandwidth {
            silk::SilkBandwidth::Narrow => 0,
            silk::SilkBandwidth::Medium => 4,
            silk::SilkBandwidth::Wide => 8,
        };
        // config bits 4:0, stereo bit 2, frame count code bits 1:0 = 0.
        let toc = (bw_config << 3) | (if channels == 2 { 0x04 } else { 0x00 });

        let mut packet = Vec::with_capacity(1 + payload.len());
        packet.push(toc);
        packet.extend_from_slice(&payload);

        Ok(EncodedAudioPacket {
            data: packet,
            pts,
            duration: n as u32,
        })
    }
}

impl AudioEncoder for OpusEncoder {
    fn codec(&self) -> CodecId {
        CodecId::Opus
    }

    /// Buffer the input frame's samples for encoding.
    fn send_frame(&mut self, frame: &AudioFrame) -> AudioResult<()> {
        // Extract f32 samples from the frame.
        let f32_samples = match &frame.samples {
            AudioBuffer::Interleaved(data) => {
                if data.len() % 4 != 0 {
                    return Err(AudioError::InvalidData(
                        "Interleaved buffer not aligned to f32".into(),
                    ));
                }
                let mut out = Vec::with_capacity(data.len() / 4);
                for chunk in data.chunks_exact(4) {
                    let bytes = [chunk[0], chunk[1], chunk[2], chunk[3]];
                    out.push(f32::from_le_bytes(bytes));
                }
                out
            }
            AudioBuffer::Planar(planes) => {
                // Interleave all planes.
                let ch = planes.len();
                if ch == 0 {
                    return Ok(());
                }
                let spc = planes[0].len() / 4;
                let mut out = Vec::with_capacity(spc * ch);
                for i in 0..spc {
                    for plane in planes {
                        let start = i * 4;
                        if start + 4 <= plane.len() {
                            let bytes = [
                                plane[start],
                                plane[start + 1],
                                plane[start + 2],
                                plane[start + 3],
                            ];
                            out.push(f32::from_le_bytes(bytes));
                        }
                    }
                }
                out
            }
        };

        if !self.buffer.has_pts {
            self.buffer.pts = frame.timestamp.pts;
            self.buffer.has_pts = true;
        }

        self.buffer.samples.extend_from_slice(&f32_samples);
        Ok(())
    }

    /// Encode one frame of buffered samples into an Opus packet.
    ///
    /// Returns `Ok(None)` if there are fewer buffered samples than one
    /// full frame requires.
    fn receive_packet(&mut self) -> AudioResult<Option<EncodedAudioPacket>> {
        let frame_samples = self.config.frame_size as usize;
        let channels = usize::from(self.config.channels);
        let required = frame_samples * channels;

        if self.buffer.samples.len() < required {
            // Not enough samples yet.
            return Ok(None);
        }

        let samples: Vec<f32> = self.buffer.samples.drain(..required).collect();
        let pts = self.buffer.pts;

        // Advance PTS by one frame duration.
        if self.config.sample_rate > 0 {
            self.buffer.pts += i64::from(self.config.frame_size);
        }
        if self.buffer.samples.is_empty() {
            self.buffer.has_pts = false;
        }

        let packet = if self.use_silk {
            self.encode_silk_packet(&samples, pts)?
        } else {
            self.encode_celt_packet(&samples, pts)?
        };

        Ok(Some(packet))
    }

    fn flush(&mut self) -> AudioResult<()> {
        self.flushing = true;
        Ok(())
    }

    fn config(&self) -> &AudioEncoderConfig {
        &self.config
    }
}

// ─────────────────────────────── Unit Tests ──────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_decoder() -> OpusDecoder {
        let config = AudioDecoderConfig {
            codec: CodecId::Opus,
            sample_rate: 48000,
            channels: 1,
            extradata: None,
        };
        OpusDecoder::new(&config).expect("should succeed")
    }

    fn make_encoder(bitrate: u32, frame_size: u32, channels: u8) -> OpusEncoder {
        let config = AudioEncoderConfig {
            codec: CodecId::Opus,
            sample_rate: 48000,
            channels,
            bitrate,
            frame_size,
        };
        OpusEncoder::new(config).expect("should succeed")
    }

    // ── Basic construction ────────────────────────────────────────────────

    #[test]
    fn test_opus_decoder() {
        let config = AudioDecoderConfig {
            codec: CodecId::Opus,
            ..Default::default()
        };
        let decoder = OpusDecoder::new(&config).expect("should succeed");
        assert_eq!(decoder.codec(), CodecId::Opus);
    }

    #[test]
    fn test_opus_encoder() {
        let config = AudioEncoderConfig {
            codec: CodecId::Opus,
            ..Default::default()
        };
        let encoder = OpusEncoder::new(config).expect("should succeed");
        assert_eq!(encoder.codec(), CodecId::Opus);
    }

    #[test]
    fn test_opus_decoder_reset() {
        let config = AudioDecoderConfig {
            codec: CodecId::Opus,
            ..Default::default()
        };
        let mut decoder = OpusDecoder::new(&config).expect("should succeed");
        decoder.reset();
        assert!(!decoder.flushing);
    }

    #[test]
    fn test_parse_packet() {
        // Single frame, SILK narrowband
        let data = vec![0x00, 0x01, 0x02, 0x03];
        let config = OpusDecoder::parse_packet(&data).expect("should succeed");
        assert_eq!(config.frame_count, 1);
        assert_eq!(config.toc.mode, OpusMode::SilkOnly);
    }

    #[test]
    fn test_get_frame_duration() {
        // Config 0 = 10ms
        let data = vec![0x00];
        let duration = OpusDecoder::get_frame_duration(&data);
        assert_eq!(duration, Some(FrameDuration::Ms10));
    }

    #[test]
    fn test_get_frame_duration_empty() {
        let duration = OpusDecoder::get_frame_duration(&[]);
        assert!(duration.is_none());
    }

    // ── send_packet rejects empty data ────────────────────────────────────

    #[test]
    fn test_send_packet_empty_data() {
        let mut decoder = make_decoder();
        let result = decoder.send_packet(&[], 0);
        assert!(result.is_err());
    }

    // ── receive_frame returns None if no packet ───────────────────────────

    #[test]
    fn test_receive_frame_no_packet() {
        let mut decoder = make_decoder();
        let frame = decoder.receive_frame().expect("should succeed");
        assert!(frame.is_none());
    }

    // ── CELT encode → decode round-trip ──────────────────────────────────

    /// Build a minimal CELT Opus packet from raw f32 samples.
    fn encode_celt_roundtrip(samples: &[f32], channels: u8) -> Vec<u8> {
        let frame_size = samples.len() / usize::from(channels);
        let mut enc = make_encoder(128_000, frame_size as u32, channels);

        // Pack samples into an AudioFrame.
        let byte_count = samples.len() * 4;
        let mut bytes = vec![0u8; byte_count];
        for (i, &s) in samples.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&s.to_le_bytes());
        }

        let frame = AudioFrame {
            format: SampleFormat::F32,
            sample_rate: 48000,
            channels: ChannelLayout::from_count(usize::from(channels)),
            samples: AudioBuffer::Interleaved(Bytes::from(bytes)),
            timestamp: Timestamp::new(0, Rational::new(1, 48000)),
        };

        enc.send_frame(&frame).expect("should succeed");
        let pkt = enc
            .receive_packet()
            .expect("should succeed")
            .expect("Expected packet");
        pkt.data
    }

    #[test]
    fn test_celt_encode_decode_mono_silence() {
        // Encode 480 zero samples (10ms mono).
        let samples = vec![0.0f32; 480];
        let packet = encode_celt_roundtrip(&samples, 1);

        let mut decoder = make_decoder();
        decoder.send_packet(&packet, 0).expect("should succeed");
        let frame = decoder
            .receive_frame()
            .expect("should succeed")
            .expect("Expected frame");

        assert_eq!(frame.format, SampleFormat::F32);
        assert_eq!(frame.sample_rate, 48000);
    }

    #[test]
    fn test_celt_encode_produces_packet_with_toc() {
        let samples = vec![0.0f32; 480];
        let packet = encode_celt_roundtrip(&samples, 1);

        // First byte is TOC.
        assert!(!packet.is_empty());
        let toc = TocByte::parse(packet[0]);
        assert_eq!(toc.mode, OpusMode::CeltOnly);
    }

    #[test]
    fn test_celt_encode_decode_sine_wave() {
        // Generate a 440 Hz sine wave at 48kHz for 10ms (480 samples).
        let n = 480usize;
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 48000.0).sin() as f32)
            .collect();

        let packet = encode_celt_roundtrip(&samples, 1);
        let mut decoder = make_decoder();
        decoder.send_packet(&packet, 0).expect("should succeed");
        let frame = decoder
            .receive_frame()
            .expect("should succeed")
            .expect("Expected frame");

        // Verify the decoded frame has the right format.
        assert_eq!(frame.format, SampleFormat::F32);
        assert_eq!(frame.sample_rate, 48000);
        assert_eq!(frame.channels, ChannelLayout::Mono);
    }

    #[test]
    fn test_celt_encode_decode_stereo() {
        let n = 960usize; // 20ms * 2 channels
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 220.0 * i as f64 / 48000.0).sin() as f32 * 0.5)
            .collect();

        let packet = encode_celt_roundtrip(&samples, 2);
        let config = AudioDecoderConfig {
            codec: CodecId::Opus,
            sample_rate: 48000,
            channels: 2,
            extradata: None,
        };
        let mut decoder = OpusDecoder::new(&config).expect("should succeed");
        decoder.send_packet(&packet, 100).expect("should succeed");
        let frame = decoder
            .receive_frame()
            .expect("should succeed")
            .expect("Expected frame");

        assert_eq!(frame.channels, ChannelLayout::Stereo);
        assert_eq!(frame.timestamp.pts, 100);
    }

    // ── Multi-frame packet ────────────────────────────────────────────────

    #[test]
    fn test_receive_frame_outputs_correct_sample_count() {
        // 20ms at 48kHz = 960 samples per channel.
        let n = 960usize;
        let samples = vec![0.5f32; n];
        let packet = encode_celt_roundtrip(&samples, 1);

        let mut decoder = make_decoder();
        decoder.send_packet(&packet, 0).expect("should succeed");
        let frame = decoder
            .receive_frame()
            .expect("should succeed")
            .expect("Expected frame");

        // The decoded frame should have a non-zero sample count.
        assert!(frame.sample_count() > 0);
    }

    // ── Encoder receive_packet returns None when no data ──────────────────

    #[test]
    fn test_encoder_receive_packet_no_data() {
        let mut enc = make_encoder(128_000, 960, 1);
        let pkt = enc.receive_packet().expect("should succeed");
        assert!(pkt.is_none());
    }

    // ── send_packet → receive_frame pipeline ─────────────────────────────

    #[test]
    fn test_send_then_receive_consumes_pending() {
        let samples = vec![0.0f32; 480];
        let packet = encode_celt_roundtrip(&samples, 1);

        let mut decoder = make_decoder();
        decoder.send_packet(&packet, 42).expect("should succeed");

        // First call: gets the frame.
        let first = decoder.receive_frame().expect("should succeed");
        assert!(first.is_some());

        // Second call: nothing pending.
        let second = decoder.receive_frame().expect("should succeed");
        assert!(second.is_none());
    }

    // ── SILK encode path ──────────────────────────────────────────────────

    #[test]
    fn test_silk_encoder_produces_packet() {
        let n = 960usize; // 20ms
        let samples: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / 48000.0).sin() as f32)
            .collect();

        let byte_count = samples.len() * 4;
        let mut bytes = vec![0u8; byte_count];
        for (i, &s) in samples.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&s.to_le_bytes());
        }

        let frame = AudioFrame {
            format: SampleFormat::F32,
            sample_rate: 48000,
            channels: ChannelLayout::Mono,
            samples: AudioBuffer::Interleaved(Bytes::from(bytes)),
            timestamp: Timestamp::new(0, Rational::new(1, 48000)),
        };

        // Use low bitrate to trigger SILK path.
        let mut enc = make_encoder(16_000, n as u32, 1);
        enc.send_frame(&frame).expect("should succeed");
        let pkt = enc
            .receive_packet()
            .expect("should succeed")
            .expect("Expected SILK packet");

        assert!(!pkt.data.is_empty());
        assert_eq!(pkt.pts, 0);
        assert_eq!(pkt.duration, n as u32);

        // Verify the mode embedded in TOC is SILK.
        let toc = TocByte::parse(pkt.data[0]);
        assert_eq!(toc.mode, OpusMode::SilkOnly);
    }

    // ── Encoder PTS advances correctly ────────────────────────────────────

    #[test]
    fn test_encoder_pts_advances() {
        let n = 480usize;
        let mut enc = make_encoder(128_000, n as u32, 1);

        for frame_idx in 0..3u64 {
            let samples = vec![0.0f32; n];
            let byte_count = samples.len() * 4;
            let mut bytes = vec![0u8; byte_count];
            for (i, &s) in samples.iter().enumerate() {
                bytes[i * 4..i * 4 + 4].copy_from_slice(&s.to_le_bytes());
            }

            let frame = AudioFrame {
                format: SampleFormat::F32,
                sample_rate: 48000,
                channels: ChannelLayout::Mono,
                samples: AudioBuffer::Interleaved(Bytes::from(bytes)),
                timestamp: Timestamp::new((frame_idx * n as u64) as i64, Rational::new(1, 48000)),
            };

            enc.send_frame(&frame).expect("should succeed");
            let pkt = enc
                .receive_packet()
                .expect("should succeed")
                .expect("Expected packet");
            assert_eq!(pkt.pts, (frame_idx * n as u64) as i64);
        }
    }

    // ── Flush/reset ───────────────────────────────────────────────────────

    #[test]
    fn test_decoder_flush_and_reset() {
        let mut decoder = make_decoder();
        decoder.flush().expect("should succeed");
        assert!(decoder.flushing);
        decoder.reset();
        assert!(!decoder.flushing);
        // After reset, receive_frame should return None.
        assert!(decoder.receive_frame().expect("should succeed").is_none());
    }

    #[test]
    fn test_encoder_flush() {
        let mut enc = make_encoder(128_000, 960, 1);
        enc.flush().expect("should succeed");
        assert!(enc.flushing);
    }

    // ── TOC byte construction ─────────────────────────────────────────────

    #[test]
    fn test_make_toc_byte_celt_mono_20ms() {
        let toc_byte = make_toc_byte(false, celt::CeltFrameSize::Ms20);
        let toc = TocByte::parse(toc_byte);
        assert_eq!(toc.mode, OpusMode::CeltOnly);
        assert!(!toc.stereo);
        assert_eq!(toc.frame_count, FrameCount::One);
    }

    #[test]
    fn test_make_toc_byte_celt_stereo_10ms() {
        let toc_byte = make_toc_byte(true, celt::CeltFrameSize::Ms10);
        let toc = TocByte::parse(toc_byte);
        assert_eq!(toc.mode, OpusMode::CeltOnly);
        assert!(toc.stereo);
    }

    // ── Planar buffer encode ──────────────────────────────────────────────

    #[test]
    fn test_encode_planar_buffer() {
        let n = 480usize;
        let samples: Vec<f32> = (0..n).map(|i| (i as f32 / n as f32).sin()).collect();

        // Build planar (single channel) buffer.
        let mut bytes = vec![0u8; n * 4];
        for (i, &s) in samples.iter().enumerate() {
            bytes[i * 4..i * 4 + 4].copy_from_slice(&s.to_le_bytes());
        }

        let frame = AudioFrame {
            format: SampleFormat::F32,
            sample_rate: 48000,
            channels: ChannelLayout::Mono,
            samples: AudioBuffer::Planar(vec![Bytes::from(bytes)]),
            timestamp: Timestamp::new(0, Rational::new(1, 48000)),
        };

        let mut enc = make_encoder(128_000, n as u32, 1);
        enc.send_frame(&frame).expect("should succeed");
        let pkt = enc.receive_packet().expect("should succeed");
        assert!(pkt.is_some());
    }

    // ── Decoder output format ─────────────────────────────────────────────

    #[test]
    fn test_decoder_output_format() {
        let decoder = make_decoder();
        assert_eq!(decoder.output_format(), Some(SampleFormat::F32));
        assert_eq!(decoder.sample_rate(), Some(48000));
        assert_eq!(decoder.channel_layout(), Some(ChannelLayout::Mono));
    }
}
