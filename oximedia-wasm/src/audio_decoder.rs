//! WASM audio decoders.
//!
//! This module provides WebAssembly-compatible audio decoders for FLAC, Vorbis,
//! and Opus codecs. Each decoder wraps the internal codec implementations from
//! `oximedia-audio` and exposes a JavaScript-friendly API.
//!
//! All decoders follow the same pattern:
//! 1. Create a decoder with `new()`
//! 2. Initialize with codec header bytes via `init()`
//! 3. Decode frames with `decode_frame()` returning `Float32Array`
//! 4. Query properties (sample_rate, channels, etc.)
//!
//! # JavaScript Example
//!
//! ```javascript
//! const decoder = new oximedia.WasmFlacDecoder();
//! decoder.init(headerBytes);
//! const pcmSamples = decoder.decode_frame(frameData);
//! console.log('Sample rate:', decoder.sample_rate());
//! ```

use wasm_bindgen::prelude::*;

use oximedia_audio::flac::{FlacDecoder, StreamInfo};
use oximedia_audio::opus::OpusDecoder;
use oximedia_audio::traits::{AudioDecoder, AudioDecoderConfig};
use oximedia_audio::vorbis::{IdentificationHeader, VorbisDecoder};
use oximedia_audio::{AudioBuffer, AudioFrame};
use oximedia_core::{CodecId, SampleFormat};

/// Extract raw bytes from an `AudioFrame`'s sample buffer.
fn extract_frame_bytes(frame: &AudioFrame) -> &[u8] {
    match &frame.samples {
        AudioBuffer::Interleaved(data) => data,
        AudioBuffer::Planar(planes) => {
            // For planar data, return the first plane as best effort
            if let Some(first) = planes.first() {
                first
            } else {
                &[]
            }
        }
    }
}

/// Convert raw sample bytes to f32 based on the sample format.
///
/// Handles S16, S32, F32, F64, and U8 formats. Returns interleaved f32 samples
/// normalized to the -1.0 to 1.0 range.
fn samples_to_f32(raw: &[u8], format: SampleFormat) -> Vec<f32> {
    match format {
        SampleFormat::S16 | SampleFormat::S16p => raw
            .chunks_exact(2)
            .map(|chunk| {
                let sample = i16::from_le_bytes([chunk[0], chunk[1]]);
                f32::from(sample) / 32768.0
            })
            .collect(),
        SampleFormat::S32 | SampleFormat::S32p => raw
            .chunks_exact(4)
            .map(|chunk| {
                let sample = i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                sample as f32 / 2_147_483_648.0
            })
            .collect(),
        SampleFormat::F32 | SampleFormat::F32p => raw
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect(),
        SampleFormat::F64 | SampleFormat::F64p => raw
            .chunks_exact(8)
            .map(|chunk| {
                f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32
            })
            .collect(),
        SampleFormat::U8 => raw
            .iter()
            .map(|&b| (f32::from(b) - 128.0) / 128.0)
            .collect(),
        _ => {
            // Unknown format: treat as silence
            vec![0.0; raw.len()]
        }
    }
}

// ---------------------------------------------------------------------------
// FLAC Decoder
// ---------------------------------------------------------------------------

/// FLAC audio decoder for WebAssembly.
///
/// Decodes FLAC compressed audio data into PCM float samples.
/// FLAC is a lossless codec that typically achieves 50-70% compression.
///
/// # Usage
///
/// ```javascript
/// const decoder = new oximedia.WasmFlacDecoder();
/// decoder.init(streamInfoBytes);
/// const samples = decoder.decode_frame(flacFrame);
/// console.log(`${decoder.channels()} channels at ${decoder.sample_rate()} Hz`);
/// ```
#[wasm_bindgen]
pub struct WasmFlacDecoder {
    /// Internal FLAC decoder.
    decoder: Option<FlacDecoder>,
    /// Detected sample rate from stream info.
    sample_rate: u32,
    /// Detected channel count.
    channels: u16,
    /// Detected bits per sample.
    bits_per_sample: u16,
    /// Whether the decoder has been successfully initialized.
    initialized: bool,
}

#[wasm_bindgen]
impl WasmFlacDecoder {
    /// Create a new FLAC decoder.
    ///
    /// The decoder must be initialized with `init()` before decoding frames.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            decoder: None,
            sample_rate: 0,
            channels: 0,
            bits_per_sample: 0,
            initialized: false,
        }
    }

    /// Initialize decoder with FLAC stream header bytes.
    ///
    /// The header should be the STREAMINFO metadata block (34 bytes).
    /// This extracts sample rate, channel count, and bits per sample.
    ///
    /// # Errors
    ///
    /// Returns an error if the header data is invalid or too short.
    pub fn init(&mut self, header: &[u8]) -> Result<(), JsValue> {
        let stream_info = StreamInfo::parse(header)
            .map_err(|e| crate::utils::js_err(&format!("FLAC init error: {e}")))?;

        self.sample_rate = stream_info.sample_rate;
        self.channels = u16::from(stream_info.channels);
        self.bits_per_sample = u16::from(stream_info.bits_per_sample);

        let config = AudioDecoderConfig {
            codec: CodecId::Flac,
            sample_rate: self.sample_rate,
            channels: self.channels as u8,
            extradata: Some(header.to_vec()),
        };

        let decoder = FlacDecoder::new(&config)
            .map_err(|e| crate::utils::js_err(&format!("FLAC decoder creation error: {e}")))?;

        self.decoder = Some(decoder);
        self.initialized = true;
        Ok(())
    }

    /// Decode a FLAC frame and return PCM samples as Float32Array.
    ///
    /// The input should be a complete FLAC frame including the frame header.
    /// Returns interleaved float samples normalized to the -1.0 to 1.0 range.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoder is not initialized or if decoding fails.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<js_sys::Float32Array, JsValue> {
        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| crate::utils::js_err("FLAC decoder not initialized"))?;

        decoder
            .send_packet(data, 0)
            .map_err(|e| crate::utils::js_err(&format!("FLAC send_packet error: {e}")))?;

        match decoder.receive_frame() {
            Ok(Some(frame)) => {
                let raw_bytes = extract_frame_bytes(&frame);
                let float_samples = samples_to_f32(raw_bytes, frame.format);
                Ok(js_sys::Float32Array::from(float_samples.as_slice()))
            }
            Ok(None) => {
                // No frame produced yet; return empty array
                Ok(js_sys::Float32Array::new_with_length(0))
            }
            Err(e) => Err(crate::utils::js_err(&format!("FLAC decode error: {e}"))),
        }
    }

    /// Get sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of audio channels.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Get bits per sample.
    pub fn bits_per_sample(&self) -> u16 {
        self.bits_per_sample
    }

    /// Check if the decoder has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Reset decoder state.
    ///
    /// After reset, the decoder can be re-initialized with new stream info.
    pub fn reset(&mut self) {
        if let Some(ref mut decoder) = self.decoder {
            decoder.reset();
        }
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Vorbis Decoder
// ---------------------------------------------------------------------------

/// Vorbis audio decoder for WebAssembly.
///
/// Decodes Ogg Vorbis compressed audio data into PCM float samples.
/// Vorbis is a lossy, royalty-free audio codec commonly used in Ogg containers.
///
/// # Usage
///
/// ```javascript
/// const decoder = new oximedia.WasmVorbisDecoder();
/// decoder.init(identificationHeaderBytes);
/// const samples = decoder.decode_frame(vorbisPacket);
/// ```
#[wasm_bindgen]
pub struct WasmVorbisDecoder {
    /// Internal Vorbis decoder.
    decoder: Option<VorbisDecoder>,
    /// Sample rate from identification header.
    sample_rate: u32,
    /// Channel count from identification header.
    channels: u16,
    /// Whether the decoder has been initialized.
    initialized: bool,
}

#[wasm_bindgen]
impl WasmVorbisDecoder {
    /// Create a new Vorbis decoder.
    ///
    /// The decoder must be initialized with `init()` before decoding frames.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            decoder: None,
            sample_rate: 0,
            channels: 0,
            initialized: false,
        }
    }

    /// Initialize decoder with Vorbis identification header bytes.
    ///
    /// The header is the first Vorbis header packet from an Ogg stream.
    /// It contains sample rate, channel count, and codec configuration.
    ///
    /// # Errors
    ///
    /// Returns an error if the header is invalid.
    pub fn init(&mut self, header: &[u8]) -> Result<(), JsValue> {
        let id_header = IdentificationHeader::parse(header)
            .map_err(|e| crate::utils::js_err(&format!("Vorbis header parse error: {e}")))?;

        self.sample_rate = id_header.audio_sample_rate;
        self.channels = u16::from(id_header.audio_channels);

        let config = AudioDecoderConfig {
            codec: CodecId::Vorbis,
            sample_rate: self.sample_rate,
            channels: self.channels as u8,
            extradata: Some(header.to_vec()),
        };

        let decoder = VorbisDecoder::new(&config)
            .map_err(|e| crate::utils::js_err(&format!("Vorbis decoder creation error: {e}")))?;

        self.decoder = Some(decoder);
        self.initialized = true;
        Ok(())
    }

    /// Decode a Vorbis audio packet and return PCM samples as Float32Array.
    ///
    /// Returns interleaved float samples normalized to -1.0 to 1.0.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoder is not initialized or decoding fails.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<js_sys::Float32Array, JsValue> {
        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| crate::utils::js_err("Vorbis decoder not initialized"))?;

        decoder
            .send_packet(data, 0)
            .map_err(|e| crate::utils::js_err(&format!("Vorbis send_packet error: {e}")))?;

        match decoder.receive_frame() {
            Ok(Some(frame)) => {
                let raw_bytes = extract_frame_bytes(&frame);
                let float_samples = samples_to_f32(raw_bytes, frame.format);
                Ok(js_sys::Float32Array::from(float_samples.as_slice()))
            }
            Ok(None) => Ok(js_sys::Float32Array::new_with_length(0)),
            Err(e) => Err(crate::utils::js_err(&format!("Vorbis decode error: {e}"))),
        }
    }

    /// Get sample rate in Hz.
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of audio channels.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Check if the decoder has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        if let Some(ref mut decoder) = self.decoder {
            decoder.reset();
        }
        self.initialized = false;
    }
}

// ---------------------------------------------------------------------------
// Opus Decoder
// ---------------------------------------------------------------------------

/// Opus audio decoder for WebAssembly.
///
/// Decodes Opus compressed audio data into PCM float samples.
/// Opus is a modern, versatile codec supporting both speech and music
/// at low latency. It combines SILK and CELT modes.
///
/// # Usage
///
/// ```javascript
/// const decoder = new oximedia.WasmOpusDecoder();
/// decoder.init(opusHeaderBytes);
/// const samples = decoder.decode_frame(opusPacket);
/// ```
#[wasm_bindgen]
pub struct WasmOpusDecoder {
    /// Internal Opus decoder.
    decoder: Option<OpusDecoder>,
    /// Output sample rate (always 48000 for Opus).
    sample_rate: u32,
    /// Channel count.
    channels: u16,
    /// Whether the decoder has been initialized.
    initialized: bool,
}

#[wasm_bindgen]
impl WasmOpusDecoder {
    /// Create a new Opus decoder.
    ///
    /// The decoder must be initialized with `init()` before decoding frames.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            decoder: None,
            sample_rate: 48000, // Opus always decodes at 48 kHz
            channels: 0,
            initialized: false,
        }
    }

    /// Initialize decoder with Opus header bytes.
    ///
    /// The header should be the OpusHead packet from an Ogg Opus stream.
    /// Minimum requirement is 19 bytes containing version, channel count,
    /// pre-skip, input sample rate, output gain, and mapping family.
    ///
    /// A simplified initialization is also supported: if the header is
    /// exactly 2 bytes, they are interpreted as `[channels, mapping_family]`.
    ///
    /// # Errors
    ///
    /// Returns an error if the header is too short or invalid.
    pub fn init(&mut self, header: &[u8]) -> Result<(), JsValue> {
        // Parse Opus header (OpusHead)
        // Minimal format: 8 bytes magic "OpusHead" + fields, or simplified 2-byte form
        let channels: u8;

        if header.len() >= 19 && header.starts_with(b"OpusHead") {
            // Standard OpusHead packet
            // Byte 8: version
            // Byte 9: channel count
            // Bytes 10-11: pre-skip (LE)
            // Bytes 12-15: input sample rate (LE)
            // Bytes 16-17: output gain (LE)
            // Byte 18: mapping family
            channels = header[9];
        } else if header.len() >= 2 {
            // Simplified header: [channels, mapping_family]
            channels = header[0];
        } else {
            return Err(crate::utils::js_err(
                "Opus header too short: need at least 2 bytes",
            ));
        }

        if channels == 0 || channels > 8 {
            return Err(crate::utils::js_err(&format!(
                "Invalid Opus channel count: {channels}"
            )));
        }

        self.channels = u16::from(channels);
        self.sample_rate = 48000; // Opus always outputs at 48 kHz

        let config = AudioDecoderConfig {
            codec: CodecId::Opus,
            sample_rate: self.sample_rate,
            channels,
            extradata: Some(header.to_vec()),
        };

        let decoder = OpusDecoder::new(&config)
            .map_err(|e| crate::utils::js_err(&format!("Opus decoder creation error: {e}")))?;

        self.decoder = Some(decoder);
        self.initialized = true;
        Ok(())
    }

    /// Decode an Opus packet and return PCM samples as Float32Array.
    ///
    /// Returns interleaved float samples normalized to -1.0 to 1.0.
    /// Opus always outputs at 48000 Hz.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoder is not initialized or decoding fails.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<js_sys::Float32Array, JsValue> {
        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| crate::utils::js_err("Opus decoder not initialized"))?;

        decoder
            .send_packet(data, 0)
            .map_err(|e| crate::utils::js_err(&format!("Opus send_packet error: {e}")))?;

        match decoder.receive_frame() {
            Ok(Some(frame)) => {
                let raw_bytes = extract_frame_bytes(&frame);
                let float_samples = samples_to_f32(raw_bytes, frame.format);
                Ok(js_sys::Float32Array::from(float_samples.as_slice()))
            }
            Ok(None) => Ok(js_sys::Float32Array::new_with_length(0)),
            Err(e) => Err(crate::utils::js_err(&format!("Opus decode error: {e}"))),
        }
    }

    /// Get output sample rate in Hz (always 48000 for Opus).
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Get number of audio channels.
    pub fn channels(&self) -> u16 {
        self.channels
    }

    /// Check if the decoder has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Reset decoder state.
    pub fn reset(&mut self) {
        if let Some(ref mut decoder) = self.decoder {
            decoder.reset();
        }
        self.initialized = false;
    }
}
