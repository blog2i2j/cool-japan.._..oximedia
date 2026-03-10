//! WASM AV1 video decoder.
//!
//! This module provides a WebAssembly-compatible AV1 video decoder that outputs
//! YUV420p planar data as `Uint8Array`. AV1 is an open, royalty-free video codec
//! developed by the Alliance for Open Media.
//!
//! # Architecture
//!
//! AV1 outputs YUV 4:2:0 planar format by default (other sub-samplings are
//! possible in the sequence header). For a frame of width W and height H:
//! - Y plane: W * H bytes
//! - U plane: ceil(W/2) * ceil(H/2) bytes
//! - V plane: ceil(W/2) * ceil(H/2) bytes
//!
//! Total size = W * H * 3 / 2 bytes (for standard 4:2:0).
//!
//! # JavaScript Example
//!
//! ```javascript
//! const decoder = new oximedia.WasmAv1Decoder();
//! decoder.init(codecParamsBytes);
//! const frame = decoder.decode_packet(av1PacketData);
//! console.log(`${decoder.width()}x${decoder.height()} ${decoder.format()}`);
//! const info = JSON.parse(decoder.codec_info());
//! console.log('Codec info:', info);
//! ```

use wasm_bindgen::prelude::*;

use oximedia_codec::traits::{DecoderConfig, VideoDecoder};
use oximedia_codec::Av1Decoder;
use oximedia_core::CodecId;

/// AV1 video decoder for WebAssembly.
///
/// Decodes AV1 compressed packets to YUV420 planar `Uint8Array` data.
/// AV1 is royalty-free, patent-clear, and optimised for web delivery.
///
/// The decoder must be initialised with optional codec parameters before
/// passing compressed AV1 packets. Unlike VP8, AV1 can self-initialise
/// from the in-band sequence header OBU embedded in the first packet.
#[wasm_bindgen]
pub struct WasmAv1Decoder {
    /// Internal AV1 decoder.
    decoder: Option<Av1Decoder>,
    /// Decoded frame width (updated after each decode).
    width: u32,
    /// Decoded frame height (updated after each decode).
    height: u32,
    /// Pixel format string (updated after first successful decode).
    pixel_format: String,
    /// Whether the decoder has been initialised.
    initialized: bool,
    /// Frame count for session tracking.
    frame_count: u64,
}

#[wasm_bindgen]
impl WasmAv1Decoder {
    /// Create a new AV1 decoder.
    ///
    /// The decoder can be used immediately for packets that contain in-band
    /// sequence headers, or explicitly initialised via `init()` first.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            decoder: None,
            width: 0,
            height: 0,
            pixel_format: "yuv420p".to_string(),
            initialized: false,
            frame_count: 0,
        }
    }

    /// Initialise the decoder with optional codec extradata bytes.
    ///
    /// The `codec_params_bytes` should be a serialised representation of the
    /// codec configuration (AV1 sequence header OBUs).  Passing an empty
    /// slice is valid; the sequence header will then be parsed from the first
    /// in-band AV1 packet.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoder cannot be created or the extradata is
    /// malformed.
    pub fn init(&mut self, codec_params_bytes: &[u8]) -> Result<(), JsValue> {
        let extradata = if codec_params_bytes.is_empty() {
            None
        } else {
            Some(codec_params_bytes.to_vec())
        };

        let config = DecoderConfig {
            codec: CodecId::Av1,
            extradata,
            threads: 1,
            low_latency: true,
        };

        let av1 = Av1Decoder::new(config)
            .map_err(|e| crate::utils::js_err(&format!("AV1 decoder init error: {e}")))?;

        self.decoder = Some(av1);
        self.initialized = true;
        Ok(())
    }

    /// Decode a single AV1 compressed packet.
    ///
    /// Returns a JSON string describing the decoded frame with the following
    /// fields:
    /// - `width` (number)
    /// - `height` (number)
    /// - `pts` (number, milliseconds)
    /// - `format` (string, e.g. `"yuv420p"`)
    /// - `data` (Uint8Array, YUV420p planar bytes as Base64-encoded string in JSON)
    ///
    /// In JavaScript, use the companion `decode_packet_data()` method to get
    /// the raw `Uint8Array` efficiently; this JSON method is provided for
    /// metadata introspection.
    ///
    /// Returns `null` (JS) / `None` if the packet produces no output frame yet.
    ///
    /// # Errors
    ///
    /// Returns an error if the packet is malformed or decoding fails.
    pub fn decode_packet(
        &mut self,
        av1_packet_data: &[u8],
    ) -> Result<Option<js_sys::Uint8Array>, JsValue> {
        // Lazily initialise if not done already
        if self.decoder.is_none() {
            self.init(&[])?;
        }

        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| crate::utils::js_err("AV1 decoder not available"))?;

        let pts = self.frame_count as i64;
        decoder
            .send_packet(av1_packet_data, pts)
            .map_err(|e| crate::utils::js_err(&format!("AV1 send_packet error: {e}")))?;

        match decoder.receive_frame() {
            Ok(Some(frame)) => {
                self.width = frame.width;
                self.height = frame.height;
                self.frame_count += 1;

                // Update pixel format string from decoded frame format
                self.pixel_format = format!("{:?}", frame.format).to_lowercase();

                let yuv = Self::assemble_yuv420p(&frame);
                Ok(Some(js_sys::Uint8Array::from(yuv.as_slice())))
            }
            Ok(None) => Ok(None),
            Err(e) => Err(crate::utils::js_err(&format!(
                "AV1 receive_frame error: {e}"
            ))),
        }
    }

    /// Decode a packet and return a JSON object with frame metadata.
    ///
    /// This method returns a JSON string containing:
    /// `{ "width": W, "height": H, "pts": P, "format": "yuv420p", "frame_number": N }`
    ///
    /// The raw pixel data is not embedded (use `decode_packet()` for that).
    ///
    /// # Errors
    ///
    /// Returns an error if decoding fails.
    pub fn decode_packet_info(&mut self, av1_packet_data: &[u8]) -> Result<String, JsValue> {
        // Lazily initialise if not done already
        if self.decoder.is_none() {
            self.init(&[])?;
        }

        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| crate::utils::js_err("AV1 decoder not available"))?;

        let pts = self.frame_count as i64;
        decoder
            .send_packet(av1_packet_data, pts)
            .map_err(|e| crate::utils::js_err(&format!("AV1 send_packet error: {e}")))?;

        match decoder.receive_frame() {
            Ok(Some(frame)) => {
                self.width = frame.width;
                self.height = frame.height;
                self.frame_count += 1;
                self.pixel_format = format!("{:?}", frame.format).to_lowercase();

                let json = format!(
                    r#"{{"width":{},"height":{},"pts":{},"format":"{}","frame_number":{}}}"#,
                    frame.width,
                    frame.height,
                    frame.timestamp.pts,
                    self.pixel_format,
                    self.frame_count,
                );
                Ok(json)
            }
            Ok(None) => Ok(r#"{"status":"pending"}"#.to_string()),
            Err(e) => Err(crate::utils::js_err(&format!(
                "AV1 receive_frame error: {e}"
            ))),
        }
    }

    /// Get decoded frame width in pixels.
    ///
    /// Returns 0 if no frame has been decoded yet.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get decoded frame height in pixels.
    ///
    /// Returns 0 if no frame has been decoded yet.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get current pixel format string (e.g. `"yuv420p"`, `"yuv444p"`).
    pub fn format(&self) -> String {
        self.pixel_format.clone()
    }

    /// Check if the decoder has been initialised.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Get total number of frames decoded in this session.
    pub fn frame_count(&self) -> u64 {
        self.frame_count
    }

    /// Return a JSON string with codec capability information.
    ///
    /// Returns `{ "codec": "AV1", "profiles": [...], "formats": [...] }`.
    pub fn codec_info(&self) -> String {
        let (w, h) = self
            .decoder
            .as_ref()
            .and_then(|d| d.dimensions())
            .unwrap_or((self.width, self.height));

        format!(
            r#"{{"codec":"AV1","width":{},"height":{},"format":"{}","frame_count":{},"initialized":{}}}"#,
            w, h, self.pixel_format, self.frame_count, self.initialized,
        )
    }

    /// Flush the decoder, signalling end of stream.
    ///
    /// After calling `flush()`, call `decode_packet()` with an empty slice
    /// to drain any buffered frames.
    ///
    /// # Errors
    ///
    /// Returns an error if the flush operation fails.
    pub fn flush(&mut self) -> Result<(), JsValue> {
        if let Some(ref mut decoder) = self.decoder {
            decoder
                .flush()
                .map_err(|e| crate::utils::js_err(&format!("AV1 flush error: {e}")))?;
        }
        Ok(())
    }

    /// Reset the decoder to a clean state.
    ///
    /// After reset, the decoder must be re-initialised with a new sequence
    /// header before decoding can resume.
    pub fn reset(&mut self) {
        if let Some(ref mut decoder) = self.decoder {
            decoder.reset();
        }
        self.width = 0;
        self.height = 0;
        self.frame_count = 0;
        self.pixel_format = "yuv420p".to_string();
        self.initialized = false;
    }
}

// Private helpers
impl WasmAv1Decoder {
    /// Assemble YUV420p planar byte buffer from a decoded `VideoFrame`.
    ///
    /// Concatenates planes in Y-U-V order.  If a plane is missing, the
    /// corresponding region is zeroed.
    fn assemble_yuv420p(frame: &oximedia_codec::VideoFrame) -> Vec<u8> {
        let y_size = (frame.width * frame.height) as usize;
        let uv_width = (frame.width + 1) / 2;
        let uv_height = (frame.height + 1) / 2;
        let uv_size = (uv_width * uv_height) as usize;
        let total = y_size + 2 * uv_size;

        let mut buf = vec![0u8; total];

        if let Some(y_plane) = frame.planes.first() {
            let copy_len = y_plane.data.len().min(y_size);
            buf[..copy_len].copy_from_slice(&y_plane.data[..copy_len]);
        }
        if let Some(u_plane) = frame.planes.get(1) {
            let copy_len = u_plane.data.len().min(uv_size);
            buf[y_size..y_size + copy_len].copy_from_slice(&u_plane.data[..copy_len]);
        }
        if let Some(v_plane) = frame.planes.get(2) {
            let copy_len = v_plane.data.len().min(uv_size);
            buf[y_size + uv_size..y_size + uv_size + copy_len]
                .copy_from_slice(&v_plane.data[..copy_len]);
        }

        buf
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decoder_new() {
        let decoder = WasmAv1Decoder::new();
        assert_eq!(decoder.width(), 0);
        assert_eq!(decoder.height(), 0);
        assert_eq!(decoder.frame_count(), 0);
        assert!(!decoder.is_initialized());
    }

    #[test]
    fn test_decoder_init_empty_params() {
        let mut decoder = WasmAv1Decoder::new();
        let result = decoder.init(&[]);
        assert!(result.is_ok());
        assert!(decoder.is_initialized());
    }

    #[test]
    fn test_decoder_reset() {
        let mut decoder = WasmAv1Decoder::new();
        decoder.init(&[]).expect("init should succeed");
        decoder.reset();
        assert!(!decoder.is_initialized());
        assert_eq!(decoder.frame_count(), 0);
        assert_eq!(decoder.width(), 0);
        assert_eq!(decoder.height(), 0);
    }

    #[test]
    fn test_codec_info_json() {
        let decoder = WasmAv1Decoder::new();
        let info = decoder.codec_info();
        assert!(info.contains("AV1"));
        assert!(info.contains("yuv420p"));
    }

    #[test]
    fn test_format_default() {
        let decoder = WasmAv1Decoder::new();
        assert_eq!(decoder.format(), "yuv420p");
    }
}
