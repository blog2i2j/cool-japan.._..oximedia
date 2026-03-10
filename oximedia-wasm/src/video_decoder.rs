//! WASM VP8 video decoder.
//!
//! This module provides a WebAssembly-compatible VP8 video decoder that outputs
//! YUV420 planar data as `Uint8Array`. VP8 is a royalty-free video codec
//! developed by Google as part of the WebM project.
//!
//! # Architecture
//!
//! VP8 always outputs YUV 4:2:0 planar format. For a frame of width W and
//! height H, the output buffer contains:
//! - Y plane: W * H bytes
//! - U plane: (W/2) * (H/2) bytes
//! - V plane: (W/2) * (H/2) bytes
//!
//! Total size = W * H * 3 / 2 bytes.
//!
//! # JavaScript Example
//!
//! ```javascript
//! const decoder = new oximedia.WasmVp8Decoder();
//! decoder.init(keyframeData);
//! const yuv = decoder.decode_frame(frameData);
//! console.log(`${decoder.width()}x${decoder.height()} ${decoder.format()}`);
//! ```

use wasm_bindgen::prelude::*;

use oximedia_codec::traits::{DecoderConfig, VideoDecoder};
use oximedia_codec::vp8::{FrameHeader, Vp8Decoder};

/// VP8 video decoder for WebAssembly.
///
/// Decodes VP8 compressed frames to YUV420 planar Uint8Array data.
/// VP8 is commonly used in WebM containers and is fully royalty-free.
///
/// The decoder must be initialized with a VP8 keyframe before inter frames
/// can be decoded. The `init()` method parses the frame header to extract
/// dimensions.
#[wasm_bindgen]
pub struct WasmVp8Decoder {
    /// Internal VP8 decoder.
    decoder: Option<Vp8Decoder>,
    /// Decoded frame width.
    width: u32,
    /// Decoded frame height.
    height: u32,
    /// Whether the decoder has been initialized.
    initialized: bool,
}

#[wasm_bindgen]
impl WasmVp8Decoder {
    /// Create a new VP8 decoder.
    ///
    /// The decoder must be initialized with `init()` before decoding frames.
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self {
            decoder: None,
            width: 0,
            height: 0,
            initialized: false,
        }
    }

    /// Initialize decoder from VP8 bitstream header.
    ///
    /// The header data should be the beginning of a VP8 keyframe that contains
    /// the frame tag (3 bytes) and for keyframes the sync code and dimensions.
    ///
    /// # Frame tag format (3 bytes)
    ///
    /// - Bit 0: frame type (0 = keyframe, 1 = inter)
    /// - Bits 1-2: version
    /// - Bit 3: show frame flag
    /// - Bits 4-23: first partition size
    ///
    /// For keyframes, bytes 3-6 contain the sync code (0x9D 0x01 0x2A) and
    /// bytes 7-10 contain width and height.
    ///
    /// # Errors
    ///
    /// Returns an error if the header data is too short or cannot be parsed.
    pub fn init(&mut self, header: &[u8]) -> Result<(), JsValue> {
        let frame_header = FrameHeader::parse(header)
            .map_err(|e| crate::utils::js_err(&format!("VP8 header parse error: {e}")))?;

        self.width = u32::from(frame_header.width);
        self.height = u32::from(frame_header.height);

        let config = DecoderConfig::default();
        let decoder = Vp8Decoder::new(config)
            .map_err(|e| crate::utils::js_err(&format!("VP8 decoder creation error: {e}")))?;

        self.decoder = Some(decoder);
        self.initialized = true;
        Ok(())
    }

    /// Decode a VP8 frame.
    ///
    /// Returns YUV420 data as Uint8Array with planes concatenated:
    /// `[Y plane (W*H)] [U plane (W/2 * H/2)] [V plane (W/2 * H/2)]`
    ///
    /// For keyframes, the entire frame is self-contained. For inter frames,
    /// previous reference frames must have been decoded first.
    ///
    /// # Errors
    ///
    /// Returns an error if the decoder is not initialized or decoding fails.
    pub fn decode_frame(&mut self, data: &[u8]) -> Result<js_sys::Uint8Array, JsValue> {
        let decoder = self
            .decoder
            .as_mut()
            .ok_or_else(|| crate::utils::js_err("VP8 decoder not initialized"))?;

        decoder
            .send_packet(data, 0)
            .map_err(|e| crate::utils::js_err(&format!("VP8 send_packet error: {e}")))?;

        match decoder.receive_frame() {
            Ok(Some(frame)) => {
                // Update dimensions from decoded frame
                self.width = frame.width;
                self.height = frame.height;

                // Assemble YUV420 planar data from the decoded frame planes
                let y_size = (self.width * self.height) as usize;
                let uv_width = (self.width + 1) / 2;
                let uv_height = (self.height + 1) / 2;
                let uv_size = (uv_width * uv_height) as usize;
                let total_size = y_size + 2 * uv_size;

                let mut yuv_data = vec![0u8; total_size];

                // Copy plane data if available
                if let Some(y_plane) = frame.planes.first() {
                    let copy_len = y_plane.data.len().min(y_size);
                    yuv_data[..copy_len].copy_from_slice(&y_plane.data[..copy_len]);
                }
                if let Some(u_plane) = frame.planes.get(1) {
                    let copy_len = u_plane.data.len().min(uv_size);
                    yuv_data[y_size..y_size + copy_len].copy_from_slice(&u_plane.data[..copy_len]);
                }
                if let Some(v_plane) = frame.planes.get(2) {
                    let copy_len = v_plane.data.len().min(uv_size);
                    yuv_data[y_size + uv_size..y_size + uv_size + copy_len]
                        .copy_from_slice(&v_plane.data[..copy_len]);
                }

                Ok(js_sys::Uint8Array::from(yuv_data.as_slice()))
            }
            Ok(None) => {
                // No frame produced; return empty array
                Ok(js_sys::Uint8Array::new_with_length(0))
            }
            Err(e) => Err(crate::utils::js_err(&format!("VP8 decode error: {e}"))),
        }
    }

    /// Get decoded frame width in pixels.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Get decoded frame height in pixels.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Get output pixel format description.
    ///
    /// VP8 always outputs YUV 4:2:0 planar format.
    pub fn format(&self) -> String {
        "yuv420p".to_string()
    }

    /// Check if the decoder has been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Reset decoder state.
    ///
    /// After reset, the decoder must be re-initialized with a keyframe.
    pub fn reset(&mut self) {
        if let Some(ref mut decoder) = self.decoder {
            decoder.reset();
        }
        self.width = 0;
        self.height = 0;
        self.initialized = false;
    }
}
