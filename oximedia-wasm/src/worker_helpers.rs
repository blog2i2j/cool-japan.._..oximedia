//! WebWorker transfer helpers for OxiMedia WASM.
//!
//! This module provides serialisation utilities that make it convenient to
//! move media frame data between the main thread and Web Workers via the
//! [structured-clone algorithm](https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API/Structured_clone_algorithm).
//!
//! # Design Goals
//!
//! Moving large `ArrayBuffer`s across the worker boundary is zero-copy when
//! the buffer is *transferred* rather than cloned.  The helpers in this module
//! serialise frame *metadata* (dimensions, presentation timestamp, pixel
//! format) alongside the raw pixel buffer so that the receiving worker can
//! reconstruct a complete frame description without a round-trip back to the
//! main thread.
//!
//! The serialisation format is a compact binary layout designed to be parsed
//! efficiently in JavaScript without a full JSON parse:
//!
//! ```text
//! [magic: 4 bytes = 0x4F 0x58 0x46 0x52]  ("OXFR")
//! [version: 1 byte = 0x01]
//! [pixel_format: 1 byte]
//! [width: u32 LE 4 bytes]
//! [height: u32 LE 4 bytes]
//! [pts: i64 LE 8 bytes]
//! [flags: u16 LE 2 bytes]
//! [plane_count: u8 1 byte]  (1 for packed, 3 for planar YUV)
//! [plane_0_len: u32 LE 4 bytes]
//! [plane_0_data: plane_0_len bytes]
//! [plane_1_len: u32 LE 4 bytes]  (if plane_count >= 2)
//! [plane_1_data: ...]
//! [plane_2_len: u32 LE 4 bytes]  (if plane_count >= 3)
//! [plane_2_data: ...]
//! ```
//!
//! # JavaScript Example
//!
//! ```javascript
//! import * as oximedia from 'oximedia-wasm';
//!
//! // Main thread:
//! const yuv = new Uint8Array(width * height * 3 / 2);
//! // … fill yuv with frame data …
//! const transfer = oximedia.transferable_frame(yuv, width, height, pts);
//! worker.postMessage({ frame: transfer }, [transfer.buffer]);
//!
//! // Worker thread:
//! self.onmessage = (e) => {
//!     const meta = oximedia.parse_transfer_header(e.data.frame);
//!     console.log(`Received ${meta.width}x${meta.height} frame @ pts=${meta.pts}`);
//!     const planes = oximedia.split_transfer_planes(e.data.frame);
//!     // planes[0] = Y plane, planes[1] = U plane, planes[2] = V plane
//! };
//! ```

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

// ---------------------------------------------------------------------------
// Constants

/// Magic bytes that identify an OxiMedia transfer frame.
const MAGIC: [u8; 4] = [0x4F, 0x58, 0x46, 0x52]; // "OXFR"

/// Current serialisation format version.
const FORMAT_VERSION: u8 = 0x01;

/// Header size in bytes (everything except the plane data).
/// magic(4) + version(1) + pixel_format(1) + width(4) + height(4) + pts(8) + flags(2) + plane_count(1)
const HEADER_SIZE: usize = 25;

// ---------------------------------------------------------------------------
// Pixel format byte values

/// YUV 4:2:0 planar (three separate planes).
const FMT_YUV420P: u8 = 0x01;
/// RGBA packed (single plane).
const FMT_RGBA: u8 = 0x02;
/// RGB packed (single plane).
const FMT_RGB: u8 = 0x03;

// ---------------------------------------------------------------------------
// Transfer frame metadata (returned to JavaScript as JSON)

/// Metadata extracted from a transfer frame header.
///
/// Returned by [`parse_transfer_header`] as a JSON string so that JavaScript
/// callers can inspect frame properties without re-parsing the raw bytes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferFrameMeta {
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Presentation timestamp in milliseconds (ms).
    pub pts: i64,
    /// Pixel format string (e.g. `"yuv420p"`, `"rgba"`, `"rgb"`).
    pub pixel_format: String,
    /// Transfer format flags (currently always 0).
    pub flags: u16,
    /// Number of pixel data planes in the transfer buffer.
    pub plane_count: u8,
    /// Total size of the serialised buffer in bytes.
    pub total_bytes: u32,
}

// ---------------------------------------------------------------------------
// Public API

/// Serialise a YUV420p frame into a transferable binary buffer.
///
/// The returned `Uint8Array` contains a self-describing header followed by the
/// raw pixel data.  It can be `postMessage`d to a Web Worker with
/// `[result.buffer]` in the transfer list so that the buffer is moved (not
/// copied) to the worker.
///
/// # Arguments
///
/// * `yuv_data` - YUV420p frame data.  Length must be exactly
///   `width * height * 3 / 2` bytes.
/// * `width` - Frame width in pixels (must be even).
/// * `height` - Frame height in pixels (must be even).
/// * `pts` - Presentation timestamp in milliseconds.
///
/// # Errors
///
/// Returns a JavaScript error if dimensions are invalid or `yuv_data` has the
/// wrong length.
#[wasm_bindgen]
pub fn transferable_frame(
    yuv_data: &[u8],
    width: u32,
    height: u32,
    pts: i64,
) -> Result<js_sys::Uint8Array, JsValue> {
    validate_yuv420_dims(width, height).map_err(|e| crate::utils::js_err(&e))?;

    let expected = yuv420p_byte_len(width, height);
    if yuv_data.len() != expected {
        return Err(crate::utils::js_err(&format!(
            "transferable_frame: yuv_data length {} does not match expected {} \
             (width={} height={})",
            yuv_data.len(),
            expected,
            width,
            height
        )));
    }

    // Split into three planes.
    let (y_plane, uv_data) = split_yuv420p_planes(yuv_data, width, height);
    let (u_plane, v_plane) = split_uv_planes(uv_data, width, height);

    let buf = build_transfer_buffer(
        FMT_YUV420P,
        width,
        height,
        pts,
        0u16,
        &[y_plane, u_plane, v_plane],
    )
    .map_err(|e| crate::utils::js_err(&e))?;

    Ok(js_sys::Uint8Array::from(buf.as_slice()))
}

/// Serialise a packed RGBA frame into a transferable binary buffer.
///
/// # Arguments
///
/// * `rgba_data` - Packed RGBA data.  Length must be exactly `width * height * 4`.
/// * `width` - Frame width in pixels.
/// * `height` - Frame height in pixels.
/// * `pts` - Presentation timestamp in milliseconds.
///
/// # Errors
///
/// Returns a JavaScript error if dimensions are invalid or `rgba_data` has the
/// wrong length.
#[wasm_bindgen]
pub fn transferable_frame_rgba(
    rgba_data: &[u8],
    width: u32,
    height: u32,
    pts: i64,
) -> Result<js_sys::Uint8Array, JsValue> {
    if width == 0 {
        return Err(crate::utils::js_err(
            "transferable_frame_rgba: width must be > 0",
        ));
    }
    if height == 0 {
        return Err(crate::utils::js_err(
            "transferable_frame_rgba: height must be > 0",
        ));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|n| n.checked_mul(4))
        .ok_or_else(|| crate::utils::js_err("transferable_frame_rgba: dimension overflow"))?;

    if rgba_data.len() != expected {
        return Err(crate::utils::js_err(&format!(
            "transferable_frame_rgba: rgba_data length {} does not match expected {} \
             (width={} height={})",
            rgba_data.len(),
            expected,
            width,
            height
        )));
    }

    let buf = build_transfer_buffer(FMT_RGBA, width, height, pts, 0u16, &[rgba_data])
        .map_err(|e| crate::utils::js_err(&e))?;

    Ok(js_sys::Uint8Array::from(buf.as_slice()))
}

/// Parse the header of a transfer buffer and return frame metadata as JSON.
///
/// # Arguments
///
/// * `buf` - A `Uint8Array` previously produced by
///   [`transferable_frame`] or [`transferable_frame_rgba`].
///
/// # Returns
///
/// JSON string encoding a [`TransferFrameMeta`] object.
///
/// # Errors
///
/// Returns a JavaScript error if `buf` is not a valid OxiMedia transfer buffer.
#[wasm_bindgen]
pub fn parse_transfer_header(buf: &[u8]) -> Result<String, JsValue> {
    let meta = decode_header(buf)
        .map_err(|e| crate::utils::js_err(&format!("parse_transfer_header: {e}")))?;
    serde_json::to_string(&meta)
        .map_err(|e| crate::utils::js_err(&format!("parse_transfer_header: JSON error: {e}")))
}

/// Extract individual plane buffers from a transfer frame.
///
/// Returns an `Array` of `Uint8Array` objects (one per plane).  For YUV420p
/// frames this will be three elements: Y, U, V.  For RGBA/RGB it will be one.
///
/// # Arguments
///
/// * `buf` - A `Uint8Array` previously produced by
///   [`transferable_frame`] or [`transferable_frame_rgba`].
///
/// # Errors
///
/// Returns a JavaScript error if `buf` is not a valid OxiMedia transfer buffer.
#[wasm_bindgen]
pub fn split_transfer_planes(buf: &[u8]) -> Result<js_sys::Array, JsValue> {
    let planes = decode_planes(buf)
        .map_err(|e| crate::utils::js_err(&format!("split_transfer_planes: {e}")))?;

    let arr = js_sys::Array::new();
    for plane in planes {
        arr.push(&js_sys::Uint8Array::from(plane.as_slice()));
    }
    Ok(arr)
}

// ---------------------------------------------------------------------------
// Internal serialisation helpers

/// Build a transfer buffer from an array of plane slices.
fn build_transfer_buffer(
    pixel_format: u8,
    width: u32,
    height: u32,
    pts: i64,
    flags: u16,
    planes: &[&[u8]],
) -> Result<Vec<u8>, String> {
    if planes.len() > 255 {
        return Err(format!("too many planes: {}", planes.len()));
    }
    let plane_count = planes.len() as u8;

    // Pre-calculate total capacity.
    let planes_total: usize = planes
        .iter()
        .map(|p| p.len() + 4 /* u32 length prefix */)
        .sum();
    let total = HEADER_SIZE + planes_total;
    let mut buf: Vec<u8> = Vec::with_capacity(total);

    // Magic
    buf.extend_from_slice(&MAGIC);
    // Version
    buf.push(FORMAT_VERSION);
    // Pixel format
    buf.push(pixel_format);
    // Width (u32 LE)
    buf.extend_from_slice(&width.to_le_bytes());
    // Height (u32 LE)
    buf.extend_from_slice(&height.to_le_bytes());
    // PTS (i64 LE)
    buf.extend_from_slice(&pts.to_le_bytes());
    // Flags (u16 LE)
    buf.extend_from_slice(&flags.to_le_bytes());
    // Plane count
    buf.push(plane_count);

    // Plane data
    for plane in planes {
        let plane_len = plane.len() as u32;
        buf.extend_from_slice(&plane_len.to_le_bytes());
        buf.extend_from_slice(plane);
    }

    Ok(buf)
}

/// Decode the header portion of a transfer buffer.
fn decode_header(buf: &[u8]) -> Result<TransferFrameMeta, String> {
    if buf.len() < HEADER_SIZE {
        return Err(format!(
            "buffer too short: {} bytes (minimum header is {} bytes)",
            buf.len(),
            HEADER_SIZE
        ));
    }

    // Check magic
    if buf[0..4] != MAGIC {
        return Err(format!(
            "invalid magic bytes: {:02X} {:02X} {:02X} {:02X}",
            buf[0], buf[1], buf[2], buf[3]
        ));
    }

    // Version
    let version = buf[4];
    if version != FORMAT_VERSION {
        return Err(format!(
            "unsupported format version: {version} (expected {FORMAT_VERSION})"
        ));
    }

    let pixel_format_byte = buf[5];
    let pixel_format = match pixel_format_byte {
        FMT_YUV420P => "yuv420p".to_string(),
        FMT_RGBA => "rgba".to_string(),
        FMT_RGB => "rgb".to_string(),
        other => format!("unknown(0x{other:02X})"),
    };

    let width = u32::from_le_bytes([buf[6], buf[7], buf[8], buf[9]]);
    let height = u32::from_le_bytes([buf[10], buf[11], buf[12], buf[13]]);
    let pts = i64::from_le_bytes([
        buf[14], buf[15], buf[16], buf[17], buf[18], buf[19], buf[20], buf[21],
    ]);
    let flags = u16::from_le_bytes([buf[22], buf[23]]);
    let plane_count = buf[24];

    Ok(TransferFrameMeta {
        width,
        height,
        pts,
        pixel_format,
        flags,
        plane_count,
        total_bytes: buf.len() as u32,
    })
}

/// Decode all plane data from a transfer buffer.
fn decode_planes(buf: &[u8]) -> Result<Vec<Vec<u8>>, String> {
    let meta = decode_header(buf)?;
    let mut offset = HEADER_SIZE;
    let mut planes = Vec::with_capacity(meta.plane_count as usize);

    for plane_idx in 0..(meta.plane_count as usize) {
        if offset + 4 > buf.len() {
            return Err(format!(
                "buffer truncated reading plane {plane_idx} length field (offset={offset})"
            ));
        }
        let plane_len = u32::from_le_bytes([
            buf[offset],
            buf[offset + 1],
            buf[offset + 2],
            buf[offset + 3],
        ]) as usize;
        offset += 4;

        if offset + plane_len > buf.len() {
            return Err(format!(
                "buffer truncated reading plane {plane_idx} data \
                 (need {plane_len} bytes at offset {offset}, buf len={})",
                buf.len()
            ));
        }
        planes.push(buf[offset..offset + plane_len].to_vec());
        offset += plane_len;
    }

    Ok(planes)
}

// ---------------------------------------------------------------------------
// YUV420p geometry helpers

/// Total byte length of a YUV420p frame.
fn yuv420p_byte_len(width: u32, height: u32) -> usize {
    let luma = (width as usize) * (height as usize);
    let chroma = ((width as usize + 1) / 2) * ((height as usize + 1) / 2);
    luma + 2 * chroma
}

/// Validate that width and height are valid for YUV420p.
///
/// Returns `Err` with a human-readable message (no `JsValue`) so this
/// function is callable from native unit tests.
fn validate_yuv420_dims(width: u32, height: u32) -> Result<(), String> {
    if width == 0 {
        return Err("width must be > 0".to_string());
    }
    if height == 0 {
        return Err("height must be > 0".to_string());
    }
    if width % 2 != 0 {
        return Err("width must be even (required by YUV420p)".to_string());
    }
    if height % 2 != 0 {
        return Err("height must be even (required by YUV420p)".to_string());
    }
    Ok(())
}

/// Split a YUV420p buffer into the Y plane and combined UV data.
fn split_yuv420p_planes<'a>(yuv: &'a [u8], width: u32, height: u32) -> (&'a [u8], &'a [u8]) {
    let luma_len = (width as usize) * (height as usize);
    (&yuv[..luma_len], &yuv[luma_len..])
}

/// Split the combined UV data into separate U and V planes.
fn split_uv_planes<'a>(uv: &'a [u8], width: u32, height: u32) -> (&'a [u8], &'a [u8]) {
    let chroma_len = ((width as usize + 1) / 2) * ((height as usize + 1) / 2);
    (&uv[..chroma_len], &uv[chroma_len..])
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_yuv420(width: u32, height: u32) -> Vec<u8> {
        let len = yuv420p_byte_len(width, height);
        (0..len).map(|i| (i & 0xFF) as u8).collect()
    }

    #[test]
    fn test_yuv420p_byte_len() {
        // 4x4 frame: 16 luma + 4 + 4 chroma = 24 bytes
        assert_eq!(yuv420p_byte_len(4, 4), 24);
        // 2x2: 4 + 1 + 1 = 6
        assert_eq!(yuv420p_byte_len(2, 2), 6);
    }

    #[test]
    fn test_round_trip_yuv420() {
        let w = 8u32;
        let h = 8u32;
        let pts = 42_000i64;
        let frame = make_yuv420(w, h);

        let buf = build_transfer_buffer(
            FMT_YUV420P,
            w,
            h,
            pts,
            0,
            &[
                &frame[..64],        // Y plane (8*8)
                &frame[64..64 + 16], // U plane (4*4)
                &frame[80..80 + 16], // V plane (4*4)
            ],
        )
        .expect("buffer build should succeed");

        let meta = decode_header(&buf).expect("header decode should succeed");
        assert_eq!(meta.width, w);
        assert_eq!(meta.height, h);
        assert_eq!(meta.pts, pts);
        assert_eq!(meta.pixel_format, "yuv420p");
        assert_eq!(meta.plane_count, 3);

        let planes = decode_planes(&buf).expect("plane decode should succeed");
        assert_eq!(planes.len(), 3);
        assert_eq!(planes[0].len(), 64); // Y
        assert_eq!(planes[1].len(), 16); // U
        assert_eq!(planes[2].len(), 16); // V
    }

    #[test]
    fn test_round_trip_rgba() {
        let w = 4u32;
        let h = 4u32;
        let rgba = vec![0xFFu8; w as usize * h as usize * 4];
        let buf = build_transfer_buffer(FMT_RGBA, w, h, 0, 0, &[&rgba])
            .expect("buffer build should succeed");

        let meta = decode_header(&buf).expect("header decode should succeed");
        assert_eq!(meta.pixel_format, "rgba");
        assert_eq!(meta.plane_count, 1);

        let planes = decode_planes(&buf).expect("plane decode should succeed");
        assert_eq!(planes.len(), 1);
        assert_eq!(planes[0].len(), 64);
    }

    #[test]
    fn test_invalid_magic() {
        let buf = vec![0u8; HEADER_SIZE + 8];
        let result = decode_header(&buf);
        assert!(result.is_err(), "expected error for bad magic");
    }

    #[test]
    fn test_buffer_too_short() {
        let result = decode_header(&[0u8; 10]);
        assert!(result.is_err());
    }

    #[test]
    fn test_validate_yuv420_rejects_odd_width() {
        assert!(validate_yuv420_dims(3, 4).is_err());
    }

    #[test]
    fn test_validate_yuv420_rejects_odd_height() {
        assert!(validate_yuv420_dims(4, 3).is_err());
    }

    #[test]
    fn test_validate_yuv420_rejects_zero() {
        assert!(validate_yuv420_dims(0, 4).is_err());
        assert!(validate_yuv420_dims(4, 0).is_err());
    }

    #[test]
    fn test_validate_yuv420_accepts_even() {
        assert!(validate_yuv420_dims(4, 4).is_ok());
        assert!(validate_yuv420_dims(1920, 1080).is_ok());
    }

    #[test]
    fn test_parse_transfer_header_json() {
        let frame = make_yuv420(4, 4);
        let buf = build_transfer_buffer(
            FMT_YUV420P,
            4,
            4,
            1234,
            0,
            &[&frame[..16], &frame[16..20], &frame[20..24]],
        )
        .expect("buffer build should succeed");
        let header = decode_header(&buf).expect("header decode should succeed");
        let json = serde_json::to_string(&header).expect("JSON serialization should succeed");
        assert!(json.contains("yuv420p"));
        assert!(json.contains("1234"));
    }

    #[test]
    fn test_split_yuv420p_planes_geometry() {
        let w = 8u32;
        let h = 8u32;
        let frame = make_yuv420(w, h);
        let (y, uv) = split_yuv420p_planes(&frame, w, h);
        assert_eq!(y.len(), 64);
        assert_eq!(uv.len(), 32); // 16 (U) + 16 (V)
        let (u, v) = split_uv_planes(uv, w, h);
        assert_eq!(u.len(), 16);
        assert_eq!(v.len(), 16);
    }
}
