//! APNG (Animated Portable Network Graphics) top-level encoder and decoder.
//!
//! APNG extends PNG with animation chunks (`acTL`, `fcTL`, `fdAT`) as specified in
//! the APNG specification (<https://wiki.mozilla.org/APNG_Spec>).
//!
//! # Container layout
//!
//! ```text
//! PNG signature  (8 bytes)
//! IHDR           canvas width, height, bit_depth=8, color_type=6 (RGBA)
//! acTL           num_frames, num_plays (loop_count)
//! — per frame —
//!   fcTL         sequence_number, w, h, x_off, y_off, delay_num, delay_den, dispose, blend
//!   IDAT / fdAT  compressed scanline data (IDAT for frame 0, fdAT for the rest)
//! IEND
//! ```
//!
//! # Example
//!
//! ```rust
//! use oximedia_codec::apng::{ApngEncoder, ApngDecoder, ApngFrame, ApngConfig};
//!
//! let config = ApngConfig {
//!     loop_count: 0,
//!     default_delay_num: 1,
//!     default_delay_den: 10,
//! };
//!
//! let frame = ApngFrame {
//!     pixels: vec![128u8; 4 * 4 * 4],
//!     width: 4,
//!     height: 4,
//!     delay_num: 1,
//!     delay_den: 10,
//!     dispose_op: 0,
//!     blend_op: 0,
//!     x_offset: 0,
//!     y_offset: 0,
//! };
//!
//! let encoded = ApngEncoder::encode(&[frame], &config).expect("encode failed");
//! assert!(encoded.starts_with(b"\x89PNG"));
//! ```

#![forbid(unsafe_code)]
#![allow(clippy::cast_possible_truncation)]

use crate::error::CodecError;
use oxiarc_deflate::{ZlibStreamDecoder, ZlibStreamEncoder};
use std::io::{Read, Write};

// =============================================================================
// CRC-32  (ISO 3309 — same polynomial as used in PNG spec)
// =============================================================================

fn crc32(data: &[u8]) -> u32 {
    const POLY: u32 = 0xEDB8_8320;
    let mut crc: u32 = 0xFFFF_FFFF;
    for &byte in data {
        let mut b = u32::from(byte);
        for _ in 0..8 {
            if (crc ^ b) & 1 != 0 {
                crc = (crc >> 1) ^ POLY;
            } else {
                crc >>= 1;
            }
            b >>= 1;
        }
    }
    !crc
}

// =============================================================================
// Public types
// =============================================================================

/// Global configuration for an APNG animation.
#[derive(Debug, Clone)]
pub struct ApngConfig {
    /// Number of times to loop the animation (0 = infinite).
    pub loop_count: u32,
    /// Default frame delay numerator used when constructing frames.
    pub default_delay_num: u16,
    /// Default frame delay denominator (e.g. 100 → centiseconds).
    pub default_delay_den: u16,
}

impl Default for ApngConfig {
    fn default() -> Self {
        Self {
            loop_count: 0,
            default_delay_num: 1,
            default_delay_den: 10,
        }
    }
}

/// A single frame of an APNG animation.
#[derive(Debug, Clone)]
pub struct ApngFrame {
    /// Raw RGBA pixel data: `width × height × 4` bytes.
    pub pixels: Vec<u8>,
    /// Frame width in pixels.
    pub width: u32,
    /// Frame height in pixels.
    pub height: u32,
    /// Frame delay numerator.
    pub delay_num: u16,
    /// Frame delay denominator (fps = delay_den / delay_num; 0 treated as 100).
    pub delay_den: u16,
    /// Disposal operation: 0 = none, 1 = clear to background, 2 = revert to previous.
    pub dispose_op: u8,
    /// Blend operation: 0 = source (overwrite), 1 = over (alpha composite).
    pub blend_op: u8,
    /// X offset of this frame on the canvas.
    pub x_offset: u32,
    /// Y offset of this frame on the canvas.
    pub y_offset: u32,
}

// =============================================================================
// Encoder
// =============================================================================

/// APNG encoder — encodes a sequence of [`ApngFrame`]s into an APNG byte stream.
pub struct ApngEncoder;

impl ApngEncoder {
    /// Encode `frames` into an APNG byte stream.
    ///
    /// The canvas dimensions are taken from the first frame.  All frames must
    /// have a `pixels` buffer of exactly `width × height × 4` bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if `frames` is empty, any pixel buffer has the
    /// wrong size, or DEFLATE compression fails.
    pub fn encode(frames: &[ApngFrame], config: &ApngConfig) -> Result<Vec<u8>, CodecError> {
        if frames.is_empty() {
            return Err(CodecError::InvalidParameter(
                "APNG requires at least one frame".to_string(),
            ));
        }

        // Canvas dimensions come from the first frame.
        let canvas_w = frames[0].width;
        let canvas_h = frames[0].height;

        // Validate all pixel buffers up-front.
        for (i, frame) in frames.iter().enumerate() {
            let expected = (frame.width as usize) * (frame.height as usize) * 4;
            if frame.pixels.len() != expected {
                return Err(CodecError::InvalidParameter(format!(
                    "frame {i}: expected {expected} bytes ({w}×{h}×4), got {}",
                    frame.pixels.len(),
                    w = frame.width,
                    h = frame.height,
                )));
            }
        }

        let mut out: Vec<u8> = Vec::new();

        // PNG signature
        out.extend_from_slice(b"\x89PNG\r\n\x1a\n");

        // IHDR
        let mut ihdr = [0u8; 13];
        ihdr[..4].copy_from_slice(&canvas_w.to_be_bytes());
        ihdr[4..8].copy_from_slice(&canvas_h.to_be_bytes());
        ihdr[8] = 8; // bit depth
        ihdr[9] = 6; // colour type: RGBA
                     // compression, filter, interlace all 0
        write_chunk(&mut out, b"IHDR", &ihdr);

        // acTL — animation control
        let mut actl = [0u8; 8];
        actl[..4].copy_from_slice(&(frames.len() as u32).to_be_bytes());
        actl[4..].copy_from_slice(&config.loop_count.to_be_bytes());
        write_chunk(&mut out, b"acTL", &actl);

        let mut seq_num: u32 = 0;

        for (frame_idx, frame) in frames.iter().enumerate() {
            // fcTL — frame control  (26 bytes of data)
            let mut fctl = [0u8; 26];
            fctl[..4].copy_from_slice(&seq_num.to_be_bytes());
            seq_num += 1;
            fctl[4..8].copy_from_slice(&frame.width.to_be_bytes());
            fctl[8..12].copy_from_slice(&frame.height.to_be_bytes());
            fctl[12..16].copy_from_slice(&frame.x_offset.to_be_bytes());
            fctl[16..20].copy_from_slice(&frame.y_offset.to_be_bytes());
            fctl[20..22].copy_from_slice(&frame.delay_num.to_be_bytes());
            fctl[22..24].copy_from_slice(&frame.delay_den.to_be_bytes());
            fctl[24] = frame.dispose_op;
            fctl[25] = frame.blend_op;
            write_chunk(&mut out, b"fcTL", &fctl);

            // Compress pixel data.
            let compressed =
                compress_frame(&frame.pixels, frame.width as usize, frame.height as usize)?;

            if frame_idx == 0 {
                // First frame: standard IDAT so non-APNG decoders show it.
                write_chunk(&mut out, b"IDAT", &compressed);
            } else {
                // Subsequent frames: fdAT prefixed with sequence number.
                let mut fdat = Vec::with_capacity(4 + compressed.len());
                fdat.extend_from_slice(&seq_num.to_be_bytes());
                seq_num += 1;
                fdat.extend_from_slice(&compressed);
                write_chunk(&mut out, b"fdAT", &fdat);
            }
        }

        // IEND
        write_chunk(&mut out, b"IEND", &[]);

        Ok(out)
    }
}

// =============================================================================
// Decoder
// =============================================================================

/// APNG decoder — parses an APNG byte stream and reconstructs [`ApngFrame`]s.
pub struct ApngDecoder;

impl ApngDecoder {
    /// Decode a raw APNG byte stream.
    ///
    /// Returns the list of frames (with decompressed RGBA pixels) and the
    /// global [`ApngConfig`].
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if the PNG signature is missing, any chunk is
    /// truncated, or pixel decompression fails.
    pub fn decode(data: &[u8]) -> Result<(Vec<ApngFrame>, ApngConfig), CodecError> {
        check_signature(data)?;

        // ── Pass 1: parse all chunks ─────────────────────────────────────────
        let chunks = parse_chunks(data)?;

        // ── Extract IHDR ─────────────────────────────────────────────────────
        let ihdr_data = find_chunk_data(&chunks, b"IHDR")
            .ok_or_else(|| CodecError::InvalidBitstream("APNG: missing IHDR chunk".to_string()))?;
        if ihdr_data.len() < 13 {
            return Err(CodecError::InvalidBitstream(
                "APNG: IHDR too short".to_string(),
            ));
        }
        let canvas_w = u32::from_be_bytes([ihdr_data[0], ihdr_data[1], ihdr_data[2], ihdr_data[3]]);
        let canvas_h = u32::from_be_bytes([ihdr_data[4], ihdr_data[5], ihdr_data[6], ihdr_data[7]]);
        let bit_depth = ihdr_data[8];
        let color_type = ihdr_data[9];

        // We only handle 8-bit RGBA for now.
        if bit_depth != 8 || color_type != 6 {
            return Err(CodecError::UnsupportedFeature(format!(
                "APNG decoder supports only 8-bit RGBA (got bit_depth={bit_depth}, color_type={color_type})"
            )));
        }

        // ── Extract acTL ─────────────────────────────────────────────────────
        let (loop_count, declared_frame_count) =
            if let Some(actl) = find_chunk_data(&chunks, b"acTL") {
                if actl.len() < 8 {
                    return Err(CodecError::InvalidBitstream(
                        "APNG: acTL too short".to_string(),
                    ));
                }
                let nf = u32::from_be_bytes([actl[0], actl[1], actl[2], actl[3]]);
                let lc = u32::from_be_bytes([actl[4], actl[5], actl[6], actl[7]]);
                (lc, nf)
            } else {
                (0u32, 0u32)
            };

        // ── Collect fcTL + IDAT/fdAT pairs ───────────────────────────────────
        // Walk the chunk list in order, keeping a pending fcTL and accumulating
        // compressed data per frame.
        struct PendingFrame {
            fctl: FctlInfo,
            compressed: Vec<u8>,
        }

        #[derive(Clone)]
        struct FctlInfo {
            width: u32,
            height: u32,
            x_offset: u32,
            y_offset: u32,
            delay_num: u16,
            delay_den: u16,
            dispose_op: u8,
            blend_op: u8,
        }

        let mut frames_raw: Vec<PendingFrame> = Vec::new();
        let mut current_fctl: Option<FctlInfo> = None;
        let mut idat_consumed = false;

        for (ctype, cdata) in &chunks {
            match ctype.as_slice() {
                b"fcTL" => {
                    // If there's an active frame being built, finalise it.
                    if let Some(fctl) = current_fctl.take() {
                        // The previous fcTL had no data yet (shouldn't happen in
                        // well-formed APNG, but be defensive).
                        frames_raw.push(PendingFrame {
                            fctl,
                            compressed: Vec::new(),
                        });
                    }
                    if cdata.len() < 26 {
                        return Err(CodecError::InvalidBitstream(
                            "APNG: fcTL too short".to_string(),
                        ));
                    }
                    let fw = u32::from_be_bytes([cdata[4], cdata[5], cdata[6], cdata[7]]);
                    let fh = u32::from_be_bytes([cdata[8], cdata[9], cdata[10], cdata[11]]);
                    let fx = u32::from_be_bytes([cdata[12], cdata[13], cdata[14], cdata[15]]);
                    let fy = u32::from_be_bytes([cdata[16], cdata[17], cdata[18], cdata[19]]);
                    let dn = u16::from_be_bytes([cdata[20], cdata[21]]);
                    let dd = u16::from_be_bytes([cdata[22], cdata[23]]);
                    current_fctl = Some(FctlInfo {
                        width: fw,
                        height: fh,
                        x_offset: fx,
                        y_offset: fy,
                        delay_num: dn,
                        delay_den: dd,
                        dispose_op: cdata[24],
                        blend_op: cdata[25],
                    });
                }
                b"IDAT" => {
                    if idat_consumed {
                        // Append to last frame's compressed data (split IDAT).
                        if let Some(last) = frames_raw.last_mut() {
                            last.compressed.extend_from_slice(cdata);
                        }
                        continue;
                    }
                    idat_consumed = true;
                    if let Some(fctl) = current_fctl.take() {
                        let mut pending = PendingFrame {
                            fctl,
                            compressed: Vec::new(),
                        };
                        pending.compressed.extend_from_slice(cdata);
                        frames_raw.push(pending);
                    } else {
                        // IDAT before any fcTL — treat canvas as default frame 0.
                        let fctl = FctlInfo {
                            width: canvas_w,
                            height: canvas_h,
                            x_offset: 0,
                            y_offset: 0,
                            delay_num: 1,
                            delay_den: 10,
                            dispose_op: 0,
                            blend_op: 0,
                        };
                        let mut pending = PendingFrame {
                            fctl,
                            compressed: Vec::new(),
                        };
                        pending.compressed.extend_from_slice(cdata);
                        frames_raw.push(pending);
                    }
                }
                b"fdAT" => {
                    // fdAT: 4-byte sequence number prefix then compressed data.
                    if cdata.len() < 4 {
                        return Err(CodecError::InvalidBitstream(
                            "APNG: fdAT too short".to_string(),
                        ));
                    }
                    let payload = &cdata[4..];
                    if let Some(fctl) = current_fctl.take() {
                        let mut pending = PendingFrame {
                            fctl,
                            compressed: Vec::new(),
                        };
                        pending.compressed.extend_from_slice(payload);
                        frames_raw.push(pending);
                    } else if let Some(last) = frames_raw.last_mut() {
                        // Continuation fdAT for current frame.
                        last.compressed.extend_from_slice(payload);
                    }
                }
                _ => {}
            }
        }

        // Flush any trailing pending fcTL with no data.
        if let Some(fctl) = current_fctl.take() {
            frames_raw.push(PendingFrame {
                fctl,
                compressed: Vec::new(),
            });
        }

        // ── Decompress + defilter each frame ─────────────────────────────────
        let mut out_frames: Vec<ApngFrame> = Vec::with_capacity(frames_raw.len());
        for pf in frames_raw {
            let w = pf.fctl.width as usize;
            let h = pf.fctl.height as usize;
            let pixels = if pf.compressed.is_empty() {
                // No data: return transparent black.
                vec![0u8; w * h * 4]
            } else {
                decompress_rgba(&pf.compressed, w, h)?
            };
            out_frames.push(ApngFrame {
                pixels,
                width: pf.fctl.width,
                height: pf.fctl.height,
                delay_num: pf.fctl.delay_num,
                delay_den: pf.fctl.delay_den,
                dispose_op: pf.fctl.dispose_op,
                blend_op: pf.fctl.blend_op,
                x_offset: pf.fctl.x_offset,
                y_offset: pf.fctl.y_offset,
            });
        }

        let config = ApngConfig {
            loop_count,
            default_delay_num: if out_frames.is_empty() {
                1
            } else {
                out_frames[0].delay_num
            },
            default_delay_den: if out_frames.is_empty() {
                10
            } else {
                out_frames[0].delay_den
            },
        };

        // Sanity: warn-or-ignore mismatched declared_frame_count (not an error).
        let _ = declared_frame_count;

        Ok((out_frames, config))
    }

    /// Quick probe: returns the number of frames declared in the `acTL` chunk.
    ///
    /// Does not decompress pixel data.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError`] if the PNG signature is invalid or the file is
    /// truncated.
    pub fn frame_count(data: &[u8]) -> Result<u32, CodecError> {
        check_signature(data)?;
        let mut pos = 8usize;
        while pos + 8 <= data.len() {
            let chunk_len =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            let chunk_type = &data[pos + 4..pos + 8];
            let data_start = pos + 8;
            let data_end = data_start + chunk_len;
            if data_end + 4 > data.len() {
                return Err(CodecError::InvalidBitstream(
                    "APNG: truncated chunk while scanning for acTL".to_string(),
                ));
            }
            if chunk_type == b"acTL" && chunk_len >= 8 {
                let fc = u32::from_be_bytes([
                    data[data_start],
                    data[data_start + 1],
                    data[data_start + 2],
                    data[data_start + 3],
                ]);
                return Ok(fc);
            }
            if chunk_type == b"IEND" {
                break;
            }
            pos = data_end + 4;
        }
        // No acTL found — not animated; treat as 1-frame static PNG.
        Ok(1)
    }

    /// Returns `true` if `data` is an APNG (valid PNG signature + `acTL` chunk).
    #[must_use]
    pub fn is_apng(data: &[u8]) -> bool {
        if data.len() < 8 || &data[..8] != b"\x89PNG\r\n\x1a\n" {
            return false;
        }
        let mut pos = 8usize;
        while pos + 8 <= data.len() {
            let chunk_len =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            let chunk_type = &data[pos + 4..pos + 8];
            if chunk_type == b"acTL" {
                return true;
            }
            if chunk_type == b"IEND" {
                break;
            }
            let data_end = pos + 8 + chunk_len;
            pos = data_end + 4; // skip CRC
        }
        false
    }
}

// =============================================================================
// Internal helpers
// =============================================================================

fn check_signature(data: &[u8]) -> Result<(), CodecError> {
    if data.len() < 8 || &data[..8] != b"\x89PNG\r\n\x1a\n" {
        return Err(CodecError::InvalidBitstream(
            "Not a PNG file (bad signature)".to_string(),
        ));
    }
    Ok(())
}

/// Parsed chunk: (4-byte type, data bytes).
type Chunk = ([u8; 4], Vec<u8>);

fn parse_chunks(data: &[u8]) -> Result<Vec<Chunk>, CodecError> {
    let mut chunks = Vec::new();
    let mut pos = 8usize; // skip signature
    while pos + 8 <= data.len() {
        let chunk_len =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        let mut ctype = [0u8; 4];
        ctype.copy_from_slice(&data[pos + 4..pos + 8]);
        let data_start = pos + 8;
        let data_end = data_start + chunk_len;
        if data_end + 4 > data.len() {
            return Err(CodecError::InvalidBitstream(format!(
                "APNG: chunk '{}' is truncated",
                String::from_utf8_lossy(&ctype)
            )));
        }
        let cdata = data[data_start..data_end].to_vec();
        let is_iend = &ctype == b"IEND";
        chunks.push((ctype, cdata));
        pos = data_end + 4; // skip 4-byte CRC
        if is_iend {
            break;
        }
    }
    Ok(chunks)
}

fn find_chunk_data<'a>(chunks: &'a [Chunk], ctype: &[u8; 4]) -> Option<&'a [u8]> {
    chunks
        .iter()
        .find(|(t, _)| t == ctype)
        .map(|(_, d)| d.as_slice())
}

/// Apply PNG Sub filter + DEFLATE-compress an RGBA frame.
fn compress_frame(rgba: &[u8], width: usize, height: usize) -> Result<Vec<u8>, CodecError> {
    let row_bytes = width * 4;
    let mut filtered: Vec<u8> = Vec::with_capacity((row_bytes + 1) * height);
    for row in 0..height {
        filtered.push(1); // Sub filter
        let base = row * row_bytes;
        for col in 0..row_bytes {
            let pixel = rgba[base + col];
            let prev = if col >= 4 { rgba[base + col - 4] } else { 0 };
            filtered.push(pixel.wrapping_sub(prev));
        }
    }
    let mut enc = ZlibStreamEncoder::new(Vec::new(), 6);
    enc.write_all(&filtered).map_err(CodecError::Io)?;
    enc.finish().map_err(CodecError::Io)
}

/// DEFLATE-decompress a zlib stream and PNG-defilter to get RGBA pixels.
fn decompress_rgba(compressed: &[u8], width: usize, height: usize) -> Result<Vec<u8>, CodecError> {
    // Inflate.
    let row_stride = width * 4; // bytes per row (no filter byte here)
    let expected_filtered = (row_stride + 1) * height;
    let mut filtered = Vec::with_capacity(expected_filtered);
    let mut decoder = ZlibStreamDecoder::new(compressed);
    decoder
        .read_to_end(&mut filtered)
        .map_err(|e| CodecError::InvalidBitstream(format!("APNG inflate error: {e}")))?;

    // Each row is: 1 filter byte + row_stride data bytes.
    if filtered.len() < (row_stride + 1) * height {
        return Err(CodecError::InvalidBitstream(format!(
            "APNG: decompressed data too short: got {} bytes, need {}",
            filtered.len(),
            (row_stride + 1) * height
        )));
    }

    let mut pixels = vec![0u8; width * height * 4];

    for row in 0..height {
        let src_row_start = row * (row_stride + 1);
        let filter_type = filtered[src_row_start];
        let src = &filtered[src_row_start + 1..src_row_start + 1 + row_stride];
        let dst_start = row * row_stride;

        // Copy the previous row into a local buffer so we can hold a mutable
        // borrow on `pixels[dst_start..]` while reading from the prior row.
        let prev_row: Vec<u8> = if row > 0 {
            pixels[(row - 1) * row_stride..row * row_stride].to_vec()
        } else {
            vec![0u8; row_stride]
        };

        let dst = &mut pixels[dst_start..dst_start + row_stride];

        match filter_type {
            0 => {
                // None
                dst.copy_from_slice(src);
            }
            1 => {
                // Sub: Recon(x) = Filt(x) + Recon(a)
                for i in 0..row_stride {
                    let a = if i >= 4 { dst[i - 4] } else { 0 };
                    dst[i] = src[i].wrapping_add(a);
                }
            }
            2 => {
                // Up: Recon(x) = Filt(x) + Recon(b)
                for i in 0..row_stride {
                    dst[i] = src[i].wrapping_add(prev_row[i]);
                }
            }
            3 => {
                // Average: Recon(x) = Filt(x) + floor((Recon(a)+Recon(b))/2)
                for i in 0..row_stride {
                    let a = if i >= 4 { dst[i - 4] } else { 0 };
                    let b = prev_row[i];
                    dst[i] = src[i].wrapping_add(((u16::from(a) + u16::from(b)) / 2) as u8);
                }
            }
            4 => {
                // Paeth
                for i in 0..row_stride {
                    let a = if i >= 4 { dst[i - 4] } else { 0 };
                    let b = prev_row[i];
                    let c = if i >= 4 { prev_row[i - 4] } else { 0 };
                    dst[i] = src[i].wrapping_add(paeth_predictor(a, b, c));
                }
            }
            ft => {
                return Err(CodecError::InvalidBitstream(format!(
                    "APNG: unknown PNG filter type {ft} on row {row}"
                )));
            }
        }
    }

    Ok(pixels)
}

/// PNG Paeth predictor function (spec §9.4).
#[inline]
fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    let ia = i32::from(a);
    let ib = i32::from(b);
    let ic = i32::from(c);
    let p = ia + ib - ic;
    let pa = (p - ia).abs();
    let pb = (p - ib).abs();
    let pc = (p - ic).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Serialise one PNG chunk: `length(4) ++ type(4) ++ data ++ crc(4)`.
fn write_chunk(out: &mut Vec<u8>, chunk_type: &[u8; 4], data: &[u8]) {
    out.extend_from_slice(&(data.len() as u32).to_be_bytes());
    out.extend_from_slice(chunk_type);
    out.extend_from_slice(data);
    let mut crc_input = Vec::with_capacity(4 + data.len());
    crc_input.extend_from_slice(chunk_type);
    crc_input.extend_from_slice(data);
    out.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba_frame(w: u32, h: u32, fill: u8) -> ApngFrame {
        ApngFrame {
            pixels: vec![fill; (w * h * 4) as usize],
            width: w,
            height: h,
            delay_num: 1,
            delay_den: 10,
            dispose_op: 0,
            blend_op: 0,
            x_offset: 0,
            y_offset: 0,
        }
    }

    fn default_config() -> ApngConfig {
        ApngConfig::default()
    }

    // ── Encoder tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_encode_png_signature() {
        let frame = rgba_frame(4, 4, 128);
        let data = ApngEncoder::encode(&[frame], &default_config()).expect("encode");
        assert!(
            data.starts_with(b"\x89PNG\r\n\x1a\n"),
            "Must start with PNG signature"
        );
    }

    #[test]
    fn test_encode_contains_actl() {
        let frames: Vec<_> = (0..3).map(|i| rgba_frame(8, 8, i * 50)).collect();
        let data = ApngEncoder::encode(&frames, &default_config()).expect("encode");
        assert!(data.windows(4).any(|w| w == b"acTL"), "Must contain acTL");
    }

    #[test]
    fn test_encode_empty_frames_errors() {
        let result = ApngEncoder::encode(&[], &default_config());
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_wrong_pixel_size_errors() {
        let bad = ApngFrame {
            pixels: vec![0u8; 10], // wrong
            width: 4,
            height: 4,
            delay_num: 1,
            delay_den: 10,
            dispose_op: 0,
            blend_op: 0,
            x_offset: 0,
            y_offset: 0,
        };
        let result = ApngEncoder::encode(&[bad], &default_config());
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_first_frame_idat() {
        let frame = rgba_frame(4, 4, 200);
        let data = ApngEncoder::encode(&[frame], &default_config()).expect("encode");
        assert!(
            data.windows(4).any(|w| w == b"IDAT"),
            "First frame must use IDAT"
        );
    }

    #[test]
    fn test_encode_second_frame_fdat() {
        let frames = vec![rgba_frame(4, 4, 100), rgba_frame(4, 4, 200)];
        let data = ApngEncoder::encode(&frames, &default_config()).expect("encode");
        assert!(
            data.windows(4).any(|w| w == b"fdAT"),
            "Frame 2+ must use fdAT"
        );
    }

    #[test]
    fn test_encode_fctl_count_matches_frame_count() {
        let frames: Vec<_> = (0..5).map(|i| rgba_frame(4, 4, i * 40)).collect();
        let data = ApngEncoder::encode(&frames, &default_config()).expect("encode");
        let fctl_count = data.windows(4).filter(|w| *w == b"fcTL").count();
        assert_eq!(fctl_count, 5, "One fcTL per frame");
    }

    #[test]
    fn test_encode_ends_with_iend() {
        let frame = rgba_frame(4, 4, 0);
        let data = ApngEncoder::encode(&[frame], &default_config()).expect("encode");
        // IEND chunk = length(0) + "IEND" + crc
        let iend_pos = data.len().saturating_sub(12);
        assert_eq!(&data[iend_pos + 4..iend_pos + 8], b"IEND");
    }

    // ── is_apng ───────────────────────────────────────────────────────────────

    #[test]
    fn test_is_apng_true_for_encoded() {
        let frame = rgba_frame(4, 4, 0);
        let data = ApngEncoder::encode(&[frame], &default_config()).expect("encode");
        assert!(ApngDecoder::is_apng(&data));
    }

    #[test]
    fn test_is_apng_false_for_random() {
        assert!(!ApngDecoder::is_apng(b"this is not a PNG"));
    }

    // ── frame_count ───────────────────────────────────────────────────────────

    #[test]
    fn test_frame_count_single() {
        let frame = rgba_frame(4, 4, 50);
        let data = ApngEncoder::encode(&[frame], &default_config()).expect("encode");
        let count = ApngDecoder::frame_count(&data).expect("frame_count");
        assert_eq!(count, 1);
    }

    #[test]
    fn test_frame_count_multi() {
        let frames: Vec<_> = (0..7).map(|i| rgba_frame(4, 4, i * 30)).collect();
        let data = ApngEncoder::encode(&frames, &default_config()).expect("encode");
        let count = ApngDecoder::frame_count(&data).expect("frame_count");
        assert_eq!(count, 7);
    }

    #[test]
    fn test_frame_count_bad_signature_errors() {
        let result = ApngDecoder::frame_count(b"not a png");
        assert!(result.is_err());
    }

    // ── decode (full roundtrip) ───────────────────────────────────────────────

    #[test]
    fn test_decode_single_frame_roundtrip() {
        let original = rgba_frame(4, 4, 123);
        let encoded = ApngEncoder::encode(&[original.clone()], &default_config()).expect("encode");
        let (frames, _config) = ApngDecoder::decode(&encoded).expect("decode");
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0].width, 4);
        assert_eq!(frames[0].height, 4);
        assert_eq!(frames[0].pixels, original.pixels);
    }

    #[test]
    fn test_decode_multi_frame_roundtrip() {
        let originals: Vec<_> = (0..3).map(|i| rgba_frame(8, 6, i * 80)).collect();
        let encoded = ApngEncoder::encode(&originals, &default_config()).expect("encode");
        let (frames, _config) = ApngDecoder::decode(&encoded).expect("decode");
        assert_eq!(frames.len(), 3);
        for (i, (original, decoded)) in originals.iter().zip(frames.iter()).enumerate() {
            assert_eq!(decoded.pixels, original.pixels, "frame {i} pixel mismatch");
        }
    }

    #[test]
    fn test_decode_loop_count_preserved() {
        let config = ApngConfig {
            loop_count: 5,
            default_delay_num: 1,
            default_delay_den: 25,
        };
        let frame = rgba_frame(4, 4, 0);
        let encoded = ApngEncoder::encode(&[frame], &config).expect("encode");
        let (_frames, out_config) = ApngDecoder::decode(&encoded).expect("decode");
        assert_eq!(out_config.loop_count, 5);
    }

    #[test]
    fn test_decode_frame_timing_preserved() {
        let mut frame = rgba_frame(4, 4, 0);
        frame.delay_num = 3;
        frame.delay_den = 25;
        let encoded = ApngEncoder::encode(&[frame], &default_config()).expect("encode");
        let (frames, _config) = ApngDecoder::decode(&encoded).expect("decode");
        assert_eq!(frames[0].delay_num, 3);
        assert_eq!(frames[0].delay_den, 25);
    }

    #[test]
    fn test_decode_frame_offsets_preserved() {
        let mut frame = rgba_frame(4, 4, 0);
        frame.x_offset = 10;
        frame.y_offset = 20;
        let encoded = ApngEncoder::encode(&[frame], &default_config()).expect("encode");
        let (frames, _config) = ApngDecoder::decode(&encoded).expect("decode");
        assert_eq!(frames[0].x_offset, 10);
        assert_eq!(frames[0].y_offset, 20);
    }

    #[test]
    fn test_decode_bad_signature_errors() {
        let result = ApngDecoder::decode(b"garbage data");
        assert!(result.is_err());
    }

    #[test]
    fn test_decode_dispose_blend_ops_preserved() {
        let mut frame = rgba_frame(4, 4, 0);
        frame.dispose_op = 1;
        frame.blend_op = 1;
        let encoded = ApngEncoder::encode(&[frame], &default_config()).expect("encode");
        let (frames, _config) = ApngDecoder::decode(&encoded).expect("decode");
        assert_eq!(frames[0].dispose_op, 1);
        assert_eq!(frames[0].blend_op, 1);
    }

    #[test]
    fn test_crc32_known_value() {
        // CRC32 of b"IHDR" = 0x4E4D4C4B (not the real value, we just check consistency)
        // Instead verify that our CRC matches what PNG spec requires for a known chunk.
        // The CRC of "IEND" (type) + "" (no data) = 0xAE426082
        let crc = crc32(b"IEND");
        assert_eq!(crc, 0xAE42_6082, "CRC of 'IEND' must match PNG spec");
    }

    #[test]
    fn test_large_frame_roundtrip() {
        let frame = rgba_frame(64, 48, 200);
        let encoded = ApngEncoder::encode(&[frame.clone()], &default_config()).expect("encode");
        let (frames, _) = ApngDecoder::decode(&encoded).expect("decode");
        assert_eq!(frames[0].pixels, frame.pixels);
    }
}
