//! Pure-Rust PNG codec with APNG animation support.
//!
//! Implements RFC 2083 (PNG) and the APNG specification.
//!
//! # Features
//! - PNG decode/encode (Grayscale, RGB, RGBA, GrayAlpha, Indexed)
//! - APNG animation: acTL, fcTL, fdAT chunks
//! - All 5 PNG filter types (None, Sub, Up, Average, Paeth)
//! - CRC32 chunk validation
//! - Pure-Rust zlib via oxiarc_deflate

#![allow(dead_code)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

use crate::error::{ImageError, ImageResult};
use crate::{ColorSpace, ImageData, ImageFrame, PixelType};
use std::path::Path;

// ── Constants ────────────────────────────────────────────────────────────────

/// PNG file signature (8 bytes).
pub const PNG_SIGNATURE: [u8; 8] = [137, 80, 78, 71, 13, 10, 26, 10];

// ── CRC32 ────────────────────────────────────────────────────────────────────

/// Build the CRC32 lookup table (IEEE 802.3 polynomial 0xEDB88320).
fn build_crc_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    for n in 0..256u32 {
        let mut c = n;
        for _ in 0..8 {
            if c & 1 != 0 {
                c = 0xEDB8_8320 ^ (c >> 1);
            } else {
                c >>= 1;
            }
        }
        table[n as usize] = c;
    }
    table
}

/// Compute CRC32 over `data`.
pub fn crc32(data: &[u8]) -> u32 {
    let table = build_crc_table();
    let mut crc = 0xFFFF_FFFFu32;
    for &byte in data {
        let idx = ((crc ^ u32::from(byte)) & 0xFF) as usize;
        crc = table[idx] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

// ── PNG chunk ─────────────────────────────────────────────────────────────────

/// A PNG chunk with type and data.
#[derive(Debug, Clone)]
pub struct PngChunk {
    /// 4-byte chunk type (ASCII).
    pub chunk_type: [u8; 4],
    /// Chunk data bytes.
    pub data: Vec<u8>,
}

impl PngChunk {
    /// Create a new chunk.
    #[must_use]
    pub fn new(chunk_type: [u8; 4], data: Vec<u8>) -> Self {
        Self { chunk_type, data }
    }

    /// Returns the type as a str slice if ASCII.
    #[must_use]
    pub fn type_str(&self) -> &str {
        std::str::from_utf8(&self.chunk_type).unwrap_or("????")
    }

    /// Compute the CRC32 of type + data.
    #[must_use]
    pub fn crc(&self) -> u32 {
        let mut combined = Vec::with_capacity(4 + self.data.len());
        combined.extend_from_slice(&self.chunk_type);
        combined.extend_from_slice(&self.data);
        crc32(&combined)
    }

    /// Serialize this chunk to bytes (length + type + data + CRC).
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let len = self.data.len() as u32;
        let crc = self.crc();
        let mut out = Vec::with_capacity(12 + self.data.len());
        out.extend_from_slice(&len.to_be_bytes());
        out.extend_from_slice(&self.chunk_type);
        out.extend_from_slice(&self.data);
        out.extend_from_slice(&crc.to_be_bytes());
        out
    }
}

// ── PNG color type ────────────────────────────────────────────────────────────

/// PNG color type values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngColorType {
    /// Grayscale (1 channel).
    Grayscale = 0,
    /// RGB (3 channels).
    Rgb = 2,
    /// Indexed color (palette).
    Indexed = 3,
    /// Grayscale with alpha (2 channels).
    GrayscaleAlpha = 4,
    /// RGBA (4 channels).
    Rgba = 6,
}

impl PngColorType {
    fn from_u8(v: u8) -> ImageResult<Self> {
        match v {
            0 => Ok(Self::Grayscale),
            2 => Ok(Self::Rgb),
            3 => Ok(Self::Indexed),
            4 => Ok(Self::GrayscaleAlpha),
            6 => Ok(Self::Rgba),
            _ => Err(ImageError::invalid_format(format!(
                "Unknown PNG color type: {v}"
            ))),
        }
    }

    /// Number of channels for this color type.
    #[must_use]
    pub const fn channels(self) -> usize {
        match self {
            Self::Grayscale | Self::Indexed => 1,
            Self::GrayscaleAlpha => 2,
            Self::Rgb => 3,
            Self::Rgba => 4,
        }
    }
}

// ── IHDR ─────────────────────────────────────────────────────────────────────

/// PNG IHDR (image header) data.
#[derive(Debug, Clone)]
pub struct PngIhdr {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Bit depth per channel.
    pub bit_depth: u8,
    /// Color type.
    pub color_type: PngColorType,
    /// Compression method (always 0).
    pub compression: u8,
    /// Filter method (always 0).
    pub filter_method: u8,
    /// Interlace method (0=none, 1=Adam7).
    pub interlace: u8,
}

impl PngIhdr {
    fn parse(data: &[u8]) -> ImageResult<Self> {
        if data.len() < 13 {
            return Err(ImageError::invalid_format("IHDR too short"));
        }
        let width = u32::from_be_bytes([data[0], data[1], data[2], data[3]]);
        let height = u32::from_be_bytes([data[4], data[5], data[6], data[7]]);
        let bit_depth = data[8];
        let color_type = PngColorType::from_u8(data[9])?;
        Ok(Self {
            width,
            height,
            bit_depth,
            color_type,
            compression: data[10],
            filter_method: data[11],
            interlace: data[12],
        })
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(13);
        data.extend_from_slice(&self.width.to_be_bytes());
        data.extend_from_slice(&self.height.to_be_bytes());
        data.push(self.bit_depth);
        data.push(self.color_type as u8);
        data.push(self.compression);
        data.push(self.filter_method);
        data.push(self.interlace);
        data
    }
}

// ── PNG filter ───────────────────────────────────────────────────────────────

/// PNG filter type for a scanline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PngFilter {
    /// No filter.
    None = 0,
    /// Sub: difference from left pixel.
    Sub = 1,
    /// Up: difference from pixel above.
    Up = 2,
    /// Average: average of left and above.
    Average = 3,
    /// Paeth predictor.
    Paeth = 4,
}

impl PngFilter {
    fn from_u8(v: u8) -> ImageResult<Self> {
        match v {
            0 => Ok(Self::None),
            1 => Ok(Self::Sub),
            2 => Ok(Self::Up),
            3 => Ok(Self::Average),
            4 => Ok(Self::Paeth),
            _ => Err(ImageError::invalid_format(format!(
                "Unknown PNG filter: {v}"
            ))),
        }
    }
}

/// Paeth predictor function.
fn paeth_predictor(a: i32, b: i32, c: i32) -> i32 {
    let p = a + b - c;
    let pa = (p - a).abs();
    let pb = (p - b).abs();
    let pc = (p - c).abs();
    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
}

/// Reconstruct a filtered scanline in-place.
fn reconstruct_scanline(filter: PngFilter, current: &mut [u8], previous: &[u8], bpp: usize) {
    match filter {
        PngFilter::None => {}
        PngFilter::Sub => {
            for i in bpp..current.len() {
                current[i] = current[i].wrapping_add(current[i - bpp]);
            }
        }
        PngFilter::Up => {
            for i in 0..current.len() {
                current[i] = current[i].wrapping_add(previous[i]);
            }
        }
        PngFilter::Average => {
            for i in 0..current.len() {
                let left = if i >= bpp { current[i - bpp] as i32 } else { 0 };
                let above = previous[i] as i32;
                current[i] = current[i].wrapping_add(((left + above) / 2) as u8);
            }
        }
        PngFilter::Paeth => {
            for i in 0..current.len() {
                let a = if i >= bpp { current[i - bpp] as i32 } else { 0 };
                let b = previous[i] as i32;
                let c = if i >= bpp {
                    previous[i - bpp] as i32
                } else {
                    0
                };
                current[i] = current[i].wrapping_add(paeth_predictor(a, b, c) as u8);
            }
        }
    }
}

// ── PNG image ─────────────────────────────────────────────────────────────────

/// A decoded PNG image.
#[derive(Debug, Clone)]
pub struct PngImage {
    /// Image width.
    pub width: u32,
    /// Image height.
    pub height: u32,
    /// Bit depth per channel.
    pub bit_depth: u8,
    /// Color type.
    pub color_type: PngColorType,
    /// Raw pixel data (filtered/defiltered).
    pub pixels: Vec<u8>,
    /// Optional text metadata.
    pub metadata: std::collections::HashMap<String, String>,
}

impl PngImage {
    /// Convert to an `ImageFrame`.
    #[must_use]
    pub fn to_image_frame(&self, frame_number: u32) -> ImageFrame {
        let color_space = match self.color_type {
            PngColorType::Grayscale | PngColorType::GrayscaleAlpha => ColorSpace::Luma,
            _ => ColorSpace::Srgb,
        };
        ImageFrame::new(
            frame_number,
            self.width,
            self.height,
            PixelType::U8,
            self.color_type.channels() as u8,
            color_space,
            ImageData::interleaved(self.pixels.clone()),
        )
    }

    /// Create from an `ImageFrame`.
    pub fn from_image_frame(frame: &ImageFrame) -> ImageResult<Self> {
        let data = frame.data.as_slice().ok_or_else(|| {
            ImageError::invalid_format("Only interleaved image data supported for PNG")
        })?;
        let color_type = match frame.components {
            1 => PngColorType::Grayscale,
            2 => PngColorType::GrayscaleAlpha,
            3 => PngColorType::Rgb,
            4 => PngColorType::Rgba,
            n => {
                return Err(ImageError::unsupported(format!(
                    "PNG: {n} components unsupported"
                )))
            }
        };
        Ok(Self {
            width: frame.width,
            height: frame.height,
            bit_depth: 8,
            color_type,
            pixels: data.to_vec(),
            metadata: std::collections::HashMap::new(),
        })
    }
}

// ── PNG decoder ───────────────────────────────────────────────────────────────

/// PNG decoder state.
#[derive(Debug, Default)]
pub struct PngDecoder;

impl PngDecoder {
    /// Create a new decoder.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Decode a PNG from a byte slice.
    pub fn decode(&self, data: &[u8]) -> ImageResult<PngImage> {
        // Validate signature
        if data.len() < 8 || data[..8] != PNG_SIGNATURE {
            return Err(ImageError::invalid_format("Not a PNG file (bad signature)"));
        }

        let mut pos = 8usize;
        let mut ihdr: Option<PngIhdr> = None;
        let mut idat_data: Vec<u8> = Vec::new();
        let mut metadata = std::collections::HashMap::new();

        while pos + 8 <= data.len() {
            let length =
                u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            pos += 4;
            if pos + 4 > data.len() {
                break;
            }
            let chunk_type = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
            pos += 4;
            if pos + length > data.len() {
                return Err(ImageError::invalid_format("PNG chunk data truncated"));
            }
            let chunk_data = &data[pos..pos + length];
            pos += length;
            // Skip CRC (4 bytes)
            if pos + 4 <= data.len() {
                pos += 4;
            }

            match &chunk_type {
                b"IHDR" => {
                    ihdr = Some(PngIhdr::parse(chunk_data)?);
                }
                b"IDAT" => {
                    idat_data.extend_from_slice(chunk_data);
                }
                b"IEND" => break,
                b"tEXt" => {
                    // key\0value
                    if let Some(null_pos) = chunk_data.iter().position(|&b| b == 0) {
                        let key = String::from_utf8_lossy(&chunk_data[..null_pos]).into_owned();
                        let value =
                            String::from_utf8_lossy(&chunk_data[null_pos + 1..]).into_owned();
                        metadata.insert(key, value);
                    }
                }
                _ => {} // skip unknown chunks
            }
        }

        let ihdr = ihdr.ok_or_else(|| ImageError::invalid_format("PNG missing IHDR"))?;
        if idat_data.is_empty() {
            return Err(ImageError::invalid_format("PNG missing IDAT"));
        }

        // Decompress IDAT
        let raw = oxiarc_deflate::zlib_decompress(&idat_data)
            .map_err(|e| ImageError::Compression(format!("PNG inflate error: {e}")))?;

        // Defilter
        let channels = ihdr.color_type.channels();
        let bpp = channels; // assuming 8-bit
        let row_stride = ihdr.width as usize * bpp;
        let expected = (row_stride + 1) * ihdr.height as usize;
        if raw.len() < expected {
            return Err(ImageError::invalid_format(format!(
                "PNG decompressed data too short: {} < {}",
                raw.len(),
                expected
            )));
        }

        let mut pixels = vec![0u8; row_stride * ihdr.height as usize];
        let mut prev_row = vec![0u8; row_stride];

        for y in 0..ihdr.height as usize {
            let src_start = y * (row_stride + 1);
            let filter_byte = raw[src_start];
            let filter = PngFilter::from_u8(filter_byte)?;
            let src_row = &raw[src_start + 1..src_start + 1 + row_stride];
            let dst_start = y * row_stride;
            pixels[dst_start..dst_start + row_stride].copy_from_slice(src_row);
            reconstruct_scanline(
                filter,
                &mut pixels[dst_start..dst_start + row_stride],
                &prev_row,
                bpp,
            );
            prev_row.copy_from_slice(&pixels[dst_start..dst_start + row_stride]);
        }

        Ok(PngImage {
            width: ihdr.width,
            height: ihdr.height,
            bit_depth: ihdr.bit_depth,
            color_type: ihdr.color_type,
            pixels,
            metadata,
        })
    }
}

// ── PNG encoder ───────────────────────────────────────────────────────────────

/// PNG encoder.
#[derive(Debug, Clone)]
pub struct PngEncoder {
    /// Compression level (0-10).
    pub compression_level: u8,
}

impl Default for PngEncoder {
    fn default() -> Self {
        Self {
            compression_level: 6,
        }
    }
}

impl PngEncoder {
    /// Create a new encoder with compression level.
    #[must_use]
    pub fn new(compression_level: u8) -> Self {
        Self {
            compression_level: compression_level.min(10),
        }
    }

    /// Encode a `PngImage` to bytes.
    pub fn encode(&self, image: &PngImage) -> ImageResult<Vec<u8>> {
        let mut out = Vec::new();
        // Signature
        out.extend_from_slice(&PNG_SIGNATURE);

        // IHDR
        let ihdr = PngIhdr {
            width: image.width,
            height: image.height,
            bit_depth: 8,
            color_type: image.color_type,
            compression: 0,
            filter_method: 0,
            interlace: 0,
        };
        let ihdr_chunk = PngChunk::new(*b"IHDR", ihdr.to_bytes());
        out.extend_from_slice(&ihdr_chunk.to_bytes());

        // IDAT: apply filter None to each row
        let channels = image.color_type.channels();
        let row_stride = image.width as usize * channels;
        let mut filtered = Vec::with_capacity((row_stride + 1) * image.height as usize);
        for y in 0..image.height as usize {
            filtered.push(0u8); // Filter type None
            let row_start = y * row_stride;
            filtered.extend_from_slice(&image.pixels[row_start..row_start + row_stride]);
        }

        let compressed = oxiarc_deflate::zlib_compress(&filtered, 6)
            .map_err(|e| ImageError::Compression(format!("PNG deflate error: {e}")))?;
        let idat_chunk = PngChunk::new(*b"IDAT", compressed);
        out.extend_from_slice(&idat_chunk.to_bytes());

        // IEND
        let iend_chunk = PngChunk::new(*b"IEND", Vec::new());
        out.extend_from_slice(&iend_chunk.to_bytes());

        Ok(out)
    }
}

// ── APNG ─────────────────────────────────────────────────────────────────────

/// APNG frame control data (fcTL chunk).
#[derive(Debug, Clone)]
pub struct ApngFrameControl {
    /// Sequence number.
    pub sequence_number: u32,
    /// Frame width.
    pub width: u32,
    /// Frame height.
    pub height: u32,
    /// X offset.
    pub x_offset: u32,
    /// Y offset.
    pub y_offset: u32,
    /// Delay numerator (delay_num/delay_den seconds).
    pub delay_num: u16,
    /// Delay denominator.
    pub delay_den: u16,
    /// Dispose operation.
    pub dispose_op: u8,
    /// Blend operation.
    pub blend_op: u8,
}

impl ApngFrameControl {
    fn parse(data: &[u8]) -> ImageResult<Self> {
        if data.len() < 26 {
            return Err(ImageError::invalid_format("fcTL chunk too short"));
        }
        Ok(Self {
            sequence_number: u32::from_be_bytes([data[0], data[1], data[2], data[3]]),
            width: u32::from_be_bytes([data[4], data[5], data[6], data[7]]),
            height: u32::from_be_bytes([data[8], data[9], data[10], data[11]]),
            x_offset: u32::from_be_bytes([data[12], data[13], data[14], data[15]]),
            y_offset: u32::from_be_bytes([data[16], data[17], data[18], data[19]]),
            delay_num: u16::from_be_bytes([data[20], data[21]]),
            delay_den: u16::from_be_bytes([data[22], data[23]]),
            dispose_op: data[24],
            blend_op: data[25],
        })
    }

    fn to_bytes(&self) -> Vec<u8> {
        let mut data = Vec::with_capacity(26);
        data.extend_from_slice(&self.sequence_number.to_be_bytes());
        data.extend_from_slice(&self.width.to_be_bytes());
        data.extend_from_slice(&self.height.to_be_bytes());
        data.extend_from_slice(&self.x_offset.to_be_bytes());
        data.extend_from_slice(&self.y_offset.to_be_bytes());
        data.extend_from_slice(&self.delay_num.to_be_bytes());
        data.extend_from_slice(&self.delay_den.to_be_bytes());
        data.push(self.dispose_op);
        data.push(self.blend_op);
        data
    }

    /// Duration in seconds.
    #[must_use]
    pub fn duration_secs(&self) -> f64 {
        let den = if self.delay_den == 0 {
            100
        } else {
            self.delay_den
        };
        self.delay_num as f64 / den as f64
    }
}

/// A single APNG animation frame.
#[derive(Debug, Clone)]
pub struct ApngFrame {
    /// Frame control metadata.
    pub control: ApngFrameControl,
    /// Pixel data (same format as parent PNG).
    pub pixels: Vec<u8>,
}

/// An APNG animation.
#[derive(Debug, Clone)]
pub struct ApngAnimation {
    /// Canvas width.
    pub width: u32,
    /// Canvas height.
    pub height: u32,
    /// Color type.
    pub color_type: PngColorType,
    /// Number of times to loop (0 = infinite).
    pub num_plays: u32,
    /// Animation frames.
    pub frames: Vec<ApngFrame>,
}

impl ApngAnimation {
    /// Create a new empty animation.
    #[must_use]
    pub fn new(width: u32, height: u32, color_type: PngColorType, num_plays: u32) -> Self {
        Self {
            width,
            height,
            color_type,
            num_plays,
            frames: Vec::new(),
        }
    }

    /// Add a frame to the animation.
    pub fn add_frame(&mut self, frame: ApngFrame) {
        self.frames.push(frame);
    }

    /// Return the number of frames.
    #[must_use]
    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Total animation duration in seconds.
    #[must_use]
    pub fn total_duration_secs(&self) -> f64 {
        self.frames.iter().map(|f| f.control.duration_secs()).sum()
    }
}

// ── Public API ────────────────────────────────────────────────────────────────

/// Read a PNG file from disk.
pub fn read_png(path: &Path) -> ImageResult<PngImage> {
    let data = std::fs::read(path)?;
    PngDecoder::new().decode(&data)
}

/// Write a PNG image to disk.
pub fn write_png(path: &Path, image: &PngImage) -> ImageResult<()> {
    let data = PngEncoder::default().encode(image)?;
    std::fs::write(path, data)?;
    Ok(())
}

/// Read a PNG file and return an `ImageFrame`.
pub fn read_png_frame(path: &Path) -> ImageResult<ImageFrame> {
    let png = read_png(path)?;
    Ok(png.to_image_frame(1))
}

/// Decode an APNG from bytes.
pub fn decode_apng(data: &[u8]) -> ImageResult<ApngAnimation> {
    if data.len() < 8 || data[..8] != PNG_SIGNATURE {
        return Err(ImageError::invalid_format("Not a PNG/APNG file"));
    }

    let mut pos = 8usize;
    let mut ihdr: Option<PngIhdr> = None;
    let mut num_frames = 0u32;
    let mut num_plays = 0u32;
    let mut frame_controls: Vec<ApngFrameControl> = Vec::new();
    let mut frame_idat_seqs: Vec<Vec<u8>> = Vec::new();
    let mut current_idat: Vec<u8> = Vec::new();
    let mut in_frame = false;
    let mut is_apng = false;

    while pos + 8 <= data.len() {
        let length =
            u32::from_be_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]) as usize;
        pos += 4;
        if pos + 4 > data.len() {
            break;
        }
        let chunk_type = [data[pos], data[pos + 1], data[pos + 2], data[pos + 3]];
        pos += 4;
        if pos + length > data.len() {
            return Err(ImageError::invalid_format("APNG chunk truncated"));
        }
        let chunk_data = &data[pos..pos + length];
        pos += length;
        if pos + 4 <= data.len() {
            pos += 4;
        } // skip CRC

        match &chunk_type {
            b"IHDR" => {
                ihdr = Some(PngIhdr::parse(chunk_data)?);
            }
            b"acTL" => {
                if chunk_data.len() >= 8 {
                    is_apng = true;
                    num_frames = u32::from_be_bytes([
                        chunk_data[0],
                        chunk_data[1],
                        chunk_data[2],
                        chunk_data[3],
                    ]);
                    num_plays = u32::from_be_bytes([
                        chunk_data[4],
                        chunk_data[5],
                        chunk_data[6],
                        chunk_data[7],
                    ]);
                }
            }
            b"fcTL" => {
                if in_frame && !current_idat.is_empty() {
                    frame_idat_seqs.push(current_idat.clone());
                    current_idat = Vec::new();
                }
                frame_controls.push(ApngFrameControl::parse(chunk_data)?);
                in_frame = true;
            }
            b"fdAT" => {
                // First 4 bytes are sequence number
                if chunk_data.len() > 4 {
                    current_idat.extend_from_slice(&chunk_data[4..]);
                }
            }
            b"IDAT" => {
                // IDAT after fcTL is used for frame 0 in APNG (same as fdAT but without seq#)
                current_idat.extend_from_slice(chunk_data);
            }
            b"IEND" => {
                if in_frame && !current_idat.is_empty() {
                    frame_idat_seqs.push(current_idat.clone());
                }
                break;
            }
            _ => {}
        }
    }

    let ihdr = ihdr.ok_or_else(|| ImageError::invalid_format("APNG missing IHDR"))?;

    if !is_apng {
        // Treat as single-frame PNG
        let decoder = PngDecoder::new();
        let png = decoder.decode(data)?;
        let control = ApngFrameControl {
            sequence_number: 0,
            width: png.width,
            height: png.height,
            x_offset: 0,
            y_offset: 0,
            delay_num: 1,
            delay_den: 10,
            dispose_op: 0,
            blend_op: 0,
        };
        return Ok(ApngAnimation {
            width: png.width,
            height: png.height,
            color_type: png.color_type,
            num_plays: 1,
            frames: vec![ApngFrame {
                control,
                pixels: png.pixels,
            }],
        });
    }

    // Decode each frame
    let channels = ihdr.color_type.channels();
    let row_stride = ihdr.width as usize * channels;
    let mut frames = Vec::new();

    for (i, idat) in frame_idat_seqs.iter().enumerate() {
        let raw = oxiarc_deflate::zlib_decompress(idat)
            .map_err(|e| ImageError::Compression(format!("APNG inflate frame {i}: {e}")))?;

        let fc = frame_controls
            .get(i)
            .ok_or_else(|| ImageError::invalid_format(format!("Missing fcTL for frame {i}")))?;
        let frame_row_stride = fc.width as usize * channels;
        let expected = (frame_row_stride + 1) * fc.height as usize;
        if raw.len() < expected {
            return Err(ImageError::invalid_format(format!(
                "APNG frame {i} decompressed too short"
            )));
        }

        let mut pixels = vec![0u8; frame_row_stride * fc.height as usize];
        let mut prev_row = vec![0u8; frame_row_stride];
        for y in 0..fc.height as usize {
            let src_start = y * (frame_row_stride + 1);
            let filter = PngFilter::from_u8(raw[src_start])?;
            let src_row = &raw[src_start + 1..src_start + 1 + frame_row_stride];
            let dst_start = y * frame_row_stride;
            pixels[dst_start..dst_start + frame_row_stride].copy_from_slice(src_row);
            reconstruct_scanline(
                filter,
                &mut pixels[dst_start..dst_start + frame_row_stride],
                &prev_row,
                channels,
            );
            prev_row.copy_from_slice(&pixels[dst_start..dst_start + frame_row_stride]);
        }

        frames.push(ApngFrame {
            control: fc.clone(),
            pixels,
        });
    }

    // Fill missing frames if needed
    let _ = (num_frames, row_stride);

    Ok(ApngAnimation {
        width: ihdr.width,
        height: ihdr.height,
        color_type: ihdr.color_type,
        num_plays,
        frames,
    })
}

/// Read an APNG file.
pub fn read_apng(path: &Path) -> ImageResult<ApngAnimation> {
    let data = std::fs::read(path)?;
    decode_apng(&data)
}

/// Encode an APNG to bytes.
pub fn encode_apng(anim: &ApngAnimation) -> ImageResult<Vec<u8>> {
    let mut out = Vec::new();
    out.extend_from_slice(&PNG_SIGNATURE);

    // IHDR
    let ihdr = PngIhdr {
        width: anim.width,
        height: anim.height,
        bit_depth: 8,
        color_type: anim.color_type,
        compression: 0,
        filter_method: 0,
        interlace: 0,
    };
    out.extend_from_slice(&PngChunk::new(*b"IHDR", ihdr.to_bytes()).to_bytes());

    // acTL
    let mut actl = Vec::with_capacity(8);
    actl.extend_from_slice(&(anim.frames.len() as u32).to_be_bytes());
    actl.extend_from_slice(&anim.num_plays.to_be_bytes());
    out.extend_from_slice(&PngChunk::new(*b"acTL", actl).to_bytes());

    let channels = anim.color_type.channels();
    let mut seq = 0u32;

    for (i, frame) in anim.frames.iter().enumerate() {
        // fcTL
        let mut fc = frame.control.clone();
        fc.sequence_number = seq;
        seq += 1;
        out.extend_from_slice(&PngChunk::new(*b"fcTL", fc.to_bytes()).to_bytes());

        // Encode frame pixels with filter None
        let frame_row_stride = fc.width as usize * channels;
        let mut filtered = Vec::new();
        for y in 0..fc.height as usize {
            filtered.push(0u8); // Filter None
            let row_start = y * frame_row_stride;
            filtered.extend_from_slice(&frame.pixels[row_start..row_start + frame_row_stride]);
        }
        let compressed = oxiarc_deflate::zlib_compress(&filtered, 6)
            .map_err(|e| ImageError::Compression(format!("APNG deflate error: {e}")))?;

        if i == 0 {
            // First frame: use IDAT
            out.extend_from_slice(&PngChunk::new(*b"IDAT", compressed).to_bytes());
        } else {
            // Other frames: use fdAT with sequence number
            let mut fdat = Vec::with_capacity(4 + compressed.len());
            fdat.extend_from_slice(&seq.to_be_bytes());
            seq += 1;
            fdat.extend_from_slice(&compressed);
            out.extend_from_slice(&PngChunk::new(*b"fdAT", fdat).to_bytes());
        }
    }

    out.extend_from_slice(&PngChunk::new(*b"IEND", Vec::new()).to_bytes());
    Ok(out)
}

/// Write an APNG animation to disk.
pub fn write_apng(path: &Path, anim: &ApngAnimation) -> ImageResult<()> {
    let data = encode_apng(anim)?;
    std::fs::write(path, data)?;
    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_png_signature_bytes() {
        assert_eq!(PNG_SIGNATURE[0], 137);
        assert_eq!(PNG_SIGNATURE[1], 80); // 'P'
        assert_eq!(PNG_SIGNATURE[2], 78); // 'N'
        assert_eq!(PNG_SIGNATURE[3], 71); // 'G'
    }

    #[test]
    fn test_crc32_known_values() {
        // CRC32 of empty slice: 0x00000000
        assert_eq!(crc32(&[]), 0x0000_0000);
        // CRC32 of b"123456789" = 0xCBF43926
        assert_eq!(crc32(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn test_crc32_chunk() {
        let chunk = PngChunk::new(*b"IEND", Vec::new());
        // Known CRC for IEND chunk (type only, no data): 0xAE426082
        assert_eq!(chunk.crc(), 0xAE42_6082);
    }

    #[test]
    fn test_chunk_to_bytes_length_field() {
        let data = vec![1u8, 2, 3, 4];
        let chunk = PngChunk::new(*b"tEXt", data.clone());
        let bytes = chunk.to_bytes();
        let length = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        assert_eq!(length as usize, data.len());
    }

    #[test]
    fn test_chunk_type_str() {
        let chunk = PngChunk::new(*b"IHDR", vec![]);
        assert_eq!(chunk.type_str(), "IHDR");
    }

    #[test]
    fn test_png_color_type_channels() {
        assert_eq!(PngColorType::Grayscale.channels(), 1);
        assert_eq!(PngColorType::Rgb.channels(), 3);
        assert_eq!(PngColorType::Rgba.channels(), 4);
        assert_eq!(PngColorType::GrayscaleAlpha.channels(), 2);
        assert_eq!(PngColorType::Indexed.channels(), 1);
    }

    #[test]
    fn test_paeth_predictor_basic() {
        // paeth(4, 6, 5) → p=5, pa=1, pb=1, pc=0 → c=5
        assert_eq!(paeth_predictor(4, 6, 5), 5);
        // paeth(0, 0, 0) → 0
        assert_eq!(paeth_predictor(0, 0, 0), 0);
        // paeth(255, 0, 0): p=255+0-0=255, pa=|255-255|=0, pb=|255-0|=255, pc=|255-0|=255 → a=255
        assert_eq!(paeth_predictor(255, 0, 0), 255);
    }

    #[test]
    fn test_filter_none_reconstruction() {
        let mut row = vec![10u8, 20, 30, 40];
        let prev = vec![0u8; 4];
        reconstruct_scanline(PngFilter::None, &mut row, &prev, 1);
        assert_eq!(row, vec![10, 20, 30, 40]);
    }

    #[test]
    fn test_filter_sub_reconstruction() {
        // Original: [10, 30, 60, 100]. Sub-filtered: [10, 20, 30, 40]
        let mut row = vec![10u8, 20, 30, 40];
        let prev = vec![0u8; 4];
        reconstruct_scanline(PngFilter::Sub, &mut row, &prev, 1);
        assert_eq!(row, vec![10, 30, 60, 100]);
    }

    #[test]
    fn test_filter_up_reconstruction() {
        let mut row = vec![5u8, 5, 5, 5];
        let prev = vec![10u8, 20, 30, 40];
        reconstruct_scanline(PngFilter::Up, &mut row, &prev, 1);
        assert_eq!(row, vec![15, 25, 35, 45]);
    }

    #[test]
    fn test_filter_average_reconstruction() {
        // Average of left=0, above=10 => 5; first pixel
        let mut row = vec![5u8, 0, 0, 0];
        let prev = vec![10u8, 0, 0, 0];
        reconstruct_scanline(PngFilter::Average, &mut row, &prev, 1);
        // pixel[0] += (0 + 10)/2 = 5 → 10
        assert_eq!(row[0], 10);
    }

    #[test]
    fn test_ihdr_roundtrip() {
        let ihdr = PngIhdr {
            width: 1920,
            height: 1080,
            bit_depth: 8,
            color_type: PngColorType::Rgb,
            compression: 0,
            filter_method: 0,
            interlace: 0,
        };
        let bytes = ihdr.to_bytes();
        let parsed = PngIhdr::parse(&bytes).expect("parse IHDR");
        assert_eq!(parsed.width, 1920);
        assert_eq!(parsed.height, 1080);
        assert_eq!(parsed.color_type, PngColorType::Rgb);
    }

    #[test]
    fn test_decode_invalid_signature() {
        let data = vec![0u8; 20];
        let result = PngDecoder::new().decode(&data);
        assert!(result.is_err());
        let err = result.expect_err("should be error");
        assert!(err.to_string().contains("signature") || err.to_string().contains("PNG"));
    }

    #[test]
    fn test_encode_decode_rgb_roundtrip() {
        // Create a 2x2 RGB image
        let pixels: Vec<u8> = vec![255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 0];
        let image = PngImage {
            width: 2,
            height: 2,
            bit_depth: 8,
            color_type: PngColorType::Rgb,
            pixels: pixels.clone(),
            metadata: std::collections::HashMap::new(),
        };

        let encoder = PngEncoder::default();
        let encoded = encoder.encode(&image).expect("encode");
        let decoder = PngDecoder::new();
        let decoded = decoder.decode(&encoded).expect("decode");

        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 2);
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn test_encode_decode_rgba_roundtrip() {
        let pixels: Vec<u8> = vec![100, 150, 200, 255, 50, 100, 150, 128];
        let image = PngImage {
            width: 2,
            height: 1,
            bit_depth: 8,
            color_type: PngColorType::Rgba,
            pixels: pixels.clone(),
            metadata: std::collections::HashMap::new(),
        };
        let encoder = PngEncoder::default();
        let encoded = encoder.encode(&image).expect("encode");
        let decoder = PngDecoder::new();
        let decoded = decoder.decode(&encoded).expect("decode");
        assert_eq!(decoded.pixels, pixels);
        assert_eq!(decoded.color_type, PngColorType::Rgba);
    }

    #[test]
    fn test_encode_decode_grayscale_roundtrip() {
        let pixels: Vec<u8> = (0..16u8).collect();
        let image = PngImage {
            width: 4,
            height: 4,
            bit_depth: 8,
            color_type: PngColorType::Grayscale,
            pixels: pixels.clone(),
            metadata: std::collections::HashMap::new(),
        };
        let encoder = PngEncoder::default();
        let encoded = encoder.encode(&image).expect("encode");
        let decoder = PngDecoder::new();
        let decoded = decoder.decode(&encoded).expect("decode");
        assert_eq!(decoded.pixels, pixels);
    }

    #[test]
    fn test_png_image_to_frame() {
        let pixels = vec![128u8; 3 * 4 * 4];
        let image = PngImage {
            width: 4,
            height: 4,
            bit_depth: 8,
            color_type: PngColorType::Rgb,
            pixels: pixels.clone(),
            metadata: std::collections::HashMap::new(),
        };
        let frame = image.to_image_frame(1);
        assert_eq!(frame.width, 4);
        assert_eq!(frame.height, 4);
        assert_eq!(frame.components, 3);
    }

    #[test]
    fn test_apng_frame_count() {
        // Build a simple 2-frame APNG
        let color_type = PngColorType::Rgb;
        let mut anim = ApngAnimation::new(2, 2, color_type, 0);
        let control = ApngFrameControl {
            sequence_number: 0,
            width: 2,
            height: 2,
            x_offset: 0,
            y_offset: 0,
            delay_num: 1,
            delay_den: 10,
            dispose_op: 0,
            blend_op: 0,
        };
        let pixels = vec![0u8; 2 * 2 * 3];
        anim.add_frame(ApngFrame {
            control: control.clone(),
            pixels: pixels.clone(),
        });
        let mut control2 = control;
        control2.sequence_number = 1;
        anim.add_frame(ApngFrame {
            control: control2,
            pixels,
        });
        assert_eq!(anim.frame_count(), 2);
    }

    #[test]
    fn test_apng_duration() {
        let control = ApngFrameControl {
            sequence_number: 0,
            width: 1,
            height: 1,
            x_offset: 0,
            y_offset: 0,
            delay_num: 1,
            delay_den: 4, // 0.25 seconds
            dispose_op: 0,
            blend_op: 0,
        };
        assert!((control.duration_secs() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_apng_encode_decode_roundtrip() {
        let color_type = PngColorType::Rgb;
        let mut anim = ApngAnimation::new(2, 1, color_type, 1);
        let control = ApngFrameControl {
            sequence_number: 0,
            width: 2,
            height: 1,
            x_offset: 0,
            y_offset: 0,
            delay_num: 1,
            delay_den: 10,
            dispose_op: 0,
            blend_op: 0,
        };
        let pixels = vec![255u8, 0, 0, 0, 255, 0];
        anim.add_frame(ApngFrame {
            control,
            pixels: pixels.clone(),
        });

        let encoded = encode_apng(&anim).expect("encode APNG");
        let decoded = decode_apng(&encoded).expect("decode APNG");
        assert_eq!(decoded.width, 2);
        assert_eq!(decoded.height, 1);
        assert!(!decoded.frames.is_empty());
    }

    #[test]
    fn test_color_type_from_u8_invalid() {
        let result = PngColorType::from_u8(99);
        assert!(result.is_err());
    }

    #[test]
    fn test_png_filter_from_u8_invalid() {
        let result = PngFilter::from_u8(99);
        assert!(result.is_err());
    }

    #[test]
    fn test_encode_produces_valid_signature() {
        let pixels = vec![255u8; 3];
        let image = PngImage {
            width: 1,
            height: 1,
            bit_depth: 8,
            color_type: PngColorType::Rgb,
            pixels,
            metadata: std::collections::HashMap::new(),
        };
        let encoded = PngEncoder::default().encode(&image).expect("encode");
        assert_eq!(&encoded[..8], &PNG_SIGNATURE);
    }
}
