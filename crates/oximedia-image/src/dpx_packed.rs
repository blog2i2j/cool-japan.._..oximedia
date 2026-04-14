//! 10-bit packed pixel read/write for DPX files.
//!
//! Implements the two standard packing methods defined in SMPTE 268M:
//!
//! - **Method A** (filled): 3 × 10-bit samples packed into a 32-bit word,
//!   with the 2 least-significant bits unused (set to 0).
//!   Layout: `[S0:10][S1:10][S2:10][pad:2]`
//!
//! - **Method B** (padded): 3 × 10-bit samples packed into a 32-bit word,
//!   with the 2 most-significant bits unused (set to 0).
//!   Layout: `[pad:2][S0:10][S1:10][S2:10]`
//!
//! Both methods store 3 samples (one pixel of RGB) per 32-bit word, giving
//! a 33% overhead vs. theoretical 30-bit packing.
//!
//! Endianness is handled at the word level — DPX headers specify whether
//! the file is big-endian or little-endian, and we convert accordingly.

#![allow(dead_code)]

use crate::error::{ImageError, ImageResult};
use crate::Endian;

// ── Packing method ──────────────────────────────────────────────────────────

/// DPX 10-bit packing method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackingMethod {
    /// Method A: padding in the 2 LSBs.
    /// Word layout (MSB first): `[S0:10][S1:10][S2:10][00:2]`
    MethodA,
    /// Method B: padding in the 2 MSBs.
    /// Word layout (MSB first): `[00:2][S0:10][S1:10][S2:10]`
    MethodB,
}

// ── Pack/unpack core ────────────────────────────────────────────────────────

/// Pack three 10-bit samples into a single 32-bit word using Method A.
///
/// Each sample must be in 0..1024.
#[must_use]
pub fn pack_method_a(s0: u16, s1: u16, s2: u16) -> u32 {
    let s0 = (s0 & 0x3FF) as u32;
    let s1 = (s1 & 0x3FF) as u32;
    let s2 = (s2 & 0x3FF) as u32;
    (s0 << 22) | (s1 << 12) | (s2 << 2)
}

/// Unpack a 32-bit word into three 10-bit samples using Method A.
#[must_use]
pub fn unpack_method_a(word: u32) -> (u16, u16, u16) {
    let s0 = ((word >> 22) & 0x3FF) as u16;
    let s1 = ((word >> 12) & 0x3FF) as u16;
    let s2 = ((word >> 2) & 0x3FF) as u16;
    (s0, s1, s2)
}

/// Pack three 10-bit samples into a single 32-bit word using Method B.
#[must_use]
pub fn pack_method_b(s0: u16, s1: u16, s2: u16) -> u32 {
    let s0 = (s0 & 0x3FF) as u32;
    let s1 = (s1 & 0x3FF) as u32;
    let s2 = (s2 & 0x3FF) as u32;
    (s0 << 20) | (s1 << 10) | s2
}

/// Unpack a 32-bit word into three 10-bit samples using Method B.
#[must_use]
pub fn unpack_method_b(word: u32) -> (u16, u16, u16) {
    let s0 = ((word >> 20) & 0x3FF) as u16;
    let s1 = ((word >> 10) & 0x3FF) as u16;
    let s2 = (word & 0x3FF) as u16;
    (s0, s1, s2)
}

/// Generic pack dispatcher.
#[must_use]
pub fn pack_triple(method: PackingMethod, s0: u16, s1: u16, s2: u16) -> u32 {
    match method {
        PackingMethod::MethodA => pack_method_a(s0, s1, s2),
        PackingMethod::MethodB => pack_method_b(s0, s1, s2),
    }
}

/// Generic unpack dispatcher.
#[must_use]
pub fn unpack_triple(method: PackingMethod, word: u32) -> (u16, u16, u16) {
    match method {
        PackingMethod::MethodA => unpack_method_a(word),
        PackingMethod::MethodB => unpack_method_b(word),
    }
}

// ── Endian-aware word I/O ───────────────────────────────────────────────────

/// Read a 32-bit word from a byte slice at the given offset, respecting endianness.
pub fn read_word(data: &[u8], offset: usize, endian: Endian) -> ImageResult<u32> {
    if offset + 4 > data.len() {
        return Err(ImageError::invalid_format(
            "Unexpected end of data reading 32-bit word",
        ));
    }
    let bytes = [
        data[offset],
        data[offset + 1],
        data[offset + 2],
        data[offset + 3],
    ];
    Ok(match endian {
        Endian::Big => u32::from_be_bytes(bytes),
        Endian::Little => u32::from_le_bytes(bytes),
    })
}

/// Write a 32-bit word into a byte buffer, respecting endianness.
pub fn write_word(buf: &mut Vec<u8>, word: u32, endian: Endian) {
    match endian {
        Endian::Big => buf.extend_from_slice(&word.to_be_bytes()),
        Endian::Little => buf.extend_from_slice(&word.to_le_bytes()),
    }
}

// ── Scanline-level operations ───────────────────────────────────────────────

/// A 10-bit packed scanline buffer holding RGB pixel data.
#[derive(Debug, Clone)]
pub struct PackedScanline {
    /// Width in pixels (each pixel = 3 samples = 1 word).
    pub width: u32,
    /// Packed 32-bit words.
    pub words: Vec<u32>,
    /// Packing method used.
    pub method: PackingMethod,
}

impl PackedScanline {
    /// Create a new empty scanline.
    #[must_use]
    pub fn new(width: u32, method: PackingMethod) -> Self {
        Self {
            width,
            words: vec![0u32; width as usize],
            method,
        }
    }

    /// Decode all pixels to 16-bit RGB triplets (values in 0..1024).
    #[must_use]
    pub fn decode_to_u16(&self) -> Vec<[u16; 3]> {
        self.words
            .iter()
            .map(|&w| {
                let (r, g, b) = unpack_triple(self.method, w);
                [r, g, b]
            })
            .collect()
    }

    /// Encode a slice of RGB triplets into this scanline.
    pub fn encode_from_u16(&mut self, pixels: &[[u16; 3]]) -> ImageResult<()> {
        if pixels.len() != self.width as usize {
            return Err(ImageError::InvalidDimensions(pixels.len() as u32, 1));
        }
        self.words = pixels
            .iter()
            .map(|px| pack_triple(self.method, px[0], px[1], px[2]))
            .collect();
        Ok(())
    }

    /// Get a single pixel as an RGB triplet.
    pub fn get_pixel(&self, x: u32) -> ImageResult<[u16; 3]> {
        if x >= self.width {
            return Err(ImageError::InvalidDimensions(x, 0));
        }
        let (r, g, b) = unpack_triple(self.method, self.words[x as usize]);
        Ok([r, g, b])
    }

    /// Set a single pixel from an RGB triplet.
    pub fn set_pixel(&mut self, x: u32, rgb: [u16; 3]) -> ImageResult<()> {
        if x >= self.width {
            return Err(ImageError::InvalidDimensions(x, 0));
        }
        self.words[x as usize] = pack_triple(self.method, rgb[0], rgb[1], rgb[2]);
        Ok(())
    }

    /// Read a packed scanline from raw bytes.
    pub fn from_bytes(
        data: &[u8],
        offset: usize,
        width: u32,
        method: PackingMethod,
        endian: Endian,
    ) -> ImageResult<Self> {
        let needed = width as usize * 4;
        if offset + needed > data.len() {
            return Err(ImageError::invalid_format(format!(
                "Need {} bytes for scanline, only {} available",
                needed,
                data.len().saturating_sub(offset)
            )));
        }
        let mut words = Vec::with_capacity(width as usize);
        for i in 0..width as usize {
            words.push(read_word(data, offset + i * 4, endian)?);
        }
        Ok(Self {
            width,
            words,
            method,
        })
    }

    /// Write this scanline as raw bytes.
    #[must_use]
    pub fn to_bytes(&self, endian: Endian) -> Vec<u8> {
        let mut buf = Vec::with_capacity(self.words.len() * 4);
        for &w in &self.words {
            write_word(&mut buf, w, endian);
        }
        buf
    }
}

// ── Frame-level packed image ────────────────────────────────────────────────

/// A complete 10-bit packed DPX image.
#[derive(Debug, Clone)]
pub struct PackedImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Packing method.
    pub method: PackingMethod,
    /// Endianness.
    pub endian: Endian,
    /// All scanlines.
    pub scanlines: Vec<PackedScanline>,
}

impl PackedImage {
    /// Create a new black packed image.
    #[must_use]
    pub fn new(width: u32, height: u32, method: PackingMethod, endian: Endian) -> Self {
        let scanlines = (0..height)
            .map(|_| PackedScanline::new(width, method))
            .collect();
        Self {
            width,
            height,
            method,
            endian,
            scanlines,
        }
    }

    /// Get a pixel at (x, y).
    pub fn get_pixel(&self, x: u32, y: u32) -> ImageResult<[u16; 3]> {
        if y >= self.height {
            return Err(ImageError::InvalidDimensions(x, y));
        }
        self.scanlines[y as usize].get_pixel(x)
    }

    /// Set a pixel at (x, y).
    pub fn set_pixel(&mut self, x: u32, y: u32, rgb: [u16; 3]) -> ImageResult<()> {
        if y >= self.height {
            return Err(ImageError::InvalidDimensions(x, y));
        }
        self.scanlines[y as usize].set_pixel(x, rgb)
    }

    /// Decode the entire image to unpacked u16 RGB data (row-major).
    #[must_use]
    pub fn decode_all(&self) -> Vec<[u16; 3]> {
        let mut out = Vec::with_capacity(self.width as usize * self.height as usize);
        for sl in &self.scanlines {
            out.extend_from_slice(&sl.decode_to_u16());
        }
        out
    }

    /// Encode from a flat slice of RGB triplets.
    pub fn encode_all(&mut self, pixels: &[[u16; 3]]) -> ImageResult<()> {
        let expected = self.width as usize * self.height as usize;
        if pixels.len() != expected {
            return Err(ImageError::invalid_format(format!(
                "Expected {expected} pixels, got {}",
                pixels.len()
            )));
        }
        let w = self.width as usize;
        for (y, sl) in self.scanlines.iter_mut().enumerate() {
            let row = &pixels[y * w..(y + 1) * w];
            sl.encode_from_u16(row)?;
        }
        Ok(())
    }

    /// Serialize the entire image to bytes.
    #[must_use]
    pub fn to_bytes(&self) -> Vec<u8> {
        let total = self.width as usize * self.height as usize * 4;
        let mut buf = Vec::with_capacity(total);
        for sl in &self.scanlines {
            buf.extend_from_slice(&sl.to_bytes(self.endian));
        }
        buf
    }

    /// Deserialize from raw bytes.
    pub fn from_bytes(
        data: &[u8],
        width: u32,
        height: u32,
        method: PackingMethod,
        endian: Endian,
    ) -> ImageResult<Self> {
        let row_bytes = width as usize * 4;
        let total_needed = row_bytes * height as usize;
        if data.len() < total_needed {
            return Err(ImageError::invalid_format(format!(
                "Need {total_needed} bytes, got {}",
                data.len()
            )));
        }
        let mut scanlines = Vec::with_capacity(height as usize);
        for y in 0..height as usize {
            scanlines.push(PackedScanline::from_bytes(
                data,
                y * row_bytes,
                width,
                method,
                endian,
            )?);
        }
        Ok(Self {
            width,
            height,
            method,
            endian,
            scanlines,
        })
    }

    /// Convert 10-bit samples to normalized f32 [0.0, 1.0].
    #[must_use]
    pub fn to_f32_normalized(&self) -> Vec<[f32; 3]> {
        self.decode_all()
            .iter()
            .map(|px| {
                [
                    px[0] as f32 / 1023.0,
                    px[1] as f32 / 1023.0,
                    px[2] as f32 / 1023.0,
                ]
            })
            .collect()
    }

    /// Create from normalized f32 data, quantizing to 10-bit.
    pub fn from_f32_normalized(
        pixels: &[[f32; 3]],
        width: u32,
        height: u32,
        method: PackingMethod,
        endian: Endian,
    ) -> ImageResult<Self> {
        let quantized: Vec<[u16; 3]> = pixels
            .iter()
            .map(|px| {
                [
                    (px[0].clamp(0.0, 1.0) * 1023.0 + 0.5) as u16,
                    (px[1].clamp(0.0, 1.0) * 1023.0 + 0.5) as u16,
                    (px[2].clamp(0.0, 1.0) * 1023.0 + 0.5) as u16,
                ]
            })
            .collect();
        let mut img = Self::new(width, height, method, endian);
        img.encode_all(&quantized)?;
        Ok(img)
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_method_a_pack_unpack() {
        let (s0, s1, s2) = (1023, 512, 0);
        let word = pack_method_a(s0, s1, s2);
        let (r0, r1, r2) = unpack_method_a(word);
        assert_eq!((r0, r1, r2), (1023, 512, 0));
    }

    #[test]
    fn test_method_b_pack_unpack() {
        let (s0, s1, s2) = (0, 341, 682);
        let word = pack_method_b(s0, s1, s2);
        let (r0, r1, r2) = unpack_method_b(word);
        assert_eq!((r0, r1, r2), (0, 341, 682));
    }

    #[test]
    fn test_method_a_bit_layout() {
        // S0=0x3FF at bits[31:22], S1=0 at bits[21:12], S2=0 at bits[11:2]
        let word = pack_method_a(0x3FF, 0, 0);
        assert_eq!(word, 0xFFC0_0000);
        // S0=0, S1=0x3FF, S2=0
        let word = pack_method_a(0, 0x3FF, 0);
        assert_eq!(word, 0x003F_F000);
        // S0=0, S1=0, S2=0x3FF
        let word = pack_method_a(0, 0, 0x3FF);
        assert_eq!(word, 0x0000_0FFC);
    }

    #[test]
    fn test_method_b_bit_layout() {
        // S0=0x3FF at bits[29:20], S1=0 at bits[19:10], S2=0 at bits[9:0]
        let word = pack_method_b(0x3FF, 0, 0);
        assert_eq!(word, 0x3FF0_0000);
        // S0=0, S1=0x3FF, S2=0
        let word = pack_method_b(0, 0x3FF, 0);
        assert_eq!(word, 0x000F_FC00);
        // S0=0, S1=0, S2=0x3FF
        let word = pack_method_b(0, 0, 0x3FF);
        assert_eq!(word, 0x0000_03FF);
    }

    #[test]
    fn test_generic_dispatch() {
        let word_a = pack_triple(PackingMethod::MethodA, 100, 200, 300);
        let (a0, a1, a2) = unpack_triple(PackingMethod::MethodA, word_a);
        assert_eq!((a0, a1, a2), (100, 200, 300));

        let word_b = pack_triple(PackingMethod::MethodB, 100, 200, 300);
        let (b0, b1, b2) = unpack_triple(PackingMethod::MethodB, word_b);
        assert_eq!((b0, b1, b2), (100, 200, 300));

        // Method A and B should produce different bit patterns
        assert_ne!(word_a, word_b);
    }

    #[test]
    fn test_endian_word_read_write() {
        let mut buf = Vec::new();
        write_word(&mut buf, 0xDEADBEEF, Endian::Big);
        let w = read_word(&buf, 0, Endian::Big).expect("read");
        assert_eq!(w, 0xDEADBEEF);

        let mut buf2 = Vec::new();
        write_word(&mut buf2, 0xDEADBEEF, Endian::Little);
        let w2 = read_word(&buf2, 0, Endian::Little).expect("read");
        assert_eq!(w2, 0xDEADBEEF);

        // Same value, different byte order
        assert_ne!(buf, buf2);
    }

    #[test]
    fn test_read_word_insufficient_data() {
        let data = [0u8; 3];
        assert!(read_word(&data, 0, Endian::Big).is_err());
    }

    #[test]
    fn test_scanline_round_trip() {
        let pixels: Vec<[u16; 3]> = (0..8)
            .map(|i| [(i * 128) as u16, (i * 64) as u16, (i * 32) as u16])
            .collect();
        let mut sl = PackedScanline::new(8, PackingMethod::MethodA);
        sl.encode_from_u16(&pixels).expect("encode");
        let decoded = sl.decode_to_u16();
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn test_scanline_get_set_pixel() {
        let mut sl = PackedScanline::new(4, PackingMethod::MethodB);
        sl.set_pixel(2, [500, 600, 700]).expect("set");
        let px = sl.get_pixel(2).expect("get");
        assert_eq!(px, [500, 600, 700]);
    }

    #[test]
    fn test_scanline_out_of_bounds() {
        let sl = PackedScanline::new(4, PackingMethod::MethodA);
        assert!(sl.get_pixel(4).is_err());
    }

    #[test]
    fn test_scanline_bytes_round_trip() {
        let mut sl = PackedScanline::new(3, PackingMethod::MethodA);
        sl.set_pixel(0, [100, 200, 300]).expect("set");
        sl.set_pixel(1, [400, 500, 600]).expect("set");
        sl.set_pixel(2, [700, 800, 900]).expect("set");

        let bytes = sl.to_bytes(Endian::Big);
        let sl2 = PackedScanline::from_bytes(&bytes, 0, 3, PackingMethod::MethodA, Endian::Big)
            .expect("from_bytes");
        assert_eq!(sl.words, sl2.words);
    }

    #[test]
    fn test_packed_image_round_trip() {
        let width = 4u32;
        let height = 3u32;
        let mut img = PackedImage::new(width, height, PackingMethod::MethodA, Endian::Big);
        img.set_pixel(1, 2, [1023, 512, 256]).expect("set");
        let px = img.get_pixel(1, 2).expect("get");
        assert_eq!(px, [1023, 512, 256]);

        let bytes = img.to_bytes();
        let img2 =
            PackedImage::from_bytes(&bytes, width, height, PackingMethod::MethodA, Endian::Big)
                .expect("from_bytes");
        assert_eq!(img.decode_all(), img2.decode_all());
    }

    #[test]
    fn test_packed_image_encode_decode_all() {
        let pixels: Vec<[u16; 3]> = (0..6)
            .map(|i| [(i * 170) as u16, (1023 - i * 170) as u16, 512])
            .collect();
        let mut img = PackedImage::new(3, 2, PackingMethod::MethodB, Endian::Little);
        img.encode_all(&pixels).expect("encode");
        let decoded = img.decode_all();
        assert_eq!(decoded, pixels);
    }

    #[test]
    fn test_f32_normalized_round_trip() {
        let f32_pixels: Vec<[f32; 3]> = vec![[0.0, 0.5, 1.0], [0.25, 0.75, 0.333]];
        let img = PackedImage::from_f32_normalized(
            &f32_pixels,
            2,
            1,
            PackingMethod::MethodA,
            Endian::Big,
        )
        .expect("from_f32");
        let back = img.to_f32_normalized();
        for (orig, conv) in f32_pixels.iter().zip(back.iter()) {
            for c in 0..3 {
                assert!(
                    (orig[c] - conv[c]).abs() < 0.002,
                    "channel {c}: {} vs {}",
                    orig[c],
                    conv[c]
                );
            }
        }
    }

    #[test]
    fn test_packed_image_out_of_bounds() {
        let img = PackedImage::new(4, 4, PackingMethod::MethodA, Endian::Big);
        assert!(img.get_pixel(0, 4).is_err());
        assert!(img.get_pixel(4, 0).is_err());
    }

    #[test]
    fn test_packed_image_from_bytes_too_short() {
        let data = vec![0u8; 10];
        assert!(PackedImage::from_bytes(&data, 4, 4, PackingMethod::MethodA, Endian::Big).is_err());
    }

    #[test]
    fn test_10bit_clamp() {
        // Values > 1023 should be masked to 10 bits
        let word = pack_method_a(1024, 2048, 4096);
        let (s0, s1, s2) = unpack_method_a(word);
        assert!(s0 <= 1023);
        assert!(s1 <= 1023);
        assert!(s2 <= 1023);
    }
}
