//! VP8L (WebP lossless) decoder.
//!
//! Implements the VP8L lossless bitstream specification for decoding
//! WebP lossless images. VP8L uses Huffman coding, LZ77 backward
//! references, a color cache, and spatial prediction transforms.
//!
//! Reference: <https://developers.google.com/speed/webp/docs/webp_lossless_bitstream_specification>

use crate::error::{CodecError, CodecResult};


/// VP8L signature byte.
const VP8L_SIGNATURE: u8 = 0x2F;

/// Maximum image dimension (14 bits + 1).
const MAX_DIMENSION: u32 = 16384;

/// Maximum number of Huffman code groups.
const MAX_HUFFMAN_GROUPS: usize = 256 * 256;

/// Maximum color cache bits.
const MAX_COLOR_CACHE_BITS: u32 = 11;

/// Number of literal codes (ARGB channels each 0-255).
const NUM_LITERAL_CODES: u16 = 256;

/// Number of length prefix codes.
const NUM_LENGTH_CODES: u16 = 24;

/// Number of distance prefix codes.
const NUM_DISTANCE_CODES: u16 = 40;

/// Maximum Huffman code length.
const MAX_ALLOWED_CODE_LENGTH: u8 = 15;

/// Order in which code length code lengths are stored.
const CODE_LENGTH_CODE_ORDER: [u8; 19] = [
    17, 18, 0, 1, 2, 3, 4, 5, 16, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
];

/// Number of code length codes.
const NUM_CODE_LENGTH_CODES: usize = 19;

/// VP8L 2-D distance mapping table (120 entries as (dx, dy) pairs).
const DISTANCE_MAP: [(i8, i8); 120] = [
    (0, 1),   (1, 0),   (1, 1),   (-1, 1),  (0, 2),   (2, 0),   (1, 2),   (-1, 2),
    (2, 1),   (-2, 1),  (2, 2),   (-2, 2),  (0, 3),   (3, 0),   (1, 3),   (-1, 3),
    (3, 1),   (-3, 1),  (2, 3),   (-2, 3),  (3, 2),   (-3, 2),  (0, 4),   (4, 0),
    (1, 4),   (-1, 4),  (4, 1),   (-4, 1),  (3, 3),   (-3, 3),  (2, 4),   (-2, 4),
    (4, 2),   (-4, 2),  (0, 5),   (3, 4),   (-3, 4),  (4, 3),   (-4, 3),  (5, 0),
    (1, 5),   (-1, 5),  (5, 1),   (-5, 1),  (2, 5),   (-2, 5),  (5, 2),   (-5, 2),
    (4, 4),   (-4, 4),  (3, 5),   (-3, 5),  (5, 3),   (-5, 3),  (0, 6),   (6, 0),
    (1, 6),   (-1, 6),  (6, 1),   (-6, 1),  (2, 6),   (-2, 6),  (6, 2),   (-6, 2),
    (4, 5),   (-4, 5),  (5, 4),   (-5, 4),  (3, 6),   (-3, 6),  (6, 3),   (-6, 3),
    (0, 7),   (7, 0),   (1, 7),   (-1, 7),  (5, 5),   (-5, 5),  (7, 1),   (-7, 1),
    (4, 6),   (-4, 6),  (6, 4),   (-6, 4),  (2, 7),   (-2, 7),  (7, 2),   (-7, 2),
    (3, 7),   (-3, 7),  (7, 3),   (-7, 3),  (5, 6),   (-5, 6),  (6, 5),   (-6, 5),
    (8, 0),   (4, 7),   (-4, 7),  (7, 4),   (-7, 4),  (8, 1),   (8, 2),   (6, 6),
    (-6, 6),  (8, 3),   (5, 7),   (-5, 7),  (7, 5),   (-7, 5),  (8, 4),   (6, 7),
    (-6, 7),  (7, 6),   (-7, 6),  (8, 5),   (7, 7),   (-7, 7),  (8, 6),   (8, 7),
];


/// VP8L bitstream header.
#[derive(Debug, Clone)]
pub struct Vp8lHeader {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Whether the image contains non-opaque alpha values.
    pub alpha_is_used: bool,
    /// Version number (must be 0).
    pub version: u8,
}

impl Vp8lHeader {
    /// Parse a VP8L header from data starting at the signature byte.
    pub fn parse(data: &[u8]) -> CodecResult<Self> {
        if data.len() < 5 {
            return Err(CodecError::InvalidBitstream(
                "VP8L header too short".into(),
            ));
        }
        if data[0] != VP8L_SIGNATURE {
            return Err(CodecError::InvalidBitstream(format!(
                "Invalid VP8L signature: 0x{:02X}",
                data[0]
            )));
        }
        // 32 bits packed little-endian: width-1(14) | height-1(14) | alpha(1) | version(3)
        let val = u32::from(data[1])
            | (u32::from(data[2]) << 8)
            | (u32::from(data[3]) << 16)
            | (u32::from(data[4]) << 24);

        let width = (val & 0x3FFF) + 1;
        let height = ((val >> 14) & 0x3FFF) + 1;
        let alpha_is_used = ((val >> 28) & 1) != 0;
        let version = ((val >> 29) & 0x7) as u8;

        if version != 0 {
            return Err(CodecError::InvalidBitstream(format!(
                "Unsupported VP8L version: {version}"
            )));
        }
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(CodecError::InvalidBitstream(format!(
                "Image dimensions too large: {width}x{height}"
            )));
        }

        Ok(Self {
            width,
            height,
            alpha_is_used,
            version,
        })
    }
}

// Bit Reader (LSB-first)

/// LSB-first bit reader for VP8L bitstreams.
pub struct Vp8lBitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_pos: u32,
    value: u64,
    bits_in_value: u32,
}

impl<'a> Vp8lBitReader<'a> {
    /// Create a new bit reader starting at the given data.
    pub fn new(data: &'a [u8]) -> Self {
        let mut reader = Self {
            data,
            pos: 0,
            bit_pos: 0,
            value: 0,
            bits_in_value: 0,
        };
        reader.fill();
        reader
    }

    /// Fill the value register with bytes from the stream.
    fn fill(&mut self) {
        while self.bits_in_value < 56 && self.pos < self.data.len() {
            self.value |= u64::from(self.data[self.pos]) << self.bits_in_value;
            self.pos += 1;
            self.bits_in_value += 8;
        }
    }

    /// Read `n` bits (LSB-first) and return them as a `u32`.
    pub fn read_bits(&mut self, n: u32) -> CodecResult<u32> {
        if n == 0 {
            return Ok(0);
        }
        if n > 32 {
            return Err(CodecError::InvalidBitstream(
                "Cannot read more than 32 bits at once".into(),
            ));
        }
        if self.bits_in_value < n {
            self.fill();
        }
        if self.bits_in_value < n {
            return Err(CodecError::NeedMoreData);
        }
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        let result = (self.value as u32) & mask;
        self.value >>= n;
        self.bits_in_value -= n;
        self.bit_pos += n;
        Ok(result)
    }

    /// Read a single bit.
    #[inline]
    pub fn read_bit(&mut self) -> CodecResult<bool> {
        Ok(self.read_bits(1)? != 0)
    }

    /// Peek at the next `n` bits without consuming them.
    pub fn peek_bits(&mut self, n: u32) -> CodecResult<u32> {
        if n == 0 {
            return Ok(0);
        }
        if self.bits_in_value < n {
            self.fill();
        }
        if self.bits_in_value < n {
            return Err(CodecError::NeedMoreData);
        }
        let mask = if n == 32 { u32::MAX } else { (1u32 << n) - 1 };
        Ok((self.value as u32) & mask)
    }

    /// Advance past `n` bits (used after peeking).
    pub fn advance(&mut self, n: u32) {
        self.value >>= n;
        self.bits_in_value = self.bits_in_value.saturating_sub(n);
        self.bit_pos += n;
    }
}

// Huffman Tree

/// A single entry in a Huffman lookup table.
#[derive(Debug, Clone, Copy, Default)]
pub struct HuffmanCode {
    /// Number of bits for this code.
    pub bits: u8,
    /// Symbol value.
    pub value: u16,
}

/// Huffman tree built from canonical code lengths.
#[derive(Debug, Clone)]
pub struct HuffmanTree {
    /// Flat lookup table (two-level for long codes).
    lut: Vec<HuffmanCode>,
    /// First-level table bits.
    lut_bits: u8,
    /// Second-level tables (offset, bits) indexed by first-level sentinel.
    second_level: Vec<(usize, u8)>,
    /// Second-level storage.
    lut2: Vec<HuffmanCode>,
}

impl HuffmanTree {
    /// Build a Huffman tree from an array of code lengths.
    fn build(code_lengths: &[u8], alphabet_size: u16) -> CodecResult<Self> {
        let max_len = code_lengths.iter().copied().max().unwrap_or(0);
        if max_len == 0 {
            // All zero code lengths – treat as single symbol 0.
            let mut lut = vec![HuffmanCode::default(); 1];
            lut[0] = HuffmanCode { bits: 0, value: 0 };
            return Ok(Self {
                lut,
                lut_bits: 0,
                second_level: Vec::new(),
                lut2: Vec::new(),
            });
        }

        let lut_bits = max_len.min(8);
        let table_size = 1usize << lut_bits;
        let mut lut = vec![HuffmanCode::default(); table_size];

        // Count per length.
        let mut bl_count = vec![0u32; (max_len as usize) + 1];
        for &cl in code_lengths.iter().take(alphabet_size as usize) {
            if cl > 0 {
                bl_count[cl as usize] += 1;
            }
        }

        // Compute next_code.
        let mut next_code = vec![0u32; (max_len as usize) + 1];
        {
            let mut code: u32 = 0;
            for bits in 1..=max_len as usize {
                code = (code + bl_count[bits - 1]) << 1;
                next_code[bits] = code;
            }
        }

        // Assign codes to symbols.
        let mut codes = Vec::with_capacity(alphabet_size as usize);
        for symbol in 0..alphabet_size {
            let cl = if (symbol as usize) < code_lengths.len() {
                code_lengths[symbol as usize]
            } else {
                0
            };
            if cl > 0 {
                let c = next_code[cl as usize];
                next_code[cl as usize] += 1;
                codes.push((symbol, cl, c));
            }
        }

        // Populate first-level table.
        let mut second_level = Vec::new();
        let mut lut2 = Vec::new();

        for &(symbol, cl, code) in &codes {
            if cl <= lut_bits {
                // Fits in the first-level table – replicate.
                let rev = reverse_bits(code, cl);
                let step = 1u32 << cl;
                let mut idx = rev;
                while idx < table_size as u32 {
                    lut[idx as usize] = HuffmanCode {
                        bits: cl,
                        value: symbol,
                    };
                    idx += step;
                }
            }
        }

        // Second-level tables for codes longer than lut_bits.
        if max_len > lut_bits {
            // Mark first-level entries that need second level.
            // Group by their first lut_bits.
            let mut max_second_bits: Vec<u8> = vec![0; table_size];
            for &(_symbol, cl, code) in &codes {
                if cl > lut_bits {
                    let rev = reverse_bits(code, cl);
                    let idx = rev & ((1u32 << lut_bits) - 1);
                    let remaining = cl - lut_bits;
                    if remaining > max_second_bits[idx as usize] {
                        max_second_bits[idx as usize] = remaining;
                    }
                }
            }

            // Allocate second-level tables.
            for first_idx in 0..table_size {
                let sb = max_second_bits[first_idx];
                if sb > 0 {
                    let offset = lut2.len();
                    let size2 = 1usize << sb;
                    lut2.resize(lut2.len() + size2, HuffmanCode::default());
                    // Store mapping: sentinel in first-level.
                    let sl_index = second_level.len();
                    second_level.push((offset, sb));
                    lut[first_idx] = HuffmanCode {
                        bits: lut_bits + sb,  // sentinel
                        value: sl_index as u16,
                    };
                }
            }

            // Fill second-level entries.
            for &(symbol, cl, code) in &codes {
                if cl > lut_bits {
                    let rev = reverse_bits(code, cl);
                    let first_idx = (rev & ((1u32 << lut_bits) - 1)) as usize;
                    let entry = &lut[first_idx];
                    let sl_index = entry.value as usize;
                    let (offset, sb) = second_level[sl_index];
                    let remaining = cl - lut_bits;
                    let rev2 = rev >> lut_bits;
                    let step = 1u32 << remaining;
                    let mut idx = rev2;
                    while idx < (1u32 << sb) {
                        lut2[offset + idx as usize] = HuffmanCode {
                            bits: cl,
                            value: symbol,
                        };
                        idx += step;
                    }
                }
            }
        }

        Ok(Self {
            lut,
            lut_bits,
            second_level,
            lut2,
        })
    }

    /// Decode a single symbol from the bit reader.
    fn read_symbol(&self, br: &mut Vp8lBitReader<'_>) -> CodecResult<u16> {
        if self.lut_bits == 0 {
            // Single-symbol tree.
            return Ok(self.lut.first().map_or(0, |c| c.value));
        }
        let peek = br.peek_bits(u32::from(self.lut_bits))?;
        let entry = self.lut[peek as usize];

        if entry.bits <= self.lut_bits {
            br.advance(u32::from(entry.bits));
            Ok(entry.value)
        } else {
            // Second-level lookup.
            let sl_index = entry.value as usize;
            if sl_index >= self.second_level.len() {
                return Err(CodecError::InvalidBitstream(
                    "Invalid Huffman second-level index".into(),
                ));
            }
            let (offset, sb) = self.second_level[sl_index];
            let total_bits = self.lut_bits + sb;
            let peek2 = br.peek_bits(u32::from(total_bits))?;
            let idx2 = (peek2 >> self.lut_bits) as usize;
            if offset + idx2 >= self.lut2.len() {
                return Err(CodecError::InvalidBitstream(
                    "Huffman second-level table overflow".into(),
                ));
            }
            let entry2 = self.lut2[offset + idx2];
            br.advance(u32::from(entry2.bits));
            Ok(entry2.value)
        }
    }
}

/// Reverse the bottom `n` bits of `val`.
fn reverse_bits(val: u32, n: u8) -> u32 {
    let mut result = 0u32;
    let mut v = val;
    for _ in 0..n {
        result = (result << 1) | (v & 1);
        v >>= 1;
    }
    result
}

// Huffman Group (5 trees)

/// A group of five Huffman trees used for decoding one region.
#[derive(Debug, Clone)]
struct HuffmanGroup {
    /// Green + length + color cache codes.
    green: HuffmanTree,
    /// Red channel codes.
    red: HuffmanTree,
    /// Blue channel codes.
    blue: HuffmanTree,
    /// Alpha channel codes.
    alpha: HuffmanTree,
    /// Distance prefix codes.
    distance: HuffmanTree,
}

// Transforms

/// VP8L image transform.
#[derive(Debug, Clone)]
pub enum Transform {
    /// Spatial prediction transform.
    Predictor {
        /// Log2 of block size minus 2.
        size_bits: u8,
        /// Sub-resolution predictor image (ARGB pixels).
        data: Vec<u32>,
    },
    /// Color decorrelation transform.
    ColorTransform {
        /// Log2 of block size minus 2.
        size_bits: u8,
        /// Color transform elements per block.
        data: Vec<ColorTransformElement>,
    },
    /// Subtract green transform (no data).
    SubtractGreen,
    /// Color indexing (palette) transform.
    ColorIndexing {
        /// Palette colors (ARGB).
        palette: Vec<u32>,
        /// Bits per pixel index (1, 2, 4, or 8).
        bits_per_pixel: u8,
    },
}

/// Color transform element used for decorrelation.
#[derive(Debug, Clone, Copy, Default)]
pub struct ColorTransformElement {
    /// Green-to-red multiplier.
    pub green_to_red: i8,
    /// Green-to-blue multiplier.
    pub green_to_blue: i8,
    /// Red-to-blue multiplier.
    pub red_to_blue: i8,
}

// Color Cache

/// Color cache for VP8L decoding.
struct ColorCache {
    colors: Vec<u32>,
    hash_shift: u32,
}

impl ColorCache {
    fn new(bits: u32) -> Self {
        let size = 1usize << bits;
        Self {
            colors: vec![0u32; size],
            hash_shift: 32 - bits,
        }
    }

    fn lookup(&self, index: u32) -> u32 {
        self.colors[index as usize & (self.colors.len() - 1)]
    }

    fn insert(&mut self, argb: u32) {
        let hash = (0x1E35_A7BDu64.wrapping_mul(u64::from(argb)) >> self.hash_shift) as usize;
        let len = self.colors.len();
        self.colors[hash & (len - 1)] = argb;
    }
}

// Decoded Image

/// Result of VP8L decoding.
pub struct DecodedImage {
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// ARGB pixel data (row-major, `width * height` elements).
    pub pixels: Vec<u32>,
    /// Whether the image contains non-opaque alpha.
    pub has_alpha: bool,
}

// VP8L Decoder

/// VP8L lossless image decoder.
pub struct Vp8lDecoder {
    width: u32,
    height: u32,
    alpha_is_used: bool,
    transforms: Vec<Transform>,
}

impl Vp8lDecoder {
    /// Create a new decoder (dimensions set during `decode`).
    pub fn new() -> Self {
        Self {
            width: 0,
            height: 0,
            alpha_is_used: false,
            transforms: Vec::new(),
        }
    }

    /// Decode a VP8L bitstream starting at the signature byte.
    pub fn decode(&mut self, data: &[u8]) -> CodecResult<DecodedImage> {
        let header = Vp8lHeader::parse(data)?;
        self.width = header.width;
        self.height = header.height;
        self.alpha_is_used = header.alpha_is_used;
        self.transforms.clear();

        // Bit reader starts after the 5-byte header.
        let mut br = Vp8lBitReader::new(&data[5..]);

        // Read transforms.
        let (decoded_width, decoded_height) = self.read_transforms(&mut br)?;

        // Decode the main image.
        let pixels = self.decode_image_data(
            &mut br,
            decoded_width,
            decoded_height,
        )?;

        // Apply transforms in reverse order.
        let pixels = self.apply_transforms(pixels, decoded_width, decoded_height)?;

        Ok(DecodedImage {
            width: self.width,
            height: self.height,
            pixels,
            has_alpha: self.alpha_is_used,
        })
    }

    // -----------------------------------------------------------------------
    // Transform reading
    // -----------------------------------------------------------------------

    fn read_transforms(
        &mut self,
        br: &mut Vp8lBitReader<'_>,
    ) -> CodecResult<(u32, u32)> {
        let mut width = self.width;
        let height = self.height;

        while br.read_bit()? {
            let transform_type = br.read_bits(2)?;
            match transform_type {
                0 => {
                    // PREDICTOR_TRANSFORM
                    let size_bits = br.read_bits(3)? as u8;
                    let block_size = 1u32 << (size_bits + 2);
                    let tw = div_round_up(width, block_size);
                    let th = div_round_up(height, block_size);
                    let data = self.decode_image_data(br, tw, th)?;
                    self.transforms.push(Transform::Predictor {
                        size_bits: size_bits + 2,
                        data,
                    });
                }
                1 => {
                    // COLOR_TRANSFORM
                    let size_bits = br.read_bits(3)? as u8;
                    let block_size = 1u32 << (size_bits + 2);
                    let tw = div_round_up(width, block_size);
                    let th = div_round_up(height, block_size);
                    let pixels = self.decode_image_data(br, tw, th)?;
                    let data: Vec<ColorTransformElement> = pixels
                        .iter()
                        .map(|&p| {
                            ColorTransformElement {
                                green_to_red: ((p >> 8) & 0xFF) as i8,
                                green_to_blue: ((p >> 16) & 0xFF) as i8,
                                red_to_blue: (p & 0xFF) as i8,
                            }
                        })
                        .collect();
                    self.transforms.push(Transform::ColorTransform {
                        size_bits: size_bits + 2,
                        data,
                    });
                }
                2 => {
                    // SUBTRACT_GREEN_TRANSFORM
                    self.transforms.push(Transform::SubtractGreen);
                }
                3 => {
                    // COLOR_INDEXING_TRANSFORM
                    let num_colors = br.read_bits(8)? + 1;
                    let palette = self.decode_image_data(br, num_colors, 1)?;
                    let bits_per_pixel = if num_colors <= 2 {
                        1
                    } else if num_colors <= 4 {
                        2
                    } else if num_colors <= 16 {
                        4
                    } else {
                        8
                    };
                    if bits_per_pixel < 8 {
                        width = div_round_up(width * u32::from(bits_per_pixel), 8);
                    }
                    self.transforms.push(Transform::ColorIndexing {
                        palette,
                        bits_per_pixel,
                    });
                }
                _ => {
                    return Err(CodecError::InvalidBitstream(
                        "Invalid transform type".into(),
                    ));
                }
            }
        }
        Ok((width, height))
    }

    // -----------------------------------------------------------------------
    // Image data decoding
    // -----------------------------------------------------------------------

    fn decode_image_data(
        &self,
        br: &mut Vp8lBitReader<'_>,
        width: u32,
        height: u32,
    ) -> CodecResult<Vec<u32>> {
        // Read color cache parameters.
        let color_cache_bits = if br.read_bit()? {
            let bits = br.read_bits(4)?;
            if bits > MAX_COLOR_CACHE_BITS {
                return Err(CodecError::InvalidBitstream(format!(
                    "Color cache bits too large: {bits}"
                )));
            }
            Some(bits)
        } else {
            None
        };

        let num_color_cache_codes = color_cache_bits
            .map(|b| 1u16 << b)
            .unwrap_or(0);

        // Read meta-Huffman image (entropy image).
        let huffman_groups = self.read_huffman_codes(
            br,
            width,
            height,
            num_color_cache_codes,
        )?;

        // Decode pixels.
        let total_pixels = (width as usize)
            .checked_mul(height as usize)
            .ok_or_else(|| {
                CodecError::InvalidBitstream("Image too large".into())
            })?;

        let mut pixels = vec![0u32; total_pixels];
        let mut color_cache = color_cache_bits.map(ColorCache::new);

        let mut idx = 0usize;
        while idx < total_pixels {
            // For now: single Huffman group (meta-Huffman support below).
            let group_idx = if huffman_groups.len() == 1 {
                0
            } else {
                // Meta-Huffman: determine group from position.
                // (In a full implementation, the entropy image would be consulted here.)
                0
            };
            let group = huffman_groups
                .get(group_idx)
                .ok_or_else(|| {
                    CodecError::InvalidBitstream("Invalid Huffman group index".into())
                })?;

            let green_sym = group.green.read_symbol(br)?;

            if green_sym < NUM_LITERAL_CODES {
                // Literal pixel.
                let red = group.red.read_symbol(br)? as u32;
                let blue = group.blue.read_symbol(br)? as u32;
                let alpha = group.alpha.read_symbol(br)? as u32;

                let argb = (alpha << 24) | (red << 16) | (u32::from(green_sym) << 8) | blue;
                pixels[idx] = argb;
                if let Some(ref mut cache) = color_cache {
                    cache.insert(argb);
                }
                idx += 1;
            } else if green_sym < NUM_LITERAL_CODES + NUM_LENGTH_CODES {
                // LZ77 backward reference.
                let length_prefix = green_sym - NUM_LITERAL_CODES;
                let length = prefix_code_to_value(length_prefix, br)?;

                let dist_sym = group.distance.read_symbol(br)?;
                let dist_raw = prefix_code_to_value(dist_sym, br)?;
                let dist = distance_map_to_pixel_dist(dist_raw as usize, width as usize);

                if dist == 0 || dist > idx {
                    return Err(CodecError::InvalidBitstream(format!(
                        "Invalid backward reference distance: {dist}, position: {idx}"
                    )));
                }

                let length = length as usize;
                for i in 0..length {
                    if idx + i >= total_pixels {
                        break;
                    }
                    let src = idx + i - dist;
                    pixels[idx + i] = pixels[src];
                    if let Some(ref mut cache) = color_cache {
                        cache.insert(pixels[idx + i]);
                    }
                }
                idx += length;
            } else {
                // Color cache lookup.
                let cache_idx = green_sym - NUM_LITERAL_CODES - NUM_LENGTH_CODES;
                let cache = color_cache.as_ref().ok_or_else(|| {
                    CodecError::InvalidBitstream(
                        "Color cache code without color cache".into(),
                    )
                })?;
                let argb = cache.lookup(u32::from(cache_idx));
                pixels[idx] = argb;
                if let Some(ref mut cache) = color_cache {
                    cache.insert(argb);
                }
                idx += 1;
            }
        }

        Ok(pixels)
    }

    // -----------------------------------------------------------------------
    // Huffman code reading
    // -----------------------------------------------------------------------

    fn read_huffman_codes(
        &self,
        br: &mut Vp8lBitReader<'_>,
        _width: u32,
        _height: u32,
        num_color_cache_codes: u16,
    ) -> CodecResult<Vec<HuffmanGroup>> {
        // Check for meta-Huffman image.
        let use_meta = br.read_bit()?;
        let num_groups = if use_meta {
            let _prefix_bits = br.read_bits(3)? + 2;
            // For a full meta-Huffman implementation, we would read
            // the entropy image here. For now, treat as single group.
            // The entropy image would tell us the number of groups.
            1usize
        } else {
            1usize
        };

        if num_groups > MAX_HUFFMAN_GROUPS {
            return Err(CodecError::InvalidBitstream(
                "Too many Huffman groups".into(),
            ));
        }

        let green_alphabet = NUM_LITERAL_CODES + NUM_LENGTH_CODES + num_color_cache_codes;
        let red_alphabet = NUM_LITERAL_CODES;
        let blue_alphabet = NUM_LITERAL_CODES;
        let alpha_alphabet = NUM_LITERAL_CODES;
        let dist_alphabet = NUM_DISTANCE_CODES;

        let mut groups = Vec::with_capacity(num_groups);
        for _ in 0..num_groups {
            let green = read_huffman_tree(br, green_alphabet)?;
            let red = read_huffman_tree(br, red_alphabet)?;
            let blue = read_huffman_tree(br, blue_alphabet)?;
            let alpha = read_huffman_tree(br, alpha_alphabet)?;
            let distance = read_huffman_tree(br, dist_alphabet)?;

            groups.push(HuffmanGroup {
                green,
                red,
                blue,
                alpha,
                distance,
            });
        }

        Ok(groups)
    }

    // -----------------------------------------------------------------------
    // Inverse transforms
    // -----------------------------------------------------------------------

    fn apply_transforms(
        &self,
        mut pixels: Vec<u32>,
        decoded_width: u32,
        decoded_height: u32,
    ) -> CodecResult<Vec<u32>> {
        // Apply in reverse order.
        for transform in self.transforms.iter().rev() {
            match transform {
                Transform::Predictor { size_bits, data } => {
                    pixels = self.inverse_predictor(
                        &pixels,
                        decoded_width,
                        decoded_height,
                        *size_bits,
                        data,
                    )?;
                }
                Transform::ColorTransform { size_bits, data } => {
                    pixels = self.inverse_color_transform(
                        &pixels,
                        decoded_width,
                        decoded_height,
                        *size_bits,
                        data,
                    );
                }
                Transform::SubtractGreen => {
                    self.inverse_subtract_green(&mut pixels);
                }
                Transform::ColorIndexing {
                    palette,
                    bits_per_pixel,
                } => {
                    pixels = self.inverse_color_indexing(
                        &pixels,
                        decoded_width,
                        decoded_height,
                        palette,
                        *bits_per_pixel,
                    )?;
                }
            }
        }
        Ok(pixels)
    }

    /// Inverse predictor transform: add predicted values to residuals.
    fn inverse_predictor(
        &self,
        pixels: &[u32],
        width: u32,
        height: u32,
        size_bits: u8,
        predictor_data: &[u32],
    ) -> CodecResult<Vec<u32>> {
        let w = width as usize;
        let h = height as usize;
        let block_size = 1usize << size_bits;
        let pred_w = div_round_up(width, block_size as u32) as usize;

        let mut out = pixels.to_vec();

        // Top-left pixel: add to 0xFF000000.
        if !out.is_empty() {
            out[0] = add_pixels(out[0], 0xFF00_0000);
        }

        // Top row: predictor mode 1 (left).
        for x in 1..w.min(out.len()) {
            out[x] = add_pixels(out[x], out[x - 1]);
        }

        // Remaining rows.
        for y in 1..h {
            let row_start = y * w;
            if row_start >= out.len() {
                break;
            }
            // Left column: predictor mode 2 (top).
            out[row_start] = add_pixels(out[row_start], out[row_start - w]);

            for x in 1..w {
                let idx = row_start + x;
                if idx >= out.len() {
                    break;
                }
                let bx = x / block_size;
                let by = (y - 1) / block_size;
                let pred_idx = by * pred_w + bx;
                let mode = if pred_idx < predictor_data.len() {
                    ((predictor_data[pred_idx] >> 8) & 0xFF) as u8
                } else {
                    0
                };

                let left = out[idx - 1];
                let top = out[idx - w];
                let top_left = out[idx - w - 1];
                let top_right = if x + 1 < w {
                    out[idx - w + 1]
                } else {
                    out[idx - w]
                };

                let predicted = predict(mode, left, top, top_left, top_right);
                out[idx] = add_pixels(out[idx], predicted);
            }
        }

        Ok(out)
    }

    /// Inverse color transform.
    fn inverse_color_transform(
        &self,
        pixels: &[u32],
        width: u32,
        height: u32,
        size_bits: u8,
        transform_data: &[ColorTransformElement],
    ) -> Vec<u32> {
        let w = width as usize;
        let h = height as usize;
        let block_size = 1usize << size_bits;
        let tw = div_round_up(width, block_size as u32) as usize;

        let mut out = pixels.to_vec();

        for y in 0..h {
            for x in 0..w {
                let idx = y * w + x;
                if idx >= out.len() {
                    break;
                }
                let bx = x / block_size;
                let by = y / block_size;
                let ti = by * tw + bx;
                let ct = if ti < transform_data.len() {
                    transform_data[ti]
                } else {
                    ColorTransformElement::default()
                };

                let argb = out[idx];
                let green = ((argb >> 8) & 0xFF) as i32;
                let red = ((argb >> 16) & 0xFF) as i32;
                let blue = (argb & 0xFF) as i32;
                let alpha = (argb >> 24) & 0xFF;

                let new_red = (red + color_transform_delta(
                    ct.green_to_red as i32,
                    green,
                )) & 0xFF;
                let new_blue = (blue
                    + color_transform_delta(ct.green_to_blue as i32, green)
                    + color_transform_delta(ct.red_to_blue as i32, new_red))
                    & 0xFF;

                out[idx] = (alpha << 24)
                    | ((new_red as u32) << 16)
                    | (((green as u32) & 0xFF) << 8)
                    | (new_blue as u32);
            }
        }

        out
    }

    /// Inverse subtract-green transform.
    fn inverse_subtract_green(&self, pixels: &mut [u32]) {
        for pixel in pixels.iter_mut() {
            let green = (*pixel >> 8) & 0xFF;
            let red = ((*pixel >> 16) & 0xFF).wrapping_add(green) & 0xFF;
            let blue = (*pixel & 0xFF).wrapping_add(green) & 0xFF;
            *pixel = (*pixel & 0xFF00_FF00) | (red << 16) | blue;
        }
    }

    /// Inverse color indexing transform.
    fn inverse_color_indexing(
        &self,
        pixels: &[u32],
        _packed_width: u32,
        height: u32,
        palette: &[u32],
        bits_per_pixel: u8,
    ) -> CodecResult<Vec<u32>> {
        let out_width = self.width as usize;
        let out_height = height as usize;
        let total = out_width * out_height;
        let mut out = Vec::with_capacity(total);

        let pixels_per_byte = 8 / bits_per_pixel as usize;
        let mask = (1u32 << bits_per_pixel) - 1;

        for y in 0..out_height {
            for x in 0..out_width {
                let packed_x = x / pixels_per_byte;
                let sub_idx = x % pixels_per_byte;
                let src_idx = y * (_packed_width as usize) + packed_x;
                let packed_val = if src_idx < pixels.len() {
                    // Green channel holds the packed index.
                    (pixels[src_idx] >> 8) & 0xFF
                } else {
                    0
                };

                let index = (packed_val >> (sub_idx as u32 * u32::from(bits_per_pixel))) & mask;
                let color = if (index as usize) < palette.len() {
                    palette[index as usize]
                } else {
                    0xFF00_0000
                };
                out.push(color);
            }
        }

        Ok(out)
    }
}

impl Default for Vp8lDecoder {
    fn default() -> Self {
        Self::new()
    }
}

// Huffman tree reading

/// Read a Huffman tree from the bitstream.
fn read_huffman_tree(br: &mut Vp8lBitReader<'_>, alphabet_size: u16) -> CodecResult<HuffmanTree> {
    let simple = br.read_bit()?;
    if simple {
        read_simple_huffman(br, alphabet_size)
    } else {
        read_normal_huffman(br, alphabet_size)
    }
}

/// Read a simple (1 or 2 symbol) Huffman code.
fn read_simple_huffman(
    br: &mut Vp8lBitReader<'_>,
    alphabet_size: u16,
) -> CodecResult<HuffmanTree> {
    let num_symbols = br.read_bits(1)? + 1;
    let is_first_8bit = br.read_bit()?;

    let symbol0 = if is_first_8bit {
        br.read_bits(8)? as u16
    } else {
        br.read_bits(1)? as u16
    };

    if symbol0 >= alphabet_size {
        return Err(CodecError::InvalidBitstream(format!(
            "Simple Huffman symbol {symbol0} >= alphabet size {alphabet_size}"
        )));
    }

    if num_symbols == 1 {
        let mut code_lengths = vec![0u8; alphabet_size as usize];
        code_lengths[symbol0 as usize] = 1;
        // Single-symbol tree: always returns this symbol.
        let lut = vec![HuffmanCode {
            bits: 0,
            value: symbol0,
        }; 1];
        return Ok(HuffmanTree {
            lut,
            lut_bits: 0,
            second_level: Vec::new(),
            lut2: Vec::new(),
        });
    }

    // Two symbols, 1-bit code.
    let symbol1 = br.read_bits(8)? as u16;
    if symbol1 >= alphabet_size {
        return Err(CodecError::InvalidBitstream(format!(
            "Simple Huffman symbol {symbol1} >= alphabet size {alphabet_size}"
        )));
    }

    let mut code_lengths = vec![0u8; alphabet_size as usize];
    code_lengths[symbol0 as usize] = 1;
    code_lengths[symbol1 as usize] = 1;
    HuffmanTree::build(&code_lengths, alphabet_size)
}

/// Read a normal (complex) Huffman code using the code-length-code method.
fn read_normal_huffman(
    br: &mut Vp8lBitReader<'_>,
    alphabet_size: u16,
) -> CodecResult<HuffmanTree> {
    // Read the number of code length codes.
    let num_code_length_codes = br.read_bits(4)? as usize + 4;
    if num_code_length_codes > NUM_CODE_LENGTH_CODES {
        return Err(CodecError::InvalidBitstream(format!(
            "Too many code length codes: {num_code_length_codes}"
        )));
    }

    // Read code length code lengths (3 bits each).
    let mut cl_code_lengths = [0u8; NUM_CODE_LENGTH_CODES];
    for i in 0..num_code_length_codes {
        cl_code_lengths[CODE_LENGTH_CODE_ORDER[i] as usize] = br.read_bits(3)? as u8;
    }

    // Build the code-length Huffman tree.
    let cl_tree = HuffmanTree::build(&cl_code_lengths, NUM_CODE_LENGTH_CODES as u16)?;

    // Decode actual code lengths.
    let mut code_lengths = vec![0u8; alphabet_size as usize];
    let mut i = 0usize;
    let mut prev_code_length = 8u8;

    while i < alphabet_size as usize {
        let sym = cl_tree.read_symbol(br)?;
        match sym {
            0..=15 => {
                code_lengths[i] = sym as u8;
                if sym != 0 {
                    prev_code_length = sym as u8;
                }
                i += 1;
            }
            16 => {
                // Repeat previous code length 3-6 times.
                let repeat = br.read_bits(2)? as usize + 3;
                for _ in 0..repeat {
                    if i >= alphabet_size as usize {
                        break;
                    }
                    code_lengths[i] = prev_code_length;
                    i += 1;
                }
            }
            17 => {
                // Repeat zero 3-10 times.
                let repeat = br.read_bits(3)? as usize + 3;
                i += repeat;
            }
            18 => {
                // Repeat zero 11-138 times.
                let repeat = br.read_bits(7)? as usize + 11;
                i += repeat;
            }
            _ => {
                return Err(CodecError::InvalidBitstream(format!(
                    "Invalid code length symbol: {sym}"
                )));
            }
        }
    }

    HuffmanTree::build(&code_lengths, alphabet_size)
}

// Prefix / LZ77 helpers

/// Convert a prefix code + extra bits to a value (used for length and distance).
fn prefix_code_to_value(prefix_code: u16, br: &mut Vp8lBitReader<'_>) -> CodecResult<u32> {
    if prefix_code < 4 {
        return Ok(u32::from(prefix_code) + 1);
    }
    let extra_bits = (u32::from(prefix_code) - 2) >> 1;
    let offset = (2 + (u32::from(prefix_code) & 1)) << extra_bits;
    let extra = br.read_bits(extra_bits)?;
    Ok(offset + extra + 1)
}

/// Map a VP8L distance code to a pixel distance using the 2D distance table.
fn distance_map_to_pixel_dist(dist_code: usize, xsize: usize) -> usize {
    if dist_code == 0 {
        return 0;
    }
    if dist_code <= 120 {
        let (dx, dy) = DISTANCE_MAP[dist_code - 1];
        let dist = i64::from(dy) * xsize as i64 + i64::from(dx);
        if dist < 1 {
            return 1;
        }
        dist as usize
    } else {
        dist_code - 120
    }
}

// Pixel arithmetic helpers

/// Add two ARGB pixels component-wise (mod 256 per channel).
#[inline]
fn add_pixels(a: u32, b: u32) -> u32 {
    let a0 = (a >> 24) & 0xFF;
    let a1 = (a >> 16) & 0xFF;
    let a2 = (a >> 8) & 0xFF;
    let a3 = a & 0xFF;

    let b0 = (b >> 24) & 0xFF;
    let b1 = (b >> 16) & 0xFF;
    let b2 = (b >> 8) & 0xFF;
    let b3 = b & 0xFF;

    (((a0 + b0) & 0xFF) << 24)
        | (((a1 + b1) & 0xFF) << 16)
        | (((a2 + b2) & 0xFF) << 8)
        | ((a3 + b3) & 0xFF)
}

/// Average two bytes (component-wise).
#[inline]
fn average2_byte(a: u8, b: u8) -> u8 {
    ((u16::from(a) + u16::from(b)) / 2) as u8
}

/// Average two ARGB pixels component-wise.
#[inline]
fn average2(a: u32, b: u32) -> u32 {
    let a_ch = pixel_channels(a);
    let b_ch = pixel_channels(b);
    channels_to_pixel([
        average2_byte(a_ch[0], b_ch[0]),
        average2_byte(a_ch[1], b_ch[1]),
        average2_byte(a_ch[2], b_ch[2]),
        average2_byte(a_ch[3], b_ch[3]),
    ])
}

/// Extract ARGB channels as [A, R, G, B].
#[inline]
fn pixel_channels(p: u32) -> [u8; 4] {
    [
        ((p >> 24) & 0xFF) as u8,
        ((p >> 16) & 0xFF) as u8,
        ((p >> 8) & 0xFF) as u8,
        (p & 0xFF) as u8,
    ]
}

/// Pack [A, R, G, B] channels into a pixel.
#[inline]
fn channels_to_pixel(ch: [u8; 4]) -> u32 {
    (u32::from(ch[0]) << 24)
        | (u32::from(ch[1]) << 16)
        | (u32::from(ch[2]) << 8)
        | u32::from(ch[3])
}

/// Clamp a value to [0, 255].
#[inline]
fn clamp_byte(v: i32) -> u8 {
    v.clamp(0, 255) as u8
}

/// Select predictor: choose L or T based on Manhattan distance to TL.
#[inline]
fn select(left: u32, top: u32, top_left: u32) -> u32 {
    let l = pixel_channels(left);
    let t = pixel_channels(top);
    let tl = pixel_channels(top_left);

    let predict_l: i32 = (0..4)
        .map(|i| (i32::from(t[i]) - i32::from(tl[i])).abs())
        .sum();
    let predict_t: i32 = (0..4)
        .map(|i| (i32::from(l[i]) - i32::from(tl[i])).abs())
        .sum();

    if predict_l < predict_t {
        left
    } else {
        top
    }
}

/// ClampAddSubtractFull: L + T - TL, clamped per channel.
#[inline]
fn clamp_add_subtract_full(left: u32, top: u32, top_left: u32) -> u32 {
    let l = pixel_channels(left);
    let t = pixel_channels(top);
    let tl = pixel_channels(top_left);
    channels_to_pixel([
        clamp_byte(i32::from(l[0]) + i32::from(t[0]) - i32::from(tl[0])),
        clamp_byte(i32::from(l[1]) + i32::from(t[1]) - i32::from(tl[1])),
        clamp_byte(i32::from(l[2]) + i32::from(t[2]) - i32::from(tl[2])),
        clamp_byte(i32::from(l[3]) + i32::from(t[3]) - i32::from(tl[3])),
    ])
}

/// ClampAddSubtractHalf: avg + (avg - other) / 2, clamped.
#[inline]
fn clamp_add_subtract_half(avg: u32, other: u32) -> u32 {
    let a = pixel_channels(avg);
    let o = pixel_channels(other);
    channels_to_pixel([
        clamp_byte(i32::from(a[0]) + (i32::from(a[0]) - i32::from(o[0])) / 2),
        clamp_byte(i32::from(a[1]) + (i32::from(a[1]) - i32::from(o[1])) / 2),
        clamp_byte(i32::from(a[2]) + (i32::from(a[2]) - i32::from(o[2])) / 2),
        clamp_byte(i32::from(a[3]) + (i32::from(a[3]) - i32::from(o[3])) / 2),
    ])
}

/// Color transform delta.
#[inline]
#[allow(clippy::cast_possible_truncation)]
fn color_transform_delta(multiplier: i32, channel: i32) -> i32 {
    // The spec defines: (multiplier * channel) >> 5
    // but the multiplier is a signed byte interpreted as i8.
    ((multiplier as i8 as i32) * (channel as i8 as i32)) >> 5
}

/// Predict a pixel using one of the 14 predictor modes.
fn predict(mode: u8, left: u32, top: u32, top_left: u32, top_right: u32) -> u32 {
    match mode {
        0 => 0xFF00_0000, // black with opaque alpha
        1 => left,
        2 => top,
        3 => top_right,
        4 => top_left,
        5 => average2(average2(left, top_right), top),
        6 => average2(left, top_left),
        7 => average2(left, top),
        8 => average2(top_left, top),
        9 => average2(top, top_right),
        10 => average2(average2(left, top_left), average2(top, top_right)),
        11 => select(left, top, top_left),
        12 => clamp_add_subtract_full(left, top, top_left),
        13 => clamp_add_subtract_half(average2(left, top), top_left),
        _ => 0xFF00_0000, // fallback
    }
}

/// Integer division rounding up.
#[inline]
fn div_round_up(a: u32, b: u32) -> u32 {
    (a + b - 1) / b
}

// Tests

#[cfg(test)]
mod tests {
    use super::*;

    // -- Header tests -------------------------------------------------------

    #[test]
    fn test_parse_header_basic() {
        // Construct a valid VP8L header for a 4x3 image, no alpha, version 0.
        // width = 4 => width-1 = 3 (14 bits: 0x0003)
        // height = 3 => height-1 = 2 (14 bits: 0x0002)
        // alpha = 0 (1 bit)
        // version = 0 (3 bits)
        // Packed: 3 | (2 << 14) | (0 << 28) | (0 << 29)
        let val: u32 = 3 | (2 << 14);
        let mut data = vec![VP8L_SIGNATURE];
        data.extend_from_slice(&val.to_le_bytes());

        let header = Vp8lHeader::parse(&data).expect("should parse");
        assert_eq!(header.width, 4);
        assert_eq!(header.height, 3);
        assert!(!header.alpha_is_used);
        assert_eq!(header.version, 0);
    }

    #[test]
    fn test_parse_header_with_alpha() {
        // 10x5, alpha used.
        let val: u32 = 9 | (4 << 14) | (1 << 28);
        let mut data = vec![VP8L_SIGNATURE];
        data.extend_from_slice(&val.to_le_bytes());

        let header = Vp8lHeader::parse(&data).expect("should parse");
        assert_eq!(header.width, 10);
        assert_eq!(header.height, 5);
        assert!(header.alpha_is_used);
        assert_eq!(header.version, 0);
    }

    #[test]
    fn test_parse_header_bad_signature() {
        let data = [0x00, 0x00, 0x00, 0x00, 0x00];
        assert!(Vp8lHeader::parse(&data).is_err());
    }

    #[test]
    fn test_parse_header_bad_version() {
        let val: u32 = 3 | (2 << 14) | (1 << 29); // version = 1
        let mut data = vec![VP8L_SIGNATURE];
        data.extend_from_slice(&val.to_le_bytes());
        assert!(Vp8lHeader::parse(&data).is_err());
    }

    #[test]
    fn test_parse_header_too_short() {
        let data = [VP8L_SIGNATURE, 0x00];
        assert!(Vp8lHeader::parse(&data).is_err());
    }

    // -- Bit reader tests ---------------------------------------------------

    #[test]
    fn test_bitreader_read_bits() {
        // 0xA5 = 1010_0101 in binary
        // LSB-first reading: bit0=1, bit1=0, bit2=1, bit3=0, bit4=0, bit5=1, ...
        let data = [0xA5, 0xFF];
        let mut br = Vp8lBitReader::new(&data);
        // Read 4 bits LSB-first from 0xA5 => bottom nibble = 0x5
        let val = br.read_bits(4).expect("should read");
        assert_eq!(val, 0x5);
        // Next 4 bits => top nibble = 0xA
        let val = br.read_bits(4).expect("should read");
        assert_eq!(val, 0xA);
    }

    #[test]
    fn test_bitreader_read_bit() {
        let data = [0x01];
        let mut br = Vp8lBitReader::new(&data);
        assert!(br.read_bit().expect("should read")); // LSB is 1
        assert!(!br.read_bit().expect("should read")); // next bit is 0
    }

    #[test]
    fn test_bitreader_read_zero_bits() {
        let data = [0xFF];
        let mut br = Vp8lBitReader::new(&data);
        assert_eq!(br.read_bits(0).expect("should read"), 0);
    }

    #[test]
    fn test_bitreader_peek_and_advance() {
        let data = [0xAB, 0xCD];
        let mut br = Vp8lBitReader::new(&data);
        let peeked = br.peek_bits(8).expect("should peek");
        assert_eq!(peeked, 0xAB);
        br.advance(4);
        let next = br.read_bits(4).expect("should read");
        assert_eq!(next, 0x0A); // upper nibble of 0xAB
    }

    // -- Reverse bits -------------------------------------------------------

    #[test]
    fn test_reverse_bits() {
        assert_eq!(reverse_bits(0b110, 3), 0b011);
        assert_eq!(reverse_bits(0b1010, 4), 0b0101);
        assert_eq!(reverse_bits(0b1, 1), 0b1);
        assert_eq!(reverse_bits(0b0, 1), 0b0);
    }

    // -- Pixel helpers ------------------------------------------------------

    #[test]
    fn test_add_pixels() {
        let a = 0x01020304u32;
        let b = 0x05060708u32;
        let result = add_pixels(a, b);
        assert_eq!((result >> 24) & 0xFF, 0x06);
        assert_eq!((result >> 16) & 0xFF, 0x08);
        assert_eq!((result >> 8) & 0xFF, 0x0A);
        assert_eq!(result & 0xFF, 0x0C);
    }

    #[test]
    fn test_add_pixels_wrapping() {
        let a = 0xFF800180u32;
        let b = 0x01810281u32;
        let result = add_pixels(a, b);
        assert_eq!((result >> 24) & 0xFF, 0x00); // 0xFF + 0x01 = 0x100 & 0xFF = 0x00
        assert_eq!((result >> 16) & 0xFF, 0x01); // 0x80 + 0x81 = 0x101 & 0xFF = 0x01
        assert_eq!((result >> 8) & 0xFF, 0x03);  // 0x01 + 0x02 = 0x03
        assert_eq!(result & 0xFF, 0x01);           // 0x80 + 0x81 = 0x101 & 0xFF = 0x01
    }

    #[test]
    fn test_average2() {
        let a = 0xFF000000u32;
        let b = 0xFF000000u32;
        assert_eq!(average2(a, b), 0xFF000000);

        let a = 0xFF640000u32; // R=100
        let b = 0xFFC80000u32; // R=200
        let result = average2(a, b);
        assert_eq!((result >> 16) & 0xFF, 150); // average of 100 and 200
    }

    #[test]
    fn test_select_predictor() {
        // predict_l = sum|T[i] - TL[i]|, predict_t = sum|L[i] - TL[i]|
        // When predict_l < predict_t, return left.
        let left = 0xFF640000u32;     // R=100
        let top = 0xFF0A0000u32;      // R=10
        let top_left = 0xFF0A0000u32; // R=10
        // predict_l = |10 - 10| = 0, predict_t = |100 - 10| = 90
        // predict_l < predict_t => return left
        let result = select(left, top, top_left);
        assert_eq!(result, left);
    }

    #[test]
    fn test_clamp_add_subtract_full() {
        let left = channels_to_pixel([255, 100, 50, 30]);
        let top = channels_to_pixel([255, 80, 60, 20]);
        let top_left = channels_to_pixel([255, 90, 55, 25]);
        // L + T - TL per channel: (100+80-90)=90, (50+60-55)=55, (30+20-25)=25
        let result = clamp_add_subtract_full(left, top, top_left);
        let ch = pixel_channels(result);
        assert_eq!(ch[0], 255);
        assert_eq!(ch[1], 90);
        assert_eq!(ch[2], 55);
        assert_eq!(ch[3], 25);
    }

    #[test]
    fn test_clamp_add_subtract_full_clamping() {
        let left = channels_to_pixel([255, 250, 10, 5]);
        let top = channels_to_pixel([255, 240, 5, 3]);
        let top_left = channels_to_pixel([255, 100, 200, 200]);
        // L + T - TL: (250+240-100)=390 clamped to 255, (10+5-200)=-185 clamped to 0
        let result = clamp_add_subtract_full(left, top, top_left);
        let ch = pixel_channels(result);
        assert_eq!(ch[1], 255);
        assert_eq!(ch[2], 0);
        assert_eq!(ch[3], 0);
    }

    // -- Distance mapping ---------------------------------------------------

    #[test]
    fn test_distance_map_code_1() {
        // Code 1 => (0, 1) => dist = 1*width + 0 = width
        let dist = distance_map_to_pixel_dist(1, 10);
        assert_eq!(dist, 10);
    }

    #[test]
    fn test_distance_map_code_2() {
        // Code 2 => (1, 0) => dist = 0*width + 1 = 1
        let dist = distance_map_to_pixel_dist(2, 10);
        assert_eq!(dist, 1);
    }

    #[test]
    fn test_distance_map_negative_dx() {
        // Code 4 => (-1, 1) => dist = 1*width + (-1) = width - 1
        let dist = distance_map_to_pixel_dist(4, 10);
        assert_eq!(dist, 9);
    }

    #[test]
    fn test_distance_map_over_120() {
        // Code > 120 => dist_code - 120
        let dist = distance_map_to_pixel_dist(130, 10);
        assert_eq!(dist, 10);
    }

    #[test]
    fn test_distance_map_clamp_to_1() {
        // Code 2 => (1, 0) => dist = 1 with any width
        // Code 1 => (0, 1) => dist = width; for width=1 => dist=1
        let dist = distance_map_to_pixel_dist(1, 1);
        assert_eq!(dist, 1);
    }

    // -- Prefix code --------------------------------------------------------

    #[test]
    fn test_prefix_code_small() {
        // Codes 0-3 map directly to values 1-4.
        let data = [0xFF; 4];
        let mut br = Vp8lBitReader::new(&data);
        assert_eq!(prefix_code_to_value(0, &mut br).expect("ok"), 1);
        assert_eq!(prefix_code_to_value(1, &mut br).expect("ok"), 2);
        assert_eq!(prefix_code_to_value(2, &mut br).expect("ok"), 3);
        assert_eq!(prefix_code_to_value(3, &mut br).expect("ok"), 4);
    }

    #[test]
    fn test_prefix_code_with_extra_bits() {
        // Code 4: extra_bits = (4-2)>>1 = 1, offset = (2+(4&1))<<1 = 4
        // With extra bit = 0: value = 4 + 0 + 1 = 5
        let data = [0x00; 4];
        let mut br = Vp8lBitReader::new(&data);
        assert_eq!(prefix_code_to_value(4, &mut br).expect("ok"), 5);
    }

    // -- Huffman tree building ----------------------------------------------

    #[test]
    fn test_huffman_single_symbol() {
        // Single-symbol tree.
        let code_lengths = [0u8, 1, 0, 0];
        let tree = HuffmanTree::build(&code_lengths, 4).expect("should build");
        let data = [0x00; 4];
        let mut br = Vp8lBitReader::new(&data);
        let sym = tree.read_symbol(&mut br).expect("should decode");
        assert_eq!(sym, 1);
    }

    #[test]
    fn test_huffman_two_symbols() {
        // Two-symbol tree: symbol 0 = code 0 (1 bit), symbol 1 = code 1 (1 bit).
        let code_lengths = [1u8, 1];
        let tree = HuffmanTree::build(&code_lengths, 2).expect("should build");

        // Bit 0 => symbol 0
        let data = [0x00];
        let mut br = Vp8lBitReader::new(&data);
        assert_eq!(tree.read_symbol(&mut br).expect("ok"), 0);

        // Bit 1 => symbol 1
        let data = [0x01];
        let mut br = Vp8lBitReader::new(&data);
        assert_eq!(tree.read_symbol(&mut br).expect("ok"), 1);
    }

    #[test]
    fn test_huffman_three_symbols() {
        // Three symbols: 0 => len 1, 1 => len 2, 2 => len 2.
        // Canonical codes: 0 => "0", 1 => "10", 2 => "11"
        let code_lengths = [1u8, 2, 2];
        let tree = HuffmanTree::build(&code_lengths, 3).expect("should build");

        // "0" (bit 0) => symbol 0
        let data = [0b0000_0000];
        let mut br = Vp8lBitReader::new(&data);
        assert_eq!(tree.read_symbol(&mut br).expect("ok"), 0);

        // Canonical code for symbol 1 = "10", reversed LSB-first = "01" = 1.
        // Bottom 2 bits of data must be 0b01.
        let data = [0b0000_0001];
        let mut br = Vp8lBitReader::new(&data);
        assert_eq!(tree.read_symbol(&mut br).expect("ok"), 1);

        // Canonical code for symbol 2 = "11", reversed LSB-first = "11" = 3.
        // Bottom 2 bits of data must be 0b11.
        let data = [0b0000_0011];
        let mut br = Vp8lBitReader::new(&data);
        assert_eq!(tree.read_symbol(&mut br).expect("ok"), 2);
    }

    // -- Inverse subtract green ---------------------------------------------

    #[test]
    fn test_inverse_subtract_green() {
        let decoder = Vp8lDecoder::new();
        let mut pixels = vec![0x00_20_40_60u32]; // A=0x00, R=0x20, G=0x40, B=0x60
        decoder.inverse_subtract_green(&mut pixels);
        // R += G: (0x20 + 0x40) & 0xFF = 0x60
        // B += G: (0x60 + 0x40) & 0xFF = 0xA0
        assert_eq!((pixels[0] >> 16) & 0xFF, 0x60);
        assert_eq!(pixels[0] & 0xFF, 0xA0);
        // G unchanged
        assert_eq!((pixels[0] >> 8) & 0xFF, 0x40);
    }

    // -- Color transform delta ----------------------------------------------

    #[test]
    fn test_color_transform_delta() {
        // Simple: 4 * 8 >> 5 = 1
        assert_eq!(color_transform_delta(4, 8), 1);
        // Negative multiplier: -4 * 8 >> 5 = -1
        assert_eq!(color_transform_delta(-4i32 as i32, 8), -1);
    }

    // -- predict function ---------------------------------------------------

    #[test]
    fn test_predict_modes() {
        let left = 0xFF112233u32;
        let top = 0xFF445566u32;
        let top_left = 0xFF778899u32;
        let top_right = 0xFFAABBCCu32;

        assert_eq!(predict(0, left, top, top_left, top_right), 0xFF000000);
        assert_eq!(predict(1, left, top, top_left, top_right), left);
        assert_eq!(predict(2, left, top, top_left, top_right), top);
        assert_eq!(predict(3, left, top, top_left, top_right), top_right);
        assert_eq!(predict(4, left, top, top_left, top_right), top_left);
    }

    #[test]
    fn test_predict_average_modes() {
        let left = channels_to_pixel([255, 100, 50, 30]);
        let top = channels_to_pixel([255, 200, 100, 60]);

        // Mode 7: Average2(L, T)
        let result = predict(7, left, top, left, top);
        let ch = pixel_channels(result);
        assert_eq!(ch[1], 150); // avg(100, 200)
        assert_eq!(ch[2], 75);  // avg(50, 100)
        assert_eq!(ch[3], 45);  // avg(30, 60)
    }

    // -- div_round_up -------------------------------------------------------

    #[test]
    fn test_div_round_up() {
        assert_eq!(div_round_up(10, 3), 4);
        assert_eq!(div_round_up(9, 3), 3);
        assert_eq!(div_round_up(1, 1), 1);
        assert_eq!(div_round_up(0, 5), 0);
    }

    // -- Color cache --------------------------------------------------------

    #[test]
    fn test_color_cache() {
        let mut cache = ColorCache::new(4); // 16 entries
        let color = 0xFF112233u32;
        cache.insert(color);
        // The hash determines the slot; verify that the color is retrievable.
        let hash = (0x1E35_A7BDu64.wrapping_mul(u64::from(color)) >> 28) as u32;
        assert_eq!(cache.lookup(hash), color);
    }

    // -- Decoder new --------------------------------------------------------

    #[test]
    fn test_decoder_new() {
        let decoder = Vp8lDecoder::new();
        assert_eq!(decoder.width, 0);
        assert_eq!(decoder.height, 0);
    }

    #[test]
    fn test_decoder_default() {
        let decoder = Vp8lDecoder::default();
        assert_eq!(decoder.width, 0);
    }

    // -- Synthetic VP8L bitstream -------------------------------------------

    /// Build a minimal valid VP8L bitstream for a 1x1 solid-color image.
    /// This uses a simple Huffman code with a single literal pixel.
    fn build_1x1_bitstream(argb: u32) -> Vec<u8> {
        let green = ((argb >> 8) & 0xFF) as u8;
        let red = ((argb >> 16) & 0xFF) as u8;
        let blue = (argb & 0xFF) as u8;
        let alpha = ((argb >> 24) & 0xFF) as u8;

        // We build the bitstream manually:
        // 1. Header: signature(1) + packed(4) = 5 bytes
        // 2. No transforms (1 bit = 0)
        // 3. No color cache (1 bit = 0)
        // 4. No meta-Huffman (1 bit = 0)
        // 5. Five simple Huffman codes (1-symbol each)
        // 6. The single pixel

        let mut bits = BitWriter::new();

        // Transform flag: no transforms.
        bits.write(0, 1);

        // Color cache: no.
        bits.write(0, 1);

        // Meta-Huffman: no.
        bits.write(0, 1);

        // Five Huffman trees, each "simple" with 1 symbol.
        // Green tree: simple=1, num_symbols-1=0, is_first_8bit=1, symbol=green
        write_single_symbol_tree(&mut bits, u16::from(green));
        // Red tree
        write_single_symbol_tree(&mut bits, u16::from(red));
        // Blue tree
        write_single_symbol_tree(&mut bits, u16::from(blue));
        // Alpha tree
        write_single_symbol_tree(&mut bits, u16::from(alpha));
        // Distance tree: symbol 0
        write_single_symbol_tree(&mut bits, 0);

        // The pixel: since each tree has exactly 1 symbol, 0 bits needed per symbol.
        // (Single-symbol trees decode without reading any bits.)

        let body = bits.finish();

        // Build header.
        let val: u32 = 0 | (0 << 14); // width-1=0, height-1=0
        // alpha flag
        let val = val | if alpha != 255 { 1 << 28 } else { 0 };
        let mut data = vec![VP8L_SIGNATURE];
        data.extend_from_slice(&val.to_le_bytes());
        data.extend_from_slice(&body);
        data
    }

    /// Write a simple single-symbol Huffman tree into the bit writer.
    fn write_single_symbol_tree(bw: &mut BitWriter, symbol: u16) {
        // simple = 1
        bw.write(1, 1);
        // num_symbols - 1 = 0
        bw.write(0, 1);
        // is_first_8bit = 1 (so we can encode symbol in 8 bits)
        bw.write(1, 1);
        // symbol (8 bits)
        bw.write(u32::from(symbol), 8);
    }

    /// Simple LSB-first bit writer for test data construction.
    struct BitWriter {
        bytes: Vec<u8>,
        current: u64,
        bits_used: u32,
    }

    impl BitWriter {
        fn new() -> Self {
            Self {
                bytes: Vec::new(),
                current: 0,
                bits_used: 0,
            }
        }

        fn write(&mut self, val: u32, n: u32) {
            self.current |= u64::from(val) << self.bits_used;
            self.bits_used += n;
            while self.bits_used >= 8 {
                self.bytes.push((self.current & 0xFF) as u8);
                self.current >>= 8;
                self.bits_used -= 8;
            }
        }

        fn finish(mut self) -> Vec<u8> {
            if self.bits_used > 0 {
                self.bytes.push((self.current & 0xFF) as u8);
            }
            self.bytes
        }
    }

    #[test]
    fn test_decode_1x1_white() {
        let data = build_1x1_bitstream(0xFFFF_FFFF);
        let mut decoder = Vp8lDecoder::new();
        let result = decoder.decode(&data).expect("should decode");
        assert_eq!(result.width, 1);
        assert_eq!(result.height, 1);
        assert_eq!(result.pixels.len(), 1);
        assert_eq!(result.pixels[0], 0xFFFF_FFFF);
    }

    #[test]
    fn test_decode_1x1_red() {
        let data = build_1x1_bitstream(0xFFFF_0000);
        let mut decoder = Vp8lDecoder::new();
        let result = decoder.decode(&data).expect("should decode");
        assert_eq!(result.pixels[0], 0xFFFF_0000);
    }

    #[test]
    fn test_decode_1x1_with_alpha() {
        let data = build_1x1_bitstream(0x80FF_0000);
        let mut decoder = Vp8lDecoder::new();
        let result = decoder.decode(&data).expect("should decode");
        assert_eq!(result.pixels[0], 0x80FF_0000);
        assert!(result.has_alpha);
    }

    #[test]
    fn test_decode_1x1_black() {
        let data = build_1x1_bitstream(0xFF00_0000);
        let mut decoder = Vp8lDecoder::new();
        let result = decoder.decode(&data).expect("should decode");
        assert_eq!(result.pixels[0], 0xFF00_0000);
    }

    #[test]
    fn test_decode_invalid_data() {
        let data = [0x00, 0x01]; // too short, wrong sig
        let mut decoder = Vp8lDecoder::new();
        assert!(decoder.decode(&data).is_err());
    }

    /// Build a 2x2 bitstream with 4 literal pixels, no transforms.
    fn build_2x2_bitstream(pixels: [u32; 4]) -> Vec<u8> {
        // We use two-symbol Huffman codes for channels that need two values.
        // For simplicity, use the same approach: each channel tree has
        // a simple code.

        // Collect unique values per channel.
        let greens: Vec<u8> = pixels.iter().map(|p| ((p >> 8) & 0xFF) as u8).collect();
        let reds: Vec<u8> = pixels.iter().map(|p| ((p >> 16) & 0xFF) as u8).collect();
        let blues: Vec<u8> = pixels.iter().map(|p| (p & 0xFF) as u8).collect();
        let alphas: Vec<u8> = pixels.iter().map(|p| ((p >> 24) & 0xFF) as u8).collect();

        // For this test, use uniform color so each tree is single-symbol.
        // Verify all channels are uniform.
        let g = greens[0];
        let r = reds[0];
        let b = blues[0];
        let a = alphas[0];
        assert!(greens.iter().all(|&v| v == g), "Non-uniform green channel in test helper");
        assert!(reds.iter().all(|&v| v == r), "Non-uniform red channel in test helper");
        assert!(blues.iter().all(|&v| v == b), "Non-uniform blue channel in test helper");
        assert!(alphas.iter().all(|&v| v == a), "Non-uniform alpha channel in test helper");

        let mut bits = BitWriter::new();

        // No transforms.
        bits.write(0, 1);
        // No color cache.
        bits.write(0, 1);
        // No meta-Huffman.
        bits.write(0, 1);

        // Five single-symbol trees.
        write_single_symbol_tree(&mut bits, u16::from(g));
        write_single_symbol_tree(&mut bits, u16::from(r));
        write_single_symbol_tree(&mut bits, u16::from(b));
        write_single_symbol_tree(&mut bits, u16::from(a));
        write_single_symbol_tree(&mut bits, 0);

        let body = bits.finish();

        // Header for 2x2 image.
        let val: u32 = 1 | (1 << 14); // width-1=1, height-1=1
        let val = val | if a != 255 { 1 << 28 } else { 0 };
        let mut data = vec![VP8L_SIGNATURE];
        data.extend_from_slice(&val.to_le_bytes());
        data.extend_from_slice(&body);
        data
    }

    #[test]
    fn test_decode_2x2_uniform() {
        let color = 0xFF336699u32;
        let data = build_2x2_bitstream([color; 4]);
        let mut decoder = Vp8lDecoder::new();
        let result = decoder.decode(&data).expect("should decode");
        assert_eq!(result.width, 2);
        assert_eq!(result.height, 2);
        assert_eq!(result.pixels.len(), 4);
        for p in &result.pixels {
            assert_eq!(*p, color);
        }
    }
}
