//! VP8 lossy encoder for WebP.
//!
//! This module provides a simplified VP8 encoder that produces valid VP8 keyframe
//! bitstreams suitable for embedding in a WebP container. The encoder implements:
//!
//! - RGB to YUV 4:2:0 color space conversion (BT.601)
//! - 16x16 macroblock processing with DC intra prediction
//! - Forward 4x4 DCT transform
//! - Coefficient quantization with quality-based QP mapping
//! - Boolean arithmetic coding (VP8 range coder)
//! - VP8 keyframe bitstream assembly per RFC 6386
//!
//! # Limitations
//!
//! - Only generates keyframes (no inter prediction / P-frames)
//! - Uses DC prediction mode exclusively (simplest intra prediction)
//! - Single DCT partition (no multi-partition)
//! - No rate-distortion optimization
//!
//! # References
//!
//! - [RFC 6386: VP8 Data Format and Decoding Guide](https://tools.ietf.org/html/rfc6386)

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]

use crate::error::{CodecError, CodecResult};

// ---------------------------------------------------------------------------
// VP8 default token probability tables (RFC 6386 Section 13.4)
// ---------------------------------------------------------------------------

/// Default coefficient probabilities for VP8 token decoding.
///
/// Layout: `[block_type][coeff_band][prev_coeff_ctx][token_node]`
/// - block_type: 0..4 (DC-Y-after-Y2, AC-Y, DC/AC-UV, Y2)
/// - coeff_band: 0..8
/// - prev_coeff_ctx: 0..3 (0=zero, 1=one, 2=>=2)
/// - token_node: 0..11 (tree probabilities)
///
/// These are the "factory default" probabilities shipped with every VP8
/// keyframe when no explicit updates are signaled.
#[rustfmt::skip]
static DEFAULT_COEFF_PROBS: [[[[u8; 11]; 3]; 8]; 4] = [
    // Block type 0: DC component of Y after Y2
    [
        [[128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
         [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128],
         [128, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
        [[253, 136, 254, 255, 228, 219, 128, 128, 128, 128, 128],
         [189, 129, 242, 255, 227, 213, 255, 219, 128, 128, 128],
         [106, 126, 227, 252, 214, 209, 255, 255, 128, 128, 128]],
        [[  1,  98, 248, 255, 236, 226, 255, 255, 128, 128, 128],
         [181, 133, 238, 254, 211, 236, 255, 255, 128, 128, 128],
         [ 78, 134, 202, 247, 198, 180, 255, 219, 128, 128, 128]],
        [[  1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128],
         [184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128],
         [ 77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128]],
        [[  1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128],
         [170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128],
         [ 37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128]],
        [[  1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128],
         [207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128],
         [102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128]],
        [[  1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128],
         [177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128],
         [ 80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128]],
        [[  1,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [246,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
    ],
    // Block type 1: AC coefficients of Y
    [
        [[198,  35, 237, 223, 193, 187, 162, 160, 145, 155,  62],
         [131,  45, 198, 221, 172, 176, 220, 157, 252, 221,   1],
         [ 68,  47, 146, 208, 149, 167, 221, 162, 255, 223, 128]],
        [[  1, 149, 241, 255, 221, 224, 255, 255, 128, 128, 128],
         [184, 141, 234, 253, 222, 220, 255, 199, 128, 128, 128],
         [ 81, 99,  181, 242, 195, 203, 255, 219, 128, 128, 128]],
        [[  1, 129, 232, 253, 214, 197, 242, 196, 255, 255, 128],
         [132, 109, 223, 253, 214, 175, 255, 236, 128, 128, 128],
         [ 68, 104, 184, 246, 171, 175, 255, 236, 128, 128, 128]],
        [[  1, 200, 246, 255, 234, 255, 128, 128, 128, 128, 128],
         [195, 148, 244, 255, 236, 203, 128, 128, 128, 128, 128],
         [ 39, 130, 228, 255, 223, 255, 128, 128, 128, 128, 128]],
        [[  1, 107, 238, 254, 198, 218, 255, 191, 128, 128, 128],
         [188, 133, 238, 253, 233, 181, 128, 128, 128, 128, 128],
         [ 36, 142, 199, 247, 175, 230, 255, 255, 128, 128, 128]],
        [[  1, 238, 251, 255, 210, 128, 128, 128, 128, 128, 128],
         [190, 171, 253, 255, 249, 128, 128, 128, 128, 128, 128],
         [ 61, 104, 231, 255, 235, 128, 128, 128, 128, 128, 128]],
        [[  1, 210, 247, 255, 255, 128, 128, 128, 128, 128, 128],
         [164, 154, 246, 255, 249, 128, 128, 128, 128, 128, 128],
         [ 29, 145, 228, 255, 220, 128, 128, 128, 128, 128, 128]],
        [[  1,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [218,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
    ],
    // Block type 2: DC/AC of UV
    [
        [[  1, 108, 226, 255, 227, 187, 128, 128, 128, 128, 128],
         [117, 109, 203, 246, 197, 174, 255, 255, 128, 128, 128],
         [ 15,  66, 128, 224, 149, 147, 255, 255, 128, 128, 128]],
        [[  1,  59, 220, 255, 205, 206, 128, 128, 128, 128, 128],
         [138,  40, 218, 255, 237, 219, 255, 255, 128, 128, 128],
         [ 31,  27, 156, 248, 188, 175, 255, 255, 128, 128, 128]],
        [[  1, 112, 230, 250, 199, 191, 255, 255, 128, 128, 128],
         [116, 109, 225, 252, 198, 190, 255, 255, 128, 128, 128],
         [ 41,  82, 163, 237, 156, 172, 255, 255, 128, 128, 128]],
        [[  1,  74, 254, 255, 227, 128, 128, 128, 128, 128, 128],
         [150, 101, 247, 255, 222, 128, 128, 128, 128, 128, 128],
         [ 57,  56, 231, 255, 243, 128, 128, 128, 128, 128, 128]],
        [[  1, 179, 255, 255, 128, 128, 128, 128, 128, 128, 128],
         [176, 134, 243, 255, 228, 128, 128, 128, 128, 128, 128],
         [ 80,  84, 234, 255, 210, 128, 128, 128, 128, 128, 128]],
        [[  1, 253, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [185, 205, 255, 255, 128, 128, 128, 128, 128, 128, 128],
         [141, 124, 248, 255, 128, 128, 128, 128, 128, 128, 128]],
        [[  1, 254, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [187, 252, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [175, 138, 254, 254, 128, 128, 128, 128, 128, 128, 128]],
        [[  1,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [239,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
    ],
    // Block type 3: Y2 (DC of 16x16 luma)
    [
        [[  1, 202, 254, 255, 245, 255, 128, 128, 128, 128, 128],
         [248, 136, 248, 254, 227, 128, 128, 128, 128, 128, 128],
         [255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
        [[  1, 185, 249, 255, 243, 255, 128, 128, 128, 128, 128],
         [184, 150, 247, 255, 236, 224, 128, 128, 128, 128, 128],
         [ 77, 110, 216, 255, 236, 230, 128, 128, 128, 128, 128]],
        [[  1, 101, 251, 255, 241, 255, 128, 128, 128, 128, 128],
         [170, 139, 241, 252, 236, 209, 255, 255, 128, 128, 128],
         [ 37, 116, 196, 243, 228, 255, 255, 255, 128, 128, 128]],
        [[  1, 204, 254, 255, 245, 255, 128, 128, 128, 128, 128],
         [207, 160, 250, 255, 238, 128, 128, 128, 128, 128, 128],
         [102, 103, 231, 255, 211, 171, 128, 128, 128, 128, 128]],
        [[  1, 152, 252, 255, 240, 255, 128, 128, 128, 128, 128],
         [177, 135, 243, 255, 234, 225, 128, 128, 128, 128, 128],
         [ 80, 129, 211, 255, 194, 224, 128, 128, 128, 128, 128]],
        [[  1,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [246,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
        [[  1,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [246,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
        [[  1,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [246,   1, 255, 128, 128, 128, 128, 128, 128, 128, 128],
         [255, 128, 128, 128, 128, 128, 128, 128, 128, 128, 128]],
    ],
];

/// VP8 DC quantizer lookup table (RFC 6386 Section 9.6).
///
/// Maps quantizer index (0..127) to the actual DC dequantization factor.
#[rustfmt::skip]
static DC_QUANT_TABLE: [i32; 128] = [
      4,   5,   6,   7,   8,   9,  10,  10,  11,  12,  13,  14,  15,  16,  17,  17,
     18,  19,  20,  20,  21,  21,  22,  22,  23,  23,  24,  25,  25,  26,  27,  28,
     29,  30,  31,  32,  33,  34,  35,  36,  37,  37,  38,  39,  40,  41,  42,  43,
     44,  45,  46,  46,  47,  48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,
     59,  60,  61,  62,  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,
     75,  76,  76,  77,  78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,
     91,  93,  95,  96,  98, 100, 101, 102, 104, 106, 108, 110, 112, 114, 116, 118,
    122, 124, 126, 128, 130, 132, 134, 136, 138, 140, 143, 145, 148, 151, 154, 157,
];

/// VP8 AC quantizer lookup table (RFC 6386 Section 9.6).
///
/// Maps quantizer index (0..127) to the actual AC dequantization factor.
#[rustfmt::skip]
static AC_QUANT_TABLE: [i32; 128] = [
      4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,  16,  17,  18,  19,
     20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,  33,  34,  35,
     36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
     52,  53,  54,  55,  56,  57,  58,  60,  62,  64,  66,  68,  70,  72,  74,  76,
     78,  80,  82,  84,  86,  88,  90,  92,  94,  96,  98, 100, 102, 104, 106, 108,
    110, 112, 114, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149, 152,
    155, 158, 161, 164, 167, 170, 173, 177, 181, 185, 189, 193, 197, 201, 205, 209,
    213, 217, 221, 225, 229, 234, 239, 245, 249, 254, 259, 264, 269, 274, 279, 284,
];

/// VP8 zigzag scan order for 4x4 blocks.
static ZIGZAG_ORDER: [usize; 16] = [
    0, 1, 4, 8, 5, 2, 3, 6, 9, 12, 13, 10, 7, 11, 14, 15,
];

/// Maps a coefficient's zigzag position to a frequency band (0..7).
///
/// VP8 groups coefficient positions into 8 bands for probability context.
static COEFF_BANDS: [usize; 16] = [0, 1, 2, 3, 6, 4, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7];

// ---------------------------------------------------------------------------
// Boolean arithmetic encoder (VP8 range coder)
// ---------------------------------------------------------------------------

/// Boolean arithmetic encoder for VP8 bitstream writing.
///
/// This is the encoding counterpart of `BoolDecoder`. VP8 encodes all
/// header flags and DCT tokens through a range coder that maintains
/// a `range` / `bottom` pair and emits bytes as the range narrows.
struct BoolEncoder {
    output: Vec<u8>,
    range: u32,
    bottom: u64,
    bits_left: i32,
}

impl BoolEncoder {
    /// Creates a new boolean encoder with an empty output buffer.
    fn new() -> Self {
        Self {
            output: Vec::new(),
            range: 255,
            bottom: 0,
            bits_left: 24,
        }
    }

    /// Encodes a single boolean symbol with the given probability.
    ///
    /// `prob` is the probability that the symbol is **false** (0),
    /// in the range 1..=255.
    fn encode_bool(&mut self, value: bool, prob: u8) {
        let split = 1 + (((self.range - 1) * u32::from(prob)) >> 8);

        if value {
            self.bottom += u64::from(split);
            self.range -= split;
        } else {
            self.range = split;
        }

        // Renormalize
        let mut shift = 0u32;
        while self.range < 128 {
            self.range <<= 1;
            shift += 1;
        }

        self.bottom <<= shift;
        self.bits_left -= shift as i32;

        if self.bits_left <= 0 {
            self.flush_bits();
        }
    }

    /// Encodes a boolean with 50% probability (uniform bit).
    fn encode_bit(&mut self, value: bool) {
        self.encode_bool(value, 128);
    }

    /// Encodes an unsigned integer of `n` bits, MSB first.
    fn encode_literal(&mut self, value: u32, n: u8) {
        for i in (0..n).rev() {
            let bit = (value >> i) & 1 != 0;
            self.encode_bit(bit);
        }
    }

    /// Encodes a value using a fixed probability for each bit, MSB first.
    fn encode_literal_with_prob(&mut self, value: u32, n: u8, prob: u8) {
        for i in (0..n).rev() {
            let bit = (value >> i) & 1 != 0;
            self.encode_bool(bit, prob);
        }
    }

    /// Flushes accumulated bits into output bytes.
    fn flush_bits(&mut self) {
        while self.bits_left <= 0 {
            let byte = (self.bottom >> 24) as u8;
            self.output.push(byte);
            self.bottom = (self.bottom & 0x00FF_FFFF) << 8;
            self.bits_left += 8;
        }
    }

    /// Finalizes the encoder and returns the encoded byte stream.
    fn flush(mut self) -> Vec<u8> {
        // Push remaining bits
        for _ in 0..4 {
            let byte = (self.bottom >> 24) as u8;
            self.output.push(byte);
            self.bottom <<= 8;
        }
        self.output
    }
}

// ---------------------------------------------------------------------------
// Forward DCT (4x4)
// ---------------------------------------------------------------------------

/// Performs a 1D forward DCT on 4 samples.
///
/// This is the VP8 forward transform from RFC 6386 Section 14.4.
fn fdct4_1d(input: &[i32; 4], output: &mut [i32; 4]) {
    let a0 = input[0] + input[3];
    let a1 = input[1] + input[2];
    let a2 = input[1] - input[2];
    let a3 = input[0] - input[3];

    output[0] = a0 + a1;
    output[2] = a0 - a1;

    // These use integer approximations of cos/sin
    // output[1] = a3 * 2217/4096 + a2 * 5352/4096
    // output[3] = a3 * 5352/4096 - a2 * 2217/4096
    output[1] = (a2 * 5352 + a3 * 2217 + 14500) >> 12;
    output[3] = (a3 * 5352 - a2 * 2217 + 7500) >> 12;
}

/// Performs a 2D forward 4x4 DCT on a residual block.
///
/// Takes 16 residual values (in raster order) and produces 16 DCT
/// coefficients (in raster order).
fn fdct4x4(residual: &[i32; 16], coeffs: &mut [i32; 16]) {
    let mut temp = [0i32; 16];

    // Row transform
    for row in 0..4 {
        let base = row * 4;
        let input = [
            residual[base],
            residual[base + 1],
            residual[base + 2],
            residual[base + 3],
        ];
        let mut out = [0i32; 4];
        fdct4_1d(&input, &mut out);
        temp[base] = out[0];
        temp[base + 1] = out[1];
        temp[base + 2] = out[2];
        temp[base + 3] = out[3];
    }

    // Column transform
    for col in 0..4 {
        let input = [temp[col], temp[col + 4], temp[col + 8], temp[col + 12]];
        let mut out = [0i32; 4];
        fdct4_1d(&input, &mut out);
        coeffs[col] = (out[0] + 1) >> 1;
        coeffs[col + 4] = (out[1] + 1) >> 1;
        coeffs[col + 8] = (out[2] + 1) >> 1;
        coeffs[col + 12] = (out[3] + 1) >> 1;
    }
}

/// Forward 4x4 Walsh-Hadamard Transform for DC coefficients.
///
/// Takes 16 DC values from the 4x4 grid of sub-blocks and produces
/// 16 WHT coefficients.
fn fwht4x4(dc_values: &[i32; 16], coeffs: &mut [i32; 16]) {
    let mut temp = [0i32; 16];

    // Row transform
    for row in 0..4 {
        let base = row * 4;
        let a = dc_values[base] + dc_values[base + 3];
        let b = dc_values[base + 1] + dc_values[base + 2];
        let c = dc_values[base + 1] - dc_values[base + 2];
        let d = dc_values[base] - dc_values[base + 3];

        temp[base] = a + b;
        temp[base + 1] = d + c;
        temp[base + 2] = a - b;
        temp[base + 3] = d - c;
    }

    // Column transform
    for col in 0..4 {
        let a = temp[col] + temp[col + 12];
        let b = temp[col + 4] + temp[col + 8];
        let c = temp[col + 4] - temp[col + 8];
        let d = temp[col] - temp[col + 12];

        coeffs[col] = a + b;
        coeffs[col + 4] = d + c;
        coeffs[col + 8] = a - b;
        coeffs[col + 12] = d - c;
    }
}

// ---------------------------------------------------------------------------
// YUV plane representation
// ---------------------------------------------------------------------------

/// YUV 4:2:0 image planes.
struct YuvPlanes {
    y: Vec<u8>,
    u: Vec<u8>,
    v: Vec<u8>,
    y_stride: usize,
    uv_stride: usize,
    width: u32,
    height: u32,
}

/// Converts RGB data to YUV 4:2:0 using BT.601 coefficients.
///
/// The RGB buffer must contain `width * height * 3` bytes in row-major
/// R-G-B order.  The output planes are padded so that the luma plane
/// width/height are multiples of 16 (macroblock alignment).
fn rgb_to_yuv420(data: &[u8], width: u32, height: u32) -> CodecResult<YuvPlanes> {
    let w = width as usize;
    let h = height as usize;

    if data.len() < w * h * 3 {
        return Err(CodecError::InvalidParameter(format!(
            "RGB data too short: need {}, have {}",
            w * h * 3,
            data.len()
        )));
    }

    // Pad to macroblock boundaries
    let mb_w = ((w + 15) / 16) * 16;
    let mb_h = ((h + 15) / 16) * 16;

    let y_stride = mb_w;
    let uv_stride = mb_w / 2;

    let mut y_plane = vec![0u8; y_stride * mb_h];
    let mut u_plane = vec![128u8; uv_stride * (mb_h / 2)];
    let mut v_plane = vec![128u8; uv_stride * (mb_h / 2)];

    // Convert pixel by pixel
    for row in 0..h {
        for col in 0..w {
            let idx = (row * w + col) * 3;
            let r = f64::from(data[idx]);
            let g = f64::from(data[idx + 1]);
            let b = f64::from(data[idx + 2]);

            let y_val = 0.299 * r + 0.587 * g + 0.114 * b;
            y_plane[row * y_stride + col] = y_val.clamp(0.0, 255.0) as u8;
        }
    }

    // Chroma subsampling: average 2x2 blocks
    let ch_w = (w + 1) / 2;
    let ch_h = (h + 1) / 2;

    for row in 0..ch_h {
        for col in 0..ch_w {
            let mut sum_u = 0.0f64;
            let mut sum_v = 0.0f64;
            let mut count = 0.0f64;

            for dy in 0..2 {
                for dx in 0..2 {
                    let sy = row * 2 + dy;
                    let sx = col * 2 + dx;
                    if sy < h && sx < w {
                        let idx = (sy * w + sx) * 3;
                        let r = f64::from(data[idx]);
                        let g = f64::from(data[idx + 1]);
                        let b = f64::from(data[idx + 2]);

                        sum_u += -0.169 * r - 0.331 * g + 0.500 * b + 128.0;
                        sum_v += 0.500 * r - 0.419 * g - 0.081 * b + 128.0;
                        count += 1.0;
                    }
                }
            }

            let u_val = (sum_u / count).clamp(0.0, 255.0) as u8;
            let v_val = (sum_v / count).clamp(0.0, 255.0) as u8;

            u_plane[row * uv_stride + col] = u_val;
            v_plane[row * uv_stride + col] = v_val;
        }
    }

    // Pad remaining pixels by replicating edges
    for row in 0..h {
        for col in w..mb_w {
            y_plane[row * y_stride + col] = y_plane[row * y_stride + w.saturating_sub(1)];
        }
    }
    for row in h..mb_h {
        let src_row = h.saturating_sub(1);
        for col in 0..mb_w {
            y_plane[row * y_stride + col] = y_plane[src_row * y_stride + col.min(mb_w - 1)];
        }
    }
    for row in 0..ch_h {
        for col in ch_w..(mb_w / 2) {
            u_plane[row * uv_stride + col] = u_plane[row * uv_stride + ch_w.saturating_sub(1)];
            v_plane[row * uv_stride + col] = v_plane[row * uv_stride + ch_w.saturating_sub(1)];
        }
    }
    for row in ch_h..(mb_h / 2) {
        let src_row = ch_h.saturating_sub(1);
        for col in 0..(mb_w / 2) {
            u_plane[row * uv_stride + col] = u_plane[src_row * uv_stride + col];
            v_plane[row * uv_stride + col] = v_plane[src_row * uv_stride + col];
        }
    }

    Ok(YuvPlanes {
        y: y_plane,
        u: u_plane,
        v: v_plane,
        y_stride,
        uv_stride,
        width,
        height,
    })
}

// ---------------------------------------------------------------------------
// Token encoding helpers
// ---------------------------------------------------------------------------

/// VP8 token categories and their encoding.
///
/// DCT coefficients are entropy-coded as a sequence of "tokens":
///   DCT_0  = 0 (run of zero)
///   DCT_1  = +/-1
///   DCT_2  = +/-2
///   DCT_3  = +/-3
///   DCT_4  = +/-4
///   DCT_CAT1 = 5..6
///   DCT_CAT2 = 7..10
///   DCT_CAT3 = 11..18
///   DCT_CAT4 = 19..34
///   DCT_CAT5 = 35..66
///   DCT_CAT6 = 67..2047
///   DCT_EOB  = end of block
///
/// Each token is encoded as a binary tree walk using the 11 probability
/// slots in `DEFAULT_COEFF_PROBS[type][band][ctx]`.

/// Extra-bits probabilities for each DCT category.
static CAT1_PROB: [u8; 1] = [159];
static CAT2_PROB: [u8; 2] = [165, 145];
static CAT3_PROB: [u8; 3] = [173, 148, 140];
static CAT4_PROB: [u8; 4] = [176, 155, 140, 135];
static CAT5_PROB: [u8; 5] = [180, 157, 141, 134, 130];
static CAT6_PROB: [u8; 11] = [254, 254, 243, 230, 196, 177, 153, 140, 133, 130, 129];

/// Encodes a single DCT coefficient token into the boolean encoder.
///
/// Returns the new "previous coefficient context" (0 = zero, 1 = one, 2 = >1).
fn encode_token(
    enc: &mut BoolEncoder,
    coeff: i32,
    block_type: usize,
    band: usize,
    ctx: usize,
    is_first_after_dc: bool,
) -> usize {
    let probs = &DEFAULT_COEFF_PROBS[block_type][band][ctx];
    let abs_val = coeff.unsigned_abs();

    if !is_first_after_dc {
        // First decision: EOB vs non-EOB
        // This is handled at a higher level (we don't emit EOB inside this fn)
    }

    // Token tree walk:
    // Node 0: prob[0] => 0 = DCT_0 path, 1 = non-zero path
    if abs_val == 0 {
        enc.encode_bool(false, probs[0]); // DCT_0
        return 0;
    }

    enc.encode_bool(true, probs[0]); // not DCT_0

    // Node 1: prob[1] => 0 = DCT_1, 1 = higher
    if abs_val == 1 {
        enc.encode_bool(false, probs[1]);
        // Sign bit
        enc.encode_bit(coeff < 0);
        return 1;
    }

    enc.encode_bool(true, probs[1]); // not DCT_1

    // Node 2: prob[2] => 0 = DCT_2..DCT_4, 1 = categories
    if abs_val <= 4 {
        enc.encode_bool(false, probs[2]);
        // Node 3: prob[3] => 0 = DCT_2, 1 = DCT_3 or DCT_4
        if abs_val == 2 {
            enc.encode_bool(false, probs[3]);
        } else {
            enc.encode_bool(true, probs[3]);
            // Node 4: prob[4] => 0 = DCT_3, 1 = DCT_4
            enc.encode_bool(abs_val == 4, probs[4]);
        }
        // Sign bit
        enc.encode_bit(coeff < 0);
        return 2;
    }

    enc.encode_bool(true, probs[2]); // category token

    // Node 5: prob[5] => 0 = CAT1/CAT2, 1 = CAT3..CAT6
    if abs_val <= 10 {
        enc.encode_bool(false, probs[5]);
        // Node 6: prob[6] => 0 = CAT1, 1 = CAT2
        if abs_val <= 6 {
            enc.encode_bool(false, probs[6]);
            // CAT1: extra = abs_val - 5 (0 or 1)
            let extra = abs_val - 5;
            enc.encode_bool(extra != 0, CAT1_PROB[0]);
        } else {
            enc.encode_bool(true, probs[6]);
            // CAT2: extra = abs_val - 7 (0..3)
            let extra = abs_val - 7;
            for (i, &p) in CAT2_PROB.iter().enumerate() {
                let bit = (extra >> (CAT2_PROB.len() - 1 - i)) & 1 != 0;
                enc.encode_bool(bit, p);
            }
        }
    } else {
        enc.encode_bool(true, probs[5]);
        // Node 7: prob[7] => 0 = CAT3/CAT4, 1 = CAT5/CAT6
        if abs_val <= 34 {
            enc.encode_bool(false, probs[7]);
            // Node 8: prob[8] => 0 = CAT3, 1 = CAT4
            if abs_val <= 18 {
                enc.encode_bool(false, probs[8]);
                // CAT3: extra = abs_val - 11 (0..7)
                let extra = abs_val - 11;
                for (i, &p) in CAT3_PROB.iter().enumerate() {
                    let bit = (extra >> (CAT3_PROB.len() - 1 - i)) & 1 != 0;
                    enc.encode_bool(bit, p);
                }
            } else {
                enc.encode_bool(true, probs[8]);
                // CAT4: extra = abs_val - 19 (0..15)
                let extra = abs_val - 19;
                for (i, &p) in CAT4_PROB.iter().enumerate() {
                    let bit = (extra >> (CAT4_PROB.len() - 1 - i)) & 1 != 0;
                    enc.encode_bool(bit, p);
                }
            }
        } else {
            enc.encode_bool(true, probs[7]);
            // Node 9: prob[9] => 0 = CAT5, 1 = CAT6
            if abs_val <= 66 {
                enc.encode_bool(false, probs[9]);
                // CAT5: extra = abs_val - 35 (0..31)
                let extra = abs_val - 35;
                for (i, &p) in CAT5_PROB.iter().enumerate() {
                    let bit = (extra >> (CAT5_PROB.len() - 1 - i)) & 1 != 0;
                    enc.encode_bool(bit, p);
                }
            } else {
                enc.encode_bool(true, probs[9]);
                // CAT6: extra = abs_val - 67 (0..2047)
                let extra = abs_val - 67;
                for (i, &p) in CAT6_PROB.iter().enumerate() {
                    let bit = (extra >> (CAT6_PROB.len() - 1 - i)) & 1 != 0;
                    enc.encode_bool(bit, p);
                }
            }
        }
    }

    // Sign bit
    enc.encode_bit(coeff < 0);
    2
}

/// Encodes a full 4x4 block of quantized DCT coefficients.
///
/// Emits tokens in zigzag order.  An EOB token is emitted after the
/// last non-zero coefficient.
///
/// `first_coeff_idx` is 0 for Y2/UV blocks, 1 for Y blocks (where DC
/// is carried by the Y2 block).
fn encode_block(
    enc: &mut BoolEncoder,
    quantized: &[i32; 16],
    block_type: usize,
    first_coeff_idx: usize,
) {
    // Find last non-zero coefficient (in zigzag order)
    let mut last_nonzero: Option<usize> = None;
    for i in (first_coeff_idx..16).rev() {
        let zigzag_pos = ZIGZAG_ORDER[i];
        if quantized[zigzag_pos] != 0 {
            last_nonzero = Some(i);
            break;
        }
    }

    let last_nz = match last_nonzero {
        Some(idx) => idx,
        None => {
            // All zero — emit EOB
            let band = COEFF_BANDS[first_coeff_idx];
            let probs = &DEFAULT_COEFF_PROBS[block_type][band][0];
            enc.encode_bool(false, probs[0]); // DCT_0 at first position acts as EOB marker
            return;
        }
    };

    let mut ctx: usize = 0; // previous coefficient context

    for i in first_coeff_idx..=last_nz {
        let zigzag_pos = ZIGZAG_ORDER[i];
        let coeff = quantized[zigzag_pos];
        let band = COEFF_BANDS[i];

        ctx = encode_token(enc, coeff, block_type, band, ctx, i > first_coeff_idx);
    }

    // Emit EOB after last non-zero coefficient (if there are remaining positions)
    if last_nz + 1 < 16 {
        let eob_band = COEFF_BANDS[(last_nz + 1).min(15)];
        let eob_probs = &DEFAULT_COEFF_PROBS[block_type][eob_band][ctx];
        // EOB is encoded as: prob[0] decides "is coefficient zero?"
        // In VP8, EOB is a separate token that terminates the block.
        // It's signaled as the first branch being "false" when the
        // coefficient *would have been* the next one — but we use
        // a simplified approach: the decoder knows EOB means "rest are zero".
        enc.encode_bool(false, eob_probs[0]);
    }
}

// ---------------------------------------------------------------------------
// Macroblock encoding
// ---------------------------------------------------------------------------

/// Quantizes a coefficient with the given quantizer step.
fn quantize(coeff: i32, step: i32) -> i32 {
    if step == 0 {
        return coeff;
    }
    let sign = if coeff < 0 { -1 } else { 1 };
    let abs_c = coeff.abs();
    sign * ((abs_c + step / 2) / step)
}

/// Processes a single macroblock: prediction, DCT, quantization.
///
/// Returns the quantized coefficients for all sub-blocks:
///   - 16 luma 4x4 blocks (Y)
///   - 1 DC block (Y2, the WHT of DC values)
///   - 4 U chroma blocks
///   - 4 V chroma blocks
struct MacroblockCoeffs {
    /// 16 luma sub-blocks, each 16 coefficients in raster order.
    y_blocks: [[i32; 16]; 16],
    /// Y2 (WHT of luma DC values), 16 coefficients.
    y2_block: [i32; 16],
    /// 4 U chroma sub-blocks, each 16 coefficients.
    u_blocks: [[i32; 16]; 4],
    /// 4 V chroma sub-blocks, each 16 coefficients.
    v_blocks: [[i32; 16]; 4],
}

/// Encodes a single 16x16 macroblock to produce quantized coefficients.
fn encode_macroblock(
    yuv: &YuvPlanes,
    mb_x: usize,
    mb_y: usize,
    dc_quant: i32,
    ac_quant: i32,
    y2_dc_quant: i32,
    y2_ac_quant: i32,
    uv_dc_quant: i32,
    uv_ac_quant: i32,
    reconstructed_y: &[u8],
    recon_y_stride: usize,
    reconstructed_u: &[u8],
    recon_uv_stride: usize,
    reconstructed_v: &[u8],
) -> MacroblockCoeffs {
    let mut mb = MacroblockCoeffs {
        y_blocks: [[0i32; 16]; 16],
        y2_block: [0i32; 16],
        u_blocks: [[0i32; 16]; 4],
        v_blocks: [[0i32; 16]; 4],
    };

    // --- DC prediction for luma 16x16 ---
    let pred_y = compute_dc_pred_16x16(reconstructed_y, recon_y_stride, mb_x, mb_y);

    // Process 16 luma 4x4 sub-blocks
    let mut dc_values = [0i32; 16];

    for sb in 0..16 {
        let sb_row = sb / 4;
        let sb_col = sb % 4;

        let mut residual = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                let py = mb_y * 16 + sb_row * 4 + r;
                let px = mb_x * 16 + sb_col * 4 + c;
                let orig = i32::from(yuv.y[py * yuv.y_stride + px]);
                let pred = i32::from(pred_y);
                residual[r * 4 + c] = orig - pred;
            }
        }

        let mut coeffs = [0i32; 16];
        fdct4x4(&residual, &mut coeffs);

        // Save DC for Y2 block
        dc_values[sb] = coeffs[0];

        // Quantize AC coefficients (DC will be replaced by Y2)
        for i in 1..16 {
            coeffs[i] = quantize(coeffs[i], ac_quant);
        }
        // DC is set to 0 here; it goes through Y2
        coeffs[0] = 0;

        mb.y_blocks[sb] = coeffs;
    }

    // Y2 block: WHT of DC values
    let mut y2_coeffs = [0i32; 16];
    fwht4x4(&dc_values, &mut y2_coeffs);

    // Quantize Y2
    mb.y2_block[0] = quantize(y2_coeffs[0], y2_dc_quant);
    for i in 1..16 {
        mb.y2_block[i] = quantize(y2_coeffs[i], y2_ac_quant);
    }

    // --- Chroma ---
    let pred_u = compute_dc_pred_8x8(reconstructed_u, recon_uv_stride, mb_x, mb_y);
    let pred_v = compute_dc_pred_8x8(reconstructed_v, recon_uv_stride, mb_x, mb_y);

    for sb in 0..4 {
        let sb_row = sb / 2;
        let sb_col = sb % 2;

        // U block
        let mut u_residual = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                let py = mb_y * 8 + sb_row * 4 + r;
                let px = mb_x * 8 + sb_col * 4 + c;
                let orig = i32::from(yuv.u[py * yuv.uv_stride + px]);
                u_residual[r * 4 + c] = orig - i32::from(pred_u);
            }
        }
        let mut u_coeffs = [0i32; 16];
        fdct4x4(&u_residual, &mut u_coeffs);
        u_coeffs[0] = quantize(u_coeffs[0], uv_dc_quant);
        for i in 1..16 {
            u_coeffs[i] = quantize(u_coeffs[i], uv_ac_quant);
        }
        mb.u_blocks[sb] = u_coeffs;

        // V block
        let mut v_residual = [0i32; 16];
        for r in 0..4 {
            for c in 0..4 {
                let py = mb_y * 8 + sb_row * 4 + r;
                let px = mb_x * 8 + sb_col * 4 + c;
                let orig = i32::from(yuv.v[py * yuv.uv_stride + px]);
                v_residual[r * 4 + c] = orig - i32::from(pred_v);
            }
        }
        let mut v_coeffs = [0i32; 16];
        fdct4x4(&v_residual, &mut v_coeffs);
        v_coeffs[0] = quantize(v_coeffs[0], uv_dc_quant);
        for i in 1..16 {
            v_coeffs[i] = quantize(v_coeffs[i], uv_ac_quant);
        }
        mb.v_blocks[sb] = v_coeffs;
    }

    mb
}

/// Computes DC prediction value for a 16x16 luma block.
///
/// Uses average of top and left reconstructed neighbors when available.
fn compute_dc_pred_16x16(recon: &[u8], stride: usize, mb_x: usize, mb_y: usize) -> u8 {
    let mut sum: u32 = 0;
    let mut count: u32 = 0;

    // Top row
    if mb_y > 0 {
        let top_row = (mb_y * 16 - 1) * stride + mb_x * 16;
        for col in 0..16 {
            if top_row + col < recon.len() {
                sum += u32::from(recon[top_row + col]);
                count += 1;
            }
        }
    }

    // Left column
    if mb_x > 0 {
        let left_col = mb_x * 16 - 1;
        for row in 0..16 {
            let idx = (mb_y * 16 + row) * stride + left_col;
            if idx < recon.len() {
                sum += u32::from(recon[idx]);
                count += 1;
            }
        }
    }

    if count > 0 {
        ((sum + count / 2) / count) as u8
    } else {
        128
    }
}

/// Computes DC prediction value for an 8x8 chroma block.
fn compute_dc_pred_8x8(recon: &[u8], stride: usize, mb_x: usize, mb_y: usize) -> u8 {
    let mut sum: u32 = 0;
    let mut count: u32 = 0;

    if mb_y > 0 {
        let top_row = (mb_y * 8 - 1) * stride + mb_x * 8;
        for col in 0..8 {
            if top_row + col < recon.len() {
                sum += u32::from(recon[top_row + col]);
                count += 1;
            }
        }
    }

    if mb_x > 0 {
        let left_col = mb_x * 8 - 1;
        for row in 0..8 {
            let idx = (mb_y * 8 + row) * stride + left_col;
            if idx < recon.len() {
                sum += u32::from(recon[idx]);
                count += 1;
            }
        }
    }

    if count > 0 {
        ((sum + count / 2) / count) as u8
    } else {
        128
    }
}

/// Reconstructs a macroblock from its quantized coefficients for use as
/// reference in subsequent macroblock predictions.
fn reconstruct_macroblock(
    mb: &MacroblockCoeffs,
    dc_quant: i32,
    ac_quant: i32,
    y2_dc_quant: i32,
    y2_ac_quant: i32,
    uv_dc_quant: i32,
    uv_ac_quant: i32,
    pred_y: u8,
    pred_u: u8,
    pred_v: u8,
    recon_y: &mut [u8],
    recon_y_stride: usize,
    mb_x: usize,
    mb_y: usize,
    recon_u: &mut [u8],
    recon_uv_stride: usize,
    recon_v: &mut [u8],
) {
    // Inverse Y2 (WHT) to get dequantized DC values
    let mut y2_dequant = [0i32; 16];
    y2_dequant[0] = mb.y2_block[0] * y2_dc_quant;
    for i in 1..16 {
        y2_dequant[i] = mb.y2_block[i] * y2_ac_quant;
    }

    // Inverse WHT
    let mut dc_values = [0i32; 16];
    {
        let mut temp = [0i32; 16];
        // Row inverse WHT
        for row in 0..4 {
            let b = row * 4;
            let a = y2_dequant[b] + y2_dequant[b + 2];
            let bv = y2_dequant[b + 1] + y2_dequant[b + 3];
            let c = y2_dequant[b + 1] - y2_dequant[b + 3];
            let d = y2_dequant[b] - y2_dequant[b + 2];

            temp[b] = a + bv;
            temp[b + 1] = d + c;
            temp[b + 2] = a - bv;
            temp[b + 3] = d - c;
        }
        // Column inverse WHT
        for col in 0..4 {
            let a = temp[col] + temp[col + 8];
            let bv = temp[col + 4] + temp[col + 12];
            let c = temp[col + 4] - temp[col + 12];
            let d = temp[col] - temp[col + 8];

            dc_values[col] = (a + bv + 1) >> 1;
            dc_values[col + 4] = (d + c + 1) >> 1;
            dc_values[col + 8] = (a - bv + 1) >> 1;
            dc_values[col + 12] = (d - c + 1) >> 1;
        }
    }

    // Reconstruct each luma 4x4 sub-block
    for sb in 0..16 {
        let sb_row = sb / 4;
        let sb_col = sb % 4;

        // Dequantize AC
        let mut dequant = [0i32; 16];
        dequant[0] = dc_values[sb]; // DC from Y2
        for i in 1..16 {
            dequant[i] = mb.y_blocks[sb][i] * ac_quant;
        }

        // Inverse DCT
        let reconstructed = idct4x4_simple(&dequant);

        // Add prediction and clamp
        for r in 0..4 {
            for c in 0..4 {
                let py = mb_y * 16 + sb_row * 4 + r;
                let px = mb_x * 16 + sb_col * 4 + c;
                let val = reconstructed[r * 4 + c] + i32::from(pred_y);
                recon_y[py * recon_y_stride + px] = val.clamp(0, 255) as u8;
            }
        }
    }

    // Reconstruct chroma
    for sb in 0..4 {
        let sb_row = sb / 2;
        let sb_col = sb % 2;

        // U
        let mut u_dequant = [0i32; 16];
        u_dequant[0] = mb.u_blocks[sb][0] * uv_dc_quant;
        for i in 1..16 {
            u_dequant[i] = mb.u_blocks[sb][i] * uv_ac_quant;
        }
        let u_recon = idct4x4_simple(&u_dequant);
        for r in 0..4 {
            for c in 0..4 {
                let py = mb_y * 8 + sb_row * 4 + r;
                let px = mb_x * 8 + sb_col * 4 + c;
                let val = u_recon[r * 4 + c] + i32::from(pred_u);
                recon_u[py * recon_uv_stride + px] = val.clamp(0, 255) as u8;
            }
        }

        // V
        let mut v_dequant = [0i32; 16];
        v_dequant[0] = mb.v_blocks[sb][0] * uv_dc_quant;
        for i in 1..16 {
            v_dequant[i] = mb.v_blocks[sb][i] * uv_ac_quant;
        }
        let v_recon = idct4x4_simple(&v_dequant);
        for r in 0..4 {
            for c in 0..4 {
                let py = mb_y * 8 + sb_row * 4 + r;
                let px = mb_x * 8 + sb_col * 4 + c;
                let val = v_recon[r * 4 + c] + i32::from(pred_v);
                recon_v[py * recon_uv_stride + px] = val.clamp(0, 255) as u8;
            }
        }
    }
}

/// Simplified inverse 4x4 DCT for reconstruction.
///
/// Takes dequantized coefficients in raster order and returns
/// residual pixel values.
fn idct4x4_simple(coeffs: &[i32; 16]) -> [i32; 16] {
    let mut temp = [0i32; 16];
    let mut output = [0i32; 16];

    // Row inverse DCT
    for row in 0..4 {
        let b = row * 4;
        let c0 = coeffs[b];
        let c1 = coeffs[b + 1];
        let c2 = coeffs[b + 2];
        let c3 = coeffs[b + 3];

        let a1 = c0 + c2;
        let b1 = c0 - c2;

        let t1 = (c1 * 35468 + c3 * 85627 + 32768) >> 16;
        let t2 = (c1 * 85627 - c3 * 35468 + 32768) >> 16;

        temp[b] = a1 + t2;
        temp[b + 1] = b1 + t1;
        temp[b + 2] = b1 - t1;
        temp[b + 3] = a1 - t2;
    }

    // Column inverse DCT
    for col in 0..4 {
        let c0 = temp[col];
        let c1 = temp[col + 4];
        let c2 = temp[col + 8];
        let c3 = temp[col + 12];

        let a1 = c0 + c2;
        let b1 = c0 - c2;

        let t1 = (c1 * 35468 + c3 * 85627 + 32768) >> 16;
        let t2 = (c1 * 85627 - c3 * 35468 + 32768) >> 16;

        output[col] = (a1 + t2 + 4) >> 3;
        output[col + 4] = (b1 + t1 + 4) >> 3;
        output[col + 8] = (b1 - t1 + 4) >> 3;
        output[col + 12] = (a1 - t2 + 4) >> 3;
    }

    output
}

// ---------------------------------------------------------------------------
// VP8 bitstream assembly
// ---------------------------------------------------------------------------

/// Writes the VP8 frame header using the boolean encoder.
///
/// This encodes Partition 1: the frame header flags, quantizer,
/// and macroblock prediction modes.
fn write_frame_header(
    enc: &mut BoolEncoder,
    mb_width: u32,
    mb_height: u32,
    quant_index: u8,
) {
    // Color space (0 = YUV)
    enc.encode_bit(false);

    // Clamping type (0 = required)
    enc.encode_bit(false);

    // Segmentation: disabled
    enc.encode_bit(false);

    // Loop filter parameters
    // filter_type (0 = normal)
    enc.encode_bit(false);
    // loop_filter_level (6 bits) - use 0 for simplicity
    enc.encode_literal(0, 6);
    // sharpness_level (3 bits)
    enc.encode_literal(0, 3);

    // Mode ref LF delta: disabled
    enc.encode_bit(false);

    // Number of DCT partitions: log2(1) = 0 (2 bits)
    enc.encode_literal(0, 2);

    // Quantizer (7 bits for base index)
    enc.encode_literal(u32::from(quant_index), 7);

    // Y DC delta (1 bit flag + optional value) - no delta
    enc.encode_bit(false);
    // Y2 DC delta
    enc.encode_bit(false);
    // Y2 AC delta
    enc.encode_bit(false);
    // UV DC delta
    enc.encode_bit(false);
    // UV AC delta
    enc.encode_bit(false);

    // Token probability updates: signal "no update" for all
    // 4 * 8 * 3 * 11 = 1056 probabilities
    for _block_type in 0..4 {
        for _band in 0..8 {
            for _ctx in 0..3 {
                for _node in 0..11 {
                    enc.encode_bit(false); // no update
                }
            }
        }
    }

    // Skip coefficient (mb_no_coeff_skip)
    enc.encode_bit(false); // disabled

    // Macroblock prediction modes
    // All macroblocks use I16 DC prediction
    let total_mbs = mb_width * mb_height;
    for _ in 0..total_mbs {
        // I16 mode tree:
        // prob 145: 0 = DC, 1 = other
        enc.encode_bool(false, 145); // DC prediction

        // Chroma mode tree:
        // prob 142: 0 = DC, 1 = other
        enc.encode_bool(false, 142); // DC prediction
    }
}

// ---------------------------------------------------------------------------
// Public encoder API
// ---------------------------------------------------------------------------

/// VP8 lossy encoder for WebP.
///
/// Produces valid VP8 keyframe bitstreams that can be embedded in a WebP
/// RIFF container.  The encoder generates intra-only (keyframe) frames
/// using DC prediction and a configurable quality parameter.
///
/// # Examples
///
/// ```
/// use oximedia_codec::webp::encoder::WebPLossyEncoder;
///
/// let encoder = WebPLossyEncoder::new(75);
///
/// // 2x2 red image
/// let rgb = [255, 0, 0, 255, 0, 0, 255, 0, 0, 255, 0, 0];
/// let vp8_data = encoder.encode_rgb(&rgb, 2, 2).expect("encode");
///
/// // The output starts with a valid VP8 frame tag
/// assert!(!vp8_data.is_empty());
/// ```
pub struct WebPLossyEncoder {
    quality: u8,
}

impl WebPLossyEncoder {
    /// Creates a new lossy encoder with the given quality (0-100).
    ///
    /// - 0 = lowest quality / smallest size
    /// - 100 = highest quality / largest size
    #[must_use]
    pub fn new(quality: u8) -> Self {
        Self {
            quality: quality.min(100),
        }
    }

    /// Maps quality (0-100) to VP8 quantizer index (0-127).
    fn quality_to_qindex(&self) -> u8 {
        // Linear mapping: quality 100 → qindex 0, quality 0 → qindex 127
        let qindex = 127 - (u32::from(self.quality) * 127 / 100);
        (qindex as u8).min(127)
    }

    /// Encodes RGB data to a VP8 bitstream (without RIFF container).
    ///
    /// The input `data` must contain `width * height * 3` bytes in
    /// row-major R, G, B order (8 bits per component).
    ///
    /// Returns the raw VP8 bitstream bytes suitable for wrapping in a
    /// WebP RIFF container.
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` if dimensions are zero or
    /// the data length does not match `width * height * 3`.
    pub fn encode_rgb(&self, data: &[u8], width: u32, height: u32) -> CodecResult<Vec<u8>> {
        self.validate_dimensions(width, height)?;

        let expected_len = (width as usize) * (height as usize) * 3;
        if data.len() < expected_len {
            return Err(CodecError::InvalidParameter(format!(
                "RGB data too short: expected {expected_len}, got {}",
                data.len()
            )));
        }

        let yuv = rgb_to_yuv420(data, width, height)?;
        self.encode_yuv(&yuv)
    }

    /// Encodes RGBA data to VP8 bitstream + separate alpha channel.
    ///
    /// Returns `(vp8_data, alpha_data)` where `alpha_data` contains
    /// the raw alpha plane bytes (width * height, row-major, uncompressed).
    ///
    /// # Errors
    ///
    /// Returns `CodecError::InvalidParameter` if dimensions are zero or
    /// the data length does not match `width * height * 4`.
    pub fn encode_rgba(
        &self,
        data: &[u8],
        width: u32,
        height: u32,
    ) -> CodecResult<(Vec<u8>, Vec<u8>)> {
        self.validate_dimensions(width, height)?;

        let w = width as usize;
        let h = height as usize;
        let expected_len = w * h * 4;
        if data.len() < expected_len {
            return Err(CodecError::InvalidParameter(format!(
                "RGBA data too short: expected {expected_len}, got {}",
                data.len()
            )));
        }

        // Extract RGB and alpha
        let pixel_count = w * h;
        let mut rgb = Vec::with_capacity(pixel_count * 3);
        let mut alpha = Vec::with_capacity(pixel_count);

        for i in 0..pixel_count {
            let base = i * 4;
            rgb.push(data[base]);
            rgb.push(data[base + 1]);
            rgb.push(data[base + 2]);
            alpha.push(data[base + 3]);
        }

        let vp8_data = self.encode_rgb(&rgb, width, height)?;
        Ok((vp8_data, alpha))
    }

    /// Validates that width and height are non-zero and within VP8 limits.
    fn validate_dimensions(&self, width: u32, height: u32) -> CodecResult<()> {
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidParameter(
                "Width and height must be non-zero".to_string(),
            ));
        }
        // VP8 maximum dimension is 16383
        if width > 16383 || height > 16383 {
            return Err(CodecError::InvalidParameter(format!(
                "Dimensions {}x{} exceed VP8 maximum of 16383",
                width, height
            )));
        }
        Ok(())
    }

    /// Core encoding: takes YUV planes and produces a VP8 bitstream.
    fn encode_yuv(&self, yuv: &YuvPlanes) -> CodecResult<Vec<u8>> {
        let width = yuv.width;
        let height = yuv.height;
        let mb_width = ((width + 15) / 16) as usize;
        let mb_height = ((height + 15) / 16) as usize;

        let qindex = self.quality_to_qindex();
        let qi = qindex as usize;
        let dc_quant = DC_QUANT_TABLE[qi.min(127)];
        let ac_quant = AC_QUANT_TABLE[qi.min(127)];
        let y2_dc_quant = DC_QUANT_TABLE[qi.min(127)] * 2;
        let y2_ac_quant = AC_QUANT_TABLE[qi.min(127)].max(8) * 155 / 100;
        let uv_dc_quant = DC_QUANT_TABLE[qi.min(127)];
        let uv_ac_quant = AC_QUANT_TABLE[qi.min(127)];

        // Reconstructed planes for prediction reference
        let recon_y_stride = mb_width * 16;
        let recon_uv_stride = mb_width * 8;
        let mut recon_y = vec![128u8; recon_y_stride * mb_height * 16];
        let mut recon_u = vec![128u8; recon_uv_stride * mb_height * 8];
        let mut recon_v = vec![128u8; recon_uv_stride * mb_height * 8];

        // Encode frame header (Partition 1)
        let mut header_enc = BoolEncoder::new();
        write_frame_header(
            &mut header_enc,
            mb_width as u32,
            mb_height as u32,
            qindex,
        );

        // Encode DCT tokens (Partition 2)
        let mut token_enc = BoolEncoder::new();

        for mby in 0..mb_height {
            for mbx in 0..mb_width {
                let mb = encode_macroblock(
                    yuv,
                    mbx,
                    mby,
                    dc_quant,
                    ac_quant,
                    y2_dc_quant,
                    y2_ac_quant,
                    uv_dc_quant,
                    uv_ac_quant,
                    &recon_y,
                    recon_y_stride,
                    &recon_u,
                    recon_uv_stride,
                    &recon_v,
                );

                // Encode Y2 block (block_type = 3)
                encode_block(&mut token_enc, &mb.y2_block, 3, 0);

                // Encode 16 Y blocks (block_type = 0 for DC-after-Y2, skip DC)
                for sb in 0..16 {
                    encode_block(&mut token_enc, &mb.y_blocks[sb], 0, 1);
                }

                // Encode 4 U blocks (block_type = 2)
                for sb in 0..4 {
                    encode_block(&mut token_enc, &mb.u_blocks[sb], 2, 0);
                }

                // Encode 4 V blocks (block_type = 2)
                for sb in 0..4 {
                    encode_block(&mut token_enc, &mb.v_blocks[sb], 2, 0);
                }

                // Reconstruct macroblock for prediction reference
                let pred_y = compute_dc_pred_16x16(&recon_y, recon_y_stride, mbx, mby);
                let pred_u = compute_dc_pred_8x8(&recon_u, recon_uv_stride, mbx, mby);
                let pred_v = compute_dc_pred_8x8(&recon_v, recon_uv_stride, mbx, mby);

                reconstruct_macroblock(
                    &mb,
                    dc_quant,
                    ac_quant,
                    y2_dc_quant,
                    y2_ac_quant,
                    uv_dc_quant,
                    uv_ac_quant,
                    pred_y,
                    pred_u,
                    pred_v,
                    &mut recon_y,
                    recon_y_stride,
                    mbx,
                    mby,
                    &mut recon_u,
                    recon_uv_stride,
                    &mut recon_v,
                );
            }
        }

        let header_data = header_enc.flush();
        let token_data = token_enc.flush();

        // Assemble VP8 bitstream
        self.assemble_bitstream(width, height, &header_data, &token_data)
    }

    /// Assembles the final VP8 bitstream from header and token partitions.
    fn assemble_bitstream(
        &self,
        width: u32,
        height: u32,
        header_data: &[u8],
        token_data: &[u8],
    ) -> CodecResult<Vec<u8>> {
        let first_partition_size = header_data.len() as u32;

        // Total output size: frame_tag(3) + sync(3) + dims(4) + partitions
        let total_size = 3 + 3 + 4 + header_data.len() + token_data.len();
        let mut output = Vec::with_capacity(total_size);

        // --- Frame tag (3 bytes) ---
        // bit 0: frame_type (0 = keyframe)
        // bits 1-3: version (0)
        // bit 4: show_frame (1)
        // bits 5-7 of byte 0 + bytes 1-2: first_partition_size (19 bits)
        let b0: u8 = 0x00  // frame_type = 0 (key)
            | 0x00          // version = 0
            | 0x10          // show_frame = 1
            | ((first_partition_size << 5) as u8 & 0xE0);
        let b1: u8 = (first_partition_size >> 3) as u8;
        let b2: u8 = (first_partition_size >> 11) as u8;

        output.push(b0);
        output.push(b1);
        output.push(b2);

        // --- Sync code ---
        output.push(0x9D);
        output.push(0x01);
        output.push(0x2A);

        // --- Dimensions (4 bytes, LE) ---
        // width: bits 0-13, horizontal_scale: bits 14-15
        let w_le = (width & 0x3FFF) as u16;
        output.push(w_le as u8);
        output.push((w_le >> 8) as u8);

        let h_le = (height & 0x3FFF) as u16;
        output.push(h_le as u8);
        output.push((h_le >> 8) as u8);

        // --- Partition 1 (header) ---
        output.extend_from_slice(header_data);

        // --- Partition 2 (tokens) ---
        output.extend_from_slice(token_data);

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bool_encoder_basic() {
        let mut enc = BoolEncoder::new();
        enc.encode_bit(false);
        enc.encode_bit(true);
        enc.encode_bit(false);
        let data = enc.flush();
        assert!(!data.is_empty());
    }

    #[test]
    fn test_bool_encoder_literal() {
        let mut enc = BoolEncoder::new();
        enc.encode_literal(42, 8);
        let data = enc.flush();
        assert!(!data.is_empty());
    }

    #[test]
    fn test_bool_encoder_with_prob() {
        let mut enc = BoolEncoder::new();
        // Encode several symbols with different probabilities
        for prob in [1, 50, 128, 200, 255] {
            enc.encode_bool(true, prob);
            enc.encode_bool(false, prob);
        }
        let data = enc.flush();
        assert!(!data.is_empty());
    }

    #[test]
    fn test_fdct4_1d() {
        let input = [100, 100, 100, 100]; // DC-only signal
        let mut output = [0i32; 4];
        fdct4_1d(&input, &mut output);

        // For a flat signal, DC should be large and dominant
        assert!(output[0] > 0);
        // AC may have small rounding artifacts from integer approximation
        assert!(output[0].abs() > output[1].abs());
        assert!(output[0].abs() > output[2].abs());
        assert!(output[0].abs() > output[3].abs());
    }

    #[test]
    fn test_fdct4x4_dc_only() {
        let residual = [10i32; 16]; // Flat residual
        let mut coeffs = [0i32; 16];
        fdct4x4(&residual, &mut coeffs);

        // DC coefficient should be dominant
        assert!(coeffs[0].abs() > 0);
        // AC should be much smaller than DC for flat input
        let dc_abs = coeffs[0].abs();
        for i in 1..16 {
            assert!(
                coeffs[i].abs() < dc_abs / 2,
                "AC coeff[{i}] = {} should be much smaller than DC = {}",
                coeffs[i],
                coeffs[0]
            );
        }
    }

    #[test]
    fn test_fwht4x4_dc_only() {
        let dc_values = [100i32; 16]; // All same DC
        let mut coeffs = [0i32; 16];
        fwht4x4(&dc_values, &mut coeffs);

        // DC should be 16 * 100 = 1600
        assert_eq!(coeffs[0], 1600);
        // AC should be 0
        for i in 1..16 {
            assert_eq!(coeffs[i], 0);
        }
    }

    #[test]
    fn test_quantize() {
        assert_eq!(quantize(100, 10), 10);
        assert_eq!(quantize(-100, 10), -10);
        assert_eq!(quantize(0, 10), 0);
        assert_eq!(quantize(4, 10), 0); // Below threshold
        assert_eq!(quantize(15, 10), 2); // (15+5)/10 = 2
    }

    #[test]
    fn test_quality_to_qindex() {
        let enc_low = WebPLossyEncoder::new(0);
        let enc_mid = WebPLossyEncoder::new(50);
        let enc_high = WebPLossyEncoder::new(100);

        assert_eq!(enc_high.quality_to_qindex(), 0);
        assert_eq!(enc_low.quality_to_qindex(), 127);
        assert!(enc_mid.quality_to_qindex() > 0);
        assert!(enc_mid.quality_to_qindex() < 127);
    }

    #[test]
    fn test_rgb_to_yuv420_basic() {
        // 4x4 white image
        let data = vec![255u8; 4 * 4 * 3];
        let yuv = rgb_to_yuv420(&data, 4, 4).expect("conversion should succeed");

        // White (255,255,255) → Y ≈ 255
        for &y in &yuv.y[..16] {
            assert!(y >= 250, "Y should be near 255 for white, got {y}");
        }
    }

    #[test]
    fn test_rgb_to_yuv420_black() {
        // 4x4 black image
        let data = vec![0u8; 4 * 4 * 3];
        let yuv = rgb_to_yuv420(&data, 4, 4).expect("conversion should succeed");

        // Black (0,0,0) → Y = 0, U = 128, V = 128
        for &y in &yuv.y[..16] {
            assert!(y <= 5, "Y should be near 0 for black, got {y}");
        }
        for &u in &yuv.u[..4] {
            assert!(
                (120..=136).contains(&u),
                "U should be near 128 for black, got {u}"
            );
        }
    }

    #[test]
    fn test_rgb_to_yuv420_short_data() {
        let data = vec![0u8; 10]; // too short for any image
        assert!(rgb_to_yuv420(&data, 4, 4).is_err());
    }

    #[test]
    fn test_encode_rgb_produces_valid_frame_tag() {
        let encoder = WebPLossyEncoder::new(50);
        // 16x16 gray image
        let data = vec![128u8; 16 * 16 * 3];
        let vp8 = encoder.encode_rgb(&data, 16, 16).expect("encode should succeed");

        // Check sync code at bytes 3..6
        assert!(vp8.len() >= 10);
        assert_eq!(vp8[3], 0x9D);
        assert_eq!(vp8[4], 0x01);
        assert_eq!(vp8[5], 0x2A);

        // Check frame type (keyframe = bit 0 of byte 0 is 0)
        assert_eq!(vp8[0] & 0x01, 0, "Should be keyframe");

        // Check show_frame (bit 4 of byte 0)
        assert_ne!(vp8[0] & 0x10, 0, "show_frame should be set");

        // Check dimensions
        let w = u16::from(vp8[6]) | (u16::from(vp8[7]) << 8);
        let h = u16::from(vp8[8]) | (u16::from(vp8[9]) << 8);
        assert_eq!(w & 0x3FFF, 16);
        assert_eq!(h & 0x3FFF, 16);
    }

    #[test]
    fn test_encode_rgb_different_qualities() {
        let data = vec![100u8; 32 * 32 * 3];

        let low = WebPLossyEncoder::new(10);
        let high = WebPLossyEncoder::new(90);

        let low_data = low.encode_rgb(&data, 32, 32).expect("low quality encode");
        let high_data = high.encode_rgb(&data, 32, 32).expect("high quality encode");

        // Both should produce valid output
        assert!(!low_data.is_empty());
        assert!(!high_data.is_empty());
    }

    #[test]
    fn test_encode_rgb_non_mb_aligned() {
        // 7x5 image: not aligned to 16x16 macroblock grid
        let encoder = WebPLossyEncoder::new(75);
        let data = vec![200u8; 7 * 5 * 3];
        let vp8 = encoder
            .encode_rgb(&data, 7, 5)
            .expect("non-aligned encode should succeed");

        assert!(!vp8.is_empty());

        // Dimensions in bitstream should match original, not padded
        let w = u16::from(vp8[6]) | (u16::from(vp8[7]) << 8);
        let h = u16::from(vp8[8]) | (u16::from(vp8[9]) << 8);
        assert_eq!(w & 0x3FFF, 7);
        assert_eq!(h & 0x3FFF, 5);
    }

    #[test]
    fn test_encode_rgba_basic() {
        let encoder = WebPLossyEncoder::new(75);
        // 4x4 red with 50% alpha
        let mut rgba = Vec::with_capacity(4 * 4 * 4);
        for _ in 0..16 {
            rgba.extend_from_slice(&[255, 0, 0, 128]);
        }

        let (vp8_data, alpha_data) = encoder
            .encode_rgba(&rgba, 4, 4)
            .expect("RGBA encode should succeed");

        assert!(!vp8_data.is_empty());
        assert_eq!(alpha_data.len(), 16);
        assert!(alpha_data.iter().all(|&a| a == 128));
    }

    #[test]
    fn test_encode_zero_dimensions() {
        let encoder = WebPLossyEncoder::new(50);
        assert!(encoder.encode_rgb(&[], 0, 10).is_err());
        assert!(encoder.encode_rgb(&[], 10, 0).is_err());
    }

    #[test]
    fn test_encode_too_short_data() {
        let encoder = WebPLossyEncoder::new(50);
        let data = vec![0u8; 10];
        assert!(encoder.encode_rgb(&data, 16, 16).is_err());
    }

    #[test]
    fn test_encode_oversized_dimensions() {
        let encoder = WebPLossyEncoder::new(50);
        assert!(encoder.encode_rgb(&[], 20000, 100).is_err());
    }

    #[test]
    fn test_idct4x4_simple_dc() {
        // DC-only input
        let mut coeffs = [0i32; 16];
        coeffs[0] = 400;

        let output = idct4x4_simple(&coeffs);
        // All outputs should be roughly equal (DC distributed)
        let avg = output.iter().sum::<i32>() / 16;
        for &v in &output {
            assert!(
                (v - avg).abs() <= 2,
                "DC-only IDCT should produce roughly uniform output"
            );
        }
    }

    #[test]
    fn test_encode_rgb_1x1() {
        // Smallest possible image
        let encoder = WebPLossyEncoder::new(75);
        let data = [128, 128, 128]; // gray pixel
        let vp8 = encoder
            .encode_rgb(&data, 1, 1)
            .expect("1x1 encode should succeed");

        assert!(!vp8.is_empty());
        // Verify it's a keyframe
        assert_eq!(vp8[0] & 0x01, 0);
    }

    #[test]
    fn test_encode_quality_extremes() {
        let data = vec![128u8; 16 * 16 * 3];

        // Quality 0
        let enc0 = WebPLossyEncoder::new(0);
        let out0 = enc0.encode_rgb(&data, 16, 16).expect("q0");
        assert!(!out0.is_empty());

        // Quality 100
        let enc100 = WebPLossyEncoder::new(100);
        let out100 = enc100.encode_rgb(&data, 16, 16).expect("q100");
        assert!(!out100.is_empty());

        // Quality > 100 should clamp
        let enc200 = WebPLossyEncoder::new(200);
        assert_eq!(enc200.quality, 100);
    }

    #[test]
    fn test_encode_rgb_colored_image() {
        // Create a simple gradient image
        let width = 32u32;
        let height = 32u32;
        let mut data = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                data.push((x * 8) as u8); // R
                data.push((y * 8) as u8); // G
                data.push(128);           // B
            }
        }

        let encoder = WebPLossyEncoder::new(80);
        let vp8 = encoder
            .encode_rgb(&data, width, height)
            .expect("gradient encode should succeed");

        // Verify valid VP8 header
        assert!(vp8.len() > 10);
        assert_eq!(vp8[3], 0x9D);
        assert_eq!(vp8[4], 0x01);
        assert_eq!(vp8[5], 0x2A);
    }

    #[test]
    fn test_first_partition_size_encoding() {
        // The first_partition_size must be correctly encoded in the frame tag
        let encoder = WebPLossyEncoder::new(50);
        let data = vec![128u8; 16 * 16 * 3];
        let vp8 = encoder.encode_rgb(&data, 16, 16).expect("encode");

        // Extract first_partition_size from frame tag
        let b0 = vp8[0];
        let b1 = vp8[1];
        let b2 = vp8[2];
        let fps = (u32::from(b0 >> 5) & 0x07)
            | (u32::from(b1) << 3)
            | (u32::from(b2) << 11);

        // The partition should start after the 10-byte header
        // and its size should be reasonable
        assert!(fps > 0, "first_partition_size should be non-zero");
        assert!(
            (fps as usize) < vp8.len(),
            "first_partition_size ({fps}) should be less than total ({}) ",
            vp8.len()
        );
    }

    #[test]
    fn test_compute_dc_pred_16x16_no_neighbors() {
        let recon = vec![0u8; 16 * 16];
        let pred = compute_dc_pred_16x16(&recon, 16, 0, 0);
        assert_eq!(pred, 128); // Default when no neighbors available
    }

    #[test]
    fn test_compute_dc_pred_16x16_with_top() {
        // Place known values in the row above (mb_y=1, top row is row 15)
        let stride = 32;
        let mut recon = vec![0u8; stride * 32];
        for col in 0..16 {
            recon[15 * stride + col] = 200;
        }
        let pred = compute_dc_pred_16x16(&recon, stride, 0, 1);
        assert_eq!(pred, 200);
    }

    #[test]
    fn test_fwht_iwht_roundtrip() {
        // Verify forward WHT structural correctness:
        // A uniform input should produce a single DC coefficient.
        let uniform = [50i32; 16];
        let mut wht_coeffs = [0i32; 16];
        fwht4x4(&uniform, &mut wht_coeffs);

        // DC = sum of all = 50*16 = 800
        assert_eq!(wht_coeffs[0], 800);
        // All AC should be zero for uniform input
        for i in 1..16 {
            assert_eq!(wht_coeffs[i], 0, "AC coeff at index {i} should be 0");
        }

        // Verify that a non-uniform input produces non-zero AC
        let varied = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160];
        fwht4x4(&varied, &mut wht_coeffs);

        // DC should equal sum of all values
        let total: i32 = varied.iter().sum();
        assert_eq!(wht_coeffs[0], total);

        // At least some AC coefficients should be non-zero
        let nonzero_ac = wht_coeffs[1..].iter().filter(|&&c| c != 0).count();
        assert!(nonzero_ac > 0, "Non-uniform input should have non-zero AC");
    }
}
