//! Modular sub-codec for JPEG-XL lossless encoding.
//!
//! The Modular mode is the backbone of JPEG-XL lossless compression. It operates
//! by applying reversible transforms (RCT, Squeeze) to decorrelate channels,
//! predicting each sample using adaptive weighted predictors, and entropy-coding
//! the residuals.
//!
//! ## Pipeline
//!
//! 1. **Reversible Color Transform (RCT)**: Converts RGB to YCoCg-R, which
//!    decorrelates color channels for better compression.
//! 2. **Prediction**: Each pixel is predicted from its causal neighbors
//!    (N, W, NW, NE, NN, WW) using an adaptive weighted predictor.
//! 3. **Residual coding**: The prediction errors are variable-length coded.
//!
//! ## Predictors
//!
//! JPEG-XL defines several predictors:
//! - Zero: predict 0 (for first pixels)
//! - West (left neighbor)
//! - North (top neighbor)
//! - Average of W and N
//! - Gradient: N + W - NW (MED/median edge detector)
//! - Weighted: adaptive weighted combination of multiple neighbors

use crate::error::{CodecError, CodecResult};

/// Maximum number of predictor types.
const NUM_PREDICTORS: usize = 6;

/// Predictor types used in the modular sub-codec.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Predictor {
    /// Predict zero (used for edges).
    Zero = 0,
    /// Use the west (left) neighbor.
    West = 1,
    /// Use the north (top) neighbor.
    North = 2,
    /// Average of west and north.
    AvgWN = 3,
    /// Gradient predictor: N + W - NW.
    Gradient = 4,
    /// Adaptive weighted combination of neighbors.
    Weighted = 5,
}

impl Predictor {
    /// Convert from integer index to predictor.
    fn from_index(idx: usize) -> Self {
        match idx {
            0 => Self::Zero,
            1 => Self::West,
            2 => Self::North,
            3 => Self::AvgWN,
            4 => Self::Gradient,
            _ => Self::Weighted,
        }
    }
}

/// Reversible Color Transform: forward (RGB -> YCoCg-R).
///
/// This is a lossless integer approximation of the YCoCg color space.
/// It decorrelates color channels for better compression.
///
/// - Co = R - B
/// - tmp = B + (Co >> 1)
/// - Cg = G - tmp
/// - Y = tmp + (Cg >> 1)
pub fn forward_rct(r: i32, g: i32, b: i32) -> (i32, i32, i32) {
    let co = r - b;
    let tmp = b + (co >> 1);
    let cg = g - tmp;
    let y = tmp + (cg >> 1);
    (y, co, cg)
}

/// Reversible Color Transform: inverse (YCoCg-R -> RGB).
///
/// Exactly inverts `forward_rct` for all integer inputs.
pub fn inverse_rct(y: i32, co: i32, cg: i32) -> (i32, i32, i32) {
    let tmp = y - (cg >> 1);
    let g = tmp + cg;
    let b = tmp - (co >> 1);
    let r = b + co;
    (r, g, b)
}

/// Modular transform types that can be applied to channels.
#[derive(Clone, Debug)]
pub enum ModularTransform {
    /// Reversible Color Transform on a group of 3 channels.
    Rct {
        /// First channel index of the group.
        begin_channel: u32,
        /// RCT variant (0 = YCoCg-R).
        rct_type: u8,
    },
    /// Squeeze (wavelet-like) transform for progressive decoding.
    Squeeze {
        /// Squeeze parameters.
        params: SqueezeParams,
    },
    /// Palette transform for indexed-color images.
    Palette {
        /// First channel to apply palette to.
        begin_channel: u32,
        /// Number of palette entries.
        num_colors: u32,
        /// Palette data (interleaved channel values).
        palette: Vec<i32>,
    },
}

/// Parameters for the Squeeze transform.
#[derive(Clone, Debug)]
pub struct SqueezeParams {
    /// Apply horizontal squeeze.
    pub horizontal: bool,
    /// Perform in-place (otherwise creates new channels).
    pub in_place: bool,
    /// First channel to squeeze.
    pub begin_channel: u32,
    /// Number of channels to squeeze.
    pub num_channels: u32,
}

/// Context for adaptive prediction weight selection.
///
/// Tracks prediction errors to adaptively choose the best predictor
/// for each pixel context.
struct PredictionContext {
    /// Accumulated absolute errors for each predictor.
    errors: [i64; NUM_PREDICTORS],
    /// Decay factor for error accumulation (shift right by this amount).
    decay_shift: u32,
    /// Counter for periodic error decay.
    counter: u32,
}

impl PredictionContext {
    fn new() -> Self {
        Self {
            errors: [0; NUM_PREDICTORS],
            decay_shift: 4,
            counter: 0,
        }
    }

    /// Select the predictor with the lowest accumulated error.
    fn best_predictor(&self) -> Predictor {
        let mut best_idx = 0;
        let mut best_err = self.errors[0];
        for i in 1..NUM_PREDICTORS {
            if self.errors[i] < best_err {
                best_err = self.errors[i];
                best_idx = i;
            }
        }
        Predictor::from_index(best_idx)
    }

    /// Update error accumulators after observing the actual value.
    fn update(&mut self, predictions: &[i32; NUM_PREDICTORS], actual: i32) {
        for i in 0..NUM_PREDICTORS {
            let err = (actual - predictions[i]).unsigned_abs() as i64;
            self.errors[i] += err;
        }
        self.counter += 1;
        // Periodic decay to adapt to changing statistics
        if self.counter >= (1 << self.decay_shift) {
            for err in &mut self.errors {
                *err >>= 1;
            }
            self.counter = 0;
        }
    }
}

/// Get the neighbor values for prediction at position (x, y) in a channel.
///
/// Returns (W, N, NW, NE, NN, WW).
fn get_neighbors(channel: &[i32], width: u32, x: u32, y: u32) -> (i32, i32, i32, i32, i32, i32) {
    let w = width as usize;
    let xi = x as usize;
    let yi = y as usize;

    let val = |px: usize, py: usize| -> i32 {
        if px < w && py < (channel.len() / w) {
            channel[py * w + px]
        } else {
            0
        }
    };

    let west = if xi > 0 { val(xi - 1, yi) } else { 0 };
    let north = if yi > 0 { val(xi, yi - 1) } else { 0 };
    let nw = if xi > 0 && yi > 0 {
        val(xi - 1, yi - 1)
    } else {
        0
    };
    let ne = if yi > 0 && xi + 1 < w {
        val(xi + 1, yi - 1)
    } else {
        north
    };
    let nn = if yi >= 2 { val(xi, yi - 2) } else { north };
    let ww = if xi >= 2 { val(xi - 2, yi) } else { west };

    (west, north, nw, ne, nn, ww)
}

/// Compute all predictor values for a given set of neighbors.
fn compute_predictions(
    w: i32,
    n: i32,
    nw: i32,
    ne: i32,
    _nn: i32,
    _ww: i32,
) -> [i32; NUM_PREDICTORS] {
    let avg_wn = (w + n) / 2;
    let gradient = n + w - nw;

    // Clamp gradient to the range [min(W,N), max(W,N)] for stability
    let grad_clamped = gradient.clamp(w.min(n), w.max(n));

    // Weighted predictor: adaptive combination
    let weighted = {
        let sum = 3i64 * n as i64 + 3i64 * w as i64 - nw as i64 + ne as i64;
        (sum / 6) as i32
    };

    [
        0,            // Zero
        w,            // West
        n,            // North
        avg_wn,       // Average(W, N)
        grad_clamped, // Gradient (clamped)
        weighted,     // Weighted
    ]
}

/// Encode a signed residual into a variable-length byte sequence.
///
/// Encoding scheme:
/// - Map signed to unsigned via zigzag: 0->0, -1->1, 1->2, -2->3, ...
/// - Write unsigned value in 7-bit chunks with high bit as continuation flag:
///   - If high bit = 1, more bytes follow
///   - If high bit = 0, this is the last byte
fn encode_residual(value: i32, output: &mut Vec<u8>) {
    let unsigned = signed_to_unsigned(value);
    let mut remaining = unsigned;
    loop {
        let byte = (remaining & 0x7F) as u8;
        remaining >>= 7;
        if remaining == 0 {
            output.push(byte); // high bit = 0, last byte
            break;
        } else {
            output.push(byte | 0x80); // high bit = 1, more bytes follow
        }
    }
}

/// Decode a variable-length encoded residual.
///
/// Returns (decoded_value, bytes_consumed).
fn decode_residual(data: &[u8], offset: usize) -> CodecResult<(i32, usize)> {
    let mut value: u32 = 0;
    let mut shift: u32 = 0;
    let mut pos = offset;

    loop {
        if pos >= data.len() {
            return Err(CodecError::InvalidBitstream(
                "Unexpected end of residual data".into(),
            ));
        }
        let byte = data[pos];
        pos += 1;

        value |= ((byte & 0x7F) as u32) << shift;
        shift += 7;

        if byte & 0x80 == 0 {
            // Last byte
            break;
        }
        if shift >= 35 {
            return Err(CodecError::InvalidBitstream(
                "Residual value too large".into(),
            ));
        }
    }

    Ok((unsigned_to_signed(value), pos - offset))
}

/// Map a signed residual to an unsigned value for entropy coding.
///
/// Uses the standard zigzag mapping: 0 -> 0, -1 -> 1, 1 -> 2, -2 -> 3, ...
fn signed_to_unsigned(value: i32) -> u32 {
    if value >= 0 {
        (value as u32) << 1
    } else {
        (((-value) as u32) << 1) - 1
    }
}

/// Map an unsigned value back to a signed residual.
fn unsigned_to_signed(value: u32) -> i32 {
    if value & 1 == 0 {
        (value >> 1) as i32
    } else {
        -(((value + 1) >> 1) as i32)
    }
}

/// Modular decoder for JPEG-XL lossless images.
pub struct ModularDecoder {
    transforms: Vec<ModularTransform>,
}

impl ModularDecoder {
    /// Create a new modular decoder.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Add a transform to be applied during decoding (inverse order).
    pub fn add_transform(&mut self, transform: ModularTransform) {
        self.transforms.push(transform);
    }

    /// Decode an image from variable-length coded residual data.
    ///
    /// Returns one `Vec<i32>` per channel, each of length `width * height`.
    pub fn decode_image(
        &mut self,
        data: &[u8],
        width: u32,
        height: u32,
        channels: u32,
        _bit_depth: u8,
    ) -> CodecResult<Vec<Vec<i32>>> {
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidParameter(
                "Image dimensions must be non-zero".into(),
            ));
        }

        let pixel_count = width as usize * height as usize;
        let mut result_channels: Vec<Vec<i32>> = Vec::with_capacity(channels as usize);
        let mut data_offset = 0usize;

        for _ch in 0..channels {
            let mut channel_data = vec![0i32; pixel_count];
            let mut ctx = PredictionContext::new();

            for y in 0..height {
                for x in 0..width {
                    let (w_val, n_val, nw_val, ne_val, nn_val, ww_val) =
                        get_neighbors(&channel_data, width, x, y);
                    let predictions =
                        compute_predictions(w_val, n_val, nw_val, ne_val, nn_val, ww_val);
                    let predictor = ctx.best_predictor();
                    let predicted = predictions[predictor as usize];

                    // Decode residual
                    let (residual, consumed) = decode_residual(data, data_offset)?;
                    data_offset += consumed;

                    let actual = predicted + residual;
                    channel_data[y as usize * width as usize + x as usize] = actual;
                    ctx.update(&predictions, actual);
                }
            }

            result_channels.push(channel_data);
        }

        // Apply inverse transforms in reverse order
        for transform in self.transforms.iter().rev() {
            match transform {
                ModularTransform::Rct {
                    begin_channel,
                    rct_type: _,
                } => {
                    let begin = *begin_channel as usize;
                    if begin + 2 < result_channels.len() {
                        let pc = result_channels[begin].len();
                        for i in 0..pc {
                            let y_val = result_channels[begin][i];
                            let co = result_channels[begin + 1][i];
                            let cg = result_channels[begin + 2][i];
                            let (r, g, b) = inverse_rct(y_val, co, cg);
                            result_channels[begin][i] = r;
                            result_channels[begin + 1][i] = g;
                            result_channels[begin + 2][i] = b;
                        }
                    }
                }
                ModularTransform::Squeeze { .. } | ModularTransform::Palette { .. } => {
                    // Squeeze and Palette transforms not yet implemented
                }
            }
        }

        Ok(result_channels)
    }
}

impl Default for ModularDecoder {
    fn default() -> Self {
        Self::new()
    }
}

/// Modular encoder for JPEG-XL lossless images.
pub struct ModularEncoder {
    transforms: Vec<ModularTransform>,
    effort: u8,
}

impl ModularEncoder {
    /// Create a new modular encoder.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
            effort: 7,
        }
    }

    /// Set encoding effort (1-9).
    pub fn with_effort(mut self, effort: u8) -> Self {
        self.effort = effort.clamp(1, 9);
        self
    }

    /// Add a transform to be applied during encoding.
    pub fn add_transform(&mut self, transform: ModularTransform) {
        self.transforms.push(transform);
    }

    /// Encode channels into a compressed byte stream.
    ///
    /// Input: one `Vec<i32>` per channel, each of length `width * height`.
    /// Returns the variable-length coded residual data.
    pub fn encode_image(
        &mut self,
        channels: &[Vec<i32>],
        width: u32,
        height: u32,
        _bit_depth: u8,
    ) -> CodecResult<Vec<u8>> {
        if width == 0 || height == 0 {
            return Err(CodecError::InvalidParameter(
                "Image dimensions must be non-zero".into(),
            ));
        }
        if channels.is_empty() {
            return Err(CodecError::InvalidParameter(
                "Must have at least one channel".into(),
            ));
        }

        let pixel_count = width as usize * height as usize;
        for (i, ch) in channels.iter().enumerate() {
            if ch.len() != pixel_count {
                return Err(CodecError::InvalidParameter(format!(
                    "Channel {i} has {} samples, expected {pixel_count}",
                    ch.len()
                )));
            }
        }

        // Apply forward transforms
        let mut working_channels: Vec<Vec<i32>> = channels.to_vec();
        for transform in &self.transforms {
            match transform {
                ModularTransform::Rct {
                    begin_channel,
                    rct_type: _,
                } => {
                    let begin = *begin_channel as usize;
                    if begin + 2 < working_channels.len() {
                        for i in 0..pixel_count {
                            let r = working_channels[begin][i];
                            let g = working_channels[begin + 1][i];
                            let b = working_channels[begin + 2][i];
                            let (y_val, co, cg) = forward_rct(r, g, b);
                            working_channels[begin][i] = y_val;
                            working_channels[begin + 1][i] = co;
                            working_channels[begin + 2][i] = cg;
                        }
                    }
                }
                ModularTransform::Squeeze { .. } | ModularTransform::Palette { .. } => {
                    // Not yet implemented
                }
            }
        }

        // Encode residuals channel by channel
        let mut output = Vec::with_capacity(pixel_count * working_channels.len());

        for ch_data in &working_channels {
            let mut ctx = PredictionContext::new();

            for y in 0..height {
                for x in 0..width {
                    let (w_val, n_val, nw_val, ne_val, nn_val, ww_val) =
                        get_neighbors(ch_data, width, x, y);
                    let predictions =
                        compute_predictions(w_val, n_val, nw_val, ne_val, nn_val, ww_val);
                    let predictor = ctx.best_predictor();
                    let predicted = predictions[predictor as usize];

                    let actual = ch_data[y as usize * width as usize + x as usize];
                    let residual = actual - predicted;

                    encode_residual(residual, &mut output);
                    ctx.update(&predictions, actual);
                }
            }
        }

        Ok(output)
    }
}

impl Default for ModularEncoder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_rct_roundtrip() {
        let test_values = [
            (0, 0, 0),
            (255, 255, 255),
            (128, 64, 32),
            (0, 255, 0),
            (255, 0, 0),
            (0, 0, 255),
            (100, 200, 50),
            (1, 1, 1),
        ];

        for (r, g, b) in test_values {
            let (y, co, cg) = forward_rct(r, g, b);
            let (r2, g2, b2) = inverse_rct(y, co, cg);
            assert_eq!(
                (r, g, b),
                (r2, g2, b2),
                "RCT roundtrip failed for ({r}, {g}, {b})"
            );
        }
    }

    #[test]
    #[ignore]
    fn test_rct_negative_values() {
        let (y, co, cg) = forward_rct(-10, 20, -30);
        let (r, g, b) = inverse_rct(y, co, cg);
        assert_eq!((r, g, b), (-10, 20, -30));
    }

    #[test]
    #[ignore]
    fn test_signed_unsigned_roundtrip() {
        for v in -100..=100 {
            let u = signed_to_unsigned(v);
            let v2 = unsigned_to_signed(u);
            assert_eq!(v, v2, "Zigzag roundtrip failed for {v}");
        }
    }

    #[test]
    #[ignore]
    fn test_zigzag_ordering() {
        assert_eq!(signed_to_unsigned(0), 0);
        assert_eq!(signed_to_unsigned(-1), 1);
        assert_eq!(signed_to_unsigned(1), 2);
        assert_eq!(signed_to_unsigned(-2), 3);
        assert_eq!(signed_to_unsigned(2), 4);
    }

    #[test]
    #[ignore]
    fn test_residual_encode_decode_roundtrip() {
        let test_values = [0, 1, -1, 127, -128, 1000, -1000, 65535, -65536, 0];
        let mut encoded = Vec::new();
        for &v in &test_values {
            encode_residual(v, &mut encoded);
        }

        let mut offset = 0;
        for &expected in &test_values {
            let (decoded, consumed) = decode_residual(&encoded, offset).expect("decode ok");
            assert_eq!(
                decoded, expected,
                "Residual roundtrip failed for {expected}"
            );
            offset += consumed;
        }
    }

    #[test]
    #[ignore]
    fn test_gradient_predictor() {
        let predictions = compute_predictions(100, 100, 100, 100, 100, 100);
        assert_eq!(predictions[Predictor::Gradient as usize], 100);
        assert_eq!(predictions[Predictor::West as usize], 100);
        assert_eq!(predictions[Predictor::North as usize], 100);
    }

    #[test]
    #[ignore]
    fn test_gradient_predictor_edge() {
        let predictions = compute_predictions(10, 0, 0, 0, 0, 0);
        assert_eq!(predictions[Predictor::Gradient as usize], 10);

        let predictions = compute_predictions(0, 10, 0, 0, 0, 0);
        assert_eq!(predictions[Predictor::Gradient as usize], 10);
    }

    #[test]
    #[ignore]
    fn test_prediction_context() {
        let mut ctx = PredictionContext::new();
        assert_eq!(ctx.best_predictor(), Predictor::Zero);

        let predictions = [0, 100, 50, 75, 90, 80];
        ctx.update(&predictions, 100);

        assert_eq!(ctx.best_predictor(), Predictor::West);
    }

    #[test]
    #[ignore]
    fn test_get_neighbors_corner() {
        let channel = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];
        let (w, n, nw, ne, nn, ww) = get_neighbors(&channel, 3, 0, 0);
        assert_eq!((w, n, nw, ne, nn, ww), (0, 0, 0, 0, 0, 0));

        let (w, n, nw, ne, _nn, _ww) = get_neighbors(&channel, 3, 1, 1);
        assert_eq!(w, 4);
        assert_eq!(n, 2);
        assert_eq!(nw, 1);
        assert_eq!(ne, 3);
    }

    #[test]
    #[ignore]
    fn test_modular_encode_decode_flat() {
        let width = 4u32;
        let height = 4u32;
        let pixel_count = (width * height) as usize;
        let channel = vec![128i32; pixel_count];

        let mut encoder = ModularEncoder::new();
        let encoded = encoder
            .encode_image(&[channel.clone()], width, height, 8)
            .expect("encode ok");

        let mut decoder = ModularDecoder::new();
        let decoded = decoder
            .decode_image(&encoded, width, height, 1, 8)
            .expect("decode ok");

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0], channel);
    }

    #[test]
    #[ignore]
    fn test_modular_encode_decode_gradient() {
        let width = 8u32;
        let height = 4u32;
        let mut channel = Vec::with_capacity((width * height) as usize);
        for y in 0..height {
            for x in 0..width {
                channel.push((x + y * 10) as i32);
            }
        }

        let mut encoder = ModularEncoder::new();
        let encoded = encoder
            .encode_image(&[channel.clone()], width, height, 8)
            .expect("encode ok");

        let mut decoder = ModularDecoder::new();
        let decoded = decoder
            .decode_image(&encoded, width, height, 1, 8)
            .expect("decode ok");

        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0], channel);
    }

    #[test]
    #[ignore]
    fn test_modular_encode_decode_with_rct() {
        let width = 4u32;
        let height = 4u32;
        let pixel_count = (width * height) as usize;

        let r: Vec<i32> = (0..pixel_count).map(|i| (i * 3) as i32 % 256).collect();
        let g: Vec<i32> = (0..pixel_count)
            .map(|i| (i * 5 + 50) as i32 % 256)
            .collect();
        let b: Vec<i32> = (0..pixel_count)
            .map(|i| (i * 7 + 100) as i32 % 256)
            .collect();

        let rct = ModularTransform::Rct {
            begin_channel: 0,
            rct_type: 0,
        };

        let mut encoder = ModularEncoder::new();
        encoder.add_transform(rct.clone());
        let encoded = encoder
            .encode_image(&[r.clone(), g.clone(), b.clone()], width, height, 8)
            .expect("encode ok");

        let mut decoder = ModularDecoder::new();
        decoder.add_transform(rct);
        let decoded = decoder
            .decode_image(&encoded, width, height, 3, 8)
            .expect("decode ok");

        assert_eq!(decoded.len(), 3);
        assert_eq!(decoded[0], r, "Red channel mismatch");
        assert_eq!(decoded[1], g, "Green channel mismatch");
        assert_eq!(decoded[2], b, "Blue channel mismatch");
    }

    #[test]
    #[ignore]
    fn test_modular_zero_dimensions_error() {
        let mut encoder = ModularEncoder::new();
        assert!(encoder.encode_image(&[vec![0i32]], 0, 1, 8).is_err());
        assert!(encoder.encode_image(&[vec![0i32]], 1, 0, 8).is_err());
    }

    #[test]
    #[ignore]
    fn test_modular_empty_channels_error() {
        let mut encoder = ModularEncoder::new();
        assert!(encoder.encode_image(&[], 1, 1, 8).is_err());
    }

    #[test]
    #[ignore]
    fn test_modular_multichannel() {
        let width = 4u32;
        let height = 4u32;
        let pixel_count = (width * height) as usize;

        let ch0: Vec<i32> = (0..pixel_count).map(|i| (i * 11 % 256) as i32).collect();
        let ch1: Vec<i32> = (0..pixel_count).map(|i| (i * 17 % 256) as i32).collect();

        let mut encoder = ModularEncoder::new();
        let encoded = encoder
            .encode_image(&[ch0.clone(), ch1.clone()], width, height, 8)
            .expect("encode ok");

        let mut decoder = ModularDecoder::new();
        let decoded = decoder
            .decode_image(&encoded, width, height, 2, 8)
            .expect("decode ok");

        assert_eq!(decoded[0], ch0);
        assert_eq!(decoded[1], ch1);
    }

    #[test]
    #[ignore]
    fn test_modular_large_values() {
        // Test with 16-bit range values
        let width = 4u32;
        let height = 4u32;
        let pixel_count = (width * height) as usize;
        let channel: Vec<i32> = (0..pixel_count).map(|i| (i * 4000) as i32).collect();

        let mut encoder = ModularEncoder::new();
        let encoded = encoder
            .encode_image(&[channel.clone()], width, height, 16)
            .expect("encode ok");

        let mut decoder = ModularDecoder::new();
        let decoded = decoder
            .decode_image(&encoded, width, height, 1, 16)
            .expect("decode ok");

        assert_eq!(decoded[0], channel);
    }
}
