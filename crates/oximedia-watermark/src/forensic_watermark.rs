//! Forensic watermarking for traitor tracing.
//!
//! A forensic watermark embeds a unique identifier (customer ID, session,
//! timestamp) into the content so that leaks can be traced back to the
//! responsible party.  This module provides payload encoding/decoding,
//! a DCT-domain embedding implementation using real 2D DCT-II/III transforms
//! with Quantization Index Modulation (QIM), and a simple traitor-tracing
//! structure.

// ── ForensicPayload ───────────────────────────────────────────────────────────

/// A 96-bit forensic payload identifying the recipient of a media asset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ForensicPayload {
    /// Customer (or user) identifier.
    pub customer_id: u32,
    /// Session identifier.
    pub session_id: u32,
    /// Unix timestamp (seconds since epoch) at embed time.
    pub timestamp_sec: u32,
}

impl ForensicPayload {
    /// Pack the payload into a 64-bit value.
    ///
    /// Layout (MSB first):
    /// - bits 63-32: `customer_id`
    /// - bits 31-16: lower 16 bits of `session_id`
    /// - bits 15-0 : lower 16 bits of `timestamp_sec`
    #[must_use]
    pub fn encode(&self) -> u64 {
        let cid = u64::from(self.customer_id);
        let sid = u64::from(self.session_id & 0xFFFF);
        let ts = u64::from(self.timestamp_sec & 0xFFFF);
        (cid << 32) | (sid << 16) | ts
    }

    /// Unpack a payload from a 64-bit value produced by `encode`.
    #[must_use]
    pub fn decode(v: u64) -> ForensicPayload {
        let customer_id = (v >> 32) as u32;
        let session_id = ((v >> 16) & 0xFFFF) as u32;
        let timestamp_sec = (v & 0xFFFF) as u32;
        ForensicPayload {
            customer_id,
            session_id,
            timestamp_sec,
        }
    }
}

// ── WatermarkVariant ──────────────────────────────────────────────────────────

/// A specific watermark variant assigned to a customer.
#[derive(Debug, Clone)]
pub struct WatermarkVariant {
    /// Variant index (used to select an embedding pattern).
    pub variant_id: u8,
    /// The forensic payload embedded in this variant.
    pub payload: ForensicPayload,
}

impl WatermarkVariant {
    /// Human-readable description of this variant.
    #[must_use]
    pub fn describe(&self) -> String {
        format!(
            "variant={} customer={} session={} ts={}",
            self.variant_id,
            self.payload.customer_id,
            self.payload.session_id,
            self.payload.timestamp_sec,
        )
    }
}

// ── DCT helpers ───────────────────────────────────────────────────────────────

/// Cosine scaling factor `C(k)` as defined in the JPEG/DCT-II standard.
/// `C(0) = 1/√2`, `C(k) = 1` for k > 0.
#[inline]
fn dct_scale(k: usize) -> f64 {
    if k == 0 {
        std::f64::consts::FRAC_1_SQRT_2
    } else {
        1.0
    }
}

/// Forward 2D DCT-II on an 8×8 block of `f64` values.
///
/// The input `block` is in row-major order (64 elements).
/// Output coefficients `F[u][v]` occupy the same row-major layout.
///
/// Formula:
/// `F[u][v] = (1/4) C(u) C(v)
///            Σ_{x=0}^{7} Σ_{y=0}^{7} f[x][y]
///            · cos((2x+1)uπ/16) · cos((2y+1)vπ/16)`
#[must_use]
fn dct2d_forward(block: &[f64; 64]) -> [f64; 64] {
    use std::f64::consts::PI;
    let mut out = [0.0_f64; 64];
    for u in 0..8_usize {
        for v in 0..8_usize {
            let cu = dct_scale(u);
            let cv = dct_scale(v);
            let mut sum = 0.0_f64;
            for x in 0..8_usize {
                let cos_u = ((2 * x + 1) as f64 * u as f64 * PI / 16.0).cos();
                for y in 0..8_usize {
                    let cos_v = ((2 * y + 1) as f64 * v as f64 * PI / 16.0).cos();
                    sum += block[x * 8 + y] * cos_u * cos_v;
                }
            }
            out[u * 8 + v] = 0.25 * cu * cv * sum;
        }
    }
    out
}

/// Inverse 2D DCT-III (reconstruction) on an 8×8 coefficient block.
///
/// Formula (inverse of DCT-II):
/// `f[x][y] = (1/4) Σ_{u=0}^{7} Σ_{v=0}^{7} C(u) C(v) F[u][v]
///            · cos((2x+1)uπ/16) · cos((2y+1)vπ/16)`
#[must_use]
fn dct2d_inverse(coeffs: &[f64; 64]) -> [f64; 64] {
    use std::f64::consts::PI;
    let mut out = [0.0_f64; 64];
    for x in 0..8_usize {
        for y in 0..8_usize {
            let mut sum = 0.0_f64;
            for u in 0..8_usize {
                let cu = dct_scale(u);
                let cos_u = ((2 * x + 1) as f64 * u as f64 * PI / 16.0).cos();
                for v in 0..8_usize {
                    let cv = dct_scale(v);
                    let cos_v = ((2 * y + 1) as f64 * v as f64 * PI / 16.0).cos();
                    sum += cu * cv * coeffs[u * 8 + v] * cos_u * cos_v;
                }
            }
            out[x * 8 + y] = 0.25 * sum;
        }
    }
    out
}

/// Mid-frequency DCT coefficient positions (u, v) used for QIM embedding.
///
/// These are zig-zag mid-band positions that avoid:
/// - DC component [0,0] (carries mean luminance — visually very sensitive)
/// - High-frequency noise zone (u+v >= 10 is too noisy)
/// The chosen positions are in the range where u+v ∈ [4, 8], giving a
/// good balance of robustness and imperceptibility.
const MID_FREQ_POSITIONS: [(usize, usize); 8] = [
    (2, 2), // u+v = 4
    (1, 3), // u+v = 4
    (3, 1), // u+v = 4
    (2, 3), // u+v = 5
    (3, 2), // u+v = 5
    (1, 4), // u+v = 5
    (4, 1), // u+v = 5
    (3, 3), // u+v = 6
];

/// QIM quantization step size for DCT coefficient embedding.
///
/// Must be large enough to survive u8 pixel round-tripping through the DCT.
/// For an 8×8 block, rounding each pixel by ±0.5 can perturb a mid-frequency
/// DCT coefficient by up to ~8.0 (sum of 64 basis products × 0.25 × 0.5).
/// The QIM decision boundary sits at `step/4` from the embedded value, so we
/// need `step/4 > 8`, i.e. `step > 32`.  We use 40.0 to provide comfortable
/// margin while keeping visible distortion acceptable for forensic use.
const QIM_STEP: f64 = 40.0;

/// Embed a single bit into a DCT coefficient array using QIM.
///
/// Quantization Index Modulation:
/// - bit `0` → round to nearest **even** multiple of `step/2`
/// - bit `1` → round to nearest **odd** multiple of `step/2`
///
/// This gives two interleaved quantization grids separated by `step/2`.
fn qim_embed_bit(coeff: f64, bit: bool, step: f64) -> f64 {
    let half = step / 2.0;
    // Index in the quantization lattice of spacing `half`.
    let index = (coeff / half).round() as i64;
    // We want even index for bit=0, odd for bit=1.
    let parity = index.rem_euclid(2) as i64;
    let target_parity = i64::from(bit);
    let adjusted_index = if parity == target_parity {
        index
    } else {
        // Shift by ±1 to fix parity — pick whichever is closer.
        let plus = index + 1;
        let minus = index - 1;
        let d_plus = ((plus as f64 * half) - coeff).abs();
        let d_minus = ((minus as f64 * half) - coeff).abs();
        if d_plus <= d_minus {
            plus
        } else {
            minus
        }
    };
    adjusted_index as f64 * half
}

/// Extract a single bit from a DCT coefficient using QIM decision.
///
/// The parity of `round(coeff / (step/2))` encodes the bit.
fn qim_extract_bit(coeff: f64, step: f64) -> bool {
    let half = step / 2.0;
    let index = (coeff / half).round() as i64;
    index.rem_euclid(2) == 1
}

// ── ForensicEmbedder ──────────────────────────────────────────────────────────

/// Embeds a forensic payload into pixel data using real DCT-domain QIM embedding.
///
/// The image is processed in non-overlapping `block_size × block_size` pixel
/// blocks (standard convention is 8×8).  For each selected block:
/// 1. Extract the luma 8×8 sub-block.
/// 2. Apply forward 2D DCT-II.
/// 3. Embed one payload bit via QIM into a mid-frequency coefficient.
/// 4. Apply inverse DCT-III.
/// 5. Write the reconstructed pixels back, clamped to `[0, 255]`.
pub struct ForensicEmbedder {
    /// Block size for DCT processing (8 for 8×8 DCT).
    pub block_size: usize,
    /// Embedding strength scalar applied to the QIM step size.
    /// Range: 0.0–1.0 mapped to 0.25×–4.0× the base step.
    pub strength: f32,
}

impl ForensicEmbedder {
    /// Create a new embedder.
    ///
    /// `block_size` is clamped to at least 8 (required for 8×8 DCT).
    /// `strength` is clamped to `[0.0, 1.0]`.
    #[must_use]
    pub fn new(block_size: usize, strength: f32) -> Self {
        Self {
            block_size: block_size.max(8),
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Embed `payload` into `pixels` using real DCT-domain QIM.
    ///
    /// # Arguments
    /// * `pixels`  – mutable flat 8-bit luma buffer (`width * height` bytes).
    /// * `width`   – image width in pixels.
    /// * `height`  – image height in pixels.
    /// * `payload` – the forensic payload to embed.
    pub fn embed(&self, pixels: &mut [u8], width: usize, height: usize, payload: &ForensicPayload) {
        let bits = payload_to_bits(payload.encode());
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;

        // We need at least 64 blocks to embed the 64-bit payload.
        if blocks_x == 0 || blocks_y == 0 {
            return;
        }

        // Scale the QIM step by strength: strength=0.5 → 1× base step.
        let step = QIM_STEP * (0.5 + f64::from(self.strength) * 1.5);

        for (bit_idx, &bit) in bits.iter().enumerate() {
            let block_col = bit_idx % blocks_x;
            let block_row = bit_idx / blocks_x;
            if block_row >= blocks_y {
                break;
            }
            let origin_x = block_col * bs;
            let origin_y = block_row * bs;

            // Extract the top-left 8×8 sub-region of the block into a flat array.
            let mut dct_in = [0.0_f64; 64];
            for r in 0..8_usize {
                for c in 0..8_usize {
                    let px = origin_x + c;
                    let py = origin_y + r;
                    if px < width && py < height {
                        let idx = py * width + px;
                        if idx < pixels.len() {
                            dct_in[r * 8 + c] = f64::from(pixels[idx]);
                        }
                    }
                }
            }

            // Forward DCT-II.
            let mut coeffs = dct2d_forward(&dct_in);

            // Embed the bit into one mid-frequency coefficient (cycle through positions).
            let pos = MID_FREQ_POSITIONS[bit_idx % MID_FREQ_POSITIONS.len()];
            let coeff_idx = pos.0 * 8 + pos.1;
            coeffs[coeff_idx] = qim_embed_bit(coeffs[coeff_idx], bit, step);

            // Inverse DCT-III.
            let reconstructed = dct2d_inverse(&coeffs);

            // Write back, clamped to [0, 255].
            for r in 0..8_usize {
                for c in 0..8_usize {
                    let px = origin_x + c;
                    let py = origin_y + r;
                    if px < width && py < height {
                        let idx = py * width + px;
                        if idx < pixels.len() {
                            let val = reconstructed[r * 8 + c].round();
                            pixels[idx] = val.clamp(0.0, 255.0) as u8;
                        }
                    }
                }
            }
        }
    }
}

// ── ForensicDetector ─────────────────────────────────────────────────────────

/// Extracts a forensic payload from pixel data using DCT-domain QIM detection.
pub struct ForensicDetector {
    /// Block size (must match the embedder).
    pub block_size: usize,
    /// Embedding strength (must match the embedder).
    pub strength: f32,
}

impl ForensicDetector {
    /// Create a new detector.
    ///
    /// `block_size` and `strength` must match the values used during embedding.
    #[must_use]
    pub fn new(block_size: usize) -> Self {
        Self {
            block_size: block_size.max(8),
            strength: 0.5,
        }
    }

    /// Create a new detector with an explicit strength matching the embedder.
    #[must_use]
    pub fn with_strength(block_size: usize, strength: f32) -> Self {
        Self {
            block_size: block_size.max(8),
            strength: strength.clamp(0.0, 1.0),
        }
    }

    /// Attempt to extract a `ForensicPayload` from `pixels`.
    ///
    /// Returns `None` if the image is too small to hold a payload or if the
    /// extracted data appears invalid (all-zero customer ID *and* session ID).
    #[must_use]
    pub fn detect(&self, pixels: &[u8], width: usize, height: usize) -> Option<ForensicPayload> {
        let bs = self.block_size;
        let blocks_x = width / bs;
        let blocks_y = height / bs;
        let total_blocks = blocks_x * blocks_y;

        // We need at least 64 blocks to read the full 64-bit payload.
        if total_blocks < 64 {
            return None;
        }

        let step = QIM_STEP * (0.5 + f64::from(self.strength) * 1.5);

        let mut bits = [false; 64];
        for i in 0..64_usize {
            let block_col = i % blocks_x;
            let block_row = i / blocks_x;
            let origin_x = block_col * bs;
            let origin_y = block_row * bs;

            // Extract 8×8 sub-block.
            let mut dct_in = [0.0_f64; 64];
            for r in 0..8_usize {
                for c in 0..8_usize {
                    let px = origin_x + c;
                    let py = origin_y + r;
                    if px < width && py < height {
                        let idx = py * width + px;
                        if idx < pixels.len() {
                            dct_in[r * 8 + c] = f64::from(pixels[idx]);
                        }
                    }
                }
            }

            // Forward DCT-II.
            let coeffs = dct2d_forward(&dct_in);

            // Read QIM decision from the same mid-frequency coefficient.
            let pos = MID_FREQ_POSITIONS[i % MID_FREQ_POSITIONS.len()];
            let coeff_idx = pos.0 * 8 + pos.1;
            bits[i] = qim_extract_bit(coeffs[coeff_idx], step);
        }

        let value = bits_to_u64(&bits);
        if value == 0 {
            return None;
        }
        Some(ForensicPayload::decode(value))
    }
}

// ── TraitorTrace ──────────────────────────────────────────────────────────────

/// A set of traitor tracing suspects with confidence scores.
#[derive(Debug, Clone, Default)]
pub struct TraitorTrace {
    /// (`customer_id`, confidence) pairs, unsorted.
    pub suspects: Vec<(u32, f32)>,
}

impl TraitorTrace {
    /// Create an empty traitor trace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a suspect.
    pub fn add_suspect(&mut self, customer_id: u32, confidence: f32) {
        self.suspects.push((customer_id, confidence));
    }

    /// Return the customer ID of the most likely suspect (highest confidence),
    /// or `None` if no suspects have been added.
    #[must_use]
    pub fn most_likely(&self) -> Option<u32> {
        self.suspects
            .iter()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|&(id, _)| id)
    }
}

// ── helpers ───────────────────────────────────────────────────────────────────

/// Convert a `u64` to a 64-element array of bits (MSB first).
fn payload_to_bits(v: u64) -> [bool; 64] {
    let mut bits = [false; 64];
    for i in 0..64 {
        bits[i] = ((v >> (63 - i)) & 1) == 1;
    }
    bits
}

/// Convert a 64-element bit array (MSB first) back to a `u64`.
fn bits_to_u64(bits: &[bool; 64]) -> u64 {
    let mut v = 0_u64;
    for (i, &b) in bits.iter().enumerate() {
        if b {
            v |= 1 << (63 - i);
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ForensicPayload ───────────────────────────────────────────────────────

    #[test]
    fn test_encode_decode_roundtrip() {
        let p = ForensicPayload {
            customer_id: 12345,
            session_id: 678,
            timestamp_sec: 910,
        };
        let encoded = p.encode();
        let decoded = ForensicPayload::decode(encoded);
        assert_eq!(decoded.customer_id, p.customer_id);
        // session_id and timestamp_sec are truncated to 16 bits
        assert_eq!(decoded.session_id, p.session_id & 0xFFFF);
        assert_eq!(decoded.timestamp_sec, p.timestamp_sec & 0xFFFF);
    }

    #[test]
    fn test_encode_zero_payload() {
        let p = ForensicPayload {
            customer_id: 0,
            session_id: 0,
            timestamp_sec: 0,
        };
        assert_eq!(p.encode(), 0);
    }

    #[test]
    fn test_encode_max_customer_id() {
        let p = ForensicPayload {
            customer_id: u32::MAX,
            session_id: 0,
            timestamp_sec: 0,
        };
        let enc = p.encode();
        assert_eq!(ForensicPayload::decode(enc).customer_id, u32::MAX);
    }

    #[test]
    fn test_encode_session_truncated_to_16_bits() {
        let p = ForensicPayload {
            customer_id: 1,
            session_id: 0x1_2345,
            timestamp_sec: 0,
        };
        let decoded = ForensicPayload::decode(p.encode());
        assert_eq!(decoded.session_id, 0x2345);
    }

    #[test]
    fn test_decode_known_value() {
        // customer_id=1 in bits 63-32, session_id=2 in bits 31-16, ts=3 in bits 15-0
        let v: u64 = (1_u64 << 32) | (2_u64 << 16) | 3;
        let p = ForensicPayload::decode(v);
        assert_eq!(p.customer_id, 1);
        assert_eq!(p.session_id, 2);
        assert_eq!(p.timestamp_sec, 3);
    }

    // ── WatermarkVariant ──────────────────────────────────────────────────────

    #[test]
    fn test_variant_describe_contains_customer_id() {
        let v = WatermarkVariant {
            variant_id: 3,
            payload: ForensicPayload {
                customer_id: 99,
                session_id: 1,
                timestamp_sec: 0,
            },
        };
        assert!(v.describe().contains("99"));
    }

    #[test]
    fn test_variant_describe_contains_variant_id() {
        let v = WatermarkVariant {
            variant_id: 7,
            payload: ForensicPayload {
                customer_id: 1,
                session_id: 1,
                timestamp_sec: 0,
            },
        };
        assert!(v.describe().contains("variant=7"));
    }

    // ── TraitorTrace ──────────────────────────────────────────────────────────

    #[test]
    fn test_most_likely_empty() {
        let trace = TraitorTrace::new();
        assert!(trace.most_likely().is_none());
    }

    #[test]
    fn test_most_likely_single() {
        let mut trace = TraitorTrace::new();
        trace.add_suspect(42, 0.9);
        assert_eq!(trace.most_likely(), Some(42));
    }

    #[test]
    fn test_most_likely_highest_confidence() {
        let mut trace = TraitorTrace::new();
        trace.add_suspect(1, 0.4);
        trace.add_suspect(2, 0.9);
        trace.add_suspect(3, 0.7);
        assert_eq!(trace.most_likely(), Some(2));
    }

    #[test]
    fn test_suspect_count() {
        let mut trace = TraitorTrace::new();
        trace.add_suspect(10, 0.5);
        trace.add_suspect(20, 0.6);
        assert_eq!(trace.suspects.len(), 2);
    }

    // ── DCT helpers ───────────────────────────────────────────────────────────

    #[test]
    fn test_dct_forward_inverse_roundtrip() {
        // A flat block should survive a forward+inverse DCT cycle.
        let mut block = [128.0_f64; 64];
        // Add some variety.
        for i in 0..64 {
            block[i] = (i as f64 * 3.7).sin() * 50.0 + 128.0;
        }
        let coeffs = dct2d_forward(&block);
        let reconstructed = dct2d_inverse(&coeffs);
        for (orig, rec) in block.iter().zip(reconstructed.iter()) {
            assert!(
                (orig - rec).abs() < 1e-6,
                "DCT roundtrip error: {} vs {}",
                orig,
                rec
            );
        }
    }

    #[test]
    fn test_qim_embed_extract_roundtrip_true() {
        let coeff = 37.5_f64;
        let embedded = qim_embed_bit(coeff, true, QIM_STEP);
        assert!(qim_extract_bit(embedded, QIM_STEP));
    }

    #[test]
    fn test_qim_embed_extract_roundtrip_false() {
        let coeff = 37.5_f64;
        let embedded = qim_embed_bit(coeff, false, QIM_STEP);
        assert!(!qim_extract_bit(embedded, QIM_STEP));
    }

    #[test]
    fn test_qim_distortion_bounded() {
        // Distortion should be at most step/2.
        let step = QIM_STEP;
        for i in 0..100_i32 {
            let coeff = i as f64 * 1.73 - 50.0;
            for &bit in &[true, false] {
                let embedded = qim_embed_bit(coeff, bit, step);
                let distortion = (embedded - coeff).abs();
                assert!(
                    distortion <= step / 2.0 + 1e-9,
                    "distortion {} exceeds step/2 for coeff={} bit={}",
                    distortion,
                    coeff,
                    bit
                );
            }
        }
    }

    // ── helpers ───────────────────────────────────────────────────────────────

    #[test]
    fn test_payload_to_bits_msb_first() {
        let bits = payload_to_bits(1_u64 << 63);
        assert!(bits[0]);
        assert!(!bits[1]);
    }

    #[test]
    fn test_bits_to_u64_roundtrip() {
        let original: u64 = 0xDEAD_BEEF_CAFE_1234;
        let bits = payload_to_bits(original);
        let recovered = bits_to_u64(&bits);
        assert_eq!(recovered, original);
    }

    // ── ForensicEmbedder / ForensicDetector ───────────────────────────────────

    #[test]
    fn test_embedder_does_not_panic_small_image() {
        let mut pixels = vec![128_u8; 4 * 4];
        let embedder = ForensicEmbedder::new(8, 0.5);
        let p = ForensicPayload {
            customer_id: 1,
            session_id: 2,
            timestamp_sec: 3,
        };
        // Image smaller than one block — should not panic, should be a no-op.
        embedder.embed(&mut pixels, 4, 4, &p);
    }

    #[test]
    fn test_detector_returns_none_for_tiny_image() {
        let pixels = vec![0_u8; 4 * 4];
        let det = ForensicDetector::new(8);
        assert!(det.detect(&pixels, 4, 4).is_none());
    }

    #[test]
    fn test_embedder_modifies_pixels() {
        // Image large enough for 64 blocks of 8×8: 8 blocks × 8 per row = 64px wide,
        // 8 blocks tall = 64px, but we need 64 blocks total so 8×8 grid = 64×64.
        let size = 64_usize;
        let original = vec![128_u8; size * size];
        let mut pixels = original.clone();
        let embedder = ForensicEmbedder::new(8, 0.5);
        let p = ForensicPayload {
            customer_id: 0xABCD,
            session_id: 0x1234,
            timestamp_sec: 100,
        };
        embedder.embed(&mut pixels, size, size, &p);
        assert_ne!(pixels, original, "embed should modify at least one pixel");
    }

    #[test]
    fn test_embed_extract_roundtrip() {
        // 8×8 grid of 8×8 blocks = 64×64 luma image.
        let size = 64_usize;
        let mut pixels: Vec<u8> = (0..size * size)
            .map(|i| ((i as f64 * 1.234).sin() * 50.0 + 128.0) as u8)
            .collect();
        let payload = ForensicPayload {
            customer_id: 42,
            session_id: 7,
            timestamp_sec: 999,
        };

        let strength = 0.5;
        let embedder = ForensicEmbedder::new(8, strength);
        embedder.embed(&mut pixels, size, size, &payload);

        let detector = ForensicDetector::with_strength(8, strength);
        let extracted = detector
            .detect(&pixels, size, size)
            .expect("payload should be detectable after embedding");

        assert_eq!(extracted.customer_id, payload.customer_id);
        assert_eq!(extracted.session_id, payload.session_id & 0xFFFF);
        assert_eq!(extracted.timestamp_sec, payload.timestamp_sec & 0xFFFF);
    }

    #[test]
    fn test_embed_extract_high_strength() {
        let size = 64_usize;
        let mut pixels = vec![200_u8; size * size];
        let payload = ForensicPayload {
            customer_id: 0xDEAD,
            session_id: 0xBEEF,
            timestamp_sec: 0x1234,
        };

        let strength = 1.0;
        let embedder = ForensicEmbedder::new(8, strength);
        embedder.embed(&mut pixels, size, size, &payload);

        let detector = ForensicDetector::with_strength(8, strength);
        let extracted = detector
            .detect(&pixels, size, size)
            .expect("payload should be detectable");

        assert_eq!(extracted.customer_id, payload.customer_id);
    }

    #[test]
    fn test_pixels_clamped_to_valid_range() {
        // Use extreme pixel values to test clamping.
        let size = 64_usize;
        let mut pixels = vec![255_u8; size * size];
        let payload = ForensicPayload {
            customer_id: 1,
            session_id: 1,
            timestamp_sec: 1,
        };
        let embedder = ForensicEmbedder::new(8, 1.0);
        embedder.embed(&mut pixels, size, size, &payload);
        // All values must remain in [0, 255] — enforced by u8.
        // Just verify no panic occurred.
        assert_eq!(pixels.len(), size * size);
    }
}
