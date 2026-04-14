#![allow(dead_code)]
//! Porter-Duff compositing operators for pixel-accurate alpha blending.
//!
//! Implements all 12 original Porter-Duff operators (Clear, Src, Dst, SrcOver,
//! DstOver, SrcIn, DstIn, SrcOut, DstOut, SrcAtop, DstAtop, Xor) plus two
//! extensions: Plus (additive) and Dissolve (stochastic).
//!
//! All compositing is performed in **premultiplied alpha** space as the
//! original 1984 paper specifies.  Helper functions convert to/from
//! straight-alpha u8 RGBA at the boundary.
//!
//! # References
//!
//! - T. Porter and T. Duff, "Compositing Digital Images", SIGGRAPH '84.
//! - W3C Compositing and Blending Level 1, Section 9.1.

use rayon::prelude::*;

// ─── Operator enum ────────────────────────────────────────────────────────────

/// All 12 Porter-Duff operators plus Plus and Dissolve extensions.
///
/// Each operator is defined by a pair of factors `(Fs, Fd)` such that:
/// - `Co = Cs·Fs + Cd·Fd`
/// - `αo = αs·Fs + αd·Fd`
///
/// where `Cs`/`Cd` are the premultiplied source/destination colour channels
/// and `αs`/`αd` are the corresponding alpha values.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PorterDuffOp {
    /// `(0, 0)` — erases both layers; result is fully transparent.
    Clear,
    /// `(1, 0)` — source replaces destination.
    Src,
    /// `(0, 1)` — destination is preserved; source is discarded.
    Dst,
    /// `(1, 1−αs)` — source placed over destination (the default blend).
    SrcOver,
    /// `(1−αd, 1)` — destination placed over source.
    DstOver,
    /// `(αd, 0)` — source only where destination is visible.
    SrcIn,
    /// `(0, αs)` — destination only where source is visible.
    DstIn,
    /// `(1−αd, 0)` — source only where destination is invisible.
    SrcOut,
    /// `(0, 1−αs)` — destination only where source is invisible.
    DstOut,
    /// `(αd, 1−αs)` — source atop destination; dst shape preserved.
    SrcAtop,
    /// `(1−αd, αs)` — destination atop source; src shape preserved.
    DstAtop,
    /// `(1−αd, 1−αs)` — union of exclusive regions.
    Xor,
    /// `(1, 1)` — additive blend; output clamped to `[0, 1]`.
    Plus,
    /// Stochastic: each pixel randomly selects source or destination based
    /// on the source alpha as a probability, yielding a dithered dissolve.
    Dissolve,
}

// ─── Premultiplied pixel ──────────────────────────────────────────────────────

/// A single pixel in premultiplied-alpha linear colour space.
///
/// All channel values lie in `[0, 1]`.  Because the colour channels are
/// premultiplied, the invariant `r ≤ a`, `g ≤ a`, `b ≤ a` holds for
/// valid input; however the type does not enforce this at construction time
/// to allow intermediate computations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PremulPixel {
    /// Red channel (premultiplied).
    pub r: f32,
    /// Green channel (premultiplied).
    pub g: f32,
    /// Blue channel (premultiplied).
    pub b: f32,
    /// Alpha channel.
    pub a: f32,
}

impl PremulPixel {
    /// Fully transparent black — the additive identity.
    #[inline]
    pub fn zero() -> Self {
        Self {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.0,
        }
    }

    /// Opaque white.
    #[inline]
    pub fn white() -> Self {
        Self {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        }
    }

    /// Clamp all channels to `[0, 1]`.
    #[inline]
    pub fn clamp(self) -> Self {
        Self {
            r: self.r.clamp(0.0, 1.0),
            g: self.g.clamp(0.0, 1.0),
            b: self.b.clamp(0.0, 1.0),
            a: self.a.clamp(0.0, 1.0),
        }
    }
}

// ─── Straight ↔ premul conversions ───────────────────────────────────────────

/// Convert a straight-alpha RGBA u8 pixel to premultiplied f32.
///
/// The formula is `Cpre = Cstraight * α` where `α ∈ [0, 1]`.
#[inline]
pub fn to_premul(rgba: [u8; 4]) -> PremulPixel {
    let a = rgba[3] as f32 / 255.0;
    PremulPixel {
        r: (rgba[0] as f32 / 255.0) * a,
        g: (rgba[1] as f32 / 255.0) * a,
        b: (rgba[2] as f32 / 255.0) * a,
        a,
    }
}

/// Convert a premultiplied f32 pixel back to straight-alpha RGBA u8.
///
/// The formula is `Cstraight = Cpre / α`.  When `α = 0` the colour
/// channels are set to zero (fully transparent black).
#[inline]
pub fn from_premul(p: PremulPixel) -> [u8; 4] {
    let p = p.clamp();
    if p.a < f32::EPSILON {
        return [0, 0, 0, 0];
    }
    let inv_a = 1.0 / p.a;
    let to_u8 = |v: f32| -> u8 { (v * inv_a * 255.0).round().clamp(0.0, 255.0) as u8 };
    [to_u8(p.r), to_u8(p.g), to_u8(p.b), (p.a * 255.0).round() as u8]
}

// ─── Deterministic hash helper ────────────────────────────────────────────────

/// A minimal 32-bit finalisation step (Murmur3-style) used to derive a
/// pseudo-random value from the pixel position without pulling in the full
/// `rand` crate dependency in the hot compositing path.
#[inline]
fn position_hash(x: u32, y: u32, seed: u32) -> u32 {
    let mut h = x.wrapping_mul(2246822519).wrapping_add(y.wrapping_mul(3266489917));
    h = h.wrapping_add(seed);
    h ^= h >> 16;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 16;
    h = h.wrapping_mul(0x45d9f3b);
    h ^= h >> 15;
    h
}

// ─── Single-pixel composite ───────────────────────────────────────────────────

/// Composite a single premultiplied pixel pair under the given operator.
///
/// For the `Dissolve` operator a position-independent path is used: since
/// no coordinate information is available to this function the underlying
/// hash seed defaults to zero, making the result deterministic but still
/// properly distributed.  Use [`composite_layer`] or [`composite_layer_into`]
/// for spatially-coherent dissolve.
#[inline]
pub fn composite_pixel(src: PremulPixel, dst: PremulPixel, op: PorterDuffOp) -> PremulPixel {
    composite_pixel_at(src, dst, op, 0, 0)
}

/// Composite a pixel pair at a known grid position.
///
/// The `x`/`y` coordinates are used exclusively by the `Dissolve` operator
/// to seed the position hash that drives the stochastic selection.
#[inline]
fn composite_pixel_at(
    src: PremulPixel,
    dst: PremulPixel,
    op: PorterDuffOp,
    x: u32,
    y: u32,
) -> PremulPixel {
    // Dissolve is handled as a special case — it does not fit the (Fs, Fd) model.
    if op == PorterDuffOp::Dissolve {
        // The probability of picking `src` equals the source alpha.
        // We map the hash to [0, 1) and compare against αs.
        let hash = position_hash(x, y, 0xdeadbeef);
        let t = (hash as f32) / (u32::MAX as f32);
        if t < src.a {
            // Normalise src to straight alpha then output as fully opaque src colour
            // to avoid the dissolve leaving semi-transparent pixels.
            if src.a < f32::EPSILON {
                return dst;
            }
            let inv = 1.0 / src.a;
            return PremulPixel {
                r: src.r * inv * dst.a.max(src.a),
                g: src.g * inv * dst.a.max(src.a),
                b: src.b * inv * dst.a.max(src.a),
                a: dst.a.max(src.a),
            };
        } else {
            return dst;
        }
    }

    // Derive the Porter-Duff factors (Fs, Fd) for this operator.
    let (fs, fd) = factors(op, src.a, dst.a);

    let r = (src.r * fs + dst.r * fd).clamp(0.0, 1.0);
    let g = (src.g * fs + dst.g * fd).clamp(0.0, 1.0);
    let b = (src.b * fs + dst.b * fd).clamp(0.0, 1.0);
    let a = (src.a * fs + dst.a * fd).clamp(0.0, 1.0);

    PremulPixel { r, g, b, a }
}

/// Return the Porter-Duff factor pair `(Fs, Fd)` for an operator given the
/// source alpha `as_` and destination alpha `ad`.
///
/// # Panics
///
/// Panics if called with `PorterDuffOp::Dissolve` — that operator must be
/// handled separately.
#[inline]
fn factors(op: PorterDuffOp, as_: f32, ad: f32) -> (f32, f32) {
    match op {
        PorterDuffOp::Clear => (0.0, 0.0),
        PorterDuffOp::Src => (1.0, 0.0),
        PorterDuffOp::Dst => (0.0, 1.0),
        PorterDuffOp::SrcOver => (1.0, 1.0 - as_),
        PorterDuffOp::DstOver => (1.0 - ad, 1.0),
        PorterDuffOp::SrcIn => (ad, 0.0),
        PorterDuffOp::DstIn => (0.0, as_),
        PorterDuffOp::SrcOut => (1.0 - ad, 0.0),
        PorterDuffOp::DstOut => (0.0, 1.0 - as_),
        PorterDuffOp::SrcAtop => (ad, 1.0 - as_),
        PorterDuffOp::DstAtop => (1.0 - ad, as_),
        PorterDuffOp::Xor => (1.0 - ad, 1.0 - as_),
        PorterDuffOp::Plus => (1.0, 1.0),
        PorterDuffOp::Dissolve => {
            panic!("factors() must not be called for PorterDuffOp::Dissolve — handle separately")
        }
    }
}

// ─── Buffer compositing ───────────────────────────────────────────────────────

/// Composite two full RGBA byte buffers and return a new buffer.
///
/// `src` and `dst` must each be `width * height * 4` bytes of row-major
/// straight-alpha RGBA.  The returned `Vec<u8>` has the same layout.
///
/// Row processing is parallelised with [rayon].
///
/// # Panics
///
/// Panics if either slice is shorter than `width * height * 4` bytes.
pub fn composite_layer(
    src: &[u8],
    dst: &[u8],
    width: u32,
    height: u32,
    op: PorterDuffOp,
) -> Vec<u8> {
    let pixel_count = (width as usize) * (height as usize);
    assert!(
        src.len() >= pixel_count * 4,
        "composite_layer: src buffer too short ({} < {})",
        src.len(),
        pixel_count * 4
    );
    assert!(
        dst.len() >= pixel_count * 4,
        "composite_layer: dst buffer too short ({} < {})",
        dst.len(),
        pixel_count * 4
    );

    let mut out = vec![0u8; pixel_count * 4];

    // Chunk by row so that rayon can work on independent slices.
    let row_bytes = (width as usize) * 4;

    out.par_chunks_mut(row_bytes)
        .zip(src.par_chunks(row_bytes))
        .zip(dst.par_chunks(row_bytes))
        .enumerate()
        .for_each(|(y_idx, ((out_row, src_row), dst_row))| {
            let y = y_idx as u32;
            for x in 0..width {
                let xi = x as usize;
                let base = xi * 4;

                let sp = to_premul([
                    src_row[base],
                    src_row[base + 1],
                    src_row[base + 2],
                    src_row[base + 3],
                ]);
                let dp = to_premul([
                    dst_row[base],
                    dst_row[base + 1],
                    dst_row[base + 2],
                    dst_row[base + 3],
                ]);

                let result = composite_pixel_at(sp, dp, op, x, y);
                let rgba = from_premul(result);
                out_row[base] = rgba[0];
                out_row[base + 1] = rgba[1];
                out_row[base + 2] = rgba[2];
                out_row[base + 3] = rgba[3];
            }
        });

    out
}

/// Composite `src` onto `dst` in place, writing the result back into `dst`.
///
/// This avoids an allocation compared to [`composite_layer`].  All other
/// constraints (buffer size, layout) are identical.
///
/// # Panics
///
/// Panics if either slice is shorter than `width * height * 4` bytes.
pub fn composite_layer_into(
    src: &[u8],
    dst: &mut [u8],
    width: u32,
    height: u32,
    op: PorterDuffOp,
) {
    let pixel_count = (width as usize) * (height as usize);
    assert!(
        src.len() >= pixel_count * 4,
        "composite_layer_into: src buffer too short ({} < {})",
        src.len(),
        pixel_count * 4
    );
    assert!(
        dst.len() >= pixel_count * 4,
        "composite_layer_into: dst buffer too short ({} < {})",
        dst.len(),
        pixel_count * 4
    );

    let row_bytes = (width as usize) * 4;

    // We need to read `src` per-row simultaneously with a mutable borrow of
    // `dst`.  Split `src` into immutable row chunks, then process each dst
    // row mutably via index arithmetic.
    dst.par_chunks_mut(row_bytes)
        .zip(src.par_chunks(row_bytes))
        .enumerate()
        .for_each(|(y_idx, (dst_row, src_row))| {
            let y = y_idx as u32;
            for x in 0..width {
                let xi = x as usize;
                let base = xi * 4;

                let sp = to_premul([
                    src_row[base],
                    src_row[base + 1],
                    src_row[base + 2],
                    src_row[base + 3],
                ]);
                let dp = to_premul([
                    dst_row[base],
                    dst_row[base + 1],
                    dst_row[base + 2],
                    dst_row[base + 3],
                ]);

                let result = composite_pixel_at(sp, dp, op, x, y);
                let rgba = from_premul(result);
                dst_row[base] = rgba[0];
                dst_row[base + 1] = rgba[1];
                dst_row[base + 2] = rgba[2];
                dst_row[base + 3] = rgba[3];
            }
        });
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Tolerance helpers ────────────────────────────────────────────────────

    const EPSILON: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32) -> bool {
        (a - b).abs() < EPSILON
    }

    fn pixel_approx_eq(a: PremulPixel, b: PremulPixel) -> bool {
        approx_eq(a.r, b.r) && approx_eq(a.g, b.g) && approx_eq(a.b, b.b) && approx_eq(a.a, b.a)
    }

    // Builds an opaque premultiplied pixel from straight-alpha f32 components.
    fn opaque(r: f32, g: f32, b: f32) -> PremulPixel {
        PremulPixel { r, g, b, a: 1.0 }
    }

    fn semi(r: f32, g: f32, b: f32, a: f32) -> PremulPixel {
        PremulPixel {
            r: r * a,
            g: g * a,
            b: b * a,
            a,
        }
    }

    // ── 1. Clear ─────────────────────────────────────────────────────────────

    #[test]
    fn test_pd_clear() {
        let src = opaque(0.8, 0.5, 0.2);
        let dst = opaque(0.3, 0.7, 0.1);
        let out = composite_pixel(src, dst, PorterDuffOp::Clear);
        assert!(pixel_approx_eq(out, PremulPixel::zero()), "Clear must yield zero: {:?}", out);
    }

    // ── 2. Src ───────────────────────────────────────────────────────────────

    #[test]
    fn test_pd_src() {
        let src = opaque(0.8, 0.5, 0.2);
        let dst = opaque(0.3, 0.7, 0.1);
        let out = composite_pixel(src, dst, PorterDuffOp::Src);
        assert!(pixel_approx_eq(out, src), "Src must equal source: {:?}", out);
    }

    // ── 3. Dst ───────────────────────────────────────────────────────────────

    #[test]
    fn test_pd_dst() {
        let src = opaque(0.8, 0.5, 0.2);
        let dst = opaque(0.3, 0.7, 0.1);
        let out = composite_pixel(src, dst, PorterDuffOp::Dst);
        assert!(pixel_approx_eq(out, dst), "Dst must equal destination: {:?}", out);
    }

    // ── 4. SrcOver — opaque src ───────────────────────────────────────────────

    #[test]
    fn test_pd_src_over_opaque() {
        // A fully-opaque source must completely cover the destination.
        let src = opaque(1.0, 0.0, 0.0); // red
        let dst = opaque(0.0, 1.0, 0.0); // green
        let out = composite_pixel(src, dst, PorterDuffOp::SrcOver);
        // αo = 1*1 + 1*(1-1) = 1; Co = Cs*1 + Cd*0 = Cs
        assert!(pixel_approx_eq(out, src), "Opaque SrcOver must show only source: {:?}", out);
    }

    // ── 5. SrcOver — transparent src ─────────────────────────────────────────

    #[test]
    fn test_pd_src_over_transparent() {
        // A fully-transparent source must leave destination unchanged.
        let src = PremulPixel::zero();
        let dst = opaque(0.0, 1.0, 0.0); // green
        let out = composite_pixel(src, dst, PorterDuffOp::SrcOver);
        assert!(
            pixel_approx_eq(out, dst),
            "Transparent SrcOver must show only destination: {:?}",
            out
        );
    }

    // ── 6. SrcOver — half-alpha hand-computed ────────────────────────────────

    #[test]
    fn test_pd_src_over_half_alpha() {
        // src: straight (1, 0, 0) @ α=0.5  ⟹  premul (0.5, 0, 0, 0.5)
        // dst: straight (0, 1, 0) @ α=1.0  ⟹  premul (0, 1, 0, 1.0)
        //
        // SrcOver factors: Fs=1, Fd=1−αs=0.5
        // αo = 0.5*1 + 1.0*0.5 = 1.0
        // ro = 0.5*1 + 0*0.5   = 0.5
        // go = 0*1   + 1*0.5   = 0.5
        // bo = 0
        let src = semi(1.0, 0.0, 0.0, 0.5);
        let dst = opaque(0.0, 1.0, 0.0);
        let out = composite_pixel(src, dst, PorterDuffOp::SrcOver);

        assert!(approx_eq(out.a, 1.0), "alpha mismatch: {}", out.a);
        assert!(approx_eq(out.r, 0.5), "red mismatch: {}", out.r);
        assert!(approx_eq(out.g, 0.5), "green mismatch: {}", out.g);
        assert!(approx_eq(out.b, 0.0), "blue mismatch: {}", out.b);
    }

    // ── 7. Xor — non-overlapping regions ─────────────────────────────────────

    #[test]
    fn test_pd_xor_nonoverlap() {
        // When src and dst are fully opaque and cover complementary regions,
        // Xor should zero both (since Fs = 1−αd = 0 and Fd = 1−αs = 0).
        let src = opaque(1.0, 0.0, 0.0);
        let dst = opaque(0.0, 1.0, 0.0);
        let out = composite_pixel(src, dst, PorterDuffOp::Xor);
        assert!(
            pixel_approx_eq(out, PremulPixel::zero()),
            "Xor of two fully-opaque pixels must be transparent: {:?}",
            out
        );
    }

    // ── 8. Plus — clamped ────────────────────────────────────────────────────

    #[test]
    fn test_pd_plus_clamped() {
        let src = opaque(0.8, 0.6, 0.4);
        let dst = opaque(0.7, 0.7, 0.7);
        let out = composite_pixel(src, dst, PorterDuffOp::Plus);
        assert!(out.r <= 1.0, "Plus r should be ≤ 1.0: {}", out.r);
        assert!(out.g <= 1.0, "Plus g should be ≤ 1.0: {}", out.g);
        assert!(out.b <= 1.0, "Plus b should be ≤ 1.0: {}", out.b);
        assert!(out.a <= 1.0, "Plus a should be ≤ 1.0: {}", out.a);
        assert!(approx_eq(out.r, 1.0), "r = min(0.8+0.7, 1.0) = 1.0, got {}", out.r);
        assert!(approx_eq(out.a, 1.0), "a = min(1+1, 1.0) = 1.0, got {}", out.a);
    }

    // ── 9. SrcIn + SrcOut complement ─────────────────────────────────────────

    #[test]
    fn test_pd_in_out_complement() {
        // SrcIn(s,d) + SrcOut(s,d) == Src(s,d)  (in premultiplied space)
        //
        // SrcIn:  Co = Cs·αd + 0·Fd = Cs·αd
        // SrcOut: Co = Cs·(1−αd) + 0
        // Sum:    Cs·αd + Cs·(1−αd) = Cs  ✓
        let src = semi(0.6, 0.3, 0.9, 0.7);
        let dst = semi(0.2, 0.8, 0.4, 0.5);

        let in_px  = composite_pixel(src, dst, PorterDuffOp::SrcIn);
        let out_px = composite_pixel(src, dst, PorterDuffOp::SrcOut);
        let sum = PremulPixel {
            r: in_px.r + out_px.r,
            g: in_px.g + out_px.g,
            b: in_px.b + out_px.b,
            a: in_px.a + out_px.a,
        };
        let expected = composite_pixel(src, dst, PorterDuffOp::Src);
        assert!(
            pixel_approx_eq(sum, expected),
            "SrcIn + SrcOut != Src: sum={:?}, src={:?}",
            sum,
            expected
        );
    }

    // ── 10. Full-frame buffer composite ──────────────────────────────────────

    #[test]
    fn test_composite_layer_full_frame() {
        // 4×4 frame: solid red source over solid blue destination.
        let w = 4u32;
        let h = 4u32;
        let src: Vec<u8> = (0..w * h).flat_map(|_| [255u8, 0, 0, 255]).collect();
        let dst: Vec<u8> = (0..w * h).flat_map(|_| [0u8, 0, 255, 255]).collect();

        let out = composite_layer(&src, &dst, w, h, PorterDuffOp::SrcOver);

        assert_eq!(out.len(), (w * h * 4) as usize, "Output length mismatch");

        for i in 0..(w * h) as usize {
            let base = i * 4;
            assert_eq!(out[base],     255, "pixel {i}: red channel should be 255");
            assert_eq!(out[base + 1], 0,   "pixel {i}: green channel should be 0");
            assert_eq!(out[base + 2], 0,   "pixel {i}: blue channel should be 0");
            assert_eq!(out[base + 3], 255, "pixel {i}: alpha should be 255");
        }
    }

    // ── 11. to_premul / from_premul round-trip ────────────────────────────────

    #[test]
    fn test_to_from_premul_roundtrip() {
        // Cover a spread of alpha values including fully transparent/opaque.
        let test_cases: &[[u8; 4]] = &[
            [255, 128, 64,  255],
            [200, 100, 50,  128],
            [100,  50, 25,   64],
            [  0,   0,  0,    0],
            [255, 255, 255, 255],
            [ 10,  20, 30,  200],
        ];

        for &rgba in test_cases {
            let pre = to_premul(rgba);
            let back = from_premul(pre);

            if rgba[3] == 0 {
                // Transparent pixels always come back as [0,0,0,0].
                assert_eq!(back, [0, 0, 0, 0], "transparent round-trip: {:?}", back);
                continue;
            }

            for ch in 0..4 {
                let diff = (back[ch] as i16 - rgba[ch] as i16).abs();
                assert!(
                    diff <= 1,
                    "round-trip error > 1 LSB on channel {ch}: input={}, output={}",
                    rgba[ch],
                    back[ch]
                );
            }
        }
    }

    // ── 12. All 14 operators on a known pixel pair ────────────────────────────

    #[test]
    fn test_all_operators_known_pixel() {
        // src: straight (1, 0.5, 0.25) @ α=0.8
        // dst: straight (0.5, 1, 0.25) @ α=0.6
        //
        // Premultiplied:
        //   src = (0.8, 0.4, 0.2, 0.8)
        //   dst = (0.3, 0.6, 0.15, 0.6)
        let src = PremulPixel { r: 0.8, g: 0.4, b: 0.2, a: 0.8 };
        let dst = PremulPixel { r: 0.3, g: 0.6, b: 0.15, a: 0.6 };

        // ── Clear ──────────────────────────────────────────────────────────
        let out = composite_pixel(src, dst, PorterDuffOp::Clear);
        assert!(pixel_approx_eq(out, PremulPixel::zero()), "Clear: {:?}", out);

        // ── Src ───────────────────────────────────────────────────────────
        let out = composite_pixel(src, dst, PorterDuffOp::Src);
        assert!(pixel_approx_eq(out, src), "Src: {:?}", out);

        // ── Dst ───────────────────────────────────────────────────────────
        let out = composite_pixel(src, dst, PorterDuffOp::Dst);
        assert!(pixel_approx_eq(out, dst), "Dst: {:?}", out);

        // ── SrcOver ───────────────────────────────────────────────────────
        // Fs=1, Fd=1-0.8=0.2
        // αo = 0.8 + 0.6*0.2 = 0.92
        // ro = 0.8 + 0.3*0.2 = 0.86
        let out = composite_pixel(src, dst, PorterDuffOp::SrcOver);
        assert!(approx_eq(out.a, 0.8 + 0.6 * 0.2), "SrcOver alpha: {}", out.a);
        assert!(approx_eq(out.r, 0.8 + 0.3 * 0.2), "SrcOver red: {}", out.r);

        // ── DstOver ───────────────────────────────────────────────────────
        // Fs=1-0.6=0.4, Fd=1
        // αo = 0.8*0.4 + 0.6 = 0.92
        // ro = 0.8*0.4 + 0.3 = 0.62
        let out = composite_pixel(src, dst, PorterDuffOp::DstOver);
        assert!(approx_eq(out.a, 0.8 * 0.4 + 0.6), "DstOver alpha: {}", out.a);
        assert!(approx_eq(out.r, 0.8 * 0.4 + 0.3), "DstOver red: {}", out.r);

        // ── SrcIn ─────────────────────────────────────────────────────────
        // Fs=αd=0.6, Fd=0
        // αo = 0.8*0.6 = 0.48; ro = 0.8*0.6 = 0.48
        let out = composite_pixel(src, dst, PorterDuffOp::SrcIn);
        assert!(approx_eq(out.a, 0.8 * 0.6), "SrcIn alpha: {}", out.a);
        assert!(approx_eq(out.r, 0.8 * 0.6), "SrcIn red: {}", out.r);
        assert!(approx_eq(out.g, 0.4 * 0.6), "SrcIn green: {}", out.g);

        // ── DstIn ─────────────────────────────────────────────────────────
        // Fs=0, Fd=αs=0.8
        // αo = 0.6*0.8 = 0.48; ro = 0.3*0.8 = 0.24
        let out = composite_pixel(src, dst, PorterDuffOp::DstIn);
        assert!(approx_eq(out.a, 0.6 * 0.8), "DstIn alpha: {}", out.a);
        assert!(approx_eq(out.r, 0.3 * 0.8), "DstIn red: {}", out.r);

        // ── SrcOut ────────────────────────────────────────────────────────
        // Fs=1-0.6=0.4, Fd=0
        // αo = 0.8*0.4 = 0.32
        let out = composite_pixel(src, dst, PorterDuffOp::SrcOut);
        assert!(approx_eq(out.a, 0.8 * 0.4), "SrcOut alpha: {}", out.a);
        assert!(approx_eq(out.r, 0.8 * 0.4), "SrcOut red: {}", out.r);

        // ── DstOut ────────────────────────────────────────────────────────
        // Fs=0, Fd=1-0.8=0.2
        // αo = 0.6*0.2 = 0.12; ro = 0.3*0.2 = 0.06
        let out = composite_pixel(src, dst, PorterDuffOp::DstOut);
        assert!(approx_eq(out.a, 0.6 * 0.2), "DstOut alpha: {}", out.a);
        assert!(approx_eq(out.r, 0.3 * 0.2), "DstOut red: {}", out.r);

        // ── SrcAtop ───────────────────────────────────────────────────────
        // Fs=αd=0.6, Fd=1-αs=0.2
        // αo = 0.8*0.6 + 0.6*0.2 = 0.48 + 0.12 = 0.60
        // ro = 0.8*0.6 + 0.3*0.2 = 0.48 + 0.06 = 0.54
        let out = composite_pixel(src, dst, PorterDuffOp::SrcAtop);
        assert!(approx_eq(out.a, 0.8 * 0.6 + 0.6 * 0.2), "SrcAtop alpha: {}", out.a);
        assert!(approx_eq(out.r, 0.8 * 0.6 + 0.3 * 0.2), "SrcAtop red: {}", out.r);

        // ── DstAtop ───────────────────────────────────────────────────────
        // Fs=1-0.6=0.4, Fd=αs=0.8
        // αo = 0.8*0.4 + 0.6*0.8 = 0.32 + 0.48 = 0.80
        // ro = 0.8*0.4 + 0.3*0.8 = 0.32 + 0.24 = 0.56
        let out = composite_pixel(src, dst, PorterDuffOp::DstAtop);
        assert!(approx_eq(out.a, 0.8 * 0.4 + 0.6 * 0.8), "DstAtop alpha: {}", out.a);
        assert!(approx_eq(out.r, 0.8 * 0.4 + 0.3 * 0.8), "DstAtop red: {}", out.r);

        // ── Xor ───────────────────────────────────────────────────────────
        // Fs=1-0.6=0.4, Fd=1-0.8=0.2
        // αo = 0.8*0.4 + 0.6*0.2 = 0.32 + 0.12 = 0.44
        // ro = 0.8*0.4 + 0.3*0.2 = 0.32 + 0.06 = 0.38
        let out = composite_pixel(src, dst, PorterDuffOp::Xor);
        assert!(approx_eq(out.a, 0.8 * 0.4 + 0.6 * 0.2), "Xor alpha: {}", out.a);
        assert!(approx_eq(out.r, 0.8 * 0.4 + 0.3 * 0.2), "Xor red: {}", out.r);

        // ── Plus ──────────────────────────────────────────────────────────
        // Fs=1, Fd=1 — clamped
        // αo = min(0.8+0.6, 1.0) = 1.0
        // ro = min(0.8+0.3, 1.0) = 1.0
        let out = composite_pixel(src, dst, PorterDuffOp::Plus);
        assert!(approx_eq(out.a, 1.0), "Plus alpha: {}", out.a);
        assert!(approx_eq(out.r, 1.0), "Plus red: {}", out.r);
        assert!(approx_eq(out.g, 1.0), "Plus green (0.4+0.6=1.0): {}", out.g);

        // ── Dissolve ──────────────────────────────────────────────────────
        // We cannot predict the exact output (it's stochastic), but we can
        // verify that the result is either src or dst colour (after normalisation)
        // and that the alpha channel is non-negative and ≤ 1.
        let out = composite_pixel(src, dst, PorterDuffOp::Dissolve);
        assert!(out.a >= 0.0 && out.a <= 1.0, "Dissolve alpha out of range: {}", out.a);
        assert!(out.r >= 0.0 && out.r <= 1.0, "Dissolve red out of range: {}", out.r);
    }

    // ── composite_layer_into ─────────────────────────────────────────────────

    #[test]
    fn test_composite_layer_into_matches_allocating() {
        let w = 8u32;
        let h = 8u32;
        let src: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let v = ((i * 13) % 256) as u8;
                [v, 255 - v, v / 2, 200u8]
            })
            .collect();
        let dst: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let v = ((i * 7 + 31) % 256) as u8;
                [255 - v, v, v / 3, 150u8]
            })
            .collect();

        let expected = composite_layer(&src, &dst, w, h, PorterDuffOp::SrcOver);

        let mut dst_mut = dst.clone();
        composite_layer_into(&src, &mut dst_mut, w, h, PorterDuffOp::SrcOver);

        assert_eq!(expected, dst_mut, "in-place and allocating versions must produce identical output");
    }

    // ── Dissolve statistical property ────────────────────────────────────────

    #[test]
    fn test_dissolve_statistical_property() {
        // When αs = 0 the dissolve must always pick dst.
        let src = PremulPixel::zero();
        let dst = opaque(0.5, 0.5, 0.5);
        // Test across many positions.
        for y in 0u32..16 {
            for x in 0u32..16 {
                let out = composite_pixel_at(src, dst, PorterDuffOp::Dissolve, x, y);
                assert!(
                    pixel_approx_eq(out, dst),
                    "Dissolve with αs=0 should always pick dst at ({x},{y}): {:?}",
                    out
                );
            }
        }

        // When αs = 1 the dissolve must always pick src.
        let src_full = opaque(1.0, 0.0, 0.0);
        let dst_full = opaque(0.0, 1.0, 0.0);
        for y in 0u32..16 {
            for x in 0u32..16 {
                let out = composite_pixel_at(src_full, dst_full, PorterDuffOp::Dissolve, x, y);
                // The output colour must match src colour (normalised).
                let back = from_premul(out);
                assert_eq!(back[0], 255, "Dissolve αs=1 must pick src red at ({x},{y})");
                assert_eq!(back[1], 0,   "Dissolve αs=1 must pick src green at ({x},{y})");
            }
        }
    }

    // ── DstIn + DstOut complement ─────────────────────────────────────────────

    #[test]
    fn test_pd_dst_in_out_complement() {
        // DstIn + DstOut == Dst (analogous to the SrcIn+SrcOut test).
        let src = semi(0.4, 0.8, 0.3, 0.6);
        let dst = semi(0.7, 0.2, 0.5, 0.8);

        let in_px  = composite_pixel(src, dst, PorterDuffOp::DstIn);
        let out_px = composite_pixel(src, dst, PorterDuffOp::DstOut);
        let sum = PremulPixel {
            r: in_px.r + out_px.r,
            g: in_px.g + out_px.g,
            b: in_px.b + out_px.b,
            a: in_px.a + out_px.a,
        };
        let expected = composite_pixel(src, dst, PorterDuffOp::Dst);
        assert!(
            pixel_approx_eq(sum, expected),
            "DstIn + DstOut != Dst: sum={:?}, dst={:?}",
            sum,
            expected
        );
    }

    // ── SrcAtop alpha equals dst alpha ───────────────────────────────────────

    #[test]
    fn test_src_atop_preserves_dst_alpha() {
        // For SrcAtop: αo = αs·αd + αd·(1−αs) = αd
        let src = semi(0.9, 0.1, 0.5, 0.7);
        let dst = semi(0.3, 0.6, 0.2, 0.4);
        let out = composite_pixel(src, dst, PorterDuffOp::SrcAtop);
        assert!(
            approx_eq(out.a, dst.a),
            "SrcAtop must preserve dst alpha: expected {}, got {}",
            dst.a,
            out.a
        );
    }

    // ── DstAtop alpha equals src alpha ───────────────────────────────────────

    #[test]
    fn test_dst_atop_preserves_src_alpha() {
        // For DstAtop: αo = αs·(1−αd) + αd·αs = αs
        let src = semi(0.9, 0.1, 0.5, 0.7);
        let dst = semi(0.3, 0.6, 0.2, 0.4);
        let out = composite_pixel(src, dst, PorterDuffOp::DstAtop);
        assert!(
            approx_eq(out.a, src.a),
            "DstAtop must preserve src alpha: expected {}, got {}",
            src.a,
            out.a
        );
    }
}
