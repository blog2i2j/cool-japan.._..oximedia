//! Lightweight color processing pipeline with chained transform operations.
//!
//! This module provides a simple, self-contained color pipeline that applies a
//! sequence of [`ColorOp`] operations to floating-point RGB triplets. Unlike
//! the more complex [`crate::pipeline`] module (which depends on full color-space
//! and LUT infrastructure), this pipeline is deliberately minimal: it operates
//! entirely on `f32` values and requires no external dependencies.
//!
//! # Supported Operations
//!
//! - **Matrix** — apply a 3×3 linear matrix to RGB
//! - **Lut1d** — apply a per-channel 1-D look-up table (with linear interpolation)
//! - **Lut3d** — apply a 3-D look-up table (trilinear interpolation)
//! - **GammaCorrect** — raise each channel to the power `1/gamma`
//! - **Clamp** — clamp each channel to a `[min, max]` range
//! - **SceneLinearToDisplay** — apply sRGB OETF (linear → encoded)
//! - **DisplayToSceneLinear** — apply sRGB EOTF (encoded → linear)
//!
//! # Built-in Presets
//!
//! | Function                    | Transform                        |
//! |-----------------------------|----------------------------------|
//! | [`ColorPipeline::srgb_to_linear`]    | Gamma 2.2 decode (approx. sRGB)  |
//! | [`ColorPipeline::linear_to_srgb`]    | Gamma 2.2 encode (approx. sRGB)  |
//! | [`ColorPipeline::bt709_to_bt2020`]   | 3×3 matrix, D65 adapted          |
//! | [`ColorPipeline::bt2020_to_bt709`]   | 3×3 matrix (inverse)             |
//!
//! # Example
//!
//! ```rust
//! use oximedia_colormgmt::color_pipeline::{ColorOp, ColorPipeline};
//!
//! // Build a simple pipeline: clamp then gamma encode.
//! let mut pipe = ColorPipeline::new();
//! pipe.add_op(ColorOp::Clamp { min: 0.0, max: 1.0 });
//! pipe.add_op(ColorOp::GammaCorrect(2.2));
//!
//! let (r, g, b) = pipe.apply(0.5, 0.5, 0.5);
//! assert!(r > 0.0 && r < 1.0);
//! ```

/// A single color processing operation in a [`ColorPipeline`].
#[derive(Debug, Clone)]
pub enum ColorOp {
    /// Multiply the RGB vector by a 3×3 column-major matrix.
    ///
    /// The matrix is stored as `[[row0_col0, row0_col1, row0_col2], ...]`.
    Matrix([[f32; 3]; 3]),

    /// Apply a 1-D look-up table independently to each channel.
    ///
    /// `input` and `output` are parallel arrays of equal length. Linear
    /// interpolation is used between sample points. Values outside the range
    /// of `input` are clamped to the nearest endpoint.
    Lut1d {
        /// Sorted input sample positions.
        input: Vec<f32>,
        /// Corresponding output values (same length as `input`).
        output: Vec<f32>,
    },

    /// Apply a 3-D look-up table via trilinear interpolation.
    ///
    /// `size` is the number of samples along each axis (so the table contains
    /// `size³` entries). `data` is stored in R-fastest order: index
    /// `r + size*g + size²*b` gives `[out_r, out_g, out_b]` as three
    /// consecutive `f32` values (i.e. `data.len() == size³ × 3`).
    Lut3d {
        /// Number of samples per axis (e.g. 17, 33, 65).
        size: u8,
        /// Flattened RGB output values, R-fastest, 3 floats per lattice point.
        data: Vec<f32>,
    },

    /// Raise each channel to the power `1/gamma` (decode / linearise).
    ///
    /// Use `GammaCorrect(2.2)` for the common sRGB approximation.
    GammaCorrect(f32),

    /// Clamp each channel independently to `[min, max]`.
    Clamp {
        /// Lower bound.
        min: f32,
        /// Upper bound.
        max: f32,
    },

    /// Convert scene-linear light to a display-encoded signal using the
    /// piecewise sRGB OETF (IEC 61966-2-1).
    SceneLinearToDisplay,

    /// Decode a display-encoded sRGB signal back to scene-linear light using
    /// the piecewise sRGB EOTF (IEC 61966-2-1).
    DisplayToSceneLinear,
}

// ─────────────────────────────────────────────────────────────────────────────
// BT.709 ↔ BT.2020 matrices (D65 white point, computed from chromaticity
// coordinates via the standard CIE XYZ method).
//
// BT.709  → XYZ (D65) → BT.2020  (combined 3×3, rounded to 6 s.f.)
// BT.2020 → XYZ (D65) → BT.709   (inverse)
// ─────────────────────────────────────────────────────────────────────────────

const BT709_TO_BT2020: [[f32; 3]; 3] = [
    [0.627_404, 0.329_283, 0.043_313],
    [0.069_097, 0.919_541, 0.011_362],
    [0.016_392, 0.088_013, 0.895_595],
];

const BT2020_TO_BT709: [[f32; 3]; 3] = [
    [1.660_491, -0.587_641, -0.072_850],
    [-0.124_551, 1.132_900, -0.008_349],
    [-0.018_151, -0.100_579, 1.118_730],
];

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline
// ─────────────────────────────────────────────────────────────────────────────

/// A sequential chain of [`ColorOp`] operations applied to floating-point RGB.
///
/// Operations are applied in insertion order. The pipeline owns its list of
/// operations and can be cloned or extended at any time.
#[derive(Debug, Clone, Default)]
pub struct ColorPipeline {
    ops: Vec<ColorOp>,
}

impl ColorPipeline {
    /// Create an empty pipeline.
    #[must_use]
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    /// Append an operation to the end of the pipeline.
    pub fn add_op(&mut self, op: ColorOp) {
        self.ops.push(op);
    }

    /// Return the number of operations in the pipeline.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    /// Return `true` if the pipeline contains no operations.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }

    /// Apply the pipeline to a single RGB triplet and return the result.
    ///
    /// Each channel is represented as a 32-bit float (scene-linear or encoded,
    /// depending on the surrounding context).
    #[must_use]
    pub fn apply(&self, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
        let mut rgb = (r, g, b);
        for op in &self.ops {
            rgb = apply_op(op, rgb.0, rgb.1, rgb.2);
        }
        rgb
    }

    /// Apply the pipeline to every pixel in `pixels` in-place.
    ///
    /// This is equivalent to calling [`apply`](Self::apply) for each element,
    /// but avoids repeated bounds checks on the operation list.
    pub fn apply_block(&self, pixels: &mut [(f32, f32, f32)]) {
        for px in pixels.iter_mut() {
            *px = self.apply(px.0, px.1, px.2);
        }
    }

    // ── Built-in presets ─────────────────────────────────────────────────────

    /// Approximate sRGB decoding: raise each channel to the power `1/2.2`.
    ///
    /// For precise sRGB handling (piecewise EOTF) use [`DisplayToSceneLinear`].
    ///
    /// [`DisplayToSceneLinear`]: ColorOp::DisplayToSceneLinear
    #[must_use]
    pub fn srgb_to_linear() -> Self {
        let mut p = Self::new();
        p.add_op(ColorOp::GammaCorrect(2.2));
        p
    }

    /// Approximate sRGB encoding: raise each channel to the power `2.2`.
    ///
    /// For precise sRGB handling (piecewise OETF) use [`SceneLinearToDisplay`].
    ///
    /// [`SceneLinearToDisplay`]: ColorOp::SceneLinearToDisplay
    #[must_use]
    pub fn linear_to_srgb() -> Self {
        let mut p = Self::new();
        // Encoding: apply gamma 1/2.2 → equivalent to GammaCorrect(1/2.2).
        // We store it as a raw GammaCorrect with the *inverse* exponent so
        // that `apply_op` applies `v^(1/gamma)` = `v^(1/(1/2.2))` = `v^2.2`.
        // That's wrong — to *encode* we want v^(1/2.2).
        //
        // The convention chosen for GammaCorrect(g) is:  out = in^(1/g)
        // so GammaCorrect(2.2) → out = in^(1/2.2)  (decode, linearise)
        //    GammaCorrect(1/2.2) → out = in^(2.2)  (encode, de-linearise)
        //
        // For linear_to_srgb (encoding) we use 1/2.2 as the gamma argument.
        p.add_op(ColorOp::GammaCorrect(1.0 / 2.2));
        p
    }

    /// Convert BT.709 linear-light RGB to BT.2020 linear-light RGB.
    ///
    /// Uses the standard 3×3 matrix derived from the respective chromaticity
    /// coordinates and the D65 white point.
    #[must_use]
    pub fn bt709_to_bt2020() -> Self {
        let mut p = Self::new();
        p.add_op(ColorOp::Matrix(BT709_TO_BT2020));
        p
    }

    /// Convert BT.2020 linear-light RGB to BT.709 linear-light RGB.
    ///
    /// Uses the matrix inverse of [`bt709_to_bt2020`](Self::bt709_to_bt2020).
    #[must_use]
    pub fn bt2020_to_bt709() -> Self {
        let mut p = Self::new();
        p.add_op(ColorOp::Matrix(BT2020_TO_BT709));
        p
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Per-operation application
// ─────────────────────────────────────────────────────────────────────────────

/// Apply a single `op` to `(r, g, b)` and return the transformed triplet.
fn apply_op(op: &ColorOp, r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    match op {
        ColorOp::Matrix(m) => {
            let nr = m[0][0] * r + m[0][1] * g + m[0][2] * b;
            let ng = m[1][0] * r + m[1][1] * g + m[1][2] * b;
            let nb = m[2][0] * r + m[2][1] * g + m[2][2] * b;
            (nr, ng, nb)
        }

        ColorOp::Lut1d { input, output } => {
            let lr = lut1d_lookup(input, output, r);
            let lg = lut1d_lookup(input, output, g);
            let lb = lut1d_lookup(input, output, b);
            (lr, lg, lb)
        }

        ColorOp::Lut3d { size, data } => lut3d_lookup(*size, data, r, g, b),

        ColorOp::GammaCorrect(gamma) => {
            let exp = if *gamma == 0.0 { 1.0 } else { 1.0 / gamma };
            let gr = apply_gamma(r, exp);
            let gg = apply_gamma(g, exp);
            let gb = apply_gamma(b, exp);
            (gr, gg, gb)
        }

        ColorOp::Clamp { min, max } => (
            r.clamp(*min, *max),
            g.clamp(*min, *max),
            b.clamp(*min, *max),
        ),

        ColorOp::SceneLinearToDisplay => (
            srgb_linear_to_encoded(r),
            srgb_linear_to_encoded(g),
            srgb_linear_to_encoded(b),
        ),

        ColorOp::DisplayToSceneLinear => (
            srgb_encoded_to_linear(r),
            srgb_encoded_to_linear(g),
            srgb_encoded_to_linear(b),
        ),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Transfer functions (piecewise sRGB OETF / EOTF)
// ─────────────────────────────────────────────────────────────────────────────

/// sRGB EOTF: encoded signal → scene-linear light.
///
/// Handles negative values symmetrically (mirrors the positive segment).
fn srgb_encoded_to_linear(v: f32) -> f32 {
    let sign = if v < 0.0 { -1.0_f32 } else { 1.0_f32 };
    let av = v.abs();
    let linear = if av <= 0.040_45 {
        av / 12.92
    } else {
        ((av + 0.055) / 1.055).powf(2.4)
    };
    sign * linear
}

/// sRGB OETF: scene-linear light → encoded signal.
///
/// Handles negative values symmetrically.
fn srgb_linear_to_encoded(v: f32) -> f32 {
    let sign = if v < 0.0 { -1.0_f32 } else { 1.0_f32 };
    let al = v.abs();
    let encoded = if al <= 0.003_130_8 {
        al * 12.92
    } else {
        1.055 * al.powf(1.0 / 2.4) - 0.055
    };
    sign * encoded
}

// ─────────────────────────────────────────────────────────────────────────────
// Gamma helper
// ─────────────────────────────────────────────────────────────────────────────

/// Raise `v` to `exp`, preserving the sign for negative inputs.
fn apply_gamma(v: f32, exp: f32) -> f32 {
    if v < 0.0 {
        -((-v).powf(exp))
    } else {
        v.powf(exp)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// 1-D LUT lookup with linear interpolation
// ─────────────────────────────────────────────────────────────────────────────

/// Sample a 1-D LUT at position `v` using linear interpolation.
///
/// Returns `output[0]` if `input` is empty, or `output.last()` if `v` is
/// beyond the last entry.
fn lut1d_lookup(input: &[f32], output: &[f32], v: f32) -> f32 {
    let n = input.len().min(output.len());
    if n == 0 {
        return v; // identity fallback
    }
    if n == 1 {
        return output[0];
    }

    // Clamp to range.
    if v <= input[0] {
        return output[0];
    }
    if v >= input[n - 1] {
        return output[n - 1];
    }

    // Binary search for the enclosing interval.
    let idx = match input.binary_search_by(|probe| probe.partial_cmp(&v).unwrap_or(std::cmp::Ordering::Less)) {
        Ok(exact) => return output[exact],
        Err(pos) => pos, // pos is the index where v would be inserted
    };

    let lo = idx.saturating_sub(1);
    let hi = lo + 1;
    if hi >= n {
        return output[n - 1];
    }

    let t = (v - input[lo]) / (input[hi] - input[lo]);
    output[lo] + t * (output[hi] - output[lo])
}

// ─────────────────────────────────────────────────────────────────────────────
// 3-D LUT lookup with trilinear interpolation
// ─────────────────────────────────────────────────────────────────────────────

/// Sample a 3-D LUT at `(r, g, b)` using trilinear interpolation.
///
/// `size` must be ≥ 2 for interpolation. Inputs are clamped to `[0, 1]`.
fn lut3d_lookup(size: u8, data: &[f32], r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let n = size as usize;
    // We need at least a 2³ table to interpolate.
    if n < 2 || data.len() < n * n * n * 3 {
        return (r, g, b); // identity fallback
    }

    let max_idx = (n - 1) as f32;
    let rc = r.clamp(0.0, 1.0) * max_idx;
    let gc = g.clamp(0.0, 1.0) * max_idx;
    let bc = b.clamp(0.0, 1.0) * max_idx;

    let ri = (rc as usize).min(n - 2);
    let gi = (gc as usize).min(n - 2);
    let bi = (bc as usize).min(n - 2);

    let rf = rc - ri as f32;
    let gf = gc - gi as f32;
    let bf = bc - bi as f32;

    // Helper closure: fetch RGB at lattice point (ri+dr, gi+dg, bi+db).
    let fetch = |dr: usize, dg: usize, db: usize| -> [f32; 3] {
        let idx = ((bi + db) * n * n + (gi + dg) * n + (ri + dr)) * 3;
        if idx + 2 < data.len() {
            [data[idx], data[idx + 1], data[idx + 2]]
        } else {
            [0.0, 0.0, 0.0]
        }
    };

    let c000 = fetch(0, 0, 0);
    let c100 = fetch(1, 0, 0);
    let c010 = fetch(0, 1, 0);
    let c110 = fetch(1, 1, 0);
    let c001 = fetch(0, 0, 1);
    let c101 = fetch(1, 0, 1);
    let c011 = fetch(0, 1, 1);
    let c111 = fetch(1, 1, 1);

    let mut out = [0.0_f32; 3];
    for ch in 0..3 {
        let v00 = c000[ch] * (1.0 - rf) + c100[ch] * rf;
        let v10 = c010[ch] * (1.0 - rf) + c110[ch] * rf;
        let v01 = c001[ch] * (1.0 - rf) + c101[ch] * rf;
        let v11 = c011[ch] * (1.0 - rf) + c111[ch] * rf;
        let v0 = v00 * (1.0 - gf) + v10 * gf;
        let v1 = v01 * (1.0 - gf) + v11 * gf;
        out[ch] = v0 * (1.0 - bf) + v1 * bf;
    }

    (out[0], out[1], out[2])
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f32 = 1e-5;

    fn approx_eq(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    fn approx_eq3(a: (f32, f32, f32), b: (f32, f32, f32), eps: f32) -> bool {
        approx_eq(a.0, b.0, eps)
            && approx_eq(a.1, b.1, eps)
            && approx_eq(a.2, b.2, eps)
    }

    // ── Identity / empty pipeline ────────────────────────────────────────────

    #[test]
    fn test_empty_pipeline_is_identity() {
        let pipe = ColorPipeline::new();
        assert_eq!(pipe.apply(0.3, 0.5, 0.7), (0.3, 0.5, 0.7));
    }

    #[test]
    fn test_len_and_is_empty() {
        let mut pipe = ColorPipeline::new();
        assert!(pipe.is_empty());
        assert_eq!(pipe.len(), 0);
        pipe.add_op(ColorOp::Clamp { min: 0.0, max: 1.0 });
        assert!(!pipe.is_empty());
        assert_eq!(pipe.len(), 1);
    }

    // ── Clamp ────────────────────────────────────────────────────────────────

    #[test]
    fn test_clamp_limits_channels() {
        let mut pipe = ColorPipeline::new();
        pipe.add_op(ColorOp::Clamp { min: 0.0, max: 1.0 });
        let (r, g, b) = pipe.apply(-0.5, 0.5, 2.0);
        assert!(approx_eq(r, 0.0, EPS));
        assert!(approx_eq(g, 0.5, EPS));
        assert!(approx_eq(b, 1.0, EPS));
    }

    // ── Gamma round-trip ─────────────────────────────────────────────────────

    #[test]
    fn test_gamma_roundtrip() {
        // srgb_to_linear(v)^(1/2.2) then linear_to_srgb should recover v.
        // GammaCorrect(2.2) = v^(1/2.2) = linearise
        // GammaCorrect(1/2.2) = v^(2.2)  = encode
        let mut pipe = ColorPipeline::new();
        pipe.add_op(ColorOp::GammaCorrect(2.2)); // decode
        pipe.add_op(ColorOp::GammaCorrect(1.0 / 2.2)); // encode back

        for &v in &[0.0_f32, 0.1, 0.3, 0.5, 0.75, 1.0] {
            let (r, _, _) = pipe.apply(v, 0.0, 0.0);
            assert!(
                approx_eq(r, v, 1e-5),
                "gamma roundtrip failed: in={v}, out={r}"
            );
        }
    }

    // ── Matrix ───────────────────────────────────────────────────────────────

    #[test]
    fn test_identity_matrix() {
        let identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut pipe = ColorPipeline::new();
        pipe.add_op(ColorOp::Matrix(identity));
        let (r, g, b) = pipe.apply(0.2, 0.5, 0.8);
        assert!(approx_eq(r, 0.2, EPS));
        assert!(approx_eq(g, 0.5, EPS));
        assert!(approx_eq(b, 0.8, EPS));
    }

    #[test]
    fn test_matrix_apply_simple() {
        // A diagonal scaling matrix.
        let m = [[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.5]];
        let mut pipe = ColorPipeline::new();
        pipe.add_op(ColorOp::Matrix(m));
        let (r, g, b) = pipe.apply(1.0, 1.0, 1.0);
        assert!(approx_eq(r, 2.0, EPS));
        assert!(approx_eq(g, 3.0, EPS));
        assert!(approx_eq(b, 0.5, EPS));
    }

    // ── Block processing ─────────────────────────────────────────────────────

    #[test]
    fn test_apply_block() {
        let mut pipe = ColorPipeline::new();
        pipe.add_op(ColorOp::Clamp { min: 0.0, max: 1.0 });

        let mut pixels = vec![(-1.0, 0.5, 2.0), (0.2, 0.4, 0.6)];
        pipe.apply_block(&mut pixels);

        assert!(approx_eq3(pixels[0], (0.0, 0.5, 1.0), EPS));
        assert!(approx_eq3(pixels[1], (0.2, 0.4, 0.6), EPS));
    }

    // ── sRGB transfer function ───────────────────────────────────────────────

    #[test]
    fn test_srgb_transfer_roundtrip() {
        for &v in &[0.0_f32, 0.05, 0.18, 0.5, 0.8, 1.0] {
            let encoded = srgb_linear_to_encoded(v);
            let decoded = srgb_encoded_to_linear(encoded);
            assert!(
                approx_eq(decoded, v, 1e-5),
                "sRGB transfer roundtrip failed: in={v}, out={decoded}"
            );
        }
    }

    #[test]
    fn test_scene_linear_to_display_op() {
        let mut pipe = ColorPipeline::new();
        pipe.add_op(ColorOp::SceneLinearToDisplay);
        pipe.add_op(ColorOp::DisplayToSceneLinear);

        for &v in &[0.0_f32, 0.18, 1.0] {
            let (r, _, _) = pipe.apply(v, 0.0, 0.0);
            assert!(
                approx_eq(r, v, 1e-5),
                "SceneLinear<->Display roundtrip: in={v}, out={r}"
            );
        }
    }

    // ── Preset pipelines ─────────────────────────────────────────────────────

    #[test]
    fn test_preset_srgb_encode_decode_roundtrip() {
        let decode = ColorPipeline::srgb_to_linear();
        let encode = ColorPipeline::linear_to_srgb();

        for &v in &[0.1_f32, 0.3, 0.5, 0.9] {
            let (lin, _, _) = decode.apply(v, 0.0, 0.0);
            let (enc, _, _) = encode.apply(lin, 0.0, 0.0);
            // Expect near-identity after round-trip (within float precision).
            assert!(
                approx_eq(enc, v, 1e-5),
                "sRGB preset roundtrip failed: in={v}, lin={lin}, out={enc}"
            );
        }
    }

    #[test]
    fn test_preset_bt709_bt2020_roundtrip() {
        let to_2020 = ColorPipeline::bt709_to_bt2020();
        let to_709 = ColorPipeline::bt2020_to_bt709();

        let input = (0.5_f32, 0.3_f32, 0.2_f32);
        let mid = to_2020.apply(input.0, input.1, input.2);
        let out = to_709.apply(mid.0, mid.1, mid.2);

        // The round-trip should recover the original value within float precision.
        assert!(
            approx_eq3(out, input, 1e-3),
            "BT.709↔BT.2020 roundtrip: in={input:?}, via2020={mid:?}, out={out:?}"
        );
    }

    #[test]
    fn test_bt709_to_bt2020_white_preserved() {
        // D65 white (1,1,1) should be preserved by the matrix (within tolerance).
        let pipe = ColorPipeline::bt709_to_bt2020();
        let (r, g, b) = pipe.apply(1.0, 1.0, 1.0);
        assert!(
            approx_eq(r, 1.0, 2e-3) && approx_eq(g, 1.0, 2e-3) && approx_eq(b, 1.0, 2e-3),
            "BT.709→BT.2020 should preserve D65 white: ({r}, {g}, {b})"
        );
    }

    // ── 1-D LUT ──────────────────────────────────────────────────────────────

    #[test]
    fn test_lut1d_identity() {
        let input: Vec<f32> = (0..=10).map(|i| i as f32 / 10.0).collect();
        let output = input.clone();
        let mut pipe = ColorPipeline::new();
        pipe.add_op(ColorOp::Lut1d { input, output });

        for &v in &[0.0_f32, 0.3, 0.7, 1.0] {
            let (r, _, _) = pipe.apply(v, 0.0, 0.0);
            assert!(approx_eq(r, v, 1e-5), "Lut1d identity failed at v={v}: got {r}");
        }
    }

    #[test]
    fn test_lut1d_clamp_extrapolation() {
        let input = vec![0.2_f32, 0.8];
        let output = vec![0.0_f32, 1.0];
        let mut pipe = ColorPipeline::new();
        pipe.add_op(ColorOp::Lut1d { input, output });

        // Below range → clamped to output[0]
        let (r, _, _) = pipe.apply(0.0, 0.0, 0.0);
        assert!(approx_eq(r, 0.0, EPS), "Below-range clamp: {r}");

        // Above range → clamped to output[last]
        let (r2, _, _) = pipe.apply(1.0, 0.0, 0.0);
        assert!(approx_eq(r2, 1.0, EPS), "Above-range clamp: {r2}");
    }
}
