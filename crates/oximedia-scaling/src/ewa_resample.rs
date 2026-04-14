//! Elliptical Weighted Average (EWA) resampling.
//!
//! EWA resampling maps each output pixel back through an inverse affine
//! transform into source space, then integrates a rotationally-symmetric
//! filter over all source samples that fall within the resulting ellipse.
//! This produces dramatically superior anti-aliasing compared to separable
//! horizontal/vertical passes.
//!
//! Reference: Paul Heckbert, "Fundamentals of Texture Mapping and Image
//! Warping", Master's thesis, UCB, 1989.

#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use std::f32::consts::PI;

// ── Kernel functions ──────────────────────────────────────────────────────────

/// Compute `sin(π·x) / (π·x)`, returning 1.0 exactly when x = 0.
#[inline]
#[must_use]
pub fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-7 {
        1.0
    } else {
        let px = PI * x;
        px.sin() / px
    }
}

/// Lanczos reconstruction kernel: `sinc(x) · sinc(x/a)` for |x| < a, else 0.
///
/// `a` is the half-width of the support window (number of lobes).
#[inline]
#[must_use]
pub fn lanczos_kernel(x: f32, a: f32) -> f32 {
    if x.abs() >= a {
        0.0
    } else {
        sinc(x) * sinc(x / a)
    }
}

/// Mitchell-Netravali piecewise cubic filter.
///
/// Classic parameterisation from Mitchell & Netravali (1988).
/// B and C control blur/ringing trade-off:
/// - `(1/3, 1/3)` is the recommended "blur-free" point.
/// - `(0, 1/2)` gives Catmull-Rom.
/// - `(1, 0)` gives B-spline.
///
/// Support radius is 2 units.
#[inline]
#[must_use]
pub fn mitchell_filter(x: f32, b: f32, c: f32) -> f32 {
    let t = x.abs();
    if t < 1.0 {
        let t2 = t * t;
        let t3 = t2 * t;
        ((12.0 - 9.0 * b - 6.0 * c) * t3 + (-18.0 + 12.0 * b + 6.0 * c) * t2 + (6.0 - 2.0 * b))
            / 6.0
    } else if t < 2.0 {
        let t2 = t * t;
        let t3 = t2 * t;
        ((-b - 6.0 * c) * t3
            + (6.0 * b + 30.0 * c) * t2
            + (-12.0 * b - 48.0 * c) * t
            + (8.0 * b + 24.0 * c))
            / 6.0
    } else {
        0.0
    }
}

/// Gaussian kernel: `exp(-x² / (2σ²))`.
///
/// Support is effectively unlimited; in practice values below 1e-6 are
/// treated as zero (|x| > ~3σ).
#[inline]
#[must_use]
pub fn gaussian_kernel(x: f32, sigma: f32) -> f32 {
    if sigma < 1e-8 {
        if x.abs() < 1e-7 {
            1.0
        } else {
            0.0
        }
    } else {
        (-x * x / (2.0 * sigma * sigma)).exp()
    }
}

// ── Filter enum ───────────────────────────────────────────────────────────────

/// The reconstruction filter applied during EWA resampling.
#[derive(Debug, Clone, PartialEq)]
pub enum EwaFilter {
    /// Mitchell-Netravali filter with explicit B and C parameters.
    ///
    /// Support radius = 2.  Good all-around choice at `(1/3, 1/3)`.
    Mitchell(f32, f32),

    /// Lanczos filter with the given tap count (1–8 recommended).
    ///
    /// `Lanczos(3)` is the classic high-quality default.
    Lanczos(u8),

    /// Gaussian filter with the given standard deviation.
    ///
    /// Very soft but guaranteed no ringing.  Effective support ~3σ.
    Gaussian(f32),

    /// Catmull-Rom spline — special case Mitchell(0, 0.5).
    ///
    /// Exact interpolation; slight ringing on sharp edges.
    Catrom,
}

impl EwaFilter {
    /// Evaluate the filter at the given normalised distance `r`.
    ///
    /// `r` is the actual continuous distance in source pixels from the sample
    /// centre; the filter is responsible for its own support bounds.
    #[inline]
    #[must_use]
    pub fn evaluate(&self, r: f32) -> f32 {
        match self {
            EwaFilter::Mitchell(b, c) => mitchell_filter(r, *b, *c),
            EwaFilter::Lanczos(taps) => lanczos_kernel(r, *taps as f32),
            EwaFilter::Gaussian(sigma) => gaussian_kernel(r, *sigma),
            EwaFilter::Catrom => mitchell_filter(r, 0.0, 0.5),
        }
    }

    /// Support radius in source pixels (used to build the ellipse bounding box).
    #[must_use]
    pub fn support_radius(&self) -> f32 {
        match self {
            EwaFilter::Mitchell(_, _) | EwaFilter::Catrom => 2.0,
            EwaFilter::Lanczos(taps) => *taps as f32,
            EwaFilter::Gaussian(sigma) => 3.0 * sigma.max(1.0),
        }
    }
}

// ── EwaResampler ──────────────────────────────────────────────────────────────

/// EWA (Elliptical Weighted Average) resampler.
///
/// For each output pixel the algorithm:
/// 1. Maps the destination pixel centre back to source coordinates via an
///    inverse scale transform.
/// 2. Constructs the ellipse whose semi-axes match the local scale factors,
///    described by the quadratic form `A·dx² + B·dx·dy + C·dy² ≤ F`.
/// 3. Iterates over all source pixels inside the bounding box of the ellipse.
/// 4. Accumulates weighted samples using the chosen filter evaluated at the
///    normalised ellipse distance `r² = (A·dx² + B·dx·dy + C·dy²) / F`.
#[derive(Debug, Clone)]
pub struct EwaResampler {
    /// Reconstruction filter to apply.
    pub filter: EwaFilter,
    /// If true, negative output values are clamped to 0 (prevents ringing
    /// artefacts at the cost of slight DC shift with Lanczos).
    pub clamp_negatives: bool,
}

impl Default for EwaResampler {
    fn default() -> Self {
        Self {
            filter: EwaFilter::Mitchell(1.0 / 3.0, 1.0 / 3.0),
            clamp_negatives: false,
        }
    }
}

impl EwaResampler {
    /// Create a resampler with the given filter and negative-clamping option.
    #[must_use]
    pub fn new(filter: EwaFilter, clamp_negatives: bool) -> Self {
        Self {
            filter,
            clamp_negatives,
        }
    }

    /// Resample a single-channel float image.
    ///
    /// - `src`: source pixels in row-major order, values in any range.
    /// - `src_w`, `src_h`: source dimensions.
    /// - `dst_w`, `dst_h`: destination dimensions.
    ///
    /// Returns a `dst_w × dst_h` output in the same value range.
    /// Returns an empty vector if any dimension is zero.
    #[must_use]
    pub fn resample(
        &self,
        src: &[f32],
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Vec<f32> {
        if src_w == 0 || src_h == 0 || dst_w == 0 || dst_h == 0 {
            return Vec::new();
        }

        // Scale factors (source pixels per destination pixel).
        let sx = src_w as f32 / dst_w as f32;
        let sy = src_h as f32 / dst_h as f32;

        // EWA ellipse quadratic form coefficients for an axis-aligned scale.
        // The ellipse equation is: A·dx² + C·dy² ≤ F
        // For axis-aligned case B = 0, A = 1/sx², C = 1/sy², F = 1.
        // We absorb F into the filter call by normalising r² below.
        let inv_sx2 = 1.0 / (sx * sx);
        let inv_sy2 = 1.0 / (sy * sy);

        // Support radius in source pixels (extended by a safety margin).
        let support = self.filter.support_radius();
        let src_radius_x = support * sx;
        let src_radius_y = support * sy;

        let mut dst = vec![0.0f32; dst_w * dst_h];

        for dy in 0..dst_h {
            for dx in 0..dst_w {
                // Map destination centre to source coordinates (centre-aligned).
                let src_cx = (dx as f32 + 0.5) * sx - 0.5;
                let src_cy = (dy as f32 + 0.5) * sy - 0.5;

                // Bounding box in source space.
                let x_min = (src_cx - src_radius_x).floor() as i64;
                let x_max = (src_cx + src_radius_x).ceil() as i64;
                let y_min = (src_cy - src_radius_y).floor() as i64;
                let y_max = (src_cy + src_radius_y).ceil() as i64;

                let mut weight_sum = 0.0f64;
                let mut value_sum = 0.0f64;

                for sy_idx in y_min..=y_max {
                    // Clamp to valid source rows.
                    let clamped_sy = sy_idx.clamp(0, src_h as i64 - 1) as usize;
                    let diff_y = sy_idx as f32 - src_cy;

                    for sx_idx in x_min..=x_max {
                        let clamped_sx = sx_idx.clamp(0, src_w as i64 - 1) as usize;
                        let diff_x = sx_idx as f32 - src_cx;

                        // Normalised ellipse distance squared: r² = A·dx² + C·dy²
                        // where A = 1/sx², C = 1/sy² (B=0 for axis-aligned).
                        let r2 = diff_x * diff_x * inv_sx2 + diff_y * diff_y * inv_sy2;

                        // Only integrate within the unit ellipse (r² ≤ 1)
                        // scaled by the filter support window.
                        if r2 > support * support {
                            continue;
                        }

                        // Evaluate filter at the actual source-pixel distance.
                        let dist = (diff_x * diff_x + diff_y * diff_y).sqrt();
                        let w = self.filter.evaluate(dist / sx.max(sy)) as f64;

                        let sample = src[clamped_sy * src_w + clamped_sx] as f64;
                        weight_sum += w;
                        value_sum += w * sample;
                    }
                }

                let raw = if weight_sum.abs() > 1e-12 {
                    (value_sum / weight_sum) as f32
                } else {
                    // Fallback: nearest-neighbour when no weights accumulate.
                    let nx = src_cx.round().clamp(0.0, (src_w - 1) as f32) as usize;
                    let ny = src_cy.round().clamp(0.0, (src_h - 1) as f32) as usize;
                    src[ny * src_w + nx]
                };

                dst[dy * dst_w + dx] = if self.clamp_negatives {
                    raw.max(0.0)
                } else {
                    raw
                };
            }
        }

        dst
    }
}

// ── FilterWeightTable ─────────────────────────────────────────────────────────

/// A precomputed 1D filter weight table for a specific (src_len, dst_len) pair.
///
/// Computing filter weights for each output sample during scaling is expensive
/// when the same scale factor is applied to many rows or columns (e.g., when
/// scaling all rows of an image to the same target width).  This structure
/// amortises that cost by precomputing all weights once and storing them in a
/// flat, cache-friendly layout.
///
/// # Memory layout
///
/// Weights are stored in row-major order: the weights for output sample `i`
/// occupy `self.entries[i]` which is a `Vec<WeightEntry>`.  To avoid
/// per-lookup allocation the table uses a flat `Vec<f32>` and a separate
/// `Vec<(usize, usize)>` slice index.
///
/// # Example
///
/// ```
/// use oximedia_scaling::ewa_resample::{EwaFilter, FilterWeightTable};
///
/// // Precompute weights for 2x downscale (8 → 4) with Mitchell filter.
/// let table = FilterWeightTable::build(
///     &EwaFilter::Mitchell(1.0 / 3.0, 1.0 / 3.0),
///     8, 4,
/// );
/// assert_eq!(table.dst_len(), 4);
/// // Apply the table to a source signal.
/// let src: Vec<f32> = (0..8).map(|i| i as f32 / 7.0).collect();
/// let dst = table.apply(&src);
/// assert_eq!(dst.len(), 4);
/// // Output should be monotonically increasing.
/// for w in dst.windows(2) {
///     assert!(w[1] >= w[0] - 0.01, "non-monotone");
/// }
/// ```
#[derive(Debug, Clone)]
pub struct FilterWeightTable {
    /// Scale ratio: src_len / dst_len.
    scale: f32,
    /// Source length this table was computed for.
    src_len: usize,
    /// Destination length this table was computed for.
    dst_len: usize,
    /// Flat weight storage.  The weights for output sample `i` start at offset
    /// `offsets[i].0` and have count `offsets[i].1`.  Weights are normalised
    /// so their sum is 1.0.
    weights: Vec<f32>,
    /// `(start_offset, count)` for each output sample.
    offsets: Vec<(usize, usize)>,
    /// Source sample index corresponding to each weight entry.
    src_indices: Vec<usize>,
}

impl FilterWeightTable {
    /// Build a weight table for resampling from `src_len` to `dst_len` samples
    /// using the given `filter`.
    ///
    /// # Panics
    ///
    /// Does not panic; returns a zero-sized table when either dimension is 0.
    #[must_use]
    pub fn build(filter: &EwaFilter, src_len: usize, dst_len: usize) -> Self {
        if src_len == 0 || dst_len == 0 {
            return Self {
                scale: 1.0,
                src_len,
                dst_len,
                weights: Vec::new(),
                offsets: Vec::new(),
                src_indices: Vec::new(),
            };
        }

        let scale = src_len as f32 / dst_len as f32;
        let support = filter.support_radius() * scale.max(1.0);

        let mut weights: Vec<f32> = Vec::new();
        let mut offsets: Vec<(usize, usize)> = Vec::with_capacity(dst_len);
        let mut src_indices: Vec<usize> = Vec::new();

        for i in 0..dst_len {
            // Centre of output sample `i` mapped back to source space.
            let centre = (i as f32 + 0.5) * scale - 0.5;
            let lo = (centre - support).floor() as i64;
            let hi = (centre + support).ceil() as i64;

            let entry_start = weights.len();
            let mut weight_sum = 0.0f32;

            for s in lo..=hi {
                let clamped = s.clamp(0, src_len as i64 - 1) as usize;
                let x = (s as f32 - centre) / scale.max(1.0);
                let w = filter.evaluate(x);
                weight_sum += w;
                weights.push(w);
                src_indices.push(clamped);
            }

            let count = weights.len() - entry_start;

            // Normalise weights so they sum to 1.0.
            if weight_sum.abs() > 1e-8 {
                let inv = 1.0 / weight_sum;
                for w in &mut weights[entry_start..entry_start + count] {
                    *w *= inv;
                }
            }

            offsets.push((entry_start, count));
        }

        Self {
            scale,
            src_len,
            dst_len,
            weights,
            offsets,
            src_indices,
        }
    }

    /// Returns the number of source samples this table was built for.
    #[inline]
    #[must_use]
    pub fn src_len(&self) -> usize {
        self.src_len
    }

    /// Returns the number of destination samples this table will produce.
    #[inline]
    #[must_use]
    pub fn dst_len(&self) -> usize {
        self.dst_len
    }

    /// Returns the scale ratio (src/dst) for this table.
    #[inline]
    #[must_use]
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Apply the precomputed weights to a source signal slice.
    ///
    /// `src` must have length `>= src_len`.  If it is shorter, missing values
    /// are treated as 0.  Returns a `Vec<f32>` of length `dst_len`.
    #[must_use]
    pub fn apply(&self, src: &[f32]) -> Vec<f32> {
        if self.dst_len == 0 || self.offsets.is_empty() {
            return Vec::new();
        }
        let mut dst = vec![0.0f32; self.dst_len];
        for (i, (start, count)) in self.offsets.iter().enumerate() {
            let mut acc = 0.0f32;
            for j in 0..*count {
                let src_idx = self.src_indices[start + j];
                let w = self.weights[start + j];
                let sample = src.get(src_idx).copied().unwrap_or(0.0);
                acc += w * sample;
            }
            dst[i] = acc;
        }
        dst
    }

    /// Apply the precomputed weights to a u8 source signal, returning `Vec<f32>`
    /// in `[0, 255]` floating-point range.
    ///
    /// Useful for integer pixel data; caller can clamp and convert back to u8.
    #[must_use]
    pub fn apply_u8(&self, src: &[u8]) -> Vec<f32> {
        if self.dst_len == 0 {
            return Vec::new();
        }
        let mut dst = vec![0.0f32; self.dst_len];
        for (i, (start, count)) in self.offsets.iter().enumerate() {
            let mut acc = 0.0f32;
            for j in 0..*count {
                let src_idx = self.src_indices[start + j];
                let w = self.weights[start + j];
                let sample = src.get(src_idx).copied().unwrap_or(0) as f32;
                acc += w * sample;
            }
            dst[i] = acc;
        }
        dst
    }

    /// Use this table to scale a 2D grayscale image (horizontal pass).
    ///
    /// The table is applied independently to each row of the `src` image.
    /// `src` must have length `src_h * src_len`.  Returns a buffer of size
    /// `src_h * dst_len`.
    #[must_use]
    pub fn apply_horizontal_pass(&self, src: &[f32], src_h: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; src_h * self.dst_len];
        for row in 0..src_h {
            let src_row = &src[row * self.src_len..(row + 1) * self.src_len];
            let dst_row = self.apply(src_row);
            out[row * self.dst_len..(row + 1) * self.dst_len].copy_from_slice(&dst_row);
        }
        out
    }

    /// Use this table to scale a 2D grayscale image (vertical pass).
    ///
    /// Each column of `src` (width `src_w`, height `src_len`) is resampled to
    /// height `dst_len`.  Returns a buffer of size `dst_len * src_w`.
    #[must_use]
    pub fn apply_vertical_pass(&self, src: &[f32], src_w: usize) -> Vec<f32> {
        let src_h = self.src_len;
        let dst_h = self.dst_len;
        let mut out = vec![0.0f32; dst_h * src_w];

        for col in 0..src_w {
            // Extract column into a temporary buffer.
            let col_buf: Vec<f32> = (0..src_h).map(|r| src[r * src_w + col]).collect();
            let resampled = self.apply(&col_buf);
            for (row, &v) in resampled.iter().enumerate() {
                out[row * src_w + col] = v;
            }
        }
        out
    }
}

/// A pair of precomputed weight tables for 2D image scaling.
///
/// Holds one table for the horizontal pass and one for the vertical pass,
/// enabling full-image scaling with a single pair of builds followed by
/// O(dst_w × src_h + dst_h × dst_w) work — far faster than the EWA
/// per-pixel approach when scaling many rows to the same target size.
///
/// # Example
///
/// ```
/// use oximedia_scaling::ewa_resample::{EwaFilter, ScaleWeightCache};
///
/// let cache = ScaleWeightCache::build(
///     &EwaFilter::Mitchell(1.0/3.0, 1.0/3.0),
///     8, 8,   // src dimensions
///     4, 4,   // dst dimensions
/// );
/// let src: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
/// let dst = cache.apply(&src);
/// assert_eq!(dst.len(), 16); // 4 × 4
/// ```
#[derive(Debug, Clone)]
pub struct ScaleWeightCache {
    h_table: FilterWeightTable,
    v_table: FilterWeightTable,
}

impl ScaleWeightCache {
    /// Build horizontal and vertical weight tables for scaling from
    /// `(src_w, src_h)` to `(dst_w, dst_h)`.
    #[must_use]
    pub fn build(
        filter: &EwaFilter,
        src_w: usize,
        src_h: usize,
        dst_w: usize,
        dst_h: usize,
    ) -> Self {
        Self {
            h_table: FilterWeightTable::build(filter, src_w, dst_w),
            v_table: FilterWeightTable::build(filter, src_h, dst_h),
        }
    }

    /// Apply the cached tables to scale `src` from the configured source
    /// dimensions to the configured destination dimensions.
    ///
    /// Performs a horizontal pass (each row scaled to `dst_w`) followed by a
    /// vertical pass (each column scaled to `dst_h`).
    ///
    /// Returns a flat `Vec<f32>` of length `dst_w × dst_h`.
    #[must_use]
    pub fn apply(&self, src: &[f32]) -> Vec<f32> {
        let src_h = self.v_table.src_len();
        let dst_w = self.h_table.dst_len();
        let dst_h = self.v_table.dst_len();

        if dst_w == 0 || dst_h == 0 || src.is_empty() {
            return vec![0.0f32; dst_w * dst_h];
        }

        // Horizontal pass: for each row, resample from src_w to dst_w.
        let h_out = self.h_table.apply_horizontal_pass(src, src_h);
        // Vertical pass: for each column of h_out, resample from src_h to dst_h.
        self.v_table.apply_vertical_pass(&h_out, dst_w)
    }

    /// Source width (from horizontal table).
    #[inline]
    #[must_use]
    pub fn src_w(&self) -> usize {
        self.h_table.src_len()
    }

    /// Source height (from vertical table).
    #[inline]
    #[must_use]
    pub fn src_h(&self) -> usize {
        self.v_table.src_len()
    }

    /// Destination width.
    #[inline]
    #[must_use]
    pub fn dst_w(&self) -> usize {
        self.h_table.dst_len()
    }

    /// Destination height.
    #[inline]
    #[must_use]
    pub fn dst_h(&self) -> usize {
        self.v_table.dst_len()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── sinc ─────────────────────────────────────────────────────────────────

    #[test]
    fn sinc_at_zero_is_one() {
        assert!((sinc(0.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn sinc_at_integers_is_zero() {
        for n in 1..=5i32 {
            let v = sinc(n as f32);
            assert!(v.abs() < 1e-5, "sinc({n}) = {v}, expected 0");
        }
    }

    #[test]
    fn sinc_is_symmetric() {
        for i in 1..=10 {
            let x = i as f32 * 0.3;
            let diff = (sinc(x) - sinc(-x)).abs();
            assert!(diff < 1e-6, "sinc not symmetric at ±{x}");
        }
    }

    #[test]
    fn sinc_value_at_half() {
        // sinc(0.5) = sin(π/2)/(π/2) = 1/(π/2) = 2/π ≈ 0.6366
        let expected = 2.0 / PI;
        let v = sinc(0.5);
        assert!(
            (v - expected).abs() < 1e-5,
            "sinc(0.5) = {v}, expected {expected}"
        );
    }

    // ── lanczos_kernel ────────────────────────────────────────────────────────

    #[test]
    fn lanczos_kernel_at_zero_is_one() {
        for a in [1, 2, 3, 5] {
            let v = lanczos_kernel(0.0, a as f32);
            assert!((v - 1.0).abs() < 1e-5, "lanczos({0}, a={a}) != 1", v);
        }
    }

    #[test]
    fn lanczos_kernel_zero_outside_support() {
        let v = lanczos_kernel(3.1, 3.0);
        assert_eq!(v, 0.0);
        let v2 = lanczos_kernel(-4.0, 3.0);
        assert_eq!(v2, 0.0);
    }

    #[test]
    fn lanczos_kernel_at_boundary_is_zero() {
        // sinc(a) = 0 for integer a
        let v = lanczos_kernel(2.9999, 3.0);
        // Near but not at boundary, should be small
        assert!(v.abs() < 0.02);
    }

    // ── mitchell_filter ───────────────────────────────────────────────────────

    #[test]
    fn mitchell_filter_at_zero_is_one_for_standard_b_c() {
        // For any (B, C), mitchell(0) = (6 - 2B) / 6
        let (b, c) = (1.0_f32 / 3.0, 1.0_f32 / 3.0);
        let expected = (6.0 - 2.0 * b) / 6.0;
        let v = mitchell_filter(0.0, b, c);
        assert!(
            (v - expected).abs() < 1e-5,
            "mitchell(0) = {v}, expected {expected}"
        );
    }

    #[test]
    fn catrom_mitchell_at_zero_is_one() {
        // Catmull-Rom: B=0, C=0.5 → (6-0)/6 = 1
        let v = mitchell_filter(0.0, 0.0, 0.5);
        assert!((v - 1.0).abs() < 1e-5);
    }

    #[test]
    fn mitchell_filter_zero_outside_support() {
        assert_eq!(mitchell_filter(2.0, 0.333, 0.333), 0.0);
        assert_eq!(mitchell_filter(-3.0, 0.0, 0.5), 0.0);
    }

    #[test]
    fn mitchell_filter_symmetry() {
        let (b, c) = (1.0_f32 / 3.0, 1.0_f32 / 3.0);
        for i in 1..=15 {
            let x = i as f32 * 0.1;
            let diff = (mitchell_filter(x, b, c) - mitchell_filter(-x, b, c)).abs();
            assert!(diff < 1e-5, "mitchell not symmetric at ±{x}");
        }
    }

    // ── EwaFilter enum ────────────────────────────────────────────────────────

    #[test]
    fn ewa_filter_mitchell_evaluate_at_zero() {
        let f = EwaFilter::Mitchell(1.0 / 3.0, 1.0 / 3.0);
        let expected = (6.0 - 2.0_f32 / 3.0) / 6.0;
        let v = f.evaluate(0.0);
        assert!((v - expected).abs() < 1e-4);
    }

    #[test]
    fn ewa_filter_catrom_same_as_mitchell_0_half() {
        let catrom = EwaFilter::Catrom;
        let mitchell = EwaFilter::Mitchell(0.0, 0.5);
        for i in 0..20 {
            let x = i as f32 * 0.1;
            let diff = (catrom.evaluate(x) - mitchell.evaluate(x)).abs();
            assert!(diff < 1e-6, "Catrom != Mitchell(0,0.5) at x={x}");
        }
    }

    #[test]
    fn ewa_filter_lanczos_support_radius() {
        let f = EwaFilter::Lanczos(3);
        assert!((f.support_radius() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn ewa_filter_gaussian_support_radius() {
        let f = EwaFilter::Gaussian(1.0);
        assert!((f.support_radius() - 3.0).abs() < 1e-6);
    }

    // ── EwaResampler ──────────────────────────────────────────────────────────

    #[test]
    fn ewa_resample_empty_returns_empty() {
        let r = EwaResampler::default();
        assert!(r.resample(&[], 0, 0, 4, 4).is_empty());
        assert!(r.resample(&[], 4, 4, 0, 4).is_empty());
    }

    #[test]
    fn ewa_resample_identity_size() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let r = EwaResampler::default();
        let dst = r.resample(&src, 4, 4, 4, 4);
        assert_eq!(dst.len(), 16);
    }

    #[test]
    fn ewa_resample_2x_upscale_size() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let r = EwaResampler::default();
        let dst = r.resample(&src, 4, 4, 8, 8);
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn ewa_resample_2x_downscale_size() {
        let src: Vec<f32> = (0..64).map(|i| i as f32 / 63.0).collect();
        let r = EwaResampler::default();
        let dst = r.resample(&src, 8, 8, 4, 4);
        assert_eq!(dst.len(), 16);
    }

    #[test]
    fn ewa_resample_uniform_image_stays_uniform() {
        let src = vec![0.5f32; 64];
        let r = EwaResampler::default();
        let dst = r.resample(&src, 8, 8, 4, 4);
        for &v in &dst {
            assert!((v - 0.5).abs() < 0.01, "uniform image produced {v}");
        }
    }

    #[test]
    fn ewa_resample_clamp_negatives_flag() {
        // Construct a signal where Lanczos can produce slight negative ring values.
        let mut src = vec![0.0f32; 16];
        src[7] = 1.0; // single spike in 4x4 grid
        let r = EwaResampler::new(EwaFilter::Lanczos(3), true);
        let dst = r.resample(&src, 4, 4, 8, 8);
        for &v in &dst {
            assert!(v >= 0.0, "clamp_negatives=true produced negative value {v}");
        }
    }

    #[test]
    fn ewa_resample_lanczos_upscale_preserves_range() {
        let src: Vec<f32> = (0..16).map(|i| i as f32 / 15.0).collect();
        let r = EwaResampler::new(EwaFilter::Lanczos(3), false);
        let dst = r.resample(&src, 4, 4, 8, 8);
        // Lanczos can ring slightly beyond [0, 1]; accept ±25 % overshoot.
        let min = dst.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = dst.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(min >= -0.25, "min out of range: {min}");
        assert!(max <= 1.25, "max out of range: {max}");
    }

    #[test]
    fn ewa_resample_gaussian_smooth() {
        // Gaussian filter output should be smoother than input (low-pass).
        let mut src = vec![0.0f32; 64];
        // Checkerboard pattern.
        for y in 0..8 {
            for x in 0..8 {
                src[y * 8 + x] = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
            }
        }
        let r = EwaResampler::new(EwaFilter::Gaussian(1.5), false);
        let dst = r.resample(&src, 8, 8, 8, 8);
        // Middle pixels should be grey-ish (averaged out).
        let centre = dst[3 * 8 + 3];
        assert!(centre > 0.2 && centre < 0.8, "centre pixel = {centre}");
    }

    #[test]
    fn ewa_resample_2x_downscale_monotone_ramp() {
        // A horizontal ramp should remain roughly monotone after 2x downscale.
        let src: Vec<f32> = (0..64).map(|i| (i % 8) as f32 / 7.0).collect();
        let r = EwaResampler::default();
        let dst = r.resample(&src, 8, 8, 4, 4);
        // Check rows are roughly monotone.
        for row in 0..4 {
            let row_vals: Vec<f32> = (0..4).map(|c| dst[row * 4 + c]).collect();
            for pair in row_vals.windows(2) {
                assert!(pair[1] >= pair[0] - 0.05, "non-monotone at row {row}");
            }
        }
    }

    #[test]
    fn ewa_resample_single_pixel_src() {
        let src = vec![0.7f32];
        let r = EwaResampler::default();
        let dst = r.resample(&src, 1, 1, 3, 3);
        assert_eq!(dst.len(), 9);
        for &v in &dst {
            assert!((v - 0.7).abs() < 0.01, "single-pixel upscale: {v}");
        }
    }

    #[test]
    fn sinc_near_zero_smooth() {
        // sinc should be smooth near zero — values should be close to 1.
        for i in 1..=5 {
            let x = i as f32 * 1e-4;
            let v = sinc(x);
            assert!((v - 1.0).abs() < 1e-4, "sinc({x}) = {v}");
        }
    }

    #[test]
    fn mitchell_filter_continuous_at_knot_one() {
        // Filter must be C0-continuous at |x|=1.
        let (b, c) = (1.0_f32 / 3.0, 1.0_f32 / 3.0);
        let left = mitchell_filter(1.0 - 1e-5, b, c);
        let right = mitchell_filter(1.0 + 1e-5, b, c);
        assert!(
            (left - right).abs() < 0.01,
            "discontinuity at 1: {left} vs {right}"
        );
    }

    // ── FilterWeightTable ─────────────────────────────────────────────────────

    #[test]
    fn filter_weight_table_dst_len_matches() {
        let f = EwaFilter::Mitchell(1.0 / 3.0, 1.0 / 3.0);
        let t = FilterWeightTable::build(&f, 8, 4);
        assert_eq!(t.dst_len(), 4);
    }

    #[test]
    fn filter_weight_table_src_len_matches() {
        let f = EwaFilter::Lanczos(3);
        let t = FilterWeightTable::build(&f, 16, 8);
        assert_eq!(t.src_len(), 16);
    }

    #[test]
    fn filter_weight_table_apply_produces_correct_length() {
        let f = EwaFilter::Mitchell(1.0 / 3.0, 1.0 / 3.0);
        let t = FilterWeightTable::build(&f, 8, 4);
        let src: Vec<f32> = (0..8).map(|i| i as f32 / 7.0).collect();
        let dst = t.apply(&src);
        assert_eq!(dst.len(), 4);
    }

    #[test]
    fn filter_weight_table_zero_dimensions_returns_empty() {
        let f = EwaFilter::Catrom;
        let t = FilterWeightTable::build(&f, 0, 4);
        assert_eq!(t.dst_len(), 4);
        let dst = t.apply(&[]);
        assert!(dst.is_empty());

        let t2 = FilterWeightTable::build(&f, 8, 0);
        assert_eq!(t2.dst_len(), 0);
        let dst2 = t2.apply(&[1.0; 8]);
        assert!(dst2.is_empty());
    }

    #[test]
    fn filter_weight_table_apply_uniform_signal_stays_constant() {
        // A uniform signal should produce a constant output regardless of scale.
        let f = EwaFilter::Mitchell(1.0 / 3.0, 1.0 / 3.0);
        let t = FilterWeightTable::build(&f, 8, 4);
        let src = vec![0.5f32; 8];
        let dst = t.apply(&src);
        for &v in &dst {
            assert!((v - 0.5).abs() < 0.02, "uniform signal deviated: {v}");
        }
    }

    #[test]
    fn filter_weight_table_downscale_monotone_ramp() {
        let f = EwaFilter::Lanczos(3);
        let t = FilterWeightTable::build(&f, 8, 4);
        let src: Vec<f32> = (0..8).map(|i| i as f32 / 7.0).collect();
        let dst = t.apply(&src);
        assert_eq!(dst.len(), 4);
        for w in dst.windows(2) {
            assert!(w[1] >= w[0] - 0.05, "non-monotone: {:?}", w);
        }
    }

    #[test]
    fn filter_weight_table_apply_u8_uniform() {
        let f = EwaFilter::Catrom;
        let t = FilterWeightTable::build(&f, 8, 4);
        let src = vec![128u8; 8];
        let dst = t.apply_u8(&src);
        for &v in &dst {
            assert!((v - 128.0).abs() < 3.0, "u8 uniform deviated: {v}");
        }
    }

    #[test]
    fn filter_weight_table_horizontal_pass_size() {
        let f = EwaFilter::Mitchell(1.0 / 3.0, 1.0 / 3.0);
        let t = FilterWeightTable::build(&f, 8, 4);
        let src = vec![0.5f32; 8 * 6]; // 8 wide, 6 rows
        let out = t.apply_horizontal_pass(&src, 6);
        assert_eq!(
            out.len(),
            4 * 6,
            "horizontal pass: expected {} got {}",
            4 * 6,
            out.len()
        );
    }

    #[test]
    fn filter_weight_table_vertical_pass_size() {
        let f = EwaFilter::Lanczos(3);
        let t = FilterWeightTable::build(&f, 8, 4);
        let src = vec![0.5f32; 8 * 10]; // 10 wide, 8 tall
        let out = t.apply_vertical_pass(&src, 10);
        assert_eq!(out.len(), 4 * 10);
    }

    #[test]
    fn filter_weight_table_scale_ratio() {
        let f = EwaFilter::Catrom;
        let t = FilterWeightTable::build(&f, 8, 4);
        assert!(
            (t.scale() - 2.0).abs() < 1e-5,
            "expected scale=2.0 got {}",
            t.scale()
        );
    }

    // ── ScaleWeightCache ──────────────────────────────────────────────────────

    #[test]
    fn scale_weight_cache_output_size() {
        let f = EwaFilter::Mitchell(1.0 / 3.0, 1.0 / 3.0);
        let cache = ScaleWeightCache::build(&f, 8, 8, 4, 4);
        let src = vec![0.5f32; 64];
        let dst = cache.apply(&src);
        assert_eq!(dst.len(), 16);
    }

    #[test]
    fn scale_weight_cache_uniform_image() {
        let f = EwaFilter::Lanczos(3);
        let cache = ScaleWeightCache::build(&f, 8, 8, 4, 4);
        let src = vec![0.7f32; 64];
        let dst = cache.apply(&src);
        for &v in &dst {
            assert!((v - 0.7).abs() < 0.05, "uniform image deviated: {v}");
        }
    }

    #[test]
    fn scale_weight_cache_dimension_accessors() {
        let f = EwaFilter::Catrom;
        let cache = ScaleWeightCache::build(&f, 16, 12, 8, 6);
        assert_eq!(cache.src_w(), 16);
        assert_eq!(cache.src_h(), 12);
        assert_eq!(cache.dst_w(), 8);
        assert_eq!(cache.dst_h(), 6);
    }

    #[test]
    fn scale_weight_cache_upscale_output_size() {
        let f = EwaFilter::Mitchell(1.0 / 3.0, 1.0 / 3.0);
        let cache = ScaleWeightCache::build(&f, 4, 4, 8, 8);
        let src = vec![0.3f32; 16];
        let dst = cache.apply(&src);
        assert_eq!(dst.len(), 64);
    }

    #[test]
    fn scale_weight_cache_monotone_ramp_2x_downscale() {
        // A horizontal ramp image should remain roughly monotone after 2x downscale.
        let f = EwaFilter::Mitchell(1.0 / 3.0, 1.0 / 3.0);
        let cache = ScaleWeightCache::build(&f, 8, 8, 4, 4);
        let src: Vec<f32> = (0..64).map(|i| (i % 8) as f32 / 7.0).collect();
        let dst = cache.apply(&src);
        // Check that each output row is roughly non-decreasing.
        for row in 0..4 {
            for col in 1..4 {
                let prev = dst[row * 4 + col - 1];
                let curr = dst[row * 4 + col];
                assert!(curr >= prev - 0.1, "row {row} col {col}: {curr} < {prev}");
            }
        }
    }

    #[test]
    fn scale_weight_cache_empty_src_returns_zeros() {
        let f = EwaFilter::Catrom;
        let cache = ScaleWeightCache::build(&f, 8, 8, 4, 4);
        let dst = cache.apply(&[]);
        assert_eq!(dst.len(), 16);
        for &v in &dst {
            assert_eq!(v, 0.0);
        }
    }
}
