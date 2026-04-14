//! Multi-illuminant chromatic adaptation beyond Bradford and Von Kries.
//!
//! Standard chromatic adaptation transforms (CAT) handle a single source
//! illuminant → single destination illuminant pair.  Real-world scenarios often
//! involve mixed or uncertain illuminants (e.g. studio + daylight, fluorescent
//! haze on a tungsten set, or adaptively detected scene illumination).
//!
//! This module provides:
//!
//! - **`IlluminantBlend`** — weighted mixture of up to N CIE standard illuminants
//!   yielding a composite white point.
//! - **`IlluminantEstimate`** — probabilistic scene illuminant hypothesis with
//!   confidence weight, suitable for Bayesian or voting-based AWB fusion.
//! - **`MultiIlluminantCat`** — performs chromatic adaptation from a blended
//!   source to a target using any of five CAT methods (Bradford, Von Kries, CAT02,
//!   Sharp, CMCCAT2000).
//! - **`IlluminantSequence`** — smooth temporal blending across a sequence of
//!   illuminant estimates for flicker-free adaptation in video.
//! - **`IlluminantClassifier`** — lightweight heuristic classifier that maps a
//!   CIE XYZ white point to the most likely standard CIE illuminant category.
//!
//! # References
//!
//! - Finlayson & Süsstrunk, "Spectral sharpening and the Bradford transform",
//!   Proc. IS&T/SID 8th Color Imaging Conf., 2000.
//! - Li et al., "CMC-CAT2000: A chromatic adaptation transform",
//!   Color Research & Application 27(1), 2002.
//! - CIE 160:2004, "A review of chromatic adaptation transforms".

use crate::error::{ColorError, Result};

// ── CIE standard illuminant white points (XYZ, Y = 1) ──────────────────────

/// CIE standard illuminant identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Illuminant {
    /// CIE Illuminant A — incandescent / tungsten (~2856 K).
    A,
    /// CIE Illuminant B — direct noon sunlight (~4874 K, deprecated).
    B,
    /// CIE Illuminant C — average daylight with UV cut (~6774 K, deprecated).
    C,
    /// CIE Illuminant D50 — horizon daylight (ICC profiles, print).
    D50,
    /// CIE Illuminant D55 — mid-morning daylight.
    D55,
    /// CIE Illuminant D65 — average daylight (sRGB/Rec.709 reference).
    D65,
    /// CIE Illuminant D75 — north sky daylight.
    D75,
    /// CIE Illuminant E — equal energy (hypothetical equal-power illuminant).
    E,
    /// CIE Illuminant F2 — cool white fluorescent.
    F2,
    /// CIE Illuminant F7 — broadband daylight fluorescent.
    F7,
    /// CIE Illuminant F11 — triband narrow-band fluorescent.
    F11,
    /// CIE Illuminant D60 — ACES reference.
    D60,
}

/// XYZ white point for a CIE standard illuminant (Y = 1.0).
///
/// Values from CIE 15:2004, 2° Standard Observer.
#[must_use]
pub fn illuminant_xyz(illuminant: Illuminant) -> [f64; 3] {
    match illuminant {
        Illuminant::A   => [1.09850, 1.00000, 0.35585],
        Illuminant::B   => [0.99072, 1.00000, 0.85223],
        Illuminant::C   => [0.98074, 1.00000, 1.18232],
        Illuminant::D50 => [0.96422, 1.00000, 0.82521],
        Illuminant::D55 => [0.95682, 1.00000, 0.92149],
        Illuminant::D65 => [0.95047, 1.00000, 1.08883],
        Illuminant::D75 => [0.94972, 1.00000, 1.22638],
        Illuminant::E   => [1.00000, 1.00000, 1.00000],
        Illuminant::F2  => [0.99186, 1.00000, 0.67393],
        Illuminant::F7  => [0.95041, 1.00000, 1.08747],
        Illuminant::F11 => [1.00962, 1.00000, 0.64350],
        Illuminant::D60 => [0.95265, 1.00000, 1.00883],
    }
}

/// Returns the nominal colour temperature in Kelvin for a CIE illuminant.
#[must_use]
pub fn illuminant_cct(illuminant: Illuminant) -> f64 {
    match illuminant {
        Illuminant::A   => 2856.0,
        Illuminant::B   => 4874.0,
        Illuminant::C   => 6774.0,
        Illuminant::D50 => 5003.0,
        Illuminant::D55 => 5503.0,
        Illuminant::D65 => 6504.0,
        Illuminant::D75 => 7504.0,
        Illuminant::E   => 5455.0,
        Illuminant::F2  => 4230.0,
        Illuminant::F7  => 6500.0,
        Illuminant::F11 => 4000.0,
        Illuminant::D60 => 6004.0,
    }
}

// ── IlluminantBlend ──────────────────────────────────────────────────────────

/// A weighted mixture of multiple CIE standard illuminants.
///
/// Models mixed-illuminant scenes (e.g. indoor window light + tungsten).
/// The composite white point is computed as the normalised weighted sum of
/// individual XYZ white points.
#[derive(Debug, Clone)]
pub struct IlluminantBlend {
    entries: Vec<(Illuminant, f64)>,
}

impl IlluminantBlend {
    /// Creates an empty blend.
    #[must_use]
    pub fn new() -> Self {
        Self { entries: Vec::new() }
    }

    /// Adds an illuminant with the given weight (clamped to [0, 1]).
    pub fn add(&mut self, illuminant: Illuminant, weight: f64) {
        self.entries.push((illuminant, weight.clamp(0.0, 1.0)));
    }

    /// Returns the number of illuminant entries.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if no illuminants have been added.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Computes the composite white point (XYZ, Y ≈ 1).
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::InvalidColor`] if the blend is empty or the
    /// total weight is zero.
    pub fn composite_xyz(&self) -> Result<[f64; 3]> {
        if self.entries.is_empty() {
            return Err(ColorError::InvalidColor(
                "IlluminantBlend: no illuminants added".into(),
            ));
        }
        let total_weight: f64 = self.entries.iter().map(|(_, w)| *w).sum();
        if total_weight < 1e-12 {
            return Err(ColorError::InvalidColor(
                "IlluminantBlend: total weight is zero".into(),
            ));
        }

        let mut sum = [0.0_f64; 3];
        for &(illum, weight) in &self.entries {
            let xyz = illuminant_xyz(illum);
            sum[0] += xyz[0] * weight;
            sum[1] += xyz[1] * weight;
            sum[2] += xyz[2] * weight;
        }

        Ok([
            sum[0] / total_weight,
            sum[1] / total_weight,
            sum[2] / total_weight,
        ])
    }

    /// Computes the weighted average CCT of the blend (arithmetic in reciprocal
    /// Mired space for perceptual linearity).
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::InvalidColor`] if the blend is empty or total weight is zero.
    pub fn composite_cct(&self) -> Result<f64> {
        if self.entries.is_empty() {
            return Err(ColorError::InvalidColor(
                "IlluminantBlend: no illuminants added".into(),
            ));
        }
        let total_weight: f64 = self.entries.iter().map(|(_, w)| *w).sum();
        if total_weight < 1e-12 {
            return Err(ColorError::InvalidColor(
                "IlluminantBlend: total weight is zero".into(),
            ));
        }

        // Interpolate in reciprocal Mired space (1/T) for perceptual uniformity
        let sum_inv_t: f64 = self
            .entries
            .iter()
            .map(|&(ill, w)| w / illuminant_cct(ill))
            .sum();

        Ok(total_weight / sum_inv_t)
    }
}

impl Default for IlluminantBlend {
    fn default() -> Self {
        Self::new()
    }
}

// ── IlluminantEstimate ────────────────────────────────────────────────────────

/// A probabilistic scene illuminant hypothesis.
///
/// Used in voting-based or Bayesian AWB pipelines where multiple algorithms
/// each produce an estimate with an associated confidence.
#[derive(Debug, Clone)]
pub struct IlluminantEstimate {
    /// Estimated XYZ white point.
    pub xyz: [f64; 3],
    /// Confidence weight ∈ [0, 1].
    pub confidence: f64,
    /// Optional human-readable label.
    pub label: Option<String>,
}

impl IlluminantEstimate {
    /// Creates a new estimate.
    ///
    /// # Arguments
    ///
    /// * `xyz` - Estimated white point (Y should be ≈ 1 for normalised XYZ).
    /// * `confidence` - Confidence weight, clamped to `[0, 1]`.
    /// * `label` - Optional descriptor.
    #[must_use]
    pub fn new(xyz: [f64; 3], confidence: f64, label: Option<String>) -> Self {
        Self {
            xyz,
            confidence: confidence.clamp(0.0, 1.0),
            label,
        }
    }

    /// Creates an estimate from a CIE standard illuminant.
    #[must_use]
    pub fn from_illuminant(illuminant: Illuminant, confidence: f64) -> Self {
        Self::new(
            illuminant_xyz(illuminant),
            confidence,
            Some(format!("{illuminant:?}")),
        )
    }
}

/// Fuses multiple illuminant estimates into a single weighted-mean XYZ.
///
/// The fusion is a confidence-weighted average in XYZ space (no gamut mapping).
///
/// # Errors
///
/// Returns [`ColorError::InvalidColor`] if `estimates` is empty or total
/// confidence is zero.
pub fn fuse_estimates(estimates: &[IlluminantEstimate]) -> Result<[f64; 3]> {
    if estimates.is_empty() {
        return Err(ColorError::InvalidColor(
            "fuse_estimates: no estimates provided".into(),
        ));
    }
    let total: f64 = estimates.iter().map(|e| e.confidence).sum();
    if total < 1e-12 {
        return Err(ColorError::InvalidColor(
            "fuse_estimates: total confidence is zero".into(),
        ));
    }

    let mut sum = [0.0_f64; 3];
    for est in estimates {
        sum[0] += est.xyz[0] * est.confidence;
        sum[1] += est.xyz[1] * est.confidence;
        sum[2] += est.xyz[2] * est.confidence;
    }

    Ok([sum[0] / total, sum[1] / total, sum[2] / total])
}

// ── CAT methods ──────────────────────────────────────────────────────────────

/// Chromatic adaptation transform method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CatMethod {
    /// Bradford (ICC recommended).
    Bradford,
    /// Von Kries (classical diagonal).
    VonKries,
    /// CAT02 (CIECAM02 standard).
    Cat02,
    /// Sharp (Finlayson & Süsstrunk 2000 — spectrally sharpened).
    Sharp,
    /// CMCCAT2000 (Li et al. 2002 — CMC model).
    CmcCat2000,
}

/// 3 × 3 cone-space forward transform matrix for each CAT method.
fn cat_forward_matrix(method: CatMethod) -> [[f64; 3]; 3] {
    match method {
        CatMethod::Bradford => [
            [ 0.8951,  0.2664, -0.1614],
            [-0.7502,  1.7135,  0.0367],
            [ 0.0389, -0.0685,  1.0296],
        ],
        CatMethod::VonKries => [
            [0.40024,  0.70760, -0.08081],
            [-0.22630, 1.16532,  0.04570],
            [0.00000,  0.00000,  0.91822],
        ],
        CatMethod::Cat02 => [
            [ 0.7328,  0.4296, -0.1624],
            [-0.7036,  1.6975,  0.0061],
            [ 0.0030,  0.0136,  0.9834],
        ],
        CatMethod::Sharp => [
            [ 1.2694, -0.0988, -0.1706],
            [-0.8364,  1.8006,  0.0357],
            [ 0.0297, -0.0315,  1.0018],
        ],
        CatMethod::CmcCat2000 => [
            [ 0.7982,  0.3389, -0.1371],
            [-0.5918,  1.5512,  0.0406],
            [ 0.0008,  0.0239,  0.9753],
        ],
    }
}

/// Multiplies a 3 × 3 matrix by a column vector.
fn mat3_mul_vec(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

/// Multiplies two 3 × 3 matrices.
fn mat3_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut c = [[0.0_f64; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
    c
}

/// Inverts a 3 × 3 matrix (Cramer's rule).
fn mat3_inv(m: &[[f64; 3]; 3]) -> Result<[[f64; 3]; 3]> {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);

    if det.abs() < 1e-15 {
        return Err(ColorError::Matrix(
            "mat3_inv: matrix is singular".into(),
        ));
    }
    let d = 1.0 / det;
    Ok([
        [
            (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * d,
            (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * d,
            (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * d,
        ],
        [
            (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * d,
            (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * d,
            (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * d,
        ],
        [
            (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * d,
            (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * d,
            (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * d,
        ],
    ])
}

// ── MultiIlluminantCat ────────────────────────────────────────────────────────

/// Chromatic adaptation transform from a (possibly blended) source illuminant
/// to a destination illuminant.
///
/// Supports five CAT cone-space models and accepts composite white points from
/// [`IlluminantBlend`] or custom XYZ values.
pub struct MultiIlluminantCat {
    /// Full 3 × 3 XYZ → XYZ adaptation matrix.
    pub matrix: [[f64; 3]; 3],
}

impl MultiIlluminantCat {
    /// Builds a CAT from explicit XYZ white points.
    ///
    /// # Arguments
    ///
    /// * `src_xyz` - Source white point XYZ (e.g. scene illuminant).
    /// * `dst_xyz` - Destination white point XYZ (e.g. D65).
    /// * `method`  - Cone-space transform method.
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::Matrix`] if the cone-space matrix or its inverse
    /// is singular (pathological inputs only).
    pub fn from_xyz(
        src_xyz: [f64; 3],
        dst_xyz: [f64; 3],
        method: CatMethod,
    ) -> Result<Self> {
        let fwd = cat_forward_matrix(method);
        let inv = mat3_inv(&fwd)?;

        // LMS of source and destination white points
        let src_lms = mat3_mul_vec(&fwd, src_xyz);
        let dst_lms = mat3_mul_vec(&fwd, dst_xyz);

        // Build diagonal gain matrix
        let diag = if src_lms[0].abs() < 1e-12 || src_lms[1].abs() < 1e-12 || src_lms[2].abs() < 1e-12 {
            return Err(ColorError::Matrix(
                "MultiIlluminantCat: source LMS has near-zero component".into(),
            ));
        } else {
            [
                [dst_lms[0] / src_lms[0], 0.0, 0.0],
                [0.0, dst_lms[1] / src_lms[1], 0.0],
                [0.0, 0.0, dst_lms[2] / src_lms[2]],
            ]
        };

        // Full matrix: M_inv * D * M
        let dm = mat3_mul(&diag, &fwd);
        let matrix = mat3_mul(&inv, &dm);

        Ok(Self { matrix })
    }

    /// Builds a CAT from standard illuminant identifiers.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Self::from_xyz`].
    pub fn from_illuminants(
        src: Illuminant,
        dst: Illuminant,
        method: CatMethod,
    ) -> Result<Self> {
        Self::from_xyz(illuminant_xyz(src), illuminant_xyz(dst), method)
    }

    /// Builds a CAT from a blended source illuminant.
    ///
    /// # Errors
    ///
    /// Returns errors from [`IlluminantBlend::composite_xyz`] or matrix construction.
    pub fn from_blend(
        src_blend: &IlluminantBlend,
        dst: Illuminant,
        method: CatMethod,
    ) -> Result<Self> {
        let src_xyz = src_blend.composite_xyz()?;
        Self::from_xyz(src_xyz, illuminant_xyz(dst), method)
    }

    /// Adapts an XYZ colour using the pre-computed matrix.
    ///
    /// Input and output are in the same XYZ space (Y = 1 for reference white).
    #[must_use]
    pub fn adapt(&self, xyz: [f64; 3]) -> [f64; 3] {
        mat3_mul_vec(&self.matrix, xyz)
    }

    /// Adapts a batch of XYZ pixels in-place.
    pub fn adapt_batch(&self, pixels: &mut [[f64; 3]]) {
        for px in pixels.iter_mut() {
            *px = self.adapt(*px);
        }
    }
}

// ── IlluminantSequence ────────────────────────────────────────────────────────

/// Smooth temporal blending of illuminant estimates for flicker-free video AWB.
///
/// Uses exponential moving average (EMA) to low-pass filter the illuminant
/// estimate across frames.
pub struct IlluminantSequence {
    /// Current smoothed XYZ estimate.
    current: [f64; 3],
    /// EMA smoothing coefficient α ∈ (0, 1]. Higher = faster response.
    alpha: f64,
}

impl IlluminantSequence {
    /// Creates a sequence starting at the given XYZ estimate.
    ///
    /// # Arguments
    ///
    /// * `initial_xyz` - Starting white point estimate.
    /// * `alpha` - EMA coefficient (0.0 = frozen, 1.0 = no smoothing).
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::InvalidColor`] if `alpha` is outside (0, 1].
    pub fn new(initial_xyz: [f64; 3], alpha: f64) -> Result<Self> {
        if alpha <= 0.0 || alpha > 1.0 {
            return Err(ColorError::InvalidColor(format!(
                "IlluminantSequence: alpha must be in (0, 1], got {alpha}"
            )));
        }
        Ok(Self {
            current: initial_xyz,
            alpha,
        })
    }

    /// Updates the running estimate with a new frame's white point measurement.
    ///
    /// Returns the new smoothed XYZ estimate.
    pub fn update(&mut self, new_xyz: [f64; 3]) -> [f64; 3] {
        for i in 0..3 {
            self.current[i] = self.alpha * new_xyz[i] + (1.0 - self.alpha) * self.current[i];
        }
        self.current
    }

    /// Returns the current smoothed XYZ estimate without updating.
    #[must_use]
    pub fn current(&self) -> [f64; 3] {
        self.current
    }

    /// Resets the sequence to a new XYZ estimate (hard cut / scene change).
    pub fn reset(&mut self, xyz: [f64; 3]) {
        self.current = xyz;
    }
}

// ── IlluminantClassifier ──────────────────────────────────────────────────────

/// Classification result from [`IlluminantClassifier`].
#[derive(Debug, Clone, PartialEq)]
pub struct ClassificationResult {
    /// Best-matching CIE standard illuminant.
    pub illuminant: Illuminant,
    /// XYZ distance to the matched illuminant white point.
    pub distance: f64,
    /// Estimated CCT in Kelvin.
    pub estimated_cct: f64,
}

/// Classifies a measured XYZ white point as the nearest CIE standard illuminant.
///
/// Uses Euclidean distance in XYZ space.  For a more perceptually accurate
/// match, callers can convert to Lab first and use ΔE.
pub struct IlluminantClassifier {
    candidates: Vec<Illuminant>,
}

impl IlluminantClassifier {
    /// Creates a classifier with all known CIE illuminants.
    #[must_use]
    pub fn new() -> Self {
        Self {
            candidates: vec![
                Illuminant::A,
                Illuminant::B,
                Illuminant::C,
                Illuminant::D50,
                Illuminant::D55,
                Illuminant::D60,
                Illuminant::D65,
                Illuminant::D75,
                Illuminant::E,
                Illuminant::F2,
                Illuminant::F7,
                Illuminant::F11,
            ],
        }
    }

    /// Creates a classifier with a custom set of candidate illuminants.
    #[must_use]
    pub fn with_candidates(candidates: Vec<Illuminant>) -> Self {
        Self { candidates }
    }

    /// Classifies the given XYZ white point.
    ///
    /// # Errors
    ///
    /// Returns [`ColorError::InvalidColor`] if no candidates have been added.
    pub fn classify(&self, xyz: [f64; 3]) -> Result<ClassificationResult> {
        if self.candidates.is_empty() {
            return Err(ColorError::InvalidColor(
                "IlluminantClassifier: no candidate illuminants".into(),
            ));
        }

        let mut best_dist = f64::MAX;
        let mut best_illum = self.candidates[0];

        for &illum in &self.candidates {
            let ref_xyz = illuminant_xyz(illum);
            let dist = xyz_distance(xyz, ref_xyz);
            if dist < best_dist {
                best_dist = dist;
                best_illum = illum;
            }
        }

        Ok(ClassificationResult {
            illuminant: best_illum,
            distance: best_dist,
            estimated_cct: illuminant_cct(best_illum),
        })
    }
}

impl Default for IlluminantClassifier {
    fn default() -> Self {
        Self::new()
    }
}

fn xyz_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    let dz = a[2] - b[2];
    (dx * dx + dy * dy + dz * dz).sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── illuminant_xyz / illuminant_cct ───────────────────────────────────────

    #[test]
    fn illuminant_xyz_d65_known_values() {
        let xyz = illuminant_xyz(Illuminant::D65);
        assert!((xyz[0] - 0.95047).abs() < 1e-4);
        assert!((xyz[1] - 1.0).abs() < 1e-6);
        assert!((xyz[2] - 1.08883).abs() < 1e-4);
    }

    #[test]
    fn illuminant_xyz_equal_energy_is_unit() {
        let xyz = illuminant_xyz(Illuminant::E);
        for v in xyz {
            assert!((v - 1.0).abs() < 1e-9, "E illuminant: {v}");
        }
    }

    #[test]
    fn illuminant_cct_order() {
        // A < D50 < D65 < D75
        assert!(illuminant_cct(Illuminant::A) < illuminant_cct(Illuminant::D50));
        assert!(illuminant_cct(Illuminant::D50) < illuminant_cct(Illuminant::D65));
        assert!(illuminant_cct(Illuminant::D65) < illuminant_cct(Illuminant::D75));
    }

    // ── IlluminantBlend ───────────────────────────────────────────────────────

    #[test]
    fn blend_empty_returns_error() {
        let blend = IlluminantBlend::new();
        assert!(blend.composite_xyz().is_err());
        assert!(blend.composite_cct().is_err());
    }

    #[test]
    fn blend_single_illuminant_is_passthrough() {
        let mut blend = IlluminantBlend::new();
        blend.add(Illuminant::D65, 1.0);
        let xyz = blend.composite_xyz().unwrap();
        let ref_xyz = illuminant_xyz(Illuminant::D65);
        for i in 0..3 {
            assert!((xyz[i] - ref_xyz[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn blend_equal_weights_midpoint() {
        let mut blend = IlluminantBlend::new();
        blend.add(Illuminant::D50, 1.0);
        blend.add(Illuminant::D65, 1.0);
        let xyz = blend.composite_xyz().unwrap();
        let d50 = illuminant_xyz(Illuminant::D50);
        let d65 = illuminant_xyz(Illuminant::D65);
        for i in 0..3 {
            let mid = (d50[i] + d65[i]) / 2.0;
            assert!((xyz[i] - mid).abs() < 1e-9, "ch {i}: {} vs {}", xyz[i], mid);
        }
    }

    #[test]
    fn blend_composite_cct_between_extremes() {
        let mut blend = IlluminantBlend::new();
        blend.add(Illuminant::A, 1.0);   // ~2856 K
        blend.add(Illuminant::D75, 1.0); // ~7504 K
        let cct = blend.composite_cct().unwrap();
        assert!(cct > 2856.0 && cct < 7504.0, "cct={cct}");
    }

    // ── fuse_estimates ────────────────────────────────────────────────────────

    #[test]
    fn fuse_single_estimate() {
        let est = IlluminantEstimate::from_illuminant(Illuminant::D65, 1.0);
        let fused = fuse_estimates(&[est]).unwrap();
        let ref_xyz = illuminant_xyz(Illuminant::D65);
        for i in 0..3 {
            assert!((fused[i] - ref_xyz[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn fuse_empty_returns_error() {
        assert!(fuse_estimates(&[]).is_err());
    }

    #[test]
    fn fuse_zero_confidence_returns_error() {
        let est = IlluminantEstimate::new([0.95, 1.0, 1.09], 0.0, None);
        assert!(fuse_estimates(&[est]).is_err());
    }

    // ── MultiIlluminantCat ────────────────────────────────────────────────────

    #[test]
    fn cat_identity_when_src_eq_dst() {
        for method in [CatMethod::Bradford, CatMethod::Cat02, CatMethod::Sharp] {
            let cat = MultiIlluminantCat::from_illuminants(
                Illuminant::D65,
                Illuminant::D65,
                method,
            )
            .unwrap();

            let xyz = [0.5, 0.6, 0.7];
            let adapted = cat.adapt(xyz);
            for i in 0..3 {
                assert!(
                    (adapted[i] - xyz[i]).abs() < 1e-9,
                    "{method:?} ch {i}: {} vs {}",
                    adapted[i],
                    xyz[i]
                );
            }
        }
    }

    #[test]
    fn cat_from_blend_d50_d65_adapts_white() {
        let mut blend = IlluminantBlend::new();
        blend.add(Illuminant::D50, 1.0);
        let cat = MultiIlluminantCat::from_blend(&blend, Illuminant::D65, CatMethod::Bradford)
            .unwrap();

        // D50 white adapted to D65 should be close to D65 white
        let d50_xyz = illuminant_xyz(Illuminant::D50);
        let adapted = cat.adapt(d50_xyz);
        let d65_xyz = illuminant_xyz(Illuminant::D65);

        for i in 0..3 {
            assert!(
                (adapted[i] - d65_xyz[i]).abs() < 0.02,
                "ch {i}: {} vs {}",
                adapted[i],
                d65_xyz[i]
            );
        }
    }

    // ── IlluminantSequence ────────────────────────────────────────────────────

    #[test]
    fn sequence_alpha1_is_no_smoothing() {
        let mut seq =
            IlluminantSequence::new(illuminant_xyz(Illuminant::D50), 1.0).unwrap();
        let new_xyz = illuminant_xyz(Illuminant::D65);
        let result = seq.update(new_xyz);
        for i in 0..3 {
            assert!((result[i] - new_xyz[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn sequence_alpha0_is_invalid() {
        assert!(IlluminantSequence::new(illuminant_xyz(Illuminant::D65), 0.0).is_err());
    }

    #[test]
    fn sequence_smoothing_converges() {
        let start = illuminant_xyz(Illuminant::A); // 2856 K
        let target = illuminant_xyz(Illuminant::D65);
        let mut seq = IlluminantSequence::new(start, 0.2).unwrap();

        for _ in 0..100 {
            seq.update(target);
        }
        let cur = seq.current();
        for i in 0..3 {
            assert!(
                (cur[i] - target[i]).abs() < 1e-4,
                "ch {i}: {} vs {}",
                cur[i],
                target[i]
            );
        }
    }

    // ── IlluminantClassifier ──────────────────────────────────────────────────

    #[test]
    fn classify_d65_exactly() {
        let clf = IlluminantClassifier::new();
        let xyz = illuminant_xyz(Illuminant::D65);
        let result = clf.classify(xyz).unwrap();
        assert_eq!(result.illuminant, Illuminant::D65);
        assert!(result.distance < 1e-9);
    }

    #[test]
    fn classify_near_a() {
        let clf = IlluminantClassifier::new();
        // Slightly perturbed A illuminant
        let xyz = illuminant_xyz(Illuminant::A);
        let perturbed = [xyz[0] + 0.001, xyz[1], xyz[2] - 0.001];
        let result = clf.classify(perturbed).unwrap();
        assert_eq!(result.illuminant, Illuminant::A);
    }

    #[test]
    fn classify_empty_candidates_errors() {
        let clf = IlluminantClassifier::with_candidates(vec![]);
        assert!(clf.classify([0.95, 1.0, 1.09]).is_err());
    }
}
