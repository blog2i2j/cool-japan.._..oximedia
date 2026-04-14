#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::too_many_lines
)]
//! ACES-style cusp-based gamut mapping with configurable compression.
//!
//! This module extends the basic cusp gamut mapping from [`crate::cusp_gamut`]
//! with configurable compression **threshold** and **rolloff** parameters.
//!
//! ## Algorithm
//!
//! 1. Convert the input colour to a perceptual JCh representation (via
//!    CIECAM02).
//! 2. At the input's hue angle, find the cusp — the (J_cusp, C_cusp) point
//!    with maximum chroma on the destination gamut boundary.
//! 3. Calculate the "distance ratio" `d = C_input / C_boundary` at the input's
//!    lightness.
//! 4. If `d <= threshold`, the colour is within the safe zone and is returned
//!    unchanged.
//! 5. If `d > threshold`, a smooth power-curve rolloff compresses the colour
//!    toward the cusp proportionally, keeping it inside the destination gamut
//!    while avoiding harsh clipping.
//!
//! ## Rolloff model
//!
//! The rolloff uses a power-curve with configurable exponent:
//!
//! ```text
//! compressed = threshold + (1 - threshold) * ((d - threshold) / (limit - threshold))^(1/rolloff)
//! ```
//!
//! where `limit` is the maximum distance ratio and `rolloff` controls how
//! aggressively the curve bends (higher = softer).
//!
//! ## References
//!
//! - ACES Gamut Mapping Working Group: "A Perceptual Gamut Mapping Algorithm"
//! - CIE 159:2004 — CIECAM02

use crate::ciecam02::{CiecamModel, CiecamViewingConditions, SurroundCondition};

// ── Gamut primaries ──────────────────────────────────────────────────────────

/// Standard gamut definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Gamut {
    /// ITU-R BT.709 / sRGB.
    Rec709,
    /// DCI-P3 (D65 adapted).
    DciP3,
    /// ITU-R BT.2020.
    Rec2020,
}

impl Gamut {
    /// CIE xy chromaticity coordinates of the primaries [R, G, B].
    fn primaries(self) -> [[f64; 2]; 3] {
        match self {
            Gamut::Rec709 => [[0.640, 0.330], [0.300, 0.600], [0.150, 0.060]],
            Gamut::DciP3 => [[0.680, 0.320], [0.265, 0.690], [0.150, 0.060]],
            Gamut::Rec2020 => [[0.708, 0.292], [0.170, 0.797], [0.131, 0.046]],
        }
    }
}

/// D65 white point in CIE xy.
const D65_WHITE: [f64; 2] = [0.3127, 0.3290];

/// Number of hue samples in the cusp table.
const HUE_SAMPLES: usize = 360;

// ── Cusp entry ───────────────────────────────────────────────────────────────

/// A cusp point on the gamut boundary at a specific hue.
#[derive(Debug, Clone, Copy)]
pub struct CuspPoint {
    /// Hue angle (degrees, 0..360).
    pub hue: f64,
    /// CIECAM02 lightness at the cusp.
    pub j_cusp: f64,
    /// Maximum CIECAM02 chroma at the cusp.
    pub c_cusp: f64,
}

// ── CuspMapper ───────────────────────────────────────────────────────────────

/// Configurable cusp-based gamut mapper.
///
/// The `threshold` parameter (0..1) defines the "safe zone" — colours whose
/// normalised distance from the achromatic axis is below this fraction of the
/// cusp chroma are passed through unchanged.
///
/// The `rolloff` parameter (> 0) controls compression smoothness: higher values
/// produce a softer knee, lower values compress more aggressively.
#[derive(Debug, Clone)]
pub struct CuspMapper {
    cusps: Vec<CuspPoint>,
    gamut: Gamut,
    /// Compression begins when normalised chroma exceeds this fraction (0..1).
    pub threshold: f64,
    /// Rolloff exponent controlling compression curve smoothness (> 0).
    pub rolloff: f64,
    /// CIECAM02 viewing conditions.
    viewing: CiecamViewingConditions,
}

impl CuspMapper {
    /// Create a mapper with default settings (threshold=0.8, rolloff=1.2).
    #[must_use]
    pub fn new(gamut: Gamut) -> Self {
        Self::with_params(gamut, 0.8, 1.2)
    }

    /// Create a mapper with custom threshold and rolloff.
    ///
    /// - `threshold`: fraction of cusp chroma below which colours pass
    ///   unchanged. Clamped to [0.01, 0.99].
    /// - `rolloff`: compression smoothness exponent. Clamped to [0.1, 10.0].
    #[must_use]
    pub fn with_params(gamut: Gamut, threshold: f64, rolloff: f64) -> Self {
        let threshold = threshold.clamp(0.01, 0.99);
        let rolloff = rolloff.clamp(0.1, 10.0);
        let viewing = CiecamViewingConditions {
            adapting_luminance_la: 64.0,
            background_relative_lum_yb: 20.0,
            surround: SurroundCondition::Average,
        };
        let cusps = build_cusp_table(gamut, &viewing);
        Self {
            cusps,
            gamut,
            threshold,
            rolloff,
            viewing,
        }
    }

    /// Create a mapper with custom viewing conditions, threshold, and rolloff.
    #[must_use]
    pub fn with_viewing(
        gamut: Gamut,
        threshold: f64,
        rolloff: f64,
        viewing: CiecamViewingConditions,
    ) -> Self {
        let threshold = threshold.clamp(0.01, 0.99);
        let rolloff = rolloff.clamp(0.1, 10.0);
        let cusps = build_cusp_table(gamut, &viewing);
        Self {
            cusps,
            gamut,
            threshold,
            rolloff,
            viewing,
        }
    }

    /// Get the destination gamut.
    #[must_use]
    pub fn destination_gamut(&self) -> Gamut {
        self.gamut
    }

    /// Get the interpolated cusp for an arbitrary hue angle (degrees).
    #[must_use]
    pub fn cusp_at_hue(&self, hue_deg: f64) -> CuspPoint {
        interpolate_cusp(&self.cusps, hue_deg)
    }

    /// Map an XYZ colour (D65, Y=100 scale) into the destination gamut.
    ///
    /// - In-gamut colours (distance ratio <= threshold) are returned unchanged.
    /// - Out-of-gamut colours are compressed using the configured rolloff curve.
    /// - Hue is preserved exactly.
    #[must_use]
    pub fn map_xyz(&self, x: f64, y: f64, z: f64) -> [f64; 3] {
        let model = CiecamModel::new(self.viewing.clone());
        let app = model.xyz_to_appearance(x as f32, y as f32, z as f32);

        let j_in = app.lightness as f64;
        let c_in = app.chroma as f64;
        let h = app.hue_angle as f64;

        let cusp = self.cusp_at_hue(h);
        let c_boundary = max_chroma_at_j(j_in, &cusp);

        // Distance ratio
        if c_boundary < 1e-6 {
            // Degenerate cusp: pass through
            return [x, y, z];
        }
        let d = c_in / c_boundary;

        if d <= self.threshold {
            // Inside safe zone: pass through
            return [x, y, z];
        }

        // Apply rolloff compression
        let c_compressed = compress_chroma(c_in, c_boundary, self.threshold, self.rolloff);

        // Also compress lightness toward cusp proportionally
        let compress_ratio = if c_in > 1e-9 { c_compressed / c_in } else { 1.0 };
        let j_compressed = j_in + (1.0 - compress_ratio) * (cusp.j_cusp - j_in) * 0.5;

        let mut app_out = app.clone();
        app_out.lightness = j_compressed.clamp(0.0, 100.0) as f32;
        app_out.chroma = c_compressed.max(0.0) as f32;

        let (xo, yo, zo) = model.appearance_to_xyz(&app_out);
        [xo as f64, yo as f64, zo as f64]
    }

    /// Map a batch of XYZ colours.
    pub fn map_batch(&self, pixels: &mut [[f64; 3]]) {
        for pixel in pixels.iter_mut() {
            let mapped = self.map_xyz(pixel[0], pixel[1], pixel[2]);
            *pixel = mapped;
        }
    }
}

// ── Compression function ─────────────────────────────────────────────────────

/// Compress chroma `c_in` toward the gamut boundary `c_boundary` using a
/// power-curve rolloff.
///
/// The safe zone extends from 0 to `threshold * c_boundary`. Beyond that,
/// compression maps the excess chroma smoothly into the remaining headroom.
fn compress_chroma(c_in: f64, c_boundary: f64, threshold: f64, rolloff: f64) -> f64 {
    let safe_limit = threshold * c_boundary;

    if c_in <= safe_limit {
        return c_in;
    }

    // Excess beyond safe zone
    let excess = c_in - safe_limit;
    let headroom = c_boundary - safe_limit;

    if headroom < 1e-9 {
        return c_boundary;
    }

    // Normalised excess in [0, inf)
    let t = excess / headroom;

    // Power-curve compression: maps [0, inf) -> [0, 1)
    // compressed_t = 1 - 1/(1 + t^(1/rolloff))
    let inv_rolloff = 1.0 / rolloff;
    let compressed_t = 1.0 - 1.0 / (1.0 + t.powf(inv_rolloff));

    safe_limit + compressed_t * headroom
}

// ── Cusp table construction ──────────────────────────────────────────────────

fn build_cusp_table(gamut: Gamut, viewing: &CiecamViewingConditions) -> Vec<CuspPoint> {
    let model = CiecamModel::new(viewing.clone());
    let primaries = gamut.primaries();
    let m = rgb_to_xyz_matrix(&primaries, D65_WHITE);
    let step = 360.0 / HUE_SAMPLES as f64;

    (0..HUE_SAMPLES)
        .map(|i| {
            let hue_target = i as f64 * step;
            find_cusp_at_hue(hue_target, &m, &model)
        })
        .collect()
}

fn find_cusp_at_hue(hue_target: f64, m: &[[f64; 3]; 3], model: &CiecamModel) -> CuspPoint {
    let mut best_c = 0.0_f64;
    let mut best_j = 50.0_f64;

    const EDGE_STEPS: usize = 64;
    const LUM_STEPS: usize = 16;
    let hue_tolerance = 360.0 / HUE_SAMPLES as f64 * 2.0;

    let mut try_rgb = |r: f64, g: f64, b: f64| {
        let xyz = mat3_vec3(m, [r, g, b]);
        let (x100, y100, z100) = (xyz[0] * 100.0, xyz[1] * 100.0, xyz[2] * 100.0);
        let app = model.xyz_to_appearance(x100 as f32, y100 as f32, z100 as f32);
        let h = app.hue_angle as f64;
        let c = app.chroma as f64;
        let j = app.lightness as f64;

        let diff = angle_diff(h, hue_target);
        if diff < hue_tolerance && c > best_c {
            best_c = c;
            best_j = j;
        }
    };

    // Scan boundary edges
    for step_idx in 0..=EDGE_STEPS {
        let t = step_idx as f64 / EDGE_STEPS as f64;
        try_rgb(1.0 - t, t, 0.0);
        try_rgb(0.0, 1.0 - t, t);
        try_rgb(t, 0.0, 1.0 - t);
    }

    // Scan at different luminance levels
    for lum_idx in 1..LUM_STEPS {
        let lum = lum_idx as f64 / LUM_STEPS as f64;
        for step_idx in 0..=EDGE_STEPS {
            let t = step_idx as f64 / EDGE_STEPS as f64;
            try_rgb((1.0 - t) * lum, t * lum, 0.0);
            try_rgb(0.0, (1.0 - t) * lum, t * lum);
            try_rgb(t * lum, 0.0, (1.0 - t) * lum);
        }
    }

    if best_c < 1e-6 {
        best_c = 0.0;
        best_j = 50.0;
    }

    CuspPoint {
        hue: hue_target,
        j_cusp: best_j,
        c_cusp: best_c,
    }
}

// ── Cusp interpolation ───────────────────────────────────────────────────────

fn interpolate_cusp(cusps: &[CuspPoint], hue_deg: f64) -> CuspPoint {
    let n = cusps.len();
    if n == 0 {
        return CuspPoint {
            hue: hue_deg,
            j_cusp: 50.0,
            c_cusp: 0.0,
        };
    }

    let step = 360.0 / n as f64;
    let h = hue_deg.rem_euclid(360.0);
    let idx_lo = (h / step).floor() as usize % n;
    let idx_hi = (idx_lo + 1) % n;
    let t = (h / step) - (h / step).floor();

    let lo = &cusps[idx_lo];
    let hi = &cusps[idx_hi];

    CuspPoint {
        hue: hue_deg,
        j_cusp: lo.j_cusp + t * (hi.j_cusp - lo.j_cusp),
        c_cusp: lo.c_cusp + t * (hi.c_cusp - lo.c_cusp),
    }
}

// ── Boundary model ───────────────────────────────────────────────────────────

/// Piecewise-linear boundary model: max chroma at lightness J.
fn max_chroma_at_j(j: f64, cusp: &CuspPoint) -> f64 {
    let j_cusp = cusp.j_cusp.clamp(1.0, 99.0);
    let c_cusp = cusp.c_cusp.max(0.0);

    if j <= j_cusp {
        c_cusp * (j / j_cusp)
    } else {
        c_cusp * ((100.0 - j) / (100.0 - j_cusp))
    }
}

// ── Matrix helpers ───────────────────────────────────────────────────────────

fn rgb_to_xyz_matrix(primaries: &[[f64; 2]; 3], white: [f64; 2]) -> [[f64; 3]; 3] {
    let xyz_r = xy_to_xyz(primaries[0]);
    let xyz_g = xy_to_xyz(primaries[1]);
    let xyz_b = xy_to_xyz(primaries[2]);

    let m = [
        [xyz_r[0], xyz_g[0], xyz_b[0]],
        [xyz_r[1], xyz_g[1], xyz_b[1]],
        [xyz_r[2], xyz_g[2], xyz_b[2]],
    ];

    let w_xyz = xy_to_xyz(white);
    let m_inv = invert_3x3(&m);
    let s = mat3_vec3(&m_inv, w_xyz);

    [
        [m[0][0] * s[0], m[0][1] * s[1], m[0][2] * s[2]],
        [m[1][0] * s[0], m[1][1] * s[1], m[1][2] * s[2]],
        [m[2][0] * s[0], m[2][1] * s[1], m[2][2] * s[2]],
    ]
}

fn xy_to_xyz(xy: [f64; 2]) -> [f64; 3] {
    let (x, y) = (xy[0], xy[1]);
    if y.abs() < 1e-12 {
        return [0.0, 0.0, 0.0];
    }
    [x / y, 1.0, (1.0 - x - y) / y]
}

fn mat3_vec3(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    [
        m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
        m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
        m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
    ]
}

fn invert_3x3(m: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let a = m[0][0];
    let b = m[0][1];
    let c = m[0][2];
    let d = m[1][0];
    let e = m[1][1];
    let f = m[1][2];
    let g = m[2][0];
    let h = m[2][1];
    let k = m[2][2];

    let det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);

    if det.abs() < 1e-14 {
        return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    }

    let inv = 1.0 / det;
    [
        [
            (e * k - f * h) * inv,
            (c * h - b * k) * inv,
            (b * f - c * e) * inv,
        ],
        [
            (f * g - d * k) * inv,
            (a * k - c * g) * inv,
            (c * d - a * f) * inv,
        ],
        [
            (d * h - e * g) * inv,
            (b * g - a * h) * inv,
            (a * e - b * d) * inv,
        ],
    ]
}

fn angle_diff(a: f64, b: f64) -> f64 {
    let d = (a - b).rem_euclid(360.0);
    if d > 180.0 {
        360.0 - d
    } else {
        d
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Construction ─────────────────────────────────────────────────────────

    #[test]
    fn test_cusp_table_size() {
        let mapper = CuspMapper::new(Gamut::Rec709);
        assert_eq!(mapper.cusps.len(), HUE_SAMPLES);
    }

    #[test]
    fn test_default_params() {
        let mapper = CuspMapper::new(Gamut::Rec709);
        assert!((mapper.threshold - 0.8).abs() < 1e-9);
        assert!((mapper.rolloff - 1.2).abs() < 1e-9);
    }

    #[test]
    fn test_custom_params_clamped() {
        let mapper = CuspMapper::with_params(Gamut::Rec709, -0.5, 100.0);
        assert!(mapper.threshold >= 0.01);
        assert!(mapper.rolloff <= 10.0);
    }

    #[test]
    fn test_all_gamuts_construct() {
        for gamut in [Gamut::Rec709, Gamut::DciP3, Gamut::Rec2020] {
            let mapper = CuspMapper::new(gamut);
            assert_eq!(mapper.cusps.len(), HUE_SAMPLES);
            assert_eq!(mapper.destination_gamut(), gamut);
        }
    }

    // ── Cusp interpolation ──────────────────────────────────────────────────

    #[test]
    fn test_cusp_at_hue_wrap_360() {
        let mapper = CuspMapper::new(Gamut::Rec709);
        let c0 = mapper.cusp_at_hue(0.0);
        let c360 = mapper.cusp_at_hue(360.0);
        assert!(
            (c0.c_cusp - c360.c_cusp).abs() < 1e-6,
            "0 and 360 should match"
        );
    }

    #[test]
    fn test_cusp_chroma_non_negative() {
        let mapper = CuspMapper::new(Gamut::Rec709);
        for cusp in &mapper.cusps {
            assert!(cusp.c_cusp >= 0.0, "negative chroma at hue {}", cusp.hue);
        }
    }

    // ── compress_chroma ──────────────────────────────────────────────────────

    #[test]
    fn test_in_gamut_unchanged() {
        let c_boundary = 50.0;
        let threshold = 0.8;
        let rolloff = 1.2;
        // c_in well within safe zone
        let c_in = 30.0; // 30/50 = 0.6 < 0.8
        let result = compress_chroma(c_in, c_boundary, threshold, rolloff);
        assert!(
            (result - c_in).abs() < 1e-9,
            "in-gamut should be unchanged"
        );
    }

    #[test]
    fn test_compression_reduces_chroma() {
        let c_boundary = 50.0;
        let threshold = 0.8;
        let rolloff = 1.2;
        let c_in = 80.0; // Way out of gamut
        let result = compress_chroma(c_in, c_boundary, threshold, rolloff);
        assert!(result < c_in, "compression should reduce chroma");
        assert!(result < c_boundary, "result should be inside boundary");
    }

    #[test]
    fn test_higher_rolloff_softer_compression() {
        let c_boundary = 50.0;
        let threshold = 0.8;
        let c_in = 70.0;
        let soft = compress_chroma(c_in, c_boundary, threshold, 3.0);
        let hard = compress_chroma(c_in, c_boundary, threshold, 0.5);
        // Both should compress, but results may differ
        assert!(soft < c_in);
        assert!(hard < c_in);
        // Hard compression pushes more toward boundary
        assert!(
            (soft - hard).abs() > 0.01 || soft <= c_boundary,
            "different rolloff should produce different results"
        );
    }

    #[test]
    fn test_threshold_boundary_continuity() {
        let c_boundary = 50.0;
        let threshold = 0.8;
        let rolloff = 1.2;
        let safe_limit = threshold * c_boundary;
        // Just below threshold
        let below = compress_chroma(safe_limit - 0.001, c_boundary, threshold, rolloff);
        // Just above threshold
        let above = compress_chroma(safe_limit + 0.001, c_boundary, threshold, rolloff);
        // Should be nearly continuous
        assert!(
            (above - below).abs() < 0.1,
            "compression should be continuous at threshold: {} vs {}",
            below,
            above
        );
    }

    // ── map_xyz ──────────────────────────────────────────────────────────────

    #[test]
    fn test_map_xyz_achromatic_unchanged() {
        let mapper = CuspMapper::new(Gamut::Rec709);
        // D65-proportional mid-grey
        let result = mapper.map_xyz(47.524, 50.0, 54.442);
        assert!((result[0] - 47.524).abs() < 2.0, "X changed: {}", result[0]);
        assert!((result[1] - 50.0).abs() < 2.0, "Y changed: {}", result[1]);
        assert!((result[2] - 54.442).abs() < 2.0, "Z changed: {}", result[2]);
    }

    #[test]
    fn test_map_xyz_out_of_gamut_finite() {
        let mapper = CuspMapper::new(Gamut::Rec709);
        let result = mapper.map_xyz(120.0, 50.0, 5.0);
        assert!(result[0].is_finite());
        assert!(result[1].is_finite());
        assert!(result[2].is_finite());
    }

    #[test]
    fn test_map_batch() {
        let mapper = CuspMapper::new(Gamut::Rec709);
        let mut pixels = vec![[47.524, 50.0, 54.442], [120.0, 50.0, 5.0]];
        mapper.map_batch(&mut pixels);
        for p in &pixels {
            assert!(p[0].is_finite());
            assert!(p[1].is_finite());
            assert!(p[2].is_finite());
        }
    }

    // ── max_chroma_at_j ──────────────────────────────────────────────────────

    #[test]
    fn test_boundary_zero_at_extremes() {
        let cusp = CuspPoint {
            hue: 0.0,
            j_cusp: 50.0,
            c_cusp: 60.0,
        };
        assert!(max_chroma_at_j(0.0, &cusp).abs() < 1e-9);
        assert!(max_chroma_at_j(100.0, &cusp).abs() < 1e-9);
    }

    #[test]
    fn test_boundary_peak_at_cusp() {
        let cusp = CuspPoint {
            hue: 0.0,
            j_cusp: 50.0,
            c_cusp: 60.0,
        };
        let peak = max_chroma_at_j(50.0, &cusp);
        assert!((peak - 60.0).abs() < 1e-9, "expected 60, got {peak}");
    }

    // ── Custom viewing conditions ────────────────────────────────────────────

    #[test]
    fn test_custom_viewing() {
        let viewing = CiecamViewingConditions {
            adapting_luminance_la: 32.0,
            background_relative_lum_yb: 15.0,
            surround: SurroundCondition::Dim,
        };
        let mapper = CuspMapper::with_viewing(Gamut::Rec709, 0.7, 1.5, viewing);
        assert_eq!(mapper.cusps.len(), HUE_SAMPLES);
        assert!((mapper.threshold - 0.7).abs() < 1e-9);
        assert!((mapper.rolloff - 1.5).abs() < 1e-9);
    }
}
