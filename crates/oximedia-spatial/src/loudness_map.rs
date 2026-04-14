//! Spatial loudness map — per-direction energy distribution.
//!
//! Given a Higher-Order Ambisonics (HOA) sound field, this module samples the
//! instantaneous or time-averaged energy across a grid of directions on the
//! sphere and returns:
//!
//! - A [`LoudnessMap`]: a discretised energy map indexed by (azimuth, elevation).
//! - Sweet-spot analysis: the direction of maximum energy and the half-power
//!   angular spread around it.
//! - Polar plot data: 1-D slices through the horizontal or median plane.
//!
//! # Algorithm
//!
//! The energy in direction `(θ, φ)` is approximated by evaluating the real
//! spherical harmonics at that direction, forming a decoding vector **d**, and
//! computing `E(θ, φ) = **d**ᵀ **B** **d**` where **B** is the order-averaged
//! B-format covariance matrix (or simply the instantaneous amplitude via the
//! ambisonic decode `p = **d**ᵀ **a**` and `E = p²`).
//!
//! For time-averaged maps the covariance matrix `**B** = (1/T) Σ **a_t** **a_t**ᵀ`
//! is accumulated over the input frames.
//!
//! # References
//! Poletti, M. A. (2005). "Three-dimensional surround sound systems based on
//! spherical harmonics." *JAES* 53(11), 1004–1025.

use crate::SpatialError;

// ─── Spherical harmonic evaluation ───────────────────────────────────────────

/// Evaluate real, SN3D-normalised spherical harmonics up to order `max_order`
/// at direction (azimuth θ, elevation φ), both in **radians**.
///
/// Returns a `Vec<f32>` of length `(max_order+1)²` in ACN order.
pub fn eval_sh(max_order: usize, azimuth: f32, elevation: f32) -> Vec<f32> {
    let n_ch = (max_order + 1) * (max_order + 1);
    let mut sh = vec![0.0_f32; n_ch];

    let _cos_el = elevation.cos();
    let sin_el = elevation.sin();

    for l in 0..=(max_order as i32) {
        // Compute associated Legendre polynomials P_l^m(sin_el) via recurrence.
        // We use the Schmidt semi-normalised form and convert to SN3D.
        let plm = assoc_legendre_sn3d(l, sin_el);

        for m in -l..=l {
            let acn = (l * l + l + m) as usize;
            let p = plm[(m + l) as usize];
            let angle_factor = if m > 0 {
                (m as f32 * azimuth).cos()
            } else if m < 0 {
                ((-m) as f32 * azimuth).sin()
            } else {
                1.0
            };
            sh[acn] = p * angle_factor;
        }
    }
    sh
}

/// Compute Schmidt semi-normalised associated Legendre polynomials
/// `P_l^m(x)` for m = [-l, l], returning a Vec of length 2l+1.
///
/// The SN3D normalisation used here matches the Ambisonics SN3D convention:
/// `Y_l^m(θ,φ) = P_l^{|m|}(sin φ) * trig(m, θ)` where
/// `trig(m>0)=cos(mθ), trig(m<0)=sin(|m|θ), trig(0)=1`.
fn assoc_legendre_sn3d(l: i32, x: f32) -> Vec<f32> {
    let size = (2 * l + 1) as usize;
    let mut plm = vec![0.0_f32; size];

    // Compute P_l^m for m = 0..l using standard recurrence, then fill ±m.
    // Recurrence: P_l^m via sectorial (P_m^m) and then zonal/tesseral.

    let sqrt_1mx2 = (1.0 - x * x).max(0.0).sqrt();

    // P_0^0 = 1
    // P_m^m = -(2m-1) * sqrt(1-x²) * P_{m-1}^{m-1}  (sectorial)
    let mut p_mm_prev = 1.0_f32; // P_0^0
    let mut p_mm = [1.0_f32; 8]; // P_m^m for m=0..7 (l ≤ 5 means m ≤ 5)
    p_mm[0] = 1.0;
    for m in 1..=l {
        p_mm[m as usize] = -(2 * m - 1) as f32 * sqrt_1mx2 * p_mm_prev;
        p_mm_prev = p_mm[m as usize];
    }

    // Now compute P_l^m(x) for fixed l, all m=0..l via the recurrence:
    // (l-m)*P_l^m = x*(2l-1)*P_{l-1}^m - (l+m-1)*P_{l-2}^m
    // We only need P_l^m for a specific l; use upward recurrence in l.
    let mut buf = vec![[0.0_f32; 6]; (l + 1) as usize]; // buf[l_idx][m_idx]
    for m in 0..=l {
        // Seed: P_m^m
        buf[m as usize][m as usize] = p_mm[m as usize];
        if m < l {
            // P_{m+1}^m = x * (2m+1) * P_m^m
            buf[(m + 1) as usize][m as usize] = x * (2 * m + 1) as f32 * p_mm[m as usize];
        }
        // Recur upward in l_idx from m+2 to l
        for li in (m + 2)..=l {
            let lf = li as f32;
            let mf = m as f32;
            buf[li as usize][m as usize] = (x * (2.0 * lf - 1.0) * buf[(li - 1) as usize][m as usize]
                - (lf + mf - 1.0) * buf[(li - 2) as usize][m as usize])
                / (lf - mf);
        }
    }

    // Apply SN3D normalisation and fill ±m slots.
    for m in 0..=l {
        let norm = sn3d_norm(l, m);
        let val = norm * buf[l as usize][m as usize];
        let idx_pos = (m + l) as usize;
        let idx_neg = (-m + l) as usize;
        if m == 0 {
            plm[idx_pos] = val;
        } else {
            plm[idx_pos] = val; // +m
            plm[idx_neg] = val; // -m (same P, different trig factor)
        }
    }
    plm
}

/// SN3D normalisation factor for (l, m).
fn sn3d_norm(l: i32, m: i32) -> f32 {
    let m = m.abs();
    let delta = if m == 0 { 1.0_f32 } else { 2.0 };
    // N_l^m = sqrt( delta * (l-m)! / (l+m)! )
    let mut factorial_ratio = 1.0_f32;
    for k in (l - m + 1)..=(l + m) {
        factorial_ratio *= k as f32;
    }
    (delta / factorial_ratio).sqrt()
}

// ─── Covariance accumulator ───────────────────────────────────────────────────

/// Accumulates the HOA covariance matrix `B = (1/T) Σ a_t * a_tᵀ` from frames.
#[derive(Debug, Clone)]
pub struct CovarianceAccumulator {
    max_order: usize,
    n_ch: usize,
    /// Upper-triangular covariance (flat, n_ch × n_ch row-major).
    cov: Vec<f32>,
    frame_count: u64,
}

impl CovarianceAccumulator {
    /// Create a new accumulator for HOA signals up to `max_order`.
    pub fn new(max_order: usize) -> Self {
        let n_ch = (max_order + 1) * (max_order + 1);
        Self {
            max_order,
            n_ch,
            cov: vec![0.0_f32; n_ch * n_ch],
            frame_count: 0,
        }
    }

    /// Return the maximum HOA order this accumulator was configured for.
    pub fn max_order(&self) -> usize {
        self.max_order
    }

    /// Accumulate one sample-frame of HOA coefficients.
    ///
    /// `frame` must have exactly `n_ch = (max_order+1)²` elements.
    pub fn accumulate(&mut self, frame: &[f32]) -> Result<(), SpatialError> {
        if frame.len() != self.n_ch {
            return Err(SpatialError::InvalidConfig(format!(
                "expected {} HOA channels, got {}",
                self.n_ch,
                frame.len()
            )));
        }
        let n = self.n_ch;
        for i in 0..n {
            for j in i..n {
                self.cov[i * n + j] += frame[i] * frame[j];
            }
        }
        self.frame_count += 1;
        Ok(())
    }

    /// Accumulate multiple frames packed as `[f0_ch0, f0_ch1, ..., f1_ch0, ...]`.
    pub fn accumulate_signal(&mut self, signal: &[f32]) -> Result<(), SpatialError> {
        if signal.len() % self.n_ch != 0 {
            return Err(SpatialError::InvalidConfig(format!(
                "signal length {} not divisible by n_ch {}",
                signal.len(),
                self.n_ch
            )));
        }
        for frame in signal.chunks_exact(self.n_ch) {
            self.accumulate(frame)?;
        }
        Ok(())
    }

    /// Return the normalised covariance matrix (divided by frame count).
    pub fn covariance(&self) -> Vec<f32> {
        if self.frame_count == 0 {
            return self.cov.clone();
        }
        let scale = 1.0 / self.frame_count as f32;
        let n = self.n_ch;
        let mut out = vec![0.0_f32; n * n];
        for i in 0..n {
            for j in i..n {
                let v = self.cov[i * n + j] * scale;
                out[i * n + j] = v;
                out[j * n + i] = v;
            }
        }
        out
    }

    /// Reset the accumulator.
    pub fn reset(&mut self) {
        self.cov.fill(0.0);
        self.frame_count = 0;
    }
}

// ─── Loudness map ─────────────────────────────────────────────────────────────

/// Resolution of the angular sampling grid.
#[derive(Debug, Clone, Copy)]
pub struct GridResolution {
    /// Number of azimuth samples in [0°, 360°).
    pub azimuth_steps: usize,
    /// Number of elevation samples in [-90°, +90°].
    pub elevation_steps: usize,
}

impl GridResolution {
    /// A reasonable default: 360 × 181 (1° grid).
    pub fn one_degree() -> Self {
        Self { azimuth_steps: 360, elevation_steps: 181 }
    }

    /// Coarser 5° grid for quick analysis.
    pub fn five_degree() -> Self {
        Self { azimuth_steps: 72, elevation_steps: 37 }
    }
}

/// A discretised energy map over the sphere, computed from HOA coefficients.
#[derive(Debug, Clone)]
pub struct LoudnessMap {
    /// Grid resolution used.
    pub resolution: GridResolution,
    /// HOA order used.
    pub max_order: usize,
    /// Energy values, indexed `[az_idx * elevation_steps + el_idx]`.
    pub energy: Vec<f32>,
}

impl LoudnessMap {
    /// Compute the energy at a given grid position (azimuth index, elevation index).
    pub fn get(&self, az_idx: usize, el_idx: usize) -> f32 {
        let idx = az_idx * self.resolution.elevation_steps + el_idx;
        self.energy.get(idx).copied().unwrap_or(0.0)
    }

    /// Azimuth angle (radians) for a grid index.
    pub fn azimuth_rad(&self, az_idx: usize) -> f32 {
        use std::f32::consts::TAU;
        az_idx as f32 / self.resolution.azimuth_steps as f32 * TAU
    }

    /// Elevation angle (radians) for a grid index.
    pub fn elevation_rad(&self, el_idx: usize) -> f32 {
        use std::f32::consts::PI;
        -PI / 2.0
            + el_idx as f32 / (self.resolution.elevation_steps.saturating_sub(1).max(1)) as f32
                * PI
    }

    /// Direction of peak energy `(azimuth_rad, elevation_rad)`.
    pub fn sweet_spot(&self) -> SweetSpot {
        let az_steps = self.resolution.azimuth_steps;
        let el_steps = self.resolution.elevation_steps;

        let max_idx = self
            .energy
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        let peak_az = max_idx / el_steps;
        let peak_el = max_idx % el_steps;
        let peak_energy = self.energy[max_idx];
        let half_power = peak_energy * 0.5;

        // Estimate half-power spread: count cells above half-power
        let above_count = self.energy.iter().filter(|&&e| e >= half_power).count();
        let total_cells = az_steps * el_steps;
        // Solid angle fraction → convert to steradians (≈ 4π × fraction)
        let spread_rad = (above_count as f32 / total_cells as f32
            * 4.0
            * std::f32::consts::PI)
            .sqrt();

        SweetSpot {
            azimuth_rad: self.azimuth_rad(peak_az),
            elevation_rad: self.elevation_rad(peak_el),
            peak_energy,
            half_power_spread_rad: spread_rad,
        }
    }

    /// Extract a horizontal-plane polar plot (elevation = 0° ≈ nearest index).
    ///
    /// Returns `(azimuth_rad_values, energy_values)` pairs.
    pub fn horizontal_polar(&self) -> Vec<(f32, f32)> {
        let el_steps = self.resolution.elevation_steps;
        // Find elevation index closest to 0 (equator).
        let eq_idx = el_steps / 2;

        (0..self.resolution.azimuth_steps)
            .map(|az| (self.azimuth_rad(az), self.get(az, eq_idx)))
            .collect()
    }

    /// Extract a median-plane polar plot (azimuth = 0° and 180°).
    ///
    /// Returns `(elevation_rad_values, energy_values)` pairs.
    pub fn median_polar(&self) -> Vec<(f32, f32)> {
        (0..self.resolution.elevation_steps)
            .map(|el| (self.elevation_rad(el), self.get(0, el)))
            .collect()
    }

    /// Normalise the energy values so that the maximum is 1.0.
    pub fn normalise(&mut self) {
        let max_e = self.energy.iter().cloned().fold(0.0_f32, f32::max);
        if max_e > 1e-12 {
            for e in &mut self.energy {
                *e /= max_e;
            }
        }
    }
}

/// Sweet-spot analysis result.
#[derive(Debug, Clone, Copy)]
pub struct SweetSpot {
    /// Azimuth of maximum energy (radians).
    pub azimuth_rad: f32,
    /// Elevation of maximum energy (radians).
    pub elevation_rad: f32,
    /// Peak energy value.
    pub peak_energy: f32,
    /// Approximate half-power angular spread (radians, area-based estimate).
    pub half_power_spread_rad: f32,
}

// ─── Map builder ──────────────────────────────────────────────────────────────

/// Build a [`LoudnessMap`] from a normalised HOA covariance matrix.
///
/// `cov` must be `n_ch × n_ch` (flat row-major) where `n_ch = (max_order+1)²`.
/// `resolution` controls the sampling grid density.
///
/// # Energy formula
/// `E(d) = d^T * B * d` where `d` is the SH decoding vector for direction
/// `(θ, φ)` and `B` is the covariance matrix.
pub fn build_map_from_covariance(
    cov: &[f32],
    max_order: usize,
    resolution: GridResolution,
) -> Result<LoudnessMap, SpatialError> {
    let n_ch = (max_order + 1) * (max_order + 1);
    if cov.len() != n_ch * n_ch {
        return Err(SpatialError::InvalidConfig(format!(
            "covariance matrix size mismatch: expected {}, got {}",
            n_ch * n_ch,
            cov.len()
        )));
    }
    let az_steps = resolution.azimuth_steps;
    let el_steps = resolution.elevation_steps;
    let mut energy = vec![0.0_f32; az_steps * el_steps];

    use std::f32::consts::{PI, TAU};

    for az_idx in 0..az_steps {
        let az = az_idx as f32 / az_steps as f32 * TAU;
        for el_idx in 0..el_steps {
            let el = -PI / 2.0
                + el_idx as f32 / (el_steps.saturating_sub(1).max(1)) as f32 * PI;
            let d = eval_sh(max_order, az, el);
            // E = d^T * B * d
            let mut e = 0.0_f32;
            for i in 0..n_ch {
                let mut bd_i = 0.0_f32;
                for j in 0..n_ch {
                    bd_i += cov[i * n_ch + j] * d[j];
                }
                e += d[i] * bd_i;
            }
            energy[az_idx * el_steps + el_idx] = e.max(0.0);
        }
    }

    Ok(LoudnessMap { resolution, max_order, energy })
}

/// Build a [`LoudnessMap`] directly from a single HOA frame (instantaneous
/// energy map: `E(d) = (d^T a)²`).
pub fn build_map_from_frame(
    frame: &[f32],
    max_order: usize,
    resolution: GridResolution,
) -> Result<LoudnessMap, SpatialError> {
    let n_ch = (max_order + 1) * (max_order + 1);
    if frame.len() != n_ch {
        return Err(SpatialError::InvalidConfig(format!(
            "expected {n_ch} channels, got {}",
            frame.len()
        )));
    }
    let az_steps = resolution.azimuth_steps;
    let el_steps = resolution.elevation_steps;
    let mut energy = vec![0.0_f32; az_steps * el_steps];

    use std::f32::consts::{PI, TAU};

    for az_idx in 0..az_steps {
        let az = az_idx as f32 / az_steps as f32 * TAU;
        for el_idx in 0..el_steps {
            let el = -PI / 2.0
                + el_idx as f32 / (el_steps.saturating_sub(1).max(1)) as f32 * PI;
            let d = eval_sh(max_order, az, el);
            let mut p = 0.0_f32;
            for i in 0..n_ch {
                p += d[i] * frame[i];
            }
            energy[az_idx * el_steps + el_idx] = p * p;
        }
    }

    Ok(LoudnessMap { resolution, max_order, energy })
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    #[test]
    fn test_eval_sh_order0_is_constant() {
        // Order-0 SH is omnidirectional: Y_0^0 = 1.
        let sh = eval_sh(0, 0.0, 0.0);
        assert_eq!(sh.len(), 1);
        assert!((sh[0] - 1.0).abs() < 1e-5);

        let sh2 = eval_sh(0, PI, PI / 4.0);
        assert!((sh2[0] - 1.0).abs() < 1e-5, "Y_0^0 must be 1 everywhere");
    }

    #[test]
    fn test_eval_sh_acn_count() {
        for order in 0..=3 {
            let sh = eval_sh(order, 0.5, 0.2);
            assert_eq!(sh.len(), (order + 1) * (order + 1));
        }
    }

    #[test]
    fn test_covariance_accumulator_accumulate() {
        let mut acc = CovarianceAccumulator::new(1);
        // All-zero frame should keep covariance zero.
        acc.accumulate(&[0.0, 0.0, 0.0, 0.0]).unwrap();
        let cov = acc.covariance();
        assert!(cov.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_covariance_accumulator_wrong_size() {
        let mut acc = CovarianceAccumulator::new(1);
        let result = acc.accumulate(&[1.0, 2.0]); // should be 4
        assert!(result.is_err());
    }

    #[test]
    fn test_build_map_from_frame_omnidirectional() {
        // Pure W channel (omnidirectional) should produce a uniform map.
        let res = GridResolution { azimuth_steps: 36, elevation_steps: 19 };
        let frame = vec![1.0_f32, 0.0, 0.0, 0.0]; // W=1, YZX=0
        let map = build_map_from_frame(&frame, 1, res).unwrap();
        // All energies should be equal (W decodes uniformly).
        let first = map.energy[0];
        for &e in &map.energy {
            assert!((e - first).abs() < 1e-4, "omnidirectional should be uniform: {e} vs {first}");
        }
    }

    #[test]
    fn test_build_map_energy_nonnegative() {
        let res = GridResolution { azimuth_steps: 36, elevation_steps: 19 };
        let frame: Vec<f32> = (0..4).map(|i| i as f32 * 0.25).collect();
        let map = build_map_from_frame(&frame, 1, res).unwrap();
        for &e in &map.energy {
            assert!(e >= 0.0, "energy must be non-negative");
        }
    }

    #[test]
    fn test_sweet_spot_returns_valid_direction() {
        let res = GridResolution { azimuth_steps: 36, elevation_steps: 19 };
        // Source pointing to azimuth ≈ 0, elevation ≈ 0: encode it.
        let d = eval_sh(1, 0.0, 0.0);
        let map = build_map_from_frame(&d, 1, res).unwrap();
        let ss = map.sweet_spot();
        // Peak should be near (0, 0)
        assert!(ss.azimuth_rad.abs() < 0.2 || (ss.azimuth_rad - 2.0 * PI).abs() < 0.2,
            "sweet spot azimuth should be near 0: {}", ss.azimuth_rad);
        assert!(ss.elevation_rad.abs() < 0.3,
            "sweet spot elevation should be near 0: {}", ss.elevation_rad);
    }

    #[test]
    fn test_horizontal_polar_length() {
        let res = GridResolution { azimuth_steps: 36, elevation_steps: 19 };
        let frame = vec![1.0_f32, 0.0, 0.0, 0.0];
        let map = build_map_from_frame(&frame, 1, res).unwrap();
        let polar = map.horizontal_polar();
        assert_eq!(polar.len(), 36);
    }

    #[test]
    fn test_normalise_sets_max_to_one() {
        let res = GridResolution { azimuth_steps: 36, elevation_steps: 19 };
        let frame: Vec<f32> = (0..4).map(|i| i as f32).collect();
        let mut map = build_map_from_frame(&frame, 1, res).unwrap();
        map.normalise();
        let max_e = map.energy.iter().cloned().fold(0.0_f32, f32::max);
        assert!((max_e - 1.0).abs() < 1e-5, "max energy after normalise should be 1.0");
    }

    #[test]
    fn test_covariance_accumulate_signal() {
        let mut acc = CovarianceAccumulator::new(1); // 4 channels
        let signal: Vec<f32> = (0..8).map(|i| i as f32 * 0.1).collect(); // 2 frames
        acc.accumulate_signal(&signal).unwrap();
        assert_eq!(acc.frame_count, 2);
    }

    #[test]
    fn test_build_map_wrong_covariance_size() {
        let res = GridResolution { azimuth_steps: 10, elevation_steps: 10 };
        let cov = vec![0.0_f32; 9]; // wrong: should be 16 for order-1
        let result = build_map_from_covariance(&cov, 1, res);
        assert!(result.is_err());
    }
}
