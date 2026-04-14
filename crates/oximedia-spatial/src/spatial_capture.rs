//! A-format to B-format microphone array conversion.
//!
//! Provides converters for two common spatial microphone array formats:
//!
//! ## Soundfield A-format → B-format
//!
//! A tetrahedral microphone (Soundfield) uses four cardioid capsules:
//! FLU (Front-Left-Up), FRD (Front-Right-Down), BLD (Back-Left-Down),
//! BRU (Back-Right-Up).
//!
//! The A-to-B matrix:
//! ```text
//! W = (A + B + C + D) / √2
//! X = (A + B − C − D) / √2
//! Y = (A − B + C − D) / √2
//! Z = (A − B − C + D) / √2
//! ```
//!
//! ## Eigenmike em32 → First-order B-format
//!
//! 32-capsule spherical array encoded to W/X/Y/Z via SH least-squares.
//!
//! # Example
//!
//! ```rust
//! use oximedia_spatial::spatial_capture::{SoundfieldConverter, AFormatFrame};
//!
//! let conv = SoundfieldConverter::new();
//! let a = AFormatFrame {
//!     flu: vec![1.0_f32; 64],
//!     frd: vec![1.0_f32; 64],
//!     bld: vec![1.0_f32; 64],
//!     bru: vec![1.0_f32; 64],
//! };
//! let b = conv.convert(&a).expect("conversion");
//! assert_eq!(b.w.len(), 64);
//! ```

use crate::SpatialError;

// ─── AFormatFrame ─────────────────────────────────────────────────────────────

/// Four-capsule A-format audio frame from a tetrahedral microphone.
#[derive(Debug, Clone)]
pub struct AFormatFrame {
    /// Front-Left-Up capsule.
    pub flu: Vec<f32>,
    /// Front-Right-Down capsule.
    pub frd: Vec<f32>,
    /// Back-Left-Down capsule.
    pub bld: Vec<f32>,
    /// Back-Right-Up capsule.
    pub bru: Vec<f32>,
}

impl AFormatFrame {
    /// Number of samples per channel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.flu.len()
    }

    /// `true` if the frame has no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.flu.is_empty()
    }
}

// ─── BFormatFrame ─────────────────────────────────────────────────────────────

/// First-order Ambisonic B-format frame (W, X, Y, Z).
#[derive(Debug, Clone)]
pub struct BFormatFrame {
    /// Omnidirectional channel (W).
    pub w: Vec<f32>,
    /// Front-back dipole (X).
    pub x: Vec<f32>,
    /// Left-right dipole (Y).
    pub y: Vec<f32>,
    /// Up-down dipole (Z).
    pub z: Vec<f32>,
}

impl BFormatFrame {
    /// Number of samples per channel.
    #[must_use]
    pub fn len(&self) -> usize {
        self.w.len()
    }

    /// `true` if the frame has no samples.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.w.is_empty()
    }

    /// Sum of squared samples across all four channels.
    #[must_use]
    pub fn total_power(&self) -> f64 {
        let sum_sq = |v: &[f32]| v.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>();
        sum_sq(&self.w) + sum_sq(&self.x) + sum_sq(&self.y) + sum_sq(&self.z)
    }
}

// ─── SoundfieldConverter ─────────────────────────────────────────────────────

/// Converts A-format tetrahedral signals to first-order B-format.
#[derive(Debug, Clone)]
pub struct SoundfieldConverter {
    /// Normalization gain. Default = 1/√2.
    pub gain: f32,
}

impl SoundfieldConverter {
    /// Create with standard normalization gain (1/√2).
    #[must_use]
    pub fn new() -> Self {
        Self { gain: std::f32::consts::FRAC_1_SQRT_2 }
    }

    /// Create with a custom normalization gain.
    #[must_use]
    pub fn with_gain(gain: f32) -> Self {
        Self { gain }
    }

    /// Convert A-format to B-format.
    pub fn convert(&self, a: &AFormatFrame) -> Result<BFormatFrame, SpatialError> {
        let n = a.flu.len();
        if n == 0 {
            return Err(SpatialError::InvalidConfig("A-format frame is empty".into()));
        }
        if a.frd.len() != n || a.bld.len() != n || a.bru.len() != n {
            return Err(SpatialError::InvalidConfig(
                "All A-format channels must have equal length".into(),
            ));
        }
        let g = self.gain;
        let mut w = Vec::with_capacity(n);
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut z = Vec::with_capacity(n);
        for i in 0..n {
            let a_s = a.flu[i];
            let b_s = a.frd[i];
            let c_s = a.bld[i];
            let d_s = a.bru[i];
            w.push(g * (a_s + b_s + c_s + d_s));
            x.push(g * (a_s + b_s - c_s - d_s));
            y.push(g * (a_s - b_s + c_s - d_s));
            z.push(g * (a_s - b_s - c_s + d_s));
        }
        Ok(BFormatFrame { w, x, y, z })
    }
}

impl Default for SoundfieldConverter {
    fn default() -> Self {
        Self::new()
    }
}

// ─── EigenmikeCapturer ───────────────────────────────────────────────────────

/// Number of capsules on the Eigenmike em32.
pub const EIGENMIKE_CAPSULES: usize = 32;

/// A single capsule position on the unit sphere.
#[derive(Debug, Clone, Copy)]
pub struct CapsulePosition {
    /// Azimuth in degrees.
    pub azimuth_deg: f32,
    /// Elevation in degrees (−90 to +90).
    pub elevation_deg: f32,
}

/// Nominal Eigenmike em32 capsule positions.
#[must_use]
pub fn eigenmike_positions() -> [CapsulePosition; EIGENMIKE_CAPSULES] {
    const TABLE: [(f32, f32); 32] = [
        (0.0, 21.0), (32.0, 0.0), (0.0, -21.0), (328.0, 0.0),
        (0.0, 58.0), (45.0, 35.0), (69.0, 0.0), (45.0, -35.0),
        (0.0, -58.0), (315.0, -35.0), (291.0, 0.0), (315.0, 35.0),
        (91.0, 69.0), (90.0, 32.0), (90.0, -31.0), (89.0, -69.0),
        (180.0, 21.0), (212.0, 0.0), (180.0, -21.0), (148.0, 0.0),
        (180.0, 58.0), (225.0, 35.0), (249.0, 0.0), (225.0, -35.0),
        (180.0, -58.0), (135.0, -35.0), (111.0, 0.0), (135.0, 35.0),
        (269.0, 69.0), (270.0, 32.0), (270.0, -31.0), (271.0, -69.0),
    ];
    let mut out = [CapsulePosition { azimuth_deg: 0.0, elevation_deg: 0.0 }; 32];
    for (i, &(az, el)) in TABLE.iter().enumerate() {
        out[i] = CapsulePosition { azimuth_deg: az, elevation_deg: el };
    }
    out
}

/// Eigenmike em32 to first-order B-format encoder.
#[derive(Debug, Clone)]
pub struct EigenmikeCapturer {
    /// Encoding matrix [4 channels × 32 capsules].
    enc: [[f32; EIGENMIKE_CAPSULES]; 4],
}

impl EigenmikeCapturer {
    /// Build the encoder from nominal capsule positions.
    #[must_use]
    pub fn new() -> Self {
        let positions = eigenmike_positions();
        let inv_sqrt4pi: f32 = (4.0 * std::f32::consts::PI).sqrt().recip();
        let sqrt_3_4pi: f32 = (3.0 / (4.0 * std::f32::consts::PI)).sqrt();
        let mut y_mat = [[0.0_f32; EIGENMIKE_CAPSULES]; 4];
        for (k, pos) in positions.iter().enumerate() {
            let az = pos.azimuth_deg.to_radians();
            let el = pos.elevation_deg.to_radians();
            y_mat[0][k] = inv_sqrt4pi;
            y_mat[1][k] = sqrt_3_4pi * az.sin() * el.cos();
            y_mat[2][k] = sqrt_3_4pi * el.sin();
            y_mat[3][k] = sqrt_3_4pi * az.cos() * el.cos();
        }
        let n_inv = 1.0 / EIGENMIKE_CAPSULES as f32;
        let enc = y_mat.map(|row| row.map(|v| v * n_inv));
        Self { enc }
    }

    /// Encode 32-capsule signals to first-order B-format.
    pub fn encode(&self, capsules: &[Vec<f32>]) -> Result<BFormatFrame, SpatialError> {
        if capsules.len() != EIGENMIKE_CAPSULES {
            return Err(SpatialError::InvalidConfig(format!(
                "Expected {EIGENMIKE_CAPSULES} capsule channels, got {}",
                capsules.len()
            )));
        }
        let n = capsules[0].len();
        if !capsules.iter().all(|c| c.len() == n) {
            return Err(SpatialError::InvalidConfig(
                "All capsule channels must have equal length".into(),
            ));
        }
        if n == 0 {
            return Err(SpatialError::InvalidConfig("Capsule channels are empty".into()));
        }
        let mut w = vec![0.0_f32; n];
        let mut y_ch = vec![0.0_f32; n];
        let mut z_ch = vec![0.0_f32; n];
        let mut x_ch = vec![0.0_f32; n];
        for (k, cap) in capsules.iter().enumerate() {
            let ew = self.enc[0][k];
            let ey = self.enc[1][k];
            let ez = self.enc[2][k];
            let ex = self.enc[3][k];
            for i in 0..n {
                w[i] += ew * cap[i];
                y_ch[i] += ey * cap[i];
                z_ch[i] += ez * cap[i];
                x_ch[i] += ex * cap[i];
            }
        }
        Ok(BFormatFrame { w, x: x_ch, y: y_ch, z: z_ch })
    }
}

impl Default for EigenmikeCapturer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn a_frame(n: usize, val: f32) -> AFormatFrame {
        AFormatFrame {
            flu: vec![val; n],
            frd: vec![val; n],
            bld: vec![val; n],
            bru: vec![val; n],
        }
    }

    #[test]
    fn test_soundfield_equal_capsules_w_only() {
        let conv = SoundfieldConverter::new();
        let b = conv.convert(&a_frame(8, 1.0)).expect("ok");
        let expected_w = 4.0 * std::f32::consts::FRAC_1_SQRT_2;
        for &s in &b.w {
            assert!((s - expected_w).abs() < 1e-4, "W={s}");
        }
        for &s in b.x.iter().chain(b.y.iter()).chain(b.z.iter()) {
            assert!(s.abs() < 1e-4, "Dipole should be 0, got {s}");
        }
    }

    #[test]
    fn test_soundfield_empty_error() {
        let conv = SoundfieldConverter::new();
        let a = AFormatFrame { flu: vec![], frd: vec![], bld: vec![], bru: vec![] };
        assert!(conv.convert(&a).is_err());
    }

    #[test]
    fn test_soundfield_mismatched_lengths_error() {
        let conv = SoundfieldConverter::new();
        let a = AFormatFrame {
            flu: vec![1.0; 8],
            frd: vec![1.0; 4],
            bld: vec![1.0; 8],
            bru: vec![1.0; 8],
        };
        assert!(conv.convert(&a).is_err());
    }

    #[test]
    fn test_soundfield_output_length() {
        let conv = SoundfieldConverter::new();
        let b = conv.convert(&a_frame(100, 0.5)).expect("ok");
        assert_eq!(b.len(), 100);
    }

    #[test]
    fn test_eigenmike_positions_count() {
        assert_eq!(eigenmike_positions().len(), EIGENMIKE_CAPSULES);
    }

    #[test]
    fn test_eigenmike_encode_output_length() {
        let encoder = EigenmikeCapturer::new();
        let capsules: Vec<Vec<f32>> = (0..EIGENMIKE_CAPSULES).map(|_| vec![0.1_f32; 128]).collect();
        let b = encoder.encode(&capsules).expect("ok");
        assert_eq!(b.len(), 128);
    }

    #[test]
    fn test_eigenmike_wrong_capsule_count() {
        let encoder = EigenmikeCapturer::new();
        let capsules: Vec<Vec<f32>> = (0..8).map(|_| vec![0.0_f32; 64]).collect();
        assert!(encoder.encode(&capsules).is_err());
    }

    #[test]
    fn test_eigenmike_silence_gives_silence() {
        let encoder = EigenmikeCapturer::new();
        let capsules: Vec<Vec<f32>> = (0..EIGENMIKE_CAPSULES).map(|_| vec![0.0_f32; 64]).collect();
        let b = encoder.encode(&capsules).expect("ok");
        for &s in b.w.iter().chain(b.x.iter()).chain(b.y.iter()).chain(b.z.iter()) {
            assert!(s.abs() < 1e-7, "Expected silence, got {s}");
        }
    }

    #[test]
    fn test_b_format_total_power() {
        let b = BFormatFrame {
            w: vec![1.0; 4],
            x: vec![0.0; 4],
            y: vec![0.0; 4],
            z: vec![0.0; 4],
        };
        assert!((b.total_power() - 4.0).abs() < 1e-9);
    }

    /// Energy preservation: converting equal-magnitude capsule signals and checking
    /// that the B-format has non-zero energy (regression guard).
    #[test]
    fn test_soundfield_energy_nonzero() {
        let conv = SoundfieldConverter::new();
        let b = conv.convert(&a_frame(16, 1.0)).expect("ok");
        assert!(b.total_power() > 0.0);
    }
}
