//! Ambisonics rotation — spherical harmonic rotation matrices.
//!
//! Applies yaw, pitch, and roll rotations to B-format (or HOA) audio streams
//! without decoding to speaker feeds.  The rotation is applied directly to the
//! spherical harmonic (SH) coefficient domain using the *Ivanic & Ruedenberg*
//! recursive algorithm (real SH convention, ACN channel ordering, SN3D normalisation).
//!
//! # Supported orders
//! - Order 0: trivial (W channel unmodified)
//! - Order 1: 4-channel first-order Ambisonics (FOA)
//! - Order 2: 9-channel second-order HOA
//! - Order 3: 16-channel third-order HOA
//! - Order 4: 25-channel fourth-order HOA
//! - Order 5: 36-channel fifth-order HOA
//!
//! # Coordinate convention
//! All angles are in **radians** using ZYZ intrinsic Euler convention.
//!
//! # References
//! Ivanic, J. & Ruedenberg, K. (1996).
//! "Rotation matrices for real spherical harmonics. Direct determination by
//!  recursion." *J. Phys. Chem.* 100(15), 6342–6347.

use crate::SpatialError;

// ─── ACN index ────────────────────────────────────────────────────────────────

/// (l, m) → ACN index: l² + l + m
#[inline]
fn acn(l: i32, m: i32) -> usize {
    (l * l + l + m) as usize
}

// ─── Compact (2l+1)×(2l+1) matrix ────────────────────────────────────────────

/// A square matrix of size `(2*order+1) × (2*order+1)` stored row-major.
/// Row/column indices map signed m ∈ [-l, l] → position `m + l`.
#[derive(Debug, Clone)]
struct BandMatrix {
    order: usize,
    data: Vec<f64>,
}

impl BandMatrix {
    fn new(order: usize) -> Self {
        let s = 2 * order + 1;
        Self { order, data: vec![0.0; s * s] }
    }

    #[inline]
    fn size(&self) -> usize {
        2 * self.order + 1
    }

    #[inline]
    fn get(&self, row: usize, col: usize) -> f64 {
        self.data[row * self.size() + col]
    }

    #[inline]
    fn set(&mut self, row: usize, col: usize, val: f64) {
        let s = self.size();
        self.data[row * s + col] = val;
    }

    /// Access by signed m, n ∈ [-l, l].  Returns 0 if out of range.
    #[inline]
    fn get_mn(&self, m: i32, n: i32) -> f64 {
        let l = self.order as i32;
        if m < -l || m > l || n < -l || n > l {
            return 0.0;
        }
        self.get((m + l) as usize, (n + l) as usize)
    }

    #[inline]
    fn set_mn(&mut self, m: i32, n: i32, val: f64) {
        let l = self.order as i32;
        self.set((m + l) as usize, (n + l) as usize, val);
    }
}

// ─── Ivanic-Ruedenberg recurrence ─────────────────────────────────────────────

/// Kronecker delta.
#[inline]
fn kd(a: i32, b: i32) -> f64 {
    if a == b { 1.0 } else { 0.0 }
}

/// u(l, m, n) — scalar coefficient, eq. (8.2) in I&R 1996.
fn coeff_u(l: i32, m: i32, n: i32) -> f64 {
    let lf = l as f64;
    let mf = m as f64;
    let nf = n as f64;
    let d = if n.abs() == l {
        2.0 * lf * (2.0 * lf - 1.0)
    } else {
        (lf + nf) * (lf - nf)
    };
    if d.abs() < 1e-15 { return 0.0; }
    ((lf + mf) * (lf - mf) / d).sqrt()
}

/// v(l, m, n) — scalar coefficient, eq. (8.3).
fn coeff_v(l: i32, m: i32, n: i32) -> f64 {
    let lf = l as f64;
    let ma = m.abs() as f64;
    let nf = n as f64;
    let d = if n.abs() == l {
        2.0 * lf * (2.0 * lf - 1.0)
    } else {
        (lf + nf) * (lf - nf)
    };
    if d.abs() < 1e-15 { return 0.0; }
    0.5 * ((1.0 + kd(m, 0)) * (lf + ma - 1.0) * (lf + ma) / d).sqrt()
}

/// w(l, m, n) — scalar coefficient, eq. (8.4).
fn coeff_w(l: i32, m: i32, n: i32) -> f64 {
    if m == 0 { return 0.0; }
    let lf = l as f64;
    let ma = m.abs() as f64;
    let nf = n as f64;
    let d = if n.abs() == l {
        2.0 * lf * (2.0 * lf - 1.0)
    } else {
        (lf + nf) * (lf - nf)
    };
    if d.abs() < 1e-15 { return 0.0; }
    -0.5 * ((lf - ma - 1.0) * (lf - ma) / d).sqrt()
}

/// P(i, a, b) helper from I&R eq. (8.1).
///
/// `i ∈ {-1, 0, 1}` indexes R^1 rows.  a, b are signed indices in R^{l-1}.
/// The expression handles the three cases for b = ±(l-1).
fn p_func(i: i32, a: i32, b: i32, l: i32, r1: &BandMatrix, rl1: &BandMatrix) -> f64 {
    let lm1 = (l - 1) as i32;
    if b == lm1 {
        r1.get_mn(i, 1) * rl1.get_mn(a, lm1 - 1)
            - r1.get_mn(i, -1) * rl1.get_mn(a, -(lm1 - 1))
    } else if b == -lm1 {
        r1.get_mn(i, 1) * rl1.get_mn(a, -(lm1 - 1))
            + r1.get_mn(i, -1) * rl1.get_mn(a, lm1 - 1)
    } else {
        r1.get_mn(i, 0) * rl1.get_mn(a, b)
    }
}

/// Compute R^l_{m,n} using I&R recurrence from R^1 and R^{l-1}.
fn r_entry(l: i32, m: i32, n: i32, r1: &BandMatrix, rl1: &BandMatrix) -> f64 {
    let u = coeff_u(l, m, n);
    let v = coeff_v(l, m, n);
    let w = coeff_w(l, m, n);

    let u_part = if u.abs() > 1e-15 {
        u * p_func(0, m, n, l, r1, rl1)
    } else {
        0.0
    };

    let v_part = if v.abs() > 1e-15 {
        let pv = if m > 0 {
            p_func(1, m - 1, n, l, r1, rl1) - p_func(-1, -(m - 1), n, l, r1, rl1)
        } else if m < 0 {
            p_func(1, m + 1, n, l, r1, rl1) + p_func(-1, -(m + 1), n, l, r1, rl1)
        } else {
            // m == 0
            p_func(1, 1, n, l, r1, rl1) + p_func(-1, -1, n, l, r1, rl1)
        };
        v * pv
    } else {
        0.0
    };

    let w_part = if w.abs() > 1e-15 {
        // w only nonzero when m != 0
        let pw = if m > 0 {
            p_func(1, m + 1, n, l, r1, rl1) + p_func(-1, -(m + 1), n, l, r1, rl1)
        } else {
            // m < 0
            p_func(1, m - 1, n, l, r1, rl1) - p_func(-1, -(m - 1), n, l, r1, rl1)
        };
        w * pw
    } else {
        0.0
    };

    u_part + v_part + w_part
}

// ─── Cartesian → order-1 SH block ────────────────────────────────────────────

/// Build the order-1 SH rotation block from a 3×3 Cartesian rotation matrix.
///
/// Real SH (SN3D/ACN) basis for l=1: m=-1 ~ y, m=0 ~ z, m=+1 ~ x.
/// Cartesian axis index: x=0, y=1, z=2.
///
/// `rot3[col][row]` — col-major column vector storage (each `rot3[j]` is column j).
fn order1_matrix(rot3: &[[f64; 3]; 3]) -> BandMatrix {
    // m → Cartesian axis
    let axis = |m: i32| match m {
        -1 => 1_usize, // y
        0  => 2,       // z
        _  => 0,       // x
    };
    let mut mat = BandMatrix::new(1);
    // R^1_{m,n} = rot3[ axis(n) ][ axis(m) ]
    // This is the rotation matrix entry mapping SH-basis-n to SH-basis-m.
    for mp in -1_i32..=1 {
        for np in -1_i32..=1 {
            let val = rot3[axis(np)][axis(mp)];
            mat.set_mn(mp, np, val);
        }
    }
    mat
}

// ─── ZYZ → 3×3 Cartesian rotation ────────────────────────────────────────────

/// ZYZ intrinsic Euler angles → 3×3 column-major Cartesian rotation matrix.
///
/// `rot3[col][row]` = element at (row, col) of R.
/// R = Rz(yaw) · Ry(pitch) · Rz(roll).
fn zyz_rotation_matrix(yaw: f64, pitch: f64, roll: f64) -> [[f64; 3]; 3] {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let (sr, cr) = roll.sin_cos();

    // Rz(α) column-major: col 0 = (c, s, 0), col 1 = (-s, c, 0), col 2 = (0, 0, 1)
    let rz = |s: f64, c: f64| -> [[f64; 3]; 3] {
        [[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]]
    };
    // Ry(α) column-major: col 0 = (c, 0, -s), col 1 = (0, 1, 0), col 2 = (s, 0, c)
    let ry = |s: f64, c: f64| -> [[f64; 3]; 3] {
        [[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]]
    };

    // Matrix multiply: C = A · B, where A,B stored column-major.
    // C[col][row] = Σ_k A[k][row] * B[col][k]
    let mul = |a: [[f64; 3]; 3], b: [[f64; 3]; 3]| -> [[f64; 3]; 3] {
        let mut c = [[0.0_f64; 3]; 3];
        for col in 0..3 {
            for row in 0..3 {
                for k in 0..3 {
                    c[col][row] += a[k][row] * b[col][k];
                }
            }
        }
        c
    };

    let rz_yaw  = rz(sy, cy);
    let ry_pitch = ry(sp, cp);
    let rz_roll  = rz(sr, cr);

    mul(rz_yaw, mul(ry_pitch, rz_roll))
}

// ─── Public API ───────────────────────────────────────────────────────────────

/// Full HOA rotation matrix for orders 0..=max_order, stored row-major `N×N`.
/// N = (max_order+1)².  Block-diagonal: band l at rows/cols `[l², l²+2l]`.
#[derive(Debug, Clone)]
pub struct HoaRotationMatrix {
    /// Maximum HOA order (0–5).
    pub max_order: usize,
    /// Row-major N×N rotation matrix (f32).
    pub matrix: Vec<f32>,
}

impl HoaRotationMatrix {
    /// Number of HOA channels: `(max_order+1)²`.
    pub fn num_channels(&self) -> usize {
        (self.max_order + 1) * (self.max_order + 1)
    }

    /// Rotate one frame of HOA coefficients.
    ///
    /// `input` length must equal `num_channels()`.
    pub fn rotate_frame(&self, input: &[f32]) -> Result<Vec<f32>, SpatialError> {
        let n = self.num_channels();
        if input.len() != n {
            return Err(SpatialError::InvalidConfig(format!(
                "expected {n} HOA channels, got {}",
                input.len()
            )));
        }
        let mut out = vec![0.0_f32; n];
        for row in 0..n {
            let base = row * n;
            let mut acc = 0.0_f32;
            for col in 0..n {
                acc += self.matrix[base + col] * input[col];
            }
            out[row] = acc;
        }
        Ok(out)
    }

    /// Rotate a multi-frame HOA signal packed as `[f0_ch0, f0_ch1, ..., f1_ch0, ...]`.
    ///
    /// Length must be a multiple of `num_channels()`.
    pub fn rotate_signal(&self, signal: &[f32]) -> Result<Vec<f32>, SpatialError> {
        let n = self.num_channels();
        if signal.len() % n != 0 {
            return Err(SpatialError::InvalidConfig(format!(
                "signal length {} is not a multiple of num_channels {}",
                signal.len(),
                n
            )));
        }
        let mut out = vec![0.0_f32; signal.len()];
        for (fi, frame) in signal.chunks_exact(n).enumerate() {
            let rotated = self.rotate_frame(frame)?;
            out[fi * n..fi * n + n].copy_from_slice(&rotated);
        }
        Ok(out)
    }
}

/// Build a HOA rotation matrix for ZYZ Euler angles (radians).
///
/// # Parameters
/// - `max_order`: highest SH order (0–5).
/// - `yaw`:   first rotation angle (about Z).
/// - `pitch`: second rotation angle (about Y′).
/// - `roll`:  third rotation angle (about Z″).
///
/// # Errors
/// Returns [`SpatialError::InvalidConfig`] if `max_order > 5`.
pub fn build_rotation_matrix(
    max_order: usize,
    yaw: f64,
    pitch: f64,
    roll: f64,
) -> Result<HoaRotationMatrix, SpatialError> {
    if max_order > 5 {
        return Err(SpatialError::InvalidConfig(
            "max_order must be ≤ 5".to_string(),
        ));
    }

    let n = (max_order + 1) * (max_order + 1);
    let mut full = vec![0.0_f32; n * n];

    // Order 0: omnidirectional W channel is invariant.
    full[0] = 1.0;

    if max_order == 0 {
        return Ok(HoaRotationMatrix { max_order, matrix: full });
    }

    // Order-1 block from Cartesian rotation.
    let rot3 = zyz_rotation_matrix(yaw, pitch, roll);
    let r1 = order1_matrix(&rot3);

    // Write order-1 block at positions [1..=3] × [1..=3].
    {
        let base = 1_usize;
        let s = 3_usize;
        for row in 0..s {
            for col in 0..s {
                full[(base + row) * n + (base + col)] = r1.get(row, col) as f32;
            }
        }
    }

    // Higher-order blocks via I&R recurrence.
    let mut rl_prev = r1.clone();
    for l in 2..=(max_order as i32) {
        let s = (2 * l + 1) as usize;
        let mut rl = BandMatrix::new(l as usize);
        for m in -l..=l {
            for np in -l..=l {
                let val = r_entry(l, m, np, &r1, &rl_prev);
                rl.set_mn(m, np, val);
            }
        }
        let base = (l * l) as usize;
        for row in 0..s {
            for col in 0..s {
                full[(base + row) * n + (base + col)] = rl.get(row, col) as f32;
            }
        }
        rl_prev = rl;
    }

    Ok(HoaRotationMatrix { max_order, matrix: full })
}

/// Convenience: build and apply a rotation to a single HOA frame.
///
/// `frame` length must equal `(max_order+1)²`.
pub fn rotate_frame(
    frame: &[f32],
    max_order: usize,
    yaw: f64,
    pitch: f64,
    roll: f64,
) -> Result<Vec<f32>, SpatialError> {
    let mat = build_rotation_matrix(max_order, yaw, pitch, roll)?;
    mat.rotate_frame(frame)
}

/// Apply a pure yaw rotation (heading change) to a HOA frame.
///
/// For yaw θ the order-l block is block-diagonal:
/// - m=0 channel is unchanged.
/// - Paired (m, -m) channels rotate as `[[cos(m·θ), -sin(m·θ)], [sin(m·θ), cos(m·θ)]]`.
///
/// This path is faster than the full I&R recurrence for real-time head tracking.
pub fn rotate_yaw(frame: &[f32], max_order: usize, yaw: f64) -> Result<Vec<f32>, SpatialError> {
    let n = (max_order + 1) * (max_order + 1);
    if frame.len() != n {
        return Err(SpatialError::InvalidConfig(format!(
            "expected {n} channels, got {}",
            frame.len()
        )));
    }
    if max_order > 5 {
        return Err(SpatialError::InvalidConfig(
            "max_order must be ≤ 5".to_string(),
        ));
    }
    let mut out = vec![0.0_f32; n];
    out[0] = frame[0]; // W

    for l in 1..=(max_order as i32) {
        let m0 = acn(l, 0);
        out[m0] = frame[m0];

        for m in 1..=l {
            let (s, c) = ((m as f64 * yaw).sin() as f32, (m as f64 * yaw).cos() as f32);
            let pos = acn(l, m);
            let neg = acn(l, -m);
            // Real SH rotation for yaw about Z:
            // Y_l^{+m}' = cos(m·θ)·Y_l^{+m} - sin(m·θ)·Y_l^{-m}
            // Y_l^{-m}' = sin(m·θ)·Y_l^{+m} + cos(m·θ)·Y_l^{-m}
            out[pos] = c * frame[pos] - s * frame[neg];
            out[neg] = s * frame[pos] + c * frame[neg];
        }
    }
    Ok(out)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    /// Order-0 rotation is always identity (W is omnidirectional).
    #[test]
    fn test_order0_identity() {
        let mat = build_rotation_matrix(0, 1.0, 0.5, 0.2).unwrap();
        let frame = vec![1.0_f32];
        let out = mat.rotate_frame(&frame).unwrap();
        assert!(close(out[0], 1.0, 1e-4));
    }

    /// Zero rotation must be exactly identity.
    #[test]
    fn test_zero_rotation_is_identity() {
        let mat = build_rotation_matrix(1, 0.0, 0.0, 0.0).unwrap();
        let frame = vec![1.0_f32, 0.5, 0.3, 0.7];
        let out = mat.rotate_frame(&frame).unwrap();
        for (a, b) in out.iter().zip(frame.iter()) {
            assert!(close(*a, *b, 1e-5), "zero rotation should be identity: {a} vs {b}");
        }
    }

    /// 360° yaw returns to original (order 1, fast path).
    #[test]
    fn test_full_yaw_cycle_order1() {
        let frame = vec![1.0_f32, 0.2, 0.4, 0.6];
        let out = rotate_yaw(&frame, 1, 2.0 * PI).unwrap();
        for (a, b) in out.iter().zip(frame.iter()) {
            assert!(close(*a, *b, 1e-4), "full yaw cycle: {a} vs {b}");
        }
    }

    /// Pure yaw fast path and the full I&R matrix should agree.
    #[test]
    fn test_yaw_fast_vs_full_matrix() {
        let yaw = PI / 3.0;
        let frame: Vec<f32> = (0..4).map(|i| (i + 1) as f32 * 0.2).collect();
        let fast = rotate_yaw(&frame, 1, yaw).unwrap();
        // Build a pure-yaw rotation matrix via ZYZ with pitch=0, roll=0, yaw = yaw.
        // ZYZ pure yaw: yaw=θ, pitch=0, roll=0  →  Rz(θ)·Ry(0)·Rz(0) = Rz(θ).
        let full_mat = build_rotation_matrix(1, yaw, 0.0, 0.0).unwrap();
        let full = full_mat.rotate_frame(&frame).unwrap();
        for (a, b) in fast.iter().zip(full.iter()) {
            assert!(close(*a, *b, 1e-4), "fast vs full yaw: {a} vs {b}");
        }
    }

    /// Energy is preserved under rotation (order 2).
    #[test]
    fn test_energy_preservation_order2() {
        let frame: Vec<f32> = (0..9).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let energy_in: f32 = frame.iter().map(|x| x * x).sum();
        let out = rotate_frame(&frame, 2, 0.7, 0.3, 1.1).unwrap();
        let energy_out: f32 = out.iter().map(|x| x * x).sum();
        assert!((energy_in - energy_out).abs() < 0.01 * energy_in,
            "energy not preserved: in={energy_in}, out={energy_out}");
    }

    /// W channel (order 0) is never modified by any rotation.
    #[test]
    fn test_w_channel_invariant() {
        let mut frame: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        frame[0] = 1.0;
        let out = rotate_frame(&frame, 3, 1.2, 0.8, 2.1).unwrap();
        assert!(close(out[0], 1.0, 1e-5), "W channel must not change: {}", out[0]);
    }

    /// Wrong channel count returns an error.
    #[test]
    fn test_wrong_channel_count() {
        let mat = build_rotation_matrix(1, 0.0, 0.0, 0.0).unwrap();
        let result = mat.rotate_frame(&[1.0, 2.0]);
        assert!(result.is_err());
    }

    /// max_order > 5 returns an error.
    #[test]
    fn test_order_too_high() {
        assert!(build_rotation_matrix(6, 0.0, 0.0, 0.0).is_err());
    }

    /// 180° yaw applied twice = identity.
    #[test]
    fn test_double_180_yaw_is_identity() {
        let frame = vec![0.5_f32, 0.3, 0.7, 0.1];
        let half = rotate_yaw(&frame, 1, PI).unwrap();
        let full = rotate_yaw(&half, 1, PI).unwrap();
        for (a, b) in full.iter().zip(frame.iter()) {
            assert!(close(*a, *b, 1e-4), "double 180°: {a} vs {b}");
        }
    }

    /// rotate_signal processes multiple frames correctly.
    #[test]
    fn test_rotate_signal_multi_frame() {
        let mat = build_rotation_matrix(1, PI / 4.0, 0.0, 0.0).unwrap();
        let frame0 = vec![1.0_f32, 0.0, 0.0, 0.0];
        let frame1 = vec![0.0_f32, 1.0, 0.0, 0.0];
        let mut signal = frame0.clone();
        signal.extend_from_slice(&frame1);

        let out = mat.rotate_signal(&signal).unwrap();
        assert_eq!(out.len(), 8);

        let expected0 = mat.rotate_frame(&frame0).unwrap();
        let expected1 = mat.rotate_frame(&frame1).unwrap();
        for i in 0..4 {
            assert!(close(out[i], expected0[i], 1e-5));
            assert!(close(out[4 + i], expected1[i], 1e-5));
        }
    }
}
