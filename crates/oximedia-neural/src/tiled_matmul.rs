//! Tiled (blocked) matrix multiplication for improved cache locality.
//!
//! Standard matrix multiplication iterates naively over rows and columns,
//! causing frequent cache misses when matrices exceed L1/L2 cache sizes.
//! This module implements **loop tiling** (also called *blocking*): the
//! matrices are partitioned into square tiles of side `TILE_SIZE` and the
//! multiply-accumulate loop is restructured so that each tile pair fits
//! comfortably in cache.
//!
//! ## Algorithm
//!
//! For `C = A × B` where `A` is `[M, K]` and `B` is `[K, N]`:
//!
//! ```text
//! for tile_i in 0..M step TILE_SIZE:
//!   for tile_j in 0..N step TILE_SIZE:
//!     for tile_k in 0..K step TILE_SIZE:
//!       C[tile_i..tile_i+T, tile_j..tile_j+T] +=
//!         A[tile_i..tile_i+T, tile_k..tile_k+T] *
//!         B[tile_k..tile_k+T, tile_j..tile_j+T]
//! ```
//!
//! ## Example
//!
//! ```rust
//! use oximedia_neural::tiled_matmul::{tiled_matmul, tiled_matmul_with_tile_size};
//! use oximedia_neural::tensor::Tensor;
//!
//! let a = Tensor::ones(vec![8, 4]).unwrap();
//! let b = Tensor::ones(vec![4, 6]).unwrap();
//! let c = tiled_matmul(&a, &b).unwrap();
//! assert_eq!(c.shape(), &[8, 6]);
//! // Each element should equal the inner dimension (dot product of all-ones).
//! assert!((c.data()[0] - 4.0).abs() < 1e-5);
//! ```

#![allow(dead_code)]

use crate::error::NeuralError;
use crate::tensor::Tensor;

/// Default tile side length.  Chosen so that three tiles (A-tile, B-tile,
/// C-tile) of `TILE_SIZE × TILE_SIZE` f32 values fit comfortably in 32 KiB L1
/// cache:  3 × 32² × 4 = 12 KiB.
pub const DEFAULT_TILE_SIZE: usize = 32;

// ─────────────────────────────────────────────────────────────────────────────
// Public API
// ─────────────────────────────────────────────────────────────────────────────

/// Performs tiled matrix multiplication `C = A × B` using the default tile size.
///
/// * `a` — 2-D tensor of shape `[M, K]`.
/// * `b` — 2-D tensor of shape `[K, N]`.
///
/// Returns a tensor of shape `[M, N]`.
pub fn tiled_matmul(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuralError> {
    tiled_matmul_with_tile_size(a, b, DEFAULT_TILE_SIZE)
}

/// Performs tiled matrix multiplication with an explicit tile size.
///
/// `tile_size` must be `>= 1`.  Values that do not evenly divide the matrix
/// dimensions are handled correctly (the edge tiles are smaller).
pub fn tiled_matmul_with_tile_size(
    a: &Tensor,
    b: &Tensor,
    tile_size: usize,
) -> Result<Tensor, NeuralError> {
    // ── Validation ───────────────────────────────────────────────────────────
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NeuralError::InvalidShape(format!(
            "tiled_matmul: both inputs must be 2-D, got ranks {} and {}",
            a.ndim(),
            b.ndim()
        )));
    }
    let m = a.shape()[0];
    let k_a = a.shape()[1];
    let k_b = b.shape()[0];
    let n = b.shape()[1];

    if k_a != k_b {
        return Err(NeuralError::ShapeMismatch(format!(
            "tiled_matmul: inner dimensions mismatch: A is [{}, {}], B is [{}, {}]",
            m, k_a, k_b, n
        )));
    }
    if tile_size == 0 {
        return Err(NeuralError::InvalidShape(
            "tiled_matmul: tile_size must be >= 1".to_string(),
        ));
    }

    let k = k_a;
    let a_data = a.data();
    let b_data = b.data();
    let mut c_data = vec![0.0_f32; m * n];

    // ── Tiled kernel ─────────────────────────────────────────────────────────
    tiled_kernel(a_data, b_data, &mut c_data, m, k, n, tile_size);

    Tensor::from_data(c_data, vec![m, n])
}

/// Performs tiled matrix multiply-accumulate: `C += A × B`.
///
/// All buffers are assumed row-major.
/// * `a` — `[M, K]` in row-major.
/// * `b` — `[K, N]` in row-major.
/// * `c` — `[M, N]` in row-major; values are **accumulated** (not overwritten).
/// * `tile_size` — tile side length.
fn tiled_kernel(
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
    m: usize,
    k: usize,
    n: usize,
    tile_size: usize,
) {
    let ts = tile_size;

    let mut ti = 0;
    while ti < m {
        let tile_m = ts.min(m - ti);

        let mut tj = 0;
        while tj < n {
            let tile_n = ts.min(n - tj);

            let mut tk = 0;
            while tk < k {
                let tile_k = ts.min(k - tk);

                // Micro-kernel: C[ti..ti+tile_m, tj..tj+tile_n] +=
                //               A[ti..ti+tile_m, tk..tk+tile_k] *
                //               B[tk..tk+tile_k, tj..tj+tile_n]
                for ii in 0..tile_m {
                    let row_a = ti + ii;
                    let row_c = ti + ii;
                    for kk in 0..tile_k {
                        let col_a = tk + kk;
                        let a_val = a[row_a * k + col_a];
                        let row_b = tk + kk;
                        for jj in 0..tile_n {
                            let col_b = tj + jj;
                            c[row_c * n + col_b] += a_val * b[row_b * n + col_b];
                        }
                    }
                }

                tk += ts;
            }

            tj += ts;
        }

        ti += ts;
    }
}

/// Performs tiled matrix multiplication with transposed B: `C = A × B^T`.
///
/// * `a` — `[M, K]`
/// * `b` — `[N, K]` (stored row-major, treated as its transpose `[K, N]`)
///
/// Returns `[M, N]`.
pub fn tiled_matmul_bt(a: &Tensor, b: &Tensor) -> Result<Tensor, NeuralError> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(NeuralError::InvalidShape(format!(
            "tiled_matmul_bt: both inputs must be 2-D, got ranks {} and {}",
            a.ndim(),
            b.ndim()
        )));
    }
    let m = a.shape()[0];
    let k_a = a.shape()[1];
    let n = b.shape()[0];
    let k_b = b.shape()[1];

    if k_a != k_b {
        return Err(NeuralError::ShapeMismatch(format!(
            "tiled_matmul_bt: K mismatch: A has K={}, B has K={}",
            k_a, k_b
        )));
    }
    let k = k_a;
    let ts = DEFAULT_TILE_SIZE;

    let a_data = a.data();
    let b_data = b.data();
    let mut c_data = vec![0.0_f32; m * n];

    // B^T[k, j] = b_data[j * k + k]
    let mut ti = 0;
    while ti < m {
        let tile_m = ts.min(m - ti);
        let mut tj = 0;
        while tj < n {
            let tile_n = ts.min(n - tj);
            let mut tk = 0;
            while tk < k {
                let tile_k = ts.min(k - tk);
                for ii in 0..tile_m {
                    let row = ti + ii;
                    for jj in 0..tile_n {
                        let col = tj + jj;
                        let mut acc = 0.0_f32;
                        for kk in 0..tile_k {
                            let a_val = a_data[row * k + (tk + kk)];
                            let b_val = b_data[col * k + (tk + kk)];
                            acc += a_val * b_val;
                        }
                        c_data[row * n + col] += acc;
                    }
                }
                tk += ts;
            }
            tj += ts;
        }
        ti += ts;
    }

    Tensor::from_data(c_data, vec![m, n])
}

/// Computes the Frobenius norm difference between two same-shaped 2-D tensors.
///
/// Useful for verifying that tiled matmul and naive matmul agree.
pub fn frobenius_diff(a: &Tensor, b: &Tensor) -> Result<f32, NeuralError> {
    if a.shape() != b.shape() {
        return Err(NeuralError::ShapeMismatch(format!(
            "frobenius_diff: shapes differ: {:?} vs {:?}",
            a.shape(),
            b.shape()
        )));
    }
    let sum_sq: f32 = a
        .data()
        .iter()
        .zip(b.data().iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum();
    Ok(sum_sq.sqrt())
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// Helper: naive matmul for reference.
    fn naive_matmul(a: &Tensor, b: &Tensor) -> Tensor {
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        let ad = a.data();
        let bd = b.data();
        let mut c = vec![0.0_f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0_f32;
                for kk in 0..k {
                    s += ad[i * k + kk] * bd[kk * n + j];
                }
                c[i * n + j] = s;
            }
        }
        Tensor::from_data(c, vec![m, n]).expect("naive_matmul")
    }

    // ── Basic correctness ────────────────────────────────────────────────────

    #[test]
    fn test_tiled_matmul_ones() {
        let a = Tensor::ones(vec![4, 3]).expect("ones");
        let b = Tensor::ones(vec![3, 5]).expect("ones");
        let c = tiled_matmul(&a, &b).expect("tiled_matmul");
        assert_eq!(c.shape(), &[4, 5]);
        for &v in c.data() {
            assert!((v - 3.0).abs() < 1e-5, "expected 3.0, got {}", v);
        }
    }

    #[test]
    fn test_tiled_matches_naive_square() {
        let data_a: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let data_b: Vec<f32> = (0..16).map(|i| 1.0 - (i as f32) * 0.05).collect();
        let a = Tensor::from_data(data_a, vec![4, 4]).expect("a");
        let b = Tensor::from_data(data_b, vec![4, 4]).expect("b");

        let c_tiled = tiled_matmul(&a, &b).expect("tiled");
        let c_naive = naive_matmul(&a, &b);
        let diff = frobenius_diff(&c_tiled, &c_naive).expect("diff");
        assert!(diff < 1e-4, "Frobenius diff = {}", diff);
    }

    #[test]
    fn test_tiled_matches_naive_non_square() {
        let data_a: Vec<f32> = (0..30).map(|i| i as f32 * 0.01).collect();
        let data_b: Vec<f32> = (0..35).map(|i| (35 - i) as f32 * 0.02).collect();
        let a = Tensor::from_data(data_a, vec![6, 5]).expect("a");
        let b = Tensor::from_data(data_b, vec![5, 7]).expect("b");

        let c_tiled = tiled_matmul(&a, &b).expect("tiled");
        let c_naive = naive_matmul(&a, &b);
        let diff = frobenius_diff(&c_tiled, &c_naive).expect("diff");
        assert!(diff < 1e-4, "Frobenius diff = {}", diff);
    }

    // ── Tile-size variations ─────────────────────────────────────────────────

    #[test]
    fn test_tile_size_1() {
        let a = Tensor::ones(vec![3, 4]).expect("ones");
        let b = Tensor::ones(vec![4, 2]).expect("ones");
        let c = tiled_matmul_with_tile_size(&a, &b, 1).expect("tiled");
        for &v in c.data() {
            assert!((v - 4.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_tile_size_larger_than_matrix() {
        let a = Tensor::ones(vec![2, 3]).expect("ones");
        let b = Tensor::ones(vec![3, 2]).expect("ones");
        let c = tiled_matmul_with_tile_size(&a, &b, 128).expect("tiled");
        assert_eq!(c.shape(), &[2, 2]);
        for &v in c.data() {
            assert!((v - 3.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_tile_size_non_divisible() {
        // 7x5 x 5x9 with tile_size = 3 → non-divisible edges.
        let data_a: Vec<f32> = (0..35).map(|i| (i % 7) as f32).collect();
        let data_b: Vec<f32> = (0..45).map(|i| (i % 5) as f32).collect();
        let a = Tensor::from_data(data_a, vec![7, 5]).expect("a");
        let b = Tensor::from_data(data_b, vec![5, 9]).expect("b");

        let c_tiled = tiled_matmul_with_tile_size(&a, &b, 3).expect("tiled");
        let c_naive = naive_matmul(&a, &b);
        let diff = frobenius_diff(&c_tiled, &c_naive).expect("diff");
        assert!(diff < 1e-3, "Frobenius diff = {}", diff);
    }

    // ── Error handling ───────────────────────────────────────────────────────

    #[test]
    fn test_rank_mismatch_error() {
        let a = Tensor::ones(vec![2, 3, 4]).expect("ones");
        let b = Tensor::ones(vec![4, 5]).expect("ones");
        assert!(tiled_matmul(&a, &b).is_err());
    }

    #[test]
    fn test_inner_dim_mismatch_error() {
        let a = Tensor::ones(vec![3, 4]).expect("ones");
        let b = Tensor::ones(vec![5, 6]).expect("ones");
        assert!(tiled_matmul(&a, &b).is_err());
    }

    #[test]
    fn test_tile_size_zero_error() {
        let a = Tensor::ones(vec![2, 2]).expect("ones");
        let b = Tensor::ones(vec![2, 2]).expect("ones");
        assert!(tiled_matmul_with_tile_size(&a, &b, 0).is_err());
    }

    // ── B-transposed variant ─────────────────────────────────────────────────

    #[test]
    fn test_tiled_matmul_bt_correctness() {
        // A=[3,4], B=[5,4] → C = A × B^T → [3,5]
        let data_a: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..20).map(|i| (20 - i) as f32 * 0.1).collect();
        let a = Tensor::from_data(data_a, vec![3, 4]).expect("a");
        let b = Tensor::from_data(data_b, vec![5, 4]).expect("b");

        let c_bt = tiled_matmul_bt(&a, &b).expect("bt");
        assert_eq!(c_bt.shape(), &[3, 5]);

        // Compare against explicit transpose + naive matmul.
        let bt_data = transpose_2d(b.data(), 5, 4);
        let b_t = Tensor::from_data(bt_data, vec![4, 5]).expect("b_t");
        let c_ref = naive_matmul(&a, &b_t);
        let diff = frobenius_diff(&c_bt, &c_ref).expect("diff");
        assert!(diff < 1e-4, "Frobenius diff = {}", diff);
    }

    #[test]
    fn test_tiled_matmul_bt_k_mismatch_error() {
        let a = Tensor::ones(vec![3, 4]).expect("ones");
        let b = Tensor::ones(vec![5, 3]).expect("ones");
        assert!(tiled_matmul_bt(&a, &b).is_err());
    }

    /// Naive row-major 2-D transpose.
    fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0_f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                out[c * rows + r] = data[r * cols + c];
            }
        }
        out
    }

    // ── Frobenius diff ───────────────────────────────────────────────────────

    #[test]
    fn test_frobenius_diff_identical() {
        let a = Tensor::ones(vec![3, 3]).expect("ones");
        let diff = frobenius_diff(&a, &a).expect("diff");
        assert!(diff < 1e-10);
    }
}
