//! LUT composition, baking, and inversion utilities.
//!
//! This module provides tools for composing multiple LUTs into a single LUT,
//! converting 1D+3D LUT chains into a baked 3D LUT, and computing approximate
//! inverses of 3D LUTs.
//!
//! # Operations
//!
//! - [`compose_3d`] - Chain two 3D LUTs: `result(x) = lut2(lut1(x))`
//! - [`bake_1d_3d`] - Combine a pre-processing 1D LUT with a 3D LUT into one
//! - [`bake_3d_1d`] - Combine a 3D LUT with a post-processing 1D LUT into one
//! - [`invert_3d`] - Compute an approximate inverse of a 3D LUT
//! - [`compose_chain`] - Chain an arbitrary slice of 3D LUTs
//!
//! # Example
//!
//! ```rust
//! use oximedia_lut::{Lut1d, Lut3d, LutInterpolation, LutSize};
//! use oximedia_lut::compose;
//!
//! // Create a gamma pre-1D LUT and a 3D grading LUT
//! let pre = Lut1d::gamma(256, 2.2);
//! let main_lut = Lut3d::identity(LutSize::Size17);
//!
//! // Bake them together into a single 3D LUT
//! let baked = compose::bake_1d_3d(&pre, &main_lut, LutSize::Size17);
//!
//! // Check that pure white input gives white output through the baked LUT
//! let white = [1.0, 1.0, 1.0];
//! let out = baked.apply(&white, LutInterpolation::Tetrahedral);
//! assert!(out[0] > 0.9);
//! ```

use crate::interpolation::LutInterpolation;
use crate::{Lut1d, Lut3d, LutSize, Rgb};

// ============================================================================
// 3D LUT Composition
// ============================================================================

/// Compose two 3D LUTs into a single LUT.
///
/// The resulting LUT is equivalent to applying `lut1` first, then `lut2`:
///
/// ```text
/// result(x) = lut2(lut1(x))
/// ```
///
/// The output LUT is sampled at `output_size` grid points, with tetrahedral
/// interpolation used for intermediate lookups to minimise error.
///
/// # Arguments
///
/// * `lut1` - First LUT to apply
/// * `lut2` - Second LUT to apply to lut1's output
/// * `output_size` - Grid dimension of the resulting LUT
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut3d, LutInterpolation, LutSize};
/// use oximedia_lut::compose;
///
/// let lut1 = Lut3d::from_fn(LutSize::Size17, |rgb| {
///     [rgb[0] * 0.9, rgb[1] * 0.9, rgb[2] * 0.9]
/// });
/// let lut2 = Lut3d::from_fn(LutSize::Size17, |rgb| {
///     [(rgb[0] + 0.05).min(1.0), rgb[1], rgb[2]]
/// });
///
/// let composed = compose::compose_3d(&lut1, &lut2, LutSize::Size17);
/// let input = [0.5, 0.5, 0.5];
/// let out = composed.apply(&input, LutInterpolation::Tetrahedral);
/// assert!(out[0] >= 0.0 && out[0] <= 1.0);
/// ```
#[must_use]
pub fn compose_3d(lut1: &Lut3d, lut2: &Lut3d, output_size: LutSize) -> Lut3d {
    Lut3d::from_fn(output_size, |rgb| {
        let intermediate = lut1.apply(&rgb, LutInterpolation::Tetrahedral);
        lut2.apply(&intermediate, LutInterpolation::Tetrahedral)
    })
}

/// Chain an arbitrary sequence of 3D LUTs into a single LUT.
///
/// Applies LUTs in the order given: `luts[0]` is applied first,
/// `luts[last]` is applied last.
///
/// # Arguments
///
/// * `luts` - Ordered slice of LUTs to chain together (must be non-empty)
/// * `output_size` - Grid dimension of the resulting LUT
///
/// # Panics
///
/// Panics if `luts` is empty.
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut3d, LutInterpolation, LutSize};
/// use oximedia_lut::compose;
///
/// let identity = Lut3d::identity(LutSize::Size17);
/// let chain = compose::compose_chain(&[&identity, &identity], LutSize::Size17);
/// let input = [0.5, 0.3, 0.7];
/// let out = chain.apply(&input, LutInterpolation::Tetrahedral);
/// assert!((out[0] - 0.5).abs() < 0.01);
/// ```
#[must_use]
pub fn compose_chain(luts: &[&Lut3d], output_size: LutSize) -> Lut3d {
    assert!(!luts.is_empty(), "compose_chain requires at least one LUT");

    Lut3d::from_fn(output_size, |rgb| {
        let mut current = rgb;
        for lut in luts {
            current = lut.apply(&current, LutInterpolation::Tetrahedral);
        }
        current
    })
}

// ============================================================================
// 1D + 3D Baking
// ============================================================================

/// Bake a 1D pre-processing LUT and a 3D LUT into a single 3D LUT.
///
/// Equivalent to the pipeline: `pre_1d → main_3d`.
///
/// This is the standard "1D+3D" workflow used in professional colour
/// management pipelines, where the 1D LUT handles scene-linear decoding
/// (e.g. log decode, gamma correction) and the 3D LUT handles the creative
/// grade.
///
/// # Arguments
///
/// * `pre` - 1D LUT applied to each channel independently before the 3D LUT
/// * `main` - 3D LUT applied after the 1D LUT
/// * `output_size` - Grid dimension of the resulting baked 3D LUT
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut1d, Lut3d, LutInterpolation, LutSize};
/// use oximedia_lut::compose;
///
/// let pre = Lut1d::gamma(1024, 2.2);
/// let main_lut = Lut3d::identity(LutSize::Size33);
/// let baked = compose::bake_1d_3d(&pre, &main_lut, LutSize::Size33);
///
/// // The baked LUT should match applying both LUTs in sequence
/// let input = [0.5, 0.3, 0.7];
/// let sequential_r1 = pre.apply(&input, LutInterpolation::Linear);
/// let sequential_out = main_lut.apply(&sequential_r1, LutInterpolation::Tetrahedral);
/// let baked_out = baked.apply(&input, LutInterpolation::Tetrahedral);
/// assert!((sequential_out[0] - baked_out[0]).abs() < 0.01);
/// ```
#[must_use]
pub fn bake_1d_3d(pre: &Lut1d, main: &Lut3d, output_size: LutSize) -> Lut3d {
    Lut3d::from_fn(output_size, |rgb| {
        // Apply 1D LUT to each channel first
        let after_1d = pre.apply(&rgb, LutInterpolation::Linear);
        // Then apply the 3D LUT
        main.apply(&after_1d, LutInterpolation::Tetrahedral)
    })
}

/// Bake a 3D LUT and a 1D post-processing LUT into a single 3D LUT.
///
/// Equivalent to the pipeline: `main_3d → post_1d`.
///
/// This is useful when a 1D LUT is applied after the grade, e.g. for
/// applying a gamma encoding or a display LUT.
///
/// # Arguments
///
/// * `main` - 3D LUT applied first
/// * `post` - 1D LUT applied after the 3D LUT
/// * `output_size` - Grid dimension of the resulting baked 3D LUT
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut1d, Lut3d, LutInterpolation, LutSize};
/// use oximedia_lut::compose;
///
/// let main_lut = Lut3d::identity(LutSize::Size17);
/// let post = Lut1d::gamma(256, 1.0 / 2.2);
/// let baked = compose::bake_3d_1d(&main_lut, &post, LutSize::Size17);
///
/// let input = [0.5, 0.5, 0.5];
/// let out = baked.apply(&input, LutInterpolation::Tetrahedral);
/// assert!(out[0] >= 0.0 && out[0] <= 1.0);
/// ```
#[must_use]
pub fn bake_3d_1d(main: &Lut3d, post: &Lut1d, output_size: LutSize) -> Lut3d {
    Lut3d::from_fn(output_size, |rgb| {
        // Apply 3D LUT first
        let after_3d = main.apply(&rgb, LutInterpolation::Tetrahedral);
        // Then apply 1D LUT per-channel
        post.apply(&after_3d, LutInterpolation::Linear)
    })
}

/// Bake a full 1D + 3D + 1D pipeline into a single 3D LUT.
///
/// Equivalent to: `pre_1d → main_3d → post_1d`.
///
/// # Arguments
///
/// * `pre` - 1D LUT applied before the 3D LUT
/// * `main` - 3D LUT in the middle
/// * `post` - 1D LUT applied after the 3D LUT
/// * `output_size` - Grid dimension of the resulting baked 3D LUT
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut1d, Lut3d, LutInterpolation, LutSize};
/// use oximedia_lut::compose;
///
/// let pre = Lut1d::gamma(256, 2.2);
/// let main_lut = Lut3d::identity(LutSize::Size17);
/// let post = Lut1d::gamma(256, 1.0 / 2.2);
/// let baked = compose::bake_full_pipeline(&pre, &main_lut, &post, LutSize::Size17);
///
/// // With identity 3D and inverse gamma 1D pair, should be close to identity
/// let input = [0.5, 0.5, 0.5];
/// let out = baked.apply(&input, LutInterpolation::Tetrahedral);
/// assert!((out[0] - input[0]).abs() < 0.02);
/// ```
#[must_use]
pub fn bake_full_pipeline(pre: &Lut1d, main: &Lut3d, post: &Lut1d, output_size: LutSize) -> Lut3d {
    Lut3d::from_fn(output_size, |rgb| {
        let after_pre = pre.apply(&rgb, LutInterpolation::Linear);
        let after_3d = main.apply(&after_pre, LutInterpolation::Tetrahedral);
        post.apply(&after_3d, LutInterpolation::Linear)
    })
}

// ============================================================================
// 3D LUT Inversion
// ============================================================================

/// Compute an approximate inverse of a 3D LUT.
///
/// Uses iterative Newton–Raphson-style refinement to find input values that
/// produce target output values. The approximation quality depends on:
/// - The smoothness and monotonicity of the original LUT
/// - The number of iterations (more = more accurate, slower)
/// - The output grid size (larger = finer accuracy)
///
/// For highly non-monotonic LUTs (e.g. strong hue rotation), the inverse
/// may not converge cleanly — the caller should validate the result.
///
/// # Arguments
///
/// * `lut` - The 3D LUT to invert
/// * `output_size` - Grid dimension of the output inverse LUT
/// * `iterations` - Number of Newton–Raphson refinement steps (4–8 typical)
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut3d, LutInterpolation, LutSize};
/// use oximedia_lut::compose;
///
/// // Create a simple LUT that scales all channels by 0.8
/// let lut = Lut3d::from_fn(LutSize::Size17, |rgb| {
///     [rgb[0] * 0.8, rgb[1] * 0.8, rgb[2] * 0.8]
/// });
///
/// // Invert it
/// let inv = compose::invert_3d(&lut, LutSize::Size17, 6);
///
/// // Applying lut then inv should approximately return to original
/// let input = [0.5, 0.3, 0.6];
/// let after_lut = lut.apply(&input, LutInterpolation::Tetrahedral);
/// let recovered = inv.apply(&after_lut, LutInterpolation::Tetrahedral);
/// assert!((recovered[0] - input[0]).abs() < 0.05);
/// assert!((recovered[1] - input[1]).abs() < 0.05);
/// assert!((recovered[2] - input[2]).abs() < 0.05);
/// ```
#[must_use]
pub fn invert_3d(lut: &Lut3d, output_size: LutSize, iterations: u32) -> Lut3d {
    let size = output_size.as_usize();
    let mut inv = Lut3d::identity(output_size);

    for r_idx in 0..size {
        for g_idx in 0..size {
            for b_idx in 0..size {
                // The target output value (what we want our forward LUT to produce)
                let target: Rgb = [
                    r_idx as f64 / (size - 1) as f64,
                    g_idx as f64 / (size - 1) as f64,
                    b_idx as f64 / (size - 1) as f64,
                ];

                // Solve: find x such that lut(x) ≈ target
                let input = find_inverse(lut, &target, iterations);
                inv.set(r_idx, g_idx, b_idx, input);
            }
        }
    }

    inv
}

/// Find the input value that produces the target output from the LUT.
///
/// Uses iterative gradient descent / Newton's method with clamping.
#[must_use]
fn find_inverse(lut: &Lut3d, target: &Rgb, iterations: u32) -> Rgb {
    // Initial guess: assume the LUT is close to identity
    let mut x = *target;

    let step = 1.0e-4; // Finite difference step for Jacobian approximation

    for _ in 0..iterations {
        // Current residual: forward(x) - target
        let fx = lut.apply(&x, LutInterpolation::Tetrahedral);
        let residual = [fx[0] - target[0], fx[1] - target[1], fx[2] - target[2]];

        // Check convergence
        let err = residual[0] * residual[0] + residual[1] * residual[1] + residual[2] * residual[2];
        if err < 1e-12 {
            break;
        }

        // Compute 3x3 Jacobian using finite differences
        let jac = compute_jacobian(lut, &x, step);

        // Solve J * delta = residual using Cramer's rule (small 3x3 system)
        let delta = solve_linear_3x3(&jac, &residual);

        // Update x with Newton step (clamped to valid range)
        x[0] = (x[0] - delta[0]).clamp(0.0, 1.0);
        x[1] = (x[1] - delta[1]).clamp(0.0, 1.0);
        x[2] = (x[2] - delta[2]).clamp(0.0, 1.0);
    }

    x
}

/// Compute the 3x3 Jacobian of the LUT at position `x` using finite differences.
///
/// Returns `jac[i][j] = d(output_i) / d(input_j)`.
#[must_use]
fn compute_jacobian(lut: &Lut3d, x: &Rgb, h: f64) -> [[f64; 3]; 3] {
    let f0 = lut.apply(x, LutInterpolation::Tetrahedral);

    let mut jac = [[0.0_f64; 3]; 3];

    for j in 0..3 {
        let mut xh = *x;
        xh[j] = (xh[j] + h).clamp(0.0, 1.0);
        let fh = lut.apply(&xh, LutInterpolation::Tetrahedral);
        // Adjust h for clamping at boundary
        let actual_h = xh[j] - x[j];
        if actual_h.abs() > 1e-10 {
            for i in 0..3 {
                jac[i][j] = (fh[i] - f0[i]) / actual_h;
            }
        } else {
            // At boundary: try backward difference
            let mut xm = *x;
            xm[j] = (xm[j] - h).clamp(0.0, 1.0);
            let fm = lut.apply(&xm, LutInterpolation::Tetrahedral);
            let actual_hm = x[j] - xm[j];
            if actual_hm.abs() > 1e-10 {
                for i in 0..3 {
                    jac[i][j] = (f0[i] - fm[i]) / actual_hm;
                }
            }
            // else leave as 0 (degenerate)
        }
    }

    jac
}

/// Solve a 3x3 linear system `A * x = b` using Cramer's rule.
///
/// Returns the zero vector if the matrix is singular.
#[must_use]
fn solve_linear_3x3(a: &[[f64; 3]; 3], b: &[f64; 3]) -> [f64; 3] {
    let det = det3x3(a);

    if det.abs() < 1e-12 {
        // Singular matrix: fall back to gradient-descent-style step
        // Use the transpose (A^T b) as a pseudo-gradient
        return [
            a[0][0] * b[0] + a[1][0] * b[1] + a[2][0] * b[2],
            a[0][1] * b[0] + a[1][1] * b[1] + a[2][1] * b[2],
            a[0][2] * b[0] + a[1][2] * b[1] + a[2][2] * b[2],
        ];
    }

    // Cramer's rule
    let det_x = det3x3(&[
        [b[0], a[0][1], a[0][2]],
        [b[1], a[1][1], a[1][2]],
        [b[2], a[2][1], a[2][2]],
    ]);
    let det_y = det3x3(&[
        [a[0][0], b[0], a[0][2]],
        [a[1][0], b[1], a[1][2]],
        [a[2][0], b[2], a[2][2]],
    ]);
    let det_z = det3x3(&[
        [a[0][0], a[0][1], b[0]],
        [a[1][0], a[1][1], b[1]],
        [a[2][0], a[2][1], b[2]],
    ]);

    [det_x / det, det_y / det, det_z / det]
}

/// Compute the determinant of a 3x3 matrix.
#[must_use]
#[inline]
fn det3x3(m: &[[f64; 3]; 3]) -> f64 {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

// ============================================================================
// Image Processing
// ============================================================================

/// Apply a 3D LUT to a flat RGBA byte image (8-bits per channel).
///
/// Input and output are contiguous RGBA byte slices. Each group of 4 bytes
/// is one pixel: `[R, G, B, A]`. The alpha channel is passed through
/// unmodified.
///
/// # Arguments
///
/// * `lut` - The 3D LUT to apply
/// * `pixels` - Input RGBA byte slice (length must be a multiple of 4)
/// * `interpolation` - Interpolation method
///
/// # Panics
///
/// Panics if `pixels.len()` is not a multiple of 4.
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut3d, LutInterpolation, LutSize};
/// use oximedia_lut::compose;
///
/// let lut = Lut3d::identity(LutSize::Size17);
/// let pixels: Vec<u8> = vec![128, 64, 200, 255, 0, 0, 0, 255];
/// let out = compose::apply_lut_to_image_rgba8(&lut, &pixels, LutInterpolation::Trilinear);
/// assert_eq!(out.len(), pixels.len());
/// // Identity LUT: output close to input
/// assert!((out[0] as i32 - 128).abs() <= 2);
/// ```
#[must_use]
pub fn apply_lut_to_image_rgba8(
    lut: &Lut3d,
    pixels: &[u8],
    interpolation: LutInterpolation,
) -> Vec<u8> {
    assert_eq!(
        pixels.len() % 4,
        0,
        "pixel data must be a multiple of 4 bytes"
    );
    let mut output = vec![0u8; pixels.len()];

    for (chunk_in, chunk_out) in pixels.chunks_exact(4).zip(output.chunks_exact_mut(4)) {
        let r = f64::from(chunk_in[0]) / 255.0;
        let g = f64::from(chunk_in[1]) / 255.0;
        let b = f64::from(chunk_in[2]) / 255.0;
        let a = chunk_in[3];

        let out_rgb = lut.apply(&[r, g, b], interpolation);

        chunk_out[0] = (out_rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        chunk_out[1] = (out_rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        chunk_out[2] = (out_rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;
        chunk_out[3] = a;
    }

    output
}

/// Apply a 3D LUT to a flat RGB byte image (8-bits per channel, no alpha).
///
/// Input and output are contiguous RGB byte slices. Each group of 3 bytes
/// is one pixel: `[R, G, B]`.
///
/// # Panics
///
/// Panics if `pixels.len()` is not a multiple of 3.
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut3d, LutInterpolation, LutSize};
/// use oximedia_lut::compose;
///
/// let lut = Lut3d::identity(LutSize::Size17);
/// let pixels: Vec<u8> = vec![128, 64, 200, 255, 128, 0];
/// let out = compose::apply_lut_to_image_rgb8(&lut, &pixels, LutInterpolation::Trilinear);
/// assert_eq!(out.len(), pixels.len());
/// assert!((out[0] as i32 - 128).abs() <= 2);
/// ```
#[must_use]
pub fn apply_lut_to_image_rgb8(
    lut: &Lut3d,
    pixels: &[u8],
    interpolation: LutInterpolation,
) -> Vec<u8> {
    assert_eq!(
        pixels.len() % 3,
        0,
        "pixel data must be a multiple of 3 bytes"
    );
    let mut output = vec![0u8; pixels.len()];

    for (chunk_in, chunk_out) in pixels.chunks_exact(3).zip(output.chunks_exact_mut(3)) {
        let r = f64::from(chunk_in[0]) / 255.0;
        let g = f64::from(chunk_in[1]) / 255.0;
        let b = f64::from(chunk_in[2]) / 255.0;

        let out_rgb = lut.apply(&[r, g, b], interpolation);

        chunk_out[0] = (out_rgb[0].clamp(0.0, 1.0) * 255.0).round() as u8;
        chunk_out[1] = (out_rgb[1].clamp(0.0, 1.0) * 255.0).round() as u8;
        chunk_out[2] = (out_rgb[2].clamp(0.0, 1.0) * 255.0).round() as u8;
    }

    output
}

/// Apply a 3D LUT to a slice of f64 RGB triples.
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut3d, LutInterpolation, LutSize};
/// use oximedia_lut::compose;
///
/// let lut = Lut3d::identity(LutSize::Size17);
/// let pixels: Vec<[f64; 3]> = vec![[0.5, 0.3, 0.7], [1.0, 0.0, 0.0]];
/// let out = compose::apply_lut_to_image_f64(&lut, &pixels, LutInterpolation::Tetrahedral);
/// assert_eq!(out.len(), 2);
/// assert!((out[0][0] - 0.5).abs() < 0.01);
/// ```
#[must_use]
pub fn apply_lut_to_image_f64(
    lut: &Lut3d,
    pixels: &[[f64; 3]],
    interpolation: LutInterpolation,
) -> Vec<[f64; 3]> {
    pixels
        .iter()
        .map(|pixel| lut.apply(pixel, interpolation))
        .collect()
}

// ============================================================================
// Blend / Mix
// ============================================================================

/// Blend two 3D LUTs together at a given mix ratio.
///
/// Returns a new LUT whose output is `lerp(lut1(x), lut2(x), mix)`.
/// A `mix` of `0.0` gives `lut1`, a `mix` of `1.0` gives `lut2`.
///
/// # Arguments
///
/// * `lut1` - First LUT (mix = 0.0)
/// * `lut2` - Second LUT (mix = 1.0)
/// * `mix` - Blend ratio in `[0.0, 1.0]`
/// * `output_size` - Grid dimension of the resulting LUT
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut3d, LutInterpolation, LutSize};
/// use oximedia_lut::compose;
///
/// let identity = Lut3d::identity(LutSize::Size17);
/// let dark = Lut3d::from_fn(LutSize::Size17, |rgb| {
///     [rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5]
/// });
///
/// // 50% blend between identity and dark → 75% of original brightness
/// let blended = compose::blend_3d(&identity, &dark, 0.5, LutSize::Size17);
/// let input = [0.8, 0.6, 0.4];
/// let out = blended.apply(&input, LutInterpolation::Tetrahedral);
/// assert!((out[0] - 0.6).abs() < 0.05); // 0.8 * 0.75
/// ```
#[must_use]
pub fn blend_3d(lut1: &Lut3d, lut2: &Lut3d, mix: f64, output_size: LutSize) -> Lut3d {
    let mix = mix.clamp(0.0, 1.0);

    Lut3d::from_fn(output_size, |rgb| {
        let out1 = lut1.apply(&rgb, LutInterpolation::Tetrahedral);
        let out2 = lut2.apply(&rgb, LutInterpolation::Tetrahedral);
        [
            lerp(out1[0], out2[0], mix),
            lerp(out1[1], out2[1], mix),
            lerp(out1[2], out2[2], mix),
        ]
    })
}

/// Linear interpolation helper.
#[inline]
#[must_use]
fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

// ============================================================================
// Analysis
// ============================================================================

/// Measure the maximum error between two 3D LUTs sampled at every grid point.
///
/// Returns the maximum per-channel absolute difference found across all
/// grid points in `lut_a`.
///
/// # Example
///
/// ```rust
/// use oximedia_lut::{Lut3d, LutSize};
/// use oximedia_lut::compose;
///
/// let a = Lut3d::identity(LutSize::Size17);
/// let b = Lut3d::identity(LutSize::Size17);
/// let err = compose::max_error(&a, &b);
/// assert!(err < 1e-10);
/// ```
#[must_use]
pub fn max_error(lut_a: &Lut3d, lut_b: &Lut3d) -> f64 {
    let size_a = lut_a.size();
    let size_b = lut_b.size();
    let size = size_a.min(size_b);

    let mut max_err = 0.0_f64;

    for r in 0..size {
        for g in 0..size {
            for b in 0..size {
                let va = lut_a.get(r, g, b);
                let vb = lut_b.get(r, g, b);
                let err = (va[0] - vb[0])
                    .abs()
                    .max((va[1] - vb[1]).abs())
                    .max((va[2] - vb[2]).abs());
                if err > max_err {
                    max_err = err;
                }
            }
        }
    }

    max_err
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::LutInterpolation;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    // --- compose_3d ---

    #[test]
    fn test_compose_3d_identity_identity() {
        let id = Lut3d::identity(LutSize::Size17);
        let composed = compose_3d(&id, &id, LutSize::Size17);
        let input = [0.5, 0.3, 0.7];
        let out = composed.apply(&input, LutInterpolation::Tetrahedral);
        assert!(approx_eq(out[0], input[0], 0.01));
        assert!(approx_eq(out[1], input[1], 0.01));
        assert!(approx_eq(out[2], input[2], 0.01));
    }

    #[test]
    fn test_compose_3d_scale() {
        // lut1 scales by 0.5, lut2 scales by 2.0 → composed should be ~identity
        let lut1 = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5]
        });
        let lut2 = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [
                (rgb[0] * 2.0).min(1.0),
                (rgb[1] * 2.0).min(1.0),
                (rgb[2] * 2.0).min(1.0),
            ]
        });

        let composed = compose_3d(&lut1, &lut2, LutSize::Size17);
        let input = [0.4, 0.3, 0.2]; // Values that won't clip after doubling
        let out = composed.apply(&input, LutInterpolation::Tetrahedral);
        assert!(approx_eq(out[0], input[0], 0.05));
        assert!(approx_eq(out[1], input[1], 0.05));
        assert!(approx_eq(out[2], input[2], 0.05));
    }

    // --- compose_chain ---

    #[test]
    fn test_compose_chain_single() {
        let id = Lut3d::identity(LutSize::Size17);
        let chain = compose_chain(&[&id], LutSize::Size17);
        let input = [0.5, 0.5, 0.5];
        let out = chain.apply(&input, LutInterpolation::Tetrahedral);
        assert!(approx_eq(out[0], 0.5, 0.01));
    }

    #[test]
    fn test_compose_chain_three() {
        let id = Lut3d::identity(LutSize::Size17);
        let chain = compose_chain(&[&id, &id, &id], LutSize::Size17);
        let input = [0.3, 0.6, 0.8];
        let out = chain.apply(&input, LutInterpolation::Tetrahedral);
        assert!(approx_eq(out[0], input[0], 0.01));
        assert!(approx_eq(out[1], input[1], 0.01));
        assert!(approx_eq(out[2], input[2], 0.01));
    }

    // --- bake_1d_3d ---

    #[test]
    fn test_bake_1d_3d_matches_sequential() {
        let pre = Lut1d::gamma(256, 2.2);
        let main_lut = Lut3d::identity(LutSize::Size17);
        let baked = bake_1d_3d(&pre, &main_lut, LutSize::Size17);

        let input = [0.5, 0.3, 0.7];
        let sequential = {
            let after_1d = pre.apply(&input, LutInterpolation::Linear);
            main_lut.apply(&after_1d, LutInterpolation::Tetrahedral)
        };
        let baked_out = baked.apply(&input, LutInterpolation::Tetrahedral);

        assert!(approx_eq(sequential[0], baked_out[0], 0.02));
        assert!(approx_eq(sequential[1], baked_out[1], 0.02));
        assert!(approx_eq(sequential[2], baked_out[2], 0.02));
    }

    #[test]
    fn test_bake_1d_3d_identity_pre() {
        let pre = Lut1d::identity(256);
        let main_lut = Lut3d::from_fn(LutSize::Size17, |rgb| [rgb[0] * 0.5, rgb[1], rgb[2]]);
        let baked = bake_1d_3d(&pre, &main_lut, LutSize::Size17);

        let input = [0.8, 0.6, 0.4];
        let out = baked.apply(&input, LutInterpolation::Tetrahedral);
        assert!(approx_eq(out[0], 0.4, 0.05)); // 0.8 * 0.5
        assert!(approx_eq(out[1], 0.6, 0.05));
    }

    // --- bake_3d_1d ---

    #[test]
    fn test_bake_3d_1d_identity_post() {
        let main_lut = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [(rgb[0] + 0.1).min(1.0), rgb[1], rgb[2]]
        });
        let post = Lut1d::identity(256);
        let baked = bake_3d_1d(&main_lut, &post, LutSize::Size17);

        let input = [0.5, 0.5, 0.5];
        let out = baked.apply(&input, LutInterpolation::Tetrahedral);
        assert!(approx_eq(out[0], 0.6, 0.05)); // 0.5 + 0.1
    }

    // --- bake_full_pipeline ---

    #[test]
    fn test_bake_full_pipeline_gamma_roundtrip() {
        let pre = Lut1d::gamma(1024, 2.2);
        let main_lut = Lut3d::identity(LutSize::Size17);
        let post = Lut1d::gamma(1024, 1.0 / 2.2);
        let baked = bake_full_pipeline(&pre, &main_lut, &post, LutSize::Size17);

        let input = [0.5, 0.5, 0.5];
        let out = baked.apply(&input, LutInterpolation::Tetrahedral);
        // gamma → identity → inverse gamma ≈ identity
        assert!(approx_eq(out[0], input[0], 0.03));
    }

    // --- invert_3d ---

    #[test]
    fn test_invert_3d_identity() {
        let id = Lut3d::identity(LutSize::Size17);
        let inv = invert_3d(&id, LutSize::Size17, 4);
        let input = [0.5, 0.3, 0.7];
        let out = inv.apply(&input, LutInterpolation::Tetrahedral);
        // Inverse of identity should be identity
        assert!(approx_eq(out[0], input[0], 0.02));
        assert!(approx_eq(out[1], input[1], 0.02));
        assert!(approx_eq(out[2], input[2], 0.02));
    }

    #[test]
    fn test_invert_3d_scale_roundtrip() {
        // Monotonic scale LUT: easy to invert accurately
        let scale = 0.75;
        let lut = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [rgb[0] * scale, rgb[1] * scale, rgb[2] * scale]
        });
        let inv = invert_3d(&lut, LutSize::Size17, 8);

        let input = [0.5, 0.3, 0.6];
        let after_fwd = lut.apply(&input, LutInterpolation::Tetrahedral);
        let recovered = inv.apply(&after_fwd, LutInterpolation::Tetrahedral);

        assert!(approx_eq(recovered[0], input[0], 0.05));
        assert!(approx_eq(recovered[1], input[1], 0.05));
        assert!(approx_eq(recovered[2], input[2], 0.05));
    }

    // --- image processing ---

    #[test]
    fn test_apply_lut_to_image_rgba8_identity() {
        let lut = Lut3d::identity(LutSize::Size17);
        let pixels: Vec<u8> = vec![200, 100, 50, 255, 0, 128, 64, 200];
        let out = apply_lut_to_image_rgba8(&lut, &pixels, LutInterpolation::Trilinear);
        assert_eq!(out.len(), 8);
        // Alpha pass-through
        assert_eq!(out[3], 255);
        assert_eq!(out[7], 200);
        // Approximate identity
        assert!((out[0] as i32 - 200).abs() <= 2);
        assert!((out[1] as i32 - 100).abs() <= 2);
    }

    #[test]
    fn test_apply_lut_to_image_rgb8_identity() {
        let lut = Lut3d::identity(LutSize::Size17);
        let pixels: Vec<u8> = vec![100, 150, 200, 255, 0, 128];
        let out = apply_lut_to_image_rgb8(&lut, &pixels, LutInterpolation::Trilinear);
        assert_eq!(out.len(), 6);
        assert!((out[0] as i32 - 100).abs() <= 2);
        assert!((out[4] as i32 - 0).abs() <= 2);
    }

    #[test]
    fn test_apply_lut_to_image_f64_identity() {
        let lut = Lut3d::identity(LutSize::Size17);
        let pixels = vec![[0.5_f64, 0.3, 0.7], [1.0, 0.0, 0.0]];
        let out = apply_lut_to_image_f64(&lut, &pixels, LutInterpolation::Tetrahedral);
        assert_eq!(out.len(), 2);
        assert!(approx_eq(out[0][0], 0.5, 0.01));
        assert!(approx_eq(out[1][0], 1.0, 0.01));
    }

    // --- blend_3d ---

    #[test]
    fn test_blend_3d_at_zero_is_lut1() {
        let id = Lut3d::identity(LutSize::Size17);
        let dark = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5]
        });
        let blended = blend_3d(&id, &dark, 0.0, LutSize::Size17);
        let input = [0.6, 0.4, 0.8];
        let out = blended.apply(&input, LutInterpolation::Tetrahedral);
        assert!(approx_eq(out[0], input[0], 0.02));
    }

    #[test]
    fn test_blend_3d_at_one_is_lut2() {
        let id = Lut3d::identity(LutSize::Size17);
        let dark = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5]
        });
        let blended = blend_3d(&id, &dark, 1.0, LutSize::Size17);
        let input = [0.6, 0.4, 0.8];
        let out = blended.apply(&input, LutInterpolation::Tetrahedral);
        assert!(approx_eq(out[0], input[0] * 0.5, 0.02));
    }

    #[test]
    fn test_blend_3d_midpoint() {
        let id = Lut3d::identity(LutSize::Size17);
        let dark = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5]
        });
        let blended = blend_3d(&id, &dark, 0.5, LutSize::Size17);
        let input = [0.8, 0.8, 0.8];
        let out = blended.apply(&input, LutInterpolation::Tetrahedral);
        // 0.5 * 0.8 + 0.5 * 0.4 = 0.6
        assert!(approx_eq(out[0], 0.6, 0.03));
    }

    // --- max_error ---

    #[test]
    fn test_max_error_identical_luts() {
        let a = Lut3d::identity(LutSize::Size17);
        let b = Lut3d::identity(LutSize::Size17);
        let err = max_error(&a, &b);
        assert!(err < 1e-10);
    }

    #[test]
    fn test_max_error_different_luts() {
        let a = Lut3d::identity(LutSize::Size17);
        let b = Lut3d::from_fn(LutSize::Size17, |rgb| {
            [rgb[0] * 0.5, rgb[1] * 0.5, rgb[2] * 0.5]
        });
        let err = max_error(&a, &b);
        assert!(err > 0.1); // Should be substantial
    }

    // --- helper functions ---

    #[test]
    fn test_det3x3_identity() {
        let m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        assert!(approx_eq(det3x3(&m), 1.0, 1e-10));
    }

    #[test]
    fn test_det3x3_singular() {
        let m = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        // Known singular
        assert!(det3x3(&m).abs() < 1e-6);
    }

    #[test]
    fn test_solve_linear_identity() {
        let a = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let b = [3.0, 5.0, 7.0];
        let x = solve_linear_3x3(&a, &b);
        assert!(approx_eq(x[0], 3.0, 1e-10));
        assert!(approx_eq(x[1], 5.0, 1e-10));
        assert!(approx_eq(x[2], 7.0, 1e-10));
    }
}
