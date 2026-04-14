//! GPU-backed (and CPU-fallback) image affine transform.
//!
//! Provides a 2×3 affine matrix type plus rotation, scale, translation,
//! and composition factories.  The `apply_affine` function uses bilinear
//! sampling with inverse-mapping for high-quality results.

use crate::error::{AccelError, AccelResult};
use rayon::prelude::*;

/// WGSL compute shader for GPU-side affine transform (inverse mapping with
/// bilinear interpolation).
pub const AFFINE_SHADER_WGSL: &str = r#"
// Affine transform shader.
// Inverse maps each destination pixel back to the source image using
// a 2x3 affine matrix (6 f32 coefficients stored in a uniform).
//
// Matrix layout:
//   [ m00  m01  tx ]
//   [ m10  m11  ty ]
//
// Uniform: [src_w, src_h, dst_w, dst_h, m00, m01, tx, m10, m11, ty]

struct AffineParams {
    src_width:  u32,
    src_height: u32,
    dst_width:  u32,
    dst_height: u32,
    m00: f32, m01: f32, tx: f32,
    m10: f32, m11: f32, ty: f32,
}

@group(0) @binding(0) var<storage, read>       src:    array<u32>;
@group(0) @binding(1) var<storage, read_write> dst:    array<u32>;
@group(0) @binding(2) var<uniform>             params: AffineParams;

fn unpack_rgba(v: u32) -> vec4<f32> {
    return vec4<f32>(
        f32((v >> 24u) & 0xFFu) / 255.0,
        f32((v >> 16u) & 0xFFu) / 255.0,
        f32((v >>  8u) & 0xFFu) / 255.0,
        f32( v         & 0xFFu) / 255.0,
    );
}

fn pack_rgba(c: vec4<f32>) -> u32 {
    let r = u32(clamp(c.r * 255.0, 0.0, 255.0));
    let g = u32(clamp(c.g * 255.0, 0.0, 255.0));
    let b = u32(clamp(c.b * 255.0, 0.0, 255.0));
    let a = u32(clamp(c.a * 255.0, 0.0, 255.0));
    return (r << 24u) | (g << 16u) | (b << 8u) | a;
}

fn fetch_bilinear(u: f32, v: f32) -> vec4<f32> {
    let x0 = u32(clamp(floor(u), 0.0, f32(params.src_width)  - 1.0));
    let y0 = u32(clamp(floor(v), 0.0, f32(params.src_height) - 1.0));
    let x1 = min(x0 + 1u, params.src_width  - 1u);
    let y1 = min(y0 + 1u, params.src_height - 1u);
    let fx = fract(max(u, 0.0));
    let fy = fract(max(v, 0.0));
    let c00 = unpack_rgba(src[y0 * params.src_width + x0]);
    let c10 = unpack_rgba(src[y0 * params.src_width + x1]);
    let c01 = unpack_rgba(src[y1 * params.src_width + x0]);
    let c11 = unpack_rgba(src[y1 * params.src_width + x1]);
    return mix(mix(c00, c10, fx), mix(c01, c11, fx), fy);
}

@compute @workgroup_size(16, 16, 1)
fn affine_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let dx = gid.x;
    let dy = gid.y;
    if (dx >= params.dst_width || dy >= params.dst_height) { return; }

    // Inverse-map destination centre to source space
    let fx = f32(dx) + 0.5;
    let fy = f32(dy) + 0.5;
    let sx = params.m00 * fx + params.m01 * fy + params.tx - 0.5;
    let sy = params.m10 * fx + params.m11 * fy + params.ty - 0.5;

    var col: vec4<f32>;
    if (sx < 0.0 || sy < 0.0 || sx >= f32(params.src_width) || sy >= f32(params.src_height)) {
        col = vec4<f32>(0.0, 0.0, 0.0, 0.0); // transparent black for out-of-bounds
    } else {
        col = fetch_bilinear(sx, sy);
    }

    dst[dy * params.dst_width + dx] = pack_rgba(col);
}
"#;

/// 2×3 affine transform matrix.
///
/// Transforms a source point `(xs, ys)` to a destination point `(xd, yd)`:
///
/// ```text
/// xd = m[0] * xs + m[1] * ys + m[2]
/// yd = m[3] * xs + m[4] * ys + m[5]
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AffineTransform {
    /// Coefficients `[m00, m01, tx, m10, m11, ty]` (row-major 2×3).
    pub m: [f32; 6],
}

impl AffineTransform {
    /// Identity transform: `(x, y) → (x, y)`.
    #[must_use]
    pub fn identity() -> Self {
        Self {
            m: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        }
    }

    /// Counter-clockwise rotation by `angle_radians` around the origin.
    ///
    /// To rotate around an image centre, chain with `translation`.
    #[must_use]
    pub fn rotation(angle_radians: f32) -> Self {
        let (s, c) = angle_radians.sin_cos();
        // Forward map: xd = c*xs - s*ys, yd = s*xs + c*ys
        Self {
            m: [c, -s, 0.0, s, c, 0.0],
        }
    }

    /// Uniform or anisotropic scale: `(x, y) → (sx*x, sy*y)`.
    #[must_use]
    pub fn scale(sx: f32, sy: f32) -> Self {
        Self {
            m: [sx, 0.0, 0.0, 0.0, sy, 0.0],
        }
    }

    /// Translation: `(x, y) → (x + tx, y + ty)`.
    #[must_use]
    pub fn translation(tx: f32, ty: f32) -> Self {
        Self {
            m: [1.0, 0.0, tx, 0.0, 1.0, ty],
        }
    }

    /// Compose `self` with `other`: the resulting transform first applies
    /// `self` then `other` (i.e. `other ∘ self`).
    ///
    /// For 2×3 matrices in homogeneous form:
    /// ```text
    /// M_composed = M_other × M_self   (with implicit [0 0 1] last row)
    /// ```
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let a = self.m;
        let b = other.m;
        // Row 0 of result
        let m00 = b[0] * a[0] + b[1] * a[3];
        let m01 = b[0] * a[1] + b[1] * a[4];
        let tx = b[0] * a[2] + b[1] * a[5] + b[2];
        // Row 1 of result
        let m10 = b[3] * a[0] + b[4] * a[3];
        let m11 = b[3] * a[1] + b[4] * a[4];
        let ty = b[3] * a[2] + b[4] * a[5] + b[5];
        Self {
            m: [m00, m01, tx, m10, m11, ty],
        }
    }

    /// Compute the inverse of this affine transform, if it is invertible.
    ///
    /// Returns `None` when the determinant is zero (degenerate matrix).
    #[must_use]
    pub fn try_inverse(&self) -> Option<Self> {
        let [m00, m01, tx, m10, m11, ty] = self.m;
        let det = m00 * m11 - m01 * m10;
        if det.abs() < 1e-10 {
            return None;
        }
        let inv_det = 1.0 / det;
        let im00 = m11 * inv_det;
        let im01 = -m01 * inv_det;
        let im10 = -m10 * inv_det;
        let im11 = m00 * inv_det;
        let itx = -(im00 * tx + im01 * ty);
        let ity = -(im10 * tx + im11 * ty);
        Some(Self {
            m: [im00, im01, itx, im10, im11, ity],
        })
    }

    /// Apply this transform to a single point `(x, y)`.
    #[must_use]
    pub fn apply_point(&self, x: f32, y: f32) -> (f32, f32) {
        let [m00, m01, tx, m10, m11, ty] = self.m;
        (m00 * x + m01 * y + tx, m10 * x + m11 * y + ty)
    }
}

/// Apply an affine transform to an RGBA image using inverse mapping with
/// bilinear interpolation.
///
/// `src` must be `width × height × 4` bytes (RGBA).
/// `dst` must be pre-allocated to `width × height × 4` bytes (same dimensions
/// as source; pixels that map outside the source are filled with transparent
/// black).
///
/// # Errors
///
/// Returns `AccelError::InvalidDimensions` when buffer sizes are inconsistent,
/// or `AccelError::Unsupported` if the transform is not invertible.
pub fn apply_affine(
    src: &[u8],
    dst: &mut [u8],
    width: usize,
    height: usize,
    transform: &AffineTransform,
) -> AccelResult<()> {
    let expected = width * height * 4;
    if src.len() < expected {
        return Err(AccelError::InvalidDimensions(format!(
            "src buffer too small: {} < {expected}",
            src.len()
        )));
    }
    if dst.len() < expected {
        return Err(AccelError::InvalidDimensions(format!(
            "dst buffer too small: {} < {expected}",
            dst.len()
        )));
    }
    if width == 0 || height == 0 {
        return Err(AccelError::InvalidDimensions("zero dimension".to_string()));
    }

    // Inverse-mapping: for each dst pixel compute the src position.
    let inv = transform.try_inverse().ok_or_else(|| {
        AccelError::Unsupported("AffineTransform is not invertible (degenerate matrix)".to_string())
    })?;

    let w = width as f32;
    let h = height as f32;

    // Parallel over output rows.
    // We operate in pixel-integer space: pixel (x,y) occupies [x, x+1).
    // The bilinear sampler samples at the sub-pixel position within [0, w/h).
    dst.par_chunks_exact_mut(width * 4)
        .enumerate()
        .for_each(|(dy, row)| {
            let fy = dy as f32;
            for dx in 0..width {
                let fx = dx as f32;
                let (sx_f, sy_f) = inv.apply_point(fx, fy);

                let pixel = if sx_f < 0.0 || sy_f < 0.0 || sx_f >= w || sy_f >= h {
                    [0u8; 4]
                } else {
                    bilinear_sample(src, width, height, sx_f, sy_f)
                };

                let base = dx * 4;
                row[base] = pixel[0];
                row[base + 1] = pixel[1];
                row[base + 2] = pixel[2];
                row[base + 3] = pixel[3];
            }
        });

    Ok(())
}

/// Bilinear interpolation on a flat RGBA u8 buffer.
fn bilinear_sample(src: &[u8], width: usize, height: usize, x: f32, y: f32) -> [u8; 4] {
    let x0 = (x.floor() as usize).min(width - 1);
    let y0 = (y.floor() as usize).min(height - 1);
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);
    let fx = x.fract();
    let fy = y.fract();

    let fetch = |row: usize, col: usize| -> [f32; 4] {
        let base = (row * width + col) * 4;
        [
            src[base] as f32,
            src[base + 1] as f32,
            src[base + 2] as f32,
            src[base + 3] as f32,
        ]
    };

    let c00 = fetch(y0, x0);
    let c10 = fetch(y0, x1);
    let c01 = fetch(y1, x0);
    let c11 = fetch(y1, x1);

    let lerp = |a: f32, b: f32, t: f32| a + t * (b - a);
    let mut out = [0u8; 4];
    for i in 0..4 {
        let top = lerp(c00[i], c10[i], fx);
        let bot = lerp(c01[i], c11[i], fx);
        out[i] = lerp(top, bot, fy).round().clamp(0.0, 255.0) as u8;
    }
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn solid_rgba(r: u8, g: u8, b: u8, a: u8, n: usize) -> Vec<u8> {
        (0..n).flat_map(|_| [r, g, b, a]).collect()
    }

    #[test]
    fn test_identity_preserves_image() {
        let src = solid_rgba(100, 150, 200, 255, 4 * 4);
        let mut dst = vec![0u8; 4 * 4 * 4];
        apply_affine(&src, &mut dst, 4, 4, &AffineTransform::identity()).unwrap();
        // Every pixel should be identical to the source
        assert_eq!(src, dst);
    }

    #[test]
    fn test_180_degree_rotation_round_trip() {
        // Build a distinguishable 8×8 image (larger to keep interior pixels
        // away from edges where bilinear sampling meets out-of-bounds pixels).
        let w = 8usize;
        let h = 8usize;
        let src: Vec<u8> = (0..(w * h))
            .flat_map(|i| {
                let v = ((i * 13) % 200 + 30) as u8;
                [v, (v / 2), 0u8, 255u8]
            })
            .collect();

        // Rotate 180° around the image centre.
        // Centre of image (pixel-coordinate system): pixel centres at 0.5..7.5
        let cx = w as f32 / 2.0;
        let cy = h as f32 / 2.0;
        let to_origin = AffineTransform::translation(-cx, -cy);
        let rot180 = AffineTransform::rotation(PI);
        let back = AffineTransform::translation(cx, cy);
        let transform = to_origin.compose(&rot180).compose(&back);

        let mut once = vec![0u8; w * h * 4];
        apply_affine(&src, &mut once, w, h, &transform).unwrap();

        let mut twice = vec![0u8; w * h * 4];
        apply_affine(&once, &mut twice, w, h, &transform).unwrap();

        // After two 180° rotations the interior pixels (skip border row/col)
        // should round-trip with small error.  Border pixels can be 0 due to
        // clamping on the first pass and 0 → bilinear edge artefact on second pass.
        let mut failures = 0usize;
        for row in 1..h - 1 {
            for col in 1..w - 1 {
                let i = (row * w + col) * 4;
                for c in 0..3 {
                    let diff = (src[i + c] as i32 - twice[i + c] as i32).unsigned_abs();
                    if diff > 8 {
                        failures += 1;
                    }
                }
            }
        }
        assert!(
            failures == 0,
            "{failures} interior pixels failed round-trip tolerance (≤8 allowed)"
        );
    }

    #[test]
    fn test_scale_2x_uniform_colour_preserved() {
        // Test scale transform: a uniform 4×4 image scaled up 2× (to 8×8).
        // We apply a 0.5× scale transform (forward), which maps dst→src as
        // src_pt = 0.5 * dst_pt.  Every dst pixel maps into the interior of
        // the 4×4 src and should read the uniform colour.
        let src = solid_rgba(128, 64, 32, 255, 4 * 4);
        let mut dst = vec![0u8; 8 * 8 * 4];
        // scale(0.5, 0.5): forward transform compresses. The inverse maps
        // each dst pixel → 0.5 * dst_coord in src space.
        let t = AffineTransform::scale(0.5, 0.5);
        apply_affine_sized(&src, 4, 4, &mut dst, 8, 8, &t).unwrap();
        // Interior pixels (away from the out-of-bounds edge) should match.
        // scale(0.5) inverse is scale(2): dst(4,4) → src(8,8) which is OOB.
        // So the valid region is dst coords where 2*dst_coord < 4, i.e. dst < 2.
        // Check only pixels in the valid dst region (first 2 rows/cols).
        for row in 0..2usize {
            for col in 0..2usize {
                let base = (row * 8 + col) * 4;
                assert!(
                    (dst[base] as i32 - 128).unsigned_abs() <= 2,
                    "R off at ({row},{col})"
                );
                assert!(
                    (dst[base + 1] as i32 - 64).unsigned_abs() <= 2,
                    "G off at ({row},{col})"
                );
                assert!(
                    (dst[base + 2] as i32 - 32).unsigned_abs() <= 2,
                    "B off at ({row},{col})"
                );
            }
        }
    }

    #[test]
    fn test_scale_half_uniform_colour_preserved() {
        // scale(2, 2) forward: src_pt = 2 * dst_coord → downscale 4→2.
        // The inverse is scale(0.5): maps dst(0,0) → src(0,0), dst(1,0) → src(0.5,0), etc.
        // All source coords are within [0,4) for dst [0,2).
        let src = solid_rgba(200, 100, 50, 255, 4 * 4);
        let mut dst = vec![0u8; 2 * 2 * 4];
        let t = AffineTransform::scale(2.0, 2.0);
        apply_affine_sized(&src, 4, 4, &mut dst, 2, 2, &t).unwrap();
        for (i, px) in dst.chunks_exact(4).enumerate() {
            assert!((px[0] as i32 - 200).unsigned_abs() <= 2, "R off pixel {i}");
            assert!((px[1] as i32 - 100).unsigned_abs() <= 2, "G off pixel {i}");
            assert!((px[2] as i32 - 50).unsigned_abs() <= 2, "B off pixel {i}");
        }
    }

    #[test]
    fn test_translation_shifts_image() {
        // 4×4 black image with a white top-left pixel
        let mut src = vec![0u8; 4 * 4 * 4];
        src[0] = 255;
        src[1] = 255;
        src[2] = 255;
        src[3] = 255;

        let mut dst = vec![0u8; 4 * 4 * 4];
        // Translate right by 1, down by 1
        let t = AffineTransform::translation(1.0, 1.0);
        apply_affine(&src, &mut dst, 4, 4, &t).unwrap();

        // The white pixel at src(0,0) should appear at dst(1,1)
        let base = (1 * 4 + 1) * 4;
        assert!(
            dst[base] > 200,
            "dst(1,1).R should be near 255, got {}",
            dst[base]
        );
    }

    #[test]
    fn test_zero_dimensions_returns_error() {
        let src = vec![0u8; 4];
        let mut dst = vec![0u8; 4];
        let result = apply_affine(&src, &mut dst, 0, 4, &AffineTransform::identity());
        assert!(result.is_err());
    }

    #[test]
    fn test_invalid_src_buffer_returns_error() {
        let src = vec![0u8; 10]; // too small for 4×4
        let mut dst = vec![0u8; 4 * 4 * 4];
        let result = apply_affine(&src, &mut dst, 4, 4, &AffineTransform::identity());
        assert!(result.is_err());
    }

    #[test]
    fn test_degenerate_transform_returns_error() {
        let src = solid_rgba(0, 0, 0, 255, 4 * 4);
        let mut dst = vec![0u8; 4 * 4 * 4];
        let degenerate = AffineTransform {
            m: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        };
        let result = apply_affine(&src, &mut dst, 4, 4, &degenerate);
        assert!(result.is_err(), "degenerate transform should return error");
    }

    #[test]
    fn test_compose_identity_is_identity() {
        let t = AffineTransform::rotation(0.5);
        let composed = t.compose(&AffineTransform::identity());
        for (a, b) in t.m.iter().zip(composed.m.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "compose with identity should be stable"
            );
        }
    }

    #[test]
    fn test_rotation_apply_point() {
        // 90° CCW: (1,0) → (0,1)
        let r = AffineTransform::rotation(PI / 2.0);
        let (x, y) = r.apply_point(1.0, 0.0);
        assert!((x - 0.0).abs() < 1e-5, "x={x}");
        assert!((y - 1.0).abs() < 1e-5, "y={y}");
    }

    #[test]
    fn test_inverse_of_rotation() {
        let r = AffineTransform::rotation(1.2);
        let inv = r.try_inverse().expect("rotation must be invertible");
        let composed = r.compose(&inv);
        // Result should be close to identity
        let id = AffineTransform::identity();
        for (a, b) in composed.m.iter().zip(id.m.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "r * r^-1 should be identity; got {a} vs {b}"
            );
        }
    }

    // ── Extra helper: sized variant for upscale test ─────────────────────────

    /// Like `apply_affine` but with separate src and dst dimensions.
    fn apply_affine_sized(
        src: &[u8],
        src_w: usize,
        src_h: usize,
        dst: &mut [u8],
        dst_w: usize,
        dst_h: usize,
        transform: &AffineTransform,
    ) -> AccelResult<()> {
        let expected_src = src_w * src_h * 4;
        let expected_dst = dst_w * dst_h * 4;
        if src.len() < expected_src || dst.len() < expected_dst {
            return Err(AccelError::InvalidDimensions(
                "buffer too small".to_string(),
            ));
        }
        if dst_w == 0 || dst_h == 0 || src_w == 0 || src_h == 0 {
            return Err(AccelError::InvalidDimensions("zero dimension".to_string()));
        }

        let inv = transform
            .try_inverse()
            .ok_or_else(|| AccelError::Unsupported("degenerate matrix".to_string()))?;

        let sw = src_w as f32;
        let sh = src_h as f32;

        dst.par_chunks_exact_mut(dst_w * 4)
            .enumerate()
            .for_each(|(dy, row)| {
                let fy = dy as f32;
                for dx in 0..dst_w {
                    let fx = dx as f32;
                    let (sx_f, sy_f) = inv.apply_point(fx, fy);

                    let pixel = if sx_f < 0.0 || sy_f < 0.0 || sx_f >= sw || sy_f >= sh {
                        [0u8; 4]
                    } else {
                        bilinear_sample(src, src_w, src_h, sx_f, sy_f)
                    };

                    let base = dx * 4;
                    row[base] = pixel[0];
                    row[base + 1] = pixel[1];
                    row[base + 2] = pixel[2];
                    row[base + 3] = pixel[3];
                }
            });

        Ok(())
    }
}
