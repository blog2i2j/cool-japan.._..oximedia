//! Laplacian pyramid blending for seamless panorama stitching.
//!
//! Multi-resolution (Laplacian pyramid) blending decomposes images into a
//! hierarchy of frequency bands and blends each band independently.  This
//! avoids the low-frequency colour seams that straight alpha blending produces
//! while still producing smooth transitions at high-frequency detail.
//!
//! ## Algorithm overview
//!
//! 1. Build a **Gaussian pyramid** by repeatedly blurring and downsampling by 2×.
//! 2. Construct a **Laplacian pyramid** where each level stores the high-frequency
//!    residual `G[i] − upsample(G[i+1])`.
//! 3. **Blend** corresponding Laplacian levels using a soft mask (also
//!    downsampled to match each pyramid level).
//! 4. **Reconstruct** the final image by iteratively upsampling and summing the
//!    blended Laplacian levels from coarsest to finest.
//!
//! All internal operations work on single-channel `f32` images.  For multi-
//! channel images, call these functions once per channel.

// ─── Gaussian pyramid ─────────────────────────────────────────────────────────

/// A Gaussian (blurred-downsampled) image pyramid.
///
/// Each level is a `(pixels, width, height)` tuple.  Level 0 is the original
/// image (or a slight Gaussian blur thereof); each subsequent level has
/// approximately half the width and height of the previous one.
pub struct GaussianPyramid {
    /// Levels from finest (index 0) to coarsest (index n-1).
    pub levels: Vec<(Vec<f32>, u32, u32)>,
}

impl GaussianPyramid {
    /// Build a Gaussian pyramid from a single-channel `f32` image.
    ///
    /// * `image`   — row-major, single-channel pixel values
    /// * `width`   — image width in pixels
    /// * `height`  — image height in pixels
    /// * `levels`  — total number of pyramid levels (including the original);
    ///   must be ≥ 1.  The actual number of levels produced may be smaller if
    ///   the image becomes too small to downsample further.
    ///
    /// Each level is produced by a 5-tap Gaussian blur (σ ≈ 1) followed by
    /// 2× sub-sampling, matching the classic Burt & Adelson formulation.
    pub fn build(image: &[f32], width: u32, height: u32, levels: usize) -> Self {
        let levels = levels.max(1);
        let mut pyramid: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(levels);

        // Level 0: store the original (no extra blur at this stage)
        pyramid.push((image.to_vec(), width, height));

        for _ in 1..levels {
            let (prev, pw, ph) = pyramid.last().expect("non-empty pyramid");
            if *pw <= 1 || *ph <= 1 {
                // Cannot downsample further
                break;
            }
            let blurred = gaussian_blur_5tap(prev, *pw, *ph);
            let (down, dw, dh) = downsample_half_f32(&blurred, *pw, *ph);
            pyramid.push((down, dw, dh));
        }

        GaussianPyramid { levels: pyramid }
    }
}

// ─── Laplacian pyramid ────────────────────────────────────────────────────────

/// A Laplacian (band-pass residual) image pyramid.
///
/// Each level stores the difference `G[i] − upsample(G[i+1])`, except the
/// coarsest level which is stored directly (no residual).  Reconstruction
/// sums all levels after upsampling each coarser level.
pub struct LaplacianPyramid {
    /// Levels from finest (index 0) to coarsest (index n-1).
    pub levels: Vec<(Vec<f32>, u32, u32)>,
}

impl LaplacianPyramid {
    /// Construct a Laplacian pyramid from a pre-built [`GaussianPyramid`].
    ///
    /// `L[i] = G[i] − upsample(G[i+1])`  for i < n-1.
    /// `L[n-1] = G[n-1]`  (coarsest residual stored as-is).
    pub fn from_gaussian(gaussian: &GaussianPyramid) -> Self {
        let n = gaussian.levels.len();
        let mut laplacian: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(n);

        for i in 0..n {
            let wi = gaussian.levels[i].1;
            let hi = gaussian.levels[i].2;
            let g_i = &gaussian.levels[i].0;
            if i + 1 < n {
                let wn = gaussian.levels[i + 1].1;
                let hn = gaussian.levels[i + 1].2;
                let g_next = &gaussian.levels[i + 1].0;
                let upsampled = upsample_double_f32(g_next, wn, hn, wi, hi);
                let lap: Vec<f32> = g_i
                    .iter()
                    .zip(upsampled.iter())
                    .map(|(&a, &b)| a - b)
                    .collect();
                laplacian.push((lap, wi, hi));
            } else {
                // Coarsest level: keep as-is
                laplacian.push((g_i.clone(), wi, hi));
            }
        }

        LaplacianPyramid { levels: laplacian }
    }
}

// ─── Blending ─────────────────────────────────────────────────────────────────

/// Blend two Laplacian pyramids level-by-level using a soft mask.
///
/// At each level, the output is `l1 * mask + l2 * (1 − mask)`, where `mask`
/// is also downsampled to the same resolution as each level.
///
/// Both input pyramids must have the same number of levels and matching
/// dimensions at every level.  The `mask` must correspond to level 0 (full
/// resolution) with values in `[0, 1]`; it is automatically downsampled for
/// deeper levels.
///
/// Returns a new `LaplacianPyramid` of the same structure.
pub fn blend_laplacian(
    l1: &LaplacianPyramid,
    l2: &LaplacianPyramid,
    mask: &[f32],
) -> LaplacianPyramid {
    let n = l1.levels.len().min(l2.levels.len());
    let mut result: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(n);

    // Build a Gaussian pyramid for the mask so we have one per level
    let w0 = l1.levels[0].1;
    let h0 = l1.levels[0].2;
    let mask_pyramid = GaussianPyramid::build(mask, w0, h0, n);

    for i in 0..n {
        let (ref lap1, lw, lh) = l1.levels[i];
        let (ref lap2, _, _) = l2.levels[i];
        let (ref m, mw, mh) = mask_pyramid.levels[i];

        // If mask pyramid dimensions don't exactly match lap dimensions, resample mask
        let m_scaled = if mw == lw && mh == lh {
            m.clone()
        } else {
            // Bilinear resample mask to (lw, lh) from (mw, mh)
            resample_f32_bilinear(m, mw, mh, lw, lh)
        };

        let n_pixels = (lw as usize) * (lh as usize);
        let mut blended = Vec::with_capacity(n_pixels);

        for j in 0..n_pixels {
            let alpha = m_scaled.get(j).copied().unwrap_or(0.5).clamp(0.0, 1.0);
            let v1 = lap1.get(j).copied().unwrap_or(0.0);
            let v2 = lap2.get(j).copied().unwrap_or(0.0);
            blended.push(v1 * alpha + v2 * (1.0 - alpha));
        }

        result.push((blended, lw, lh));
    }

    LaplacianPyramid { levels: result }
}

/// Reconstruct a full-resolution image from a Laplacian pyramid.
///
/// Starting from the coarsest level, iteratively upsamples and adds each
/// finer level's residual until the finest level is reached.
///
/// Returns the reconstructed single-channel `f32` pixel buffer at the
/// resolution of `lap.levels[0]`.
pub fn reconstruct(lap: &LaplacianPyramid) -> Vec<f32> {
    let n = lap.levels.len();
    if n == 0 {
        return Vec::new();
    }

    // Start with coarsest level
    let mut cw = lap.levels[n - 1].1;
    let mut ch = lap.levels[n - 1].2;
    let mut current = lap.levels[n - 1].0.clone();

    // Collapse from coarsest to finest
    for i in (0..n - 1).rev() {
        let tw = lap.levels[i].1;
        let th = lap.levels[i].2;
        let residual = &lap.levels[i].0;
        // Upsample current to the dimensions of the next finer level
        let upsampled = upsample_double_f32(&current, cw, ch, tw, th);
        // Add the residual
        current = upsampled
            .iter()
            .zip(residual.iter())
            .map(|(&u, &r)| u + r)
            .collect();
        cw = tw;
        ch = th;
    }

    current
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

/// Separable 5-tap Gaussian blur kernel coefficients (σ ≈ 1).
/// Burt & Adelson (1983) "a" = 0.375 row: [0.0625, 0.25, 0.375, 0.25, 0.0625]
const GAUSS5: [f32; 5] = [0.0625, 0.25, 0.375, 0.25, 0.0625];

/// Apply a separable 5-tap Gaussian blur to a single-channel `f32` image.
fn gaussian_blur_5tap(src: &[f32], w: u32, h: u32) -> Vec<f32> {
    let ww = w as usize;
    let hh = h as usize;
    let mut temp = vec![0.0f32; ww * hh];
    let mut out = vec![0.0f32; ww * hh];

    // Horizontal pass
    for row in 0..hh {
        for col in 0..ww {
            let mut acc = 0.0f32;
            for (ki, &k) in GAUSS5.iter().enumerate() {
                let x = (col as i64 + ki as i64 - 2).clamp(0, ww as i64 - 1) as usize;
                acc += src[row * ww + x] * k;
            }
            temp[row * ww + col] = acc;
        }
    }

    // Vertical pass
    for row in 0..hh {
        for col in 0..ww {
            let mut acc = 0.0f32;
            for (ki, &k) in GAUSS5.iter().enumerate() {
                let y = (row as i64 + ki as i64 - 2).clamp(0, hh as i64 - 1) as usize;
                acc += temp[y * ww + col] * k;
            }
            out[row * ww + col] = acc;
        }
    }

    out
}

/// Downsample a single-channel `f32` image to half the size using a 2×2 box average.
fn downsample_half_f32(src: &[f32], w: u32, h: u32) -> (Vec<f32>, u32, u32) {
    let ow = (w / 2).max(1);
    let oh = (h / 2).max(1);
    let ww = w as usize;
    let oww = ow as usize;
    let mut out = vec![0.0f32; oww * oh as usize];

    for oy in 0..oh as usize {
        for ox in 0..oww {
            let sx = ox * 2;
            let sy = oy * 2;
            let sx1 = (sx + 1).min(ww - 1);
            let sy1 = (sy + 1).min(h as usize - 1);
            let v =
                src[sy * ww + sx] + src[sy * ww + sx1] + src[sy1 * ww + sx] + src[sy1 * ww + sx1];
            out[oy * oww + ox] = v * 0.25;
        }
    }

    (out, ow, oh)
}

/// Upsample a single-channel `f32` image to `(tw, th)` using bilinear interpolation.
fn upsample_double_f32(src: &[f32], sw: u32, sh: u32, tw: u32, th: u32) -> Vec<f32> {
    let sww = sw as usize;
    let shh = sh as usize;
    let tww = tw as usize;
    let thh = th as usize;
    let mut out = vec![0.0f32; tww * thh];

    for oy in 0..thh {
        for ox in 0..tww {
            // Map to source coordinate space (centre-aligned)
            let sx_f = (ox as f32 + 0.5) * sww as f32 / tww as f32 - 0.5;
            let sy_f = (oy as f32 + 0.5) * shh as f32 / thh as f32 - 0.5;

            let sx0 = (sx_f.floor() as i64).clamp(0, sww as i64 - 1) as usize;
            let sy0 = (sy_f.floor() as i64).clamp(0, shh as i64 - 1) as usize;
            let sx1 = (sx0 + 1).min(sww - 1);
            let sy1 = (sy0 + 1).min(shh - 1);

            let tx = sx_f - sx_f.floor();
            let ty = sy_f - sy_f.floor();

            let p00 = src[sy0 * sww + sx0];
            let p10 = src[sy0 * sww + sx1];
            let p01 = src[sy1 * sww + sx0];
            let p11 = src[sy1 * sww + sx1];

            let top = p00 + (p10 - p00) * tx;
            let bot = p01 + (p11 - p01) * tx;
            out[oy * tww + ox] = top + (bot - top) * ty;
        }
    }

    out
}

/// Bilinear resample a `f32` image from `(sw, sh)` to `(tw, th)`.
fn resample_f32_bilinear(src: &[f32], sw: u32, sh: u32, tw: u32, th: u32) -> Vec<f32> {
    upsample_double_f32(src, sw, sh, tw, th)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_image(w: u32, h: u32, value: f32) -> Vec<f32> {
        vec![value; (w * h) as usize]
    }

    // ── GaussianPyramid ──────────────────────────────────────────────────────

    #[test]
    fn gaussian_pyramid_level_count() {
        let img = solid_image(64, 64, 0.5);
        let pyr = GaussianPyramid::build(&img, 64, 64, 4);
        assert_eq!(pyr.levels.len(), 4);
    }

    #[test]
    fn gaussian_pyramid_dimensions_halve() {
        let img = solid_image(64, 64, 0.5);
        let pyr = GaussianPyramid::build(&img, 64, 64, 4);
        let (_, w0, h0) = &pyr.levels[0];
        let (_, w1, h1) = &pyr.levels[1];
        let (_, w2, h2) = &pyr.levels[2];
        assert_eq!(*w0, 64);
        assert_eq!(*h0, 64);
        assert_eq!(*w1, 32);
        assert_eq!(*h1, 32);
        assert_eq!(*w2, 16);
        assert_eq!(*h2, 16);
    }

    #[test]
    fn gaussian_pyramid_solid_colour_preserved() {
        // A solid image should produce approximately the same value at every level
        let img = solid_image(32, 32, 0.6);
        let pyr = GaussianPyramid::build(&img, 32, 32, 3);
        for (level_img, _, _) in &pyr.levels {
            for &v in level_img {
                assert!((v - 0.6).abs() < 0.02, "expected ~0.6, got {v}");
            }
        }
    }

    #[test]
    fn gaussian_pyramid_single_level() {
        let img = vec![1.0f32, 2.0, 3.0, 4.0];
        let pyr = GaussianPyramid::build(&img, 2, 2, 1);
        assert_eq!(pyr.levels.len(), 1);
        assert_eq!(pyr.levels[0].0, img);
    }

    // ── LaplacianPyramid ─────────────────────────────────────────────────────

    #[test]
    fn laplacian_pyramid_same_level_count() {
        let img = solid_image(64, 64, 0.5);
        let gauss = GaussianPyramid::build(&img, 64, 64, 4);
        let lap = LaplacianPyramid::from_gaussian(&gauss);
        assert_eq!(lap.levels.len(), gauss.levels.len());
    }

    #[test]
    fn laplacian_pyramid_coarsest_equals_gaussian_coarsest() {
        let img = solid_image(32, 32, 0.8);
        let gauss = GaussianPyramid::build(&img, 32, 32, 3);
        let lap = LaplacianPyramid::from_gaussian(&gauss);
        let n = gauss.levels.len();
        let (ref g_last, _, _) = gauss.levels[n - 1];
        let (ref l_last, _, _) = lap.levels[n - 1];
        assert_eq!(
            g_last, l_last,
            "coarsest Laplacian level should equal coarsest Gaussian"
        );
    }

    #[test]
    fn laplacian_pyramid_residuals_near_zero_for_solid_image() {
        // For a solid colour image, all Laplacian residuals (except coarsest) should
        // be very close to zero because blurring doesn't change a constant image.
        let img = solid_image(32, 32, 0.5);
        let gauss = GaussianPyramid::build(&img, 32, 32, 4);
        let lap = LaplacianPyramid::from_gaussian(&gauss);
        // Check all levels except the last (coarsest) for near-zero residuals
        let n = lap.levels.len();
        for i in 0..n.saturating_sub(1) {
            let (ref l, _, _) = lap.levels[i];
            let max_abs: f32 = l.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(max_abs < 0.02, "level {i} residual too large: {max_abs}");
        }
    }

    // ── blend_laplacian ──────────────────────────────────────────────────────

    #[test]
    fn blend_laplacian_full_mask_returns_l1() {
        // With mask = all 1.0, the result should equal l1
        let img1 = solid_image(32, 32, 0.8);
        let img2 = solid_image(32, 32, 0.2);
        let mask = solid_image(32, 32, 1.0);

        let g1 = GaussianPyramid::build(&img1, 32, 32, 3);
        let g2 = GaussianPyramid::build(&img2, 32, 32, 3);
        let l1 = LaplacianPyramid::from_gaussian(&g1);
        let l2 = LaplacianPyramid::from_gaussian(&g2);

        let blended = blend_laplacian(&l1, &l2, &mask);

        for i in 0..blended.levels.len() {
            let (ref b, _, _) = blended.levels[i];
            let (ref l, _, _) = l1.levels[i];
            for (bv, lv) in b.iter().zip(l.iter()) {
                assert!((bv - lv).abs() < 1e-5, "level {i}: blended={bv}, l1={lv}");
            }
        }
    }

    #[test]
    fn blend_laplacian_zero_mask_returns_l2() {
        let img1 = solid_image(32, 32, 0.9);
        let img2 = solid_image(32, 32, 0.1);
        let mask = solid_image(32, 32, 0.0);

        let g1 = GaussianPyramid::build(&img1, 32, 32, 3);
        let g2 = GaussianPyramid::build(&img2, 32, 32, 3);
        let l1 = LaplacianPyramid::from_gaussian(&g1);
        let l2 = LaplacianPyramid::from_gaussian(&g2);

        let blended = blend_laplacian(&l1, &l2, &mask);

        for i in 0..blended.levels.len() {
            let (ref b, _, _) = blended.levels[i];
            let (ref l, _, _) = l2.levels[i];
            for (bv, lv) in b.iter().zip(l.iter()) {
                assert!((bv - lv).abs() < 1e-5, "level {i}: blended={bv}, l2={lv}");
            }
        }
    }

    #[test]
    fn blend_laplacian_half_mask_interpolates() {
        let img1 = solid_image(32, 32, 1.0);
        let img2 = solid_image(32, 32, 0.0);
        let mask = solid_image(32, 32, 0.5);

        let g1 = GaussianPyramid::build(&img1, 32, 32, 3);
        let g2 = GaussianPyramid::build(&img2, 32, 32, 3);
        let l1 = LaplacianPyramid::from_gaussian(&g1);
        let l2 = LaplacianPyramid::from_gaussian(&g2);

        let blended = blend_laplacian(&l1, &l2, &mask);
        let recon = reconstruct(&blended);

        // Reconstructed result should be approximately 0.5
        let mean: f32 = recon.iter().sum::<f32>() / recon.len() as f32;
        assert!((mean - 0.5).abs() < 0.05, "expected ~0.5, got {mean}");
    }

    // ── reconstruct ─────────────────────────────────────────────────────────

    #[test]
    fn reconstruct_roundtrip_solid_colour() {
        // Build Laplacian pyramid from a solid image, then reconstruct — should
        // recover the original value at every pixel.
        let val = 0.75f32;
        let img = solid_image(64, 64, val);
        let gauss = GaussianPyramid::build(&img, 64, 64, 4);
        let lap = LaplacianPyramid::from_gaussian(&gauss);
        let recon = reconstruct(&lap);
        assert_eq!(recon.len(), 64 * 64);
        for &v in &recon {
            assert!((v - val).abs() < 0.02, "expected {val}, got {v}");
        }
    }

    #[test]
    fn reconstruct_empty_pyramid() {
        let empty = LaplacianPyramid { levels: vec![] };
        let result = reconstruct(&empty);
        assert!(result.is_empty());
    }

    #[test]
    fn reconstruct_output_dimensions_match_finest_level() {
        let img = solid_image(48, 32, 0.3);
        let gauss = GaussianPyramid::build(&img, 48, 32, 3);
        let lap = LaplacianPyramid::from_gaussian(&gauss);
        let recon = reconstruct(&lap);
        let (_, w0, h0) = &lap.levels[0];
        assert_eq!(recon.len(), (*w0 as usize) * (*h0 as usize));
    }

    // ── Full pipeline ────────────────────────────────────────────────────────

    #[test]
    fn full_pipeline_blends_two_images() {
        // Left half: bright (0.9), right half: dark (0.1)
        // Mask: 1.0 on left, 0.0 on right → result should have brightness gradient
        let w = 64u32;
        let h = 32u32;
        let n = (w * h) as usize;

        let img1: Vec<f32> = vec![0.9; n];
        let img2: Vec<f32> = vec![0.1; n];

        // Soft horizontal mask
        let mask: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i % w as usize) as f32 / w as f32;
                if x < 0.5 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();

        let g1 = GaussianPyramid::build(&img1, w, h, 4);
        let g2 = GaussianPyramid::build(&img2, w, h, 4);
        let l1 = LaplacianPyramid::from_gaussian(&g1);
        let l2 = LaplacianPyramid::from_gaussian(&g2);
        let blended = blend_laplacian(&l1, &l2, &mask);
        let result = reconstruct(&blended);

        // Left-quarter mean should be significantly brighter than right-quarter
        let left_mean: f32 = (0..h as usize)
            .flat_map(|row| (0..w as usize / 4).map(move |col| (row, col)))
            .map(|(row, col)| result[row * w as usize + col])
            .sum::<f32>()
            / (h as usize * w as usize / 4) as f32;

        let right_mean: f32 = (0..h as usize)
            .flat_map(|row| (3 * w as usize / 4..w as usize).map(move |col| (row, col)))
            .map(|(row, col)| result[row * w as usize + col])
            .sum::<f32>()
            / (h as usize * w as usize / 4) as f32;

        assert!(
            left_mean > right_mean + 0.3,
            "left_mean={left_mean:.3} should be much brighter than right_mean={right_mean:.3}"
        );
    }
}
