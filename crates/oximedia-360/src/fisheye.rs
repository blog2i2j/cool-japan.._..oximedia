//! Fisheye lens projection models and dual-fisheye stitching.
//!
//! Supports equidistant, equisolid, orthographic and stereographic fisheye
//! projections, plus conversion to/from equirectangular and a dual-fisheye
//! stitcher with linear-alpha blending in the overlap zone.

use crate::{
    projection::{
        bilinear_sample_u8, equirect_to_sphere, sphere_to_equirect, SphericalCoord, UvCoord,
    },
    VrError,
};

// ─── Fisheye model ───────────────────────────────────────────────────────────

/// Mathematical model describing a fisheye lens' projection law.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FisheyeModel {
    /// r = f × θ  (most common, also known as "f-theta")
    Equidistant,
    /// r = 2f × sin(θ/2)
    Equisolid,
    /// r = f × sin(θ)
    Orthographic,
    /// r = 2f × tan(θ/2)
    Stereographic,
}

// ─── Fisheye parameters ──────────────────────────────────────────────────────

/// Parameters describing a fisheye lens and how it maps to a sensor.
#[derive(Debug, Clone, PartialEq)]
pub struct FisheyeParams {
    /// Full field-of-view in degrees.
    pub fov_deg: f32,
    /// Optical centre X as a fraction of image width (0..1).
    pub center_x: f32,
    /// Optical centre Y as a fraction of image height (0..1).
    pub center_y: f32,
    /// Radius of the fisheye circle as a fraction of image half-min-dimension.
    /// A value of 1.0 means the circle just touches the nearest image edge.
    pub radius: f32,
    /// Lens projection model.
    pub model: FisheyeModel,
}

impl FisheyeParams {
    /// Convenience constructor with equidistant model.
    pub fn equidistant(fov_deg: f32) -> Self {
        Self {
            fov_deg,
            center_x: 0.5,
            center_y: 0.5,
            radius: 1.0,
            model: FisheyeModel::Equidistant,
        }
    }

    /// Half FOV in radians.
    fn half_fov_rad(&self) -> f32 {
        self.fov_deg * 0.5 * std::f32::consts::PI / 180.0
    }

    /// Compute the normalised radial distance `r` (in the fisheye image, as a
    /// fraction of `radius`) for a given incident angle `theta` (radians from
    /// the optical axis).
    ///
    /// Returns `None` if `theta` exceeds the half-FOV.
    pub fn theta_to_r(&self, theta: f32) -> Option<f32> {
        let half_fov = self.half_fov_rad();
        if theta > half_fov {
            return None;
        }
        // f is chosen so that r == 1.0 at theta == half_fov
        let r = match self.model {
            FisheyeModel::Equidistant => {
                // r = f * theta,  f = 1.0 / half_fov
                theta / half_fov
            }
            FisheyeModel::Equisolid => {
                // r = 2f * sin(θ/2),  f = 1 / (2 * sin(half_fov/2))
                let denom = 2.0 * (half_fov * 0.5).sin();
                if denom.abs() < f32::EPSILON {
                    0.0
                } else {
                    2.0 * (theta * 0.5).sin() / denom
                }
            }
            FisheyeModel::Orthographic => {
                // r = f * sin(θ),  f = 1 / sin(half_fov)
                let denom = half_fov.sin();
                if denom.abs() < f32::EPSILON {
                    0.0
                } else {
                    theta.sin() / denom
                }
            }
            FisheyeModel::Stereographic => {
                // r = 2f * tan(θ/2),  f = 1 / (2 * tan(half_fov/2))
                let denom = 2.0 * (half_fov * 0.5).tan();
                if denom.abs() < f32::EPSILON {
                    0.0
                } else {
                    2.0 * (theta * 0.5).tan() / denom
                }
            }
        };
        Some(r.clamp(0.0, 1.0))
    }

    /// Invert `theta_to_r`: given a normalised radial distance, compute the
    /// incident angle `theta` in radians.
    fn r_to_theta(&self, r: f32) -> f32 {
        let r = r.clamp(0.0, 1.0);
        let half_fov = self.half_fov_rad();
        match self.model {
            FisheyeModel::Equidistant => r * half_fov,
            FisheyeModel::Equisolid => {
                let denom = 2.0 * (half_fov * 0.5).sin();
                // r = 2f*sin(θ/2)  →  sin(θ/2) = r*denom/2  →  θ = 2*asin(...)
                let sin_half = r * denom * 0.5;
                2.0 * sin_half.clamp(-1.0, 1.0).asin()
            }
            FisheyeModel::Orthographic => {
                let denom = half_fov.sin();
                // r = f*sin(θ)  →  sin(θ) = r*denom
                (r * denom).clamp(-1.0, 1.0).asin()
            }
            FisheyeModel::Stereographic => {
                let denom = 2.0 * (half_fov * 0.5).tan();
                // r = 2f*tan(θ/2)  →  tan(θ/2) = r*denom/2  →  θ = 2*atan(...)
                2.0 * (r * denom * 0.5).atan()
            }
        }
    }
}

// ─── Fisheye → equirectangular ───────────────────────────────────────────────

/// Convert a fisheye image to equirectangular.
///
/// * `src`        — source pixel data (RGB, 3 bpp, row-major)
/// * `src_width`  — source image width in pixels
/// * `src_height` — source image height in pixels
/// * `params`     — fisheye lens parameters
/// * `out_width`  — output equirectangular width in pixels
/// * `out_height` — output equirectangular height in pixels
///
/// Pixels outside the fisheye circle are black.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero.
pub fn fisheye_to_equirect(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    params: &FisheyeParams,
    out_width: u32,
    out_height: u32,
) -> Result<Vec<u8>, VrError> {
    validate_fisheye(src, src_width, src_height, out_width, out_height)?;

    const CH: u32 = 3;
    let mut out = vec![0u8; (out_width * out_height * CH) as usize];

    // The physical radius of the fisheye circle in source pixels
    let half_min_dim = (src_width.min(src_height) as f32) * 0.5;
    let circle_r_px = params.radius * half_min_dim;

    for oy in 0..out_height {
        for ox in 0..out_width {
            let u = (ox as f32 + 0.5) / out_width as f32;
            let v = (oy as f32 + 0.5) / out_height as f32;

            let sphere = equirect_to_sphere(&UvCoord { u, v });
            // theta = angle from the optical axis (forward = +Z)
            let theta = std::f32::consts::FRAC_PI_2 - sphere.elevation_rad;
            let phi = sphere.azimuth_rad; // longitude

            if let Some(r_norm) = params.theta_to_r(theta) {
                let r_px = r_norm * circle_r_px;
                let cx_px = params.center_x * src_width as f32;
                let cy_px = params.center_y * src_height as f32;

                let fx = cx_px + r_px * phi.sin();
                let fy = cy_px - r_px * phi.cos();

                let fu = fx / src_width as f32;
                let fv = fy / src_height as f32;

                if fu >= 0.0 && fu <= 1.0 && fv >= 0.0 && fv <= 1.0 {
                    let sample = bilinear_sample_u8(src, src_width, src_height, fu, fv, CH);
                    let dst = (oy * out_width + ox) as usize * CH as usize;
                    out[dst..dst + CH as usize].copy_from_slice(&sample);
                }
            }
        }
    }

    Ok(out)
}

/// Convert an equirectangular image to fisheye projection.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero.
pub fn equirect_to_fisheye(
    src: &[u8],
    src_width: u32,
    src_height: u32,
    params: &FisheyeParams,
    out_width: u32,
    out_height: u32,
) -> Result<Vec<u8>, VrError> {
    validate_fisheye(src, src_width, src_height, out_width, out_height)?;

    const CH: u32 = 3;
    let mut out = vec![0u8; (out_width * out_height * CH) as usize];

    let half_min_dim = (out_width.min(out_height) as f32) * 0.5;
    let circle_r_px = params.radius * half_min_dim;
    let cx_px = params.center_x * out_width as f32;
    let cy_px = params.center_y * out_height as f32;

    for oy in 0..out_height {
        for ox in 0..out_width {
            let dx = ox as f32 + 0.5 - cx_px;
            let dy = oy as f32 + 0.5 - cy_px;
            let r_px = (dx * dx + dy * dy).sqrt();
            let r_norm = r_px / circle_r_px;

            if r_norm > 1.0 {
                // Outside the fisheye circle — leave black
                continue;
            }

            let theta = params.r_to_theta(r_norm);
            // phi: azimuth angle in the fisheye image plane
            // dx = r*sin(phi), dy = -r*cos(phi)
            let phi = dy.atan2(dx) + std::f32::consts::FRAC_PI_2;

            let elevation = std::f32::consts::FRAC_PI_2 - theta;
            let sphere = SphericalCoord {
                azimuth_rad: phi,
                elevation_rad: elevation,
            };
            let uv = sphere_to_equirect(&sphere);

            let sample = bilinear_sample_u8(src, src_width, src_height, uv.u, uv.v, CH);
            let dst = (oy * out_width + ox) as usize * CH as usize;
            out[dst..dst + CH as usize].copy_from_slice(&sample);
        }
    }

    Ok(out)
}

// ─── Exposure compensation ───────────────────────────────────────────────────

/// Per-channel gain (multiplicative) applied to an equirectangular half-image
/// before blending, used to match brightness between front and back lenses.
///
/// Values near 1.0 leave the image unchanged; values above 1.0 brighten,
/// below 1.0 darken.  Each channel (R, G, B) can be adjusted independently
/// to correct colour-cast differences between lenses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExposureGain {
    /// Gain for the red channel.
    pub r: f32,
    /// Gain for the green channel.
    pub g: f32,
    /// Gain for the blue channel.
    pub b: f32,
}

impl ExposureGain {
    /// No adjustment (identity gain).
    pub fn identity() -> Self {
        Self {
            r: 1.0,
            g: 1.0,
            b: 1.0,
        }
    }

    /// Uniform gain applied equally to all channels.
    pub fn uniform(gain: f32) -> Self {
        Self {
            r: gain,
            g: gain,
            b: gain,
        }
    }

    /// Apply this gain to a single RGB pixel, returning clamped u8 values.
    #[inline]
    fn apply_u8(&self, pixel: &[u8]) -> [u8; 3] {
        let r = (pixel[0] as f32 * self.r).round().clamp(0.0, 255.0) as u8;
        let g = (pixel[1] as f32 * self.g).round().clamp(0.0, 255.0) as u8;
        let b = (pixel[2] as f32 * self.b).round().clamp(0.0, 255.0) as u8;
        [r, g, b]
    }
}

impl Default for ExposureGain {
    fn default() -> Self {
        Self::identity()
    }
}

// ─── Laplacian pyramid blending ───────────────────────────────────────────────

/// Number of pyramid levels used in Laplacian multi-resolution blending.
///
/// More levels = smoother blending over wider spatial frequencies but more
/// memory usage.  4 is a good default for most use-cases.
pub const DEFAULT_PYRAMID_LEVELS: usize = 4;

/// Compute the mean pixel intensity of an RGB image buffer.
///
/// Used by [`DualFisheyeStitcher`] to auto-compute exposure gain when
/// `auto_exposure` is enabled.
fn mean_intensity(data: &[u8]) -> f32 {
    if data.is_empty() {
        return 0.0;
    }
    let sum: u64 = data.iter().map(|&p| p as u64).sum();
    sum as f32 / data.len() as f32
}

/// Downsample an RGB image to half size using a simple 2×2 box filter.
fn downsample_half(src: &[u8], w: u32, h: u32) -> (Vec<u8>, u32, u32) {
    let ow = (w / 2).max(1);
    let oh = (h / 2).max(1);
    let stride = w as usize * 3;
    let mut out = vec![0u8; ow as usize * oh as usize * 3];

    for oy in 0..oh as usize {
        for ox in 0..ow as usize {
            let sx = ox * 2;
            let sy = oy * 2;
            // 4 neighbours — clamp to image boundary
            let sx1 = (sx + 1).min(w as usize - 1);
            let sy1 = (sy + 1).min(h as usize - 1);
            let b00 = sy * stride + sx * 3;
            let b10 = sy * stride + sx1 * 3;
            let b01 = sy1 * stride + sx * 3;
            let b11 = sy1 * stride + sx1 * 3;
            for c in 0..3 {
                let avg = (src[b00 + c] as u32
                    + src[b10 + c] as u32
                    + src[b01 + c] as u32
                    + src[b11 + c] as u32)
                    / 4;
                let dst = (oy * ow as usize + ox) * 3 + c;
                out[dst] = avg as u8;
            }
        }
    }
    (out, ow, oh)
}

/// Upsample an RGB image to double size using nearest-neighbour.
fn upsample_double(src: &[u8], sw: u32, sh: u32, tw: u32, th: u32) -> Vec<u8> {
    let mut out = vec![0u8; tw as usize * th as usize * 3];
    for oy in 0..th as usize {
        for ox in 0..tw as usize {
            let sx = ((ox * sw as usize) / tw as usize).min(sw as usize - 1);
            let sy = ((oy * sh as usize) / th as usize).min(sh as usize - 1);
            let src_base = (sy * sw as usize + sx) * 3;
            let dst_base = (oy * tw as usize + ox) * 3;
            out[dst_base..dst_base + 3].copy_from_slice(&src[src_base..src_base + 3]);
        }
    }
    out
}

/// Subtract two RGB images (a - b), storing results as f32 for the Laplacian.
fn image_sub_f32(a: &[u8], b: &[u8]) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .map(|(&av, &bv)| av as f32 - bv as f32)
        .collect()
}

/// Add f32 Laplacian residual back to a u8 base, clamping to [0,255].
fn image_add_laplacian(base: &[u8], lap: &[f32]) -> Vec<u8> {
    debug_assert_eq!(base.len(), lap.len());
    base.iter()
        .zip(lap.iter())
        .map(|(&b, &l)| (b as f32 + l).round().clamp(0.0, 255.0) as u8)
        .collect()
}

/// Blend two f32 Laplacian layers together using a per-pixel alpha mask.
fn blend_laplacian(
    lap_a: &[f32],
    lap_b: &[f32],
    alpha: &[f32], // same pixel count, one value per pixel
    channels: usize,
) -> Vec<f32> {
    debug_assert_eq!(lap_a.len(), lap_b.len());
    debug_assert_eq!(lap_a.len(), alpha.len() * channels);
    let mut out = vec![0.0f32; lap_a.len()];
    let n = alpha.len();
    for i in 0..n {
        for c in 0..channels {
            let idx = i * channels + c;
            out[idx] = lap_a[idx] * alpha[i] + lap_b[idx] * (1.0 - alpha[i]);
        }
    }
    out
}

/// Build a horizontal-seam alpha mask for a half-image of size `w × h`.
///
/// The blend zone spans `blend_width` pixels around the seam (column 0 for the
/// right half, column `w-1` for the left half).
fn build_seam_alpha(w: u32, h: u32, blend_half: u32, is_front: bool) -> Vec<f32> {
    let n = w as usize * h as usize;
    let mut alpha = vec![0.0f32; n];
    for row in 0..h as usize {
        for col in 0..w as usize {
            let dist = if is_front {
                // Front is on the left; seam is at col = w-1
                let dist_to_seam = (w as usize - 1 - col) as u32;
                dist_to_seam
            } else {
                // Back is on the right; seam is at col = 0
                col as u32
            };
            let a = if blend_half == 0 {
                if is_front {
                    1.0
                } else {
                    0.0
                }
            } else if dist < blend_half {
                dist as f32 / blend_half as f32
            } else {
                1.0
            };
            alpha[row * w as usize + col] = a;
        }
    }
    alpha
}

/// Multi-resolution (Laplacian pyramid) blend of two same-size RGB images.
///
/// * `img_a`  — "dominant" image (at the seam, alpha → 0)
/// * `img_b`  — "other" image
/// * `alpha`  — per-pixel blend weight for `img_a`; `1.0` = fully `img_a`
/// * `w`, `h` — image dimensions
/// * `levels` — number of pyramid levels (typically 3–6)
///
/// Builds a Gaussian pyramid (downsampled versions) and Laplacian residuals
/// for each image, blends each level separately, then collapses the blended
/// pyramid back to full resolution.
fn laplacian_pyramid_blend(
    img_a: &[u8],
    img_b: &[u8],
    alpha: &[f32],
    w: u32,
    h: u32,
    levels: usize,
) -> Vec<u8> {
    // ── Build Gaussian pyramids ───────────────────────────────────────────────
    let mut gauss_a: Vec<(Vec<u8>, u32, u32)> = Vec::with_capacity(levels + 1);
    let mut gauss_b: Vec<(Vec<u8>, u32, u32)> = Vec::with_capacity(levels + 1);
    let mut alpha_pyr: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(levels + 1);

    gauss_a.push((img_a.to_vec(), w, h));
    gauss_b.push((img_b.to_vec(), w, h));
    alpha_pyr.push((alpha.to_vec(), w, h));

    for _ in 0..levels {
        // SAFETY: gauss_a/gauss_b/alpha_pyr all received an initial push before this loop
        let (prev_a, pw, ph) = &gauss_a[gauss_a.len() - 1];
        let (pw, ph) = (*pw, *ph);
        let (down_a, dw, dh) = downsample_half(prev_a, pw, ph);
        let (prev_b, _, _) = &gauss_b[gauss_b.len() - 1];
        let (down_b, _, _) = downsample_half(prev_b, pw, ph);

        // Downsample alpha by averaging 2×2 blocks.
        // downsample_half expects 3-channel RGB, so expand alpha to 3 channels,
        // downsample, then extract every 3rd value (all channels are equal).
        // SAFETY: alpha_pyr received an initial push before this loop
        let (prev_alpha, aw, ah) = &alpha_pyr[alpha_pyr.len() - 1];
        let (aw, ah) = (*aw, *ah);
        let alpha_rgb: Vec<u8> = prev_alpha
            .iter()
            .flat_map(|&v| {
                let byte = (v * 255.0) as u8;
                [byte, byte, byte]
            })
            .collect();
        let (tmp_u8, _, _) = downsample_half(&alpha_rgb, aw, ah);
        let down_alpha: Vec<f32> = tmp_u8
            .chunks_exact(3)
            .map(|c| c[0] as f32 / 255.0)
            .collect();

        gauss_a.push((down_a, dw, dh));
        gauss_b.push((down_b, dw, dh));
        alpha_pyr.push((down_alpha, dw, dh));
    }

    // ── Build Laplacian pyramids ──────────────────────────────────────────────
    let mut lap_a: Vec<Vec<f32>> = Vec::with_capacity(levels);
    let mut lap_b: Vec<Vec<f32>> = Vec::with_capacity(levels);

    for i in 0..levels {
        let (ga, gw, gh) = &gauss_a[i];
        let (ga_next, gw_next, gh_next) = &gauss_a[i + 1];
        let up_a = upsample_double(ga_next, *gw_next, *gh_next, *gw, *gh);
        lap_a.push(image_sub_f32(ga, &up_a));

        let (gb, _, _) = &gauss_b[i];
        let (gb_next, _, _) = &gauss_b[i + 1];
        let up_b = upsample_double(gb_next, *gw_next, *gh_next, *gw, *gh);
        lap_b.push(image_sub_f32(gb, &up_b));
    }

    // ── Blend each level ──────────────────────────────────────────────────────
    let mut blended_lap: Vec<(Vec<f32>, u32, u32)> = Vec::with_capacity(levels);
    for i in 0..levels {
        let (la, lw, lh) = (&lap_a[i], gauss_a[i].1, gauss_a[i].2);
        let (lb, _, _) = (&lap_b[i], gauss_b[i].1, gauss_b[i].2);
        let (ap, _, _) = &alpha_pyr[i];
        let blended = blend_laplacian(la, lb, ap, 3);
        blended_lap.push((blended, lw, lh));
    }

    // ── Blend the coarsest Gaussian level ────────────────────────────────────
    let (coarse_a, cw, ch) = &gauss_a[levels];
    let (coarse_b, _, _) = &gauss_b[levels];
    let (coarse_alpha, _, _) = &alpha_pyr[levels];
    let coarse_fa: Vec<f32> = coarse_a.iter().map(|&v| v as f32).collect();
    let coarse_fb: Vec<f32> = coarse_b.iter().map(|&v| v as f32).collect();
    let coarse_blended_f: Vec<f32> = blend_laplacian(&coarse_fa, &coarse_fb, coarse_alpha, 3);
    let mut current: Vec<u8> = coarse_blended_f
        .iter()
        .map(|&v| v.round().clamp(0.0, 255.0) as u8)
        .collect();
    let mut cur_w = *cw;
    let mut cur_h = *ch;

    // ── Collapse pyramid ──────────────────────────────────────────────────────
    for i in (0..levels).rev() {
        let (lap_blended, lw, lh) = &blended_lap[i];
        let up = upsample_double(&current, cur_w, cur_h, *lw, *lh);
        current = image_add_laplacian(&up, lap_blended);
        cur_w = *lw;
        cur_h = *lh;
    }

    current
}

// ─── Dual-fisheye stitcher ───────────────────────────────────────────────────

/// Blend mode used by the [`DualFisheyeStitcher`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendMode {
    /// Simple linear alpha ramp — fast, may show colour/frequency seams.
    #[default]
    Linear,
    /// Multi-resolution Laplacian pyramid blend — slower but seamless.
    Laplacian,
}

/// Builder for [`DualFisheyeStitcher`] with all optional settings.
///
/// # Example
/// ```rust
/// use oximedia_360::fisheye::{DualFisheyeStitcherBuilder, FisheyeParams, BlendMode};
///
/// let stitcher = DualFisheyeStitcherBuilder::new()
///     .front_params(FisheyeParams::equidistant(185.0))
///     .back_params(FisheyeParams::equidistant(185.0))
///     .overlap_blend_width(48)
///     .blend_mode(BlendMode::Laplacian)
///     .pyramid_levels(4)
///     .build();
/// ```
#[derive(Debug, Clone, Default)]
pub struct DualFisheyeStitcherBuilder {
    front_params: Option<FisheyeParams>,
    back_params: Option<FisheyeParams>,
    overlap_blend_width: Option<u32>,
    front_gain: Option<ExposureGain>,
    back_gain: Option<ExposureGain>,
    auto_exposure: bool,
    blend_mode: Option<BlendMode>,
    pyramid_levels: Option<usize>,
}

impl DualFisheyeStitcherBuilder {
    /// Create a builder with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the front lens fisheye parameters.
    pub fn front_params(mut self, p: FisheyeParams) -> Self {
        self.front_params = Some(p);
        self
    }

    /// Set the rear lens fisheye parameters.
    pub fn back_params(mut self, p: FisheyeParams) -> Self {
        self.back_params = Some(p);
        self
    }

    /// Set the overlap blend zone width in output pixels.
    pub fn overlap_blend_width(mut self, w: u32) -> Self {
        self.overlap_blend_width = Some(w);
        self
    }

    /// Set a fixed exposure gain for the front lens image.
    pub fn front_gain(mut self, g: ExposureGain) -> Self {
        self.front_gain = Some(g);
        self
    }

    /// Set a fixed exposure gain for the rear lens image.
    pub fn back_gain(mut self, g: ExposureGain) -> Self {
        self.back_gain = Some(g);
        self
    }

    /// Enable automatic exposure equalisation (adjusts back gain to match
    /// front mean intensity at stitch time).
    pub fn auto_exposure(mut self, enabled: bool) -> Self {
        self.auto_exposure = enabled;
        self
    }

    /// Select the blend mode (linear or Laplacian pyramid).
    pub fn blend_mode(mut self, mode: BlendMode) -> Self {
        self.blend_mode = Some(mode);
        self
    }

    /// Number of Laplacian pyramid levels (ignored for [`BlendMode::Linear`]).
    pub fn pyramid_levels(mut self, n: usize) -> Self {
        self.pyramid_levels = Some(n);
        self
    }

    /// Build the stitcher.
    pub fn build(self) -> DualFisheyeStitcher {
        DualFisheyeStitcher {
            front_params: self
                .front_params
                .unwrap_or_else(|| FisheyeParams::equidistant(180.0)),
            back_params: self
                .back_params
                .unwrap_or_else(|| FisheyeParams::equidistant(180.0)),
            overlap_blend_width: self.overlap_blend_width.unwrap_or(32),
            front_gain: self.front_gain.unwrap_or_default(),
            back_gain: self.back_gain.unwrap_or_default(),
            auto_exposure: self.auto_exposure,
            blend_mode: self.blend_mode.unwrap_or_default(),
            pyramid_levels: self.pyramid_levels.unwrap_or(DEFAULT_PYRAMID_LEVELS),
        }
    }
}

/// Stitcher for dual-fisheye cameras (front + back lens on opposite sides).
///
/// Converts both fisheye images to equirectangular half-images, optionally
/// applies per-lens exposure compensation, then blends them in the overlap zone
/// using either linear-alpha or multi-resolution Laplacian pyramid blending.
///
/// Use [`DualFisheyeStitcherBuilder`] for fine-grained configuration, or the
/// convenience constructor [`DualFisheyeStitcher::symmetric_180`] for the
/// common case of identical 180° lenses.
#[derive(Debug, Clone)]
pub struct DualFisheyeStitcher {
    /// Parameters for the front-facing lens.
    pub front_params: FisheyeParams,
    /// Parameters for the rear-facing lens.
    pub back_params: FisheyeParams,
    /// Width of the overlap blend zone in output pixels.
    pub overlap_blend_width: u32,
    /// Multiplicative gain applied to the front equirectangular half-image.
    pub front_gain: ExposureGain,
    /// Multiplicative gain applied to the rear equirectangular half-image.
    pub back_gain: ExposureGain,
    /// When `true`, override `back_gain` at stitch time to equalise mean
    /// intensity between front and back.
    pub auto_exposure: bool,
    /// Algorithm used to blend the two half-images at the seam.
    pub blend_mode: BlendMode,
    /// Number of Laplacian pyramid levels (only used with `BlendMode::Laplacian`).
    pub pyramid_levels: usize,
}

impl DualFisheyeStitcher {
    /// Create a new stitcher with symmetric 180° equidistant fisheye lenses.
    pub fn symmetric_180() -> Self {
        DualFisheyeStitcherBuilder::new().build()
    }

    /// Apply exposure gain to an equirectangular half-image in-place.
    fn apply_gain(data: &mut [u8], gain: ExposureGain) {
        if gain == ExposureGain::identity() {
            return;
        }
        for pixel in data.chunks_exact_mut(3) {
            let adjusted = gain.apply_u8(pixel);
            pixel.copy_from_slice(&adjusted);
        }
    }

    /// Compute a per-channel gain that rescales `src` mean intensity to match
    /// `reference` mean intensity.
    fn compute_auto_gain(reference: &[u8], src: &[u8]) -> ExposureGain {
        let ref_mean = mean_intensity(reference);
        let src_mean = mean_intensity(src);
        if src_mean < 1.0 {
            return ExposureGain::identity();
        }
        let g = (ref_mean / src_mean).clamp(0.1, 10.0);
        ExposureGain::uniform(g)
    }

    /// Stitch front and back fisheye images into a single equirectangular frame.
    ///
    /// * `front`  — front fisheye pixel data (RGB, 3 bpp, square, row-major)
    /// * `back`   — rear  fisheye pixel data (same dimensions as `front`)
    /// * `width`  — input image width in pixels
    /// * `height` — input image height in pixels
    ///
    /// The output equirectangular image has width `width * 2` and height `height`.
    ///
    /// # Errors
    /// Returns [`VrError::InvalidDimensions`] if dimensions are zero.
    pub fn stitch(
        &self,
        front: &[u8],
        back: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Vec<u8>, VrError> {
        if width == 0 || height == 0 {
            return Err(VrError::InvalidDimensions(
                "width/height must be > 0".into(),
            ));
        }

        let out_w = width * 2;
        let out_h = height;
        let half_w = out_w / 2;

        // Convert each fisheye to equirectangular at half-width
        let mut front_eq =
            fisheye_to_equirect(front, width, height, &self.front_params, half_w, out_h)?;
        let mut back_eq =
            fisheye_to_equirect(back, width, height, &self.back_params, half_w, out_h)?;

        // Apply exposure compensation
        Self::apply_gain(&mut front_eq, self.front_gain);
        if self.auto_exposure {
            let auto_gain = Self::compute_auto_gain(&front_eq, &back_eq);
            Self::apply_gain(&mut back_eq, auto_gain);
        } else {
            Self::apply_gain(&mut back_eq, self.back_gain);
        }

        match self.blend_mode {
            BlendMode::Linear => self.blend_linear(&front_eq, &back_eq, half_w, out_h),
            BlendMode::Laplacian => {
                self.blend_laplacian_pyramid(&front_eq, &back_eq, half_w, out_h)
            }
        }
    }

    /// Linear-alpha seam blending (original algorithm).
    fn blend_linear(
        &self,
        front_eq: &[u8],
        back_eq: &[u8],
        half_w: u32,
        out_h: u32,
    ) -> Result<Vec<u8>, VrError> {
        const CH: usize = 3;
        let blend_half = self.overlap_blend_width / 2;
        let hw = half_w as usize;
        let out_w = half_w as usize * 2;
        let mut out = vec![0u8; out_w * out_h as usize * CH];

        for row in 0..out_h as usize {
            for col in 0..out_w {
                let dst_base = (row * out_w + col) * CH;

                let (pixel, alpha_front) = if col < hw {
                    let dist_to_seam = (hw - col) as u32;
                    let alpha = if blend_half > 0 && dist_to_seam < blend_half {
                        dist_to_seam as f32 / blend_half as f32
                    } else {
                        1.0
                    };
                    let src_base = (row * hw + col) * CH;
                    (&front_eq[src_base..src_base + CH], alpha)
                } else {
                    let back_col = col - hw;
                    let dist_to_seam = back_col as u32;
                    let alpha = if blend_half > 0 && dist_to_seam < blend_half {
                        dist_to_seam as f32 / blend_half as f32
                    } else {
                        1.0
                    };
                    let src_base = (row * hw + back_col) * CH;
                    (&back_eq[src_base..src_base + CH], alpha)
                };

                if alpha_front < 1.0 {
                    let other_col = if col < hw {
                        let seam_offset = hw - col;
                        seam_offset.min(hw - 1)
                    } else {
                        let seam_offset = col - hw;
                        seam_offset.min(hw - 1)
                    };
                    let other_is_front = col >= hw;
                    let other_src_base = (row * hw + other_col) * CH;
                    let other = if other_is_front {
                        &front_eq[other_src_base..other_src_base + CH]
                    } else {
                        &back_eq[other_src_base..other_src_base + CH]
                    };
                    for c in 0..CH {
                        let blended =
                            pixel[c] as f32 * alpha_front + other[c] as f32 * (1.0 - alpha_front);
                        out[dst_base + c] = blended.round().clamp(0.0, 255.0) as u8;
                    }
                } else {
                    out[dst_base..dst_base + CH].copy_from_slice(pixel);
                }
            }
        }
        Ok(out)
    }

    /// Multi-resolution Laplacian pyramid seam blending.
    fn blend_laplacian_pyramid(
        &self,
        front_eq: &[u8],
        back_eq: &[u8],
        half_w: u32,
        out_h: u32,
    ) -> Result<Vec<u8>, VrError> {
        let blend_half = self.overlap_blend_width / 2;
        let levels = self.pyramid_levels.max(1);

        // Build alpha masks for each half — front is on the left, back on the right
        let front_alpha = build_seam_alpha(half_w, out_h, blend_half, true);
        let back_alpha = build_seam_alpha(half_w, out_h, blend_half, false);

        // Blend each half using Laplacian pyramid with its own alpha mask
        // The "A" image in each case is the dominant half, "B" is the blending source.
        // For the left half: dominant = front, blend target = back (mirrored seam).
        // For the right half: dominant = back, blend target = front (mirrored seam).
        // We concatenate the two halves after blending.

        let blended_front =
            laplacian_pyramid_blend(front_eq, back_eq, &front_alpha, half_w, out_h, levels);
        let blended_back =
            laplacian_pyramid_blend(back_eq, front_eq, &back_alpha, half_w, out_h, levels);

        // Concatenate left (front) and right (back) halves
        const CH: usize = 3;
        let out_w = half_w as usize * 2;
        let mut out = vec![0u8; out_w * out_h as usize * CH];
        for row in 0..out_h as usize {
            let src_row_front =
                &blended_front[row * half_w as usize * CH..(row + 1) * half_w as usize * CH];
            let src_row_back =
                &blended_back[row * half_w as usize * CH..(row + 1) * half_w as usize * CH];
            let dst_row_start = row * out_w * CH;
            out[dst_row_start..dst_row_start + half_w as usize * CH].copy_from_slice(src_row_front);
            out[dst_row_start + half_w as usize * CH..dst_row_start + out_w * CH]
                .copy_from_slice(src_row_back);
        }
        Ok(out)
    }
}

// ─── Horizon detection ───────────────────────────────────────────────────────

/// Compute the elevation angle of the fisheye horizon.
///
/// The "horizon" is the circle at the edge of the fisheye view.
/// For a lens with `fov_deg` total FOV, the elevation at the horizon is
/// `π/2 − fov_rad/2`.
pub fn detect_horizon(params: &FisheyeParams) -> f32 {
    let fov_rad = params.fov_deg * std::f32::consts::PI / 180.0;
    std::f32::consts::FRAC_PI_2 - fov_rad * 0.5
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn validate_fisheye(
    src: &[u8],
    src_w: u32,
    src_h: u32,
    out_w: u32,
    out_h: u32,
) -> Result<(), VrError> {
    if src_w == 0 || src_h == 0 || out_w == 0 || out_h == 0 {
        return Err(VrError::InvalidDimensions(
            "all image dimensions must be > 0".into(),
        ));
    }
    let expected = src_w as usize * src_h as usize * 3;
    if src.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: src.len(),
        });
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn solid(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity(w as usize * h as usize * 3);
        for _ in 0..(w * h) {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    // ── FisheyeParams: theta ↔ r ─────────────────────────────────────────────

    #[test]
    fn equidistant_r_at_half_fov_is_one() {
        let p = FisheyeParams::equidistant(180.0);
        let r = p.theta_to_r(p.half_fov_rad()).expect("should be Some");
        assert!((r - 1.0).abs() < 1e-5);
    }

    #[test]
    fn equidistant_r_at_zero_is_zero() {
        let p = FisheyeParams::equidistant(180.0);
        let r = p.theta_to_r(0.0).expect("should be Some");
        assert!(r.abs() < 1e-5);
    }

    #[test]
    fn equidistant_theta_to_r_roundtrip() {
        let p = FisheyeParams::equidistant(180.0);
        for i in 0..=10 {
            let theta = p.half_fov_rad() * (i as f32 / 10.0);
            if let Some(r) = p.theta_to_r(theta) {
                let theta_back = p.r_to_theta(r);
                assert!(
                    (theta_back - theta).abs() < 1e-4,
                    "roundtrip failed: theta={theta:.4} back={theta_back:.4}"
                );
            }
        }
    }

    #[test]
    fn equisolid_r_at_half_fov_is_one() {
        let p = FisheyeParams {
            fov_deg: 180.0,
            model: FisheyeModel::Equisolid,
            ..FisheyeParams::equidistant(180.0)
        };
        let r = p.theta_to_r(p.half_fov_rad()).expect("Some");
        assert!((r - 1.0).abs() < 1e-4);
    }

    #[test]
    fn orthographic_r_at_half_fov_is_one() {
        let p = FisheyeParams {
            fov_deg: 160.0, // orthographic can't do 180° (sin(90°)=1 boundary)
            model: FisheyeModel::Orthographic,
            ..FisheyeParams::equidistant(160.0)
        };
        let r = p.theta_to_r(p.half_fov_rad()).expect("Some");
        assert!((r - 1.0).abs() < 1e-4);
    }

    #[test]
    fn stereographic_r_at_half_fov_is_one() {
        let p = FisheyeParams {
            fov_deg: 170.0,
            model: FisheyeModel::Stereographic,
            ..FisheyeParams::equidistant(170.0)
        };
        let r = p.theta_to_r(p.half_fov_rad()).expect("Some");
        assert!((r - 1.0).abs() < 1e-4);
    }

    #[test]
    fn theta_outside_fov_returns_none() {
        let p = FisheyeParams::equidistant(180.0);
        let result = p.theta_to_r(PI * 0.6); // > π/2
        assert!(result.is_none());
    }

    // ── fisheye_to_equirect ──────────────────────────────────────────────────

    #[test]
    fn fisheye_to_equirect_output_size() {
        let src = solid(64, 64, 100, 150, 200);
        let p = FisheyeParams::equidistant(180.0);
        let out = fisheye_to_equirect(&src, 64, 64, &p, 128, 64).expect("ok");
        assert_eq!(out.len(), 128 * 64 * 3);
    }

    #[test]
    fn fisheye_to_equirect_zero_dimension_error() {
        let src = solid(64, 64, 0, 0, 0);
        let p = FisheyeParams::equidistant(180.0);
        assert!(fisheye_to_equirect(&src, 0, 64, &p, 128, 64).is_err());
        assert!(fisheye_to_equirect(&src, 64, 0, &p, 128, 64).is_err());
        assert!(fisheye_to_equirect(&src, 64, 64, &p, 0, 64).is_err());
        assert!(fisheye_to_equirect(&src, 64, 64, &p, 128, 0).is_err());
    }

    #[test]
    fn fisheye_to_equirect_buffer_too_small_error() {
        let src = vec![0u8; 10];
        let p = FisheyeParams::equidistant(180.0);
        assert!(fisheye_to_equirect(&src, 64, 64, &p, 128, 64).is_err());
    }

    // ── equirect_to_fisheye ──────────────────────────────────────────────────

    #[test]
    fn equirect_to_fisheye_output_size() {
        let src = solid(128, 64, 200, 100, 50);
        let p = FisheyeParams::equidistant(180.0);
        let out = equirect_to_fisheye(&src, 128, 64, &p, 64, 64).expect("ok");
        assert_eq!(out.len(), 64 * 64 * 3);
    }

    #[test]
    fn equirect_to_fisheye_zero_dimension_error() {
        let src = solid(128, 64, 0, 0, 0);
        let p = FisheyeParams::equidistant(180.0);
        assert!(equirect_to_fisheye(&src, 0, 64, &p, 64, 64).is_err());
    }

    // ── fisheye → equirect → fisheye centre pixel roundtrip ─────────────────

    #[test]
    fn fisheye_equirect_fisheye_centre_colour_preserved() {
        // A solid-colour fisheye image: after forward+inverse the centre should
        // match fairly closely (the exact boundaries will be black)
        let src = solid(64, 64, 180, 90, 45);
        let p = FisheyeParams::equidistant(180.0);
        let mid_eq = fisheye_to_equirect(&src, 64, 64, &p, 128, 64).expect("forward");
        let back = equirect_to_fisheye(&mid_eq, 128, 64, &p, 64, 64).expect("inverse");

        // centre pixel of the fisheye should still be close to 180
        let cx = 32usize;
        let cy = 32usize;
        let base = (cy * 64 + cx) * 3;
        let err_r = (back[base] as i32 - 180).abs();
        assert!(err_r <= 20, "R error at centre: {err_r}");
    }

    // ── detect_horizon ───────────────────────────────────────────────────────

    #[test]
    fn horizon_180_fov_is_zero_elevation() {
        let p = FisheyeParams::equidistant(180.0);
        let h = detect_horizon(&p);
        assert!(h.abs() < 1e-4);
    }

    #[test]
    fn horizon_90_fov_is_pi_over_4() {
        let p = FisheyeParams::equidistant(90.0);
        let h = detect_horizon(&p);
        assert!((h - PI / 4.0).abs() < 1e-4);
    }

    #[test]
    fn horizon_360_fov_is_negative_pi_over_2() {
        let p = FisheyeParams::equidistant(360.0);
        let h = detect_horizon(&p);
        assert!((h + PI / 2.0).abs() < 1e-4);
    }

    // ── DualFisheyeStitcher ──────────────────────────────────────────────────

    #[test]
    fn dual_fisheye_stitch_output_size() {
        let front = solid(64, 64, 100, 0, 0);
        let back = solid(64, 64, 0, 100, 0);
        let stitcher = DualFisheyeStitcher::symmetric_180();
        let out = stitcher.stitch(&front, &back, 64, 64).expect("stitch");
        assert_eq!(out.len(), 128 * 64 * 3);
    }

    #[test]
    fn dual_fisheye_stitch_zero_dimension_error() {
        let front = solid(64, 64, 0, 0, 0);
        let back = solid(64, 64, 0, 0, 0);
        let stitcher = DualFisheyeStitcher::symmetric_180();
        assert!(stitcher.stitch(&front, &back, 0, 64).is_err());
        assert!(stitcher.stitch(&front, &back, 64, 0).is_err());
    }

    #[test]
    fn dual_fisheye_stitch_non_black_pixels() {
        let front = solid(32, 32, 200, 100, 50);
        let back = solid(32, 32, 50, 200, 100);
        let stitcher = DualFisheyeStitcherBuilder::new()
            .front_params(FisheyeParams::equidistant(180.0))
            .back_params(FisheyeParams::equidistant(180.0))
            .overlap_blend_width(4)
            .build();
        let out = stitcher.stitch(&front, &back, 32, 32).expect("stitch");
        // At least some pixels should be non-black (fisheye circle is non-empty)
        let non_black = out.iter().any(|&p| p > 0);
        assert!(non_black);
    }

    // ── Exposure compensation ─────────────────────────────────────────────────

    #[test]
    fn exposure_gain_identity_no_change() {
        let pixel = [100u8, 150, 200];
        let gain = ExposureGain::identity();
        let out = gain.apply_u8(&pixel);
        assert_eq!(out, [100, 150, 200]);
    }

    #[test]
    fn exposure_gain_double() {
        let pixel = [50u8, 100, 200];
        let gain = ExposureGain::uniform(2.0);
        let out = gain.apply_u8(&pixel);
        assert_eq!(out[0], 100);
        assert_eq!(out[1], 200);
        assert_eq!(out[2], 255); // clamped from 400
    }

    #[test]
    fn exposure_gain_halve() {
        let pixel = [100u8, 200, 60];
        let gain = ExposureGain::uniform(0.5);
        let out = gain.apply_u8(&pixel);
        assert_eq!(out[0], 50);
        assert_eq!(out[1], 100);
        assert_eq!(out[2], 30);
    }

    #[test]
    fn auto_exposure_stitch_does_not_panic() {
        let front = solid(32, 32, 200, 200, 200);
        let back = solid(32, 32, 100, 100, 100);
        let stitcher = DualFisheyeStitcherBuilder::new()
            .auto_exposure(true)
            .overlap_blend_width(4)
            .build();
        let out = stitcher
            .stitch(&front, &back, 32, 32)
            .expect("auto-exposure stitch");
        assert_eq!(out.len(), 64 * 32 * 3);
    }

    #[test]
    fn fixed_exposure_gain_applied() {
        let front = solid(32, 32, 100, 100, 100);
        let back = solid(32, 32, 50, 50, 50);
        // back_gain = 2.0 to bring back up to front level
        let stitcher = DualFisheyeStitcherBuilder::new()
            .back_gain(ExposureGain::uniform(2.0))
            .overlap_blend_width(0)
            .build();
        let out = stitcher.stitch(&front, &back, 32, 32).expect("gain stitch");
        assert_eq!(out.len(), 64 * 32 * 3);
        // Right half should be brighter than original (50 → ~100)
        let non_zero = out.iter().any(|&p| p > 60);
        assert!(non_zero, "expected brightened pixels from back gain");
    }

    // ── BlendMode builder ────────────────────────────────────────────────────

    #[test]
    fn builder_linear_blend_output_size() {
        let front = solid(32, 32, 150, 50, 100);
        let back = solid(32, 32, 50, 150, 50);
        let stitcher = DualFisheyeStitcherBuilder::new()
            .blend_mode(BlendMode::Linear)
            .overlap_blend_width(8)
            .build();
        let out = stitcher
            .stitch(&front, &back, 32, 32)
            .expect("linear blend");
        assert_eq!(out.len(), 64 * 32 * 3);
    }

    #[test]
    fn builder_laplacian_blend_output_size() {
        let front = solid(32, 32, 180, 90, 45);
        let back = solid(32, 32, 45, 180, 90);
        let stitcher = DualFisheyeStitcherBuilder::new()
            .blend_mode(BlendMode::Laplacian)
            .pyramid_levels(3)
            .overlap_blend_width(8)
            .build();
        let out = stitcher
            .stitch(&front, &back, 32, 32)
            .expect("laplacian blend");
        assert_eq!(out.len(), 64 * 32 * 3);
    }

    #[test]
    fn laplacian_blend_solid_colour_preserved() {
        // Both images same colour: output should match that colour everywhere
        let front = solid(32, 32, 128, 64, 32);
        let back = solid(32, 32, 128, 64, 32);
        let stitcher = DualFisheyeStitcherBuilder::new()
            .blend_mode(BlendMode::Laplacian)
            .pyramid_levels(3)
            .overlap_blend_width(8)
            .build();
        let out = stitcher
            .stitch(&front, &back, 32, 32)
            .expect("laplacian same-colour");
        // Check a pixel well inside the fisheye circle (row=4, col=8 in 64×32 output).
        // Row=16, col=16 in equirect maps to theta≈93° which is outside the 180° FOV, so
        // those pixels are always black.  Row=4, col=8 maps to theta≈48° — well inside.
        let base = (4 * 64 + 8) * 3;
        let err_r = (out[base] as i32 - 128).abs();
        assert!(err_r <= 20, "R error at (row=4,col=8): {err_r}");
    }

    // ── Laplacian pyramid internals ──────────────────────────────────────────

    #[test]
    fn downsample_half_reduces_size() {
        let img: Vec<u8> = (0..16 * 8 * 3).map(|i| (i % 256) as u8).collect();
        let (out, ow, oh) = super::downsample_half(&img, 16, 8);
        assert_eq!(ow, 8);
        assert_eq!(oh, 4);
        assert_eq!(out.len(), 8 * 4 * 3);
    }

    #[test]
    fn upsample_double_increases_size() {
        let img: Vec<u8> = (0..4 * 4 * 3).map(|i| (i % 256) as u8).collect();
        let out = super::upsample_double(&img, 4, 4, 8, 8);
        assert_eq!(out.len(), 8 * 8 * 3);
    }

    #[test]
    fn build_seam_alpha_front_is_one_at_left() {
        // Front is dominant on left; at col=0 it should be far from seam → 1.0
        let alpha = super::build_seam_alpha(16, 4, 4, true);
        // col 0 is furthest from seam → alpha = 1.0
        assert!((alpha[0] - 1.0).abs() < 1e-5, "alpha[0]={}", alpha[0]);
    }

    #[test]
    fn build_seam_alpha_back_is_one_at_right() {
        // Back is dominant on right; at col=15 it should be far from seam → 1.0
        let alpha = super::build_seam_alpha(16, 1, 4, false);
        // col 15 is furthest from seam (right edge) → alpha = 1.0
        assert!((alpha[15] - 1.0).abs() < 1e-5, "alpha[15]={}", alpha[15]);
    }
}
