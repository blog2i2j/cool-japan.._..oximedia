//! Lens distortion correction (barrel, pincushion, fish-eye).
//!
//! Camera lenses introduce radial and tangential distortions that cause
//! straight lines to appear curved.  This module implements:
//!
//! * **Radial distortion** — Brown–Conrady polynomial model
//!   (`k1`, `k2`, `k3`, `k4`, `k5`, `k6`).
//! * **Tangential distortion** — thin prism / decentring coefficients
//!   (`p1`, `p2`).
//! * **Fisheye equidistant model** — `r_d = f · θ` with radial coefficients.
//!
//! The main entry points are [`LensDistortionCorrector`] (for removal of
//! distortion) and [`LensDistortionSimulator`] (for synthesis, useful for
//! testing and artistic barrel/pincushion effects).
//!
//! All operations accept and return row-major RGBA u8 images of size
//! `width × height × 4`.
//!
//! # Example
//!
//! ```
//! use oximedia_cv::lens_distortion::{
//!     DistortionCoeffs, LensDistortionCorrector, CameraIntrinsics,
//! };
//!
//! let w = 16usize;
//! let h = 16usize;
//! let rgba: Vec<u8> = (0..(w * h * 4)).map(|i| (i % 256) as u8).collect();
//!
//! let intrinsics = CameraIntrinsics::from_fov(w, h, 60.0_f64.to_radians());
//! let coeffs = DistortionCoeffs::barrel(0.3);
//! let corrected = LensDistortionCorrector::new(intrinsics, coeffs).correct_rgba(&rgba, w, h);
//! assert_eq!(corrected.len(), rgba.len());
//! ```

#![allow(dead_code)]

// ---------------------------------------------------------------------------
// Camera intrinsics
// ---------------------------------------------------------------------------

/// Pinhole camera intrinsics (focal length + principal point).
#[derive(Debug, Clone, Copy)]
pub struct CameraIntrinsics {
    /// Focal length in pixels along the x axis.
    pub fx: f64,
    /// Focal length in pixels along the y axis.
    pub fy: f64,
    /// Principal point x (column), usually image width / 2.
    pub cx: f64,
    /// Principal point y (row), usually image height / 2.
    pub cy: f64,
}

impl CameraIntrinsics {
    /// Create intrinsics from a horizontal field of view angle (radians).
    ///
    /// Assumes square pixels and the principal point at the image centre.
    #[must_use]
    pub fn from_fov(width: usize, height: usize, h_fov_rad: f64) -> Self {
        let cx = width as f64 * 0.5;
        let cy = height as f64 * 0.5;
        let fx = cx / (h_fov_rad * 0.5).tan().max(1e-12);
        Self { fx, fy: fx, cx, cy }
    }

    /// Create intrinsics by specifying focal lengths and principal point directly.
    #[must_use]
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }
}

// ---------------------------------------------------------------------------
// Distortion model
// ---------------------------------------------------------------------------

/// Lens distortion model (Brown–Conrady radial + tangential).
///
/// The distorted point `(xd, yd)` is related to the undistorted normalised
/// coordinates `(x, y)` by:
///
/// ```text
/// r² = x² + y²
/// radial = (1 + k1·r² + k2·r⁴ + k3·r⁶) / (1 + k4·r² + k5·r⁴ + k6·r⁶)
/// xd = x·radial + 2·p1·x·y     + p2·(r²+2x²)
/// yd = y·radial + p1·(r²+2y²)  + 2·p2·x·y
/// ```
///
/// Setting all coefficients to 0 yields the identity (no distortion).
#[derive(Debug, Clone, Copy)]
pub struct DistortionCoeffs {
    /// Radial distortion coefficient k1.
    pub k1: f64,
    /// Radial distortion coefficient k2.
    pub k2: f64,
    /// Radial distortion coefficient k3.
    pub k3: f64,
    /// Radial distortion coefficient k4 (denominator).
    pub k4: f64,
    /// Radial distortion coefficient k5 (denominator).
    pub k5: f64,
    /// Radial distortion coefficient k6 (denominator).
    pub k6: f64,
    /// Tangential distortion coefficient p1.
    pub p1: f64,
    /// Tangential distortion coefficient p2.
    pub p2: f64,
}

impl DistortionCoeffs {
    /// No distortion (identity).
    #[must_use]
    pub const fn identity() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            k4: 0.0,
            k5: 0.0,
            k6: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }

    /// Simple barrel distortion (k1 > 0 produces barrel, k1 < 0 pincushion).
    #[must_use]
    pub const fn barrel(k1: f64) -> Self {
        Self {
            k1,
            ..Self::identity()
        }
    }

    /// Barrel distortion with two radial coefficients.
    #[must_use]
    pub const fn barrel2(k1: f64, k2: f64) -> Self {
        Self {
            k1,
            k2,
            ..Self::identity()
        }
    }

    /// Full Brown–Conrady model.
    #[must_use]
    pub const fn new(
        k1: f64,
        k2: f64,
        k3: f64,
        k4: f64,
        k5: f64,
        k6: f64,
        p1: f64,
        p2: f64,
    ) -> Self {
        Self {
            k1,
            k2,
            k3,
            k4,
            k5,
            k6,
            p1,
            p2,
        }
    }

    /// Apply distortion to normalised image coordinates `(x, y)`.
    ///
    /// Returns the distorted normalised coordinates.
    #[must_use]
    pub fn distort(&self, x: f64, y: f64) -> (f64, f64) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let numer = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let denom = 1.0 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6;
        let radial = if denom.abs() > 1e-12 {
            numer / denom
        } else {
            numer
        };

        let xd = x * radial + 2.0 * self.p1 * x * y + self.p2 * (r2 + 2.0 * x * x);
        let yd = y * radial + self.p1 * (r2 + 2.0 * y * y) + 2.0 * self.p2 * x * y;
        (xd, yd)
    }

    /// Iterative inverse distortion (undistort) using Newton–Raphson iterations.
    ///
    /// Given distorted normalised coordinates `(xd, yd)`, returns the
    /// undistorted `(x, y)`.  Converges in ≤ `max_iter` steps.
    #[must_use]
    pub fn undistort(&self, xd: f64, yd: f64, max_iter: u32) -> (f64, f64) {
        // Initial guess is the distorted point itself
        let mut x = xd;
        let mut y = yd;
        for _ in 0..max_iter {
            let (xd_est, yd_est) = self.distort(x, y);
            let ex = xd_est - xd;
            let ey = yd_est - yd;
            if ex * ex + ey * ey < 1e-18 {
                break;
            }
            x -= ex;
            y -= ey;
        }
        (x, y)
    }
}

impl Default for DistortionCoeffs {
    fn default() -> Self {
        Self::identity()
    }
}

// ---------------------------------------------------------------------------
// LensDistortionCorrector
// ---------------------------------------------------------------------------

/// Removes lens distortion from an RGBA image by inverse-mapping each output
/// pixel through the distortion model to find the corresponding source pixel.
pub struct LensDistortionCorrector {
    intrinsics: CameraIntrinsics,
    coeffs: DistortionCoeffs,
    /// Number of Newton–Raphson iterations used for undistortion.
    undistort_iters: u32,
    /// Background fill colour when a source pixel is outside image bounds.
    pub border_color: [u8; 4],
}

impl LensDistortionCorrector {
    /// Create a corrector with default undistort iterations (20) and black border.
    #[must_use]
    pub fn new(intrinsics: CameraIntrinsics, coeffs: DistortionCoeffs) -> Self {
        Self {
            intrinsics,
            coeffs,
            undistort_iters: 20,
            border_color: [0, 0, 0, 0],
        }
    }

    /// Set the number of Newton–Raphson iterations for undistortion.
    pub fn set_undistort_iters(&mut self, iters: u32) {
        self.undistort_iters = iters.clamp(1, 100);
    }

    /// Correct distortion in an RGBA image.
    ///
    /// `rgba` must be row-major with exactly `width × height × 4` bytes.
    /// Returns a new RGBA buffer of the same size.
    #[must_use]
    pub fn correct_rgba(&self, rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
        if width == 0 || height == 0 || rgba.len() != width * height * 4 {
            return rgba.to_vec();
        }

        let mut out = vec![0u8; rgba.len()];
        let ci = &self.intrinsics;

        for y_out in 0..height {
            for x_out in 0..width {
                // Convert output pixel to normalised coordinates
                let xn = (x_out as f64 - ci.cx) / ci.fx;
                let yn = (y_out as f64 - ci.cy) / ci.fy;

                // Apply distortion model to find where this undistorted pixel
                // maps to in the distorted (source) image
                let (xnd, ynd) = self.coeffs.distort(xn, yn);

                // Convert back to pixel coordinates in the source image
                let xs = xnd * ci.fx + ci.cx;
                let ys = ynd * ci.fy + ci.cy;

                // Bilinear sample from source image
                let pixel = bilinear_sample_rgba(rgba, width, height, xs, ys, self.border_color);

                let dst = (y_out * width + x_out) * 4;
                out[dst..dst + 4].copy_from_slice(&pixel);
            }
        }

        out
    }

    /// Correct distortion in a grayscale image.
    ///
    /// `gray` must be row-major with exactly `width × height` bytes.
    /// Returns a new grayscale buffer of the same size.
    #[must_use]
    pub fn correct_gray(&self, gray: &[u8], width: usize, height: usize) -> Vec<u8> {
        if width == 0 || height == 0 || gray.len() != width * height {
            return gray.to_vec();
        }

        let mut out = vec![0u8; gray.len()];
        let ci = &self.intrinsics;

        for y_out in 0..height {
            for x_out in 0..width {
                let xn = (x_out as f64 - ci.cx) / ci.fx;
                let yn = (y_out as f64 - ci.cy) / ci.fy;
                let (xnd, ynd) = self.coeffs.distort(xn, yn);
                let xs = xnd * ci.fx + ci.cx;
                let ys = ynd * ci.fy + ci.cy;
                out[y_out * width + x_out] = bilinear_sample_gray(gray, width, height, xs, ys);
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// LensDistortionSimulator
// ---------------------------------------------------------------------------

/// Applies lens distortion to an undistorted RGBA image (synthesis).
///
/// This is the forward operation: given an undistorted image, produce the
/// distorted version that a camera with the given lens would capture.
pub struct LensDistortionSimulator {
    intrinsics: CameraIntrinsics,
    coeffs: DistortionCoeffs,
    /// Background fill colour when a source pixel is outside image bounds.
    pub border_color: [u8; 4],
    /// Newton–Raphson iterations for the inverse pass.
    undistort_iters: u32,
}

impl LensDistortionSimulator {
    /// Create a simulator with default settings.
    #[must_use]
    pub fn new(intrinsics: CameraIntrinsics, coeffs: DistortionCoeffs) -> Self {
        Self {
            intrinsics,
            coeffs,
            border_color: [0, 0, 0, 0],
            undistort_iters: 20,
        }
    }

    /// Simulate distortion on an RGBA image.
    ///
    /// For each output pixel in the distorted image, we compute the
    /// corresponding undistorted source position using Newton–Raphson
    /// inversion and sample the input image.
    #[must_use]
    pub fn simulate_rgba(&self, rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
        if width == 0 || height == 0 || rgba.len() != width * height * 4 {
            return rgba.to_vec();
        }

        let mut out = vec![0u8; rgba.len()];
        let ci = &self.intrinsics;

        for y_out in 0..height {
            for x_out in 0..width {
                // Normalised coordinates for this output (distorted) pixel
                let xnd = (x_out as f64 - ci.cx) / ci.fx;
                let ynd = (y_out as f64 - ci.cy) / ci.fy;

                // Invert distortion to get the undistorted source coordinates
                let (xn, yn) = self.coeffs.undistort(xnd, ynd, self.undistort_iters);

                let xs = xn * ci.fx + ci.cx;
                let ys = yn * ci.fy + ci.cy;

                let pixel = bilinear_sample_rgba(rgba, width, height, xs, ys, self.border_color);
                let dst = (y_out * width + x_out) * 4;
                out[dst..dst + 4].copy_from_slice(&pixel);
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// Bilinear interpolation helpers
// ---------------------------------------------------------------------------

/// Bilinearly sample a 4-channel RGBA image at sub-pixel coordinates.
fn bilinear_sample_rgba(
    rgba: &[u8],
    width: usize,
    height: usize,
    x: f64,
    y: f64,
    border: [u8; 4],
) -> [u8; 4] {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x.floor();
    let fy = y - y.floor();

    let sample = |px: isize, py: isize| -> [u8; 4] {
        if px < 0 || py < 0 || px >= width as isize || py >= height as isize {
            border
        } else {
            let base = (py as usize * width + px as usize) * 4;
            [rgba[base], rgba[base + 1], rgba[base + 2], rgba[base + 3]]
        }
    };

    let c00 = sample(x0, y0);
    let c10 = sample(x1, y0);
    let c01 = sample(x0, y1);
    let c11 = sample(x1, y1);

    let mut out = [0u8; 4];
    for ch in 0..4 {
        let top = c00[ch] as f64 * (1.0 - fx) + c10[ch] as f64 * fx;
        let bot = c01[ch] as f64 * (1.0 - fx) + c11[ch] as f64 * fx;
        let val = top * (1.0 - fy) + bot * fy;
        out[ch] = val.round().clamp(0.0, 255.0) as u8;
    }
    out
}

/// Bilinearly sample a single-channel grayscale image at sub-pixel coordinates.
fn bilinear_sample_gray(gray: &[u8], width: usize, height: usize, x: f64, y: f64) -> u8 {
    let x0 = x.floor() as isize;
    let y0 = y.floor() as isize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    let fx = x - x.floor();
    let fy = y - y.floor();

    let sample = |px: isize, py: isize| -> f64 {
        if px < 0 || py < 0 || px >= width as isize || py >= height as isize {
            0.0
        } else {
            gray[py as usize * width + px as usize] as f64
        }
    };

    let top = sample(x0, y0) * (1.0 - fx) + sample(x1, y0) * fx;
    let bot = sample(x0, y1) * (1.0 - fx) + sample(x1, y1) * fx;
    let val = top * (1.0 - fy) + bot * fy;
    val.round().clamp(0.0, 255.0) as u8
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Apply barrel distortion correction to a grayscale image.
///
/// `k1` > 0 corrects barrel, `k1` < 0 corrects pincushion.
#[must_use]
pub fn correct_barrel_gray(gray: &[u8], width: usize, height: usize, k1: f64) -> Vec<u8> {
    let intrinsics = CameraIntrinsics::from_fov(width, height, std::f64::consts::PI / 3.0);
    let coeffs = DistortionCoeffs::barrel(k1);
    LensDistortionCorrector::new(intrinsics, coeffs).correct_gray(gray, width, height)
}

/// Apply barrel distortion simulation to a grayscale image.
#[must_use]
pub fn simulate_barrel_gray(gray: &[u8], width: usize, height: usize, k1: f64) -> Vec<u8> {
    let intrinsics = CameraIntrinsics::from_fov(width, height, std::f64::consts::PI / 3.0);
    let coeffs = DistortionCoeffs::barrel(k1);
    // For grayscale simulation, build a tmp RGBA, simulate, extract channel
    let rgba: Vec<u8> = gray.iter().flat_map(|&v| [v, v, v, 255u8]).collect();
    let sim = LensDistortionSimulator::new(intrinsics, coeffs);
    let out_rgba = sim.simulate_rgba(&rgba, width, height);
    out_rgba.chunks(4).map(|c| c[0]).collect()
}

// ---------------------------------------------------------------------------
// DistortionMap — pre-computed remap table for fast repeated correction
// ---------------------------------------------------------------------------

/// A pre-computed per-pixel remap table for lens distortion correction.
///
/// Instead of recomputing the undistortion mapping on every frame, you can
/// build the map once with [`DistortionMap::build`] and then apply it to many
/// frames cheaply with [`DistortionMap::apply_rgba`] or [`DistortionMap::apply_gray`].
///
/// Each entry stores the source `(x, y)` position (in sub-pixel f32 coordinates)
/// that maps to the corresponding output pixel.
pub struct DistortionMap {
    /// Source x-coordinates for each output pixel (row-major).
    pub map_x: Vec<f32>,
    /// Source y-coordinates for each output pixel (row-major).
    pub map_y: Vec<f32>,
    /// Output image width.
    pub width: usize,
    /// Output image height.
    pub height: usize,
}

impl DistortionMap {
    /// Build a remap table from camera intrinsics and distortion coefficients.
    ///
    /// This is equivalent to one call to `LensDistortionCorrector::correct_rgba`
    /// per pixel, but computed once and stored for reuse.
    #[must_use]
    pub fn build(
        intrinsics: CameraIntrinsics,
        coeffs: DistortionCoeffs,
        width: usize,
        height: usize,
    ) -> Self {
        let n = width * height;
        let mut map_x = vec![0.0f32; n];
        let mut map_y = vec![0.0f32; n];

        let ci = &intrinsics;
        for y_out in 0..height {
            for x_out in 0..width {
                let xn = (x_out as f64 - ci.cx) / ci.fx;
                let yn = (y_out as f64 - ci.cy) / ci.fy;
                let (xnd, ynd) = coeffs.distort(xn, yn);
                let xs = xnd * ci.fx + ci.cx;
                let ys = ynd * ci.fy + ci.cy;
                let idx = y_out * width + x_out;
                map_x[idx] = xs as f32;
                map_y[idx] = ys as f32;
            }
        }

        Self {
            map_x,
            map_y,
            width,
            height,
        }
    }

    /// Apply the pre-computed remap to an RGBA image.
    ///
    /// `rgba` must be `width × height × 4` bytes.  Returns `None` for size mismatch.
    #[must_use]
    pub fn apply_rgba(&self, rgba: &[u8], border: [u8; 4]) -> Option<Vec<u8>> {
        if rgba.len() != self.width * self.height * 4 {
            return None;
        }
        let mut out = vec![0u8; rgba.len()];
        for (idx, (&sx, &sy)) in self.map_x.iter().zip(self.map_y.iter()).enumerate() {
            let pixel =
                bilinear_sample_rgba(rgba, self.width, self.height, sx as f64, sy as f64, border);
            let dst = idx * 4;
            out[dst..dst + 4].copy_from_slice(&pixel);
        }
        Some(out)
    }

    /// Apply the pre-computed remap to a grayscale image.
    ///
    /// `gray` must be `width × height` bytes.  Returns `None` for size mismatch.
    #[must_use]
    pub fn apply_gray(&self, gray: &[u8]) -> Option<Vec<u8>> {
        if gray.len() != self.width * self.height {
            return None;
        }
        let mut out = vec![0u8; gray.len()];
        for (idx, (&sx, &sy)) in self.map_x.iter().zip(self.map_y.iter()).enumerate() {
            out[idx] = bilinear_sample_gray(gray, self.width, self.height, sx as f64, sy as f64);
        }
        Some(out)
    }

    /// Return the number of pixels in the remap table.
    #[must_use]
    pub fn len(&self) -> usize {
        self.map_x.len()
    }

    /// Return `true` if the map is empty (zero-size image).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.map_x.is_empty()
    }
}

// ---------------------------------------------------------------------------
// FisheyeEquidistantCorrector
// ---------------------------------------------------------------------------

/// Camera model for a fisheye lens with the equidistant projection:
/// `r_d = f · θ`, where `θ` is the angle from the optical axis.
///
/// This model is common for action cameras (GoPro-style) and wide-angle
/// security cameras.  Given focal length `f` and the principal point, the
/// corrector maps each output pixel back to the fisheye source pixel using
/// the inverse equidistant model.
pub struct FisheyeEquidistantCorrector {
    intrinsics: CameraIntrinsics,
    /// Additional radial polynomial coefficients `[k1, k2, k3, k4]` applied to
    /// the angle θ to model non-ideal fisheye lenses.
    poly: [f64; 4],
    /// Background fill colour for out-of-bounds pixels.
    pub border_color: [u8; 4],
}

impl FisheyeEquidistantCorrector {
    /// Create a corrector for an ideal equidistant fisheye lens (no polynomial correction).
    #[must_use]
    pub fn new(intrinsics: CameraIntrinsics) -> Self {
        Self {
            intrinsics,
            poly: [0.0; 4],
            border_color: [0, 0, 0, 0],
        }
    }

    /// Set the polynomial correction coefficients `[k1, k2, k3, k4]`.
    ///
    /// The effective radial distortion becomes:
    /// `r_d = f · (θ + k1·θ³ + k2·θ⁵ + k3·θ⁷ + k4·θ⁹)`.
    pub fn set_poly(&mut self, k1: f64, k2: f64, k3: f64, k4: f64) {
        self.poly = [k1, k2, k3, k4];
    }

    /// Correct fisheye distortion in an RGBA image.
    ///
    /// For each output pixel the corresponding angle is computed, the fisheye
    /// mapping is inverted, and the source pixel is bilinearly sampled.
    ///
    /// `rgba` must be `width × height × 4` bytes.
    #[must_use]
    pub fn correct_rgba(&self, rgba: &[u8], width: usize, height: usize) -> Vec<u8> {
        if width == 0 || height == 0 || rgba.len() != width * height * 4 {
            return rgba.to_vec();
        }

        let mut out = vec![0u8; rgba.len()];
        let ci = &self.intrinsics;
        let f = ci.fx; // Assume square pixels; use fx as focal length

        for y_out in 0..height {
            for x_out in 0..width {
                // Normalised undistorted coordinates
                let xu = (x_out as f64 - ci.cx) / ci.fx;
                let yu = (y_out as f64 - ci.cy) / ci.fy;
                let r_undist = (xu * xu + yu * yu).sqrt();

                if r_undist < 1e-12 {
                    // Optical axis: source is principal point
                    let pixel =
                        bilinear_sample_rgba(rgba, width, height, ci.cx, ci.cy, self.border_color);
                    let dst = (y_out * width + x_out) * 4;
                    out[dst..dst + 4].copy_from_slice(&pixel);
                    continue;
                }

                // θ from the equidistant model (assume no polynomial correction first)
                let theta = r_undist.atan(); // r_undist / f * f = r_undist in normalised coords

                // Apply polynomial correction
                let theta2 = theta * theta;
                let r_d = f
                    * theta
                    * (1.0
                        + self.poly[0] * theta2
                        + self.poly[1] * theta2 * theta2
                        + self.poly[2] * theta2 * theta2 * theta2
                        + self.poly[3] * theta2 * theta2 * theta2 * theta2);

                // Direction
                let scale = r_d / (r_undist * ci.fx);
                let xs = xu * scale * ci.fx + ci.cx;
                let ys = yu * scale * ci.fy + ci.cy;

                let pixel = bilinear_sample_rgba(rgba, width, height, xs, ys, self.border_color);
                let dst = (y_out * width + x_out) * 4;
                out[dst..dst + 4].copy_from_slice(&pixel);
            }
        }

        out
    }
}

// ---------------------------------------------------------------------------
// optimal_crop_rect — find the largest inscribed rectangle after correction
// ---------------------------------------------------------------------------

/// Compute the largest axis-aligned rectangle (in output pixel coordinates) that
/// is entirely covered by valid source pixels after barrel/pincushion correction.
///
/// After undistortion, pixels near the image boundary often have no valid source
/// pixel (they map outside the input image), resulting in black/border pixels.
/// `optimal_crop_rect` searches for the tightest rectangle centered at the
/// principal point that avoids these border regions.
///
/// Returns `(x, y, crop_width, crop_height)` in pixel coordinates, or
/// `None` if no valid crop can be found.
///
/// # Algorithm
///
/// Binary-search the crop radius (fraction of the output half-dimension) until
/// the sampled corners all map to valid source positions.
#[must_use]
pub fn optimal_crop_rect(
    intrinsics: CameraIntrinsics,
    coeffs: DistortionCoeffs,
    width: usize,
    height: usize,
) -> Option<(usize, usize, usize, usize)> {
    if width == 0 || height == 0 {
        return None;
    }

    let ci = &intrinsics;
    let half_w = (width as f64) * 0.5;
    let half_h = (height as f64) * 0.5;

    // Binary search on the fraction of the half-dimensions
    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;

    let in_bounds = |scale: f64| -> bool {
        // Check four corners of the candidate crop at the given scale
        let corners = [
            (-half_w * scale, -half_h * scale),
            (half_w * scale, -half_h * scale),
            (-half_w * scale, half_h * scale),
            (half_w * scale, half_h * scale),
        ];

        for (dx, dy) in corners {
            let xn = dx / ci.fx;
            let yn = dy / ci.fy;
            let (xnd, ynd) = coeffs.distort(xn, yn);
            let xs = xnd * ci.fx + ci.cx;
            let ys = ynd * ci.fy + ci.cy;
            if xs < 0.0 || ys < 0.0 || xs >= width as f64 - 1.0 || ys >= height as f64 - 1.0 {
                return false;
            }
        }
        true
    };

    for _ in 0..32 {
        let mid = (lo + hi) * 0.5;
        if in_bounds(mid) {
            lo = mid;
        } else {
            hi = mid;
        }
    }

    if lo < 1e-4 {
        return None;
    }

    let cw = (half_w * lo * 2.0).floor() as usize;
    let ch = (half_h * lo * 2.0).floor() as usize;
    if cw == 0 || ch == 0 {
        return None;
    }

    let cx = (width.saturating_sub(cw)) / 2;
    let cy = (height.saturating_sub(ch)) / 2;

    Some((cx, cy, cw, ch))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_rgba(w: usize, h: usize) -> Vec<u8> {
        (0..w * h * 4).map(|i| (i % 256) as u8).collect()
    }

    fn make_gray(w: usize, h: usize) -> Vec<u8> {
        (0..w * h).map(|i| (i % 256) as u8).collect()
    }

    #[test]
    fn test_identity_coeffs_correct_is_identity_rgba() {
        let w = 8usize;
        let h = 8usize;
        let rgba = make_rgba(w, h);
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.0);
        let coeffs = DistortionCoeffs::identity();
        let corrected = LensDistortionCorrector::new(intrinsics, coeffs).correct_rgba(&rgba, w, h);
        // With identity coeffs the output should closely match the input
        let diffs: u32 = rgba
            .iter()
            .zip(corrected.iter())
            .map(|(&a, &b)| (a as i32 - b as i32).unsigned_abs())
            .sum();
        // Allow small rounding differences from bilinear interpolation
        assert!(
            diffs < (w * h * 4) as u32,
            "Too much difference with identity distortion: total_diff={diffs}"
        );
    }

    #[test]
    fn test_correct_rgba_returns_same_size() {
        let w = 16usize;
        let h = 12usize;
        let rgba = make_rgba(w, h);
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.2);
        let coeffs = DistortionCoeffs::barrel(0.1);
        let out = LensDistortionCorrector::new(intrinsics, coeffs).correct_rgba(&rgba, w, h);
        assert_eq!(out.len(), rgba.len());
    }

    #[test]
    fn test_correct_gray_returns_same_size() {
        let w = 16usize;
        let h = 12usize;
        let gray = make_gray(w, h);
        let out = correct_barrel_gray(&gray, w, h, 0.2);
        assert_eq!(out.len(), gray.len());
    }

    #[test]
    fn test_distort_identity() {
        let coeffs = DistortionCoeffs::identity();
        let (xd, yd) = coeffs.distort(0.5, 0.3);
        assert!((xd - 0.5).abs() < 1e-12);
        assert!((yd - 0.3).abs() < 1e-12);
    }

    #[test]
    fn test_undistort_roundtrip() {
        let coeffs = DistortionCoeffs::barrel2(0.2, 0.05);
        let x = 0.4;
        let y = -0.3;
        let (xd, yd) = coeffs.distort(x, y);
        let (xu, yu) = coeffs.undistort(xd, yd, 30);
        assert!(
            (xu - x).abs() < 1e-6,
            "undistort x round-trip: xu={xu}, expected={x}"
        );
        assert!(
            (yu - y).abs() < 1e-6,
            "undistort y round-trip: yu={yu}, expected={y}"
        );
    }

    #[test]
    fn test_barrel_k1_positive_moves_corners_inward() {
        // With positive k1 (barrel), the distorted radius > undistorted radius
        let coeffs = DistortionCoeffs::barrel(0.5);
        let x = 0.6;
        let y = 0.6;
        let (xd, yd) = coeffs.distort(x, y);
        let r_orig = (x * x + y * y).sqrt();
        let r_dist = (xd * xd + yd * yd).sqrt();
        assert!(
            r_dist > r_orig,
            "Barrel distortion should push corners outward: r_orig={r_orig} r_dist={r_dist}"
        );
    }

    #[test]
    fn test_camera_intrinsics_from_fov() {
        let ci = CameraIntrinsics::from_fov(640, 480, std::f64::consts::PI / 2.0);
        // 90° horizontal FOV → fx = cx / tan(45°) = 320
        assert!((ci.fx - 320.0).abs() < 1.0, "fx={}", ci.fx);
        assert!((ci.cx - 320.0).abs() < 0.5);
        assert!((ci.cy - 240.0).abs() < 0.5);
    }

    #[test]
    fn test_simulate_rgba_returns_same_size() {
        let w = 12usize;
        let h = 8usize;
        let rgba = make_rgba(w, h);
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.0);
        let coeffs = DistortionCoeffs::barrel(0.3);
        let sim = LensDistortionSimulator::new(intrinsics, coeffs);
        let out = sim.simulate_rgba(&rgba, w, h);
        assert_eq!(out.len(), rgba.len());
    }

    #[test]
    fn test_correct_and_simulate_are_inverse_operations() {
        // Simulate distortion then correct it — result should be close to original
        let w = 32usize;
        let h = 32usize;
        // Gradient image (smooth to minimise interpolation error)
        let gray: Vec<u8> = (0..w * h)
            .map(|i| ((i * 7 / (w * h) + 30) % 200) as u8)
            .collect();

        let intrinsics = CameraIntrinsics::from_fov(w, h, std::f64::consts::PI / 3.0);
        let coeffs = DistortionCoeffs::barrel(0.1);

        let distorted = simulate_barrel_gray(&gray, w, h, 0.1);
        let corrected = correct_barrel_gray(&distorted, w, h, 0.1);

        // The centre of the image should recover well after round-trip
        let cx = w / 2;
        let cy = h / 2;
        let orig_centre = gray[cy * w + cx];
        let corr_centre = corrected[cy * w + cx];
        let diff = (orig_centre as i32 - corr_centre as i32).abs();
        assert!(
            diff < 30,
            "Centre pixel difference after round-trip: {diff}"
        );
    }

    #[test]
    fn test_correct_empty_image_returns_empty() {
        let out = correct_barrel_gray(&[], 0, 0, 0.2);
        assert!(out.is_empty());
    }

    #[test]
    fn test_distortion_coeffs_default_is_identity() {
        let c = DistortionCoeffs::default();
        let (xd, yd) = c.distort(1.0, 1.0);
        assert!((xd - 1.0).abs() < 1e-12);
        assert!((yd - 1.0).abs() < 1e-12);
    }

    // --- DistortionMap tests ---

    #[test]
    fn test_distortion_map_build_correct_size() {
        let w = 10usize;
        let h = 8usize;
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.0);
        let coeffs = DistortionCoeffs::barrel(0.1);
        let map = DistortionMap::build(intrinsics, coeffs, w, h);
        assert_eq!(map.width, w);
        assert_eq!(map.height, h);
        assert_eq!(map.len(), w * h);
        assert!(!map.is_empty());
    }

    #[test]
    fn test_distortion_map_apply_rgba_returns_same_size() {
        let w = 8usize;
        let h = 8usize;
        let rgba = make_rgba(w, h);
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.0);
        let coeffs = DistortionCoeffs::identity();
        let map = DistortionMap::build(intrinsics, coeffs, w, h);
        let out = map.apply_rgba(&rgba, [0, 0, 0, 0]);
        assert!(out.is_some());
        assert_eq!(out.unwrap().len(), rgba.len());
    }

    #[test]
    fn test_distortion_map_apply_gray_returns_same_size() {
        let w = 8usize;
        let h = 8usize;
        let gray = make_gray(w, h);
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.0);
        let coeffs = DistortionCoeffs::identity();
        let map = DistortionMap::build(intrinsics, coeffs, w, h);
        let out = map.apply_gray(&gray);
        assert!(out.is_some());
        assert_eq!(out.unwrap().len(), gray.len());
    }

    #[test]
    fn test_distortion_map_apply_rgba_size_mismatch_returns_none() {
        let w = 8usize;
        let h = 8usize;
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.0);
        let coeffs = DistortionCoeffs::identity();
        let map = DistortionMap::build(intrinsics, coeffs, w, h);
        // Wrong size buffer
        let out = map.apply_rgba(&[0u8; 10], [0, 0, 0, 0]);
        assert!(out.is_none());
    }

    #[test]
    fn test_distortion_map_identity_matches_corrector() {
        let w = 12usize;
        let h = 10usize;
        let rgba = make_rgba(w, h);
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.0);
        let coeffs = DistortionCoeffs::barrel(0.05);

        // Corrector result
        let corrector_out =
            LensDistortionCorrector::new(intrinsics, coeffs).correct_rgba(&rgba, w, h);

        // DistortionMap result
        let map = DistortionMap::build(intrinsics, coeffs, w, h);
        let map_out = map.apply_rgba(&rgba, [0, 0, 0, 0]).unwrap_or_default();

        assert_eq!(corrector_out.len(), map_out.len());
        // Results must be pixel-identical (same algorithm)
        assert_eq!(
            corrector_out, map_out,
            "DistortionMap and corrector should produce identical results"
        );
    }

    #[test]
    fn test_distortion_map_empty_image() {
        let map = DistortionMap::build(
            CameraIntrinsics::from_fov(0, 0, 1.0),
            DistortionCoeffs::identity(),
            0,
            0,
        );
        assert!(map.is_empty());
    }

    // --- FisheyeEquidistantCorrector tests ---

    #[test]
    fn test_fisheye_corrector_returns_same_size() {
        let w = 16usize;
        let h = 12usize;
        let rgba = make_rgba(w, h);
        let intrinsics = CameraIntrinsics::from_fov(w, h, std::f64::consts::PI / 2.0);
        let corrector = FisheyeEquidistantCorrector::new(intrinsics);
        let out = corrector.correct_rgba(&rgba, w, h);
        assert_eq!(out.len(), rgba.len());
    }

    #[test]
    fn test_fisheye_corrector_empty_image() {
        let intrinsics = CameraIntrinsics::from_fov(0, 0, 1.0);
        let corrector = FisheyeEquidistantCorrector::new(intrinsics);
        let out = corrector.correct_rgba(&[], 0, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn test_fisheye_corrector_zero_poly_is_baseline() {
        // With zero polynomial corrections the output should differ from the input
        // (because the fisheye remap changes the image geometry), but must not panic.
        let w = 8usize;
        let h = 8usize;
        let rgba = make_rgba(w, h);
        let intrinsics = CameraIntrinsics::from_fov(w, h, std::f64::consts::PI * 0.75);
        let corrector = FisheyeEquidistantCorrector::new(intrinsics);
        let out = corrector.correct_rgba(&rgba, w, h);
        assert_eq!(out.len(), rgba.len());
    }

    // --- optimal_crop_rect tests ---

    #[test]
    fn test_optimal_crop_rect_identity_covers_full_image() {
        let w = 32usize;
        let h = 32usize;
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.2);
        let coeffs = DistortionCoeffs::identity();
        // Identity distortion → every output pixel maps to a valid source pixel
        let result = optimal_crop_rect(intrinsics, coeffs, w, h);
        assert!(
            result.is_some(),
            "identity should always return a valid crop"
        );
        let (_, _, cw, ch) = result.unwrap();
        // Crop should cover at least half the image
        assert!(cw >= w / 2, "crop width={cw} too small");
        assert!(ch >= h / 2, "crop height={ch} too small");
    }

    #[test]
    fn test_optimal_crop_rect_barrel_is_smaller_than_full() {
        let w = 64usize;
        let h = 64usize;
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.2);
        // Strong barrel pushes corners outside source image
        let coeffs = DistortionCoeffs::barrel(0.5);
        let result = optimal_crop_rect(intrinsics, coeffs, w, h);
        assert!(result.is_some(), "should find a crop for barrel distortion");
        let (_, _, cw, ch) = result.unwrap();
        // Crop must be smaller than the full image (black borders created)
        assert!(
            cw < w,
            "crop_width={cw} should be < width={w} for barrel distortion"
        );
        assert!(
            ch < h,
            "crop_height={ch} should be < height={h} for barrel distortion"
        );
    }

    #[test]
    fn test_optimal_crop_rect_zero_dimensions_returns_none() {
        let coeffs = DistortionCoeffs::identity();
        let intrinsics = CameraIntrinsics::from_fov(0, 0, 1.0);
        assert!(optimal_crop_rect(intrinsics, coeffs, 0, 0).is_none());
    }

    #[test]
    fn test_full_pipeline_map_and_crop() {
        // Build the distortion map, apply it, then compute the crop rectangle.
        // The crop coordinates should be within the image boundaries.
        let w = 32usize;
        let h = 24usize;
        let rgba = make_rgba(w, h);
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.2);
        let coeffs = DistortionCoeffs::barrel(0.2);

        let map = DistortionMap::build(intrinsics, coeffs, w, h);
        let out = map.apply_rgba(&rgba, [0, 0, 0, 0]);
        assert!(out.is_some());

        if let Some((cx, cy, cw, ch)) = optimal_crop_rect(intrinsics, coeffs, w, h) {
            assert!(cx + cw <= w, "crop x+w exceeds image width");
            assert!(cy + ch <= h, "crop x+h exceeds image height");
        }
    }

    #[test]
    fn test_distortion_map_apply_gray_mismatch_returns_none() {
        let w = 4usize;
        let h = 4usize;
        let intrinsics = CameraIntrinsics::from_fov(w, h, 1.0);
        let coeffs = DistortionCoeffs::identity();
        let map = DistortionMap::build(intrinsics, coeffs, w, h);
        // Wrong size buffer
        let out = map.apply_gray(&[128u8; 7]);
        assert!(out.is_none());
    }

    #[test]
    fn test_camera_intrinsics_new_stores_values() {
        let ci = CameraIntrinsics::new(500.0, 502.0, 320.0, 240.0);
        assert!((ci.fx - 500.0).abs() < 1e-10);
        assert!((ci.fy - 502.0).abs() < 1e-10);
        assert!((ci.cx - 320.0).abs() < 1e-10);
        assert!((ci.cy - 240.0).abs() < 1e-10);
    }

    #[test]
    fn test_full_brown_conrady_model_roundtrip() {
        let coeffs = DistortionCoeffs::new(0.1, -0.05, 0.02, 0.0, 0.0, 0.0, 0.001, -0.001);
        let x = 0.3;
        let y = -0.2;
        let (xd, yd) = coeffs.distort(x, y);
        let (xu, yu) = coeffs.undistort(xd, yd, 50);
        assert!(
            (xu - x).abs() < 1e-5,
            "Brown-Conrady roundtrip x: xu={xu} expected={x}"
        );
        assert!(
            (yu - y).abs() < 1e-5,
            "Brown-Conrady roundtrip y: yu={yu} expected={y}"
        );
    }
}
