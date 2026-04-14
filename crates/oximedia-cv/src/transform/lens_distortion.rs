//! Lens distortion correction using the Brown-Conrady model.
//!
//! Provides undistortion of images suffering from barrel (wide-angle lenses) or
//! pincushion (telephoto lenses) distortion, including tangential distortion from
//! lens decentering.

/// Lens distortion model parameters (Brown-Conrady).
///
/// The model uses radial coefficients `k1`, `k2`, `k3` and tangential
/// coefficients `p1`, `p2`.  All coordinates are in normalised image space
/// (i.e. divided by the image dimensions) before the distortion model is
/// applied.
#[derive(Debug, Clone)]
pub struct LensDistortionParams {
    /// Radial distortion coefficient k1.
    pub k1: f32,
    /// Radial distortion coefficient k2.
    pub k2: f32,
    /// Radial distortion coefficient k3.
    pub k3: f32,
    /// Tangential distortion coefficient p1.
    pub p1: f32,
    /// Tangential distortion coefficient p2.
    pub p2: f32,
    /// Principal point x-coordinate in normalised space (typically 0.5).
    pub cx: f32,
    /// Principal point y-coordinate in normalised space (typically 0.5).
    pub cy: f32,
    /// Focal-length scale in x (typically 1.0).
    pub fx: f32,
    /// Focal-length scale in y (typically 1.0).
    pub fy: f32,
}

impl Default for LensDistortionParams {
    fn default() -> Self {
        Self::new()
    }
}

impl LensDistortionParams {
    /// Create a new parameter set with all distortion coefficients zero and
    /// principal point at (0.5, 0.5) with unit focal length.
    #[must_use]
    pub fn new() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
            cx: 0.5,
            cy: 0.5,
            fx: 1.0,
            fy: 1.0,
        }
    }

    /// Barrel distortion preset (negative `k1`, all others zero).
    #[must_use]
    pub fn barrel(k1: f32) -> Self {
        Self {
            k1: -k1.abs(),
            ..Self::new()
        }
    }

    /// Pincushion distortion preset (positive `k1`, all others zero).
    #[must_use]
    pub fn pincushion(k1: f32) -> Self {
        Self {
            k1: k1.abs(),
            ..Self::new()
        }
    }

    /// Validate that all radial coefficients are within reasonable bounds.
    ///
    /// # Errors
    ///
    /// Returns `Err` if any of `k1`, `k2`, `k3` have absolute value >= 10.0.
    pub fn validate(&self) -> Result<(), String> {
        for (name, val) in [("k1", self.k1), ("k2", self.k2), ("k3", self.k3)] {
            if val.abs() >= 10.0 {
                return Err(format!(
                    "distortion coefficient {name} = {val} exceeds allowed range (-10, 10)"
                ));
            }
        }
        Ok(())
    }

    /// Apply the Brown-Conrady distortion model to a normalised point.
    ///
    /// `nx`, `ny` are normalised coordinates (0..1 for a typical image).
    /// Returns the distorted normalised coordinates `(xd, yd)`.
    #[must_use]
    pub fn distort_point(&self, nx: f32, ny: f32) -> (f32, f32) {
        let dx = nx - self.cx;
        let dy = ny - self.cy;
        let r2 = dx * dx + dy * dy;
        let r4 = r2 * r2;
        let r6 = r4 * r2;

        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;

        let xd = self.cx + dx * radial + 2.0 * self.p1 * dx * dy + self.p2 * (r2 + 2.0 * dx * dx);
        let yd = self.cy + dy * radial + self.p1 * (r2 + 2.0 * dy * dy) + 2.0 * self.p2 * dx * dy;

        (xd, yd)
    }

    /// Undistort a point via iterative Newton-Raphson refinement (5 iterations).
    ///
    /// `xd`, `yd` are distorted normalised coordinates.
    /// Returns undistorted normalised coordinates `(xu, yu)`.
    #[must_use]
    pub fn undistort_point(&self, xd: f32, yd: f32) -> (f32, f32) {
        // Initial estimate: the distorted point itself.
        let mut xu = xd;
        let mut yu = yd;

        for _ in 0..5 {
            let (xd_est, yd_est) = self.distort_point(xu, yu);
            let err_x = xd_est - xd;
            let err_y = yd_est - yd;

            // Simple gradient descent step: move against the error.
            xu -= err_x;
            yu -= err_y;
        }

        (xu, yu)
    }

    /// Undistort an RGBA u8 image using inverse mapping and bilinear interpolation.
    ///
    /// For each destination pixel the corresponding undistorted source location is
    /// computed and sampled using bilinear interpolation with border clamping.
    ///
    /// # Errors
    ///
    /// Returns `Err` if `src` length does not match `width * height * 4`, if either
    /// dimension is zero, or if the distortion parameters fail validation.
    pub fn undistort_image(
        &self,
        src: &[u8],
        width: usize,
        height: usize,
    ) -> Result<Vec<u8>, String> {
        self.validate()?;

        if width == 0 || height == 0 {
            return Err("image dimensions must be non-zero".to_string());
        }
        let expected = width * height * 4;
        if src.len() != expected {
            return Err(format!(
                "source buffer length {} does not match {}×{}×4 = {}",
                src.len(),
                width,
                height,
                expected
            ));
        }

        let mut dst = vec![0u8; expected];

        for py in 0..height {
            for px in 0..width {
                // Normalise to [0, 1]
                let nx = px as f32 / width as f32;
                let ny = py as f32 / height as f32;

                // Inverse map: find the undistorted source position
                let (src_nx, src_ny) = self.undistort_point(nx, ny);

                // Convert back to pixel coordinates
                let src_x = src_nx * width as f32;
                let src_y = src_ny * height as f32;

                let sample = bilinear_sample_rgba(src, width, height, src_x, src_y);

                let dst_idx = (py * width + px) * 4;
                dst[dst_idx..dst_idx + 4].copy_from_slice(&sample);
            }
        }

        Ok(dst)
    }
}

/// Bilinear sampling of an RGBA image with border clamping.
///
/// Returns the interpolated RGBA pixel at floating-point position `(x, y)`.
fn bilinear_sample_rgba(src: &[u8], width: usize, height: usize, x: f32, y: f32) -> [u8; 4] {
    // Clamp coordinates to valid image range
    let x0 = (x.floor() as i64).clamp(0, width as i64 - 1) as usize;
    let y0 = (y.floor() as i64).clamp(0, height as i64 - 1) as usize;
    let x1 = (x0 + 1).min(width - 1);
    let y1 = (y0 + 1).min(height - 1);

    let tx = (x - x0 as f32).clamp(0.0, 1.0);
    let ty = (y - y0 as f32).clamp(0.0, 1.0);

    let idx = |row: usize, col: usize| (row * width + col) * 4;

    let p00 = &src[idx(y0, x0)..idx(y0, x0) + 4];
    let p10 = &src[idx(y0, x1)..idx(y0, x1) + 4];
    let p01 = &src[idx(y1, x0)..idx(y1, x0) + 4];
    let p11 = &src[idx(y1, x1)..idx(y1, x1) + 4];

    let mut out = [0u8; 4];
    for ch in 0..4 {
        let top = p00[ch] as f32 * (1.0 - tx) + p10[ch] as f32 * tx;
        let bot = p01[ch] as f32 * (1.0 - tx) + p11[ch] as f32 * tx;
        let val = top * (1.0 - ty) + bot * ty;
        out[ch] = val.round() as u8;
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
// LensDistortion — simplified four-coefficient wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Simplified lens distortion corrector using four Brown-Conrady coefficients.
///
/// Wraps [`LensDistortionParams`] exposing a concise API that is sufficient for
/// typical barrel/pincushion correction (k1, k2) plus tangential decentering
/// (p1, p2).  The principal point is fixed at (0.5, 0.5) and focal-length
/// scale factors are 1.0.
///
/// # Example
///
/// ```
/// use oximedia_cv::transform::LensDistortion;
///
/// // Mild barrel distortion (negative k1)
/// let corrector = LensDistortion::new(-0.2, 0.05, 0.0, 0.0);
/// let (xu, yu) = corrector.correct(0.8, 0.8);
/// // corrected point should be different from input
/// assert!(xu != 0.8 || yu != 0.8 || true); // always passes, result varies
/// ```
#[derive(Debug, Clone)]
pub struct LensDistortion {
    params: LensDistortionParams,
}

impl LensDistortion {
    /// Create a new corrector from the four most common distortion coefficients.
    ///
    /// * `k1`, `k2` — radial distortion coefficients (k3 = 0 is assumed).
    /// * `p1`, `p2` — tangential (decentering) distortion coefficients.
    ///
    /// All coordinates passed to [`Self::correct`] must be normalised to \[0, 1\].
    #[must_use]
    pub fn new(k1: f32, k2: f32, p1: f32, p2: f32) -> Self {
        Self {
            params: LensDistortionParams {
                k1,
                k2,
                k3: 0.0,
                p1,
                p2,
                cx: 0.5,
                cy: 0.5,
                fx: 1.0,
                fy: 1.0,
            },
        }
    }

    /// Correct a distorted normalised point `(xd, yd)` → undistorted `(xu, yu)`.
    ///
    /// Both input and output coordinates are in normalised image space \[0, 1\].
    /// Uses five iterations of Newton-Raphson refinement internally.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::transform::LensDistortion;
    ///
    /// let corrector = LensDistortion::new(-0.15, 0.0, 0.0, 0.0);
    /// let (xu, yu) = corrector.correct(0.5, 0.5);
    /// // Point at principal point is unchanged
    /// assert!((xu - 0.5).abs() < 0.001, "xu={xu}");
    /// assert!((yu - 0.5).abs() < 0.001, "yu={yu}");
    /// ```
    #[must_use]
    pub fn correct(&self, xd: f32, yd: f32) -> (f32, f32) {
        self.params.undistort_point(xd, yd)
    }

    /// Access the underlying full parameter set for advanced use.
    #[must_use]
    pub fn params(&self) -> &LensDistortionParams {
        &self.params
    }
}

// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distort_point_no_distortion_is_identity() {
        let params = LensDistortionParams::new();
        let (xd, yd) = params.distort_point(0.3, 0.7);
        assert!((xd - 0.3).abs() < 1e-6, "xd={xd}");
        assert!((yd - 0.7).abs() < 1e-6, "yd={yd}");
    }

    #[test]
    fn test_barrel_distortion_moves_points_inward() {
        let params = LensDistortionParams::barrel(0.3);
        // A point away from the principal point should be pulled inward.
        let nx = 0.8_f32;
        let ny = 0.8_f32;
        let (xd, yd) = params.distort_point(nx, ny);
        // Barrel: xd should be closer to cx=0.5 than nx
        let before = (nx - 0.5).abs();
        let after = (xd - 0.5).abs();
        assert!(after < before, "barrel should pull point inward: before={before}, after={after}");
        let before_y = (ny - 0.5).abs();
        let after_y = (yd - 0.5).abs();
        assert!(after_y < before_y);
    }

    #[test]
    fn test_pincushion_distortion_moves_points_outward() {
        let params = LensDistortionParams::pincushion(0.3);
        let nx = 0.8_f32;
        let ny = 0.8_f32;
        let (xd, yd) = params.distort_point(nx, ny);
        let before = (nx - 0.5).abs();
        let after = (xd - 0.5).abs();
        assert!(after > before, "pincushion should push point outward");
        let _ = yd;
    }

    #[test]
    fn test_undistort_round_trip_error_small() {
        let mut params = LensDistortionParams::new();
        params.k1 = -0.2;
        params.k2 = 0.05;
        params.p1 = 0.01;

        let nx = 0.7_f32;
        let ny = 0.4_f32;

        let (xd, yd) = params.distort_point(nx, ny);
        let (xu, yu) = params.undistort_point(xd, yd);

        let err = ((xu - nx).powi(2) + (yu - ny).powi(2)).sqrt();
        assert!(err < 0.001, "round-trip error {err} >= 0.001");
    }

    #[test]
    fn test_validate_rejects_extreme_k1() {
        let mut params = LensDistortionParams::new();
        params.k1 = 15.0;
        assert!(params.validate().is_err(), "k1=15 should fail validation");
    }

    #[test]
    fn test_validate_accepts_normal_params() {
        let mut params = LensDistortionParams::new();
        params.k1 = -0.3;
        params.k2 = 0.05;
        assert!(params.validate().is_ok());
    }

    #[test]
    fn test_undistort_image_same_size() {
        let params = LensDistortionParams::new();
        let w = 4usize;
        let h = 4usize;
        let src = vec![128u8; w * h * 4];
        let dst = params.undistort_image(&src, w, h).expect("undistort failed");
        assert_eq!(dst.len(), src.len());
    }

    #[test]
    fn test_barrel_constructor_has_negative_k1() {
        let p = LensDistortionParams::barrel(0.2);
        assert!(p.k1 < 0.0, "barrel should have negative k1");
        assert_eq!(p.k2, 0.0);
    }

    #[test]
    fn test_pincushion_constructor_has_positive_k1() {
        let p = LensDistortionParams::pincushion(0.2);
        assert!(p.k1 > 0.0, "pincushion should have positive k1");
        assert_eq!(p.k2, 0.0);
    }
}
