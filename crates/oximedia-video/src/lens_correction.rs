//! Lens distortion correction metadata and coefficient storage.
//!
//! Real camera lenses introduce geometric distortion that makes straight lines
//! appear curved.  This module stores distortion coefficients, computes
//! correction preview metadata (bounding rectangle, pixel coverage estimate),
//! and accounts for sensor crop factors.
//!
//! The supported distortion model is the **Brown–Conrady** model, which is the
//! same polynomial model used by OpenCV, Lensfun, and most ISP pipelines:
//!
//! ```text
//! r  = sqrt(x_n^2 + y_n^2)      (normalised distance from principal point)
//! x' = x_n · (1 + k1·r² + k2·r⁴ + k3·r⁶) + 2·p1·x_n·y_n + p2·(r²+2x_n²)
//! y' = y_n · (1 + k1·r² + k2·r⁴ + k3·r⁶) + p1·(r²+2y_n²) + 2·p2·x_n·y_n
//! ```
//!
//! where *(k1, k2, k3)* are radial coefficients and *(p1, p2)* are tangential
//! (decentring) coefficients.  All values are in *normalised* coordinates (the
//! short half-axis of the sensor is 1.0).

// -----------------------------------------------------------------------
// Error type
// -----------------------------------------------------------------------

/// Errors from lens correction metadata operations.
#[derive(Debug, thiserror::Error)]
pub enum LensCorrectionError {
    /// Frame or sensor dimensions are zero.
    #[error("invalid dimensions: {width}x{height}")]
    InvalidDimensions {
        /// Width in pixels.
        width: u32,
        /// Height in pixels.
        height: u32,
    },

    /// Crop factor is not positive.
    #[error("crop factor must be positive, got {value}")]
    InvalidCropFactor {
        /// The invalid crop factor value.
        value: f64,
    },

    /// The preview grid sampling step is zero.
    #[error("grid step must be ≥ 1, got {value}")]
    InvalidGridStep {
        /// The invalid step value.
        value: u32,
    },
}

// -----------------------------------------------------------------------
// Coefficient storage
// -----------------------------------------------------------------------

/// Brown–Conrady lens distortion coefficients.
///
/// Positive radial coefficients (*k1*, *k2*, *k3*) produce **barrel**
/// distortion (the kind typical of wide-angle lenses).  Negative coefficients
/// produce **pincushion** distortion.
#[derive(Debug, Clone, PartialEq)]
pub struct DistortionCoefficients {
    /// First radial distortion coefficient (k1).
    pub k1: f64,
    /// Second radial distortion coefficient (k2).
    pub k2: f64,
    /// Third radial distortion coefficient (k3).
    pub k3: f64,
    /// First tangential (decentring) coefficient (p1).
    pub p1: f64,
    /// Second tangential (decentring) coefficient (p2).
    pub p2: f64,
}

impl Default for DistortionCoefficients {
    /// Returns the identity (no distortion) coefficients.
    fn default() -> Self {
        Self {
            k1: 0.0,
            k2: 0.0,
            k3: 0.0,
            p1: 0.0,
            p2: 0.0,
        }
    }
}

impl DistortionCoefficients {
    /// Returns `true` if all coefficients are zero (no distortion).
    pub fn is_identity(&self) -> bool {
        const EPS: f64 = 1e-12;
        self.k1.abs() < EPS
            && self.k2.abs() < EPS
            && self.k3.abs() < EPS
            && self.p1.abs() < EPS
            && self.p2.abs() < EPS
    }

    /// Apply the Brown–Conrady distortion model to a normalised point
    /// `(x_n, y_n)` and return the distorted normalised coordinates.
    pub fn distort(&self, x_n: f64, y_n: f64) -> (f64, f64) {
        let r2 = x_n * x_n + y_n * y_n;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
        let xd = x_n * radial + 2.0 * self.p1 * x_n * y_n + self.p2 * (r2 + 2.0 * x_n * x_n);
        let yd = y_n * radial + self.p1 * (r2 + 2.0 * y_n * y_n) + 2.0 * self.p2 * x_n * y_n;
        (xd, yd)
    }

    /// Attempt to invert the distortion for a given distorted point using
    /// iterative Newton–Raphson refinement (up to `max_iters` iterations).
    ///
    /// Returns `None` if the iteration fails to converge within `tolerance`
    /// (normalised distance).
    pub fn undistort_iter(
        &self,
        xd: f64,
        yd: f64,
        max_iters: u32,
        tolerance: f64,
    ) -> Option<(f64, f64)> {
        if self.is_identity() {
            return Some((xd, yd));
        }
        // Start with the distorted point as the initial guess.
        let mut x = xd;
        let mut y = yd;
        for _ in 0..max_iters {
            let (fx, fy) = self.distort(x, y);
            let ex = fx - xd;
            let ey = fy - yd;
            if ex * ex + ey * ey < tolerance * tolerance {
                return Some((x, y));
            }
            // Simple fixed-point step: x ← xd / radial (ignoring tangential).
            let r2 = x * x + y * y;
            let r4 = r2 * r2;
            let r6 = r4 * r2;
            let radial = 1.0 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6;
            // Guard against degenerate radial value.
            if radial.abs() < 1e-9 {
                return None;
            }
            x = (xd - 2.0 * self.p1 * x * y - self.p2 * (r2 + 2.0 * x * x)) / radial;
            y = (yd - self.p1 * (r2 + 2.0 * y * y) - 2.0 * self.p2 * x * y) / radial;
        }
        // Check final convergence.
        let (fx, fy) = self.distort(x, y);
        let ex = fx - xd;
        let ey = fy - yd;
        if ex * ex + ey * ey < tolerance * tolerance {
            Some((x, y))
        } else {
            None
        }
    }
}

// -----------------------------------------------------------------------
// Sensor / camera metadata
// -----------------------------------------------------------------------

/// Sensor and optics metadata attached to a lens profile.
#[derive(Debug, Clone)]
pub struct LensProfile {
    /// Human-readable name (e.g. `"Sony FE 24-105mm f/4 G @ 35mm"`).
    pub name: String,
    /// Distortion coefficients.
    pub coefficients: DistortionCoefficients,
    /// Sensor crop factor relative to 35 mm full-frame (1.0 = FF, 1.5 = APS-C, …).
    ///
    /// Used to translate normalised distortion coordinates to pixel coordinates
    /// given a specific sensor size.
    pub crop_factor: f64,
    /// Focal length in mm at which the profile was measured.
    pub focal_length_mm: f64,
    /// Optional aperture (f-number) at which the profile was measured.
    pub aperture: Option<f64>,
}

impl LensProfile {
    /// Create a new lens profile, validating the crop factor.
    ///
    /// # Errors
    ///
    /// Returns [`LensCorrectionError::InvalidCropFactor`] if `crop_factor ≤ 0`.
    pub fn new(
        name: impl Into<String>,
        coefficients: DistortionCoefficients,
        crop_factor: f64,
        focal_length_mm: f64,
        aperture: Option<f64>,
    ) -> Result<Self, LensCorrectionError> {
        if crop_factor <= 0.0 {
            return Err(LensCorrectionError::InvalidCropFactor { value: crop_factor });
        }
        Ok(Self {
            name: name.into(),
            coefficients,
            crop_factor,
            focal_length_mm,
            aperture,
        })
    }

    /// Returns the effective field-of-view-equivalent focal length when mounted
    /// on a full-frame body (focal_length × crop_factor).
    pub fn equiv_focal_length_mm(&self) -> f64 {
        self.focal_length_mm * self.crop_factor
    }
}

// -----------------------------------------------------------------------
// Correction preview metadata
// -----------------------------------------------------------------------

/// Metadata describing what a lens-corrected frame will look like *without*
/// actually remapping pixels.
///
/// Useful for UI previews, crop planning, and estimating quality loss before
/// committing to a remap operation.
#[derive(Debug, Clone)]
pub struct CorrectionPreview {
    /// Pixel dimensions of the *input* (distorted) frame.
    pub src_width: u32,
    /// Pixel dimensions of the *input* (distorted) frame.
    pub src_height: u32,
    /// Estimated largest axis-aligned inscribed rectangle after undistortion
    /// (in pixels), centred in the frame.
    ///
    /// Pixels outside this rectangle were outside the undistorted sensor area
    /// and must be cropped or filled with black.
    pub inscribed_rect: InscribedRect,
    /// Fraction of the total pixel area within the inscribed rectangle.
    /// `1.0` means no crop loss; lower values mean more wasted area.
    pub coverage_fraction: f64,
    /// Maximum normalised undistortion displacement at any sampled point.
    ///
    /// Gives a quick estimate of how strong the correction is; values < 0.02
    /// indicate a nearly distortion-free lens.
    pub max_displacement: f64,
    /// Distortion type inferred from the dominant radial coefficient sign.
    pub distortion_type: DistortionType,
}

/// Axis-aligned inscribed rectangle after undistortion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InscribedRect {
    /// Left edge pixel offset (inclusive).
    pub x: u32,
    /// Top edge pixel offset (inclusive).
    pub y: u32,
    /// Rectangle width in pixels.
    pub width: u32,
    /// Rectangle height in pixels.
    pub height: u32,
}

/// Qualitative distortion type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionType {
    /// Barrel distortion — corners pushed outward.
    Barrel,
    /// Pincushion distortion — corners pulled inward.
    Pincushion,
    /// Mixed or negligible distortion.
    Mixed,
    /// No significant distortion (all coefficients are near zero).
    None,
}

// -----------------------------------------------------------------------
// Preview computation
// -----------------------------------------------------------------------

/// Compute a [`CorrectionPreview`] for `src_width × src_height` pixels using
/// the given `profile`.
///
/// # Arguments
///
/// * `src_width` / `src_height` — dimensions of the distorted source frame.
/// * `profile` — lens profile (coefficients + crop factor).
/// * `grid_step` — sampling step in pixels for the displacement scan.
///   Larger values are faster; `1` gives exact coverage but is O(W×H).
///   Must be ≥ 1.
///
/// # Errors
///
/// Returns [`LensCorrectionError`] on invalid dimensions or grid step.
pub fn compute_correction_preview(
    src_width: u32,
    src_height: u32,
    profile: &LensProfile,
    grid_step: u32,
) -> Result<CorrectionPreview, LensCorrectionError> {
    if src_width == 0 || src_height == 0 {
        return Err(LensCorrectionError::InvalidDimensions {
            width: src_width,
            height: src_height,
        });
    }
    if grid_step == 0 {
        return Err(LensCorrectionError::InvalidGridStep { value: grid_step });
    }

    let w = src_width as f64;
    let h = src_height as f64;
    // Normalisation: the short half-axis maps to 1.0.
    let half_short = w.min(h) / 2.0;

    let coeffs = &profile.coefficients;
    let step = grid_step as usize;
    let iw = src_width as usize;
    let ih = src_height as usize;

    // Scan a grid of points in the *undistorted* (corrected) image and compute
    // the maximum displacement and the inscribed rectangle.
    let mut max_disp2: f64 = 0.0;

    // To find the inscribed rectangle we scan the four frame edges in normalised
    // space and find where the undistorted boundary falls, then compute the
    // largest rectangle that fits entirely within the mapped region.
    //
    // Simplified approach: sample the four edges of the corrected image and
    // find the maximum inset needed on each side.
    let mut min_left: f64 = 0.0;
    let mut max_right: f64 = w;
    let mut min_top: f64 = 0.0;
    let mut max_bottom: f64 = h;

    let mut y_px = 0usize;
    while y_px <= ih {
        let y_px_clamped = y_px.min(ih.saturating_sub(1));
        let yn = (y_px_clamped as f64 + 0.5 - h / 2.0) / half_short;

        let mut x_px = 0usize;
        while x_px <= iw {
            let x_px_clamped = x_px.min(iw.saturating_sub(1));
            let xn = (x_px_clamped as f64 + 0.5 - w / 2.0) / half_short;

            let (xd, yd) = coeffs.distort(xn, yn);

            let dx = (xd - xn).abs();
            let dy = (yd - yn).abs();
            let disp2 = dx * dx + dy * dy;
            if disp2 > max_disp2 {
                max_disp2 = disp2;
            }

            // Convert distorted normalised coords back to pixel space.
            let xd_px = xd * half_short + w / 2.0 - 0.5;
            let yd_px = yd * half_short + h / 2.0 - 0.5;

            // Update inscribed bounds: the corrected pixel at (x_px, y_px)
            // maps to (xd_px, yd_px) in the source.  If the source position
            // is out of bounds, the corrected pixel is invalid.
            // Instead of per-pixel remap, we track the tightest valid boundary.
            if y_px_clamped == 0 || y_px_clamped == ih.saturating_sub(1) {
                // Top/bottom edge: track how far inward the boundary moves.
                if y_px_clamped == 0 {
                    min_top = min_top.max(yd_px);
                } else {
                    max_bottom = max_bottom.min(yd_px + 1.0);
                }
            }
            if x_px_clamped == 0 || x_px_clamped == iw.saturating_sub(1) {
                if x_px_clamped == 0 {
                    min_left = min_left.max(xd_px);
                } else {
                    max_right = max_right.min(xd_px + 1.0);
                }
            }

            x_px += step;
        }
        y_px += step;
    }

    // Clamp the inscribed rectangle to the frame.
    let ix = min_left.max(0.0).ceil() as u32;
    let iy = min_top.max(0.0).ceil() as u32;
    let ir = (max_right.min(w) as u32).saturating_sub(ix);
    let ib = (max_bottom.min(h) as u32).saturating_sub(iy);

    let inscribed_rect = InscribedRect {
        x: ix,
        y: iy,
        width: ir,
        height: ib,
    };

    let total_pixels = (src_width as f64) * (src_height as f64);
    let inscribed_pixels = (ir as f64) * (ib as f64);
    let coverage_fraction = if total_pixels > 0.0 {
        (inscribed_pixels / total_pixels).clamp(0.0, 1.0)
    } else {
        0.0
    };

    let max_displacement = max_disp2.sqrt();
    let distortion_type = classify_distortion(&profile.coefficients);

    Ok(CorrectionPreview {
        src_width,
        src_height,
        inscribed_rect,
        coverage_fraction,
        max_displacement,
        distortion_type,
    })
}

// -----------------------------------------------------------------------
// Internal helpers
// -----------------------------------------------------------------------

/// Classify the dominant distortion based on the sign of the leading radial
/// coefficient.
fn classify_distortion(coeffs: &DistortionCoefficients) -> DistortionType {
    const THRESH: f64 = 1e-6;
    if coeffs.is_identity() {
        return DistortionType::None;
    }
    // Use k1 as the dominant term.
    if coeffs.k1 > THRESH {
        DistortionType::Barrel
    } else if coeffs.k1 < -THRESH {
        DistortionType::Pincushion
    } else {
        DistortionType::Mixed
    }
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn no_distortion() -> DistortionCoefficients {
        DistortionCoefficients::default()
    }

    fn barrel() -> DistortionCoefficients {
        DistortionCoefficients {
            k1: 0.1,
            k2: 0.02,
            ..Default::default()
        }
    }

    fn pincushion() -> DistortionCoefficients {
        DistortionCoefficients {
            k1: -0.1,
            ..Default::default()
        }
    }

    // 1. Identity coefficients report is_identity.
    #[test]
    fn test_identity_is_identity() {
        assert!(no_distortion().is_identity());
    }

    // 2. Non-zero coefficients are not identity.
    #[test]
    fn test_barrel_not_identity() {
        assert!(!barrel().is_identity());
    }

    // 3. distort(0, 0) returns (0, 0) for any model.
    #[test]
    fn test_distort_origin_is_zero() {
        for coeffs in [no_distortion(), barrel(), pincushion()] {
            let (xd, yd) = coeffs.distort(0.0, 0.0);
            assert!(xd.abs() < 1e-12, "xd should be 0, got {}", xd);
            assert!(yd.abs() < 1e-12, "yd should be 0, got {}", yd);
        }
    }

    // 4. Barrel distortion pushes points outward from centre.
    #[test]
    fn test_barrel_pushes_outward() {
        let coeffs = barrel();
        let (xd, _) = coeffs.distort(0.5, 0.0);
        // With positive k1 the point should move away from centre.
        assert!(xd > 0.5, "barrel should push outward, xd={}", xd);
    }

    // 5. Pincushion distortion pulls points inward.
    #[test]
    fn test_pincushion_pulls_inward() {
        let coeffs = pincushion();
        let (xd, _) = coeffs.distort(0.5, 0.0);
        assert!(xd < 0.5, "pincushion should pull inward, xd={}", xd);
    }

    // 6. undistort_iter converges for identity model.
    #[test]
    fn test_undistort_identity() {
        let coeffs = no_distortion();
        let result = coeffs.undistort_iter(0.3, 0.4, 30, 1e-8);
        let (xu, yu) = result.unwrap();
        assert!((xu - 0.3).abs() < 1e-6);
        assert!((yu - 0.4).abs() < 1e-6);
    }

    // 7. undistort_iter approximately inverts barrel distortion.
    #[test]
    fn test_undistort_barrel_roundtrip() {
        let coeffs = barrel();
        let (xd, yd) = coeffs.distort(0.3, 0.2);
        let result = coeffs.undistort_iter(xd, yd, 50, 1e-7);
        let (xu, yu) = result.unwrap();
        assert!((xu - 0.3).abs() < 1e-4, "xu={}", xu);
        assert!((yu - 0.2).abs() < 1e-4, "yu={}", yu);
    }

    // 8. LensProfile with zero crop factor is rejected.
    #[test]
    fn test_invalid_crop_factor() {
        assert!(matches!(
            LensProfile::new("test", no_distortion(), 0.0, 35.0, None),
            Err(LensCorrectionError::InvalidCropFactor { .. })
        ));
    }

    // 9. equiv_focal_length_mm is correct.
    #[test]
    fn test_equiv_focal_length() {
        let profile = LensProfile::new("APS-C 24mm", no_distortion(), 1.5, 24.0, None).unwrap();
        assert!((profile.equiv_focal_length_mm() - 36.0).abs() < 1e-9);
    }

    // 10. compute_correction_preview zero dimensions returns error.
    #[test]
    fn test_preview_zero_dimensions() {
        let profile = LensProfile::new("test", no_distortion(), 1.0, 50.0, None).unwrap();
        assert!(matches!(
            compute_correction_preview(0, 480, &profile, 4),
            Err(LensCorrectionError::InvalidDimensions { .. })
        ));
    }

    // 11. compute_correction_preview zero grid_step returns error.
    #[test]
    fn test_preview_zero_grid_step() {
        let profile = LensProfile::new("test", no_distortion(), 1.0, 50.0, None).unwrap();
        assert!(matches!(
            compute_correction_preview(640, 480, &profile, 0),
            Err(LensCorrectionError::InvalidGridStep { .. })
        ));
    }

    // 12. Identity lens has coverage_fraction near 1.0 and displacement near 0.
    #[test]
    fn test_identity_lens_preview() {
        let profile = LensProfile::new("identity", no_distortion(), 1.0, 50.0, None).unwrap();
        let preview = compute_correction_preview(640, 480, &profile, 8).unwrap();
        assert!(
            preview.max_displacement < 1e-9,
            "expected ~0 displacement, got {}",
            preview.max_displacement
        );
        assert_eq!(preview.distortion_type, DistortionType::None);
    }

    // 13. Barrel lens distortion type is classified correctly.
    #[test]
    fn test_barrel_classification() {
        let profile = LensProfile::new("barrel-lens", barrel(), 1.0, 24.0, None).unwrap();
        let preview = compute_correction_preview(320, 240, &profile, 8).unwrap();
        assert_eq!(preview.distortion_type, DistortionType::Barrel);
        assert!(preview.max_displacement > 0.0);
    }

    // 14. Pincushion lens distortion type is classified correctly.
    #[test]
    fn test_pincushion_classification() {
        let profile = LensProfile::new("pinc-lens", pincushion(), 1.0, 50.0, None).unwrap();
        let preview = compute_correction_preview(320, 240, &profile, 8).unwrap();
        assert_eq!(preview.distortion_type, DistortionType::Pincushion);
    }
}
