//! Lens distortion and vignetting profiling.
//!
//! Implements Brown–Conrady radial distortion correction, polynomial vignetting
//! models, a lens profile database, and linear interpolation between profiles.

#![allow(dead_code)]

// ── DistortionModel ───────────────────────────────────────────────────────────

/// Mathematical model used to represent radial distortion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistortionModel {
    /// Standard polynomial (Brown–Conrady) model (k1, k2, k3 coefficients).
    Polynomial,
    /// Division model (single-parameter).
    Division,
    /// `PTLens` / Hugin database model.
    PtLens,
    /// Fisheye equidistant projection.
    FisheyeEquidistant,
}

impl DistortionModel {
    /// Returns a human-readable name for the model.
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Polynomial => "Polynomial (Brown-Conrady)",
            Self::Division => "Division",
            Self::PtLens => "PTLens",
            Self::FisheyeEquidistant => "Fisheye Equidistant",
        }
    }
}

// ── LensProfile ───────────────────────────────────────────────────────────────

/// A complete lens distortion and vignetting profile.
#[derive(Debug, Clone)]
pub struct LensProfile {
    /// Camera body make/model.
    pub camera: String,
    /// Lens name (e.g. "Canon EF 50mm f/1.4").
    pub lens: String,
    /// Focal length in mm at which this profile was measured.
    pub focal_length_mm: f32,
    /// Aperture (f-stop) at which this profile was measured.
    pub aperture: f32,
    /// Radial distortion coefficient k1.
    pub distortion_k1: f32,
    /// Radial distortion coefficient k2.
    pub distortion_k2: f32,
    /// Radial distortion coefficient k3.
    pub distortion_k3: f32,
    /// Polynomial vignetting coefficients [a0, a1, a2, a3].
    pub vignetting: [f32; 4],
}

impl LensProfile {
    /// Apply Brown–Conrady radial distortion correction to a point.
    ///
    /// # Arguments
    /// * `x`, `y` – distorted pixel coordinates.
    /// * `cx`, `cy` – principal point (optical centre).
    ///
    /// Returns corrected `(x, y)` coordinates.
    #[must_use]
    pub fn correct_distortion(&self, x: f32, y: f32, cx: f32, cy: f32) -> (f32, f32) {
        let dx = x - cx;
        let dy = y - cy;
        let r2 = dx * dx + dy * dy;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let scale =
            1.0 + self.distortion_k1 * r2 + self.distortion_k2 * r4 + self.distortion_k3 * r6;
        (cx + dx * scale, cy + dy * scale)
    }
}

// ── VignettingModel ───────────────────────────────────────────────────────────

/// Polynomial vignetting model (gain at radius r).
pub struct VignettingModel;

impl VignettingModel {
    /// Compute the vignetting gain at normalised radius `r` (0 = centre, 1 = corner).
    ///
    /// Uses the polynomial  `gain = a0 + a1·r² + a2·r⁴ + a3·r⁶`.
    #[must_use]
    pub fn compute(r: f32, coeffs: &[f32; 4]) -> f32 {
        let r2 = r * r;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        coeffs[0] + coeffs[1] * r2 + coeffs[2] * r4 + coeffs[3] * r6
    }
}

// ── LensDatabase ─────────────────────────────────────────────────────────────

/// A collection of `LensProfile` entries that can be searched by camera/lens/focal length.
#[derive(Debug, Default)]
pub struct LensDatabase {
    profiles: Vec<LensProfile>,
}

impl LensDatabase {
    /// Create an empty database.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a profile to the database.
    pub fn add(&mut self, profile: LensProfile) {
        self.profiles.push(profile);
    }

    /// Find the best-matching profile for the given camera / lens / focal length.
    ///
    /// Exact match on camera + lens, then nearest focal length.
    #[must_use]
    pub fn find(&self, camera: &str, lens: &str, focal_mm: f32) -> Option<&LensProfile> {
        let candidates: Vec<&LensProfile> = self
            .profiles
            .iter()
            .filter(|p| p.camera == camera && p.lens == lens)
            .collect();

        if candidates.is_empty() {
            return None;
        }

        candidates.into_iter().min_by(|a, b| {
            let da = (a.focal_length_mm - focal_mm).abs();
            let db = (b.focal_length_mm - focal_mm).abs();
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
    }

    /// Returns the number of profiles stored.
    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    /// Returns `true` if no profiles are stored.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

// ── ProfileInterpolator ───────────────────────────────────────────────────────

/// Interpolates between two `LensProfile`s at the same camera/lens.
pub struct ProfileInterpolator;

impl ProfileInterpolator {
    /// Linearly blend all distortion and vignetting coefficients.
    ///
    /// `t = 0.0` returns `near`, `t = 1.0` returns `far`.
    #[must_use]
    pub fn interpolate(near: &LensProfile, far: &LensProfile, t: f32) -> LensProfile {
        let t = t.clamp(0.0, 1.0);
        let lerp = |a: f32, b: f32| a + (b - a) * t;
        LensProfile {
            camera: near.camera.clone(),
            lens: near.lens.clone(),
            focal_length_mm: lerp(near.focal_length_mm, far.focal_length_mm),
            aperture: lerp(near.aperture, far.aperture),
            distortion_k1: lerp(near.distortion_k1, far.distortion_k1),
            distortion_k2: lerp(near.distortion_k2, far.distortion_k2),
            distortion_k3: lerp(near.distortion_k3, far.distortion_k3),
            vignetting: [
                lerp(near.vignetting[0], far.vignetting[0]),
                lerp(near.vignetting[1], far.vignetting[1]),
                lerp(near.vignetting[2], far.vignetting[2]),
                lerp(near.vignetting[3], far.vignetting[3]),
            ],
        }
    }
}

// ── tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_profile(focal: f32, k1: f32) -> LensProfile {
        LensProfile {
            camera: "TestCam".to_string(),
            lens: "TestLens 50mm".to_string(),
            focal_length_mm: focal,
            aperture: 2.8,
            distortion_k1: k1,
            distortion_k2: 0.0,
            distortion_k3: 0.0,
            vignetting: [1.0, -0.1, 0.05, -0.01],
        }
    }

    #[test]
    fn test_distortion_model_names() {
        assert!(DistortionModel::Polynomial.name().contains("Polynomial"));
        assert!(DistortionModel::Division.name().contains("Division"));
        assert!(DistortionModel::PtLens.name().contains("PTLens"));
        assert!(DistortionModel::FisheyeEquidistant
            .name()
            .contains("Fisheye"));
    }

    #[test]
    fn test_correct_distortion_zero_coefficients() {
        let profile = sample_profile(50.0, 0.0);
        let (x2, y2) = profile.correct_distortion(100.0, 100.0, 50.0, 50.0);
        // With k1=k2=k3=0, scale=1, so result should equal input
        assert!((x2 - 100.0).abs() < 1e-5);
        assert!((y2 - 100.0).abs() < 1e-5);
    }

    #[test]
    fn test_correct_distortion_at_center() {
        let profile = sample_profile(50.0, -0.1);
        // Point at the optical centre: (cx, cy) → unchanged
        let (x2, y2) = profile.correct_distortion(50.0, 50.0, 50.0, 50.0);
        assert!((x2 - 50.0).abs() < 1e-5);
        assert!((y2 - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_correct_distortion_barrel() {
        // Negative k1 → barrel distortion → points move towards centre
        let profile = sample_profile(50.0, -0.0001);
        let cx = 100.0f32;
        let cy = 100.0f32;
        let (x2, _y2) = profile.correct_distortion(200.0, 100.0, cx, cy);
        // With negative k1, corrected x should be closer to center than 200
        assert!(x2 < 200.0, "Expected barrel correction: x2={}", x2);
    }

    #[test]
    fn test_vignetting_gain_center() {
        let coeffs = [1.0f32, -0.1, 0.05, -0.01];
        let gain = VignettingModel::compute(0.0, &coeffs);
        assert!((gain - coeffs[0]).abs() < 1e-6);
    }

    #[test]
    fn test_vignetting_gain_edge() {
        let coeffs = [1.0f32, -0.5, 0.2, -0.05];
        let gain = VignettingModel::compute(1.0, &coeffs);
        let expected = 1.0 - 0.5 + 0.2 - 0.05;
        assert!((gain - expected).abs() < 1e-5);
    }

    #[test]
    fn test_database_add_and_find() {
        let mut db = LensDatabase::new();
        db.add(sample_profile(50.0, -0.01));
        assert_eq!(db.len(), 1);
        let found = db.find("TestCam", "TestLens 50mm", 50.0);
        assert!(found.is_some());
    }

    #[test]
    fn test_database_not_found() {
        let mut db = LensDatabase::new();
        db.add(sample_profile(50.0, -0.01));
        assert!(db.find("OtherCam", "TestLens 50mm", 50.0).is_none());
    }

    #[test]
    fn test_database_nearest_focal() {
        let mut db = LensDatabase::new();
        db.add(sample_profile(24.0, -0.02));
        db.add(sample_profile(70.0, -0.005));
        // 35 mm is closer to 24 mm than to 70 mm
        let found = db
            .find("TestCam", "TestLens 50mm", 35.0)
            .expect("expected item to be found");
        assert!((found.focal_length_mm - 24.0).abs() < 1e-6);
    }

    #[test]
    fn test_database_is_empty() {
        let db = LensDatabase::new();
        assert!(db.is_empty());
    }

    #[test]
    fn test_interpolate_t0() {
        let near = sample_profile(24.0, -0.02);
        let far = sample_profile(70.0, -0.005);
        let interp = ProfileInterpolator::interpolate(&near, &far, 0.0);
        assert!((interp.focal_length_mm - 24.0).abs() < 1e-5);
        assert!((interp.distortion_k1 - (-0.02)).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_t1() {
        let near = sample_profile(24.0, -0.02);
        let far = sample_profile(70.0, -0.005);
        let interp = ProfileInterpolator::interpolate(&near, &far, 1.0);
        assert!((interp.focal_length_mm - 70.0).abs() < 1e-5);
        assert!((interp.distortion_k1 - (-0.005)).abs() < 1e-6);
    }

    #[test]
    fn test_interpolate_midpoint() {
        let near = sample_profile(24.0, -0.02);
        let far = sample_profile(70.0, -0.005);
        let interp = ProfileInterpolator::interpolate(&near, &far, 0.5);
        assert!((interp.focal_length_mm - 47.0).abs() < 1e-4);
    }

    #[test]
    fn test_interpolate_clamps_t() {
        let near = sample_profile(24.0, -0.02);
        let far = sample_profile(70.0, -0.005);
        let interp_neg = ProfileInterpolator::interpolate(&near, &far, -0.5);
        let interp_gt1 = ProfileInterpolator::interpolate(&near, &far, 1.5);
        assert!((interp_neg.focal_length_mm - 24.0).abs() < 1e-5);
        assert!((interp_gt1.focal_length_mm - 70.0).abs() < 1e-5);
    }
}
