//! Camera lens projection models for virtual production.
//!
//! Provides geometric projection/unprojection between 3D world space and 2D
//! normalised image coordinates, supporting multiple optical models used in
//! broadcast and virtual production cameras:
//!
//! - **Rectilinear** (standard pinhole) – straight lines remain straight.
//! - **Fisheye** – equidistant angular mapping for ultra-wide optics.
//! - **Equidistant** – alias for the fisheye model (r = f·θ).
//! - **Equisolid** – area-preserving fisheye (r = 2f·sin(θ/2)).
//! - **Orthographic** – parallel-projection fisheye (r = f·sin(θ)).
//!
//! All image coordinates are normalised to **[-1, 1]** in both axes
//! (independent of sensor aspect ratio), with (0, 0) at the optical centre.

/// Camera lens projection model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ProjectionModel {
    /// Standard rectilinear (pinhole) projection.
    ///
    /// Preserves straight lines; FOV is limited to <180°.
    Rectilinear,
    /// Equidistant fisheye: r = f·θ.
    ///
    /// Equal angle increments map to equal radial distances.
    Fisheye,
    /// Equidistant fisheye (alias for [`ProjectionModel::Fisheye`]).
    Equidistant,
    /// Equisolid-angle fisheye: r = 2f·sin(θ/2).
    ///
    /// Equal solid-angle elements map to equal image areas.
    Equisolid,
    /// Orthographic fisheye: r = f·sin(θ).
    ///
    /// Captures up to exactly 180° (hemisphere).
    Orthographic,
}

/// Physical lens and sensor parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LensParams {
    /// Effective focal length in millimetres.
    pub focal_length_mm: f32,
    /// Sensor width in millimetres (e.g. 36.0 for full-frame 35 mm).
    pub sensor_width_mm: f32,
    /// Sensor height in millimetres (e.g. 24.0 for full-frame 35 mm).
    pub sensor_height_mm: f32,
    /// First radial distortion coefficient (Brown-Conrady k1).
    pub distortion_k1: f32,
    /// Second radial distortion coefficient (Brown-Conrady k2).
    pub distortion_k2: f32,
}

impl LensParams {
    /// Construct a new `LensParams`.
    #[must_use]
    pub fn new(
        focal_length_mm: f32,
        sensor_width_mm: f32,
        sensor_height_mm: f32,
        distortion_k1: f32,
        distortion_k2: f32,
    ) -> Self {
        Self {
            focal_length_mm,
            sensor_width_mm,
            sensor_height_mm,
            distortion_k1,
            distortion_k2,
        }
    }

    /// Returns the normalised focal length along the X axis.
    ///
    /// `fx_norm = 2 * focal_length_mm / sensor_width_mm`
    ///
    /// This is the focal length in units where the sensor half-width is 1.
    #[must_use]
    pub fn fx_norm(&self) -> f32 {
        if self.sensor_width_mm == 0.0 {
            return 0.0;
        }
        2.0 * self.focal_length_mm / self.sensor_width_mm
    }

    /// Returns the normalised focal length along the Y axis.
    ///
    /// `fy_norm = 2 * focal_length_mm / sensor_height_mm`
    #[must_use]
    pub fn fy_norm(&self) -> f32 {
        if self.sensor_height_mm == 0.0 {
            return 0.0;
        }
        2.0 * self.focal_length_mm / self.sensor_height_mm
    }
}

impl Default for LensParams {
    fn default() -> Self {
        // 50 mm lens on a full-frame (36×24 mm) sensor, no distortion.
        Self {
            focal_length_mm: 50.0,
            sensor_width_mm: 36.0,
            sensor_height_mm: 24.0,
            distortion_k1: 0.0,
            distortion_k2: 0.0,
        }
    }
}

/// Projection and un-projection utilities for a given lens model.
///
/// All coordinates use a right-handed camera-space coordinate system where
/// the camera looks along the **+Z** axis: X is right, Y is up, Z is depth.
///
/// Normalised image coordinates span [-1, 1] on both axes with (0, 0) at the
/// principal point. Distortion is applied *after* the linear projection and
/// removed *before* un-projection.
pub struct ProjectionMapper;

impl ProjectionMapper {
    /// Project a 3-D camera-space point onto the 2-D sensor plane.
    ///
    /// # Arguments
    /// * `x`, `y`, `z` – 3-D position in camera space (Z is depth along optical axis).
    /// * `params` – lens parameters.
    /// * `model` – projection model.
    ///
    /// # Returns
    /// `Some((px, py))` in normalised image coordinates [-1, 1], or `None` if the
    /// point is behind (or on) the camera plane.
    #[must_use]
    pub fn project(
        x: f32,
        y: f32,
        z: f32,
        params: &LensParams,
        model: ProjectionModel,
    ) -> Option<(f32, f32)> {
        if z <= 0.0 {
            return None;
        }

        let fx = params.fx_norm();
        let fy = params.fy_norm();

        if fx == 0.0 || fy == 0.0 {
            return None;
        }

        let (nx, ny) = match model {
            ProjectionModel::Rectilinear => {
                // Standard pinhole: divide by depth.
                (x / z * fx, y / z * fy)
            }
            ProjectionModel::Fisheye | ProjectionModel::Equidistant => {
                // r = f * theta; theta = atan(sqrt(x²+y²) / z)
                let r_3d = (x * x + y * y).sqrt();
                let theta = r_3d.atan2(z); // angle from optical axis
                if r_3d < 1e-9 {
                    (0.0, 0.0)
                } else {
                    let r_img = theta; // equidistant: r = theta (in units of 1 radian)
                    let scale = r_img / r_3d;
                    (x * scale * fx, y * scale * fy)
                }
            }
            ProjectionModel::Equisolid => {
                // r = 2 * f * sin(theta / 2)
                let r_3d = (x * x + y * y).sqrt();
                let theta = r_3d.atan2(z);
                if r_3d < 1e-9 {
                    (0.0, 0.0)
                } else {
                    let r_img = 2.0 * (theta / 2.0).sin();
                    let scale = r_img / r_3d;
                    (x * scale * fx, y * scale * fy)
                }
            }
            ProjectionModel::Orthographic => {
                // r = f * sin(theta)
                let r_3d = (x * x + y * y).sqrt();
                let theta = r_3d.atan2(z);
                if theta > std::f32::consts::FRAC_PI_2 {
                    // Behind the lens plane for orthographic projection.
                    return None;
                }
                if r_3d < 1e-9 {
                    (0.0, 0.0)
                } else {
                    let r_img = theta.sin();
                    let scale = r_img / r_3d;
                    (x * scale * fx, y * scale * fy)
                }
            }
        };

        // Apply radial distortion after projection.
        let (dx, dy) =
            Self::apply_distortion(nx, ny, params.distortion_k1, params.distortion_k2);

        Some((dx, dy))
    }

    /// Un-project a 2-D normalised image point to a 3-D ray direction.
    ///
    /// The returned direction vector has unit length and points from the optical
    /// centre into the scene. The Z component is always non-negative.
    ///
    /// # Arguments
    /// * `px`, `py` – normalised image coordinates [-1, 1].
    /// * `params` – lens parameters.
    /// * `model` – projection model.
    ///
    /// # Returns
    /// Normalised 3-D direction vector `(dx, dy, dz)`.
    #[must_use]
    pub fn unproject(
        px: f32,
        py: f32,
        params: &LensParams,
        model: ProjectionModel,
    ) -> (f32, f32, f32) {
        // Remove radial distortion first.
        let (ux, uy) =
            Self::remove_distortion(px, py, params.distortion_k1, params.distortion_k2);

        let fx = params.fx_norm();
        let fy = params.fy_norm();

        // Convert from normalised image coords to angular/direction.
        let nx = if fx != 0.0 { ux / fx } else { 0.0 };
        let ny = if fy != 0.0 { uy / fy } else { 0.0 };

        let (dir_x, dir_y, dir_z) = match model {
            ProjectionModel::Rectilinear => {
                // Direction: (nx, ny, 1) normalised.
                let len = (nx * nx + ny * ny + 1.0).sqrt();
                (nx / len, ny / len, 1.0 / len)
            }
            ProjectionModel::Fisheye | ProjectionModel::Equidistant => {
                // r = theta; theta = r_img; recover 3D direction.
                let r_img = (nx * nx + ny * ny).sqrt();
                let theta = r_img; // equidistant: r = theta (radians)
                if r_img < 1e-9 {
                    (0.0, 0.0, 1.0)
                } else {
                    let sin_t = theta.sin();
                    let cos_t = theta.cos();
                    let scale = sin_t / r_img;
                    (nx * scale, ny * scale, cos_t)
                }
            }
            ProjectionModel::Equisolid => {
                // r_img = 2 * sin(theta/2) → theta = 2 * asin(r_img/2)
                let r_img = (nx * nx + ny * ny).sqrt();
                let half_arg = (r_img / 2.0).clamp(-1.0, 1.0);
                let theta = 2.0 * half_arg.asin();
                if r_img < 1e-9 {
                    (0.0, 0.0, 1.0)
                } else {
                    let sin_t = theta.sin();
                    let cos_t = theta.cos();
                    let scale = sin_t / r_img;
                    (nx * scale, ny * scale, cos_t)
                }
            }
            ProjectionModel::Orthographic => {
                // r_img = sin(theta) → theta = asin(r_img)
                let r_img = (nx * nx + ny * ny).sqrt();
                let arg = r_img.clamp(-1.0, 1.0);
                let theta = arg.asin();
                if r_img < 1e-9 {
                    (0.0, 0.0, 1.0)
                } else {
                    let sin_t = theta.sin(); // = r_img
                    let cos_t = theta.cos();
                    let scale = sin_t / r_img;
                    (nx * scale, ny * scale, cos_t)
                }
            }
        };

        // Return already-normalised direction (each branch above produces unit vec).
        let len = (dir_x * dir_x + dir_y * dir_y + dir_z * dir_z).sqrt();
        if len < 1e-12 {
            (0.0, 0.0, 1.0)
        } else {
            (dir_x / len, dir_y / len, dir_z / len)
        }
    }

    /// Apply Brown-Conrady radial distortion to normalised image coordinates.
    ///
    /// The distorted coordinates `(xd, yd)` satisfy:
    /// ```text
    /// r² = x² + y²
    /// xd = x * (1 + k1*r² + k2*r⁴)
    /// yd = y * (1 + k1*r² + k2*r⁴)
    /// ```
    #[must_use]
    pub fn apply_distortion(x: f32, y: f32, k1: f32, k2: f32) -> (f32, f32) {
        let r2 = x * x + y * y;
        let r4 = r2 * r2;
        let factor = 1.0 + k1 * r2 + k2 * r4;
        (x * factor, y * factor)
    }

    /// Remove Brown-Conrady radial distortion using iterative fixed-point refinement.
    ///
    /// Inverts `apply_distortion` by iterating the residual correction up to
    /// 50 times or until the correction magnitude drops below 1 × 10⁻⁷.
    #[must_use]
    pub fn remove_distortion(x: f32, y: f32, k1: f32, k2: f32) -> (f32, f32) {
        // Short circuit when there is no distortion.
        if k1 == 0.0 && k2 == 0.0 {
            return (x, y);
        }

        // Iterative scheme: start from the distorted point as an initial guess
        // for the undistorted point, then refine.
        let mut xu = x;
        let mut yu = y;

        for _ in 0..50 {
            let r2 = xu * xu + yu * yu;
            let r4 = r2 * r2;
            let factor = 1.0 + k1 * r2 + k2 * r4;
            if factor.abs() < 1e-15 {
                break;
            }
            let xu_new = x / factor;
            let yu_new = y / factor;

            let dx = xu_new - xu;
            let dy = yu_new - yu;
            xu = xu_new;
            yu = yu_new;

            if dx * dx + dy * dy < 1e-14 {
                break;
            }
        }

        (xu, yu)
    }
}

/// Field-of-view calculator for a given lens and sensor combination.
pub struct FovCalculator;

impl FovCalculator {
    /// Calculate the horizontal field of view in degrees for a rectilinear lens.
    ///
    /// Uses the standard formula:
    /// ```text
    /// HFOV = 2 * atan(sensor_width / (2 * focal_length))
    /// ```
    ///
    /// For fisheye and other wide-angle models the actual FOV can exceed this
    /// value; this method always returns the rectilinear equivalent.
    #[must_use]
    pub fn horizontal_fov(params: &LensParams) -> f32 {
        if params.focal_length_mm <= 0.0 {
            return 180.0;
        }
        let half_angle = (params.sensor_width_mm / (2.0 * params.focal_length_mm)).atan();
        half_angle.to_degrees() * 2.0
    }

    /// Calculate the vertical field of view in degrees for a rectilinear lens.
    #[must_use]
    pub fn vertical_fov(params: &LensParams) -> f32 {
        if params.focal_length_mm <= 0.0 {
            return 180.0;
        }
        let half_angle = (params.sensor_height_mm / (2.0 * params.focal_length_mm)).atan();
        half_angle.to_degrees() * 2.0
    }

    /// Compute the diagonal field of view in degrees.
    #[must_use]
    pub fn diagonal_fov(params: &LensParams) -> f32 {
        if params.focal_length_mm <= 0.0 {
            return 180.0;
        }
        let diag_mm =
            (params.sensor_width_mm * params.sensor_width_mm
                + params.sensor_height_mm * params.sensor_height_mm)
                .sqrt();
        let half_angle = (diag_mm / (2.0 * params.focal_length_mm)).atan();
        half_angle.to_degrees() * 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Helper
    // -----------------------------------------------------------------------

    fn default_lens() -> LensParams {
        LensParams::default() // 50 mm / 36×24 / no distortion
    }

    fn approx_eq(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    // -----------------------------------------------------------------------
    // Rectilinear project / unproject round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_rectilinear_project_centre() {
        let p = ProjectionMapper::project(0.0, 0.0, 1.0, &default_lens(), ProjectionModel::Rectilinear);
        let (px, py) = p.expect("centre should project successfully");
        assert!(approx_eq(px, 0.0, 1e-5), "centre x should be 0, got {px}");
        assert!(approx_eq(py, 0.0, 1e-5), "centre y should be 0, got {py}");
    }

    #[test]
    fn test_rectilinear_project_unproject_roundtrip() {
        let lens = default_lens();
        let model = ProjectionModel::Rectilinear;

        // Original 3-D point.
        let (x, y, z) = (0.3, -0.2, 2.0);
        let (px, py) = ProjectionMapper::project(x, y, z, &lens, model)
            .expect("point should be in front of camera");

        // Unproject back to a ray and verify it points in the right direction.
        let (dx, dy, dz) =
            ProjectionMapper::unproject(px, py, &lens, model);

        // Ray should be parallel to the original direction (x, y, z) / |(x,y,z)|.
        let orig_len = (x * x + y * y + z * z).sqrt();
        let ox = x / orig_len;
        let oy = y / orig_len;
        let oz = z / orig_len;

        assert!(approx_eq(dx, ox, 1e-4), "dx: {dx} vs {ox}");
        assert!(approx_eq(dy, oy, 1e-4), "dy: {dy} vs {oy}");
        assert!(approx_eq(dz, oz, 1e-4), "dz: {dz} vs {oz}");
    }

    #[test]
    fn test_rectilinear_behind_camera_returns_none() {
        let lens = default_lens();
        let result =
            ProjectionMapper::project(0.0, 0.0, -1.0, &lens, ProjectionModel::Rectilinear);
        assert!(
            result.is_none(),
            "point behind camera should return None"
        );
    }

    #[test]
    fn test_rectilinear_z_zero_returns_none() {
        let lens = default_lens();
        let result =
            ProjectionMapper::project(1.0, 1.0, 0.0, &lens, ProjectionModel::Rectilinear);
        assert!(result.is_none(), "z=0 should return None");
    }

    // -----------------------------------------------------------------------
    // Fisheye / Equidistant
    // -----------------------------------------------------------------------

    #[test]
    fn test_fisheye_centre_projects_to_origin() {
        let lens = default_lens();
        let p = ProjectionMapper::project(0.0, 0.0, 1.0, &lens, ProjectionModel::Fisheye);
        let (px, py) = p.expect("centre should project");
        assert!(approx_eq(px, 0.0, 1e-5));
        assert!(approx_eq(py, 0.0, 1e-5));
    }

    #[test]
    fn test_fisheye_equidistant_alias_same_result() {
        let lens = default_lens();
        let (x, y, z) = (0.4, 0.2, 1.5);
        let pf = ProjectionMapper::project(x, y, z, &lens, ProjectionModel::Fisheye);
        let pe = ProjectionMapper::project(x, y, z, &lens, ProjectionModel::Equidistant);
        let (fx, fy) = pf.expect("fisheye");
        let (ex, ey) = pe.expect("equidistant");
        assert!(approx_eq(fx, ex, 1e-6), "fisheye == equidistant in x");
        assert!(approx_eq(fy, ey, 1e-6), "fisheye == equidistant in y");
    }

    #[test]
    fn test_fisheye_roundtrip() {
        let lens = default_lens();
        let model = ProjectionModel::Fisheye;
        let (x, y, z) = (0.2, 0.1, 1.0);
        let (px, py) = ProjectionMapper::project(x, y, z, &lens, model)
            .expect("fisheye project");
        let (dx, dy, dz) = ProjectionMapper::unproject(px, py, &lens, model);

        let orig_len = (x * x + y * y + z * z).sqrt();
        assert!(approx_eq(dx, x / orig_len, 5e-4), "dx mismatch: {dx}");
        assert!(approx_eq(dy, y / orig_len, 5e-4), "dy mismatch: {dy}");
        assert!(approx_eq(dz, z / orig_len, 5e-4), "dz mismatch: {dz}");
    }

    // -----------------------------------------------------------------------
    // Equisolid
    // -----------------------------------------------------------------------

    #[test]
    fn test_equisolid_roundtrip() {
        let lens = default_lens();
        let model = ProjectionModel::Equisolid;
        let (x, y, z) = (0.15, -0.1, 0.8);
        let (px, py) = ProjectionMapper::project(x, y, z, &lens, model)
            .expect("equisolid project");
        let (dx, dy, dz) = ProjectionMapper::unproject(px, py, &lens, model);

        let orig_len = (x * x + y * y + z * z).sqrt();
        assert!(approx_eq(dx, x / orig_len, 5e-4));
        assert!(approx_eq(dy, y / orig_len, 5e-4));
        assert!(approx_eq(dz, z / orig_len, 5e-4));
    }

    // -----------------------------------------------------------------------
    // Orthographic
    // -----------------------------------------------------------------------

    #[test]
    fn test_orthographic_roundtrip() {
        let lens = default_lens();
        let model = ProjectionModel::Orthographic;
        // Keep angle < 90° so point is not behind plane.
        let (x, y, z) = (0.05, 0.05, 1.0);
        let (px, py) = ProjectionMapper::project(x, y, z, &lens, model)
            .expect("orthographic project");
        let (dx, dy, dz) = ProjectionMapper::unproject(px, py, &lens, model);

        let orig_len = (x * x + y * y + z * z).sqrt();
        assert!(approx_eq(dx, x / orig_len, 5e-4));
        assert!(approx_eq(dy, y / orig_len, 5e-4));
        assert!(approx_eq(dz, z / orig_len, 5e-4));
    }

    // -----------------------------------------------------------------------
    // Distortion / Undistortion round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_distortion_roundtrip_k1_only() {
        let k1 = -0.2_f32;
        let k2 = 0.0_f32;
        let (x, y) = (0.4, 0.3);
        let (dx, dy) = ProjectionMapper::apply_distortion(x, y, k1, k2);
        let (ux, uy) = ProjectionMapper::remove_distortion(dx, dy, k1, k2);
        assert!(approx_eq(ux, x, 1e-4), "undistort x: {ux} vs {x}");
        assert!(approx_eq(uy, y, 1e-4), "undistort y: {uy} vs {y}");
    }

    #[test]
    fn test_distortion_roundtrip_k1_k2() {
        let k1 = 0.15_f32;
        let k2 = -0.03_f32;
        let (x, y) = (-0.5, 0.25);
        let (dx, dy) = ProjectionMapper::apply_distortion(x, y, k1, k2);
        let (ux, uy) = ProjectionMapper::remove_distortion(dx, dy, k1, k2);
        assert!(approx_eq(ux, x, 1e-4), "undistort x: {ux} vs {x}");
        assert!(approx_eq(uy, y, 1e-4), "undistort y: {uy} vs {y}");
    }

    #[test]
    fn test_zero_distortion_identity() {
        let (x, y) = (0.7, -0.3);
        let (dx, dy) = ProjectionMapper::apply_distortion(x, y, 0.0, 0.0);
        assert!(approx_eq(dx, x, 1e-7));
        assert!(approx_eq(dy, y, 1e-7));
        let (ux, uy) = ProjectionMapper::remove_distortion(dx, dy, 0.0, 0.0);
        assert!(approx_eq(ux, x, 1e-7));
        assert!(approx_eq(uy, y, 1e-7));
    }

    // -----------------------------------------------------------------------
    // FOV calculator
    // -----------------------------------------------------------------------

    #[test]
    fn test_horizontal_fov_50mm_fullframe() {
        // 50 mm on 36 mm sensor → HFOV ≈ 39.6°
        let lens = LensParams::new(50.0, 36.0, 24.0, 0.0, 0.0);
        let hfov = FovCalculator::horizontal_fov(&lens);
        assert!(
            approx_eq(hfov, 39.6, 0.2),
            "Expected ~39.6°, got {hfov}"
        );
    }

    #[test]
    fn test_horizontal_fov_wide_angle() {
        // 20 mm on 36 mm sensor → HFOV ≈ 83.97°
        let lens = LensParams::new(20.0, 36.0, 24.0, 0.0, 0.0);
        let hfov = FovCalculator::horizontal_fov(&lens);
        assert!(hfov > 80.0, "Wide angle should have >80° HFOV, got {hfov}");
    }

    #[test]
    fn test_fov_proportional_to_focal_length() {
        let lens_long = LensParams::new(100.0, 36.0, 24.0, 0.0, 0.0);
        let lens_short = LensParams::new(25.0, 36.0, 24.0, 0.0, 0.0);
        let fov_long = FovCalculator::horizontal_fov(&lens_long);
        let fov_short = FovCalculator::horizontal_fov(&lens_short);
        assert!(
            fov_short > fov_long,
            "Shorter focal length should give wider FOV"
        );
    }

    #[test]
    fn test_unproject_returns_unit_vector() {
        let lens = default_lens();
        for model in [
            ProjectionModel::Rectilinear,
            ProjectionModel::Fisheye,
            ProjectionModel::Equisolid,
            ProjectionModel::Orthographic,
        ] {
            let (dx, dy, dz) = ProjectionMapper::unproject(0.1, -0.1, &lens, model);
            let len = (dx * dx + dy * dy + dz * dz).sqrt();
            assert!(
                approx_eq(len, 1.0, 1e-5),
                "Model {model:?}: unproject direction should be unit-length, got {len}"
            );
        }
    }
}
