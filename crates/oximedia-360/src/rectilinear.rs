//! Rectilinear (perspective) sub-region extraction from equirectangular panoramas.
//!
//! This module provides a [`VirtualCamera`] model and the
//! [`extract_rectilinear`] function to render an arbitrary perspective view
//! from an equirectangular 360° image.  The virtual camera is defined by its
//! horizontal and vertical fields of view, and an orientation described by
//! yaw, pitch, and roll angles.
//!
//! ## Coordinate conventions
//!
//! * **Yaw** — rotation around the global Y (up) axis.  Positive yaw rotates
//!   the camera to the right (East).
//! * **Pitch** — rotation around the camera-local X axis after yaw.  Positive
//!   pitch tilts the camera upward.
//! * **Roll** — rotation around the camera-local Z (forward) axis.  Positive
//!   roll rotates the image clockwise.
//!
//! All angles are in **degrees**.
//!
//! ## Usage
//!
//! ```rust
//! use oximedia_360::rectilinear::{VirtualCamera, extract_rectilinear};
//!
//! // A tiny solid-colour equirectangular image (2×1 pixels, RGB)
//! let equirect = vec![100u8, 150, 200,  100u8, 150, 200];
//! let camera = VirtualCamera {
//!     fov_h_deg: 90.0,
//!     fov_v_deg: 60.0,
//!     yaw_deg:   0.0,
//!     pitch_deg: 0.0,
//!     roll_deg:  0.0,
//! };
//! let view = extract_rectilinear(&equirect, 2, 1, &camera, 4, 3).expect("ok");
//! assert_eq!(view.len(), 4 * 3 * 3);
//! ```

use crate::{
    projection::{bilinear_sample_u8, sphere_to_equirect, SphericalCoord},
    VrError,
};

// ─── RectilinearProjection ───────────────────────────────────────────────────

/// A rectilinear (perspective) projection model parameterised by a single
/// horizontal field-of-view angle.
///
/// This struct converts 2-D image coordinates (in the range `[−1, +1]` for
/// both axes, with the origin at the image centre) to equirectangular UV
/// coordinates on the sphere.
///
/// ## Usage
///
/// ```rust
/// use oximedia_360::rectilinear::RectilinearProjection;
///
/// let proj = RectilinearProjection::new(90.0);
/// // Centre of the image maps to (0.5, 0.5) in equirect UV (forward direction)
/// let (u, v) = proj.to_equirect(0.0, 0.0);
/// assert!((u - 0.5).abs() < 0.01, "u={u}");
/// assert!((v - 0.5).abs() < 0.01, "v={v}");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RectilinearProjection {
    /// Horizontal field of view in degrees.
    pub fov_deg: f32,
    /// Precomputed half-tan of the horizontal FOV.
    half_tan_hfov: f32,
}

impl RectilinearProjection {
    /// Create a new rectilinear projection with the given horizontal FOV.
    ///
    /// # Parameters
    ///
    /// * `fov_deg` — horizontal field of view in degrees; must be in `(0°, 180°)`.
    ///   Values outside this range are clamped to a valid range.
    #[must_use]
    pub fn new(fov_deg: f32) -> Self {
        let clamped = fov_deg.clamp(1.0, 179.0);
        let half_tan = (clamped * 0.5).to_radians().tan();
        Self {
            fov_deg: clamped,
            half_tan_hfov: half_tan,
        }
    }

    /// Convert a normalised 2-D image coordinate to equirectangular UV.
    ///
    /// The input coordinates `(x, y)` are in the range `[−1, +1]`, where
    /// `(0, 0)` is the image centre, `(−1, 0)` is the left edge, and
    /// `(0, −1)` is the bottom edge.
    ///
    /// The camera is assumed to look along the positive Z axis with no
    /// orientation offset (yaw = 0, pitch = 0, roll = 0).  For a rotated
    /// camera, use [`extract_rectilinear`] with a configured [`VirtualCamera`].
    ///
    /// # Returns
    ///
    /// `(u, v)` — equirectangular UV in `[0, 1]`, where `(0.5, 0.5)` is the
    /// forward direction (azimuth = 0, elevation = 0).
    #[must_use]
    pub fn to_equirect(&self, x: f32, y: f32) -> (f32, f32) {
        // Ray direction in camera space: Z = 1 (looking forward)
        let rx = x * self.half_tan_hfov;
        let ry = y * self.half_tan_hfov;
        let rz = 1.0f32;

        // Normalise
        let len = (rx * rx + ry * ry + rz * rz).sqrt();
        if len < f32::EPSILON {
            return (0.5, 0.5);
        }
        let (nx, ny, nz) = (rx / len, ry / len, rz / len);

        // Convert to spherical (same convention as `sphere_to_equirect`)
        let elevation = ny.clamp(-1.0, 1.0).asin();
        let azimuth = nx.atan2(nz);

        let sphere = SphericalCoord {
            azimuth_rad: azimuth,
            elevation_rad: elevation,
        };
        let uv = sphere_to_equirect(&sphere);
        (uv.u, uv.v)
    }

    /// Vertical field of view in degrees for a given output aspect ratio.
    ///
    /// # Parameters
    ///
    /// * `aspect` — width / height ratio of the output image (e.g. 16.0/9.0)
    #[must_use]
    pub fn vfov_deg(&self, aspect: f32) -> f32 {
        if aspect <= 0.0 {
            return 0.0;
        }
        let vfov_rad = 2.0 * (self.half_tan_hfov / aspect).atan();
        vfov_rad.to_degrees()
    }
}

// ─── VirtualCamera ───────────────────────────────────────────────────────────

/// Parameters for a rectilinear (perspective) virtual camera embedded in a
/// 360° equirectangular panorama.
///
/// The camera looks toward the sphere's forward direction by default (az = 0,
/// el = 0).  Orientation is applied in yaw → pitch → roll order.
#[derive(Debug, Clone, PartialEq)]
pub struct VirtualCamera {
    /// Horizontal field of view in degrees.
    pub fov_h_deg: f32,
    /// Vertical field of view in degrees.
    pub fov_v_deg: f32,
    /// Yaw (azimuth) rotation in degrees.  Positive = rotate right / East.
    pub yaw_deg: f32,
    /// Pitch rotation in degrees.  Positive = tilt upward.
    pub pitch_deg: f32,
    /// Roll rotation in degrees.  Positive = clockwise roll.
    pub roll_deg: f32,
}

impl VirtualCamera {
    /// Convenience constructor: 90°×60° FOV, looking straight ahead.
    pub fn default_forward() -> Self {
        Self {
            fov_h_deg: 90.0,
            fov_v_deg: 60.0,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        }
    }
}

// ─── 3-D rotation matrix ──────────────────────────────────────────────────────

/// 3×3 rotation matrix stored in row-major order.
struct RotMat3x3([[f32; 3]; 3]);

impl RotMat3x3 {
    /// Identity rotation.
    #[allow(dead_code)]
    fn identity() -> Self {
        RotMat3x3([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Rotation around the Y axis by angle `theta` (radians).
    fn rot_y(theta: f32) -> Self {
        let (s, c) = theta.sin_cos();
        RotMat3x3([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    }

    /// Rotation around the X axis by angle `theta` (radians).
    fn rot_x(theta: f32) -> Self {
        let (s, c) = theta.sin_cos();
        RotMat3x3([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    }

    /// Rotation around the Z axis by angle `theta` (radians).
    fn rot_z(theta: f32) -> Self {
        let (s, c) = theta.sin_cos();
        RotMat3x3([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    }

    /// Matrix multiplication: `self × other`.
    fn mul(&self, other: &Self) -> Self {
        let a = &self.0;
        let b = &other.0;
        let mut r = [[0.0f32; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    r[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        RotMat3x3(r)
    }

    /// Apply this rotation to a 3-D vector.
    #[inline]
    fn apply(&self, v: [f32; 3]) -> [f32; 3] {
        let m = &self.0;
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }
}

// ─── extract_rectilinear ─────────────────────────────────────────────────────

/// Render a rectilinear (perspective) view from an equirectangular panorama.
///
/// For each output pixel `(u, v)` the function:
/// 1. Computes the 3-D ray direction from a perspective projection model using
///    the camera's horizontal and vertical FOV.
/// 2. Rotates the ray by the camera's yaw → pitch → roll orientation.
/// 3. Converts the rotated direction to equirectangular UV coordinates.
/// 4. Samples the equirectangular source image with bilinear interpolation.
///
/// # Parameters
///
/// * `equirect`    — source equirectangular image (RGB, 3 bpp, row-major)
/// * `equirect_w`  — source image width in pixels
/// * `equirect_h`  — source image height in pixels
/// * `camera`      — virtual camera parameters
/// * `out_w`       — output image width in pixels
/// * `out_h`       — output image height in pixels
///
/// # Errors
///
/// Returns [`VrError::InvalidDimensions`] if any dimension is zero or if
/// `equirect` is too small for the declared dimensions.
pub fn extract_rectilinear(
    equirect: &[u8],
    equirect_w: u32,
    equirect_h: u32,
    camera: &VirtualCamera,
    out_w: u32,
    out_h: u32,
) -> Result<Vec<u8>, VrError> {
    if equirect_w == 0 || equirect_h == 0 || out_w == 0 || out_h == 0 {
        return Err(VrError::InvalidDimensions(
            "all dimensions must be > 0".into(),
        ));
    }
    let min_required = equirect_w as usize * equirect_h as usize * 3;
    if equirect.len() < min_required {
        return Err(VrError::BufferTooSmall {
            expected: min_required,
            got: equirect.len(),
        });
    }

    // Pre-compute camera rotation matrix: yaw (Y) → pitch (X) → roll (Z)
    // Positive pitch = tilt upward → negate angle for rot_x so that +pitch raises the view.
    let r_yaw = RotMat3x3::rot_y(camera.yaw_deg.to_radians());
    let r_pitch = RotMat3x3::rot_x(-camera.pitch_deg.to_radians());
    let r_roll = RotMat3x3::rot_z(camera.roll_deg.to_radians());
    // Combined: R = R_yaw * R_pitch * R_roll
    let rot = r_yaw.mul(&r_pitch).mul(&r_roll);

    // Focal lengths from FOV (perspective projection)
    let half_fov_h = (camera.fov_h_deg * 0.5).to_radians().tan();
    let half_fov_v = (camera.fov_v_deg * 0.5).to_radians().tan();

    const CH: u32 = 3;
    let mut out = vec![0u8; (out_w * out_h * CH) as usize];

    for oy in 0..out_h {
        for ox in 0..out_w {
            // Normalised device coordinates in [-1, +1]
            let ndx = (ox as f32 + 0.5) / out_w as f32 * 2.0 - 1.0;
            let ndy = 1.0 - (oy as f32 + 0.5) / out_h as f32 * 2.0; // Y up

            // Ray direction in camera space (Z forward, X right, Y up)
            let ray_cam = [ndx * half_fov_h, ndy * half_fov_v, 1.0f32];

            // Rotate ray to world space
            let [rx, ry, rz] = rot.apply(ray_cam);

            // Normalise
            let len = (rx * rx + ry * ry + rz * rz).sqrt();
            if len < f32::EPSILON {
                continue;
            }
            let (nx, ny, nz) = (rx / len, ry / len, rz / len);

            // Convert direction to spherical coordinates
            // Convention: x = cos(el)*sin(az), y = sin(el), z = cos(el)*cos(az)
            let elevation = ny.clamp(-1.0, 1.0).asin();
            let azimuth = nx.atan2(nz);

            let sphere = SphericalCoord {
                azimuth_rad: azimuth,
                elevation_rad: elevation,
            };
            let uv = sphere_to_equirect(&sphere);

            let sample = bilinear_sample_u8(equirect, equirect_w, equirect_h, uv.u, uv.v, CH);
            let dst = (oy * out_w + ox) as usize * CH as usize;
            out[dst..dst + CH as usize].copy_from_slice(&sample);
        }
    }

    Ok(out)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn solid_equirect(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    // ── RectilinearProjection ────────────────────────────────────────────────

    #[test]
    fn rectilinear_centre_maps_to_equirect_centre() {
        let proj = RectilinearProjection::new(90.0);
        let (u, v) = proj.to_equirect(0.0, 0.0);
        assert!((u - 0.5).abs() < 0.01, "u={u}");
        assert!((v - 0.5).abs() < 0.01, "v={v}");
    }

    #[test]
    fn rectilinear_fov_clamped_to_valid_range() {
        let p = RectilinearProjection::new(200.0);
        assert!(p.fov_deg <= 179.0, "fov should be clamped");

        let q = RectilinearProjection::new(-10.0);
        assert!(q.fov_deg >= 1.0, "fov should be clamped");
    }

    #[test]
    fn rectilinear_vfov_less_than_hfov_for_widescreen() {
        // For a 16:9 aspect ratio, vfov should be less than hfov
        let proj = RectilinearProjection::new(90.0);
        let vfov = proj.vfov_deg(16.0 / 9.0);
        assert!(vfov < 90.0, "vfov={vfov} should be < 90 for 16:9");
        assert!(vfov > 0.0, "vfov={vfov} should be positive");
    }

    #[test]
    fn rectilinear_symmetric_lr_and_tb() {
        let proj = RectilinearProjection::new(90.0);
        let (u_right, _) = proj.to_equirect(1.0, 0.0);
        let (u_left, _) = proj.to_equirect(-1.0, 0.0);
        // Left and right should be symmetric around 0.5
        let centre = 0.5f32;
        assert!(
            (u_right - centre).abs() - (u_left - centre).abs() < 0.01,
            "u_right={u_right}, u_left={u_left}"
        );
    }

    // ── VirtualCamera ────────────────────────────────────────────────────────

    #[test]
    fn virtual_camera_default_forward_fields() {
        let cam = VirtualCamera::default_forward();
        assert!((cam.fov_h_deg - 90.0).abs() < 1e-6);
        assert!((cam.fov_v_deg - 60.0).abs() < 1e-6);
        assert!((cam.yaw_deg).abs() < 1e-6);
        assert!((cam.pitch_deg).abs() < 1e-6);
        assert!((cam.roll_deg).abs() < 1e-6);
    }

    // ── extract_rectilinear ──────────────────────────────────────────────────

    #[test]
    fn extract_rectilinear_output_size() {
        let src = solid_equirect(64, 32, 128, 64, 32);
        let cam = VirtualCamera::default_forward();
        let result =
            extract_rectilinear(&src, 64, 32, &cam, 32, 24).expect("extract_rectilinear failed");
        assert_eq!(result.len(), 32 * 24 * 3);
    }

    #[test]
    fn extract_rectilinear_solid_colour_preserved() {
        // A solid-colour panorama should produce the same colour in any view
        let src = solid_equirect(64, 32, 180, 120, 60);
        let cam = VirtualCamera::default_forward();
        let result = extract_rectilinear(&src, 64, 32, &cam, 16, 12).expect("extract failed");

        // Check the centre pixel
        let cx = 8usize;
        let cy = 6usize;
        let base = (cy * 16 + cx) * 3;
        assert_eq!(result[base], 180, "R mismatch");
        assert_eq!(result[base + 1], 120, "G mismatch");
        assert_eq!(result[base + 2], 60, "B mismatch");
    }

    #[test]
    fn extract_rectilinear_yaw_rotates_view() {
        // Create a panorama where the left half is red and the right half is blue.
        let w = 64u32;
        let h = 32u32;
        let mut src = vec![0u8; (w * h * 3) as usize];
        for row in 0..h as usize {
            for col in 0..w as usize {
                let base = (row * w as usize + col) * 3;
                if col < w as usize / 2 {
                    src[base] = 200; // left = red
                } else {
                    src[base + 2] = 200; // right = blue
                }
            }
        }

        // Looking forward (yaw=0) → centre view should see boundary area
        let cam_center = VirtualCamera {
            fov_h_deg: 60.0,
            fov_v_deg: 45.0,
            yaw_deg: 0.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        };
        // Looking 90° right (yaw=90) → should see more blue
        let cam_right = VirtualCamera {
            fov_h_deg: 60.0,
            fov_v_deg: 45.0,
            yaw_deg: 90.0,
            pitch_deg: 0.0,
            roll_deg: 0.0,
        };
        let out_w = 16u32;
        let out_h = 12u32;

        let view_center =
            extract_rectilinear(&src, w, h, &cam_center, out_w, out_h).expect("center view");
        let view_right =
            extract_rectilinear(&src, w, h, &cam_right, out_w, out_h).expect("right view");

        let mean_blue_center: f32 = view_center
            .chunks_exact(3)
            .map(|p| p[2] as f32)
            .sum::<f32>()
            / (out_w * out_h) as f32;
        let mean_blue_right: f32 =
            view_right.chunks_exact(3).map(|p| p[2] as f32).sum::<f32>() / (out_w * out_h) as f32;

        assert!(
            mean_blue_right > mean_blue_center,
            "yaw=90 should see more blue: center={mean_blue_center:.1}, right={mean_blue_right:.1}"
        );
    }

    #[test]
    fn extract_rectilinear_error_on_zero_dims() {
        let src = solid_equirect(4, 2, 0, 0, 0);
        // out_w = 0
        assert!(extract_rectilinear(&src, 4, 2, &VirtualCamera::default_forward(), 0, 4).is_err());
        // equirect_h = 0
        assert!(extract_rectilinear(&src, 4, 0, &VirtualCamera::default_forward(), 4, 4).is_err());
    }

    #[test]
    fn extract_rectilinear_error_on_buffer_too_small() {
        let src = vec![0u8; 6]; // too small for 4×2×3
        assert!(extract_rectilinear(&src, 4, 2, &VirtualCamera::default_forward(), 2, 2).is_err());
    }

    #[test]
    fn extract_rectilinear_pitch_changes_view() {
        // A panorama: top half white, bottom half black
        let w = 64u32;
        let h = 32u32;
        let mut src = vec![0u8; (w * h * 3) as usize];
        for row in 0..h as usize / 2 {
            for col in 0..w as usize {
                let base = (row * w as usize + col) * 3;
                src[base] = 255;
                src[base + 1] = 255;
                src[base + 2] = 255;
            }
        }

        let cam_up = VirtualCamera {
            fov_h_deg: 60.0,
            fov_v_deg: 45.0,
            yaw_deg: 0.0,
            pitch_deg: 60.0,
            roll_deg: 0.0,
        };
        let cam_down = VirtualCamera {
            fov_h_deg: 60.0,
            fov_v_deg: 45.0,
            yaw_deg: 0.0,
            pitch_deg: -60.0,
            roll_deg: 0.0,
        };

        let out_w = 16u32;
        let out_h = 12u32;

        let view_up = extract_rectilinear(&src, w, h, &cam_up, out_w, out_h).expect("view up");
        let view_down =
            extract_rectilinear(&src, w, h, &cam_down, out_w, out_h).expect("view down");

        let mean_bright_up: f32 =
            view_up.iter().map(|&v| v as f32).sum::<f32>() / view_up.len() as f32;
        let mean_bright_down: f32 =
            view_down.iter().map(|&v| v as f32).sum::<f32>() / view_down.len() as f32;

        assert!(
            mean_bright_up > mean_bright_down + 20.0,
            "pitch up should be brighter: up={mean_bright_up:.1}, down={mean_bright_down:.1}"
        );
    }
}
