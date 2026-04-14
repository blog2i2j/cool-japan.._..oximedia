//! Spherical projection conversions: equirectangular ↔ sphere ↔ cubemap.
//!
//! All angle values are in radians unless noted otherwise.
//! UV coordinates are normalised to 0..1 for both axes.

use std::collections::HashMap;

use crate::VrError;

// ─── Coordinate types ────────────────────────────────────────────────────────

/// A point on the unit sphere expressed in spherical coordinates.
///
/// * `azimuth_rad`   — longitude, ranging from −π (west) to +π (east).
/// * `elevation_rad` — latitude,  ranging from −π/2 (south pole) to +π/2 (north pole).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SphericalCoord {
    pub azimuth_rad: f32,
    pub elevation_rad: f32,
}

/// Normalised 2-D texture coordinate.
///
/// Both `u` and `v` are in the range `0.0 ..= 1.0`.
/// `u` increases to the right, `v` increases downward (image convention).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct UvCoord {
    pub u: f32,
    pub v: f32,
}

/// One face of a cube-map.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeFace {
    Front,
    Back,
    Left,
    Right,
    Top,
    Bottom,
}

impl CubeFace {
    /// Returns all six faces in a deterministic order.
    pub fn all() -> [CubeFace; 6] {
        [
            CubeFace::Front,
            CubeFace::Back,
            CubeFace::Left,
            CubeFace::Right,
            CubeFace::Top,
            CubeFace::Bottom,
        ]
    }
}

/// A location within one cube-map face.
///
/// `u` and `v` are face-local coordinates in `0.0 ..= 1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CubeFaceCoord {
    pub face: CubeFace,
    pub u: f32,
    pub v: f32,
}

// ─── Equirectangular ↔ sphere ────────────────────────────────────────────────

/// Convert a normalised equirectangular UV to a spherical coordinate.
///
/// Mapping:
/// * `az  = (u − 0.5) × 2π`
/// * `el  = (0.5 − v) × π`
pub fn equirect_to_sphere(uv: &UvCoord) -> SphericalCoord {
    let az = (uv.u - 0.5) * std::f32::consts::TAU;
    let el = (0.5 - uv.v) * std::f32::consts::PI;
    SphericalCoord {
        azimuth_rad: az,
        elevation_rad: el,
    }
}

/// Convert a spherical coordinate back to a normalised equirectangular UV.
pub fn sphere_to_equirect(s: &SphericalCoord) -> UvCoord {
    let u = s.azimuth_rad / std::f32::consts::TAU + 0.5;
    let v = 0.5 - s.elevation_rad / std::f32::consts::PI;
    UvCoord {
        u: u.clamp(0.0, 1.0),
        v: v.clamp(0.0, 1.0),
    }
}

// ─── Sphere ↔ cube-face ───────────────────────────────────────────────────────

/// Convert a spherical coordinate to a cube-face coordinate.
///
/// The direction vector `(x, y, z)` is computed as:
/// * `x = cos(el) × sin(az)`   (right)
/// * `y = sin(el)`              (up)
/// * `z = cos(el) × cos(az)`   (forward / into screen)
///
/// The dominant axis selects the face; the two remaining components are
/// divided by the dominant one to obtain the face-local UV.
pub fn sphere_to_cube_face(s: &SphericalCoord) -> CubeFaceCoord {
    let x = s.elevation_rad.cos() * s.azimuth_rad.sin();
    let y = s.elevation_rad.sin();
    let z = s.elevation_rad.cos() * s.azimuth_rad.cos();

    let ax = x.abs();
    let ay = y.abs();
    let az = z.abs();

    if ax >= ay && ax >= az {
        // Left / Right face — dominant X
        if x > 0.0 {
            // Right face: looking in +X direction
            // s = -z / x, t = -y / x
            let face_u = (-z / x) * 0.5 + 0.5;
            let face_v = (-y / x) * 0.5 + 0.5;
            CubeFaceCoord {
                face: CubeFace::Right,
                u: face_u.clamp(0.0, 1.0),
                v: face_v.clamp(0.0, 1.0),
            }
        } else {
            // Left face: looking in −X direction
            let face_u = (z / (-x)) * 0.5 + 0.5;
            let face_v = (-y / (-x)) * 0.5 + 0.5;
            CubeFaceCoord {
                face: CubeFace::Left,
                u: face_u.clamp(0.0, 1.0),
                v: face_v.clamp(0.0, 1.0),
            }
        }
    } else if ay >= ax && ay >= az {
        // Top / Bottom face — dominant Y
        if y > 0.0 {
            // Top face: looking in +Y direction
            let face_u = (x / y) * 0.5 + 0.5;
            let face_v = (z / y) * 0.5 + 0.5;
            CubeFaceCoord {
                face: CubeFace::Top,
                u: face_u.clamp(0.0, 1.0),
                v: face_v.clamp(0.0, 1.0),
            }
        } else {
            // Bottom face: looking in −Y direction
            let face_u = (x / (-y)) * 0.5 + 0.5;
            let face_v = (-z / (-y)) * 0.5 + 0.5;
            CubeFaceCoord {
                face: CubeFace::Bottom,
                u: face_u.clamp(0.0, 1.0),
                v: face_v.clamp(0.0, 1.0),
            }
        }
    } else {
        // Front / Back face — dominant Z
        if z > 0.0 {
            // Front face: looking in +Z direction
            let face_u = (x / z) * 0.5 + 0.5;
            let face_v = (-y / z) * 0.5 + 0.5;
            CubeFaceCoord {
                face: CubeFace::Front,
                u: face_u.clamp(0.0, 1.0),
                v: face_v.clamp(0.0, 1.0),
            }
        } else {
            // Back face: looking in −Z direction
            let face_u = (-x / (-z)) * 0.5 + 0.5;
            let face_v = (-y / (-z)) * 0.5 + 0.5;
            CubeFaceCoord {
                face: CubeFace::Back,
                u: face_u.clamp(0.0, 1.0),
                v: face_v.clamp(0.0, 1.0),
            }
        }
    }
}

/// Convert a cube-face coordinate back to a spherical coordinate.
///
/// Reconstructs the 3-D direction vector from the face and face-local UV,
/// then computes azimuth and elevation via `atan2` / `asin`.
pub fn cube_face_to_sphere(c: &CubeFaceCoord) -> SphericalCoord {
    // Map face-local UV back to [-1, 1]
    let s = c.u * 2.0 - 1.0; // horizontal
    let t = c.v * 2.0 - 1.0; // vertical

    let (x, y, z) = match c.face {
        CubeFace::Right => (1.0, -t, -s),
        CubeFace::Left => (-1.0, -t, s),
        CubeFace::Top => (s, 1.0, t),
        CubeFace::Bottom => (s, -1.0, -t),
        CubeFace::Front => (s, -t, 1.0),
        CubeFace::Back => (-s, -t, -1.0),
    };

    let len = (x * x + y * y + z * z).sqrt();
    let (nx, ny, nz) = (x / len, y / len, z / len);

    let elevation_rad = ny.asin();
    let azimuth_rad = nx.atan2(nz);

    SphericalCoord {
        azimuth_rad,
        elevation_rad,
    }
}

// ─── Bilinear sampler ────────────────────────────────────────────────────────

/// Bilinear sample from an 8-bit image buffer with edge clamping.
///
/// * `data`     — packed row-major pixel data (channels interleaved)
/// * `w`, `h`   — image dimensions in pixels
/// * `u`, `v`   — normalised sampling coordinates (0..1)
/// * `channels` — number of colour channels per pixel (e.g. 3 for RGB)
///
/// Returns a `Vec` of length `channels`.
pub fn bilinear_sample_u8(data: &[u8], w: u32, h: u32, u: f32, v: f32, channels: u32) -> Vec<u8> {
    let ch = channels as usize;
    let fw = w as f32;
    let fh = h as f32;

    // Pixel-space coordinates (centre of pixel 0 is at 0.5/w)
    let px = (u * fw - 0.5).max(0.0);
    let py = (v * fh - 0.5).max(0.0);

    let x0 = (px.floor() as u32).min(w.saturating_sub(1));
    let y0 = (py.floor() as u32).min(h.saturating_sub(1));
    let x1 = (x0 + 1).min(w.saturating_sub(1));
    let y1 = (y0 + 1).min(h.saturating_sub(1));

    let tx = px - px.floor();
    let ty = py - py.floor();

    let row_stride = w as usize * ch;

    let base_00 = y0 as usize * row_stride + x0 as usize * ch;
    let base_10 = y0 as usize * row_stride + x1 as usize * ch;
    let base_01 = y1 as usize * row_stride + x0 as usize * ch;
    let base_11 = y1 as usize * row_stride + x1 as usize * ch;

    let mut result = vec![0u8; ch];
    for c in 0..ch {
        let p00 = data[base_00 + c] as f32;
        let p10 = data[base_10 + c] as f32;
        let p01 = data[base_01 + c] as f32;
        let p11 = data[base_11 + c] as f32;

        let top = p00 + (p10 - p00) * tx;
        let bottom = p01 + (p11 - p01) * tx;
        let value = top + (bottom - top) * ty;

        result[c] = value.round().clamp(0.0, 255.0) as u8;
    }
    result
}

// ─── Equirectangular → cubemap ───────────────────────────────────────────────

/// Convert an equirectangular image to six cube-map face images.
///
/// * `src`       — source pixel data (RGB, 3 bytes per pixel, row-major)
/// * `width`     — source image width in pixels
/// * `height`    — source image height in pixels
/// * `face_size` — output cube-face size (square); each output is `face_size × face_size`
///
/// Returns a `HashMap` mapping each `CubeFace` to its RGB pixel data.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if `width`, `height`, or `face_size` is zero,
/// or if `src` is too small for the declared dimensions.
pub fn equirect_to_cube(
    src: &[u8],
    width: u32,
    height: u32,
    face_size: u32,
) -> Result<HashMap<CubeFace, Vec<u8>>, VrError> {
    validate_image_buffer(src, width, height, 3)?;
    if face_size == 0 {
        return Err(VrError::InvalidDimensions("face_size must be > 0".into()));
    }

    let channels: u32 = 3;
    let face_pixels = (face_size * face_size * channels) as usize;
    let mut map: HashMap<CubeFace, Vec<u8>> = HashMap::new();

    for face in CubeFace::all() {
        let mut face_data = vec![0u8; face_pixels];

        for fy in 0..face_size {
            for fx in 0..face_size {
                // Face-local UV
                let fu = (fx as f32 + 0.5) / face_size as f32;
                let fv = (fy as f32 + 0.5) / face_size as f32;

                let cube_coord = CubeFaceCoord { face, u: fu, v: fv };
                let sphere = cube_face_to_sphere(&cube_coord);
                let uv = sphere_to_equirect(&sphere);

                let sample = bilinear_sample_u8(src, width, height, uv.u, uv.v, channels);
                let dst_base = (fy * face_size + fx) as usize * channels as usize;
                face_data[dst_base..dst_base + channels as usize].copy_from_slice(&sample);
            }
        }

        map.insert(face, face_data);
    }

    Ok(map)
}

/// Convert six cube-map face images back to an equirectangular image.
///
/// * `faces`      — map from `CubeFace` to pixel data (RGB, 3 bpp, square, row-major)
/// * `face_size`  — side length of each cube face in pixels
/// * `out_width`  — output equirectangular image width
/// * `out_height` — output equirectangular image height
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if dimensions are zero.
/// Returns [`VrError::MissingFace`] if any of the six faces is absent from `faces`.
pub fn cube_to_equirect(
    faces: &HashMap<CubeFace, Vec<u8>>,
    face_size: u32,
    out_width: u32,
    out_height: u32,
) -> Result<Vec<u8>, VrError> {
    if face_size == 0 || out_width == 0 || out_height == 0 {
        return Err(VrError::InvalidDimensions(
            "face_size, out_width and out_height must be > 0".into(),
        ));
    }
    for face in CubeFace::all() {
        if !faces.contains_key(&face) {
            return Err(VrError::MissingFace(format!("{face:?}")));
        }
    }

    let channels: u32 = 3;
    let mut out = vec![0u8; (out_width * out_height * channels) as usize];

    for oy in 0..out_height {
        for ox in 0..out_width {
            let u = (ox as f32 + 0.5) / out_width as f32;
            let v = (oy as f32 + 0.5) / out_height as f32;

            let sphere = equirect_to_sphere(&UvCoord { u, v });
            let cfc = sphere_to_cube_face(&sphere);

            let face_data = &faces[&cfc.face];
            let sample =
                bilinear_sample_u8(face_data, face_size, face_size, cfc.u, cfc.v, channels);
            let dst_base = (oy * out_width + ox) as usize * channels as usize;
            out[dst_base..dst_base + channels as usize].copy_from_slice(&sample);
        }
    }

    Ok(out)
}

// ─── Round-trip validation utilities ─────────────────────────────────────────

/// Compute the maximum round-trip UV error for `sphere_to_equirect` followed by
/// `equirect_to_sphere` over a regular `grid_size × grid_size` sample grid.
///
/// Each sample in the grid is converted sphere → equirect → sphere, and the
/// maximum angular distance between the original and recovered spherical
/// coordinate (in radians) is returned.
///
/// This is useful as a regression test: for the standard equirectangular mapping
/// the round-trip error should be near machine epsilon for most positions.
///
/// # Errors
/// Returns [`VrError::InvalidDimensions`] if `grid_size` is zero.
pub fn sphere_equirect_max_roundtrip_error_rad(grid_size: u32) -> Result<f32, VrError> {
    if grid_size == 0 {
        return Err(VrError::InvalidDimensions("grid_size must be > 0".into()));
    }
    let n = grid_size as f32;
    let mut max_err = 0.0f32;

    for row in 0..grid_size {
        for col in 0..grid_size {
            let u = (col as f32 + 0.5) / n;
            let v = (row as f32 + 0.5) / n;
            let uv_in = UvCoord { u, v };
            let sphere = equirect_to_sphere(&uv_in);
            let uv_out = sphere_to_equirect(&sphere);

            // Measure the angular error as the great-circle distance between the
            // two spherical directions derived from the input and output UVs.
            let s_in = equirect_to_sphere(&uv_in);
            let s_out = equirect_to_sphere(&uv_out);

            let err = angular_distance_rad(&s_in, &s_out);
            if err > max_err {
                max_err = err;
            }
        }
    }
    Ok(max_err)
}

/// Compute the Peak Signal-to-Noise Ratio (PSNR) in dB between two 8-bit
/// image buffers of the same size.
///
/// PSNR = 10 · log10(255² / MSE)
///
/// Returns `f64::INFINITY` when the two images are identical (MSE == 0).
///
/// # Errors
/// Returns [`VrError::BufferTooSmall`] if `a` and `b` have different lengths
/// or are empty.
pub fn compute_psnr(a: &[u8], b: &[u8]) -> Result<f64, VrError> {
    if a.is_empty() || a.len() != b.len() {
        return Err(VrError::BufferTooSmall {
            expected: a.len(),
            got: b.len(),
        });
    }
    let mse: f64 = a
        .iter()
        .zip(b.iter())
        .map(|(&av, &bv)| {
            let diff = av as f64 - bv as f64;
            diff * diff
        })
        .sum::<f64>()
        / a.len() as f64;
    if mse < 1e-12 {
        return Ok(f64::INFINITY);
    }
    Ok(10.0 * (255.0_f64 * 255.0 / mse).log10())
}

/// Compute the great-circle angular distance (in radians) between two
/// [`SphericalCoord`] values using the haversine formula.
///
/// Numerically stable for both small and large angles.
#[inline]
pub fn angular_distance_rad(a: &SphericalCoord, b: &SphericalCoord) -> f32 {
    let dlat = b.elevation_rad - a.elevation_rad;
    let dlon = b.azimuth_rad - a.azimuth_rad;
    let h = (dlat * 0.5).sin().powi(2)
        + a.elevation_rad.cos() * b.elevation_rad.cos() * (dlon * 0.5).sin().powi(2);
    2.0 * h.sqrt().clamp(0.0, 1.0).asin()
}

// ─── Internal helpers ─────────────────────────────────────────────────────────

fn validate_image_buffer(data: &[u8], w: u32, h: u32, channels: u32) -> Result<(), VrError> {
    if w == 0 || h == 0 {
        return Err(VrError::InvalidDimensions(
            "image width and height must be > 0".into(),
        ));
    }
    let expected = w as usize * h as usize * channels as usize;
    if data.len() < expected {
        return Err(VrError::BufferTooSmall {
            expected,
            got: data.len(),
        });
    }
    Ok(())
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    const EPSILON: f32 = 0.01;

    fn uv(u: f32, v: f32) -> UvCoord {
        UvCoord { u, v }
    }

    fn sphere(az: f32, el: f32) -> SphericalCoord {
        SphericalCoord {
            azimuth_rad: az,
            elevation_rad: el,
        }
    }

    // ── equirect ↔ sphere roundtrips ─────────────────────────────────────────

    #[test]
    fn equirect_sphere_centre_roundtrip() {
        let original = uv(0.5, 0.5);
        let s = equirect_to_sphere(&original);
        let back = sphere_to_equirect(&s);
        assert!((back.u - original.u).abs() < EPSILON);
        assert!((back.v - original.v).abs() < EPSILON);
    }

    #[test]
    fn equirect_sphere_topleft_roundtrip() {
        let original = uv(0.0, 0.0);
        let s = equirect_to_sphere(&original);
        let back = sphere_to_equirect(&s);
        assert!((back.u - original.u).abs() < EPSILON);
        assert!((back.v - original.v).abs() < EPSILON);
    }

    #[test]
    fn equirect_sphere_bottomright_roundtrip() {
        let original = uv(1.0, 1.0);
        let s = equirect_to_sphere(&original);
        let back = sphere_to_equirect(&s);
        assert!((back.u - original.u).abs() < EPSILON);
        assert!((back.v - original.v).abs() < EPSILON);
    }

    #[test]
    fn equirect_to_sphere_centre() {
        let s = equirect_to_sphere(&uv(0.5, 0.5));
        assert!(s.azimuth_rad.abs() < EPSILON);
        assert!(s.elevation_rad.abs() < EPSILON);
    }

    #[test]
    fn equirect_to_sphere_left_edge() {
        let s = equirect_to_sphere(&uv(0.0, 0.5));
        assert!((s.azimuth_rad + PI).abs() < EPSILON);
        assert!(s.elevation_rad.abs() < EPSILON);
    }

    #[test]
    fn equirect_to_sphere_right_edge() {
        let s = equirect_to_sphere(&uv(1.0, 0.5));
        assert!((s.azimuth_rad - PI).abs() < EPSILON);
        assert!(s.elevation_rad.abs() < EPSILON);
    }

    #[test]
    fn equirect_to_sphere_top_edge() {
        let s = equirect_to_sphere(&uv(0.5, 0.0));
        assert!(s.elevation_rad > 0.0);
    }

    #[test]
    fn equirect_to_sphere_bottom_edge() {
        let s = equirect_to_sphere(&uv(0.5, 1.0));
        assert!(s.elevation_rad < 0.0);
    }

    #[test]
    fn sphere_to_equirect_origin() {
        let uvc = sphere_to_equirect(&sphere(0.0, 0.0));
        assert!((uvc.u - 0.5).abs() < EPSILON);
        assert!((uvc.v - 0.5).abs() < EPSILON);
    }

    // ── sphere ↔ cube-face roundtrips ────────────────────────────────────────

    fn roundtrip_sphere_cube(az: f32, el: f32) {
        let s = sphere(az, el);
        let c = sphere_to_cube_face(&s);
        let back = cube_face_to_sphere(&c);
        assert!(
            (back.azimuth_rad - s.azimuth_rad).abs() < EPSILON
                || (back.azimuth_rad - s.azimuth_rad + 2.0 * PI).abs() < EPSILON
                || (back.azimuth_rad - s.azimuth_rad - 2.0 * PI).abs() < EPSILON,
            "azimuth mismatch: in={:.4} out={:.4}",
            s.azimuth_rad,
            back.azimuth_rad
        );
        assert!(
            (back.elevation_rad - s.elevation_rad).abs() < EPSILON,
            "elevation mismatch: in={:.4} out={:.4}",
            s.elevation_rad,
            back.elevation_rad
        );
    }

    #[test]
    fn roundtrip_front() {
        roundtrip_sphere_cube(0.0, 0.0);
    }

    #[test]
    fn roundtrip_right() {
        roundtrip_sphere_cube(PI / 2.0, 0.0);
    }

    #[test]
    fn roundtrip_back() {
        roundtrip_sphere_cube(PI * 0.9, 0.0);
    }

    #[test]
    fn roundtrip_left() {
        roundtrip_sphere_cube(-PI / 2.0, 0.0);
    }

    #[test]
    fn roundtrip_top() {
        roundtrip_sphere_cube(0.0, PI / 2.0 * 0.9);
    }

    #[test]
    fn roundtrip_bottom() {
        roundtrip_sphere_cube(0.0, -PI / 2.0 * 0.9);
    }

    #[test]
    fn roundtrip_diagonal() {
        roundtrip_sphere_cube(PI / 4.0, PI / 6.0);
    }

    #[test]
    fn sphere_to_cube_front_face() {
        let s = sphere(0.0, 0.0);
        let c = sphere_to_cube_face(&s);
        assert_eq!(c.face, CubeFace::Front);
    }

    #[test]
    fn sphere_to_cube_right_face() {
        let s = sphere(PI / 2.0, 0.0);
        let c = sphere_to_cube_face(&s);
        assert_eq!(c.face, CubeFace::Right);
    }

    #[test]
    fn sphere_to_cube_left_face() {
        let s = sphere(-PI / 2.0, 0.0);
        let c = sphere_to_cube_face(&s);
        assert_eq!(c.face, CubeFace::Left);
    }

    #[test]
    fn sphere_to_cube_top_face() {
        let s = sphere(0.0, PI / 2.0 * 0.95);
        let c = sphere_to_cube_face(&s);
        assert_eq!(c.face, CubeFace::Top);
    }

    #[test]
    fn sphere_to_cube_bottom_face() {
        let s = sphere(0.0, -PI / 2.0 * 0.95);
        let c = sphere_to_cube_face(&s);
        assert_eq!(c.face, CubeFace::Bottom);
    }

    // ── bilinear sampler ─────────────────────────────────────────────────────

    #[test]
    fn bilinear_solid_colour() {
        // A 4×4 solid-red image should return red everywhere
        let img: Vec<u8> = (0..4 * 4 * 3)
            .map(|i| if i % 3 == 0 { 255 } else { 0 })
            .collect();
        let p = bilinear_sample_u8(&img, 4, 4, 0.5, 0.5, 3);
        assert_eq!(p[0], 255);
        assert_eq!(p[1], 0);
        assert_eq!(p[2], 0);
    }

    #[test]
    fn bilinear_corner_clamp() {
        let img = vec![128u8; 2 * 2 * 3];
        let p = bilinear_sample_u8(&img, 2, 2, 0.0, 0.0, 3);
        assert_eq!(p.len(), 3);
        assert_eq!(p[0], 128);
    }

    #[test]
    fn bilinear_exact_pixel() {
        // 2×1 image: left pixel = 0, right pixel = 255 (R channel)
        let img = vec![0u8, 0, 0, 255, 0, 0];
        let left = bilinear_sample_u8(&img, 2, 1, 0.25, 0.5, 3);
        let right = bilinear_sample_u8(&img, 2, 1, 0.75, 0.5, 3);
        assert!(left[0] < 128, "left pixel should be near 0");
        assert!(right[0] > 128, "right pixel should be near 255");
    }

    // ── equirect_to_cube / cube_to_equirect ──────────────────────────────────

    fn solid_equirect(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = Vec::with_capacity((w * h * 3) as usize);
        for _ in 0..(w * h) {
            v.push(r);
            v.push(g);
            v.push(b);
        }
        v
    }

    #[test]
    fn equirect_to_cube_produces_six_faces() {
        let src = solid_equirect(64, 32, 128, 64, 32);
        let faces = equirect_to_cube(&src, 64, 32, 8).expect("equirect_to_cube failed");
        assert_eq!(faces.len(), 6);
        for face in CubeFace::all() {
            assert!(faces.contains_key(&face));
            assert_eq!(faces[&face].len(), 8 * 8 * 3);
        }
    }

    #[test]
    fn equirect_to_cube_invalid_zero_face_size() {
        let src = solid_equirect(64, 32, 0, 0, 0);
        let result = equirect_to_cube(&src, 64, 32, 0);
        assert!(result.is_err());
    }

    #[test]
    fn equirect_to_cube_invalid_buffer_too_small() {
        let result = equirect_to_cube(&[0u8; 10], 64, 32, 8);
        assert!(result.is_err());
    }

    #[test]
    fn cube_to_equirect_produces_correct_size() {
        let src = solid_equirect(64, 32, 200, 100, 50);
        let faces = equirect_to_cube(&src, 64, 32, 16).expect("equirect_to_cube failed");
        let out = cube_to_equirect(&faces, 16, 64, 32).expect("cube_to_equirect failed");
        assert_eq!(out.len(), 64 * 32 * 3);
    }

    #[test]
    fn cube_to_equirect_missing_face_error() {
        let mut faces: HashMap<CubeFace, Vec<u8>> = HashMap::new();
        // Add only 5 faces (omit Back)
        for face in [
            CubeFace::Front,
            CubeFace::Left,
            CubeFace::Right,
            CubeFace::Top,
            CubeFace::Bottom,
        ] {
            faces.insert(face, vec![0u8; 8 * 8 * 3]);
        }
        let result = cube_to_equirect(&faces, 8, 32, 16);
        assert!(result.is_err());
    }

    #[test]
    fn cube_to_equirect_zero_dimensions_error() {
        let faces: HashMap<CubeFace, Vec<u8>> = HashMap::new();
        assert!(cube_to_equirect(&faces, 0, 32, 16).is_err());
        assert!(cube_to_equirect(&faces, 8, 0, 16).is_err());
        assert!(cube_to_equirect(&faces, 8, 32, 0).is_err());
    }

    #[test]
    fn cube_face_all_returns_six() {
        assert_eq!(CubeFace::all().len(), 6);
    }

    #[test]
    fn equirect_cube_equirect_roundtrip_colour_preservation() {
        // A uniform-colour equirect should mostly survive a round-trip
        let src = solid_equirect(128, 64, 180, 120, 60);
        let faces = equirect_to_cube(&src, 128, 64, 32).expect("to cube");
        let out = cube_to_equirect(&faces, 32, 128, 64).expect("to equirect");

        // Sample the centre pixel — should be close to input colour
        let cx = 64usize;
        let cy = 32usize;
        let base = (cy * 128 + cx) * 3;
        let err_r = (out[base] as i32 - 180).abs();
        let err_g = (out[base + 1] as i32 - 120).abs();
        let err_b = (out[base + 2] as i32 - 60).abs();
        assert!(err_r <= 5, "R error too large: {err_r}");
        assert!(err_g <= 5, "G error too large: {err_g}");
        assert!(err_b <= 5, "B error too large: {err_b}");
    }

    // ── Polar singularity edge cases ─────────────────────────────────────────

    #[test]
    fn north_pole_elevation_clamped_at_half_pi() {
        // v=0 → elevation = π/2 (north pole)
        let s = equirect_to_sphere(&uv(0.5, 0.0));
        assert!(
            (s.elevation_rad - PI / 2.0).abs() < 0.01,
            "el={}",
            s.elevation_rad
        );
    }

    #[test]
    fn south_pole_elevation_clamped_at_neg_half_pi() {
        // v=1 → elevation = −π/2 (south pole)
        let s = equirect_to_sphere(&uv(0.5, 1.0));
        assert!(
            (s.elevation_rad + PI / 2.0).abs() < 0.01,
            "el={}",
            s.elevation_rad
        );
    }

    #[test]
    fn pole_roundtrip_does_not_panic() {
        // Both poles: equirect → sphere → equirect should be stable
        for &v_val in &[0.0f32, 1.0f32] {
            for &u_val in &[0.0f32, 0.25, 0.5, 0.75, 1.0] {
                let uv_in = uv(u_val, v_val);
                let s = equirect_to_sphere(&uv_in);
                let _ = sphere_to_equirect(&s);
            }
        }
    }

    #[test]
    fn antimeridian_roundtrip_stable() {
        // Pixels at u ≈ 0 and u ≈ 1 should produce the same sphere direction
        // (the antimeridian wrap)
        let s0 = equirect_to_sphere(&uv(0.0, 0.5));
        let s1 = equirect_to_sphere(&uv(1.0, 0.5));
        // Both give az ≈ ±π which differ by 2π → same physical direction
        let az_diff = (s0.azimuth_rad.abs() - PI).abs();
        let az_diff1 = (s1.azimuth_rad.abs() - PI).abs();
        assert!(az_diff < 0.01, "az0={}", s0.azimuth_rad);
        assert!(az_diff1 < 0.01, "az1={}", s1.azimuth_rad);
    }

    #[test]
    fn cubemap_roundtrip_at_poles_does_not_panic() {
        // Ensure equirect_to_cube handles polar pixels without panicking
        let src = solid_equirect(64, 32, 100, 150, 200);
        let faces = equirect_to_cube(&src, 64, 32, 8).expect("to cube");
        for face in CubeFace::all() {
            let face_data = &faces[&face];
            assert_eq!(face_data.len(), 8 * 8 * 3, "face {face:?} wrong size");
        }
    }

    // ── Round-trip PSNR test ─────────────────────────────────────────────────

    #[test]
    fn equirect_cube_equirect_psnr_above_threshold() {
        // A uniform-colour panorama through equirect→cube→equirect should
        // achieve PSNR ≥ 30 dB at the sampled pixels (ignoring poles).
        let src = solid_equirect(128, 64, 200, 100, 50);
        let faces = equirect_to_cube(&src, 128, 64, 32).expect("to cube");
        let out = cube_to_equirect(&faces, 32, 128, 64).expect("to equirect");
        let psnr = compute_psnr(&src, &out).expect("psnr");
        assert!(psnr >= 30.0, "PSNR too low: {psnr:.1} dB");
    }

    #[test]
    fn psnr_identical_images_is_infinite() {
        let img = solid_equirect(32, 16, 128, 64, 32);
        let psnr = compute_psnr(&img, &img).expect("psnr");
        assert!(psnr.is_infinite() && psnr > 0.0);
    }

    #[test]
    fn psnr_different_lengths_error() {
        let a = vec![0u8; 10];
        let b = vec![0u8; 20];
        assert!(compute_psnr(&a, &b).is_err());
    }

    #[test]
    fn psnr_empty_error() {
        assert!(compute_psnr(&[], &[]).is_err());
    }

    #[test]
    fn psnr_max_difference_below_9_db() {
        // All-zeros vs all-255: PSNR = 10*log10(255²/255²) = 0 dB? No:
        // MSE = 255², PSNR = 10*log10(1.0) = 0 dB — worst case.
        let a = vec![0u8; 8 * 8 * 3];
        let b = vec![255u8; 8 * 8 * 3];
        let psnr = compute_psnr(&a, &b).expect("psnr");
        // PSNR = 10 * log10(255^2 / 255^2) = 10 * log10(1) = 0 dB
        assert!((psnr - 0.0).abs() < 0.01, "psnr={psnr}");
    }

    // ── Round-trip validation utility ────────────────────────────────────────

    #[test]
    fn sphere_equirect_roundtrip_error_near_zero() {
        let max_err = sphere_equirect_max_roundtrip_error_rad(16).expect("ok");
        // Standard equirectangular mapping is lossless to floating-point precision
        // (error should be sub-milliradian)
        assert!(max_err < 0.001, "max_err={max_err} rad");
    }

    #[test]
    fn sphere_equirect_roundtrip_error_zero_grid_size_error() {
        assert!(sphere_equirect_max_roundtrip_error_rad(0).is_err());
    }

    // ── angular_distance_rad ─────────────────────────────────────────────────

    #[test]
    fn angular_distance_same_point_is_zero() {
        let s = sphere(PI / 4.0, PI / 6.0);
        let d = angular_distance_rad(&s, &s);
        assert!(d < 1e-6, "d={d}");
    }

    #[test]
    fn angular_distance_antipodal_is_pi() {
        let a = sphere(0.0, 0.0);
        let b = sphere(PI, 0.0);
        let d = angular_distance_rad(&a, &b);
        assert!((d - PI).abs() < 0.05, "d={d}");
    }
}
