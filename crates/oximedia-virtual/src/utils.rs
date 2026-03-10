//! Utility functions and helpers for virtual production
//!
//! Provides common utility functions, conversions, and helpers.

use crate::math::{Matrix4, Point3, Quaternion, UnitQuaternion, Vector3};
use std::time::{SystemTime, UNIX_EPOCH};

/// Convert timestamp to nanoseconds since epoch
#[must_use]
pub fn system_time_to_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

/// Convert nanoseconds to seconds
#[must_use]
pub fn nanos_to_seconds(nanos: u64) -> f64 {
    nanos as f64 / 1_000_000_000.0
}

/// Convert seconds to nanoseconds
#[must_use]
pub fn seconds_to_nanos(seconds: f64) -> u64 {
    (seconds * 1_000_000_000.0) as u64
}

/// Linear interpolation
#[must_use]
pub fn lerp(a: f64, b: f64, t: f64) -> f64 {
    a + (b - a) * t
}

/// Clamp value to range
#[must_use]
pub fn clamp<T: PartialOrd>(value: T, min: T, max: T) -> T {
    if value < min {
        min
    } else if value > max {
        max
    } else {
        value
    }
}

/// Convert degrees to radians
#[must_use]
pub fn deg_to_rad(degrees: f64) -> f64 {
    degrees * std::f64::consts::PI / 180.0
}

/// Convert radians to degrees
#[must_use]
pub fn rad_to_deg(radians: f64) -> f64 {
    radians * 180.0 / std::f64::consts::PI
}

/// Create look-at matrix
#[must_use]
pub fn look_at_matrix(eye: &Point3<f64>, target: &Point3<f64>, up: &Vector3<f64>) -> Matrix4<f64> {
    let f = (target - eye).normalize();
    let s = f.cross(up).normalize();
    let u = s.cross(&f);

    let mut result = Matrix4::identity();
    result[(0, 0)] = s.x;
    result[(0, 1)] = s.y;
    result[(0, 2)] = s.z;
    result[(1, 0)] = u.x;
    result[(1, 1)] = u.y;
    result[(1, 2)] = u.z;
    result[(2, 0)] = -f.x;
    result[(2, 1)] = -f.y;
    result[(2, 2)] = -f.z;
    result[(0, 3)] = -s.dot(&eye.coords());
    result[(1, 3)] = -u.dot(&eye.coords());
    result[(2, 3)] = f.dot(&eye.coords());

    result
}

/// Create perspective projection matrix
#[must_use]
pub fn perspective_matrix(fov_y: f64, aspect: f64, near: f64, far: f64) -> Matrix4<f64> {
    let f = 1.0 / (fov_y / 2.0).tan();

    let mut result = Matrix4::zeros();
    result[(0, 0)] = f / aspect;
    result[(1, 1)] = f;
    result[(2, 2)] = (far + near) / (near - far);
    result[(2, 3)] = (2.0 * far * near) / (near - far);
    result[(3, 2)] = -1.0;

    result
}

/// Convert quaternion to Euler angles (pitch, yaw, roll)
#[must_use]
pub fn quaternion_to_euler(q: &Quaternion<f64>) -> (f64, f64, f64) {
    let unit_q = UnitQuaternion::from_quaternion(*q);
    let euler = unit_q.euler_angles();
    (euler.0, euler.1, euler.2)
}

/// Convert Euler angles to quaternion
#[must_use]
pub fn euler_to_quaternion(pitch: f64, yaw: f64, roll: f64) -> Quaternion<f64> {
    let unit_q = UnitQuaternion::from_euler_angles(pitch, yaw, roll);
    *unit_q.quaternion()
}

/// RGB to HSV conversion
#[must_use]
pub fn rgb_to_hsv(r: f32, g: f32, b: f32) -> (f32, f32, f32) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let h = if delta < 1e-6 {
        0.0
    } else if (max - r).abs() < 1e-6 {
        60.0 * (((g - b) / delta) % 6.0)
    } else if (max - g).abs() < 1e-6 {
        60.0 * (((b - r) / delta) + 2.0)
    } else {
        60.0 * (((r - g) / delta) + 4.0)
    };

    let s = if max < 1e-6 { 0.0 } else { delta / max };
    let v = max;

    (h, s, v)
}

/// HSV to RGB conversion
#[must_use]
pub fn hsv_to_rgb(h: f32, s: f32, v: f32) -> (f32, f32, f32) {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = if h < 60.0 {
        (c, x, 0.0)
    } else if h < 120.0 {
        (x, c, 0.0)
    } else if h < 180.0 {
        (0.0, c, x)
    } else if h < 240.0 {
        (0.0, x, c)
    } else if h < 300.0 {
        (x, 0.0, c)
    } else {
        (c, 0.0, x)
    };

    (r + m, g + m, b + m)
}

/// Calculate distance between two points
#[must_use]
pub fn distance(a: &Point3<f64>, b: &Point3<f64>) -> f64 {
    (b - a).norm()
}

/// Calculate angle between two vectors
#[must_use]
pub fn angle_between(a: &Vector3<f64>, b: &Vector3<f64>) -> f64 {
    let dot = a.dot(b);
    let mag_product = a.norm() * b.norm();
    if mag_product < 1e-10 {
        0.0
    } else {
        (dot / mag_product).acos()
    }
}

/// Smooth step function
#[must_use]
pub fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Calculate frame time from FPS
#[must_use]
pub fn fps_to_frame_time_ns(fps: f64) -> u64 {
    (1_000_000_000.0 / fps) as u64
}

/// Calculate FPS from frame time
#[must_use]
pub fn frame_time_ns_to_fps(frame_time_ns: u64) -> f64 {
    1_000_000_000.0 / frame_time_ns as f64
}

/// Convert RGB u8 to f32
#[must_use]
pub fn rgb_u8_to_f32(rgb: [u8; 3]) -> [f32; 3] {
    [
        f32::from(rgb[0]) / 255.0,
        f32::from(rgb[1]) / 255.0,
        f32::from(rgb[2]) / 255.0,
    ]
}

/// Convert RGB f32 to u8
#[must_use]
pub fn rgb_f32_to_u8(rgb: [f32; 3]) -> [u8; 3] {
    [
        (rgb[0] * 255.0).min(255.0) as u8,
        (rgb[1] * 255.0).min(255.0) as u8,
        (rgb[2] * 255.0).min(255.0) as u8,
    ]
}

/// Apply gamma correction
#[must_use]
pub fn apply_gamma(value: f32, gamma: f32) -> f32 {
    value.powf(gamma)
}

/// Remove gamma correction
#[must_use]
pub fn remove_gamma(value: f32, gamma: f32) -> f32 {
    value.powf(1.0 / gamma)
}

/// sRGB to linear conversion
#[must_use]
pub fn srgb_to_linear(value: f32) -> f32 {
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

/// Linear to sRGB conversion
#[must_use]
pub fn linear_to_srgb(value: f32) -> f32 {
    if value <= 0.0031308 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_time_conversions() {
        let nanos = 1_000_000_000u64;
        let seconds = nanos_to_seconds(nanos);
        assert_eq!(seconds, 1.0);

        let back_to_nanos = seconds_to_nanos(seconds);
        assert_eq!(back_to_nanos, nanos);
    }

    #[test]
    fn test_lerp() {
        assert_eq!(lerp(0.0, 10.0, 0.5), 5.0);
        assert_eq!(lerp(0.0, 10.0, 0.0), 0.0);
        assert_eq!(lerp(0.0, 10.0, 1.0), 10.0);
    }

    #[test]
    fn test_clamp() {
        assert_eq!(clamp(5.0, 0.0, 10.0), 5.0);
        assert_eq!(clamp(-5.0, 0.0, 10.0), 0.0);
        assert_eq!(clamp(15.0, 0.0, 10.0), 10.0);
    }

    #[test]
    fn test_deg_rad_conversion() {
        let deg = 180.0;
        let rad = deg_to_rad(deg);
        assert!((rad - std::f64::consts::PI).abs() < 1e-10);

        let back_to_deg = rad_to_deg(rad);
        assert!((back_to_deg - deg).abs() < 1e-10);
    }

    #[test]
    fn test_rgb_hsv_conversion() {
        let (h, s, v) = rgb_to_hsv(1.0, 0.0, 0.0);
        assert!((h - 0.0).abs() < 1.0);
        assert!((s - 1.0).abs() < 1e-6);
        assert!((v - 1.0).abs() < 1e-6);

        let (r, g, b) = hsv_to_rgb(0.0, 1.0, 1.0);
        assert!((r - 1.0).abs() < 1e-6);
        assert!(g.abs() < 1e-6);
        assert!(b.abs() < 1e-6);
    }

    #[test]
    fn test_rgb_conversions() {
        let rgb_u8 = [255, 128, 64];
        let rgb_f32 = rgb_u8_to_f32(rgb_u8);
        let back_to_u8 = rgb_f32_to_u8(rgb_f32);

        assert_eq!(rgb_u8, back_to_u8);
    }

    #[test]
    fn test_gamma() {
        let value = 0.5f32;
        let gamma = 2.2f32;

        let corrected = apply_gamma(value, gamma);
        let original = remove_gamma(corrected, gamma);

        assert!((original - value).abs() < 1e-6);
    }

    #[test]
    fn test_srgb_linear() {
        let linear = 0.5f32;
        let srgb = linear_to_srgb(linear);
        let back_to_linear = srgb_to_linear(srgb);

        assert!((back_to_linear - linear).abs() < 1e-5);
    }

    #[test]
    fn test_distance() {
        let a = Point3::new(0.0, 0.0, 0.0);
        let b = Point3::new(3.0, 4.0, 0.0);
        let dist = distance(&a, &b);

        assert!((dist - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_angle_between() {
        let a = Vector3::new(1.0, 0.0, 0.0);
        let b = Vector3::new(0.0, 1.0, 0.0);
        let angle = angle_between(&a, &b);

        assert!((angle - std::f64::consts::PI / 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_smoothstep() {
        assert_eq!(smoothstep(0.0, 1.0, 0.0), 0.0);
        assert_eq!(smoothstep(0.0, 1.0, 1.0), 1.0);
        assert!(smoothstep(0.0, 1.0, 0.5) > 0.4);
        assert!(smoothstep(0.0, 1.0, 0.5) < 0.6);
    }

    #[test]
    fn test_fps_conversion() {
        let fps = 60.0;
        let frame_time = fps_to_frame_time_ns(fps);
        let back_to_fps = frame_time_ns_to_fps(frame_time);

        assert!((back_to_fps - fps).abs() < 0.1);
    }

    #[test]
    fn test_perspective_matrix() {
        let fov = deg_to_rad(90.0);
        let aspect = 16.0 / 9.0;
        let near = 0.1;
        let far = 100.0;

        let matrix = perspective_matrix(fov, aspect, near, far);
        assert!(matrix[(0, 0)] != 0.0);
        assert!(matrix[(1, 1)] != 0.0);
    }

    #[test]
    fn test_look_at_matrix() {
        let eye = Point3::new(0.0, 0.0, 5.0);
        let target = Point3::origin();
        let up = Vector3::new(0.0, 1.0, 0.0);

        let matrix = look_at_matrix(&eye, &target, &up);
        assert!(matrix[(3, 3)] != 0.0);
    }
}
