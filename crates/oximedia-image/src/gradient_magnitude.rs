//! Gradient magnitude computation for image processing and analysis.
//!
//! Implements classic first-order derivative (gradient) operators used extensively
//! in edge detection, feature extraction, and image quality analysis:
//!
//! - **Sobel** — 3×3 isotropic gradient, robust general-purpose operator
//! - **Prewitt** — 3×3 uniform-weight gradient, simpler alternative to Sobel
//! - **Scharr** — 3×3 improved Sobel with near-perfect rotational symmetry
//!
//! Each operator produces:
//! - A **magnitude map** (`|∇I|`) indicating edge strength at every pixel
//! - An optional **direction map** (`atan2(Gy, Gx)`) in radians (−π … π)
//!
//! Additionally, **non-maximum suppression (NMS)** thins thick gradient ridges
//! to single-pixel-wide edges by retaining only local maxima along the gradient
//! direction.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_lossless)]

use crate::error::{ImageError, ImageResult};

// ─── Operator enum ─────────────────────────────────────────────────────────────

/// First-order gradient operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum GradientOperator {
    /// Sobel 3×3 operator — standard, noise-robust.
    Sobel,
    /// Prewitt 3×3 operator — uniform weights.
    Prewitt,
    /// Scharr 3×3 operator — near-perfect rotational symmetry.
    Scharr,
}

// ─── Kernel coefficients ───────────────────────────────────────────────────────

/// Returns the (Kx, Ky) 3×3 kernel pair for the requested operator.
///
/// Each kernel is stored in row-major order (top-left → bottom-right).
/// `Kx` detects horizontal edges, `Ky` detects vertical edges.
fn kernel_for(op: GradientOperator) -> ([f64; 9], [f64; 9]) {
    match op {
        GradientOperator::Sobel => (
            // Kx
            [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0],
            // Ky
            [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0],
        ),
        GradientOperator::Prewitt => (
            [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0],
            [-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0],
        ),
        GradientOperator::Scharr => (
            [-3.0, 0.0, 3.0, -10.0, 0.0, 10.0, -3.0, 0.0, 3.0],
            [-3.0, -10.0, -3.0, 0.0, 0.0, 0.0, 3.0, 10.0, 3.0],
        ),
    }
}

// ─── GrayImage ────────────────────────────────────────────────────────────────

/// A floating-point grayscale image buffer.
///
/// Pixel values are expected in the range `[0.0, 1.0]` (normalised), stored
/// in row-major order: index = `y * width + x`.
#[derive(Debug, Clone)]
pub struct GrayImage {
    /// Pixel data (row-major, values in 0.0–1.0).
    pub data: Vec<f64>,
    /// Image width in pixels.
    pub width: usize,
    /// Image height in pixels.
    pub height: usize,
}

impl GrayImage {
    /// Create a new `GrayImage` from raw data.
    ///
    /// Returns an error if `data.len() != width * height`.
    pub fn new(data: Vec<f64>, width: usize, height: usize) -> ImageResult<Self> {
        let expected = width * height;
        if data.len() != expected {
            return Err(ImageError::invalid_format(format!(
                "GrayImage: data length {} does not match {}×{} = {}",
                data.len(),
                width,
                height,
                expected
            )));
        }
        Ok(Self {
            data,
            width,
            height,
        })
    }

    /// Create a blank (all-zero) image.
    #[must_use]
    pub fn zeros(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0; width * height],
            width,
            height,
        }
    }

    /// Get the pixel value at `(x, y)`, returning 0.0 for out-of-bounds access.
    #[inline]
    #[must_use]
    pub fn get(&self, x: isize, y: isize) -> f64 {
        if x < 0 || y < 0 || x as usize >= self.width || y as usize >= self.height {
            return 0.0;
        }
        self.data[y as usize * self.width + x as usize]
    }

    /// Set the pixel value at `(x, y)`. Does nothing if out of bounds.
    #[inline]
    pub fn set(&mut self, x: usize, y: usize, value: f64) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = value;
        }
    }

    /// Create a `GrayImage` from an interleaved RGB u8 buffer (luminance only).
    ///
    /// Uses the Rec.709 luma coefficients: Y = 0.2126 R + 0.7152 G + 0.0722 B.
    pub fn from_rgb_u8(data: &[u8], width: usize, height: usize) -> ImageResult<Self> {
        let expected = width * height * 3;
        if data.len() != expected {
            return Err(ImageError::invalid_format(format!(
                "from_rgb_u8: buffer length {} != {}",
                data.len(),
                expected
            )));
        }
        let pixels: Vec<f64> = data
            .chunks_exact(3)
            .map(|px| {
                let r = px[0] as f64 / 255.0;
                let g = px[1] as f64 / 255.0;
                let b = px[2] as f64 / 255.0;
                0.2126 * r + 0.7152 * g + 0.0722 * b
            })
            .collect();
        Self::new(pixels, width, height)
    }
}

// ─── GradientMaps ─────────────────────────────────────────────────────────────

/// Output of a gradient computation.
#[derive(Debug, Clone)]
pub struct GradientMaps {
    /// Gradient magnitude at each pixel.  Range depends on normalization.
    pub magnitude: GrayImage,
    /// Gradient direction at each pixel in radians `[−π, π]`.
    pub direction: GrayImage,
    /// Gx component (horizontal derivative).
    pub gx: GrayImage,
    /// Gy component (vertical derivative).
    pub gy: GrayImage,
}

// ─── Core computation ─────────────────────────────────────────────────────────

/// Compute gradient maps for a grayscale image using the given operator.
///
/// The magnitude map is **not** normalised by default so that absolute gradient
/// values can be compared across images.  Call [`normalize_magnitude`] when a
/// `[0.0, 1.0]` range is needed.
///
/// # Errors
///
/// Returns an error if the input image dimensions are zero.
pub fn compute_gradient(image: &GrayImage, op: GradientOperator) -> ImageResult<GradientMaps> {
    let w = image.width;
    let h = image.height;
    if w == 0 || h == 0 {
        return Err(ImageError::invalid_format(
            "gradient: image must be non-empty",
        ));
    }

    let (kx, ky) = kernel_for(op);

    let n = w * h;
    let mut mag_data = vec![0.0f64; n];
    let mut dir_data = vec![0.0f64; n];
    let mut gx_data = vec![0.0f64; n];
    let mut gy_data = vec![0.0f64; n];

    for y in 0..h {
        for x in 0..w {
            let xi = x as isize;
            let yi = y as isize;

            // Apply 3×3 kernels (border pixels treated as zero-padded)
            let gx = kx[0] * image.get(xi - 1, yi - 1)
                + kx[1] * image.get(xi, yi - 1)
                + kx[2] * image.get(xi + 1, yi - 1)
                + kx[3] * image.get(xi - 1, yi)
                + kx[4] * image.get(xi, yi)
                + kx[5] * image.get(xi + 1, yi)
                + kx[6] * image.get(xi - 1, yi + 1)
                + kx[7] * image.get(xi, yi + 1)
                + kx[8] * image.get(xi + 1, yi + 1);

            let gy = ky[0] * image.get(xi - 1, yi - 1)
                + ky[1] * image.get(xi, yi - 1)
                + ky[2] * image.get(xi + 1, yi - 1)
                + ky[3] * image.get(xi - 1, yi)
                + ky[4] * image.get(xi, yi)
                + ky[5] * image.get(xi + 1, yi)
                + ky[6] * image.get(xi - 1, yi + 1)
                + ky[7] * image.get(xi, yi + 1)
                + ky[8] * image.get(xi + 1, yi + 1);

            let idx = y * w + x;
            gx_data[idx] = gx;
            gy_data[idx] = gy;
            mag_data[idx] = (gx * gx + gy * gy).sqrt();
            dir_data[idx] = gy.atan2(gx);
        }
    }

    Ok(GradientMaps {
        magnitude: GrayImage {
            data: mag_data,
            width: w,
            height: h,
        },
        direction: GrayImage {
            data: dir_data,
            width: w,
            height: h,
        },
        gx: GrayImage {
            data: gx_data,
            width: w,
            height: h,
        },
        gy: GrayImage {
            data: gy_data,
            width: w,
            height: h,
        },
    })
}

/// Normalise a magnitude image to `[0.0, 1.0]` in place.
///
/// If all values are equal the image is filled with zeros.
pub fn normalize_magnitude(mag: &mut GrayImage) {
    let max = mag.data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min = mag.data.iter().cloned().fold(f64::INFINITY, f64::min);
    let range = max - min;
    if range < f64::EPSILON {
        mag.data.iter_mut().for_each(|v| *v = 0.0);
        return;
    }
    mag.data.iter_mut().for_each(|v| *v = (*v - min) / range);
}

// ─── Non-maximum suppression ──────────────────────────────────────────────────

/// Discretised gradient direction for NMS.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NmsDirection {
    Horizontal,  // 0° / 180°
    Diagonal45,  // 45° / 225°
    Vertical,    // 90° / 270°
    Diagonal135, // 135° / 315°
}

fn quantize_direction(angle_rad: f64) -> NmsDirection {
    // Normalise to [0, π)
    let mut a = angle_rad;
    if a < 0.0 {
        a += std::f64::consts::PI;
    }
    let deg = a.to_degrees();
    if deg < 22.5 || deg >= 157.5 {
        NmsDirection::Horizontal
    } else if deg < 67.5 {
        NmsDirection::Diagonal45
    } else if deg < 112.5 {
        NmsDirection::Vertical
    } else {
        NmsDirection::Diagonal135
    }
}

/// Apply non-maximum suppression to thin gradient ridges to single-pixel width.
///
/// Each pixel is retained only if it is a local maximum compared to its two
/// neighbours along the gradient direction (8-connectivity).  Non-maxima are set
/// to 0.
///
/// # Arguments
///
/// * `magnitude` — gradient magnitude map (values in any non-negative range)
/// * `direction` — gradient direction map in radians (−π … π)
///
/// # Errors
///
/// Returns an error if the magnitude and direction images have different sizes.
pub fn non_maximum_suppression(
    magnitude: &GrayImage,
    direction: &GrayImage,
) -> ImageResult<GrayImage> {
    let w = magnitude.width;
    let h = magnitude.height;
    if w != direction.width || h != direction.height {
        return Err(ImageError::invalid_format(
            "NMS: magnitude and direction images must have the same size",
        ));
    }

    let mut out = GrayImage::zeros(w, h);

    for y in 0..h {
        for x in 0..w {
            let xi = x as isize;
            let yi = y as isize;
            let idx = y * w + x;
            let mag = magnitude.data[idx];
            let dir = direction.data[idx];

            let (n1, n2) = match quantize_direction(dir) {
                NmsDirection::Horizontal => (magnitude.get(xi - 1, yi), magnitude.get(xi + 1, yi)),
                NmsDirection::Diagonal45 => {
                    (magnitude.get(xi + 1, yi - 1), magnitude.get(xi - 1, yi + 1))
                }
                NmsDirection::Vertical => (magnitude.get(xi, yi - 1), magnitude.get(xi, yi + 1)),
                NmsDirection::Diagonal135 => {
                    (magnitude.get(xi - 1, yi - 1), magnitude.get(xi + 1, yi + 1))
                }
            };

            if mag >= n1 && mag >= n2 {
                out.data[idx] = mag;
            }
            // else remains 0.0
        }
    }

    Ok(out)
}

// ─── Convenience pipeline ─────────────────────────────────────────────────────

/// High-level configuration for the full gradient-magnitude pipeline.
#[derive(Debug, Clone)]
pub struct GradientConfig {
    /// Which gradient operator to apply.
    pub operator: GradientOperator,
    /// Whether to normalise the magnitude map to `[0.0, 1.0]`.
    pub normalize: bool,
    /// Whether to apply non-maximum suppression after computing gradients.
    pub apply_nms: bool,
}

impl Default for GradientConfig {
    fn default() -> Self {
        Self {
            operator: GradientOperator::Sobel,
            normalize: true,
            apply_nms: false,
        }
    }
}

impl GradientConfig {
    /// Create a config for the given operator with default settings.
    #[must_use]
    pub fn new(operator: GradientOperator) -> Self {
        Self {
            operator,
            ..Self::default()
        }
    }

    /// Enable/disable magnitude normalisation.
    #[must_use]
    pub fn with_normalize(mut self, v: bool) -> Self {
        self.normalize = v;
        self
    }

    /// Enable/disable non-maximum suppression.
    #[must_use]
    pub fn with_nms(mut self, v: bool) -> Self {
        self.apply_nms = v;
        self
    }
}

/// Run the full gradient magnitude pipeline and return the processed magnitude map.
///
/// Steps:
/// 1. Compute `Gx`, `Gy`, magnitude, and direction using the chosen operator.
/// 2. Optionally normalise the magnitude to `[0.0, 1.0]`.
/// 3. Optionally thin ridges with non-maximum suppression.
///
/// # Errors
///
/// Propagates errors from [`compute_gradient`] and [`non_maximum_suppression`].
pub fn gradient_magnitude_pipeline(
    image: &GrayImage,
    cfg: &GradientConfig,
) -> ImageResult<GrayImage> {
    let mut maps = compute_gradient(image, cfg.operator)?;

    if cfg.normalize {
        normalize_magnitude(&mut maps.magnitude);
    }

    if cfg.apply_nms {
        let thinned = non_maximum_suppression(&maps.magnitude, &maps.direction)?;
        return Ok(thinned);
    }

    Ok(maps.magnitude)
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn flat_image(w: usize, h: usize, v: f64) -> GrayImage {
        GrayImage {
            data: vec![v; w * h],
            width: w,
            height: h,
        }
    }

    /// Build a small vertical-edge image: left half=0, right half=1.
    fn vertical_edge_image(w: usize, h: usize) -> GrayImage {
        let data = (0..h)
            .flat_map(|_| (0..w).map(|x| if x < w / 2 { 0.0 } else { 1.0 }))
            .collect();
        GrayImage {
            data,
            width: w,
            height: h,
        }
    }

    #[test]
    fn test_gray_image_new_ok() {
        let img = GrayImage::new(vec![0.0; 6], 3, 2).unwrap();
        assert_eq!(img.width, 3);
        assert_eq!(img.height, 2);
    }

    #[test]
    fn test_gray_image_new_size_mismatch() {
        let err = GrayImage::new(vec![0.0; 5], 3, 2);
        assert!(err.is_err());
    }

    #[test]
    fn test_gray_image_get_out_of_bounds() {
        let img = GrayImage::zeros(4, 4);
        assert_eq!(img.get(-1, 0), 0.0);
        assert_eq!(img.get(4, 0), 0.0);
        assert_eq!(img.get(0, -1), 0.0);
        assert_eq!(img.get(0, 4), 0.0);
    }

    #[test]
    fn test_flat_image_zero_gradient() {
        // Uniform image: interior pixels should produce zero gradients.
        // Border pixels may have non-zero gradients due to zero-padding at boundary.
        let w = 8;
        let h = 8;
        let img = flat_image(w, h, 0.5);
        let maps = compute_gradient(&img, GradientOperator::Sobel).unwrap();
        // Check only interior pixels (not on the 1-pixel border)
        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                let m = maps.magnitude.data[y * w + x];
                assert!(
                    m.abs() < 1e-10,
                    "flat interior pixel ({x},{y}) should have zero gradient, got {m}"
                );
            }
        }
    }

    #[test]
    fn test_vertical_edge_detected() {
        let img = vertical_edge_image(10, 10);
        let maps = compute_gradient(&img, GradientOperator::Sobel).unwrap();
        // There should be high-magnitude pixels near the centre column.
        let max_mag = maps.magnitude.data.iter().cloned().fold(0.0f64, f64::max);
        assert!(max_mag > 0.0, "Expected non-zero gradient at vertical edge");
    }

    #[test]
    fn test_normalize_magnitude() {
        let img = vertical_edge_image(10, 6);
        let mut maps = compute_gradient(&img, GradientOperator::Sobel).unwrap();
        normalize_magnitude(&mut maps.magnitude);
        let max = maps
            .magnitude
            .data
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min = maps
            .magnitude
            .data
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        // After normalization the range should be [0, 1].
        assert!(
            (max - 1.0).abs() < 1e-10 || max == 0.0,
            "max should be 1.0, got {max}"
        );
        assert!(min >= 0.0, "min should be >= 0");
    }

    #[test]
    fn test_prewitt_and_scharr_operators() {
        let img = vertical_edge_image(8, 8);
        for op in [GradientOperator::Prewitt, GradientOperator::Scharr] {
            let maps = compute_gradient(&img, op).unwrap();
            let max_mag = maps.magnitude.data.iter().cloned().fold(0.0f64, f64::max);
            assert!(max_mag > 0.0, "Operator {op:?} should detect edge");
        }
    }

    #[test]
    fn test_non_maximum_suppression_thins_edges() {
        let img = vertical_edge_image(12, 12);
        let maps = compute_gradient(&img, GradientOperator::Sobel).unwrap();
        let nms = non_maximum_suppression(&maps.magnitude, &maps.direction).unwrap();
        // NMS output should have at most as many non-zero pixels as the raw magnitude.
        let raw_nonzero = maps.magnitude.data.iter().filter(|&&v| v > 1e-10).count();
        let nms_nonzero = nms.data.iter().filter(|&&v| v > 1e-10).count();
        assert!(
            nms_nonzero <= raw_nonzero,
            "NMS should not add new non-zero pixels"
        );
    }

    #[test]
    fn test_nms_size_mismatch_error() {
        let mag = GrayImage::zeros(4, 4);
        let dir = GrayImage::zeros(5, 4);
        assert!(non_maximum_suppression(&mag, &dir).is_err());
    }

    #[test]
    fn test_pipeline_with_nms() {
        let img = vertical_edge_image(10, 10);
        let cfg = GradientConfig::new(GradientOperator::Sobel)
            .with_normalize(true)
            .with_nms(true);
        let result = gradient_magnitude_pipeline(&img, &cfg).unwrap();
        assert_eq!(result.width, 10);
        assert_eq!(result.height, 10);
        // All values should be in [0, 1] after normalisation.
        for &v in &result.data {
            assert!((0.0..=1.0 + 1e-10).contains(&v), "value {v} out of range");
        }
    }

    #[test]
    fn test_from_rgb_u8() {
        // Create a 2×2 RGB image: all white
        let data = vec![255u8; 4 * 3];
        let gray = GrayImage::from_rgb_u8(&data, 2, 2).unwrap();
        for &v in &gray.data {
            assert!((v - 1.0).abs() < 1e-10, "white pixel should map to 1.0");
        }
    }

    #[test]
    fn test_direction_range() {
        // Gradient directions must lie in [-π, π].
        let img = vertical_edge_image(8, 8);
        let maps = compute_gradient(&img, GradientOperator::Sobel).unwrap();
        for &d in &maps.direction.data {
            assert!(
                d >= -PI - 1e-10 && d <= PI + 1e-10,
                "direction {d} out of [-π, π]"
            );
        }
    }

    #[test]
    fn test_zero_size_error() {
        let img = GrayImage {
            data: vec![],
            width: 0,
            height: 5,
        };
        assert!(compute_gradient(&img, GradientOperator::Sobel).is_err());
    }
}
