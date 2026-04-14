//! Full Canny edge detection pipeline.
//
// Implements the classic five-stage Canny algorithm:
//
// 1. **Gaussian blur** — reduce noise using a separable 1D Gaussian kernel
//    whose sigma and radius are configurable.
// 2. **Gradient computation** — Sobel 3x3 operator produces per-pixel
//    magnitude and direction maps.
// 3. **Non-maximum suppression** — thin edges to single-pixel width by
//    discarding pixels that are not local maxima along the gradient direction.
// 4. **Double thresholding** — classify pixels as strong, weak, or non-edge.
// 5. **Hysteresis edge tracking** — promote weak pixels that are 8-connected
//    to at least one strong pixel.
//
// The module operates on `GrayF32` (row-major `f32` buffer) so it remains
// independent of the higher-level `ImageData` / `ImageFrame` types and can
// be called directly on any floating-point luminance buffer.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A single-channel floating-point image (row-major).
#[derive(Clone, Debug)]
pub struct GrayF32 {
    /// Pixel buffer, values nominally in [0.0, 1.0].
    pub data: Vec<f32>,
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
}

impl GrayF32 {
    /// Create a new image filled with zeros.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            data: vec![0.0f32; width * height],
            width,
            height,
        }
    }

    /// Create from existing data, returning `None` if the size does not match.
    #[must_use]
    pub fn from_data(width: usize, height: usize, data: Vec<f32>) -> Option<Self> {
        if data.len() == width * height {
            Some(Self {
                data,
                width,
                height,
            })
        } else {
            None
        }
    }

    /// Get a pixel value, clamping out-of-range coordinates to the border.
    #[must_use]
    pub fn get_clamped(&self, x: i32, y: i32) -> f32 {
        let cx = x.clamp(0, self.width as i32 - 1) as usize;
        let cy = y.clamp(0, self.height as i32 - 1) as usize;
        self.data[cy * self.width + cx]
    }

    /// Set a pixel value, silently ignoring out-of-range coordinates.
    pub fn set(&mut self, x: usize, y: usize, v: f32) {
        if x < self.width && y < self.height {
            self.data[y * self.width + x] = v;
        }
    }

    /// Get pixel value, returning `None` for out-of-range coordinates.
    #[must_use]
    pub fn get(&self, x: usize, y: usize) -> Option<f32> {
        if x < self.width && y < self.height {
            Some(self.data[y * self.width + x])
        } else {
            None
        }
    }

    /// Normalize the image so its values span [0.0, 1.0].
    pub fn normalize(&mut self) {
        let min = self.data.iter().copied().fold(f32::INFINITY, f32::min);
        let max = self.data.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let range = max - min;
        if range > f32::EPSILON {
            for v in &mut self.data {
                *v = (*v - min) / range;
            }
        }
    }

    /// Total number of pixels.
    #[must_use]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// True if the image has no pixels.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

// ---------------------------------------------------------------------------
// Gaussian blur (separable)
// ---------------------------------------------------------------------------

/// Build a 1-D Gaussian kernel of the given sigma.
///
/// The kernel is normalised to sum to 1.0.  `radius` controls how many
/// samples are taken each side of the centre (the full kernel has length
/// `2 * radius + 1`).
#[must_use]
pub fn gaussian_kernel_1d(sigma: f64, radius: usize) -> Vec<f64> {
    let len = 2 * radius + 1;
    let mut k = Vec::with_capacity(len);
    let two_sigma_sq = 2.0 * sigma * sigma;
    let mut sum = 0.0f64;
    for i in 0..len {
        let x = i as f64 - radius as f64;
        let v = (-x * x / two_sigma_sq).exp();
        k.push(v);
        sum += v;
    }
    for v in &mut k {
        *v /= sum;
    }
    k
}

/// Apply a 1-D horizontal convolution to every row.
fn convolve_h(image: &GrayF32, kernel: &[f64]) -> GrayF32 {
    let r = kernel.len() / 2;
    let mut out = GrayF32::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let mut acc = 0.0f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sx = x as i32 + ki as i32 - r as i32;
                acc += image.get_clamped(sx, y as i32) as f64 * kv;
            }
            out.set(x, y, acc as f32);
        }
    }
    out
}

/// Apply a 1-D vertical convolution to every column.
fn convolve_v(image: &GrayF32, kernel: &[f64]) -> GrayF32 {
    let r = kernel.len() / 2;
    let mut out = GrayF32::new(image.width, image.height);
    for y in 0..image.height {
        for x in 0..image.width {
            let mut acc = 0.0f64;
            for (ki, &kv) in kernel.iter().enumerate() {
                let sy = y as i32 + ki as i32 - r as i32;
                acc += image.get_clamped(x as i32, sy) as f64 * kv;
            }
            out.set(x, y, acc as f32);
        }
    }
    out
}

/// Blur a grayscale image with a separable Gaussian filter.
#[must_use]
pub fn gaussian_blur(image: &GrayF32, sigma: f64, radius: usize) -> GrayF32 {
    let k = gaussian_kernel_1d(sigma, radius);
    let tmp = convolve_h(image, &k);
    convolve_v(&tmp, &k)
}

// ---------------------------------------------------------------------------
// Sobel gradient
// ---------------------------------------------------------------------------

/// Gradient components and derived magnitude / direction.
pub struct GradientMap {
    /// Horizontal gradient (Gx).
    pub gx: GrayF32,
    /// Vertical gradient (Gy).
    pub gy: GrayF32,
    /// Gradient magnitude sqrt(Gx² + Gy²).
    pub magnitude: GrayF32,
    /// Gradient direction in radians.
    pub direction: Vec<f64>,
}

/// Compute the Sobel gradient of a grayscale image.
#[must_use]
pub fn sobel_gradient(image: &GrayF32) -> GradientMap {
    let kx: [f64; 9] = [-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let ky: [f64; 9] = [-1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];

    let mut gx = GrayF32::new(image.width, image.height);
    let mut gy = GrayF32::new(image.width, image.height);
    let mut magnitude = GrayF32::new(image.width, image.height);
    let mut direction = vec![0.0f64; image.width * image.height];

    for y in 0..image.height {
        for x in 0..image.width {
            let xi = x as i32;
            let yi = y as i32;
            let mut sx = 0.0f64;
            let mut sy = 0.0f64;
            for ky_off in -1i32..=1 {
                for kx_off in -1i32..=1 {
                    let ki = ((ky_off + 1) * 3 + (kx_off + 1)) as usize;
                    let pv = image.get_clamped(xi + kx_off, yi + ky_off) as f64;
                    sx += pv * kx[ki];
                    sy += pv * ky[ki];
                }
            }
            let mag = (sx * sx + sy * sy).sqrt() as f32;
            let idx = y * image.width + x;
            gx.data[idx] = sx as f32;
            gy.data[idx] = sy as f32;
            magnitude.data[idx] = mag;
            direction[idx] = sy.atan2(sx);
        }
    }

    GradientMap {
        gx,
        gy,
        magnitude,
        direction,
    }
}

// ---------------------------------------------------------------------------
// Non-maximum suppression
// ---------------------------------------------------------------------------

/// Thin edges by retaining only local maxima along the gradient direction.
#[must_use]
pub fn non_maximum_suppression(grad: &GradientMap) -> GrayF32 {
    let w = grad.magnitude.width;
    let h = grad.magnitude.height;
    let mut out = GrayF32::new(w, h);

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let idx = y * w + x;
            let angle = grad.direction[idx].to_degrees().rem_euclid(180.0);
            let mag = grad.magnitude.data[idx];
            let xi = x as i32;
            let yi = y as i32;

            let (n1, n2) = if angle < 22.5 || angle >= 157.5 {
                // 0° — horizontal edge
                (
                    grad.magnitude.get_clamped(xi - 1, yi),
                    grad.magnitude.get_clamped(xi + 1, yi),
                )
            } else if angle < 67.5 {
                // 45°
                (
                    grad.magnitude.get_clamped(xi - 1, yi - 1),
                    grad.magnitude.get_clamped(xi + 1, yi + 1),
                )
            } else if angle < 112.5 {
                // 90° — vertical edge
                (
                    grad.magnitude.get_clamped(xi, yi - 1),
                    grad.magnitude.get_clamped(xi, yi + 1),
                )
            } else {
                // 135°
                (
                    grad.magnitude.get_clamped(xi + 1, yi - 1),
                    grad.magnitude.get_clamped(xi - 1, yi + 1),
                )
            };

            if mag >= n1 && mag >= n2 {
                out.set(x, y, mag);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Double threshold + hysteresis
// ---------------------------------------------------------------------------

/// Pixel classification after double thresholding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum EdgeClass {
    None,
    Weak,
    Strong,
}

/// Apply double thresholding and hysteresis to produce the final binary edge map.
///
/// `low` and `high` are absolute magnitude thresholds (not normalised).
/// After calling [`non_maximum_suppression`] the magnitude values are in the
/// same scale as the input image so typical values are 0–1 for float inputs.
#[must_use]
pub fn hysteresis(suppressed: &GrayF32, low: f32, high: f32) -> GrayF32 {
    let w = suppressed.width;
    let h = suppressed.height;
    let n = w * h;
    let mut cls = vec![EdgeClass::None; n];

    // Double threshold classification
    for i in 0..n {
        let v = suppressed.data[i];
        if v >= high {
            cls[i] = EdgeClass::Strong;
        } else if v >= low {
            cls[i] = EdgeClass::Weak;
        }
    }

    // Hysteresis: promote weak pixels connected to strong pixels
    let mut changed = true;
    while changed {
        changed = false;
        for y in 1..h.saturating_sub(1) {
            for x in 1..w.saturating_sub(1) {
                let idx = y * w + x;
                if cls[idx] == EdgeClass::Weak {
                    let has_strong = [
                        (x - 1, y - 1),
                        (x, y - 1),
                        (x + 1, y - 1),
                        (x - 1, y),
                        (x + 1, y),
                        (x - 1, y + 1),
                        (x, y + 1),
                        (x + 1, y + 1),
                    ]
                    .iter()
                    .any(|&(nx, ny)| nx < w && ny < h && cls[ny * w + nx] == EdgeClass::Strong);

                    if has_strong {
                        cls[idx] = EdgeClass::Strong;
                        changed = true;
                    }
                }
            }
        }
    }

    // Convert to output image
    let mut out = GrayF32::new(w, h);
    for i in 0..n {
        if cls[i] == EdgeClass::Strong {
            out.data[i] = 1.0;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// High-level Canny pipeline
// ---------------------------------------------------------------------------

/// Configuration for the Canny edge detector.
#[derive(Clone, Debug)]
pub struct CannyConfig {
    /// Standard deviation for the Gaussian blur pre-filter.
    pub sigma: f64,
    /// Kernel half-width for the Gaussian blur.
    pub blur_radius: usize,
    /// Low threshold for hysteresis (relative to maximum gradient magnitude
    /// when `auto_threshold` is true, otherwise absolute).
    pub low_threshold: f32,
    /// High threshold for hysteresis.
    pub high_threshold: f32,
    /// If true, `low_threshold` and `high_threshold` are treated as fractions
    /// of the maximum gradient magnitude rather than absolute values.
    pub auto_threshold: bool,
}

impl Default for CannyConfig {
    fn default() -> Self {
        Self {
            sigma: 1.4,
            blur_radius: 2,
            low_threshold: 0.05,
            high_threshold: 0.15,
            auto_threshold: true,
        }
    }
}

impl CannyConfig {
    /// Create a new config with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set Gaussian blur parameters.
    #[must_use]
    pub fn with_blur(mut self, sigma: f64, radius: usize) -> Self {
        self.sigma = sigma;
        self.blur_radius = radius;
        self
    }

    /// Set threshold values.
    #[must_use]
    pub fn with_thresholds(mut self, low: f32, high: f32) -> Self {
        self.low_threshold = low;
        self.high_threshold = high;
        self
    }

    /// Use absolute thresholds instead of relative ones.
    #[must_use]
    pub fn absolute_thresholds(mut self) -> Self {
        self.auto_threshold = false;
        self
    }
}

/// Statistics about the detected edges.
#[derive(Clone, Debug)]
pub struct CannyStats {
    /// Fraction of pixels classified as edges.
    pub edge_density: f64,
    /// Maximum gradient magnitude before thresholding.
    pub max_gradient: f32,
    /// Mean gradient magnitude before thresholding.
    pub mean_gradient: f32,
}

/// Full Canny edge detection result.
pub struct CannyResult {
    /// Binary edge map (1.0 = edge, 0.0 = non-edge).
    pub edges: GrayF32,
    /// Gradient magnitude map (after NMS, before thresholding).
    pub magnitude: GrayF32,
    /// Statistics.
    pub stats: CannyStats,
}

/// Run the complete Canny edge detection pipeline on a grayscale image.
#[must_use]
pub fn canny(image: &GrayF32, config: &CannyConfig) -> CannyResult {
    // Stage 1: Gaussian blur
    let blurred = if config.sigma > 0.0 && config.blur_radius > 0 {
        gaussian_blur(image, config.sigma, config.blur_radius)
    } else {
        image.clone()
    };

    // Stage 2: Sobel gradient
    let grad = sobel_gradient(&blurred);

    // Stage 3: Non-maximum suppression
    let suppressed = non_maximum_suppression(&grad);

    // Compute threshold values
    let max_mag = suppressed.data.iter().copied().fold(0.0f32, f32::max);
    let sum_mag: f64 = suppressed.data.iter().map(|&v| v as f64).sum();
    let mean_mag = if suppressed.len() > 0 {
        (sum_mag / suppressed.len() as f64) as f32
    } else {
        0.0
    };

    let (low, high) = if config.auto_threshold {
        (
            config.low_threshold * max_mag,
            config.high_threshold * max_mag,
        )
    } else {
        (config.low_threshold, config.high_threshold)
    };

    // Stage 4 + 5: Double threshold + hysteresis
    let edges = hysteresis(&suppressed, low, high);

    let edge_count = edges.data.iter().filter(|&&v| v > 0.5).count();
    let edge_density = if edges.len() > 0 {
        edge_count as f64 / edges.len() as f64
    } else {
        0.0
    };

    CannyResult {
        edges,
        magnitude: suppressed,
        stats: CannyStats {
            edge_density,
            max_gradient: max_mag,
            mean_gradient: mean_mag,
        },
    }
}

// ---------------------------------------------------------------------------
// Utility: compute gradient angle histogram
// ---------------------------------------------------------------------------

/// Compute a histogram of gradient directions (in degrees, 0–179) from a
/// `GradientMap`.  `bins` must be in 1..=180.
#[must_use]
pub fn direction_histogram(grad: &GradientMap, bins: usize) -> Vec<f64> {
    let bins = bins.clamp(1, 180);
    let mut hist = vec![0.0f64; bins];
    let bin_size = 180.0 / bins as f64;

    for (i, &dir) in grad.direction.iter().enumerate() {
        let mag = grad.magnitude.data[i];
        if mag < f32::EPSILON {
            continue;
        }
        let deg = dir.to_degrees().rem_euclid(180.0);
        let bin = (deg / bin_size) as usize;
        let bin = bin.min(bins - 1);
        hist[bin] += mag as f64;
    }
    hist
}

// ---------------------------------------------------------------------------
// Utility: Gaussian sigma estimation from kernel radius
// ---------------------------------------------------------------------------

/// Estimate the Gaussian sigma that produces a kernel of the given radius
/// covering approximately 99% of the distribution.
#[must_use]
pub fn sigma_for_radius(radius: usize) -> f64 {
    radius as f64 / 3.0
}

/// Estimate the minimum radius needed to represent a Gaussian of the given sigma.
#[must_use]
pub fn radius_for_sigma(sigma: f64) -> usize {
    ((sigma * 3.0).ceil() as usize).max(1)
}

// ---------------------------------------------------------------------------
// Utility: pi constant re-export for downstream use in tests
// ---------------------------------------------------------------------------

/// π constant (re-exported for convenience in tests).
pub const PI_CONST: f64 = PI;

#[cfg(test)]
mod tests {
    use super::*;

    fn vertical_edge_image(w: usize, h: usize) -> GrayF32 {
        let mut img = GrayF32::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.set(x, y, if x < w / 2 { 0.0 } else { 1.0 });
            }
        }
        img
    }

    fn uniform_image(w: usize, h: usize, v: f32) -> GrayF32 {
        let data = vec![v; w * h];
        GrayF32::from_data(w, h, data).expect("valid uniform image")
    }

    #[test]
    fn test_gray_f32_construction() {
        let img = GrayF32::new(8, 6);
        assert_eq!(img.width, 8);
        assert_eq!(img.height, 6);
        assert_eq!(img.len(), 48);
        assert!(img.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gray_f32_from_data_size_mismatch() {
        assert!(GrayF32::from_data(4, 4, vec![0.0; 10]).is_none());
        assert!(GrayF32::from_data(4, 4, vec![0.0; 16]).is_some());
    }

    #[test]
    fn test_gaussian_kernel_sums_to_one() {
        let k = gaussian_kernel_1d(1.0, 3);
        let sum: f64 = k.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "kernel sum = {sum}");
    }

    #[test]
    fn test_gaussian_blur_reduces_edge_sharpness() {
        let sharp = vertical_edge_image(10, 10);
        let blurred = gaussian_blur(&sharp, 1.0, 2);
        // Pixel at x=4 (just left of edge) should be non-zero after blur
        let v = blurred.get(4, 5).expect("valid coords");
        assert!(v > 0.0, "blur should spread the edge");
    }

    #[test]
    fn test_sobel_gradient_detects_vertical_edge() {
        let img = vertical_edge_image(10, 10);
        let grad = sobel_gradient(&img);
        // Pixels at x=5 (the edge column) should have non-zero Gx
        let gx_at_edge = grad.gx.get(5, 5).expect("valid");
        assert!(gx_at_edge.abs() > 0.0, "Sobel should detect vertical edge");
    }

    #[test]
    fn test_non_maximum_suppression_reduces_edge_width() {
        let img = vertical_edge_image(12, 12);
        let grad = sobel_gradient(&img);
        let suppressed = non_maximum_suppression(&grad);
        // Count edge pixels in the central row — NMS should reduce to 1–2
        let row_edges = (0..12)
            .filter(|&x| suppressed.get(x, 6).map(|v| v > 0.0).unwrap_or(false))
            .count();
        assert!(row_edges <= 3, "NMS should thin edges; found {row_edges}");
    }

    #[test]
    fn test_hysteresis_promotes_weak_adjacent_to_strong() {
        let mut suppressed = GrayF32::new(7, 7);
        // Strong pixel
        suppressed.set(3, 3, 0.9);
        // Weak adjacent pixel
        suppressed.set(4, 3, 0.3);
        let edges = hysteresis(&suppressed, 0.2, 0.7);
        // Strong pixel must be an edge
        assert!(
            (edges.get(3, 3).expect("valid") - 1.0).abs() < f32::EPSILON,
            "strong pixel should be edge"
        );
        // Weak adjacent pixel should be promoted
        assert!(
            (edges.get(4, 3).expect("valid") - 1.0).abs() < f32::EPSILON,
            "weak pixel adjacent to strong should be promoted"
        );
    }

    #[test]
    fn test_canny_uniform_image_no_edges() {
        // Use a large uniform image so border effects are negligible relative
        // to the auto-threshold derived from the maximum gradient magnitude.
        // With auto_threshold the thresholds are fractions of max_magnitude,
        // so all pixels below those fractions are suppressed — a uniform image
        // has zero gradient everywhere hence max_magnitude ≈ 0 and both
        // absolute thresholds collapse to 0, meaning nothing survives.
        // Use absolute thresholds instead to make the test deterministic.
        let img = uniform_image(10, 10, 0.5);
        let config = CannyConfig::new()
            .with_thresholds(0.01, 0.05)
            .absolute_thresholds();
        let result = canny(&img, &config);
        let edge_sum: f32 = result.edges.data.iter().sum();
        assert!(
            edge_sum < f32::EPSILON,
            "uniform image should yield no edges"
        );
    }

    #[test]
    fn test_canny_vertical_edge_detected() {
        let img = vertical_edge_image(20, 20);
        let config = CannyConfig::new()
            .with_blur(1.0, 1)
            .with_thresholds(0.05, 0.2);
        let result = canny(&img, &config);
        // There should be at least one edge pixel
        let edge_count = result.edges.data.iter().filter(|&&v| v > 0.5).count();
        assert!(edge_count > 0, "Canny should detect the vertical step edge");
    }

    #[test]
    fn test_canny_stats_populated() {
        let img = vertical_edge_image(10, 10);
        let config = CannyConfig::default();
        let result = canny(&img, &config);
        assert!(
            result.stats.max_gradient >= 0.0,
            "max_gradient must be non-negative"
        );
    }

    #[test]
    fn test_direction_histogram_bin_count() {
        let img = vertical_edge_image(8, 8);
        let grad = sobel_gradient(&img);
        let hist = direction_histogram(&grad, 8);
        assert_eq!(hist.len(), 8);
        let total: f64 = hist.iter().sum();
        assert!(total > 0.0, "histogram should have non-zero entries");
    }

    #[test]
    fn test_radius_sigma_roundtrip() {
        let sigma = 2.0;
        let r = radius_for_sigma(sigma);
        assert!(r >= 1, "radius must be at least 1");
        // sigma_for_radius is a rough inverse
        let estimated = sigma_for_radius(r);
        assert!(estimated > 0.0);
    }
}
