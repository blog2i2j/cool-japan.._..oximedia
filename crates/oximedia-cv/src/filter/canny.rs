//! Canny edge detection with hysteresis thresholding.
//!
//! [`CannyEdge`] implements the full Canny pipeline:
//! 1. Gaussian smoothing (5×5 kernel)
//! 2. Gradient magnitude and direction via Sobel operators
//! 3. Non-maximum suppression along the gradient direction
//! 4. Double-threshold classification: strong / weak / suppressed edges
//! 5. Edge tracking by hysteresis: weak edges connected to strong edges survive
//!
//! # Example
//!
//! ```
//! use oximedia_cv::filter::canny::CannyEdge;
//!
//! let width = 8u32;
//! let height = 8u32;
//! // Synthetic image: top half bright, bottom half dark (horizontal edge)
//! let mut img = vec![0u8; (width * height) as usize];
//! for y in 0..4 {
//!     for x in 0..width as usize {
//!         img[y * width as usize + x] = 200;
//!     }
//! }
//! let edges = CannyEdge::detect(&img, width, height, 30.0, 80.0);
//! assert_eq!(edges.len(), (width * height) as usize);
//! // At least one edge pixel should be detected near the boundary
//! assert!(edges.iter().any(|&e| e));
//! ```

/// Canny edge detector.
///
/// Stateless — call [`CannyEdge::detect`] directly.  The struct holds only
/// construction-time parameters to allow reuse across frames.
#[derive(Debug, Clone)]
pub struct CannyEdge {
    /// Low hysteresis threshold.
    low: f32,
    /// High hysteresis threshold.
    high: f32,
}

impl CannyEdge {
    /// Create a new [`CannyEdge`] detector with the given thresholds.
    ///
    /// * `low`  — weak-edge threshold (gradient magnitude above this survives
    ///   if connected to a strong edge).
    /// * `high` — strong-edge threshold (gradient magnitude above this is
    ///   immediately an edge).
    #[must_use]
    pub fn new(low: f32, high: f32) -> Self {
        Self { low, high }
    }

    /// Run the full Canny pipeline on a grayscale image and return an edge mask.
    ///
    /// Delegates to [`detect`] using the thresholds stored in `self`.
    #[must_use]
    pub fn run(&self, img: &[u8], w: u32, h: u32) -> Vec<bool> {
        detect(img, w, h, self.low, self.high)
    }

    /// Detect edges in a grayscale image using the Canny algorithm.
    ///
    /// This is the primary entry point.  It is equivalent to
    /// `CannyEdge::new(low, high).run(img, w, h)`.
    ///
    /// * `img`  — flat row-major grayscale pixel data (`w × h` bytes).
    /// * `w`, `h` — image dimensions.
    /// * `low`, `high` — hysteresis thresholds in gradient-magnitude units
    ///   (approximately 0 – 1 440 for 8-bit images after Sobel).
    ///
    /// Returns a `Vec<bool>` of length `w × h` where `true` marks an edge pixel.
    #[must_use]
    pub fn detect(img: &[u8], w: u32, h: u32, low: f32, high: f32) -> Vec<bool> {
        detect(img, w, h, low, high)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Implementation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Run the Canny pipeline.
fn detect(img: &[u8], w: u32, h: u32, low: f32, high: f32) -> Vec<bool> {
    let width = w as usize;
    let height = h as usize;
    let n = width * height;

    if n == 0 || img.len() < n {
        return vec![false; n];
    }

    // Step 1: Gaussian blur (5×5 approximation)
    let blurred = gaussian_blur_5x5(img, width, height);

    // Step 2: Sobel gradient magnitude + direction
    let (mag, dir) = sobel_gradients(&blurred, width, height);

    // Step 3: Non-maximum suppression
    let suppressed = non_maximum_suppression(&mag, &dir, width, height);

    // Step 4 + 5: Double-threshold + hysteresis
    hysteresis(&suppressed, width, height, low, high)
}

/// 5×5 Gaussian blur using a separable approximation [1 4 6 4 1].
fn gaussian_blur_5x5(src: &[u8], w: usize, h: usize) -> Vec<f32> {
    let kernel: [f32; 5] = [1.0, 4.0, 6.0, 4.0, 1.0];
    let norm = 16.0_f32; // sum of kernel
    let n = w * h;

    // Horizontal pass
    let mut tmp = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let xi = (x as i64 + k as i64 - 2).clamp(0, w as i64 - 1) as usize;
                acc += src[y * w + xi] as f32 * kv;
            }
            tmp[y * w + x] = acc / norm;
        }
    }

    // Vertical pass
    let mut out = vec![0.0f32; n];
    for y in 0..h {
        for x in 0..w {
            let mut acc = 0.0f32;
            for (k, &kv) in kernel.iter().enumerate() {
                let yi = (y as i64 + k as i64 - 2).clamp(0, h as i64 - 1) as usize;
                acc += tmp[yi * w + x] * kv;
            }
            out[y * w + x] = acc / norm;
        }
    }
    out
}

/// Sobel gradients. Returns (magnitude, direction_quantised_0..4).
///
/// Direction is quantised into four sectors:
/// * 0 → horizontal (0°)
/// * 1 → diagonal (45°)
/// * 2 → vertical (90°)
/// * 3 → anti-diagonal (135°)
fn sobel_gradients(src: &[f32], w: usize, h: usize) -> (Vec<f32>, Vec<u8>) {
    let n = w * h;
    let mut mag = vec![0.0f32; n];
    let mut dir = vec![0u8; n];

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let tl = src[(y - 1) * w + (x - 1)];
            let t  = src[(y - 1) * w + x];
            let tr = src[(y - 1) * w + (x + 1)];
            let l  = src[y * w + (x - 1)];
            let r  = src[y * w + (x + 1)];
            let bl = src[(y + 1) * w + (x - 1)];
            let b  = src[(y + 1) * w + x];
            let br = src[(y + 1) * w + (x + 1)];

            let gx = -tl - 2.0 * l - bl + tr + 2.0 * r + br;
            let gy = -tl - 2.0 * t - tr + bl + 2.0 * b + br;

            mag[y * w + x] = (gx * gx + gy * gy).sqrt();

            // Quantise angle to 0/45/90/135
            let angle = gy.atan2(gx).to_degrees();
            let angle = if angle < 0.0 { angle + 180.0 } else { angle };
            dir[y * w + x] = if angle < 22.5 || angle >= 157.5 {
                0 // 0°
            } else if angle < 67.5 {
                1 // 45°
            } else if angle < 112.5 {
                2 // 90°
            } else {
                3 // 135°
            };
        }
    }

    (mag, dir)
}

/// Non-maximum suppression: keep only local maxima along gradient direction.
fn non_maximum_suppression(mag: &[f32], dir: &[u8], w: usize, h: usize) -> Vec<f32> {
    let n = w * h;
    let mut out = vec![0.0f32; n];

    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let idx = y * w + x;
            let m = mag[idx];
            let (n1, n2) = match dir[idx] {
                0 => (mag[y * w + x + 1], mag[y * w + x - 1]),           // horizontal
                1 => (mag[(y + 1) * w + x + 1], mag[(y - 1) * w + x - 1]), // 45°
                2 => (mag[(y + 1) * w + x], mag[(y - 1) * w + x]),         // vertical
                _ => (mag[(y + 1) * w + x - 1], mag[(y - 1) * w + x + 1]), // 135°
            };
            if m >= n1 && m >= n2 {
                out[idx] = m;
            }
        }
    }
    out
}

/// Double-threshold and hysteresis edge tracking.
///
/// Strong edge (> high) → immediate edge.
/// Weak edge (low..=high) → edge only if connected to a strong edge.
/// Below low → suppressed.
fn hysteresis(mag: &[f32], w: usize, h: usize, low: f32, high: f32) -> Vec<bool> {
    let n = w * h;
    // 0 = suppressed, 1 = weak, 2 = strong
    let mut state = vec![0u8; n];

    for (i, &m) in mag.iter().enumerate().take(n) {
        if m > high {
            state[i] = 2;
        } else if m > low {
            state[i] = 1;
        }
    }

    // BFS/DFS from every strong pixel to promote connected weak pixels
    let mut visited = vec![false; n];
    let mut stack: Vec<usize> = Vec::new();

    for i in 0..n {
        if state[i] == 2 && !visited[i] {
            stack.push(i);
            visited[i] = true;
        }
    }

    while let Some(idx) = stack.pop() {
        let y = idx / w;
        let x = idx % w;

        for dy in -1_i64..=1 {
            for dx in -1_i64..=1 {
                if dx == 0 && dy == 0 {
                    continue;
                }
                let ny = y as i64 + dy;
                let nx = x as i64 + dx;
                if ny < 0 || ny >= h as i64 || nx < 0 || nx >= w as i64 {
                    continue;
                }
                let ni = ny as usize * w + nx as usize;
                if !visited[ni] && state[ni] == 1 {
                    state[ni] = 2; // promote to strong
                    visited[ni] = true;
                    stack.push(ni);
                }
            }
        }
    }

    state.iter().map(|&s| s == 2).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_output_length() {
        let img = vec![128u8; 16];
        let edges = CannyEdge::detect(&img, 4, 4, 10.0, 30.0);
        assert_eq!(edges.len(), 16);
    }

    #[test]
    fn test_uniform_image_no_edges() {
        let img = vec![200u8; 64];
        let edges = CannyEdge::detect(&img, 8, 8, 10.0, 30.0);
        // Uniform image has zero gradient → no edges
        assert!(edges.iter().all(|&e| !e));
    }

    #[test]
    fn test_horizontal_step_edge() {
        let w = 8usize;
        let h = 8usize;
        let mut img = vec![0u8; w * h];
        // Top half white, bottom half black
        for y in 0..4 {
            for x in 0..w {
                img[y * w + x] = 200;
            }
        }
        let edges = CannyEdge::detect(&img, w as u32, h as u32, 20.0, 60.0);
        assert_eq!(edges.len(), w * h);
        assert!(edges.iter().any(|&e| e), "Expected at least one edge pixel");
    }

    #[test]
    fn test_zero_size_image() {
        let img: Vec<u8> = vec![];
        let edges = CannyEdge::detect(&img, 0, 0, 10.0, 30.0);
        assert_eq!(edges.len(), 0);
    }

    #[test]
    fn test_new_and_run_equivalent_to_detect() {
        let img: Vec<u8> = (0..64).map(|i| (i * 4) as u8).collect();
        let direct = CannyEdge::detect(&img, 8, 8, 15.0, 40.0);
        let via_struct = CannyEdge::new(15.0, 40.0).run(&img, 8, 8);
        assert_eq!(direct, via_struct);
    }

    #[test]
    fn test_high_threshold_suppresses_all() {
        // With an absurdly high threshold, nothing is an edge
        let img: Vec<u8> = (0..64).map(|i| (i % 255) as u8).collect();
        let edges = CannyEdge::detect(&img, 8, 8, 1_000_000.0, 2_000_000.0);
        assert!(edges.iter().all(|&e| !e));
    }
}
