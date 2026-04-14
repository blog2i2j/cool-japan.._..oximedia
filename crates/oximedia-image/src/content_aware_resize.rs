//! Content-aware image resizing via seam carving.
//!
//! Implements the Avidan & Shamir (2007) seam carving algorithm for
//! intelligent image resizing that preserves salient content.
//!
//! # Algorithm
//!
//! 1. **Energy map**: Compute gradient magnitude (Sobel) for each pixel.
//! 2. **Cumulative energy (DP)**: Build a minimum-energy path table
//!    from top to bottom (for horizontal seams) or left to right (vertical).
//! 3. **Seam tracing**: Back-trace the minimum cumulative-energy path.
//! 4. **Seam removal**: Remove the seam pixels from the image.
//! 5. Repeat until target dimensions are reached.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]

use crate::error::{ImageError, ImageResult};

// ── Image buffer ──────────────────────────────────────────────────────────────

/// A floating-point RGBA image buffer for seam carving.
#[derive(Debug, Clone)]
pub struct CarvingImage {
    /// Width in pixels.
    pub width: usize,
    /// Height in pixels.
    pub height: usize,
    /// Number of channels (1, 3, or 4).
    pub channels: usize,
    /// Pixel data: [y * width * channels + x * channels + c].
    pub data: Vec<f32>,
}

impl CarvingImage {
    /// Create from raw u8 interleaved data.
    pub fn from_u8(data: &[u8], width: usize, height: usize, channels: usize) -> ImageResult<Self> {
        let expected = width * height * channels;
        if data.len() != expected {
            return Err(ImageError::invalid_format(format!(
                "Buffer length {} != expected {}",
                data.len(),
                expected
            )));
        }
        let float_data = data.iter().map(|&v| v as f32 / 255.0).collect();
        Ok(Self {
            width,
            height,
            channels,
            data: float_data,
        })
    }

    /// Sample a pixel channel at (x, y).
    #[must_use]
    pub fn get(&self, x: usize, y: usize, c: usize) -> f32 {
        if x < self.width && y < self.height && c < self.channels {
            self.data[y * self.width * self.channels + x * self.channels + c]
        } else {
            0.0
        }
    }

    /// Set a pixel channel at (x, y).
    pub fn set(&mut self, x: usize, y: usize, c: usize, v: f32) {
        if x < self.width && y < self.height && c < self.channels {
            self.data[y * self.width * self.channels + x * self.channels + c] = v;
        }
    }

    /// Convert to u8 bytes.
    #[must_use]
    pub fn to_u8(&self) -> Vec<u8> {
        self.data
            .iter()
            .map(|&v| (v.clamp(0.0, 1.0) * 255.0).round() as u8)
            .collect()
    }

    /// Compute BT.709 luma for pixel (x, y).
    #[must_use]
    pub fn luma(&self, x: usize, y: usize) -> f32 {
        match self.channels {
            1 => self.get(x, y, 0),
            2 => self.get(x, y, 0),
            _ => {
                let r = self.get(x, y, 0);
                let g = self.get(x, y, 1);
                let b = self.get(x, y, 2);
                0.2126 * r + 0.7152 * g + 0.0722 * b
            }
        }
    }
}

// ── Energy map ────────────────────────────────────────────────────────────────

/// Compute the gradient-magnitude energy map using Sobel operators.
///
/// Returns a 2D energy map (row-major) of size `width × height`.
pub fn compute_energy_map(image: &CarvingImage) -> Vec<f32> {
    let w = image.width;
    let h = image.height;
    let mut energy = vec![0.0f32; w * h];

    for y in 0..h {
        for x in 0..w {
            // Horizontal Sobel
            let xl = if x > 0 { x - 1 } else { 0 };
            let xr = if x + 1 < w { x + 1 } else { w - 1 };
            let yu = if y > 0 { y - 1 } else { 0 };
            let yd = if y + 1 < h { y + 1 } else { h - 1 };

            let gx = image.luma(xr, yu) + 2.0 * image.luma(xr, y) + image.luma(xr, yd)
                - image.luma(xl, yu)
                - 2.0 * image.luma(xl, y)
                - image.luma(xl, yd);
            let gy = image.luma(xl, yd) + 2.0 * image.luma(x, yd) + image.luma(xr, yd)
                - image.luma(xl, yu)
                - 2.0 * image.luma(x, yu)
                - image.luma(xr, yu);

            energy[y * w + x] = (gx * gx + gy * gy).sqrt();
        }
    }

    energy
}

/// Compute a forward-energy map (Rubinstein 2008) for better quality on natural images.
///
/// Uses pixel differences instead of gradient magnitude to avoid halos.
pub fn compute_forward_energy(image: &CarvingImage) -> Vec<f32> {
    let w = image.width;
    let h = image.height;
    let mut energy = vec![0.0f32; w * h];

    for y in 0..h {
        for x in 0..w {
            let xl = if x > 0 { x - 1 } else { 0 };
            let xr = if x + 1 < w { x + 1 } else { w - 1 };
            let yu = if y > 0 { y - 1 } else { 0 };

            let cu = (image.luma(xr, y) - image.luma(xl, y)).abs();
            let cl = cu + (image.luma(x, yu) - image.luma(xl, y)).abs();
            let cr = cu + (image.luma(x, yu) - image.luma(xr, y)).abs();

            energy[y * w + x] = (cu + cl + cr) / 3.0;
        }
    }

    energy
}

// ── Cumulative energy (DP) ────────────────────────────────────────────────────

/// Compute the cumulative minimum energy table for vertical seam carving.
///
/// Returns a `(dp, parent)` pair where `dp[y*w+x]` is the minimum energy
/// to reach pixel (x, y) from the top, and `parent[y*w+x]` is the x-offset
/// of the parent pixel in row `y-1` (0=left, 1=straight, 2=right from centre).
pub fn cumulative_energy_vertical(
    energy: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<i8>) {
    let mut dp = energy.to_vec();
    let mut parent = vec![0i8; width * height];

    for y in 1..height {
        for x in 0..width {
            let xl = if x > 0 { x - 1 } else { x };
            let xr = if x + 1 < width { x + 1 } else { x };

            let el = dp[(y - 1) * width + xl];
            let ec = dp[(y - 1) * width + x];
            let er = dp[(y - 1) * width + xr];

            let (min_e, px) = if el <= ec && el <= er {
                (el, -1i8)
            } else if ec <= er {
                (ec, 0i8)
            } else {
                (er, 1i8)
            };

            dp[y * width + x] = energy[y * width + x] + min_e;
            parent[y * width + x] = px;
        }
    }

    (dp, parent)
}

/// Compute the cumulative minimum energy for horizontal seam carving.
pub fn cumulative_energy_horizontal(
    energy: &[f32],
    width: usize,
    height: usize,
) -> (Vec<f32>, Vec<i8>) {
    let mut dp = energy.to_vec();
    let mut parent = vec![0i8; width * height];

    for x in 1..width {
        for y in 0..height {
            let yt = if y > 0 { y - 1 } else { y };
            let yb = if y + 1 < height { y + 1 } else { y };

            let et = dp[yt * width + (x - 1)];
            let ec = dp[y * width + (x - 1)];
            let eb = dp[yb * width + (x - 1)];

            let (min_e, py) = if et <= ec && et <= eb {
                (et, -1i8)
            } else if ec <= eb {
                (ec, 0i8)
            } else {
                (eb, 1i8)
            };

            dp[y * width + x] = energy[y * width + x] + min_e;
            parent[y * width + x] = py;
        }
    }

    (dp, parent)
}

// ── Seam tracing ──────────────────────────────────────────────────────────────

/// Trace a vertical seam from bottom to top.
///
/// Returns a vector of x-coordinates, one per row (length = `height`).
pub fn trace_vertical_seam(dp: &[f32], parent: &[i8], width: usize, height: usize) -> Vec<usize> {
    let mut seam = vec![0usize; height];

    // Find minimum in last row
    let last_row_start = (height - 1) * width;
    let min_x = (0..width)
        .min_by(|&a, &b| {
            dp[last_row_start + a]
                .partial_cmp(&dp[last_row_start + b])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    seam[height - 1] = min_x;

    // Back-trace
    for y in (0..height - 1).rev() {
        let px = parent[(y + 1) * width + seam[y + 1]];
        let x = seam[y + 1] as isize + px as isize;
        seam[y] = x.clamp(0, width as isize - 1) as usize;
    }

    seam
}

/// Trace a horizontal seam from right to left.
pub fn trace_horizontal_seam(dp: &[f32], parent: &[i8], width: usize, height: usize) -> Vec<usize> {
    let mut seam = vec![0usize; width];

    // Find minimum in last column
    let min_y = (0..height)
        .min_by(|&a, &b| {
            dp[a * width + (width - 1)]
                .partial_cmp(&dp[b * width + (width - 1)])
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or(0);
    seam[width - 1] = min_y;

    for x in (0..width - 1).rev() {
        let py = parent[seam[x + 1] * width + (x + 1)];
        let y = seam[x + 1] as isize + py as isize;
        seam[x] = y.clamp(0, height as isize - 1) as usize;
    }

    seam
}

// ── Seam removal ─────────────────────────────────────────────────────────────

/// Remove a vertical seam from an image (reduces width by 1).
pub fn remove_vertical_seam(image: &CarvingImage, seam: &[usize]) -> CarvingImage {
    let w = image.width - 1;
    let h = image.height;
    let c = image.channels;
    let mut data = Vec::with_capacity(w * h * c);

    for y in 0..h {
        let skip_x = seam[y];
        for x in 0..image.width {
            if x == skip_x {
                continue;
            }
            for ch in 0..c {
                data.push(image.get(x, y, ch));
            }
        }
    }

    CarvingImage {
        width: w,
        height: h,
        channels: c,
        data,
    }
}

/// Remove a horizontal seam from an image (reduces height by 1).
pub fn remove_horizontal_seam(image: &CarvingImage, seam: &[usize]) -> CarvingImage {
    let w = image.width;
    let h = image.height - 1;
    let c = image.channels;
    let mut data = Vec::with_capacity(w * h * c);

    for x in 0..w {
        let skip_y = seam[x];
        for y in 0..image.height {
            if y == skip_y {
                continue;
            }
            for ch in 0..c {
                data.push(image.get(x, y, ch));
            }
        }
    }

    // Rearrange: data is column-major above, need to transpose to row-major
    // Actually we need row-major; let's redo:
    let mut row_major = Vec::with_capacity(w * h * c);
    for y in 0..h {
        for x in 0..w {
            // Find actual y after removal
            let skip_y = seam[x];
            let actual_y = if y < skip_y { y } else { y + 1 };
            for ch in 0..c {
                row_major.push(image.get(x, actual_y, ch));
            }
        }
    }

    CarvingImage {
        width: w,
        height: h,
        channels: c,
        data: row_major,
    }
}

// ── Energy visualisation ──────────────────────────────────────────────────────

/// Convert an energy map to a normalized u8 image.
pub fn energy_to_u8(energy: &[f32]) -> Vec<u8> {
    let max_e = energy.iter().copied().fold(0.0f32, f32::max);
    if max_e < f32::EPSILON {
        return vec![0u8; energy.len()];
    }
    energy
        .iter()
        .map(|&e| ((e / max_e) * 255.0).round() as u8)
        .collect()
}

// ── Seam carver ───────────────────────────────────────────────────────────────

/// Configuration for the seam carver.
#[derive(Debug, Clone)]
pub struct SeamCarvingConfig {
    /// Use forward energy (better quality, slightly slower).
    pub use_forward_energy: bool,
    /// Maximum seams to carve in one batch (chunking for efficiency).
    pub batch_size: usize,
}

impl Default for SeamCarvingConfig {
    fn default() -> Self {
        Self {
            use_forward_energy: true,
            batch_size: 1,
        }
    }
}

/// Content-aware seam carver.
pub struct SeamCarver {
    config: SeamCarvingConfig,
}

impl SeamCarver {
    /// Create a new carver with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: SeamCarvingConfig::default(),
        }
    }

    /// Create with custom config.
    #[must_use]
    pub fn with_config(config: SeamCarvingConfig) -> Self {
        Self { config }
    }

    /// Resize the image to `target_width × target_height` by seam carving.
    ///
    /// Only carving (reduction) is supported. For enlarging, use standard interpolation.
    pub fn resize(
        &self,
        image: &CarvingImage,
        target_width: usize,
        target_height: usize,
    ) -> ImageResult<CarvingImage> {
        if target_width > image.width {
            return Err(ImageError::unsupported(
                "Seam carving can only reduce width (target_width <= current width)",
            ));
        }
        if target_height > image.height {
            return Err(ImageError::unsupported(
                "Seam carving can only reduce height (target_height <= current height)",
            ));
        }

        let mut current = image.clone();

        // Remove vertical seams (reduce width)
        let seams_to_remove_x = current.width - target_width;
        for _ in 0..seams_to_remove_x {
            let energy = if self.config.use_forward_energy {
                compute_forward_energy(&current)
            } else {
                compute_energy_map(&current)
            };
            let (dp, parent) = cumulative_energy_vertical(&energy, current.width, current.height);
            let seam = trace_vertical_seam(&dp, &parent, current.width, current.height);
            current = remove_vertical_seam(&current, &seam);
        }

        // Remove horizontal seams (reduce height)
        let seams_to_remove_y = current.height - target_height;
        for _ in 0..seams_to_remove_y {
            let energy = if self.config.use_forward_energy {
                compute_forward_energy(&current)
            } else {
                compute_energy_map(&current)
            };
            let (dp, parent) = cumulative_energy_horizontal(&energy, current.width, current.height);
            let seam = trace_horizontal_seam(&dp, &parent, current.width, current.height);
            current = remove_horizontal_seam(&current, &seam);
        }

        Ok(current)
    }

    /// Find a single minimum-energy vertical seam without removing it.
    pub fn find_vertical_seam(&self, image: &CarvingImage) -> Vec<usize> {
        let energy = compute_energy_map(image);
        let (dp, parent) = cumulative_energy_vertical(&energy, image.width, image.height);
        trace_vertical_seam(&dp, &parent, image.width, image.height)
    }

    /// Find a single minimum-energy horizontal seam without removing it.
    pub fn find_horizontal_seam(&self, image: &CarvingImage) -> Vec<usize> {
        let energy = compute_energy_map(image);
        let (dp, parent) = cumulative_energy_horizontal(&energy, image.width, image.height);
        trace_horizontal_seam(&dp, &parent, image.width, image.height)
    }
}

impl Default for SeamCarver {
    fn default() -> Self {
        Self::new()
    }
}

// ── Seam highlight ────────────────────────────────────────────────────────────

/// Highlight a vertical seam in red in an RGB image (for visualisation).
pub fn highlight_vertical_seam(image: &mut CarvingImage, seam: &[usize]) {
    if image.channels < 3 {
        return;
    }
    for (y, &x) in seam.iter().enumerate() {
        image.set(x, y, 0, 1.0); // R=1
        image.set(x, y, 1, 0.0); // G=0
        image.set(x, y, 2, 0.0); // B=0
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn checkerboard(w: usize, h: usize) -> CarvingImage {
        let c = 3usize;
        let mut data = vec![0.0f32; w * h * c];
        for y in 0..h {
            for x in 0..w {
                let v = if (x + y) % 2 == 0 { 1.0 } else { 0.0 };
                for ch in 0..c {
                    data[y * w * c + x * c + ch] = v;
                }
            }
        }
        CarvingImage {
            width: w,
            height: h,
            channels: c,
            data,
        }
    }

    fn uniform_image(w: usize, h: usize, val: f32) -> CarvingImage {
        CarvingImage {
            width: w,
            height: h,
            channels: 3,
            data: vec![val; w * h * 3],
        }
    }

    #[test]
    fn test_carving_image_from_u8() {
        let data = vec![255u8; 4 * 4 * 3];
        let img = CarvingImage::from_u8(&data, 4, 4, 3).expect("from_u8");
        assert_eq!(img.width, 4);
        assert_eq!(img.height, 4);
        assert!((img.get(0, 0, 0) - 1.0).abs() < 1e-3);
    }

    #[test]
    fn test_carving_image_from_u8_bad_size() {
        let data = vec![0u8; 10];
        let result = CarvingImage::from_u8(&data, 4, 4, 3);
        assert!(result.is_err());
    }

    #[test]
    fn test_carving_image_to_u8_roundtrip() {
        let data: Vec<u8> = (0..12u8).collect();
        let img = CarvingImage::from_u8(&data, 4, 1, 3).expect("from_u8");
        let back = img.to_u8();
        for (i, (&a, &b)) in data.iter().zip(back.iter()).enumerate() {
            assert!(
                (a as i32 - b as i32).abs() <= 1,
                "mismatch at {i}: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_luma_grayscale() {
        let mut img = uniform_image(2, 2, 0.5);
        img.channels = 1;
        img.data = vec![0.5; 4];
        assert!((img.luma(0, 0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_luma_rgb() {
        let img = CarvingImage {
            width: 1,
            height: 1,
            channels: 3,
            data: vec![1.0, 0.0, 0.0],
        };
        // Pure red → luma = 0.2126
        assert!((img.luma(0, 0) - 0.2126).abs() < 1e-4);
    }

    #[test]
    fn test_energy_map_size() {
        let img = checkerboard(8, 8);
        let energy = compute_energy_map(&img);
        assert_eq!(energy.len(), 64);
    }

    #[test]
    fn test_energy_uniform_image() {
        let img = uniform_image(6, 6, 0.5);
        let energy = compute_energy_map(&img);
        // Uniform image should have near-zero energy everywhere
        for &e in &energy {
            assert!(e < 1e-4, "Uniform image should have ~0 energy, got {e}");
        }
    }

    #[test]
    fn test_energy_checkerboard_nonzero() {
        let img = checkerboard(6, 6);
        let energy = compute_energy_map(&img);
        let total: f32 = energy.iter().sum();
        assert!(total > 0.1, "Checkerboard should have significant energy");
    }

    #[test]
    fn test_forward_energy_size() {
        let img = checkerboard(8, 8);
        let energy = compute_forward_energy(&img);
        assert_eq!(energy.len(), 64);
    }

    #[test]
    fn test_cumulative_energy_vertical_shape() {
        let energy = vec![1.0f32; 4 * 4];
        let (dp, parent) = cumulative_energy_vertical(&energy, 4, 4);
        assert_eq!(dp.len(), 16);
        assert_eq!(parent.len(), 16);
    }

    #[test]
    fn test_cumulative_energy_vertical_increases() {
        let energy = vec![1.0f32; 4 * 4];
        let (dp, _) = cumulative_energy_vertical(&energy, 4, 4);
        // Cumulative energy in last row should be > first row
        let first_row_min = dp[..4].iter().copied().fold(f32::MAX, f32::min);
        let last_row_min = dp[12..].iter().copied().fold(f32::MAX, f32::min);
        assert!(last_row_min > first_row_min, "cumulative must increase");
    }

    #[test]
    fn test_trace_vertical_seam_length() {
        let energy = vec![1.0f32; 6 * 4];
        let (dp, parent) = cumulative_energy_vertical(&energy, 6, 4);
        let seam = trace_vertical_seam(&dp, &parent, 6, 4);
        assert_eq!(seam.len(), 4);
    }

    #[test]
    fn test_trace_vertical_seam_valid_coords() {
        let img = checkerboard(8, 8);
        let energy = compute_energy_map(&img);
        let (dp, parent) = cumulative_energy_vertical(&energy, 8, 8);
        let seam = trace_vertical_seam(&dp, &parent, 8, 8);
        for &x in &seam {
            assert!(x < 8, "seam x coordinate out of bounds: {x}");
        }
    }

    #[test]
    fn test_remove_vertical_seam_reduces_width() {
        let img = checkerboard(8, 8);
        let energy = compute_energy_map(&img);
        let (dp, parent) = cumulative_energy_vertical(&energy, 8, 8);
        let seam = trace_vertical_seam(&dp, &parent, 8, 8);
        let carved = remove_vertical_seam(&img, &seam);
        assert_eq!(carved.width, 7);
        assert_eq!(carved.height, 8);
    }

    #[test]
    fn test_remove_horizontal_seam_reduces_height() {
        let img = checkerboard(8, 6);
        let energy = compute_energy_map(&img);
        let (dp, parent) = cumulative_energy_horizontal(&energy, 8, 6);
        let seam = trace_horizontal_seam(&dp, &parent, 8, 6);
        let carved = remove_horizontal_seam(&img, &seam);
        assert_eq!(carved.width, 8);
        assert_eq!(carved.height, 5);
    }

    #[test]
    fn test_seam_carver_reduce_width() {
        let img = checkerboard(10, 8);
        let carver = SeamCarver::new();
        let result = carver.resize(&img, 8, 8).expect("carve");
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_seam_carver_reduce_height() {
        let img = checkerboard(8, 10);
        let carver = SeamCarver::new();
        let result = carver.resize(&img, 8, 8).expect("carve");
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_seam_carver_no_resize() {
        let img = checkerboard(8, 8);
        let carver = SeamCarver::new();
        let result = carver.resize(&img, 8, 8).expect("carve");
        assert_eq!(result.width, 8);
        assert_eq!(result.height, 8);
    }

    #[test]
    fn test_seam_carver_enlarge_rejected() {
        let img = checkerboard(8, 8);
        let carver = SeamCarver::new();
        let result = carver.resize(&img, 10, 8);
        assert!(result.is_err());
    }

    #[test]
    fn test_seam_carver_data_integrity() {
        // Uniform image: after carving, values should still be ~0.5
        let img = uniform_image(8, 8, 0.5);
        let carver = SeamCarver::new();
        let result = carver.resize(&img, 6, 6).expect("carve");
        for &v in &result.data {
            assert!(
                (v - 0.5).abs() < 1e-3,
                "Uniform values should be preserved: {v}"
            );
        }
    }

    #[test]
    fn test_energy_to_u8_range() {
        let energy = vec![0.0f32, 0.5, 1.0, 2.0, 0.25];
        let u8_vals = energy_to_u8(&energy);
        assert_eq!(u8_vals.len(), energy.len());
        assert_eq!(u8_vals[0], 0);
        assert!((u8_vals[2] as i32 - 127).abs() <= 1); // 1.0 / 2.0 * 255 ≈ 127-128
    }

    #[test]
    fn test_energy_to_u8_all_zeros() {
        let energy = vec![0.0f32; 10];
        let u8_vals = energy_to_u8(&energy);
        assert!(u8_vals.iter().all(|&v| v == 0));
    }

    #[test]
    fn test_find_vertical_seam_returns_correct_length() {
        let img = checkerboard(6, 4);
        let carver = SeamCarver::new();
        let seam = carver.find_vertical_seam(&img);
        assert_eq!(seam.len(), 4);
    }

    #[test]
    fn test_find_horizontal_seam_returns_correct_length() {
        let img = checkerboard(6, 4);
        let carver = SeamCarver::new();
        let seam = carver.find_horizontal_seam(&img);
        assert_eq!(seam.len(), 6);
    }

    #[test]
    fn test_highlight_seam_sets_red() {
        let mut img = uniform_image(4, 4, 0.5);
        let seam = vec![2usize; 4]; // column 2 for all rows
        highlight_vertical_seam(&mut img, &seam);
        for y in 0..4 {
            assert!((img.get(2, y, 0) - 1.0).abs() < 1e-6, "R should be 1");
            assert!((img.get(2, y, 1)).abs() < 1e-6, "G should be 0");
            assert!((img.get(2, y, 2)).abs() < 1e-6, "B should be 0");
        }
    }
}
