#![allow(dead_code)]
//! Content-aware scaling using seam carving for intelligent resizing.
//!
//! Implements seam carving algorithms that can resize images and video frames
//! while preserving visually important content by removing low-energy seams.

use std::fmt;

/// Energy function used to compute pixel importance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnergyFunction {
    /// Gradient magnitude (Sobel-like).
    Gradient,
    /// Forward energy (preserves structure better).
    ForwardEnergy,
    /// Entropy-based energy.
    Entropy,
    /// Saliency-based energy from a provided saliency map.
    Saliency,
}

impl fmt::Display for EnergyFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Gradient => write!(f, "Gradient"),
            Self::ForwardEnergy => write!(f, "ForwardEnergy"),
            Self::Entropy => write!(f, "Entropy"),
            Self::Saliency => write!(f, "Saliency"),
        }
    }
}

/// Direction of seam removal.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeamDirection {
    /// Vertical seams (reduces width).
    Vertical,
    /// Horizontal seams (reduces height).
    Horizontal,
}

impl fmt::Display for SeamDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Vertical => write!(f, "Vertical"),
            Self::Horizontal => write!(f, "Horizontal"),
        }
    }
}

/// Configuration for content-aware scaling.
#[derive(Debug, Clone)]
pub struct ContentAwareConfig {
    /// Energy function to use.
    pub energy_function: EnergyFunction,
    /// Target width (number of pixels).
    pub target_width: u32,
    /// Target height (number of pixels).
    pub target_height: u32,
    /// Whether to protect masked regions from removal.
    pub use_protection_mask: bool,
    /// Whether to remove masked regions preferentially.
    pub use_removal_mask: bool,
    /// Weight for protection mask (higher = stronger protection).
    pub protection_weight: f64,
    /// Maximum seams to remove per pass.
    pub max_seams_per_pass: u32,
}

impl ContentAwareConfig {
    /// Creates a new configuration targeting the given dimensions.
    pub fn new(target_width: u32, target_height: u32) -> Self {
        Self {
            energy_function: EnergyFunction::Gradient,
            target_width,
            target_height,
            use_protection_mask: false,
            use_removal_mask: false,
            protection_weight: 1000.0,
            max_seams_per_pass: 1,
        }
    }

    /// Sets the energy function.
    pub fn with_energy_function(mut self, func: EnergyFunction) -> Self {
        self.energy_function = func;
        self
    }

    /// Enables the protection mask.
    pub fn with_protection_mask(mut self, weight: f64) -> Self {
        self.use_protection_mask = true;
        self.protection_weight = weight;
        self
    }

    /// Enables the removal mask.
    pub fn with_removal_mask(mut self) -> Self {
        self.use_removal_mask = true;
        self
    }
}

/// Represents a single seam (a connected path through the image).
#[derive(Debug, Clone)]
pub struct Seam {
    /// Direction of this seam.
    pub direction: SeamDirection,
    /// The indices along the seam (column indices for vertical, row for horizontal).
    pub indices: Vec<u32>,
    /// Total energy of this seam (lower = less visually important).
    pub total_energy: f64,
}

impl Seam {
    /// Creates a new seam.
    pub fn new(direction: SeamDirection, indices: Vec<u32>, total_energy: f64) -> Self {
        Self {
            direction,
            indices,
            total_energy,
        }
    }

    /// Returns the length of the seam.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns true if the seam has no indices.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Simple 2D energy map for seam carving computations.
#[derive(Debug, Clone)]
pub struct EnergyMap {
    /// Width of the energy map.
    width: u32,
    /// Height of the energy map.
    height: u32,
    /// Energy values stored row-major.
    data: Vec<f64>,
}

impl EnergyMap {
    /// Creates a new energy map with zero energy.
    pub fn new(width: u32, height: u32) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; (width as usize) * (height as usize)],
        }
    }

    /// Creates an energy map from raw data.
    pub fn from_data(width: u32, height: u32, data: Vec<f64>) -> Option<Self> {
        if data.len() == (width as usize) * (height as usize) {
            Some(Self {
                width,
                height,
                data,
            })
        } else {
            None
        }
    }

    /// Returns the width.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// Returns the height.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Gets the energy at a given position.
    pub fn get(&self, x: u32, y: u32) -> f64 {
        if x < self.width && y < self.height {
            self.data[(y as usize) * (self.width as usize) + (x as usize)]
        } else {
            f64::MAX
        }
    }

    /// Sets the energy at a given position.
    pub fn set(&mut self, x: u32, y: u32, energy: f64) {
        if x < self.width && y < self.height {
            self.data[(y as usize) * (self.width as usize) + (x as usize)] = energy;
        }
    }

    /// Returns the minimum energy value in the map.
    pub fn min_energy(&self) -> f64 {
        self.data.iter().cloned().fold(f64::MAX, f64::min)
    }

    /// Returns the maximum energy value in the map.
    pub fn max_energy(&self) -> f64 {
        self.data.iter().cloned().fold(f64::MIN, f64::max)
    }

    /// Returns the average energy value.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_energy(&self) -> f64 {
        if self.data.is_empty() {
            return 0.0;
        }
        self.data.iter().sum::<f64>() / self.data.len() as f64
    }

    /// Computes the cumulative energy map for vertical seam finding.
    pub fn compute_cumulative_vertical(&self) -> EnergyMap {
        let mut cumulative = self.clone();
        for y in 1..self.height {
            for x in 0..self.width {
                let up = cumulative.get(x, y - 1);
                let up_left = if x > 0 {
                    cumulative.get(x - 1, y - 1)
                } else {
                    f64::MAX
                };
                let up_right = if x + 1 < self.width {
                    cumulative.get(x + 1, y - 1)
                } else {
                    f64::MAX
                };
                let min_above = up.min(up_left).min(up_right);
                let current = cumulative.get(x, y);
                cumulative.set(x, y, current + min_above);
            }
        }
        cumulative
    }

    /// Finds the minimum-energy vertical seam using the cumulative map.
    pub fn find_vertical_seam(&self) -> Seam {
        let cumulative = self.compute_cumulative_vertical();
        let mut indices = vec![0u32; self.height as usize];

        // Find minimum in last row
        let last_row = self.height - 1;
        let mut min_x = 0u32;
        let mut min_energy = f64::MAX;
        for x in 0..self.width {
            let e = cumulative.get(x, last_row);
            if e < min_energy {
                min_energy = e;
                min_x = x;
            }
        }
        indices[last_row as usize] = min_x;

        // Trace back
        for y in (0..last_row).rev() {
            let prev_x = indices[(y + 1) as usize];
            let mut best_x = prev_x;
            let mut best_e = cumulative.get(prev_x, y);

            if prev_x > 0 {
                let e = cumulative.get(prev_x - 1, y);
                if e < best_e {
                    best_e = e;
                    best_x = prev_x - 1;
                }
            }
            if prev_x + 1 < self.width {
                let e = cumulative.get(prev_x + 1, y);
                if e < best_e {
                    let _ = best_e;
                    best_x = prev_x + 1;
                }
            }

            indices[y as usize] = best_x;
        }

        Seam::new(SeamDirection::Vertical, indices, min_energy)
    }

    /// Computes gradient energy from pixel brightness values.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_gradient_energy(pixels: &[u8], width: u32, height: u32) -> Self {
        let mut map = Self::new(width, height);
        for y in 0..height {
            for x in 0..width {
                let idx = (y as usize) * (width as usize) + (x as usize);
                let left = if x > 0 { pixels[idx - 1] } else { pixels[idx] };
                let right = if x + 1 < width {
                    pixels[idx + 1]
                } else {
                    pixels[idx]
                };
                let up = if y > 0 {
                    pixels[idx - width as usize]
                } else {
                    pixels[idx]
                };
                let down = if y + 1 < height {
                    pixels[idx + width as usize]
                } else {
                    pixels[idx]
                };
                let dx = (right as f64) - (left as f64);
                let dy = (down as f64) - (up as f64);
                let energy = (dx * dx + dy * dy).sqrt();
                map.set(x, y, energy);
            }
        }
        map
    }
}

/// Content-aware scaler engine.
#[derive(Debug)]
pub struct ContentAwareScaler {
    /// Configuration.
    config: ContentAwareConfig,
}

impl ContentAwareScaler {
    /// Creates a new content-aware scaler.
    pub fn new(config: ContentAwareConfig) -> Self {
        Self { config }
    }

    /// Returns the configuration.
    pub fn config(&self) -> &ContentAwareConfig {
        &self.config
    }

    /// Computes the number of vertical seams to remove.
    pub fn vertical_seams_to_remove(&self, current_width: u32) -> u32 {
        if current_width > self.config.target_width {
            current_width - self.config.target_width
        } else {
            0
        }
    }

    /// Computes the number of horizontal seams to remove.
    pub fn horizontal_seams_to_remove(&self, current_height: u32) -> u32 {
        if current_height > self.config.target_height {
            current_height - self.config.target_height
        } else {
            0
        }
    }

    /// Performs content-aware scaling on an RGB image.
    ///
    /// The image is provided as packed 3-byte RGB pixels in row-major order.
    /// Vertical seams are removed first (reducing width), then horizontal
    /// seams (reducing height). Returns the resized pixel buffer together
    /// with the final `(width, height)`.
    ///
    /// If the target dimensions are larger than the source, those axes are
    /// left unchanged (seam carving only shrinks).
    #[allow(clippy::cast_precision_loss)]
    pub fn scale_rgb(&self, pixels: &[u8], width: u32, height: u32) -> Option<(Vec<u8>, u32, u32)> {
        if pixels.len() < (width as usize) * (height as usize) * 3 {
            return None;
        }

        let mut buf = pixels.to_vec();
        let mut w = width;
        let mut h = height;

        // --- Remove vertical seams (reduce width) ---
        let v_seams = self.vertical_seams_to_remove(w);
        for _ in 0..v_seams {
            let luma = rgb_to_luma(&buf, w, h);
            let energy = EnergyMap::compute_gradient_energy(&luma, w, h);
            let seam = energy.find_vertical_seam();
            buf = remove_vertical_seam_rgb(&buf, w, h, &seam);
            w -= 1;
        }

        // --- Remove horizontal seams (reduce height) ---
        let h_seams = self.horizontal_seams_to_remove(h);
        for _ in 0..h_seams {
            // Transpose, remove vertical seam, transpose back
            let transposed = transpose_rgb(&buf, w, h);
            let luma = rgb_to_luma(&transposed, h, w);
            let energy = EnergyMap::compute_gradient_energy(&luma, h, w);
            let seam = energy.find_vertical_seam();
            let carved = remove_vertical_seam_rgb(&transposed, h, w, &seam);
            h -= 1;
            buf = transpose_rgb(&carved, h, w);
        }

        Some((buf, w, h))
    }
}

/// Convert packed RGB buffer to single-channel luma (BT.709 weights).
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn rgb_to_luma(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let count = (width as usize) * (height as usize);
    let mut luma = vec![0u8; count];
    for i in 0..count {
        let base = i * 3;
        if base + 2 < pixels.len() {
            let r = pixels[base] as f32;
            let g = pixels[base + 1] as f32;
            let b = pixels[base + 2] as f32;
            // BT.709 luma
            luma[i] = (0.2126 * r + 0.7152 * g + 0.0722 * b)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
    }
    luma
}

/// Remove a single vertical seam from a packed RGB buffer.
///
/// The seam indices give the column to remove at each row.
/// Returns a buffer whose width is `width - 1`.
fn remove_vertical_seam_rgb(pixels: &[u8], width: u32, height: u32, seam: &Seam) -> Vec<u8> {
    let w = width as usize;
    let new_w = w - 1;
    let h = height as usize;
    let mut out = Vec::with_capacity(new_w * h * 3);

    for y in 0..h {
        let seam_x = if y < seam.indices.len() {
            seam.indices[y] as usize
        } else {
            w // skip nothing
        };
        for x in 0..w {
            if x == seam_x {
                continue;
            }
            let base = (y * w + x) * 3;
            if base + 2 < pixels.len() {
                out.push(pixels[base]);
                out.push(pixels[base + 1]);
                out.push(pixels[base + 2]);
            }
        }
    }

    out
}

/// Transpose a packed RGB image (swap width and height).
fn transpose_rgb(pixels: &[u8], width: u32, height: u32) -> Vec<u8> {
    let w = width as usize;
    let h = height as usize;
    let mut out = vec![0u8; w * h * 3];

    for y in 0..h {
        for x in 0..w {
            let src_base = (y * w + x) * 3;
            let dst_base = (x * h + y) * 3;
            if src_base + 2 < pixels.len() && dst_base + 2 < out.len() {
                out[dst_base] = pixels[src_base];
                out[dst_base + 1] = pixels[src_base + 1];
                out[dst_base + 2] = pixels[src_base + 2];
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_function_display() {
        assert_eq!(EnergyFunction::Gradient.to_string(), "Gradient");
        assert_eq!(EnergyFunction::ForwardEnergy.to_string(), "ForwardEnergy");
        assert_eq!(EnergyFunction::Entropy.to_string(), "Entropy");
        assert_eq!(EnergyFunction::Saliency.to_string(), "Saliency");
    }

    #[test]
    fn test_seam_direction_display() {
        assert_eq!(SeamDirection::Vertical.to_string(), "Vertical");
        assert_eq!(SeamDirection::Horizontal.to_string(), "Horizontal");
    }

    #[test]
    fn test_config_builder() {
        let config = ContentAwareConfig::new(640, 480)
            .with_energy_function(EnergyFunction::ForwardEnergy)
            .with_protection_mask(500.0)
            .with_removal_mask();
        assert_eq!(config.target_width, 640);
        assert_eq!(config.target_height, 480);
        assert_eq!(config.energy_function, EnergyFunction::ForwardEnergy);
        assert!(config.use_protection_mask);
        assert!(config.use_removal_mask);
        assert!((config.protection_weight - 500.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_seam_new() {
        let seam = Seam::new(SeamDirection::Vertical, vec![1, 2, 1, 0], 10.5);
        assert_eq!(seam.direction, SeamDirection::Vertical);
        assert_eq!(seam.len(), 4);
        assert!(!seam.is_empty());
        assert!((seam.total_energy - 10.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_seam_empty() {
        let seam = Seam::new(SeamDirection::Horizontal, vec![], 0.0);
        assert!(seam.is_empty());
        assert_eq!(seam.len(), 0);
    }

    #[test]
    fn test_energy_map_new() {
        let map = EnergyMap::new(4, 3);
        assert_eq!(map.width(), 4);
        assert_eq!(map.height(), 3);
        assert!((map.get(0, 0)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_energy_map_set_get() {
        let mut map = EnergyMap::new(3, 3);
        map.set(1, 2, 42.5);
        assert!((map.get(1, 2) - 42.5).abs() < f64::EPSILON);
    }

    #[test]
    fn test_energy_map_out_of_bounds() {
        let map = EnergyMap::new(2, 2);
        assert_eq!(map.get(5, 5), f64::MAX);
    }

    #[test]
    fn test_energy_map_from_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let map = EnergyMap::from_data(3, 2, data).expect("should succeed in test");
        assert!((map.get(2, 1) - 6.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_energy_map_from_data_invalid() {
        let data = vec![1.0, 2.0];
        assert!(EnergyMap::from_data(3, 2, data).is_none());
    }

    #[test]
    fn test_energy_map_statistics() {
        let data = vec![1.0, 5.0, 3.0, 7.0];
        let map = EnergyMap::from_data(2, 2, data).expect("should succeed in test");
        assert!((map.min_energy() - 1.0).abs() < f64::EPSILON);
        assert!((map.max_energy() - 7.0).abs() < f64::EPSILON);
        assert!((map.average_energy() - 4.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_find_vertical_seam() {
        // 3x3 energy map where center column has lowest energy
        let data = vec![10.0, 1.0, 10.0, 10.0, 1.0, 10.0, 10.0, 1.0, 10.0];
        let map = EnergyMap::from_data(3, 3, data).expect("should succeed in test");
        let seam = map.find_vertical_seam();
        assert_eq!(seam.direction, SeamDirection::Vertical);
        assert_eq!(seam.len(), 3);
        // The seam should go through column 1 (lowest energy)
        for &idx in &seam.indices {
            assert_eq!(idx, 1);
        }
    }

    #[test]
    fn test_compute_gradient_energy() {
        let pixels = vec![10, 20, 30, 40, 50, 60, 70, 80, 90];
        let map = EnergyMap::compute_gradient_energy(&pixels, 3, 3);
        assert_eq!(map.width(), 3);
        assert_eq!(map.height(), 3);
        // Center pixel should have non-zero energy
        assert!(map.get(1, 1) > 0.0);
    }

    #[test]
    fn test_content_aware_scaler_seam_counts() {
        let config = ContentAwareConfig::new(640, 360);
        let scaler = ContentAwareScaler::new(config);
        assert_eq!(scaler.vertical_seams_to_remove(800), 160);
        assert_eq!(scaler.horizontal_seams_to_remove(480), 120);
        assert_eq!(scaler.vertical_seams_to_remove(320), 0);
        assert_eq!(scaler.horizontal_seams_to_remove(200), 0);
    }

    #[test]
    fn test_rgb_to_luma_basic() {
        // Pure white pixel
        let pixels = vec![255u8, 255, 255, 0, 0, 0];
        let luma = rgb_to_luma(&pixels, 2, 1);
        assert_eq!(luma.len(), 2);
        assert!(luma[0] > 250); // white -> high luma
        assert_eq!(luma[1], 0); // black -> 0 luma
    }

    #[test]
    fn test_remove_vertical_seam_rgb() {
        // 3x2 image with known pixels
        let pixels = vec![
            10, 20, 30, 40, 50, 60, 70, 80, 90, // row 0: 3 pixels
            100, 110, 120, 130, 140, 150, 160, 170, 180, // row 1: 3 pixels
        ];
        let seam = Seam::new(SeamDirection::Vertical, vec![1, 1], 0.0);
        let result = remove_vertical_seam_rgb(&pixels, 3, 2, &seam);
        // Should remove column 1 from each row -> 2x2 image
        assert_eq!(result.len(), 2 * 2 * 3);
        // Row 0: pixel 0 and pixel 2
        assert_eq!(&result[0..3], &[10, 20, 30]);
        assert_eq!(&result[3..6], &[70, 80, 90]);
        // Row 1: pixel 0 and pixel 2
        assert_eq!(&result[6..9], &[100, 110, 120]);
        assert_eq!(&result[9..12], &[160, 170, 180]);
    }

    #[test]
    fn test_transpose_rgb_roundtrip() {
        let pixels = vec![
            1, 2, 3, 4, 5, 6, // row 0
            7, 8, 9, 10, 11, 12, // row 1
            13, 14, 15, 16, 17, 18, // row 2
        ];
        let transposed = transpose_rgb(&pixels, 2, 3);
        // transposed dimensions: width=3, height=2
        let back = transpose_rgb(&transposed, 3, 2);
        assert_eq!(pixels, back);
    }

    #[test]
    fn test_scale_rgb_reduces_width() {
        // 4x4 uniform grey image
        let pixels = vec![128u8; 4 * 4 * 3];
        let config = ContentAwareConfig::new(2, 4);
        let scaler = ContentAwareScaler::new(config);
        let result = scaler.scale_rgb(&pixels, 4, 4);
        assert!(result.is_some());
        let (buf, w, h) = result.expect("should succeed");
        assert_eq!(w, 2);
        assert_eq!(h, 4);
        assert_eq!(buf.len(), (w as usize) * (h as usize) * 3);
    }

    #[test]
    fn test_scale_rgb_reduces_height() {
        let pixels = vec![128u8; 4 * 4 * 3];
        let config = ContentAwareConfig::new(4, 2);
        let scaler = ContentAwareScaler::new(config);
        let result = scaler.scale_rgb(&pixels, 4, 4);
        assert!(result.is_some());
        let (buf, w, h) = result.expect("should succeed");
        assert_eq!(w, 4);
        assert_eq!(h, 2);
        assert_eq!(buf.len(), (w as usize) * (h as usize) * 3);
    }

    #[test]
    fn test_scale_rgb_invalid_input() {
        let config = ContentAwareConfig::new(2, 2);
        let scaler = ContentAwareScaler::new(config);
        // Buffer too small
        assert!(scaler.scale_rgb(&[0u8; 5], 4, 4).is_none());
    }

    #[test]
    fn test_scale_rgb_no_change_when_target_larger() {
        let pixels = vec![128u8; 4 * 4 * 3];
        let config = ContentAwareConfig::new(8, 8); // larger than source
        let scaler = ContentAwareScaler::new(config);
        let result = scaler.scale_rgb(&pixels, 4, 4);
        assert!(result.is_some());
        let (buf, w, h) = result.expect("should succeed");
        assert_eq!(w, 4);
        assert_eq!(h, 4);
        assert_eq!(buf.len(), pixels.len());
    }
}
