//! Composition analysis (rule of thirds, symmetry, balance, etc.).

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FloatImage, FrameBuffer, GrayImage};
use crate::types::CompositionAnalysis;

/// Composition analyzer.
pub struct CompositionAnalyzer;

impl CompositionAnalyzer {
    /// Create a new composition analyzer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Analyze composition of a frame.
    ///
    /// # Errors
    ///
    /// Returns error if frame is invalid.
    pub fn analyze(&self, frame: &FrameBuffer) -> ShotResult<CompositionAnalysis> {
        let shape = frame.dim();
        if shape.2 < 3 {
            return Err(ShotError::InvalidFrame(
                "Frame must have at least 3 channels".to_string(),
            ));
        }

        let rule_of_thirds = self.analyze_rule_of_thirds(frame)?;
        let symmetry = self.analyze_symmetry(frame)?;
        let balance = self.analyze_balance(frame)?;
        let leading_lines = self.analyze_leading_lines(frame)?;
        let depth = self.analyze_depth(frame)?;

        Ok(CompositionAnalysis {
            rule_of_thirds,
            symmetry,
            balance,
            leading_lines,
            depth,
        })
    }

    /// Analyze rule of thirds compliance.
    fn analyze_rule_of_thirds(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();
        let height = shape.0;
        let width = shape.1;

        // Rule of thirds intersections
        let intersections = [
            (width / 3, height / 3),
            (2 * width / 3, height / 3),
            (width / 3, 2 * height / 3),
            (2 * width / 3, 2 * height / 3),
        ];

        // Calculate saliency map (simplified)
        let saliency = self.calculate_saliency(frame)?;

        // Check if interesting content is near intersections
        let mut score = 0.0;
        for (ix, iy) in intersections {
            let window_size = 20;
            let mut local_saliency = 0.0;

            for dy in -(window_size as i32)..(window_size as i32) {
                for dx in -(window_size as i32)..(window_size as i32) {
                    let y = (iy as i32 + dy).clamp(0, height as i32 - 1) as usize;
                    let x = (ix as i32 + dx).clamp(0, width as i32 - 1) as usize;
                    local_saliency += saliency.get(y, x);
                }
            }

            score += local_saliency / (window_size * window_size * 4) as f32;
        }

        Ok((score / 4.0).min(1.0))
    }

    /// Analyze symmetry (vertical and horizontal).
    fn analyze_symmetry(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();
        let height = shape.0;
        let width = shape.1;

        // Vertical symmetry
        let mut v_symmetry = 0.0;
        let mid_x = width / 2;

        for y in 0..height {
            for dx in 0..mid_x {
                let left_x = mid_x - dx;
                let right_x = mid_x + dx;

                if right_x < width {
                    for c in 0..3 {
                        let left = f32::from(frame.get(y, left_x, c));
                        let right = f32::from(frame.get(y, right_x, c));
                        v_symmetry += 1.0 - ((left - right).abs() / 255.0);
                    }
                }
            }
        }

        v_symmetry /= (height * mid_x * 3) as f32;

        // Horizontal symmetry
        let mut h_symmetry = 0.0;
        let mid_y = height / 2;

        for x in 0..width {
            for dy in 0..mid_y {
                let top_y = mid_y - dy;
                let bottom_y = mid_y + dy;

                if bottom_y < height {
                    for c in 0..3 {
                        let top = f32::from(frame.get(top_y, x, c));
                        let bottom = f32::from(frame.get(bottom_y, x, c));
                        h_symmetry += 1.0 - ((top - bottom).abs() / 255.0);
                    }
                }
            }
        }

        h_symmetry /= (width * mid_y * 3) as f32;

        // Average of vertical and horizontal symmetry
        Ok((v_symmetry + h_symmetry) / 2.0)
    }

    /// Analyze visual balance.
    fn analyze_balance(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();
        let height = shape.0;
        let width = shape.1;

        // Calculate center of mass for visual weight
        let mut total_weight = 0.0;
        let mut weighted_x = 0.0;
        let mut weighted_y = 0.0;

        for y in 0..height {
            for x in 0..width {
                let mut pixel_weight = 0.0;
                for c in 0..3 {
                    pixel_weight += f32::from(frame.get(y, x, c));
                }
                pixel_weight /= 3.0;

                total_weight += pixel_weight;
                weighted_x += pixel_weight * x as f32;
                weighted_y += pixel_weight * y as f32;
            }
        }

        if total_weight == 0.0 {
            return Ok(0.0);
        }

        let center_of_mass_x = weighted_x / total_weight;
        let center_of_mass_y = weighted_y / total_weight;

        // Calculate distance from center
        let center_x = width as f32 / 2.0;
        let center_y = height as f32 / 2.0;

        let dx = (center_of_mass_x - center_x).abs() / center_x;
        let dy = (center_of_mass_y - center_y).abs() / center_y;

        let distance = (dx * dx + dy * dy).sqrt();

        // Balance score (1.0 = perfectly balanced)
        Ok((1.0 - distance.min(1.0)).max(0.0))
    }

    /// Analyze leading lines.
    fn analyze_leading_lines(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let gray = self.to_grayscale(frame);
        let edges = self.detect_edges(&gray);
        let shape = edges.dim();

        // Detect diagonal lines (common leading lines)
        let mut diagonal_strength = 0.0;
        let mut edge_count = 0;

        for y in 1..(shape.0.saturating_sub(1)) {
            for x in 1..(shape.1.saturating_sub(1)) {
                if edges.get(y, x) > 128 {
                    edge_count += 1;

                    // Check if this edge is part of a diagonal line
                    let is_diagonal = self.is_diagonal_edge(&edges, x, y);
                    if is_diagonal {
                        diagonal_strength += 1.0;
                    }
                }
            }
        }

        if edge_count == 0 {
            return Ok(0.0);
        }

        Ok((diagonal_strength / edge_count as f32).min(1.0))
    }

    /// Analyze depth perception.
    fn analyze_depth(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();
        let height = shape.0;

        // Analyze brightness and detail gradient from top to bottom
        let mut top_brightness = 0.0;
        let mut bottom_brightness = 0.0;
        let mut top_detail = 0.0;
        let mut bottom_detail = 0.0;

        let gray = self.to_grayscale(frame);
        let edges = self.detect_edges(&gray);

        // Top third
        for y in 0..(height / 3) {
            for x in 0..shape.1 {
                top_brightness += f32::from(gray.get(y, x));
                if edges.get(y, x) > 128 {
                    top_detail += 1.0;
                }
            }
        }

        // Bottom third
        for y in (2 * height / 3)..height {
            for x in 0..shape.1 {
                bottom_brightness += f32::from(gray.get(y, x));
                if edges.get(y, x) > 128 {
                    bottom_detail += 1.0;
                }
            }
        }

        let pixels_per_third = (height / 3 * shape.1) as f32;
        top_brightness /= pixels_per_third;
        bottom_brightness /= pixels_per_third;
        top_detail /= pixels_per_third;
        bottom_detail /= pixels_per_third;

        // Depth cues: bottom usually has more detail and different brightness
        let detail_gradient = (bottom_detail - top_detail).abs() / 255.0;
        let brightness_gradient = (bottom_brightness - top_brightness).abs() / 255.0;

        Ok(((detail_gradient + brightness_gradient) / 2.0).min(1.0))
    }

    /// Calculate saliency map.
    fn calculate_saliency(&self, frame: &FrameBuffer) -> ShotResult<FloatImage> {
        let shape = frame.dim();
        let mut saliency = FloatImage::zeros(shape.0, shape.1);

        // Simple saliency based on local contrast
        for y in 1..(shape.0.saturating_sub(1)) {
            for x in 1..(shape.1.saturating_sub(1)) {
                let mut local_variance = 0.0;

                for c in 0..3 {
                    let center = f32::from(frame.get(y, x, c));
                    let mut neighbor_sum = 0.0;

                    for dy in -1..=1 {
                        for dx in -1..=1 {
                            if dx != 0 || dy != 0 {
                                let ny = (y as i32 + dy) as usize;
                                let nx = (x as i32 + dx) as usize;
                                neighbor_sum += f32::from(frame.get(ny, nx, c));
                            }
                        }
                    }

                    let mean = neighbor_sum / 8.0;
                    local_variance += (center - mean).abs();
                }

                saliency.set(y, x, local_variance / 3.0 / 255.0);
            }
        }

        Ok(saliency)
    }

    /// Convert RGB to grayscale.
    fn to_grayscale(&self, frame: &FrameBuffer) -> GrayImage {
        let shape = frame.dim();
        let mut gray = GrayImage::zeros(shape.0, shape.1);

        for y in 0..shape.0 {
            for x in 0..shape.1 {
                let r = f32::from(frame.get(y, x, 0));
                let g = f32::from(frame.get(y, x, 1));
                let b = f32::from(frame.get(y, x, 2));
                gray.set(y, x, ((r * 0.299) + (g * 0.587) + (b * 0.114)) as u8);
            }
        }

        gray
    }

    /// Detect edges.
    fn detect_edges(&self, gray: &GrayImage) -> GrayImage {
        let shape = gray.dim();
        let mut edges = GrayImage::zeros(shape.0, shape.1);

        for y in 1..(shape.0.saturating_sub(1)) {
            for x in 1..(shape.1.saturating_sub(1)) {
                let mut gx = 0i32;
                let mut gy = 0i32;

                // Sobel operator
                gx += -i32::from(gray.get(y - 1, x - 1));
                gx += i32::from(gray.get(y - 1, x + 1));
                gx += -2 * i32::from(gray.get(y, x - 1));
                gx += 2 * i32::from(gray.get(y, x + 1));
                gx += -i32::from(gray.get(y + 1, x - 1));
                gx += i32::from(gray.get(y + 1, x + 1));

                gy += -i32::from(gray.get(y - 1, x - 1));
                gy += -2 * i32::from(gray.get(y - 1, x));
                gy += -i32::from(gray.get(y - 1, x + 1));
                gy += i32::from(gray.get(y + 1, x - 1));
                gy += 2 * i32::from(gray.get(y + 1, x));
                gy += i32::from(gray.get(y + 1, x + 1));

                let magnitude = ((gx * gx + gy * gy) as f32).sqrt();
                edges.set(y, x, magnitude.min(255.0) as u8);
            }
        }

        edges
    }

    /// Check if edge is part of a diagonal line.
    fn is_diagonal_edge(&self, edges: &GrayImage, x: usize, y: usize) -> bool {
        let shape = edges.dim();

        if y < 2 || y >= shape.0 - 2 || x < 2 || x >= shape.1 - 2 {
            return false;
        }

        // Check diagonal neighbors
        let tl = edges.get(y - 1, x - 1);
        let tr = edges.get(y - 1, x + 1);
        let bl = edges.get(y + 1, x - 1);
        let br = edges.get(y + 1, x + 1);

        // Strong diagonal if both diagonal neighbors are edges
        (tl > 128 && br > 128) || (tr > 128 && bl > 128)
    }
}

impl Default for CompositionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composition_analyzer_creation() {
        let _analyzer = CompositionAnalyzer::new();
    }

    #[test]
    fn test_analyze_uniform_frame() {
        let analyzer = CompositionAnalyzer::new();
        let frame = FrameBuffer::from_elem(100, 100, 3, 128);
        let result = analyzer.analyze(&frame);
        assert!(result.is_ok());
        if let Ok(comp) = result {
            assert!(comp.symmetry > 0.9); // Uniform frame is highly symmetric
        }
    }

    #[test]
    fn test_analyze_black_frame() {
        let analyzer = CompositionAnalyzer::new();
        let frame = FrameBuffer::zeros(100, 100, 3);
        let result = analyzer.analyze(&frame);
        assert!(result.is_ok());
    }
}
