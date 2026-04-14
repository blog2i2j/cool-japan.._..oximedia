//! Composition analysis (rule of thirds, symmetry, balance, etc.).

use crate::error::{ShotError, ShotResult};
use crate::frame_buffer::{FloatImage, FrameBuffer, GrayImage};
use crate::types::CompositionAnalysis;

/// Extended composition analysis with golden ratio and phi grid scores.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtendedComposition {
    /// Standard composition analysis.
    pub base: CompositionAnalysis,
    /// Golden ratio compliance (0.0-1.0). Measures how well key visual
    /// elements align with the golden ratio (1.618:1) divisions.
    pub golden_ratio: f32,
    /// Phi grid compliance (0.0-1.0). The phi grid places lines at 1/phi
    /// (~0.382) and 1 - 1/phi (~0.618) of each dimension, a tighter
    /// alternative to the rule of thirds.
    pub phi_grid: f32,
    /// Golden spiral conformity (0.0-1.0). Measures how well salient
    /// regions follow a logarithmic spiral based on the golden ratio.
    pub golden_spiral: f32,
    /// Diagonal dominance score (0.0-1.0). Measures alignment with the
    /// baroque (lower-left to upper-right) and sinister (upper-left to
    /// lower-right) diagonals.
    pub diagonal_dominance: f32,
}

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

    /// Perform extended composition analysis including golden ratio and phi grid.
    ///
    /// # Errors
    ///
    /// Returns error if frame is invalid.
    pub fn analyze_extended(&self, frame: &FrameBuffer) -> ShotResult<ExtendedComposition> {
        let base = self.analyze(frame)?;
        let golden_ratio = self.analyze_golden_ratio(frame)?;
        let phi_grid = self.analyze_phi_grid(frame)?;
        let golden_spiral = self.analyze_golden_spiral(frame)?;
        let diagonal_dominance = self.analyze_diagonal_dominance(frame)?;

        Ok(ExtendedComposition {
            base,
            golden_ratio,
            phi_grid,
            golden_spiral,
            diagonal_dominance,
        })
    }

    /// Analyse golden ratio compliance.
    ///
    /// Divides the frame at 1:phi and phi:1 ratios (approximately 38.2%
    /// and 61.8%) on both axes, then measures saliency concentration near
    /// the four intersection points.
    fn analyze_golden_ratio(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();
        let height = shape.0;
        let width = shape.1;
        let phi_inv: f32 = 1.0 / 1.618_034;

        // Golden ratio division points
        let gx1 = (width as f32 * phi_inv) as usize;
        let gx2 = (width as f32 * (1.0 - phi_inv)) as usize;
        let gy1 = (height as f32 * phi_inv) as usize;
        let gy2 = (height as f32 * (1.0 - phi_inv)) as usize;

        let intersections = [(gx1, gy1), (gx2, gy1), (gx1, gy2), (gx2, gy2)];
        let saliency = self.calculate_saliency(frame)?;

        let window = (width.min(height) / 10).max(3);
        let mut score = 0.0_f32;

        for (ix, iy) in intersections {
            let mut local_sal = 0.0_f32;
            let mut count = 0u32;
            let y_start = iy.saturating_sub(window);
            let y_end = (iy + window).min(height);
            let x_start = ix.saturating_sub(window);
            let x_end = (ix + window).min(width);

            for y in y_start..y_end {
                for x in x_start..x_end {
                    local_sal += saliency.get(y, x);
                    count += 1;
                }
            }
            if count > 0 {
                score += local_sal / count as f32;
            }
        }
        Ok((score / 4.0).min(1.0))
    }

    /// Analyse phi grid compliance.
    ///
    /// The phi grid lines sit at ~38.2% and ~61.8% (exactly 1/phi and
    /// 1-1/phi). This measures how much edge energy concentrates along
    /// those four lines.
    fn analyze_phi_grid(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let gray = self.to_grayscale(frame);
        let edges = self.detect_edges(&gray);
        let (h, w) = edges.dim();
        let phi_inv: f32 = 1.0 / 1.618_034;

        let line_positions_x = [
            (w as f32 * phi_inv) as usize,
            (w as f32 * (1.0 - phi_inv)) as usize,
        ];
        let line_positions_y = [
            (h as f32 * phi_inv) as usize,
            (h as f32 * (1.0 - phi_inv)) as usize,
        ];

        let tolerance = (w.min(h) / 20).max(2);
        let mut on_grid_energy = 0.0_f64;
        let mut total_energy = 0.0_f64;

        for y in 0..h {
            for x in 0..w {
                let e = f64::from(edges.get(y, x));
                total_energy += e;

                let near_v = line_positions_x
                    .iter()
                    .any(|&lx| x.abs_diff(lx) <= tolerance);
                let near_h = line_positions_y
                    .iter()
                    .any(|&ly| y.abs_diff(ly) <= tolerance);

                if near_v || near_h {
                    on_grid_energy += e;
                }
            }
        }

        if total_energy < 1.0 {
            return Ok(0.0);
        }
        // Normalise: ideal fraction of area covered by grid tolerance bands
        let grid_area_fraction = {
            let band_v = 2 * tolerance * h * line_positions_x.len();
            let band_h = 2 * tolerance * w * line_positions_y.len();
            (band_v + band_h) as f64 / (w * h).max(1) as f64
        };
        let energy_fraction = on_grid_energy / total_energy;
        let ratio = if grid_area_fraction > 0.0 {
            (energy_fraction / grid_area_fraction) as f32
        } else {
            0.0
        };
        // A ratio > 1 means edges concentrate on the grid more than random
        Ok(((ratio - 1.0).max(0.0) / 2.0).min(1.0) + (energy_fraction as f32).min(0.5))
    }

    /// Analyse golden spiral conformity.
    ///
    /// Approximates a golden spiral using quarter-circle arcs in
    /// progressively smaller golden rectangles. Measures how much
    /// saliency sits near the spiral path.
    fn analyze_golden_spiral(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let shape = frame.dim();
        let height = shape.0;
        let width = shape.1;
        let saliency = self.calculate_saliency(frame)?;

        let phi: f64 = 1.618_033_988_749_895;
        let mut spiral_points: Vec<(usize, usize)> = Vec::with_capacity(200);

        // Generate spiral points via successive golden rectangle subdivisions
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let mut radius = (width.min(height) as f64) / 2.0;
        let mut angle = 0.0_f64;

        for _ in 0..80 {
            let px = cx + radius * angle.cos();
            let py = cy + radius * angle.sin();
            let ix = (px as usize).min(width.saturating_sub(1));
            let iy = (py as usize).min(height.saturating_sub(1));
            spiral_points.push((ix, iy));
            angle += std::f64::consts::PI / 12.0;
            radius /= phi.powf(1.0 / 6.0);
        }

        let tolerance = (width.min(height) / 15).max(3);
        let mut spiral_sal = 0.0_f32;
        let mut total_sal = 0.0_f32;

        for y in 0..height {
            for x in 0..width {
                let s = saliency.get(y, x);
                total_sal += s;
                let near_spiral = spiral_points
                    .iter()
                    .any(|&(sx, sy)| x.abs_diff(sx) <= tolerance && y.abs_diff(sy) <= tolerance);
                if near_spiral {
                    spiral_sal += s;
                }
            }
        }

        if total_sal < f32::EPSILON {
            return Ok(0.0);
        }
        Ok((spiral_sal / total_sal).min(1.0))
    }

    /// Analyse diagonal dominance.
    ///
    /// Measures how strongly edges align with the two main diagonals:
    /// baroque (lower-left to upper-right) and sinister (upper-left to
    /// lower-right).
    fn analyze_diagonal_dominance(&self, frame: &FrameBuffer) -> ShotResult<f32> {
        let gray = self.to_grayscale(frame);
        let edges = self.detect_edges(&gray);
        let (h, w) = edges.dim();
        if h == 0 || w == 0 {
            return Ok(0.0);
        }

        let tolerance = (w.min(h) / 20).max(2);
        let mut diag_energy = 0.0_f64;
        let mut total_energy = 0.0_f64;

        for y in 0..h {
            for x in 0..w {
                let e = f64::from(edges.get(y, x));
                total_energy += e;

                // Sinister diagonal: y/h ~ x/w  =>  y*w ~ x*h
                let sinister_dist = ((y * w) as i64 - (x * h) as i64).unsigned_abs();
                let sinister_norm = sinister_dist as f64 / (w * h).max(1) as f64;

                // Baroque diagonal: y/h ~ 1 - x/w  =>  y*w + x*h ~ w*h
                let baroque_val = (y * w + x * h) as f64;
                let baroque_dist = (baroque_val - (w * h) as f64).abs();
                let baroque_norm = baroque_dist / (w * h).max(1) as f64;

                let near = sinister_norm < (tolerance as f64 / h.max(1) as f64)
                    || baroque_norm < (tolerance as f64 / h.max(1) as f64);
                if near {
                    diag_energy += e;
                }
            }
        }

        if total_energy < 1.0 {
            return Ok(0.0);
        }
        Ok((diag_energy / total_energy).min(1.0) as f32)
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

    fn make_frame(h: usize, w: usize, r: u8, g: u8, b: u8) -> FrameBuffer {
        let mut frame = FrameBuffer::zeros(h, w, 3);
        for y in 0..h {
            for x in 0..w {
                frame.set(y, x, 0, r);
                frame.set(y, x, 1, g);
                frame.set(y, x, 2, b);
            }
        }
        frame
    }

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

    // ── Golden ratio / phi grid extension tests (TODO item 3) ──────────────

    /// `analyze_extended` must succeed on a valid frame.
    #[test]
    fn test_analyze_extended_returns_ok() {
        let analyzer = CompositionAnalyzer::new();
        let frame = make_frame(80, 80, 150, 100, 50);
        let result = analyzer.analyze_extended(&frame);
        assert!(
            result.is_ok(),
            "analyze_extended should succeed on a valid frame"
        );
    }

    /// All extended scores must be in [0.0, 1.0].
    #[test]
    fn test_analyze_extended_scores_in_range() {
        let analyzer = CompositionAnalyzer::new();
        let frame = make_frame(100, 120, 80, 80, 200);
        let ext = analyzer.analyze_extended(&frame).expect("should succeed");
        assert!(
            ext.golden_ratio >= 0.0 && ext.golden_ratio <= 1.0,
            "golden_ratio out of range: {}",
            ext.golden_ratio
        );
        assert!(
            ext.phi_grid >= 0.0 && ext.phi_grid <= 1.0,
            "phi_grid out of range: {}",
            ext.phi_grid
        );
        assert!(
            ext.golden_spiral >= 0.0 && ext.golden_spiral <= 1.0,
            "golden_spiral out of range: {}",
            ext.golden_spiral
        );
        assert!(
            ext.diagonal_dominance >= 0.0 && ext.diagonal_dominance <= 1.0,
            "diagonal_dominance out of range: {}",
            ext.diagonal_dominance
        );
    }

    /// `analyze_extended` returns an error for a frame with fewer than 3 channels.
    #[test]
    fn test_analyze_extended_wrong_channels_error() {
        let analyzer = CompositionAnalyzer::new();
        let frame = FrameBuffer::zeros(50, 50, 1);
        assert!(
            analyzer.analyze_extended(&frame).is_err(),
            "should fail on 1-channel frame"
        );
    }

    /// The `base` field in `ExtendedComposition` contains the standard analysis.
    #[test]
    fn test_analyze_extended_base_field_populated() {
        let analyzer = CompositionAnalyzer::new();
        // A uniform frame should have high symmetry in the base result.
        let frame = make_frame(64, 64, 200, 200, 200);
        let ext = analyzer.analyze_extended(&frame).expect("should succeed");
        assert!(
            ext.base.symmetry > 0.8,
            "uniform frame base symmetry should be high: {}",
            ext.base.symmetry
        );
    }

    /// A frame with content at the phi-grid intersection (≈38.2% / 61.8%) should
    /// score higher on golden_ratio than a completely uniform frame, because the
    /// luminance-weighted energy is aligned with the phi divisions.
    #[test]
    fn test_analyze_extended_phi_grid_vs_uniform() {
        let analyzer = CompositionAnalyzer::new();

        // Uniform frame — no specific edge alignment
        let uniform = make_frame(100, 100, 128, 128, 128);

        // Frame with a bright vertical band at the phi-grid split (≈38.2% of width)
        let phi_x = 38usize; // ≈38.2% of 100
        let mut phi_frame = FrameBuffer::zeros(100, 100, 3);
        for y in 0..100 {
            for x in 0..100 {
                // Bright band of 2 pixels centred on the phi column
                let v = if x == phi_x || x == phi_x + 1 {
                    255
                } else {
                    50
                };
                phi_frame.set(y, x, 0, v);
                phi_frame.set(y, x, 1, v);
                phi_frame.set(y, x, 2, v);
            }
        }

        let ext_uniform = analyzer.analyze_extended(&uniform).expect("uniform ok");
        let ext_phi = analyzer.analyze_extended(&phi_frame).expect("phi_frame ok");

        // The phi-aligned frame should have at least as high a phi_grid score as uniform.
        assert!(
            ext_phi.phi_grid >= ext_uniform.phi_grid - 0.05,
            "phi-aligned frame should not score lower than uniform: phi_frame={}, uniform={}",
            ext_phi.phi_grid,
            ext_uniform.phi_grid
        );
    }

    /// The extended analysis is consistent: calling it twice on the same frame
    /// must return identical results.
    #[test]
    fn test_analyze_extended_deterministic() {
        let analyzer = CompositionAnalyzer::new();
        let frame = make_frame(80, 80, 90, 120, 60);
        let r1 = analyzer.analyze_extended(&frame).expect("first call ok");
        let r2 = analyzer.analyze_extended(&frame).expect("second call ok");
        assert!(
            (r1.golden_ratio - r2.golden_ratio).abs() < f32::EPSILON,
            "golden_ratio not deterministic"
        );
        assert!(
            (r1.phi_grid - r2.phi_grid).abs() < f32::EPSILON,
            "phi_grid not deterministic"
        );
        assert!(
            (r1.golden_spiral - r2.golden_spiral).abs() < f32::EPSILON,
            "golden_spiral not deterministic"
        );
    }

    /// A frame with a clear diagonal edge should have higher diagonal_dominance
    /// than a completely uniform frame.
    #[test]
    fn test_analyze_extended_diagonal_frame_higher_dominance() {
        let analyzer = CompositionAnalyzer::new();
        let uniform = make_frame(80, 80, 100, 100, 100);

        // Diagonal split: upper-left half white, lower-right half black
        let mut diag = FrameBuffer::zeros(80, 80, 3);
        for y in 0..80 {
            for x in 0..80 {
                let v = if x + y < 80 { 255 } else { 0 };
                diag.set(y, x, 0, v);
                diag.set(y, x, 1, v);
                diag.set(y, x, 2, v);
            }
        }

        let ext_uniform = analyzer.analyze_extended(&uniform).expect("uniform ok");
        let ext_diag = analyzer.analyze_extended(&diag).expect("diag ok");

        assert!(
            ext_diag.diagonal_dominance >= ext_uniform.diagonal_dominance,
            "diagonal frame should have >= diagonal_dominance vs uniform: diag={}, uniform={}",
            ext_diag.diagonal_dominance,
            ext_uniform.diagonal_dominance
        );
    }

    /// Works on a minimal 4×4 frame without panicking.
    #[test]
    fn test_analyze_extended_tiny_frame() {
        let analyzer = CompositionAnalyzer::new();
        let frame = make_frame(4, 4, 200, 100, 50);
        let result = analyzer.analyze_extended(&frame);
        assert!(result.is_ok(), "should handle 4×4 frame without error");
    }

    /// Works on a non-square frame (landscape aspect ratio).
    #[test]
    fn test_analyze_extended_landscape_frame() {
        let analyzer = CompositionAnalyzer::new();
        let frame = make_frame(54, 96, 100, 150, 200);
        let ext = analyzer.analyze_extended(&frame).expect("landscape frame");
        assert!(ext.golden_ratio >= 0.0 && ext.golden_ratio <= 1.0);
        assert!(ext.phi_grid >= 0.0 && ext.phi_grid <= 1.0);
    }

    /// `ExtendedComposition` preserves the base rule_of_thirds from `analyze()`.
    #[test]
    fn test_analyze_extended_base_matches_analyze() {
        let analyzer = CompositionAnalyzer::new();
        let frame = make_frame(100, 100, 50, 50, 50);
        let base = analyzer.analyze(&frame).expect("analyze ok");
        let ext = analyzer
            .analyze_extended(&frame)
            .expect("analyze_extended ok");
        assert!(
            (ext.base.rule_of_thirds - base.rule_of_thirds).abs() < 0.001,
            "extended base.rule_of_thirds should match analyze(): {} vs {}",
            ext.base.rule_of_thirds,
            base.rule_of_thirds
        );
    }
}
