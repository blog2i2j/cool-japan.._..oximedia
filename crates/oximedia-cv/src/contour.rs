#![allow(dead_code)]
//! Contour finding and analysis for binary/edge images.
//!
//! Provides contour extraction, shape analysis, convexity testing, and filtering.

use std::f32::consts::PI;

/// Approximation method used when extracting contours.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ContourApprox {
    /// Store every point along the contour boundary
    None,
    /// Douglas-Peucker simplification — store only inflection points
    #[default]
    Simple,
    /// Tight chain-code representation
    ChainCode,
}

impl ContourApprox {
    /// Returns `true` if this method stores all boundary pixels.
    pub fn is_lossless(self) -> bool {
        matches!(self, Self::None)
    }
}

/// A 2-D integer point used in contour coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Point2i {
    /// Horizontal pixel coordinate
    pub x: i32,
    /// Vertical pixel coordinate
    pub y: i32,
}

impl Point2i {
    /// Create a new point.
    pub const fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }
}

/// A closed or open contour represented as an ordered sequence of points.
#[derive(Debug, Clone)]
pub struct Contour {
    /// Ordered boundary points
    pub points: Vec<Point2i>,
    /// Whether this contour is a hole inside another
    pub is_hole: bool,
    /// Approximation method used during extraction
    pub approx: ContourApprox,
}

impl Contour {
    /// Create a new contour from a list of points.
    pub fn new(points: Vec<Point2i>) -> Self {
        Self {
            points,
            is_hole: false,
            approx: ContourApprox::default(),
        }
    }

    /// Mark this contour as a hole.
    #[must_use]
    pub fn as_hole(mut self) -> Self {
        self.is_hole = true;
        self
    }

    /// Number of points in the contour.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Returns `true` if the contour has no points.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Compute the signed area using the shoelace formula.
    ///
    /// Positive → counter-clockwise, negative → clockwise.
    #[allow(clippy::cast_precision_loss)]
    pub fn signed_area(&self) -> f32 {
        let n = self.points.len();
        if n < 3 {
            return 0.0;
        }
        let mut sum = 0i64;
        for i in 0..n {
            let j = (i + 1) % n;
            sum += self.points[i].x as i64 * self.points[j].y as i64;
            sum -= self.points[j].x as i64 * self.points[i].y as i64;
        }
        sum as f32 / 2.0
    }

    /// Absolute area enclosed by the contour.
    pub fn area(&self) -> f32 {
        self.signed_area().abs()
    }

    /// Perimeter (arc length) of the contour.
    #[allow(clippy::cast_precision_loss)]
    pub fn perimeter(&self) -> f32 {
        let n = self.points.len();
        if n < 2 {
            return 0.0;
        }
        let mut total = 0.0f32;
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = (self.points[j].x - self.points[i].x) as f32;
            let dy = (self.points[j].y - self.points[i].y) as f32;
            total += (dx * dx + dy * dy).sqrt();
        }
        total
    }

    /// Check convexity: a contour is convex if all cross products of consecutive
    /// edge pairs have the same sign.
    pub fn is_convex(&self) -> bool {
        let n = self.points.len();
        if n < 3 {
            return false;
        }
        let mut sign = 0i32;
        for i in 0..n {
            let a = self.points[i];
            let b = self.points[(i + 1) % n];
            let c = self.points[(i + 2) % n];
            let cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x);
            if cross != 0 {
                let s = cross.signum();
                if sign == 0 {
                    sign = s;
                } else if sign != s {
                    return false;
                }
            }
        }
        true
    }

    /// Compute the centroid of the contour points.
    #[allow(clippy::cast_precision_loss)]
    pub fn centroid(&self) -> (f32, f32) {
        if self.points.is_empty() {
            return (0.0, 0.0);
        }
        let n = self.points.len() as f32;
        let sx: i64 = self.points.iter().map(|p| p.x as i64).sum();
        let sy: i64 = self.points.iter().map(|p| p.y as i64).sum();
        (sx as f32 / n, sy as f32 / n)
    }

    /// Circularity measure: 1.0 for a perfect circle, <1.0 otherwise.
    pub fn circularity(&self) -> f32 {
        let area = self.area();
        let peri = self.perimeter();
        if peri < 1e-6 {
            return 0.0;
        }
        (4.0 * PI * area) / (peri * peri)
    }

    /// Axis-aligned bounding box: (min_x, min_y, max_x, max_y).
    pub fn bounding_box(&self) -> Option<(i32, i32, i32, i32)> {
        if self.points.is_empty() {
            return None;
        }
        let min_x = self.points.iter().map(|p| p.x).min()?;
        let min_y = self.points.iter().map(|p| p.y).min()?;
        let max_x = self.points.iter().map(|p| p.x).max()?;
        let max_y = self.points.iter().map(|p| p.y).max()?;
        Some((min_x, min_y, max_x, max_y))
    }
}

/// Configuration for contour finding.
#[derive(Debug, Clone)]
pub struct ContourConfig {
    /// Approximation method
    pub approx: ContourApprox,
    /// Minimum contour area to retain
    pub min_area: f32,
    /// Maximum contour area (0 = unlimited)
    pub max_area: f32,
    /// Whether to retrieve holes
    pub retrieve_holes: bool,
}

impl Default for ContourConfig {
    fn default() -> Self {
        Self {
            approx: ContourApprox::Simple,
            min_area: 1.0,
            max_area: 0.0,
            retrieve_holes: false,
        }
    }
}

/// Finds contours in a binary image (stored as a flat `u8` buffer, nonzero = foreground).
#[derive(Debug, Default)]
pub struct ContourFinder {
    config: ContourConfig,
}

impl ContourFinder {
    /// Create a finder with default configuration.
    pub fn new() -> Self {
        Self {
            config: ContourConfig::default(),
        }
    }

    /// Create a finder with custom configuration.
    pub fn with_config(config: ContourConfig) -> Self {
        Self { config }
    }

    /// Find contours in a binary image using a simplified boundary tracing.
    ///
    /// `pixels` is a row-major `u8` buffer of size `width × height`.
    /// Nonzero pixels are treated as foreground.
    pub fn find_contours(&self, pixels: &[u8], width: usize, height: usize) -> Vec<Contour> {
        if pixels.len() != width * height || width == 0 || height == 0 {
            return Vec::new();
        }

        let mut visited = vec![false; width * height];
        let mut contours = Vec::new();

        for y in 0..height {
            for x in 0..width {
                let idx = y * width + x;
                if pixels[idx] == 0 || visited[idx] {
                    continue;
                }
                // Check if this is a boundary pixel (has at least one background 4-neighbour)
                if !self.is_boundary(pixels, width, height, x, y) {
                    continue;
                }
                let points = self.trace_boundary(pixels, &mut visited, width, height, x, y);
                if !points.is_empty() {
                    let c = Contour::new(points);
                    contours.push(c);
                }
            }
        }
        contours
    }

    /// Filter contours that fall within the configured area range.
    pub fn filter_by_area<'a>(&self, contours: &'a [Contour]) -> Vec<&'a Contour> {
        contours
            .iter()
            .filter(|c| {
                let area = c.area();
                if area < self.config.min_area {
                    return false;
                }
                if self.config.max_area > 0.0 && area > self.config.max_area {
                    return false;
                }
                true
            })
            .collect()
    }

    /// Returns `true` if the pixel at (x, y) is a foreground boundary pixel.
    fn is_boundary(&self, pixels: &[u8], width: usize, height: usize, x: usize, y: usize) -> bool {
        let neighbours = [
            (x.wrapping_sub(1), y),
            (x + 1, y),
            (x, y.wrapping_sub(1)),
            (x, y + 1),
        ];
        for (nx, ny) in neighbours {
            if nx >= width || ny >= height || pixels[ny * width + nx] == 0 {
                return true;
            }
        }
        false
    }

    /// Simple 8-connected boundary tracing (Moore neighbour walk).
    #[allow(clippy::cast_possible_wrap)]
    fn trace_boundary(
        &self,
        pixels: &[u8],
        visited: &mut [bool],
        width: usize,
        height: usize,
        start_x: usize,
        start_y: usize,
    ) -> Vec<Point2i> {
        let directions: [(i32, i32); 8] = [
            (1, 0),
            (1, 1),
            (0, 1),
            (-1, 1),
            (-1, 0),
            (-1, -1),
            (0, -1),
            (1, -1),
        ];
        let mut points = Vec::new();
        let mut cx = start_x as i32;
        let mut cy = start_y as i32;

        loop {
            let idx = (cy as usize) * width + cx as usize;
            if visited[idx] {
                break;
            }
            visited[idx] = true;
            points.push(Point2i::new(cx, cy));

            let mut found = false;
            for (dx, dy) in directions {
                let nx = cx + dx;
                let ny = cy + dy;
                if nx < 0 || ny < 0 || nx as usize >= width || ny as usize >= height {
                    continue;
                }
                let nidx = ny as usize * width + nx as usize;
                if pixels[nidx] != 0 && !visited[nidx] {
                    cx = nx;
                    cy = ny;
                    found = true;
                    break;
                }
            }
            if !found {
                break;
            }
            // Prevent runaway traces on very large objects
            if points.len() > width * height {
                break;
            }
        }
        points
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_square_contour() -> Contour {
        // 4×4 square corners
        Contour::new(vec![
            Point2i::new(0, 0),
            Point2i::new(4, 0),
            Point2i::new(4, 4),
            Point2i::new(0, 4),
        ])
    }

    #[test]
    fn test_contour_approx_default() {
        assert_eq!(ContourApprox::default(), ContourApprox::Simple);
    }

    #[test]
    fn test_contour_approx_lossless() {
        assert!(ContourApprox::None.is_lossless());
        assert!(!ContourApprox::Simple.is_lossless());
    }

    #[test]
    fn test_contour_area_square() {
        let c = make_square_contour();
        // Shoelace area of 4×4 square = 16
        assert!((c.area() - 16.0).abs() < 1e-3);
    }

    #[test]
    fn test_contour_perimeter_square() {
        let c = make_square_contour();
        // Perimeter of 4×4 square = 16
        assert!((c.perimeter() - 16.0).abs() < 1e-3);
    }

    #[test]
    fn test_contour_is_convex_square() {
        let c = make_square_contour();
        assert!(c.is_convex());
    }

    #[test]
    fn test_contour_is_not_convex() {
        // Non-convex "L"-like shape
        let c = Contour::new(vec![
            Point2i::new(0, 0),
            Point2i::new(4, 0),
            Point2i::new(4, 2),
            Point2i::new(2, 2),
            Point2i::new(2, 4),
            Point2i::new(0, 4),
        ]);
        assert!(!c.is_convex());
    }

    #[test]
    fn test_contour_too_small_for_convex() {
        let c = Contour::new(vec![Point2i::new(0, 0), Point2i::new(1, 1)]);
        assert!(!c.is_convex()); // fewer than 3 points
    }

    #[test]
    fn test_contour_centroid() {
        let c = make_square_contour();
        let (cx, cy) = c.centroid();
        assert!((cx - 2.0).abs() < 1e-3);
        assert!((cy - 2.0).abs() < 1e-3);
    }

    #[test]
    fn test_contour_bounding_box() {
        let c = make_square_contour();
        let bb = c.bounding_box().expect("bounding_box should succeed");
        assert_eq!(bb, (0, 0, 4, 4));
    }

    #[test]
    fn test_contour_empty() {
        let c = Contour::new(vec![]);
        assert!(c.is_empty());
        assert!((c.area()).abs() < 1e-6);
        assert!((c.perimeter()).abs() < 1e-6);
        assert!(c.bounding_box().is_none());
    }

    #[test]
    fn test_contour_is_hole_flag() {
        let c = Contour::new(vec![Point2i::new(0, 0)]).as_hole();
        assert!(c.is_hole);
    }

    #[test]
    fn test_finder_empty_image() {
        let finder = ContourFinder::new();
        let result = finder.find_contours(&[], 0, 0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_finder_all_black() {
        let finder = ContourFinder::new();
        let pixels = vec![0u8; 10 * 10];
        let result = finder.find_contours(&pixels, 10, 10);
        assert!(result.is_empty());
    }

    #[test]
    fn test_finder_single_pixel() {
        let finder = ContourFinder::new();
        let mut pixels = vec![0u8; 5 * 5];
        pixels[2 * 5 + 2] = 255; // centre pixel
        let result = finder.find_contours(&pixels, 5, 5);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_filter_by_area() {
        let finder = ContourFinder::with_config(ContourConfig {
            min_area: 5.0,
            max_area: 100.0,
            ..Default::default()
        });
        let big = make_square_contour(); // area = 16 → passes
        let tiny = Contour::new(vec![
            Point2i::new(0, 0),
            Point2i::new(1, 0),
            Point2i::new(0, 1),
        ]); // area = 0.5 → filtered
        let all = vec![big, tiny];
        let kept = finder.filter_by_area(&all);
        assert_eq!(kept.len(), 1);
        assert!((kept[0].area() - 16.0).abs() < 1e-3);
    }

    #[test]
    fn test_point2i_equality() {
        assert_eq!(Point2i::new(3, 4), Point2i::new(3, 4));
        assert_ne!(Point2i::new(3, 4), Point2i::new(4, 3));
    }

    #[test]
    fn test_contour_circularity_square_lt_one() {
        let c = make_square_contour();
        // Square circularity ≈ PI/4 ≈ 0.785 — must be < 1
        assert!(c.circularity() < 1.0);
        assert!(c.circularity() > 0.0);
    }
}
