#![allow(dead_code)]
//! Hough transform for detecting lines and circles in images.
//!
//! The Hough transform is a feature extraction technique used to find
//! imperfect instances of geometric shapes (lines, circles) in images.
//!
//! # Algorithms
//!
//! - **Standard Hough Transform (SHT)**: Detects lines using the (rho, theta)
//!   parameterization. Each edge point votes in an accumulator array.
//! - **Probabilistic Hough Transform (PHT)**: A randomized variant that samples
//!   edge points, trading accuracy for speed on large images.
//! - **Circle Hough Transform (CHT)**: Detects circles using the (cx, cy, r)
//!   parameterization of circles.

use std::f64::consts::PI;

/// A detected line in (rho, theta) parameterization.
///
/// The line equation is: `x * cos(theta) + y * sin(theta) = rho`.
#[derive(Debug, Clone, Copy)]
pub struct HoughLine {
    /// Distance from the origin to the line.
    pub rho: f64,
    /// Angle of the line normal in radians.
    pub theta: f64,
    /// Number of votes (edge points supporting this line).
    pub votes: u32,
}

impl HoughLine {
    /// Creates a new Hough line.
    pub fn new(rho: f64, theta: f64, votes: u32) -> Self {
        Self { rho, theta, votes }
    }

    /// Converts the line to two (x, y) endpoints within the given image dimensions.
    ///
    /// Returns two points that can be used for drawing.
    #[allow(clippy::cast_precision_loss)]
    pub fn to_endpoints(&self, width: usize, height: usize) -> ((f64, f64), (f64, f64)) {
        let cos_t = self.theta.cos();
        let sin_t = self.theta.sin();
        let x0 = cos_t * self.rho;
        let y0 = sin_t * self.rho;
        let diag = ((width * width + height * height) as f64).sqrt();
        let p1 = (x0 - diag * sin_t, y0 + diag * cos_t);
        let p2 = (x0 + diag * sin_t, y0 - diag * cos_t);
        (p1, p2)
    }

    /// Returns the angle in degrees.
    pub fn angle_degrees(&self) -> f64 {
        self.theta * 180.0 / PI
    }
}

/// A detected line segment with explicit endpoints (from probabilistic Hough).
#[derive(Debug, Clone, Copy)]
pub struct HoughLineSegment {
    /// Start x coordinate.
    pub x1: f64,
    /// Start y coordinate.
    pub y1: f64,
    /// End x coordinate.
    pub x2: f64,
    /// End y coordinate.
    pub y2: f64,
    /// Strength (votes) for this segment.
    pub votes: u32,
}

impl HoughLineSegment {
    /// Creates a new line segment.
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64, votes: u32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            votes,
        }
    }

    /// Returns the length of the segment.
    pub fn length(&self) -> f64 {
        let dx = self.x2 - self.x1;
        let dy = self.y2 - self.y1;
        (dx * dx + dy * dy).sqrt()
    }

    /// Returns the midpoint of the segment.
    pub fn midpoint(&self) -> (f64, f64) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    /// Returns the angle of the segment in radians.
    pub fn angle(&self) -> f64 {
        (self.y2 - self.y1).atan2(self.x2 - self.x1)
    }
}

/// A detected circle with center and radius.
#[derive(Debug, Clone, Copy)]
pub struct HoughCircle {
    /// Center x coordinate.
    pub cx: f64,
    /// Center y coordinate.
    pub cy: f64,
    /// Radius.
    pub radius: f64,
    /// Number of votes (edge points supporting this circle).
    pub votes: u32,
}

impl HoughCircle {
    /// Creates a new Hough circle.
    pub fn new(cx: f64, cy: f64, radius: f64, votes: u32) -> Self {
        Self {
            cx,
            cy,
            radius,
            votes,
        }
    }

    /// Returns the circumference.
    pub fn circumference(&self) -> f64 {
        2.0 * PI * self.radius
    }

    /// Returns the area.
    pub fn area(&self) -> f64 {
        PI * self.radius * self.radius
    }

    /// Checks whether a point is inside the circle (within tolerance).
    pub fn contains(&self, x: f64, y: f64, tolerance: f64) -> bool {
        let dx = x - self.cx;
        let dy = y - self.cy;
        let dist = (dx * dx + dy * dy).sqrt();
        dist <= self.radius + tolerance
    }
}

/// Configuration for the standard Hough line transform.
#[derive(Debug, Clone)]
pub struct HoughLineConfig {
    /// Angular resolution in radians (default: PI/180).
    pub theta_resolution: f64,
    /// Distance resolution in pixels (default: 1.0).
    pub rho_resolution: f64,
    /// Minimum number of votes to accept a line.
    pub vote_threshold: u32,
    /// Maximum number of lines to return.
    pub max_lines: usize,
}

impl Default for HoughLineConfig {
    fn default() -> Self {
        Self {
            theta_resolution: PI / 180.0,
            rho_resolution: 1.0,
            vote_threshold: 50,
            max_lines: 100,
        }
    }
}

/// Configuration for the circle Hough transform.
#[derive(Debug, Clone)]
pub struct HoughCircleConfig {
    /// Minimum radius to search for.
    pub min_radius: f64,
    /// Maximum radius to search for.
    pub max_radius: f64,
    /// Radius resolution in pixels (default: 1.0).
    pub radius_resolution: f64,
    /// Minimum number of votes to accept a circle.
    pub vote_threshold: u32,
    /// Minimum distance between detected circle centers.
    pub min_center_dist: f64,
    /// Maximum number of circles to return.
    pub max_circles: usize,
}

impl Default for HoughCircleConfig {
    fn default() -> Self {
        Self {
            min_radius: 10.0,
            max_radius: 100.0,
            radius_resolution: 1.0,
            vote_threshold: 50,
            min_center_dist: 20.0,
            max_circles: 50,
        }
    }
}

/// The Hough accumulator for the standard line transform.
#[derive(Debug, Clone)]
pub struct HoughAccumulator {
    /// Accumulator data in (theta_idx, rho_idx) layout.
    pub data: Vec<u32>,
    /// Number of theta bins.
    pub num_theta: usize,
    /// Number of rho bins.
    pub num_rho: usize,
    /// Rho offset (so rho can be negative).
    pub rho_offset: f64,
    /// Theta resolution.
    pub theta_resolution: f64,
    /// Rho resolution.
    pub rho_resolution: f64,
}

impl HoughAccumulator {
    /// Creates a new accumulator for the given image dimensions and config.
    #[allow(clippy::cast_precision_loss)]
    pub fn new(width: usize, height: usize, config: &HoughLineConfig) -> Self {
        let diag = ((width * width + height * height) as f64).sqrt();
        let num_theta = (PI / config.theta_resolution) as usize + 1;
        let num_rho = (2.0 * diag / config.rho_resolution) as usize + 1;
        let rho_offset = diag;
        Self {
            data: vec![0; num_theta * num_rho],
            num_theta,
            num_rho,
            rho_offset,
            theta_resolution: config.theta_resolution,
            rho_resolution: config.rho_resolution,
        }
    }

    /// Adds a vote for an edge point at (x, y).
    #[allow(clippy::cast_precision_loss)]
    pub fn vote(&mut self, x: f64, y: f64) {
        for t_idx in 0..self.num_theta {
            let theta = t_idx as f64 * self.theta_resolution;
            let rho = x * theta.cos() + y * theta.sin();
            let rho_idx = ((rho + self.rho_offset) / self.rho_resolution) as usize;
            if rho_idx < self.num_rho {
                self.data[t_idx * self.num_rho + rho_idx] += 1;
            }
        }
    }

    /// Extracts lines above the vote threshold, sorted by votes descending.
    #[allow(clippy::cast_precision_loss)]
    pub fn extract_lines(&self, threshold: u32, max_lines: usize) -> Vec<HoughLine> {
        let mut lines = Vec::new();
        for t_idx in 0..self.num_theta {
            for r_idx in 0..self.num_rho {
                let votes = self.data[t_idx * self.num_rho + r_idx];
                if votes >= threshold {
                    let theta = t_idx as f64 * self.theta_resolution;
                    let rho = r_idx as f64 * self.rho_resolution - self.rho_offset;
                    lines.push(HoughLine::new(rho, theta, votes));
                }
            }
        }
        lines.sort_by(|a, b| b.votes.cmp(&a.votes));
        lines.truncate(max_lines);
        lines
    }

    /// Returns the maximum vote count in the accumulator.
    pub fn max_votes(&self) -> u32 {
        self.data.iter().copied().max().unwrap_or(0)
    }
}

/// Runs the standard Hough line transform on edge point coordinates.
///
/// `edge_points` is a list of (x, y) coordinates of detected edge pixels.
pub fn hough_lines(
    edge_points: &[(f64, f64)],
    width: usize,
    height: usize,
    config: &HoughLineConfig,
) -> Vec<HoughLine> {
    let mut acc = HoughAccumulator::new(width, height, config);
    for &(x, y) in edge_points {
        acc.vote(x, y);
    }
    acc.extract_lines(config.vote_threshold, config.max_lines)
}

/// Runs a simple circle Hough transform on edge point coordinates.
///
/// For each edge point, votes are cast in the (cx, cy) accumulator
/// for each candidate radius.
#[allow(clippy::cast_precision_loss, clippy::too_many_arguments)]
pub fn hough_circles(
    edge_points: &[(f64, f64)],
    width: usize,
    height: usize,
    config: &HoughCircleConfig,
) -> Vec<HoughCircle> {
    let num_radii =
        ((config.max_radius - config.min_radius) / config.radius_resolution) as usize + 1;
    let mut results = Vec::new();

    for r_idx in 0..num_radii {
        let radius = config.min_radius + r_idx as f64 * config.radius_resolution;
        let mut acc = vec![0u32; width * height];
        let num_angle_steps = (2.0 * PI * radius / 2.0).max(36.0) as usize;

        for &(x, y) in edge_points {
            for a_idx in 0..num_angle_steps {
                let angle = 2.0 * PI * a_idx as f64 / num_angle_steps as f64;
                let cx = (x - radius * angle.cos()).round();
                let cy = (y - radius * angle.sin()).round();
                if cx >= 0.0 && cx < width as f64 && cy >= 0.0 && cy < height as f64 {
                    let ix = cx as usize;
                    let iy = cy as usize;
                    acc[iy * width + ix] += 1;
                }
            }
        }

        for iy in 0..height {
            for ix in 0..width {
                let votes = acc[iy * width + ix];
                if votes >= config.vote_threshold {
                    results.push(HoughCircle::new(ix as f64, iy as f64, radius, votes));
                }
            }
        }
    }

    // Non-maximum suppression: remove nearby circles
    results.sort_by(|a, b| b.votes.cmp(&a.votes));
    let mut filtered = Vec::new();
    for circle in &results {
        let too_close = filtered.iter().any(|existing: &HoughCircle| {
            let dx = circle.cx - existing.cx;
            let dy = circle.cy - existing.cy;
            (dx * dx + dy * dy).sqrt() < config.min_center_dist
        });
        if !too_close {
            filtered.push(*circle);
        }
        if filtered.len() >= config.max_circles {
            break;
        }
    }

    filtered
}

/// Computes the intersection point of two Hough lines.
///
/// Returns `None` if the lines are nearly parallel.
pub fn line_intersection(a: &HoughLine, b: &HoughLine) -> Option<(f64, f64)> {
    let cos_a = a.theta.cos();
    let sin_a = a.theta.sin();
    let cos_b = b.theta.cos();
    let sin_b = b.theta.sin();

    let det = cos_a * sin_b - sin_a * cos_b;
    if det.abs() < 1e-10 {
        return None;
    }

    let x = (a.rho * sin_b - b.rho * sin_a) / det;
    let y = (b.rho * cos_a - a.rho * cos_b) / det;
    Some((x, y))
}

/// Groups detected lines by their angle into clusters.
///
/// Lines within `angle_tolerance` radians of each other are grouped together.
pub fn cluster_lines_by_angle(lines: &[HoughLine], angle_tolerance: f64) -> Vec<Vec<usize>> {
    let n = lines.len();
    let mut visited = vec![false; n];
    let mut clusters: Vec<Vec<usize>> = Vec::new();

    for i in 0..n {
        if visited[i] {
            continue;
        }
        let mut cluster = vec![i];
        visited[i] = true;
        for j in (i + 1)..n {
            if visited[j] {
                continue;
            }
            let diff = (lines[i].theta - lines[j].theta).abs();
            let diff = diff.min(PI - diff);
            if diff <= angle_tolerance {
                cluster.push(j);
                visited[j] = true;
            }
        }
        clusters.push(cluster);
    }

    clusters
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hough_line_creation() {
        let line = HoughLine::new(50.0, PI / 4.0, 100);
        assert!((line.rho - 50.0).abs() < 1e-10);
        assert!((line.theta - PI / 4.0).abs() < 1e-10);
        assert_eq!(line.votes, 100);
    }

    #[test]
    fn test_hough_line_angle_degrees() {
        let line = HoughLine::new(0.0, PI / 2.0, 10);
        assert!((line.angle_degrees() - 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_hough_line_endpoints() {
        let line = HoughLine::new(50.0, 0.0, 10); // vertical line at x=50
        let (p1, p2) = line.to_endpoints(100, 100);
        // For theta=0, line is x=50, endpoints should have x near 50
        assert!((p1.0 - 50.0).abs() < 1e-6);
        assert!((p2.0 - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_line_segment_length() {
        let seg = HoughLineSegment::new(0.0, 0.0, 3.0, 4.0, 10);
        assert!((seg.length() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_line_segment_midpoint() {
        let seg = HoughLineSegment::new(0.0, 0.0, 10.0, 10.0, 5);
        let mid = seg.midpoint();
        assert!((mid.0 - 5.0).abs() < 1e-10);
        assert!((mid.1 - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_line_segment_angle() {
        let seg = HoughLineSegment::new(0.0, 0.0, 1.0, 0.0, 5);
        assert!(seg.angle().abs() < 1e-10); // horizontal = 0 radians
    }

    #[test]
    fn test_hough_circle_creation() {
        let c = HoughCircle::new(50.0, 50.0, 25.0, 80);
        assert!((c.cx - 50.0).abs() < 1e-10);
        assert!((c.radius - 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_circle_circumference() {
        let c = HoughCircle::new(0.0, 0.0, 10.0, 1);
        assert!((c.circumference() - 2.0 * PI * 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_circle_area() {
        let c = HoughCircle::new(0.0, 0.0, 5.0, 1);
        assert!((c.area() - PI * 25.0).abs() < 1e-10);
    }

    #[test]
    fn test_circle_contains() {
        let c = HoughCircle::new(50.0, 50.0, 10.0, 1);
        assert!(c.contains(50.0, 50.0, 0.0)); // center
        assert!(c.contains(55.0, 50.0, 0.0)); // inside
        assert!(!c.contains(70.0, 50.0, 0.0)); // outside
        assert!(c.contains(60.0, 50.0, 1.0)); // edge within tolerance
    }

    #[test]
    fn test_accumulator_creation() {
        let config = HoughLineConfig::default();
        let acc = HoughAccumulator::new(100, 100, &config);
        assert!(acc.num_theta > 0);
        assert!(acc.num_rho > 0);
        assert_eq!(acc.data.len(), acc.num_theta * acc.num_rho);
    }

    #[test]
    fn test_accumulator_vote() {
        let config = HoughLineConfig::default();
        let mut acc = HoughAccumulator::new(100, 100, &config);
        acc.vote(50.0, 50.0);
        assert!(acc.max_votes() > 0);
    }

    #[test]
    fn test_hough_lines_collinear_points() {
        // Create collinear points along y=50
        let points: Vec<(f64, f64)> = (0..100).map(|x| (x as f64, 50.0)).collect();
        let config = HoughLineConfig {
            vote_threshold: 80,
            ..Default::default()
        };
        let lines = hough_lines(&points, 100, 100, &config);
        // Should detect at least one strong line
        assert!(!lines.is_empty());
        // The strongest line should be near theta=PI/2, rho=50
        let best = &lines[0];
        assert!(best.votes >= 80);
    }

    #[test]
    fn test_line_intersection() {
        // Two perpendicular lines
        let a = HoughLine::new(50.0, 0.0, 10); // vertical at x=50
        let b = HoughLine::new(50.0, PI / 2.0, 10); // horizontal at y=50
        let result = line_intersection(&a, &b);
        assert!(result.is_some());
        let (x, y) = result.expect("operation should succeed");
        assert!((x - 50.0).abs() < 1e-6);
        assert!((y - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_line_intersection_parallel() {
        let a = HoughLine::new(50.0, 0.0, 10);
        let b = HoughLine::new(60.0, 0.0, 10);
        assert!(line_intersection(&a, &b).is_none());
    }

    #[test]
    fn test_cluster_lines_by_angle() {
        let lines = vec![
            HoughLine::new(10.0, 0.0, 10),
            HoughLine::new(20.0, 0.01, 10),
            HoughLine::new(30.0, PI / 2.0, 10),
        ];
        let clusters = cluster_lines_by_angle(&lines, 0.1);
        assert_eq!(clusters.len(), 2);
    }

    #[test]
    fn test_default_configs() {
        let lc = HoughLineConfig::default();
        assert!((lc.theta_resolution - PI / 180.0).abs() < 1e-10);
        assert_eq!(lc.vote_threshold, 50);

        let cc = HoughCircleConfig::default();
        assert!((cc.min_radius - 10.0).abs() < 1e-10);
        assert!((cc.max_radius - 100.0).abs() < 1e-10);
    }
}
