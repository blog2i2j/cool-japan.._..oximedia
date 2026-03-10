#![allow(dead_code)]
//! Lane detection for video frames.
//!
//! This module provides lane detection on road imagery using a simplified
//! Hough transform approach. It identifies lane points, groups them into
//! lane lines, and classifies left/right boundaries.

use std::f64::consts::PI;

/// A single detected point that lies on a lane marking.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LanePoint {
    /// X coordinate in pixels.
    pub x: f64,
    /// Y coordinate in pixels.
    pub y: f64,
    /// Confidence score in `[0, 1]`.
    pub confidence: f64,
}

impl LanePoint {
    /// Create a new lane point.
    #[must_use]
    pub fn new(x: f64, y: f64, confidence: f64) -> Self {
        Self {
            x,
            y,
            confidence: confidence.clamp(0.0, 1.0),
        }
    }

    /// Euclidean distance to another point.
    #[must_use]
    pub fn distance_to(&self, other: &Self) -> f64 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

/// Side classification for a detected lane line.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LaneSide {
    /// Left lane boundary.
    Left,
    /// Right lane boundary.
    Right,
    /// Could not be classified.
    Unknown,
}

/// A detected lane line represented by its Hough parameters and constituent points.
#[derive(Debug, Clone)]
pub struct LaneLine {
    /// Perpendicular distance from the origin (pixels).
    pub rho: f64,
    /// Angle of the normal in radians.
    pub theta: f64,
    /// Accumulator votes that produced this line.
    pub votes: u32,
    /// Points that were associated with this line.
    pub points: Vec<LanePoint>,
    /// Which side of the frame the lane falls on.
    pub side: LaneSide,
}

impl LaneLine {
    /// Create a new lane line from Hough parameters.
    #[must_use]
    pub fn new(rho: f64, theta: f64, votes: u32) -> Self {
        Self {
            rho,
            theta,
            votes,
            points: Vec::new(),
            side: LaneSide::Unknown,
        }
    }

    /// Angle of the line itself (perpendicular to the normal).
    #[must_use]
    pub fn line_angle_deg(&self) -> f64 {
        (self.theta.to_degrees() + 90.0) % 180.0
    }

    /// Evaluate the x coordinate at a given y for this line.
    ///
    /// Returns `None` when `sin(theta)` is zero (horizontal normal).
    #[must_use]
    pub fn x_at_y(&self, y: f64) -> Option<f64> {
        let cos_t = self.theta.cos();
        if cos_t.abs() < 1e-12 {
            return None;
        }
        let sin_t = self.theta.sin();
        Some((self.rho - y * sin_t) / cos_t)
    }

    /// Average confidence of the associated points.
    #[must_use]
    pub fn avg_confidence(&self) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        let sum: f64 = self.points.iter().map(|p| p.confidence).sum();
        sum / self.points.len() as f64
    }
}

/// Configuration for the lane detector.
#[derive(Debug, Clone)]
pub struct LaneDetectorConfig {
    /// Rho resolution in pixels for the accumulator.
    pub rho_resolution: f64,
    /// Theta resolution in radians for the accumulator.
    pub theta_resolution: f64,
    /// Minimum votes to accept a line.
    pub vote_threshold: u32,
    /// Edge intensity threshold.
    pub edge_threshold: u8,
    /// Image width.
    pub width: usize,
    /// Image height.
    pub height: usize,
}

impl Default for LaneDetectorConfig {
    fn default() -> Self {
        Self {
            rho_resolution: 1.0,
            theta_resolution: PI / 180.0,
            vote_threshold: 50,
            edge_threshold: 128,
            width: 640,
            height: 480,
        }
    }
}

/// Performs lane detection on edge-map images using a Hough line transform.
#[derive(Debug)]
pub struct LaneDetector {
    config: LaneDetectorConfig,
}

impl LaneDetector {
    /// Create a new detector with the given configuration.
    #[must_use]
    pub fn new(config: LaneDetectorConfig) -> Self {
        Self { config }
    }

    /// Run the Hough line transform on a grayscale edge map.
    ///
    /// `edge_map` must have length `width * height`. Pixels above `edge_threshold`
    /// are treated as edge pixels.
    ///
    /// Returns detected [`LaneLine`] instances sorted by descending vote count.
    pub fn hough_lines(&self, edge_map: &[u8]) -> Vec<LaneLine> {
        let w = self.config.width;
        let h = self.config.height;
        if edge_map.len() != w * h {
            return Vec::new();
        }

        let diag = ((w * w + h * h) as f64).sqrt();
        let max_rho = diag;
        let rho_bins = ((2.0 * max_rho) / self.config.rho_resolution).ceil() as usize + 1;
        let theta_bins = (PI / self.config.theta_resolution).ceil() as usize;
        let mut accumulator = vec![0u32; rho_bins * theta_bins];

        // Pre-compute sin/cos table
        let thetas: Vec<f64> = (0..theta_bins)
            .map(|i| i as f64 * self.config.theta_resolution)
            .collect();
        let cos_table: Vec<f64> = thetas.iter().map(|t| t.cos()).collect();
        let sin_table: Vec<f64> = thetas.iter().map(|t| t.sin()).collect();

        for y in 0..h {
            for x in 0..w {
                if edge_map[y * w + x] < self.config.edge_threshold {
                    continue;
                }
                let xf = x as f64;
                let yf = y as f64;
                for ti in 0..theta_bins {
                    let rho = xf * cos_table[ti] + yf * sin_table[ti];
                    let rho_idx = ((rho + max_rho) / self.config.rho_resolution).round() as usize;
                    if rho_idx < rho_bins {
                        accumulator[rho_idx * theta_bins + ti] += 1;
                    }
                }
            }
        }

        let mut lines: Vec<LaneLine> = Vec::new();
        for ri in 0..rho_bins {
            for ti in 0..theta_bins {
                let votes = accumulator[ri * theta_bins + ti];
                if votes >= self.config.vote_threshold {
                    let rho = ri as f64 * self.config.rho_resolution - max_rho;
                    let theta = thetas[ti];
                    let mut line = LaneLine::new(rho, theta, votes);
                    // Classify side based on angle
                    let angle_deg = line.line_angle_deg();
                    if (100.0..170.0).contains(&angle_deg) {
                        line.side = LaneSide::Left;
                    } else if (10.0..80.0).contains(&angle_deg) {
                        line.side = LaneSide::Right;
                    }
                    lines.push(line);
                }
            }
        }
        lines.sort_by(|a, b| b.votes.cmp(&a.votes));
        lines
    }

    /// Convenience: detect lanes and return only the strongest left and right lines.
    #[must_use]
    pub fn detect_lane_boundaries(&self, edge_map: &[u8]) -> (Option<LaneLine>, Option<LaneLine>) {
        let lines = self.hough_lines(edge_map);
        let left = lines.iter().find(|l| l.side == LaneSide::Left).cloned();
        let right = lines.iter().find(|l| l.side == LaneSide::Right).cloned();
        (left, right)
    }

    /// Non-maximum suppression on the accumulator results.
    ///
    /// Merges lines whose `(rho, theta)` are within the given tolerances.
    #[must_use]
    pub fn nms(lines: &[LaneLine], rho_tol: f64, theta_tol: f64) -> Vec<LaneLine> {
        let mut kept: Vec<LaneLine> = Vec::new();
        for line in lines {
            let dominated = kept.iter().any(|k| {
                (k.rho - line.rho).abs() < rho_tol && (k.theta - line.theta).abs() < theta_tol
            });
            if !dominated {
                kept.push(line.clone());
            }
        }
        kept
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_config(w: usize, h: usize) -> LaneDetectorConfig {
        LaneDetectorConfig {
            width: w,
            height: h,
            vote_threshold: 3,
            edge_threshold: 128,
            ..LaneDetectorConfig::default()
        }
    }

    #[test]
    fn test_lane_point_new() {
        let p = LanePoint::new(10.0, 20.0, 0.9);
        assert!((p.x - 10.0).abs() < f64::EPSILON);
        assert!((p.y - 20.0).abs() < f64::EPSILON);
        assert!((p.confidence - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lane_point_clamp_confidence() {
        let p = LanePoint::new(0.0, 0.0, 1.5);
        assert!((p.confidence - 1.0).abs() < f64::EPSILON);
        let p2 = LanePoint::new(0.0, 0.0, -0.3);
        assert!((p2.confidence - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lane_point_distance() {
        let a = LanePoint::new(0.0, 0.0, 1.0);
        let b = LanePoint::new(3.0, 4.0, 1.0);
        assert!((a.distance_to(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_lane_line_new() {
        let line = LaneLine::new(100.0, 0.5, 42);
        assert!((line.rho - 100.0).abs() < f64::EPSILON);
        assert_eq!(line.votes, 42);
        assert_eq!(line.side, LaneSide::Unknown);
    }

    #[test]
    fn test_lane_line_angle_deg() {
        let line = LaneLine::new(0.0, 0.0, 1);
        let angle = line.line_angle_deg();
        assert!((angle - 90.0).abs() < 1e-9);
    }

    #[test]
    fn test_lane_line_avg_confidence_empty() {
        let line = LaneLine::new(0.0, 0.0, 1);
        assert!((line.avg_confidence() - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_lane_line_avg_confidence_with_points() {
        let mut line = LaneLine::new(0.0, 0.0, 1);
        line.points.push(LanePoint::new(0.0, 0.0, 0.8));
        line.points.push(LanePoint::new(1.0, 1.0, 0.6));
        assert!((line.avg_confidence() - 0.7).abs() < 1e-9);
    }

    #[test]
    fn test_hough_lines_empty_map() {
        let cfg = make_config(10, 10);
        let det = LaneDetector::new(cfg);
        let map = vec![0u8; 100];
        let lines = det.hough_lines(&map);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_hough_lines_wrong_size() {
        let cfg = make_config(10, 10);
        let det = LaneDetector::new(cfg);
        let map = vec![255u8; 50]; // wrong size
        let lines = det.hough_lines(&map);
        assert!(lines.is_empty());
    }

    #[test]
    fn test_hough_lines_horizontal_edge() {
        // A row of bright pixels should produce at least one line.
        let w = 20;
        let h = 20;
        let cfg = LaneDetectorConfig {
            width: w,
            height: h,
            vote_threshold: 5,
            edge_threshold: 128,
            ..LaneDetectorConfig::default()
        };
        let det = LaneDetector::new(cfg);
        let mut map = vec![0u8; w * h];
        for x in 0..w {
            map[10 * w + x] = 255;
        }
        let lines = det.hough_lines(&map);
        assert!(!lines.is_empty());
    }

    #[test]
    fn test_nms_merges_similar() {
        let lines = vec![
            LaneLine::new(100.0, 1.0, 50),
            LaneLine::new(100.5, 1.01, 30),
            LaneLine::new(200.0, 0.5, 20),
        ];
        let kept = LaneDetector::nms(&lines, 2.0, 0.05);
        assert_eq!(kept.len(), 2);
    }

    #[test]
    fn test_detect_lane_boundaries_returns_none_on_blank() {
        let cfg = make_config(10, 10);
        let det = LaneDetector::new(cfg);
        let map = vec![0u8; 100];
        let (left, right) = det.detect_lane_boundaries(&map);
        assert!(left.is_none());
        assert!(right.is_none());
    }

    #[test]
    fn test_lane_line_x_at_y_vertical_normal() {
        // theta = 0 => normal along x => x = rho for all y.
        let line = LaneLine::new(50.0, 0.0, 1);
        // sin(0) = 0, so x_at_y returns None because cos is in denominator... wait:
        // x = (rho - y*sin) / cos = (50 - 0) / 1 = 50
        let val = line.x_at_y(100.0);
        assert!(val.is_some());
        assert!((val.expect("value should be valid") - 50.0).abs() < 1e-9);
    }

    #[test]
    fn test_lane_line_x_at_y_horizontal_normal() {
        // theta = PI/2 => cos = 0 => formula degenerates
        let line = LaneLine::new(50.0, PI / 2.0, 1);
        let val = line.x_at_y(10.0);
        // cos(pi/2) ~ 0 => None
        assert!(val.is_none());
    }

    #[test]
    fn test_default_config() {
        let cfg = LaneDetectorConfig::default();
        assert_eq!(cfg.width, 640);
        assert_eq!(cfg.height, 480);
        assert_eq!(cfg.vote_threshold, 50);
    }

    #[test]
    fn test_lane_side_equality() {
        assert_eq!(LaneSide::Left, LaneSide::Left);
        assert_ne!(LaneSide::Left, LaneSide::Right);
    }
}
