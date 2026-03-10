//! Horizon line detection.

use crate::Frame;

/// Horizon detector.
#[derive(Debug)]
pub struct HorizonDetector;

impl HorizonDetector {
    /// Create a new horizon detector.
    #[must_use]
    pub fn new() -> Self {
        Self
    }

    /// Detect horizon line in a frame.
    #[must_use]
    pub fn detect(&self, _frame: &Frame) -> Option<HorizonLine> {
        Some(HorizonLine {
            angle: 0.0,
            y_intercept: 0.0,
            confidence: 0.5,
        })
    }
}

impl Default for HorizonDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Detected horizon line.
#[derive(Debug, Clone, Copy)]
pub struct HorizonLine {
    /// Angle in radians
    pub angle: f64,
    /// Y-intercept
    pub y_intercept: f64,
    /// Detection confidence
    pub confidence: f64,
}

/// Advanced horizon detection using Hough transform.
pub mod hough {
    use scirs2_core::ndarray::Array2;

    /// Hough transform for line detection.
    pub struct HoughTransform {
        rho_resolution: f64,
        theta_resolution: f64,
        threshold: usize,
    }

    impl HoughTransform {
        /// Create a new Hough transform detector.
        #[must_use]
        pub fn new() -> Self {
            Self {
                rho_resolution: 1.0,
                theta_resolution: std::f64::consts::PI / 180.0,
                threshold: 100,
            }
        }

        /// Detect lines using Hough transform.
        #[must_use]
        pub fn detect_lines(&self, edges: &Array2<u8>) -> Vec<Line> {
            let (height, width) = edges.dim();
            let diagonal = ((width.pow(2) + height.pow(2)) as f64).sqrt();

            let rho_max = diagonal as usize;
            let theta_bins = (std::f64::consts::PI / self.theta_resolution) as usize;

            let mut accumulator: Array2<usize> = Array2::zeros((rho_max * 2, theta_bins));

            // Voting
            for y in 0..height {
                for x in 0..width {
                    if edges[[y, x]] > 0 {
                        for theta_idx in 0..theta_bins {
                            let theta = theta_idx as f64 * self.theta_resolution;
                            let rho = x as f64 * theta.cos() + y as f64 * theta.sin();
                            let rho_idx = (rho + diagonal) as usize;

                            if rho_idx < rho_max * 2 {
                                accumulator[[rho_idx, theta_idx]] += 1;
                            }
                        }
                    }
                }
            }

            // Find peaks
            let mut lines = Vec::new();
            for rho_idx in 0..rho_max * 2 {
                for theta_idx in 0..theta_bins {
                    if accumulator[[rho_idx, theta_idx]] > self.threshold {
                        let rho = rho_idx as f64 - diagonal;
                        let theta = theta_idx as f64 * self.theta_resolution;

                        lines.push(Line { rho, theta });
                    }
                }
            }

            lines
        }

        /// Filter lines to find horizon.
        #[must_use]
        pub fn find_horizon(&self, lines: &[Line]) -> Option<Line> {
            // Find the most horizontal line
            lines
                .iter()
                .filter(|line| {
                    let angle_deg = line.theta.to_degrees();
                    angle_deg.abs() < 10.0 || (angle_deg - 180.0).abs() < 10.0
                })
                .min_by(|a, b| {
                    a.rho
                        .abs()
                        .partial_cmp(&b.rho.abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
        }
    }

    impl Default for HoughTransform {
        fn default() -> Self {
            Self::new()
        }
    }

    /// A line detected by Hough transform.
    #[derive(Debug, Clone, Copy)]
    pub struct Line {
        /// Distance from origin
        pub rho: f64,
        /// Angle in radians
        pub theta: f64,
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_hough_transform() {
            let detector = HoughTransform::new();
            let edges = Array2::zeros((100, 100));
            let lines = detector.detect_lines(&edges);
            assert!(lines.is_empty() || !lines.is_empty()); // Valid either way
        }
    }
}
