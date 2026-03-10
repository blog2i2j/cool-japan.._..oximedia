#![allow(dead_code)]
//! Keypoint detection and matching for feature-based computer vision tasks.
//!
//! Provides corner detection, descriptor computation, and brute-force matching.

/// Strength classification for detected keypoints.
#[derive(Debug, Clone, Copy)]
pub enum KeypointStrength {
    /// Weak corner response — marginal feature
    Weak,
    /// Moderate corner response
    Moderate,
    /// Strong, reliable corner response
    Strong,
    /// Exceptionally salient point
    VeryStrong,
}

impl KeypointStrength {
    /// Map strength to a numeric score in 0.0–1.0.
    #[allow(clippy::cast_precision_loss)]
    pub fn numeric_value(self) -> f32 {
        match self {
            Self::Weak => 0.25,
            Self::Moderate => 0.50,
            Self::Strong => 0.75,
            Self::VeryStrong => 1.00,
        }
    }

    /// Classify a raw corner response value into a strength level.
    pub fn from_response(response: f32) -> Self {
        if response >= 0.75 {
            Self::VeryStrong
        } else if response >= 0.50 {
            Self::Strong
        } else if response >= 0.25 {
            Self::Moderate
        } else {
            Self::Weak
        }
    }
}

/// A single detected keypoint in image space.
#[derive(Debug, Clone)]
pub struct Keypoint {
    /// Pixel x-coordinate
    pub x: f32,
    /// Pixel y-coordinate
    pub y: f32,
    /// Corner response score (higher is better)
    pub response: f32,
    /// Scale at which the keypoint was detected
    pub scale: f32,
    /// Dominant orientation in radians
    pub angle: f32,
    /// Optional descriptor vector (e.g., 128-D SIFT-like)
    pub descriptor: Option<Vec<f32>>,
}

impl Keypoint {
    /// Create a new keypoint without a descriptor.
    pub fn new(x: f32, y: f32, response: f32) -> Self {
        Self {
            x,
            y,
            response,
            scale: 1.0,
            angle: 0.0,
            descriptor: None,
        }
    }

    /// Create a keypoint with full parameters.
    #[allow(clippy::too_many_arguments)]
    pub fn with_params(x: f32, y: f32, response: f32, scale: f32, angle: f32) -> Self {
        Self {
            x,
            y,
            response,
            scale,
            angle,
            descriptor: None,
        }
    }

    /// Attach a descriptor to this keypoint.
    #[must_use]
    pub fn with_descriptor(mut self, desc: Vec<f32>) -> Self {
        self.descriptor = Some(desc);
        self
    }

    /// Returns `true` if the keypoint response meets the strong threshold.
    pub fn is_strong(&self) -> bool {
        KeypointStrength::from_response(self.response) >= KeypointStrength::Strong
    }

    /// Distance from the origin (image corner) — useful for spatial bucketing.
    #[allow(clippy::cast_precision_loss)]
    pub fn distance_from_origin(&self) -> f32 {
        (self.x * self.x + self.y * self.y).sqrt()
    }

    /// Euclidean distance in image space to another keypoint.
    pub fn spatial_distance(&self, other: &Keypoint) -> f32 {
        let dx = self.x - other.x;
        let dy = self.y - other.y;
        (dx * dx + dy * dy).sqrt()
    }
}

impl PartialEq for KeypointStrength {
    fn eq(&self, other: &Self) -> bool {
        self.numeric_value() == other.numeric_value()
    }
}

impl Eq for KeypointStrength {}

impl PartialOrd for KeypointStrength {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for KeypointStrength {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.numeric_value()
            .partial_cmp(&other.numeric_value())
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Configuration for the keypoint detector.
#[derive(Debug, Clone)]
pub struct DetectorConfig {
    /// Minimum response threshold to accept a keypoint
    pub response_threshold: f32,
    /// Non-maximum suppression radius in pixels
    pub nms_radius: f32,
    /// Maximum number of keypoints to return (0 = unlimited)
    pub max_keypoints: usize,
    /// Whether to compute descriptors
    pub compute_descriptors: bool,
    /// Descriptor dimension
    pub descriptor_dim: usize,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            response_threshold: 0.01,
            nms_radius: 4.0,
            max_keypoints: 1000,
            compute_descriptors: true,
            descriptor_dim: 32,
        }
    }
}

/// Keypoint detector implementing a simple Harris-style corner measure.
#[derive(Debug, Default)]
pub struct KeypointDetector {
    config: DetectorConfig,
}

impl KeypointDetector {
    /// Create a detector with default configuration.
    pub fn new() -> Self {
        Self {
            config: DetectorConfig::default(),
        }
    }

    /// Create a detector with custom configuration.
    pub fn with_config(config: DetectorConfig) -> Self {
        Self { config }
    }

    /// Detect corners in a flat grayscale pixel buffer (`width × height`).
    ///
    /// Uses a simplified Harris response approximation.
    #[allow(clippy::cast_precision_loss)]
    pub fn detect_corners(&self, pixels: &[f32], width: usize, height: usize) -> Vec<Keypoint> {
        let mut keypoints = Vec::new();

        if width < 3 || height < 3 || pixels.len() != width * height {
            return keypoints;
        }

        // Compute gradient images first, then sum structure tensor over 3×3 windows
        let npix = width * height;
        let mut grad_x = vec![0.0f32; npix];
        let mut grad_y = vec![0.0f32; npix];
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;
                grad_x[idx] = pixels[idx + 1] - pixels[idx - 1];
                grad_y[idx] = pixels[idx + width] - pixels[idx - width];
            }
        }

        for y in 2..height.saturating_sub(2) {
            for x in 2..width.saturating_sub(2) {
                // Accumulate structure tensor M = [[sum_ix2, sum_ixy],[sum_ixy, sum_iy2]]
                // over a 3×3 neighbourhood
                let mut sum_ix2 = 0.0f32;
                let mut sum_iy2 = 0.0f32;
                let mut sum_ixy = 0.0f32;
                for wy in y.wrapping_sub(1)..=y + 1 {
                    for wx in x.wrapping_sub(1)..=x + 1 {
                        let gi = wy * width + wx;
                        sum_ix2 += grad_x[gi] * grad_x[gi];
                        sum_iy2 += grad_y[gi] * grad_y[gi];
                        sum_ixy += grad_x[gi] * grad_y[gi];
                    }
                }

                // Harris response: det(M) - k * trace(M)^2, k = 0.04
                let det = sum_ix2 * sum_iy2 - sum_ixy * sum_ixy;
                let trace = sum_ix2 + sum_iy2;
                let response = det - 0.04 * trace * trace;

                if response > self.config.response_threshold {
                    let mut kp = Keypoint::new(x as f32, y as f32, response);
                    if self.config.compute_descriptors {
                        kp = kp.with_descriptor(
                            self.compute_simple_descriptor(pixels, width, height, x, y),
                        );
                    }
                    keypoints.push(kp);
                }
            }
        }

        // Non-maximum suppression
        let keypoints = self.non_max_suppression(keypoints);

        // Sort by response descending and cap
        let mut keypoints = keypoints;
        keypoints.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        if self.config.max_keypoints > 0 {
            keypoints.truncate(self.config.max_keypoints);
        }
        keypoints
    }

    /// Compute a simple patch-based descriptor.
    #[allow(clippy::cast_precision_loss)]
    fn compute_simple_descriptor(
        &self,
        pixels: &[f32],
        width: usize,
        height: usize,
        cx: usize,
        cy: usize,
    ) -> Vec<f32> {
        let dim = self.config.descriptor_dim;
        let mut desc = vec![0.0f32; dim];
        let radius = 3usize;

        let mut idx = 0;
        'outer: for dy in 0..=radius * 2 {
            for dx in 0..=radius * 2 {
                let px = cx as isize + dx as isize - radius as isize;
                let py = cy as isize + dy as isize - radius as isize;
                if px >= 0 && py >= 0 && (px as usize) < width && (py as usize) < height {
                    if idx < dim {
                        desc[idx] = pixels[py as usize * width + px as usize];
                        idx += 1;
                    } else {
                        break 'outer;
                    }
                }
            }
        }
        desc
    }

    /// Remove keypoints that are dominated by a stronger neighbour within `nms_radius`.
    fn non_max_suppression(&self, mut kps: Vec<Keypoint>) -> Vec<Keypoint> {
        kps.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut kept: Vec<Keypoint> = Vec::new();
        for kp in kps {
            let dominated = kept
                .iter()
                .any(|k| kp.spatial_distance(k) < self.config.nms_radius);
            if !dominated {
                kept.push(kp);
            }
        }
        kept
    }
}

/// A matched pair of keypoints between two images.
#[derive(Debug, Clone)]
pub struct KeypointMatch {
    /// Index into the query keypoint list
    pub query_idx: usize,
    /// Index into the train keypoint list
    pub train_idx: usize,
    /// Descriptor distance (lower is better)
    pub distance: f32,
}

impl KeypointMatch {
    /// Create a new match record.
    pub fn new(query_idx: usize, train_idx: usize, distance: f32) -> Self {
        Self {
            query_idx,
            train_idx,
            distance,
        }
    }

    /// Returns `true` if the match distance is below the given threshold.
    pub fn is_good(&self, threshold: f32) -> bool {
        self.distance < threshold
    }
}

/// Brute-force descriptor matcher.
#[derive(Debug, Default)]
pub struct KeypointMatcher {
    /// Maximum distance ratio for Lowe's ratio test (0 = disabled)
    ratio_threshold: f32,
}

impl KeypointMatcher {
    /// Create a matcher without ratio test.
    pub fn new() -> Self {
        Self {
            ratio_threshold: 0.0,
        }
    }

    /// Create a matcher with Lowe's ratio test threshold (e.g., 0.75).
    pub fn with_ratio_test(ratio: f32) -> Self {
        Self {
            ratio_threshold: ratio,
        }
    }

    /// Match descriptors from `query` keypoints against `train` keypoints.
    ///
    /// Returns matches sorted by distance ascending.
    pub fn match_descriptors(&self, query: &[Keypoint], train: &[Keypoint]) -> Vec<KeypointMatch> {
        let mut matches = Vec::new();

        for (qi, q) in query.iter().enumerate() {
            let Some(q_desc) = &q.descriptor else {
                continue;
            };

            let mut best_dist = f32::MAX;
            let mut second_dist = f32::MAX;
            let mut best_ti = 0usize;

            for (ti, t) in train.iter().enumerate() {
                let Some(t_desc) = &t.descriptor else {
                    continue;
                };
                let dist = l2_distance(q_desc, t_desc);
                if dist < best_dist {
                    second_dist = best_dist;
                    best_dist = dist;
                    best_ti = ti;
                } else if dist < second_dist {
                    second_dist = dist;
                }
            }

            #[allow(clippy::float_cmp)]
            if best_dist == f32::MAX {
                continue;
            }

            // Lowe's ratio test
            if self.ratio_threshold > 0.0
                && second_dist < f32::MAX
                && best_dist / second_dist > self.ratio_threshold
            {
                continue;
            }

            matches.push(KeypointMatch::new(qi, best_ti, best_dist));
        }

        matches.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        matches
    }

    /// Filter matches using Lowe's ratio test after matching.
    pub fn filter_by_ratio(&self, matches: &[KeypointMatch], ratio: f32) -> Vec<KeypointMatch> {
        matches
            .iter()
            .filter(|m| m.distance < ratio)
            .cloned()
            .collect()
    }
}

/// Compute the L2 (Euclidean) distance between two descriptor vectors.
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_keypoint_strength_numeric() {
        assert!((KeypointStrength::Weak.numeric_value() - 0.25).abs() < 1e-6);
        assert!((KeypointStrength::Moderate.numeric_value() - 0.50).abs() < 1e-6);
        assert!((KeypointStrength::Strong.numeric_value() - 0.75).abs() < 1e-6);
        assert!((KeypointStrength::VeryStrong.numeric_value() - 1.00).abs() < 1e-6);
    }

    #[test]
    fn test_keypoint_strength_from_response() {
        assert_eq!(
            KeypointStrength::from_response(0.80),
            KeypointStrength::VeryStrong
        );
        assert_eq!(
            KeypointStrength::from_response(0.60),
            KeypointStrength::Strong
        );
        assert_eq!(
            KeypointStrength::from_response(0.30),
            KeypointStrength::Moderate
        );
        assert_eq!(
            KeypointStrength::from_response(0.10),
            KeypointStrength::Weak
        );
    }

    #[test]
    fn test_keypoint_strength_ordering() {
        assert!(KeypointStrength::Strong > KeypointStrength::Weak);
        assert!(KeypointStrength::VeryStrong > KeypointStrength::Strong);
        assert!(KeypointStrength::Moderate < KeypointStrength::VeryStrong);
    }

    #[test]
    fn test_keypoint_new() {
        let kp = Keypoint::new(10.0, 20.0, 0.8);
        assert!((kp.x - 10.0).abs() < 1e-6);
        assert!((kp.y - 20.0).abs() < 1e-6);
        assert!((kp.response - 0.8).abs() < 1e-6);
        assert!(kp.descriptor.is_none());
    }

    #[test]
    fn test_keypoint_is_strong() {
        let strong = Keypoint::new(0.0, 0.0, 0.9);
        let weak = Keypoint::new(0.0, 0.0, 0.1);
        assert!(strong.is_strong());
        assert!(!weak.is_strong());
    }

    #[test]
    fn test_keypoint_with_descriptor() {
        let kp = Keypoint::new(5.0, 5.0, 0.5).with_descriptor(vec![1.0, 2.0, 3.0]);
        assert!(kp.descriptor.is_some());
        assert_eq!(
            kp.descriptor.as_ref().expect("as_ref should succeed").len(),
            3
        );
    }

    #[test]
    fn test_keypoint_spatial_distance() {
        let a = Keypoint::new(0.0, 0.0, 0.5);
        let b = Keypoint::new(3.0, 4.0, 0.5);
        let dist = a.spatial_distance(&b);
        assert!((dist - 5.0).abs() < 1e-5);
    }

    #[test]
    fn test_detector_empty_image() {
        let det = KeypointDetector::new();
        let kps = det.detect_corners(&[], 0, 0);
        assert!(kps.is_empty());
    }

    #[test]
    fn test_detector_small_image() {
        let det = KeypointDetector::new();
        let pixels = vec![0.0f32; 2 * 2];
        let kps = det.detect_corners(&pixels, 2, 2);
        assert!(kps.is_empty()); // too small for 3×3 window
    }

    #[test]
    fn test_detector_flat_image_no_corners() {
        let det = KeypointDetector::new();
        let pixels = vec![0.5f32; 10 * 10];
        let kps = det.detect_corners(&pixels, 10, 10);
        // Flat image has zero gradient — no corners expected
        assert!(kps.is_empty());
    }

    #[test]
    fn test_detector_checkerboard_finds_corners() {
        let det = KeypointDetector::with_config(DetectorConfig {
            response_threshold: 0.0001,
            nms_radius: 2.0,
            max_keypoints: 100,
            compute_descriptors: false,
            descriptor_dim: 32,
        });
        let w = 8usize;
        let h = 8usize;
        // Create a 16x16 image with a bright rectangle in the centre.
        // The four corners of the rectangle produce Harris responses because
        // gradients in x and y are independently non-zero there.
        let w = 16usize;
        let h = 16usize;
        let pixels: Vec<f32> = (0..w * h)
            .map(|i| {
                let x = i % w;
                let y = i / w;
                if x >= 4 && x < 12 && y >= 4 && y < 12 {
                    1.0
                } else {
                    0.0
                }
            })
            .collect();
        let kps = det.detect_corners(&pixels, w, h);
        // Should find at least some corners in a checkerboard
        assert!(!kps.is_empty());
    }

    #[test]
    fn test_match_descriptors_basic() {
        let desc_a = vec![1.0f32, 0.0, 0.0];
        let desc_b = vec![0.0f32, 1.0, 0.0];
        let kp_q = vec![Keypoint::new(0.0, 0.0, 0.5).with_descriptor(desc_a.clone())];
        let kp_t = vec![
            Keypoint::new(1.0, 1.0, 0.5).with_descriptor(desc_a),
            Keypoint::new(2.0, 2.0, 0.5).with_descriptor(desc_b),
        ];
        let matcher = KeypointMatcher::new();
        let matches = matcher.match_descriptors(&kp_q, &kp_t);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].train_idx, 0); // closer descriptor is index 0
    }

    #[test]
    fn test_keypoint_match_is_good() {
        let m = KeypointMatch::new(0, 0, 0.1);
        assert!(m.is_good(0.5));
        assert!(!m.is_good(0.05));
    }

    #[test]
    fn test_l2_distance_zero() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![1.0f32, 2.0, 3.0];
        assert!(l2_distance(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_l2_distance_known() {
        let a = vec![0.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert!((l2_distance(&a, &b) - 1.0).abs() < 1e-6);
    }
}
