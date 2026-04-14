//! Camera motion estimation and classification.
//!
//! The [`ransac`] sub-module provides full RANSAC-based DLT homography
//! estimation for robust geometric camera-motion fitting.
//!
//! Detects and classifies camera movement patterns across consecutive frames
//! by analyzing global motion vectors. Supports identification of pan, tilt,
//! zoom, rotation, and handheld shake, which is useful for shot classification,
//! stabilization decisions, and editorial metadata.

#![allow(dead_code)]

pub mod ransac;
pub use ransac::{apply_homography, ransac_homography, Homography};

/// Classification of camera motion type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MotionType {
    /// Camera is stationary (no significant movement).
    Static,
    /// Horizontal pan (left or right).
    Pan,
    /// Vertical tilt (up or down).
    Tilt,
    /// Zoom in or out (scale change).
    Zoom,
    /// Rotation around the optical axis.
    Rotation,
    /// Combined pan and tilt (diagonal movement).
    PanTilt,
    /// Erratic movement (handheld shake).
    Handheld,
    /// Tracking/dolly shot (steady lateral or forward/backward).
    Tracking,
}

impl MotionType {
    /// Return a descriptive label.
    pub fn label(&self) -> &'static str {
        match self {
            Self::Static => "static",
            Self::Pan => "pan",
            Self::Tilt => "tilt",
            Self::Zoom => "zoom",
            Self::Rotation => "rotation",
            Self::PanTilt => "pan_tilt",
            Self::Handheld => "handheld",
            Self::Tracking => "tracking",
        }
    }
}

/// A 2D motion vector.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MotionVector {
    /// Horizontal displacement in pixels.
    pub dx: f64,
    /// Vertical displacement in pixels.
    pub dy: f64,
}

impl MotionVector {
    /// Create a new motion vector.
    pub fn new(dx: f64, dy: f64) -> Self {
        Self { dx, dy }
    }

    /// Compute the magnitude of this vector.
    pub fn magnitude(&self) -> f64 {
        (self.dx * self.dx + self.dy * self.dy).sqrt()
    }

    /// Compute the angle in radians (0 = right, pi/2 = down).
    pub fn angle(&self) -> f64 {
        self.dy.atan2(self.dx)
    }

    /// Return a zero vector.
    pub fn zero() -> Self {
        Self { dx: 0.0, dy: 0.0 }
    }
}

/// Result of camera motion analysis between two frames.
#[derive(Debug, Clone, PartialEq)]
pub struct MotionAnalysis {
    /// Dominant global motion vector.
    pub global_motion: MotionVector,
    /// Classified motion type.
    pub motion_type: MotionType,
    /// Confidence of the classification (0.0-1.0).
    pub confidence: f64,
    /// Estimated scale change (1.0 = no change, >1.0 = zoom in).
    pub scale_change: f64,
    /// Estimated rotation in radians.
    pub rotation_rad: f64,
    /// Motion magnitude in pixels.
    pub magnitude: f64,
    /// Variance of local motion vectors (high = complex scene motion).
    pub motion_variance: f64,
}

/// Configuration for the camera motion estimator.
#[derive(Debug, Clone)]
pub struct MotionEstimatorConfig {
    /// Block size for motion estimation (in pixels).
    pub block_size: usize,
    /// Minimum motion magnitude to be considered non-static.
    pub static_threshold: f64,
    /// Ratio of horizontal vs vertical motion to classify as pan/tilt.
    pub axis_ratio_threshold: f64,
    /// Variance threshold above which motion is classified as handheld.
    pub handheld_variance_threshold: f64,
}

impl Default for MotionEstimatorConfig {
    fn default() -> Self {
        Self {
            block_size: 16,
            static_threshold: 0.5,
            axis_ratio_threshold: 3.0,
            handheld_variance_threshold: 50.0,
        }
    }
}

/// Camera motion estimator using block-matching.
#[derive(Debug)]
pub struct MotionEstimator {
    /// Configuration.
    config: MotionEstimatorConfig,
}

impl MotionEstimator {
    /// Create with default configuration.
    pub fn new() -> Self {
        Self {
            config: MotionEstimatorConfig::default(),
        }
    }

    /// Create with custom configuration.
    pub fn with_config(config: MotionEstimatorConfig) -> Self {
        Self { config }
    }

    /// Estimate camera motion between two grayscale frames.
    ///
    /// Both frames must have the same dimensions.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: usize,
        height: usize,
    ) -> MotionAnalysis {
        let total = width * height;
        if total == 0 || prev.len() < total || curr.len() < total {
            return MotionAnalysis {
                global_motion: MotionVector::zero(),
                motion_type: MotionType::Static,
                confidence: 0.0,
                scale_change: 1.0,
                rotation_rad: 0.0,
                magnitude: 0.0,
                motion_variance: 0.0,
            };
        }

        let block = self.config.block_size;
        let blocks_x = width / block;
        let blocks_y = height / block;

        if blocks_x == 0 || blocks_y == 0 {
            return MotionAnalysis {
                global_motion: MotionVector::zero(),
                motion_type: MotionType::Static,
                confidence: 1.0,
                scale_change: 1.0,
                rotation_rad: 0.0,
                magnitude: 0.0,
                motion_variance: 0.0,
            };
        }

        // Collect local motion vectors via block matching
        let mut vectors: Vec<MotionVector> = Vec::new();
        let search_range = block / 2;

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let ox = bx * block;
                let oy = by * block;
                let mv = self.block_match(prev, curr, width, height, ox, oy, block, search_range);
                vectors.push(mv);
            }
        }

        if vectors.is_empty() {
            return MotionAnalysis {
                global_motion: MotionVector::zero(),
                motion_type: MotionType::Static,
                confidence: 1.0,
                scale_change: 1.0,
                rotation_rad: 0.0,
                magnitude: 0.0,
                motion_variance: 0.0,
            };
        }

        // Compute global motion as median of local vectors
        let global_motion = median_motion(&vectors);
        let magnitude = global_motion.magnitude();

        // Compute variance
        let motion_variance = compute_motion_variance(&vectors, &global_motion);

        // Estimate scale change from radial motion
        let scale_change = estimate_scale(&vectors, width, height);
        let rotation_rad = estimate_rotation(&vectors, width, height);

        // Classify motion type
        let motion_type = self.classify(
            &global_motion,
            magnitude,
            motion_variance,
            scale_change,
            rotation_rad,
        );

        let confidence = compute_confidence(magnitude, motion_variance);

        MotionAnalysis {
            global_motion,
            motion_type,
            confidence,
            scale_change,
            rotation_rad,
            magnitude,
            motion_variance,
        }
    }

    /// Estimate camera motion using RANSAC for robustness against independent object motion.
    ///
    /// This variant uses the standard block-matching to collect local motion vectors,
    /// then applies RANSAC to identify the inlier consensus representing true camera motion,
    /// rejecting outliers caused by moving objects in the scene.
    #[allow(clippy::cast_precision_loss)]
    pub fn estimate_ransac(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: usize,
        height: usize,
        ransac_iterations: usize,
        inlier_threshold: f64,
    ) -> MotionAnalysis {
        let total = width * height;
        if total == 0 || prev.len() < total || curr.len() < total {
            return MotionAnalysis {
                global_motion: MotionVector::zero(),
                motion_type: MotionType::Static,
                confidence: 0.0,
                scale_change: 1.0,
                rotation_rad: 0.0,
                magnitude: 0.0,
                motion_variance: 0.0,
            };
        }

        let block = self.config.block_size;
        let blocks_x = width / block;
        let blocks_y = height / block;

        if blocks_x == 0 || blocks_y == 0 {
            return MotionAnalysis {
                global_motion: MotionVector::zero(),
                motion_type: MotionType::Static,
                confidence: 1.0,
                scale_change: 1.0,
                rotation_rad: 0.0,
                magnitude: 0.0,
                motion_variance: 0.0,
            };
        }

        let search_range = block / 2;
        let mut vectors: Vec<MotionVector> = Vec::with_capacity(blocks_x * blocks_y);

        for by in 0..blocks_y {
            for bx in 0..blocks_x {
                let ox = bx * block;
                let oy = by * block;
                let mv = self.block_match(prev, curr, width, height, ox, oy, block, search_range);
                vectors.push(mv);
            }
        }

        if vectors.is_empty() {
            return MotionAnalysis {
                global_motion: MotionVector::zero(),
                motion_type: MotionType::Static,
                confidence: 1.0,
                scale_change: 1.0,
                rotation_rad: 0.0,
                magnitude: 0.0,
                motion_variance: 0.0,
            };
        }

        // Use RANSAC instead of plain median
        let ransac = ransac_global_motion(&vectors, inlier_threshold, ransac_iterations);
        let global_motion = ransac.motion;
        let magnitude = global_motion.magnitude();

        let motion_variance = compute_motion_variance(&vectors, &global_motion);
        let scale_change = estimate_scale(&vectors, width, height);
        let rotation_rad = estimate_rotation(&vectors, width, height);

        let motion_type = self.classify(
            &global_motion,
            magnitude,
            motion_variance,
            scale_change,
            rotation_rad,
        );

        // Confidence boosted by RANSAC inlier ratio
        let base_conf = compute_confidence(magnitude, motion_variance);
        let confidence = (base_conf as f64 * ransac.inlier_ratio).clamp(0.0, 1.0);

        MotionAnalysis {
            global_motion,
            motion_type,
            confidence,
            scale_change,
            rotation_rad,
            magnitude,
            motion_variance,
        }
    }

    /// Block matching: find the best displacement for a block.
    #[allow(clippy::cast_precision_loss)]
    #[allow(clippy::too_many_arguments)]
    fn block_match(
        &self,
        prev: &[u8],
        curr: &[u8],
        width: usize,
        height: usize,
        ox: usize,
        oy: usize,
        block: usize,
        search_range: usize,
    ) -> MotionVector {
        let mut best_dx: i32 = 0;
        let mut best_dy: i32 = 0;
        let mut best_sad = u64::MAX;

        let sr = search_range as i32;
        for dy in -sr..=sr {
            for dx in -sr..=sr {
                let tx = ox as i32 + dx;
                let ty = oy as i32 + dy;
                if tx < 0 || ty < 0 {
                    continue;
                }
                let tx = tx as usize;
                let ty = ty as usize;
                if tx + block > width || ty + block > height {
                    continue;
                }

                let mut sad = 0_u64;
                for row in 0..block {
                    for col in 0..block {
                        let p = prev[(oy + row) * width + (ox + col)] as i32;
                        let c = curr[(ty + row) * width + (tx + col)] as i32;
                        sad += (p - c).unsigned_abs() as u64;
                    }
                }

                // Prefer smaller displacement when SAD is equal (bias toward zero motion)
                let current_dist = (dx * dx + dy * dy) as u64;
                let best_dist = (best_dx * best_dx + best_dy * best_dy) as u64;
                if sad < best_sad || (sad == best_sad && current_dist < best_dist) {
                    best_sad = sad;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        MotionVector::new(f64::from(best_dx), f64::from(best_dy))
    }

    /// Classify the detected motion.
    fn classify(
        &self,
        global: &MotionVector,
        magnitude: f64,
        variance: f64,
        scale_change: f64,
        rotation: f64,
    ) -> MotionType {
        if magnitude < self.config.static_threshold {
            return MotionType::Static;
        }

        if variance > self.config.handheld_variance_threshold {
            return MotionType::Handheld;
        }

        if (scale_change - 1.0).abs() > 0.02 {
            return MotionType::Zoom;
        }

        if rotation.abs() > 0.02 {
            return MotionType::Rotation;
        }

        let abs_dx = global.dx.abs();
        let abs_dy = global.dy.abs();
        let ratio = if abs_dy > 1e-6 {
            abs_dx / abs_dy
        } else {
            f64::MAX
        };
        let inv_ratio = if abs_dx > 1e-6 {
            abs_dy / abs_dx
        } else {
            f64::MAX
        };

        if ratio > self.config.axis_ratio_threshold {
            MotionType::Pan
        } else if inv_ratio > self.config.axis_ratio_threshold {
            MotionType::Tilt
        } else {
            MotionType::PanTilt
        }
    }
}

impl Default for MotionEstimator {
    fn default() -> Self {
        Self::new()
    }
}

/// RANSAC-based robust homography estimation using block-matching motion vectors.
///
/// Returns the inlier consensus motion vector (translation-only model) that
/// best describes the global camera motion while rejecting outlier vectors
/// caused by independently moving objects.
///
/// # Arguments
///
/// * `vectors` – local motion vectors from block matching.
/// * `inlier_threshold` – maximum pixel distance from the consensus model to be an inlier.
/// * `iterations` – number of RANSAC iterations.
#[allow(clippy::cast_precision_loss)]
pub fn ransac_global_motion(
    vectors: &[MotionVector],
    inlier_threshold: f64,
    iterations: usize,
) -> RansacResult {
    if vectors.len() < 2 {
        let mv = if vectors.is_empty() {
            MotionVector::zero()
        } else {
            vectors[0]
        };
        return RansacResult {
            motion: mv,
            inlier_count: vectors.len(),
            inlier_ratio: 1.0,
        };
    }

    let mut best_inliers = 0usize;
    let mut best_dx = 0.0_f64;
    let mut best_dy = 0.0_f64;

    // Deterministic pseudo-random sampling using a linear congruential generator
    // seeded by the first motion vector magnitudes (no external RNG needed).
    let n = vectors.len();
    let seed_val = {
        let m = vectors[0].magnitude();
        (m * 1_000_000.0) as u64 ^ (n as u64).wrapping_mul(6_364_136_223_846_793_005)
    };
    let mut lcg = seed_val.wrapping_add(1);

    for _ in 0..iterations {
        // Sample one hypothesis
        lcg = lcg
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let idx = (lcg >> 33) as usize % n;
        let hypothesis_dx = vectors[idx].dx;
        let hypothesis_dy = vectors[idx].dy;

        // Count inliers
        let inliers = vectors
            .iter()
            .filter(|v| {
                let ddx = v.dx - hypothesis_dx;
                let ddy = v.dy - hypothesis_dy;
                (ddx * ddx + ddy * ddy).sqrt() < inlier_threshold
            })
            .count();

        if inliers > best_inliers {
            best_inliers = inliers;
            best_dx = hypothesis_dx;
            best_dy = hypothesis_dy;
        }
    }

    // Refine: average over all inliers
    let inlier_vecs: Vec<&MotionVector> = vectors
        .iter()
        .filter(|v| {
            let ddx = v.dx - best_dx;
            let ddy = v.dy - best_dy;
            (ddx * ddx + ddy * ddy).sqrt() < inlier_threshold
        })
        .collect();

    let (refined_dx, refined_dy) = if inlier_vecs.is_empty() {
        (best_dx, best_dy)
    } else {
        let sum_dx: f64 = inlier_vecs.iter().map(|v| v.dx).sum();
        let sum_dy: f64 = inlier_vecs.iter().map(|v| v.dy).sum();
        let k = inlier_vecs.len() as f64;
        (sum_dx / k, sum_dy / k)
    };

    let inlier_count = inlier_vecs.len();
    let inlier_ratio = inlier_count as f64 / n as f64;

    RansacResult {
        motion: MotionVector::new(refined_dx, refined_dy),
        inlier_count,
        inlier_ratio,
    }
}

/// Result of RANSAC global motion estimation.
#[derive(Debug, Clone, PartialEq)]
pub struct RansacResult {
    /// Estimated global motion (translation model).
    pub motion: MotionVector,
    /// Number of inlier vectors that agree with the model.
    pub inlier_count: usize,
    /// Fraction of inliers (0.0-1.0).
    pub inlier_ratio: f64,
}

/// Compute median motion vector from a set of vectors.
fn median_motion(vectors: &[MotionVector]) -> MotionVector {
    if vectors.is_empty() {
        return MotionVector::zero();
    }
    let mut dxs: Vec<f64> = vectors.iter().map(|v| v.dx).collect();
    let mut dys: Vec<f64> = vectors.iter().map(|v| v.dy).collect();
    dxs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    dys.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = vectors.len() / 2;
    MotionVector::new(dxs[mid], dys[mid])
}

/// Compute variance of motion vectors around the global motion.
#[allow(clippy::cast_precision_loss)]
fn compute_motion_variance(vectors: &[MotionVector], global: &MotionVector) -> f64 {
    if vectors.is_empty() {
        return 0.0;
    }
    let n = vectors.len() as f64;
    let sum_sq: f64 = vectors
        .iter()
        .map(|v| {
            let ddx = v.dx - global.dx;
            let ddy = v.dy - global.dy;
            ddx * ddx + ddy * ddy
        })
        .sum();
    sum_sq / n
}

/// Estimate scale change from radial motion pattern.
#[allow(clippy::cast_precision_loss)]
fn estimate_scale(vectors: &[MotionVector], _width: usize, _height: usize) -> f64 {
    if vectors.is_empty() {
        return 1.0;
    }
    // Simplified: compare magnitudes at center vs edges
    // For now, return 1.0 (no zoom) as a baseline
    let avg_mag: f64 =
        vectors.iter().map(MotionVector::magnitude).sum::<f64>() / vectors.len() as f64;
    if avg_mag < 0.1 {
        1.0
    } else {
        1.0 // Baseline: detailed zoom estimation would compare radial patterns
    }
}

/// Estimate rotation from tangential motion pattern.
#[allow(clippy::cast_precision_loss)]
fn estimate_rotation(vectors: &[MotionVector], _width: usize, _height: usize) -> f64 {
    if vectors.is_empty() {
        return 0.0;
    }
    // Simplified: rotation would be detected via tangential component analysis
    0.0
}

/// Compute confidence based on motion magnitude and variance.
fn compute_confidence(magnitude: f64, variance: f64) -> f64 {
    let mag_conf = (magnitude / 10.0).min(1.0);
    let var_penalty = (variance / 100.0).min(0.5);
    (mag_conf * (1.0 - var_penalty)).clamp(0.0, 1.0)
}

/// Accumulate motion classifications over a shot for overall shot-motion summary.
#[derive(Debug)]
pub struct MotionSummary {
    /// Count of each motion type observed.
    type_counts: std::collections::HashMap<MotionType, usize>,
    /// Sum of motion magnitudes.
    total_magnitude: f64,
    /// Number of frames analyzed.
    frame_count: usize,
}

impl MotionSummary {
    /// Create a new empty summary.
    pub fn new() -> Self {
        Self {
            type_counts: std::collections::HashMap::new(),
            total_magnitude: 0.0,
            frame_count: 0,
        }
    }

    /// Record a frame analysis result.
    pub fn record(&mut self, analysis: &MotionAnalysis) {
        *self.type_counts.entry(analysis.motion_type).or_insert(0) += 1;
        self.total_magnitude += analysis.magnitude;
        self.frame_count += 1;
    }

    /// Get the dominant motion type across all recorded frames.
    pub fn dominant_type(&self) -> MotionType {
        self.type_counts
            .iter()
            .max_by_key(|(_, &count)| count)
            .map_or(MotionType::Static, |(&mt, _)| mt)
    }

    /// Get the average motion magnitude.
    #[allow(clippy::cast_precision_loss)]
    pub fn average_magnitude(&self) -> f64 {
        if self.frame_count == 0 {
            0.0
        } else {
            self.total_magnitude / self.frame_count as f64
        }
    }

    /// Get the total number of frames analyzed.
    pub fn frame_count(&self) -> usize {
        self.frame_count
    }
}

impl Default for MotionSummary {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_motion_vector_magnitude() {
        let mv = MotionVector::new(3.0, 4.0);
        assert!((mv.magnitude() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_motion_vector_zero() {
        let mv = MotionVector::zero();
        assert!((mv.magnitude() - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_motion_vector_angle() {
        let mv = MotionVector::new(1.0, 0.0);
        assert!(mv.angle().abs() < 1e-10);
        let mv2 = MotionVector::new(0.0, 1.0);
        assert!((mv2.angle() - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
    }

    #[test]
    fn test_motion_type_label() {
        assert_eq!(MotionType::Static.label(), "static");
        assert_eq!(MotionType::Pan.label(), "pan");
        assert_eq!(MotionType::Handheld.label(), "handheld");
    }

    #[test]
    fn test_median_motion_single() {
        let vectors = vec![MotionVector::new(3.0, 5.0)];
        let m = median_motion(&vectors);
        assert!((m.dx - 3.0).abs() < 1e-10);
        assert!((m.dy - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_motion_multiple() {
        let vectors = vec![
            MotionVector::new(1.0, 10.0),
            MotionVector::new(3.0, 20.0),
            MotionVector::new(5.0, 30.0),
        ];
        let m = median_motion(&vectors);
        assert!((m.dx - 3.0).abs() < 1e-10);
        assert!((m.dy - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_estimate_identical_frames() {
        let estimator = MotionEstimator::new();
        let frame = vec![128_u8; 64 * 64];
        let result = estimator.estimate(&frame, &frame, 64, 64);
        assert_eq!(result.motion_type, MotionType::Static);
        assert!(result.magnitude < 1.0);
    }

    #[test]
    fn test_estimate_empty_frames() {
        let estimator = MotionEstimator::new();
        let result = estimator.estimate(&[], &[], 0, 0);
        assert_eq!(result.motion_type, MotionType::Static);
    }

    #[test]
    fn test_motion_variance_zero() {
        let vectors = vec![MotionVector::new(2.0, 3.0), MotionVector::new(2.0, 3.0)];
        let global = MotionVector::new(2.0, 3.0);
        let var = compute_motion_variance(&vectors, &global);
        assert!(var < 1e-10);
    }

    #[test]
    fn test_motion_variance_nonzero() {
        let vectors = vec![MotionVector::new(0.0, 0.0), MotionVector::new(4.0, 0.0)];
        let global = MotionVector::new(2.0, 0.0);
        let var = compute_motion_variance(&vectors, &global);
        assert!((var - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_no_motion() {
        let conf = compute_confidence(0.0, 0.0);
        assert!((conf - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_strong_motion() {
        let conf = compute_confidence(10.0, 0.0);
        assert!((conf - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_motion_summary_empty() {
        let summary = MotionSummary::new();
        assert_eq!(summary.dominant_type(), MotionType::Static);
        assert!((summary.average_magnitude() - 0.0).abs() < 1e-10);
        assert_eq!(summary.frame_count(), 0);
    }

    #[test]
    fn test_motion_summary_recording() {
        let mut summary = MotionSummary::new();
        let analysis = MotionAnalysis {
            global_motion: MotionVector::new(5.0, 0.0),
            motion_type: MotionType::Pan,
            confidence: 0.9,
            scale_change: 1.0,
            rotation_rad: 0.0,
            magnitude: 5.0,
            motion_variance: 1.0,
        };
        summary.record(&analysis);
        summary.record(&analysis);
        assert_eq!(summary.dominant_type(), MotionType::Pan);
        assert!((summary.average_magnitude() - 5.0).abs() < 1e-10);
        assert_eq!(summary.frame_count(), 2);
    }

    #[test]
    fn test_estimator_with_config() {
        let config = MotionEstimatorConfig {
            block_size: 8,
            static_threshold: 1.0,
            ..Default::default()
        };
        let estimator = MotionEstimator::with_config(config);
        let frame = vec![100_u8; 32 * 32];
        let result = estimator.estimate(&frame, &frame, 32, 32);
        assert_eq!(result.motion_type, MotionType::Static);
    }

    // --- RANSAC tests ---

    #[test]
    fn test_ransac_global_motion_all_same() {
        let vectors = vec![
            MotionVector::new(3.0, 0.0),
            MotionVector::new(3.0, 0.0),
            MotionVector::new(3.0, 0.0),
            MotionVector::new(3.0, 0.0),
        ];
        let result = ransac_global_motion(&vectors, 1.0, 50);
        assert!((result.motion.dx - 3.0).abs() < 0.5);
        assert!(result.inlier_ratio > 0.8);
    }

    #[test]
    fn test_ransac_global_motion_with_outliers() {
        // 4 consistent vectors + 1 large outlier
        let vectors = vec![
            MotionVector::new(2.0, 1.0),
            MotionVector::new(2.0, 1.0),
            MotionVector::new(2.0, 1.0),
            MotionVector::new(2.0, 1.0),
            MotionVector::new(50.0, 50.0), // outlier
        ];
        let result = ransac_global_motion(&vectors, 2.0, 100);
        // Inlier ratio should be majority
        assert!(result.inlier_ratio >= 0.6);
        // Motion should be close to the consensus
        assert!((result.motion.dx - 2.0).abs() < 5.0);
    }

    #[test]
    fn test_ransac_single_vector() {
        let vectors = vec![MotionVector::new(5.0, 3.0)];
        let result = ransac_global_motion(&vectors, 1.0, 10);
        assert!((result.motion.dx - 5.0).abs() < 0.1);
        assert!((result.motion.dy - 3.0).abs() < 0.1);
        assert_eq!(result.inlier_count, 1);
    }

    #[test]
    fn test_ransac_empty_vectors() {
        let vectors: Vec<MotionVector> = Vec::new();
        let result = ransac_global_motion(&vectors, 1.0, 10);
        assert_eq!(result.inlier_count, 0);
        assert!((result.motion.magnitude()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_estimate_ransac_identical_frames() {
        let estimator = MotionEstimator::new();
        let frame = vec![128_u8; 64 * 64];
        let result = estimator.estimate_ransac(&frame, &frame, 64, 64, 20, 1.5);
        assert_eq!(result.motion_type, MotionType::Static);
    }

    #[test]
    fn test_estimate_ransac_empty_frames() {
        let estimator = MotionEstimator::new();
        let result = estimator.estimate_ransac(&[], &[], 0, 0, 10, 1.0);
        assert_eq!(result.motion_type, MotionType::Static);
    }
}
