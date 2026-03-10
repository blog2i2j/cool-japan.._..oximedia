//! Patent-free feature detection and matching.
//!
//! This module provides implementations of feature detection and description
//! algorithms that are free from patent restrictions:
//!
//! - FAST corner detection
//! - BRIEF binary descriptors
//! - ORB (Oriented FAST and Rotated BRIEF)
//! - Brute-force matching with Hamming distance

use crate::{AlignError, AlignResult, Point2D};
use rayon::prelude::*;
use std::f64::consts::PI;

/// A detected keypoint feature
#[derive(Debug, Clone)]
pub struct Keypoint {
    /// Location of the keypoint
    pub point: Point2D,
    /// Scale (size) of the feature
    pub scale: f32,
    /// Orientation in radians
    pub orientation: f32,
    /// Response (strength) of the feature
    pub response: f32,
}

impl Keypoint {
    /// Create a new keypoint
    #[must_use]
    pub fn new(x: f64, y: f64, scale: f32, orientation: f32, response: f32) -> Self {
        Self {
            point: Point2D::new(x, y),
            scale,
            orientation,
            response,
        }
    }
}

/// Binary descriptor (256 bits = 32 bytes)
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryDescriptor {
    /// Binary descriptor data
    pub data: [u8; 32],
}

impl BinaryDescriptor {
    /// Create a new binary descriptor
    #[must_use]
    pub fn new(data: [u8; 32]) -> Self {
        Self { data }
    }

    /// Compute Hamming distance to another descriptor
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }

    /// Create zero descriptor
    #[must_use]
    pub fn zero() -> Self {
        Self { data: [0; 32] }
    }
}

/// A pair of matched features
#[derive(Debug, Clone)]
pub struct MatchPair {
    /// Index in first image
    pub idx1: usize,
    /// Index in second image
    pub idx2: usize,
    /// Hamming distance
    pub distance: u32,
    /// Point in first image
    pub point1: Point2D,
    /// Point in second image
    pub point2: Point2D,
}

impl MatchPair {
    /// Create a new match pair
    #[must_use]
    pub fn new(idx1: usize, idx2: usize, distance: u32, point1: Point2D, point2: Point2D) -> Self {
        Self {
            idx1,
            idx2,
            distance,
            point1,
            point2,
        }
    }
}

/// FAST corner detector
pub struct FastDetector {
    /// Threshold for corner detection
    pub threshold: u8,
    /// Non-maximum suppression window size
    pub nms_window: usize,
}

impl Default for FastDetector {
    fn default() -> Self {
        Self {
            threshold: 20,
            nms_window: 3,
        }
    }
}

impl FastDetector {
    /// Create a new FAST detector
    #[must_use]
    pub fn new(threshold: u8) -> Self {
        Self {
            threshold,
            nms_window: 3,
        }
    }

    /// Detect FAST corners in a grayscale image
    ///
    /// # Errors
    /// Returns error if image dimensions are invalid
    pub fn detect(&self, image: &[u8], width: usize, height: usize) -> AlignResult<Vec<Keypoint>> {
        if image.len() != width * height {
            return Err(AlignError::InvalidConfig("Image size mismatch".to_string()));
        }

        let mut corners = Vec::new();
        let radius = 3;

        // FAST-9 circle offsets (16 pixels in a circle of radius 3)
        let circle = self.get_circle_offsets(width);

        // Detect corners (skip border)
        for y in radius..height - radius {
            for x in radius..width - radius {
                let idx = y * width + x;
                let center = image[idx];

                if self.is_corner(image, idx, center, &circle) {
                    let response = self.compute_response(image, idx, center, &circle);
                    corners.push(Keypoint::new(x as f64, y as f64, 1.0, 0.0, response));
                }
            }
        }

        // Apply non-maximum suppression
        let corners = self.non_maximum_suppression(&corners, width, height);

        Ok(corners)
    }

    /// Get FAST-9 circle offsets
    fn get_circle_offsets(&self, width: usize) -> Vec<isize> {
        let w = width as isize;
        vec![
            -w * 3,     // 0: top
            -w * 3 + 1, // 1
            -w * 2 + 2, // 2
            -w + 3,     // 3: right
            3,          // 4
            w + 3,      // 5
            w * 2 + 2,  // 6
            w * 3 + 1,  // 7: bottom
            w * 3,      // 8
            w * 3 - 1,  // 9
            w * 2 - 2,  // 10
            w - 3,      // 11: left
            -3,         // 12
            -w - 3,     // 13
            -w * 2 - 2, // 14
            -w * 3 - 1, // 15
        ]
    }

    /// Check if pixel is a corner
    fn is_corner(&self, image: &[u8], center_idx: usize, center_val: u8, circle: &[isize]) -> bool {
        let threshold = i16::from(self.threshold);
        let center = i16::from(center_val);

        // Count consecutive brighter or darker pixels
        let mut brighter = 0;
        let mut darker = 0;
        let mut max_consecutive_brighter = 0;
        let mut max_consecutive_darker = 0;

        for &offset in circle {
            let idx = (center_idx as isize + offset) as usize;
            let val = i16::from(image[idx]);
            let diff = val - center;

            if diff > threshold {
                brighter += 1;
                darker = 0;
                max_consecutive_brighter = max_consecutive_brighter.max(brighter);
            } else if diff < -threshold {
                darker += 1;
                brighter = 0;
                max_consecutive_darker = max_consecutive_darker.max(darker);
            } else {
                brighter = 0;
                darker = 0;
            }
        }

        // Need at least 9 consecutive pixels (FAST-9)
        max_consecutive_brighter >= 9 || max_consecutive_darker >= 9
    }

    /// Compute corner response (strength)
    fn compute_response(
        &self,
        image: &[u8],
        center_idx: usize,
        center_val: u8,
        circle: &[isize],
    ) -> f32 {
        let center = i16::from(center_val);
        let mut sum_abs_diff = 0i16;

        for &offset in circle {
            let idx = (center_idx as isize + offset) as usize;
            let val = i16::from(image[idx]);
            sum_abs_diff += (val - center).abs();
        }

        f32::from(sum_abs_diff)
    }

    /// Non-maximum suppression
    fn non_maximum_suppression(
        &self,
        corners: &[Keypoint],
        _width: usize,
        _height: usize,
    ) -> Vec<Keypoint> {
        let mut suppressed = vec![false; corners.len()];
        let window = self.nms_window;

        for i in 0..corners.len() {
            if suppressed[i] {
                continue;
            }

            let ki = &corners[i];

            for (j, kj) in corners.iter().enumerate().skip(i + 1) {
                if suppressed[j] {
                    continue;
                }

                let dx = (ki.point.x - kj.point.x).abs();
                let dy = (ki.point.y - kj.point.y).abs();

                if dx < window as f64 && dy < window as f64 {
                    // Keep the one with higher response
                    if ki.response > kj.response {
                        suppressed[j] = true;
                    } else {
                        suppressed[i] = true;
                        break;
                    }
                }
            }
        }

        corners
            .iter()
            .enumerate()
            .filter(|(i, _)| !suppressed[*i])
            .map(|(_, k)| k.clone())
            .collect()
    }
}

/// BRIEF descriptor extractor
pub struct BriefDescriptor {
    /// Patch size
    pub patch_size: usize,
    /// Sampling pattern
    pattern: Vec<(isize, isize, isize, isize)>,
}

impl Default for BriefDescriptor {
    fn default() -> Self {
        Self::new(31)
    }
}

impl BriefDescriptor {
    /// Create a new BRIEF descriptor extractor
    #[must_use]
    pub fn new(patch_size: usize) -> Self {
        let pattern = Self::generate_pattern(patch_size);
        Self {
            patch_size,
            pattern,
        }
    }

    /// Generate sampling pattern (deterministic)
    fn generate_pattern(patch_size: usize) -> Vec<(isize, isize, isize, isize)> {
        let mut pattern = Vec::with_capacity(256);
        let half = (patch_size / 2) as isize;

        // Use a deterministic pattern based on a simple PRNG
        let mut seed = 42u32;
        for _ in 0..256 {
            let x1 = (Self::next_random(&mut seed) % patch_size as u32) as isize - half;
            let y1 = (Self::next_random(&mut seed) % patch_size as u32) as isize - half;
            let x2 = (Self::next_random(&mut seed) % patch_size as u32) as isize - half;
            let y2 = (Self::next_random(&mut seed) % patch_size as u32) as isize - half;
            pattern.push((x1, y1, x2, y2));
        }

        pattern
    }

    /// Simple LCG random number generator
    fn next_random(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (*seed / 65536) % 32768
    }

    /// Extract BRIEF descriptor at a keypoint
    ///
    /// # Errors
    /// Returns error if keypoint is too close to image border
    pub fn extract(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        keypoint: &Keypoint,
    ) -> AlignResult<BinaryDescriptor> {
        let x = keypoint.point.x as isize;
        let y = keypoint.point.y as isize;
        let half = (self.patch_size / 2) as isize;

        // Check bounds
        if x < half || y < half || x >= (width as isize - half) || y >= (height as isize - half) {
            return Err(AlignError::FeatureError(
                "Keypoint too close to border".to_string(),
            ));
        }

        let mut descriptor = [0u8; 32];

        for (bit_idx, &(x1, y1, x2, y2)) in self.pattern.iter().enumerate() {
            let px1 = (y + y1) as usize * width + (x + x1) as usize;
            let px2 = (y + y2) as usize * width + (x + x2) as usize;

            if image[px1] < image[px2] {
                let byte_idx = bit_idx / 8;
                let bit_pos = bit_idx % 8;
                descriptor[byte_idx] |= 1 << bit_pos;
            }
        }

        Ok(BinaryDescriptor::new(descriptor))
    }

    /// Extract descriptors for multiple keypoints in parallel
    ///
    /// # Errors
    /// Returns error if any extraction fails
    pub fn extract_batch(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        keypoints: &[Keypoint],
    ) -> AlignResult<Vec<BinaryDescriptor>> {
        keypoints
            .par_iter()
            .map(|kp| self.extract(image, width, height, kp))
            .collect()
    }
}

/// ORB feature detector and descriptor
pub struct OrbDetector {
    /// FAST detector
    fast: FastDetector,
    /// BRIEF descriptor
    brief: BriefDescriptor,
    /// Number of features to retain
    pub max_features: usize,
}

impl Default for OrbDetector {
    fn default() -> Self {
        Self::new(500)
    }
}

impl OrbDetector {
    /// Create a new ORB detector
    #[must_use]
    pub fn new(max_features: usize) -> Self {
        Self {
            fast: FastDetector::default(),
            brief: BriefDescriptor::default(),
            max_features,
        }
    }

    /// Detect and describe ORB features
    ///
    /// # Errors
    /// Returns error if detection or description fails
    pub fn detect_and_compute(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
    ) -> AlignResult<(Vec<Keypoint>, Vec<BinaryDescriptor>)> {
        // Detect FAST corners
        let mut keypoints = self.fast.detect(image, width, height)?;

        // Compute orientation for each keypoint
        for kp in &mut keypoints {
            kp.orientation = self.compute_orientation(image, width, height, kp);
        }

        // Keep top N features by response
        keypoints.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        keypoints.truncate(self.max_features);

        // Extract BRIEF descriptors
        let descriptors = self.brief.extract_batch(image, width, height, &keypoints)?;

        Ok((keypoints, descriptors))
    }

    /// Compute orientation using intensity centroid
    fn compute_orientation(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        keypoint: &Keypoint,
    ) -> f32 {
        let x = keypoint.point.x as isize;
        let y = keypoint.point.y as isize;
        let radius = 15isize;

        let mut m01 = 0i64;
        let mut m10 = 0i64;

        for dy in -radius..=radius {
            for dx in -radius..=radius {
                if dx * dx + dy * dy > radius * radius {
                    continue;
                }

                let px = x + dx;
                let py = y + dy;

                if px >= 0 && py >= 0 && px < width as isize && py < height as isize {
                    let idx = py as usize * width + px as usize;
                    let intensity = i64::from(image[idx]);

                    m01 += dy as i64 * intensity;
                    m10 += dx as i64 * intensity;
                }
            }
        }

        (m01 as f64).atan2(m10 as f64) as f32
    }
}

/// Brute-force feature matcher
pub struct FeatureMatcher {
    /// Maximum Hamming distance for a valid match
    pub max_distance: u32,
    /// Ratio test threshold (Lowe's ratio)
    pub ratio_threshold: f32,
}

impl Default for FeatureMatcher {
    fn default() -> Self {
        Self {
            max_distance: 50,
            ratio_threshold: 0.8,
        }
    }
}

impl FeatureMatcher {
    /// Create a new feature matcher
    #[must_use]
    pub fn new(max_distance: u32, ratio_threshold: f32) -> Self {
        Self {
            max_distance,
            ratio_threshold,
        }
    }

    /// Match features using brute-force with ratio test
    #[must_use]
    pub fn match_features(
        &self,
        keypoints1: &[Keypoint],
        descriptors1: &[BinaryDescriptor],
        keypoints2: &[Keypoint],
        descriptors2: &[BinaryDescriptor],
    ) -> Vec<MatchPair> {
        descriptors1
            .par_iter()
            .enumerate()
            .filter_map(|(i, desc1)| {
                // Find two best matches
                let mut best_dist = u32::MAX;
                let mut second_best_dist = u32::MAX;
                let mut best_idx = 0;

                for (j, desc2) in descriptors2.iter().enumerate() {
                    let dist = desc1.hamming_distance(desc2);

                    if dist < best_dist {
                        second_best_dist = best_dist;
                        best_dist = dist;
                        best_idx = j;
                    } else if dist < second_best_dist {
                        second_best_dist = dist;
                    }
                }

                // Apply distance threshold and ratio test
                if best_dist <= self.max_distance {
                    let ratio = best_dist as f32 / second_best_dist as f32;
                    if ratio < self.ratio_threshold {
                        return Some(MatchPair::new(
                            i,
                            best_idx,
                            best_dist,
                            keypoints1[i].point,
                            keypoints2[best_idx].point,
                        ));
                    }
                }

                None
            })
            .collect()
    }

    /// Filter matches using geometric consistency
    #[must_use]
    pub fn filter_matches_geometric(
        &self,
        matches: &[MatchPair],
        threshold: f64,
    ) -> Vec<MatchPair> {
        if matches.len() < 4 {
            return matches.to_vec();
        }

        // Simple geometric filter: check consistency with median displacement
        let median_dx = Self::median(
            &matches
                .iter()
                .map(|m| m.point2.x - m.point1.x)
                .collect::<Vec<_>>(),
        );
        let median_dy = Self::median(
            &matches
                .iter()
                .map(|m| m.point2.y - m.point1.y)
                .collect::<Vec<_>>(),
        );

        matches
            .iter()
            .filter(|m| {
                let dx = m.point2.x - m.point1.x;
                let dy = m.point2.y - m.point1.y;
                (dx - median_dx).abs() < threshold && (dy - median_dy).abs() < threshold
            })
            .cloned()
            .collect()
    }

    /// Compute median of a slice
    fn median(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let mid = sorted.len() / 2;
        if sorted.len() % 2 == 0 {
            (sorted[mid - 1] + sorted[mid]) / 2.0
        } else {
            sorted[mid]
        }
    }
}

/// Harris corner detector (alternative to FAST)
pub struct HarrisDetector {
    /// Threshold for corner detection
    pub threshold: f32,
    /// Window size for gradient computation
    pub window_size: usize,
}

impl Default for HarrisDetector {
    fn default() -> Self {
        Self {
            threshold: 0.01,
            window_size: 3,
        }
    }
}

impl HarrisDetector {
    /// Create a new Harris detector
    #[must_use]
    pub fn new(threshold: f32) -> Self {
        Self {
            threshold,
            window_size: 3,
        }
    }

    /// Detect Harris corners
    ///
    /// # Errors
    /// Returns error if image dimensions are invalid
    pub fn detect(&self, image: &[u8], width: usize, height: usize) -> AlignResult<Vec<Keypoint>> {
        if image.len() != width * height {
            return Err(AlignError::InvalidConfig("Image size mismatch".to_string()));
        }

        // Compute gradients
        let (grad_x, grad_y) = self.compute_gradients(image, width, height);

        // Compute structure tensor
        let mut corners = Vec::new();
        let k = 0.04; // Harris parameter

        for y in self.window_size..height - self.window_size {
            for x in self.window_size..width - self.window_size {
                let mut ixx = 0.0;
                let mut iyy = 0.0;
                let mut ixy = 0.0;

                // Sum over window
                for dy in 0..self.window_size {
                    for dx in 0..self.window_size {
                        let idx = (y + dy - self.window_size / 2) * width
                            + (x + dx - self.window_size / 2);
                        let gx = grad_x[idx];
                        let gy = grad_y[idx];

                        ixx += gx * gx;
                        iyy += gy * gy;
                        ixy += gx * gy;
                    }
                }

                // Compute Harris response
                let det = ixx * iyy - ixy * ixy;
                let trace = ixx + iyy;
                let response = det - k * trace * trace;

                if response > self.threshold {
                    corners.push(Keypoint::new(x as f64, y as f64, 1.0, 0.0, response));
                }
            }
        }

        Ok(corners)
    }

    /// Compute image gradients using Sobel operator
    fn compute_gradients(&self, image: &[u8], width: usize, height: usize) -> (Vec<f32>, Vec<f32>) {
        let mut grad_x = vec![0.0; width * height];
        let mut grad_y = vec![0.0; width * height];

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let idx = y * width + x;

                // Sobel X
                let gx = -f32::from(image[idx - width - 1])
                    - 2.0 * f32::from(image[idx - 1])
                    - f32::from(image[idx + width - 1])
                    + f32::from(image[idx - width + 1])
                    + 2.0 * f32::from(image[idx + 1])
                    + f32::from(image[idx + width + 1]);

                // Sobel Y
                let gy = -f32::from(image[idx - width - 1])
                    - 2.0 * f32::from(image[idx - width])
                    - f32::from(image[idx - width + 1])
                    + f32::from(image[idx + width - 1])
                    + 2.0 * f32::from(image[idx + width])
                    + f32::from(image[idx + width + 1]);

                grad_x[idx] = gx / 8.0;
                grad_y[idx] = gy / 8.0;
            }
        }

        (grad_x, grad_y)
    }
}

/// Feature pyramid for multi-scale feature detection
pub struct FeaturePyramid {
    /// Number of pyramid levels
    pub num_levels: usize,
    /// Scale factor between levels
    pub scale_factor: f32,
}

impl Default for FeaturePyramid {
    fn default() -> Self {
        Self {
            num_levels: 4,
            scale_factor: 0.5,
        }
    }
}

impl FeaturePyramid {
    /// Create a new feature pyramid
    #[must_use]
    pub fn new(num_levels: usize, scale_factor: f32) -> Self {
        Self {
            num_levels,
            scale_factor,
        }
    }

    /// Build image pyramid
    #[must_use]
    pub fn build_pyramid(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<(Vec<u8>, usize, usize)> {
        let mut pyramid = Vec::new();
        pyramid.push((image.to_vec(), width, height));

        let mut current_image = image.to_vec();
        let mut current_width = width;
        let mut current_height = height;

        for _ in 1..self.num_levels {
            let new_width = (current_width as f32 * self.scale_factor) as usize;
            let new_height = (current_height as f32 * self.scale_factor) as usize;

            if new_width < 16 || new_height < 16 {
                break;
            }

            let downsampled = self.downsample(
                &current_image,
                current_width,
                current_height,
                new_width,
                new_height,
            );
            pyramid.push((downsampled.clone(), new_width, new_height));

            current_image = downsampled;
            current_width = new_width;
            current_height = new_height;
        }

        pyramid
    }

    /// Downsample image
    fn downsample(
        &self,
        image: &[u8],
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
    ) -> Vec<u8> {
        let mut output = vec![0u8; dst_width * dst_height];

        let scale_x = src_width as f32 / dst_width as f32;
        let scale_y = src_height as f32 / dst_height as f32;

        for y in 0..dst_height {
            for x in 0..dst_width {
                let src_x = (x as f32 * scale_x) as usize;
                let src_y = (y as f32 * scale_y) as usize;

                if src_x < src_width && src_y < src_height {
                    output[y * dst_width + x] = image[src_y * src_width + src_x];
                }
            }
        }

        output
    }

    /// Detect features at all pyramid levels
    ///
    /// # Errors
    /// Returns error if detection fails
    pub fn detect_multiscale(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        detector: &FastDetector,
    ) -> AlignResult<Vec<Keypoint>> {
        let pyramid = self.build_pyramid(image, width, height);
        let mut all_keypoints = Vec::new();

        for (level, (img, w, h)) in pyramid.iter().enumerate() {
            let keypoints = detector.detect(img, *w, *h)?;

            // Rescale keypoints to original image coordinates
            let scale = self.scale_factor.powi(level as i32);
            for mut kp in keypoints {
                kp.point.x /= f64::from(scale);
                kp.point.y /= f64::from(scale);
                kp.scale *= scale;
                all_keypoints.push(kp);
            }
        }

        Ok(all_keypoints)
    }
}

/// Adaptive non-maximum suppression for better feature distribution
pub struct AdaptiveNMS {
    /// Radius for NMS
    pub radius: f32,
    /// Number of features to retain
    pub num_features: usize,
}

impl AdaptiveNMS {
    /// Create a new adaptive NMS
    #[must_use]
    pub fn new(radius: f32, num_features: usize) -> Self {
        Self {
            radius,
            num_features,
        }
    }

    /// Apply adaptive NMS to keypoints
    #[must_use]
    pub fn apply(&self, keypoints: &[Keypoint]) -> Vec<Keypoint> {
        if keypoints.len() <= self.num_features {
            return keypoints.to_vec();
        }

        let mut result: Vec<Keypoint> = Vec::new();
        let mut sorted = keypoints.to_vec();
        sorted.sort_by(|a, b| {
            b.response
                .partial_cmp(&a.response)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for candidate in &sorted {
            if result.len() >= self.num_features {
                break;
            }

            let mut too_close = false;
            for kept in &result {
                let dist = candidate.point.distance(&kept.point);
                if dist < f64::from(self.radius) {
                    too_close = true;
                    break;
                }
            }

            if !too_close {
                result.push(candidate.clone());
            }
        }

        result
    }
}

/// Outlier filter using statistical methods
pub struct OutlierFilter {
    /// Distance threshold multiplier
    pub threshold_multiplier: f32,
}

impl Default for OutlierFilter {
    fn default() -> Self {
        Self {
            threshold_multiplier: 2.0,
        }
    }
}

impl OutlierFilter {
    /// Create a new outlier filter
    #[must_use]
    pub fn new(threshold_multiplier: f32) -> Self {
        Self {
            threshold_multiplier,
        }
    }

    /// Filter outlier matches
    #[must_use]
    pub fn filter(&self, matches: &[MatchPair]) -> Vec<MatchPair> {
        if matches.len() < 3 {
            return matches.to_vec();
        }

        // Compute distance statistics
        let distances: Vec<f64> = matches
            .iter()
            .map(|m| m.point1.distance(&m.point2))
            .collect();

        let mean = distances.iter().sum::<f64>() / distances.len() as f64;

        let variance: f64 = distances
            .iter()
            .map(|&d| (d - mean) * (d - mean))
            .sum::<f64>()
            / distances.len() as f64;

        let std_dev = variance.sqrt();
        let threshold = mean + std_dev * f64::from(self.threshold_multiplier);

        matches
            .iter()
            .filter(|m| {
                let dist = m.point1.distance(&m.point2);
                dist <= threshold
            })
            .cloned()
            .collect()
    }
}

/// Cross-check matcher for bidirectional matching
pub struct CrossCheckMatcher {
    /// Base matcher
    base_matcher: FeatureMatcher,
}

impl Default for CrossCheckMatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl CrossCheckMatcher {
    /// Create a new cross-check matcher
    #[must_use]
    pub fn new() -> Self {
        Self {
            base_matcher: FeatureMatcher::default(),
        }
    }

    /// Match with cross-check (mutual nearest neighbors)
    #[must_use]
    pub fn match_with_cross_check(
        &self,
        keypoints1: &[Keypoint],
        descriptors1: &[BinaryDescriptor],
        keypoints2: &[Keypoint],
        descriptors2: &[BinaryDescriptor],
    ) -> Vec<MatchPair> {
        // Forward matches (1 -> 2)
        let forward =
            self.base_matcher
                .match_features(keypoints1, descriptors1, keypoints2, descriptors2);

        // Backward matches (2 -> 1)
        let backward =
            self.base_matcher
                .match_features(keypoints2, descriptors2, keypoints1, descriptors1);

        // Keep only mutual matches
        let mut cross_checked = Vec::new();

        for fwd in &forward {
            for bwd in &backward {
                if fwd.idx1 == bwd.idx2 && fwd.idx2 == bwd.idx1 {
                    cross_checked.push(fwd.clone());
                    break;
                }
            }
        }

        cross_checked
    }
}

/// Feature descriptor with FREAK-like pattern
pub struct FreakDescriptor {
    /// Number of sampling pairs
    pub num_pairs: usize,
    /// Pattern scale
    pub pattern_scale: f32,
}

impl Default for FreakDescriptor {
    fn default() -> Self {
        Self {
            num_pairs: 256,
            pattern_scale: 1.0,
        }
    }
}

impl FreakDescriptor {
    /// Create a new FREAK descriptor
    #[must_use]
    pub fn new(num_pairs: usize, pattern_scale: f32) -> Self {
        Self {
            num_pairs,
            pattern_scale,
        }
    }

    /// Extract descriptor (simplified FREAK-like)
    ///
    /// # Errors
    /// Returns error if extraction fails
    pub fn extract(
        &self,
        image: &[u8],
        width: usize,
        height: usize,
        keypoint: &Keypoint,
    ) -> AlignResult<BinaryDescriptor> {
        let x = keypoint.point.x as isize;
        let y = keypoint.point.y as isize;
        let radius = (20.0 * self.pattern_scale) as isize;

        if x < radius
            || y < radius
            || x >= (width as isize - radius)
            || y >= (height as isize - radius)
        {
            return Err(AlignError::FeatureError(
                "Keypoint too close to border".to_string(),
            ));
        }

        let mut descriptor = [0u8; 32];

        // Simplified retina-inspired sampling pattern
        let mut seed = 123u32;
        for bit_idx in 0..256 {
            let r1 = (Self::lcg(&mut seed) % (radius as u32)) as isize;
            let theta1 = (Self::lcg(&mut seed) as f32 / u32::MAX as f32) * 2.0 * PI as f32;
            let x1 = x + (r1 as f32 * theta1.cos()) as isize;
            let y1 = y + (r1 as f32 * theta1.sin()) as isize;

            let r2 = (Self::lcg(&mut seed) % (radius as u32)) as isize;
            let theta2 = (Self::lcg(&mut seed) as f32 / u32::MAX as f32) * 2.0 * PI as f32;
            let x2 = x + (r2 as f32 * theta2.cos()) as isize;
            let y2 = y + (r2 as f32 * theta2.sin()) as isize;

            if x1 >= 0
                && x1 < width as isize
                && y1 >= 0
                && y1 < height as isize
                && x2 >= 0
                && x2 < width as isize
                && y2 >= 0
                && y2 < height as isize
            {
                let idx1 = y1 as usize * width + x1 as usize;
                let idx2 = y2 as usize * width + x2 as usize;

                if idx1 < image.len() && idx2 < image.len() && image[idx1] < image[idx2] {
                    let byte_idx = bit_idx / 8;
                    let bit_pos = bit_idx % 8;
                    descriptor[byte_idx] |= 1 << bit_pos;
                }
            }
        }

        Ok(BinaryDescriptor::new(descriptor))
    }

    /// Simple linear congruential generator
    fn lcg(seed: &mut u32) -> u32 {
        *seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
        (*seed / 65536) % 32768
    }
}

/// Descriptor variance filter to remove low-quality features
pub struct DescriptorVarianceFilter {
    /// Minimum variance threshold
    pub min_variance: f32,
}

impl Default for DescriptorVarianceFilter {
    fn default() -> Self {
        Self { min_variance: 0.1 }
    }
}

impl DescriptorVarianceFilter {
    /// Create a new variance filter
    #[must_use]
    pub fn new(min_variance: f32) -> Self {
        Self { min_variance }
    }

    /// Filter keypoints by descriptor variance
    #[must_use]
    pub fn filter(
        &self,
        keypoints: &[Keypoint],
        descriptors: &[BinaryDescriptor],
    ) -> (Vec<Keypoint>, Vec<BinaryDescriptor>) {
        let mut filtered_kp = Vec::new();
        let mut filtered_desc = Vec::new();

        for (kp, desc) in keypoints.iter().zip(descriptors.iter()) {
            let variance = self.compute_variance(desc);
            if variance >= self.min_variance {
                filtered_kp.push(kp.clone());
                filtered_desc.push(desc.clone());
            }
        }

        (filtered_kp, filtered_desc)
    }

    /// Compute descriptor variance (ratio of set bits)
    fn compute_variance(&self, descriptor: &BinaryDescriptor) -> f32 {
        let num_set_bits: u32 = descriptor.data.iter().map(|b| b.count_ones()).sum();
        let total_bits = descriptor.data.len() * 8;

        let ratio = num_set_bits as f32 / total_bits as f32;

        // Variance is highest when ratio is close to 0.5
        1.0 - (ratio - 0.5).abs() * 2.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_binary_descriptor_hamming() {
        let desc1 = BinaryDescriptor::new([0xFF; 32]);
        let desc2 = BinaryDescriptor::new([0x00; 32]);
        assert_eq!(desc1.hamming_distance(&desc2), 256);

        let desc3 = BinaryDescriptor::new([0xFF; 32]);
        assert_eq!(desc1.hamming_distance(&desc3), 0);
    }

    #[test]
    fn test_keypoint_creation() {
        let kp = Keypoint::new(10.0, 20.0, 1.5, 0.5, 100.0);
        assert_eq!(kp.point.x, 10.0);
        assert_eq!(kp.point.y, 20.0);
        assert_eq!(kp.scale, 1.5);
    }

    #[test]
    fn test_fast_detector() {
        let detector = FastDetector::new(20);
        let image = vec![128u8; 100 * 100];
        let result = detector.detect(&image, 100, 100);
        assert!(result.is_ok());
    }

    #[test]
    fn test_brief_pattern_generation() {
        let brief = BriefDescriptor::new(31);
        assert_eq!(brief.pattern.len(), 256);
    }

    #[test]
    fn test_feature_matcher() {
        let matcher = FeatureMatcher::default();
        assert_eq!(matcher.max_distance, 50);
        assert!((matcher.ratio_threshold - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn test_median_computation() {
        let values = vec![1.0, 3.0, 2.0, 5.0, 4.0];
        let median = FeatureMatcher::median(&values);
        assert_eq!(median, 3.0);

        let values2 = vec![1.0, 2.0, 3.0, 4.0];
        let median2 = FeatureMatcher::median(&values2);
        assert_eq!(median2, 2.5);
    }

    #[test]
    fn test_feature_pyramid() {
        let pyramid = FeaturePyramid::new(4, 0.5);
        assert_eq!(pyramid.num_levels, 4);
        assert_eq!(pyramid.scale_factor, 0.5);
    }

    #[test]
    fn test_pyramid_building() {
        let pyramid = FeaturePyramid::default();
        let image = vec![128u8; 100 * 100];
        let levels = pyramid.build_pyramid(&image, 100, 100);

        assert!(!levels.is_empty());
        assert_eq!(levels[0].1, 100); // First level width
        assert_eq!(levels[0].2, 100); // First level height
    }

    #[test]
    fn test_adaptive_nms() {
        let nms = AdaptiveNMS::new(10.0, 5);
        let keypoints = vec![
            Keypoint::new(0.0, 0.0, 1.0, 0.0, 100.0),
            Keypoint::new(5.0, 5.0, 1.0, 0.0, 90.0),
            Keypoint::new(50.0, 50.0, 1.0, 0.0, 80.0),
        ];

        let filtered = nms.apply(&keypoints);
        assert!(!filtered.is_empty());
        assert!(filtered.len() <= 5);
    }

    #[test]
    fn test_outlier_filter() {
        let filter = OutlierFilter::default();
        assert_eq!(filter.threshold_multiplier, 2.0);
    }

    #[test]
    fn test_cross_check_matcher() {
        let matcher = CrossCheckMatcher::new();
        let kp1 = vec![Keypoint::new(0.0, 0.0, 1.0, 0.0, 1.0)];
        let kp2 = vec![Keypoint::new(0.0, 0.0, 1.0, 0.0, 1.0)];
        let desc1 = vec![BinaryDescriptor::zero()];
        let desc2 = vec![BinaryDescriptor::zero()];

        let matches = matcher.match_with_cross_check(&kp1, &desc1, &kp2, &desc2);
        assert_eq!(matches.len(), 1);
    }

    #[test]
    fn test_freak_descriptor() {
        let freak = FreakDescriptor::default();
        assert_eq!(freak.num_pairs, 256);
        assert_eq!(freak.pattern_scale, 1.0);
    }

    #[test]
    fn test_descriptor_variance_filter() {
        let filter = DescriptorVarianceFilter::new(0.1);
        assert_eq!(filter.min_variance, 0.1);
    }

    #[test]
    fn test_descriptor_variance() {
        let filter = DescriptorVarianceFilter::default();
        let desc = BinaryDescriptor::new([0xAA; 32]); // 50% set bits
        let variance = filter.compute_variance(&desc);
        assert!((variance - 1.0).abs() < 0.01);
    }
}
