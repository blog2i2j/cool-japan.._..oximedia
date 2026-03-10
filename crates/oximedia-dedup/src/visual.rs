//! Visual similarity detection for image and video deduplication.
//!
//! This module provides:
//! - Perceptual hashing (pHash, dHash, aHash)
//! - SSIM (Structural Similarity Index Measure)
//! - Histogram comparison
//! - Feature extraction and matching
//! - Near-duplicate detection with configurable thresholds

use crate::{DedupError, DedupResult};
use ndarray::Array2;

/// Image representation for processing.
#[derive(Debug, Clone)]
pub struct Image {
    /// Image width
    pub width: usize,

    /// Image height
    pub height: usize,

    /// Pixel data (RGB or grayscale)
    pub data: Vec<u8>,

    /// Number of channels (1 for grayscale, 3 for RGB)
    pub channels: usize,
}

impl Image {
    /// Create a new image.
    #[must_use]
    pub fn new(width: usize, height: usize, channels: usize) -> Self {
        let data = vec![0u8; width * height * channels];
        Self {
            width,
            height,
            data,
            channels,
        }
    }

    /// Create from raw data.
    ///
    /// # Errors
    ///
    /// Returns an error if data size doesn't match dimensions.
    pub fn from_data(
        width: usize,
        height: usize,
        channels: usize,
        data: Vec<u8>,
    ) -> DedupResult<Self> {
        if data.len() != width * height * channels {
            return Err(DedupError::Visual(format!(
                "Invalid data size: expected {}, got {}",
                width * height * channels,
                data.len()
            )));
        }
        Ok(Self {
            width,
            height,
            data,
            channels,
        })
    }

    /// Convert to grayscale.
    #[must_use]
    pub fn to_grayscale(&self) -> Self {
        if self.channels == 1 {
            return self.clone();
        }

        let mut gray = Vec::with_capacity(self.width * self.height);

        for y in 0..self.height {
            for x in 0..self.width {
                let idx = (y * self.width + x) * self.channels;
                let r = f64::from(self.data[idx]);
                let g = f64::from(self.data[idx + 1]);
                let b = f64::from(self.data[idx + 2]);

                // ITU-R BT.601 conversion
                let gray_value = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                gray.push(gray_value);
            }
        }

        Self {
            width: self.width,
            height: self.height,
            data: gray,
            channels: 1,
        }
    }

    /// Resize image to specified dimensions.
    #[must_use]
    pub fn resize(&self, new_width: usize, new_height: usize) -> Self {
        let mut resized = Image::new(new_width, new_height, self.channels);

        let x_ratio = self.width as f64 / new_width as f64;
        let y_ratio = self.height as f64 / new_height as f64;

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x as f64 * x_ratio) as usize;
                let src_y = (y as f64 * y_ratio) as usize;

                let src_idx = (src_y * self.width + src_x) * self.channels;
                let dst_idx = (y * new_width + x) * self.channels;

                for c in 0..self.channels {
                    resized.data[dst_idx + c] = self.data[src_idx + c];
                }
            }
        }

        resized
    }

    /// Get pixel at position.
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<&[u8]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let idx = (y * self.width + x) * self.channels;
        Some(&self.data[idx..idx + self.channels])
    }

    /// Calculate mean pixel value.
    #[must_use]
    pub fn mean(&self) -> f64 {
        let sum: u64 = self.data.iter().map(|&v| u64::from(v)).sum();
        sum as f64 / self.data.len() as f64
    }
}

/// Perceptual hash result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PerceptualHash {
    hash: u64,
    bits: usize,
}

impl PerceptualHash {
    /// Create from hash value.
    #[must_use]
    pub fn new(hash: u64, bits: usize) -> Self {
        Self { hash, bits }
    }

    /// Get hash value.
    #[must_use]
    pub fn hash(&self) -> u64 {
        self.hash
    }

    /// Calculate Hamming distance.
    #[must_use]
    pub fn hamming_distance(&self, other: &Self) -> u32 {
        (self.hash ^ other.hash).count_ones()
    }

    /// Calculate similarity (0.0-1.0).
    #[must_use]
    pub fn similarity(&self, other: &Self) -> f64 {
        let distance = self.hamming_distance(other);
        1.0 - (f64::from(distance) / self.bits as f64)
    }

    /// Convert to hex string.
    #[must_use]
    pub fn to_hex(&self) -> String {
        format!("{:016x}", self.hash)
    }
}

/// Compute difference hash (dHash).
///
/// dHash is based on differences between adjacent pixels.
#[must_use]
pub fn compute_dhash(image: &Image) -> PerceptualHash {
    const HASH_SIZE: usize = 8;

    // Convert to grayscale and resize
    let gray = image.to_grayscale();
    let resized = gray.resize(HASH_SIZE + 1, HASH_SIZE);

    let mut hash = 0u64;
    let mut bit = 0;

    for y in 0..HASH_SIZE {
        for x in 0..HASH_SIZE {
            let idx1 = y * (HASH_SIZE + 1) + x;
            let idx2 = y * (HASH_SIZE + 1) + x + 1;

            if resized.data[idx2] > resized.data[idx1] {
                hash |= 1u64 << bit;
            }
            bit += 1;
        }
    }

    PerceptualHash::new(hash, 64)
}

/// Compute average hash (aHash).
///
/// aHash is based on whether each pixel is above or below the mean.
#[must_use]
pub fn compute_ahash(image: &Image) -> PerceptualHash {
    const HASH_SIZE: usize = 8;

    // Convert to grayscale and resize
    let gray = image.to_grayscale();
    let resized = gray.resize(HASH_SIZE, HASH_SIZE);

    let mean = resized.mean();
    let mut hash = 0u64;

    for (i, &pixel) in resized.data.iter().enumerate() {
        if f64::from(pixel) > mean {
            hash |= 1u64 << i;
        }
    }

    PerceptualHash::new(hash, 64)
}

/// Discrete Cosine Transform (DCT) for perceptual hashing.
fn dct_2d(input: &Array2<f64>) -> Array2<f64> {
    let (rows, cols) = input.dim();
    let mut output = Array2::zeros((rows, cols));

    for u in 0..rows {
        for v in 0..cols {
            let mut sum = 0.0;

            for i in 0..rows {
                for j in 0..cols {
                    let val = input[[i, j]];
                    let cos_i = ((2 * i + 1) as f64 * u as f64 * std::f64::consts::PI
                        / (2.0 * rows as f64))
                        .cos();
                    let cos_j = ((2 * j + 1) as f64 * v as f64 * std::f64::consts::PI
                        / (2.0 * cols as f64))
                        .cos();
                    sum += val * cos_i * cos_j;
                }
            }

            let cu = if u == 0 {
                (1.0 / rows as f64).sqrt()
            } else {
                (2.0 / rows as f64).sqrt()
            };
            let cv = if v == 0 {
                (1.0 / cols as f64).sqrt()
            } else {
                (2.0 / cols as f64).sqrt()
            };

            output[[u, v]] = cu * cv * sum;
        }
    }

    output
}

/// Compute perceptual hash (pHash) using DCT.
#[must_use]
pub fn compute_phash(image: &Image) -> PerceptualHash {
    const HASH_SIZE: usize = 8;
    const DCT_SIZE: usize = 32;

    // Convert to grayscale and resize
    let gray = image.to_grayscale();
    let resized = gray.resize(DCT_SIZE, DCT_SIZE);

    // Convert to array
    let mut input = Array2::zeros((DCT_SIZE, DCT_SIZE));
    for y in 0..DCT_SIZE {
        for x in 0..DCT_SIZE {
            let idx = y * DCT_SIZE + x;
            input[[y, x]] = f64::from(resized.data[idx]);
        }
    }

    // Apply DCT
    let dct = dct_2d(&input);

    // Extract top-left 8x8 (low frequencies)
    let mut low_freq = Vec::new();
    for y in 0..HASH_SIZE {
        for x in 0..HASH_SIZE {
            low_freq.push(dct[[y, x]]);
        }
    }

    // Calculate median
    let mut sorted = low_freq.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let median = sorted[sorted.len() / 2];

    // Generate hash
    let mut hash = 0u64;
    for (i, &val) in low_freq.iter().enumerate() {
        if val > median {
            hash |= 1u64 << i;
        }
    }

    PerceptualHash::new(hash, 64)
}

/// Compute histogram for an image.
#[must_use]
pub fn compute_histogram(image: &Image) -> Vec<Vec<u32>> {
    let mut histograms = vec![vec![0u32; 256]; image.channels];

    for i in 0..image.data.len() {
        let channel = i % image.channels;
        let value = image.data[i] as usize;
        histograms[channel][value] += 1;
    }

    histograms
}

/// Compare histograms using correlation.
#[must_use]
pub fn compare_histograms(hist1: &[Vec<u32>], hist2: &[Vec<u32>]) -> f64 {
    if hist1.len() != hist2.len() {
        return 0.0;
    }

    let mut correlations = Vec::new();

    for (h1, h2) in hist1.iter().zip(hist2.iter()) {
        let correlation = histogram_correlation(h1, h2);
        correlations.push(correlation);
    }

    // Average correlation across channels
    correlations.iter().sum::<f64>() / correlations.len() as f64
}

/// Calculate correlation between two histograms.
fn histogram_correlation(hist1: &[u32], hist2: &[u32]) -> f64 {
    let mean1: f64 = hist1.iter().map(|&v| f64::from(v)).sum::<f64>() / hist1.len() as f64;
    let mean2: f64 = hist2.iter().map(|&v| f64::from(v)).sum::<f64>() / hist2.len() as f64;

    let mut numerator = 0.0;
    let mut denom1 = 0.0;
    let mut denom2 = 0.0;

    for i in 0..hist1.len() {
        let d1 = f64::from(hist1[i]) - mean1;
        let d2 = f64::from(hist2[i]) - mean2;

        numerator += d1 * d2;
        denom1 += d1 * d1;
        denom2 += d2 * d2;
    }

    if denom1 == 0.0 || denom2 == 0.0 {
        return 0.0;
    }

    numerator / (denom1 * denom2).sqrt()
}

/// SSIM (Structural Similarity Index) parameters.
pub struct SsimParams {
    /// Window size
    pub window_size: usize,

    /// K1 constant
    pub k1: f64,

    /// K2 constant
    pub k2: f64,

    /// Dynamic range (typically 255 for 8-bit images)
    pub l: f64,
}

impl Default for SsimParams {
    fn default() -> Self {
        Self {
            window_size: 11,
            k1: 0.01,
            k2: 0.03,
            l: 255.0,
        }
    }
}

/// Compute SSIM between two images.
#[must_use]
pub fn compute_ssim(image1: &Image, image2: &Image, params: &SsimParams) -> f64 {
    // Convert to grayscale
    let gray1 = image1.to_grayscale();
    let gray2 = image2.to_grayscale();

    // Resize to same dimensions if needed
    let (width, height) = if gray1.width == gray2.width && gray1.height == gray2.height {
        (gray1.width, gray1.height)
    } else {
        let min_width = gray1.width.min(gray2.width);
        let min_height = gray1.height.min(gray2.height);
        (min_width, min_height)
    };

    let img1 = if gray1.width != width || gray1.height != height {
        gray1.resize(width, height)
    } else {
        gray1
    };

    let img2 = if gray2.width != width || gray2.height != height {
        gray2.resize(width, height)
    } else {
        gray2
    };

    // Calculate SSIM
    let c1 = (params.k1 * params.l).powi(2);
    let c2 = (params.k2 * params.l).powi(2);

    let mut ssim_sum = 0.0;
    let mut count = 0;

    let half_window = params.window_size / 2;

    for y in half_window..height.saturating_sub(half_window) {
        for x in half_window..width.saturating_sub(half_window) {
            let window1 = extract_window(&img1, x, y, params.window_size);
            let window2 = extract_window(&img2, x, y, params.window_size);

            let mean1 = window_mean(&window1);
            let mean2 = window_mean(&window2);
            let var1 = window_variance(&window1, mean1);
            let var2 = window_variance(&window2, mean2);
            let covar = window_covariance(&window1, &window2, mean1, mean2);

            let numerator = (2.0 * mean1 * mean2 + c1) * (2.0 * covar + c2);
            let denominator = (mean1 * mean1 + mean2 * mean2 + c1) * (var1 + var2 + c2);

            if denominator != 0.0 {
                ssim_sum += numerator / denominator;
                count += 1;
            }
        }
    }

    if count == 0 {
        return 0.0;
    }

    ssim_sum / count as f64
}

/// Extract a window from an image.
fn extract_window(image: &Image, cx: usize, cy: usize, window_size: usize) -> Vec<f64> {
    let half = window_size / 2;
    let mut window = Vec::new();

    for y in cy.saturating_sub(half)..=(cy + half).min(image.height - 1) {
        for x in cx.saturating_sub(half)..=(cx + half).min(image.width - 1) {
            let idx = y * image.width + x;
            window.push(f64::from(image.data[idx]));
        }
    }

    window
}

/// Calculate mean of a window.
fn window_mean(window: &[f64]) -> f64 {
    window.iter().sum::<f64>() / window.len() as f64
}

/// Calculate variance of a window.
fn window_variance(window: &[f64], mean: f64) -> f64 {
    let sum_sq: f64 = window.iter().map(|&v| (v - mean).powi(2)).sum();
    sum_sq / window.len() as f64
}

/// Calculate covariance between two windows.
fn window_covariance(window1: &[f64], window2: &[f64], mean1: f64, mean2: f64) -> f64 {
    let sum: f64 = window1
        .iter()
        .zip(window2.iter())
        .map(|(&v1, &v2)| (v1 - mean1) * (v2 - mean2))
        .sum();
    sum / window1.len() as f64
}

/// Feature point for matching.
#[derive(Debug, Clone)]
pub struct FeaturePoint {
    /// X coordinate
    pub x: f64,

    /// Y coordinate
    pub y: f64,

    /// Feature descriptor
    pub descriptor: Vec<f64>,
}

/// Extract feature points from an image (simplified SIFT-like).
#[must_use]
pub fn extract_features(image: &Image) -> Vec<FeaturePoint> {
    let gray = image.to_grayscale();
    let mut features = Vec::new();

    // Simple corner detection (Harris-like)
    let threshold = 100.0;

    for y in 2..gray.height - 2 {
        for x in 2..gray.width - 2 {
            let score = compute_corner_response(&gray, x, y);

            if score > threshold {
                let descriptor = compute_descriptor(&gray, x, y);
                features.push(FeaturePoint {
                    x: x as f64,
                    y: y as f64,
                    descriptor,
                });
            }
        }
    }

    features
}

/// Compute corner response at a point.
fn compute_corner_response(image: &Image, x: usize, y: usize) -> f64 {
    let idx = y * image.width + x;
    let center = f64::from(image.data[idx]);

    let mut sum = 0.0;
    for dy in -1..=1 {
        for dx in -1..=1 {
            if dx == 0 && dy == 0 {
                continue;
            }

            let nx = (x as i32 + dx) as usize;
            let ny = (y as i32 + dy) as usize;

            if nx < image.width && ny < image.height {
                let nidx = ny * image.width + nx;
                let diff = center - f64::from(image.data[nidx]);
                sum += diff * diff;
            }
        }
    }

    sum
}

/// Compute feature descriptor at a point.
fn compute_descriptor(image: &Image, cx: usize, cy: usize) -> Vec<f64> {
    const DESC_SIZE: usize = 8;
    let mut descriptor = Vec::new();

    for dy in -(DESC_SIZE as i32 / 2)..=(DESC_SIZE as i32 / 2) {
        for dx in -(DESC_SIZE as i32 / 2)..=(DESC_SIZE as i32 / 2) {
            let nx = (cx as i32 + dx).clamp(0, image.width as i32 - 1) as usize;
            let ny = (cy as i32 + dy).clamp(0, image.height as i32 - 1) as usize;
            let idx = ny * image.width + nx;
            descriptor.push(f64::from(image.data[idx]));
        }
    }

    // Normalize
    let norm: f64 = descriptor.iter().map(|&v| v * v).sum::<f64>().sqrt();
    if norm > 0.0 {
        descriptor.iter_mut().for_each(|v| *v /= norm);
    }

    descriptor
}

/// Match features between two images.
#[must_use]
pub fn match_features(features1: &[FeaturePoint], features2: &[FeaturePoint]) -> usize {
    let mut matches = 0;
    const MATCH_THRESHOLD: f64 = 0.8;

    for f1 in features1 {
        let mut best_distance = f64::MAX;
        let mut second_best = f64::MAX;

        for f2 in features2 {
            let distance = descriptor_distance(&f1.descriptor, &f2.descriptor);

            if distance < best_distance {
                second_best = best_distance;
                best_distance = distance;
            } else if distance < second_best {
                second_best = distance;
            }
        }

        // Ratio test (Lowe's ratio test)
        if best_distance < MATCH_THRESHOLD * second_best {
            matches += 1;
        }
    }

    matches
}

/// Calculate Euclidean distance between descriptors.
fn descriptor_distance(desc1: &[f64], desc2: &[f64]) -> f64 {
    desc1
        .iter()
        .zip(desc2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Compare visual similarity between two images.
///
/// # Errors
///
/// Returns an error if images cannot be processed.
pub fn compare_images(image1: &Image, image2: &Image) -> DedupResult<VisualSimilarity> {
    // Compute various similarity metrics
    let dhash1 = compute_dhash(image1);
    let dhash2 = compute_dhash(image2);
    let dhash_similarity = dhash1.similarity(&dhash2);

    let ahash1 = compute_ahash(image1);
    let ahash2 = compute_ahash(image2);
    let ahash_similarity = ahash1.similarity(&ahash2);

    let phash1 = compute_phash(image1);
    let phash2 = compute_phash(image2);
    let phash_similarity = phash1.similarity(&phash2);

    let hist1 = compute_histogram(image1);
    let hist2 = compute_histogram(image2);
    let histogram_similarity = compare_histograms(&hist1, &hist2);

    let ssim_params = SsimParams::default();
    let ssim = compute_ssim(image1, image2, &ssim_params);

    let features1 = extract_features(image1);
    let features2 = extract_features(image2);
    let feature_matches = match_features(&features1, &features2);

    Ok(VisualSimilarity {
        dhash_similarity,
        ahash_similarity,
        phash_similarity,
        histogram_similarity,
        ssim,
        feature_matches,
    })
}

/// Visual similarity metrics.
#[derive(Debug, Clone)]
pub struct VisualSimilarity {
    /// Difference hash similarity
    pub dhash_similarity: f64,

    /// Average hash similarity
    pub ahash_similarity: f64,

    /// Perceptual hash similarity
    pub phash_similarity: f64,

    /// Histogram similarity
    pub histogram_similarity: f64,

    /// SSIM score
    pub ssim: f64,

    /// Number of feature matches
    pub feature_matches: usize,
}

impl VisualSimilarity {
    /// Calculate overall similarity score.
    #[must_use]
    pub fn overall_score(&self) -> f64 {
        // Weighted average of all metrics
        let hash_score =
            (self.dhash_similarity + self.ahash_similarity + self.phash_similarity) / 3.0;
        let feature_score = (self.feature_matches as f64 / 100.0).min(1.0);

        hash_score * 0.3 + self.histogram_similarity * 0.2 + self.ssim * 0.3 + feature_score * 0.2
    }

    /// Check if images are similar above threshold.
    #[must_use]
    pub fn is_similar(&self, threshold: f64) -> bool {
        self.overall_score() >= threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_image(width: usize, height: usize) -> Image {
        let data = (0..width * height).map(|i| (i % 256) as u8).collect();
        Image {
            width,
            height,
            data,
            channels: 1,
        }
    }

    #[test]
    fn test_image_creation() {
        let img = Image::new(100, 100, 3);
        assert_eq!(img.width, 100);
        assert_eq!(img.height, 100);
        assert_eq!(img.channels, 3);
        assert_eq!(img.data.len(), 100 * 100 * 3);
    }

    #[test]
    fn test_grayscale_conversion() {
        let img = create_test_image(10, 10);
        let gray = img.to_grayscale();
        assert_eq!(gray.channels, 1);
        assert_eq!(gray.width, 10);
        assert_eq!(gray.height, 10);
    }

    #[test]
    fn test_image_resize() {
        let img = create_test_image(100, 100);
        let resized = img.resize(50, 50);
        assert_eq!(resized.width, 50);
        assert_eq!(resized.height, 50);
    }

    #[test]
    fn test_dhash() {
        let img = create_test_image(64, 64);
        let hash = compute_dhash(&img);
        assert!(hash.hash() != 0);
    }

    #[test]
    fn test_ahash() {
        let img = create_test_image(64, 64);
        let hash = compute_ahash(&img);
        assert!(hash.hash() != 0);
    }

    #[test]
    fn test_phash() {
        let img = create_test_image(64, 64);
        let hash = compute_phash(&img);
        assert!(hash.hash() != 0);
    }

    #[test]
    fn test_hash_similarity() {
        let img1 = create_test_image(64, 64);
        let img2 = create_test_image(64, 64);

        let hash1 = compute_dhash(&img1);
        let hash2 = compute_dhash(&img2);

        // Same images should have high similarity
        assert_eq!(hash1.similarity(&hash2), 1.0);
    }

    #[test]
    fn test_histogram() {
        let img = create_test_image(10, 10);
        let hist = compute_histogram(&img);
        assert_eq!(hist.len(), 1); // Grayscale
        assert_eq!(hist[0].len(), 256);
    }

    #[test]
    fn test_histogram_comparison() {
        let img1 = create_test_image(10, 10);
        let img2 = create_test_image(10, 10);

        let hist1 = compute_histogram(&img1);
        let hist2 = compute_histogram(&img2);

        let similarity = compare_histograms(&hist1, &hist2);
        assert!(similarity >= 0.0 && similarity <= 1.0);
    }

    #[test]
    fn test_ssim() {
        let img1 = create_test_image(64, 64);
        let img2 = create_test_image(64, 64);

        let params = SsimParams::default();
        let ssim = compute_ssim(&img1, &img2, &params);

        // Same images should have SSIM close to 1.0
        assert!(ssim > 0.9);
    }

    #[test]
    fn test_feature_extraction() {
        let img = create_test_image(64, 64);
        let features = extract_features(&img);
        assert!(!features.is_empty());

        for feature in &features {
            assert!(!feature.descriptor.is_empty());
        }
    }

    #[test]
    fn test_feature_matching() {
        // Use a small image to keep feature extraction and O(n^2) matching fast
        let img = create_test_image(16, 16);
        let features1 = extract_features(&img);
        let features2 = extract_features(&img);

        let matches = match_features(&features1, &features2);
        assert!(matches > 0);
    }
}
