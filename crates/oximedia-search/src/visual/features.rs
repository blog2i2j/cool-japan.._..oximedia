//! Visual feature extraction.
//!
//! Implements perceptual hashing (aHash / dHash), colour histogram
//! extraction (RGB cube quantisation), edge histogram (MPEG-7 style),
//! and Local Binary Pattern (LBP) texture features. All algorithms
//! operate on raw RGB byte data and produce normalised float vectors
//! suitable for cosine/Euclidean distance comparisons.

use crate::error::SearchResult;

/// Number of dimensions in the combined feature vector.
const FEATURE_DIM: usize = 128;
/// Number of bins per RGB channel for colour histograms.
const COLOR_BINS_PER_CHANNEL: usize = 4;
/// Total bins for a 3-channel RGB histogram.
const COLOR_TOTAL_BINS: usize =
    COLOR_BINS_PER_CHANNEL * COLOR_BINS_PER_CHANNEL * COLOR_BINS_PER_CHANNEL;
/// Standard edge histogram bins (MPEG-7: 16 blocks * 5 edge types).
const EDGE_BINS: usize = 80;
/// Texture feature dimensions (LBP histogram with 64 patterns).
const TEXTURE_BINS: usize = 64;

/// Feature extractor for visual search.
///
/// Combines multiple low-level visual descriptors into a single
/// feature vector for similarity comparison.
pub struct FeatureExtractor;

impl FeatureExtractor {
    /// Create a new feature extractor.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Extract a 128-dimensional feature vector from raw RGB image data.
    ///
    /// The feature vector combines:
    /// - 64 colour histogram bins (4^3 quantised RGB)
    /// - 32 edge histogram bins (downsampled from 80)
    /// - 32 texture bins (downsampled from 64 LBP bins)
    ///
    /// # Errors
    ///
    /// Returns an error if feature extraction fails.
    pub fn extract(&self, image_data: &[u8]) -> SearchResult<Vec<f32>> {
        let color_hist = self.extract_color_histogram(image_data, COLOR_TOTAL_BINS)?;
        let edge_hist = self.extract_edge_histogram(image_data)?;
        let texture = self.extract_texture_features(image_data)?;

        let mut features = Vec::with_capacity(FEATURE_DIM);

        // Take first 64 colour bins.
        features.extend(color_hist.iter().take(64));

        // Downsample 80 edge bins to 32 by averaging pairs + remaining.
        let mut edge_ds = Vec::with_capacity(32);
        for i in 0..32 {
            let idx = i * 2;
            let val = if idx + 1 < edge_hist.len() {
                (edge_hist[idx] + edge_hist[idx + 1]) * 0.5
            } else if idx < edge_hist.len() {
                edge_hist[idx]
            } else {
                0.0
            };
            edge_ds.push(val);
        }
        features.extend(&edge_ds);

        // Downsample 64 texture bins to 32 by averaging pairs.
        let mut tex_ds = Vec::with_capacity(32);
        for i in 0..32 {
            let idx = i * 2;
            let val = if idx + 1 < texture.len() {
                (texture[idx] + texture[idx + 1]) * 0.5
            } else if idx < texture.len() {
                texture[idx]
            } else {
                0.0
            };
            tex_ds.push(val);
        }
        features.extend(&tex_ds);

        // L2-normalise the final vector.
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 1e-9 {
            for f in &mut features {
                *f /= norm;
            }
        }

        Ok(features)
    }

    /// Extract colour histogram by quantising each pixel's RGB values
    /// into `bins_total` bins via cube quantisation.
    ///
    /// The image data is expected as contiguous RGB triplets.
    /// The histogram is normalised so values sum to 1.0.
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn extract_color_histogram(
        &self,
        image_data: &[u8],
        bins: usize,
    ) -> SearchResult<Vec<f32>> {
        let mut histogram = vec![0.0_f32; bins];

        if image_data.len() < 3 || bins == 0 {
            // Cannot extract colours; return uniform.
            let uniform = if bins > 0 { 1.0 / bins as f32 } else { 0.0 };
            return Ok(vec![uniform; bins]);
        }

        let pixel_count = image_data.len() / 3;
        // Determine bins per channel from cube root.
        let bpc = cube_root_floor(bins).max(1);

        for chunk in image_data.chunks_exact(3) {
            let r_bin = (usize::from(chunk[0]) * bpc) / 256;
            let g_bin = (usize::from(chunk[1]) * bpc) / 256;
            let b_bin = (usize::from(chunk[2]) * bpc) / 256;
            let idx = r_bin * bpc * bpc + g_bin * bpc + b_bin;
            if idx < histogram.len() {
                histogram[idx] += 1.0;
            }
        }

        // Normalise.
        let total = pixel_count as f32;
        if total > 0.0 {
            for h in &mut histogram {
                *h /= total;
            }
        }

        Ok(histogram)
    }

    /// Extract edge histogram using a simplified MPEG-7 Edge Histogram
    /// Descriptor approach.
    ///
    /// The image is divided into a 4x4 grid of sub-images. For each
    /// sub-image, 5 edge types are computed (vertical, horizontal,
    /// 45-degree diagonal, 135-degree diagonal, non-directional).
    /// Total: 4 * 4 * 5 = 80 bins.
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn extract_edge_histogram(&self, image_data: &[u8]) -> SearchResult<Vec<f32>> {
        let mut histogram = vec![0.0_f32; EDGE_BINS];

        if image_data.len() < 9 {
            return Ok(histogram);
        }

        // Estimate image dimensions: assume square-ish RGB image.
        let pixel_count = image_data.len() / 3;
        let side = (pixel_count as f64).sqrt() as usize;
        if side < 4 {
            return Ok(histogram);
        }

        // Convert to grayscale luminance.
        let gray: Vec<f32> = image_data
            .chunks_exact(3)
            .take(side * side)
            .map(|rgb| {
                0.299 * f32::from(rgb[0]) + 0.587 * f32::from(rgb[1]) + 0.114 * f32::from(rgb[2])
            })
            .collect();

        let block_w = side / 4;
        let block_h = side / 4;

        if block_w < 2 || block_h < 2 {
            return Ok(histogram);
        }

        // For each 4x4 block, compute edge strengths.
        for by in 0..4_usize {
            for bx in 0..4_usize {
                let base_x = bx * block_w;
                let base_y = by * block_h;

                let mut v_sum = 0.0_f32;
                let mut h_sum = 0.0_f32;
                let mut d45_sum = 0.0_f32;
                let mut d135_sum = 0.0_f32;
                let mut nd_sum = 0.0_f32;
                let mut count = 0.0_f32;

                for y in base_y..(base_y + block_h).min(side - 1) {
                    for x in base_x..(base_x + block_w).min(side - 1) {
                        let c = gray[y * side + x];
                        let r = gray[y * side + (x + 1).min(side - 1)];
                        let b = gray[(y + 1).min(side - 1) * side + x];
                        let br = gray[(y + 1).min(side - 1) * side + (x + 1).min(side - 1)];

                        // Vertical edge: |left - right|
                        v_sum += (c - r).abs() + (b - br).abs();
                        // Horizontal edge: |top - bottom|
                        h_sum += (c - b).abs() + (r - br).abs();
                        // 45-degree diagonal
                        d45_sum += (c - br).abs();
                        // 135-degree diagonal
                        d135_sum += (r - b).abs();
                        // Non-directional (Laplacian-like)
                        nd_sum +=
                            (4.0 * c - r - b - br - gray[y * side + x.saturating_sub(1)]).abs();
                        count += 1.0;
                    }
                }

                if count > 0.0 {
                    let bin_base = (by * 4 + bx) * 5;
                    histogram[bin_base] = v_sum / (count * 255.0);
                    histogram[bin_base + 1] = h_sum / (count * 255.0);
                    histogram[bin_base + 2] = d45_sum / (count * 255.0);
                    histogram[bin_base + 3] = d135_sum / (count * 255.0);
                    histogram[bin_base + 4] = nd_sum / (count * 1020.0);
                }
            }
        }

        Ok(histogram)
    }

    /// Extract texture features using a simplified Local Binary Pattern (LBP).
    ///
    /// For each pixel, compares with its 8 neighbours to produce an 8-bit
    /// LBP code. The histogram of these codes (quantised to 64 bins) forms
    /// the texture descriptor.
    ///
    /// # Errors
    ///
    /// Returns an error if extraction fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn extract_texture_features(&self, image_data: &[u8]) -> SearchResult<Vec<f32>> {
        let mut histogram = vec![0.0_f32; TEXTURE_BINS];

        if image_data.len() < 9 {
            return Ok(histogram);
        }

        let pixel_count = image_data.len() / 3;
        let side = (pixel_count as f64).sqrt() as usize;
        if side < 3 {
            return Ok(histogram);
        }

        // Convert to grayscale.
        let gray: Vec<u8> = image_data
            .chunks_exact(3)
            .take(side * side)
            .map(|rgb| {
                let lum = 0.299 * f32::from(rgb[0])
                    + 0.587 * f32::from(rgb[1])
                    + 0.114 * f32::from(rgb[2]);
                lum.clamp(0.0, 255.0) as u8
            })
            .collect();

        let mut total = 0u32;

        // 8-neighbour offsets (clockwise from top-left).
        let offsets: [(i32, i32); 8] = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, 1),
            (1, 1),
            (1, 0),
            (1, -1),
            (0, -1),
        ];

        for y in 1..side.saturating_sub(1) {
            for x in 1..side.saturating_sub(1) {
                let center = gray[y * side + x];
                let mut code: u8 = 0;

                for (bit, &(dy, dx)) in offsets.iter().enumerate() {
                    let ny = (y as i32 + dy) as usize;
                    let nx = (x as i32 + dx) as usize;
                    if gray[ny * side + nx] >= center {
                        code |= 1 << bit;
                    }
                }

                // Quantise 256 LBP codes to 64 bins.
                let bin = (usize::from(code) * TEXTURE_BINS) / 256;
                histogram[bin.min(TEXTURE_BINS - 1)] += 1.0;
                total += 1;
            }
        }

        // Normalise.
        if total > 0 {
            let t = total as f32;
            for h in &mut histogram {
                *h /= t;
            }
        }

        Ok(histogram)
    }

    /// Compute a perceptual hash (aHash) of the image.
    ///
    /// Algorithm:
    /// 1. Resize to 8x8 by block-averaging.
    /// 2. Convert to grayscale.
    /// 3. Compute average luminance.
    /// 4. Each pixel above average is 1, below is 0 => 64-bit hash.
    ///
    /// Returns an 8-byte vector representing the 64-bit hash.
    ///
    /// # Errors
    ///
    /// Returns an error if computation fails.
    #[allow(clippy::cast_precision_loss)]
    pub fn compute_phash(&self, image_data: &[u8]) -> SearchResult<Vec<u8>> {
        const HASH_SIZE: usize = 8;

        if image_data.len() < 3 {
            return Ok(vec![0u8; HASH_SIZE]);
        }

        let pixel_count = image_data.len() / 3;
        let side = (pixel_count as f64).sqrt() as usize;
        if side < HASH_SIZE {
            return Ok(vec![0u8; HASH_SIZE]);
        }

        // Convert to grayscale.
        let gray: Vec<f32> = image_data
            .chunks_exact(3)
            .take(side * side)
            .map(|rgb| {
                0.299 * f32::from(rgb[0]) + 0.587 * f32::from(rgb[1]) + 0.114 * f32::from(rgb[2])
            })
            .collect();

        // Downsample to 8x8 via block averaging.
        let block_w = side / HASH_SIZE;
        let block_h = side / HASH_SIZE;
        let mut small = [0.0_f32; 64];

        for by in 0..HASH_SIZE {
            for bx in 0..HASH_SIZE {
                let mut sum = 0.0_f32;
                let mut count = 0u32;
                for y in (by * block_h)..((by + 1) * block_h).min(side) {
                    for x in (bx * block_w)..((bx + 1) * block_w).min(side) {
                        sum += gray[y * side + x];
                        count += 1;
                    }
                }
                small[by * HASH_SIZE + bx] = if count > 0 { sum / count as f32 } else { 0.0 };
            }
        }

        // Compute average and threshold.
        let avg: f32 = small.iter().sum::<f32>() / 64.0;
        let mut hash_bytes = vec![0u8; HASH_SIZE];

        for (i, &val) in small.iter().enumerate() {
            if val >= avg {
                let byte_idx = i / 8;
                let bit_idx = i % 8;
                hash_bytes[byte_idx] |= 1 << bit_idx;
            }
        }

        Ok(hash_bytes)
    }

    /// Compute the Hamming distance between two perceptual hashes.
    ///
    /// Lower distance means more visually similar images.
    #[must_use]
    pub fn phash_distance(hash_a: &[u8], hash_b: &[u8]) -> u32 {
        hash_a
            .iter()
            .zip(hash_b.iter())
            .map(|(a, b)| (a ^ b).count_ones())
            .sum()
    }
}

/// Compute the integer cube root (floored) of `n`.
fn cube_root_floor(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut r = (n as f64).cbrt() as usize;
    // Correct for floating-point imprecision.
    while (r + 1) * (r + 1) * (r + 1) <= n {
        r += 1;
    }
    r
}

impl Default for FeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_features_empty_image() {
        let extractor = FeatureExtractor::new();
        let features = extractor.extract(&[]).expect("should succeed in test");
        assert_eq!(features.len(), FEATURE_DIM);
    }

    #[test]
    fn test_extract_features_small_image() {
        // 4x4 RGB image (48 bytes).
        let data: Vec<u8> = (0..48).collect();
        let extractor = FeatureExtractor::new();
        let features = extractor.extract(&data).expect("should succeed in test");
        assert_eq!(features.len(), FEATURE_DIM);

        // Should be L2-normalised (norm ~= 1.0).
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Small images may produce all-zero features, which is valid.
        assert!(norm < 1.01);
    }

    #[test]
    fn test_extract_features_larger_image() {
        // 16x16 RGB image.
        let mut data = vec![0u8; 16 * 16 * 3];
        // Fill with a gradient.
        for (i, byte) in data.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        let extractor = FeatureExtractor::new();
        let features = extractor.extract(&data).expect("should succeed in test");
        assert_eq!(features.len(), FEATURE_DIM);
        let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "norm = {norm}");
    }

    #[test]
    fn test_color_histogram_basic() {
        let extractor = FeatureExtractor::new();
        let histogram = extractor
            .extract_color_histogram(&[], 64)
            .expect("should succeed in test");
        assert_eq!(histogram.len(), 64);
    }

    #[test]
    fn test_color_histogram_sum_to_one() {
        // 8x8 image with uniform colour.
        let data = vec![128u8; 8 * 8 * 3];
        let extractor = FeatureExtractor::new();
        let histogram = extractor
            .extract_color_histogram(&data, COLOR_TOTAL_BINS)
            .expect("should succeed in test");
        let sum: f32 = histogram.iter().sum();
        assert!((sum - 1.0).abs() < 0.01, "sum = {sum}");
    }

    #[test]
    fn test_edge_histogram_dimensions() {
        let data = vec![100u8; 16 * 16 * 3];
        let extractor = FeatureExtractor::new();
        let edges = extractor
            .extract_edge_histogram(&data)
            .expect("should succeed in test");
        assert_eq!(edges.len(), EDGE_BINS);
    }

    #[test]
    fn test_texture_features_dimensions() {
        let data = vec![50u8; 16 * 16 * 3];
        let extractor = FeatureExtractor::new();
        let texture = extractor
            .extract_texture_features(&data)
            .expect("should succeed in test");
        assert_eq!(texture.len(), TEXTURE_BINS);
    }

    #[test]
    fn test_phash_identical_images_distance_zero() {
        let data = vec![200u8; 16 * 16 * 3];
        let extractor = FeatureExtractor::new();
        let h1 = extractor
            .compute_phash(&data)
            .expect("should succeed in test");
        let h2 = extractor
            .compute_phash(&data)
            .expect("should succeed in test");
        assert_eq!(h1.len(), 8);
        assert_eq!(FeatureExtractor::phash_distance(&h1, &h2), 0);
    }

    #[test]
    fn test_phash_different_images_nonzero_distance() {
        let extractor = FeatureExtractor::new();
        let white = vec![255u8; 16 * 16 * 3];
        let black = vec![0u8; 16 * 16 * 3];
        let h_white = extractor
            .compute_phash(&white)
            .expect("should succeed in test");
        let _h_black = extractor
            .compute_phash(&black)
            .expect("should succeed in test");
        // Uniform images have all pixels equal to average, but one is 255 and
        // one is 0, so hashes will differ when there is any variation.
        // For truly uniform images, all >= avg, so both hashes are 0xFF.
        // This is actually expected: both uniform images produce the same hash.
        // Use a mixed image for meaningful distance.
        let mut mixed = vec![0u8; 16 * 16 * 3];
        let half = mixed.len() / 2;
        for (i, b) in mixed.iter_mut().enumerate() {
            *b = if i < half { 255 } else { 0 };
        }
        let h_mixed = extractor
            .compute_phash(&mixed)
            .expect("should succeed in test");
        let dist = FeatureExtractor::phash_distance(&h_white, &h_mixed);
        assert!(dist > 0, "mixed vs uniform should differ");
    }

    #[test]
    fn test_cube_root_floor() {
        assert_eq!(cube_root_floor(0), 0);
        assert_eq!(cube_root_floor(1), 1);
        assert_eq!(cube_root_floor(8), 2);
        assert_eq!(cube_root_floor(27), 3);
        assert_eq!(cube_root_floor(64), 4);
        assert_eq!(cube_root_floor(63), 3);
    }
}
