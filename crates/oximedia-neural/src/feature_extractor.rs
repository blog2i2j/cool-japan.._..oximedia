//! Neural feature extraction for similarity search and content tagging.
//!
//! This module provides [`ImageFeatureExtractor`] which computes a compact,
//! deterministic feature vector from raw image data without requiring a
//! pre-trained model.  The representation combines three complementary
//! descriptors:
//!
//! 1. **Colour histogram** (32 bins per channel, quantised to a single
//!    luminance histogram for single-channel images, or per-channel
//!    histograms summed to 32 bins for RGB).
//! 2. **Edge histogram** (8 orientation bins computed via finite-difference
//!    gradient approximation, inspired by HOG-lite).
//! 3. **Spatial pyramid** (2×2 quadrant grid; per-quadrant channel mean →
//!    4 quadrants × 3 channels = 12 values for RGB, or 4 values for grey).
//!
//! These are concatenated and either **projected** down to `output_dim` via a
//! deterministic pseudo-random matrix (generated from a fixed seed using a
//! PCG-style PRNG) or **truncated / zero-padded** if the natural dimension
//! already matches.
//!
//! The resulting [`FeatureVector`] supports cosine similarity and L2
//! normalisation, enabling efficient nearest-neighbour lookup via
//! [`FeatureSimilaritySearch`].

// ─────────────────────────────────────────────────────────────────────────────
// FeatureVector
// ─────────────────────────────────────────────────────────────────────────────

/// A dense f32 feature vector of fixed dimensionality.
///
/// Created by [`ImageFeatureExtractor::extract`] and consumed by
/// [`FeatureSimilaritySearch`].
#[derive(Debug, Clone, PartialEq)]
pub struct FeatureVector {
    /// Raw feature values.
    pub data: Vec<f32>,
    /// Declared dimensionality — always equals `data.len()`.
    pub dim: usize,
}

impl FeatureVector {
    /// Creates a `FeatureVector` from a `Vec<f32>`.
    ///
    /// Sets `dim` to `data.len()`.
    pub fn new(data: Vec<f32>) -> Self {
        let dim = data.len();
        Self { data, dim }
    }

    /// Creates a zero-filled `FeatureVector` of the given dimension.
    pub fn zeros(dim: usize) -> Self {
        Self {
            data: vec![0.0; dim],
            dim,
        }
    }

    /// Computes the **cosine similarity** between `self` and `other`.
    ///
    /// Returns a value in `[-1, 1]`.  Returns `0.0` when either vector has
    /// zero L2 norm so that the function is always well-defined.
    pub fn cosine_similarity(&self, other: &FeatureVector) -> f32 {
        if self.dim == 0 || other.dim == 0 {
            return 0.0;
        }
        let min_len = self.data.len().min(other.data.len());
        let dot: f32 = self.data[..min_len]
            .iter()
            .zip(other.data[..min_len].iter())
            .map(|(&a, &b)| a * b)
            .sum();
        let norm_a = self.l2_norm();
        let norm_b = other.l2_norm();
        if norm_a < 1e-10 || norm_b < 1e-10 {
            return 0.0;
        }
        (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
    }

    /// Computes the L2 (Euclidean) norm of the vector.
    pub fn l2_norm(&self) -> f32 {
        self.data.iter().map(|&x| x * x).sum::<f32>().sqrt()
    }

    /// Normalises the vector **in place** so that its L2 norm equals 1.
    ///
    /// If the vector has zero norm, all elements remain zero.
    pub fn normalize(&mut self) {
        let norm = self.l2_norm();
        if norm > 1e-10 {
            for x in self.data.iter_mut() {
                *x /= norm;
            }
        }
    }

    /// Returns `true` if this vector is approximately unit-length (within
    /// `1e-5` tolerance).
    pub fn is_unit(&self) -> bool {
        (self.l2_norm() - 1.0).abs() < 1e-5
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// FeatureExtractionConfig
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for [`ImageFeatureExtractor`].
#[derive(Debug, Clone)]
pub struct FeatureExtractionConfig {
    /// Dimensionality of the output feature vector.  Defaults to `128`.
    pub output_dim: usize,
    /// Whether to L2-normalise the feature vector after extraction.
    /// Defaults to `true`.
    pub normalize_output: bool,
}

impl FeatureExtractionConfig {
    /// Creates a configuration with the given output dimension.
    pub fn new(output_dim: usize, normalize_output: bool) -> Self {
        Self {
            output_dim: output_dim.max(1),
            normalize_output,
        }
    }
}

impl Default for FeatureExtractionConfig {
    fn default() -> Self {
        Self {
            output_dim: 128,
            normalize_output: true,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ImageFeatureExtractor
// ─────────────────────────────────────────────────────────────────────────────

/// Extracts compact feature vectors from raw image buffers.
///
/// The input `image` slice is interpreted as packed **RGB** bytes when its
/// length equals `width × height × 3`; otherwise it is treated as a
/// single-channel (greyscale) buffer.
///
/// ## Feature layout (before projection)
///
/// | Segment | Size |
/// |---------|------|
/// | Colour histogram (32 bins) | 32 |
/// | Edge-orientation histogram (8 bins) | 8 |
/// | Spatial pyramid 2×2 (RGB: 4×3 = 12, grey: 4×1 = 4) | 12 or 4 |
///
/// Total natural dim: 52 (RGB) or 44 (grey).  These are projected to
/// `output_dim` via a fixed-seed pseudo-random projection matrix.
pub struct ImageFeatureExtractor {
    config: FeatureExtractionConfig,
}

impl ImageFeatureExtractor {
    /// Creates an extractor with the given configuration.
    pub fn new(config: FeatureExtractionConfig) -> Self {
        Self { config }
    }

    /// Creates an extractor with default settings (`output_dim = 128`,
    /// `normalize = true`).
    pub fn default_config() -> Self {
        Self {
            config: FeatureExtractionConfig::default(),
        }
    }

    /// Extracts a feature vector from a raw image buffer.
    ///
    /// `image` must contain at least `width * height` bytes.  RGB images
    /// should contain exactly `width * height * 3` bytes.
    ///
    /// Returns an error string if the input is empty or the buffer is too
    /// short.
    pub fn extract(&self, image: &[u8], width: u32, height: u32) -> Result<FeatureVector, String> {
        let w = width as usize;
        let h = height as usize;
        let total_pixels = w * h;

        if image.is_empty() || total_pixels == 0 {
            return Err("ImageFeatureExtractor: empty image".to_string());
        }
        if image.len() < total_pixels {
            return Err(format!(
                "ImageFeatureExtractor: buffer too short ({} < {})",
                image.len(),
                total_pixels
            ));
        }

        let channels = if image.len() >= total_pixels * 3 {
            3usize
        } else {
            1
        };

        // Extract luma plane.
        let luma: Vec<f32> = if channels == 3 {
            image
                .chunks_exact(3)
                .take(total_pixels)
                .map(|px| 0.299 * px[0] as f32 + 0.587 * px[1] as f32 + 0.114 * px[2] as f32)
                .collect()
        } else {
            image.iter().take(total_pixels).map(|&b| b as f32).collect()
        };

        // ── Segment 1: 32-bin colour histogram (luma) ─────────────────────
        let mut colour_hist = [0.0f32; 32];
        for &v in &luma {
            let bin = ((v / 255.0) * 31.999) as usize;
            colour_hist[bin.min(31)] += 1.0;
        }
        // L2-normalise.
        let ch_norm = colour_hist
            .iter()
            .map(|&v| v * v)
            .sum::<f32>()
            .sqrt()
            .max(1e-8);
        for v in colour_hist.iter_mut() {
            *v /= ch_norm;
        }

        // ── Segment 2: 8-bin edge orientation histogram ───────────────────
        let edge_hist = compute_edge_histogram(&luma, w, h);

        // ── Segment 3: spatial pyramid 2×2 per-quadrant channel means ─────
        let spatial = compute_spatial_pyramid(image, w, h, channels);

        // Concatenate all segments.
        let mut raw: Vec<f32> = Vec::with_capacity(32 + 8 + spatial.len());
        raw.extend_from_slice(&colour_hist);
        raw.extend_from_slice(&edge_hist);
        raw.extend_from_slice(&spatial);

        // ── Project to output_dim ─────────────────────────────────────────
        let projected = project_features(&raw, self.config.output_dim);

        let mut fv = FeatureVector::new(projected);
        if self.config.normalize_output {
            fv.normalize();
        }
        Ok(fv)
    }

    /// Returns the configured output dimensionality.
    pub fn output_dim(&self) -> usize {
        self.config.output_dim
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

/// 8-bin gradient-orientation histogram (HOG-lite), L2-normalised.
fn compute_edge_histogram(luma: &[f32], w: usize, h: usize) -> [f32; 8] {
    let mut hist = [0.0f32; 8];
    for row in 0..h {
        for col in 0..w {
            let idx = row * w + col;
            let gx = if col + 1 < w {
                luma[idx + 1] - luma[idx]
            } else {
                0.0
            };
            let gy = if row + 1 < h {
                luma[idx + w] - luma[idx]
            } else {
                0.0
            };
            let mag = (gx * gx + gy * gy).sqrt();
            if mag < 1e-6 {
                continue;
            }
            // Map atan2 ∈ (-π, π] → [0, π) unsigned orientation → 8 bins.
            let mut angle = gy.atan2(gx);
            if angle < 0.0 {
                angle += std::f32::consts::PI;
            }
            let bin = ((angle / std::f32::consts::PI) * 7.999) as usize;
            hist[bin.min(7)] += mag;
        }
    }
    let norm = hist.iter().map(|&v| v * v).sum::<f32>().sqrt().max(1e-8);
    for v in hist.iter_mut() {
        *v /= norm;
    }
    hist
}

/// Spatial pyramid: divide image into 2×2 quadrants, compute per-channel
/// mean for each quadrant.  Returns 4*channels values.
fn compute_spatial_pyramid(image: &[u8], w: usize, h: usize, channels: usize) -> Vec<f32> {
    let half_h = h / 2;
    let half_w = w / 2;
    // Quadrant offsets: (row_start, row_end, col_start, col_end).
    let quads = [
        (0, half_h.max(1), 0, half_w.max(1)),
        (0, half_h.max(1), half_w, w.max(half_w + 1)),
        (half_h, h.max(half_h + 1), 0, half_w.max(1)),
        (half_h, h.max(half_h + 1), half_w, w.max(half_w + 1)),
    ];

    let mut result = Vec::with_capacity(4 * channels);

    for (r0, r1, c0, c1) in quads {
        let mut sums = [0.0f64; 3];
        let mut count = 0usize;
        for row in r0..r1.min(h) {
            for col in c0..c1.min(w) {
                let pixel_idx = row * w + col;
                if channels == 3 {
                    let byte_idx = pixel_idx * 3;
                    if byte_idx + 2 < image.len() {
                        sums[0] += image[byte_idx] as f64;
                        sums[1] += image[byte_idx + 1] as f64;
                        sums[2] += image[byte_idx + 2] as f64;
                    }
                } else if pixel_idx < image.len() {
                    sums[0] += image[pixel_idx] as f64;
                }
                count += 1;
            }
        }
        let n = count.max(1) as f64;
        for ch in 0..channels {
            result.push((sums[ch] / n / 255.0) as f32);
        }
    }
    result
}

/// Projects `input` to `output_dim` using a deterministic pseudo-random
/// matrix generated from a fixed seed (PCG-style PRNG).
///
/// When `output_dim <= input.len()` the first `output_dim` values of the
/// matrix–vector product are returned (equivalent to a random projection).
/// The projection is scaled so the expected output variance matches the
/// input variance.
fn project_features(input: &[f32], output_dim: usize) -> Vec<f32> {
    let in_dim = input.len();
    let mut output = vec![0.0f32; output_dim];

    // PCG-like PRNG seeded with a fixed constant.
    let mut state: u64 = 0x853c_49e6_748f_ea9b;
    let increment: u64 = 0xda3e_39cb_94b9_5bdb;

    let scale = 1.0 / (in_dim.max(1) as f32).sqrt();

    for out_i in 0..output_dim {
        let mut acc = 0.0f32;
        for in_j in 0..in_dim {
            // Generate next pseudo-random u32.
            state = state
                .wrapping_mul(0x5851_f42d_4c95_7f2d)
                .wrapping_add(increment);
            let xor_shifted = (((state >> 18) ^ state) >> 27) as u32;
            let rot = (state >> 59) as u32;
            let rand_u32 = xor_shifted.rotate_right(rot);
            // Map to a weight in [-1, 1].
            let w = (rand_u32 as i32 as f32) / (i32::MAX as f32);
            acc += w * input[in_j];
            let _ = (out_i, in_j); // suppress unused warning
        }
        output[out_i] = acc * scale;
    }
    output
}

// ─────────────────────────────────────────────────────────────────────────────
// FeatureSimilaritySearch
// ─────────────────────────────────────────────────────────────────────────────

/// A simple in-memory nearest-neighbour store backed by an exhaustive cosine
/// similarity scan.
///
/// Suitable for libraries of up to tens of thousands of vectors.  For larger
/// scales, integrate with an approximate-nearest-neighbour index.
pub struct FeatureSimilaritySearch {
    entries: Vec<(String, FeatureVector)>,
}

impl FeatureSimilaritySearch {
    /// Creates an empty search index.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Adds a named feature vector to the index.
    ///
    /// Vectors with duplicate IDs are all retained; no deduplication is
    /// performed.
    pub fn add(&mut self, id: String, vector: FeatureVector) {
        self.entries.push((id, vector));
    }

    /// Searches the index for the `top_k` vectors most similar to `query`.
    ///
    /// Returns a `Vec` of `(id, similarity)` pairs sorted by cosine
    /// similarity **descending** (most similar first).
    ///
    /// If the index contains fewer than `top_k` entries, all entries are
    /// returned.
    pub fn search(&self, query: &FeatureVector, top_k: usize) -> Vec<(String, f32)> {
        if self.entries.is_empty() || top_k == 0 {
            return Vec::new();
        }
        let mut scored: Vec<(usize, f32)> = self
            .entries
            .iter()
            .enumerate()
            .map(|(i, (_, v))| (i, query.cosine_similarity(v)))
            .collect();

        // Sort descending by similarity.
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(i, sim)| (self.entries[i].0.clone(), sim))
            .collect()
    }

    /// Returns the number of vectors currently in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Removes all entries from the index.
    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

impl Default for FeatureSimilaritySearch {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FeatureVector ────────────────────────────────────────────────────────

    #[test]
    fn test_cosine_similarity_identical_is_one() {
        let mut v = FeatureVector::new(vec![1.0, 2.0, 3.0, 4.0]);
        v.normalize();
        let sim = v.cosine_similarity(&v.clone());
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "cosine_similarity of identical normalised vectors must be 1.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_orthogonal_is_zero() {
        let a = FeatureVector::new(vec![1.0, 0.0]);
        let b = FeatureVector::new(vec![0.0, 1.0]);
        let sim = a.cosine_similarity(&b);
        assert!(
            sim.abs() < 1e-5,
            "Orthogonal vectors must have cosine_similarity ≈ 0.0, got {sim}"
        );
    }

    #[test]
    fn test_cosine_similarity_opposite_is_minus_one() {
        let a = FeatureVector::new(vec![1.0, 0.0]);
        let b = FeatureVector::new(vec![-1.0, 0.0]);
        let sim = a.cosine_similarity(&b);
        assert!(
            (sim + 1.0).abs() < 1e-5,
            "Opposite vectors must have cosine_similarity ≈ -1.0"
        );
    }

    #[test]
    fn test_l2_norm_known_value() {
        let v = FeatureVector::new(vec![3.0, 4.0]);
        assert!(
            (v.l2_norm() - 5.0).abs() < 1e-5,
            "L2 norm of [3,4] must be 5.0"
        );
    }

    #[test]
    fn test_normalize_makes_unit_length() {
        let mut v = FeatureVector::new(vec![3.0, 4.0]);
        v.normalize();
        assert!(v.is_unit(), "Normalised vector must be unit-length");
    }

    #[test]
    fn test_normalize_zero_vector_stays_zero() {
        let mut v = FeatureVector::zeros(4);
        v.normalize();
        assert!(v.data.iter().all(|&x| x == 0.0));
    }

    // ── ImageFeatureExtractor ────────────────────────────────────────────────

    #[test]
    fn test_extractor_output_dim_matches_config() {
        let config = FeatureExtractionConfig::new(64, true);
        let extractor = ImageFeatureExtractor::new(config);
        let image = vec![128u8; 16 * 16 * 3];
        let fv = extractor.extract(&image, 16, 16).expect("extract failed");
        assert_eq!(fv.dim, 64);
        assert_eq!(fv.data.len(), 64);
    }

    #[test]
    fn test_extractor_default_dim_128() {
        let extractor = ImageFeatureExtractor::default_config();
        let image = vec![100u8; 32 * 32 * 3];
        let fv = extractor.extract(&image, 32, 32).expect("extract failed");
        assert_eq!(fv.dim, 128);
    }

    #[test]
    fn test_extractor_normalised_output_is_unit() {
        let config = FeatureExtractionConfig::new(64, true);
        let extractor = ImageFeatureExtractor::new(config);
        let image: Vec<u8> = (0u8..=255).cycle().take(32 * 32 * 3).collect();
        let fv = extractor.extract(&image, 32, 32).expect("extract failed");
        assert!(
            fv.is_unit(),
            "Normalised feature vector must be unit-length (norm={})",
            fv.l2_norm()
        );
    }

    #[test]
    fn test_extractor_empty_returns_error() {
        let extractor = ImageFeatureExtractor::default_config();
        assert!(extractor.extract(&[], 0, 0).is_err());
    }

    #[test]
    fn test_extractor_different_images_different_features() {
        let extractor = ImageFeatureExtractor::default_config();
        let img1 = vec![0u8; 16 * 16 * 3];
        let img2 = vec![200u8; 16 * 16 * 3];
        let fv1 = extractor.extract(&img1, 16, 16).expect("extract fv1");
        let fv2 = extractor.extract(&img2, 16, 16).expect("extract fv2");
        let sim = fv1.cosine_similarity(&fv2);
        // Two very different images should not have near-identical features.
        assert!(
            sim < 0.999,
            "Different images must produce different feature vectors (sim={sim})"
        );
    }

    #[test]
    fn test_extractor_identical_images_identical_features() {
        let extractor = ImageFeatureExtractor::default_config();
        let img = vec![123u8; 16 * 16 * 3];
        let fv1 = extractor.extract(&img, 16, 16).expect("extract fv1");
        let fv2 = extractor.extract(&img, 16, 16).expect("extract fv2");
        let sim = fv1.cosine_similarity(&fv2);
        assert!(
            (sim - 1.0).abs() < 1e-5,
            "Identical images must produce identical feature vectors (sim={sim})"
        );
    }

    // ── FeatureSimilaritySearch ──────────────────────────────────────────────

    #[test]
    fn test_search_returns_top_k_sorted() {
        let mut index = FeatureSimilaritySearch::new();
        // Insert 5 vectors: first = query, others gradually orthogonal.
        let query = FeatureVector::new(vec![1.0, 0.0, 0.0]);
        index.add("a".to_string(), FeatureVector::new(vec![1.0, 0.0, 0.0]));
        index.add("b".to_string(), FeatureVector::new(vec![0.9, 0.1, 0.0]));
        index.add("c".to_string(), FeatureVector::new(vec![0.5, 0.5, 0.0]));
        index.add("d".to_string(), FeatureVector::new(vec![0.0, 1.0, 0.0]));
        index.add("e".to_string(), FeatureVector::new(vec![-1.0, 0.0, 0.0]));

        let results = index.search(&query, 3);
        assert_eq!(results.len(), 3);
        // First result must be "a" (identical).
        assert_eq!(results[0].0, "a");
        assert!((results[0].1 - 1.0).abs() < 1e-5);
        // Results must be sorted descending.
        for i in 0..results.len() - 1 {
            assert!(
                results[i].1 >= results[i + 1].1,
                "Results must be sorted descending by similarity"
            );
        }
    }

    #[test]
    fn test_search_empty_index_returns_empty() {
        let index = FeatureSimilaritySearch::new();
        let query = FeatureVector::new(vec![1.0, 0.0]);
        let results = index.search(&query, 5);
        assert!(results.is_empty());
    }

    #[test]
    fn test_search_top_k_larger_than_index() {
        let mut index = FeatureSimilaritySearch::new();
        index.add("x".to_string(), FeatureVector::new(vec![1.0, 0.0]));
        let query = FeatureVector::new(vec![1.0, 0.0]);
        let results = index.search(&query, 100);
        assert_eq!(
            results.len(),
            1,
            "Should return all available entries when top_k > index size"
        );
    }

    #[test]
    fn test_search_most_similar_is_first() {
        let mut index = FeatureSimilaritySearch::new();
        index.add("close".to_string(), FeatureVector::new(vec![1.0, 0.1]));
        index.add("far".to_string(), FeatureVector::new(vec![0.0, 1.0]));
        let query = FeatureVector::new(vec![1.0, 0.0]);
        let results = index.search(&query, 2);
        assert_eq!(results[0].0, "close", "Most similar must be first");
    }
}
