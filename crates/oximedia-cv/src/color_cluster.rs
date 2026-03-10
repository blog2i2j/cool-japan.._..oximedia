#![allow(dead_code)]
//! Color clustering via K-Means for dominant color extraction.
//!
//! This module implements a simplified K-Means algorithm that operates on
//! RGB pixel data to extract dominant colors from images.

/// A single RGB color value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColorCluster {
    /// Red channel `[0, 255]`.
    pub r: f64,
    /// Green channel `[0, 255]`.
    pub g: f64,
    /// Blue channel `[0, 255]`.
    pub b: f64,
}

impl ColorCluster {
    /// Create a new color cluster centroid.
    #[must_use]
    pub fn new(r: f64, g: f64, b: f64) -> Self {
        Self { r, g, b }
    }

    /// Squared Euclidean distance to another color.
    #[must_use]
    pub fn distance_sq(&self, other: &Self) -> f64 {
        let dr = self.r - other.r;
        let dg = self.g - other.g;
        let db = self.b - other.b;
        dr * dr + dg * dg + db * db
    }

    /// Euclidean distance to another color.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f64 {
        self.distance_sq(other).sqrt()
    }

    /// Convert to an `[u8; 3]` by clamping.
    #[must_use]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn to_rgb_u8(&self) -> [u8; 3] {
        [
            self.r.clamp(0.0, 255.0).round() as u8,
            self.g.clamp(0.0, 255.0).round() as u8,
            self.b.clamp(0.0, 255.0).round() as u8,
        ]
    }

    /// Approximate luminance (ITU-R BT.601).
    #[must_use]
    pub fn luminance(&self) -> f64 {
        0.299 * self.r + 0.587 * self.g + 0.114 * self.b
    }
}

/// Result of a K-Means clustering run.
#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Centroids of the discovered clusters (dominant colors).
    pub centroids: Vec<ColorCluster>,
    /// Number of pixels assigned to each cluster.
    pub counts: Vec<usize>,
    /// Final inertia (sum of squared distances to nearest centroid).
    pub inertia: f64,
    /// Number of iterations executed.
    pub iterations: usize,
}

impl ClusterResult {
    /// Return the dominant color (largest cluster).
    #[must_use]
    pub fn dominant_color(&self) -> Option<ColorCluster> {
        self.counts
            .iter()
            .enumerate()
            .max_by_key(|(_, &c)| c)
            .map(|(i, _)| self.centroids[i])
    }

    /// Return centroids sorted by cluster size (largest first).
    #[must_use]
    pub fn sorted_by_count(&self) -> Vec<(ColorCluster, usize)> {
        let mut pairs: Vec<(ColorCluster, usize)> = self
            .centroids
            .iter()
            .copied()
            .zip(self.counts.iter().copied())
            .collect();
        pairs.sort_by(|a, b| b.1.cmp(&a.1));
        pairs
    }
}

/// K-Means color clustering engine.
#[derive(Debug)]
pub struct KMeansColorCluster {
    /// Number of clusters (k).
    pub k: usize,
    /// Maximum iterations.
    pub max_iter: usize,
    /// Convergence tolerance for centroid movement.
    pub tolerance: f64,
}

impl KMeansColorCluster {
    /// Create a new clusterer.
    #[must_use]
    pub fn new(k: usize, max_iter: usize, tolerance: f64) -> Self {
        Self {
            k: k.max(1),
            max_iter: max_iter.max(1),
            tolerance,
        }
    }

    /// Run K-Means on an array of RGB pixels.
    ///
    /// `pixels` is a flat slice of `[R, G, B, R, G, B, ...]`.
    /// Its length must be divisible by 3.
    ///
    /// Returns `None` if the input is empty or malformed.
    #[must_use]
    pub fn k_means(&self, pixels: &[u8]) -> Option<ClusterResult> {
        if pixels.is_empty() || pixels.len() % 3 != 0 {
            return None;
        }
        let n = pixels.len() / 3;
        let k = self.k.min(n);

        // Initialise centroids from evenly-spaced samples.
        let mut centroids: Vec<ColorCluster> = (0..k)
            .map(|i| {
                let idx = i * n / k;
                let off = idx * 3;
                ColorCluster::new(
                    f64::from(pixels[off]),
                    f64::from(pixels[off + 1]),
                    f64::from(pixels[off + 2]),
                )
            })
            .collect();

        let mut assignments = vec![0usize; n];
        let mut iterations = 0;

        for _ in 0..self.max_iter {
            iterations += 1;

            // Assignment step.
            for i in 0..n {
                let off = i * 3;
                let px = ColorCluster::new(
                    f64::from(pixels[off]),
                    f64::from(pixels[off + 1]),
                    f64::from(pixels[off + 2]),
                );
                let mut best = 0;
                let mut best_dist = f64::MAX;
                for (ci, c) in centroids.iter().enumerate() {
                    let d = px.distance_sq(c);
                    if d < best_dist {
                        best_dist = d;
                        best = ci;
                    }
                }
                assignments[i] = best;
            }

            // Update step.
            let mut sums_r = vec![0.0f64; k];
            let mut sums_g = vec![0.0f64; k];
            let mut sums_b = vec![0.0f64; k];
            let mut counts = vec![0usize; k];

            for i in 0..n {
                let off = i * 3;
                let ci = assignments[i];
                sums_r[ci] += f64::from(pixels[off]);
                sums_g[ci] += f64::from(pixels[off + 1]);
                sums_b[ci] += f64::from(pixels[off + 2]);
                counts[ci] += 1;
            }

            let mut max_shift = 0.0f64;
            for ci in 0..k {
                if counts[ci] == 0 {
                    continue;
                }
                let new_c = ColorCluster::new(
                    sums_r[ci] / counts[ci] as f64,
                    sums_g[ci] / counts[ci] as f64,
                    sums_b[ci] / counts[ci] as f64,
                );
                let shift = centroids[ci].distance(&new_c);
                if shift > max_shift {
                    max_shift = shift;
                }
                centroids[ci] = new_c;
            }

            if max_shift < self.tolerance {
                break;
            }
        }

        // Compute final inertia and counts.
        let mut inertia = 0.0f64;
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let off = i * 3;
            let px = ColorCluster::new(
                f64::from(pixels[off]),
                f64::from(pixels[off + 1]),
                f64::from(pixels[off + 2]),
            );
            let ci = assignments[i];
            inertia += px.distance_sq(&centroids[ci]);
            counts[ci] += 1;
        }

        Some(ClusterResult {
            centroids,
            counts,
            inertia,
            iterations,
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_color_cluster_new() {
        let c = ColorCluster::new(100.0, 150.0, 200.0);
        assert!((c.r - 100.0).abs() < f64::EPSILON);
        assert!((c.g - 150.0).abs() < f64::EPSILON);
        assert!((c.b - 200.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_distance_sq_zero() {
        let c = ColorCluster::new(1.0, 2.0, 3.0);
        assert!((c.distance_sq(&c)).abs() < f64::EPSILON);
    }

    #[test]
    fn test_distance_sq_known() {
        let a = ColorCluster::new(0.0, 0.0, 0.0);
        let b = ColorCluster::new(3.0, 4.0, 0.0);
        assert!((a.distance_sq(&b) - 25.0).abs() < 1e-9);
    }

    #[test]
    fn test_distance_known() {
        let a = ColorCluster::new(0.0, 0.0, 0.0);
        let b = ColorCluster::new(3.0, 4.0, 0.0);
        assert!((a.distance(&b) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_to_rgb_u8() {
        let c = ColorCluster::new(128.4, 0.0, 255.9);
        let rgb = c.to_rgb_u8();
        assert_eq!(rgb, [128, 0, 255]);
    }

    #[test]
    fn test_to_rgb_u8_clamp() {
        let c = ColorCluster::new(-10.0, 300.0, 127.5);
        let rgb = c.to_rgb_u8();
        assert_eq!(rgb, [0, 255, 128]);
    }

    #[test]
    fn test_luminance() {
        let white = ColorCluster::new(255.0, 255.0, 255.0);
        assert!((white.luminance() - 255.0).abs() < 1e-9);
        let black = ColorCluster::new(0.0, 0.0, 0.0);
        assert!((black.luminance()).abs() < f64::EPSILON);
    }

    #[test]
    fn test_kmeans_empty() {
        let km = KMeansColorCluster::new(3, 10, 0.1);
        assert!(km.k_means(&[]).is_none());
    }

    #[test]
    fn test_kmeans_bad_len() {
        let km = KMeansColorCluster::new(2, 10, 0.1);
        assert!(km.k_means(&[1, 2]).is_none()); // not divisible by 3
    }

    #[test]
    fn test_kmeans_single_pixel() {
        let km = KMeansColorCluster::new(1, 10, 0.1);
        let result = km
            .k_means(&[100, 150, 200])
            .expect("k_means should succeed");
        assert_eq!(result.centroids.len(), 1);
        assert!((result.centroids[0].r - 100.0).abs() < 1e-9);
        assert_eq!(result.counts[0], 1);
    }

    #[test]
    fn test_kmeans_two_clusters() {
        // 4 red pixels and 4 blue pixels
        let mut pixels = Vec::new();
        for _ in 0..4 {
            pixels.extend_from_slice(&[255, 0, 0]);
        }
        for _ in 0..4 {
            pixels.extend_from_slice(&[0, 0, 255]);
        }
        let km = KMeansColorCluster::new(2, 50, 0.01);
        let result = km.k_means(&pixels).expect("k_means should succeed");
        assert_eq!(result.centroids.len(), 2);
        // Both clusters should have 4 members
        assert!(result.counts.iter().all(|&c| c == 4));
    }

    #[test]
    fn test_dominant_color() {
        // 6 red, 2 green
        let mut pixels = Vec::new();
        for _ in 0..6 {
            pixels.extend_from_slice(&[255, 0, 0]);
        }
        for _ in 0..2 {
            pixels.extend_from_slice(&[0, 255, 0]);
        }
        let km = KMeansColorCluster::new(2, 50, 0.01);
        let result = km.k_means(&pixels).expect("k_means should succeed");
        let dom = result
            .dominant_color()
            .expect("dominant_color should succeed");
        // The dominant should be the red cluster
        assert!(dom.r > dom.g);
    }

    #[test]
    fn test_sorted_by_count() {
        let result = ClusterResult {
            centroids: vec![
                ColorCluster::new(0.0, 0.0, 0.0),
                ColorCluster::new(255.0, 255.0, 255.0),
            ],
            counts: vec![10, 50],
            inertia: 0.0,
            iterations: 1,
        };
        let sorted = result.sorted_by_count();
        assert_eq!(sorted[0].1, 50);
        assert_eq!(sorted[1].1, 10);
    }

    #[test]
    fn test_kmeans_k_exceeds_n() {
        // k=5 but only 2 pixels => clamped to k=2
        let km = KMeansColorCluster::new(5, 10, 0.1);
        let result = km
            .k_means(&[10, 20, 30, 40, 50, 60])
            .expect("k_means should succeed");
        assert_eq!(result.centroids.len(), 2);
    }

    #[test]
    fn test_kmeans_converges() {
        let mut pixels = Vec::new();
        for _ in 0..20 {
            pixels.extend_from_slice(&[100, 100, 100]);
        }
        let km = KMeansColorCluster::new(1, 100, 0.001);
        let result = km.k_means(&pixels).expect("k_means should succeed");
        // Should converge in 1 or 2 iterations since all points are the same.
        assert!(result.iterations <= 2);
        assert!(result.inertia < 1e-6);
    }

    #[test]
    fn test_cluster_result_dominant_empty() {
        let result = ClusterResult {
            centroids: Vec::new(),
            counts: Vec::new(),
            inertia: 0.0,
            iterations: 0,
        };
        assert!(result.dominant_color().is_none());
    }
}
