//! Color histogram for indexing.
//!
//! Provides an RGB cube-quantised colour histogram where each colour
//! channel is divided into equal-width bins.  The total number of bins
//! is `bins_per_channel^3`, so creating a histogram with 64 bins uses
//! 4 bins per channel (4^3 = 64).

/// Color histogram using RGB cube quantisation.
pub struct ColorHistogram {
    /// Bins holding colour counts / weights.
    bins: Vec<f32>,
    /// Number of bins per channel (derived from cube root of total bins).
    bins_per_channel: usize,
}

impl ColorHistogram {
    /// Create a new histogram with `num_bins` bins.
    ///
    /// `num_bins` should ideally be a perfect cube (e.g. 8, 27, 64, 125).
    /// Non-perfect cubes are rounded down to the nearest cube root.
    #[must_use]
    pub fn new(num_bins: usize) -> Self {
        let bpc = cube_root_floor(num_bins).max(1);
        let actual_bins = bpc * bpc * bpc;
        Self {
            bins: vec![0.0; actual_bins],
            bins_per_channel: bpc,
        }
    }

    /// Add a single colour sample to the histogram.
    ///
    /// Each channel value (0-255) is quantised into `bins_per_channel`
    /// equal-width bins and the corresponding 3D bin is incremented.
    pub fn add_color(&mut self, r: u8, g: u8, b: u8) {
        let bpc = self.bins_per_channel;
        let r_bin = (usize::from(r) * bpc) / 256;
        let g_bin = (usize::from(g) * bpc) / 256;
        let b_bin = (usize::from(b) * bpc) / 256;
        let idx = r_bin * bpc * bpc + g_bin * bpc + b_bin;
        if idx < self.bins.len() {
            self.bins[idx] += 1.0;
        }
    }

    /// Add all pixels from raw RGB data (contiguous R, G, B triplets).
    pub fn add_rgb_data(&mut self, data: &[u8]) {
        for chunk in data.chunks_exact(3) {
            self.add_color(chunk[0], chunk[1], chunk[2]);
        }
    }

    /// Normalize the histogram so all bins sum to 1.0.
    pub fn normalize(&mut self) {
        let sum: f32 = self.bins.iter().sum();
        if sum > 0.0 {
            for bin in &mut self.bins {
                *bin /= sum;
            }
        }
    }

    /// Get the bins as a slice.
    #[must_use]
    pub fn bins(&self) -> &[f32] {
        &self.bins
    }

    /// Return the total number of bins.
    #[must_use]
    pub fn num_bins(&self) -> usize {
        self.bins.len()
    }

    /// Compute the histogram intersection similarity with `other`.
    ///
    /// Both histograms should be normalised. Returns a value in [0, 1]
    /// where 1.0 means identical distributions.
    #[must_use]
    pub fn intersection(&self, other: &Self) -> f32 {
        self.bins
            .iter()
            .zip(other.bins.iter())
            .map(|(a, b)| a.min(*b))
            .sum()
    }

    /// Compute the chi-squared distance with `other`.
    ///
    /// Lower values indicate more similar histograms.
    #[must_use]
    pub fn chi_squared_distance(&self, other: &Self) -> f32 {
        self.bins
            .iter()
            .zip(other.bins.iter())
            .map(|(a, b)| {
                let sum = a + b;
                if sum > 1e-9 {
                    (a - b).powi(2) / sum
                } else {
                    0.0
                }
            })
            .sum()
    }
}

/// Compute the integer cube root (floored) of `n`.
fn cube_root_floor(n: usize) -> usize {
    if n == 0 {
        return 0;
    }
    let mut r = (n as f64).cbrt() as usize;
    while (r + 1) * (r + 1) * (r + 1) <= n {
        r += 1;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_bins() {
        let histogram = ColorHistogram::new(64);
        assert_eq!(histogram.bins().len(), 64);
        assert_eq!(histogram.bins_per_channel, 4);
    }

    #[test]
    fn test_add_color_increments_bin() {
        let mut h = ColorHistogram::new(64);
        h.add_color(0, 0, 0);
        h.add_color(0, 0, 0);
        assert!((h.bins()[0] - 2.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_add_color_different_bins() {
        let mut h = ColorHistogram::new(64);
        h.add_color(0, 0, 0);
        h.add_color(255, 255, 255);
        // 0,0,0 => bin 0; 255,255,255 => bin 63
        assert!((h.bins()[0] - 1.0).abs() < f32::EPSILON);
        assert!((h.bins()[63] - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_normalize() {
        let mut h = ColorHistogram::new(8);
        h.add_color(0, 0, 0);
        h.add_color(128, 128, 128);
        h.add_color(255, 255, 255);
        h.normalize();
        let sum: f32 = h.bins().iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum = {sum}");
    }

    #[test]
    fn test_add_rgb_data() {
        let mut h = ColorHistogram::new(64);
        let data = vec![0, 0, 0, 255, 255, 255, 128, 128, 128];
        h.add_rgb_data(&data);
        let total: f32 = h.bins().iter().sum();
        assert!((total - 3.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_intersection_identical() {
        let mut h1 = ColorHistogram::new(8);
        let mut h2 = ColorHistogram::new(8);
        h1.add_color(100, 100, 100);
        h2.add_color(100, 100, 100);
        h1.normalize();
        h2.normalize();
        let sim = h1.intersection(&h2);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_chi_squared_identical_is_zero() {
        let mut h1 = ColorHistogram::new(8);
        let mut h2 = ColorHistogram::new(8);
        h1.add_color(50, 50, 50);
        h2.add_color(50, 50, 50);
        h1.normalize();
        h2.normalize();
        let d = h1.chi_squared_distance(&h2);
        assert!(d < 1e-5);
    }
}
