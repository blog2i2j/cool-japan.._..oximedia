//! Histogram operations.
//!
//! This module provides histogram computation and equalization algorithms
//! including standard histogram equalization and CLAHE (Contrast Limited
//! Adaptive Histogram Equalization).
//!
//! # Example
//!
//! ```
//! use oximedia_cv::image::Histogram;
//!
//! let data = vec![0u8, 64, 128, 192, 255];
//! let hist = Histogram::compute(&data);
//! assert_eq!(hist.bins().len(), 256);
//! ```

use crate::error::{CvError, CvResult};

/// Histogram structure for grayscale images.
///
/// Stores the frequency distribution of pixel values (0-255).
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Frequency bins for each intensity value (0-255).
    bins: [u32; 256],
    /// Total number of pixels.
    total: u32,
}

impl Histogram {
    /// Create a new empty histogram.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::Histogram;
    ///
    /// let hist = Histogram::new();
    /// assert_eq!(hist.total(), 0);
    /// ```
    #[must_use]
    pub const fn new() -> Self {
        Self {
            bins: [0; 256],
            total: 0,
        }
    }

    /// Compute histogram from image data.
    ///
    /// # Arguments
    ///
    /// * `data` - Grayscale image data
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::Histogram;
    ///
    /// let data = vec![0u8, 128, 128, 255];
    /// let hist = Histogram::compute(&data);
    /// assert_eq!(hist.bin(0), 1);
    /// assert_eq!(hist.bin(128), 2);
    /// assert_eq!(hist.bin(255), 1);
    /// ```
    #[must_use]
    pub fn compute(data: &[u8]) -> Self {
        let mut hist = Self::new();

        for &pixel in data {
            hist.bins[pixel as usize] += 1;
        }

        hist.total = data.len() as u32;
        hist
    }

    /// Get the frequency for a specific bin.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::Histogram;
    ///
    /// let hist = Histogram::compute(&[100u8, 100, 100]);
    /// assert_eq!(hist.bin(100), 3);
    /// assert_eq!(hist.bin(50), 0);
    /// ```
    #[must_use]
    pub const fn bin(&self, index: u8) -> u32 {
        self.bins[index as usize]
    }

    /// Get all bins as a slice.
    #[must_use]
    pub const fn bins(&self) -> &[u32; 256] {
        &self.bins
    }

    /// Get the total number of pixels.
    #[must_use]
    pub const fn total(&self) -> u32 {
        self.total
    }

    /// Get the minimum non-zero bin index.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::Histogram;
    ///
    /// let hist = Histogram::compute(&[50u8, 100, 150]);
    /// assert_eq!(hist.min_bin(), Some(50));
    /// ```
    #[must_use]
    pub fn min_bin(&self) -> Option<u8> {
        self.bins.iter().position(|&b| b > 0).map(|i| i as u8)
    }

    /// Get the maximum non-zero bin index.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::Histogram;
    ///
    /// let hist = Histogram::compute(&[50u8, 100, 150]);
    /// assert_eq!(hist.max_bin(), Some(150));
    /// ```
    #[must_use]
    pub fn max_bin(&self) -> Option<u8> {
        self.bins.iter().rposition(|&b| b > 0).map(|i| i as u8)
    }

    /// Compute the mean intensity.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::Histogram;
    ///
    /// let hist = Histogram::compute(&[100u8, 100, 100]);
    /// assert!((hist.mean() - 100.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }

        let sum: u64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &count)| i as u64 * count as u64)
            .sum();

        sum as f64 / self.total as f64
    }

    /// Compute the standard deviation.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::Histogram;
    ///
    /// let hist = Histogram::compute(&[100u8, 100, 100]);
    /// assert!(hist.std_dev() < 0.001); // All same value
    /// ```
    #[must_use]
    pub fn std_dev(&self) -> f64 {
        if self.total == 0 {
            return 0.0;
        }

        let mean = self.mean();
        let variance: f64 = self
            .bins
            .iter()
            .enumerate()
            .map(|(i, &count)| {
                let diff = i as f64 - mean;
                diff * diff * count as f64
            })
            .sum::<f64>()
            / self.total as f64;

        variance.sqrt()
    }

    /// Compute the cumulative distribution function (CDF).
    ///
    /// Returns an array where `cdf[i]` is the cumulative probability up to intensity `i`.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::Histogram;
    ///
    /// let hist = Histogram::compute(&[0u8, 128, 255]);
    /// let cdf = hist.cdf();
    /// assert!((cdf[255] - 1.0).abs() < 0.001);
    /// ```
    #[must_use]
    pub fn cdf(&self) -> [f64; 256] {
        let mut cdf = [0.0; 256];

        if self.total == 0 {
            return cdf;
        }

        let mut cumsum = 0u64;
        for (i, &count) in self.bins.iter().enumerate() {
            cumsum += count as u64;
            cdf[i] = cumsum as f64 / self.total as f64;
        }

        cdf
    }

    /// Compute the cumulative histogram (integer counts).
    #[must_use]
    pub fn cumulative(&self) -> [u32; 256] {
        let mut cumulative = [0u32; 256];
        let mut cumsum = 0u32;

        for (i, &count) in self.bins.iter().enumerate() {
            cumsum += count;
            cumulative[i] = cumsum;
        }

        cumulative
    }

    /// Normalize the histogram to sum to 1.0.
    #[must_use]
    pub fn normalized(&self) -> [f64; 256] {
        let mut normalized = [0.0; 256];

        if self.total == 0 {
            return normalized;
        }

        for (i, &count) in self.bins.iter().enumerate() {
            normalized[i] = count as f64 / self.total as f64;
        }

        normalized
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

/// Perform histogram equalization on a grayscale image.
///
/// This technique improves the contrast by spreading out the intensity
/// distribution to use the full available range.
///
/// # Arguments
///
/// * `src` - Source grayscale image data
/// * `width` - Image width
/// * `height` - Image height
///
/// # Returns
///
/// Equalized grayscale image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::histogram::histogram_equalization;
///
/// let src = vec![50u8; 100];
/// let result = histogram_equalization(&src, 10, 10)?;
/// assert_eq!(result.len(), 100);
/// ```
pub fn histogram_equalization(src: &[u8], width: u32, height: u32) -> CvResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected_size = width as usize * height as usize;
    if src.len() < expected_size {
        return Err(CvError::insufficient_data(expected_size, src.len()));
    }

    let hist = Histogram::compute(src);
    let cdf = hist.cumulative();

    // Find the minimum non-zero CDF value
    let cdf_min = cdf.iter().copied().find(|&c| c > 0).unwrap_or(0);

    let mut dst = vec![0u8; expected_size];
    let total = hist.total();

    if total <= cdf_min {
        // Edge case: all pixels have the same value
        return Ok(src.to_vec());
    }

    for (i, &pixel) in src.iter().enumerate().take(expected_size) {
        let cdf_val = cdf[pixel as usize];
        let equalized = ((cdf_val - cdf_min) as f64 * 255.0 / (total - cdf_min) as f64)
            .round()
            .clamp(0.0, 255.0);
        dst[i] = equalized as u8;
    }

    Ok(dst)
}

/// Compute histogram for a specific region of an image.
#[must_use]
pub fn compute_region_histogram(
    src: &[u8],
    width: u32,
    x: u32,
    y: u32,
    region_width: u32,
    region_height: u32,
) -> Histogram {
    let mut hist = Histogram::new();

    for ry in 0..region_height {
        let row_start = ((y + ry) * width + x) as usize;
        for rx in 0..region_width {
            let idx = row_start + rx as usize;
            if idx < src.len() {
                hist.bins[src[idx] as usize] += 1;
                hist.total += 1;
            }
        }
    }

    hist
}

/// CLAHE (Contrast Limited Adaptive Histogram Equalization) configuration.
#[derive(Debug, Clone, Copy)]
pub struct ClaheConfig {
    /// Number of tiles in the x direction.
    pub tiles_x: u32,
    /// Number of tiles in the y direction.
    pub tiles_y: u32,
    /// Clip limit for contrast limiting (1.0 = no clipping).
    pub clip_limit: f64,
}

impl Default for ClaheConfig {
    fn default() -> Self {
        Self {
            tiles_x: 8,
            tiles_y: 8,
            clip_limit: 2.0,
        }
    }
}

impl ClaheConfig {
    /// Create a new CLAHE configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::histogram::ClaheConfig;
    ///
    /// let config = ClaheConfig::new(8, 8, 2.0);
    /// assert_eq!(config.tiles_x, 8);
    /// ```
    #[must_use]
    pub const fn new(tiles_x: u32, tiles_y: u32, clip_limit: f64) -> Self {
        Self {
            tiles_x,
            tiles_y,
            clip_limit,
        }
    }
}

/// Perform adaptive histogram equalization (CLAHE).
///
/// CLAHE divides the image into tiles and applies histogram equalization
/// locally, with contrast limiting to prevent noise amplification.
///
/// # Arguments
///
/// * `src` - Source grayscale image data
/// * `width` - Image width
/// * `height` - Image height
/// * `config` - CLAHE configuration
///
/// # Returns
///
/// Equalized grayscale image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::histogram::{adaptive_histogram_equalization, ClaheConfig};
///
/// let src = vec![128u8; 256];
/// let config = ClaheConfig::new(4, 4, 2.0);
/// let result = adaptive_histogram_equalization(&src, 16, 16, &config)?;
/// assert_eq!(result.len(), 256);
/// ```
pub fn adaptive_histogram_equalization(
    src: &[u8],
    width: u32,
    height: u32,
    config: &ClaheConfig,
) -> CvResult<Vec<u8>> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected_size = width as usize * height as usize;
    if src.len() < expected_size {
        return Err(CvError::insufficient_data(expected_size, src.len()));
    }

    if config.tiles_x == 0 || config.tiles_y == 0 {
        return Err(CvError::invalid_parameter(
            "tiles",
            "must be greater than 0",
        ));
    }

    let tile_width = width / config.tiles_x;
    let tile_height = height / config.tiles_y;

    if tile_width == 0 || tile_height == 0 {
        // Fall back to regular histogram equalization for small images
        return histogram_equalization(src, width, height);
    }

    // Compute lookup tables for each tile
    let mut tile_luts: Vec<[u8; 256]> = Vec::new();

    for ty in 0..config.tiles_y {
        for tx in 0..config.tiles_x {
            let start_x = tx * tile_width;
            let start_y = ty * tile_height;

            // Handle edge tiles
            let tw = if tx == config.tiles_x - 1 {
                width - start_x
            } else {
                tile_width
            };
            let th = if ty == config.tiles_y - 1 {
                height - start_y
            } else {
                tile_height
            };

            let hist = compute_region_histogram(src, width, start_x, start_y, tw, th);
            let lut = create_clahe_lut(&hist, config.clip_limit);
            tile_luts.push(lut);
        }
    }

    // Apply with bilinear interpolation between tiles
    let mut dst = vec![0u8; expected_size];

    for y in 0..height {
        for x in 0..width {
            let pixel = src[(y * width + x) as usize];

            // Find which tile(s) this pixel belongs to
            let fx = (x as f64 / tile_width as f64 - 0.5).max(0.0);
            let fy = (y as f64 / tile_height as f64 - 0.5).max(0.0);

            let tx0 = (fx.floor() as u32).min(config.tiles_x - 1);
            let ty0 = (fy.floor() as u32).min(config.tiles_y - 1);
            let tx1 = (tx0 + 1).min(config.tiles_x - 1);
            let ty1 = (ty0 + 1).min(config.tiles_y - 1);

            let x_weight = fx - fx.floor();
            let y_weight = fy - fy.floor();

            // Get LUT indices
            let idx00 = (ty0 * config.tiles_x + tx0) as usize;
            let idx10 = (ty0 * config.tiles_x + tx1) as usize;
            let idx01 = (ty1 * config.tiles_x + tx0) as usize;
            let idx11 = (ty1 * config.tiles_x + tx1) as usize;

            // Bilinear interpolation
            let v00 = tile_luts[idx00][pixel as usize] as f64;
            let v10 = tile_luts[idx10][pixel as usize] as f64;
            let v01 = tile_luts[idx01][pixel as usize] as f64;
            let v11 = tile_luts[idx11][pixel as usize] as f64;

            let top = v00 * (1.0 - x_weight) + v10 * x_weight;
            let bottom = v01 * (1.0 - x_weight) + v11 * x_weight;
            let value = top * (1.0 - y_weight) + bottom * y_weight;

            dst[(y * width + x) as usize] = value.round().clamp(0.0, 255.0) as u8;
        }
    }

    Ok(dst)
}

/// Create a CLAHE lookup table for a histogram.
fn create_clahe_lut(hist: &Histogram, clip_limit: f64) -> [u8; 256] {
    let mut clipped_hist = *hist.bins();
    let total = hist.total();

    if total == 0 {
        let mut lut = [0u8; 256];
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as u8;
        }
        return lut;
    }

    // Calculate actual clip limit
    let clip_threshold = ((clip_limit * total as f64) / 256.0).round() as u32;

    if clip_threshold > 0 {
        // Clip histogram and redistribute excess
        let mut excess = 0u32;

        for bin in &mut clipped_hist {
            if *bin > clip_threshold {
                excess += *bin - clip_threshold;
                *bin = clip_threshold;
            }
        }

        // Redistribute excess equally
        let redistribution = excess / 256;
        let remainder = (excess % 256) as usize;

        for (i, bin) in clipped_hist.iter_mut().enumerate() {
            *bin += redistribution;
            if i < remainder {
                *bin += 1;
            }
        }
    }

    // Create CDF and lookup table
    let mut cdf = [0u32; 256];
    let mut cumsum = 0u32;

    for (i, &count) in clipped_hist.iter().enumerate() {
        cumsum += count;
        cdf[i] = cumsum;
    }

    let cdf_min = cdf.iter().copied().find(|&c| c > 0).unwrap_or(0);
    let cdf_max = cumsum;

    let mut lut = [0u8; 256];

    if cdf_max > cdf_min {
        for (i, &c) in cdf.iter().enumerate() {
            let value = ((c - cdf_min) as f64 * 255.0 / (cdf_max - cdf_min) as f64)
                .round()
                .clamp(0.0, 255.0);
            lut[i] = value as u8;
        }
    } else {
        for (i, v) in lut.iter_mut().enumerate() {
            *v = i as u8;
        }
    }

    lut
}

/// Compute the histogram intersection (similarity measure).
///
/// Returns a value between 0.0 (completely different) and 1.0 (identical).
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::Histogram;
/// use oximedia_cv::image::histogram::histogram_intersection;
///
/// let h1 = Histogram::compute(&[100u8, 100, 100]);
/// let h2 = Histogram::compute(&[100u8, 100, 100]);
/// assert!((histogram_intersection(&h1, &h2) - 1.0).abs() < 0.001);
/// ```
#[must_use]
pub fn histogram_intersection(h1: &Histogram, h2: &Histogram) -> f64 {
    if h1.total() == 0 || h2.total() == 0 {
        return 0.0;
    }

    let n1 = h1.normalized();
    let n2 = h2.normalized();

    let mut intersection = 0.0;
    for i in 0..256 {
        intersection += n1[i].min(n2[i]);
    }

    intersection
}

/// Compute the chi-squared distance between histograms.
///
/// Lower values indicate more similar histograms.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::Histogram;
/// use oximedia_cv::image::histogram::histogram_chi_squared;
///
/// let h1 = Histogram::compute(&[100u8, 100, 100]);
/// let h2 = Histogram::compute(&[100u8, 100, 100]);
/// assert!(histogram_chi_squared(&h1, &h2) < 0.001);
/// ```
#[must_use]
pub fn histogram_chi_squared(h1: &Histogram, h2: &Histogram) -> f64 {
    if h1.total() == 0 || h2.total() == 0 {
        return f64::INFINITY;
    }

    let n1 = h1.normalized();
    let n2 = h2.normalized();

    let mut chi_sq = 0.0;
    for i in 0..256 {
        let sum = n1[i] + n2[i];
        if sum > f64::EPSILON {
            let diff = n1[i] - n2[i];
            chi_sq += diff * diff / sum;
        }
    }

    chi_sq / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_histogram_new() {
        let hist = Histogram::new();
        assert_eq!(hist.total(), 0);
        assert_eq!(hist.bin(0), 0);
    }

    #[test]
    fn test_histogram_compute() {
        let data = vec![0u8, 0, 128, 255, 255, 255];
        let hist = Histogram::compute(&data);

        assert_eq!(hist.total(), 6);
        assert_eq!(hist.bin(0), 2);
        assert_eq!(hist.bin(128), 1);
        assert_eq!(hist.bin(255), 3);
    }

    #[test]
    fn test_histogram_min_max() {
        let hist = Histogram::compute(&[50u8, 100, 150]);
        assert_eq!(hist.min_bin(), Some(50));
        assert_eq!(hist.max_bin(), Some(150));
    }

    #[test]
    fn test_histogram_mean() {
        let hist = Histogram::compute(&[0u8, 100, 200]);
        assert!((hist.mean() - 100.0).abs() < 0.001);
    }

    #[test]
    fn test_histogram_std_dev() {
        let hist = Histogram::compute(&[100u8, 100, 100]);
        assert!(hist.std_dev() < 0.001);

        let hist2 = Histogram::compute(&[0u8, 100, 200]);
        assert!(hist2.std_dev() > 0.0);
    }

    #[test]
    fn test_histogram_cdf() {
        let hist = Histogram::compute(&[0u8, 0, 255, 255]);
        let cdf = hist.cdf();

        assert!((cdf[0] - 0.5).abs() < 0.001);
        assert!((cdf[255] - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_histogram_equalization() {
        let src: Vec<u8> = (0..256).map(|i| (i / 4) as u8).collect();
        let result =
            histogram_equalization(&src, 16, 16).expect("histogram_equalization should succeed");
        assert_eq!(result.len(), 256);

        // After equalization, the histogram should be more uniform
        let hist = Histogram::compute(&result);
        assert!(hist.std_dev() < 100.0);
    }

    #[test]
    fn test_histogram_equalization_uniform() {
        let src = vec![128u8; 100];
        let result =
            histogram_equalization(&src, 10, 10).expect("histogram_equalization should succeed");

        // Uniform input should remain relatively uniform
        assert_eq!(result.len(), 100);
    }

    #[test]
    fn test_clahe() {
        let src = vec![128u8; 256];
        let config = ClaheConfig::new(4, 4, 2.0);
        let result = adaptive_histogram_equalization(&src, 16, 16, &config)
            .expect("adaptive_histogram_equalization should succeed");
        assert_eq!(result.len(), 256);
    }

    #[test]
    fn test_histogram_intersection() {
        let h1 = Histogram::compute(&[100u8; 100]);
        let h2 = Histogram::compute(&[100u8; 100]);
        assert!((histogram_intersection(&h1, &h2) - 1.0).abs() < 0.001);

        let h3 = Histogram::compute(&[200u8; 100]);
        assert!(histogram_intersection(&h1, &h3) < 0.001);
    }

    #[test]
    fn test_histogram_chi_squared() {
        let h1 = Histogram::compute(&[100u8; 100]);
        let h2 = Histogram::compute(&[100u8; 100]);
        assert!(histogram_chi_squared(&h1, &h2) < 0.001);
    }

    #[test]
    fn test_invalid_dimensions() {
        let src = vec![0u8; 100];
        assert!(histogram_equalization(&src, 0, 10).is_err());
        assert!(histogram_equalization(&src, 10, 0).is_err());
    }

    #[test]
    fn test_region_histogram() {
        let src = vec![
            0, 0, 255, 255, 0, 0, 255, 255, 128, 128, 128, 128, 128, 128, 128, 128,
        ];
        let hist = compute_region_histogram(&src, 4, 0, 0, 2, 2);
        assert_eq!(hist.bin(0), 4);
        assert_eq!(hist.total(), 4);
    }
}
