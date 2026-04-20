//! Statistical analysis utilities for video scope data.

#![allow(dead_code)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::must_use_candidate)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::manual_is_multiple_of)]
#![allow(clippy::manual_midpoint)]

/// Per-frame statistical summary of scope values.
#[derive(Debug, Clone, PartialEq)]
pub struct FrameStats {
    /// Minimum value in the frame.
    pub min: f64,
    /// Maximum value in the frame.
    pub max: f64,
    /// Arithmetic mean.
    pub mean: f64,
    /// Population standard deviation.
    pub std_dev: f64,
    /// 1st-percentile value.
    pub percentile_1: f64,
    /// 99th-percentile value.
    pub percentile_99: f64,
}

/// Compute `FrameStats` for a slice of `f64` values.
///
/// Returns a zeroed `FrameStats` for an empty slice.
pub fn compute_stats(values: &[f64]) -> FrameStats {
    if values.is_empty() {
        return FrameStats {
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std_dev: 0.0,
            percentile_1: 0.0,
            percentile_99: 0.0,
        };
    }

    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let min = sorted[0];
    // sorted is non-empty: the empty case returned early above, so index is valid.
    let max = sorted[sorted.len() - 1];
    let mean = sorted.iter().sum::<f64>() / sorted.len() as f64;
    let variance = sorted.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / sorted.len() as f64;
    let std_dev = variance.sqrt();
    let p1 = percentile(&sorted, 1.0);
    let p99 = percentile(&sorted, 99.0);

    FrameStats {
        min,
        max,
        mean,
        std_dev,
        percentile_1: p1,
        percentile_99: p99,
    }
}

/// Return the `p`-th percentile (0–100) of a **pre-sorted** slice.
///
/// Uses linear interpolation between adjacent elements.
/// Returns 0.0 for an empty slice.
pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let p_clamped = p.clamp(0.0, 100.0);
    let idx = p_clamped / 100.0 * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi {
        return sorted[lo];
    }
    let frac = idx - lo as f64;
    sorted[lo] + frac * (sorted[hi] - sorted[lo])
}

/// Rolling window of `FrameStats` for temporal analysis.
#[derive(Debug, Clone)]
pub struct TemporalStats {
    /// Ordered collection of frame statistics within the window.
    pub frames: Vec<FrameStats>,
    /// Maximum number of frames to retain.
    pub window: usize,
}

impl TemporalStats {
    /// Create a new `TemporalStats` with the given rolling window size.
    pub fn new(window: usize) -> Self {
        Self {
            frames: Vec::with_capacity(window),
            window: window.max(1),
        }
    }

    /// Add a new `FrameStats` entry, evicting the oldest if the window is full.
    pub fn add(&mut self, stats: FrameStats) {
        if self.frames.len() == self.window {
            self.frames.remove(0);
        }
        self.frames.push(stats);
    }

    /// Compute the rolling mean of the `mean` field across all retained frames.
    pub fn rolling_mean(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        self.frames.iter().map(|f| f.mean).sum::<f64>() / self.frames.len() as f64
    }

    /// Compute the population standard deviation of the `mean` field.
    pub fn rolling_std_dev(&self) -> f64 {
        if self.frames.is_empty() {
            return 0.0;
        }
        let mean = self.rolling_mean();
        let variance = self
            .frames
            .iter()
            .map(|f| (f.mean - mean).powi(2))
            .sum::<f64>()
            / self.frames.len() as f64;
        variance.sqrt()
    }

    /// Estimate the linear trend of the `mean` field using simple linear regression slope.
    ///
    /// Returns 0.0 when fewer than two frames are present.
    pub fn trend(&self) -> f64 {
        let n = self.frames.len();
        if n < 2 {
            return 0.0;
        }
        let x_mean = (n as f64 - 1.0) / 2.0;
        let y_mean = self.rolling_mean();
        let mut numerator = 0.0_f64;
        let mut denominator = 0.0_f64;
        for (i, frame) in self.frames.iter().enumerate() {
            let dx = i as f64 - x_mean;
            numerator += dx * (frame.mean - y_mean);
            denominator += dx * dx;
        }
        if denominator == 0.0 {
            0.0
        } else {
            numerator / denominator
        }
    }
}

// ---------------------------------------------------------------------------
// ImageStats
// ---------------------------------------------------------------------------

/// Statistical summary of a single-channel image or pixel array.
#[derive(Debug, Clone)]
pub struct ImageStats {
    /// Arithmetic mean of pixel values.
    pub mean: f32,
    /// Population standard deviation.
    pub std_dev: f32,
    /// Minimum pixel value.
    pub min: f32,
    /// Maximum pixel value.
    pub max: f32,
    /// Median pixel value.
    pub median: f32,
}

impl ImageStats {
    /// Returns the dynamic range (`max − min`).
    #[must_use]
    pub fn dynamic_range(&self) -> f32 {
        self.max - self.min
    }
}

/// Computes `ImageStats` for a slice of pixel values.
///
/// Uses a sorted copy for the median and a single pass for mean and variance.
/// Returns zeroed stats for an empty slice.
#[must_use]
pub fn compute_image_stats(pixels: &[f32]) -> ImageStats {
    if pixels.is_empty() {
        return ImageStats {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
        };
    }
    // Single pass for mean, min, max.
    let mut min = pixels[0];
    let mut max = pixels[0];
    let mut sum = 0.0_f32;
    for &p in pixels {
        if p < min {
            min = p;
        }
        if p > max {
            max = p;
        }
        sum += p;
    }
    let mean = sum / pixels.len() as f32;

    // Variance.
    let variance: f32 =
        pixels.iter().map(|&p| (p - mean) * (p - mean)).sum::<f32>() / pixels.len() as f32;
    let std_dev = variance.sqrt();

    // Median via sorted copy.
    let mut sorted = pixels.to_vec();
    sorted.sort_unstable_by(f32::total_cmp);
    let n = sorted.len();
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };

    ImageStats {
        mean,
        std_dev,
        min,
        max,
        median,
    }
}

// ---------------------------------------------------------------------------
// Histogram
// ---------------------------------------------------------------------------

/// A fixed-bin histogram for pixel/sample data.
#[derive(Debug, Clone)]
pub struct Histogram {
    /// Bin counts.
    pub bins: Vec<u32>,
    /// Lower bound of the value range.
    pub min_val: f32,
    /// Upper bound of the value range.
    pub max_val: f32,
}

impl Histogram {
    /// Creates a new `Histogram` with `num_bins` equally wide bins over [`min`, `max`].
    #[must_use]
    pub fn new(num_bins: usize, min: f32, max: f32) -> Self {
        Self {
            bins: vec![0; num_bins.max(1)],
            min_val: min,
            max_val: max,
        }
    }

    /// Adds `value` to the appropriate bin (values outside range are clamped to edge bins).
    pub fn add(&mut self, value: f32) {
        let range = self.max_val - self.min_val;
        let n = self.bins.len() as f32;
        if range <= 0.0 {
            self.bins[0] = self.bins[0].saturating_add(1);
            return;
        }
        let t = ((value - self.min_val) / range * n).floor() as isize;
        let idx = t.clamp(0, (self.bins.len() as isize) - 1) as usize;
        self.bins[idx] = self.bins[idx].saturating_add(1);
    }

    /// Returns the `p`-th percentile (0–100) value via linear interpolation over bins.
    ///
    /// Returns `min_val` for empty histograms.
    #[must_use]
    pub fn percentile(&self, p: f32) -> f32 {
        let total = self.total_count();
        if total == 0 {
            return self.min_val;
        }
        let target = (p.clamp(0.0, 100.0) / 100.0 * total as f32).ceil() as u64;
        let mut cumulative: u64 = 0;
        let n = self.bins.len();
        for (i, &count) in self.bins.iter().enumerate() {
            cumulative += u64::from(count);
            if cumulative >= target {
                let range = self.max_val - self.min_val;
                return self.min_val + (i as f32 + 0.5) / n as f32 * range;
            }
        }
        self.max_val
    }

    /// Returns the lower edge of the most-populated bin.
    #[must_use]
    pub fn mode(&self) -> f32 {
        let (idx, _) = self
            .bins
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .unwrap_or((0, &0));
        let n = self.bins.len() as f32;
        let range = self.max_val - self.min_val;
        self.min_val + idx as f32 / n * range
    }

    /// Returns the total number of values added to all bins.
    #[must_use]
    pub fn total_count(&self) -> u64 {
        self.bins.iter().map(|&c| u64::from(c)).sum()
    }
}

// ---------------------------------------------------------------------------
// ChannelStats
// ---------------------------------------------------------------------------

/// Per-channel RGB image statistics.
#[derive(Debug, Clone)]
pub struct ChannelStats {
    /// Red channel stats.
    pub r: ImageStats,
    /// Green channel stats.
    pub g: ImageStats,
    /// Blue channel stats.
    pub b: ImageStats,
}

impl ChannelStats {
    /// Computes per-channel stats from a slice of `[r, g, b]` pixel triplets.
    #[must_use]
    pub fn from_rgb_pixels(pixels: &[[f32; 3]]) -> Self {
        let r: Vec<f32> = pixels.iter().map(|p| p[0]).collect();
        let g: Vec<f32> = pixels.iter().map(|p| p[1]).collect();
        let b: Vec<f32> = pixels.iter().map(|p| p[2]).collect();
        Self {
            r: compute_image_stats(&r),
            g: compute_image_stats(&g),
            b: compute_image_stats(&b),
        }
    }

    /// Returns `"r"`, `"g"`, or `"b"` — the channel with the highest mean value.
    #[must_use]
    pub fn most_saturated(&self) -> &str {
        if self.r.mean >= self.g.mean && self.r.mean >= self.b.mean {
            "r"
        } else if self.g.mean >= self.b.mean {
            "g"
        } else {
            "b"
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_stats_empty() {
        let s = compute_stats(&[]);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 0.0);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.std_dev, 0.0);
    }

    #[test]
    fn test_compute_stats_single() {
        let s = compute_stats(&[42.0]);
        assert!((s.min - 42.0).abs() < 1e-10);
        assert!((s.max - 42.0).abs() < 1e-10);
        assert!((s.mean - 42.0).abs() < 1e-10);
        assert!(s.std_dev.abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_mean() {
        let s = compute_stats(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((s.mean - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_std_dev() {
        // Constant values → std_dev = 0.
        let s = compute_stats(&[5.0, 5.0, 5.0, 5.0]);
        assert!(s.std_dev.abs() < 1e-10);
    }

    #[test]
    fn test_compute_stats_min_max() {
        let s = compute_stats(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0]);
        assert!((s.min - 1.0).abs() < 1e-10);
        assert!((s.max - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_empty() {
        assert_eq!(percentile(&[], 50.0), 0.0);
    }

    #[test]
    fn test_percentile_endpoints() {
        let sorted = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        assert!((percentile(&sorted, 0.0) - 0.0).abs() < 1e-10);
        assert!((percentile(&sorted, 100.0) - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile_median() {
        let sorted = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&sorted, 50.0) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_stats_rolling_mean() {
        let mut ts = TemporalStats::new(3);
        ts.add(compute_stats(&[1.0]));
        ts.add(compute_stats(&[3.0]));
        ts.add(compute_stats(&[5.0]));
        assert!((ts.rolling_mean() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_stats_window_eviction() {
        let mut ts = TemporalStats::new(2);
        ts.add(compute_stats(&[1.0]));
        ts.add(compute_stats(&[2.0]));
        ts.add(compute_stats(&[10.0])); // evicts the first
        assert_eq!(ts.frames.len(), 2);
        assert!((ts.frames[0].mean - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_temporal_stats_rolling_std_dev_constant() {
        let mut ts = TemporalStats::new(5);
        for _ in 0..5 {
            ts.add(compute_stats(&[4.0]));
        }
        assert!(ts.rolling_std_dev().abs() < 1e-10);
    }

    #[test]
    fn test_temporal_stats_trend_increasing() {
        let mut ts = TemporalStats::new(5);
        for v in [1.0, 2.0, 3.0, 4.0, 5.0_f64] {
            ts.add(compute_stats(&[v]));
        }
        assert!(ts.trend() > 0.0);
    }

    #[test]
    fn test_temporal_stats_trend_decreasing() {
        let mut ts = TemporalStats::new(5);
        for v in [5.0, 4.0, 3.0, 2.0, 1.0_f64] {
            ts.add(compute_stats(&[v]));
        }
        assert!(ts.trend() < 0.0);
    }

    #[test]
    fn test_temporal_stats_empty_rolling_mean() {
        let ts = TemporalStats::new(5);
        assert_eq!(ts.rolling_mean(), 0.0);
    }

    // --- ImageStats tests ---

    #[test]
    fn test_image_stats_empty() {
        let s = compute_image_stats(&[]);
        assert_eq!(s.mean, 0.0);
        assert_eq!(s.std_dev, 0.0);
        assert_eq!(s.min, 0.0);
        assert_eq!(s.max, 0.0);
    }

    #[test]
    fn test_image_stats_single() {
        let s = compute_image_stats(&[0.5]);
        assert!((s.mean - 0.5).abs() < 1e-6);
        assert!((s.median - 0.5).abs() < 1e-6);
        assert!(s.std_dev.abs() < 1e-6);
    }

    #[test]
    fn test_image_stats_mean() {
        let s = compute_image_stats(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((s.mean - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_image_stats_min_max() {
        let s = compute_image_stats(&[0.1, 0.9, 0.5]);
        assert!((s.min - 0.1).abs() < 1e-6);
        assert!((s.max - 0.9).abs() < 1e-6);
    }

    #[test]
    fn test_image_stats_dynamic_range() {
        let s = compute_image_stats(&[0.0, 0.5, 1.0]);
        assert!((s.dynamic_range() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_image_stats_median_odd() {
        let s = compute_image_stats(&[3.0, 1.0, 2.0]);
        assert!((s.median - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_image_stats_median_even() {
        let s = compute_image_stats(&[1.0, 2.0, 3.0, 4.0]);
        assert!((s.median - 2.5).abs() < 1e-5);
    }

    // --- Histogram tests ---

    #[test]
    fn test_histogram_total_count() {
        let mut h = Histogram::new(10, 0.0, 1.0);
        for v in [0.1, 0.5, 0.9] {
            h.add(v);
        }
        assert_eq!(h.total_count(), 3);
    }

    #[test]
    fn test_histogram_add_clamped() {
        let mut h = Histogram::new(4, 0.0, 1.0);
        h.add(-1.0); // below min → bin 0
        h.add(2.0); // above max → last bin
        assert_eq!(h.total_count(), 2);
    }

    #[test]
    fn test_histogram_mode() {
        let mut h = Histogram::new(4, 0.0, 1.0);
        // Heavily populate bin 1 (0.25–0.5)
        for _ in 0..10 {
            h.add(0.3);
        }
        h.add(0.8);
        let mode = h.mode();
        // Mode should be near 0.25 (start of bin 1)
        assert!(mode < 0.5);
    }

    #[test]
    fn test_histogram_percentile_empty() {
        let h = Histogram::new(10, 0.0, 1.0);
        assert_eq!(h.percentile(50.0), 0.0);
    }

    #[test]
    fn test_histogram_percentile_100() {
        let mut h = Histogram::new(10, 0.0, 1.0);
        for v in [0.1f32, 0.5, 0.9] {
            h.add(v);
        }
        let p = h.percentile(100.0);
        assert!(p <= 1.0);
    }

    // --- ChannelStats tests ---

    #[test]
    fn test_channel_stats_from_rgb_pixels() {
        let pixels: Vec<[f32; 3]> = vec![[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]];
        let cs = ChannelStats::from_rgb_pixels(&pixels);
        assert!((cs.r.mean - 1.0).abs() < 1e-5);
        assert!((cs.g.mean - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_channel_stats_most_saturated_red() {
        let pixels: Vec<[f32; 3]> = vec![[0.8, 0.1, 0.1]];
        let cs = ChannelStats::from_rgb_pixels(&pixels);
        assert_eq!(cs.most_saturated(), "r");
    }

    #[test]
    fn test_channel_stats_most_saturated_blue() {
        let pixels: Vec<[f32; 3]> = vec![[0.1, 0.2, 0.9]];
        let cs = ChannelStats::from_rgb_pixels(&pixels);
        assert_eq!(cs.most_saturated(), "b");
    }
}
