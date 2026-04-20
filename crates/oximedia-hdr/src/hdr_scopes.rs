//! Waveform and vectorscope analysis for HDR content.
//!
//! Provides signal-analysis tools analogous to broadcast scopes but extended
//! for high-dynamic-range content:
//!
//! - **Waveform monitor**: per-column or per-row luminance plot showing the
//!   distribution of signal levels across the spatial extent of the frame.
//! - **Vectorscope**: plots chroma (Cb/Cr or ICtCp Ct/Cp) position for each
//!   pixel, revealing colour gamut usage and saturation.
//! - **Luma statistics**: per-frame minimum, maximum, mean, and percentile
//!   luminance for HDR quality control.

use crate::{HdrError, Result};

// ── WaveformMode ─────────────────────────────────────────────────────────────

/// Axis along which waveform columns are gathered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WaveformAxis {
    /// One waveform column per horizontal pixel column.
    Horizontal,
    /// One waveform column per horizontal pixel row.
    Vertical,
}

// ── LumaStatistics ───────────────────────────────────────────────────────────

/// Per-frame luminance statistics for HDR quality control.
#[derive(Debug, Clone)]
pub struct LumaStatistics {
    /// Minimum luminance across the frame (normalised [0, 1]).
    pub min: f32,
    /// Maximum luminance across the frame (normalised [0, 1]).
    pub max: f32,
    /// Arithmetic mean luminance.
    pub mean: f32,
    /// Median luminance (50th percentile).
    pub median: f32,
    /// 95th-percentile luminance (near-peak without specular hotspots).
    pub p95: f32,
    /// 99th-percentile luminance (almost-peak).
    pub p99: f32,
    /// Total number of pixels analysed.
    pub pixel_count: u64,
}

impl LumaStatistics {
    /// Compute luma statistics from an interleaved linear-light RGB frame.
    ///
    /// BT.2100 luma coefficients (Kr=0.2627, Kb=0.0593) are used.
    ///
    /// # Arguments
    /// - `pixels`: interleaved linear-light RGB values (length divisible by 3)
    ///
    /// # Errors
    /// Returns `HdrError::ToneMappingError` if `pixels.len() % 3 != 0` or the
    /// frame is empty.
    pub fn from_frame(pixels: &[f32]) -> Result<Self> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "pixel buffer length {} is not divisible by 3",
                pixels.len()
            )));
        }
        if pixels.is_empty() {
            return Err(HdrError::ToneMappingError("empty pixel buffer".to_string()));
        }

        // Compute per-pixel luma values.
        let mut luma_values: Vec<f32> = pixels
            .chunks_exact(3)
            .map(|c| 0.2627 * c[0] + 0.6780 * c[1] + 0.0593 * c[2])
            .collect();

        let n = luma_values.len() as f32;

        // Sort for percentile computation.
        luma_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = luma_values[0];
        // luma_values is non-empty: pixels non-empty guard is checked above.
        let max = luma_values[luma_values.len() - 1];
        let mean = luma_values.iter().copied().map(|v| v as f64).sum::<f64>() as f32 / n;

        let percentile = |p: f32| -> f32 {
            let idx = ((p / 100.0 * n) as usize).min(luma_values.len() - 1);
            luma_values[idx]
        };

        Ok(Self {
            min,
            max,
            mean,
            median: percentile(50.0),
            p95: percentile(95.0),
            p99: percentile(99.0),
            pixel_count: luma_values.len() as u64,
        })
    }
}

// ── WaveformMonitor ───────────────────────────────────────────────────────────

/// HDR waveform monitor: plots luminance distribution across spatial columns.
///
/// Each column in the waveform represents one spatial position (horizontal or
/// vertical), and contains a histogram of the luminance values at that position.
/// The resulting 2D array can be rendered as a false-colour scope image.
pub struct WaveformMonitor {
    /// Number of spatial columns (one per pixel row or column of the source frame).
    pub columns: usize,
    /// Number of luminance bins per column (vertical resolution of the scope).
    pub bins: usize,
    /// 2D histogram: `data[column * bins + bin_index] = pixel_count`.
    pub data: Vec<u64>,
    /// Spatial axis that defines each column.
    pub axis: WaveformAxis,
}

impl WaveformMonitor {
    /// Build a waveform from an interleaved linear-light RGB frame.
    ///
    /// # Arguments
    /// - `pixels`: interleaved RGB values (length = `width * height * 3`)
    /// - `width`: frame width in pixels
    /// - `height`: frame height in pixels
    /// - `axis`: axis along which to gather columns
    /// - `bins`: luminance resolution (number of bins in the Y axis)
    ///
    /// # Errors
    /// Returns `HdrError::ToneMappingError` if the buffer length does not match
    /// `width * height * 3`.
    pub fn compute(
        pixels: &[f32],
        width: usize,
        height: usize,
        axis: WaveformAxis,
        bins: usize,
    ) -> Result<Self> {
        let expected = width * height * 3;
        if pixels.len() != expected {
            return Err(HdrError::ToneMappingError(format!(
                "pixel buffer length {} does not match {}×{}×3 = {}",
                pixels.len(),
                width,
                height,
                expected
            )));
        }
        if bins == 0 {
            return Err(HdrError::ToneMappingError("bins must be > 0".to_string()));
        }

        let columns = match axis {
            WaveformAxis::Horizontal => width,
            WaveformAxis::Vertical => height,
        };

        let mut data = vec![0u64; columns * bins];

        for row in 0..height {
            for col in 0..width {
                let idx = (row * width + col) * 3;
                let r = pixels[idx];
                let g = pixels[idx + 1];
                let b = pixels[idx + 2];
                let luma = 0.2627 * r + 0.6780 * g + 0.0593 * b;
                let luma_clamped = luma.clamp(0.0, 1.0);

                let col_idx = match axis {
                    WaveformAxis::Horizontal => col,
                    WaveformAxis::Vertical => row,
                };

                let bin_idx = ((luma_clamped * bins as f32) as usize).min(bins - 1);
                data[col_idx * bins + bin_idx] += 1;
            }
        }

        Ok(Self {
            columns,
            bins,
            data,
            axis,
        })
    }

    /// Return the pixel count at a specific (column, bin) position.
    pub fn get(&self, column: usize, bin: usize) -> u64 {
        if column < self.columns && bin < self.bins {
            self.data[column * self.bins + bin]
        } else {
            0
        }
    }

    /// Return the peak pixel count across all (column, bin) cells.
    pub fn peak_count(&self) -> u64 {
        self.data.iter().copied().max().unwrap_or(0)
    }

    /// Return the luminance value at the centre of a given bin (normalised [0, 1]).
    pub fn bin_to_luma(&self, bin: usize) -> f32 {
        (bin as f32 + 0.5) / self.bins as f32
    }
}

// ── Vectorscope ──────────────────────────────────────────────────────────────

/// A 2D colour distribution plot in the Cb/Cr (YCbCr) chrominance plane.
///
/// Each pixel in the source frame contributes to the (Cb, Cr) bin that
/// corresponds to its chrominance, building up a density map.  The resulting
/// 2D histogram can be rendered as a false-colour scope image.
///
/// BT.2020 YCbCr coefficients are used:
///   Y  =  0.2627·R + 0.6780·G + 0.0593·B
///   Cb = (B - Y) / (2 · (1 - Kb))  with Kb = 0.0593
///   Cr = (R - Y) / (2 · (1 - Kr))  with Kr = 0.2627
pub struct Vectorscope {
    /// Number of bins in the Cb axis.
    pub bins_cb: usize,
    /// Number of bins in the Cr axis.
    pub bins_cr: usize,
    /// 2D histogram: `data[cb_bin * bins_cr + cr_bin] = pixel_count`.
    pub data: Vec<u64>,
}

impl Vectorscope {
    /// Build a vectorscope from an interleaved linear-light RGB frame.
    ///
    /// # Arguments
    /// - `pixels`: interleaved RGB values (length divisible by 3)
    /// - `bins_cb`: number of Cb bins (horizontal resolution of the scope)
    /// - `bins_cr`: number of Cr bins (vertical resolution of the scope)
    ///
    /// # Errors
    /// Returns `HdrError::ToneMappingError` if `pixels.len() % 3 != 0`.
    pub fn compute(pixels: &[f32], bins_cb: usize, bins_cr: usize) -> Result<Self> {
        if !pixels.len().is_multiple_of(3) {
            return Err(HdrError::ToneMappingError(format!(
                "pixel buffer length {} is not divisible by 3",
                pixels.len()
            )));
        }
        if bins_cb == 0 || bins_cr == 0 {
            return Err(HdrError::ToneMappingError("bins must be > 0".to_string()));
        }

        let mut data = vec![0u64; bins_cb * bins_cr];

        // BT.2020 YCbCr scaling factors (per ITU-R BT.2020)
        let kb = 0.0593_f32;
        let kr = 0.2627_f32;
        let cb_scale = 2.0 * (1.0 - kb);
        let cr_scale = 2.0 * (1.0 - kr);

        for chunk in pixels.chunks_exact(3) {
            let r = chunk[0];
            let g = chunk[1];
            let b = chunk[2];

            let y = kr * r + 0.6780 * g + kb * b;
            let cb = (b - y) / cb_scale; // range [-0.5, +0.5]
            let cr = (r - y) / cr_scale; // range [-0.5, +0.5]

            // Normalise to [0, 1]
            let cb_norm = (cb + 0.5).clamp(0.0, 1.0);
            let cr_norm = (cr + 0.5).clamp(0.0, 1.0);

            let cb_bin = ((cb_norm * bins_cb as f32) as usize).min(bins_cb - 1);
            let cr_bin = ((cr_norm * bins_cr as f32) as usize).min(bins_cr - 1);

            data[cb_bin * bins_cr + cr_bin] += 1;
        }

        Ok(Self {
            bins_cb,
            bins_cr,
            data,
        })
    }

    /// Return the pixel count at a specific (cb_bin, cr_bin) position.
    pub fn get(&self, cb_bin: usize, cr_bin: usize) -> u64 {
        if cb_bin < self.bins_cb && cr_bin < self.bins_cr {
            self.data[cb_bin * self.bins_cr + cr_bin]
        } else {
            0
        }
    }

    /// Return the peak count across all bins.
    pub fn peak_count(&self) -> u64 {
        self.data.iter().copied().max().unwrap_or(0)
    }

    /// Return the total number of pixels accumulated.
    pub fn total_pixels(&self) -> u64 {
        self.data.iter().sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, eps: f32) -> bool {
        (a - b).abs() < eps
    }

    // ── LumaStatistics tests ────────────────────────────────────────────────

    // 1. Black frame: min=max=mean=0
    #[test]
    fn test_luma_stats_black_frame() {
        let pixels = vec![0.0f32; 300]; // 100 black pixels
        let stats = LumaStatistics::from_frame(&pixels).expect("black frame");
        assert!(approx(stats.min, 0.0, 1e-6));
        assert!(approx(stats.max, 0.0, 1e-6));
        assert!(approx(stats.mean, 0.0, 1e-6));
        assert_eq!(stats.pixel_count, 100);
    }

    // 2. White frame: min=max=mean≈1
    #[test]
    fn test_luma_stats_white_frame() {
        let pixels = vec![1.0f32; 300];
        let stats = LumaStatistics::from_frame(&pixels).expect("white frame");
        assert!(approx(stats.min, 1.0, 1e-4));
        assert!(approx(stats.max, 1.0, 1e-4));
    }

    // 3. Mixed frame: min < max
    #[test]
    fn test_luma_stats_min_max_order() {
        let mut pixels = vec![0.0f32; 300];
        // Make one bright pixel
        pixels[0] = 1.0;
        pixels[1] = 1.0;
        pixels[2] = 1.0;
        let stats = LumaStatistics::from_frame(&pixels).expect("mixed");
        assert!(stats.max > stats.min, "max should exceed min");
    }

    // 4. Empty buffer: error
    #[test]
    fn test_luma_stats_empty_error() {
        assert!(LumaStatistics::from_frame(&[]).is_err());
    }

    // 5. Invalid length: error
    #[test]
    fn test_luma_stats_invalid_length() {
        assert!(LumaStatistics::from_frame(&[0.5f32; 5]).is_err());
    }

    // 6. p95 ≥ median
    #[test]
    fn test_luma_stats_percentile_order() {
        let pixels: Vec<f32> = (0..300)
            .map(|i| {
                if i % 3 == 0 {
                    (i / 3) as f32 / 100.0
                } else {
                    0.0
                }
            })
            .collect();
        let stats = LumaStatistics::from_frame(&pixels).expect("stats");
        assert!(stats.p95 >= stats.median, "p95 should be >= median");
        assert!(stats.p99 >= stats.p95, "p99 should be >= p95");
    }

    // ── WaveformMonitor tests ────────────────────────────────────────────────

    // 7. Waveform column count matches width (horizontal axis)
    #[test]
    fn test_waveform_horizontal_columns() {
        let width = 8;
        let height = 4;
        let pixels = vec![0.5f32; width * height * 3];
        let wf = WaveformMonitor::compute(&pixels, width, height, WaveformAxis::Horizontal, 64)
            .expect("waveform");
        assert_eq!(wf.columns, width);
    }

    // 8. Waveform column count matches height (vertical axis)
    #[test]
    fn test_waveform_vertical_columns() {
        let width = 8;
        let height = 4;
        let pixels = vec![0.5f32; width * height * 3];
        let wf = WaveformMonitor::compute(&pixels, width, height, WaveformAxis::Vertical, 64)
            .expect("waveform");
        assert_eq!(wf.columns, height);
    }

    // 9. Waveform total count equals pixel count
    #[test]
    fn test_waveform_total_count() {
        let width = 8;
        let height = 4;
        let pixels = vec![0.5f32; width * height * 3];
        let wf = WaveformMonitor::compute(&pixels, width, height, WaveformAxis::Horizontal, 64)
            .expect("waveform");
        let total: u64 = wf.data.iter().sum();
        assert_eq!(total, (width * height) as u64);
    }

    // 10. Waveform invalid buffer length: error
    #[test]
    fn test_waveform_invalid_length() {
        let result = WaveformMonitor::compute(&[0.5f32; 10], 4, 4, WaveformAxis::Horizontal, 64);
        assert!(result.is_err());
    }

    // 11. Waveform: black frame concentrates in bin 0
    #[test]
    fn test_waveform_black_frame_bin0() {
        let width = 4;
        let height = 4;
        let pixels = vec![0.0f32; width * height * 3];
        let wf = WaveformMonitor::compute(&pixels, width, height, WaveformAxis::Horizontal, 64)
            .expect("black");
        // All pixels should map to bin 0
        for col in 0..width {
            assert_eq!(wf.get(col, 0), height as u64, "col {col} bin 0");
        }
    }

    // 12. Waveform get() returns 0 for out-of-range indices
    #[test]
    fn test_waveform_get_oob() {
        let pixels = vec![0.5f32; 4 * 4 * 3];
        let wf = WaveformMonitor::compute(&pixels, 4, 4, WaveformAxis::Horizontal, 64).expect("wf");
        assert_eq!(wf.get(999, 999), 0);
    }

    // 13. bin_to_luma returns value in [0, 1]
    #[test]
    fn test_waveform_bin_to_luma() {
        let pixels = vec![0.5f32; 4 * 4 * 3];
        let wf = WaveformMonitor::compute(&pixels, 4, 4, WaveformAxis::Horizontal, 64).expect("wf");
        for bin in 0..64 {
            let v = wf.bin_to_luma(bin);
            assert!((0.0..=1.0).contains(&v), "bin_to_luma({bin}) = {v}");
        }
    }

    // ── Vectorscope tests ────────────────────────────────────────────────────

    // 14. Grey frame concentrates in the centre of the Cb/Cr plane
    #[test]
    fn test_vectorscope_grey_centred() {
        let pixels = vec![0.5f32; 300]; // 100 grey pixels
        let vs = Vectorscope::compute(&pixels, 64, 64).expect("vectorscope");
        let total = vs.total_pixels();
        assert_eq!(total, 100);
        // The centre bin should have all pixels
        let mid_cb = 32;
        let mid_cr = 32;
        let count = vs.get(mid_cb, mid_cr);
        assert!(count > 0, "grey should be near centre: count={count}");
    }

    // 15. Vectorscope total pixel count matches input
    #[test]
    fn test_vectorscope_total_count() {
        let pixels = vec![0.3f32, 0.5, 0.7, 0.8, 0.2, 0.4];
        let vs = Vectorscope::compute(&pixels, 64, 64).expect("vs");
        assert_eq!(vs.total_pixels(), 2);
    }

    // 16. Vectorscope invalid buffer length: error
    #[test]
    fn test_vectorscope_invalid_length() {
        assert!(Vectorscope::compute(&[0.5f32; 5], 64, 64).is_err());
    }

    // 17. Vectorscope zero bins: error
    #[test]
    fn test_vectorscope_zero_bins() {
        let pixels = vec![0.5f32; 9];
        assert!(Vectorscope::compute(&pixels, 0, 64).is_err());
        assert!(Vectorscope::compute(&pixels, 64, 0).is_err());
    }

    // 18. get() returns 0 out of range
    #[test]
    fn test_vectorscope_get_oob() {
        let pixels = vec![0.5f32; 9];
        let vs = Vectorscope::compute(&pixels, 8, 8).expect("vs");
        assert_eq!(vs.get(999, 999), 0);
    }

    // 19. Peak count is >= any individual bin
    #[test]
    fn test_vectorscope_peak_count() {
        let pixels = vec![0.5f32; 300];
        let vs = Vectorscope::compute(&pixels, 64, 64).expect("vs");
        for cb in 0..64 {
            for cr in 0..64 {
                assert!(vs.peak_count() >= vs.get(cb, cr));
            }
        }
    }

    // 20. Red-dominant pixel produces different Cr from grey
    #[test]
    fn test_vectorscope_red_vs_grey_cr() {
        // Red-dominant
        let red_pixels = vec![1.0f32, 0.0, 0.0];
        let grey_pixels = vec![0.33f32, 0.33, 0.33];

        let vs_red = Vectorscope::compute(&red_pixels, 64, 64).expect("red");
        let vs_grey = Vectorscope::compute(&grey_pixels, 64, 64).expect("grey");

        // Find the bin with peak count for each
        let peak_red = vs_red.data.iter().position(|&v| v == vs_red.peak_count());
        let peak_grey = vs_grey.data.iter().position(|&v| v == vs_grey.peak_count());

        assert_ne!(
            peak_red, peak_grey,
            "Red and grey should occupy different vectorscope positions"
        );
    }
}
