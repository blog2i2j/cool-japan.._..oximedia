//! Luma parade scope showing per-channel luminance distribution.
//!
//! The luma parade renders each requested channel (Red, Green, Blue, or
//! composite Luma) as a side-by-side waveform histogram, giving colorists a
//! fast visual reference for per-channel exposure, clipping, and balance.
//!
//! # Channel Layout
//!
//! Channels are laid out left-to-right in the order they appear in
//! [`LumaParadeConfig::channels`].  A thin black gutter separates each pane.
//!
//! # Colour Coding
//!
//! | Channel | Colour |
//! |---------|--------|
//! | Red     | `#FF4040` |
//! | Green   | `#40FF40` |
//! | Blue    | `#4040FF` |
//! | Luma    | `#E0E0E0` |

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

/// A single channel displayed in the luma parade.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParadeChannel {
    /// Red channel (index 0 in RGB24 input).
    Red,
    /// Green channel (index 1 in RGB24 input).
    Green,
    /// Blue channel (index 2 in RGB24 input).
    Blue,
    /// BT.709 composite luminance: 0.2126·R + 0.7152·G + 0.0722·B.
    Luma,
}

impl ParadeChannel {
    /// Returns the RGBA colour used to render this channel's waveform.
    #[must_use]
    pub fn rgba_color(self) -> [u8; 4] {
        match self {
            Self::Red => [255, 64, 64, 220],
            Self::Green => [64, 255, 64, 220],
            Self::Blue => [64, 64, 255, 220],
            Self::Luma => [224, 224, 224, 200],
        }
    }

    /// Extract the scalar value [0, 255] for this channel from a single RGB pixel.
    #[must_use]
    fn extract(self, r: u8, g: u8, b: u8) -> u8 {
        match self {
            Self::Red => r,
            Self::Green => g,
            Self::Blue => b,
            Self::Luma => {
                let luma = 0.2126 * f32::from(r)
                    + 0.7152 * f32::from(g)
                    + 0.0722 * f32::from(b);
                luma.round().clamp(0.0, 255.0) as u8
            }
        }
    }
}

// ─── Configuration ────────────────────────────────────────────────────────────

/// Configuration for the luma parade scope.
#[derive(Debug, Clone)]
pub struct LumaParadeConfig {
    /// Output image width in pixels.
    pub width: u32,
    /// Output image height in pixels.
    pub height: u32,
    /// Channels to display (in left-to-right order).
    pub channels: Vec<ParadeChannel>,
    /// Whether to overlay a 10/25/50/75/90/100 % graticule.
    pub show_grid: bool,
}

impl Default for LumaParadeConfig {
    fn default() -> Self {
        Self {
            width: 512,
            height: 256,
            channels: vec![ParadeChannel::Red, ParadeChannel::Green, ParadeChannel::Blue],
            show_grid: true,
        }
    }
}

// ─── Histogram data ───────────────────────────────────────────────────────────

/// Histogram data for a single channel.
#[derive(Debug, Clone)]
pub struct ChannelHistogram {
    /// Which channel this histogram represents.
    pub channel: ParadeChannel,
    /// 256 bins, each counting how many pixels fell in that intensity level.
    pub bins: Vec<u32>,
    /// Intensity value [0, 255] of the highest-populated bin.
    pub peak_value: u8,
    /// Mean intensity [0.0, 255.0] across all pixels.
    pub mean_value: f32,
}

impl ChannelHistogram {
    /// Build a histogram from raw pixel data.
    fn build(pixels: &[u8], image_width: u32, image_height: u32, channel: ParadeChannel) -> Self {
        let mut bins = vec![0u32; 256];
        let total_pixels = (image_width * image_height) as usize;

        for idx in 0..total_pixels {
            let base = idx * 3;
            if base + 2 < pixels.len() {
                let val = channel.extract(pixels[base], pixels[base + 1], pixels[base + 2]);
                bins[val as usize] += 1;
            }
        }

        // peak_value = bin index with the highest count
        let peak_value = bins
            .iter()
            .enumerate()
            .max_by_key(|&(_, &c)| c)
            .map(|(i, _)| i as u8)
            .unwrap_or(0);

        // mean_value = weighted average of bin index * count
        let total: u64 = bins.iter().map(|&c| u64::from(c)).sum();
        let mean_value = if total == 0 {
            0.0
        } else {
            let weighted: u64 = bins
                .iter()
                .enumerate()
                .map(|(i, &c)| i as u64 * u64::from(c))
                .sum();
            weighted as f32 / total as f32
        };

        Self {
            channel,
            bins,
            peak_value,
            mean_value,
        }
    }
}

// ─── Luma Parade ─────────────────────────────────────────────────────────────

/// A computed luma parade with per-channel histograms ready for rendering.
#[derive(Debug, Clone)]
pub struct LumaParade {
    /// Per-channel histogram data, in the same order as `config.channels`.
    pub histograms: Vec<ChannelHistogram>,
    /// Configuration used to generate this parade.
    config: LumaParadeConfig,
}

impl LumaParade {
    /// Analyse `pixels` (RGB24, row-major) and build per-channel histograms.
    ///
    /// # Arguments
    ///
    /// * `pixels`       – RGB24 byte slice (`image_width * image_height * 3` bytes).
    /// * `image_width`  – Source image width.
    /// * `image_height` – Source image height.
    /// * `config`       – Parade configuration.
    #[must_use]
    pub fn generate(
        pixels: &[u8],
        image_width: u32,
        image_height: u32,
        config: &LumaParadeConfig,
    ) -> Self {
        let histograms = config
            .channels
            .iter()
            .map(|&ch| ChannelHistogram::build(pixels, image_width, image_height, ch))
            .collect();

        Self {
            histograms,
            config: config.clone(),
        }
    }

    /// Render the parade to an RGBA image (`width * height * 4` bytes).
    ///
    /// Each channel occupies an equal horizontal strip separated by a 1-pixel
    /// black gutter.  Pixel intensity is shown vertically: high values (bright)
    /// appear at the top, low values (dark) at the bottom.
    #[must_use]
    pub fn render(&self) -> Vec<u8> {
        let w = self.config.width as usize;
        let h = self.config.height as usize;
        let num_channels = self.histograms.len().max(1);

        // Total gutter pixels between channels
        const GUTTER: usize = 2;
        let total_gutters = if num_channels > 1 { (num_channels - 1) * GUTTER } else { 0 };
        let channel_w = if w > total_gutters {
            (w - total_gutters) / num_channels
        } else {
            1
        };

        // RGBA buffer, initialised to black
        let mut buf = vec![0u8; w * h * 4];

        // Render each channel pane
        for (ch_idx, hist) in self.histograms.iter().enumerate() {
            let x_start = ch_idx * (channel_w + GUTTER);
            let color = hist.channel.rgba_color();

            // Find the maximum bin count for normalisation
            let max_count = hist.bins.iter().copied().max().unwrap_or(1).max(1);

            // For each of the 256 intensity levels, determine how tall the bar is.
            // Map level → x column within this channel's pane (0 = darkest, left).
            // Map count → y height from the bottom (normalised to channel height).
            for (level, &count) in hist.bins.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                // Column within this channel pane (level 0..=255 mapped to channel_w)
                let col = (level * channel_w / 256).min(channel_w.saturating_sub(1));
                let abs_x = x_start + col;
                if abs_x >= w {
                    continue;
                }

                // Height of this bar (normalised, minimum 1 pixel)
                let bar_h = ((count as f64 / max_count as f64) * (h as f64 - 1.0))
                    .round() as usize;
                let bar_h = bar_h.clamp(1, h);

                // Draw from the bottom up
                for row in (h - bar_h)..h {
                    let px = (row * w + abs_x) * 4;
                    if px + 3 < buf.len() {
                        // Additive blend for overlapping pixels
                        buf[px] = buf[px].saturating_add(color[0] / 3);
                        buf[px + 1] = buf[px + 1].saturating_add(color[1] / 3);
                        buf[px + 2] = buf[px + 2].saturating_add(color[2] / 3);
                        buf[px + 3] = color[3];
                    }
                }
            }

            // Full-brightness top pixel for visibility
            for (level, &count) in hist.bins.iter().enumerate() {
                if count == 0 {
                    continue;
                }
                let col = (level * channel_w / 256).min(channel_w.saturating_sub(1));
                let abs_x = x_start + col;
                if abs_x >= w {
                    continue;
                }
                let bar_h = ((count as f64 / max_count as f64) * (h as f64 - 1.0))
                    .round() as usize;
                let bar_h = bar_h.clamp(1, h);
                let top_row = h - bar_h;
                let px = (top_row * w + abs_x) * 4;
                if px + 3 < buf.len() {
                    buf[px] = color[0];
                    buf[px + 1] = color[1];
                    buf[px + 2] = color[2];
                    buf[px + 3] = 255;
                }
            }
        }

        // Optional graticule (horizontal lines at 10/25/50/75/90/100 %)
        if self.config.show_grid {
            let graticule_levels = [10u32, 25, 50, 75, 90, 100];
            for &pct in &graticule_levels {
                let row = h.saturating_sub(1) - (pct as usize * (h - 1) / 100);
                for x in 0..w {
                    let px = (row * w + x) * 4;
                    if px + 3 < buf.len() {
                        // Dim white grid line (alpha blend on top)
                        let base_a = buf[px + 3];
                        if base_a < 40 {
                            buf[px] = 60;
                            buf[px + 1] = 60;
                            buf[px + 2] = 60;
                            buf[px + 3] = 80;
                        }
                    }
                }
            }
        }

        buf
    }

    /// Returns the mean value for a given channel, or `None` if that channel
    /// was not included in the parade configuration.
    #[must_use]
    pub fn channel_mean(&self, channel: ParadeChannel) -> Option<f32> {
        self.histograms
            .iter()
            .find(|h| h.channel == channel)
            .map(|h| h.mean_value)
    }

    /// Returns `true` if any pixel in `channel` has intensity ≥ `threshold`.
    ///
    /// Returns `false` (conservatively) if the channel is not present.
    #[must_use]
    pub fn is_clipping(&self, channel: ParadeChannel, threshold: u8) -> bool {
        self.histograms
            .iter()
            .find(|h| h.channel == channel)
            .map(|h| {
                // Any bin at or above `threshold` with at least one pixel
                h.bins[threshold as usize..]
                    .iter()
                    .any(|&count| count > 0)
            })
            .unwrap_or(false)
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a flat RGB24 image of `w × h` pixels all set to `(r, g, b)`.
    fn solid_rgb(w: u32, h: u32, r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut v = vec![0u8; (w * h * 3) as usize];
        for i in 0..(w * h) as usize {
            v[i * 3] = r;
            v[i * 3 + 1] = g;
            v[i * 3 + 2] = b;
        }
        v
    }

    #[test]
    fn test_histogram_bin_count_single_value() {
        let pixels = solid_rgb(4, 4, 128, 0, 0);
        let config = LumaParadeConfig {
            channels: vec![ParadeChannel::Red],
            ..Default::default()
        };
        let parade = LumaParade::generate(&pixels, 4, 4, &config);
        assert_eq!(parade.histograms.len(), 1);
        // All 16 pixels are at intensity 128
        assert_eq!(parade.histograms[0].bins[128], 16);
        // All other bins are zero
        assert_eq!(parade.histograms[0].bins[0], 0);
        assert_eq!(parade.histograms[0].bins[255], 0);
    }

    #[test]
    fn test_histogram_peak_value() {
        let pixels = solid_rgb(4, 4, 200, 100, 50);
        let config = LumaParadeConfig {
            channels: vec![ParadeChannel::Red],
            ..Default::default()
        };
        let parade = LumaParade::generate(&pixels, 4, 4, &config);
        assert_eq!(parade.histograms[0].peak_value, 200);
    }

    #[test]
    fn test_histogram_mean_value() {
        let pixels = solid_rgb(4, 4, 128, 0, 0);
        let config = LumaParadeConfig {
            channels: vec![ParadeChannel::Red],
            ..Default::default()
        };
        let parade = LumaParade::generate(&pixels, 4, 4, &config);
        let mean = parade.histograms[0].mean_value;
        assert!((mean - 128.0).abs() < 0.01, "mean={mean}");
    }

    #[test]
    fn test_channel_mean_present() {
        let pixels = solid_rgb(2, 2, 100, 200, 50);
        let config = LumaParadeConfig {
            channels: vec![ParadeChannel::Green],
            ..Default::default()
        };
        let parade = LumaParade::generate(&pixels, 2, 2, &config);
        let mean = parade.channel_mean(ParadeChannel::Green);
        assert!(mean.is_some());
        let m = mean.expect("should be Some");
        assert!((m - 200.0).abs() < 0.01, "mean={m}");
    }

    #[test]
    fn test_channel_mean_absent() {
        let pixels = solid_rgb(2, 2, 100, 200, 50);
        let config = LumaParadeConfig {
            channels: vec![ParadeChannel::Red],
            ..Default::default()
        };
        let parade = LumaParade::generate(&pixels, 2, 2, &config);
        // Blue was not requested
        assert!(parade.channel_mean(ParadeChannel::Blue).is_none());
    }

    #[test]
    fn test_clipping_detects_full_white() {
        // All pixels at 255 — should be clipping at threshold 240
        let pixels = solid_rgb(4, 4, 255, 255, 255);
        let config = LumaParadeConfig {
            channels: vec![ParadeChannel::Red, ParadeChannel::Green, ParadeChannel::Blue],
            ..Default::default()
        };
        let parade = LumaParade::generate(&pixels, 4, 4, &config);
        assert!(parade.is_clipping(ParadeChannel::Red, 240));
        assert!(parade.is_clipping(ParadeChannel::Green, 240));
        assert!(parade.is_clipping(ParadeChannel::Blue, 240));
    }

    #[test]
    fn test_clipping_not_triggered_below_threshold() {
        let pixels = solid_rgb(4, 4, 100, 100, 100);
        let config = LumaParadeConfig {
            channels: vec![ParadeChannel::Red],
            ..Default::default()
        };
        let parade = LumaParade::generate(&pixels, 4, 4, &config);
        // Threshold of 200 — pixel value 100 is well below
        assert!(!parade.is_clipping(ParadeChannel::Red, 200));
    }

    #[test]
    fn test_clipping_returns_false_for_absent_channel() {
        let pixels = solid_rgb(2, 2, 255, 0, 0);
        let config = LumaParadeConfig {
            channels: vec![ParadeChannel::Red],
            ..Default::default()
        };
        let parade = LumaParade::generate(&pixels, 2, 2, &config);
        // Blue not in config → conservatively false
        assert!(!parade.is_clipping(ParadeChannel::Blue, 0));
    }

    #[test]
    fn test_luma_channel_extraction() {
        // Pure white → luma ≈ 255
        let ch = ParadeChannel::Luma;
        assert_eq!(ch.extract(255, 255, 255), 255);
        // Pure black → luma = 0
        assert_eq!(ch.extract(0, 0, 0), 0);
    }

    #[test]
    fn test_channel_separation_rgb() {
        // Red channel only → green and blue histograms not generated
        let pixels = solid_rgb(4, 4, 200, 100, 50);
        let config = LumaParadeConfig {
            channels: vec![ParadeChannel::Red, ParadeChannel::Green, ParadeChannel::Blue],
            ..Default::default()
        };
        let parade = LumaParade::generate(&pixels, 4, 4, &config);
        assert_eq!(parade.histograms.len(), 3);
        assert_eq!(parade.histograms[0].channel, ParadeChannel::Red);
        assert_eq!(parade.histograms[1].channel, ParadeChannel::Green);
        assert_eq!(parade.histograms[2].channel, ParadeChannel::Blue);
        // Values are correct per-channel
        assert_eq!(parade.histograms[0].bins[200], 16);
        assert_eq!(parade.histograms[1].bins[100], 16);
        assert_eq!(parade.histograms[2].bins[50], 16);
    }

    #[test]
    fn test_render_output_size() {
        let pixels = solid_rgb(8, 8, 128, 128, 128);
        let config = LumaParadeConfig {
            width: 120,
            height: 60,
            channels: vec![ParadeChannel::Red, ParadeChannel::Green, ParadeChannel::Blue],
            show_grid: false,
        };
        let parade = LumaParade::generate(&pixels, 8, 8, &config);
        let rendered = parade.render();
        assert_eq!(rendered.len(), 120 * 60 * 4);
    }

    #[test]
    fn test_render_with_grid_enabled() {
        let pixels = solid_rgb(4, 4, 200, 200, 200);
        let config = LumaParadeConfig {
            width: 64,
            height: 64,
            channels: vec![ParadeChannel::Luma],
            show_grid: true,
        };
        let parade = LumaParade::generate(&pixels, 4, 4, &config);
        // Should not panic and produce correct length
        let rendered = parade.render();
        assert_eq!(rendered.len(), 64 * 64 * 4);
    }
}
