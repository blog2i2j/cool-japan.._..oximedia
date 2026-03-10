//! Color analysis scopes: vectorscope, waveform monitor, histogram.

use super::types::{ColorChannel, RgbColor};

/// Vectorscope analyzer for color distribution analysis.
#[derive(Clone, Debug)]
pub struct VectorscopeAnalyzer {
    /// Resolution of the vectorscope (size x size).
    resolution: usize,
    /// Accumulated data.
    data: Vec<f64>,
}

impl VectorscopeAnalyzer {
    /// Create a new vectorscope analyzer.
    #[must_use]
    pub fn new(resolution: usize) -> Self {
        Self {
            resolution,
            data: vec![0.0; resolution * resolution],
        }
    }

    /// Add a color sample to the vectorscope.
    pub fn add_sample(&mut self, color: RgbColor) {
        let hsl = color.to_hsl();

        // Convert hue and saturation to vectorscope coordinates
        let angle = hsl.h * std::f64::consts::PI / 180.0;
        let radius = hsl.s * 0.5; // Map saturation to radius

        let x = 0.5 + radius * angle.cos();
        let y = 0.5 + radius * angle.sin();

        let ix = (x * self.resolution as f64) as usize;
        let iy = (y * self.resolution as f64) as usize;

        if ix < self.resolution && iy < self.resolution {
            let index = iy * self.resolution + ix;
            self.data[index] += 1.0;
        }
    }

    /// Get the vectorscope data.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Clear the vectorscope data.
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Get peak saturation in a hue range.
    #[must_use]
    pub fn peak_saturation_in_range(&self, hue_min: f64, hue_max: f64) -> f64 {
        let mut max_sat: f64 = 0.0;

        for y in 0..self.resolution {
            for x in 0..self.resolution {
                let fx = x as f64 / self.resolution as f64 - 0.5;
                let fy = y as f64 / self.resolution as f64 - 0.5;

                let radius = (fx * fx + fy * fy).sqrt();
                let angle = fy.atan2(fx) * 180.0 / std::f64::consts::PI;
                let hue = (angle + 360.0).rem_euclid(360.0);

                if hue >= hue_min && hue <= hue_max {
                    let sat = radius * 2.0;
                    max_sat = max_sat.max(sat);
                }
            }
        }

        max_sat.min(1.0)
    }
}

/// Waveform monitor for luminance analysis.
#[derive(Clone, Debug)]
pub struct WaveformMonitor {
    /// Width of the waveform.
    width: usize,
    /// Height of the waveform (represents luminance range 0-1).
    height: usize,
    /// Accumulated data.
    data: Vec<f64>,
}

impl WaveformMonitor {
    /// Create a new waveform monitor.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            data: vec![0.0; width * height],
        }
    }

    /// Add a sample at a specific x position.
    pub fn add_sample(&mut self, x_pos: f64, luma: f64) {
        let x = (x_pos * self.width as f64) as usize;
        let y = ((1.0 - luma.clamp(0.0, 1.0)) * (self.height - 1) as f64) as usize;

        if x < self.width && y < self.height {
            let index = y * self.width + x;
            self.data[index] += 1.0;
        }
    }

    /// Get the waveform data.
    #[must_use]
    pub fn data(&self) -> &[f64] {
        &self.data
    }

    /// Clear the waveform data.
    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    /// Get average luminance in a horizontal range.
    #[must_use]
    pub fn average_luma_in_range(&self, x_min: f64, x_max: f64) -> f64 {
        let x_start = (x_min * self.width as f64) as usize;
        let x_end = (x_max * self.width as f64) as usize;

        let mut sum = 0.0;
        let mut count = 0;

        for y in 0..self.height {
            for x in x_start..x_end.min(self.width) {
                let index = y * self.width + x;
                if self.data[index] > 0.0 {
                    let luma = 1.0 - (y as f64 / (self.height - 1) as f64);
                    sum += luma * self.data[index];
                    count += self.data[index] as usize;
                }
            }
        }

        if count > 0 {
            sum / count as f64
        } else {
            0.0
        }
    }
}

/// Histogram for analyzing color channel distributions.
#[derive(Clone, Debug)]
pub struct ColorHistogram {
    /// Number of bins.
    bins: usize,
    /// Red channel histogram.
    pub red: Vec<u32>,
    /// Green channel histogram.
    pub green: Vec<u32>,
    /// Blue channel histogram.
    pub blue: Vec<u32>,
    /// Luminance histogram.
    pub luma: Vec<u32>,
}

impl ColorHistogram {
    /// Create a new histogram.
    #[must_use]
    pub fn new(bins: usize) -> Self {
        Self {
            bins,
            red: vec![0; bins],
            green: vec![0; bins],
            blue: vec![0; bins],
            luma: vec![0; bins],
        }
    }

    /// Add a color sample.
    pub fn add_sample(&mut self, color: RgbColor) {
        let r_bin = (color.r.clamp(0.0, 1.0) * (self.bins - 1) as f64) as usize;
        let g_bin = (color.g.clamp(0.0, 1.0) * (self.bins - 1) as f64) as usize;
        let b_bin = (color.b.clamp(0.0, 1.0) * (self.bins - 1) as f64) as usize;

        let luma = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
        let l_bin = (luma.clamp(0.0, 1.0) * (self.bins - 1) as f64) as usize;

        self.red[r_bin] += 1;
        self.green[g_bin] += 1;
        self.blue[b_bin] += 1;
        self.luma[l_bin] += 1;
    }

    /// Clear the histogram.
    pub fn clear(&mut self) {
        self.red.fill(0);
        self.green.fill(0);
        self.blue.fill(0);
        self.luma.fill(0);
    }

    /// Get the median value for a channel.
    #[must_use]
    pub fn median(&self, channel: ColorChannel) -> f64 {
        let histogram = match channel {
            ColorChannel::Red => &self.red,
            ColorChannel::Green => &self.green,
            ColorChannel::Blue => &self.blue,
            ColorChannel::Luma => &self.luma,
        };

        let total: u32 = histogram.iter().sum();
        if total == 0 {
            return 0.5;
        }

        let half = total / 2;
        let mut acc = 0;

        for (i, &count) in histogram.iter().enumerate() {
            acc += count;
            if acc >= half {
                return i as f64 / (self.bins - 1) as f64;
            }
        }

        0.5
    }

    /// Get the mean value for a channel.
    #[must_use]
    pub fn mean(&self, channel: ColorChannel) -> f64 {
        let histogram = match channel {
            ColorChannel::Red => &self.red,
            ColorChannel::Green => &self.green,
            ColorChannel::Blue => &self.blue,
            ColorChannel::Luma => &self.luma,
        };

        let total: u32 = histogram.iter().sum();
        if total == 0 {
            return 0.5;
        }

        let mut sum = 0.0;
        for (i, &count) in histogram.iter().enumerate() {
            sum += (i as f64 / (self.bins - 1) as f64) * count as f64;
        }

        sum / total as f64
    }

    /// Get percentile value for a channel.
    #[must_use]
    pub fn percentile(&self, channel: ColorChannel, percentile: f64) -> f64 {
        let histogram = match channel {
            ColorChannel::Red => &self.red,
            ColorChannel::Green => &self.green,
            ColorChannel::Blue => &self.blue,
            ColorChannel::Luma => &self.luma,
        };

        let total: u32 = histogram.iter().sum();
        if total == 0 {
            return 0.5;
        }

        let target = (total as f64 * percentile.clamp(0.0, 1.0)) as u32;
        let mut acc = 0;

        for (i, &count) in histogram.iter().enumerate() {
            acc += count;
            if acc >= target {
                return i as f64 / (self.bins - 1) as f64;
            }
        }

        1.0
    }
}
