//! LUT analysis, caching, procedural generation, and utility functions.

use std::path::Path;

use super::io::load_lut_file;
use super::lut1d::Lut1d;
use super::lut3d::{overlay_blend, Lut3d};
use super::types::{ColorChannel, RgbColor};

/// Analyze a LUT for statistics and characteristics.
#[derive(Clone, Debug, Default)]
pub struct LutAnalysis {
    /// Average change magnitude.
    pub avg_change: f64,
    /// Maximum change magnitude.
    pub max_change: f64,
    /// Is approximately identity.
    pub is_identity: bool,
    /// Dynamic range min.
    pub range_min: RgbColor,
    /// Dynamic range max.
    pub range_max: RgbColor,
}

impl LutAnalysis {
    /// Analyze a 3D LUT.
    #[must_use]
    pub fn analyze(lut: &Lut3d) -> Self {
        let mut total_change = 0.0;
        let mut max_change: f64 = 0.0;
        let mut min_vals = RgbColor::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max_vals = RgbColor::new(f64::MIN, f64::MIN, f64::MIN);
        let mut is_identity = true;

        let size = lut.size;
        let count = size * size * size;

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let input = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );

                    let output = lut.get(r, g, b);

                    // Calculate change
                    let change = ((output.r - input.r).powi(2)
                        + (output.g - input.g).powi(2)
                        + (output.b - input.b).powi(2))
                    .sqrt();

                    total_change += change;
                    max_change = max_change.max(change);

                    if change > 0.01 {
                        is_identity = false;
                    }

                    // Track range
                    min_vals.r = min_vals.r.min(output.r);
                    min_vals.g = min_vals.g.min(output.g);
                    min_vals.b = min_vals.b.min(output.b);

                    max_vals.r = max_vals.r.max(output.r);
                    max_vals.g = max_vals.g.max(output.g);
                    max_vals.b = max_vals.b.max(output.b);
                }
            }
        }

        Self {
            avg_change: total_change / count as f64,
            max_change,
            is_identity,
            range_min: min_vals,
            range_max: max_vals,
        }
    }
}

/// LUT caching for performance optimization.
#[derive(Clone, Debug)]
pub struct LutCache {
    /// Cached LUTs by path.
    cache: std::collections::HashMap<String, Lut3d>,
    /// Maximum cache size.
    max_size: usize,
}

impl LutCache {
    /// Create a new LUT cache.
    #[must_use]
    pub fn new(max_size: usize) -> Self {
        Self {
            cache: std::collections::HashMap::new(),
            max_size,
        }
    }

    /// Get a LUT from cache or load it.
    pub fn get_or_load(&mut self, path: &str) -> Result<Lut3d, String> {
        if let Some(lut) = self.cache.get(path) {
            Ok(lut.clone())
        } else {
            let lut = load_lut_file(Path::new(path))?;

            // Evict oldest if cache is full
            if self.cache.len() >= self.max_size {
                if let Some(key) = self.cache.keys().next().cloned() {
                    self.cache.remove(&key);
                }
            }

            self.cache.insert(path.to_string(), lut.clone());
            Ok(lut)
        }
    }

    /// Clear the cache.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// Get cache statistics.
    #[must_use]
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.cache.len(),
            max_size: self.max_size,
        }
    }
}

impl Default for LutCache {
    fn default() -> Self {
        Self::new(10)
    }
}

/// Cache statistics.
#[derive(Clone, Copy, Debug)]
pub struct CacheStats {
    /// Number of cached entries.
    pub entries: usize,
    /// Maximum cache size.
    pub max_size: usize,
}

/// Procedural LUT generation functions.
pub mod procedural {
    use super::*;

    /// Generate a contrast adjustment LUT.
    #[must_use]
    pub fn contrast_lut(size: usize, contrast: f64) -> Lut3d {
        let mut lut = Lut3d::new(size);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let input = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );

                    let output = RgbColor::new(
                        apply_contrast(input.r, contrast),
                        apply_contrast(input.g, contrast),
                        apply_contrast(input.b, contrast),
                    );

                    lut.set(r, g, b, output);
                }
            }
        }

        lut
    }

    /// Generate a saturation adjustment LUT.
    #[must_use]
    pub fn saturation_lut(size: usize, saturation: f64) -> Lut3d {
        let mut lut = Lut3d::new(size);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let input = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );

                    // Calculate luminance
                    let luma = 0.2126 * input.r + 0.7152 * input.g + 0.0722 * input.b;

                    let output = RgbColor::new(
                        luma + (input.r - luma) * saturation,
                        luma + (input.g - luma) * saturation,
                        luma + (input.b - luma) * saturation,
                    );

                    lut.set(r, g, b, output.clamp());
                }
            }
        }

        lut
    }

    /// Generate a hue shift LUT.
    #[must_use]
    pub fn hue_shift_lut(size: usize, hue_shift: f64) -> Lut3d {
        let mut lut = Lut3d::new(size);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let input = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );

                    let output = shift_hue(input, hue_shift);
                    lut.set(r, g, b, output.clamp());
                }
            }
        }

        lut
    }

    /// Generate a temperature adjustment LUT.
    #[must_use]
    pub fn temperature_lut(size: usize, temperature: f64) -> Lut3d {
        let mut lut = Lut3d::new(size);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let input = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );

                    let output = if temperature > 0.0 {
                        // Warm
                        RgbColor::new(
                            input.r * (1.0 + temperature * 0.1),
                            input.g,
                            input.b * (1.0 - temperature * 0.1),
                        )
                    } else {
                        // Cool
                        RgbColor::new(
                            input.r * (1.0 + temperature * 0.1),
                            input.g,
                            input.b * (1.0 - temperature * 0.1),
                        )
                    };

                    lut.set(r, g, b, output.clamp());
                }
            }
        }

        lut
    }

    /// Generate a vibrance adjustment LUT.
    #[must_use]
    pub fn vibrance_lut(size: usize, vibrance: f64) -> Lut3d {
        let mut lut = Lut3d::new(size);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let input = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );

                    let output = apply_vibrance(input, vibrance);
                    lut.set(r, g, b, output.clamp());
                }
            }
        }

        lut
    }

    /// Generate an exposure adjustment LUT.
    #[must_use]
    pub fn exposure_lut(size: usize, exposure: f64) -> Lut3d {
        let mut lut = Lut3d::new(size);
        let multiplier = 2_f64.powf(exposure);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let input = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );

                    let output = RgbColor::new(
                        input.r * multiplier,
                        input.g * multiplier,
                        input.b * multiplier,
                    );

                    lut.set(r, g, b, output.clamp());
                }
            }
        }

        lut
    }

    /// Generate a sepia tone LUT.
    #[must_use]
    pub fn sepia_lut(size: usize, strength: f64) -> Lut3d {
        let mut lut = Lut3d::new(size);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let input = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );

                    let sepia_r = input.r * 0.393 + input.g * 0.769 + input.b * 0.189;
                    let sepia_g = input.r * 0.349 + input.g * 0.686 + input.b * 0.168;
                    let sepia_b = input.r * 0.272 + input.g * 0.534 + input.b * 0.131;

                    let sepia = RgbColor::new(sepia_r, sepia_g, sepia_b);
                    let output = input.lerp(&sepia, strength);

                    lut.set(r, g, b, output.clamp());
                }
            }
        }

        lut
    }

    /// Generate a bleach bypass LUT.
    #[must_use]
    pub fn bleach_bypass_lut(size: usize, strength: f64) -> Lut3d {
        let mut lut = Lut3d::new(size);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let input = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );

                    let luma = 0.2126 * input.r + 0.7152 * input.g + 0.0722 * input.b;

                    // Bleach bypass blends the image with its luminance
                    let bleach = RgbColor::new(
                        overlay_blend(input.r, luma),
                        overlay_blend(input.g, luma),
                        overlay_blend(input.b, luma),
                    );

                    let output = input.lerp(&bleach, strength);
                    lut.set(r, g, b, output.clamp());
                }
            }
        }

        lut
    }

    /// Generate a channel mixer LUT.
    #[must_use]
    pub fn channel_mixer_lut(
        size: usize,
        rr: f64,
        rg: f64,
        rb: f64,
        gr: f64,
        gg: f64,
        gb: f64,
        br: f64,
        bg: f64,
        bb: f64,
    ) -> Lut3d {
        let mut lut = Lut3d::new(size);

        for r in 0..size {
            for g in 0..size {
                for b in 0..size {
                    let input = RgbColor::new(
                        r as f64 / (size - 1) as f64,
                        g as f64 / (size - 1) as f64,
                        b as f64 / (size - 1) as f64,
                    );

                    let output = RgbColor::new(
                        input.r * rr + input.g * rg + input.b * rb,
                        input.r * gr + input.g * gg + input.b * gb,
                        input.r * br + input.g * bg + input.b * bb,
                    );

                    lut.set(r, g, b, output.clamp());
                }
            }
        }

        lut
    }

    fn apply_contrast(value: f64, contrast: f64) -> f64 {
        ((value - 0.5) * contrast + 0.5).clamp(0.0, 1.0)
    }

    fn shift_hue(color: RgbColor, hue_shift: f64) -> RgbColor {
        let (h, s, v) = rgb_to_hsv(color);
        let new_h = (h + hue_shift).rem_euclid(360.0);
        hsv_to_rgb(new_h, s, v)
    }

    fn apply_vibrance(color: RgbColor, vibrance: f64) -> RgbColor {
        let max_val = color.r.max(color.g).max(color.b);
        let min_val = color.r.min(color.g).min(color.b);
        let saturation = if max_val > 0.0 {
            (max_val - min_val) / max_val
        } else {
            0.0
        };

        // Apply vibrance more to less saturated colors
        let adjustment = vibrance * (1.0 - saturation);
        let luma = 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;

        RgbColor::new(
            luma + (color.r - luma) * (1.0 + adjustment),
            luma + (color.g - luma) * (1.0 + adjustment),
            luma + (color.b - luma) * (1.0 + adjustment),
        )
    }

    fn rgb_to_hsv(color: RgbColor) -> (f64, f64, f64) {
        let max_val = color.r.max(color.g).max(color.b);
        let min_val = color.r.min(color.g).min(color.b);
        let delta = max_val - min_val;

        let v = max_val;
        let s = if max_val > 0.0 { delta / max_val } else { 0.0 };

        let h = if delta == 0.0 {
            0.0
        } else if (max_val - color.r).abs() < f64::EPSILON {
            60.0 * (((color.g - color.b) / delta).rem_euclid(6.0))
        } else if (max_val - color.g).abs() < f64::EPSILON {
            60.0 * (((color.b - color.r) / delta) + 2.0)
        } else {
            60.0 * (((color.r - color.g) / delta) + 4.0)
        };

        (h, s, v)
    }

    fn hsv_to_rgb(h: f64, s: f64, v: f64) -> RgbColor {
        let c = v * s;
        let x = c * (1.0 - ((h / 60.0).rem_euclid(2.0) - 1.0).abs());
        let m = v - c;

        let (r, g, b) = if h < 60.0 {
            (c, x, 0.0)
        } else if h < 120.0 {
            (x, c, 0.0)
        } else if h < 180.0 {
            (0.0, c, x)
        } else if h < 240.0 {
            (0.0, x, c)
        } else if h < 300.0 {
            (x, 0.0, c)
        } else {
            (c, 0.0, x)
        };

        RgbColor::new(r + m, g + m, b + m)
    }
}

/// Advanced LUT manipulation utilities.
pub mod utils {
    use super::*;

    /// Extract a 1D LUT from a 3D LUT along a specific axis.
    #[must_use]
    pub fn extract_1d_lut(lut: &Lut3d, channel: ColorChannel) -> Lut1d {
        let size = lut.size;
        let mut lut_1d = Lut1d::new(size);

        for i in 0..size {
            let t = i as f64 / (size - 1) as f64;
            let color = match channel {
                ColorChannel::Red => RgbColor::new(t, 0.0, 0.0),
                ColorChannel::Green => RgbColor::new(0.0, t, 0.0),
                ColorChannel::Blue => RgbColor::new(0.0, 0.0, t),
                ColorChannel::Luminance => RgbColor::new(t, t, t),
            };

            let result = lut.apply_trilinear(color);

            lut_1d.r_lut[i] = result.r;
            lut_1d.g_lut[i] = result.g;
            lut_1d.b_lut[i] = result.b;
        }

        lut_1d
    }

    /// Smooth a LUT by applying a simple averaging filter.
    #[must_use]
    pub fn smooth_lut(lut: &Lut3d, iterations: usize) -> Lut3d {
        let mut result = lut.clone();

        for _ in 0..iterations {
            let mut new_data = result.data.clone();

            for r in 1..(result.size - 1) {
                for g in 1..(result.size - 1) {
                    for b in 1..(result.size - 1) {
                        let mut sum = RgbColor::new(0.0, 0.0, 0.0);
                        let mut count = 0.0;

                        // Average with neighbors
                        for dr in -1_i32..=1 {
                            for dg in -1_i32..=1 {
                                for db in -1_i32..=1 {
                                    let nr = (r as i32 + dr) as usize;
                                    let ng = (g as i32 + dg) as usize;
                                    let nb = (b as i32 + db) as usize;

                                    let color = result.get(nr, ng, nb);
                                    sum.r += color.r;
                                    sum.g += color.g;
                                    sum.b += color.b;
                                    count += 1.0;
                                }
                            }
                        }

                        let idx = result.index(r, g, b);
                        new_data[idx] = RgbColor::new(sum.r / count, sum.g / count, sum.b / count);
                    }
                }
            }

            result.data = new_data;
        }

        result
    }

    /// Detect if a LUT is monotonic (values always increase).
    #[must_use]
    pub fn is_monotonic(lut: &Lut3d) -> bool {
        let size = lut.size;

        for r in 0..(size - 1) {
            for g in 0..(size - 1) {
                for b in 0..(size - 1) {
                    let curr = lut.get(r, g, b);
                    let next_r = lut.get(r + 1, g, b);
                    let next_g = lut.get(r, g + 1, b);
                    let next_b = lut.get(r, g, b + 1);

                    if next_r.r < curr.r || next_g.g < curr.g || next_b.b < curr.b {
                        return false;
                    }
                }
            }
        }

        true
    }

    /// Clamp all LUT values to a specific range.
    #[must_use]
    pub fn clamp_lut(lut: &Lut3d, min: f64, max: f64) -> Lut3d {
        let mut result = lut.clone();

        for color in &mut result.data {
            color.r = color.r.clamp(min, max);
            color.g = color.g.clamp(min, max);
            color.b = color.b.clamp(min, max);
        }

        result
    }

    /// Normalize a LUT to use the full [0, 1] range.
    #[must_use]
    pub fn normalize_lut(lut: &Lut3d) -> Lut3d {
        let mut result = lut.clone();

        let mut min_vals = RgbColor::new(f64::MAX, f64::MAX, f64::MAX);
        let mut max_vals = RgbColor::new(f64::MIN, f64::MIN, f64::MIN);

        // Find range
        for color in &lut.data {
            min_vals.r = min_vals.r.min(color.r);
            min_vals.g = min_vals.g.min(color.g);
            min_vals.b = min_vals.b.min(color.b);

            max_vals.r = max_vals.r.max(color.r);
            max_vals.g = max_vals.g.max(color.g);
            max_vals.b = max_vals.b.max(color.b);
        }

        // Normalize
        for color in &mut result.data {
            if max_vals.r > min_vals.r {
                color.r = (color.r - min_vals.r) / (max_vals.r - min_vals.r);
            }
            if max_vals.g > min_vals.g {
                color.g = (color.g - min_vals.g) / (max_vals.g - min_vals.g);
            }
            if max_vals.b > min_vals.b {
                color.b = (color.b - min_vals.b) / (max_vals.b - min_vals.b);
            }
        }

        result
    }
}
