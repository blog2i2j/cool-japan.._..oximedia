#![allow(dead_code)]
//! Color quantization and palette extraction for image processing.
//!
//! Implements the median-cut algorithm for reducing a set of colors to a
//! representative palette, plus utilities for nearest-palette-color lookup
//! and dithering error computation.

/// An sRGB color with 8-bit channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rgb8 {
    /// Red channel (0–255).
    pub r: u8,
    /// Green channel (0–255).
    pub g: u8,
    /// Blue channel (0–255).
    pub b: u8,
}

impl Rgb8 {
    /// Creates a new 8-bit RGB color.
    #[must_use]
    pub const fn new(r: u8, g: u8, b: u8) -> Self {
        Self { r, g, b }
    }

    /// Returns the color as an `[r, g, b]` array.
    #[must_use]
    pub const fn to_array(self) -> [u8; 3] {
        [self.r, self.g, self.b]
    }

    /// Squared Euclidean distance in RGB space.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn distance_sq(self, other: Self) -> u32 {
        let dr = i32::from(self.r) - i32::from(other.r);
        let dg = i32::from(self.g) - i32::from(other.g);
        let db = i32::from(self.b) - i32::from(other.b);
        (dr * dr + dg * dg + db * db) as u32
    }
}

/// Which color channel to split on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Channel {
    Red,
    Green,
    Blue,
}

/// A bounding box (bucket) of colors used by median-cut.
#[derive(Debug, Clone)]
struct ColorBox {
    colors: Vec<Rgb8>,
}

impl ColorBox {
    /// Creates a new color box.
    fn new(colors: Vec<Rgb8>) -> Self {
        Self { colors }
    }

    /// Returns the channel with the widest range.
    fn widest_channel(&self) -> Channel {
        let (mut rmin, mut rmax) = (u8::MAX, u8::MIN);
        let (mut gmin, mut gmax) = (u8::MAX, u8::MIN);
        let (mut bmin, mut bmax) = (u8::MAX, u8::MIN);

        for c in &self.colors {
            rmin = rmin.min(c.r);
            rmax = rmax.max(c.r);
            gmin = gmin.min(c.g);
            gmax = gmax.max(c.g);
            bmin = bmin.min(c.b);
            bmax = bmax.max(c.b);
        }

        let r_range = rmax - rmin;
        let g_range = gmax - gmin;
        let b_range = bmax - bmin;

        if r_range >= g_range && r_range >= b_range {
            Channel::Red
        } else if g_range >= b_range {
            Channel::Green
        } else {
            Channel::Blue
        }
    }

    /// Returns the average color of this box.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn average_color(&self) -> Rgb8 {
        if self.colors.is_empty() {
            return Rgb8::new(0, 0, 0);
        }
        let n = self.colors.len() as f64;
        let mut r_sum = 0u64;
        let mut g_sum = 0u64;
        let mut b_sum = 0u64;
        for c in &self.colors {
            r_sum += u64::from(c.r);
            g_sum += u64::from(c.g);
            b_sum += u64::from(c.b);
        }
        Rgb8::new(
            (r_sum as f64 / n).round() as u8,
            (g_sum as f64 / n).round() as u8,
            (b_sum as f64 / n).round() as u8,
        )
    }

    /// Splits this box into two along the widest channel at the median.
    fn split(mut self) -> (Self, Self) {
        let ch = self.widest_channel();
        match ch {
            Channel::Red => self.colors.sort_by_key(|c| c.r),
            Channel::Green => self.colors.sort_by_key(|c| c.g),
            Channel::Blue => self.colors.sort_by_key(|c| c.b),
        }
        let mid = self.colors.len() / 2;
        let right = self.colors.split_off(mid);
        (Self::new(self.colors), Self::new(right))
    }
}

/// Extracts a palette from a list of colors using the median-cut algorithm.
///
/// # Arguments
///
/// * `colors` - Input colors (can contain duplicates)
/// * `palette_size` - Desired number of palette entries (will be rounded to nearest power of 2 in iterations)
///
/// # Returns
///
/// A vector of representative palette colors.
#[must_use]
pub fn median_cut(colors: &[Rgb8], palette_size: usize) -> Vec<Rgb8> {
    if colors.is_empty() || palette_size == 0 {
        return Vec::new();
    }
    if palette_size >= colors.len() {
        let mut unique: Vec<Rgb8> = colors.to_vec();
        unique.dedup();
        return unique;
    }

    let mut boxes = vec![ColorBox::new(colors.to_vec())];

    while boxes.len() < palette_size {
        // Find box with most colors
        let idx = boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| b.colors.len() > 1)
            .max_by_key(|(_, b)| b.colors.len())
            .map(|(i, _)| i);

        let Some(idx) = idx else { break };

        let biggest = boxes.remove(idx);
        let (a, b) = biggest.split();
        if !a.colors.is_empty() {
            boxes.push(a);
        }
        if !b.colors.is_empty() {
            boxes.push(b);
        }
    }

    boxes.iter().map(ColorBox::average_color).collect()
}

/// A quantized palette with fast nearest-color lookup.
#[derive(Debug, Clone)]
pub struct Palette {
    /// The palette colors.
    entries: Vec<Rgb8>,
}

impl Palette {
    /// Creates a palette from a set of representative colors.
    #[must_use]
    pub fn new(entries: Vec<Rgb8>) -> Self {
        Self { entries }
    }

    /// Creates a palette by extracting representative colors from input data.
    #[must_use]
    pub fn from_colors(colors: &[Rgb8], size: usize) -> Self {
        Self {
            entries: median_cut(colors, size),
        }
    }

    /// Returns the palette entries.
    #[must_use]
    pub fn entries(&self) -> &[Rgb8] {
        &self.entries
    }

    /// Returns the number of entries in the palette.
    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the palette has no entries.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Finds the nearest palette color to the given color (brute-force).
    ///
    /// Returns the index and the color.
    #[must_use]
    pub fn nearest(&self, color: Rgb8) -> Option<(usize, Rgb8)> {
        self.entries
            .iter()
            .enumerate()
            .min_by_key(|(_, &c)| color.distance_sq(c))
            .map(|(i, &c)| (i, c))
    }

    /// Quantizes a list of colors to palette indices.
    #[must_use]
    pub fn quantize(&self, colors: &[Rgb8]) -> Vec<usize> {
        colors
            .iter()
            .map(|&c| self.nearest(c).map_or(0, |(i, _)| i))
            .collect()
    }
}

/// Computes the quantization error (mean squared error) over a set of colors.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn quantization_mse(palette: &Palette, colors: &[Rgb8]) -> f64 {
    if colors.is_empty() || palette.is_empty() {
        return 0.0;
    }
    let total: u64 = colors
        .iter()
        .map(|&c| {
            let (_, nearest) = palette.nearest(c).unwrap_or((0, Rgb8::new(0, 0, 0)));
            u64::from(c.distance_sq(nearest))
        })
        .sum();
    total as f64 / colors.len() as f64
}

/// Computes the Floyd-Steinberg dithering error for a single pixel.
///
/// Returns `(error_r, error_g, error_b)` as signed values.
#[must_use]
pub fn dithering_error(original: Rgb8, quantized: Rgb8) -> (i16, i16, i16) {
    (
        i16::from(original.r) - i16::from(quantized.r),
        i16::from(original.g) - i16::from(quantized.g),
        i16::from(original.b) - i16::from(quantized.b),
    )
}

/// Applies a dithering error to a pixel with clamping.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn apply_error(pixel: Rgb8, error: (i16, i16, i16), factor: f64) -> Rgb8 {
    let clamp = |v: f64| -> u8 { v.round().clamp(0.0, 255.0) as u8 };
    Rgb8::new(
        clamp(f64::from(pixel.r) + f64::from(error.0) * factor),
        clamp(f64::from(pixel.g) + f64::from(error.1) * factor),
        clamp(f64::from(pixel.b) + f64::from(error.2) * factor),
    )
}

/// Computes the color histogram from a list of colors.
///
/// Returns a vector of `(color, count)` pairs sorted by count descending.
#[must_use]
pub fn color_histogram(colors: &[Rgb8]) -> Vec<(Rgb8, usize)> {
    use std::collections::HashMap;
    let mut map: HashMap<Rgb8, usize> = HashMap::new();
    for &c in colors {
        *map.entry(c).or_insert(0) += 1;
    }
    let mut hist: Vec<_> = map.into_iter().collect();
    hist.sort_by(|a, b| b.1.cmp(&a.1));
    hist
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rgb8_new() {
        let c = Rgb8::new(10, 20, 30);
        assert_eq!(c.r, 10);
        assert_eq!(c.g, 20);
        assert_eq!(c.b, 30);
    }

    #[test]
    fn test_rgb8_to_array() {
        let c = Rgb8::new(1, 2, 3);
        assert_eq!(c.to_array(), [1, 2, 3]);
    }

    #[test]
    fn test_distance_sq_same() {
        let c = Rgb8::new(100, 100, 100);
        assert_eq!(c.distance_sq(c), 0);
    }

    #[test]
    fn test_distance_sq_known() {
        let a = Rgb8::new(0, 0, 0);
        let b = Rgb8::new(3, 4, 0);
        assert_eq!(a.distance_sq(b), 25); // 9 + 16
    }

    #[test]
    fn test_median_cut_empty() {
        let result = median_cut(&[], 4);
        assert!(result.is_empty());
    }

    #[test]
    fn test_median_cut_single() {
        let colors = vec![Rgb8::new(128, 128, 128)];
        let result = median_cut(&colors, 1);
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_median_cut_two_clusters() {
        let mut colors = Vec::new();
        for _ in 0..50 {
            colors.push(Rgb8::new(200, 50, 50));
        }
        for _ in 0..50 {
            colors.push(Rgb8::new(50, 50, 200));
        }
        let palette = median_cut(&colors, 2);
        assert_eq!(palette.len(), 2);
    }

    #[test]
    fn test_palette_nearest() {
        let p = Palette::new(vec![Rgb8::new(0, 0, 0), Rgb8::new(255, 255, 255)]);
        let (idx, _) = p
            .nearest(Rgb8::new(200, 200, 200))
            .expect("nearest color should be found");
        assert_eq!(idx, 1);
    }

    #[test]
    fn test_palette_quantize() {
        let p = Palette::new(vec![Rgb8::new(0, 0, 0), Rgb8::new(255, 255, 255)]);
        let indices = p.quantize(&[Rgb8::new(10, 10, 10), Rgb8::new(245, 245, 245)]);
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn test_quantization_mse_exact() {
        let colors = vec![Rgb8::new(100, 100, 100)];
        let p = Palette::new(vec![Rgb8::new(100, 100, 100)]);
        let mse = quantization_mse(&p, &colors);
        assert!((mse).abs() < f64::EPSILON);
    }

    #[test]
    fn test_dithering_error() {
        let orig = Rgb8::new(200, 100, 50);
        let quant = Rgb8::new(190, 110, 60);
        let (er, eg, eb) = dithering_error(orig, quant);
        assert_eq!(er, 10);
        assert_eq!(eg, -10);
        assert_eq!(eb, -10);
    }

    #[test]
    fn test_apply_error_clamping() {
        let pixel = Rgb8::new(250, 5, 128);
        let error = (20, -20, 0);
        let result = apply_error(pixel, error, 1.0);
        assert_eq!(result.r, 255); // clamped
        assert_eq!(result.g, 0); // clamped
        assert_eq!(result.b, 128);
    }

    #[test]
    fn test_color_histogram() {
        let colors = vec![
            Rgb8::new(1, 1, 1),
            Rgb8::new(2, 2, 2),
            Rgb8::new(1, 1, 1),
            Rgb8::new(1, 1, 1),
        ];
        let hist = color_histogram(&colors);
        assert_eq!(hist[0].0, Rgb8::new(1, 1, 1));
        assert_eq!(hist[0].1, 3);
        assert_eq!(hist[1].0, Rgb8::new(2, 2, 2));
        assert_eq!(hist[1].1, 1);
    }

    #[test]
    fn test_palette_from_colors() {
        let colors = vec![
            Rgb8::new(10, 10, 10),
            Rgb8::new(20, 20, 20),
            Rgb8::new(240, 240, 240),
            Rgb8::new(250, 250, 250),
        ];
        let p = Palette::from_colors(&colors, 2);
        assert_eq!(p.len(), 2);
    }
}
