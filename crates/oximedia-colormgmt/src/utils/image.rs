//! Image buffer utilities for color processing.

use crate::colorspaces::ColorSpace;
use crate::pipeline::ColorPipeline;

/// RGB image buffer.
#[derive(Clone, Debug)]
pub struct RgbImage {
    /// Width in pixels
    pub width: usize,
    /// Height in pixels
    pub height: usize,
    /// RGB data (interleaved, row-major)
    pub data: Vec<f64>,
}

impl RgbImage {
    /// Creates a new RGB image.
    ///
    /// # Arguments
    ///
    /// * `width` - Image width
    /// * `height` - Image height
    ///
    /// # Panics
    ///
    /// Panics if width or height is zero.
    #[must_use]
    pub fn new(width: usize, height: usize) -> Self {
        assert!(width > 0 && height > 0);
        Self {
            width,
            height,
            data: vec![0.0; width * height * 3],
        }
    }

    /// Creates an RGB image from existing data.
    ///
    /// # Panics
    ///
    /// Panics if data length doesn't match width * height * 3.
    #[must_use]
    pub fn from_data(width: usize, height: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), width * height * 3);
        Self {
            width,
            height,
            data,
        }
    }

    /// Gets a pixel value at (x, y).
    ///
    /// # Returns
    ///
    /// RGB triplet or None if coordinates are out of bounds.
    #[must_use]
    pub fn get_pixel(&self, x: usize, y: usize) -> Option<[f64; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }

        let idx = (y * self.width + x) * 3;
        Some([self.data[idx], self.data[idx + 1], self.data[idx + 2]])
    }

    /// Sets a pixel value at (x, y).
    ///
    /// Returns true if successful, false if coordinates are out of bounds.
    pub fn set_pixel(&mut self, x: usize, y: usize, rgb: [f64; 3]) -> bool {
        if x >= self.width || y >= self.height {
            return false;
        }

        let idx = (y * self.width + x) * 3;
        self.data[idx] = rgb[0];
        self.data[idx + 1] = rgb[1];
        self.data[idx + 2] = rgb[2];
        true
    }

    /// Applies a color pipeline to the entire image.
    pub fn apply_pipeline(&mut self, pipeline: &ColorPipeline) {
        pipeline.transform_image(&mut self.data);
    }

    /// Converts the image to a different color space.
    pub fn convert_color_space(&mut self, from: &ColorSpace, to: &ColorSpace) {
        for y in 0..self.height {
            for x in 0..self.width {
                if let Some(rgb) = self.get_pixel(x, y) {
                    let converted = crate::transforms::rgb_to_rgb(&rgb, from, to);
                    self.set_pixel(x, y, converted);
                }
            }
        }
    }

    /// Returns the total number of pixels.
    #[must_use]
    pub const fn pixel_count(&self) -> usize {
        self.width * self.height
    }

    /// Returns a mutable reference to the raw data.
    #[must_use]
    pub fn data_mut(&mut self) -> &mut [f64] {
        &mut self.data
    }

    /// Fills the image with a solid color.
    pub fn fill(&mut self, rgb: [f64; 3]) {
        for y in 0..self.height {
            for x in 0..self.width {
                self.set_pixel(x, y, rgb);
            }
        }
    }

    /// Creates a gradient image for testing.
    #[must_use]
    pub fn gradient(width: usize, height: usize) -> Self {
        let mut img = Self::new(width, height);

        for y in 0..height {
            for x in 0..width {
                let r = x as f64 / (width - 1) as f64;
                let g = y as f64 / (height - 1) as f64;
                let b = 0.5;
                img.set_pixel(x, y, [r, g, b]);
            }
        }

        img
    }

    /// Resizes the image (simple nearest neighbor).
    #[must_use]
    pub fn resize_nearest(&self, new_width: usize, new_height: usize) -> Self {
        let mut result = Self::new(new_width, new_height);

        for y in 0..new_height {
            for x in 0..new_width {
                let src_x = (x * self.width) / new_width;
                let src_y = (y * self.height) / new_height;

                if let Some(pixel) = self.get_pixel(src_x, src_y) {
                    result.set_pixel(x, y, pixel);
                }
            }
        }

        result
    }

    /// Flips the image vertically.
    pub fn flip_vertical(&mut self) {
        for y in 0..self.height / 2 {
            for x in 0..self.width {
                let top_idx = (y * self.width + x) * 3;
                let bottom_idx = ((self.height - 1 - y) * self.width + x) * 3;

                for i in 0..3 {
                    self.data.swap(top_idx + i, bottom_idx + i);
                }
            }
        }
    }

    /// Flips the image horizontally.
    pub fn flip_horizontal(&mut self) {
        for y in 0..self.height {
            for x in 0..self.width / 2 {
                let left_idx = (y * self.width + x) * 3;
                let right_idx = (y * self.width + (self.width - 1 - x)) * 3;

                for i in 0..3 {
                    self.data.swap(left_idx + i, right_idx + i);
                }
            }
        }
    }

    /// Crops the image to a rectangular region.
    ///
    /// # Returns
    ///
    /// A new image containing the cropped region, or the original if invalid.
    #[must_use]
    pub fn crop(&self, x: usize, y: usize, width: usize, height: usize) -> Self {
        if x + width > self.width || y + height > self.height {
            return self.clone();
        }

        let mut result = Self::new(width, height);

        for dy in 0..height {
            for dx in 0..width {
                if let Some(pixel) = self.get_pixel(x + dx, y + dy) {
                    result.set_pixel(dx, dy, pixel);
                }
            }
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_image_creation() {
        let img = RgbImage::new(100, 100);
        assert_eq!(img.width, 100);
        assert_eq!(img.height, 100);
        assert_eq!(img.data.len(), 100 * 100 * 3);
    }

    #[test]
    fn test_pixel_access() {
        let mut img = RgbImage::new(10, 10);
        let rgb = [0.5, 0.3, 0.7];

        assert!(img.set_pixel(5, 5, rgb));
        let retrieved = img.get_pixel(5, 5).expect("pixel should be within bounds");

        assert_eq!(retrieved, rgb);
    }

    #[test]
    fn test_out_of_bounds() {
        let img = RgbImage::new(10, 10);
        assert!(img.get_pixel(10, 10).is_none());
        assert!(img.get_pixel(5, 10).is_none());
    }

    #[test]
    fn test_fill() {
        let mut img = RgbImage::new(10, 10);
        let color = [1.0, 0.0, 0.0];

        img.fill(color);

        for y in 0..10 {
            for x in 0..10 {
                assert_eq!(
                    img.get_pixel(x, y).expect("pixel should be within bounds"),
                    color
                );
            }
        }
    }

    #[test]
    fn test_gradient() {
        let img = RgbImage::gradient(10, 10);
        let top_left = img.get_pixel(0, 0).expect("pixel should be within bounds");
        let bottom_right = img.get_pixel(9, 9).expect("pixel should be within bounds");

        assert!((top_left[0] - 0.0).abs() < 0.01);
        assert!((bottom_right[0] - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_flip_vertical() {
        let mut img = RgbImage::new(10, 10);
        img.set_pixel(0, 0, [1.0, 0.0, 0.0]);
        img.set_pixel(0, 9, [0.0, 1.0, 0.0]);

        img.flip_vertical();

        assert_eq!(
            img.get_pixel(0, 0).expect("pixel should be within bounds"),
            [0.0, 1.0, 0.0]
        );
        assert_eq!(
            img.get_pixel(0, 9).expect("pixel should be within bounds"),
            [1.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_crop() {
        let img = RgbImage::gradient(100, 100);
        let cropped = img.crop(10, 10, 20, 20);

        assert_eq!(cropped.width, 20);
        assert_eq!(cropped.height, 20);
    }
}
