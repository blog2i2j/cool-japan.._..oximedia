//! Integral image (summed-area table) for fast box filter operations.
//!
//! An integral image `S` of a source image `I` is defined as:
//!
//! ```text
//! S(x, y) = Σ I(i, j) for all i ≤ x, j ≤ y
//! ```
//!
//! Once the integral image is built in O(W×H) time, a rectangular box sum
//! `Σ I(x, y) for x ∈ [x0, x0+bw), y ∈ [y0, y0+bh)` can be computed
//! in O(1) using the standard four-corner formula.
//!
//! This makes integral images the backbone of fast Haar-like feature
//! detection (Viola-Jones), as well as fast BRIEF test sampling.

/// A 2-D integral image providing O(1) rectangular box sums.
///
/// # Example
///
/// ```rust
/// use oximedia_align::integral_image::IntegralImage;
///
/// let pixels = vec![
///     1u8, 2, 3,
///     4,   5, 6,
///     7,   8, 9,
/// ];
/// let ii = IntegralImage::build(&pixels, 3, 3);
/// // Sum of the full 3×3 image should be 45
/// assert_eq!(ii.box_sum(0, 0, 3, 3), 45);
/// ```
#[derive(Debug, Clone)]
pub struct IntegralImage {
    /// The integral image data.  Layout: row-major, `(width+1) × (height+1)`
    /// with a zero-padded top row and left column.
    data: Vec<u64>,
    /// Width of the *source* image (not the integral image).
    pub width: u32,
    /// Height of the *source* image.
    pub height: u32,
}

impl IntegralImage {
    /// Build an integral image from a grayscale pixel buffer.
    ///
    /// `img` must have length `w * h`.  If the buffer is shorter the extra
    /// entries are treated as zero.
    #[must_use]
    pub fn build(img: &[u8], w: u32, h: u32) -> Self {
        let iw = (w as usize) + 1;
        let ih = (h as usize) + 1;
        let mut data = vec![0u64; iw * ih];

        for y in 0..h as usize {
            for x in 0..w as usize {
                let src_idx = y * w as usize + x;
                let pixel = if src_idx < img.len() {
                    img[src_idx] as u64
                } else {
                    0
                };

                // Integral at (x+1, y+1) in the padded image:
                // I(x,y) + S(x-1,y) + S(x,y-1) - S(x-1,y-1)
                let above = if y > 0 { data[y * iw + (x + 1)] } else { 0 };
                let left = if x > 0 { data[(y + 1) * iw + x] } else { 0 };
                let diag = if x > 0 && y > 0 { data[y * iw + x] } else { 0 };

                data[(y + 1) * iw + (x + 1)] = pixel + above + left - diag;
            }
        }

        Self {
            data,
            width: w,
            height: h,
        }
    }

    /// Compute the sum of pixel values in the rectangle
    /// `[x, x+bw) × [y, y+bh)`.
    ///
    /// Coordinates are clamped to the image boundary so out-of-bounds
    /// rectangles do not panic.
    #[must_use]
    pub fn box_sum(&self, x: u32, y: u32, bw: u32, bh: u32) -> u64 {
        let iw = self.width as usize + 1;

        // Clamp to image bounds
        let x1 = (x as usize).min(self.width as usize);
        let y1 = (y as usize).min(self.height as usize);
        let x2 = ((x as usize) + bw as usize).min(self.width as usize);
        let y2 = ((y as usize) + bh as usize).min(self.height as usize);

        if x2 <= x1 || y2 <= y1 {
            return 0;
        }

        // Four-corner formula:
        // sum = S(x2,y2) - S(x1,y2) - S(x2,y1) + S(x1,y1)
        let s_x2_y2 = self.data[y2 * iw + x2];
        let s_x1_y2 = self.data[y2 * iw + x1];
        let s_x2_y1 = self.data[y1 * iw + x2];
        let s_x1_y1 = self.data[y1 * iw + x1];

        s_x2_y2 + s_x1_y1 - s_x1_y2 - s_x2_y1
    }

    /// Return the average pixel value in the box `[x, x+bw) × [y, y+bh)`.
    ///
    /// Returns `None` if the box is empty.
    #[must_use]
    pub fn box_mean(&self, x: u32, y: u32, bw: u32, bh: u32) -> Option<f64> {
        let area = bw as u64 * bh as u64;
        if area == 0 {
            return None;
        }
        Some(self.box_sum(x, y, bw, bh) as f64 / area as f64)
    }

    /// Total sum of all pixels in the image.
    #[must_use]
    pub fn total_sum(&self) -> u64 {
        self.box_sum(0, 0, self.width, self.height)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_and_full_sum() {
        // 3×3 image: values 1..9, sum = 45
        let pixels: Vec<u8> = (1..=9).collect();
        let ii = IntegralImage::build(&pixels, 3, 3);
        assert_eq!(ii.box_sum(0, 0, 3, 3), 45, "full sum must be 45");
        assert_eq!(ii.total_sum(), 45);
    }

    #[test]
    fn test_single_pixel_sum() {
        let pixels = vec![100u8, 200, 50, 75];
        let ii = IntegralImage::build(&pixels, 2, 2);
        assert_eq!(ii.box_sum(0, 0, 1, 1), 100);
        assert_eq!(ii.box_sum(1, 0, 1, 1), 200);
        assert_eq!(ii.box_sum(0, 1, 1, 1), 50);
        assert_eq!(ii.box_sum(1, 1, 1, 1), 75);
    }

    #[test]
    fn test_full_2x2_sum() {
        let pixels = vec![100u8, 200, 50, 75];
        let ii = IntegralImage::build(&pixels, 2, 2);
        assert_eq!(ii.box_sum(0, 0, 2, 2), 425);
    }

    #[test]
    fn test_empty_box_returns_zero() {
        let pixels = vec![255u8; 9];
        let ii = IntegralImage::build(&pixels, 3, 3);
        assert_eq!(ii.box_sum(1, 1, 0, 2), 0, "zero width must give 0");
        assert_eq!(ii.box_sum(1, 1, 2, 0), 0, "zero height must give 0");
    }

    #[test]
    fn test_out_of_bounds_clamped() {
        let pixels = vec![10u8; 4];
        let ii = IntegralImage::build(&pixels, 2, 2);
        // Ask for a box that extends beyond the image
        let clamped = ii.box_sum(0, 0, 100, 100);
        let expected = ii.box_sum(0, 0, 2, 2);
        assert_eq!(clamped, expected);
    }

    #[test]
    fn test_uniform_image_box_mean() {
        let pixels = vec![10u8; 25];
        let ii = IntegralImage::build(&pixels, 5, 5);
        let mean = ii.box_mean(1, 1, 3, 3).expect("mean must be Some");
        assert!((mean - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_integral_image_dimensions() {
        let ii = IntegralImage::build(&vec![0u8; 20], 5, 4);
        assert_eq!(ii.width, 5);
        assert_eq!(ii.height, 4);
    }

    /// Build integral image for a checkerboard and verify that alternating
    /// 2×2 box sums have expected values.
    #[test]
    fn test_checkerboard_2x4() {
        // 4×2 checkerboard: [255,0,255,0, 0,255,0,255]
        let pixels = vec![255u8, 0, 255, 0, 0, 255, 0, 255];
        let ii = IntegralImage::build(&pixels, 4, 2);
        // Full sum = 4 × 255 = 1020
        assert_eq!(ii.total_sum(), 1020);
    }
}
