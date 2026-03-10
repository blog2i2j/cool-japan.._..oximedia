//! Morphological operations.
//!
//! This module provides morphological image processing operations including:
//! - Erosion and dilation
//! - Opening and closing
//! - Structuring elements (rectangle, ellipse, cross)
//!
//! # Example
//!
//! ```
//! use oximedia_cv::image::{MorphOperation, StructuringElement};
//!
//! let element = StructuringElement::rectangle(3, 3);
//! assert_eq!(element.width(), 3);
//! ```

use crate::error::{CvError, CvResult};

/// Morphological operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MorphOperation {
    /// Erosion - shrinks bright regions.
    Erode,
    /// Dilation - expands bright regions.
    Dilate,
    /// Opening - erosion followed by dilation (removes small bright spots).
    Open,
    /// Closing - dilation followed by erosion (fills small dark holes).
    Close,
    /// Morphological gradient - difference between dilation and erosion.
    Gradient,
    /// Top-hat - difference between image and opening (extracts bright spots).
    TopHat,
    /// Black-hat - difference between closing and image (extracts dark spots).
    BlackHat,
}

/// Structuring element for morphological operations.
#[derive(Debug, Clone)]
pub struct StructuringElement {
    /// Element data (true = part of element).
    data: Vec<bool>,
    /// Element width.
    width: usize,
    /// Element height.
    height: usize,
    /// Anchor point x (center of element).
    anchor_x: usize,
    /// Anchor point y (center of element).
    anchor_y: usize,
}

impl StructuringElement {
    /// Create a rectangular structuring element.
    ///
    /// # Arguments
    ///
    /// * `width` - Element width
    /// * `height` - Element height
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::StructuringElement;
    ///
    /// let element = StructuringElement::rectangle(5, 5);
    /// assert_eq!(element.width(), 5);
    /// assert_eq!(element.height(), 5);
    /// ```
    #[must_use]
    pub fn rectangle(width: usize, height: usize) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        Self {
            data: vec![true; width * height],
            width,
            height,
            anchor_x: width / 2,
            anchor_y: height / 2,
        }
    }

    /// Create an elliptical structuring element.
    ///
    /// # Arguments
    ///
    /// * `width` - Element width
    /// * `height` - Element height
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::StructuringElement;
    ///
    /// let element = StructuringElement::ellipse(5, 5);
    /// assert_eq!(element.width(), 5);
    /// ```
    #[must_use]
    pub fn ellipse(width: usize, height: usize) -> Self {
        let width = width.max(1);
        let height = height.max(1);
        let mut data = vec![false; width * height];

        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let rx = cx;
        let ry = cy;

        for y in 0..height {
            for x in 0..width {
                let dx = (x as f64 + 0.5 - cx) / rx;
                let dy = (y as f64 + 0.5 - cy) / ry;
                if dx * dx + dy * dy <= 1.0 {
                    data[y * width + x] = true;
                }
            }
        }

        Self {
            data,
            width,
            height,
            anchor_x: width / 2,
            anchor_y: height / 2,
        }
    }

    /// Create a cross-shaped structuring element.
    ///
    /// # Arguments
    ///
    /// * `size` - Size of the cross (must be odd)
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::StructuringElement;
    ///
    /// let element = StructuringElement::cross(5);
    /// assert_eq!(element.width(), 5);
    /// ```
    #[must_use]
    pub fn cross(size: usize) -> Self {
        let size = if size % 2 == 0 { size + 1 } else { size };
        let size = size.max(3);
        let mut data = vec![false; size * size];

        let center = size / 2;

        // Horizontal line
        for x in 0..size {
            data[center * size + x] = true;
        }

        // Vertical line
        for y in 0..size {
            data[y * size + center] = true;
        }

        Self {
            data,
            width: size,
            height: size,
            anchor_x: center,
            anchor_y: center,
        }
    }

    /// Create a custom structuring element.
    ///
    /// # Arguments
    ///
    /// * `data` - Element data (row-major, true = part of element)
    /// * `width` - Element width
    /// * `height` - Element height
    ///
    /// # Examples
    ///
    /// ```
    /// use oximedia_cv::image::StructuringElement;
    ///
    /// let element = StructuringElement::custom(
    ///     vec![true, true, true, true, true, true, true, true, true],
    ///     3, 3
    /// );
    /// ```
    #[must_use]
    pub fn custom(data: Vec<bool>, width: usize, height: usize) -> Self {
        assert_eq!(data.len(), width * height);
        Self {
            data,
            width,
            height,
            anchor_x: width / 2,
            anchor_y: height / 2,
        }
    }

    /// Get element width.
    #[must_use]
    pub const fn width(&self) -> usize {
        self.width
    }

    /// Get element height.
    #[must_use]
    pub const fn height(&self) -> usize {
        self.height
    }

    /// Get anchor point.
    #[must_use]
    pub const fn anchor(&self) -> (usize, usize) {
        (self.anchor_x, self.anchor_y)
    }

    /// Check if a position is part of the element.
    #[must_use]
    pub fn contains(&self, x: usize, y: usize) -> bool {
        if x < self.width && y < self.height {
            self.data[y * self.width + x]
        } else {
            false
        }
    }

    /// Get iterator over element positions that are true.
    pub fn iter_active(&self) -> impl Iterator<Item = (i32, i32)> + '_ {
        self.data
            .iter()
            .enumerate()
            .filter(|(_, &v)| v)
            .map(|(i, _)| {
                let x = (i % self.width) as i32 - self.anchor_x as i32;
                let y = (i / self.width) as i32 - self.anchor_y as i32;
                (x, y)
            })
    }
}

impl Default for StructuringElement {
    fn default() -> Self {
        Self::rectangle(3, 3)
    }
}

/// Perform erosion on a grayscale image.
///
/// Erosion takes the minimum value in the neighborhood defined by the
/// structuring element.
///
/// # Arguments
///
/// * `src` - Source grayscale image data
/// * `width` - Image width
/// * `height` - Image height
/// * `element` - Structuring element
///
/// # Returns
///
/// Eroded image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::{StructuringElement, morph::erode};
///
/// let src = vec![255u8; 25];
/// let element = StructuringElement::rectangle(3, 3);
/// let result = erode(&src, 5, 5, &element)?;
/// ```
pub fn erode(
    src: &[u8],
    width: u32,
    height: u32,
    element: &StructuringElement,
) -> CvResult<Vec<u8>> {
    validate_input(src, width, height)?;

    let w = width as usize;
    let h = height as usize;
    let mut dst = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut min_val = 255u8;

            for (dx, dy) in element.iter_active() {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                    let idx = ny as usize * w + nx as usize;
                    min_val = min_val.min(src[idx]);
                }
            }

            dst[y * w + x] = min_val;
        }
    }

    Ok(dst)
}

/// Perform dilation on a grayscale image.
///
/// Dilation takes the maximum value in the neighborhood defined by the
/// structuring element.
///
/// # Arguments
///
/// * `src` - Source grayscale image data
/// * `width` - Image width
/// * `height` - Image height
/// * `element` - Structuring element
///
/// # Returns
///
/// Dilated image data.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::{StructuringElement, morph::dilate};
///
/// let src = vec![0u8; 25];
/// let element = StructuringElement::rectangle(3, 3);
/// let result = dilate(&src, 5, 5, &element)?;
/// ```
pub fn dilate(
    src: &[u8],
    width: u32,
    height: u32,
    element: &StructuringElement,
) -> CvResult<Vec<u8>> {
    validate_input(src, width, height)?;

    let w = width as usize;
    let h = height as usize;
    let mut dst = vec![0u8; w * h];

    for y in 0..h {
        for x in 0..w {
            let mut max_val = 0u8;

            for (dx, dy) in element.iter_active() {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;

                if nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32 {
                    let idx = ny as usize * w + nx as usize;
                    max_val = max_val.max(src[idx]);
                }
            }

            dst[y * w + x] = max_val;
        }
    }

    Ok(dst)
}

/// Perform morphological opening (erosion followed by dilation).
///
/// Opening removes small bright spots and smooths object boundaries.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::{StructuringElement, morph::morphological_open};
///
/// let src = vec![128u8; 25];
/// let element = StructuringElement::rectangle(3, 3);
/// let result = morphological_open(&src, 5, 5, &element)?;
/// ```
pub fn morphological_open(
    src: &[u8],
    width: u32,
    height: u32,
    element: &StructuringElement,
) -> CvResult<Vec<u8>> {
    let eroded = erode(src, width, height, element)?;
    dilate(&eroded, width, height, element)
}

/// Perform morphological closing (dilation followed by erosion).
///
/// Closing fills small dark holes and smooths object boundaries.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
///
/// # Examples
///
/// ```
/// use oximedia_cv::image::{StructuringElement, morph::morphological_close};
///
/// let src = vec![128u8; 25];
/// let element = StructuringElement::rectangle(3, 3);
/// let result = morphological_close(&src, 5, 5, &element)?;
/// ```
pub fn morphological_close(
    src: &[u8],
    width: u32,
    height: u32,
    element: &StructuringElement,
) -> CvResult<Vec<u8>> {
    let dilated = dilate(src, width, height, element)?;
    erode(&dilated, width, height, element)
}

/// Perform morphological gradient (dilation minus erosion).
///
/// Highlights edges in the image.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn morphological_gradient(
    src: &[u8],
    width: u32,
    height: u32,
    element: &StructuringElement,
) -> CvResult<Vec<u8>> {
    let dilated = dilate(src, width, height, element)?;
    let eroded = erode(src, width, height, element)?;

    let size = width as usize * height as usize;
    let mut result = vec![0u8; size];

    for i in 0..size {
        result[i] = dilated[i].saturating_sub(eroded[i]);
    }

    Ok(result)
}

/// Perform top-hat transform (image minus opening).
///
/// Extracts bright spots smaller than the structuring element.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn top_hat(
    src: &[u8],
    width: u32,
    height: u32,
    element: &StructuringElement,
) -> CvResult<Vec<u8>> {
    let opened = morphological_open(src, width, height, element)?;

    let size = width as usize * height as usize;
    let mut result = vec![0u8; size];

    for i in 0..size {
        result[i] = src[i].saturating_sub(opened[i]);
    }

    Ok(result)
}

/// Perform black-hat transform (closing minus image).
///
/// Extracts dark spots smaller than the structuring element.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn black_hat(
    src: &[u8],
    width: u32,
    height: u32,
    element: &StructuringElement,
) -> CvResult<Vec<u8>> {
    let closed = morphological_close(src, width, height, element)?;

    let size = width as usize * height as usize;
    let mut result = vec![0u8; size];

    for i in 0..size {
        result[i] = closed[i].saturating_sub(src[i]);
    }

    Ok(result)
}

/// Apply a morphological operation.
///
/// # Errors
///
/// Returns an error if dimensions are invalid.
pub fn apply_morph(
    src: &[u8],
    width: u32,
    height: u32,
    operation: MorphOperation,
    element: &StructuringElement,
) -> CvResult<Vec<u8>> {
    match operation {
        MorphOperation::Erode => erode(src, width, height, element),
        MorphOperation::Dilate => dilate(src, width, height, element),
        MorphOperation::Open => morphological_open(src, width, height, element),
        MorphOperation::Close => morphological_close(src, width, height, element),
        MorphOperation::Gradient => morphological_gradient(src, width, height, element),
        MorphOperation::TopHat => top_hat(src, width, height, element),
        MorphOperation::BlackHat => black_hat(src, width, height, element),
    }
}

/// Validate input dimensions.
fn validate_input(data: &[u8], width: u32, height: u32) -> CvResult<()> {
    if width == 0 || height == 0 {
        return Err(CvError::invalid_dimensions(width, height));
    }

    let expected_size = width as usize * height as usize;
    if data.len() < expected_size {
        return Err(CvError::insufficient_data(expected_size, data.len()));
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rectangle_element() {
        let element = StructuringElement::rectangle(3, 3);
        assert_eq!(element.width(), 3);
        assert_eq!(element.height(), 3);
        assert!(element.contains(0, 0));
        assert!(element.contains(2, 2));
    }

    #[test]
    fn test_ellipse_element() {
        let element = StructuringElement::ellipse(5, 5);
        assert_eq!(element.width(), 5);
        assert_eq!(element.height(), 5);

        // Center should be included
        assert!(element.contains(2, 2));
    }

    #[test]
    fn test_cross_element() {
        let element = StructuringElement::cross(5);
        assert_eq!(element.width(), 5);

        // Center should be included
        assert!(element.contains(2, 2));

        // Corners should not be included
        assert!(!element.contains(0, 0));
        assert!(!element.contains(4, 4));

        // Arms should be included
        assert!(element.contains(0, 2));
        assert!(element.contains(2, 0));
    }

    #[test]
    fn test_erode() {
        // Create image with a bright spot
        let mut src = vec![0u8; 25];
        src[12] = 255; // Center pixel

        let element = StructuringElement::rectangle(3, 3);
        let result = erode(&src, 5, 5, &element).expect("erode should succeed");

        // Bright spot should be removed by erosion
        assert_eq!(result[12], 0);
    }

    #[test]
    fn test_dilate() {
        // Create image with a bright spot
        let mut src = vec![0u8; 25];
        src[12] = 255; // Center pixel

        let element = StructuringElement::rectangle(3, 3);
        let result = dilate(&src, 5, 5, &element).expect("dilate should succeed");

        // Bright spot should expand
        assert_eq!(result[12], 255);
        assert_eq!(result[11], 255); // Left neighbor
        assert_eq!(result[7], 255); // Top neighbor
    }

    #[test]
    fn test_morphological_open() {
        let src = vec![128u8; 25];
        let element = StructuringElement::rectangle(3, 3);
        let result =
            morphological_open(&src, 5, 5, &element).expect("morphological_open should succeed");
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_morphological_close() {
        let src = vec![128u8; 25];
        let element = StructuringElement::rectangle(3, 3);
        let result =
            morphological_close(&src, 5, 5, &element).expect("morphological_close should succeed");
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_morphological_gradient() {
        let src = vec![128u8; 25];
        let element = StructuringElement::rectangle(3, 3);
        let result = morphological_gradient(&src, 5, 5, &element)
            .expect("morphological_gradient should succeed");
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_top_hat() {
        let src = vec![128u8; 25];
        let element = StructuringElement::rectangle(3, 3);
        let result = top_hat(&src, 5, 5, &element).expect("top_hat should succeed");
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_black_hat() {
        let src = vec![128u8; 25];
        let element = StructuringElement::rectangle(3, 3);
        let result = black_hat(&src, 5, 5, &element).expect("black_hat should succeed");
        assert_eq!(result.len(), 25);
    }

    #[test]
    fn test_apply_morph() {
        let src = vec![128u8; 25];
        let element = StructuringElement::rectangle(3, 3);

        for op in [
            MorphOperation::Erode,
            MorphOperation::Dilate,
            MorphOperation::Open,
            MorphOperation::Close,
            MorphOperation::Gradient,
            MorphOperation::TopHat,
            MorphOperation::BlackHat,
        ] {
            let result = apply_morph(&src, 5, 5, op, &element).expect("apply_morph should succeed");
            assert_eq!(result.len(), 25);
        }
    }

    #[test]
    fn test_invalid_dimensions() {
        let src = vec![0u8; 25];
        let element = StructuringElement::rectangle(3, 3);
        assert!(erode(&src, 0, 5, &element).is_err());
        assert!(dilate(&src, 5, 0, &element).is_err());
    }

    #[test]
    fn test_structuring_element_iter() {
        let element = StructuringElement::rectangle(3, 3);
        let positions: Vec<_> = element.iter_active().collect();
        assert_eq!(positions.len(), 9);
    }
}
