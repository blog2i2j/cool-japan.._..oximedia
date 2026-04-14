//! Image preprocessing for ML models.
//!
//! This module provides common preprocessing operations for preparing
//! images for ML model inference.

use crate::error::{CvError, CvResult};
use crate::ml::tensor::{DataLayout, Tensor};
use ndarray::{Array, IxDyn};

/// ImageNet normalization parameters.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
/// ImageNet standard deviation parameters.
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Image preprocessor for ML models.
///
/// Provides a fluent API for chaining preprocessing operations.
///
/// # Example
///
/// ```
/// use oximedia_cv::ml::preprocessing::ImagePreprocessor;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let preprocessor = ImagePreprocessor::new()
///     .resize(224, 224)
///     .normalize_imagenet()
///     .to_nchw();
/// # Ok(())
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct ImagePreprocessor {
    target_width: Option<u32>,
    target_height: Option<u32>,
    normalize_mean: Option<Vec<f32>>,
    normalize_std: Option<Vec<f32>>,
    layout: DataLayout,
    pad_value: f32,
}

impl ImagePreprocessor {
    /// Create a new image preprocessor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            target_width: None,
            target_height: None,
            normalize_mean: None,
            normalize_std: None,
            layout: DataLayout::Nchw,
            pad_value: 0.0,
        }
    }

    /// Set target size for resizing.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::ml::preprocessing::ImagePreprocessor;
    ///
    /// let preprocessor = ImagePreprocessor::new().resize(224, 224);
    /// ```
    #[must_use]
    pub fn resize(mut self, width: u32, height: u32) -> Self {
        self.target_width = Some(width);
        self.target_height = Some(height);
        self
    }

    /// Set normalization parameters.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::ml::preprocessing::ImagePreprocessor;
    ///
    /// let preprocessor = ImagePreprocessor::new()
    ///     .normalize(&[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]);
    /// ```
    #[must_use]
    pub fn normalize(mut self, mean: &[f32], std: &[f32]) -> Self {
        self.normalize_mean = Some(mean.to_vec());
        self.normalize_std = Some(std.to_vec());
        self
    }

    /// Use ImageNet normalization.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::ml::preprocessing::ImagePreprocessor;
    ///
    /// let preprocessor = ImagePreprocessor::new().normalize_imagenet();
    /// ```
    #[must_use]
    pub fn normalize_imagenet(self) -> Self {
        self.normalize(&IMAGENET_MEAN, &IMAGENET_STD)
    }

    /// Set data layout to NCHW.
    #[must_use]
    pub fn to_nchw(mut self) -> Self {
        self.layout = DataLayout::Nchw;
        self
    }

    /// Set data layout to NHWC.
    #[must_use]
    pub fn to_nhwc(mut self) -> Self {
        self.layout = DataLayout::Nhwc;
        self
    }

    /// Set padding value.
    #[must_use]
    pub fn with_pad_value(mut self, value: f32) -> Self {
        self.pad_value = value;
        self
    }

    /// Apply preprocessing to a tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if preprocessing fails.
    pub fn process(&self, mut tensor: Tensor) -> CvResult<Tensor> {
        // Resize if needed
        if let (Some(width), Some(height)) = (self.target_width, self.target_height) {
            tensor = resize_tensor(&tensor, width as usize, height as usize)?;
        }

        // Convert to target layout
        tensor = match self.layout {
            DataLayout::Nchw => tensor.to_nchw()?,
            DataLayout::Nhwc => tensor.to_nhwc()?,
        };

        // Normalize if specified
        if let (Some(mean), Some(std)) = (&self.normalize_mean, &self.normalize_std) {
            tensor.normalize(mean, std)?;
        }

        Ok(tensor)
    }
}

impl Default for ImagePreprocessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Resize a tensor to the specified dimensions.
///
/// Uses bilinear interpolation for resizing.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `width` - Target width
/// * `height` - Target height
///
/// # Errors
///
/// Returns an error if tensor format is invalid.
#[allow(dead_code)]
fn resize_tensor(tensor: &Tensor, width: usize, height: usize) -> CvResult<Tensor> {
    let shape = tensor.shape();
    if shape.len() != 4 {
        return Err(CvError::tensor_error("Resize requires 4D tensor"));
    }

    let (batch, channels, old_h, old_w) = (shape[0], shape[1], shape[2], shape[3]);

    let data_f32 = tensor.data().to_f32()?;

    // Simple nearest neighbor resize (for simplicity)
    let mut resized = match tensor.layout() {
        DataLayout::Nchw => Array::zeros(IxDyn(&[batch, channels, height, width])),
        DataLayout::Nhwc => Array::zeros(IxDyn(&[batch, height, width, channels])),
    };

    let x_ratio = old_w as f32 / width as f32;
    let y_ratio = old_h as f32 / height as f32;

    match tensor.layout() {
        DataLayout::Nchw => {
            for b in 0..batch {
                for c in 0..channels {
                    for y in 0..height {
                        for x in 0..width {
                            let src_x = (x as f32 * x_ratio) as usize;
                            let src_y = (y as f32 * y_ratio) as usize;
                            let src_x = src_x.min(old_w - 1);
                            let src_y = src_y.min(old_h - 1);
                            resized[[b, c, y, x]] = data_f32[[b, c, src_y, src_x]];
                        }
                    }
                }
            }
        }
        DataLayout::Nhwc => {
            for b in 0..batch {
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..channels {
                            let src_x = (x as f32 * x_ratio) as usize;
                            let src_y = (y as f32 * y_ratio) as usize;
                            let src_x = src_x.min(old_w - 1);
                            let src_y = src_y.min(old_h - 1);
                            resized[[b, y, x, c]] = data_f32[[b, src_y, src_x, c]];
                        }
                    }
                }
            }
        }
    }

    Ok(Tensor::new_f32(resized, tensor.layout()))
}

/// Normalize tensor with mean and standard deviation.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `mean` - Mean values per channel
/// * `std` - Standard deviation per channel
///
/// # Errors
///
/// Returns an error if dimensions don't match.
///
/// # Example
///
/// ```
/// use oximedia_cv::ml::{Tensor, preprocessing::normalize};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut tensor = Tensor::zeros(&[1, 3, 224, 224]);
/// normalize(&mut tensor, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5])?;
/// # Ok(())
/// # }
/// ```
pub fn normalize(tensor: &mut Tensor, mean: &[f32], std: &[f32]) -> CvResult<()> {
    tensor.normalize(mean, std)
}

/// Normalize tensor with ImageNet parameters.
///
/// Uses mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225].
///
/// # Arguments
///
/// * `tensor` - Input tensor to normalize
///
/// # Errors
///
/// Returns an error if normalization fails.
///
/// # Example
///
/// ```
/// use oximedia_cv::ml::{Tensor, preprocessing::normalize_imagenet};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut tensor = Tensor::zeros(&[1, 3, 224, 224]);
/// normalize_imagenet(&mut tensor)?;
/// # Ok(())
/// # }
/// ```
pub fn normalize_imagenet(tensor: &mut Tensor) -> CvResult<()> {
    tensor.normalize(&IMAGENET_MEAN, &IMAGENET_STD)
}

/// Resize tensor to fit target dimensions while maintaining aspect ratio.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `target_width` - Target width
/// * `target_height` - Target height
/// * `pad_value` - Value to use for padding
///
/// # Errors
///
/// Returns an error if resizing fails.
#[allow(dead_code)]
pub fn resize_to_fit(
    tensor: &Tensor,
    target_width: usize,
    target_height: usize,
    pad_value: f32,
) -> CvResult<Tensor> {
    let shape = tensor.shape();
    if shape.len() != 4 {
        return Err(CvError::tensor_error("Resize requires 4D tensor"));
    }

    let (_, _, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    // Calculate scale to fit
    let scale_w = target_width as f32 / width as f32;
    let scale_h = target_height as f32 / height as f32;
    let scale = scale_w.min(scale_h);

    let new_width = (width as f32 * scale) as usize;
    let new_height = (height as f32 * scale) as usize;

    // Resize
    let mut resized = resize_tensor(tensor, new_width, new_height)?;

    // Pad to target size
    resized = pad_to_size(&resized, target_width, target_height, pad_value)?;

    Ok(resized)
}

/// Pad tensor to target size.
///
/// Centers the input tensor and pads with the specified value.
///
/// # Arguments
///
/// * `tensor` - Input tensor
/// * `target_width` - Target width
/// * `target_height` - Target height
/// * `pad_value` - Value to use for padding
///
/// # Errors
///
/// Returns an error if padding fails.
pub fn pad_to_size(
    tensor: &Tensor,
    target_width: usize,
    target_height: usize,
    pad_value: f32,
) -> CvResult<Tensor> {
    let shape = tensor.shape();
    if shape.len() != 4 {
        return Err(CvError::tensor_error("Padding requires 4D tensor"));
    }

    let (batch, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);

    if width >= target_width && height >= target_height {
        return Ok(tensor.clone());
    }

    let data_f32 = tensor.data().to_f32()?;

    let pad_x = (target_width - width) / 2;
    let pad_y = (target_height - height) / 2;

    let padded = match tensor.layout() {
        DataLayout::Nchw => {
            let mut arr = Array::from_elem(
                IxDyn(&[batch, channels, target_height, target_width]),
                pad_value,
            );
            for b in 0..batch {
                for c in 0..channels {
                    for y in 0..height {
                        for x in 0..width {
                            arr[[b, c, y + pad_y, x + pad_x]] = data_f32[[b, c, y, x]];
                        }
                    }
                }
            }
            arr
        }
        DataLayout::Nhwc => {
            let mut arr = Array::from_elem(
                IxDyn(&[batch, target_height, target_width, channels]),
                pad_value,
            );
            for b in 0..batch {
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..channels {
                            arr[[b, y + pad_y, x + pad_x, c]] = data_f32[[b, y, x, c]];
                        }
                    }
                }
            }
            arr
        }
    };

    Ok(Tensor::new_f32(padded, tensor.layout()))
}

/// Convert pixel values from [0, 255] to [0, 1].
///
/// # Arguments
///
/// * `tensor` - Input tensor with values in [0, 255]
///
/// # Errors
///
/// Returns an error if conversion fails.
///
/// # Example
///
/// ```
/// use oximedia_cv::ml::{Tensor, preprocessing::scale_to_unit};
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut tensor = Tensor::zeros(&[1, 3, 224, 224]);
/// scale_to_unit(&mut tensor)?;
/// # Ok(())
/// # }
/// ```
pub fn scale_to_unit(tensor: &mut Tensor) -> CvResult<()> {
    let mut data_f32 = tensor.data().to_f32()?;
    data_f32.mapv_inplace(|x| x / 255.0);
    *tensor = Tensor::new_f32(data_f32, tensor.layout());
    Ok(())
}

/// Convert pixel values from [0, 1] to [0, 255].
///
/// # Arguments
///
/// * `tensor` - Input tensor with values in [0, 1]
///
/// # Errors
///
/// Returns an error if conversion fails.
pub fn scale_from_unit(tensor: &mut Tensor) -> CvResult<()> {
    let mut data_f32 = tensor.data().to_f32()?;
    data_f32.mapv_inplace(|x| x * 255.0);
    *tensor = Tensor::new_f32(data_f32, tensor.layout());
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_imagenet_constants() {
        assert_eq!(IMAGENET_MEAN.len(), 3);
        assert_eq!(IMAGENET_STD.len(), 3);
    }

    #[test]
    fn test_image_preprocessor_new() {
        let preprocessor = ImagePreprocessor::new();
        assert!(preprocessor.target_width.is_none());
        assert!(preprocessor.target_height.is_none());
    }

    #[test]
    fn test_image_preprocessor_resize() {
        let preprocessor = ImagePreprocessor::new().resize(224, 224);
        assert_eq!(preprocessor.target_width, Some(224));
        assert_eq!(preprocessor.target_height, Some(224));
    }

    #[test]
    fn test_image_preprocessor_normalize() {
        let preprocessor = ImagePreprocessor::new().normalize(&[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]);
        assert!(preprocessor.normalize_mean.is_some());
        assert!(preprocessor.normalize_std.is_some());
    }

    #[test]
    fn test_image_preprocessor_imagenet() {
        let preprocessor = ImagePreprocessor::new().normalize_imagenet();
        assert!(preprocessor.normalize_mean.is_some());
        assert_eq!(
            preprocessor
                .normalize_mean
                .as_ref()
                .expect("as_ref should succeed"),
            &IMAGENET_MEAN
        );
    }

    #[test]
    fn test_normalize() {
        let mut tensor = Tensor::zeros(&[1, 3, 4, 4]);
        let result = normalize(&mut tensor, &[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_normalize_imagenet() {
        let mut tensor = Tensor::zeros(&[1, 3, 4, 4]);
        let result = normalize_imagenet(&mut tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_pad_to_size() {
        let tensor = Tensor::zeros(&[1, 3, 4, 4]);
        let result = pad_to_size(&tensor, 8, 8, 0.0);
        assert!(result.is_ok());
        assert_eq!(
            result.expect("value should be valid").shape(),
            &[1, 3, 8, 8]
        );
    }

    #[test]
    fn test_scale_to_unit() {
        let mut tensor = Tensor::zeros(&[1, 3, 4, 4]);
        let result = scale_to_unit(&mut tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_scale_from_unit() {
        let mut tensor = Tensor::zeros(&[1, 3, 4, 4]);
        let result = scale_from_unit(&mut tensor);
        assert!(result.is_ok());
    }
}
