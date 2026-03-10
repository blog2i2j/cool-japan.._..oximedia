//! Tensor operations and conversions.
//!
//! This module provides tensor abstractions for ML operations,
//! including conversions to/from video frames and various data layouts.

use crate::error::{CvError, CvResult};
use ndarray::{Array, ArrayD, IxDyn};
use ort::value::Value;
use oximedia_codec::VideoFrame;
use oximedia_core::PixelFormat;

/// Data layout for tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DataLayout {
    /// Batch, Channels, Height, Width (NCHW) - PyTorch default.
    #[default]
    Nchw,
    /// Batch, Height, Width, Channels (NHWC) - TensorFlow default.
    Nhwc,
}

/// Tensor data type.
#[derive(Debug, Clone)]
pub enum TensorData {
    /// 32-bit floating point tensor.
    F32(ArrayD<f32>),
    /// 8-bit unsigned integer tensor.
    U8(ArrayD<u8>),
    /// 64-bit floating point tensor.
    F64(ArrayD<f64>),
    /// 32-bit signed integer tensor.
    I32(ArrayD<i32>),
    /// 64-bit signed integer tensor.
    I64(ArrayD<i64>),
}

impl TensorData {
    /// Get the shape of the tensor.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::F32(arr) => arr.shape(),
            Self::U8(arr) => arr.shape(),
            Self::F64(arr) => arr.shape(),
            Self::I32(arr) => arr.shape(),
            Self::I64(arr) => arr.shape(),
        }
    }

    /// Get the number of dimensions.
    #[must_use]
    pub fn ndim(&self) -> usize {
        self.shape().len()
    }

    /// Get the total number of elements.
    #[must_use]
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Check if the tensor is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Convert to f32 tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails.
    pub fn to_f32(&self) -> CvResult<ArrayD<f32>> {
        match self {
            Self::F32(arr) => Ok(arr.clone()),
            Self::U8(arr) => Ok(arr.mapv(f32::from)),
            Self::F64(arr) => Ok(arr.mapv(|x| x as f32)),
            Self::I32(arr) => Ok(arr.mapv(|x| x as f32)),
            Self::I64(arr) => Ok(arr.mapv(|x| x as f32)),
        }
    }

    /// Convert to u8 tensor.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails.
    pub fn to_u8(&self) -> CvResult<ArrayD<u8>> {
        match self {
            Self::U8(arr) => Ok(arr.clone()),
            Self::F32(arr) => Ok(arr.mapv(|x| x.clamp(0.0, 255.0) as u8)),
            Self::F64(arr) => Ok(arr.mapv(|x| x.clamp(0.0, 255.0) as u8)),
            Self::I32(arr) => Ok(arr.mapv(|x| x.clamp(0, 255) as u8)),
            Self::I64(arr) => Ok(arr.mapv(|x| x.clamp(0, 255) as u8)),
        }
    }
}

/// Multi-dimensional tensor for ML operations.
///
/// Provides a high-level interface for tensor operations,
/// conversions, and data layout transformations.
#[derive(Debug, Clone)]
pub struct Tensor {
    data: TensorData,
    layout: DataLayout,
}

impl Tensor {
    /// Create a new tensor from f32 data.
    ///
    /// # Arguments
    ///
    /// * `data` - N-dimensional array of f32 values
    /// * `layout` - Data layout (NCHW or NHWC)
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::ml::{Tensor, TensorData, DataLayout};
    /// use ndarray::ArrayD;
    ///
    /// let data = ArrayD::zeros(vec![1, 3, 224, 224]);
    /// let tensor = Tensor::new_f32(data, DataLayout::Nchw);
    /// ```
    #[must_use]
    pub fn new_f32(data: ArrayD<f32>, layout: DataLayout) -> Self {
        Self {
            data: TensorData::F32(data),
            layout,
        }
    }

    /// Create a new tensor from u8 data.
    ///
    /// # Arguments
    ///
    /// * `data` - N-dimensional array of u8 values
    /// * `layout` - Data layout (NCHW or NHWC)
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::ml::{Tensor, DataLayout};
    /// use ndarray::ArrayD;
    ///
    /// let data = ArrayD::zeros(vec![1, 224, 224, 3]);
    /// let tensor = Tensor::new_u8(data, DataLayout::Nhwc);
    /// ```
    #[must_use]
    pub fn new_u8(data: ArrayD<u8>, layout: DataLayout) -> Self {
        Self {
            data: TensorData::U8(data),
            layout,
        }
    }

    /// Create a zero-filled f32 tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - Tensor shape
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::ml::Tensor;
    ///
    /// let tensor = Tensor::zeros(&[1, 3, 224, 224]);
    /// ```
    #[must_use]
    pub fn zeros(shape: &[usize]) -> Self {
        Self {
            data: TensorData::F32(ArrayD::zeros(IxDyn(shape))),
            layout: DataLayout::Nchw,
        }
    }

    /// Create a one-filled f32 tensor.
    ///
    /// # Arguments
    ///
    /// * `shape` - Tensor shape
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::ml::Tensor;
    ///
    /// let tensor = Tensor::ones(&[1, 3, 224, 224]);
    /// ```
    #[must_use]
    pub fn ones(shape: &[usize]) -> Self {
        Self {
            data: TensorData::F32(ArrayD::ones(IxDyn(shape))),
            layout: DataLayout::Nchw,
        }
    }

    /// Create a tensor from a video frame.
    ///
    /// Converts video frame to RGB tensor in NCHW format with values in [0, 255].
    ///
    /// # Arguments
    ///
    /// * `frame` - Video frame to convert
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Frame format is not supported
    /// - Frame data is invalid
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oximedia_cv::ml::Tensor;
    /// use oximedia_codec::VideoFrame;
    /// use oximedia_core::PixelFormat;
    ///
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let frame = VideoFrame::new(PixelFormat::Yuv420p, 640, 480);
    /// let tensor = Tensor::from_frame(&frame)?;
    /// # Ok(())
    /// # }
    /// ```
    pub fn from_frame(frame: &VideoFrame) -> CvResult<Self> {
        Self::from_frame_with_layout(frame, DataLayout::Nchw)
    }

    /// Create a tensor from a video frame with specified layout.
    ///
    /// # Arguments
    ///
    /// * `frame` - Video frame to convert
    /// * `layout` - Desired data layout
    ///
    /// # Errors
    ///
    /// Returns an error if frame format is not supported.
    pub fn from_frame_with_layout(frame: &VideoFrame, layout: DataLayout) -> CvResult<Self> {
        let width = frame.width as usize;
        let height = frame.height as usize;

        // For now, we'll create a simple RGB conversion
        // In a real implementation, you'd want proper YUV to RGB conversion
        let rgb_data = Self::frame_to_rgb(frame)?;

        // Create tensor based on layout
        let tensor_data = match layout {
            DataLayout::Nchw => {
                // Shape: [1, 3, H, W]
                let mut data = Array::zeros(IxDyn(&[1, 3, height, width]));
                for c in 0..3 {
                    for y in 0..height {
                        for x in 0..width {
                            let idx = (y * width + x) * 3 + c;
                            data[[0, c, y, x]] = f32::from(rgb_data[idx]);
                        }
                    }
                }
                data
            }
            DataLayout::Nhwc => {
                // Shape: [1, H, W, 3]
                let mut data = Array::zeros(IxDyn(&[1, height, width, 3]));
                for y in 0..height {
                    for x in 0..width {
                        for c in 0..3 {
                            let idx = (y * width + x) * 3 + c;
                            data[[0, y, x, c]] = f32::from(rgb_data[idx]);
                        }
                    }
                }
                data
            }
        };

        Ok(Self {
            data: TensorData::F32(tensor_data),
            layout,
        })
    }

    /// Convert frame to RGB data (simplified).
    fn frame_to_rgb(frame: &VideoFrame) -> CvResult<Vec<u8>> {
        match frame.format {
            PixelFormat::Yuv420p | PixelFormat::Yuv422p | PixelFormat::Yuv444p => {
                Self::yuv_to_rgb(frame)
            }
            PixelFormat::Rgb24 => {
                // Already RGB
                if frame.planes.is_empty() {
                    return Err(CvError::tensor_error("Frame has no plane data"));
                }
                Ok(frame.planes[0].data.clone())
            }
            _ => Err(CvError::unsupported_format(format!("{:?}", frame.format))),
        }
    }

    /// Convert YUV to RGB (simplified BT.601 conversion).
    fn yuv_to_rgb(frame: &VideoFrame) -> CvResult<Vec<u8>> {
        if frame.planes.len() < 3 {
            return Err(CvError::tensor_error("YUV frame requires 3 planes"));
        }

        let width = frame.width as usize;
        let height = frame.height as usize;
        let mut rgb = vec![0u8; width * height * 3];

        let y_plane = &frame.planes[0].data;
        let u_plane = &frame.planes[1].data;
        let v_plane = &frame.planes[2].data;

        let (u_width, _u_height) = frame.plane_dimensions(1);
        let (h_ratio, v_ratio) = frame.format.chroma_subsampling();

        for y in 0..height {
            for x in 0..width {
                let y_val = y_plane[y * width + x] as i32;
                let u_idx = (y / v_ratio as usize) * u_width as usize + (x / h_ratio as usize);
                let v_idx = (y / v_ratio as usize) * u_width as usize + (x / h_ratio as usize);
                let u_val = u_plane[u_idx] as i32 - 128;
                let v_val = v_plane[v_idx] as i32 - 128;

                // BT.601 conversion
                let r = (y_val + (1.402 * v_val as f32) as i32).clamp(0, 255);
                let g = (y_val - (0.344 * u_val as f32) as i32 - (0.714 * v_val as f32) as i32)
                    .clamp(0, 255);
                let b = (y_val + (1.772 * u_val as f32) as i32).clamp(0, 255);

                let rgb_idx = (y * width + x) * 3;
                rgb[rgb_idx] = r as u8;
                rgb[rgb_idx + 1] = g as u8;
                rgb[rgb_idx + 2] = b as u8;
            }
        }

        Ok(rgb)
    }

    /// Get the tensor shape.
    #[must_use]
    pub fn shape(&self) -> &[usize] {
        self.data.shape()
    }

    /// Get the data layout.
    #[must_use]
    pub const fn layout(&self) -> DataLayout {
        self.layout
    }

    /// Get the tensor data.
    #[must_use]
    pub const fn data(&self) -> &TensorData {
        &self.data
    }

    /// Convert to NCHW layout.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails or tensor shape is invalid.
    pub fn to_nchw(&self) -> CvResult<Self> {
        if self.layout == DataLayout::Nchw {
            return Ok(self.clone());
        }

        // Convert NHWC to NCHW
        if self.shape().len() != 4 {
            return Err(CvError::tensor_error(
                "Layout conversion requires 4D tensor",
            ));
        }

        let data_f32 = self.data.to_f32()?;
        let shape = data_f32.shape();
        let (n, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);

        let mut nchw_data = Array::zeros(IxDyn(&[n, c, h, w]));

        for ni in 0..n {
            for ci in 0..c {
                for hi in 0..h {
                    for wi in 0..w {
                        nchw_data[[ni, ci, hi, wi]] = data_f32[[ni, hi, wi, ci]];
                    }
                }
            }
        }

        Ok(Self {
            data: TensorData::F32(nchw_data),
            layout: DataLayout::Nchw,
        })
    }

    /// Convert to NHWC layout.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails or tensor shape is invalid.
    pub fn to_nhwc(&self) -> CvResult<Self> {
        if self.layout == DataLayout::Nhwc {
            return Ok(self.clone());
        }

        // Convert NCHW to NHWC
        if self.shape().len() != 4 {
            return Err(CvError::tensor_error(
                "Layout conversion requires 4D tensor",
            ));
        }

        let data_f32 = self.data.to_f32()?;
        let shape = data_f32.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);

        let mut nhwc_data = Array::zeros(IxDyn(&[n, h, w, c]));

        for ni in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    for ci in 0..c {
                        nhwc_data[[ni, hi, wi, ci]] = data_f32[[ni, ci, hi, wi]];
                    }
                }
            }
        }

        Ok(Self {
            data: TensorData::F32(nhwc_data),
            layout: DataLayout::Nhwc,
        })
    }

    /// Normalize tensor values.
    ///
    /// # Arguments
    ///
    /// * `mean` - Mean values for each channel
    /// * `std` - Standard deviation for each channel
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions don't match.
    ///
    /// # Example
    ///
    /// ```
    /// use oximedia_cv::ml::Tensor;
    ///
    /// let mut tensor = Tensor::zeros(&[1, 3, 224, 224]);
    /// // ImageNet normalization
    /// tensor.normalize(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225])?;
    /// ```
    pub fn normalize(&mut self, mean: &[f32], std: &[f32]) -> CvResult<()> {
        let mut data_f32 = self.data.to_f32()?;

        match self.layout {
            DataLayout::Nchw => {
                let shape = data_f32.shape().to_vec();
                if shape.len() != 4 {
                    return Err(CvError::tensor_error("Normalization requires 4D tensor"));
                }

                let channels = shape[1];
                if mean.len() != channels || std.len() != channels {
                    return Err(CvError::tensor_error("Mean/std length must match channels"));
                }

                let (batch, _, height, width) = (shape[0], shape[1], shape[2], shape[3]);

                for c in 0..channels {
                    for n in 0..batch {
                        for h in 0..height {
                            for w in 0..width {
                                let val = data_f32[[n, c, h, w]];
                                data_f32[[n, c, h, w]] = (val - mean[c]) / std[c];
                            }
                        }
                    }
                }
            }
            DataLayout::Nhwc => {
                let shape = data_f32.shape().to_vec();
                if shape.len() != 4 {
                    return Err(CvError::tensor_error("Normalization requires 4D tensor"));
                }

                let channels = shape[3];
                if mean.len() != channels || std.len() != channels {
                    return Err(CvError::tensor_error("Mean/std length must match channels"));
                }

                let (batch, height, width, _) = (shape[0], shape[1], shape[2], shape[3]);

                for c in 0..channels {
                    for n in 0..batch {
                        for h in 0..height {
                            for w in 0..width {
                                let val = data_f32[[n, h, w, c]];
                                data_f32[[n, h, w, c]] = (val - mean[c]) / std[c];
                            }
                        }
                    }
                }
            }
        }

        self.data = TensorData::F32(data_f32);
        Ok(())
    }

    /// Convert to ONNX Runtime value.
    ///
    /// # Errors
    ///
    /// Returns an error if conversion fails.
    #[cfg(feature = "onnx")]
    pub fn to_ort_value(&self) -> CvResult<ort::value::DynValue> {
        match &self.data {
            TensorData::F32(arr) => {
                let val = Value::from_array(arr.clone()).map_err(|e| {
                    CvError::tensor_error(format!("Failed to create ORT value: {e}"))
                })?;
                Ok(val.into_dyn())
            }
            TensorData::U8(arr) => {
                // Convert u8 to f32 since ORT may not support u8 directly
                let f32_arr: ArrayD<f32> = arr.mapv(f32::from);
                let val = Value::from_array(f32_arr).map_err(|e| {
                    CvError::tensor_error(format!("Failed to create ORT value: {e}"))
                })?;
                Ok(val.into_dyn())
            }
            TensorData::F64(arr) => {
                let val = Value::from_array(arr.clone()).map_err(|e| {
                    CvError::tensor_error(format!("Failed to create ORT value: {e}"))
                })?;
                Ok(val.into_dyn())
            }
            TensorData::I32(arr) => {
                let val = Value::from_array(arr.clone()).map_err(|e| {
                    CvError::tensor_error(format!("Failed to create ORT value: {e}"))
                })?;
                Ok(val.into_dyn())
            }
            TensorData::I64(arr) => {
                let val = Value::from_array(arr.clone()).map_err(|e| {
                    CvError::tensor_error(format!("Failed to create ORT value: {e}"))
                })?;
                Ok(val.into_dyn())
            }
        }
    }

    /// Create tensor from ONNX Runtime value.
    ///
    /// # Errors
    ///
    /// Returns an error if the value type is not supported.
    #[cfg(feature = "onnx")]
    pub fn from_ort_value(value: &ort::value::DynValue) -> CvResult<Self> {
        // Extract the tensor data
        let (shape, data) = value
            .try_extract_tensor::<f32>()
            .map_err(|e| CvError::tensor_error(format!("Failed to extract f32 tensor: {e}")))?;

        let shape: Vec<usize> = shape.iter().map(|&x| x as usize).collect();
        let data_vec: Vec<f32> = data.to_vec();

        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data_vec)
            .map_err(|e| CvError::tensor_error(format!("Failed to create array: {e}")))?;

        Ok(Self {
            data: TensorData::F32(arr),
            layout: DataLayout::Nchw,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_layout_default() {
        assert_eq!(DataLayout::default(), DataLayout::Nchw);
    }

    #[test]
    fn test_tensor_zeros() {
        let tensor = Tensor::zeros(&[1, 3, 224, 224]);
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_tensor_ones() {
        let tensor = Tensor::ones(&[1, 3, 224, 224]);
        assert_eq!(tensor.shape(), &[1, 3, 224, 224]);
    }

    #[test]
    fn test_tensor_layout() {
        let tensor = Tensor::zeros(&[1, 3, 224, 224]);
        assert_eq!(tensor.layout(), DataLayout::Nchw);
    }

    #[test]
    fn test_nchw_to_nhwc_conversion() {
        let nchw = Tensor::zeros(&[1, 3, 4, 4]);
        let nhwc = nchw.to_nhwc().expect("to_nhwc should succeed");
        assert_eq!(nhwc.shape(), &[1, 4, 4, 3]);
        assert_eq!(nhwc.layout(), DataLayout::Nhwc);
    }

    #[test]
    fn test_nhwc_to_nchw_conversion() {
        let data = ArrayD::zeros(IxDyn(&[1, 4, 4, 3]));
        let nhwc = Tensor::new_f32(data, DataLayout::Nhwc);
        let nchw = nhwc.to_nchw().expect("to_nchw should succeed");
        assert_eq!(nchw.shape(), &[1, 3, 4, 4]);
        assert_eq!(nchw.layout(), DataLayout::Nchw);
    }

    #[test]
    fn test_tensor_normalize() {
        let mut tensor = Tensor::zeros(&[1, 3, 4, 4]);
        let result = tensor.normalize(&[0.5, 0.5, 0.5], &[0.5, 0.5, 0.5]);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tensor_data_shape() {
        let data = ArrayD::zeros(IxDyn(&[1, 3, 224, 224]));
        let tensor_data = TensorData::F32(data);
        assert_eq!(tensor_data.shape(), &[1, 3, 224, 224]);
        assert_eq!(tensor_data.ndim(), 4);
    }

    #[test]
    fn test_tensor_data_len() {
        let data = ArrayD::zeros(IxDyn(&[1, 3, 4, 4]));
        let tensor_data = TensorData::F32(data);
        assert_eq!(tensor_data.len(), 48);
        assert!(!tensor_data.is_empty());
    }
}
